"""NixOrb VRAM manager — GPU memory lifecycle for GTX 1080 (8 GB).

Keeps models in VRAM with priority-based eviction to prevent OOM crashes.
"""
from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import subprocess
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

# VRAM constants for GTX 1080 8GB
VRAM_TOTAL_MB = 8_192
VRAM_SYSTEM_RESERVE_MB = 512
VRAM_SAFETY_BUFFER_MB = 256
VRAM_PRESSURE_THRESHOLD_MB = 1_024
VRAM_BUDGET_MB = VRAM_TOTAL_MB - VRAM_SYSTEM_RESERVE_MB - VRAM_SAFETY_BUFFER_MB


class ModelPriority(Enum):
    """Priority levels for VRAM residency."""

    CRITICAL = 0  # Never evicted
    HIGH = 1  # Evicted last
    MEDIUM = 2  # Evicted after LOW
    LOW = 3  # Evicted first


@dataclass
class ManagedModel:
    """A model registered with the VRAM manager."""

    name: str
    vram_mb: int
    priority: ModelPriority
    load_fn: Callable[[], Any]
    unload_fn: Callable[[Any], None]
    obj: Any = field(default=None, repr=False)
    loaded: bool = False
    last_used: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock | None = field(default=None, repr=False, init=False)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


class VRAMManager:
    """Manages GPU VRAM with priority-based model eviction."""

    def __init__(self) -> None:
        self._models: dict[str, ManagedModel] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._monitor: asyncio.Task | None = None

    def register(
        self,
        name: str,
        vram_mb: int,
        priority: ModelPriority,
        load_fn: Callable,
        unload_fn: Callable,
    ) -> None:
        """Register a model for VRAM management."""
        self._models[name] = ManagedModel(
            name=name,
            vram_mb=vram_mb,
            priority=priority,
            load_fn=load_fn,
            unload_fn=unload_fn,
        )
        log.info("VRAM: registered '%s' (%d MB)", name, vram_mb)

    @asynccontextmanager
    async def lease(self, name: str):
        """Context manager: load model if needed, yield it, keep loaded."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered")
        m = self._models[name]
        async with m.lock:
            if not m.loaded:
                await self._ensure_space(m.vram_mb, exclude=name)
                await self._load(m)
            m.last_used = time.monotonic()
            try:
                yield m.obj
            finally:
                pass

    async def evict(self, name: str) -> None:
        """Unload a model from VRAM."""
        m = self._models.get(name)
        if m and m.loaded:
            async with m.lock:
                if m.loaded:
                    await self._unload(m)

    async def evict_all_except(self, keep: str) -> None:
        """Evict all models except the named one."""
        for name, m in list(self._models.items()):
            if name != keep and m.loaded:
                await self.evict(name)

    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded."""
        return self._models.get(name, ManagedModel(
            "", 0, ModelPriority.LOW, lambda: None, lambda _: None
        )).loaded

    def free_vram_mb(self) -> int:
        """Query current free VRAM in megabytes."""
        return self._query_free_vram()

    def used_vram_mb(self) -> int:
        """Calculate VRAM used by managed models."""
        return sum(m.vram_mb for m in self._models.values() if m.loaded)

    async def _ensure_space(self, needed_mb: int, exclude: str) -> None:
        """Free VRAM by evicting lower-priority models."""
        if self._query_free_vram() >= needed_mb:
            return
        log.warning("VRAM: need %d MB, evicting low-priority models", needed_mb)
        candidates = sorted(
            [m for n, m in self._models.items() if m.loaded and n != exclude],
            key=lambda m: (-m.priority.value, m.last_used),
        )
        for candidate in candidates:
            if self._query_free_vram() >= needed_mb:
                break
            await self.evict(candidate.name)
        if self._query_free_vram() < needed_mb:
            free = self._query_free_vram()
            raise MemoryError(
                f"Cannot free enough VRAM for '{exclude}': "
                f"need {needed_mb} MB, have {free} MB"
            )

    async def _load(self, m: ManagedModel) -> None:
        """Load a model into VRAM."""
        log.info("VRAM: loading '%s' …", m.name)
        loop = asyncio.get_running_loop()
        m.obj = await loop.run_in_executor(None, m.load_fn)
        m.loaded = True
        m.last_used = time.monotonic()
        log.info("VRAM: '%s' loaded", m.name)

    async def _unload(self, m: ManagedModel) -> None:
        """Unload a model from VRAM."""
        log.info("VRAM: unloading '%s'", m.name)
        loop = asyncio.get_running_loop()
        with contextlib.suppress(Exception):
            await loop.run_in_executor(None, m.unload_fn, m.obj)
        m.obj = None
        m.loaded = False
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def _query_free_vram() -> int:
        """Query free VRAM via nvidia-smi or torch fallback."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                timeout=2,
                stderr=subprocess.DEVNULL,
            )
            return int(out.decode().strip().split("\n")[0])
        except Exception:
            pass
        if _HAS_TORCH and torch is not None and torch.cuda.is_available():
            try:
                free, _ = torch.cuda.mem_get_info(0)
                return free // (1024 * 1024)
            except Exception:
                pass
        # Fallback: assume full budget available
        return VRAM_BUDGET_MB

    async def start_monitor(self, poll_interval: float = 6.0) -> None:
        """Start the VRAM pressure monitor."""
        self._loop = asyncio.get_running_loop()
        self._monitor = asyncio.create_task(
            self._monitor_loop(poll_interval), name="vram-monitor"
        )

    async def _monitor_loop(self, interval: float) -> None:
        """Periodically check VRAM and emit pressure events."""
        while True:
            await asyncio.sleep(interval)
            try:
                free = self._query_free_vram()
                if free < VRAM_PRESSURE_THRESHOLD_MB:
                    await bus.emit(
                        Event.VRAM_PRESSURE,
                        data={"free_mb": free, "used_mb": self.used_vram_mb()},
                        source="VRAMManager",
                        priority=1,
                    )
            except Exception:
                log.exception("VRAM monitor error")

    async def stop(self) -> None:
        """Stop monitoring and unload all models."""
        if self._monitor:
            self._monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor
        for m in list(self._models.values()):
            if m.loaded:
                with contextlib.suppress(Exception):
                    await self._unload(m)


# Global singleton
vram = VRAMManager()
