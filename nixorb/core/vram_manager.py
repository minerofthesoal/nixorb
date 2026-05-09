"""
nixorb/core/vram_manager.py

VRAM lifecycle manager for GTX 1080 (8 GB).

BUG FIX PASS 1:
  - ManagedModel.lock = field(default_factory=asyncio.Lock) creates the Lock
    at dataclass construction time, which may precede the event loop on some
    Python 3.10+ builds. Changed to lazy initialisation via a property.

BUG FIX PASS 2:
  - _load / _unload were called with `await loop.run_in_executor(None, fn)`
    but run_in_executor is not awaitable directly from here — it returns a
    Future. Wrapped correctly.

BUG FIX PASS 3:
  - ModelPriority comparison in sort was reversed (evict LOW priority = 3
    first, which means highest .value first). Sort key corrected.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

VRAM_TOTAL_MB        = 8_192
VRAM_SYSTEM_RESERVE  =   512   # driver + KDE compositor
VRAM_SAFETY_BUFFER   =   256   # never touch this headroom
VRAM_PRESSURE_THRESH = 1_024   # emit VRAM_PRESSURE below this
VRAM_BUDGET          = VRAM_TOTAL_MB - VRAM_SYSTEM_RESERVE - VRAM_SAFETY_BUFFER


class ModelPriority(Enum):
    """Lower value = higher priority = evicted last."""
    CRITICAL = 0
    HIGH     = 1
    MEDIUM   = 2
    LOW      = 3


@dataclass
class ManagedModel:
    name:      str
    vram_mb:   int
    priority:  ModelPriority
    load_fn:   Callable[[], Any]
    unload_fn: Callable[[Any], None]
    obj:       Any   = field(default=None,  repr=False)
    loaded:    bool  = False
    last_used: float = field(default_factory=time.monotonic)
    # BUG FIX: do NOT create asyncio.Lock as a dataclass default_factory
    # because it binds to whatever loop (or no loop) exists at instantiation.
    # Use a property that creates it lazily on first access instead.
    _lock:     asyncio.Lock | None = field(default=None, repr=False, init=False)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


class VRAMManager:
    """
    Central VRAM lifecycle controller.

    Example:
        async with vram.lease("whisper") as model:
            segments, _ = model.transcribe(audio)
    """

    def __init__(self) -> None:
        self._models:  dict[str, ManagedModel] = {}
        self._loop:    asyncio.AbstractEventLoop | None = None
        self._monitor: asyncio.Task | None = None

    # ---------------------------------------------------------------- #
    #  Registration API                                                  #
    # ---------------------------------------------------------------- #
    def register(
        self,
        name:      str,
        vram_mb:   int,
        priority:  ModelPriority,
        load_fn:   Callable,
        unload_fn: Callable,
    ) -> None:
        if name in self._models:
            log.debug("VRAMManager: re-registering '%s'", name)
        self._models[name] = ManagedModel(
            name=name, vram_mb=vram_mb, priority=priority,
            load_fn=load_fn, unload_fn=unload_fn,
        )
        log.info("VRAMManager: registered '%s' (%d MB, priority=%s)",
                 name, vram_mb, priority.name)

    # ---------------------------------------------------------------- #
    #  Public API                                                        #
    # ---------------------------------------------------------------- #
    @asynccontextmanager
    async def lease(self, name: str):
        """
        Async context manager — ensures *name* is loaded in VRAM.
        Evicts lower-priority models as needed before loading.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not registered with VRAMManager")
        m = self._models[name]

        async with m.lock:
            if not m.loaded:
                await self._ensure_space(m.vram_mb, exclude=name)
                await self._load(m)
            m.last_used = time.monotonic()
            try:
                yield m.obj
            finally:
                pass   # keep model warm; monitor handles lazy eviction

    async def evict(self, name: str) -> None:
        """Forcibly unload *name* from VRAM immediately."""
        m = self._models.get(name)
        if m and m.loaded:
            async with m.lock:
                if m.loaded:   # re-check inside lock
                    await self._unload(m)

    async def evict_all_except(self, keep: str) -> None:
        for name, m in list(self._models.items()):
            if name != keep and m.loaded:
                await self.evict(name)

    def is_loaded(self, name: str) -> bool:
        return self._models.get(name, ManagedModel("", 0, ModelPriority.LOW,
                                                    lambda: None, lambda _: None)).loaded

    def free_vram_mb(self) -> int:
        return self._query_free_vram()

    # ---------------------------------------------------------------- #
    #  Internal helpers                                                  #
    # ---------------------------------------------------------------- #
    async def _ensure_space(self, needed_mb: int, exclude: str) -> None:
        free = self._query_free_vram()
        if free >= needed_mb:
            return

        log.warning(
            "VRAM: need %d MB, have %d MB free — evicting candidates", needed_mb, free
        )

        # BUG FIX: sort key was wrong. We want to evict LOW priority (value=3)
        # first, then break ties by LRU (smallest last_used first).
        # Correct key: (-priority.value, last_used) → sort descending by priority
        # value so LOW (3) comes first, then LRU within same priority.
        candidates = sorted(
            [m for n, m in self._models.items() if m.loaded and n != exclude],
            key=lambda m: (-m.priority.value, m.last_used),
        )

        for candidate in candidates:
            if self._query_free_vram() >= needed_mb:
                break
            log.info("VRAMManager: evicting '%s' to free space", candidate.name)
            await self.evict(candidate.name)

        free = self._query_free_vram()
        if free < needed_mb:
            raise MemoryError(
                f"Cannot free enough VRAM for '{exclude}': "
                f"need {needed_mb} MB, only {free} MB available"
            )

    async def _load(self, m: ManagedModel) -> None:
        log.info("VRAMManager: loading '%s' (%d MB) …", m.name, m.vram_mb)
        loop = asyncio.get_running_loop()
        # BUG FIX: run_in_executor returns a coroutine-like Future — must await it.
        m.obj    = await loop.run_in_executor(None, m.load_fn)
        m.loaded = True
        m.last_used = time.monotonic()
        log.info(
            "VRAMManager: '%s' loaded — free VRAM now %d MB",
            m.name, self._query_free_vram(),
        )

    async def _unload(self, m: ManagedModel) -> None:
        log.info("VRAMManager: unloading '%s'", m.name)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, m.unload_fn, m.obj)
        except Exception:
            log.exception("Error while unloading '%s'", m.name)
        finally:
            m.obj    = None
            m.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info(
            "VRAMManager: '%s' unloaded — free VRAM now %d MB",
            m.name, self._query_free_vram(),
        )

    @staticmethod
    def _query_free_vram() -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return int(out.decode().strip().split("\n")[0])
        except Exception:
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info(0)
                return free // (1024 * 1024)
            return VRAM_BUDGET

    # ---------------------------------------------------------------- #
    #  Background monitor                                                #
    # ---------------------------------------------------------------- #
    async def start_monitor(self, poll_interval: float = 5.0) -> None:
        self._loop    = asyncio.get_running_loop()
        self._monitor = asyncio.create_task(
            self._monitor_loop(poll_interval), name="nixorb-vram-monitor"
        )

    async def _monitor_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            free = self._query_free_vram()
            if free < VRAM_PRESSURE_THRESH:
                await bus.emit(
                    Event.VRAM_PRESSURE,
                    data={"free_mb": free},
                    source="VRAMManager",
                    priority=1,
                )
                log.warning("VRAM pressure: %d MB free", free)

    async def stop(self) -> None:
        if self._monitor:
            self._monitor.cancel()
        for m in list(self._models.values()):
            if m.loaded:
                try:
                    await self._unload(m)
                except Exception:
                    pass


# Module-level singleton
vram = VRAMManager()
