"""
nixorb/core/vram_manager.py

Aggressive VRAM paging for GTX 1080 (8 GB).

Strategy:
  - Each model is registered with an estimated VRAM cost in MB.
  - Only one "heavy" model (Whisper Large v3 ~2.8 GB or LLM) is
    resident at a time; others are offloaded to CPU RAM.
  - A context-manager API makes load/unload transparent to callers.
  - Real-time NVIDIA SMI polling raises VRAM_PRESSURE events via the
    EventBus when headroom drops below a threshold.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

import torch

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  VRAM budget constants (MB)                                                  #
# --------------------------------------------------------------------------- #
VRAM_TOTAL_MB        = 8192
VRAM_SYSTEM_RESERVE  = 512    # driver + KDE compositor overhead
VRAM_SAFETY_BUFFER   = 256    # emergency headroom
VRAM_PRESSURE_THRESH = 1024   # emit VRAM_PRESSURE below this free MB
VRAM_BUDGET          = VRAM_TOTAL_MB - VRAM_SYSTEM_RESERVE - VRAM_SAFETY_BUFFER


class ModelPriority(Enum):
    """Higher priority = evicted last."""
    CRITICAL = 0   # never evict (tiny utility models)
    HIGH     = 1   # LLM currently responding
    MEDIUM   = 2   # TTS
    LOW      = 3   # ASR (loaded on demand)


@dataclass
class ManagedModel:
    name: str
    vram_mb: int
    priority: ModelPriority
    load_fn: Callable[[], Any]       # returns the model object
    unload_fn: Callable[[Any], None] # releases it
    obj: Any = field(default=None, repr=False)
    loaded: bool = False
    last_used: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class VRAMManager:
    """
    Central VRAM lifecycle controller.

    Usage:
        async with vram.lease("whisper") as model:
            result = model.transcribe(audio)
    """

    def __init__(self) -> None:
        self._models: dict[str, ManagedModel] = {}
        self._global_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task | None = None

    # ---------------------------------------------------------------------- #
    #  Registration                                                            #
    # ---------------------------------------------------------------------- #
    def register(
        self,
        name: str,
        vram_mb: int,
        priority: ModelPriority,
        load_fn: Callable,
        unload_fn: Callable,
    ) -> None:
        self._models[name] = ManagedModel(
            name=name,
            vram_mb=vram_mb,
            priority=priority,
            load_fn=load_fn,
            unload_fn=unload_fn,
        )
        log.info("VRAMManager: registered '%s' (%d MB)", name, vram_mb)

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #
    @asynccontextmanager
    async def lease(self, name: str):
        """
        Async context manager that ensures the named model is in VRAM.
        Evicts lower-priority models if needed.
        """
        m = self._models[name]
        async with m.lock:
            if not m.loaded:
                await self._ensure_space(m.vram_mb, exclude=name)
                await self._load(m)
            m.last_used = time.monotonic()
            try:
                yield m.obj
            finally:
                pass   # keep loaded; monitor handles lazy eviction

    async def evict(self, name: str) -> None:
        """Forcibly unload a model from VRAM."""
        m = self._models.get(name)
        if m and m.loaded:
            async with m.lock:
                await self._unload(m)

    async def evict_all_except(self, keep: str) -> None:
        for name, m in self._models.items():
            if name != keep and m.loaded:
                async with m.lock:
                    await self._unload(m)

    # ---------------------------------------------------------------------- #
    #  Internal helpers                                                        #
    # ---------------------------------------------------------------------- #
    async def _ensure_space(self, needed_mb: int, exclude: str) -> None:
        free = self._free_vram_mb()
        if free >= needed_mb:
            return

        log.warning("VRAM low (%d MB free), need %d MB — evicting", free, needed_mb)

        # Sort loaded models by priority (evict LOW first) then by LRU
        candidates = sorted(
            [m for n, m in self._models.items() if m.loaded and n != exclude],
            key=lambda m: (-m.priority.value, m.last_used),
        )
        for candidate in candidates:
            if self._free_vram_mb() >= needed_mb:
                break
            async with candidate.lock:
                await self._unload(candidate)

        if self._free_vram_mb() < needed_mb:
            raise MemoryError(
                f"Cannot free enough VRAM for '{exclude}' "
                f"(need {needed_mb} MB, have {self._free_vram_mb()} MB)"
            )

    async def _load(self, m: ManagedModel) -> None:
        log.info("VRAMManager: loading '%s' (%d MB)", m.name, m.vram_mb)
        loop = asyncio.get_running_loop()
        m.obj = await loop.run_in_executor(None, m.load_fn)
        m.loaded = True
        m.last_used = time.monotonic()
        log.info("VRAMManager: '%s' loaded — free VRAM now %d MB",
                 m.name, self._free_vram_mb())

    async def _unload(self, m: ManagedModel) -> None:
        log.info("VRAMManager: unloading '%s'", m.name)
        try:
            m.unload_fn(m.obj)
        except Exception:
            log.exception("Error during unload of '%s'", m.name)
        m.obj = None
        m.loaded = False
        # Force CUDA to release cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("VRAMManager: '%s' unloaded — free VRAM now %d MB",
                 m.name, self._free_vram_mb())

    @staticmethod
    def _free_vram_mb() -> int:
        """Query nvidia-smi for live free VRAM."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                timeout=2,
            )
            return int(out.decode().strip().split("\n")[0])
        except Exception:
            # Fallback to torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info(0)
                return free // (1024 * 1024)
            return VRAM_BUDGET

    # ---------------------------------------------------------------------- #
    #  Background monitor                                                      #
    # ---------------------------------------------------------------------- #
    async def start_monitor(self, poll_interval: float = 5.0) -> None:
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(poll_interval), name="vram-monitor"
        )

    async def _monitor_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            free = self._free_vram_mb()
            if free < VRAM_PRESSURE_THRESH:
                await bus.emit(
                    Event.VRAM_PRESSURE,
                    data={"free_mb": free},
                    source="vram_manager",
                    priority=1,
                )
                log.warning("VRAM pressure: %d MB free", free)

    async def stop(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
        for m in self._models.values():
            if m.loaded:
                await self._unload(m)


vram = VRAMManager()
