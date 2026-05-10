"""nixorb/core/vram_manager.py — VRAM lifecycle manager for GTX 1080 (8 GB)."""
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

import torch

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

VRAM_TOTAL_MB        = 8_192
VRAM_SYSTEM_RESERVE  =   512
VRAM_SAFETY_BUFFER   =   256
VRAM_PRESSURE_THRESH = 1_024
VRAM_BUDGET          = VRAM_TOTAL_MB - VRAM_SYSTEM_RESERVE - VRAM_SAFETY_BUFFER


class ModelPriority(Enum):
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
    obj:       Any            = field(default=None,  repr=False)
    loaded:    bool           = False
    last_used: float          = field(default_factory=time.monotonic)
    _lock:     asyncio.Lock | None = field(default=None, repr=False, init=False)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock


class VRAMManager:
    def __init__(self) -> None:
        self._models:  dict[str, ManagedModel] = {}
        self._loop:    asyncio.AbstractEventLoop | None = None
        self._monitor: asyncio.Task | None = None

    def register(
        self,
        name:      str,
        vram_mb:   int,
        priority:  ModelPriority,
        load_fn:   Callable,
        unload_fn: Callable,
    ) -> None:
        self._models[name] = ManagedModel(
            name=name, vram_mb=vram_mb, priority=priority,
            load_fn=load_fn, unload_fn=unload_fn,
        )
        log.info("VRAMManager: registered '%s' (%d MB)", name, vram_mb)

    @asynccontextmanager
    async def lease(self, name: str):
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
                pass

    async def evict(self, name: str) -> None:
        m = self._models.get(name)
        if m and m.loaded:
            async with m.lock:
                if m.loaded:
                    await self._unload(m)

    async def evict_all_except(self, keep: str) -> None:
        for name, m in list(self._models.items()):
            if name != keep and m.loaded:
                await self.evict(name)

    def is_loaded(self, name: str) -> bool:
        return self._models.get(name, ManagedModel(
            "", 0, ModelPriority.LOW, lambda: None, lambda _: None
        )).loaded

    def free_vram_mb(self) -> int:
        return self._query_free_vram()

    async def _ensure_space(self, needed_mb: int, exclude: str) -> None:
        if self._query_free_vram() >= needed_mb:
            return
        log.warning("VRAM: need %d MB — evicting", needed_mb)
        candidates = sorted(
            [m for n, m in self._models.items() if m.loaded and n != exclude],
            key=lambda m: (-m.priority.value, m.last_used),
        )
        for candidate in candidates:
            if self._query_free_vram() >= needed_mb:
                break
            await self.evict(candidate.name)
        if self._query_free_vram() < needed_mb:
            raise MemoryError(
                f"Cannot free enough VRAM for '{exclude}': "
                f"need {needed_mb} MB, have {self._query_free_vram()} MB"
            )

    async def _load(self, m: ManagedModel) -> None:
        log.info("VRAMManager: loading '%s' …", m.name)
        loop = asyncio.get_running_loop()
        m.obj    = await loop.run_in_executor(None, m.load_fn)
        m.loaded = True
        m.last_used = time.monotonic()

    async def _unload(self, m: ManagedModel) -> None:
        log.info("VRAMManager: unloading '%s'", m.name)
        loop = asyncio.get_running_loop()
        with contextlib.suppress(Exception):
            await loop.run_in_executor(None, m.unload_fn, m.obj)
        m.obj    = None
        m.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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

    async def start_monitor(self, poll_interval: float = 6.0) -> None:
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
                    Event.VRAM_PRESSURE, data={"free_mb": free},
                    source="VRAMManager", priority=1,
                )

    async def stop(self) -> None:
        if self._monitor:
            self._monitor.cancel()
        for m in list(self._models.values()):
            if m.loaded:
                with contextlib.suppress(Exception):
                    await self._unload(m)


vram = VRAMManager()
