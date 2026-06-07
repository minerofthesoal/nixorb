"""
nixorb/core/event_bus.py — Central asyncio EventBus.

BUG FIXES:
  - PriorityQueue tiebreaker counter (Python compares tuple elements in order;
    without a counter two equal-priority payloads would compare EventPayload
    objects which have no __lt__, raising TypeError).
  - emit_sync() now guards against None _loop gracefully.
  - _ensure_init() called lazily so the singleton works across test resets.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, cast

log = logging.getLogger(__name__)

_counter = itertools.count()   # tiebreaker — never compared as EventPayload


class Event(Enum):
    HOTKEY_TRIGGERED    = auto()
    WAKE_WORD_DETECTED  = auto()
    RECORDING_START     = auto()
    RECORDING_STOP      = auto()
    TRANSCRIPT_READY    = auto()
    LLM_THINKING        = auto()
    LLM_CHUNK           = auto()
    LLM_DONE            = auto()
    LLM_ERROR           = auto()
    TTS_START           = auto()
    TTS_AUDIO_CHUNK     = auto()
    TTS_DONE            = auto()
    ORB_IDLE            = auto()
    ORB_LISTENING       = auto()
    ORB_THINKING        = auto()
    ORB_SPEAKING        = auto()
    ORB_ERROR           = auto()
    ACTION_REQUESTED    = auto()
    ACTION_RESULT       = auto()
    SCREEN_CAPTURE_REQ  = auto()
    SCREEN_CAPTURE_DONE = auto()
    VRAM_PRESSURE       = auto()
    SETTINGS_CHANGED    = auto()
    SHUTDOWN            = auto()
    LOG                 = auto()
    MIC_MUTED           = auto()
    MIC_LEVEL           = auto()


@dataclass
class EventPayload:
    event:    Event
    data:     Any  = None
    source:   str  = "unknown"
    priority: int  = 5


_Handler = Callable[[EventPayload], Awaitable[None]]


class EventBus:
    _instance: EventBus | None = None
    _initialised: bool
    _handlers: dict[Event, list[tuple[int, _Handler]]]
    _wildcard: list[tuple[int, _Handler]]
    _queue: asyncio.PriorityQueue[tuple[int, int, EventPayload]]
    _loop: asyncio.AbstractEventLoop | None
    _running: bool

    def __new__(cls) -> EventBus:
        if cls._instance is None:
            obj = cast(EventBus, super().__new__(cls))
            obj._initialised = False
            cls._instance = obj
        return cls._instance

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        self._handlers: dict[Event, list[tuple[int, _Handler]]] = defaultdict(list)
        self._wildcard:  list[tuple[int, _Handler]] = []
        self._queue: asyncio.PriorityQueue[tuple[int, int, EventPayload]] = (
            asyncio.PriorityQueue()
        )
        self._loop:    asyncio.AbstractEventLoop | None = None
        self._running: bool = False
        self._initialised  = True

    async def start(self) -> None:
        self._ensure_init()
        self._loop    = asyncio.get_running_loop()
        self._running = True
        asyncio.create_task(self._dispatch_loop(), name="nixorb-event-bus")
        log.info("EventBus started")

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                _pri, _seq, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            handlers = (
                list(self._handlers.get(payload.event, [])) + list(self._wildcard)
            )
            handlers.sort(key=lambda t: t[0])

            for _hpri, handler in handlers:
                try:
                    await handler(payload)
                except Exception:
                    log.exception("Handler %s raised for %s", handler, payload.event)

            self._queue.task_done()

    async def stop(self) -> None:
        self._running = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=3.0)
        except TimeoutError:
            log.warning("EventBus drain timed out")
        log.info("EventBus stopped")

    async def emit(
        self,
        event:    Event,
        data:     Any = None,
        source:   str = "unknown",
        priority: int = 5,
    ) -> None:
        self._ensure_init()
        payload = EventPayload(event=event, data=data, source=source, priority=priority)
        await self._queue.put((priority, next(_counter), payload))

    def emit_sync(
        self,
        event:    Event,
        data:     Any = None,
        source:   str = "unknown",
        priority: int = 5,
    ) -> None:
        self._ensure_init()
        loop = self._loop
        if loop is None or not loop.is_running():
            log.warning("emit_sync: loop not running — event %s dropped", event)
            return
        payload = EventPayload(event=event, data=data, source=source, priority=priority)
        asyncio.run_coroutine_threadsafe(
            self._queue.put((priority, next(_counter), payload)), loop
        )

    def subscribe(
        self,
        event:    Event | None,
        handler:  _Handler,
        priority: int = 5,
    ) -> None:
        self._ensure_init()
        if event is None:
            self._wildcard.append((priority, handler))
        else:
            self._handlers[event].append((priority, handler))

    def unsubscribe(self, event: Event | None, handler: _Handler) -> None:
        self._ensure_init()
        if event is None:
            self._wildcard = [(p, h) for p, h in self._wildcard if h is not handler]
        else:
            self._handlers[event] = [
                (p, h) for p, h in self._handlers[event] if h is not handler
            ]


bus = EventBus()
