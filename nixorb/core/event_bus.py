"""
nixorb/core/event_bus.py
Central asyncio-based EventBus. All inter-component communication
goes through here — zero direct imports between subsystems.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)


class Event(Enum):
    # Audio / ASR
    HOTKEY_TRIGGERED    = auto()
    WAKE_WORD_DETECTED  = auto()
    RECORDING_START     = auto()
    RECORDING_STOP      = auto()
    TRANSCRIPT_READY    = auto()

    # LLM
    LLM_THINKING        = auto()
    LLM_CHUNK           = auto()
    LLM_DONE            = auto()
    LLM_ERROR           = auto()

    # TTS
    TTS_START           = auto()
    TTS_AUDIO_CHUNK     = auto()   # carries raw PCM for orb animation
    TTS_DONE            = auto()

    # Orb UI
    ORB_IDLE            = auto()
    ORB_LISTENING       = auto()
    ORB_THINKING        = auto()
    ORB_SPEAKING        = auto()
    ORB_ERROR           = auto()

    # Action engine
    ACTION_REQUESTED    = auto()
    ACTION_RESULT       = auto()

    # Vision
    SCREEN_CAPTURE_REQ  = auto()
    SCREEN_CAPTURE_DONE = auto()

    # System
    VRAM_PRESSURE       = auto()
    SETTINGS_CHANGED    = auto()
    SHUTDOWN            = auto()
    LOG                 = auto()


@dataclass
class EventPayload:
    event: Event
    data: Any = None
    source: str = "unknown"
    priority: int = 5           # 0 = highest


_Handler = Callable[[EventPayload], Coroutine]


class EventBus:
    """
    Singleton async event bus with priority queuing and wildcard subscription.
    Thread-safe: call bus.emit_sync() from non-async contexts.
    """

    _instance: EventBus | None = None

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._handlers: dict[Event, list[tuple[int, _Handler]]] = defaultdict(list)
        self._wildcard: list[tuple[int, _Handler]] = []
        self._queue: asyncio.PriorityQueue[tuple[int, EventPayload]] = None  # set in start()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.PriorityQueue()
        self._running = True
        asyncio.create_task(self._dispatch_loop(), name="event-bus-dispatch")
        log.info("EventBus started")

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                _prio, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            handlers = self._handlers.get(payload.event, []) + self._wildcard
            handlers_sorted = sorted(handlers, key=lambda x: x[0])

            for _pri, handler in handlers_sorted:
                try:
                    await handler(payload)
                except Exception:
                    log.exception("Handler %s raised for event %s", handler, payload.event)

            self._queue.task_done()

    async def emit(self, event: Event, data: Any = None,
                   source: str = "unknown", priority: int = 5) -> None:
        payload = EventPayload(event=event, data=data,
                               source=source, priority=priority)
        await self._queue.put((priority, payload))

    def emit_sync(self, event: Event, data: Any = None,
                  source: str = "unknown", priority: int = 5) -> None:
        """Thread-safe emit from non-async code (e.g. Qt slots)."""
        if self._loop and self._loop.is_running():
            payload = EventPayload(event=event, data=data,
                                   source=source, priority=priority)
            asyncio.run_coroutine_threadsafe(
                self._queue.put((priority, payload)), self._loop
            )

    def subscribe(self, event: Event | None, handler: _Handler,
                  priority: int = 5) -> None:
        """Subscribe to a specific event, or None for all events (wildcard)."""
        if event is None:
            self._wildcard.append((priority, handler))
        else:
            self._handlers[event].append((priority, handler))

    def unsubscribe(self, event: Event | None, handler: _Handler) -> None:
        if event is None:
            self._wildcard = [(p, h) for p, h in self._wildcard if h != handler]
        else:
            self._handlers[event] = [
                (p, h) for p, h in self._handlers[event] if h != handler
            ]

    async def stop(self) -> None:
        self._running = False
        await self._queue.join()
        log.info("EventBus stopped")


bus = EventBus()
