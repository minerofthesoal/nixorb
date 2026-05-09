"""
nixorb/core/event_bus.py

Central asyncio EventBus with priority queuing and wildcard subscriptions.

BUG FIX PASS 1:
  - EventPayload was placed directly into PriorityQueue as the second element
    of a tuple. When two priorities are equal Python falls through to comparing
    the second element (EventPayload). EventPayload has no __lt__ so this
    raised a TypeError at runtime. Fixed by inserting a monotonic integer
    counter as the tiebreaker: (priority, counter, payload).

BUG FIX PASS 2:
  - _Handler type alias used Coroutine which is not the right abstract for
    async callables. Changed to Awaitable to be compatible with both coroutine
    functions and async generators.

BUG FIX PASS 3:
  - emit() raised AttributeError if called before start() (queue was None).
    Added a guard that initialises the queue lazily.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# Monotonic counter used as tiebreaker in the priority queue so that
# EventPayload objects are never compared directly (they have no ordering).
_counter = itertools.count()


class Event(Enum):
    # ── Audio / ASR ──────────────────────────────────────────────── #
    HOTKEY_TRIGGERED    = auto()
    WAKE_WORD_DETECTED  = auto()
    RECORDING_START     = auto()
    RECORDING_STOP      = auto()
    TRANSCRIPT_READY    = auto()

    # ── LLM ──────────────────────────────────────────────────────── #
    LLM_THINKING        = auto()
    LLM_CHUNK           = auto()
    LLM_DONE            = auto()
    LLM_ERROR           = auto()

    # ── TTS ──────────────────────────────────────────────────────── #
    TTS_START           = auto()
    TTS_AUDIO_CHUNK     = auto()   # carries raw PCM bytes for orb animation
    TTS_DONE            = auto()

    # ── Orb UI ───────────────────────────────────────────────────── #
    ORB_IDLE            = auto()
    ORB_LISTENING       = auto()
    ORB_THINKING        = auto()
    ORB_SPEAKING        = auto()
    ORB_ERROR           = auto()

    # ── Action engine ────────────────────────────────────────────── #
    ACTION_REQUESTED    = auto()
    ACTION_RESULT       = auto()

    # ── Vision ───────────────────────────────────────────────────── #
    SCREEN_CAPTURE_REQ  = auto()
    SCREEN_CAPTURE_DONE = auto()

    # ── System ───────────────────────────────────────────────────── #
    VRAM_PRESSURE       = auto()
    SETTINGS_CHANGED    = auto()
    SHUTDOWN            = auto()
    LOG                 = auto()


@dataclass
class EventPayload:
    event:    Event
    data:     Any  = None
    source:   str  = "unknown"
    priority: int  = 5       # 0 = highest priority


# BUG FIX: Callable[[EventPayload], Coroutine] is wrong — use Awaitable.
_Handler = Callable[[EventPayload], Awaitable[None]]


class EventBus:
    """
    Singleton async event bus.

    Usage from async code:
        await bus.emit(Event.HOTKEY_TRIGGERED)

    Usage from Qt slots / threads:
        bus.emit_sync(Event.HOTKEY_TRIGGERED)
    """

    _instance: "EventBus | None" = None

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._initialised = False
            cls._instance = obj
        return cls._instance

    def _ensure_init(self) -> None:
        if self._initialised:
            return
        self._handlers: dict[Event, list[tuple[int, _Handler]]] = defaultdict(list)
        self._wildcard:  list[tuple[int, _Handler]] = []
        # BUG FIX: queue is None until start(); emit() now lazily creates it
        # so callers don't crash if they emit before start() is awaited.
        self._queue: asyncio.PriorityQueue[tuple[int, int, EventPayload]] = (
            asyncio.PriorityQueue()
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._initialised = True

    # ---------------------------------------------------------------- #
    #  Lifecycle                                                         #
    # ---------------------------------------------------------------- #
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
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Merge specific + wildcard handlers, sorted by handler priority
            handlers: list[tuple[int, _Handler]] = (
                list(self._handlers.get(payload.event, [])) + list(self._wildcard)
            )
            handlers.sort(key=lambda t: t[0])

            for _hpri, handler in handlers:
                try:
                    await handler(payload)
                except Exception:
                    log.exception(
                        "Handler %s raised for event %s", handler, payload.event
                    )

            self._queue.task_done()

    async def stop(self) -> None:
        self._running = False
        # Drain remaining items
        try:
            await asyncio.wait_for(self._queue.join(), timeout=3.0)
        except asyncio.TimeoutError:
            log.warning("EventBus drain timed out — some events may be lost")
        log.info("EventBus stopped")

    # ---------------------------------------------------------------- #
    #  Emit                                                              #
    # ---------------------------------------------------------------- #
    async def emit(
        self,
        event:    Event,
        data:     Any = None,
        source:   str = "unknown",
        priority: int = 5,
    ) -> None:
        self._ensure_init()
        payload = EventPayload(event=event, data=data, source=source, priority=priority)
        # BUG FIX: tuple is (priority, counter, payload) — counter breaks ties
        # so Python never compares two EventPayload objects.
        await self._queue.put((priority, next(_counter), payload))

    def emit_sync(
        self,
        event:    Event,
        data:     Any = None,
        source:   str = "unknown",
        priority: int = 5,
    ) -> None:
        """Thread-safe emit from Qt slots or non-async code."""
        self._ensure_init()
        loop = self._loop
        if loop is None or not loop.is_running():
            log.warning("emit_sync called but event loop not running — event dropped")
            return
        payload = EventPayload(event=event, data=data, source=source, priority=priority)
        asyncio.run_coroutine_threadsafe(
            self._queue.put((priority, next(_counter), payload)), loop
        )

    # ---------------------------------------------------------------- #
    #  Subscribe / Unsubscribe                                           #
    # ---------------------------------------------------------------- #
    def subscribe(
        self,
        event:    Event | None,
        handler:  _Handler,
        priority: int = 5,
    ) -> None:
        """
        Subscribe *handler* to *event*.
        Pass ``event=None`` to receive every event (wildcard).
        Lower *priority* values are called first.
        """
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


# Module-level singleton — import and use directly.
bus = EventBus()
