"""NixOrb event bus — central async event system.

All communication between modules happens through typed events on this bus.
This eliminates direct coupling between UI, ASR, LLM, TTS, and other components.
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast

log = logging.getLogger(__name__)

# Global tiebreaker counter — prevents PriorityQueue comparison errors
counter = itertools.count()

# A handler holding the dispatch loop longer than this is a bug worth
# reporting: everything else on the bus is stuck behind it.
SLOW_HANDLER_SECONDS = 5.0


class Event(Enum):
    """All events that can flow through the NixOrb event bus."""

    # Trigger events
    HOTKEY_TRIGGERED = auto()
    WAKE_WORD_DETECTED = auto()
    ORB_CLICKED = auto()

    # Recording / ASR
    RECORDING_START = auto()
    RECORDING_STOP = auto()
    MIC_LEVEL = auto()
    MIC_MUTED = auto()
    TRANSCRIPT_READY = auto()
    ASR_READY = auto()
    ASR_ERROR = auto()

    # LLM
    LLM_START = auto()
    LLM_CHUNK = auto()
    LLM_DONE = auto()
    LLM_ERROR = auto()

    # TTS
    TTS_START = auto()
    TTS_AUDIO_CHUNK = auto()
    TTS_DONE = auto()
    TTS_ERROR = auto()

    # Orb state
    ORB_IDLE = auto()
    ORB_LISTENING = auto()
    ORB_THINKING = auto()
    ORB_SPEAKING = auto()
    ORB_ERROR = auto()

    # Actions
    ACTION_REQUESTED = auto()
    ACTION_CONFIRMED = auto()
    ACTION_DENIED = auto()
    ACTION_RESULT = auto()

    # Screen / vision
    SCREEN_CAPTURE_REQ = auto()
    SCREEN_CAPTURE_DONE = auto()

    # VRAM
    VRAM_PRESSURE = auto()

    # Settings
    SETTINGS_CHANGED = auto()

    # Lifecycle
    SHUTDOWN = auto()

    # Logging
    LOG = auto()

    # Plugins
    PLUGIN_LOADED = auto()


@dataclass
class EventPayload:
    """Payload delivered with each event."""

    event: Event
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    priority: int = 5


# Type alias for event handlers
Handler = Callable[[EventPayload], Awaitable[None]]


class EventBus:
    """Singleton async event bus with priority queue dispatch."""

    _instance: EventBus | None = None
    _initialized: bool = False

    def __new__(cls) -> EventBus:
        if cls._instance is None:
            obj = cast(EventBus, super().__new__(cls))
            obj._initialized = False
            cls._instance = obj
        return cls._instance

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._handlers: dict[Event, list[tuple[int, Handler]]] = defaultdict(list)
        self._wildcard: list[tuple[int, Handler]] = []
        self._queue: asyncio.PriorityQueue[tuple[int, int, EventPayload]] = (
            asyncio.PriorityQueue()
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._initialized = True

    def reset(self) -> None:
        """Reset all state — useful for tests."""
        self._ensure_init()
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._handlers = defaultdict(list)
        self._wildcard = []
        self._queue = asyncio.PriorityQueue()
        self._loop = None
        self._running = False
        self._task = None

    # conftest.py and the rest of the suite call it by this name.
    reset_for_tests = reset

    async def start(self) -> None:
        """Start the event dispatch loop (idempotent)."""
        self._ensure_init()
        if self._running and self._task is not None and not self._task.done():
            log.debug("EventBus already running")
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        # Hold the reference: a bare create_task() can be garbage-collected
        # while it is suspended, which silently kills the whole bus.
        self._task = asyncio.create_task(self._dispatch_loop(), name="event-bus")
        log.info("EventBus started")

    async def _dispatch_loop(self) -> None:
        """Main dispatch loop — runs until stopped."""
        while self._running:
            try:
                _pri, _seq, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                handlers = list(self._handlers.get(payload.event, [])) + list(
                    self._wildcard
                )
                handlers.sort(key=lambda t: t[0])

                for _priority, handler in handlers:
                    started = time.monotonic()
                    try:
                        await handler(payload)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        log.exception(
                            "Handler %s raised for %s", handler, payload.event
                        )
                    elapsed = time.monotonic() - started
                    if elapsed > SLOW_HANDLER_SECONDS:
                        # Handlers run inline, so a slow one stalls every other
                        # event behind it. Say so instead of looking hung.
                        log.warning(
                            "EventBus: handler %s blocked dispatch for %.1fs "
                            "on %s — it should spawn a task instead",
                            getattr(handler, "__qualname__", handler),
                            elapsed,
                            payload.event.name,
                        )
            finally:
                # Must always run, or stop()'s queue.join() never returns.
                self._queue.task_done()

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        self._ensure_init()
        if not self._running and self._task is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=3.0)
        except TimeoutError:
            log.warning("EventBus drain timed out")
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        log.info("EventBus stopped")

    async def emit(
        self,
        event: Event,
        data: dict[str, Any] | None = None,
        source: str = "unknown",
        priority: int = 5,
    ) -> None:
        """Emit an event asynchronously."""
        self._ensure_init()
        payload = EventPayload(
            event=event, data=data or {}, source=source, priority=priority
        )
        await self._queue.put((priority, next(counter), payload))

    def emit_sync(
        self,
        event: Event,
        data: dict[str, Any] | None = None,
        source: str = "unknown",
        priority: int = 5,
    ) -> None:
        """Emit an event synchronously from any thread."""
        self._ensure_init()
        loop = self._loop
        if loop is None or not loop.is_running():
            log.warning("emit_sync: loop not running — event %s dropped", event.name)
            return
        payload = EventPayload(
            event=event, data=data or {}, source=source, priority=priority
        )
        asyncio.run_coroutine_threadsafe(
            self._queue.put((priority, next(counter), payload)), loop
        )

    def subscribe(
        self,
        event: Event | None,
        handler: Handler,
        priority: int = 5,
    ) -> None:
        """Subscribe to an event. Use event=None for wildcard."""
        self._ensure_init()
        if event is None:
            self._wildcard.append((priority, handler))
        else:
            self._handlers[event].append((priority, handler))

    def unsubscribe(self, event: Event | None, handler: Handler) -> None:
        """Unsubscribe a handler from an event."""
        self._ensure_init()
        if event is None:
            self._wildcard = [(p, h) for p, h in self._wildcard if h is not handler]
        else:
            self._handlers[event] = [
                (p, h) for p, h in self._handlers[event] if h is not handler
            ]


# Global singleton instance
bus = EventBus()
