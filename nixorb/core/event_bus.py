"""NixOrb event bus — central async event system.

All communication between modules happens through typed events on this bus.
This eliminates direct coupling between UI, ASR, LLM, TTS, and other components.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast

log = logging.getLogger(__name__)

# Global tiebreaker counter — prevents PriorityQueue comparison errors
counter = itertools.count()


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
        self._initialized = True

    def reset(self) -> None:
        """Reset all state — useful for tests."""
        self._handlers = defaultdict(list)
        self._wildcard = []
        self._queue = asyncio.PriorityQueue()
        self._loop = None
        self._running = False

    async def start(self) -> None:
        """Start the event dispatch loop."""
        self._ensure_init()
        self._loop = asyncio.get_running_loop()
        self._running = True
        asyncio.create_task(self._dispatch_loop(), name="event-bus")
        log.info("EventBus started")

    async def _dispatch_loop(self) -> None:
        """Main dispatch loop — runs forever until stopped."""
        while self._running:
            try:
                _pri, _seq, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=0.5
                )
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            handlers = list(self._handlers.get(payload.event, [])) + list(
                self._wildcard
            )
            handlers.sort(key=lambda t: t[0])

            for _priority, handler in handlers:
                try:
                    await handler(payload)
                except Exception:
                    log.exception("Handler %s raised for %s", handler, payload.event)

            self._queue.task_done()

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        self._running = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=3.0)
        except TimeoutError:
            log.warning("EventBus drain timed out")
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
