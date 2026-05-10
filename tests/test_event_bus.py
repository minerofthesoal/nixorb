"""tests/test_event_bus.py — EventBus unit tests."""
from __future__ import annotations
import asyncio
import pytest
from nixorb.core.event_bus import Event, EventPayload, EventBus

pytestmark = pytest.mark.asyncio


async def test_basic_emit_and_receive(started_bus):
    received = []
    async def handler(p: EventPayload):
        received.append(p)
    started_bus.subscribe(Event.LOG, handler)
    await started_bus.emit(Event.LOG, data={"msg": "hello"}, source="test")
    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].data["msg"] == "hello"


async def test_wildcard_subscription(started_bus):
    seen = []
    async def handler(p):
        seen.append(p.event)
    started_bus.subscribe(None, handler)  # wildcard
    await started_bus.emit(Event.ORB_IDLE, source="test")
    await started_bus.emit(Event.ORB_SPEAKING, source="test")
    await asyncio.sleep(0.05)
    assert Event.ORB_IDLE in seen
    assert Event.ORB_SPEAKING in seen


async def test_priority_ordering(started_bus):
    order = []
    async def h_high(p): order.append("high")
    async def h_low(p):  order.append("low")
    started_bus.subscribe(Event.LOG, h_high, priority=1)
    started_bus.subscribe(Event.LOG, h_low,  priority=9)
    await started_bus.emit(Event.LOG, source="test")
    await asyncio.sleep(0.05)
    assert order == ["high", "low"], f"Got: {order}"


async def test_unsubscribe(started_bus):
    calls = []
    async def handler(p): calls.append(1)
    started_bus.subscribe(Event.LOG, handler)
    started_bus.unsubscribe(Event.LOG, handler)
    await started_bus.emit(Event.LOG, source="test")
    await asyncio.sleep(0.05)
    assert calls == []


async def test_emit_before_start_does_not_crash():
    """emit_sync on a bus with no loop should log a warning, not raise."""
    EventBus._instance = None
    bus = EventBus()
    # Should not raise; bus._loop is None
    bus.emit_sync(Event.LOG, data={"msg": "before start"}, source="test")
    EventBus._instance = None


async def test_priority_queue_tiebreaker(started_bus):
    """Emit many events at same priority — no TypeError from payload comparison."""
    for i in range(20):
        await started_bus.emit(Event.LOG, data={"i": i}, priority=5, source="test")
    await asyncio.sleep(0.1)  # all dispatched without crash
