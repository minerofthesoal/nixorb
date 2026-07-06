"""tests/conftest.py — shared fixtures for NixOrb test suite."""
from __future__ import annotations

import pytest

from nixorb.core.event_bus import bus as _bus


@pytest.fixture
def fresh_bus():
    """
    The real EventBus singleton, reset in place.

    NOTE: this intentionally does *not* do `EventBus._instance = None;
    EventBus()` — that would create a second, disconnected instance.
    Every module that already did `from nixorb.core.event_bus import bus`
    at import time (executor.py, backends.py, wake_word.py, ...) holds a
    direct reference to the *original* object; swapping in a new instance
    silently orphans all of them, so events emitted through those modules
    would never reach handlers registered on the "fresh" bus, or vice
    versa. Resetting the same object's internal state keeps everyone
    talking to the same bus.
    """
    _bus.reset_for_tests()
    return _bus

@pytest.fixture
async def started_bus(fresh_bus):
    await fresh_bus.start()
    yield fresh_bus
    await fresh_bus.stop()
    fresh_bus.reset_for_tests()

@pytest.fixture
def tmp_memory(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d
