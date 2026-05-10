"""tests/conftest.py — shared fixtures for NixOrb test suite."""
from __future__ import annotations
import pytest
from nixorb.core.event_bus import EventBus

@pytest.fixture
def fresh_bus():
    EventBus._instance = None
    bus = EventBus()
    return bus

@pytest.fixture
async def started_bus(fresh_bus):
    await fresh_bus.start()
    yield fresh_bus
    await fresh_bus.stop()
    EventBus._instance = None

@pytest.fixture
def tmp_memory(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d
