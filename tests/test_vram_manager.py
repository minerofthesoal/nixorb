"""tests/test_vram_manager.py — VRAMManager unit tests (CPU-only, no CUDA required)."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import patch, MagicMock

from nixorb.core.vram_manager import VRAMManager, ModelPriority

pytestmark = pytest.mark.asyncio


def _make_manager() -> VRAMManager:
    return VRAMManager()


@pytest.fixture
async def vram():
    m = _make_manager()
    # Patch nvidia-smi so tests run without a GPU
    with patch.object(m, "_query_free_vram", return_value=6_000):
        yield m


async def test_register_and_lease(vram):
    loaded = []
    unloaded = []

    def load_fn():
        loaded.append(True)
        return object()

    def unload_fn(obj):
        unloaded.append(True)

    vram.register("test_model", vram_mb=512, priority=ModelPriority.HIGH,
                  load_fn=load_fn, unload_fn=unload_fn)

    async with vram.lease("test_model") as obj:
        assert obj is not None
        assert vram.is_loaded("test_model")

    assert len(loaded) == 1


async def test_eviction_by_priority(vram):
    """LOW-priority model should be evicted before HIGH when space is needed."""
    evicted = []

    def make_unload(name):
        def fn(obj): evicted.append(name)
        return fn

    vram.register("low_model",  vram_mb=2_000, priority=ModelPriority.LOW,
                  load_fn=lambda: "low",  unload_fn=make_unload("low_model"))
    vram.register("high_model", vram_mb=2_000, priority=ModelPriority.HIGH,
                  load_fn=lambda: "high", unload_fn=make_unload("high_model"))

    # Manually mark low_model as loaded
    vram._models["low_model"].obj    = "low"
    vram._models["low_model"].loaded = True

    # Force free_vram to be low so eviction triggers
    with patch.object(vram, "_query_free_vram", side_effect=[500, 500, 3_000]):
        await vram._ensure_space(2_000, exclude="high_model")

    assert "low_model" in evicted


async def test_evict_explicit(vram):
    unloaded = []
    vram.register("m", vram_mb=100, priority=ModelPriority.MEDIUM,
                  load_fn=lambda: "obj", unload_fn=lambda o: unloaded.append(o))
    vram._models["m"].obj    = "obj"
    vram._models["m"].loaded = True
    await vram.evict("m")
    assert not vram.is_loaded("m")
    assert "obj" in unloaded


async def test_lease_unknown_model_raises(vram):
    with pytest.raises(KeyError):
        async with vram.lease("does_not_exist"):
            pass


async def test_double_evict_is_safe(vram):
    vram.register("x", vram_mb=100, priority=ModelPriority.LOW,
                  load_fn=lambda: "obj", unload_fn=lambda o: None)
    vram._models["x"].obj    = "obj"
    vram._models["x"].loaded = True
    await vram.evict("x")
    await vram.evict("x")  # second call should be a no-op
