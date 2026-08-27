"""tests/test_asr.py — WhisperEngine unit tests (no GPU or mic required)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def settings():
    s = MagicMock()
    s.asr_model = "large-v3"
    s.asr_language = ""
    s.microphone_index = None
    return s


@pytest.fixture
def engine(settings, started_bus):
    from nixorb.asr.whisper_engine import WhisperEngine
    return WhisperEngine(settings)


def test_language_defaults_to_english(engine):
    assert engine._language == "en"


def test_vad_silence_detection():
    """Silence must sit below the VAD threshold."""
    from nixorb.asr.whisper_engine import SILENCE_THRESHOLD

    silent = np.zeros(1024, dtype=np.float32)
    assert np.sqrt(np.mean(silent**2)) < SILENCE_THRESHOLD


def test_vad_speech_detection():
    """A loud signal must sit above the VAD threshold."""
    from nixorb.asr.whisper_engine import SILENCE_THRESHOLD

    speech = np.random.uniform(-0.8, 0.8, 1024).astype(np.float32)
    assert np.sqrt(np.mean(speech**2)) > SILENCE_THRESHOLD


async def test_transcribe_raises_without_a_model(engine):
    with pytest.raises(RuntimeError, match="not loaded"):
        engine._transcribe_sync(np.zeros(100, dtype=np.float32))


async def test_record_and_transcribe_emits_events(engine, started_bus):
    """record_and_transcribe emits RECORDING_START even when nothing is heard."""
    from nixorb.core.event_bus import Event

    received = []

    async def _handler(p):
        received.append(p.event)

    started_bus.subscribe(Event.RECORDING_START, _handler)
    started_bus.subscribe(Event.RECORDING_STOP, _handler)

    engine._model = object()  # skip preload
    with patch.object(engine, "_record_audio_sync", return_value=None):
        result = await engine.record_and_transcribe()

    await asyncio.sleep(0.1)
    assert result is None
    assert Event.RECORDING_START in received


async def test_short_audio_is_discarded(engine, started_bus):
    """Sub-300ms captures are noise, not speech."""
    engine._model = object()
    tiny = np.zeros(100, dtype=np.float32)
    with patch.object(engine, "_record_audio_sync", return_value=tiny):
        assert await engine.record_and_transcribe() is None


async def test_missing_faster_whisper_is_a_clear_error(engine):
    """A missing optional backend must say so, not raise ImportError deep down."""
    import builtins

    real_import = builtins.__import__

    def _no_faster_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", _no_faster_whisper):
        with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
            engine._load_model()


async def test_load_model_falls_back_to_cpu(engine):
    """A CUDA failure must fall back to CPU rather than killing ASR."""
    attempts = []

    class _FakeModel:
        def __init__(self, name, device=None, compute_type=None, **kw):
            attempts.append(device)
            if device == "cuda":
                raise RuntimeError("no CUDA driver")

    fake_module = MagicMock()
    fake_module.WhisperModel = _FakeModel

    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        model = engine._load_model()

    assert attempts == ["cuda", "cpu"]
    assert isinstance(model, _FakeModel)
