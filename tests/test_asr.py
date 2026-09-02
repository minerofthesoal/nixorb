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
        engine._transcribe(np.zeros(100, dtype=np.float32))


async def test_record_and_transcribe_emits_events(engine, started_bus):
    """record_and_transcribe emits RECORDING_START even when nothing is heard."""
    from nixorb.core.event_bus import Event

    received = []

    async def _handler(p):
        received.append(p.event)

    started_bus.subscribe(Event.RECORDING_START, _handler)
    started_bus.subscribe(Event.RECORDING_STOP, _handler)

    engine._model = object()  # skip preload
    with patch("nixorb.asr.base.record_with_vad", return_value=None):
        result = await engine.record_and_transcribe()

    await asyncio.sleep(0.1)
    assert result is None
    assert Event.RECORDING_START in received


async def test_short_audio_is_discarded(engine, started_bus):
    """Sub-300ms captures are noise, not speech."""
    engine._model = object()
    tiny = np.zeros(100, dtype=np.float32)
    with patch("nixorb.asr.base.record_with_vad", return_value=tiny):
        assert await engine.record_and_transcribe() is None


async def test_missing_faster_whisper_is_a_clear_error(engine, monkeypatch):
    """A missing optional backend must say so, not raise ImportError deep down."""
    import builtins
    import importlib.util

    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def _no_faster_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ModuleNotFoundError(
                "No module named 'faster_whisper'", name="faster_whisper"
            )
        return real_import(name, *args, **kwargs)

    # Absent means absent: gone from disk, not merely unimportable. The two
    # cases get different messages on purpose.
    def _gone(name, *args, **kwargs):
        if name.split(".")[0] == "faster_whisper":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _gone)
    with patch.object(builtins, "__import__", _no_faster_whisper):
        with pytest.raises(RuntimeError, match="is not installed") as caught:
            engine._load()

    message = str(caught.value)
    assert "faster_whisper" in message
    assert "pip install faster-whisper" in message


async def test_an_unimportable_faster_whisper_is_not_called_missing(
    engine, monkeypatch
):
    """It is on disk, so "install it" is the one thing that will not help."""
    import builtins
    import importlib.util

    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec

    def _broken(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError(
                "cannot import name 'WhisperModel'", name="faster_whisper"
            )
        return real_import(name, *args, **kwargs)

    # Say it is present regardless of this environment: the case under test
    # is "on disk but unimportable", and whether the runner happens to have
    # faster-whisper installed must not decide which message we assert.
    def _present(name, *args, **kwargs):
        if name.split(".")[0] == "faster_whisper":
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", _present)
    with patch.object(builtins, "__import__", _broken):
        with pytest.raises(RuntimeError) as caught:
            engine._load()

    message = str(caught.value)
    assert "though it is installed" in message
    assert "pip install" not in message


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
        model = engine._load()

    assert attempts == ["cuda", "cpu"]
    assert isinstance(model, _FakeModel)
