"""tests/test_asr.py — WhisperEngine unit tests (no GPU required)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def settings():
    s = MagicMock()
    s.asr_model        = "large-v3"
    s.asr_language     = ""
    s.microphone_index = None
    return s


@pytest.fixture
async def engine(settings, started_bus):
    from nixorb.asr.whisper_engine import WhisperEngine
    return WhisperEngine(settings)


async def test_transcribe_returns_none_on_empty_audio(engine):
    """Empty audio should return None, not crash."""
    audio = np.zeros(100, dtype=np.float32)

    async def _fake_lease(_name):
        class _Ctx:
            async def __aenter__(self): return None
            async def __aexit__(self, *a): pass
        return _Ctx()

    with patch("nixorb.core.vram_manager.vram.lease", _fake_lease):
        result = await engine._transcribe_async(audio)
    assert result is None


async def test_vad_silence_detection():
    """VAD should detect silence below threshold."""
    import numpy as np

    from nixorb.asr.whisper_engine import SILENCE_DB

    # Silence: RMS should be below threshold
    silent = np.zeros(1024, dtype=np.float32)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(silent ** 2)) + 1e-10)
    assert rms_db < SILENCE_DB


async def test_vad_speech_detection():
    """Loud signal should be detected as speech."""
    import numpy as np

    from nixorb.asr.whisper_engine import SILENCE_DB

    speech = np.random.uniform(-0.8, 0.8, 1024).astype(np.float32)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(speech ** 2)) + 1e-10)
    assert rms_db > SILENCE_DB


async def test_record_and_transcribe_emits_events(engine, started_bus):
    """record_and_transcribe should emit RECORDING_START and RECORDING_STOP."""
    from nixorb.core.event_bus import Event
    received = []

    async def _handler(p):
        received.append(p.event)

    started_bus.subscribe(Event.RECORDING_START, _handler)
    started_bus.subscribe(Event.RECORDING_STOP,  _handler)

    with patch.object(engine, "_record_blocking", return_value=None):
        await engine.record_and_transcribe()

    import asyncio
    await asyncio.sleep(0.1)
    assert Event.RECORDING_START in received
    assert Event.RECORDING_STOP  in received
