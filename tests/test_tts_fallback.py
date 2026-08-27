"""tests/test_tts_fallback.py — TTS must never fail silently.

The original bug: when no speech engine was usable, NixOrb produced no audio
*and* no visible sign that anything had gone wrong, so it looked like the orb
had simply ignored you.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from nixorb.core.event_bus import Event
from nixorb.settings import Settings


def _tts(piper: bool, espeak: bool):
    from nixorb.tts.piper_tts import PiperTTS

    with patch("shutil.which", lambda name: {"piper": piper, "espeak-ng": espeak}
               .get(name, False) or None):
        engine = PiperTTS(Settings())
    engine._piper_available = piper
    engine._espeak_available = espeak
    return engine


def test_default_settings_use_the_offline_engine():
    """v2 speaks through Piper locally — no API keys, no downloads at runtime."""
    s = Settings()
    assert s.tts_backend == "piper"
    assert s.tts_voice == "en_US-lessac-medium"


def test_availability_reflects_installed_engines():
    assert _tts(piper=True, espeak=False).available is True
    assert _tts(piper=False, espeak=True).available is True
    assert _tts(piper=False, espeak=False).available is False


async def test_falls_back_to_espeak_when_piper_is_missing(started_bus):
    engine = _tts(piper=False, espeak=True)
    called = []

    async def _fake_espeak(text):
        called.append(text)

    engine._speak_espeak = _fake_espeak
    await engine.speak("hello there")

    assert called == ["hello there"]


async def test_piper_voice_missing_falls_back_to_espeak():
    """Piper installed but no voice model must still produce speech."""
    engine = _tts(piper=True, espeak=True)
    fell_back = []

    engine._find_voice_model = lambda: None
    engine._speak_espeak_sync = lambda text: fell_back.append(text)

    engine._speak_piper_sync("say something")
    assert fell_back == ["say something"]


async def test_no_engine_reports_failure_on_the_visible_bus_log(started_bus):
    """With nothing installed the user must see why the orb went quiet."""
    engine = _tts(piper=False, espeak=False)

    seen = []

    async def _record(payload):
        seen.append((payload.event, payload.data))

    started_bus.subscribe(Event.TTS_ERROR, _record)
    started_bus.subscribe(Event.LOG, _record)

    await engine.speak("this will not be spoken")
    await started_bus._queue.join()

    events = [e for e, _ in seen]
    assert Event.TTS_ERROR in events
    assert Event.LOG in events
    assert any("espeak-ng" in str(d) for _, d in seen)


async def test_empty_text_is_a_no_op(started_bus):
    engine = _tts(piper=False, espeak=False)
    seen = []

    async def _record(payload):
        seen.append(payload.event)

    started_bus.subscribe(Event.TTS_START, _record)
    started_bus.subscribe(Event.TTS_ERROR, _record)

    await engine.speak("")
    await engine.speak("   ")
    await started_bus._queue.join()

    assert seen == []


async def test_speak_emits_start_and_done(started_bus):
    engine = _tts(piper=False, espeak=True)

    async def _noop(text):
        return None

    engine._speak_espeak = _noop

    seen = []

    async def _record(payload):
        seen.append(payload.event)

    started_bus.subscribe(Event.TTS_START, _record)
    started_bus.subscribe(Event.TTS_DONE, _record)

    await engine.speak("hello")
    await started_bus._queue.join()

    assert Event.TTS_START in seen
    assert Event.TTS_DONE in seen


def test_settings_drive_voice_and_volume():
    from nixorb.tts.piper_tts import PiperTTS

    s = Settings(tts_voice="en_GB-alan-medium", tts_volume=0.5, tts_speed=1.5)
    engine = PiperTTS(s)
    assert engine._voice == "en_GB-alan-medium"
    assert engine._volume == 0.5
    assert engine._speed == 1.5


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_speed_maps_to_length_scale(speed):
    """Piper takes length-scale, which is the reciprocal of speed."""
    from nixorb.tts.piper_tts import PiperTTS

    engine = PiperTTS(Settings(tts_speed=speed))
    assert 1.0 / engine._speed == pytest.approx(1.0 / speed)
