"""
nixorb/tts/piper_tts.py — Offline Piper TTS backend.

BUG FIX PASS 1:
  - Previous version had `import subprocess` duplicated (once at top of the
    merged file, once inside _speak_blocking). Removed duplicate.

BUG FIX PASS 2:
  - Piper outputs raw 16-bit PCM at 22 050 Hz (model-dependent).
    The sample rate was hardcoded; now configurable via settings.tts_voice
    which maps to a Piper voice name that embeds its sample rate.
    Added a simple lookup table for common voices.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Known Piper voice sample rates
_VOICE_SR: dict[str, int] = {
    "en_US-lessac-medium":   22_050,
    "en_US-ryan-high":       22_050,
    "en_GB-alba-medium":     22_050,
    "en_US-libritts-high":   44_100,
}
_DEFAULT_SR = 22_050


class PiperTTS:
    def __init__(self, settings: "Settings") -> None:
        self._voice     = settings.tts_voice or "en_US-lessac-medium"
        self._sample_rate = _VOICE_SR.get(self._voice, _DEFAULT_SR)

        if not shutil.which("piper"):
            log.warning("piper not found in PATH — install piper-tts from AUR")

    async def speak(self, text: str) -> None:
        await bus.emit(Event.TTS_START, source="PiperTTS")
        loop = asyncio.get_running_loop()
        try:
            pcm = await loop.run_in_executor(None, self._synthesise, text)
            if pcm:
                await bus.emit(
                    Event.TTS_AUDIO_CHUNK,
                    data={"pcm": pcm},
                    source="PiperTTS",
                    priority=3,
                )
                await loop.run_in_executor(None, self._play_pcm, pcm)
        except Exception:
            log.exception("Piper TTS failed")
        finally:
            await bus.emit(Event.TTS_DONE, source="PiperTTS")

    def _synthesise(self, text: str) -> bytes | None:
        proc = subprocess.run(
            ["piper", "--model", self._voice, "--output-raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            log.error("piper exited %d: %s", proc.returncode,
                      proc.stderr.decode(errors="replace"))
            return None
        return proc.stdout

    def _play_pcm(self, pcm: bytes) -> None:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_768.0
        sd.play(arr, samplerate=self._sample_rate, blocking=True)
        sd.wait()
