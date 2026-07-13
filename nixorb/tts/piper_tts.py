"""NixOrb Piper TTS — offline text-to-speech.

Uses Piper (https://github.com/rhasspy/piper) for fast, local, neural
text-to-speech. Falls back to espeak-ng if Piper is not installed.
"""
from __future__ import annotations

import asyncio
import io
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Default Piper voice model
DEFAULT_VOICE = "en_US-lessac-medium"
PIPER_VOICES_DIR = Path.home() / ".local" / "share" / "piper" / "voices"


class PiperTTS:
    """Offline TTS using Piper with fallback to espeak-ng."""

    def __init__(self, settings: Settings | None = None) -> None:
        if settings:
            self._voice = settings.tts_voice
            self._speed = settings.tts_speed
            self._volume = settings.tts_volume
        else:
            self._voice = DEFAULT_VOICE
            self._speed = 1.0
            self._volume = 1.0

        self._piper_available = shutil.which("piper") is not None
        self._espeak_available = shutil.which("espeak-ng") is not None
        self._aplay_available = shutil.which("aplay") is not None

    def _find_voice_model(self) -> Path | None:
        """Find the Piper voice model file."""
        # Check in standard locations
        locations = [
            PIPER_VOICES_DIR / f"{self._voice}.onnx",
            Path("/usr/share/piper-voices") / f"{self._voice}.onnx",
            Path.home() / ".piper" / "voices" / f"{self._voice}.onnx",
        ]
        for loc in locations:
            if loc.exists():
                return loc
        return None

    async def speak(self, text: str) -> None:
        """Speak the given text aloud."""
        if not text or not text.strip():
            return

        text = text.strip()
        log.info("TTS: speaking '%s…'", text[:60])

        if self._piper_available:
            await self._speak_piper(text)
        elif self._espeak_available:
            await self._speak_espeak(text)
        else:
            log.error("TTS: no TTS engine available (install piper or espeak-ng)")

    def _speak_piper_sync(self, text: str) -> None:
        """Synchronous Piper TTS (runs in executor)."""
        model_path = self._find_voice_model()

        if model_path is None:
            log.warning("TTS: Piper voice model not found, falling back to espeak")
            self._speak_espeak_sync(text)
            return

        config_path = model_path.with_suffix(".onnx.json")

        try:
            # Run piper to generate WAV audio
            proc = subprocess.Popen(
                [
                    "piper",
                    "--model", str(model_path),
                    "--config", str(config_path) if config_path.exists() else "",
                    "--output_file", "-",
                    "--length-scale", str(1.0 / self._speed),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(text.encode("utf-8"), timeout=30)

            if proc.returncode != 0:
                log.error("TTS: piper failed: %s", stderr.decode())
                self._speak_espeak_sync(text)
                return

            # Play the audio
            self._play_wav_bytes(stdout)

        except subprocess.TimeoutExpired:
            log.error("TTS: piper timed out")
            proc.kill()
        except Exception as exc:
            log.error("TTS: piper error: %s", exc)
            self._speak_espeak_sync(text)

    def _speak_espeak_sync(self, text: str) -> None:
        """Synchronous espeak-ng TTS fallback."""
        try:
            # Generate WAV with espeak-ng
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            subprocess.run(
                ["espeak-ng", "-w", wav_path, "-s", "150", text],
                check=True,
                capture_output=True,
                timeout=30,
            )

            # Read and play
            import wave

            with wave.open(wav_path, "rb") as wf:
                data = wf.readframes(wf.getnframes())
                self._play_pcm(data, wf.getframerate(), wf.getnchannels())

            Path(wav_path).unlink(missing_ok=True)

        except Exception as exc:
            log.error("TTS: espeak-ng error: %s", exc)

    def _play_wav_bytes(self, wav_data: bytes) -> None:
        """Play WAV audio data using sounddevice."""
        try:
            import wave

            with io.BytesIO(wav_data) as f:
                with wave.open(f, "rb") as wf:
                    data = wf.readframes(wf.getnframes())
                    sample_rate = wf.getframerate()
                    channels = wf.getnchannels()
                    self._play_pcm(data, sample_rate, channels)
        except Exception as exc:
            log.error("TTS: WAV playback error: %s", exc)

    def _play_pcm(self, data: bytes, sample_rate: int, channels: int) -> None:
        """Play raw PCM audio data."""
        try:
            # Convert bytes to numpy array
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if channels > 1:
                audio = audio.reshape(-1, channels)

            # Apply volume
            audio = audio * self._volume

            # Play
            sd.play(audio, samplerate=sample_rate)
            sd.wait()

        except Exception as exc:
            log.error("TTS: PCM playback error: %s", exc)

    async def _speak_piper(self, text: str) -> None:
        """Async wrapper for Piper TTS."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._speak_piper_sync, text)

    async def _speak_espeak(self, text: str) -> None:
        """Async wrapper for espeak TTS."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._speak_espeak_sync, text)

    async def synthesize_to_file(self, text: str, output_path: Path) -> bool:
        """Synthesize speech to a WAV file."""
        if not self._piper_available:
            return False

        model_path = self._find_voice_model()
        if model_path is None:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", str(model_path),
                "--output_file", str(output_path),
                "--length-scale", str(1.0 / self._speed),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        except Exception as exc:
            log.error("TTS: synthesize error: %s", exc)
            return False
