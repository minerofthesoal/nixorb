"""nixorb/tts/openai_tts.py — OpenAI TTS backend."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


class OpenAITTS:
    SAMPLE_RATE = 24_000

    def __init__(self, settings: "Settings") -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._voice  = settings.tts_voice or "alloy"

    async def speak(self, text: str) -> None:
        await bus.emit(Event.TTS_START, source="OpenAITTS")
        try:
            response = await self._client.audio.speech.create(
                model="tts-1",
                voice=self._voice,
                input=text[:4_096],   # API limit
                response_format="pcm",
            )
            pcm_bytes: bytes = response.content

            # Emit PCM so orb can animate to the audio
            await bus.emit(
                Event.TTS_AUDIO_CHUNK,
                data={"pcm": pcm_bytes},
                source="OpenAITTS",
                priority=3,
            )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._play_pcm, pcm_bytes)
        except Exception:
            log.exception("OpenAI TTS failed")
        finally:
            await bus.emit(Event.TTS_DONE, source="OpenAITTS")

    def _play_pcm(self, pcm: bytes) -> None:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32_768.0
        sd.play(arr, samplerate=self.SAMPLE_RATE, blocking=True)
        sd.wait()
