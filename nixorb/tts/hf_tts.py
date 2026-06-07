"""nixorb/tts/hf_tts.py — HuggingFace TTS backend (e.g. SpeechT5, Parler)."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import ModelPriority, vram

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


def _load_hf_tts(repo_id: str, token: str | None) -> Any:
    from transformers import pipeline
    task: Any = "text-to-speech"
    return pipeline(
        task,
        model=repo_id,
        token=token or None,
        device=0 if torch.cuda.is_available() else -1,
    )


def _unload_hf_tts(pipe: Any) -> None:
    del pipe


class HuggingFaceTTS:
    def __init__(self, settings: Settings) -> None:
        self._repo_id = settings.tts_hf_repo
        self._token   = settings.hf_token or None
        self._voice   = settings.tts_voice

        if not self._repo_id:
            raise ValueError("tts_hf_repo must be set when using HuggingFace TTS")

        vram.register(
            name="hf_tts",
            vram_mb=1_500,
            priority=ModelPriority.MEDIUM,
            load_fn=lambda: _load_hf_tts(self._repo_id, self._token),
            unload_fn=_unload_hf_tts,
        )

    async def speak(self, text: str) -> None:
        await bus.emit(Event.TTS_START, source="HuggingFaceTTS")
        loop = asyncio.get_running_loop()
        try:
            async with vram.lease("hf_tts") as pipe:
                output = await loop.run_in_executor(None, cast(Any, pipe), text)
            audio = output["audio"]           # numpy array
            sr    = output["sampling_rate"]
            pcm   = (audio * 32_767).astype(np.int16).tobytes()
            await bus.emit(
                Event.TTS_AUDIO_CHUNK, data={"pcm": pcm}, source="HuggingFaceTTS"
            )
            await loop.run_in_executor(
                None, lambda: (sd.play(audio, samplerate=sr, blocking=True), sd.wait())
            )
        except Exception:
            log.exception("HuggingFace TTS failed")
        finally:
            await bus.emit(Event.TTS_DONE, source="HuggingFaceTTS")
