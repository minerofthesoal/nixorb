"""nixorb/tts/hf_tts.py — HuggingFace TTS backend (e.g. SpeechT5, Parler)."""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import ModelPriority, vram

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


def _hf_tts_device() -> int:
    if importlib.util.find_spec("torch") is None:
        return -1
    import torch
    return 0 if torch.cuda.is_available() else -1


_FALLBACK_REPO = "microsoft/speecht5_tts"


def _load_hf_tts(repo_id: str, token: str | None) -> tuple[str, Any]:
    from transformers import pipeline
    task: Any = "text-to-speech"
    try:
        pipe = pipeline(task, model=repo_id, token=token or None, device=_hf_tts_device())
        return ("pipeline", pipe)
    except Exception as exc:
        # Not every HF repo id is actually a TTS model (e.g. a plain causal-LM
        # checkpoint used elsewhere in the config) — fall back to a model we
        # know transformers' text-to-speech pipeline can load, instead of
        # failing silently forever.
        log.warning(
            "HF TTS: '%s' is not a usable text-to-speech pipeline (%s); "
            "falling back to %s", repo_id, exc, _FALLBACK_REPO,
        )
        from nixorb.tts.glados_tts import _load_speecht5
        return _load_speecht5(token)


def _unload_hf_tts(obj: tuple[str, Any]) -> None:
    del obj


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
            async with vram.lease("hf_tts") as obj:
                kind, pipe = obj
                if kind == "pipeline":
                    output = await loop.run_in_executor(None, cast(Any, pipe), text)
                    audio  = output["audio"]
                    sr     = output["sampling_rate"]
                else:
                    from nixorb.tts.glados_tts import _synthesise_speecht5
                    pcm_bytes = await loop.run_in_executor(
                        None, _synthesise_speecht5, pipe, text
                    )
                    if pcm_bytes is None:
                        return
                    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    sr    = 16_000
            pcm = (audio * 32_767).astype(np.int16).tobytes()
            await bus.emit(
                Event.TTS_AUDIO_CHUNK, data={"pcm": pcm}, source="HuggingFaceTTS"
            )
            await loop.run_in_executor(
                None, lambda: (sd.play(audio, samplerate=sr, blocking=True), sd.wait())
            )
        except Exception as exc:
            log.exception("HuggingFace TTS failed")
            await bus.emit(
                Event.LOG,
                data={"level": "error", "msg": f"❌ TTS failed: {exc}"},
                source="HuggingFaceTTS",
            )
        finally:
            await bus.emit(Event.TTS_DONE, source="HuggingFaceTTS")
