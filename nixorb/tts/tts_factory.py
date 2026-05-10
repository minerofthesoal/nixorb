"""nixorb/tts/tts_factory.py — Build the correct TTS backend from settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nixorb.settings import Settings


def build_tts(settings: Settings):
    backend = settings.tts_backend.lower()
    if backend == "openai":
        from nixorb.tts.openai_tts import OpenAITTS
        return OpenAITTS(settings)
    elif backend == "huggingface":
        from nixorb.tts.hf_tts import HuggingFaceTTS
        return HuggingFaceTTS(settings)
    elif backend == "piper":
        from nixorb.tts.piper_tts import PiperTTS
        return PiperTTS(settings)
    else:
        raise ValueError(f"Unknown TTS backend: {backend!r}")
