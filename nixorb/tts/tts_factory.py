"""nixorb/tts/tts_factory.py — Build the correct TTS backend from settings."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nixorb.settings import Settings


def build_tts(settings: Settings):
    """
    Factory: return the appropriate TTS backend based on settings.

    Backends:
      huggingface  — HuggingFaceTTS (SpeechT5, Parler, etc.)
      glados       — GladosTTS (torphix/stablelm-2-glados-v1 voice)
      openai       — OpenAITTS (alloy, nova, echo, shimmer, fable, onyx)
      piper        — PiperTTS (fully offline, Piper binary)
    """
    backend = settings.tts_backend.lower()

    if backend == "glados":
        from nixorb.tts.glados_tts import GladosTTS
        return GladosTTS(settings)

    if backend == "openai":
        from nixorb.tts.openai_tts import OpenAITTS
        return OpenAITTS(settings)

    if backend == "piper":
        from nixorb.tts.piper_tts import PiperTTS
        return PiperTTS(settings)

    # default: huggingface
    from nixorb.tts.hf_tts import HuggingFaceTTS
    return HuggingFaceTTS(settings)
