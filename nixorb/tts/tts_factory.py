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

    # "espeak-ng" is a settings-UI choice, not a distinct engine: PiperTTS
    # already falls back to espeak-ng automatically when the piper binary
    # or voice model isn't available, so both names route to it. Routing
    # "espeak-ng" to HuggingFaceTTS instead (the old unconditional default)
    # would raise ValueError, since that backend requires tts_hf_repo.
    if backend in ("piper", "espeak-ng", "espeak"):
        from nixorb.tts.piper_tts import PiperTTS
        return PiperTTS(settings)

    if backend == "huggingface":
        from nixorb.tts.hf_tts import HuggingFaceTTS
        return HuggingFaceTTS(settings)

    from nixorb.tts.piper_tts import PiperTTS
    return PiperTTS(settings)
