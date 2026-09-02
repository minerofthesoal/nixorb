"""Build the TTS backend named by settings.

    tts_backend = "piper"        # offline, AUR piper-tts, espeak fallback
    tts_backend = "huggingface"  # any TTS model on the Hub
    tts_backend = "glados"       # the GLaDOS voice, via SpeechT5
    tts_backend = "openai"       # any OpenAI-compatible speech endpoint
    tts_backend = "espeak"       # force the always-available fallback
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_ALIASES = {"hf": "huggingface", "transformers": "huggingface",
            "espeak-ng": "espeak", "piper-tts": "piper"}

BACKENDS = ("piper", "huggingface", "glados", "openai", "espeak")


def normalise_backend(name: str | None) -> str:
    key = (name or "piper").strip().lower()
    return _ALIASES.get(key, key)


def build_tts(settings: Settings) -> Any:
    """Build the configured TTS engine.

    Falls back to Piper (which itself falls back to espeak-ng) whenever the
    chosen backend cannot run, so the orb is never silently mute.
    """
    backend = normalise_backend(getattr(settings, "tts_backend", None))

    if backend == "huggingface":
        from nixorb.tts.hf_tts import HuggingFaceTTS

        hf_engine = HuggingFaceTTS(settings)
        if hf_engine.available:
            return hf_engine
        log.warning(
            "TTS: huggingface backend selected but transformers is not "
            "installed — falling back to piper. Install with: "
            "pip install 'nixorb[hf]'"
        )
        backend = "piper"

    if backend == "glados":
        try:
            from nixorb.tts.glados_tts import GladosTTS

            return GladosTTS(settings)
        except Exception as exc:
            log.warning("TTS: glados backend unavailable (%s) — using piper", exc)
            backend = "piper"

    if backend == "openai":
        try:
            from nixorb.tts.openai_tts import OpenAITTS

            return OpenAITTS(settings)
        except Exception as exc:
            log.warning("TTS: openai backend unavailable (%s) — using piper", exc)
            backend = "piper"

    if backend not in ("piper", "espeak"):
        log.warning(
            "TTS: unknown tts_backend '%s' (choose one of %s) — using piper",
            getattr(settings, "tts_backend", None), ", ".join(BACKENDS),
        )

    from nixorb.tts.piper_tts import PiperTTS

    piper = PiperTTS(settings)
    if backend == "espeak":
        # Skip Piper even if installed; the user asked for espeak.
        piper._piper_available = False
    return piper


create_tts = build_tts
