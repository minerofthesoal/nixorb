"""Choose a text-to-speech backend from settings.

    tts_backend = "piper"        # default — offline, fast, AUR piper-tts
    tts_backend = "huggingface"  # any TTS model on the Hub
    tts_backend = "espeak"       # the always-available fallback
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_ALIASES = {"hf": "huggingface", "transformers": "huggingface",
            "espeak-ng": "espeak", "piper-tts": "piper"}

BACKENDS = ("piper", "huggingface", "espeak")


def normalise_backend(name: str | None) -> str:
    key = (name or "piper").strip().lower()
    return _ALIASES.get(key, key)


def create_tts(settings: Settings) -> Any:
    """Build the configured TTS engine.

    Falls back to Piper (which itself falls back to espeak-ng) when the
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

    if backend not in ("piper", "espeak"):
        log.warning(
            "TTS: unknown backend %r (choose one of %s) — using piper",
            settings.tts_backend, ", ".join(BACKENDS),
        )

    from nixorb.tts.piper_tts import PiperTTS

    piper = PiperTTS(settings)
    if backend == "espeak":
        # Skip Piper even if it is installed; the user asked for espeak.
        piper._piper_available = False
    return piper
