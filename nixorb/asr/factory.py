"""Choose a speech-to-text backend from settings.

    asr_backend = "faster-whisper"   # default, self-contained
    asr_backend = "huggingface"      # any ASR model on the Hub
    asr_backend = "nemotron"         # NVIDIA Nemotron 3.5, native streaming
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nixorb.asr.base import ASREngine

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Aliases people will reasonably type, mapped to the canonical name.
_ALIASES = {
    "whisper": "faster-whisper",
    "faster_whisper": "faster-whisper",
    "fasterwhisper": "faster-whisper",
    "hf": "huggingface",
    "transformers": "huggingface",
    "nvidia": "nemotron",
    "nemotron-3.5": "nemotron",
}

BACKENDS = ("faster-whisper", "huggingface", "nemotron")


def normalise_backend(name: str | None) -> str:
    key = (name or "faster-whisper").strip().lower()
    return _ALIASES.get(key, key)


def create_asr(settings: Settings) -> ASREngine:
    """Build the configured ASR engine.

    A model id that is obviously a Nemotron checkpoint selects the native
    backend even if the user left asr_backend alone — otherwise pointing
    asr_model at it would silently lose streaming.
    """
    backend = normalise_backend(getattr(settings, "asr_backend", None))
    model_id = (getattr(settings, "asr_model", "") or "").lower()

    if backend == "huggingface" and "nemotron" in model_id and "asr" in model_id:
        log.info("ASR: %s is a Nemotron checkpoint — using the native backend",
                 settings.asr_model)
        backend = "nemotron"

    if backend == "nemotron":
        from nixorb.asr.nemotron_asr import NemotronASREngine

        return NemotronASREngine(settings)

    if backend == "huggingface":
        from nixorb.asr.hf_asr import HFASREngine

        return HFASREngine(settings)

    if backend != "faster-whisper":
        log.warning(
            "ASR: unknown backend %r (choose one of %s) — using faster-whisper",
            settings.asr_backend, ", ".join(BACKENDS),
        )

    from nixorb.asr.whisper_engine import WhisperEngine

    return WhisperEngine(settings)
