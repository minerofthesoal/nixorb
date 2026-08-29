"""nixorb/asr/factory.py — pick an ASR engine from settings.

``settings.asr_backend`` selects the engine; both implementations share the
same public interface (``is_loaded``, ``preload``, ``unload``,
``stop_recording``, ``record_and_transcribe``), so callers never need to
know which one they got.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nixorb.asr.hf_asr_engine import HFASREngine
    from nixorb.asr.whisper_engine import WhisperEngine
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

AnyASREngine = Union["WhisperEngine", "HFASREngine"]


def build_asr(settings: Settings) -> AnyASREngine:
    backend = (getattr(settings, "asr_backend", "") or "faster_whisper").lower()

    if backend == "huggingface":
        from nixorb.asr.hf_asr_engine import HFASREngine
        log.info("ASR: using HuggingFace backend, model '%s'", settings.asr_model)
        return HFASREngine(settings)

    if backend != "faster_whisper":
        log.warning(
            "ASR: unknown asr_backend '%s' — falling back to faster_whisper",
            backend,
        )

    from nixorb.asr.whisper_engine import WhisperEngine
    log.info("ASR: using faster-whisper backend, model '%s'", settings.asr_model)
    return WhisperEngine(settings)
