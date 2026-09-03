"""faster-whisper speech-to-text.

The default backend: a single self-contained CTranslate2 model, no
transformers stack, INT8 on the GPU. Recording and VAD live in
`nixorb.asr.base`; this file is only the transcribe step.
"""
from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.asr.base import (  # noqa: F401  (re-exported for callers/tests)
    CHANNELS,
    CHUNK_DURATION,
    DTYPE,
    MAX_RECORDING_DURATION,
    MAX_SILENCE_THRESHOLD,
    MIN_SILENCE_THRESHOLD,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
    SILENCE_TIMEOUT,
    ASREngine,
    record_with_vad,
    sensitivity_to_threshold,
)

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# INT8 on the GPU (~2.1 GB for large-v3 on a GTX 1080), plain int8 on CPU.
# Trying only CUDA left CPU-only machines — and CUDA boxes without cuDNN —
# with no speech recognition at all and an opaque ctranslate2 error.
_DEVICE_ATTEMPTS = (("cuda", "int8_float16"), ("cpu", "int8"))


class WhisperEngine(ASREngine):
    """ASR engine using faster-whisper."""

    name = "faster-whisper"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model_name = settings.asr_model
        self._language = settings.asr_language or "en"

    def _load(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            from nixorb.hf import explain_import_error

            raise RuntimeError(
                explain_import_error(
                    exc, "faster_whisper", "Speech recognition"
                )
            ) from exc

        # Only try CUDA when there is a CUDA device. Attempting it blindly
        # logs a WARNING about int8_float16 on every single load of every
        # CPU-only machine, which reads as a fault and is not one.
        from nixorb.hf import resolve_device

        attempts: tuple[tuple[str, str], ...] = _DEVICE_ATTEMPTS
        if resolve_device("auto") != "cuda":
            attempts = tuple(a for a in _DEVICE_ATTEMPTS if a[0] != "cuda")

        last_exc: Exception | None = None
        for device, compute_type in attempts:
            try:
                log.info(
                    "ASR: loading Whisper %s (%s, %s)",
                    self._model_name, device, compute_type,
                )
                model = WhisperModel(
                    self._model_name,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4,
                )
                log.info("ASR: Whisper loaded on %s", device)
                return model
            except Exception as exc:
                last_exc = exc
                log.warning("ASR: could not load on %s: %s", device, exc)

        log.error("ASR: failed to load Whisper model on any device")
        raise RuntimeError(
            f"Could not load Whisper model '{self._model_name}': {last_exc}"
        ) from last_exc

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        import soundfile as sf

        wav = io.BytesIO()
        sf.write(wav, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav.seek(0)

        segments, info = self._model.transcribe(
            wav,
            language=self._language or None,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=True,
        )
        text = " ".join(segment.text for segment in segments).strip()
        log.info(
            "ASR: transcribed (lang=%s, prob=%.2f): %s",
            info.language, info.language_probability, text[:100],
        )
        return text
