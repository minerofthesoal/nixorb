"""Generic Hugging Face speech-to-text — any ASR model on the Hub.

Uses the `automatic-speech-recognition` pipeline, which is the one interface
every ASR architecture on the Hub agrees on: Whisper, Wav2Vec2, HuBERT, MMS,
SeamlessM4T, Parakeet, Moonshine and anything else that declares the task.

    asr_backend = "huggingface"
    asr_model   = "openai/whisper-small"        # or distil-whisper, MMS, …
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb import hf
from nixorb.asr.base import SAMPLE_RATE, ASREngine

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Anything longer than the model's own window has to be chunked, or the
# pipeline silently truncates to the first 30 seconds.
DEFAULT_CHUNK_SECONDS = 30.0

# Models whose generate() accepts a `language` argument. For everything else
# passing one is an error, not a no-op.
_MULTILINGUAL_HINTS = ("whisper", "seamless", "mms", "canary")


class HFASREngine(ASREngine):
    """Speech-to-text through the HF `automatic-speech-recognition` pipeline."""

    name = "huggingface"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model_id = settings.asr_model
        self._language = (settings.asr_language or "").strip()

    def _load(self) -> Any:
        transformers = hf.require("transformers")
        device = hf.resolve_device(getattr(self._settings, "hf_device", "auto"))

        kwargs = hf.load_kwargs(self._settings, device)
        dtype = hf.torch_dtype(device)
        if dtype is not None:
            kwargs["dtype"] = dtype

        log.info("ASR: loading HF model %s on %s", self._model_id, device)
        pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=self._model_id,
            device=0 if device == "cuda" else -1,
            **kwargs,
        )
        log.info("ASR: %s ready", self._model_id)
        return pipe

    def _wants_language(self) -> bool:
        """Only pass `language=` to models that actually accept it."""
        if not self._language:
            return False
        return any(hint in self._model_id.lower() for hint in _MULTILINGUAL_HINTS)

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("ASR model not loaded")

        payload = {"raw": np.asarray(audio, dtype=np.float32),
                   "sampling_rate": SAMPLE_RATE}
        kwargs: dict[str, Any] = {
            "chunk_length_s": DEFAULT_CHUNK_SECONDS,
        }
        if self._wants_language():
            kwargs["generate_kwargs"] = {"language": self._language}

        try:
            result = self._model(payload, **kwargs)
        except (TypeError, ValueError) as exc:
            # Plenty of CTC models reject chunk_length_s or generate_kwargs.
            # Retry bare rather than losing the utterance over an argument.
            log.debug("ASR: retrying %s without extras (%s)", self._model_id, exc)
            result = self._model(payload)

        return _extract_text(result)


def _extract_text(result: Any) -> str:
    """Pull the transcript out of whatever shape the pipeline returned."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", "")).strip()
    if isinstance(result, list):
        return " ".join(_extract_text(item) for item in result).strip()
    return str(result).strip()
