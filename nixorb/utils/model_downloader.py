"""
nixorb/utils/model_downloader.py — Download NixOrb's local models.

Downloads:
  • faster-whisper (ASR)             — configurable size, default large-v3 INT8
  • openwakeword wake-word models    — required for wake-word detection

HuggingFace models (LLM, TTS, Vision) are auto-downloaded by `transformers`
the first time they're used, so they aren't handled here.

This is the single source of truth used by both:
  - `nixorb download-models` (CLI)
  - `scripts/download_models.sh` (works even without the package installed)
"""
from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


def download_whisper(model_name: str = "large-v3", on_progress: ProgressFn = _noop) -> None:
    """Download/cache the faster-whisper model."""
    on_progress(f"Downloading faster-whisper {model_name} (this may take a few minutes)…")
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    del model
    on_progress(f"faster-whisper {model_name} cached.")


def download_wake_word(model_name: str | None = None, on_progress: ProgressFn = _noop) -> None:
    """Download OpenWakeWord model weights.

    Args:
        model_name: A specific pretrained model to fetch (e.g. "hey_jarvis_v0.1").
                     If None, all bundled openWakeWord models are downloaded.
    """
    on_progress("Downloading OpenWakeWord models…")
    import openwakeword.utils as oww_utils

    oww_utils.download_models(model_names=[model_name] if model_name else [])
    on_progress("OpenWakeWord models downloaded.")


def download_all(
    settings=None,
    whisper: bool = True,
    wake_word: bool = True,
    on_progress: ProgressFn = _noop,
) -> list[str]:
    """Download everything NixOrb needs locally. Returns a list of error
    messages for anything that failed (empty list = fully successful)."""
    errors: list[str] = []

    if whisper:
        try:
            model_name = settings.asr_model if settings else "large-v3"
            download_whisper(model_name, on_progress)
        except Exception as exc:
            log.exception("Whisper model download failed")
            errors.append(f"faster-whisper download failed: {exc}")
            on_progress(f"⚠ faster-whisper download failed: {exc}")

    if wake_word:
        try:
            model_name = settings.wake_word_model if settings else None
            download_wake_word(model_name, on_progress)
        except Exception as exc:
            log.exception("OpenWakeWord model download failed")
            errors.append(f"OpenWakeWord download failed: {exc}")
            on_progress(f"⚠ OpenWakeWord download failed: {exc}")

    return errors
