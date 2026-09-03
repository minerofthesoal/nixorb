"""Shared ASR plumbing — microphone capture, VAD, and the engine interface.

Every ASR backend needs the same "record until they stop talking, then
transcribe" loop; only the transcribe step differs. That loop lives here so
faster-whisper, a generic Hugging Face model and Nemotron all behave the
same way from the orb's point of view.
"""
from __future__ import annotations

import asyncio
import copy
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Audio capture parameters. 16 kHz mono is what every speech model expects.
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
CHUNK_DURATION = 0.5  # seconds per captured block
# RMS below this counts as silence, at the default mic_sensitivity of 0.5.
SILENCE_THRESHOLD = 0.015
# Ends of the sensitivity dial: 1.0 triggers on quiet/distant speech, 0.0
# needs a loud, close voice.
MIN_SILENCE_THRESHOLD = 0.004
MAX_SILENCE_THRESHOLD = 0.045
SILENCE_TIMEOUT = 2.0  # seconds of silence after speech before stopping
MAX_RECORDING_DURATION = 30.0
# Spend the first fraction of a second measuring the room rather than
# listening: a noisy room otherwise trips the VAD on its own hum and records
# thirty seconds of nothing.
CALIBRATION_SECONDS = 0.3
MIN_SPEECH_SECONDS = 0.3  # anything shorter is a click, not a sentence


def sensitivity_to_threshold(sensitivity: object) -> float:
    """Map a 0..1 mic_sensitivity onto a VAD RMS threshold.

    Higher sensitivity must *lower* the threshold. Defensive about the value
    not being a plain float (an unconfigured test mock, say) — falls back to
    the default rather than raising out of a constructor.
    """
    try:
        s = max(0.0, min(1.0, float(sensitivity)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return SILENCE_THRESHOLD
    return MAX_SILENCE_THRESHOLD + s * (MIN_SILENCE_THRESHOLD - MAX_SILENCE_THRESHOLD)


def record_with_vad(
    mic_index: int | None = None,
    threshold: float = SILENCE_THRESHOLD,
    max_seconds: float = MAX_RECORDING_DURATION,
    silence_seconds: float = SILENCE_TIMEOUT,
    should_continue: Any = None,
    mic_name: str = "",
) -> np.ndarray | None:
    """Record from the microphone until silence. Blocking — run in a thread.

    Returns the captured audio, or None if nothing was said.
    """
    import sounddevice as sd

    from nixorb.utils.audio import resolve_input_device

    # Never just take sd.default.device: on PipeWire that is routinely a
    # monitor/loopback source, which records the speakers instead of a mic
    # and yields a confident transcript of the assistant's own voice.
    resolved = resolve_input_device(mic_index, mic_name)
    log.info(
        "ASR: recording from device[%s] '%s' (%s), threshold=%.4f",
        resolved.index, resolved.name, resolved.reason, threshold,
    )

    block = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_blocks = int(SAMPLE_RATE * silence_seconds)
    max_samples = int(SAMPLE_RATE * max_seconds)

    buffer: list[np.ndarray] = []
    silence_run = 0
    total = 0
    heard_speech = False

    calib_needed = int(SAMPLE_RATE * CALIBRATION_SECONDS)
    calib_seen = 0
    calib_rms: list[float] = []

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=resolved.index,
            blocksize=block,
        ) as stream:
            while total < max_samples:
                if should_continue is not None and not should_continue():
                    break

                chunk, _ = stream.read(block)
                chunk = chunk.flatten()
                buffer.append(chunk)
                total += len(chunk)

                rms = float(np.sqrt(np.mean(chunk**2)))

                # Calibrate against the room before trusting the threshold.
                if calib_seen < calib_needed:
                    calib_rms.append(rms)
                    calib_seen += len(chunk)
                    if calib_seen >= calib_needed:
                        noise_floor = float(np.mean(calib_rms))
                        adaptive = noise_floor * 2.5
                        if adaptive > threshold:
                            threshold = min(adaptive, MAX_SILENCE_THRESHOLD * 2)
                            log.info(
                                "ASR: ambient noise floor %.4f raises the VAD "
                                "threshold to %.4f", noise_floor, threshold,
                            )
                    continue

                if rms > threshold:
                    heard_speech = True
                    silence_run = 0
                elif heard_speech:
                    silence_run += len(chunk)

                bus.emit_sync(
                    Event.MIC_LEVEL,
                    data={"level": min(1.0, rms / threshold if threshold else 0.0)},
                    source="asr",
                )

                if heard_speech and silence_run >= silence_blocks:
                    log.info("ASR: silence detected, stopping recording")
                    break
    except Exception as exc:
        log.error("ASR: recording failed: %s", exc)
        return None

    if not heard_speech or not buffer:
        log.info("ASR: no speech detected")
        return None

    audio = np.concatenate(buffer)
    log.info("ASR: recorded %.1fs of audio", len(audio) / SAMPLE_RATE)
    return audio


# faster-whisper model names that can be given directly; anything else is a
# repo id, and a Nemotron or Wav2Vec2 repo id means nothing to CTranslate2.
WHISPER_SIZES = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo", "distil-small.en", "distil-medium.en",
    "distil-large-v2", "distil-large-v3",
})
# Small enough to fetch and run on a CPU while the real backend is broken.
FALLBACK_WHISPER_MODEL = "base"


def whisper_fallback_settings(settings: Any) -> Any:
    """Settings for a stand-in faster-whisper engine.

    `asr_model` usually names a model the fallback cannot load — the default
    is a Nemotron repo id — so point it at a Whisper size unless it already
    is one. The original settings object is never mutated.
    """
    model = str(getattr(settings, "asr_model", "") or "").strip()
    if model in WHISPER_SIZES:
        return settings

    copier = getattr(settings, "model_copy", None)
    if callable(copier):
        try:
            return copier(update={"asr_model": FALLBACK_WHISPER_MODEL})
        except Exception:
            pass

    try:
        clone = copy.copy(settings)
        clone.asr_model = FALLBACK_WHISPER_MODEL
        return clone
    except Exception:
        return settings


def build_whisper_fallback(
    settings: Any, engine_name: str, exc: Exception
) -> Any | None:
    """Stand faster-whisper up in place of a backend that cannot load.

    A backend that fails in `_load` fails identically on every trigger, so
    without this the assistant is deaf for the rest of the session and each
    turn dies with the same unreadable error. faster-whisper is a base
    dependency running on CTranslate2 — no torch, no torchaudio — so it
    survives the installs the other two die on.

    Returns None when there is nothing better to run, which tells the
    caller to re-raise.
    """
    from nixorb.hf import describe_native_import_error

    advice = describe_native_import_error(exc)
    log.error(
        "ASR: %s could not load (%s) — falling back to faster-whisper. %s",
        engine_name, exc, advice or "",
    )

    try:
        from nixorb.asr.whisper_engine import WhisperEngine

        return WhisperEngine(whisper_fallback_settings(settings))
    except Exception as build_exc:
        log.error("ASR: faster-whisper is unavailable too: %s", build_exc)
        return None


class ASREngine(ABC):
    """Base class for speech-to-text backends."""

    #: Human-readable name, used in logs and `nixorb status`.
    name: str = "asr"
    #: Whether stream_transcribe() does anything other than fall back.
    supports_streaming: bool = False
    #: Whether faster-whisper may stand in when this backend cannot load.
    #: True for the torch-backed engines: faster-whisper is a base
    #: dependency and runs on CTranslate2, so it survives a torch install
    #: that the others die on.
    whisper_fallback: bool = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._mic_index = getattr(settings, "microphone_index", None)
        self._mic_name = getattr(settings, "microphone_name", "") or ""
        self._recording = False
        #: Set when this engine failed to load and another took over.
        self._delegate: ASREngine | None = None
        #: Why the last attempt returned nothing, or "" if it was silence.
        #: Callers cannot tell those apart from a None return alone, and
        #: reporting a crash as "No speech detected" hides real faults.
        self.last_error: str = ""

    # ── Backend hooks ────────────────────────────────────────────── #

    @abstractmethod
    def _load(self) -> Any:
        """Load the model. Blocking; called in a worker thread."""

    @abstractmethod
    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe 16 kHz mono float32 audio. Blocking; called in a thread."""

    def _release(self, model: Any) -> None:
        """Free the model. Override only if a backend needs more than GC."""
        del model
        from nixorb.hf import free_cuda

        free_cuda()

    # ── Lifecycle ────────────────────────────────────────────────── #

    @property
    def is_loaded(self) -> bool:
        if self._delegate is not None:
            return self._delegate.is_loaded
        return self._model is not None

    @property
    def active_name(self) -> str:
        """The engine actually doing the work — not always this one."""
        if self._delegate is not None:
            return self._delegate.active_name
        return self.name

    async def preload(self) -> None:
        """Load the model into memory, off the event loop."""
        if self._delegate is not None:
            # A stand-in owns loading now. Retrying our own _load would
            # fail exactly as it did the first time.
            await self._delegate.preload()
            return
        if self._model is not None:
            return
        try:
            self._model = await asyncio.to_thread(self._load)
        except Exception as exc:
            delegate = self._build_fallback(exc)
            if delegate is None:
                raise
            self._delegate = delegate
            try:
                # Emits ASR_READY for the engine that actually loaded.
                await delegate.preload()
            except Exception as fallback_exc:
                self._delegate = None
                log.error(
                    "ASR: faster-whisper could not stand in either: %s",
                    fallback_exc,
                )
                raise exc from fallback_exc
            return
        await bus.emit(Event.ASR_READY, data={"engine": self.name}, source=self.name)

    def _build_fallback(self, exc: Exception) -> ASREngine | None:
        if not self.whisper_fallback:
            return None
        return build_whisper_fallback(self._settings, self.name, exc)

    async def unload(self) -> None:
        if self._delegate is not None:
            # Unload its model, but KEEP the stand-in. Whether this backend
            # can load is a property of the installation, not of this turn;
            # dropping the delegate here made every turn re-run a load
            # already known to fail, re-log the whole explanation, and
            # rebuild the stand-in from scratch.
            await self._delegate.unload()
            return
        if self._model is None:
            return
        model, self._model = self._model, None
        await asyncio.to_thread(self._release, model)
        log.info("ASR: %s unloaded", self.name)

    def stop_recording(self) -> None:
        self._recording = False
        if self._delegate is not None:
            self._delegate.stop_recording()

    # ── Pipeline ─────────────────────────────────────────────────── #

    async def record_and_transcribe(self) -> str | None:
        """Record until silence, then transcribe. The orb's main entry point."""
        if self._delegate is None and self._model is None:
            await self.preload()
        if self._delegate is not None:
            transcript = await self._delegate.record_and_transcribe()
            # getattr: a stand-in need not be an ASREngine subclass.
            self.last_error = getattr(self._delegate, "last_error", "")
            return transcript

        await bus.emit(Event.RECORDING_START, source=self.name)
        self._recording = True
        self.last_error = ""

        try:
            audio = await asyncio.to_thread(
                record_with_vad,
                self._mic_index,
                self._vad_threshold,
                MAX_RECORDING_DURATION,
                SILENCE_TIMEOUT,
                lambda: self._recording,
                self._mic_name,
            )
            await bus.emit(Event.RECORDING_STOP, source=self.name)

            if audio is None or len(audio) < SAMPLE_RATE * MIN_SPEECH_SECONDS:
                log.info("ASR: audio too short or empty")
                return None

            await bus.emit(Event.ORB_THINKING, source=self.name)
            text = (await asyncio.to_thread(self._transcribe, audio)).strip()

            if text:
                await bus.emit(
                    Event.TRANSCRIPT_READY, data={"text": text}, source=self.name
                )
            return text or None

        except Exception as exc:
            log.error("ASR: %s failed: %s", self.name, exc)
            self.last_error = str(exc)
            await bus.emit(
                Event.ASR_ERROR, data={"error": str(exc)}, source=self.name
            )
            return None
        finally:
            self._recording = False

    async def stream_transcribe(self) -> AsyncIterator[str]:
        """Yield partial transcripts while the user is still speaking.

        Backends that cannot stream fall back to one final result, so callers
        never need to branch on supports_streaming.
        """
        if self._delegate is None and self._model is None:
            await self.preload()
        if self._delegate is not None:
            async for partial in self._delegate.stream_transcribe():
                yield partial
            return

        text = await self.record_and_transcribe()
        if text:
            yield text

    @property
    def _vad_threshold(self) -> float:
        return sensitivity_to_threshold(
            getattr(self._settings, "mic_sensitivity", 0.5)
        )
