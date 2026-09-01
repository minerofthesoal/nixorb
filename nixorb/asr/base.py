"""Shared ASR plumbing — microphone capture, VAD, and the engine interface.

Every ASR backend needs the same "record until they stop talking, then
transcribe" loop; only the transcribe step differs. That loop lives here so
faster-whisper, a generic Hugging Face model and Nemotron all behave the
same way from the orb's point of view.
"""
from __future__ import annotations

import asyncio
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
SILENCE_THRESHOLD = 0.015  # RMS below this counts as silence
SILENCE_TIMEOUT = 2.0  # seconds of silence after speech before stopping
MAX_RECORDING_DURATION = 30.0
MIN_SPEECH_SECONDS = 0.3  # anything shorter is a click, not a sentence


def record_with_vad(
    mic_index: int | None = None,
    threshold: float = SILENCE_THRESHOLD,
    max_seconds: float = MAX_RECORDING_DURATION,
    silence_seconds: float = SILENCE_TIMEOUT,
    should_continue: Any = None,
) -> np.ndarray | None:
    """Record from the microphone until silence. Blocking — run in a thread.

    Returns the captured audio, or None if nothing was said.
    """
    import sounddevice as sd

    block = int(SAMPLE_RATE * CHUNK_DURATION)
    silence_blocks = int(SAMPLE_RATE * silence_seconds)
    max_samples = int(SAMPLE_RATE * max_seconds)

    buffer: list[np.ndarray] = []
    silence_run = 0
    total = 0
    heard_speech = False

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=mic_index,
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


class ASREngine(ABC):
    """Base class for speech-to-text backends."""

    #: Human-readable name, used in logs and `nixorb status`.
    name: str = "asr"
    #: Whether stream_transcribe() does anything other than fall back.
    supports_streaming: bool = False

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._mic_index = getattr(settings, "microphone_index", None)
        self._recording = False

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
        return self._model is not None

    async def preload(self) -> None:
        """Load the model into memory, off the event loop."""
        if self._model is not None:
            return
        self._model = await asyncio.to_thread(self._load)
        await bus.emit(Event.ASR_READY, data={"engine": self.name}, source=self.name)

    async def unload(self) -> None:
        if self._model is None:
            return
        model, self._model = self._model, None
        await asyncio.to_thread(self._release, model)
        log.info("ASR: %s unloaded", self.name)

    def stop_recording(self) -> None:
        self._recording = False

    # ── Pipeline ─────────────────────────────────────────────────── #

    async def record_and_transcribe(self) -> str | None:
        """Record until silence, then transcribe. The orb's main entry point."""
        if self._model is None:
            await self.preload()

        await bus.emit(Event.RECORDING_START, source=self.name)
        self._recording = True

        try:
            audio = await asyncio.to_thread(
                record_with_vad,
                self._mic_index,
                self._vad_threshold,
                MAX_RECORDING_DURATION,
                SILENCE_TIMEOUT,
                lambda: self._recording,
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
        text = await self.record_and_transcribe()
        if text:
            yield text

    @property
    def _vad_threshold(self) -> float:
        """Mic sensitivity maps onto the VAD threshold, inverted.

        Higher sensitivity should trigger on quieter speech, so it must
        *lower* the RMS threshold.
        """
        sensitivity = float(getattr(self._settings, "mic_sensitivity", 0.5) or 0.5)
        sensitivity = max(0.05, min(1.0, sensitivity))
        return SILENCE_THRESHOLD * (1.0 / (sensitivity * 2))
