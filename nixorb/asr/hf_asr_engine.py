"""NixOrb HuggingFace ASR engine — speech-to-text via transformers.

Unlike WhisperEngine (which only loads CTranslate2 Whisper checkpoints
through faster-whisper), this engine loads *any* model transformers'
``automatic-speech-recognition`` pipeline supports: Whisper variants
(including ones faster-whisper can't load, like fine-tunes only published
as safetensors), distil-whisper, Wav2Vec2/HuBERT-family CTC models, NVIDIA
Parakeet, Moonshine, or a private/custom fine-tune — anything on the Hub or
a local path works as long as it registers with that pipeline.

It shares WhisperEngine's public interface (``is_loaded``, ``preload``,
``unload``, ``stop_recording``, ``record_and_transcribe``) so main.py can
use either interchangeably, and reuses its recording/VAD constants and mic
device resolution so both engines behave identically at the microphone —
the only bugs worth having twice are new ones, not old ones.

Streaming mode (``settings.asr_streaming``): most HF ASR models have no
native streaming API, so this approximates it by re-transcribing the
buffered audio so far every ``asr_streaming_chunk_seconds`` on a background
thread, emitting ``Event.ASR_PARTIAL_TRANSCRIPT`` as the transcript
firms up. The audio capture loop is never blocked on this — a partial
decode in progress is simply skipped rather than queued, so a slow model
degrades to fewer partial updates rather than lagging capture.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.asr.base import (
    CALIBRATION_SECONDS,
    CHUNK_DURATION,
    MAX_RECORDING_DURATION,
    MAX_SILENCE_THRESHOLD,
    SAMPLE_RATE,
    SILENCE_TIMEOUT,
)
from nixorb.asr.base import sensitivity_to_threshold as _sensitivity_to_threshold
from nixorb.core.event_bus import Event, bus
from nixorb.utils.audio import describe_devices, resolve_input_device

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

CHANNELS = 1
DTYPE = np.float32


class HFASREngine:
    """ASR engine using any transformers automatic-speech-recognition model."""

    #: Reported in logs and `nixorb status`, like every other engine.
    name = "huggingface"
    #: It has no native streaming API; partials come from re-transcribing a
    #: rolling buffer, which is approximate but works for any HF model.
    supports_streaming = True


    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._repo_id = settings.asr_model
        self._token = getattr(settings, "hf_token", "") or None
        self._language = settings.asr_language or None
        self._mic_index = settings.microphone_index
        self._mic_name = getattr(settings, "microphone_name", "") or ""
        self._base_threshold = _sensitivity_to_threshold(
            getattr(settings, "mic_sensitivity", 0.5)
        )
        self._streaming = bool(getattr(settings, "asr_streaming", False))
        self._chunk_seconds = float(
            getattr(settings, "asr_streaming_chunk_seconds", 2.5) or 2.5
        )
        self._pipe: Any = None
        self._recording = False
        #: Set when this engine failed to load and faster-whisper took over.
        self._delegate: Any = None
        #: Why the last attempt returned nothing — see ASREngine.last_error.
        self.last_error: str = ""
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._partial_busy = threading.Event()
        self._devices_logged = False

    @property
    def is_loaded(self) -> bool:
        if self._delegate is not None:
            return bool(self._delegate.is_loaded)
        return self._pipe is not None

    @property
    def active_name(self) -> str:
        """The engine actually doing the work — not always this one."""
        if self._delegate is not None:
            return str(getattr(self._delegate, "active_name", self._delegate.name))
        return self.name

    # ── model lifecycle ─────────────────────────────────────────── #

    def _load_pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:
            # Not necessarily transformers: this fires for anything in its
            # import chain too, torch most often. Let the error say which.
            from nixorb.hf import explain_import_error

            raise RuntimeError(
                explain_import_error(exc, "transformers", "HuggingFace ASR")
            ) from exc

        device = -1
        try:
            import torch
            if torch.cuda.is_available():
                device = 0
        except ImportError:
            pass

        log.info(
            "ASR: loading HuggingFace model '%s' (device=%s)",
            self._repo_id, "cuda" if device == 0 else "cpu",
        )
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self._repo_id,
                token=self._token,
                device=device,
                chunk_length_s=30,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load HuggingFace ASR model '{self._repo_id}': {exc}"
            ) from exc
        log.info("ASR: HuggingFace model loaded: %s", self._repo_id)
        return pipe

    def _unload_pipeline(self, _pipe: Any) -> None:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("ASR: HuggingFace model unloaded")

    async def preload(self) -> None:
        if self._delegate is not None:
            await self._delegate.preload()
            return
        if self._pipe is not None:
            return
        loop = asyncio.get_running_loop()
        try:
            self._pipe = await loop.run_in_executor(None, self._load_pipeline)
        except Exception as exc:
            # transformers>=5 imports torchaudio at module scope, so a torch
            # install whose pieces disagree takes this engine down before it
            # sees a single sample. Keep the assistant hearing.
            from nixorb.asr.base import build_whisper_fallback

            delegate = build_whisper_fallback(self._settings, self.name, exc)
            if delegate is None:
                raise
            self._delegate = delegate
            try:
                await delegate.preload()
            except Exception as fallback_exc:
                self._delegate = None
                log.error(
                    "ASR: faster-whisper could not stand in either: %s",
                    fallback_exc,
                )
                raise exc from fallback_exc
            return
        await bus.emit(Event.ASR_READY, source="HFASREngine")

    async def unload(self) -> None:
        if self._delegate is not None:
            # Keep the stand-in across unloads — see ASREngine.unload.
            await self._delegate.unload()
            return
        if self._pipe is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._unload_pipeline, self._pipe)
            self._pipe = None

    def stop_recording(self) -> None:
        self._recording = False
        if self._delegate is not None:
            self._delegate.stop_recording()

    # ── transcription ───────────────────────────────────────────── #

    def _transcribe_array(self, audio: np.ndarray) -> str:
        if self._pipe is None:
            raise RuntimeError("HuggingFace ASR model not loaded")
        kwargs: dict[str, Any] = {}
        if self._language:
            kwargs["generate_kwargs"] = {"language": self._language}
        try:
            result = self._pipe({"array": audio, "sampling_rate": SAMPLE_RATE}, **kwargs)
        except (TypeError, ValueError):
            # Not every architecture (e.g. plain CTC models) accepts
            # generate_kwargs or a language hint — retry without them
            # rather than losing the whole transcript over it.
            result = self._pipe({"array": audio, "sampling_rate": SAMPLE_RATE})
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip()

    def _emit_partial(self, snapshot: np.ndarray) -> None:
        """Run one partial decode on a background thread. Never blocks capture."""
        if self._partial_busy.is_set():
            return
        self._partial_busy.set()

        def _worker() -> None:
            try:
                text = self._transcribe_array(snapshot)
                if text:
                    bus.emit_sync(
                        Event.ASR_PARTIAL_TRANSCRIPT,
                        data={"text": text},
                        source="HFASREngine",
                    )
            except Exception as exc:
                log.debug("ASR: partial decode failed (continuing): %s", exc)
            finally:
                self._partial_busy.clear()

        threading.Thread(target=_worker, daemon=True).start()

    # ── recording ────────────────────────────────────────────────── #

    def _record_audio_sync(self) -> np.ndarray | None:
        """Synchronous audio recording with VAD (runs in thread).

        Mirrors WhisperEngine's recorder (see that module for why each
        piece — device resolution, ambient-noise calibration — exists),
        with an added streaming hook that fires partial decodes on a
        separate thread every ``asr_streaming_chunk_seconds``.
        """
        self._recording = True
        self._audio_buffer = []

        resolved = resolve_input_device(self._mic_index, self._mic_name)
        if not self._devices_logged:
            log.info("ASR: available input devices:\n%s", describe_devices())
            self._devices_logged = True
        log.info(
            "ASR: starting recording… (device[%s]='%s', reason: %s, streaming=%s)",
            resolved.index, resolved.name, resolved.reason, self._streaming,
        )

        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        silence_samples = int(SAMPLE_RATE * SILENCE_TIMEOUT)
        max_samples = int(SAMPLE_RATE * MAX_RECORDING_DURATION)
        partial_interval_samples = max(chunk_samples, int(SAMPLE_RATE * self._chunk_seconds))

        silence_counter = 0
        total_samples = 0
        samples_since_partial = 0
        has_speech = False
        peak_rms = 0.0
        threshold = self._base_threshold

        try:
            import sounddevice as sd

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=resolved.index,
                blocksize=chunk_samples,
            ) as stream:
                calib_samples_needed = int(SAMPLE_RATE * CALIBRATION_SECONDS)
                calib_samples_seen = 0
                calib_rms_values: list[float] = []

                while self._recording and total_samples < max_samples:
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk.flatten()
                    with self._buffer_lock:
                        self._audio_buffer.append(chunk)
                    total_samples += len(chunk)
                    samples_since_partial += len(chunk)

                    rms = float(np.sqrt(np.mean(chunk**2)))
                    peak_rms = max(peak_rms, rms)

                    if calib_samples_seen < calib_samples_needed:
                        calib_rms_values.append(rms)
                        calib_samples_seen += len(chunk)
                        if calib_samples_seen >= calib_samples_needed:
                            noise_floor = float(np.mean(calib_rms_values))
                            adaptive = noise_floor * 2.5
                            if adaptive > threshold:
                                threshold = min(adaptive, MAX_SILENCE_THRESHOLD * 2)
                        continue

                    if rms > threshold:
                        has_speech = True
                        silence_counter = 0
                    elif has_speech:
                        silence_counter += len(chunk)

                    level = min(1.0, rms / threshold)
                    bus.emit_sync(
                        Event.MIC_LEVEL, data={"level": float(level)},
                        source="HFASREngine",
                    )

                    if (
                        self._streaming and has_speech
                        and samples_since_partial >= partial_interval_samples
                    ):
                        samples_since_partial = 0
                        with self._buffer_lock:
                            snapshot = np.concatenate(self._audio_buffer)
                        self._emit_partial(snapshot)

                    if has_speech and silence_counter >= silence_samples:
                        log.info("ASR: silence detected, stopping recording")
                        break

            if not has_speech:
                log.info(
                    "ASR: no speech detected (peak input level %.4f, "
                    "threshold %.4f, device '%s')",
                    peak_rms, threshold, resolved.name,
                )
                return None

            with self._buffer_lock:
                audio = np.concatenate(self._audio_buffer)
            log.info("ASR: recorded %.1fs of audio", len(audio) / SAMPLE_RATE)
            return audio

        except Exception as exc:
            log.error(
                "ASR: recording failed on device[%s]='%s': %s",
                resolved.index, resolved.name, exc,
            )
            return None
        finally:
            self._recording = False

    async def record_and_transcribe(self) -> str | None:
        """Record audio and return transcript. Full pipeline."""
        if self._delegate is None and self._pipe is None:
            await self.preload()
        if self._delegate is not None:
            transcript = await self._delegate.record_and_transcribe()
            self.last_error = getattr(self._delegate, "last_error", "")
            return transcript

        await bus.emit(Event.RECORDING_START, source="HFASREngine")
        self.last_error = ""

        try:
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, self._record_audio_sync)

            if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                log.info("ASR: audio too short or empty")
                return None

            await bus.emit(Event.RECORDING_STOP, source="HFASREngine")
            await bus.emit(Event.ORB_THINKING, source="HFASREngine")

            # A final full-buffer pass, even in streaming mode — it's more
            # accurate than the last partial (which may have decoded a
            # truncated clip), and costs one extra inference per turn.
            text = await loop.run_in_executor(None, self._transcribe_array, audio)

            if text:
                await bus.emit(
                    Event.TRANSCRIPT_READY, data={"text": text}, source="HFASREngine"
                )
            return text

        except Exception as exc:
            log.error("ASR: record_and_transcribe failed: %s", exc)
            self.last_error = str(exc)
            await bus.emit(
                Event.ASR_ERROR, data={"error": str(exc)}, source="HFASREngine"
            )
            return None

    async def stream_transcribe(self) -> AsyncIterator[str]:
        """Yield the transcript once recording ends.

        This engine has no native streaming API — its partials come from
        re-transcribing a rolling buffer and go out on the bus as
        ASR_PARTIAL, not through this iterator. It still has to exist:
        `supports_streaming` is True, so main calls this whenever
        asr_streaming is on, and without it the turn died on an
        AttributeError.
        """
        if self._delegate is None and self._pipe is None:
            await self.preload()
        if self._delegate is not None:
            async for partial in self._delegate.stream_transcribe():
                yield partial
            return

        text = await self.record_and_transcribe()
        if text:
            yield text
