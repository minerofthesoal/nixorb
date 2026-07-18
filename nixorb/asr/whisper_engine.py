"""NixOrb Whisper ASR engine — speech-to-text using faster-whisper.

Records audio from the microphone, detects voice activity, and transcribes
using Whisper Large v3 optimized for GTX 1080 (INT8 quantization).
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
import soundfile as sf

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
CHUNK_DURATION = 0.5  # seconds per audio chunk
SILENCE_THRESHOLD = 0.015
SILENCE_TIMEOUT = 2.0  # seconds of silence before stopping
MAX_RECORDING_DURATION = 30.0  # maximum recording length
VAD_WINDOW_MS = 30  # voice activity detection window


class WhisperEngine:
    """ASR engine using faster-whisper for local speech-to-text."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._model_name = settings.asr_model
        self._language = settings.asr_language or "en"
        self._mic_index = settings.microphone_index
        self._recording = False
        self._audio_buffer: list[np.ndarray] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self):
        """Load the faster-whisper model (runs in executor)."""
        try:
            from faster_whisper import WhisperModel

            # Use INT8 for GTX 1080 — ~2.1 GB VRAM
            compute_type = "int8_float16"
            device = "cuda"

            log.info("ASR: loading Whisper %s (%s, %s)", self._model_name, device, compute_type)
            model = WhisperModel(
                self._model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
            )
            log.info("ASR: Whisper model loaded successfully")
            return model
        except Exception as exc:
            log.error("ASR: failed to load Whisper model: %s", exc)
            raise

    def _unload_model(self, model) -> None:
        """Unload the model and free VRAM."""
        del model
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("ASR: Whisper model unloaded")

    async def preload(self) -> None:
        """Preload the Whisper model into VRAM."""
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        await bus.emit(Event.ASR_READY, source="WhisperEngine")

    async def unload(self) -> None:
        """Unload the model from VRAM."""
        if self._model is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._unload_model, self._model)
            self._model = None

    def _record_audio_sync(self) -> np.ndarray | None:
        """Synchronous audio recording with VAD (runs in thread)."""
        log.info("ASR: starting recording…")
        self._recording = True
        self._audio_buffer = []

        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        silence_samples = int(SAMPLE_RATE * SILENCE_TIMEOUT)
        max_samples = int(SAMPLE_RATE * MAX_RECORDING_DURATION)

        silence_counter = 0
        total_samples = 0
        has_speech = False

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                device=self._mic_index,
                blocksize=chunk_samples,
            ) as stream:
                while self._recording and total_samples < max_samples:
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk.flatten()
                    self._audio_buffer.append(chunk)
                    total_samples += len(chunk)

                    # Voice activity detection
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms > SILENCE_THRESHOLD:
                        has_speech = True
                        silence_counter = 0
                    elif has_speech:
                        silence_counter += len(chunk)

                    # Emit mic level for UI visualization
                    level = min(1.0, rms / SILENCE_THRESHOLD)
                    bus.emit_sync(
                        Event.MIC_LEVEL,
                        data={"level": float(level)},
                        source="WhisperEngine",
                    )

                    # Stop on prolonged silence after speech
                    if has_speech and silence_counter >= silence_samples:
                        log.info("ASR: silence detected, stopping recording")
                        break

            if not has_speech:
                log.info("ASR: no speech detected")
                return None

            audio = np.concatenate(self._audio_buffer)
            log.info("ASR: recorded %.1fs of audio", len(audio) / SAMPLE_RATE)
            return audio

        except Exception as exc:
            log.error("ASR: recording failed: %s", exc)
            return None
        finally:
            self._recording = False

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription (runs in thread)."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            # Convert float32 to int16 WAV in memory
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            wav_buffer.seek(0)

            segments, info = self._model.transcribe(
                wav_buffer,
                language=self._language,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=True,
            )

            text = " ".join(segment.text for segment in segments).strip()
            log.info(
                "ASR: transcribed (lang=%s, prob=%.2f): %s",
                info.language,
                info.language_probability,
                text[:100],
            )
            return text

        except Exception as exc:
            log.error("ASR: transcription failed: %s", exc)
            raise

    def stop_recording(self) -> None:
        """Signal the recorder to stop."""
        self._recording = False

    async def record_and_transcribe(self) -> str | None:
        """Record audio and return transcript. Full pipeline."""
        # Ensure model is loaded
        if self._model is None:
            await self.preload()

        await bus.emit(Event.RECORDING_START, source="WhisperEngine")

        try:
            # Record audio in thread
            loop = asyncio.get_running_loop()
            audio = await loop.run_in_executor(None, self._record_audio_sync)

            if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                log.info("ASR: audio too short or empty")
                return None

            await bus.emit(Event.RECORDING_STOP, source="WhisperEngine")
            await bus.emit(Event.ORB_THINKING, source="WhisperEngine")

            # Transcribe in thread
            text = await loop.run_in_executor(None, self._transcribe_sync, audio)

            if text:
                await bus.emit(
                    Event.TRANSCRIPT_READY,
                    data={"text": text},
                    source="WhisperEngine",
                )
            return text

        except Exception as exc:
            log.error("ASR: record_and_transcribe failed: %s", exc)
            await bus.emit(
                Event.ASR_ERROR,
                data={"error": str(exc)},
                source="WhisperEngine",
            )
            return None
