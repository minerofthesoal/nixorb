"""
nixorb/asr/whisper_engine.py

Faster-Whisper Large v3 ASR with:
  - INT8 quantization for GTX 1080
  - VRAM paging (unloaded while LLM/TTS runs)
  - Streaming VAD-gated recording via sounddevice
  - Thread-safe result queue
"""
from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import time
from typing import Generator

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, EventPayload, bus
from nixorb.core.vram_manager import ModelPriority, vram

log = logging.getLogger(__name__)

SAMPLE_RATE   = 16_000
CHANNELS      = 1
DTYPE         = "float32"
CHUNK_FRAMES  = 1024
SILENCE_DB    = -40.0          # dBFS threshold for VAD
SILENCE_SECS  = 1.2            # seconds of silence to stop recording
MAX_RECORD_S  = 30             # hard cap


def _load_whisper():
    from faster_whisper import WhisperModel
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="int8_float16",   # INT8 weights, FP16 compute — saves ~1 GB vs FP16
        cpu_threads=4,
        num_workers=2,
    )
    log.info("Whisper Large v3 (INT8) loaded")
    return model


def _unload_whisper(model) -> None:
    del model


# Register with VRAM manager (~2.0 GB for INT8 Large v3)
vram.register(
    name="whisper",
    vram_mb=2048,
    priority=ModelPriority.LOW,
    load_fn=_load_whisper,
    unload_fn=_unload_whisper,
)


class WhisperEngine:
    def __init__(self, settings) -> None:
        self._settings  = settings
        self._recording  = False
        self._audio_buf: list[np.ndarray] = []
        self._lock       = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Recording                                                           #
    # ------------------------------------------------------------------ #
    async def record_and_transcribe(self) -> str | None:
        """
        Record from mic until silence, then transcribe.
        Runs audio capture on a dedicated thread; transcription in executor.
        """
        await bus.emit(Event.RECORDING_START, source="whisper")
        loop = asyncio.get_running_loop()

        audio_data = await loop.run_in_executor(None, self._record_blocking)

        if audio_data is None or len(audio_data) < SAMPLE_RATE * 0.3:
            await bus.emit(Event.RECORDING_STOP, source="whisper")
            return None

        await bus.emit(Event.RECORDING_STOP, source="whisper")

        transcript = await self._transcribe(audio_data)
        if transcript:
            await bus.emit(
                Event.TRANSCRIPT_READY,
                data={"text": transcript},
                source="whisper",
                priority=2,
            )
        return transcript

    def _record_blocking(self) -> np.ndarray | None:
        """Blocking VAD-gated mic recording (runs in thread pool)."""
        chunks: list[np.ndarray] = []
        silence_start: float | None = None
        recording_start = time.monotonic()

        device_idx = self._settings.microphone_index

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_FRAMES,
            device=device_idx,
        ) as stream:
            log.info("Recording started (device=%s)", device_idx)
            while True:
                chunk, _ = stream.read(CHUNK_FRAMES)
                chunks.append(chunk.copy())

                rms_db = 20 * np.log10(np.sqrt(np.mean(chunk ** 2)) + 1e-9)
                elapsed = time.monotonic() - recording_start

                if rms_db > SILENCE_DB:
                    silence_start = None
                else:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif (time.monotonic() - silence_start) > SILENCE_SECS:
                        log.info("Silence detected after %.1f s", elapsed)
                        break

                if elapsed > MAX_RECORD_S:
                    log.warning("Max recording duration reached")
                    break

        if not chunks:
            return None
        return np.concatenate(chunks, axis=0).flatten()

    # ------------------------------------------------------------------ #
    #  Transcription                                                       #
    # ------------------------------------------------------------------ #
    async def _transcribe(self, audio: np.ndarray) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_blocking, audio)

    def _transcribe_blocking(self, audio: np.ndarray) -> str | None:
        # Synchronous — called from thread pool
        # We use asyncio.run_coroutine_threadsafe to get the model via VRAMManager
        future = asyncio.run_coroutine_threadsafe(
            self._transcribe_async(audio),
            asyncio.get_event_loop(),
        )
        return future.result(timeout=60)

    async def _transcribe_async(self, audio: np.ndarray) -> str | None:
        async with vram.lease("whisper") as model:
            segments, info = model.transcribe(
                audio,
                beam_size=5,
                language=self._settings.asr_language or None,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                word_timestamps=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            log.info("Transcript [%.1f s]: %s", info.duration, text[:80])
            return text or None
