"""
nixorb/asr/whisper_engine.py

faster-whisper ASR with non-blocking recording and VRAM paging.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd
import torch

from nixorb.core.event_bus import Event, EventBus
from nixorb.core.vram_manager import ModelPriority, vram

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
CHUNK_FRAMES = 1_024
SILENCE_DB = -38.0
SILENCE_SECS = 1.2
INITIAL_LISTEN_S = 8.0
MAX_RECORD_S = 30.0


def _preferred_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_whisper():
    from faster_whisper import WhisperModel

    from nixorb.settings import Settings

    settings = Settings.load()
    model_name = settings.asr_model or "large-v3"
    device = _preferred_device()
    compute_type = "int8_float16" if device == "cuda" else "int8"
    log.info("Loading faster-whisper %s on %s (%s)", model_name, device, compute_type)
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=2,
        )
    except Exception:
        if device == "cuda":
            log.exception("CUDA Whisper load failed; retrying on CPU")
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=2,
            )
        else:
            raise
    log.info("faster-whisper model ready: %s", model_name)
    return model


def _unload_whisper(model) -> None:
    del model


vram.register(
    name="whisper",
    vram_mb=2_100,
    priority=ModelPriority.LOW,
    load_fn=_load_whisper,
    unload_fn=_unload_whisper,
)


class WhisperEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def preload(self) -> None:
        """Download/load the configured ASR model without recording."""
        async with await self._whisper_lease():
            return

    async def record_and_transcribe(self) -> str | None:
        await EventBus().emit(Event.RECORDING_START, source="whisper")
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(None, self._record_blocking)
        await EventBus().emit(Event.RECORDING_STOP, source="whisper")

        if audio is None or len(audio) < int(SAMPLE_RATE * 0.3):
            log.info("No usable microphone audio captured")
            return None
        return await self._transcribe_async(audio)

    def _emit_mic_level(self, level: float, rms_db: float) -> None:
        try:
            EventBus().emit_sync(
                Event.MIC_LEVEL,
                data={"level": max(0.0, min(1.0, level)), "rms_db": rms_db},
                source="whisper",
                priority=3,
            )
        except Exception:
            log.debug("Unable to emit microphone level", exc_info=True)

    def _record_blocking(self) -> np.ndarray | None:
        chunks: list[np.ndarray] = []
        speech_started = False
        silence_start: float | None = None
        start = time.monotonic()
        device = self._settings.microphone_index

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_FRAMES,
                device=device,
            ) as stream:
                log.info("Recording started (device=%s)", device)
                while True:
                    chunk, overflowed = stream.read(CHUNK_FRAMES)
                    if overflowed:
                        log.warning("Microphone input overflowed")
                    chunk = chunk.copy()

                    rms = float(np.sqrt(np.mean(chunk**2)) + 1e-10)
                    rms_db = 20.0 * np.log10(rms)
                    level = min(1.0, max(0.0, (rms_db - SILENCE_DB) / 38.0))
                    self._emit_mic_level(level, rms_db)
                    elapsed = time.monotonic() - start

                    if rms_db > SILENCE_DB:
                        speech_started = True
                        silence_start = None
                        chunks.append(chunk)
                    elif speech_started:
                        chunks.append(chunk)
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elif (time.monotonic() - silence_start) >= SILENCE_SECS:
                            log.info("End of speech detected (%.1f s)", elapsed)
                            break
                    elif elapsed >= INITIAL_LISTEN_S:
                        log.info("No microphone activity detected within %.1f s", INITIAL_LISTEN_S)
                        break

                    if elapsed >= MAX_RECORD_S:
                        log.warning("Max recording duration reached")
                        break
        except sd.PortAudioError as exc:
            log.error("PortAudio error during recording: %s", exc)
            return None

        if not chunks:
            return None
        return np.concatenate(chunks, axis=0).flatten()

    async def _whisper_lease(self) -> Any:
        """Return the Whisper VRAM lease, tolerating async test doubles."""
        lease = vram.lease("whisper")
        if inspect.isawaitable(lease):
            lease = await lease
        return lease

    async def _transcribe_async(self, audio: np.ndarray) -> str | None:
        if audio is None or len(audio) == 0 or not np.any(audio):
            return None

        async with await self._whisper_lease() as model:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._transcribe_sync, model, audio)
        if text:
            await EventBus().emit(
                Event.TRANSCRIPT_READY,
                data={"text": text},
                source="whisper",
                priority=2,
            )
            log.info("Transcript: %s", text[:120])
        return text or None

    def _transcribe_sync(self, model, audio: np.ndarray) -> str:
        language = self._settings.asr_language or None
        segments, _info = model.transcribe(
            audio,
            beam_size=5,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
