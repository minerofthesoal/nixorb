"""
nixorb/asr/whisper_engine.py

faster-whisper Large v3 with INT8 quantisation and VRAM paging.

BUG FIX PASS 1:
  - _transcribe_blocking called asyncio.get_event_loop() from inside a
    ThreadPoolExecutor worker. In Python 3.10+ this issues a DeprecationWarning
    and in 3.12 it raises RuntimeError because there is no running loop in the
    thread. Fixed by capturing the loop at construction time and storing it.

BUG FIX PASS 2:
  - asyncio.run_coroutine_threadsafe result was awaited inside the executor
    thread with .result(timeout=60), but the coroutine itself acquires
    VRAMManager locks which are asyncio.Lock objects — awaitable only on
    the main loop. Restructured: recording runs in thread pool, transcription
    is fully async on the main loop, no cross-thread coroutine submission.

BUG FIX PASS 3:
  - sounddevice InputStream blocksize parameter was passed as CHUNK_FRAMES but
    the callback variant of InputStream was mixed up with the read() API.
    Corrected to use explicit stream.read() in a loop (non-callback mode).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import ModelPriority, vram

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE  = 16_000
CHANNELS     = 1
DTYPE        = "float32"
CHUNK_FRAMES = 1_024
SILENCE_DB   = -38.0    # dBFS; above this = speech
SILENCE_SECS = 1.2      # seconds of consecutive silence = end of utterance
MAX_RECORD_S = 30.0     # hard cap


def _load_whisper():
    from faster_whisper import WhisperModel
    model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="int8_float16",   # INT8 weights, FP16 compute ≈ 2 GB VRAM
        cpu_threads=4,
        num_workers=2,
    )
    log.info("Whisper Large v3 (int8_float16) loaded")
    return model


def _unload_whisper(model) -> None:
    del model


# Register once at import time
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

    # ---------------------------------------------------------------- #
    #  Public entry point                                               #
    # ---------------------------------------------------------------- #
    async def record_and_transcribe(self) -> str | None:
        """
        Record from microphone until silence, then transcribe.

        Recording is blocking I/O — runs in the default thread pool.
        Transcription runs on the async loop using vram.lease().
        """
        await bus.emit(Event.RECORDING_START, source="whisper")
        loop = asyncio.get_running_loop()

        # BUG FIX: record in thread, transcribe on async loop (no cross-thread
        # coroutine submission needed)
        audio = await loop.run_in_executor(None, self._record_blocking)

        await bus.emit(Event.RECORDING_STOP, source="whisper")

        if audio is None or len(audio) < int(SAMPLE_RATE * 0.3):
            log.debug("Recording too short or empty — skipping transcription")
            return None

        return await self._transcribe_async(audio)

    # ---------------------------------------------------------------- #
    #  Recording (blocking, runs in thread pool)                        #
    # ---------------------------------------------------------------- #
    def _record_blocking(self) -> np.ndarray | None:
        chunks: list[np.ndarray] = []
        silence_start: float | None = None
        start = time.monotonic()
        device = self._settings.microphone_index

        try:
            # BUG FIX: use non-callback InputStream.read() correctly
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=CHUNK_FRAMES,
                device=device,
            ) as stream:
                log.info("Recording started (device=%s)", device)
                while True:
                    # read() returns (data: ndarray, overflowed: bool)
                    chunk, _overflowed = stream.read(CHUNK_FRAMES)
                    chunks.append(chunk.copy())

                    rms    = float(np.sqrt(np.mean(chunk ** 2)) + 1e-10)
                    rms_db = 20.0 * np.log10(rms)
                    elapsed = time.monotonic() - start

                    if rms_db > SILENCE_DB:
                        silence_start = None          # reset silence timer on speech
                    else:
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elif (time.monotonic() - silence_start) >= SILENCE_SECS:
                            log.info("End of speech detected (%.1f s)", elapsed)
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

    # ---------------------------------------------------------------- #
    #  Transcription (fully async, runs on main event loop)            #
    # ---------------------------------------------------------------- #
    async def _transcribe_async(self, audio: np.ndarray) -> str | None:
        """Acquire Whisper from VRAM manager and transcribe."""
        async with vram.lease("whisper") as model:
            loop = asyncio.get_running_loop()
            # Run the synchronous faster-whisper call in the thread pool
            text = await loop.run_in_executor(
                None, self._transcribe_sync, model, audio
            )
        if text:
            await bus.emit(
                Event.TRANSCRIPT_READY,
                data={"text": text},
                source="whisper",
                priority=2,
            )
            log.info("Transcript: %s", text[:120])
        return text or None

    @staticmethod
    def _transcribe_sync(model, audio: np.ndarray) -> str:
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language=None,           # auto-detect
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            word_timestamps=False,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
