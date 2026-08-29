"""NixOrb Whisper ASR engine — speech-to-text using faster-whisper.

Records audio from the microphone, detects voice activity, and transcribes
using Whisper Large v3 optimized for GTX 1080 (INT8 quantization).
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.core.event_bus import Event, bus
from nixorb.utils.audio import describe_devices, resolve_input_device

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32
CHUNK_DURATION = 0.5  # seconds per audio chunk
# Base VAD threshold at the default mic_sensitivity (0.5) — kept as a plain
# module constant because it doubles as the documented "this is roughly how
# loud speech needs to be" baseline, and tests pin it directly.
SILENCE_THRESHOLD = 0.015
# Sensitivity 0.0 maps to this (needs a loud/close voice), 1.0 maps to
# MIN_SILENCE_THRESHOLD (picks up quiet/distant speech, more false triggers).
MAX_SILENCE_THRESHOLD = 0.045
MIN_SILENCE_THRESHOLD = 0.004
SILENCE_TIMEOUT = 2.0  # seconds of silence before stopping
MAX_RECORDING_DURATION = 30.0  # maximum recording length
VAD_WINDOW_MS = 30  # voice activity detection window
# How long to sample ambient noise before recording actually starts, to
# adapt the threshold to a genuinely noisy room or a hot mic gain. Set to 0
# to disable and use the sensitivity-derived threshold as-is.
CALIBRATION_SECONDS = 0.3


def _sensitivity_to_threshold(sensitivity: object) -> float:
    """Map a 0..1 mic_sensitivity setting onto a VAD RMS threshold.

    Defensive about ``sensitivity`` not being a plain float (e.g. an
    unconfigured test mock) — falls back to the original fixed threshold
    rather than raising out of a constructor.
    """
    try:
        s = max(0.0, min(1.0, float(sensitivity)))
    except (TypeError, ValueError):
        return SILENCE_THRESHOLD
    return MAX_SILENCE_THRESHOLD + s * (MIN_SILENCE_THRESHOLD - MAX_SILENCE_THRESHOLD)


class WhisperEngine:
    """ASR engine using faster-whisper for local speech-to-text."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._model_name = settings.asr_model
        self._language = settings.asr_language or "en"
        self._mic_index = settings.microphone_index
        self._mic_name = getattr(settings, "microphone_name", "") or ""
        self._base_threshold = _sensitivity_to_threshold(
            getattr(settings, "mic_sensitivity", 0.5)
        )
        self._recording = False
        self._audio_buffer: list[np.ndarray] = []
        self._devices_logged = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        """Load the faster-whisper model (runs in executor)."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed — speech recognition is "
                "unavailable. Install it with: pip install faster-whisper"
            ) from exc

        # INT8 on the GPU (~2.1 GB VRAM on a GTX 1080), then plain int8 on the
        # CPU. Trying only CUDA meant a CPU-only box — or a CUDA box missing
        # cuDNN — got no ASR at all, with an opaque ctranslate2 error.
        attempts = [("cuda", "int8_float16"), ("cpu", "int8")]
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
        self._recording = True
        self._audio_buffer = []

        resolved = resolve_input_device(self._mic_index, self._mic_name)
        if not self._devices_logged:
            log.info("ASR: available input devices:\n%s", describe_devices())
            self._devices_logged = True
        log.info(
            "ASR: starting recording… (device[%s]='%s', reason: %s, "
            "vad_threshold=%.4f)",
            resolved.index, resolved.name, resolved.reason, self._base_threshold,
        )

        chunk_samples = int(SAMPLE_RATE * CHUNK_DURATION)
        silence_samples = int(SAMPLE_RATE * SILENCE_TIMEOUT)
        max_samples = int(SAMPLE_RATE * MAX_RECORDING_DURATION)

        silence_counter = 0
        total_samples = 0
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
                    self._audio_buffer.append(chunk)
                    total_samples += len(chunk)

                    rms = float(np.sqrt(np.mean(chunk**2)))
                    peak_rms = max(peak_rms, rms)

                    # Briefly sample the room's noise floor before applying
                    # VAD, so a noisy fan or a hot mic gain doesn't either
                    # trigger immediately or need a threshold so high it
                    # misses quiet speech.
                    if calib_samples_seen < calib_samples_needed:
                        calib_rms_values.append(rms)
                        calib_samples_seen += len(chunk)
                        if calib_samples_seen >= calib_samples_needed:
                            noise_floor = float(np.mean(calib_rms_values))
                            adaptive = noise_floor * 2.5
                            if adaptive > threshold:
                                log.info(
                                    "ASR: ambient noise floor (%.4f) raises "
                                    "VAD threshold %.4f → %.4f",
                                    noise_floor, threshold, adaptive,
                                )
                                threshold = min(adaptive, MAX_SILENCE_THRESHOLD * 2)
                        continue

                    # Voice activity detection
                    if rms > threshold:
                        has_speech = True
                        silence_counter = 0
                    elif has_speech:
                        silence_counter += len(chunk)

                    # Emit mic level for UI visualization
                    level = min(1.0, rms / threshold)
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
                log.info(
                    "ASR: no speech detected (peak input level %.4f, "
                    "threshold %.4f, device '%s')%s",
                    peak_rms, threshold, resolved.name,
                    " — input was essentially silent; check that this is "
                    "really your microphone and that it isn't muted"
                    if peak_rms < threshold * 0.1 else
                    " — sound was picked up but stayed under the threshold; "
                    "try raising mic_sensitivity in settings",
                )
                return None

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

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription (runs in thread)."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            import soundfile as sf

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
