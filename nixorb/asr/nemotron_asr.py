"""NVIDIA Nemotron 3.5 ASR — native cache-aware streaming speech-to-text.

    https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b

A 0.6B FastConformer-RNNT that transcribes 40 language-locales from one
model, with punctuation and capitalisation, and — the reason it gets its own
backend rather than going through the generic pipeline — *cache-aware*
streaming. It processes strictly non-overlapping audio chunks while reusing
encoder state, so partial transcripts arrive while you are still talking
instead of after you stop.

Two modes:
  * offline   — record until silence, then transcribe (default, simplest)
  * streaming — feed the microphone in chunks, emit partials as they land

Both need `transformers >= 5.13`, where Nemotron3_5Asr landed. Older
installs fall back to the generic ASR pipeline, which still works for
offline transcription.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb import hf
from nixorb.asr.base import (
    MAX_RECORDING_DURATION,
    MIN_SPEECH_SECONDS,
    SAMPLE_RATE,
    SILENCE_TIMEOUT,
    ASREngine,
    record_with_vad,
)
from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# Nemotron3_5Asr was added to transformers in 5.13. Below that the
# architecture is not registered at all and from_pretrained fails with an
# unrecognised-model-type error that says nothing useful.
MIN_TRANSFORMERS = (5, 13)

# Right-context sizes the checkpoint was trained for, and the streaming
# latency each one buys. Anything else is rejected by the processor.
LOOKAHEAD_LATENCY_MS = {0: 80, 3: 320, 6: 560, 13: 1120}
DEFAULT_LOOKAHEAD = 3

# In auto mode the model appends the detected locale after the terminal
# punctuation, e.g. "Hello there. <en-US>".
LANG_TAG = re.compile(r"\s*<([a-z]{2}(?:-[A-Za-z]{2})?)>\s*$")


def normalise_language(language: str | None) -> str:
    """Map NixOrb's asr_language onto the model's prompt vocabulary.

    Accepts a locale ("en-US"), a bare code ("de"), or blank/"auto" for
    automatic detection.
    """
    value = (language or "").strip()
    if not value or value.lower() == "auto":
        return "auto"
    return value


def resolve_lookahead(tokens: Any) -> int:
    """Clamp the configured right-context to one the model supports."""
    try:
        value = int(tokens)
    except (TypeError, ValueError):
        return DEFAULT_LOOKAHEAD
    if value in LOOKAHEAD_LATENCY_MS:
        return value
    nearest = min(LOOKAHEAD_LATENCY_MS, key=lambda k: abs(k - value))
    log.warning(
        "ASR: lookahead %s is not supported (choose one of %s) — using %d",
        tokens, sorted(LOOKAHEAD_LATENCY_MS), nearest,
    )
    return nearest


def split_language_tag(text: str) -> tuple[str, str | None]:
    """Separate the transcript from the trailing <xx-XX> detection tag."""
    match = LANG_TAG.search(text)
    if not match:
        return text.strip(), None
    return text[: match.start()].strip(), match.group(1)


def chunk_window(mel_frame_idx: int, hop_length: int, window: int,
                 samples_per_chunk: int) -> tuple[int, int]:
    """Sample range feeding the next streaming chunk.

    The half-window back-shift keeps the mel frames continuous across chunk
    boundaries; without it every chunk restarts the STFT and the transcript
    stutters at the seams. `window` is the feature extractor's win_length
    (what the processor itself derives its chunk sizes from), falling back to
    n_fft when a feature extractor does not expose one.
    """
    start = mel_frame_idx * hop_length - window // 2
    return start, start + samples_per_chunk


def feature_window(extractor: Any) -> int:
    """The STFT window the processor uses to size its chunks."""
    for attr in ("win_length", "n_fft"):
        value = getattr(extractor, attr, None)
        if value:
            return int(value)
    return 0


class _LiveAudio:
    """Growing audio buffer fed by the microphone thread.

    Slices block until enough samples have arrived, so the feature generator
    can be written as if it had the whole utterance up front.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._length = 0
        self._closed = False
        self._cond = threading.Condition()

    def append(self, samples: np.ndarray) -> None:
        with self._cond:
            self._chunks.append(np.asarray(samples, dtype=np.float32))
            self._length += len(samples)
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def take(self, start: int, end: int, timeout: float = 10.0) -> np.ndarray | None:
        """Return audio[start:end], waiting for it. None if the stream ended."""
        start = max(0, start)
        with self._cond:
            while self._length < end and not self._closed:
                if not self._cond.wait(timeout=timeout):
                    return None
            if self._length < end:
                return None
            audio = np.concatenate(self._chunks) if self._chunks else np.array([])
        return audio[start:end]


class NemotronASREngine(ASREngine):
    """Cache-aware streaming ASR using NVIDIA Nemotron 3.5."""

    name = "nemotron"
    supports_streaming = True
    whisper_fallback = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model_id = settings.asr_model or DEFAULT_MODEL
        self._language = normalise_language(settings.asr_language)
        self._lookahead = resolve_lookahead(
            getattr(settings, "asr_nemotron_lookahead", DEFAULT_LOOKAHEAD)
        )
        self._detected_language: str | None = None
        self._via_pipeline = False

    @property
    def detected_language(self) -> str | None:
        """Locale reported by the last transcription in `auto` mode."""
        return self._detected_language

    @property
    def latency_ms(self) -> int:
        """Approximate streaming latency for the active right-context."""
        return LOOKAHEAD_LATENCY_MS.get(self._lookahead, 80 * (self._lookahead + 1))

    # ── Loading ──────────────────────────────────────────────────── #

    def _load(self) -> Any:
        transformers = hf.require("transformers")
        device = hf.resolve_device(getattr(self._settings, "hf_device", "auto"))
        kwargs = hf.load_kwargs(self._settings, device)

        too_old = hf.transformers_version() < MIN_TRANSFORMERS
        if too_old or not hasattr(transformers, "AutoModelForRNNT"):
            # Older installs can still transcribe offline through the generic
            # pipeline, so degrade with a clear message rather than refusing
            # to start the assistant.
            want = ".".join(str(n) for n in MIN_TRANSFORMERS)
            log.warning(
                "ASR: Nemotron needs transformers >= %s but %s is installed — "
                "falling back to offline transcription with no streaming. "
                "Upgrade with: pip install -U 'transformers>=%s'",
                want, getattr(transformers, "__version__", "unknown"), want,
            )
            self._via_pipeline = True
            from nixorb.asr.hf_asr_engine import HFASREngine

            delegate = HFASREngine(self._settings)
            delegate._pipe = delegate._load_pipeline()
            return delegate

        log.info(
            "ASR: loading Nemotron %s on %s (lookahead=%d, ~%dms)",
            self._model_id, device, self._lookahead, self.latency_ms,
        )
        processor = transformers.AutoProcessor.from_pretrained(
            self._model_id, **kwargs
        )
        model = transformers.AutoModelForRNNT.from_pretrained(
            self._model_id, device_map="auto" if device == "cuda" else None,
            **kwargs,
        )
        if device != "cuda":
            model = model.to("cpu")

        self._apply_lookahead(processor)

        log.info("ASR: Nemotron ready (language=%s)", self._language)
        return {"model": model, "processor": processor}

    def _apply_lookahead(self, processor: Any) -> None:
        """Set the streaming right-context, honouring what this checkpoint allows."""
        setter = getattr(processor, "set_num_lookahead_tokens", None)
        if not callable(setter):
            return

        supported = list(
            getattr(processor, "supported_num_lookahead_tokens", None) or []
        )
        wanted = self._lookahead
        if supported and wanted not in supported:
            wanted = min(supported, key=lambda k: abs(k - wanted))
            log.warning(
                "ASR: lookahead %d is not supported by %s (supports %s) — using %d",
                self._lookahead, self._model_id, sorted(supported), wanted,
            )

        try:
            setter(wanted)
            self._lookahead = wanted
        except ValueError as exc:
            log.warning(
                "ASR: could not set lookahead %d (%s) — keeping the model default",
                wanted, exc,
            )
            self._lookahead = int(
                getattr(processor, "default_num_lookahead_tokens", DEFAULT_LOOKAHEAD)
            )

        latency = getattr(processor, "streaming_latency_ms", None)
        if latency:
            log.info("ASR: streaming latency ≈ %sms", latency)

    # ── Offline ──────────────────────────────────────────────────── #

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._model is None:
            raise RuntimeError("ASR model not loaded")

        if self._via_pipeline:
            # self._model is an HFASREngine standing in for us.
            return self._model._transcribe_array(
                np.asarray(audio, dtype=np.float32)
            )

        model = self._model["model"]
        processor = self._model["processor"]

        inputs = processor(
            np.asarray(audio, dtype=np.float32),
            sampling_rate=self._sampling_rate(processor),
            language=self._language,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)

        output = model.generate(**inputs, return_dict_in_generate=True)
        # Decode with the tag intact so auto-detection is readable, then strip.
        raw = processor.decode(output.sequences, skip_special_tokens=False)
        text, detected = split_language_tag(_strip_specials(raw))
        self._detected_language = detected
        if detected:
            log.info("ASR: detected language %s", detected)
        return text

    @staticmethod
    def _sampling_rate(processor: Any) -> int:
        extractor = getattr(processor, "feature_extractor", None)
        return int(getattr(extractor, "sampling_rate", SAMPLE_RATE))

    # ── Streaming ────────────────────────────────────────────────── #

    async def stream_transcribe(self) -> AsyncIterator[str]:
        """Yield partial transcripts while the user is still speaking."""
        if self._delegate is None and self._model is None:
            await self.preload()

        # A delegate means loading failed and something else is listening;
        # the base class knows how to hand off to it.
        if self._delegate is not None or self._via_pipeline \
                or not self._streaming_supported():
            async for text in super().stream_transcribe():
                yield text
            return

        loop = asyncio.get_running_loop()
        out: asyncio.Queue[str | None] = asyncio.Queue()
        live = _LiveAudio()

        await bus.emit(Event.RECORDING_START, source=self.name)
        self._recording = True

        mic = threading.Thread(
            target=self._capture, args=(live,), daemon=True, name="nemotron-mic"
        )
        decode = threading.Thread(
            target=self._decode,
            args=(live, loop, out),
            daemon=True,
            name="nemotron-decode",
        )
        mic.start()
        decode.start()

        spoken = False
        try:
            while True:
                piece = await out.get()
                if piece is None:
                    break
                spoken = True
                yield piece
        finally:
            self._recording = False
            live.close()
            await bus.emit(Event.RECORDING_STOP, source=self.name)
            if not spoken:
                log.info("ASR: streaming produced no speech")

    def _streaming_supported(self) -> bool:
        if not isinstance(self._model, dict):
            return False
        processor = self._model.get("processor")
        return all(
            hasattr(processor, attr)
            for attr in (
                "num_samples_first_audio_chunk",
                "num_mel_frames_first_audio_chunk",
                "num_samples_per_audio_chunk",
                "num_mel_frames_per_audio_chunk",
            )
        )

    def _capture(self, live: _LiveAudio) -> None:
        """Push microphone blocks into the live buffer until silence."""
        try:
            import sounddevice as sd
        except Exception as exc:
            log.error("ASR: cannot open microphone: %s", exc)
            live.close()
            return

        block = max(256, int(SAMPLE_RATE * 0.08))  # one 80ms model frame
        threshold = self._vad_threshold
        silence_needed = int(SAMPLE_RATE * SILENCE_TIMEOUT)
        max_samples = int(SAMPLE_RATE * MAX_RECORDING_DURATION)
        silence_run = 0
        heard = False
        total = 0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
                device=self._mic_index, blocksize=block,
            ) as stream:
                while self._recording and total < max_samples:
                    chunk, _ = stream.read(block)
                    chunk = chunk.flatten()
                    live.append(chunk)
                    total += len(chunk)

                    rms = float(np.sqrt(np.mean(chunk**2)))
                    if rms > threshold:
                        heard = True
                        silence_run = 0
                    elif heard:
                        silence_run += len(chunk)

                    bus.emit_sync(
                        Event.MIC_LEVEL,
                        data={"level": min(1.0, rms / threshold if threshold else 0.0)},
                        source=self.name,
                    )

                    if heard and silence_run >= silence_needed:
                        break
        except Exception as exc:
            log.error("ASR: streaming capture failed: %s", exc)
        finally:
            live.close()

    def _decode(
        self,
        live: _LiveAudio,
        loop: asyncio.AbstractEventLoop,
        out: asyncio.Queue,
    ) -> None:
        """Run cache-aware generation, forwarding partials to the loop."""
        try:
            import transformers
        except ImportError:
            loop.call_soon_threadsafe(out.put_nowait, None)
            return

        model = self._model["model"]
        processor = self._model["processor"]
        sr = self._sampling_rate(processor)
        extractor = processor.feature_extractor
        hop = int(extractor.hop_length)
        window = feature_window(extractor)

        first = live.take(0, int(processor.num_samples_first_audio_chunk))
        if first is None:
            loop.call_soon_threadsafe(out.put_nowait, None)
            return

        first_inputs = processor(
            first, sampling_rate=sr, is_streaming=True,
            is_first_audio_chunk=True, language=self._language,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)

        def features():
            yield first_inputs.input_features[
                :, : processor.num_mel_frames_first_audio_chunk, :
            ]
            mel_idx = int(processor.num_mel_frames_first_audio_chunk)
            per_chunk = int(processor.num_samples_per_audio_chunk)

            while True:
                start, end = chunk_window(mel_idx, hop, window, per_chunk)
                block = live.take(start, end)
                if block is None:
                    return
                chunk_inputs = processor(
                    block, sampling_rate=sr, is_streaming=True,
                    is_first_audio_chunk=False, language=self._language,
                    return_tensors="pt",
                ).to(model.device, dtype=model.dtype)
                yield chunk_inputs.input_features
                mel_idx += int(processor.num_mel_frames_per_audio_chunk)

        streamer = transformers.TextIteratorStreamer(
            processor.tokenizer, skip_special_tokens=True
        )
        worker = threading.Thread(
            target=self._generate_safely,
            args=(model, {**first_inputs, "input_features": features(),
                          "streamer": streamer}),
            daemon=True,
            name="nemotron-generate",
        )
        worker.start()

        try:
            for piece in streamer:
                if piece:
                    loop.call_soon_threadsafe(out.put_nowait, piece)
        except Exception as exc:
            log.error("ASR: streaming decode failed: %s", exc)
        finally:
            worker.join(timeout=5.0)
            loop.call_soon_threadsafe(out.put_nowait, None)

    @staticmethod
    def _generate_safely(model: Any, kwargs: dict[str, Any]) -> None:
        try:
            model.generate(**kwargs)
        except Exception as exc:
            log.error("ASR: Nemotron generation failed: %s", exc)


def _strip_specials(text: str) -> str:
    """Remove leftover special tokens the tokenizer kept for the lang tag."""
    return re.sub(r"<\|[^|>]*\|>", "", text).strip()


# Re-exported so callers can size a minimum utterance the same way.
__all__ = [
    "DEFAULT_MODEL",
    "LOOKAHEAD_LATENCY_MS",
    "MIN_SPEECH_SECONDS",
    "NemotronASREngine",
    "chunk_window",
    "feature_window",
    "normalise_language",
    "record_with_vad",
    "resolve_lookahead",
    "split_language_tag",
]
