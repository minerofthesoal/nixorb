"""NixOrb wake word detector — always-on voice activation.

Uses openwakeword for low-CPU wake word detection. When the wake word
is detected, emits WAKE_WORD_DETECTED on the event bus to trigger a
conversation turn.

Three independent bugs used to make this permanently silent, even once
openwakeword was installed:

  1. ``Model(wakeword_models=[...])`` — current openwakeword renamed that
     constructor argument to ``wakeword_model_paths``, and it takes file
     paths, not bare names. The old kwarg fell through to ``**kwargs`` and
     was forwarded into the internal ``AudioFeatures`` preprocessor, which
     doesn't accept it — hence the ``AudioFeatures.__init__() got an
     unexpected keyword argument 'wakeword_models'`` crash.
  2. openwakeword's feature extractor requires 16-bit PCM *integer* samples
     (roughly -32768..32767). This module captured float32 samples
     normalised to [-1.0, 1.0] straight from ``sounddevice`` and handed them
     to the model as-is. Internally it casts that buffer with
     ``.astype(np.int16)``, which truncates every sample to 0 — the model
     would have been listening to true digital silence regardless of what
     the microphone picked up.
  3. "hey_nixorb" was never a real model. openwakeword ships a handful of
     pretrained wake words (alexa, hey_jarvis, hey_mycroft, timer, weather);
     anything else has to be a path to a model you trained yourself. With
     no such file, detection silently did nothing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nixorb.core.event_bus import Event, bus
from nixorb.utils.audio import describe_devices, resolve_input_device

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08  # 80ms chunks for low latency
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
# openwakeword's own docs recommend 0.5 as the baseline score threshold —
# used here as the effective threshold at the *default* sensitivity (0.5),
# not as a fixed value the sensitivity setting gets multiplied against.
ACTIVATION_THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0  # minimum time between activations
# Wake words openwakeword ships pretrained models for out of the box.
BUNDLED_WAKE_WORDS = ("alexa", "hey_jarvis", "hey_mycroft", "timer", "weather")
FALLBACK_WAKE_WORD = "hey_jarvis"


def _sensitivity_to_threshold(sensitivity: object) -> float:
    """Map 0..1 sensitivity onto a score threshold, higher sensitivity = lower bar.

    The previous implementation computed ``ACTIVATION_THRESHOLD *
    sensitivity``, which moved the wrong direction: turning sensitivity
    *down* made the detector fire on *less* signal, and the default
    (0.5) landed at half of openwakeword's own recommended threshold.
    """
    try:
        s = max(0.0, min(1.0, float(sensitivity)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        s = 0.5
    # s=0 → 1.5x threshold (needs a clear, confident match)
    # s=0.5 → 1.0x threshold (openwakeword's recommended default)
    # s=1 → 0.5x threshold (fires on weaker/quieter matches)
    return ACTIVATION_THRESHOLD * (1.5 - s)


def resolve_wake_word_model(model_name: str) -> tuple[str | None, str]:
    """Resolve a configured wake-word model to a loadable file path.

    Returns ``(path_or_None, resolved_label)``. ``path`` is ``None`` only
    when nothing usable could be found at all (openwakeword not installed
    correctly / no bundled resources).
    """
    import openwakeword

    # A real, existing file — a custom-trained .onnx/.tflite model.
    candidate = Path(model_name).expanduser()
    if candidate.is_file():
        return str(candidate), model_name

    # A bundled pretrained name.
    if model_name in openwakeword.models:
        return openwakeword.models[model_name]["model_path"], model_name

    # Anything else (including the "hey_nixorb" default, which nobody has
    # trained) falls back to a real bundled model so wake word actually
    # works out of the box, with a loud explanation of how to fix it
    # properly.
    log.warning(
        "WakeWord: '%s' is not one of openwakeword's bundled models (%s) "
        "and is not a path to a file that exists — falling back to '%s'. "
        "To use your own wake word, train a custom model with "
        "openwakeword's training tools and set wake_word_model to the "
        "resulting .onnx/.tflite path in settings.",
        model_name, ", ".join(BUNDLED_WAKE_WORDS), FALLBACK_WAKE_WORD,
    )
    if FALLBACK_WAKE_WORD in openwakeword.models:
        return openwakeword.models[FALLBACK_WAKE_WORD]["model_path"], FALLBACK_WAKE_WORD
    return None, model_name


class WakeWordDetector:
    """Always-on wake word detection using openwakeword."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._enabled = settings.wake_word_enabled
        self._model_name = settings.wake_word_model
        self._resolved_label = self._model_name
        self._threshold = _sensitivity_to_threshold(settings.wake_word_sensitivity)
        self._mic_index = getattr(settings, "microphone_index", None)
        self._mic_name = getattr(settings, "microphone_name", "") or ""
        self._last_activation = 0.0
        self._running = False

    def _load_model(self) -> Any:
        """Load the openwakeword model."""
        try:
            import openwakeword
            from openwakeword.model import Model
        except ImportError:
            # openwakeword is the optional "wakeword" extra, not a base
            # dependency — a missing import is a config problem, not a crash.
            log.warning(
                "WakeWord: openwakeword is not installed — wake word disabled. "
                "Install it with: pip install 'nixorb[wakeword]'"
            )
            return None

        # Older openwakeword releases ship without any bundled model files
        # and require an explicit one-time download; newer ones (and the
        # version this was tested against) bundle them in the package. Try
        # the download hook if one exists, but don't treat its absence — or
        # failure, e.g. no network — as fatal, since the files may already
        # be present.
        downloader = getattr(getattr(openwakeword, "utils", None), "download_models", None)
        if callable(downloader):
            try:
                downloader()
            except Exception as exc:
                log.debug("WakeWord: model download step skipped: %s", exc)

        model_paths: list[str] = []
        labels: list[str] = []
        for raw_name in self._model_name.split(","):
            raw_name = raw_name.strip()
            if not raw_name:
                continue
            model_path, label = resolve_wake_word_model(raw_name)
            if model_path is not None:
                model_paths.append(model_path)
                labels.append(label)
        self._resolved_label = ", ".join(labels) if labels else self._model_name
        if not model_paths:
            log.error(
                "WakeWord: no usable model found (not '%s', and the bundled "
                "fallback is unavailable) — wake word disabled", self._model_name
            )
            return None

        try:
            log.info(
                "WakeWord: loading model(s) '%s' (threshold=%.2f)",
                self._resolved_label, self._threshold,
            )
            model = Model(wakeword_model_paths=model_paths)
            log.info("WakeWord: model(s) loaded")
            return model
        except Exception as exc:
            log.error("WakeWord: failed to load model(s) '%s': %s", self._resolved_label, exc)
            return None

    def _unload_model(self, model) -> None:
        """Unload the wake word model."""
        del model
        import gc

        gc.collect()
        log.info("WakeWord: model unloaded")

    async def preload(self) -> None:
        """Preload the wake word model."""
        if self._model is not None or not self._enabled:
            return
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)

    async def unload(self) -> None:
        """Unload the wake word model."""
        if self._model is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._unload_model, self._model)
            self._model = None

    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process a single audio chunk, return True if wake word detected."""
        if self._model is None:
            return False
        try:
            # openwakeword's feature extractor requires 16-bit PCM integer
            # samples. sounddevice gives us float32 in [-1.0, 1.0]; feeding
            # that straight in gets truncated to all-zero int16 internally
            # (see module docstring, bug #2) and the model never fires.
            if np.issubdtype(audio_chunk.dtype, np.floating):
                audio_chunk = np.clip(audio_chunk * 32767.0, -32768, 32767).astype(np.int16)
            prediction = self._model.predict(audio_chunk)
            if prediction is None:
                return False
            # Get the score for our wake word
            scores = list(prediction.values())
            if not scores:
                return False
            max_score = max(scores)
            return max_score > self._threshold
        except Exception:
            log.exception("WakeWord: prediction failed")
            return False

    def _detect_loop(self) -> None:
        """Blocking capture + detection loop. Runs in a worker thread.

        This must not run on the event loop: sd.InputStream.read() blocks for
        a full chunk (~80 ms) every iteration, which stalls Qt rendering, the
        event bus and every other coroutine for as long as NixOrb is running.
        """
        import sounddevice as sd

        resolved = resolve_input_device(self._mic_index, self._mic_name)
        log.info(
            "WakeWord: listening for '%s' on device[%s]='%s' (%s)",
            self._resolved_label, resolved.index, resolved.name, resolved.reason,
        )
        if resolved.index is None and self._mic_index is not None:
            log.info("WakeWord: input devices:\n%s", describe_devices())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            device=resolved.index,
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            while self._running:
                chunk, _ = stream.read(CHUNK_SAMPLES)
                if not self._process_audio_chunk(chunk.flatten()):
                    continue

                now = time.monotonic()
                if now - self._last_activation > COOLDOWN_SECONDS:
                    self._last_activation = now
                    log.info("WakeWord: '%s' detected!", self._resolved_label)
                    # emit_sync marshals back onto the loop thread for us.
                    bus.emit_sync(
                        Event.WAKE_WORD_DETECTED,
                        data={"model": self._model_name},
                        source="WakeWordDetector",
                    )

    async def run_forever(self) -> None:
        """Main detection loop — runs until cancelled. Never raises."""
        if not self._enabled:
            log.info("WakeWord: disabled in settings")
            return

        try:
            await self.preload()
        except Exception as exc:
            log.error("WakeWord: preload failed, wake word disabled: %s", exc)
            return

        if self._model is None:
            # _load_model already explained why.
            return

        self._running = True
        log.info(
            "WakeWord: detection loop started (listening for '%s')",
            self._resolved_label,
        )

        try:
            await asyncio.to_thread(self._detect_loop)
        except asyncio.CancelledError:
            log.info("WakeWord: detection loop cancelled")
            raise
        except Exception as exc:
            log.error("WakeWord: detection error — wake word disabled: %s", exc)
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the detector to stop."""
        self._running = False
