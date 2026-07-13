"""NixOrb wake word detector — always-on voice activation.

Uses openwakeword for low-CPU wake word detection. When the wake word
is detected, emits WAKE_WORD_DETECTED on the event bus to trigger a
conversation turn.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08  # 80ms chunks for low latency
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
ACTIVATION_THRESHOLD = 0.5
COOLDOWN_SECONDS = 2.0  # minimum time between activations


class WakeWordDetector:
    """Always-on wake word detection using openwakeword."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._enabled = settings.wake_word_enabled
        self._model_name = settings.wake_word_model
        self._sensitivity = settings.wake_word_sensitivity
        self._last_activation = 0.0
        self._running = False

    def _load_model(self):
        """Load the openwakeword model."""
        try:
            from openwakeword.model import Model

            log.info("WakeWord: loading model '%s'", self._model_name)
            model = Model(wakeword_models=[self._model_name])
            log.info("WakeWord: model loaded")
            return model
        except Exception as exc:
            log.error("WakeWord: failed to load model: %s", exc)
            raise

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
            prediction = self._model.predict(audio_chunk)
            if prediction is None:
                return False
            # Get the score for our wake word
            scores = list(prediction.values())
            if not scores:
                return False
            max_score = max(scores)
            return max_score > (ACTIVATION_THRESHOLD * self._sensitivity)
        except Exception:
            return False

    async def run_forever(self) -> None:
        """Main detection loop — runs until cancelled."""
        if not self._enabled:
            log.info("WakeWord: disabled in settings")
            return

        await self.preload()
        self._running = True
        log.info("WakeWord: detection loop started (listening for '%s')", self._model_name)

        try:
            import sounddevice as sd

            chunk_samples = CHUNK_SAMPLES

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_samples,
            ) as stream:
                while self._running:
                    chunk, _ = stream.read(chunk_samples)
                    chunk = chunk.flatten()

                    loop = asyncio.get_running_loop()
                    detected = await loop.run_in_executor(
                        None, self._process_audio_chunk, chunk
                    )

                    if detected:
                        now = time.monotonic()
                        if now - self._last_activation > COOLDOWN_SECONDS:
                            self._last_activation = now
                            log.info("WakeWord: '%s' detected!", self._model_name)
                            bus.emit_sync(
                                Event.WAKE_WORD_DETECTED,
                                data={"model": self._model_name},
                                source="WakeWordDetector",
                            )

                    await asyncio.sleep(0.01)  # small yield

        except asyncio.CancelledError:
            log.info("WakeWord: detection loop cancelled")
        except Exception as exc:
            log.error("WakeWord: detection error: %s", exc)
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the detector to stop."""
        self._running = False
