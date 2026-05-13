"""
nixorb/asr/wake_word.py — OpenWakeWord always-on wake word detector.

Compatible with openwakeword >= 0.4.0 (the version available via pip).
The API changed between versions; this code works with both.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

CHUNK      = 1_280   # 80 ms at 16 kHz (OWW requirement)
CONFIDENCE = 0.50    # lowered threshold for better detection
COOLDOWN_S = 2.5     # seconds between detections


class WakeWordDetector:
    def __init__(self, settings: Settings) -> None:
        self._settings    = settings
        self._last_fired  = 0.0
        self._loop:  asyncio.AbstractEventLoop | None = None
        self._model  = None

    def _load_model(self):
        """Load OWW model — handles API differences between versions."""
        try:
            from openwakeword.model import Model  # type: ignore[import]
            model_name = self._settings.wake_word_model
            try:
                # OWW >= 0.5 API
                m = Model(wakeword_models=[model_name], inference_framework="onnx")
            except TypeError:
                # OWW 0.4.x API — positional args
                m = Model(model_name, inference_framework="onnx")  # type: ignore[call-arg]
            log.info("WakeWord: loaded model '%s'", model_name)
            return m
        except Exception as exc:
            log.error("WakeWord: failed to load model: %s", exc)
            log.error(
                "Try: pip install openwakeword  "
                "and run: python -m openwakeword --download_models %s",
                self._settings.wake_word_model,
            )
            return None

    async def run_forever(self) -> None:
        self._loop  = asyncio.get_running_loop()
        self._model = await self._loop.run_in_executor(None, self._load_model)
        if self._model is None:
            log.error("WakeWord detector disabled — model not loaded")
            return

        log.info(
            "WakeWord detector running (model=%s, threshold=%.2f)",
            self._settings.wake_word_model, CONFIDENCE,
        )

        def _callback(indata: np.ndarray, frames: int, t, status) -> None:
            if status:
                log.debug("sounddevice: %s", status)
            pcm = (indata[:, 0] * 32_767).astype(np.int16)

            try:
                preds = self._model.predict(pcm)
            except Exception as exc:
                log.debug("WakeWord predict error: %s", exc)
                return

            # Handle both dict and list return types across versions
            items = preds.items() if isinstance(preds, dict) else enumerate(preds)  # type: ignore[arg-type]

            for name, score in items:
                score_f = float(score)
                if score_f >= CONFIDENCE:
                    now = time.monotonic()
                    if (now - self._last_fired) < COOLDOWN_S:
                        return
                    self._last_fired = now
                    log.info("🎤 Wake word detected: %s (score=%.3f)", name, score_f)
                    asyncio.run_coroutine_threadsafe(
                        bus.emit(
                            Event.WAKE_WORD_DETECTED,
                            data={"word": str(name), "score": score_f},
                            source="WakeWordDetector",
                        ),
                        self._loop,
                    )

        with sd.InputStream(
            samplerate=16_000,
            channels=1,
            dtype="float32",
            blocksize=CHUNK,
            callback=_callback,
        ):
            log.info("WakeWord: audio stream open")
            while True:
                await asyncio.sleep(0.5)
