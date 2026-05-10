"""
nixorb/asr/wake_word.py

OpenWakeWord always-on detector.

BUG FIX PASS 1:
  - Previous version called asyncio.ensure_future() from the sounddevice
    callback thread. ensure_future() requires a running event loop in the
    calling thread, which the audio callback thread does not have.
    Fixed by capturing the running loop at startup and using
    loop.call_soon_threadsafe() with asyncio.run_coroutine_threadsafe().

BUG FIX PASS 2:
  - Cooldown guard added. Without it, a single wake word detection fires
    dozens of events during the ~80 ms chunk window while confidence is high.
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

CHUNK        = 1_280    # ~80 ms at 16 kHz (OpenWakeWord requirement)
CONFIDENCE   = 0.70     # minimum score to fire
COOLDOWN_S   = 2.0      # seconds before another wake-word fires


class WakeWordDetector:
    def __init__(self, settings: Settings) -> None:
        from openwakeword.model import Model
        self._model = Model(
            wakeword_models=[settings.wake_word_model],
            inference_framework="onnx",
        )
        self._settings   = settings
        self._last_fired = 0.0
        self._loop:  asyncio.AbstractEventLoop | None = None

    async def run_forever(self) -> None:
        # BUG FIX: capture loop here, on the async thread, before starting
        # the sounddevice stream whose callbacks run on a C audio thread.
        self._loop = asyncio.get_running_loop()
        log.info(
            "Wake-word detector started (model=%s, threshold=%.2f)",
            self._settings.wake_word_model, CONFIDENCE,
        )

        def _callback(
            indata: np.ndarray, frames: int, time_info, status
        ) -> None:
            if status:
                log.debug("sounddevice status: %s", status)
            pcm = (indata[:, 0] * 32_767).astype(np.int16)
            preds = self._model.predict(pcm)
            for name, score in preds.items():
                if score >= CONFIDENCE:
                    now = time.monotonic()
                    # BUG FIX: cooldown so we don't flood the bus
                    if (now - self._last_fired) < COOLDOWN_S:
                        return
                    self._last_fired = now
                    log.info("Wake word: %s (score=%.3f)", name, score)
                    # BUG FIX: use run_coroutine_threadsafe from audio thread
                    asyncio.run_coroutine_threadsafe(
                        bus.emit(
                            Event.WAKE_WORD_DETECTED,
                            data={"word": name, "score": float(score)},
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
            # Keep the coroutine alive; the callback does the real work.
            while True:
                await asyncio.sleep(0.5)
