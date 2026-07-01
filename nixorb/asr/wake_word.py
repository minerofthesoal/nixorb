"""
nixorb/asr/wake_word.py — OpenWakeWord always-on wake word detector.

Compatible with openwakeword >= 0.4.0 (the version available via pip).
The API changed between versions; this code works with both.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

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
        self._model: Any | None = None
        self._muted:  bool = False

    async def _on_mic_muted(self, payload) -> None:
        self._muted = bool((payload.data or {}).get("muted", False))

    def _load_model(self) -> Any | None:
        """Load OWW model — handles API differences between versions.

        The pip-installed openwakeword package ships *no* model weights —
        they must be downloaded separately (via ``scripts/download_models.sh``
        or ``nixorb download-models``). If they're missing, download them
        here automatically instead of failing silently: NixOrb usually runs
        backgrounded with no visible console, so a python-logger-only error
        is invisible to the user and looks exactly like "nothing happens".
        """
        model_name = self._settings.wake_word_model
        try:
            from openwakeword.model import Model  # type: ignore[import]
        except ImportError as exc:
            self._log_bus_error(f"openwakeword is not installed ({exc}); wake word disabled")
            return None

        for attempt in (1, 2):
            try:
                try:
                    # OWW >= 0.5 API
                    m = Model(wakeword_models=[model_name], inference_framework="onnx")
                except TypeError:
                    # OWW 0.4.x API — positional args
                    m = Model(model_name, inference_framework="onnx")  # type: ignore[call-arg]
                log.info("WakeWord: loaded model '%s'", model_name)
                return m
            except Exception as exc:
                if attempt == 1 and self._try_download_model(model_name):
                    continue  # retry now that the weights should exist
                self._log_bus_error(f"failed to load wake-word model '{model_name}': {exc}")
                return None
        return None

    def _try_download_model(self, model_name: str) -> bool:
        """Attempt to fetch the missing wake-word model weights.
        Returns True if the download appears to have succeeded."""
        try:
            import openwakeword.utils as oww_utils
            log.info("WakeWord: model '%s' missing — downloading…", model_name)
            oww_utils.download_models(model_names=[model_name])
            log.info("WakeWord: download of '%s' complete", model_name)
            return True
        except Exception as exc:
            log.error("WakeWord: automatic download failed: %s", exc)
            self._log_bus_error(
                f"could not auto-download wake-word model '{model_name}': {exc}. "
                "Run: nixorb download-models --wake-only"
            )
            return False

    def _log_bus_error(self, msg: str) -> None:
        log.error("WakeWord: %s", msg)
        loop = self._loop
        with contextlib.suppress(Exception):
            if loop is None:
                loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(
                bus.emit(
                    Event.LOG,
                    data={"level": "error", "msg": f"❌ Wake word: {msg}"},
                    source="WakeWordDetector",
                ),
                loop,
            )

    async def run_forever(self) -> None:
        self._loop  = asyncio.get_running_loop()
        bus.subscribe(Event.MIC_MUTED, self._on_mic_muted)
        self._model = await self._loop.run_in_executor(None, self._load_model)
        model = self._model
        loop = self._loop
        if model is None:
            log.error("WakeWord detector disabled — model not loaded")
            await bus.emit(
                Event.LOG,
                data={
                    "level": "warning",
                    "msg": "⚠ Wake word disabled — model failed to load "
                           "(see log above, or run: nixorb download-models --wake-only)",
                },
                source="WakeWordDetector",
            )
            return

        log.info(
            "WakeWord detector running (model=%s, threshold=%.2f)",
            self._settings.wake_word_model, CONFIDENCE,
        )

        def _callback(indata: np.ndarray, frames: int, t, status) -> None:
            if status:
                log.debug("sounddevice: %s", status)
            if self._muted:
                return
            pcm = (indata[:, 0] * 32_767).astype(np.int16)

            try:
                preds = model.predict(pcm)
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
                        loop,
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
