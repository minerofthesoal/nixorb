"""
nixorb/ui/orb_window.py

Frameless floating orb window (PySide6 QQuickView + QML).

BUG FIX PASS 1:
  - setSource() was passed a plain string. QQuickView.setSource() requires
    a QUrl; passing a str raises TypeError in PySide6 6.7+. Fixed with
    QUrl.fromLocalFile(str(resolved_path)).

BUG FIX PASS 2:
  - OrbBridge.clicked() called bus.emit_sync() which uses the stored
    event-loop reference. If the orb window is shown before bus.start()
    is awaited, that reference is None and emit_sync silently drops the
    event. Moved bus.start() before QQuickView creation in main.py.
    Added a None-check guard here as a safety net.

BUG FIX PASS 3:
  - QTimer.singleShot used a lambda capturing loop variables that could
    be garbage-collected before the timer fired. Changed to explicit
    method calls with stable argument binding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Property, QObject, QPointF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickView
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from nixorb.core.event_bus import Event, EventPayload, bus

log = logging.getLogger(__name__)

# Resolved at import time so the QUrl is always absolute
_QML_PATH = Path(__file__).parents[2] / "assets" / "orb.qml"

QML_IMPORT_NAME    = "NixOrb"
QML_IMPORT_VERSION = "1.0"


# ------------------------------------------------------------------ #
#  Python → QML bridge                                                #
# ------------------------------------------------------------------ #
@QmlElement
class OrbBridge(QObject):
    """Exposes orb state and audio amplitude to QML via Q_PROPERTY."""

    stateChanged     = Signal(str)
    amplitudeChanged = Signal(float)
    colorChanged     = Signal(str)

    _STATE_COLORS: dict[str, str] = {
        "idle":      "#4A90D9",
        "listening": "#2ECC71",
        "thinking":  "#F39C12",
        "speaking":  "#9B59B6",
        "error":     "#E74C3C",
    }

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state     = "idle"
        self._amplitude = 0.0
        self._color     = self._STATE_COLORS["idle"]

    @Property(str, notify=stateChanged)
    def state(self) -> str:
        return self._state

    @Property(float, notify=amplitudeChanged)
    def amplitude(self) -> float:
        return self._amplitude

    @Property(str, notify=colorChanged)
    def color(self) -> str:
        return self._color

    @Slot(str)
    def setState(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._color = self._STATE_COLORS.get(state, "#FFFFFF")
        self.stateChanged.emit(state)
        self.colorChanged.emit(self._color)

    @Slot(float)
    def setAmplitude(self, amp: float) -> None:
        clamped = max(0.0, min(1.0, float(amp)))
        if abs(clamped - self._amplitude) > 0.004:
            self._amplitude = clamped
            self.amplitudeChanged.emit(clamped)

    @Slot()
    def clicked(self) -> None:
        # BUG FIX: guard against emit_sync before bus is started
        if bus._loop is not None:
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_click")
        else:
            log.warning("OrbBridge.clicked() — event loop not ready yet")

    @Slot()
    def openSettings(self) -> None:
        from nixorb.ui.settings_window import SettingsWindow
        SettingsWindow.show_singleton()


# ------------------------------------------------------------------ #
#  Floating orb window                                                #
# ------------------------------------------------------------------ #
class OrbWindow(QQuickView):
    def __init__(self, settings, app: QApplication) -> None:
        super().__init__()
        self._settings        = settings
        self._app             = app
        self._drag_pos:       Optional[QPointF] = None
        self._bridge          = OrbBridge()
        self._target_amp      = 0.0
        self._current_amp     = 0.0

        self._setup_window()
        self._setup_qml()
        self._subscribe()
        self._start_amp_smoother()

    # ---------------------------------------------------------------- #
    #  Window setup                                                     #
    # ---------------------------------------------------------------- #
    def _setup_window(self) -> None:
        self.setFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setColor(QColor(0, 0, 0, 0))
        size = self._settings.orb_size
        self.resize(QSize(size, size))

        screen    = QApplication.primaryScreen()
        screen_w  = screen.geometry().width() if screen else 1920
        x = self._settings.orb_x if self._settings.orb_x is not None else screen_w - size - 40
        y = self._settings.orb_y if self._settings.orb_y is not None else 40
        self.setPosition(x, y)

    def _setup_qml(self) -> None:
        self.rootContext().setContextProperty("orbBridge", self._bridge)
        # BUG FIX: must use QUrl, not a plain string
        url = QUrl.fromLocalFile(str(_QML_PATH.resolve()))
        self.setSource(url)
        if self.status() == QQuickView.Status.Error:
            for err in self.errors():
                log.error("QML error: %s", err.toString())
            raise RuntimeError(f"Failed to load {_QML_PATH}")

    # ---------------------------------------------------------------- #
    #  EventBus subscriptions                                           #
    # ---------------------------------------------------------------- #
    def _subscribe(self) -> None:
        for evt in (Event.ORB_IDLE, Event.ORB_LISTENING,
                    Event.ORB_THINKING, Event.ORB_SPEAKING, Event.ORB_ERROR):
            bus.subscribe(evt, self._on_orb_state)
        bus.subscribe(Event.TTS_AUDIO_CHUNK, self._on_audio_chunk)

    _EVENT_TO_STATE: dict[Event, str] = {
        Event.ORB_IDLE:      "idle",
        Event.ORB_LISTENING: "listening",
        Event.ORB_THINKING:  "thinking",
        Event.ORB_SPEAKING:  "speaking",
        Event.ORB_ERROR:     "error",
    }

    async def _on_orb_state(self, payload: EventPayload) -> None:
        state = self._EVENT_TO_STATE.get(payload.event, "idle")
        # BUG FIX: store state in a local var so the lambda captures
        # the value, not a mutable reference.
        _state = state
        QTimer.singleShot(0, lambda s=_state: self._bridge.setState(s))

    async def _on_audio_chunk(self, payload: EventPayload) -> None:
        import numpy as np
        data = payload.data or {}
        pcm  = data.get("pcm")
        if pcm:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            self._target_amp = float(np.sqrt(np.mean(arr ** 2)) / 32_768.0)

    # ---------------------------------------------------------------- #
    #  Amplitude smoother (Qt timer, ~60 FPS)                          #
    # ---------------------------------------------------------------- #
    def _start_amp_smoother(self) -> None:
        self._amp_timer = QTimer(self)
        self._amp_timer.setInterval(16)
        self._amp_timer.timeout.connect(self._tick_amplitude)
        self._amp_timer.start()

    def _tick_amplitude(self) -> None:
        # Fast attack, slow release exponential smoothing
        alpha = 0.35 if self._target_amp > self._current_amp else 0.08
        self._current_amp += alpha * (self._target_amp - self._current_amp)
        self._bridge.setAmplitude(self._current_amp)

    # ---------------------------------------------------------------- #
    #  Drag to reposition                                               #
    # ---------------------------------------------------------------- #
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition() - QPointF(self.x(), self.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            new = event.globalPosition() - self._drag_pos
            self.setPosition(int(new.x()), int(new.y()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._drag_pos = None
            self._settings.orb_x = self.x()
            self._settings.orb_y = self.y()
            self._settings.save()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_double_click")

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Scroll wheel changes orb opacity (not size, to avoid layout thrash)
        delta  = event.angleDelta().y()
        effect = self.rootObject()
        if effect:
            opacity = getattr(effect, "opacity", 1.0)
            new_op  = max(0.2, min(1.0, opacity + delta / 1200.0))
            effect.setProperty("opacity", new_op)
