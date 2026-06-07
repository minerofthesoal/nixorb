"""nixorb/ui/orb_window.py — Frameless floating orb window."""
from __future__ import annotations

import logging

from PySide6.QtCore import (
    Property,
    QObject,
    QPointF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QMouseEvent, QWheelEvent
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QApplication

from nixorb.core.event_bus import Event, EventPayload, bus
from nixorb.utils.paths import asset_path

log = logging.getLogger(__name__)

_QML_PATH = asset_path("orb.qml")

QML_IMPORT_NAME    = "NixOrb"
QML_IMPORT_VERSION = "1.0"


@QmlElement
class OrbBridge(QObject):
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
        if bus._loop is not None:  # noqa: SLF001
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_click")

    @Slot()
    def openSettings(self) -> None:
        from nixorb.ui.settings_window import SettingsWindow
        SettingsWindow.show_singleton()


class OrbWindow(QQuickView):
    def __init__(self, settings, app: QApplication) -> None:
        super().__init__()
        self._settings    = settings
        self._drag_pos:   QPointF | None = None
        self._bridge      = OrbBridge()
        self._target_amp  = 0.0
        self._current_amp = 0.0

        self._setup_window()
        self._setup_qml()
        self._subscribe()
        self._start_amp_smoother()

    def _setup_window(self) -> None:
        self.setFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setColor(QColor(0, 0, 0, 0))
        size = self._settings.orb_size
        self.resize(QSize(size, size))
        screen = QApplication.primaryScreen()
        sw     = screen.geometry().width() if screen else 1920
        x = self._settings.orb_x if self._settings.orb_x is not None else sw - size - 40
        y = self._settings.orb_y if self._settings.orb_y is not None else 40
        self.setPosition(x, y)

    def _setup_qml(self) -> None:
        self.rootContext().setContextProperty("orbBridge", self._bridge)
        url = QUrl.fromLocalFile(str(_QML_PATH.resolve()))
        self.setSource(url)
        if self.status() == QQuickView.Status.Error:
            for err in self.errors():
                log.error("QML error: %s", err.toString())
            log.warning("Orb QML failed to load from %s", _QML_PATH)

    def _subscribe(self) -> None:
        for evt in (Event.ORB_IDLE, Event.ORB_LISTENING,
                    Event.ORB_THINKING, Event.ORB_SPEAKING, Event.ORB_ERROR):
            bus.subscribe(evt, self._on_orb_state)
        bus.subscribe(Event.TTS_AUDIO_CHUNK, self._on_audio_chunk)
        bus.subscribe(Event.MIC_LEVEL, self._on_mic_level)
        bus.subscribe(Event.RECORDING_STOP, self._on_recording_stop)

    _EVENT_STATE: dict[Event, str] = {
        Event.ORB_IDLE:      "idle",
        Event.ORB_LISTENING: "listening",
        Event.ORB_THINKING:  "thinking",
        Event.ORB_SPEAKING:  "speaking",
        Event.ORB_ERROR:     "error",
    }

    async def _on_orb_state(self, payload: EventPayload) -> None:
        state = self._EVENT_STATE.get(payload.event, "idle")
        QTimer.singleShot(0, lambda s=state: self._bridge.setState(s))

    async def _on_audio_chunk(self, payload: EventPayload) -> None:
        import numpy as np
        data = payload.data or {}
        pcm  = data.get("pcm")
        if pcm:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            self._target_amp = float(np.sqrt(np.mean(arr ** 2)) / 32_768.0)

    async def _on_mic_level(self, payload: EventPayload) -> None:
        data = payload.data or {}
        self._target_amp = float(data.get("level", 0.0))

    async def _on_recording_stop(self, _payload: EventPayload) -> None:
        self._target_amp = 0.0

    def _start_amp_smoother(self) -> None:
        self._amp_timer = QTimer(self)
        self._amp_timer.setInterval(16)
        self._amp_timer.timeout.connect(self._tick_amplitude)
        self._amp_timer.start()

    def _tick_amplitude(self) -> None:
        alpha = 0.35 if self._target_amp > self._current_amp else 0.08
        self._current_amp += alpha * (self._target_amp - self._current_amp)
        self._bridge.setAmplitude(self._current_amp)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition() - QPointF(self.x(), self.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            new = event.globalPosition() - self._drag_pos
            self.setPosition(int(new.x()), int(new.y()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._drag_pos        = None
            self._settings.orb_x = self.x()
            self._settings.orb_y = self.y()
            self._settings.save()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_double_click")

    def wheelEvent(self, event: QWheelEvent) -> None:
        obj = self.rootObject()
        if obj:
            opacity = obj.property("opacity") or 1.0
            obj.setProperty("opacity", max(0.2, min(1.0, opacity + event.angleDelta().y() / 1200.0)))
