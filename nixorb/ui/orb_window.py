"""NixOrb floating orb window — frameless Qt6/QML orb.

A glowing, always-on-top orb that animates based on AI state:
  idle (blue) → listening (green) → thinking (amber) → speaking (purple)

Features:
- Drag to reposition
- Scroll wheel to adjust opacity
- Double-click to activate
- Right-click for context menu
"""
from __future__ import annotations

import logging

from PySide6.QtCore import (
    QObject,
    QPointF,
    QSize,
    Qt,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QWheelEvent
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QApplication

from nixorb.core.event_bus import Event, EventPayload, bus
from nixorb.utils.paths import asset_path

log = logging.getLogger(__name__)

QML_IMPORT_NAME = "NixOrb"
QML_IMPORT_MAJOR_VERSION = 1

# State → color mapping
STATE_COLORS: dict[str, str] = {
    "idle": "#4A90D9",
    "listening": "#2ECC71",
    "thinking": "#F39C12",
    "speaking": "#9B59B6",
    "error": "#E74C3C",
}


@QmlElement
class OrbBridge(QObject):
    """Bridge object exposed to QML for state/control communication."""

    stateChanged = Signal(str)
    amplitudeChanged = Signal(float)
    colorChanged = Signal(str)
    opacityChanged = Signal(float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = "idle"
        self._amplitude = 0.0
        self._color = STATE_COLORS["idle"]
        self._opacity = 1.0

    @Property(str, notify=stateChanged)
    def state(self) -> str:
        return self._state

    @Property(float, notify=amplitudeChanged)
    def amplitude(self) -> float:
        return self._amplitude

    @Property(str, notify=colorChanged)
    def color(self) -> str:
        return self._color

    @Property(float, notify=opacityChanged)
    def opacity(self) -> float:
        return self._opacity

    @Slot(str)
    def setState(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        self._color = STATE_COLORS.get(state, "#FFFFFF")
        self.stateChanged.emit(state)
        self.colorChanged.emit(self._color)

    @Slot(float)
    def setAmplitude(self, amp: float) -> None:
        clamped = max(0.0, min(1.0, float(amp)))
        if abs(clamped - self._amplitude) > 0.004:
            self._amplitude = clamped
            self.amplitudeChanged.emit(clamped)

    @Slot(float)
    def setOpacity(self, opacity: float) -> None:
        self._opacity = max(0.2, min(1.0, opacity))
        self.opacityChanged.emit(self._opacity)

    @Slot()
    def clicked(self) -> None:
        bus.emit_sync(Event.ORB_CLICKED, source="orb_window")

    @Slot()
    def openSettings(self) -> None:
        from nixorb.ui.settings_window import SettingsWindow

        SettingsWindow.show_singleton()


class OrbWindow(QQuickView):
    """The main floating orb window."""

    def __init__(self, settings, app: QApplication) -> None:
        super().__init__()
        self._settings = settings
        self._drag_pos: QPointF | None = None
        self._bridge = OrbBridge()
        self._target_amp = 0.0
        self._current_amp = 0.0

        self._setup_window()
        self._setup_qml()
        self._subscribe_events()
        self._start_amp_smoother()

    def _setup_window(self) -> None:
        """Configure window flags and geometry."""
        self.setFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setColor(QColor(0, 0, 0, 0))

        size = self._settings.orb_size
        self.resize(QSize(size, size))

        # Position: top-right corner as default
        screen = QApplication.primaryScreen()
        if screen:
            sw = screen.geometry().width()
            sh = screen.geometry().height()
        else:
            sw, sh = 1920, 1080

        x = self._settings.orb_x if self._settings.orb_x is not None else sw - size - 40
        y = self._settings.orb_y if self._settings.orb_y is not None else 40

        # Clamp to screen bounds
        x = max(0, min(x, sw - size))
        y = max(0, min(y, sh - size))

        platform = QGuiApplication.platformName()
        if platform == "wayland":
            log.info("Orb: Wayland session — compositor controls placement")
        else:
            self.setPosition(x, y)

        # Set opacity from settings
        self._bridge.setOpacity(self._settings.orb_opacity)

    def _setup_qml(self) -> None:
        """Load the QML orb interface."""
        self.rootContext().setContextProperty("orbBridge", self._bridge)

        qml_path = asset_path("orb.qml")
        if not qml_path.exists():
            log.error("Orb: QML file not found at %s", qml_path)
            return

        self.setSource(QUrl.fromLocalFile(str(qml_path.resolve())))

        if self.status() == QQuickView.Status.Error:
            for err in self.errors():
                log.error("Orb QML error: %s", err.toString())

    def _subscribe_events(self) -> None:
        """Subscribe to event bus for state changes."""
        event_state_map = {
            Event.ORB_IDLE: "idle",
            Event.ORB_LISTENING: "listening",
            Event.ORB_THINKING: "thinking",
            Event.ORB_SPEAKING: "speaking",
            Event.ORB_ERROR: "error",
        }

        for evt, state in event_state_map.items():
            handler = self._make_state_handler(state)
            bus.subscribe(evt, handler)

        bus.subscribe(Event.TTS_AUDIO_CHUNK, self._on_audio_chunk)
        bus.subscribe(Event.MIC_LEVEL, self._on_mic_level)
        bus.subscribe(Event.RECORDING_STOP, self._on_recording_stop)

    def _make_state_handler(self, state: str):
        """Create a handler that sets the orb to a specific state."""
        async def handler(_payload: EventPayload) -> None:
            QTimer.singleShot(0, lambda s=state: self._bridge.setState(s))
        return handler

    async def _on_audio_chunk(self, payload: EventPayload) -> None:
        """Update amplitude from TTS audio data."""
        data = payload.data or {}
        pcm = data.get("pcm")
        if pcm:
            import numpy as np
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            self._target_amp = float(np.sqrt(np.mean(arr ** 2)) / 32768.0)

    async def _on_mic_level(self, payload: EventPayload) -> None:
        """Update amplitude from microphone level."""
        data = payload.data or {}
        self._target_amp = float(data.get("level", 0.0))

    async def _on_recording_stop(self, _payload: EventPayload) -> None:
        """Reset amplitude when recording stops."""
        self._target_amp = 0.0

    def _start_amp_smoother(self) -> None:
        """Start the amplitude smoothing timer (60fps)."""
        self._amp_timer = QTimer(self)
        self._amp_timer.setInterval(16)  # ~60fps
        self._amp_timer.timeout.connect(self._tick_amplitude)
        self._amp_timer.start()

    def _tick_amplitude(self) -> None:
        """Smooth amplitude transitions."""
        alpha = 0.35 if self._target_amp > self._current_amp else 0.08
        self._current_amp += alpha * (self._target_amp - self._current_amp)
        self._bridge.setAmplitude(self._current_amp)

    # ── Mouse interaction ────────────────────────────────────────── #

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition() - QPointF(self.x(), self.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            new_pos = event.globalPosition() - self._drag_pos
            self.setPosition(int(new_pos.x()), int(new_pos.y()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self._drag_pos = None
            # Save position
            self._settings.orb_x = self.x()
            self._settings.orb_y = self.y()
            self._settings.save()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            log.info("Orb: double-clicked — triggering activation")
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_double_click")

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Scroll wheel adjusts orb opacity."""
        delta = event.angleDelta().y() / 1200.0
        new_opacity = self._bridge.opacity + delta
        self._bridge.setOpacity(new_opacity)
        self._settings.orb_opacity = self._bridge.opacity
        self._settings.save()

    def log_visibility(self) -> None:
        """Log the visibility state for debugging."""
        log.info(
            "Orb: visible=%s exposed=%s platform=%s geometry=%s",
            self.isVisible(),
            self.isExposed(),
            QGuiApplication.platformName(),
            self.geometry(),
        )
