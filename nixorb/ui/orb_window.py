"""
nixorb/ui/orb_window.py

Frameless, borderless, always-on-top floating orb for KDE Plasma 6 / Wayland.

Architecture:
  - QQuickView hosts the QML particle shader orb (assets/orb.qml)
  - Python bridge exposes OrbBridge (QObject) to QML for state/amplitude
  - PySide6 Qt.FramelessWindowHint + WA_TranslucentBackground = true glass
  - Mouse dragging for repositioning
  - Wayland: uses layer-shell protocol via QWindow setProperty hints
    (requires qt6-wayland and kwin layer-shell support)
"""
from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Optional

from PySide6.QtCore import (
    Property, QObject, QPointF, QSize, Qt, QTimer, Signal, Slot,
)
from PySide6.QtGui import QColor, QMouseEvent, QScreen
from PySide6.QtQml import QmlElement
from PySide6.QtQuick import QQuickView
from PySide6.QtWidgets import QApplication

from nixorb.core.event_bus import Event, EventPayload, bus

log = logging.getLogger(__name__)

ORB_SIZE     = 120          # px
ORB_QML_PATH = "assets/orb.qml"

QML_IMPORT_NAME    = "NixOrb"
QML_IMPORT_VERSION = "1.0"


# ------------------------------------------------------------------ #
#  QML Bridge — exposed to QML as context property                    #
# ------------------------------------------------------------------ #
@QmlElement
class OrbBridge(QObject):
    """
    Exposes orb state and audio amplitude to QML.
    QML properties are updated from Python via signals.
    """

    stateChanged    = Signal(str)
    amplitudeChanged = Signal(float)
    colorChanged    = Signal(str)

    STATE_COLORS = {
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
        self._color     = self.STATE_COLORS["idle"]

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
        if state != self._state:
            self._state = state
            self._color = self.STATE_COLORS.get(state, "#FFFFFF")
            self.stateChanged.emit(state)
            self.colorChanged.emit(self._color)

    @Slot(float)
    def setAmplitude(self, amp: float) -> None:
        clamped = max(0.0, min(1.0, amp))
        if abs(clamped - self._amplitude) > 0.005:
            self._amplitude = clamped
            self.amplitudeChanged.emit(clamped)

    @Slot()
    def clicked(self) -> None:
        """QML calls this on orb click to trigger hotkey action."""
        bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_click")

    @Slot()
    def openSettings(self) -> None:
        from nixorb.ui.settings_window import SettingsWindow
        SettingsWindow.show_singleton()


# ------------------------------------------------------------------ #
#  Floating Orb Window                                                #
# ------------------------------------------------------------------ #
class OrbWindow(QQuickView):
    def __init__(self, settings, app: QApplication) -> None:
        super().__init__()
        self._settings = settings
        self._app      = app
        self._drag_pos: Optional[QPointF] = None
        self._bridge   = OrbBridge()

        self._setup_window()
        self._setup_qml()
        self._setup_event_subscriptions()
        self._setup_amplitude_smoother()

    # ---------------------------------------------------------------- #
    #  Window setup                                                     #
    # ---------------------------------------------------------------- #
    def _setup_window(self) -> None:
        # Frameless transparent always-on-top
        self.setFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.X11BypassWindowManagerHint  # ignored on Wayland
        )
        self.setColor(QColor(0, 0, 0, 0))   # fully transparent bg
        self.resize(QSize(ORB_SIZE, ORB_SIZE))

        # Wayland layer-shell positioning hint
        # KWin respects _KDE_NET_WM_BLUR_BEHIND_REGION for glass effect
        self.setProperty("_q_waylandSurface_role", "overlay")

        # Restore saved position
        x = self._settings.orb_x or (
            QApplication.primaryScreen().geometry().width() - ORB_SIZE - 40
        )
        y = self._settings.orb_y or 40
        self.setPosition(x, y)

    def _setup_qml(self) -> None:
        self.rootContext().setContextProperty("orbBridge", self._bridge)
        self.setSource(ORB_QML_PATH)  # type: ignore[arg-type]
        if self.status() == QQuickView.Status.Error:
            for err in self.errors():
                log.error("QML Error: %s", err.toString())
            raise RuntimeError("Failed to load orb.qml")

    # ---------------------------------------------------------------- #
    #  Mouse dragging                                                   #
    # ---------------------------------------------------------------- #
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition() - QPointF(self.x(), self.y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_pos and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition() - self._drag_pos
            self.setPosition(int(new_pos.x()), int(new_pos.y()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_pos:
            self._drag_pos = None
            # Save position
            self._settings.orb_x = self.x()
            self._settings.orb_y = self.y()
            self._settings.save()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="orb_double_click")

    # ---------------------------------------------------------------- #
    #  EventBus subscriptions (called from async context)              #
    # ---------------------------------------------------------------- #
    def _setup_event_subscriptions(self) -> None:
        bus.subscribe(Event.ORB_IDLE,      self._on_orb_state)
        bus.subscribe(Event.ORB_LISTENING, self._on_orb_state)
        bus.subscribe(Event.ORB_THINKING,  self._on_orb_state)
        bus.subscribe(Event.ORB_SPEAKING,  self._on_orb_state)
        bus.subscribe(Event.ORB_ERROR,     self._on_orb_state)
        bus.subscribe(Event.TTS_AUDIO_CHUNK, self._on_audio_chunk)

    async def _on_orb_state(self, payload: EventPayload) -> None:
        state_map = {
            Event.ORB_IDLE:      "idle",
            Event.ORB_LISTENING: "listening",
            Event.ORB_THINKING:  "thinking",
            Event.ORB_SPEAKING:  "speaking",
            Event.ORB_ERROR:     "error",
        }
        state = state_map.get(payload.event, "idle")
        # Qt GUI updates must be on the main thread
        QTimer.singleShot(0, lambda: self._bridge.setState(state))

    async def _on_audio_chunk(self, payload: EventPayload) -> None:
        """Receive PCM amplitude from TTS for orb animation."""
        import numpy as np
        pcm = payload.data.get("pcm") if payload.data else None
        if pcm is not None:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            amp = float(np.sqrt(np.mean(arr ** 2)) / 32768.0)
            self._target_amplitude = amp

    # ---------------------------------------------------------------- #
    #  Amplitude smoother (runs on Qt timer, 60 FPS)                  #
    # ---------------------------------------------------------------- #
    def _setup_amplitude_smoother(self) -> None:
        self._target_amplitude  = 0.0
        self._current_amplitude = 0.0
        self._amp_timer = QTimer(self)
        self._amp_timer.setInterval(16)   # ~60 FPS
        self._amp_timer.timeout.connect(self._smooth_amplitude)
        self._amp_timer.start()

    def _smooth_amplitude(self) -> None:
        # Exponential smoothing: fast attack, slow release
        alpha = 0.4 if self._target_amplitude > self._current_amplitude else 0.1
        self._current_amplitude += alpha * (self._target_amplitude - self._current_amplitude)
        self._bridge.setAmplitude(self._current_amplitude)

    def show(self) -> None:
        super().show()
        log.info("Orb window shown at (%d, %d)", self.x(), self.y())
