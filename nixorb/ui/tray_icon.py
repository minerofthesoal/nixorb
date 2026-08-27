"""NixOrb system tray icon for KDE Plasma 6."""
from __future__ import annotations

import logging

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from nixorb.core.event_bus import Event, EventPayload, bus
from nixorb.ui.orb_window import STATE_COLORS

log = logging.getLogger(__name__)

# The orb has no caption at 88px, so the tray tooltip is where the state is
# spelled out in words.
STATE_LABELS: dict[str, str] = {
    "idle": "Idle",
    "listening": "Listening…",
    "thinking": "Thinking…",
    "speaking": "Speaking…",
    "error": "Error",
}


class NixOrbTray(QSystemTrayIcon):
    """System tray icon with context menu."""

    def __init__(self, settings, app: QApplication) -> None:
        super().__init__(app)
        self._settings = settings
        self._app = app

        self._state = "idle"
        self._setup_icon()
        self._setup_menu()
        self._subscribe_events()
        self.setVisible(True)

    def _setup_icon(self) -> None:
        """Draw the tray icon in the current state colour."""
        self.setIcon(self._render_icon(STATE_COLORS.get(self._state, STATE_COLORS["idle"])))
        self._refresh_tooltip()

    @staticmethod
    def _render_icon(hex_color: str) -> QIcon:
        """A small shaded sphere matching the orb, not a flat cyan dot."""
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        base = QColor(hex_color)
        gradient = QRadialGradient(size * 0.38, size * 0.34, size * 0.62)
        gradient.setColorAt(0.0, base.lighter(155))
        gradient.setColorAt(0.55, base)
        gradient.setColorAt(1.0, base.darker(165))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRect(4, 4, size - 8, size - 8))
        painter.end()

        return QIcon(pixmap)

    def _refresh_tooltip(self) -> None:
        label = STATE_LABELS.get(self._state, self._state.title())
        self.setToolTip(f"NixOrb — {label}")

    def _subscribe_events(self) -> None:
        """Mirror the orb's state in the tray icon and tooltip."""
        for event, state in (
            (Event.ORB_IDLE, "idle"),
            (Event.ORB_LISTENING, "listening"),
            (Event.ORB_THINKING, "thinking"),
            (Event.ORB_SPEAKING, "speaking"),
            (Event.ORB_ERROR, "error"),
        ):
            bus.subscribe(event, self._make_state_handler(state))

    def _make_state_handler(self, state: str):
        async def handler(_payload: EventPayload) -> None:
            if state == self._state:
                return
            self._state = state
            self.setIcon(self._render_icon(STATE_COLORS[state]))
            self._refresh_tooltip()

        return handler

    def _setup_menu(self) -> None:
        """Create the context menu."""
        menu = QMenu()

        # Activate action
        activate_action = QAction("Activate", menu)
        activate_action.triggered.connect(self._on_activate)
        menu.addAction(activate_action)

        menu.addSeparator()

        # Mute/Unmute
        self._mute_action = QAction("Mute Microphone", menu)
        self._mute_action.setCheckable(True)
        self._mute_action.triggered.connect(self._on_mute_toggle)
        menu.addAction(self._mute_action)

        menu.addSeparator()

        # Settings
        settings_action = QAction("Settings…", menu)
        settings_action.triggered.connect(self._on_settings)
        menu.addAction(settings_action)

        menu.addSeparator()

        # Quit
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._on_quit)
        menu.addAction(quit_action)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._on_activate()

    def _on_activate(self) -> None:
        """Trigger activation via event bus."""
        log.info("Tray: activate triggered")
        bus.emit_sync(Event.HOTKEY_TRIGGERED, source="tray_icon")

    def _on_mute_toggle(self) -> None:
        """Toggle microphone mute."""
        muted = self._mute_action.isChecked()
        bus.emit_sync(
            Event.MIC_MUTED,
            data={"muted": muted},
            source="tray_icon",
        )
        log.info("Tray: microphone %s", "muted" if muted else "unmuted")

    def _on_settings(self) -> None:
        """Open settings window."""
        from nixorb.ui.settings_window import SettingsWindow

        SettingsWindow.show_singleton()

    def _on_quit(self) -> None:
        """Quit NixOrb."""
        log.info("Tray: quit requested")
        # SHUTDOWN lets main() unwind and release models, the HTTP session and
        # the bus. Calling app.quit() here would kill the loop first and skip
        # all of it.
        bus.emit_sync(Event.SHUTDOWN, source="tray_icon")
