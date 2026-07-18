"""NixOrb system tray icon for KDE Plasma 6."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)


class NixOrbTray(QSystemTrayIcon):
    """System tray icon with context menu."""

    def __init__(self, settings, app: QApplication) -> None:
        super().__init__(app)
        self._settings = settings
        self._app = app

        self._setup_icon()
        self._setup_menu()
        self.setVisible(True)

    def _setup_icon(self) -> None:
        """Set up the tray icon."""
        # Use a standard icon as fallback
        icon = QIcon.fromTheme("nixorb", QIcon.fromTheme("audio-input-microphone"))
        if icon.isNull():
            # Create a simple colored circle icon programmatically
            from PySide6.QtCore import QRect
            from PySide6.QtGui import QPainter, QPixmap

            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setBrush(Qt.GlobalColor.cyan)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRect(2, 2, 28, 28))
            painter.end()
            icon = QIcon(pixmap)

        self.setIcon(icon)
        self.setToolTip("NixOrb — AI Assistant")

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
        bus.emit_sync(Event.SHUTDOWN, source="tray_icon")
        self._app.quit()
