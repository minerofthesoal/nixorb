"""nixorb/ui/tray_icon.py — KDE Plasma 6 system tray icon."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from nixorb.core.event_bus import Event, bus
from nixorb.utils.paths import asset_path

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_ICON = asset_path("tray_icon.png")


class NixOrbTray(QSystemTrayIcon):
    def __init__(self, settings: Settings, parent=None) -> None:
        icon = (
            QIcon(str(_ICON))
            if _ICON.exists()
            else QIcon.fromTheme("audio-input-microphone")
        )
        super().__init__(icon, parent)
        self._settings = settings
        self._muted    = False
        self.setToolTip("NixOrb — floating AI assistant")
        self._build_menu()
        self.activated.connect(self._on_activated)

    def _build_menu(self) -> None:
        menu = QMenu()
        menu.addAction("🎙  Activate",        self._trigger)
        self._mute_action = menu.addAction("🔇  Mute Microphone", self._toggle_mute)
        self._mute_action.setCheckable(True)
        menu.addSeparator()
        menu.addAction("⚙  Settings",         self._open_settings)
        menu.addSeparator()
        menu.addAction("✕  Quit NixOrb",       self._quit)
        self.setContextMenu(menu)

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._trigger()

    def _trigger(self) -> None:
        bus.emit_sync(Event.HOTKEY_TRIGGERED, source="tray")

    def _toggle_mute(self) -> None:
        self._muted = not self._muted
        self._mute_action.setChecked(self._muted)
        bus.emit_sync(Event.MIC_MUTED, data={"muted": self._muted}, source="tray")

    def _open_settings(self) -> None:
        from nixorb.ui.settings_window import SettingsWindow
        SettingsWindow.show_singleton()

    def _quit(self) -> None:
        from PySide6.QtWidgets import QApplication
        bus.emit_sync(Event.SHUTDOWN, source="tray")
        QApplication.quit()
