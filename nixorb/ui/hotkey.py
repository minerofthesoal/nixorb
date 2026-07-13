"""NixOrb global hotkey manager.

Uses pynput for global hotkey capture. On Wayland, pynput requires
XWayland (DISPLAY env var). If unavailable, falls back to orb double-click.

KDE Plasma 6 users can also set a KWin shortcut that runs `nixorb trigger`.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import TYPE_CHECKING

from PySide6.QtCore import QMetaObject, QObject, Qt, Slot

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


def _ensure_display() -> bool:
    """Ensure DISPLAY is set for pynput (XWayland support)."""
    if os.environ.get("DISPLAY"):
        return True

    # Try to find XWayland display
    try:
        out = subprocess.check_output(
            ["bash", "-c", "ls /tmp/.X11-unix/X* 2>/dev/null | head -1"],
            timeout=2,
        ).decode().strip()
        if out:
            display = ":" + out.split("X")[-1]
            os.environ["DISPLAY"] = display
            log.info("Hotkey: auto-detected XWayland DISPLAY=%s", display)
            return True
    except Exception:
        pass

    log.warning(
        "Hotkey: DISPLAY not set — global hotkeys need XWayland.\n"
        "  Fix: export DISPLAY=:0\n"
        "  Or use KDE System Settings → Shortcuts to assign a hotkey\n"
        "  that runs: nixorb trigger"
    )
    return False


def _pynput_combo(hotkey: str) -> str:
    """Convert a hotkey string to pynput format."""
    mapping = {
        "ctrl": "<ctrl>",
        "alt": "<alt>",
        "shift": "<shift>",
        "meta": "<cmd>",
        "super": "<cmd>",
        "space": "<space>",
        "return": "<enter>",
        "enter": "<enter>",
        "tab": "<tab>",
        "esc": "<esc>",
    }
    return "+".join(
        mapping.get(p.strip().lower(), p.strip().lower())
        for p in hotkey.split("+")
    )


class _Relay(QObject):
    """Relay object for thread-safe hotkey callback."""

    @Slot()
    def fire(self) -> None:
        log.info("🔔 Hotkey triggered")
        bus.emit_sync(Event.HOTKEY_TRIGGERED, source="HotkeyManager")


class HotkeyManager:
    """Global hotkey manager using pynput."""

    def __init__(self, settings: Settings) -> None:
        self._hotkey = settings.hotkey
        self._relay = _Relay()

    def start(self) -> None:
        """Start the hotkey listener in a background thread."""
        t = threading.Thread(target=self._run, daemon=True, name="hotkey")
        t.start()

    def _run(self) -> None:
        """Run the pynput hotkey listener."""
        if not _ensure_display():
            return

        try:
            from pynput import keyboard

            combo = _pynput_combo(self._hotkey)
            log.info("Hotkey: registering %s → pynput %s", self._hotkey, combo)

            def _activate() -> None:
                # Thread-safe: use QMetaObject to dispatch to Qt main thread
                QMetaObject.invokeMethod(
                    self._relay, "fire", Qt.ConnectionType.QueuedConnection
                )

            listener = keyboard.GlobalHotKeys({combo: _activate})
            listener.start()
            log.info("Hotkey listener active: %s", self._hotkey)
            listener.join()

        except Exception as exc:
            log.error("Hotkey setup failed: %s", exc)
            log.error(
                "Workaround: double-click the orb, or use the tray icon.\n"
                "For native Wayland hotkeys, set up a KDE shortcut that runs:\n"
                "  nixorb trigger"
            )
