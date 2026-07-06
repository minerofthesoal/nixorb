"""
nixorb/ui/hotkey.py — Global hotkey for NixOrb.

On Wayland, pynput's GlobalHotKeys uses XLib which requires a running
X server (XWayland). If DISPLAY is not set, we try to find XWayland
automatically and set DISPLAY before pynput initialises.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import TYPE_CHECKING

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


def _ensure_display() -> bool:
    """Make sure DISPLAY is set for pynput (XWayland support)."""
    if os.environ.get("DISPLAY"):
        return True

    # Try to find an active XWayland display
    try:
        out = subprocess.check_output(
            ["bash", "-c",
             "ls /tmp/.X11-unix/X* 2>/dev/null | head -1"],
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
        "  Fix: export DISPLAY=:0  (or whatever your XWayland display is)\n"
        "  Or use KDE System Settings → Shortcuts to assign a hotkey\n"
        "  that runs: nixorb start --trigger"
    )
    return False


def _pynput_combo(hotkey: str) -> str:
    mapping = {
        "ctrl":   "<ctrl>",
        "alt":    "<alt>",
        "shift":  "<shift>",
        "meta":   "<cmd>",
        "super":  "<cmd>",
        "space":  "<space>",
        "return": "<enter>",
        "enter":  "<enter>",
        "tab":    "<tab>",
        "esc":    "<esc>",
    }
    return "+".join(
        mapping.get(p.strip().lower(), p.strip().lower())
        for p in hotkey.split("+")
    )


class HotkeyManager:
    def __init__(self, settings: Settings) -> None:
        self._hotkey = settings.hotkey
        # A tiny persistent QObject living on the Qt/main thread purely so
        # QMetaObject.invokeMethod has somewhere safe to land callbacks
        # dispatched from pynput's own (non-QThread) listener thread.
        from PySide6.QtCore import QObject, Slot

        class _MainThreadRelay(QObject):
            @Slot()
            def fire(self_inner) -> None:
                log.info("🔔 Hotkey triggered: %s", self._hotkey)
                bus.emit_sync(Event.HOTKEY_TRIGGERED, source="HotkeyManager")

        self._relay = _MainThreadRelay()

    def start(self) -> None:
        t = threading.Thread(target=self._run, daemon=True, name="nixorb-hotkey-init")
        t.start()

    def _run(self) -> None:
        if not _ensure_display():
            return

        try:
            from pynput import keyboard  # type: ignore[import]
            from PySide6.QtCore import Qt, QMetaObject
            combo = _pynput_combo(self._hotkey)
            log.info("Hotkey: registering %s → pynput %s", self._hotkey, combo)

            def _activate() -> None:
                # This runs on pynput's own listener thread, which Qt does
                # not recognise as a QThread. QMetaObject.invokeMethod with
                # a queued connection is Qt's documented thread-safe way to
                # get from here to the main/Qt thread — unlike touching
                # bus.emit_sync() (and therefore the qasync loop) directly
                # from this thread, which is what previously produced
                # "QSocketNotifier / QObject::startTimer: Can only be used
                # with threads started with QThread" warnings.
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
                "Workaround: double-click the orb, or use the tray icon to activate.\n"
                "For native Wayland hotkeys, set up a KDE shortcut that runs:\n"
                "  nixorb start --trigger"
            )
