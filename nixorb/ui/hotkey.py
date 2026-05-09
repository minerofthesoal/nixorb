"""nixorb/ui/hotkey.py — Global hotkey via KGlobalAccel D-Bus or pynput fallback."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


def _pynput_combo(hotkey: str) -> str:
    """Convert 'Ctrl+Alt+Space' → '<ctrl>+<alt>+<space>' for pynput."""
    mapping = {
        "ctrl": "<ctrl>", "alt": "<alt>", "shift": "<shift>",
        "meta": "<cmd>",  "super": "<cmd>", "space": "<space>",
        "return": "<enter>", "enter": "<enter>",
    }
    parts = [p.strip() for p in hotkey.split("+")]
    return "+".join(mapping.get(p.lower(), p.lower()) for p in parts)


class HotkeyManager:
    def __init__(self, settings: "Settings") -> None:
        self._hotkey = settings.hotkey

    def start(self) -> None:
        try:
            self._try_kglobalaccel()
        except Exception as exc:
            log.info("KGlobalAccel unavailable (%s), using pynput fallback", exc)
            self._start_pynput()

    def _try_kglobalaccel(self) -> None:
        import dbus
        session = dbus.SessionBus()
        # Verify kglobalacceld is present; raises if not
        session.get_object("org.kde.kglobalaccel", "/kglobalaccel")
        # Full KGlobalAccel registration requires component registration via
        # the org.kde.KGlobalAccel interface — simplified here to verify
        # the daemon exists. A full implementation would use
        # registerGlobalShortcut() with component/action metadata.
        log.info("KGlobalAccel found — hotkey: %s", self._hotkey)
        # For now fall through to pynput for the actual listener
        raise NotImplementedError("KGlobalAccel listener not fully implemented")

    def _start_pynput(self) -> None:
        from pynput import keyboard
        combo = _pynput_combo(self._hotkey)

        def _activate():
            log.info("Global hotkey fired: %s", self._hotkey)
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="HotkeyManager")

        listener = keyboard.GlobalHotKeys({combo: _activate})
        t = threading.Thread(target=listener.start, daemon=True, name="nixorb-hotkey")
        t.start()
        log.info("pynput hotkey listener started: %s", combo)
