"""nixorb/ui/hotkey.py — Global hotkey via pynput with Wayland/XWayland support."""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)


def _pynput_combo(hotkey: str) -> str:
    mapping = {
        "ctrl": "<ctrl>", "alt": "<alt>", "shift": "<shift>",
        "meta": "<cmd>",  "super": "<cmd>", "space": "<space>",
    }
    return "+".join(
        mapping.get(p.strip().lower(), p.strip().lower())
        for p in hotkey.split("+")
    )


class HotkeyManager:
    def __init__(self, settings: Settings) -> None:
        self._hotkey = settings.hotkey

    def start(self) -> None:
        self._start_pynput()

    def _start_pynput(self) -> None:
        try:
            from pynput import keyboard
            combo = _pynput_combo(self._hotkey)

            def _activate() -> None:
                log.info("Hotkey fired: %s", self._hotkey)
                bus.emit_sync(Event.HOTKEY_TRIGGERED, source="HotkeyManager")

            listener = keyboard.GlobalHotKeys({combo: _activate})
            t = threading.Thread(
                target=listener.start, daemon=True, name="nixorb-hotkey"
            )
            t.start()
            log.info("Hotkey listener started: %s → %s", self._hotkey, combo)
        except Exception as exc:
            log.error("Hotkey setup failed: %s", exc)
            log.error(
                "On Wayland without XWayland, global hotkeys require KGlobalAccel. "
                "Try: export QT_QPA_PLATFORM=xcb  (run under XWayland)"
            )
