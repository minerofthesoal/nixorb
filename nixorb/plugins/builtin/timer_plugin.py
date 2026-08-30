"""
nixorb/plugins/builtin/timer_plugin.py

Built-in plugin: set a timer/reminder that fires a desktop notification
(and speaks, if a TTS callback is registered — see set_speak_callback)
after a delay. Non-blocking: the tool call returns immediately once the
timer is scheduled; the notification fires later on its own thread.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": (
            "Set a timer/reminder that notifies you after a delay. "
            "Returns immediately; the notification fires later in the "
            "background."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Delay before the timer fires, in seconds. "
                                   "E.g. 300 for 5 minutes, 3600 for 1 hour.",
                },
                "label": {
                    "type": "string",
                    "description": "What to remind about, e.g. 'check the oven'.",
                },
            },
            "required": ["seconds"],
        },
    },
}

# Optional hook so main.py can wire this up to actually speak the reminder
# through the active TTS backend, not just pop a desktop notification.
_speak_callback: Callable[[str], None] | None = None


def set_speak_callback(callback: Callable[[str], None] | None) -> None:
    global _speak_callback
    _speak_callback = callback


def _fire(label: str) -> None:
    message = label or "Timer's up"
    if shutil.which("notify-send"):
        try:
            subprocess.run(
                ["notify-send", "-a", "NixOrb", "-i", "alarm-symbolic", "Timer", message],
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            pass
    if _speak_callback is not None:
        try:
            _speak_callback(message)
        except Exception:
            pass


def set_timer(seconds: float, label: str = "") -> str:
    if seconds <= 0:
        return "Timer duration must be a positive number of seconds."
    if seconds > 24 * 3600:
        return "Timer duration is capped at 24 hours."

    timer = threading.Timer(seconds, _fire, args=(label,))
    timer.daemon = True
    timer.start()

    if seconds < 60:
        when = f"{seconds:g}s"
    elif seconds < 3600:
        when = f"{seconds / 60:.1f} min"
    else:
        when = f"{seconds / 3600:.1f} hr"
    return f"Timer set for {when}" + (f" — {label}" if label else "") + "."
