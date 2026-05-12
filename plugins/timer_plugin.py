"""plugins/timer_plugin.py — Set timers and alarms via notify-send."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import threading

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": (
            "Set a timer or reminder. Sends a desktop notification when done. "
            "Use when user says 'set a timer for 5 minutes', "
            "'remind me in 10 minutes', 'alarm in 1 hour'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Duration in seconds.",
                },
                "message": {
                    "type": "string",
                    "description": "Notification message when timer fires.",
                },
                "minutes": {
                    "type": "number",
                    "description": "Duration in minutes (alternative to seconds).",
                },
            },
        },
    },
}

_HAS_NOTIFY = bool(shutil.which("notify-send"))
_timers: list[threading.Timer] = []


def set_timer(
    seconds: int = 0,
    message: str = "Timer complete!",
    minutes: float = 0.0,
) -> str:
    total_seconds = int(seconds + minutes * 60)
    if total_seconds <= 0:
        return "Please specify a duration (seconds or minutes)."

    def _fire():
        if _HAS_NOTIFY:
            subprocess.run(
                ["notify-send", "--urgency=normal",
                 "--icon=dialog-information",
                 "⏰ NixOrb Timer", message],
                timeout=5,
            )
        # Also emit to event bus
        try:
            from nixorb.core.event_bus import Event, bus
            bus.emit_sync(
                Event.LOG,
                data={"level": "info", "msg": f"⏰ Timer: {message}"},
                source="timer_plugin",
            )
        except Exception:
            pass

    t = threading.Timer(total_seconds, _fire)
    t.daemon = True
    t.start()
    _timers.append(t)

    mins, secs = divmod(total_seconds, 60)
    hrs,  mins = divmod(mins, 60)
    if hrs:
        human = f"{hrs}h {mins}m {secs}s"
    elif mins:
        human = f"{mins}m {secs}s"
    else:
        human = f"{secs}s"

    return f"⏰ Timer set for {human}: {message}"
