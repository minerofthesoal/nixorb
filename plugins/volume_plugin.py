"""plugins/volume_plugin.py — Control system volume via PipeWire/PulseAudio."""
from __future__ import annotations

import shutil
import subprocess

_HAS_WPCTL  = bool(shutil.which("wpctl"))   # PipeWire
_HAS_PACTL  = bool(shutil.which("pactl"))   # PulseAudio

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "control_volume",
        "description": (
            "Get or set the system audio volume. "
            "Use when the user says 'volume up', 'mute', 'set volume to 50%', etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "up", "down", "mute", "unmute"],
                    "description": "Volume action to perform.",
                },
                "percent": {
                    "type": "integer",
                    "description": "Volume percentage (0-100) for 'set' action.",
                },
                "step": {
                    "type": "integer",
                    "description": "Step percentage for up/down (default 10).",
                },
            },
            "required": ["action"],
        },
    },
}


def control_volume(
    action: str, percent: int | None = None, step: int = 10
) -> str:
    if _HAS_WPCTL:
        return _wpctl(action, percent, step)
    if _HAS_PACTL:
        return _pactl(action, percent, step)
    return "No volume control found (install PipeWire or PulseAudio)"


def _wpctl(action: str, percent: int | None, step: int) -> str:
    def run(*args) -> str:
        r = subprocess.run(
            ["wpctl"] + list(args),
            capture_output=True, text=True, timeout=5
        )
        return (r.stdout + r.stderr).strip()

    if action == "get":
        return run("get-volume", "@DEFAULT_AUDIO_SINK@")
    if action == "set" and percent is not None:
        return run("set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent / 100:.2f}")
    if action == "up":
        return run("set-volume", "-l", "1.0", "@DEFAULT_AUDIO_SINK@", f"{step / 100:.2f}+")
    if action == "down":
        return run("set-volume", "@DEFAULT_AUDIO_SINK@", f"{step / 100:.2f}-")
    if action == "mute":
        return run("set-mute", "@DEFAULT_AUDIO_SINK@", "1")
    if action == "unmute":
        return run("set-mute", "@DEFAULT_AUDIO_SINK@", "0")
    return f"Unknown action: {action}"


def _pactl(action: str, percent: int | None, step: int) -> str:
    def run(*args) -> str:
        r = subprocess.run(
            ["pactl"] + list(args),
            capture_output=True, text=True, timeout=5
        )
        return (r.stdout + r.stderr).strip()

    if action == "get":
        return run("get-sink-volume", "@DEFAULT_SINK@")
    if action == "set" and percent is not None:
        return run("set-sink-volume", "@DEFAULT_SINK@", f"{percent}%")
    if action == "up":
        return run("set-sink-volume", "@DEFAULT_SINK@", f"+{step}%")
    if action == "down":
        return run("set-sink-volume", "@DEFAULT_SINK@", f"-{step}%")
    if action == "mute":
        return run("set-sink-mute", "@DEFAULT_SINK@", "1")
    if action == "unmute":
        return run("set-sink-mute", "@DEFAULT_SINK@", "0")
    return f"Unknown action: {action}"
