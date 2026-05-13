"""
nixorb/plugins/builtin/kdeconnect_plugin.py

Built-in plugin: KDE Connect integration.
Send notifications, SMS, and clipboard to paired phone.
"""
from __future__ import annotations

import shutil
import subprocess

_HAS_KDECONNECT = bool(shutil.which("kdeconnect-cli"))

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "kdeconnect",
        "description": (
            "Interact with a paired phone via KDE Connect. "
            "Actions: list_devices, ping, send_sms, share_url, ring_phone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_devices", "ping", "send_sms", "share_url", "ring_phone"],
                },
                "device_id": {
                    "type": "string",
                    "description": "KDE Connect device ID (optional if only one device is paired)",
                },
                "message": {
                    "type": "string",
                    "description": "SMS message body or URL to share",
                },
                "phone_number": {
                    "type": "string",
                    "description": "Recipient phone number for send_sms",
                },
            },
            "required": ["action"],
        },
    },
}


def _run(*args: str) -> str:
    if not _HAS_KDECONNECT:
        return "kdeconnect-cli not found. Install kdeconnect from pacman."
    result = subprocess.run(
        ["kdeconnect-cli"] + list(args),
        capture_output=True, text=True, timeout=10,
    )
    return (result.stdout + result.stderr).strip() or "(no output)"


def _first_device_id() -> str | None:
    out = _run("-l", "--id-only")
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return lines[0] if lines else None


def kdeconnect(
    action: str,
    device_id: str = "",
    message: str = "",
    phone_number: str = "",
) -> str:
    dev = device_id or _first_device_id() or ""

    if action == "list_devices":
        return _run("--list-available", "-l")

    if not dev:
        return "No KDE Connect device found. Pair a device first."

    if action == "ping":
        return _run(f"--device={dev}", "--ping-msg", message or "Hello from NixOrb!")

    elif action == "send_sms":
        if not phone_number or not message:
            return "send_sms requires both phone_number and message."
        return _run(f"--device={dev}", "--send-sms", message,
                    "--destination", phone_number)

    elif action == "share_url":
        if not message:
            return "share_url requires a URL in the message field."
        return _run(f"--device={dev}", "--share", message)

    elif action == "ring_phone":
        return _run(f"--device={dev}", "--ring")

    return f"Unknown action: {action}"
