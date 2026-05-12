"""
plugins/kdeconnect_plugin.py

NixOrb plugin: KDE Connect integration.
Send text, URLs and notifications to paired phone/device.
"""
from __future__ import annotations

import subprocess

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "kde_connect_send",
        "description": (
            "Send a message, URL, or notification to the user's phone or "
            "another device paired via KDE Connect. Use when the user says "
            "'send to my phone', 'share this link', or similar."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The text or URL to send"
                },
                "device_id": {
                    "type": "string",
                    "description": "KDE Connect device ID (optional; omit to use first available)"
                },
            },
            "required": ["message"],
        },
    },
}


def kde_connect_send(message: str, device_id: str = "") -> str:
    """Send *message* to a KDE Connect device."""
    import shutil
    if not shutil.which("kdeconnect-cli"):
        return "kdeconnect-cli not found. Install kdeconnect from the AUR."

    # Resolve device if not specified
    if not device_id:
        r = subprocess.run(
            ["kdeconnect-cli", "--list-available", "--id-only"],
            capture_output=True, text=True, timeout=5,
        )
        ids = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        if not ids:
            return "No KDE Connect devices available."
        device_id = ids[0]

    result = subprocess.run(
        ["kdeconnect-cli", "--device", device_id, "--ping-msg", message],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return f"✓ Sent to {device_id}: {message[:60]}"
    return f"Failed: {result.stderr.strip()}"
