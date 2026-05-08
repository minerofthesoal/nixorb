"""
plugins/systemd_status.py

Example NixOrb plugin: query systemd service status.

NixOrb discovers this automatically from the plugins/ directory.
Expose TOOL_DEFINITION (OpenAI tool schema) and a matching function.
"""
from __future__ import annotations

import subprocess

# ------------------------------------------------------------------ #
#  Tool definition (OpenAI function-calling format)                   #
# ------------------------------------------------------------------ #
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_service_status",
        "description": (
            "Get the status of a systemd service on the Arch Linux host. "
            "Use when the user asks about a running service or daemon."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The systemd service name, e.g. 'nginx' or 'sshd'",
                }
            },
            "required": ["service_name"],
        },
    },
}


# ------------------------------------------------------------------ #
#  Implementation — name must match function.name above               #
# ------------------------------------------------------------------ #
def get_service_status(service_name: str) -> str:
    """Returns a concise systemd status string."""
    result = subprocess.run(
        ["systemctl", "status", "--no-pager", "--lines=5", service_name],
        capture_output=True, text=True, timeout=10,
    )
    output = (result.stdout + result.stderr).strip()
    return output[:1000] if output else f"No status found for {service_name}"


# ------------------------------------------------------------------ #
#  Optional: KDE Connect plugin example stub                          #
# ------------------------------------------------------------------ #
TOOL_DEFINITION_KDECONNECT = {
    "type": "function",
    "function": {
        "name": "send_to_phone",
        "description": "Send a text message or URL to the paired phone via KDE Connect.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message or URL to send"},
                "device_id": {"type": "string", "description": "KDE Connect device ID"},
            },
            "required": ["message"],
        },
    },
}


def send_to_phone(message: str, device_id: str = "") -> str:
    cmd = ["kdeconnect-cli", "--ping-msg", message]
    if device_id:
        cmd += ["--device", device_id]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return result.stdout.strip() or result.stderr.strip() or "Sent."
