"""
nixorb/plugins/builtin/systemd_plugin.py

Built-in plugin: query and manage systemd services.
"""
from __future__ import annotations

import subprocess

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "systemd_service",
        "description": (
            "Query or control a systemd service. "
            "Actions: status, start, stop, restart, enable, disable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name, e.g. 'nginx' or 'sshd.service'",
                },
                "action": {
                    "type": "string",
                    "enum": ["status", "start", "stop", "restart", "enable", "disable"],
                    "description": "Action to perform on the service",
                },
            },
            "required": ["service", "action"],
        },
    },
}


def systemd_service(service: str, action: str) -> str:
    if action == "status":
        cmd = ["systemctl", "status", "--no-pager", "--lines=8", service]
    else:
        cmd = ["systemctl", action, service]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    output = (result.stdout + result.stderr).strip()
    return output[:1_500] if output else f"No output for `systemctl {action} {service}`"
