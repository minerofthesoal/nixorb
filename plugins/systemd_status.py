"""
plugins/systemd_status.py

NixOrb built-in plugin: query and control systemd services.

Provides three tools:
  get_service_status  — check if a service is running
  start_service       — start a service (requires confirmation)
  stop_service        — stop a service (requires confirmation)
"""
from __future__ import annotations

import subprocess

# ── Tool definitions (OpenAI function-calling schema) ─────────────── #
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_service_status",
        "description": (
            "Get the current status of a systemd service. Use when the user "
            "asks whether a service is running, its logs, or its state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "The systemd unit name, e.g. 'nginx.service' or 'sshd'"
                }
            },
            "required": ["service_name"],
        },
    },
}


def get_service_status(service_name: str) -> str:
    """Returns a concise status string for *service_name*."""
    result = subprocess.run(
        ["systemctl", "status", "--no-pager", "--lines=8", service_name],
        capture_output=True, text=True, timeout=10,
    )
    output = (result.stdout + result.stderr).strip()
    # Trim to reasonable length
    lines = output.splitlines()[:20]
    return "\n".join(lines) if lines else f"No status for {service_name}"


TOOL_DEFINITION_START = {
    "type": "function",
    "function": {
        "name": "start_service",
        "description": "Start a stopped systemd service.",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string"}
            },
            "required": ["service_name"],
        },
    },
}


def start_service(service_name: str) -> str:
    r = subprocess.run(
        ["systemctl", "--user", "start", service_name],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode == 0:
        return f"✓ {service_name} started."
    # Retry with sudo
    r2 = subprocess.run(
        ["sudo", "systemctl", "start", service_name],
        capture_output=True, text=True, timeout=15,
    )
    return r2.stdout.strip() or r2.stderr.strip() or f"Started {service_name} (exit {r2.returncode})"


TOOL_DEFINITION_STOP = {
    "type": "function",
    "function": {
        "name": "stop_service",
        "description": "Stop a running systemd service.",
        "parameters": {
            "type": "object",
            "properties": {
                "service_name": {"type": "string"}
            },
            "required": ["service_name"],
        },
    },
}


def stop_service(service_name: str) -> str:
    r = subprocess.run(
        ["sudo", "systemctl", "stop", service_name],
        capture_output=True, text=True, timeout=15,
    )
    return r.stdout.strip() or r.stderr.strip() or f"Stopped {service_name} (exit {r.returncode})"
