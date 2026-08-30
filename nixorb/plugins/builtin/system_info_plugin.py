"""
nixorb/plugins/builtin/system_info_plugin.py

Built-in plugin: read-only system status (CPU load, RAM, disk, uptime).

Linux-only (reads /proc), stdlib only — no psutil dependency.
"""
from __future__ import annotations

import os
import shutil

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "system_status",
        "description": (
            "Get current system status: CPU load, RAM usage, disk free "
            "space on /, and uptime. Read-only, no arguments."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, _, rest = line.partition(":")
            digits = "".join(ch for ch in rest.split()[0] if ch.isdigit()) if rest.split() else ""
            if digits:
                values[key] = int(digits)  # kB
    return values


def _read_uptime_seconds() -> float:
    with open("/proc/uptime", encoding="utf-8") as f:
        return float(f.read().split()[0])


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _fmt_duration(seconds: float) -> str:
    days, rem = divmod(int(seconds), 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def system_status() -> str:
    lines = []

    try:
        load1, load5, load15 = os.getloadavg()
        ncpu = os.cpu_count() or 1
        lines.append(
            f"CPU load: {load1:.2f}, {load5:.2f}, {load15:.2f} "
            f"(1/5/15 min, {ncpu} cores)"
        )
    except OSError as exc:
        lines.append(f"CPU load: unavailable ({exc})")

    try:
        mem = _read_meminfo()
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", mem.get("MemFree", 0))
        used_kb = max(total_kb - avail_kb, 0)
        pct = (used_kb / total_kb * 100) if total_kb else 0.0
        lines.append(
            f"RAM: {_fmt_bytes(used_kb * 1024)} / {_fmt_bytes(total_kb * 1024)} "
            f"used ({pct:.0f}%)"
        )
    except OSError as exc:
        lines.append(f"RAM: unavailable ({exc})")

    try:
        usage = shutil.disk_usage("/")
        pct = (usage.used / usage.total * 100) if usage.total else 0.0
        lines.append(
            f"Disk (/): {_fmt_bytes(usage.used)} / {_fmt_bytes(usage.total)} "
            f"used ({pct:.0f}%), {_fmt_bytes(usage.free)} free"
        )
    except OSError as exc:
        lines.append(f"Disk: unavailable ({exc})")

    try:
        lines.append(f"Uptime: {_fmt_duration(_read_uptime_seconds())}")
    except OSError as exc:
        lines.append(f"Uptime: unavailable ({exc})")

    return "\n".join(lines)
