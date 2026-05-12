"""nixorb/utils/system_info.py — Runtime system information helpers."""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess


def get_system_info() -> dict:
    """Return a dict of key system facts for diagnostics."""
    info: dict = {
        "os":      platform.system(),
        "distro":  _read_os_release().get("PRETTY_NAME", "unknown"),
        "kernel":  platform.release(),
        "arch":    platform.machine(),
        "python":  platform.python_version(),
        "display": os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY", "none"),
        "de":      os.environ.get("XDG_CURRENT_DESKTOP", "unknown"),
    }
    # GPU
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"],
                timeout=3,
            ).decode().strip()
            info["gpu"] = out
        except Exception:
            info["gpu"] = "nvidia-smi failed"
    else:
        info["gpu"] = "no nvidia-smi"

    # Wayland compositor
    info["compositor"] = os.environ.get("XDG_SESSION_TYPE", "unknown")
    return info


def _read_os_release() -> dict:
    try:
        with open("/etc/os-release") as f:
            return dict(
                line.strip().split("=", 1)
                for line in f
                if "=" in line and not line.startswith("#")
            )
    except OSError:
        return {}


async def check_wayland_tools() -> dict[str, bool]:
    """Check which Wayland integration tools are available."""
    tools = ["grim", "wl-paste", "wl-copy", "kglobalacceld", "bwrap"]
    return {t: bool(shutil.which(t)) for t in tools}


def get_vram_info() -> dict:
    """Return VRAM stats without importing torch."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3,
        ).decode().strip()
        used, free, total = (int(x.strip()) for x in out.split(","))
        return {"used_mb": used, "free_mb": free, "total_mb": total}
    except Exception:
        return {"used_mb": 0, "free_mb": 0, "total_mb": 0}
