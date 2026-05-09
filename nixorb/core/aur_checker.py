"""nixorb/core/aur_checker.py — Arch/AUR package dependency checker."""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

# (package_name, source, reason)
REQUIRED: list[tuple[str, str, str]] = [
    ("qt6-wayland",   "pacman", "Wayland Qt6 platform plugin"),
    ("cuda",          "pacman", "CUDA runtime for GPU inference"),
    ("cudnn",         "pacman", "cuDNN for faster-whisper"),
    ("python",        "pacman", "Python 3.12 interpreter"),
    ("grim",          "pacman", "Wayland screenshot tool (screen context)"),
    ("wl-clipboard",  "pacman", "Wayland clipboard access"),
    ("ffmpeg",        "pacman", "Audio/video codec support"),
    ("kglobalacceld", "aur",   "KDE global shortcut daemon"),
    ("piper-tts",     "aur",   "Offline Piper TTS engine"),
]


def check_dependencies() -> list[str]:
    """
    Returns a list of missing package names.
    Logs a warning for each one found absent.
    """
    missing: list[str] = []
    for pkg, source, reason in REQUIRED:
        try:
            result = subprocess.run(
                ["pacman", "-Q", pkg],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                log.warning("⚠  Missing [%s] %s — %s", source.upper(), pkg, reason)
                missing.append(pkg)
        except FileNotFoundError:
            log.error("pacman not found — is this Arch Linux?")
            break
        except subprocess.TimeoutExpired:
            log.warning("pacman query timed out for: %s", pkg)
    return missing
