"""nixorb/action/clipboard.py — Wayland clipboard integration."""
from __future__ import annotations

import asyncio
import logging
import shutil

log = logging.getLogger(__name__)

_HAS_WL_PASTE = bool(shutil.which("wl-paste"))
_HAS_WL_COPY  = bool(shutil.which("wl-copy"))


async def read_clipboard() -> str | None:
    """Read current Wayland clipboard text content."""
    if not _HAS_WL_PASTE:
        log.warning("wl-paste not found — install wl-clipboard")
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "wl-paste", "--no-newline",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode(errors="replace").strip() or None
    except Exception as exc:
        log.error("Clipboard read failed: %s", exc)
        return None


async def write_clipboard(text: str) -> bool:
    """Write *text* to the Wayland clipboard."""
    if not _HAS_WL_COPY:
        log.warning("wl-copy not found — install wl-clipboard")
        return False
    try:
        proc = await asyncio.create_subprocess_exec(
            "wl-copy",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(
            proc.communicate(input=text.encode("utf-8")), timeout=5
        )
        return proc.returncode == 0
    except Exception as exc:
        log.error("Clipboard write failed: %s", exc)
        return False
