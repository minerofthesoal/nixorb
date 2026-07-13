"""NixOrb clipboard integration — read/write clipboard on Wayland.

Uses wl-paste and wl-copy for Wayland clipboard integration.
Falls back to xclip/xsel on X11.
"""
from __future__ import annotations

import asyncio
import logging
import shutil

log = logging.getLogger(__name__)

# Detect available clipboard tools
_HAS_WL_COPY = shutil.which("wl-copy") is not None
_HAS_WL_PASTE = shutil.which("wl-paste") is not None
_HAS_XCLIP = shutil.which("xclip") is not None


async def read_clipboard() -> str | None:
    """Read text from the clipboard."""
    try:
        if _HAS_WL_PASTE:
            proc = await asyncio.create_subprocess_exec(
                "wl-paste",
                "--no-newline",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode("utf-8", errors="replace")

        elif _HAS_XCLIP:
            proc = await asyncio.create_subprocess_exec(
                "xclip", "-selection", "clipboard", "-o",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return stdout.decode("utf-8", errors="replace")

        else:
            log.warning("Clipboard: no clipboard tool found (install wl-clipboard)")
            return None

    except Exception as exc:
        log.error("Clipboard: read error: %s", exc)
        return None


async def write_clipboard(text: str) -> bool:
    """Write text to the clipboard."""
    try:
        if _HAS_WL_COPY:
            proc = await asyncio.create_subprocess_exec(
                "wl-copy",
                stdin=asyncio.subprocess.PIPE,
            )
            await proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0

        elif _HAS_XCLIP:
            proc = await asyncio.create_subprocess_exec(
                "xclip", "-selection", "clipboard",
                stdin=asyncio.subprocess.PIPE,
            )
            await proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0

        else:
            log.warning("Clipboard: no clipboard tool found")
            return False

    except Exception as exc:
        log.error("Clipboard: write error: %s", exc)
        return False
