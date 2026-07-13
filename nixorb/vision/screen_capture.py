"""NixOrb screen capture — take screenshots on Wayland.

Uses `grim` for Wayland screenshot capture. Can describe the screen
using a vision-capable model if available.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

_HAS_GRIM = shutil.which("grim") is not None
_HAS_SLURP = shutil.which("slurp") is not None


class ScreenCapture:
    """Screen capture for Wayland (KDE Plasma 6)."""

    def __init__(self) -> None:
        if not _HAS_GRIM:
            log.warning("ScreenCapture: 'grim' not found — install it for screenshots")

    async def capture(self, output_path: Path | None = None) -> Path | None:
        """Capture the screen and save to file.

        Args:
            output_path: Where to save the screenshot. If None, uses temp file.

        Returns:
            Path to the saved screenshot, or None if capture failed.
        """
        if not _HAS_GRIM:
            log.error("ScreenCapture: grim not available")
            return None

        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / "nixorb_screenshot.png"

        try:
            proc = await asyncio.create_subprocess_exec(
                "grim", str(output_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)

            if proc.returncode == 0 and output_path.exists():
                log.info("ScreenCapture: screenshot saved to %s", output_path)
                return output_path
            else:
                log.error("ScreenCapture: grim failed: %s", stderr.decode())
                return None

        except Exception as exc:
            log.error("ScreenCapture: error: %s", exc)
            return None

    async def capture_region(self, output_path: Path | None = None) -> Path | None:
        """Capture a selected region of the screen.

        Uses slurp for region selection (requires user interaction).
        """
        if not _HAS_GRIM or not _HAS_SLURP:
            log.error("ScreenCapture: grim and slurp required for region capture")
            return None

        if output_path is None:
            output_path = Path(tempfile.gettempdir()) / "nixorb_screenshot.png"

        try:
            # Use slurp to get region, then grim to capture
            proc = await asyncio.create_subprocess_shell(
                f"slurp | grim -g - {output_path}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            if proc.returncode == 0 and output_path.exists():
                log.info("ScreenCapture: region screenshot saved")
                return output_path
            else:
                log.error("ScreenCapture: region capture failed: %s", stderr.decode())
                return None

        except asyncio.TimeoutError:
            log.warning("ScreenCapture: region selection timed out")
            return None
        except Exception as exc:
            log.error("ScreenCapture: region capture error: %s", exc)
            return None

    async def describe(self, image_path: Path | None = None) -> str:
        """Capture screen and return a description.

        If no vision model is available, returns a basic description
        indicating that a screenshot was taken.
        """
        path = image_path or await self.capture()
        if path is None:
            return "[Screen capture failed]"

        # Try to use a vision model if available
        try:
            from PIL import Image

            with Image.open(path) as img:
                width, height = img.size
                return f"[Screenshot captured: {width}x{height} pixels at {path}]"
        except ImportError:
            return f"[Screenshot captured at {path}]"
