"""nixorb/vision/screen_capture.py — Wayland screen capture via grim."""
from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nixorb.llm.backends import LLMBackend

log = logging.getLogger(__name__)

_HAS_GRIM = bool(shutil.which("grim"))


class ScreenCapture:
    async def capture_b64(self) -> str | None:
        """Capture full screen; return base64 PNG or None on failure."""
        if not _HAS_GRIM:
            log.error("grim not found — install it: sudo pacman -S grim")
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                "grim", str(tmp),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                log.error("grim failed: %s", stderr.decode(errors="replace"))
                return None
            return base64.b64encode(tmp.read_bytes()).decode()
        except asyncio.TimeoutError:
            log.error("grim timed out")
            return None
        finally:
            tmp.unlink(missing_ok=True)

    async def describe(
        self,
        llm: "LLMBackend",
        question: str = "Describe what is shown on this screen.",
    ) -> str:
        b64 = await self.capture_b64()
        if b64 is None:
            return "⚠️  Screen capture unavailable."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        chunks: list[str] = []
        async for chunk in llm.stream(messages):
            chunks.append(chunk)
        return "".join(chunks).strip() or "No description returned."
