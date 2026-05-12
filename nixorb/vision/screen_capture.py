"""nixorb/vision/screen_capture.py — Wayland capture + CogFlorence/VLM vision."""
from __future__ import annotations

import asyncio
import base64 as _b64
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
        if not _HAS_GRIM:
            log.error("grim not found — sudo pacman -S grim")
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
            return _b64.b64encode(tmp.read_bytes()).decode()
        except TimeoutError:
            log.error("grim timed out")
            return None
        finally:
            tmp.unlink(missing_ok=True)

    async def describe(self, llm: LLMBackend,
                       question: str = "Describe this screen concisely.") -> str:
        b64 = await self.capture_b64()
        if not b64:
            return "⚠️ Screen capture unavailable."
        messages = [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
            {"type": "text", "text": question},
        ]}]
        chunks: list[str] = []
        try:
            async for chunk in llm.stream(messages):
                chunks.append(chunk)
        except Exception as exc:
            log.error("VLM describe failed: %s", exc)
            return "Screen description unavailable."
        return "".join(chunks).strip() or "Nothing notable on screen."

    async def describe_cogflorence(
        self,
        model_id: str = "thwri/CogFlorence-2.2-Large",
        hf_token: str = "",
    ) -> str:
        """Lightweight captioning via CogFlorence-2.2-Large."""
        b64 = await self.capture_b64()
        if not b64:
            return "⚠️ Screen capture unavailable."
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._cogflorence_sync, b64, model_id, hf_token
        )

    @staticmethod
    def _cogflorence_sync(b64: str, model_id: str, hf_token: str) -> str:
        import io
        try:
            from PIL import Image
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch
        except ImportError:
            return "CogFlorence unavailable — pip install transformers pillow"
        try:
            img = Image.open(io.BytesIO(_b64.b64decode(b64))).convert("RGB")
            tok = None
            processor = AutoProcessor.from_pretrained(
                model_id, token=hf_token or None, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, token=hf_token or None, trust_remote_code=True
            )
            model.eval()
            inputs = processor(images=img, text="<DETAILED_CAPTION>", return_tensors="pt")
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            return processor.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as exc:
            log.error("CogFlorence failed: %s", exc)
            return f"Vision error: {exc}"
