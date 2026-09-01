"""Generic Hugging Face LLM — any causal language model on the Hub.

    llm_backend  = "huggingface"
    llm_hf_model = "Qwen/Qwen3-4B-Instruct"   # or Llama, Phi, Gemma, SmolLM…

Streams tokens through TextIteratorStreamer so the orb animates while the
model is still generating, exactly as the Ollama backend does. Generation
runs in a worker thread: transformers' generate() is blocking, and on the
event loop it would freeze the UI for the whole answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nixorb import hf
from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# How long to wait for the first token before deciding generation is wedged.
FIRST_TOKEN_TIMEOUT = 180.0


class HFLLMError(RuntimeError):
    """Raised when a Hugging Face model cannot load or generate."""


class HuggingFaceBackend:
    """Local text generation with any HF causal LM."""

    name = "huggingface"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_id = getattr(settings, "llm_hf_model", "") or DEFAULT_MODEL
        self._bundle: Any = None
        self._lock = asyncio.Lock()
        # Kept for interface parity with the Ollama backend; HF chat models
        # report tool calls in-band rather than as a separate field.
        self.last_tool_calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model_id

    # ── Loading ──────────────────────────────────────────────────── #

    def _load(self) -> dict[str, Any]:
        transformers = hf.require("transformers")
        hf.require("torch")

        device = hf.resolve_device(getattr(self._settings, "hf_device", "auto"))
        kwargs = hf.load_kwargs(self._settings, device)

        log.info("LLM: loading %s on %s", self._model_id, device)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._model_id, **kwargs
        )

        model_kwargs = dict(kwargs)
        dtype = hf.torch_dtype(device)
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        if getattr(self._settings, "llm_hf_load_in_4bit", False):
            # Quantisation is what makes a 7B fit next to Whisper on an 8 GB
            # card; without bitsandbytes it is a clear message, not a crash.
            try:
                hf.require("bitsandbytes")
                model_kwargs["quantization_config"] = (
                    transformers.BitsAndBytesConfig(load_in_4bit=True)
                )
                log.info("LLM: loading %s in 4-bit", self._model_id)
            except hf.MissingDependency as exc:
                log.warning("LLM: 4-bit requested but unavailable — %s", exc)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self._model_id, **model_kwargs
        )
        if device != "cuda":
            model = model.to("cpu")
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        log.info("LLM: %s ready", self._model_id)
        return {"model": model, "tokenizer": tokenizer, "device": device}

    async def _ensure_loaded(self) -> dict[str, Any]:
        async with self._lock:
            if self._bundle is None:
                self._bundle = await asyncio.to_thread(self._load)
        return self._bundle

    async def close(self) -> None:
        """Drop the model and release GPU memory."""
        async with self._lock:
            if self._bundle is None:
                return
            self._bundle = None
        await asyncio.to_thread(hf.free_cuda)

    # ── Health ───────────────────────────────────────────────────── #

    async def health_check(self) -> dict[str, Any]:
        """Check the model can be resolved. Never raises."""
        try:
            hf.require("transformers")
            hf.require("torch")
        except hf.MissingDependency as exc:
            return {"ok": False, "error": str(exc), "models": []}

        try:
            transformers = __import__("transformers")
            transformers.AutoConfig.from_pretrained(
                self._model_id, **hf.load_kwargs(self._settings)
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": (
                    f"Cannot load '{self._model_id}' from Hugging Face ({exc}). "
                    f"Check the model id, and set hf_token for a gated repo."
                ),
                "models": [],
            }

        return {"ok": True, "error": "", "models": [self._model_id]}

    # ── Generation ───────────────────────────────────────────────── #

    def _build_prompt(self, tokenizer: Any, messages: list[dict],
                      tools: list[dict] | None) -> str:
        """Render messages with the model's own chat template when it has one."""
        if getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **({"tools": tools} if tools else {}),
                )
            except Exception as exc:
                log.debug("LLM: chat template rejected the input (%s)", exc)
                try:
                    return tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass

        # Base models without a template still deserve a coherent prompt.
        lines = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            lines.append(f"{role}: {message.get('content', '')}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion, yielding text as the model produces it."""
        import transformers

        self.last_tool_calls = []
        bundle = await self._ensure_loaded()
        model, tokenizer = bundle["model"], bundle["tokenizer"]

        prompt = self._build_prompt(tokenizer, messages, tools)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        streamer = transformers.TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": int(self._settings.llm_max_tokens),
            "do_sample": float(self._settings.llm_temperature) > 0,
            "temperature": max(0.01, float(self._settings.llm_temperature)),
            "top_p": 0.95,
            "pad_token_id": tokenizer.pad_token_id,
        }

        error: list[BaseException] = []

        def _generate() -> None:
            try:
                import torch

                with torch.no_grad():
                    model.generate(**generate_kwargs)
            except BaseException as exc:  # noqa: BLE001 — re-raised on the loop
                error.append(exc)
                streamer.end()

        await bus.emit(Event.LLM_START, data={"model": self._model_id},
                       source=self.name, priority=3)

        worker = threading.Thread(target=_generate, daemon=True, name="hf-generate")
        worker.start()

        loop = asyncio.get_running_loop()
        collected: list[str] = []
        try:
            # Draining a blocking iterator on the loop would freeze the UI for
            # the whole answer, so each next() goes to a thread.
            iterator = iter(streamer)
            while True:
                chunk = await asyncio.wait_for(
                    asyncio.to_thread(next, iterator, None),
                    timeout=FIRST_TOKEN_TIMEOUT,
                )
                if chunk is None:
                    break
                if not chunk:
                    continue
                collected.append(chunk)
                await bus.emit(Event.LLM_CHUNK, data={"chunk": chunk},
                               source=self.name, priority=3)
                yield chunk
        except TimeoutError as exc:
            raise HFLLMError(
                f"'{self._model_id}' produced no output within "
                f"{FIRST_TOKEN_TIMEOUT:.0f}s"
            ) from exc
        finally:
            await asyncio.to_thread(worker.join, 5.0)

        if error:
            raise HFLLMError(f"{self._model_id} generation failed: {error[0]}") from error[0]

        self.last_tool_calls = _parse_tool_calls("".join(collected))
        await bus.emit(Event.LLM_DONE, data={"model": self._model_id},
                       source=self.name, priority=3)
        del loop

    async def generate(self, messages: list[dict]) -> str:
        chunks = [chunk async for chunk in self.stream(messages)]
        return "".join(chunks)


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from an in-band <tool_call> block.

    Qwen, Hermes and several other instruct models emit function calls as
    JSON inside <tool_call> tags rather than in a structured field, so a
    plugin call would otherwise be dropped as ordinary prose.
    """
    import re

    calls: list[dict[str, Any]] = []
    for blob in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        name = parsed.get("name")
        if name:
            calls.append({"name": name,
                          "arguments": parsed.get("arguments") or {}})
    return calls
