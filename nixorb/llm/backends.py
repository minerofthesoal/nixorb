"""nixorb/llm/backends.py — LLM backend abstraction."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import aiohttp

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import ModelPriority, vram

log = logging.getLogger(__name__)


class LLMBackend(ABC):
    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete(self, messages: list[dict]) -> str:
        chunks: list[str] = []
        async for chunk in self.stream(messages):
            chunks.append(chunk)
        result = "".join(chunks)
        await bus.emit(Event.LLM_DONE, data={"text": result},
                       source=self.__class__.__name__, priority=2)
        return result


# ── HuggingFace transformers backend ─────────────────────────────── #
class HuggingFaceBackend(LLMBackend):
    """
    Local HuggingFace model via transformers.generate() with streaming.
    Works for: stablelm, gemma, qwen, phi, and most causal LMs.
    """

    def __init__(self, model_id: str, token: str = "", vram_mb: int = 4096,
                 max_new_tokens: int = 512) -> None:
        self._model_id      = model_id
        self._token         = token or None
        self._max_new_tokens = max_new_tokens
        vram.register(
            name="hf_llm",
            vram_mb=vram_mb,
            priority=ModelPriority.HIGH,
            load_fn=lambda: self._load(),
            unload_fn=lambda obj: self._unload_model(obj),
        )

    def _load(self) -> dict:
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        log.info("Loading HF model: %s", self._model_id)
        tok = AutoTokenizer.from_pretrained(
            self._model_id, token=self._token, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            token=self._token,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
        log.info("HF model loaded: %s", self._model_id)
        return {"model": model, "tokenizer": tok, "streamer_cls": TextIteratorStreamer}

    @staticmethod
    def _unload_model(obj: dict) -> None:
        del obj["model"]
        del obj["tokenizer"]

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        import threading
        await vram.evict("whisper")
        loop = asyncio.get_running_loop()

        async with vram.lease("hf_llm") as obj:
            model     = obj["model"]
            tokenizer = obj["tokenizer"]
            StreamerCls = obj["streamer_cls"]

            # Build prompt using chat template if available
            try:
                prompt_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
            except Exception:
                # Fallback: simple concatenation
                text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in messages
                ) + "\nASSISTANT:"
                prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

            streamer = StreamerCls(tokenizer, skip_prompt=True, skip_special_tokens=True)
            q: asyncio.Queue[str | None] = asyncio.Queue()

            def _generate() -> None:
                import torch
                with torch.no_grad():
                    model.generate(
                        prompt_ids,
                        max_new_tokens=self._max_new_tokens,
                        streamer=streamer,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                loop.call_soon_threadsafe(q.put_nowait, None)

            for token_text in streamer:
                pass  # consumed by thread; we read from queue

            # Reset streamer state and run generate in thread
            gen_thread = threading.Thread(target=_generate, daemon=True)
            # Re-create streamer each call
            streamer = StreamerCls(tokenizer, skip_prompt=True, skip_special_tokens=True)

            def _generate_and_stream() -> None:
                import torch
                with torch.no_grad():
                    model.generate(
                        prompt_ids,
                        max_new_tokens=self._max_new_tokens,
                        streamer=streamer,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # streamer yields in the for loop below via __iter__

            def _stream_to_queue() -> None:
                import torch
                new_streamer = StreamerCls(tokenizer, skip_prompt=True, skip_special_tokens=True)
                with torch.no_grad():
                    model.generate(
                        prompt_ids,
                        max_new_tokens=self._max_new_tokens,
                        streamer=new_streamer,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                for tok_text in new_streamer:
                    loop.call_soon_threadsafe(q.put_nowait, tok_text)
                loop.call_soon_threadsafe(q.put_nowait, None)

            t = threading.Thread(target=_stream_to_queue, daemon=True)
            t.start()

            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                await bus.emit(Event.LLM_CHUNK, data={"chunk": chunk},
                               source="HuggingFaceBackend", priority=3)
                yield chunk


# ── OpenAI-compatible ─────────────────────────────────────────────── #
class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1") -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model  = model

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=self._model, messages=messages, stream=True, max_tokens=2048
        )
        if tools:
            kwargs["tools"] = tools
        try:
            async with self._client.chat.completions.stream(**kwargs) as s:
                async for event in s:
                    if not event.choices:
                        continue
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        chunk = delta.content
                        await bus.emit(Event.LLM_CHUNK, data={"chunk": chunk},
                                       source="OpenAIBackend", priority=3)
                        yield chunk
        except Exception as exc:
            log.error("OpenAI stream error: %s", exc)
            await bus.emit(Event.LLM_ERROR, data={"error": str(exc)},
                           source="OpenAIBackend")
            raise


# ── Local llama.cpp ───────────────────────────────────────────────── #
class LocalLLMBackend(LLMBackend):
    def __init__(self, model_path: str, vram_mb: int = 4096) -> None:
        self._model_path = model_path
        vram.register(
            name="local_llm",
            vram_mb=vram_mb,
            priority=ModelPriority.HIGH,
            load_fn=lambda: self._load_llama(model_path),
            unload_fn=lambda m: None,
        )

    @staticmethod
    def _load_llama(path: str):
        from llama_cpp import Llama
        return Llama(model_path=path, n_gpu_layers=-1, n_ctx=8192, verbose=False)

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        await vram.evict("whisper")
        loop = asyncio.get_running_loop()
        async with vram.lease("local_llm") as model:
            prompt = "\n".join(
                f"{m['role'].upper()}: {m.get('content','')}" for m in messages
            ) + "\nASSISTANT:"
            q: asyncio.Queue[str | None] = asyncio.Queue()

            def _gen() -> None:
                for tok in model(prompt, max_tokens=2048, stream=True):
                    text = tok["choices"][0].get("text", "")
                    if text:
                        loop.call_soon_threadsafe(q.put_nowait, text)
                loop.call_soon_threadsafe(q.put_nowait, None)

            t_task = loop.run_in_executor(None, _gen)
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                await bus.emit(Event.LLM_CHUNK, data={"chunk": chunk},
                               source="LocalLLMBackend", priority=3)
                yield chunk
            await t_task


# ── Ollama ───────────────────────────────────────────────────────── #
class OllamaBackend(LLMBackend):
    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host  = host

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        payload = {"model": self._model, "messages": messages, "stream": True}
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self._host}/api/chat", json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp,
            ):
                resp.raise_for_status()
                async for raw in resp.content:
                    if not raw.strip():
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        obj     = json.loads(raw)
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            await bus.emit(Event.LLM_CHUNK, data={"chunk": content},
                                           source="OllamaBackend", priority=3)
                            yield content
        except aiohttp.ClientError as exc:
            log.error("Ollama error: %s", exc)
            raise



# ── Offline fallback manager ──────────────────────────────────────── #
class OfflineFallbackManager:
    FAIL_THRESHOLD = 3

    def __init__(self, primary: LLMBackend, fallback: LLMBackend) -> None:
        self._primary        = primary
        self._fallback       = fallback
        self._fail_count     = 0
        self._using_fallback = False

    @property
    def active(self) -> LLMBackend:
        return self._fallback if self._using_fallback else self._primary

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        error_occurred = False
        try:
            async for chunk in self.active.stream(messages, tools):
                yield chunk
            self._fail_count = 0
        except Exception as exc:
            error_occurred = True
            self._fail_count += 1
            log.error("LLM backend error (%d/%d): %s",
                      self._fail_count, self.FAIL_THRESHOLD, exc)

        if error_occurred:
            if self._fail_count >= self.FAIL_THRESHOLD and not self._using_fallback:
                self._using_fallback = True
                log.warning("Switching to offline fallback LLM")
                await bus.emit(
                    Event.LOG,
                    data={"level": "warning",
                          "msg": "⚠️  API unreachable — switched to offline model"},
                    source="OfflineFallbackManager",
                )
            if self._using_fallback:
                async for chunk in self._fallback.stream(messages, tools):
                    yield chunk
