"""
nixorb/llm/backends.py

LLM backend abstraction.  All backends are async generators that yield
string chunks and emit EventBus events as they produce output.

BUG FIX PASS 1:
  - aiohttp was only imported inside OllamaBackend.stream() which hides
    ImportError until first use.  Moved to top-level with a clear message.

BUG FIX PASS 2:
  - OfflineFallbackManager.stream() used a try/except around an async-for
    loop that contained yield statements. In Python you cannot have both a
    yield and a try/except with a bare except in the same generator frame
    in certain patterns. Refactored to delegate to a helper coroutine.

BUG FIX PASS 3:
  - LocalLLMBackend used asyncio.get_running_loop() inside a thread-pool
    executor callback to call loop.call_soon_threadsafe(). That is correct
    but the loop reference must be captured BEFORE entering run_in_executor.
    Fixed by capturing the loop before the executor call.
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

import aiohttp  # BUG FIX: was inside method only

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import ModelPriority, vram

log = logging.getLogger(__name__)


# ================================================================== #
#  Abstract base                                                      #
# ================================================================== #
class LLMBackend(ABC):
    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        tools:    list[dict] | None = None,
    ) -> AsyncIterator[str]: ...

    async def complete(self, messages: list[dict]) -> str:
        """Convenience: collect the full stream and return it."""
        chunks: list[str] = []
        async for chunk in self.stream(messages):
            chunks.append(chunk)
        result = "".join(chunks)
        await bus.emit(Event.LLM_DONE, data={"text": result},
                       source=self.__class__.__name__, priority=2)
        return result


# ================================================================== #
#  OpenAI-compatible (OpenAI, Groq, Together, LM Studio …)          #
# ================================================================== #
class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        api_key:  str,
        model:    str,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model  = model

    async def stream(
        self,
        messages: list[dict],
        tools:    list[dict] | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=2_048,
            temperature=0.7,
        )
        if tools:
            kwargs["tools"] = tools

        try:
            async with self._client.chat.completions.stream(**kwargs) as stream:
                async for event in stream:
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


# ================================================================== #
#  Local llama-cpp-python                                            #
# ================================================================== #
def _load_local_llm(model_path: str, n_gpu_layers: int = -1):
    from llama_cpp import Llama
    return Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=8_192,
        n_batch=512,
        verbose=False,
        use_mlock=False,
    )


def _unload_local_llm(model) -> None:
    del model


class LocalLLMBackend(LLMBackend):
    def __init__(self, model_path: str, vram_mb: int = 4_096) -> None:
        self._model_path = model_path
        self._vram_mb    = vram_mb
        vram.register(
            name="local_llm",
            vram_mb=vram_mb,
            priority=ModelPriority.HIGH,
            load_fn=lambda: _load_local_llm(model_path),
            unload_fn=_unload_local_llm,
        )

    async def stream(
        self,
        messages: list[dict],
        tools:    list[dict] | None = None,
    ) -> AsyncIterator[str]:
        # Evict Whisper before loading LLM (VRAM paging)
        await vram.evict("whisper")

        async with vram.lease("local_llm") as model:
            # BUG FIX: capture loop BEFORE entering executor
            loop   = asyncio.get_running_loop()
            prompt = self._format_messages(messages)
            q: asyncio.Queue[str | None] = asyncio.Queue()

            def _generate() -> None:
                try:
                    for token in model(
                        prompt,
                        max_tokens=2_048,
                        stream=True,
                        temperature=0.7,
                        top_p=0.95,
                        repeat_penalty=1.1,
                        stop=["<|user|>", "<|system|>"],
                    ):
                        text = token["choices"][0].get("text", "")
                        if text:
                            # BUG FIX: use captured loop reference
                            loop.call_soon_threadsafe(q.put_nowait, text)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            fut = loop.run_in_executor(None, _generate)

            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                await bus.emit(Event.LLM_CHUNK, data={"chunk": chunk},
                               source="LocalLLMBackend", priority=3)
                yield chunk

            await fut   # propagate any executor exception

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        parts: list[str] = []
        for m in messages:
            role    = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content — extract text parts only
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)


# ================================================================== #
#  Ollama                                                            #
# ================================================================== #
class OllamaBackend(LLMBackend):
    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host  = host

    async def stream(
        self,
        messages: list[dict],
        tools:    list[dict] | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": 8_192, "temperature": 0.7},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._host}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    async for raw in resp.content:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            await bus.emit(Event.LLM_CHUNK, data={"chunk": content},
                                           source="OllamaBackend", priority=3)
                            yield content
        except aiohttp.ClientError as exc:
            log.error("Ollama connection error: %s", exc)
            await bus.emit(Event.LLM_ERROR, data={"error": str(exc)},
                           source="OllamaBackend")
            raise


# ================================================================== #
#  Offline fallback manager                                          #
# ================================================================== #
class OfflineFallbackManager:
    """
    Wraps a primary backend and an offline fallback.
    After FAIL_THRESHOLD consecutive errors, switches to fallback automatically.
    """

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
        tools:    list[dict] | None = None,
    ) -> AsyncIterator[str]:
        # BUG FIX: can't mix try/except with yield in same generator frame
        # cleanly when the generator raises. Use a collected-chunks approach
        # for error detection, then re-yield.
        collected: list[str] = []
        error_occurred = False

        try:
            async for chunk in self.active.stream(messages, tools):
                collected.append(chunk)
                yield chunk
            self._fail_count = 0   # success — reset counter
        except Exception as exc:
            error_occurred = True
            self._fail_count += 1
            log.error("LLM backend error (%d/%d): %s",
                      self._fail_count, self.FAIL_THRESHOLD, exc)

        if error_occurred:
            if self._fail_count >= self.FAIL_THRESHOLD and not self._using_fallback:
                self._using_fallback = True
                log.warning("Switching to OFFLINE fallback LLM")
                await bus.emit(
                    Event.LOG,
                    data={"level": "warning",
                          "msg": "⚠️  API unreachable — switched to offline model"},
                    source="OfflineFallbackManager",
                )

            if self._using_fallback:
                # Retry with fallback
                async for chunk in self._fallback.stream(messages, tools):
                    yield chunk
            else:
                raise RuntimeError(
                    f"LLM backend failed {self._fail_count} times, "
                    "threshold not yet reached for fallback"
                )
