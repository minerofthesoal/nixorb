"""
nixorb/llm/backend_base.py  +  local_backend.py  +  openai_backend.py
+  offline_fallback.py  (multi-file merged for brevity)

LLM backend abstraction layer. Each backend yields streaming text chunks
and emits LLM_CHUNK events on the EventBus.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from typing import AsyncIterator

from nixorb.core.event_bus import Event, bus
from nixorb.core.vram_manager import ModelPriority, vram

log = logging.getLogger(__name__)


# ============================================================ #
#  Base                                                        #
# ============================================================ #
class LLMBackend(ABC):
    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        ...

    async def _emit_chunks(self, iterator: AsyncIterator[str]) -> str:
        full = []
        async for chunk in iterator:
            full.append(chunk)
            await bus.emit(Event.LLM_CHUNK, data={"chunk": chunk},
                           source=self.__class__.__name__, priority=3)
        result = "".join(full)
        await bus.emit(Event.LLM_DONE, data={"text": result},
                       source=self.__class__.__name__, priority=2)
        return result


# ============================================================ #
#  OpenAI-compatible backend (OpenAI, Together, Groq, LM Studio)
# ============================================================ #
class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.openai.com/v1") -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model  = model

    async def stream(self, messages: list[dict],
                     tools: list[dict] | None = None) -> AsyncIterator[str]:
        kwargs: dict = dict(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=2048,
        )
        if tools:
            kwargs["tools"] = tools

        async with self._client.chat.completions.stream(**kwargs) as s:
            async for event in s:
                delta = event.choices[0].delta if event.choices else None
                if delta and delta.content:
                    yield delta.content


# ============================================================ #
#  Local backend via llama-cpp-python                          #
# ============================================================ #
def _load_local_llm(model_path: str, n_gpu_layers: int = -1):
    from llama_cpp import Llama
    return Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=8192,
        n_batch=512,
        verbose=False,
        use_mlock=False,
    )


def _unload_local_llm(model) -> None:
    del model


class LocalLLMBackend(LLMBackend):
    def __init__(self, model_path: str, vram_mb: int = 4096) -> None:
        self._model_path = model_path
        vram.register(
            name="local_llm",
            vram_mb=vram_mb,
            priority=ModelPriority.HIGH,
            load_fn=lambda: _load_local_llm(model_path),
            unload_fn=_unload_local_llm,
        )

    async def stream(self, messages: list[dict],
                     tools: list[dict] | None = None) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()

        async with vram.lease("local_llm") as model:
            # Evict Whisper before loading LLM (VRAM paging)
            await vram.evict("whisper")

            prompt = self._format_messages(messages)
            q: asyncio.Queue[str | None] = asyncio.Queue()

            def _gen():
                for chunk in model(
                    prompt,
                    max_tokens=2048,
                    stream=True,
                    temperature=0.7,
                    top_p=0.95,
                    repeat_penalty=1.1,
                ):
                    text = chunk["choices"][0]["text"]
                    if text:
                        loop.call_soon_threadsafe(q.put_nowait, text)
                loop.call_soon_threadsafe(q.put_nowait, None)

            thread = asyncio.get_running_loop().run_in_executor(None, _gen)

            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk

            await thread

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)


# ============================================================ #
#  Ollama backend                                              #
# ============================================================ #
class OllamaBackend(LLMBackend):
    """Streams from a running Ollama daemon (no VRAM management needed)."""

    def __init__(self, model: str, host: str = "http://localhost:11434") -> None:
        self._model = model
        self._host  = host

    async def stream(self, messages: list[dict],
                     tools: list[dict] | None = None) -> AsyncIterator[str]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {"num_ctx": 8192},
            }
            async with session.post(
                f"{self._host}/api/chat", json=payload
            ) as resp:
                async for line in resp.content:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    content = obj.get("message", {}).get("content", "")
                    if content:
                        yield content


# ============================================================ #
#  Offline fallback manager                                    #
# ============================================================ #
class OfflineFallbackManager:
    """
    Monitors API availability. If OpenAI/HF calls fail ≥3 times,
    automatically switches to the local backend.
    """

    def __init__(self, primary: LLMBackend, fallback: LLMBackend) -> None:
        self._primary  = primary
        self._fallback = fallback
        self._fail_count = 0
        self._using_fallback = False
        self.FAIL_THRESHOLD = 3

    @property
    def active(self) -> LLMBackend:
        return self._fallback if self._using_fallback else self._primary

    async def stream(self, messages: list[dict],
                     tools: list[dict] | None = None) -> AsyncIterator[str]:
        try:
            async for chunk in self.active.stream(messages, tools):
                yield chunk
            self._fail_count = 0
        except Exception as exc:
            self._fail_count += 1
            log.error("LLM backend error (%d/%d): %s",
                      self._fail_count, self.FAIL_THRESHOLD, exc)
            if self._fail_count >= self.FAIL_THRESHOLD and not self._using_fallback:
                log.warning("Switching to OFFLINE FALLBACK LLM")
                self._using_fallback = True
                await bus.emit(
                    Event.LOG,
                    data={"level": "warning",
                          "msg": "⚠️ API unreachable — using local offline model"},
                    source="OfflineFallbackManager",
                )
            if self._using_fallback:
                async for chunk in self._fallback.stream(messages, tools):
                    yield chunk
            else:
                raise
