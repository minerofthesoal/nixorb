"""nixorb/llm/openai_compat_backend.py — any OpenAI-compatible chat endpoint.

Covers real OpenAI, and every local server that serves an arbitrary
HuggingFace model behind an OpenAI-shaped API: vLLM, text-generation-
inference, LM Studio, llama.cpp's own ``server`` binary, etc. Pointing
``openai_base_url`` at one of those is often the most practical way to run
a "custom HuggingFace model" that's too large or needs batching beyond what
loading it in-process (``HuggingFaceLLMBackend``) is good for.

Mirrors ``OllamaBackend``'s interface for the same reason ``hf_llm_backend``
does: main.py's tool-calling loop doesn't need to know which backend it's
talking to.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import aiohttp

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 10.0
HEALTH_TIMEOUT = 8.0
FIRST_TOKEN_TIMEOUT = 120.0
INTER_TOKEN_TIMEOUT = 60.0


class OpenAICompatError(RuntimeError):
    """Raised when the endpoint is unreachable or returns an error."""


class OpenAICompatBackend:
    """Streaming chat client for any OpenAI-compatible /chat/completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_url = str(getattr(settings, "openai_base_url", "") or
                              "https://api.openai.com/v1").rstrip("/")
        self._api_key = getattr(settings, "openai_api_key", "") or ""
        self._model = settings.llm_model
        self._session: aiohttp.ClientSession | None = None
        self.last_tool_calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _session_for_loop(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=CONNECT_TIMEOUT, sock_read=INTER_TOKEN_TIMEOUT
                )
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def health_check(self) -> dict[str, Any]:
        """Check the endpoint is reachable. Most local servers (vLLM, LM
        Studio, llama.cpp server) expose /models; real OpenAI does too."""
        url = f"{self._base_url}/models"
        try:
            session = await self._session_for_loop()
            async with session.get(
                url, headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    return {
                        "ok": False,
                        "error": f"{self._base_url} returned HTTP {resp.status}",
                        "models": [],
                    }
                body = await resp.json()
        except aiohttp.ClientError as exc:
            return {
                "ok": False,
                "error": f"Cannot reach {self._base_url} ({exc})",
                "models": [],
            }
        except TimeoutError:
            return {
                "ok": False,
                "error": f"{self._base_url} did not respond within {HEALTH_TIMEOUT:.0f}s",
                "models": [],
            }
        except Exception as exc:
            return {"ok": False, "error": f"Health check failed: {exc}", "models": []}

        models = [m.get("id", "") for m in body.get("data", [])]
        if not models or any(m == self._model for m in models):
            return {"ok": True, "error": "", "models": models}
        return {
            "ok": False,
            "error": f"Model '{self._model}' not listed by {self._base_url} "
                     f"(available: {', '.join(models)})",
            "models": models,
        }

    def _payload(self, messages: list[dict], tools: list[dict] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": float(self._settings.llm_temperature),
            "max_tokens": int(self._settings.llm_max_tokens),
        }
        if tools:
            payload["tools"] = tools
        return payload

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        url = f"{self._base_url}/chat/completions"
        self.last_tool_calls = []
        # Accumulates streamed tool_call deltas by index, per the OpenAI
        # streaming tool-call format (name/arguments arrive in fragments
        # across multiple chunks, keyed by an "index" the server assigns).
        tool_call_acc: dict[int, dict[str, Any]] = {}

        await bus.emit(Event.LLM_START, data={"model": self._model},
                       source="OpenAICompatBackend", priority=3)
        try:
            session = await self._session_for_loop()
            async with session.post(
                url, json=self._payload(messages, tools), headers=self._headers(),
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=CONNECT_TIMEOUT, sock_read=FIRST_TOKEN_TIMEOUT,
                ),
            ) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:400]
                    raise OpenAICompatError(
                        f"{self._base_url} HTTP {resp.status}: {detail or 'no detail'}"
                    )

                async for raw in resp.content:
                    line = raw.strip()
                    if not line or not line.startswith(b"data:"):
                        continue
                    data = line[len(b"data:"):].strip()
                    if data == b"[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("error"):
                        raise OpenAICompatError(str(obj["error"]))

                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {}) or {}

                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        acc = tool_call_acc.setdefault(
                            idx, {"name": "", "arguments": ""}
                        )
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            acc["name"] += fn["name"]
                        if fn.get("arguments"):
                            acc["arguments"] += fn["arguments"]

                    chunk = delta.get("content") or ""
                    if chunk:
                        await bus.emit(
                            Event.LLM_CHUNK, data={"chunk": chunk},
                            source="OpenAICompatBackend", priority=3,
                        )
                        yield chunk

                    if choices[0].get("finish_reason"):
                        break

        except OpenAICompatError:
            raise
        except aiohttp.ClientError as exc:
            raise OpenAICompatError(f"Cannot reach {self._base_url} ({exc})") from exc
        except TimeoutError as exc:
            raise OpenAICompatError(
                f"{self._base_url} timed out while generating with '{self._model}'"
            ) from exc

        for acc in tool_call_acc.values():
            if not acc["name"]:
                continue
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            self.last_tool_calls.append({"name": acc["name"], "arguments": args})

        await bus.emit(Event.LLM_DONE, data={"model": self._model},
                       source="OpenAICompatBackend", priority=3)

    async def generate(self, messages: list[dict]) -> str:
        chunks: list[str] = []
        async for chunk in self.stream(messages):
            chunks.append(chunk)
        return "".join(chunks)
