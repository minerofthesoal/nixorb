"""NixOrb Ollama backend — local LLM over the Ollama HTTP API.

This is the only LLM backend NixOrb v2 uses: everything runs locally
against `ollama serve` (default http://localhost:11434), so there are no
API keys and no network egress.

Design notes:
  * One shared aiohttp session, created lazily *inside* the running loop —
    building a ClientSession at import/construction time binds it to the
    wrong loop and raises at first use.
  * Streaming uses a socket-read timeout rather than a total timeout, so a
    genuinely long answer is not killed halfway while a dead connection
    still fails fast.
  * Every failure path raises OllamaError with a message that says what to
    actually do about it, because these surface in the orb's log panel.
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

# Connecting to a local daemon should be instant; failing fast here is what
# turns "the orb is frozen" into "Ollama isn't running".
CONNECT_TIMEOUT = 5.0
HEALTH_TIMEOUT = 5.0
# Ollama may have to pull a cold model into VRAM before the first token.
FIRST_TOKEN_TIMEOUT = 120.0
# Gap between tokens once generation has started.
INTER_TOKEN_TIMEOUT = 60.0


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an error."""


class OllamaBackend:
    """Streaming chat client for a local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._host = str(settings.ollama_host).rstrip("/")
        self._model = settings.llm_model
        self._session: aiohttp.ClientSession | None = None
        # Tool calls returned by the most recent stream(). Ollama reports
        # these out-of-band from the text, and dropping them meant plugins
        # were advertised to the model and then silently ignored.
        self.last_tool_calls: list[dict[str, Any]] = []

    # ── session lifecycle ────────────────────────────────────────── #

    async def _session_for_loop(self) -> aiohttp.ClientSession:
        """Return the shared session, creating it on first use."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    connect=CONNECT_TIMEOUT,
                    sock_read=INTER_TOKEN_TIMEOUT,
                )
            )
        return self._session

    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ── health ───────────────────────────────────────────────────── #

    async def health_check(self) -> dict[str, Any]:
        """Check that Ollama is up and the configured model is present.

        Returns a dict with ``ok``, plus ``error`` / ``models`` for callers
        that want to tell the user how to fix things. Never raises.
        """
        url = f"{self._host}/api/tags"
        try:
            session = await self._session_for_loop()
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=HEALTH_TIMEOUT)
            ) as resp:
                if resp.status != 200:
                    return {
                        "ok": False,
                        "error": f"Ollama returned HTTP {resp.status} from {url}",
                        "models": [],
                    }
                body = await resp.json()
        except aiohttp.ClientError as exc:
            return {
                "ok": False,
                "error": (
                    f"Cannot reach Ollama at {self._host} ({exc}). "
                    f"Start it with: ollama serve"
                ),
                "models": [],
            }
        except TimeoutError:
            return {
                "ok": False,
                "error": f"Ollama at {self._host} did not respond within "
                         f"{HEALTH_TIMEOUT:.0f}s",
                "models": [],
            }
        except Exception as exc:  # malformed JSON, etc.
            return {"ok": False, "error": f"Ollama health check failed: {exc}",
                    "models": []}

        models = [m.get("name", "") for m in body.get("models", [])]
        # Ollama reports "llama3.2:latest" for a model pulled as "llama3.2".
        wanted = self._model
        if any(m == wanted or m.split(":")[0] == wanted.split(":")[0] for m in models):
            return {"ok": True, "error": "", "models": models}

        return {
            "ok": False,
            "error": (
                f"Model '{wanted}' is not installed "
                f"(available: {', '.join(models) or 'none'})"
            ),
            "models": models,
        }

    # ── generation ───────────────────────────────────────────────── #

    def _payload(self, messages: list[dict], tools: list[dict] | None,
                 stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": float(self._settings.llm_temperature),
                "num_predict": int(self._settings.llm_max_tokens),
            },
        }
        if tools:
            payload["tools"] = tools
        return payload

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion, yielding text chunks as they arrive."""
        url = f"{self._host}/api/chat"
        self.last_tool_calls = []
        await bus.emit(Event.LLM_START, data={"model": self._model},
                       source="OllamaBackend", priority=3)

        try:
            session = await self._session_for_loop()
            async with session.post(
                url,
                json=self._payload(messages, tools, stream=True),
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    connect=CONNECT_TIMEOUT,
                    sock_read=FIRST_TOKEN_TIMEOUT,
                ),
            ) as resp:
                if resp.status != 200:
                    detail = (await resp.text())[:400]
                    raise OllamaError(
                        f"Ollama HTTP {resp.status}: {detail or 'no detail'}"
                    )

                async for raw in resp.content:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        log.debug("Ollama: skipping non-JSON line %r", line[:120])
                        continue

                    # Ollama reports generation errors in-band, mid-stream.
                    if obj.get("error"):
                        raise OllamaError(str(obj["error"]))

                    message = obj.get("message", {}) or {}

                    for call in message.get("tool_calls") or []:
                        fn = call.get("function", {}) or {}
                        if fn.get("name"):
                            self.last_tool_calls.append(fn)

                    chunk = message.get("content", "")
                    if chunk:
                        await bus.emit(
                            Event.LLM_CHUNK, data={"chunk": chunk},
                            source="OllamaBackend", priority=3,
                        )
                        yield chunk

                    if obj.get("done"):
                        break

        except OllamaError:
            raise
        except aiohttp.ClientError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self._host} ({exc}). "
                f"Start it with: ollama serve"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama timed out while generating with '{self._model}'"
            ) from exc

        await bus.emit(Event.LLM_DONE, data={"model": self._model},
                       source="OllamaBackend", priority=3)

    async def generate(self, messages: list[dict]) -> str:
        """Run a full (non-incremental) completion and return the text."""
        chunks: list[str] = []
        async for chunk in self.stream(messages):
            chunks.append(chunk)
        return "".join(chunks)

    # Kept so callers can log which model actually answered.
    @property
    def model(self) -> str:
        return self._model
