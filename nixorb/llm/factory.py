"""Choose an LLM backend from settings.

    llm_backend = "ollama"        # default — local daemon, no Python deps
    llm_backend = "huggingface"   # any causal LM on the Hub, in-process
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_ALIASES = {
    "hf": "huggingface",
    "transformers": "huggingface",
    "local": "huggingface",
}

BACKENDS = ("ollama", "huggingface")


class LLMBackend(Protocol):
    """What main.py needs from any language model backend."""

    last_tool_calls: list[dict[str, Any]]

    @property
    def model(self) -> str:
        """The model id actually in use, for logs and `nixorb status`."""
        ...

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]: ...

    async def generate(self, messages: list[dict]) -> str: ...

    async def health_check(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


def normalise_backend(name: str | None) -> str:
    key = (name or "ollama").strip().lower()
    return _ALIASES.get(key, key)


def create_llm(settings: Settings) -> LLMBackend:
    """Build the configured LLM backend."""
    backend = normalise_backend(getattr(settings, "llm_backend", None))

    if backend == "huggingface":
        from nixorb.llm.hf_backend import HuggingFaceBackend

        return HuggingFaceBackend(settings)

    if backend != "ollama":
        log.warning(
            "LLM: unknown backend %r (choose one of %s) — using ollama",
            settings.llm_backend, ", ".join(BACKENDS),
        )

    from nixorb.llm.ollama_backend import OllamaBackend

    return OllamaBackend(settings)


class OfflineFallbackManager:
    """Switch to a second backend after the first fails repeatedly.

    Useful when the primary is a network-dependent daemon and the fallback
    is an in-process model (or the other way round).
    """

    FAIL_THRESHOLD = 3

    def __init__(self, primary: Any, fallback: Any) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fail_count = 0
        self._using_fallback = False
        self.last_tool_calls: list[dict[str, Any]] = []

    @property
    def active(self) -> Any:
        return self._fallback if self._using_fallback else self._primary

    @property
    def model(self) -> str:
        return str(getattr(self.active, "model", "unknown"))

    async def health_check(self) -> dict[str, Any]:
        return await self.active.health_check()

    async def close(self) -> None:
        for backend in (self._primary, self._fallback):
            close = getattr(backend, "close", None)
            if close is not None:
                await close()

    async def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[str]:
        failed = False
        try:
            async for chunk in self.active.stream(messages, tools):
                yield chunk
            self._fail_count = 0
        except Exception as exc:
            failed = True
            self._fail_count += 1
            log.error(
                "LLM backend error (%d/%d): %s",
                self._fail_count, self.FAIL_THRESHOLD, exc,
            )

        if not failed:
            self.last_tool_calls = list(
                getattr(self.active, "last_tool_calls", [])
            )
            return

        if self._fail_count >= self.FAIL_THRESHOLD and not self._using_fallback:
            self._using_fallback = True
            log.warning("Switching to the fallback LLM backend")
            await bus.emit(
                Event.LOG,
                data={"level": "warning",
                      "msg": "⚠️ Primary model unreachable — switched to fallback"},
                source="OfflineFallbackManager",
            )

        if self._using_fallback:
            async for chunk in self._fallback.stream(messages, tools):
                yield chunk
            self.last_tool_calls = list(
                getattr(self._fallback, "last_tool_calls", [])
            )

    async def generate(self, messages: list[dict]) -> str:
        return "".join([chunk async for chunk in self.stream(messages)])
