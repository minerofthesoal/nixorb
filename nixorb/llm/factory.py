"""nixorb/llm/factory.py — pick an LLM backend from settings.

Before this existed, main.py always constructed ``OllamaBackend`` directly,
regardless of ``settings.llm_backend`` — the setting was logged in the
startup banner but never actually consulted, so choosing "huggingface" in
settings silently kept running Ollama (and ran Ollama's health check
against a HuggingFace repo id, producing exactly the confusing "Model 'X'
is not installed" warning this project's users ran into). This is the one
place that decision gets made now.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend
    from nixorb.llm.ollama_backend import OllamaBackend
    from nixorb.llm.openai_compat_backend import OpenAICompatBackend
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

AnyLLMBackend = Union["OllamaBackend", "HuggingFaceLLMBackend", "OpenAICompatBackend"]


def build_llm(settings: Settings) -> AnyLLMBackend:
    backend = (getattr(settings, "llm_backend", "") or "ollama").lower()

    if backend == "huggingface":
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend
        log.info("LLM: using HuggingFace backend, model '%s'", settings.llm_model)
        return HuggingFaceLLMBackend(settings)

    if backend == "openai":
        from nixorb.llm.openai_compat_backend import OpenAICompatBackend
        log.info(
            "LLM: using OpenAI-compatible backend at %s, model '%s'",
            getattr(settings, "openai_base_url", ""), settings.llm_model,
        )
        return OpenAICompatBackend(settings)

    if backend != "ollama":
        log.warning("LLM: unknown llm_backend '%s' — falling back to ollama", backend)

    from nixorb.llm.ollama_backend import OllamaBackend
    log.info("LLM: using Ollama backend, model '%s'", settings.llm_model)
    return OllamaBackend(settings)
