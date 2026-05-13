"""nixorb/llm/prompt_builder.py — Chat template / prompt formatting utilities."""
from __future__ import annotations

import re

GLADOS_PREFIX = (
    "[GLaDOS voice, dry wit, precise, occasionally sardonic] "
)


def build_messages(
    transcript: str,
    system_prompt: str,
    history: list[dict],
    memory_context: str = "",
    web_context: str = "",
    screen_context: str = "",
    clipboard_context: str = "",
) -> list[dict]:
    """
    Assemble the full messages list for a single turn.
    Context blocks are injected into the user message in order.
    """
    user_parts: list[str] = []

    if memory_context:
        user_parts.append(memory_context)
    if web_context:
        user_parts.append(web_context)
    if screen_context:
        user_parts.append(f"[Screen context]\n{screen_context}")
    if clipboard_context:
        user_parts.append(f"[Clipboard]\n{clipboard_context}")

    user_parts.append(transcript)
    user_content = "\n\n".join(user_parts)

    messages = [{"role": "system", "content": system_prompt}, *history,
                {"role": "user", "content": user_content}]
    return messages


def extract_action_blocks(text: str) -> list[str]:
    """Return all bash commands from <ACTION>...</ACTION> blocks."""
    return [m.strip() for m in re.findall(
        r"<ACTION>(.*?)</ACTION>", text, re.DOTALL | re.IGNORECASE
    )]


def strip_action_blocks(text: str) -> str:
    """Remove ACTION blocks from text (for TTS)."""
    return re.sub(r"<ACTION>.*?</ACTION>", "", text, flags=re.DOTALL).strip()


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return list of (language, code) from fenced code blocks."""
    return re.findall(r"```(\w*)\n(.*?)```", text, re.DOTALL)


def truncate_for_tts(text: str, max_sentences: int = 6) -> str:
    """Limit TTS to first N sentences to avoid very long responses."""
    clean    = strip_action_blocks(text)
    clean    = re.sub(r"```.*?```", "[code block]", clean, flags=re.DOTALL)
    sentences = re.split(r"(?<=[.!?])\s+", clean.strip())
    return " ".join(sentences[:max_sentences])


def format_results_for_llm(results: list) -> str:
    """Format ActionResult list for LLM consumption."""
    if not results:
        return ""
    parts = [str(r) for r in results]
    return "<RESULT>\n" + "\n\n".join(parts) + "\n</RESULT>"


def trim_history(
    history: list[dict],
    max_turns: int = 20,
    system: dict | None = None,
) -> list[dict]:
    """Keep conversation within token budget."""
    if len(history) <= max_turns:
        return history
    trimmed = history[-max_turns:]
    if system:
        return [system] + trimmed
    return trimmed
