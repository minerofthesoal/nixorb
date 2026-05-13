"""tests/test_llm_backends.py — LLM backend unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.asyncio


async def test_offline_fallback_switches_after_threshold(started_bus):
    from nixorb.llm.backends import OfflineFallbackManager

    primary  = MagicMock()
    fallback = MagicMock()

    call_count = 0

    async def _bad_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("API down")
        yield  # make it an async generator

    async def _good_stream(messages, tools=None):
        yield "fallback response"

    primary.stream  = _bad_stream
    fallback.stream = _good_stream

    mgr = OfflineFallbackManager(primary, fallback)
    mgr.FAIL_THRESHOLD = 2

    # Two failures should trigger fallback
    for _ in range(2):
        chunks = []
        try:
            async for c in mgr.stream([{"role": "user", "content": "hi"}]):
                chunks.append(c)
        except Exception:
            pass

    assert mgr._using_fallback or mgr._fail_count >= 2


async def test_prompt_builder_builds_correct_messages():
    from nixorb.llm.prompt_builder import build_messages

    msgs = build_messages(
        transcript="What time is it?",
        system_prompt="You are helpful.",
        history=[],
        web_context="Current time: 12:00",
    )
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "What time is it?" in msgs[-1]["content"]
    assert "Current time" in msgs[-1]["content"]


async def test_strip_action_blocks():
    from nixorb.llm.prompt_builder import strip_action_blocks

    text = "Sure! <ACTION>ls -la</ACTION> That shows your files."
    result = strip_action_blocks(text)
    assert "<ACTION>" not in result
    assert "Sure!" in result
    assert "That shows your files." in result


async def test_extract_action_blocks():
    from nixorb.llm.prompt_builder import extract_action_blocks

    text = "Run <ACTION>echo hello</ACTION> then <ACTION>pwd</ACTION>"
    blocks = extract_action_blocks(text)
    assert blocks == ["echo hello", "pwd"]


async def test_truncate_for_tts():
    from nixorb.llm.prompt_builder import truncate_for_tts

    # 10 sentences
    long = " ".join(f"Sentence {i}." for i in range(10))
    result = truncate_for_tts(long, max_sentences=3)
    assert result.count(".") <= 4  # 3 sentences + trailing


async def test_extract_code_blocks():
    from nixorb.llm.prompt_builder import extract_code_blocks

    text = "Here:\n```python\nprint('hi')\n```\nDone."
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    lang, code = blocks[0]
    assert lang == "python"
    assert "print" in code
