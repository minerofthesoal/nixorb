"""tests/test_tts_fallback.py — regression tests for the TTS silent-failure bug.

Before this fix, the default config (tts_backend="huggingface",
tts_hf_repo="torphix/stablelm-2-glados-v1") loaded a text-only causal-LM
checkpoint into transformers' text-to-speech pipeline, which always raised,
was swallowed by a bare `except Exception: log.exception(...)`, and never
produced audio — with nothing visible anywhere since NixOrb usually runs
without an attached terminal.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_default_settings_use_a_real_tts_repo():
    from nixorb.settings import Settings

    s = Settings()
    # Regression guard: this must be an id that transformers' text-to-speech
    # pipeline can actually load, not a repurposed chat-LLM checkpoint.
    assert s.tts_hf_repo == "microsoft/speecht5_tts"
    assert s.tts_backend == "glados"


@pytest.mark.asyncio
async def test_hf_tts_falls_back_when_repo_is_not_a_tts_model(started_bus):
    """If tts_hf_repo points at something the TTS pipeline can't load,
    HuggingFaceTTS must fall back to SpeechT5 instead of failing silently.

    NOTE: `transformers` uses a lazy-loading module (`_LazyModule`) for its
    top-level namespace, so `patch("transformers.pipeline", ...)` does not
    reliably intercept calls — the real function lives in
    `transformers.pipelines.pipeline` and that's what must be patched.
    """
    from nixorb.tts.hf_tts import _load_hf_tts

    def _boom(*_a, **_kw):
        raise OSError("not a valid text-to-speech model")

    fake_speecht5 = ("speecht5", object())

    with (
        patch("transformers.pipelines.pipeline", side_effect=_boom),
        patch("nixorb.tts.glados_tts._load_speecht5", return_value=fake_speecht5) as mock_fb,
    ):
        kind, obj = _load_hf_tts("some/text-only-model", None)

    mock_fb.assert_called_once()
    assert kind == "speecht5"
    assert obj is fake_speecht5[1]


@pytest.mark.asyncio
async def test_hf_tts_reports_failure_on_the_visible_bus_log(started_bus):
    """Even the fallback path can fail (e.g. no network) — that must show up
    on Event.LOG (the Settings > Log tab), not just the hidden python logger."""
    from nixorb.core.event_bus import Event
    from nixorb.tts.hf_tts import HuggingFaceTTS

    class _Settings:
        tts_hf_repo = "microsoft/speecht5_tts"
        hf_token    = ""
        tts_voice   = "alloy"

    seen = []

    async def _capture(payload):
        seen.append(payload.data or {})

    started_bus.subscribe(Event.LOG, _capture)

    tts = HuggingFaceTTS(_Settings())

    with patch("nixorb.core.vram_manager.vram.lease", side_effect=RuntimeError("boom")):
        await tts.speak("hello there")

    # tts.speak() only enqueues the LOG event (bus.emit awaits queue.put,
    # which returns almost immediately) — the bus's own dispatch task needs
    # a chance to actually run and deliver it before we can assert on it.
    await started_bus._queue.join()

    error_logs = [d for d in seen if d.get("level") == "error"]
    assert error_logs, "TTS failure must be posted to the visible bus log"
    assert "TTS failed" in error_logs[0]["msg"]
