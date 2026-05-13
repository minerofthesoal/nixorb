"""tests/test_executor.py — ActionExecutor unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _make_settings(confirm=False):
    s = MagicMock()
    s.require_action_confirmation = confirm
    return s


@pytest.fixture
def executor(started_bus):
    from nixorb.action.executor import ActionExecutor
    with patch("os.geteuid", return_value=1000):  # not root
        ex = ActionExecutor(_make_settings(confirm=False))
    return ex


async def test_simple_echo(executor):
    results = await executor.handle_llm_output("<ACTION>echo hello</ACTION>")
    assert len(results) == 1
    assert "hello" in results[0].stdout
    assert results[0].success


async def test_hard_deny_rm_rf(executor):
    results = await executor.handle_llm_output("<ACTION>rm -rf /</ACTION>")
    assert len(results) == 1
    assert results[0].returncode == -1
    assert "denied" in results[0].stderr.lower()


async def test_no_action_blocks(executor):
    results = await executor.handle_llm_output("Just a plain text response, no actions.")
    assert results == []


async def test_multiple_action_blocks(executor):
    results = await executor.handle_llm_output(
        "<ACTION>echo one</ACTION> then <ACTION>echo two</ACTION>"
    )
    assert len(results) == 2
    assert "one" in results[0].stdout
    assert "two" in results[1].stdout


async def test_timeout_produces_result(executor):
    # Sleep longer than TIMEOUT_SECONDS override
    from nixorb import action
    with patch.object(action.executor, "TIMEOUT_SECONDS", 0.1):
        results = await executor.handle_llm_output("<ACTION>sleep 10</ACTION>")
    assert results[0].timed_out


async def test_root_raises():
    from nixorb.action.executor import ActionExecutor
    with patch("os.geteuid", return_value=0), pytest.raises(RuntimeError, match="root"):
        ActionExecutor(_make_settings())
