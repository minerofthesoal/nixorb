"""tests/test_action_confirmation.py — regression test for the
ACTION_REQUESTED/ACTION_RESULT confirmation handshake.

Before this fix, nothing subscribed to Event.ACTION_REQUESTED, so
ActionExecutor's confirmation wait always timed out after 30s and denied
*every* command, with no dialog ever shown and no error surfaced anywhere.
`register_confirmation_handler` (nixorb/ui/confirm_dialog.py) is the fix;
it's wired in for real in nixorb/main.py. These tests exercise it directly,
injecting a fake `ask_fn` so no real Qt dialog needs to appear.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _make_settings(confirm=True):
    s = MagicMock()
    s.require_action_confirmation = confirm
    return s


async def test_registered_handler_approves_and_unblocks_executor(started_bus):
    from nixorb.action.executor import ActionExecutor
    from nixorb.core.event_bus import Event
    from nixorb.ui.confirm_dialog import register_confirmation_handler

    register_confirmation_handler(started_bus, Event.ACTION_REQUESTED, ask_fn=lambda cmd: True)

    with patch("os.geteuid", return_value=1000):  # not root
        executor = ActionExecutor(_make_settings())

    results = await executor.handle_llm_output("<ACTION>echo hi</ACTION>")

    assert len(results) == 1
    assert results[0].success
    assert "hi" in results[0].stdout


async def test_registered_handler_denies_and_executor_reports_denial(started_bus):
    from nixorb.action.executor import ActionExecutor
    from nixorb.core.event_bus import Event
    from nixorb.ui.confirm_dialog import register_confirmation_handler

    register_confirmation_handler(started_bus, Event.ACTION_REQUESTED, ask_fn=lambda cmd: False)

    with patch("os.geteuid", return_value=1000):  # not root
        executor = ActionExecutor(_make_settings())

    results = await executor.handle_llm_output("<ACTION>echo hi</ACTION>")

    assert len(results) == 1
    assert not results[0].success
    assert "denied" in results[0].stderr.lower()


async def test_ask_fn_receives_the_actual_command(started_bus):
    from nixorb.action.executor import ActionExecutor
    from nixorb.core.event_bus import Event
    from nixorb.ui.confirm_dialog import register_confirmation_handler

    seen_commands = []

    def _record_and_approve(cmd):
        seen_commands.append(cmd)
        return True

    register_confirmation_handler(started_bus, Event.ACTION_REQUESTED, ask_fn=_record_and_approve)

    with patch("os.geteuid", return_value=1000):  # not root
        executor = ActionExecutor(_make_settings())

    await executor.handle_llm_output("<ACTION>echo specific-marker</ACTION>")

    assert seen_commands == ["echo specific-marker"]


async def test_default_ask_fn_is_the_real_confirm_dialog(started_bus):
    """Without an explicit ask_fn, the default must be the real Qt
    ConfirmDialog.ask — that's what actually shows the dialog in
    production. We patch it so no real Qt window needs to appear, but
    assert it's genuinely the thing that gets called."""
    from nixorb.core.event_bus import Event
    from nixorb.ui.confirm_dialog import register_confirmation_handler

    with patch("nixorb.ui.confirm_dialog.ConfirmDialog.ask", return_value=True) as mock_ask:
        register_confirmation_handler(started_bus, Event.ACTION_REQUESTED)
        await started_bus.emit(
            Event.ACTION_REQUESTED, data={"command": "echo hi"}, source="test"
        )
        await started_bus._queue.join()  # let the dispatch loop process it

    mock_ask.assert_called_once_with("echo hi")


