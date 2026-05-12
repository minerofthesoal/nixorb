"""tests/test_clipboard.py — Clipboard integration tests."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

pytestmark = pytest.mark.asyncio


async def test_read_clipboard_success():
    from nixorb.action.clipboard import read_clipboard

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("nixorb.action.clipboard._HAS_WL_PASTE", True),
        patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ),
        patch(
            "asyncio.wait_for",
            AsyncMock(return_value=(b"hello clipboard", b"")),
        ),
    ):
        result = await read_clipboard()

    assert result == "hello clipboard"


async def test_read_clipboard_no_wl_paste():
    from nixorb.action.clipboard import read_clipboard

    with patch("nixorb.action.clipboard._HAS_WL_PASTE", False):
        result = await read_clipboard()

    assert result is None


async def test_write_clipboard_success():
    from nixorb.action.clipboard import write_clipboard

    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
        patch("nixorb.action.clipboard._HAS_WL_COPY", True),
        patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        ),
        patch(
            "asyncio.wait_for",
            AsyncMock(return_value=(b"", b"")),
        ),
    ):
        result = await write_clipboard("test text")

    assert result is True


async def test_write_clipboard_no_wl_copy():
    from nixorb.action.clipboard import write_clipboard

    with patch("nixorb.action.clipboard._HAS_WL_COPY", False):
        result = await write_clipboard("anything")

    assert result is False
