"""tests/test_ipc.py — the control socket behind `nixorb trigger`.

`nixorb trigger` exists so a KDE global shortcut can activate a running orb.
It was a TODO stub that printed "not yet implemented"; these cover the
channel that now backs it.
"""
from __future__ import annotations

import asyncio

import pytest

from nixorb.core.ipc import IPCServer, is_running, send, socket_path


@pytest.fixture(autouse=True)
def isolated_socket(tmp_path, monkeypatch):
    """Keep tests off the real user socket."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


async def _echo(command: str) -> str:
    return f"ok: {command}"


@pytest.fixture
async def server():
    srv = IPCServer({"trigger": _echo, "quit": _echo})
    assert await srv.start()
    yield srv
    await srv.stop()


def _client(command: str, timeout: float = 3.0) -> str:
    """Run the blocking client off the loop it is talking to."""
    return send(command, timeout=timeout)


def test_socket_path_follows_xdg_runtime_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert socket_path() == tmp_path / "nixorb.sock"


def test_socket_path_falls_back_when_xdg_is_unset(monkeypatch):
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert socket_path().name.startswith("nixorb-")


async def test_ping_answers(server):
    assert (await asyncio.to_thread(_client, "ping")).startswith("ok")


async def test_trigger_reaches_its_handler(server):
    assert await asyncio.to_thread(_client, "trigger") == "ok: trigger"


async def test_unknown_command_is_reported(server):
    reply = await asyncio.to_thread(_client, "nonsense")
    assert reply.startswith("error:")
    assert "nonsense" in reply
    # The reply should say what it *would* accept.
    assert "trigger" in reply


async def test_empty_command_is_reported(server):
    assert (await asyncio.to_thread(_client, "")).startswith("error:")


async def test_is_running_true_while_served(server):
    assert await asyncio.to_thread(is_running) is True


async def test_is_running_false_after_stop(server):
    await server.stop()
    assert await asyncio.to_thread(is_running) is False


async def test_send_raises_when_nothing_listens():
    with pytest.raises(ConnectionError):
        await asyncio.to_thread(_client, "trigger")


async def test_stale_socket_file_is_reclaimed(isolated_socket):
    """A crash leaves the socket file behind; the next start must take it."""
    stale = socket_path()
    stale.touch()
    assert stale.exists()

    srv = IPCServer({"trigger": _echo})
    try:
        assert await srv.start() is True
        assert await asyncio.to_thread(_client, "trigger") == "ok: trigger"
    finally:
        await srv.stop()


async def test_second_instance_is_refused(server):
    """Two orbs would fight over the microphone."""
    other = IPCServer({"trigger": _echo})
    assert await other.start() is False


async def test_stop_removes_the_socket_file(isolated_socket):
    srv = IPCServer({"trigger": _echo})
    await srv.start()
    assert srv.path.exists()
    await srv.stop()
    assert not srv.path.exists()


async def test_socket_is_private_to_the_user(server):
    """It accepts commands that run code — nobody else gets to connect."""
    mode = server.path.stat().st_mode
    assert mode & 0o077 == 0, oct(mode)


async def test_handler_errors_do_not_kill_the_server(isolated_socket):
    async def _boom(_command: str) -> str:
        raise RuntimeError("handler exploded")

    srv = IPCServer({"boom": _boom, "trigger": _echo})
    await srv.start()
    try:
        # The client gets a real error string, not an empty reply...
        reply = await asyncio.to_thread(_client, "boom")
        assert reply.startswith("error:")
        assert "exploded" in reply
        # ...and the server is still up for the next command.
        assert await asyncio.to_thread(_client, "trigger") == "ok: trigger"
    finally:
        await srv.stop()
