"""tests/test_pipeline_integration.py — the whole turn, end to end.

Every bug this file guards against made NixOrb look like it had hung on
startup or on activation:

  * ``nixorb.llm.ollama_backend`` did not exist, so ``nixorb start`` died
    with ModuleNotFoundError before the orb was usable.
  * ActionExecutor waited on a future nobody held, so every command sat for
    the full timeout and was then reported as denied.
  * The turn ran *as* an event-bus handler, so the bus could not deliver the
    ACTION_REQUESTED the turn itself was waiting on — a genuine deadlock,
    and orb state events never arrived either.
  * Tool calls returned by the model were dropped on the floor.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web

from nixorb.core.event_bus import Event

pytestmark = pytest.mark.asyncio


def _settings(host: str):
    s = MagicMock()
    s.ollama_host = host
    s.llm_model = "llama3.2"
    s.llm_temperature = 0.7
    s.llm_max_tokens = 512
    s.require_action_confirmation = True
    s.sandbox_actions = False
    s.action_confirm_timeout = 5.0
    return s


def _ndjson(*objects) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objects)


@pytest.fixture
async def ollama(aiohttp_server):
    """A stand-in Ollama whose reply is scripted per test."""
    script: dict = {"chunks": [{"message": {"content": "hello"}, "done": True}]}

    async def tags(_request):
        return web.json_response({"models": [{"name": "llama3.2:latest"}]})

    async def chat(request):
        await request.json()
        resp = web.StreamResponse()
        await resp.prepare(request)
        turn = script["chunks"]
        if turn and isinstance(turn[0], list):  # a queue of turns
            turn = script["chunks"].pop(0)
        await resp.write(_ndjson(*turn))
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/api/tags", tags)
    app.router.add_post("/api/chat", chat)
    server = await aiohttp_server(app)
    server.script = script
    return server


def _url(server) -> str:
    return f"http://{server.host}:{server.port}"


async def test_backend_module_is_importable():
    """main.py and cli.py both import this by name."""
    from nixorb.llm.ollama_backend import OllamaBackend

    assert hasattr(OllamaBackend, "health_check")
    assert hasattr(OllamaBackend, "stream")
    assert hasattr(OllamaBackend, "generate")
    assert hasattr(OllamaBackend, "close")


async def test_health_check_and_stream(ollama, started_bus):
    from nixorb.llm.ollama_backend import OllamaBackend

    llm = OllamaBackend(_settings(_url(ollama)))
    try:
        assert (await llm.health_check())["ok"] is True

        ollama.script["chunks"] = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " world"}},
            {"message": {"content": ""}, "done": True},
        ]
        assert await llm.generate([{"role": "user", "content": "hi"}]) == "Hello world"
    finally:
        await llm.close()


async def test_action_round_trip_completes_promptly(ollama, started_bus):
    """The model asks for a command, the user approves, the command runs."""
    from nixorb.action.executor import ActionExecutor
    from nixorb.llm.ollama_backend import OllamaBackend
    from nixorb.ui.confirm_dialog import register_confirmation_handler

    ollama.script["chunks"] = [
        {"message": {"content": "<ACTION>echo integration-ok</ACTION>"}},
        {"message": {"content": ""}, "done": True},
    ]

    register_confirmation_handler(started_bus, ask_fn=lambda _cmd: True)
    llm = OllamaBackend(_settings(_url(ollama)))
    with patch("os.geteuid", return_value=1000):
        executor = ActionExecutor(_settings(_url(ollama)))

    try:
        started = time.monotonic()
        reply = await llm.generate([{"role": "user", "content": "run it"}])
        results = await executor.handle_llm_output(reply)
        elapsed = time.monotonic() - started
    finally:
        await llm.close()

    assert len(results) == 1
    assert results[0].success
    assert "integration-ok" in results[0].stdout
    # The old code sat here for the whole confirmation timeout.
    assert elapsed < 3.0, f"round trip took {elapsed:.1f}s — something is waiting"


async def test_turn_as_a_task_does_not_deadlock_the_bus(ollama, started_bus):
    """A turn must not run *as* a bus handler.

    Handlers are awaited inline by the dispatch loop, so a turn that waits on
    ACTION_REQUESTED can never be answered — and no orb state event gets
    through while it waits. main.py spawns the turn as its own task; this is
    that arrangement, verified.
    """
    from nixorb.action.executor import ActionExecutor
    from nixorb.ui.confirm_dialog import register_confirmation_handler

    register_confirmation_handler(started_bus, ask_fn=lambda _cmd: True)
    with patch("os.geteuid", return_value=1000):
        executor = ActionExecutor(_settings(_url(ollama)))

    orb_states: list[str] = []

    async def _track(payload):
        orb_states.append(payload.event.name)

    started_bus.subscribe(Event.ORB_THINKING, _track)
    started_bus.subscribe(Event.ORB_IDLE, _track)

    finished = asyncio.Event()
    captured: dict = {}

    async def _turn():
        await started_bus.emit(Event.ORB_THINKING, source="test")
        captured["results"] = await executor.handle_llm_output(
            "<ACTION>echo from-a-task</ACTION>"
        )
        await started_bus.emit(Event.ORB_IDLE, source="test")
        finished.set()

    async def _on_trigger(_payload):
        asyncio.get_running_loop().create_task(_turn())

    started_bus.subscribe(Event.HOTKEY_TRIGGERED, _on_trigger)
    await started_bus.emit(Event.HOTKEY_TRIGGERED, source="test")

    await asyncio.wait_for(finished.wait(), timeout=5.0)
    await started_bus._queue.join()

    assert captured["results"][0].success
    # Orb state reached the UI *during* the turn — it used to be stuck behind it.
    assert orb_states == ["ORB_THINKING", "ORB_IDLE"]


async def test_model_tool_calls_reach_the_plugin(ollama, started_bus, tmp_path):
    """A tool call must actually run the plugin, not vanish."""
    from nixorb.llm.ollama_backend import OllamaBackend
    from nixorb.plugins.loader import PluginLoader

    (tmp_path / "greeter.py").write_text(
        'TOOL_DEFINITION = {"type": "function", "function": '
        '{"name": "greet", "description": "d", "parameters": {}}}\n'
        'def greet(name: str) -> str: return f"hi {name}"\n'
    )
    loader = PluginLoader(str(tmp_path))
    loader.load_all()
    assert loader.get_tool_definitions()

    ollama.script["chunks"] = [
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "greet", "arguments": {"name": "Mason"}}}
        ]}},
        {"message": {"content": ""}, "done": True},
    ]

    llm = OllamaBackend(_settings(_url(ollama)))
    try:
        await llm.generate([{"role": "user", "content": "greet me"}])
        assert llm.last_tool_calls, "tool calls were dropped"
        call = llm.last_tool_calls[0]
        result = await loader.dispatch(call["name"], call["arguments"])
    finally:
        await llm.close()

    assert result == "hi Mason"


async def test_unreachable_ollama_fails_fast_with_advice(started_bus):
    """No Ollama must produce a fixable message quickly, not a hang."""
    from nixorb.llm.ollama_backend import OllamaBackend

    llm = OllamaBackend(_settings("http://127.0.0.1:1"))
    try:
        started = time.monotonic()
        health = await llm.health_check()
        elapsed = time.monotonic() - started
    finally:
        await llm.close()

    assert health["ok"] is False
    assert "ollama serve" in health["error"]
    assert elapsed < 6.0
