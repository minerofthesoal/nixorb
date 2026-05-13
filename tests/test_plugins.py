"""tests/test_plugins.py — Plugin loader tests."""
from __future__ import annotations

import pytest

from nixorb.plugins.loader import PluginLoader


def test_empty_dir(tmp_path):
    loader = PluginLoader(tmp_path)
    loader.load_all()
    assert loader.plugin_names() == []
    assert loader.get_tool_definitions() == []


def test_loads_valid_plugin(tmp_path):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text("""\
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "greet",
        "description": "Says hello",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
}

def greet() -> str:
    return "hello"
""")
    loader = PluginLoader(tmp_path)
    loader.load_all()
    assert "my_plugin" in loader.plugin_names()
    tools = loader.get_tool_definitions()
    assert any(t["function"]["name"] == "greet" for t in tools)


@pytest.mark.asyncio
async def test_dispatch_sync_function(tmp_path):
    plugin = tmp_path / "calc.py"
    plugin.write_text("def add(a: int, b: int) -> int: return a + b\n")
    loader = PluginLoader(tmp_path)
    loader.load_all()
    result = await loader.dispatch("add", {"a": 3, "b": 4})
    assert result == "7"


@pytest.mark.asyncio
async def test_dispatch_async_function(tmp_path):
    plugin = tmp_path / "async_plug.py"
    plugin.write_text("async def echo(msg: str) -> str: return msg\n")
    loader = PluginLoader(tmp_path)
    loader.load_all()
    result = await loader.dispatch("echo", {"msg": "ping"})
    assert result == "ping"


@pytest.mark.asyncio
async def test_dispatch_unknown_returns_message(tmp_path):
    loader = PluginLoader(tmp_path)
    loader.load_all()
    result = await loader.dispatch("nonexistent", {})
    assert "not found" in result


def test_skips_dunder_files(tmp_path):
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "_private.py").write_text("x = 1")
    loader = PluginLoader(tmp_path)
    loader.load_all()
    assert loader.plugin_names() == []


def test_reload(tmp_path):
    plugin = tmp_path / "hot.py"
    plugin.write_text("VALUE = 1\n")
    loader = PluginLoader(tmp_path)
    loader.load_all()
    plugin.write_text("VALUE = 2\n")
    loader.reload_all()
    assert loader._plugins["hot"].VALUE == 2
