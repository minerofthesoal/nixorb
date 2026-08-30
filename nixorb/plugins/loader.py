"""NixOrb plugin loader — hot-reloadable drop-in Python tools.

Plugins are Python files dropped into the plugins directory that define
TOOL_DEFINITION and an implementation function. The LLM can call these
tools via function calling.
"""
from __future__ import annotations

import asyncio
import functools
import importlib
import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

# A plugin needs TOOL_DEFINITION to be *advertised* to the LLM, but any
# public callable it defines can still be dispatched by name.
TOOL_DEFINITION_ATTR = "TOOL_DEFINITION"

# How long a single plugin call may run before we give up on it.
TOOL_TIMEOUT_SECONDS = 30.0


class PluginLoader:
    """Loads and manages NixOrb plugins."""

    def __init__(self, plugin_dir: str | None = None) -> None:
        self._plugin_dir = Path(plugin_dir) if plugin_dir else Path.home() / ".local" / "share" / "nixorb" / "plugins"
        self._plugin_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, ModuleType] = {}
        self._tools: dict[str, Callable] = {}

    def load_all(self) -> int:
        """Load all plugins: bundled builtins first, then the user directory."""
        count = 0
        # Pick up plugin files created since this process started.
        importlib.invalidate_caches()

        builtin_dir = Path(__file__).parent / "builtin"
        for file_path in sorted(builtin_dir.glob("*.py")):
            if file_path.name.startswith("_"):
                continue
            try:
                self._load_plugin(file_path)
                count += 1
            except Exception as exc:
                log.error("Plugin: failed to load builtin %s: %s", file_path.name, exc)

        if not self._plugin_dir.exists():
            log.warning("Plugin dir not found: %s", self._plugin_dir)
        else:
            for file_path in sorted(self._plugin_dir.glob("*.py")):
                if file_path.name.startswith("_"):
                    continue
                try:
                    self._load_plugin(file_path)
                    count += 1
                except Exception as exc:
                    log.error("Plugin: failed to load %s: %s", file_path.name, exc)

        log.info("Plugin: loaded %d plugin(s)", count)
        bus.emit_sync(
            Event.PLUGIN_LOADED,
            data={"count": count, "plugins": list(self._plugins.keys())},
            source="PluginLoader",
        )
        return count

    def _load_plugin(self, file_path: Path) -> None:
        """Load a single plugin file."""
        name = file_path.stem
        spec = importlib.util.spec_from_file_location(name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[f"nixorb.plugins.loaded.{name}"] = module

        # Execute the source directly instead of spec.loader.exec_module().
        # Python validates its __pycache__ on (mtime, size), so editing a
        # plugin without changing its length and reloading in the same second
        # silently re-runs the *old* bytecode — exactly the case "Reload
        # Plugins" exists for.
        source = file_path.read_text(encoding="utf-8")
        exec(compile(source, str(file_path), "exec"), module.__dict__)

        registered: list[str] = []

        # Every public callable *defined in this file* is dispatchable. The
        # `__module__` check keeps imported helpers (json.dumps, shutil.which)
        # out of the tool table.
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            obj = getattr(module, attr_name)
            if callable(obj) and getattr(obj, "__module__", None) == module.__name__:
                self._tools[attr_name] = obj
                registered.append(attr_name)

        # TOOL_DEFINITION's declared name wins, so a plugin can expose its
        # entry point under a different name than the Python function.
        tool_def = getattr(module, TOOL_DEFINITION_ATTR, None)
        if isinstance(tool_def, dict):
            func_name = tool_def.get("function", {}).get("name", name)
            impl = getattr(module, func_name, None)
            if impl is None:
                # Fall back to the first public callable the module defines.
                impl = next(
                    (getattr(module, n) for n in registered), None
                )
            if impl is None:
                raise AttributeError(
                    f"Plugin {name} declares tool '{func_name}' but defines "
                    f"no callable to implement it"
                )
            self._tools[func_name] = impl
            registered.append(func_name)

        self._plugins[name] = module
        log.debug("Plugin: loaded '%s' exposing %s", name, registered or "nothing")

    def reload(self, name: str) -> bool:
        """Reload a specific plugin."""
        if name not in self._plugins:
            return False

        file_path = self._plugin_dir / f"{name}.py"
        if not file_path.exists():
            return False

        # Remove old module
        del self._plugins[name]
        module_name = f"nixorb.plugins.loaded.{name}"
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Reload
        try:
            self._load_plugin(file_path)
            return True
        except Exception as exc:
            log.error("Plugin: reload failed for %s: %s", name, exc)
            return False

    def reload_all(self) -> int:
        """Reload all plugins from disk."""
        for name in list(self._plugins):
            sys.modules.pop(f"nixorb.plugins.loaded.{name}", None)
        self._plugins.clear()
        self._tools.clear()
        return self.load_all()

    def plugin_names(self) -> list[str]:
        """Get list of loaded plugin names."""
        return list(self._plugins.keys())

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for LLM function calling."""
        tools = []
        for _name, module in self._plugins.items():
            tool_def = getattr(module, TOOL_DEFINITION_ATTR, None)
            if isinstance(tool_def, dict):
                tools.append(tool_def)
        return tools

    def get_tool_function(self, name: str) -> Callable | None:
        """Get a tool implementation function by name."""
        return self._tools.get(name)

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Call a tool by name with arguments."""
        func = self._tools.get(name)
        if func is None:
            raise KeyError(f"Tool '{name}' not found")
        return func(**kwargs)

    async def dispatch(self, name: str, args: dict[str, Any] | None = None) -> str:
        """Call a plugin tool by name and return its result as a string.

        Handles sync and async plugins alike, and never raises: the return
        value goes straight back to the LLM, so an error message is more
        useful than an exception that kills the turn.
        """
        func = self._tools.get(name)
        if func is None:
            available = ", ".join(sorted(self._tools)) or "none"
            log.warning("Plugin: tool '%s' not found (have: %s)", name, available)
            return f"Tool '{name}' not found. Available tools: {available}"

        kwargs = dict(args or {})
        try:
            if inspect.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(**kwargs), timeout=TOOL_TIMEOUT_SECONDS
                )
            else:
                # Plugins are third-party code and may block on I/O; keep them
                # off the event loop.
                result = await asyncio.wait_for(
                    asyncio.to_thread(functools.partial(func, **kwargs)),
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
                if inspect.isawaitable(result):
                    result = await result
        except TimeoutError:
            log.error("Plugin: tool '%s' timed out", name)
            return f"Tool '{name}' timed out after {TOOL_TIMEOUT_SECONDS:g}s"
        except TypeError as exc:
            log.error("Plugin: bad arguments for '%s': %s", name, exc)
            return f"Tool '{name}' rejected those arguments: {exc}"
        except Exception as exc:
            log.exception("Plugin: tool '%s' raised", name)
            return f"Tool '{name}' failed: {exc}"

        return "" if result is None else str(result)
