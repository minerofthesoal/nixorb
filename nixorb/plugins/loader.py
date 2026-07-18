"""NixOrb plugin loader — hot-reloadable drop-in Python tools.

Plugins are Python files dropped into the plugins directory that define
TOOL_DEFINITION and an implementation function. The LLM can call these
tools via function calling.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

# Required attributes for a valid plugin
REQUIRED_ATTRS = ["TOOL_DEFINITION"]


class PluginLoader:
    """Loads and manages NixOrb plugins."""

    def __init__(self, plugin_dir: str | None = None) -> None:
        self._plugin_dir = Path(plugin_dir) if plugin_dir else Path.home() / ".local" / "share" / "nixorb" / "plugins"
        self._plugin_dir.mkdir(parents=True, exist_ok=True)
        self._plugins: dict[str, ModuleType] = {}
        self._tools: dict[str, Callable] = {}

    def load_all(self) -> int:
        """Load all plugins from the plugin directory."""
        count = 0
        if not self._plugin_dir.exists():
            log.warning("Plugin dir not found: %s", self._plugin_dir)
            return 0

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
        spec.loader.exec_module(module)

        # Validate required attributes
        for attr in REQUIRED_ATTRS:
            if not hasattr(module, attr):
                raise AttributeError(f"Plugin {name} missing required attribute: {attr}")

        # Register tool functions
        tool_def = module.TOOL_DEFINITION
        func_name = tool_def.get("function", {}).get("name", name)

        if hasattr(module, func_name):
            self._tools[func_name] = getattr(module, func_name)
        else:
            # Try to find any callable that matches
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if callable(obj) and not attr_name.startswith("_"):
                    self._tools[func_name] = obj
                    break

        self._plugins[name] = module
        log.debug("Plugin: loaded '%s' with tool '%s'", name, func_name)

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
        """Reload all plugins."""
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
            if hasattr(module, "TOOL_DEFINITION"):
                tools.append(module.TOOL_DEFINITION)
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
