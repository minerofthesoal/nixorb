"""nixorb/plugins/loader.py — Hot-reloadable plugin system."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class PluginLoader:
    def __init__(self, plugin_dir: str | Path) -> None:
        self._dir     = Path(plugin_dir)
        self._plugins: dict[str, Any] = {}

    def load_all(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for py in sorted(self._dir.glob("*.py")):
            if py.name.startswith("_"):
                continue
            self._load_file(py)

    def reload_all(self) -> None:
        self._plugins.clear()
        self.load_all()

    def _load_file(self, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            self._plugins[path.stem] = module
            log.info("Plugin loaded: %s", path.stem)
        except Exception:
            log.exception("Failed to load plugin: %s", path)

    def plugin_names(self) -> list[str]:
        return sorted(self._plugins.keys())

    def get_tool_definitions(self) -> list[dict]:
        tools: list[dict] = []
        for module in self._plugins.values():
            defn = getattr(module, "TOOL_DEFINITION", None)
            if defn:
                tools.append(defn)
        return tools

    async def dispatch(self, tool_name: str, args: dict) -> str:
        import asyncio
        for module in self._plugins.values():
            fn = getattr(module, tool_name, None)
            if fn is not None:
                try:
                    if asyncio.iscoroutinefunction(fn):
                        return str(await fn(**args))
                    return str(fn(**args))
                except Exception as exc:
                    log.exception("Plugin %s raised:", tool_name)
                    return f"Error in plugin {tool_name}: {exc}"
        return f"Tool '{tool_name}' not found in any loaded plugin."
