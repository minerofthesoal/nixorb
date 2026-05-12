"""nixorb/llm/tool_dispatcher.py — Dispatch LLM tool/function calls to plugins."""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


class ToolDispatcher:
    """
    Handles OpenAI-style function-call tool_use responses.
    Parses tool_use content blocks and dispatches to plugin_loader.
    """

    def __init__(self, plugin_loader) -> None:
        self._loader = plugin_loader

    async def handle_response_chunks(self, chunks: list[str]) -> tuple[str, list[str]]:
        """
        Given collected LLM chunks, separate text content from tool calls.
        Returns (text_response, list_of_tool_results).
        """
        full = "".join(chunks)
        tool_results: list[str] = []

        # Handle JSON tool call embedded in response (some HF models)
        if full.strip().startswith('{"tool":') or '"function_call"' in full:
            try:
                obj = json.loads(full.strip())
                name = (obj.get("tool") or obj.get("function_call", {}).get("name", ""))
                args = (obj.get("parameters") or obj.get("function_call", {}).get("arguments", {}))
                if isinstance(args, str):
                    args = json.loads(args)
                if name:
                    result = await self._loader.dispatch(name, args)
                    tool_results.append(f"[{name}] → {result}")
                    return "", tool_results
            except (json.JSONDecodeError, Exception):
                pass

        return full, tool_results

    async def dispatch_tool(self, name: str, args: dict) -> str:
        """Dispatch a single tool call and return the result string."""
        log.info("Tool call: %s(%s)", name, args)
        try:
            result = await self._loader.dispatch(name, args)
            log.info("Tool result: %s → %s", name, str(result)[:100])
            return result
        except Exception as exc:
            log.error("Tool %s failed: %s", name, exc)
            return f"Error calling {name}: {exc}"
