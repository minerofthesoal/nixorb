# NixOrb Plugin System

## Plugin Directory

User plugins: `~/.local/share/nixorb/plugins/`
Built-in plugins: `nixorb/plugins/builtin/`
Repo example plugins: `plugins/`

## Built-in Plugins

| Plugin | Tools | Description |
|---|---|---|
| `systemd_plugin` | `systemd_service` | Query/control systemd services |
| `kdeconnect_plugin` | `kdeconnect` | KDE Connect phone integration |

## Bundled Example Plugins

| Plugin | Tools | Description |
|---|---|---|
| `weather_plugin` | `get_weather` | Live weather via open-meteo |
| `volume_plugin` | `control_volume` | PipeWire/PulseAudio volume |
| `notes_plugin` | `manage_note` | Quick note-taking |
| `timer_plugin` | `set_timer` | Desktop notification timers |

## Writing a Plugin

```python
# ~/.local/share/nixorb/plugins/my_plugin.py

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "Short description — the LLM decides when to call this.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The input query"
                }
            },
            "required": ["query"]
        }
    }
}

def my_tool(query: str) -> str:
    return f"Result for: {query}"

# Async plugins also work:
async def my_async_tool(query: str) -> str:
    import asyncio
    await asyncio.sleep(0.1)
    return f"Async result for: {query}"
```

## Reloading Plugins

- GUI: Settings → Plugins → Reload Plugins
- CLI: `nixorb list-plugins` (auto-reloads)
- Plugins use `compile()+exec()` so file changes are always picked up
