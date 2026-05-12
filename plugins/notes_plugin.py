"""plugins/notes_plugin.py — Quick note-taking to ~/.local/share/nixorb/notes/."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

_NOTES_DIR = Path.home() / ".local" / "share" / "nixorb" / "notes"

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "manage_note",
        "description": (
            "Create, read, list, or delete quick notes. "
            "Use when the user says 'make a note', 'remind me', "
            "'what are my notes', 'delete my note about X'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "read", "delete"],
                    "description": "Note action to perform.",
                },
                "title": {
                    "type": "string",
                    "description": "Note title / filename (without extension).",
                },
                "content": {
                    "type": "string",
                    "description": "Note content for 'create' action.",
                },
            },
            "required": ["action"],
        },
    },
}


def manage_note(
    action: str,
    title: str = "",
    content: str = "",
) -> str:
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)

    if action == "create":
        if not title:
            title = datetime.now().strftime("note_%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        path = _NOTES_DIR / f"{safe}.md"
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
        path.write_text(f"# {title}\n_{ts}_\n\n{content}\n")
        return f"Note saved: {path.name}"

    if action == "list":
        notes = sorted(_NOTES_DIR.glob("*.md"))
        if not notes:
            return "No notes found."
        return "\n".join(f"• {n.stem}" for n in notes)

    if action == "read":
        if not title:
            return "Specify a note title to read."
        matches = list(_NOTES_DIR.glob(f"*{title}*.md"))
        if not matches:
            return f"No note matching '{title}'"
        return matches[0].read_text()

    if action == "delete":
        if not title:
            return "Specify a note title to delete."
        matches = list(_NOTES_DIR.glob(f"*{title}*.md"))
        if not matches:
            return f"No note matching '{title}'"
        matches[0].unlink()
        return f"Deleted: {matches[0].stem}"

    return f"Unknown action: {action}"
