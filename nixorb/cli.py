"""NixOrb CLI — command-line interface.

Usage:
    nixorb start              Launch the GUI orb
    nixorb start --headless   Daemon mode (no Qt)
    nixorb trigger            Trigger activation (for KDE shortcuts)
    nixorb ask "query"        One-shot text query
    nixorb tts "text"         Speak text
    nixorb status             Show system status
    nixorb config             Open config in editor
    nixorb check              Check dependencies
    nixorb version            Show version
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import typer

import nixorb
from nixorb.settings import Settings

app = typer.Typer(
    name="nixorb",
    help="NixOrb — AI assistant for Arch Linux + KDE Plasma 6",
    no_args_is_help=True,
)

log = logging.getLogger(__name__)


@app.command()
def start(
    headless: bool = typer.Option(False, "--headless", help="Run without GUI"),
) -> None:
    """Start NixOrb."""
    if headless:
        typer.echo("Headless mode not yet implemented — use GUI mode")
        raise typer.Exit(1)
    else:
        from nixorb.main import main

        main()


@app.command()
def trigger() -> None:
    """Trigger NixOrb activation (for KDE shortcuts)."""
    # This would communicate with a running NixOrb instance
    typer.echo("Triggering NixOrb activation…")
    # TODO: Implement IPC to running instance
    typer.echo("(Not yet implemented — double-click the orb instead)")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Text query to send to the AI"),
) -> None:
    """Send a one-shot query to the AI."""

    async def _ask() -> None:
        settings = Settings.load()
        from nixorb.llm.ollama_backend import OllamaBackend

        llm = OllamaBackend(settings)
        messages = [{"role": "user", "content": query}]

        typer.echo(f"🤔 Querying {settings.llm_model}…")
        try:
            response = await llm.generate(messages)
            typer.echo(f"\n🤖 {response}")
        except Exception as exc:
            typer.echo(f"❌ Error: {exc}", err=True)
            raise typer.Exit(1)
        finally:
            await llm.close()

    asyncio.run(_ask())


@app.command()
def tts(
    text: str = typer.Argument(..., help="Text to speak"),
) -> None:
    """Speak text using TTS."""

    async def _speak() -> None:
        settings = Settings.load()
        from nixorb.tts.piper_tts import PiperTTS

        tts_engine = PiperTTS(settings)
        typer.echo(f"🔊 Speaking: {text}")
        await tts_engine.speak(text)

    asyncio.run(_speak())


@app.command()
def status() -> None:
    """Show NixOrb system status."""
    settings = Settings.load()

    typer.echo("═" * 50)
    typer.echo(f"  NixOrb {nixorb.__version__}")
    typer.echo("═" * 50)

    # Check Ollama
    typer.echo("\n🤖 AI Backend:")
    typer.echo(f"  Backend: {settings.llm_backend}")
    typer.echo(f"  Model: {settings.llm_model}")
    typer.echo(f"  Host: {settings.ollama_host}")

    # Check dependencies
    typer.echo("\n📦 Dependencies:")
    deps = {
        "piper": "TTS engine",
        "wl-paste": "Clipboard (Wayland)",
        "grim": "Screenshot",
        "espeak-ng": "TTS fallback",
        "bwrap": "Sandbox",
        "ollama": "AI backend",
    }
    for dep, desc in deps.items():
        found = "✅" if shutil.which(dep) else "❌"
        typer.echo(f"  {found} {dep} — {desc}")

    # VRAM
    typer.echo("\n🎮 GPU:")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        name, free, total = [x.strip() for x in out.split(",")]
        typer.echo(f"  {name}")
        typer.echo(f"  VRAM: {free} / {total} MB free")
    except Exception:
        typer.echo("  (No NVIDIA GPU detected)")

    typer.echo("")


@app.command()
def config() -> None:
    """Open NixOrb configuration in default editor."""
    config_path = Path.home() / ".config" / "nixorb" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        Settings().save()

    editor = shutil.which("nano") or shutil.which("vim") or shutil.which("vi")
    if editor:
        subprocess.call([editor, str(config_path)])
    else:
        typer.echo(f"Config file: {config_path}")


@app.command()
def check() -> None:
    """Check system dependencies."""
    required = ["python", "pip"]
    recommended = [
        "piper", "wl-paste", "wl-copy", "grim", "espeak-ng",
        "bwrap", "ollama", "nvidia-smi",
    ]

    typer.echo("Required:")
    for dep in required:
        found = shutil.which(dep)
        typer.echo(f"  {'✅' if found else '❌'} {dep}")

    typer.echo("\nRecommended:")
    for dep in recommended:
        found = shutil.which(dep)
        typer.echo(f"  {'✅' if found else '❌'} {dep}")


@app.command()
def version() -> None:
    """Show NixOrb version."""
    typer.echo(f"NixOrb {nixorb.__version__}")


if __name__ == "__main__":
    app()
