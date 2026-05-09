"""
nixorb/cli.py

Typer-powered CLI for NixOrb.

Commands
--------
  nixorb start          Start the GUI daemon
  nixorb start --headless   Run without Qt (daemon-only, for servers/SSH)
  nixorb ask "prompt"   One-shot LLM query, prints to stdout
  nixorb transcribe     Transcribe a local audio file
  nixorb check-deps     Verify Arch/AUR packages are installed
  nixorb export-config  Encrypt and export settings + memory
  nixorb import-config  Restore from encrypted archive
  nixorb list-devices   List available microphone devices
  nixorb list-plugins   List loaded plugins
  nixorb tts "text"     Speak text using the configured TTS engine
  nixorb memory-search  Query the long-term vector memory
  nixorb memory-clear   Wipe all stored memories
  nixorb version        Print version and exit
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app     = typer.Typer(name="nixorb", help="NixOrb — floating AI assistant for Arch Linux",
                      add_completion=True, rich_markup_mode="rich")
console = Console()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _load_settings():
    from nixorb.settings import Settings
    return Settings.load()


# ──────────────────────────────────────────────────────────────────── #
#  start                                                               #
# ──────────────────────────────────────────────────────────────────── #
@app.command()
def start(
    headless: bool = typer.Option(False,  "--headless", "-H",
                                  help="Run without Qt GUI"),
    debug:    bool = typer.Option(False,  "--debug",    "-d",
                                  help="Enable debug logging"),
    config:   Optional[Path] = typer.Option(None, "--config", "-c",
                                            help="Path to config TOML"),
) -> None:
    """[bold green]Start the NixOrb daemon.[/bold green]"""
    _setup_logging(debug)

    if config:
        import os
        os.environ["NIXORB_CONFIG"] = str(config)

    if headless:
        console.print(Panel("[yellow]Starting in headless mode (no GUI)[/yellow]"))
        settings = _load_settings()

        async def _headless_main():
            from nixorb.core.event_bus import Event, bus
            from nixorb.core.vram_manager import vram
            await bus.start()
            await vram.start_monitor()
            from nixorb.asr.whisper_engine import WhisperEngine
            from nixorb.tts.tts_factory import build_tts
            from nixorb.action.executor import ActionExecutor
            asr      = WhisperEngine(settings)
            tts      = build_tts(settings)
            executor = ActionExecutor(settings)
            console.print("[green]Headless daemon running. Ctrl-C to quit.[/green]")
            import signal
            stop = asyncio.Event()
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT,  stop.set)
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
            await stop.wait()
            await vram.stop()
            await bus.stop()

        asyncio.run(_headless_main())
    else:
        from nixorb.main import main as _gui_main
        _gui_main()


# ──────────────────────────────────────────────────────────────────── #
#  ask                                                                 #
# ──────────────────────────────────────────────────────────────────── #
@app.command()
def ask(
    prompt:  str  = typer.Argument(..., help="Prompt to send to the LLM"),
    model:   Optional[str] = typer.Option(None,  "--model",   "-m"),
    raw:     bool = typer.Option(False, "--raw",  "-r",  help="No formatting"),
    debug:   bool = typer.Option(False, "--debug", "-d"),
) -> None:
    """[bold]Send a one-shot text prompt and stream the response.[/bold]"""
    _setup_logging(debug)
    settings = _load_settings()
    if model:
        settings.llm_model = model

    async def _run() -> None:
        from nixorb.core.event_bus import bus
        await bus.start()
        from nixorb.llm.backends import OpenAIBackend, LocalLLMBackend, OllamaBackend
        b = settings.llm_backend.lower()
        if b == "openai":
            llm = OpenAIBackend(settings.openai_api_key, settings.llm_model,
                                settings.llm_base_url)
        elif b == "ollama":
            llm = OllamaBackend(settings.llm_model)
        else:
            llm = LocalLLMBackend(settings.local_model_path, settings.llm_vram_mb)

        messages = [{"role": "user", "content": prompt}]
        if raw:
            async for chunk in llm.stream(messages):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print()
        else:
            from rich.live import Live
            from rich.markdown import Markdown
            buf = ""
            with Live(Markdown(buf), console=console, refresh_per_second=15) as live:
                async for chunk in llm.stream(messages):
                    buf += chunk
                    live.update(Markdown(buf))
        await bus.stop()

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────── #
#  tts                                                                 #
# ──────────────────────────────────────────────────────────────────── #
@app.command()
def tts(
    text:  str  = typer.Argument(..., help="Text to speak"),
    debug: bool = typer.Option(False, "--debug", "-d"),
) -> None:
    """Speak *text* using the configured TTS engine."""
    _setup_logging(debug)
    settings = _load_settings()

    async def _run() -> None:
        from nixorb.core.event_bus import bus
        from nixorb.core.vram_manager import vram
        await bus.start()
        await vram.start_monitor()
        from nixorb.tts.tts_factory import build_tts
        engine = build_tts(settings)
        console.print(f"[cyan]Speaking:[/cyan] {text[:80]}…")
        await engine.speak(text)
        await vram.stop()
        await bus.stop()

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────── #
#  transcribe                                                          #
# ──────────────────────────────────────────────────────────────────── #
@app.command()
def transcribe(
    audio_file: Path = typer.Argument(..., help="Audio file to transcribe",
                                      exists=True, readable=True),
    debug: bool = typer.Option(False, "--debug", "-d"),
) -> None:
    """Transcribe a local audio file with Whisper Large v3."""
    _setup_logging(debug)
    settings = _load_settings()

    async def _run() -> None:
        import numpy as np
        import soundfile as sf
        from nixorb.core.event_bus import bus
        from nixorb.core.vram_manager import vram
        await bus.start()
        await vram.start_monitor()

        audio, sr = sf.read(str(audio_file), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            console.print(f"[yellow]Resampling {sr} Hz → 16000 Hz…[/yellow]")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)

        from nixorb.asr.whisper_engine import WhisperEngine
        engine = WhisperEngine(settings)
        text   = await engine._transcribe_async(audio)
        console.print(Panel(text or "(empty)", title="Transcript",
                            border_style="green"))
        await vram.stop()
        await bus.stop()

    asyncio.run(_run())


# ──────────────────────────────────────────────────────────────────── #
#  check-deps                                                          #
# ──────────────────────────────────────────────────────────────────── #
@app.command(name="check-deps")
def check_deps() -> None:
    """Verify all required Arch Linux packages are installed."""
    from nixorb.core.aur_checker import check_dependencies, REQUIRED
    missing = check_dependencies()

    table = Table(title="NixOrb Dependency Check", show_header=True)
    table.add_column("Package",  style="cyan")
    table.add_column("Source",   style="magenta")
    table.add_column("Reason",   style="white")
    table.add_column("Status",   justify="center")

    installed_set = {pkg for pkg, *_ in REQUIRED if pkg not in missing}
    for pkg, source, reason in REQUIRED:
        status = "[green]✓ OK[/green]" if pkg not in missing else "[red]✗ MISSING[/red]"
        table.add_row(pkg, source, reason, status)

    console.print(table)
    if missing:
        console.print(f"\n[red]Install missing:[/red] sudo pacman -S {' '.join(p for p in missing if p not in ('kglobalacceld','piper-tts'))} && yay -S {' '.join(p for p in missing if p in ('kglobalacceld','piper-tts'))}")
        raise typer.Exit(1)
    else:
        console.print("[green]All dependencies satisfied ✓[/green]")


# ──────────────────────────────────────────────────────────────────── #
#  list-devices                                                        #
# ──────────────────────────────────────────────────────────────────── #
@app.command(name="list-devices")
def list_devices() -> None:
    """List available audio input devices."""
    import sounddevice as sd
    table = Table(title="Microphone Devices")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name",  style="white")
    table.add_column("Channels", justify="center")
    table.add_column("Default SR", justify="right")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            default = " [green]★[/green]" if i == sd.default.device[0] else ""
            table.add_row(
                str(i),
                d["name"] + default,
                str(d["max_input_channels"]),
                str(int(d["default_samplerate"])),
            )
    console.print(table)


# ──────────────────────────────────────────────────────────────────── #
#  list-plugins                                                        #
# ──────────────────────────────────────────────────────────────────── #
@app.command(name="list-plugins")
def list_plugins() -> None:
    """List plugins found in the plugin directory."""
    settings = _load_settings()
    from nixorb.plugins.loader import PluginLoader
    loader = PluginLoader(settings.plugin_dir)
    loader.load_all()
    names = loader.plugin_names()
    tools = loader.get_tool_definitions()
    table = Table(title=f"Plugins  ({settings.plugin_dir})")
    table.add_column("Name",        style="cyan")
    table.add_column("Tool Name",   style="green")
    table.add_column("Description", style="white")
    for t in tools:
        fn = t.get("function", {})
        table.add_row("–", fn.get("name", "?"), fn.get("description", "")[:60])
    if not tools:
        console.print("[yellow]No plugins loaded.[/yellow]")
    else:
        console.print(table)


# ──────────────────────────────────────────────────────────────────── #
#  memory-search                                                       #
# ──────────────────────────────────────────────────────────────────── #
@app.command(name="memory-search")
def memory_search(
    query:   str = typer.Argument(..., help="Query string"),
    results: int = typer.Option(5, "--results", "-n"),
) -> None:
    """Search long-term vector memory."""
    settings = _load_settings()
    from nixorb.memory.vector_store import VectorMemory
    mem  = VectorMemory(settings.memory_dir)
    hits = mem.query(query, n_results=results)
    if not hits:
        console.print("[yellow]No memories found.[/yellow]")
        return
    table = Table(title=f"Memory results for: {query!r}")
    table.add_column("#",    style="cyan", justify="right")
    table.add_column("Text", style="white")
    for i, h in enumerate(hits, 1):
        table.add_row(str(i), h[:120])
    console.print(table)


# ──────────────────────────────────────────────────────────────────── #
#  memory-clear                                                        #
# ──────────────────────────────────────────────────────────────────── #
@app.command(name="memory-clear")
def memory_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Wipe all long-term memories."""
    if not yes:
        typer.confirm("Delete ALL NixOrb memories? This cannot be undone.", abort=True)
    settings = _load_settings()
    import shutil
    from pathlib import Path
    mem_dir = Path(settings.memory_dir)
    if mem_dir.exists():
        shutil.rmtree(mem_dir)
        mem_dir.mkdir(parents=True)
        console.print("[green]Memory cleared.[/green]")
    else:
        console.print("[yellow]Memory directory does not exist — nothing to clear.[/yellow]")


# ──────────────────────────────────────────────────────────────────── #
#  export-config / import-config                                       #
# ──────────────────────────────────────────────────────────────────── #
@app.command(name="export-config")
def export_config(
    output:   Path = typer.Option(Path("nixorb_backup.tar.gz.enc"), "--out", "-o"),
    password: str  = typer.Option("nixorb", "--password", "-p",
                                  help="Encryption password", prompt=True,
                                  hide_input=True, confirmation_prompt=True),
) -> None:
    """Export settings and memory into an encrypted archive."""
    from nixorb.utils.crypto import export_config as _export
    _export(_load_settings(), str(output), password)
    console.print(f"[green]✓ Exported →[/green] {output}")


@app.command(name="import-config")
def import_config(
    input_file: Path = typer.Argument(..., help="Encrypted archive", exists=True),
    password:   str  = typer.Option("nixorb", "--password", "-p",
                                    prompt=True, hide_input=True),
) -> None:
    """Restore settings and memory from an encrypted archive."""
    from nixorb.utils.crypto import import_config as _import
    _import(_load_settings(), str(input_file), password)
    console.print(f"[green]✓ Imported ←[/green] {input_file}")


# ──────────────────────────────────────────────────────────────────── #
#  version                                                             #
# ──────────────────────────────────────────────────────────────────── #
@app.command()
def version() -> None:
    """Print version information."""
    from nixorb import __version__
    import platform, torch
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[cyan]NixOrb[/cyan]",     __version__)
    table.add_row("[cyan]Python[/cyan]",     platform.python_version())
    table.add_row("[cyan]PyTorch[/cyan]",    torch.__version__)
    try:
        import PySide6
        table.add_row("[cyan]PySide6[/cyan]", PySide6.__version__)
    except Exception:
        pass
    try:
        import faster_whisper
        table.add_row("[cyan]faster-whisper[/cyan]", faster_whisper.__version__)
    except Exception:
        pass
    console.print(Panel(table, title="NixOrb Version Info", border_style="cyan"))


if __name__ == "__main__":
    app()
