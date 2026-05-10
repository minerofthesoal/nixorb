"""
nixorb/cli.py — NixOrb command-line interface.

Usage:
  nixorb start                launch orb GUI
  nixorb start --headless     daemon only, no Qt
  nixorb ask "query"          one-shot LLM
  nixorb tts "text"           speak text
  nixorb transcribe file.wav  transcribe audio
  nixorb config               show all config
  nixorb config key           read one config key
  nixorb config key value     set a config key
  nixorb config-gui           open Settings window standalone
  nixorb check-deps           verify Arch packages
  nixorb list-devices         list microphones
  nixorb list-plugins         show plugins
  nixorb memory-search query  search memories
  nixorb memory-clear         delete all memories
  nixorb export-config        encrypted backup
  nixorb import-config file   restore backup
  nixorb version              show version info
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app     = typer.Typer(
    name="nixorb",
    help="NixOrb — floating AI assistant for Arch Linux.",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_settings(config: Path | None = None):
    if config:
        os.environ["NIXORB_CONFIG"] = str(config)
    from nixorb.settings import Settings
    return Settings.load()


def _ask_password() -> str:
    import getpass
    return getpass.getpass("Encryption password: ") or "nixorb"


# ── start ─────────────────────────────────────────────────────────── #

@app.command()
def start(
    headless: bool       = typer.Option(False, "--headless", "-H", help="No GUI"),
    debug:    bool       = typer.Option(False, "--debug",    "-d", help="Debug logging"),
    config:   Path | None = typer.Option(None,  "--config",   "-c", help="Config TOML path"),
) -> None:
    """[bold green]Start the NixOrb daemon and orb window.[/bold green]"""
    _setup_logging(debug)
    settings = _load_settings(config)
    if headless:
        console.print(Panel("[yellow]Headless mode — no GUI[/yellow]", title="NixOrb"))
        asyncio.run(_headless_daemon(settings))
    else:
        from nixorb.main import main as _gui_main
        _gui_main()


async def _headless_daemon(settings) -> None:
    import signal as _signal
    from nixorb.core.event_bus import bus
    from nixorb.core.vram_manager import vram
    from nixorb.action.executor import ActionExecutor
    from nixorb.asr.whisper_engine import WhisperEngine
    from nixorb.tts.tts_factory import build_tts

    await bus.start()
    await vram.start_monitor()
    WhisperEngine(settings)
    build_tts(settings)
    ActionExecutor(settings)
    stop = asyncio.Event()

    def _stop(*_) -> None:
        stop.set()

    _signal.signal(_signal.SIGINT,  _stop)
    _signal.signal(_signal.SIGTERM, _stop)
    console.print("[green]Headless daemon running. Ctrl-C to quit.[/green]")
    await stop.wait()
    await vram.stop()
    await bus.stop()


# ── ask ───────────────────────────────────────────────────────────── #

@app.command()
def ask(
    prompt:  str        = typer.Argument(..., help="Question for the LLM"),
    model:   str | None = typer.Option(None, "--model", "-m"),
    config:  Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Send a one-shot text prompt and print the response."""
    settings = _load_settings(config)
    if model:
        settings.llm_model = model

    async def _run() -> None:
        from nixorb.core.event_bus import bus
        from nixorb.core.vram_manager import vram
        from nixorb.llm.backends import HuggingFaceBackend, OllamaBackend, OpenAIBackend

        await bus.start()
        await vram.start_monitor()
        b = settings.llm_backend.lower()
        if b == "openai":
            llm = OpenAIBackend(settings.openai_api_key, settings.llm_model, settings.llm_base_url)
        elif b == "ollama":
            llm = OllamaBackend(settings.llm_model)
        else:
            llm = HuggingFaceBackend(settings.llm_model, settings.hf_token)
        async for chunk in llm.stream([{"role": "user", "content": prompt}]):
            print(chunk, end="", flush=True)
        print()
        await vram.stop()
        await bus.stop()

    asyncio.run(_run())


# ── tts ───────────────────────────────────────────────────────────── #

@app.command()
def tts(
    text:   str        = typer.Argument(..., help="Text to speak"),
    voice:  str | None = typer.Option(None, "--voice", "-v"),
    config: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Speak text using the configured TTS engine."""
    settings = _load_settings(config)
    if voice:
        settings.tts_voice = voice

    async def _run() -> None:
        from nixorb.core.event_bus import bus
        from nixorb.tts.tts_factory import build_tts
        await bus.start()
        await build_tts(settings).speak(text)
        await bus.stop()

    asyncio.run(_run())


# ── transcribe ────────────────────────────────────────────────────── #

@app.command()
def transcribe(
    audio_file: Path       = typer.Argument(..., help="Audio file path"),
    language:   str | None = typer.Option(None, "--language", "-l"),
    config:     Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Transcribe an audio file with Whisper Large v3."""
    if not audio_file.exists():
        console.print(f"[red]File not found: {audio_file}[/red]")
        raise typer.Exit(1)
    settings = _load_settings(config)
    if language:
        settings.asr_language = language

    async def _run() -> None:
        import soundfile as sf
        from nixorb.asr.whisper_engine import WhisperEngine
        from nixorb.core.event_bus import bus
        from nixorb.core.vram_manager import vram

        await bus.start()
        await vram.start_monitor()
        audio, sr = sf.read(str(audio_file), dtype="float32", always_2d=False)
        if sr != 16_000:
            console.print(f"[yellow]Resampling {sr} Hz → 16000 Hz[/yellow]")
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
            except ImportError:
                console.print("[red]pip install librosa for resampling[/red]")
        text = await WhisperEngine(settings)._transcribe_async(audio)  # noqa: SLF001
        console.print(f"\n[bold]Transcript:[/bold] {text or '(empty)'}\n")
        await vram.stop()
        await bus.stop()

    asyncio.run(_run())


# ── config ────────────────────────────────────────────────────────── #

@app.command()
def config(
    key:         str | None = typer.Argument(None, help="Key to read or set"),
    value:       str | None = typer.Argument(None, help="New value (omit to read)"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """
    View or edit config values from the terminal.

    Examples:\n
      nixorb config                     # show all settings\n
      nixorb config llm_model           # read one key\n
      nixorb config llm_model gpt-4o    # set a key\n
    """
    settings = _load_settings(config_path)

    if key is None:
        t = Table("Key", "Value", title="NixOrb Config", show_lines=False)
        for k, v in settings.model_dump().items():
            display = str(v)
            if any(s in k for s in ("key", "token", "password")):
                display = "••••" if v else "(not set)"
            t.add_row(k, display)
        console.print(t)
        return

    if not hasattr(settings, key):
        console.print(f"[red]Unknown key: {key}[/red]")
        raise typer.Exit(1)

    if value is None:
        console.print(f"[bold]{key}[/bold] = {getattr(settings, key)!r}")
        return

    current = getattr(settings, key)
    if isinstance(current, bool):
        typed: bool | int | str = value.lower() in ("true", "1", "yes")
    elif isinstance(current, int):
        typed = int(value)
    else:
        typed = value

    setattr(settings, key, typed)
    settings.save()
    console.print(f"[green]✅[/green] [bold]{key}[/bold] = {typed!r}")


# ── config-gui ────────────────────────────────────────────────────── #

@app.command(name="config-gui")
def config_gui(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Open the Settings window as a standalone app (no orb required)."""
    settings = _load_settings(config_path)
    from PySide6.QtWidgets import QApplication
    from nixorb.ui.settings_window import SettingsWindow

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("NixOrb Settings")
    qt_app.setQuitOnLastWindowClosed(True)
    SettingsWindow.init_settings(settings)
    win = SettingsWindow(settings)
    win.show()
    qt_app.exec()


# ── check-deps ────────────────────────────────────────────────────── #

@app.command(name="check-deps")
def check_deps() -> None:
    """Verify all required Arch Linux packages are installed."""
    from nixorb.core.aur_checker import check_dependencies
    missing = check_dependencies()
    if missing:
        console.print(f"[red]Missing:[/red] {', '.join(missing)}")
        raise typer.Exit(1)
    console.print("[green]✅  All dependencies satisfied[/green]")


# ── list-devices ──────────────────────────────────────────────────── #

@app.command(name="list-devices")
def list_devices() -> None:
    """List available microphone input devices."""
    from nixorb.utils.audio import list_input_devices
    t = Table("Index", "Name", "Channels", "Sample Rate")
    for d in list_input_devices():
        t.add_row(str(d["index"]), d["name"], str(d["channels"]), str(d["sample_rate"]))
    console.print(t)


# ── list-plugins ──────────────────────────────────────────────────── #

@app.command(name="list-plugins")
def list_plugins(config_path: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """Show currently loaded plugins and their tools."""
    settings = _load_settings(config_path)
    from nixorb.plugins.loader import PluginLoader
    loader = PluginLoader(settings.plugin_dir)
    loader.load_all()
    names = loader.plugin_names()
    if not names:
        console.print("[yellow]No plugins loaded[/yellow]")
        return
    t = Table("Plugin", "Tools")
    for name in names:
        tools = [d["function"]["name"] for d in loader.get_tool_definitions()]
        t.add_row(name, ", ".join(tools) or "—")
    console.print(t)


# ── memory-search ─────────────────────────────────────────────────── #

@app.command(name="memory-search")
def memory_search(
    query:       str      = typer.Argument(...),
    n_results:   int      = typer.Option(5, "--n", "-n"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Search long-term vector memory."""
    settings = _load_settings(config_path)
    from nixorb.memory.vector_store import VectorMemory
    results = VectorMemory(settings.memory_dir).query(query, n_results)
    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return
    for i, r in enumerate(results, 1):
        console.print(f"[bold]{i}.[/bold] {r[:200]}")


# ── memory-clear ──────────────────────────────────────────────────── #

@app.command(name="memory-clear")
def memory_clear(
    yes:         bool     = typer.Option(False, "--yes", "-y"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Delete all long-term memories."""
    if not yes:
        typer.confirm("Delete ALL memories?", abort=True)
    import shutil
    settings = _load_settings(config_path)
    shutil.rmtree(settings.memory_dir, ignore_errors=True)
    console.print("[green]Memory cleared.[/green]")


# ── export-config ─────────────────────────────────────────────────── #

@app.command(name="export-config")
def export_config(
    output:      Path      = typer.Option(Path("nixorb_backup.tar.gz.enc"), "--out", "-o"),
    password:    str | None = typer.Option(None, "--password", "-p"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Export settings + memory to an encrypted archive."""
    settings = _load_settings(config_path)
    from nixorb.utils.crypto import export_config as _exp
    _exp(settings, str(output), password or _ask_password())
    console.print(f"[green]Exported →[/green] {output}")


# ── import-config ─────────────────────────────────────────────────── #

@app.command(name="import-config")
def import_config(
    input_file:  Path      = typer.Argument(...),
    password:    str | None = typer.Option(None, "--password", "-p"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Import settings + memory from an encrypted archive."""
    if not input_file.exists():
        console.print(f"[red]Not found: {input_file}[/red]")
        raise typer.Exit(1)
    settings = _load_settings(config_path)
    from nixorb.utils.crypto import import_config as _imp
    _imp(settings, str(input_file), password or _ask_password())
    console.print(f"[green]Imported ←[/green] {input_file}")


# ── version ───────────────────────────────────────────────────────── #

@app.command()
def version() -> None:
    """Show version and runtime info."""
    from nixorb import __version__
    console.print(f"[bold]NixOrb[/bold] {__version__}")
    try:
        import torch
        cuda = "[green]yes[/green]" if torch.cuda.is_available() else "[red]no[/red]"
        console.print(f"  PyTorch {torch.__version__}  CUDA: {cuda}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            console.print(f"  VRAM {free//1024**2} MB free / {total//1024**2} MB")
    except ImportError:
        console.print("  [yellow]PyTorch not installed[/yellow]")
    try:
        import PySide6.QtCore as qc
        console.print(f"  Qt {qc.__version__}")
    except ImportError:
        console.print("  [yellow]PySide6 not installed[/yellow]")


if __name__ == "__main__":
    app()
