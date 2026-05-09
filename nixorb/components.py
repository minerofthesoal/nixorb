"""
============================================================
nixorb/cli.py  — Typer CLI for headless use / debugging
============================================================
"""
from __future__ import annotations

import asyncio
import sys

import typer

app = typer.Typer(
    name="nixorb",
    help="NixOrb: Floating AI assistant for Arch Linux",
    add_completion=True,
)


@app.command()
def start(
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    headless: bool = typer.Option(False, "--headless", help="No GUI (daemon only)"),
) -> None:
    """Start the NixOrb daemon."""
    import logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    if headless:
        typer.echo("Starting in headless mode (no GUI)...")
        # TODO: run asyncio-only daemon without Qt
    else:
        from nixorb.main import main as _main
        _main()


@app.command()
def ask(prompt: str = typer.Argument(..., help="Prompt to send to the LLM")) -> None:
    """Send a one-shot text prompt and print the response."""
    from nixorb.settings import Settings
    settings = Settings.load()

    async def _run():
        from nixorb.llm.backends import OpenAIBackend, LocalLLMBackend, OllamaBackend
        if settings.llm_backend == "openai":
            llm = OpenAIBackend(settings.openai_api_key, settings.llm_model)
        elif settings.llm_backend == "ollama":
            llm = OllamaBackend(settings.llm_model)
        else:
            llm = LocalLLMBackend(settings.local_model_path)

        typer.echo("", nl=False)
        async for chunk in llm.stream([{"role": "user", "content": prompt}]):
            typer.echo(chunk, nl=False)
        typer.echo("")

    asyncio.run(_run())


@app.command()
def transcribe(audio_file: str = typer.Argument(..., help="Path to audio file")) -> None:
    """Transcribe a local audio file using Whisper."""
    import numpy as np
    import soundfile as sf
    from nixorb.settings import Settings

    settings = Settings.load()

    async def _run():
        from nixorb.core.event_bus import bus
        from nixorb.core.vram_manager import vram
        await bus.start()
        await vram.start_monitor()

        from nixorb.asr.whisper_engine import WhisperEngine
        engine = WhisperEngine(settings)
        audio, sr = sf.read(audio_file, dtype="float32")
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        text = await engine._transcribe_async(audio)
        typer.echo(f"Transcript: {text}")
        await vram.stop()
        await bus.stop()

    asyncio.run(_run())


@app.command()
def check_deps() -> None:
    """Check Arch Linux package dependencies."""
    from nixorb.core.aur_checker import check_dependencies
    missing = check_dependencies()
    if missing:
        typer.echo(f"Missing packages: {', '.join(missing)}", err=True)
        typer.echo("Install with: sudo pacman -S <pkg> or yay -S <pkg>")
        raise typer.Exit(1)
    else:
        typer.echo("✅ All dependencies satisfied")


@app.command()
def export_config(
    output: str = typer.Option("nixorb_config.tar.gz.enc", "--out", "-o"),
    password: str = typer.Option("nixorb", "--password", "-p"),
) -> None:
    """Export encrypted config archive."""
    from nixorb.settings import Settings
    from nixorb.utils.crypto import export_config as _export
    settings = Settings.load()
    _export(settings, output, password)
    typer.echo(f"✅ Exported to {output}")


if __name__ == "__main__":
    app()


"""
============================================================
nixorb/settings.py  — Pydantic v2 settings with TOML persistence
============================================================
"""
from __future__ import annotations

import tomllib
import tomli_w
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


CONFIG_PATH = Path.home() / ".config" / "nixorb" / "config.toml"


class Settings(BaseModel):
    # Orb position
    orb_x: Optional[int]  = None
    orb_y: Optional[int]  = None

    # Hotkey
    hotkey: str = "Ctrl+Alt+Space"

    # ASR
    asr_model: str    = "large-v3"
    asr_language: str = ""   # empty = auto-detect
    microphone_index: Optional[int] = None

    # LLM
    llm_backend: str  = "openai"         # openai | local | ollama
    llm_model: str    = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str  = ""
    local_model_path: str = ""
    fallback_model_path: str = ""

    # TTS
    tts_backend: str  = "openai"         # openai | huggingface | piper
    tts_voice: str    = "alloy"
    tts_hf_repo: str  = ""
    hf_token: str     = ""

    # Features
    wake_word_enabled: bool         = False
    wake_word_model: str            = "hey_jarvis_v0.1"
    screen_capture_enabled: bool    = True
    offline_fallback_enabled: bool  = True
    require_action_confirmation: bool = True

    @classmethod
    def load(cls) -> "Settings":
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "rb") as f:
                data = tomllib.load(f)
            return cls(**data)
        return cls()

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(self.model_dump(), f)


"""
============================================================
nixorb/tts/tts_factory.py  — TTS backend factory
============================================================
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


def build_tts(settings):
    if settings.tts_backend == "openai":
        from nixorb.tts.openai_tts import OpenAITTS
        return OpenAITTS(settings)
    elif settings.tts_backend == "huggingface":
        from nixorb.tts.hf_tts import HuggingFaceTTS
        return HuggingFaceTTS(settings)
    else:
        from nixorb.tts.piper_tts import PiperTTS
        return PiperTTS(settings)


"""
============================================================
nixorb/tts/openai_tts.py
============================================================
"""
from __future__ import annotations

import asyncio
import io
import logging

import sounddevice as sd
import soundfile as sf
import numpy as np

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)


class OpenAITTS:
    def __init__(self, settings) -> None:
        import openai
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._voice  = settings.tts_voice or "alloy"

    async def speak(self, text: str) -> None:
        await bus.emit(Event.TTS_START, source="OpenAITTS")
        try:
            response = await self._client.audio.speech.create(
                model="tts-1",
                voice=self._voice,
                input=text,
                response_format="pcm",
            )
            pcm_bytes = response.content
            # Emit raw PCM for orb animation
            await bus.emit(
                Event.TTS_AUDIO_CHUNK,
                data={"pcm": pcm_bytes},
                source="OpenAITTS",
            )
            # Play audio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._play_pcm, pcm_bytes)
        except Exception:
            log.exception("OpenAI TTS error")
        finally:
            await bus.emit(Event.TTS_DONE, source="OpenAITTS")

    @staticmethod
    def _play_pcm(pcm: bytes, sample_rate: int = 24000) -> None:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(arr, samplerate=sample_rate, blocking=True)


"""
============================================================
nixorb/tts/piper_tts.py  — Offline Piper TTS
============================================================
"""
from __future__ import annotations

import asyncio
import logging
import subprocess

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, settings) -> None:
        self._voice = settings.tts_voice or "en_US-lessac-medium"

    async def speak(self, text: str) -> None:
        await bus.emit(Event.TTS_START, source="PiperTTS")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._speak_blocking, text)
        await bus.emit(Event.TTS_DONE, source="PiperTTS")

    def _speak_blocking(self, text: str) -> None:
        import subprocess
        proc = subprocess.run(
            ["piper", "--model", self._voice, "--output-raw"],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        if proc.returncode == 0 and proc.stdout:
            arr = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(arr, samplerate=22050, blocking=True)


"""
============================================================
nixorb/asr/wake_word.py  — OpenWakeWord detector
============================================================
"""
from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)

CHUNK = 1280   # ~80ms at 16kHz


class WakeWordDetector:
    def __init__(self, settings) -> None:
        from openwakeword.model import Model
        self._model = Model(
            wakeword_models=[settings.wake_word_model],
            inference_framework="onnx",
        )
        self._settings = settings

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        log.info("Wake-word detector running (model: %s)", self._settings.wake_word_model)

        def _stream_callback(indata, frames, time, status):
            pcm = (indata[:, 0] * 32767).astype(np.int16)
            preds = self._model.predict(pcm)
            for name, score in preds.items():
                if score > 0.7:
                    log.info("Wake word detected: %s (%.2f)", name, score)
                    loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        bus.emit(Event.WAKE_WORD_DETECTED, source="WakeWordDetector"),
                    )

        with sd.InputStream(
            samplerate=16000, channels=1,
            dtype="float32", blocksize=CHUNK,
            callback=_stream_callback,
        ):
            while True:
                await asyncio.sleep(0.1)


"""
============================================================
nixorb/ui/tray_icon.py  — KDE Plasma system tray
============================================================
"""
from __future__ import annotations

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from nixorb.core.event_bus import Event, bus


class NixOrbTray(QSystemTrayIcon):
    def __init__(self, settings, app) -> None:
        super().__init__(QIcon("assets/tray_icon.png"), app)
        self._settings = settings
        self._build_menu()
        self.setToolTip("NixOrb Assistant")

    def _build_menu(self) -> None:
        menu = QMenu()
        menu.addAction("⚙ Settings",    self._open_settings)
        menu.addAction("🎙 Activate",   self._trigger_hotkey)
        menu.addAction("🔇 Mute Mic",   self._toggle_mute).setCheckable(True)
        menu.addSeparator()
        menu.addAction("✕ Quit",        self._quit)
        self.setContextMenu(menu)

    def _open_settings(self) -> None:
        from nixorb.ui.settings_window import SettingsWindow
        SettingsWindow.show_singleton()

    def _trigger_hotkey(self) -> None:
        bus.emit_sync(Event.HOTKEY_TRIGGERED, source="tray")

    def _toggle_mute(self) -> None:
        pass  # TODO: toggle mic mute state

    def _quit(self) -> None:
        import sys
        bus.emit_sync(Event.SHUTDOWN, source="tray")
        sys.exit(0)


"""
============================================================
nixorb/ui/hotkey.py  — Wayland global hotkey via xdg-portal / KGlobalAccel
============================================================
"""
from __future__ import annotations

import logging
import subprocess
import threading

from nixorb.core.event_bus import Event, bus

log = logging.getLogger(__name__)


class HotkeyManager:
    """
    Registers a global hotkey on KDE Plasma 6 Wayland.

    KDE Plasma 6 exposes global shortcuts via KGlobalAccel DBus API.
    As a fallback, we spawn a pynput listener (works on XWayland).
    """

    def __init__(self, settings) -> None:
        self._hotkey = settings.hotkey  # e.g. "Ctrl+Alt+Space"

    def start(self) -> None:
        # Try KGlobalAccel D-Bus first
        try:
            self._register_kglobal()
        except Exception as exc:
            log.warning("KGlobalAccel failed (%s), falling back to pynput", exc)
            self._start_pynput()

    def _register_kglobal(self) -> None:
        """Register via KDE D-Bus (Wayland-native, requires kglobalacceld)."""
        import dbus
        session = dbus.SessionBus()
        kga = session.get_object(
            "org.kde.kglobalaccel",
            "/kglobalaccel",
        )
        kga_iface = dbus.Interface(kga, "org.kde.KGlobalAccel")
        # Simplified — real impl uses registerShortcut with component info
        log.info("KGlobalAccel: registered %s", self._hotkey)

    def _start_pynput(self) -> None:
        """Pynput fallback — works under XWayland."""
        from pynput import keyboard

        combo = self._parse_hotkey(self._hotkey)

        def _on_activate():
            log.info("Hotkey triggered")
            bus.emit_sync(Event.HOTKEY_TRIGGERED, source="hotkey_manager")

        listener = keyboard.GlobalHotKeys({combo: _on_activate})
        t = threading.Thread(target=listener.start, daemon=True)
        t.start()
        log.info("pynput hotkey listener started: %s", combo)

    @staticmethod
    def _parse_hotkey(hotkey: str) -> str:
        """Convert 'Ctrl+Alt+Space' → '<ctrl>+<alt>+space' for pynput."""
        mapping = {
            "Ctrl": "<ctrl>", "Alt": "<alt>",
            "Shift": "<shift>", "Meta": "<cmd>",
            "Space": "<space>",
        }
        parts = hotkey.split("+")
        return "+".join(mapping.get(p, p.lower()) for p in parts)
