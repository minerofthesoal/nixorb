"""NixOrb settings — Pydantic v2 settings with TOML persistence."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel

_CONFIG_ENV = "NIXORB_CONFIG"
_DEFAULT_CONFIG = Path.home() / ".config" / "nixorb" / "config.toml"


def _config_path() -> Path:
    return Path(os.environ[_CONFIG_ENV]) if _CONFIG_ENV in os.environ else _DEFAULT_CONFIG


class Settings(BaseModel):
    """NixOrb configuration — all user-tunable parameters."""

    # ── Orb UI ───────────────────────────────────────────────────── #
    orb_x: int | None = None
    orb_y: int | None = None
    orb_size: int = 120
    orb_opacity: float = 1.0
    hotkey: str = "Ctrl+Alt+Space"

    # ── ASR ──────────────────────────────────────────────────────── #
    asr_model: str = "large-v3"
    asr_language: str = "en"
    microphone_index: int | None = None
    mic_sensitivity: float = 0.5

    # ── LLM (Local-only: Ollama) ─────────────────────────────────── #
    llm_backend: str = "ollama"
    llm_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    llm_system_prompt: str = (
        "You are NixOrb, a helpful AI assistant running on Arch Linux with KDE Plasma 6. "
        "You have a witty, slightly sardonic personality like GLaDOS. "
        "Keep responses concise unless asked for detail. "
        "You can execute bash commands, search the web, capture the screen, and remember conversations."
    )
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7

    # ── TTS ──────────────────────────────────────────────────────── #
    tts_backend: str = "piper"
    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0
    tts_volume: float = 1.0

    # ── Wake Word ────────────────────────────────────────────────── #
    wake_word_enabled: bool = True
    wake_word_model: str = "hey_nixorb"
    wake_word_sensitivity: float = 0.5

    # ── Features ─────────────────────────────────────────────────── #
    screen_capture_enabled: bool = True
    web_search_enabled: bool = True
    clipboard_enabled: bool = True
    require_action_confirmation: bool = True
    memory_enabled: bool = True
    plugins_enabled: bool = True

    # ── VRAM ─────────────────────────────────────────────────────── #
    vram_total_mb: int = 8192
    vram_system_reserve_mb: int = 512
    vram_safety_buffer_mb: int = 256

    # ── Paths ────────────────────────────────────────────────────── #
    plugin_dir: str = str(Path.home() / ".local" / "share" / "nixorb" / "plugins")
    memory_dir: str = str(Path.home() / ".local" / "share" / "nixorb" / "memory")

    @classmethod
    def load(cls) -> Settings:
        """Load settings from config file, creating defaults if missing."""
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            try:
                with open(p, "rb") as f:
                    data = tomllib.load(f)
                return cls(**data)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    "Config load failed, using defaults: %s", exc
                )
        return cls()

    def save(self) -> None:
        """Persist current settings to config file."""
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.model_dump().items() if v is not None}
        with open(p, "wb") as f:
            tomli_w.dump(data, f)
