"""nixorb/settings.py — Pydantic v2 settings with TOML persistence."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

import tomli_w
from pydantic import BaseModel, Field

CONFIG_PATH = Path.home() / ".config" / "nixorb" / "config.toml"


class Settings(BaseModel):
    # ── Orb UI ───────────────────────────────────────────────────── #
    orb_x: Optional[int] = None
    orb_y: Optional[int] = None
    hotkey: str          = "Ctrl+Alt+Space"

    # ── ASR ──────────────────────────────────────────────────────── #
    asr_model:        str          = "large-v3"
    asr_language:     str          = ""          # empty = auto-detect
    microphone_index: Optional[int] = None

    # ── LLM ──────────────────────────────────────────────────────── #
    llm_backend:       str = "openai"            # openai | local | ollama
    llm_model:         str = "gpt-4o-mini"
    llm_base_url:      str = "https://api.openai.com/v1"
    openai_api_key:    str = ""
    local_model_path:  str = ""
    fallback_model_path: str = ""
    llm_vram_mb:       int = 4096               # budget for local LLM

    # ── TTS ──────────────────────────────────────────────────────── #
    tts_backend:  str = "openai"                 # openai | huggingface | piper
    tts_voice:    str = "alloy"
    tts_hf_repo:  str = ""
    hf_token:     str = ""

    # ── Features ─────────────────────────────────────────────────── #
    wake_word_enabled:            bool = False
    wake_word_model:              str  = "hey_jarvis_v0.1"
    screen_capture_enabled:       bool = True
    offline_fallback_enabled:     bool = True
    require_action_confirmation:  bool = True
    clipboard_enabled:            bool = True

    # ── Paths ────────────────────────────────────────────────────── #
    plugin_dir:  str = str(Path.home() / ".local" / "share" / "nixorb" / "plugins")
    memory_dir:  str = str(Path.home() / ".local" / "share" / "nixorb" / "memory")

    # ── Appearance ───────────────────────────────────────────────── #
    orb_size: int = 120

    @classmethod
    def load(cls) -> "Settings":
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "rb") as f:
                    data = tomllib.load(f)
                return cls(**data)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    "Failed to load config, using defaults: %s", exc
                )
        return cls()

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump()
        # TOML doesn't support None values
        data = {k: v for k, v in data.items() if v is not None}
        with open(CONFIG_PATH, "wb") as f:
            tomli_w.dump(data, f)
