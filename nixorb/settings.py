"""nixorb/settings.py — Pydantic v2 settings with TOML persistence."""
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
    # ── Orb UI ───────────────────────────────────────────────────── #
    orb_x:    int | None = None
    orb_y:    int | None = None
    orb_size: int        = 120
    hotkey:   str        = "Ctrl+Alt+Space"

    # ── ASR ──────────────────────────────────────────────────────── #
    asr_model:        str      = "large-v3"
    asr_language:     str      = ""
    microphone_index: int | None = None

    # ── LLM ──────────────────────────────────────────────────────── #
    llm_backend:         str = "huggingface"
    # Default: GLaDOS-voice StableLM for local inference
    llm_model:           str = "torphix/stablelm-2-glados-v1"
    # Fast assistant model (0.5 B — runs on CPU if needed)
    llm_fast_model:      str = "google/gemma-4-31B-it-assistant"
    llm_base_url:        str = "https://api.openai.com/v1"
    openai_api_key:      str = ""
    hf_token:            str = ""
    local_model_path:    str = ""
    fallback_model_path: str = ""
    llm_vram_mb:         int = 4096
    llm_max_new_tokens:  int = 512

    # ── TTS ──────────────────────────────────────────────────────── #
    tts_backend:  str = "huggingface"
    # GLaDOS TTS voice
    tts_hf_repo:  str = "torphix/stablelm-2-glados-v1"
    tts_voice:    str = "alloy"

    # ── Vision ───────────────────────────────────────────────────── #
    vision_enabled:   bool = True
    # CogFlorence for captioning; Qwen3.5 for full vision+LLM
    vision_model:     str  = "thwri/CogFlorence-2.2-Large"
    vlm_model:        str  = "Qwen/Qwen3.5-4B"
    use_vlm:          bool = False   # True = use Qwen VLM; False = CogFlorence

    # ── Web search ───────────────────────────────────────────────── #
    web_search_enabled:     bool = True
    web_search_max_results: int  = 4

    # ── Features ─────────────────────────────────────────────────── #
    wake_word_enabled:            bool = False
    wake_word_model:              str  = "hey_jarvis_v0.1"
    screen_capture_enabled:       bool = True
    offline_fallback_enabled:     bool = True
    require_action_confirmation:  bool = True
    clipboard_enabled:            bool = True

    # ── Paths ────────────────────────────────────────────────────── #
    plugin_dir: str = str(Path.home() / ".local" / "share" / "nixorb" / "plugins")
    memory_dir: str = str(Path.home() / ".local" / "share" / "nixorb" / "memory")

    @classmethod
    def load(cls) -> Settings:
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
        p = _config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in self.model_dump().items() if v is not None}
        with open(p, "wb") as f:
            tomli_w.dump(data, f)
