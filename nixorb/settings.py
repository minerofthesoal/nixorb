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
    orb_size: int = 88
    orb_opacity: float = 1.0
    hotkey: str = "Ctrl+Alt+Space"

    # ── ASR ──────────────────────────────────────────────────────── #
    # "faster-whisper" | "huggingface" (any ASR model on the Hub) | "nemotron"
    asr_backend: str = "faster-whisper"
    # For faster-whisper a size ("large-v3"); for the others a Hub repo id.
    asr_model: str = "large-v3"
    # Locale ("en-US"), bare code ("en"), or "auto" where the model supports it.
    asr_language: str = "en"
    microphone_index: int | None = None
    mic_sensitivity: float = 0.5

    # Nemotron right-context, in 80ms frames: 0 | 3 | 6 | 13, trading latency
    # (80/320/560/1120 ms) against accuracy.
    asr_nemotron_lookahead: int = 3
    # Emit partial transcripts while you are still speaking. Only Nemotron
    # can actually do this; other backends ignore it.
    asr_streaming: bool = False

    # ── LLM ──────────────────────────────────────────────────────── #
    # "ollama" (local daemon) | "huggingface" (any causal LM, in-process)
    llm_backend: str = "ollama"
    llm_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    # Used when llm_backend = "huggingface".
    llm_hf_model: str = "Qwen/Qwen2.5-3B-Instruct"
    # 4-bit quantisation, so a 7B fits alongside the ASR model on 8 GB.
    # Needs bitsandbytes.
    llm_hf_load_in_4bit: bool = False
    llm_system_prompt: str = (
        "You are NixOrb, a voice assistant on Arch Linux with KDE Plasma 6. "
        "You are being spoken to and your reply is read aloud, so answer the "
        "way a person would out loud: one or two short sentences, the answer "
        "first, no preamble, no markdown, no bullet lists, no code blocks "
        "unless asked. Never say 'as an AI'. Spell out numbers and units the "
        "way you would say them. If you need to run a shell command, put it "
        "in <ACTION>…</ACTION> and say briefly what you are doing. Only give "
        "a long answer when explicitly asked for detail. "
        "You can run commands, search the web, look at the screen, and "
        "remember earlier conversations."
    )
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    # Seconds to wait for the user to answer a command-confirmation dialog.
    action_confirm_timeout: float = 60.0

    # ── TTS ──────────────────────────────────────────────────────── #
    # "piper" (offline, AUR piper-tts) | "huggingface" | "espeak"
    tts_backend: str = "piper"
    tts_voice: str = "en_US-lessac-medium"
    tts_speed: float = 1.0
    tts_volume: float = 1.0
    # Speak each sentence as the model produces it, instead of waiting for
    # the whole answer. This is most of what makes the orb feel responsive.
    tts_streaming: bool = True
    # Used when tts_backend = "huggingface".
    tts_hf_model: str = "microsoft/speecht5_tts"
    # SpeechT5 speaker x-vector: a .npy path, a cmu-arctic-xvectors index,
    # or blank for the default voice.
    tts_hf_speaker: str = ""

    # ── Wake Word ────────────────────────────────────────────────── #
    # openwakeword ships as the optional "wakeword" extra, so this stays
    # off until the user installs it and opts in.
    wake_word_enabled: bool = False
    wake_word_model: str = "hey_nixorb"
    wake_word_sensitivity: float = 0.5

    # ── Conversation ─────────────────────────────────────────────── #
    # Triggering while the orb is talking stops it and starts listening,
    # rather than queueing behind the current answer.
    barge_in: bool = True
    # After answering, keep listening this long for a follow-up without
    # needing the hotkey again. 0 disables it.
    follow_up_seconds: float = 6.0
    # Cap spoken replies. Long answers are for the screen, not the speakers.
    tts_max_sentences: int = 6

    # ── Features ─────────────────────────────────────────────────── #
    screen_capture_enabled: bool = True
    web_search_enabled: bool = True
    web_search_max_results: int = 4
    clipboard_enabled: bool = True
    require_action_confirmation: bool = True
    # bubblewrap sandbox for <ACTION> commands. Off by default: the
    # sandbox is read-only with no network, so an approved command that
    # writes a file or installs a package fails for no visible reason.
    sandbox_actions: bool = False
    memory_enabled: bool = True
    plugins_enabled: bool = True

    # ── Hugging Face ─────────────────────────────────────────────── #
    # Only needed for gated or private repos. Falls back to $HF_TOKEN.
    hf_token: str = ""
    # "auto" | "cuda" | "cpu"
    hf_device: str = "auto"
    # Override the model cache location (default: ~/.cache/huggingface).
    hf_cache_dir: str = ""
    # Off by default: this executes Python from the model repo. Some models
    # (custom architectures) do not load without it.
    hf_trust_remote_code: bool = False

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
