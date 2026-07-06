"""tests/test_settings.py — Settings persistence tests."""
from __future__ import annotations

from nixorb.settings import _CONFIG_ENV, Settings


def test_defaults():
    s = Settings()
    assert s.llm_backend == "huggingface"
    # "glados" is the default because it has an automatic SpeechT5 fallback
    # baked in, so TTS actually produces audio out of the box. The old
    # default ("huggingface" backend + tts_hf_repo pointed at a text-only
    # StableLM checkpoint) silently produced no audio at all — see
    # nixorb/tts/hf_tts.py and nixorb/tts/glados_tts.py.
    assert s.tts_backend == "glados"
    assert s.tts_hf_repo == "microsoft/speecht5_tts"
    assert s.hotkey == "Ctrl+Alt+Space"
    assert s.require_action_confirmation is True


def test_save_and_reload(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))

    s = Settings(llm_model="gpt-4o", openai_api_key="sk-test123")
    s.save()

    assert cfg.exists()
    s2 = Settings.load()
    assert s2.llm_model == "gpt-4o"
    assert s2.openai_api_key == "sk-test123"


def test_load_missing_config_returns_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "nonexistent.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    s = Settings.load()
    assert s.llm_backend == "huggingface"


def test_none_values_not_in_toml(tmp_path, monkeypatch):
    """None values must be excluded from the TOML file."""
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    s = Settings(orb_x=None, orb_y=None)
    s.save()
    content = cfg.read_text()
    assert "None" not in content
    assert "null" not in content


def test_config_env_override(tmp_path, monkeypatch):
    """NIXORB_CONFIG env var overrides the default config path."""
    cfg = tmp_path / "custom.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    s = Settings(llm_model="custom-model")
    s.save()
    assert cfg.exists()
    s2 = Settings.load()
    assert s2.llm_model == "custom-model"
