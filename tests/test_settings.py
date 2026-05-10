"""tests/test_settings.py — Settings persistence tests."""
from __future__ import annotations

import pytest
from nixorb.settings import Settings


def test_defaults():
    s = Settings()
    assert s.llm_backend == "openai"
    assert s.tts_backend == "openai"
    assert s.hotkey == "Ctrl+Alt+Space"
    assert s.require_action_confirmation is True


def test_save_and_reload(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr("nixorb.settings.CONFIG_PATH", cfg)

    s = Settings(llm_model="gpt-4o", openai_api_key="sk-test123")
    s.save()

    assert cfg.exists()
    s2 = Settings.load()
    assert s2.llm_model == "gpt-4o"
    assert s2.openai_api_key == "sk-test123"


def test_load_missing_config_returns_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "nonexistent.toml"
    monkeypatch.setattr("nixorb.settings.CONFIG_PATH", cfg)
    s = Settings.load()
    assert s.llm_backend == "openai"


def test_none_values_not_in_toml(tmp_path, monkeypatch):
    """None values must be excluded from the TOML file (TOML doesn't support null)."""
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr("nixorb.settings.CONFIG_PATH", cfg)
    s = Settings(orb_x=None, orb_y=None)
    s.save()
    content = cfg.read_text()
    assert "None" not in content
    assert "null" not in content
