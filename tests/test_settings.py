"""tests/test_settings.py — Settings persistence tests."""
from __future__ import annotations

from nixorb.settings import _CONFIG_ENV, Settings


def test_defaults():
    """v2 is local-only: Ollama for the LLM, Piper for speech."""
    s = Settings()
    assert s.llm_backend == "ollama"
    assert s.llm_model == "llama3.2"
    assert s.ollama_host == "http://localhost:11434"
    assert s.tts_backend == "piper"
    assert s.tts_voice == "en_US-lessac-medium"
    assert s.hotkey == "Ctrl+Alt+Space"
    assert s.require_action_confirmation is True


def test_wake_word_defaults_off():
    """openwakeword is the optional 'wakeword' extra, so it cannot default on."""
    assert Settings().wake_word_enabled is False


def test_sandbox_defaults_off():
    """The bwrap sandbox is read-only with no network; opt-in, not default."""
    assert Settings().sandbox_actions is False


def test_web_search_max_results_exists():
    """main.py reads this on every web-search turn."""
    assert Settings().web_search_max_results >= 1


def test_save_and_reload(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))

    s = Settings(llm_model="mistral", orb_size=200)
    s.save()

    assert cfg.exists()
    s2 = Settings.load()
    assert s2.llm_model == "mistral"
    assert s2.orb_size == 200


def test_load_missing_config_returns_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "nonexistent.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    assert Settings.load().llm_backend == "ollama"


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
    Settings(llm_model="custom-model").save()
    assert cfg.exists()
    assert Settings.load().llm_model == "custom-model"


def test_unknown_keys_are_ignored(tmp_path, monkeypatch):
    """A config left over from an older version must not break startup."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('llm_model = "keepme"\nlegacy_removed_option = "gone"\n')
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    assert Settings.load().llm_model == "keepme"


def test_bad_value_falls_back_to_defaults(tmp_path, monkeypatch):
    """A corrupt config must not stop NixOrb from starting."""
    cfg = tmp_path / "config.toml"
    cfg.write_text('orb_size = "not a number"\n')
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    assert Settings.load().orb_size == 120


def test_malformed_toml_falls_back_to_defaults(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text("this is not [ valid toml =\n")
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    assert Settings.load().llm_model == "llama3.2"


def test_round_trip_preserves_every_field(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))
    original = Settings(orb_x=10, orb_y=20)
    original.save()
    assert Settings.load().model_dump() == original.model_dump()
