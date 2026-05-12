"""tests/test_crypto.py — Config export/import crypto tests."""
from __future__ import annotations

import pytest
from nixorb.settings import Settings, _CONFIG_ENV
from nixorb.utils.crypto import export_config, import_config


def test_roundtrip(tmp_path, monkeypatch):
    archive = tmp_path / "backup.tar.gz.enc"
    cfg     = tmp_path / "config.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))

    s = Settings(llm_model="claude-3", openai_api_key="sk-abc")
    s.save()

    export_config(s, str(archive), password="testpass")
    assert archive.exists()
    assert archive.stat().st_size > 100

    # Modify in-memory then restore
    s.llm_model = "something-else"
    import_config(s, str(archive), password="testpass")
    assert s.llm_model == "claude-3"


def test_wrong_password_raises(tmp_path, monkeypatch):
    archive = tmp_path / "backup.tar.gz.enc"
    cfg     = tmp_path / "config.toml"
    monkeypatch.setenv(_CONFIG_ENV, str(cfg))

    s = Settings()
    s.save()
    export_config(s, str(archive), password="correct")

    with pytest.raises(ValueError, match="Wrong password"):
        import_config(s, str(archive), password="wrong")
