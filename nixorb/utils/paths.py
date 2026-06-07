"""Path helpers for bundled NixOrb assets and data files."""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_ENV_DATA_DIR = "NIXORB_DATA_DIR"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return the best available NixOrb data directory.

    Editable installs use the repository checkout. Wheels install assets via
    hatchling shared-data into ``<prefix>/share/nixorb``. Packagers may override
    with ``NIXORB_DATA_DIR``.
    """
    candidates: list[Path] = []
    if override := os.environ.get(_ENV_DATA_DIR):
        candidates.append(Path(override).expanduser())

    root = repo_root()
    candidates.append(root)

    for key in ("data", "platdata"):
        base = sysconfig.get_paths().get(key)
        if base:
            candidates.append(Path(base) / "share" / "nixorb")

    candidates.extend(
        [
            Path(sys.prefix) / "share" / "nixorb",
            Path(sys.base_prefix) / "share" / "nixorb",
            Path("/usr/local/share/nixorb"),
            Path("/usr/share/nixorb"),
        ]
    )

    for candidate in candidates:
        if (candidate / "assets").exists():
            return candidate
    return candidates[0]


def asset_path(*parts: str) -> Path:
    return data_dir() / "assets" / Path(*parts)


def config_path(*parts: str) -> Path:
    return data_dir() / "config" / Path(*parts)
