"""NixOrb path utilities — find assets, config, data directories."""
from __future__ import annotations

import sys
from pathlib import Path


def _find_asset_root() -> Path:
    """Locate the assets directory across install methods."""
    # 1. Development / source tree
    src = Path(__file__).resolve().parents[2] / "assets"
    if src.exists():
        return src

    # 2. pip install / wheel, with assets bundled inside the package itself
    pip = Path(__file__).resolve().parents[1] / "assets"
    if pip.exists():
        return pip

    # 3. pip/uv/pipx install of our actual wheel: hatchling's "shared-data"
    # (see pyproject.toml) packs assets/ under `<wheel>.data/data/share/...`,
    # and the wheel spec extracts that relative to the *active interpreter's*
    # install base — sys.prefix — not into site-packages at all. For a venv
    # or a uv-managed Python (like this one) that is NOT /usr, so it has to
    # be resolved at runtime rather than assumed.
    prefix_share = Path(sys.prefix) / "share" / "nixorb" / "assets"
    if prefix_share.exists():
        return prefix_share

    # 4. system install via a distro package (e.g. an AUR PKGBUILD) that
    # installs straight to /usr/share rather than through pip/wheel data.
    system = Path("/usr/share/nixorb/assets")
    if system.exists():
        return system

    # 5. user local install (~/.local/share/nixorb/assets)
    local = Path.home() / ".local" / "share" / "nixorb" / "assets"
    if local.exists():
        return local

    # Fallback — return source location and let caller handle missing files
    return src


def _find_config_root() -> Path:
    """Locate the config directory."""
    src = Path(__file__).resolve().parents[2] / "config"
    if src.exists():
        return src

    pip = Path(__file__).resolve().parents[1] / "config"
    if pip.exists():
        return pip

    # See the matching comment in _find_asset_root(): hatchling's
    # shared-data lands under sys.prefix, which for a venv/uv install is
    # not /usr.
    prefix_share = Path(sys.prefix) / "share" / "nixorb" / "config"
    if prefix_share.exists():
        return prefix_share

    system = Path("/usr/share/nixorb/config")
    if system.exists():
        return system

    local = Path.home() / ".local" / "share" / "nixorb" / "config"
    if local.exists():
        return local

    return src


_ASSET_ROOT = _find_asset_root()
_CONFIG_ROOT = _find_config_root()


def asset_path(name: str) -> Path:
    """Get the full path to an asset file."""
    return _ASSET_ROOT / name


def config_path(name: str) -> Path:
    """Get the full path to a config file."""
    return _CONFIG_ROOT / name


def data_dir() -> Path:
    """Get the NixOrb data directory (~/.local/share/nixorb)."""
    d = Path.home() / ".local" / "share" / "nixorb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    """Get the NixOrb cache directory (~/.cache/nixorb)."""
    d = Path.home() / ".cache" / "nixorb"
    d.mkdir(parents=True, exist_ok=True)
    return d
