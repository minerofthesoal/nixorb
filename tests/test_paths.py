"""tests/test_paths.py — asset/config root resolution across install methods.

_find_asset_root()/_find_config_root() try several candidate locations in
priority order and return the first that exists. This pins that order and,
specifically, the sys.prefix-based "wheel shared-data" candidate: hatchling
packs assets/config under `<wheel>.data/data/share/nixorb/...`, which the
wheel spec extracts relative to sys.prefix — not into site-packages, and
not necessarily /usr (a venv or a uv-managed Python is its own prefix).
Missing that candidate is exactly what left `asset_path("orb.qml")`
resolving to a path that never existed on a real venv/uv install.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from nixorb.utils import paths as paths_mod


def _only_this_exists(*wanted: Path):
    """Path.exists() returns True only for paths in `wanted`."""
    wanted_strs = {str(p) for p in wanted}

    def _exists(self) -> bool:
        return str(self) in wanted_strs

    return _exists


def test_dev_tree_wins_when_present(tmp_path):
    dev_root = tmp_path / "assets"
    with patch.object(paths_mod.Path, "exists", _only_this_exists(dev_root)), \
         patch.object(paths_mod, "__file__", str(tmp_path / "nixorb" / "utils" / "paths.py")):
        assert paths_mod._find_asset_root() == dev_root


def test_package_internal_wins_over_prefix_share(tmp_path):
    pkg_assets = tmp_path / "site-packages" / "nixorb" / "assets"
    prefix_share = tmp_path / "prefix" / "share" / "nixorb" / "assets"
    with patch.object(
        paths_mod.Path, "exists", _only_this_exists(pkg_assets, prefix_share)
    ), patch.object(
        paths_mod,
        "__file__",
        str(tmp_path / "site-packages" / "nixorb" / "utils" / "paths.py"),
    ), patch.object(paths_mod.sys, "prefix", str(tmp_path / "prefix")):
        assert paths_mod._find_asset_root() == pkg_assets


def test_sys_prefix_shared_data_used_when_nothing_else_matches(tmp_path):
    """The exact bug: a venv/uv wheel install with nothing under
    site-packages, /usr/share, or ~/.local/share — only sys.prefix/share."""
    prefix_share = tmp_path / "prefix" / "share" / "nixorb" / "assets"
    with patch.object(paths_mod.Path, "exists", _only_this_exists(prefix_share)), \
         patch.object(
             paths_mod,
             "__file__",
             str(tmp_path / "prefix" / "lib" / "site-packages" / "nixorb" / "utils" / "paths.py"),
         ), \
         patch.object(paths_mod.sys, "prefix", str(tmp_path / "prefix")), \
         patch.object(paths_mod.Path, "home", return_value=tmp_path / "home"):
        assert paths_mod._find_asset_root() == prefix_share


def test_falls_back_to_usr_share_for_distro_installs(tmp_path):
    usr_share = Path("/usr/share/nixorb/assets")
    with patch.object(paths_mod.Path, "exists", _only_this_exists(usr_share)), \
         patch.object(
             paths_mod,
             "__file__",
             str(tmp_path / "site-packages" / "nixorb" / "utils" / "paths.py"),
         ), \
         patch.object(paths_mod.sys, "prefix", str(tmp_path / "prefix")):
        assert paths_mod._find_asset_root() == usr_share


def test_falls_back_to_user_local_share(tmp_path):
    local_share = tmp_path / "home" / ".local" / "share" / "nixorb" / "assets"
    with patch.object(paths_mod.Path, "exists", _only_this_exists(local_share)), \
         patch.object(
             paths_mod,
             "__file__",
             str(tmp_path / "site-packages" / "nixorb" / "utils" / "paths.py"),
         ), \
         patch.object(paths_mod.sys, "prefix", str(tmp_path / "prefix")), \
         patch.object(paths_mod.Path, "home", return_value=tmp_path / "home"):
        assert paths_mod._find_asset_root() == local_share


def test_config_root_also_checks_sys_prefix_share(tmp_path):
    prefix_config = tmp_path / "prefix" / "share" / "nixorb" / "config"
    with patch.object(paths_mod.Path, "exists", _only_this_exists(prefix_config)), \
         patch.object(
             paths_mod,
             "__file__",
             str(tmp_path / "prefix" / "lib" / "site-packages" / "nixorb" / "utils" / "paths.py"),
         ), \
         patch.object(paths_mod.sys, "prefix", str(tmp_path / "prefix")), \
         patch.object(paths_mod.Path, "home", return_value=tmp_path / "home"):
        assert paths_mod._find_config_root() == prefix_config
