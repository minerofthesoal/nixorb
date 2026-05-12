#!/usr/bin/env bash
# nixorb-pkg/build-pkg.sh — Build the NixOrb pacman package on Arch Linux.
# Run from inside nixorb-pkg/:  bash build-pkg.sh
set -euo pipefail

[[ "$(uname -s)" == "Linux" ]] || { echo "Must run on Linux"; exit 1; }
command -v makepkg &>/dev/null || { echo "Install base-devel first: sudo pacman -S base-devel"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing makedepends..."
sudo pacman -S --needed --noconfirm \
    base-devel python python-build python-installer \
    python-wheel python-hatchling qt6-tools python-pip

echo ""
echo "==> Building nixorb pacman package..."
makepkg --noconfirm --cleanbuild 2>&1 | tee makepkg.log

PKG=$(ls nixorb-*.pkg.tar.zst 2>/dev/null | tail -1)
if [[ -z "$PKG" ]]; then
    echo "ERROR: Build failed. Check makepkg.log"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Package built: $PKG"
echo "╚══════════════════════════════════════════════════╝"
echo ""
read -rp "Install now? [y/N] " REPLY
[[ "${REPLY,,}" == "y" ]] && sudo pacman -U --noconfirm "$PKG"
