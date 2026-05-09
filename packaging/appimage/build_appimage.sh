#!/usr/bin/env bash
# packaging/appimage/build_appimage.sh
# Builds a portable NixOrb AppImage using appimage-builder.
#
# Prerequisites (Arch Linux):
#   sudo pacman -S python fuse2 wget
#   pip install appimage-builder
#
# Usage:
#   cd packaging/appimage
#   bash build_appimage.sh [version]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
VERSION="${1:-$(python3 -c 'import tomllib; d=tomllib.load(open("'"$PROJECT_ROOT/pyproject.toml"'","rb")); print(d["project"]["version"])')}"

echo "==> Building NixOrb AppImage v${VERSION}"
export NIXORB_VERSION="$VERSION"

# ── Create AppDir skeleton ────────────────────────────────────────── #
APPDIR="$SCRIPT_DIR/AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/nixorb" "$APPDIR/usr/share/icons/hicolor/256x256/apps"

# ── Install nixorb into AppDir ────────────────────────────────────── #
pip3 install --target="$APPDIR/usr/lib/python3/dist-packages" \
    --no-deps "$PROJECT_ROOT"

# Compile shaders
if command -v qsb &>/dev/null; then
    qsb --glsl "100es,120,150" --hlsl 50 --msl 12 \
        "$PROJECT_ROOT/assets/shaders/orb_glow.vert" \
        -o "$PROJECT_ROOT/assets/shaders/orb_glow.vert.qsb"
    qsb --glsl "100es,120,150" --hlsl 50 --msl 12 \
        "$PROJECT_ROOT/assets/shaders/orb_glow.frag" \
        -o "$PROJECT_ROOT/assets/shaders/orb_glow.frag.qsb"
fi

# ── Copy assets ───────────────────────────────────────────────────── #
cp -r "$PROJECT_ROOT/assets" "$APPDIR/usr/share/nixorb/"
cp -r "$PROJECT_ROOT/config" "$APPDIR/usr/share/nixorb/"
cp "$PROJECT_ROOT/assets/nixorb_256.png" \
   "$APPDIR/usr/share/icons/hicolor/256x256/apps/nixorb.png"
cp "$PROJECT_ROOT/assets/nixorb_256.png" "$APPDIR/nixorb.png"

# ── AppRun entrypoint ─────────────────────────────────────────────── #
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$HERE/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
export NIXORB_ASSETS="$HERE/usr/share/nixorb/assets"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland;xcb}"
exec python3 -m nixorb.cli "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# ── Desktop file ──────────────────────────────────────────────────── #
cat > "$APPDIR/nixorb.desktop" << 'DESKTOP'
[Desktop Entry]
Type=Application
Name=NixOrb
Exec=AppRun start
Icon=nixorb
Categories=Utility;
DESKTOP

# ── Run appimage-builder ──────────────────────────────────────────── #
cd "$SCRIPT_DIR"
appimage-builder --recipe AppImageBuilder.yml --skip-tests

OUTPUT="NixOrb-${VERSION}-x86_64.AppImage"
if [ -f "$OUTPUT" ]; then
    echo "==> SUCCESS: $OUTPUT"
    ls -lh "$OUTPUT"
else
    echo "==> WARNING: AppImage not found at expected path; check appimage-builder output above."
fi
