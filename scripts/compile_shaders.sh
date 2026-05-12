#!/usr/bin/env bash
# scripts/compile_shaders.sh — Compile QML GLSL shaders to .qsb binaries.
#
# qsb is part of qt6-tools. On Arch Linux it lives at /usr/lib/qt6/bin/qsb
# and is NOT in $PATH by default.
#
# Usage:
#   bash scripts/compile_shaders.sh
#   bash scripts/compile_shaders.sh --check   # just verify qsb exists

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"
SHADER_DIR="$PROJECT_ROOT/assets/shaders"
GLSL_VERSIONS="100es,120,150"

# ── Find qsb ─────────────────────────────────────────────────────── #
QSB=""
for candidate in \
    /usr/lib/qt6/bin/qsb \
    /usr/lib64/qt6/bin/qsb \
    "$(command -v qsb 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]]; then
        QSB="$candidate"
        break
    fi
done

if [[ -z "$QSB" ]]; then
    echo "❌ qsb not found."
    echo "   Install: sudo pacman -S qt6-tools"
    echo "   Expected: /usr/lib/qt6/bin/qsb"
    exit 1
fi

echo "✅ qsb found: $QSB"
"$QSB" --version | head -1

if [[ "${1:-}" == "--check" ]]; then
    echo "Check passed."
    exit 0
fi

# ── Compile each shader ───────────────────────────────────────────── #
compile_shader() {
    local src="$1"
    local out="${src%.vert}.vert.qsb"
    [[ "$src" == *.frag ]] && out="${src%.frag}.frag.qsb"
    echo "  Compiling: $(basename "$src") → $(basename "$out")"
    "$QSB" \
        --glsl "$GLSL_VERSIONS" \
        --hlsl 50 \
        --msl 12 \
        "$src" -o "$out"
}

echo ""
echo "==> Compiling NixOrb GLSL shaders…"

compile_shader "$SHADER_DIR/orb_glow.vert"
compile_shader "$SHADER_DIR/orb_glow.frag"
compile_shader "$SHADER_DIR/particle.vert"
compile_shader "$SHADER_DIR/particle.frag"

echo ""
echo "✅ All shaders compiled:"
ls -lh "$SHADER_DIR"/*.qsb
