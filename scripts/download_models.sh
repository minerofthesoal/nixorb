#!/usr/bin/env bash
# scripts/download_models.sh — Download default NixOrb AI models.
#
# Downloads:
#   • faster-whisper large-v3 (INT8) — ASR
#   • openwakeword hey_jarvis model — wake word
#
# HuggingFace models (GLaDOS LLM, CogFlorence, Qwen) are auto-downloaded
# by transformers on first use.
#
# Usage:
#   bash scripts/download_models.sh
#   bash scripts/download_models.sh --whisper-only
#   bash scripts/download_models.sh --wake-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/..")"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        NixOrb Model Downloader                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

MODE="${1:-all}"

# Prefer the installed CLI (keeps this script and `nixorb download-models`
# in sync via nixorb/utils/model_downloader.py) but fall back to a plain
# python invocation so this script still works from a fresh checkout
# before NixOrb itself is installed.
if command -v nixorb &>/dev/null; then
    case "$MODE" in
        --whisper-only) exec nixorb download-models --whisper-only ;;
        --wake-only)    exec nixorb download-models --wake-only ;;
        *)              exec nixorb download-models ;;
    esac
fi

echo "  [INFO] 'nixorb' CLI not found on PATH — falling back to a direct"
echo "         python invocation (this still works, just less integrated)."
echo ""

# ── Whisper Large v3 INT8 ─────────────────────────────────────────── #
download_whisper() {
    echo "==> Downloading faster-whisper large-v3 (INT8)…"
    echo "    ~1.6 GB — this may take a few minutes."
    python3 - << 'PYEOF'
from faster_whisper import WhisperModel
print("  Loading model (downloads if not cached)…")
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print("  ✅ faster-whisper large-v3 cached.")
del model
PYEOF
}

# ── OpenWakeWord models ───────────────────────────────────────────── #
download_wake_word() {
    echo ""
    echo "==> Downloading OpenWakeWord models…"
    python3 - << 'PYEOF'
try:
    import openwakeword.utils
    openwakeword.utils.download_models()
    print("  ✅ OpenWakeWord models downloaded.")
except Exception as e:
    print(f"  ⚠  OpenWakeWord download: {e}")
    print("     Try: pip install openwakeword")
PYEOF
}

case "$MODE" in
    --whisper-only) download_whisper ;;
    --wake-only)    download_wake_word ;;
    *)
        download_whisper
        download_wake_word
        ;;
esac

echo ""
echo "✅ Done. HuggingFace models (LLM, TTS, Vision) download automatically on first use."
echo "   Set your HF token if using gated models: nixorb config hf_token hf_xxxx"
