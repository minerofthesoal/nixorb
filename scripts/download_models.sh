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
    import openwakeword
    openwakeword.utils.download_models()
    print("  ✅ OpenWakeWord models downloaded.")
except Exception as e:
    print(f"  ⚠  OpenWakeWord download: {e}")
    print("     Try: python -m openwakeword --download_models hey_jarvis_v0.1")
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
