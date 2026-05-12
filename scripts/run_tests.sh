#!/usr/bin/env bash
# scripts/run_tests.sh — Run the NixOrb test suite.
# Tests requiring GPU/audio hardware are skipped automatically.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate venv if present
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Determine skip list based on available hardware
SKIP=""
python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null || \
    SKIP="$SKIP --ignore=tests/test_vram_manager.py"
python3 -c "import sounddevice; sounddevice.query_devices()" 2>/dev/null || \
    SKIP="$SKIP --ignore=tests/test_asr.py --ignore=tests/test_clipboard.py"
python3 -c "import chromadb" 2>/dev/null || \
    SKIP="$SKIP --ignore=tests/test_memory.py"

echo "==> Running NixOrb tests (skipping: ${SKIP:-none})"
python3 -m pytest tests/ $SKIP -v --tb=short "$@"
