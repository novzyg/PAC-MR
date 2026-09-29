#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
exec "$PYTHON_BIN" "$ROOT/src/main.py" evaluate --run "$OUTPUT_DIR/full" --device "$DEVICE" "$@"
