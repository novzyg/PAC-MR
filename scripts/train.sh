#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$OUTPUT_DIR"
common=(--data "$DATA_DIR" --split "$OUTPUT_DIR/split.json" --split-mode ordered --split-seed 2026 --device "$DEVICE" --dim "${DIM:-256}" --graph-layers "${GRAPH_LAYERS:-2}" --attention-heads "${ATTENTION_HEADS:-4}" --layers 1 --batch-size 4 --eval-batch-size 8 --seed "${SEED:-42}" --threshold-mode fixed --threshold 0.5 --resume)
"$PYTHON_BIN" "$ROOT/src/main.py" train "${common[@]}" --variant base --output "$OUTPUT_DIR/base" --epochs "${BASE_EPOCHS:-30}" --patience 8 --ddi-weight 0 --aux-weight 0 "$@"
"$PYTHON_BIN" "$ROOT/src/main.py" train "${common[@]}" --variant full --output "$OUTPUT_DIR/full" --init "$OUTPUT_DIR/base/best.pt" --epochs "${ADJUST_EPOCHS:-20}" --patience 8 --lr 0.001 --weight-decay 0 --ddi-weight "${DDI_WEIGHT:-1.0}" --aux-weight 0 "$@"
echo "Training complete: $OUTPUT_DIR/full/best.pt"
echo "Run scripts/evaluate.sh separately for test evaluation."
