#!/bin/bash
# Usage: run_train.sh <SEED> <TAG> [EXTRA_ARGS...]
set -u
SEED=$1; TAG=$2; shift 2
SHARED=/mnt/shared-workspace/r2m
OUT=/workspace/out_${TAG}
mkdir -p "$OUT"

python3 "$SHARED/v52_rna2meth.py" \
  --preprocessed "$SHARED/preprocessed_v52.npz" \
  --out_dir "$OUT" --seed "$SEED" --epochs 80 --patience 10 "$@" \
  > "$OUT/train.log" 2>&1
RC=$?

# final sync of all outputs
DEST=$SHARED/results_${TAG}
mkdir -p "$DEST"
cp -f "$OUT"/train.log "$OUT"/training_log.csv "$OUT"/test_summary.json \
      "$OUT"/test_gene_metrics.csv "$OUT"/test_sample_metrics.csv "$DEST"/ 2>/dev/null
cp -f "$OUT"/test_predictions.npz "$OUT"/val_predictions.npz "$OUT"/best_model.pth "$DEST"/ 2>/dev/null
echo "training exited rc=$RC, outputs synced to $DEST"
