#!/bin/bash
# Usage: run_job.sh <TAG> <SEED> <EPOCHS> [EXTRA_ARGS...]
set -u
TAG=$1; SEED=$2; EPOCHS=$3; shift 3
SHARED=/mnt/shared-workspace/r2m
OUT=/workspace/out_${TAG}
mkdir -p "$OUT"
python3 "$SHARED/v52_rna2meth.py" \
  --preprocessed "$SHARED/preprocessed_v52.npz" \
  --out_dir "$OUT" --seed "$SEED" --epochs "$EPOCHS" --patience 10 "$@" \
  > "$OUT/train.log" 2>&1
RC=$?
DEST=$SHARED/results_${TAG}
mkdir -p "$DEST"
cp -f "$OUT"/train.log "$OUT"/training_log.csv "$OUT"/test_summary.json \
      "$OUT"/test_gene_metrics.csv "$OUT"/test_sample_metrics.csv "$DEST"/ 2>/dev/null
cp -f "$OUT"/test_predictions.npz "$OUT"/val_predictions.npz "$OUT"/best_model.pth "$DEST"/ 2>/dev/null
echo "job ${TAG} exited rc=$RC, synced to $DEST"
