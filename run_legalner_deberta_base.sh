#!/usr/bin/env bash
# Fine-tune GLiNER2 (deberta-v3-base encoder) on LeNER-Br and score best/ on the test split.
# Hyperparameters mirror legalner/configs/base.yaml + legalner/configs/std_ner/deberta_base.yaml,
# with batch 4 x grad-accum 4 (same effective batch 16) so it fits a 12 GB GPU without OOMs.
#
# Usage: bash run_legalner_deberta_base.sh [output_dir]
#   GPU=0 bash run_legalner_deberta_base.sh      # pick another GPU (default: 1)
#   SEED=42 bash run_legalner_deberta_base.sh    # another seed (default: 914)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-1}"
SEED="${SEED:-914}"
OUT="${1:-/home/ubuntu/dev/legalner-gliner2/output/deberta_base_ulysses_s${SEED}}"

if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "Output dir $OUT already exists and is not empty; pass a different one." >&2
    exit 1
fi
mkdir -p "$OUT"
cd "$REPO_DIR"

echo "Training on GPU $GPU, seed $SEED -> $OUT (log: $OUT/train.log)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU" \
python train_legalner_br.py \
  --model fastino/gliner2.5-base-v1 \
  --dataset ulysses \
  --output-dir "$OUT" --run-name "gliner2_deberta_base_ulysses_s${SEED}" \
  --epochs 30 --batch-size 8 --grad-accum 2 --eval-batch-size 8 \
  --encoder-lr 2e-5 --task-lr 2e-5 --weight-decay 0 \
  --scheduler cosine --warmup-steps 500 \
  --early-stopping-patience 5 --early-stopping-threshold 0.0005 \
  --max-len 512 --allow-invalid-samples \
  --seed "$SEED" --eval-test > "$OUT/train.log" 2>&1
