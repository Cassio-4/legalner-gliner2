#!/usr/bin/env bash
# EXP-04: LoRA fine-tune of GLiNER2 (gliner2.5-multi-v1, mDeBERTa-v3-base encoder) on LeNER-Br,
# then score best/ on the test split.
# This one is for healing 5 pruned models by the ReplaceMe method with lora
# LoRA r=8, alpha=16, dropout 0.05 on encoder + extractive_head, and lr 2e-4.
# Batch 8 x grad-accum 2 (effective batch 16), eval batch 8.
# LoRA modules (claude): this model uses the boundary architecture, so extractive_head resolves to
#   boundary_head (43 Linear layers); encoder resolves to 36 Linear layers (q/k/v + dense) on the
#   6-layer pruned encoder (72 on the full 12-layer model).
# Learning rate (claude): only task-lr applies to LoRA modules; encoder-lr has no effect in a LoRA run.
# Usage: bash run_exp04.sh [output_dir]
#   GPU=1 bash run_exp04.sh      # pick another GPU (default: 0)
#   SEED=0 bash run_exp04.sh     # seed for training AND which pruned model to heal (default: 42)
#                                # pruned models exist for seeds 0, 42, 135, 295, 512
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
MODEL="/home/ubuntu/dev/gliner2-replaceme/output_models/gliner2.5-multi-v1_3_layers_1_4__6_9_lener_dev_ReplaceMe_lstsq_2_s${SEED}"
PYTHON="$REPO_DIR/.venv/bin/python"
OUT="${1:-/home/ubuntu/dev/legalner-gliner2/output/EXP-04/gliner2_mDeBERTa-v3-base_pruned_lener_s${SEED}}"

if [ ! -f "$MODEL/model.safetensors" ]; then
    echo "No pruned model for seed $SEED at $MODEL" >&2
    exit 1
fi

if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "Output dir $OUT already exists and is not empty; pass a different one." >&2
    exit 1
fi
mkdir -p "$OUT"
cd "$REPO_DIR"

echo "Training on GPU $GPU, seed $SEED -> $OUT (log: $OUT/train.log)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU" \
"$PYTHON" train_legalner_br.py \
  --model "$MODEL" \
  --dataset lener \
  --output-dir "$OUT" --run-name "gliner2_mDeBERTa-v3-base_pruned_lener_s${SEED}" \
  --epochs 30 --batch-size 8 --grad-accum 2 --eval-batch-size 8 \
  --task-lr 2e-4 --weight-decay 0 \
  --scheduler cosine --warmup-steps 500 \
  --early-stopping-patience 5 --early-stopping-threshold 0.0005 \
  --max-len 512 --allow-invalid-samples \
  --lora-target-modules encoder extractive_head \
  --lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 \
  --seed "$SEED" --eval-test > "$OUT/train.log" 2>&1
