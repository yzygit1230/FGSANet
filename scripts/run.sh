#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DR_DATA_ROOT:?Set DR_DATA_ROOT to the directory containing BiDR and 'Messidor-2 Dataset'}"

MODEL="${MODEL:-fgsanet}"
DATASETS="${DATASETS:-BiDR Messidor2}"
SEEDS="${SEEDS:-0 42 123}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
WORKERS="${WORKERS:-4}"
LR="${LR:-3e-4}"

for dataset in $DATASETS; do
  for seed in $SEEDS; do
    echo "Training ${MODEL} on ${dataset}, seed=${seed}"
    python dr_pipeline/train.py \
      --data-root "$DR_DATA_ROOT" \
      --dataset "$dataset" \
      --model "$MODEL" \
      --image-size "$IMAGE_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --workers "$WORKERS" \
      --lr "$LR" \
      --seed "$seed"
  done
done
