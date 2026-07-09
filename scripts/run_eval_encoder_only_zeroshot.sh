#!/usr/bin/env bash
# Encoder-only zero-shot: tasks NOT in Stage-1 translation training (e.g. MGSM).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_MODEL="${BASE_MODEL:-$ROOT/model/XBridge-base}"
LANGS="${LANGS:-zh}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/encoder_only_zeroshot}"

echo "=== Encoder-only zero-shot task eval ==="
echo "model: $BASE_MODEL"
echo "langs: $LANGS"

cd "$ROOT"
"$PYTHON" scripts/eval_encoder_only_zeroshot_tasks.py \
  --base_model "$BASE_MODEL" \
  --langs "$LANGS" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 4
