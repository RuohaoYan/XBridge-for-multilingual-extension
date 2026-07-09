#!/usr/bin/env bash
# Encoder-only task eval: FLORES x->en (NLLB encoder + mapping_enc2llm + LLM).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_MODEL="${BASE_MODEL:-$ROOT/model/XBridge-base}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/encoder_only_eval}"
LANGS="${LANGS:-}"   # empty = all 9 non-en langs

echo "=== Encoder-only evaluation (FLORES x->en) ==="
echo "base_model: $BASE_MODEL"
echo "gpu:        $CUDA_VISIBLE_DEVICES"

ARGS=(--base_model "$BASE_MODEL" --output_dir "$OUTPUT_DIR")
if [[ -n "$LANGS" ]]; then
  ARGS+=(--langs $LANGS)
fi

cd "$ROOT"
"$PYTHON" -m pip install -q sacrebleu -i https://pypi.tuna.tsinghua.edu.cn/simple
"$PYTHON" scripts/eval_encoder_only.py "${ARGS[@]}"
