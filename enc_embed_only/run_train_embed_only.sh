#!/usr/bin/env bash
# Embed-only XBridge ablation: NLLB token embeddings -> mapping_embed -> frozen LLaMA3.
# Drops the mapping_enc2llm(Enc(x)) path; trains mapping_embed only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"

MT_PATH="${MT_PATH:-$ROOT/model/nllb-200-1.3B}"
LLM_PATH="${LLM_PATH:-$ROOT/model/Meta-Llama-3-8B}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/encoder_only/opus100_zh_en_100k.jsonl}"
OUTPUT="${OUTPUT:-$ROOT/outputs/enc_mt_embed_only_zh_en_100k}"

mkdir -p "$OUTPUT"

echo "[embed-only] Train mapping_embed on $TRAIN_FILE -> $OUTPUT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" "$PYTHON" "$ROOT/enc_embed_only/train_encoder_only.py" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT" \
  --mt_path "$MT_PATH" \
  --llm_path "$LLM_PATH" \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --reinit_mapping True \
  --bf16 True \
  2>&1 | tee "$OUTPUT/train.log"

echo "Done. Trainable checkpoint: $OUTPUT/checkpoint-final/mapping_embed.pt"
