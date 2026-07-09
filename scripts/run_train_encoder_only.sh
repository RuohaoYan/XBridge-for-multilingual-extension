#!/usr/bin/env bash
# Encoder-only XBridge training: NLLB encoder + mapping_enc2llm + frozen LLaMA3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"

MT_PATH="${MT_PATH:-$ROOT/model/nllb-200-1.3B}"
LLM_PATH="${LLM_PATH:-$ROOT/model/Meta-Llama-3-8B}"
DATA_OUT="$ROOT/data/encoder_only/train.jsonl"
TRAIN_OUT="$ROOT/outputs/encoder_only_train"
MERGED_OUT="$ROOT/outputs/encoder_only_merged"

echo "[1/4] Prepare large x→en parallel JSONL (NLLB mined, English side only)..."
"$PYTHON" "$ROOT/scripts/download_parallel_xen.py" \
  --output "$ROOT/data/encoder_only/train_nllb_xen.jsonl" \
  --sources nllb,seed,flores \
  --max_samples_per_lang 200000 \
  --cache_dir "$ROOT/data/parallel_cache"

echo "[2/4] (optional small set) FLORES-only JSONL still available via prepare_encoder_only_data.py"

TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/encoder_only/train_nllb_xen.jsonl}"

echo "[3/4] Train mapping_enc2llm (llm_only, L_LLM only)..."
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" "$ROOT/train_encoder_only.py" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$TRAIN_OUT" \
  --mt_path "$MT_PATH" \
  --llm_path "$LLM_PATH" \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --reinit_mapping True \
  --bf16 True

echo "[4/4] Merge mapping into full checkpoint for inference..."
"$PYTHON" "$ROOT/scripts/merge_encoder_only_ckpt.py" \
  --base_checkpoint "$ROOT/model/XBridge-base" \
  --mapping_pt "$TRAIN_OUT/checkpoint-final/mapping_enc2llm.pt" \
  --output_dir "$MERGED_OUT"

echo "Done. Inference with:"
echo "  --base_model $MERGED_OUT"
