#!/usr/bin/env bash
# 4-GPU (gloo grad-sync) baseline training: mapping_enc2llm on 200k zh->en.
# Effective batch 32 (=4 gpus x bs4 x accum2), lr 2e-5, 3 epochs, reinit from scratch.
# NCCL is unusable on this box (RTX 5090 illegal-memory-access on collectives),
# so grads sync via gloo (CPU); all heavy compute stays on-GPU per rank.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
TORCHRUN="${TORCHRUN:-/home/yrh/.conda/envs/xbridge/bin/torchrun}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
PORT="${PORT:-29550}"

TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/encoder_only/opus100_zh_en_200k.jsonl}"
OUT="${OUT:-$ROOT/outputs/enc_mt_xen_200k}"

mkdir -p "$OUT"
cd "$ROOT"
exec "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$PORT" \
  "$ROOT/train_encoder_only_mp.py" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUT" \
  --per_device_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --reinit_mapping True \
  --save_steps 500 \
  --logging_steps 10 \
  --bf16 True
