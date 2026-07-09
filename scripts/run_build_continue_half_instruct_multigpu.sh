#!/usr/bin/env bash
# Multi-GPU detached generation (survives terminal close via nohup).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

INPUT="${ROOT}/data/encoder_only/continue_en_half_50k.jsonl"
OUTPUT="${ROOT}/data/encoder_only/continue_en_half_50k_instruct.jsonl"
LOG_DIR="${ROOT}/outputs/continue_instruct_shards"
PID_FILE="${LOG_DIR}/pids.txt"
SCRIPT="${ROOT}/scripts/build_continue_half_instruct_greedy.py"

mkdir -p "$LOG_DIR"
: > "$PID_FILE"

SKIP_ARG=()
if [[ -f "$OUTPUT" ]]; then
  SKIP_ARG=(--also_skip_from "$OUTPUT")
fi

echo "Starting ${NUM_GPUS} GPU workers (nohup)..."
for ((i=0; i<NUM_GPUS; i++)); do
  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" "$SCRIPT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --device cuda:0 \
    --shard_id "$i" \
    --num_shards "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --resume \
    "${SKIP_ARG[@]}" \
    > "${LOG_DIR}/shard${i}.log" 2>&1 &
  echo $! >> "$PID_FILE"
  echo "  shard${i} pid=$! gpu=${i} log=${LOG_DIR}/shard${i}.log"
done

echo ""
echo "PIDs saved to: $PID_FILE"
echo "Monitor: tail -f ${LOG_DIR}/shard0.log"
echo "After all shards finish, merge:"
echo "  $PYTHON ${ROOT}/scripts/merge_continue_instruct_shards.py"
