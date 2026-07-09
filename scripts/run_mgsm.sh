#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MT_PATH="${ROOT}/model/nllb-200-1.3B"
LLM_PATH="${ROOT}/model/Meta-Llama-3-8B"
BASE_MODEL="${ROOT}/model/XBridge-SFT"
TESTSET_DIR="${ROOT}/data/mgsm"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/mgsm}"
TEST_LANGS="${TEST_LANGS:-en,bn,de,es,fr,ja,ru,sw,th,zh}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

mkdir -p "$OUTPUT_DIR"
find "$OUTPUT_DIR" -mindepth 1 ! -name 'run.log' -delete

echo "=== MGSM Evaluation (XBridge-SFT) ==="
echo "base_model:  $BASE_MODEL"
echo "test_langs:  $TEST_LANGS"
echo "output_dir:  $OUTPUT_DIR"
echo "gpu:         $CUDA_VISIBLE_DEVICES"

cd "$ROOT"
"$PYTHON" inference_xbridge_stage2_and_3.py \
    --mt_tokenizer_path "$MT_PATH" \
    --llm_tokenizer_path "$LLM_PATH" \
    --base_model "$BASE_MODEL" \
    --batch_size "$BATCH_SIZE" \
    --testset_dir "$TESTSET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --test_langs "$TEST_LANGS" \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo "=== Done. Results in $OUTPUT_DIR/accuracy ==="
cat "$OUTPUT_DIR/accuracy"
