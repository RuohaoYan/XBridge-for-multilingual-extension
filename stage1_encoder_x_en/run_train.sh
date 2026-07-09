#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAGE_DIR="stage1_encoder_x_en"
CONFIG_FILE="${CONFIG_FILE:-${STAGE_DIR}/config.env}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  echo "Create it with: cp ${STAGE_DIR}/sample_config.env ${CONFIG_FILE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_FILE}"

mkdir -p "$(dirname "${TRAIN_FILE}")" "${OUTPUT_DIR}"

DATA_MODE="${DATA_MODE:-parallel}"
if [[ "${DATA_MODE}" == "parallel" ]]; then
  python "${STAGE_DIR}/build_x_en_data.py" \
    --source_file "${SOURCE_FILE}" \
    --english_file "${ENGLISH_FILE}" \
    --output_file "${TRAIN_FILE}" \
    --src_lang "${SRC_LANG}" \
    --prompt "${PROMPT}" \
    --min_chars "${MIN_CHARS:-1}" \
    --max_chars "${MAX_CHARS:-0}" \
    --limit "${LIMIT:-0}"
elif [[ "${DATA_MODE}" == "json" ]]; then
  python "${STAGE_DIR}/build_x_en_data.py" \
    --input_file "${INPUT_FILE}" \
    --output_file "${TRAIN_FILE}" \
    --src_lang "${SRC_LANG}" \
    --prompt "${PROMPT}" \
    --min_chars "${MIN_CHARS:-1}" \
    --max_chars "${MAX_CHARS:-0}" \
    --limit "${LIMIT:-0}"
else
  echo "Unsupported DATA_MODE=${DATA_MODE}. Use parallel or json."
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE:-1,2,3}" \
torchrun --nproc_per_node="${NPROC_PER_NODE:-3}" "${STAGE_DIR}/train_encoder_x_en_mp.py" \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --mt_path "${MT_PATH}" \
  --llm_path "${LLM_PATH}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-5}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --num_epochs "${NUM_EPOCHS:-3}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --max_src_len "${MAX_SRC_LEN:-256}" \
  --max_tgt_len "${MAX_TGT_LEN:-256}" \
  --max_prompt_len "${MAX_PROMPT_LEN:-128}" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --max_steps "${MAX_STEPS:-0}" \
  --seed "${SEED:-42}" \
  $( [[ "${BF16:-true}" == "true" ]] && echo --bf16 || echo --no-bf16 ) \
  $( [[ "${REINIT_MAPPING:-true}" == "true" ]] && echo --reinit_mapping || echo --no-reinit_mapping ) \
  $( [[ -n "${RESUME_MAPPING:-}" ]] && echo --resume_mapping "${RESUME_MAPPING}" )

echo "Encoder x->English training finished. Checkpoint: ${OUTPUT_DIR}/checkpoint-final"
