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

CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES_VALUE%%,*}}" \
python "${STAGE_DIR}/infer_encoder_x_en.py" \
  --base_model "${BASE_MODEL:-${OUTPUT_DIR}/checkpoint-final}" \
  --mt_path "${MT_PATH}" \
  --llm_path "${LLM_PATH}" \
  --input_file "${INFER_INPUT_FILE}" \
  --output_file "${INFER_OUTPUT_FILE}" \
  --src_lang "${SRC_LANG}" \
  --prompt "${PROMPT}" \
  --batch_size "${INFER_BATCH_SIZE:-16}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-128}"

echo "English outputs written to: ${INFER_OUTPUT_FILE}"
