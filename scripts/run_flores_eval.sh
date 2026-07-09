#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"

echo "=== FLORES-101 metric evaluation ==="
"$PYTHON" -m pip install -q sacrebleu -i https://pypi.tuna.tsinghua.edu.cn/simple

cd "$ROOT"
"$PYTHON" scripts/eval_flores.py --tokenize flores200

if [[ "${COMET:-0}" == "1" ]]; then
  echo "=== COMET (optional, slower) ==="
  "$PYTHON" -m pip install -q unbabel-comet -i https://pypi.tuna.tsinghua.edu.cn/simple
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PYTHON" scripts/eval_flores.py --tokenize flores200 --comet --gpus 1 \
    --out "$ROOT/outputs/flores101/metrics_comet.json"
fi

echo "=== Done ==="
