#!/usr/bin/env bash
# XBridge paper reproduction pipeline
# Stage 1: FLORES-101 translation (XBridge-base)
# Stage 2/3: MGSM math reasoning (XBridge-SFT)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
SCRIPTS="${ROOT}/scripts"

echo "========================================"
echo " XBridge Paper Reproduction"
echo "========================================"

echo ""
echo "[1/3] Downloading evaluation datasets..."
"$PYTHON" "${SCRIPTS}/download_datasets.py" --all

echo ""
echo "[2/3] Running FLORES-101 evaluation (Stage 1)..."
bash "${SCRIPTS}/run_flores.sh"

echo ""
echo "[3/3] Running MGSM evaluation (Stage 2 & 3)..."
bash "${SCRIPTS}/run_mgsm.sh"

echo ""
echo "========================================"
echo " Reproduction complete!"
echo " FLORES outputs: ${ROOT}/outputs/flores101/"
echo " MGSM accuracy:  ${ROOT}/outputs/mgsm/accuracy"
echo "========================================"
