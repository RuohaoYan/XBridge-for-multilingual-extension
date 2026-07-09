#!/usr/bin/env bash
# XBridge 环境安装脚本（优先清华源）
# RTX 5090 (sm_120) 需 PyTorch cu128；清华暂未镜像 cu128 wheel，PyTorch 使用阿里云镜像
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/yrh/.conda/envs/xbridge/bin/python}"
PIP="${PIP:-/home/yrh/.conda/envs/xbridge/bin/pip}"

TSINGHUA_PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"
ALIYUN_TORCH="https://mirrors.aliyun.com/pytorch-wheels/cu128"

echo "========================================"
echo " XBridge 环境安装"
echo " 普通包: 清华源 ${TSINGHUA_PYPI}"
echo " PyTorch: 阿里云 cu128 ${ALIYUN_TORCH}"
echo "========================================"

mkdir -p "${HOME}/.pip"
if ! grep -q "pypi.tuna.tsinghua.edu.cn" "${HOME}/.pip/pip.conf" 2>/dev/null; then
  cat > "${HOME}/.pip/pip.conf" <<EOF
[global]
index-url = ${TSINGHUA_PYPI}
trusted-host = pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com
EOF
fi

echo ""
echo "[1/3] 安装 PyTorch (cu128, 支持 RTX 5090)..."
"${PIP}" uninstall -y torch torchvision torchaudio 2>/dev/null || true
"${PIP}" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  -f "${ALIYUN_TORCH}/" \
  -i "${TSINGHUA_PYPI}" \
  --progress-bar on

echo ""
echo "[2/3] 安装项目依赖 (清华源, 跳过 torch)..."
grep -v '^torch' "${ROOT}/requirements.txt" > /tmp/xbridge_requirements_no_torch.txt
"${PIP}" install -r /tmp/xbridge_requirements_no_torch.txt -i "${TSINGHUA_PYPI}" --progress-bar on
"${PIP}" install "pyarrow>=14.0.0" safetensors -i "${TSINGHUA_PYPI}" --progress-bar on

echo ""
echo "[2.5/3] 再次确保 PyTorch cu128 (防止被覆盖)..."
"${PIP}" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  -f "${ALIYUN_TORCH}/" \
  -i "${TSINGHUA_PYPI}" \
  --progress-bar on

echo ""
echo "[3/3] 验证环境..."
"${PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability())
    print("arch list:", torch.cuda.get_arch_list())
    assert "sm_120" in torch.cuda.get_arch_list(), f"需要 sm_120 支持, 当前: {torch.cuda.get_arch_list()}"
import transformers, fire, sentencepiece
print("transformers:", transformers.__version__)
print("OK")
PY

echo ""
echo "环境安装完成。"
