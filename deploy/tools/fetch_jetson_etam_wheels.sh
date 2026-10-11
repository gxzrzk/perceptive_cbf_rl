#!/usr/bin/env bash
# ==============================================================================
# Jetson aarch64 边缘端 EfficientTAM 依赖 Wheel 包下载与安装脚本
#
# 在部署架构中的角色：
# -------------------
# 本脚本用于在实机 Jetson Orin (JetPack 5.x / aarch64 / Python 3.8) 上一键准备
# EfficientTAM 目标分割网络所需的 PyTorch 运行时环境：
# 1. 下载：从 NVIDIA 开发者官方 CDN 获取针对 JP5 定制的 PyTorch aarch64 wheel 包；
# 2. 校验：使用 deploy/wheels/SHA256SUMS 强校验文件哈希防篡改；
# 3. 安装：若当前设备为 aarch64 架构，自动安装至 deploy 虚拟环境中；
#    若在 x86 开发机上运行，仅下载并缓存至 deploy/wheels/ 目录供后续分发。
#
# 使用方式：
#   bash deploy/tools/fetch_jetson_etam_wheels.sh            # 下载并安装 torch 与 torchvision
#   NO_INSTALL=1 bash deploy/tools/fetch_jetson_etam_wheels.sh   # 仅下载镜像，不执行安装
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # deploy/tools
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                     # deploy
WHEELS="$DEPLOY_DIR/wheels"
TORCH_WHL="torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/${TORCH_WHL}"

mkdir -p "$WHEELS"

# 1. 从 NVIDIA 官方 CDN 下载 PyTorch aarch64 wheel 包并进行哈希校验
if [[ ! -f "$WHEELS/$TORCH_WHL" ]]; then
  echo "[fetch] 正在从 NVIDIA 官方 CDN 下载 PyTorch 轮子包 (~170 MB)..."
  curl -fSL --connect-timeout 20 --retry 3 -o "$WHEELS/$TORCH_WHL" "$TORCH_URL"
else
  echo "[fetch] PyTorch wheel 已经存在于本地缓存: $WHEELS/$TORCH_WHL"
fi
if [[ -f "$WHEELS/SHA256SUMS" ]]; then
  echo "[fetch] 正在执行 SHA256 校验和完整性校验..."
  ( cd "$WHEELS" && sha256sum -c --ignore-missing SHA256SUMS )
fi

# 2. 检查 torchvision wheel 轮子包是否存在
#    注意：NVIDIA 官方不提供预编译的 torchvision，且 jetson-ai-lab 源常不稳定；
#    若尚未编译，请参照 deploy/wheels/README.md 在 Jetson 本地编译一次
TV_WHL="$(ls "$WHEELS"/torchvision-0.16.*-cp38-*-linux_aarch64.whl 2>/dev/null | head -1 || true)"
if [[ -z "$TV_WHL" ]]; then
  echo "[fetch] 警告: 在 $WHEELS 未找到 torchvision wheel 轮子包。" >&2
  echo "        建议首次在 Jetson 上安装 torch 后执行一次源码编译并存入该目录。" >&2
  echo "        详见 deploy/wheels/README.md / SETUP.md 第 9 步:" >&2
  echo "          git clone --branch v0.16.2 --depth 1 https://github.com/pytorch/vision /tmp/torchvision" >&2
  echo "          cd /tmp/torchvision && export BUILD_VERSION=0.16.2 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=4" >&2
  echo "          $DEPLOY_DIR/.venv/bin/python setup.py bdist_wheel && cp dist/torchvision-0.16.2*.whl $WHEELS/" >&2
fi

# 3. 在实机 aarch64 环境下安装至 deploy 虚拟环境中 (x86 开发机或 NO_INSTALL=1 时自动跳过)
if [[ "${NO_INSTALL:-0}" == "1" ]]; then
  echo "[fetch] 检测到 NO_INSTALL=1: 仅下载缓存，跳过安装步骤。"; exit 0
fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "[fetch] 当前架构为 $(uname -m) (非 aarch64): 轮子包已下载，但跳过安装 (仅适用于 Jetson 实机)。"
  exit 0
fi

echo "[fetch] 正在安装 PyTorch (+ torchvision 若存在) 以及纯 Python 依赖至 deploy 虚拟环境..."
uv pip install --project "$DEPLOY_DIR" "$WHEELS/$TORCH_WHL" ${TV_WHL:+"$TV_WHL"} \
  "hydra-core>=1.3.2" "iopath>=0.1.10" "pillow>=9.4.0"

# 4. 验证 Python 导入
echo "[fetch] 正在验证 PyTorch 导入与 CUDA 状态..."
"$DEPLOY_DIR/.venv/bin/python" - <<'PY'
import torch
print("  PyTorch版本:", torch.__version__, "CUDA可用状态:", torch.cuda.is_available())
try:
    import torchvision; print("  torchvision版本:", torchvision.__version__)
except Exception as e:
    print("  torchvision 暂不可用:", type(e).__name__, e)
PY
