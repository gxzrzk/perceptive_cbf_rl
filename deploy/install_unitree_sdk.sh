#!/usr/bin/env bash
# ==============================================================================
# Unitree SDK2 Wrapper (unitree_interface) 实机 Python 绑定编译与安装脚本
#
# 在部署架构中的角色：
# -------------------
# 本脚本用于在实机边缘计算设备 Jetson Orin (Ubuntu 20.04 / aarch64 / Python 3.8)
# 上编译 C++ 编写的 Unitree SDK2 硬件驱动接口，并将其生成的共享动态库
# `unitree_interface.so` 正确安装至 deploy 虚拟环境的 site-packages 中。
# 为 hardware_node.py 提供底层的 LowCmd/LowState 和遥控手柄数据交互能力。
#
# 使用方式：
#   cd deploy && bash install_unitree_sdk.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../deploy 目录
WRAPPER_DIR="$SCRIPT_DIR/common/unitree_sdk2_wrapper"
ARCH="$(uname -m)"   # 硬件架构 (Jetson 为 aarch64, 开发机为 x86_64)

# 指定与 Jetson 系统原生 Python (3.8) 严格一致的版本，确保 C++ ABI 兼容
UV_PYTHON="3.8"
UV_PROJECT_ARGS=(--project "$SCRIPT_DIR" --python "$UV_PYTHON")

echo "[install] 硬件架构=$ARCH  Python版本=$UV_PYTHON  Wrapper源码路径=$WRAPPER_DIR"

# 1. 安装系统级 C++ 编译工具链与依赖库
echo "[install] 步骤 1/6: 安装系统级构建依赖库..."
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev pybind11-dev \
  libspdlog-dev libboost-all-dev libyaml-cpp-dev libfmt-dev

# 2. 在 deploy 的 uv 虚拟环境中安装 Python 构建工具与库
echo "[install] 步骤 2/6: 在 uv 虚拟环境中安装构建依赖 (pybind11, numpy)..."
uv pip install "${UV_PROJECT_ARGS[@]}" pybind11 numpy

# 3. 确保 unitree_sdk2_wrapper git 子模块已拉取就绪
echo "[install] 步骤 3/6: 同步并拉取 unitree_sdk2_wrapper 子模块源码..."
git -C "$SCRIPT_DIR/.." submodule update --init deploy/common/unitree_sdk2_wrapper

# 4. 执行 CMake 构建流程，编译生成 Python 绑定的 C++ .so 动态库
echo "[install] 步骤 4/6: 编译 C++ Python 绑定库..."
cd "$WRAPPER_DIR/python_binding"
export UNITREE_SDK2_PATH="$(cd .. && pwd)"
uv run "${UV_PROJECT_ARGS[@]}" bash build.sh --sdk-path "$UNITREE_SDK2_PATH"

# 5. 将生成的 unitree_interface.so 移动并重命名复制至 uv 虚拟环境 site-packages 中
echo "[install] 步骤 5/6: 安装 unitree_interface.so 至虚拟环境 site-packages..."
SITE_PACKAGES=$(uv run "${UV_PROJECT_ARGS[@]}" python -c "import site; print(site.getsitepackages()[0])")
echo "[install] 安装目标路径 -> $SITE_PACKAGES"
cp build/lib/unitree_interface.cpython-*-linux-gnu.so "$SITE_PACKAGES/unitree_interface.so"

# 6. 导入测试：验证 unitree_interface 是否能够被正常 import
echo "[install] 步骤 6/6: 验证动态库加载与导入..."
uv run "${UV_PROJECT_ARGS[@]}" python -c "import unitree_interface; print('unitree_interface 安装验证成功 (OK)')"
