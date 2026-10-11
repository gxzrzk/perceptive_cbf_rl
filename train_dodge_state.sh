#!/usr/bin/env bash
# ==============================================================================
# 状态真值 Oracle 避障策略训练脚本 (Train State-based Oracle Dodge Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-MimicKit-Flat (MimicKit/SMP 规范基线)
#    - 感知模式: 状态真值 Oracle（直接观测球体在机体航向坐标系下的 6 维相对位置与速度向量）
#               无相机渲染、无深度图，作为全论文体系中避障控制能力的理论“感知性能上限”。
#    - 训练优势: 无需执行光栅化/光线追踪深度渲染，是训练速度最快、吞吐量最高的避障任务。
#    - 核心环境超参数:
#      * STAND_RATIO (0.2): 恒定平稳站立且不抛球的环境比例，防止网络步态退化。
#      * INPLACE_RATIO (0.4): 原地停滞受击环境比例，专门强化机器人零初始速度下的瞬间闪避爆发力。
#    - 论文实验组推荐: 建议优先使用 train_runs/state_*.sh 封装脚本启动对齐实验。
#
# 2. 环境变量重载选项:
#    - TASK: 任务环境名称 (默认: Unitree-G1-AMP-Dodge-MimicKit-Flat)
#    - NUM_ENVS: 并行环境数 (默认: 8192)
#    - STAND_RATIO: 纯原地站立不抛球环境比例 (默认: 0.2)
#    - INPLACE_RATIO: 原地停滞受击环境比例 (默认: 0.4)
#    - EXP_NAME: 实验日志名称 (默认: state_link)
#
# 3. 使用示例:
#    ./train_dodge_state.sh
#    ./train_dodge_state.sh --agent.max-iterations=40000
#    NUM_ENVS=4096 ./train_dodge_state.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# CUDA compilation can exhaust the small system /tmp partition. Keep temporary
# files and Warp kernels on the project disk; honor explicit user overrides.
export TMPDIR="${TMPDIR:-$REPO_ROOT/.cache/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$REPO_ROOT/.cache/warp}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

TASK="${TASK:-Unitree-G1-AMP-Dodge-MimicKit-Flat}"
NUM_ENVS="${NUM_ENVS:-8192}"
STAND_RATIO="${STAND_RATIO:-0.2}"
INPLACE_RATIO="${INPLACE_RATIO:-0.4}"
EXP_NAME="${EXP_NAME:-state_link}"  # honor EXP_NAME so runs get distinct log dirs

echo "[train_dodge_state.sh] task=${TASK} num_envs=${NUM_ENVS} experiment_name=${EXP_NAME}"
echo "[train_dodge_state.sh] rel_standing_envs=${STAND_RATIO} rel_inplace_throw_envs=${INPLACE_RATIO}"
echo "[train_dodge_state.sh] extra args: $*"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec uv run python scripts/train.py "$TASK" \
    --env.scene.num-envs="$NUM_ENVS" \
    --env.commands.twist.rel-inplace-throw-envs="$INPLACE_RATIO" \
    --env.commands.twist.rel-standing-envs="$STAND_RATIO" \
    --agent.experiment-name="$EXP_NAME" \
    --agent.max-iterations=25000 \
    --video True \
    --video-interval 48000 \
    "$@"
