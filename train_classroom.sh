#!/usr/bin/env bash
# ==============================================================================
# 教室走廊环境全向避障与巡航策略训练脚本 (Train Classroom Dodge Task)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat
#    - 感知模式: 状态真值 Oracle 模态（观测球体真值 6 维状态）+ WallWalk 步态运动先验
#    - 场景特性: 包含课桌课椅构建的真实教室过道走廊环境，随机初始站位与目标路径巡航。
#               结合 360° 全向定时高速抛球，要求机器人在沿过道行进的同时成功闪避来球且不与桌椅碰撞。
#    - 输出日志: logs/rsl_rl/state_classroom
#
# 2. 环境变量重载选项:
#    - TASK: 任务环境名称 (默认: Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat)
#    - NUM_ENVS: 并行环境数 (默认: 1024)
#    - STAND_RATIO: 纯原地站立环境比例 (默认: 0.0，保持机器人在行走状态)
#    - INPLACE_RATIO: 原地停滞受击环境比例 (默认: 0.0)
#    - EXP_NAME: 实验日志名称 (默认: state_classroom)
#    - TMPDIR / WARP_CACHE_PATH: 缓存与编译路径，防止系统盘 /tmp 分区占满
#
# 3. 使用示例:
#    ./train_classroom.sh
#    NUM_ENVS=512 ./train_classroom.sh --agent.max-iterations=40000
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# CUDA compilation can exhaust the small system /tmp partition. Keep temporary
# files and Warp kernels on the project disk; honor explicit user overrides.
export TMPDIR="${TMPDIR:-$REPO_ROOT/.cache/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$REPO_ROOT/.cache/warp}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

TASK="${TASK:-Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat}"
NUM_ENVS="${NUM_ENVS:-1024}"
STAND_RATIO="${STAND_RATIO:-0.0}"
INPLACE_RATIO="${INPLACE_RATIO:-0.0}"
EXP_NAME="${EXP_NAME:-state_classroom}"  # honor EXP_NAME so runs get distinct log dirs

echo "[train_classroom.sh] task=${TASK} num_envs=${NUM_ENVS} experiment_name=${EXP_NAME}"
echo "[train_classroom.sh] rel_standing_envs=${STAND_RATIO} rel_inplace_throw_envs=${INPLACE_RATIO}"
echo "[train_classroom.sh] extra args: $*"

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
