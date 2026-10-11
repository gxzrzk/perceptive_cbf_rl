#!/usr/bin/env bash
# ==============================================================================
# 纯本体感觉行走步态策略训练脚本 (Train Proprio-only Walk Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Flat (基于对抗运动先验 AMP 学习基础双足行走步态)
#    - 感知模式: 纯本体感知 (Proprio-only, 无视觉/无球体)，观测为 4 帧历史本体向量 (384维) -> 29 关节目标
#    - 核心作用: 该策略用于基准测试的部署恢复模式 (--walk-recover) 以及实机部署中的模式切换
#               (walk <-> dodge)，负责在每次躲避球后控制机器人走回原站立工位并调正朝向。
#    - 输出日志: logs/rsl_rl/walk，可被 dodge_benchmark.py --walk-recover 自动识别加载。
#
# 2. 环境变量重载选项:
#    - NUM_ENVS: 并行环境数 (默认: 8192)
#    - EXP_NAME: 实验日志名称 (默认: walk)
#
# 3. 使用示例:
#    ./train_walk.sh
#    ./train_walk.sh --agent.max-iterations=100000
#    NUM_ENVS=4096 ./train_walk.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="Unitree-G1-AMP-Flat"
NUM_ENVS="${NUM_ENVS:-8192}"
EXP_NAME="${EXP_NAME:-walk}"

echo "[train_walk.sh] task=${TASK} num_envs=${NUM_ENVS} experiment_name=${EXP_NAME}"
echo "[train_walk.sh] extra args: $*"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec uv run python scripts/train.py "$TASK" \
    --env.scene.num-envs="$NUM_ENVS" \
    --agent.experiment-name="$EXP_NAME" \
    --agent.max-iterations=60000 \
    --video True \
    --video-interval 48000 \
    "$@"
