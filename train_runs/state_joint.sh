#!/usr/bin/env bash
# ==============================================================================
# 实验组: state_joint (状态真值 × 关节级 CBF 引导)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 状态真值 Oracle 模态（直接观测球体真值 6 维相对状态，无相机渲染）。
#    - 安全模式: 关节空间 CBF 奖励引导 (Joint-space CBF, CBF_JOINT=1)
#    - 评测对照: 裸模型评测对应 Joint-CBF 单元格；附加 BENCH_CBF_FILTER=1 评测对应 '+filter' 单元格。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 state_joint_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export CBF_JOINT=1
export EXP_NAME="state_joint${_SUF}"
exec ./train_dodge_state.sh "$@"
