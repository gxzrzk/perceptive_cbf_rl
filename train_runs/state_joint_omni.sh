#!/usr/bin/env bash
# ==============================================================================
# 实验组: state_joint_omni (全向抛球校准: 状态真值 × 关节级 CBF 引导)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 状态真值 Oracle 模态（直接观测 360° 全向飞来球体的 6 维相对状态，无相机渲染）。
#    - 场景特性: OMNI_THROW=1，360° 全向高速抛球（距离 8-10m，球速 12-15 m/s），作为极限避障能力上限。
#    - 安全模式: 关节空间 CBF 奖励引导 (Joint-space CBF, CBF_JOINT=1)
#    - 评测要求: 必须使用 scripts/dodge_benchmark.py --omni 进行评测（加 BENCH_CBF_FILTER=1 评测过滤单元格）。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 state_joint_omni_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export OMNI_THROW=1
export CBF_JOINT=1
export EXP_NAME="state_joint_omni${_SUF}"
exec ./train_dodge_state.sh "$@"
