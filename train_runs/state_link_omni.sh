#!/usr/bin/env bash
# ==============================================================================
# 实验组: state_link_omni (全向抛球校准: 状态真值 × 连杆级 CBF 屏障)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 状态真值 Oracle 模态（直接观测 360° 全向飞来球体的 6 维相对状态，无相机渲染）。
#    - 场景特性: OMNI_THROW=1，360° 全向高速抛球（距离 8-10m，球速 12-15 m/s）。
#    - 安全模式: 连杆级 CBF 屏障奖励 (Per-link clearance CBF reward)
#    - 评测要求: 必须使用 scripts/dodge_benchmark.py --omni 进行评测。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 state_link_omni_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export OMNI_THROW=1
export EXP_NAME="state_link_omni${_SUF}"
exec ./train_dodge_state.sh "$@"
