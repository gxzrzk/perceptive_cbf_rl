#!/usr/bin/env bash
# ==============================================================================
# 实验组: state_link (状态真值 × 连杆级 CBF 屏障)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 状态真值 Oracle 模态（直接观测球体真值 6 维相对状态，无相机渲染）。
#    - 安全模式: 连杆级 CBF 屏障奖励 (Per-link clearance CBF reward)。
#               本实验组所确立的 CBF 连杆屏障奖励堆栈是全论文核心方法的基础，后续直接迁移至实机视觉策略中。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 state_link_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export EXP_NAME="state_link${_SUF}"
exec ./train_dodge_state.sh "$@"
