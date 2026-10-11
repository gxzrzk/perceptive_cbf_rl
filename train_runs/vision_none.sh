#!/usr/bin/env bash
# ==============================================================================
# 实验组: vision_none (固定视觉相机 × 无屏障基线)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 固定俯仰仰角 (+20°) 单目头部深度相机，输入为 BallOnly 掩码深度图。
#    - 安全模式: 无 CBF 屏障基线 (NO_LINK_CBF=1, 剔除连杆级 CBF 屏障项，仅保留质心距离排斥奖励)。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 vision_none_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export NO_LINK_CBF=1
export EXP_NAME="vision_none${_SUF}"
exec ./train_dodge_single_ballonly.sh "$@"
