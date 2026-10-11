#!/usr/bin/env bash
# ==============================================================================
# 实验组: gimbal_link (云台相机 × 连杆级 CBF 屏障)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 动态俯仰云台相机 (Gimbal Camera, CAMERA_GIMBAL=1, CAMERA_PROPRIO=1)
#                带有真值引导的主动俯仰云台 (+/-30°)，观测为 BallOnly 深度图与云台状态。
#    - 安全模式: 连杆级 CBF 屏障奖励 (Per-link clearance CBF reward)
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 gimbal_link_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export CAMERA_GIMBAL=1
export CAMERA_PROPRIO=1
export EXP_NAME="gimbal_link${_SUF}"
exec ./train_dodge_single_ballonly.sh "$@"
