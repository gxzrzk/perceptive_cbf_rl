#!/usr/bin/env bash
# ==============================================================================
# 实验组: gimbal_none (云台相机 × 无屏障基线)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 动态俯仰云台相机 (Gimbal Camera, CAMERA_GIMBAL=1, CAMERA_PROPRIO=1)
#                带有真值引导的主动俯仰云台 (+/-30°)，观测为 BallOnly 深度图与云台状态。
#    - 安全模式: 无 CBF 屏障基线 (NO_LINK_CBF=1, 仅依赖基于质心距离的球体躲避惩罚奖励)
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 gimbal_none_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export CAMERA_GIMBAL=1
export CAMERA_PROPRIO=1
export NO_LINK_CBF=1
export EXP_NAME="gimbal_none${_SUF}"
exec ./train_dodge_single_ballonly.sh "$@"
