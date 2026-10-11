#!/usr/bin/env bash
# ==============================================================================
# 实验组: gimbal_joint (云台相机 × 关节级 CBF 引导)
# ==============================================================================
# 1. 论文实验矩阵定位:
#    - 感知模态: 动态俯仰云台相机 (Gimbal Camera, CAMERA_GIMBAL=1, CAMERA_PROPRIO=1)
#                基于真值瞄准追踪球体，观测输入为单目 BallOnly 掩码深度图与云台关节状态。
#    - 安全模式: 关节空间 CBF 奖励引导 (Joint-space CBF, CBF_JOINT=1)
#    - 评测对照: 裸模型评测对应 Joint-CBF 单元格；若附加 BENCH_CBF_FILTER=1 评测则对应 '+filter' 单元格。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 gimbal_joint_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export CAMERA_GIMBAL=1
export CAMERA_PROPRIO=1
export CBF_JOINT=1
export EXP_NAME="gimbal_joint${_SUF}"
exec ./train_dodge_single_ballonly.sh "$@"
