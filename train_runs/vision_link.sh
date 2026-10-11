#!/usr/bin/env bash
# ==============================================================================
# 实验组: vision_link (固定视觉相机 × 连杆级 CBF 屏障 —— 实机物理部署配置)
# ==============================================================================
# 1. 论文实验矩阵定位与特殊地位:
#    - 核心地位: 本脚本为论文实机硬件部署的主力策略配方！
#               导出的部署权重 deploy/ckpts/dodge_link_cbf.onnx 完全由此训练生成。
#    - 感知模态: 固定俯仰仰角 (+20°) 单目头部深度相机，输入为 BallOnly 掩码深度图
#                (对齐实机端部署的 EfficientTAM 视频分割输出)。
#    - 安全模式: 连杆级 CBF 屏障奖励 (Per-link clearance CBF barrier reward)。
# 2. 环境变量重载:
#    - NUM_ENVS: 并行环境数 (默认: 8192，非默认值自动追加后缀如 vision_link_4096)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export NUM_ENVS="${NUM_ENVS:-8192}"
export STAND_RATIO=0.2
export INPLACE_RATIO=0.4
_SUF=""; [ "$NUM_ENVS" != "8192" ] && _SUF="_${NUM_ENVS}"
export EXP_NAME="vision_link${_SUF}"
exec ./train_dodge_single_ballonly.sh "$@"
