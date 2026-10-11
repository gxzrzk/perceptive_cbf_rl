#!/usr/bin/env bash
# ==============================================================================
# 单目视觉 BallOnly 掩码深度避障策略训练脚本 (Train Single-Camera Ball-Only Dodge)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat
#    - 感知模式: 单目头部深度相机，物理安装俯仰上倾 +20° (CAMERA_TILT_DEG=20，覆盖垂直视场 -7°..+47°)。
#               输入深度图为 BallOnlyDepthObs 格式（仅保留球体真实深度，其余所有背景全填充为极远值 far），
#               精确对齐实机端部署的 EfficientTAM 视觉分割模型输出表示。
#    - 域随机化 (Sim2Real DR): 训练期施加整球瞬态丢帧 (whole-ball dropout)、边缘毛刺 (pixel dropout)、
#               球体深度连贯抖动 (depth jitter) 以及网格滤波闪烁等扰动，大幅提升实机鲁棒性。
#    - 部署要点: 导出的模型在硬件端要求 ZED 相机物理安装倾角严格对齐 (+20°)，且需要部署 EfficientTAM 分割节点。
#    - 论文实验组推荐: 建议优先使用 train_runs/vision_*.sh 或 train_runs/gimbal_*.sh 封装脚本启动对齐实验。
#
# 2. 环境变量重载选项:
#    - TASK: 任务环境名称 (默认: Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat)
#    - NUM_ENVS: 并行环境数 (默认: 8192)
#    - STAND_RATIO: 纯原地站立不抛球环境比例 (默认: 0.2，锚定策略站立能力)
#    - INPLACE_RATIO: 原地停滞受击环境比例 (默认: 0.4，强化零速下的瞬间躲避反应)
#    - DEPTH_DR_SCALE: 深度图域随机化扰动强度倍率 (默认: 1.0)
#    - CAMERA_TILT_DEG: 头部相机仰角，单位为度 (默认: 20)
#    - FRAME_OFFSETS: 历史时序深度帧偏移量元组 (默认: (0,3,8,18))
#    - EXP_NAME: 实验日志名称 (默认: vision_link)
#
# 3. 使用示例:
#    ./train_dodge_single_ballonly.sh
#    ./train_dodge_single_ballonly.sh --agent.max-iterations=40000
#    NUM_ENVS=4096 ./train_dodge_single_ballonly.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="${TASK:-Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat}"
NUM_ENVS="${NUM_ENVS:-8192}"
STAND_RATIO="${STAND_RATIO:-0.2}"
INPLACE_RATIO="${INPLACE_RATIO:-0.4}"
export DEPTH_DR_SCALE="${DEPTH_DR_SCALE:-1.0}"
export CAMERA_TILT_DEG="${CAMERA_TILT_DEG:-20}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"
EXP_NAME="${EXP_NAME:-vision_link}"

echo "[train_dodge_single_ballonly.sh] task=${TASK} num_envs=${NUM_ENVS} CAMERA_TILT_DEG=${CAMERA_TILT_DEG}"
echo "[train_dodge_single_ballonly.sh] rel_standing_envs=${STAND_RATIO} rel_inplace_throw_envs=${INPLACE_RATIO} depth_dr_scale=${DEPTH_DR_SCALE} depth_frame_offsets=${FRAME_OFFSETS}"
echo "[train_dodge_single_ballonly.sh] experiment_name=${EXP_NAME} (REMINDER: physical mount + play must use the same tilt)"
echo "[train_dodge_single_ballonly.sh] extra args: $*"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec uv run python scripts/train.py "$TASK" \
    --env.scene.num-envs="$NUM_ENVS" \
    --env.commands.twist.rel-inplace-throw-envs="$INPLACE_RATIO" \
    --env.depth-frame-offsets "$FRAME_OFFSETS" \
    --env.commands.twist.rel-standing-envs="$STAND_RATIO" \
    --agent.experiment-name="$EXP_NAME" \
    --agent.max-iterations=25000 \
    --video True \
    --video-interval 48000 \
    "$@"
