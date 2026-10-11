#!/usr/bin/env bash
# ==============================================================================
# 单目固定深度视觉避障策略可视化回放与域随机化预览脚本 (Play Vision Dodge Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat (论文核心部署任务)
#    - 感知模式: 固定仰角 (+20°) 单目头部深度相机，输入 BallOnlyDepthObs 掩码深度图
#               （仅保留真实深度球体，其余背景填充为 far 远景值）。
#    - 域随机化 (DR) 预览:
#      * 默认开启 BALLONLY_AUG=1，在 Viser 网页端的 'head_depth_single' 深度面板中实时呈现
#        训练阶段引入的真实噪声（球体深度抖动、整球瞬态丢帧、边缘像素闪烁等），确保评估处于分布内。
#      * 设为 BALLONLY_AUG=0 可切换为纯净无噪的干净掩码球进行评估。
#    - 倾角一致性: CAMERA_TILT_DEG 必须严格与训练时保持一致（默认 20°），否则会导致球体落在错误视场中。
#    - 可视化平台: 启动 Viser Web 端可视化服务器 (默认访问 http://localhost:8080)。
#
# 2. 检查点 (Checkpoint) 解析机制:
#    - 第一个非 '-' 参数可显式传入检查点路径 (如 logs/rsl_rl/vision_link/<run>/model_20000.pt)。
#    - 若未传入检查点，则启动纯站立 Dummy Agent (agent=zero)，方便单独检验传感器输入与抛球轨迹。
#
# 3. 环境变量重载选项:
#    - NUM_ENVS: 可视化同屏机器人数量 (默认: 1)
#    - FRAME_OFFSETS: 历史时序深度帧偏移量元组 (默认: (0,3,8,18))
#    - CAMERA_TILT_DEG: 头部相机仰角 (默认: 20)
#    - BALLONLY_AUG: 是否开启深度域随机化预览 (默认: 1, 0 表示干净评测)
#    - AGENT: 未指定模型时的 Dummy 类型 (默认: zero)
#
# 4. 使用示例:
#    ./play_vision.sh                                      # 启动静止 Dummy + 观察深度 DR
#    ./play_vision.sh logs/rsl_rl/vision_link/<run>/model_20000.pt
#    BALLONLY_AUG=0 ./play_vision.sh <ckpt>                # 干净无噪深度图评测
#    NUM_ENVS=4 ./play_vision.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat"

# First positional arg (not a flag) = checkpoint path -> trained policy; else standing dummy.
CKPT=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

# Camera tilt MUST match the checkpoint's training tilt (the release trains at +20 deg).
export CAMERA_TILT_DEG="${CAMERA_TILT_DEG:-20}"      # read at import by dodge_env_cfgs

# Play WITH the masked-depth DR by default (the policy trained on it -- in-distribution). Set
# BALLONLY_AUG=0 to eval on the clean masked ball instead.
export BALLONLY_AUG="${BALLONLY_AUG:-1}"

NUM_ENVS="${NUM_ENVS:-1}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"
AGENT="${AGENT:-zero}"  # standing dummy by default (no checkpoint needed)

echo "[play_vision.sh] task=${TASK} CAMERA_TILT_DEG=${CAMERA_TILT_DEG} BALLONLY_AUG=${BALLONLY_AUG} (1=DR on, 0=clean eval)"
echo "[play_vision.sh] depth_frame_offsets=${FRAME_OFFSETS} num_envs=${NUM_ENVS} viewer=viser"

COMMON=( "$TASK" --num-envs "$NUM_ENVS" --viewer viser --depth-frame-offsets "$FRAME_OFFSETS" )

if [[ -n "$CKPT" ]]; then
  [[ -f "$CKPT" ]] || { echo "[play_vision.sh] checkpoint not found: ${CKPT}" >&2; exit 1; }
  echo "[play_vision.sh] policy checkpoint=${CKPT}"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    exec uv run python scripts/play.py "${COMMON[@]}" --checkpoint-file "$CKPT" "$@"
else
  echo "[play_vision.sh] dummy agent=${AGENT} (standing; pass a checkpoint to drive a policy)"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    exec uv run python scripts/play.py "${COMMON[@]}" --agent "$AGENT" "$@"
fi
