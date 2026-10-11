#!/usr/bin/env bash
# ==============================================================================
# 带真值瞄准云台的单目深度避障策略回放与域随机化预览脚本 (Play Gimbal Dodge Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat
#    - 感知与云台模式:
#      * 启用真值引导的头部相机俯仰云台铰链 (CAMERA_GIMBAL=1, 范围 +/-30°)，每步自动追踪飞来球体。
#      * 将云台俯仰关节角度及速度纳入本体感知观测 (CAMERA_PROPRIO=1)，匹配 gimbal_* 系列训练模型。
#      * 输入观测为 BallOnlyDepthObs 掩码深度图（仅保留球体深度，其余为 far）。
#    - 域随机化 (DR) 预览:
#      * 默认开启 BALLONLY_AUG=1，在 Viser 面板中实时展示策略训练时所承受的球体深度抖动、整球瞬态丢帧
#        和边缘闪烁等 Sim2Real 噪声。
#      * 设为 BALLONLY_AUG=0 可切换为纯净无噪的干净掩码球进行评估对比。
#    - 倾角一致性: CAMERA_TILT_DEG 必须严格与训练时保持一致（默认 20°），否则球体成像高度失配。
#
# 2. 检查点 (Checkpoint) 解析机制:
#    - 第一个非 '-' 参数传入 Checkpoint 路径（如 logs/.../gimbal_link/<run>/model_*.pt）。
#    - 若不传 Checkpoint，则默认启动静止站立 Dummy Agent (agent=zero)，方便单独检查云台与深度观测。
#
# 3. 环境变量重载选项:
#    - NUM_ENVS: 并行环境数 (默认: 1)
#    - FRAME_OFFSETS: 历史时序深度帧偏移量元组 (默认: (0,3,8,18))
#    - CAMERA_TILT_DEG: 头部相机基准仰角 (默认: 20)
#    - BALLONLY_AUG: 是否开启掩码深度域随机化 (默认: 1, 0 表示干净评测)
#    - AGENT: 未指定模型时的 Dummy 类型 (默认: zero)
#
# 4. 使用示例:
#    ./play_gimbal.sh                                      # 启动静止 Dummy + 观察云台与深度 DR
#    ./play_gimbal.sh logs/rsl_rl/gimbal_link/<run>/model_20000.pt
#    BALLONLY_AUG=0 ./play_gimbal.sh <ckpt>                # 干净无噪深度图评测
#    NUM_ENVS=4 ./play_gimbal.sh
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

# Oracle-aimed camera-pitch gimbal: the camera sits on a +/-30 deg pitch hinge driven by a
# per-step oracle aim event; the joint state enters proprio (matches gimbal_* checkpoints).
export CAMERA_GIMBAL=1
export CAMERA_PROPRIO=1

# Play WITH the masked-depth DR by default (the policy trained on it -- in-distribution). Set
# BALLONLY_AUG=0 to eval on the clean masked ball instead.
export BALLONLY_AUG="${BALLONLY_AUG:-1}"

NUM_ENVS="${NUM_ENVS:-1}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"
AGENT="${AGENT:-zero}"  # standing dummy by default (no checkpoint needed)

echo "[play_gimbal.sh] task=${TASK} CAMERA_TILT_DEG=${CAMERA_TILT_DEG} BALLONLY_AUG=${BALLONLY_AUG} (1=DR on, 0=clean eval)"
echo "[play_gimbal.sh] depth_frame_offsets=${FRAME_OFFSETS} num_envs=${NUM_ENVS} viewer=viser"

COMMON=( "$TASK" --num-envs "$NUM_ENVS" --viewer viser --depth-frame-offsets "$FRAME_OFFSETS" )

if [[ -n "$CKPT" ]]; then
  [[ -f "$CKPT" ]] || { echo "[play_gimbal.sh] checkpoint not found: ${CKPT}" >&2; exit 1; }
  echo "[play_gimbal.sh] policy checkpoint=${CKPT}"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    exec uv run python scripts/play.py "${COMMON[@]}" --checkpoint-file "$CKPT" "$@"
else
  echo "[play_gimbal.sh] dummy agent=${AGENT} (standing; pass a checkpoint to drive a policy)"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    exec uv run python scripts/play.py "${COMMON[@]}" --agent "$AGENT" "$@"
fi
