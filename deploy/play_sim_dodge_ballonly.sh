#!/usr/bin/env bash
# ==============================================================================
# 纯净掩码躲避策略 (Ball-Only Policy) 真实感知在环 Sim2sim 验证脚本
#
# 在部署架构中的角色：
# -------------------
# 本脚本搭建了“真实物理相机感知 + 仿真机器人动力学”的混合闭环验证系统：
#   * dodge_policy     : 加载纯净球体掩码深度策略 ONNX，运行在 DEPLOY 虚拟环境；
#   * camera_node_etam : 真实物理 ZED 相机 + 实时 EfficientTAM 目标分割网络，
#                        生成“唯有球体保留真实深度、背景全置为 5.0m”的纯净深度图并 UDP 发送；
#   * sim_node         : 运行在 DEV 虚拟环境的 MuJoCo G1 动力学仿真器，禁用自身深度渲染
#                        (--external-depth)，接收外部真实深度并驱动仿真机器人。
#
# 交互与测试方式：
# ---------------
# 1. 启动脚本后，在弹出的 EfficientTAM 视频窗口中**鼠标点击真实网球/排球一次**；
# 2. 仿真环境已自动暂停内置的虚拟抛球 (--pause-throws) 并将基准速度置零 (--zero-command)；
# 3. 操作人员向真实 ZED 相机抛掷真球，即可实时观察仿真环境中的 G1 人形机器人做出敏捷的闪避动作！
#
# 常用环境变量：
# -------------
#   CAMERA_TILT_DEG : 相机物理安装仰角 (默认 20 度，必须与模型训练 tilt 严格一致)
#   TINY            : 设为 1 启用高效 tiny 分割模型 (提速约 2 倍)
#   STATIC_MASK     : 设为 1 启用逼近门控 (屏蔽静态小球，仅对飞来抛球反应)
#   FRAME_OFFSETS   : 深度时间堆叠步数 (默认 "(0,3,8,18)")
# ==============================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../deploy 目录
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                     # 仓库根目录
TASK="Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat"
EXP_NAME="${EXP_NAME:-g1_amp_dodge_depth_single_ballonly}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"

# 相机物理安装仰角（需对齐模型训练时的俯仰角度）
export CAMERA_TILT_DEG="${CAMERA_TILT_DEG:-20}"

cleanup() {
  trap '' SIGINT SIGTERM EXIT
  echo -e '\n[play_sim_dodge_ballonly] 正在停止所有子进程...'
  pkill -f "deploy/policy/dodge_policy.py"   2>/dev/null || true
  pkill -f "deploy/sim/sim_node.py"          2>/dev/null || true
  pkill -f "deploy/real/camera_node_etam.py" 2>/dev/null || true
  fuser -k -KILL 9870/udp 2>/dev/null || true
  fuser -k -KILL 9871/udp 2>/dev/null || true
  fuser -k -KILL 9872/udp 2>/dev/null || true
}

cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 1. 解析纯净球体掩码 ONNX 策略路径
# ---------------------------------------------------------------------------
ONNX_MODEL="${1:-}"
EXP_DIR="${ROOT_DIR}/logs/rsl_rl/${EXP_NAME}"
if [[ -z "${ONNX_MODEL}" ]]; then
  SEARCH_DIR="${EXP_DIR}"
  [[ -n "${RUN:-}" ]] && SEARCH_DIR="${EXP_DIR}/${RUN}"
  ONNX_MODEL="$(find "${SEARCH_DIR}" -path '*/export/*.onnx' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -z "${ONNX_MODEL}" ]] && {
    echo "错误: 未找到导出的 ONNX 模型。请先执行导出:" >&2
    echo "  uv run python deploy/export_onnx.py ${TASK} <ckpt.pt> --frame-offsets '${FRAME_OFFSETS}'" >&2
    exit 1
  }
elif [[ "${ONNX_MODEL}" == *.pt ]]; then
  echo "错误: 本脚本直接运行导出的 ONNX 模型；请先执行导出:" >&2
  echo "  uv run python deploy/export_onnx.py ${TASK} ${ONNX_MODEL} --frame-offsets '${FRAME_OFFSETS}'" >&2
  exit 1
fi
[[ -f "${ONNX_MODEL}" ]] || { echo "错误: ONNX 文件不存在: ${ONNX_MODEL}" >&2; exit 1; }
echo "[play_sim_dodge_ballonly] 使用纯净掩码策略模型: ${ONNX_MODEL}"
echo "[play_sim_dodge_ballonly] 相机仰角 CAMERA_TILT_DEG=${CAMERA_TILT_DEG} (请校准物理相机指向)"

trap cleanup SIGINT SIGTERM EXIT

# ---------------------------------------------------------------------------
# 2. 在 DEPLOY 环境后台拉起策略推理节点
# ---------------------------------------------------------------------------
echo "[play_sim_dodge_ballonly] 正在启动 dodge_policy (部署环境, 后台运行)..."
uv run --project "${SCRIPT_DIR}" python deploy/policy/dodge_policy.py \
  "${ONNX_MODEL}" --frame-offsets "${FRAME_OFFSETS}" &

# ---------------------------------------------------------------------------
# 3. 后台启动真实 ZED 相机 + EfficientTAM 实时掩码节点
# ---------------------------------------------------------------------------
echo "[play_sim_dodge_ballonly] 正在启动 camera_node_etam (真实 ZED + EfficientTAM 掩码深度, 后台运行)..."
echo "  -> 请在弹出的视频窗口中【鼠标单击球体】以启动跟踪 (按 r 重置, q 退出)。"
ETAM_FLAGS=(--hz 50)
[[ "${TINY:-0}" == "1" ]] && ETAM_FLAGS+=(--tiny)
[[ "${STATIC_MASK:-0}" == "1" ]] && ETAM_FLAGS+=(--static-mask)
"${SCRIPT_DIR}/.venv/bin/python" deploy/real/camera_node_etam.py "${ETAM_FLAGS[@]}" &

sleep 2.0

# ---------------------------------------------------------------------------
# 4. 在 DEV 环境前台启动 MuJoCo 仿真器
# ---------------------------------------------------------------------------
VIEWER="${VIEWER:-viser}"
RESET_STAND="${RESET_STAND:-True}"
[[ "${RESET_STAND}" == "True" || "${RESET_STAND}" == "true" || "${RESET_STAND}" == "1" ]] \
  && RESET_FLAG="--reset-stand" || RESET_FLAG="--no-reset-stand"

echo "[play_sim_dodge_ballonly] 正在启动 sim_node (前台常驻, 外部真实深度接入, 原地站立等待)..."
uv run python deploy/sim/sim_node.py "${TASK}" --frame-offsets "${FRAME_OFFSETS}" \
  --viewer "${VIEWER}" ${RESET_FLAG} --no-depth-aug --external-depth --pause-throws --zero-command \
  ${DEVICE:+--device "${DEVICE}"}
