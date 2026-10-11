#!/usr/bin/env bash
# ==============================================================================
# 行走与躲避双模态策略 (Walk <-> Dodge Dual-Mode) Sim2sim 验证脚本
#
# 在部署架构中的角色：
# -------------------
# 本脚本在仿真环境中同时加载两个预训练的 ONNX 策略网络，并在 Viser 网页交互界面
# 中支持实时动态无缝切换：
#   1. 行走策略 (WALK_ONNX) : 纯本体感受 (384 -> 29) AMP-Flat 双足全向移动步态；
#   2. 躲避策略 (DODGE_ONNX): 本体感受 + 深度图时序堆叠 (960 -> 29) 敏捷躲避策略。
#
# 平滑切换技术细节：
# -----------------
# - 零增益切换：两个策略采用完全相同的 G1 电机刚度 KP 和阻尼 KD，无需切换执行器参数；
# - 关节目标线性过渡 (Target Blending)：策略节点在切换瞬间捕获最后目标位置，
#   并在 BLEND_TICKS (默认 25 步，50Hz 下约 0.5 秒) 内执行平滑线性插值，杜绝步态跃变冲击；
# - 视觉环形缓冲区保活：在行走模式期间，策略节点依然每步推入最新深度帧，
#   使得切回躲避模式时具备即时反应能力，无感知冷启动延迟。
#
# Viser 交互界面控制项：
# ---------------------
#   * "Walk mode" 复选框       : 勾选切为行走策略，取消勾选切回躲避策略；
#   * "Manual cmd" 速度滑条   : 调节 vx (前向)、vy (横向)、wz (偏航角速度) 操控行走；
#   * "Ball throws" 抛球控制箱 : 支持高球、低球、随机抛球或暂停自动投掷。
# ==============================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../deploy 目录
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                     # 仓库根目录
TASK="${TASK:-Unitree-G1-AMP-Dodge-Depth-Single-Flat}"
EXP_NAME="${EXP_NAME:-g1_amp_dodge_depth_single}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"
WALK_ONNX="${WALK_ONNX:-${SCRIPT_DIR}/ckpts/walk_policy.onnx}"
BLEND_TICKS="${BLEND_TICKS:-25}"  # 模式切换时的线性插值步数 (25 步约 0.5 秒)
DEPTH_DECIM="${DEPTH_DECIM:-1}"   # 深度图发布间隔

cleanup() {
  trap '' SIGINT SIGTERM EXIT
  echo -e '\n[play_sim_walk] 正在停止双模态仿真与策略进程...'
  pkill -f "deploy/policy/dodge_policy.py" 2>/dev/null || true
  pkill -f "deploy/sim/sim_node.py"        2>/dev/null || true
  for p in 9870 9871 9872; do fuser -k -KILL ${p}/udp 2>/dev/null || true; done
}

cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

# 1. 查找解析躲避策略 ONNX 模型
DODGE_ONNX="${1:-}"
if [[ -z "${DODGE_ONNX}" ]]; then
  DODGE_ONNX="$(find "${ROOT_DIR}/logs/rsl_rl/${EXP_NAME}" -name '*.onnx' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
fi
[[ -f "${DODGE_ONNX}" ]] || { echo "错误: 未找到躲避策略 ONNX 文件: ${DODGE_ONNX} (可通过参数 1 传入)" >&2; exit 1; }
[[ -f "${WALK_ONNX}" ]]  || { echo "错误: 未找到行走策略 ONNX 文件: ${WALK_ONNX} (可通过 WALK_ONNX=... 设定)" >&2; exit 1; }
echo "[play_sim_walk] 躲避策略: ${DODGE_ONNX}"
echo "[play_sim_walk] 行走策略: ${WALK_ONNX}"

trap cleanup SIGINT SIGTERM EXIT

# 2. 在 DEPLOY 环境后台启动双模态策略推理节点
echo "[play_sim_walk] 正在启动双模态策略推理节点 (后台运行)..."
uv run --project "${SCRIPT_DIR}" python deploy/policy/dodge_policy.py \
  "${DODGE_ONNX}" --walk-onnx "${WALK_ONNX}" \
  --frame-offsets "${FRAME_OFFSETS}" --blend-ticks "${BLEND_TICKS}" &

sleep 2.0

# 3. 在 DEV 环境前台启动带有双模态控件的 Viser 仿真器
VIEWER="${VIEWER:-viser}"
echo "[play_sim_walk] 正在启动 sim_node 仿真器 (前台常驻, viewer=${VIEWER})..."
uv run python deploy/sim/sim_node.py "${TASK}" --frame-offsets "${FRAME_OFFSETS}" \
  --viewer "${VIEWER}" --depth-aug --depth-decimation "${DEPTH_DECIM}" ${DEVICE:+--device "${DEVICE}"}
