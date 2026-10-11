#!/usr/bin/env bash
# ==============================================================================
# Sim2sim 仿真验证总启动脚本 (开发机专用，零硬件风险验证全栈部署)
#
# 在部署架构中的角色：
# -------------------
# 本脚本在开发机 (Dev Box, x86_64) 上启动两个独立进程，通过本地回路 UDP 通信：
#   * dodge_policy : 运行在 DEPLOY 虚拟环境 (onnxruntime)，加载真实实机导出的 ONNX 模型；
#   * sim_node     : 运行在 DEV 虚拟环境 (mjlab/MuJoCo)，同时扮演 hardware_node 与 camera_node。
#
# 验证意义：
# ---------
# 机器人若在仿真中能够稳健站立、对抛球做出有效躲避动作（存活时间长，被击中少），
# 则证明实机部署软件栈的观测组装顺序、深度堆叠环形缓冲区、动作反归一化及关节映射完全正确，
# 彻底杜绝实机上电即摔倒的风险。
#
# 使用方式：
#   bash deploy/play_sim_dodge.sh                      # 自动寻找实验目录下最新导出的 *.onnx
#   bash deploy/play_sim_dodge.sh path/to/model.onnx   # 指定特定的 ONNX 模型路径
#   RUN=2026-06-06_18-50-05 bash deploy/play_sim_dodge.sh   # 指定某次实验运行目录
#   ZED=1 bash deploy/play_sim_dodge.sh                # 接入真实物理 ZED 相机替代仿真渲染深度
#
# 常用环境变量配置：
# -----------------
#   FRAME_OFFSETS : 深度时间堆叠步数偏移 (默认 "(0,3,8,18)")
#   DEPTH_DECIM   : 深度图发布步数间隔 (默认 1 = 50 Hz; 5 = 10 Hz 模拟低帧率相机)
#   VIEWER        : 可视化界面 (viser: 网页端，默认; native: 本地窗口; none: 无头压测)
#   RESET_STAND   : 回合重置姿态 (True: 默认标称微屈膝站立; False: 保持 RSI 运动随机位姿)
#   DEPTH_AUG     : 深度加噪模式 (True: 默认发送训练级加噪深度; False: 发送纯净几何深度)
#   DEVICE        : 仿真计算设备 (cuda:0 或 cpu)
# ==============================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../deploy 目录
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                     # 仓库根目录
TASK="${TASK:-Unitree-G1-AMP-Dodge-Depth-Single-Flat}"
EXP_NAME="${EXP_NAME:-g1_amp_dodge_depth_single}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"
DEPTH_DECIM="${DEPTH_DECIM:-1}"   # 默认每步发送一帧深度 (50 Hz)

# 优雅退出与资源清理函数
cleanup() {
  trap '' SIGINT SIGTERM EXIT
  echo -e '\n[play_sim_dodge] 正在停止所有仿真与策略进程...'
  pkill -f "deploy/policy/dodge_policy.py" 2>/dev/null || true
  pkill -f "deploy/sim/sim_node.py"        2>/dev/null || true
  pkill -f "deploy/real/camera_node.py"    2>/dev/null || true
  pkill -f "deploy/real/camera_node_etam.py" 2>/dev/null || true
  fuser -k -KILL 9870/udp 2>/dev/null || true
  fuser -k -KILL 9871/udp 2>/dev/null || true
  fuser -k -KILL 9872/udp 2>/dev/null || true
}

cd "$ROOT_DIR"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 1. 解析待运行的 ONNX 策略模型路径
# ---------------------------------------------------------------------------
ONNX_MODEL="${1:-}"
EXP_DIR="${ROOT_DIR}/logs/rsl_rl/${EXP_NAME}"
if [[ -z "${ONNX_MODEL}" ]]; then
  SEARCH_DIR="${EXP_DIR}"
  [[ -n "${RUN:-}" ]] && SEARCH_DIR="${EXP_DIR}/${RUN}"
  ONNX_MODEL="$(find "${SEARCH_DIR}" -name '*.onnx' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -z "${ONNX_MODEL}" ]] && {
    echo "错误: 在 ${SEARCH_DIR} 未找到 .onnx 模型。请先执行导出:" >&2
    echo "  uv run python deploy/export_onnx.py ${TASK} <ckpt.pt> --frame-offsets '${FRAME_OFFSETS}'" >&2
    exit 1
  }
elif [[ "${ONNX_MODEL}" == *.pt ]]; then
  echo "错误: play_sim_dodge.sh 直接执行 ONNX 推理；请先导出模型:" >&2
  echo "  uv run python deploy/export_onnx.py ${TASK} ${ONNX_MODEL} --frame-offsets '${FRAME_OFFSETS}'" >&2
  exit 1
fi
[[ -f "${ONNX_MODEL}" ]] || { echo "错误: ONNX 文件不存在: ${ONNX_MODEL}" >&2; exit 1; }
echo "[play_sim_dodge] 使用 ONNX 模型: ${ONNX_MODEL}"

trap cleanup SIGINT SIGTERM EXIT

# ---------------------------------------------------------------------------
# 2. 在 DEPLOY 虚拟环境 (onnxruntime) 中后台启动策略推理节点
# ---------------------------------------------------------------------------
echo "[play_sim_dodge] 正在启动 dodge_policy (部署环境, 后台运行)..."
uv run --project "${SCRIPT_DIR}" python deploy/policy/dodge_policy.py \
  "${ONNX_MODEL}" --frame-offsets "${FRAME_OFFSETS}" &

# ---------------------------------------------------------------------------
# 2b. 外部相机模式处理 (ZED=1 或 ETAM=1): 接入物理相机信号
# ---------------------------------------------------------------------------
ZED="${ZED:-0}"
ETAM="${ETAM:-0}"
EXT_DEPTH_FLAGS=()
if [[ "${ETAM}" == "1" ]]; then
  # 真实 ZED + EfficientTAM 球体分割掩码深度
  echo "[play_sim_dodge] 正在启动 camera_node_etam (EfficientTAM 掩码深度, 后台运行)..."
  "${SCRIPT_DIR}/.venv/bin/python" deploy/real/camera_node_etam.py --hz 50 ${TINY:+--tiny} &
  EXT_DEPTH_FLAGS=(--external-depth --pause-throws)
elif [[ "${ZED}" == "1" ]]; then
  # 真实 ZED 原始池化深度
  echo "[play_sim_dodge] 正在启动 camera_node (真实 ZED 相机原始深度, 后台运行)..."
  uv run python deploy/real/camera_node.py --hz 50 \
    ${VIEW:+--view} &
  EXT_DEPTH_FLAGS=(--external-depth --pause-throws)
fi

sleep 2.0

# ---------------------------------------------------------------------------
# 3. 在 DEV 环境 (mjlab/MuJoCo) 中前台启动仿真节点
# ---------------------------------------------------------------------------
VIEWER="${VIEWER:-viser}"
RESET_STAND="${RESET_STAND:-True}"
[[ "${RESET_STAND}" == "True" || "${RESET_STAND}" == "true" || "${RESET_STAND}" == "1" ]] \
  && RESET_FLAG="--reset-stand" || RESET_FLAG="--no-reset-stand"
DEPTH_AUG="${DEPTH_AUG:-True}"
[[ "${DEPTH_AUG}" == "True" || "${DEPTH_AUG}" == "true" || "${DEPTH_AUG}" == "1" ]] \
  && AUG_FLAG="--depth-aug" || AUG_FLAG="--no-depth-aug"

echo "[play_sim_dodge] 正在启动 sim_node (DEV环境, 前台常驻, viewer=${VIEWER}, reset_stand=${RESET_STAND}, depth_aug=${DEPTH_AUG}, zed=${ZED})..."
uv run python deploy/sim/sim_node.py "${TASK}" --frame-offsets "${FRAME_OFFSETS}" \
  --viewer "${VIEWER}" ${RESET_FLAG} ${AUG_FLAG} "${EXT_DEPTH_FLAGS[@]}" \
  --depth-decimation "${DEPTH_DECIM}" ${DEVICE:+--device "${DEVICE}"}
