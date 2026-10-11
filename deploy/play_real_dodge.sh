#!/usr/bin/env zsh
# ==============================================================================
# Unitree G1 实机单目深度躲避策略总启动脚本 (Jetson Orin 硬件平台)
#
# 在部署架构中的角色：
# -------------------
# 本脚本是真实人形机器人实机部署的规范化总入口，负责一键编排并拉起相互隔离的三进程架构：
#   1. camera_node / camera_node_etam (后台运行): ZED Mini 相机采集与深度下采样/分割；
#   2. dodge_policy (后台运行): 基于 ONNXRuntime 的 CPU 策略推理节点；
#   3. hardware_node (前台运行): 硬件驱动与无线遥控安全状态机，常驻前台便于实时监控。
#
# 使用方式 (需使用 zsh 运行以支持路径扩展)：
#   zsh deploy/play_real_dodge.sh                         # 自动寻找 deploy/ckpts/ 下最新的 *.onnx
#   zsh deploy/play_real_dodge.sh path/to/model.onnx      # 指定特定的 ONNX 模型路径
#   (传入 .pt 文件将报错：实机上不安装 mjlab，必须先在开发机导出为 .onnx)
#
# 生产环境推荐启动范式 (启用 EfficientTAM 球体分割与逼近门控)：
#   STATIC_MASK=1 ETAM=1 TINY=1 NET=eth0 zsh deploy/play_real_dodge.sh
#
# 常用环境变量配置说明：
# ---------------------
#   NET              : 机器人 DDS 通信物理网卡名 (默认自动提示或取 eth0)
#   CKPT_DIR         : 默认查找 ONNX 模型的目录 (默认 deploy/ckpts)
#   WALK_ONNX        : 双足行走策略路径 (默认 deploy/ckpts/walk_policy.onnx，若存在则启用 Y 键切换)
#   FRAME_OFFSETS    : 深度图时间堆叠步数偏移 (默认 "(0,3,8,18)"，必须与模型导出时完全一致)
#   CAMERA_HZ        : 相机 UDP 发布频率 (默认 50 Hz，匹配电机控制周期)
#   KP_SCALE         : 关节位置刚度缩放比例 (默认 1.0；初次试跑可设为 0.7~0.8，更柔顺安全)
#   ETAM             : 设为 1 启用 EfficientTAM 目标分割纯净深度图 (替代原始池化深度)
#   TINY             : 设为 1 使用高效的 tiny EfficientTAM 模型 (推荐在 Orin 边缘端开启)
#   STATIC_MASK      : 设为 1 启用逼近门控 (屏蔽世界静止小球，仅对逼近飞球做出反应)
#   FREEZE           : 设为 1 锁存首帧有效深度图用于诊断静态平衡
#   VIEW             : 设为 1 开启 OpenCV 图形监控窗口 (需要显示环境)
#   OBS_FAR          : 设为 1 强制策略接收全远深度 (相机运行但机器人不反应，纯站立测试)
#   DRY_RUN          : 设为 1 纯虚拟无相机回环测试 (上电前的网络与状态机联调)
# ==============================================================================
set -e
set -o pipefail

SCRIPT_DIR=${0:A:h}                 # .../deploy 绝对路径
ROOT_DIR=${SCRIPT_DIR:h}           # 仓库根目录绝对路径
TASK="Unitree-G1-AMP-Dodge-Depth-Single-Flat"
EXP_NAME="${EXP_NAME:-g1_amp_dodge_depth_single}"
FRAME_OFFSETS="${FRAME_OFFSETS:-(0,3,8,18)}"
CAMERA_HZ="${CAMERA_HZ:-50}"   # 相机发布频率对齐 50 Hz 控制率；ZED 硬件以 60 FPS 抓取
KP_SCALE="${KP_SCALE:-1.0}"   # 刚度缩放因子 (KD 阻尼保持标称值)
DRY_RUN="${DRY_RUN:-0}"

# 节点命令行匹配模式（用于在启动前清理残留进程及退出时级联终止）
NODE_PATS=(
  "deploy/real/camera_node.py"
  "deploy/real/camera_node_etam.py"
  "deploy/policy/dodge_policy.py"
  "deploy/real/hardware_node.py"
)

# 级联终止残留节点：
# 先发送 SIGTERM 允许各节点执行 finally 块（例如 camera_node 优雅关闭 ZED 硬件相机句柄）；
# 等待最长约 3 秒，若仍有未退出的顽固进程，则升级为 SIGKILL 强行终止。
# 该递进机制极其重要：若 ZED SDK 底层在 C++ open() 阻塞，SIGTERM 无法响应，
# 必须 SIGKILL 释放相机独占锁，否则下次启动必死锁失败。
reap_nodes() {
  local pat
  for pat in "${NODE_PATS[@]}"; do
    pkill -TERM -f "${pat}" 2>/dev/null || true
  done
  local i
  for i in {1..10}; do
    pgrep -f "deploy/real/camera_node.py"    >/dev/null 2>&1 ||
    pgrep -f "deploy/policy/dodge_policy.py"  >/dev/null 2>&1 ||
    pgrep -f "deploy/real/hardware_node.py"   >/dev/null 2>&1 || break
    sleep 0.3
  done
  for pat in "${NODE_PATS[@]}"; do
    pkill -KILL -f "${pat}" 2>/dev/null || true
  done
}

CLEANED_UP=0
cleanup() {
  [[ "${CLEANED_UP}" == "1" ]] && return
  CLEANED_UP=1
  trap '' SIGINT SIGTERM EXIT
  echo -e '\n[play_real_dodge] 正在停止所有子进程与清理资源...'
  reap_nodes
  kill -- -$$ 2>/dev/null || true
  # 强制释放绑定的 UDP 端口
  fuser -k -KILL 9870/udp 2>/dev/null || true
  fuser -k -KILL 9871/udp 2>/dev/null || true
  fuser -k -KILL 9872/udp 2>/dev/null || true
  fuser -k -KILL 9873/udp 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 1. 运行环境配置 (Deploy uv 虚拟环境, Python 3.8, 无 ROS 依赖)
# ---------------------------------------------------------------------------
UV_PYTHON="3.8"
UV_DEPLOY=(uv run --project "${SCRIPT_DIR}" --python "${UV_PYTHON}")

# ---------------------------------------------------------------------------
# 2. 解析与校验待执行的 ONNX 模型文件
# ---------------------------------------------------------------------------
ONNX_MODEL="${1:-}"
CKPT_DIR="${CKPT_DIR:-${SCRIPT_DIR}/ckpts}"
# 双模态行走策略检查点 (纯本体感受 384->29，若存在则自动启用手柄 Y 键双模态切换)
WALK_ONNX="${WALK_ONNX:-${CKPT_DIR}/walk_policy.onnx}"
if [[ -z "${ONNX_MODEL}" ]]; then
  # 自动寻找 ckpts 目录下最新的 ONNX (排除 walk_policy)
  ONNX_MODEL="$(find "${CKPT_DIR}" -name '*.onnx' ! -name 'walk_policy*.onnx' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -z "${ONNX_MODEL}" ]] && {
    echo "错误: 在 ${CKPT_DIR} 下未找到 .onnx 策略文件。请在开发机导出并复制过来:" >&2
    echo "  uv run python deploy/export_onnx.py ${TASK} <ckpt.pt> --frame-offsets '${FRAME_OFFSETS}'" >&2
    exit 1
  }
elif [[ "${ONNX_MODEL}" == *.pt ]]; then
  echo "错误: play_real_dodge.sh 直接运行 ONNX 模型；实机不包含 mjlab，无法运行 .pt 检查点。" >&2
  echo "请先在开发机执行导出:" >&2
  echo "  uv run python deploy/export_onnx.py ${TASK} ${ONNX_MODEL} --frame-offsets '${FRAME_OFFSETS}'" >&2
  exit 1
fi
[[ -f "${ONNX_MODEL}" ]] || { echo "错误: ONNX 文件不存在: ${ONNX_MODEL}" >&2; exit 1; }
echo "[play_real_dodge] 当前使用的躲避策略模型: ${ONNX_MODEL}"

WALK_FLAG=()
if [[ -f "${WALK_ONNX}" ]]; then
  WALK_FLAG=(--walk-onnx "${WALK_ONNX}")
  echo "[play_real_dodge] 找到行走策略模型: ${WALK_ONNX} (遥控手柄 Y 键可切换 行走<->躲避)"
else
  echo "[play_real_dodge] 未找到行走策略 ${WALK_ONNX}；将以纯躲避模式运行 (Y 键无动作)。"
fi

trap cleanup SIGINT SIGTERM EXIT
cd "${ROOT_DIR}"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 3. 确定机器人 DDS 通信物理网络接口
# ---------------------------------------------------------------------------
if [[ -z "${NET:-}" ]]; then
  printf "请输入连接机器人的物理网络接口 [默认: eth0]: "
  read -r NET
  NET="${NET:-eth0}"
fi
echo "[play_real_dodge] 使用物理网卡: ${NET}"

# ---------------------------------------------------------------------------
# 4. 进程启动流程: 相机节点 (后台) -> 策略节点 (后台) -> 硬件节点 (前台)
# ---------------------------------------------------------------------------
echo "[play_real_dodge] 正在清理前次运行可能残留的孤儿进程..."
reap_nodes

CAM_ARGS=(--hz "${CAMERA_HZ}")
[[ "${DRY_RUN}" == "1" ]] && CAM_ARGS+=(--dry-run)
[[ "${VIEW:-0}" == "1" ]] && CAM_ARGS+=(--view)
[[ "${OBS_FAR:-0}" == "1" ]] && CAM_ARGS+=(--obs-far)
[[ -n "${ZED_CONFIDENCE:-}" ]]   && CAM_ARGS+=(--zed-confidence "${ZED_CONFIDENCE}")
[[ -n "${ZED_TEXTURE_CONF:-}" ]] && CAM_ARGS+=(--zed-texture-conf "${ZED_TEXTURE_CONF}")
[[ "${FILL_HOLES:-0}" == "1" ]]  && CAM_ARGS+=(--fill-holes)
[[ -n "${EDGE_CROP:-}" ]]        && CAM_ARGS+=(--edge-crop "${EDGE_CROP}")
HW_ARGS=(--kp-scale "${KP_SCALE}")

# 根据 ETAM 环境变量决定启动 EfficientTAM 掩码相机还是原始池化相机
if [[ "${ETAM:-0}" == "1" ]]; then
  ETAM_ARGS=(--hz "${CAMERA_HZ}")
  [[ "${TINY:-0}" == "1" ]] && ETAM_ARGS+=(--tiny)
  [[ "${FREEZE:-0}" == "1" ]] && ETAM_ARGS+=(--freeze-first)
  [[ "${STATIC_MASK:-0}" == "1" ]] && ETAM_ARGS+=(--static-mask)
  echo "[play_real_dodge] 正在启动 camera_node_etam (EfficientTAM 掩码深度相机节点，后台运行)..."
  "${UV_DEPLOY[@]}" python deploy/real/camera_node_etam.py "${ETAM_ARGS[@]}" &
else
  echo "[play_real_dodge] 正在启动 camera_node (原始池化深度相机节点，后台运行)${DRY_RUN:+ [虚拟模式]}..."
  "${UV_DEPLOY[@]}" python deploy/real/camera_node.py "${CAM_ARGS[@]}" &
fi

echo "[play_real_dodge] 正在启动 dodge_policy (策略推理节点，后台运行)..."
"${UV_DEPLOY[@]}" python deploy/policy/dodge_policy.py "${ONNX_MODEL}" "${WALK_FLAG[@]}" --frame-offsets "${FRAME_OFFSETS}" &

sleep 2.0

echo "[play_real_dodge] 正在启动 hardware_node (硬件驱动与安全状态机，前台常驻) [kp_scale=${KP_SCALE}]..."
"${UV_DEPLOY[@]}" python deploy/real/hardware_node.py --net "${NET}" "${HW_ARGS[@]}"
