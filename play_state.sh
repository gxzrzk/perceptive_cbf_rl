#!/usr/bin/env bash
# ==============================================================================
# 状态真值 Oracle 避障策略可视化回放脚本 (Play State-based Oracle Dodge Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-MimicKit-Flat
#    - 感知模式: 状态真值 Oracle 模态（策略直接读取球体在自身坐标系下的 6 维相对位置与速度）
#               无相机渲染、无深度图，作为评估避障能力的“感知上限”对照基线。
#    - 特点: 纯几何与物理交互，Viser 界面中无需打开深度面板，也不需要配置帧偏移参数。
#    - 可视化平台: 启动 Viser Web 端可视化服务器 (默认访问 http://localhost:8080)。
#
# 2. 检查点 (Checkpoint) 解析机制:
#    - 第一个非 '-' 参数可显式传入检查点路径 (如 logs/rsl_rl/state_link/<run>/model_25000.pt)。
#    - 若未指定，则自动搜索 logs/rsl_rl/${EXP_NAME}（或指定 RUN 目录下）最新的 model_*.pt 文件。
#
# 3. 环境变量重载选项:
#    - RUN: 指定训练子目录名 (如 2026-06-02_19-35-53)
#    - NUM_ENVS: 可视化同屏机器人数量 (默认: 1)
#    - EXP_NAME: 实验日志根目录名称 (默认: state_link)
#    - RESET_STAND: 是否重置为标准稳定站立姿态 (默认: True)
#
# 4. 使用示例:
#    ./play_state.sh                                          # 加载最新模型
#    ./play_state.sh logs/rsl_rl/state_link/<run>/model_25000.pt
#    RUN=2026-06-02_19-35-53 ./play_state.sh
#    NUM_ENVS=4 ./play_state.sh
#    ./play_state.sh --video True                             # 录制回放视频
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="Unitree-G1-AMP-Dodge-MimicKit-Flat"

# First positional arg (anything not starting with '-') = direct checkpoint path.
CKPT=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

EXP_NAME="${EXP_NAME:-state_link}"
EXP_DIR="logs/rsl_rl/${EXP_NAME}"

# No explicit checkpoint path -> newest model_*.pt under the RUN dir (if set) or the experiment dir.
if [[ -z "$CKPT" ]]; then
  SEARCH_DIR="$EXP_DIR"
  [[ -n "${RUN:-}" ]] && SEARCH_DIR="${EXP_DIR}/${RUN}"
  CKPT="$(find "$SEARCH_DIR" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$CKPT" ]]; then
    echo "[play_state.sh] No model_*.pt under ${SEARCH_DIR}. Pass a checkpoint path as arg 1." >&2
    exit 1
  fi
elif [[ ! -f "$CKPT" ]]; then
  echo "[play_state.sh] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
RESET_STAND="${RESET_STAND:-True}"

echo "[play_state.sh] task=${TASK} (state-based oracle, ground-truth ball obs)"
echo "[play_state.sh] checkpoint=${CKPT}"
echo "[play_state.sh] num_envs=${NUM_ENVS} viewer=viser reset_stand=${RESET_STAND}"
echo "[play_state.sh] extra args: $*"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec uv run python scripts/play.py "$TASK" \
    --checkpoint-file "$CKPT" \
    --num-envs "$NUM_ENVS" \
    --viewer viser \
    --reset-stand "$RESET_STAND" \
    "$@"
