#!/usr/bin/env bash
# ==============================================================================
# 带静态障碍墙的真值状态避障策略可视化回放脚本 (Play State Wall Dodge Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-MimicKit-Wall-Flat
#    - 感知模式: 状态真值 Oracle 模态 + 侧向静态墙体位置感知。
#               注意：由于拼接了墙体相对坐标，本任务的球体观测组维度从 6 维扩展为 9 维，
#               无法直接加载基础无墙任务 (Unitree-G1-AMP-Dodge-MimicKit-Flat) 的检查点！
#    - 场景特性: 机器人出生点旁立有一面静态障碍墙（每回合随机选择左/右侧与间距），
#               机器人必须在躲避高速袭来球体的同时严格避免接触墙壁（碰墙即判定失败终止）。
#    - 可视化平台: 启动 Viser Web 端可视化服务器 (默认访问 http://localhost:8080)。
#
# 2. 检查点 (Checkpoint) 解析机制:
#    - 第一个非 '-' 参数可显式传入检查点路径 (如 logs/rsl_rl/state_wall_link/<run>/model_25000.pt)。
#    - 若未指定，则自动搜索 logs/rsl_rl/${EXP_NAME}（或指定 RUN 目录下）按迭代次数版本号最高
#      (sort -V) 的最新 model_*.pt 文件。
#
# 3. 环境变量重载选项:
#    - RUN: 指定训练子目录名 (如 2026-09-19_10-00-00)
#    - NUM_ENVS: 可视化同屏机器人数量 (默认: 1)
#    - EXP_NAME: 实验日志根目录名称 (默认: state_wall_link)
#    - RESET_STAND: 是否重置为标准稳定站立姿态 (默认: True)
#
# 4. 使用示例:
#    ./play_wall.sh                                          # 加载最新模型
#    ./play_wall.sh logs/rsl_rl/state_wall_link/<run>/model_25000.pt
#    RUN=2026-09-19_10-00-00 ./play_wall.sh
#    NUM_ENVS=4 ./play_wall.sh
#    ./play_wall.sh --agent zero                             # 无模型观察墙体与抛球
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="Unitree-G1-AMP-Dodge-MimicKit-Wall-Flat"

# First positional arg (anything not starting with '-') = direct checkpoint path.
CKPT=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

EXP_NAME="${EXP_NAME:-state_wall_link}"
EXP_DIR="logs/rsl_rl/${EXP_NAME}"

# No explicit checkpoint path -> newest model_*.pt under the RUN dir (if set) or the
# experiment dir. "Newest" = highest ITERATION NUMBER in the filename (sort -V), NOT
# file mtime: logs synced from a training server all share the copy time, so mtime
# order is arbitrary transfer order and can pick e.g. model_8000 over model_24999.
if [[ -z "$CKPT" ]]; then
  SEARCH_DIR="$EXP_DIR"
  [[ -n "${RUN:-}" ]] && SEARCH_DIR="${EXP_DIR}/${RUN}"
  CKPT="$(find "$SEARCH_DIR" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1)"
  if [[ -z "$CKPT" ]]; then
    echo "[play_wall.sh] No model_*.pt under ${SEARCH_DIR}. Pass a checkpoint path as arg 1," >&2
    echo "[play_wall.sh] or use '--agent zero' to watch the untrained scene." >&2
    exit 1
  fi
elif [[ ! -f "$CKPT" ]]; then
  echo "[play_wall.sh] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
RESET_STAND="${RESET_STAND:-True}"

echo "[play_wall.sh] task=${TASK} (state oracle + static wall)"
echo "[play_wall.sh] checkpoint=${CKPT}"
echo "[play_wall.sh] num_envs=${NUM_ENVS} viewer=viser reset_stand=${RESET_STAND}"
echo "[play_wall.sh] extra args: $*"

CKPT_ARGS=()
if [[ -n "$CKPT" ]]; then
  CKPT_ARGS=(--checkpoint-file "$CKPT")
fi

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec uv run python scripts/play.py "$TASK" \
    "${CKPT_ARGS[@]}" \
    --num-envs "$NUM_ENVS" \
    --viewer viser \
    --reset-stand "$RESET_STAND" \
    "$@"
