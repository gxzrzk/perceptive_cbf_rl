#!/usr/bin/env bash
# ==============================================================================
# 教室过道环境全向避障与巡航策略可视化回放脚本 (Play Classroom Dodge Policy)
# ==============================================================================
# 1. 任务与算法定位:
#    - 任务环境: Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat
#    - 感知与场景模式: 状态真值 Oracle 模态 + 真实教室多排课桌过道几何 + 360° 全向高速抛球。
#    - 动作特性: 机器人遵循随机生成的过道巡航路径持续行进，并在遭遇全向来球时灵活闪避且避免触碰课桌。
#    - 可视化平台: 启动 Viser Web 端可视化服务器 (默认访问 http://localhost:8080)。
#
# 2. 检查点 (Checkpoint) 解析机制:
#    - 第一个非 '-' 参数可显式传入检查点路径 (如 logs/rsl_rl/state_classroom/<run>/model_25000.pt)。
#    - 若未指定且未开启 dummy 模式，自动搜索 logs/rsl_rl/${EXP_NAME}（或指定 RUN 目录下）按迭代
#      版本号最高 (sort -V) 的最新 model_*.pt 文件。
#    - 支持传入 `--agent zero` 或 `--agent random` 直接在无模型状态下交互预览空场景。
#
# 3. 环境变量重载选项:
#    - RUN: 指定训练子目录名 (如 2026-09-19_10-00-00)
#    - NUM_ENVS: 可视化同屏机器人数量 (默认: 1)
#    - EXP_NAME: 实验日志根目录名称 (默认: state_classroom)
#    - RESET_STAND: 是否重置为标准稳定站立姿态 (默认: True)
#    - TMPDIR / WARP_CACHE_PATH: 缓存目录配置，避免系统盘空间不足
#
# 4. 使用示例:
#    ./play_classroom.sh                                          # 加载最新模型
#    ./play_classroom.sh logs/rsl_rl/state_classroom/<run>/model_25000.pt
#    ./play_classroom.sh --agent zero                             # 无模型场景预览
#    RUN=<run> NUM_ENVS=4 ./play_classroom.sh
# ==============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# CUDA compilation can exhaust the small system /tmp partition. Keep temporary
# files and Warp kernels on the project disk; honor explicit user overrides.
export TMPDIR="${TMPDIR:-$REPO_ROOT/.cache/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$REPO_ROOT/.cache/warp}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

TASK="Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat"

# First positional arg (anything not starting with '-') = direct checkpoint path.
CKPT=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

EXP_NAME="${EXP_NAME:-state_classroom}"
EXP_DIR="logs/rsl_rl/${EXP_NAME}"

# Dummy agents use the registered task without loading a checkpoint.
DUMMY_MODE=0
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[i]}" in
    --agent=zero|--agent=random) DUMMY_MODE=1 ;;
    --agent)
      if [[ "${ARGS[i+1]:-}" == zero || "${ARGS[i+1]:-}" == random ]]; then
        DUMMY_MODE=1
      fi
      ;;
  esac
done

# No explicit checkpoint path -> newest model_*.pt under the RUN dir (if set) or the
# experiment dir. "Newest" = highest ITERATION NUMBER in the filename (sort -V), NOT
# file mtime: logs synced from a training server all share the copy time, so mtime
# order is arbitrary transfer order and can pick e.g. model_8000 over model_24999.
if [[ -z "$CKPT" && "$DUMMY_MODE" == 0 ]]; then
  SEARCH_DIR="$EXP_DIR"
  [[ -n "${RUN:-}" ]] && SEARCH_DIR="${EXP_DIR}/${RUN}"
  CKPT="$(find "$SEARCH_DIR" -name 'model_*.pt' 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "$CKPT" ]]; then
    echo "[play_classroom.sh] No model_*.pt under ${SEARCH_DIR}. Pass a checkpoint path as arg 1," >&2
    echo "[play_classroom.sh] or use '--agent zero' to watch the untrained scene." >&2
    exit 1
  fi
elif [[ -n "$CKPT" && ! -f "$CKPT" ]]; then
  echo "[play_classroom.sh] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
RESET_STAND="${RESET_STAND:-True}"

echo "[play_classroom.sh] task=${TASK} (state oracle + random classroom routes + omni throws)"
echo "[play_classroom.sh] checkpoint=${CKPT}"
echo "[play_classroom.sh] num_envs=${NUM_ENVS} viewer=viser reset_stand=${RESET_STAND}"
echo "[play_classroom.sh] extra args: $*"

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
