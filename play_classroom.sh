#!/usr/bin/env bash
# Play the registered classroom policy in the viser web viewer.
# Loads the newest checkpoint under logs/rsl_rl/state_classroom by default.
#
# Usage:
#   ./play_classroom.sh
#   ./play_classroom.sh logs/rsl_rl/state_classroom/<run>/model_25000.pt
#   ./play_classroom.sh --agent zero                  # preview without a model
#   RUN=<run> NUM_ENVS=9 ./play_classroom.sh
#
# Env overrides: RUN, NUM_ENVS, EXP_NAME, RESET_STAND.
# Extra args -> scripts/play.py. WallWalk checkpoints have different obs dimensions.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

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
