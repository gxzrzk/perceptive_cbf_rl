#!/usr/bin/env bash
# Play / visualize the STATE-BASED (oracle) dodge + WALL policy
# (Unitree-G1-AMP-Dodge-MimicKit-Wall-Flat) in the viser web viewer (default
# http://localhost:8080). Same perception regime as play_state.sh (ground-truth
# ball state, no camera), but a static wall stands beside the spawn point --
# randomized side/distance each episode -- and the robot must dodge WITHOUT
# touching it (wall contact terminates the episode).
#
# NOTE: checkpoints from the plain state task (Unitree-G1-AMP-Dodge-MimicKit-Flat)
# DO NOT load here -- the ball_state obs group grew from 6 to 9 dims (wall_state
# appended). Train a wall-task checkpoint first:
#   TASK=Unitree-G1-AMP-Dodge-MimicKit-Wall-Flat EXP_NAME=state_wall_link ./train_dodge_state.sh
#
# The first positional arg is a direct checkpoint path; omit it to use the newest
# model_*.pt under the experiment dir (logs/rsl_rl/state_wall_link).
#
# Usage:
#   ./play_wall.sh                                          # newest checkpoint
#   ./play_wall.sh logs/rsl_rl/state_wall_link/<run>/model_25000.pt
#   RUN=2026-09-19_10-00-00 ./play_wall.sh                  # newest under a specific run dir
#   NUM_ENVS=9 ./play_wall.sh                               # more robots on screen
#   ./play_wall.sh --agent zero                             # untrained (see the wall + throws now)
#
# Env overrides: RUN, NUM_ENVS, EXP_NAME, RESET_STAND. Extra args -> scripts/play.py.
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
