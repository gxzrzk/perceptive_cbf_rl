#!/usr/bin/env bash
# Play / visualize the STATE-BASED (oracle) dodge WHILE WALKING through a wall
# corridor policy (Unitree-G1-AMP-Dodge-MimicKit-WallWalk-Flat) in the viser web
# viewer (default http://localhost:8080). Each episode gets a RANDOMLY SHAPED
# walking path (gentle-curve polyline); the robot walks along it continuously
# (~1.3 m/s, pure-pursuit lookahead) while dodging the thrown ball AND slaloming
# WALLWALK_NUM_WALLS (default 3) static walls flanking the path -- walked-past
# walls recycle back ahead along the curve, so the corridor is endless.
#
# NOTE: checkpoints from the plain state task OR the standing wall task DO NOT
# load here -- the ball_state obs group is 6 + 3*WALLWALK_NUM_WALLS dims (15 by
# default). Train a wallwalk checkpoint first:
#   TASK=Unitree-G1-AMP-Dodge-MimicKit-WallWalk-Flat EXP_NAME=state_wallwalk ./train_dodge_state.sh
#
# The first positional arg is a direct checkpoint path; omit it to use the newest
# model_*.pt under the experiment dir (logs/rsl_rl/state_wallwalk).
#
# Usage:
#   ./play_wallwalk.sh                                          # newest checkpoint
#   ./play_wallwalk.sh logs/rsl_rl/state_wallwalk/<run>/model_25000.pt
#   RUN=2026-09-19_10-00-00 ./play_wallwalk.sh                  # newest under a specific run dir
#   NUM_ENVS=9 ./play_wallwalk.sh                               # more robots on screen
#   WALLWALK_NUM_WALLS=5 ./play_wallwalk.sh --agent zero        # denser corridor, untrained
#
# Env overrides: RUN, NUM_ENVS, EXP_NAME, RESET_STAND, WALLWALK_NUM_WALLS,
# WALK_PATH_LOOKAHEAD, WALK_PATH_TURN_MAX, WALK_MAX_VEL_X. Extra args -> scripts/play.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TASK="Unitree-G1-AMP-Dodge-MimicKit-WallWalk-Flat"

# First positional arg (anything not starting with '-') = direct checkpoint path.
CKPT=""
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

EXP_NAME="${EXP_NAME:-state_wallwalk}"
EXP_DIR="logs/rsl_rl/${EXP_NAME}"

# No explicit checkpoint path -> newest model_*.pt under the RUN dir (if set) or the experiment dir.
if [[ -z "$CKPT" ]]; then
  SEARCH_DIR="$EXP_DIR"
  [[ -n "${RUN:-}" ]] && SEARCH_DIR="${EXP_DIR}/${RUN}"
  CKPT="$(find "$SEARCH_DIR" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
            | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -z "$CKPT" ]]; then
    echo "[play_wallwalk.sh] No model_*.pt under ${SEARCH_DIR}. Pass a checkpoint path as arg 1," >&2
    echo "[play_wallwalk.sh] or use '--agent zero' to watch the untrained scene." >&2
    exit 1
  fi
elif [[ ! -f "$CKPT" ]]; then
  echo "[play_wallwalk.sh] Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
RESET_STAND="${RESET_STAND:-True}"

echo "[play_wallwalk.sh] task=${TASK} (state oracle + walk + wall corridor)"
echo "[play_wallwalk.sh] checkpoint=${CKPT}"
echo "[play_wallwalk.sh] num_envs=${NUM_ENVS} viewer=viser reset_stand=${RESET_STAND}"
echo "[play_wallwalk.sh] walls=${WALLWALK_NUM_WALLS:-3} extra args: $*"

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
