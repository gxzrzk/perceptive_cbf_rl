#!/usr/bin/env bash
# Train the registered classroom task: random spawn pose and aisle routes,
# 360-degree timed throws, state oracle, and the WallWalk AMP locomotion prior.
#
# Usage:
#   ./train_classroom.sh
#   NUM_ENVS=512 ./train_classroom.sh --agent.max-iterations=40000
#
# Env overrides: TASK, NUM_ENVS, EXP_NAME, STAND_RATIO, INPLACE_RATIO.
# Extra args -> scripts/train.py. The classroom defaults keep every robot walking.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# CUDA compilation can exhaust the small system /tmp partition. Keep temporary
# files and Warp kernels on the project disk; honor explicit user overrides.
export TMPDIR="${TMPDIR:-$REPO_ROOT/.cache/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$REPO_ROOT/.cache/warp}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

TASK="${TASK:-Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat}"
NUM_ENVS="${NUM_ENVS:-1024}"
STAND_RATIO="${STAND_RATIO:-0.0}"
INPLACE_RATIO="${INPLACE_RATIO:-0.0}"
EXP_NAME="${EXP_NAME:-state_classroom}"  # honor EXP_NAME so runs get distinct log dirs

echo "[train_classroom.sh] task=${TASK} num_envs=${NUM_ENVS} experiment_name=${EXP_NAME}"
echo "[train_classroom.sh] rel_standing_envs=${STAND_RATIO} rel_inplace_throw_envs=${INPLACE_RATIO}"
echo "[train_classroom.sh] extra args: $*"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  exec uv run python scripts/train.py "$TASK" \
    --env.scene.num-envs="$NUM_ENVS" \
    --env.commands.twist.rel-inplace-throw-envs="$INPLACE_RATIO" \
    --env.commands.twist.rel-standing-envs="$STAND_RATIO" \
    --agent.experiment-name="$EXP_NAME" \
    --agent.max-iterations=25000 \
    --video True \
    --video-interval 48000 \
    "$@"
