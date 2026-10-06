#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TASK=Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat
export EXP_NAME="${EXP_NAME:-state_classroom}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export STAND_RATIO=0
export INPLACE_RATIO=0
exec bash "$REPO_ROOT/train_dodge_state.sh" "$@"
