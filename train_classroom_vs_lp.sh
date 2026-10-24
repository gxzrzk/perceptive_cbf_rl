#!/usr/bin/env bash
# LP-NavOA comparison run: pure-navigation classroom, 0.8 m/s cruise, no ball
# throws. Wraps train_classroom.sh with WALK_THROW_ENABLED=0 + WALK_V_REF=0.8
# and logs under a dedicated experiment name.
#
# Usage:
#   ./train_classroom_vs_lp.sh
#   NUM_ENVS=512 ./train_classroom_vs_lp.sh --agent.max-iterations=40000
#
# Extra args are forwarded to train_classroom.sh -> scripts/train.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WALK_THROW_ENABLED="${WALK_THROW_ENABLED:-0}"
export WALK_V_REF="${WALK_V_REF:-0.8}"
export EXP_NAME="${EXP_NAME:-classroom_vs_lp}"

echo "[train_classroom_vs_lp.sh] WALK_THROW_ENABLED=${WALK_THROW_ENABLED} WALK_V_REF=${WALK_V_REF} EXP_NAME=${EXP_NAME}"

exec "$REPO_ROOT/train_classroom.sh" "$@"
