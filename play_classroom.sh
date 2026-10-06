#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
ARGS=(--agent zero)
if [[ $# -gt 0 && "$1" != -* ]]; then
  ARGS=(--agent trained --checkpoint-file "$1")
  shift
fi
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" exec uv run python scripts/play.py \
  Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat \
  "${ARGS[@]}" --num-envs "${NUM_ENVS:-1}" --viewer viser --reset-stand True "$@"
