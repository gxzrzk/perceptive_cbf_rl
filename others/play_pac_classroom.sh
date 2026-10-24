#!/usr/bin/env bash
# PAC classroom Viser viewer. By default, replay case 0 from the formal comparison.
#
#   ./others/play_pac_classroom.sh
#   ./others/play_pac_classroom.sh --case-id 7
#   ./others/play_pac_classroom.sh --random-routes
#   ./others/play_pac_classroom.sh /path/to/model.pt --case-id 7
#
# The fixed-case mode repeats the selected route after every reset. The random
# mode delegates to scripts/play.py, like play_classroom.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export TMPDIR="${TMPDIR:-$REPO_ROOT/.cache/tmp}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$REPO_ROOT/.cache/warp}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WALK_THROW_ENABLED=0
export WALK_V_REF="${WALK_V_REF:-0.6}"
mkdir -p "$TMPDIR" "$WARP_CACHE_PATH"

CKPT="${PAC_CHECKPOINT:-$REPO_ROOT/logs/rsl_rl/classroom_vs_lp/2026-10-22_15-42-09/model_24999.pt}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CKPT="$1"
  shift
fi

CASE_ID="${CASE_ID:-0}"
MANIFEST="${ROUTES_MANIFEST:-$REPO_ROOT/logs/classroom_comparison/stage7_final_pac_unclipped_20261023/routes.json}"
MAX_EPISODE_S="${MAX_EPISODE_S:-40}"
DEVICE="${DEVICE:-cuda:0}"
RANDOM_ROUTES=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      echo "Usage: $0 [checkpoint.pt] [--case-id N] [--manifest routes.json] [--max-episode-s SECONDS] [--device DEVICE] [--random-routes]"
      echo "Fixed-case mode replays the formal comparison route; random mode forwards extra flags to scripts/play.py."
      exit 0 ;;
    --case-id) CASE_ID="${2:?--case-id needs a value}"; shift 2 ;;
    --manifest) MANIFEST="${2:?--manifest needs a path}"; shift 2 ;;
    --max-episode-s) MAX_EPISODE_S="${2:?--max-episode-s needs a value}"; shift 2 ;;
    --device) DEVICE="${2:?--device needs a value}"; shift 2 ;;
    --random-routes) RANDOM_ROUTES=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ ! -f "$CKPT" ]]; then
  echo "PAC checkpoint not found: $CKPT" >&2
  exit 1
fi

if (( RANDOM_ROUTES )); then
  echo "[PAC viewer] random classroom routes, checkpoint=$CKPT"
  exec uv run python scripts/play.py Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat \
    --checkpoint-file "$CKPT" --num-envs "${NUM_ENVS:-1}" --viewer viser \
    --reset-stand "${RESET_STAND:-False}" --device "$DEVICE" "${EXTRA[@]}"
fi

if (( ${#EXTRA[@]} )); then
  echo "Unknown fixed-case arguments: ${EXTRA[*]}" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Route manifest not found: $MANIFEST" >&2
  exit 1
fi
echo "[PAC viewer] case=$CASE_ID manifest=$MANIFEST checkpoint=$CKPT speed=$WALK_V_REF"
exec uv run python others/play_pac_classroom.py \
  --checkpoint "$CKPT" --manifest "$MANIFEST" --case-id "$CASE_ID" \
  --max-episode-s "$MAX_EPISODE_S" --device "$DEVICE"
