#!/usr/bin/env bash
# One-shot watcher for the lp-stage67 tmux training session.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LP_DIR="$ROOT_DIR/others/LP-NavOA-code"
TRAIN_SESSION=lp-stage67
TRAIN_LOG="$LP_DIR/logs/steering/stage67_20261023.log"
PLANNER="$LP_DIR/logs/steering/local_planner_classroom_v6_511_final.pt"
PAC_CHECKPOINT="$ROOT_DIR/logs/rsl_rl/classroom_vs_lp/2026-10-22_15-42-09/model_24999.pt"
LP_BACKBONE="$LP_DIR/logs/rsl_rl/g1_velocity_raycast_h1/2026-10-22_00-33-49/model_12497.pt"
OUTPUT_DIR="$ROOT_DIR/logs/classroom_comparison/stage7_final_20261023"
COMPLETED='[Stage 7] complete: logs/steering/local_planner_classroom_v6_511_final.pt'

mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$OUTPUT_DIR/watcher.log") 2>&1

for required in "$ROOT_DIR/.venv/bin/python" "$LP_DIR/.venv-stage67/bin/python" \
                "$ROOT_DIR/others/compare_classroom.py" "$PAC_CHECKPOINT" "$LP_BACKBONE"; do
    [[ -s "$required" ]] || { echo "Missing comparison input: $required" >&2; exit 1; }
done
command -v tmux >/dev/null || { echo 'tmux is unavailable' >&2; exit 1; }
command -v rg >/dev/null || { echo 'rg is unavailable' >&2; exit 1; }

echo "Waiting for $TRAIN_SESSION to finish Stage 7..."
while ! { [[ -f "$TRAIN_LOG" ]] && rg -Fq "$COMPLETED" "$TRAIN_LOG"; }; do
    if ! tmux has-session -t "=$TRAIN_SESSION" 2>/dev/null; then
        echo "Training session ended without the Stage 7 completion marker: $TRAIN_LOG" >&2
        exit 1
    fi
    sleep 2
done
[[ -s "$PLANNER" ]] || { echo "Stage 7 completion marker found, but checkpoint is missing: $PLANNER" >&2; exit 1; }

export G1_ROBOT_VERSION=v15
export CUDA_VISIBLE_DEVICES="${COMPARE_CUDA_VISIBLE_DEVICES:-0}"
echo "Stage 7 complete. Starting 128 paired classroom cases on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
if "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/others/compare_classroom.py" \
    --episodes 128 --batch-size 32 \
    --pac-checkpoint "$PAC_CHECKPOINT" \
    --lp-backbone "$LP_BACKBONE" \
    --lp-planner "$PLANNER" \
    --output-dir "$OUTPUT_DIR"; then
    echo "Comparison complete: $OUTPUT_DIR/summary.json"
else
    status=$?
    echo "Comparison failed (exit $status); see $OUTPUT_DIR/watcher.log" >&2
    exit "$status"
fi
