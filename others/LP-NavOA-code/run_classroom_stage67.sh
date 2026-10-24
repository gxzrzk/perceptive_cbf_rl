#!/usr/bin/env bash
# Continue the classroom curriculum from the completed Stage 3 and Stage 5.
# Runs Stage 6 (indoor planner), then Stage 7 (classroom planner).

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

BACKBONE_CKPT="logs/rsl_rl/g1_velocity_raycast_h1/2026-10-22_00-33-49/model_12497.pt"
STAGE5_CKPT="logs/steering/local_planner_low_speed_v6_511_final.pt"
PYTHON_BIN="${LP_PYTHON:-./.venv-stage67/bin/python}"
ROBOT_VERSION=v15

usage() {
    echo "Usage: $0 [--backbone-checkpoint PATH] [--stage5-checkpoint PATH] [--python PATH] [--robot-version v5|v15]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone-checkpoint) BACKBONE_CKPT="$2"; shift 2 ;;
        --stage5-checkpoint) STAGE5_CKPT="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --robot-version) ROBOT_VERSION="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ "$ROBOT_VERSION" == v5 || "$ROBOT_VERSION" == v15 ]] || { echo "Invalid robot version: $ROBOT_VERSION" >&2; exit 2; }
[[ -f "$BACKBONE_CKPT" ]] || { echo "Missing backbone checkpoint: $BACKBONE_CKPT" >&2; exit 1; }
[[ -f "$STAGE5_CKPT" ]] || { echo "Missing Stage 5 checkpoint: $STAGE5_CKPT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python interpreter: $PYTHON_BIN" >&2; exit 1; }
export G1_ROBOT_VERSION="$ROBOT_VERSION"

OUTPUT_DIR=logs/steering
STAGE6_CKPT="$OUTPUT_DIR/local_planner_indoor_approach_v6_511_final.pt"
STAGE7_CKPT="$OUTPUT_DIR/local_planner_classroom_v6_511_final.pt"
mkdir -p "$OUTPUT_DIR"

echo "[Stage 6] indoor planner, warm start: $STAGE5_CKPT"
"$PYTHON_BIN" -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task Mjlab-IndoorStraightApproach-G1-v6 \
    --checkpoint "$BACKBONE_CKPT" \
    --local_planner_checkpoint "$STAGE5_CKPT" \
    --num_envs 1024 \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name indoor_approach_v6_511 \
    --raycast-obs-names '("raycast", "raycast_left", "raycast_right")' \
    --max_iterations 20000

[[ -s "$STAGE6_CKPT" ]] || { echo "Stage 6 finished without final checkpoint: $STAGE6_CKPT" >&2; exit 1; }
echo "[Stage 7] classroom planner, warm start: $STAGE6_CKPT"
"$PYTHON_BIN" -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task Mjlab-Classroom-G1-v6 \
    --checkpoint "$BACKBONE_CKPT" \
    --local_planner_checkpoint "$STAGE6_CKPT" \
    --num_envs 1024 \
    --output_dir "$OUTPUT_DIR" \
    --experiment_name classroom_v6_511 \
    --raycast-obs-names '("raycast", "raycast_left", "raycast_right")' \
    --max_iterations 20000

[[ -s "$STAGE7_CKPT" ]] || { echo "Stage 7 finished without final checkpoint: $STAGE7_CKPT" >&2; exit 1; }
echo "[Stage 7] complete: $STAGE7_CKPT"
