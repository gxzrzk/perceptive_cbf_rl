#!/bin/bash
# Orchestrator: train the two navigation tasks, then run the formal benchmark.
#
# Run this after the locomotion backbone curriculum
# (curriculum_train_v6.sh, stages 1-3) has produced a Stage 3 checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="formal_v6"
SAFETY_BRAKE_SCALE_THRESHOLDS="0.85 0.90 0.925 0.95 0.975"
export MJLAB_VL53L8CX_SQUARE_FOVY_DEG=45.0

LOW_SPEED_DR_CHECKPOINT="logs/steering/local_planner_low_speed_v6_511_final.pt"
LOW_SPEED_ODOM_CHECKPOINT="logs/steering/local_planner_low_speed_v6_511_odom_final.pt"
INDOOR_DR_CHECKPOINT="logs/steering/local_planner_indoor_approach_v6_511_final.pt"
INDOOR_ODOM_CHECKPOINT="logs/steering/local_planner_indoor_approach_v6_511_odom_final.pt"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Error: required file not found: ${path}" >&2
    exit 1
  fi
}

echo "=========================================="
echo "Navigation training + benchmark orchestrator"
echo "Execution order:"
echo "  1. curriculum_train_straight_approach.sh  (train both navigation tasks)"
echo "  2. run_paper_experiments_v6.sh            (${RUN_NAME} formal benchmark)"
echo "VL53L8CX square fovy override: ${MJLAB_VL53L8CX_SQUARE_FOVY_DEG:-default 65° diagonal conversion}"
echo "=========================================="

echo ""
echo "[1/2] Starting curriculum_train_straight_approach.sh"
echo "=========================================="
"${SCRIPT_DIR}/curriculum_train_straight_approach.sh"

echo ""
echo "[1/2] curriculum_train_straight_approach.sh finished"
echo "=========================================="

echo ""
echo "[2/2] Checking local planner checkpoints"
echo "=========================================="
require_file "${LOW_SPEED_DR_CHECKPOINT}"
require_file "${LOW_SPEED_ODOM_CHECKPOINT}"
require_file "${INDOOR_DR_CHECKPOINT}"
require_file "${INDOOR_ODOM_CHECKPOINT}"
echo "Low-speed DR:   ${LOW_SPEED_DR_CHECKPOINT}"
echo "Low-speed odom: ${LOW_SPEED_ODOM_CHECKPOINT}"
echo "Indoor DR:      ${INDOOR_DR_CHECKPOINT}"
echo "Indoor odom:    ${INDOOR_ODOM_CHECKPOINT}"

echo ""
echo "[2/2] Starting the formal v6 benchmark"
echo "=========================================="
"${SCRIPT_DIR}/experiments/run_paper_experiments_v6.sh" \
  --run-name "${RUN_NAME}" \
  --safety-brake-scale-thresholds "${SAFETY_BRAKE_SCALE_THRESHOLDS}" \
  --low-speed-local-planner-checkpoint "${LOW_SPEED_DR_CHECKPOINT}" \
  --low-speed-odom-local-planner-checkpoint "${LOW_SPEED_ODOM_CHECKPOINT}" \
  --indoor-local-planner-checkpoint "${INDOOR_DR_CHECKPOINT}" \
  --indoor-odom-local-planner-checkpoint "${INDOOR_ODOM_CHECKPOINT}"

echo ""
echo "[2/2] Formal v6 benchmark finished"
echo "=========================================="
echo "All pipelines finished"
