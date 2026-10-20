#!/bin/bash
# Curriculum learning training script (five steps, historical Stage numbers 1/2/3/5/6 preserved)
# Stage 1: Flat CircularGoal, G1Rw-Pure reward, command safety envelope, lightweight raycast DR
# Stage 2: Flat obstacle + A* waypoint guidance, G1Rw-Pure reward, resumed from Stage 1
# Stage 3: ObstacleMildRough + A* waypoint guidance, G1Rw-Pure reward, resumed from Stage 2
#          terrain is 70% obstacle + 30% mild heightfield rough, rough difficulty <= 0.16
# Stage 5: Steering LowSpeed StraightApproach (0.5-1.2m/s, d_turn=2.4m), resumed from the Stage 3 velocity checkpoint
# Stage 6: Steering IndoorStraightApproach (0.5-1.2m/s, indoor room environment), resumed from the Stage 5 local planner

set -e  # exit immediately on error

DISABLE_RAYCAST_NEAR_OCCLUSION=false
USE_ODOMETRY=false
ROBOT_VERSION=v15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disable-raycast-near-occlusion)
            DISABLE_RAYCAST_NEAR_OCCLUSION=true
            shift
            ;;
        --use-odometry)
            USE_ODOMETRY=true
            shift
            ;;
        --robot-version)
            ROBOT_VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--disable-raycast-near-occlusion] [--use-odometry] [--robot-version v5|v15]"
            exit 0
            ;;
        *)
            echo "Error: unknown argument $1"
            echo "Usage: $0 [--disable-raycast-near-occlusion] [--use-odometry] [--robot-version v5|v15]"
            exit 1
            ;;
    esac
done

# Export robot version for Python code to pick up.
export G1_ROBOT_VERSION="${ROBOT_VERSION}"

RAYCAST_NEAR_OCCLUSION_ARGS=()
if [[ "${DISABLE_RAYCAST_NEAR_OCCLUSION}" == "true" ]]; then
    RAYCAST_NEAR_OCCLUSION_ARGS=(--disable-raycast-near-occlusion True)
fi

ODOMETRY_ARGS=()
if [[ "${USE_ODOMETRY}" == "true" ]]; then
    ODOMETRY_ARGS=(--use-odometry True --include-joint-vel False)
fi

# ====== Configuration ======
# Stage 1: simple environment training
STAGE1_ENV="Mjlab-CircularGoal-Flat-G1Rw-Pure-v6"
STAGE1_NUM_ENVS=8192
STAGE1_MAX_ITERATIONS=4000

# Stage 2: flat obstacle + A* waypoint guidance training
STAGE2_ENV="Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6"
STAGE2_NUM_ENVS=4096
STAGE2_MAX_ITERATIONS=4000

# Stage 3: mixed obstacles + mildly rough terrain + A* waypoint guidance
# 70% obstacles + 30% mild rough (difficulty<=0.16); rough is a heightfield and adds no extra obstacle geom.
STAGE3_ENV="Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6"
STAGE3_NUM_ENVS=4096
STAGE3_MAX_ITERATIONS=4500

# Stage 5: Steering training - LowSpeed straight approach
STAGE5_TASK="Mjlab-StraightApproach-LowSpeed-G1-v6"
STAGE5_NUM_ENVS=1024
STAGE5_MAX_ITERATIONS=80000
STAGE5_OUTPUT_DIR="logs/steering"
STAGE5_EXPERIMENT_NAME="low_speed_v6_511"

# Stage 6: Indoor straight approach (resumed from the Stage 5 local planner)
# Transfer the Stage 5 low-speed local planner to an indoor environment with room-wall constraints.
STAGE6_TASK="Mjlab-IndoorStraightApproach-G1-v6"
STAGE6_NUM_ENVS=1024
STAGE6_MAX_ITERATIONS=20000
STAGE6_OUTPUT_DIR="logs/steering"
STAGE6_EXPERIMENT_NAME="indoor_approach_v6_511"

# Stage 7 removed: only the open-wall (LowSpeed) and indoor navigation tasks are
# part of this release.

# Three-camera raycast
RAYCAST_OBS_NAMES="raycast raycast_left raycast_right"

# Log directory (adjust to match your task configuration)
LOG_DIR="logs/rsl_rl/g1_velocity_raycast_h1"
# =======================

echo "=========================================="
echo "Curriculum learning training script (five steps, historical Stage numbers 1/2/3/5/6 preserved)"
echo "Stage 1: ${STAGE1_ENV} (Flat CircularGoal, G1Rw-Pure, max speed 1.5)"
echo "Stage 2: ${STAGE2_ENV} (Flat obstacle + A*, G1Rw-Pure, max speed 1.5)"
echo "Stage 3: ${STAGE3_ENV} (70% obstacle + 30% mild rough + A*, max speed 1.2)"
echo "Stage 5: ${STAGE5_TASK} (LowSpeed Steering, 0.5-1.2m/s, resumed from Stage 3)"
echo "Stage 6: ${STAGE6_TASK} (Indoor Steering, 0.5-1.2m/s, resumed from Stage 5)"
echo "raycast close-range random occlusion: $([[ "${DISABLE_RAYCAST_NEAR_OCCLUSION}" == "true" ]] && echo "disabled" || echo "lightweight, enabled by default")"
echo "Robot version: ${G1_ROBOT_VERSION}"
echo "=========================================="

# Record the start time, used to find the newly created training directory
START_TIME=$(date +%Y-%m-%d_%H-%M-%S)

echo ""
echo "[Stage 1] Starting training for ${STAGE1_ENV}..."
echo "Max iterations: ${STAGE1_MAX_ITERATIONS}"
echo "=========================================="

# Run Stage 1 training
uv run train ${STAGE1_ENV} \
    --env.scene.num-envs ${STAGE1_NUM_ENVS} \
    --agent.max-iterations ${STAGE1_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}"

echo ""
echo "[Stage 1] Training complete!"
echo "=========================================="

# Find the most recent training directory
LATEST_RUN=$(ls -td ${LOG_DIR}/*/ 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "Error: could not find the training directory ${LOG_DIR}"
    exit 1
fi

# Extract the directory name (strip the path prefix and the trailing slash)
RUN_NAME=$(basename "$LATEST_RUN")

echo "Found latest training directory: ${RUN_NAME}"

# Find the latest checkpoint (sort numerically and pick the largest)
LATEST_CHECKPOINT=$(ls ${LATEST_RUN}/model_*.pt 2>/dev/null | \
    sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | \
    sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: no checkpoint file found in ${LATEST_RUN}"
    exit 1
fi

CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")

echo "Found latest checkpoint: ${CHECKPOINT_NAME}"
echo ""
echo "=========================================="
echo "[Stage 2] Starting training for ${STAGE2_ENV}..."
echo "Resuming from ${RUN_NAME}/${CHECKPOINT_NAME}"
echo "Max iterations: ${STAGE2_MAX_ITERATIONS}"
echo "=========================================="

# Run Stage 2 training
uv run train ${STAGE2_ENV} \
    --agent.resume True \
    --agent.load-run "${RUN_NAME}" \
    --agent.load-checkpoint "${CHECKPOINT_NAME}" \
    --env.scene.num-envs ${STAGE2_NUM_ENVS} \
    --agent.max-iterations ${STAGE2_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}"

echo ""
echo "=========================================="
echo "[Stage 2] Training complete!"
echo "=========================================="

# Find the most recent Stage 2 training directory
STAGE2_LATEST_RUN=$(ls -td ${LOG_DIR}/*/ 2>/dev/null | head -1)
if [ -z "$STAGE2_LATEST_RUN" ]; then
    echo "Error: could not find the Stage 2 training directory ${LOG_DIR}"
    exit 1
fi

STAGE2_RUN_NAME=$(basename "$STAGE2_LATEST_RUN")
echo "Found latest Stage 2 training directory: ${STAGE2_RUN_NAME}"

# Find the latest Stage 2 checkpoint
STAGE2_LATEST_CHECKPOINT=$(ls ${STAGE2_LATEST_RUN}/model_*.pt 2>/dev/null | \
    sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | \
    sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$STAGE2_LATEST_CHECKPOINT" ]; then
    echo "Error: no checkpoint file found in ${STAGE2_LATEST_RUN}"
    exit 1
fi

echo "Found latest Stage 2 checkpoint: ${STAGE2_LATEST_CHECKPOINT}"
echo ""
echo "=========================================="
echo "[Stage 3] Starting training for ${STAGE3_ENV}..."
echo "Resuming from ${STAGE2_LATEST_CHECKPOINT}"
echo "Max iterations: ${STAGE3_MAX_ITERATIONS}"
echo "=========================================="

# Run Stage 3 mixed-obstacle + mild-rough training
uv run train ${STAGE3_ENV} \
    --agent.resume True \
    --agent.load-run "${STAGE2_RUN_NAME}" \
    --agent.load-checkpoint "$(basename "${STAGE2_LATEST_CHECKPOINT}")" \
    --env.scene.num-envs ${STAGE3_NUM_ENVS} \
    --agent.max-iterations ${STAGE3_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}"

echo ""
echo "=========================================="
echo "[Stage 3] Training complete!"
echo "=========================================="

# Find the most recent Stage 3 training directory
STAGE3_LATEST_RUN=$(ls -td ${LOG_DIR}/*/ 2>/dev/null | head -1)
if [ -z "$STAGE3_LATEST_RUN" ]; then
    echo "Error: could not find the Stage 3 training directory ${LOG_DIR}"
    exit 1
fi

STAGE3_RUN_NAME=$(basename "$STAGE3_LATEST_RUN")
echo "Found latest Stage 3 training directory: ${STAGE3_RUN_NAME}"

# Find the latest Stage 3 checkpoint
STAGE3_LATEST_CHECKPOINT=$(ls ${STAGE3_LATEST_RUN}/model_*.pt 2>/dev/null | \
    sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | \
    sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$STAGE3_LATEST_CHECKPOINT" ]; then
    echo "Error: no checkpoint file found in ${STAGE3_LATEST_RUN}"
    exit 1
fi

echo "Found latest Stage 3 checkpoint: ${STAGE3_LATEST_CHECKPOINT}"

# ==============================================================
# Stage 5: LowSpeed straight approach (resumed from the Stage 3 base checkpoint)
# ==============================================================
echo ""
echo "=========================================="
echo "[Stage 5] Starting LowSpeed Steering training ${STAGE5_TASK}..."
echo "Base velocity checkpoint: ${STAGE3_LATEST_CHECKPOINT}"
echo "Max iterations: ${STAGE5_MAX_ITERATIONS}"
echo "=========================================="

uv run python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE5_TASK}" \
    --checkpoint "${STAGE3_LATEST_CHECKPOINT}" \
    --num_envs ${STAGE5_NUM_ENVS} \
    --output_dir "${STAGE5_OUTPUT_DIR}" \
    --experiment_name "${STAGE5_EXPERIMENT_NAME}" \
    --raycast-obs-names ${RAYCAST_OBS_NAMES} \
    --max_iterations ${STAGE5_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}" \
    "${ODOMETRY_ARGS[@]}"

echo ""
echo "[Stage 5] LowSpeed Steering training complete!"
echo "=========================================="

# ==============================================================
# Find the best Stage 5 local planner checkpoint.
# Prefer _final; otherwise take the most recently modified file with the same prefix.
STAGE5_FINAL_CKPT="${STAGE5_OUTPUT_DIR}/local_planner_${STAGE5_EXPERIMENT_NAME}_final.pt"
if [ -f "${STAGE5_FINAL_CKPT}" ]; then
    STAGE5_LOCAL_PLANNER_CHECKPOINT="${STAGE5_FINAL_CKPT}"
else
    STAGE5_LOCAL_PLANNER_CHECKPOINT=$(ls -t "${STAGE5_OUTPUT_DIR}"/local_planner_"${STAGE5_EXPERIMENT_NAME}"*.pt 2>/dev/null | head -1)
fi

if [ -z "${STAGE5_LOCAL_PLANNER_CHECKPOINT}" ]; then
    echo "Error: could not find a Stage 5 local planner checkpoint in ${STAGE5_OUTPUT_DIR} (local_planner_${STAGE5_EXPERIMENT_NAME}*.pt)"
    exit 1
fi

echo "Found Stage 5 local planner checkpoint: ${STAGE5_LOCAL_PLANNER_CHECKPOINT}"

# ==============================================================
# Stage 6: Indoor straight approach (resumed from the Stage 5 local planner)
# ==============================================================
echo ""
echo "=========================================="
echo "[Stage 6] Starting Indoor Steering training ${STAGE6_TASK}..."
echo "Base velocity checkpoint: ${STAGE3_LATEST_CHECKPOINT}"
echo "Local planner checkpoint: ${STAGE5_LOCAL_PLANNER_CHECKPOINT}"
echo "Max iterations: ${STAGE6_MAX_ITERATIONS}"
echo "=========================================="

uv run python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE6_TASK}" \
    --checkpoint "${STAGE3_LATEST_CHECKPOINT}" \
    --local_planner_checkpoint "${STAGE5_LOCAL_PLANNER_CHECKPOINT}" \
    --num_envs ${STAGE6_NUM_ENVS} \
    --output_dir "${STAGE6_OUTPUT_DIR}" \
    --experiment_name "${STAGE6_EXPERIMENT_NAME}" \
    --raycast-obs-names ${RAYCAST_OBS_NAMES} \
    --max_iterations ${STAGE6_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}" \
    "${ODOMETRY_ARGS[@]}"

echo ""
echo "[Stage 6] Indoor Steering training complete!"
echo "=========================================="

echo ""
echo "=========================================="
echo "All curriculum learning training complete!"
echo ""
echo "Output directories:"
echo "  Stage 3 velocity checkpoint: ${STAGE3_LATEST_CHECKPOINT}"
echo "  LowSpeed:  ${STAGE5_OUTPUT_DIR}/local_planner_${STAGE5_EXPERIMENT_NAME}_final.pt"
echo "  Indoor:    ${STAGE6_OUTPUT_DIR}/local_planner_${STAGE6_EXPERIMENT_NAME}_final.pt"
echo "=========================================="
