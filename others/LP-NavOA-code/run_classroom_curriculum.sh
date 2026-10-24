#!/bin/bash
# LP-NavOA full curriculum for the PAC-MAN classroom comparison.
#
# Mirrors src/mjlab/tool/curriculum_train_v6.sh exactly (stages 1/2/3 backbone,
# stages 5/6 native navigation planners, official env counts and iterations),
# then adds a Stage 7: local-planner distillation on Mjlab-Classroom-G1-v6,
# warm-started from the Stage 6 indoor planner.
#
# Differences from the upstream script are mechanical only:
#   - `uv run --no-sync` + PYTHONPATH, using the local locked environment.
#
# Usage: bash run_classroom_curriculum.sh [--robot-version v5|v15]

set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src"
export UV_PROJECT_ENVIRONMENT="${LP_VENV:-$(pwd)/.venv-stage67}"
UV_RUN="uv run --no-sync"

ROBOT_VERSION=v15
while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-version)
            ROBOT_VERSION="$2"
            shift 2
            ;;
        *)
            echo "Error: unknown argument $1"
            exit 1
            ;;
    esac
done
export G1_ROBOT_VERSION="${ROBOT_VERSION}"

# ====== Configuration (identical to curriculum_train_v6.sh) ======
STAGE1_ENV="Mjlab-CircularGoal-Flat-G1Rw-Pure-v6"
STAGE1_NUM_ENVS=8192
STAGE1_MAX_ITERATIONS=4000

STAGE2_ENV="Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6"
STAGE2_NUM_ENVS=4096
STAGE2_MAX_ITERATIONS=4000

STAGE3_ENV="Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6"
STAGE3_NUM_ENVS=4096
STAGE3_MAX_ITERATIONS=4500

STAGE5_TASK="Mjlab-StraightApproach-LowSpeed-G1-v6"
STAGE5_NUM_ENVS=1024
STAGE5_MAX_ITERATIONS=80000
STAGE5_OUTPUT_DIR="logs/steering"
STAGE5_EXPERIMENT_NAME="low_speed_v6_511"

STAGE6_TASK="Mjlab-IndoorStraightApproach-G1-v6"
STAGE6_NUM_ENVS=1024
STAGE6_MAX_ITERATIONS=20000
STAGE6_OUTPUT_DIR="logs/steering"
STAGE6_EXPERIMENT_NAME="indoor_approach_v6_511"

# Stage 7 (added): classroom distillation, warm-started from the Stage 6 planner.
STAGE7_TASK="Mjlab-Classroom-G1-v6"
STAGE7_NUM_ENVS=1024
STAGE7_MAX_ITERATIONS=20000
STAGE7_OUTPUT_DIR="logs/steering"
STAGE7_EXPERIMENT_NAME="classroom_v6_511"

RAYCAST_OBS_NAMES='("raycast", "raycast_left", "raycast_right")'
LOG_DIR="logs/rsl_rl/g1_velocity_raycast_h1"
# =======================

latest_checkpoint() {
    # $1 = run dir; prints the model_*.pt with the largest iteration number.
    ls "$1"/model_*.pt 2>/dev/null | \
        sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | \
        sort -rn | head -1 | cut -d' ' -f2
}

echo "=========================================="
echo "LP-NavOA full curriculum + classroom stage"
echo "Stage 1: ${STAGE1_ENV} (${STAGE1_NUM_ENVS} envs, ${STAGE1_MAX_ITERATIONS} iters)"
echo "Stage 2: ${STAGE2_ENV} (${STAGE2_NUM_ENVS} envs, ${STAGE2_MAX_ITERATIONS} iters)"
echo "Stage 3: ${STAGE3_ENV} (${STAGE3_NUM_ENVS} envs, ${STAGE3_MAX_ITERATIONS} iters)"
echo "Stage 5: ${STAGE5_TASK} (${STAGE5_NUM_ENVS} envs, ${STAGE5_MAX_ITERATIONS} iters)"
echo "Stage 6: ${STAGE6_TASK} (${STAGE6_NUM_ENVS} envs, ${STAGE6_MAX_ITERATIONS} iters)"
echo "Stage 7: ${STAGE7_TASK} (${STAGE7_NUM_ENVS} envs, ${STAGE7_MAX_ITERATIONS} iters)"
echo "Robot version: ${G1_ROBOT_VERSION}"
echo "=========================================="

# ---------- Stage 1 ----------
echo "[Stage 1] ${STAGE1_ENV}"
${UV_RUN} train ${STAGE1_ENV} \
    --env.scene.num-envs ${STAGE1_NUM_ENVS} \
    --agent.max-iterations ${STAGE1_MAX_ITERATIONS}

RUN_NAME=$(basename "$(ls -td ${LOG_DIR}/*/ | head -1)")
CKPT=$(latest_checkpoint "${LOG_DIR}/${RUN_NAME}")
echo "[Stage 1] done: ${RUN_NAME}/$(basename "${CKPT}")"

# ---------- Stage 2 ----------
echo "[Stage 2] ${STAGE2_ENV} (resume ${RUN_NAME})"
${UV_RUN} train ${STAGE2_ENV} \
    --agent.resume True \
    --agent.load-run "${RUN_NAME}" \
    --agent.load-checkpoint "$(basename "${CKPT}")" \
    --env.scene.num-envs ${STAGE2_NUM_ENVS} \
    --agent.max-iterations ${STAGE2_MAX_ITERATIONS}

RUN_NAME=$(basename "$(ls -td ${LOG_DIR}/*/ | head -1)")
CKPT=$(latest_checkpoint "${LOG_DIR}/${RUN_NAME}")
echo "[Stage 2] done: ${RUN_NAME}/$(basename "${CKPT}")"

# ---------- Stage 3 ----------
echo "[Stage 3] ${STAGE3_ENV} (resume ${RUN_NAME})"
${UV_RUN} train ${STAGE3_ENV} \
    --agent.resume True \
    --agent.load-run "${RUN_NAME}" \
    --agent.load-checkpoint "$(basename "${CKPT}")" \
    --env.scene.num-envs ${STAGE3_NUM_ENVS} \
    --agent.max-iterations ${STAGE3_MAX_ITERATIONS}

RUN_NAME=$(basename "$(ls -td ${LOG_DIR}/*/ | head -1)")
CKPT=$(latest_checkpoint "${LOG_DIR}/${RUN_NAME}")
BACKBONE_CKPT="${LOG_DIR}/${RUN_NAME}/$(basename "${CKPT}")"
echo "[Stage 3] backbone done: ${BACKBONE_CKPT}"

# ---------- Stage 5 ----------
echo "[Stage 5] ${STAGE5_TASK}"
${UV_RUN} python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE5_TASK}" \
    --checkpoint "${BACKBONE_CKPT}" \
    --num_envs ${STAGE5_NUM_ENVS} \
    --output_dir "${STAGE5_OUTPUT_DIR}" \
    --experiment_name "${STAGE5_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE5_MAX_ITERATIONS}

LP_CKPT="${STAGE5_OUTPUT_DIR}/local_planner_${STAGE5_EXPERIMENT_NAME}_final.pt"
[ -f "${LP_CKPT}" ] || LP_CKPT=$(ls -t "${STAGE5_OUTPUT_DIR}"/local_planner_"${STAGE5_EXPERIMENT_NAME}"*.pt | head -1)
echo "[Stage 5] done: ${LP_CKPT}"

# ---------- Stage 6 ----------
echo "[Stage 6] ${STAGE6_TASK}"
${UV_RUN} python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE6_TASK}" \
    --checkpoint "${BACKBONE_CKPT}" \
    --local_planner_checkpoint "${LP_CKPT}" \
    --num_envs ${STAGE6_NUM_ENVS} \
    --output_dir "${STAGE6_OUTPUT_DIR}" \
    --experiment_name "${STAGE6_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE6_MAX_ITERATIONS}

LP_CKPT="${STAGE6_OUTPUT_DIR}/local_planner_${STAGE6_EXPERIMENT_NAME}_final.pt"
[ -f "${LP_CKPT}" ] || LP_CKPT=$(ls -t "${STAGE6_OUTPUT_DIR}"/local_planner_"${STAGE6_EXPERIMENT_NAME}"*.pt | head -1)
echo "[Stage 6] done: ${LP_CKPT}"

# ---------- Stage 7: classroom ----------
echo "[Stage 7] ${STAGE7_TASK} (warm start from ${LP_CKPT})"
${UV_RUN} python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE7_TASK}" \
    --checkpoint "${BACKBONE_CKPT}" \
    --local_planner_checkpoint "${LP_CKPT}" \
    --num_envs ${STAGE7_NUM_ENVS} \
    --output_dir "${STAGE7_OUTPUT_DIR}" \
    --experiment_name "${STAGE7_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE7_MAX_ITERATIONS}

echo ""
echo "=========================================="
echo "All stages complete!"
echo "  Backbone:          ${BACKBONE_CKPT}"
echo "  Classroom planner: ${STAGE7_OUTPUT_DIR}/local_planner_${STAGE7_EXPERIMENT_NAME}_final.pt"
echo "=========================================="
