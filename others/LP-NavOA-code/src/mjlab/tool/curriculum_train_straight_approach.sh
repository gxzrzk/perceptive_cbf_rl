#!/bin/bash
# Straight-Approach Steering curriculum training script
#
# Starting from an existing locomotion checkpoint, train the local planner steering network in four stages:
#
# Non-odometry mode (Stages 1-2):
#   Stage 1: LowSpeed straight approach (0.5-1.2 m/s, d_turn=2.4m, d_safe=1.2m)
#   Stage 2: Indoor straight approach (0.5-1.2 m/s, d_turn=2.4m, d_safe=1.2m)
#            Continue training from the Stage 1 local planner checkpoint
#
# Odometry mode (Stages 3-4):
#   Stage 3: LowSpeed straight approach (odometry mode)
#   Stage 4: Indoor straight approach (odometry mode, continued from Stage 3)
#
# Design rationale:
#   - LowSpeed → Indoor is a continuous curriculum: first practice low-speed obstacle avoidance in a simple single-wall scene,
#     then transfer to an indoor environment constrained by room walls, with the speed slightly raised to 1.5 m/s.
#   - Odometry mode and non-odometry mode have different observation spaces, so each trains its own local planner independently.

set -e

START_STAGE=1
LOCAL_PLANNER_MODE=both
DISABLE_RAYCAST_NEAR_OCCLUSION=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)
      if [[ $# -lt 2 ]]; then
        echo "Error: --start-stage requires a stage number (1-4)"
        exit 1
      fi
      START_STAGE="$2"
      shift 2
      ;;
    --local-planner-mode)
      if [[ $# -lt 2 ]]; then
        echo "Error: --local-planner-mode requires dr, odom, or both"
        exit 1
      fi
      LOCAL_PLANNER_MODE="$2"
      shift 2
      ;;
    --disable-raycast-near-occlusion)
      DISABLE_RAYCAST_NEAR_OCCLUSION=true
      shift
      ;;
    --enable-raycast-near-occlusion)
      DISABLE_RAYCAST_NEAR_OCCLUSION=false
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--start-stage N] [--local-planner-mode dr|odom|both] [--enable-raycast-near-occlusion]"
      echo "  --start-stage N: start from stage N, N can be 1-4, default 1"
      echo "  --local-planner-mode: dr runs only Stages 1-2; odom runs only Stages 3-4; both runs all, default both"
      echo "Raycast near-range random occlusion is disabled by default; pass --enable-raycast-near-occlusion to enable it"
      echo "odom mode passes --use-odometry True --include-joint-vel False"
      exit 0
      ;;
    *)
      echo "Error: unknown argument $1"
      echo "Usage: $0 [--start-stage N] [--local-planner-mode dr|odom|both] [--enable-raycast-near-occlusion]"
      exit 1
      ;;
  esac
done

if ! [[ "${START_STAGE}" =~ ^[1-4]$ ]]; then
  echo "Error: --start-stage must be 1, 2, 3, or 4, current value: ${START_STAGE}"
  exit 1
fi

if [[ "${LOCAL_PLANNER_MODE}" != "dr" && "${LOCAL_PLANNER_MODE}" != "odom" && "${LOCAL_PLANNER_MODE}" != "both" ]]; then
  echo "Error: --local-planner-mode must be dr, odom, or both, current value: ${LOCAL_PLANNER_MODE}"
  exit 1
fi

RUN_DR=false
RUN_ODOM=false
if [[ "${LOCAL_PLANNER_MODE}" == "dr" || "${LOCAL_PLANNER_MODE}" == "both" ]]; then
  RUN_DR=true
fi
if [[ "${LOCAL_PLANNER_MODE}" == "odom" || "${LOCAL_PLANNER_MODE}" == "both" ]]; then
  RUN_ODOM=true
fi

RAYCAST_NEAR_OCCLUSION_ARGS=()
if [[ "${DISABLE_RAYCAST_NEAR_OCCLUSION}" == "true" ]]; then
  RAYCAST_NEAR_OCCLUSION_ARGS=(--disable-raycast-near-occlusion True)
  RAYCAST_NEAR_OCCLUSION_STATUS="disabled"
else
  RAYCAST_NEAR_OCCLUSION_STATUS="enabled by default"
fi

ODOMETRY_ARGS=(--use-odometry True --include-joint-vel False)
ONNX_OUTPUT_DIR="logs/steering/onnx"

find_local_planner_checkpoint() {
  local stage_name="$1"
  local output_dir="$2"
  local experiment_name="$3"
  local final_ckpt="${output_dir}/local_planner_${experiment_name}_final.pt"
  local checkpoint=""

  if [ -f "${final_ckpt}" ]; then
    checkpoint="${final_ckpt}"
  else
    checkpoint=$(ls -t "${output_dir}"/local_planner_"${experiment_name}"*.pt 2>/dev/null | head -1)
  fi

  if [ -z "${checkpoint}" ]; then
    echo "Error: could not find ${stage_name} local planner checkpoint (local_planner_${experiment_name}*.pt) in ${output_dir}" >&2
    exit 1
  fi

  echo "${checkpoint}"
}

export_final_local_planner_onnx() {
  local stage_name="$1"
  local output_dir="$2"
  local experiment_name="$3"
  local final_ckpt="${output_dir}/local_planner_${experiment_name}_final.pt"
  local onnx_path="${ONNX_OUTPUT_DIR}/local_planner_${experiment_name}_final.onnx"

  if [ ! -f "${final_ckpt}" ]; then
    echo "Error: ${stage_name} final checkpoint does not exist, cannot export ONNX: ${final_ckpt}" >&2
    exit 1
  fi

  mkdir -p "${ONNX_OUTPUT_DIR}"
  echo ""
  echo "[${stage_name}] Exporting local planner ONNX..."
  echo "Checkpoint: ${final_ckpt}"
  echo "ONNX: ${onnx_path}"
  uv run python -m mjlab.tasks.steering.export_local_planner_to_onnx \
    --checkpoint "${final_ckpt}" \
    --output "${onnx_path}"
  echo "[${stage_name}] ONNX export complete: ${onnx_path}"
}

# ====== Base Locomotion Checkpoint ======
LOCOMOTION_CHECKPOINT="logs/rsl_rl/g1_velocity_raycast_h1/2026-04-26_06-57-30/model_12497.pt"

# ====== Non-odometry mode: Stages 1-2 ======
# Stage 1: LowSpeed straight approach
STAGE1_TASK="Mjlab-StraightApproach-LowSpeed-G1-v6"
STAGE1_NUM_ENVS=1024
STAGE1_MAX_ITERATIONS=100000
STAGE1_OUTPUT_DIR="logs/steering"
STAGE1_EXPERIMENT_NAME="low_speed_v6_511"

# Stage 2: Indoor straight approach (continued from Stage 1)
STAGE2_TASK="Mjlab-IndoorStraightApproach-G1-v6"
STAGE2_NUM_ENVS=1024
STAGE2_MAX_ITERATIONS=100000
STAGE2_OUTPUT_DIR="logs/steering"
STAGE2_EXPERIMENT_NAME="indoor_approach_v6_511"

# ====== Odometry mode: Stages 3-4 ======
# Stage 3: LowSpeed straight approach (odometry mode)
STAGE3_TASK="Mjlab-StraightApproach-LowSpeed-G1-v6"
STAGE3_NUM_ENVS=1024
STAGE3_MAX_ITERATIONS=100000
STAGE3_OUTPUT_DIR="logs/steering"
STAGE3_EXPERIMENT_NAME="low_speed_v6_511_odom"

# Stage 4: Indoor straight approach (odometry mode, continued from Stage 3)
STAGE4_TASK="Mjlab-IndoorStraightApproach-G1-v6"
STAGE4_NUM_ENVS=1024
STAGE4_MAX_ITERATIONS=100000
STAGE4_OUTPUT_DIR="logs/steering"
STAGE4_EXPERIMENT_NAME="indoor_approach_v6_511_odom"

# Three-camera raycast. mjlab.TYRO_FLAGS requires tuples to use Python literal syntax.
RAYCAST_OBS_NAMES="('raycast','raycast_left','raycast_right')"
# =======================

if [ ! -f "${LOCOMOTION_CHECKPOINT}" ]; then
  echo "Error: locomotion checkpoint does not exist: ${LOCOMOTION_CHECKPOINT}"
  exit 1
fi

echo "=========================================="
echo "Straight-Approach Steering curriculum training (4 stages)"
echo "Locomotion: ${LOCOMOTION_CHECKPOINT}"
echo ""
echo "Non-odometry mode:"
echo "  Stage 1: ${STAGE1_TASK} (LowSpeed 0.5-1.2 m/s)"
echo "  Stage 2: ${STAGE2_TASK} (Indoor, continued from Stage 1)"
echo "Odometry mode:"
echo "  Stage 3: ${STAGE3_TASK} (LowSpeed 0.5-1.2 m/s)"
echo "  Stage 4: ${STAGE4_TASK} (Indoor, continued from Stage 3)"
echo "Starting stage: Stage ${START_STAGE}"
echo "Local planner mode: ${LOCAL_PLANNER_MODE}"
echo "Odometry arguments: --use-odometry True --include-joint-vel False"
echo "Raycast near-range random occlusion: ${RAYCAST_NEAR_OCCLUSION_STATUS}"
echo "=========================================="

# ==============================================================
# Stage 1: LowSpeed straight approach (non-odometry)
# ==============================================================
if [[ "${RUN_DR}" == "true" && "${START_STAGE}" -le 1 ]]; then
  echo ""
  echo "[Stage 1] Starting LowSpeed Steering training (non-odometry)..."
  echo "Task: ${STAGE1_TASK}"
  echo "Max iterations: ${STAGE1_MAX_ITERATIONS}"
  echo "=========================================="

  uv run python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE1_TASK}" \
    --checkpoint "${LOCOMOTION_CHECKPOINT}" \
    --num_envs ${STAGE1_NUM_ENVS} \
    --output_dir "${STAGE1_OUTPUT_DIR}" \
    --experiment_name "${STAGE1_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE1_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}"

  echo ""
  echo "[Stage 1] LowSpeed Steering training complete!"
  echo "=========================================="
  export_final_local_planner_onnx "Stage 1" "${STAGE1_OUTPUT_DIR}" "${STAGE1_EXPERIMENT_NAME}"
else
  echo ""
  echo "[Stage 1] Skipped (mode=${LOCAL_PLANNER_MODE}, starting stage is Stage ${START_STAGE})"
fi

# Find the best Stage 1 local planner checkpoint
# train_local_planner.py saves checkpoints under output_dir, with the following filename rules:
#   local_planner_{experiment_name}_final.pt   (finished normally)
#   local_planner_{experiment_name}_step{N}.pt (periodic save)
#   local_planner_{experiment_name}_interrupted_step{N}.pt (when interrupted)
# Prefer _final, otherwise take the most recently modified file with the same prefix
if [[ "${RUN_DR}" == "true" && "${START_STAGE}" -le 2 ]]; then
  STAGE1_LOCAL_PLANNER_CHECKPOINT=$(find_local_planner_checkpoint "Stage 1" "${STAGE1_OUTPUT_DIR}" "${STAGE1_EXPERIMENT_NAME}")
  echo "Found Stage 1 local planner checkpoint: ${STAGE1_LOCAL_PLANNER_CHECKPOINT}"
fi

# ==============================================================
# Stage 2: Indoor straight approach (non-odometry, continued from the Stage 1 local planner)
# ==============================================================
if [[ "${RUN_DR}" == "true" && "${START_STAGE}" -le 2 ]]; then
  echo ""
  echo "[Stage 2] Starting Indoor Steering training (non-odometry)..."
  echo "Task: ${STAGE2_TASK}"
  echo "Local planner checkpoint: ${STAGE1_LOCAL_PLANNER_CHECKPOINT}"
  echo "Max iterations: ${STAGE2_MAX_ITERATIONS}"
  echo "=========================================="

  uv run python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE2_TASK}" \
    --checkpoint "${LOCOMOTION_CHECKPOINT}" \
    --local_planner_checkpoint "${STAGE1_LOCAL_PLANNER_CHECKPOINT}" \
    --num_envs ${STAGE2_NUM_ENVS} \
    --output_dir "${STAGE2_OUTPUT_DIR}" \
    --experiment_name "${STAGE2_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE2_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}"

  echo ""
  echo "[Stage 2] Indoor Steering training complete!"
  echo "=========================================="
  export_final_local_planner_onnx "Stage 2" "${STAGE2_OUTPUT_DIR}" "${STAGE2_EXPERIMENT_NAME}"
else
  echo ""
  echo "[Stage 2] Skipped (mode=${LOCAL_PLANNER_MODE}, starting stage is Stage ${START_STAGE})"
fi

# ==============================================================
# Stage 3: LowSpeed straight approach (odometry mode)
# ==============================================================
if [[ "${RUN_ODOM}" == "true" && "${START_STAGE}" -le 3 ]]; then
  echo ""
  echo "[Stage 3] Starting LowSpeed Steering training (odometry mode)..."
  echo "Task: ${STAGE3_TASK}"
  echo "Max iterations: ${STAGE3_MAX_ITERATIONS}"
  echo "=========================================="

  uv run python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE3_TASK}" \
    --checkpoint "${LOCOMOTION_CHECKPOINT}" \
    --num_envs ${STAGE3_NUM_ENVS} \
    --output_dir "${STAGE3_OUTPUT_DIR}" \
    --experiment_name "${STAGE3_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE3_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}" \
    "${ODOMETRY_ARGS[@]}"

  echo ""
  echo "[Stage 3] LowSpeed Steering training complete!"
  echo "=========================================="
  export_final_local_planner_onnx "Stage 3" "${STAGE3_OUTPUT_DIR}" "${STAGE3_EXPERIMENT_NAME}"
else
  echo ""
  echo "[Stage 3] Skipped (mode=${LOCAL_PLANNER_MODE}, starting stage is Stage ${START_STAGE})"
fi

# Find the best Stage 3 local planner checkpoint
if [[ "${RUN_ODOM}" == "true" ]]; then
  STAGE3_LOCAL_PLANNER_CHECKPOINT=$(find_local_planner_checkpoint "Stage 3" "${STAGE3_OUTPUT_DIR}" "${STAGE3_EXPERIMENT_NAME}")
  echo "Found Stage 3 local planner checkpoint: ${STAGE3_LOCAL_PLANNER_CHECKPOINT}"
fi

# ==============================================================
# Stage 4: Indoor straight approach (odometry mode, continued from the Stage 3 local planner)
# ==============================================================
if [[ "${RUN_ODOM}" == "true" && "${START_STAGE}" -le 4 ]]; then
  echo ""
  echo "[Stage 4] Starting Indoor Steering training (odometry mode)..."
  echo "Task: ${STAGE4_TASK}"
  echo "Local planner checkpoint: ${STAGE3_LOCAL_PLANNER_CHECKPOINT}"
  echo "Max iterations: ${STAGE4_MAX_ITERATIONS}"
  echo "=========================================="

  uv run python -m mjlab.tasks.steering.train_local_planner \
    --mode train \
    --task "${STAGE4_TASK}" \
    --checkpoint "${LOCOMOTION_CHECKPOINT}" \
    --local_planner_checkpoint "${STAGE3_LOCAL_PLANNER_CHECKPOINT}" \
    --num_envs ${STAGE4_NUM_ENVS} \
    --output_dir "${STAGE4_OUTPUT_DIR}" \
    --experiment_name "${STAGE4_EXPERIMENT_NAME}" \
    --raycast-obs-names "${RAYCAST_OBS_NAMES}" \
    --max_iterations ${STAGE4_MAX_ITERATIONS} \
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}" \
    "${ODOMETRY_ARGS[@]}"

  echo ""
  echo "[Stage 4] Indoor Steering training complete!"
  echo "=========================================="
  export_final_local_planner_onnx "Stage 4" "${STAGE4_OUTPUT_DIR}" "${STAGE4_EXPERIMENT_NAME}"
else
  echo ""
  echo "[Stage 4] Skipped (mode=${LOCAL_PLANNER_MODE}, starting stage is Stage ${START_STAGE})"
fi

echo ""
echo "=========================================="
echo "All Straight-Approach Steering training complete!"
echo ""
echo "Output directories (non-odometry):"
echo "  LowSpeed:  ${STAGE1_OUTPUT_DIR}/local_planner_${STAGE1_EXPERIMENT_NAME}_final.pt"
echo "  Indoor:    ${STAGE2_OUTPUT_DIR}/local_planner_${STAGE2_EXPERIMENT_NAME}_final.pt"
echo "Output directories (odometry mode):"
echo "  LowSpeed:  ${STAGE3_OUTPUT_DIR}/local_planner_${STAGE3_EXPERIMENT_NAME}_final.pt"
echo "  Indoor:    ${STAGE4_OUTPUT_DIR}/local_planner_${STAGE4_EXPERIMENT_NAME}_final.pt"
echo "ONNX output directory:"
echo "  ${ONNX_OUTPUT_DIR}"
echo "=========================================="
