#!/bin/bash
# Navigation local-planner ablation training for paper v6 experiments.
#
# Default behavior trains the low-speed ablation checkpoints used by the paper:
#   - DR/no-odometry mode: waypoint_only, key_goal_only, no_teacher_forcing,
#     no_dr_curriculum
#   - Odometry mode: waypoint_only, key_goal_only
#
# Indoor ablation training is available with --task-scope all; the ablation
# table only needs the low-speed (open-wall) setting.

set -euo pipefail

VARIANT="all"
MODE="both"
TASK_SCOPE="low_speed"
DISABLE_RAYCAST_NEAR_OCCLUSION=false
ROBOT_VERSION="v15"
LOW_SPEED_ITERS=80000
INDOOR_ITERS=20000
NUM_ENVS=1024

usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "options:"
  echo "  --variant all|waypoint_only|key_goal_only|no_teacher_forcing|no_dr_curriculum"
  echo "  --mode dr|odom|both"
  echo "  --odometry                       equivalent to --mode odom"
  echo "  --task-scope low_speed|all       default low_speed (all = add indoor)"
  echo "  --disable-raycast-near-occlusion"
  echo "  --robot-version v5|v15"
  echo "  --low-speed-iters N"
  echo "  --indoor-iters N"
  echo "  --num-envs N"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --odometry)
      MODE="odom"
      shift
      ;;
    --task-scope)
      TASK_SCOPE="$2"
      shift 2
      ;;
    --disable-raycast-near-occlusion)
      DISABLE_RAYCAST_NEAR_OCCLUSION=true
      shift
      ;;
    --robot-version)
      ROBOT_VERSION="$2"
      shift 2
      ;;
    --low-speed-iters)
      LOW_SPEED_ITERS="$2"
      shift 2
      ;;
    --indoor-iters)
      INDOOR_ITERS="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument $1"
      usage
      exit 1
      ;;
  esac
done

case "${VARIANT}" in
  all|waypoint_only|key_goal_only|no_teacher_forcing|no_dr_curriculum)
    ;;
  *)
    echo "Error: --variant must be one of all, waypoint_only, key_goal_only, no_teacher_forcing, no_dr_curriculum"
    exit 1
    ;;
esac

case "${MODE}" in
  dr|odom|both)
    ;;
  *)
    echo "Error: --mode must be one of dr, odom, both"
    exit 1
    ;;
esac

case "${TASK_SCOPE}" in
  low_speed|all)
    ;;
  *)
    echo "Error: --task-scope must be one of low_speed, all"
    exit 1
    ;;
esac

export G1_ROBOT_VERSION="${ROBOT_VERSION}"

LOCOMOTION_CHECKPOINT="logs/rsl_rl/g1_velocity_raycast_h1/2026-04-26_06-57-30/model_12497.pt"
OUTPUT_DIR="logs/steering"
RAYCAST_OBS_NAMES="raycast raycast_left raycast_right"

LOW_SPEED_TASK="Mjlab-StraightApproach-LowSpeed-G1-v6"
INDOOR_TASK="Mjlab-IndoorStraightApproach-G1-v6"

RAYCAST_NEAR_OCCLUSION_ARGS=()
if [[ "${DISABLE_RAYCAST_NEAR_OCCLUSION}" == "true" ]]; then
  RAYCAST_NEAR_OCCLUSION_ARGS=(--disable-raycast-near-occlusion True)
fi

ODOMETRY_ARGS=(--use-odometry --no-include-joint-vel)

if [ ! -f "${LOCOMOTION_CHECKPOINT}" ]; then
  echo "Error: locomotion checkpoint not found: ${LOCOMOTION_CHECKPOINT}"
  exit 1
fi

run_items=()
add_run_item() {
  run_items+=("$1:$2")
}

if [[ "${VARIANT}" == "all" ]]; then
  if [[ "${MODE}" == "dr" || "${MODE}" == "both" ]]; then
    add_run_item dr waypoint_only
    add_run_item dr key_goal_only
    add_run_item dr no_teacher_forcing
    add_run_item dr no_dr_curriculum
  fi
  if [[ "${MODE}" == "odom" || "${MODE}" == "both" ]]; then
    add_run_item odom waypoint_only
    add_run_item odom key_goal_only
  fi
else
  if [[ "${MODE}" == "odom" && "${VARIANT}" != "waypoint_only" && "${VARIANT}" != "key_goal_only" ]]; then
    echo "Error: odometry mode only trains waypoint_only / key_goal_only; ${VARIANT} is not applicable."
    exit 1
  fi
  if [[ "${MODE}" == "both" ]]; then
    add_run_item dr "${VARIANT}"
    if [[ "${VARIANT}" == "waypoint_only" || "${VARIANT}" == "key_goal_only" ]]; then
      add_run_item odom "${VARIANT}"
    fi
  else
    add_run_item "${MODE}" "${VARIANT}"
  fi
fi

if [[ "${#run_items[@]}" -eq 0 ]]; then
  echo "Error: no ablation runs to train."
  exit 1
fi

find_local_planner_checkpoint() {
  local experiment_name="$1"
  local final_ckpt="${OUTPUT_DIR}/local_planner_${experiment_name}_final.pt"
  local latest_ckpt=""

  if [ -f "${final_ckpt}" ]; then
    echo "${final_ckpt}"
    return
  fi

  latest_ckpt=$(ls -t "${OUTPUT_DIR}"/local_planner_"${experiment_name}"*.pt 2>/dev/null | head -1)
  if [ -n "${latest_ckpt}" ]; then
    echo "${latest_ckpt}"
    return
  fi

  return 1
}

train_stage() {
  local stage_label="$1"
  local task="$2"
  local experiment_name="$3"
  local max_iterations="$4"
  local previous_checkpoint="$5"
  shift 5
  local extra_args=("$@")

  echo ""
  echo "=========================================="
  echo "[${stage_label}] starting training"
  echo "Task: ${task}"
  echo "Experiment: ${experiment_name}"
  echo "Iterations: ${max_iterations}"
  if [ -n "${previous_checkpoint}" ]; then
    echo "Resume local planner: ${previous_checkpoint}"
  fi
  echo "=========================================="

  cmd=(
    uv run python -m mjlab.tasks.steering.train_local_planner
    --mode train
    --task "${task}"
    --checkpoint "${LOCOMOTION_CHECKPOINT}"
    --num_envs "${NUM_ENVS}"
    --output_dir "${OUTPUT_DIR}"
    --experiment_name "${experiment_name}"
    --raycast-obs-names ${RAYCAST_OBS_NAMES}
    --max_iterations "${max_iterations}"
    "${RAYCAST_NEAR_OCCLUSION_ARGS[@]}"
    "${extra_args[@]}"
  )

  if [ -n "${previous_checkpoint}" ]; then
    cmd+=(--local_planner_checkpoint "${previous_checkpoint}")
  fi

  "${cmd[@]}"
}

args_for_variant() {
  local variant_name="$1"
  case "${variant_name}" in
    waypoint_only)
      echo "--target-label-mode waypoint_only"
      ;;
    key_goal_only)
      echo "--target-label-mode key_goal_only"
      ;;
    no_teacher_forcing)
      echo "--disable-teacher-forcing"
      ;;
    no_dr_curriculum)
      echo "--disable-dr-curriculum"
      ;;
  esac
}

train_variant() {
  local planner_mode="$1"
  local variant_name="$2"
  local mode_prefix=""
  local mode_title="no odometry"
  local mode_args=()

  if [[ "${planner_mode}" == "odom" ]]; then
    mode_prefix="odom_"
    mode_title="odometry"
    mode_args=("${ODOMETRY_ARGS[@]}")
  fi

  read -r -a variant_args <<< "$(args_for_variant "${variant_name}")"

  local low_exp="ablation_${mode_prefix}${variant_name}_low_speed_v6"
  local indoor_exp="ablation_${mode_prefix}${variant_name}_indoor_v6"

  echo ""
  echo "##########################################"
  echo "Mode: ${mode_title}"
  echo "Variant: ${variant_name}"
  echo "##########################################"

  train_stage \
    "${mode_title} / ${variant_name} / LowSpeed" \
    "${LOW_SPEED_TASK}" \
    "${low_exp}" \
    "${LOW_SPEED_ITERS}" \
    "" \
    "${variant_args[@]}" \
    "${mode_args[@]}"

  local low_ckpt
  low_ckpt=$(find_local_planner_checkpoint "${low_exp}") || {
    echo "Error: LowSpeed checkpoint not found: ${low_exp}"
    exit 1
  }

  local indoor_ckpt="-"

  if [[ "${TASK_SCOPE}" == "all" ]]; then
    train_stage \
      "${mode_title} / ${variant_name} / Indoor" \
      "${INDOOR_TASK}" \
      "${indoor_exp}" \
      "${INDOOR_ITERS}" \
      "${low_ckpt}" \
      "${variant_args[@]}" \
      "${mode_args[@]}"

    indoor_ckpt=$(find_local_planner_checkpoint "${indoor_exp}") || {
      echo "Error: Indoor checkpoint not found: ${indoor_exp}"
      exit 1
    }
  fi

  echo ""
  echo "Mode ${mode_title}, variant ${variant_name} complete. Fill these into the evaluation config:"
  echo "  Low-speed:  ${low_ckpt}"
  if [[ "${TASK_SCOPE}" == "all" ]]; then
    echo "  Indoor:     ${indoor_ckpt}"
  fi
}

echo "=========================================="
echo "Navigation ablation local-planner training"
echo "Locomotion checkpoint: ${LOCOMOTION_CHECKPOINT}"
echo "Mode: ${MODE}"
echo "Task scope: ${TASK_SCOPE}"
echo "Run items: ${run_items[*]}"
echo "Num envs: ${NUM_ENVS}"
echo "LowSpeed/Indoor iterations: ${LOW_SPEED_ITERS}/${INDOOR_ITERS}"
echo "Robot version: ${G1_ROBOT_VERSION}"
echo "raycast near-distance random occlusion: $([[ "${DISABLE_RAYCAST_NEAR_OCCLUSION}" == "true" ]] && echo "disabled" || echo "enabled by default")"
echo "=========================================="

for item in "${run_items[@]}"; do
  planner_mode="${item%%:*}"
  variant_name="${item#*:}"
  train_variant "${planner_mode}" "${variant_name}"
done

echo ""
echo "=========================================="
echo "Navigation ablation training complete."
echo "Template: src/mjlab/tool/experiments/ablation_checkpoints_v6.template.json"
echo "=========================================="
