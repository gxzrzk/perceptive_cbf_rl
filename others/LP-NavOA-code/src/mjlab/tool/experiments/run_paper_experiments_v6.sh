#!/bin/bash
# Run the full paper v6 navigation benchmark and generate figures.
#
# This wrapper evaluates the main table with both planner-free modes. Defaults
# come from eval_utils.py, and can be overridden with the checkpoint flags below.
# It also evaluates the low-speed ablation matrix when an ablation config is
# provided, then runs the plotting script on the same run directory.

set -euo pipefail

OUTPUT_ROOT="logs/benchmark/paper"
RUN_NAME=""
EXACT_RUN_NAME=false
SEEDS=(0 1 2)
EPISODES_PER_SEED=500
NUM_ENVS=256
DEVICE="cuda"
ABLATION_CONFIG="src/mjlab/tool/experiments/ablation_checkpoints_v6.template.json"
EXPERIMENT_GROUPS="main ablation"
MAX_TRAJECTORIES_PER_CONDITION=8
LOW_SPEED_LOCAL_PLANNER_CHECKPOINT=""
LOW_SPEED_ODOM_LOCAL_PLANNER_CHECKPOINT=""
INDOOR_LOCAL_PLANNER_CHECKPOINT=""
INDOOR_ODOM_LOCAL_PLANNER_CHECKPOINT=""
SMOKE=false
DRY_RUN=false
PLOT=true
SAFETY_BRAKE_SCALE_THRESHOLD="0.95"
SAFETY_BRAKE_SCALE_THRESHOLDS="0.85 0.90 0.925 0.95 0.975"
SAFETY_BRAKE_MIN_TIME_S="0.5"

usage() {
  echo "Usage: $0 [options]"
  echo ""
  echo "options:"
  echo "  --run-name NAME"
  echo "  --exact-run-name"
  echo "  --output-root DIR"
  echo "  --seeds \"0 1 2\""
  echo "  --episodes-per-seed N"
  echo "  --num-envs N"
  echo "  --device cuda|cpu"
  echo "  --ablation-config PATH"
  echo "  --experiment-groups \"main ablation\""
  echo "  --max-trajectories-per-condition N"
  echo "  --low-speed-local-planner-checkpoint PATH"
  echo "  --low-speed-odom-local-planner-checkpoint PATH"
  echo "  --indoor-local-planner-checkpoint PATH"
  echo "  --indoor-odom-local-planner-checkpoint PATH"
  echo "  --smoke"
  echo "  --dry-run"
  echo "  --no-plot"
  echo "  --safety-brake-scale-threshold X"
  echo "  --safety-brake-scale-thresholds \"0.85 0.90 0.925 0.95 0.975\""
  echo "  --safety-brake-min-time-s X"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --exact-run-name)
      EXACT_RUN_NAME=true
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --seeds)
      read -r -a SEEDS <<< "$2"
      shift 2
      ;;
    --episodes-per-seed)
      EPISODES_PER_SEED="$2"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --ablation-config)
      ABLATION_CONFIG="$2"
      shift 2
      ;;
    --experiment-groups)
      EXPERIMENT_GROUPS="$2"
      shift 2
      ;;
    --max-trajectories-per-condition)
      MAX_TRAJECTORIES_PER_CONDITION="$2"
      shift 2
      ;;
    --low-speed-local-planner-checkpoint)
      LOW_SPEED_LOCAL_PLANNER_CHECKPOINT="$2"
      shift 2
      ;;
    --low-speed-odom-local-planner-checkpoint)
      LOW_SPEED_ODOM_LOCAL_PLANNER_CHECKPOINT="$2"
      shift 2
      ;;
    --indoor-local-planner-checkpoint)
      INDOOR_LOCAL_PLANNER_CHECKPOINT="$2"
      shift 2
      ;;
    --indoor-odom-local-planner-checkpoint)
      INDOOR_ODOM_LOCAL_PLANNER_CHECKPOINT="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --no-plot)
      PLOT=false
      shift
      ;;
    --safety-brake-scale-threshold)
      SAFETY_BRAKE_SCALE_THRESHOLD="$2"
      shift 2
      ;;
    --safety-brake-scale-thresholds)
      SAFETY_BRAKE_SCALE_THRESHOLDS="$2"
      shift 2
      ;;
    --safety-brake-min-time-s)
      SAFETY_BRAKE_MIN_TIME_S="$2"
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

GROUP_MAIN=false
GROUP_ABLATION=false
read -r -a EXPERIMENT_GROUP_ARRAY <<< "${EXPERIMENT_GROUPS}"
if [[ "${#EXPERIMENT_GROUP_ARRAY[@]}" -eq 0 ]]; then
  echo "Error: --experiment-groups cannot be empty; valid values: main ablation"
  exit 1
fi
for group in "${EXPERIMENT_GROUP_ARRAY[@]}"; do
  case "${group}" in
    main)
      GROUP_MAIN=true
      ;;
    ablation)
      GROUP_ABLATION=true
      ;;
    *)
      echo "Error: unknown experiment group: ${group}; valid values: main ablation"
      exit 1
      ;;
  esac
done

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="${TIMESTAMP}"
elif [[ "${EXACT_RUN_NAME}" != "true" ]]; then
  RUN_NAME="${RUN_NAME}_${TIMESTAMP}"
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ "${#SEEDS[@]}" -eq 1 ]]; then
  SEEDS_LITERAL="(${SEEDS[0]},)"
else
  SEEDS_LITERAL="($(IFS=,; echo "${SEEDS[*]}"))"
fi

cmd=(
  uv run python -m mjlab.tool.experiments.run_paper_experiments
  --preset v6
  --run-name "${RUN_NAME}"
  --output-root "${OUTPUT_ROOT}"
  --seeds "${SEEDS_LITERAL}"
  --episodes-per-seed "${EPISODES_PER_SEED}"
  --num-envs "${NUM_ENVS}"
  --device "${DEVICE}"
  --experiment-groups "${EXPERIMENT_GROUPS}"
  --max-trajectories-per-condition "${MAX_TRAJECTORIES_PER_CONDITION}"
  --safety-brake-scale-threshold "${SAFETY_BRAKE_SCALE_THRESHOLD}"
  --safety-brake-scale-thresholds "${SAFETY_BRAKE_SCALE_THRESHOLDS}"
  --safety-brake-min-time-s "${SAFETY_BRAKE_MIN_TIME_S}"
)

if [[ "${GROUP_ABLATION}" == "true" && -n "${ABLATION_CONFIG}" ]]; then
  cmd+=(--ablation-config "${ABLATION_CONFIG}")
fi
if [[ "${SMOKE}" == "true" ]]; then
  cmd+=(--smoke True)
fi
if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run True)
fi
if [[ -n "${LOW_SPEED_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  cmd+=(--low-speed-local-planner-checkpoint "${LOW_SPEED_LOCAL_PLANNER_CHECKPOINT}")
fi
if [[ -n "${LOW_SPEED_ODOM_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  cmd+=(
    --low-speed-odom-local-planner-checkpoint
    "${LOW_SPEED_ODOM_LOCAL_PLANNER_CHECKPOINT}"
  )
fi
if [[ -n "${INDOOR_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  cmd+=(--indoor-local-planner-checkpoint "${INDOOR_LOCAL_PLANNER_CHECKPOINT}")
fi
if [[ -n "${INDOOR_ODOM_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  cmd+=(--indoor-odom-local-planner-checkpoint "${INDOOR_ODOM_LOCAL_PLANNER_CHECKPOINT}")
fi

echo "=========================================="
echo "Paper v6 benchmark"
echo "Run dir: ${RUN_DIR}"
echo "Seeds: ${SEEDS[*]}"
echo "Episodes/seed: ${EPISODES_PER_SEED}"
echo "Num envs: ${NUM_ENVS}"
echo "Device: ${DEVICE}"
echo "Experiment groups: ${EXPERIMENT_GROUPS}"
if [[ "${GROUP_ABLATION}" == "true" ]]; then
  echo "Ablation config: ${ABLATION_CONFIG:-none}"
else
  echo "Ablation config: none"
fi
echo "Max trajectories/task-method-seed: ${MAX_TRAJECTORIES_PER_CONDITION}"
if [[ -n "${LOW_SPEED_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  echo "Low-speed DR checkpoint: ${LOW_SPEED_LOCAL_PLANNER_CHECKPOINT}"
fi
if [[ -n "${LOW_SPEED_ODOM_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  echo "Low-speed odom checkpoint: ${LOW_SPEED_ODOM_LOCAL_PLANNER_CHECKPOINT}"
fi
if [[ -n "${INDOOR_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  echo "Indoor DR checkpoint: ${INDOOR_LOCAL_PLANNER_CHECKPOINT}"
fi
if [[ -n "${INDOOR_ODOM_LOCAL_PLANNER_CHECKPOINT}" ]]; then
  echo "Indoor odom checkpoint: ${INDOOR_ODOM_LOCAL_PLANNER_CHECKPOINT}"
fi
echo "Safety brake brush: primary scale<${SAFETY_BRAKE_SCALE_THRESHOLD} for ${SAFETY_BRAKE_MIN_TIME_S}s"
echo "Safety brake scale grid: ${SAFETY_BRAKE_SCALE_THRESHOLDS}"
echo "=========================================="

"${cmd[@]}"

if [[ "${DRY_RUN}" == "true" || "${PLOT}" != "true" ]]; then
  exit 0
fi

uv run python -m mjlab.tool.experiments.plot_paper_results \
  --run-dir "${RUN_DIR}"
