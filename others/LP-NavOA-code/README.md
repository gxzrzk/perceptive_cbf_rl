# LP-NavOA

Local navigation and obstacle avoidance for a Unitree G1 humanoid under limited
perception. A perception-action PPO locomotion backbone is trained with a
robot-centred circular heading-speed command and a shared command-side safety
filter. With the backbone frozen, planner and waypoint teachers generate
rollouts that distil a recurrent local planner which overwrites **only the
heading command** at deployment. Inference needs no global map, no waypoint
stream, and no external planner — just proprioception, short-range range
sensing, and a body-frame goal direction.

**Paper:** [LP-NavOA: Integrated Local Navigation and Obstacle Avoidance for
Humanoid Robots under Limited Perception](https://arxiv.org/abs/2606.23249)
(arXiv:2606.23249)

**Project page:** [shenqiqishi.github.io/LP-NavOA](https://shenqiqishi.github.io/LP-NavOA/)
— real-robot demo videos of the obstacle-bypass experiments
([source](https://github.com/shenqiqishi/LP-NavOA))

This repository is a research fork of [mjlab](https://github.com/mujocolab/mjlab)
(Apache-2.0), which supplies the manager-based RL framework. The Python
distribution is still named `mjlab` and the `mjlab.*` imports and `uv run train`
/ `uv run play` entry points are unchanged from upstream. See [`NOTICE`](NOTICE)
for attribution.

---

## Requirements

- An NVIDIA GPU with CUDA 12.8 support for training (CPU works for the tests).
- Python 3.10–3.13 and [`uv`](https://docs.astral.sh/uv/).

## Installation

```bash
git clone https://github.com/shenqiqishi/LP-NavOA-code.git
cd LP-NavOA
uv sync --extra cu128 --group dev     # use --extra cpu without an NVIDIA GPU
uv run list_envs                      # prints the 7 registered tasks
```

## The two navigation tasks

| | Open-wall | Indoor |
| --- | --- | --- |
| Teacher task | `Mjlab-StraightApproach-LowSpeed-G1-v6` | `Mjlab-IndoorStraightApproach-G1-v6` |
| Planner-free twin | `Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6` | `Mjlab-IndoorStraightApproach-NoPP-G1-v6` |
| Scene | robot approaches a straight wall, must bypass it | room walls, a central wall, box obstacles |
| Commanded speed | 0.5–1.2 m/s | 0.5–1.2 m/s |

The `-NoPP-` twins remove the planner waypoints, so the frozen backbone is
steered only by the learned local planner. Evaluate them to measure
deployment-style, planner-free performance.

Both tasks need a locomotion backbone first.

---

## 1. Train the locomotion backbone (once, shared by both tasks)

Three curriculum stages, ~12.5k iterations total. The first stage teaches the
compact `[heading error, target speed]` command interface; the later two add
A\*-guided obstacle bypassing and mild rough terrain.

| Stage | Task | Envs | Iterations |
| --- | --- | --- | --- |
| 1 | `Mjlab-CircularGoal-Flat-G1Rw-Pure-v6` | 8192 | 4000 |
| 2 | `Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6` | 4096 | 4000 |
| 3 | `Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6` | 4096 | 4500 |

```bash
export G1_ROBOT_VERSION=v15                       # v5 or v15 (default v15)
export MJLAB_VL53L8CX_SQUARE_FOVY_DEG=45.0        # datasheet range-sensor FoV
bash src/mjlab/tool/curriculum_train_v6.sh
```

The script runs the three stages in order (each resuming the previous one) and
then both local planners. To run a single stage by hand:

```bash
uv run train Mjlab-CircularGoal-Flat-G1Rw-Pure-v6 \
  --env.scene.num-envs 8192 --agent.max-iterations 4000

uv run train Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6 \
  --agent.resume True --agent.load-run <stage1-run> \
  --agent.load-checkpoint <stage1-checkpoint> \
  --env.scene.num-envs 4096 --agent.max-iterations 4000
```

Checkpoints land in `logs/rsl_rl/g1_velocity_raycast_h1/<timestamp>/`. Use the
Stage 3 checkpoint as `<backbone.pt>` below.

---

## 2. Train the open-wall local planner

Distil the recurrent planner from planner/waypoint teachers while the backbone
stays frozen.

```bash
uv run python -m mjlab.tasks.steering.train_local_planner \
  --mode train \
  --task Mjlab-StraightApproach-LowSpeed-G1-v6 \
  --checkpoint <backbone.pt> \
  --num_envs 1024 \
  --output_dir logs/steering \
  --experiment_name low_speed_v6_511 \
  --raycast-obs-names raycast raycast_left raycast_right \
  --max_iterations 80000
```

Output: `logs/steering/local_planner_low_speed_v6_511_final.pt`

## 3. Train the indoor local planner

Resume from the open-wall planner; the teacher changes, the backbone does not.

```bash
uv run python -m mjlab.tasks.steering.train_local_planner \
  --mode train \
  --task Mjlab-IndoorStraightApproach-G1-v6 \
  --checkpoint <backbone.pt> \
  --local_planner_checkpoint logs/steering/local_planner_low_speed_v6_511_final.pt \
  --num_envs 1024 \
  --output_dir logs/steering \
  --experiment_name indoor_approach_v6_511 \
  --raycast-obs-names raycast raycast_left raycast_right \
  --max_iterations 20000
```

Output: `logs/steering/local_planner_indoor_approach_v6_511_final.pt`

### Odometry variant

The paper also reports a variant that derives the relative goal from odometry,
drops the velocity head, and excludes joint velocities. Append this to either
training command:

```bash
  --use-odometry True --include-joint-vel False
```

Name the output with an `_odom` suffix, e.g. `--experiment_name low_speed_v6_511_odom`.

---

## Evaluate a trained planner

Planner-free rollout metrics for both tasks:

```bash
uv run python -m mjlab.tool.experiments.benchmark_local_planner \
  --stage1-task Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6 \
  --stage2-task Mjlab-IndoorStraightApproach-NoPP-G1-v6 \
  --stage1-local-planner-checkpoint logs/steering/local_planner_low_speed_v6_511_final.pt \
  --stage2-local-planner-checkpoint logs/steering/local_planner_indoor_approach_v6_511_final.pt \
  --num-episodes 500 --num-envs 256
```

The full benchmark (both tasks, dead-reckoning and odometry, plus optional
ablations and figures) is driven by:

```bash
bash src/mjlab/tool/curriculum_train_v6_all.sh            # train + benchmark
bash src/mjlab/tool/experiments/run_paper_experiments_v6.sh --smoke   # benchmark only
```

Watch a policy interactively, and export the frozen planner for deployment:

```bash
uv run python -m mjlab.tasks.steering.train_local_planner --mode test \
  --task Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6 \
  --checkpoint <backbone.pt> \
  --local_planner_checkpoint logs/steering/local_planner_low_speed_v6_511_final.pt \
  --num_envs 1 --viewer native

uv run python -m mjlab.tasks.steering.export_local_planner_to_onnx \
  --checkpoint logs/steering/local_planner_low_speed_v6_511_final.pt \
  --output logs/steering/local_planner_low_speed_v6_511.onnx
```

---

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `G1_ROBOT_VERSION` | `v15` | Selects the G1 model: `v5` or `v15` |
| `MJLAB_VL53L8CX_SQUARE_FOVY_DEG` | `65°` diagonal-derived square FoV | Set to `45.0` for the VL53L8CX datasheet detection-volume angle |
| `KEYBOARD_CONTROL` | unset | `1` enables keyboard goal control in play mode |
| `WALL_LEVEL` | unset (difficulty `0.3–0.7`) | Clamp maze difficulty to `0`, `1`, or `2` |
| `MJLAB_WARP_QUIET` | `0` | `1` silences Warp kernel-compilation logging |
| `WANDB_TAGS` | unset | Comma-separated tags attached to the W&B run |

## Implementation notes

| Item | Setting |
| --- | --- |
| Range sensing | three VL53L8CX-style front/left/right groups, each 8×8 pinhole rays, 45° FoV, 4 m range, base-aligned, 12 Hz |
| Backbone network | actor/critic MLP `(512, 256, 128)`, ELU, observation normalisation; a small convolutional encoder per raycast group |
| PPO | clip 0.2, entropy 0.01, GAE `γ=0.99`, `λ=0.95`, adaptive LR `1e-3`, 5 epochs, 4 mini-batches, 24 steps/env |
| Evaluation | seeds `0 1 2`, 500 episodes per seed, 256 parallel environments |

The actor observation is a fixed 102-dimensional vector (critic 249) whose
concatenation order includes a constant zero-filled seven-element slot, so the
trained interface stays stable across stages and on hardware.

## Repository layout

```
src/mjlab/
├── envs/ managers/ rl/ sensor/ terrains/ scene/ sim/ viewer/ utils/   # mjlab framework
├── asset_zoo/robots/unitree_g1/                                       # G1 description
├── tasks/velocity/config/obstacle_avoidance/g1_raycast_circular_n3_v3/ # the 7 tasks
├── tasks/steering/                                                    # recurrent local planner
└── tool/
    ├── curriculum_train_v6.sh              # backbone + both planners
    ├── curriculum_train_straight_approach.sh
    ├── train_navigation_ablation_v6.sh
    └── experiments/                        # benchmark, ablations, plotting
tests/                                      # framework and task-config tests
typings/                                    # MuJoCo type stubs
```

Only the tasks listed above are registered; the remaining configuration
builders in the task module are internal bases reached by those tasks.

## Development

```bash
make format     # ruff format + ruff check --fix
make type       # ty + pyright
make test       # full test suite (CPU, no GPU required)
make sync       # refresh the environment
```

CI gates on lint, format, and the test suite. The type checkers run as advisory
jobs: this fork inherits pre-existing `ty` and `pyright` diagnostics from the
research code, so they report regressions without failing the build.

## License and citation

Licensed under the [Apache License, Version 2.0](LICENSE), inherited from mjlab.
See [`NOTICE`](NOTICE) for upstream attribution (mjlab, Isaac Lab's BSD-3-Clause
`lab_api` fork, the Unitree G1 model). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
