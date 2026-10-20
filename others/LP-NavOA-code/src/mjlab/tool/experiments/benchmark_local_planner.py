#!/usr/bin/env python3
"""Local planner network performance benchmark.

For the two navigation tasks (LowSpeed open-wall / Indoor), load the
locomotion checkpoint and the local planner checkpoint respectively, run a number of episodes in parallel environments,
collect the following metrics and print a summary table:

  1. Goal reach rate       ── sequence_complete (reached the last key_goal)
  2. Key goal 1 reach rate ── current_key_goal_idx ≥ 1 (reached WP7, past the wall)
  3. Mean waypoints done   ── current_goal_idx at timeout (0-9, 9 WP in total)
  4. Fall/error term. rate ── terminated (hard termination other than timeout,
                              i.e. falling etc.)
  5. Mean dist. at timeout ── when incomplete, distance to the final key_goal
                              at the timeout instant (m)

Additional diagnostic metrics (collision cause analysis):
  6. Collision WP dist.    ── index of the waypoint being approached on collision
  7. Collision position    ── mean XY of the robot relative to spawn on collision (m)
  8. Forward speed         ── actual forward speed of collided vs successful episodes
  9. Local planner corr.   ── whether the local planner is attempting avoidance
  10. Spawn X offset       ── collision rate grouped by spawn lateral offset (±DR)

Usage examples:
  # Training tasks (with intermediate waypoints, for full collision WP analysis)
  uv run python -m mjlab.tool.experiments.benchmark_local_planner \\
    --stage1-local-planner-checkpoint logs/steering/local_planner_low_speed_v8_final.pt \\
    --stage2-local-planner-checkpoint logs/steering/local_planner_indoor_approach_v8_final.pt \\
    --num-episodes 500 --num-envs 256

  # Odometry checkpoints
  uv run python -m mjlab.tool.experiments.benchmark_local_planner \\
    --use-odometry --no-include-joint-vel \\
    --stage1-local-planner-checkpoint logs/steering/local_planner_low_speed_v8_odom_final.pt \\
    --stage2-local-planner-checkpoint logs/steering/local_planner_indoor_approach_v8_odom_final.pt \\
    --num-episodes 500 --num-envs 256

  # NoPP test tasks (no intermediate waypoints, goal reach and collision only)
  uv run python -m mjlab.tool.experiments.benchmark_local_planner \\
    --stage1-task Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6 \\
    --stage2-task Mjlab-IndoorStraightApproach-NoPP-G1-v6 \\
    --stage1-local-planner-checkpoint logs/steering/local_planner_low_speed_v8_final.pt \\
    --stage2-local-planner-checkpoint logs/steering/local_planner_indoor_approach_v8_final.pt \\
    --num-episodes 500 --num-envs 256
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

import numpy as np
import torch
import tyro

import mjlab
from mjlab.tasks.steering.local_planner_inference import (
  LocalPlannerInference,
  LocalPlannerInferenceCfg,
)
from mjlab.tasks.steering.train_local_planner import (
  TrainLocalPlannerArgs,
  _apply_checkpoint_cfg_to_inference_cfg,
  _load_local_planner_network_for_inference,
  load_policy,
  setup_environment,
)
from mjlab.tasks.velocity.mdp.reachable_goal_command import FixedSequentialGoalCommand
from mjlab.tool.experiments.eval_utils import run_policy_eval
from mjlab.utils.torch import configure_torch_backends

# ──────────────────────────────────────────────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkArgs:
  """Local planner performance benchmark arguments."""

  # ── Locomotion checkpoint (all three tasks share one base locomotion net) ──
  locomotion_checkpoint: str = (
    "logs/rsl_rl/g1_velocity_raycast_h1/2026-04-26_06-57-30/model_12497.pt"
  )
  """Base locomotion policy checkpoint (shared by all three stages)."""

  # ── Stage 1: LowSpeed ──
  stage1_task: str = "Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6"
  """Stage 1 task name (v6 NoPP test task by default)."""

  stage1_local_planner_checkpoint: str = (
    "logs/steering/local_planner_low_speed_v6_511_final.pt"
  )
  """Stage 1 local planner checkpoint path."""

  # ── Stage 2: Indoor ──
  stage2_task: str = "Mjlab-IndoorStraightApproach-NoPP-G1-v6"
  """Stage 2 task name (v6 NoPP test task by default)."""

  stage2_local_planner_checkpoint: str = (
    "logs/steering/local_planner_indoor_approach_v6_511_final.pt"
  )
  """Stage 2 local planner checkpoint path."""

  # ── Run configuration ──
  num_envs: int = 256
  """Number of parallel environments (affects episode collection speed)."""

  num_episodes: int = 200
  """Number of independent episodes collected per task (statistics base)."""

  device: str = "cuda"
  """Run device ("cuda" or "cpu")."""

  seed: int | None = None
  """Random seed; when None, the environment default randomness is used."""

  raycast_obs_names: tuple[str, ...] = ("raycast", "raycast_left", "raycast_right")
  """Raycast observation group names (three-camera n3 configuration)."""

  correction_gain: float = 1.0
  """Local planner correction gain."""

  use_odometry: bool = False
  """Use odometry-mode local planner checkpoints."""

  include_joint_vel: bool = True
  """Include joint velocities in local-planner observations.
  Pass --no-include-joint-vel with the v8 odometry checkpoints."""

  use_measured_velocity: bool = False
  use_ahrs_heading: bool = True
  use_projected_yaw_rate: bool = False

  log_dir: str = "logs/benchmark"
  """Benchmark log output directory (JSON format, full per-episode records)."""

  wall_y_distances: tuple[float, float] = (6.0, 5.0)
  """Forward wall distance for the three stages (m, relative to spawn), used for
  episode collision position classification.
  Defaults: Stage1(LowSpeed)=6.0, Stage2(Indoor)=5.0."""

  wall_proximity_threshold: float = 0.15
  """Wall proximity metric threshold (normalized raycast [0,1]); below this value
  the robot is considered "close to the wall"."""

  success_direct_max_steps: tuple[int, int] = (1200, 1300)
  """Maximum episode step threshold for "direct arrival" per stage.
  Successful episodes exceeding this step count are treated as "wandering
  arrivals" (the robot wandered before reaching the goal by luck).
  Defaults: Stage1(LowSpeed, 0.5-1.2m/s)=1200, Stage2(Indoor, 0.5-1.2m/s)=1300."""


# ──────────────────────────────────────────────────────────────────────────────
# Episode metric container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EpisodeRecord:
  """Result record of a single episode."""

  sequence_complete: bool  # reached the final goal (WP8)
  key_goal1_reached: bool  # reached the first key_goal (WP7)
  waypoints_completed: int  # total waypoints completed (= current_goal_idx)
  hard_terminated: bool  # hard termination (fall/collision), not a timeout
  dist_to_final: float  # distance to the final goal at timeout/termination (m)
  episode_steps: int  # number of steps this episode went through

  # ── Collision diagnostic fields ───────────────────────────────────────────
  collision_waypoint_idx: int = -1
  """Index of the waypoint being approached on collision; -1 if no collision."""

  collision_pos_x_rel: float = float("nan")
  """X coordinate of the robot relative to spawn on collision (m); NaN otherwise."""

  collision_pos_y_rel: float = float("nan")
  """Y coordinate of the robot relative to spawn on collision (m); NaN otherwise."""

  actual_velocity: float = float("nan")
  """Actual forward speed of the robot at episode end (body frame, m/s)."""

  local_planner_correction_at_end: float = float("nan")
  """delta_yaw correction output by the local planner at the last step (rad/s)."""

  min_raycast_at_end: float = float("nan")
  """Nearest normalized raycast reading at the last step [0, 1] (1 = farthest/no
  obstacle)."""

  spawn_x_offset: float = float("nan")
  """Lateral X offset of the robot relative to env_origin at spawn (m, DR value)."""

  min_raycast_ever: float = float("nan")
  """Minimum normalized raycast reading over the whole episode (lowest value)."""


# ──────────────────────────────────────────────────────────────────────────────
# Single-task benchmark
# ──────────────────────────────────────────────────────────────────────────────


def _make_local_planner_enhanced_policy(
  base_policy: Callable[..., torch.Tensor],
  local_planner_inference: LocalPlannerInference,
  env: object,
  num_envs: int,
  correction_gain: float,
  device: torch.device,
  last_corrections: torch.Tensor,
) -> Callable[..., torch.Tensor]:
  """Apply the local planner correction to the policy observation, then return
  the action.

  The modification follows the same policy wrapping used by
  test_local_planner() in train_local_planner.py: find the offset of
  goal_command (delta_yaw, vel) inside the policy actor obs and directly
  replace its delta_yaw component with the local planner output.

  Args:
    last_corrections: [num_envs] tensor; written in-place with the correction
      of the current step after every call.
  """

  _goal_cmd_offset: int | None = None

  def _find_offset(obs: object) -> int:
    try:
      obs_manager = env.observation_manager  # type: ignore[union-attr]
      term_names = obs_manager.active_terms.get("actor", [])
      term_dims = obs_manager.group_obs_term_dim.get("actor", [])
      offset = 0
      for name, dims in zip(term_names, term_dims, strict=False):
        if name == "goal_command":
          return offset
        offset += int(np.prod(dims))
      return 93  # G1 fallback
    except Exception:
      return 93

  def policy(obs: object) -> torch.Tensor:
    nonlocal _goal_cmd_offset

    # Build obs_dict for the local planner
    local_planner_obs: dict[str, torch.Tensor] = {}
    if hasattr(obs, "keys"):
      for k in obs.keys():  # type: ignore[union-attr]
        v = obs[k]  # type: ignore[index]
        if isinstance(v, torch.Tensor):
          local_planner_obs[k] = v
        elif hasattr(v, "keys"):
          for sk in v.keys():
            sv = v[sk]
            if isinstance(sv, torch.Tensor):
              local_planner_obs[sk] = sv
    else:
      local_planner_obs = {"obs": obs}  # type: ignore[assignment]

    correction = local_planner_inference.compute_correction(env, local_planner_obs)  # type: ignore[arg-type]

    # Record the correction of this step (read by the caller on collision)
    last_corrections.copy_(correction.squeeze(-1))

    # Apply to policy obs
    has_actor = hasattr(obs, "keys") and "actor" in list(obs.keys())  # type: ignore[union-attr]
    if has_actor:
      if _goal_cmd_offset is None:
        _goal_cmd_offset = _find_offset(obs)
      policy_obs: torch.Tensor = obs["actor"]  # type: ignore[index]
      modified = policy_obs.clone()
      local_planner_delta_yaw = correction.squeeze(-1) * correction_gain
      modified[:, _goal_cmd_offset] = local_planner_delta_yaw
      obs["actor"] = modified  # type: ignore[index]

    with torch.no_grad():
      return base_policy(obs)

  return policy


def run_task_benchmark(
  task: str,
  locomotion_checkpoint: str,
  local_planner_checkpoint: str,
  num_envs: int,
  num_episodes: int,
  raycast_obs_names: tuple[str, ...],
  device: torch.device,
  correction_gain: float,
  use_odometry: bool,
  include_joint_vel: bool,
  use_measured_velocity: bool,
  use_ahrs_heading: bool,
  use_projected_yaw_rate: bool,
  seed: int | None = None,
) -> tuple[list[EpisodeRecord], int, list[int]]:
  """Run the benchmark for a single task and return per-episode metric lists.

  Args:
    task: Task name (a registered Mjlab task ID).
    locomotion_checkpoint: Path to the base locomotion policy checkpoint.
    local_planner_checkpoint: Path to the local planner checkpoint.
    num_envs: Number of parallel environments.
    num_episodes: Target number of episodes to collect.
    raycast_obs_names: Raycast observation group names.
    device: Run device.
    correction_gain: Local planner correction gain.
    use_measured_velocity: Whether to use measured velocity for DR.
    use_ahrs_heading: Whether to use the AHRS attitude estimate for heading.
    use_projected_yaw_rate: Whether to project the yaw rate.

  Returns:
    (records, num_goals, key_goal_indices)
  """
  shared_result = run_policy_eval(
    task=task,
    locomotion_checkpoint=locomotion_checkpoint,
    local_planner_checkpoint=local_planner_checkpoint,
    num_envs=num_envs,
    num_episodes=num_episodes,
    raycast_obs_names=raycast_obs_names,
    device=device,
    seed=seed,
    mode="test",
    correction_gain=correction_gain,
    use_odometry=use_odometry,
    include_joint_vel=include_joint_vel,
    use_measured_velocity=use_measured_velocity,
    use_ahrs_heading=use_ahrs_heading,
    use_projected_yaw_rate=use_projected_yaw_rate,
  )
  return (
    shared_result.records,  # type: ignore[return-value]
    shared_result.num_goals,
    shared_result.key_goal_indices,
  )

  configure_torch_backends()

  # ── Build environment ───────────────────────────────────────────────────────
  train_args = TrainLocalPlannerArgs(
    mode="test",
    task=task,
    num_envs=num_envs,
    checkpoint=locomotion_checkpoint,
    local_planner_checkpoint=local_planner_checkpoint,
    device=str(device),
    raycast_obs_names=raycast_obs_names,
    use_measured_velocity=use_measured_velocity,
    use_ahrs_heading=use_ahrs_heading,
    use_projected_yaw_rate=use_projected_yaw_rate,
    correction_gain=correction_gain,
  )
  env, rl_cfg = setup_environment(train_args)

  # ── Load locomotion policy ──────────────────────────────────────────────────
  base_policy = load_policy(locomotion_checkpoint, env, rl_cfg, device)

  # ── Build local planner network ───────────────────────────────────────────
  local_planner_network, checkpoint_cfg = _load_local_planner_network_for_inference(
    local_planner_checkpoint,
    raycast_obs_names=raycast_obs_names,
    device=device,
    use_odometry=train_args.use_odometry,
    include_joint_vel=train_args.include_joint_vel,
  )

  # ── Local planner inference module ────────────────────────────────────────
  local_planner_cfg = LocalPlannerInferenceCfg()
  _apply_checkpoint_cfg_to_inference_cfg(local_planner_cfg, checkpoint_cfg, train_args)
  local_planner_cfg.use_measured_velocity = use_measured_velocity
  local_planner_cfg.use_ahrs_heading = use_ahrs_heading
  local_planner_cfg.use_projected_yaw_rate = use_projected_yaw_rate

  local_planner_inference = LocalPlannerInference(
    cfg=local_planner_cfg,
    local_planner_network=local_planner_network,
    device=device,
  )
  local_planner_inference.initialize(num_envs)

  # Fetch and validate the goal command term
  goal_cmd_term = env.command_manager.get_term("goal")
  if goal_cmd_term is None:
    raise RuntimeError("Cannot find the 'goal' command term in the environment.")
  if not isinstance(goal_cmd_term, FixedSequentialGoalCommand):
    raise RuntimeError(
      f"'goal' term has wrong type: expected FixedSequentialGoalCommand, "
      f"got {type(goal_cmd_term)}"
    )

  num_goals: int = goal_cmd_term.num_goals
  key_goal_indices: list[int] = list(goal_cmd_term._key_goal_indices)
  print(f"  Total waypoints: {num_goals}, key goal indices: {key_goal_indices}")

  # ── Per-step correction cache (written by enhanced_policy) ───────────────
  last_corrections = torch.zeros(num_envs, device=device)

  # ── Wrap policy + local planner ──────────────────────────────────────────
  initial_goal_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
  enhanced_policy = _make_local_planner_enhanced_policy(
    base_policy=base_policy,
    local_planner_inference=local_planner_inference,
    env=env,
    num_envs=num_envs,
    correction_gain=correction_gain,
    device=device,
    last_corrections=last_corrections,
  )

  # ── Per-env spawn X offset tracking (robot pos - env_origin after spawn) ──
  env_spawn_x_offset = torch.zeros(num_envs, device=device)

  def _update_spawn_x_offset(env_ids: torch.Tensor) -> None:
    """Call right after an episode reset to record the new spawn X offset."""
    robot = env.scene["robot"]
    robot_x = robot.data.root_link_pos_w[env_ids, 0]
    origin_x = env.scene.env_origins[env_ids, 0]
    env_spawn_x_offset[env_ids] = robot_x - origin_x

  def _get_min_raycast(obs_d: object) -> torch.Tensor:
    """Take the minimum over the raycast groups in obs_dict (normalized [0,1])."""
    parts: list[torch.Tensor] = []
    for rc_name in raycast_obs_names:
      rc_tensor: torch.Tensor | None = None
      if hasattr(obs_d, "keys"):
        if rc_name in obs_d.keys():  # type: ignore[union-attr]
          v = obs_d[rc_name]  # type: ignore[index]
          rc_tensor = v if isinstance(v, torch.Tensor) else None
        else:
          # Try to find it inside a nested dict
          for k in obs_d.keys():  # type: ignore[union-attr]
            sub = obs_d[k]  # type: ignore[index]
            if hasattr(sub, "keys") and rc_name in sub.keys():
              sv = sub[rc_name]
              if isinstance(sv, torch.Tensor):
                rc_tensor = sv
                break
      if rc_tensor is not None:
        parts.append(rc_tensor.min(dim=-1).values)
    if parts:
      return torch.stack(parts, dim=-1).min(dim=-1).values
    return torch.zeros(num_envs, device=device)

  # ── Environment initialization ─────────────────────────────────────────────
  obs_dict, _ = env.reset()
  local_planner_inference.obs_builder.update_initial_goal(env, initial_goal_mask)
  _update_spawn_x_offset(torch.arange(num_envs, device=device))

  # ── Current episode step count tracking per env ────────────────────────────
  env_step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
  env_min_raycast_ever = torch.ones(num_envs, device=device)

  records: list[EpisodeRecord] = []
  t_start = time.time()

  print(f"  Collecting {num_episodes} episodes ({num_envs} parallel envs)...")

  while len(records) < num_episodes:
    # ── Snapshot the state before this step (read when an episode ends) ─────
    prev_goal_idx = goal_cmd_term.current_goal_idx.clone()
    prev_dist_to_key_goal = goal_cmd_term.distance_to_key_goal.clone()
    prev_env_steps = env_step_counts.clone()

    # Robot position / velocity / raycast (used for collision diagnostics)
    robot = env.scene["robot"]
    prev_robot_pos = robot.data.root_link_pos_w[:, :2].clone()
    prev_spawn_pos = goal_cmd_term._spawn_pos_w.clone()
    prev_robot_vel = robot.data.root_link_lin_vel_b[:, 0].clone()  # forward vel (m/s)
    prev_min_raycast = _get_min_raycast(obs_dict)
    env_min_raycast_ever = torch.minimum(env_min_raycast_ever, prev_min_raycast)
    prev_local_planner_correction = last_corrections.clone()
    prev_spawn_x = env_spawn_x_offset.clone()

    # ── Forward inference & environment stepping ────────────────────────────
    with torch.no_grad():
      actions = enhanced_policy(obs_dict)
    obs_dict, _reward, terminated, time_outs, _extras = env.step(actions)

    env_step_counts += 1
    dones = terminated | time_outs
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

    if done_ids.numel() > 0:
      # Collect this episode's metrics (using the pre-step snapshot)
      for idx in done_ids:
        i = int(idx.item())
        prev_goal = int(prev_goal_idx[i].item())
        first_key_wp = (
          key_goal_indices[0] if key_goal_indices else max(0, num_goals - 1)
        )

        # Note: the command term sets sequence_complete to True on timeout,
        # so prev_seq_complete cannot be trusted directly here.
        # Success definition: not a timeout and the final waypoint has already
        # been completed on the last step.
        is_timeout = bool(time_outs[i].item())
        is_success = (not is_timeout) and (prev_goal >= num_goals - 1)

        # Hard termination: terminated and not successful (timeouts excluded).
        is_hard = bool(terminated[i].item()) and (not is_success) and (not is_timeout)

        # key_goal1 reached: the current goal index has passed the first key
        # goal waypoint. E.g. with first key goal = WP7, current_goal_idx
        # becomes 8 once it is reached.
        key1_reached = prev_goal >= (first_key_wp + 1)

        if is_hard:
          col_wp = int(prev_goal_idx[i].item())
          col_x = float((prev_robot_pos[i, 0] - prev_spawn_pos[i, 0]).item())
          col_y = float((prev_robot_pos[i, 1] - prev_spawn_pos[i, 1]).item())
        else:
          col_wp = -1
          col_x = float("nan")
          col_y = float("nan")

        records.append(
          EpisodeRecord(
            sequence_complete=is_success,
            key_goal1_reached=key1_reached,
            waypoints_completed=prev_goal,
            hard_terminated=is_hard,
            dist_to_final=float(prev_dist_to_key_goal[i].item()),
            episode_steps=int(prev_env_steps[i].item()),
            collision_waypoint_idx=col_wp,
            collision_pos_x_rel=col_x,
            collision_pos_y_rel=col_y,
            actual_velocity=float(prev_robot_vel[i].item()),
            local_planner_correction_at_end=float(
              prev_local_planner_correction[i].item()
            ),
            min_raycast_at_end=float(prev_min_raycast[i].item()),
            spawn_x_offset=float(prev_spawn_x[i].item()),
            min_raycast_ever=float(env_min_raycast_ever[i].item()),
          )
        )

      # Record the new spawn position (done envs were reset by env.step)
      _update_spawn_x_offset(done_ids)
      # Reset the step counters
      env_step_counts[done_ids] = 0
      env_min_raycast_ever[done_ids] = 1.0
      # compute_correction() resets automatically when episode_length_buf == 0,
      # so there is no need to call local_planner_inference.reset() explicitly.

    # Progress report (printed once every 50 episodes collected)
    n_done = len(records)
    if n_done > 0 and n_done // 50 > (n_done - done_ids.numel()) // 50:
      elapsed = time.time() - t_start
      rate = len(records) / elapsed if elapsed > 0 else 0.0
      print(
        f"  Progress: {len(records)}/{num_episodes} eps ({elapsed:.0f}s, "
        f"{rate:.1f} eps/s)"
      )

  env.close()
  elapsed = time.time() - t_start
  print(f"  Finished: {len(records)} episodes in {elapsed:.1f}s.")
  return records[:num_episodes], num_goals, key_goal_indices


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics & table printing
# ──────────────────────────────────────────────────────────────────────────────


def compute_stats(records: list[EpisodeRecord], num_goals: int = 0) -> dict[str, float]:
  """Compute summary statistics from a list of EpisodeRecord.

  Args:
    records: List of episode records.
    num_goals: Total number of waypoints. Used to detect NoPP tasks
      (num_goals ≤ 2).
  """
  n = len(records)
  if n == 0:
    return {}

  seq_complete = np.array([r.sequence_complete for r in records], dtype=float)
  key1_reached = np.array([r.key_goal1_reached for r in records], dtype=float)
  wp_completed = np.array([r.waypoints_completed for r in records], dtype=float)
  hard_term = np.array([r.hard_terminated for r in records], dtype=float)
  timed_out = np.array(
    [
      getattr(r, "timed_out", not r.sequence_complete and not r.hard_terminated)
      for r in records
    ],
    dtype=float,
  )
  dist_final = np.array([r.dist_to_final for r in records], dtype=float)
  ep_steps = np.array([r.episode_steps for r in records], dtype=float)

  # Distance to the goal at timeout (only timed-out, incomplete episodes)
  timeout_incomplete = np.array(
    [
      getattr(r, "timed_out", not r.sequence_complete and not r.hard_terminated)
      and not r.sequence_complete
      for r in records
    ]
  )
  dist_at_timeout = (
    float(dist_final[timeout_incomplete].mean()) if timeout_incomplete.any() else 0.0
  )

  stats: dict[str, float] = {
    "success_rate (%)": float(seq_complete.mean() * 100),
    "key_goal_1_reach_rate (%)": float(key1_reached.mean() * 100),
  }
  # Only training tasks (with intermediate waypoints) report this metric; NoPP
  # tasks have just 2 key goals and no intermediate waypoints.
  if num_goals > 2:
    stats[f"mean_waypoints_completed/{num_goals}"] = float(wp_completed.mean())

  stats["termination_rate (%)"] = float(hard_term.mean() * 100)
  stats["timeout_rate (%)"] = float(timed_out.mean() * 100)
  stats["distance_to_goal_at_timeout (m)"] = dist_at_timeout
  stats["mean_episode_steps"] = float(ep_steps.mean())
  stats["mean_action_acc"] = float(
    np.nanmean([getattr(r, "mean_action_acc", float("nan")) for r in records])
  )
  stats["action_rate_l2"] = float(
    np.nanmean([getattr(r, "action_rate_l2", float("nan")) for r in records])
  )
  stats["base_ang_vel_var"] = float(
    np.nanmean([getattr(r, "base_ang_vel_var", float("nan")) for r in records])
  )
  stats["gait_preservation_score"] = float(
    np.nanmean([getattr(r, "gait_preservation_score", float("nan")) for r in records])
  )

  return stats


def classify_episodes(
  records: list[EpisodeRecord],
  wall_y: float,
  wall_proximity: float = 0.15,
  direct_max_steps: int = 600,
) -> dict[str, int]:
  """Classify episodes into eight categories by failure mode.

  Collision (hard_terminated), three categories ── by collision_pos_y_rel:
    spawn_failure:        Y_rel < 0 (collision behind the spawn)
    avoidance_failure:    0 ≤ Y_rel < wall_y + 2.0 (collision before the wall
                          = avoidance failure)
    goal_seeking_failure: Y_rel ≥ wall_y + 2.0 (collision after the wall
                          = goal-seeking failure)

  Success (sequence_complete), three categories ──
    direct_clean_success: episode_steps ≤ direct_max_steps and never near wall
    direct_brush_success: episode_steps ≤ direct_max_steps and was near wall
    wandering_success:    episode_steps > direct_max_steps (reached the goal
                          after wandering)

  Not collision and not success (timeout) ── by min_raycast_ever:
    wall_brush_timeout:   min_raycast_ever ≤ proximity (brushed the wall but
                          survived, then timed out)
    timeout_clean:        min_raycast_ever > proximity (timed out without
                          approaching the wall)

  Args:
    records: List of episode records.
    wall_y: Wall Y distance (m, relative to spawn).
    wall_proximity: Wall proximity threshold (normalized raycast).
    direct_max_steps: Maximum step count for a "direct arrival"; beyond this the
      arrival counts as a "wandering arrival".
  """
  result = {
    "total": len(records),
    "direct_clean_success": 0,
    "direct_brush_success": 0,
    "wandering_success": 0,
    "spawn_failure": 0,
    "avoidance_failure": 0,
    "goal_seeking_failure": 0,
    "wall_brush_timeout": 0,
    "timeout_clean": 0,
  }
  post_wall_cutoff = wall_y + 2.0

  for r in records:
    if r.hard_terminated:
      y = r.collision_pos_y_rel
      if math.isnan(y) or y < 0:
        result["spawn_failure"] += 1
      elif y < post_wall_cutoff:
        result["avoidance_failure"] += 1
      else:
        result["goal_seeking_failure"] += 1
    elif r.sequence_complete:
      is_direct = r.episode_steps <= direct_max_steps
      near_wall = (not math.isnan(r.min_raycast_ever)) and (
        r.min_raycast_ever <= wall_proximity
      )
      if not is_direct:
        result["wandering_success"] += 1
      elif near_wall:
        result["direct_brush_success"] += 1
      else:
        result["direct_clean_success"] += 1
    else:
      # timeout
      near_wall = (not math.isnan(r.min_raycast_ever)) and (
        r.min_raycast_ever <= wall_proximity
      )
      if near_wall:
        result["wall_brush_timeout"] += 1
      else:
        result["timeout_clean"] += 1

  return result


_CLASSIFICATION_LABELS: dict[str, tuple[str, str]] = {
  "direct_clean_success": (
    "✅ Direct (no wall contact)",
    "steps within limit, never near wall",
  ),
  "direct_brush_success": (
    "✅ Direct (brushed wall)",
    "steps within limit but was near wall",
  ),
  "wandering_success": (
    "⚠ Wandering arrival",
    "steps over limit, arrived after wandering",
  ),
  "avoidance_failure": (
    "⚠ Avoidance failure (pre-wall)",
    "collision terminated in the wall zone",
  ),
  "goal_seeking_failure": (
    "⚠ Goal-seeking failure (post-wall)",
    "late wandering collision past the wall",
  ),
  "spawn_failure": (
    "⚠ Collision behind spawn",
    "collision near init (incl. Y<0)",
  ),
  "wall_brush_timeout": (
    "⏱ Wall brush, no termination, timeout",
    "near wall, survived, timed out",
  ),
  "timeout_clean": (
    "⏱ Timeout (no wall contact)",
    "timed out without approaching the wall",
  ),
}

_CLASSIFICATION_ORDER = [
  "direct_clean_success",
  "direct_brush_success",
  "wandering_success",
  "avoidance_failure",
  "goal_seeking_failure",
  "spawn_failure",
  "wall_brush_timeout",
  "timeout_clean",
]


def print_episode_classification(
  task_label: str,
  cls: dict[str, int],
  wall_y: float,
  wall_proximity: float,
  direct_max_steps: int = 600,
) -> None:
  """Print the episode eight-category table."""
  n = cls["total"]
  if n == 0:
    return

  print()
  print(
    f"  [{task_label}] Episode classification"
    f" (wall_y={wall_y:.1f}m, proximity threshold={wall_proximity},"
    f" direct step threshold={direct_max_steps})"
  )
  print(f"  {'─' * 76}")

  for key in _CLASSIFICATION_ORDER:
    cnt = cls[key]
    label, desc = _CLASSIFICATION_LABELS[key]
    pct = cnt / n * 100 if n > 0 else 0
    print(f"  {_rpad(label, 36)} {cnt:4d}  ({pct:5.1f}%)  {desc}")

  # Pure avoidance failure rate (excluding spawn and late wandering)
  pure_avoid = cls["avoidance_failure"]
  print(
    f"\n  Pure avoidance failure rate: {pure_avoid}/{n} = {pure_avoid / n * 100:.1f}%"
    " (excluding spawn failures and post-wall wandering)"
  )
  print()


def _wp_label(wp_idx: int, key_goal_indices: list[int]) -> str:
  """Build a waypoint label; key goals get a star marker."""
  tag = "★" if wp_idx in key_goal_indices else "  "
  return f"{tag}WP{wp_idx}"


def _cjk_len(s: str) -> int:
  """Display width of a string (CJK and full-width chars count as width 2)."""
  w = 0
  for c in s:
    if (
      "\u4e00" <= c <= "\u9fff"
      or "\u3000" <= c <= "\u303f"
      or "\uff00" <= c <= "\uffef"
      or "\u2600" <= c <= "\u26ff"
    ):
      w += 2
    else:
      w += 1
  return w


def _rpad(s: str, width: int) -> str:
  """Right-pad with spaces to the given display width."""
  return s + " " * max(0, width - _cjk_len(s))


def _lpad(s: str, width: int) -> str:
  """Left-pad with spaces to the given display width."""
  return " " * max(0, width - _cjk_len(s)) + s


def compute_collision_table(
  records: list[EpisodeRecord],
  num_goals: int,
  key_goal_indices: list[int],
) -> list[dict]:
  """Count collisions per waypoint index and average collision position.

  Returns:
    One row per waypoint, each containing: wp_idx, label, collision_count,
    collision_rate_pct, collision_pct_of_all, mean_x, mean_y
  """
  n_total = len(records)
  collision_records = [r for r in records if r.hard_terminated]
  n_collision = len(collision_records)

  rows = []
  for wp in range(num_goals):
    wp_cols = [r for r in collision_records if r.collision_waypoint_idx == wp]
    cnt = len(wp_cols)
    if cnt == 0:
      rows.append(
        {
          "wp_idx": wp,
          "label": _wp_label(wp, key_goal_indices),
          "collision_count": 0,
          "collision_rate_pct": 0.0,
          "collision_pct_of_all": 0.0,
          "mean_x": float("nan"),
          "mean_y": float("nan"),
        }
      )
      continue
    xs = np.array([r.collision_pos_x_rel for r in wp_cols], dtype=float)
    ys = np.array([r.collision_pos_y_rel for r in wp_cols], dtype=float)
    rows.append(
      {
        "wp_idx": wp,
        "label": _wp_label(wp, key_goal_indices),
        "collision_count": cnt,
        "collision_rate_pct": cnt / n_total * 100.0,
        "collision_pct_of_all": cnt / n_collision * 100.0 if n_collision > 0 else 0.0,
        "mean_x": float(np.nanmean(xs)),
        "mean_y": float(np.nanmean(ys)),
      }
    )
  return rows


def analyze_collision_causes(
  records: list[EpisodeRecord],
  num_goals: int,
  key_goal_indices: list[int],
) -> None:
  """Print cross-analysis of collision causes: speed, Spawn X, correction, raycast."""
  n = len(records)
  if n == 0:
    return

  collision_recs = [r for r in records if r.hard_terminated]
  success_recs = [r for r in records if r.sequence_complete]
  n_col = len(collision_recs)
  n_suc = len(success_recs)

  sep_len = 72
  print()
  print("=" * sep_len)
  print("  Collision cause analysis")
  print("=" * sep_len)
  print(f"  Total episodes: {n}, collisions: {n_col}, successes: {n_suc}")
  print()

  if n_col == 0:
    print("  No collision episodes, analysis skipped.")
    print()
    return

  # ── 1. Collision WP distribution ─────────────────────────────────────────
  print("── Collision WP distribution ──")
  wp_counter: dict[int, int] = {}
  for r in collision_recs:
    wp_counter[r.collision_waypoint_idx] = (
      wp_counter.get(r.collision_waypoint_idx, 0) + 1
    )
  for wp in sorted(wp_counter):
    label = _wp_label(wp, key_goal_indices) if wp >= 0 else "   ?"
    bar = "█" * int(wp_counter[wp] / n_col * 30)
    print(f"  {_rpad(label, 8)} {_lpad(str(wp_counter[wp]), 4)}  {bar}")
  print()

  # ── 2. Speed distribution comparison (collision vs success) ──────────────
  col_vel = np.array([r.actual_velocity for r in collision_recs], dtype=float)
  suc_vel = np.array([r.actual_velocity for r in success_recs], dtype=float)
  col_vel = col_vel[np.isfinite(col_vel)]
  suc_vel = suc_vel[np.isfinite(suc_vel)]

  print("── Speed distribution (actual forward speed m/s) ──")
  if col_vel.size > 0:
    print(
      f"  Collision (n={col_vel.size}): "
      f"mean={col_vel.mean():.2f}  median={np.median(col_vel):.2f}  "
      f"p25={np.percentile(col_vel, 25):.2f}  p75={np.percentile(col_vel, 75):.2f}"
    )
  if suc_vel.size > 0:
    print(
      f"  Success (n={suc_vel.size}): "
      f"mean={suc_vel.mean():.2f}  median={np.median(suc_vel):.2f}  "
      f"p25={np.percentile(suc_vel, 25):.2f}  p75={np.percentile(suc_vel, 75):.2f}"
    )
  print()

  # ── 3. Spawn X offset distribution comparison ─────────────────────────────
  col_spx = np.array([r.spawn_x_offset for r in collision_recs], dtype=float)
  suc_spx = np.array([r.spawn_x_offset for r in success_recs], dtype=float)
  col_spx = col_spx[np.isfinite(col_spx)]
  suc_spx = suc_spx[np.isfinite(suc_spx)]

  print("── Spawn X offset distribution (positive = right, vs env_origin, m) ──")
  if col_spx.size > 0:
    print(
      f"  Collision (n={col_spx.size}): "
      f"mean={col_spx.mean():.2f}  median={np.median(col_spx):.2f}  "
      f"p25={np.percentile(col_spx, 25):.2f}  p75={np.percentile(col_spx, 75):.2f}"
    )
  if suc_spx.size > 0:
    print(
      f"  Success (n={suc_spx.size}): "
      f"mean={suc_spx.mean():.2f}  median={np.median(suc_spx):.2f}  "
      f"p25={np.percentile(suc_spx, 25):.2f}  p75={np.percentile(suc_spx, 75):.2f}"
    )

  # Spawn X bucketed collision rate (-3m ~ +3m, one bucket per 1m)
  all_spx = np.array([r.spawn_x_offset for r in records], dtype=float)
  if np.isfinite(all_spx).any():
    print()
    print("  Spawn X bucketed collision rate:")
    bins = np.arange(-3.0, 4.0, 1.0)
    for lo in bins[:-1]:
      hi = lo + 1.0
      in_bin = [r for r in records if lo <= r.spawn_x_offset < hi]
      col_in = [r for r in in_bin if r.hard_terminated]
      if not in_bin:
        continue
      rate = len(col_in) / len(in_bin) * 100
      bar = "█" * int(rate / 5)
      print(
        f"    [{lo:+.0f}, {hi:+.0f})  {len(in_bin):4d} eps  "
        f"collision rate {rate:5.1f}%  {bar}"
      )
  print()

  # ── 4. Local planner correction analysis ─────────────────────────────────
  col_local_planner = np.array(
    [r.local_planner_correction_at_end for r in collision_recs],
    dtype=float,
  )
  suc_local_planner = np.array(
    [r.local_planner_correction_at_end for r in success_recs],
    dtype=float,
  )
  col_local_planner = col_local_planner[np.isfinite(col_local_planner)]
  suc_local_planner = suc_local_planner[np.isfinite(suc_local_planner)]

  print("── Local planner correction (collision vs success) ──")
  print("  (large = local planner steering hard, small = barely correcting)")
  if col_local_planner.size > 0:
    print(
      f"  Collision (n={col_local_planner.size}): "
      f"mean={col_local_planner.mean():.3f}  "
      f"|mean|={np.abs(col_local_planner).mean():.3f}  "
      f"median={np.median(col_local_planner):.3f}"
    )
    # Whether the local planner tried to avoid (|correction| > 0.3 = active)
    active_avoid = int((np.abs(col_local_planner) > 0.3).sum())
    passive = int((np.abs(col_local_planner) <= 0.3).sum())
    print(
      f"    In collisions, local planner |corr| > 0.3: "
      f"{active_avoid}/{col_local_planner.size} "
      f"({active_avoid / col_local_planner.size * 100:.0f}%)  "
      f"<- saw the wall but still collided"
    )
    print(
      f"    In collisions, local planner |corr| <= 0.3: "
      f"{passive}/{col_local_planner.size} "
      f"({passive / col_local_planner.size * 100:.0f}%)  "
      f"<- did not effectively detect the obstacle"
    )
  if suc_local_planner.size > 0:
    print(
      f"  Success (n={suc_local_planner.size}): "
      f"mean={suc_local_planner.mean():.3f}  "
      f"|mean|={np.abs(suc_local_planner).mean():.3f}  "
      f"median={np.median(suc_local_planner):.3f}"
    )
  print()

  # ── 5. Raycast minimum analysis ──────────────────────────────────────────
  col_ray = np.array([r.min_raycast_at_end for r in collision_recs], dtype=float)
  suc_ray = np.array([r.min_raycast_at_end for r in success_recs], dtype=float)
  col_ray = col_ray[np.isfinite(col_ray)]
  suc_ray = suc_ray[np.isfinite(suc_ray)]

  print("── Raycast minimum (collision vs success) ──")
  print("  (smaller = obstacle closer; 0 = nearest, 1 = farthest)")
  if col_ray.size > 0:
    print(
      f"  Collision (n={col_ray.size}): "
      f"mean={col_ray.mean():.3f}  median={np.median(col_ray):.3f}  "
      f"p25={np.percentile(col_ray, 25):.3f}"
    )
  if suc_ray.size > 0:
    print(
      f"  Success (n={suc_ray.size}): "
      f"mean={suc_ray.mean():.3f}  median={np.median(suc_ray):.3f}  "
      f"p25={np.percentile(suc_ray, 25):.3f}"
    )
  print()


def print_collision_table(
  task_label: str,
  records: list[EpisodeRecord],
  num_episodes: int,
  num_goals: int,
  key_goal_indices: list[int],
) -> None:
  """Print the per-task collision distribution table."""
  collision_rows = compute_collision_table(records, num_goals, key_goal_indices)
  total_collisions = sum(r["collision_count"] for r in collision_rows)

  def _fmt_f(v: float, fmt: str = ".2f") -> str:
    return "N/A" if math.isnan(v) else format(v, fmt)

  print()
  print(
    f"  [{task_label}] Collision distribution "
    f"({total_collisions} collisions, {num_episodes} eps)"
  )
  col_hdrs = ["Waypoint", "Collisions", "Rate/eps", "% of all", "MeanX(m)", "MeanY(m)"]
  col_ws = [10, 10, 12, 10, 10, 10]

  hdr_parts = [_rpad(col_hdrs[i], col_ws[i]) for i in range(len(col_hdrs))]
  print("  " + " │ ".join(hdr_parts))
  print("  " + "─" * (sum(col_ws) + 5 * 3))

  for row in collision_rows:
    cnt = row["collision_count"]
    if cnt == 0 and total_collisions > 0:
      continue

    cells = [
      row["label"],
      str(cnt),
      f"{row['collision_rate_pct']:.1f}%",
      f"{row['collision_pct_of_all']:.1f}%",
      _fmt_f(row["mean_x"]),
      _fmt_f(row["mean_y"]),
    ]
    print("  " + " │ ".join(_rpad(cells[i], col_ws[i]) for i in range(len(cells))))

  if total_collisions == 0:
    print("  (no collision events)")


def _pad(s: str, width: int) -> str:
  """Left-aligned padding, CJK width aware (reuses _cjk_len)."""
  pad = max(0, width - _cjk_len(s))
  return s + " " * pad


def print_summary_table(
  task_names: list[str],
  stats_list: list[dict[str, float]],
) -> None:
  """Print the summary comparison table."""
  if not stats_list:
    return

  # Collect every metric key across all tasks (keep order, dedup)
  metric_keys: list[str] = []
  for s in stats_list:
    for k in s:
      if k not in metric_keys:
        metric_keys.append(k)
  col_width = 22  # width of each metric column
  name_width = 42  # width of the task name column

  sep = "─" * (name_width + col_width * len(metric_keys) + len(metric_keys) + 2)

  print()
  print("=" * len(sep))
  print("  Local planner network performance benchmark results (summary)")
  print("=" * len(sep))
  print()

  # Header row
  header = _pad("Task", name_width)
  for k in metric_keys:
    header += "│" + _pad(k, col_width)
  print(header)
  print(sep)

  # Data rows
  for name, stats in zip(task_names, stats_list, strict=True):
    row = _pad(name, name_width)
    for k in metric_keys:
      v = stats.get(k, float("nan"))
      if math.isnan(v):
        cell = "──"
      elif "%" in k:
        cell = f"{v:.1f}%"
      elif "steps" in k:
        cell = f"{v:.0f}"
      else:
        cell = f"{v:.2f}"
      row += "│" + _pad(cell, col_width)
    print(row)

  print(sep)
  print()

  # Supplementary notes
  print("Notes:")
  print("  success_rate           ── share of episodes reaching the last key_goal")
  print("  key_goal_1_reach_rate  ── share of episodes passing the obstacle to the")
  print("                            first key_goal")
  print("  mean_waypoints/N       ── waypoints completed at timeout (training tasks")
  print("                            only, N = total number of waypoints)")
  print("  termination_rate       ── share of hard terminations (fall/collision/wall)")
  print("                            that are not timeouts")
  print("  timeout_rate           ── share of episodes that neither succeeded nor")
  print("                            collided (ran out of time)")
  print("  distance_to_goal       ── mean distance to the current key goal over")
  print("                            timed-out episodes (m)")
  print("  mean_episode_steps     ── mean number of decision steps per episode")
  print()
  print("  ★ = key goal (key_goal); MeanX/Y: mean XY relative to spawn on collision")
  print("  Collision = terminated=True and sequence_complete=False (wall hits and")
  print("  falls included)")
  print()


# ──────────────────────────────────────────────────────────────────────────────
# Log saving
# ──────────────────────────────────────────────────────────────────────────────


def _sanitize_for_json(v: float) -> float | str:
  """Convert NaN/inf to strings for JSON standard compatibility."""
  if math.isnan(v):
    return "NaN"
  if math.isinf(v):
    return "Inf" if v > 0 else "-Inf"
  return v


def _build_collision_analysis(records: list[EpisodeRecord]) -> dict:
  """Build the collision cause summary (reused by JSON and Markdown)."""
  collision_recs = [r for r in records if r.hard_terminated]
  success_recs = [r for r in records if r.sequence_complete]

  col_vel = np.array([r.actual_velocity for r in collision_recs], dtype=float)
  suc_vel = np.array([r.actual_velocity for r in success_recs], dtype=float)
  col_spx = np.array([r.spawn_x_offset for r in collision_recs], dtype=float)
  col_local_planner = np.array(
    [r.local_planner_correction_at_end for r in collision_recs],
    dtype=float,
  )
  col_ray = np.array([r.min_raycast_at_end for r in collision_recs], dtype=float)
  col_vel = col_vel[np.isfinite(col_vel)]
  suc_vel = suc_vel[np.isfinite(suc_vel)]
  col_spx = col_spx[np.isfinite(col_spx)]
  col_local_planner = col_local_planner[np.isfinite(col_local_planner)]
  col_ray = col_ray[np.isfinite(col_ray)]

  # WP collision distribution
  wp_dist: dict[str, int] = {}
  for r in collision_recs:
    key = f"WP{r.collision_waypoint_idx}"
    wp_dist[key] = wp_dist.get(key, 0) + 1

  return {
    "total_collisions": len(collision_recs),
    "total_successes": len(success_recs),
    "wp_distribution": wp_dist,
    "collision_velocity": {
      "mean": float(col_vel.mean()) if col_vel.size > 0 else None,
      "median": float(np.median(col_vel)) if col_vel.size > 0 else None,
    },
    "success_velocity": {
      "mean": float(suc_vel.mean()) if suc_vel.size > 0 else None,
      "median": float(np.median(suc_vel)) if suc_vel.size > 0 else None,
    },
    "collision_spawn_x": {
      "mean": float(col_spx.mean()) if col_spx.size > 0 else None,
      "median": float(np.median(col_spx)) if col_spx.size > 0 else None,
    },
    "collision_local_planner_correction": {
      "mean": float(col_local_planner.mean()) if col_local_planner.size > 0 else None,
      "abs_mean": float(np.abs(col_local_planner).mean())
      if col_local_planner.size > 0
      else None,
      "active_avoid_count": int((np.abs(col_local_planner) > 0.3).sum())
      if col_local_planner.size > 0
      else 0,
      "passive_count": int((np.abs(col_local_planner) <= 0.3).sum())
      if col_local_planner.size > 0
      else 0,
    },
    "collision_min_raycast": {
      "mean": float(col_ray.mean()) if col_ray.size > 0 else None,
      "median": float(np.median(col_ray)) if col_ray.size > 0 else None,
    },
  }


def save_benchmark_log(
  log_dir: str,
  args: BenchmarkArgs,
  labels: list[str],
  task_configs: list[dict],
  all_records: list[list[EpisodeRecord]],
  all_stats: list[dict[str, float]],
  all_num_goals: list[int],
  all_key_goal_indices: list[list[int]],
) -> str:
  """Save the complete benchmark results as a JSON log.

  Returns:
    Path of the saved log file.
  """
  os.makedirs(log_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  log_path = os.path.join(log_dir, f"benchmark_local_planner_{timestamp}.json")

  log_data: dict = {
    "timestamp": timestamp,
    "args": {
      "locomotion_checkpoint": args.locomotion_checkpoint,
      "num_envs": args.num_envs,
      "num_episodes": args.num_episodes,
      "correction_gain": args.correction_gain,
      "raycast_obs_names": list(args.raycast_obs_names),
      "use_odometry": args.use_odometry,
      "include_joint_vel": args.include_joint_vel,
      "use_measured_velocity": args.use_measured_velocity,
      "use_ahrs_heading": args.use_ahrs_heading,
      "use_projected_yaw_rate": args.use_projected_yaw_rate,
      "seed": args.seed,
    },
    "tasks": [],
  }

  for stage_idx, (label, cfg, records, stats, ng, kgi) in enumerate(
    zip(
      labels,
      task_configs,
      all_records,
      all_stats,
      all_num_goals,
      all_key_goal_indices,
      strict=True,
    )
  ):
    # Collision distribution table (structured)
    collision_rows = compute_collision_table(records, ng, kgi)

    # Full record of every episode
    episode_dicts = []
    for r in records:
      d = asdict(r)
      for k, v in d.items():
        if isinstance(v, float):
          d[k] = _sanitize_for_json(v)
      episode_dicts.append(d)

    collision_analysis = _build_collision_analysis(records)

    # Episode classification
    wall_y = (
      args.wall_y_distances[stage_idx]
      if stage_idx < len(args.wall_y_distances)
      else 6.0
    )
    direct_max_steps = (
      args.success_direct_max_steps[stage_idx]
      if stage_idx < len(args.success_direct_max_steps)
      else 600
    )
    cls = classify_episodes(
      records, wall_y, args.wall_proximity_threshold, direct_max_steps
    )

    task_entry = {
      "label": label,
      "task": cfg["task"],
      "local_planner_checkpoint": cfg["local_planner_checkpoint"],
      "num_goals": ng,
      "key_goal_indices": kgi,
      "summary": {k: _sanitize_for_json(v) for k, v in stats.items()},
      "episode_classification": cls,
      "collision_analysis": collision_analysis,
      "collision_table": [
        {
          k: _sanitize_for_json(v) if isinstance(v, float) else v
          for k, v in row.items()
        }
        for row in collision_rows
      ],
      "episodes": episode_dicts,
    }
    log_data["tasks"].append(task_entry)

  with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=2)

  print(f"\nLog saved: {log_path}")
  return log_path


def save_markdown_report(
  log_dir: str,
  args: BenchmarkArgs,
  labels: list[str],
  task_configs: list[dict],
  all_records: list[list[EpisodeRecord]],
  all_stats: list[dict[str, float]],
  all_num_goals: list[int],
  all_key_goal_indices: list[list[int]],
) -> str:
  """Save the Markdown report (readable snapshot of the key terminal results)."""
  os.makedirs(log_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  md_path = os.path.join(log_dir, f"benchmark_local_planner_{timestamp}.md")

  # Summary header (collect every metric that appeared in any task)
  metric_keys: list[str] = []
  for s in all_stats:
    for k in s:
      if k not in metric_keys:
        metric_keys.append(k)

  def _fmt(v: float, key: str) -> str:
    if math.isnan(v):
      return "--"
    if "%" in key:
      return f"{v:.1f}%"
    if "steps" in key:
      return f"{v:.0f}"
    return f"{v:.2f}"

  def _opt_f(v: float | None, fmt: str = ".2f") -> str:
    return "--" if v is None else format(v, fmt)

  lines: list[str] = []
  lines.append("# Local planner network performance benchmark report")
  lines.append("")
  lines.append(f"- time: {timestamp}")
  lines.append(f"- locomotion checkpoint: {args.locomotion_checkpoint}")
  lines.append(f"- num_envs: {args.num_envs}")
  lines.append(f"- num_episodes: {args.num_episodes}")
  lines.append(f"- seed: {args.seed}")
  lines.append("")

  lines.append("## Summary")
  header = ["Task", *metric_keys]
  lines.append("| " + " | ".join(header) + " |")
  lines.append("|" + "|".join(["---"] * len(header)) + "|")
  for label, stats in zip(labels, all_stats, strict=True):
    cells = [label]
    for k in metric_keys:
      cells.append(_fmt(float(stats.get(k, float("nan"))), k))
    lines.append("| " + " | ".join(cells) + " |")
  lines.append("")

  for stage_idx, (label, cfg, records, ng, kgi) in enumerate(
    zip(
      labels,
      task_configs,
      all_records,
      all_num_goals,
      all_key_goal_indices,
      strict=True,
    )
  ):
    col_rows = compute_collision_table(records, ng, kgi)
    total_col = sum(r["collision_count"] for r in col_rows)
    n = len(records)
    n_succ = sum(1 for r in records if r.sequence_complete)
    wall_y = (
      args.wall_y_distances[stage_idx]
      if stage_idx < len(args.wall_y_distances)
      else 6.0
    )
    direct_max_steps = (
      args.success_direct_max_steps[stage_idx]
      if stage_idx < len(args.success_direct_max_steps)
      else 600
    )

    lines.append(f"## {label}")
    lines.append("")
    lines.append(f"- task: {cfg['task']}")
    lines.append(f"- local planner checkpoint: {cfg['local_planner_checkpoint']}")
    lines.append(f"- num_goals: {ng}, key_goal_indices: {kgi}")
    lines.append(f"- episodes: {n}, success: {n_succ}, collisions: {total_col}")
    lines.append(
      f"- wall_y: {wall_y:.1f}m, proximity threshold: {args.wall_proximity_threshold},"
      f" direct step threshold: {direct_max_steps}"
    )
    lines.append("")

    # ── Episode classification table ──
    cls = classify_episodes(
      records, wall_y, args.wall_proximity_threshold, direct_max_steps
    )
    lines.append("### Episode classification")
    lines.append("| Category | Count | Ratio | Description |")
    lines.append("|---|---:|---:|---|")
    for key in _CLASSIFICATION_ORDER:
      cnt = cls[key]
      label_text, desc = _CLASSIFICATION_LABELS[key]
      pct = cnt / n * 100 if n > 0 else 0
      lines.append(f"| {label_text} | {cnt} | {pct:.1f}% | {desc} |")
    pure_avoid = cls["avoidance_failure"]
    lines.append("")
    lines.append(
      f"> Pure avoidance failure rate: {pure_avoid}/{n} = {pure_avoid / n * 100:.1f}%"
      " (excluding spawn failures and post-wall wandering)"
    )
    lines.append("")

    analysis = _build_collision_analysis(records)
    active = analysis["collision_local_planner_correction"]["active_avoid_count"]
    passive = analysis["collision_local_planner_correction"]["passive_count"]
    n_col = int(analysis["total_collisions"])
    active_rate = (active / n_col * 100.0) if n_col > 0 else 0.0
    passive_rate = (passive / n_col * 100.0) if n_col > 0 else 0.0

    lines.append("### Collision cause summary")
    lines.append(
      f"- mean collision speed: {_opt_f(analysis['collision_velocity']['mean'])} m/s, "
      f"mean success speed: {_opt_f(analysis['success_velocity']['mean'])} m/s"
    )
    lines.append(
      f"- median collision speed: "
      f"{_opt_f(analysis['collision_velocity']['median'])} m/s, "
      f"median success speed: {_opt_f(analysis['success_velocity']['median'])} m/s"
    )
    lines.append(
      f"- mean local planner |correction| on collision: "
      f"{_opt_f(analysis['collision_local_planner_correction']['abs_mean'], '.3f')}"
    )
    lines.append(
      f"- active local planner correction on collision (|corr|>0.3): "
      f"{active}/{n_col} ({active_rate:.1f}%)"
      " — saw the obstacle but still collided"
    )
    lines.append(
      f"- low local planner correction on collision (|corr|<=0.3): "
      f"{passive}/{n_col} ({passive_rate:.1f}%)"
      " — did not effectively detect the obstacle"
    )
    lines.append(
      f"- mean minimum raycast on collision: "
      f"{_opt_f(analysis['collision_min_raycast']['mean'], '.3f')}, "
      f"median: {_opt_f(analysis['collision_min_raycast']['median'], '.3f')}"
    )
    lines.append(
      f"- mean Spawn X offset on collision: "
      f"{_opt_f(analysis['collision_spawn_x']['mean'])} m"
    )
    lines.append("")

    lines.append("### Collision distribution")
    if ng <= 2:
      lines.append(
        f"> NoPP mode ({ng} goals); all collisions happen on the way to the first goal."
      )
      lines.append("")
    lines.append(
      "| Waypoint | Collisions | Rate/eps | % of all | MeanX(m) | MeanY(m) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in col_rows:
      if row["collision_count"] == 0 and total_col > 0:
        continue
      mx = row["mean_x"]
      my = row["mean_y"]
      mx_s = "--" if math.isnan(mx) else f"{mx:.2f}"
      my_s = "--" if math.isnan(my) else f"{my:.2f}"
      lines.append(
        "| "
        + " | ".join(
          [
            str(row["label"]),
            str(row["collision_count"]),
            f"{row['collision_rate_pct']:.1f}%",
            f"{row['collision_pct_of_all']:.1f}%",
            mx_s,
            my_s,
          ]
        )
        + " |"
      )
    lines.append("")

  with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

  print(f"Report saved: {md_path}")
  return md_path


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
  args: BenchmarkArgs = tyro.cli(BenchmarkArgs, config=mjlab.TYRO_FLAGS)

  device = torch.device(args.device)
  if args.use_odometry and args.include_joint_vel:
    print("[ODOM] --use-odometry is best paired with --no-include-joint-vel.")

  # ── Task configuration table ────────────────────────────────────────────────
  task_configs = [
    {
      "label": "Stage1 LowSpeed",
      "task": args.stage1_task,
      "local_planner_checkpoint": args.stage1_local_planner_checkpoint,
    },
    {
      "label": "Stage2 Indoor",
      "task": args.stage2_task,
      "local_planner_checkpoint": args.stage2_local_planner_checkpoint,
    },
  ]

  # ── Drop stages that have no checkpoint provided ─────────────────────────
  task_configs = [cfg for cfg in task_configs if cfg["local_planner_checkpoint"]]

  # ── Validate checkpoints ───────────────────────────────────────────────────
  missing = []
  if not args.locomotion_checkpoint or not os.path.isfile(args.locomotion_checkpoint):
    missing.append(f"locomotion_checkpoint: {args.locomotion_checkpoint}")
  for cfg in task_configs:
    ckpt = cfg["local_planner_checkpoint"]
    if not os.path.isfile(ckpt):
      missing.append(f"{cfg['label']} local_planner_checkpoint: {ckpt}")
  if not task_configs:
    print("Error: at least one stage local_planner_checkpoint is required.")
    raise SystemExit(1)
  if missing:
    print("Error: the following checkpoint files do not exist:")
    for m in missing:
      print(f"  {m}")
    raise SystemExit(1)

  # ── Run the benchmark task by task ──────────────────────────────────────────
  all_stats: list[dict[str, float]] = []
  all_records: list[list[EpisodeRecord]] = []
  all_num_goals: list[int] = []
  all_key_goal_indices: list[list[int]] = []
  labels: list[str] = []

  for cfg in task_configs:
    print()
    print(f"{'─' * 60}")
    print(f"  [{cfg['label']}] {cfg['task']}")
    print(f"  Local planner ckpt: {cfg['local_planner_checkpoint']}")
    print(f"{'─' * 60}")

    records, num_goals, key_goal_indices = run_task_benchmark(
      task=cfg["task"],
      locomotion_checkpoint=args.locomotion_checkpoint,
      local_planner_checkpoint=cfg["local_planner_checkpoint"],
      num_envs=args.num_envs,
      num_episodes=args.num_episodes,
      raycast_obs_names=args.raycast_obs_names,
      device=device,
      correction_gain=args.correction_gain,
      use_odometry=args.use_odometry,
      include_joint_vel=args.include_joint_vel,
      use_measured_velocity=args.use_measured_velocity,
      use_ahrs_heading=args.use_ahrs_heading,
      use_projected_yaw_rate=args.use_projected_yaw_rate,
      seed=args.seed,
    )

    stats = compute_stats(records, num_goals=num_goals)
    all_stats.append(stats)
    all_records.append(records)
    all_num_goals.append(num_goals)
    all_key_goal_indices.append(key_goal_indices)
    labels.append(cfg["label"])

    # Print this task's results immediately
    print(f"\n  [{cfg['label']}] preliminary statistics:")
    for k, v in stats.items():
      if "%" in k:
        print(f"    {k}: {v:.1f}%")
      elif "steps" in k:
        print(f"    {k}: {v:.0f}")
      else:
        print(f"    {k}: {v:.2f}")

  # ── Print the summary table ────────────────────────────────────────────────
  print_summary_table(labels, all_stats)

  # ── Per task: collision table + cause analysis + episode classification ────
  for stage_idx, (label, records, num_goals, key_goal_indices) in enumerate(
    zip(labels, all_records, all_num_goals, all_key_goal_indices, strict=True)
  ):
    print_collision_table(
      task_label=label,
      records=records,
      num_episodes=len(records),
      num_goals=num_goals,
      key_goal_indices=key_goal_indices,
    )
    analyze_collision_causes(records, num_goals, key_goal_indices)

    wall_y = (
      args.wall_y_distances[stage_idx]
      if stage_idx < len(args.wall_y_distances)
      else 6.0
    )
    direct_max_steps = (
      args.success_direct_max_steps[stage_idx]
      if stage_idx < len(args.success_direct_max_steps)
      else 600
    )
    cls = classify_episodes(
      records, wall_y, args.wall_proximity_threshold, direct_max_steps
    )
    print_episode_classification(
      label, cls, wall_y, args.wall_proximity_threshold, direct_max_steps
    )

  # ── Save the JSON log ──────────────────────────────────────────────────────
  save_benchmark_log(
    log_dir=args.log_dir,
    args=args,
    labels=labels,
    task_configs=task_configs,
    all_records=all_records,
    all_stats=all_stats,
    all_num_goals=all_num_goals,
    all_key_goal_indices=all_key_goal_indices,
  )

  # ── Save the Markdown report (readable version of the summary) ─────────────
  save_markdown_report(
    log_dir=args.log_dir,
    args=args,
    labels=labels,
    task_configs=task_configs,
    all_records=all_records,
    all_stats=all_stats,
    all_num_goals=all_num_goals,
    all_key_goal_indices=all_key_goal_indices,
  )


if __name__ == "__main__":
  main()
