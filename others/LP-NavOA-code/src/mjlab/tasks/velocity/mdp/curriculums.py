from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def terrain_levels_goal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Terrain level curriculum based on goal tracking progress.

  Similar to terrain_levels_vel, this function uses target_velocity from
  the goal tracking command to calculate expected distance and adjust
  terrain difficulty accordingly.

  Args:
    env: The environment instance.
    env_ids: Environment indices being reset.
    command_name: Name of the goal tracking command.
    asset_cfg: Scene entity configuration for the robot.

  Returns:
    Mean terrain level across all environments.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  # Get the goal tracking command term to access target_velocity
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None

  # Compute the distance the robot walked from origin
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains
  move_up = distance > terrain_generator.size[0] / 2

  # Use target_velocity to compute expected distance (similar to terrain_levels_vel)
  # command[:, 2] is target_velocity in GoalTrackingCommand
  target_velocity = torch.abs(command_term.command[env_ids, 2])
  expected_distance = target_velocity * env.max_episode_length_s * 0.5
  move_down = distance < expected_distance
  move_down *= ~move_up

  # Update terrain levels
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def terrain_levels_reachable_goal(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Performance-driven terrain curriculum for ReachableGoalCommand.

  ReachableGoalCommand provides command tensor as [delta_yaw, target_velocity].
  This helper mirrors ``terrain_levels_goal`` but reads target velocity from
  index 1 instead of index 2.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  move_up = distance > terrain_generator.size[0] / 2

  target_velocity = torch.abs(command_term.command[env_ids, 1])
  expected_distance = target_velocity * env.max_episode_length_s * 0.5
  move_down = distance < expected_distance
  move_down *= ~move_up

  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


class GoalVelocityStage(TypedDict):
  step: int
  target_velocity_range: tuple[float, float]


def goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[GoalVelocityStage],
) -> dict[str, torch.Tensor]:
  """Goal tracking velocity curriculum.

  Updates the target_velocity_range of the goal tracking command based on
  training step count.

  Args:
    env: The environment instance.
    env_ids: Environment indices (unused, kept for API compatibility).
    command_name: Name of the goal tracking command.
    velocity_stages: List of {step, target_velocity_range} dicts.

  Returns:
    Dictionary with current velocity range values.
  """
  del env_ids  # Unused.

  from mjlab.tasks.velocity.mdp.goal_command import GoalTrackingCommandCfg

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(GoalTrackingCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      cfg.target_velocity_range = stage["target_velocity_range"]

  # Only update config range - new target velocities will be sampled
  # when commands are resampled (on reset or resample timeout).
  # This matches the behavior of commands_vel for velocity tracking.
  vel_min, vel_max = cfg.target_velocity_range

  return {
    "target_vel_min": torch.tensor(vel_min),
    "target_vel_max": torch.tensor(vel_max),
  }


def fixed_sequential_goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[GoalVelocityStage],
) -> dict[str, torch.Tensor]:
  """Fixed sequential goal velocity curriculum.

  Updates ``FixedSequentialGoalCommandCfg.target_velocity_range`` based on
  training step count. This mirrors ``goal_velocity`` for the fixed waypoint
  command used by goal-tracking ablations.
  """
  del env_ids  # Unused.

  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
  )

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(FixedSequentialGoalCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      cfg.target_velocity_range = stage["target_velocity_range"]

  target_velocity_range = cfg.target_velocity_range
  if target_velocity_range is None:
    vel_min = vel_max = cfg.target_velocity
  else:
    vel_min, vel_max = target_velocity_range

  return {
    "target_vel_min": torch.tensor(vel_min),
    "target_vel_max": torch.tensor(vel_max),
  }


class TerrainLevelStage(TypedDict):
  step: int
  level: int


def terrain_levels_by_step(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  level_stages: list[TerrainLevelStage],
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  """Force terrain level based on training step count.

  This curriculum function sets terrain levels directly based on iteration count,
  useful for obstacle avoidance training where difficulty stages are time-based.

  Args:
    env: The environment instance.
    env_ids: Environment indices being reset.
    command_name: Name of the velocity command (for compatibility).
    level_stages: List of {step, level} dicts defining when to advance levels.
    asset_cfg: Scene entity configuration for the robot.

  Returns:
    Mean terrain level across all environments.
  """
  del command_name, asset_cfg  # Unused but kept for API compatibility.

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  # Determine target level based on current step
  target_level = 0
  for stage in level_stages:
    if env.common_step_counter >= stage["step"]:
      target_level = stage["level"]

  # Clamp to valid range
  max_level = terrain_generator.num_rows - 1
  target_level = min(target_level, max_level)

  # Force all resetting environments to the target level
  current_levels = terrain.terrain_levels[env_ids]
  move_up = current_levels < target_level
  move_down = current_levels > target_level

  # Repeat until at target level (update_env_origins moves by 1 step)
  for _ in range(terrain_generator.num_rows):
    terrain.update_env_origins(env_ids, move_up, move_down)
    current_levels = terrain.terrain_levels[env_ids]
    move_up = current_levels < target_level
    move_down = current_levels > target_level

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


class CircularGoalStage(TypedDict, total=False):
  """Type definition for circular goal curriculum stage."""

  step: int
  target_velocity_range: tuple[float, float]
  goal_angle_range: float  # Radians, optional


def circular_goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[CircularGoalStage],
) -> dict[str, torch.Tensor]:
  """Circular goal tracking velocity and angle curriculum.

  Updates both target_velocity_range and goal_angle_range of the
  circular goal tracking command based on training step count.

  Args:
    env: The environment instance.
    env_ids: Environment indices (unused, kept for API compatibility).
    command_name: Name of the circular goal tracking command.
    velocity_stages: List of {step, target_velocity_range, goal_angle_range} dicts.
      goal_angle_range is optional; if not specified, it won't be updated.

  Returns:
    Dictionary with current velocity and angle range values.
  """
  del env_ids  # Unused.

  from mjlab.tasks.velocity.mdp.circular_goal_command import (
    CircularGoalTrackingCommandCfg,
  )

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(CircularGoalTrackingCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    step = stage.get("step")
    target_velocity_range = stage.get("target_velocity_range")
    if step is not None and env.common_step_counter > step:
      if target_velocity_range is not None:
        cfg.target_velocity_range = target_velocity_range
      if "goal_angle_range" in stage:
        cfg.goal_angle_range = stage["goal_angle_range"]

  vel_min, vel_max = cfg.target_velocity_range
  angle_range_deg = math.degrees(cfg.goal_angle_range)

  return {
    "target_vel_min": torch.tensor(vel_min),
    "target_vel_max": torch.tensor(vel_max),
    "goal_angle_range_deg": torch.tensor(angle_range_deg),
  }


class EventParamStage(TypedDict, total=False):
  """Type definition for event parameter curriculum stage."""

  step: int
  # Any additional keys are event parameters to update


def modify_event_params(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  param_stages: list[dict],
) -> dict[str, torch.Tensor]:
  """Modify event term parameters based on training step count.

  Updates parameters of an event term (like launch_ball) based on
  curriculum stages. This allows gradual increase in difficulty
  for tasks like ball dodge.

  Args:
    env: The environment instance.
    env_ids: Environment indices (unused, kept for API compatibility).
    event_name: Name of the event term to modify.
    param_stages: List of dicts with 'step' and parameter key-value pairs.
      Example: [{"step": 0, "ball_speed_range": (2.0, 4.0)},
                {"step": 1000, "ball_speed_range": (3.0, 5.0)}]

  Returns:
    Dictionary with current parameter values for logging.
  """
  del env_ids  # Unused.

  event_cfg = env.event_manager.get_term_cfg(event_name)

  current_step = env.common_step_counter
  applied_stage = None

  for stage in param_stages:
    if current_step >= stage["step"]:
      applied_stage = stage

  if applied_stage is not None:
    for key, value in applied_stage.items():
      if key != "step" and key in event_cfg.params:
        event_cfg.params[key] = value

  # Return current parameter values for logging
  result = {}
  if "ball_speed_range" in event_cfg.params:
    result["ball_speed_min"] = torch.tensor(event_cfg.params["ball_speed_range"][0])
    result["ball_speed_max"] = torch.tensor(event_cfg.params["ball_speed_range"][1])
  if "launch_interval_range" in event_cfg.params:
    result["interval_min"] = torch.tensor(event_cfg.params["launch_interval_range"][0])
    result["interval_max"] = torch.tensor(event_cfg.params["launch_interval_range"][1])

  return result


# ==============================================================================
# Reachable Goal Curriculum
# ==============================================================================


class ReachableGoalStage(TypedDict, total=False):
  """Type definition for reachable goal curriculum stage."""

  step: int
  target_velocity_range: tuple[float, float]
  goal_angle_range: float  # Radians, optional


def reachable_goal_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[ReachableGoalStage],
) -> dict[str, torch.Tensor]:
  """Reachable goal tracking velocity and angle curriculum.

  Updates both target_velocity_range and goal_angle_range of the
  reachable goal tracking command based on training step count.

  Args:
    env: The environment instance.
    env_ids: Environment indices (unused, kept for API compatibility).
    command_name: Name of the reachable goal tracking command.
    velocity_stages: List of {step, target_velocity_range, goal_angle_range} dicts.
      goal_angle_range is optional; if not specified, it won't be updated.

  Returns:
    Dictionary with current velocity and angle range values.
  """
  del env_ids  # Unused.

  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    ReachableGoalCommandCfg,
  )

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(ReachableGoalCommandCfg, command_term.cfg)

  for stage in velocity_stages:
    step = stage.get("step")
    target_velocity_range = stage.get("target_velocity_range")
    if step is not None and env.common_step_counter > step:
      if target_velocity_range is not None:
        cfg.target_velocity_range = target_velocity_range
      if "goal_angle_range" in stage:
        cfg.goal_angle_range = stage["goal_angle_range"]

  vel_min, vel_max = cfg.target_velocity_range
  angle_range_deg = math.degrees(cfg.goal_angle_range)

  return {
    "target_vel_min": torch.tensor(vel_min),
    "target_vel_max": torch.tensor(vel_max),
    "goal_angle_range_deg": torch.tensor(angle_range_deg),
  }
