from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from .safety_velocity import (
  compute_directional_approach_speeds,
  compute_safe_speed_limit,
  min_raycast_clearance,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_command_speed(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Extract effective speed from any command type.

  Works with:
  - Velocity commands: sqrt(lin_x^2 + lin_y^2) + |ang_z|
  - CircularGoalTrackingCommand: target_velocity
  - ReachableGoalCommand: target_velocity
  - GoalTrackingCommand: target_velocity

  Returns:
    Tensor of shape [num_envs] with speed values.
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import (
    KeyboardCircularGoalTrackingCommand,
    KeyboardGoalTrackingCommand,
  )
  from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  if command_term is None:
    return torch.zeros(env.num_envs, device=env.device)

  if isinstance(
    command_term,
    (
      CircularGoalTrackingCommand,
      KeyboardCircularGoalTrackingCommand,
      ReachableGoalCommand,
      FixedSequentialGoalCommand,
    ),
  ):
    target_vel = command_term.target_velocity
    if isinstance(target_vel, float):
      return torch.full((env.num_envs,), target_vel, device=env.device)
    return target_vel
  elif isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)):
    return command_term.target_velocity
  else:
    # Standard velocity command
    command = env.command_manager.get_command(command_name)
    if command is None:
      return torch.zeros(env.num_envs, device=env.device)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    return linear_norm + angular_norm


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_linear_velocity_z(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping z-axis linear velocity close to zero.

  This is useful for goal tracking tasks where xy velocity is controlled
  by goal direction but z velocity should remain zero (no vertical motion).
  """
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_lin_vel_b
  z_error = torch.square(actual[:, 2])
  return torch.exp(-z_error / std**2)


def track_angular_velocity_xy(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping xy-axis angular velocities close to zero.

  This penalizes roll and pitch angular velocities, which helps prevent
  the robot from wobbling/oscillating up and down during locomotion.
  This is especially important for goal tracking tasks where the yaw
  is controlled separately but xy angular velocities should be minimized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-xy_error / std**2)


def track_angular_velocity_z(
  env: ManagerBasedRlEnv,
  std: float,
  threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for keeping z-axis angular velocity smooth and controlled.

  This penalizes excessive yaw angular velocity to encourage smooth turning.
  Unlike heading_rate_penalty which penalizes heading changes towards a target,
  this simply limits how fast the robot can rotate around the z-axis.

  This is useful for goal tracking with obstacle avoidance where:
  - The robot needs to turn to face goals (tracking_yaw reward)
  - The robot needs to turn to avoid obstacles
  - But all turns should be smooth, not jerky

  Mathematical form: r = exp(-max(0, |omega_z| - threshold)^2 / std^2)
  - When |omega_z| <= threshold: reward = 1 (no penalty, free to turn)
  - When |omega_z| > threshold: reward decays exponentially

  Args:
    env: The environment instance.
    std: Standard deviation controlling sensitivity beyond threshold.
    threshold: Angular velocity threshold (rad/s) below which no penalty applies.
      Default 0.5 rad/s (~29 deg/s) allows normal turning without penalty.
    asset_cfg: Configuration for the robot entity.
  """
  asset: Entity = env.scene[asset_cfg.name]
  actual = asset.data.root_link_ang_vel_b
  omega_z_abs = torch.abs(actual[:, 2])
  # Only penalize angular velocity exceeding the threshold
  excess = torch.clamp(omega_z_abs - threshold, min=0.0)
  z_error = torch.square(excess)
  return torch.exp(-z_error / std**2)


def track_lateral_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for minimizing lateral velocity perpendicular to goal direction.

  For goal tracking tasks, this penalizes velocity components that are
  perpendicular to the direction towards the goal. This encourages the
  robot to move directly towards the goal without excessive side-to-side
  motion, resulting in smoother and more efficient locomotion.
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  asset: Entity = env.scene[asset_cfg.name]

  # Get relative position to goal
  # KeyboardGoalTrackingCommand stores goal_pos_rel in robot frame
  if isinstance(command_term, KeyboardGoalTrackingCommand):
    target_pos_rel = command_term.goal_pos_rel  # [num_envs, 2] in robot frame
    # Transform robot velocity to robot frame for consistency
    cur_vel = asset.data.root_link_lin_vel_b[:, :2]  # [num_envs, 2]
  else:
    target_pos_rel = command_term.target_pos_rel  # [num_envs, 2] in world frame
    cur_vel = asset.data.root_link_lin_vel_w[:, :2]  # [num_envs, 2]

  norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
  target_vec_norm = target_pos_rel / (norm + 1e-5)

  # Compute lateral velocity (perpendicular to goal direction)
  # Perpendicular direction: rotate target_vec_norm by 90 degrees
  perp_vec = torch.stack([-target_vec_norm[:, 1], target_vec_norm[:, 0]], dim=-1)
  lateral_vel = torch.abs(torch.sum(perp_vec * cur_vel, dim=-1))  # [num_envs]

  # Use exponential reward for smooth gradients
  lateral_error = torch.square(lateral_vel)
  return torch.exp(-lateral_error / (std**2))


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def flat_orientation(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_collision_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize body collisions (excluding feet).

  This reward penalizes collisions of body parts like thigh, shin, hip, and torso
  with any object (ground, obstacles, etc.). Based on ABS training reward design.

  Args:
    env: The environment.
    sensor_name: Name of the contact sensor that tracks body collisions.

  Returns:
    Tensor of shape [num_envs] with collision count (number of body parts in contact).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  # Check if any body part has contact force above threshold
  # Shape: [num_envs, num_bodies, 3]
  contact_force_magnitude = torch.norm(sensor.data.force, dim=-1)
  # Count bodies with significant contact (force > 0.1 N)
  in_contact = (contact_force_magnitude > 0.1).float()
  return torch.sum(in_contact, dim=-1)


def feet_collision_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Penalize feet horizontal collisions (kicking obstacles/walls).

  This reward detects when feet hit obstacles with large horizontal forces,
  such as kicking a step or wall. Based on ABS training reward design.

  The penalty triggers when horizontal force > 2 * vertical force + 10.0,
  indicating the foot is pushing sideways against something rather than
  standing on ground.

  Args:
    env: The environment.
    sensor_name: Name of the contact sensor that tracks feet contact forces.

  Returns:
    Tensor of shape [num_envs] with 1.0 if any foot has horizontal collision, 0.0 otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  # Force shape: [num_envs, num_feet, 3]
  # Horizontal force: XY components
  hor_footforce = torch.norm(sensor.data.force[:, :, 0:2], dim=-1)
  # Vertical force: Z component (absolute value)
  ver_footforce = torch.abs(sensor.data.force[:, :, 2])
  # Detect horizontal collision: horizontal > 2 * vertical + 10
  foot_hor_col = torch.any(hor_footforce > 2 * ver_footforce + 10.0, dim=-1)
  return foot_hor_col.float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    total_command = _get_command_speed(env, command_name)
    scale = (total_command > command_threshold).float()
    reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    total_command = _get_command_speed(env, command_name)
    active = (total_command > command_threshold).float()
    cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    total_command = _get_command_speed(env, command_name)
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  total_command = _get_command_speed(env, command_name)
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    total_command = _get_command_speed(env, command_name)
    active = (total_command > command_threshold).float()
    cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.
    from .circular_goal_command import CircularGoalTrackingCommand
    from .keyboard_command import KeyboardCircularGoalTrackingCommand
    from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

    asset: Entity = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None

    # Handle different command types
    if isinstance(
      command_term,
      (
        CircularGoalTrackingCommand,
        KeyboardCircularGoalTrackingCommand,
        ReachableGoalCommand,
        FixedSequentialGoalCommand,
      ),
    ):
      # CircularGoalTrackingCommand/ReachableGoalCommand/FixedSequentialGoalCommand:
      # use target_velocity directly
      total_speed = command_term.target_velocity
      if isinstance(total_speed, float):
        total_speed = torch.full((env.num_envs,), total_speed, device=env.device)
    else:
      # Standard velocity command: [lin_x, lin_y, ang_z]
      command = env.command_manager.get_command(command_name)
      assert command is not None
      linear_speed = torch.norm(command[:, :2], dim=1)
      angular_speed = torch.abs(command[:, 2])
      total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


# ============================================================================
# Circular Goal Tracking Rewards
# ============================================================================


def circular_goal_tracking_yaw(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  std: float = 0.5,
) -> torch.Tensor:
  """Reward for tracking heading towards the goal.

  Works with CircularGoalTrackingCommand and ReachableGoalCommand which
  provide stable delta_yaw because the goal angle is in world frame.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    std: Standard deviation for the exponential reward.

  Returns:
    Reward tensor of shape [num_envs] in range [0, 1].
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand
  from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(
    command_term,
    (
      CircularGoalTrackingCommand,
      KeyboardCircularGoalTrackingCommand,
      ReachableGoalCommand,
      FixedSequentialGoalCommand,
    ),
  ), (
    f"Command '{command_name}' is not a CircularGoalTrackingCommand, "
    f"KeyboardCircularGoalTrackingCommand, ReachableGoalCommand, or FixedSequentialGoalCommand."
  )

  delta_yaw = command_term.delta_yaw  # [num_envs]
  return torch.exp(-torch.square(delta_yaw) / (std**2))


def circular_goal_tracking_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking velocity towards the goal.

  Unlike waypoint-based tracking, this rewards forward velocity in the
  direction of the goal.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    std: Standard deviation for the exponential reward.
    asset_cfg: Configuration for the robot entity.

  Returns:
    Reward tensor of shape [num_envs] in range [0, 1].
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand
  from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(
    command_term,
    (
      CircularGoalTrackingCommand,
      KeyboardCircularGoalTrackingCommand,
      ReachableGoalCommand,
      FixedSequentialGoalCommand,
    ),
  ), (
    f"Command '{command_name}' is not a CircularGoalTrackingCommand, "
    f"KeyboardCircularGoalTrackingCommand, ReachableGoalCommand, or FixedSequentialGoalCommand."
  )

  asset: Entity = env.scene[asset_cfg.name]

  # Get robot forward velocity
  lin_vel_b = asset.data.root_link_lin_vel_b  # [num_envs, 3]

  # Get delta_yaw (angle from robot heading to goal direction)
  delta_yaw = command_term.delta_yaw  # [num_envs]

  # Project velocity onto goal direction
  # In robot frame: goal direction is at angle delta_yaw from forward
  vel_towards_goal = lin_vel_b[:, 0] * torch.cos(delta_yaw) + lin_vel_b[
    :, 1
  ] * torch.sin(delta_yaw)

  # Target velocity
  target_vel = command_term.target_velocity  # [num_envs]
  if isinstance(target_vel, float):
    target_vel = torch.full((env.num_envs,), target_vel, device=env.device)

  # Compute error
  vel_error = torch.square(vel_towards_goal - target_vel)

  return torch.exp(-vel_error / (std**2))


def circular_goal_ang_vel_tracking(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  std: float = 0.5,
  heading_control_stiffness: float = 0.5,
  max_ang_vel: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking angular velocity command derived from heading error.

  This is the KEY reward for stable training. Instead of directly penalizing
  the heading error (which produces noisy gradients), we:
  1. Convert heading error to an angular velocity command
  2. Clip it to a reasonable range
  3. Reward matching the actual angular velocity to this command

  This provides the SAME smooth training signal as velocity tracking,
  eliminating the oscillation problem.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    std: Standard deviation for the exponential reward.
    heading_control_stiffness: Gain for heading error to ang_vel conversion.
    max_ang_vel: Maximum angular velocity command [rad/s].
    asset_cfg: Configuration for the robot entity.

  Returns:
    Reward tensor of shape [num_envs] in range [0, 1].
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand
  from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(
    command_term,
    (
      CircularGoalTrackingCommand,
      KeyboardCircularGoalTrackingCommand,
      ReachableGoalCommand,
      FixedSequentialGoalCommand,
    ),
  ), (
    f"Command '{command_name}' is not a CircularGoalTrackingCommand, "
    f"KeyboardCircularGoalTrackingCommand, ReachableGoalCommand, or FixedSequentialGoalCommand."
  )

  asset: Entity = env.scene[asset_cfg.name]

  # Convert heading error to angular velocity command
  delta_yaw = command_term.delta_yaw  # [num_envs]
  ang_vel_command = torch.clamp(
    heading_control_stiffness * delta_yaw,
    min=-max_ang_vel,
    max=max_ang_vel,
  )

  # Get actual angular velocity
  actual_ang_vel_z = asset.data.root_link_ang_vel_b[:, 2]

  # Compute error
  ang_vel_error = torch.square(ang_vel_command - actual_ang_vel_z)

  return torch.exp(-ang_vel_error / (std**2))


def in_place_turn_bonus(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  heading_threshold: float = 0.8,
  max_ang_vel: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for turning in place when heading error is large.

  Provides an additional reward signal specifically for rotating toward the goal
  when the heading error exceeds ``heading_threshold``.  The bonus scales
  linearly from zero at the threshold up to 1.0 at ±π, and is further
  multiplied by how well the robot is actually rotating in the goal direction.

  This complements ``circular_goal_ang_vel_tracking`` which rewards angular
  velocity tracking at all times, but does not amplify the signal for large
  heading errors that demand in-place rotation rather than forward travel.

  Reward::

    gate         = clamp((|delta_yaw| − threshold) / (π − threshold), 0, 1)
    correct_turn = clamp(ω_z · sign(delta_yaw) / max_ang_vel,         0, 1)
    reward       = gate · correct_turn

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    heading_threshold: Heading error below which no bonus is given [rad].
      Default 0.8 rad (≈ 46°).
    max_ang_vel: Angular velocity normalisation factor [rad/s]. Default 1.0.
    asset_cfg: Configuration for the robot entity.

  Returns:
    Reward tensor of shape [num_envs] in range [0, 1].
  """
  import math

  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand
  from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(
    command_term,
    (
      CircularGoalTrackingCommand,
      KeyboardCircularGoalTrackingCommand,
      ReachableGoalCommand,
      FixedSequentialGoalCommand,
    ),
  ), (
    f"Command '{command_name}' is not a recognised goal command type. "
    f"Got {type(command_term).__name__}."
  )

  asset: Entity = env.scene[asset_cfg.name]
  delta_yaw = command_term.delta_yaw  # [num_envs], range [−π, π]
  actual_ang_vel_z = asset.data.root_link_ang_vel_b[:, 2]  # [num_envs]

  # Gate: 0 when |delta_yaw| <= threshold, grows linearly to 1 at ±π.
  denominator = math.pi - heading_threshold
  gate = ((delta_yaw.abs() - heading_threshold) / denominator).clamp(min=0.0, max=1.0)

  # Correct-turn ratio: how well the robot rotates toward the goal.
  # 1.0 when turning at max_ang_vel in the right direction, 0 otherwise.
  correct_turn = (actual_ang_vel_z * delta_yaw.sign() / max_ang_vel).clamp(
    min=0.0, max=1.0
  )

  return gate * correct_turn


# ============================================================================
# Goal Tracking Rewards
# ============================================================================


def tracking_goal_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking velocity towards the goal.

  Uses exponential reward similar to track_linear_velocity for smooth gradients
  and stable training. Computes error between current velocity projection
  onto goal direction and target velocity magnitude.

  The target_velocity represents the desired speed (magnitude) at which the
  robot should move towards the goal. It is always non-negative.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    std: Standard deviation for the exponential reward (controls sensitivity).
    asset_cfg: Configuration for the robot entity.

  Returns:
    Reward tensor of shape [num_envs] in range [0, 1].
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  asset: Entity = env.scene[asset_cfg.name]

  # Get relative position to goal
  # KeyboardGoalTrackingCommand stores goal_pos_rel in robot frame
  if isinstance(command_term, KeyboardGoalTrackingCommand):
    target_pos_rel = command_term.goal_pos_rel  # [num_envs, 2] in robot frame
    # Transform robot velocity to robot frame for consistency
    cur_vel = asset.data.root_link_lin_vel_b[:, :2]  # [num_envs, 2]
  else:
    target_pos_rel = command_term.target_pos_rel  # [num_envs, 2] in world frame
    cur_vel = asset.data.root_link_lin_vel_w[:, :2]  # [num_envs, 2]

  norm = torch.norm(target_pos_rel, dim=-1, keepdim=True)
  target_vec_norm = target_pos_rel / (norm + 1e-5)

  # Project velocity onto goal direction
  vel_projection = torch.sum(target_vec_norm * cur_vel, dim=-1)  # [num_envs]

  # Get target velocity magnitude (speed towards goal, always >= 0)
  target_vel = command_term.target_velocity  # [num_envs]

  # Compute velocity error (difference between projected and target)
  vel_error = torch.square(vel_projection - target_vel)

  # Use exponential reward for smooth gradients (same form as track_linear_velocity)
  reward = torch.exp(-vel_error / (std**2))

  return reward


def tracking_yaw(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  std: float = 0.5,
) -> torch.Tensor:
  """Reward for tracking the heading towards the goal.

  Based on extreme-parkour's _reward_tracking_yaw:
  Computes exponential reward based on heading error.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    std: Standard deviation for the exponential reward.

  Returns:
    Reward tensor of shape [num_envs].
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  delta_yaw = command_term.delta_yaw  # [num_envs]
  return torch.exp(-torch.square(delta_yaw) / (std**2))


def tracking_ang_vel_yaw(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  std: float = 0.5,
  heading_control_stiffness: float = 1.0,
  max_ang_vel: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking yaw via angular velocity command (like velocity tracking).

  This converts heading error to an angular velocity command, then rewards
  tracking that command. This is the SAME mechanism used in velocity tracking
  with heading_command=True, which produces much smoother training curves.

  The key insight is that velocity tracking doesn't directly penalize heading error.
  Instead, it:
  1. Converts heading_error to ang_vel_command = stiffness * heading_error
  2. Clips the command to [-max_ang_vel, max_ang_vel]
  3. Rewards matching actual ang_vel to the command

  This provides a smooth, bounded control signal instead of directly penalizing
  the heading error which changes unpredictably as the robot moves.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    std: Standard deviation for the exponential reward.
    heading_control_stiffness: Gain for converting heading error to ang_vel command.
      Higher values = more aggressive turning. Default 1.0.
    max_ang_vel: Maximum angular velocity command magnitude [rad/s]. Default 1.0.
    asset_cfg: Configuration for the robot entity.

  Returns:
    Reward tensor of shape [num_envs] in range [0, 1].
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  asset: Entity = env.scene[asset_cfg.name]

  # Convert heading error to angular velocity command (same as velocity tracking)
  delta_yaw = command_term.delta_yaw  # [num_envs]
  ang_vel_command = torch.clamp(
    heading_control_stiffness * delta_yaw,
    min=-max_ang_vel,
    max=max_ang_vel,
  )

  # Get actual angular velocity
  actual_ang_vel_z = asset.data.root_link_ang_vel_b[:, 2]

  # Compute error between command and actual (same form as track_angular_velocity)
  ang_vel_error = torch.square(ang_vel_command - actual_ang_vel_z)

  return torch.exp(-ang_vel_error / (std**2))


def goal_approach_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  scale: float = 2.0,
) -> torch.Tensor:
  """Smooth reward for approaching the goal based on distance reduction.

  This implements potential-based reward shaping for stable PPO training:
  - Uses exponential decay based on distance to goal
  - Provides dense, continuous gradient signal (no sparse jumps)
  - Scales reward based on how close the robot is to the goal

  The reward is computed as: exp(-distance / scale)
  This gives higher reward when closer to the goal, with smooth gradients.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    scale: Distance scale parameter. Larger values give rewards at greater distances.
      Default 2.0 means significant reward within ~4m of goal.

  Returns:
    Reward tensor of shape [num_envs] in range (0, 1].
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  # Get distance to current goal
  distance = command_term.target_distance  # [num_envs]

  # Exponential reward: closer = higher reward, with smooth gradients
  reward = torch.exp(-distance / scale)

  return reward


def goal_reached_bonus(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
) -> torch.Tensor:
  """Bonus reward for reaching a goal.

  Gives a one-time bonus when the robot reaches a goal (detected by
  reach_goal_timer transitioning from 0 to positive).

  This is a sparse reward that supplements the dense goal_approach_reward.
  Using it alone can cause high-variance gradients and training instability.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.

  Returns:
    Reward tensor of shape [num_envs] with 1.0 for envs that just reached a goal.
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  # KeyboardGoalTrackingCommand doesn't track goal reaching
  if isinstance(command_term, KeyboardGoalTrackingCommand):
    return torch.zeros(env.num_envs, device=env.device)

  # Reward when newly reached (timer is at exactly dt)
  just_reached = (command_term.reach_goal_timer > 0) & (
    command_term.reach_goal_timer <= env.step_dt * 1.5
  )
  return just_reached.float()


def heading_rate_penalty(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive angular velocity when not needing to turn.

  Encourages smooth turning by penalizing high yaw rate when heading
  error is small.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    asset_cfg: Configuration for the robot entity.

  Returns:
    Penalty tensor of shape [num_envs].
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  asset: Entity = env.scene[asset_cfg.name]

  # Get heading error and angular velocity
  delta_yaw = torch.abs(command_term.delta_yaw)  # [num_envs]
  yaw_rate = asset.data.root_link_ang_vel_b[:, 2]  # [num_envs]

  # Penalize high yaw rate when heading error is small
  # Scale factor decreases penalty when large turn is needed
  turn_needed_scale = torch.clamp(1.0 - delta_yaw / 0.5, min=0.0, max=1.0)
  penalty = torch.square(yaw_rate) * turn_needed_scale

  return penalty


class goal_tracking_posture:
  """Penalize deviation from default pose for goal tracking tasks.

  Similar to variable_posture but adapted for goal tracking where the command
  format is [delta_yaw, delta_next_yaw, target_velocity] instead of
  [lin_vel_x, lin_vel_y, ang_vel_z].

  Uses the target_velocity from the goal tracking command to determine the
  motion regime (standing, walking, running).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.
    from .goal_command import GoalTrackingCommand
    from .keyboard_command import KeyboardGoalTrackingCommand

    asset: Entity = env.scene[asset_cfg.name]

    # Get command term to access target velocity
    command_term = env.command_manager.get_term(command_name)

    if isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)):
      # For goal tracking: use target_velocity as the speed indicator
      total_speed = command_term.target_velocity.clone()
    else:
      # Fallback for velocity commands
      command = env.command_manager.get_command(command_name)
      assert command is not None
      linear_speed = torch.norm(command[:, :2], dim=1)
      angular_speed = torch.abs(command[:, 2])
      total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


# ============================================================================
# Velocity Tracking Metrics Logging (Zero-weight reward for WandB logging)
# ============================================================================


def circular_goal_velocity_tracking_log(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Compute and log velocity tracking metrics for goal tasks.

  This is a zero-weight reward function used purely for logging metrics to WandB.
  It computes the velocity tracking error (difference between actual velocity
  and target velocity) and logs it along with actual and target velocities.

  The metrics logged are:
  - Metrics/actual_forward_velocity: Robot's actual forward velocity (m/s)
  - Metrics/target_velocity: Commanded target velocity (m/s)
  - Metrics/velocity_error: Absolute difference between actual and target (m/s)
  - Metrics/velocity_error_signed: Signed difference (actual - target) (m/s)
  - Metrics/heading_error: Heading error in radians

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.
    asset_cfg: Configuration for the robot asset.

  Returns:
    Tensor of shape [num_envs] with zeros (this is purely for logging).
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand
  from .reachable_goal_command import FixedSequentialGoalCommand, ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  if not isinstance(
    command_term,
    (
      CircularGoalTrackingCommand,
      KeyboardCircularGoalTrackingCommand,
      ReachableGoalCommand,
      FixedSequentialGoalCommand,
    ),
  ):
    return torch.zeros(env.num_envs, device=env.device)

  # Get robot entity
  asset: Entity = env.scene[asset_cfg.name]

  # Get robot's linear velocity in body frame
  lin_vel_b = asset.data.root_link_lin_vel_b[:, :2]  # [num_envs, 2]

  # Get heading error (delta_yaw)
  delta_yaw = command_term.delta_yaw  # [num_envs]

  # Compute velocity towards goal direction (projected onto goal direction)
  # Goal direction in robot frame is delta_yaw
  # Forward velocity component in goal direction
  actual_velocity = lin_vel_b[:, 0] * torch.cos(delta_yaw) + lin_vel_b[
    :, 1
  ] * torch.sin(delta_yaw)

  # Get target velocity
  target_velocity = command_term.target_velocity  # [num_envs]

  # Compute velocity errors
  velocity_error = torch.abs(actual_velocity - target_velocity)
  velocity_error_signed = actual_velocity - target_velocity

  # Log metrics to WandB via env.extras["log"]
  # Only log if the "log" dict exists (it's created during reset, not at init)
  if "log" in env.extras:
    env.extras["log"]["Metrics/actual_forward_velocity"] = torch.mean(actual_velocity)
    env.extras["log"]["Metrics/target_velocity"] = torch.mean(target_velocity)
    env.extras["log"]["Metrics/velocity_error"] = torch.mean(velocity_error)
    env.extras["log"]["Metrics/velocity_error_signed"] = torch.mean(
      velocity_error_signed
    )

    # Also log heading error for completeness
    heading_error = torch.abs(delta_yaw)
    env.extras["log"]["Metrics/heading_error"] = torch.mean(heading_error)

  # Return zeros (this reward has weight=0, used only for logging)
  return torch.zeros(env.num_envs, device=env.device)


def directional_safety_speed_penalty(
  env: ManagerBasedRlEnv,
  forward_sensor_name: str = "raycast",
  left_sensor_name: str = "raycast_left",
  right_sensor_name: str = "raycast_right",
  skip_ground_cols: int = 3,
  safety_margin_m: float = 0.10,
  reaction_time_s: float = 0.02,
  braking_time_s: float = 0.25,
  side_point_forward_offset_m: float = 0.35,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize directional approach speed that exceeds the stop-time envelope."""
  asset: Entity = env.scene[asset_cfg.name]

  front_clearance = min_raycast_clearance(
    env.scene[forward_sensor_name].data.distances,
    skip_ground_cols=skip_ground_cols,
  )
  left_clearance = min_raycast_clearance(
    env.scene[left_sensor_name].data.distances,
    skip_ground_cols=skip_ground_cols,
  )
  right_clearance = min_raycast_clearance(
    env.scene[right_sensor_name].data.distances,
    skip_ground_cols=skip_ground_cols,
  )

  front_speed, left_speed, right_speed = compute_directional_approach_speeds(
    asset.data.root_link_lin_vel_b[:, :2],
    asset.data.root_link_ang_vel_b[:, 2],
    side_point_forward_offset_m=side_point_forward_offset_m,
  )

  speed_limits = torch.stack(
    [
      compute_safe_speed_limit(
        front_clearance,
        safety_margin_m=safety_margin_m,
        reaction_time_s=reaction_time_s,
        braking_time_s=braking_time_s,
      ),
      compute_safe_speed_limit(
        left_clearance,
        safety_margin_m=safety_margin_m,
        reaction_time_s=reaction_time_s,
        braking_time_s=braking_time_s,
      ),
      compute_safe_speed_limit(
        right_clearance,
        safety_margin_m=safety_margin_m,
        reaction_time_s=reaction_time_s,
        braking_time_s=braking_time_s,
      ),
    ],
    dim=1,
  )
  directional_speeds = torch.stack([front_speed, left_speed, right_speed], dim=1)
  return torch.clamp(directional_speeds - speed_limits, min=0.0).amax(dim=1)
