from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _raycast_corner_mask(
  num_rays: int,
  sensor_width: int,
  corner_width: int,
  device: torch.device,
) -> torch.Tensor:
  """Return a per-ray mask for square-grid corner zones."""
  corner_mask = torch.zeros(num_rays, dtype=torch.bool, device=device)
  if sensor_width <= 0 or num_rays % sensor_width != 0:
    return corner_mask

  sensor_height = num_rays // sensor_width
  corner_width = max(0, min(corner_width, sensor_width, sensor_height))
  if corner_width <= 0:
    return corner_mask

  ray_ids = torch.arange(num_rays, device=device)
  rows = ray_ids // sensor_width
  cols = ray_ids % sensor_width
  row_corner = (rows < corner_width) | (rows >= sensor_height - corner_width)
  col_corner = (cols < corner_width) | (cols >= sensor_width - corner_width)
  return row_corner & col_corner


def _vl53l8cx_false_max_probability(
  distances_m: torch.Tensor,
  max_distance: float,
  hit_mask: torch.Tensor,
  corner_mask: torch.Tensor,
  ray_dirs_w: torch.Tensor | None = None,
  normals_w: torch.Tensor | None = None,
  invalid_distance_start_m: float = 0.0,
  invalid_prob_at_max: float = 0.0,
  corner_invalid_prob_extra: float = 0.0,
  shallow_incidence_invalid_prob: float = 0.0,
  shallow_incidence_cos_start: float = 0.35,
  shallow_incidence_cos_full: float = 0.08,
) -> torch.Tensor:
  """Approximate false-max probability for low-confidence VL53L8CX hits."""
  invalid_prob = distances_m.new_zeros(distances_m.shape)

  if invalid_prob_at_max > 0.0:
    invalid_start_m = max(0.0, min(invalid_distance_start_m, max_distance))
    ramp_den = max(max_distance - invalid_start_m, 1e-6)
    distance_ramp = ((distances_m - invalid_start_m) / ramp_den).clamp(0.0, 1.0)
    invalid_prob = invalid_prob + distance_ramp * invalid_prob_at_max

  if corner_invalid_prob_extra > 0.0 and bool(torch.any(corner_mask)):
    invalid_prob = invalid_prob + (
      corner_mask.to(distances_m.dtype).unsqueeze(0) * corner_invalid_prob_extra
    )

  if (
    shallow_incidence_invalid_prob > 0.0
    and ray_dirs_w is not None
    and normals_w is not None
  ):
    ray_dirs_unit = ray_dirs_w / ray_dirs_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normals_unit = normals_w / normals_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    normal_alignment = torch.abs(torch.sum(ray_dirs_unit * normals_unit, dim=-1))
    shallow_den = max(
      shallow_incidence_cos_start - shallow_incidence_cos_full,
      1e-6,
    )
    shallow_ramp = (
      (shallow_incidence_cos_start - normal_alignment) / shallow_den
    ).clamp(0.0, 1.0)
    invalid_prob = invalid_prob + shallow_ramp * shallow_incidence_invalid_prob

  return torch.where(hit_mask, invalid_prob.clamp(0.0, 1.0), invalid_prob.new_zeros(()))


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def raycast_distances(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  near_occlusion_prob: float = 0.0,
  near_occlusion_distance_range: tuple[float, float] = (0.02, 0.08),
  near_occlusion_ray_prob: float = 0.9,
  near_occlusion_full_frame_prob: float = 0.35,
  spike_max_prob: float = 0.0,
  spike_bottom_rows_prob: float = 0.0,
  sensor_width: int = 8,
  vl53l8cx_noise_std_min_m: float = 0.0,
  vl53l8cx_noise_rel_std: float = 0.0,
  vl53l8cx_corner_rel_std_extra: float = 0.0,
  vl53l8cx_invalid_distance_start_m: float = 0.0,
  vl53l8cx_invalid_prob_at_max: float = 0.0,
  vl53l8cx_corner_invalid_prob_extra: float = 0.0,
  vl53l8cx_corner_width: int = 2,
  vl53l8cx_shallow_incidence_invalid_prob: float = 0.0,
  vl53l8cx_shallow_incidence_cos_start: float = 0.35,
  vl53l8cx_shallow_incidence_cos_full: float = 0.08,
) -> torch.Tensor:
  """Get normalized raycast distances.

  Distances are normalized to [0, 1] range by dividing by max_distance.
  A value of 0 means hit at origin, 1 means no hit (at max distance).

  Args:
    env: The environment instance.
    sensor_name: Name of the raycast sensor.
    near_occlusion_prob: Per-env probability to inject near-field occlusion.
    near_occlusion_distance_range: Replacement distance range in meters.
    near_occlusion_ray_prob: Per-ray probability for partial occlusion masks.
    near_occlusion_full_frame_prob: Probability to occlude the full raycast frame.
    spike_max_prob: Per-pixel per-step probability of spiking to 1.0 (max
      distance / no-obstacle reading) for all rows.
    spike_bottom_rows_prob: Higher per-pixel spike probability for the bottom
      2 rows of the sensor grid (ground-proximity false-max readings).
    sensor_width: Width of the sensor grid in pixels; used to identify which
      rays belong to the bottom 2 rows (last 2 * sensor_width entries).
    vl53l8cx_noise_std_min_m: Minimum distance noise standard deviation in
      meters for valid hits.
    vl53l8cx_noise_rel_std: Distance-proportional noise standard deviation for
      valid hits.
    vl53l8cx_corner_rel_std_extra: Additional relative noise standard deviation
      applied to corner zones.
    vl53l8cx_invalid_distance_start_m: Distance at which false-max invalid
      readings begin ramping up.
    vl53l8cx_invalid_prob_at_max: False-max invalid probability at max range.
    vl53l8cx_corner_invalid_prob_extra: Additional false-max probability for
      corner zones.
    vl53l8cx_corner_width: Width, in zones, of each corner region.
    vl53l8cx_shallow_incidence_invalid_prob: Additional probability of
      returning max range for grazing, low-confidence hits.
    vl53l8cx_shallow_incidence_cos_start: Surface-normal alignment below which
      shallow-incidence invalidation starts to ramp up.
    vl53l8cx_shallow_incidence_cos_full: Surface-normal alignment below which
      the shallow-incidence invalidation reaches its full probability.

  Returns:
    Tensor of shape [num_envs, num_rays] with normalized distances in [0, 1].
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  distances = sensor.data.distances
  max_distance = sensor.cfg.max_distance

  distances_m = distances
  if distances.ndim == 2 and (
    vl53l8cx_noise_std_min_m > 0.0
    or vl53l8cx_noise_rel_std > 0.0
    or vl53l8cx_corner_rel_std_extra > 0.0
    or vl53l8cx_invalid_prob_at_max > 0.0
    or vl53l8cx_corner_invalid_prob_extra > 0.0
    or vl53l8cx_shallow_incidence_invalid_prob > 0.0
  ):
    num_envs, num_rays = distances.shape
    if num_envs > 0 and num_rays > 0:
      distances_m = distances.clone()
      hit_mask = distances_m < max_distance

      corner_mask = _raycast_corner_mask(
        num_rays,
        sensor_width,
        vl53l8cx_corner_width,
        distances.device,
      )

      if (
        vl53l8cx_noise_std_min_m > 0.0
        or vl53l8cx_noise_rel_std > 0.0
        or vl53l8cx_corner_rel_std_extra > 0.0
      ):
        rel_std = distances_m.new_full((num_rays,), vl53l8cx_noise_rel_std)
        if vl53l8cx_corner_rel_std_extra > 0.0:
          rel_std = rel_std.clone()
          rel_std[corner_mask] += vl53l8cx_corner_rel_std_extra
        noise_std_m = torch.maximum(
          distances_m * rel_std.unsqueeze(0),
          distances_m.new_full(distances_m.shape, vl53l8cx_noise_std_min_m),
        )
        noise = torch.randn_like(distances_m) * noise_std_m
        distances_m = torch.where(hit_mask, distances_m + noise, distances_m)
        distances_m = distances_m.clamp(0.0, max_distance)

      if (
        vl53l8cx_invalid_prob_at_max > 0.0
        or vl53l8cx_corner_invalid_prob_extra > 0.0
        or vl53l8cx_shallow_incidence_invalid_prob > 0.0
      ):
        ray_dirs_w = sensor.data.hit_pos_w - sensor.data.pos_w.unsqueeze(1)
        invalid_prob = _vl53l8cx_false_max_probability(
          distances_m=distances_m,
          max_distance=float(max_distance),
          hit_mask=hit_mask,
          corner_mask=corner_mask,
          ray_dirs_w=ray_dirs_w,
          normals_w=sensor.data.normals_w,
          invalid_distance_start_m=vl53l8cx_invalid_distance_start_m,
          invalid_prob_at_max=vl53l8cx_invalid_prob_at_max,
          corner_invalid_prob_extra=vl53l8cx_corner_invalid_prob_extra,
          shallow_incidence_invalid_prob=vl53l8cx_shallow_incidence_invalid_prob,
          shallow_incidence_cos_start=vl53l8cx_shallow_incidence_cos_start,
          shallow_incidence_cos_full=vl53l8cx_shallow_incidence_cos_full,
        )
        invalid_mask = (torch.rand_like(distances_m) < invalid_prob) & hit_mask
        distances_m = torch.where(
          invalid_mask,
          distances_m.new_full(distances_m.shape, max_distance),
          distances_m,
        )

  # Normalize to [0, 1] range.
  normalized = distances_m / max_distance

  # --- Near-field occlusion DR ---
  result = normalized
  if near_occlusion_prob > 0.0 and normalized.ndim == 2:
    num_envs, num_rays = normalized.shape
    if num_envs > 0 and num_rays > 0:
      trigger_mask = (
        torch.rand(num_envs, device=normalized.device) < near_occlusion_prob
      )
      if bool(torch.any(trigger_mask)):
        lo_m, hi_m = near_occlusion_distance_range
        lo_m = max(0.0, min(float(lo_m), float(max_distance)))
        hi_m = max(lo_m + 1e-6, min(float(hi_m), float(max_distance)))
        lo_n = lo_m / max_distance
        hi_n = hi_m / max_distance

        occluded = normalized.clone()
        trigger_ids = torch.nonzero(trigger_mask, as_tuple=False).squeeze(1)
        n_trigger = int(trigger_ids.numel())
        if n_trigger > 0:
          # Per-ray random distances for realism.
          near_values = (
            torch.rand((n_trigger, num_rays), device=normalized.device) * (hi_n - lo_n)
            + lo_n
          )
          full_mask = (
            torch.rand(n_trigger, device=normalized.device)
            < near_occlusion_full_frame_prob
          )
          if bool(torch.any(full_mask)):
            full_ids = trigger_ids[full_mask]
            occluded[full_ids, :] = near_values[full_mask]
          partial_mask = ~full_mask
          if bool(torch.any(partial_mask)):
            partial_ids = trigger_ids[partial_mask]
            ray_mask = (
              torch.rand((partial_ids.numel(), num_rays), device=normalized.device)
              < near_occlusion_ray_prob
            )
            partial_obs = occluded[partial_ids, :]
            occluded[partial_ids, :] = torch.where(
              ray_mask, near_values[partial_mask], partial_obs
            )
        result = occluded

  # --- Spike DR: random pixels jump to 1.0 (max-distance / no-obstacle) ---
  # Bottom 2 rows have higher probability due to ground-proximity false-max
  # readings observed on the real sensor.
  if (spike_max_prob > 0.0 or spike_bottom_rows_prob > 0.0) and result.ndim == 2:
    num_envs, num_rays = result.shape
    if num_envs > 0 and num_rays > 0:
      spike_probs = result.new_full((num_rays,), spike_max_prob)
      n_bottom = 2 * sensor_width
      if spike_bottom_rows_prob > 0.0 and 0 < n_bottom <= num_rays:
        spike_probs[-n_bottom:] = spike_bottom_rows_prob
      spike_mask = torch.rand(
        num_envs, num_rays, device=result.device
      ) < spike_probs.unsqueeze(0)
      result = torch.where(spike_mask, result.new_ones(result.shape), result)

  return result.clamp(0.0, 1.0)


def height_scan_observations(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  base_height: float = 0.9,
) -> torch.Tensor:
  """Get terrain height observations relative to robot base height.

  Uses a raycast sensor pointing downward (-Z) to measure terrain heights
  in a grid pattern around the robot. Returns heights as offsets from the
  expected robot base height, clipped to [-1, 1] range.

  This follows the "more" approach where the critic observes ground truth
  terrain heights rather than noisy depth camera observations.

  Args:
    env: The environment instance.
    sensor_name: Name of the height scan raycast sensor.
    base_height: Expected robot base height above ground (default 0.9m for G1).

  Returns:
    Tensor of shape [num_envs, num_rays] with height offsets in [-1, 1].
  """
  sensor: RayCastSensor = env.scene[sensor_name]

  # Get ray distances (distance from sensor to ground)
  distances = sensor.data.distances  # [num_envs, num_rays]

  # Height scan sensor is at robot base, pointing down
  # Distance = robot_height - terrain_height
  # So terrain_height = robot_height - distance
  # Relative height = robot_height - base_height - terrain_height
  #                 = robot_height - base_height - (robot_height - distance)
  #                 = distance - base_height
  # But we want: terrain_height - (robot_height - base_height)
  #            = (robot_height - distance) - (robot_height - base_height)
  #            = base_height - distance

  # Compute terrain heights relative to expected base height
  # Positive value = terrain is above expected base height (obstacle)
  # Negative value = terrain is below expected base height (pit)
  relative_heights = base_height - distances

  # Clip to [-1, 1] range for normalization
  return torch.clip(relative_heights, -1.0, 1.0)


# ============================================================================
# Goal Tracking Observations
# ============================================================================


def goal_heading_command(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get the goal tracking command [delta_yaw, delta_next_yaw, target_velocity].

  This is the primary command observation for goal tracking tasks.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 3] with [delta_yaw, delta_next_yaw, target_velocity].
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def goal_relative_position(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get relative position to current goal in robot frame.

  Returns the relative x, y position to the current goal, transformed
  to the robot's local coordinate frame.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 2] with [rel_x, rel_y] in robot frame.
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  # KeyboardGoalTrackingCommand stores goal_pos_rel directly in robot frame
  if isinstance(command_term, KeyboardGoalTrackingCommand):
    return command_term.goal_pos_rel

  # GoalTrackingCommand: Get relative position in world frame
  target_pos_rel_w = command_term.target_pos_rel  # [num_envs, 2]

  # Transform to robot frame
  robot: Entity = env.scene[command_term.cfg.entity_name]
  robot_quat = robot.data.root_link_quat_w  # [num_envs, 4]

  # Create 3D vector for transformation
  target_pos_3d = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
  target_pos_3d[:, :2] = target_pos_rel_w

  # Transform to robot frame
  from mjlab.utils.lab_api.math import quat_apply_inverse

  target_pos_b = quat_apply_inverse(robot_quat, target_pos_3d)

  return target_pos_b[:, :2]


def goal_distance(env: ManagerBasedRlEnv, command_name: str = "goal") -> torch.Tensor:
  """Get distance to current goal.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 1] with distance to current goal.
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )
  return command_term.target_distance.unsqueeze(-1)


def goal_heading_error(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get heading error to current and next goal.

  Returns delta_yaw to current goal and delta_yaw to next goal.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 2] with [delta_yaw, delta_next_yaw].
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  # KeyboardGoalTrackingCommand uses delta_yaw for both (no next goal concept)
  if isinstance(command_term, KeyboardGoalTrackingCommand):
    return torch.stack([command_term.delta_yaw, command_term.delta_yaw], dim=-1)

  return torch.stack([command_term.delta_yaw, command_term.delta_next_yaw], dim=-1)


def goal_reached(env: ManagerBasedRlEnv, command_name: str = "goal") -> torch.Tensor:
  """Get whether the current goal is reached.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 1] with 1.0 if goal reached, 0.0 otherwise.
  """
  from .goal_command import GoalTrackingCommand
  from .keyboard_command import KeyboardGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, (GoalTrackingCommand, KeyboardGoalTrackingCommand)), (
    f"Command '{command_name}' is not a GoalTrackingCommand or KeyboardGoalTrackingCommand."
  )

  # KeyboardGoalTrackingCommand doesn't have reached_goal concept
  if isinstance(command_term, KeyboardGoalTrackingCommand):
    return torch.zeros(env.num_envs, 1, device=env.device)

  return command_term.reached_goal.float().unsqueeze(-1)


# ============================================================================
# Circular Goal Tracking Observations
# ============================================================================


def circular_goal_command(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get the circular goal tracking command [delta_yaw, target_velocity].

  This is the primary command observation for circular goal tracking tasks.
  The command is stable because goal angle is in world frame and changes smoothly.

  Args:
    env: The environment instance.
    command_name: Name of the circular goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 2] with [delta_yaw, target_velocity].
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(
    command_term, (CircularGoalTrackingCommand, KeyboardCircularGoalTrackingCommand)
  ), (
    f"Command '{command_name}' is not a CircularGoalTrackingCommand "
    f"or KeyboardCircularGoalTrackingCommand."
  )
  return command_term.command


# ============================================================================
# Reachable Goal Tracking Observations
# ============================================================================


def reachable_goal_command(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get the reachable goal tracking command [delta_yaw, target_velocity].

  This is the primary command observation for reachable goal tracking tasks.
  Unlike circular goal tracking, the goal is a FIXED position in world frame
  that robot can actually reach.

  Args:
    env: The environment instance.
    command_name: Name of the reachable goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 2] with [delta_yaw, target_velocity].
  """
  from .reachable_goal_command import ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, ReachableGoalCommand), (
    f"Command '{command_name}' is not a ReachableGoalCommand."
  )
  return command_term.command


def fixed_sequential_goal_command(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get the fixed sequential goal tracking command [delta_yaw, target_velocity].

  This observation is for the FixedSequentialGoalCommand which provides
  a fixed sequence of goal positions (e.g., Goal 1 then Goal 2).

  Args:
    env: The environment instance.
    command_name: Name of the fixed sequential goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 2] with [delta_yaw, target_velocity].
  """
  from .reachable_goal_command import FixedSequentialGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, FixedSequentialGoalCommand), (
    f"Command '{command_name}' is not a FixedSequentialGoalCommand."
  )
  return command_term.command


def reachable_goal_distance(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Get the distance to the reachable goal.

  Args:
    env: The environment instance.
    command_name: Name of the reachable goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 1] with distance to goal.
  """
  from .reachable_goal_command import ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, ReachableGoalCommand), (
    f"Command '{command_name}' is not a ReachableGoalCommand."
  )
  return command_term.distance_to_goal.unsqueeze(-1)


def reachable_goal_reached(
  env: ManagerBasedRlEnv, command_name: str = "goal"
) -> torch.Tensor:
  """Check if the reachable goal has been reached.

  Args:
    env: The environment instance.
    command_name: Name of the reachable goal tracking command term.

  Returns:
    Tensor of shape [num_envs, 1] with 1.0 if reached, 0.0 otherwise.
  """
  from .reachable_goal_command import ReachableGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, ReachableGoalCommand), (
    f"Command '{command_name}' is not a ReachableGoalCommand."
  )
  return command_term.goal_reached.float().unsqueeze(-1)


# ============================================================================
# Deployment observation layout
# ============================================================================


def ball_observation_placeholder(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reserve a zero-filled observation slot in the deployed actor layout.

  The LP-NavOA actor observation is a fixed concatenation order that is also
  mirrored by the exported deployment configuration. This term keeps a
  constant-width, seven-dimensional slot in that order so the interface stays
  stable across training stages and on hardware; it carries no information and
  is always zero at inference time.

  Args:
    env: The environment instance.

  Returns:
    Tensor of shape [num_envs, 7] filled with zeros.
  """
  return torch.zeros(env.num_envs, 7, device=env.device, dtype=torch.float32)


# ============================================================================
# Velocity Tracking Metrics (WandB Logging)
# ============================================================================


def circular_goal_velocity_tracking_metrics(
  env: ManagerBasedRlEnv,
  command_name: str = "goal",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Compute and log velocity tracking metrics for circular goal tasks.

  This function computes the velocity tracking error (difference between
  actual velocity and target velocity) and logs it to WandB along with
  the actual and target velocities.

  The metrics logged are:
  - Metrics/actual_forward_velocity: Robot's actual forward velocity (m/s)
  - Metrics/target_velocity: Commanded target velocity (m/s)
  - Metrics/velocity_error: Absolute difference between actual and target (m/s)
  - Metrics/velocity_error_signed: Signed difference (actual - target) (m/s)

  Args:
    env: The environment instance.
    command_name: Name of the circular goal tracking command term.
    asset_cfg: Configuration for the robot asset.

  Returns:
    Tensor of shape [num_envs, 1] with zeros (dummy output for observation manager).
  """
  from .circular_goal_command import CircularGoalTrackingCommand
  from .keyboard_command import KeyboardCircularGoalTrackingCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(
    command_term, (CircularGoalTrackingCommand, KeyboardCircularGoalTrackingCommand)
  ), (
    f"Command '{command_name}' is not a CircularGoalTrackingCommand "
    f"or KeyboardCircularGoalTrackingCommand."
  )

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

  # Return dummy tensor (this observation is just for logging, not for policy)
  return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)
