from __future__ import annotations

import torch


def min_raycast_clearance(
  distances: torch.Tensor,
  skip_ground_cols: int = 3,
) -> torch.Tensor:
  """Return the minimum horizontal clearance from an 8x8 raycast image."""
  if distances.ndim != 2:
    raise ValueError(
      f"Expected raycast distances with shape [num_envs, num_rays], got {tuple(distances.shape)}."
    )

  num_envs, num_rays = distances.shape
  if num_rays % 64 != 0:
    raise ValueError(
      f"Expected raycast distances to be a multiple of 64, got {num_rays}."
    )

  start_col = min(max(int(skip_ground_cols), 0), 7)
  num_groups = num_rays // 64
  grid = distances.reshape(num_envs, num_groups, 8, 8)
  return grid[..., start_col:].amin(dim=(-1, -2)).amin(dim=-1)


def compute_directional_approach_speeds(
  lin_vel_b: torch.Tensor,
  yaw_rate_b: torch.Tensor,
  side_point_forward_offset_m: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Estimate approach speeds for front, left, and right obstacle bands."""
  forward_speed = torch.clamp(lin_vel_b[:, 0], min=0.0)
  side_sweep_speed = lin_vel_b[:, 1] + yaw_rate_b * float(side_point_forward_offset_m)
  left_speed = torch.clamp(side_sweep_speed, min=0.0)
  right_speed = torch.clamp(-side_sweep_speed, min=0.0)
  return forward_speed, left_speed, right_speed


def compute_safe_speed_limit(
  clearance_m: torch.Tensor,
  safety_margin_m: float,
  reaction_time_s: float,
  braking_time_s: float,
) -> torch.Tensor:
  """Convert available clearance into a speed limit using a stopping-time model."""
  stop_horizon = max(float(reaction_time_s) + 0.5 * float(braking_time_s), 1e-6)
  available_distance = torch.clamp(clearance_m - float(safety_margin_m), min=0.0)
  return available_distance / stop_horizon


def compute_required_stop_distance(
  approach_speed_mps: torch.Tensor,
  safety_margin_m: float,
  reaction_time_s: float,
  braking_time_s: float,
) -> torch.Tensor:
  """Return the clearance needed to stop while leaving the safety margin."""
  stop_horizon = max(float(reaction_time_s) + 0.5 * float(braking_time_s), 1e-6)
  return (
    float(safety_margin_m) + torch.clamp(approach_speed_mps, min=0.0) * stop_horizon
  )


def damp_blocked_delta_yaw(
  delta_yaw: torch.Tensor,
  front_clearance_m: torch.Tensor,
  left_clearance_m: torch.Tensor,
  right_clearance_m: torch.Tensor,
  stop_distance_m: float,
  blocked_turn_scale: float = 0.0,
) -> torch.Tensor:
  """Damp yaw commands that would rotate further into a blocked direction.

  Front blocking does NOT suppress turning — only forward velocity is clipped
  (handled separately by ``compute_safe_speed_limit``).  This keeps the robot
  able to turn away from a wall it is facing.

  Only side blocking is enforced:
  - left blocked + positive yaw (turning left)  → damped
  - right blocked + negative yaw (turning right) → damped
  """
  turn_scale = min(max(float(blocked_turn_scale), 0.0), 1.0)
  adjusted = delta_yaw.clone()

  left_blocked = (left_clearance_m <= float(stop_distance_m)) & (adjusted > 0.0)
  if left_blocked.any():
    adjusted[left_blocked] = adjusted[left_blocked] * turn_scale

  right_blocked = (right_clearance_m <= float(stop_distance_m)) & (adjusted < 0.0)
  if right_blocked.any():
    adjusted[right_blocked] = adjusted[right_blocked] * turn_scale

  return adjusted
