"""Circular goal tracking command for stable goal-based training.

This module implements a goal tracking command where the goal point moves on a
fixed-radius circle centered on the robot. Unlike waypoint-based goal tracking,
this approach provides smoother reward signals similar to velocity tracking:

Key design principles:
1. Goal position is defined in WORLD frame at fixed radius from robot
2. Goal angle resamples gradually (like velocity commands) every 3-8 seconds
3. Robot needs to face the goal direction (yaw tracking)
4. Goal follows robot position but NOT robot orientation

This eliminates the reward oscillation caused by:
- Discrete goal switching (waypoint approach)
- Goal-robot relative position changing with robot movement
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from .safety_velocity import (
  compute_safe_speed_limit,
  damp_blocked_delta_yaw,
  min_raycast_clearance,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class CircularGoalTrackingCommand(CommandTerm):
  """Command term for circular goal tracking.

  The goal point is always at a fixed radius from the robot's current position,
  defined by an angle in the WORLD frame. This provides stable reward signals
  because:
  - Goal radius is constant (no distance-based reward oscillation)
  - Goal angle changes smoothly via periodic resampling
  - Robot linear velocity doesn't affect the goal-robot relationship

  The command output is [delta_yaw, target_velocity] where:
  - delta_yaw: angle difference between robot heading and goal direction
  - target_velocity: desired forward speed towards the goal
  """

  cfg: CircularGoalTrackingCommandCfg

  def __init__(self, cfg: CircularGoalTrackingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._env = env

    # Goal angle in WORLD frame (not relative to robot heading)
    # This is the key difference from waypoint tracking
    self.goal_angle_w = torch.zeros(self.num_envs, device=self.device)

    # Target velocity (forward speed towards goal)
    self._base_target_velocity_cmd = torch.full(
      (self.num_envs,), float(cfg.target_velocity), device=self.device
    )
    self._target_velocity_cmd = torch.full(
      (self.num_envs,), float(cfg.target_velocity), device=self.device
    )
    self.target_velocity = self._target_velocity_cmd

    # Computed values for observations
    self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
    self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Command output: [delta_yaw, target_velocity]
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._command[:, 1] = self._target_velocity_cmd

    # Metrics
    self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Return the command tensor [delta_yaw, target_velocity]."""
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample goal angle for specified environments.

    Goal angle is sampled RELATIVE to robot heading within configured range.
    This ensures the goal is always in front of the robot.
    """
    n = len(env_ids)
    if n == 0:
      return

    # Get current robot heading for these environments
    robot_yaw = self.robot.data.heading_w[env_ids]

    # Sample relative angle within configured range
    # goal_angle_range defines how far left/right of robot heading
    angle_range = self.cfg.goal_angle_range
    relative_angle = torch.empty(n, device=self.device).uniform_(
      -angle_range, angle_range
    )

    # Convert to world frame: goal_angle_w = robot_yaw + relative_angle
    self.goal_angle_w[env_ids] = wrap_to_pi(robot_yaw + relative_angle)

    # Resample target velocity from configured range
    self._base_target_velocity_cmd[env_ids] = torch.empty(
      n, device=self.device
    ).uniform_(self.cfg.target_velocity_range[0], self.cfg.target_velocity_range[1])
    self._target_velocity_cmd[env_ids] = self._base_target_velocity_cmd[env_ids]

    # Initial update
    self._update_command()

  def _get_raycast_clearance(self, sensor_name: str) -> torch.Tensor:
    """Return the minimum horizontal clearance for one raycast sensor."""
    sensor = self._env.scene[sensor_name]
    return min_raycast_clearance(
      sensor.data.distances,
      skip_ground_cols=self.cfg.safety_velocity_envelope.skip_ground_cols,
    )

  def _apply_safety_velocity_envelope(
    self, delta_yaw: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip target velocity using direction-aware time-to-stop limits."""
    safety_cfg = self.cfg.safety_velocity_envelope
    if not safety_cfg.enabled:
      self._target_velocity_cmd.copy_(self._base_target_velocity_cmd)
      return delta_yaw, self._target_velocity_cmd

    front_clearance = self._get_raycast_clearance(safety_cfg.forward_sensor_name)
    left_clearance = self._get_raycast_clearance(safety_cfg.left_sensor_name)
    right_clearance = self._get_raycast_clearance(safety_cfg.right_sensor_name)

    safe_speed_limit = compute_safe_speed_limit(
      front_clearance,
      safety_margin_m=safety_cfg.safety_margin_m,
      reaction_time_s=safety_cfg.reaction_time_s,
      braking_time_s=safety_cfg.braking_time_s,
    )

    clipped_target_velocity = torch.minimum(
      self._base_target_velocity_cmd, safe_speed_limit
    )

    abs_yaw = torch.abs(delta_yaw)
    turn_needed = abs_yaw > safety_cfg.turn_threshold_rad
    front_blocked = front_clearance < (
      safety_cfg.safety_margin_m + safety_cfg.stop_velocity_epsilon
    )
    side_clear = (left_clearance > safety_cfg.hard_stop_distance_m) | (
      right_clearance > safety_cfg.hard_stop_distance_m
    )
    allow_turn = turn_needed & front_blocked & side_clear
    min_turn_vel = safety_cfg.min_turn_velocity
    clipped_target_velocity = torch.where(
      allow_turn & (clipped_target_velocity < min_turn_vel),
      torch.full_like(clipped_target_velocity, min_turn_vel),
      clipped_target_velocity,
    )

    clipped_target_velocity = torch.where(
      ~allow_turn & (clipped_target_velocity < safety_cfg.stop_velocity_epsilon),
      torch.zeros_like(clipped_target_velocity),
      clipped_target_velocity,
    )

    delta_yaw = damp_blocked_delta_yaw(
      delta_yaw,
      front_clearance_m=front_clearance,
      left_clearance_m=left_clearance,
      right_clearance_m=right_clearance,
      stop_distance_m=safety_cfg.hard_stop_distance_m,
      blocked_turn_scale=safety_cfg.blocked_turn_scale,
    )

    self._target_velocity_cmd.copy_(clipped_target_velocity)
    return delta_yaw, clipped_target_velocity

  def _update_command(self) -> None:
    """Update command based on current robot state."""
    # Get robot position and heading
    robot_pos = self.robot.data.root_link_pos_w[:, :2]  # [num_envs, 2]
    robot_yaw = self.robot.data.heading_w  # [num_envs]

    # Compute goal position in world frame
    # Goal is at fixed radius from robot, at angle goal_angle_w in world frame
    goal_x = robot_pos[:, 0] + self.cfg.goal_radius * torch.cos(self.goal_angle_w)
    goal_y = robot_pos[:, 1] + self.cfg.goal_radius * torch.sin(self.goal_angle_w)
    self.goal_pos_w[:, 0] = goal_x
    self.goal_pos_w[:, 1] = goal_y
    self.goal_pos_w[:, 2] = self.robot.data.root_link_pos_w[
      :, 2
    ]  # Same height as robot

    # Compute delta_yaw: difference between robot heading and goal direction
    # Goal direction is goal_angle_w (in world frame)
    self.delta_yaw = wrap_to_pi(self.goal_angle_w - robot_yaw)
    self.delta_yaw, target_velocity = self._apply_safety_velocity_envelope(
      self.delta_yaw
    )

    # Update command tensor
    self._command[:, 0] = self.delta_yaw
    self._command[:, 1] = target_velocity

    # Update metrics
    self.metrics["heading_error"] = torch.abs(self.delta_yaw)

  def _update_metrics(self) -> None:
    """Update tracking metrics."""
    # Velocity towards goal direction
    lin_vel_b = self.robot.data.root_link_lin_vel_b[:, :2]

    # Project velocity onto goal direction (in robot frame)
    # Goal direction in robot frame is delta_yaw
    vel_towards_goal = lin_vel_b[:, 0] * torch.cos(self.delta_yaw) + lin_vel_b[
      :, 1
    ] * torch.sin(self.delta_yaw)

    self.metrics["velocity_error"] = torch.abs(
      vel_towards_goal - self._target_velocity_cmd
    )

    # Store actual velocity for visualization
    self._actual_velocity = vel_towards_goal

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize goal point, heading vector, and velocity information."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    robot_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    goal_pos = self.goal_pos_w[batch].cpu().numpy()
    target_vel = float(self.target_velocity[batch].item())

    # Get actual velocity towards goal
    if hasattr(self, "_actual_velocity"):
      actual_vel = float(self._actual_velocity[batch].item())
    else:
      # Compute actual velocity if not yet computed
      lin_vel_b = self.robot.data.root_link_lin_vel_b[batch, :2].cpu().numpy()
      delta_yaw = float(self.delta_yaw[batch].item())
      actual_vel = float(
        lin_vel_b[0] * np.cos(delta_yaw) + lin_vel_b[1] * np.sin(delta_yaw)
      )

    velocity_error = actual_vel - target_vel

    # Skip if robot appears uninitialized
    if np.linalg.norm(robot_pos) < 1e-6:
      return

    z_offset = self.cfg.viz.z_offset

    # Print velocity information to console
    print(
      f"\r[Velocity] Target: {target_vel:+.2f} m/s | "
      f"Actual: {actual_vel:+.2f} m/s | "
      f"Error: {velocity_error:+.2f} m/s | "
      f"Heading Error: {np.degrees(float(self.delta_yaw[batch].item())):+.1f}°",
      end="",
    )

    # Draw goal point as a sphere
    goal_vis_pos = goal_pos.copy()
    goal_vis_pos[2] += z_offset
    visualizer.add_sphere(
      center=goal_vis_pos,
      radius=0.15,
      color=(0.2, 0.8, 0.2, 0.8),  # Green
    )

    # Draw line from robot to goal
    robot_vis_pos = robot_pos.copy()
    robot_vis_pos[2] += z_offset
    visualizer.add_arrow(
      robot_vis_pos,
      goal_vis_pos,
      color=(0.2, 0.6, 0.8, 0.6),  # Cyan
      width=0.02,
    )

    # Draw robot heading direction
    robot_yaw = float(self.robot.data.heading_w[batch].item())
    heading_end = robot_pos.copy()
    heading_end[0] += 0.8 * np.cos(robot_yaw)
    heading_end[1] += 0.8 * np.sin(robot_yaw)
    heading_end[2] += z_offset
    visualizer.add_arrow(
      robot_vis_pos,
      heading_end,
      color=(1.0, 0.5, 0.0, 0.8),  # Orange - robot heading
      width=0.02,
    )

    # Draw target velocity indicator (purple arrow - command velocity)
    # Arrow length proportional to target velocity
    vel_scale = 0.3 * target_vel
    target_vel_end = robot_pos.copy()
    target_vel_end[0] += vel_scale * np.cos(robot_yaw)
    target_vel_end[1] += vel_scale * np.sin(robot_yaw)
    target_vel_end[2] += z_offset + 0.25
    visualizer.add_arrow(
      np.array([robot_vis_pos[0], robot_vis_pos[1], robot_vis_pos[2] + 0.25]),
      target_vel_end,
      color=(0.8, 0.2, 0.8, 0.7),  # Purple - target velocity
      width=0.015,
      label="target_vel",
    )

    # Draw actual velocity indicator (cyan arrow - actual velocity)
    # Arrow length proportional to actual velocity
    actual_vel_scale = 0.3 * actual_vel
    actual_vel_end = robot_pos.copy()
    actual_vel_end[0] += actual_vel_scale * np.cos(robot_yaw)
    actual_vel_end[1] += actual_vel_scale * np.sin(robot_yaw)
    actual_vel_end[2] += z_offset + 0.35
    visualizer.add_arrow(
      np.array([robot_vis_pos[0], robot_vis_pos[1], robot_vis_pos[2] + 0.35]),
      actual_vel_end,
      color=(0.2, 0.8, 0.8, 0.7),  # Cyan - actual velocity
      width=0.015,
      label="actual_vel",
    )

    # Draw velocity error indicator
    # Green if error < 0.2, yellow if < 0.5, red otherwise
    abs_error = abs(velocity_error)
    if abs_error < 0.2:
      error_color = (0.2, 0.9, 0.2, 0.8)  # Green - good tracking
    elif abs_error < 0.5:
      error_color = (0.9, 0.9, 0.2, 0.8)  # Yellow - moderate error
    else:
      error_color = (0.9, 0.2, 0.2, 0.8)  # Red - large error

    # Draw a small sphere above robot indicating tracking quality
    error_indicator_pos = robot_pos.copy()
    error_indicator_pos[2] += z_offset + 0.5
    visualizer.add_sphere(
      center=error_indicator_pos,
      radius=0.08,
      color=error_color,
      label="tracking_quality",
    )


@dataclass(kw_only=True)
class CircularGoalTrackingCommandCfg(CommandTermCfg):
  """Configuration for circular goal tracking command.

  The goal point moves on a circle of fixed radius centered on the robot.
  The goal angle is RELATIVE to robot heading (always in front of robot).
  Goal angle resamples periodically, similar to velocity commands,
  providing stable training signals.

  Goal angle is sampled within [-goal_angle_range, +goal_angle_range]
  relative to robot heading. This ensures the goal is always in front.
  """

  entity_name: str

  # Goal radius (fixed distance from robot to goal)
  goal_radius: float = 3.0

  # Goal angle range relative to robot heading (radians)
  # Goal will be sampled in [-goal_angle_range, +goal_angle_range]
  # e.g., pi/6 (~30 deg) means goal is within ±30° of robot heading
  goal_angle_range: float = math.pi / 6  # ±30 degrees by default

  # Target velocity (forward speed towards goal)
  target_velocity: float = 0.5

  # Target velocity range for curriculum
  target_velocity_range: tuple[float, float] = (0.0, 1.0)

  @dataclass
  class SafetyVelocityEnvelopeCfg:
    """Configuration for time-to-stop-based obstacle braking."""

    enabled: bool = False
    """Whether to enable command-side safety clipping."""

    forward_sensor_name: str = "raycast"
    """Front raycast sensor name used for frontal clearance."""

    left_sensor_name: str = "raycast_left"
    """Left raycast sensor name used for blocked-side detection."""

    right_sensor_name: str = "raycast_right"
    """Right raycast sensor name used for blocked-side detection."""

    skip_ground_cols: int = 3
    """Ignore the downward-looking raycast columns when computing clearance."""

    safety_margin_m: float = 0.25
    """Final standstill clearance that should remain after braking [m]."""

    reaction_time_s: float = 0.15
    """Reaction delay before braking starts [s]."""

    braking_time_s: float = 0.8
    """Assumed time to decay to zero velocity once braking begins [s]."""

    stop_velocity_epsilon: float = 0.15
    """Snap small clipped speeds to zero to avoid unsafe creeping [m/s]."""

    side_point_forward_offset_m: float = 0.35
    """Forward lever arm kept for config parity with reachable-goal safety."""

    hard_stop_distance_m: float = 0.40
    """Distance at which yaw commands into the blocked side are damped [m]."""

    blocked_turn_scale: float = 0.0
    """Multiplier for yaw commands that rotate into a blocked direction."""

    min_turn_velocity: float = 0.20
    """Minimum target velocity preserved when turning away from obstacles [m/s]."""

    turn_threshold_rad: float = 0.35
    """delta_yaw above which the robot is considered to need turning [rad]."""

  safety_velocity_envelope: SafetyVelocityEnvelopeCfg = field(
    default_factory=SafetyVelocityEnvelopeCfg
  )
  """Time-to-stop safety-envelope configuration."""

  @dataclass
  class VizCfg:
    z_offset: float = 0.3

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> CircularGoalTrackingCommand:
    return CircularGoalTrackingCommand(self, env)
