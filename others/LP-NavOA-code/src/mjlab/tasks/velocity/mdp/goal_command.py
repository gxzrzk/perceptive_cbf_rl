"""Goal tracking command for waypoint-based navigation.

This module implements a goal/waypoint tracking command system inspired by
extreme-parkour's goal tracking approach. Instead of velocity commands,
the robot is given a series of waypoints to track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class GoalTrackingCommand(CommandTerm):
  """Command term for goal/waypoint tracking.

  This command generates a series of waypoints for the robot to track.
  The robot's observation includes the relative position and heading to the
  current goal and optionally the next goal.

  Based on the extreme-parkour goal tracking implementation:
  - Multiple goals per environment (num_goals)
  - Current goal index tracking
  - Goal reached detection with threshold and delay
  - Relative position and heading calculation
  """

  cfg: GoalTrackingCommandCfg

  def __init__(self, cfg: GoalTrackingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    # Note: terrain is available via env.scene.terrain if needed

    # Goal buffers: [num_envs, num_goals, 3] for (x, y, z)
    self.goals = torch.zeros(
      self.num_envs, cfg.num_goals, 3, device=self.device, dtype=torch.float32
    )
    # Current goal index per environment
    self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    # Timer for goal reached delay
    self.reach_goal_timer = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    # Whether the robot has reached the current goal
    self.reached_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    # Whether the robot has completed all goals (reached the last one)
    self.all_goals_completed = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )

    # Computed values for observations
    self.target_pos_rel = torch.zeros(
      self.num_envs, 2, device=self.device, dtype=torch.float32
    )
    self.next_target_pos_rel = torch.zeros(
      self.num_envs, 2, device=self.device, dtype=torch.float32
    )
    self.target_yaw = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self.next_target_yaw = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self.delta_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    self.delta_next_yaw = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self.target_distance = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    # Target velocity (for compatibility with velocity-based rewards)
    self.target_velocity = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self.target_velocity[:] = cfg.target_velocity

    # Command output: [delta_yaw, delta_next_yaw, target_velocity]
    self._command = torch.zeros(
      self.num_envs, 3, device=self.device, dtype=torch.float32
    )

    # Metrics
    self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Return the command tensor [delta_yaw, delta_next_yaw, target_velocity]."""
    return self._command

  @property
  def cur_goals(self) -> torch.Tensor:
    """Get current goals for all environments. Shape: [num_envs, 3]."""
    return self._gather_goals(future=0)

  @property
  def next_goals(self) -> torch.Tensor:
    """Get next goals for all environments. Shape: [num_envs, 3]."""
    return self._gather_goals(future=1)

  def _gather_goals(self, future: int = 0) -> torch.Tensor:
    """Gather goals at current index + future offset."""
    indices = (self.cur_goal_idx + future) % self.cfg.num_goals
    return self.goals[torch.arange(self.num_envs, device=self.device), indices]

  def _check_obstacle_collision(
    self, goal_x: torch.Tensor, goal_y: torch.Tensor, goal_z: torch.Tensor
  ) -> torch.Tensor:
    """Check if goal positions collide with obstacles using raycasting.

    Args:
      goal_x: X coordinates of goals [n]
      goal_y: Y coordinates of goals [n]
      goal_z: Z coordinates of goals [n]

    Returns:
      Boolean tensor [n] where True means collision detected.
    """
    import mujoco

    n = len(goal_x)
    collisions = torch.zeros(n, device=self.device, dtype=torch.bool)

    # Get the mujoco model and data
    sim = self._env.sim
    # Access the underlying mujoco model/data
    model = sim.model._model  # type: ignore
    data = sim.data._data  # type: ignore

    # Check height range around each goal
    check_height = 0.5  # Check from ground up to this height

    for i in range(n):
      x = float(goal_x[i].cpu())
      y = float(goal_y[i].cpu())
      z = float(goal_z[i].cpu()) if goal_z is not None else 0.0

      # Cast ray from above the goal position downward
      ray_start = np.array([x, y, z + check_height + 1.0])
      ray_dir = np.array([0.0, 0.0, -1.0])

      geom_id = np.array([-1], dtype=np.int32)
      result = mujoco.mj_ray(model, data, ray_start, ray_dir, None, 1, -1, geom_id)

      # If hit something and it's not the ground (terrain)
      if result > 0 and result < check_height + 0.5:
        # Check if the hit geom is an obstacle (not terrain ground)
        hit_geom_id = geom_id[0]
        if hit_geom_id >= 0:
          geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hit_geom_id)
          # Terrain ground geoms are typically the first or named 'terrain'
          if geom_name is not None and "terrain" not in geom_name.lower():
            collisions[i] = True

    return collisions

  def _generate_goals(self, env_ids: torch.Tensor) -> None:
    """Generate waypoints for the specified environments.

    Goals are generated in a forward direction with some randomness.
    The first goal is placed relative to the robot's current position.
    If obstacle_avoidance_enabled is True, goals are checked against obstacles.

    Note: We read position from qpos (generalized coordinates) instead of xpos
    (forward kinematics output) because during reset, qpos is updated before
    command_manager.reset() is called, but xpos may not be updated until the
    next mj_forward() call.
    """
    n = len(env_ids)
    if n == 0:
      return

    # Get robot positions from qpos (which is updated immediately after write)
    # This is more reliable during reset than root_link_pos_w (based on xpos)
    robot_data = self.robot.data
    qpos_adr = robot_data.indexing.free_joint_q_adr[:3]  # First 3 elements are x, y, z
    robot_pos = robot_data.data.qpos[env_ids][:, qpos_adr][:, :2]  # [n, 2]

    # For heading, we need to compute from quat in qpos
    quat_adr = robot_data.indexing.free_joint_q_adr[3:7]
    robot_quat = robot_data.data.qpos[env_ids][:, quat_adr]  # [n, 4] (w, x, y, z)
    # Extract yaw from quaternion (w, x, y, z format)
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    terrain = self._env.scene.terrain

    for goal_idx in range(self.cfg.num_goals):
      if goal_idx == 0:
        # First goal: relative to robot's current position
        # This ensures the first goal is always within first_goal_distance_range
        # centered around the robot, not the environment origin
        base_yaw = robot_yaw
        base_pos = robot_pos  # Use robot position for first goal
      else:
        # Subsequent goals: relative to previous goal
        prev_goal = self.goals[env_ids, goal_idx - 1, :2]
        if goal_idx > 1:
          prev_prev_goal = self.goals[env_ids, goal_idx - 2, :2]
        else:
          prev_prev_goal = robot_pos
        # Direction from prev_prev to prev goal
        direction = prev_goal - prev_prev_goal
        base_yaw = torch.atan2(direction[:, 1], direction[:, 0])
        base_pos = prev_goal

      # Try multiple times to find valid goal positions
      max_attempts = (
        self.cfg.max_goal_placement_attempts
        if self.cfg.obstacle_avoidance_enabled
        else 1
      )
      valid_mask = torch.zeros(n, device=self.device, dtype=torch.bool)
      goal_x = torch.zeros(n, device=self.device)
      goal_y = torch.zeros(n, device=self.device)
      goal_z = torch.zeros(n, device=self.device)

      # Sample distances once at the start (outside the attempt loop)
      if goal_idx == 0:
        # First goal: use first_goal_distance_range (within 4m circle by default)
        goal_distances = torch.empty(n, device=self.device).uniform_(
          self.cfg.first_goal_distance_range[0], self.cfg.first_goal_distance_range[1]
        )
      else:
        goal_distances = torch.empty(n, device=self.device).uniform_(
          self.cfg.goal_distance_range[0], self.cfg.goal_distance_range[1]
        )

      for attempt in range(max_attempts):
        # Only regenerate for invalid positions
        invalid_mask = ~valid_mask
        num_invalid = invalid_mask.sum().item()
        if num_invalid == 0:
          break

        # Use pre-sampled distances
        distance = goal_distances

        # Angle variation - increase range on retry attempts
        angle_range = self.cfg.goal_angle_range * (1.0 + attempt * 0.3)
        angle_offset = torch.empty(n, device=self.device).uniform_(
          -angle_range, angle_range
        )

        # Calculate goal position
        candidate_yaw = base_yaw + angle_offset
        candidate_x = base_pos[:, 0] + distance * torch.cos(candidate_yaw)
        candidate_y = base_pos[:, 1] + distance * torch.sin(candidate_yaw)

        # Clamp goals to stay within terrain bounds
        if terrain is not None and terrain.cfg.terrain_generator is not None:
          gen = terrain.cfg.terrain_generator
          half_width = (gen.num_cols * gen.size[1]) / 2
          half_length = (gen.num_rows * gen.size[0]) / 2
          candidate_x = torch.clamp(candidate_x, -half_length + 1.0, half_length - 1.0)
          candidate_y = torch.clamp(candidate_y, -half_width + 1.0, half_width - 1.0)

        # Set goal height
        candidate_z = torch.zeros(n, device=self.device)
        if terrain is not None:
          candidate_z = (
            self._env.scene.env_origins[env_ids, 2] + self.cfg.goal_height_offset
          )

        # Update only invalid positions
        goal_x = torch.where(invalid_mask, candidate_x, goal_x)
        goal_y = torch.where(invalid_mask, candidate_y, goal_y)
        goal_z = torch.where(invalid_mask, candidate_z, goal_z)

        # Check for obstacle collisions if enabled
        if self.cfg.obstacle_avoidance_enabled and attempt < max_attempts - 1:
          try:
            collisions = self._check_obstacle_collision(goal_x, goal_y, goal_z)
            valid_mask = ~collisions
          except Exception:
            # If collision check fails, accept all positions
            valid_mask[:] = True
        else:
          valid_mask[:] = True

      self.goals[env_ids, goal_idx, 0] = goal_x
      self.goals[env_ids, goal_idx, 1] = goal_y
      self.goals[env_ids, goal_idx, 2] = goal_z

  def _update_goals(self) -> None:
    """Update goal tracking state.

    This method:
    1. Checks if robots have reached their current goal
    2. Advances to next goal after reach_goal_delay
    3. Computes relative positions and headings
    """
    dt = self._env.step_dt

    # Check for goal advancement (after delay)
    next_flag = self.reach_goal_timer > self.cfg.reach_goal_delay

    # Check if we're about to complete all goals (reaching the last goal)
    is_last_goal = self.cur_goal_idx == (self.cfg.num_goals - 1)
    completing_all_goals = next_flag & is_last_goal
    self.all_goals_completed = completing_all_goals

    # Only advance if not completing (to avoid wrap-around)
    advance_flag = next_flag & ~is_last_goal
    self.cur_goal_idx[advance_flag] = self.cur_goal_idx[advance_flag] + 1
    self.reach_goal_timer[next_flag] = 0.0
    self.metrics["goals_reached"][next_flag] += 1

    # Check if robots have reached the current goal
    cur_goals = self.cur_goals
    robot_pos = self.robot.data.root_link_pos_w[:, :2]
    distance = torch.norm(robot_pos - cur_goals[:, :2], dim=1)
    self.target_distance = distance
    self.reached_goal = distance < self.cfg.next_goal_threshold
    self.reach_goal_timer[self.reached_goal] += dt

    # Compute relative positions
    self.target_pos_rel = cur_goals[:, :2] - robot_pos
    next_goals = self.next_goals
    self.next_target_pos_rel = next_goals[:, :2] - robot_pos

    # Compute target yaw angles
    norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
    target_vec_norm = self.target_pos_rel / (norm + 1e-5)
    self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    norm_next = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
    next_target_vec_norm = self.next_target_pos_rel / (norm_next + 1e-5)
    self.next_target_yaw = torch.atan2(
      next_target_vec_norm[:, 1], next_target_vec_norm[:, 0]
    )

    # Compute delta yaw (heading error)
    robot_yaw = self.robot.data.heading_w
    self.delta_yaw = wrap_to_pi(self.target_yaw - robot_yaw)
    self.delta_next_yaw = wrap_to_pi(self.next_target_yaw - robot_yaw)

    # Update command tensor
    self._command[:, 0] = self.delta_yaw
    self._command[:, 1] = self.delta_next_yaw
    self._command[:, 2] = self.target_velocity

    # Update metrics
    self.metrics["goal_distance"] = self.target_distance

  def _update_metrics(self) -> None:
    """Update tracking metrics."""
    pass  # Metrics updated in _update_goals

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample goals for the specified environments."""
    self.cur_goal_idx[env_ids] = 0
    self.reach_goal_timer[env_ids] = 0.0
    self.reached_goal[env_ids] = False
    self.all_goals_completed[env_ids] = False
    self._generate_goals(env_ids)
    # Resample target velocity from the configured range
    n = len(env_ids)
    self.target_velocity[env_ids] = torch.empty(n, device=self.device).uniform_(
      self.cfg.target_velocity_range[0], self.cfg.target_velocity_range[1]
    )
    # Initial update to compute relative positions
    self._update_goals()

  def _update_command(self) -> None:
    """Update command based on current state."""
    self._update_goals()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize goals and heading vectors."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    robot_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    goals = self.goals[batch].cpu().numpy()
    cur_idx = int(self.cur_goal_idx[batch].item())

    # Skip if robot appears uninitialized
    if np.linalg.norm(robot_pos) < 1e-6:
      return

    # Draw all goals
    for i in range(self.cfg.num_goals):
      goal = goals[i]
      # Color: green for current, yellow for next, gray for others
      if i == cur_idx:
        color = (0.0, 1.0, 0.0, 0.9)  # Green
        radius = 0.15
      elif i == (cur_idx + 1) % self.cfg.num_goals:
        color = (1.0, 1.0, 0.0, 0.7)  # Yellow
        radius = 0.12
      else:
        color = (0.5, 0.5, 0.5, 0.4)  # Gray
        radius = 0.08

      # Draw goal sphere
      visualizer.add_sphere(goal, radius=radius, color=color)

      # Draw line to next goal using cylinder
      next_i = (i + 1) % self.cfg.num_goals
      if i < self.cfg.num_goals - 1:  # Don't wrap around for visualization
        next_goal = goals[next_i]
        visualizer.add_cylinder(
          goal, next_goal, radius=0.02, color=(0.3, 0.3, 0.8, 0.5)
        )

    # Draw heading arrow from robot to current goal (length proportional to target velocity)
    cur_goal = goals[cur_idx]
    arrow_start = robot_pos.copy()
    arrow_start[2] += 0.5  # Raise arrow above robot

    # Calculate direction to goal and set length proportional to target velocity
    direction = cur_goal[:2] - robot_pos[:2]
    dist = np.linalg.norm(direction)
    if dist > 0.1:
      direction = direction / dist  # Normalize
      # Arrow length proportional to target velocity (0.5m per 1 m/s)
      target_vel = float(self.target_velocity[batch].cpu())
      arrow_length = max(0.3, abs(target_vel) * 0.5)  # Min 0.3m, scale by velocity
      arrow_end = arrow_start.copy()
      arrow_end[0] = arrow_start[0] + direction[0] * arrow_length
      arrow_end[1] = arrow_start[1] + direction[1] * arrow_length
    else:
      arrow_end = arrow_start.copy()
      arrow_end[0] += 0.3  # Default forward direction

    visualizer.add_arrow(arrow_start, arrow_end, color=(0.2, 0.8, 0.2, 0.8), width=0.02)


@dataclass(kw_only=True)
class GoalTrackingCommandCfg(CommandTermCfg):
  """Configuration for goal tracking command."""

  entity_name: str
  """Name of the robot entity in the scene."""

  num_goals: int = 4
  """Number of waypoints to generate per episode."""

  first_goal_distance_range: tuple[float, float] = (4.0, 8.0)
  """Range of distances for the first goal from robot [m]. Default: 4-8m."""

  goal_distance_range: tuple[float, float] = (4.0, 8.0)
  """Range of distances between consecutive goals [m]. 4-8m for natural obstacle avoidance."""

  goal_angle_range: float = 0.8
  """Maximum angle deviation from forward direction for goal placement [rad]."""

  next_goal_threshold: float = 0.5
  """Distance threshold to consider a goal as reached [m]."""

  reach_goal_delay: float = 0.1
  """Delay after reaching a goal before advancing to the next [s]."""

  target_velocity_range: tuple[float, float] = (0.0, 1.0)
  """Range of target velocity magnitudes for goal tracking [m/s].

  This represents the speed (magnitude) at which the robot should move towards
  the goal. Values are always non-negative. Used by curriculum to gradually
  increase the required speed during training.
  """

  target_velocity: float = 0.5
  """Initial target velocity magnitude for goal tracking [m/s].

  This is the speed (magnitude) towards the goal. Always non-negative.
  Overridden by curriculum during training.
  """

  goal_height_offset: float = 0.0
  """Height offset for goals above terrain [m]."""

  obstacle_check_radius: float = 0.5
  """Radius to check for obstacles around goal position [m]."""

  obstacle_avoidance_enabled: bool = False
  """Whether to check for obstacles when generating goals."""

  max_goal_placement_attempts: int = 10
  """Maximum attempts to find a valid goal position avoiding obstacles."""

  @dataclass
  class VizCfg:
    """Visualization configuration."""

    z_offset: float = 0.5
    scale: float = 1.0

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoalTrackingCommand:
    return GoalTrackingCommand(self, env)
