"""Reachable goal tracking command for RA (Reach-Avoid) training.

This module implements a goal tracking command where the goal point is a FIXED
position in the world frame that the robot can actually reach. Unlike
CircularGoalTrackingCommand which maintains a constant distance to the goal,
this command:

1. Computes target distance based on: resample_interval * target_velocity
2. Places goal at that distance in robot's forward direction at resample time
3. Goal position is FIXED in world frame until next resample
4. Robot tracks the FIXED goal, so distance to goal decreases as robot walks
5. When robot gets close (<shrink_threshold), the "circular ring" shrinks
6. When robot reaches the goal (<reach_threshold), marks as reached

This enables RA training to correctly identify:
- V(init): Value at start (should decrease as robot learns to reach goal)
- V(die): Value when colliding (should be low/negative)
- V(turn): Value when direction changes (intermediate)

Key design:
- Goal is computed at resample time and fixed in world frame
- Each frame recalculates heading error from current position to fixed goal
- Distance to goal decreases as robot walks towards it
- Reach success when distance < reach_threshold

Path Planning feature for maze environments (NEW):
- 2D obstacle map extracted from terrain geometry (walls)
- A* path planning to find optimal route to goal
- Reachability check: if goal is unreachable, find alternative on robot's right
- Intermediate waypoints inserted between robot and final goal
- Robot tracks waypoints sequentially, switches to next when reached
- Resample duration calculated from planned path length
- Prefer turning right to maintain policy consistency
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from .safety_velocity import (
  compute_directional_approach_speeds,
  compute_required_stop_distance,
  compute_safe_speed_limit,
  damp_blocked_delta_yaw,
  min_raycast_clearance,
)

# Setup logger for path planning
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


# ==============================================================================
# 2D Path Planner for Maze Navigation
# ==============================================================================


class MazePathPlanner:
  """2D path planner for maze navigation using A* pathfinding.

  This class extracts obstacle information from terrain geometry and provides:
  - 2D occupancy grid generation from wall geometries
  - A* pathfinding from start to goal
  - Alternative goal finding when direct goal is unreachable
  - Waypoint generation along the planned path

  The planner operates in the local coordinate frame of each terrain cell
  (60m x 60m), with the terrain origin at the center of each cell.
  """

  def __init__(
    self,
    terrain_size: tuple[float, float] = (60.0, 60.0),
    grid_resolution: float = 0.5,
    robot_radius: float = 0.4,
    waypoint_spacing: float = 2.0,
    device: str = "cpu",
  ):
    """Initialize the path planner.

    Args:
      terrain_size: Size of terrain cell (width, height) in meters.
      grid_resolution: Resolution of occupancy grid in meters per cell.
      robot_radius: Robot collision radius for path planning.
      waypoint_spacing: Minimum distance between waypoints in simplified path [m].
      device: Torch device for tensor operations.
    """
    self.terrain_size = terrain_size
    self.grid_resolution = grid_resolution
    self.robot_radius = robot_radius
    self.waypoint_spacing = waypoint_spacing
    self.device = device

    # Grid dimensions
    self.grid_width = int(terrain_size[0] / grid_resolution)
    self.grid_height = int(terrain_size[1] / grid_resolution)

    # Occupancy grid: True = obstacle, False = free
    # Shape: [grid_height, grid_width]
    self.occupancy_grid: np.ndarray | None = None

    # Wall information cache: list of (x, y, half_size_x, half_size_y, angle)
    self._wall_cache: list[tuple[float, float, float, float, float]] = []

    # Boundary walls (always present)
    self._boundary_margin = 0.5

    # Cache of occupancy grids keyed by terrain cell origin (rounded to int)
    # This avoids rebuilding the grid when revisiting a terrain cell
    self._grid_cache: dict[tuple[int, int], np.ndarray] = {}

  def build_occupancy_grid_from_walls(
    self,
    walls: list[tuple[float, float, float, float, float]],
    env_origin: np.ndarray,
  ) -> None:
    """Build occupancy grid from wall geometry information.

    Args:
      walls: List of wall tuples (center_x, center_y, half_size_x, half_size_y, angle).
             Positions are in world frame.
      env_origin: Environment origin in world frame [x, y, z].
    """
    self._wall_cache = walls

    # Check cache first — key is rounded env origin (terrain cell identity)
    cache_key = (round(env_origin[0]), round(env_origin[1]))
    if cache_key in self._grid_cache:
      self.occupancy_grid = self._grid_cache[cache_key]
      return

    self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)

    # Inflate robot radius for collision checking
    inflate_cells = int(np.ceil(self.robot_radius / self.grid_resolution)) + 1

    # Terrain bounds in local coordinates (centered at origin)
    half_w = self.terrain_size[0] / 2
    half_h = self.terrain_size[1] / 2

    walls_in_range = 0
    for wx, wy, half_sx, half_sy, angle in walls:
      # Convert to local coordinates relative to terrain center
      local_x = wx - env_origin[0]
      local_y = wy - env_origin[1]

      # Skip walls that are completely outside the terrain cell
      # Account for wall size when checking bounds
      wall_max_extent = max(half_sx, half_sy) + self.robot_radius
      if (
        local_x + wall_max_extent < -half_w
        or local_x - wall_max_extent > half_w
        or local_y + wall_max_extent < -half_h
        or local_y - wall_max_extent > half_h
      ):
        continue

      walls_in_range += 1

      # Convert to grid coordinates (terrain center is at grid center)
      center_gx = int((local_x + self.terrain_size[0] / 2) / self.grid_resolution)
      center_gy = int((local_y + self.terrain_size[1] / 2) / self.grid_resolution)

      # Compute rotated bounding box
      cos_a = np.cos(angle)
      sin_a = np.sin(angle)

      # Corners of the wall in local frame
      corners = [
        (half_sx, half_sy),
        (-half_sx, half_sy),
        (-half_sx, -half_sy),
        (half_sx, -half_sy),
      ]

      # Rotate corners and find bounding box
      rotated_corners = []
      for cx, cy in corners:
        rx = cx * cos_a - cy * sin_a
        ry = cx * sin_a + cy * cos_a
        rotated_corners.append((rx, ry))

      min_x = min(c[0] for c in rotated_corners)
      max_x = max(c[0] for c in rotated_corners)
      min_y = min(c[1] for c in rotated_corners)
      max_y = max(c[1] for c in rotated_corners)

      # Convert to grid cells and inflate
      # First compute raw grid coordinates
      raw_min_gx = center_gx + int(min_x / self.grid_resolution) - inflate_cells
      raw_max_gx = center_gx + int(max_x / self.grid_resolution) + inflate_cells
      raw_min_gy = center_gy + int(min_y / self.grid_resolution) - inflate_cells
      raw_max_gy = center_gy + int(max_y / self.grid_resolution) + inflate_cells

      # Clamp to grid bounds
      min_gx = max(0, raw_min_gx)
      max_gx = min(self.grid_width - 1, raw_max_gx)
      min_gy = max(0, raw_min_gy)
      max_gy = min(self.grid_height - 1, raw_max_gy)

      # Skip if wall doesn't overlap with grid at all
      if max_gx < min_gx or max_gy < min_gy:
        walls_in_range -= 1  # Undo the counter increment
        continue

      # Mark cells as occupied
      self.occupancy_grid[min_gy : max_gy + 1, min_gx : max_gx + 1] = True

    # Store count of walls in range for debugging
    self._walls_in_range_count = walls_in_range

    # Add boundary walls
    boundary_cells = int(
      np.ceil((self._boundary_margin + self.robot_radius) / self.grid_resolution)
    )
    # Bottom boundary
    self.occupancy_grid[:boundary_cells, :] = True
    # Top boundary
    self.occupancy_grid[-boundary_cells:, :] = True
    # Left boundary
    self.occupancy_grid[:, :boundary_cells] = True
    # Right boundary
    self.occupancy_grid[:, -boundary_cells:] = True

    # Store in cache for future reuse
    self._grid_cache[cache_key] = self.occupancy_grid

  def world_to_grid(
    self, world_pos: np.ndarray, env_origin: np.ndarray
  ) -> tuple[int, int]:
    """Convert world position to grid coordinates.

    Args:
      world_pos: Position in world frame [x, y] or [x, y, z].
      env_origin: Environment origin in world frame.

    Returns:
      Grid coordinates (gx, gy).
    """
    local_x = world_pos[0] - env_origin[0]
    local_y = world_pos[1] - env_origin[1]
    gx = int((local_x + self.terrain_size[0] / 2) / self.grid_resolution)
    gy = int((local_y + self.terrain_size[1] / 2) / self.grid_resolution)
    gx = max(0, min(self.grid_width - 1, gx))
    gy = max(0, min(self.grid_height - 1, gy))
    return gx, gy

  def grid_to_world(
    self, gx: int, gy: int, env_origin: np.ndarray
  ) -> tuple[float, float]:
    """Convert grid coordinates to world position.

    Args:
      gx, gy: Grid coordinates.
      env_origin: Environment origin in world frame.

    Returns:
      World position (x, y).
    """
    local_x = gx * self.grid_resolution - self.terrain_size[0] / 2
    local_y = gy * self.grid_resolution - self.terrain_size[1] / 2
    world_x = local_x + env_origin[0]
    world_y = local_y + env_origin[1]
    return world_x, world_y

  def is_cell_free(self, gx: int, gy: int) -> bool:
    """Check if a grid cell is free (not occupied)."""
    if self.occupancy_grid is None:
      return True
    if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
      return not self.occupancy_grid[gy, gx]
    return False

  def is_position_free(self, world_pos: np.ndarray, env_origin: np.ndarray) -> bool:
    """Check if a world position is free."""
    gx, gy = self.world_to_grid(world_pos, env_origin)
    return self.is_cell_free(gx, gy)

  # 8-directional movement: (dx, dy, cost)
  _DIRS_8: tuple[tuple[int, int, float], ...] = (
    (1, 0, 1.0),
    (1, -1, 1.414),
    (0, -1, 1.0),
    (-1, -1, 1.414),
    (-1, 0, 1.0),
    (-1, 1, 1.414),
    (0, 1, 1.0),
    (1, 1, 1.414),
  )

  def find_path(
    self,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    env_origin: np.ndarray,
    prefer_right: bool = True,
  ) -> list[tuple[float, float]] | None:
    """Find path from start to goal using A* pathfinding.

    Args:
      start_pos: Start position in world frame [x, y] or [x, y, z].
      goal_pos: Goal position in world frame [x, y] or [x, y, z].
      env_origin: Environment origin in world frame.
      prefer_right: If True, add small bias to prefer right-side paths.

    Returns:
      List of waypoints [(x, y), ...] in world frame, or None if no path found.
    """
    if self.occupancy_grid is None:
      return [(float(goal_pos[0]), float(goal_pos[1]))]

    start_gx, start_gy = self.world_to_grid(start_pos, env_origin)
    goal_gx, goal_gy = self.world_to_grid(goal_pos, env_origin)

    # If start is inside an obstacle, snap to nearest free cell
    if not self.is_cell_free(start_gx, start_gy):
      result = self._find_nearest_free_cell(start_gx, start_gy)
      if result[0] is not None:
        start_gx, start_gy = result[0], result[1]

    # If goal is in obstacle, snap to nearest free cell
    if not self.is_cell_free(goal_gx, goal_gy):
      result = self._find_nearest_free_cell(goal_gx, goal_gy)
      if result[0] is None:
        return None
      goal_gx, goal_gy = result[0], result[1]

    # A* search
    w = self.grid_width
    h = self.grid_height
    grid = self.occupancy_grid
    dirs = self._DIRS_8

    # Heuristic: Chebyshev-ish distance (admissible for 8-dir)
    _abs = abs
    goal_gx_val = goal_gx
    goal_gy_val = goal_gy

    # Priority queue: (f_score, counter, gx, gy)
    counter = 0
    _heappush = heapq.heappush
    _heappop = heapq.heappop

    g_score = np.full((h, w), float("inf"), dtype=np.float32)
    g_score[start_gy, start_gx] = 0.0
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed = np.zeros((h, w), dtype=bool)

    dx0 = _abs(start_gx - goal_gx_val)
    dy0 = _abs(start_gy - goal_gy_val)
    h0 = max(dx0, dy0) + 0.414 * min(dx0, dy0)
    heap: list[tuple[float, int, int, int]] = [(h0, counter, start_gx, start_gy)]

    while heap:
      _, _, cx, cy = _heappop(heap)

      if cx == goal_gx_val and cy == goal_gy_val:
        # Reconstruct path
        path_grid: list[tuple[int, int]] = []
        node = (goal_gx_val, goal_gy_val)
        while node in came_from:
          path_grid.append(node)
          node = came_from[node]
        path_grid.append(node)
        path_grid.reverse()
        return self._simplify_path(
          path_grid, env_origin, waypoint_spacing=self.waypoint_spacing
        )

      if closed[cy, cx]:
        continue
      closed[cy, cx] = True

      g_cur = g_score[cy, cx]

      for ddx, ddy, cost in dirs:
        nx, ny = cx + ddx, cy + ddy
        if not (0 <= nx < w and 0 <= ny < h):
          continue
        if closed[ny, nx]:
          continue
        if grid is not None and grid[ny, nx]:
          continue
        ng = g_cur + cost
        if ng < g_score[ny, nx]:
          g_score[ny, nx] = ng
          came_from[(nx, ny)] = (cx, cy)
          dx1 = _abs(nx - goal_gx_val)
          dy1 = _abs(ny - goal_gy_val)
          heur = max(dx1, dy1) + 0.414 * min(dx1, dy1)
          counter += 1
          _heappush(heap, (float(ng + heur), counter, nx, ny))  # type: ignore[arg-type]

    return None  # No path found

  def _find_nearest_free_cell(
    self, gx: int, gy: int, max_radius: int = 10
  ) -> tuple[int, int] | tuple[None, None]:
    """Find nearest free cell to given grid position."""
    grid = self.occupancy_grid
    w = self.grid_width
    h = self.grid_height
    for r in range(1, max_radius + 1):
      for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
          if abs(dx) == r or abs(dy) == r:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < w and 0 <= ny < h and (grid is None or not grid[ny, nx]):
              return nx, ny
    return None, None

  def _simplify_path(
    self,
    path_grid: list[tuple[int, int]],
    env_origin: np.ndarray,
    waypoint_spacing: float = 2.0,
  ) -> list[tuple[float, float]]:
    """Simplify grid path to sparse waypoints.

    Args:
      path_grid: Path as list of grid coordinates.
      env_origin: Environment origin in world frame.
      waypoint_spacing: Minimum distance between waypoints in meters.

    Returns:
      Simplified path as list of world coordinates.
    """
    n = len(path_grid)
    if n <= 2:
      return [self.grid_to_world(gx, gy, env_origin) for gx, gy in path_grid]

    # Batch convert grid coords to world coords using vectorized math
    grid_arr = np.array(path_grid, dtype=np.float64)  # [N, 2] — (gx, gy)
    world_x = (
      grid_arr[:, 0] * self.grid_resolution - self.terrain_size[0] / 2 + env_origin[0]
    )
    world_y = (
      grid_arr[:, 1] * self.grid_resolution - self.terrain_size[1] / 2 + env_origin[1]
    )

    # Compute segment distances (vectorized)
    dx = np.diff(world_x)
    dy = np.diff(world_y)
    segment_dists = np.sqrt(dx * dx + dy * dy)

    # Simplify: pick points where accumulated distance >= waypoint_spacing
    simplified = [(float(world_x[0]), float(world_y[0]))]
    accumulated_dist = 0.0
    last_idx = n - 1

    for i in range(len(segment_dists)):
      accumulated_dist += segment_dists[i]
      if accumulated_dist >= waypoint_spacing or (i + 1) == last_idx:
        simplified.append((float(world_x[i + 1]), float(world_y[i + 1])))
        accumulated_dist = 0.0

    return simplified

  def find_alternative_goal(
    self,
    robot_pos: np.ndarray,
    robot_yaw: float,
    target_distance: float,
    env_origin: np.ndarray,
    search_angle_range: float = np.pi / 2,
    num_candidates: int = 12,
    min_distance: float = 3.0,
  ) -> tuple[float, float] | None:
    """Find an alternative reachable goal, preferring robot's right side.

    Checks candidate goals by verifying they are in free cells and that
    A* can find a path to them.

    Args:
      robot_pos: Robot position in world frame [x, y] or [x, y, z].
      robot_yaw: Robot heading in world frame (radians).
      target_distance: Desired distance to goal.
      env_origin: Environment origin in world frame.
      search_angle_range: Angle range to search (negative = right side).
      num_candidates: Number of candidate angles to try.
      min_distance: Minimum distance to search.

    Returns:
      Alternative goal position (x, y) or None if no reachable goal found.
    """
    half_terrain = min(self.terrain_size[0], self.terrain_size[1]) / 2 - 2.0
    max_search_dist = min(target_distance, half_terrain)

    # Build distances to try
    distances_to_try: list[float] = []
    current_dist = max_search_dist
    while current_dist >= min_distance:
      distances_to_try.append(current_dist)
      current_dist -= 3.0
    if not distances_to_try:
      distances_to_try = [min_distance]

    _cos = math.cos
    _sin = math.sin

    for search_dist in distances_to_try:
      # Right side first
      angles = np.linspace(0, -search_angle_range, num_candidates)
      for angle_offset in angles:
        goal_angle = robot_yaw + float(angle_offset)
        goal_x = robot_pos[0] + search_dist * _cos(goal_angle)
        goal_y = robot_pos[1] + search_dist * _sin(goal_angle)
        gx, gy = self.world_to_grid(np.array([goal_x, goal_y]), env_origin)
        if self.is_cell_free(gx, gy):
          path = self.find_path(robot_pos, np.array([goal_x, goal_y]), env_origin)
          if path is not None:
            return float(goal_x), float(goal_y)

      # Left side fallback
      angles_left = np.linspace(0, search_angle_range, num_candidates)
      for angle_offset in angles_left[1:]:
        goal_angle = robot_yaw + float(angle_offset)
        goal_x = robot_pos[0] + search_dist * _cos(goal_angle)
        goal_y = robot_pos[1] + search_dist * _sin(goal_angle)
        gx, gy = self.world_to_grid(np.array([goal_x, goal_y]), env_origin)
        if self.is_cell_free(gx, gy):
          path = self.find_path(robot_pos, np.array([goal_x, goal_y]), env_origin)
          if path is not None:
            return float(goal_x), float(goal_y)

    return None

  def compute_path_length(self, path: list[tuple[float, float]]) -> float:
    """Compute total length of a path."""
    if len(path) < 2:
      return 0.0

    total_length = 0.0
    _hypot = math.hypot
    for i in range(1, len(path)):
      total_length += _hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
    return total_length

  def estimate_travel_time(
    self,
    path: list[tuple[float, float]],
    forward_velocity: float,
    turn_velocity: float = 0.5,
  ) -> float:
    """Estimate travel time for a path considering turns.

    Args:
      path: Path as list of (x, y) waypoints.
      forward_velocity: Robot forward velocity.
      turn_velocity: Angular velocity for turning (rad/s).

    Returns:
      Estimated travel time in seconds.
    """
    if len(path) < 2:
      return 0.0

    total_time = 0.0
    inv_vel = 1.0 / max(forward_velocity, 0.1)
    inv_turn_vel = 1.0 / max(turn_velocity, 0.1)
    _pi = math.pi
    _two_pi = 2.0 * _pi
    _atan2 = math.atan2
    _hypot = math.hypot

    for i in range(1, len(path)):
      # Distance to next waypoint
      dx = path[i][0] - path[i - 1][0]
      dy = path[i][1] - path[i - 1][1]
      dist = _hypot(dx, dy)

      # Time to travel straight
      total_time += dist * inv_vel

      # Angle change at this waypoint
      if i < len(path) - 1:
        # Direction to next waypoint
        next_dx = path[i + 1][0] - path[i][0]
        next_dy = path[i + 1][1] - path[i][1]
        current_angle = _atan2(dy, dx)
        next_angle = _atan2(next_dy, next_dx)
        # Inline wrap_to_pi for scalar (avoid creating torch tensor)
        angle_diff = next_angle - current_angle
        angle_diff = (angle_diff + _pi) % _two_pi - _pi
        total_time += abs(angle_diff) * inv_turn_vel

    return total_time


class ReachableGoalCommand(CommandTerm):
  """Command term for reachable goal tracking with path planning.

  The goal point is computed based on resample interval and target velocity,
  then FIXED in world frame. Robot walks towards this fixed goal, and the
  distance decreases as it moves.

  NEW: Path planning feature for maze environments:
  - 2D occupancy grid built from terrain wall geometry
  - A* path planning to find optimal route to goal
  - If goal is unreachable, finds alternative goal on robot's right side
  - Intermediate waypoints guide robot around obstacles
  - Resample duration computed from planned path length (travel + turn time)

  Command output: [delta_yaw, target_velocity]
  - delta_yaw: angle from robot heading to CURRENT WAYPOINT (not final goal)
  - target_velocity: desired forward speed towards the waypoint

  Additional properties for RA training:
  - goal_pos_w: Fixed FINAL goal position in world frame [num_envs, 3]
  - current_waypoint_w: Current intermediate waypoint [num_envs, 3]
  - distance_to_goal: Distance to final goal [num_envs]
  - goal_reached: Whether robot has reached the final goal [num_envs]
  """

  cfg: "ReachableGoalCommandCfg"

  def __init__(self, cfg: "ReachableGoalCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._env = env

    # === Path Planning ===
    # Initialize path planner (will be configured when terrain is available)
    self._path_planner: MazePathPlanner | None = None
    # Cache the last env_origin used for building occupancy grid
    # This allows us to detect when robot moves to a different terrain cell
    self._last_grid_origin: np.ndarray | None = None

    # Maximum number of waypoints per environment (terrain 60m / spacing 2m ≈ 30)
    self._MAX_WAYPOINTS = 32

    # Planned waypoints for each environment (Python lists, used only for debug_vis)
    self._planned_waypoints: list[list[tuple[float, float]]] = [
      [] for _ in range(self.num_envs)
    ]
    # Padded waypoints tensor: [num_envs, MAX_WAYPOINTS, 2]
    # The last valid waypoint is always the final goal.
    # Entries beyond num_waypoints are copies of the last valid waypoint.
    self._waypoints_tensor = torch.zeros(
      self.num_envs, self._MAX_WAYPOINTS, 2, device=self.device
    )
    # Number of valid waypoints per env
    self._num_waypoints = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    # Current waypoint index for each environment
    self._current_waypoint_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Current waypoint position (intermediate target)
    self.current_waypoint_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Fixed FINAL goal position in world frame (computed at resample, stays fixed)
    self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Target goal distance (computed based on resample_time * velocity)
    self.target_distance = torch.zeros(self.num_envs, device=self.device)

    # Current shrink radius for the "circular ring" visualization
    self.current_ring_radius = torch.zeros(self.num_envs, device=self.device)
    self.current_ring_radius[:] = cfg.initial_ring_radius

    # Base target velocity is sampled on goal resample; the runtime command can
    # be clipped further by the time-to-stop safety envelope.
    self._base_target_velocity_cmd = torch.full(
      (self.num_envs,), float(cfg.target_velocity), device=self.device
    )
    self._target_velocity_cmd = torch.full(
      (self.num_envs,), float(cfg.target_velocity), device=self.device
    )
    self.target_velocity = self._target_velocity_cmd

    # Computed values for observations
    self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
    self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
    self.distance_to_waypoint = torch.zeros(self.num_envs, device=self.device)
    self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # Timer for delayed resample after goal reach (counts down from reach_resample_delay to 0)
    # Negative or zero means no pending resample.  Positive means waiting.
    self._reach_resample_timer = torch.zeros(self.num_envs, device=self.device)

    # Flag set to True for one step after a goal resample (reach / timeout / reset).
    # External consumers (e.g. RA trainer) can use this to treat each goal
    # attempt as a separate mini-episode.
    self.just_resampled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Store the resample interval for each environment (for distance calculation)
    self.resample_interval = torch.zeros(self.num_envs, device=self.device)

    # Path planning statistics
    self.path_length = torch.zeros(self.num_envs, device=self.device)
    self.num_waypoints = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.alternative_goal_used = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Flag for environments that need goal resampling (robot position was invalid at reset)
    self._needs_goal_resample = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Command output: [delta_yaw, target_velocity]
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._command[:, 1] = self._target_velocity_cmd

    # Metrics
    self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["distance_to_waypoint"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["goal_reached_count"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["path_replanned_count"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["alternative_goal_count"] = torch.zeros(
      self.num_envs, device=self.device
    )

    self._safety_front_clearance = torch.full(
      (self.num_envs,), float("inf"), device=self.device
    )
    self._safety_left_clearance = torch.full(
      (self.num_envs,), float("inf"), device=self.device
    )
    self._safety_right_clearance = torch.full(
      (self.num_envs,), float("inf"), device=self.device
    )
    self._safety_required_stop_distance = torch.zeros(self.num_envs, device=self.device)
    self._safety_velocity_scale = torch.ones(self.num_envs, device=self.device)
    self._safety_hard_stop = torch.zeros(self.num_envs, device=self.device)

  def _query_terrain_height(self, x: float, y: float, fallback_z: float) -> float:
    """Query terrain surface height at (x, y) using a downward ray cast.

    Shoots a ray from high above straight down. If it hits any static geom,
    returns the hit-point Z plus a small offset so the goal sits on the
    surface. Falls back to *fallback_z* when the ray misses (e.g. outside
    the terrain).
    """
    try:
      model = self._env.sim.model._model  # type: ignore[attr-defined]
      data = self._env.sim.data._data  # type: ignore[attr-defined]
    except AttributeError:
      return fallback_z

    ray_start = np.array([x, y, fallback_z + 50.0])
    ray_dir = np.array([0.0, 0.0, -1.0])
    geom_id_arr = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, ray_start, ray_dir, None, 1, -1, geom_id_arr)
    if dist >= 0:
      surface_z = ray_start[2] - dist
      return float(surface_z) + 0.05  # Slight offset above surface
    return fallback_z

  def _initialize_path_planner(self) -> None:
    """Initialize the path planner with terrain information."""
    if self._path_planner is not None:
      return

    # When path planning is disabled, skip initialisation entirely so that
    # _resample_command falls back to scene.env_origins for env_origin and
    # uses a generous default clamp radius.  This avoids snapping goals to
    # per-tile cell centres which causes goals to cluster near the robot
    # when it walks close to a tile boundary.
    if not self.cfg.path_planning.enabled:
      _logger.info("[PathPlanner] Path planning disabled — skipping planner init.")
      return

    _logger.info("[PathPlanner] Initializing path planner...")
    # print("\n[PathPlanner] ====== Initializing Path Planner ======")
    # print(f"[PathPlanner] Path planning enabled: {self.cfg.path_planning.enabled}")

    # Get terrain configuration
    terrain = self._env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
      # No terrain, use simple planner without obstacles
      # print("[PathPlanner] No terrain generator found, using default 60x60m terrain")
      self._path_planner = MazePathPlanner(
        terrain_size=(60.0, 60.0),
        grid_resolution=self.cfg.path_planning.grid_resolution,
        robot_radius=self.cfg.path_planning.robot_radius,
        waypoint_spacing=self.cfg.path_planning.waypoint_spacing,
        device=self.device,
      )
      return

    # Get terrain size from generator config
    gen_cfg = terrain.cfg.terrain_generator
    terrain_size = gen_cfg.size

    # print(f"[PathPlanner] Terrain size: {terrain_size}")
    # print(f"[PathPlanner] Grid resolution: {self.cfg.path_planning.grid_resolution}m")
    # print(f"[PathPlanner] Robot radius: {self.cfg.path_planning.robot_radius}m")

    self._path_planner = MazePathPlanner(
      terrain_size=terrain_size,
      grid_resolution=self.cfg.path_planning.grid_resolution,
      robot_radius=self.cfg.path_planning.robot_radius,
      waypoint_spacing=self.cfg.path_planning.waypoint_spacing,
      device=self.device,
    )

    # print(
    #   f"[PathPlanner] Grid dimensions: {self._path_planner.grid_width} x {self._path_planner.grid_height}"
    # )
    # print("[PathPlanner] =======================================\n")

  def _build_occupancy_grid_for_env(self, env_id: int) -> None:
    """Build occupancy grid for a specific environment based on its terrain."""
    env_origin = self._env.scene.env_origins[env_id].cpu().numpy()
    self._build_occupancy_grid_for_env_at(env_id, env_origin)

  def _build_occupancy_grid_for_env_at(
    self, env_id: int, env_origin: np.ndarray
  ) -> None:
    """Build occupancy grid for a specific environment at given origin.

    This extracts wall geometry from the MuJoCo model and builds a 2D occupancy grid.
    The grid is only rebuilt when the terrain cell changes (different env_origin).

    Args:
      env_id: Environment ID (used for logging only)
      env_origin: The terrain cell center in world coordinates [x, y, z]
    """
    if self._path_planner is None:
      self._initialize_path_planner()

    if self._path_planner is None:
      # print(f"[PathPlanner] ERROR: Path planner is None for env {env_id}")
      return

    # Check if we need to rebuild the grid (different terrain cell)
    need_rebuild = self._last_grid_origin is None
    if not need_rebuild and self._last_grid_origin is not None:
      # Check if the terrain cell has changed (origin moved by more than half terrain size)
      origin_diff = np.linalg.norm(env_origin[:2] - self._last_grid_origin[:2])
      terrain_half = min(self._path_planner.terrain_size) / 2
      need_rebuild = origin_diff > terrain_half * 0.5  # Allow some tolerance

    if not need_rebuild:
      # Same terrain cell — occupancy grid is still valid.
      # Flood-fill will be recomputed by find_path when start position changes.
      return

    # Extract wall geometries from the simulation
    # Walls are BOX geoms in the terrain body
    walls: list[tuple[float, float, float, float, float]] = []

    try:
      # Use mj_model for accessing model structure
      mj_model = self._env.sim.mj_model

      # if need_rebuild:
      #   print(
      #     f"[PathPlanner] MuJoCo model has {mj_model.nbody} bodies, {mj_model.ngeom} geoms"
      #   )

      # Find terrain body
      terrain_body_id = -1
      for i in range(mj_model.nbody):
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and "terrain" in body_name.lower():
          terrain_body_id = i
          break

      if terrain_body_id < 0:
        # No terrain body found, build grid with just boundaries
        # if need_rebuild:
        #   print("[PathPlanner] WARNING: No terrain body found!")
        #   print("[PathPlanner] Available bodies:")
        #   for i in range(min(20, mj_model.nbody)):  # Print first 20 bodies
        #     body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"  - Body {i}: {body_name}")
        self._path_planner.build_occupancy_grid_from_walls([], env_origin)
        self._last_grid_origin = env_origin.copy()
        return

      # if need_rebuild:
      #   print(
      #     f"[PathPlanner] Found terrain body: '{terrain_body_name}' (id={terrain_body_id})"
      #   )

      # Iterate through geoms in terrain body.
      # Include ALL solid geoms that could block navigation:
      #   - BOX: walls, pillars, furniture (any aspect ratio)
      #   - CYLINDER / CAPSULE / SPHERE / ELLIPSOID: post-like obstacles
      # Exclude flat ground planes (half-height < 0.05 m for BOX,
      # or radius < 0.05 m for round shapes).
      _OBSTACLE_TYPES = {
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
      }
      _GROUND_HEIGHT_THRESHOLD = 0.05  # ignore flat ground-plane geoms
      obstacle_geoms_in_terrain = 0
      for geom_id in range(mj_model.ngeom):
        # Check if this geom belongs to terrain body
        if mj_model.geom_bodyid[geom_id] != terrain_body_id:
          continue

        geom_type = mj_model.geom_type[geom_id]
        if geom_type not in _OBSTACLE_TYPES:
          continue

        obstacle_geoms_in_terrain += 1

        # Get geom position and size
        geom_pos = mj_model.geom_pos[geom_id]
        geom_size = mj_model.geom_size[geom_id]  # half-sizes

        # Get geom rotation (quaternion → z-axis rotation angle)
        geom_quat = mj_model.geom_quat[geom_id]
        angle = 2 * np.arctan2(geom_quat[3], geom_quat[0])

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
          # geom_size = [half_x, half_y, half_z]
          # Skip flat ground planes (half_z is very small)
          if geom_size[2] < _GROUND_HEIGHT_THRESHOLD:
            continue
          half_sx = float(geom_size[0])
          half_sy = float(geom_size[1])
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
          # geom_size = [radius, 0, 0]
          r = float(geom_size[0])
          if r < _GROUND_HEIGHT_THRESHOLD:
            continue
          half_sx = r
          half_sy = r
        elif geom_type in (
          mujoco.mjtGeom.mjGEOM_CYLINDER,
          mujoco.mjtGeom.mjGEOM_CAPSULE,
        ):
          # geom_size = [radius, half_length, 0]
          r = float(geom_size[0])
          if r < _GROUND_HEIGHT_THRESHOLD:
            continue
          half_sx = r
          half_sy = r
        else:  # ELLIPSOID: geom_size = [half_x, half_y, half_z]
          if geom_size[2] < _GROUND_HEIGHT_THRESHOLD:
            continue
          half_sx = float(geom_size[0])
          half_sy = float(geom_size[1])

        walls.append(
          (
            float(geom_pos[0]),
            float(geom_pos[1]),
            half_sx,
            half_sy,
            float(angle),
          )
        )

      # if need_rebuild:
      #   print(f"[PathPlanner] Found {box_geoms_in_terrain} BOX geoms in terrain")
      #   print(f"[PathPlanner] Identified {len(walls)} walls (thin boxes)")
      #   if len(walls) > 0:
      #     print(
      #       f"[PathPlanner] First wall: pos=({walls[0][0]:.1f}, {walls[0][1]:.1f}), "
      #       f"size=({walls[0][2]:.2f}, {walls[0][3]:.2f}), angle={np.degrees(walls[0][4]):.1f}°"
      #     )

    except Exception as e:
      _logger.error(f"[PathPlanner] ERROR: Failed to extract wall geometry: {e}")
      # import traceback
      # traceback.print_exc()
      # Build grid with just boundaries
      self._path_planner.build_occupancy_grid_from_walls([], env_origin)
      self._last_grid_origin = env_origin.copy()
      return

    # Build occupancy grid
    self._path_planner.build_occupancy_grid_from_walls(walls, env_origin)

    # if need_rebuild:
    #   # Count occupied cells and walls in range
    #   walls_in_range = getattr(self._path_planner, "_walls_in_range_count", 0)
    #   print(
    #     f"[PathPlanner] Walls in current terrain cell: {walls_in_range}/{len(walls)}"
    #   )
    #   if self._path_planner.occupancy_grid is not None:
    #     occupied_count = np.sum(self._path_planner.occupancy_grid)
    #     total_cells = self._path_planner.occupancy_grid.size
    #     print(
    #       f"[PathPlanner] Occupancy grid built: {occupied_count}/{total_cells} cells occupied "
    #       f"({100 * occupied_count / total_cells:.1f}%)"
    #     )
    #   print("[PathPlanner] =======================================\n")

    # Update cached origin
    self._last_grid_origin = env_origin.copy()

  def compute(self, dt: float) -> None:
    """Override base to clear ``just_resampled`` before any resamples.

    The flag is cleared here (before ``_resample`` / ``_update_command``)
    so that any resample triggered during this step will set it to True
    and it will stay True until the *next* ``compute()`` call.
    """
    self.just_resampled[:] = False
    super().compute(dt)

  @property
  def command(self) -> torch.Tensor:
    """Return the command tensor [delta_yaw, target_velocity]."""
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample goal position for specified environments with path planning.

    1. Sample initial goal position based on robot heading and target distance
    2. Check if goal is reachable using A* path planning
    3. If unreachable, find alternative goal on robot's right side
    4. Plan path with intermediate waypoints
    5. Compute resample duration from path length (travel + turn time)
    """
    n = len(env_ids)
    if n == 0:
      return

    # Initialize path planner if needed
    if self._path_planner is None:
      self._initialize_path_planner()

    # Get current robot state for these environments
    robot_pos = self.robot.data.root_link_pos_w[env_ids, :]  # [n, 3]
    robot_yaw = self.robot.data.heading_w[env_ids]  # [n]

    # NOTE: At reset time, robot position data might not be updated yet because
    # physics simulation hasn't stepped. Check if robot position is near world origin (0,0).
    # If so, we'll set a temporary goal nearby the env_origin.
    # The goal will be resampled on the next update once robot position is valid.
    env_origins = self._env.scene.env_origins[env_ids, :]  # [n, 3]

    # Check for uninitialized robot position (near world origin 0,0)
    robot_near_world_origin = torch.norm(robot_pos[:, :2], dim=1) < 1.0
    needs_resample_later = robot_near_world_origin.clone()

    if robot_near_world_origin.any():
      robot_pos = robot_pos.clone()
      robot_pos[robot_near_world_origin] = env_origins[robot_near_world_origin]

    # Sample target velocity from configured range
    vel_low, vel_high = self.cfg.target_velocity_range
    self._base_target_velocity_cmd[env_ids] = torch.empty(
      n, device=self.device
    ).uniform_(vel_low, vel_high)
    self._target_velocity_cmd[env_ids] = self._base_target_velocity_cmd[env_ids]

    # Sample initial resample interval to compute initial target distance
    low, high = self.cfg.resampling_time_range
    initial_interval = torch.empty(n, device=self.device).uniform_(low, high)
    # Process each environment
    for i, env_id in enumerate(env_ids.tolist()):
      robot_pos_i = robot_pos[i].cpu().numpy()
      robot_yaw_i = float(robot_yaw[i].item())
      target_vel_i = float(self._base_target_velocity_cmd[env_id].item())
      initial_time_i = float(initial_interval[i].item())

      # Calculate the actual terrain cell center based on robot position
      # In play mode, robot might be in a different terrain cell than scene_env_origins[env_id]
      if self._path_planner is not None:
        terrain_w, terrain_h = self._path_planner.terrain_size
        # Calculate which terrain cell the robot is in (snap to cell center)
        cell_x = np.floor(robot_pos_i[0] / terrain_w) * terrain_w + terrain_w / 2
        cell_y = np.floor(robot_pos_i[1] / terrain_h) * terrain_h + terrain_h / 2
        env_origin = np.array([cell_x, cell_y, robot_pos_i[2]])
      else:
        env_origin = self._env.scene.env_origins[env_id].cpu().numpy()

      # If robot position was invalid (near world origin), skip path planning for this resample.
      # Just set a temporary goal - it will be resampled properly once robot position is valid.
      if needs_resample_later[i]:
        # print(
        #   f"[PathPlanner] Env {env_id}: Skipping path planning (robot position invalid, will resample later)"
        # )
        # Set a temporary goal near the robot's estimated position (robot_pos_i was already
        # replaced with env_origin for invalid positions, so it should be close to where
        # the robot will actually spawn)
        temp_goal_dist = 3.0  # Short distance
        # Use robot_pos_i which has been adjusted to env_origin for invalid positions
        temp_goal_x = robot_pos_i[0] + temp_goal_dist
        temp_goal_y = robot_pos_i[1]

        self.goal_pos_w[env_id] = torch.tensor(
          [temp_goal_x, temp_goal_y, robot_pos_i[2]], device=self.device
        )
        self.current_waypoint_w[env_id] = self.goal_pos_w[env_id]
        self._planned_waypoints[env_id] = [(float(temp_goal_x), float(temp_goal_y))]
        self._current_waypoint_idx[env_id] = 0
        self.num_waypoints[env_id] = 1
        self._num_waypoints[env_id] = 1
        self._waypoints_tensor[env_id, 0, 0] = float(temp_goal_x)
        self._waypoints_tensor[env_id, 0, 1] = float(temp_goal_y)
        self._waypoints_tensor[env_id, 1:] = self._waypoints_tensor[env_id, 0]
        self.alternative_goal_used[env_id] = False
        self.path_length[env_id] = temp_goal_dist
        self.resample_interval[env_id] = 1.0  # Very short interval - resample ASAP
        self.target_distance[env_id] = temp_goal_dist
        self.time_left[env_id] = 1.0  # Will be re-set after valid resample
        # Mark for resample once robot position becomes valid
        self._needs_goal_resample[env_id] = True
        continue  # Skip to next environment

      # Build occupancy grid if path planning is enabled
      if self.cfg.path_planning.enabled and self._path_planner is not None:
        self._build_occupancy_grid_for_env_at(env_id, env_origin)

      # Compute initial target distance
      target_dist = initial_time_i * target_vel_i
      target_dist = max(target_dist, self.cfg.min_target_distance)

      # Limit target distance to stay within terrain bounds (half of terrain size minus margin)
      if self._path_planner is not None:
        max_dist = min(self._path_planner.terrain_size) / 2 - 5.0  # 25m for 60m terrain
        original_dist = target_dist
        target_dist = min(target_dist, max_dist)
        if original_dist > max_dist:
          _logger.debug(
            f"[PathPlanner] Env {env_id}: Limiting target_dist from {original_dist:.1f}m to {target_dist:.1f}m (terrain bound)"
          )

      # Sample relative angle within configured range
      angle_range = self.cfg.goal_angle_range
      relative_angle = np.random.uniform(-angle_range, angle_range)
      goal_direction = robot_yaw_i + relative_angle

      # Compute initial goal position (relative to current terrain cell)
      goal_x = robot_pos_i[0] + target_dist * np.cos(goal_direction)
      goal_y = robot_pos_i[1] + target_dist * np.sin(goal_direction)

      # Clamp goal to stay within current terrain cell
      terrain_half = (
        min(self._path_planner.terrain_size) / 2 - 2.0 if self._path_planner else 28.0
      )
      local_goal_x = goal_x - env_origin[0]
      local_goal_y = goal_y - env_origin[1]
      local_goal_x = np.clip(local_goal_x, -terrain_half, terrain_half)
      local_goal_y = np.clip(local_goal_y, -terrain_half, terrain_half)
      goal_x = env_origin[0] + local_goal_x
      goal_y = env_origin[1] + local_goal_y

      goal_pos = np.array([goal_x, goal_y, robot_pos_i[2]])

      # Path planning
      path: list[tuple[float, float]] | None = None
      alternative_used = False

      if self.cfg.path_planning.enabled and self._path_planner is not None:
        # Try to find path to initial goal
        path = self._path_planner.find_path(
          robot_pos_i[:2], goal_pos[:2], env_origin, prefer_right=True
        )

        if path is None:
          # Goal is unreachable, find alternative on right side
          _logger.info(
            f"[PathPlanner] Env {env_id}: Initial goal unreachable, searching alternative..."
          )
          alternative = self._path_planner.find_alternative_goal(
            robot_pos_i,
            robot_yaw_i,
            target_dist,
            env_origin,
            search_angle_range=self.cfg.path_planning.alternative_search_range,
            num_candidates=self.cfg.path_planning.alternative_candidates,
            min_distance=self.cfg.min_target_distance,
          )

          if alternative is not None:
            goal_x, goal_y = alternative
            goal_pos = np.array([goal_x, goal_y, robot_pos_i[2]])
            alternative_used = True
            _logger.info(
              f"[PathPlanner] Env {env_id}: Found alternative goal at ({goal_x:.1f}, {goal_y:.1f})"
            )

            # Plan path to new goal
            path = self._path_planner.find_path(
              robot_pos_i[:2], goal_pos[:2], env_origin, prefer_right=True
            )
          else:
            _logger.warning(f"[PathPlanner] Env {env_id}: No alternative goal found!")

        if path is None:
          # Still no path, use direct line (fallback)
          # But limit the goal to a reachable distance within terrain
          max_safe_dist = min(self._path_planner.terrain_size) / 2 - 5.0
          fallback_dist = min(target_dist, max_safe_dist, 10.0)  # Max 10m for fallback
          fallback_x = robot_pos_i[0] + fallback_dist * np.cos(goal_direction)
          fallback_y = robot_pos_i[1] + fallback_dist * np.sin(goal_direction)
          goal_x, goal_y = fallback_x, fallback_y
          goal_pos = np.array([goal_x, goal_y, robot_pos_i[2]])
          _logger.warning(
            f"[PathPlanner] Env {env_id}: Using fallback goal at ({goal_x:.1f}, {goal_y:.1f}), dist={fallback_dist:.1f}m"
          )
          path = [(float(goal_x), float(goal_y))]
        else:
          # Path found - use the last waypoint as the actual goal
          # The last waypoint from A* is guaranteed to be within the terrain
          goal_x, goal_y = path[-1]
          goal_pos = np.array([goal_x, goal_y, robot_pos_i[2]])
          _logger.debug(
            f"[PathPlanner] Env {env_id}: Path {len(path)} waypoints, goal at ({goal_x:.1f},{goal_y:.1f})"
          )

      else:
        # No path planning, direct line to goal
        path = [(float(goal_x), float(goal_y))]

      # Store planned waypoints
      self._planned_waypoints[env_id] = path
      self._current_waypoint_idx[env_id] = 0
      n_wp = len(path)
      self.num_waypoints[env_id] = n_wp
      self._num_waypoints[env_id] = n_wp
      self.alternative_goal_used[env_id] = alternative_used

      # Populate padded waypoints tensor
      clamped_n = min(n_wp, self._MAX_WAYPOINTS)
      for wi in range(clamped_n):
        self._waypoints_tensor[env_id, wi, 0] = float(path[wi][0])
        self._waypoints_tensor[env_id, wi, 1] = float(path[wi][1])
      # Pad remaining slots with the last waypoint (so gather beyond n_wp is safe)
      if clamped_n > 0:
        last_wp = self._waypoints_tensor[env_id, clamped_n - 1]
        self._waypoints_tensor[env_id, clamped_n:] = last_wp

      if alternative_used:
        self.metrics["alternative_goal_count"][env_id] += 1

      # Set final goal position to the actual path endpoint (convert numpy to float for CUDA tensor assignment)
      # Query terrain surface height so the goal sits on top of the terrain
      # (not at robot Z, which may differ on rough/sloped terrain).
      goal_z = self._query_terrain_height(
        float(goal_x), float(goal_y), float(robot_pos_i[2])
      )
      self.goal_pos_w[env_id, 0] = float(goal_x)
      self.goal_pos_w[env_id, 1] = float(goal_y)
      self.goal_pos_w[env_id, 2] = goal_z

      # Set current waypoint (first in path)
      if len(path) > 0:
        self.current_waypoint_w[env_id, 0] = float(path[0][0])
        self.current_waypoint_w[env_id, 1] = float(path[0][1])
        self.current_waypoint_w[env_id, 2] = goal_z

      # Compute path length and resample duration
      if self._path_planner is not None and len(path) > 0:
        # Add robot position to path for length calculation
        full_path = [(float(robot_pos_i[0]), float(robot_pos_i[1]))] + path
        path_length = self._path_planner.compute_path_length(full_path)
        self.path_length[env_id] = float(path_length)
      else:
        # Fallback: use direct distance
        direct_dist = np.sqrt(
          (goal_x - robot_pos_i[0]) ** 2 + (goal_y - robot_pos_i[1]) ** 2
        )
        self.path_length[env_id] = float(direct_dist)

      # Compute per-environment resample time
      if self.cfg.path_planning.dynamic_resample_time and target_vel_i > 0:
        # Dynamic: just enough time to walk straight to the goal + margin
        direct_dist = np.sqrt(
          (goal_x - robot_pos_i[0]) ** 2 + (goal_y - robot_pos_i[1]) ** 2
        )
        dyn_time = (
          direct_dist / target_vel_i + self.cfg.path_planning.resample_time_margin
        )
        resample_time = min(dyn_time, self.cfg.path_planning.max_resample_time)
      else:
        # Fixed max resample time (legacy behaviour)
        resample_time = self.cfg.path_planning.max_resample_time

      self.resample_interval[env_id] = float(resample_time)
      self.target_distance[env_id] = float(target_dist)
      # Store per-env time_left (will be written to tensor after loop)
      self.time_left[env_id] = float(resample_time)

      # Mark environment for resample if robot position was invalid
      if needs_resample_later[i]:
        self._needs_goal_resample[env_id] = True

    # Reset ring radius, goal reached status, and resample timer
    self.current_ring_radius[env_ids] = self.cfg.initial_ring_radius
    self.goal_reached[env_ids] = False
    self._reach_resample_timer[env_ids] = 0.0

    # Mark these environments as just-resampled so the RA trainer can
    # treat each goal attempt as a separate mini-episode.
    self.just_resampled[env_ids] = True

    # Note: time_left is already set per-environment in the loop above.
    # When dynamic_resample_time is enabled each env has its own deadline;
    # otherwise every env gets max_resample_time.

    # Update command for these environments
    self._update_command_for_envs(env_ids)

  def _update_command(self) -> None:
    """Update command based on current robot state.

    Tracks current waypoint, switches to next when reached.
    Triggers early resample when goal is reached.
    """
    # Check if any environments need goal resampling (robot position was invalid at reset)
    needs_resample = self._needs_goal_resample.nonzero(as_tuple=True)[0]
    if len(needs_resample) > 0:
      # Check if robot positions are now valid
      robot_pos = self.robot.data.root_link_pos_w[needs_resample, :2]
      valid_mask = torch.norm(robot_pos, dim=1) > 1.0
      if valid_mask.any():
        valid_env_ids = needs_resample[valid_mask]
        self._needs_goal_resample[valid_env_ids] = False
        # print(
        #   f"[PathPlanner] Resampling goal for {len(valid_env_ids)} envs (robot position now valid)"
        # )
        self._resample_command(valid_env_ids)

    # Check if any environments have reached their goal
    # Instead of immediately resampling, start a delay timer so the robot
    # lingers near the goal and accumulates reach samples for RA training.
    newly_reached = self.goal_reached & (self._reach_resample_timer <= 0)
    if newly_reached.any():
      newly_reached_ids = newly_reached.nonzero(as_tuple=True)[0]
      if self.cfg.reach_resample_delay <= 0:
        # Immediate resample when delay is zero or negative
        _logger.debug(
          f"[PathPlanner] Goal reached for {len(newly_reached_ids)} envs, "
          f"resampling immediately (delay={self.cfg.reach_resample_delay})"
        )
        self._resample_command(newly_reached_ids)
      else:
        self._reach_resample_timer[newly_reached_ids] = self.cfg.reach_resample_delay
        _logger.debug(
          f"[PathPlanner] Goal reached for {len(newly_reached_ids)} envs, "
          f"waiting {self.cfg.reach_resample_delay}s before resample"
        )

    # Tick down the timer for environments that are waiting
    dt = self._env.step_dt
    waiting = self._reach_resample_timer > 0
    if waiting.any():
      self._reach_resample_timer[waiting] -= dt
      # Resample environments whose timer has expired
      expired = (self._reach_resample_timer <= 0) & self.goal_reached
      if expired.any():
        expired_ids = expired.nonzero(as_tuple=True)[0]
        _logger.debug(
          f"[PathPlanner] Reach delay expired for {len(expired_ids)} envs, resampling"
        )
        self._resample_command(expired_ids)

    all_env_ids = torch.arange(self.num_envs, device=self.device)
    self._update_command_for_envs(all_env_ids)

  def _get_raycast_clearance(
    self, sensor_name: str, env_ids: torch.Tensor
  ) -> torch.Tensor:
    """Return the horizontal clearance for one raycast sensor."""
    sensor = self._env.scene[sensor_name]
    distances = sensor.data.distances[env_ids]
    return min_raycast_clearance(
      distances,
      skip_ground_cols=self.cfg.safety_velocity_envelope.skip_ground_cols,
    )

  def _apply_safety_velocity_envelope(
    self, env_ids: torch.Tensor, delta_yaw: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip target velocity using direction-aware time-to-stop limits."""
    safety_cfg = self.cfg.safety_velocity_envelope
    if not safety_cfg.enabled:
      self._target_velocity_cmd[env_ids] = self._base_target_velocity_cmd[env_ids]
      self._safety_front_clearance[env_ids] = float("inf")
      self._safety_left_clearance[env_ids] = float("inf")
      self._safety_right_clearance[env_ids] = float("inf")
      self._safety_required_stop_distance[env_ids] = 0.0
      self._safety_velocity_scale[env_ids] = 1.0
      self._safety_hard_stop[env_ids] = 0.0
      return delta_yaw, self._target_velocity_cmd[env_ids]

    front_clearance = self._get_raycast_clearance(
      safety_cfg.forward_sensor_name, env_ids
    )
    left_clearance = self._get_raycast_clearance(safety_cfg.left_sensor_name, env_ids)
    right_clearance = self._get_raycast_clearance(safety_cfg.right_sensor_name, env_ids)

    lin_vel_b = self.robot.data.root_link_lin_vel_b[env_ids, :2]
    yaw_rate_b = self.robot.data.root_link_ang_vel_b[env_ids, 2]
    front_speed, left_speed, right_speed = compute_directional_approach_speeds(
      lin_vel_b,
      yaw_rate_b,
      side_point_forward_offset_m=safety_cfg.side_point_forward_offset_m,
    )

    front_required = compute_required_stop_distance(
      front_speed,
      safety_margin_m=safety_cfg.safety_margin_m,
      reaction_time_s=safety_cfg.reaction_time_s,
      braking_time_s=safety_cfg.braking_time_s,
    )
    left_required = compute_required_stop_distance(
      left_speed,
      safety_margin_m=safety_cfg.safety_margin_m,
      reaction_time_s=safety_cfg.reaction_time_s,
      braking_time_s=safety_cfg.braking_time_s,
    )
    right_required = compute_required_stop_distance(
      right_speed,
      safety_margin_m=safety_cfg.safety_margin_m,
      reaction_time_s=safety_cfg.reaction_time_s,
      braking_time_s=safety_cfg.braking_time_s,
    )

    # Speed limit uses FRONT clearance only.  Side clearances are handled by
    # yaw damping — they should not drag the speed to zero and strand the
    # robot in place when it needs to turn away from a frontal obstacle.
    safe_speed_limit = compute_safe_speed_limit(
      front_clearance,
      safety_margin_m=safety_cfg.safety_margin_m,
      reaction_time_s=safety_cfg.reaction_time_s,
      braking_time_s=safety_cfg.braking_time_s,
    )

    clipped_target_velocity = torch.minimum(
      self._base_target_velocity_cmd[env_ids], safe_speed_limit
    )

    # When front is blocked but the robot needs to turn (large delta_yaw)
    # and at least one side is clear, keep a minimum velocity so the policy
    # can execute turning motion instead of standing still.
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

    # Snap to zero only when NOT in turning mode
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

    self._target_velocity_cmd[env_ids] = clipped_target_velocity
    self._safety_front_clearance[env_ids] = front_clearance
    self._safety_left_clearance[env_ids] = left_clearance
    self._safety_right_clearance[env_ids] = right_clearance
    self._safety_required_stop_distance[env_ids] = torch.stack(
      [front_required, left_required, right_required], dim=1
    ).amax(dim=1)

    base_speed = torch.clamp(self._base_target_velocity_cmd[env_ids], min=1e-6)
    self._safety_velocity_scale[env_ids] = clipped_target_velocity / base_speed
    self._safety_hard_stop[env_ids] = (
      (front_clearance <= safety_cfg.hard_stop_distance_m)
      | (left_clearance <= safety_cfg.hard_stop_distance_m)
      | (right_clearance <= safety_cfg.hard_stop_distance_m)
    ).to(clipped_target_velocity.dtype)

    if "log" in self._env.extras:
      log = self._env.extras["log"]
      log["Metrics/base_target_velocity"] = torch.mean(
        self._base_target_velocity_cmd[env_ids]
      )
      log["Metrics/safety_clipped_target_velocity"] = torch.mean(
        clipped_target_velocity
      )
      log["Metrics/safety_velocity_scale"] = torch.mean(
        self._safety_velocity_scale[env_ids]
      )
      log["Metrics/safety_front_clearance_m"] = torch.mean(front_clearance)
      log["Metrics/safety_left_clearance_m"] = torch.mean(left_clearance)
      log["Metrics/safety_right_clearance_m"] = torch.mean(right_clearance)
      log["Metrics/safety_required_stop_distance_m"] = torch.mean(
        self._safety_required_stop_distance[env_ids]
      )
      log["Metrics/safety_hard_stop_fraction"] = torch.mean(
        self._safety_hard_stop[env_ids]
      )

    return delta_yaw, clipped_target_velocity

  def _update_command_for_envs(self, env_ids: torch.Tensor) -> None:
    """Update command for specific environments.

    Fully vectorized — NO Python for-loop. Waypoint data is stored in a padded
    tensor (_waypoints_tensor) so the entire update is batch tensor math.
    """
    n = len(env_ids)
    if n == 0:
      return

    # Get robot position and heading (batch)
    robot_pos = self.robot.data.root_link_pos_w[env_ids, :2]  # [n, 2]
    robot_yaw = self.robot.data.heading_w[env_ids]  # [n]

    # Check for uninitialized robot position (near world origin 0,0)
    robot_near_world_origin = torch.norm(robot_pos, dim=1) < 1.0
    if robot_near_world_origin.any():
      env_origins = self._env.scene.env_origins[env_ids, :2]
      robot_pos = robot_pos.clone()
      robot_pos[robot_near_world_origin] = env_origins[robot_near_world_origin]

    # ── Fully vectorized waypoint tracking ──
    # Gather current waypoint from padded tensor: [n, 2]
    wp_idx = self._current_waypoint_idx[env_ids]  # [n] long
    # Clamp index to valid range (max = MAX_WAYPOINTS-1, but already padded)
    n_wp = self._num_waypoints[env_ids]  # [n] long
    wp_idx = torch.clamp(wp_idx, max=self._MAX_WAYPOINTS - 1)
    # Also clamp to num_waypoints-1 for envs that have waypoints
    has_wp = n_wp > 0
    wp_idx = torch.where(has_wp, torch.min(wp_idx, n_wp - 1), wp_idx)

    # Gather: _waypoints_tensor[env_ids, wp_idx, :] → [n, 2]
    idx_expanded = wp_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)  # [n, 1, 2]
    current_wp = torch.gather(self._waypoints_tensor[env_ids], 1, idx_expanded).squeeze(
      1
    )  # [n, 2]

    # For envs with no waypoints, fall back to goal_pos_w
    no_wp = ~has_wp
    if no_wp.any():
      current_wp[no_wp] = self.goal_pos_w[env_ids[no_wp], :2]

    # Distance from robot to current waypoint (batch)
    to_wp = current_wp - robot_pos  # [n, 2]
    dist_to_wp_sq = (to_wp * to_wp).sum(dim=1)  # [n]

    # Check which envs reached their current waypoint
    wp_threshold = self.cfg.path_planning.waypoint_reach_threshold
    reached_wp = dist_to_wp_sq < (wp_threshold * wp_threshold)  # [n] bool

    # Advance waypoint index for envs that reached current wp and have more
    can_advance = reached_wp & (wp_idx < n_wp - 1)
    if can_advance.any():
      # Advance index
      new_idx = wp_idx.clone()
      new_idx[can_advance] += 1
      self._current_waypoint_idx[env_ids] = new_idx

      # Re-gather the updated waypoint for advanced envs
      new_idx_expanded = new_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
      current_wp = torch.gather(
        self._waypoints_tensor[env_ids], 1, new_idx_expanded
      ).squeeze(1)
      if no_wp.any():
        current_wp[no_wp] = self.goal_pos_w[env_ids[no_wp], :2]

    # Write current waypoint to self.current_waypoint_w
    self.current_waypoint_w[env_ids, 0] = current_wp[:, 0]
    self.current_waypoint_w[env_ids, 1] = current_wp[:, 1]

    # ── Vectorized geometry (same as before) ──
    to_waypoint = current_wp - robot_pos  # [n, 2]
    dist_to_waypoint = torch.norm(to_waypoint, dim=1)  # [n]
    self.distance_to_waypoint[env_ids] = dist_to_waypoint

    # Heading to current waypoint (batch atan2 + wrap_to_pi)
    waypoint_angle = torch.atan2(to_waypoint[:, 1], to_waypoint[:, 0])  # [n]
    delta_yaw = wrap_to_pi(waypoint_angle - robot_yaw)  # [n]
    delta_yaw, target_velocity = self._apply_safety_velocity_envelope(
      env_ids, delta_yaw
    )
    self.delta_yaw[env_ids] = delta_yaw

    # Distance to final goal  [n]
    goal_xy = self.goal_pos_w[env_ids, :2]
    to_goal = goal_xy - robot_pos
    dist_to_goal = torch.norm(to_goal, dim=1)  # [n]
    self.distance_to_goal[env_ids] = dist_to_goal

    # Goal reached check (batch)
    reached_mask = dist_to_goal < self.cfg.reach_threshold
    if reached_mask.any():
      self.goal_reached[env_ids[reached_mask]] = True
      self.metrics["goal_reached_count"][env_ids[reached_mask]] += 1

    # Shrink ring (batch)
    shrink_mask = dist_to_goal < self.cfg.shrink_threshold
    if shrink_mask.any():
      shrink_ids = env_ids[shrink_mask]
      self.current_ring_radius[shrink_ids] = dist_to_goal[shrink_mask]

    # Update command tensor (batch)
    self._command[env_ids, 0] = delta_yaw
    self._command[env_ids, 1] = target_velocity

    # Update metrics (batch)
    self.metrics["heading_error"][env_ids] = torch.abs(delta_yaw)
    self.metrics["distance_to_goal"][env_ids] = dist_to_goal
    self.metrics["distance_to_waypoint"][env_ids] = dist_to_waypoint

  def _update_metrics(self) -> None:
    """Update tracking metrics."""
    # Velocity towards waypoint direction
    lin_vel_b = self.robot.data.root_link_lin_vel_b[:, :2]

    # Project velocity onto waypoint direction (in robot frame)
    vel_towards_waypoint = lin_vel_b[:, 0] * torch.cos(self.delta_yaw) + lin_vel_b[
      :, 1
    ] * torch.sin(self.delta_yaw)

    self.metrics["velocity_error"] = torch.abs(
      vel_towards_waypoint - self.target_velocity
    )
    self._actual_velocity = vel_towards_waypoint

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize goal, waypoints, planned path, and status information."""
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    robot_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    goal_pos = self.goal_pos_w[batch].cpu().numpy()
    waypoint_pos = self.current_waypoint_w[batch].cpu().numpy()
    ring_radius = float(self.current_ring_radius[batch].item())
    reached = bool(self.goal_reached[batch].item())
    current_wp_idx = int(self._current_waypoint_idx[batch].item())
    alt_used = bool(self.alternative_goal_used[batch].item())

    # Skip if robot appears uninitialized
    if np.linalg.norm(robot_pos) < 1e-6:
      return

    z_offset = self.cfg.viz.z_offset

    # Print status information (disabled for training)
    # status = "REACHED!" if reached else f"Goal: {distance:.1f}m | WP: {dist_to_wp:.1f}m"
    # alt_str = " [ALT]" if alt_used else ""
    # print(
    #   f"\r[PathPlan] V:{target_vel:+.1f}/{actual_vel:+.1f} m/s | "
    #   f"Path:{path_len:.1f}m | WP:{current_wp_idx + 1}/{num_wp} | "
    #   f"{status}{alt_str} | "
    #   f"Yaw:{np.degrees(float(self.delta_yaw[batch].item())):+.0f}°",
    #   end="",
    # )

    # Goal color: green if reached, cyan otherwise, yellow if alternative
    if reached:
      goal_color = (0.2, 0.9, 0.2, 0.9)
    elif alt_used:
      goal_color = (0.9, 0.9, 0.2, 0.9)  # Yellow for alternative
    else:
      goal_color = (0.2, 0.8, 0.8, 0.8)

    # Draw final goal point as a sphere
    goal_vis_pos = goal_pos.copy()
    goal_vis_pos[2] += z_offset
    visualizer.add_sphere(
      center=goal_vis_pos,
      radius=0.25,
      color=goal_color,
    )

    # Draw current waypoint as smaller sphere (magenta)
    if not np.allclose(waypoint_pos[:2], goal_pos[:2], atol=0.1):
      wp_vis_pos = waypoint_pos.copy()
      wp_vis_pos[2] += z_offset
      visualizer.add_sphere(
        center=wp_vis_pos,
        radius=0.15,
        color=(0.8, 0.2, 0.8, 0.9),  # Magenta
      )

    # Draw planned path
    waypoints = self._planned_waypoints[batch]
    if len(waypoints) > 0:
      # Draw line from robot to first waypoint
      robot_vis_pos = robot_pos.copy()
      robot_vis_pos[2] += z_offset

      prev_pos = robot_vis_pos
      for i, (wx, wy) in enumerate(waypoints):
        wp_pos = np.array([wx, wy, robot_pos[2] + z_offset])

        # Color: green for completed, magenta for current, cyan for future
        if i < current_wp_idx:
          color = (0.2, 0.8, 0.2, 0.6)  # Green - completed
          line_width = 0.02
        elif i == current_wp_idx:
          color = (1.0, 0.2, 1.0, 0.9)  # Bright magenta - current
          line_width = 0.03
        else:
          color = (0.2, 0.8, 0.8, 0.6)  # Cyan - future
          line_width = 0.02

        visualizer.add_arrow(
          prev_pos,
          wp_pos,
          color=color,
          width=line_width,
        )

        # Draw small sphere at each waypoint (except the goal which is already drawn)
        if i < len(waypoints) - 1 or not np.allclose(
          wp_pos[:2], goal_pos[:2], atol=0.5
        ):
          wp_sphere_color = (color[0], color[1], color[2], 0.7)
          visualizer.add_sphere(
            center=wp_pos,
            radius=0.12,
            color=wp_sphere_color,
          )

        prev_pos = wp_pos

    # Draw robot heading direction
    robot_yaw = float(self.robot.data.heading_w[batch].item())
    heading_end = robot_pos.copy()
    heading_end[0] += 0.8 * np.cos(robot_yaw)
    heading_end[1] += 0.8 * np.sin(robot_yaw)
    heading_end[2] += z_offset
    robot_vis_pos = robot_pos.copy()
    robot_vis_pos[2] += z_offset
    visualizer.add_arrow(
      robot_vis_pos,
      heading_end,
      color=(1.0, 0.5, 0.0, 0.8),  # Orange - robot heading
      width=0.02,
    )

    # Draw reach threshold circle around goal
    num_segments = 32
    theta = np.linspace(0, 2 * np.pi, num_segments + 1)
    for i in range(num_segments):
      p1 = goal_pos.copy()
      p2 = goal_pos.copy()
      p1[0] += self.cfg.reach_threshold * np.cos(theta[i])
      p1[1] += self.cfg.reach_threshold * np.sin(theta[i])
      p1[2] += z_offset
      p2[0] += self.cfg.reach_threshold * np.cos(theta[i + 1])
      p2[1] += self.cfg.reach_threshold * np.sin(theta[i + 1])
      p2[2] += z_offset
      visualizer.add_arrow(
        p1,
        p2,
        color=(0.8, 0.8, 0.2, 0.5),  # Yellow - reach zone
        width=0.01,
      )

    # Draw shrinking ring around robot (if not at initial radius)
    if ring_radius < self.cfg.initial_ring_radius * 0.95:
      ring_color = (0.8, 0.4, 0.8, 0.4)  # Purple - shrinking ring
      for i in range(num_segments):
        p1 = robot_pos.copy()
        p2 = robot_pos.copy()
        p1[0] += ring_radius * np.cos(theta[i])
        p1[1] += ring_radius * np.sin(theta[i])
        p1[2] += z_offset - 0.1
        p2[0] += ring_radius * np.cos(theta[i + 1])
        p2[1] += ring_radius * np.sin(theta[i + 1])
        p2[2] += z_offset - 0.1
        visualizer.add_arrow(
          p1,
          p2,
          color=ring_color,
          width=0.008,
        )


@dataclass(kw_only=True)
class ReachableGoalCommandCfg(CommandTermCfg):
  """Configuration for reachable goal tracking command with path planning.

  Unlike CircularGoalTrackingCommand, this creates a FIXED goal in world frame
  that the robot can actually reach. The target distance is computed as:
    target_distance = resample_interval * target_velocity

  This enables proper RA (Reach-Avoid) training where:
  - Robot starts at some distance from goal
  - Distance decreases as robot walks towards goal
  - Goal is reached when distance < reach_threshold

  Path planning feature (NEW):
  - 2D occupancy grid built from terrain wall geometry
  - A* path planning to find optimal route avoiding obstacles
  - If goal unreachable, find alternative on robot's right side
  - Intermediate waypoints guide robot around obstacles
  - Resample duration computed from planned path length
  """

  entity_name: str
  """Name of the robot entity in the scene."""

  # Goal angle range relative to robot heading (radians)
  goal_angle_range: float = math.pi / 4
  """Maximum angle offset from robot heading for goal placement.
  Goal direction = robot_heading + random(-goal_angle_range, +goal_angle_range)
  Default ±45° means goal is generally in front of robot.
  """

  # Target velocity settings
  target_velocity: float = 0.8
  """Default target velocity [m/s]."""

  target_velocity_range: tuple[float, float] = (0.5, 1.3)
  """Range for sampling target velocity [m/s]."""

  # Distance settings
  min_target_distance: float = 2.0
  """Minimum target distance to goal [m]. Clamps computed distance."""

  initial_ring_radius: float = 3.0
  """Initial radius of the visualization ring around robot [m]."""

  shrink_threshold: float = 3.0
  """Distance at which the ring starts shrinking [m].
  When robot gets closer than this, ring radius = distance to goal.
  """

  reach_threshold: float = 0.5
  """Distance at which goal is considered reached [m]."""

  reach_resample_delay: float = 0.6
  """Delay in seconds before resampling after goal is reached [s].

  When the robot reaches the goal (distance < reach_threshold), the goal
  is NOT immediately resampled.  Instead, the robot stays near the current
  goal for this duration, allowing the RA buffer to accumulate multiple
  ls ≤ 0 ("reached") samples before the goal moves.  Set to 0 for instant
  resample (legacy behaviour).
  """

  # === Path Planning Settings ===
  @dataclass
  class PathPlanningCfg:
    """Configuration for path planning in maze environments."""

    enabled: bool = True
    """Whether to enable path planning. If False, uses direct line to goal."""

    grid_resolution: float = 0.5
    """Resolution of occupancy grid in meters per cell."""

    robot_radius: float = 0.4
    """Robot collision radius for path planning (inflate obstacles by this)."""

    waypoint_spacing: float = 2.0
    """Minimum distance between waypoints in simplified path [m]."""

    waypoint_reach_threshold: float = 1.0
    """Distance at which a waypoint is considered reached [m]."""

    max_resample_time: float = 30.0
    """Maximum time before forced resample [s]. Actual resample triggered early when goal reached."""

    dynamic_resample_time: bool = False
    """When True, set time_left = direct_distance / target_velocity + resample_time_margin
    instead of a fixed ``max_resample_time``.  This gives the robot just enough
    time to walk straight to the goal plus a small buffer, encouraging quick
    recovery after wall collisions.  Falls back to ``max_resample_time`` as an
    upper cap."""

    resample_time_margin: float = 2.0
    """Extra seconds added when ``dynamic_resample_time`` is enabled [s]."""

    alternative_search_range: float = math.pi / 2
    """Angle range to search for alternative goal (radians, negative = right)."""

    alternative_candidates: int = 12
    """Number of candidate angles to try when finding alternative goal."""

  path_planning: PathPlanningCfg = field(default_factory=PathPlanningCfg)
  """Path planning configuration."""

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
    """Forward lever arm used to convert yaw motion into side approach speed [m]."""

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
    """Visualization configuration."""

    z_offset: float = 0.3

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> ReachableGoalCommand:
    return ReachableGoalCommand(self, env)


# ==============================================================================
# Fixed Sequential Goal Command (for SingleWall environment)
# ==============================================================================


class FixedSequentialGoalCommand(CommandTerm):
  """Command term with fixed sequential goal positions.

  This command term provides a fixed sequence of goal positions:
  - Intermediate waypoints: Guide the robot around obstacles
  - Key goals: The actual target positions (e.g., behind the wall)

  The robot follows waypoints in order, but for local planner training we also
  compute the angle directly to the current key goal (ignoring waypoints).

  Command output: [delta_yaw, target_velocity]
  - delta_yaw: angle from robot heading to current WAYPOINT (used by policy)
  - target_velocity: desired forward speed

  Additional outputs for local planner training (not used by policy):
  - delta_yaw_to_key_goal: angle directly to the current key goal
  - key_goal_pos_w: world position of current key goal

  This is designed for simple, repeatable obstacle avoidance tests
  where goals are always in the same positions relative to spawn.
  """

  cfg: "FixedSequentialGoalCommandCfg"

  def __init__(self, cfg: "FixedSequentialGoalCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._env = env

    # Goal sequence: [num_goals, 2] relative positions from spawn
    # These are in LOCAL coordinates (relative to robot spawn position)
    self._goal_offsets = torch.tensor(
      cfg.goal_offsets, device=self.device, dtype=torch.float32
    )
    self.num_goals = len(cfg.goal_offsets)

    # Noise for each waypoint (optional)
    if cfg.goal_offset_noise is not None:
      self._goal_offset_noise = torch.tensor(
        cfg.goal_offset_noise, device=self.device, dtype=torch.float32
      )
    else:
      self._goal_offset_noise = None

    # Per-environment goal offsets (with noise applied at reset)
    # Shape: [num_envs, num_goals, 2]
    self._env_goal_offsets = (
      self._goal_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
    )

    # Key goal indices (the "real" targets for local planner training)
    # Default: last waypoint is the only key goal
    if cfg.key_goal_indices is not None:
      self._key_goal_indices = list(cfg.key_goal_indices)
    else:
      # Default: only the final waypoint is a key goal
      self._key_goal_indices = [self.num_goals - 1]
    self._key_goal_indices_t = torch.tensor(
      self._key_goal_indices, dtype=torch.long, device=self.device
    )
    self._is_key_goal_idx = torch.zeros(
      self.num_goals, dtype=torch.bool, device=self.device
    )
    self._is_key_goal_idx[self._key_goal_indices_t] = True
    self._all_env_ids = torch.arange(self.num_envs, device=self.device)

    # Intermediate reach threshold (smaller for waypoints, larger for key goals)
    self._intermediate_threshold = (
      cfg.intermediate_reach_threshold or cfg.reach_threshold
    )

    # Current goal index for each environment
    self.current_goal_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Current key goal index for each environment (which key goal we're heading to)
    self.current_key_goal_idx = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Spawn position for each environment (set at reset)
    self._spawn_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

    # Current goal position in world frame (the waypoint being tracked)
    self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Current KEY goal position in world frame (for local planner training)
    # This is the actual target, ignoring intermediate waypoints
    self.key_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

    # Command output: [delta_yaw, target_velocity]
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._base_target_velocity_cmd = torch.full(
      (self.num_envs,), float(cfg.target_velocity), device=self.device
    )
    self._target_velocity_cmd = torch.full(
      (self.num_envs,), float(cfg.target_velocity), device=self.device
    )
    self._command[:, 1] = self._target_velocity_cmd

    # Optional short near-zero phase at episode start.
    self._near_zero_time_left = torch.zeros(self.num_envs, device=self.device)

    # Tracking
    self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
    self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
    self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.sequence_complete = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # local planner training outputs: angle directly to key goal (ignoring waypoints)
    self.delta_yaw_to_key_goal = torch.zeros(self.num_envs, device=self.device)
    self.distance_to_key_goal = torch.zeros(self.num_envs, device=self.device)

    # Metrics
    self.metrics = {
      "heading_error": torch.zeros(self.num_envs, device=self.device),
      "distance_to_goal": torch.zeros(self.num_envs, device=self.device),
      "goals_reached": torch.zeros(self.num_envs, device=self.device),
    }

    # Expose per-environment target velocity for metrics/rewards compatibility.
    self.target_velocity = self._target_velocity_cmd

    # Wall AABBs for obstacle clearance (lazily extracted on first reset)
    # Each entry: (center_x, center_y, half_x, half_y) in world coords
    self._wall_aabbs: list[tuple[float, float, float, float]] | None = None

    # All-obstacle AABBs (walls + furniture + other solids) for clearance checks
    # when velocity_adaptive_offsets is True.
    self._all_obstacle_aabbs: list[tuple[float, float, float, float]] | None = None

    # Terrain cell size for filtering obstacles to the robot's own cell.
    # Extracted lazily from TerrainGeneratorCfg.size on first reset.
    self._terrain_cell_size: tuple[float, float] | None = None

    # Safety-envelope diagnostics (mirrors ReachableGoalCommand).
    self._safety_velocity_scale = torch.ones(self.num_envs, device=self.device)

    # Trajectory recording for debug visualization.
    # Each env has a list of (x, y, z) world-coordinate points.
    self._trajectory_points: list[list[tuple[float, float, float]]] = [
      [] for _ in range(self.num_envs)
    ]

  @property
  def command(self) -> torch.Tensor:
    """Command tensor: [delta_yaw, target_velocity]."""
    return self._command

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    """Reset command state for specified environments."""
    assert isinstance(env_ids, torch.Tensor)

    extras = {}
    for metric_name, metric_value in self.metrics.items():
      extras[metric_name] = torch.mean(metric_value[env_ids]).item()
      metric_value[env_ids] = 0.0

    # Reset goal index to 0 (first goal)
    self.current_goal_idx[env_ids] = 0
    self.current_key_goal_idx[env_ids] = 0  # Start with first key goal
    self.goal_reached[env_ids] = False
    self.sequence_complete[env_ids] = False
    self.command_counter[env_ids] = 0

    if self.cfg.debug_vis:
      # Trajectories are only recorded for the first few environments.
      for env_id in env_ids:
        env_id_int = int(env_id.item())
        if env_id_int < 4:
          self._trajectory_points[env_id_int] = []

    # Optional per-environment speed sampling for data collection diversity.
    if self.cfg.target_velocity_range is not None:
      low, high = self.cfg.target_velocity_range
      self._base_target_velocity_cmd[env_ids] = torch.empty(
        len(env_ids), device=self.device
      ).uniform_(float(low), float(high))
    else:
      self._base_target_velocity_cmd[env_ids] = float(self.cfg.target_velocity)

    self._target_velocity_cmd[env_ids] = self._base_target_velocity_cmd[env_ids]

    # Optional: inject short low-speed fragments for startup behavior learning.
    self._near_zero_time_left[env_ids] = 0.0
    if self.cfg.near_zero_probability > 0.0:
      near_zero_mask = (
        torch.rand(len(env_ids), device=self.device) < self.cfg.near_zero_probability
      )
      if near_zero_mask.any():
        selected_env_ids = env_ids[near_zero_mask]
        vz_low, vz_high = self.cfg.near_zero_velocity_range
        dz_low, dz_high = self.cfg.near_zero_duration_range
        self._target_velocity_cmd[selected_env_ids] = torch.empty(
          len(selected_env_ids), device=self.device
        ).uniform_(float(vz_low), float(vz_high))
        self._near_zero_time_left[selected_env_ids] = torch.empty(
          len(selected_env_ids), device=self.device
        ).uniform_(float(dz_low), float(dz_high))

    # Store spawn position from terrain origins
    # Note: robot.data.root_link_pos_w may not be updated yet at reset time
    # Use env_origins which is the terrain spawn origin (already set during terrain generation)
    env_origins = self._env.scene.env_origins[env_ids, :2]
    self._spawn_pos_w[env_ids] = env_origins

    # ── Velocity-adaptive waypoint generation ─────────────────────────────
    if getattr(self.cfg, "velocity_adaptive_offsets", False):
      # Lazily initialise AABBs (walls only, for _find_nearest_wall_relative)
      if self._wall_aabbs is None:
        self._wall_aabbs = self._extract_wall_aabbs()
      if self._all_obstacle_aabbs is None:
        self._all_obstacle_aabbs = self._extract_all_obstacle_aabbs()
      # Process each environment individually
      for env_id in env_ids:
        env_id_int = int(env_id.item())
        speed = float(self._base_target_velocity_cmd[env_id_int].item())
        wall_info = self._find_nearest_wall_relative(env_id_int)

        # Detect room boundaries for indoor scenes (v5) so the bypass
        # path stays within the room walls.
        sx = float(self._spawn_pos_w[env_id_int, 0].item())
        sy = float(self._spawn_pos_w[env_id_int, 1].item())
        cell_bounds = self._get_cell_bounds(env_id_int)
        cell_obs = self._filter_obstacles_to_cell(self._all_obstacle_aabbs, cell_bounds)
        room = self._find_room_bounds(cell_obs, sx, sy)
        # room_bounds_rel: (left_dist, right_dist, south_dist, north_dist)
        # All four values are positive distances from spawn to inner wall face.
        room_bounds_rel: tuple[float, float, float, float] | None = None
        if room is not None:
          rx_min, rx_max, ry_min, ry_max = room
          room_bounds_rel = (
            sx - rx_min,  # dist to west wall
            rx_max - sx,  # dist to east wall
            sy - ry_min,  # dist to south wall (behind robot)
            ry_max - sy,  # dist to north wall (ahead)
          )

        if getattr(self.cfg, "straight_approach", False):
          offsets = self._generate_straight_approach_offsets(
            speed, wall_info, room_bounds_rel
          )
        else:
          offsets = self._generate_speed_adapted_offsets(
            speed, wall_info, room_bounds_rel
          )
        # Pad or truncate to exactly num_goals entries
        while len(offsets) < self.num_goals:
          offsets.append(offsets[-1])
        offsets = offsets[: self.num_goals]
        self._env_goal_offsets[env_id_int] = torch.tensor(
          offsets, device=self.device, dtype=torch.float32
        )
      # Apply small noise ONLY to key-goal waypoints for diversity,
      # then clamp key goals inside room bounds so they stay reachable.
      if self._goal_offset_noise is not None:
        for env_id in env_ids:
          env_id_int = int(env_id.item())
          for wp_idx in self._key_goal_indices:
            if wp_idx < self.num_goals:
              noise_range = self._goal_offset_noise[wp_idx]
              noise = (torch.rand(2, device=self.device) * 2.0 - 1.0) * noise_range
              self._env_goal_offsets[env_id_int, wp_idx] += noise
              # Clamp noisy key goal inside room bounds (v5 indoor) if detected
              _sx2 = float(self._spawn_pos_w[env_id_int, 0].item())
              _sy2 = float(self._spawn_pos_w[env_id_int, 1].item())
              _cb2 = self._get_cell_bounds(env_id_int)
              _co2 = self._filter_obstacles_to_cell(
                self._all_obstacle_aabbs or [], _cb2
              )
              _room2 = self._find_room_bounds(_co2, _sx2, _sy2)
              if _room2 is not None:
                _rx_min2, _rx_max2, _ry_min2, _ry_max2 = _room2
                _clr = max(float(self.cfg.min_obstacle_clearance), 0.6)
                # Clamp Y offset so key goal stays below north wall
                _max_y_off = _ry_max2 - _sy2 - _clr
                if (
                  float(self._env_goal_offsets[env_id_int, wp_idx, 1].item())
                  > _max_y_off
                ):
                  self._env_goal_offsets[env_id_int, wp_idx, 1] = _max_y_off
                # Clamp X offset inside east/west walls
                _min_x_off = _rx_min2 - _sx2 + _clr
                _max_x_off = _rx_max2 - _sx2 - _clr
                _off_x = float(self._env_goal_offsets[env_id_int, wp_idx, 0].item())
                if _off_x < _min_x_off:
                  self._env_goal_offsets[env_id_int, wp_idx, 0] = _min_x_off
                elif _off_x > _max_x_off:
                  self._env_goal_offsets[env_id_int, wp_idx, 0] = _max_x_off
    else:
      # Apply random noise to goal offsets for these environments
      if self._goal_offset_noise is not None:
        # noise_range is [num_goals, 2] with max noise for each dimension.
        # Keep this on GPU; the previous per-env loop forced CUDA syncs.
        noise = (
          torch.rand(len(env_ids), self.num_goals, 2, device=self.device) * 2.0 - 1.0
        ) * self._goal_offset_noise.unsqueeze(0)
        self._env_goal_offsets[env_ids] = self._goal_offsets.unsqueeze(0) + noise
      else:
        self._env_goal_offsets[env_ids] = self._goal_offsets.unsqueeze(0)

    # Randomly mirror X-coordinates to produce left-side bypass paths
    mirror_prob = getattr(self.cfg, "mirror_x_probability", 0.0)
    if mirror_prob > 0.0:
      # Draw a Bernoulli sample per environment
      mirror_mask = torch.rand(len(env_ids), device=self.device) < mirror_prob
      self._env_goal_offsets[env_ids, :, 0] = torch.where(
        mirror_mask.unsqueeze(1),
        -self._env_goal_offsets[env_ids, :, 0],
        self._env_goal_offsets[env_ids, :, 0],
      )

    # Enforce minimum obstacle clearance (push waypoints away from walls)
    self._enforce_obstacle_clearance(env_ids)

    # Enforce segment clearance: ensure path segments between consecutive
    # waypoints do not pass through obstacles.
    self._enforce_segment_clearance(env_ids)

    # --- Debug: print waypoint / key-goal positions & wall distances ---
    # if self.cfg.min_obstacle_clearance > 0.0:
    #   # Lazily ensure wall AABBs are available
    #   if self._wall_aabbs is None:
    #     self._wall_aabbs = self._extract_wall_aabbs()
    #   # Only print for the first 2 envs to avoid log spam
    #   for _i, env_id in enumerate(env_ids[:2]):
    #     env_id_int = int(env_id.item())
    #     sx = float(self._spawn_pos_w[env_id_int, 0].item())
    #     sy = float(self._spawn_pos_w[env_id_int, 1].item())
    #     print(f"\n=== Env {env_id_int}  spawn=({sx:.2f}, {sy:.2f}) ===")
    #     for wp_idx in range(self.num_goals):
    #       off = self._env_goal_offsets[env_id_int, wp_idx]
    #       wx = sx + float(off[0].item())
    #       wy = sy + float(off[1].item())
    #       d = self._min_wall_distance(wx, wy)
    #       is_key = (
    #         self._key_goal_indices is not None and wp_idx in self._key_goal_indices
    #       )
    #       tag = " [KEY]" if is_key else ""
    #       print(f"  WP{wp_idx}{tag}: ({wx:.3f}, {wy:.3f})  wall_dist={d:.3f} m")
    # --- End debug ---

    # Set first goal position in world frame (waypoint)
    self._update_goal_position(env_ids)

    # Set first key goal position in world frame (for local planner training)
    self._update_key_goal_position(env_ids)

    # Reset timer (fixed range or dynamic by per-episode expected speed).
    if self.cfg.dynamic_timeout_by_velocity:
      # Approximate travel length: spawn -> wp0 -> ... -> wpN.
      segment_start = torch.zeros(len(env_ids), 2, device=self.device)
      path_length = torch.zeros(len(env_ids), device=self.device)
      for wp_idx in range(self.num_goals):
        segment_end = self._env_goal_offsets[env_ids, wp_idx, :]
        path_length += torch.linalg.norm(segment_end - segment_start, dim=1)
        segment_start = segment_end

      speed_ref = torch.clamp(
        self._base_target_velocity_cmd[env_ids],
        min=float(self.cfg.timeout_speed_floor),
      )
      timeout_s = path_length / speed_ref + float(self.cfg.timeout_margin_s)

      if self.cfg.timeout_min_s is not None:
        timeout_s = torch.clamp(timeout_s, min=float(self.cfg.timeout_min_s))
      if self.cfg.timeout_max_s is not None:
        timeout_s = torch.clamp(timeout_s, max=float(self.cfg.timeout_max_s))

      self.time_left[env_ids] = timeout_s
    else:
      self.time_left[env_ids] = self.time_left[env_ids].uniform_(
        *self.cfg.resampling_time_range
      )

    return extras

  def _extract_wall_aabbs(self) -> list[tuple[float, float, float, float]]:
    """Extract axis-aligned bounding boxes for all walls from the MuJoCo model.

    Returns list of (center_x, center_y, half_x, half_y) in world coords.
    Only wall-like BOX geoms are included (thin in one horizontal dimension).
    """
    walls: list[tuple[float, float, float, float]] = []
    try:
      mj_model = self._env.sim.mj_model

      # Find terrain body
      terrain_body_id = -1
      for i in range(mj_model.nbody):
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and "terrain" in body_name.lower():
          terrain_body_id = i
          break

      if terrain_body_id < 0:
        return walls

      for geom_id in range(mj_model.ngeom):
        if mj_model.geom_bodyid[geom_id] != terrain_body_id:
          continue
        if mj_model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
          continue

        geom_pos = mj_model.geom_pos[geom_id]
        geom_size = mj_model.geom_size[geom_id]  # half-sizes

        # Skip ground planes (very thin in Z)
        if geom_size[2] < 0.05:
          continue

        # Wall-like: one thin horizontal dimension, one long
        min_xy = min(geom_size[0], geom_size[1])
        max_xy = max(geom_size[0], geom_size[1])
        if min_xy < 0.5 and max_xy > 0.5:
          walls.append(
            (
              float(geom_pos[0]),
              float(geom_pos[1]),
              float(geom_size[0]),
              float(geom_size[1]),
            )
          )
    except Exception as e:
      _logger.warning(f"Failed to extract wall geometry for clearance check: {e}")
    return walls

  def _min_wall_distance(self, px: float, py: float) -> float:
    """Return the minimum distance from point (px, py) to any wall surface."""
    if not self._wall_aabbs:
      return float("inf")
    min_d = float("inf")
    for cx, cy, hx, hy in self._wall_aabbs:
      closest_x = max(cx - hx, min(px, cx + hx))
      closest_y = max(cy - hy, min(py, cy + hy))
      dx = px - closest_x
      dy = py - closest_y
      d = math.sqrt(dx * dx + dy * dy)
      # If inside the box, distance is 0
      if (cx - hx) <= px <= (cx + hx) and (cy - hy) <= py <= (cy + hy):
        d = 0.0
      if d < min_d:
        min_d = d
    return min_d

  def _get_terrain_cell_size(self) -> tuple[float, float]:
    """Return (width_x, width_y) of one terrain cell.

    Lazily cached from ``TerrainGeneratorCfg.size``.  Falls back to a
    generous default if the terrain config is unavailable.
    """
    if self._terrain_cell_size is not None:
      return self._terrain_cell_size
    try:
      terrain = self._env.scene.terrain
      if terrain is not None:
        gen_cfg = terrain.cfg.terrain_generator
        if gen_cfg is not None:
          self._terrain_cell_size = (float(gen_cfg.size[0]), float(gen_cfg.size[1]))
        else:
          self._terrain_cell_size = (60.0, 60.0)
      else:
        self._terrain_cell_size = (60.0, 60.0)
    except Exception:
      self._terrain_cell_size = (60.0, 60.0)
    return self._terrain_cell_size

  def _get_cell_bounds(self, env_id_int: int) -> tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) of the terrain cell for *env_id_int*.

    The cell bounds are estimated from the spawn position: the spawn is at
    ``(cell_center_x, cell_center_y - robot_distance_from_wall)`` so the
    cell center Y ≈ spawn_y + robot_distance_from_wall, and the cell
    extends ±half_size from its center.
    """
    cw, ch = self._get_terrain_cell_size()
    sx = float(self._spawn_pos_w[env_id_int, 0].item())
    sy = float(self._spawn_pos_w[env_id_int, 1].item())
    # Spawn sits robot_distance_from_wall below the cell centre Y.
    dist = float(self.cfg.robot_distance_from_wall)
    cell_cx = sx  # spawn X ≈ cell centre X (modulo random ± noise)
    cell_cy = sy + dist  # approximate cell centre Y
    return (
      cell_cx - cw / 2.0,
      cell_cx + cw / 2.0,
      cell_cy - ch / 2.0,
      cell_cy + ch / 2.0,
    )

  def _filter_obstacles_to_cell(
    self,
    obstacles: list[tuple[float, float, float, float]],
    cell_bounds: tuple[float, float, float, float],
  ) -> list[tuple[float, float, float, float]]:
    """Return only *obstacles* whose centre falls inside *cell_bounds*."""
    x_min, x_max, y_min, y_max = cell_bounds
    return [
      (cx, cy, hx, hy)
      for cx, cy, hx, hy in obstacles
      if x_min <= cx <= x_max and y_min <= cy <= y_max
    ]

  def _filter_path_relevant_obstacles(
    self,
    cell_obstacles: list[tuple[float, float, float, float]],
    env_id_int: int,
    clearance: float,
  ) -> list[tuple[float, float, float, float]]:
    """Keep only obstacles whose AABB overlaps the path corridor.

    The corridor is defined by the bounding box of all waypoint offsets
    (including key goals), expanded by *clearance* in every direction.
    Obstacles outside this region cannot affect the path and are dropped.
    """
    spawn = self._spawn_pos_w[env_id_int]
    sx = float(spawn[0].item())
    sy = float(spawn[1].item())

    # Compute bounding box of the full path (spawn + all waypoint offsets).
    path_x_min = 0.0
    path_x_max = 0.0
    path_y_min = 0.0
    path_y_max = 0.0
    for wp_idx in range(self.num_goals):
      dx = float(self._env_goal_offsets[env_id_int, wp_idx, 0].item())
      dy = float(self._env_goal_offsets[env_id_int, wp_idx, 1].item())
      path_x_min = min(path_x_min, dx)
      path_x_max = max(path_x_max, dx)
      path_y_min = min(path_y_min, dy)
      path_y_max = max(path_y_max, dy)

    # Expand to world coordinates with clearance margin.
    margin = clearance + 0.5
    corridor_x_min = sx + path_x_min - margin
    corridor_x_max = sx + path_x_max + margin
    corridor_y_min = sy + path_y_min - margin
    corridor_y_max = sy + path_y_max + margin

    return [
      (cx, cy, hx, hy)
      for cx, cy, hx, hy in cell_obstacles
      if (
        cx + hx >= corridor_x_min
        and cx - hx <= corridor_x_max
        and cy + hy >= corridor_y_min
        and cy - hy <= corridor_y_max
      )
    ]

  def _find_room_bounds(
    self,
    cell_obstacles: list[tuple[float, float, float, float]],
    sx: float,
    sy: float,
  ) -> tuple[float, float, float, float] | None:
    """Detect the rectangular indoor room boundary from cell-local obstacles.

    Scans for large rectangular walls that enclose the spawn point.
    Returns ``(room_x_min, room_x_max, room_y_min, room_y_max)``
    (inner faces, world coords) or *None* if no room detected.

    A "room wall" is a wall whose centre is near the cell edges (further
    from spawn than the centre obstacle wall) AND whose long axis is
    parallel to a cell boundary (i.e. either thin in X or thin in Y).
    """
    dist = float(self.cfg.robot_distance_from_wall)
    # Approximate cell centre
    cell_cx = sx
    cell_cy = sy + dist

    # Candidate walls for each side (N/S/E/W inner faces)
    best_north: float | None = None  # smallest Y from above
    best_south: float | None = None  # largest Y from below
    best_east: float | None = None  # smallest X from right
    best_west: float | None = None  # largest X from left

    for cx, cy, hx, hy in cell_obstacles:
      # Outer walls are "thin" in one dimension and "long" in the other
      min_h = min(hx, hy)
      max_h = max(hx, hy)
      if min_h > 0.5 or max_h < 1.0:
        continue  # not a wall-shaped obstacle

      if hy < hx:
        # Horizontal wall (extends along X, thin in Y) → N or S boundary
        inner_south_face = cy + hy
        inner_north_face = cy - hy
        if cy > cell_cy:
          # North wall — pick the one closest to centre from above
          if best_north is None or inner_north_face < best_north:
            best_north = inner_north_face
        else:
          # South wall — pick the one closest to centre from below
          if best_south is None or inner_south_face > best_south:
            best_south = inner_south_face
      else:
        # Vertical wall (extends along Y, thin in X) → E or W boundary
        inner_west_face = cx + hx
        inner_east_face = cx - hx
        if cx > cell_cx:
          # East wall
          if best_east is None or inner_east_face < best_east:
            best_east = inner_east_face
        else:
          # West wall
          if best_west is None or inner_west_face > best_west:
            best_west = inner_west_face

    # Only return bounds if at least E/W walls detected (critical for bypass_x)
    if best_east is not None and best_west is not None:
      # Use cell bounds as fallback for N/S if not detected
      cw, ch = self._get_terrain_cell_size()
      if best_north is None:
        best_north = cell_cy + ch / 2.0
      if best_south is None:
        best_south = cell_cy - ch / 2.0
      return (best_west, best_east, best_south, best_north)
    return None

  def _extract_all_obstacle_aabbs(self) -> list[tuple[float, float, float, float]]:
    """Extract AABBs for ALL non-ground collision geoms (walls + furniture + pillars).

    Unlike _extract_wall_aabbs() which only finds thin wall-like boxes, this
    method returns every solid obstacle so that _enforce_obstacle_clearance()
    can push waypoints away from furniture and other non-wall obstacles.

    Returns list of (center_x, center_y, half_x, half_y) in world coords.
    """
    obstacles: list[tuple[float, float, float, float]] = []
    try:
      import mujoco as _mujoco  # noqa: PLC0415

      mj_model = self._env.sim.mj_model

      # Find terrain body index
      terrain_body_id = -1
      for i in range(mj_model.nbody):
        body_name = _mujoco.mj_id2name(mj_model, _mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and "terrain" in body_name.lower():
          terrain_body_id = i
          break
      if terrain_body_id < 0:
        return obstacles

      for geom_id in range(mj_model.ngeom):
        if mj_model.geom_bodyid[geom_id] != terrain_body_id:
          continue

        gtype = mj_model.geom_type[geom_id]
        gpos = mj_model.geom_pos[geom_id]
        gsize = mj_model.geom_size[geom_id]

        if gtype == _mujoco.mjtGeom.mjGEOM_BOX:
          # Skip ground planes (very thin in Z, height < 0.05 m)
          if gsize[2] < 0.05:
            continue
          obstacles.append(
            (float(gpos[0]), float(gpos[1]), float(gsize[0]), float(gsize[1]))
          )
        elif gtype == _mujoco.mjtGeom.mjGEOM_CYLINDER:
          # radius is gsize[0]; treat as square AABB
          r = float(gsize[0])
          obstacles.append((float(gpos[0]), float(gpos[1]), r, r))
        elif gtype == _mujoco.mjtGeom.mjGEOM_CAPSULE:
          r = float(gsize[0])
          obstacles.append((float(gpos[0]), float(gpos[1]), r, r))
    except Exception as e:
      _logger.warning(f"Failed to extract obstacle geometry: {e}")
    return obstacles

  def _find_nearest_wall_relative(
    self, env_id_int: int
  ) -> tuple[float, float, float] | None:
    """Find the nearest forward-facing *centre obstacle* wall in this env's cell.

    Returns (wall_cx_rel, wall_y_rel, wall_half_width) in spawn-relative
    coordinates, or None if no wall is found (open-field cell).

    Filtering:
      1. Obstacle must lie within this env's terrain cell bounds.
      2. Must be in front of the robot (wall_y_rel > 0).
      3. Must be a horizontal wall (extends along X, thin along Y).
      4. Must be within a reasonable forward distance (< 3 × dist).
    """
    if self._wall_aabbs is None:
      self._wall_aabbs = self._extract_wall_aabbs()

    sx = float(self._spawn_pos_w[env_id_int, 0].item())
    sy = float(self._spawn_pos_w[env_id_int, 1].item())

    cell_bounds = self._get_cell_bounds(env_id_int)
    cell_walls = self._filter_obstacles_to_cell(self._wall_aabbs, cell_bounds)

    max_forward_dist = 3.0 * float(self.cfg.robot_distance_from_wall)

    best: tuple[float, float, float] | None = None
    best_y_rel = float("inf")

    for cx, cy, hx, hy in cell_walls:
      wall_y_rel = cy - sy
      if wall_y_rel <= 0.1 or wall_y_rel > max_forward_dist:
        continue

      # Only horizontal walls (extends along X, thin in Y)
      if hy > hx:
        continue

      wall_half_width = float(hx)

      if wall_y_rel < best_y_rel:
        best_y_rel = wall_y_rel
        best = (cx - sx, wall_y_rel, wall_half_width)

    return best

  def _generate_speed_adapted_offsets(
    self,
    speed: float,
    wall_info: tuple[float, float, float] | None,
    room_bounds_rel: tuple[float, float, float, float] | None = None,
  ) -> list[tuple[float, float]]:
    """Generate waypoint offsets (relative to spawn) adapted to the robot speed.

    Uses a smooth bell-curve lateral-displacement profile with a flat-top
    section around the wall.  This guarantees:
      - Y-monotonic path (always moving forward)
      - Smooth heading transitions (Hermite interpolation)
      - The flat section scales with turning radius R so heading changes
        at the rising→flat and flat→falling boundaries stay feasible

    Post-processing blends any remaining heading-constraint violations
    toward neighbor midpoints (always reduces heading change).

    For open-field cells (wall_info=None) a straight path is generated.

    Args:
      speed: Target velocity sampled from target_velocity_range [m/s].
      wall_info: (wall_cx_rel, wall_y_rel, wall_half_width) or None.
      room_bounds_rel: (left_dist, right_dist, south_dist, north_dist) —
        distances from spawn to each inner room-wall face (all positive).
        None if no room detected (outdoor / open-field).

    Returns:
      List of (dx, dy) offsets from spawn, length = self.num_goals.
    """
    omega_max = max(self.cfg.max_angular_velocity, 0.1)
    R = max(speed / omega_max, 0.5)
    clearance = max(float(self.cfg.min_obstacle_clearance), 0.6)

    n_goals = self.num_goals
    n_key = 2

    # ── Open-field (no wall) ───────────────────────────────────────────────
    if wall_info is None:
      D = float(self.cfg.robot_distance_from_wall)
      key1_y = D + max(5.0, R * 3.0)
      key2_y = key1_y + max(2.5, R * 1.5)
      # Clamp to north room wall for indoor open-field cells
      if room_bounds_rel is not None:
        max_y = room_bounds_rel[3] - clearance
        key2_y = min(key2_y, max_y)
        key1_y = min(key1_y, key2_y - 1.0)
      n_mid = n_goals - n_key
      if n_mid <= 0:
        return [(0.0, key1_y), (0.0, key2_y)]
      step = key1_y / float(n_mid + 1)
      offsets: list[tuple[float, float]] = [(0.0, step * (i + 1)) for i in range(n_mid)]
      offsets.append((0.0, key1_y))
      offsets.append((0.0, key2_y))
      return offsets

    # ── Bypass path around obstacle wall ──────────────────────────────────
    wall_cx_rel, wall_y, wall_hw = wall_info

    # bypass_x must clear both right and mirrored-left side (mirror_x flips)
    bypass_x = wall_hw + abs(wall_cx_rel) + clearance

    # Clamp bypass_x to stay within room boundaries if detected.
    # room_bounds_rel[0] = dist to left (west) wall,  [1] = dist to right (east) wall
    if room_bounds_rel is not None:
      max_lateral = min(room_bounds_rel[0], room_bounds_rel[1]) - clearance
      if max_lateral < clearance:
        max_lateral = clearance
      bypass_x = min(bypass_x, max_lateral)

    # Flat section around the wall — scales with R for feasible turns
    flat_half = max(1.5, R * 1.0)

    # Key goal positions (unclamped first, then constrained to room)
    key1_y = wall_y + max(5.0, R * 3.0)
    key2_y = key1_y + max(2.5, R * 1.5)

    # Clamp key goal Y to stay inside north room wall if detected.
    # room_bounds_rel[3] = dist from spawn to north inner wall face.
    if room_bounds_rel is not None:
      max_y = room_bounds_rel[3] - clearance
      if max_y < wall_y + 1.0:
        max_y = wall_y + 1.0  # always at least 1 m past wall
      key2_y = min(key2_y, max_y)
      # Keep key1 at least 1 m before key2, and past the wall
      key1_y = min(key1_y, key2_y - 1.0)
      if key1_y <= wall_y:
        key1_y = wall_y + 0.5

    # Phase boundaries
    rise_end_y = max(1.5, wall_y - flat_half)
    flat_end_y = wall_y + flat_half

    # ── Hermite smooth-step helper ─────────────────────────────────────────
    def _smooth(t: float) -> float:
      t = max(0.0, min(1.0, t))
      return 3.0 * t * t - 2.0 * t * t * t

    # ── X-profile: bell-curve with flat top ────────────────────────────────
    def _x_at(y: float) -> float:
      if y <= rise_end_y:
        return bypass_x * _smooth(y / rise_end_y)
      if y <= flat_end_y:
        return bypass_x
      t = (y - flat_end_y) / max(0.1, key1_y - flat_end_y)
      return bypass_x * (1.0 - _smooth(t))

    # ── Build intermediate waypoints (evenly spaced in Y) ──────────────────
    n_mid = n_goals - n_key
    y_step = key1_y / (n_mid + 1)
    mid_offsets: list[tuple[float, float]] = []
    for i in range(n_mid):
      y = y_step * (i + 1)
      mid_offsets.append((_x_at(y), y))

    result = mid_offsets + [(0.0, key1_y), (0.0, key2_y)]

    # ── Post-process: smooth heading-constraint violations ─────────────────
    # Blend violating waypoints toward their neighbors' midpoint.  This
    # always reduces the heading change (moves the path toward a straight
    # line through the point).  Key goals (last 2) are never moved.
    all_pts: list[tuple[float, float]] = [(0.0, 0.0)] + result
    n_fixed_tail = n_key  # don't move the last n_key entries
    for _pass in range(5):
      changed = False
      for i in range(1, len(all_pts) - n_fixed_tail):
        px, py = all_pts[i - 1]
        qx, qy = all_pts[i]
        rx_pt, ry_pt = all_pts[i + 1]
        d_out = math.sqrt((rx_pt - qx) ** 2 + (ry_pt - qy) ** 2)
        d_in = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
        if d_in < 1e-6 or d_out < 1e-6:
          continue
        h_in = math.atan2(qy - py, qx - px)
        h_out = math.atan2(ry_pt - qy, rx_pt - qx)
        dtheta = abs(math.atan2(math.sin(h_out - h_in), math.cos(h_out - h_in)))
        if dtheta * R > d_out * 0.9 + 1e-4:
          mid_x = (px + rx_pt) / 2.0
          mid_y = (py + ry_pt) / 2.0
          blend = 0.4
          all_pts[i] = (
            qx * (1.0 - blend) + mid_x * blend,
            qy * (1.0 - blend) + mid_y * blend,
          )
          changed = True
      if not changed:
        break

    result = list(all_pts[1:])

    # ── Enforce Y-monotonicity ─────────────────────────────────────────────
    for i in range(1, len(result)):
      if result[i][1] <= result[i - 1][1]:
        result[i] = (result[i][0], result[i - 1][1] + 0.3)

    return [(round(x, 3), round(y, 3)) for x, y in result]

  def _generate_straight_approach_offsets(
    self,
    speed: float,
    wall_info: tuple[float, float, float] | None,
    room_bounds_rel: tuple[float, float, float, float] | None = None,
  ) -> list[tuple[float, float]]:
    """Generate straight-approach waypoints for reactive avoidance training.

    Mimics real-world no-map avoidance: walk straight → detect wall →
    turn >90° while drifting away → walk along wall → detect wall end →
    turn toward goal.

    The turn is intentionally MORE than 90°: the robot walks forward while
    simultaneously moving away from the wall until it reaches d_safe lateral
    clearance.  This prevents the robot from getting stuck when brushing
    past the wall at only 90°.

    Core constraint:
      d_turn − R ≥ d_safe   where R = speed / omega_max
      Ensures the robot never gets closer than d_safe to the wall.

    Args:
      speed: Sampled target velocity [m/s].
      wall_info: (wall_cx_rel, wall_y_rel, wall_half_width) or None.
      room_bounds_rel: Optional indoor room bounds. Clamps lateral
        bypass offsets (drift_x, wall_edge_x, corner_x) to the room
        interior so waypoints do not pierce outer room walls.
    """
    omega_max = max(self.cfg.max_angular_velocity, 0.1)
    R = max(speed / omega_max, 0.2)  # turning radius [m]
    d_safe = max(float(self.cfg.min_obstacle_clearance), 0.6)
    n_goals = self.num_goals
    n_key = 2  # last 2 waypoints are key goals

    # Determine turn threshold
    if self.cfg.turn_threshold is not None:
      d_turn = float(self.cfg.turn_threshold)
    else:
      d_turn = R + d_safe

    # Clamp d_turn: must fit within raycast range and enforce safety
    d_turn = min(d_turn, 4.0)
    d_turn = max(d_turn, R + d_safe)

    # ── Open field (no wall) ─────────────────────────────────────────────
    if wall_info is None:
      D = float(self.cfg.robot_distance_from_wall)
      key1_y = D + max(3.0, R * 2.0)
      _kg_gap = getattr(self.cfg, "min_key_goal_gap", 2.5)
      _kg_mult = getattr(self.cfg, "key_goal_gap_multiplier", 1.5)
      key2_y = key1_y + max(_kg_gap, R * _kg_mult)
      if room_bounds_rel is not None:
        max_y = room_bounds_rel[3] - d_safe
        key2_y = min(key2_y, max_y)
        key1_y = min(key1_y, key2_y - 1.0)
      n_mid = n_goals - n_key
      if n_mid <= 0:
        return [(0.0, key1_y), (0.0, key2_y)]
      step = key1_y / float(n_mid + 1)
      offsets: list[tuple[float, float]] = [(0.0, step * (i + 1)) for i in range(n_mid)]
      offsets.append((0.0, key1_y))
      offsets.append((0.0, key2_y))
      return offsets

    # ── Wall bypass path ─────────────────────────────────────────────────
    # Every intermediate waypoint (except the turn-trigger point) must be
    # ≥ d_safe from the nearest wall surface.  At the wall corner an
    # explicit waypoint is placed facing the short edge at d_safe distance
    # to guide the robot cleanly around the end of the wall.
    wall_cx, wall_y, wall_hw = wall_info

    # Phase 1: straight approach — turn starts at distance d_turn from wall
    turn_y = wall_y - d_turn

    # Phase 2: turn + drift away from wall
    # After a ~90° arc with radius R the robot is at approximately
    # (R, turn_y + R).  If R < d_safe the robot continues turning
    # (>90°), walking diagonally forward-and-away until it reaches
    # d_safe Y-clearance from the wall surface.
    turn_end_x = R
    turn_end_y = min(turn_y + R, wall_y - d_safe)

    # After the turn, NEVER approach the wall.  The post-turn distance
    # (wall_y − turn_end_y = d_turn − R) is always ≥ d_safe (guaranteed
    # by the clamp d_turn ≥ R + d_safe).  Walking along the wall at
    # this distance prevents the "drift toward wall" behaviour that
    # causes the robot to get progressively closer during the
    # along-wall phase, especially at lower speeds where d_turn − R
    # is much larger than d_safe.
    drift_x = max(R, d_safe)
    drift_y = turn_end_y  # maintain post-turn distance from wall

    # Phase 3: along wall to edge (at constant distance from long face)
    wall_edge_x = wall_hw + abs(wall_cx)  # actual wall edge
    along_wall_y = drift_y

    # Distance the robot maintains from the wall's long face.
    # Always ≥ d_safe; can be much larger at low speeds.
    along_wall_clearance = wall_y - along_wall_y

    # Clamp lateral offsets to room bounds for indoor environments.
    # Prevents bypass waypoints from penetrating outer room walls.
    _max_lat: float | None = None
    if room_bounds_rel is not None:
      _max_lat = max(min(room_bounds_rel[0], room_bounds_rel[1]) - d_safe, d_safe)
      drift_x = min(drift_x, _max_lat)
      wall_edge_x = min(wall_edge_x, _max_lat)

    # Phase 4: corner — explicit waypoint facing the wall's short edge.
    # The diagonal path from the last along-wall waypoint to the corner
    # passes within d_safe/√2 of the wall corner when both offsets equal
    # d_safe, which is too close.  Scale the clearance by ≈√2 so the
    # diagonal path maintains at least d_safe clearance from the corner,
    # and never use less clearance than the along-wall distance.
    corner_clearance = max(d_safe * 1.4, along_wall_clearance)
    corner_x = wall_edge_x + corner_clearance
    if _max_lat is not None:
      corner_x = min(corner_x, _max_lat)
    corner_y = wall_y  # directly facing the short edge

    # Phase 5: past wall — clearance from back face matches along-wall
    past_wall_x = corner_x
    past_wall_y = wall_y + max(d_safe, along_wall_clearance * 0.8)

    # Phase 6: key goals (return to centreline, behind wall)
    _kg_gap = getattr(self.cfg, "min_key_goal_gap", 2.5)
    _kg_mult = getattr(self.cfg, "key_goal_gap_multiplier", 1.5)
    key1_y = wall_y + max(3.0, R * 2.0)
    key2_y = key1_y + max(_kg_gap, R * _kg_mult)
    if room_bounds_rel is not None:
      max_y = room_bounds_rel[3] - d_safe
      key2_y = min(key2_y, max_y)
      key1_y = min(key1_y, key2_y - 1.0)

    # Build raw intermediate waypoint sequence
    n_mid = n_goals - n_key
    raw_path: list[tuple[float, float]] = []

    # Phase 1: approach waypoints
    if turn_y > 0.5:
      raw_path.append((0.0, turn_y * 0.5))
      raw_path.append((0.0, turn_y))
    else:
      raw_path.append((0.0, max(0.3, turn_y)))

    # Phase 2: turn end (initial lateral displacement R)
    raw_path.append((turn_end_x, turn_end_y))

    # Phase 2b: lateral drift (if the robot needs to move further in X
    # to reach at least d_safe lateral displacement after the turn).
    needs_drift = drift_x - turn_end_x > 0.3
    if needs_drift:
      raw_path.append((drift_x, drift_y))

    # Phase 3: along wall to edge (all at y = wall_y − d_safe)
    last_x = drift_x if needs_drift else turn_end_x
    if wall_edge_x > last_x + 0.5:
      mid_x = (last_x + wall_edge_x) / 2.0
      raw_path.append((mid_x, along_wall_y))
      raw_path.append((wall_edge_x, along_wall_y))
    elif wall_edge_x > last_x:
      raw_path.append((wall_edge_x, along_wall_y))

    # Phase 4: corner (face short edge at d_safe)
    raw_path.append((corner_x, corner_y))

    # Phase 5: past wall (d_safe from back face)
    raw_path.append((past_wall_x, past_wall_y))

    # Trim or pad to match n_mid
    if len(raw_path) > n_mid:
      indices = [0]
      for i in range(1, n_mid - 1):
        idx = int(round(i * (len(raw_path) - 1) / (n_mid - 1)))
        indices.append(idx)
      indices.append(len(raw_path) - 1)
      raw_path = [raw_path[i] for i in indices]
    elif len(raw_path) < n_mid:
      while len(raw_path) < n_mid:
        last = raw_path[-1]
        raw_path.append((last[0] * 0.5, last[1] + (key1_y - last[1]) * 0.5))

    # Append key goals
    result: list[tuple[float, float]] = raw_path + [(0.0, key1_y), (0.0, key2_y)]

    # Enforce Y-monotonicity
    for i in range(1, len(result)):
      if result[i][1] <= result[i - 1][1]:
        result[i] = (result[i][0], result[i - 1][1] + 0.3)

    return [(round(x, 3), round(y, 3)) for x, y in result]

  def _enforce_obstacle_clearance(self, env_ids: torch.Tensor) -> None:
    """Push waypoints away from walls to maintain minimum clearance.

    For each waypoint in each env, computes the distance to every wall
    surface using closest-point-on-AABB. If distance < min_obstacle_clearance,
    the waypoint is pushed outward to exactly the clearance distance.

    Only **path-relevant** obstacles are considered: those whose Y-range
    overlaps with the path corridor (spawn → last key goal) and whose
    X-range is within lateral reach of the bypass path.  This prevents
    furniture and room walls far from the intended route from distorting
    the planned waypoints.
    """
    clearance = self.cfg.min_obstacle_clearance
    if clearance <= 0.0:
      return

    # Lazily extract AABBs (they don't change between resets).
    if getattr(self.cfg, "velocity_adaptive_offsets", False):
      if self._all_obstacle_aabbs is None:
        self._all_obstacle_aabbs = self._extract_all_obstacle_aabbs()
      all_obs = self._all_obstacle_aabbs
    else:
      if self._wall_aabbs is None:
        self._wall_aabbs = self._extract_wall_aabbs()
      all_obs = self._wall_aabbs

    if not all_obs:
      return

    for env_id in env_ids:
      env_id_int = int(env_id.item())
      spawn = self._spawn_pos_w[env_id_int]  # [2]
      spawn_x = float(spawn[0].item())
      spawn_y = float(spawn[1].item())

      # Filter obstacles to this env's terrain cell only.
      cell_bounds = self._get_cell_bounds(env_id_int)
      obstacle_aabbs = self._filter_obstacles_to_cell(all_obs, cell_bounds)
      if not obstacle_aabbs:
        continue

      # Further filter to path-relevant obstacles only.
      obstacle_aabbs = self._filter_path_relevant_obstacles(
        obstacle_aabbs, env_id_int, clearance
      )
      if not obstacle_aabbs:
        continue

      for wp_idx in range(self.num_goals):
        offset = self._env_goal_offsets[env_id_int, wp_idx]
        wp_x = spawn_x + float(offset[0].item())
        wp_y = spawn_y + float(offset[1].item())

        for cx, cy, hx, hy in obstacle_aabbs:
          # Obstacle AABB bounds
          x_min = cx - hx
          x_max = cx + hx
          y_min = cy - hy
          y_max = cy + hy

          # Closest point on AABB to the waypoint
          closest_x = max(x_min, min(wp_x, x_max))
          closest_y = max(y_min, min(wp_y, y_max))

          dx = wp_x - closest_x
          dy = wp_y - closest_y
          dist = math.sqrt(dx * dx + dy * dy)

          inside = x_min <= wp_x <= x_max and y_min <= wp_y <= y_max

          if inside:
            # Push out through nearest face
            d_left = wp_x - x_min
            d_right = x_max - wp_x
            d_bottom = wp_y - y_min
            d_top = y_max - wp_y
            d_min = min(d_left, d_right, d_bottom, d_top)
            if d_min == d_left:
              wp_x = x_min - clearance
            elif d_min == d_right:
              wp_x = x_max + clearance
            elif d_min == d_bottom:
              wp_y = y_min - clearance
            else:
              wp_y = y_max + clearance
          elif dist < clearance:
            # Push outward from closest surface point
            if dist < 1e-6:
              dx, dy, dist = 1.0, 0.0, 1.0
            scale = clearance / dist
            wp_x = closest_x + dx * scale
            wp_y = closest_y + dy * scale
          else:
            continue

          # Write back as offset from spawn
          self._env_goal_offsets[env_id_int, wp_idx, 0] = wp_x - spawn_x
          self._env_goal_offsets[env_id_int, wp_idx, 1] = wp_y - spawn_y

      # Ensure waypoints remain Y-monotonic after obstacle pushes.
      # Pushes can move waypoints backward; fix by enforcing strictly
      # increasing Y offsets with a small minimum gap.
      for wp_idx in range(1, self.num_goals):
        prev_y = float(self._env_goal_offsets[env_id_int, wp_idx - 1, 1].item())
        curr_y = float(self._env_goal_offsets[env_id_int, wp_idx, 1].item())
        if curr_y <= prev_y:
          self._env_goal_offsets[env_id_int, wp_idx, 1] = prev_y + 0.3

  @staticmethod
  def _segment_intersects_aabb(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
  ) -> bool:
    """Check if line segment (a→b) intersects axis-aligned bounding box.

    Uses the slab method (Liang-Barsky) for 2D segment-AABB intersection.
    """
    dx = bx - ax
    dy = by - ay
    t_near = 0.0
    t_far = 1.0
    for d, lo, hi, orig in [(dx, x_min, x_max, ax), (dy, y_min, y_max, ay)]:
      if abs(d) < 1e-10:
        if orig < lo or orig > hi:
          return False
      else:
        t1 = (lo - orig) / d
        t2 = (hi - orig) / d
        if t1 > t2:
          t1, t2 = t2, t1
        t_near = max(t_near, t1)
        t_far = min(t_far, t2)
        if t_near > t_far:
          return False
    return True

  def _enforce_segment_clearance(self, env_ids: torch.Tensor) -> None:
    """Ensure path segments between consecutive waypoints don't cross obstacles.

    After per-waypoint clearance, two waypoints can each be safely away from
    an obstacle yet the straight-line segment between them passes through it
    (e.g. furniture or room walls in indoor environments).

    This check uses a SMALL inflation margin (0.15 m) — just enough to catch
    segments that clip or penetrate the physical obstacle body.  It does NOT
    re-enforce the full min_obstacle_clearance; that is already handled by
    _enforce_obstacle_clearance on each individual waypoint.  Using the full
    clearance here would incorrectly flag approach/along-wall segments in
    straight-approach paths that are intentionally close to the wall.

    For each crossing segment, the downstream waypoint is pushed laterally
    to the same side of the obstacle as the upstream waypoint, at full
    clearance distance from the obstacle surface.
    """
    clearance = self.cfg.min_obstacle_clearance
    if clearance <= 0.0:
      return

    # Small margin for segment-body intersection detection.
    # Only catches segments that actually clip the obstacle surface.
    _SEGMENT_MARGIN = 0.15

    # Reuse cached obstacle AABBs (already populated by _enforce_obstacle_clearance).
    if getattr(self.cfg, "velocity_adaptive_offsets", False):
      all_obs = self._all_obstacle_aabbs
    else:
      all_obs = self._wall_aabbs
    if not all_obs:
      return

    for env_id in env_ids:
      env_id_int = int(env_id.item())
      spawn = self._spawn_pos_w[env_id_int]
      sx = float(spawn[0].item())
      sy = float(spawn[1].item())

      cell_bounds = self._get_cell_bounds(env_id_int)
      obstacle_aabbs = self._filter_obstacles_to_cell(all_obs, cell_bounds)
      if not obstacle_aabbs:
        continue

      # Only check path-relevant obstacles.
      obstacle_aabbs = self._filter_path_relevant_obstacles(
        obstacle_aabbs, env_id_int, clearance
      )
      if not obstacle_aabbs:
        continue

      # Multiple passes to allow cascading fixes.
      for _pass in range(3):
        changed = False
        for wp_idx in range(self.num_goals):
          # Segment: prev → current
          if wp_idx == 0:
            ax, ay = sx, sy
          else:
            off_prev = self._env_goal_offsets[env_id_int, wp_idx - 1]
            ax = sx + float(off_prev[0].item())
            ay = sy + float(off_prev[1].item())

          off_curr = self._env_goal_offsets[env_id_int, wp_idx]
          bx = sx + float(off_curr[0].item())
          by = sy + float(off_curr[1].item())

          for cx, cy, hx, hy in obstacle_aabbs:
            # Inflate AABB by a small margin only — detect actual
            # body penetration, not clearance violations.
            inf_xmin = cx - hx - _SEGMENT_MARGIN
            inf_xmax = cx + hx + _SEGMENT_MARGIN
            inf_ymin = cy - hy - _SEGMENT_MARGIN
            inf_ymax = cy + hy + _SEGMENT_MARGIN

            if not self._segment_intersects_aabb(
              ax, ay, bx, by, inf_xmin, inf_ymin, inf_xmax, inf_ymax
            ):
              continue

            # Segment crosses the obstacle body.  Push B to the
            # same side of the obstacle as A at full clearance distance
            # so the path routes cleanly around.
            rel_ax = ax - cx
            rel_ay = ay - cy

            # Choose push axis: route around the obstacle's shorter
            # extent (easier to bypass).  For a wide wall (hx > hy)
            # route in X; for a tall/narrow one route in Y.
            if hx >= hy:
              # Wide obstacle → push in X (go around the X-end).
              if rel_ax >= 0:
                new_bx = cx + hx + clearance
              else:
                new_bx = cx - hx - clearance
              self._env_goal_offsets[env_id_int, wp_idx, 0] = new_bx - sx
              bx = new_bx
            else:
              # Tall/narrow obstacle → push in Y.
              if rel_ay >= 0:
                new_by = cy + hy + clearance
              else:
                new_by = cy - hy - clearance
              self._env_goal_offsets[env_id_int, wp_idx, 1] = new_by - sy
              by = new_by
            changed = True

        if not changed:
          break

      # Re-enforce Y-monotonicity after segment fixes.
      for wp_idx in range(1, self.num_goals):
        prev_y = float(self._env_goal_offsets[env_id_int, wp_idx - 1, 1].item())
        curr_y = float(self._env_goal_offsets[env_id_int, wp_idx, 1].item())
        if curr_y <= prev_y:
          self._env_goal_offsets[env_id_int, wp_idx, 1] = prev_y + 0.3

  def _update_goal_position(self, env_ids: torch.Tensor) -> None:
    """Update goal position for specified environments based on current goal index."""
    if env_ids.numel() == 0:
      return

    goal_idx = self.current_goal_idx[env_ids]
    valid = goal_idx < self.num_goals
    valid_env_ids = env_ids[valid]
    if valid_env_ids.numel() == 0:
      return

    valid_goal_idx = torch.clamp(goal_idx[valid], 0, self.num_goals - 1)
    offset = self._env_goal_offsets[valid_env_ids, valid_goal_idx]
    self.goal_pos_w[valid_env_ids, :2] = self._spawn_pos_w[valid_env_ids] + offset
    self.goal_pos_w[valid_env_ids, 2] = 0.0

  def _update_key_goal_position(self, env_ids: torch.Tensor) -> None:
    """Update key goal position for local planner training.

    The key goal is the actual target (not intermediate waypoints).
    This is used by the local planner to learn the "direct path" angle for comparison.
    """
    if env_ids.numel() == 0:
      return

    key_idx = self.current_key_goal_idx[env_ids]
    valid = key_idx < len(self._key_goal_indices)
    valid_env_ids = env_ids[valid]
    if valid_env_ids.numel() == 0:
      return

    valid_key_idx = torch.clamp(key_idx[valid], 0, len(self._key_goal_indices) - 1)
    waypoint_idx = self._key_goal_indices_t[valid_key_idx]
    offset = self._env_goal_offsets[valid_env_ids, waypoint_idx]
    self.key_goal_pos_w[valid_env_ids, :2] = self._spawn_pos_w[valid_env_ids] + offset
    self.key_goal_pos_w[valid_env_ids, 2] = 0.0

  def _get_raycast_clearance(
    self, sensor_name: str, env_ids: torch.Tensor
  ) -> torch.Tensor:
    """Return the minimum horizontal clearance for one raycast sensor."""
    sensor = self._env.scene[sensor_name]
    distances = sensor.data.distances[env_ids]
    return min_raycast_clearance(
      distances,
      skip_ground_cols=self.cfg.safety_velocity_envelope.skip_ground_cols,
    )

  def _apply_safety_velocity_envelope(
    self, delta_yaw: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip target velocity using direction-aware time-to-stop limits.

    Mirrors ReachableGoalCommand._apply_safety_velocity_envelope but
    operates over all environments at once.
    """
    safety_cfg = self.cfg.safety_velocity_envelope
    if not safety_cfg.enabled:
      self._safety_velocity_scale[:] = 1.0
      return delta_yaw, self._target_velocity_cmd

    front_clearance = self._get_raycast_clearance(
      safety_cfg.forward_sensor_name, self._all_env_ids
    )
    left_clearance = self._get_raycast_clearance(
      safety_cfg.left_sensor_name, self._all_env_ids
    )
    right_clearance = self._get_raycast_clearance(
      safety_cfg.right_sensor_name, self._all_env_ids
    )

    safe_speed_limit = compute_safe_speed_limit(
      front_clearance,
      safety_margin_m=safety_cfg.safety_margin_m,
      reaction_time_s=safety_cfg.reaction_time_s,
      braking_time_s=safety_cfg.braking_time_s,
    )

    clipped_target_velocity = torch.minimum(
      self._base_target_velocity_cmd, safe_speed_limit
    )

    # Preserve minimum turn velocity when turning in a blocked-front situation.
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

    # Snap to zero only when not in turning mode.
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
    base_speed = torch.clamp(self._base_target_velocity_cmd, min=1e-6)
    self._safety_velocity_scale = clipped_target_velocity / base_speed

    return delta_yaw, clipped_target_velocity

  def _advance_reached_waypoints(self, robot_pos: torch.Tensor) -> None:
    """Advance past the farthest reached waypoint without moving backward."""
    current_idx = self.current_goal_idx.clone()
    valid_current = current_idx < self.num_goals
    if not valid_current.any():
      return

    waypoint_xy = self._spawn_pos_w.unsqueeze(1) + self._env_goal_offsets
    dist_to_waypoints = torch.linalg.norm(waypoint_xy - robot_pos.unsqueeze(1), dim=2)

    waypoint_thresholds = torch.where(
      self._is_key_goal_idx,
      torch.full(
        (self.num_goals,),
        float(self.cfg.reach_threshold),
        device=self.device,
        dtype=dist_to_waypoints.dtype,
      ),
      torch.full(
        (self.num_goals,),
        float(self._intermediate_threshold),
        device=self.device,
        dtype=dist_to_waypoints.dtype,
      ),
    )
    waypoint_indices = torch.arange(self.num_goals, device=self.device)
    future_waypoints = waypoint_indices.unsqueeze(0) >= current_idx.unsqueeze(1)
    reached_future = (
      valid_current.unsqueeze(1)
      & future_waypoints
      & (dist_to_waypoints < waypoint_thresholds.unsqueeze(0))
    )
    reached_any = reached_future.any(dim=1)
    if not reached_any.any():
      return

    last_reached_idx = (
      torch.where(
        reached_future,
        waypoint_indices.unsqueeze(0),
        torch.full_like(waypoint_indices.unsqueeze(0), -1),
      )
      .max(dim=1)
      .values
    )
    new_goal_idx = torch.where(reached_any, last_reached_idx + 1, current_idx)
    completed_counts = torch.clamp(new_goal_idx - current_idx, min=0)

    self.current_goal_idx.copy_(new_goal_idx)
    self.metrics["goals_reached"] += completed_counts.to(
      self.metrics["goals_reached"].dtype
    )

    key_goal_complete_counts = (
      self._key_goal_indices_t.unsqueeze(0) < new_goal_idx.unsqueeze(1)
    ).sum(dim=1)
    self.current_key_goal_idx.copy_(
      torch.maximum(self.current_key_goal_idx, key_goal_complete_counts)
    )

    completed_mask = reached_any & (new_goal_idx >= self.num_goals)
    self.sequence_complete |= completed_mask
    self.goal_reached |= completed_mask

    advanced_env_ids = reached_any.nonzero().flatten()
    active_advanced_env_ids = (reached_any & ~completed_mask).nonzero().flatten()
    self._update_goal_position(active_advanced_env_ids)
    self._update_key_goal_position(advanced_env_ids)

    if completed_mask.any():
      completed_env_ids = completed_mask.nonzero().flatten()
      final_goal_idx = self.num_goals - 1
      final_goal_xy = waypoint_xy[completed_env_ids, final_goal_idx]
      self.goal_pos_w[completed_env_ids, :2] = final_goal_xy
      self.goal_pos_w[completed_env_ids, 2] = 0.0
      self.key_goal_pos_w[completed_env_ids, :2] = final_goal_xy
      self.key_goal_pos_w[completed_env_ids, 2] = 0.0

  def compute(self, dt: float) -> None:
    """Compute command based on current state."""
    if self.cfg.debug_vis:
      # Only the first few environments are visualized, so avoid recording
      # Python trajectory tuples for every parallel training environment.
      robot_pos = self.robot.data.root_link_pos_w
      for env_idx in range(min(self.num_envs, 4)):
        pos = robot_pos[env_idx]
        self._trajectory_points[env_idx].append(
          (float(pos[0].item()), float(pos[1].item()), float(pos[2].item()))
        )

    self._update_near_zero_phase(dt)
    self._update_metrics()

    # Check for timeout
    self.time_left -= dt
    # Mark sequence complete on timeout (episode should reset).
    self.sequence_complete |= self.time_left <= 0.0

    self._update_command()

  def _update_command(self) -> None:
    """Update command based on current robot and goal state."""
    # Get robot position and heading
    robot_pos = self.robot.data.root_link_pos_w[:, :2]
    robot_quat = self.robot.data.root_link_quat_w

    # Compute robot yaw
    yaw = torch.atan2(
      2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
      1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
    )

    self._advance_reached_waypoints(robot_pos)

    # Direction to current waypoint (used by policy)
    goal_xy = self.goal_pos_w[:, :2]
    to_goal = goal_xy - robot_pos
    dist_to_goal = torch.norm(to_goal, dim=1)
    self.distance_to_goal = dist_to_goal

    # Heading to waypoint
    goal_angle = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    delta_yaw = wrap_to_pi(goal_angle - yaw)
    self.delta_yaw = delta_yaw

    # Direction to key goal (for local planner training - ignores waypoints)
    key_goal_xy = self.key_goal_pos_w[:, :2]
    to_key_goal = key_goal_xy - robot_pos
    dist_to_key_goal = torch.norm(to_key_goal, dim=1)
    self.distance_to_key_goal = dist_to_key_goal

    # Heading to key goal
    key_goal_angle = torch.atan2(to_key_goal[:, 1], to_key_goal[:, 0])
    delta_yaw_to_key = wrap_to_pi(key_goal_angle - yaw)
    self.delta_yaw_to_key_goal = delta_yaw_to_key

    # Update command (uses waypoint, not key goal)
    delta_yaw, target_velocity = self._apply_safety_velocity_envelope(delta_yaw)
    self._command[:, 0] = delta_yaw
    self._command[:, 1] = target_velocity

    # Update metrics
    self.metrics["heading_error"] = torch.abs(delta_yaw)
    self.metrics["distance_to_goal"] = dist_to_goal

  def _update_metrics(self) -> None:
    """Update tracking metrics."""
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Resample command (not used for fixed goals)."""
    pass

  def _update_near_zero_phase(self, dt: float) -> None:
    """Restore base speed after short near-zero startup fragment ends."""
    if self.cfg.near_zero_probability <= 0.0:
      return

    active = self._near_zero_time_left > 0.0
    next_time_left = torch.clamp(self._near_zero_time_left - dt, min=0.0)
    expired = active & (next_time_left <= 0.0)
    self._target_velocity_cmd.copy_(
      torch.where(expired, self._base_target_velocity_cmd, self._target_velocity_cmd)
    )
    self._near_zero_time_left.copy_(next_time_left)

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize all waypoints, connecting path, and steering information."""

    import numpy as np

    # Draw full path for first few environments
    for env_idx in range(min(self.num_envs, 4)):
      robot_pos = self.robot.data.root_link_pos_w[env_idx].cpu().numpy()
      current_idx = int(self.current_goal_idx[env_idx].item())
      spawn = self._spawn_pos_w[env_idx].cpu().numpy()

      z_base = 0.3

      # Determine which waypoint indices are key goals.
      key_set = set(
        int(self._key_goal_indices[k]) for k in range(len(self._key_goal_indices))
      )

      # Draw all waypoints and connecting lines.
      prev_pos = np.array([robot_pos[0], robot_pos[1], z_base])
      for wp_i in range(self.num_goals):
        off = self._env_goal_offsets[env_idx, wp_i].cpu().numpy()
        wp_world = np.array(
          [
            float(spawn[0]) + float(off[0]),
            float(spawn[1]) + float(off[1]),
            z_base,
          ]
        )

        # Color coding: green=completed, magenta=current, cyan=future,
        # red-orange=key goal (future), bright green=key goal (completed).
        is_key = wp_i in key_set
        if wp_i < current_idx:
          color = (0.2, 0.8, 0.2, 0.5)  # green, completed
          sphere_r = 0.12
          line_w = 0.015
        elif wp_i == current_idx:
          color = (1.0, 0.2, 1.0, 0.9)  # bright magenta, current
          sphere_r = 0.25
          line_w = 0.03
        elif is_key:
          color = (0.0, 1.0, 1.0, 0.8)  # cyan, key goal
          sphere_r = 0.35
          line_w = 0.02
        else:
          color = (0.8, 0.6, 0.2, 0.7)  # orange, future intermediate
          sphere_r = 0.15
          line_w = 0.02

        # Connecting line from previous point to this waypoint.
        visualizer.add_arrow(
          start=prev_pos,
          end=wp_world,
          color=color,
          width=line_w,
        )

        # Sphere at waypoint.
        wp_sphere = wp_world.copy()
        wp_sphere[2] = z_base + 0.2 if is_key else z_base + 0.1
        visualizer.add_sphere(
          center=wp_sphere,
          radius=sphere_r,
          color=color,
          label=f"WP{wp_i}" if wp_i == current_idx else "",
        )

        prev_pos = wp_world

      # Draw arrow from robot to current waypoint (green overlay).
      goal_pos = self.goal_pos_w[env_idx].cpu().numpy()
      visualizer.add_arrow(
        start=np.array([robot_pos[0], robot_pos[1], z_base + 0.1]),
        end=np.array([goal_pos[0], goal_pos[1], z_base + 0.1]),
        color=(0.2, 0.8, 0.2, 0.8),
        width=0.02,
        label="to_waypoint",
      )

      # Draw arrow from robot to key goal (cyan).
      key_goal_pos = self.key_goal_pos_w[env_idx].cpu().numpy()
      visualizer.add_arrow(
        start=np.array([robot_pos[0], robot_pos[1], z_base + 0.2]),
        end=np.array([key_goal_pos[0], key_goal_pos[1], z_base + 0.2]),
        color=(0.0, 0.8, 1.0, 0.6),
        width=0.015,
        label="to_key_goal",
      )

      # Draw recorded trajectory (yellow line following the robot).
      traj = self._trajectory_points[env_idx]
      if len(traj) >= 2:
        z_traj = 0.05
        # Subsample to at most 200 segments to limit draw overhead.
        step = max(1, (len(traj) - 1) // 200)
        sampled = traj[::step]
        if sampled[-1] is not traj[-1]:
          sampled.append(traj[-1])
        for i in range(len(sampled) - 1):
          p0 = sampled[i]
          p1 = sampled[i + 1]
          visualizer.add_arrow(
            start=np.array([p0[0], p0[1], z_traj]),
            end=np.array([p1[0], p1[1], z_traj]),
            color=(1.0, 1.0, 0.0, 0.9),  # bright yellow
            width=0.012,
          )

  def get_overlay_info(self) -> tuple[str, str]:
    """Get overlay information for display in viewer.

    Returns:
      Tuple of (labels, values) strings for HUD display.
    """
    import math

    env_idx = 0  # Show info for first environment

    delta_yaw_deg = math.degrees(self.delta_yaw[env_idx].item())
    delta_key_deg = math.degrees(self.delta_yaw_to_key_goal[env_idx].item())
    correction = self.delta_yaw[env_idx] - self.delta_yaw_to_key_goal[env_idx]
    correction_deg = math.degrees(correction.item())

    current_wp = int(self.current_goal_idx[env_idx].item())
    current_key = int(self.current_key_goal_idx[env_idx].item())
    dist_wp = self.distance_to_goal[env_idx].item()
    dist_key = self.distance_to_key_goal[env_idx].item()

    labels = (
      "Goal Tracking\n"
      "-------------\n"
      "Current WP\n"
      "Key Goal\n"
      "Dist to WP\n"
      "Dist to Key\n"
      "dYaw (WP)\n"
      "dYaw (Key)\n"
      "Correction"
    )
    values = (
      "\n"
      "\n"
      f"{current_wp}/{self.num_goals}\n"
      f"{current_key}/{len(self._key_goal_indices)}\n"
      f"{dist_wp:.2f} m\n"
      f"{dist_key:.2f} m\n"
      f"{delta_yaw_deg:+.1f} deg\n"
      f"{delta_key_deg:+.1f} deg\n"
      f"{correction_deg:+.2f} deg ({correction.item():+.4f} rad)"
    )
    return labels, values


@dataclass(kw_only=True)
class FixedSequentialGoalCommandCfg(CommandTermCfg):
  """Configuration for fixed sequential goal command.

  This command provides a fixed sequence of goal positions that the robot
  must reach in order. Designed for simple, repeatable obstacle avoidance tests.

  For local planner training, you can specify which waypoints are "key goals" (the actual
  targets). The local planner network learns from the difference between:
  - delta_yaw: angle to current waypoint (what the policy uses)
  - delta_yaw_to_key_goal: angle directly to key goal (ignoring waypoints)
  """

  entity_name: str
  """Name of the robot entity in the scene."""

  goal_offsets: list[tuple[float, float]]
  """List of goal positions as (x, y) offsets from spawn point.

  Example for SingleWall environment with waypoints around the wall:
    goal_offsets = [
      (0.0, 2.0),   # WP0: in front of wall
      (4.5, 3.5),   # WP1: near wall's right edge
      (5.5, 5.0),   # WP2: past wall
      (0.0, 10.0),  # WP3: final goal behind wall
    ]
  """

  goal_offset_noise: list[tuple[float, float]] | None = None
  """Per-waypoint random noise range for (x, y) offsets.

  If provided, must have same length as goal_offsets.
  Each tuple (noise_x, noise_y) specifies the max uniform noise for that waypoint.
  Noise is sampled uniformly from [-noise, +noise] for each dimension.

  Example:
    goal_offset_noise = [
      (0.0, 0.0),   # WP0: no noise (fixed start)
      (0.5, 0.3),   # WP1: x ± 0.5m, y ± 0.3m
      (0.3, 0.2),   # WP2: x ± 0.3m, y ± 0.2m
      (0.0, 0.0),   # WP3: no noise (fixed final goal)
    ]
  """

  key_goal_indices: list[int] | None = None
  """Indices of waypoints that are "key goals" for local planner training.

  These are the actual targets (not intermediate waypoints). local planner learns
  to predict the steering correction by comparing:
  - delta_yaw: angle to current waypoint
  - delta_yaw_to_key_goal: angle to current key goal

  Example: key_goal_indices=[3] means WP3 is the only real goal.
  Example: key_goal_indices=[2, 5] means WP2 is Goal 1, WP5 is Goal 2.

  If None, defaults to [num_goals - 1] (only final waypoint is key goal).
  """

  target_velocity: float = 0.8
  """Target forward velocity [m/s]."""

  target_velocity_range: tuple[float, float] | None = None
  """Optional target velocity sampling range [m/s].

  If set, each environment samples target velocity uniformly on reset and
  outputs that value in command[1]. If None, uses fixed ``target_velocity``.
  """

  near_zero_probability: float = 0.0
  """Probability of injecting a short near-zero-speed fragment per episode."""

  near_zero_velocity_range: tuple[float, float] = (0.0, 0.1)
  """Speed range [m/s] for short near-zero fragments."""

  near_zero_duration_range: tuple[float, float] = (0.5, 1.0)
  """Duration range [s] for short near-zero fragments."""

  dynamic_timeout_by_velocity: bool = False
  """If True, set episode timeout from path length and sampled speed."""

  timeout_margin_s: float = 4.0
  """Extra timeout seconds added on top of path_length / expected_speed."""

  timeout_speed_floor: float = 0.2
  """Lower bound for expected speed when computing dynamic timeout."""

  timeout_min_s: float | None = None
  """Optional minimum dynamic timeout [s]."""

  timeout_max_s: float | None = None
  """Optional maximum dynamic timeout [s]."""

  reach_threshold: float = 1.0
  """Distance at which a goal is considered reached [m]."""

  intermediate_reach_threshold: float | None = None
  """Smaller threshold for intermediate waypoints [m].

  If provided, this threshold is used for all non-key-goal waypoints,
  while reach_threshold is used for key goals. This allows tighter
  tracking of intermediate waypoints.

  If None, reach_threshold is used for all waypoints.
  """

  mirror_x_probability: float = 0.0
  """Probability of mirroring all waypoint X-coordinates (flipping left/right).

  At each episode reset, with this probability all goal_offsets have their
  X-component negated, producing a left-side bypass path that is the mirror
  image of the original right-side path.

  This doubles the effective training diversity for obstacle avoidance:
  - Original waypoints: robot bypasses obstacles from the right
  - Mirrored waypoints:  robot bypasses obstacles from the left

  Set to 0.5 for equal left/right training.  Default 0.0 preserves legacy
  behaviour (right-side bypass only).
  """

  min_obstacle_clearance: float = 0.0
  """Minimum distance from each waypoint to any obstacle surface [m].

  If > 0, waypoints that are closer than this distance to any wall are
  pushed outward after noise and mirroring are applied. This ensures
  the robot has enough space to maneuver near obstacles.

  Default 0.0 disables the constraint (backward compatible).
  """

  # ── Velocity-adaptive waypoint generation ─────────────────────────────────

  velocity_adaptive_offsets: bool = False
  """If True, dynamically generate waypoint offsets at each episode reset
  based on the sampled target velocity.

  The algorithm computes a minimum turning radius R = v / max_angular_velocity
  and generates a bypass path whose curvature never exceeds 1/R, so the robot
  can always complete every direction change at the commanded speed.

  When True, ``goal_offsets`` values are used only to determine the number of
  waypoints (num_goals = len(goal_offsets)) and as a fallback if wall geometry
  cannot be extracted.  ``goal_offset_noise`` is still applied to key-goal
  waypoints for diversity.
  """

  max_angular_velocity: float = 1.0
  """Maximum robot angular velocity [rad/s] used to compute the minimum turning
  radius R = v / max_angular_velocity.

  Should match the ``max_ang_vel`` parameter in ``track_ang_vel_cmd`` reward
  (default 1.0 rad/s).
  """

  robot_distance_from_wall: float = 4.0
  """Distance from robot spawn to the nearest obstacle wall [m].

  Must match ``SingleWallTerrainCfg.robot_distance_from_wall`` (or
  ``IndoorRoomTerrainCfg.robot_distance_from_wall``) for the terrain in use.
  Used by ``_generate_speed_adapted_offsets()`` to compute key-goal positions
  correctly.
  """

  default_wall_half_width: float = 4.0
  """Fallback wall half-width [m] used by the adaptive offset generator when
  no wall geometry can be extracted from the terrain (e.g. open-field cells).
  Should match the dominant obstacle wall's half-width in the terrain mix.
  """

  max_heading_change: float = math.pi / 1.9
  """Maximum allowed heading change between consecutive waypoint segments [rad].

  Used as a reference threshold for heading-constraint post-processing.
  The adaptive path generator smooths any violations by blending waypoints
  toward their neighbors' midpoint.  Default π/1.9 (≈94.7°).
  """

  straight_approach: bool = False
  """If True, generate a straight-approach waypoint path instead of a bell-curve.

  The path follows real-world reactive avoidance behavior:
    1. Walk straight toward the wall (front raycast detects obstacle)
    2. At distance ``turn_threshold`` from the wall, execute 90° turn
    3. Walk along the wall at ``min_obstacle_clearance`` distance
    4. After passing the wall edge, turn back toward key goal

  Key constraint:  turn_threshold − v/max_angular_velocity ≥ min_obstacle_clearance
  This guarantees the robot never gets closer than min_obstacle_clearance to the
  wall surface during the turn.
  """

  turn_threshold: float | None = None
  """Forward distance at which the robot initiates a 90° turn [m].

  If None and ``straight_approach`` is True, computed automatically:
    turn_threshold = max_speed / max_angular_velocity + min_obstacle_clearance

  If set, the maximum safe speed is:
    v_max = max_angular_velocity × (turn_threshold − min_obstacle_clearance)

  Must satisfy:  turn_threshold ≤ raycast_max_range (4.0 m).
  """

  min_key_goal_gap: float = 2.5
  """Minimum distance [m] between the two key goals (WP7 → WP8) in
  straight-approach paths.  The actual gap is
  ``max(min_key_goal_gap, R * key_goal_gap_multiplier)``.

  Increase for high-speed tasks where the robot needs more room to
  turn from the wall-bypass path toward the final goal.
  Default 2.5 m (original value).
  """

  key_goal_gap_multiplier: float = 1.5
  """Multiplier on the turning radius R applied to the key-goal gap:
  ``gap = max(min_key_goal_gap, R * key_goal_gap_multiplier)``.

  Larger values give more runway between key goals relative to speed.
  Default 1.5 (original value).
  """

  safety_velocity_envelope: ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg = field(
    default_factory=ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg
  )
  """Time-to-stop safety-envelope configuration (same semantics as
  ReachableGoalCommandCfg).  Disabled by default."""

  def build(self, env: "ManagerBasedRlEnv") -> FixedSequentialGoalCommand:
    return FixedSequentialGoalCommand(self, env)
