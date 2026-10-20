"""Training loop for the steering local planner.

This module collects demonstrations from the global path planner and trains the
GRU-based local planner to reproduce those navigation decisions from local
observations only.

Training Approach:
1. Run pre-trained locomotion policy with path planner enabled
2. Record the path planner's heading commands and robot velocity
3. Train the local planner to predict these from local observations
4. The local planner learns to replace the global path planner

Goal-related quantities (bearing, distance) are computed analytically outside
the network using dead reckoning with the network's predicted linear velocity.

Network outputs (2D):
- delta_yaw: heading command for Actor (main task)
- pred_lin_vel: predicted current linear velocity (auxiliary, feeds DR)

Reference: Inspired by RA trainer from ABS training methodology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from mjlab.tasks.steering.local_planner_checkpoint_compat import (
  load_local_planner_checkpoint,
)
from mjlab.tasks.steering.local_planner_network import (
  LocalPlannerNetwork,
  LocalPlannerNetworkCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# G1 joint velocity indices used by local planner observations:
#   Lower body (12): hip pitch/roll/yaw × 2, knee × 2, ankle pitch/roll × 2
#   Waist (3): yaw, roll, pitch
#   Left upper arm (4): shoulder pitch/roll/yaw, elbow
#   Right upper arm (4): shoulder pitch/roll/yaw, elbow
#   Skipped: left wrist (19-21), right wrist (26-28)
G1_LOCAL_PLANNER_JOINT_VEL_INDICES: tuple[int, ...] = (*range(19), 22, 23, 24, 25)


@dataclass
class LocalPlannerTrainerCfg:
  """Configuration for the steering local planner trainer."""

  # Experience buffer
  buffer_size: int = 10000
  """Size of experience replay buffer."""

  batch_size: int = 256
  """Batch size for gradient updates."""

  sequence_length: int = 32
  """Sequence length for recurrent training."""

  # Training hyperparameters
  lr: float = 1e-4
  """Learning rate for Adam optimizer."""

  weight_decay: float = 1e-5
  """Weight decay for regularization."""

  grad_clip: float = 1.0
  """Gradient clipping threshold."""

  update_interval: int = 20
  """Number of steps between training updates."""

  warmup_steps: int = 1000
  """Number of steps before starting training."""

  # Loss weights
  correction_loss_weight: float = 1.0
  """Weight for delta_yaw (main task) prediction loss."""

  vel_pred_loss_weight: float = 0.3
  """Weight for linear velocity prediction loss (auxiliary)."""

  smoothness_loss_weight: float = 0.1
  """Weight for output smoothness loss (penalize rapid changes)."""

  proximity_loss_weight: float = 0.5
  """Weight for proximity-aware steering loss.
  When the robot is near obstacles (small raycast values), the yaw
  prediction error is amplified.  This incentivises higher steering
  accuracy in collision-risk zones without requiring explicit collision
  labels.  Set to 0 to disable."""

  proximity_loss_threshold: float = 0.5
  """Raycast distance threshold (normalised [0, 1]; 0 = contact, 1 = max
  range ≈ 4 m) below which the proximity loss activates.  Value 0.5
  corresponds to obstacles within ≈ 2 m.  Smaller values restrict the
  penalty to very close encounters.  Set to 0 to disable."""

  proximity_raycast_floor: float = 0.05
  """Minimum effective raycast value used in proximity weight computation
  (normalised [0, 1]).  Near-field occlusion DR injects values ≈ 0.005–
  0.020; clamping to this floor prevents those artefacts from being
  treated as contact (would inflate prox_w to ≈ 1.0).  Value 0.05
  corresponds to ≈ 0.2 m physical distance with a 4 m sensor."""

  hidden_state_loss_weight: float = 0.01
  """Weight for hidden state regularization."""

  # Curriculum learning – 2-phase schedule
  #
  # Phase 1 (vel):  Only vel_pred loss is active.  The hidden state learns to
  #   encode velocity from joint velocities (easy kinematic mapping).
  # Phase 2 (yaw):  correction (delta_yaw) + smoothness losses ramp in.  The
  #   network learns obstacle-avoiding steering output using goal bearing/
  #   distance computed analytically from the velocity it already knows.
  #
  # Set all durations to 0 to disable curriculum (all losses full from start).

  curriculum_phase1_updates: int = 1000
  """Phase 1 duration (gradient updates): vel_pred only."""

  curriculum_phase2_updates: int = 2000
  """Phase 2 ramp duration (gradient updates): linearly ramp correction
  (delta_yaw) & smoothness weights from 0 → configured value."""

  # Observation configuration
  raycast_dim: int = 64
  """Dimension of a single raycast observation (8x8 = 64)."""

  raycast_obs_names: tuple[str, ...] = ("raycast",)
  """Observation key names for raycast groups.
  Single camera: ("raycast",)
  Three cameras: ("raycast", "raycast_left", "raycast_right")"""

  heading_dim: int = 2
  """Dimension of robot heading from IMU quaternion: [cos_yaw, sin_yaw].
  Drift-free absolute orientation with no wrapping discontinuity."""

  joint_vel_dim: int = 23
  """Dimension of joint velocities from motor encoders (23 for G1).
  Includes legs (12) + waist (3) + both upper arms (4+4)."""

  joint_vel_indices: tuple[int, ...] | None = G1_LOCAL_PLANNER_JOINT_VEL_INDICES
  """Indices into robot.data.joint_vel to select. When None, falls back to
  simple truncation [:joint_vel_dim]. Use G1_LOCAL_PLANNER_JOINT_VEL_INDICES
  for G1."""

  goal_direction_dim: int = 4
  """Dimension of goal direction vector: [dx_scaled, dy_scaled, cos_bearing, sin_bearing]."""

  # --- Teacher Forcing: Ground-Truth Goal Inputs during Training ---
  #
  # The DR-computed bearing/distance drifts early in training because
  # pred_lin_vel is initially inaccurate (chicken-and-egg).  By replacing
  # the DR inputs with ground-truth values during training (plus noise to
  # simulate DR residual error), the network sees clean goal information
  # from step 1, breaking the dependency loop.
  #
  # At test time, DR is used as-is.  The noise makes the policy robust to
  # the gap between perfect GT and imperfect DR.

  use_gt_goal_inputs: bool = True
  """Use simulator ground-truth bearing/distance during training (with noise).
  Set False to use DR-computed values (original behaviour)."""

  gt_bearing_noise_std: float = 0.25
  """Std of Gaussian noise added to GT bearing during training (radians).
  ~0.25 rad ≈ 14 deg covers typical DR / localisation drift.
  Set to 0 for clean GT (may overfit; zero noise at test time)."""

  gt_distance_noise_frac: float = 0.10
  """Distance noise std as a fraction of true distance (e.g. 0.10 = 10 %).
  Absolute minimum noise floor is 0.05 m.
  Set to 0 for clean GT."""

  # --- DR sensor source ---
  use_measured_velocity: bool = False
  """Use measured forward velocity from the state estimator (leg kinematics
  + IMU fusion) instead of the network's pred_lin_vel for DR position.
  Default False = use network prediction (pred_lin_vel is sufficiently
  accurate after training)."""

  use_ahrs_heading: bool = True
  """Use AHRS (IMU orientation → yaw) for DR heading instead of
  integrating angular velocity.  Eliminates heading drift during turns.
  Default True because IMU quaternion is always available and avoids
  the largest source of DR drift (heading accumulation error)."""

  use_projected_yaw_rate: bool = False
  """Project body-frame angular velocity to world yaw rate using IMU
  quaternion before DR heading integration.  Fixes heading error when
  the robot pitches or rolls (body z ≠ world z).
  Default False = use body-frame ang_vel_z directly."""

  decision_dt: float = 0.02
  """Decision timestep in seconds (1 / policy frequency).
  Used for dead reckoning integration.  Default 0.02 = 50 Hz."""

  # Observation noise – sim-to-real robustness
  joint_vel_noise_std: float = 1.5
  """Std of uniform symmetric noise added to joint velocities during training
  (rad/s).  Real motor encoders report non-zero dq() even at standstill
  due to PD micro-corrections and sensor noise (~0.3-0.5 rad/s RMS per
  joint).  Injecting ±noise_std during training forces the network to
  be robust to this bias, preventing the ~0.1-0.2 m/s pred_lin_vel
  overestimate observed at standstill on hardware.
  Match the locomotion policy's joint_vel noise range: Unoise(-1.5, 1.5).
  Set to 0 to disable (clean sim values, legacy behaviour)."""

  # --- Odometry mode ---
  use_odometry: bool = False
  """Enable odometry mode.  When True:
  - Network output is 1D (delta_yaw only, no pred_lin_vel)
  - Dead reckoning is disabled; goal_direction computed from odom position
  - joint_vel is excluded from observations (unless include_joint_vel=True)
  - Curriculum learning is disabled (all losses active from start)
  - Loss weights are adjusted for obstacle-focused training"""

  odom_vel_dim: int = 2
  """Dimension of odometry body-frame velocity input (vx, vy).
  Only used when use_odometry=True."""

  include_joint_vel: bool = True
  """Include joint velocities in observations.
  Legacy (DR) mode: True (needed for velocity estimation).
  Odometry mode: False (velocity from odom; joint_vel is redundant noise)."""

  odom_bearing_noise_std: float = 0.06
  """Std of Gaussian noise added to odom-derived bearing (radians).
  Odometry has much lower drift than DR, so this is smaller than
  gt_bearing_noise_std (0.25).  ~0.06 rad ≈ 3.4°."""

  odom_distance_noise_frac: float = 0.03
  """Distance noise std as a fraction of true distance for odom mode.
  3% covers typical short-range odometry drift."""

  # Odometry-mode loss weight overrides
  odom_smoothness_loss_weight: float = 0.05
  """Smoothness loss weight in odom mode (lower than DR's 0.1 because
  odom goal_direction is stable; high smoothness delays avoidance)."""

  odom_proximity_loss_weight: float = 1.0
  """Proximity loss weight in odom mode (higher than DR's 0.5 because
  obstacle avoidance is the primary learning challenge)."""

  # Saving configuration
  save_dir: str = "logs/steering"
  """Directory to save local planner checkpoints."""

  save_interval: int = 1000
  """Interval for saving checkpoints (in updates)."""

  # Network configuration
  network_cfg: LocalPlannerNetworkCfg = field(default_factory=LocalPlannerNetworkCfg)
  """Configuration for the local planner network."""


class LocalPlannerObservationBuilder:
  """Builds local planner observations from environment state.

  Local planner observation format:
  - base_heading: [2] robot heading [cos_yaw, sin_yaw] from IMU quaternion
  - joint_vel: [23] joint velocities from motor encoders
    (legs 12 + waist 3 + left upper arm 4 + right upper arm 4)
  - raycast: [64] raycast distances (8x8 grid)
  - goal_direction: [4] body-frame goal vector
    [dx_scaled, dy_scaled, cos_bearing, sin_bearing]

  Total: 93 dimensions

  Dead Reckoning (DR):
    The builder maintains an analytic DR state to compute the goal
    direction vector for the network.  The DR integrates:

      displacement_x += pred_lin_vel * cos(heading) * dt
      displacement_y += pred_lin_vel * sin(heading) * dt
      heading += yaw_rate * dt

    From the DR state, the estimated goal position is:
      est_goal = initial_goal - displacement    (world frame)

    The goal direction vector is then computed in body frame:
      goal_body = R_world_to_body @ est_goal
      dx_scaled = goal_body_x / 10   (distance-carrying)
      dy_scaled = goal_body_y / 10
      cos_bearing = goal_body_x / |goal_body|  (pure direction)
      sin_bearing = goal_body_y / |goal_body|

    DR is reset on episode reset.
    Call ``update_dr()`` after each network forward pass to advance the state.
  """

  def __init__(
    self,
    heading_dim: int = 2,
    joint_vel_dim: int = 23,
    raycast_dim: int = 64,
    goal_direction_dim: int = 4,
    decision_dt: float = 0.02,
    raycast_obs_names: tuple[str, ...] | str = "raycast",
    command_name: str = "goal",
    joint_vel_indices: tuple[int, ...] | None = None,
    joint_vel_noise_std: float = 0.0,
    use_odometry: bool = False,
    odom_vel_dim: int = 0,
    include_joint_vel: bool = True,
  ):
    self.heading_dim = heading_dim
    self.joint_vel_dim = joint_vel_dim if include_joint_vel else 0
    self.joint_vel_indices = joint_vel_indices if include_joint_vel else None
    self.joint_vel_noise_std = joint_vel_noise_std if include_joint_vel else 0.0
    self.include_joint_vel = include_joint_vel
    # Pre-build index tensor (will be moved to device on first use)
    self._joint_vel_idx_tensor: torch.Tensor | None = None
    self.raycast_dim = raycast_dim
    self.goal_direction_dim = goal_direction_dim
    self.decision_dt = decision_dt
    # Odometry mode
    self.use_odometry = use_odometry
    self.odom_vel_dim = odom_vel_dim if use_odometry else 0
    # Normalise to tuple for uniform handling
    if isinstance(raycast_obs_names, str):
      self.raycast_obs_names: tuple[str, ...] = (raycast_obs_names,)
    else:
      self.raycast_obs_names = tuple(raycast_obs_names)
    self.num_raycast_groups = len(self.raycast_obs_names)
    # Keep single-name attribute for backward compat
    self.raycast_obs_name = self.raycast_obs_names[0]
    self.command_name = command_name

    # Store initial goal position for each environment (world-frame relative)
    self._initial_goal_pos: torch.Tensor | None = None  # [num_envs, 2]
    self._goal_assigned: torch.Tensor | None = None

    # Dead reckoning state
    self._dr_pos: torch.Tensor | None = None  # [num_envs, 2] displacement
    self._dr_heading: torch.Tensor | None = None  # [num_envs] heading

    # Key goal tracking: detect when key_goal_pos_w changes mid-episode
    self._prev_key_goal_idx: torch.Tensor | None = None  # [num_envs] int64

  def reset(self, num_envs: int, device: torch.device | str) -> None:
    """Reset the observation builder state.

    Args:
      num_envs: Number of parallel environments
      device: Device for tensors
    """
    dev = torch.device(device) if isinstance(device, str) else device
    if not self.use_odometry:
      # DR mode: allocate goal + DR state
      self._initial_goal_pos = torch.zeros(num_envs, 2, device=dev)
      self._goal_assigned = torch.zeros(num_envs, dtype=torch.bool, device=dev)
      self._dr_pos = torch.zeros(num_envs, 2, device=dev)
      self._dr_heading = torch.zeros(num_envs, device=dev)
      self._prev_key_goal_idx = torch.zeros(num_envs, dtype=torch.long, device=dev)
    else:
      # Odometry mode: no DR state needed
      self._initial_goal_pos = None
      self._goal_assigned = None
      self._dr_pos = None
      self._dr_heading = None
      self._prev_key_goal_idx = None

  def update_initial_goal(
    self,
    env: ManagerBasedRlEnv,
    new_goal_mask: torch.Tensor,
  ) -> None:
    """Update initial goal position when new goals are assigned.

    This should be called when the goal is resampled. The initial goal
    position is stored in the robot's body frame at the moment of assignment.

    Args:
      env: The RL environment
      new_goal_mask: Boolean mask indicating which environments got new goals
    """
    if self.use_odometry:
      return  # No DR state to update in odometry mode

    if self._initial_goal_pos is None:
      self.reset(env.num_envs, env.device)

    if not new_goal_mask.any():
      return

    assert self._initial_goal_pos is not None
    assert self._goal_assigned is not None

    # Get goal position from command manager
    # Prefer key_goal_pos_w (final goal) so DR tracks the same target
    # as at test time, rather than intermediate waypoints.
    try:
      term = env.command_manager.get_term(self.command_name)
      nav_target_w = None
      if term is not None and hasattr(term, "key_goal_pos_w"):
        nav_target_w = cast(torch.Tensor, term.key_goal_pos_w)
      elif term is not None and hasattr(term, "goal_pos_w"):
        nav_target_w = cast(torch.Tensor, term.goal_pos_w)

      if nav_target_w is not None:
        robot = env.scene["robot"]
        robot_pos_w = robot.data.root_link_pos_w[:, :2]
        # World-frame relative position (no rotation)
        rel_x = nav_target_w[:, 0] - robot_pos_w[:, 0]
        rel_y = nav_target_w[:, 1] - robot_pos_w[:, 1]

        # Update only for environments with new goals
        self._initial_goal_pos[new_goal_mask, 0] = rel_x[new_goal_mask]
        self._initial_goal_pos[new_goal_mask, 1] = rel_y[new_goal_mask]
        self._goal_assigned[new_goal_mask] = True

    except (KeyError, AttributeError):
      pass

  def check_key_goal_change(
    self,
    env: ManagerBasedRlEnv,
  ) -> None:
    """Detect key goal progression and re-initialise DR for affected envs.

    When `FixedSequentialGoalCommand` advances `current_key_goal_idx`
    (robot reached a key goal), the local planner observation builder must:
    1. Re-capture `_initial_goal_pos` relative to *current* robot position.
    2. Reset DR displacement to zero (goal is now relative to *here*).
    3. Reset DR heading to current heading.

    Call this **before** ``build()`` on every step.
    """
    if self.use_odometry:
      return  # No DR state in odometry mode

    try:
      term = env.command_manager.get_term(self.command_name)
      if term is None or not hasattr(term, "current_key_goal_idx"):
        return
      current_idx = cast(torch.Tensor, term.current_key_goal_idx)
    except (KeyError, AttributeError):
      return

    if self._prev_key_goal_idx is None:
      self._prev_key_goal_idx = current_idx.clone()
      return

    changed = current_idx != self._prev_key_goal_idx
    if not changed.any():
      return

    # Re-capture initial goal for changed envs (uses key_goal_pos_w)
    self.update_initial_goal(env, changed)

    # Reset DR for changed envs (goal offset is now relative to current pos)
    robot = env.scene["robot"]
    robot_quat = robot.data.root_link_quat_w
    robot_yaw = torch.atan2(
      2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
      1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
    )
    changed_ids = changed.nonzero(as_tuple=False).squeeze(-1)
    self.reset_dr(changed_ids, initial_heading=robot_yaw[changed_ids])

    self._prev_key_goal_idx = current_idx.clone()

  def build(
    self,
    env: ManagerBasedRlEnv,
    obs_dict: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Build local planner observations from environment state.

    Args:
      env: The RL environment
      obs_dict: Dictionary of observations from the environment

    Returns:
      Local planner observation tensor.
      Legacy (DR):  [heading(2), joint_vel(23), raycast(64×N), goal_dir(4)]
      Odometry:     [heading(2), odom_vel(2), raycast(64×N), goal_dir(4)]
    """
    if not self.use_odometry and self._initial_goal_pos is None:
      self.reset(env.num_envs, env.device)

    # Get robot heading from IMU quaternion (drift-free)
    robot = env.scene["robot"]
    robot_quat = robot.data.root_link_quat_w  # [B, 4] wxyz
    robot_yaw = torch.atan2(
      2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
      1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
    )  # [B]
    base_heading = torch.stack(
      [torch.cos(robot_yaw), torch.sin(robot_yaw)], dim=-1
    )  # [B, 2]
    cos_h = base_heading[:, 0]  # [B]
    sin_h = base_heading[:, 1]  # [B]

    # --- Odometry velocity (odom mode only) ---
    odom_vel: torch.Tensor | None = None
    if self.use_odometry and self.odom_vel_dim > 0:
      odom_vel = robot.data.root_link_lin_vel_b[:, : self.odom_vel_dim]  # [B, 2]

    # --- Joint velocities (legacy mode, or if explicitly included) ---
    joint_vel: torch.Tensor | None = None
    if self.include_joint_vel and self.joint_vel_dim > 0:
      raw_jv = robot.data.joint_vel  # [B, num_joints]
      if self.joint_vel_indices is not None:
        if (
          self._joint_vel_idx_tensor is None
          or self._joint_vel_idx_tensor.device != raw_jv.device
        ):
          self._joint_vel_idx_tensor = torch.tensor(
            self.joint_vel_indices, dtype=torch.long, device=raw_jv.device
          )
        joint_vel = raw_jv[:, self._joint_vel_idx_tensor]
      elif raw_jv.shape[-1] > self.joint_vel_dim:
        joint_vel = raw_jv[:, : self.joint_vel_dim]
      elif raw_jv.shape[-1] < self.joint_vel_dim:
        pad = torch.zeros(
          env.num_envs,
          self.joint_vel_dim - raw_jv.shape[-1],
          device=env.device,
        )
        joint_vel = torch.cat([raw_jv, pad], dim=-1)
      else:
        joint_vel = raw_jv

      # Sim-to-real noise
      if self.joint_vel_noise_std > 0.0:
        noise = torch.empty_like(joint_vel).uniform_(
          -self.joint_vel_noise_std, self.joint_vel_noise_std
        )
        joint_vel = joint_vel + noise

    # --- Raycast observations ---
    raycast_parts: list[torch.Tensor] = []
    for rc_name in self.raycast_obs_names:
      if rc_name in obs_dict:
        rc = obs_dict[rc_name]
      else:
        if not hasattr(self, "_raycast_warn_printed"):
          print(
            f"[WARN] '{rc_name}' not found in obs_dict. "
            f"Available keys: {list(obs_dict.keys())}. Using ones fallback!"
          )
          self._raycast_warn_printed = True
        rc = torch.ones(env.num_envs, self.raycast_dim, device=env.device)
      if rc.shape[-1] > self.raycast_dim:
        rc = rc[..., -self.raycast_dim :]
      raycast_parts.append(rc)
    raycast = torch.cat(raycast_parts, dim=-1)

    # --- Goal direction ---
    if self.use_odometry:
      goal_direction = self._compute_odom_goal_direction(env, cos_h, sin_h)
    else:
      goal_direction = self._compute_dr_goal_direction(cos_h, sin_h)

    # --- Assemble observation ---
    parts: list[torch.Tensor] = [base_heading]
    if odom_vel is not None:
      parts.append(odom_vel)
    if joint_vel is not None:
      parts.append(joint_vel)
    parts.append(raycast)
    parts.append(goal_direction)
    return torch.cat(parts, dim=-1)

  def _compute_dr_goal_direction(
    self, cos_h: torch.Tensor, sin_h: torch.Tensor
  ) -> torch.Tensor:
    """Compute 4D goal direction from dead-reckoning state."""
    assert self._initial_goal_pos is not None and self._dr_pos is not None
    est_goal = self._initial_goal_pos - self._dr_pos  # [B, 2]
    goal_body_x = est_goal[:, 0] * cos_h + est_goal[:, 1] * sin_h
    goal_body_y = -est_goal[:, 0] * sin_h + est_goal[:, 1] * cos_h
    distance = torch.sqrt(goal_body_x**2 + goal_body_y**2).clamp(min=0.1)
    dx_scaled = (goal_body_x / 10.0).unsqueeze(-1)
    dy_scaled = (goal_body_y / 10.0).unsqueeze(-1)
    cos_bearing = (goal_body_x / distance).unsqueeze(-1)
    sin_bearing = (goal_body_y / distance).unsqueeze(-1)
    return torch.cat([dx_scaled, dy_scaled, cos_bearing, sin_bearing], dim=-1)

  def _compute_odom_goal_direction(
    self,
    env: ManagerBasedRlEnv,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
  ) -> torch.Tensor:
    """Compute 4D goal direction directly from odometry (no DR).

    Uses the simulator's robot position and goal position to compute
    body-frame goal direction.  Equivalent to ground-truth computation.
    """
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_link_pos_w[:, :2]  # [B, 2]

    # Get goal position from command manager
    goal_pos_w = robot_pos_w  # fallback: zero direction
    try:
      term = env.command_manager.get_term(self.command_name)
      if term is not None:
        if hasattr(term, "key_goal_pos_w"):
          goal_pos_w = cast(torch.Tensor, term.key_goal_pos_w)[:, :2]
        elif hasattr(term, "goal_pos_w"):
          goal_pos_w = cast(torch.Tensor, term.goal_pos_w)[:, :2]
    except (KeyError, AttributeError):
      pass

    # World-frame relative goal
    rel_x = goal_pos_w[:, 0] - robot_pos_w[:, 0]  # [B]
    rel_y = goal_pos_w[:, 1] - robot_pos_w[:, 1]  # [B]

    # Transform to body frame
    goal_body_x = rel_x * cos_h + rel_y * sin_h  # forward
    goal_body_y = -rel_x * sin_h + rel_y * cos_h  # left

    distance = torch.sqrt(goal_body_x**2 + goal_body_y**2).clamp(min=0.1)
    dx_scaled = (goal_body_x / 10.0).unsqueeze(-1)
    dy_scaled = (goal_body_y / 10.0).unsqueeze(-1)
    cos_bearing = (goal_body_x / distance).unsqueeze(-1)
    sin_bearing = (goal_body_y / distance).unsqueeze(-1)
    return torch.cat([dx_scaled, dy_scaled, cos_bearing, sin_bearing], dim=-1)

  def get_initial_goal(self) -> torch.Tensor:
    """Get the current initial goal positions.

    Returns:
      Initial goal positions [num_envs, 2]
    """
    if self._initial_goal_pos is None:
      raise RuntimeError("Observation builder not initialized. Call reset() first.")
    return self._initial_goal_pos

  def compute_true_bearing_distance(
    self,
    env: ManagerBasedRlEnv,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute ground-truth goal bearing and distance from the simulator.

    Uses the actual robot pose and goal position, completely independent of
    the dead-reckoning state.  Intended for teacher-forcing during training.

    Returns:
      (bearing [B, 1] in radians wrapped to [-pi, pi],
       distance [B, 1] in metres) or (None, None) on failure.
    """
    try:
      term = env.command_manager.get_term(self.command_name)
      if term is None:
        return None, None

      # CRITICAL: use key_goal_pos_w (final goal, e.g. behind wall) instead
      # of goal_pos_w (current waypoint).  With goal_pos_w, the GT bearing
      # equals the delta_yaw target (both point to the same waypoint),
      # allowing the network to learn a trivial identity mapping
      # (output ≈ bearing) that completely ignores raycast → no avoidance.
      # Using key_goal_pos_w creates a gap between bearing (to final goal)
      # and target (to current waypoint around obstacle), forcing the
      # network to rely on raycast to predict the avoidance detour.
      if hasattr(term, "key_goal_pos_w"):
        goal_pos_w = cast(torch.Tensor, term.key_goal_pos_w)
      elif hasattr(term, "goal_pos_w"):
        goal_pos_w = cast(torch.Tensor, term.goal_pos_w)
      else:
        return None, None

      robot = env.scene["robot"]
      robot_pos_w = robot.data.root_link_pos_w[:, :2]  # [B, 2]
      robot_quat = robot.data.root_link_quat_w  # [B, 4] wxyz

      # Robot yaw from quaternion
      robot_yaw = torch.atan2(
        2.0
        * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
      )  # [B]

      # World-frame relative goal
      rel_x = goal_pos_w[:, 0] - robot_pos_w[:, 0]  # [B]
      rel_y = goal_pos_w[:, 1] - robot_pos_w[:, 1]  # [B]

      # Bearing: angle from robot forward to goal (body frame)
      angle_w = torch.atan2(rel_y, rel_x)
      bearing = angle_w - robot_yaw  # [B]
      bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))  # wrap

      # Distance
      distance = torch.sqrt(rel_x**2 + rel_y**2)  # [B]

      return bearing.unsqueeze(-1), distance.unsqueeze(-1)  # [B,1], [B,1]
    except (KeyError, AttributeError):
      return None, None

  def update_dr(
    self,
    pred_lin_vel: torch.Tensor,
    ang_vel_z: torch.Tensor,
    done: torch.Tensor | None = None,
    robot_yaw: torch.Tensor | None = None,
    measured_heading: torch.Tensor | None = None,
  ) -> None:
    """Update dead reckoning state.

    Call this **after** each network forward pass.

    Args:
      pred_lin_vel: [num_envs, 1] forward velocity (measured or predicted)
      ang_vel_z: [num_envs, 1] observed angular velocity (yaw rate).
        Used for heading integration ONLY when *measured_heading* is None.
      done: [num_envs] episode termination flags; DR is reset for done envs
      robot_yaw: [num_envs] world-frame yaw for done-env heading reset.
        Only used when *done* has True entries AND *measured_heading*
        is None (legacy integration mode).
      measured_heading: [num_envs] or [num_envs, 1] world-frame yaw from
        IMU / AHRS.  When provided, heading is set directly to this value
        instead of being integrated from *ang_vel_z*, completely
        eliminating heading drift.
    """
    if self._dr_pos is None or self._dr_heading is None:
      return  # Odometry mode or not initialized

    dt = self.decision_dt
    v = pred_lin_vel.detach().squeeze(-1)  # [num_envs]

    if measured_heading is not None:
      # Use measured heading directly – no integration drift.
      heading_now = measured_heading.detach().view(-1)  # [num_envs]
      # Position integration uses the MEASURED heading (most accurate)
      self._dr_pos[:, 0] = self._dr_pos[:, 0] + v * torch.cos(heading_now) * dt
      self._dr_pos[:, 1] = self._dr_pos[:, 1] + v * torch.sin(heading_now) * dt
      # Set heading directly (zero drift) – use copy_ to preserve tensor shape
      self._dr_heading.copy_(heading_now)
    else:
      # Legacy: integrate heading from angular velocity (drifts over time)
      w = ang_vel_z.detach().squeeze(-1)  # [num_envs]
      # Integrate position FIRST using the CURRENT heading (before update).
      self._dr_pos[:, 0] = self._dr_pos[:, 0] + v * torch.cos(self._dr_heading) * dt
      self._dr_pos[:, 1] = self._dr_pos[:, 1] + v * torch.sin(self._dr_heading) * dt
      # THEN integrate heading
      self._dr_heading = self._dr_heading + w * dt
      # Wrap heading to [-π, π] to prevent angle flip / numerical issues
      self._dr_heading = torch.atan2(
        torch.sin(self._dr_heading), torch.cos(self._dr_heading)
      )

    # Reset DR for terminated episodes
    if done is not None and done.any():
      self._dr_pos[done] = 0.0
      if measured_heading is not None:
        self._dr_heading[done] = measured_heading.detach().view(-1)[done]
      elif robot_yaw is not None:
        self._dr_heading[done] = robot_yaw[done]
      else:
        self._dr_heading[done] = 0.0
      # Reset key goal tracking
      if self._prev_key_goal_idx is not None:
        self._prev_key_goal_idx[done] = 0

  def reset_dr(
    self,
    env_ids: torch.Tensor | None = None,
    initial_heading: torch.Tensor | None = None,
  ) -> None:
    """Reset dead reckoning state for specific environments.

    Args:
      env_ids: Environment indices to reset.  If None, reset all.
      initial_heading: World-frame yaw angles to initialise DR heading.
        Shape ``[num_envs]`` when *env_ids* is ``None``, or
        ``[len(env_ids)]`` otherwise.  If ``None``, heading is zeroed
        (incorrect for world-frame DR – always pass the robot yaw).
    """
    if self._dr_pos is None or self._dr_heading is None:
      return
    if env_ids is None:
      self._dr_pos.zero_()
      if initial_heading is not None:
        self._dr_heading.copy_(initial_heading)
      else:
        self._dr_heading.zero_()
      # Reset key goal tracking
      if self._prev_key_goal_idx is not None:
        self._prev_key_goal_idx.zero_()
    else:
      self._dr_pos[env_ids] = 0.0
      if initial_heading is not None:
        self._dr_heading[env_ids] = initial_heading
      else:
        self._dr_heading[env_ids] = 0.0
      # Reset key goal tracking for these envs
      if self._prev_key_goal_idx is not None:
        self._prev_key_goal_idx[env_ids] = 0

  @staticmethod
  def project_yaw_rate(
    ang_vel_b: torch.Tensor,
    quat_w: torch.Tensor,
  ) -> torch.Tensor:
    """Project body-frame angular velocity to world-frame yaw rate.

    When the robot pitches or rolls, body z-axis is not aligned with
    world z.  Simply using ``ang_vel_b[:, 2]`` as yaw rate introduces
    heading error proportional to pitch/roll angle.

    This method computes the correct world-frame yaw rate:

      ω_world = R(q) @ ω_body
      yaw_rate = ω_world_z

    Args:
      ang_vel_b: [B, 3] body-frame angular velocity [ωx, ωy, ωz].
      quat_w: [B, 4] orientation quaternion in wxyz convention.

    Returns:
      [B, 1] world-frame yaw rate.
    """
    qw, qx, qy, qz = quat_w[:, 0], quat_w[:, 1], quat_w[:, 2], quat_w[:, 3]
    # Third row of rotation matrix R(q) applied to ω_body
    yaw_rate = (
      2.0 * (qx * qz - qw * qy) * ang_vel_b[:, 0]
      + 2.0 * (qy * qz + qw * qx) * ang_vel_b[:, 1]
      + (1.0 - 2.0 * (qx**2 + qy**2)) * ang_vel_b[:, 2]
    )
    return yaw_rate.unsqueeze(-1)  # [B, 1]


class SteeringExperienceBuffer:
  """Experience buffer for local planner training with importance sampling.

  Stores sequences of (observation, target_correction, hidden_state) for
  training the recurrent network.

  Key improvement: Uses raycast change rate for importance-weighted sampling.
  Samples with larger raycast changes (approaching or leaving obstacles) are
  more important and should be sampled more frequently.
  """

  def __init__(
    self,
    buffer_size: int,
    num_envs: int,
    obs_dim: int,
    hidden_dim: int,
    device: torch.device,
    raycast_dim: int = 64,
    raycast_offset: int = 25,  # heading(2) + joint_vel(23)
    target_dim: int = 2,
  ):
    self.buffer_size = buffer_size
    self.num_envs = num_envs
    self.obs_dim = obs_dim
    self.hidden_dim = hidden_dim
    self.device = device
    self.raycast_dim = raycast_dim
    self.raycast_offset = raycast_offset
    self.target_dim = target_dim

    # Circular buffer index
    self._write_idx = 0
    self._full = False

    # Observation buffer
    self.obs_buffer = torch.zeros(
      (buffer_size, num_envs, obs_dim),
      device=device,
      dtype=torch.float32,
    )

    # Target buffer: [delta_yaw] (odom) or [delta_yaw, lin_vel] (legacy)
    self.target_buffer = torch.zeros(
      (buffer_size, num_envs, target_dim),
      device=device,
      dtype=torch.float32,
    )

    # Hidden state buffer (for recurrent training)
    self.hidden_buffer = torch.zeros(
      (buffer_size, num_envs, hidden_dim),
      device=device,
      dtype=torch.float32,
    )

    # Done flag buffer (for sequence boundaries)
    self.done_buffer = torch.zeros(
      (buffer_size, num_envs),
      device=device,
      dtype=torch.bool,
    )

    # Importance weight buffer based on raycast change rate
    # Higher weight = more likely to be sampled
    self.importance_buffer = torch.ones(
      (buffer_size, num_envs),
      device=device,
      dtype=torch.float32,
    )

    # Previous raycast for computing change rate
    self._prev_raycast: torch.Tensor | None = None

    self.step_count = 0

  def add(
    self,
    obs: torch.Tensor,
    target: torch.Tensor,
    hidden_state: torch.Tensor,
    done: torch.Tensor,
  ) -> None:
    """Add experience to buffer with importance weight based on raycast change.

    Args:
      obs: Local planner observation [num_envs, obs_dim]
      target: Training targets [num_envs, 2]: [delta_yaw, lin_vel]
      hidden_state: Current hidden state [num_envs, hidden_dim]
      done: Episode termination flags [num_envs]
    """
    # Extract raycast from observation
    raycast_start = self.raycast_offset
    raycast_end = raycast_start + self.raycast_dim
    current_raycast = obs[:, raycast_start:raycast_end]

    # Compute importance weight based on raycast change AND proximity
    if self._prev_raycast is not None:
      # Compute L1 change rate of raycast
      raycast_change = (current_raycast - self._prev_raycast).abs().mean(dim=-1)
      # Also consider large corrections as important
      correction_magnitude = target[:, 0].abs()  # delta_yaw magnitude
      # CRITICAL: Also consider raycast absolute value (proximity to obstacles)
      # When raycast values are small (close to wall), importance should be HIGH
      # Use min raycast value - smaller = closer to obstacle = more important
      #
      # Due to camera euler (0, -π/2, 0), image columns map to robot
      # vertical: col 0 = 24° downward (always hits ground at ~1.8m),
      # col 7 = 24° upward.  Skip the first 3 downward-pointing columns
      # to get only horizontal/upward rays for clearance estimation.
      #
      # For multi-group raycast (n3): reshape each 64-dim group to 8x8,
      # compute per-group min, then take the overall min across groups.
      num_envs = obs.shape[0]
      num_groups = self.raycast_dim // 64 if self.raycast_dim > 64 else 1
      _grid = current_raycast.reshape(num_envs * num_groups, 8, 8)
      _horiz = _grid[:, :, 3:]  # skip 3 ground-hitting cols
      _min_per_group = _horiz.reshape(num_envs * num_groups, -1).min(dim=-1).values
      # Aggregate across groups: [num_envs * num_groups] -> [num_envs, num_groups] -> min
      min_raycast = (
        _min_per_group.reshape(num_envs, num_groups).min(dim=-1).values
      )  # [num_envs]
      # Proximity importance: high when close to obstacles (small raycast)
      # min_raycast is normalised [0, 1] (max_distance ≈ 4 m).
      # threshold=0.5 → activate below ~2 m physical distance.
      proximity_importance = torch.clamp(1.0 - min_raycast / 0.5, 0.0, 1.0)
      # Extra boost when very close (< 0.5 m physical ≈ 0.125 normalised)
      proximity_importance = proximity_importance + 5.0 * torch.clamp(
        1.0 - min_raycast / 0.125, 0.0, 1.0
      )

      # Goal-seeking importance: reward samples where correction is large
      # AND space is clear (open-space goal-seeking samples).
      # Without this, open-space samples (small raycast change, small
      # proximity) get very low importance and the network never learns
      # goal-seeking behaviour.
      open_space_mask = (min_raycast > 0.6).float()  # 1 when clearly open (> ~2.4 m)
      goal_seeking_importance = open_space_mask * (1.0 + 2.0 * correction_magnitude)

      # Combined importance: change rate + proximity + correction + goal-seeking
      importance = (
        1.0
        + 5.0 * raycast_change
        + 3.0 * correction_magnitude
        + 4.0 * proximity_importance
        + 3.0 * goal_seeking_importance
      )
    else:
      importance = torch.ones(self.num_envs, device=self.device)

    # Reset importance for done environments (episode boundaries are important)
    importance[done] = 2.0

    self.obs_buffer[self._write_idx] = obs
    self.target_buffer[self._write_idx] = target
    self.hidden_buffer[self._write_idx] = hidden_state
    self.done_buffer[self._write_idx] = done
    self.importance_buffer[self._write_idx] = importance

    # Update previous raycast
    self._prev_raycast = current_raycast.clone()
    # Reset prev_raycast for done envs so next step doesn't compute change
    if done.any():
      # We can't easily reset only some envs, so we'll handle this differently
      # by setting importance=1 after done in next step (already done above)
      pass

    self._write_idx = (self._write_idx + 1) % self.buffer_size
    if self._write_idx == 0:
      self._full = True
    self.step_count += 1

  def is_ready(self, min_samples: int) -> bool:
    """Check if buffer has enough samples for training."""
    return self.step_count >= min_samples

  def sample_sequences(
    self,
    batch_size: int,
    sequence_length: int,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample sequences for recurrent training with importance weighting.

    Sequences starting at timesteps with higher raycast change rates
    are more likely to be sampled.

    Args:
      batch_size: Number of sequences to sample
      sequence_length: Length of each sequence

    Returns:
      Tuple of (obs, target, initial_hidden, mask)
      - obs: [batch_size, sequence_length, obs_dim]
      - target: [batch_size, sequence_length, 2]
      - initial_hidden: [batch_size, hidden_dim]
      - mask: [batch_size, sequence_length] valid timestep mask
    """
    max_start = (
      self.buffer_size - sequence_length
      if self._full
      else self._write_idx - sequence_length
    )

    if max_start <= 0:
      raise ValueError("Not enough data in buffer for the requested sequence length")

    # Compute importance weights for sequence starting points
    # Use mean importance over sequence as weight
    weights_list = []
    for start_idx in range(max_start):
      indices = torch.arange(sequence_length, device=self.device) + start_idx
      # Mean importance over all envs and all timesteps in sequence
      seq_importance = self.importance_buffer[indices].mean()
      weights_list.append(seq_importance)

    weights = torch.stack(weights_list)
    # Normalize to probability distribution
    probs = weights / weights.sum()

    # Sample start indices with importance weighting
    start_indices = torch.multinomial(probs, batch_size, replacement=True)

    # Gather sequences
    obs_seqs = []
    target_seqs = []
    hidden_seqs = []
    masks = []

    for _i, start_idx in enumerate(start_indices):
      # Get sequence indices
      indices = torch.arange(sequence_length, device=self.device) + start_idx.item()

      # Sample env with importance weighting too
      env_importance = self.importance_buffer[indices].mean(dim=0)
      env_probs = env_importance / env_importance.sum()
      env_idx = torch.multinomial(env_probs, 1).item()

      obs_seqs.append(self.obs_buffer[indices, int(env_idx)])
      target_seqs.append(self.target_buffer[indices, int(env_idx)])
      hidden_seqs.append(self.hidden_buffer[int(start_idx.item()), int(env_idx)])

      # Create mask (valid until done)
      done_flags = self.done_buffer[indices, int(env_idx)]
      mask = (~done_flags).float()
      # Propagate mask: once done, all subsequent steps are invalid
      for j in range(1, sequence_length):
        mask[j] = mask[j] * mask[j - 1]
      masks.append(mask)

    obs = torch.stack(obs_seqs, dim=0)
    targets = torch.stack(target_seqs, dim=0)
    initial_hidden = torch.stack(hidden_seqs, dim=0)
    mask = torch.stack(masks, dim=0)

    return obs, targets, initial_hidden, mask

  def sample_batch(
    self,
    batch_size: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a simple batch with importance weighting.

    Args:
      batch_size: Number of samples

    Returns:
      Tuple of (obs, target)
    """
    max_idx = self.buffer_size if self._full else self._write_idx

    # Flatten importance weights
    flat_importance = self.importance_buffer[:max_idx].reshape(-1)
    probs = flat_importance / flat_importance.sum()

    # Sample with importance weighting
    flat_indices = torch.multinomial(probs, batch_size, replacement=True)

    # Convert flat indices to (time_idx, env_idx)
    time_indices = flat_indices // self.num_envs
    env_indices = flat_indices % self.num_envs

    obs = self.obs_buffer[time_indices, env_indices]
    targets = self.target_buffer[time_indices, env_indices]

    return obs, targets


class LocalPlannerTrainer:
  """Trainer for the steering local planner.

  Trains the local planner to imitate the path planner's steering corrections
  using behavior cloning.
  """

  def __init__(
    self,
    cfg: LocalPlannerTrainerCfg,
    num_envs: int,
    device: torch.device,
    network: nn.Module | None = None,
  ):
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device

    # --- Odometry mode: propagate to network config ---
    if cfg.use_odometry:
      cfg.network_cfg.output_dim = 1
      cfg.network_cfg.odom_vel_dim = cfg.odom_vel_dim
      if not cfg.include_joint_vel:
        cfg.network_cfg.joint_vel_dim = 0

    # Calculate observation dimension
    num_raycast_groups = len(cfg.raycast_obs_names)
    total_raycast_dim = cfg.raycast_dim * num_raycast_groups
    effective_joint_vel_dim = cfg.joint_vel_dim if cfg.include_joint_vel else 0
    effective_odom_vel_dim = cfg.odom_vel_dim if cfg.use_odometry else 0
    self.obs_dim = (
      cfg.heading_dim
      + effective_odom_vel_dim
      + effective_joint_vel_dim
      + total_raycast_dim
      + cfg.goal_direction_dim
    )
    self.target_dim = 1 if cfg.use_odometry else 2

    # Create or use provided network
    if network is None:
      self.network = LocalPlannerNetwork(cfg.network_cfg).to(device)
    else:
      self.network = network.to(device)

    # Optimizer
    self.optimizer = torch.optim.AdamW(
      self.network.parameters(),
      lr=cfg.lr,
      weight_decay=cfg.weight_decay,
    )

    # Learning rate scheduler
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      self.optimizer,
      T_max=10000,
      eta_min=cfg.lr * 0.1,
    )

    # Experience buffer with importance weighting based on raycast change
    raycast_offset = cfg.heading_dim + effective_odom_vel_dim + effective_joint_vel_dim
    self.buffer = SteeringExperienceBuffer(
      buffer_size=cfg.buffer_size,
      num_envs=num_envs,
      obs_dim=self.obs_dim,
      hidden_dim=cfg.network_cfg.state_dim,
      device=device,
      raycast_dim=total_raycast_dim,
      raycast_offset=raycast_offset,
      target_dim=self.target_dim,
    )

    # Observation builder
    self.obs_builder = LocalPlannerObservationBuilder(
      heading_dim=cfg.heading_dim,
      joint_vel_dim=cfg.joint_vel_dim,
      raycast_dim=cfg.raycast_dim,
      goal_direction_dim=cfg.goal_direction_dim,
      decision_dt=cfg.decision_dt,
      raycast_obs_names=cfg.raycast_obs_names,
      joint_vel_indices=cfg.joint_vel_indices,
      joint_vel_noise_std=cfg.joint_vel_noise_std,
      use_odometry=cfg.use_odometry,
      odom_vel_dim=cfg.odom_vel_dim,
      include_joint_vel=cfg.include_joint_vel,
    )

    # Hidden state for each environment
    self._hidden_states = torch.zeros(
      num_envs, cfg.network_cfg.state_dim, device=device
    )

    # Training statistics
    self.step_count = 0
    self.update_count = 0
    self.training_stats: dict[str, float] = {}
    self.best_loss = float("inf")

  def collect_experience(
    self,
    env: ManagerBasedRlEnv,
    obs_dict: dict[str, torch.Tensor],
    target: torch.Tensor,
    done: torch.Tensor,
    new_goal_mask: torch.Tensor | None = None,
  ) -> None:
    """Collect experience from environment step.

    Args:
      env: The RL environment
      obs_dict: Observation dictionary
      target: Training targets [num_envs, 2]: [delta_yaw, lin_vel]
      done: Episode termination flags [num_envs]
      new_goal_mask: Mask indicating which environments got new goals
    """
    # Update initial goal if new goals were assigned
    if new_goal_mask is not None:
      self.obs_builder.update_initial_goal(env, new_goal_mask)
    elif done.any():
      # On episode reset, consider it as a new goal
      self.obs_builder.update_initial_goal(env, done)

    # Detect key goal progression (Goal 1 → Goal 2) and re-init DR
    self.obs_builder.check_key_goal_change(env)

    # Build local planner observation
    local_planner_obs = self.obs_builder.build(env, obs_dict)

    # Teacher forcing: replace goal direction with GT + noise.
    # In odometry mode, goal direction is already odom-derived (near GT),
    # so we add smaller noise calibrated for odom drift.
    # In DR mode, this breaks the chicken-and-egg dependency between
    # pred_lin_vel accuracy and DR quality.
    if self.cfg.use_gt_goal_inputs or self.cfg.use_odometry:
      if self.cfg.use_odometry:
        bearing_noise = self.cfg.odom_bearing_noise_std
        distance_noise = self.cfg.odom_distance_noise_frac
      else:
        bearing_noise = self.cfg.gt_bearing_noise_std
        distance_noise = self.cfg.gt_distance_noise_frac

      true_bearing, true_distance = self.obs_builder.compute_true_bearing_distance(env)
      if true_bearing is not None and true_distance is not None:
        # Bearing noise (radians)
        if bearing_noise > 0:
          noise_b = torch.randn_like(true_bearing) * bearing_noise
        else:
          noise_b = torch.zeros_like(true_bearing)
        noisy_bearing = true_bearing + noise_b
        noisy_bearing = torch.atan2(torch.sin(noisy_bearing), torch.cos(noisy_bearing))

        # Distance noise (proportional to distance, with absolute floor)
        if distance_noise > 0:
          noise_std_d = (true_distance * distance_noise).clamp(min=0.05)
          noise_d = torch.randn_like(true_distance) * noise_std_d
        else:
          noise_d = torch.zeros_like(true_distance)
        noisy_distance = (true_distance + noise_d).clamp(min=0.1)

        # Build 4D goal direction from noisy bearing and distance
        cos_b = torch.cos(noisy_bearing)  # [B, 1]
        sin_b = torch.sin(noisy_bearing)  # [B, 1]
        dx_scaled = noisy_distance * cos_b / 10.0
        dy_scaled = noisy_distance * sin_b / 10.0

        # Replace last 4 dims with GT-derived 4D goal direction
        local_planner_obs = local_planner_obs.clone()
        local_planner_obs[:, -4:] = torch.cat(
          [dx_scaled, dy_scaled, cos_b, sin_b], dim=-1
        )

    # Add to buffer
    self.buffer.add(
      obs=local_planner_obs,
      target=target,
      hidden_state=self._hidden_states.clone(),
      done=done,
    )

    # Update hidden state (forward through network)
    with torch.no_grad():
      output, new_hidden = self.network(
        local_planner_obs,
        self._hidden_states,
        return_hidden=True,
      )
      self._hidden_states = new_hidden

      # Feed velocity back to obs_builder for dead reckoning
      # (no-op in odometry mode — obs_builder.update_dr returns immediately)
      robot = env.scene["robot"]
      # Compute world-frame yaw from quaternion (always needed for done-reset)
      robot_quat = robot.data.root_link_quat_w
      robot_yaw = torch.atan2(
        2.0
        * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
      )

      # Determine yaw rate source for DR heading integration
      if self.cfg.use_projected_yaw_rate:
        ang_vel_for_dr = LocalPlannerObservationBuilder.project_yaw_rate(
          robot.data.root_link_ang_vel_b, robot_quat
        )  # [B, 1] world yaw rate
      else:
        ang_vel_for_dr = robot.data.root_link_ang_vel_b[:, 2:3]  # body z

      if self.cfg.use_measured_velocity:
        # Use measured velocity from state estimator (less drift)
        dr_vel = robot.data.root_link_lin_vel_b[:, 0:1]  # [B, 1]
      elif self.network.cfg.output_dim >= 2:
        # Use network's predicted velocity (legacy)
        dr_vel = output[:, 1:2]
      else:
        # Odometry mode: no pred_lin_vel output, use measured as fallback
        dr_vel = robot.data.root_link_lin_vel_b[:, 0:1]

      if self.cfg.use_ahrs_heading:
        # Use AHRS heading – no integration drift
        self.obs_builder.update_dr(
          dr_vel,
          ang_vel_for_dr,
          done,
          robot_yaw,
          measured_heading=robot_yaw,
        )
      else:
        # Legacy: integrate angular velocity (heading drifts)
        self.obs_builder.update_dr(dr_vel, ang_vel_for_dr, done, robot_yaw)

    # Reset hidden state for done environments
    if done.any():
      self._hidden_states[done] = 0.0

    self.step_count += 1

  def update(self) -> dict[str, float] | None:
    """Perform training update if ready.

    Returns:
      Training statistics if update was performed, None otherwise.
    """
    # Check if ready for update
    if self.step_count < self.cfg.warmup_steps:
      return None
    if self.step_count % self.cfg.update_interval != 0:
      return None
    if not self.buffer.is_ready(self.cfg.batch_size * 2):
      return None

    self.network.train()

    # Sample batch
    try:
      obs, targets, initial_hidden, mask = self.buffer.sample_sequences(
        batch_size=self.cfg.batch_size,
        sequence_length=self.cfg.sequence_length,
      )
    except ValueError:
      # Not enough data yet
      obs, targets = self.buffer.sample_batch(self.cfg.batch_size)
      mask = None
      initial_hidden = None

    # Forward pass
    if initial_hidden is not None and mask is not None:
      # Sequential training
      batch_size, seq_len = obs.shape[:2]
      hidden = initial_hidden

      predictions = []
      for t in range(seq_len):
        pred, hidden = self.network(obs[:, t], hidden, return_hidden=True)
        predictions.append(pred)

      predictions = torch.stack(predictions, dim=1)  # [B, T, output_dim]

      # Main task loss: delta_yaw prediction
      yaw_pred = predictions[:, :, 0:1]
      yaw_target = targets[:, :, 0:1]
      correction_loss = ((yaw_pred - yaw_target) ** 2 * mask.unsqueeze(-1)).sum()
      correction_loss = correction_loss / (mask.sum() + 1e-8)

      # Auxiliary loss: linear velocity prediction (disabled in odom mode)
      if predictions.shape[-1] >= 2 and targets.shape[-1] >= 2:
        vel_pred = predictions[:, :, 1:2]
        vel_target = targets[:, :, 1:2]
        vel_pred_loss = ((vel_pred - vel_target) ** 2 * mask.unsqueeze(-1)).sum()
        vel_pred_loss = vel_pred_loss / (mask.sum() + 1e-8)
      else:
        vel_pred_loss = torch.tensor(0.0, device=self.device)

      # Smoothness loss (only on delta_yaw)
      yaw_diff = yaw_pred[:, 1:] - yaw_pred[:, :-1]
      smoothness_loss = (yaw_diff**2 * mask[:, 1:].unsqueeze(-1)).sum()
      smoothness_loss = smoothness_loss / (mask[:, 1:].sum() + 1e-8)

      # -- Proximity loss: amplify yaw error near obstacles ----------------
      # Extract raycast from obs: [B, T, obs_dim] → [B, T, raycast_total]
      rc_start = self.buffer.raycast_offset
      rc_end = rc_start + self.buffer.raycast_dim
      raycast_seq = obs[:, :, rc_start:rc_end]  # [B, T, raycast_dim]
      num_groups = self.buffer.raycast_dim // 64 if self.buffer.raycast_dim > 64 else 1
      B, T = raycast_seq.shape[:2]
      # Reshape to per-group 8x8, skip 3 ground-hitting columns (same as
      # importance weighting logic), then take min across horizontal rays.
      _grid = raycast_seq.reshape(B * T * num_groups, 8, 8)
      _horiz = _grid[:, :, 3:]  # skip 3 downward cols
      _flat_min = _horiz.reshape(B * T * num_groups, -1).min(dim=-1).values
      # Aggregate across groups → [B, T]
      min_ray = _flat_min.reshape(B, T, num_groups).min(dim=-1).values
      # Proximity weight: 1 when near real wall, 0 when far or occluded.
      # Near-field occlusion DR injects ~0.005–0.020 normalised values to
      # simulate the robot's own body blocking rays.  These are NOT real
      # obstacles, so zero out prox_w entirely when min_ray < floor.
      real_obstacle = (min_ray >= self.cfg.proximity_raycast_floor).float()
      prox_w = torch.clamp(1.0 - min_ray / self.cfg.proximity_loss_threshold, 0.0, 1.0)
      prox_w = prox_w * real_obstacle
      yaw_err_sq = (yaw_pred - yaw_target) ** 2  # [B, T, 1]
      proximity_loss = (yaw_err_sq.squeeze(-1) * prox_w * mask).sum()
      proximity_loss = proximity_loss / (mask.sum() + 1e-8)

    else:
      # Non-sequential training
      predictions = self.network(obs)
      correction_loss = nn.functional.mse_loss(predictions[:, 0:1], targets[:, 0:1])
      if predictions.shape[-1] >= 2 and targets.shape[-1] >= 2:
        vel_pred_loss = nn.functional.mse_loss(predictions[:, 1:2], targets[:, 1:2])
      else:
        vel_pred_loss = torch.tensor(0.0, device=self.device)
      smoothness_loss = torch.tensor(0.0, device=self.device)
      proximity_loss = torch.tensor(0.0, device=self.device)

    # Total loss
    # -- curriculum learning: per-task weight schedule ---------------------
    w_vel, w_corr = self._curriculum_weights()

    # In odom mode, use odom-specific loss weights
    if self.cfg.use_odometry:
      smoothness_w = self.cfg.odom_smoothness_loss_weight
      proximity_w = self.cfg.odom_proximity_loss_weight
      vel_pred_w = 0.0  # no velocity prediction in odom mode
    else:
      smoothness_w = self.cfg.smoothness_loss_weight
      proximity_w = self.cfg.proximity_loss_weight
      vel_pred_w = self.cfg.vel_pred_loss_weight

    total_loss = (
      self.cfg.correction_loss_weight * w_corr * correction_loss
      + vel_pred_w * w_vel * vel_pred_loss
      + smoothness_w * w_corr * smoothness_loss
      + proximity_w * w_corr * proximity_loss
    )

    # Backward pass
    self.optimizer.zero_grad()
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
      self.network.parameters(),
      self.cfg.grad_clip,
    )

    self.optimizer.step()
    self.scheduler.step()

    self.network.eval()
    self.update_count += 1

    # ---- Compute detailed diagnostic metrics (no grad) ----
    with torch.no_grad():
      vel_rmse = (
        vel_pred_loss.sqrt()
        if isinstance(vel_pred_loss, torch.Tensor)
        else torch.tensor(vel_pred_loss).sqrt()
      )
      yaw_rmse = (
        correction_loss.sqrt()
        if isinstance(correction_loss, torch.Tensor)
        else torch.tensor(correction_loss).sqrt()
      )

    # Update statistics
    self.training_stats = {
      "loss": total_loss.item(),
      "correction_loss": correction_loss.item(),
      "vel_pred_loss": vel_pred_loss.item(),
      "smoothness_loss": smoothness_loss.item(),
      "proximity_loss": proximity_loss.item(),
      "lr": self.optimizer.param_groups[0]["lr"],
      "cur_w_vel": w_vel,
      "cur_w_corr": w_corr,
      # Diagnostic RMSE metrics
      "vel_rmse": vel_rmse.item(),
      "yaw_rmse": yaw_rmse.item(),
    }

    # Track best loss
    if total_loss.item() < self.best_loss:
      self.best_loss = total_loss.item()

    return self.training_stats

  def _curriculum_weights(self) -> tuple[float, float]:
    """Compute per-task curriculum scaling factors.

    Returns ``(w_vel, w_corr)`` each in [0, 1]:

    - **Phase 1** (update 0 … p1):  w_vel=1, w_corr=0
      → Only velocity estimation is trained.  The hidden state learns to
        map joint velocities → body linear velocity.
    - **Phase 2** (p1 … p1+p2):  w_vel=1, w_corr ramps 0→1
      → Correction (delta_yaw) and smoothness losses are linearly
        introduced.  The network learns obstacle-avoiding steering
        using goal bearing / distance computed analytically from the
        velocity it already knows.
    - **Phase 3** (after p1+p2):  all = 1
      → All losses at full configured weight.

    In odometry mode, curriculum is disabled (all weights = 1 from start)
    because there is no velocity estimation chicken-and-egg problem.
    """
    # Odometry mode: no curriculum needed
    if self.cfg.use_odometry:
      return 1.0, 1.0

    p1 = self.cfg.curriculum_phase1_updates
    p2 = self.cfg.curriculum_phase2_updates
    u = self.update_count

    if p1 + p2 <= 0:
      return 1.0, 1.0  # curriculum disabled

    # vel: always on (enabled from update 0)
    w_vel = 1.0

    # correction: ramp during phase 2
    if u < p1:
      w_corr = 0.0
    elif u < p1 + p2:
      w_corr = (u - p1) / max(p2, 1)
    else:
      w_corr = 1.0

    return w_vel, w_corr

  def save(self, save_dir: str, filename: str) -> str:
    """Save the trained network.

    Args:
      save_dir: Directory to save to
      filename: Filename for the checkpoint

    Returns:
      Full path to saved file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filepath = save_path / filename

    # Save full state
    torch.save(
      {
        "network_state_dict": self.network.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.scheduler.state_dict(),
        "config": self.cfg,
        "step_count": self.step_count,
        "update_count": self.update_count,
        "best_loss": self.best_loss,
      },
      filepath,
    )

    return str(filepath)

  # ------------------------------------------------------------------
  # Checkpoint compatibility helpers
  # ------------------------------------------------------------------

  def _filter_ckpt_for_shape(
    self, ckpt_sd: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    """Return a state-dict safe to pass to ``load_state_dict(strict=False)``.

    ``strict=False`` skips missing / unexpected *keys* but still raises on
    shape mismatches for the **same** key.  This helper detects those
    mismatches, partially copies the overlapping slice, and returns a
    filtered dict that will load cleanly.
    """
    model_sd = self.network.state_dict()
    filtered: dict[str, torch.Tensor] = {}

    for key, ckpt_val in ckpt_sd.items():
      if key not in model_sd:
        continue
      model_val = model_sd[key]
      if ckpt_val.shape == model_val.shape:
        filtered[key] = ckpt_val
      else:
        print(
          f"[WARN] Partial load '{key}': ckpt {list(ckpt_val.shape)} → "
          f"model {list(model_val.shape)}"
        )
        slices = tuple(
          slice(0, min(c, m))
          for c, m in zip(ckpt_val.shape, model_val.shape, strict=False)
        )
        model_val[slices] = ckpt_val[slices]
        filtered[key] = model_val

    return filtered

  def load(self, checkpoint_path: str) -> None:
    """Load a saved checkpoint.

    Args:
      checkpoint_path: Path to the checkpoint file
    """
    checkpoint = load_local_planner_checkpoint(
      checkpoint_path,
      map_location=self.device,
    )

    missing, unexpected = self.network.load_state_dict(
      self._filter_ckpt_for_shape(checkpoint["network_state_dict"]), strict=False
    )
    if missing:
      print(
        f"[WARN] Missing keys in checkpoint (new layers use random init): {missing}"
      )
    if unexpected:
      print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    self.step_count = checkpoint.get("step_count", 0)
    self.update_count = checkpoint.get("update_count", 0)
    self.best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Step count: {self.step_count}")
    print(f"  Update count: {self.update_count}")
    print(f"  Best loss: {self.best_loss:.6f}")

  def print_stats(self) -> None:
    """Print training statistics."""
    if not self.training_stats:
      return

    s = self.training_stats
    w_corr = s.get("cur_w_corr", 1.0)

    # Determine curriculum phase label
    p1 = self.cfg.curriculum_phase1_updates
    p2 = self.cfg.curriculum_phase2_updates
    u = self.update_count
    if u <= p1:
      phase = f"P1-vel ({u}/{p1})"
    elif u <= p1 + p2:
      phase = f"P2-yaw ({u - p1}/{p2} w={w_corr:.2f})"
    else:
      phase = "Full"

    print(
      f"[LocalPlanner] Step {self.step_count:6d} | "
      f"Update {self.update_count:4d} | "
      f"{phase} | "
      f"Loss: {s['loss']:.4f} | "
      f"Yaw: {s['correction_loss']:.4f} | "
      f"Vel: {s['vel_pred_loss']:.4f} | "
      f"Smooth: {s['smoothness_loss']:.4f} | "
      f"Prox: {s.get('proximity_loss', 0.0):.4f} | "
      f"LR: {s['lr']:.6f}"
    )

  def should_save(self) -> bool:
    """Check if model should be saved (periodic saving)."""
    return self.update_count > 0 and self.update_count % self.cfg.save_interval == 0

  def reset_hidden_states(self, mask: torch.Tensor | None = None) -> None:
    """Reset hidden states for specified environments.

    Args:
      mask: Boolean mask indicating which environments to reset.
            If None, reset all environments.
    """
    if mask is None:
      self._hidden_states.zero_()
    else:
      self._hidden_states[mask] = 0.0
