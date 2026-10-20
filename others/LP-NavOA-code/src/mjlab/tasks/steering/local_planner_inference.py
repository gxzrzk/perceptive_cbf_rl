"""Runtime inference for the steering local planner.

This module runs the trained local planner and outputs delta_yaw commands for
the robot's heading control, replacing the global path planner.

The network outputs 2D: [delta_yaw, pred_lin_vel].
- delta_yaw is used for steering
- pred_lin_vel feeds back into the analytic dead reckoning to compute
  goal bearing and distance for the next step's input.

Key Features:
- Real-time delta_yaw output for Actor heading control
- Analytic goal bearing/distance computed from DR state
- Hidden state management for recurrent computation
- Output smoothing and saturation
- Integration with the command system

Usage:
  # Create inference module
  inference = LocalPlannerInference(cfg, local_planner_network, device=device)

  # Get delta_yaw command during control loop
  delta_yaw = inference.compute_correction(env, obs_dict)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from mjlab.tasks.steering.local_planner_checkpoint_compat import (
  load_local_planner_checkpoint,
)
from mjlab.tasks.steering.local_planner_trainer import (
  G1_LOCAL_PLANNER_JOINT_VEL_INDICES,
  LocalPlannerObservationBuilder,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class LocalPlannerInferenceCfg:
  """Configuration for local planner inference."""

  # Correction parameters
  max_correction: float = 1.7453
  """Maximum delta_yaw correction in radians. Must match the network's
  max_yaw_correction (1.7453 rad ≈ 100°) to avoid clipping trained output."""

  smoothing_alpha: float = 0.3
  """Exponential smoothing factor (0 = no smoothing, 1 = max smoothing)."""

  activation_threshold: float = 0.0
  """Minimum correction magnitude to apply (for deadband). Set to 0 to disable."""

  # Confidence-based gating
  use_confidence_gating: bool = False
  """Use confidence output to gate corrections."""

  confidence_threshold: float = 0.5
  """Minimum confidence to apply correction."""

  # Observation configuration (must match training)
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

  decision_dt: float = 0.02
  """Decision timestep for DR integration (1 / policy_freq). Default 50 Hz."""

  # Observation noise – sim-to-real robustness
  joint_vel_noise_std: float = 1.5
  """Std of uniform symmetric noise added to joint velocities during inference
  (rad/s).  Matches the training noise range to avoid the distribution shift
  where the network sees clean joint velocities in simulation but noisy
  encoder readings on hardware (causing ~0.1-0.2 m/s pred_lin_vel bias).
  Set to 0 to disable."""

  # --- DR sensor source ---
  use_measured_velocity: bool = False
  """Use measured forward velocity from the state estimator (leg kinematics
  + IMU fusion) instead of the network's pred_lin_vel for DR position.
  Default False = use network prediction (pred_lin_vel is sufficiently
  accurate after training)."""

  use_ahrs_heading: bool = True
  """Use AHRS (IMU orientation → yaw) for DR heading instead of
  integrating angular velocity.  Eliminates heading drift during turns.
  Default True because IMU quaternion yaw is always available and
  drift-free."""

  use_projected_yaw_rate: bool = False
  """Project body-frame angular velocity to world yaw rate using IMU
  quaternion before DR heading integration.  Corrects heading error
  when the robot pitches or rolls.
  Default False = use body-frame ang_vel_z directly."""

  # --- Goal-seeking gate (inference-time only) ---
  # Blends the learned avoidance output with an analytical goal-seeking
  # P-controller based on raycast clearance.  This keeps training pure
  # (imitation only), while adding goal-directed behaviour at test time.
  #
  #   goal_yaw = clamp(bearing * goal_seek_gain, -max_correction, max_correction)
  #   gate     = sigmoid((clearance_threshold - min_ray_top) * gate_scale)
  #              gate ~ 1 near obstacles, ~ 0 in open space
  #   final    = gate * avoidance_yaw + (1 - gate) * goal_yaw

  use_goal_seeking_gate: bool = False
  """Enable goal-seeking gate at inference time.
  Default False: the network learns conditional avoidance/goal-seeking
  end-to-end via dynamic target training (Plan A).  Set True to use the
  legacy analytical gate blending instead."""

  goal_seek_gain: float = 0.8
  """P-controller gain: goal_yaw = clamp(bearing * gain, -max, max).
  0.8 means bearing of 1.25 rad (72°) saturates the output."""

  raycast_max_distance: float = 4.0
  """Max raycast sensor distance (m).  Observations are normalised to
  [0, 1]; we multiply back before comparing against thresholds."""

  gate_clearance_threshold: float = 3.8
  """Raycast distance (m) below which avoidance dominates.
  Above this threshold the gate opens toward goal-seeking."""

  gate_scale: float = 3.0
  """Sigmoid steepness.  Larger = sharper transition.
  3.0 gives: 0.5m below threshold → gate≈0.82, 0.5m above → gate≈0.18."""

  gate_raycast_quantile: float = 0.5
  """Quantile (0–1) of all raycast distances used for gate clearance.
  Uses median to robustly ignore ground-hitting rays caused by the
  camera site's 90° roll (euler 0,-π/2,0)."""

  # --- Odometry mode ---
  use_odometry: bool = False
  """Enable odometry mode.  When True the network outputs 1D (delta_yaw
  only), DR is disabled, and goal direction is computed from odometry
  position instead of dead reckoning."""

  odom_vel_dim: int = 0
  """Dimension of odometry velocity observation (2 when use_odometry)."""

  include_joint_vel: bool = True
  """Include joint velocities in observations.  Typically False in odom mode."""

  enable_odom_goal_noise: bool = True
  """Add small Gaussian noise to odometry-mode goal direction at inference.
  This matches the odom goal-input noise used during training and makes the
  test input fluctuate around the simulator truth instead of being perfectly
  clean."""

  odom_bearing_noise_std: float = 0.06
  """Std of Gaussian noise added to odom-derived bearing in radians."""

  odom_distance_noise_frac: float = 0.03
  """Distance noise std as a fraction of odom-derived goal distance."""

  odom_distance_noise_min: float = 0.05
  """Minimum std in metres for odom-derived distance noise."""


@dataclass
class LocalPlannerMetrics:
  """Metrics for local planner inference monitoring."""

  correction_magnitude: float = 0.0
  """Average correction magnitude (post-clamp)."""

  raw_correction: float = 0.0
  """Raw correction value before smoothing/clamping (delta_yaw, dim 0)."""

  smoothed_correction: float = 0.0
  """Smoothed correction value before clamping."""

  full_output: tuple[float, float] = (0.0, 0.0)
  """Full 2D network output: (delta_yaw, pred_lin_vel)."""

  correction_count: int = 0
  """Number of corrections applied."""

  confidence_mean: float = 0.0
  """Average confidence value."""

  hidden_state_norm: float = 0.0
  """Average hidden state norm."""

  goal_seek_gate: float = 0.0
  """Current gate value: 1 = avoidance mode, 0 = goal-seeking mode."""

  goal_seek_yaw: float = 0.0
  """Analytical goal-seeking delta_yaw (before gating)."""

  avoidance_yaw: float = 0.0
  """Network avoidance delta_yaw (before gating)."""


class LocalPlannerInference:
  """Runtime inference for the steering local planner.

  This class manages the runtime inference of the local planner network as a
  navigation planner. The network directly outputs delta_yaw commands for
  the Actor, replacing the path planner's heading command.

  The network outputs [delta_yaw, pred_lin_vel].
  - delta_yaw (dim 0) is used for steering.
  - pred_lin_vel (dim 1) feeds back into analytic dead reckoning to
    compute goal bearing and distance for the next observation.

  Post-processing:
  - Smoothing (exponential moving average)
  - Deadband (threshold for minimum output)
  - Confidence gating (optional)
  """

  def __init__(
    self,
    cfg: LocalPlannerInferenceCfg,
    local_planner_network: nn.Module,
    device: torch.device | str = "cuda",
  ):
    self.cfg = cfg
    self.device = torch.device(device)

    # Local planner network
    self.local_planner_network = local_planner_network.to(self.device)
    self.local_planner_network.eval()

    # State management
    self._hidden_states: torch.Tensor | None = None
    self._prev_correction: torch.Tensor | None = None
    self._num_envs: int | None = None

    # Observation builder
    _jv_dim = (
      0 if (cfg.use_odometry and not cfg.include_joint_vel) else cfg.joint_vel_dim
    )
    self.obs_builder = LocalPlannerObservationBuilder(
      heading_dim=cfg.heading_dim,
      joint_vel_dim=_jv_dim,
      raycast_dim=cfg.raycast_dim,
      goal_direction_dim=cfg.goal_direction_dim,
      decision_dt=cfg.decision_dt,
      raycast_obs_names=cfg.raycast_obs_names,
      joint_vel_indices=cfg.joint_vel_indices if _jv_dim > 0 else None,
      joint_vel_noise_std=cfg.joint_vel_noise_std if _jv_dim > 0 else 0.0,
      use_odometry=cfg.use_odometry,
      odom_vel_dim=cfg.odom_vel_dim,
    )
    self._last_local_planner_obs: torch.Tensor | None = None

    # Metrics
    self.metrics = LocalPlannerMetrics()

  def initialize(self, num_envs: int) -> None:
    """Initialize inference state for given number of environments.

    Args:
      num_envs: Number of parallel environments
    """
    self._num_envs = num_envs
    state_dim = getattr(self.local_planner_network, "cfg", None)
    if state_dim is not None:
      state_dim = state_dim.state_dim
    else:
      # Try to infer from network
      state_dim = 32  # Default

    self._hidden_states = torch.zeros(num_envs, state_dim, device=self.device)
    self._prev_correction = torch.zeros(num_envs, 1, device=self.device)
    self.obs_builder.reset(num_envs, self.device)

  def reset(
    self,
    env_ids: torch.Tensor | None = None,
    initial_heading: torch.Tensor | None = None,
  ) -> None:
    """Reset inference state for specified environments.

    Args:
      env_ids: Environment indices to reset. If None, reset all.
      initial_heading: World-frame yaw angles for DR heading initialisation.
        Shape ``[num_envs]`` when *env_ids* is None, or ``[len(env_ids)]``
        otherwise.
    """
    if self._hidden_states is None or self._prev_correction is None:
      return

    if env_ids is None:
      self._hidden_states.zero_()
      self._prev_correction.zero_()
      self.obs_builder.reset_dr(initial_heading=initial_heading)
    else:
      self._hidden_states[env_ids] = 0.0
      self._prev_correction[env_ids] = 0.0
      self.obs_builder.reset_dr(env_ids, initial_heading=initial_heading)

  def load_network(self, path: str) -> None:
    """Load local planner network from file.

    Args:
      path: Path to saved local planner checkpoint
    """
    checkpoint = load_local_planner_checkpoint(path, map_location=self.device)

    if "network_state_dict" in checkpoint:
      self.local_planner_network.load_state_dict(checkpoint["network_state_dict"])
    else:
      # Assume it's just the state dict
      self.local_planner_network.load_state_dict(checkpoint)

    self.local_planner_network.eval()
    print(f"Loaded local planner network from {path}")

  def compute_correction(
    self,
    env: ManagerBasedRlEnv,
    obs_dict: dict[str, torch.Tensor],
    new_goal_mask: torch.Tensor | None = None,
    min_goal_bias: float = 0.0,
  ) -> torch.Tensor:
    """Compute steering correction from current observations.

    Args:
      env: The RL environment
      obs_dict: Observation dictionary
      new_goal_mask: Mask indicating which environments got new goals
      min_goal_bias: Minimum goal-seeking component added even when
        gate=1 (fully avoidance).  Prevents pure avoidance lock-in.

    Returns:
      Yaw rate correction [num_envs, 1]
    """
    if self._num_envs is None or self._num_envs != env.num_envs:
      self.initialize(env.num_envs)

    # ------------------------------------------------------------------
    # Detect episode resets and reset DR / hidden state BEFORE building
    # observations so the first post-reset obs has DR = [0, 0].
    # ------------------------------------------------------------------
    robot = env.scene["robot"]
    _reset_mask = env.episode_length_buf == 0
    if _reset_mask.any():
      _reset_ids = _reset_mask.nonzero(as_tuple=False).squeeze(-1)
      if not self.cfg.use_odometry:
        # Compute world-frame yaw for DR heading initialisation
        _quat = robot.data.root_link_quat_w
        _robot_yaw = torch.atan2(
          2.0 * (_quat[:, 0] * _quat[:, 3] + _quat[:, 1] * _quat[:, 2]),
          1.0 - 2.0 * (_quat[:, 2] ** 2 + _quat[:, 3] ** 2),
        )
        self.reset(_reset_ids, initial_heading=_robot_yaw[_reset_ids])
      else:
        self.reset(_reset_ids)
      # Update initial goal for reset environments
      self.obs_builder.update_initial_goal(env, _reset_mask)

    # Update initial goal if new goals were assigned
    if new_goal_mask is not None:
      self.obs_builder.update_initial_goal(env, new_goal_mask)

    # Detect key goal progression (Goal 1 → Goal 2) and re-init DR
    if not self.cfg.use_odometry:
      self.obs_builder.check_key_goal_change(env)

    # Build local planner observation
    local_planner_obs = self.obs_builder.build(env, obs_dict)
    local_planner_obs = self._apply_odom_goal_noise(local_planner_obs)
    self._last_local_planner_obs = local_planner_obs.detach()

    # Forward pass through network
    with torch.no_grad():
      if hasattr(self.local_planner_network, "forward"):
        result = self.local_planner_network(
          local_planner_obs,
          self._hidden_states,
          return_hidden=True,
        )
        if isinstance(result, tuple):
          full_output, new_hidden = result
          raw_correction = full_output[:, 0:1]  # Only delta_yaw for steering
          confidence = None
        elif isinstance(result, dict):
          raw_correction = result.get("delta_yaw", result.get("yaw_correction"))
          if raw_correction is None:
            raise KeyError(
              "Local planner output dict missing 'delta_yaw' and 'yaw_correction' keys"
            )
          new_hidden = result["hidden_state"]
          confidence = result.get("confidence", None)
          full_output = None
        else:
          raw_correction = result[:, 0:1] if result.shape[-1] > 1 else result
          new_hidden = self._hidden_states
          confidence = None
          full_output = result if result.shape[-1] >= 2 else None
      else:
        result = self.local_planner_network(local_planner_obs)
        raw_correction = result[:, 0:1] if result.shape[-1] > 1 else result
        new_hidden = self._hidden_states
        confidence = None
        full_output = result if result.shape[-1] >= 2 else None

    # Update hidden state
    self._hidden_states = new_hidden

    # Update dead reckoning (skip in odometry mode)
    if not self.cfg.use_odometry:
      # Determine yaw rate source for DR heading integration
      if self.cfg.use_projected_yaw_rate:
        _quat_dr = robot.data.root_link_quat_w
        ang_vel_for_dr = LocalPlannerObservationBuilder.project_yaw_rate(
          robot.data.root_link_ang_vel_b, _quat_dr
        )  # [B, 1] world yaw rate
      else:
        ang_vel_for_dr = robot.data.root_link_ang_vel_b[:, 2:3]  # body z

      # Determine velocity source
      if self.cfg.use_measured_velocity:
        dr_vel = robot.data.root_link_lin_vel_b[:, 0:1]  # [B, 1]
      elif full_output is not None and full_output.shape[-1] >= 2:
        dr_vel = full_output[:, 1:2]
      else:
        dr_vel = None

      if dr_vel is not None:
        # Determine heading source
        if self.cfg.use_ahrs_heading:
          _quat = robot.data.root_link_quat_w
          measured_heading = torch.atan2(
            2.0 * (_quat[:, 0] * _quat[:, 3] + _quat[:, 1] * _quat[:, 2]),
            1.0 - 2.0 * (_quat[:, 2] ** 2 + _quat[:, 3] ** 2),
          )  # [B]
          self.obs_builder.update_dr(
            dr_vel,
            ang_vel_for_dr,
            measured_heading=measured_heading,
          )
        else:
          # Legacy: integrate angular velocity (heading drifts)
          self.obs_builder.update_dr(dr_vel, ang_vel_for_dr)

    # Store raw correction value before any processing
    self.metrics.raw_correction = raw_correction[0, 0].item()

    # Store full 2D output if available
    if full_output is not None and full_output.shape[-1] >= 2:
      self.metrics.full_output = (
        full_output[0, 0].item(),
        full_output[0, 1].item(),
      )

    # Apply smoothing
    alpha = self.cfg.smoothing_alpha
    if self._prev_correction is None:
      self._prev_correction = raw_correction.clone()
    # Type assertion for type checker
    assert self._prev_correction is not None
    smoothed_correction = (1 - alpha) * raw_correction + alpha * self._prev_correction
    self._prev_correction = smoothed_correction.clone()

    # Store smoothed value before clamping
    self.metrics.smoothed_correction = smoothed_correction[0, 0].item()

    # Apply deadband
    correction = torch.where(
      torch.abs(smoothed_correction) < self.cfg.activation_threshold,
      torch.zeros_like(smoothed_correction),
      smoothed_correction,
    )

    # Apply confidence gating
    if self.cfg.use_confidence_gating and confidence is not None:
      gate = (confidence > self.cfg.confidence_threshold).float()
      correction = correction * gate

    # Clamp to max correction
    correction = torch.clamp(
      correction,
      -self.cfg.max_correction,
      self.cfg.max_correction,
    )

    # --- Goal-seeking gate: blend avoidance with analytical P-controller ---
    if self.cfg.use_goal_seeking_gate:
      # Extract bearing from 4D goal direction (last 4 dims)
      # Layout: [..., dx_scaled, dy_scaled, cos_bearing, sin_bearing]
      cos_b = local_planner_obs[:, -2:-1]  # cos_bearing [B, 1]
      sin_b = local_planner_obs[:, -1:]  # sin_bearing [B, 1]
      bearing = torch.atan2(sin_b, cos_b)  # [B, 1]

      # Analytical goal-seeking: simple P-controller on bearing
      goal_yaw = torch.clamp(
        bearing * self.cfg.goal_seek_gain,
        -self.cfg.max_correction,
        self.cfg.max_correction,
      )  # [B, 1]

      # Compute gate from raycast clearance (quantile of all rays)
      _jv = (
        0
        if (self.cfg.use_odometry and not self.cfg.include_joint_vel)
        else self.cfg.joint_vel_dim
      )
      _ray_s = self.cfg.heading_dim + _jv + self.cfg.odom_vel_dim
      _total_ray_dim = self.cfg.raycast_dim * len(self.cfg.raycast_obs_names)
      _ray_e = _ray_s + _total_ray_dim
      raycast_all = local_planner_obs[:, _ray_s:_ray_e]  # [B, total_raycast_dim]
      _q_idx = int(self.cfg.gate_raycast_quantile * raycast_all.shape[-1])
      _q_idx = min(_q_idx, raycast_all.shape[-1] - 1)
      _sorted_ray, _ = raycast_all.sort(dim=-1)
      min_ray = _sorted_ray[:, _q_idx : _q_idx + 1]  # [B, 1] normalised
      min_ray = min_ray * self.cfg.raycast_max_distance  # → metres

      # gate ~ 1 when obstacle close (avoidance), ~ 0 when clear (goal-seeking)
      gate = torch.sigmoid(
        (self.cfg.gate_clearance_threshold - min_ray) * self.cfg.gate_scale
      )  # [B, 1]

      # Store pre-blend values for metrics
      self.metrics.avoidance_yaw = correction[0, 0].item()
      self.metrics.goal_seek_yaw = goal_yaw[0, 0].item()
      self.metrics.goal_seek_gate = gate[0, 0].item()

      # Blend: final = gate * avoidance + (1 - gate) * goal + min_bias * goal
      # The min_bias ensures some goal-seeking even when gate=1 (obstacle present)
      correction = gate * correction + (1.0 - gate + min_goal_bias) * goal_yaw
    else:
      self.metrics.avoidance_yaw = correction[0, 0].item()
      self.metrics.goal_seek_yaw = 0.0
      self.metrics.goal_seek_gate = 1.0

    # Update metrics
    self.metrics.correction_magnitude = correction.abs().mean().item()
    self.metrics.correction_count = int((correction.abs() > 0.01).sum().item())
    if self._hidden_states is not None:
      self.metrics.hidden_state_norm = self._hidden_states.norm(dim=-1).mean().item()
    if confidence is not None:
      self.metrics.confidence_mean = confidence.mean().item()

    return correction

  def _apply_odom_goal_noise(self, obs: torch.Tensor) -> torch.Tensor:
    """Apply inference-time odometry noise to the final 4D goal direction."""
    if not (self.cfg.use_odometry and self.cfg.enable_odom_goal_noise):
      return obs

    bearing_std = float(self.cfg.odom_bearing_noise_std)
    distance_frac = float(self.cfg.odom_distance_noise_frac)
    if bearing_std <= 0.0 and distance_frac <= 0.0:
      return obs

    goal_dir = obs[:, -4:]
    dx_scaled = goal_dir[:, 0:1]
    dy_scaled = goal_dir[:, 1:2]
    cos_b = goal_dir[:, 2:3]
    sin_b = goal_dir[:, 3:4]

    bearing = torch.atan2(sin_b, cos_b)
    distance = (torch.sqrt(dx_scaled**2 + dy_scaled**2) * 10.0).clamp(min=0.1)

    if bearing_std > 0.0:
      noisy_bearing = bearing + torch.randn_like(bearing) * bearing_std
      noisy_bearing = torch.atan2(torch.sin(noisy_bearing), torch.cos(noisy_bearing))
    else:
      noisy_bearing = bearing

    if distance_frac > 0.0:
      distance_std = (distance * distance_frac).clamp(
        min=float(self.cfg.odom_distance_noise_min)
      )
      noisy_distance = (distance + torch.randn_like(distance) * distance_std).clamp(
        min=0.1
      )
    else:
      noisy_distance = distance

    noisy_cos = torch.cos(noisy_bearing)
    noisy_sin = torch.sin(noisy_bearing)
    noisy_goal_dir = torch.cat(
      [
        noisy_distance * noisy_cos / 10.0,
        noisy_distance * noisy_sin / 10.0,
        noisy_cos,
        noisy_sin,
      ],
      dim=-1,
    )

    noisy_obs = obs.clone()
    noisy_obs[:, -4:] = noisy_goal_dir
    return noisy_obs

  def apply_correction(
    self,
    original_wz: torch.Tensor,
    env: ManagerBasedRlEnv,
    obs_dict: dict[str, torch.Tensor],
    new_goal_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Get local planner delta_yaw output as navigation command.

    In the local planner paradigm, the network output directly provides
    the heading command rather than being added to the original.

    Args:
      original_wz: Original angular velocity command [num_envs] or [num_envs, 1] (unused)
      env: The RL environment
      obs_dict: Observation dictionary
      new_goal_mask: Mask indicating which environments got new goals

    Returns:
      Delta yaw command [num_envs] or [num_envs, 1]
    """
    delta_yaw = self.compute_correction(env, obs_dict, new_goal_mask)

    # Handle input shape
    if original_wz.dim() == 1:
      delta_yaw = delta_yaw.squeeze(-1)

    return delta_yaw

  def get_correction_only(
    self,
    local_planner_obs: torch.Tensor,
  ) -> torch.Tensor:
    """Get correction from pre-built observation (no environment needed).

    Useful for batch inference or deployment.

    Args:
      local_planner_obs: Local planner observation tensor [B, obs_dim]

    Returns:
      Yaw rate correction [B, 1]
    """
    B = local_planner_obs.shape[0]

    # Initialize hidden state if needed
    if self._hidden_states is None or self._hidden_states.shape[0] != B:
      state_dim = getattr(self.local_planner_network, "cfg", None)
      if state_dim is not None:
        state_dim = state_dim.state_dim
      else:
        state_dim = 32
      self._hidden_states = torch.zeros(B, state_dim, device=self.device)
      self._prev_correction = torch.zeros(B, 1, device=self.device)

    with torch.no_grad():
      result = self.local_planner_network(
        local_planner_obs,
        self._hidden_states,
        return_hidden=True,
      )
      if isinstance(result, tuple):
        full_output, new_hidden = result
        raw_correction = full_output[:, 0:1]  # Only delta_yaw
      else:
        raw_correction = result[:, 0:1] if result.shape[-1] > 1 else result
        new_hidden = self._hidden_states

    self._hidden_states = new_hidden

    # Apply smoothing
    alpha = self.cfg.smoothing_alpha
    if self._prev_correction is None:
      self._prev_correction = raw_correction.clone()
    # Type assertion for type checker
    assert self._prev_correction is not None
    smoothed = (1 - alpha) * raw_correction + alpha * self._prev_correction
    self._prev_correction = smoothed.clone()

    return torch.clamp(smoothed, -self.cfg.max_correction, self.cfg.max_correction)

  def get_metrics(self) -> dict[str, float | tuple[float, float]]:
    """Get current inference metrics.

    Returns:
      Dictionary of metrics including raw/smoothed corrections and full output.
    """
    return {
      "correction_magnitude": self.metrics.correction_magnitude,
      "raw_correction": self.metrics.raw_correction,
      "smoothed_correction": self.metrics.smoothed_correction,
      "full_output": self.metrics.full_output,
      "correction_count": self.metrics.correction_count,
      "confidence_mean": self.metrics.confidence_mean,
      "hidden_state_norm": self.metrics.hidden_state_norm,
      "goal_seek_gate": self.metrics.goal_seek_gate,
      "goal_seek_yaw": self.metrics.goal_seek_yaw,
      "avoidance_yaw": self.metrics.avoidance_yaw,
    }


class LocalPlannerCommandModifier:
  """Command modifier that integrates local planner correction with commands.

  This class wraps local planner inference and provides a clean interface for
  modifying velocity commands in the control loop.
  """

  def __init__(
    self,
    local_planner_inference: LocalPlannerInference,
  ):
    self.local_planner_inference = local_planner_inference
    self._enabled = True

  def enable(self) -> None:
    """Enable local planner correction."""
    self._enabled = True

  def disable(self) -> None:
    """Disable local planner correction."""
    self._enabled = False

  @property
  def enabled(self) -> bool:
    """Check if correction is enabled."""
    return self._enabled

  def modify_command(
    self,
    command: torch.Tensor,
    env: ManagerBasedRlEnv,
    obs_dict: dict[str, torch.Tensor],
    new_goal_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Modify velocity command with local planner correction.

    Args:
      command: Original velocity command [B, 3] (vx, vy, wz)
      env: The RL environment
      obs_dict: Observation dictionary
      new_goal_mask: Mask indicating which environments got new goals

    Returns:
      Modified velocity command [B, 3]
    """
    if not self._enabled:
      return command

    # Get local planner delta_yaw and replace heading component
    correction = self.local_planner_inference.compute_correction(
      env,
      obs_dict,
      new_goal_mask,
    )

    # Replace delta_yaw (heading) component with the local planner output
    modified_command = command.clone()
    modified_command[:, 0:1] = correction

    return modified_command

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset modifier state.

    Args:
      env_ids: Environment indices to reset
    """
    self.local_planner_inference.reset(env_ids)
