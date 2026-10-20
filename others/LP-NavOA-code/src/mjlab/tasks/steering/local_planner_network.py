"""Steering local planner network implementations.

The primary deployment model is ``LocalPlannerNetwork``, a GRU-based recurrent
local planner trained by behavior cloning from the global path planner.  This
module also keeps a legacy equilibrium-network variant for experiments that need
explicit contraction-style dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaycastCNNEncoder(nn.Module):
  """CNN encoder for a single frame of raycast distances.

  Same architecture as RaycastFrameEncoder in the velocity tasks:
  Conv2d(1, output_dim, kernel=3, stride=2, padding=1) on 8x8 grid,
  followed by Global Average Pooling.

  For 8x8 input:
    Conv2d: [B, 1, 8, 8] -> [B, C, 4, 4]
    GAP:    [B, C, 4, 4] -> [B, C]
  """

  def __init__(
    self,
    input_dim: int = 64,
    output_dim: int = 32,
    activation: str = "elu",
    grid_size: tuple[int, int] = (8, 8),
  ):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.grid_height, self.grid_width = grid_size

    expected_dim = self.grid_height * self.grid_width
    if input_dim != expected_dim:
      raise ValueError(
        f"Input dim {input_dim} doesn't match grid {grid_size} = {expected_dim}"
      )

    act = {
      "elu": nn.ELU(),
      "relu": nn.ReLU(),
      "tanh": nn.Tanh(),
      "leaky_relu": nn.LeakyReLU(),
      "gelu": nn.GELU(),
    }[activation]

    self.conv = nn.Conv2d(
      in_channels=1,
      out_channels=output_dim,
      kernel_size=3,
      stride=2,
      padding=1,
    )
    self.act = act
    self.gap = nn.AdaptiveAvgPool2d(1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode raycast distances.

    Args:
      x: Raycast distances [B, input_dim] (flattened 8x8 grid).

    Returns:
      Encoded features [B, output_dim].
    """
    B = x.shape[0]
    x = x.view(B, 1, self.grid_height, self.grid_width)
    x = self.act(self.conv(x))
    x = self.gap(x).view(B, -1)
    return x


@dataclass
class LocalPlannerNetworkCfg:
  """Configuration for steering local planner networks."""

  # Input dimensions
  base_heading_dim: int = 2
  """Dimension of robot heading from IMU quaternion: [cos_yaw, sin_yaw].
  Provides drift-free absolute orientation to the network.  The 2D
  trigonometric encoding avoids ±π wrapping discontinuities."""

  odom_vel_dim: int = 0
  """Dimension of odometry body-frame velocity input.
  0 = legacy (no odometry, use joint_vel for implicit velocity estimation).
  2 = odometry mode (vx, vy from odometry; typically joint_vel_dim=0)."""

  joint_vel_dim: int = 23
  """Dimension of joint velocities from motor encoders (23 for G1).
  Includes legs (12) + waist (3) + both upper arms (4+4).
  Set to 0 in odometry mode when joint velocities are not needed."""

  raycast_dim: int = 64
  """Dimension of a single raycast observation (8x8 grid = 64)."""

  num_raycast_groups: int = 1
  """Number of raycast observation groups.
  1 for single forward camera (n1), 3 for forward+left+right (n3).
  Each group is encoded independently through the same shared CNN,
  and the resulting latent vectors are concatenated."""

  goal_direction_dim: int = 4
  """Dimension of goal direction vector in body frame:
  [dx_scaled, dy_scaled, cos_bearing, sin_bearing].
  - dx/dy_scaled: body-frame goal displacement / 10 (distance info)
  - cos/sin_bearing: unit direction to goal (pure bearing info)
  This 4D representation provides a stronger directional signal than
  the original 1D scalar bearing, crucial for goal-seeking in open space."""

  # Legacy equilibrium-network parameters
  state_dim: int = 48
  """Dimension of the internal recurrent state (hidden state).
  48D gives enough capacity to remember avoidance/goal-seeking mode."""

  hidden_dims: tuple[int, ...] = (96, 64)
  """Hidden layer dimensions for input/output MLPs.
  First dim (96) is the input projection; second (64) is the output hidden."""

  contraction_rate: float = 0.9
  """Contraction rate for equilibrium constraint (0 < rate < 1).
  Smaller values = stronger contraction = more stable but less expressive.
  """

  equilibrium_iterations: int = 10
  """Number of iterations for implicit equilibrium computation."""

  equilibrium_tolerance: float = 1e-4
  """Tolerance for equilibrium convergence."""

  # Output configuration
  output_dim: int = 2
  """Output dimension: [delta_yaw, pred_lin_vel]."""

  max_yaw_correction: float = 1.7453
  """Maximum delta yaw output (rad). Output is clipped to [-max, max].
  1.7453 rad ≈ 100° allows aggressive high-speed avoidance turns."""

  max_vel_pred: float = 2.0
  """Maximum predicted linear velocity (m/s)."""

  # Hidden state decay (clearance-based, Proposal 2a)
  raycast_max_distance: float = 4.0
  """Max raycast sensor distance (m).  Observations are normalised to
  [0, 1] by dividing by this value; we multiply back before comparing
  against metre-based thresholds."""

  hidden_decay_threshold: float = 3.0
  """Raycast clearance (m) below which hidden state is fully preserved.
  Above this threshold, the hidden state decays toward zero."""

  hidden_decay_scale: float = 3.0
  """Sigmoid steepness for decay transition."""

  hidden_decay_min: float = 0.9
  """Minimum decay factor in open space. 0.9 means 10% decay per step,
  flushing stale avoidance memory over ~10 steps."""

  hidden_decay_quantile: float = 0.5
  """Quantile (0–1) of all raycast distances used for clearance estimate.
  Using the median (0.5) is robust to the ~25% of rays that always hit
  the ground regardless of obstacles, because the camera site euler
  (0, -π/2, 0) rotates image rows into left↔right and columns into
  down↔up, so every row contains downward-looking rays."""

  output_smoothing: float = 0.3
  """Exponential smoothing factor for output (0 = no smoothing, 1 = max smoothing)."""

  # Raycast CNN encoder configuration
  use_raycast_cnn: bool = True
  """Use CNN encoder for raycast observations instead of raw flattened input.
  When True, the 8x8 raycast grid is processed by a Conv2d encoder
  (same architecture as RaycastFrameEncoder in the velocity tasks)
  before being concatenated with other observations."""

  raycast_grid_size: tuple[int, int] = (8, 8)
  """Raycast grid dimensions (height, width) for CNN reshape."""

  raycast_cnn_output_dim: int = 32
  """Output dimension of the per-frame CNN encoder.
  The raw 64-dim raycast is replaced by this lower-dimensional latent."""

  raycast_cnn_activation: str = "elu"
  """Activation function used inside the CNN encoder."""

  use_independent_raycast_cnn: bool = False
  """Use independent CNN encoders for each raycast group instead of sharing.
  When True, each raycast group (front/left/right) gets its own CNN with
  independent weights, allowing direction-specific feature extraction.
  When False (default), all groups share the same CNN weights (legacy)."""

  # Activation and initialization
  activation: str = "tanh"
  """Activation function: 'tanh', 'relu', 'elu'."""

  init_scale: float = 0.1
  """Scale for weight initialization."""


class EquilibriumPlannerLayer(nn.Module):
  """Legacy equilibrium recurrent layer.

  Implements an implicit fixed-point update with contracting dynamics.

  The layer computes: h = σ(W_h @ h + W_x @ x + b)
  subject to the equilibrium constraint that makes the dynamics contracting.

  This is achieved by parameterizing W_h such that its spectral radius < 1.
  """

  def __init__(
    self,
    state_dim: int,
    input_dim: int,
    contraction_rate: float = 0.9,
    equilibrium_iterations: int = 10,
    equilibrium_tolerance: float = 1e-4,
  ):
    super().__init__()
    self.state_dim = state_dim
    self.input_dim = input_dim
    self.contraction_rate = contraction_rate
    self.equilibrium_iterations = equilibrium_iterations
    self.equilibrium_tolerance = equilibrium_tolerance

    # State transition weights (parameterized for contraction)
    # We use a skew-symmetric + diagonal parameterization for stability
    self.W_skew = nn.Parameter(torch.zeros(state_dim, state_dim))
    self.W_diag = nn.Parameter(torch.ones(state_dim) * -0.5)

    # Input projection
    self.W_x = nn.Linear(input_dim, state_dim)

    # Bias
    self.bias = nn.Parameter(torch.zeros(state_dim))

    # Initialize
    self._init_weights()

  def _init_weights(self) -> None:
    """Initialize weights for stability."""
    nn.init.orthogonal_(self.W_skew, gain=0.1)
    nn.init.uniform_(self.W_diag, -0.8, -0.2)
    nn.init.orthogonal_(self.W_x.weight, gain=0.5)
    nn.init.zeros_(self.W_x.bias)

  def _get_stable_W_h(self) -> torch.Tensor:
    """Construct stable state transition matrix.

    Uses skew-symmetric + negative diagonal parameterization to ensure
    the eigenvalues have negative real parts, guaranteeing contraction.

    Returns:
      State transition matrix with spectral radius < contraction_rate.
    """
    # Skew-symmetric part (eigenvalues are purely imaginary)
    W_skew = self.W_skew - self.W_skew.t()

    # Negative diagonal ensures stability
    W_diag = torch.diag(self.contraction_rate * torch.tanh(self.W_diag))

    # Combined matrix
    W_h = W_skew + W_diag

    return W_h

  def forward(
    self,
    x: torch.Tensor,
    h_prev: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Forward pass through the equilibrium layer.

    Args:
      x: Input tensor [B, input_dim]
      h_prev: Previous hidden state [B, state_dim], or None for initial

    Returns:
      New hidden state [B, state_dim]
    """
    B = x.shape[0]
    device = x.device

    # Initialize hidden state if not provided
    if h_prev is None:
      h_prev = torch.zeros(B, self.state_dim, device=device)

    # Get stable transition matrix
    W_h = self._get_stable_W_h()

    # Input contribution
    x_proj = self.W_x(x)

    # Implicit equilibrium iteration
    # h = tanh(W_h @ h + W_x @ x + bias)
    # We iterate to find the fixed point
    h = h_prev
    h_new = h_prev  # Initialize h_new
    for _ in range(self.equilibrium_iterations):
      h_new = torch.tanh(F.linear(h, W_h) + x_proj + self.bias)

      # Check convergence
      if torch.norm(h_new - h) < self.equilibrium_tolerance:
        break

      # Mix with previous for stability
      h = 0.5 * h_new + 0.5 * h

    return h_new


class EquilibriumLocalPlannerNetwork(nn.Module):
  """Legacy equilibrium local planner network.

  This network serves as a local navigation planner based on:
  - Current yaw rate (1D)
  - Joint velocities from motor encoders (19D)
  - Raycast observations (64D)
  - Goal bearing angle from DR estimate (1D)
  - Goal distance from DR estimate (1D)

  Outputs a 2D vector:
  - delta_yaw (1D): heading control command for Actor
  - pred_lin_vel (1D): predicted current linear velocity (for external DR)

  Architecture:
    Input MLP -> equilibrium layer -> Output MLP -> [delta_yaw, pred_lin_vel]
  """

  def __init__(self, cfg: LocalPlannerNetworkCfg):
    super().__init__()
    self.cfg = cfg

    # --- Raycast CNN encoder (optional) ---
    if cfg.use_raycast_cnn:
      if cfg.use_independent_raycast_cnn and cfg.num_raycast_groups > 1:
        # Independent CNN for each raycast group (front/left/right)
        self.raycast_cnn = None
        self.raycast_cnns = nn.ModuleList(
          [
            RaycastCNNEncoder(
              input_dim=cfg.raycast_dim,
              output_dim=cfg.raycast_cnn_output_dim,
              activation=cfg.raycast_cnn_activation,
              grid_size=cfg.raycast_grid_size,
            )
            for _ in range(cfg.num_raycast_groups)
          ]
        )
      else:
        # Shared CNN across all groups (legacy)
        self.raycast_cnn = RaycastCNNEncoder(
          input_dim=cfg.raycast_dim,
          output_dim=cfg.raycast_cnn_output_dim,
          activation=cfg.raycast_cnn_activation,
          grid_size=cfg.raycast_grid_size,
        )
        self.raycast_cnns = None
      raycast_feature_dim = cfg.raycast_cnn_output_dim * cfg.num_raycast_groups
    else:
      self.raycast_cnn = None
      self.raycast_cnns = None
      raycast_feature_dim = cfg.raycast_dim * cfg.num_raycast_groups

    # Total raw raycast dimension in the observation tensor
    self._total_raw_raycast_dim = cfg.raycast_dim * cfg.num_raycast_groups

    # Offset to the start of raycast in the raw observation tensor
    self._raycast_obs_offset = (
      cfg.base_heading_dim + cfg.odom_vel_dim + cfg.joint_vel_dim
    )

    # Calculate total input dimension
    self.input_dim = (
      cfg.base_heading_dim
      + cfg.odom_vel_dim
      + cfg.joint_vel_dim
      + raycast_feature_dim
      + cfg.goal_direction_dim
    )

    # Input encoder
    input_layers = []
    prev_dim = self.input_dim
    activation = self._get_activation(cfg.activation)

    for hidden_dim in cfg.hidden_dims:
      input_layers.append(nn.Linear(prev_dim, hidden_dim))
      input_layers.append(nn.LayerNorm(hidden_dim))
      input_layers.append(activation)
      prev_dim = hidden_dim

    self.input_encoder = nn.Sequential(*input_layers)

    # Equilibrium layer with contraction-style dynamics.
    self.equilibrium_layer = EquilibriumPlannerLayer(
      state_dim=cfg.state_dim,
      input_dim=prev_dim,
      contraction_rate=cfg.contraction_rate,
      equilibrium_iterations=cfg.equilibrium_iterations,
      equilibrium_tolerance=cfg.equilibrium_tolerance,
    )

    # Output decoder
    output_layers = []
    prev_dim = cfg.state_dim
    for hidden_dim in reversed(cfg.hidden_dims):
      output_layers.append(nn.Linear(prev_dim, hidden_dim))
      output_layers.append(nn.LayerNorm(hidden_dim))
      output_layers.append(activation)
      prev_dim = hidden_dim

    # Final output layer: [delta_yaw, pred_lin_vel]
    output_layers.append(nn.Linear(prev_dim, cfg.output_dim))
    # No activation here - per-dimension scaling applied in forward()

    self.output_decoder = nn.Sequential(*output_layers)

    # Output smoothing state
    self._prev_output: torch.Tensor | None = None

    # Initialize weights
    self._init_weights()

  def _get_activation(self, name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
      "relu": nn.ReLU(),
      "tanh": nn.Tanh(),
      "elu": nn.ELU(),
      "leaky_relu": nn.LeakyReLU(),
      "gelu": nn.GELU(),
    }
    if name not in activations:
      raise ValueError(
        f"Unknown activation: {name}. Choose from {list(activations.keys())}"
      )
    return activations[name]

  def _init_weights(self) -> None:
    """Initialize network weights."""
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=self.cfg.init_scale)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def forward(
    self,
    obs: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
    return_hidden: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward pass.

    Args:
      obs: Input observation tensor of shape [B, input_dim] where:
        - obs[:, 0:2]:  base_heading [cos_yaw, sin_yaw] (from IMU quat)
        - obs[:, 2:25]: joint_vel (23 joint velocities from encoders)
        - obs[:, 25:89]: raycast (8x8 = 64 values)
        - obs[:, 89:93]: goal_direction (4D body-frame goal vector)
          [dx_scaled, dy_scaled, cos_bearing, sin_bearing]

      hidden_state: Previous recurrent hidden state [B, state_dim], or None

      return_hidden: If True, also return the new hidden state

    Returns:
      If return_hidden=False:
        Output tensor [B, 2]: [delta_yaw, pred_lin_vel]
      If return_hidden=True:
        Tuple of (output [B, 2], new_hidden_state)
    """
    # --- Extract and encode raycast group(s) through CNN ---
    if self.raycast_cnn is not None or self.raycast_cnns is not None:
      ray_s = self._raycast_obs_offset
      ray_e = ray_s + self._total_raw_raycast_dim
      pre_ray = obs[:, :ray_s]  # heading [+ odom_vel] [+ joint_vel]
      post_ray = obs[:, ray_e:]  # goal_direction
      latents = []
      for g in range(self.cfg.num_raycast_groups):
        g_s = ray_s + g * self.cfg.raycast_dim
        g_e = g_s + self.cfg.raycast_dim
        if self.raycast_cnns is not None:
          latents.append(self.raycast_cnns[g](obs[:, g_s:g_e]))
        else:
          assert self.raycast_cnn is not None
          latents.append(self.raycast_cnn(obs[:, g_s:g_e]))
      obs_encoded = torch.cat([pre_ray, *latents, post_ray], dim=-1)
    else:
      obs_encoded = obs

    # Encode input
    encoded = self.input_encoder(obs_encoded)

    # Legacy equilibrium recurrence
    new_hidden = self.equilibrium_layer(encoded, hidden_state)

    # Decode output [B, output_dim]
    raw_output = self.output_decoder(new_hidden)

    # Per-dimension scaling
    delta_yaw = torch.tanh(raw_output[:, 0:1]) * self.cfg.max_yaw_correction
    if self.cfg.output_dim >= 2:
      pred_lin_vel = torch.tanh(raw_output[:, 1:2]) * self.cfg.max_vel_pred
      output = torch.cat([delta_yaw, pred_lin_vel], dim=-1)
    else:
      output = delta_yaw

    if return_hidden:
      return output, new_hidden
    return output

  def predict(
    self,
    base_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
    raycast: torch.Tensor,
    goal_direction: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict navigation outputs from separate observations.

    Args:
      base_ang_vel: [B, 1] base angular velocity (yaw rate only)
      joint_vel: [B, 23] joint velocities from motor encoders
      raycast: [B, 64] raycast distances (8x8 grid)
      goal_direction: [B, 4] body-frame goal direction vector
        [dx_scaled, dy_scaled, cos_bearing, sin_bearing]
      hidden_state: [B, state_dim] previous hidden state or None

    Returns:
      Tuple of (output [B, 2], new_hidden_state [B, state_dim])
    """
    obs = torch.cat([base_ang_vel, joint_vel, raycast, goal_direction], dim=-1)
    return self.forward(obs, hidden_state, return_hidden=True)  # type: ignore

  def reset_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Get initial hidden state (zeros).

    Args:
      batch_size: Number of environments
      device: Device for the tensor

    Returns:
      Zero-initialized hidden state [batch_size, state_dim]
    """
    return torch.zeros(batch_size, self.cfg.state_dim, device=device)


class LocalPlannerNetwork(nn.Module):
  """GRU-based local planner used for training and deployment.

  The network consumes local sensor observations and emits two values:
  ``delta_yaw`` for steering and ``pred_lin_vel`` for dead reckoning.
  """

  def __init__(self, cfg: LocalPlannerNetworkCfg):
    super().__init__()
    self.cfg = cfg

    # --- Raycast CNN encoder (optional) ---
    if cfg.use_raycast_cnn:
      if cfg.use_independent_raycast_cnn and cfg.num_raycast_groups > 1:
        # Independent CNN for each raycast group (front/left/right)
        self.raycast_cnn = None
        self.raycast_cnns = nn.ModuleList(
          [
            RaycastCNNEncoder(
              input_dim=cfg.raycast_dim,
              output_dim=cfg.raycast_cnn_output_dim,
              activation=cfg.raycast_cnn_activation,
              grid_size=cfg.raycast_grid_size,
            )
            for _ in range(cfg.num_raycast_groups)
          ]
        )
      else:
        # Shared CNN across all groups (legacy)
        self.raycast_cnn = RaycastCNNEncoder(
          input_dim=cfg.raycast_dim,
          output_dim=cfg.raycast_cnn_output_dim,
          activation=cfg.raycast_cnn_activation,
          grid_size=cfg.raycast_grid_size,
        )
        self.raycast_cnns = None
      raycast_feature_dim = cfg.raycast_cnn_output_dim * cfg.num_raycast_groups
    else:
      self.raycast_cnn = None
      self.raycast_cnns = None
      raycast_feature_dim = cfg.raycast_dim * cfg.num_raycast_groups

    # Total raw raycast dimension in the observation tensor
    self._total_raw_raycast_dim = cfg.raycast_dim * cfg.num_raycast_groups

    # Offset to the start of raycast in the raw observation tensor
    self._raycast_obs_offset = (
      cfg.base_heading_dim + cfg.odom_vel_dim + cfg.joint_vel_dim
    )

    # Calculate total input dimension
    self.input_dim = (
      cfg.base_heading_dim
      + cfg.odom_vel_dim
      + cfg.joint_vel_dim
      + raycast_feature_dim
      + cfg.goal_direction_dim
    )

    # Input projection
    self.input_proj = nn.Sequential(
      nn.Linear(self.input_dim, cfg.hidden_dims[0]),
      nn.LayerNorm(cfg.hidden_dims[0]),
      nn.Tanh(),
    )

    # Recurrent layer (GRU for simplicity)
    self.gru = nn.GRUCell(
      input_size=cfg.hidden_dims[0],
      hidden_size=cfg.state_dim,
    )

    # Note: Spectral normalization can be applied during training
    # by using hooks or custom forward methods

    # Output layers: [delta_yaw, pred_lin_vel]
    self.output_hidden = nn.Sequential(
      nn.Linear(cfg.state_dim, cfg.hidden_dims[-1]),
      nn.LayerNorm(cfg.hidden_dims[-1]),
      nn.Tanh(),
    )
    self.output_head = nn.Linear(cfg.hidden_dims[-1], cfg.output_dim)

    self._init_weights()

  def _init_weights(self) -> None:
    """Initialize weights."""
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.5)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def forward(
    self,
    obs: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
    return_hidden: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Forward pass.

    Args:
      obs: Input observation tensor [B, input_dim]
      hidden_state: Previous hidden state [B, state_dim], or None
      return_hidden: If True, also return the new hidden state

    Returns:
      Output [B, 2]: [delta_yaw, pred_lin_vel],
      optionally with hidden state
    """
    B = obs.shape[0]
    device = obs.device

    # Initialize hidden state if needed
    if hidden_state is None:
      hidden_state = torch.zeros(B, self.cfg.state_dim, device=device)

    # --- Clearance-based hidden state decay (Proposal 2a) --- DISABLED
    # Temporarily disabled for A/B comparison.  With decay, the network
    # forgets avoidance memory too quickly in open space, causing abrupt
    # rotation when raycast clears.  Without decay, the GRU naturally
    # forgets over time via its own forget gate.
    #
    # _ray_s = self.cfg.base_heading_dim + self.cfg.joint_vel_dim
    # _ray_e = _ray_s + self.cfg.raycast_dim
    # _raycast = obs[:, _ray_s:_ray_e]  # [B, raycast_dim]
    # _q_idx = int(self.cfg.hidden_decay_quantile * _raycast.shape[-1])
    # _q_idx = min(_q_idx, _raycast.shape[-1] - 1)
    # _sorted_ray, _ = _raycast.sort(dim=-1)
    # _clearance = _sorted_ray[:, _q_idx]  # [B] normalised [0,1]
    # _clearance = _clearance * self.cfg.raycast_max_distance  # → metres
    # _decay = self.cfg.hidden_decay_min + (
    #   1.0 - self.cfg.hidden_decay_min
    # ) * torch.sigmoid(
    #   (self.cfg.hidden_decay_threshold - _clearance) * self.cfg.hidden_decay_scale
    # )  # [B]
    # hidden_state = hidden_state * _decay.unsqueeze(-1)
    _decay = torch.ones(B, device=device)  # no decay (keep full hidden state)

    # --- Extract and encode raycast group(s) through CNN ---
    if self.raycast_cnn is not None or self.raycast_cnns is not None:
      ray_s = self._raycast_obs_offset
      ray_e = ray_s + self._total_raw_raycast_dim
      pre_ray = obs[:, :ray_s]  # heading [+ odom_vel] [+ joint_vel]
      post_ray = obs[:, ray_e:]  # goal_direction
      latents = []
      for g in range(self.cfg.num_raycast_groups):
        g_s = ray_s + g * self.cfg.raycast_dim
        g_e = g_s + self.cfg.raycast_dim
        if self.raycast_cnns is not None:
          latents.append(self.raycast_cnns[g](obs[:, g_s:g_e]))
        else:
          assert self.raycast_cnn is not None
          latents.append(self.raycast_cnn(obs[:, g_s:g_e]))
      obs_for_proj = torch.cat([pre_ray, *latents, post_ray], dim=-1)
    else:
      obs_for_proj = obs

    # Input projection
    x = self.input_proj(obs_for_proj)

    # GRU step
    new_hidden = self.gru(x, hidden_state)

    # Output: per-dimension scaling
    hidden_out = self.output_hidden(new_hidden)
    raw_output = self.output_head(hidden_out)  # [B, output_dim]
    delta_yaw = torch.tanh(raw_output[:, 0:1]) * self.cfg.max_yaw_correction
    if self.cfg.output_dim >= 2:
      pred_lin_vel = torch.tanh(raw_output[:, 1:2]) * self.cfg.max_vel_pred
      output = torch.cat([delta_yaw, pred_lin_vel], dim=-1)
    else:
      output = delta_yaw

    if return_hidden:
      return output, new_hidden
    return output

  def reset_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
    """Get initial hidden state (zeros)."""
    return torch.zeros(batch_size, self.cfg.state_dim, device=device)


class LocalPlannerSteeringCorrector(nn.Module):
  """Wrapper that adds confidence estimation on top of a planner network.

  This module wraps the local planner network and provides:
  - delta_yaw: primary heading command for Actor
  - pred_lin_vel: predicted linear velocity (auxiliary, feeds DR)
  - confidence: optional confidence estimate
  - State management for recurrent computation
  """

  def __init__(
    self,
    cfg: LocalPlannerNetworkCfg,
    use_equilibrium_network: bool = False,
  ):
    super().__init__()
    self.cfg = cfg

    # Main network
    if use_equilibrium_network:
      self.network = EquilibriumLocalPlannerNetwork(cfg)
    else:
      self.network = LocalPlannerNetwork(cfg)

    # Auxiliary head for confidence estimation
    self.confidence_head = nn.Sequential(
      nn.Linear(cfg.state_dim, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid(),
    )

    # Hidden state storage
    self._hidden_state: torch.Tensor | None = None

  def forward(
    self,
    obs: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Forward pass with all outputs.

    Args:
      obs: Input observations [B, input_dim]
      hidden_state: Optional hidden state override

    Returns:
      Dictionary with:
        - 'yaw_correction': [B, 1] steering correction
        - 'confidence': [B, 1] confidence in the correction
        - 'hidden_state': [B, state_dim] new hidden state
    """
    if hidden_state is None:
      hidden_state = self._hidden_state

    full_output, new_hidden = self.network(obs, hidden_state, return_hidden=True)
    confidence = self.confidence_head(new_hidden)

    # Update stored hidden state
    self._hidden_state = new_hidden

    result = {
      "delta_yaw": full_output[:, 0:1],
      "output": full_output,
      "confidence": confidence,
      "hidden_state": new_hidden,
    }
    if full_output.shape[-1] >= 2:
      result["pred_lin_vel"] = full_output[:, 1:2]
    return result

  def reset(self, batch_size: int, device: torch.device) -> None:
    """Reset hidden state for new episodes."""
    self._hidden_state = self.network.reset_hidden(batch_size, device)

  def get_correction(
    self,
    obs: torch.Tensor,
    hidden_state: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Get only the yaw correction (convenience method).

    Args:
      obs: Input observations [B, input_dim]
      hidden_state: Optional hidden state override

    Returns:
      Yaw correction [B, 1]
    """
    outputs = self.forward(obs, hidden_state)
    return outputs["delta_yaw"]
