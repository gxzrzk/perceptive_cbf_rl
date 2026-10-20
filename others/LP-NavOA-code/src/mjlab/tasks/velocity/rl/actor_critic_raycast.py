"""Actor-Critic with Raycast Encoding for obstacle avoidance.

This module provides a raycast-aware model that processes raycast distance
observations using a temporal encoder similar to depth image processing.
The encoder extracts features from each frame independently, then uses 1D
convolution to aggregate temporal information across frames.

Compatible with rsl_rl >= 5.0 where actor and critic are separate model
instances. Each instance of ActorCriticRaycast handles a single role
("actor" or "critic").

Architecture:
1. Per-frame CNN feature extraction (for 8x8 raycast grid)
2. 1D Conv temporal aggregation
3. MLP fusion to latent space
4. Concatenation with other observations for the role's MLP
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from rsl_rl.modules import MLP, EmpiricalNormalization

if TYPE_CHECKING:
  from rsl_rl.modules.distribution import Distribution


def get_activation(act_name: str) -> nn.Module:
  """Get activation function by name."""
  activations = {
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
  }
  if act_name not in activations:
    raise ValueError(
      f"Invalid activation: {act_name}. Choose from {list(activations.keys())}"
    )
  return activations[act_name]


class RaycastFrameEncoder(nn.Module):
  """Encodes a single frame of raycast distances using a minimal CNN.

  Designed for 8x8 raycast grid. Uses a single-layer CNN with Global Average
  Pooling (GAP) for maximum parameter efficiency.

  Architecture for 8x8 input:
    Conv2d(1, output_dim, kernel=3, stride=2, padding=1) -> [B, output_dim, 4, 4]
    GlobalAvgPool -> [B, output_dim]

  Design rationale:
    - 8x8 = 64 pixels is very small, deep networks unnecessary
    - 3x3 kernel with stride=2 provides 2x2 effective receptive field per output
    - GAP eliminates FC layer parameters, provides spatial invariance
    - Total params: 1 * output_dim * 3 * 3 + output_dim = 10 * output_dim (~320 for dim=32)
  """

  def __init__(
    self,
    input_dim: int,
    output_dim: int = 32,
    activation: str = "elu",
    grid_size: tuple[int, int] = (8, 8),
  ):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.grid_height, self.grid_width = grid_size

    # Verify input dimension matches grid
    expected_dim = self.grid_height * self.grid_width
    if input_dim != expected_dim:
      raise ValueError(
        f"Input dim {input_dim} doesn't match grid {grid_size} = {expected_dim}"
      )

    act = get_activation(activation)

    # Minimal CNN: single conv layer + global average pooling
    # Conv: 8x8x1 -> 4x4xC (stride=2 downsamples by 2x)
    # GAP: 4x4xC -> 1x1xC -> C
    self.conv = nn.Conv2d(
      in_channels=1,
      out_channels=output_dim,
      kernel_size=3,
      stride=2,
      padding=1,
    )
    self.act = act
    self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

    # Count parameters
    num_params = sum(p.numel() for p in self.parameters())
    print(
      f"RaycastFrameEncoder: grid={grid_size}, output={output_dim}, params={num_params}"
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode raycast distances using minimal CNN.

    Args:
      x: Raycast distances of shape [batch_size, input_dim]

    Returns:
      Encoded features of shape [batch_size, output_dim]
    """
    batch_size = x.shape[0]
    # Reshape to image format: [B, D] -> [B, 1, H, W]
    x = x.view(batch_size, 1, self.grid_height, self.grid_width)
    # Single conv layer
    x = self.conv(x)
    x = self.act(x)
    # Global average pooling: [B, C, H', W'] -> [B, C, 1, 1] -> [B, C]
    x = self.gap(x)
    x = x.view(batch_size, -1)
    return x


class StackedRaycastEncoder(nn.Module):
  """Encodes a sequence of raycast frames using temporal convolution.

  Processes each frame with a shared CNN encoder, then applies 1D convolution
  along the temporal dimension to capture motion patterns.

  Supports sparse frame sampling to efficiently capture both recent detail and
  long-term context. For example, with history_length=30 and frame_indices=(0,1,2,5,10,20,29),
  we sample 7 frames: 3 most recent (0.06s) + 4 sparse older frames spanning 0.6s.

  Architecture:
    1. Sparse frame selection: [B, T_full, H*W] -> [B, T_sparse, H*W]
    2. Per-frame CNN encoding: [B, T_sparse, H*W] -> [B, T_sparse, D_frame]
    3. 1D Conv temporal: [B, D_frame, T_sparse] -> [B, C_out, T']
    4. MLP fusion: [B, C_out * T'] -> [B, D_latent]
  """

  def __init__(
    self,
    input_dim: int,
    history_length: int = 3,
    frame_indices: tuple[int, ...] | None = None,
    frame_output_dim: int = 32,
    conv_out_channels: int = 16,
    conv_kernel_size: int = 2,
    latent_dim: int = 32,
    activation: str = "elu",
    grid_size: tuple[int, int] = (8, 8),
  ):
    """Initialize the raycast encoder.

    Args:
      input_dim: Dimension of single frame (H * W for raycast grid).
      history_length: Total number of frames in the history buffer.
      frame_indices: Indices of frames to sample from history (0=oldest, -1=newest).
                     If None, uses all frames. Example: (0, 1, 2, 5, 10, 20, 29)
                     means sample frames at those positions from a 30-frame buffer.
      frame_output_dim: Output dimension of per-frame CNN encoder.
      conv_out_channels: Number of output channels for temporal 1D convolution.
      conv_kernel_size: Kernel size for temporal 1D convolution.
      latent_dim: Final latent dimension after temporal fusion.
      activation: Activation function name.
      grid_size: Raycast grid dimensions (height, width).
    """
    super().__init__()
    self.input_dim = input_dim
    self.history_length = history_length
    self.latent_dim = latent_dim
    self.grid_size = grid_size

    # Sparse frame sampling: if frame_indices provided, only process those frames
    if frame_indices is not None:
      self.frame_indices = tuple(
        idx if idx >= 0 else history_length + idx for idx in frame_indices
      )
      self.num_sampled_frames = len(self.frame_indices)
    else:
      self.frame_indices = None
      self.num_sampled_frames = history_length

    act = get_activation(activation)

    # Per-frame CNN encoder (shared across all frames)
    self.frame_encoder = RaycastFrameEncoder(
      input_dim=input_dim,
      output_dim=frame_output_dim,
      activation=activation,
      grid_size=grid_size,
    )

    # 1D convolution for temporal aggregation
    # Input: [B, T_sampled, frame_output_dim] -> permute to [B, frame_output_dim, T_sampled]
    # Output: [B, conv_out_channels, T']
    # With kernel_size=k, stride=1: T' = T_sampled - k + 1
    conv_output_length = self.num_sampled_frames - conv_kernel_size + 1
    self.temporal_conv = nn.Sequential(
      nn.Conv1d(
        in_channels=frame_output_dim,
        out_channels=conv_out_channels,
        kernel_size=conv_kernel_size,
        stride=1,
      ),
      act,
    )

    # MLP to fuse temporal features into final latent
    fusion_input_dim = conv_out_channels * conv_output_length
    self.fusion_mlp = nn.Sequential(
      nn.Linear(fusion_input_dim, latent_dim),
      act,
    )

    if frame_indices is not None:
      print(
        f"StackedRaycastEncoder: grid={grid_size}, history={history_length}, "
        f"sparse_frames={self.frame_indices} ({self.num_sampled_frames} frames), "
        f"frame_out={frame_output_dim}, conv_out={conv_out_channels}, "
        f"conv_length={conv_output_length}, latent={latent_dim}"
      )
    else:
      print(
        f"StackedRaycastEncoder: grid={grid_size}, history={history_length}, "
        f"frame_out={frame_output_dim}, conv_out={conv_out_channels}, "
        f"conv_length={conv_output_length}, latent={latent_dim}"
      )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode stacked raycast history with optional sparse sampling.

    Args:
      x: Stacked raycast history of shape [batch_size, history_length * input_dim]
         (flattened) or [batch_size, history_length, input_dim]

    Returns:
      Encoded latent of shape [batch_size, latent_dim]
    """
    batch_size = x.shape[0]

    # Handle flattened input (from observation manager with flatten_history_dim=True)
    if x.dim() == 2:
      # Reshape: [B, T_full * D] -> [B, T_full, D]
      x = x.view(batch_size, self.history_length, self.input_dim)

    # Apply sparse frame sampling if configured
    if self.frame_indices is not None:
      # Select only the specified frames: [B, T_full, D] -> [B, T_sparse, D]
      x = x[:, self.frame_indices, :]

    # Encode each sampled frame independently: [B, T_sparse, D] -> [B*T_sparse, D]
    x_flat = x.reshape(batch_size * self.num_sampled_frames, self.input_dim)
    frame_features = self.frame_encoder(x_flat)
    # Reshape back: [B*T_sparse, D_frame] -> [B, T_sparse, D_frame]
    frame_features = frame_features.view(batch_size, self.num_sampled_frames, -1)

    # Temporal convolution: [B, T_sparse, D_frame] -> [B, D_frame, T_sparse] -> [B, C_out, T']
    frame_features = frame_features.permute(0, 2, 1)  # [B, D_frame, T_sparse]
    temporal_features = self.temporal_conv(frame_features)  # [B, C_out, T']

    # Flatten and fuse: [B, C_out, T'] -> [B, C_out * T'] -> [B, latent]
    temporal_features = temporal_features.flatten(1)
    latent = self.fusion_mlp(temporal_features)

    return latent


class ActorCriticRaycast(nn.Module):
  """Raycast-aware model compatible with rsl_rl >= 5.0 MLPModel interface.

  Each instance handles a single role ("actor" or "critic"). Observations are
  separated into:
  - raycast groups: Processed through a StackedRaycastEncoder (CNN + temporal conv)
  - base groups: Concatenated and optionally normalized

  The encoded raycast features are concatenated with base observations before
  passing through the role's MLP head.

  Supports multiple raycast observation groups (e.g. forward, left, right).
  When ``share_raycast_encoder=True`` (default), all groups are processed by
  a single shared CNN encoder.  When ``False``, each group gets its own
  independent encoder, allowing direction-specific feature extraction.
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: dict[str, torch.Tensor],
    obs_groups: dict[str, list[str] | tuple[str, ...]],
    obs_set: str,  # "actor" or "critic"
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    # Raycast encoder config
    raycast_obs_name: str | tuple[str, ...] = "raycast",
    raycast_history_length: int = 3,
    raycast_frame_indices: tuple[int, ...] | None = None,
    raycast_grid_size: tuple[int, int] = (8, 8),
    raycast_frame_output_dim: int = 32,
    raycast_conv_out_channels: int = 16,
    raycast_conv_kernel_size: int = 2,
    raycast_latent_dim: int = 32,
    share_raycast_encoder: bool = True,
    **kwargs,
  ):
    if kwargs:
      print(
        "ActorCriticRaycast.__init__ got unexpected arguments, which will be ignored: "
        + str(list(kwargs.keys()))
      )
    super().__init__()

    self.obs_set = obs_set
    self.obs_groups_all = obs_groups
    self.my_groups: list[str] | tuple[str, ...] = obs_groups[obs_set]

    # Normalize raycast_obs_name to a tuple for uniform handling
    if isinstance(raycast_obs_name, str):
      self.raycast_obs_names: tuple[str, ...] = (raycast_obs_name,)
    else:
      self.raycast_obs_names = tuple(raycast_obs_name)
    self.raycast_obs_names_set = set(self.raycast_obs_names)
    self.raycast_history_length = raycast_history_length

    # Determine which raycast groups are in this role's observation set
    self.my_raycast_names = [
      name for name in self.raycast_obs_names if name in self.my_groups
    ]
    self.has_raycast = len(self.my_raycast_names) > 0

    # Calculate observation dimensions
    total_obs_dim = 0
    total_raycast_dim = 0
    raycast_raw_dim = 0

    for obs_group in self.my_groups:
      obs_tensor = obs[obs_group]
      assert len(obs_tensor.shape) == 2, (
        f"Only 1D observations supported, got shape {obs_tensor.shape} for '{obs_group}'."
      )
      total_obs_dim += obs_tensor.shape[-1]
      if obs_group in self.raycast_obs_names_set:
        rc_dim = obs_tensor.shape[-1]
        rc_raw = rc_dim // raycast_history_length
        if raycast_raw_dim == 0:
          raycast_raw_dim = rc_raw
        else:
          assert rc_raw == raycast_raw_dim, (
            f"All raycast groups must have same raw dim, got {rc_raw} for '{obs_group}' "
            f"but expected {raycast_raw_dim}"
          )
        total_raycast_dim += rc_dim

    # Store per-group dimensions for ONNX export (flat tensor splitting).
    self._obs_group_dims: dict[str, int] = {}
    for obs_group in self.my_groups:
      self._obs_group_dims[obs_group] = obs[obs_group].shape[-1]

    base_obs_dim = total_obs_dim - total_raycast_dim
    self.obs_dim = total_obs_dim  # Required by rsl_rl interface

    # Observation normalization (only for base obs, not raycast)
    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(base_obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    # Raycast encoder (only if this role uses raycast observations)
    self.share_raycast_encoder = share_raycast_encoder
    if self.has_raycast:

      def _make_encoder() -> StackedRaycastEncoder:
        return StackedRaycastEncoder(
          input_dim=raycast_raw_dim,
          history_length=raycast_history_length,
          frame_indices=raycast_frame_indices,
          frame_output_dim=raycast_frame_output_dim,
          conv_out_channels=raycast_conv_out_channels,
          conv_kernel_size=raycast_conv_kernel_size,
          latent_dim=raycast_latent_dim,
          activation=activation,
          grid_size=raycast_grid_size,
        )

      if share_raycast_encoder:
        self.raycast_encoder = _make_encoder()
      else:
        self.raycast_encoders = nn.ModuleDict(
          {name: _make_encoder() for name in self.my_raycast_names}
        )
      total_raycast_latent = raycast_latent_dim * len(self.my_raycast_names)
      mlp_input_dim = base_obs_dim + total_raycast_latent
    else:
      mlp_input_dim = base_obs_dim

    # Distribution (actor only)
    if distribution_cfg is not None:
      from rsl_rl.utils import resolve_callable

      dist_class: type[Distribution] = resolve_callable(
        distribution_cfg.pop("class_name")
      )  # type: ignore[assignment]
      self.distribution: Distribution | None = dist_class(
        output_dim, **distribution_cfg
      )
      mlp_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    # MLP head
    self.mlp = MLP(mlp_input_dim, mlp_output_dim, list(hidden_dims), activation)

    # Initialize distribution-specific MLP weights
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

    print(
      f"ActorCriticRaycast [{obs_set}]: input={mlp_input_dim} "
      f"(base={base_obs_dim}"
      + (
        f" + raycast_latent={raycast_latent_dim}x{len(self.my_raycast_names)}"
        f" [{'shared' if share_raycast_encoder else 'independent'}]"
        if self.has_raycast
        else ""
      )
      + f"), output={output_dim}, MLP={self.mlp}"
    )

  def _split_obs(
    self, obs: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Split observations into base and raycast components."""
    base_parts: list[torch.Tensor] = []
    raycast_obs_list: list[torch.Tensor] = []

    for obs_group in self.my_groups:
      if obs_group in self.raycast_obs_names_set:
        raycast_obs_list.append(obs[obs_group])
      else:
        base_parts.append(obs[obs_group])

    base_obs = torch.cat(base_parts, dim=-1) if base_parts else torch.empty(0)
    return base_obs, raycast_obs_list

  def get_latent(
    self,
    obs: dict[str, torch.Tensor],
    masks: torch.Tensor | None = None,
    hidden_state: object = None,
  ) -> torch.Tensor:
    """Build model latent: normalized base obs + encoded raycast."""
    base_obs, raycast_obs_list = self._split_obs(obs)
    base_normalized = self.obs_normalizer(base_obs)

    if self.has_raycast and raycast_obs_list:
      if self.share_raycast_encoder:
        latents = [self.raycast_encoder(rc) for rc in raycast_obs_list]
      else:
        latents = [
          self.raycast_encoders[name](rc)
          for name, rc in zip(self.my_raycast_names, raycast_obs_list, strict=True)
        ]
      raycast_latent = torch.cat(latents, dim=-1)
      return torch.cat([base_normalized, raycast_latent], dim=-1)
    return base_normalized

  def forward(
    self,
    obs: dict[str, torch.Tensor],
    masks: torch.Tensor | None = None,
    hidden_state: object = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    """Forward pass compatible with rsl_rl MLPModel interface."""
    latent = self.get_latent(obs, masks, hidden_state)
    mlp_output = self.mlp(latent)

    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  # --- rsl_rl MLPModel interface ---

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: object = None
  ) -> None:
    pass

  def get_hidden_state(self) -> None:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    pass

  @property
  def output_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    assert self.distribution is not None
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
  ) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.kl_divergence(old_params, new_params)

  def update_normalization(self, obs: dict[str, torch.Tensor]) -> None:
    """Update observation-normalization statistics."""
    if self.obs_normalization:
      base_obs, _ = self._split_obs(obs)
      self.obs_normalizer.update(base_obs)  # type: ignore[union-attr]

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    """Return a version of the model compatible with ONNX export."""
    return _OnnxActorCriticRaycast(self, verbose)

  def _get_latent_dim(self) -> int:
    """Return the latent dimensionality consumed by the MLP head."""
    return self.mlp[0].in_features  # type: ignore[union-attr]


class _OnnxActorCriticRaycast(nn.Module):
  """ONNX-exportable wrapper for ActorCriticRaycast.

  Accepts a single flat observation tensor (all groups concatenated in order)
  and returns deterministic actions.
  """

  is_recurrent: bool = False

  def __init__(self, model: ActorCriticRaycast, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_groups = list(model.my_groups)
    self.obs_group_dims = dict(model._obs_group_dims)
    self.input_size = model.obs_dim
    self.raycast_obs_names_set = model.raycast_obs_names_set
    self.has_raycast = model.has_raycast
    self._share_raycast_encoder = model.share_raycast_encoder
    self._my_raycast_names = list(model.my_raycast_names)

    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    if self.has_raycast:
      if self._share_raycast_encoder:
        self.raycast_encoder = copy.deepcopy(model.raycast_encoder)
      else:
        self.raycast_encoders = copy.deepcopy(model.raycast_encoders)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Split flat tensor back into per-group slices.
    base_parts: list[torch.Tensor] = []
    raycast_obs_list: list[torch.Tensor] = []
    offset = 0
    for group in self.obs_groups:
      dim = self.obs_group_dims[group]
      obs_slice = x[:, offset : offset + dim]
      offset += dim
      if group in self.raycast_obs_names_set:
        raycast_obs_list.append(obs_slice)
      else:
        base_parts.append(obs_slice)

    base_obs = torch.cat(base_parts, dim=-1)
    base_normalized = self.obs_normalizer(base_obs)

    if self.has_raycast and raycast_obs_list:
      if self._share_raycast_encoder:
        latents = [self.raycast_encoder(rc) for rc in raycast_obs_list]
      else:
        latents = [
          self.raycast_encoders[name](rc)
          for name, rc in zip(self._my_raycast_names, raycast_obs_list, strict=True)
        ]
      raycast_latent = torch.cat(latents, dim=-1)
      combined = torch.cat([base_normalized, raycast_latent], dim=-1)
    else:
      combined = base_normalized

    out = self.mlp(combined)
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
