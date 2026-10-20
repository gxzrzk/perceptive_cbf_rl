#!/usr/bin/env python3
"""Train and test the steering local planner module.

This script trains a GRU-based local planner from global path planner
demonstrations using only local observations. The trained model can then be
deployed as a local navigation module.

The training process:
1. Load a pre-trained locomotion policy with path planning enabled
2. Run the policy in the environment and collect path planner decisions
3. Train the local planner to predict heading commands, goal positions, and velocity
4. The local planner learns to replace the global path planner with local sensing

Usage:
  # Train the local planner (requires a trained locomotion policy)
  python -m mjlab.tasks.steering.train_local_planner --mode train \\
    --task Mjlab-StraightApproach-LowSpeed-G1-v6 \\
    --checkpoint /path/to/policy.pt

  # Test the local planner with steering correction
  python -m mjlab.tasks.steering.train_local_planner --mode test \\
    --task Mjlab-StraightApproach-LowSpeed-G1-v6 \\
    --checkpoint /path/to/policy.pt \\
    --local_planner_checkpoint /path/to/local_planner.pt

  # Play mode (visualization without local planner)
  python -m mjlab.tasks.steering.train_local_planner --mode play \\
    --task Mjlab-StraightApproach-LowSpeed-G1-v6

  # Diagnose locomotion angular velocity at high speed
  python -m mjlab.tasks.steering.train_local_planner --mode diagnose \\
    --task Mjlab-CircularGoal-Flat-G1Rw-Pure-v6 \\
    --checkpoint /path/to/policy.pt \\
    --viewer native --num_envs 1

Example:
  uv run python -m mjlab.tasks.steering.train_local_planner --mode train \\
    --task Mjlab-StraightApproach-LowSpeed-G1-v6 \\
    --num_envs 1024 \\
    --checkpoint logs/rsl_rl/g1_velocity_raycast_h1/2026-02-04_01-07-23/model_14998.pt
"""

from __future__ import annotations

import os
import signal
import threading
import time
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TypedDict, cast

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.raycast_occlusion import disable_raycast_near_occlusion
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.steering.local_planner_checkpoint_compat import (
  load_local_planner_checkpoint,
)
from mjlab.tasks.steering.local_planner_inference import (
  LocalPlannerInference,
  LocalPlannerInferenceCfg,
)
from mjlab.tasks.steering.local_planner_network import (
  LocalPlannerNetwork,
  LocalPlannerNetworkCfg,
)
from mjlab.tasks.steering.local_planner_trainer import (
  LocalPlannerTrainer,
  LocalPlannerTrainerCfg,
)
from mjlab.tasks.velocity.mdp.reachable_goal_command import (
  FixedSequentialGoalCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer.debug_visualizer import DebugVisualizer


def _dict_like_keys(value: object) -> Iterable[object] | None:
  keys_fn = getattr(value, "keys", None)
  if not callable(keys_fn):
    return None
  keys = keys_fn()
  return keys if isinstance(keys, Iterable) else None


def _dict_like_has_key(value: object, key: str) -> bool:
  keys = _dict_like_keys(value)
  return keys is not None and key in keys


def _dict_like_get(value: object, key: object) -> Any:
  getitem_fn = getattr(value, "__getitem__", None)
  if not callable(getitem_fn):
    raise TypeError(f"Object of type {type(value).__name__} is not indexable.")
  return getitem_fn(key)


def _dict_like_set(value: object, key: object, item: object) -> None:
  setitem_fn = getattr(value, "__setitem__", None)
  if not callable(setitem_fn):
    raise TypeError(
      f"Object of type {type(value).__name__} does not support item assignment."
    )
  setitem_fn(key, item)


class LocalPlannerOverlayState(TypedDict):
  original_yaw: float
  raw_correction: float
  smoothed_correction: float
  correction: float
  corrected_yaw: float
  hidden_norm: float
  step: int
  true_goal_x: float
  true_goal_y: float
  target_lin_vel: float
  true_lin_vel: float
  pred_lin_vel: float
  dr_x: float
  dr_y: float
  dr_heading_deg: float
  initial_goal_x: float
  initial_goal_y: float
  dr_est_goal_x: float
  dr_est_goal_y: float
  goal_bearing_deg: float
  goal_distance: float
  true_bearing_deg: float
  true_distance: float
  goal_seek_gate: float
  goal_seek_yaw: float
  avoidance_yaw: float
  current_key_goal: str
  goal_dir_dx: float
  goal_dir_dy: float
  goal_dir_cos: float
  goal_dir_sin: float
  true_dir_dx: float
  true_dir_dy: float
  true_dir_cos: float
  true_dir_sin: float
  hidden_decay: float


class _TeacherLabelPolicyProtocol(Protocol):
  def get_last_target_yaw(self) -> torch.Tensor | None: ...


@dataclass(frozen=True)
class FixedScenePreset:
  """Deterministic low-speed screenshot preset for paper figures."""

  terrain_seed: int
  terrain_level: int
  terrain_type: int
  spawn_x: float
  spawn_y: float
  spawn_yaw: float
  target_velocity: float


_FIXED_SCENE_PRESETS: dict[str, FixedScenePreset] = {
  "paper_low_speed_v1": FixedScenePreset(
    terrain_seed=1234,
    terrain_level=2,
    terrain_type=0,
    spawn_x=0.0,
    spawn_y=0.0,
    spawn_yaw=1.5707963267948966,
    target_velocity=0.8,
  )
}


def _get_fixed_scene_preset(name: str | None) -> FixedScenePreset | None:
  if name is None:
    return None
  preset = _FIXED_SCENE_PRESETS.get(name)
  if preset is None:
    raise ValueError(
      f"Unknown --fixed-scene preset '{name}'. "
      f"Available: {sorted(_FIXED_SCENE_PRESETS)}"
    )
  return preset


def _configure_fixed_scene(
  env_cfg: ManagerBasedRlEnvCfg,
  preset: FixedScenePreset,
) -> None:
  """Freeze config-level reset randomness before environment creation."""
  scene = getattr(env_cfg, "scene", None)
  if scene is None or getattr(scene, "num_envs", None) != 1:
    raise ValueError("Fixed-scene presets currently require --num-envs 1.")

  terrain_cfg = getattr(scene, "terrain", None)
  terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
  if terrain_cfg is None or terrain_generator is None:
    raise ValueError("Fixed-scene preset requires generator terrain.")
  terrain_generator.seed = preset.terrain_seed
  terrain_cfg.max_init_terrain_level = preset.terrain_level

  if getattr(env_cfg, "curriculum", None) is not None:
    env_cfg.curriculum.pop("terrain_levels", None)
    env_cfg.curriculum.pop("goal_velocity", None)
    env_cfg.curriculum.pop("command_vel", None)

  if getattr(env_cfg, "events", None) is not None:
    env_cfg.events.pop("randomize_terrain", None)
    reset_base_cfg = env_cfg.events.get("reset_base")
    if reset_base_cfg is not None and reset_base_cfg.params is not None:
      pose_range = reset_base_cfg.params.get("pose_range")
      if isinstance(pose_range, dict):
        pose_range["x"] = (preset.spawn_x, preset.spawn_x)
        pose_range["y"] = (preset.spawn_y, preset.spawn_y)
        pose_range["yaw"] = (preset.spawn_yaw, preset.spawn_yaw)

  commands = getattr(env_cfg, "commands", None)
  if commands is None:
    return
  goal_cmd = commands.get("goal")
  if not isinstance(goal_cmd, FixedSequentialGoalCommandCfg):
    return

  goal_cmd.target_velocity = preset.target_velocity
  goal_cmd.target_velocity_range = (preset.target_velocity, preset.target_velocity)
  if goal_cmd.goal_offset_noise is not None:
    goal_cmd.goal_offset_noise = [(0.0, 0.0)] * len(goal_cmd.goal_offset_noise)
  goal_cmd.mirror_x_probability = 0.0
  goal_cmd.near_zero_probability = 0.0


def _ensure_fixed_scene_after_reset(
  env: ManagerBasedRlEnv,
  preset: FixedScenePreset,
) -> None:
  """Re-apply the pinned terrain cell and reset once so commands use it."""
  _apply_fixed_scene_runtime(env, preset)
  env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
  env.reset(env_ids=env_ids)
  _apply_fixed_scene_runtime(env, preset)


def _install_fixed_scene_reset_hook(
  env: ManagerBasedRlEnv,
  preset: FixedScenePreset,
) -> None:
  """Keep viewer/test resets on the same terrain cell."""
  original_reset = env.reset

  def _reset_with_fixed_scene(
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[dict[str, torch.Tensor | dict[str, torch.Tensor]], dict]:
    obs, extras = original_reset(seed=seed, env_ids=env_ids, options=options)
    if env_ids is None:
      pinned_env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    else:
      pinned_env_ids = env_ids
    if len(pinned_env_ids) > 0:
      _apply_fixed_scene_runtime(env, preset)
      obs, extras = original_reset(env_ids=pinned_env_ids)
      _apply_fixed_scene_runtime(env, preset)
    return obs, extras

  env.reset = _reset_with_fixed_scene  # type: ignore[method-assign]


def _apply_fixed_scene_runtime(
  env: ManagerBasedRlEnv,
  preset: FixedScenePreset,
) -> None:
  """Pin terrain cell and origins after environment construction."""
  if env.num_envs != 1:
    raise ValueError("Fixed-scene presets currently require a single environment.")

  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    raise ValueError("Fixed-scene preset requires terrain origins.")

  env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
  terrain.terrain_levels[env_ids] = preset.terrain_level
  terrain.terrain_types[env_ids] = preset.terrain_type
  terrain.env_origins[env_ids] = terrain.terrain_origins[
    preset.terrain_level, preset.terrain_type
  ]
  env.scene.env_origins[env_ids] = terrain.env_origins[env_ids]


def _robot_yaw_from_quat(robot_quat: torch.Tensor) -> torch.Tensor:
  return torch.atan2(
    2.0 * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
    1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
  )


def _compute_overlay_target_yaw(
  correction_extractor: "PathPlannerCorrectionExtractor",
  obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
) -> torch.Tensor:
  done = torch.zeros(
    correction_extractor.num_envs, dtype=torch.bool, device=correction_extractor.device
  )
  target, _ = correction_extractor.extract_correction(done, obs_dict=obs_dict)
  return target[:, 0]


def _flatten_tensor_observations(obs: object) -> dict[str, torch.Tensor]:
  """Collect tensor observations from nested dict/TensorDict structures."""
  flat_obs: dict[str, torch.Tensor] = {}

  def _collect(value: object, key: str | None = None) -> None:
    if isinstance(value, torch.Tensor):
      if key is not None:
        flat_obs[key] = value
      return
    value_keys = _dict_like_keys(value)
    if value_keys is not None:
      for sub_key in value_keys:
        _collect(_dict_like_get(value, sub_key), str(sub_key))

  _collect(obs)
  if not flat_obs and isinstance(obs, torch.Tensor):
    flat_obs["obs"] = obs
  return flat_obs


def _format_raycast_debug_stats(
  obs_dict: Mapping[str, torch.Tensor],
  raycast_obs_names: tuple[str, ...],
  raycast_dim: int,
) -> str:
  """Summarize the raycast tensors that will be fed to the local planner."""
  parts: list[str] = []
  for obs_name in raycast_obs_names:
    ray = obs_dict.get(obs_name)
    if ray is None:
      parts.append(f"{obs_name}=missing")
      continue
    if ray.shape[-1] > raycast_dim:
      ray = ray[..., -raycast_dim:]
    ray = ray.detach()
    all_ones = torch.allclose(ray, torch.ones_like(ray), atol=1.0e-6, rtol=0.0)
    parts.append(
      f"{obs_name}=min/mean/max "
      f"{ray.min().item():.3f}/{ray.mean().item():.3f}/{ray.max().item():.3f} "
      f"all1={all_ones}"
    )
  return " | ".join(parts)


def _find_goal_command_offset(env: ManagerBasedRlEnv) -> int | None:
  """Find the actor-observation offset of the goal command term."""
  try:
    obs_manager = env.observation_manager
    term_names = obs_manager.active_terms.get("actor", [])
    term_dims = obs_manager.group_obs_term_dim.get("actor", [])

    offset = 0
    for term_name, dims in zip(term_names, term_dims, strict=False):
      if term_name == "goal_command":
        return offset
      offset += int(np.prod(dims))

    print(f"[WARN] goal_command not found in terms: {term_names}")
    return 93
  except Exception as e:
    print(f"[WARN] Could not determine goal_command offset: {e}")
    return 93


def _make_label_overlay_debug_vis(
  env: ManagerBasedRlEnv,
  correction_extractor: "PathPlannerCorrectionExtractor | None" = None,
  target_yaw_getter: Callable[[], torch.Tensor | None] | None = None,
) -> Callable[[DebugVisualizer], None]:
  if correction_extractor is None and target_yaw_getter is None:
    raise ValueError(
      "Label overlay debug visualization requires a correction extractor "
      "or a target_yaw_getter."
    )

  def _debug_vis(visualizer: DebugVisualizer) -> None:
    if target_yaw_getter is not None:
      target_yaw = target_yaw_getter()
      if target_yaw is None:
        return
    else:
      assert correction_extractor is not None
      obs_dict = env.observation_manager.compute(update_history=False)
      target_yaw = _compute_overlay_target_yaw(correction_extractor, obs_dict)
    robot = env.scene["robot"]
    robot_pos = robot.data.root_link_pos_w[:, :3]
    robot_quat = robot.data.root_link_quat_w
    robot_yaw = _robot_yaw_from_quat(robot_quat)
    arrow_yaw = robot_yaw + target_yaw
    arrow_length = 1.25

    for env_idx in visualizer.get_env_indices(env.num_envs):
      start = robot_pos[env_idx].detach().cpu().numpy().copy()
      start[2] += 0.55
      yaw = float(arrow_yaw[env_idx].item())
      direction = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
      end = start + arrow_length * direction
      visualizer.add_arrow(
        start,
        end,
        color=(0.9, 0.25, 0.2, 0.95),
        width=0.02,
        label=f"label_target_{env_idx}",
      )

  return _debug_vis


def _attach_debug_vis(
  env: ManagerBasedRlEnv,
  extra_debug_vis: Callable[[DebugVisualizer], None],
) -> None:
  original_update_visualizers = env.update_visualizers

  def _combined_update_visualizers(visualizer: DebugVisualizer) -> None:
    original_update_visualizers(visualizer)
    extra_debug_vis(visualizer)

  env.update_visualizers = _combined_update_visualizers  # type: ignore[method-assign]


def _load_local_planner_checkpoint_safe(
  network: torch.nn.Module,
  checkpoint_state_dict: dict[str, torch.Tensor],
) -> None:
  """Load a local planner checkpoint with tolerance for shape mismatches.

  When the network architecture changes (e.g., output_dim 4 → 5), the final
  layer weights will have different shapes.  ``strict=False`` in
  ``load_state_dict`` only handles *missing* / *unexpected* keys – it still
  raises on size mismatches for the **same** key.

  This helper filters out size-mismatched keys, partially copies matching
  dimensions, and loads the rest normally.
  """
  model_sd = network.state_dict()
  filtered_sd: dict[str, torch.Tensor] = {}
  partial_keys: list[str] = []

  for key, ckpt_val in checkpoint_state_dict.items():
    if key not in model_sd:
      continue  # unexpected key – will be reported by strict=False
    model_val = model_sd[key]
    if ckpt_val.shape == model_val.shape:
      filtered_sd[key] = ckpt_val
    else:
      # Partial copy: copy the overlapping slice, keep random init for the rest
      partial_keys.append(
        f"{key}: ckpt {list(ckpt_val.shape)} → model {list(model_val.shape)}"
      )
      slices = tuple(
        slice(0, min(c, m))
        for c, m in zip(ckpt_val.shape, model_val.shape, strict=False)
      )
      model_val[slices] = ckpt_val[slices]
      filtered_sd[key] = model_val

  missing, unexpected = network.load_state_dict(filtered_sd, strict=False)
  if partial_keys:
    print(
      "[WARN] Partially loaded keys with shape mismatch (new dims use random init):"
    )
    for k in partial_keys:
      print(f"       {k}")
  if missing:
    print(f"[WARN] Missing keys (new layers use random init): {missing}")
  if unexpected:
    print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")


def _default_local_planner_network_cfg(
  raycast_obs_names: tuple[str, ...],
  *,
  use_odometry: bool = False,
  include_joint_vel: bool = True,
) -> LocalPlannerNetworkCfg:
  """Build a network config from CLI flags when a checkpoint has no config."""
  net_cfg = LocalPlannerNetworkCfg()
  net_cfg.num_raycast_groups = len(raycast_obs_names)
  if use_odometry:
    net_cfg.output_dim = 1
    net_cfg.odom_vel_dim = 2
    if not include_joint_vel:
      net_cfg.joint_vel_dim = 0
  if len(raycast_obs_names) > 1:
    net_cfg.use_independent_raycast_cnn = True
    net_cfg.hidden_dims = (128, 96)
    net_cfg.state_dim = 64
    net_cfg.raycast_cnn_output_dim = 48
  return net_cfg


def _network_cfg_from_checkpoint_config(
  cfg_obj: object | None,
) -> LocalPlannerNetworkCfg | None:
  """Recover the exact local-planner network config saved in a checkpoint."""
  if cfg_obj is None:
    return None

  if isinstance(cfg_obj, dict) and "network_cfg" in cfg_obj:
    raw_cfg = cfg_obj["network_cfg"]
  else:
    raw_cfg = getattr(cfg_obj, "network_cfg", cfg_obj)
  if isinstance(raw_cfg, LocalPlannerNetworkCfg):
    return deepcopy(raw_cfg)

  if is_dataclass(raw_cfg) and not isinstance(raw_cfg, type):
    raw_cfg = asdict(cast(Any, raw_cfg))

  if isinstance(raw_cfg, dict):
    allowed = {f.name for f in fields(LocalPlannerNetworkCfg)}
    return LocalPlannerNetworkCfg(**{k: v for k, v in raw_cfg.items() if k in allowed})

  return None


def _trainer_cfg_value(
  cfg_obj: object | None,
  name: str,
  default: Any,
) -> Any:
  if cfg_obj is None:
    return default
  if isinstance(cfg_obj, dict):
    return cfg_obj.get(name, default)
  return getattr(cfg_obj, name, default)


def _apply_trainer_cfg_to_network_cfg(
  net_cfg: LocalPlannerNetworkCfg,
  trainer_cfg: object | None,
) -> None:
  """Backfill network odometry fields from older trainer configs."""
  if not bool(_trainer_cfg_value(trainer_cfg, "use_odometry", False)):
    return

  net_cfg.output_dim = 1
  net_cfg.odom_vel_dim = int(_trainer_cfg_value(trainer_cfg, "odom_vel_dim", 2))
  include_joint_vel = bool(_trainer_cfg_value(trainer_cfg, "include_joint_vel", True))
  if not include_joint_vel:
    net_cfg.joint_vel_dim = 0


def _load_local_planner_network_for_inference(
  checkpoint_path: str,
  *,
  raycast_obs_names: tuple[str, ...],
  device: torch.device | str,
  use_odometry: bool = False,
  include_joint_vel: bool = True,
) -> tuple[LocalPlannerNetwork, object | None]:
  """Load a local planner using the checkpoint's own architecture config.

  Falling back to CLI-derived dimensions is only safe for legacy checkpoints
  that did not save a config.  For modern checkpoints, especially odometry
  checkpoints, the saved config is the source of truth; otherwise a 1D odom
  network can be partially loaded into a 2D DR network with shifted inputs.
  """
  checkpoint = load_local_planner_checkpoint(checkpoint_path, map_location=device)
  if isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
    state_dict = checkpoint["network_state_dict"]
    trainer_cfg = checkpoint.get("config", None)
  elif isinstance(checkpoint, dict):
    state_dict = checkpoint
    trainer_cfg = None
  else:
    raise ValueError(f"Unsupported local planner checkpoint type: {type(checkpoint)}")

  net_cfg = _network_cfg_from_checkpoint_config(trainer_cfg)
  if net_cfg is None:
    net_cfg = _default_local_planner_network_cfg(
      raycast_obs_names,
      use_odometry=use_odometry,
      include_joint_vel=include_joint_vel,
    )
  else:
    _apply_trainer_cfg_to_network_cfg(net_cfg, trainer_cfg)

  if net_cfg.num_raycast_groups != len(raycast_obs_names):
    has_saved_cfg = trainer_cfg is not None
    if has_saved_cfg:
      raise ValueError(
        "Local planner checkpoint expects "
        f"{net_cfg.num_raycast_groups} raycast observation group(s), but "
        f"{len(raycast_obs_names)} name(s) were provided: {raycast_obs_names}. "
        "Pass the raycast names used during training."
      )
    net_cfg.num_raycast_groups = len(raycast_obs_names)

  network = LocalPlannerNetwork(net_cfg)
  _load_local_planner_checkpoint_safe(network, state_dict)
  network.to(device)
  network.eval()
  return network, trainer_cfg


def _apply_checkpoint_cfg_to_inference_cfg(
  inference_cfg: LocalPlannerInferenceCfg,
  checkpoint_cfg: object | None,
  args: TrainLocalPlannerArgs,
) -> None:
  """Make runtime observation construction match the loaded checkpoint."""
  ckpt_use_odometry = _trainer_cfg_value(checkpoint_cfg, "use_odometry", None)
  if ckpt_use_odometry is not None and bool(ckpt_use_odometry) != args.use_odometry:
    print(
      "[WARN] CLI --use-odometry does not match local planner checkpoint config; "
      f"using checkpoint value use_odometry={bool(ckpt_use_odometry)}."
    )

  use_odometry = (
    bool(ckpt_use_odometry) if ckpt_use_odometry is not None else args.use_odometry
  )
  include_joint_vel = bool(
    _trainer_cfg_value(checkpoint_cfg, "include_joint_vel", args.include_joint_vel)
  )

  inference_cfg.raycast_obs_names = args.raycast_obs_names
  inference_cfg.use_odometry = use_odometry
  inference_cfg.include_joint_vel = include_joint_vel
  if use_odometry:
    inference_cfg.odom_vel_dim = int(
      _trainer_cfg_value(checkpoint_cfg, "odom_vel_dim", 2)
    )
  else:
    inference_cfg.odom_vel_dim = 0
  inference_cfg.raycast_dim = int(
    _trainer_cfg_value(checkpoint_cfg, "raycast_dim", inference_cfg.raycast_dim)
  )
  inference_cfg.heading_dim = int(
    _trainer_cfg_value(checkpoint_cfg, "heading_dim", inference_cfg.heading_dim)
  )
  inference_cfg.joint_vel_dim = int(
    _trainer_cfg_value(checkpoint_cfg, "joint_vel_dim", inference_cfg.joint_vel_dim)
  )
  inference_cfg.goal_direction_dim = int(
    _trainer_cfg_value(
      checkpoint_cfg,
      "goal_direction_dim",
      inference_cfg.goal_direction_dim,
    )
  )
  inference_cfg.decision_dt = float(
    _trainer_cfg_value(checkpoint_cfg, "decision_dt", inference_cfg.decision_dt)
  )
  inference_cfg.odom_bearing_noise_std = float(
    _trainer_cfg_value(
      checkpoint_cfg,
      "odom_bearing_noise_std",
      inference_cfg.odom_bearing_noise_std,
    )
  )
  inference_cfg.odom_distance_noise_frac = float(
    _trainer_cfg_value(
      checkpoint_cfg,
      "odom_distance_noise_frac",
      inference_cfg.odom_distance_noise_frac,
    )
  )
  inference_cfg.odom_distance_noise_min = float(
    _trainer_cfg_value(
      checkpoint_cfg,
      "odom_distance_noise_min",
      inference_cfg.odom_distance_noise_min,
    )
  )

  inference_cfg.enable_odom_goal_noise = args.enable_odom_goal_noise
  if args.odom_bearing_noise_std is not None:
    inference_cfg.odom_bearing_noise_std = args.odom_bearing_noise_std
  if args.odom_distance_noise_frac is not None:
    inference_cfg.odom_distance_noise_frac = args.odom_distance_noise_frac
  if args.odom_distance_noise_min is not None:
    inference_cfg.odom_distance_noise_min = args.odom_distance_noise_min


def _goal_pos_w_from_term(term: object) -> torch.Tensor | None:
  """Get the navigation target position in world frame from a command term.

  Prefers key_goal_pos_w (the final/key goal) over goal_pos_w (current waypoint)
  so that the overlay's 'True Bearing' matches the reference point used by the
  network input (which is trained with bearing to the key goal).

  Returns:
    [num_envs, 3] world-frame position, or None if unavailable.
  """
  if hasattr(term, "key_goal_pos_w"):
    return term.key_goal_pos_w  # type: ignore[union-attr]
  if hasattr(term, "goal_pos_w"):
    return term.goal_pos_w  # type: ignore[union-attr]
  return None


@dataclass
class TrainLocalPlannerArgs:
  """Arguments for local planner training and testing."""

  # Mode
  mode: Literal["train", "test", "play", "diagnose"] = "train"
  """Mode: train (train local planner), test (test with steering correction),
  play (visualization only), diagnose (keyboard-controlled locomotion diagnostic)."""

  # Task configuration
  task: str = "Mjlab-StraightApproach-LowSpeed-G1-v6"
  """Task name to use for training/testing."""

  num_envs: int = 1024
  """Number of parallel environments."""

  seed: int | None = None
  """Optional random seed for reproducible environment construction/evaluation."""

  # Checkpoints
  checkpoint: str = ""
  """Path to trained locomotion policy checkpoint."""

  local_planner_checkpoint: str = ""
  """Path to a trained local planner checkpoint (for test mode)."""

  # Training configuration
  max_iterations: int = 50000
  """Maximum training iterations (environment steps)."""

  save_interval: int = 5000
  """Interval for saving checkpoints (in steps)."""

  log_interval: int = 500
  """Interval for logging statistics."""

  # Output
  output_dir: str = "logs/steering"
  """Directory to save trained models."""

  experiment_name: str = "steering"
  """Experiment name for logging."""

  # WandB logging
  use_wandb: bool = True
  """Enable WandB logging."""

  wandb_project: str = "mjlab-steering"
  """WandB project name."""

  wandb_entity: str | None = None
  """WandB entity (team/user name). If None, uses default."""

  # Viewer
  viewer: Literal["native", "viser", "none"] = "none"
  """Viewer type: native (mujoco), viser (web), or none."""

  fixed_scene: str | None = None
  """Optional fixed-scene screenshot preset. Currently supports paper_low_speed_v1."""

  show_label_target_overlay: bool = False
  """In play mode, show the current teacher label target as a debug arrow."""

  # Video recording
  video: bool = False
  """Record offscreen videos during training."""

  video_length: int = 200
  """Number of frames per training video."""

  video_interval: int = 2000
  """Start a new training video every N environment steps."""

  video_height: int | None = None
  """Optional video frame height override."""

  video_width: int | None = None
  """Optional video frame width override."""

  # Device
  device: str = "cuda"
  """Device to use (cuda or cpu)."""

  # Test mode parameters
  correction_gain: float = 1.0
  """Gain factor for local planner correction in test mode."""

  # --- DR sensor source (test/play mode) ---
  use_measured_velocity: bool = False
  """Use measured forward velocity from state estimator instead of network
  prediction for dead-reckoning.  Default False = use pred_lin_vel."""

  use_ahrs_heading: bool = True
  """Use AHRS (IMU → yaw) for DR heading instead of integrating angular
  velocity.  Eliminates heading drift during turns."""

  use_projected_yaw_rate: bool = False
  """Project body-frame angular velocity to world yaw rate using IMU
  quaternion.  Corrects heading integration when robot pitches/rolls."""

  # Ablation controls
  target_label_mode: Literal["dynamic", "waypoint_only", "key_goal_only"] = "dynamic"
  """Training target mode for navigation ablations.
  dynamic blends waypoint and key-goal labels from raycast clearance;
  waypoint_only always imitates the waypoint teacher;
  key_goal_only always points to the active key goal."""

  apply_teacher_label_to_policy: bool = True
  """In train mode, inject the current teacher yaw label into the locomotion
  policy observation so target_label_mode also controls data-collection motion."""

  disable_teacher_forcing: bool = False
  """Disable GT goal-direction inputs during training."""

  disable_dr_curriculum: bool = False
  """Disable the two-phase dead-reckoning curriculum schedule."""

  # --- Goal-seeking gate tuning (test/play mode) ---
  smoothing_alpha: float | None = None
  """Override local planner smoothing factor (0=no smoothing, 1=max). Lowering this
  reduces lag when switching between avoidance and goal-seeking.
  If None, uses default (0.3)."""

  goal_seek_gain: float | None = None
  """Override P-controller gain for goal-seeking.  Increase to turn toward
  goal more aggressively.  If None, uses default (0.8)."""

  gate_clearance_threshold: float | None = None
  """Override raycast clearance threshold (m).  Lower to switch to
  goal-seeking sooner after passing obstacles.  If None, uses default (3.8)."""

  min_goal_bias: float = 0.0
  """Minimum goal-seeking component added even when gate=1 (fully avoidance).
  E.g. 0.15 means final = gate*avoidance + (1-gate)*goal + 0.15*goal.
  This prevents the robot from getting stuck in pure avoidance mode."""

  # Raycast observation configuration
  raycast_obs_names: tuple[str, ...] = ("raycast",)
  """Observation key names for raycast groups.
  Single camera (n1): ("raycast",)
  Three cameras (n3): ("raycast", "raycast_left", "raycast_right")"""

  # TTC-based alpha blending for training
  use_ttc_blending: bool = False
  """Use Time-To-Collision alpha blending instead of fixed distance threshold.
  Enable for high-speed tasks where the raycast detection window is too short
  relative to the robot's stopping/turning distance."""

  ttc_threshold: float = 2.0
  """TTC threshold (seconds) for alpha blending transition centre.
  Only used when use_ttc_blending=True."""

  ttc_scale: float = 3.0
  """Sigmoid steepness for TTC blending.  Only used when
  use_ttc_blending=True."""

  # Lookahead teacher target for training
  use_lookahead_teacher: bool = False
  """Enable lookahead teacher target for high-speed training.
  When the current waypoint is roughly straight ahead (|yaw_to_waypoint|
  < lookahead_yaw_threshold), replace yaw_to_waypoint with yaw to the
  NEXT waypoint. This gives the network early warning to start turning
  before the path planner advances the waypoint, recovering ~0.25s of
  reaction time at 12Hz raycast.
  Only affects training targets, not task waypoints or inference."""

  lookahead_yaw_threshold: float = 0.3
  """Yaw threshold (rad, ~17 deg) below which current waypoint is
  considered 'straight ahead' and lookahead is activated."""

  # Local planner trainer overrides
  sequence_length: int | None = None
  """Override trainer sequence_length (BPTT window). If None, uses
  default from LocalPlannerTrainerCfg (32). For high-speed tasks, 96 gives the
  network enough temporal context to learn the full detect→turn sequence
  across multiple 12Hz raycast frames."""

  # Safety velocity envelope
  enable_safety_envelope: bool = False
  """Enable the time-to-stop safety velocity envelope on the goal command.
  When True, the command's speed is clipped based on front/side raycast
  clearance so the robot brakes before hitting obstacles.  Applies the
  same parameters used in Stage 4 SafeStop training."""

  disable_raycast_near_occlusion: bool = False
  """Disable task-configured near-field raycast occlusion randomization."""

  # --- Odometry mode ---
  use_odometry: bool = False
  """Enable odometry mode for the local planner.
  When True:
  - Network output is 1D (delta_yaw only, no velocity prediction)
  - Dead reckoning is disabled; goal direction computed from odom position
  - Joint velocities excluded from observations by default
  - Curriculum learning disabled; loss weights adjusted for obstacle focus"""

  include_joint_vel: bool = True
  """Include joint velocities in observations.  Automatically set to False
  in odometry mode unless explicitly overridden."""

  enable_odom_goal_noise: bool = True
  """In test/eval odometry mode, add small zero-mean noise to the odom-derived
  goal bearing/distance so the goal input fluctuates around simulator truth."""

  odom_bearing_noise_std: float | None = None
  """Override test/eval odom bearing noise std in radians.  If None, inherits
  the local-planner checkpoint config, falling back to 0.06."""

  odom_distance_noise_frac: float | None = None
  """Override test/eval odom distance noise std as a fraction of distance.  If
  None, inherits the checkpoint config, falling back to 0.03."""

  odom_distance_noise_min: float | None = None
  """Override test/eval odom distance noise floor in metres.  If None, uses
  the inference default, currently 0.05."""

  # Local planner configuration
  local_planner_cfg: LocalPlannerTrainerCfg = field(
    default_factory=LocalPlannerTrainerCfg
  )
  """Configuration for the local planner trainer."""


def setup_environment(args: TrainLocalPlannerArgs) -> tuple[ManagerBasedRlEnv, object]:
  """Setup the training/testing environment.

  Args:
    args: Training arguments.

  Returns:
    Tuple of (environment, rl_config).
  """
  # Load configurations
  # Use play config for both 'play' and 'test' modes to avoid robot position reset
  play = args.mode in ("play", "test")
  env_cfg = load_env_cfg(args.task, play=play)
  rl_cfg = load_rl_cfg(args.task)

  # Override number of environments
  env_cfg.scene.num_envs = args.num_envs
  if args.video_height is not None:
    env_cfg.viewer.height = args.video_height
  if args.video_width is not None:
    env_cfg.viewer.width = args.video_width
  if args.seed is not None:
    env_cfg.seed = args.seed
  fixed_scene_preset = _get_fixed_scene_preset(args.fixed_scene)
  if fixed_scene_preset is not None:
    _configure_fixed_scene(env_cfg, fixed_scene_preset)

  if args.disable_raycast_near_occlusion:
    disabled_groups = disable_raycast_near_occlusion(env_cfg)
    if disabled_groups:
      disabled_names = ", ".join(disabled_groups)
      print(
        "[INFO] Disabled raycast near-occlusion DR for "
        f"observation groups: {disabled_names}"
      )
    else:
      print(
        "[WARN] --disable-raycast-near-occlusion was requested, "
        "but no task-configured near-occlusion params were found."
      )

  # Optionally enable safety velocity envelope on the goal command.
  if args.enable_safety_envelope and env_cfg.commands is not None:
    from mjlab.tasks.velocity.mdp.reachable_goal_command import (
      FixedSequentialGoalCommandCfg,
      ReachableGoalCommandCfg,
    )

    goal_cmd = env_cfg.commands.get("goal")
    if isinstance(goal_cmd, (FixedSequentialGoalCommandCfg, ReachableGoalCommandCfg)):
      goal_cmd.safety_velocity_envelope = (
        ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
          enabled=True,
          forward_sensor_name="raycast",
          left_sensor_name="raycast_left",
          right_sensor_name="raycast_right",
          skip_ground_cols=3,
          safety_margin_m=0.25,
          reaction_time_s=0.15,
          braking_time_s=0.8,
          stop_velocity_epsilon=0.15,
          side_point_forward_offset_m=0.35,
          hard_stop_distance_m=0.40,
          blocked_turn_scale=0.0,
        )
      )
      print("[INFO] Safety velocity envelope enabled on goal command.")

  # Create environment
  device = args.device
  if args.mode == "train" and args.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
  render_mode = "rgb_array" if args.mode == "train" and args.video else None
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode=render_mode)
  if fixed_scene_preset is not None:
    _ensure_fixed_scene_after_reset(env, fixed_scene_preset)
    _install_fixed_scene_reset_hook(env, fixed_scene_preset)

  return env, rl_cfg


def load_policy(
  checkpoint: str,
  env: ManagerBasedRlEnv,
  rl_cfg: object,
  device: torch.device,
) -> Callable[..., torch.Tensor]:
  """Load trained locomotion policy.

  Args:
    checkpoint: Path to checkpoint.
    env: The environment.
    rl_cfg: RL configuration.
    device: Device to load to.

  Returns:
    Policy callable that takes observations and returns actions.
  """
  from dataclasses import asdict, is_dataclass

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.runner import MjlabOnPolicyRunner

  if not checkpoint:
    raise ValueError("Checkpoint path required for policy loading.")

  checkpoint_path = Path(checkpoint)

  # Try loading as JIT model first
  try:
    policy = torch.jit.load(checkpoint, map_location=device)
    print(f"Loaded JIT policy from {checkpoint}")
    return policy
  except Exception:
    pass

  # Load using runner (proper way for RSL-RL checkpoints)
  try:
    # Wrap environment for runner compatibility
    wrapped_env = RslRlVecEnvWrapper(
      env, clip_actions=getattr(rl_cfg, "clip_actions", 1.0)
    )

    # Get the appropriate runner class (MjlabOnPolicyRunner strips
    # None-valued optional configs like cnn_cfg that MLPModel rejects,
    # and handles legacy checkpoint migration).
    runner_cls = MjlabOnPolicyRunner

    # Create runner with config
    if is_dataclass(rl_cfg):
      runner_cfg = asdict(rl_cfg)  # type: ignore[arg-type]
    else:
      runner_cfg = rl_cfg if isinstance(rl_cfg, dict) else {}

    runner = runner_cls(wrapped_env, runner_cfg, device=str(device))

    # Load checkpoint
    runner.load(str(checkpoint_path), map_location=str(device))

    # Get inference policy
    policy = runner.get_inference_policy(device=str(device))
    print(f"Loaded policy from {checkpoint} using runner")
    return policy

  except Exception as e:
    print(f"Failed to load with runner: {e}")
    raise ValueError(
      f"Could not load policy from {checkpoint}.\n"
      f"Error: {e}\n"
      "Make sure the checkpoint is compatible with the task."
    ) from e


class PathPlannerCorrectionExtractor:
  """Extracts training targets from the path planner and robot state.

  This class collects 2D training targets:
  - delta_yaw: heading command (dynamically blended between waypoint and key_goal)
  - lin_vel_x: robot's forward linear velocity in body frame

  **Dynamic Target (Plan A):**
  The delta_yaw target is a soft blend between:
    - yaw_to_waypoint (path planner's local heading, good near obstacles)
    - yaw_to_key_goal (direct bearing to final goal, good in open space)

  The blend is controlled by raycast clearance via a sigmoid gate:
    alpha = sigmoid((min_ray_top - threshold) * scale)
    target_yaw = alpha * yaw_to_key_goal + (1 - alpha) * yaw_to_waypoint

  NOTE: Raycast observations are normalised to [0, 1] (max distance = 4m).
  The min_ray value is denormalised to metres before comparing against the
  threshold so that the blending operates in physical units.

  Near obstacles (min_ray small): alpha ≈ 0 → follow waypoint (avoidance)
  In open space (min_ray large): alpha ≈ 1 → steer toward key_goal
  """

  # --- Dynamic target configuration ---
  dynamic_target_threshold: float = 3.5
  """Raycast clearance (m) centre of the waypoint→key_goal transition.
  With scale=10 the transition band is roughly [3.0, 4.0] m:
    ≤ 3.0 m  →  alpha ≤ 1%   (follow waypoint / avoidance)
    = 3.5 m  →  alpha = 50%  (midpoint)
    ≥ 4.0 m  →  alpha ≥ 99%  (steer toward key_goal)"""

  dynamic_target_scale: float = 10.0
  """Sigmoid steepness for the soft blend. 10.0 compresses the full
  transition into ~1 m (from threshold−0.5 to threshold+0.5)."""

  dynamic_target_skip_ground_cols: int = 3
  """Number of ground-hitting image columns to skip (out of 8).

  Due to the forward_camera site euler (0, -π/2, 0), image axes are rotated:
    - Image columns (u axis) → robot vertical (col 0 = 24° downward, col 7 = 24° upward)
    - Image rows (v axis) → robot left/right
  The lowest columns always hit the ground even on flat terrain:
    col 0 (u=-1):    24° down → ground at ~1.8 m
    col 1 (u=-5/7):  17° down → ground at ~2.5 m
    col 2 (u=-3/7):  10° down → ground at ~3.9 m (pitch-dependent)
    col 3+ :          ~horizontal or upward → clean in open terrain
  Default 3 skips all reliably-ground-hitting columns."""

  alpha_smooth_up_rate: float = 0.05
  """EMA rate for alpha increasing (obstacle clearing → open space).
  Smaller = slower transition.
  At 50 Hz with rate=0.03:  ~0.5 s to reach 50%, ~1.5 s to reach 90%.
  This prevents the local planner training target from jumping instantly from
  waypoint (avoidance) to key_goal (goal-seeking) when the robot
  passes a wall, teaching the network to keep walking forward briefly
  before gradually rotating toward the goal."""

  alpha_smooth_down_rate: float = 0.5
  """EMA rate for alpha decreasing (open space → obstacle detected).
  Larger = faster response (safety-critical).
  At 50 Hz with rate=0.5: ~0.1 s to respond to a new obstacle."""

  raycast_max_distance: float = 4.0
  """Max raycast sensor distance (m).  Observations are normalised to
  [0, 1] by dividing by this value; we multiply back before comparing
  against metre-based thresholds."""

  raycast_obs_name: str = "raycast"
  """Key for primary (forward) raycast observations in obs_dict."""

  raycast_obs_names: tuple[str, ...] = ("raycast",)
  """All raycast observation keys. When len > 1, side raycasts are used
  in the alpha blending logic: if the goal-side raycast min distance
  is below ``side_raycast_avoidance_threshold``, alpha is forced to 0
  (avoidance mode) regardless of forward clearance."""

  side_raycast_avoidance_threshold: float = 1.6
  """Side raycast clearance threshold (metres). When the minimum distance
  from a side raycast (left or right, chosen by goal direction) is below
  this value, alpha is forced to 0 (full avoidance / follow waypoint).
  Default 1.6m — obstacles within 1.6m on the goal side block goal-seeking."""

  side_raycast_skip_ground_cols: int = 3
  """Ground columns to skip for side raycasts (same camera geometry)."""

  # --- TTC-based alpha blending (for high-speed tasks) ---
  use_ttc_blending: bool = False
  """Use Time-To-Collision (TTC) instead of fixed distance threshold for
  alpha blending.  When enabled, the blending sigmoid operates on
  ``TTC = min_ray_m / v_forward`` instead of ``min_ray_m``.

  This automatically adapts the avoidance trigger to the robot's current
  speed: at 2.2 m/s the robot enters avoidance mode much earlier (in
  distance) than at 1.0 m/s, matching the larger turning radius needed.

  Only affects training targets; does not change the local planner architecture or
  inference.  Enable via ``--use-ttc-blending`` CLI flag for high-speed
  tasks; leave disabled for low-speed / indoor tasks."""

  ttc_threshold: float = 2.0
  """TTC threshold (seconds) centre of the waypoint→key_goal transition.
  Approximation: π / (2 * ω_max) + margin ≈ 1.57 + 0.43 = 2.0 s.
  At 2.2 m/s a wall at 4.0m gives TTC = 1.82s < 2.0 → avoidance.
  At 1.5 m/s a wall at 4.0m gives TTC = 2.67s > 2.0 → goal-seeking."""

  ttc_scale: float = 3.0
  """Sigmoid steepness for TTC blending.  With scale=3.0 the transition
  band spans roughly ±0.7 s around ttc_threshold (1.3–2.7 s)."""

  ttc_min_velocity: float = 0.3
  """Minimum forward velocity (m/s) for TTC computation.  Prevents
  division by zero or extreme TTC values when robot is near-stationary."""

  # --- Lookahead teacher target (for high-speed tasks) ---
  use_lookahead_teacher: bool = False
  """When enabled and the current waypoint is roughly straight ahead,
  replace yaw_to_waypoint with yaw to the NEXT waypoint in the sequence.
  This provides an early turning signal before the path planner advances,
  recovering critical reaction time at high speed."""

  lookahead_yaw_threshold: float = 0.3
  """Yaw threshold (rad) below which current waypoint is considered
  'straight ahead' and the lookahead target is used instead."""

  max_target_yaw: float = 1.7453
  """Clamp teacher delta_yaw targets to ±max_target_yaw (rad).
  Must match the network's max_yaw_correction so that large bearing
  angles (> 100°) are clamped instead of saturating tanh and wasting
  gradient capacity.  Default 1.7453 rad ≈ 100°."""

  def __init__(self, env: ManagerBasedRlEnv, command_name: str = "goal"):
    self.env = env
    self.command_name = command_name
    self.device = env.device
    self.num_envs = env.num_envs
    self.use_odometry = False
    self.target_label_mode: Literal["dynamic", "waypoint_only", "key_goal_only"] = (
      "dynamic"
    )

    # Auto-derive blending thresholds from the task's goal command config.
    # This ensures the avoidance/goal-seeking transition matches the
    # physical parameters of each task (d_turn, d_safe) rather than
    # using a single hardcoded value for all tasks.
    #   dynamic_target_threshold = d_turn  (switch to avoidance at turn distance)
    #   side_raycast_avoidance_threshold = d_safe + 0.8  (keep avoidance while along wall)
    try:
      term = env.command_manager.get_term(command_name)
      if term is not None:
        term_cfg = getattr(term, "cfg", None)
        if term_cfg is not None:
          _tt = getattr(term_cfg, "turn_threshold", None)
          if _tt is not None:
            self.dynamic_target_threshold = float(_tt)
          _moc = getattr(term_cfg, "min_obstacle_clearance", 0.0)
          if _moc > 0:
            self.side_raycast_avoidance_threshold = float(_moc) + 0.8
    except (KeyError, AttributeError):
      pass  # keep class defaults (3.5, 1.6)

    # Previous commanded yaw for computing corrections
    self._prev_commanded_yaw: torch.Tensor | None = None
    self._goal_resampled: torch.Tensor | None = None
    # Smoothed alpha from previous step (for temporal EMA)
    self._prev_alpha: torch.Tensor | None = None
    self.last_alpha_raw: torch.Tensor | None = None
    self.last_alpha: torch.Tensor | None = None
    self.last_forward_clearance_m: torch.Tensor | None = None
    self.last_goal_side_clearance_m: torch.Tensor | None = None
    self.last_goal_side_blocked: torch.Tensor | None = None
    self.last_yaw_to_waypoint: torch.Tensor | None = None
    self.last_yaw_to_key_goal: torch.Tensor | None = None
    self.last_target_yaw: torch.Tensor | None = None

  def reset(self) -> None:
    """Reset the extractor state."""
    self._prev_commanded_yaw = None
    self._goal_resampled = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._prev_alpha = None
    self.last_alpha_raw = None
    self.last_alpha = None
    self.last_forward_clearance_m = None
    self.last_goal_side_clearance_m = None
    self.last_goal_side_blocked = None
    self.last_yaw_to_waypoint = None
    self.last_yaw_to_key_goal = None
    self.last_target_yaw = None

  def extract_correction(
    self,
    done: torch.Tensor,
    obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract training targets from path planner and robot state.

    Returns 2D target:
    - [0]: delta_yaw - dynamically blended heading target
           Near obstacles: angle to current waypoint (avoidance)
           Open space: angle to key_goal (goal-seeking)
    - [1]: lin_vel_x - robot's forward linear velocity in body frame

    Args:
      done: Episode termination flags [num_envs]
      obs_dict: Observation dictionary containing raycast for dynamic target.
        If None, falls back to waypoint-only target (legacy behaviour).

    Returns:
      Tuple of (target [num_envs, 2], new_goal_mask [num_envs])
    """
    target_dim = 1 if self.use_odometry else 2
    target = torch.zeros(self.num_envs, target_dim, device=self.device)
    new_goal_mask = done.clone()

    try:
      term = self.env.command_manager.get_term(self.command_name)
      robot = self.env.scene["robot"]

      # --- Compute robot yaw (needed for both waypoint and key_goal targets) ---
      robot_pos_w = robot.data.root_link_pos_w[:, :2]
      robot_quat = robot.data.root_link_quat_w
      robot_yaw = torch.atan2(
        2.0
        * (robot_quat[:, 0] * robot_quat[:, 3] + robot_quat[:, 1] * robot_quat[:, 2]),
        1.0 - 2.0 * (robot_quat[:, 2] ** 2 + robot_quat[:, 3] ** 2),
      )

      # --- Target 0a: delta_yaw to WAYPOINT (avoidance target) ---
      yaw_to_waypoint = torch.zeros(self.num_envs, device=self.device)
      if term is not None and hasattr(term, "delta_yaw"):
        yaw_to_waypoint = term.delta_yaw.clone()  # type: ignore
      elif term is not None:
        if hasattr(term, "_current_waypoint_w"):
          wp_w = term._current_waypoint_w[:, :2]  # type: ignore
        elif hasattr(term, "goal_pos_w"):
          wp_w = term.goal_pos_w[:, :2]  # type: ignore
        else:
          wp_w = robot_pos_w
        to_wp = wp_w - robot_pos_w
        angle_to_wp = torch.atan2(to_wp[:, 1], to_wp[:, 0])
        delta = angle_to_wp - robot_yaw
        yaw_to_waypoint = torch.atan2(torch.sin(delta), torch.cos(delta))

      # --- Target 0b: delta_yaw to KEY_GOAL (goal-seeking target) ---
      yaw_to_key_goal = yaw_to_waypoint.clone()  # default: same as waypoint
      if term is not None and hasattr(term, "key_goal_pos_w"):
        kg_pos = term.key_goal_pos_w[:, :2]  # type: ignore
        to_kg = kg_pos - robot_pos_w
        angle_to_kg = torch.atan2(to_kg[:, 1], to_kg[:, 0])
        delta_kg = angle_to_kg - robot_yaw
        yaw_to_key_goal = torch.atan2(torch.sin(delta_kg), torch.cos(delta_kg))

      # --- Lookahead teacher target (high-speed only) ---
      # When the current waypoint is roughly straight ahead, the teacher
      # gives a near-zero yaw signal — wasting ~0.25s (3 raycast frames)
      # that the network could use to start turning.  Replace with the
      # yaw to the NEXT waypoint so the network learns to anticipate.
      if self.use_lookahead_teacher and term is not None:
        yaw_to_waypoint = self._apply_lookahead_teacher(
          term, robot_pos_w, robot_yaw, yaw_to_waypoint
        )

      # --- Dynamic target blending based on raycast clearance ---
      # Prefer raw sensor data (bypasses observation-level DR such as
      # near-field occlusion) to keep training targets clean even when
      # the env config injects synthetic close-range readings into obs_dict.
      # Falls back to obs_dict when the sensor is not accessible.
      _raw_fwd = self._get_raw_sensor_distances(self.raycast_obs_name)
      if _raw_fwd is not None:
        raycast: torch.Tensor | None = _raw_fwd
      elif obs_dict is not None and self.raycast_obs_name in obs_dict:
        _ray_val = obs_dict[self.raycast_obs_name]
        if isinstance(_ray_val, torch.Tensor):
          _rv = _ray_val
          if _rv.shape[-1] > 64:
            _rv = _rv[..., -64:]
          raycast = _rv
        else:
          raycast = None
      else:
        raycast = None

      if raycast is not None:
        # Reshape to (N, 8_rows, 8_cols) grid.
        # Due to camera euler (0, -π/2, 0), image cols map to robot
        # vertical: col 0 = downward (ground), col 7 = upward.
        # Skip ground-hitting columns to get horizontal/upward rays.
        grid = raycast.reshape(-1, 8, 8)
        horiz_rays = grid[:, :, self.dynamic_target_skip_ground_cols :]  # [N, 8, K]
        min_ray = horiz_rays.reshape(grid.shape[0], -1).min(dim=-1).values  # [N]

        # Denormalise to metres for comparison against metre-based threshold
        min_ray_m = min_ray * self.raycast_max_distance  # [num_envs], in metres

        if self.use_ttc_blending:
          # TTC-based blending: alpha depends on time-to-collision rather
          # than raw distance.  High speed → small TTC → earlier avoidance.
          v_fwd = robot.data.root_link_lin_vel_b[:, 0].clamp(min=self.ttc_min_velocity)
          ttc = min_ray_m / v_fwd  # [num_envs], in seconds
          alpha = torch.sigmoid(
            (ttc - self.ttc_threshold) * self.ttc_scale
          )  # [num_envs]
        else:
          # Distance-based blending (original behaviour)
          alpha = torch.sigmoid(
            (min_ray_m - self.dynamic_target_threshold) * self.dynamic_target_scale
          )  # [num_envs]
        alpha_raw = alpha.clone()
        goal_side_min_diag = torch.full_like(alpha, float("nan"))
        side_blocked_diag = torch.zeros_like(alpha, dtype=torch.bool)

        # --- Side raycast constraint (n3 only) ---
        # When side raycasts are available, check the goal-direction side:
        # if obstacles are within side_raycast_avoidance_threshold on the
        # side the robot needs to turn toward, force avoidance mode.
        if len(self.raycast_obs_names) >= 3:
          left_name = self.raycast_obs_names[1]
          right_name = self.raycast_obs_names[2]
          left_min_m = self._compute_side_min_distance(obs_dict, left_name)
          right_min_m = self._compute_side_min_distance(obs_dict, right_name)

          if left_min_m is not None and right_min_m is not None:
            # Goal-side check: yaw_to_key_goal > 0 → target is left,
            # yaw_to_key_goal < 0 → target is right
            goal_side_min = torch.where(
              yaw_to_key_goal > 0,
              left_min_m,
              right_min_m,
            )
            # Force alpha to 0 when goal-side obstacle is too close
            side_blocked = goal_side_min < self.side_raycast_avoidance_threshold
            alpha = torch.where(side_blocked, torch.zeros_like(alpha), alpha)
            goal_side_min_diag = goal_side_min
            side_blocked_diag = side_blocked

        # Temporal smoothing of alpha (asymmetric EMA):
        #   - When obstacle clears (alpha up): slow ramp → robot keeps walking
        #     forward briefly before gradually turning toward key_goal.
        #   - When obstacle appears (alpha down): fast response for safety.
        if self._prev_alpha is None:
          self._prev_alpha = alpha.clone()
        else:
          rate = torch.where(
            alpha > self._prev_alpha,
            torch.tensor(self.alpha_smooth_up_rate, device=self.device),
            torch.tensor(self.alpha_smooth_down_rate, device=self.device),
          )
          smoothed = self._prev_alpha + rate * (alpha - self._prev_alpha)
          # Reset to raw alpha for environments that just finished an episode
          alpha = torch.where(done, alpha, smoothed)
        self._prev_alpha = alpha.clone()
        self.last_alpha_raw = alpha_raw.detach().clone()
        self.last_alpha = alpha.detach().clone()
        self.last_forward_clearance_m = min_ray_m.detach().clone()
        self.last_goal_side_clearance_m = goal_side_min_diag.detach().clone()
        self.last_goal_side_blocked = side_blocked_diag.detach().clone()

        # Blended target
        if self.target_label_mode == "waypoint_only":
          target[:, 0] = yaw_to_waypoint
        elif self.target_label_mode == "key_goal_only":
          target[:, 0] = yaw_to_key_goal
        else:
          target[:, 0] = alpha * yaw_to_key_goal + (1.0 - alpha) * yaw_to_waypoint
      else:
        if self.target_label_mode == "key_goal_only":
          target[:, 0] = yaw_to_key_goal
        else:
          # No raycast available: fallback to waypoint-only target
          target[:, 0] = yaw_to_waypoint

      self.last_yaw_to_waypoint = yaw_to_waypoint.detach().clone()
      self.last_yaw_to_key_goal = yaw_to_key_goal.detach().clone()
      self.last_target_yaw = target[:, 0].detach().clone()

      # Clamp delta_yaw target to network's representable range.
      # Targets beyond ±max_target_yaw would only saturate tanh,
      # wasting gradient capacity without teaching useful behaviour.
      target[:, 0] = target[:, 0].clamp(-self.max_target_yaw, self.max_target_yaw)
      self.last_target_yaw = target[:, 0].detach().clone()

      # --- Target 1: forward linear velocity in body frame ---
      # Only in legacy DR mode; odom mode uses 1D target (delta_yaw only)
      if target.shape[-1] >= 2:
        target[:, 1] = robot.data.root_link_lin_vel_b[:, 0]

      # Detect goal resampling
      if term is not None and hasattr(term, "_goal_resampled"):
        new_goal_mask = term._goal_resampled.clone() | done  # type: ignore
      else:
        new_goal_mask = done.clone()

    except (KeyError, AttributeError) as e:
      print(f"Warning: Could not extract training targets: {e}")

    return target, new_goal_mask

  def _get_raw_sensor_distances(
    self,
    sensor_name: str,
  ) -> torch.Tensor | None:
    """Read raw (pre-DR) normalized raycast distances directly from the sensor.

    Bypasses observation-level domain randomization (e.g. near-field occlusion
    injected by ``_enable_near_occlusion_dr``) so that training targets are
    always computed from clean physical sensor readings.

    Args:
      sensor_name: Scene sensor key (same as the observation group name).

    Returns:
      Normalized distances tensor [num_envs, num_rays] in [0, 1], or None
      if the sensor is not found in the scene.
    """
    try:
      sensor = self.env.scene[sensor_name]
      return sensor.data.distances / self.raycast_max_distance
    except (KeyError, AttributeError):
      return None

  def _compute_side_min_distance(
    self,
    obs_dict: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]] | None,
    raycast_key: str,
  ) -> torch.Tensor | None:
    """Compute minimum horizontal raycast distance for a side sensor.

    Prefers raw sensor data (bypasses observation-level DR) and falls back
    to obs_dict when the sensor is not accessible.

    Args:
      obs_dict: Observation dictionary.
      raycast_key: Key for the side raycast in obs_dict.

    Returns:
      Minimum distance in metres [num_envs], or None if unavailable.
    """
    if obs_dict is None:
      return None

    # Prefer raw sensor data to bypass observation-level DR
    raw_rc = self._get_raw_sensor_distances(raycast_key)
    if raw_rc is not None:
      rc = raw_rc
    else:
      if raycast_key not in obs_dict:
        return None
      _val = obs_dict[raycast_key]
      if not isinstance(_val, torch.Tensor):
        return None
      rc = _val
      if rc.shape[-1] > 64:
        rc = rc[..., -64:]
    grid = rc.reshape(-1, 8, 8)
    horiz = grid[:, :, self.side_raycast_skip_ground_cols :]
    min_ray = horiz.reshape(grid.shape[0], -1).min(dim=-1).values
    return min_ray * self.raycast_max_distance

  def _apply_lookahead_teacher(
    self,
    term: Any,
    robot_pos_w: torch.Tensor,
    robot_yaw: torch.Tensor,
    yaw_to_waypoint: torch.Tensor,
  ) -> torch.Tensor:
    """Replace yaw_to_waypoint with yaw-to-next-waypoint when current WP is ahead.

    For FixedSequentialGoalCommand: uses current_goal_idx, _env_goal_offsets,
    _spawn_pos_w, and num_goals to compute the next waypoint position.

    For ReachableGoalCommand: uses _current_waypoint_idx, _waypoints_tensor,
    and _num_waypoints.

    When |yaw_to_waypoint| < lookahead_yaw_threshold AND a next waypoint
    exists, the returned yaw target points to the next waypoint instead,
    giving the network an early turning signal.

    Args:
      term: The goal command term.
      robot_pos_w: Robot XY positions [num_envs, 2].
      robot_yaw: Robot heading [num_envs].
      yaw_to_waypoint: Current yaw-to-waypoint target [num_envs].

    Returns:
      Modified yaw_to_waypoint with lookahead applied where appropriate.
    """
    result = yaw_to_waypoint.clone()
    threshold = self.lookahead_yaw_threshold

    # Mask: current WP is roughly straight ahead
    straight_ahead = yaw_to_waypoint.abs() < threshold

    if not straight_ahead.any():
      return result

    # ── FixedSequentialGoalCommand path ──
    if hasattr(term, "current_goal_idx") and hasattr(term, "_env_goal_offsets"):
      cur_idx = term.current_goal_idx  # [num_envs] long
      num_goals = term.num_goals  # scalar int
      has_next = (cur_idx < num_goals - 1) & straight_ahead

      if has_next.any():
        next_idx = cur_idx[has_next] + 1  # [K] long
        # Gather next WP offset: [K, 2]
        idx_exp = next_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        next_offset = torch.gather(
          term._env_goal_offsets[has_next], 1, idx_exp
        ).squeeze(1)
        # World position of next WP
        next_wp_w = term._spawn_pos_w[has_next] + next_offset  # [K, 2]
        to_next = next_wp_w - robot_pos_w[has_next]
        angle_to_next = torch.atan2(to_next[:, 1], to_next[:, 0])
        delta_next = angle_to_next - robot_yaw[has_next]
        yaw_to_next = torch.atan2(torch.sin(delta_next), torch.cos(delta_next))
        result[has_next] = yaw_to_next

    # ── ReachableGoalCommand path ──
    elif hasattr(term, "_current_waypoint_idx") and hasattr(term, "_waypoints_tensor"):
      cur_idx = term._current_waypoint_idx  # [num_envs] long
      n_wp = term._num_waypoints  # [num_envs] long
      has_next = (cur_idx < n_wp - 1) & straight_ahead

      if has_next.any():
        next_idx = cur_idx[has_next] + 1
        idx_exp = next_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
        next_wp = torch.gather(term._waypoints_tensor[has_next], 1, idx_exp).squeeze(1)
        to_next = next_wp - robot_pos_w[has_next]
        angle_to_next = torch.atan2(to_next[:, 1], to_next[:, 0])
        delta_next = angle_to_next - robot_yaw[has_next]
        yaw_to_next = torch.atan2(torch.sin(delta_next), torch.cos(delta_next))
        result[has_next] = yaw_to_next

    return result


def train_local_planner(args: TrainLocalPlannerArgs) -> None:
  """Train the steering local planner.

  Args:
    args: Training arguments.
  """
  device = torch.device(args.device)
  configure_torch_backends()

  # Validate checkpoint argument
  if not args.checkpoint:
    raise ValueError(
      "Checkpoint path required for training. Please provide a trained locomotion policy:\n"
      "  --checkpoint /path/to/model.pt\n\n"
      "Example checkpoint locations:\n"
      "  logs/rsl_rl/<task_name>/<timestamp>/model_XXX.pt"
    )

  if not os.path.isfile(args.checkpoint):
    raise FileNotFoundError(
      f"Checkpoint file not found: {args.checkpoint}\n"
      "Please provide a valid path to a trained locomotion policy."
    )

  print(f"Training local planner for task: {args.task}")
  print(f"Number of environments: {args.num_envs}")
  print(f"Output directory: {args.output_dir}")

  # Initialize WandB
  wandb_run = None
  if args.use_wandb:
    try:
      import wandb

      wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        config={
          "task": args.task,
          "num_envs": args.num_envs,
          "max_iterations": args.max_iterations,
          "checkpoint": args.checkpoint,
          "local_planner_cfg": {
            "buffer_size": args.local_planner_cfg.buffer_size,
            "batch_size": args.local_planner_cfg.batch_size,
            "lr": args.local_planner_cfg.lr,
            "sequence_length": args.local_planner_cfg.sequence_length,
          },
        },
        reinit=True,
      )
      print(f"WandB initialized: {wandb_run.url}")
    except ImportError:
      print("WandB not installed, skipping logging.")
      args.use_wandb = False
    except Exception as e:
      print(f"WandB init failed: {e}")
      args.use_wandb = False

  # Setup environment
  env, rl_cfg = setup_environment(args)

  # Load locomotion policy
  policy = load_policy(args.checkpoint, env, rl_cfg, device)

  if args.video:
    if args.video_interval <= 0:
      raise ValueError("--video-interval must be positive when --video is enabled.")
    if args.video_length <= 0:
      raise ValueError("--video-length must be positive when --video is enabled.")
    video_folder = Path(args.output_dir) / "videos" / "train" / args.experiment_name
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step % args.video_interval == 0,
      video_length=args.video_length,
      name_prefix=f"local-planner-{args.experiment_name}",
      disable_logger=False,
    )
    print(f"[INFO] Recording local planner training videos to {video_folder}")

  # Create local planner trainer
  local_planner_cfg = args.local_planner_cfg
  local_planner_cfg.save_dir = args.output_dir
  local_planner_cfg.raycast_obs_names = args.raycast_obs_names
  if args.disable_teacher_forcing:
    local_planner_cfg.use_gt_goal_inputs = False
    print("[Ablation] Teacher forcing disabled: using DR goal inputs in training.")
  if args.disable_dr_curriculum:
    local_planner_cfg.curriculum_phase1_updates = 0
    local_planner_cfg.curriculum_phase2_updates = 0
    print("[Ablation] DR curriculum disabled: all loss weights active from start.")
  # Override sequence_length if provided (e.g. 96 for high-speed tasks)
  if args.sequence_length is not None:
    local_planner_cfg.sequence_length = args.sequence_length
    print(f"[SeqLen] Overriding sequence_length to {args.sequence_length}")
  # Propagate num_raycast_groups to network config
  local_planner_cfg.network_cfg.num_raycast_groups = len(args.raycast_obs_names)

  # --- Odometry mode propagation ---
  if args.use_odometry:
    local_planner_cfg.use_odometry = True
    local_planner_cfg.odom_vel_dim = 2
    local_planner_cfg.include_joint_vel = args.include_joint_vel
    # Auto-disable joint_vel in odom mode unless user explicitly included it
    # (tyro will keep include_joint_vel=True by default; check if it was
    # explicitly set by the user by comparing with the class default when
    # use_odometry was False — if the user passes --include-joint-vel True
    # alongside --use-odometry True, respect that)
    print(
      f"[ODOM] Odometry mode enabled: output_dim=1, odom_vel_dim=2, "
      f"include_joint_vel={local_planner_cfg.include_joint_vel}"
    )

  # For multi-camera (n3+): use independent CNN encoders and larger capacity
  if len(args.raycast_obs_names) > 1:
    local_planner_cfg.network_cfg.use_independent_raycast_cnn = True
    local_planner_cfg.network_cfg.hidden_dims = (128, 96)
    local_planner_cfg.network_cfg.state_dim = 64
    local_planner_cfg.network_cfg.raycast_cnn_output_dim = 48
  trainer = LocalPlannerTrainer(
    cfg=local_planner_cfg,
    num_envs=args.num_envs,
    device=device,
  )

  # Load existing local planner checkpoint if available
  if args.local_planner_checkpoint and os.path.isfile(args.local_planner_checkpoint):
    print(f"Loading existing local planner from {args.local_planner_checkpoint}")
    trainer.load(args.local_planner_checkpoint)

  # Create path planner correction extractor
  if not args.raycast_obs_names:
    raise ValueError("raycast_obs_names must contain at least one observation key.")

  def _configure_extractor(
    extractor: PathPlannerCorrectionExtractor,
  ) -> PathPlannerCorrectionExtractor:
    extractor.use_odometry = args.use_odometry
    extractor.raycast_obs_name = args.raycast_obs_names[0]
    extractor.raycast_obs_names = args.raycast_obs_names
    extractor.target_label_mode = args.target_label_mode
    if args.use_ttc_blending:
      extractor.use_ttc_blending = True
      extractor.ttc_threshold = args.ttc_threshold
      extractor.ttc_scale = args.ttc_scale
    if args.use_lookahead_teacher:
      extractor.use_lookahead_teacher = True
      extractor.lookahead_yaw_threshold = args.lookahead_yaw_threshold
    return extractor

  correction_extractor = _configure_extractor(PathPlannerCorrectionExtractor(env))
  print(f"[Ablation] Target label mode: {args.target_label_mode}")
  if args.use_ttc_blending:
    print(
      f"[TTC] Enabled TTC-based alpha blending: "
      f"threshold={args.ttc_threshold:.1f}s, scale={args.ttc_scale:.1f}"
    )
  if args.use_lookahead_teacher:
    print(
      f"[Lookahead] Enabled lookahead teacher target: "
      f"yaw_threshold={args.lookahead_yaw_threshold:.2f} rad"
    )

  behavior_correction_extractor: PathPlannerCorrectionExtractor | None = None
  if args.apply_teacher_label_to_policy:
    behavior_correction_extractor = _configure_extractor(
      PathPlannerCorrectionExtractor(env)
    )

    class TeacherLabelControlPolicy:
      """Policy wrapper that lets the teacher label control data collection."""

      def __init__(self, base_policy: Callable[..., torch.Tensor]):
        self._base_policy = base_policy
        self._goal_cmd_offset: int | None = None
        self._warned_missing_actor = False

      def __call__(self, obs: object) -> torch.Tensor:
        assert behavior_correction_extractor is not None
        flat_obs = _flatten_tensor_observations(obs)
        if flat_obs:
          done = env.episode_length_buf == 0
          target, _ = behavior_correction_extractor.extract_correction(
            done, obs_dict=flat_obs
          )
          target_yaw = target[:, 0]

          if _dict_like_has_key(obs, "actor"):
            if self._goal_cmd_offset is None:
              self._goal_cmd_offset = _find_goal_command_offset(env)
              print(
                f"[INFO] goal_command offset in policy obs: {self._goal_cmd_offset}"
              )
            if self._goal_cmd_offset is not None:
              modified_actor_obs = _dict_like_get(obs, "actor").clone()
              modified_actor_obs[:, self._goal_cmd_offset] = target_yaw
              _dict_like_set(obs, "actor", modified_actor_obs)
          elif not self._warned_missing_actor:
            print(
              "[WARN] Teacher-label control requested, but actor observation "
              "was not found; policy actions are unchanged."
            )
            self._warned_missing_actor = True

        return self._base_policy(obs)

    policy = TeacherLabelControlPolicy(policy)
    print(
      "[Train] Teacher label control enabled: "
      f"target_label_mode={args.target_label_mode}"
    )

  # Training loop
  obs_dict, _ = env.reset()
  correction_extractor.reset()
  if behavior_correction_extractor is not None:
    behavior_correction_extractor.reset()
  trainer.obs_builder.reset(args.num_envs, device)

  # Initialize DR heading (only needed in non-odometry mode)
  if not args.use_odometry:
    _robot = env.scene["robot"]
    _quat = _robot.data.root_link_quat_w
    _init_yaw = torch.atan2(
      2.0 * (_quat[:, 0] * _quat[:, 3] + _quat[:, 1] * _quat[:, 2]),
      1.0 - 2.0 * (_quat[:, 2] ** 2 + _quat[:, 3] ** 2),
    )
    trainer.obs_builder.reset_dr(initial_heading=_init_yaw)

    # Initialize goals at start
    initial_goal_mask = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    trainer.obs_builder.update_initial_goal(env, initial_goal_mask)

  step = 0

  # Performance timing
  t_policy = 0.0
  t_env_step = 0.0
  t_extract = 0.0
  t_collect = 0.0
  t_update = 0.0
  t_start = time.time()

  print("Starting local planner training...")
  print("Collecting path planner demonstrations...")

  interrupted = False
  try:
    while step < args.max_iterations:
      # Get policy action
      t0 = time.time()
      with torch.no_grad():
        actions = policy(obs_dict)
      t_policy += time.time() - t0

      # Environment step
      t0 = time.time()
      step_result = env.step(actions)
      obs_dict = step_result[0]
      dones = step_result[2]
      t_env_step += time.time() - t0

      # Extract path planner correction (with dynamic target blending)
      t0 = time.time()
      target, new_goal_mask = correction_extractor.extract_correction(dones, obs_dict)
      t_extract += time.time() - t0

      # Collect experience
      t0 = time.time()
      if isinstance(obs_dict, dict):
        trainer.collect_experience(
          env=env,
          obs_dict=obs_dict,  # type: ignore
          target=target,
          done=dones,
          new_goal_mask=new_goal_mask,
        )
      else:
        trainer.collect_experience(
          env=env,
          obs_dict={"obs": obs_dict},
          target=target,
          done=dones,
          new_goal_mask=new_goal_mask,
        )
      t_collect += time.time() - t0

      # Training update
      t0 = time.time()
      stats = trainer.update()
      t_update += time.time() - t0

      # Logging
      if stats is not None:
        trainer.print_stats()
        if args.use_wandb and wandb_run is not None:
          import wandb

          wandb.log(
            {
              "local_planner/loss": stats["loss"],
              "local_planner/yaw_loss": stats["correction_loss"],
              "local_planner/vel_pred_loss": stats["vel_pred_loss"],
              "local_planner/smoothness_loss": stats["smoothness_loss"],
              "local_planner/proximity_loss": stats.get("proximity_loss", 0.0),
              "local_planner/lr": stats["lr"],
              "local_planner/cur_w_vel": stats.get("cur_w_vel", 1.0),
              "local_planner/cur_w_corr": stats.get("cur_w_corr", 1.0),
              # RMSE diagnostics
              "local_planner/vel_rmse": stats.get("vel_rmse", 0.0),
              "local_planner/yaw_rmse": stats.get("yaw_rmse", 0.0),
              "step": step,
            }
          )

      # Performance report
      if step > 0 and step % 1000 == 0:
        elapsed = time.time() - t_start
        fps = step / elapsed
        print(
          f"[Perf] Step {step}: {fps:.1f} FPS | "
          f"policy: {t_policy / step * 1000:.1f}ms, "
          f"env_step: {t_env_step / step * 1000:.1f}ms, "
          f"extract: {t_extract / step * 1000:.1f}ms, "
          f"collect: {t_collect / step * 1000:.1f}ms, "
          f"update: {t_update / step * 1000:.1f}ms"
        )

      # Saving
      if trainer.should_save():
        save_path = trainer.save(
          args.output_dir,
          f"local_planner_{args.experiment_name}.pt",
        )
        print(f"\033[92mSaved local planner to {save_path}\033[0m")

      # Periodic saving
      if step > 0 and step % args.save_interval == 0:
        save_path = trainer.save(
          args.output_dir,
          f"local_planner_{args.experiment_name}_step{step}.pt",
        )
        print(f"Checkpoint saved to {save_path}")

      step += 1

  except KeyboardInterrupt:
    interrupted = True
    print(f"\n[INFO] Training interrupted by user (Ctrl+C) at step {step}")
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
      save_path = trainer.save(
        args.output_dir,
        f"local_planner_{args.experiment_name}_interrupted_step{step}.pt",
      )
      print(f"Interrupted checkpoint saved to {save_path}")
    except Exception as e:
      print(f"[WARN] Failed to save interrupted checkpoint: {e}")

  if not interrupted:
    save_path = trainer.save(
      args.output_dir,
      f"local_planner_{args.experiment_name}_final.pt",
    )
    print(f"Final model saved to {save_path}")

  print(f"Training {'interrupted' if interrupted else 'completed'}.")

  # Finish WandB
  if args.use_wandb and wandb_run is not None:
    _WANDB_TIMEOUT = 10

    def _finish_wandb() -> None:
      try:
        import wandb

        wandb.finish(exit_code=130 if interrupted else 0)
      except Exception:
        pass

    t = threading.Thread(target=_finish_wandb, daemon=True)
    t.start()
    t.join(timeout=_WANDB_TIMEOUT)
    if t.is_alive():
      print(f"[WARN] WandB finish timed out after {_WANDB_TIMEOUT}s")
    else:
      print("WandB run finished.")

  # Close environment
  try:
    env.close()
  except Exception as e:
    print(f"[WARN] Environment close failed: {e}")

  if interrupted:
    print("[INFO] Force-exiting to avoid residual background tasks.")
    os._exit(130)


def test_local_planner(args: TrainLocalPlannerArgs) -> None:
  """Test the local planner with steering correction and visualization.

  Args:
    args: Training arguments.
  """
  import os as _os

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  device = torch.device(args.device)
  configure_torch_backends()

  print(f"Testing local planner for task: {args.task}")

  # Validate local planner checkpoint
  if not args.local_planner_checkpoint:
    raise ValueError("Local planner checkpoint required for test mode.")

  if not os.path.isfile(args.local_planner_checkpoint):
    raise FileNotFoundError(
      f"Local planner checkpoint not found: {args.local_planner_checkpoint}"
    )

  # Setup environment
  env, rl_cfg = setup_environment(args)

  # Load locomotion policy
  base_policy = load_policy(args.checkpoint, env, rl_cfg, device)

  # Create and load local planner network.  The checkpoint config is the
  # source of truth here; odometry checkpoints have a different input layout
  # and 1D output, so CLI-derived defaults can silently scramble inputs.
  local_planner_network, checkpoint_cfg = _load_local_planner_network_for_inference(
    args.local_planner_checkpoint,
    raycast_obs_names=args.raycast_obs_names,
    device=device,
    use_odometry=args.use_odometry,
    include_joint_vel=args.include_joint_vel,
  )
  net_cfg = local_planner_network.cfg
  print(
    "[LP] checkpoint architecture: "
    f"input_dim={local_planner_network.input_dim}, "
    f"output_dim={net_cfg.output_dim}, "
    f"state_dim={net_cfg.state_dim}, "
    f"raycast_groups={net_cfg.num_raycast_groups}, "
    f"odom_vel_dim={net_cfg.odom_vel_dim}, "
    f"joint_vel_dim={net_cfg.joint_vel_dim}"
  )

  # Create local planner inference module
  local_planner_cfg = LocalPlannerInferenceCfg()
  _apply_checkpoint_cfg_to_inference_cfg(local_planner_cfg, checkpoint_cfg, args)
  local_planner_cfg.use_measured_velocity = args.use_measured_velocity
  local_planner_cfg.use_ahrs_heading = args.use_ahrs_heading
  local_planner_cfg.use_projected_yaw_rate = args.use_projected_yaw_rate
  # Apply optional gate tuning overrides
  if args.smoothing_alpha is not None:
    local_planner_cfg.smoothing_alpha = args.smoothing_alpha
  if args.goal_seek_gain is not None:
    local_planner_cfg.goal_seek_gain = args.goal_seek_gain
  if args.gate_clearance_threshold is not None:
    local_planner_cfg.gate_clearance_threshold = args.gate_clearance_threshold
  if local_planner_cfg.use_odometry:
    print(
      "[ODOM] enabled: "
      f"odom_vel_dim={local_planner_cfg.odom_vel_dim}, "
      f"include_joint_vel={local_planner_cfg.include_joint_vel}, "
      f"goal_noise={local_planner_cfg.enable_odom_goal_noise} "
      f"(bearing_std={local_planner_cfg.odom_bearing_noise_std:.3f} rad, "
      f"distance_frac={local_planner_cfg.odom_distance_noise_frac:.3f}, "
      f"distance_min={local_planner_cfg.odom_distance_noise_min:.3f} m). "
      "Goal direction is computed from the current odometry/sim pose."
    )
  else:
    _vel_src = (
      "measured (state estimator)"
      if args.use_measured_velocity
      else "network pred_lin_vel"
    )
    _hdg_src = (
      "AHRS (IMU -> yaw)"
      if args.use_ahrs_heading
      else (
        "projected yaw rate (quat)"
        if args.use_projected_yaw_rate
        else "ang_vel_z integration"
      )
    )
    print(f"[DR] velocity: {_vel_src} | heading: {_hdg_src}")
  print(
    f"[Gate] smooth={local_planner_cfg.smoothing_alpha:.2f} "
    f"gain={local_planner_cfg.goal_seek_gain:.2f} "
    f"threshold={local_planner_cfg.gate_clearance_threshold:.1f}m "
    f"min_bias={args.min_goal_bias:.2f}"
  )
  local_planner_inference = LocalPlannerInference(
    cfg=local_planner_cfg,
    local_planner_network=local_planner_network,
    device=device,
  )
  local_planner_inference.initialize(args.num_envs)

  # Initialize goals
  initial_goal_mask = torch.ones(args.num_envs, dtype=torch.bool, device=device)

  # Statistics tracking
  step_count = [0]
  total_correction = [0.0]
  correction_count = [0]

  # Get the goal command term to modify its command
  goal_cmd_term = env.command_manager.get_term("goal")
  if goal_cmd_term is None:
    raise ValueError("No 'goal' command term found in environment.")

  # Check if this command has the _command attribute we can modify
  if not hasattr(goal_cmd_term, "_command"):
    print(
      "[WARN] Command term doesn't have _command attribute, corrections won't be applied."
    )

  # State for overlay display
  local_planner_state: LocalPlannerOverlayState = {
    "original_yaw": 0.0,
    "raw_correction": 0.0,
    "smoothed_correction": 0.0,
    "correction": 0.0,
    "corrected_yaw": 0.0,
    "hidden_norm": 0.0,
    "step": 0,
    "true_goal_x": 0.0,
    "true_goal_y": 0.0,
    "target_lin_vel": 0.0,
    "true_lin_vel": 0.0,
    "pred_lin_vel": 0.0,
    "dr_x": 0.0,
    "dr_y": 0.0,
    "dr_heading_deg": 0.0,
    "initial_goal_x": 0.0,
    "initial_goal_y": 0.0,
    "dr_est_goal_x": 0.0,
    "dr_est_goal_y": 0.0,
    "goal_bearing_deg": 0.0,
    "goal_distance": 0.0,
    "true_bearing_deg": 0.0,
    "true_distance": 0.0,
    "goal_seek_gate": 1.0,
    "goal_seek_yaw": 0.0,
    "avoidance_yaw": 0.0,
    "current_key_goal": "0/?",
    # 4D goal direction (network input)
    "goal_dir_dx": 0.0,
    "goal_dir_dy": 0.0,
    "goal_dir_cos": 0.0,
    "goal_dir_sin": 0.0,
    # True 4D goal direction (from GT)
    "true_dir_dx": 0.0,
    "true_dir_dy": 0.0,
    "true_dir_cos": 0.0,
    "true_dir_sin": 0.0,
    # Hidden decay factor
    "hidden_decay": 1.0,
  }

  # Create a wrapper policy that applies local planner corrections
  class LocalPlannerEnhancedPolicy:
    """Policy wrapper that applies local planner steering corrections.

    The key insight is that we must modify the OBSERVATION passed to the policy,
    not the command term itself. The observation is already computed when env.step()
    returns, so modifying the command term has no effect on the current step.

    The policy observation structure for raycast tasks is:
    - obs["actor"]: concatenated tensor of [base_ang_vel, proj_gravity, joint_pos,
                    joint_vel, actions, goal_command, ball_state]
    - goal_command is at a specific offset and has shape [num_envs, 2] as
      [delta_yaw, target_velocity]

    We need to find and modify the delta_yaw component in the policy observation.
    """

    def __init__(self):
      self.last_correction = None
      self.initialized = False
      # Cache the observation structure info
      self._goal_cmd_offset: int | None = None
      self._policy_obs_dim: int | None = None
      # Auxiliary output cache
      self.last_pred_lin_vel: float = 0.0
      # Ground truth cache
      self.last_true_goal_x: float = 0.0
      self.last_true_goal_y: float = 0.0
      self.last_target_lin_vel: float = 0.0
      self.last_true_lin_vel: float = 0.0
      self.last_true_bearing_deg: float = 0.0
      self.last_true_distance: float = 0.0

    def _find_goal_command_offset(self, obs: dict) -> int | None:
      """Find the offset of goal_command in the policy observation tensor.

      The policy observation is concatenated from multiple terms. We need to
      find where goal_command starts. The structure is typically:
      - base_ang_vel: 3
      - projected_gravity: 3
      - joint_pos_rel: num_joints (29 for G1)
      - joint_vel_rel: num_joints (29 for G1)
      - actions: num_actions (29 for G1)
      - goal_command: 2 (delta_yaw, target_velocity)
      - ball_state: 7 (placeholder)

      Total before goal_command: 3 + 3 + 29 + 29 + 29 = 93
      """
      del obs
      return _find_goal_command_offset(env)

    def __call__(self, obs: Any) -> torch.Tensor:
      # Initialize local planner state on first call
      if not self.initialized:
        local_planner_inference.obs_builder.update_initial_goal(
          env,
          initial_goal_mask,
        )
        self.initialized = True

      # Get local planner correction BEFORE modifying observations
      local_planner_obs_dict = _flatten_tensor_observations(obs)
      missing_raycast_obs = [
        obs_name
        for obs_name in args.raycast_obs_names
        if obs_name not in local_planner_obs_dict
      ]
      if missing_raycast_obs:
        full_obs_dict = _flatten_tensor_observations(
          env.observation_manager.compute(update_history=False)
        )
        for obs_name in missing_raycast_obs:
          if obs_name in full_obs_dict:
            local_planner_obs_dict[obs_name] = full_obs_dict[obs_name]
        still_missing = [
          obs_name
          for obs_name in args.raycast_obs_names
          if obs_name not in local_planner_obs_dict
        ]
        if still_missing and step_count[0] == 0:
          print(
            "[WARN] Local planner observation is missing raycast keys "
            f"{still_missing}. Available keys: {list(local_planner_obs_dict.keys())}. "
            "These keys will use the observation builder's ones fallback."
          )
        elif step_count[0] == 0:
          print(
            "[LP] merged raycast observations from full env observation for "
            f"{missing_raycast_obs}."
          )

      if step_count[0] == 0 or (step_count[0] + 1) % 100 == 0:
        print(
          "[LP Raycast] "
          + _format_raycast_debug_stats(
            local_planner_obs_dict,
            args.raycast_obs_names,
            local_planner_cfg.raycast_dim,
          )
        )
      correction = local_planner_inference.compute_correction(
        env,
        local_planner_obs_dict if local_planner_obs_dict else {"obs": obs},
        min_goal_bias=args.min_goal_bias,
      )

      # Get auxiliary predictions from metrics (already computed in compute_correction)
      _full_out = local_planner_inference.get_metrics()["full_output"]
      assert isinstance(_full_out, tuple), "full_output metric must be a tuple"
      self.last_pred_lin_vel = _full_out[1]

      # Extract ground truth from environment for comparison
      with torch.no_grad():
        robot = env.scene["robot"]
        # Target linear velocity from command term
        try:
          _cmd_term = env.command_manager.get_term("goal")
          _tv = (
            getattr(_cmd_term, "target_velocity", None)
            if _cmd_term is not None
            else None
          )
          if _tv is not None:
            self.last_target_lin_vel = (
              float(_tv.item()) if isinstance(_tv, torch.Tensor) else float(_tv)
            )
        except (KeyError, AttributeError):
          pass
        # Ground truth linear velocity (forward, body frame)
        self.last_true_lin_vel = float(robot.data.root_link_lin_vel_b[0, 0].item())
        # Ground truth relative goal position in WORLD frame
        # Pure translation: goal_w - robot_w (no rotation applied)
        try:
          _term = env.command_manager.get_term("goal")
          _nav_target_w = _goal_pos_w_from_term(_term) if _term is not None else None
          if _nav_target_w is not None:
            _robot_xy = robot.data.root_link_pos_w[0, :2]
            self.last_true_goal_x = float((_nav_target_w[0, 0] - _robot_xy[0]).item())
            self.last_true_goal_y = float((_nav_target_w[0, 1] - _robot_xy[1]).item())
            # Compute true bearing and distance for comparison
            import math as _m

            _robot_quat = robot.data.root_link_quat_w
            _robot_yaw = float(
              torch.atan2(
                2.0
                * (
                  _robot_quat[0, 0] * _robot_quat[0, 3]
                  + _robot_quat[0, 1] * _robot_quat[0, 2]
                ),
                1.0 - 2.0 * (_robot_quat[0, 2] ** 2 + _robot_quat[0, 3] ** 2),
              ).item()
            )
            _gx, _gy = self.last_true_goal_x, self.last_true_goal_y
            _angle_to_goal = _m.atan2(_gy, _gx)
            _raw_bearing = _angle_to_goal - _robot_yaw
            self.last_true_bearing_deg = _m.degrees(
              _m.atan2(_m.sin(_raw_bearing), _m.cos(_raw_bearing))
            )
            self.last_true_distance = _m.sqrt(_gx**2 + _gy**2)
        except (KeyError, AttributeError):
          pass

      self.last_correction = correction

      # APPLY CORRECTION TO OBSERVATION
      # We need to modify the delta_yaw in the policy observation tensor
      original_yaw_val = 0.0
      corrected_yaw_val = 0.0
      correction_val = float(correction[0].item())

      # Check if obs has dict-like interface (dict or TensorDict)
      has_actor_key = _dict_like_has_key(obs, "actor")
      if has_actor_key:
        # Find goal_command offset if not cached
        if self._goal_cmd_offset is None:
          self._goal_cmd_offset = self._find_goal_command_offset(obs)
          print(f"[INFO] goal_command offset in policy obs: {self._goal_cmd_offset}")

        if self._goal_cmd_offset is not None:
          policy_obs = _dict_like_get(obs, "actor")
          # goal_command is [delta_yaw, target_velocity] at offset
          goal_cmd_idx = self._goal_cmd_offset

          # Clone to avoid modifying the original observation in-place
          # (which could affect other systems)
          modified_policy_obs = policy_obs.clone()

          # Get original delta_yaw
          original_yaw = modified_policy_obs[:, goal_cmd_idx].clone()
          original_yaw_val = float(original_yaw[0].item())

          # The local planner directly outputs delta_yaw as the command.
          local_planner_delta_yaw = correction.squeeze(-1) * args.correction_gain
          corrected_yaw = local_planner_delta_yaw  # Direct output, not additive
          modified_policy_obs[:, goal_cmd_idx] = corrected_yaw
          corrected_yaw_val = float(corrected_yaw[0].item())

          # For TensorDict, we need to set the value directly
          # Creating a new dict won't work for TensorDict
          _dict_like_set(obs, "actor", modified_policy_obs)

      # Update state for overlay
      metrics = local_planner_inference.get_metrics()
      raw_correction = metrics["raw_correction"]
      smoothed_correction = metrics["smoothed_correction"]
      hidden_state_norm = metrics["hidden_state_norm"]
      goal_seek_gate = metrics.get("goal_seek_gate", 1.0)
      goal_seek_yaw = metrics.get("goal_seek_yaw", 0.0)
      avoidance_yaw = metrics.get("avoidance_yaw", 0.0)

      assert not isinstance(raw_correction, tuple)
      assert not isinstance(smoothed_correction, tuple)
      assert not isinstance(hidden_state_norm, tuple)
      assert not isinstance(goal_seek_gate, tuple)
      assert not isinstance(goal_seek_yaw, tuple)
      assert not isinstance(avoidance_yaw, tuple)

      local_planner_state["original_yaw"] = original_yaw_val
      local_planner_state["raw_correction"] = float(raw_correction)
      local_planner_state["smoothed_correction"] = float(smoothed_correction)
      local_planner_state["correction"] = (
        correction_val * args.correction_gain
      )  # clamped & scaled
      local_planner_state["corrected_yaw"] = corrected_yaw_val
      local_planner_state["hidden_norm"] = float(hidden_state_norm)
      local_planner_state["step"] = step_count[0]
      local_planner_state["true_goal_x"] = self.last_true_goal_x
      local_planner_state["true_goal_y"] = self.last_true_goal_y
      local_planner_state["true_bearing_deg"] = self.last_true_bearing_deg
      local_planner_state["true_distance"] = self.last_true_distance
      local_planner_state["target_lin_vel"] = self.last_target_lin_vel
      local_planner_state["true_lin_vel"] = self.last_true_lin_vel
      local_planner_state["pred_lin_vel"] = self.last_pred_lin_vel
      local_planner_state["goal_seek_gate"] = float(goal_seek_gate)
      local_planner_state["goal_seek_yaw"] = float(goal_seek_yaw)
      local_planner_state["avoidance_yaw"] = float(avoidance_yaw)

      # Key goal progress
      try:
        _gt = env.command_manager.get_term("goal")
        if _gt is not None and hasattr(_gt, "current_key_goal_idx"):
          _ki = int(_gt.current_key_goal_idx[0].item())  # type: ignore
          _kt = len(_gt._key_goal_indices)  # type: ignore
          local_planner_state["current_key_goal"] = f"{_ki}/{_kt}"
      except (KeyError, AttributeError):
        pass

      # Extract DR state from obs_builder
      _dr_pos = local_planner_inference.obs_builder._dr_pos
      _dr_heading = local_planner_inference.obs_builder._dr_heading
      _initial_goal = local_planner_inference.obs_builder._initial_goal_pos
      _last_lp_obs = local_planner_inference._last_local_planner_obs
      if _dr_pos is not None:
        local_planner_state["dr_x"] = float(_dr_pos[0, 0].item())
        local_planner_state["dr_y"] = float(_dr_pos[0, 1].item())
      if _dr_heading is not None:
        import math as _math

        local_planner_state["dr_heading_deg"] = float(
          _math.degrees(_dr_heading[0].item())
        )
      if _initial_goal is not None:
        local_planner_state["initial_goal_x"] = float(_initial_goal[0, 0].item())
        local_planner_state["initial_goal_y"] = float(_initial_goal[0, 1].item())
      # DR-estimated current relative goal = initial_goal - DR_displacement
      if _initial_goal is not None and _dr_pos is not None:
        _est_goal_x = float((_initial_goal[0, 0] - _dr_pos[0, 0]).item())
        _est_goal_y = float((_initial_goal[0, 1] - _dr_pos[0, 1]).item())
        local_planner_state["dr_est_goal_x"] = _est_goal_x
        local_planner_state["dr_est_goal_y"] = _est_goal_y
        import math as _math2

        if _dr_heading is not None:
          _heading = float(_dr_heading[0].item())
          # Compute body-frame goal (same as obs_builder.build())
          _cos_h = _math2.cos(_heading)
          _sin_h = _math2.sin(_heading)
          _body_x = _est_goal_x * _cos_h + _est_goal_y * _sin_h
          _body_y = -_est_goal_x * _sin_h + _est_goal_y * _cos_h
          _dist = _math2.sqrt(_body_x**2 + _body_y**2)
          _dist_safe = max(_dist, 0.1)

          # DR 4D goal direction (what network sees)
          local_planner_state["goal_dir_dx"] = _body_x / 10.0
          local_planner_state["goal_dir_dy"] = _body_y / 10.0
          local_planner_state["goal_dir_cos"] = _body_x / _dist_safe
          local_planner_state["goal_dir_sin"] = _body_y / _dist_safe

          # Derived bearing and distance for human readability
          _bearing = _math2.atan2(_body_y, _body_x)
          local_planner_state["goal_bearing_deg"] = _math2.degrees(_bearing)
          local_planner_state["goal_distance"] = _dist
      elif _last_lp_obs is not None and _last_lp_obs.shape[-1] >= 4:
        import math as _math2

        _g = _last_lp_obs[0, -4:].detach()
        _gdx = float(_g[0].item())
        _gdy = float(_g[1].item())
        _gc = float(_g[2].item())
        _gs = float(_g[3].item())
        local_planner_state["goal_dir_dx"] = _gdx
        local_planner_state["goal_dir_dy"] = _gdy
        local_planner_state["goal_dir_cos"] = _gc
        local_planner_state["goal_dir_sin"] = _gs
        local_planner_state["goal_bearing_deg"] = _math2.degrees(_math2.atan2(_gs, _gc))
        local_planner_state["goal_distance"] = 10.0 * _math2.sqrt(_gdx**2 + _gdy**2)

      # Compute TRUE 4D goal direction from GT (for comparison)
      _true_gx = local_planner_state["true_goal_x"]
      _true_gy = local_planner_state["true_goal_y"]
      if abs(_true_gx) + abs(_true_gy) > 0.01:
        import math as _math3

        # Get true robot yaw for body-frame transform
        _rq = robot.data.root_link_quat_w
        _true_yaw = float(
          torch.atan2(
            2.0 * (_rq[0, 0] * _rq[0, 3] + _rq[0, 1] * _rq[0, 2]),
            1.0 - 2.0 * (_rq[0, 2] ** 2 + _rq[0, 3] ** 2),
          ).item()
        )
        _tc = _math3.cos(_true_yaw)
        _ts = _math3.sin(_true_yaw)
        _tbx = _true_gx * _tc + _true_gy * _ts
        _tby = -_true_gx * _ts + _true_gy * _tc
        _td = _math3.sqrt(_tbx**2 + _tby**2)
        _td_safe = max(_td, 0.1)
        local_planner_state["true_dir_dx"] = _tbx / 10.0
        local_planner_state["true_dir_dy"] = _tby / 10.0
        local_planner_state["true_dir_cos"] = _tbx / _td_safe
        local_planner_state["true_dir_sin"] = _tby / _td_safe

      # Compute hidden decay factor for display.
      # This mirrors the heuristic from LocalPlannerNetwork.forward.
      try:
        _ray_obs_name = "raycast"
        _ray_vals = local_planner_obs_dict.get(_ray_obs_name)
        if _ray_vals is not None and isinstance(_ray_vals, torch.Tensor):
          _rv = _ray_vals[0]
          if _rv.shape[-1] > 64:
            _rv = _rv[-64:]
          # Use median (quantile 0.5) of all rays for robust clearance
          _sorted_rv, _ = _rv.sort(dim=-1)
          _q_idx = min(len(_sorted_rv) // 2, len(_sorted_rv) - 1)
          _clearance = float(_sorted_rv[_q_idx].item()) * 4.0  # denorm → metres
          import math as _math4

          _decay = 0.9 + 0.1 / (1.0 + _math4.exp(-(3.0 - _clearance) * 3.0))
          local_planner_state["hidden_decay"] = _decay
      except Exception:
        pass

      # Get base policy action with MODIFIED observations
      actions = base_policy(obs)

      # Track statistics
      step_count[0] += 1
      total_correction[0] += correction.abs().mean().item() * args.correction_gain
      correction_count[0] += 1

      # Print stats periodically
      if step_count[0] % 100 == 0:
        avg_corr = (
          total_correction[0] / correction_count[0] if correction_count[0] > 0 else 0
        )
        raw_val = metrics["raw_correction"]
        scaled_corr = correction_val * args.correction_gain

        _b_dr = local_planner_state["goal_bearing_deg"]
        _b_true = local_planner_state["true_bearing_deg"]
        _d_dr = local_planner_state["goal_distance"]
        _d_true = local_planner_state["true_distance"]
        _gate: float = local_planner_state["goal_seek_gate"]
        _gate_s = "A" if _gate > 0.7 else ("G" if _gate < 0.3 else "B")
        _decay: float = local_planner_state["hidden_decay"]
        _gdx = local_planner_state["goal_dir_dx"]
        _gdy = local_planner_state["goal_dir_dy"]
        _gc = local_planner_state["goal_dir_cos"]
        _gs = local_planner_state["goal_dir_sin"]
        print(
          f"Step {step_count[0]}: "
          f"KG={local_planner_state['current_key_goal']} "
          f"|Δyaw|={avg_corr:.3f} "
          f"raw={raw_val:+.3f} final={scaled_corr:+.3f} "
          f"Gate={_gate:.2f}[{_gate_s}] "
          f"Decay={_decay:.3f} "
          f"Dir=({_gdx:+.2f},{_gdy:+.2f},{_gc:+.2f},{_gs:+.2f}) "
          f"Brg DR/True={_b_dr:+.0f}/{_b_true:+.0f}° "
          f"Dist={_d_dr:.1f}/{_d_true:.1f}m "
          f"vel={self.last_pred_lin_vel:+.2f}/{self.last_true_lin_vel:+.2f}m/s "
          f"H={metrics['hidden_state_norm']:.2f}"
        )
        total_correction[0] = 0.0
        correction_count[0] = 0

      return actions

  policy = LocalPlannerEnhancedPolicy()

  # Create overlay callback for native viewer
  # NOTE: MuJoCo mjr_overlay has a 500-char buffer limit per text column.
  # Keep both labels and values strings under 450 chars to avoid truncation.
  def local_planner_overlay_callback() -> tuple[str, str]:
    """Generate compact local planner status overlay text."""
    import math

    # --- pre-compute values ---
    orig_deg = math.degrees(local_planner_state["original_yaw"])
    raw_deg = math.degrees(local_planner_state["raw_correction"])
    final_deg = math.degrees(local_planner_state["corrected_yaw"])
    err_vel = local_planner_state["pred_lin_vel"] - local_planner_state["true_lin_vel"]
    # Bearing / distance
    g_brg = local_planner_state["goal_bearing_deg"]
    t_brg = local_planner_state["true_bearing_deg"]
    brg_err = (g_brg - t_brg + 180) % 360 - 180
    g_dist = local_planner_state["goal_distance"]
    t_dist = local_planner_state["true_distance"]
    d_err = g_dist - t_dist
    # DR state
    dr_est_gx = local_planner_state["dr_est_goal_x"]
    dr_est_gy = local_planner_state["dr_est_goal_y"]
    true_gx = local_planner_state["true_goal_x"]
    true_gy = local_planner_state["true_goal_y"]
    dr_goal_err = math.sqrt((dr_est_gx - true_gx) ** 2 + (dr_est_gy - true_gy) ** 2)
    # 4D goal direction
    gdx = local_planner_state["goal_dir_dx"]
    gdy = local_planner_state["goal_dir_dy"]
    gc = local_planner_state["goal_dir_cos"]
    gs = local_planner_state["goal_dir_sin"]
    tdx = local_planner_state["true_dir_dx"]
    tdy = local_planner_state["true_dir_dy"]
    tc = local_planner_state["true_dir_cos"]
    ts = local_planner_state["true_dir_sin"]
    decay_val = local_planner_state["hidden_decay"]

    labels = (
      "Local Planner"
      "\nStep / Key Goal"
      "\nPP Yaw (GT)"
      "\nLP Raw"
      "\nLP Final"
      "\nHidden N / Decay"
      "\n"
      "\n4D Dir DR"
      "\n4D Dir True"
      "\nBrg DR/True (err)"
      "\nDist DR/True (err)"
      "\n"
      "\nInit Goal"
      "\nDR Disp / Hdg"
      "\nDR Est Goal"
      "\nTrue Goal"
      "\nDR Goal Err"
      "\nVel True / Pred"
    )

    ig_x = local_planner_state["initial_goal_x"]
    ig_y = local_planner_state["initial_goal_y"]
    dr_x = local_planner_state["dr_x"]
    dr_y = local_planner_state["dr_y"]
    dr_hdg = local_planner_state["dr_heading_deg"]

    values = (
      " "
      f"\n{local_planner_state['step']} / {local_planner_state['current_key_goal']}"
      f"\n{orig_deg:+.1f}deg"
      f"\n{raw_deg:+.2f}deg ({local_planner_state['raw_correction']:+.4f})"
      f"\n{final_deg:+.1f}deg"
      f"\n{local_planner_state['hidden_norm']:.3f} / {decay_val:.3f}"
      "\n"
      f"\n({gdx:+.2f},{gdy:+.2f},{gc:+.2f},{gs:+.2f})"
      f"\n({tdx:+.2f},{tdy:+.2f},{tc:+.2f},{ts:+.2f})"
      f"\n{g_brg:+.0f} / {t_brg:+.0f}deg ({brg_err:+.0f})"
      f"\n{g_dist:.1f} / {t_dist:.1f}m ({d_err:+.1f})"
      "\n"
      f"\n({ig_x:+.1f},{ig_y:+.1f})m"
      f"\n({dr_x:+.2f},{dr_y:+.2f}) / {dr_hdg:+.1f}deg"
      f"\n({dr_est_gx:+.1f},{dr_est_gy:+.1f})m"
      f"\n({true_gx:+.1f},{true_gy:+.1f})m"
      f"\n{dr_goal_err:.2f}m"
      f"\n{local_planner_state['true_lin_vel']:+.2f} / {local_planner_state['pred_lin_vel']:+.2f} m/s ({err_vel:+.2f})"
    )
    return labels, values

  # Wrap environment for viewer compatibility
  wrapped_env = RslRlVecEnvWrapper(
    env, clip_actions=getattr(rl_cfg, "clip_actions", 1.0)
  )

  # Handle viewer selection
  viewer_type = args.viewer

  print(f"Starting visualization with {viewer_type} viewer...")
  print("Local planner corrections will be computed and displayed in console.")
  print("Press Ctrl+C or close the viewer to stop.")

  if viewer_type == "native":
    has_display = bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
      print("[WARN] No display detected, falling back to headless mode.")
      viewer_type = "none"

  if viewer_type == "native":
    NativeMujocoViewer(
      wrapped_env,
      policy,
      overlay_callback=local_planner_overlay_callback,
    ).run()
  elif viewer_type == "viser":
    ViserPlayViewer(wrapped_env, policy).run()
  elif viewer_type == "none":
    # Headless test mode - run for a limited number of steps
    print("Running in headless mode...")
    obs = wrapped_env.get_observations()
    for _ in range(500):
      actions = policy(obs)
      obs, _, _, _ = wrapped_env.step(actions)
    print("Headless test completed.")
  else:
    raise ValueError(f"Unknown viewer type: {viewer_type}")

  env.close()
  print("Test completed.")


def diagnose_locomotion(args: TrainLocalPlannerArgs) -> None:
  """Diagnose locomotion policy angular velocity capability via keyboard.

  Launches a keyboard terminal for controlling ``[target_velocity, delta_yaw]``
  and prints real-time commanded vs actual linear/angular velocities.  This is
  used to verify the locomotion policy's turning ability at high speeds.

  Key bindings (in the separate keyboard terminal):
    W / UP:    Increase forward speed (+10%)
    S / DOWN:  Decrease forward speed (-10%)
    Q:         Increase yaw command (turn left, +10%)
    E:         Decrease yaw command (turn right, -10%)
    SPACE:     Reset all to zero

  Args:
    args: Training arguments.  Only ``checkpoint``, ``task``, ``num_envs``,
      ``viewer``, and ``device`` are used.
  """
  import os as _os

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.velocity.mdp.keyboard_command import KeyboardListener
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  device = torch.device(args.device)
  configure_torch_backends()

  print(f"[Diagnose] Locomotion angular velocity diagnostic for: {args.task}")

  # Setup environment
  env, rl_cfg = setup_environment(args)

  # Load locomotion policy
  if not args.checkpoint or not os.path.isfile(args.checkpoint):
    raise ValueError(
      f"Locomotion checkpoint required for diagnose mode: {args.checkpoint}"
    )
  base_policy = load_policy(args.checkpoint, env, rl_cfg, device)

  # Find goal_command offset in the policy observation tensor
  obs_manager = env.observation_manager
  term_names = obs_manager.active_terms.get("actor", [])
  term_dims = obs_manager.group_obs_term_dim.get("actor", [])
  goal_cmd_offset: int | None = None
  offset = 0
  for term_name, dims in zip(term_names, term_dims, strict=False):
    if term_name == "goal_command":
      goal_cmd_offset = offset
      break
    offset += int(np.prod(dims))
  if goal_cmd_offset is None:
    print(f"[WARN] goal_command not found in terms: {term_names}, using fallback=93")
    goal_cmd_offset = 93
  print(f"[Diagnose] goal_command offset in actor obs: {goal_cmd_offset}")

  # Start keyboard listener — reuse KeyboardListener with
  # lin_x → target_velocity, ang_z → delta_yaw
  max_lin_vel = 2.5
  max_ang_vel = 1.5  # max_yaw_correction from LocalPlannerNetworkCfg
  listener = KeyboardListener(
    max_lin_vel_x=max_lin_vel,
    max_lin_vel_y=0.0,  # unused
    max_ang_vel_z=max_ang_vel,
    step_ratio=0.1,
  )
  listener.start()
  print(
    f"[Diagnose] Keyboard terminal launched.\n"
    f"  W/S: target velocity (max {max_lin_vel:.1f} m/s)\n"
    f"  Q/E: delta_yaw command (max ±{max_ang_vel:.1f} rad)\n"
    f"  SPACE: reset to zero"
  )

  # Diagnostics state
  diag_state = {
    "cmd_vel": 0.0,
    "cmd_yaw": 0.0,
    "actual_lin_vel_x": 0.0,
    "actual_ang_vel_z": 0.0,
    "step": 0,
  }

  def diag_overlay_callback() -> tuple[str, str]:
    """Overlay callback for native viewer."""
    cmd_v = diag_state["cmd_vel"]
    cmd_y = diag_state["cmd_yaw"]
    act_v = diag_state["actual_lin_vel_x"]
    act_w = diag_state["actual_ang_vel_z"]
    ratio = abs(act_w / cmd_y) if abs(cmd_y) > 0.05 else float("nan")
    ratio_str = f"{ratio:.2f}" if ratio == ratio else "N/A"
    labels = (
      "Locomotion Diagnostic"
      "\nCmd Velocity"
      "\nCmd Delta Yaw"
      "\nActual lin_vel_x"
      "\nActual ang_vel_z"
      "\nomega Ratio"
    )
    values = (
      f"\nStep {diag_state['step']}"
      f"\n{cmd_v:+.2f} m/s"
      f"\n{cmd_y:+.3f} rad"
      f"\n{act_v:+.2f} m/s"
      f"\n{act_w:+.3f} rad/s"
      f"\n{ratio_str}"
    )
    return labels, values

  # Wrap environment for viewer compatibility
  wrapped_env = RslRlVecEnvWrapper(
    env, clip_actions=getattr(rl_cfg, "clip_actions", 1.0)
  )

  # Create policy wrapper that overrides goal_command from keyboard
  class DiagnosticPolicy:
    def __init__(self, base_policy: Callable, listener: KeyboardListener):
      self._base = base_policy
      self._listener = listener

    def __call__(self, obs: Any) -> torch.Tensor:
      # Read keyboard targets
      target_lin_x, _target_lin_y, target_ang_z = self._listener.get_targets()

      # Override goal_command in actor observation
      has_actor = _dict_like_has_key(obs, "actor")
      if has_actor:
        policy_obs = _dict_like_get(obs, "actor").clone()
        policy_obs[:, goal_cmd_offset] = target_ang_z  # delta_yaw
        policy_obs[:, goal_cmd_offset + 1] = target_lin_x  # target_velocity
        _dict_like_set(obs, "actor", policy_obs)

      # Update diagnostic state
      robot = env.scene["robot"]
      diag_state["cmd_vel"] = target_lin_x
      diag_state["cmd_yaw"] = target_ang_z
      diag_state["actual_lin_vel_x"] = float(
        robot.data.root_link_lin_vel_b[0, 0].item()
      )
      diag_state["actual_ang_vel_z"] = float(
        robot.data.root_link_ang_vel_b[0, 2].item()
      )
      diag_state["step"] = diag_state.get("step", 0) + 1

      # Print diagnostics every 50 steps (~1 second at 50 Hz)
      if diag_state["step"] % 50 == 0:
        cmd_v = diag_state["cmd_vel"]
        cmd_y = diag_state["cmd_yaw"]
        act_v = diag_state["actual_lin_vel_x"]
        act_w = diag_state["actual_ang_vel_z"]
        ratio = abs(act_w / cmd_y) if abs(cmd_y) > 0.05 else float("nan")
        ratio_str = f"{ratio:.2f}" if ratio == ratio else "N/A"
        print(
          f"[Diag] Step {diag_state['step']:6d} | "
          f"Cmd: v={cmd_v:+.2f} m/s  Δyaw={cmd_y:+.3f} rad | "
          f"Act: v={act_v:+.2f} m/s  ω_z={act_w:+.3f} rad/s | "
          f"ω ratio={ratio_str}"
        )

      return self._base(obs)

  policy = DiagnosticPolicy(base_policy, listener)

  # Handle viewer selection
  viewer_type = args.viewer

  print(f"[Diagnose] Starting with {viewer_type} viewer...")
  print("Control the robot with the keyboard terminal. Press Ctrl+C to stop.")

  if viewer_type == "native":
    has_display = bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
      print("[WARN] No display detected, falling back to headless mode.")
      viewer_type = "none"

  if viewer_type == "native":
    NativeMujocoViewer(
      wrapped_env, policy, overlay_callback=diag_overlay_callback
    ).run()
  elif viewer_type == "viser":
    ViserPlayViewer(wrapped_env, policy).run()
  elif viewer_type == "none":
    # Headless mode — run until Ctrl+C
    print("Running in headless mode (console output only)...")
    obs = wrapped_env.get_observations()
    try:
      while True:
        actions = policy(obs)
        obs, _, _, _ = wrapped_env.step(actions)
    except KeyboardInterrupt:
      print("\nDiagnostic interrupted.")
  else:
    raise ValueError(f"Unknown viewer type: {viewer_type}")

  listener.stop()
  env.close()
  print("Diagnostic completed.")


def play_env(args: TrainLocalPlannerArgs) -> None:
  """Play/visualize the environment without the local planner.

  Args:
    args: Arguments.
  """
  import os as _os

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  device = torch.device(args.device)
  configure_torch_backends()

  print(f"Playing task: {args.task}")

  # Setup environment
  env, rl_cfg = setup_environment(args)

  # Load policy if checkpoint provided
  if args.checkpoint and os.path.isfile(args.checkpoint):
    base_policy = load_policy(args.checkpoint, env, rl_cfg, device)
  else:
    # Zero policy for visualization
    print("No checkpoint provided, using zero policy.")
    action_dim = env.action_space.shape[-1]

    def zero_policy(obs: object) -> torch.Tensor:
      return torch.zeros(args.num_envs, action_dim, device=device)

    base_policy = zero_policy

  policy = base_policy

  correction_extractor: PathPlannerCorrectionExtractor | None = None
  if args.checkpoint and os.path.isfile(args.checkpoint):
    if not args.raycast_obs_names:
      raise ValueError("raycast_obs_names must contain at least one observation key.")
    correction_extractor = PathPlannerCorrectionExtractor(env)
    correction_extractor.raycast_obs_name = args.raycast_obs_names[0]
    correction_extractor.raycast_obs_names = args.raycast_obs_names
    correction_extractor.target_label_mode = args.target_label_mode

    class TeacherLabelPolicy:
      """Policy wrapper that overrides goal-command yaw with teacher labels."""

      def __init__(self):
        self._goal_cmd_offset: int | None = None
        self._last_target_yaw: torch.Tensor | None = None

      def reset(self) -> None:
        assert correction_extractor is not None
        correction_extractor.reset()
        self._last_target_yaw = None

      def get_last_target_yaw(self) -> torch.Tensor | None:
        if self._last_target_yaw is None:
          return None
        return self._last_target_yaw.clone()

      def __call__(self, obs: object) -> torch.Tensor:
        assert correction_extractor is not None
        flat_obs = _flatten_tensor_observations(obs)
        if not flat_obs:
          return base_policy(obs)

        done = env.episode_length_buf == 0
        target, _ = correction_extractor.extract_correction(done, obs_dict=flat_obs)
        target_yaw = target[:, 0]
        self._last_target_yaw = target_yaw.detach().clone()

        has_actor_key = _dict_like_has_key(obs, "actor")
        if has_actor_key:
          if self._goal_cmd_offset is None:
            self._goal_cmd_offset = _find_goal_command_offset(env)
            print(f"[INFO] goal_command offset in policy obs: {self._goal_cmd_offset}")

          if self._goal_cmd_offset is not None:
            modified_actor_obs = _dict_like_get(obs, "actor").clone()
            modified_actor_obs[:, self._goal_cmd_offset] = target_yaw
            _dict_like_set(obs, "actor", modified_actor_obs)

        return base_policy(obs)

    policy = TeacherLabelPolicy()
    print(
      "[Play] Teacher control target override enabled: "
      f"target_label_mode={args.target_label_mode}"
    )

  overlay_callback: Callable[[], tuple[str, str] | None] | None = None
  if args.show_label_target_overlay:
    if correction_extractor is None or not hasattr(policy, "get_last_target_yaw"):
      raise ValueError(
        "show_label_target_overlay requires a locomotion checkpoint and "
        "teacher label policy support."
      )
    teacher_policy = cast(_TeacherLabelPolicyProtocol, policy)
    _attach_debug_vis(
      env,
      _make_label_overlay_debug_vis(
        env,
        target_yaw_getter=teacher_policy.get_last_target_yaw,
      ),
    )

    def _label_overlay_callback() -> tuple[str, str]:
      return (
        "Teacher Label Overlay\nMode\nFixed Scene",
        f" \n{args.target_label_mode}\n{args.fixed_scene or 'random'}",
      )

    overlay_callback = _label_overlay_callback

  wrapped_env = RslRlVecEnvWrapper(
    env, clip_actions=getattr(rl_cfg, "clip_actions", 1.0)
  )

  viewer_type = args.viewer
  print(f"Starting play mode with {viewer_type} viewer...")
  print("Press Ctrl+C or close the viewer to stop.")

  if viewer_type == "native":
    has_display = bool(_os.environ.get("DISPLAY") or _os.environ.get("WAYLAND_DISPLAY"))
    if not has_display:
      print("[WARN] No display detected, falling back to headless mode.")
      viewer_type = "none"

  if viewer_type == "native":
    NativeMujocoViewer(
      wrapped_env,
      policy,
      overlay_callback=overlay_callback,
    ).run()
  elif viewer_type == "viser":
    ViserPlayViewer(wrapped_env, policy).run()
  elif viewer_type == "none":
    obs = wrapped_env.get_observations()
    for step in range(500):
      with torch.no_grad():
        actions = policy(obs)
      obs, _, _, _ = wrapped_env.step(actions)
      if step % 100 == 0:
        print(f"Step {step}")
  else:
    raise ValueError(f"Unknown viewer type: {viewer_type}")

  env.close()
  print("Play completed.")


def main() -> None:
  """Main entry point."""
  args: TrainLocalPlannerArgs = tyro.cli(TrainLocalPlannerArgs, config=mjlab.TYRO_FLAGS)

  if args.mode == "train":
    train_local_planner(args)
  elif args.mode == "test":
    test_local_planner(args)
  elif args.mode == "play":
    play_env(args)
  elif args.mode == "diagnose":
    diagnose_locomotion(args)
  else:
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
  main()
