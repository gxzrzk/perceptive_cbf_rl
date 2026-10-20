"""Play an exported ONNX policy inside an mjlab environment.

This helper is meant for quick visual inspection of exported velocity ONNX
policies produced by ``uv run train`` (see the ``onnx`` directory of a run).

Example usage:

  uv run python src/mjlab/tool/play_onnx.py
  uv run python src/mjlab/tool/play_onnx.py --viewer viser
  uv run python src/mjlab/tool/play_onnx.py --viewer none --max_steps 200
  uv run python src/mjlab/tool/play_onnx.py \
    --onnx_file path/to/policy.onnx \
    --task Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

try:
  import yaml
except ImportError:  # pragma: no cover - optional fallback only
  yaml = None


DEFAULT_ONNX_FILE = Path(
  "logs/rsl_rl/g1_velocity_raycast_h1/<timestamp>/<timestamp>.onnx"
)
DEFAULT_TASK_ID = "Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6"

# Map a backbone checkpoint's recorded max_iterations to the curriculum task it
# was trained with (cumulative iteration counts from curriculum_train_v6.sh).
_CURRICULUM_TASKS_BY_MAX_ITERS = {
  4000: "Mjlab-CircularGoal-Flat-G1Rw-Pure-v6",
  8000: "Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6",
  12500: "Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6",
}


@dataclass(frozen=True)
class PlayOnnxConfig:
  onnx_file: Path = DEFAULT_ONNX_FILE
  task: str | None = None
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser", "none"] = "auto"
  max_steps: int = 200


class OnnxPolicy:
  """Small adapter that turns an ONNX model into a viewer-compatible policy."""

  def __init__(
    self,
    onnx_file: Path,
    actor_obs_groups: tuple[str, ...],
  ) -> None:
    self.onnx_file = onnx_file.resolve()
    self.actor_obs_groups = actor_obs_groups
    self.providers = self._resolve_providers()
    self.session = ort.InferenceSession(str(self.onnx_file), providers=self.providers)

    inputs = self.session.get_inputs()
    outputs = self.session.get_outputs()
    if len(inputs) != 1:
      raise ValueError(
        f"Expected a single ONNX input, got {len(inputs)} for {self.onnx_file}."
      )
    if len(outputs) != 1:
      raise ValueError(
        f"Expected a single ONNX output, got {len(outputs)} for {self.onnx_file}."
      )

    self.input_name = inputs[0].name
    self.output_name = outputs[0].name
    self.expected_obs_dim = self._shape_last_dim(inputs[0].shape)
    self.expected_action_dim = self._shape_last_dim(outputs[0].shape)

  @staticmethod
  def _resolve_providers() -> list[str]:
    available = set(ort.get_available_providers())
    providers: list[str] = []
    for provider in ("CUDAExecutionProvider", "CPUExecutionProvider"):
      if provider in available:
        providers.append(provider)
    if providers:
      return providers
    return ["CPUExecutionProvider"]

  @staticmethod
  def _shape_last_dim(shape: list[int | str | None]) -> int | None:
    if not shape:
      return None
    last_dim = shape[-1]
    return last_dim if isinstance(last_dim, int) else None

  def _flatten_obs(self, obs: Any) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    missing: list[str] = []
    for group in self.actor_obs_groups:
      try:
        value = obs[group]
      except Exception:
        missing.append(group)
        continue
      if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
      if value.ndim == 1:
        value = value.unsqueeze(0)
      parts.append(value)

    if missing:
      available = []
      keys_fn = getattr(obs, "keys", None)
      if callable(keys_fn):
        try:
          raw_keys = keys_fn()
          if isinstance(raw_keys, Iterable):
            available = sorted(str(key) for key in raw_keys)
        except Exception:
          available = []
      raise KeyError(
        "Missing actor observation groups for ONNX policy: "
        f"{missing}. Available groups: {available}"
      )
    if not parts:
      raise ValueError("No actor observation tensors were available for ONNX policy.")

    flat_obs = torch.cat(parts, dim=-1)
    if (
      self.expected_obs_dim is not None and flat_obs.shape[-1] != self.expected_obs_dim
    ):
      raise ValueError(
        f"ONNX expects obs dim {self.expected_obs_dim}, but environment produced "
        f"{flat_obs.shape[-1]} from groups {self.actor_obs_groups}."
      )
    return flat_obs.to(dtype=torch.float32)

  def validate(self, obs: Any, expected_action_dim: int) -> None:
    flat_obs = self._flatten_obs(obs)
    if (
      self.expected_action_dim is not None
      and self.expected_action_dim != expected_action_dim
    ):
      raise ValueError(
        f"ONNX action dim {self.expected_action_dim} does not match environment "
        f"action dim {expected_action_dim}."
      )

    _ = self._run_batch(flat_obs)

  def _run_batch(self, flat_obs: torch.Tensor) -> torch.Tensor:
    obs_np = flat_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    outputs: list[np.ndarray] = []
    for env_idx in range(obs_np.shape[0]):
      action_value = self.session.run(
        [self.output_name],
        {self.input_name: obs_np[env_idx : env_idx + 1]},
      )[0]
      if not isinstance(action_value, np.ndarray):
        action_np = np.asarray(action_value, dtype=np.float32)
      else:
        action_np = action_value.astype(np.float32, copy=False)
      outputs.append(action_np.astype(np.float32, copy=False))
    actions_np = np.concatenate(outputs, axis=0)
    return torch.from_numpy(actions_np).to(device=flat_obs.device)

  def __call__(self, obs: Any) -> torch.Tensor:
    flat_obs = self._flatten_obs(obs)
    return self._run_batch(flat_obs)

  def reset(self, dones: torch.Tensor | None = None) -> None:
    del dones


def _infer_task_from_agent_yaml(onnx_file: Path) -> str | None:
  if yaml is None:
    return None

  agent_yaml = onnx_file.resolve().parent / "params" / "agent.yaml"
  if not agent_yaml.exists():
    return None

  try:
    data = yaml.unsafe_load(agent_yaml.read_text(encoding="utf-8"))
  except Exception as exc:
    print(f"[WARN] Failed to parse {agent_yaml}: {exc}")
    return None

  if not isinstance(data, dict):
    return None
  if data.get("experiment_name") != "g1_velocity_raycast_h1":
    return None

  max_iterations = data.get("max_iterations")
  if not isinstance(max_iterations, int):
    return None
  return _CURRICULUM_TASKS_BY_MAX_ITERS.get(max_iterations)


def _resolve_task_id(cfg: PlayOnnxConfig) -> str:
  if cfg.task is not None:
    return cfg.task

  inferred = _infer_task_from_agent_yaml(cfg.onnx_file)
  if inferred is not None:
    print(f"[INFO] Inferred task from run config: {inferred}")
    return inferred

  print(f"[INFO] Falling back to default task: {DEFAULT_TASK_ID}")
  return DEFAULT_TASK_ID


def _resolve_viewer(viewer: Literal["auto", "native", "viser", "none"]) -> str:
  if viewer != "auto":
    return viewer
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


def _build_overlay_callback(env: RslRlVecEnvWrapper):
  from mjlab.tasks.velocity.mdp.reachable_goal_command import FixedSequentialGoalCommand

  real_env = env.unwrapped
  if not hasattr(real_env, "command_manager"):
    return None

  for term in real_env.command_manager._terms.values():
    if isinstance(term, FixedSequentialGoalCommand):
      print("[INFO] Goal tracking overlay enabled.")
      return term.get_overlay_info
  return None


def _run_headless(
  env: RslRlVecEnvWrapper,
  policy: OnnxPolicy,
  max_steps: int,
) -> None:
  if max_steps <= 0:
    raise ValueError("max_steps must be > 0 when viewer is set to 'none'.")

  for step_idx in range(max_steps):
    with torch.no_grad():
      obs = env.get_observations()
      actions = policy(obs)
      _, reward, dones, _ = env.step(actions)

    if step_idx in {0, max_steps - 1} or (step_idx + 1) % 50 == 0:
      mean_reward = float(reward.float().mean().item())
      done_count = int(dones.sum().item())
      print(
        f"[HEADLESS] step={step_idx + 1}/{max_steps} "
        f"mean_reward={mean_reward:.4f} done={done_count}"
      )


def main() -> None:
  import mjlab.tasks  # noqa: F401

  configure_torch_backends()
  args = tyro.cli(PlayOnnxConfig, config=mjlab.TYRO_FLAGS)

  onnx_file = args.onnx_file.resolve()
  if not onnx_file.exists():
    raise FileNotFoundError(f"ONNX file not found: {onnx_file}")
  if args.num_envs <= 0:
    raise ValueError("num_envs must be >= 1.")

  task_id = _resolve_task_id(args)
  available_tasks = set(list_tasks())
  if task_id not in available_tasks:
    raise ValueError(
      f"Unknown task '{task_id}'. Pass --task explicitly from registered tasks."
    )

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = args.num_envs

  actor_obs_groups = tuple(agent_cfg.obs_groups["actor"])
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  policy = OnnxPolicy(onnx_file, actor_obs_groups=actor_obs_groups)
  policy.validate(
    env.get_observations(),
    expected_action_dim=env.unwrapped.action_manager.total_action_dim,
  )

  print(f"[INFO] Loaded ONNX: {onnx_file}")
  print(f"[INFO] Task: {task_id}")
  print(f"[INFO] Device: {device}")
  print(f"[INFO] ONNX providers: {policy.providers}")
  print(f"[INFO] Actor obs groups: {actor_obs_groups}")

  resolved_viewer = _resolve_viewer(args.viewer)
  try:
    if resolved_viewer == "none":
      _run_headless(env, policy, max_steps=args.max_steps)
      return

    overlay_callback = _build_overlay_callback(env)
    if resolved_viewer == "native":
      NativeMujocoViewer(
        env,
        policy,
        overlay_callback=overlay_callback,
      ).run()
      return
    if resolved_viewer == "viser":
      ViserPlayViewer(env, policy).run()
      return
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
