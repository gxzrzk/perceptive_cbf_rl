#!/usr/bin/env python3
"""Export alpha_t diagnostics for the low-speed wall local-planner teacher."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.tasks.steering.train_local_planner import (
  PathPlannerCorrectionExtractor,
  TrainLocalPlannerArgs,
  _find_goal_command_offset,
  _flatten_tensor_observations,
  load_policy,
  setup_environment,
)
from mjlab.tool.experiments.eval_utils import (
  DEFAULT_LOCOMOTION_CHECKPOINT,
  DEFAULT_RAYCAST_OBS_NAMES,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass
class AlphaTimeseriesArgs:
  """Arguments for alpha_t diagnostic export."""

  task: str = "Mjlab-StraightApproach-LowSpeed-G1-v6"
  """Low-speed wall task used for the diagnostic rollout."""

  checkpoint: str = DEFAULT_LOCOMOTION_CHECKPOINT
  """Locomotion checkpoint used for teacher-controlled rollout."""

  num_envs: int = 1
  """Parallel environments. The exported time series uses env 0."""

  steps: int = 1000
  """Number of control steps to record."""

  seed: int = 0
  """Environment seed."""

  device: str = "cuda"
  """Simulation device."""

  raycast_obs_names: tuple[str, ...] = DEFAULT_RAYCAST_OBS_NAMES
  """Raycast observation names used by the label extractor."""

  output_root: str = "logs/benchmark/alpha_timeseries"
  """Root directory for CSV and plot artifacts."""

  run_name: str = ""
  """Optional output subdirectory name."""


def _value(tensor: torch.Tensor | None, env_idx: int = 0) -> float:
  if tensor is None:
    return float("nan")
  return float(tensor.reshape(-1)[env_idx].detach().cpu().item())


def _write_plot(csv_path: Path, png_path: Path, pdf_path: Path) -> None:
  import matplotlib.pyplot as plt

  data: dict[str, list[float]] = {}
  with csv_path.open(encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      for key, value in row.items():
        data.setdefault(key, []).append(float(value) if value else float("nan"))
  fig, axes = plt.subplots(4, 1, figsize=(8.0, 7.5), sharex=True)
  axes[0].plot(data["time_s"], data["alpha_raw"], label="raw alpha", lw=1.0)
  axes[0].plot(data["time_s"], data["alpha"], label="smoothed alpha", lw=1.5)
  axes[0].set_ylabel("alpha")
  axes[0].set_ylim(-0.05, 1.05)
  axes[0].legend(loc="best", fontsize=8)

  axes[1].plot(data["time_s"], data["forward_clearance_m"], label="front", lw=1.2)
  axes[1].plot(
    data["time_s"],
    data["goal_side_clearance_m"],
    label="goal-side",
    lw=1.2,
  )
  axes[1].set_ylabel("clearance [m]")
  axes[1].legend(loc="best", fontsize=8)

  axes[2].plot(data["time_s"], data["yaw_to_waypoint"], label="waypoint", lw=1.2)
  axes[2].plot(data["time_s"], data["yaw_to_key_goal"], label="key goal", lw=1.2)
  axes[2].plot(data["time_s"], data["target_yaw"], label="blend", lw=1.5)
  axes[2].set_ylabel("yaw [rad]")
  axes[2].legend(loc="best", fontsize=8)

  axes[3].plot(data["time_s"], data["goal_side_blocked"], lw=1.2)
  axes[3].set_ylabel("side blocked")
  axes[3].set_xlabel("time [s]")
  axes[3].set_ylim(-0.05, 1.05)

  for axis in axes:
    axis.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(png_path, dpi=200)
  fig.savefig(pdf_path)
  plt.close(fig)


def main() -> None:
  args = tyro.cli(AlphaTimeseriesArgs, config=mjlab.TYRO_FLAGS)
  configure_torch_backends()
  device = torch.device(args.device)
  run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_dir = Path(args.output_root) / run_name
  run_dir.mkdir(parents=True, exist_ok=True)

  train_args = TrainLocalPlannerArgs(
    mode="test",
    task=args.task,
    checkpoint=args.checkpoint,
    num_envs=args.num_envs,
    device=args.device,
    seed=args.seed,
    raycast_obs_names=args.raycast_obs_names,
  )
  env, rl_cfg = setup_environment(train_args)
  base_policy = load_policy(args.checkpoint, env, rl_cfg, device)
  extractor = PathPlannerCorrectionExtractor(env)
  extractor.raycast_obs_name = args.raycast_obs_names[0]
  extractor.raycast_obs_names = args.raycast_obs_names
  extractor.reset()
  goal_cmd_offset = _find_goal_command_offset(env)

  obs, _ = env.reset(seed=args.seed)
  rows: list[dict[str, float | int]] = []
  for step in range(int(args.steps)):
    flat_obs = _flatten_tensor_observations(obs)
    done = env.episode_length_buf == 0
    target, _ = extractor.extract_correction(done, obs_dict=flat_obs)
    if goal_cmd_offset is not None and hasattr(obs, "keys") and "actor" in obs:
      modified_actor = obs["actor"].clone()
      modified_actor[:, goal_cmd_offset] = target[:, 0]
      obs["actor"] = modified_actor
    with torch.no_grad():
      action = base_policy(obs)
    rows.append(
      {
        "step": step,
        "time_s": float(step * env.step_dt),
        "alpha_raw": _value(extractor.last_alpha_raw),
        "alpha": _value(extractor.last_alpha),
        "forward_clearance_m": _value(extractor.last_forward_clearance_m),
        "goal_side_clearance_m": _value(extractor.last_goal_side_clearance_m),
        "goal_side_blocked": int(_value(extractor.last_goal_side_blocked) > 0.5),
        "yaw_to_waypoint": _value(extractor.last_yaw_to_waypoint),
        "yaw_to_key_goal": _value(extractor.last_yaw_to_key_goal),
        "target_yaw": _value(extractor.last_target_yaw),
      }
    )
    obs, _reward, _terminated, _timeouts, _extras = env.step(action)

  env.close()

  csv_path = run_dir / "alpha_timeseries.csv"
  with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  png_path = run_dir / "alpha_timeseries.png"
  pdf_path = run_dir / "alpha_timeseries.pdf"
  try:
    _write_plot(csv_path, png_path, pdf_path)
  except Exception as exc:
    print(f"[WARN] Could not render alpha plot: {type(exc).__name__}: {exc}")

  print(f"Outputs written to: {run_dir}")
  print(f"  csv: {csv_path}")
  print(f"  png: {png_path}")
  print(f"  pdf: {pdf_path}")


if __name__ == "__main__":
  main()
