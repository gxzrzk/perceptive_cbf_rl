#!/usr/bin/env python3
"""Backbone-only benchmark for FixedSequentialGoalCommand tasks.

This script evaluates the frozen locomotion policy without a learned local
planner.  It is useful for teacher/waypoint quality checks and for the R1
backbone-only paper baseline.

Examples:
  uv run python -m mjlab.tool.experiments.benchmark_play \\
    --task Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6 \\
    --num-episodes 200 --num-envs 256 --seed 0

  # Backward-compatible wrapper:
  uv run python -m mjlab.tool.benchmark_play --help
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime

import torch
import tyro

import mjlab
from mjlab.tool.experiments.eval_utils import (
  DEFAULT_LOCOMOTION_CHECKPOINT,
  EvalEpisodeRecord,
  compute_collision_table,
  compute_summary,
  record_to_dict,
  run_policy_eval,
  write_json,
)

EpisodeRecord = EvalEpisodeRecord


@dataclass
class PlayBenchmarkArgs:
  """Teacher-data quality-check arguments."""

  task: str = "Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6"
  """Task name to evaluate (any task using FixedSequentialGoalCommand)."""

  checkpoint: str = DEFAULT_LOCOMOTION_CHECKPOINT
  """Path to the locomotion policy checkpoint."""

  num_envs: int = 256
  """Number of parallel environments."""

  num_episodes: int = 200
  """Number of goal-collection episodes (statistics sample size)."""

  device: str = "cuda"
  """Execution device ("cuda" or "cpu")."""

  seed: int | None = None
  """Random seed; None keeps the environment's default randomness."""

  play_mode: bool = False
  """Whether to use the play-mode env config; False keeps train behavior."""

  log_dir: str = "logs/benchmark"
  """Output directory for structured JSON logs."""

  save_log: bool = True
  """Whether to save a JSON log."""


def run_play_benchmark(
  task: str,
  checkpoint: str,
  num_envs: int,
  num_episodes: int,
  device: torch.device,
  play_mode: bool = False,
  seed: int | None = None,
) -> tuple[list[EpisodeRecord], int, list[int]]:
  """Run the backbone-only benchmark."""
  mode = "play" if play_mode else "train"
  result = run_policy_eval(
    task=task,
    locomotion_checkpoint=checkpoint,
    num_envs=num_envs,
    num_episodes=num_episodes,
    device=device,
    seed=seed,
    mode=mode,
  )
  return result.records, result.num_goals, result.key_goal_indices


def compute_summary_stats(records: list[EpisodeRecord]) -> dict[str, float]:
  """Return summary metrics with human-readable display labels."""
  summary = compute_summary(records)
  return {
    "Final goal arrival rate (%)": summary.get("success_rate", float("nan")),
    "Key goal 1 arrival rate (%)": summary.get("key_goal1_rate", float("nan")),
    "Mean waypoints completed": summary.get("mean_waypoints_completed", float("nan")),
    "Collision termination (%)": summary.get("collision_rate", float("nan")),
    "Timeout rate (%)": summary.get("timeout_rate", float("nan")),
    "Timeout dist. to final (m)": summary.get("timeout_dist_m", float("nan")),
    "Mean episode steps": summary.get("mean_episode_steps", float("nan")),
    "mean_action_acc": summary.get("mean_action_acc", float("nan")),
    "action_rate_l2": summary.get("action_rate_l2", float("nan")),
    "base_ang_vel_var": summary.get("base_ang_vel_var", float("nan")),
    "gait_preservation_score": summary.get("gait_preservation_score", float("nan")),
  }


def _cjk_len(s: str) -> int:
  width = 0
  for char in s:
    if (
      "\u4e00" <= char <= "\u9fff"
      or "\u3000" <= char <= "\u303f"
      or "\uff00" <= char <= "\uffef"
      or "\u2600" <= char <= "\u26ff"
    ):
      width += 2
    else:
      width += 1
  return width


def _rpad(s: str, width: int) -> str:
  return s + " " * max(0, width - _cjk_len(s))


def _lpad(s: str, width: int) -> str:
  return " " * max(0, width - _cjk_len(s)) + s


def print_results(
  task: str,
  checkpoint: str,
  num_episodes: int,
  seed: int | None,
  summary: dict[str, float],
  collision_rows: list[dict],
  num_goals: int,
  key_goal_indices: list[int],
) -> None:
  """Print the summary statistics and the collision-distribution table."""
  sep_len = 72
  print()
  print("=" * sep_len)
  print("  Backbone-only Benchmark Report")
  print("=" * sep_len)
  print(f"  Task:       {task}")
  print(f"  Checkpoint: {checkpoint}")
  print(f"  Seed:       {seed}")
  print(f"  Episodes:   {num_episodes}")
  print(f"  Waypoints:  {num_goals} (key goals: WP{key_goal_indices})")
  print()

  print("── Summary Metrics ──")
  for key, value in summary.items():
    if math.isnan(value):
      val_str = "N/A"
    elif "%" in key:
      val_str = f"{value:.1f}%"
    elif "steps" in key:
      val_str = f"{value:.0f}"
    else:
      val_str = (
        f"{value:.4f}"
        if key.startswith(("mean_", "action_", "base_", "gait_"))
        else f"{value:.2f}"
      )
    print(f"  {_rpad(key, 28)} {_lpad(val_str, 14)}")

  print()
  total_collisions = sum(row["collision_count"] for row in collision_rows)
  print(f"── Collisions: {total_collisions} total, {num_episodes} episodes ──")
  headers = ["Waypoint", "Collisions", "Rate/eps", "Share", "Mean X(m)", "Mean Y(m)"]
  widths = [10, 10, 12, 10, 10, 10]
  print("  " + " │ ".join(_rpad(headers[i], widths[i]) for i in range(len(headers))))
  print("  " + "─" * (sum(widths) + 5 * 3))

  for row in collision_rows:
    if row["collision_count"] == 0 and total_collisions > 0:
      continue
    mean_x = row["mean_x"]
    mean_y = row["mean_y"]
    cells = [
      row["label"],
      str(row["collision_count"]),
      f"{row['collision_rate_pct']:.1f}%",
      f"{row['collision_pct_of_all']:.1f}%",
      "N/A" if math.isnan(mean_x) else f"{mean_x:.2f}",
      "N/A" if math.isnan(mean_y) else f"{mean_y:.2f}",
    ]
    print("  " + " │ ".join(_rpad(cells[i], widths[i]) for i in range(len(cells))))
  if total_collisions == 0:
    print("  (no collision events)")
  print()


def save_benchmark_log(
  log_dir: str,
  args: PlayBenchmarkArgs,
  records: list[EpisodeRecord],
  num_goals: int,
  key_goal_indices: list[int],
  summary: dict[str, float],
  collision_rows: list[dict],
) -> str:
  """Save structured benchmark JSON."""
  os.makedirs(log_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = os.path.join(log_dir, f"benchmark_play_{timestamp}.json")
  write_json(
    path,
    {
      "timestamp": timestamp,
      "task": args.task,
      "checkpoint": args.checkpoint,
      "num_envs": args.num_envs,
      "num_episodes": args.num_episodes,
      "seed": args.seed,
      "num_goals": num_goals,
      "key_goal_indices": key_goal_indices,
      "summary": summary,
      "collision_table": collision_rows,
      "episodes": [record_to_dict(record) for record in records],
    },
  )
  print(f"Log saved: {path}")
  return path


def main() -> None:
  args: PlayBenchmarkArgs = tyro.cli(PlayBenchmarkArgs, config=mjlab.TYRO_FLAGS)
  device = torch.device(args.device)

  if not args.checkpoint or not os.path.isfile(args.checkpoint):
    print(f"Error: checkpoint file not found: {args.checkpoint}")
    raise SystemExit(1)

  print()
  print(f"{'─' * 60}")
  print(f"  Task:       {args.task}")
  print(f"  Checkpoint: {args.checkpoint}")
  print(f"  Episodes:   {args.num_episodes}  Envs: {args.num_envs}  Seed: {args.seed}")
  print(f"{'─' * 60}")

  records, num_goals, key_goal_indices = run_play_benchmark(
    task=args.task,
    checkpoint=args.checkpoint,
    num_envs=args.num_envs,
    num_episodes=args.num_episodes,
    device=device,
    play_mode=args.play_mode,
    seed=args.seed,
  )
  summary = compute_summary_stats(records)
  collision_rows = compute_collision_table(records, num_goals, key_goal_indices)
  print_results(
    task=args.task,
    checkpoint=args.checkpoint,
    num_episodes=len(records),
    seed=args.seed,
    summary=summary,
    collision_rows=collision_rows,
    num_goals=num_goals,
    key_goal_indices=key_goal_indices,
  )
  if args.save_log:
    save_benchmark_log(
      args.log_dir,
      args,
      records,
      num_goals,
      key_goal_indices,
      summary,
      collision_rows,
    )


if __name__ == "__main__":
  main()
