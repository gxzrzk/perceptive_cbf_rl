"""Curriculum terms for AMP tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def goal_radius_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  start_radius: float = 1.0,
  final_radius: float = 4.0,
  steps_to_full: float = 40_000.0,
) -> float:
  """Grow the goal-sampling radius from ``start_radius`` to ``final_radius``.

  Progress is measured by ``env.common_step_counter`` -- the global env-step counter,
  which increments once per env step (~``num_steps_per_env`` per training iteration,
  24 by default) and, crucially, is **never reset on episode resets**. The radius ramps
  linearly to ``final_radius`` over the first ``steps_to_full`` env steps (e.g. 40k
  steps / 24 per iter ~= 1.7k iterations), then holds. Returns the current max radius
  for logging.

  (The previous proxy ``command.command_counter`` was wrong here: ``CommandTerm.reset``
  zeroes it on every episode reset, so it plateaued at the steady-state per-episode
  resample count (~2.4) and the radius never grew past ~0.5 m.)
  """
  command = env.command_manager.get_term(command_name)
  frac = min(env.common_step_counter / steps_to_full, 1.0)
  new_max = start_radius + frac * (final_radius - start_radius)
  command.cfg.radius = (command.cfg.radius[0], new_max)
  return new_max


def classroom_training_iteration(env: ManagerBasedRlEnv) -> float:
  """Training progress in rollout iterations, anchored when a runner resumes."""
  steps = getattr(env, "_classroom_rollout_steps", 24)
  return getattr(env, "_classroom_iteration_anchor", 0.) + (
    env.common_step_counter - getattr(env, "_classroom_step_anchor", 0)) / steps


def classroom_throw_interval_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None = None,
  warmup_iterations: int = 5000,
  ramp_iterations: int = 3000,
  start_interval: tuple[float, float] = (4., 8.),
  final_interval: tuple[float, float] = (2., 5.),
) -> dict[str, float]:
  """Keep sparse throws during warmup, then linearly tighten both throw timers.

  Called before reset events and before the step throw event. Existing countdowns
  finish normally; only newly sampled countdowns use the updated interval.
  """
  if (warmup_iterations < 0 or ramp_iterations <= 0
      or any(not math.isfinite(v) for v in (*start_interval, *final_interval))
      or not 0 < start_interval[0] <= start_interval[1]
      or not 0 < final_interval[0] <= final_interval[1]
      or any(end > start for start, end in zip(start_interval, final_interval))):
    raise ValueError("Invalid classroom throw curriculum parameters")
  iteration = classroom_training_iteration(env)
  fraction = min(max((iteration - warmup_iterations) / ramp_iterations, 0.), 1.)
  interval = tuple(start + fraction * (end - start)
                   for start, end in zip(start_interval, final_interval))
  for name in ("reset_dodge_state", "throw_ball_on_dwell"):
    env.event_manager.get_term_cfg(name).params["throw_interval_range"] = interval
  return {"iteration": iteration, "progress": fraction,
          "interval_min_s": interval[0], "interval_max_s": interval[1]}
