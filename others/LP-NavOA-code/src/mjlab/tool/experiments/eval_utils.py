#!/usr/bin/env python3
"""Shared utilities for local-navigation benchmark scripts.

The helpers in this module intentionally keep the benchmark policy surface
small: every evaluation condition ultimately runs a locomotion policy in a
``FixedSequentialGoalCommand`` task, optionally replacing the heading command
with a learned local planner.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

from mjlab.tasks.steering.local_planner_inference import (
  LocalPlannerInference,
  LocalPlannerInferenceCfg,
)
from mjlab.tasks.steering.train_local_planner import (
  TrainLocalPlannerArgs,
  _apply_checkpoint_cfg_to_inference_cfg,
  _load_local_planner_network_for_inference,
  load_policy,
  setup_environment,
)
from mjlab.tasks.velocity.mdp.reachable_goal_command import (
  FixedSequentialGoalCommand,
  ReachableGoalCommand,
)
from mjlab.utils import random as random_utils
from mjlab.utils.torch import configure_torch_backends

# Placeholder: point this at the Stage 3 backbone checkpoint you trained with
# `uv run train Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6`, or pass
# --locomotion-checkpoint explicitly.
DEFAULT_LOCOMOTION_CHECKPOINT = (
  "logs/rsl_rl/g1_velocity_raycast_h1/<run>/model_<step>.pt"
)
DEFAULT_RAYCAST_OBS_NAMES = ("raycast", "raycast_left", "raycast_right")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_DIRECT_CALIBRATION_QUANTILE = 0.95
DEFAULT_DIRECT_SENSITIVITY_QUANTILES = (0.9, 0.95, 0.975)
DEFAULT_ARRIVAL_TARGET_RATE = 0.98
DEFAULT_ENVELOPE_TARGET_RATE = 0.03
DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS = (0.85, 0.9, 0.925, 0.95, 0.975)


@dataclass(frozen=True)
class TaskSpec:
  """Task pair used by paper evaluation."""

  label: str
  pp_task: str
  nopp_task: str
  local_planner_checkpoint: str
  odom_local_planner_checkpoint: str
  wall_y: float
  direct_max_steps: int


V6_TASK_SPECS: tuple[TaskSpec, ...] = (
  TaskSpec(
    label="Low-speed",
    pp_task="Mjlab-StraightApproach-LowSpeed-G1-v6",
    nopp_task="Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6",
    local_planner_checkpoint="logs/steering/local_planner_low_speed_v6_511_final.pt",
    odom_local_planner_checkpoint=(
      "logs/steering/local_planner_low_speed_v6_511_odom_final.pt"
    ),
    wall_y=6.0,
    direct_max_steps=1200,
  ),
  TaskSpec(
    label="Indoor",
    pp_task="Mjlab-IndoorStraightApproach-G1-v6",
    nopp_task="Mjlab-IndoorStraightApproach-NoPP-G1-v6",
    local_planner_checkpoint=(
      "logs/steering/local_planner_indoor_approach_v6_511_final.pt"
    ),
    odom_local_planner_checkpoint=(
      "logs/steering/local_planner_indoor_approach_v6_511_odom_final.pt"
    ),
    wall_y=5.0,
    direct_max_steps=1300,
  ),
)


@dataclass
class EvalEpisodeRecord:
  """A single episode result with paper-table and diagnostic metrics."""

  sequence_complete: bool
  key_goal1_reached: bool
  waypoints_completed: int
  hard_terminated: bool
  timed_out: bool
  dist_to_final: float
  episode_steps: int
  collision_waypoint_idx: int = -1
  collision_pos_x_rel: float = float("nan")
  collision_pos_y_rel: float = float("nan")
  actual_velocity: float = float("nan")
  target_velocity_cmd: float = float("nan")
  episode_time_s: float = float("nan")
  objective_path_length_m: float = float("nan")
  objective_budget_time_s: float = float("nan")
  objective_budget_steps: int = -1
  objective_budget_available: bool = False
  is_objective_on_time_arrival: bool | None = None
  is_objective_late_arrival: bool | None = None
  direct_budget_steps: int = -1
  is_wandering_success: bool | None = None
  is_direct_success: bool | None = None
  is_late_arrival: bool | None = None
  is_on_time_arrival: bool | None = None
  body_contact_ever: bool = False
  feet_obstacle_contact_ever: bool = False
  safety_envelope_brake_ever: bool = False
  safety_envelope_brake_steps: int = 0
  safety_envelope_brake_time_s: float = 0.0
  safety_envelope_brake_steps_by_threshold: dict[str, int] = field(default_factory=dict)
  safety_envelope_brake_time_s_by_threshold: dict[str, float] = field(
    default_factory=dict
  )
  brush_contact_ever: bool = False
  max_contact_force: float = float("nan")
  local_planner_correction_at_end: float = float("nan")
  min_raycast_at_end: float = float("nan")
  spawn_x_offset: float = float("nan")
  min_raycast_ever: float = float("nan")
  mean_action_acc: float = float("nan")
  action_rate_l2: float = float("nan")
  base_ang_vel_var: float = float("nan")
  gait_preservation_score: float = float("nan")
  trajectory_xy: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class EvalRunResult:
  """Records and metadata for one task/method/seed run."""

  records: list[EvalEpisodeRecord]
  num_goals: int
  key_goal_indices: list[int]
  elapsed_s: float


def set_eval_seed(seed: int | None) -> None:
  """Seed Python, NumPy, Torch, and project RNG helpers."""
  if seed is None:
    return
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  random_utils.seed_rng(seed)


def sanitize_for_json(value: object) -> object:
  """Convert NaN/inf and tuples into JSON-safe values."""
  if isinstance(value, float):
    if math.isnan(value):
      return None
    if math.isinf(value):
      return "Inf" if value > 0 else "-Inf"
    return value
  if isinstance(value, tuple):
    return [sanitize_for_json(v) for v in value]
  if isinstance(value, list):
    return [sanitize_for_json(v) for v in value]
  if isinstance(value, dict):
    return {str(k): sanitize_for_json(v) for k, v in value.items()}
  return value


def write_json(path: str | os.PathLike[str], data: object) -> None:
  """Write JSON with benchmark-friendly NaN handling."""
  with open(path, "w", encoding="utf-8") as f:
    json.dump(sanitize_for_json(data), f, ensure_ascii=False, indent=2)


def append_jsonl(path: str | os.PathLike[str], rows: list[dict]) -> None:
  """Append JSONL rows."""
  with open(path, "a", encoding="utf-8") as f:
    for row in rows:
      f.write(json.dumps(sanitize_for_json(row), ensure_ascii=False) + "\n")


def record_to_dict(record: EvalEpisodeRecord) -> dict:
  """Dataclass to JSON-safe dict."""
  return sanitize_for_json(asdict(record))  # type: ignore[return-value]


def record_from_dict(data: dict[str, Any]) -> EvalEpisodeRecord:
  """Load an episode record from a JSON-safe dict, tolerating old outputs."""
  field_names = {f.name for f in fields(EvalEpisodeRecord)}
  kwargs = {key: value for key, value in data.items() if key in field_names}
  for key in (
    "collision_pos_x_rel",
    "collision_pos_y_rel",
    "local_planner_correction_at_end",
    "min_raycast_at_end",
    "spawn_x_offset",
    "min_raycast_ever",
  ):
    if kwargs.get(key) is None:
      kwargs[key] = float("nan")
  for key in (
    "actual_velocity",
    "target_velocity_cmd",
    "episode_time_s",
    "objective_path_length_m",
    "objective_budget_time_s",
    "max_contact_force",
    "mean_action_acc",
    "action_rate_l2",
    "base_ang_vel_var",
    "gait_preservation_score",
  ):
    if kwargs.get(key) is None:
      kwargs[key] = float("nan")
  if kwargs.get("trajectory_xy") is None:
    kwargs["trajectory_xy"] = []
  return EvalEpisodeRecord(**kwargs)


def direct_quantile_key(quantile: float) -> str:
  """Return a compact stable key for a direct-budget quantile."""
  value = int(round(float(quantile) * 1000.0))
  return f"q{value}"


def direct_metric_suffix(quantile: float) -> str:
  """Return metric suffix used for direct/wandering sensitivity summaries."""
  return direct_quantile_key(quantile)


def _safe_target_velocity(record: EvalEpisodeRecord) -> float:
  value = float(record.target_velocity_cmd)
  if not math.isfinite(value) or value <= 1e-6:
    return 1.0
  return value


def _record_effective_distance(record: EvalEpisodeRecord) -> float:
  """Velocity-normalized episode duration in metres."""
  time_s = float(record.episode_time_s)
  if not math.isfinite(time_s):
    time_s = float(record.episode_steps) * 0.02
  return float(time_s * _safe_target_velocity(record))


def calibrate_direct_budgets(
  records: list[EvalEpisodeRecord],
  *,
  quantiles: tuple[float, ...] = DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
  min_success_episodes: int = 5,
) -> dict[str, float]:
  """Calibrate direct-success budgets from T1 successful episodes.

  The budget is stored as velocity-normalized distance:
  ``episode_time_s * target_velocity_cmd``.  Applying it to another episode
  converts the distance back into a per-episode step budget using that
  episode's target velocity.
  """
  effective_distances = np.array(
    [
      _record_effective_distance(record)
      for record in records
      if record.sequence_complete
    ],
    dtype=float,
  )
  effective_distances = effective_distances[np.isfinite(effective_distances)]
  if effective_distances.size < min_success_episodes:
    return {}
  return {
    direct_quantile_key(q): float(np.quantile(effective_distances, float(q)))
    for q in quantiles
  }


def calibrate_arrival_budget_for_target(
  records: list[EvalEpisodeRecord],
  *,
  target_rate: float = DEFAULT_ARRIVAL_TARGET_RATE,
  min_success_episodes: int = 5,
) -> dict[str, float | bool | int]:
  """Choose the smallest T1 budget that gives target on-time arrival rate.

  The target is measured against all episodes, not just successful episodes.
  If the T1 success rate itself is below the target, the widest successful
  budget is used and ``target_reached`` is false.
  """
  total = len(records)
  if total == 0:
    return {
      "budget_effective_distance": float("nan"),
      "target_rate": float(target_rate),
      "target_reached": False,
      "success_count": 0,
      "total_count": 0,
      "on_time_count": 0,
    }
  effective_distances = sorted(
    _record_effective_distance(record)
    for record in records
    if record.sequence_complete and math.isfinite(_record_effective_distance(record))
  )
  success_count = len(effective_distances)
  if success_count < int(min_success_episodes):
    return {
      "budget_effective_distance": float("nan"),
      "target_rate": float(target_rate),
      "target_reached": False,
      "success_count": success_count,
      "total_count": total,
      "on_time_count": 0,
    }

  required = int(math.ceil(float(target_rate) * float(total)))
  target_reached = success_count >= required
  index = min(max(required, 1), success_count) - 1
  budget = float(effective_distances[index])
  return {
    "budget_effective_distance": budget,
    "target_rate": float(target_rate),
    "target_reached": bool(target_reached),
    "success_count": success_count,
    "total_count": total,
    "on_time_count": min(required, success_count),
  }


def apply_direct_budget(
  records: list[EvalEpisodeRecord],
  *,
  budget_effective_distance: float | None,
  quantile: float = DEFAULT_DIRECT_CALIBRATION_QUANTILE,
) -> None:
  """Annotate records with T1-calibrated direct/wandering labels.

  This is a teacher-relative arrival budget: the effective-distance
  threshold is calibrated from T1 PP episodes and then re-used for other
  methods in the same task/seed.
  """
  if budget_effective_distance is None or not math.isfinite(budget_effective_distance):
    for record in records:
      record.direct_budget_steps = -1
      record.is_direct_success = bool(record.sequence_complete)
      record.is_wandering_success = False
      record.is_on_time_arrival = bool(record.sequence_complete)
      record.is_late_arrival = False
    return

  step_dt_values = [
    float(record.episode_time_s) / float(record.episode_steps)
    for record in records
    if record.episode_steps > 0 and math.isfinite(float(record.episode_time_s))
  ]
  fallback_step_dt = float(np.nanmedian(step_dt_values)) if step_dt_values else 0.02

  for record in records:
    step_dt = (
      float(record.episode_time_s) / float(record.episode_steps)
      if record.episode_steps > 0 and math.isfinite(float(record.episode_time_s))
      else fallback_step_dt
    )
    budget_steps = int(
      math.ceil(
        float(budget_effective_distance) / _safe_target_velocity(record) / step_dt
      )
    )
    record.direct_budget_steps = max(1, budget_steps)
    is_direct = bool(record.sequence_complete and record.episode_steps <= budget_steps)
    record.is_direct_success = is_direct
    record.is_wandering_success = bool(record.sequence_complete and not is_direct)
    record.is_on_time_arrival = is_direct
    record.is_late_arrival = bool(record.sequence_complete and not is_direct)


def apply_objective_budget(
  records: list[EvalEpisodeRecord],
  *,
  objective_timeout_margin_s: float | None = None,
) -> None:
  """Annotate records with teacher-independent objective on-time labels.

  The objective budget is stored per episode at collection time. New v6
  runs derive it from the generated A*/waypoint path length, sampled target
  velocity, and a fixed timeout margin. Old logs that do not contain these
  fields are marked unavailable instead of falling back to T1 calibration.
  When ``objective_timeout_margin_s`` is provided, recompute the budget from
  the recorded objective path length and sampled target velocity. This keeps
  paper arrival metrics independent from long environment timeout margins used
  by NoPP rollouts to avoid truncating late arrivals.
  """
  for record in records:
    if objective_timeout_margin_s is not None:
      path_length = float(record.objective_path_length_m)
      target_velocity = float(record.target_velocity_cmd)
      if math.isfinite(path_length) and path_length > 0.0 and target_velocity > 0.0:
        budget_time_s = path_length / target_velocity + float(
          objective_timeout_margin_s
        )
        record.objective_budget_time_s = budget_time_s
        step_dt = (
          float(record.episode_time_s) / float(record.episode_steps)
          if record.episode_steps > 0 and math.isfinite(float(record.episode_time_s))
          else 0.02
        )
        record.objective_budget_steps = max(
          1, int(math.ceil(budget_time_s / max(step_dt, 1e-6)))
        )
        record.objective_budget_available = True

    budget_steps = int(record.objective_budget_steps)
    budget_time_s = float(record.objective_budget_time_s)
    available = bool(record.objective_budget_available) and (
      budget_steps > 0 or math.isfinite(budget_time_s)
    )
    if not available:
      record.objective_budget_available = False
      record.is_objective_on_time_arrival = None
      record.is_objective_late_arrival = None
      continue

    record.objective_budget_available = True
    if budget_steps <= 0:
      step_dt = (
        float(record.episode_time_s) / float(record.episode_steps)
        if record.episode_steps > 0 and math.isfinite(float(record.episode_time_s))
        else 0.02
      )
      budget_steps = int(math.ceil(budget_time_s / max(step_dt, 1e-6)))
      record.objective_budget_steps = max(1, budget_steps)

    episode_time_s = float(record.episode_time_s)
    if math.isfinite(episode_time_s) and math.isfinite(budget_time_s):
      is_on_time = bool(record.sequence_complete and episode_time_s <= budget_time_s)
    else:
      is_on_time = bool(
        record.sequence_complete and int(record.episode_steps) <= int(budget_steps)
      )
    record.is_objective_on_time_arrival = is_on_time
    record.is_objective_late_arrival = bool(record.sequence_complete and not is_on_time)


def direct_budget_summary(
  records: list[EvalEpisodeRecord],
  *,
  budgets: dict[str, float],
  quantiles: tuple[float, ...] = DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
) -> dict[str, float]:
  """Compute direct/wandering rates for all sensitivity budgets."""
  if not records:
    return {}
  total = float(len(records))
  out: dict[str, float] = {}
  step_dt_values = [
    float(record.episode_time_s) / float(record.episode_steps)
    for record in records
    if record.episode_steps > 0 and math.isfinite(float(record.episode_time_s))
  ]
  fallback_step_dt = float(np.nanmedian(step_dt_values)) if step_dt_values else 0.02

  for quantile in quantiles:
    key = direct_quantile_key(quantile)
    budget = budgets.get(key)
    if budget is None or not math.isfinite(float(budget)):
      continue
    direct = 0
    wandering = 0
    budget_steps_values: list[float] = []
    for record in records:
      step_dt = (
        float(record.episode_time_s) / float(record.episode_steps)
        if record.episode_steps > 0 and math.isfinite(float(record.episode_time_s))
        else fallback_step_dt
      )
      budget_steps = int(
        math.ceil(float(budget) / _safe_target_velocity(record) / step_dt)
      )
      budget_steps_values.append(float(budget_steps))
      if record.sequence_complete and record.episode_steps <= budget_steps:
        direct += 1
      elif record.sequence_complete:
        wandering += 1
    suffix = direct_metric_suffix(quantile)
    out[f"direct_success_rate_{suffix}"] = direct / total * 100.0
    out[f"wandering_success_rate_{suffix}"] = wandering / total * 100.0
    out[f"mean_direct_budget_steps_{suffix}"] = float(np.mean(budget_steps_values))
  return out


def calibrate_envelope_brush_threshold(
  records: list[EvalEpisodeRecord],
  *,
  target_rate: float = DEFAULT_ENVELOPE_TARGET_RATE,
  scale_thresholds: tuple[float, ...] = DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS,
  existing_scale_only: bool = False,
) -> dict[str, float | bool | str]:
  """Select a safety-envelope brush threshold close to the target T1 rate."""
  if not records:
    return {
      "scale_threshold": float("nan"),
      "min_time_s": float("nan"),
      "rate": float("nan"),
      "target_rate": float(target_rate),
      "existing_scale_only": bool(existing_scale_only),
    }

  candidates: list[tuple[float, float, float, bool]] = []
  if existing_scale_only:
    times = sorted(float(r.safety_envelope_brake_time_s) for r in records)
    scale = 0.95
    for threshold in scale_thresholds:
      if math.isclose(float(threshold), 0.95, rel_tol=0.0, abs_tol=1e-9):
        scale = float(threshold)
        break
    candidates.extend(_envelope_time_candidates(scale, times, target_rate))
  else:
    for scale in scale_thresholds:
      key = safety_scale_key(scale)
      times = []
      for record in records:
        by_threshold = record.safety_envelope_brake_time_s_by_threshold or {}
        value = by_threshold.get(key)
        if value is None:
          value = record.safety_envelope_brake_time_s
        times.append(float(value))
      candidates.extend(
        _envelope_time_candidates(float(scale), sorted(times), target_rate)
      )

  if not candidates:
    return {
      "scale_threshold": float("nan"),
      "min_time_s": float("nan"),
      "rate": float("nan"),
      "target_rate": float(target_rate),
      "existing_scale_only": bool(existing_scale_only),
    }
  best_scale, best_time, best_rate, exact = min(
    candidates,
    key=lambda item: (
      abs(item[2] - float(target_rate) * 100.0),
      item[1],
      item[0],
    ),
  )
  return {
    "scale_threshold": float(best_scale),
    "min_time_s": float(best_time),
    "rate": float(best_rate),
    "target_rate": float(target_rate),
    "existing_scale_only": bool(existing_scale_only),
    "exact": bool(exact),
  }


def _envelope_time_candidates(
  scale_threshold: float,
  times: list[float],
  target_rate: float,
) -> list[tuple[float, float, float, bool]]:
  total = len(times)
  if total == 0:
    return []
  clean_times = sorted(t for t in times if math.isfinite(float(t)))
  if not clean_times:
    return [(scale_threshold, float("inf"), 0.0, False)]

  candidates: list[tuple[float, float, float, bool]] = [
    (scale_threshold, 0.0, 100.0, False)
  ]
  positive = sorted({float(t) for t in clean_times if float(t) > 0.0})
  for threshold in positive:
    rate = sum(float(t) >= threshold for t in clean_times) / total * 100.0
    candidates.append((scale_threshold, float(threshold), float(rate), True))
  max_time = positive[-1] if positive else 0.0
  candidates.append((scale_threshold, float(max_time + 1e-6), 0.0, False))
  return candidates


def safety_scale_key(scale_threshold: float) -> str:
  """Stable JSON key for a safety-envelope scale threshold."""
  value = int(round(float(scale_threshold) * 1000.0))
  return f"s{value}"


def safety_scale_thresholds_from_string(value: str) -> tuple[float, ...]:
  """Parse whitespace/comma separated safety scale thresholds."""
  parts = value.replace(",", " ").split()
  return tuple(float(part) for part in parts)


def safety_envelope_brake_time(
  record: EvalEpisodeRecord,
  *,
  scale_threshold: float | None = None,
  existing_scale_only: bool = False,
) -> float:
  """Return cumulative safety-envelope braking time for the requested scale."""
  if existing_scale_only or scale_threshold is None:
    return float(record.safety_envelope_brake_time_s)
  key = safety_scale_key(float(scale_threshold))
  value = (record.safety_envelope_brake_time_s_by_threshold or {}).get(key)
  if value is None:
    return float(record.safety_envelope_brake_time_s)
  return float(value)


def is_safety_envelope_brush(
  record: EvalEpisodeRecord,
  *,
  scale_threshold: float | None = None,
  min_time_s: float | None = None,
  existing_scale_only: bool = False,
) -> bool:
  """Classify safety-envelope brush with either stored or calibrated thresholds."""
  if min_time_s is None:
    return bool(record.safety_envelope_brake_ever)
  return safety_envelope_brake_time(
    record,
    scale_threshold=scale_threshold,
    existing_scale_only=existing_scale_only,
  ) >= float(min_time_s)


def is_contact_brush(record: EvalEpisodeRecord) -> bool:
  """Classify physical contact brush."""
  return bool(record.body_contact_ever or record.feet_obstacle_contact_ever)


def is_any_brush(
  record: EvalEpisodeRecord,
  *,
  envelope_scale_threshold: float | None = None,
  envelope_min_time_s: float | None = None,
  existing_scale_only: bool = False,
) -> bool:
  """Classify any brush as contact brush or safety-envelope brush."""
  return bool(
    is_contact_brush(record)
    or is_safety_envelope_brush(
      record,
      scale_threshold=envelope_scale_threshold,
      min_time_s=envelope_min_time_s,
      existing_scale_only=existing_scale_only,
    )
  )


def outcome_from_snapshot(
  *,
  prev_sequence_complete: bool,
  prev_key_goal_idx: int,
  prev_goal_idx: int,
  num_goals: int,
  terminated: bool,
  timed_out: bool,
) -> tuple[bool, bool, bool, bool]:
  """Compute success, key-goal, hard termination, and timeout consistently.

  ``FixedSequentialGoalCommand`` can set ``sequence_complete`` when its own
  time budget expires. A real navigation success therefore also requires that
  the current waypoint index has advanced past the final goal.
  """
  command_timeout = bool(prev_sequence_complete) and int(prev_goal_idx) < int(num_goals)
  is_timeout = bool(timed_out) or command_timeout
  is_success = (
    bool(prev_sequence_complete)
    and int(prev_goal_idx) >= int(num_goals)
    and not bool(timed_out)
  )
  is_hard = bool(terminated) and not is_success and not is_timeout
  key1_reached = bool(prev_key_goal_idx >= 1 or prev_goal_idx > 0)
  return is_success, key1_reached, is_hard, is_timeout


def get_min_raycast(
  obs_dict: object,
  raycast_obs_names: tuple[str, ...],
  num_envs: int,
  device: torch.device,
) -> torch.Tensor:
  """Return per-env minimum normalized raycast value across configured groups."""
  parts: list[torch.Tensor] = []
  for rc_name in raycast_obs_names:
    rc_tensor: torch.Tensor | None = None
    if hasattr(obs_dict, "keys"):
      keys = list(obs_dict.keys())  # type: ignore[union-attr]
      if rc_name in keys:
        v = obs_dict[rc_name]  # type: ignore[index]
        rc_tensor = v if isinstance(v, torch.Tensor) else None
      else:
        for key in keys:
          sub = obs_dict[key]  # type: ignore[index]
          if hasattr(sub, "keys") and rc_name in list(sub.keys()):
            sv = sub[rc_name]
            if isinstance(sv, torch.Tensor):
              rc_tensor = sv
              break
    if rc_tensor is not None:
      parts.append(rc_tensor.reshape(num_envs, -1).min(dim=-1).values)
  if parts:
    return torch.stack(parts, dim=-1).min(dim=-1).values
  return torch.ones(num_envs, device=device)


def get_contact_brush_state(
  env: object,
  num_envs: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return body/feet brush masks plus max contact force for each env."""
  body_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
  feet_contact = torch.zeros(num_envs, dtype=torch.bool, device=device)
  max_force = torch.zeros(num_envs, device=device)
  scene = getattr(env, "scene", None)
  if scene is None:
    return body_contact, feet_contact, max_force

  try:
    body_sensor = scene["body_collision"]
  except Exception:
    body_sensor = None
  if body_sensor is not None:
    force = getattr(getattr(body_sensor, "data", None), "force", None)
    if isinstance(force, torch.Tensor):
      mag = torch.norm(force, dim=-1)
      body_contact |= mag.max(dim=-1).values > 1.0
      max_force = torch.maximum(max_force, mag.max(dim=-1).values)

  try:
    feet_sensor = scene["feet_collision"]
  except Exception:
    feet_sensor = None
  if feet_sensor is not None:
    force = getattr(getattr(feet_sensor, "data", None), "force", None)
    if isinstance(force, torch.Tensor):
      mag = torch.norm(force, dim=-1)
      hor_force = force[..., :2].norm(dim=-1)
      ver_force = force[..., 2].abs()
      feet_contact |= (hor_force > 2.0 * ver_force + 10.0).any(dim=-1)
      max_force = torch.maximum(max_force, mag.max(dim=-1).values)

  return body_contact, feet_contact, max_force


def get_safety_brake_state(
  goal_term: FixedSequentialGoalCommand,
  num_envs: int,
  device: torch.device,
  *,
  scale_threshold: float = 0.95,
) -> torch.Tensor:
  """Return envs whose command safety envelope is actively clipping speed."""
  scale = getattr(goal_term, "_safety_velocity_scale", None)
  if not isinstance(scale, torch.Tensor) or scale.numel() != num_envs:
    return torch.zeros(num_envs, dtype=torch.bool, device=device)
  return scale.to(device=device) < float(scale_threshold)


def objective_budget_snapshot(
  goal_term: object,
  num_envs: int,
  device: torch.device,
  *,
  step_dt: float,
  objective_timeout_margin_s: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return teacher-independent path execution budgets for current episodes.

  For the v6 fixed-sequential tasks, the budget is the length of the generated
  waypoint path divided by the episode's sampled target velocity, plus an
  objective paper margin. This margin is intentionally separate from the command
  term's environment timeout margin, which can be much longer for NoPP rollouts
  so late arrivals are not truncated before diagnostics are collected. For A*
  reachable-goal tasks, the stored planner path length is used with the
  path-planning resample margin unless an objective margin override is given.
  """
  path_length = torch.full((num_envs,), float("nan"), device=device)
  budget_time_s = torch.full((num_envs,), float("nan"), device=device)
  budget_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)
  available = torch.zeros(num_envs, dtype=torch.bool, device=device)

  if isinstance(goal_term, FixedSequentialGoalCommand):
    offsets = getattr(goal_term, "_env_goal_offsets", None)
    if not isinstance(offsets, torch.Tensor) or offsets.ndim != 3:
      return path_length, budget_time_s, budget_steps, available
    num_goals = int(getattr(goal_term, "num_goals", offsets.shape[1]))
    offsets = offsets[:, :num_goals, :2].to(device=device)
    starts = torch.cat([torch.zeros_like(offsets[:, :1, :]), offsets[:, :-1, :]], dim=1)
    path_length = torch.linalg.norm(offsets - starts, dim=-1).sum(dim=1)
    base_speed = getattr(goal_term, "_base_target_velocity_cmd", None)
    if isinstance(base_speed, torch.Tensor):
      speed = base_speed.to(device=device)
    else:
      speed = torch.full(
        (num_envs,),
        float(getattr(goal_term.cfg, "target_velocity", 1.0)),
        device=device,
      )
    speed_floor = float(getattr(goal_term.cfg, "timeout_speed_floor", 0.2))
    margin_s = (
      float(objective_timeout_margin_s)
      if objective_timeout_margin_s is not None
      else float(getattr(goal_term.cfg, "timeout_margin_s", 4.0))
    )
    speed = torch.clamp(speed, min=max(speed_floor, 1e-6))
    budget_time_s = path_length / speed + margin_s
  elif isinstance(goal_term, ReachableGoalCommand):
    stored_path_length = getattr(goal_term, "path_length", None)
    if not isinstance(stored_path_length, torch.Tensor):
      return path_length, budget_time_s, budget_steps, available
    path_length = stored_path_length.to(device=device).reshape(num_envs)
    base_speed = getattr(goal_term, "_base_target_velocity_cmd", None)
    if isinstance(base_speed, torch.Tensor):
      speed = base_speed.to(device=device)
    else:
      speed = torch.full(
        (num_envs,),
        float(getattr(goal_term.cfg, "target_velocity", 1.0)),
        device=device,
      )
    planning_cfg = getattr(goal_term.cfg, "path_planning", None)
    margin_s = (
      float(objective_timeout_margin_s)
      if objective_timeout_margin_s is not None
      else float(getattr(planning_cfg, "resample_time_margin", 2.0))
    )
    speed = torch.clamp(speed, min=1e-6)
    budget_time_s = path_length / speed + margin_s
  else:
    return path_length, budget_time_s, budget_steps, available

  available = (
    torch.isfinite(path_length) & torch.isfinite(budget_time_s) & (path_length > 0.0)
  )
  budget_steps = torch.where(
    available,
    torch.ceil(budget_time_s / max(float(step_dt), 1e-6)).to(torch.long),
    budget_steps,
  )
  budget_steps = torch.where(
    available & (budget_steps < 1),
    torch.ones_like(budget_steps),
    budget_steps,
  )
  return path_length, budget_time_s, budget_steps, available


def make_local_planner_policy(
  *,
  base_policy: Callable[..., torch.Tensor],
  local_planner_inference: LocalPlannerInference,
  env: object,
  correction_gain: float,
  last_corrections: torch.Tensor,
) -> Callable[..., torch.Tensor]:
  """Wrap the locomotion policy with local-planner heading replacement."""

  goal_cmd_offset: int | None = None

  def find_offset() -> int:
    try:
      obs_manager = env.observation_manager  # type: ignore[union-attr]
      term_names = obs_manager.active_terms.get("actor", [])
      term_dims = obs_manager.group_obs_term_dim.get("actor", [])
      offset = 0
      for name, dims in zip(term_names, term_dims, strict=False):
        if name == "goal_command":
          return offset
        offset += int(np.prod(dims))
    except Exception:
      pass
    return 93

  def policy(obs: object) -> torch.Tensor:
    nonlocal goal_cmd_offset
    local_planner_obs: dict[str, torch.Tensor] = {}
    if hasattr(obs, "keys"):
      for key in obs.keys():  # type: ignore[union-attr]
        value = obs[key]  # type: ignore[index]
        if isinstance(value, torch.Tensor):
          local_planner_obs[key] = value
        elif hasattr(value, "keys"):
          for sub_key in value.keys():
            sub_value = value[sub_key]
            if isinstance(sub_value, torch.Tensor):
              local_planner_obs[sub_key] = sub_value
    else:
      local_planner_obs = {"obs": obs}  # type: ignore[assignment]

    correction = local_planner_inference.compute_correction(env, local_planner_obs)  # type: ignore[arg-type]
    last_corrections.copy_(correction.squeeze(-1))

    if hasattr(obs, "keys") and "actor" in list(obs.keys()):  # type: ignore[union-attr]
      if goal_cmd_offset is None:
        goal_cmd_offset = find_offset()
      policy_obs: torch.Tensor = obs["actor"]  # type: ignore[index]
      modified = policy_obs.clone()
      modified[:, goal_cmd_offset] = correction.squeeze(-1) * correction_gain
      obs["actor"] = modified  # type: ignore[index]

    with torch.no_grad():
      return base_policy(obs)

  return policy


def load_local_planner_inference(
  *,
  checkpoint: str,
  raycast_obs_names: tuple[str, ...],
  device: torch.device,
  num_envs: int,
  task: str = "",
  locomotion_checkpoint: str = "",
  use_odometry: bool = False,
  include_joint_vel: bool = True,
  use_measured_velocity: bool = False,
  use_ahrs_heading: bool = True,
  use_projected_yaw_rate: bool = False,
) -> LocalPlannerInference:
  """Load a local planner checkpoint and create its inference helper."""
  network, checkpoint_cfg = _load_local_planner_network_for_inference(
    checkpoint,
    raycast_obs_names=raycast_obs_names,
    device=device,
    use_odometry=use_odometry,
    include_joint_vel=include_joint_vel,
  )

  cfg = LocalPlannerInferenceCfg()
  args = TrainLocalPlannerArgs(
    mode="test",
    task=task,
    checkpoint=locomotion_checkpoint,
    local_planner_checkpoint=checkpoint,
    num_envs=num_envs,
    device=str(device),
    raycast_obs_names=raycast_obs_names,
    use_odometry=use_odometry,
    include_joint_vel=include_joint_vel,
  )
  _apply_checkpoint_cfg_to_inference_cfg(cfg, checkpoint_cfg, args)
  cfg.use_measured_velocity = use_measured_velocity
  cfg.use_ahrs_heading = use_ahrs_heading
  cfg.use_projected_yaw_rate = use_projected_yaw_rate
  inference = LocalPlannerInference(
    cfg=cfg, local_planner_network=network, device=device
  )
  inference.initialize(num_envs)
  return inference


def run_policy_eval(
  *,
  task: str,
  locomotion_checkpoint: str,
  num_envs: int,
  num_episodes: int,
  device: torch.device,
  seed: int | None,
  mode: Literal["train", "test", "play"] = "test",
  local_planner_checkpoint: str = "",
  raycast_obs_names: tuple[str, ...] = DEFAULT_RAYCAST_OBS_NAMES,
  correction_gain: float = 1.0,
  use_odometry: bool = False,
  include_joint_vel: bool = True,
  use_measured_velocity: bool = False,
  use_ahrs_heading: bool = True,
  use_projected_yaw_rate: bool = False,
  disable_safety_envelope: bool = False,
  capture_trajectories: bool = False,
  max_trajectories: int = 6,
  trajectory_stride: int = 5,
  safety_brake_scale_threshold: float = 0.95,
  safety_brake_scale_thresholds: tuple[
    float, ...
  ] = DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS,
  safety_brake_min_time_s: float = 0.5,
  objective_timeout_margin_s: float | None = 10.0,
) -> EvalRunResult:
  """Run one benchmark condition."""
  configure_torch_backends()
  set_eval_seed(seed)

  train_args = TrainLocalPlannerArgs(
    mode=mode,
    task=task,
    num_envs=num_envs,
    checkpoint=locomotion_checkpoint,
    local_planner_checkpoint=local_planner_checkpoint,
    device=str(device),
    raycast_obs_names=raycast_obs_names,
    use_odometry=use_odometry,
    include_joint_vel=include_joint_vel,
    use_measured_velocity=use_measured_velocity,
    use_ahrs_heading=use_ahrs_heading,
    use_projected_yaw_rate=use_projected_yaw_rate,
    correction_gain=correction_gain,
    seed=seed,
  )
  env, rl_cfg = setup_environment(train_args)
  base_policy = load_policy(locomotion_checkpoint, env, rl_cfg, device)
  last_corrections = torch.full((num_envs,), float("nan"), device=device)

  if local_planner_checkpoint:
    inference = load_local_planner_inference(
      checkpoint=local_planner_checkpoint,
      raycast_obs_names=raycast_obs_names,
      device=device,
      num_envs=num_envs,
      task=task,
      locomotion_checkpoint=locomotion_checkpoint,
      use_odometry=train_args.use_odometry,
      include_joint_vel=train_args.include_joint_vel,
      use_measured_velocity=use_measured_velocity,
      use_ahrs_heading=use_ahrs_heading,
      use_projected_yaw_rate=use_projected_yaw_rate,
    )
    policy = make_local_planner_policy(
      base_policy=base_policy,
      local_planner_inference=inference,
      env=env,
      correction_gain=correction_gain,
      last_corrections=last_corrections,
    )
  else:
    inference = None
    policy = base_policy

  goal_term = env.command_manager.get_term("goal")
  if goal_term is None:
    raise RuntimeError("Cannot find the 'goal' command term in the environment.")
  if not isinstance(goal_term, FixedSequentialGoalCommand):
    raise RuntimeError(
      f"'goal' term is not FixedSequentialGoalCommand; got {type(goal_term)}"
    )
  if disable_safety_envelope:
    safety_cfg = getattr(goal_term.cfg, "safety_velocity_envelope", None)
    if safety_cfg is not None:
      safety_cfg.enabled = False
    if hasattr(goal_term.cfg, "safety_enabled"):
      goal_term.cfg.safety_enabled = False
    print("  [Ablation] Safety velocity envelope disabled for this run.")

  num_goals = int(goal_term.num_goals)
  key_goal_indices = list(goal_term._key_goal_indices)
  print(f"  Total waypoints: {num_goals}, key goal indices: {key_goal_indices}")

  obs_dict, _ = env.reset(seed=seed)
  if inference is not None:
    initial_goal_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    inference.obs_builder.update_initial_goal(env, initial_goal_mask)

  robot = env.scene["robot"]
  env_spawn_x_offset = (
    robot.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  ).clone()

  step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
  min_raycast_ever = torch.ones(num_envs, device=device)
  action_sum_acc = torch.zeros(num_envs, device=device)
  action_rate_sum = torch.zeros(num_envs, device=device)
  yaw_sum = torch.zeros(num_envs, device=device)
  yaw_sq_sum = torch.zeros(num_envs, device=device)
  body_contact_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
  feet_contact_ever = torch.zeros(num_envs, dtype=torch.bool, device=device)
  safety_brake_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  safety_thresholds = tuple(
    sorted(
      {float(safety_brake_scale_threshold), *map(float, safety_brake_scale_thresholds)}
    )
  )
  safety_brake_steps_by_threshold = {
    safety_scale_key(threshold): torch.zeros(num_envs, dtype=torch.long, device=device)
    for threshold in safety_thresholds
  }
  max_contact_force = torch.zeros(num_envs, device=device)
  action_dim = int(env.action_manager.action.shape[-1])
  prev_action = torch.zeros(num_envs, action_dim, device=device)
  prev_prev_action = torch.zeros_like(prev_action)
  trajectories: list[list[tuple[float, float]]] = [[] for _ in range(num_envs)]
  trajectory_enabled = [
    capture_trajectories and i < max_trajectories for i in range(num_envs)
  ]
  saved_trajectory_count = 0

  records: list[EvalEpisodeRecord] = []
  t_start = time.time()
  print(f"  Collecting {num_episodes} episodes ({num_envs} parallel envs)...")

  while len(records) < num_episodes:
    prev_goal_idx = goal_term.current_goal_idx.clone()
    prev_key_goal_idx = goal_term.current_key_goal_idx.clone()
    prev_seq_complete = goal_term.sequence_complete.clone()
    prev_dist_to_key_goal = goal_term.distance_to_key_goal.clone()
    prev_robot_pos = robot.data.root_link_pos_w[:, :2].clone()
    prev_spawn_pos = goal_term._spawn_pos_w.clone()
    prev_robot_vel = robot.data.root_link_lin_vel_b[:, 0].clone()
    prev_target_velocity = goal_term._base_target_velocity_cmd.clone()
    (
      prev_objective_path_length,
      prev_objective_budget_time,
      prev_objective_budget_steps,
      prev_objective_budget_available,
    ) = objective_budget_snapshot(
      goal_term,
      num_envs,
      device,
      step_dt=float(env.step_dt),
      objective_timeout_margin_s=objective_timeout_margin_s,
    )
    prev_min_raycast = get_min_raycast(obs_dict, raycast_obs_names, num_envs, device)
    prev_spawn_x = env_spawn_x_offset.clone()
    prev_local_planner_correction = last_corrections.clone()

    if capture_trajectories:
      should_sample = (step_counts % trajectory_stride) == 0
      for i in range(num_envs):
        if (
          trajectory_enabled[i]
          and saved_trajectory_count < max_trajectories
          and bool(should_sample[i].item())
        ):
          trajectories[i].append(
            (float(prev_robot_pos[i, 0].item()), float(prev_robot_pos[i, 1].item()))
          )

    min_raycast_ever = torch.minimum(min_raycast_ever, prev_min_raycast)

    yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
    yaw_sum += yaw_rate
    yaw_sq_sum += yaw_rate * yaw_rate

    with torch.no_grad():
      actions = policy(obs_dict)

    action_rate = torch.mean((actions - prev_action) ** 2, dim=-1)
    action_acc = torch.mean(
      torch.abs(actions - 2.0 * prev_action + prev_prev_action), dim=-1
    )
    action_rate_sum += action_rate
    action_sum_acc += action_acc
    prev_prev_action.copy_(prev_action)
    prev_action.copy_(actions)

    obs_dict, _reward, terminated, time_outs, _extras = env.step(actions)
    step_counts += 1
    body_contact, feet_contact, contact_force = get_contact_brush_state(
      env, num_envs, device
    )
    safety_brake = get_safety_brake_state(
      goal_term,
      num_envs,
      device,
      scale_threshold=safety_brake_scale_threshold,
    )
    for threshold in safety_thresholds:
      key = safety_scale_key(threshold)
      safety_brake_steps_by_threshold[key] += get_safety_brake_state(
        goal_term,
        num_envs,
        device,
        scale_threshold=threshold,
      ).to(torch.long)
    body_contact_ever |= body_contact
    feet_contact_ever |= feet_contact
    safety_brake_steps += safety_brake.to(torch.long)
    max_contact_force = torch.maximum(max_contact_force, contact_force)

    dones = terminated | time_outs
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

    if done_ids.numel() > 0:
      for idx in done_ids:
        i = int(idx.item())
        steps = max(1, int(step_counts[i].item()))
        success, key1_reached, hard, timeout = outcome_from_snapshot(
          prev_sequence_complete=bool(prev_seq_complete[i].item()),
          prev_key_goal_idx=int(prev_key_goal_idx[i].item()),
          prev_goal_idx=int(prev_goal_idx[i].item()),
          num_goals=num_goals,
          terminated=bool(terminated[i].item()),
          timed_out=bool(time_outs[i].item()),
        )
        if hard:
          col_wp = int(prev_goal_idx[i].item())
          col_x = float((prev_robot_pos[i, 0] - prev_spawn_pos[i, 0]).item())
          col_y = float((prev_robot_pos[i, 1] - prev_spawn_pos[i, 1]).item())
        else:
          col_wp = -1
          col_x = float("nan")
          col_y = float("nan")

        mean_action_acc = float((action_sum_acc[i] / steps).item())
        action_rate_l2 = float((action_rate_sum[i] / steps).item())
        yaw_mean = yaw_sum[i] / steps
        base_ang_vel_var = float(
          torch.clamp(yaw_sq_sum[i] / steps - yaw_mean * yaw_mean, min=0.0).item()
        )
        gait_score = 1.0 / (1.0 + mean_action_acc + base_ang_vel_var)
        safety_brake_time_s = float(safety_brake_steps[i].item() * env.step_dt)
        safety_brake_steps_by_scale = {
          key: int(value[i].item())
          for key, value in safety_brake_steps_by_threshold.items()
        }
        safety_brake_time_by_scale = {
          key: float(steps_at_threshold * env.step_dt)
          for key, steps_at_threshold in safety_brake_steps_by_scale.items()
        }
        safety_brake_ever = safety_brake_time_s >= float(safety_brake_min_time_s)
        brush_contact = bool(
          body_contact_ever[i].item()
          or feet_contact_ever[i].item()
          or safety_brake_ever
        )

        records.append(
          EvalEpisodeRecord(
            sequence_complete=success,
            key_goal1_reached=key1_reached,
            waypoints_completed=int(prev_goal_idx[i].item()),
            hard_terminated=hard,
            timed_out=timeout,
            dist_to_final=float(prev_dist_to_key_goal[i].item()),
            episode_steps=steps,
            collision_waypoint_idx=col_wp,
            collision_pos_x_rel=col_x,
            collision_pos_y_rel=col_y,
            actual_velocity=float(prev_robot_vel[i].item()),
            target_velocity_cmd=float(prev_target_velocity[i].item()),
            episode_time_s=float(steps * env.step_dt),
            objective_path_length_m=float(prev_objective_path_length[i].item()),
            objective_budget_time_s=float(prev_objective_budget_time[i].item()),
            objective_budget_steps=int(prev_objective_budget_steps[i].item()),
            objective_budget_available=bool(prev_objective_budget_available[i].item()),
            body_contact_ever=bool(body_contact_ever[i].item()),
            feet_obstacle_contact_ever=bool(feet_contact_ever[i].item()),
            safety_envelope_brake_ever=safety_brake_ever,
            safety_envelope_brake_steps=int(safety_brake_steps[i].item()),
            safety_envelope_brake_time_s=safety_brake_time_s,
            safety_envelope_brake_steps_by_threshold=safety_brake_steps_by_scale,
            safety_envelope_brake_time_s_by_threshold=safety_brake_time_by_scale,
            brush_contact_ever=brush_contact,
            max_contact_force=float(max_contact_force[i].item()),
            local_planner_correction_at_end=float(
              prev_local_planner_correction[i].item()
            ),
            min_raycast_at_end=float(prev_min_raycast[i].item()),
            spawn_x_offset=float(prev_spawn_x[i].item()),
            min_raycast_ever=float(min_raycast_ever[i].item()),
            mean_action_acc=mean_action_acc,
            action_rate_l2=action_rate_l2,
            base_ang_vel_var=base_ang_vel_var,
            gait_preservation_score=gait_score,
            trajectory_xy=(
              list(trajectories[i])
              if trajectory_enabled[i] and saved_trajectory_count < max_trajectories
              else []
            ),
          )
        )
        if trajectory_enabled[i] and saved_trajectory_count < max_trajectories:
          saved_trajectory_count += 1
          if saved_trajectory_count >= max_trajectories:
            for tr in trajectories:
              tr.clear()

      robot = env.scene["robot"]
      env_spawn_x_offset[done_ids] = (
        robot.data.root_link_pos_w[done_ids, 0] - env.scene.env_origins[done_ids, 0]
      )
      step_counts[done_ids] = 0
      min_raycast_ever[done_ids] = 1.0
      action_sum_acc[done_ids] = 0.0
      action_rate_sum[done_ids] = 0.0
      yaw_sum[done_ids] = 0.0
      yaw_sq_sum[done_ids] = 0.0
      body_contact_ever[done_ids] = False
      feet_contact_ever[done_ids] = False
      safety_brake_steps[done_ids] = 0
      for value in safety_brake_steps_by_threshold.values():
        value[done_ids] = 0
      max_contact_force[done_ids] = 0.0
      prev_action[done_ids] = 0.0
      prev_prev_action[done_ids] = 0.0
      for idx in done_ids:
        i = int(idx.item())
        trajectories[i] = []
        trajectory_enabled[i] = (
          capture_trajectories and saved_trajectory_count < max_trajectories
        )

    n_done = len(records)
    if n_done > 0 and n_done // 50 > (n_done - done_ids.numel()) // 50:
      elapsed = time.time() - t_start
      rate = n_done / elapsed if elapsed > 0 else 0.0
      print(f"  Progress: {n_done}/{num_episodes} ({elapsed:.0f}s, {rate:.1f} eps/s)")

  env.close()
  elapsed = time.time() - t_start
  print(f"  Done: {len(records)} episodes in {elapsed:.1f}s.")
  result = EvalRunResult(
    records=records[:num_episodes],
    num_goals=num_goals,
    key_goal_indices=key_goal_indices,
    elapsed_s=elapsed,
  )
  del obs_dict, policy, base_policy, env
  if inference is not None:
    del inference
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
  return result


def compute_summary(
  records: list[EvalEpisodeRecord],
  num_goals: int = 0,
  direct_budgets: dict[str, float] | None = None,
  envelope_scale_threshold: float | None = None,
  envelope_min_time_s: float | None = None,
  existing_scale_only: bool = False,
) -> dict[str, float]:
  """Compute paper and diagnostic summary metrics."""
  if not records:
    return {}
  success = np.array([r.sequence_complete for r in records], dtype=float)
  key1 = np.array([r.key_goal1_reached for r in records], dtype=float)
  hard = np.array([r.hard_terminated for r in records], dtype=float)
  timeout = np.array([r.timed_out for r in records], dtype=float)
  envelope_brush = np.array(
    [
      is_safety_envelope_brush(
        r,
        scale_threshold=envelope_scale_threshold,
        min_time_s=envelope_min_time_s,
        existing_scale_only=existing_scale_only,
      )
      for r in records
    ],
    dtype=float,
  )
  contact_brush = np.array(
    [is_contact_brush(r) for r in records],
    dtype=float,
  )
  brush = np.array(
    [
      is_any_brush(
        r,
        envelope_scale_threshold=envelope_scale_threshold,
        envelope_min_time_s=envelope_min_time_s,
        existing_scale_only=existing_scale_only,
      )
      for r in records
    ],
    dtype=float,
  )
  brush_success = np.array(
    [r.sequence_complete and bool(brush[idx]) for idx, r in enumerate(records)],
    dtype=float,
  )
  direct_success = np.array(
    [
      bool(r.is_direct_success)
      if r.is_direct_success is not None
      else bool(r.sequence_complete)
      for r in records
    ],
    dtype=float,
  )
  wandering_success = np.array(
    [
      bool(r.is_wandering_success) if r.is_wandering_success is not None else False
      for r in records
    ],
    dtype=float,
  )
  on_time_arrival = np.array(
    [
      bool(r.is_on_time_arrival)
      if r.is_on_time_arrival is not None
      else bool(r.is_direct_success)
      if r.is_direct_success is not None
      else bool(r.sequence_complete)
      for r in records
    ],
    dtype=float,
  )
  late_arrival = np.array(
    [
      bool(r.is_late_arrival)
      if r.is_late_arrival is not None
      else bool(r.is_wandering_success)
      if r.is_wandering_success is not None
      else False
      for r in records
    ],
    dtype=float,
  )
  slow_brush_arrival = np.array(
    [
      bool(r.sequence_complete) and bool(late_arrival[idx]) and bool(brush[idx])
      for idx, r in enumerate(records)
    ],
    dtype=float,
  )
  late_clean_arrival = np.array(
    [
      bool(r.sequence_complete) and bool(late_arrival[idx]) and not bool(brush[idx])
      for idx, r in enumerate(records)
    ],
    dtype=float,
  )
  objective_available = np.array(
    [bool(r.objective_budget_available) for r in records],
    dtype=bool,
  )
  objective_on_time = np.array(
    [
      bool(r.is_objective_on_time_arrival)
      if r.is_objective_on_time_arrival is not None
      else False
      for r in records
    ],
    dtype=float,
  )
  objective_late = np.array(
    [
      bool(r.is_objective_late_arrival)
      if r.is_objective_late_arrival is not None
      else False
      for r in records
    ],
    dtype=float,
  )
  if objective_available.any():
    objective_on_time_rate = float(
      objective_on_time[objective_available].mean() * 100.0
    )
    objective_late_rate = float(objective_late[objective_available].mean() * 100.0)
    objective_slow_brush_rate = float(
      np.array(
        [
          bool(r.sequence_complete)
          and bool(r.is_objective_late_arrival)
          and bool(brush[idx])
          for idx, r in enumerate(records)
          if objective_available[idx]
        ],
        dtype=float,
      ).mean()
      * 100.0
    )
    objective_late_clean_rate = float(
      np.array(
        [
          bool(r.sequence_complete)
          and bool(r.is_objective_late_arrival)
          and not bool(brush[idx])
          for idx, r in enumerate(records)
          if objective_available[idx]
        ],
        dtype=float,
      ).mean()
      * 100.0
    )
  else:
    objective_on_time_rate = float("nan")
    objective_late_rate = float("nan")
    objective_slow_brush_rate = float("nan")
    objective_late_clean_rate = float("nan")
  wp = np.array([r.waypoints_completed for r in records], dtype=float)
  steps = np.array([r.episode_steps for r in records], dtype=float)
  dist = np.array([r.dist_to_final for r in records], dtype=float)
  timeout_mask = np.array([r.timed_out and not r.sequence_complete for r in records])
  mean_timeout_dist = float(dist[timeout_mask].mean()) if timeout_mask.any() else 0.0

  summary: dict[str, float] = {
    "success_rate": float(success.mean() * 100.0),
    "key_goal1_rate": float(key1.mean() * 100.0),
    "collision_rate": float(hard.mean() * 100.0),
    "hard_collision_rate": float(hard.mean() * 100.0),
    "timeout_rate": float(timeout.mean() * 100.0),
    "direct_success_rate": float(direct_success.mean() * 100.0),
    "wandering_success_rate": float(wandering_success.mean() * 100.0),
    "on_time_arrival_rate": float(on_time_arrival.mean() * 100.0),
    "late_arrival_rate": float(late_arrival.mean() * 100.0),
    "slow_brush_arrival_rate": float(slow_brush_arrival.mean() * 100.0),
    "late_clean_arrival_rate": float(late_clean_arrival.mean() * 100.0),
    "teacher_relative_on_time_arrival_rate": float(on_time_arrival.mean() * 100.0),
    "teacher_relative_late_arrival_rate": float(late_arrival.mean() * 100.0),
    "teacher_relative_slow_brush_arrival_rate": float(
      slow_brush_arrival.mean() * 100.0
    ),
    "teacher_relative_late_clean_arrival_rate": float(
      late_clean_arrival.mean() * 100.0
    ),
    "objective_budget_available_rate": float(objective_available.mean() * 100.0),
    "objective_on_time_arrival_rate": objective_on_time_rate,
    "objective_late_arrival_rate": objective_late_rate,
    "objective_slow_brush_arrival_rate": objective_slow_brush_rate,
    "objective_late_clean_arrival_rate": objective_late_clean_rate,
    "brush_rate": float(brush.mean() * 100.0),
    "any_brush_rate": float(brush.mean() * 100.0),
    "brush_success_rate": float(brush_success.mean() * 100.0),
    "contact_brush_rate": float(contact_brush.mean() * 100.0),
    "safety_envelope_brush_rate": float(envelope_brush.mean() * 100.0),
    "envelope_brush_rate": float(envelope_brush.mean() * 100.0),
    "mean_safety_envelope_brake_time_s": float(
      np.nanmean([r.safety_envelope_brake_time_s for r in records])
    ),
    "mean_max_contact_force": float(np.nanmean([r.max_contact_force for r in records])),
    "timeout_dist_m": mean_timeout_dist,
    "mean_episode_steps": float(steps.mean()),
    "mean_episode_time_s": float(np.nanmean([r.episode_time_s for r in records])),
    "mean_target_velocity_cmd": float(
      np.nanmean([r.target_velocity_cmd for r in records])
    ),
    "mean_direct_budget_steps": float(
      np.mean(
        [r.direct_budget_steps for r in records if int(r.direct_budget_steps) > 0]
      )
      if any(int(r.direct_budget_steps) > 0 for r in records)
      else float("nan")
    ),
    "mean_objective_budget_steps": float(
      np.mean(
        [
          r.objective_budget_steps
          for r in records
          if bool(r.objective_budget_available) and int(r.objective_budget_steps) > 0
        ]
      )
      if any(
        bool(r.objective_budget_available) and int(r.objective_budget_steps) > 0
        for r in records
      )
      else float("nan")
    ),
    "mean_objective_budget_time_s": float(
      np.nanmean(
        [
          r.objective_budget_time_s
          for r in records
          if bool(r.objective_budget_available)
        ]
      )
      if any(bool(r.objective_budget_available) for r in records)
      else float("nan")
    ),
    "mean_objective_path_length_m": float(
      np.nanmean(
        [
          r.objective_path_length_m
          for r in records
          if bool(r.objective_budget_available)
        ]
      )
      if any(bool(r.objective_budget_available) for r in records)
      else float("nan")
    ),
    "mean_action_acc": float(np.nanmean([r.mean_action_acc for r in records])),
    "action_rate_l2": float(np.nanmean([r.action_rate_l2 for r in records])),
    "base_ang_vel_var": float(np.nanmean([r.base_ang_vel_var for r in records])),
    "gait_preservation_score": float(
      np.nanmean([r.gait_preservation_score for r in records])
    ),
  }
  if direct_budgets:
    summary.update(direct_budget_summary(records, budgets=direct_budgets))
  if num_goals > 2:
    summary["mean_waypoints_completed"] = float(wp.mean())
  return summary


def compute_collision_table(
  records: list[EvalEpisodeRecord],
  num_goals: int,
  key_goal_indices: list[int],
) -> list[dict]:
  """Per-waypoint collision distribution."""
  total = len(records)
  collisions = [r for r in records if r.hard_terminated]
  n_col = len(collisions)
  rows: list[dict] = []
  for wp in range(num_goals):
    wp_records = [r for r in collisions if r.collision_waypoint_idx == wp]
    count = len(wp_records)
    xs = np.array([r.collision_pos_x_rel for r in wp_records], dtype=float)
    ys = np.array([r.collision_pos_y_rel for r in wp_records], dtype=float)
    rows.append(
      {
        "wp_idx": wp,
        "label": wp_label(wp, key_goal_indices),
        "collision_count": count,
        "collision_rate_pct": count / total * 100.0 if total else 0.0,
        "collision_pct_of_all": count / n_col * 100.0 if n_col else 0.0,
        "mean_x": float(np.nanmean(xs)) if count else float("nan"),
        "mean_y": float(np.nanmean(ys)) if count else float("nan"),
      }
    )
  return rows


def classify_episodes(
  records: list[EvalEpisodeRecord],
  wall_y: float,
  wall_proximity: float = 0.15,
  direct_max_steps: int = 1200,
) -> dict[str, int]:
  """Classify episodes into success/failure modes."""
  result = {
    "total": len(records),
    "direct_clean_success": 0,
    "direct_brush_success": 0,
    "wandering_success": 0,
    "spawn_failure": 0,
    "avoidance_failure": 0,
    "goal_seeking_failure": 0,
    "wall_brush_timeout": 0,
    "timeout_clean": 0,
  }
  post_wall_cutoff = wall_y + 2.0
  for record in records:
    near_wall = (
      not math.isnan(record.min_raycast_ever)
      and record.min_raycast_ever <= wall_proximity
    )
    if record.hard_terminated:
      y = record.collision_pos_y_rel
      if math.isnan(y) or y < 0:
        result["spawn_failure"] += 1
      elif y < post_wall_cutoff:
        result["avoidance_failure"] += 1
      else:
        result["goal_seeking_failure"] += 1
    elif record.sequence_complete:
      if record.is_wandering_success is True or (
        record.is_wandering_success is None and record.episode_steps > direct_max_steps
      ):
        result["wandering_success"] += 1
      elif near_wall:
        result["direct_brush_success"] += 1
      else:
        result["direct_clean_success"] += 1
    elif near_wall:
      result["wall_brush_timeout"] += 1
    else:
      result["timeout_clean"] += 1
  return result


def aggregate_seed_summaries(
  seed_summaries: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
  """Aggregate per-seed summaries as mean/std."""
  keys: list[str] = []
  for summary in seed_summaries:
    for key in summary:
      if key not in keys:
        keys.append(key)
  out: dict[str, dict[str, float]] = {}
  for key in keys:
    values = np.array(
      [summary[key] for summary in seed_summaries if key in summary],
      dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
      out[key] = {"mean": float("nan"), "std": float("nan")}
    else:
      out[key] = {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
      }
  return out


def wp_label(wp_idx: int, key_goal_indices: list[int]) -> str:
  """Generate a waypoint label."""
  return f"{'*' if wp_idx in key_goal_indices else ' '}WP{wp_idx}"


def ensure_empty_jsonl(path: str | os.PathLike[str]) -> None:
  """Create or truncate a JSONL file."""
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8"):
    pass
