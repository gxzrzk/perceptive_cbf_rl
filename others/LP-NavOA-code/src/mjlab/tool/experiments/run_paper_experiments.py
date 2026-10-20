#!/usr/bin/env python3
"""Run paper-scale v6 navigation benchmarks.

Default command:
  uv run python -m mjlab.tool.experiments.run_paper_experiments \\
    --preset v6 --seeds 0 1 2 --episodes-per-seed 500 --num-envs 256
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.tool.experiments.eval_utils import (
  DEFAULT_ARRIVAL_TARGET_RATE,
  DEFAULT_DIRECT_CALIBRATION_QUANTILE,
  DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
  DEFAULT_ENVELOPE_TARGET_RATE,
  DEFAULT_LOCOMOTION_CHECKPOINT,
  DEFAULT_RAYCAST_OBS_NAMES,
  DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS,
  DEFAULT_SEEDS,
  V6_TASK_SPECS,
  TaskSpec,
  aggregate_seed_summaries,
  append_jsonl,
  apply_direct_budget,
  apply_objective_budget,
  calibrate_arrival_budget_for_target,
  calibrate_direct_budgets,
  calibrate_envelope_brush_threshold,
  compute_summary,
  direct_quantile_key,
  ensure_empty_jsonl,
  record_to_dict,
  run_policy_eval,
  safety_scale_thresholds_from_string,
  sanitize_for_json,
  write_json,
)


@dataclass
class PaperExperimentArgs:
  """Arguments for paper benchmark reproduction."""

  preset: Literal["v6"] = "v6"
  """Task preset. Only v6 is currently defined for final paper evaluation."""

  locomotion_checkpoint: str = DEFAULT_LOCOMOTION_CHECKPOINT
  """Frozen locomotion policy checkpoint."""

  seeds: tuple[int, ...] = DEFAULT_SEEDS
  """Random seeds used for mean/std aggregation."""

  episodes_per_seed: int = 500
  """Episodes collected for each task/method/seed."""

  num_envs: int = 256
  """Parallel environments."""

  device: str = "cuda"
  """Evaluation device."""

  raycast_obs_names: tuple[str, ...] = DEFAULT_RAYCAST_OBS_NAMES
  """Raycast observation group names."""

  low_speed_local_planner_checkpoint: str = ""
  """Override Low-speed DR local planner checkpoint."""

  low_speed_odom_local_planner_checkpoint: str = ""
  """Override Low-speed odom local planner checkpoint."""

  indoor_local_planner_checkpoint: str = ""
  """Override Indoor DR local planner checkpoint."""

  indoor_odom_local_planner_checkpoint: str = ""
  """Override Indoor odom local planner checkpoint."""

  correction_gain: float = 1.0
  """Local planner correction gain for Ours and ablation variants."""

  output_root: str = "logs/benchmark/paper"
  """Root directory for paper benchmark outputs."""

  run_name: str = ""
  """Optional run directory name. Defaults to timestamp."""

  dry_run: bool = False
  """Print the planned matrix without launching simulation."""

  smoke: bool = False
  """Run a tiny full-matrix smoke test. Forces episodes=1 and num_envs=1."""

  ablation_config: str = ""
  """Optional JSON config with ablation variants and existing checkpoints."""

  experiment_groups: str = "main ablation"
  """Experiment groups to run. Supported groups: main, ablation."""

  capture_trajectories: bool = True
  """Store a few representative trajectories per condition."""

  max_trajectories_per_condition: int = 3
  """Maximum trajectories stored for plotting per task/method/seed run."""

  direct_calibration_quantile: float = DEFAULT_DIRECT_CALIBRATION_QUANTILE
  """Primary T1 effective-distance quantile for direct/wandering classification."""

  arrival_target_rate: float = DEFAULT_ARRIVAL_TARGET_RATE
  """Target T1 on-time arrival rate for calibrated arrival budgets."""

  min_direct_calibration_episodes: int = 5
  """Minimum T1 successful episodes required before applying direct budgets."""

  safety_brake_scale_threshold: float = 0.95
  """Safety-envelope speed scale below this value counts as active braking."""

  safety_brake_scale_thresholds: str = " ".join(
    str(value) for value in DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS
  )
  """Safety scale thresholds recorded for post-hoc envelope-brush calibration."""

  safety_brake_min_time_s: float = 0.5
  """Minimum cumulative safety braking time before counting a brush episode."""

  envelope_target_rate: float = DEFAULT_ENVELOPE_TARGET_RATE
  """Target T1 envelope-brush rate used for calibrated safety tables."""

  objective_timeout_margin_s: float = 10.0
  """Paper objective-arrival margin added to nominal path length / target speed."""


@dataclass(frozen=True)
class Condition:
  task_label: str
  method: str
  task: str
  local_planner_checkpoint: str = ""
  use_odometry: bool = False
  include_joint_vel: bool = True
  disable_safety_envelope: bool = False


def _task_specs_for_preset(
  preset: str,
  *,
  args: PaperExperimentArgs | None = None,
) -> tuple[TaskSpec, ...]:
  if preset == "v6":
    specs = V6_TASK_SPECS
    if args is not None:
      specs = _with_local_planner_overrides(specs, args)
    return specs
  raise ValueError(f"Unknown preset: {preset}")


def _with_local_planner_overrides(
  specs: tuple[TaskSpec, ...], args: PaperExperimentArgs
) -> tuple[TaskSpec, ...]:
  updated: list[TaskSpec] = []
  for spec in specs:
    local_planner_checkpoint = spec.local_planner_checkpoint
    odom_local_planner_checkpoint = spec.odom_local_planner_checkpoint
    if spec.label == "Low-speed":
      local_planner_checkpoint = (
        args.low_speed_local_planner_checkpoint or local_planner_checkpoint
      )
      odom_local_planner_checkpoint = (
        args.low_speed_odom_local_planner_checkpoint or odom_local_planner_checkpoint
      )
    elif spec.label == "Indoor":
      local_planner_checkpoint = (
        args.indoor_local_planner_checkpoint or local_planner_checkpoint
      )
      odom_local_planner_checkpoint = (
        args.indoor_odom_local_planner_checkpoint or odom_local_planner_checkpoint
      )
    updated.append(
      replace(
        spec,
        local_planner_checkpoint=local_planner_checkpoint,
        odom_local_planner_checkpoint=odom_local_planner_checkpoint,
      )
    )
  return tuple(updated)


def _parse_experiment_groups(groups: str) -> tuple[str, ...]:
  requested = tuple(part.strip().lower() for part in groups.split() if part.strip())
  if not requested:
    raise SystemExit("experiment_groups cannot be empty. Use: main, ablation.")
  valid = {"main", "ablation"}
  unknown = sorted({group for group in requested if group not in valid})
  if unknown:
    raise SystemExit(
      f"Unknown experiment group(s): {', '.join(unknown)}. "
      "Supported groups: main, ablation."
    )
  ordered = tuple(group for group in ("main", "ablation") if group in requested)
  return ordered


def _main_conditions(task_specs: tuple[TaskSpec, ...]) -> list[Condition]:
  conditions: list[Condition] = []
  for spec in task_specs:
    conditions.extend(
      [
        Condition(spec.label, "T1 (PP upper bound)", spec.pp_task),
        Condition(spec.label, "R1 (Backbone-only)", spec.nopp_task),
        Condition(
          spec.label,
          "Ours (DR)",
          spec.nopp_task,
          spec.local_planner_checkpoint,
        ),
        Condition(
          spec.label,
          "Ours (Odom)",
          spec.nopp_task,
          spec.odom_local_planner_checkpoint,
          use_odometry=True,
          include_joint_vel=False,
        ),
      ]
    )
  return conditions


def _make_run_dir(args: PaperExperimentArgs) -> Path:
  name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  path = Path(args.output_root) / name
  path.mkdir(parents=True, exist_ok=True)
  (path / "figures").mkdir(exist_ok=True)
  return path


def _format_mean_std(metric: dict[str, float] | None, suffix: str = "") -> str:
  if metric is None:
    return "MISSING"
  mean = metric.get("mean", float("nan"))
  std = metric.get("std", float("nan"))
  if mean != mean:
    return "--"
  return f"{mean:.1f}{suffix} $\\pm$ {std:.1f}"


def _format_plain(metric: dict[str, float] | None) -> str:
  if metric is None:
    return "MISSING"
  mean = metric.get("mean", float("nan"))
  std = metric.get("std", float("nan"))
  if mean != mean:
    return "--"
  return f"{mean:.2f} ± {std:.2f}"


def _format_plain_pct(metric: dict[str, float] | None) -> str:
  if metric is None:
    return "MISSING"
  mean = metric.get("mean", float("nan"))
  std = metric.get("std", float("nan"))
  if mean != mean:
    return "--"
  return f"{mean:.1f} ± {std:.1f}%"


def _format_gap(value: float) -> str:
  if value != value:
    return "--"
  return f"{value:.2f}%"


def _format_gap_tex(value: float) -> str:
  if value != value:
    return "--"
  return f"{value:.2f}\\%"


def _metric_mean(entry: dict, metric: str) -> float:
  aggregate = entry.get("aggregate", {})
  data = aggregate.get(metric)
  if not isinstance(data, dict):
    return float("nan")
  return float(data.get("mean", float("nan")))


def _write_main_tables(run_dir: Path, summary: dict) -> None:
  conditions = summary["conditions"]
  by_key = {(entry["task_label"], entry["method"]): entry for entry in conditions}
  task_labels = summary["task_labels"]
  display_methods = [
    "R1 (Backbone-only)",
    "Ours (DR)",
    "Ours (Odom)",
  ]

  tex_lines = [
    "\\begin{tabular}{l l c c c c c c c c c}",
    "\\toprule",
    "Task & Method & Success & Obj. on-time & $L_{obj}$ & $T_{obj}$ & $T_{ep}$ & Any brush & Obj. slow-brush & Timeout & Hard coll. \\\\",
    "\\midrule",
  ]
  md_lines = [
    "# Paper Benchmark Summary",
    "",
    "| Task | Method | Success | Obj. on-time | L_obj (m) | T_obj (s) | T_ep (s) | Any brush | Obj. slow-brush | Timeout | Hard coll. |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]

  for task_idx, task_label in enumerate(task_labels):
    for method_idx, method in enumerate(display_methods):
      entry = by_key.get((task_label, method))
      metrics = entry.get("aggregate", {}) if entry else {}
      task_cell = task_label if method_idx == 0 else ""
      pct = "\\%"
      tex_lines.append(
        f"{task_cell} & {method} & "
        f"{_format_mean_std(metrics.get('success_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('objective_on_time_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('mean_objective_path_length_m'))} & "
        f"{_format_mean_std(metrics.get('mean_objective_budget_time_s'))} & "
        f"{_format_mean_std(metrics.get('mean_episode_time_s'))} & "
        f"{_format_mean_std(metrics.get('any_brush_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('objective_slow_brush_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('timeout_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('hard_collision_rate'), pct)} \\\\"
      )
      md_lines.append(
        f"| {task_label} | {method} | "
        f"{_format_plain(metrics.get('success_rate'))}% | "
        f"{_format_plain(metrics.get('objective_on_time_arrival_rate'))}% | "
        f"{_format_plain(metrics.get('mean_objective_path_length_m'))} | "
        f"{_format_plain(metrics.get('mean_objective_budget_time_s'))} | "
        f"{_format_plain(metrics.get('mean_episode_time_s'))} | "
        f"{_format_plain(metrics.get('any_brush_rate'))}% | "
        f"{_format_plain(metrics.get('objective_slow_brush_arrival_rate'))}% | "
        f"{_format_plain(metrics.get('timeout_rate'))}% | "
        f"{_format_plain(metrics.get('hard_collision_rate'))}% |"
      )
    if task_idx < len(task_labels) - 1:
      tex_lines.append("\\midrule")
  tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / "main_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")

  gap_rows = _compute_gap_rows(summary)
  md_lines.extend(
    [
      "",
      "## PP-NoPP Gap",
      "",
      "| Task | T1-DR success gap | T1-Odom success gap | T1-R1 success gap |",
      "|---|---:|---:|---:|",
    ]
  )
  gap_tex = [
    "\\begin{tabular}{l c c c}",
    "\\toprule",
    "Task & T1--DR gap & T1--Odom gap & T1--R1 gap \\\\",
    "\\midrule",
  ]
  for row in gap_rows:
    md_lines.append(
      f"| {row['task_label']} | "
      f"{_format_gap(float(row['t1_dr_gap']))} | "
      f"{_format_gap(float(row['t1_odom_gap']))} | "
      f"{_format_gap(float(row['t1_r1_gap']))} |"
    )
    gap_tex.append(
      f"{row['task_label']} & "
      f"{_format_gap_tex(float(row['t1_dr_gap']))} & "
      f"{_format_gap_tex(float(row['t1_odom_gap']))} & "
      f"{_format_gap_tex(float(row['t1_r1_gap']))} \\\\"
    )
  gap_tex.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / "pp_nopp_gap_table.tex").write_text("\n".join(gap_tex), encoding="utf-8")
  (run_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _write_safety_table(run_dir: Path, summary: dict) -> None:
  conditions = summary["conditions"]
  by_key = {(entry["task_label"], entry["method"]): entry for entry in conditions}
  methods = [
    "R1 (Backbone-only)",
    "T1 (PP upper bound)",
    "Ours (DR)",
    "Ours (Odom)",
  ]

  md_lines = [
    "",
    "## Safety / Brush Diagnostics",
    "",
    "| Task | Method | Any brush | T-rel. slow-brush | T-rel. late-clean | Hard collision | Obj. on-time | Contact brush |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  tex_lines = [
    "\\begin{tabular}{l l c c c c c c}",
    "\\toprule",
    "Task & Method & Any brush & T-rel. slow-brush & T-rel. late-clean & Hard coll. & Obj. on-time & Contact brush \\\\",
    "\\midrule",
  ]
  pct = "\\%"

  for task_idx, task_label in enumerate(summary["task_labels"]):
    for method_idx, method in enumerate(methods):
      entry = by_key.get((task_label, method))
      metrics = entry.get("aggregate", {}) if entry else {}
      task_cell = task_label if method_idx == 0 else ""
      md_lines.append(
        f"| {task_label} | {method} | "
        f"{_format_plain_pct(metrics.get('any_brush_rate'))} | "
        f"{_format_plain_pct(metrics.get('slow_brush_arrival_rate'))} | "
        f"{_format_plain_pct(metrics.get('late_clean_arrival_rate'))} | "
        f"{_format_plain_pct(metrics.get('hard_collision_rate'))} | "
        f"{_format_plain_pct(metrics.get('objective_on_time_arrival_rate'))} | "
        f"{_format_plain_pct(metrics.get('contact_brush_rate'))} |"
      )
      tex_lines.append(
        f"{task_cell} & {method} & "
        f"{_format_mean_std(metrics.get('any_brush_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('slow_brush_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('late_clean_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('hard_collision_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('objective_on_time_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('contact_brush_rate'), pct)} \\\\"
      )
    if task_idx < len(summary["task_labels"]) - 1:
      tex_lines.append("\\midrule")

  md_lines.extend(
    [
      "",
      "Objective arrival metrics use the per-episode generated waypoint/A* path budget.",
      "Teacher-relative arrival metrics remain in summary.json as T1-calibrated reference diagnostics.",
      "Slow-brush arrival is late arrival with any brush, including envelope or contact brush.",
    ]
  )
  tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
  with (run_dir / "summary.md").open("a", encoding="utf-8") as f:
    f.write("\n".join(md_lines) + "\n")
  (run_dir / "safety_table.tex").write_text(
    "\n".join(tex_lines) + "\n", encoding="utf-8"
  )
  (run_dir / "diagnostic_table.tex").write_text(
    "\n".join(tex_lines) + "\n", encoding="utf-8"
  )


def _compute_gap_rows(summary: dict) -> list[dict[str, float | str]]:
  by_key = {
    (entry["task_label"], entry["method"]): entry for entry in summary["conditions"]
  }

  def metric_mean(task_label: str, method: str, metric: str) -> float:
    entry = by_key.get((task_label, method), {})
    aggregate = entry.get("aggregate", {})
    data = aggregate.get(metric)
    if not isinstance(data, dict):
      return float("nan")
    return float(data.get("mean", float("nan")))

  rows = []
  for task_label in summary["task_labels"]:
    t1 = metric_mean(task_label, "T1 (PP upper bound)", "success_rate")
    dr = metric_mean(task_label, "Ours (DR)", "success_rate")
    odom = metric_mean(task_label, "Ours (Odom)", "success_rate")
    r1 = metric_mean(task_label, "R1 (Backbone-only)", "success_rate")
    rows.append(
      {
        "task_label": task_label,
        "t1_dr_gap": float(t1 - dr),
        "t1_odom_gap": float(t1 - odom),
        "t1_ours_gap": float(t1 - dr),
        "t1_r1_gap": float(t1 - r1),
      }
    )
  return rows


def _load_ablation_variants(path: str) -> list[dict]:
  if not path:
    return []
  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  if isinstance(data, list):
    return data
  return list(data.get("variants", []))


def _resolve_variant_checkpoint(variant: dict, spec: TaskSpec) -> str:
  use_odometry = bool(variant.get("use_odometry", False))
  default_checkpoint = (
    spec.odom_local_planner_checkpoint
    if use_odometry
    else spec.local_planner_checkpoint
  )
  checkpoints = variant.get("checkpoints")
  if isinstance(checkpoints, dict):
    value = checkpoints.get(spec.label) or checkpoints.get(spec.label.lower())
    if value:
      return str(value)
  value = variant.get("checkpoint", "")
  if value == "default":
    return default_checkpoint
  if not value and variant.get("name", "").lower() in {"full", "ours"}:
    return default_checkpoint
  return str(value)


def _variant_task_specs(
  variant: dict,
  task_specs: tuple[TaskSpec, ...],
) -> list[TaskSpec]:
  requested = variant.get("tasks") or variant.get("task_labels")
  if not requested:
    return list(task_specs)
  labels = {str(label).strip().lower() for label in requested}
  return [spec for spec in task_specs if spec.label.lower() in labels]


def _write_ablation_table(run_dir: Path, ablation_summary: dict) -> None:
  rows = [
    row
    for row in ablation_summary.get("rows", [])
    if str(row.get("variant", "")).lower() != "dr no curriculum"
  ]
  lines = [
    "\\begin{tabular}{l l c c c c c c}",
    "\\toprule",
    "Mode & Variant & Success & Obj. on-time & Obj. slow-brush & Any brush & Hard coll. & PP--NoPP gap \\\\",
    "\\midrule",
  ]
  for row in rows:
    mode = "Odom" if row.get("use_odometry") else "DR"
    variant = row["variant"]
    task_label = row["task_label"]
    variant_cell = f"{variant} ({task_label})"
    if row.get("status") == "missing":
      lines.append(
        f"{mode} & {variant_cell} & MISSING & MISSING & MISSING & MISSING & MISSING & MISSING \\\\"
      )
    else:
      lines.append(
        f"{mode} & {variant_cell} & "
        f"{row['success_rate']:.1f}\\% & "
        f"{row['objective_on_time_arrival_rate']:.1f}\\% & "
        f"{row['objective_slow_brush_arrival_rate']:.1f}\\% & "
        f"{row['any_brush_rate']:.1f}\\% & "
        f"{row['hard_collision_rate']:.1f}\\% & "
        f"{row['pp_nopp_gap']:.1f}\\% \\\\"
      )
  if not rows:
    lines.append("\\multicolumn{8}{c}{No ablation config provided.} \\\\")
  lines.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / "ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")


def _print_plan(
  conditions: list[Condition],
  args: PaperExperimentArgs,
  task_specs: tuple[TaskSpec, ...],
  ablation_variants: list[dict],
  experiment_groups: Sequence[str],
) -> None:
  episodes = 1 if args.smoke else args.episodes_per_seed
  envs = 1 if args.smoke else args.num_envs
  print("Paper benchmark matrix:")
  print(f"  preset: {args.preset}")
  print(f"  experiment_groups: {' '.join(experiment_groups)}")
  print(f"  seeds: {list(args.seeds)}")
  print(f"  episodes_per_seed: {episodes}")
  print(f"  num_envs: {envs}")
  print(f"  locomotion checkpoint: {args.locomotion_checkpoint}")
  if conditions:
    print("Main conditions:")
    for cond in conditions:
      ckpt = cond.local_planner_checkpoint or "-"
      flags = []
      if cond.use_odometry:
        flags.append("odom")
      if cond.disable_safety_envelope:
        flags.append("no-safety")
      flag_text = f" [{', '.join(flags)}]" if flags else ""
      print(
        f"  - {cond.task_label:10s} | {cond.method:20s}{flag_text} | "
        f"{cond.task} | {ckpt}"
      )
  if "ablation" in experiment_groups and ablation_variants:
    print("Ablation variants:")
    for variant in ablation_variants:
      name = variant.get("name", "<unnamed>")
      for spec in _variant_task_specs(variant, task_specs):
        ckpt = _resolve_variant_checkpoint(variant, spec)
        status = "ok" if ckpt and os.path.isfile(ckpt) else "missing"
        flags = []
        if bool(variant.get("use_odometry", False)):
          flags.append("odom")
        if bool(variant.get("disable_safety_envelope", False)):
          flags.append("no-safety")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        print(f"  - {name} / {spec.label}{flag_text}: {ckpt or '-'} ({status})")


DirectBudgetMap = dict[tuple[str, int], dict[str, float]]
RecordMap = dict[tuple[str, str, int], dict]
ArrivalBudgetMap = dict[tuple[str, int], dict[str, float | bool | int]]
EnvelopeThresholdMap = dict[str, dict[str, float | bool | str]]


def _parse_calibration_key(key: str) -> tuple[str, int]:
  try:
    task_label, seed_text = key.rsplit("/seed_", 1)
    return task_label, int(seed_text)
  except ValueError as exc:
    raise SystemExit(f"Invalid calibration key in summary.json: {key}") from exc


def _calibration_maps_from_summary(
  summary: dict,
) -> tuple[DirectBudgetMap, ArrivalBudgetMap, EnvelopeThresholdMap]:
  direct_budget_map: DirectBudgetMap = {}
  for key, value in summary.get("direct_budget_calibration", {}).items():
    if not isinstance(value, dict):
      continue
    task_label, seed = _parse_calibration_key(str(key))
    direct_budget_map[(task_label, seed)] = {
      str(metric): float(metric_value)
      for metric, metric_value in value.items()
      if isinstance(metric_value, (float, int))
    }

  arrival_budget_map: ArrivalBudgetMap = {}
  for key, value in summary.get("arrival_budget_calibration", {}).items():
    if not isinstance(value, dict):
      continue
    task_label, seed = _parse_calibration_key(str(key))
    arrival_budget_map[(task_label, seed)] = {
      str(metric): metric_value
      for metric, metric_value in value.items()
      if isinstance(metric_value, (float, int, bool))
    }

  envelope_threshold_map: EnvelopeThresholdMap = {
    str(task_label): dict(value)
    for task_label, value in summary.get("envelope_brush_calibration", {}).items()
    if isinstance(value, dict)
  }
  return direct_budget_map, arrival_budget_map, envelope_threshold_map


def _load_main_summary_for_ablations(
  run_dir: Path,
) -> tuple[dict, DirectBudgetMap, ArrivalBudgetMap, EnvelopeThresholdMap]:
  summary_path = run_dir / "summary.json"
  if not summary_path.is_file():
    raise SystemExit(
      "Ablation-only runs require an existing main summary at "
      f"{summary_path}. Run with --experiment-groups 'main ablation' first, "
      "or rerun the wrapper with the same --run-name/--output-root and "
      "--exact-run-name."
    )
  with open(summary_path, encoding="utf-8") as f:
    summary = json.load(f)
  if not summary.get("conditions"):
    raise SystemExit(f"Main summary has no conditions: {summary_path}")
  (
    direct_budget_map,
    arrival_budget_map,
    envelope_threshold_map,
  ) = _calibration_maps_from_summary(summary)
  return summary, direct_budget_map, arrival_budget_map, envelope_threshold_map


def _primary_direct_budget(
  direct_budget_map: DirectBudgetMap,
  task_label: str,
  seed: int,
  primary_quantile: float,
) -> float | None:
  budgets = direct_budget_map.get((task_label, seed), {})
  return budgets.get(direct_quantile_key(primary_quantile))


def _primary_arrival_budget(
  arrival_budget_map: ArrivalBudgetMap,
  task_label: str,
  seed: int,
) -> float | None:
  data = arrival_budget_map.get((task_label, seed), {})
  value = data.get("budget_effective_distance")
  return float(value) if isinstance(value, (float, int)) else None


def _envelope_threshold_for_task(
  envelope_threshold_map: EnvelopeThresholdMap,
  task_label: str,
) -> tuple[float | None, float | None, bool]:
  data = envelope_threshold_map.get(task_label, {})
  scale = data.get("scale_threshold")
  min_time = data.get("min_time_s")
  existing_scale_only = bool(data.get("existing_scale_only", False))
  scale_value = float(scale) if isinstance(scale, (float, int)) else None
  min_time_value = float(min_time) if isinstance(min_time, (float, int)) else None
  return scale_value, min_time_value, existing_scale_only


def _calibrate_from_t1_records(
  *,
  record_entries: RecordMap,
  task_specs: tuple[TaskSpec, ...],
  seeds: tuple[int, ...],
  args: PaperExperimentArgs,
  direct_budget_map: DirectBudgetMap,
) -> tuple[ArrivalBudgetMap, EnvelopeThresholdMap]:
  arrival_budget_map: ArrivalBudgetMap = {}
  envelope_threshold_map: EnvelopeThresholdMap = {}
  scale_thresholds = safety_scale_thresholds_from_string(
    args.safety_brake_scale_thresholds
  )
  for spec in task_specs:
    all_t1_records = []
    for seed in seeds:
      data = record_entries.get((spec.label, "T1 (PP upper bound)", int(seed)))
      if data is None:
        continue
      records = data["records"]
      direct_budget_map[(spec.label, int(seed))] = calibrate_direct_budgets(
        records,
        quantiles=DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
        min_success_episodes=args.min_direct_calibration_episodes,
      )
      arrival_budget_map[(spec.label, int(seed))] = calibrate_arrival_budget_for_target(
        records,
        target_rate=args.arrival_target_rate,
        min_success_episodes=args.min_direct_calibration_episodes,
      )
      all_t1_records.extend(records)
    envelope_threshold_map[spec.label] = calibrate_envelope_brush_threshold(
      all_t1_records,
      target_rate=args.envelope_target_rate,
      scale_thresholds=scale_thresholds,
      existing_scale_only=False,
    )
  return arrival_budget_map, envelope_threshold_map


def _summarize_records(
  *,
  record_entries: RecordMap,
  conditions: list[Condition],
  args: PaperExperimentArgs,
  direct_budget_map: DirectBudgetMap,
  arrival_budget_map: ArrivalBudgetMap,
  envelope_threshold_map: EnvelopeThresholdMap,
  run_dir: Path,
) -> list[dict]:
  condition_entries: list[dict] = []
  for condition in conditions:
    seed_summaries: list[dict[str, float]] = []
    seed_entries: list[dict] = []
    missing = False
    for seed in args.seeds:
      data = record_entries.get((condition.task_label, condition.method, int(seed)))
      if data is None:
        missing = True
        continue
      records = data["records"]
      apply_direct_budget(
        records,
        budget_effective_distance=_primary_arrival_budget(
          arrival_budget_map,
          condition.task_label,
          int(seed),
        ),
        quantile=args.direct_calibration_quantile,
      )
      apply_objective_budget(
        records,
        objective_timeout_margin_s=args.objective_timeout_margin_s,
      )
      scale, min_time, existing_scale_only = _envelope_threshold_for_task(
        envelope_threshold_map, condition.task_label
      )
      seed_summary = compute_summary(
        records,
        data["num_goals"],
        direct_budgets=direct_budget_map.get((condition.task_label, int(seed)), {}),
        envelope_scale_threshold=scale,
        envelope_min_time_s=min_time,
        existing_scale_only=existing_scale_only,
      )
      seed_summaries.append(seed_summary)
      seed_entries.append(
        {
          "seed": seed,
          "summary": seed_summary,
          "num_goals": data["num_goals"],
          "key_goal_indices": data["key_goal_indices"],
          "elapsed_s": data["elapsed_s"],
          "direct_budget_calibration": direct_budget_map.get(
            (condition.task_label, int(seed)), {}
          ),
          "arrival_budget_calibration": arrival_budget_map.get(
            (condition.task_label, int(seed)), {}
          ),
        }
      )

    if missing and not seed_summaries:
      condition_entries.append(
        {
          "task_label": condition.task_label,
          "method": condition.method,
          "task": condition.task,
          "local_planner_checkpoint": condition.local_planner_checkpoint,
          "use_odometry": condition.use_odometry,
          "include_joint_vel": condition.include_joint_vel,
          "disable_safety_envelope": condition.disable_safety_envelope,
          "status": "missing",
          "aggregate": {},
        }
      )
      continue

    entry = {
      "task_label": condition.task_label,
      "method": condition.method,
      "task": condition.task,
      "local_planner_checkpoint": condition.local_planner_checkpoint,
      "use_odometry": condition.use_odometry,
      "include_joint_vel": condition.include_joint_vel,
      "disable_safety_envelope": condition.disable_safety_envelope,
      "seeds": seed_entries,
      "aggregate": aggregate_seed_summaries(seed_summaries),
    }
    write_json(
      run_dir
      / f"summary_{condition.task_label}_{condition.method}.json".replace(
        " ", "_"
      ).replace("/", "_"),
      entry,
    )
    condition_entries.append(entry)
  return condition_entries


def _run_condition(
  *,
  condition: Condition,
  args: PaperExperimentArgs,
  run_dir: Path,
  episodes_per_seed: int,
  num_envs: int,
  device: torch.device,
  jsonl_path: Path,
  direct_budget_map: DirectBudgetMap,
) -> RecordMap:
  record_entries: RecordMap = {}

  for seed in args.seeds:
    print()
    print("=" * 80)
    print(f"{condition.task_label} | {condition.method} | seed={seed}")
    print("=" * 80)
    result = run_policy_eval(
      task=condition.task,
      locomotion_checkpoint=args.locomotion_checkpoint,
      local_planner_checkpoint=condition.local_planner_checkpoint,
      num_envs=num_envs,
      num_episodes=episodes_per_seed,
      raycast_obs_names=args.raycast_obs_names,
      device=device,
      seed=seed,
      mode="test",
      correction_gain=args.correction_gain,
      use_odometry=condition.use_odometry,
      include_joint_vel=condition.include_joint_vel,
      disable_safety_envelope=condition.disable_safety_envelope,
      capture_trajectories=args.capture_trajectories,
      max_trajectories=args.max_trajectories_per_condition,
      safety_brake_scale_threshold=args.safety_brake_scale_threshold,
      safety_brake_scale_thresholds=safety_scale_thresholds_from_string(
        args.safety_brake_scale_thresholds
      ),
      safety_brake_min_time_s=args.safety_brake_min_time_s,
      objective_timeout_margin_s=args.objective_timeout_margin_s,
    )
    if condition.method == "T1 (PP upper bound)":
      direct_budget_map[(condition.task_label, int(seed))] = calibrate_direct_budgets(
        result.records,
        quantiles=DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
        min_success_episodes=args.min_direct_calibration_episodes,
      )
    rows = []
    for episode_idx, record in enumerate(result.records):
      rows.append(
        {
          "task_label": condition.task_label,
          "method": condition.method,
          "task": condition.task,
          "use_odometry": condition.use_odometry,
          "include_joint_vel": condition.include_joint_vel,
          "disable_safety_envelope": condition.disable_safety_envelope,
          "seed": seed,
          "episode_idx": episode_idx,
          "record": record_to_dict(record),
        }
      )
    append_jsonl(jsonl_path, rows)
    record_entries[(condition.task_label, condition.method, int(seed))] = {
      "records": result.records,
      "num_goals": result.num_goals,
      "key_goal_indices": result.key_goal_indices,
      "elapsed_s": result.elapsed_s,
    }

  return record_entries


def _run_ablations(
  *,
  variants: list[dict],
  task_specs: tuple[TaskSpec, ...],
  args: PaperExperimentArgs,
  run_dir: Path,
  episodes_per_seed: int,
  num_envs: int,
  device: torch.device,
  jsonl_path: Path,
  main_summary: dict,
  direct_budget_map: DirectBudgetMap,
  arrival_budget_map: ArrivalBudgetMap,
  envelope_threshold_map: EnvelopeThresholdMap,
) -> dict:
  rows: list[dict] = []
  if not variants:
    return {"rows": rows}

  by_key = {
    (entry["task_label"], entry["method"]): entry
    for entry in main_summary["conditions"]
  }
  for variant in variants:
    name = str(variant.get("name", "Variant"))
    use_odometry = bool(variant.get("use_odometry", False))
    include_joint_vel = bool(variant.get("include_joint_vel", not use_odometry))
    disable_safety_envelope = bool(variant.get("disable_safety_envelope", False))
    for spec in _variant_task_specs(variant, task_specs):
      ckpt = _resolve_variant_checkpoint(variant, spec)
      if not ckpt or not os.path.isfile(ckpt):
        rows.append(
          {
            "variant": name,
            "task_label": spec.label,
            "checkpoint": ckpt,
            "use_odometry": use_odometry,
            "include_joint_vel": include_joint_vel,
            "disable_safety_envelope": disable_safety_envelope,
            "status": "missing",
          }
        )
        continue
      condition = Condition(
        spec.label,
        f"Ablation:{name}",
        spec.nopp_task,
        ckpt,
        use_odometry=use_odometry,
        include_joint_vel=include_joint_vel,
        disable_safety_envelope=disable_safety_envelope,
      )
      entry = _run_condition(
        condition=condition,
        args=args,
        run_dir=run_dir,
        episodes_per_seed=episodes_per_seed,
        num_envs=num_envs,
        device=device,
        jsonl_path=jsonl_path,
        direct_budget_map=direct_budget_map,
      )
      entries = _summarize_records(
        record_entries=entry,
        conditions=[condition],
        args=args,
        direct_budget_map=direct_budget_map,
        arrival_budget_map=arrival_budget_map,
        envelope_threshold_map=envelope_threshold_map,
        run_dir=run_dir,
      )
      summary_entry = entries[0]
      success = _metric_mean(summary_entry, "success_rate")
      t1_success = by_key[(spec.label, "T1 (PP upper bound)")]["aggregate"][
        "success_rate"
      ]["mean"]
      rows.append(
        {
          "variant": name,
          "task_label": spec.label,
          "checkpoint": ckpt,
          "use_odometry": use_odometry,
          "include_joint_vel": include_joint_vel,
          "disable_safety_envelope": disable_safety_envelope,
          "status": "ok",
          "success_rate": success,
          "objective_on_time_arrival_rate": _metric_mean(
            summary_entry, "objective_on_time_arrival_rate"
          ),
          "teacher_relative_on_time_arrival_rate": _metric_mean(
            summary_entry, "teacher_relative_on_time_arrival_rate"
          ),
          "on_time_arrival_rate": _metric_mean(
            summary_entry, "teacher_relative_on_time_arrival_rate"
          ),
          "objective_late_arrival_rate": _metric_mean(
            summary_entry, "objective_late_arrival_rate"
          ),
          "late_arrival_rate": _metric_mean(
            summary_entry, "teacher_relative_late_arrival_rate"
          ),
          "objective_slow_brush_arrival_rate": _metric_mean(
            summary_entry, "objective_slow_brush_arrival_rate"
          ),
          "slow_brush_arrival_rate": _metric_mean(
            summary_entry, "teacher_relative_slow_brush_arrival_rate"
          ),
          "late_clean_arrival_rate": _metric_mean(
            summary_entry, "late_clean_arrival_rate"
          ),
          "any_brush_rate": _metric_mean(summary_entry, "any_brush_rate"),
          "contact_brush_rate": _metric_mean(summary_entry, "contact_brush_rate"),
          "hard_collision_rate": _metric_mean(summary_entry, "hard_collision_rate"),
          "collision_rate": _metric_mean(summary_entry, "collision_rate"),
          "pp_nopp_gap": float(t1_success - success),
        }
      )
  return {"rows": rows}


def main() -> None:
  args = tyro.cli(PaperExperimentArgs, config=mjlab.TYRO_FLAGS)
  experiment_groups = _parse_experiment_groups(args.experiment_groups)
  task_specs = _task_specs_for_preset(args.preset, args=args)
  conditions = _main_conditions(task_specs) if "main" in experiment_groups else []
  ablation_variants = (
    _load_ablation_variants(args.ablation_config)
    if "ablation" in experiment_groups
    else []
  )

  if args.smoke:
    args.episodes_per_seed = 1
    args.num_envs = 1

  _print_plan(conditions, args, task_specs, ablation_variants, experiment_groups)
  if args.dry_run:
    return

  if not os.path.isfile(args.locomotion_checkpoint):
    raise SystemExit(f"Missing locomotion checkpoint: {args.locomotion_checkpoint}")

  run_dir = _make_run_dir(args)
  jsonl_path = run_dir / "episodes.jsonl"
  device = torch.device(args.device)

  if "main" in experiment_groups:
    ensure_empty_jsonl(jsonl_path)
    direct_budget_map: DirectBudgetMap = {}
    record_entries: RecordMap = {}

    for condition in conditions:
      if condition.local_planner_checkpoint and not os.path.isfile(
        condition.local_planner_checkpoint
      ):
        continue
      record_entries.update(
        _run_condition(
          condition=condition,
          args=args,
          run_dir=run_dir,
          episodes_per_seed=args.episodes_per_seed,
          num_envs=args.num_envs,
          device=device,
          jsonl_path=jsonl_path,
          direct_budget_map=direct_budget_map,
        )
      )

    arrival_budget_map, envelope_threshold_map = _calibrate_from_t1_records(
      record_entries=record_entries,
      task_specs=task_specs,
      seeds=args.seeds,
      args=args,
      direct_budget_map=direct_budget_map,
    )
    condition_entries = _summarize_records(
      record_entries=record_entries,
      conditions=conditions,
      args=args,
      direct_budget_map=direct_budget_map,
      arrival_budget_map=arrival_budget_map,
      envelope_threshold_map=envelope_threshold_map,
      run_dir=run_dir,
    )

    summary = {
      "timestamp": datetime.now().isoformat(timespec="seconds"),
      "preset": args.preset,
      "task_labels": [spec.label for spec in task_specs],
      "args": sanitize_for_json(args.__dict__),
      "conditions": condition_entries,
      "direct_budget_calibration": {
        f"{task_label}/seed_{seed}": budgets
        for (task_label, seed), budgets in direct_budget_map.items()
      },
      "arrival_budget_calibration": {
        f"{task_label}/seed_{seed}": data
        for (task_label, seed), data in arrival_budget_map.items()
      },
      "envelope_brush_calibration": envelope_threshold_map,
    }
    summary["pp_nopp_gap"] = _compute_gap_rows(summary)
    write_json(run_dir / "summary.json", summary)
    _write_main_tables(run_dir, summary)
    _write_safety_table(run_dir, summary)
  else:
    (
      summary,
      direct_budget_map,
      arrival_budget_map,
      envelope_threshold_map,
    ) = _load_main_summary_for_ablations(run_dir)
    if jsonl_path.exists():
      print(
        f"  [Info] Reusing existing episode log for ablation-only run: {jsonl_path}"
      )

  if "ablation" in experiment_groups:
    ablation_summary = _run_ablations(
      variants=ablation_variants,
      task_specs=task_specs,
      args=args,
      run_dir=run_dir,
      episodes_per_seed=args.episodes_per_seed,
      num_envs=args.num_envs,
      device=device,
      jsonl_path=jsonl_path,
      main_summary=summary,
      direct_budget_map=direct_budget_map,
      arrival_budget_map=arrival_budget_map,
      envelope_threshold_map=envelope_threshold_map,
    )
    write_json(run_dir / "ablation_summary.json", ablation_summary)
    _write_ablation_table(run_dir, ablation_summary)

  print()
  print(f"Paper benchmark outputs written to: {run_dir}")
  print(f"  episodes: {jsonl_path}")
  print(f"  summary:  {run_dir / 'summary.json'}")
  if "main" in experiment_groups:
    print(f"  table:    {run_dir / 'main_table.tex'}")
  if "ablation" in experiment_groups:
    print(f"  ablation: {run_dir / 'ablation_table.tex'}")


if __name__ == "__main__":
  main()
