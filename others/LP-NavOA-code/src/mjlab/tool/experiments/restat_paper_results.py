#!/usr/bin/env python3
"""Post-hoc restat paper benchmark episodes with the v2 arrival/brush metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.tool.experiments.eval_utils import (
  DEFAULT_ARRIVAL_TARGET_RATE,
  DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
  DEFAULT_ENVELOPE_TARGET_RATE,
  DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS,
  EvalEpisodeRecord,
  aggregate_seed_summaries,
  apply_direct_budget,
  apply_objective_budget,
  calibrate_arrival_budget_for_target,
  calibrate_direct_budgets,
  calibrate_envelope_brush_threshold,
  compute_summary,
  record_from_dict,
  safety_scale_thresholds_from_string,
  sanitize_for_json,
  write_json,
)
from mjlab.tool.experiments.run_paper_experiments import (
  _compute_gap_rows,
  _format_gap,
  _format_gap_tex,
  _format_mean_std,
  _format_plain,
  _format_plain_pct,
)

METHOD_ORDER = (
  "R1 (Backbone-only)",
  "T1 (PP upper bound)",
  "Ours (DR)",
  "Ours (Odom)",
)

DISPLAY_METHOD_ORDER = (
  "R1 (Backbone-only)",
  "Ours (DR)",
  "Ours (Odom)",
)


@dataclass
class RestatArgs:
  """Arguments for post-hoc v2 paper-result restat."""

  run_dir: str
  """Benchmark run directory containing episodes.jsonl."""

  arrival_target_rate: float = DEFAULT_ARRIVAL_TARGET_RATE
  """Target T1 on-time arrival rate for calibrated arrival budgets."""

  envelope_target_rate: float = DEFAULT_ENVELOPE_TARGET_RATE
  """Target T1 envelope-brush rate."""

  crawl_mode: Literal["any-brush"] = "any-brush"
  """Slow-brush arrival definition. Currently fixed to late arrival plus any brush."""

  existing_scale_only: bool = False
  """Use only legacy safety_envelope_brake_time_s when threshold grids are absent."""

  safety_brake_scale_thresholds: str = " ".join(
    str(value) for value in DEFAULT_SAFETY_BRAKE_SCALE_THRESHOLDS
  )
  """Scale thresholds to consider when per-threshold times exist."""

  output_suffix: str = "_v2"
  """Suffix used for generated summary/table files."""

  min_direct_calibration_episodes: int = 5
  """Minimum T1 successes needed to calibrate budgets."""

  objective_timeout_margin_s: float = 10.0
  """Paper objective-arrival margin added to nominal path length / target speed."""


def _load_rows(path: Path) -> list[dict]:
  rows = []
  with path.open(encoding="utf-8") as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def _task_labels(rows: list[dict]) -> list[str]:
  labels: list[str] = []
  for row in rows:
    label = str(row["task_label"])
    if label not in labels and not str(row["method"]).startswith("Ablation:"):
      labels.append(label)
  return labels


def _build_record_groups(
  rows: list[dict],
) -> dict[tuple[str, str, int], list[EvalEpisodeRecord]]:
  groups: dict[tuple[str, str, int], list[EvalEpisodeRecord]] = defaultdict(list)
  for row in rows:
    key = (str(row["task_label"]), str(row["method"]), int(row["seed"]))
    groups[key].append(record_from_dict(row["record"]))
  return groups


def _calibrate(
  groups: dict[tuple[str, str, int], list[EvalEpisodeRecord]],
  task_labels: list[str],
  seeds: list[int],
  args: RestatArgs,
) -> tuple[dict, dict, dict]:
  direct_budget_map = {}
  arrival_budget_map = {}
  envelope_threshold_map = {}
  scale_thresholds = safety_scale_thresholds_from_string(
    args.safety_brake_scale_thresholds
  )
  for task_label in task_labels:
    all_t1_records: list[EvalEpisodeRecord] = []
    for seed in seeds:
      records = groups.get((task_label, "T1 (PP upper bound)", seed), [])
      if not records:
        continue
      direct_budget_map[(task_label, seed)] = calibrate_direct_budgets(
        records,
        quantiles=DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
        min_success_episodes=args.min_direct_calibration_episodes,
      )
      arrival_budget_map[(task_label, seed)] = calibrate_arrival_budget_for_target(
        records,
        target_rate=args.arrival_target_rate,
        min_success_episodes=args.min_direct_calibration_episodes,
      )
      all_t1_records.extend(records)
    envelope_threshold_map[task_label] = calibrate_envelope_brush_threshold(
      all_t1_records,
      target_rate=args.envelope_target_rate,
      scale_thresholds=scale_thresholds,
      existing_scale_only=args.existing_scale_only,
    )
  return direct_budget_map, arrival_budget_map, envelope_threshold_map


def _budget_value(arrival_budget_map: dict, task_label: str, seed: int) -> float | None:
  data = arrival_budget_map.get((task_label, seed), {})
  value = data.get("budget_effective_distance")
  return float(value) if isinstance(value, (float, int)) else None


def _envelope_values(envelope_threshold_map: dict, task_label: str) -> tuple:
  data = envelope_threshold_map.get(task_label, {})
  scale = data.get("scale_threshold")
  min_time = data.get("min_time_s")
  return (
    float(scale) if isinstance(scale, (float, int)) else None,
    float(min_time) if isinstance(min_time, (float, int)) else None,
    bool(data.get("existing_scale_only", False)),
  )


def _summarize_main(
  groups: dict[tuple[str, str, int], list[EvalEpisodeRecord]],
  rows: list[dict],
  task_labels: list[str],
  seeds: list[int],
  direct_budget_map: dict,
  arrival_budget_map: dict,
  envelope_threshold_map: dict,
  objective_timeout_margin_s: float | None,
) -> list[dict]:
  row_meta = {
    (str(row["task_label"]), str(row["method"])): row
    for row in rows
    if not str(row["method"]).startswith("Ablation:")
  }
  conditions = []
  for task_label in task_labels:
    for method in METHOD_ORDER:
      seed_summaries = []
      seed_entries = []
      for seed in seeds:
        records = groups.get((task_label, method, seed), [])
        if not records:
          continue
        apply_direct_budget(
          records,
          budget_effective_distance=_budget_value(arrival_budget_map, task_label, seed),
        )
        apply_objective_budget(
          records,
          objective_timeout_margin_s=objective_timeout_margin_s,
        )
        scale, min_time, existing_scale_only = _envelope_values(
          envelope_threshold_map, task_label
        )
        summary = compute_summary(
          records,
          direct_budgets=direct_budget_map.get((task_label, seed), {}),
          envelope_scale_threshold=scale,
          envelope_min_time_s=min_time,
          existing_scale_only=existing_scale_only,
        )
        seed_summaries.append(summary)
        seed_entries.append(
          {
            "seed": seed,
            "summary": summary,
            "arrival_budget_calibration": arrival_budget_map.get(
              (task_label, seed), {}
            ),
            "direct_budget_calibration": direct_budget_map.get((task_label, seed), {}),
          }
        )
      meta = row_meta.get((task_label, method), {})
      conditions.append(
        {
          "task_label": task_label,
          "method": method,
          "task": meta.get("task", ""),
          "use_odometry": bool(meta.get("use_odometry", False)),
          "include_joint_vel": bool(meta.get("include_joint_vel", True)),
          "disable_safety_envelope": bool(meta.get("disable_safety_envelope", False)),
          "seeds": seed_entries,
          "aggregate": aggregate_seed_summaries(seed_summaries),
        }
      )
  return conditions


def _ablation_rows(
  groups: dict[tuple[str, str, int], list[EvalEpisodeRecord]],
  rows: list[dict],
  seeds: list[int],
  direct_budget_map: dict,
  arrival_budget_map: dict,
  envelope_threshold_map: dict,
  main_conditions: list[dict],
  objective_timeout_margin_s: float | None,
) -> list[dict]:
  meta_by_key = {}
  for row in rows:
    method = str(row["method"])
    if method.startswith("Ablation:"):
      meta_by_key[(str(row["task_label"]), method)] = row
  t1_success = {
    entry["task_label"]: entry.get("aggregate", {})
    .get("success_rate", {})
    .get("mean", float("nan"))
    for entry in main_conditions
    if entry["method"] == "T1 (PP upper bound)"
  }
  out = []
  for (task_label, method), meta in sorted(meta_by_key.items()):
    seed_summaries = []
    for seed in seeds:
      records = groups.get((task_label, method, seed), [])
      if not records:
        continue
      apply_direct_budget(
        records,
        budget_effective_distance=_budget_value(arrival_budget_map, task_label, seed),
      )
      apply_objective_budget(
        records,
        objective_timeout_margin_s=objective_timeout_margin_s,
      )
      scale, min_time, existing_scale_only = _envelope_values(
        envelope_threshold_map, task_label
      )
      seed_summaries.append(
        compute_summary(
          records,
          direct_budgets=direct_budget_map.get((task_label, seed), {}),
          envelope_scale_threshold=scale,
          envelope_min_time_s=min_time,
          existing_scale_only=existing_scale_only,
        )
      )
    aggregate = aggregate_seed_summaries(seed_summaries)
    success = aggregate.get("success_rate", {}).get("mean", float("nan"))
    variant = method.replace("Ablation:", "", 1)
    out.append(
      {
        "variant": variant,
        "task_label": task_label,
        "use_odometry": bool(meta.get("use_odometry", False)),
        "include_joint_vel": bool(meta.get("include_joint_vel", True)),
        "disable_safety_envelope": bool(meta.get("disable_safety_envelope", False)),
        "status": "ok",
        "success_rate": success,
        "objective_on_time_arrival_rate": aggregate.get(
          "objective_on_time_arrival_rate", {}
        ).get("mean", float("nan")),
        "teacher_relative_on_time_arrival_rate": aggregate.get(
          "teacher_relative_on_time_arrival_rate", {}
        ).get("mean", float("nan")),
        "on_time_arrival_rate": aggregate.get(
          "teacher_relative_on_time_arrival_rate", {}
        ).get("mean", float("nan")),
        "objective_late_arrival_rate": aggregate.get(
          "objective_late_arrival_rate", {}
        ).get("mean", float("nan")),
        "late_arrival_rate": aggregate.get(
          "teacher_relative_late_arrival_rate", {}
        ).get("mean", float("nan")),
        "objective_slow_brush_arrival_rate": aggregate.get(
          "objective_slow_brush_arrival_rate", {}
        ).get("mean", float("nan")),
        "slow_brush_arrival_rate": aggregate.get(
          "teacher_relative_slow_brush_arrival_rate", {}
        ).get("mean", float("nan")),
        "late_clean_arrival_rate": aggregate.get("late_clean_arrival_rate", {}).get(
          "mean", float("nan")
        ),
        "any_brush_rate": aggregate.get("any_brush_rate", {}).get("mean", float("nan")),
        "contact_brush_rate": aggregate.get("contact_brush_rate", {}).get(
          "mean", float("nan")
        ),
        "hard_collision_rate": aggregate.get("hard_collision_rate", {}).get(
          "mean", float("nan")
        ),
        "collision_rate": aggregate.get("collision_rate", {}).get("mean", float("nan")),
        "pp_nopp_gap": float(t1_success.get(task_label, float("nan")) - success),
      }
    )
  return out


def _write_main_table(run_dir: Path, suffix: str, summary: dict) -> None:
  by_key = {
    (entry["task_label"], entry["method"]): entry for entry in summary["conditions"]
  }
  pct = "\\%"
  tex = [
    "\\begin{tabular}{l l c c c c c c c c c}",
    "\\toprule",
    "Task & Method & Success & Obj. on-time & $L_{obj}$ & $T_{obj}$ & $T_{ep}$ & Any brush & Obj. slow-brush & Timeout & Hard coll. \\\\",
    "\\midrule",
  ]
  md = [
    "# Paper Benchmark Summary (v2)",
    "",
    "| Task | Method | Success | Obj. on-time | L_obj (m) | T_obj (s) | T_ep (s) | Any brush | Obj. slow-brush | Timeout | Hard coll. |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for task_idx, task_label in enumerate(summary["task_labels"]):
    for method_idx, method in enumerate(DISPLAY_METHOD_ORDER):
      entry = by_key.get((task_label, method), {})
      metrics = entry.get("aggregate", {})
      task_cell = task_label if method_idx == 0 else ""
      tex.append(
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
      md.append(
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
    if task_idx < len(summary["task_labels"]) - 1:
      tex.append("\\midrule")
  tex.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / f"main_table{suffix}.tex").write_text("\n".join(tex), encoding="utf-8")

  md.extend(
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
  for row in summary.get("pp_nopp_gap", []):
    md.append(
      f"| {row['task_label']} | {_format_gap(float(row['t1_dr_gap']))} | "
      f"{_format_gap(float(row['t1_odom_gap']))} | {_format_gap(float(row['t1_r1_gap']))} |"
    )
    gap_tex.append(
      f"{row['task_label']} & {_format_gap_tex(float(row['t1_dr_gap']))} & "
      f"{_format_gap_tex(float(row['t1_odom_gap']))} & {_format_gap_tex(float(row['t1_r1_gap']))} \\\\"
    )
  gap_tex.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / f"pp_nopp_gap_table{suffix}.tex").write_text(
    "\n".join(gap_tex), encoding="utf-8"
  )
  (run_dir / f"summary{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _write_safety_table(run_dir: Path, suffix: str, summary: dict) -> None:
  by_key = {
    (entry["task_label"], entry["method"]): entry for entry in summary["conditions"]
  }
  pct = "\\%"
  tex = [
    "\\begin{tabular}{l l c c c c c c}",
    "\\toprule",
    "Task & Method & Any brush & Envelope brush & Contact brush & T-rel. slow-brush & T-rel. late-clean & Hard coll. \\\\",
    "\\midrule",
  ]
  md = [
    "",
    "## Safety / Brush Diagnostics",
    "",
    "| Task | Method | Any brush | Envelope brush | Contact brush | T-rel. slow-brush | T-rel. late-clean | Hard coll. |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
  ]
  for task_idx, task_label in enumerate(summary["task_labels"]):
    for method_idx, method in enumerate(METHOD_ORDER):
      entry = by_key.get((task_label, method), {})
      metrics = entry.get("aggregate", {})
      task_cell = task_label if method_idx == 0 else ""
      tex.append(
        f"{task_cell} & {method} & "
        f"{_format_mean_std(metrics.get('any_brush_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('envelope_brush_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('contact_brush_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('slow_brush_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('late_clean_arrival_rate'), pct)} & "
        f"{_format_mean_std(metrics.get('hard_collision_rate'), pct)} \\\\"
      )
      md.append(
        f"| {task_label} | {method} | "
        f"{_format_plain_pct(metrics.get('any_brush_rate'))} | "
        f"{_format_plain_pct(metrics.get('envelope_brush_rate'))} | "
        f"{_format_plain_pct(metrics.get('contact_brush_rate'))} | "
        f"{_format_plain_pct(metrics.get('slow_brush_arrival_rate'))} | "
        f"{_format_plain_pct(metrics.get('late_clean_arrival_rate'))} | "
        f"{_format_plain_pct(metrics.get('hard_collision_rate'))} |"
      )
    if task_idx < len(summary["task_labels"]) - 1:
      tex.append("\\midrule")
  tex.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / f"safety_table{suffix}.tex").write_text("\n".join(tex), encoding="utf-8")
  with (run_dir / f"summary{suffix}.md").open("a", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")


def _write_ablation_table(run_dir: Path, suffix: str, rows: list[dict]) -> None:
  lines = [
    "\\begin{tabular}{l l c c c c c c c c}",
    "\\toprule",
    "Mode & Variant & Success & Obj. on-time & Obj. late & Obj. slow-brush & Any brush & Contact brush & Hard coll. & PP--NoPP gap \\\\",
    "\\midrule",
  ]
  for row in rows:
    mode = "Odom" if row.get("use_odometry") else "DR"
    variant_cell = f"{row['variant']} ({row['task_label']})"
    lines.append(
      f"{mode} & {variant_cell} & "
      f"{row['success_rate']:.1f}\\% & {row['objective_on_time_arrival_rate']:.1f}\\% & "
      f"{row['objective_late_arrival_rate']:.1f}\\% & {row['objective_slow_brush_arrival_rate']:.1f}\\% & "
      f"{row['any_brush_rate']:.1f}\\% & {row['contact_brush_rate']:.1f}\\% & "
      f"{row['hard_collision_rate']:.1f}\\% & {row['pp_nopp_gap']:.1f}\\% \\\\"
    )
  if not rows:
    lines.append("\\multicolumn{10}{c}{No ablation rows found.} \\\\")
  lines.extend(["\\bottomrule", "\\end{tabular}", ""])
  (run_dir / f"ablation_table{suffix}.tex").write_text(
    "\n".join(lines), encoding="utf-8"
  )


def main() -> None:
  args = tyro.cli(RestatArgs, config=mjlab.TYRO_FLAGS)
  run_dir = Path(args.run_dir)
  episodes_path = run_dir / "episodes.jsonl"
  rows = _load_rows(episodes_path)
  groups = _build_record_groups(rows)
  task_labels = _task_labels(rows)
  seeds = sorted({int(row["seed"]) for row in rows})
  direct_budget_map, arrival_budget_map, envelope_threshold_map = _calibrate(
    groups, task_labels, seeds, args
  )
  conditions = _summarize_main(
    groups,
    rows,
    task_labels,
    seeds,
    direct_budget_map,
    arrival_budget_map,
    envelope_threshold_map,
    args.objective_timeout_margin_s,
  )
  summary = {
    "source_run_dir": str(run_dir),
    "episode_count": len(rows),
    "task_labels": task_labels,
    "args": sanitize_for_json(args.__dict__),
    "conditions": conditions,
    "direct_budget_calibration": {
      f"{task_label}/seed_{seed}": data
      for (task_label, seed), data in direct_budget_map.items()
    },
    "arrival_budget_calibration": {
      f"{task_label}/seed_{seed}": data
      for (task_label, seed), data in arrival_budget_map.items()
    },
    "envelope_brush_calibration": envelope_threshold_map,
  }
  summary["pp_nopp_gap"] = _compute_gap_rows(summary)
  ablation_rows = _ablation_rows(
    groups,
    rows,
    seeds,
    direct_budget_map,
    arrival_budget_map,
    envelope_threshold_map,
    conditions,
    args.objective_timeout_margin_s,
  )
  ablation_summary = {"rows": ablation_rows}

  suffix = args.output_suffix
  write_json(run_dir / f"summary{suffix}.json", summary)
  write_json(run_dir / f"ablation_summary{suffix}.json", ablation_summary)
  _write_main_table(run_dir, suffix, summary)
  _write_safety_table(run_dir, suffix, summary)
  _write_ablation_table(run_dir, suffix, ablation_rows)

  print(f"Restated {len(rows)} episodes from: {episodes_path}")
  print(f"  summary: {run_dir / f'summary{suffix}.json'}")
  print(f"  main:    {run_dir / f'main_table{suffix}.tex'}")
  print(f"  safety:  {run_dir / f'safety_table{suffix}.tex'}")
  print(f"  ablation:{run_dir / f'ablation_table{suffix}.tex'}")


if __name__ == "__main__":
  main()
