#!/usr/bin/env python3
"""Plot paper-ready benchmark summaries.

Example:
  uv run python -m mjlab.tool.experiments.plot_paper_results \\
    --run-dir logs/benchmark/paper/formal_v8_1_2026-05-06_01-53-48
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tyro

import mjlab

METHOD_ORDER = (
  "R1 (Backbone-only)",
  "T1 (PP upper bound)",
  "Ours (DR)",
  "Ours (Odom)",
)
METHOD_SHORT = {
  "R1 (Backbone-only)": "R1",
  "T1 (PP upper bound)": "T1",
  "Ours (DR)": "Ours DR",
  "Ours (Odom)": "Ours Odom",
}
METHOD_COLORS = {
  "R1 (Backbone-only)": "#6f6f6f",
  "T1 (PP upper bound)": "#009E73",
  "Ours (DR)": "#0072B2",
  "Ours (Odom)": "#7B5EA7",
}
METRIC_LABELS = {
  "success_rate": "Success",
  "teacher_relative_on_time_arrival_rate": "T-rel. on-time arrival",
  "mean_episode_overhead_s": "Mean overhead vs $T_0$",
  "teacher_relative_slow_brush_arrival_rate": "T-rel. slow-brush arrival",
  "any_brush_rate": "Any brush",
  "contact_brush_rate": "Contact brush",
  "hard_collision_rate": "Hard collision",
}
METRIC_YLABELS = {
  "mean_episode_overhead_s": "Time over best path (s)",
}
RATE_METRICS = {
  "success_rate",
  "teacher_relative_on_time_arrival_rate",
  "teacher_relative_slow_brush_arrival_rate",
  "any_brush_rate",
  "contact_brush_rate",
  "hard_collision_rate",
}
MAIN_METRICS = (
  "success_rate",
  "teacher_relative_on_time_arrival_rate",
  "mean_episode_overhead_s",
  "teacher_relative_slow_brush_arrival_rate",
  "any_brush_rate",
  "contact_brush_rate",
  "hard_collision_rate",
)
ABLATION_METRICS = (
  "teacher_relative_on_time_arrival_rate",
  "teacher_relative_slow_brush_arrival_rate",
  "any_brush_rate",
  "hard_collision_rate",
)
DIAGNOSTIC_BAR_METRICS = (
  "any_brush_rate",
  "contact_brush_rate",
  "hard_collision_rate",
)


@dataclass
class PlotArgs:
  """Plotting options for paper benchmark outputs."""

  run_dir: str
  """Directory created by run_paper_experiments.py."""

  training_curves_csv: str = ""
  """Optional CSV export with local-planner training curves."""

  output_dir: str = ""
  """Optional figure output directory. Defaults to <run_dir>/figures."""


@dataclass(frozen=True)
class MetricValue:
  """Mean and standard deviation for a paper metric."""

  mean: float
  std: float = 0.0


@dataclass(frozen=True)
class ConditionMetrics:
  """Aggregated metrics for one task-method condition."""

  task_label: str
  method: str
  values: dict[str, MetricValue]


def _load_json(path: Path) -> dict[str, Any]:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  if not path.exists():
    return rows
  with open(path, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
      if limit is not None and len(rows) >= limit:
        break
  return rows


def _setup_matplotlib() -> None:
  import matplotlib as mpl

  mpl.rcParams.update(
    {
      "pdf.fonttype": 42,
      "ps.fonttype": 42,
      "font.family": "DejaVu Sans",
      "font.size": 8.5,
      "axes.titlesize": 9.5,
      "axes.labelsize": 8.5,
      "axes.linewidth": 0.8,
      "xtick.labelsize": 7.5,
      "ytick.labelsize": 7.5,
      "legend.fontsize": 7.5,
      "figure.titlesize": 10.5,
      "savefig.bbox": "tight",
      "savefig.pad_inches": 0.03,
    }
  )


def _save_figure(fig: Any, fig_dir: Path, stem: str) -> None:
  fig_dir.mkdir(parents=True, exist_ok=True)
  fig.savefig(fig_dir / f"{stem}.pdf")
  fig.savefig(fig_dir / f"{stem}.png", dpi=300)


def _remove_legacy_figure_aliases(fig_dir: Path) -> None:
  """Drop aliases from older plot scripts that duplicated canonical figures."""

  legacy_aliases = {
    "trajectories": "trajectory_diagnostics",
    "pp_nopp_gap": "main_metrics",
    "ablation": "ablation_diagnostics",
  }
  for alias, canonical in legacy_aliases.items():
    for suffix in (".pdf", ".png"):
      alias_path = fig_dir / f"{alias}{suffix}"
      canonical_path = fig_dir / f"{canonical}{suffix}"
      if alias_path.exists() and canonical_path.exists():
        alias_path.unlink()


def _style_axis(ax: Any, *, y_grid: bool = True) -> None:
  if y_grid:
    ax.grid(axis="y", color="#d7dde2", linewidth=0.6, alpha=0.8)
  ax.set_axisbelow(True)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.spines["left"].set_color("#7b8790")
  ax.spines["bottom"].set_color("#7b8790")
  ax.tick_params(colors="#263238", width=0.7, length=3)


def _extract_metric(metric_data: Any) -> MetricValue:
  if isinstance(metric_data, dict):
    return MetricValue(
      mean=float(metric_data.get("mean", float("nan"))),
      std=float(metric_data.get("std", float("nan"))),
    )
  if isinstance(metric_data, (float, int)):
    return MetricValue(mean=float(metric_data), std=0.0)
  return MetricValue(mean=float("nan"), std=float("nan"))


def _extract_metric_for_plot(aggregate: dict[str, Any], metric: str) -> MetricValue:
  if metric != "mean_episode_overhead_s":
    return _extract_metric(aggregate.get(metric, {}))

  episode_time = _extract_metric(aggregate.get("mean_episode_time_s", {}))
  path_length = _extract_metric(aggregate.get("mean_objective_path_length_m", {}))
  target_velocity = _extract_metric(aggregate.get("mean_target_velocity_cmd", {}))
  if (
    math.isfinite(episode_time.mean)
    and math.isfinite(path_length.mean)
    and math.isfinite(target_velocity.mean)
    and target_velocity.mean > 1e-6
  ):
    best_path_time = path_length.mean / target_velocity.mean
    return MetricValue(mean=episode_time.mean - best_path_time, std=episode_time.std)
  return MetricValue(mean=float("nan"), std=float("nan"))


def _task_labels(summary: dict[str, Any]) -> list[str]:
  labels = [str(label) for label in summary.get("task_labels", [])]
  if labels:
    return labels

  out: list[str] = []
  for condition in summary.get("conditions", []):
    label = str(condition.get("task_label", ""))
    if label and label not in out:
      out.append(label)
  return out


def _episode_overhead_metrics(
  rows: list[dict[str, Any]],
) -> dict[tuple[str, str], MetricValue]:
  import numpy as np

  by_seed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
  for row in rows:
    task = str(row.get("task_label", ""))
    method = str(row.get("method", ""))
    if not task or not method:
      continue
    record = row.get("record", {})
    try:
      episode_time_s = float(record.get("episode_time_s"))
      path_length_m = float(record.get("objective_path_length_m"))
      target_velocity = float(record.get("target_velocity_cmd"))
    except (TypeError, ValueError):
      continue
    if not (
      math.isfinite(episode_time_s)
      and math.isfinite(path_length_m)
      and math.isfinite(target_velocity)
      and target_velocity > 1e-6
    ):
      continue
    seed = int(row.get("seed", 0))
    by_seed[(task, method, seed)].append(
      episode_time_s - path_length_m / target_velocity
    )

  seed_means: dict[tuple[str, str], list[float]] = defaultdict(list)
  for (task, method, _seed), values in by_seed.items():
    if values:
      seed_means[(task, method)].append(float(np.mean(values)))

  return {
    key: MetricValue(mean=float(np.mean(values)), std=float(np.std(values)))
    for key, values in seed_means.items()
    if values
  }


def _condition_metrics(
  summary: dict[str, Any],
  *,
  overhead_metrics: dict[tuple[str, str], MetricValue] | None = None,
) -> dict[tuple[str, str], ConditionMetrics]:
  conditions: dict[tuple[str, str], ConditionMetrics] = {}
  for condition in summary.get("conditions", []):
    task = str(condition.get("task_label", ""))
    method = str(condition.get("method", ""))
    if not task or not method:
      continue
    aggregate = condition.get("aggregate", {})
    values = {
      metric: _extract_metric_for_plot(aggregate, metric)
      for metric in set(MAIN_METRICS + ("gait_preservation_score", "action_rate_l2"))
    }
    if overhead_metrics is not None and (task, method) in overhead_metrics:
      values["mean_episode_overhead_s"] = overhead_metrics[(task, method)]
    conditions[(task, method)] = ConditionMetrics(
      task_label=task,
      method=method,
      values=values,
    )
  return conditions


def _metric(
  conditions: dict[tuple[str, str], ConditionMetrics],
  task: str,
  method: str,
  metric: str,
) -> MetricValue:
  entry = conditions.get((task, method))
  if entry is None:
    return MetricValue(0.0, 0.0)
  return entry.values.get(metric, MetricValue(0.0, 0.0))


def _trajectory_groups(
  rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[list[list[float]]]]:
  groups: dict[tuple[str, str], list[list[list[float]]]] = defaultdict(list)
  for row in rows:
    method = str(row.get("method", ""))
    if method not in METHOD_ORDER:
      continue
    record = row.get("record", {})
    traj = record.get("trajectory_xy") or []
    if len(traj) < 2:
      continue
    task = str(row.get("task_label", ""))
    groups[(task, method)].append(traj)
  return groups


def _warn_sparse_trajectories(
  groups: dict[tuple[str, str], list[list[list[float]]]],
  task_labels: list[str],
) -> None:
  for task in task_labels:
    for method in METHOD_ORDER:
      count = len(groups.get((task, method), []))
      if count < 20:
        print(
          "Warning: "
          f"{task} / {method} has only {count} non-empty saved trajectories. "
          "Plotting all available trajectories without synthesizing more."
        )


def _plot_trajectory_panel(
  ax: Any,
  groups: dict[tuple[str, str], list[list[list[float]]]],
  task: str,
  *,
  show_legend: bool,
) -> None:
  all_x: list[float] = []
  all_y: list[float] = []
  for method in METHOD_ORDER:
    trajectories = groups.get((task, method), [])
    for idx, traj in enumerate(trajectories):
      xs = [float(point[0]) for point in traj]
      ys = [float(point[1]) for point in traj]
      all_x.extend(xs)
      all_y.extend(ys)
      ax.plot(
        xs,
        ys,
        color=METHOD_COLORS[method],
        linewidth=1.15,
        alpha=0.42,
        label=METHOD_SHORT[method] if show_legend and idx == 0 else None,
      )
      ax.scatter(
        xs[0],
        ys[0],
        s=9,
        color=METHOD_COLORS[method],
        alpha=0.55,
        linewidths=0,
      )
      ax.scatter(
        xs[-1],
        ys[-1],
        s=12,
        marker="x",
        color=METHOD_COLORS[method],
        alpha=0.68,
        linewidths=0.8,
      )

  ax.set_title(f"{task} trajectories", loc="left", pad=4)
  ax.set_xlabel("World X (m)")
  ax.set_ylabel("World Y (m)")
  ax.set_aspect("equal", adjustable="box")
  if all_x and all_y:
    x_pad = max(1.0, (max(all_x) - min(all_x)) * 0.05)
    y_pad = max(1.0, (max(all_y) - min(all_y)) * 0.08)
    ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
    ax.set_ylim(min(all_y) - y_pad, max(all_y) + y_pad)
  _style_axis(ax)
  ax.grid(color="#d7dde2", linewidth=0.55, alpha=0.75)


def _plot_metric_bars(
  ax: Any,
  conditions: dict[tuple[str, str], ConditionMetrics],
  task_labels: list[str],
  metric: str,
  *,
  title: str,
  ylabel: str = "Rate (%)",
) -> None:
  import numpy as np

  x = np.arange(len(task_labels), dtype=float)
  width = 0.18
  offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))
  for idx, method in enumerate(METHOD_ORDER):
    values = [_metric(conditions, task, method, metric) for task in task_labels]
    means = [value.mean for value in values]
    stds = [0.0 if np.isnan(value.std) else value.std for value in values]
    ax.bar(
      x + offsets[idx],
      means,
      width,
      yerr=stds,
      capsize=2.0,
      color=METHOD_COLORS[method],
      edgecolor="white",
      linewidth=0.45,
      label=METHOD_SHORT[method],
    )
  ax.set_title(title, loc="left", pad=3)
  ax.set_xticks(x, task_labels)
  ax.set_ylabel(ylabel)
  finite_means = [
    _metric(conditions, t, m, metric).mean
    for t in task_labels
    for m in METHOD_ORDER
    if np.isfinite(_metric(conditions, t, m, metric).mean)
  ]
  upper = 100.0 if finite_means and max(finite_means) > 35 else None
  if upper is not None:
    ax.set_ylim(0, 105)
  else:
    ax.set_ylim(bottom=0)
  _style_axis(ax)


def _plot_safety_dots(
  ax: Any,
  conditions: dict[tuple[str, str], ConditionMetrics],
  task_labels: list[str],
) -> None:
  import numpy as np

  row_labels: list[str] = []
  y_positions: list[float] = []
  y = 0
  for task in task_labels:
    for method in METHOD_ORDER:
      row_labels.append(f"{task}  {METHOD_SHORT[method]}")
      y_positions.append(float(y))
      y += 1
    y += 0.55

  metric_markers = {
    "any_brush_rate": "o",
    "contact_brush_rate": "s",
    "hard_collision_rate": "^",
  }
  metric_colors = {
    "any_brush_rate": "#C44E52",
    "contact_brush_rate": "#E69F00",
    "hard_collision_rate": "#222222",
  }
  for metric in DIAGNOSTIC_BAR_METRICS:
    xs: list[float] = []
    ys: list[float] = []
    xerrs: list[float] = []
    for task in task_labels:
      for method in METHOD_ORDER:
        value = _metric(conditions, task, method, metric)
        xs.append(value.mean)
        xerrs.append(value.std)
    ys = y_positions
    ax.errorbar(
      xs,
      ys,
      xerr=xerrs,
      fmt=metric_markers[metric],
      markersize=4.0,
      linewidth=0.8,
      capsize=1.8,
      color=metric_colors[metric],
      label=METRIC_LABELS[metric],
      alpha=0.95,
    )

  ax.set_title("Safety diagnostics", loc="left", pad=3)
  ax.set_xlabel("Episode rate (%)")
  ax.set_yticks(y_positions, row_labels)
  ax.invert_yaxis()
  ax.set_xlim(left=0)
  _style_axis(ax)
  ax.legend(frameon=False, loc="lower right")
  for sep_idx in range(len(task_labels) - 1):
    sep_y = (sep_idx + 1) * len(METHOD_ORDER) + sep_idx * 0.55 - 0.25
    ax.axhline(sep_y, color="#b8c2cc", linewidth=0.6)
  ax.set_xticks(np.linspace(0, 80, 5))


def _plot_trajectory_diagnostics(
  summary: dict[str, Any],
  rows: list[dict[str, Any]],
  fig_dir: Path,
) -> None:
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D

  task_labels = _task_labels(summary)
  conditions = _condition_metrics(summary)
  trajectories = _trajectory_groups(rows)
  _warn_sparse_trajectories(trajectories, task_labels)

  fig = plt.figure(figsize=(7.15, 5.2), constrained_layout=True)
  gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1.35, 1.0],
    height_ratios=[1.0, 1.0, 1.15],
  )
  traj_axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
  ]
  for idx, task in enumerate(task_labels[:2]):
    _plot_trajectory_panel(
      traj_axes[idx],
      trajectories,
      task,
      show_legend=idx == 0,
    )
  if len(task_labels) < 2:
    traj_axes[1].set_visible(False)

  method_handles = [
    Line2D([0], [0], color=METHOD_COLORS[method], linewidth=1.6)
    for method in METHOD_ORDER
  ]
  fig.legend(
    method_handles,
    [METHOD_SHORT[method] for method in METHOD_ORDER],
    frameon=False,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.565, 1.035),
    handlelength=1.7,
    columnspacing=1.0,
  )

  ax_on_time = fig.add_subplot(gs[0, 1])
  ax_slow_brush = fig.add_subplot(gs[1, 1])
  ax_safety = fig.add_subplot(gs[2, :])
  _plot_metric_bars(
    ax_on_time,
    conditions,
    task_labels,
    "teacher_relative_on_time_arrival_rate",
    title="T-rel. on-time arrival",
  )
  _plot_metric_bars(
    ax_slow_brush,
    conditions,
    task_labels,
    "teacher_relative_slow_brush_arrival_rate",
    title="T-rel. late arrivals with brush",
  )
  _plot_safety_dots(ax_safety, conditions, task_labels)

  _save_figure(fig, fig_dir, "trajectory_diagnostics")
  plt.close(fig)


def _plot_main_metrics(
  summary: dict[str, Any],
  episode_rows: list[dict[str, Any]],
  fig_dir: Path,
) -> None:
  import matplotlib.pyplot as plt
  import numpy as np

  task_labels = _task_labels(summary)
  conditions = _condition_metrics(
    summary,
    overhead_metrics=_episode_overhead_metrics(episode_rows),
  )
  fig, axes = plt.subplots(2, 4, figsize=(7.15, 4.35), constrained_layout=True)
  axes_flat = list(axes.flat)
  x = np.arange(len(task_labels), dtype=float)
  width = 0.18
  offsets = np.linspace(-1.5 * width, 1.5 * width, len(METHOD_ORDER))

  for ax, metric in zip(axes_flat, MAIN_METRICS, strict=False):
    for idx, method in enumerate(METHOD_ORDER):
      values = [_metric(conditions, task, method, metric) for task in task_labels]
      ax.bar(
        x + offsets[idx],
        [value.mean for value in values],
        width,
        yerr=[0.0 if np.isnan(value.std) else value.std for value in values],
        capsize=1.8,
        color=METHOD_COLORS[method],
        edgecolor="white",
        linewidth=0.45,
        label=METHOD_SHORT[method],
      )
    ax.set_title(METRIC_LABELS[metric], loc="left", pad=3)
    ax.set_xticks(x, task_labels)
    if metric in RATE_METRICS:
      ax.set_ylim(0, 105)
    else:
      finite = [
        value.mean + (0.0 if np.isnan(value.std) else value.std)
        for task in task_labels
        for method in METHOD_ORDER
        if np.isfinite((value := _metric(conditions, task, method, metric)).mean)
      ]
      ax.set_ylim(0, max(finite) * 1.15 if finite else 1.0)
    ax.set_ylabel(METRIC_YLABELS.get(metric, "Rate (%)"))
    _style_axis(ax)

  for ax in axes_flat[len(MAIN_METRICS) :]:
    ax.set_visible(False)

  handles, labels = axes_flat[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.04),
  )
  _save_figure(fig, fig_dir, "main_metrics")
  plt.close(fig)


def _ablation_group(variant: str) -> str:
  if variant.startswith("Odom"):
    return "Odom"
  if variant.startswith("DR"):
    return "DR"
  return "Other"


def _ablation_label(variant: str) -> str:
  for prefix in ("DR ", "Odom "):
    if variant.startswith(prefix):
      variant = variant[len(prefix) :]
  return variant.replace(" labels", "").replace("safety filter", "safety")


def _plot_ablation_diagnostics(summary: dict[str, Any], fig_dir: Path) -> None:
  import matplotlib.pyplot as plt
  import numpy as np

  rows = [row for row in summary.get("rows", []) if row.get("status") == "ok"]
  if not rows:
    return

  groups = {
    "DR": [row for row in rows if _ablation_group(str(row.get("variant", ""))) == "DR"],
    "Odom": [
      row for row in rows if _ablation_group(str(row.get("variant", ""))) == "Odom"
    ],
  }
  fig, axes = plt.subplots(
    len(groups),
    1,
    figsize=(7.15, 4.75),
    sharey=True,
    constrained_layout=True,
  )
  if len(groups) == 1:
    axes = [axes]

  metric_colors = {
    "teacher_relative_on_time_arrival_rate": "#0072B2",
    "teacher_relative_slow_brush_arrival_rate": "#D55E00",
    "any_brush_rate": "#C44E52",
    "hard_collision_rate": "#222222",
  }
  for ax, (group, group_rows) in zip(axes, groups.items(), strict=True):
    labels = [_ablation_label(str(row.get("variant", ""))) for row in group_rows]
    x = np.arange(len(group_rows), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ABLATION_METRICS))
    for idx, metric in enumerate(ABLATION_METRICS):
      ax.bar(
        x + offsets[idx],
        [float(row.get(metric, 0.0)) for row in group_rows],
        width,
        color=metric_colors[metric],
        edgecolor="white",
        linewidth=0.45,
        label=METRIC_LABELS[metric],
      )
    ax.set_title(f"{group} ablations", loc="left", pad=3)
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Episode rate (%)")
    _style_axis(ax)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    ncol=4,
    bbox_to_anchor=(0.5, 1.04),
  )
  _save_figure(fig, fig_dir, "ablation_diagnostics")
  plt.close(fig)


def _plot_training_curves(csv_path: Path, fig_dir: Path) -> None:
  import matplotlib.pyplot as plt

  if not csv_path.exists():
    return
  with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
  if not rows:
    return
  step_key = "step" if "step" in rows[0] else next(iter(rows[0].keys()))
  metric_keys = [
    key for key in rows[0].keys() if key != step_key and key.lower() != "wall_time"
  ][:6]
  if not metric_keys:
    return
  fig, ax = plt.subplots(figsize=(7.15, 3.1), constrained_layout=True)
  steps = [float(row[step_key]) for row in rows]
  for key in metric_keys:
    vals: list[float] = []
    for row in rows:
      try:
        vals.append(float(row[key]))
      except ValueError:
        vals.append(float("nan"))
    ax.plot(steps, vals, label=key, linewidth=1.2)
  ax.set_xlabel(step_key)
  ax.set_title("Local planner training curves", loc="left", pad=3)
  _style_axis(ax)
  ax.legend(frameon=False, fontsize=7)
  _save_figure(fig, fig_dir, "training_curves")
  plt.close(fig)


def main() -> None:
  args = tyro.cli(PlotArgs, config=mjlab.TYRO_FLAGS)
  run_dir = Path(args.run_dir)
  fig_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
  fig_dir.mkdir(parents=True, exist_ok=True)

  try:
    import matplotlib  # noqa: F401
  except Exception as exc:
    raise SystemExit(
      "matplotlib is required for plotting. Install it or run only the benchmark."
    ) from exc

  _setup_matplotlib()
  _remove_legacy_figure_aliases(fig_dir)
  summary = _load_json(run_dir / "summary.json")
  episode_rows = _load_jsonl(run_dir / "episodes.jsonl")
  _plot_trajectory_diagnostics(summary, episode_rows, fig_dir)
  _plot_main_metrics(summary, episode_rows, fig_dir)

  ablation_path = run_dir / "ablation_summary.json"
  if ablation_path.exists():
    _plot_ablation_diagnostics(_load_json(ablation_path), fig_dir)
  if args.training_curves_csv:
    _plot_training_curves(Path(args.training_curves_csv), fig_dir)
  print(f"Figures written to: {fig_dir}")


if __name__ == "__main__":
  main()
