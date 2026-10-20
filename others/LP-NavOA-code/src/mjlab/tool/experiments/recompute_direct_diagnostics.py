#!/usr/bin/env python3
"""Recompute direct/wandering diagnostics from an existing paper run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

import mjlab
from mjlab.tool.experiments.eval_utils import (
  DEFAULT_DIRECT_CALIBRATION_QUANTILE,
  DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
  EvalEpisodeRecord,
  apply_direct_budget,
  calibrate_direct_budgets,
  compute_summary,
  direct_quantile_key,
  write_json,
)


@dataclass
class RecomputeDirectDiagnosticsArgs:
  run_dir: str
  """Existing paper benchmark run directory."""

  output_name: str = "direct_diagnostics_recomputed"
  """Output basename without extension."""

  primary_quantile: float = DEFAULT_DIRECT_CALIBRATION_QUANTILE
  """Primary T1 quantile used to annotate direct/wandering success."""

  min_direct_calibration_episodes: int = 5
  """Minimum T1 successful episodes required before applying direct budgets."""


def _record_from_dict(data: dict) -> EvalEpisodeRecord:
  allowed = set(EvalEpisodeRecord.__dataclass_fields__.keys())
  filtered = {key: value for key, value in data.items() if key in allowed}
  record = EvalEpisodeRecord(**filtered)
  if not record.episode_time_s or record.episode_time_s != record.episode_time_s:
    record.episode_time_s = float(record.episode_steps) * 0.02
  if (
    not record.target_velocity_cmd
    or record.target_velocity_cmd != record.target_velocity_cmd
  ):
    actual_velocity = float(record.actual_velocity)
    record.target_velocity_cmd = (
      abs(actual_velocity) if actual_velocity == actual_velocity else 1.0
    )
  return record


def _load_records(path: Path) -> dict[tuple[str, str, int], list[EvalEpisodeRecord]]:
  by_key: dict[tuple[str, str, int], list[EvalEpisodeRecord]] = {}
  with path.open(encoding="utf-8") as f:
    for line in f:
      row = json.loads(line)
      key = (str(row["task_label"]), str(row["method"]), int(row["seed"]))
      by_key.setdefault(key, []).append(_record_from_dict(row["record"]))
  return by_key


def main() -> None:
  args = tyro.cli(RecomputeDirectDiagnosticsArgs, config=mjlab.TYRO_FLAGS)
  run_dir = Path(args.run_dir)
  episodes_path = run_dir / "episodes.jsonl"
  if not episodes_path.is_file():
    raise SystemExit(f"Missing episodes file: {episodes_path}")

  by_key = _load_records(episodes_path)
  budget_map: dict[tuple[str, int], dict[str, float]] = {}
  for (task_label, method, seed), records in by_key.items():
    if method == "T1 (PP upper bound)":
      budget_map[(task_label, seed)] = calibrate_direct_budgets(
        records,
        quantiles=DEFAULT_DIRECT_SENSITIVITY_QUANTILES,
        min_success_episodes=args.min_direct_calibration_episodes,
      )

  rows = []
  for (task_label, method, seed), records in sorted(by_key.items()):
    budgets = budget_map.get((task_label, seed), {})
    primary_budget = budgets.get(direct_quantile_key(args.primary_quantile))
    apply_direct_budget(records, budget_effective_distance=primary_budget)
    summary = compute_summary(records, direct_budgets=budgets)
    rows.append(
      {
        "task_label": task_label,
        "method": method,
        "seed": seed,
        "summary": summary,
      }
    )

  output = {
    "run_dir": str(run_dir),
    "primary_quantile": args.primary_quantile,
    "direct_budget_calibration": {
      f"{task_label}/seed_{seed}": budgets
      for (task_label, seed), budgets in sorted(budget_map.items())
    },
    "rows": rows,
  }
  write_json(run_dir / f"{args.output_name}.json", output)

  md = [
    "# Recomputed Direct Diagnostics",
    "",
    "| Task | Method | Seed | Success | Direct | Wandering | q95 budget eff. m |",
    "|---|---|---:|---:|---:|---:|---:|",
  ]
  q95 = direct_quantile_key(0.95)
  for row in rows:
    summary = row["summary"]
    budgets = budget_map.get((row["task_label"], row["seed"]), {})
    md.append(
      f"| {row['task_label']} | {row['method']} | {row['seed']} | "
      f"{summary.get('success_rate', 0.0):.1f}% | "
      f"{summary.get('direct_success_rate', 0.0):.1f}% | "
      f"{summary.get('wandering_success_rate', 0.0):.1f}% | "
      f"{budgets.get(q95, float('nan')):.2f} |"
    )
  (run_dir / f"{args.output_name}.md").write_text(
    "\n".join(md) + "\n", encoding="utf-8"
  )
  print(f"Wrote {run_dir / f'{args.output_name}.json'}")
  print(f"Wrote {run_dir / f'{args.output_name}.md'}")


if __name__ == "__main__":
  main()
