#!/usr/bin/env python3
"""Validate local planner ONNX against the PyTorch checkpoint numerically."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mjlab.tasks.steering.export_local_planner_to_onnx import (
  _build_network_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Validate local planner PT vs ONNX")
  ap.add_argument(
    "--checkpoint", required=True, help="Local planner checkpoint .pt path"
  )
  ap.add_argument("--onnx", required=True, help="Local planner ONNX path")
  ap.add_argument("--num-raycast-groups", type=int, default=3)
  ap.add_argument("--num-tests", type=int, default=200)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--tol-max", type=float, default=5e-4)
  ap.add_argument("--tol-mean", type=float, default=1e-4)
  return ap.parse_args()


def main() -> int:
  args = parse_args()
  rng = np.random.default_rng(args.seed)

  device = torch.device("cpu")
  net, cfg = _build_network_from_checkpoint(
    Path(args.checkpoint),
    num_raycast_groups=args.num_raycast_groups,
    device=device,
  )
  net.eval()

  input_dim = (
    cfg.base_heading_dim
    + cfg.joint_vel_dim
    + cfg.odom_vel_dim
    + cfg.raycast_dim * cfg.num_raycast_groups
    + cfg.goal_direction_dim
  )
  state_dim = cfg.state_dim

  sess = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])

  output_abs_max: list[float] = []
  output_abs_mean: list[float] = []
  hidden_abs_max: list[float] = []
  hidden_abs_mean: list[float] = []

  for _ in range(args.num_tests):
    obs = rng.standard_normal((1, input_dim), dtype=np.float32)
    hidden = rng.standard_normal((1, state_dim), dtype=np.float32)

    with torch.no_grad():
      pt_out, pt_hidden = net(
        torch.from_numpy(obs),
        torch.from_numpy(hidden),
        return_hidden=True,
      )
    pt_out_np = pt_out.cpu().numpy()
    pt_hidden_np = pt_hidden.cpu().numpy()

    onnx_out, onnx_hidden = sess.run(
      ["output", "next_hidden_state"],
      {"obs": obs, "hidden_state": hidden},
    )

    out_abs = np.abs(pt_out_np - onnx_out)
    hid_abs = np.abs(pt_hidden_np - onnx_hidden)
    output_abs_max.append(float(out_abs.max()))
    output_abs_mean.append(float(out_abs.mean()))
    hidden_abs_max.append(float(hid_abs.max()))
    hidden_abs_mean.append(float(hid_abs.mean()))

  out_max = max(output_abs_max)
  out_mean = float(np.mean(output_abs_mean))
  hid_max = max(hidden_abs_max)
  hid_mean = float(np.mean(hidden_abs_mean))

  print("PT vs ONNX validation result:")
  print(f"  tests:            {args.num_tests}")
  print(f"  output max abs:   {out_max:.6e}")
  print(f"  output mean abs:  {out_mean:.6e}")
  print(f"  hidden max abs:   {hid_max:.6e}")
  print(f"  hidden mean abs:  {hid_mean:.6e}")

  ok = (
    out_max <= args.tol_max
    and out_mean <= args.tol_mean
    and hid_max <= args.tol_max
    and hid_mean <= args.tol_mean
  )

  if ok:
    print("PASS")
    return 0

  print("FAIL")
  return 2


if __name__ == "__main__":
  raise SystemExit(main())
