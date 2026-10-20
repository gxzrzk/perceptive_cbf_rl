#!/usr/bin/env python3
"""Export a local planner checkpoint (.pt) to ONNX for deployment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab.tasks.steering.local_planner_checkpoint_compat import (
  load_local_planner_checkpoint,
)
from mjlab.tasks.steering.local_planner_network import (
  LocalPlannerNetwork,
  LocalPlannerNetworkCfg,
)


class _LocalPlannerExportWrapper(torch.nn.Module):
  """Wrap the local planner with a stable 2-input/2-output ONNX signature."""

  def __init__(self, network: LocalPlannerNetwork):
    super().__init__()
    self.network = network

  def forward(
    self,
    obs: torch.Tensor,
    hidden_state: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    out, new_hidden = self.network(obs, hidden_state, return_hidden=True)
    assert isinstance(out, torch.Tensor)
    assert isinstance(new_hidden, torch.Tensor)
    return out, new_hidden


def _cfg_from_obj(cfg_obj: Any) -> LocalPlannerNetworkCfg:
  if isinstance(cfg_obj, LocalPlannerNetworkCfg):
    return cfg_obj

  if hasattr(cfg_obj, "network_cfg"):
    net_cfg = cfg_obj.network_cfg
    if isinstance(net_cfg, LocalPlannerNetworkCfg):
      return net_cfg
    if is_dataclass(net_cfg):
      return LocalPlannerNetworkCfg(**asdict(net_cfg))

  if is_dataclass(cfg_obj):
    return LocalPlannerNetworkCfg(**asdict(cfg_obj))

  if isinstance(cfg_obj, dict):
    if "network_cfg" in cfg_obj:
      raw = cfg_obj["network_cfg"]
      if isinstance(raw, LocalPlannerNetworkCfg):
        return raw
      if is_dataclass(raw):
        return LocalPlannerNetworkCfg(**asdict(raw))
      if isinstance(raw, dict):
        return LocalPlannerNetworkCfg(**raw)
    return LocalPlannerNetworkCfg(**cfg_obj)

  return LocalPlannerNetworkCfg()


def _infer_num_raycast_groups(
  state_dict: dict[str, torch.Tensor], fallback: int
) -> int:
  max_group = -1
  for key in state_dict:
    if key.startswith("raycast_cnns."):
      parts = key.split(".")
      if len(parts) >= 2 and parts[1].isdigit():
        max_group = max(max_group, int(parts[1]))
  if max_group >= 0:
    return max_group + 1
  return fallback


def _build_network_from_checkpoint(
  checkpoint_path: Path,
  num_raycast_groups: int,
  device: torch.device,
) -> tuple[LocalPlannerNetwork, LocalPlannerNetworkCfg]:
  ckpt = load_local_planner_checkpoint(checkpoint_path, map_location=device)

  if isinstance(ckpt, dict) and "network_state_dict" in ckpt:
    state_dict = ckpt["network_state_dict"]
    cfg_obj = ckpt.get("config", None)
  elif isinstance(ckpt, dict):
    state_dict = ckpt
    cfg_obj = None
  else:
    raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

  cfg = _cfg_from_obj(cfg_obj)
  cfg.num_raycast_groups = _infer_num_raycast_groups(state_dict, num_raycast_groups)

  network = LocalPlannerNetwork(cfg)
  missing, unexpected = network.load_state_dict(state_dict, strict=False)
  if missing:
    print(f"[WARN] missing keys: {missing}")
  if unexpected:
    print(f"[WARN] unexpected keys: {unexpected}")

  network.to(device)
  network.eval()
  return network, cfg


def parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(
    description="Export local planner .pt checkpoint to ONNX"
  )
  ap.add_argument(
    "--checkpoint", required=True, help="Local planner checkpoint .pt path"
  )
  ap.add_argument("--output", required=True, help="Output ONNX path")
  ap.add_argument(
    "--num-raycast-groups",
    type=int,
    default=3,
    help="Fallback number of raycast groups when checkpoint has no config",
  )
  ap.add_argument(
    "--opset",
    type=int,
    default=18,
    help="ONNX opset version",
  )
  ap.add_argument(
    "--dynamic-batch",
    action="store_true",
    help="Export with dynamic batch axis",
  )
  return ap.parse_args()


def main() -> int:
  args = parse_args()
  ckpt_path = Path(args.checkpoint)
  out_path = Path(args.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  device = torch.device("cpu")
  net, cfg = _build_network_from_checkpoint(
    ckpt_path,
    num_raycast_groups=args.num_raycast_groups,
    device=device,
  )

  input_dim = (
    cfg.base_heading_dim
    + cfg.joint_vel_dim
    + cfg.odom_vel_dim
    + cfg.raycast_dim * cfg.num_raycast_groups
    + cfg.goal_direction_dim
  )
  state_dim = cfg.state_dim

  wrapper = _LocalPlannerExportWrapper(net)
  obs = torch.zeros(1, input_dim, dtype=torch.float32, device=device)
  hidden = torch.zeros(1, state_dim, dtype=torch.float32, device=device)

  dynamic_axes = None
  if args.dynamic_batch:
    dynamic_axes = {
      "obs": {0: "batch"},
      "hidden_state": {0: "batch"},
      "output": {0: "batch"},
      "next_hidden_state": {0: "batch"},
    }

  torch.onnx.export(
    wrapper,
    (obs, hidden),
    out_path,
    export_params=True,
    opset_version=args.opset,
    do_constant_folding=True,
    input_names=["obs", "hidden_state"],
    output_names=["output", "next_hidden_state"],
    dynamic_axes=dynamic_axes,
  )

  print("Exported local planner ONNX:")
  print(f"  checkpoint: {ckpt_path}")
  print(f"  onnx:       {out_path}")
  print(f"  input_dim:  {input_dim}")
  print(f"  output_dim: {cfg.output_dim}")
  print(f"  state_dim:  {state_dim}")
  print(f"  groups:     {cfg.num_raycast_groups}")
  print(f"  odom_vel:   {cfg.odom_vel_dim}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
