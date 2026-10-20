"""Compatibility helpers for loading legacy steering checkpoints.

The canonical steering modules now live under ``local_planner_*`` names.
Older checkpoints may still pickle dataclass configs from the historical
``ren_*`` modules, so loading must temporarily expose those module names
during unpickling.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import torch


def _build_legacy_modules() -> dict[str, types.ModuleType]:
  network_mod = importlib.import_module("mjlab.tasks.steering.local_planner_network")
  trainer_mod = importlib.import_module("mjlab.tasks.steering.local_planner_trainer")
  inference_mod = importlib.import_module(
    "mjlab.tasks.steering.local_planner_inference"
  )

  legacy_network = types.ModuleType("mjlab.tasks.steering.ren_network")
  legacy_network.RENNetworkCfg = network_mod.LocalPlannerNetworkCfg
  legacy_network.RENLayer = network_mod.EquilibriumPlannerLayer
  legacy_network.RENNetwork = network_mod.EquilibriumLocalPlannerNetwork
  legacy_network.SimplifiedRENNetwork = network_mod.LocalPlannerNetwork
  legacy_network.RENSteeringCorrector = network_mod.LocalPlannerSteeringCorrector

  legacy_trainer = types.ModuleType("mjlab.tasks.steering.ren_trainer")
  legacy_trainer.G1_REN_JOINT_VEL_INDICES = (
    trainer_mod.G1_LOCAL_PLANNER_JOINT_VEL_INDICES
  )
  legacy_trainer.RENTrainerCfg = trainer_mod.LocalPlannerTrainerCfg
  legacy_trainer.RENObservationBuilder = trainer_mod.LocalPlannerObservationBuilder
  legacy_trainer.RENTrainer = trainer_mod.LocalPlannerTrainer

  legacy_inference = types.ModuleType("mjlab.tasks.steering.ren_inference")
  legacy_inference.RENInferenceCfg = inference_mod.LocalPlannerInferenceCfg
  legacy_inference.RENMetrics = inference_mod.LocalPlannerMetrics
  legacy_inference.RENInference = inference_mod.LocalPlannerInference
  legacy_inference.RENCommandModifier = inference_mod.LocalPlannerCommandModifier

  return {
    legacy_network.__name__: legacy_network,
    legacy_trainer.__name__: legacy_trainer,
    legacy_inference.__name__: legacy_inference,
  }


@contextlib.contextmanager
def legacy_checkpoint_modules() -> Iterator[None]:
  legacy_modules = _build_legacy_modules()
  old_modules = {name: sys.modules.get(name) for name in legacy_modules}

  try:
    sys.modules.update(legacy_modules)
    yield
  finally:
    for name, old_module in old_modules.items():
      if old_module is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = old_module


def load_local_planner_checkpoint(
  checkpoint_path: str | Path,
  *,
  map_location: Any,
) -> Any:
  with legacy_checkpoint_modules():
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
