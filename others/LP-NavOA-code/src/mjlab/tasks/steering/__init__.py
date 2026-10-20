"""Steering local planner package.

The canonical deployment architecture is a GRU-based local planner that learns
steering corrections from the planner/waypoint teacher using only local
observations. A legacy equilibrium-network variant is retained for research use.

Usage:
  # Training
  python -m mjlab.tasks.steering.train_local_planner --mode train \\
    --task Mjlab-StraightApproach-LowSpeed-G1-v6 \\
    --checkpoint /path/to/policy.pt

  # Testing with steering correction
  python -m mjlab.tasks.steering.train_local_planner --mode test \\
    --task Mjlab-StraightApproach-LowSpeed-G1-v6 \\
    --local_planner_checkpoint /path/to/local_planner.pt
"""

from mjlab.tasks.steering.local_planner_inference import (
  LocalPlannerCommandModifier as LocalPlannerCommandModifier,
)
from mjlab.tasks.steering.local_planner_inference import (
  LocalPlannerInference as LocalPlannerInference,
)
from mjlab.tasks.steering.local_planner_inference import (
  LocalPlannerInferenceCfg as LocalPlannerInferenceCfg,
)
from mjlab.tasks.steering.local_planner_network import (
  EquilibriumLocalPlannerNetwork as EquilibriumLocalPlannerNetwork,
)
from mjlab.tasks.steering.local_planner_network import (
  LocalPlannerNetwork as LocalPlannerNetwork,
)
from mjlab.tasks.steering.local_planner_network import (
  LocalPlannerNetworkCfg as LocalPlannerNetworkCfg,
)
from mjlab.tasks.steering.local_planner_trainer import (
  LocalPlannerObservationBuilder as LocalPlannerObservationBuilder,
)
from mjlab.tasks.steering.local_planner_trainer import (
  LocalPlannerTrainer as LocalPlannerTrainer,
)
from mjlab.tasks.steering.local_planner_trainer import (
  LocalPlannerTrainerCfg as LocalPlannerTrainerCfg,
)

__all__ = [
  # Network
  "LocalPlannerNetworkCfg",
  "LocalPlannerNetwork",
  "EquilibriumLocalPlannerNetwork",
  # Trainer
  "LocalPlannerTrainerCfg",
  "LocalPlannerTrainer",
  "LocalPlannerObservationBuilder",
  # Inference
  "LocalPlannerInferenceCfg",
  "LocalPlannerInference",
  "LocalPlannerCommandModifier",
]
