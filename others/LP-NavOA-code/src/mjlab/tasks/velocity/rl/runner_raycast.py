"""Velocity task runner with raycast temporal encoding support.

This module provides a specialized runner that uses ActorCriticRaycast policy
for processing raycast observations through a temporal encoder.
"""

import builtins
import os

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.actor_critic_raycast import ActorCriticRaycast

# Make ActorCriticRaycast available to rsl_rl's eval()
# This is needed because rsl_rl uses eval(policy_class_name) to instantiate policies
builtins.ActorCriticRaycast = ActorCriticRaycast  # type: ignore[attr-defined]


class VelocityRaycastOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner for velocity tasks with raycast temporal encoding.

  Uses ActorCriticRaycast policy that processes multi-frame raycast
  observations through a temporal encoder (per-frame MLP + 1D conv + MLP fusion).
  """

  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
    try:
      self.export_policy_to_onnx(policy_path, filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      onnx_path = os.path.join(policy_path, filename)
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(onnx_path, metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
