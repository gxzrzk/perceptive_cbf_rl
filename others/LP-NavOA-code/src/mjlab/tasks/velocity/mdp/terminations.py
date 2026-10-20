from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def all_goals_reached(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Terminate when all goals have been reached.

  This termination condition checks if the robot has completed all waypoints
  in the goal tracking task.

  Args:
    env: The environment instance.
    command_name: Name of the goal tracking command.

  Returns:
    Boolean tensor indicating which environments should terminate.
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  # Access the all_goals_completed flag from GoalTrackingCommand
  return command_term.all_goals_completed  # type: ignore[attr-defined]


def fixed_sequence_complete(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Terminate when the fixed goal sequence is complete.

  This termination condition checks if the robot has reached all goals
  in a FixedSequentialGoalCommand, or if the timeout has expired.

  Args:
    env: The environment instance.
    command_name: Name of the fixed sequential goal tracking command.

  Returns:
    Boolean tensor indicating which environments should terminate.
  """
  from .reachable_goal_command import FixedSequentialGoalCommand

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, FixedSequentialGoalCommand), (
    f"Command '{command_name}' is not a FixedSequentialGoalCommand."
  )
  return command_term.sequence_complete
