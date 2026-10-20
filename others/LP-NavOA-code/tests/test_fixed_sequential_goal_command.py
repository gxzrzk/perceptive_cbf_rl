from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.velocity.mdp.reachable_goal_command import (
  FixedSequentialGoalCommand,
  FixedSequentialGoalCommandCfg,
)


class _FakeScene:
  def __init__(self, robot: SimpleNamespace, env_origins: torch.Tensor):
    self._robot = robot
    self.env_origins = env_origins

  def __getitem__(self, name: str) -> SimpleNamespace:
    assert name == "robot"
    return self._robot


class _FakeEnv:
  def __init__(self, robot: SimpleNamespace, env_origins: torch.Tensor):
    self.num_envs = env_origins.shape[0]
    self.device = str(env_origins.device)
    self.scene = _FakeScene(robot, env_origins)


def _make_command() -> FixedSequentialGoalCommand:
  device = "cpu"
  goal_offsets = [(0.0, float(i + 1)) for i in range(9)]
  root_pos = torch.zeros(1, 3, device=device)
  root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=root_pos,
      root_link_quat_w=root_quat,
    )
  )
  env_origins = torch.zeros(1, 3, device=device)
  env = _FakeEnv(robot, env_origins)
  cfg = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),
    goal_offsets=goal_offsets,
    key_goal_indices=[7, 8],
    target_velocity=0.8,
    reach_threshold=1.0,
    intermediate_reach_threshold=0.5,
  )
  command = FixedSequentialGoalCommand(cfg, env)  # type: ignore[arg-type]
  env_ids = torch.tensor([0], device=device)
  command.reset(env_ids)
  return command


def _set_robot_xy(command: FixedSequentialGoalCommand, x: float, y: float) -> None:
  command.robot.data.root_link_pos_w[0, 0] = x
  command.robot.data.root_link_pos_w[0, 1] = y
  command.robot.data.root_link_pos_w[0, 2] = 0.0


def _set_current_waypoint(command: FixedSequentialGoalCommand, idx: int) -> None:
  env_ids = torch.tensor([0], device=command.device)
  command.current_goal_idx[0] = idx
  key_idx = sum(key_goal_idx < idx for key_goal_idx in command._key_goal_indices)
  command.current_key_goal_idx[0] = key_idx
  command.sequence_complete[0] = False
  command.goal_reached[0] = False
  command._update_goal_position(env_ids)
  command._update_key_goal_position(env_ids)


def test_reaching_later_key_goal_completes_previous_waypoints() -> None:
  command = _make_command()
  _set_current_waypoint(command, 2)
  _set_robot_xy(command, 0.0, 8.0)

  command._update_command()

  assert command.current_goal_idx[0].item() == 8
  assert command.current_key_goal_idx[0].item() == 1
  assert not command.sequence_complete[0].item()
  assert command.goal_pos_w[0, 1].item() == pytest.approx(9.0)
  assert command.key_goal_pos_w[0, 1].item() == pytest.approx(9.0)
  assert command.metrics["goals_reached"][0].item() == pytest.approx(6.0)


def test_reaching_later_intermediate_waypoint_completes_previous_waypoints() -> None:
  command = _make_command()
  _set_current_waypoint(command, 2)
  _set_robot_xy(command, 0.0, 6.0)

  command._update_command()

  assert command.current_goal_idx[0].item() == 6
  assert command.current_key_goal_idx[0].item() == 0
  assert not command.sequence_complete[0].item()
  assert command.goal_pos_w[0, 1].item() == pytest.approx(7.0)
  assert command.key_goal_pos_w[0, 1].item() == pytest.approx(8.0)
  assert command.metrics["goals_reached"][0].item() == pytest.approx(4.0)


def test_reaching_final_key_goal_completes_sequence() -> None:
  command = _make_command()
  _set_current_waypoint(command, 2)
  _set_robot_xy(command, 0.0, 9.0)

  command._update_command()

  assert command.current_goal_idx[0].item() == 9
  assert command.current_key_goal_idx[0].item() == 2
  assert command.sequence_complete[0].item()
  assert command.goal_reached[0].item()
  assert command.goal_pos_w[0, 1].item() == pytest.approx(9.0)
  assert command.key_goal_pos_w[0, 1].item() == pytest.approx(9.0)
  assert command.metrics["goals_reached"][0].item() == pytest.approx(7.0)


def test_no_reached_future_waypoint_keeps_indices_unchanged() -> None:
  command = _make_command()
  _set_current_waypoint(command, 2)
  _set_robot_xy(command, 2.0, 4.0)

  command._update_command()

  assert command.current_goal_idx[0].item() == 2
  assert command.current_key_goal_idx[0].item() == 0
  assert not command.sequence_complete[0].item()
  assert command.goal_pos_w[0, 1].item() == pytest.approx(3.0)
  assert command.key_goal_pos_w[0, 1].item() == pytest.approx(8.0)
  assert command.metrics["goals_reached"][0].item() == pytest.approx(0.0)
