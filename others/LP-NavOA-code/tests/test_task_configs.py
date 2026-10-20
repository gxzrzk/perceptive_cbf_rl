"""Generic tests for task config integrity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.raycast_occlusion import (
  RAYCAST_NEAR_OCCLUSION_PARAM_KEYS,
  disable_raycast_near_occlusion,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg

VL53L8CX_SENSOR_PARAM_KEYS = (
  "vl53l8cx_invalid_prob_at_max",
  "vl53l8cx_shallow_incidence_invalid_prob",
  "vl53l8cx_shallow_incidence_cos_start",
  "vl53l8cx_shallow_incidence_cos_full",
)


@pytest.fixture(scope="module")
def all_task_ids() -> list[str]:
  """Get all registered task IDs."""
  return list_tasks()


def test_all_tasks_loadable(all_task_ids: list[str]) -> None:
  """All registered tasks should be loadable without errors."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}': {e}")


def test_all_tasks_have_play_config(all_task_ids: list[str]) -> None:
  """All tasks should be loadable in play mode."""
  for task_id in all_task_ids:
    try:
      cfg = load_env_cfg(task_id, play=True)
      assert isinstance(cfg, ManagerBasedRlEnvCfg), (
        f"Task {task_id} play mode did not return ManagerBasedRlEnvCfg"
      )
    except Exception as e:
      pytest.fail(f"Failed to load task '{task_id}' in play mode: {e}")


def test_play_mode_episode_length(all_task_ids: list[str]) -> None:
  """Play mode tasks should have infinite episode length."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.episode_length_s >= 1e9, (
      f"{task_id} (play mode) episode_length_s={cfg.episode_length_s}, expected >= 1e9"
    )


def test_play_mode_observation_corruption_matches_training(
  all_task_ids: list[str],
) -> None:
  """Play mode tasks should preserve training actor observation corruption.

  Play mode is used for policy deployment and interactive checks against the
  same noisy actor input distribution used during training.  Critic observation
  corruption remains disabled because the critic is not part of deployment.
  Data collection tasks have clean training actor inputs and therefore keep
  clean play actor inputs too.
  """
  for task_id in all_task_ids:
    train_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    assert "actor" in play_cfg.observations, (
      f"Play mode task {task_id} missing 'policy' observation group"
    )

    train_actor_obs = train_cfg.observations["actor"]
    play_actor_obs = play_cfg.observations["actor"]
    assert isinstance(play_actor_obs, ObservationGroupCfg), (
      f"Play mode task {task_id} policy observation is not ObservationGroupCfg"
    )
    assert isinstance(train_actor_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert play_actor_obs.enable_corruption == train_actor_obs.enable_corruption, (
      f"Play mode task {task_id} has enable_corruption="
      f"{play_actor_obs.enable_corruption}, expected "
      f"{train_actor_obs.enable_corruption} to match training"
    )


def test_training_mode_observation_corruption_enabled(all_task_ids: list[str]) -> None:
  """Training mode tasks should have observation corruption enabled for policy.

  Note: Data collection tasks (containing '-Data-') are excluded as they
  intentionally disable corruption for clean data collection.
  """
  for task_id in all_task_ids:
    # Skip data collection tasks - they intentionally disable corruption
    if "-Data-" in task_id:
      continue

    cfg = load_env_cfg(task_id)

    assert "actor" in cfg.observations, (
      f"Training task {task_id} missing 'policy' observation group"
    )

    policy_obs = cfg.observations["actor"]
    assert isinstance(policy_obs, ObservationGroupCfg), (
      f"Training task {task_id} policy observation is not ObservationGroupCfg"
    )

    assert policy_obs.enable_corruption, (
      f"Training task {task_id} has enable_corruption=False, expected True"
    )


@pytest.mark.parametrize(
  "task_id",
  [
    "Mjlab-CircularGoal-Flat-G1Rw-Pure-v6",
    "Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6",
    "Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6",
    "Mjlab-StraightApproach-LowSpeed-G1-v6",
    "Mjlab-IndoorStraightApproach-G1-v6",
  ],
)
def test_disable_raycast_near_occlusion_removes_task_overrides(task_id: str) -> None:
  """Task-configured near-field occlusion should be removable via CLI helper."""
  cfg = load_env_cfg(task_id)
  front_params = cfg.observations["raycast"].terms["raycast"].params

  assert front_params is not None
  assert "near_occlusion_prob" in front_params
  assert front_params["near_occlusion_prob"] == pytest.approx(0.05)

  disabled_groups = disable_raycast_near_occlusion(cfg)

  assert "raycast" in disabled_groups
  for obs_name in disabled_groups:
    params = cfg.observations[obs_name].terms[obs_name].params
    assert params is not None
    assert all(key not in params for key in RAYCAST_NEAR_OCCLUSION_PARAM_KEYS)


def test_v6_raycast_sensor_model_keeps_interface() -> None:
  cfg = load_env_cfg("Mjlab-CircularGoal-Flat-G1Rw-Pure-v6")

  for obs_name in ("raycast", "raycast_left", "raycast_right"):
    group = cfg.observations[obs_name]
    term = group.terms[obs_name]
    params = term.params
    assert params is not None
    assert term.history_length == 1
    assert params["sensor_width"] == 8
    assert all(key in params for key in VL53L8CX_SENSOR_PARAM_KEYS)

  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
  for sensor_name in ("raycast", "raycast_left", "raycast_right"):
    sensor = sensors[sensor_name]
    assert sensor.pattern.width == 8
    assert sensor.pattern.height == 8
    assert sensor.max_distance == pytest.approx(4.0)


def test_vl53l8cx_fovy_override_does_not_change_invalid_model() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  script = """
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Mjlab-CircularGoal-Flat-G1Rw-Pure-v6")
params = cfg.observations["raycast"].terms["raycast"].params
assert params is not None
sensors = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
assert abs(sensors["raycast"].pattern.fovy - 45.0) < 1e-6
assert params["vl53l8cx_invalid_prob_at_max"] == 0.10
assert params["vl53l8cx_shallow_incidence_invalid_prob"] == 0.75
"""
  env = os.environ.copy()
  env["MJLAB_VL53L8CX_SQUARE_FOVY_DEG"] = "45.0"
  result = subprocess.run(
    [sys.executable, "-c", script],
    capture_output=True,
    cwd=repo_root,
    env=env,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr or result.stdout


def test_vl53l8cx_fovy_default_differs_from_override() -> None:
  base_cfg = load_env_cfg("Mjlab-CircularGoal-Flat-G1Rw-Pure-v6")
  base_params = base_cfg.observations["raycast"].terms["raycast"].params
  assert base_params is not None

  base_sensors = {sensor.name: sensor for sensor in base_cfg.scene.sensors or ()}
  assert base_sensors["raycast"].pattern.fovy != pytest.approx(45.0)
  for key in VL53L8CX_SENSOR_PARAM_KEYS:
    assert key in base_params


@pytest.mark.parametrize(
  ("task_id", "command_kind"),
  [
    ("Mjlab-CircularGoal-Flat-G1Rw-Pure-v6", "circular"),
    ("Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6", "reachable"),
    ("Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6", "reachable"),
    ("Mjlab-StraightApproach-LowSpeed-G1-v6", "sequential"),
    ("Mjlab-IndoorStraightApproach-G1-v6", "sequential"),
  ],
)
def test_navigation_tasks_enable_safety_envelope(
  task_id: str,
  command_kind: str,
) -> None:
  """Every task should enable the shared command-side safety envelope."""
  from mjlab.tasks.velocity.mdp.circular_goal_command import (
    CircularGoalTrackingCommandCfg,
  )
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  cfg = load_env_cfg(task_id)
  goal_cfg = cfg.commands["goal"]

  if command_kind == "circular":
    assert isinstance(goal_cfg, CircularGoalTrackingCommandCfg)
  elif command_kind == "sequential":
    assert isinstance(goal_cfg, FixedSequentialGoalCommandCfg)
  else:
    assert isinstance(goal_cfg, ReachableGoalCommandCfg)

  assert goal_cfg.safety_velocity_envelope.enabled
  assert goal_cfg.safety_velocity_envelope.safety_margin_m == pytest.approx(0.25)
  assert goal_cfg.safety_velocity_envelope.reaction_time_s == pytest.approx(0.15)
  assert goal_cfg.safety_velocity_envelope.braking_time_s == pytest.approx(0.8)


def test_critic_observation_corruption_always_disabled(all_task_ids: list[str]) -> None:
  """Critic observations should always have corruption disabled."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id)

    if "critic" not in cfg.observations:
      continue

    critic_obs = cfg.observations["critic"]
    assert isinstance(critic_obs, ObservationGroupCfg), (
      f"Task {task_id} critic observation is not ObservationGroupCfg"
    )

    assert not critic_obs.enable_corruption, (
      f"Task {task_id} has critic enable_corruption=True, expected False"
    )


def test_play_training_observation_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching observation structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    # Same observation groups.
    assert set(training_cfg.observations.keys()) == set(play_cfg.observations.keys()), (
      f"Observation groups mismatch between {task_id} training and play modes"
    )

    # Same observation terms within each group.
    for obs_group_name in training_cfg.observations:
      training_terms = set(training_cfg.observations[obs_group_name].terms.keys())
      play_terms = set(play_cfg.observations[obs_group_name].terms.keys())

      assert training_terms == play_terms, (
        f"Observation terms mismatch in group '{obs_group_name}' "
        f"between {task_id} training and play modes"
      )


def test_play_training_action_structure_match(all_task_ids: list[str]) -> None:
  """Play and training configs should have matching action structure."""
  for task_id in all_task_ids:
    training_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)

    assert set(training_cfg.actions.keys()) == set(play_cfg.actions.keys()), (
      f"Action structure mismatch between {task_id} training and play modes"
    )


def test_play_mode_disables_push_robot(all_task_ids: list[str]) -> None:
  """Play mode tasks should disable push_robot event."""
  for task_id in all_task_ids:
    cfg = load_env_cfg(task_id, play=True)
    assert "push_robot" not in cfg.events, (
      f"Play mode task {task_id} has push_robot event, expected it to be removed"
    )


def test_backbone_tasks_enable_keyboard_play_on_cold_start() -> None:
  """Backbone curriculum tasks should expose keyboard play commands."""
  repo_root = Path(__file__).resolve().parents[1]
  script = """
import os

os.environ[\"KEYBOARD_CONTROL\"] = \"1\"

import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp.keyboard_command import (
  KeyboardCircularGoalTrackingCommandCfg,
)
from mjlab.tasks.velocity.mdp.observations import circular_goal_command

task_ids = [
  \"Mjlab-CircularGoal-Flat-G1Rw-Pure-v6\",
  \"Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6\",
  \"Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6\",
]

for task_id in task_ids:
  cfg = load_env_cfg(task_id, play=True)
  goal_cfg = cfg.commands[\"goal\"]
  assert isinstance(goal_cfg, KeyboardCircularGoalTrackingCommandCfg), (
    task_id,
    type(goal_cfg).__name__,
  )
  assert goal_cfg.safety_enabled, f\"{task_id} keyboard play must enable safety\"
  assert cfg.observations[\"actor\"].terms[\"goal_command\"].func is circular_goal_command
  assert cfg.observations[\"critic\"].terms[\"goal_command\"].func is circular_goal_command
"""
  result = subprocess.run(
    [sys.executable, "-c", script],
    capture_output=True,
    cwd=repo_root,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize(
  "task_id",
  [
    "Mjlab-StraightApproach-LowSpeed-G1-v6",
    "Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6",
    "Mjlab-IndoorStraightApproach-G1-v6",
    "Mjlab-IndoorStraightApproach-NoPP-G1-v6",
  ],
)
def test_navigation_tasks_share_the_command_side_safety_envelope(task_id: str) -> None:
  """Both navigation tasks and their planner-free twins share the safety filter.

  The braking behaviour lives entirely in the shared command-side envelope, which
  every evaluated condition reuses; there is no separate speed-penalty reward.
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
  )

  cfg = load_env_cfg(task_id)
  goal_cfg = cfg.commands["goal"]

  assert isinstance(goal_cfg, FixedSequentialGoalCommandCfg)
  assert goal_cfg.safety_velocity_envelope.enabled
  assert goal_cfg.safety_velocity_envelope.safety_margin_m == pytest.approx(0.25)
  assert goal_cfg.safety_velocity_envelope.reaction_time_s == pytest.approx(0.15)
  assert goal_cfg.safety_velocity_envelope.braking_time_s == pytest.approx(0.8)
  assert "directional_safety_speed" not in cfg.rewards
