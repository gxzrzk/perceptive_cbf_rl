"""Unitree G1 velocity environment configurations."""

import math
import os
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots import (
  get_active_g1_action_scale,
  get_active_g1_n3_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.heightfield_terrains import (
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfWaveTerrainCfg,
)
from mjlab.terrains.primitive_terrains import (
  BoxFlatTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

_ROBOT_HEIGHT_OFFSET = 0.35

_VL53L8CX_SENSOR_WIDTH = 8
_VL53L8CX_SENSOR_HEIGHT = 8
_VL53L8CX_MAX_DISTANCE_M = 4.0
_VL53L8CX_DIAGONAL_FOV_DEG = 65.0
_VL53L8CX_UPDATE_PERIOD_S = 1.0 / 12.0
_VL53L8CX_NOISE_STD_MIN_M = 0.01
_VL53L8CX_NOISE_REL_STD = 0.05
_VL53L8CX_CORNER_REL_STD_EXTRA = 0.04
_VL53L8CX_INVALID_DISTANCE_START_M = 2.0
_VL53L8CX_INVALID_PROB_AT_MAX = 0.10
_VL53L8CX_CORNER_INVALID_PROB_EXTRA = 0.05
_VL53L8CX_CORNER_WIDTH = 2
_VL53L8CX_SHALLOW_INCIDENCE_INVALID_PROB = 0.75
_VL53L8CX_SHALLOW_INCIDENCE_COS_START = 0.35
_VL53L8CX_SHALLOW_INCIDENCE_COS_FULL = 0.08
_VL53L8CX_SQUARE_FOVY_ENV = "MJLAB_VL53L8CX_SQUARE_FOVY_DEG"
_RAYCAST_CAMERA_ORIENT_RANGE_RAD = (-0.262, 0.262)
_RAYCAST_FORWARD_UPWARD_PITCH_RANGE_RAD = (-0.157, -0.035)
_RAYCAST_LEFT_UPWARD_ROLL_RANGE_RAD = (0.035, 0.157)
_RAYCAST_RIGHT_UPWARD_ROLL_RANGE_RAD = (-0.157, -0.035)

# Stage 1/2/3 training-only near-field occlusion DR for tiny range sensors.
_NEAR_OCCLUSION_PROB = 0.05
_NEAR_OCCLUSION_DISTANCE_RANGE_M = (0.02, 0.08)
_NEAR_OCCLUSION_RAY_PROB = 0.9
_NEAR_OCCLUSION_FULL_FRAME_PROB = 0.35
# Spike DR: random pixels jump to 1.0 (max distance / no-obstacle).
# Bottom 2 rows have higher probability due to ground-proximity false-max readings.
_SPIKE_MAX_PROB = 0.01
_SPIKE_BOTTOM_ROWS_PROB = 0.05
_SPIKE_SENSOR_WIDTH = 8
_SAFETY_MARGIN_M = 0.25
_REACTION_TIME_S = 0.15
_BRAKING_TIME_S = 0.8
_STOP_EPSILON = 0.15
_SIDE_OFFSET_M = 0.35
_HARD_STOP_M = 0.40
_BLOCKED_TURN_SCALE = 0.0
_MIN_TURN_VELOCITY = 0.20
_TURN_THRESHOLD_RAD = 0.35
_CURRICULUM_V6_PLAY_MAX_SPEED_MPS = 1.5
_MANUAL_PLAY_OBSTACLE_ENTITY_PREFIX = "manual_obstacle"
_MANUAL_PLAY_OBSTACLE_ENTITY_NAMES = (
  f"{_MANUAL_PLAY_OBSTACLE_ENTITY_PREFIX}_a",
  f"{_MANUAL_PLAY_OBSTACLE_ENTITY_PREFIX}_b",
)


def _square_pinhole_fovy_from_diagonal(diagonal_fov_deg: float) -> float:
  """Convert square diagonal FoV to pinhole horizontal/vertical FoV."""
  diagonal_rad = math.radians(diagonal_fov_deg)
  return math.degrees(2.0 * math.atan(math.tan(diagonal_rad / 2.0) / math.sqrt(2.0)))


_VL53L8CX_PINHOLE_SQUARE_FOVY_DEG = _square_pinhole_fovy_from_diagonal(
  _VL53L8CX_DIAGONAL_FOV_DEG
)


def _vl53l8cx_square_fovy_deg() -> float:
  """Return the VL53L8CX pinhole FoV, optionally overridden by env var."""
  override = os.getenv(_VL53L8CX_SQUARE_FOVY_ENV)
  if override is None or override.strip() == "":
    return _VL53L8CX_PINHOLE_SQUARE_FOVY_DEG

  try:
    fovy_deg = float(override)
  except ValueError as exc:
    raise ValueError(
      f"{_VL53L8CX_SQUARE_FOVY_ENV} must be a numeric degree value, got {override!r}."
    ) from exc

  if not 0.0 < fovy_deg < 180.0:
    raise ValueError(
      f"{_VL53L8CX_SQUARE_FOVY_ENV} must be in (0, 180) degrees, got {fovy_deg}."
    )
  return fovy_deg


def _vl53l8cx_raycast_observation_params(sensor_name: str) -> dict[str, object]:
  """Observation params for a VL53L8CX-like 8x8 normalized range grid."""
  return {
    "sensor_name": sensor_name,
    "sensor_width": _VL53L8CX_SENSOR_WIDTH,
    "vl53l8cx_noise_std_min_m": _VL53L8CX_NOISE_STD_MIN_M,
    "vl53l8cx_noise_rel_std": _VL53L8CX_NOISE_REL_STD,
    "vl53l8cx_corner_rel_std_extra": _VL53L8CX_CORNER_REL_STD_EXTRA,
    "vl53l8cx_invalid_distance_start_m": _VL53L8CX_INVALID_DISTANCE_START_M,
    "vl53l8cx_invalid_prob_at_max": _VL53L8CX_INVALID_PROB_AT_MAX,
    "vl53l8cx_corner_invalid_prob_extra": _VL53L8CX_CORNER_INVALID_PROB_EXTRA,
    "vl53l8cx_corner_width": _VL53L8CX_CORNER_WIDTH,
    "vl53l8cx_shallow_incidence_invalid_prob": (
      _VL53L8CX_SHALLOW_INCIDENCE_INVALID_PROB
    ),
    "vl53l8cx_shallow_incidence_cos_start": _VL53L8CX_SHALLOW_INCIDENCE_COS_START,
    "vl53l8cx_shallow_incidence_cos_full": _VL53L8CX_SHALLOW_INCIDENCE_COS_FULL,
  }


def _estimate_g1_height() -> float:
  """Approximate robot standing height from the base pose."""
  cfg = get_active_g1_n3_robot_cfg()
  try:
    base_height = float(cfg.init_state.pos[2])
  except Exception:
    base_height = 0.9
  return base_height + _ROBOT_HEIGHT_OFFSET


def _enable_near_occlusion_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Inject close-range random occlusions into raycast observations."""
  for raycast_group_name in ("raycast", "raycast_left", "raycast_right"):
    rc_term = cfg.observations[raycast_group_name].terms[raycast_group_name]
    rc_params = dict(rc_term.params or {})
    rc_params.update(
      {
        "near_occlusion_prob": _NEAR_OCCLUSION_PROB,
        "near_occlusion_distance_range": _NEAR_OCCLUSION_DISTANCE_RANGE_M,
        "near_occlusion_ray_prob": _NEAR_OCCLUSION_RAY_PROB,
        "near_occlusion_full_frame_prob": _NEAR_OCCLUSION_FULL_FRAME_PROB,
      }
    )
    rc_term.params = rc_params


def _enable_spike_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Inject random max-distance spikes into raycast observations.

  Simulates real sensor false-max readings where a pixel randomly reports
  maximum distance (no obstacle).  Bottom 2 rows have a higher spike
  probability due to ground-proximity effects observed on hardware.
  """
  for raycast_group_name in ("raycast", "raycast_left", "raycast_right"):
    rc_term = cfg.observations[raycast_group_name].terms[raycast_group_name]
    rc_params = dict(rc_term.params or {})
    rc_params.update(
      {
        "spike_max_prob": _SPIKE_MAX_PROB,
        "spike_bottom_rows_prob": _SPIKE_BOTTOM_ROWS_PROB,
        "sensor_width": _SPIKE_SENSOR_WIDTH,
      }
    )
    rc_term.params = rc_params


def _enable_command_safety_envelope(cfg: ManagerBasedRlEnvCfg) -> None:
  """Enable Stage-4-style command-side braking on the active goal command."""
  from mjlab.tasks.velocity.mdp.circular_goal_command import (
    CircularGoalTrackingCommandCfg,
  )
  from mjlab.tasks.velocity.mdp.keyboard_command import (
    KeyboardCircularGoalTrackingCommandCfg,
  )
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  assert cfg.commands is not None
  goal_cmd = cfg.commands["goal"]

  if isinstance(goal_cmd, ReachableGoalCommandCfg):
    goal_cmd.safety_velocity_envelope = (
      ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
        enabled=True,
        forward_sensor_name="raycast",
        left_sensor_name="raycast_left",
        right_sensor_name="raycast_right",
        skip_ground_cols=3,
        safety_margin_m=_SAFETY_MARGIN_M,
        reaction_time_s=_REACTION_TIME_S,
        braking_time_s=_BRAKING_TIME_S,
        stop_velocity_epsilon=_STOP_EPSILON,
        side_point_forward_offset_m=_SIDE_OFFSET_M,
        hard_stop_distance_m=_HARD_STOP_M,
        blocked_turn_scale=_BLOCKED_TURN_SCALE,
        min_turn_velocity=_MIN_TURN_VELOCITY,
        turn_threshold_rad=_TURN_THRESHOLD_RAD,
      )
    )
  elif isinstance(goal_cmd, FixedSequentialGoalCommandCfg):
    goal_cmd.safety_velocity_envelope = (
      ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
        enabled=True,
        forward_sensor_name="raycast",
        left_sensor_name="raycast_left",
        right_sensor_name="raycast_right",
        skip_ground_cols=3,
        safety_margin_m=_SAFETY_MARGIN_M,
        reaction_time_s=_REACTION_TIME_S,
        braking_time_s=_BRAKING_TIME_S,
        stop_velocity_epsilon=_STOP_EPSILON,
        side_point_forward_offset_m=_SIDE_OFFSET_M,
        hard_stop_distance_m=_HARD_STOP_M,
        blocked_turn_scale=_BLOCKED_TURN_SCALE,
        min_turn_velocity=_MIN_TURN_VELOCITY,
        turn_threshold_rad=_TURN_THRESHOLD_RAD,
      )
    )
  elif isinstance(goal_cmd, CircularGoalTrackingCommandCfg):
    goal_cmd.safety_velocity_envelope = (
      CircularGoalTrackingCommandCfg.SafetyVelocityEnvelopeCfg(
        enabled=True,
        forward_sensor_name="raycast",
        left_sensor_name="raycast_left",
        right_sensor_name="raycast_right",
        skip_ground_cols=3,
        safety_margin_m=_SAFETY_MARGIN_M,
        reaction_time_s=_REACTION_TIME_S,
        braking_time_s=_BRAKING_TIME_S,
        stop_velocity_epsilon=_STOP_EPSILON,
        side_point_forward_offset_m=_SIDE_OFFSET_M,
        hard_stop_distance_m=_HARD_STOP_M,
        blocked_turn_scale=_BLOCKED_TURN_SCALE,
        min_turn_velocity=_MIN_TURN_VELOCITY,
        turn_threshold_rad=_TURN_THRESHOLD_RAD,
      )
    )
  elif isinstance(goal_cmd, KeyboardCircularGoalTrackingCommandCfg):
    goal_cmd.safety_enabled = True
    goal_cmd.safety_forward_sensor = "raycast"
    goal_cmd.safety_left_sensor = "raycast_left"
    goal_cmd.safety_right_sensor = "raycast_right"
    goal_cmd.safety_skip_ground_cols = 3
    goal_cmd.safety_margin_m = _SAFETY_MARGIN_M
    goal_cmd.safety_reaction_time_s = _REACTION_TIME_S
    goal_cmd.safety_braking_time_s = _BRAKING_TIME_S
    goal_cmd.safety_stop_epsilon = _STOP_EPSILON
    goal_cmd.safety_side_offset_m = _SIDE_OFFSET_M
    goal_cmd.safety_hard_stop_m = _HARD_STOP_M
    goal_cmd.safety_blocked_turn_scale = _BLOCKED_TURN_SCALE
    goal_cmd.safety_min_turn_velocity = _MIN_TURN_VELOCITY
    goal_cmd.safety_turn_threshold_rad = _TURN_THRESHOLD_RAD


def _apply_g1_reward_body(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace non-tracking rewards with the g1 deployed reward body.

  Keeps the 4 goal-tracking rewards (``track_ang_vel_cmd``,
  ``track_goal_velocity``, ``track_linear_velocity_z``,
  ``track_ang_vel_xy``) intact and replaces everything else with the
  reward body from
  ``src/mjlab/tasks/velocity/config/g1``.

  Does **not** modify terminations — callers manage those independently.
  """
  assert cfg.rewards is not None

  # ---- Remove v2/v3-only rewards not part of g1 reward body ----
  for key in (
    "velocity_tracking_log",
    "air_time",
    "foot_swing_height",
    "body_collision",
    "feet_collision",
    "directional_safety_speed",
    "dof_pos_limits",
  ):
    cfg.rewards.pop(key, None)

  # ---- Replace upright (exp-gaussian) with flat_orientation_l2 (g1 style) ----
  cfg.rewards.pop("upright", None)
  cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
    func=envs_mdp.flat_orientation_l2,
    weight=-5.0,
  )

  # ---- Add g1-specific constraint rewards ----
  cfg.rewards["is_terminated"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-200.0,
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=envs_mdp.joint_acc_l2,
    weight=-2.5e-7,
  )
  cfg.rewards["joint_pos_limits"] = RewardTermCfg(
    func=envs_mdp.joint_pos_limits,
    weight=-10.0,
  )

  # ---- Adjust shared reward weights to g1 values ----
  cfg.rewards["angular_momentum"].weight = -0.025
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["foot_clearance"].weight = -1.0
  cfg.rewards["foot_slip"].weight = -0.25
  cfg.rewards["soft_landing"].weight = -1e-3

  # ---- In-place turn bonus ----
  cfg.rewards["in_place_turn_bonus"] = RewardTermCfg(
    func=mdp.in_place_turn_bonus,
    weight=1.0,
    params={
      "command_name": "goal",
      "heading_threshold": 0.8,
      "max_ang_vel": 1.0,
    },
  )

  # ---- Update pose std values to g1 deployed values ----
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
    r".*waist_roll.*": 0.1,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
  }


def _apply_g1rw_pure_reward_adjustments(cfg: ManagerBasedRlEnvCfg) -> None:
  """Align tracking reward weights to Unitree-G1-Flat (Pure) convention.

  Assumes ``_apply_g1_reward_body`` has already been applied.  Adjusts
  tracking weights and removes stability terms already covered by the g1
  reward body, matching ``unitree_g1_circular_goal_flat_g1rw_pure_h1_env_cfg``.
  """
  assert cfg.rewards is not None

  # Align tracking weights to Unitree-G1-Flat (1.0 / 1.0)
  cfg.rewards["track_goal_velocity"].weight = 1.0
  cfg.rewards["track_ang_vel_cmd"].weight = 1.0

  # Remove stability terms already covered by G1 reward body
  cfg.rewards.pop("track_linear_velocity_z", None)
  cfg.rewards.pop("track_ang_vel_xy", None)
  cfg.rewards.pop("velocity_tracking_log", None)

  # Remove ang_vel_cmd_weight curriculum (G1-Flat has no reward curriculum)
  assert cfg.curriculum is not None
  cfg.curriculum.pop("ang_vel_cmd_weight", None)

  # Align command_threshold / walking_threshold to Unitree-G1-Flat
  cfg.rewards["pose"].params["walking_threshold"] = 0.1
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.1
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.1
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.1


def _enable_keyboard_circular_goal_play(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace goal tracking play commands with keyboard circular goal control."""
  from mjlab.tasks.velocity.mdp.keyboard_command import (
    KeyboardCircularGoalTrackingCommandCfg,
  )

  assert cfg.commands is not None
  goal_cfg = cfg.commands["goal"]

  default_velocity = float(getattr(goal_cfg, "target_velocity", 0.5))
  velocity_range = getattr(goal_cfg, "target_velocity_range", None)
  max_velocity = (
    float(velocity_range[1])
    if velocity_range is not None and len(velocity_range) >= 2
    else 2.0
  )

  cfg.commands["goal"] = KeyboardCircularGoalTrackingCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    goal_radius=3.0,
    max_velocity=max_velocity,
    default_velocity=default_velocity,
    angle_step_deg=15.0,
    smoothing_alpha=0.15,
    debug_vis=True,
  )
  cfg.commands["goal"].viz.z_offset = 0.3

  for obs_group_name in ("actor", "critic"):
    obs_terms = cfg.observations[obs_group_name].terms
    if "goal_command" in obs_terms:
      obs_terms["goal_command"] = ObservationTermCfg(
        func=mdp.circular_goal_command,
        params={"command_name": "goal"},
      )

  if cfg.curriculum is not None and "goal_velocity" in cfg.curriculum:
    del cfg.curriculum["goal_velocity"]


def _cap_play_goal_max_speed(cfg: ManagerBasedRlEnvCfg, max_speed: float) -> None:
  """Set play-mode goal command maximum speed without affecting training."""
  from mjlab.tasks.velocity.mdp.circular_goal_command import (
    CircularGoalTrackingCommandCfg,
  )
  from mjlab.tasks.velocity.mdp.keyboard_command import (
    KeyboardCircularGoalTrackingCommandCfg,
  )
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  assert cfg.commands is not None
  goal_cmd = cfg.commands["goal"]

  if isinstance(goal_cmd, KeyboardCircularGoalTrackingCommandCfg):
    goal_cmd.max_velocity = max_speed
    goal_cmd.default_velocity = min(float(goal_cmd.default_velocity), max_speed)
    return

  if isinstance(
    goal_cmd,
    (
      CircularGoalTrackingCommandCfg,
      ReachableGoalCommandCfg,
      FixedSequentialGoalCommandCfg,
    ),
  ):
    velocity_range = goal_cmd.target_velocity_range
    if velocity_range is None:
      return
    min_velocity, _ = velocity_range
    capped_min_velocity = min(float(min_velocity), max_speed)
    goal_cmd.target_velocity_range = (capped_min_velocity, max_speed)
    goal_cmd.target_velocity = min(float(goal_cmd.target_velocity), max_speed)


def _get_manual_play_obstacle_entity_cfg(
  *,
  geom_type: str,
  size: str,
  height_center: float,
  main_rgba: tuple[float, float, float, float],
  marker_rgba: tuple[float, float, float, float],
  marker_pos: str,
  marker_size: str,
) -> EntityCfg:
  """Create a play-only obstacle that can be dragged in the native viewer.

  The main geom stays in group 0 so the existing raycast sensors detect it.
  A second visual-only marker geom makes the draggable obstacle easy to spot.
  Collisions are disabled so the obstacle acts as a pure sensing target.
  Gravity compensation keeps it suspended without ground contact.
  """
  xml = f"""
    <mujoco>
      <default>
        <joint damping="6.0"/>
      </default>
      <worldbody>
        <body name="obstacle" gravcomp="1">
          <freejoint name="obstacle_joint"/>
          <geom
            name="obstacle_geom"
            type="{geom_type}"
            size="{size}"
            mass="8.0"
            rgba="{main_rgba[0]} {main_rgba[1]} {main_rgba[2]} {main_rgba[3]}"
            group="0"
            contype="0"
            conaffinity="0"
          />
          <geom
            name="marker_geom"
            type="sphere"
            pos="{marker_pos}"
            size="{marker_size}"
            rgba="{marker_rgba[0]} {marker_rgba[1]} {marker_rgba[2]} {marker_rgba[3]}"
            group="3"
            contype="0"
            conaffinity="0"
          />
        </body>
      </worldbody>
    </mujoco>
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(6.0, 4.0, height_center)),
    spec_fn=lambda: mujoco.MjSpec.from_string(xml),
  )


def _reset_manual_play_obstacles(
  env,
  env_ids: torch.Tensor | None,
  robot_cfg: SceneEntityCfg | None = None,
) -> None:
  """Spawn draggable sensing obstacles near the robot on reset."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if robot_cfg is None:
    robot_cfg = SceneEntityCfg("robot")

  robot = env.scene[robot_cfg.name]
  robot_default_root_state = robot.data.default_root_state[env_ids]
  robot_spawn_pos = robot_default_root_state[:, 0:3] + env.scene.env_origins[env_ids]

  obstacle_layouts = {
    _MANUAL_PLAY_OBSTACLE_ENTITY_NAMES[0]: {
      "xy_offset": (1.6, 0.85),
      "height": 0.95,
    },
    _MANUAL_PLAY_OBSTACLE_ENTITY_NAMES[1]: {
      "xy_offset": (2.0, -0.95),
      "height": 0.75,
    },
  }

  for obstacle_name, layout in obstacle_layouts.items():
    obstacle = env.scene[obstacle_name]
    root_state = torch.zeros((len(env_ids), 13), device=env.device)
    planar_offset = torch.tensor(
      layout["xy_offset"], device=env.device, dtype=robot_spawn_pos.dtype
    )
    root_state[:, 0:2] = robot_spawn_pos[:, 0:2] + planar_offset
    root_state[:, 2] = layout["height"]
    root_state[:, 3] = 1.0
    obstacle.write_root_state_to_sim(root_state, env_ids=env_ids)


def _enable_manual_play_obstacle(cfg: ManagerBasedRlEnvCfg) -> None:
  """Inject draggable play-only obstacles for manual approach testing."""
  cfg.scene.entities[_MANUAL_PLAY_OBSTACLE_ENTITY_NAMES[0]] = (
    _get_manual_play_obstacle_entity_cfg(
      geom_type="box",
      size="0.25 0.75 0.9",
      height_center=0.95,
      main_rgba=(0.95, 0.45, 0.10, 0.78),
      marker_rgba=(1.0, 0.95, 0.15, 0.95),
      marker_pos="0 0 1.2",
      marker_size="0.12",
    )
  )
  cfg.scene.entities[_MANUAL_PLAY_OBSTACLE_ENTITY_NAMES[1]] = (
    _get_manual_play_obstacle_entity_cfg(
      geom_type="cylinder",
      size="0.30 0.75",
      height_center=0.75,
      main_rgba=(0.10, 0.80, 0.92, 0.78),
      marker_rgba=(0.95, 0.20, 0.85, 0.95),
      marker_pos="0 0 1.0",
      marker_size="0.11",
    )
  )
  cfg.events["reset_manual_obstacles"] = EventTermCfg(
    func=_reset_manual_play_obstacles,
    mode="reset",
    params={
      "robot_cfg": SceneEntityCfg("robot"),
    },
  )


@dataclass(kw_only=True)
@dataclass(kw_only=True)
@dataclass(kw_only=True)
@dataclass(kw_only=True)
@dataclass(kw_only=True)
@dataclass(kw_only=True)
class TrueMazeTerrainCfg(SubTerrainCfg):
  """True maze terrain using recursive backtracking algorithm.

  This creates a proper maze structure with connected corridors, unlike scattered
  random walls. The maze ensures:
  - All passages are connected (solvable maze)
  - Wide corridors for robot navigation (configurable width)
  - Multiple paths/loops via wall removal for more open exploration
  - Central spawn area with guaranteed open space

  Layout:
    - Boundary walls: Surround the terrain on all four sides
    - Maze grid: N x M cells, each cell can have walls on its edges
    - Corridor width: Minimum passage width for robot movement
    - Extra openings: Random walls removed to create loops and wider spaces
    - Asymmetric margins: Two sides have narrow margins, two sides have wide margins

  Curriculum based on difficulty:
    - Lower difficulty: More walls removed (easier navigation)
    - Higher difficulty: Fewer walls removed (more maze-like)
  """

  robot_height: float
  """Estimated robot standing height."""

  wall_height: float = 2.0
  """Height of all walls (should be taller than robot)."""

  wall_thickness: float = 0.2
  """Thickness of walls."""

  boundary_margin: float = 0.5
  """Margin from terrain edge to boundary wall center."""

  corridor_width: float = 4.0
  """Width of corridors between walls. Should be wide enough for robot."""

  maze_cells_x: int = 6
  """Number of maze cells in X direction."""

  maze_cells_y: int = 6
  """Number of maze cells in Y direction."""

  extra_openings_ratio: float = 0.3
  """Ratio of extra walls to remove for creating loops (0 to 1).
  Higher values create more open maze with multiple paths."""

  spawn_clear_cells: int = 2
  """Number of cells around spawn to keep clear (no walls)."""

  curriculum_step_threshold: int = 2000
  """Training iteration threshold for full maze difficulty."""

  friction: tuple[float, float, float] = (1.0, 0.1, 0.01)
  """Ground and wall friction coefficients."""

  # Asymmetric margin settings
  narrow_margin: float = 2.0
  """Narrow margin on two sides (left and bottom)."""

  wide_margin: float = 8.0
  """Wide margin on two sides (right and top)."""

  def _generate_maze(
    self, cells_x: int, cells_y: int, rng: np.random.Generator
  ) -> tuple[set[tuple[int, int, str]], tuple[int, int]]:
    """Generate maze using recursive backtracking (DFS).

    Returns:
      - Set of walls to keep (cell_x, cell_y, direction)
      - Spawn cell (center or near-center cell)
    """
    # Each cell has potential walls on 'N' (north/top) and 'E' (east/right)
    # Bottom and left walls are represented by adjacent cells
    # Initialize all walls as present
    walls: set[tuple[int, int, str]] = set()
    for x in range(cells_x):
      for y in range(cells_y):
        if y < cells_y - 1:  # North wall (not for top row)
          walls.add((x, y, "N"))
        if x < cells_x - 1:  # East wall (not for rightmost column)
          walls.add((x, y, "E"))

    # Track visited cells
    visited = [[False] * cells_y for _ in range(cells_x)]

    # Start from center cell
    start_x, start_y = cells_x // 2, cells_y // 2
    spawn_cell = (start_x, start_y)

    # DFS stack
    stack = [(start_x, start_y)]
    visited[start_x][start_y] = True

    directions = [
      (0, 1, "N", 0, 0, "N"),  # North: remove current cell's N wall
      (0, -1, "S", 0, -1, "N"),  # South: remove neighbor's N wall
      (1, 0, "E", 0, 0, "E"),  # East: remove current cell's E wall
      (-1, 0, "W", -1, 0, "E"),  # West: remove neighbor's E wall
    ]

    while stack:
      cx, cy = stack[-1]

      # Find unvisited neighbors
      unvisited = []
      for dx, dy, _, wall_cx_offset, wall_cy_offset, wall_dir in directions:
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < cells_x and 0 <= ny < cells_y and not visited[nx][ny]:
          unvisited.append((nx, ny, cx + wall_cx_offset, cy + wall_cy_offset, wall_dir))

      if unvisited:
        # Choose random unvisited neighbor
        nx, ny, wall_x, wall_y, wall_dir = unvisited[rng.integers(len(unvisited))]

        # Remove wall between current and neighbor
        wall_key = (wall_x, wall_y, wall_dir)
        if wall_key in walls:
          walls.remove(wall_key)

        # Move to neighbor
        visited[nx][ny] = True
        stack.append((nx, ny))
      else:
        # Backtrack
        stack.pop()

    return walls, spawn_cell

  def _remove_extra_walls(
    self,
    walls: set[tuple[int, int, str]],
    cells_x: int,
    cells_y: int,
    ratio: float,
    spawn_cell: tuple[int, int],
    rng: np.random.Generator,
  ) -> set[tuple[int, int, str]]:
    """Remove additional walls to create loops and more open space.

    Also FORCEFULLY clears walls within spawn_clear_cells radius of spawn point.
    """
    spawn_x, spawn_y = spawn_cell

    # First, FORCEFULLY remove ALL walls within spawn_clear_cells radius
    # This ensures the spawn area is completely clear
    walls_to_remove = set()
    for wall in walls:
      x, y, direction = wall
      # Check if this wall is within the spawn clear zone
      # A wall at (x, y, 'N') is between cell (x,y) and (x, y+1)
      # A wall at (x, y, 'E') is between cell (x,y) and (x+1, y)
      if direction == "N":
        # Wall is at north edge of cell (x, y)
        if (
          abs(x - spawn_x) <= self.spawn_clear_cells
          and abs(y - spawn_y) < self.spawn_clear_cells  # y to y+1 should be in range
          and abs(y + 1 - spawn_y) <= self.spawn_clear_cells
        ):
          walls_to_remove.add(wall)
      else:  # direction == "E"
        # Wall is at east edge of cell (x, y)
        if (
          abs(x - spawn_x) < self.spawn_clear_cells  # x to x+1 should be in range
          and abs(x + 1 - spawn_x) <= self.spawn_clear_cells
          and abs(y - spawn_y) <= self.spawn_clear_cells
        ):
          walls_to_remove.add(wall)

    # Remove all walls in spawn clear zone
    for wall in walls_to_remove:
      walls.discard(wall)

    if ratio <= 0:
      return walls

    walls_list = list(walls)
    rng.shuffle(walls_list)

    # Calculate how many to remove
    num_to_remove = int(len(walls_list) * ratio)

    # Remove walls to create loops
    removed = 0
    for wall in walls_list:
      if removed >= num_to_remove:
        break

      # Remove this wall to create a loop
      walls.discard(wall)
      removed += 1

    return walls

  def _cell_to_world(
    self,
    cell_x: int,
    cell_y: int,
    cells_x: int,
    cells_y: int,
    maze_origin_x: float,
    maze_origin_y: float,
    cell_size: float,
  ) -> tuple[float, float]:
    """Convert cell coordinates to world coordinates (cell center)."""
    world_x = maze_origin_x + (cell_x + 0.5) * cell_size
    world_y = maze_origin_y + (cell_y + 0.5) * cell_size
    return world_x, world_y

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    # Calculate terrain center
    center_x = self.size[0] / 2.0
    center_y = self.size[1] / 2.0

    # Ground plane
    plane = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, 0.02),
      pos=(center_x, center_y, -0.02),
      rgba=(0.45, 0.45, 0.5, 1.0),
    )
    plane.friction = np.array(self.friction)
    geometries = [TerrainGeometry(geom=plane, color=(0.45, 0.45, 0.5, 1.0))]

    half_height = self.wall_height / 2.0
    wall_color = (0.6, 0.55, 0.5, 1.0)  # Stone gray
    internal_wall_color = (0.5, 0.45, 0.4, 1.0)

    # === Boundary walls ===
    margin = self.boundary_margin

    # Bottom wall
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.wall_thickness / 2.0, half_height),
      pos=(center_x, margin, half_height),
      rgba=wall_color,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    # Top wall
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.wall_thickness / 2.0, half_height),
      pos=(center_x, self.size[1] - margin, half_height),
      rgba=wall_color,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    # Left wall
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.wall_thickness / 2.0, self.size[1] / 2.0, half_height),
      pos=(margin, center_y, half_height),
      rgba=wall_color,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    # Right wall
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.wall_thickness / 2.0, self.size[1] / 2.0, half_height),
      pos=(self.size[0] - margin, center_y, half_height),
      rgba=wall_color,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    # === Generate maze ===
    # Calculate maze area with asymmetric margins
    # Left and bottom sides have narrow margins, right and top have wide margins
    maze_margin_left = self.narrow_margin + self.wall_thickness
    maze_margin_right = self.wide_margin + self.wall_thickness
    maze_margin_bottom = self.narrow_margin + self.wall_thickness
    maze_margin_top = self.wide_margin + self.wall_thickness

    maze_width = self.size[0] - maze_margin_left - maze_margin_right
    maze_height = self.size[1] - maze_margin_bottom - maze_margin_top

    # Cell size = corridor width + wall thickness
    cell_size = self.corridor_width + self.wall_thickness

    # Adjust cells to fit the area
    cells_x = min(self.maze_cells_x, int(maze_width / cell_size))
    cells_y = min(self.maze_cells_y, int(maze_height / cell_size))

    if cells_x < 2 or cells_y < 2:
      # Terrain too small for maze, just return with boundary walls
      origin = np.array((center_x, center_y, 0.0))
      return TerrainOutput(origin=origin, geometries=geometries)

    # Position maze with asymmetric margins (closer to left and bottom)
    actual_maze_width = cells_x * cell_size
    actual_maze_height = cells_y * cell_size
    # Center the maze within the available asymmetric space
    maze_origin_x = maze_margin_left + (maze_width - actual_maze_width) / 2.0
    maze_origin_y = maze_margin_bottom + (maze_height - actual_maze_height) / 2.0

    # Generate base maze structure
    walls, spawn_cell = self._generate_maze(cells_x, cells_y, rng)

    # Map difficulty to extra openings
    # Higher difficulty = fewer openings (harder maze)
    # Lower difficulty = more openings (easier, more open)
    estimated_iteration = int(difficulty * 10000)
    if estimated_iteration < self.curriculum_step_threshold:
      # Early training: very open maze (many walls removed)
      opening_ratio = self.extra_openings_ratio + 0.3
    else:
      # Later training: scale with difficulty
      # difficulty 0.2 -> 0.4 ratio, difficulty 1.0 -> 0.1 ratio
      progress = (difficulty - 0.2) / 0.8
      opening_ratio = self.extra_openings_ratio * (1.0 - progress * 0.7)

    opening_ratio = max(0.1, min(0.6, opening_ratio))

    # Remove extra walls for loops
    walls = self._remove_extra_walls(
      walls, cells_x, cells_y, opening_ratio, spawn_cell, rng
    )

    # === Create wall geometries ===
    for wall_x, wall_y, direction in walls:
      if direction == "N":
        # North wall: horizontal wall at top of cell
        wx = maze_origin_x + (wall_x + 0.5) * cell_size
        wy = maze_origin_y + (wall_y + 1) * cell_size
        wall_length = cell_size / 2.0

        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(wall_length, self.wall_thickness / 2.0, half_height),
          pos=(wx, wy, half_height),
          rgba=internal_wall_color,
        )
      else:  # direction == "E"
        # East wall: vertical wall at right of cell
        wx = maze_origin_x + (wall_x + 1) * cell_size
        wy = maze_origin_y + (wall_y + 0.5) * cell_size
        wall_length = cell_size / 2.0

        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(self.wall_thickness / 2.0, wall_length, half_height),
          pos=(wx, wy, half_height),
          rgba=internal_wall_color,
        )

      geom.contype = 1
      geom.conaffinity = 1
      geom.friction = np.array(self.friction)
      geometries.append(TerrainGeometry(geom=geom, color=internal_wall_color))

    # Spawn origin at spawn cell center (with small random offset)
    spawn_world_x, spawn_world_y = self._cell_to_world(
      spawn_cell[0],
      spawn_cell[1],
      cells_x,
      cells_y,
      maze_origin_x,
      maze_origin_y,
      cell_size,
    )
    spawn_offset_x = float(rng.uniform(-0.5, 0.5))
    spawn_offset_y = float(rng.uniform(-0.5, 0.5))
    origin = np.array(
      (spawn_world_x + spawn_offset_x, spawn_world_y + spawn_offset_y, 0.0)
    )

    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
@dataclass(kw_only=True)
class UniformPillarFieldTerrainCfg(SubTerrainCfg):
  """Pillar field with pillars placed uniformly at random across the terrain.

  Unlike ``PillarFieldTerrainCfg`` which derives pillars from a maze structure,
  this terrain scatters short pillars (boxes) uniformly across the entire arena
  with random orientations.  There is **no spawn clear zone** — pillars may
  appear right next to the spawn point.

  Parameters control pillar count, dimensions and spacing.  A minimum distance
  between pillar centres is enforced via rejection sampling to avoid overlap.
  """

  robot_height: float
  """Estimated robot standing height."""

  wall_height: float = 2.0
  """Height of pillars."""

  wall_height_range: tuple[float, float] | None = None
  """Optional per-pillar random height range (min, max) in meters.

  If set, each pillar samples its own height independently in this range.
  If ``None``, all pillars use ``wall_height``.
  """

  wall_thickness: float = 0.2
  """Thickness (narrow dimension) of each pillar."""

  pillar_length: float = 2.0
  """Length (long dimension) of each pillar."""

  num_pillars: int = 50
  """Number of pillars to place."""

  min_pillar_spacing: float = 2.5
  """Minimum centre-to-centre distance between pillars [m]."""

  boundary_margin: float = 1.0
  """Margin from terrain edge where pillars are not placed [m]."""

  friction: tuple[float, float, float] = (1.0, 0.1, 0.01)
  """Ground and pillar friction coefficients."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    center_x = self.size[0] / 2.0
    center_y = self.size[1] / 2.0

    # Ground plane
    plane = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, 0.02),
      pos=(center_x, center_y, -0.02),
      rgba=(0.45, 0.45, 0.5, 1.0),
    )
    plane.friction = np.array(self.friction)
    geometries = [TerrainGeometry(geom=plane, color=(0.45, 0.45, 0.5, 1.0))]

    pillar_color = (0.55, 0.50, 0.45, 1.0)

    # Placeable region (inside boundary margin)
    x_lo = self.boundary_margin
    x_hi = self.size[0] - self.boundary_margin
    y_lo = self.boundary_margin
    y_hi = self.size[1] - self.boundary_margin

    # Place pillars with rejection sampling for minimum spacing
    placed: list[tuple[float, float]] = []
    max_attempts = self.num_pillars * 20  # generous budget
    attempts = 0

    while len(placed) < self.num_pillars and attempts < max_attempts:
      px = float(rng.uniform(x_lo, x_hi))
      py = float(rng.uniform(y_lo, y_hi))
      attempts += 1

      # Check minimum spacing against all placed pillars
      too_close = False
      for ox, oy in placed:
        if (px - ox) ** 2 + (py - oy) ** 2 < self.min_pillar_spacing**2:
          too_close = True
          break
      if too_close:
        continue

      placed.append((px, py))

      # Random orientation (angle in radians)
      angle = float(rng.uniform(0, math.pi))
      cos_a = math.cos(angle)
      sin_a = math.sin(angle)
      # MuJoCo quaternion: (w, x, y, z) for rotation around Z
      quat = (cos_a, 0.0, 0.0, sin_a)  # half-angle already in cos/sin

      if self.wall_height_range is not None:
        h_min, h_max = self.wall_height_range
        height = float(rng.uniform(min(h_min, h_max), max(h_min, h_max)))
        height = max(0.08, height)
      else:
        height = self.wall_height
      half_height = height / 2.0

      geom = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.pillar_length / 2.0, self.wall_thickness / 2.0, half_height),
        pos=(px, py, half_height),
        quat=quat,
        rgba=pillar_color,
      )
      geom.contype = 1
      geom.conaffinity = 1
      geom.friction = np.array(self.friction)
      geometries.append(TerrainGeometry(geom=geom, color=pillar_color))

    # Spawn at terrain centre (no clear zone)
    spawn_offset_x = float(rng.uniform(-0.5, 0.5))
    spawn_offset_y = float(rng.uniform(-0.5, 0.5))
    origin = np.array((center_x + spawn_offset_x, center_y + spawn_offset_y, 0.0))

    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
@dataclass(kw_only=True)
class SingleWallTerrainCfg(SubTerrainCfg):
  """Terrain with a single wall for simple obstacle avoidance training.

  This is the simplest obstacle avoidance scenario:
  - One horizontal wall in the center of the terrain
  - Robot spawns in front of the wall at a fixed distance
  - Goal is placed behind the wall
  - Robot must navigate around the wall to reach the goal

  Layout (top view, Y is forward):
    +---------------------------+
    |                           |
    |        Goal 2 (Y+10)      |
    |        Goal 1 (Y+6)       |
    |                           |
    |   ==================      |  <-- Wall at Y=center
    |                           |
    |       Robot (Y-4)         |
    |                           |
    +---------------------------+

  This terrain is designed for:
  - Simple, repeatable obstacle avoidance tests
  - Debugging local planner steering correction
  - Baseline comparisons before complex mazes
  """

  robot_height: float
  """Estimated robot standing height."""

  wall_height: float = 2.0
  """Height of the wall (should be taller than robot)."""

  wall_thickness: float = 0.2
  """Thickness of the wall."""

  wall_width: float = 8.0
  """Width of the wall (horizontal extent)."""

  wall_width_range: tuple[float, float] | None = None
  """If set, randomize wall width uniformly within (min, max). Overrides wall_width."""

  wall_bottom_gap_range: tuple[float, float] | None = None
  """If set, cut a random bottom gap while keeping the wall top height unchanged."""

  robot_distance_from_wall: float = 4.0
  """Distance from robot spawn point to wall center."""

  friction: tuple[float, float, float] = (1.0, 0.1, 0.01)
  """Ground and wall friction coefficients."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    # Calculate terrain center
    center_x = self.size[0] / 2.0
    center_y = self.size[1] / 2.0

    # Ground plane
    plane = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, 0.02),
      pos=(center_x, center_y, -0.02),
      rgba=(0.45, 0.45, 0.5, 1.0),
    )
    plane.friction = np.array(self.friction)
    geometries = [TerrainGeometry(geom=plane, color=(0.45, 0.45, 0.5, 1.0))]

    wall_color = (0.5, 0.45, 0.4, 1.0)  # Brown/stone color

    # Determine wall width (randomize if range is set)
    if self.wall_width_range is not None:
      actual_wall_width = rng.uniform(
        self.wall_width_range[0], self.wall_width_range[1]
      )
    else:
      actual_wall_width = self.wall_width

    if self.wall_bottom_gap_range is not None:
      wall_bottom_gap = float(
        rng.uniform(self.wall_bottom_gap_range[0], self.wall_bottom_gap_range[1])
      )
    else:
      wall_bottom_gap = 0.0
    wall_solid_height = max(float(self.wall_height) - wall_bottom_gap, 1e-3)
    wall_half_height = wall_solid_height / 2.0

    # === Single horizontal wall at terrain center ===
    # Wall is centered at (center_x, center_y), extending along X axis
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(actual_wall_width / 2.0, self.wall_thickness / 2.0, wall_half_height),
      pos=(center_x, center_y, wall_bottom_gap + wall_half_height),
      rgba=wall_color,
    )
    geom.contype = 1
    geom.conaffinity = 1
    geom.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=geom, color=wall_color))

    # Spawn origin: robot spawns in front of the wall (Y - robot_distance_from_wall)
    # Robot faces +Y direction (towards the wall)
    spawn_x = center_x
    spawn_y = center_y - self.robot_distance_from_wall
    origin = np.array((spawn_x, spawn_y, 0.0))

    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class OpenFieldTerrainCfg(SubTerrainCfg):
  """Flat terrain with no obstacles for pure goal-directed navigation training.

  This is the simplest possible scenario: a flat ground plane with no walls.
  The robot must learn to navigate toward a goal using only direction information.
  Spawn point is at the same relative position as SingleWallTerrainCfg (center_x,
  center_y - robot_distance_from_wall) for consistency with the mixed-scenario
  training pipeline.

  Used in mixed-scenario training to teach the local planner network stable
  goal-directed behavior in open spaces without obstacle avoidance bias.
  """

  robot_height: float
  """Estimated robot standing height."""

  robot_distance_from_wall: float = 4.0
  """Distance from robot spawn to terrain center (same as SingleWall for consistency)."""

  friction: tuple[float, float, float] = (1.0, 0.1, 0.01)
  """Ground friction coefficients."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")

    # Calculate terrain center
    center_x = self.size[0] / 2.0
    center_y = self.size[1] / 2.0

    # Ground plane only — no walls or obstacles
    plane = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, 0.02),
      pos=(center_x, center_y, -0.02),
      rgba=(0.45, 0.45, 0.5, 1.0),
    )
    plane.friction = np.array(self.friction)
    geometries = [TerrainGeometry(geom=plane, color=(0.45, 0.45, 0.5, 1.0))]

    # Spawn origin: same as SingleWall layout for consistency
    spawn_x = center_x
    spawn_y = center_y - self.robot_distance_from_wall
    origin = np.array((spawn_x, spawn_y, 0.0))

    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class IndoorRoomTerrainCfg(SubTerrainCfg):
  """Indoor room terrain: four outer walls + centre obstacle wall + wall furniture.

  Layout (top view, +Y is forward / robot heading direction):

    ╔═══════════════════════════════════╗
    ║                                   ║  ← North outer wall
    ║       Key Goal 2 (Y+15)           ║
    ║       Key Goal 1 (Y+12.5)         ║
    ║                                   ║
    ║  ☐ ☐                  ☐ ☐         ║  ← furniture near walls
    ║                                   ║
    ║   ════════════════════════         ║  ← centre obstacle wall at Y=terrain_cy
    ║                                   ║
    ║  ☐ ☐                  ☐ ☐         ║  ← furniture near walls
    ║                                   ║
    ║       Robot (Y-distance_from_wall) ║
    ║                                   ║
    ╚═══════════════════════════════════╝
      ↑                             ↑
    West wall                   East wall

  The outer walls create a realistic indoor environment so raycast sensors
  encounter wall returns in all directions, improving sim-to-real transfer.
  Furniture (tables, chairs, cabinets, columns) placed near the outer walls
  gives local planner exposure to diverse obstacle shapes without blocking the path.

  Key design rules:
  1. Furniture is placed within furniture_max_wall_dist of outer walls.
  2. Furniture is excluded from the centre corridor:
       |x - room_center_x| < furniture_clear_zone_x_half
       AND within y-band: (spawn_y, spawn_y + room_length)
  3. Spawn position same as SingleWallTerrainCfg for config compatibility.
  """

  robot_height: float
  """Estimated robot standing height [m]."""

  # ── Outer room dimensions ──────────────────────────────────────────────────
  room_length_range: tuple[float, float] = (12.0, 16.0)
  """Y-extent of the room interior [m], sampled uniformly each episode."""

  room_width_range: tuple[float, float] = (12.0, 16.0)
  """X-extent of the room interior [m], sampled uniformly each episode."""

  outer_wall_height: float = 2.5
  """Height of the four outer walls [m]."""

  outer_wall_thickness: float = 0.15
  """Thickness of the outer walls [m]."""

  # ── Centre obstacle wall ──────────────────────────────────────────────────
  center_wall_width: float = 8.0
  """Width of the horizontal centre obstacle wall [m] (along X axis)."""

  center_wall_width_range: tuple[float, float] | None = None
  """If set, randomise centre wall width uniformly in [min, max]. Overrides
  center_wall_width."""

  center_wall_height: float | None = None
  """Height of the centre obstacle wall [m].  Defaults to outer_wall_height."""

  center_wall_thickness: float = 0.2
  """Thickness of the centre obstacle wall [m]."""

  center_wall_bottom_gap_range: tuple[float, float] | None = None
  """If set, cut a bottom gap while keeping the centre wall top height unchanged."""

  # ── Furniture ─────────────────────────────────────────────────────────────
  num_furniture_range: tuple[int, int] = (4, 8)
  """Min/max number of furniture pieces to place per episode."""

  furniture_min_wall_dist: float = 0.1
  """Minimum distance from furniture surface to outer wall interior face [m]."""

  furniture_max_wall_dist: float = 2.0
  """Maximum distance from furniture centre to outer wall interior face [m]."""

  furniture_clear_zone_x_half: float = 3.0
  """Half-width of the central corridor (|Δx| < this value) where no furniture
  is placed, ensuring the robot's bypass path is unobstructed [m]."""

  robot_distance_from_wall: float = 5.0
  """Distance from robot spawn origin to the centre obstacle wall along +Y [m].
  Must match the corresponding terrain generator setting."""

  friction: tuple[float, float, float] = (1.0, 0.1, 0.01)
  """Ground and wall friction coefficients (sliding, torsional, rolling)."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    tx = float(self.size[0])
    ty = float(self.size[1])
    cx = tx / 2.0
    cy = ty / 2.0
    geometries: list[TerrainGeometry] = []

    wall_color = (0.55, 0.50, 0.45, 1.0)  # warm beige – indoor wall colour
    floor_color = (0.45, 0.45, 0.5, 1.0)
    furniture_color = (0.4, 0.35, 0.3, 1.0)  # dark brown furniture

    # ── Ground plane ──────────────────────────────────────────────────────
    plane = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(tx / 2.0, ty / 2.0, 0.02),
      pos=(cx, cy, -0.02),
      rgba=floor_color,
    )
    plane.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=plane, color=floor_color))

    # ── Room dimensions ───────────────────────────────────────────────────
    room_w = float(rng.uniform(self.room_width_range[0], self.room_width_range[1]))
    room_l = float(rng.uniform(self.room_length_range[0], self.room_length_range[1]))
    # Room is centred on the terrain cell
    rx_min = cx - room_w / 2.0
    rx_max = cx + room_w / 2.0
    ry_min = cy - room_l / 2.0
    ry_max = cy + room_l / 2.0
    owt = float(self.outer_wall_thickness)
    owh = float(self.outer_wall_height)

    def _add_wall(
      pos: tuple[float, float, float],
      size: tuple[float, float, float],
    ) -> None:
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        rgba=wall_color,
      )
      g.contype = 1
      g.conaffinity = 1
      g.friction = np.array(self.friction)
      geometries.append(TerrainGeometry(geom=g, color=wall_color))

    # North wall  (Y = ry_max, extends along X)
    _add_wall(
      pos=(cx, ry_max + owt / 2.0, owh / 2.0),
      size=(room_w / 2.0 + owt, owt / 2.0, owh / 2.0),
    )
    # South wall  (Y = ry_min)
    _add_wall(
      pos=(cx, ry_min - owt / 2.0, owh / 2.0),
      size=(room_w / 2.0 + owt, owt / 2.0, owh / 2.0),
    )
    # East wall   (X = rx_max, extends along Y)
    _add_wall(
      pos=(rx_max + owt / 2.0, cy, owh / 2.0),
      size=(owt / 2.0, room_l / 2.0, owh / 2.0),
    )
    # West wall   (X = rx_min)
    _add_wall(
      pos=(rx_min - owt / 2.0, cy, owh / 2.0),
      size=(owt / 2.0, room_l / 2.0, owh / 2.0),
    )

    # ── Centre obstacle wall (horizontal, along X axis) ────────────────────
    if self.center_wall_width_range is not None:
      actual_cw = float(
        rng.uniform(self.center_wall_width_range[0], self.center_wall_width_range[1])
      )
    else:
      actual_cw = float(self.center_wall_width)
    cwh = float(self.center_wall_height if self.center_wall_height is not None else owh)
    cwt = float(self.center_wall_thickness)
    if self.center_wall_bottom_gap_range is not None:
      cwall_bottom_gap = float(
        rng.uniform(
          self.center_wall_bottom_gap_range[0],
          self.center_wall_bottom_gap_range[1],
        )
      )
    else:
      cwall_bottom_gap = 0.0
    cwall_solid_height = max(cwh - cwall_bottom_gap, 1e-3)
    cwall_half_height = cwall_solid_height / 2.0
    g_cwall = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(actual_cw / 2.0, cwt / 2.0, cwall_half_height),
      pos=(cx, cy, cwall_bottom_gap + cwall_half_height),
      rgba=wall_color,
    )
    g_cwall.contype = 1
    g_cwall.conaffinity = 1
    g_cwall.friction = np.array(self.friction)
    geometries.append(TerrainGeometry(geom=g_cwall, color=wall_color))

    # ── Furniture placement ────────────────────────────────────────────────
    num_furn = int(
      rng.integers(self.num_furniture_range[0], self.num_furniture_range[1] + 1)
    )
    # Spawn is at (cx, cy - robot_distance_from_wall); robot travels in +Y.
    spawn_y = cy - float(self.robot_distance_from_wall)
    clear_x_half = float(self.furniture_clear_zone_x_half)

    # Build a list of candidate (wall_side, distance_from_wall) pairs and
    # choose randomly.  Each attempt tries at most 20 positions.
    max_attempts = 20
    placed = 0
    for _ in range(num_furn * max_attempts):
      if placed >= num_furn:
        break
      # Random wall side: 0=North, 1=South, 2=East, 3=West
      side = int(rng.integers(0, 4))
      d_from_wall = float(
        rng.uniform(self.furniture_min_wall_dist, self.furniture_max_wall_dist)
      )

      if side == 0:  # North wall interior face: Y = ry_max - owt
        fy = ry_max - owt - d_from_wall
        fx = float(rng.uniform(rx_min + owt + 0.5, rx_max - owt - 0.5))
      elif side == 1:  # South wall interior face: Y = ry_min + owt
        fy = ry_min + owt + d_from_wall
        fx = float(rng.uniform(rx_min + owt + 0.5, rx_max - owt - 0.5))
      elif side == 2:  # East wall interior face: X = rx_max - owt
        fx = rx_max - owt - d_from_wall
        fy = float(rng.uniform(ry_min + owt + 0.5, ry_max - owt - 0.5))
      else:  # West wall interior face: X = rx_min + owt
        fx = rx_min + owt + d_from_wall
        fy = float(rng.uniform(ry_min + owt + 0.5, ry_max - owt - 0.5))

      # Reject if inside the centre corridor (|fx - cx| < clear_x_half)
      if abs(fx - cx) < clear_x_half:
        continue

      # Place a random furniture piece
      furn_kind = int(rng.integers(0, 4))  # 0=desk, 1=chair, 2=cabinet, 3=column

      if furn_kind == 0:  # desk: Box ~ 1.0 × 0.6 × 0.7
        hw = float(rng.uniform(0.25, 0.60))
        hd = float(rng.uniform(0.15, 0.35))
        ht = 0.35  # half-height of 0.7m desk
        if rng.random() > 0.5:
          hw, hd = hd, hw  # random rotation 90°
        g = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(hw, hd, ht),
          pos=(fx, fy, ht),
          rgba=furniture_color,
        )
      elif furn_kind == 1:  # chair: Box ~ 0.45 × 0.45 × 0.45
        hs = float(rng.uniform(0.18, 0.25))
        ht = 0.225  # half-height of 0.45m chair
        g = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(hs, hs, ht),
          pos=(fx, fy, ht),
          rgba=furniture_color,
        )
      elif furn_kind == 2:  # cabinet: Box ~ 0.6 × 0.4 × 1.2
        hw = float(rng.uniform(0.20, 0.40))
        hd = float(rng.uniform(0.15, 0.30))
        ht = float(rng.uniform(0.35, 0.75))  # half-height 0.7–1.5m
        if rng.random() > 0.5:
          hw, hd = hd, hw
        g = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=(hw, hd, ht),
          pos=(fx, fy, ht),
          rgba=furniture_color,
        )
      else:  # column/pillar: Cylinder r=0.2, h=0.8–1.2
        r = float(rng.uniform(0.12, 0.25))
        ht = float(rng.uniform(0.40, 0.60))
        g = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_CYLINDER,
          size=(r, ht),
          pos=(fx, fy, ht),
          rgba=furniture_color,
        )

      g.contype = 1
      g.conaffinity = 1
      g.friction = np.array(self.friction)
      geometries.append(TerrainGeometry(geom=g, color=furniture_color))
      placed += 1

    # ── Spawn origin ──────────────────────────────────────────────────────
    origin = np.array((cx, spawn_y, 0.0))
    return TerrainOutput(origin=origin, geometries=geometries)


def _make_mixed_scenario_generator_cfg(
  robot_height: float, play: bool
) -> TerrainGeneratorCfg:
  """Create terrain generator for mixed-scenario training (40/30/30 split).

  Three scenario types mixed by proportion:
  - 40% SingleWall (fixed 8m width): standard right-bypass avoidance
  - 30% OpenField (no walls): pure goal-directed navigation
  - 30% SingleWall (random 2-8m width): generalized wall avoidance

  Uses a 5x2 grid (10 terrain cells) so that with 2048 envs, each cell
  gets ~205 envs with diverse terrain types.

  Terrain cell size increased to 30×30m (from 20×20m) and
  robot_distance_from_wall increased from 4m to 6m to give the robot
  enough room to execute wide-radius turns at higher speeds (up to 3 m/s).

  For the open-field cells, the FixedSequentialGoalCommand still defines
  waypoints, but the dynamic target blending in train_local_planner.py
  (alpha = sigmoid((min_ray - threshold) * scale)) will automatically
  ignore waypoints because min_ray is large → alpha ≈ 1 → target = key_goal.
  """
  return TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=2.0,
    num_rows=5,
    num_cols=2,
    curriculum=False,
    sub_terrains={
      # 30%: Fixed 8m wide wall (standard right-bypass obstacle avoidance)
      "single_wall_fixed": SingleWallTerrainCfg(
        proportion=0.30,
        robot_height=robot_height,
        wall_height=max(2.0, robot_height + 0.5),
        wall_thickness=0.2,
        wall_width=8.0,
        robot_distance_from_wall=6.0,
      ),
      # 30%: Open field — no obstacles, pure goal navigation
      "open_field": OpenFieldTerrainCfg(
        proportion=0.3,
        robot_height=robot_height,
        robot_distance_from_wall=6.0,
      ),
      # 25%: Random wall width 2-8m for generalization
      "single_wall_random": SingleWallTerrainCfg(
        proportion=0.25,
        robot_height=robot_height,
        wall_height=max(2.0, robot_height + 0.5),
        wall_thickness=0.2,
        wall_width=5.0,  # Fallback (overridden by wall_width_range)
        wall_width_range=(2.0, 8.0),
        robot_distance_from_wall=6.0,
      ),
      # 15%: Suspended wall, matching real cases with empty space near the floor
      "single_wall_suspended": SingleWallTerrainCfg(
        proportion=0.15,
        robot_height=robot_height,
        wall_height=max(2.0, robot_height + 0.5),
        wall_thickness=0.2,
        wall_width=8.0,
        wall_bottom_gap_range=(0.20, 0.45),
        robot_distance_from_wall=6.0,
      ),
    },
    difficulty_range=(0.0, 1.0),
    add_lights=True,
  )


def _make_short_corridor_maze_generator_cfg(
  robot_height: float, play: bool, curriculum_step_threshold: int = 2000
) -> TerrainGeneratorCfg:
  """Create terrain generator for short-corridor maze with many openings.

  This maze is designed for RIGHT-SIDE BYPASS navigation:
  - Maintains proper maze structure (connected corridors)
  - MORE maze cells to fill the available space
  - MORE OPENINGS (higher extra_openings_ratio = more escape routes)
  - Wide corridors (4m+) for comfortable robot navigation
  - LARGE central spawn area with cleared surrounding cells

  Key differences from standard true_maze:
  - maze_cells: 10x10 to fill available space
  - extra_openings_ratio: 0.55 instead of 0.35 (more openings/exits)
  - spawn_clear_cells: 2 (larger clear area = 5x5 cells = ~21m diameter)
  - narrow_margin/wide_margin: 1.0/2.0 (smaller margins to maximize maze area)

  This ensures robot can always find a relatively short detour path,
  especially when bypassing obstacles from the right side.

  Curriculum is based on difficulty:
  - Lower difficulty: Even more walls removed (very open maze)
  - Higher difficulty: Standard openings (proper maze with many exits)
  """
  grid_rows = 3
  grid_cols = 8 if not play else 4

  return TerrainGeneratorCfg(
    size=(50.0, 50.0),  # Medium terrain size
    border_width=12.0 if play else 8.0,
    num_rows=grid_rows,
    num_cols=grid_cols,
    curriculum=True,
    sub_terrains={
      "short_corridor_maze": TrueMazeTerrainCfg(
        proportion=1.0,
        robot_height=robot_height,
        wall_height=max(2.0, robot_height + 0.5),
        wall_thickness=0.2,
        boundary_margin=0.5,
        corridor_width=4.0,  # Wide corridors for robot
        maze_cells_x=10,  # Fill the available space
        maze_cells_y=10,
        extra_openings_ratio=0.55,  # MORE openings = more exit options
        spawn_clear_cells=2,  # Clear area around spawn (5x5 cells = ~21m)
        curriculum_step_threshold=curriculum_step_threshold,
        # Smaller margins to maximize maze area
        narrow_margin=1.0,  # Minimal margin (left and bottom sides)
        wide_margin=2.0,  # Small margin (right and top sides)
      )
    },
    difficulty_range=(0.0, 1.0),
    add_lights=True,
  )


def _make_true_maze_generator_cfg(
  robot_height: float, play: bool, curriculum_step_threshold: int = 2000
) -> TerrainGeneratorCfg:
  """Create terrain generator for true maze environment.

  The maze uses recursive backtracking algorithm to generate proper maze structure:
  - Connected corridors forming a solvable maze
  - Wide passages (4m+) for comfortable robot navigation
  - Extra openings create loops for multiple path options
  - Asymmetric margins: narrow on left/bottom, wide on right/top
  - Central spawn area with cleared surrounding cells

  Curriculum is based on difficulty:
  - Lower difficulty: More walls removed (easier exploration)
  - Higher difficulty: Fewer walls removed (proper maze challenge)
  """
  grid_rows = 3
  grid_cols = 8 if not play else 4

  return TerrainGeneratorCfg(
    size=(60.0, 60.0),  # Large terrain for proper maze
    border_width=15.0 if play else 10.0,
    num_rows=grid_rows,
    num_cols=grid_cols,
    curriculum=True,
    sub_terrains={
      "true_maze": TrueMazeTerrainCfg(
        proportion=1.0,
        robot_height=robot_height,
        wall_height=max(2.0, robot_height + 0.5),
        wall_thickness=0.2,
        boundary_margin=0.5,
        corridor_width=4.0,  # Wide corridors for robot
        maze_cells_x=10,  # Larger maze grid to fill more space
        maze_cells_y=10,
        extra_openings_ratio=0.35,  # Create loops for multiple paths
        spawn_clear_cells=1,  # Clear cells around spawn
        curriculum_step_threshold=curriculum_step_threshold,
        # Asymmetric margins: narrow on left/bottom, wide on right/top
        narrow_margin=2.0,  # Narrow margin (left and bottom sides)
        wide_margin=6.0,  # Wide margin (right and top sides)
      )
    },
    difficulty_range=(0.0, 1.0),
    add_lights=True,
  )


@dataclass(kw_only=True)
class ObstacleFieldTerrainCfg(SubTerrainCfg):
  """Flat patch populated with mixed-shape obstacles.

  Curriculum stages based on difficulty:
    - Stage 1 (difficulty < 0.33): All obstacles at robot height
    - Stage 2 (0.33 <= difficulty < 0.66): 1/2 at robot height, 1/2 at 0.5x
    - Stage 3 (difficulty >= 0.66): 1/3 at robot height, 1/3 at 0.5x, 1/3 at 0.25x

  Layout: ~1/3 of area is clear space, obstacles are sparse with safe spawn zone.
  """

  robot_height: float
  min_obstacles: int = 6
  max_obstacles: int = 10
  radius_range: tuple[float, float] = (0.15, 0.35)
  height_jitter: float = 0.03
  friction: tuple[float, float, float] = (1.0, 0.1, 0.01)
  # Minimum gap between obstacles for robot passage (robot width ~0.4m + margin)
  min_gap: float = 0.6
  # Safe spawn radius around origin where no obstacles are placed
  spawn_safe_radius: float = 1.2
  # Lower bound of obstacle height as a fraction of robot height.
  min_height_scale: float = 0.25

  def _height_scales(
    self, num_obstacles: int, difficulty: float, rng: np.random.Generator
  ) -> list[float]:
    """Compute height scales based on curriculum stage (difficulty)."""
    if difficulty < 1.0 / 3.0:
      # Stage 1: all obstacles at full robot height
      scales = [1.0] * num_obstacles
    elif difficulty < 2.0 / 3.0:
      # Stage 2: half at full height, half at 0.5x
      num_half_height = num_obstacles // 2
      num_full_height = num_obstacles - num_half_height
      scales = [1.0] * num_full_height + [0.5] * num_half_height
    else:
      # Stage 3: 1/3 at full, 1/3 at 0.5x, 1/3 at 0.25x
      num_each = max(1, num_obstacles // 3)
      num_full = num_each
      num_half = num_each
      num_quarter = num_obstacles - num_full - num_half
      scales = [1.0] * num_full + [0.5] * num_half + [0.25] * num_quarter
    rng.shuffle(scales)
    return scales

  def _sample_position(
    self,
    radius: float,
    rng: np.random.Generator,
    placed: list[tuple[float, float, float]],
    origin: tuple[float, float],
    obstacle_zone_x_start: float,
  ) -> tuple[float, float] | None:
    """Sample obstacle position ensuring sparse placement and safe spawn."""
    for _ in range(30):
      # Place obstacles only in the obstacle zone (right 2/3 of terrain)
      x = float(
        rng.uniform(obstacle_zone_x_start + radius + 0.2, self.size[0] - radius - 0.2)
      )
      y = float(rng.uniform(radius + 0.2, self.size[1] - radius - 0.2))

      # Check distance from spawn origin
      dist_to_origin = np.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
      if dist_to_origin < self.spawn_safe_radius + radius:
        continue

      # Check distance from all placed obstacles (ensure passage gap)
      too_close = False
      for px, py, pr in placed:
        dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
        if dist < radius + pr + self.min_gap:
          too_close = True
          break
      if not too_close:
        return x, y
    return None

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    # Spawn origin at center of clear zone (left 1/3)
    clear_zone_width = self.size[0] / 3.0
    origin = np.array((clear_zone_width / 2.0, self.size[1] / 2.0, 0.0))

    # Ground plane
    plane = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, 0.02),
      pos=(self.size[0] / 2.0, self.size[1] / 2.0, -0.02),
      rgba=(0.45, 0.45, 0.5, 1.0),
    )
    plane.friction = np.array(self.friction)
    geometries = [TerrainGeometry(geom=plane, color=(0.45, 0.45, 0.5, 1.0))]

    # Obstacle zone starts after clear zone
    obstacle_zone_x_start = clear_zone_width

    num_obstacles = int(rng.integers(self.min_obstacles, self.max_obstacles + 1))
    height_scales = self._height_scales(num_obstacles, difficulty, rng)

    shapes = (
      mujoco.mjtGeom.mjGEOM_CYLINDER,
      mujoco.mjtGeom.mjGEOM_SPHERE,
      mujoco.mjtGeom.mjGEOM_BOX,
      mujoco.mjtGeom.mjGEOM_CAPSULE,
      mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    )

    placed: list[tuple[float, float, float]] = []
    for scale in height_scales:
      scale = max(scale, self.min_height_scale)
      target_height = max(0.15, self.robot_height * scale)
      target_height += rng.uniform(-self.height_jitter, self.height_jitter)
      target_height = max(0.12, target_height)

      radius = float(rng.uniform(*self.radius_range))
      pos = self._sample_position(
        radius, rng, placed, (origin[0], origin[1]), obstacle_zone_x_start
      )
      if pos is None:
        continue  # Skip if cannot find valid position
      x, y = pos
      placed.append((x, y, radius))

      geom_type = shapes[int(rng.integers(0, len(shapes)))]
      if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = max(radius, target_height / 2.0)
        geom = body.add_geom(
          type=geom_type,
          size=(r, 0.0, 0.0),
          pos=(x, y, r),
        )
      elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        half_h = target_height / 2.0
        geom = body.add_geom(
          type=geom_type,
          size=(radius, radius, half_h),
          pos=(x, y, half_h),
        )
      elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        sx = radius * float(rng.uniform(0.9, 1.3))
        sy = radius * float(rng.uniform(0.9, 1.3))
        half_h = target_height / 2.0
        geom = body.add_geom(
          type=geom_type,
          size=(sx, sy, half_h),
          pos=(x, y, half_h),
        )
      elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half_length = max(target_height / 2.0 - radius, 0.02)
        geom = body.add_geom(
          type=geom_type,
          size=(radius, half_length, 0.0),
          pos=(x, y, radius + half_length),
        )
      else:  # Ellipsoid
        half_h = target_height / 2.0
        geom = body.add_geom(
          type=geom_type,
          size=(radius, radius * 0.8, half_h),
          pos=(x, y, half_h),
        )

      geom.contype = 1
      geom.conaffinity = 1
      geom.friction = np.array(self.friction)
      color = rng.uniform(0.25, 0.85, size=3)
      geom.rgba = np.array([color[0], color[1], color[2], 1.0], dtype=np.float32)
      geometries.append(TerrainGeometry(geom=geom, color=tuple(geom.rgba)))

    return TerrainOutput(origin=origin, geometries=geometries)


def _make_obstacle_with_mild_rough_generator_cfg(
  robot_height: float, play: bool
) -> TerrainGeneratorCfg:
  """Create mixed terrain: obstacles (70%) + mild rough heightfield (30%).

  Combines the obstacle families from ComplexGeneralization with lightweight
  heightfield-based rough terrain at very low difficulty. The rough tiles use
  only heightfields (no extra geom bodies), so they do NOT increase nconmax
  pressure — in fact they *reduce* average contact load compared to pure
  obstacle terrain.

  Rough difficulty is capped at 0.16 to represent near-flat real-world ground:
  - Stair step_height ≤ 1.6 cm
  - Slope factor ≤ 0.16
  - Surface noise 0.02–0.10 m (independent of difficulty)
  """
  grid_rows = 4
  grid_cols = 8 if not play else 4

  # --- 70% obstacle tiles (same families as ComplexGeneralization) ---
  obstacle_scale = 0.70
  obstacle_sub_terrains: dict[str, SubTerrainCfg] = {
    "thin_pillar_field": UniformPillarFieldTerrainCfg(
      proportion=0.40 * obstacle_scale,
      robot_height=robot_height,
      wall_height=max(2.0, robot_height + 0.5),
      wall_height_range=(
        (4.0 / 7.0) * robot_height,
        max(2.0, robot_height + 0.5),
      ),
      wall_thickness=0.08,
      pillar_length=0.9,
      num_pillars=60,
      min_pillar_spacing=2.2,
      boundary_margin=1.0,
    ),
    "irregular_obstacle_field": ObstacleFieldTerrainCfg(
      proportion=0.60 * obstacle_scale,
      robot_height=robot_height,
      min_obstacles=24,
      max_obstacles=42,
      radius_range=(0.06, 0.30),
      height_jitter=0.04,
      min_gap=0.45,
      spawn_safe_radius=1.2,
      min_height_scale=(2.0 / 3.0),
    ),
  }

  # --- 30% mild rough tiles (heightfield only, no geom bodies) ---
  rough_scale = 0.30
  rough_sub_terrains: dict[str, SubTerrainCfg] = {
    "mild_flat": BoxFlatTerrainCfg(
      proportion=0.20 * rough_scale,
    ),
    "mild_pyramid_stairs": BoxPyramidStairsTerrainCfg(
      proportion=0.20 * rough_scale,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "mild_pyramid_stairs_inv": BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.20 * rough_scale,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "mild_hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
      proportion=0.10 * rough_scale,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
    ),
    "mild_random_rough": HfRandomUniformTerrainCfg(
      proportion=0.15 * rough_scale,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "mild_wave_terrain": HfWaveTerrainCfg(
      proportion=0.15 * rough_scale,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
  }

  return TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=12.0 if play else 8.0,
    num_rows=grid_rows,
    num_cols=grid_cols,
    curriculum=True,
    sub_terrains={**obstacle_sub_terrains, **rough_sub_terrains},
    difficulty_range=(0.0, 0.16),
    add_lights=True,
  )


def _make_obstacle_flat_generator_cfg(
  robot_height: float, play: bool
) -> TerrainGeneratorCfg:
  """Create flat terrain with obstacles only (no rough heightfield tiles).

  Same obstacle families as ``_make_obstacle_with_mild_rough_generator_cfg``
  but the 30% mild-rough tiles are replaced with flat tiles, so the robot
  trains purely on flat ground with obstacle avoidance.
  """
  grid_rows = 4
  grid_cols = 8 if not play else 4

  sub_terrains: dict[str, SubTerrainCfg] = {
    "thin_pillar_field": UniformPillarFieldTerrainCfg(
      proportion=0.40,
      robot_height=robot_height,
      wall_height=max(2.0, robot_height + 0.5),
      wall_height_range=(
        (4.0 / 7.0) * robot_height,
        max(2.0, robot_height + 0.5),
      ),
      wall_thickness=0.08,
      pillar_length=0.9,
      num_pillars=60,
      min_pillar_spacing=2.2,
      boundary_margin=1.0,
    ),
    "irregular_obstacle_field": ObstacleFieldTerrainCfg(
      proportion=0.60,
      robot_height=robot_height,
      min_obstacles=24,
      max_obstacles=42,
      radius_range=(0.06, 0.30),
      height_jitter=0.04,
      min_gap=0.45,
      spawn_safe_radius=1.2,
      min_height_scale=(2.0 / 3.0),
    ),
  }

  return TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=12.0 if play else 8.0,
    num_rows=grid_rows,
    num_cols=grid_cols,
    curriculum=True,
    sub_terrains=sub_terrains,
    difficulty_range=(0.0, 0.16),
    add_lights=True,
  )


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_active_g1_n3_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Body collision sensor: detects collisions of body parts (thigh, shin, hip, pelvis)
  # excluding feet. Used for ABS-style collision penalty.
  body_collision_cfg = ContactSensorCfg(
    name="body_collision",
    primary=ContactMatch(
      mode="body",
      pattern=(
        r"pelvis",
        r".*_hip_.*_link",
        r".*_knee_link",  # knee/shin bodies
        r"torso_link",
        r"waist_.*_link",
      ),
      entity="robot",
    ),
    secondary=None,  # Any contact (ground, obstacles, etc.)
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  # Feet horizontal collision sensor: detects feet kicking obstacles/walls
  # Used for ABS-style feet_collision penalty.
  feet_collision_cfg = ContactSensorCfg(
    name="feet_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=None,  # Any contact
    fields=("force",),
    reduce="netforce",
    num_slots=1,
  )
  # VL53L8CX reports a measured detection volume of 45° horizontal/vertical
  # and a 65° diagonal square FoV.  Those are optical/algorithmic detection
  # volumes, not an exact pinhole identity; for this pinhole ray model we match
  # the headline 65° diagonal and convert it to a square horizontal/vertical FoV.
  # Set MJLAB_VL53L8CX_SQUARE_FOVY_DEG=45.0 to train with the datasheet
  # horizontal/vertical detection-volume angle instead.
  square_fovy_deg = _vl53l8cx_square_fovy_deg()
  raycast_update_period_s = _VL53L8CX_UPDATE_PERIOD_S
  raycast_cfg = RayCastSensorCfg(
    name="raycast",
    frame=ObjRef(type="site", name="forward_camera", entity="robot"),
    pattern=PinholeCameraPatternCfg(
      width=_VL53L8CX_SENSOR_WIDTH,
      height=_VL53L8CX_SENSOR_HEIGHT,
      fovy=square_fovy_deg,
    ),
    ray_alignment="base",
    max_distance=_VL53L8CX_MAX_DISTANCE_M,
    update_period=raycast_update_period_s,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  # Left-side raycast sensor: same 8x8 pinhole pattern, mounted on neck
  # (left_camera site on torso_link body, neck left side, facing outward +Y — above shoulders, avoids arm occlusion)
  raycast_left_cfg = RayCastSensorCfg(
    name="raycast_left",
    frame=ObjRef(type="site", name="left_camera", entity="robot"),
    pattern=PinholeCameraPatternCfg(
      width=_VL53L8CX_SENSOR_WIDTH,
      height=_VL53L8CX_SENSOR_HEIGHT,
      fovy=square_fovy_deg,
    ),
    ray_alignment="base",
    max_distance=_VL53L8CX_MAX_DISTANCE_M,
    update_period=raycast_update_period_s,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  # Right-side raycast sensor: same 8x8 pinhole pattern, mounted on neck
  # (right_camera site on torso_link body, neck right side, facing outward -Y — above shoulders, avoids arm occlusion)
  raycast_right_cfg = RayCastSensorCfg(
    name="raycast_right",
    frame=ObjRef(type="site", name="right_camera", entity="robot"),
    pattern=PinholeCameraPatternCfg(
      width=_VL53L8CX_SENSOR_WIDTH,
      height=_VL53L8CX_SENSOR_HEIGHT,
      fovy=square_fovy_deg,
    ),
    ray_alignment="base",
    max_distance=_VL53L8CX_MAX_DISTANCE_M,
    update_period=raycast_update_period_s,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  # Critic terrain scan sensor for privileged terrain height observations.
  # Asymmetric grid: front 4m, back 0.2m, left/right 0.5m
  # Size: X = 4.0 + 0.2 = 4.2m, Y = 0.5 + 0.5 = 1.0m
  # Offset: X = (4.0 - 0.2) / 2 = 1.9m forward (center shifted forward)
  # With 0.2m resolution: X has 22 points, Y has 6 points -> 132 total points
  critic_terrain_scan_cfg = RayCastSensorCfg(
    name="critic_terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=GridPatternCfg(
      size=(4.2, 1.0),  # Front 4m + back 0.2m, left/right 0.5m each
      resolution=0.2,  # 0.2m spacing
      direction=(0.0, 0.0, -1.0),  # Pointing downward
      offset=(1.9, 0.0),  # Shift center 1.9m forward
    ),
    ray_alignment="yaw",  # Stay level on slopes (only yaw follows robot)
    max_distance=3.0,  # Max height measurement range
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Only terrain geoms
  )
  if play:
    # Enable debug rendering of the raycast grids only during play.
    for _rc_cfg in (raycast_cfg, raycast_left_cfg, raycast_right_cfg):
      _rc_cfg.debug_vis = True
      _rc_cfg.viz = RayCastSensorCfg.VizCfg(
        show_rays=True,
        show_normals=False,
        hit_color=(0.1, 0.8, 0.2, 0.9),
        miss_color=(0.8, 0.2, 0.2, 0.5),
      )
  cfg.scene.sensors = (
    feet_ground_cfg,
    self_collision_cfg,
    body_collision_cfg,
    feet_collision_cfg,
    raycast_cfg,
    raycast_left_cfg,
    raycast_right_cfg,
    critic_terrain_scan_cfg,
  )

  # Remove inherited height_scan from both actor and critic (from base terrain_scan).
  # Critic gets terrain via our own critic_terrain_scan with normalized observations.
  for group_name in ("actor", "critic"):
    group = cfg.observations.get(group_name)
    if group is not None and "height_scan" in group.terms:
      del group.terms["height_scan"]

  # Add terrain_heights to critic using the normalized observation function.
  cfg.observations["critic"].terms["terrain_heights"] = ObservationTermCfg(
    func=mdp.height_scan_observations,
    params={
      "sensor_name": critic_terrain_scan_cfg.name,
      "base_height": 0.9,
    },
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = get_active_g1_action_scale()

  cfg.viewer.body_name = "torso_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Increase COM randomization to match others' magnitudes (±0.05 all axes).
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.05, 0.05),
    1: (-0.05, 0.05),
    2: (-0.05, 0.05),
  }

  # DR: slightly more than g1/ velocity (±0.15) to add robustness for obstacle tasks,
  # but well below the original ±0.5 that caused catastrophic init poses.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.2, 0.2)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.2, 0.2)

  # Raycast camera site position DR (±5cm per axis).
  cfg.events["camera_pos_forward"] = EventTermCfg(
    func=dr.site_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("forward_camera",)),
      "operation": "add",
      "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
    },
  )
  cfg.events["camera_pos_left"] = EventTermCfg(
    func=dr.site_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_camera",)),
      "operation": "add",
      "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
    },
  )
  cfg.events["camera_pos_right"] = EventTermCfg(
    func=dr.site_pos,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("right_camera",)),
      "operation": "add",
      "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
    },
  )

  # Raycast camera orientation DR.  Keep all three sensors randomized, but
  # tighten the floor-facing tilt into a mild upward-biased jitter so open black
  # floors are often invisible while vertical obstacles remain in view.
  cfg.events["camera_orient_forward"] = EventTermCfg(
    func=dr.site_quat,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("forward_camera",)),
      "roll_range": _RAYCAST_CAMERA_ORIENT_RANGE_RAD,
      "pitch_range": _RAYCAST_FORWARD_UPWARD_PITCH_RANGE_RAD,
      "yaw_range": _RAYCAST_CAMERA_ORIENT_RANGE_RAD,
    },
  )
  cfg.events["camera_orient_left"] = EventTermCfg(
    func=dr.site_quat,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_camera",)),
      "roll_range": _RAYCAST_LEFT_UPWARD_ROLL_RANGE_RAD,
      "pitch_range": _RAYCAST_CAMERA_ORIENT_RANGE_RAD,
      "yaw_range": _RAYCAST_CAMERA_ORIENT_RANGE_RAD,
    },
  )
  cfg.events["camera_orient_right"] = EventTermCfg(
    func=dr.site_quat,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=("right_camera",)),
      "roll_range": _RAYCAST_RIGHT_UPWARD_ROLL_RANGE_RAD,
      "pitch_range": _RAYCAST_CAMERA_ORIENT_RANGE_RAD,
      "yaw_range": _RAYCAST_CAMERA_ORIENT_RANGE_RAD,
    },
  )

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Body collision penalty: penalizes collisions of body parts (thigh, shin, hip, torso)
  # excluding feet. Based on ABS training reward design.
  cfg.rewards["body_collision"] = RewardTermCfg(
    func=mdp.body_collision_penalty,
    weight=-1.0,
    params={"sensor_name": body_collision_cfg.name},
  )

  # Feet collision penalty: penalizes feet kicking obstacles/walls horizontally.
  # Triggers when horizontal force > 2 * vertical force + 10.0.
  # Based on ABS training reward design.
  cfg.rewards["feet_collision"] = RewardTermCfg(
    func=mdp.feet_collision_penalty,
    weight=-0.5,
    params={"sensor_name": feet_collision_cfg.name},
  )

  # Create raycast observation as a separate observation group with 3-frame history.
  # This allows the ActorCriticRaycast policy to process it through a temporal encoder
  # instead of directly concatenating raw values.
  raycast_history_length = 3
  raycast_term = ObservationTermCfg(
    func=mdp.raycast_distances,
    params=_vl53l8cx_raycast_observation_params(raycast_cfg.name),
    history_length=raycast_history_length,
    flatten_history_dim=True,  # [B, T * D] for encoder input
  )

  # Add raycast as a separate observation group (not concatenated with policy/critic)
  cfg.observations["raycast"] = ObservationGroupCfg(
    terms={"raycast": raycast_term},
    concatenate_terms=True,
    enable_corruption=False,  # Sensor model noise is handled by term params.
  )

  # Left raycast observation group (same config as forward raycast)
  raycast_left_term = ObservationTermCfg(
    func=mdp.raycast_distances,
    params=_vl53l8cx_raycast_observation_params(raycast_left_cfg.name),
    history_length=raycast_history_length,
    flatten_history_dim=True,
  )
  cfg.observations["raycast_left"] = ObservationGroupCfg(
    terms={"raycast_left": raycast_left_term},
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Right raycast observation group (same config as forward raycast)
  raycast_right_term = ObservationTermCfg(
    func=mdp.raycast_distances,
    params=_vl53l8cx_raycast_observation_params(raycast_right_cfg.name),
    history_length=raycast_history_length,
    flatten_history_dim=True,
  )
  cfg.observations["raycast_right"] = ObservationGroupCfg(
    terms={"raycast_right": raycast_right_term},
    concatenate_terms=True,
    enable_corruption=False,
  )

  # Note: raycast/raycast_left/raycast_right are NOT added to policy/critic groups
  # since ActorCriticRaycast will handle them separately through CNN encoders

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    # Keep actor observation noise enabled to match real-robot sensor noise
    # (motor encoder dq() noise, IMU noise, etc.).
    # enable_corruption=True preserves the per-term noise defined in the
    # actor observation terms (e.g. joint_vel Unoise(-1.5, 1.5)).
    cfg.observations["actor"].enable_corruption = True
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain velocity configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def unitree_g1_circular_goal_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 circular goal tracking on flat terrain.

  This configuration uses a STABLE goal tracking approach:
  - Goal is at fixed radius from robot (circular goal)
  - Goal ANGLE is in WORLD frame (not robot frame)
  - Goal angle resamples smoothly every 3-8 seconds (like velocity commands)
  - Reward is based on angular velocity command (not direct heading error)

  This eliminates the reward oscillation problem of waypoint-based goal tracking
  by providing the same smooth control signal as velocity tracking.

  Key insight: velocity tracking's stability comes from tracking ANGULAR VELOCITY
  commands (a smooth control signal), not from direct heading error.
  """
  from mjlab.tasks.velocity.mdp.circular_goal_command import (
    CircularGoalTrackingCommandCfg,
  )

  # Start from flat velocity configuration
  cfg = unitree_g1_flat_env_cfg(play=play)

  # Replace velocity command with circular goal tracking command
  assert cfg.commands is not None
  cfg.commands = {
    "goal": CircularGoalTrackingCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),  # Same as velocity tracking
      goal_radius=3.0,  # Fixed distance to goal
      goal_angle_range=math.pi / 6,  # Initial: ±30° (front only)
      target_velocity=0.5,  # Initial velocity
      target_velocity_range=(0.0, 1.0),  # Initial curriculum stage
      debug_vis=True,
    )
  }

  # Update observations for circular goal tracking.
  # Rebuild actor terms in deploy-compatible order:
  #   base_ang_vel → projected_gravity → goal_command → joint_pos → joint_vel
  #   → actions → ball_state
  # (deploy.yaml lists observations in this exact concatenation order)
  policy_terms = cfg.observations["actor"].terms
  new_actor_terms = {
    "base_ang_vel": policy_terms["base_ang_vel"],
    "projected_gravity": policy_terms["projected_gravity"],
    "goal_command": ObservationTermCfg(
      func=mdp.circular_goal_command,
      params={"command_name": "goal"},
    ),
    "joint_pos": policy_terms["joint_pos"],
    "joint_vel": policy_terms["joint_vel"],
    "actions": policy_terms["actions"],
    "ball_state": ObservationTermCfg(
      func=mdp.ball_observation_placeholder,
      params={},
    ),
  }
  policy_terms.clear()
  policy_terms.update(new_actor_terms)

  # Rebuild critic terms in the same base order, keeping privileged extras at the end.
  critic_terms = cfg.observations["critic"].terms
  # Collect privileged critic-only terms (everything not in base set).
  _base_keys = {
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "terrain_heights",
  }
  privileged = {k: v for k, v in critic_terms.items() if k not in _base_keys}
  new_critic_terms = {
    "base_lin_vel": critic_terms["base_lin_vel"],
    "base_ang_vel": critic_terms["base_ang_vel"],
    "projected_gravity": critic_terms["projected_gravity"],
    "goal_command": ObservationTermCfg(
      func=mdp.circular_goal_command,
      params={"command_name": "goal"},
    ),
    "joint_pos": critic_terms["joint_pos"],
    "joint_vel": critic_terms["joint_vel"],
    "actions": critic_terms["actions"],
    "ball_state": ObservationTermCfg(
      func=mdp.ball_observation_placeholder,
      params={},
    ),
    "terrain_heights": critic_terms["terrain_heights"],
    **privileged,
  }
  critic_terms.clear()
  critic_terms.update(new_critic_terms)

  # Replace velocity tracking rewards with circular goal tracking rewards
  assert cfg.rewards is not None

  # Remove velocity tracking rewards
  del cfg.rewards["track_linear_velocity"]
  del cfg.rewards["track_angular_velocity"]

  # Add circular goal tracking rewards
  # KEY REWARD: Track angular velocity command derived from heading error
  # This is the SAME mechanism as velocity tracking, providing smooth gradients
  cfg.rewards["track_ang_vel_cmd"] = RewardTermCfg(
    func=mdp.circular_goal_ang_vel_tracking,
    weight=1.3,  # Same as track_angular_velocity
    params={
      "command_name": "goal",
      "std": math.sqrt(0.5),
      "heading_control_stiffness": 0.5,  # Moderate, like velocity tracking
      "max_ang_vel": 1.0,
    },
  )

  # Track forward velocity towards goal
  cfg.rewards["track_goal_velocity"] = RewardTermCfg(
    func=mdp.circular_goal_tracking_velocity,
    weight=2.0,  # Same as track_linear_velocity
    params={"command_name": "goal", "std": math.sqrt(0.25)},
  )

  # Track z-axis linear velocity (should be zero - no vertical motion)
  cfg.rewards["track_linear_velocity_z"] = RewardTermCfg(
    func=mdp.track_linear_velocity_z,
    weight=2.2,
    params={"std": math.sqrt(0.25)},
  )

  # Keep xy angular velocity penalty for stability
  cfg.rewards["track_ang_vel_xy"] = RewardTermCfg(
    func=mdp.track_angular_velocity_xy,
    weight=2.0,
    params={"std": math.sqrt(0.5)},
  )

  # Add velocity tracking metrics logging (zero-weight reward for WandB logging)
  # This logs: actual velocity, target velocity, velocity error, heading error
  cfg.rewards["velocity_tracking_log"] = RewardTermCfg(
    func=mdp.circular_goal_velocity_tracking_log,
    weight=0.0,  # Zero weight - purely for logging
    params={"command_name": "goal"},
  )

  # NOTE: track_ang_vel_z is NOT needed because:
  # 1. Goal is always in front of robot (within goal_angle_range)
  # 2. track_ang_vel_cmd already limits angular velocity via max_ang_vel
  # 3. Adding track_ang_vel_z would conflict with track_ang_vel_cmd

  # Update pose reward to use target_velocity for speed thresholding
  cfg.rewards["pose"] = RewardTermCfg(
    func=mdp.variable_posture,
    weight=1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "command_name": "goal",
      "std_standing": {".*": 0.05},
      "std_walking": {
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.2,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.1,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
      },
      "std_running": {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        r".*waist_yaw.*": 0.3,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.2,
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
      },
      "walking_threshold": 0.05,
      "running_threshold": 1.5,
    },
  )

  # Update other rewards that reference command
  for reward_name in [
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ]:
    if reward_name in cfg.rewards:
      reward_cfg = cfg.rewards[reward_name]
      if "command_name" in reward_cfg.params:
        reward_cfg.params["command_name"] = "goal"

  if "air_time" in cfg.rewards:
    cfg.rewards["air_time"].params["command_name"] = "goal"

  # Update curriculum for goal velocity and angle range
  assert cfg.curriculum is not None

  # Remove velocity curriculum if present
  if "command_vel" in cfg.curriculum:
    del cfg.curriculum["command_vel"]

  # Add goal velocity and angle curriculum
  # Curriculum stages:
  #   0-250 iters: vel [0,1], angle ±30° (easy: goal always nearly in front)
  #   250-500 iters: vel [0,1.5], angle ±60° (medium: wider angle range)
  #   500+ iters: vel [0,3], angle ±90° (hard: goal can be to the side)
  cfg.curriculum["goal_velocity"] = CurriculumTermCfg(
    func=mdp.circular_goal_velocity,
    params={
      "command_name": "goal",
      "velocity_stages": [
        {
          "step": 0,
          "target_velocity_range": (0.0, 1.0),
          "goal_angle_range": math.pi / 6,  # ±30°
        },
        {
          "step": 1750 * 24,
          "target_velocity_range": (0.0, 2.0),
          "goal_angle_range": math.pi / 3,  # ±60°
        },
        {
          "step": 3500 * 24,
          "target_velocity_range": (0.0, 3.0),
          "goal_angle_range": math.pi / 2,  # ±90°
        },
      ],
    },
  )

  # Add angular velocity tracking reward weight curriculum
  # As goal angle range increases, we need stronger rotation tracking
  # Curriculum stages synchronized with goal_velocity stages:
  #   0-250 iters: weight 1.3 (baseline, goal mostly in front)
  #   250-500 iters: weight 1.8 (increased for wider angles)
  #   500+ iters: weight 2.5 (strong rotation reward for ±90° goals)
  cfg.curriculum["ang_vel_cmd_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "track_ang_vel_cmd",
      "weight_stages": [
        {"step": 0, "weight": 1.3},
        {"step": 2500 * 24, "weight": 1.5},
        {"step": 3600 * 24, "weight": 1.8},
      ],
    },
  )

  # Apply play mode overrides
  if play:
    cfg.episode_length_s = int(1e9)
    # Keep actor observation noise enabled to match real-robot sensor noise.
    cfg.observations["actor"].enable_corruption = True
    cfg.events.pop("push_robot", None)

    # Set final stage velocity and angle for play
    goal_cmd = cfg.commands["goal"]
    if isinstance(goal_cmd, CircularGoalTrackingCommandCfg):
      goal_cmd.target_velocity_range = (0.8, 3.0)
      goal_cmd.target_velocity = 1.5
      goal_cmd.goal_angle_range = math.pi / 2  # ±90°

    # Enable keyboard control if KEYBOARD_CONTROL env var is set
    if mdp.is_keyboard_control_enabled():
      from mjlab.tasks.velocity.mdp.keyboard_command import (
        KeyboardCircularGoalTrackingCommandCfg,
      )

      cfg.commands["goal"] = KeyboardCircularGoalTrackingCommandCfg(
        entity_name="robot",
        resampling_time_range=(1e9, 1e9),  # Never resample
        goal_radius=3.0,  # Same as CircularGoalTrackingCommandCfg
        max_velocity=2.0,
        default_velocity=0.5,
        angle_step_deg=15.0,
        smoothing_alpha=0.15,
        debug_vis=True,
      )
      cfg.commands["goal"].viz.z_offset = 0.3

      # Disable goal velocity curriculum when using keyboard control
      if cfg.curriculum is not None and "goal_velocity" in cfg.curriculum:
        del cfg.curriculum["goal_velocity"]

  return cfg


def unitree_g1_circular_goal_flat_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 circular goal tracking on flat terrain with single-frame raycast.

  This is the single-frame (h1) variant that uses only 1 frame of raycast observation,
  removing temporal fusion from the depth encoder.
  """
  cfg = unitree_g1_circular_goal_flat_env_cfg(play=play)

  # Update all raycast observations to use single frame (no history)
  for raycast_group_name in ("raycast", "raycast_left", "raycast_right"):
    rc_term = cfg.observations[raycast_group_name].terms[raycast_group_name]
    rc_term.history_length = 1
    rc_term.flatten_history_dim = True

  return cfg


def unitree_g1_circular_goal_flat_h1_stage1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-1 variant with safety braking and light near-occlusion DR."""
  cfg = unitree_g1_circular_goal_flat_h1_env_cfg(play=play)
  _enable_command_safety_envelope(cfg)
  if play:
    _cap_play_goal_max_speed(cfg, _CURRICULUM_V6_PLAY_MAX_SPEED_MPS)
  if not play:
    _enable_near_occlusion_dr(cfg)
    _enable_spike_dr(cfg)

  # Override curriculum: max speed 1.5, scaled to 4000-iteration budget.
  assert cfg.curriculum is not None
  cfg.curriculum["goal_velocity"] = CurriculumTermCfg(
    func=mdp.circular_goal_velocity,
    params={
      "command_name": "goal",
      "velocity_stages": [
        {
          "step": 0,
          "target_velocity_range": (0.0, 1.0),
          "goal_angle_range": math.pi / 4,  # ±30°
        },
        {
          "step": 1500 * 24,
          "target_velocity_range": (0.0, 1.2),
          "goal_angle_range": math.pi / 3,  # ±60°
        },
        {
          "step": 3000 * 24,
          "target_velocity_range": (0.0, 1.5),
          "goal_angle_range": math.pi / 1.9,  # ±90°
        },
      ],
    },
  )
  cfg.curriculum["ang_vel_cmd_weight"] = CurriculumTermCfg(
    func=mdp.reward_weight,
    params={
      "reward_name": "track_ang_vel_cmd",
      "weight_stages": [
        {"step": 0, "weight": 1.3},
        {"step": 1500 * 24, "weight": 1.5},
        {"step": 3000 * 24, "weight": 1.8},
      ],
    },
  )

  _apply_g1_reward_body(cfg)

  return cfg


def unitree_g1_circular_goal_flat_g1rw_pure_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """CircularGoal-Flat with the pure Unitree-G1-Flat reward body.

  Starts from the Stage-1 circular goal flat config (which already applies
  ``_apply_g1_reward_body``) and adjusts tracking reward weights to match
  ``src/mjlab/tasks/velocity/config/g1``.

  Weight derivation from Unitree-G1-Flat velocity tracking:
  - ``track_linear_velocity`` (w=1.0) → ``track_goal_velocity`` (w=1.0)
  - ``track_angular_velocity`` (w=1.0) → ``track_ang_vel_cmd`` (w=1.0)
  - z-velocity and xy-angular-velocity penalties embedded in the originals
    are already covered by ``flat_orientation_l2`` (w=-5.0) and
    ``body_ang_vel`` (w=-0.05), so ``track_linear_velocity_z`` and
    ``track_ang_vel_xy`` are removed.

  Keeps ``in_place_turn_bonus`` for large heading-error turning assistance.
  """
  cfg = unitree_g1_circular_goal_flat_h1_stage1_env_cfg(play=play)

  assert cfg.rewards is not None

  # ---- Align tracking weights to Unitree-G1-Flat (1.0 / 1.0) ----
  cfg.rewards["track_goal_velocity"].weight = 1.0
  cfg.rewards["track_ang_vel_cmd"].weight = 1.0

  # ---- Remove stability terms already covered by G1 reward body ----
  # flat_orientation_l2 (-5.0) and body_ang_vel (-0.05) handle these.
  cfg.rewards.pop("track_linear_velocity_z", None)
  cfg.rewards.pop("track_ang_vel_xy", None)
  cfg.rewards.pop("velocity_tracking_log", None)

  # ---- Remove ang_vel_cmd_weight curriculum (G1-Flat has no reward curriculum) ----
  assert cfg.curriculum is not None
  cfg.curriculum.pop("ang_vel_cmd_weight", None)

  # ---- Align command_threshold / walking_threshold to Unitree-G1-Flat ----
  cfg.rewards["pose"].params["walking_threshold"] = 0.1
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.1
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.1
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.1

  return cfg


def unitree_g1_wall_maze_h1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 circular goal tracking with true maze terrain.

  This configuration combines circular goal tracking with a proper maze:
  - LARGER boundary walls (60x60m terrain)
  - True maze structure using recursive backtracking algorithm
  - Connected corridors with wide passages (4m+)
  - Multiple paths/loops for flexible robot navigation
  - 7x7 maze grid providing sufficient complexity
  - Central spawn area with cleared cells around it

  Uses single-frame (h1) raycast observation for simpler architecture.

  Curriculum stages based on training iterations:
    - 0-2000 iterations: More open maze (many walls removed)
    - 2000+ iterations: Proper maze structure (fewer walls removed)

  Internal base configuration. It is not registered as a task; navigation
  tasks reach it through the wall-maze and straight-approach chains.
  """

  # Start from circular goal flat configuration (single-frame)
  cfg = unitree_g1_circular_goal_flat_h1_env_cfg(play=play)

  # Increase contact buffer for wall-rich environment
  cfg.sim.nconmax = 128

  # Add true maze terrain (proper maze with connected corridors)
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_type = "generator"
  terrain_cfg.terrain_generator = _make_true_maze_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
    curriculum_step_threshold=2000,
  )

  # Start all environments at level 0 (no internal walls)
  if terrain_cfg.terrain_generator is not None:
    terrain_cfg.max_init_terrain_level = 0

  # Add terrain curriculum based on steps
  # Maps training iterations to difficulty levels
  assert cfg.curriculum is not None
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_by_step,
    params={
      "command_name": "goal",
      "level_stages": [
        {"step": 7000, "level": 0},  # No internal walls
        {"step": 9000 * 24, "level": 1},  # Enable internal walls (difficulty >= 0.2)
        {"step": 10000 * 24, "level": 2},  # Full difficulty
      ],
    },
  )

  # Play mode overrides
  if play:
    if terrain_cfg.terrain_generator is not None:
      terrain_cfg.terrain_generator.curriculum = False
      terrain_cfg.terrain_generator.num_cols = 4
      terrain_cfg.terrain_generator.border_width = 12.0
      terrain_cfg.terrain_generator.num_rows = 3

      # Allow selecting difficulty level via WALL_LEVEL env var (0, 1, or 2)
      # 0: No internal walls, 1-2: Internal walls enabled
      wall_level_str = os.environ.get("WALL_LEVEL", "")
      if wall_level_str:
        try:
          wall_level = int(wall_level_str)
          wall_level = max(0, min(2, wall_level))
          terrain_cfg.terrain_generator.difficulty_range = (
            wall_level / 3.0,
            (wall_level + 1) / 3.0 - 0.01,
          )
        except ValueError:
          pass
      else:
        # Default: use moderate difficulty (some internal walls)
        terrain_cfg.terrain_generator.difficulty_range = (0.3, 0.7)

    # Enable keyboard control if KEYBOARD_CONTROL env var is set
    if mdp.is_keyboard_control_enabled():
      from mjlab.tasks.velocity.mdp.keyboard_command import (
        KeyboardCircularGoalTrackingCommandCfg,
      )

      cfg.commands["goal"] = KeyboardCircularGoalTrackingCommandCfg(
        entity_name="robot",
        resampling_time_range=(1e9, 1e9),  # Never resample
        goal_radius=3.0,
        max_velocity=2.0,
        default_velocity=0.5,
        angle_step_deg=15.0,
        smoothing_alpha=0.15,
        debug_vis=True,
      )
      cfg.commands["goal"].viz.z_offset = 0.3

      # Disable goal velocity curriculum when using keyboard control
      if cfg.curriculum is not None and "goal_velocity" in cfg.curriculum:
        del cfg.curriculum["goal_velocity"]

  return cfg


# ==============================================================================
# Reachable Goal Short-Corridor Maze Variants
# Environment for RA (Reach-Avoid) training where robot must reach a fixed goal.
# Unlike CircularGoalTrackingCommand which maintains constant 3m distance,
# this uses ReachableGoalCommand where the goal is a fixed position that can
# be reached, enabling proper Reach-Avoid value learning.
#
# Key features for short-corridor maze navigation:
# - Proper maze structure with connected corridors (6x6 grid)
# - SHORTER corridors than standard maze (fewer maze cells)
# - MORE OPENINGS/EXITS (55% extra openings vs 35%)
# - Wide passages (4m+) for robot navigation
# - Path planning can find short detour paths (especially right-side bypass)
# ==============================================================================


def unitree_g1_wall_maze_reachable_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 reachable goal tracking with short-corridor maze terrain.

  This configuration is designed for RA (Reach-Avoid) training with
  a maze that has SHORT CORRIDORS and MANY OPENINGS:
  - Goal is a FIXED position in world frame (not moving with robot)
  - Target distance = resample_interval * target_velocity
  - Robot walks towards the fixed goal, distance decreases
  - When robot reaches goal (distance < threshold), marked as success
  - Uses SHORT-CORRIDOR maze: proper maze structure but easier to navigate

  Key features of the SHORT-CORRIDOR maze:
  - 6x6 maze grid (vs 10x10 standard) = shorter corridor segments
  - 55% extra openings (vs 35% standard) = more escape routes/exits
  - Wide corridors (4m+) for comfortable robot navigation
  - Central spawn area with 2-cell clear zone
  - Maintains proper maze structure (connected corridors via DFS)

  Key differences from WallMaze (CircularGoalTracking):
  - Goal doesn't move with robot - it's fixed in world frame
  - Distance to goal decreases as robot walks towards it
  - Goal is "reachable" - robot can actually get there
  - Enables multi-waypoint goal tracking with reachable goals

  Curriculum stages:
    - 0-2000 iterations: Very open maze (even more walls removed)
    - 2000+ iterations: Standard short-corridor maze (many exits)

  Internal base configuration. It is not registered as a task; the navigation
  tasks reach it through the wall-maze and straight-approach chains.
  """
  from mjlab.tasks.velocity.mdp.curriculums import (
    reachable_goal_velocity as reachable_goal_velocity_curriculum,
  )
  from mjlab.tasks.velocity.mdp.observations import (
    reachable_goal_command as reachable_goal_command_obs,
  )
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    ReachableGoalCommandCfg,
  )

  # Start from wall maze h1 configuration (includes wide street maze terrain)
  cfg = unitree_g1_wall_maze_h1_env_cfg(play=play)

  # Override terrain with short-corridor maze (proper maze structure but shorter
  # corridors and more openings for easier right-side bypass navigation)
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_type = "generator"
  terrain_cfg.terrain_generator = _make_short_corridor_maze_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
    curriculum_step_threshold=2000,
  )

  # Start all environments at level 0 (more open maze)
  if terrain_cfg.terrain_generator is not None:
    terrain_cfg.max_init_terrain_level = 0

  # Replace CircularGoalTrackingCommand with ReachableGoalCommand
  # This is the key change for RA training
  # Enable path planning for maze navigation
  assert cfg.commands is not None
  cfg.commands = {
    "goal": ReachableGoalCommandCfg(
      entity_name="robot",
      resampling_time_range=(4.0, 12.0),  # Base time range (adjusted by path planner)
      goal_angle_range=math.pi / 3,  # ±60° - wider range for diverse goals
      target_velocity=0.8,  # Default target velocity
      target_velocity_range=(0.5, 1.2),  # Moderate speed for careful navigation
      min_target_distance=3.0,  # Larger minimum distance
      initial_ring_radius=4.0,  # Larger visualization ring
      shrink_threshold=4.0,  # Start shrinking when close
      reach_threshold=0.6,  # Goal resample threshold
      debug_vis=True,
      # Path planning settings for maze navigation
      path_planning=ReachableGoalCommandCfg.PathPlanningCfg(
        enabled=True,  # Enable A* path planning
        grid_resolution=0.5,  # 0.5m per grid cell
        robot_radius=0.4,  # Robot collision radius
        waypoint_spacing=2.5,  # Distance between waypoints
        waypoint_reach_threshold=1.2,  # Threshold to switch to next waypoint
        max_resample_time=30.0,  # Max time before forced resample (early resample on goal reach)
        alternative_search_range=math.pi / 2,  # Search up to 90° right for alternatives
        alternative_candidates=16,  # Try 16 angles when finding alternatives
      ),
    )
  }

  # Update observations for reachable goal tracking
  policy_terms = cfg.observations["actor"].terms

  # Replace circular_goal_command with reachable_goal_command
  if "goal_command" in policy_terms:
    policy_terms["goal_command"] = ObservationTermCfg(
      func=reachable_goal_command_obs,
      params={"command_name": "goal"},
    )

  # Update critic observations similarly
  critic_terms = cfg.observations["critic"].terms
  if "goal_command" in critic_terms:
    critic_terms["goal_command"] = ObservationTermCfg(
      func=reachable_goal_command_obs,
      params={"command_name": "goal"},
    )

  # NOTE: We do NOT add goal_distance to critic observations because:
  # 1. It would break checkpoint compatibility with pre-trained policies
  # 2. The command already contains delta_yaw and target_velocity
  # 3. RA training computes distance separately via GoalDistanceComputer

  # Update rewards for reachable goal tracking
  # The key rewards are the same (angular velocity + velocity tracking)
  # because ReachableGoalCommand provides the same command interface
  assert cfg.rewards is not None

  # Note: We keep most rewards the same because:
  # 1. ReachableGoalCommand provides [delta_yaw, target_velocity] - same as circular
  # 2. The tracking rewards work with any command that has delta_yaw
  # 3. The key difference is in the goal dynamics, not the reward structure

  # Update goal velocity curriculum to use reachable goal range
  if cfg.curriculum is not None:
    # Remove circular goal curriculum if present
    if "goal_velocity" in cfg.curriculum:
      del cfg.curriculum["goal_velocity"]

    # Add reachable goal curriculum
    cfg.curriculum["goal_velocity"] = CurriculumTermCfg(
      func=reachable_goal_velocity_curriculum,
      params={
        "command_name": "goal",
        "velocity_stages": [
          {
            "step": 0,
            "target_velocity_range": (0.5, 1.0),
            "goal_angle_range": math.pi / 6,  # ±30°
          },
          {
            "step": 500 * 24,
            "target_velocity_range": (0.5, 1.3),
            "goal_angle_range": math.pi / 4,  # ±45°
          },
          {
            "step": 1000 * 24,
            "target_velocity_range": (0.5, 1.5),
            "goal_angle_range": math.pi / 3,  # ±60°
          },
        ],
      },
    )

  # Play mode overrides
  if play:
    goal_cmd = cfg.commands["goal"]
    assert isinstance(goal_cmd, ReachableGoalCommandCfg)
    goal_cmd.target_velocity_range = (0.8, 3.0)
    goal_cmd.target_velocity = 1.0
    goal_cmd.goal_angle_range = math.pi / 3  # ±60°

    # Longer resampling for manual observation
    goal_cmd.resampling_time_range = (5.0, 10.0)

  return cfg


def unitree_g1_obstacle_mild_rough_reachable_nopp_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Mixed obstacles + mild rough terrain NoPP task (replaces Stage 3 + Stage 4).

  Combines 70% obstacle tiles (thin pillars + irregular obstacles) with 30%
  mild heightfield-based rough terrain (difficulty ≤ 0.16). This prevents
  catastrophic forgetting by training on both terrain types simultaneously.

  The rough tiles are pure heightfield — they add zero extra geom bodies, so
  nconmax stays at the same 256 as the pure-obstacle ComplexGeneralization task.
  num_envs = 4096 works without OOM.

  Behavior goals:
  - High obstacles → robot learns to slow down / avoid / not fear contact
  - Mild rough → robot learns to walk through at full speed (up to 3 m/s)
  """
  from mjlab.tasks.velocity.mdp.curriculums import (
    reachable_goal_velocity as reachable_goal_velocity_curriculum,
  )
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    ReachableGoalCommandCfg,
  )

  cfg = unitree_g1_wall_maze_reachable_h1_env_cfg(play=play)

  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_type = "generator"
  terrain_cfg.terrain_generator = _make_obstacle_with_mild_rough_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
  )

  if terrain_cfg.terrain_generator is not None:
    terrain_cfg.max_init_terrain_level = 0

  # Terrain difficulty curriculum: step-based progression across difficulty
  # levels. Checkpoint from Stage 2 is at ~14000 iterations (336k steps).
  # Start at level 0 (easiest) at load time, then gradually increase.
  # Matches the proven ComplexGeneralization pattern.
  assert cfg.curriculum is not None
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_by_step,
    params={
      "command_name": "goal",
      "level_stages": [
        {"step": 10000 * 24, "level": 0},
        {"step": 11500 * 24, "level": 1},
        {"step": 12500 * 24, "level": 2},
      ],
    },
  )

  # Velocity curriculum: already fully open at checkpoint load time.
  # Range includes 0.0 so robot self-regulates: fast on easy ground,
  # slow near hard obstacles. Matches ComplexGeneralization approach.
  cfg.curriculum["goal_velocity"] = CurriculumTermCfg(
    func=reachable_goal_velocity_curriculum,
    params={
      "command_name": "goal",
      "velocity_stages": [
        {
          "step": 10000 * 24,
          "target_velocity_range": (0.0, 1.0),
          "goal_angle_range": math.pi / 4,
        },
        {
          "step": 11500 * 24,
          "target_velocity_range": (0.0, 1.5),
          "goal_angle_range": math.pi / 3,
        },
        {
          "step": 12500 * 24,
          "target_velocity_range": (0.0, 2.2),
          "goal_angle_range": math.pi / 1.85,
        },
      ],
    },
  )

  goal_cmd = cfg.commands["goal"]
  assert isinstance(goal_cmd, ReachableGoalCommandCfg)
  assert goal_cmd.path_planning is not None

  # NoPP: disable A* waypoints, rely on local sensing only.
  goal_cmd.path_planning.enabled = False
  goal_cmd.path_planning.waypoint_spacing = 30.0
  goal_cmd.path_planning.dynamic_resample_time = True
  goal_cmd.path_planning.resample_time_margin = 3.0

  goal_cmd.reach_resample_delay = 0.0
  goal_cmd.goal_angle_range = math.pi / 2
  goal_cmd.min_target_distance = 6.0
  goal_cmd.resampling_time_range = (8.0, 18.0)
  goal_cmd.initial_ring_radius = 6.0
  goal_cmd.shrink_threshold = 6.0

  # Contact settings: inherit nconmax=128 from WallMaze base config.
  # Do NOT set ccd_iterations=500 — that inflates the EPA convex collision
  # buffer to ~15 GB when obstacle geoms (boxes/cylinders) are present.
  # The mild rough heightfield at difficulty ≤ 0.16 is nearly flat and does
  # not need aggressive CCD.  The working ComplexGeneralization task uses
  # the same defaults successfully.

  if play and terrain_cfg.terrain_generator is not None:
    terrain_cfg.terrain_generator.curriculum = False
    terrain_cfg.terrain_generator.num_cols = 4
    terrain_cfg.terrain_generator.num_rows = 4
    terrain_cfg.terrain_generator.border_width = 12.0
    terrain_cfg.terrain_generator.difficulty_range = (0.0, 0.16)
    terrain_cfg.max_init_terrain_level = None

  if play and "randomize_terrain" in cfg.events:
    randomize_terrain_cfg = cfg.events.pop("randomize_terrain")
    cfg.events = {"randomize_terrain": randomize_terrain_cfg, **cfg.events}

  if play:
    goal_cmd.target_velocity_range = (0.0, 1.5)
    goal_cmd.target_velocity = 1.0

  return cfg


def unitree_g1_obstacle_mild_rough_reachable_nopp_h1_stage3_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-3 variant with safety braking and light near-occlusion DR."""
  from mjlab.tasks.velocity.mdp.curriculums import (
    reachable_goal_velocity as reachable_goal_velocity_curriculum,
  )

  cfg = unitree_g1_obstacle_mild_rough_reachable_nopp_h1_env_cfg(play=play)
  if play and mdp.is_keyboard_control_enabled():
    _enable_keyboard_circular_goal_play(cfg)
  if play and mdp.is_manual_obstacle_control_enabled():
    _enable_manual_play_obstacle(cfg)
  _enable_command_safety_envelope(cfg)
  if play:
    _cap_play_goal_max_speed(cfg, _CURRICULUM_V6_PLAY_MAX_SPEED_MPS)
  if not play:
    _enable_near_occlusion_dr(cfg)
    _enable_spike_dr(cfg)

  # Override curriculum: starts at iter 8000, max speed 1.2, budget 4500 iters.
  assert cfg.curriculum is not None
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_by_step,
    params={
      "command_name": "goal",
      "level_stages": [
        {"step": 8000 * 24, "level": 0},
        {"step": 9800 * 24, "level": 1},
        {"step": 12500 * 24, "level": 2},
      ],
    },
  )
  if not play:
    cfg.curriculum["goal_velocity"] = CurriculumTermCfg(
      func=reachable_goal_velocity_curriculum,
      params={
        "command_name": "goal",
        "velocity_stages": [
          {
            "step": 8000 * 24,
            "target_velocity_range": (0.0, 1.0),
            "goal_angle_range": math.pi / 4,  # ±45°
          },
          {
            "step": 9800 * 24,
            "target_velocity_range": (0.0, 1.2),
            "goal_angle_range": math.pi / 3,  # ±60°
          },
          {
            "step": 12500 * 24,
            "target_velocity_range": (0.0, 1.5),
            "goal_angle_range": math.pi / 1.9,  # ±75°
          },
        ],
      },
    )

  _apply_g1_reward_body(cfg)

  return cfg


def unitree_g1_obstacle_mild_rough_g1rw_astar_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-3 A* with g1 reward body and wider terrain gaps.

  Builds on ``unitree_g1_obstacle_mild_rough_reachable_nopp_h1_stage3_env_cfg``
  (which already applies the g1 reward body) and adds:
  - A* obstacle-guided waypoints
  - Wider obstacle gaps and raised obstacle heights for raycast visibility
  - Removes ``goal_sequence_complete`` termination (not present in parent chain)
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    ReachableGoalCommandCfg,
  )

  cfg = unitree_g1_obstacle_mild_rough_reachable_nopp_h1_stage3_env_cfg(play=play)

  # ---- Enable A* path planning ----
  goal_cmd = cfg.commands["goal"]
  if isinstance(goal_cmd, ReachableGoalCommandCfg):
    assert goal_cmd.path_planning is not None
    goal_cmd.path_planning.enabled = True
    goal_cmd.path_planning.waypoint_spacing = 2.5
    goal_cmd.path_planning.waypoint_reach_threshold = 1.2
    goal_cmd.path_planning.dynamic_resample_time = True
    goal_cmd.path_planning.resample_time_margin = 2.0

  # ---- Terminations: remove goal_sequence_complete if present ----
  if cfg.terminations is not None:
    cfg.terminations.pop("goal_sequence_complete", None)

  # ---- Terrain: wider gaps + raise all obstacles to >= robot height ----
  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    sub_terrains = cfg.scene.terrain.terrain_generator.sub_terrains

    if "irregular_obstacle_field" in sub_terrains:
      obs_cfg = sub_terrains["irregular_obstacle_field"]
      assert isinstance(obs_cfg, ObstacleFieldTerrainCfg)
      obs_cfg.min_gap = 0.8
      obs_cfg.min_height_scale = 1.0

    if "thin_pillar_field" in sub_terrains:
      pillar_cfg = sub_terrains["thin_pillar_field"]
      assert isinstance(pillar_cfg, UniformPillarFieldTerrainCfg)
      pillar_cfg.min_pillar_spacing = 3.0
      _rh = pillar_cfg.robot_height
      _upper = max(2.0, _rh + 0.5)
      pillar_cfg.wall_height_range = (_rh, _upper)

  return cfg


def unitree_g1_obstacle_flat_g1rw_astar_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat-terrain obstacle avoidance with A* and G1 reward body.

  Same as ``unitree_g1_obstacle_mild_rough_g1rw_astar_h1_env_cfg`` but
  replaces the mixed obstacle + mild-rough terrain with flat ground plus
  obstacles only (no heightfield rough tiles).

  The flat-obstacle variant is used as Stage 2 by
  ``src/mjlab/tool/curriculum_train_v6.sh``, so it carries its own cumulative
  curriculum window: start at iter 4000 and spend the next 4000 iterations
  ramping to the final 1.5 m/s target.
  """
  from mjlab.tasks.velocity.mdp.curriculums import (
    reachable_goal_velocity as reachable_goal_velocity_curriculum,
  )

  cfg = unitree_g1_obstacle_mild_rough_g1rw_astar_h1_env_cfg(play=play)

  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_generator = _make_obstacle_flat_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
  )

  # Re-apply wider gaps + raised obstacles (same as parent A* config)
  if terrain_cfg.terrain_generator is not None:
    sub_terrains = terrain_cfg.terrain_generator.sub_terrains

    if "irregular_obstacle_field" in sub_terrains:
      obs_cfg = sub_terrains["irregular_obstacle_field"]
      assert isinstance(obs_cfg, ObstacleFieldTerrainCfg)
      obs_cfg.min_gap = 0.8
      obs_cfg.min_height_scale = 1.0

    if "thin_pillar_field" in sub_terrains:
      pillar_cfg = sub_terrains["thin_pillar_field"]
      assert isinstance(pillar_cfg, UniformPillarFieldTerrainCfg)
      pillar_cfg.min_pillar_spacing = 3.0
      _rh = pillar_cfg.robot_height
      _upper = max(2.0, _rh + 0.5)
      pillar_cfg.wall_height_range = (_rh, _upper)

  # Override curriculum: starts at iter 4000, max speed 1.5, budget 4000 iters.
  assert cfg.curriculum is not None
  cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
    func=mdp.terrain_levels_by_step,
    params={
      "command_name": "goal",
      "level_stages": [
        {"step": 4000 * 24, "level": 0},
        {"step": 6000 * 24, "level": 1},
        {"step": 8000 * 24, "level": 2},
      ],
    },
  )
  cfg.curriculum["goal_velocity"] = CurriculumTermCfg(
    func=reachable_goal_velocity_curriculum,
    params={
      "command_name": "goal",
      "velocity_stages": [
        {
          "step": 4000 * 24,
          "target_velocity_range": (0.0, 1.0),
          "goal_angle_range": math.pi / 4,  # ±45°
        },
        {
          "step": 6000 * 24,
          "target_velocity_range": (0.0, 1.3),
          "goal_angle_range": math.pi / 3,  # ±60°
        },
        {
          "step": 8000 * 24,
          "target_velocity_range": (0.0, 1.5),
          "goal_angle_range": math.pi / 1.9,  # ±75°
        },
      ],
    },
  )

  if play and terrain_cfg.terrain_generator is not None:
    terrain_cfg.terrain_generator.curriculum = False
    terrain_cfg.terrain_generator.num_cols = 4
    terrain_cfg.terrain_generator.num_rows = 4
    terrain_cfg.terrain_generator.border_width = 12.0
    terrain_cfg.max_init_terrain_level = None

  return cfg


def unitree_g1_obstacle_mild_rough_g1rw_pure_astar_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Obstacle + mild-rough terrain with A*, using G1Rw-Pure reward body.

  Same as ``unitree_g1_obstacle_mild_rough_g1rw_astar_h1_env_cfg`` but
  replaces the reward body with the Unitree-G1-Flat (Pure) convention:
  tracking weights 1.0/1.0, no z/xy stability tracking terms.
  """
  cfg = unitree_g1_obstacle_mild_rough_g1rw_astar_h1_env_cfg(play=play)
  _apply_g1rw_pure_reward_adjustments(cfg)
  return cfg


def unitree_g1_obstacle_flat_g1rw_pure_astar_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat-terrain obstacles with A*, using G1Rw-Pure reward body.

  Same as ``unitree_g1_obstacle_flat_g1rw_astar_h1_env_cfg`` but
  replaces the reward body with the Unitree-G1-Flat (Pure) convention:
  tracking weights 1.0/1.0, no z/xy stability tracking terms.
  """
  cfg = unitree_g1_obstacle_flat_g1rw_astar_h1_env_cfg(play=play)
  _apply_g1rw_pure_reward_adjustments(cfg)
  return cfg


def unitree_g1_mixed_scenario_reachable_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 config with MIXED SCENARIOS for diverse local planner training.

  This configuration combines THREE terrain types to address training bias:
  - 40%: SingleWall (fixed 8m width) — standard right-bypass obstacle avoidance
  - 30%: Open field (no walls) — pure goal-directed navigation
  - 30%: SingleWall (random 2-8m width) — generalized wall avoidance

  Key improvements over single-scenario training:
  1. Open-field envs teach the local planner to stably navigate toward goals in open space,
     preventing the "continue turning right after passing the wall" failure mode.
  2. Random wall-width envs improve generalization to different obstacle sizes.
  3. Spawn position randomization (±1.5m X, ±0.5m Y) breaks overfitting to
     a single starting trajectory.
  4. Goal position noise (±1.0m X, ±0.5m Y for key goals) diversifies target
     approach angles.
  5. Robot orientation is EXACTLY aligned with world +Y axis (yaw = π/2),
     no variance, to ensure consistent forward direction across scenarios.

  In open-field terrain cells, the dynamic target blending in train_local_planner.py
  automatically handles the lack of walls: min_ray is always large →
  alpha ≈ 1 → target = key_goal (waypoints are effectively ignored).

  Layout is identical to SingleWall tasks (20×20m terrain cells), but uses
  a 5×2 grid (10 cells) to support the 40/30/30 proportion split.

  Usage:
    python -m mjlab.tasks.steering.train_local_planner --mode train \\
      --task Mjlab-StraightApproach-LowSpeed-G1-v6 \\
      --checkpoint <pretrained_policy> \\
      --num_envs 2048 --output_dir logs/steering \\
      --experiment_name mixed_scenario_v1
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
  )

  # Start from circular goal flat h1 configuration
  cfg = unitree_g1_circular_goal_flat_h1_env_cfg(play=play)

  # Override terrain with mixed-scenario generator (40/30/30 split)
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_type = "generator"
  terrain_cfg.terrain_generator = _make_mixed_scenario_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
  )

  # No terrain curriculum for mixed scenario
  if terrain_cfg.terrain_generator is not None:
    terrain_cfg.max_init_terrain_level = None

  # Robot initial orientation: EXACTLY facing +Y direction (yaw = π/2)
  # No variance — ensures consistent forward direction across all scenarios.
  # Spawn POSITION is randomized: X ± 2.0m, Y ± 0.5m relative to terrain origin.
  # Increased X range (from ±1.5m) with the larger 30×30m terrain cells to
  # further break overfitting to a single starting trajectory.
  _yaw = math.pi / 2.0
  if cfg.events is not None and "reset_base" in cfg.events:
    reset_base_cfg = cfg.events["reset_base"]
    if reset_base_cfg.params is not None and "pose_range" in reset_base_cfg.params:
      reset_base_cfg.params["pose_range"]["yaw"] = (_yaw, _yaw)  # Exactly π/2
      reset_base_cfg.params["pose_range"]["x"] = (-2.0, 2.0)  # ±2.0m lateral
      reset_base_cfg.params["pose_range"]["y"] = (-0.5, 0.5)  # ±0.5m fore/aft

  # Velocity-adaptive waypoints for bypass path.
  # The FixedSequentialGoalCommand.reset() will call
  # _generate_speed_adapted_offsets() to compute a turning-radius-compatible
  # path for the sampled speed at each episode reset.
  # goal_offsets below serve as (a) the source of num_goals=9, and (b) a
  # fallback path for v=0.8 m/s (the nominal speed) in case geometry
  # extraction fails.
  # mirror_x_probability=0.5 still flips 50% of episodes to a left-side bypass.
  # Key goals (WP7 and WP8) get ±1.0m / ±0.5m noise for diverse approach angles.
  assert cfg.commands is not None
  cfg.commands = {
    "goal": FixedSequentialGoalCommandCfg(
      entity_name="robot",
      resampling_time_range=(30.0, 30.0),  # fallback; overridden by dynamic_timeout
      goal_offsets=[
        # Fallback right-side bypass path at v=0.8 m/s (robot_distance_from_wall=6m)
        # wall is 6m in front → wall_y=6.0, bypass_x=4.0+1.4=5.4
        (0.0, 2.0),  # WP0: straight approach
        (2.0, 3.5),  # WP1: start rightward arc
        (4.5, 5.5),  # WP2: near wall's right edge
        (5.4, 6.0),  # WP3: at wall level
        (5.4, 7.0),  # WP4: just past wall
        (3.5, 9.0),  # WP5: beginning return arc
        (1.0, 11.0),  # WP6: returning toward centre
        (0.0, 12.5),  # WP7 (Key Goal 1): behind wall centre, ~6.5m past wall
        (0.0, 15.0),  # WP8 (Key Goal 2): final goal
      ],
      # Noise: only applied to key-goal waypoints (WP7, WP8) in adaptive mode;
      # all waypoints get noise in legacy (non-adaptive) mode.
      goal_offset_noise=[
        (0.0, 0.0),  # WP0
        (0.5, 0.3),  # WP1
        (0.3, 0.3),  # WP2
        (0.3, 0.2),  # WP3
        (0.3, 0.2),  # WP4
        (0.5, 0.4),  # WP5
        (0.5, 0.4),  # WP6
        (1.0, 0.5),  # WP7 (Key Goal 1)
        (1.0, 0.5),  # WP8 (Key Goal 2)
      ],
      key_goal_indices=[7, 8],
      target_velocity=0.8,
      target_velocity_range=(0.2, 3.0),
      near_zero_probability=0.1,
      near_zero_velocity_range=(0.0, 0.1),
      near_zero_duration_range=(0.5, 1.0),
      dynamic_timeout_by_velocity=True,
      timeout_margin_s=10.0,
      timeout_speed_floor=0.2,
      timeout_min_s=20.0,
      timeout_max_s=120.0,
      reach_threshold=1.0,
      # Increased from 0.5 to 1.2m: at 3 m/s (R=3m) the robot can't hit a
      # 0.5m circle without spinning.  1.2m < 1.375m (min waypoint spacing
      # at slowest speed) so consecutive thresholds never overlap.
      intermediate_reach_threshold=1.2,
      mirror_x_probability=0.5,  # 50% left-side bypass for balanced training
      # Increased from 1.4 to 2.0m: must be > intermediate_reach_threshold
      # (1.2m) + robot half-width (~0.3m) so a waypoint touch never puts the
      # robot inside a wall.  Widens bypass_x from ~5.4m to ~6.0m (safe in
      # 30×30m cells).
      min_obstacle_clearance=2.0,
      # Velocity-adaptive waypoint generation
      velocity_adaptive_offsets=True,
      max_angular_velocity=1.0,
      robot_distance_from_wall=6.0,
      default_wall_half_width=4.0,
      max_heading_change=math.pi / 1.9,
      debug_vis=True,
    )
  }

  # Update observations for fixed sequential goal tracking
  from mjlab.tasks.velocity.mdp.observations import (
    fixed_sequential_goal_command as fixed_goal_command_obs,
  )

  policy_terms = cfg.observations["actor"].terms
  if "goal_command" in policy_terms:
    policy_terms["goal_command"] = ObservationTermCfg(
      func=fixed_goal_command_obs,
      params={"command_name": "goal"},
    )

  critic_terms = cfg.observations["critic"].terms
  if "goal_command" in critic_terms:
    critic_terms["goal_command"] = ObservationTermCfg(
      func=fixed_goal_command_obs,
      params={"command_name": "goal"},
    )

  # Add termination condition
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.tasks.velocity.mdp.terminations import fixed_sequence_complete

  if cfg.terminations is None:
    cfg.terminations = {}
  cfg.terminations["goal_sequence_complete"] = TerminationTermCfg(
    func=fixed_sequence_complete,
    params={"command_name": "goal"},
  )

  # Must not be lower than dynamic timeout upper bound.
  cfg.episode_length_s = 90.0

  # Increase contact buffer for multi-cell terrain with multiple wall geoms.
  # Base velocity config only reserves nconmax=35 which overflows with 10 terrain
  # cells each containing wall + ground geoms.
  cfg.sim.nconmax = 128

  # Fix event ordering: randomize_terrain must run BEFORE reset_base.
  # The base config (play=True) appends randomize_terrain at the END of the events dict,
  # which means it runs AFTER reset_base. This causes env_origins to be modified in-place
  # between robot placement (reset_base) and goal placement (command_manager.reset()),
  # making robot and goals end up in different terrain cells.
  # Fix: move randomize_terrain to the FRONT of the dict so it runs first.
  if "randomize_terrain" in cfg.events:
    _rt_cfg = cfg.events.pop("randomize_terrain")
    cfg.events = {"randomize_terrain": _rt_cfg, **cfg.events}

  # Remove terrain curriculum
  if cfg.curriculum is not None:
    if "terrain_levels" in cfg.curriculum:
      del cfg.curriculum["terrain_levels"]
    if "goal_velocity" in cfg.curriculum:
      del cfg.curriculum["goal_velocity"]

  return cfg


def _make_indoor_mixed_scenario_generator_cfg(
  robot_height: float, play: bool
) -> TerrainGeneratorCfg:
  """Create terrain generator for indoor mixed-scenario training (v5).

  Five scenario types to give the local planner network exposure to diverse indoor
  environments while preserving compatibility with the v3 bypass-path training:

    25%: IndoorRoomTerrainCfg — fixed 6m centre wall, outer room walls, furniture
    20%: IndoorRoomTerrainCfg — random 2-6m centre wall, outer room walls, furniture
    20%: SingleWallTerrainCfg — traditional single wall, no outer room (v3 compat)
    20%: OpenFieldTerrainCfg  — no obstacles, pure goal navigation (v3 compat)
    15%: IndoorRoomTerrainCfg — suspended centre wall with 20-45cm bottom gap

  Grid: 5×2 = 10 terrain cells at 30×30m each.
  Larger terrain cells (30×30m vs v3's 20×20m) give room for wider turning arcs
  at the reduced maximum speed of 1.5 m/s (R_max = 1.5 m).

  Speed range: 0.2–1.5 m/s (safe indoor speed; R varies 0.2–1.5 m).
  """
  return TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=2.0,
    num_rows=5,
    num_cols=2,
    curriculum=False,
    sub_terrains={
      # 25%: Indoor room with fixed 6m centre obstacle wall
      "indoor_fixed_wall": IndoorRoomTerrainCfg(
        proportion=0.25,
        robot_height=robot_height,
        center_wall_width=6.0,
        center_wall_height=max(2.0, robot_height + 0.5),
        center_wall_thickness=0.2,
        robot_distance_from_wall=5.0,
        room_width_range=(14, 16),
        room_length_range=(14, 16),
        num_furniture_range=(4, 8),
        furniture_clear_zone_x_half=3.0,
      ),
      # 20%: Indoor room with randomised 2-6m centre wall for generalisation
      "indoor_random_wall": IndoorRoomTerrainCfg(
        proportion=0.20,
        robot_height=robot_height,
        center_wall_width=4.0,  # fallback; overridden by range
        center_wall_width_range=(2.0, 6.0),
        center_wall_height=max(2.0, robot_height + 0.5),
        center_wall_thickness=0.2,
        robot_distance_from_wall=5.0,
        room_width_range=(14, 16),
        room_length_range=(14, 16),
        num_furniture_range=(4, 8),
        furniture_clear_zone_x_half=3.0,
      ),
      # 20%: Classic single wall (no outer room) — preserves v3 experience
      "single_wall": SingleWallTerrainCfg(
        proportion=0.20,
        robot_height=robot_height,
        wall_height=max(2.0, robot_height + 0.5),
        wall_thickness=0.2,
        wall_width=8.0,
        robot_distance_from_wall=5.0,
      ),
      # 20%: Open field — no obstacles, pure goal-directed navigation
      "open_field": OpenFieldTerrainCfg(
        proportion=0.20,
        robot_height=robot_height,
        robot_distance_from_wall=5.0,
      ),
      # 15%: Indoor centre wall lifted 20-45cm above the floor
      "indoor_suspended_wall": IndoorRoomTerrainCfg(
        proportion=0.15,
        robot_height=robot_height,
        center_wall_width=6.0,
        center_wall_height=max(2.0, robot_height + 0.5),
        center_wall_thickness=0.2,
        center_wall_bottom_gap_range=(0.20, 0.45),
        robot_distance_from_wall=5.0,
        room_width_range=(14, 16),
        room_length_range=(14, 16),
        num_furniture_range=(4, 8),
        furniture_clear_zone_x_half=3.0,
      ),
    },
    difficulty_range=(0.0, 1.0),
    add_lights=True,
  )


def unitree_g1_indoor_mixed_scenario_reachable_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 indoor mixed-scenario config for local planner training (v5).

  Extends the v3 Mixed Scenario by adding realistic indoor environments:
  - Outer room walls (simulating indoor rooms / corridors, 12-16m wide)
  - Furniture near walls (tables, chairs, cabinets, columns)
  - Speed range reduced to 0.2-1.5 m/s (safe indoor operating speed)
  - Velocity-adaptive waypoints (turning-radius-compatible paths)

  Terrain composition (30×30m cells, 5×2 grid):
    35%: Indoor + fixed 8m wall — standard avoidance in room
    25%: Indoor + random 2-8m wall — generalised avoidance in room
    20%: Single wall (no outer room) — backward-compatible with v3
    20%: Open field — pure goal-directed navigation

  The outer walls and furniture give raycast sensors realistic returns in all
  directions, improving real-world deployment (sim-to-real gap reduction).
  Furniture in the periphery (>3m from robot path) does NOT block navigation
  but trains the local planner to handle incidental raycasts from non-blocking objects.

  Usage:
    python -m mjlab.tasks.steering.train_local_planner --mode train \\
      --task Mjlab-IndoorStraightApproach-G1-v6 \\
      --checkpoint <pretrained_policy> \\
      --num_envs 1024 --output_dir logs/steering \\
      --experiment_name indoor_v5
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
  )

  # Build from the v3 training config as the base so all common settings
  # (observations, rewards, events, terminations) are correctly inherited.
  cfg = unitree_g1_mixed_scenario_reachable_h1_env_cfg(play=play)

  # Replace terrain generator with the indoor variant
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_generator = _make_indoor_mixed_scenario_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
  )

  # Modify goal command: lower speed ceiling to 1.5 m/s for safe indoor nav
  # and adjust distances to robot_distance_from_wall=5.0m (indoor terrain).
  # Clearance reduced to 0.8m (from 1.4m) to fit within 14-16m room width.
  assert cfg.commands is not None
  cfg.commands["goal"] = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),  # fallback; overridden by dynamic_timeout
    goal_offsets=[
      # Fallback right-side bypass (v=0.8 m/s, dist_from_wall=5, wall_hw=3)
      (0.0, 1.6),  # WP0: straight approach
      (1.5, 3.1),  # WP1: veering right
      (3.5, 4.4),  # WP2: near wall edge
      (3.8, 5.0),  # WP3: at wall level
      (3.8, 5.8),  # WP4: just past wall
      (2.5, 7.5),  # WP5: returning toward centre
      (0.5, 9.0),  # WP6: nearly centred
      (0.0, 10.0),  # WP7 (Key Goal 1)
      (0.0, 12.5),  # WP8 (Key Goal 2)
    ],
    goal_offset_noise=[
      (0.0, 0.0),  # WP0
      (0.3, 0.2),  # WP1
      (0.3, 0.2),  # WP2
      (0.2, 0.2),  # WP3
      (0.2, 0.2),  # WP4
      (0.4, 0.3),  # WP5
      (0.4, 0.3),  # WP6
      (0.8, 0.5),  # WP7 (Key Goal 1)
      (0.8, 0.5),  # WP8 (Key Goal 2)
    ],
    key_goal_indices=[7, 8],
    target_velocity=0.8,
    target_velocity_range=(0.2, 1.5),  # indoor speed limit
    near_zero_probability=0.1,
    near_zero_velocity_range=(0.0, 0.1),
    near_zero_duration_range=(0.5, 1.0),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=10.0,
    timeout_speed_floor=0.2,
    timeout_min_s=20.0,
    timeout_max_s=120.0,
    reach_threshold=1.0,
    # Increased from 0.5 to 1.0m: at 1.5 m/s (R=1.5m) a 0.5m circle is
    # hard to hit.  1.0m < 1.25m (min waypoint spacing for v5) so no overlap.
    intermediate_reach_threshold=1.0,
    mirror_x_probability=0.5,
    # Increased from 0.8 to 1.5m: must be > intermediate_reach_threshold
    # (1.0m) + robot half-width.  bypass_x = 3 + 1.5 = 4.5m, well within
    # 14-16m indoor room (max_lateral ≈ 5.4m after room-bound clamping).
    min_obstacle_clearance=1.5,  # tighter clearance for indoor rooms
    # Velocity-adaptive waypoints with indoor turning parameters
    velocity_adaptive_offsets=True,
    max_angular_velocity=1.0,
    robot_distance_from_wall=5.0,  # match indoor terrain
    default_wall_half_width=3.0,  # match reduced v5 wall widths
    max_heading_change=math.pi / 1.9,
    debug_vis=True,
  )

  # Play mode: infinite episode length; training: dynamic timeout (up to 120s)
  cfg.episode_length_s = 1e10 if play else 120.0

  # Indoor rooms with furniture have many more contact geoms than outdoor.
  cfg.sim.nconmax = 256

  # Reduce spawn X range for indoor — wider offsets push bypass paths into
  # room walls on small (14m) rooms.
  if cfg.events is not None and "reset_base" in cfg.events:
    reset_base_cfg = cfg.events["reset_base"]
    if reset_base_cfg.params is not None and "pose_range" in reset_base_cfg.params:
      reset_base_cfg.params["pose_range"]["x"] = (-1.0, 1.0)  # ±1.0m indoor

  return cfg


def unitree_g1_indoor_straight_approach_reachable_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Indoor straight-approach local planner training config (v5, 0.5–1.2 m/s).

  Extends IndoorMixedScenario-v5 by switching from bell-curve bypass paths
  to causal straight-approach waypoints.  The robot walks straight toward the
  centre wall, turns >90° at d_turn=2.4m, drifts to d_safe=1.2m lateral
  clearance, travels along the wall, rounds the corner, then returns to goal.

  Terrain: same IndoorMixedScenario-v5 layout:
    35%: Indoor room + fixed 6m wall
    25%: Indoor room + random 2-6m wall
    20%: SingleWall (no outer room, v3 compat)
    20%: OpenField

  The velocity-adaptive waypoints recompute the bypass path from MuJoCo
  geometry at each episode reset, respecting room_bounds_rel lateral limits
  so no waypoint exits through the outer room walls.

  Usage:
    python -m mjlab.tasks.steering.train_local_planner --mode train \\
      --task Mjlab-IndoorStraightApproach-G1-v6 \\
      --checkpoint <pretrained_policy> \\
      --num_envs 1024 --output_dir logs/steering \\
      --experiment_name indoor_straight_approach_v5
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
  )

  # Base: inherit indoor terrain, nconmax=256, spawn ±1.0m X, events, rewards
  cfg = unitree_g1_indoor_mixed_scenario_reachable_h1_env_cfg(play=play)

  # Replace goal command with straight-approach waypoints.
  # Fallback offsets (v=0.8, dist_from_wall=5, wall_hw=3, d_turn=2.4, d_safe=1.2):
  #   turn_y = 5 - 2.4 = 2.6  |  R=0.8 → turn_end = (0.8, 3.4)
  #   drift_x = max(0.8, 1.2) = 1.2 → along_wall_y = 5 - 1.2 = 3.8
  #   corner_clearance = max(1.2*1.4, 1.2) = 1.68 ≈ 1.7
  #   wall_edge_x = 3.0 → corner_x = 4.7, past_wall_y = 6.2
  assert cfg.commands is not None
  cfg.commands["goal"] = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),  # fallback; overridden by dynamic_timeout
    goal_offsets=[
      # Fallback straight-approach path (v=0.8 m/s nominal)
      (0.0, 1.3),  # WP0: approach midpoint (turn_y / 2 = 1.3)
      (0.0, 2.6),  # WP1: turn trigger (wall_y - d_turn = 2.6)  ← NO noise
      (0.8, 3.4),  # WP2: turn end (R lateral)
      (2.2, 3.8),  # WP3: along wall (1.2m clearance from wall face)
      (3.0, 3.8),  # WP4: near wall edge (1.2m clearance)
      (4.7, 5.0),  # WP5: corner (wall_edge + corner_clearance=1.7, wall_y)
      (4.7, 6.2),  # WP6: past wall (corner_x, wall_y + d_safe)
      (0.0, 8.0),  # WP7 (Key Goal 1): wall_y + max(3, R*2) = 5+3
      (0.0, 10.5),  # WP8 (Key Goal 2): key1_y + max(2.5, R*1.5) = 8+2.5
    ],
    goal_offset_noise=[
      (0.0, 0.0),  # WP0
      (0.0, 0.0),  # WP1 — turn trigger must not be displaced
      (0.2, 0.2),  # WP2
      (0.3, 0.2),  # WP3
      (0.3, 0.2),  # WP4
      (0.3, 0.3),  # WP5
      (0.3, 0.3),  # WP6
      (0.8, 0.5),  # WP7 (Key Goal 1)
      (0.8, 0.5),  # WP8 (Key Goal 2)
    ],
    key_goal_indices=[7, 8],
    target_velocity=0.8,
    target_velocity_range=(0.5, 1.2),  # v_min raised; v_max = 1.2 m/s (d_turn = 2.4m)
    near_zero_probability=0.0,
    near_zero_velocity_range=(0.0, 0.1),
    near_zero_duration_range=(0.5, 1.0),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=10.0,
    timeout_speed_floor=0.5,
    timeout_min_s=20.0,
    timeout_max_s=120.0,
    reach_threshold=1.0,
    intermediate_reach_threshold=0.85,  # must be < d_safe (1.2m) to avoid skipping close waypoints
    mirror_x_probability=0.5,
    min_obstacle_clearance=1.2,  # d_safe = 1.2m (increased for safer wall clearance)
    velocity_adaptive_offsets=True,
    max_angular_velocity=1.0,
    robot_distance_from_wall=5.0,  # match indoor terrain
    default_wall_half_width=3.0,  # match reduced v5 wall widths
    max_heading_change=math.pi / 1.9,
    straight_approach=True,  # ★ causal avoidance sequence
    turn_threshold=2.4,  # ★ d_turn = v_max/ω + d_safe = 1.2 + 1.2 = 2.4m
    debug_vis=True,
  )

  # Play mode: infinite episode length; training: dynamic timeout (up to 120s)
  cfg.episode_length_s = 1e10 if play else 120.0

  # No joint-position randomisation on reset for straight-approach steering.
  if cfg.events is not None:
    cfg.events.pop("reset_robot_joints", None)

  if play:
    cfg.episode_length_s = 1e10

  return cfg


def unitree_g1_indoor_straight_approach_reachable_h1_stage4_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-4 variant with near-field raycast occlusion domain randomization
  and time-to-stop safety velocity envelope."""
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  cfg = unitree_g1_indoor_straight_approach_reachable_h1_env_cfg(play=play)
  if not play:
    _enable_near_occlusion_dr(cfg)
    _enable_spike_dr(cfg)

  assert cfg.commands is not None
  goal_cmd = cfg.commands["goal"]
  assert isinstance(goal_cmd, FixedSequentialGoalCommandCfg)
  goal_cmd.safety_velocity_envelope = ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
    enabled=True,
    forward_sensor_name="raycast",
    left_sensor_name="raycast_left",
    right_sensor_name="raycast_right",
    skip_ground_cols=3,
    safety_margin_m=0.25,
    reaction_time_s=0.15,
    braking_time_s=0.8,
    stop_velocity_epsilon=0.15,
    side_point_forward_offset_m=0.35,
    hard_stop_distance_m=0.40,
    blocked_turn_scale=0.0,
  )

  if play:
    _cap_play_goal_max_speed(cfg, 1.2)

  _apply_g1_reward_body(cfg)

  return cfg


def unitree_g1_indoor_straight_approach_reachable_nopp_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Indoor straight-approach local planner testing config (v5, NO path planning).

  Same indoor terrain as the training task. Robot must rely solely on trained
  local planner steering for obstacle avoidance. Two key goals placed to match the
  training config: 8m (wall_y=5 + 3) and 10.5m ahead of spawn.

  Usage:
    python -m mjlab.tasks.steering.train_local_planner --mode test \\
      --task Mjlab-IndoorStraightApproach-NoPP-G1-v6 \\
      --checkpoint <pretrained_policy> \\
      --local_planner_checkpoint logs/steering/local_planner_indoor_straight_approach_v5.pt \\
      --num_envs 1 --viewer native
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  cfg = unitree_g1_indoor_straight_approach_reachable_h1_env_cfg(play=play)

  assert cfg.commands is not None
  cfg.commands["goal"] = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(60.0, 60.0),
    goal_offsets=[
      (0.0, 8.0),  # Key Goal 1: matches training WP7 (wall_y=5 + 3)
      (0.0, 10.5),  # Key Goal 2: matches training WP8
    ],
    key_goal_indices=[0, 1],
    target_velocity=0.8,
    target_velocity_range=(0.5, 1.2),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=50.0,
    timeout_speed_floor=0.5,
    timeout_min_s=45.0,
    timeout_max_s=300.0,
    reach_threshold=1.0,
    debug_vis=True,
    safety_velocity_envelope=ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
      enabled=True,
      forward_sensor_name="raycast",
      left_sensor_name="raycast_left",
      right_sensor_name="raycast_right",
      skip_ground_cols=3,
      safety_margin_m=0.25,
      reaction_time_s=0.15,
      braking_time_s=0.8,
      stop_velocity_epsilon=0.15,
      side_point_forward_offset_m=0.35,
      hard_stop_distance_m=0.40,
      blocked_turn_scale=0.0,
    ),
  )

  cfg.episode_length_s = 1e10 if play else 300.0

  _apply_g1_reward_body(cfg)

  return cfg


# ==============================================================================
# Low-Speed Straight-Approach (v3) — local planner training with straight-approach
# waypoints at v_max=1.2 m/s.  The turning radius at max speed R=1.2m gives
# d_turn=2.4m with d_safe=1.2m safety margin.
#
# Mathematical constraint (ω_max=1.0, d_safe=1.2m):
#   v_max=1.2 m/s → d_turn = v/ω + d_safe = 1.2 + 1.2 = 2.4 m
#   At d_turn=2.4m: alpha = sigmoid((2.4−3.5)×10) ≈ 0.0  (full avoidance mode)
#
# Uses same terrain as MixedScenario-v3 (8m/2-8m walls + OpenField).
# The close-range turn (2.4m) gives a strong raycast training signal.
# ==============================================================================
def unitree_g1_straight_approach_low_speed_reachable_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Low-speed (0.5–1.2 m/s) straight-approach local planner training config.

  Uses the same terrain as MixedScenario-v3 (8m fixed wall, 2-8m random,
  30% open field, 30×30m cells). With v_max=1.2 m/s, the turning radius at
  maximum speed is R=1.2m, requiring d_turn=2.4m with d_safe=1.2m clearance.

  The robot approaches within 2.4m of the wall before turning >90° and
  drifting to d_safe=1.2m lateral clearance. Minimum speed is raised to
  0.5 m/s to avoid degenerate near-stationary episodes.

  Mathematical constraint (d_safe=1.2m, ω_max=1.0 rad/s):
    d_turn = v_max/ω_max + d_safe = 1.2/1.0 + 1.2 = 2.4 m
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
  )

  # Inherit all settings from MixedScenario-v3 (same terrain, observations,
  # terminations, events, curriculum setup, nconmax).
  cfg = unitree_g1_mixed_scenario_reachable_h1_env_cfg(play=play)

  # Override ONLY the goal command with straight-approach behavior.
  # Fallback 9-WP path computed for v=0.8, d_turn=2.4, d_safe=1.2,
  # wall_y=6, wall_hw=4:
  #   turn_y = 3.6, R = 0.8
  #   turn_end = (0.8, 4.4)  drift → (1.2, 4.8)  along_wall_y = 4.8
  #   corner_clearance = max(1.2*1.4, 1.2) = 1.68 ≈ 1.7
  #   wall_edge = 4.0, corner = (5.7, 6.0), past = (5.7, 7.2)
  #   key1_y = 9.0   key2_y = 11.5
  assert cfg.commands is not None
  cfg.commands["goal"] = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),
    goal_offsets=[
      (0.0, 1.8),  # WP0: approach midpoint (turn_y / 2 = 1.8)
      (0.0, 3.6),  # WP1: turn point (wall_y − d_turn = 3.6)
      (0.8, 4.4),  # WP2: turn end (R, turn_y + R)
      (2.6, 4.8),  # WP3: mid along wall (1.2m clearance from long face)
      (4.0, 4.8),  # WP4: wall edge (1.2m clearance from long face)
      (5.7, 6.0),  # WP5: corner — face short edge at d_safe*1.4
      (5.7, 7.2),  # WP6: past wall (corner_x, wall_y + d_safe)
      (0.0, 9.0),  # WP7 (Key Goal 1): wall_y + max(3, R*2) = 9.0
      (0.0, 11.5),  # WP8 (Key Goal 2): key1 + max(2.5, R*1.5) = 11.5
    ],
    goal_offset_noise=[
      (0.0, 0.0),  # WP0
      (0.0, 0.0),  # WP1 (turn point — no noise for reliable trigger)
      (0.2, 0.2),  # WP2
      (0.3, 0.2),  # WP3
      (0.3, 0.2),  # WP4
      (0.3, 0.3),  # WP5 (corner)
      (0.3, 0.3),  # WP6 (past wall)
      (1.0, 0.5),  # WP7 (Key Goal 1)
      (1.0, 0.5),  # WP8 (Key Goal 2)
    ],
    key_goal_indices=[7, 8],
    target_velocity=0.8,
    target_velocity_range=(0.5, 1.2),  # v_min raised; v_max = 1.2 m/s (d_turn = 2.4m)
    near_zero_probability=0.0,
    near_zero_velocity_range=(0.0, 0.1),
    near_zero_duration_range=(0.5, 1.0),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=10.0,
    timeout_speed_floor=0.5,
    timeout_min_s=20.0,
    timeout_max_s=120.0,
    reach_threshold=1.0,
    intermediate_reach_threshold=0.73,  # must be < d_safe (1.2m) to avoid skipping close waypoints
    mirror_x_probability=0.5,
    min_obstacle_clearance=1.2,  # d_safe = 1.2m (increased for safer wall clearance)
    velocity_adaptive_offsets=True,
    max_angular_velocity=1.0,
    robot_distance_from_wall=6.0,
    default_wall_half_width=4.0,  # original 8m wall (restored)
    max_heading_change=math.pi / 1.9,
    straight_approach=True,  # straight-approach path generation
    turn_threshold=2.4,  # d_turn = v/ω + d_safe = 1.2 + 1.2 = 2.4m
    debug_vis=True,
  )

  # No joint-position randomisation on reset for straight-approach steering.
  if cfg.events is not None:
    cfg.events.pop("reset_robot_joints", None)

  if play:
    cfg.episode_length_s = 1e10

  return cfg


def unitree_g1_straight_approach_low_speed_reachable_h1_stage4_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Stage-4 variant with near-field raycast occlusion domain randomization
  and time-to-stop safety velocity envelope."""
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  cfg = unitree_g1_straight_approach_low_speed_reachable_h1_env_cfg(play=play)
  if not play:
    _enable_near_occlusion_dr(cfg)
    _enable_spike_dr(cfg)

  assert cfg.commands is not None
  goal_cmd = cfg.commands["goal"]
  assert isinstance(goal_cmd, FixedSequentialGoalCommandCfg)
  goal_cmd.safety_velocity_envelope = ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
    enabled=True,
    forward_sensor_name="raycast",
    left_sensor_name="raycast_left",
    right_sensor_name="raycast_right",
    skip_ground_cols=3,
    safety_margin_m=0.25,
    reaction_time_s=0.15,
    braking_time_s=0.8,
    stop_velocity_epsilon=0.15,
    side_point_forward_offset_m=0.35,
    hard_stop_distance_m=0.40,
    blocked_turn_scale=0.0,
  )

  if play:
    _cap_play_goal_max_speed(cfg, 1.2)

  _apply_g1_reward_body(cfg)

  return cfg


def unitree_g1_straight_approach_low_speed_reachable_nopp_h1_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Low-speed straight-approach local planner testing config (NO path planning).

  Same terrain as the training task but with only two key goal positions.
  Robot must rely solely on trained local planner steering for obstacle avoidance.
  """
  from mjlab.tasks.velocity.mdp.reachable_goal_command import (
    FixedSequentialGoalCommandCfg,
    ReachableGoalCommandCfg,
  )

  cfg = unitree_g1_straight_approach_low_speed_reachable_h1_env_cfg(play=play)

  assert cfg.commands is not None
  cfg.commands["goal"] = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(60.0, 60.0),
    goal_offsets=[
      (0.0, 9.0),  # Key Goal 1: wall_y + max(3, R*2) at typical speed
      (0.0, 11.5),  # Key Goal 2: final goal
    ],
    key_goal_indices=[0, 1],
    target_velocity=0.8,
    target_velocity_range=(0.5, 1.2),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=50.0,
    timeout_speed_floor=0.5,
    timeout_min_s=45.0,
    timeout_max_s=300.0,
    reach_threshold=1.0,
    debug_vis=True,
    safety_velocity_envelope=ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
      enabled=True,
      forward_sensor_name="raycast",
      left_sensor_name="raycast_left",
      right_sensor_name="raycast_right",
      skip_ground_cols=3,
      safety_margin_m=0.25,
      reaction_time_s=0.15,
      braking_time_s=0.8,
      stop_velocity_epsilon=0.15,
      side_point_forward_offset_m=0.35,
      hard_stop_distance_m=0.40,
      blocked_turn_scale=0.0,
    ),
  )

  cfg.episode_length_s = 1e10 if play else 300.0

  _apply_g1_reward_body(cfg)

  return cfg
