"""Classroom navigation task: PAC-MAN classroom layout ported to LP-NavOA.

The scene replicates the classroom used by the PAC-MAN project
(perceptive_cbf_rl, ``src/assets/objects/classroom.py``): a walled room with a
5x3 grid of desk+chair pairs and uniform 1.5 m aisles. It is used to retrain
and evaluate the LP-NavOA local planner in the same scenario as PAC-MAN for a
side-by-side comparison.

Coordinate convention: LP-NavOA navigation tasks walk along +Y. The PAC-MAN
classroom's aisles run along its X axis, so the layout is rotated 90 degrees:
LP forward ``v`` = PAC-MAN ``x``, LP lateral ``u`` = PAC-MAN ``y``. Room walls
(centre lines): v in [-0.83, 16.33], u in [-6.23, 3.53]; the robot spawns in
the front cross-aisle (v = 0.2) on one of the four longitudinal lanes
(u in {-5.4, -2.7, 0.0, 2.7}) and walks the lane to the back cross-aisle.

Commanded cruise speed is 0.6 m/s (``target_velocity_range=(0.4, 0.8)``),
matching the PAC-MAN classroom cruise for a fair comparison.

Waypoints are EXPLICIT and fixed (``velocity_adaptive_offsets=False``,
``min_obstacle_clearance=0.0``). Do NOT enable the velocity-adaptive waypoint
generation here: its wall auto-detection (``_extract_wall_aabbs`` /
``_find_room_bounds`` in reachable_goal_command.py) accepts any geom AABB with
min-half-extent < 0.5 m as "wall-like", which the desks (0.4x0.6) satisfy, so
adaptive generation would treat the furniture grid as walls and corrupt the
room-bounds estimate.

Upgrade path if straight lanes prove too easy: subclass
``FixedSequentialGoalCommand`` to sample per-env S-routes (lane changes via
the cross-aisles) at reset, keeping explicit waypoint lists per route.

NOTE on running this copy: the venv is shared with the original LP-NavOA
checkout via symlink, and the venv's ``mjlab.pth`` points at the ORIGINAL
sources. Always run from this copy with::

  export PYTHONPATH=<this-copy>/src
  uv run --no-sync ...

Never run bare ``uv run`` / ``uv sync`` in this copy -- it would rewrite the
shared venv's ``mjlab.pth`` and break the original project.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.mdp.reachable_goal_command import (
  FixedSequentialGoalCommandCfg,
  ReachableGoalCommandCfg,
)
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

from .env_cfgs import (
  _apply_g1_reward_body,
  _cap_play_goal_max_speed,
  _enable_near_occlusion_dr,
  _enable_spike_dr,
  _estimate_g1_height,
  unitree_g1_indoor_straight_approach_reachable_h1_env_cfg,
)

# ── Classroom layout (rotated frame: u = lateral, v = forward) ───────────────
# Ported from perceptive_cbf_rl src/assets/objects/classroom.py.
CLASSROOM_LANES_U = (-5.4, -2.7, 0.0, 2.7)
"""Centres of the four longitudinal (v-direction) 1.5 m aisles."""

CLASSROOM_COLUMN_U = (-4.05, -1.35, 1.35)
"""Lateral positions of the three desk/chair columns."""

CLASSROOM_ROW_V = tuple(1.95 + 3.1 * k for k in range(5))
"""Forward positions of the five desk rows (chairs sit 0.95 m behind)."""

_DESK_HALF_SIZE = (0.6, 0.4, 0.38)  # (u, v, z); long side lateral
_CHAIR_HALF_SIZE = (0.25, 0.25, 0.45)
_CHAIR_OFFSET_V = 0.95  # chair centre this far behind (negative v) its desk
_WALL_HALF_THICKNESS = 0.08
_WALL_HEIGHT = 3.0

# Wall centre lines.
_ROOM_U = (-6.23, 3.53)  # west, east
_ROOM_V = (-0.83, 16.33)  # south (behind spawn), north (ahead)

# Terrain cell mapping: cell (12 x 20) centres the room exactly.
_U_SHIFT = 7.35
_V_SHIFT = 2.25
_SPAWN_V = 0.2  # front cross-aisle


@dataclass(kw_only=True)
class ClassroomTerrainCfg(SubTerrainCfg):
  """Walled classroom: 4 perimeter walls + 5x3 grid of desk/chair boxes.

  Layout is deterministic (every cell is the same classroom); the only per-cell
  randomisation is which of the four lanes the robot spawns on. Desks and
  chairs are collision-only boxes -- the 4 m raycast range does not need the
  visual detail (legs, desktop) of the PAC-MAN original.
  """

  robot_height: float
  """Estimated robot standing height [m] (unused; kept for cfg symmetry)."""

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

    wall_color = (0.55, 0.50, 0.45, 1.0)
    floor_color = (0.45, 0.45, 0.5, 1.0)
    desk_color = (0.45, 0.35, 0.25, 1.0)
    chair_color = (0.35, 0.30, 0.25, 1.0)

    def _add_box(pos, size, color) -> None:
      g = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        rgba=color,
      )
      g.contype = 1
      g.conaffinity = 1
      g.friction = np.array(self.friction)
      geometries.append(TerrainGeometry(geom=g, color=color))

    # ── Ground plane ──────────────────────────────────────────────────────
    _add_box((cx, cy, -0.02), (tx / 2.0, ty / 2.0, 0.02), floor_color)

    # ── Perimeter walls (centre lines from PAC-MAN constants) ─────────────
    hwt = _WALL_HALF_THICKNESS
    wz = _WALL_HEIGHT / 2.0
    u_mid = sum(_ROOM_U) / 2.0 + _U_SHIFT  # 6.0
    v_mid = sum(_ROOM_V) / 2.0 + _V_SHIFT  # 10.0
    # South / North walls span the full width including the corner overlaps.
    sn_half_u = (_ROOM_U[1] - _ROOM_U[0]) / 2.0 + hwt  # 4.96
    _add_wall_s = (u_mid, _ROOM_V[0] + _V_SHIFT, wz)
    _add_box(_add_wall_s, (sn_half_u, hwt, wz), wall_color)
    _add_box((u_mid, _ROOM_V[1] + _V_SHIFT, wz), (sn_half_u, hwt, wz), wall_color)
    # West / East walls span the full length including the corner overlaps.
    we_half_v = (_ROOM_V[1] - _ROOM_V[0]) / 2.0 + hwt  # 8.66
    _add_box((_ROOM_U[0] + _U_SHIFT, v_mid, wz), (hwt, we_half_v, wz), wall_color)
    _add_box((_ROOM_U[1] + _U_SHIFT, v_mid, wz), (hwt, we_half_v, wz), wall_color)

    # ── Desk/chair grid (5 rows x 3 columns) ──────────────────────────────
    for u in CLASSROOM_COLUMN_U:
      for v in CLASSROOM_ROW_V:
        _add_box(
          (u + _U_SHIFT, v + _V_SHIFT, _DESK_HALF_SIZE[2]),
          _DESK_HALF_SIZE,
          desk_color,
        )
        _add_box(
          (
            u + _U_SHIFT,
            v - _CHAIR_OFFSET_V + _V_SHIFT,
            _CHAIR_HALF_SIZE[2],
          ),
          _CHAIR_HALF_SIZE,
          chair_color,
        )

    # ── Spawn: front cross-aisle, on a randomly chosen lane ───────────────
    lane_u = CLASSROOM_LANES_U[int(rng.integers(0, len(CLASSROOM_LANES_U)))]
    origin = np.array((lane_u + _U_SHIFT, _SPAWN_V + _V_SHIFT, 0.0))
    return TerrainOutput(origin=origin, geometries=geometries)


def _make_classroom_generator_cfg(robot_height: float, play: bool) -> TerrainGeneratorCfg:
  """Single-terrain classroom grid: 3x4 = 12 cells, one classroom per cell."""
  return TerrainGeneratorCfg(
    size=(12.0, 20.0),
    border_width=2.0,
    num_rows=3,
    num_cols=4,
    curriculum=False,
    sub_terrains={
      "classroom": ClassroomTerrainCfg(
        proportion=1.0,
        robot_height=robot_height,
      ),
    },
    difficulty_range=(0.0, 1.0),
    add_lights=True,
  )


def _classroom_goal_command_cfg(play: bool) -> FixedSequentialGoalCommandCfg:
  """Straight-lane teacher waypoints down the spawn lane, 0.6 m/s cruise.

  All waypoints sit on the lane centreline (x-offset 0), so no mirroring and
  no clearance enforcement are needed. Key goals: WP7 just past the last desk
  row, WP8 in the back cross-aisle -- the NoPP twin targets exactly these two.
  """
  return FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(30.0, 30.0),  # fallback; overridden by dynamic_timeout
    goal_offsets=[
      (0.0, 2.0),  # WP0
      (0.0, 4.0),  # WP1
      (0.0, 6.0),  # WP2
      (0.0, 8.0),  # WP3
      (0.0, 10.0),  # WP4
      (0.0, 12.0),  # WP5
      (0.0, 13.5),  # WP6
      (0.0, 14.2),  # WP7 (Key Goal 1): just past the last desk row (v=14.4)
      (0.0, 15.0),  # WP8 (Key Goal 2): back cross-aisle (v=15.2)
    ],
    goal_offset_noise=[
      (0.0, 0.0),  # WP0
      (0.0, 0.0),  # WP1
      (0.0, 0.0),  # WP2
      (0.0, 0.0),  # WP3
      (0.0, 0.0),  # WP4
      (0.0, 0.0),  # WP5
      (0.0, 0.0),  # WP6
      (0.3, 0.5),  # WP7 (Key Goal 1) — x noise < 0.75 m lane half-width
      (0.3, 0.5),  # WP8 (Key Goal 2)
    ],
    key_goal_indices=[7, 8],
    target_velocity=0.6,  # PAC-MAN classroom cruise speed
    target_velocity_range=(0.4, 0.8),
    near_zero_probability=0.0,
    near_zero_velocity_range=(0.0, 0.1),
    near_zero_duration_range=(0.5, 1.0),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=10.0,
    timeout_speed_floor=0.4,
    timeout_min_s=20.0,
    timeout_max_s=120.0,
    reach_threshold=1.0,
    intermediate_reach_threshold=1.0,  # < 2.0 m waypoint spacing: no overlap
    mirror_x_probability=0.0,  # all x-offsets are 0; lanes already give diversity
    # CRITICAL: keep both disabled. Waypoints are exact for the deterministic
    # layout, and the adaptive/clearance machinery mis-classifies desks as
    # walls (see module docstring).
    min_obstacle_clearance=0.0,
    velocity_adaptive_offsets=False,
    debug_vis=True,
  )


def _safety_velocity_envelope_cfg() -> ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg:
  """Same time-to-stop safety envelope as the indoor stage-4 task."""
  return ReachableGoalCommandCfg.SafetyVelocityEnvelopeCfg(
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


def unitree_g1_classroom_reachable_h1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Classroom local-planner config: classroom terrain + straight-lane waypoints.

  Inherits everything (robot, raycast sensors, observations, rewards, events,
  nconmax=256, episode_length_s=120) from the indoor straight-approach chain
  and swaps in the classroom terrain generator and goal command.

  Usage:
    python -m mjlab.tasks.steering.train_local_planner --mode train \\
      --task Mjlab-Classroom-G1-v6 \\
      --checkpoint <pretrained_backbone> \\
      --num-envs 1024 --output-dir logs/steering \\
      --experiment-name classroom_v1
  """
  cfg = unitree_g1_indoor_straight_approach_reachable_h1_env_cfg(play=play)

  # Swap terrain for the classroom.
  terrain_cfg = cfg.scene.terrain
  assert terrain_cfg is not None
  terrain_cfg.terrain_generator = _make_classroom_generator_cfg(
    robot_height=_estimate_g1_height(),
    play=play,
  )

  # Swap the goal command for the classroom lane route.
  assert cfg.commands is not None
  cfg.commands["goal"] = _classroom_goal_command_cfg(play)

  # Keep the spawn jitter inside the 1.5 m lane (half-width 0.75 m).
  if cfg.events is not None and "reset_base" in cfg.events:
    reset_base_cfg = cfg.events["reset_base"]
    if reset_base_cfg.params is not None and "pose_range" in reset_base_cfg.params:
      reset_base_cfg.params["pose_range"]["x"] = (-0.3, 0.3)
      reset_base_cfg.params["pose_range"]["y"] = (-0.3, 0.3)

  return cfg


def unitree_g1_classroom_reachable_h1_stage4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Classroom with raycast DR and the time-to-stop safety velocity envelope.

  This is the registered teacher task (``Mjlab-Classroom-G1-v6``) that the
  local planner is distilled on.
  """
  cfg = unitree_g1_classroom_reachable_h1_env_cfg(play=play)
  if not play:
    _enable_near_occlusion_dr(cfg)
    _enable_spike_dr(cfg)

  assert cfg.commands is not None
  goal_cmd = cfg.commands["goal"]
  assert isinstance(goal_cmd, FixedSequentialGoalCommandCfg)
  goal_cmd.safety_velocity_envelope = _safety_velocity_envelope_cfg()

  if play:
    _cap_play_goal_max_speed(cfg, 0.8)  # classroom speed ceiling

  _apply_g1_reward_body(cfg)

  return cfg


def unitree_g1_classroom_reachable_nopp_h1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Planner-free classroom twin (``Mjlab-Classroom-NoPP-G1-v6``).

  Same terrain, but the command carries only the two key goals, so the frozen
  backbone is steered solely by the learned local planner. Used for
  deployment-style evaluation against PAC-MAN.

  Usage:
    python -m mjlab.tasks.steering.train_local_planner --mode test \\
      --task Mjlab-Classroom-NoPP-G1-v6 \\
      --checkpoint <pretrained_backbone> \\
      --local-planner-checkpoint logs/steering/local_planner_classroom_v1_final.pt \\
      --num-envs 4
  """
  cfg = unitree_g1_classroom_reachable_h1_env_cfg(play=play)

  assert cfg.commands is not None
  cfg.commands["goal"] = FixedSequentialGoalCommandCfg(
    entity_name="robot",
    resampling_time_range=(60.0, 60.0),
    goal_offsets=[
      (0.0, 14.2),  # Key Goal 1: matches training WP7
      (0.0, 15.0),  # Key Goal 2: matches training WP8
    ],
    key_goal_indices=[0, 1],
    target_velocity=0.6,
    target_velocity_range=(0.4, 0.8),
    dynamic_timeout_by_velocity=True,
    timeout_margin_s=50.0,
    timeout_speed_floor=0.4,
    timeout_min_s=45.0,
    timeout_max_s=300.0,
    reach_threshold=1.0,
    debug_vis=True,
    safety_velocity_envelope=_safety_velocity_envelope_cfg(),
  )

  cfg.episode_length_s = 1e10 if play else 300.0

  _apply_g1_reward_body(cfg)

  return cfg
