"""Unitree G1 constants."""

import os as _os
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

G1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists()

G1_N3_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1_n3_v5.xml"
)
assert G1_N3_XML.exists()

G1_V5_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1_v5.xml"
)
assert G1_V5_XML.exists()

G1_N3_V15_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1_n3_v15.xml"
)
assert G1_N3_V15_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_n3_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_N3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_v5_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_V5_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_n3_v15_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_N3_V15_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

# Motor specs (from Unitree).
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
# V5 real-machine aligned wrist effort limit uses +/-5 Nm in XML.
# Reflected inertia is approximated with the previous 4010 estimate until a
# dedicated identification model is added.
ACTUATOR_5010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=27.0,
  effort_limit=5.0,
)
# Backward-compatible alias name kept for existing imports/tests.
ACTUATOR_4010 = ACTUATOR_5010

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_5010 = ARMATURE_4010 * NATURAL_FREQ**2
STIFFNESS_4010 = STIFFNESS_5010

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_5010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
DAMPING_4010 = DAMPING_5010

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_5010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_5010,
  damping=DAMPING_5010,
  effort_limit=ACTUATOR_5010.effort_limit,
  armature=ACTUATOR_5010.reflected_inertia,
)
# Backward-compatible alias name kept for existing imports/tests.
G1_ACTUATOR_4010 = G1_ACTUATOR_5010

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "right_shoulder_roll_joint": -0.18,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22_HIP_PITCH,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_5010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_n3_robot_cfg() -> EntityCfg:
  """Get a fresh G1-N3 robot configuration instance.

  Uses g1_n3_v5.xml which has left/right camera sites on the torso_link
  (above shoulder joints, below head, avoids arm occlusion during walking).
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_n3_spec,
    articulation=G1_ARTICULATION,
  )


def get_g1_n3_v5_robot_cfg() -> EntityCfg:
  """Get a fresh G1-N3 v5 robot configuration instance.

  Uses g1_n3_v5.xml with corrected wrist inertial parameters and hip_pitch
  assigned to the 7520_14 motor group matching the official URDF. Retains
  left/right camera sites on torso_link required by obstacle_avoidance tasks.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_n3_spec,
    articulation=G1_V5_ARTICULATION,
  )


##
# V5 real-machine aligned configuration.
#
# Differences from the default G1_ARTICULATION:
# - hip_pitch uses 7520_14 motor (effort=88 Nm, velocity=32 rad/s) matching the
#   official g1_29dof_rev_1_0 URDF, instead of the 7520_22 armature override.
# - Uses g1_v5.xml which has corrected wrist inertial parameters matching the
#   official model, collision geometry, and no camera sites.
##

G1_ACTUATOR_7520_14_V5 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

G1_V5_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14_V5,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_5010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_v5_robot_cfg() -> EntityCfg:
  """Get a fresh G1-V5 robot configuration aligned with the real machine.

  Uses g1_v5.xml with corrected wrist inertial parameters and hip_pitch
  assigned to the 7520_14 motor group matching the official URDF.
  Initial state uses HOME_KEYFRAME (height=0.8m) consistent with the
  successfully deployed configuration.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_v5_spec,
    articulation=G1_V5_ARTICULATION,
  )


G1_V5_ACTION_SCALE: dict[str, float] = {}
for _a in G1_V5_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    G1_V5_ACTION_SCALE[_n] = 0.25 * _e / _s


##
# V15 (mode_15) configuration.
#
# Differences from V5:
# - hip_pitch uses 7520_22 motor (effort=139 Nm, velocity=20 rad/s) with full
#   7520_22 armature — matching the official g1_29dof_mode_15 URDF.
#   (V5 used 7520_14 motor with effort=88 Nm for hip_pitch.)
# - wrist_pitch/yaw uses 5010(new) motor (effort=13.4 Nm, velocity=27 rad/s).
#   (V5 used effort=5.0 Nm.)
# - Uses g1_n3_v15.xml with 5010 wrist meshes and mode_15 wrist inertials.
##

ACTUATOR_5010_V15 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=27.0,
  effort_limit=13.4,
)

STIFFNESS_5010_V15 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5010_V15 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5010_V15 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_5010_V15,
  damping=DAMPING_5010_V15,
  effort_limit=ACTUATOR_5010_V15.effort_limit,
  armature=ACTUATOR_5010_V15.reflected_inertia,
)

# hip_pitch in mode_15 uses 7520_22 motor with full effort (139 Nm).
G1_ACTUATOR_7520_22_HIP_PITCH_V15 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

G1_ACTUATOR_7520_14_V15 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

G1_V15_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14_V15,
    G1_ACTUATOR_7520_22_HIP_PITCH_V15,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_5010_V15,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_n3_v15_robot_cfg() -> EntityCfg:
  """Get a fresh G1-N3 v15 (mode_15) robot configuration instance.

  Uses g1_n3_v15.xml with hip_pitch assigned to the full 7520_22 motor
  group (effort=139 Nm) and 5010(new) wrist motors (effort=13.4 Nm).
  Retains left/right camera sites on torso_link for obstacle avoidance.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_n3_v15_spec,
    articulation=G1_V15_ARTICULATION,
  )


G1_V15_ACTION_SCALE: dict[str, float] = {}
for _a15 in G1_V15_ARTICULATION.actuators:
  assert isinstance(_a15, BuiltinPositionActuatorCfg)
  _e15 = _a15.effort_limit
  _s15 = _a15.stiffness
  _names15 = _a15.target_names_expr
  assert _e15 is not None
  for _n15 in _names15:
    G1_V15_ACTION_SCALE[_n15] = 0.25 * _e15 / _s15


G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


##
# Version switching.
#
# Set environment variable G1_ROBOT_VERSION to "v5" or "v15" (default: "v15")
# to control which robot model/articulation is used by active-config helpers.
##

G1_ROBOT_VERSION: str = _os.environ.get("G1_ROBOT_VERSION", "v15")

_VALID_VERSIONS = ("v5", "v15")
if G1_ROBOT_VERSION not in _VALID_VERSIONS:
  raise ValueError(
    f"Invalid G1_ROBOT_VERSION={G1_ROBOT_VERSION!r}, expected one of {_VALID_VERSIONS}"
  )


def get_active_g1_n3_robot_cfg() -> EntityCfg:
  """Get robot config for the active G1 version (v5 or v15).

  Controlled by the G1_ROBOT_VERSION environment variable.
  """
  if G1_ROBOT_VERSION == "v15":
    return get_g1_n3_v15_robot_cfg()
  return get_g1_n3_v5_robot_cfg()


def get_active_g1_action_scale() -> dict[str, float]:
  """Get action scale dict for the active G1 version (v5 or v15)."""
  if G1_ROBOT_VERSION == "v15":
    return G1_V15_ACTION_SCALE
  return G1_V5_ACTION_SCALE


def get_active_g1_articulation() -> EntityArticulationInfoCfg:
  """Get articulation config for the active G1 version (v5 or v15)."""
  if G1_ROBOT_VERSION == "v15":
    return G1_V15_ARTICULATION
  return G1_V5_ARTICULATION


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
