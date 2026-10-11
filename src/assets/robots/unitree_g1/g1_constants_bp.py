"""Unitree G1 机器人原生位置执行器基线常量配置 (Unitree G1 Builtin Position Actuator Baseline Constants)。

【系统定位与文件作用】
本文件是 `g1_constants.py` 的基线对照版本（BP = Builtin Position / Baseline）：
- 与 `g1_constants.py` 的核心区别在于：本文件完全采用 MuJoCo 原生线性 `<position>` 驱动模型
  (`BuiltinPositionActuatorCfg`)，不引入非线性 T-N 扭矩-转速包络截断与 Stribeck 摩擦力模型。
- 应用场景：用于控制算法对照基线 (Baseline)、消融实验 (Ablation Study)、或对物理仿真执行效率
  有极端高频要求的初级阶段强化学习预训练。
- 其余物理建模（两级行星齿轮折算惯量、二阶系统 PD 增益推导、关键帧姿态、足底摩擦锥及动作缩放）
  与 `g1_constants.py` 保持完全一致的数学逻辑。
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

# ==============================================================================
# MJCF 模型路径与模型规范 (MJCF XML Model Path & Spec)
# ==============================================================================

# G1 全自由度机器人 MJCF XML 描述文件绝对路径
G1_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)
assert G1_XML.exists(), f"G1 XML 文件不存在: {G1_XML}"


def get_spec() -> mujoco.MjSpec:
  """加载并返回 Unitree G1 的 MuJoCo MjSpec 规范对象。

  在 mjlab 1.4.0+ 中，网格模型通过 XML 中的 `meshdir` 在编译阶段直接从磁盘解析。
  """
  return mujoco.MjSpec.from_file(str(G1_XML))


# ==============================================================================
# 电机物理规格与等效电枢转动惯量折算 (Motor Specs & Reflected Inertias)
# ==============================================================================

# 5020 电机模块：
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

# 7520-14 电机模块 (减速比 14.3)：
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

# 7520-22 电机模块 (减速比 22.5)：
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

# 4010 电机模块：
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

# 5010-16 电机模块：
ARMATURE_5010_16 = 0.0021812

# 电机参数容器 (等效惯量、极限角速度 rad/s、极限输出力矩 N·m)
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
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)
ACTUATOR_5010_16 = ElectricActuator(
  reflected_inertia=ARMATURE_5010_16,
  velocity_limit=22.0,
  effort_limit=10.0,
)

# ==============================================================================
# PD 控制增益推导 (二阶系统参数: 刚度 Kp 与 阻尼 Kd)
# ==============================================================================
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 固有频率 10Hz
DAMPING_RATIO = 2.0                    # 阻尼比 2.0 (过阻尼系统)

# 刚度 K_p = I * omega_n^2
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
STIFFNESS_5010_16 = ARMATURE_5010_16 * NATURAL_FREQ**2

# 阻尼 K_d = 2 * zeta * I * omega_n
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
DAMPING_5010_16 = 2.0 * DAMPING_RATIO * ARMATURE_5010_16 * NATURAL_FREQ

# ==============================================================================
# 原生位置执行器分组配置 (Builtin Position Actuators)
# ==============================================================================

# 手臂及手腕翻滚关节
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

# 髋航向与腰航向关节
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

# 腿部承重关节（髋俯仰、髋侧倾、膝关节）
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

# 手腕俯仰与手腕偏航关节
G1_ACTUATOR_5010_16 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_5010_16,
  damping=DAMPING_5010_16,
  effort_limit=ACTUATOR_5010_16.effort_limit,
  armature=ACTUATOR_5010_16.reflected_inertia,
)

# 腰部与踝部双电机并联机构（标称 1:1 传动比，参数加倍）
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

# ==============================================================================
# 机器人关键帧姿态 (Keyframe Configurations)
# ==============================================================================

# 标准直立站立关键帧
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

# 屈膝微蹲关键帧（默认训练初始姿势）
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.78),
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

# ==============================================================================
# 碰撞与接触条件配置 (Collision and Contact Dimension Config)
# ==============================================================================

# 包含自碰撞的全碰撞模式 (足底 condim=3 支持摩擦锥，连杆自碰 condim=1)
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# 仅与环境发生碰撞模式（屏蔽连杆自碰撞）
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# 仅足底碰撞模式
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# ==============================================================================
# 机器人关节铰接与实体完整配置 (Articulation and Entity Configuration)
# ==============================================================================

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_5010_16,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_robot_cfg() -> EntityCfg:
  """获取全新的 Unitree G1 原生执行器基线配置实例。"""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


# ==============================================================================
# 动作空间缩放比例映射 (RL Action Scaling Mapping)
# ==============================================================================
G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
