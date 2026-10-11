"""Unitree G1 23 自由度精简版仿人机器人常量与配置模块 (Unitree G1 23-DOF Humanoid Constants and Configuration Module)。

【系统定位与文件作用】
本模块定义了宇树 G1 仿人机器人的 23 自由度 (23-DOF) 精简版模型配置与控制常量：
1. **模型拓扑与自由度裁剪 (DOF Reduction)**:
   相较于 29 自由度全构型，23 自由度模型锁定了部分末端手腕自由度（腕部俯仰与偏航）以及腰部复杂双电机俯仰/侧滚差动机构，
   仅保留对全身下肢平衡、双足行走步态和上肢基本摆臂最关键的 23 个自由度。
   这显著降低了强化学习动作探索空间的维度，能够显著加速运动先验 (AMP) 模仿学习与下肢导航策略的收敛。
2. **执行器分配与 PD 增益配置**:
   - 臂部/手腕侧滚：5020 电机
   - 髋俯仰/偏航及腰偏航：7520-14 电机
   - 髋侧滚与膝关节：7520-22 高力矩电机
   - 踝部并联四连杆：双 5020 电机等效驱动
3. **碰撞掩码与动作空间缩放**:
   提供高保真足底接触摩擦锥定义与力矩-刚度动作空间映射。
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

# G1 23-DOF 机器人 MJCF XML 描述文件绝对路径
G1_23DOF_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml"
)
assert G1_23DOF_XML.exists(), f"G1 23-DOF XML 文件不存在: {G1_23DOF_XML}"


def get_spec() -> mujoco.MjSpec:
  """加载并返回 Unitree G1 23-DOF 的 MuJoCo MjSpec 规范对象。

  在 mjlab 1.4.0+ 中，网格模型通过 XML 中的 `meshdir` 在编译阶段直接从磁盘解析。
  """
  return mujoco.MjSpec.from_file(str(G1_23DOF_XML))


# ==============================================================================
# 执行器电枢转动惯量与电机规格 (Actuator Motor Specs & Armature Inertias)
# ==============================================================================

# 5020 电机模块 (手臂及脚踝并联驱动单元)
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

# 7520-14 电机模块 (减速比 14.3，用于髋俯仰、髋航向及腰航向)
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

# 7520-22 电机模块 (减速比 22.5，高力矩，用于髋侧滚及膝关节)
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

# 4010 电机模块
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

# 电机参数容器 (等效惯量、角速度上限 rad/s、输出力矩上限 N·m)
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

# ==============================================================================
# PD 控制增益推导 (二阶振荡系统参数: 刚度 Kp 与 阻尼 Kd)
# ==============================================================================
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 固有频率 10Hz
DAMPING_RATIO = 2.0                    # 阻尼比 2.0 (过阻尼系统，防止振荡发散)

# 刚度系数推导公式: K_p = I * omega_n^2
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# 阻尼系数推导公式: K_d = 2 * zeta * I * omega_n
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# ==============================================================================
# 23-DOF 执行器分组与配置实例化 (23-DOF Actuator Configurations)
# ==============================================================================

# 手臂主要关节（肩关节俯仰/侧滚/航向、肘关节及手腕侧滚）
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

# 髋俯仰、髋航向与腰航向关节 (在 23-DOF 构型中，hip pitch 亦配置为 7520_14 电机)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

# 髋侧滚与膝关节（高负载腿部关节）
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

# 脚踝并联四连杆机构（双 5020 电机驱动，标称 1:1 传动比近似，等效参数加倍）
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

# 标准直立站立关键帧 (高度 z=0.8 m)
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

# 屈膝微蹲姿态关键帧 (高度 z=0.78 m)
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
# 23-DOF 铰接系统与机器人完整实体配置 (Articulation & Entity Configuration)
# ==============================================================================

G1_23DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_23dof_robot_cfg() -> EntityCfg:
  """获取全新的 Unitree G1 23-DOF 机器人实体配置对象 (Get a fresh G1_23DOF robot config)。

  默认初始状态为直立站姿 `HOME_KEYFRAME`，启用 `FULL_COLLISION` 全碰撞检测。
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_23DOF_ARTICULATION,
  )


# ==============================================================================
# 强化学习动作缩放字典 (RL Action Scale Dictionary)
# ==============================================================================
G1_23DOF_ACTION_SCALE: dict[str, float] = {}
for a in G1_23DOF_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    G1_23DOF_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_23dof_robot_cfg())

  viewer.launch(robot.spec.compile())
