"""Unitree G1 全自由度 (29 DOF) 仿人机器人常量与配置模块 (Unitree G1 Humanoid Constants and Configuration Module)。

本模块定义了 Unitree G1 仿人机器人在 MuJoCo 仿真环境中的全套核心物理配置与控制参数：
1. **MJCF 模型规范 (XML Specification)**:
   加载包含 29 个可驱动自由度的高精度 G1 仿人机器人模型结构，自动解析几何网格与运动学拓扑。
2. **两级行星减速器与电枢惯量折算 (Reflected Armature Inertia)**:
   基于各型号电机（5020, 7520, 4010, 5010）的转子转动惯量及两级行星齿轮传动比，精确折算电机端到关节输出端的等效电枢惯量。
3. **二阶临界/过阻尼 PD 增益推导 (Critically-damped/Overdamped PD Tuning)**:
   将关节视为二阶质量-弹簧-阻尼系统，以统一的固有频率 $\\omega_n$ (10 Hz) 和过阻尼比 $\\zeta=2.0$ 自适应计算各关节的
   刚度系数 (Stiffness / Kp) 与阻尼系数 (Damping / Kd)，避免仿真中出现数值失稳与高频抖振。
4. **并联四连杆机构处理 (Waist & Ankle 4-Bar Parallel Linkages)**:
   腰部俯仰/侧滚以及踝部俯仰/侧滚各由 2 台 5020 电机通过差动/四连杆并联机构驱动；在公称位姿下按 1:1 传动比近似，
   将其等效电枢惯量、刚度、阻尼与力矩上限加倍。
5. **宇树非线性执行器模型绑定 (Non-linear Unitree Actuators)**:
   将各关节组与对应的 `UnitreeActuatorCfg_...` 绑定，应用 T-N 扭矩-转速限制及非线性摩擦力模型。
6. **碰撞过滤与接触维度 (Collisions & Contact Dimensions)**:
   配置自碰撞与环境接触规则；足底 7 点接触网格赋予 3 维接触条件 (condim=3) 与高接触优先级，连杆自碰撞赋予 condim=1。
7. **动作缩放映射 (Action Scaling)**:
   基于各关节的力矩上限与刚度比例关系（$\\Delta q = 0.25 \\cdot \\tau_{max} / K_p$）推导策略动作缩放因子。
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
# 备用导入：内置位置执行器配置（当前采用基于实机特性建模的宇树专用非线性执行器 UnitreeActuator）：
# from mjlab.actuator import BuiltinPositionActuatorCfg  # 内置位置执行器类定义
from src.assets.robots.unitree_g1.unitree_actuators import *
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

  在 mjlab 1.4.0+ 中，网格模型通过 XML 中的 `meshdir` 在编译阶段直接从磁盘解析，
  因此直接通过 MjSpec.from_file 编译加载。
  """
  return mujoco.MjSpec.from_file(str(G1_XML))


# ==============================================================================
# 执行器电枢转动惯量与电机规格 (Actuator Motor Specs & Armature Inertias)
# ==============================================================================

# 宇树官方电机硬件参数 (Motor specs from Unitree):
# 5020 电机模块 (手臂关节及腰/踝驱动单元):
# rotor 转子、gear_1 一级行星架、gear_2 二级太阳/行星齿轮惯量 (kg·m²)
ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
# 5020 两级齿轮传动比
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
# 折算到关节输出轴的等效转动惯量 (Reflected Armature Inertia)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

# 7520-14 电机模块 (减速比 14.3，用于髋偏航及腰偏航):
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

# 7520-22 电机模块 (减速比 22.5，高力矩，用于髋俯仰、髋侧滚及膝关节):
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

# 4010 电机模块:
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

# 5010-16 电机模块 (用于腕部俯仰与偏航关节):
ARMATURE_5010_16 = 0.0021812

# 电机参数容器对象 (包含折算惯量、角速度上限 rad/s、输出力矩上限 N·m)
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
# PD 控制增益推导 (二阶振荡系统参数: 刚度 Kp 与 阻尼 Kd)
# ==============================================================================
# 目标固有频率 omega_n: 10 Hz -> 2 * pi * 10 rad/s
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 设定二阶振荡系统的目标固有振动频率为 10Hz (omega_n = 2 * pi * f)
# 阻尼比 zeta: 设为 2.0 (过阻尼状态 overdamped)，确保受扰动后平稳单调回归，抑制双足奔跑/跳跃时的冲击振荡
DAMPING_RATIO = 2.0

# 刚度系数推导公式: K_p = I * omega_n^2
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
STIFFNESS_5010_16 = ARMATURE_5010_16 * NATURAL_FREQ**2

# 阻尼系数推导公式: K_d = 2 * zeta * I * omega_n
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ
DAMPING_5010_16 = 2.0 * DAMPING_RATIO * ARMATURE_5010_16 * NATURAL_FREQ

# ==============================================================================
# 执行器分组与配置实例化 (Actuator Group Configurations)
# ==============================================================================

# 手臂及手腕翻滚关节（5020 电机）
G1_ACTUATOR_5020 = UnitreeActuatorCfg_N5020_16(
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

# 髋航向角与腰航向角（7520-14 电机，中等减速比）
G1_ACTUATOR_7520_14 = UnitreeActuatorCfg_N7520_14p3(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)

# 腿部高负荷承重关节：髋俯仰、髋侧倾、膝关节（7520-22 电机，大减速比与高力矩）
G1_ACTUATOR_7520_22 = UnitreeActuatorCfg_N7520_22p5(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)

# 手腕末端俯仰与偏航关节（5010-16 电机）
G1_ACTUATOR_5010_16 = UnitreeActuatorCfg_N5010_16(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_5010_16,
  damping=DAMPING_5010_16,
  effort_limit=ACTUATOR_5010_16.effort_limit,
  armature=ACTUATOR_5010_16.reflected_inertia,
)

# 腰部俯仰/侧滚与脚踝俯仰/侧滚为双 5020 电机驱动的并联四连杆机构 (4-bar linkages)。
# 由于并联连杆机构的有效等效电枢惯量与当前关节位形相关，在连杆精确解析未知的工程前提下，
# 假设公称工况传动比为 1:1，因此标称等效电枢惯量、刚度、阻尼与力矩上限均近似为 2 台电机直接叠加 (2倍)。
G1_ACTUATOR_WAIST = UnitreeActuatorCfg_N5020_16(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = UnitreeActuatorCfg_N5020_16(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

# ==============================================================================
# 机器人关键帧姿态 (Keyframe Configurations)
# ==============================================================================

# 标准直立站姿关键帧 (高度 z=0.8 m)
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

# 屈膝微蹲姿态关键帧 (高度 z=0.78 m，重心更低更平稳，作为训练的默认初始位姿)
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

# 全碰撞模式（启用包括自碰撞在内的所有碰撞）
# 脚部碰撞体 (left/right_foot1-7_collision) 设定接触约束维度 condim=3 (支持三维摩擦锥切向阻力)，
# 接触优先级为 1，摩擦系数为 0.6；连杆自碰撞接触维度设定为 condim=1 (仅无摩擦法向接触力)。
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# 全环境碰撞模式（关闭自碰撞）：contype=0, conaffinity=1
FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# 仅足底碰撞模式：仅保留双脚与地面的接触检测，彻底屏蔽躯干与手臂碰撞以最大化仿真吞吐率
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

# G1 铰接系统综合定义，包含各执行器组及软限位缩放因子 (0.9，防止硬撞击关节限位边界)
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
  """获取全新的 Unitree G1 机器人实体配置对象 (Get a fresh G1 robot configuration instance)。

  每次调用均返回独立的 `EntityCfg` 实例，以杜绝多任务或多环境共享同一对象带来的潜在变量修改副作用。
  默认初始姿态为屈膝微蹲姿态 `KNEES_BENT_KEYFRAME`，启用包含自碰撞的 `FULL_COLLISION`。
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_ARTICULATION,
  )


# ==============================================================================
# 强化学习动作缩放因子字典 (RL Action Scale Dictionary)
# ==============================================================================
# 将网络策略输出的无量纲动作 a in [-1, 1] 映射为关节目标角度增量 Delta_q。
# 映射公式基于静力平衡：Delta_q = 0.25 * (effort_limit / stiffness)
# 即满额策略动作 (+1 或 -1) 产生的瞬时位置偏移对应电机最大连续力矩下弹性变形量的 25%，
# 确保在探索过程中动作具有合理的物理量纲与探索步长。
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
  # 直接运行脚本可在交互式 MuJoCo 可视化器中预览加载后的机器人模型与初始位姿
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
