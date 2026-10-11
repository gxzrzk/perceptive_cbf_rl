"""Unitree G1 人形机器人可驱动头部俯仰云台相机配置（Camera Gimbal Configuration）。

设计背景与硬件架构：
在标准 Unitree G1 模型中，头部深度相机（`head_camera_single`）通常以固定朝向（抬高 +20 度）刚性安装在躯干
（`torso_link`）上。当机器人执行深蹲、侧扑、跳跃或在小球从极高/极低角度飞来时，固定的视场角（FOV）极易丢失
对飞球的持续视觉跟踪。

本模块通过 MuJoCo 的 `MjSpec` 运行时脚本接口，以编程式（Programmatic Injection）动态为 G1 插入一个
1 自由度俯仰旋转云台（1-DoF Pitch Gimbal）：
1. 连杆重父化（Re-parenting）：
   在 `torso_link` 下新增子刚体 `camera_pitch_link`，安装点位于原相机安装坐标 `(0.08, 0.0, 0.45)`；
2. 自由度与关节限位：
   在躯干与云台之间添加绕 Y 轴旋转的铰链关节 `camera_pitch_joint`（控制相机上下俯仰），
   中立零位为正前方水平视线（0 度），活动行程范围为 `[-30度, +30度]`；
3. 执行器配置（Actuator）：
   复用 Unitree G1 最小型号的电机（4010 规格电机），采用内置 PD 位置控制器（BuiltinPositionActuatorCfg）；
   其刚度、阻尼与力矩限制严格对齐 G1 硬件常量；
4. 先知视线瞄准机制（Oracle-Aimed Tracking）：
   在训练与仿真中，云台由先知视线自动驱动俯仰角对准小球，从而模拟出视场始终紧盯移动目标的动态视觉感知机制；
5. 零污染与精确复现（Exact Repro）：
   该机制仅在环境变量 `CAMERA_GIMBAL=1` 显式开启时通过 `spec_fn` 包装器生效，不开启时模型完全保持原始无修改状态。
"""

from __future__ import annotations

import dataclasses
import math

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from src.assets.robots.unitree_g1 import g1_constants as gc

# 云台刚体与铰链关节名称定义
GIMBAL_BODY = "camera_pitch_link"
GIMBAL_JOINT = "camera_pitch_joint"
# 俯仰角度限位：+/- 30 度（弧度制约 0.5236 rad）
_PITCH_LIMIT = math.radians(30.0)
# 相机在躯干 torso_link 上的基准安装相对三维坐标 [x, y, z]
_MOUNT_POS = (0.08, 0.0, 0.45)


def _edit_spec(spec: mujoco.MjSpec) -> None:
    """在 MuJoCo MjSpec 规范中插入云台刚体、关节并将头部相机重挂载至云台。

    Args:
        spec: MuJoCo 模型的 MjSpec 规格定义对象。
    """
    torso = spec.body("torso_link")
    # 在躯干下新增云台刚体
    gimbal = torso.add_body(name=GIMBAL_BODY, pos=list(_MOUNT_POS))
    # 为云台添加微小的轻质球体几何体（质量仅 50g），赋予其良态的惯性张量，同时其动力学对机器人整体运动影响可忽略不计
    # group=3, contype=0, conaffinity=0 表示无碰撞检测且在默认渲染视角中隐藏
    gimbal.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.02, 0.0, 0.0],
        mass=0.05,
        group=3,
        contype=0,
        conaffinity=0,
    )
    # 添加绕 Y 轴俯仰旋转的铰链关节
    gimbal.add_joint(
        name=GIMBAL_JOINT,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        range=[-_PITCH_LIMIT, _PITCH_LIMIT],
    )
    # 移除躯干上原有的固定相机，并重新挂载到云台坐标系的原点 [0, 0, 0]
    # 在 MuJoCo 中相机光轴默认指向 -Z 方向，通过 xyaxes 设置：
    # right=(0, -1, 0), up=(0, 0, 1) -> 使得关节角度为 0 时相机的光轴恰好沿 +X 水平向前
    old = spec.camera("head_camera_single")
    fovy = float(old.fovy)
    spec.delete(old)  # 通过 spec.delete 删除原有相机实体
    gimbal.add_camera(
        name="head_camera_single",
        pos=[0, 0, 0],
        fovy=fovy,
        xyaxes=[0, -1, 0, 0, 0, 1],
    )


def _camera_pitch_actuator() -> BuiltinPositionActuatorCfg:
    """构建用于驱动 ``camera_pitch_joint`` 的位置控制器配置。

    采用 Unitree G1 最小规格的 4010 电机参数驱动云台：
    - 刚度（Stiffness）: 对应 gc.STIFFNESS_4010
    - 阻尼（Damping）: 对应 gc.DAMPING_4010
    - 峰值力矩限位: gc.ACTUATOR_4010.effort_limit (5 Nm)
    - 转子反射惯量（Armature）: gc.ACTUATOR_4010.reflected_inertia
    """
    return BuiltinPositionActuatorCfg(
        target_names_expr=(GIMBAL_JOINT,),
        stiffness=gc.STIFFNESS_4010,
        damping=gc.DAMPING_4010,
        effort_limit=gc.ACTUATOR_4010.effort_limit,
        armature=gc.ACTUATOR_4010.reflected_inertia,
    )


def add_camera_gimbal(robot_cfg) -> None:
    """为机器人配置原地附加俯仰云台：包装 ``spec_fn`` 并追加执行器。

    注意：本函数直接原地修改传入的 `robot_cfg`。为了防止多环境构建时污染全局共享的
    `G1_ARTICULATION` 模块变量，内部使用 dataclasses.replace 对关节配置执行了安全副本替换。

    Args:
        robot_cfg: G1 机器人的实体配置对象。
    """
    orig_spec_fn = robot_cfg.spec_fn

    def spec_fn():
        spec = orig_spec_fn()
        _edit_spec(spec)
        return spec

    robot_cfg.spec_fn = spec_fn

    # 追加云台执行器配置
    art = robot_cfg.articulation
    new_actuators = tuple(art.actuators) + (_camera_pitch_actuator(),)
    robot_cfg.articulation = dataclasses.replace(art, actuators=new_actuators)
