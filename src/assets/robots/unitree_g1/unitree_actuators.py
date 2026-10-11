"""宇树 Unitree 电驱执行器非线性动力学模型 (Unitree Actuator Non-linear Dynamics Model)。

本模块实现了宇树 (Unitree) 机器人电驱关节的高保真物理仿真模型，主要包含以下核心物理特性：
1. **T-N (扭矩-转速) 动力学包络截断**:
   模拟电机在不同转速下的输出力矩能力限制（包含低速恒扭矩区、高速弱磁/反电动势受限的恒功率线性衰减区，
   以及转速达到空载转速时的力矩归零点）。
2. **正反向旋转峰值转矩非对称性 (Four-Quadrant Operation)**:
   电机做正功（转矩与转速同向，电动状态）和电机做负功（转矩与转速反向，制动/反拖状态）具有不同的峰值扭矩包络 (Y1 vs Y2)。
3. **基于双曲正切的光滑静/动摩擦力模型 (Smooth Static & Dynamic Friction Model)**:
   采用 Stribeck 式的光滑连续近似，结合库仑静摩擦力 $F_s \cdot \tanh(\dot{q} / V_a)$ 与粘滞摩擦力 $F_d \cdot \dot{q}$，
   在有效避免零速数值颤振 (chattering) 的前提下逼真模拟减速箱阻尼与摩擦损失。
4. **MuJoCo 原生位置伺服映射**:
   通过逆向推导，将经过 T-N 限制与摩擦力扣除后的目标有效力矩，解析转换为 MuJoCo 内置 `<position>` 执行器
   所需的目标驱动控制信号，确保物理求解器在具备稳定数值隐式积分的同时实现非线性力矩特性。
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco

from mjlab.actuator import Actuator, ActuatorCmd, BuiltinPositionActuatorCfg
from mjlab.utils.spec import create_position_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


class UnitreeActuator(Actuator):
    """宇树执行器物理实现类，具备扭矩-转速 (T-N) 曲线截断与摩擦力补偿。

    T-N (Torque-Speed) 特性曲线定义如下：

            Torque Limit, N·m (输出扭矩极限)
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s (关节角速度)
                               X1   X2

    关键特征参数说明：
    - Y1: 同向峰值扭矩 (Peak Torque: 扭矩与转速同向，电机处于电动做功状态，第 1 / 3 象限)
    - Y2: 反向峰值扭矩 (Peak Torque: 扭矩与转速反向，电机处于发电机/制动反拖状态，第 2 / 4 象限)
    - X1: 满扭矩转速拐点 (Knee Point / Base Speed: 恒扭矩区的最大转速，单位 rad/s)
    - X2: 空载极限转速 (No-Load Speed: 弱磁/反电动势达到极限，输出扭矩降为 0 时的转速，单位 rad/s)

    摩擦力模型参数说明：
    - Fs: 静摩擦力矩系数 (Static friction coefficient, N·m)
    - Fd: 动/粘滞摩擦系数 (Dynamic / viscous friction coefficient, N·m·s/rad)
    - Va: 摩擦力平滑完全激活特征转速 (Velocity threshold for full friction activation, rad/s)
    """

    cfg: UnitreeActuatorCfg

    # 并行环境下的 PyTorch 张量缓存 (shape: [num_envs, num_joints])
    _joint_vel: torch.Tensor          # 当前各关节角速度
    _effort_y1: torch.Tensor          # 同向峰值扭矩张量
    _effort_y2: torch.Tensor          # 反向峰值扭矩张量
    _velocity_x1: torch.Tensor        # 满扭矩转速拐点张量
    _velocity_x2: torch.Tensor        # 空载极限转速张量
    _friction_static: torch.Tensor    # 静摩擦系数张量
    _friction_dynamic: torch.Tensor   # 动摩擦系数张量
    _activation_vel: torch.Tensor     # 摩擦力平滑过渡特征转速张量

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        """在 MuJoCo MjSpec 模型规范中为每个目标关节添加原生 `<position>` 位置执行器。

        参数:
            spec: MuJoCo 模型规范对象 (MjSpec)。
            target_names: 该执行器组所绑定的目标关节名称列表。
        """
        # 为每个受控关节在 MuJoCo 规范中创建内置 position 类型的执行器
        for target_name in target_names:
            actuator = create_position_actuator(
                spec,
                target_name,
                stiffness=self.cfg.stiffness,
                damping=self.cfg.damping,
                effort_limit=self.cfg.effort_limit,
                armature=self.cfg.armature,
                frictionloss=self.cfg.frictionloss,
                transmission_type=self.cfg.transmission_type,
            )
            self._mjs_actuators.append(actuator)

    def initialize(self, mj_model, model, data, device: str) -> None:
        """初始化执行器内部状态张量，并在指定计算设备 (CPU/CUDA) 上预分配缓冲区。

        参数:
            mj_model: MuJoCo C 结构体模型对象。
            model: 仿真封装模型。
            data: 仿真运行时数据对象 (包含并行环境世界数量 nworld)。
            device: 运算设备字符串 (如 "cuda:0" 或 "cpu")。
        """
        super().initialize(mj_model, model, data, device)

        num_envs = data.nworld
        num_joints = len(self.target_names)
        shape = (num_envs, num_joints)

        # 预分配张量以避免在 step 循环中频繁分配显存/内存
        self._joint_vel = torch.zeros(shape, dtype=torch.float, device=device)
        self._effort_y1 = torch.full(shape, self.cfg.Y1, dtype=torch.float, device=device)
        self._effort_y2 = torch.full(
            shape,
            self.cfg.Y1 if self.cfg.Y2 is None else self.cfg.Y2,
            dtype=torch.float,
            device=device,
        )
        self._velocity_x1 = torch.full(shape, self.cfg.X1, dtype=torch.float, device=device)
        self._velocity_x2 = torch.full(shape, self.cfg.X2, dtype=torch.float, device=device)
        self._friction_static = torch.full(shape, self.cfg.Fs, dtype=torch.float, device=device)
        self._friction_dynamic = torch.full(shape, self.cfg.Fd, dtype=torch.float, device=device)
        self._activation_vel = torch.full(shape, self.cfg.Va, dtype=torch.float, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """执行器前向动力学计算步：计算实际力矩，并逆解算为 MuJoCo 位置伺服信号。

        计算流程：
        1. 记录当前关节角速度用于后续 T-N 限制判断；
        2. 基于 PD 控制律 + 前馈力矩计算期望输出力矩：
           $$\\tau = K_p (q_{target} - q) + K_d (\\dot{q}_{target} - \\dot{q}) + \\tau_{ff}$$
        3. 调用 `_clip_effort` 执行 T-N 曲线截断与全局力矩限制；
        4. 施加非线性摩擦力模型，扣除阻力矩：
           $$\\tau_{net} = \\tau_{clipped} - [F_s \\cdot \\tanh(\\dot{q} / V_a) + F_d \\cdot \\dot{q}]$$
        5. 逆解算 MuJoCo 原生 BuiltinPositionActuator 的目标位置输入信号：
           MuJoCo 原生位置伺服产生的力矩为 $\\tau_{sim} = K_p (u - q) - K_d \\dot{q}$，
           令 $\\tau_{sim} = \\tau_{net}$，解得：
           $$u = q + \\frac{\\tau_{net} + K_d \\dot{q}}{K_p}$$

        参数:
            cmd: 包含目标位置、目标速度、目标前馈力矩以及当前实际位置和速度的执行器控制指令。

        返回:
            torch.Tensor: 写入 MuJoCo 执行器的目标控制输入张量 (u)。
        """
        # 保存当前关节速度供力矩-转速曲线截断使用
        self._joint_vel[:] = cmd.vel

        # 计算 PD 期望输出力矩 + 前馈扭矩
        effort = self.cfg.stiffness * (cmd.position_target - cmd.pos)
        effort += self.cfg.damping * (cmd.velocity_target - cmd.vel)
        effort += cmd.effort_target

        # 经过 T-N 曲线和硬件力矩上限进行非线性截断
        effort = self._clip_effort(effort)

        # 施加摩擦力模型扣除阻力项：平滑库仑静摩擦 + 线性粘滞摩擦
        effort -= (
            self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
            + self._friction_dynamic * cmd.vel
        )

        # MuJoCo BuiltinPositionActuator 期望接收位置控制信号 u：
        # 根据 tau = kp * (u - pos) - kd * vel，反解出 u = pos + (tau + kd * vel) / kp
        kp = torch.as_tensor(self.cfg.stiffness, dtype=cmd.pos.dtype, device=cmd.pos.device)
        kd = torch.as_tensor(self.cfg.damping, dtype=cmd.pos.dtype, device=cmd.pos.device)
        kp = torch.clamp(kp, min=1e-6)  # 防止除以零
        return cmd.pos + (effort + kd * cmd.vel) / kp

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """根据关节转速与旋转方向，利用 T-N 曲线截断期望输出扭矩。

        参数:
            effort: 期望驱动力矩张量。

        返回:
            torch.Tensor: 截断后的有效力矩张量。
        """
        # 判断力矩方向与当前关节旋转速度是否相同（同向做正功，反向做负功/制动）
        same_direction = (self._joint_vel * effort) > 0
        # 同向使用 Y1 峰值力矩限制，反向使用 Y2 制动力矩限制
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)

        # 若配置了显式全局力矩上限 effort_limit，取二者的较小值
        if self.cfg.effort_limit is not None:
            limit = torch.as_tensor(
                self.cfg.effort_limit,
                dtype=max_effort.dtype,
                device=max_effort.device,
            )
            max_effort = torch.minimum(max_effort, limit)

        # 检查关节转速是否在恒转矩区 (转速绝对值 < X1)
        # 若在恒转矩区则保持 max_effort；若超出 X1，则调用 _compute_effort_limit 进行线性衰减
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        # 将输出力矩截断在 [-max_effort, max_effort] 区间内
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
        """计算当转速超过拐点 X1 时，输出力矩沿 T-N 曲线斜率下降后的最大允许值。

        公式：
        $$\\text{limit} = -\\frac{\\text{max\\_effort}}{X_2 - X_1} (|\\dot{q}| - X_1) + \\text{max\\_effort}$$
        并在达到空载转速 X2 时下限截断至 0.0。
        """
        denom = torch.clamp(self._velocity_x2 - self._velocity_x1, min=1e-6)
        k = -max_effort / denom
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@dataclass(kw_only=True)
class UnitreeActuatorCfg(BuiltinPositionActuatorCfg):
    """宇树执行器基础配置类 (Configuration dataclass for Unitree actuators)。

    继承自 `BuiltinPositionActuatorCfg`，扩展了 T-N 曲线与摩擦力学模型参数。
    """

    X1: float = 1e9
    """满扭矩最大角速度（T-N 曲线拐点转速 / Knee Point），单位: rad/s"""

    X2: float = 1e9
    """空载极限角速度（输出力矩衰减至 0 时的转速），单位: rad/s"""

    Y1: float = 0.0
    """同向峰值扭矩（力矩与角速度方向相同，做正功），单位: N·m"""

    Y2: float | None = None
    """反向峰值扭矩（力矩与角速度方向相反，制动反拖），单位: N·m。若为 None 则默认取 Y1"""

    Fs: float = 0.0
    """静摩擦力矩系数 (Static friction coefficient)，单位: N·m"""

    Fd: float = 0.0
    """动/粘滞摩擦系数 (Dynamic friction coefficient)，单位: N·m·s/rad"""

    Va: float = 0.01
    """摩擦力完全激活的平滑过渡特征速度阈值，单位: rad/s"""

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> UnitreeActuator:
        """根据配置构建对应的 `UnitreeActuator` 实例。"""
        return UnitreeActuator(self, entity, target_ids, target_names)


# ==============================================================================
# 宇树特定型号关节电机模块配置 (Unitree Motor Module Configurations)
# ==============================================================================

@dataclass(kw_only=True)
class UnitreeActuatorCfg_M107_15(UnitreeActuatorCfg):
    """宇树 M107-15 电驱模块配置。

    通常用于高负载关节（如髋关节）。
    - 拐点转速 X1: 14.0 rad/s
    - 空载转速 X2: 25.6 rad/s
    - 同向峰值力矩 Y1: 150.0 N·m
    - 反向峰值力矩 Y2: 182.8 N·m
    - 折算到输出端的电枢转动惯量 armature: 0.06326 kg·m²
    """
    X1: float = 14.0
    X2: float = 25.6
    Y1: float = 150.0
    Y2: float = 182.8

    armature: float = 0.063259741


@dataclass(kw_only=True)
class UnitreeActuatorCfg_M107_24(UnitreeActuatorCfg):
    """宇树 M107-24 电驱模块配置（超高减速比/高力矩版本）。

    - 拐点转速 X1: 8.8 rad/s
    - 空载转速 X2: 16.0 rad/s
    - 同向峰值力矩 Y1: 240.0 N·m
    - 反向峰值力矩 Y2: 292.5 N·m
    - 折算到输出端的电枢转动惯量 armature: 0.16048 kg·m²
    """
    X1: float = 8.8
    X2: float = 16.0
    Y1: float = 240.0
    Y2: float = 292.5

    armature: float = 0.160478022


@dataclass(kw_only=True)
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    """宇树 Go2 四足机器人高压版 (Go2-HV) 电机配置。

    - 拐点转速 X1: 13.5 rad/s
    - 空载转速 X2: 30.0 rad/s
    - 同向峰值力矩 Y1: 20.2 N·m
    - 反向峰值力矩 Y2: 23.4 N·m
    """
    X1: float = 13.5
    X2: float = 30.0
    Y1: float = 20.2
    Y2: float = 23.4


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N7520_14p3(UnitreeActuatorCfg):
    """宇树 N7520 电机模块配置，减速比 14.3 (变量名中的 p 代表小数点)。

    用于 G1 髋偏航关节 (hip yaw) 与腰偏航关节 (waist yaw)。
    - 拐点转速 X1: 22.63 rad/s
    - 空载转速 X2: 35.52 rad/s
    - 同向峰值力矩 Y1: 71.0 N·m
    - 反向峰值力矩 Y2: 83.3 N·m
    - 静摩擦 Fs: 1.6 N·m, 动摩擦 Fd: 0.16 N·m·s/rad
    - 减速器结构：两级行星减速器
      | 转子惯量 rotor  | 0.489e-4 kg·m²
      | 一级行星齿轮 gear_1 | 0.098e-4 kg·m² | 一级传动比 ratio | 4.5
      | 二级行星齿轮 gear_2 | 0.533e-4 kg·m² | 二级传动比 ratio | 48/22+1
    - 折算到输出端的等效电枢惯量 armature: 0.01018 kg·m²
    """
    X1: float = 22.63
    X2: float = 35.52
    Y1: float = 71.0
    Y2: float = 83.3

    Fs: float = 1.6
    Fd: float = 0.16

    armature: float = 0.01017752


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N7520_22p5(UnitreeActuatorCfg):
    """宇树 N7520 电机模块配置，减速比 22.5。

    用于 G1 腿部高负载关节：髋俯仰 (hip pitch)、髋侧滚 (hip roll) 以及膝关节 (knee)。
    - 拐点转速 X1: 14.5 rad/s
    - 空载转速 X2: 22.7 rad/s
    - 同向峰值力矩 Y1: 111.0 N·m
    - 反向峰值力矩 Y2: 131.0 N·m
    - 静摩擦 Fs: 2.4 N·m, 动摩擦 Fd: 0.24 N·m·s/rad
    - 减速器结构：两级行星减速器
      | 转子惯量 rotor  | 0.489e-4 kg·m²
      | 一级行星齿轮 gear_1 | 0.109e-4 kg·m² | 一级传动比 ratio | 4.5
      | 二级行星齿轮 gear_2 | 0.738e-4 kg·m² | 二级传动比 ratio | 5.0
    - 折算到输出端的等效电枢惯量 armature: 0.02510 kg·m²
    """
    X1: float = 14.5
    X2: float = 22.7
    Y1: float = 111.0
    Y2: float = 131.0

    Fs: float = 2.4
    Fd: float = 0.24

    armature: float = 0.025101925


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N5010_16(UnitreeActuatorCfg):
    """宇树 N5010 电机模块配置，减速比 16。

    用于 G1 手腕俯仰与偏航等灵巧末端关节。
    - 拐点转速 X1: 27.0 rad/s
    - 空载转速 X2: 41.5 rad/s
    - 同向峰值力矩 Y1: 9.5 N·m
    - 反向峰值力矩 Y2: 17.0 N·m
    - 减速器结构：
      | 转子惯量 rotor  | 0.084e-4 kg·m²
      | 一级行星齿轮 gear_1 | 0.015e-4 kg·m² | 一级传动比 ratio | 4
      | 二级行星齿轮 gear_2 | 0.068e-4 kg·m² | 二级传动比 ratio | 4
    - 折算电枢惯量 armature: 0.0021812 kg·m²
    """
    X1: float = 27.0
    X2: float = 41.5
    Y1: float = 9.5
    Y2: float = 17.0

    armature: float = 0.0021812


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N5020_16(UnitreeActuatorCfg):
    """宇树 N5020 电机模块配置，减速比 16。

    用于 G1 手臂主要关节（肩、肘、手腕侧滚）及腰部/踝部四连杆并联机构驱动单元。
    - 拐点转速 X1: 30.86 rad/s
    - 空载转速 X2: 40.13 rad/s
    - 同向峰值力矩 Y1: 24.8 N·m
    - 反向峰值力矩 Y2: 31.9 N·m
    - 静摩擦 Fs: 0.6 N·m, 动摩擦 Fd: 0.06 N·m·s/rad
    - 减速器结构：两级行星减速器
      | 转子惯量 rotor  | 0.139e-4 kg·m²
      | 一级行星齿轮 gear_1 | 0.017e-4 kg·m² | 一级传动比 ratio | 46/18+1
      | 二级行星齿轮 gear_2 | 0.169e-4 kg·m² | 二级传动比 ratio | 56/16+1
    - 折算电枢惯量 armature: 0.0036097 kg·m²
    """
    X1: float = 30.86
    X2: float = 40.13
    Y1: float = 24.8
    Y2: float = 31.9

    Fs: float = 0.6
    Fd: float = 0.06

    armature: float = 0.003609725


@dataclass(kw_only=True)
class UnitreeActuatorCfg_W4010_25(UnitreeActuatorCfg):
    """宇树 W4010 腕部电机模块配置，减速比 25。

    - 拐点转速 X1: 15.3 rad/s
    - 空载转速 X2: 24.76 rad/s
    - 同向峰值力矩 Y1: 4.8 N·m
    - 反向峰值力矩 Y2: 8.6 N·m
    - 静摩擦 Fs: 0.6 N·m, 动摩擦 Fd: 0.06 N·m·s/rad
    - 减速器结构：
      | 转子惯量 rotor  | 0.068e-4 kg·m²
      | 一级传动比 ratio | 5
      | 二级传动比 ratio | 5
    - 折算电枢惯量 armature: 0.00425 kg·m²
    """
    X1: float = 15.3
    X2: float = 24.76
    Y1: float = 4.8
    Y2: float = 8.6

    Fs: float = 0.6
    Fd: float = 0.06

    armature: float = 0.00425
