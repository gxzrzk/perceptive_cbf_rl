from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco

from mjlab.actuator import Actuator, ActuatorCmd, BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.spec import create_position_actuator

if TYPE_CHECKING:
    from mjlab.entity import Entity


class UnitreeActuator(Actuator):
    """Unitree actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque Limit, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    - Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    - Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    - X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    - X2: No-Load Speed Test

    - Fs: Static friction coefficient
    - Fd: Dynamic friction coefficient
    - Va: Velocity at which the friction is fully activated
    """

    cfg: UnitreeActuatorCfg

    _joint_vel: torch.Tensor
    _effort_y1: torch.Tensor
    _effort_y2: torch.Tensor
    _velocity_x1: torch.Tensor
    _velocity_x2: torch.Tensor
    _friction_static: torch.Tensor
    _friction_dynamic: torch.Tensor
    _activation_vel: torch.Tensor

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        # Add MuJoCo builtin <position> actuators, one per target.
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
        super().initialize(mj_model, model, data, device)

        num_envs = data.nworld
        num_joints = len(self.target_names)
        shape = (num_envs, num_joints)

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
        # Save current joint velocity for torque-speed clipping.
        self._joint_vel[:] = cmd.vel

        # Compute desired effort with PD + feedforward, then apply custom limits.
        effort = self.cfg.stiffness * (cmd.position_target - cmd.pos)
        effort += self.cfg.damping * (cmd.velocity_target - cmd.vel)
        effort += cmd.effort_target
        effort = self._clip_effort(effort)

        # Apply friction model on output torque.
        effort -= (
            self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
            + self._friction_dynamic * cmd.vel
        )

        # BuiltinPositionActuator expects a position control signal.
        # These config values are scalars. Tensor arithmetic already casts scalar
        # operands to the working dtype, without allocating/copying CUDA tensors
        # for every actuator group at every physics substep.
        kp = max(self.cfg.stiffness, 1e-6)
        return cmd.pos + (effort + self.cfg.damping * cmd.vel) / kp

    def get_command(self, data) -> ActuatorCmd:
        if self.transmission_type != TransmissionType.JOINT:
            return super().get_command(data)
        # EntityData.joint_pos/vel first gather every robot joint. Cache this
        # group's global addresses on the working device and gather only its
        # joints directly from the live sim buffers.
        if getattr(self, "_command_indexing", None) is not data.indexing:
            self._command_q_adr = torch.as_tensor(
                data.indexing.joint_q_adr[self._target_ids_list],
                dtype=torch.long, device=data.device,
            )
            self._command_v_adr = torch.as_tensor(
                data.indexing.joint_v_adr[self._target_ids_list],
                dtype=torch.long, device=data.device,
            )
            self._command_indexing = data.indexing
        return ActuatorCmd(
            position_target=data.joint_pos_target[:, self.target_ids],
            velocity_target=data.joint_vel_target[:, self.target_ids],
            effort_target=data.joint_effort_target[:, self.target_ids],
            pos=data.data.qpos[:, self._command_q_adr],
            vel=data.data.qvel[:, self._command_v_adr],
        )

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)

        if self.cfg.effort_limit is not None:
            max_effort = max_effort.clamp(max=self.cfg.effort_limit)

        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        denom = torch.clamp(self._velocity_x2 - self._velocity_x1, min=1e-6)
        k = -max_effort / denom
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@dataclass(kw_only=True)
class UnitreeActuatorCfg(BuiltinPositionActuatorCfg):
    """
    Configuration for Unitree actuators.
    """

    X1: float = 1e9
    """Maximum Speed at Full Torque(T-N Curve Knee Point) Unit: rad/s"""

    X2: float = 1e9
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = 0.0
    """Peak Torque Test(Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float | None = None
    """Peak Torque Test(Torque and Speed in the Opposite Direction) Unit: N*m"""

    Fs: float = 0.0
    """ Static friction coefficient """

    Fd: float = 0.0
    """ Dynamic friction coefficient """

    Va: float = 0.01
    """ Velocity at which the friction is fully activated """

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> UnitreeActuator:
        return UnitreeActuator(self, entity, target_ids, target_names)


@dataclass(kw_only=True)
class UnitreeActuatorCfg_M107_15(UnitreeActuatorCfg):
    X1: float = 14.0
    X2: float = 25.6
    Y1: float = 150.0
    Y2: float = 182.8

    armature: float = 0.063259741


@dataclass(kw_only=True)
class UnitreeActuatorCfg_M107_24(UnitreeActuatorCfg):
    X1: float = 8.8
    X2: float = 16.0
    Y1: float = 240.0
    Y2: float = 292.5

    armature: float = 0.160478022


@dataclass(kw_only=True)
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    X1: float = 13.5
    X2: float = 30.0
    Y1: float = 20.2
    Y2: float = 23.4


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N7520_14p3(UnitreeActuatorCfg):
    # Decimal point cannot be used as variable name, use `p` instead
    X1: float = 22.63
    X2: float = 35.52
    Y1: float = 71.0
    Y2: float = 83.3

    Fs: float = 1.6
    Fd: float = 0.16

    """
    | rotor  | 0.489e-4 kg·m²
    | gear_1 | 0.098e-4 kg·m² | ratio | 4.5
    | gear_2 | 0.533e-4 kg·m² | ratio | 48/22+1
    """
    armature: float = 0.01017752


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N7520_22p5(UnitreeActuatorCfg):
    # Decimal point cannot be used as variable name, use `p` instead
    X1: float = 14.5
    X2: float = 22.7
    Y1: float = 111.0
    Y2: float = 131.0

    Fs: float = 2.4
    Fd: float = 0.24

    """
    | rotor  | 0.489e-4 kg·m²
    | gear_1 | 0.109e-4 kg·m² | ratio | 4.5
    | gear_2 | 0.738e-4 kg·m² | ratio | 5.0
    """
    armature: float = 0.025101925


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N5010_16(UnitreeActuatorCfg):
    X1: float = 27.0
    X2: float = 41.5
    Y1: float = 9.5
    Y2: float = 17.0

    """
    | rotor  | 0.084e-4 kg·m²
    | gear_1 | 0.015e-4 kg·m² | ratio | 4
    | gear_2 | 0.068e-4 kg·m² | ratio | 4
    """
    armature: float = 0.0021812


@dataclass(kw_only=True)
class UnitreeActuatorCfg_N5020_16(UnitreeActuatorCfg):
    X1: float = 30.86
    X2: float = 40.13
    Y1: float = 24.8
    Y2: float = 31.9

    Fs: float = 0.6
    Fd: float = 0.06

    """
    | rotor  | 0.139e-4 kg·m²
    | gear_1 | 0.017e-4 kg·m² | ratio | 46/18+1
    | gear_2 | 0.169e-4 kg·m² | ratio | 56/16+1
    """
    armature: float = 0.003609725


@dataclass(kw_only=True)
class UnitreeActuatorCfg_W4010_25(UnitreeActuatorCfg):
    X1: float = 15.3
    X2: float = 24.76
    Y1: float = 4.8
    Y2: float = 8.6

    Fs: float = 0.6
    Fd: float = 0.06

    """
    | rotor  | 0.068e-4 kg·m²
    | gear_1 |                | ratio | 5
    | gear_2 |                | ratio | 5
    """
    armature: float = 0.00425
