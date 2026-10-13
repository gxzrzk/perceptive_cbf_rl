"""Torque-speed/friction control remains equivalent without CUDA scalar copies."""
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.actuator import Actuator, ActuatorCmd
from mjlab.actuator.actuator import TransmissionType
from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuator


@pytest.mark.parametrize('stiffness', [0., 32.4])
@pytest.mark.parametrize('effort_limit', [None, 10.2])
@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_control_matches_torque_speed_and_friction_reference(stiffness, effort_limit, device):
    torch.manual_seed(17)
    shape = (32, 5)
    actuator = UnitreeActuator.__new__(UnitreeActuator)
    actuator.cfg = NS(stiffness=stiffness, damping=2.6, effort_limit=effort_limit)
    actuator._joint_vel = torch.zeros(shape, device=device)
    for name, value in [('effort_y1', 20.2), ('effort_y2', 23.4),
                        ('velocity_x1', 13.5), ('velocity_x2', 30.),
                        ('friction_static', .6), ('friction_dynamic', .06),
                        ('activation_vel', .01)]:
        setattr(actuator, '_' + name, torch.full(shape, value, device=device))
    cmd = ActuatorCmd(position_target=torch.randn(shape, device=device),
                      velocity_target=torch.randn(shape, device=device),
                      effort_target=50. * torch.randn(shape, device=device),
                      pos=torch.randn(shape, device=device),
                      vel=50. * torch.randn(shape, device=device))
    effort = stiffness * (cmd.position_target - cmd.pos)
    effort += 2.6 * (cmd.velocity_target - cmd.vel)
    effort += cmd.effort_target
    limit = torch.where(cmd.vel * effort > 0, 20.2, 23.4)
    if effort_limit is not None:
        limit = torch.minimum(limit, torch.tensor(effort_limit, device=device))
    falling_limit = (limit - limit / (30. - 13.5) * (cmd.vel.abs() - 13.5)).clamp_min(0)
    limit = torch.where(cmd.vel.abs() < 13.5, limit, falling_limit)
    effort = torch.clamp(effort, -limit, limit)
    effort -= .6 * torch.tanh(cmd.vel / .01) + .06 * cmd.vel
    kp = torch.tensor(stiffness, device=device).clamp_min(1e-6)
    expected = cmd.pos + (effort + torch.tensor(2.6, device=device) * cmd.vel) / kp
    torch.testing.assert_close(actuator.compute(cmd), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_group_addresses_preserve_order_and_read_live_simulation_state(device):
    class Data:
        @property
        def joint_pos(self):
            return self.data.qpos[:, self.indexing.joint_q_adr]

        @property
        def joint_vel(self):
            return self.data.qvel[:, self.indexing.joint_v_adr]

    data = Data()
    data.device = device
    data.indexing = NS(joint_q_adr=torch.tensor([7, 9, 8, 11, 10], device=device),
                       joint_v_adr=torch.tensor([6, 10, 8, 7, 9], device=device))
    data.data = NS(qpos=torch.randn(4, 12, device=device),
                   qvel=torch.randn(4, 11, device=device))
    data.joint_pos_target = torch.randn(4, 5, device=device)
    data.joint_vel_target = torch.randn(4, 5, device=device)
    data.joint_effort_target = torch.randn(4, 5, device=device)
    actuator = UnitreeActuator(NS(transmission_type=TransmissionType.JOINT), None,
                               [4, 1, 3], ['a', 'b', 'c'])
    actuator._target_ids = torch.tensor([4, 1, 3], device=device)
    for _ in range(2):
        actual, expected = actuator.get_command(data), Actuator.get_command(actuator, data)
        for field in ('position_target', 'velocity_target', 'effort_target', 'pos', 'vel'):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
        data.data.qpos.add_(1.)
        data.data.qvel.sub_(2.)
    # Reinitialization can replace entity indexing; do not retain old addresses.
    data.indexing = NS(joint_q_adr=torch.tensor([11, 10, 9, 8, 7], device=device),
                       joint_v_adr=torch.tensor([10, 9, 8, 7, 6], device=device))
    actual, expected = actuator.get_command(data), Actuator.get_command(actuator, data)
    torch.testing.assert_close(actual.pos, expected.pos, rtol=0, atol=0)
    torch.testing.assert_close(actual.vel, expected.vel, rtol=0, atol=0)
