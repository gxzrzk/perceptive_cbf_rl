"""Exercise real command updates across heading and incoming-threat transitions."""
from types import SimpleNamespace as NS

import pytest
import torch

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommand
from src.tasks.amp_loco.mdp.rewards import walk_path_stillness_when_safe


class Scene(dict):
    pass


def make_command(angles):
    n = len(angles)
    heading = -torch.deg2rad(torch.tensor(angles, dtype=torch.float32))
    quat = torch.zeros(n, 4)
    quat[:, 0], quat[:, 3] = torch.cos(heading / 2), torch.sin(heading / 2)
    scene = Scene(
        robot=NS(data=NS(root_link_pos_w=torch.zeros(n, 3),
                         root_link_quat_w=quat, heading_w=heading)),
        ball=NS(data=NS(root_link_pos_w=torch.tensor([[0., 1000., .8]]).repeat(n, 1),
                        root_link_lin_vel_w=torch.zeros(n, 3))),
    )
    scene.env_origins = torch.zeros(n, 3)
    env = NS(num_envs=n, device='cpu', step_dt=.02, scene=scene,
             sim=NS(mj_model=NS(geom=lambda _: NS(id=0)),
                    model=NS(geom_size=torch.full((n, 1, 3), .1))))
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg().commands['twist']
    cfg.path_follow = False
    cfg.forward_offset = 0.
    command = DodgeGoToGoalCommand(cfg, env)
    command.goal_pos_w[:, 0] = 1.
    env.command_manager = NS(get_term=lambda _: command)
    return command, env


@pytest.mark.parametrize('cbf_enabled', [False, True])
def test_turn_then_smooth_forward_and_matching_stillness(cbf_enabled):
    command, env = make_command([180., 120., 90., 75., 60., 45., 30., 0., -60., -120.])
    command.cfg.cbf_enabled = cbf_enabled
    command._update_command()
    vx, vy, wz = command.command.unbind(-1)
    assert torch.allclose(vx[:3], torch.zeros(3), atol=1e-6)
    assert torch.all(vx[3:7].diff() > 0)
    assert vx[4].item() == pytest.approx(.4, abs=1e-6)
    assert torch.allclose(vx[6:8], torch.full((2,), .8))
    assert vx[8].item() == pytest.approx(vx[4].item())
    assert vx[9] == 0 and torch.all(vy == 0) and torch.all(vx >= 0)
    assert torch.all(wz[1:7] > 0) and torch.all(wz[8:] < 0) and wz[7] == 0
    assert torch.allclose(command.vel_command_nominal_b, command.command)
    cost = walk_path_stillness_when_safe(env)
    assert torch.allclose(cost[:3], torch.zeros(3), atol=1e-6)
    assert cost[4].item() == pytest.approx(.5, abs=1e-6)
    assert torch.allclose(cost[6:8], torch.ones(2))


@pytest.mark.parametrize('filter_command', [False, True])
def test_threat_releases_gate_immediately_and_safe_restores_it(filter_command):
    command, env = make_command([120., -120.])
    command.cfg.cbf_filter_command = filter_command
    command._update_command()
    assert not command._dodge_threat.any()
    assert torch.all(command.command[:, :2] == 0)
    # One environment receives a threatening ball, the other stays safe.
    ball = env.scene['ball'].data
    ball.root_link_pos_w[0] = torch.tensor([2., 0., .8])
    ball.root_link_lin_vel_w[0] = torch.tensor([-4., 0., 0.])
    command._update_command()
    assert command._dodge_threat.tolist() == [True, False]
    assert command.vel_command_nominal_b[0, 0] < 0
    assert command.vel_command_nominal_b[0, 1] > 0
    assert torch.all(command.command[1, :2] == 0)
    released = command.command[0].clone()
    command.cfg.turn_before_walk = False
    command._update_command()
    assert torch.allclose(command.command[0], released)
    command.cfg.turn_before_walk = True
    assert walk_path_stillness_when_safe(env)[0] == 0
    ball.root_link_pos_w[0, 1] = 1000.
    ball.root_link_lin_vel_w.zero_()
    command._update_command()
    assert not command._dodge_threat.any()
    assert torch.all(command.command[:, :2] == 0)


def test_arrived_standing_and_inplace_commands_remain_zero():
    command, _ = make_command([0., 0., 0.])
    command.goal_pos_w[0, 0] = .1
    command.is_standing_env[1] = True
    command.is_inplace_env[2] = True
    command._update_command()
    assert torch.all(command.command == 0)


def test_other_tasks_retain_backward_commands_and_stillness():
    assert not g1_amp_dodge_mimickit_wallwalk_flat_env_cfg().commands['twist'].turn_before_walk
    command, env = make_command([120.])
    command.cfg.turn_before_walk = False
    command._update_command()
    assert command.command[0, 0] < 0 and command.command[0, 1] > 0
    assert walk_path_stillness_when_safe(env).item() == 1.
    for play in (False, True):
        assert g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=play).commands['twist'].turn_before_walk
