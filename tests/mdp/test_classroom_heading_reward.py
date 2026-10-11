"""Misaligned robots must actually turn toward the goal at a reasonable rate."""
import math

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.rewards import walk_goal_heading_when_safe
from tests.mdp.test_classroom_heading_command import make_command


@pytest.fixture
def term():
    return g1_amp_dodge_mimickit_classroom_flat_env_cfg().rewards[
        "walk_goal_heading_when_safe"
    ]


@pytest.mark.parametrize("angle", [30.001, 31., 60., 90., 120., 179., -30.001, -31., -90., -179.])
def test_reasonable_turns_pay_but_stopping_wrong_way_and_overspeed_cost(term, angle):
    command, env = make_command([angle] * 5)
    command._update_command()
    target = command.command[:, 2].clone()
    yaw_rates = target * torch.tensor([1., 0., -1., 3., .1])
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = yaw_rates
    reward = term.func(env, **term.params)
    assert reward[0].item() == pytest.approx(1.)
    assert torch.all(reward[1:] < 0.)
    assert reward[1].item() == -1. and reward[2].item() == -1.


def test_zero_or_wrong_way_rate_remains_negative_with_wide_tolerance():
    command, env = make_command([60., 60.])
    command._update_command()
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([0., -.5])
    assert torch.all(walk_goal_heading_when_safe(env, turn_velocity_std=2.) == -1.)


@pytest.mark.parametrize("angle", [0., 20., 30., -20., -30.])
def test_aligned_and_exact_threshold_keep_heading_reward(term, angle):
    _, env = make_command([angle, angle])
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([0., -.5])
    result = term.func(env, **term.params)
    expected = math.cos(math.radians(angle)) ** 4
    assert torch.allclose(result, torch.full((2,), expected), atol=1e-6)


def test_threat_and_coincident_goal_disable_both_branches(term):
    command, env = make_command([60., 60., 20., 20., 90.])
    command._dodge_threat[:] = torch.tensor([True, True, True, True, False])
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([.5, -.5, .5, 0., .5])
    command.goal_pos_w[4] = 0.
    assert torch.all(term.func(env, **term.params) == 0.)


def test_current_bearing_controls_turn_direction_not_cached_command(term):
    command, env = make_command([60.])
    command._update_command()
    assert command.command[0, 2] > 0.
    # Change the current bearing without refreshing the cached command.
    command.goal_pos_w[0] = torch.tensor([-1., -1.])
    env.scene["robot"].data.root_link_ang_vel_w[0, 2] = -.5
    assert term.func(env, **term.params).item() == pytest.approx(1.)
    env.scene["robot"].data.root_link_ang_vel_w[0, 2] = .5
    assert term.func(env, **term.params).item() == -1.


def test_pi_wrap_chooses_short_turn_direction(term):
    command, env = make_command([-179.])
    command.goal_pos_w[0] = torch.tensor([
        math.cos(math.radians(-179.)), math.sin(math.radians(-179.))
    ])
    env.scene["robot"].data.root_link_ang_vel_w[0, 2] = .5
    # Heading +179deg and bearing -179deg differ by +2deg, not -358deg.
    assert term.func(env, **term.params).item() == pytest.approx(
        math.cos(math.radians(2.)) ** 4, abs=1e-6
    )


def test_turn_target_respects_controller_gain_and_speed_limit(term):
    command, env = make_command([60., 60.])
    command.cfg.kp_yaw = .2
    command.cfg.max_ang_vel_z = .4
    desired = .2 * math.radians(60.)
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([desired, .5])
    score = term.func(env, **term.params)
    assert score[0].item() == pytest.approx(1.) and score[1] < 0.


def test_root_world_z_rate_and_real_reward_scaling(term):
    command, env = make_command([90., 90., 90.])
    env.scene["robot"].data.root_link_ang_vel_w[:] = torch.tensor([
        [3., -2., .5], [0., 0., 0.], [0., 0., -.5]
    ])
    manager = RewardManager({"heading": term}, env, scale_by_dt=True)
    task = manager.compute(dt=env.step_dt)
    assert torch.allclose(.5 * task, torch.tensor([.02, -.02, -.02]), atol=1e-6)


@pytest.mark.parametrize("std", [0., -.1])
def test_invalid_turn_tolerance_is_rejected(std):
    _, env = make_command([90.])
    with pytest.raises(ValueError, match="turn_velocity_std"):
        walk_goal_heading_when_safe(env, turn_velocity_std=std)
