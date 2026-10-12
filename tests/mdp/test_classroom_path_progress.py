"""Spawn-relative path rewards across forward, idle, retreat, turn and reset states."""
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommand
from src.tasks.amp_loco.mdp.classroom import round_classroom_route
from src.tasks.amp_loco.mdp.rewards import walk_path_progress_when_safe
from src.tasks.amp_loco.mdp.walk_path import walk_path_query
from tests.mdp.test_classroom_turns import make_command


@pytest.fixture
def term():
    # Retain coverage of the legacy helper used by WallWalk.
    return RewardTermCfg(func=walk_path_progress_when_safe, weight=2., params={
        "command_name": "twist", "v_max": .75,
        "stillness_penalty": .6, "stillness_threshold": .02,
    })


def progress_command(points=None):
    if points is None:
        points = [(10., -3.), (20., -3.), (20., 5.), (20., 15.)]
    command = make_command(points, points[0])
    command.cfg.path_allow_backtracking = True
    command._dodge_threat = torch.zeros(1, dtype=torch.bool)
    command._walk_forward_scale = torch.ones(1)
    env = command._env
    env.step_dt = .02
    env.command_manager = NS(get_term=lambda _: command)
    return command, env


def move(command, xy):
    command.robot.data.root_link_pos_w[0, :2] = torch.as_tensor(xy)
    DodgeGoToGoalCommand._update_path_goal(command)


def test_uniform_forward_rewards_and_idle_or_retreat_costs(term):
    command, env = progress_command()
    # Spawn has s=0 even though its world coordinates are nonzero.
    move(command, (10., -3.))
    assert command._path_s.item() == 0.
    assert term.func(env, **term.params).item() == pytest.approx(-.6)
    for step in range(1, 5):
        move(command, (10. + step * .012, -3.))
        assert term.func(env, **term.params).item() == pytest.approx(.6, abs=1e-4)
    # No progress, even while moving laterally, pays the fixed idle cost.
    move(command, (10.048, -2.5))
    assert term.func(env, **term.params).item() == pytest.approx(-.6)
    move(command, (10.042, -2.5))
    assert term.func(env, **term.params).item() == pytest.approx(-.3, abs=1e-4)


def test_reward_depends_on_progress_change_not_absolute_distance(term):
    command, env = progress_command()
    rewards = []
    for start in (10., 18.):
        move(command, (start, -3.))
        move(command, (start + .012, -3.))
        rewards.append(term.func(env, **term.params).clone())
    assert torch.allclose(rewards[0], rewards[1], atol=1e-4)
    assert rewards[0].item() == pytest.approx(.6, abs=1e-4)


def test_retreat_through_corner_and_behind_spawn(term):
    command, env = progress_command()
    move(command, (20., -2.8))
    assert command._path_seg_idx.item() == 1
    assert command._path_s.item() == pytest.approx(10.2)
    move(command, (19.8, -3.))
    assert command._path_seg_idx.item() == 0
    assert command._path_s.item() == pytest.approx(9.8)
    assert term.func(env, **term.params).item() < 0
    move(command, (10., -3.))
    move(command, (9.994, -3.))
    assert command._path_s.item() == pytest.approx(-.006, abs=1e-6)
    assert term.func(env, **term.params).item() == pytest.approx(-.3, abs=1e-4)


def test_turn_exemption_and_threat_gate(term):
    command, env = progress_command()
    move(command, (10., -3.))
    for scale in (0., .5, 1.):
        command._walk_forward_scale[:] = scale
        assert term.func(env, **term.params).item() == pytest.approx(-.6 * scale)
    command._walk_forward_scale.zero_()
    move(command, (9.994, -3.))
    assert term.func(env, **term.params).item() < 0  # Turning does not waive retreat.
    command._dodge_threat[:] = True
    for xy in ((10.006, -3.), (10.006, -3.), (10., -3.)):
        move(command, xy)
        assert term.func(env, **term.params).item() == 0.


@pytest.mark.parametrize("rate,expected", [
    (-10., -2.), (-.3, -.3), (-.01, -.6), (0., -.6), (.01, -.6),
    (.03, .03), (.6, .6), (10., .75),
])
def test_idle_deadband_and_rate_caps(term, rate, expected):
    command, env = progress_command()
    command._path_s_delta = torch.tensor([rate * env.step_dt])
    assert term.func(env, **term.params).item() == pytest.approx(expected)


def test_new_spawn_resets_progress_without_reward_spike(term):
    command, env = progress_command()
    move(command, (18., -3.))
    env._walk_path_pts += torch.tensor([100., 50.])
    env._walk_path_epoch += 1
    move(command, (110., 47.))
    assert command._path_s.item() == 0.
    assert command._path_s_delta.item() == 0.
    assert term.func(env, **term.params).item() == pytest.approx(-.6)
    move(command, (110.012, 47.))
    assert term.func(env, **term.params).item() == pytest.approx(.6, abs=2e-4)


def test_backtracking_does_not_jump_to_another_lap(term):
    command, env = progress_command([
        (0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.), (5., 0.),
    ])
    move(command, (1., 0.))
    move(command, (.994, 0.))
    assert command._path_seg_idx.item() == 0
    assert command._path_s.item() == pytest.approx(.994)
    assert term.func(env, **term.params).item() == pytest.approx(-.3, abs=1e-4)


def test_rounded_corner_progress_decreases_continuously_when_retraced(term):
    points, counts = round_classroom_route(
        torch.tensor([[[0., 0.], [5., 0.], [5., 4.], [5., 20.]]])
    )
    command, env = progress_command(points[0, :counts[0]].tolist())
    # Walk over the short arc segments, then retrace them through the corner.
    stations = list(torch.linspace(0., 8., 81)) + list(torch.linspace(7.9, 0., 80))
    last_s = 0.
    for station in stations:
        xy, _ = walk_path_query(env, torch.tensor([0]), station.reshape(1))
        move(command, xy[0])
        assert command._path_s.item() == pytest.approx(station.item(), abs=1e-4)
        delta = station.item() - last_s
        reward = term.func(env, **term.params).item()
        if delta > 1e-4:
            assert reward > 0.
        elif delta < -1e-4:
            assert reward < 0.
        last_s = station.item()


def test_classroom_reward_scaling_and_wallwalk_defaults(term):
    command, env = progress_command()
    env.num_envs, env.device = 1, "cpu"
    manager = RewardManager({"progress": term}, env, scale_by_dt=True)
    move(command, (10.012, -3.))
    task_reward = manager.compute(dt=env.step_dt).clone()
    assert task_reward.item() == pytest.approx(.024, abs=1e-5)
    assert (.5 * task_reward).item() == pytest.approx(.012, abs=1e-5)
    move(command, (10.012, -3.))
    assert (.5 * manager.compute(dt=env.step_dt)).item() == pytest.approx(-.012)
    wallwalk = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    assert not wallwalk.commands["twist"].path_allow_backtracking
    assert wallwalk.rewards["walk_path_progress_when_safe"].params == {"command_name": "twist"}
    assert walk_path_progress_when_safe(env).item() == 0.
