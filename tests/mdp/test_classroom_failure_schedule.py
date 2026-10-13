"""Failure-rate warmup/ramp, event scaling and checkpoint recovery."""
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
from src.tasks.amp_loco.mdp.path_navigation import classroom_failure_cost
from tests.mdp.test_classroom_throw_curriculum import make_env, make_runner


def failure_reward(env):
    env.max_episode_length_s = 25.
    env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
    env.termination_manager = NS(
        active_terms=['bad_base_height'],
        get_term=lambda _: torch.ones(env.num_envs, dtype=torch.bool))
    manager = RewardManager({'is_terminated': env.cfg.rewards['is_terminated']},
                            env, scale_by_dt=True)
    return manager.compute(dt=env.step_dt)


@pytest.mark.parametrize('steps_per_iteration', [12, 24, 48])
@pytest.mark.parametrize('iteration,rate', [
    (0, 2.), (999, 2.), (1000, 2.), (1001, 2.0028),
    (3500, 9.), (5999, 15.9972), (6000, 16.), (10000, 16.),
])
def test_configured_failure_reward_at_schedule_boundaries(steps_per_iteration, iteration, rate):
    env = make_env(steps_per_iteration)
    env.common_step_counter = iteration * steps_per_iteration
    assert failure_reward(env).tolist() == pytest.approx([-(10. + rate * 25.)] * 4)


@pytest.mark.parametrize('dt', [.01, .02, .1])
def test_scheduled_cost_preserves_failure_masks_and_remaining_time(dt):
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    terms = {
        'time_out': torch.tensor([False, False, False, False, True, True, False]),
        'ball_hit': torch.tensor([False, True, False, True, False, False, False]),
        'bad_base_height': torch.tensor([False, False, True, True, False, True, False]),
        'wall_hit': torch.tensor([False, False, False, False, False, False, True]),
        'navigation_update': torch.zeros(7, dtype=torch.bool),
    }
    env = NS(num_envs=7, device='cpu', step_dt=dt, max_episode_length_s=25.,
             common_step_counter=3500 * 24,
             episode_length_buf=torch.tensor([0, 0, round(5 / dt), round(5 / dt),
                                               round(25 / dt), round(25 / dt), round(5 / dt)]),
             termination_manager=NS(active_terms=list(terms), get_term=terms.__getitem__))
    manager = RewardManager({'is_terminated': cfg.rewards['is_terminated']},
                            env, scale_by_dt=True)
    # Mid-ramp rate is 9; ball-only and timeout-only events remain exempt.
    assert manager.compute(dt=dt).tolist() == pytest.approx([0., 0., -190., -190., 0., -10., -190.])


@pytest.mark.parametrize('new_steps', [12, 24, 48])
def test_resume_keeps_mid_ramp_rate_and_weights_only_restarts(monkeypatch, tmp_path, new_steps):
    env = make_env()
    runner = make_runner(monkeypatch, env)
    runner.current_learning_iteration = 3499
    env.common_step_counter = 3500 * 24
    path = str(tmp_path / 'model.pt')
    runner.save(path)
    restored = make_env(new_steps)
    new_runner = make_runner(monkeypatch, restored, new_steps)
    new_runner.load(path)
    assert failure_reward(restored).tolist() == pytest.approx([-235.] * 4)
    restored.common_step_counter += new_steps
    assert failure_reward(restored).tolist() == pytest.approx([-(10. + 9.0028 * 25.)] * 4)
    new_runner.load(path, load_cfg={'iteration': False})
    assert failure_reward(restored).tolist() == pytest.approx([-60.] * 4)


def test_play_uses_final_rate_and_schedule_parameters_are_configurable(monkeypatch):
    play = make_env()
    play.cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=True)
    assert failure_reward(play).tolist() == pytest.approx([-410.] * 4)
    monkeypatch.setenv('WALK_FAILURE_INITIAL_REMAINING_RATE', '3')
    monkeypatch.setenv('WALK_FAILURE_REMAINING_RATE', '11')
    monkeypatch.setenv('WALK_FAILURE_WARMUP_ITERS', '200')
    monkeypatch.setenv('WALK_FAILURE_RAMP_ITERS', '800')
    env = make_env()
    env.common_step_counter = 600 * 24
    assert failure_reward(env).tolist() == pytest.approx([-185.] * 4)


@pytest.mark.parametrize('params', [
    {'warmup_iterations': -1}, {'ramp_iterations': 0},
    {'initial_remaining_rate': -1.}, {'remaining_rate': 1.},
    {'remaining_rate': float('nan')}, {'initial_remaining_rate': float('inf')},
])
def test_invalid_failure_rate_schedule_is_rejected(params):
    kwargs = {'remaining_rate': 16., 'initial_remaining_rate': 2., **params}
    with pytest.raises(ValueError, match='failure rate schedule'):
        classroom_failure_cost(NS(), **kwargs)
