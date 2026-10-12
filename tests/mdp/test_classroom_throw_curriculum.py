"""Iteration-based throw schedules, timer integration and checkpoint recovery."""
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.curriculums import (
    classroom_throw_interval_curriculum, classroom_training_iteration,
)
from src.tasks.amp_loco.mdp.events import reset_dodge_state, throw_ball_on_dwell
from src.tasks.amp_loco.rl.runner import AMPOnPolicyRunner, MjlabOnPolicyRunner


def make_env(steps_per_iteration=24):
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    terms = deepcopy(cfg.events)
    ball = NS(write_root_link_pose_to_sim=Mock(), write_root_link_velocity_to_sim=Mock())
    robot = NS(data=NS(root_link_lin_vel_w=torch.zeros(4, 3)))
    class Scene(dict):
        env_origins = torch.zeros(4, 3)
    return NS(cfg=cfg, common_step_counter=0, num_envs=4, device='cpu', step_dt=.02,
              scene=Scene(ball=ball, robot=robot),
              command_manager=NS(get_term=lambda _: NS(is_standing_env=torch.zeros(4, dtype=torch.bool))),
              event_manager=NS(get_term_cfg=terms.__getitem__),
              _classroom_rollout_steps=steps_per_iteration)


@pytest.mark.parametrize('steps_per_iteration', [12, 24, 48])
@pytest.mark.parametrize('iteration,expected', [
    (0, (4., 8.)), (4999, (4., 8.)), (5000, (4., 8.)),
    (6500, (3., 6.5)), (8000, (2., 5.)), (12000, (2., 5.)),
])
def test_schedule_boundaries_and_both_timers(steps_per_iteration, iteration, expected):
    env = make_env(steps_per_iteration)
    env.common_step_counter = iteration * steps_per_iteration
    env._dodge_throw_countdown = torch.tensor([10, 20, 30, 40])
    result = classroom_throw_interval_curriculum(env, torch.tensor([0]))
    for name in ('reset_dodge_state', 'throw_ball_on_dwell'):
        assert env.event_manager.get_term_cfg(name).params['throw_interval_range'] == pytest.approx(expected)
    assert result['iteration'] == iteration
    assert (result['interval_min_s'], result['interval_max_s']) == pytest.approx(expected)
    assert env._dodge_throw_countdown.tolist() == [10, 20, 30, 40]


def test_resets_sample_new_interval_and_step_countdowns_keep_advancing():
    env = make_env()
    env.common_step_counter = 6500 * 24
    classroom_throw_interval_curriculum(env)
    reset_params = env.event_manager.get_term_cfg('reset_dodge_state').params
    reset_dodge_state(env, None, **reset_params)
    first = env._dodge_throw_countdown.clone()
    assert ((first >= 150) & (first <= 325)).all()
    throw_params = env.event_manager.get_term_cfg('throw_ball_on_dwell').params
    throw_ball_on_dwell(env, None, **throw_params)
    assert torch.equal(env._dodge_throw_countdown, first - 1)
    env.common_step_counter = 8000 * 24
    classroom_throw_interval_curriculum(env, torch.tensor([1]))
    reset_dodge_state(env, torch.tensor([1]), **reset_params)
    assert 100 <= env._dodge_throw_countdown[1] <= 250
    assert torch.equal(env._dodge_throw_countdown[[0, 2, 3]], (first - 1)[[0, 2, 3]])


def test_training_and_play_configuration():
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    keys = list(cfg.events)
    assert keys.index('classroom_throw_interval_update') < keys.index('throw_ball_on_dwell')
    assert cfg.curriculum['classroom_throw_interval'].params['warmup_iterations'] == 5000
    assert cfg.curriculum['classroom_throw_interval'].params['ramp_iterations'] == 3000
    assert cfg.events['classroom_throw_interval_update'].params == cfg.curriculum['classroom_throw_interval'].params
    for name in ('reset_dodge_state', 'throw_ball_on_dwell'):
        assert cfg.events[name].params['throw_interval_range'] == (4., 8.)
    play = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=True)
    assert 'classroom_throw_interval' not in play.curriculum
    assert 'classroom_throw_interval_update' not in play.events
    for name in ('reset_dodge_state', 'throw_ball_on_dwell'):
        assert play.events[name].params['throw_interval_range'] == (2., 5.)
    wallwalk = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    assert 'classroom_throw_interval' not in wallwalk.curriculum


def make_runner(monkeypatch, env, steps=24):
    def initialize(self, wrapped_env, cfg, log_dir=None, device='cpu'):
        self.env, self.cfg = wrapped_env, cfg
        self.current_learning_iteration = 0
        self.alg = NS(save=lambda: {}, actor=NS(is_recurrent=True),
                      load=lambda data, load_cfg, strict: load_cfg is None or load_cfg.get('iteration', False))
        self.logger = NS(logger_type='tensorboard')
    monkeypatch.setattr(MjlabOnPolicyRunner, '__init__', initialize)
    return AMPOnPolicyRunner(NS(unwrapped=env), {'num_steps_per_env': steps})


@pytest.mark.parametrize('new_steps', [12, 24, 48])
def test_checkpoint_preserves_progress_when_rollout_length_changes(monkeypatch, tmp_path, new_steps):
    env = make_env()
    runner = make_runner(monkeypatch, env)
    runner.current_learning_iteration = 6499
    env.common_step_counter = 6500 * 24
    path = str(tmp_path / 'model.pt')
    runner.save(path, infos={'note': 'keep'})
    restored = make_env()
    new_runner = make_runner(monkeypatch, restored, new_steps)
    infos = new_runner.load(path)
    assert infos['note'] == 'keep'
    assert classroom_training_iteration(restored) == 6500
    assert restored.event_manager.get_term_cfg('reset_dodge_state').params['throw_interval_range'] == (3., 6.5)
    assert ((restored._dodge_throw_countdown >= 150) & (restored._dodge_throw_countdown <= 325)).all()
    restored.common_step_counter += new_steps
    assert classroom_training_iteration(restored) == 6501


def test_legacy_checkpoint_and_weights_only_loading(monkeypatch, tmp_path):
    path = str(tmp_path / 'legacy.pt')
    torch.save({'iter': 7999, 'infos': None}, path)
    env = make_env()
    runner = make_runner(monkeypatch, env)
    runner.load(path)
    assert classroom_training_iteration(env) == 8000
    assert env.event_manager.get_term_cfg('throw_ball_on_dwell').params['throw_interval_range'] == (2., 5.)
    runner.load(path, load_cfg={'iteration': False})
    assert classroom_training_iteration(env) == 0
    assert env.event_manager.get_term_cfg('throw_ball_on_dwell').params['throw_interval_range'] == (4., 8.)


@pytest.mark.parametrize('params', [
    {'ramp_iterations': 0}, {'warmup_iterations': -1},
    {'start_interval': (8., 4.)}, {'final_interval': (5., 9.)},
])
def test_invalid_schedule_is_rejected(params):
    with pytest.raises(ValueError, match='curriculum'):
        classroom_throw_interval_curriculum(make_env(), **params)
