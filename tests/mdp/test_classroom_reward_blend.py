"""Verify the exact classroom reward equation through both reward stages."""
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.config.g1.goto_rl_cfg import (
    g1_amp_dodge_mimickit_classroom_ppo_runner_cfg,
    g1_amp_dodge_mimickit_wallwalk_ppo_runner_cfg,
)
from src.tasks.amp_loco.rl.amp_discriminator import Discriminator


def sample_term(env, key):
    return env.samples[key]


@pytest.mark.parametrize('disc_output,style_score', [(-1., 0.), (0., .75), (1., 1.)])
def test_dt_scaled_task_sum_and_standing_style_blend(disc_output, style_score):
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    runner = g1_amp_dodge_mimickit_classroom_ppo_runner_cfg()
    # Three separate cases: heading reward, full stillness cost, termination cost.
    env = NS(num_envs=3, device='cpu', samples={
        'heading': torch.tensor([1., 0., 0.]),
        'stillness': torch.tensor([0., 1., 0.]),
        'termination': torch.tensor([0., 0., 1.]),
    })
    terms = {
        key: RewardTermCfg(func=sample_term, weight=weight, params={'key': key})
        for key, weight in [('heading', 2.), ('stillness', -2.), ('termination', -200.)]
    }
    manager = RewardManager(terms, env, scale_by_dt=cfg.scale_rewards_by_dt)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    assert dt == pytest.approx(.02)
    task_reward = manager.compute(dt=dt).clone()
    assert torch.allclose(task_reward, torch.tensor([.04, -.04, -4.]))
    assert torch.allclose(manager.compute(dt=dt / 2), task_reward / 2)
    discriminator = Discriminator(
        input_dim=4, hidden_layer_sizes=(4,), device='cpu',
        amp_reward_coef=runner.amp_reward_coef,
        task_reward_lerp=runner.amp_task_reward_lerp,
    )
    with torch.no_grad():
        for parameter in discriminator.parameters():
            parameter.zero_()
        discriminator.amp_linear.bias.fill_(disc_output)
    state = torch.zeros(3, 2)
    blended, _ = discriminator.predict_amp_reward(state, state, task_reward)
    assert torch.allclose(blended, torch.tensor([.02, -.02, -2.]) + .25 * style_score)


def test_registered_classroom_matches_standing_reward_scaling():
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    task = 'Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat'
    for play in (False, True):
        assert load_env_cfg(task, play=play).scale_rewards_by_dt
    runner = load_rl_cfg(task)
    assert runner.amp_reward_coef == .5 and runner.amp_task_reward_lerp == .5
    standing_task = 'Unitree-G1-AMP-Dodge-MimicKit-Flat'
    standing_env = load_env_cfg(standing_task)
    standing_runner = load_rl_cfg(standing_task)
    assert standing_env.scale_rewards_by_dt
    assert runner.amp_reward_coef == standing_runner.amp_reward_coef
    assert runner.amp_task_reward_lerp == standing_runner.amp_task_reward_lerp
    wall = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    wall_runner = g1_amp_dodge_mimickit_wallwalk_ppo_runner_cfg()
    assert wall.scale_rewards_by_dt
    assert wall_runner.amp_reward_coef == .5 and wall_runner.amp_task_reward_lerp == .5
