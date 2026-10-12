"""教室任务奖励混合与 AMP 判别器风格奖励加权测试 (Classroom Reward Blend Tests)。

测试功能：
- 验证任务奖励在两个关键计算阶段的精确方程：
  第一阶段：RewardManager 基于时间步长 dt 对各任务奖励项求和加权；
  第二阶段：AMP 判别器 (Discriminator) 将任务奖励与风格判别奖励进行插值混合 (blend)；
- 验证判别器输出在不同取值范围（如 -1.0, 0.0, 1.0）下风格得分的映射计算；
- 验证教室导航任务与基准站立任务 (Standing Task) 在奖励缩放参数配置上的严格对齐。

关键断言条件：
- dt 缩放：dt = 0.02s 时，权重为 2 的项计算结果为 0.04；若 dt 减半，计算结果精确减半；
- AMP 风格混合：blended == task_reward * lerp + amp_coef * style_score；
- 参数对齐：scale_rewards_by_dt 为 True，amp_reward_coef == 0.5，amp_task_reward_lerp == 0.5。

覆盖边界情况：
- 判别器输出极端值 (-1, 0, +1)；
- 正奖励（朝向）、负惩罚（停滞）与大额终止惩罚（-200）的综合叠加；
- 跨任务（Classroom, Standing, WallWalk）的超参数一致性校验。
"""
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
    """用于测试的模拟奖励项函数，直接从环境样本中读取指定键的值。"""
    return env.samples[key]


@pytest.mark.parametrize('disc_output,style_score', [(-1., 0.), (0., .75), (1., 1.)])
def test_dt_scaled_task_sum_and_standing_style_blend(disc_output, style_score):
    """测试场景：多奖励项的 dt 缩放加权求和，以及 AMP 判别器对任务奖励与对抗风格奖励的混合。

    输入假数据：
        3 个环境样本：朝向对齐正奖励 (+1.0)、静止停滞惩罚 (+1.0 配合权重 -2.0)、终止大额惩罚 (+1.0 配合权重 -200.0)。
    预期输出及验证逻辑：
        1. dt = 0.02s 时，任务总奖励为 [.04, -.04, -4.]；
        2. dt 减半时，任务总奖励严格减半；
        3. 判别器权重归零、偏置设置为 disc_output，混合奖励输出严格等于
           0.5 * task_reward + 0.25 * style_score。
    """
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    runner = g1_amp_dodge_mimickit_classroom_ppo_runner_cfg()
    # 构造 3 个独立测试用例环境：分别激活航向对齐奖励、完全停滞代价、终止惩罚 (Three separate cases: heading reward, full stillness cost, termination cost).
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
    """测试场景：验证注册的教室任务奖励参数与标准站立任务及 WallWalk 任务保持一致。

    验证逻辑：
        1. scale_rewards_by_dt 在所有任务中均为 True；
        2. amp_reward_coef == 0.5 且 amp_task_reward_lerp == 0.5，确保风格与任务目标平衡一致。
    """
    import src.tasks  # noqa: F401 -- 触发任务注册（通过副作用导入注册所有任务配置环境） (triggers task registration via side-effect import)
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
