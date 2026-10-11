"""教室导航目标朝向奖励函数测试 (Classroom Heading Reward Tests)。

测试功能：
- 验证当机器人朝向与目标存在角度偏差时，安全状态下的朝向对齐奖励 ``walk_goal_heading_when_safe``；
- 验证转向角速度激励与惩罚机制：朝目标合理角速度转向时获得满分奖励 (+1.0)，静止不转或反向旋转受到严厉惩罚 (-1.0)，过快超速旋转受到扣分；
- 验证小角度对齐状态（偏差角 <= 30°）下的朝向余弦指数平滑奖励（如 cos(angle)^sharpness）；
- 验证来球威胁 (dodge_threat) 与目标重合情况下的奖励清零门控（避障躲避时不强求朝向目标）；
- 验证转向方向由实时方位角 (bearing) 动态决定，而非依赖过期的缓存指令；
- 验证 [-pi, pi] 角度环绕逻辑：当在 ±180° 附近跨越周期时，算法能正确选择最小转向角（例如 +179° 到 -179° 实际仅相差 2°）；
- 验证角速度目标受控制器比例增益 kp_yaw 和最大角速度 max_ang_vel_z 的物理约束。

关键断言条件：
- 转向速度匹配目标时：reward == 1.0；静止或反向旋转：reward == -1.0；
- 小角度对齐时：reward == cos(rad)^4；
- 存在威胁时：reward == 0.0；
- 周期环绕角度差为 2° 时：reward == cos(2°)^4；
- 非法参数（如 turn_velocity_std <= 0）：抛出 ValueError。

覆盖边界情况：
- 临界对齐阈值（±30.001° vs ±30°）；
- 极端大角度（±179°、跨越 ±pi 奇点）；
- 各种角速度比例（1x、0x、-1x、3x超速、0.1x极慢）；
- RewardManager 的 dt 时间步缩放集成。
"""
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
    """获取教室环境配置中注册的 walk_goal_heading_when_safe 奖励配置项。"""
    return g1_amp_dodge_mimickit_classroom_flat_env_cfg().rewards[
        "walk_goal_heading_when_safe"
    ]


@pytest.mark.parametrize("angle", [30.001, 31., 60., 90., 120., 179., -30.001, -31., -90., -179.])
def test_reasonable_turns_pay_but_stopping_wrong_way_and_overspeed_cost(term, angle):
    """测试场景：当朝向偏差超出阈值时，以合理速度转向获得满分，而不转向、反向转或超速转则受罚。

    输入假数据：
        设置 5 个环境，朝向偏差角为 angle（> 30°），机器人的航向角速度 yaw_rate 分别设置为
        目标值的 [1.0倍, 0.0倍(停滞), -1.0倍(反向), 3.0倍(严重超速), 0.1倍(转动过慢)]。
    预期输出及验证逻辑：
        1. 完美匹配期望转向速度 (1.0倍) 时，奖励恰好为 1.0；
        2. 其他 4 种情况的奖励均小于 0；
        3. 停滞不转 (0倍) 与反向旋转 (-1倍) 均受到全额惩罚 -1.0。
    """
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
    """测试场景：即使调大角速度容忍标准差 (turn_velocity_std=2.0)，静止或反向旋转仍必须为负值惩罚。

    输入假数据：
        2 个偏差 60° 的环境，航向角速度分别为 0 与 -0.5 rad/s。
    预期输出及验证逻辑：
        两者的奖励输出值均严格保持为 -1.0。
    """
    command, env = make_command([60., 60.])
    command._update_command()
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([0., -.5])
    assert torch.all(walk_goal_heading_when_safe(env, turn_velocity_std=2.) == -1.)


@pytest.mark.parametrize("angle", [0., 20., 30., -20., -30.])
def test_aligned_and_exact_threshold_keep_heading_reward(term, angle):
    """测试场景：当偏差角在 30° 阈值以内时，平滑退化为静态航向对齐奖励，不受角速度影响。

    输入假数据：
        角度分别取 0°, ±20°, ±30°，设置不同的角速度 (0.0 与 -0.5 rad/s)。
    预期输出及验证逻辑：
        奖励计算退化为静态对齐项：cos(angle)^4，两个环境的奖励值严格等于该理论值。
    """
    _, env = make_command([angle, angle])
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([0., -.5])
    result = term.func(env, **term.params)
    expected = math.cos(math.radians(angle)) ** 4
    assert torch.allclose(result, torch.full((2,), expected), atol=1e-6)


def test_threat_and_coincident_goal_disable_both_branches(term):
    """测试场景：来球威胁或目标重合时，朝向奖励完全关闭（置 0）。

    输入假数据：
        前 4 个环境标记为有威胁 (_dodge_threat = True)；第 5 个环境目标点与机器人重合。
    预期输出及验证逻辑：
        所有环境的奖励值输出均为 0.0，避免机器人在躲避危险或已到达目标时强行对齐。
    """
    command, env = make_command([60., 60., 20., 20., 90.])
    command._dodge_threat[:] = torch.tensor([True, True, True, True, False])
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([.5, -.5, .5, 0., .5])
    command.goal_pos_w[4] = 0.
    assert torch.all(term.func(env, **term.params) == 0.)


def test_current_bearing_controls_turn_direction_not_cached_command(term):
    """测试场景：验证转向奖励基于当前几何方位角 (bearing) 实时计算，而非依赖过期的指令缓存。

    输入假数据与测试流程：
        1. 初始指令缓存为正向偏转；
        2. 在不重新触发 _update_command 的情况下将目标点修改到反方向；
        3. 机器人的转向角速度如果配合新的目标方向，应获得奖励；若配合过期的缓存指令方向，则受罚。
    预期输出及验证逻辑：
        转向角速度为 -0.5 时奖励为 1.0，转向角速度为 +0.5 时奖励为 -1.0。
    """
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
    """测试场景：跨越 ±pi 边界时的环绕最短路径判定。

    输入假数据：
        机器人朝向为 +179°，目标方向为 -179°。
    预期输出及验证逻辑：
        两者的实际最短夹角仅为 2°（小于 30° 阈值），奖励值应等于 cos(2°)^4，
        证明算法没有错误地当作 358° 大角度偏差而触发原地掉头转向惩罚。
    """
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
    """测试场景：期望转向角速度的计算严格遵守控制器增益 (kp_yaw) 与最大角速度上限。

    输入假数据：
        设置 kp_yaw = 0.2，max_ang_vel_z = 0.4，朝向偏差 60°。
        期望速度 = 0.2 * (60 * pi / 180) ≈ 0.2094 rad/s。
    预期输出及验证逻辑：
        航向角速度为期望值时得分 1.0，过高速度 (0.5 rad/s) 得分为负。
    """
    command, env = make_command([60., 60.])
    command.cfg.kp_yaw = .2
    command.cfg.max_ang_vel_z = .4
    desired = .2 * math.radians(60.)
    env.scene["robot"].data.root_link_ang_vel_w[:, 2] = torch.tensor([desired, .5])
    score = term.func(env, **term.params)
    assert score[0].item() == pytest.approx(1.) and score[1] < 0.


def test_root_world_z_rate_and_real_reward_scaling(term):
    """测试场景：结合 RewardManager 验证多维度角速度输入与时间步长 dt 缩放机制。

    输入假数据：
        3 个环境输入不同的 3 轴角速度向量，由 RewardManager 在 scale_by_dt=True 模式下计算。
    预期输出及验证逻辑：
        输出的单步任务奖励乘以权重因子后，与期望的缩放值精确匹配。
    """
    command, env = make_command([90., 90., 90.])
    env.scene["robot"].data.root_link_ang_vel_w[:] = torch.tensor([
        [3., -2., .5], [0., 0., 0.], [0., 0., -.5]
    ])
    manager = RewardManager({"heading": term}, env, scale_by_dt=True)
    task = manager.compute(dt=env.step_dt)
    assert torch.allclose(.5 * task, torch.tensor([.02, -.02, -.02]), atol=1e-6)


@pytest.mark.parametrize("std", [0., -.1])
def test_invalid_turn_tolerance_is_rejected(std):
    """测试场景：异常参数防御——传入非正的标准差容差。

    预期输出及验证逻辑：
        抛出 ValueError 异常，异常信息匹配 'turn_velocity_std'。
    """
    _, env = make_command([90.])
    with pytest.raises(ValueError, match="turn_velocity_std"):
        walk_goal_heading_when_safe(env, turn_velocity_std=std)
