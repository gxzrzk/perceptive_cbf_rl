"""教室路径进度奖励与倒退/停滞惩罚测试 (Classroom Path Progress Tests)。

测试功能：
- 验证教室导航任务中的沿路径前进奖励 ``walk_path_progress_when_safe``；
- 验证出生点相对坐标机制：即使出生点的世界坐标非零，路径弧长进度起点始终归零 (s = 0)；
- 验证前向行进奖励的正向激励（由进度增量 delta_s 决定，与绝对路径距离无关）；
- 验证停滞不动（侧向移动或原地踏步）时的固定怠速惩罚 (-0.6)，以及沿路径后退倒退时的负奖励；
- 验证路径拐角倒退与回到出生点后方时的平滑回退过渡；
- 验证原地转向期间对怠速惩罚的豁免衰减机制；
- 验证来球威胁 (dodge_threat) 对进度奖励与怠速惩罚的即时清零门控（躲球期间免除怠速与后退惩罚）；
- 验证圆弧倒角拐弯路径 (rounded route) 上的连续平滑进度计算。

关键断言条件：
- 前向正常步进：reward ≈ +0.6；
- 停滞无进展：reward ≈ -0.6；
- 沿路径后退：reward < 0；
- 转向比例为 0 时：停滞惩罚缩放为 0；
- 存在威胁时：无论前进还是后退，奖励项输出强制为 0.0；
- 新剧集重置路径时：无跳跃性超大奖励脉冲（delta 自动平滑重置为 0）。

覆盖边界情况：
- 回退穿越分段拐点与退到出生点起点后方 (s < 0)；
- 自身相交/重合自闭环路径的回退，不错误跳跃到后续圈数；
- 极快行进速率与极大后退速率下的饱和截断；
- 圆弧平滑过渡拐角处的微小微分变化。
"""
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager
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
    """获取教室环境配置中注册的 walk_path_progress_when_safe 奖励配置项。"""
    return g1_amp_dodge_mimickit_classroom_flat_env_cfg().rewards[
        "walk_path_progress_when_safe"
    ]


def progress_command(points=None):
    """构造用于测试路径进度的模拟指令环境。

    参数：
        points: 路径折线顶点序列（2D 坐标列表），若为 None 则默认生成包含拐弯的 4 节点路径。
    """
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
    """辅助函数：传送机器人到指定 (x, y) 坐标并更新路径目标追踪状态。"""
    command.robot.data.root_link_pos_w[0, :2] = torch.as_tensor(xy)
    DodgeGoToGoalCommand._update_path_goal(command)


def test_uniform_forward_rewards_and_idle_or_retreat_costs(term):
    """测试场景：沿路径前行获取正奖励，纯侧向无进展支付停滞惩罚，后退支付惩罚。

    输入假数据与测试流程：
        1. 机器人位于出生点 (10, -3)，弧长进度 s=0；
        2. 原地未动时，支付怠速惩罚 -0.6；
        3. 连续 4 步向前移动 0.012m（折合 0.6 m/s），每步均稳定获得 +0.6 正奖励；
        4. 仅侧向偏离而不前行，继续支付 -0.6 怠速惩罚；
        5. 沿路径倒退时，扣减倒退惩罚。
    """
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
    """测试场景：验证奖励严格取决于步间进度差值 (delta_s)，与当前所在的绝对距离无关。

    输入假数据：
        分别在路径起点 (X=10) 与路径中段 (X=18) 执行完全相同的 0.012m 前向移动。
    预期输出及验证逻辑：
        两次获得的奖励均为 +0.6，且两者绝对误差小于 1e-4。
    """
    command, env = progress_command()
    rewards = []
    for start in (10., 18.):
        move(command, (start, -3.))
        move(command, (start + .012, -3.))
        rewards.append(term.func(env, **term.params).clone())
    assert torch.allclose(rewards[0], rewards[1], atol=1e-4)
    assert rewards[0].item() == pytest.approx(.6, abs=1e-4)


def test_retreat_through_corner_and_behind_spawn(term):
    """测试场景：倒退越过拐点分段以及退到出生点原点后方 (s < 0) 的分段索引回退。

    输入假数据与测试流程：
        1. 移动到拐角后方 (20, -2.8)，确认处于分段 1，s ≈ 10.2m；
        2. 倒退回拐角前 (19.8, -3.0)，确认分段成功退回 0，s ≈ 9.8m，奖励为负；
        3. 倒退到出生点起点后方 (9.994, -3.0)，进度 s 为负值 (-0.006m)，惩罚值为 -0.3。
    """
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
    """测试场景：原地转向期间怠速惩罚按比例衰减，来球威胁激活时奖励项全部关闭。

    输入假数据与测试流程：
        1. 当行走前向比例 _walk_forward_scale 分别为 0, 0.5, 1.0 时，停滞惩罚分别为 0, -0.3, -0.6；
        2. 转向仅免除原地怠速，不豁免倒退惩罚（后退仍受罚）；
        3. 威胁激活 (_dodge_threat = True) 时，不论前行、后退或停滞，奖励一律为 0.0。
    """
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
    """测试场景：速度死区、线性区及最大上下限截断。

    输入假数据：
        覆盖从 -10 m/s（严重倒退）到 +10 m/s（极速前行）的不同行进速率。
    预期输出及验证逻辑：
        1. 极速倒退 (-10 m/s) 截断在下限 -2.0；
        2. 微小速度在 [-0.01, 0.01] 死区内归为停滞惩罚 -0.6；
        3. 正常行进速度线性计分；
        4. 极速前行 (+10 m/s) 截断在上限 +0.75。
    """
    command, env = progress_command()
    command._path_s_delta = torch.tensor([rate * env.step_dt])
    assert term.func(env, **term.params).item() == pytest.approx(expected)


def test_new_spawn_resets_progress_without_reward_spike(term):
    """测试场景：新剧集重置路径时，进度起点重置，不发生奖励异常突变。

    输入假数据与测试流程：
        1. 先在原路径行进至 X=18；
        2. 触发新一轮路径重置（epoch 递增，路径顶点整体平移到 100m 外）；
        3. 机器人出现在新出生点 (110, 47)，进度 s 与 delta 自动归零；
        4. 首次计算读取怠速惩罚 -0.6（无由于坐标跳变导致的数十米虚假 delta 惩罚或奖励）；
        5. 继续迈出第一步正常获得 +0.6 奖励。
    """
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
    """测试场景：自相交/回环路径在拐弯回退时不错误跳转至后续圈数。

    输入假数据：
        一条构成回环并开始第二圈的折线路径。
    预期输出及验证逻辑：
        微量后退时，分段索引仍稳定锚定在第 0 段，进度平稳退回 0.994m，避免跳跃。
    """
    command, env = progress_command([
        (0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.), (5., 0.),
    ])
    move(command, (1., 0.))
    move(command, (.994, 0.))
    assert command._path_seg_idx.item() == 0
    assert command._path_s.item() == pytest.approx(.994)
    assert term.func(env, **term.params).item() == pytest.approx(-.3, abs=1e-4)


def test_rounded_corner_progress_decreases_continuously_when_retraced(term):
    """测试场景：圆弧平滑倒角拐弯路径 (rounded corner) 上的单调连续性。

    输入假数据：
        由 round_classroom_route 生成带圆弧倒角的直角弯折路径。
    预期输出及验证逻辑：
        沿圆弧采样 81 个递增桩号并反向采样 80 个递减桩号：
        前进阶段奖励恒大于 0，倒退折返阶段奖励恒小于 0，无几何奇异点。
    """
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
    """测试场景：RewardManager dt 缩放与任务间默认参数一致性。

    预期输出及验证逻辑：
        1. 在 dt=0.02s 缩放模式下计算单步奖励，缩放后数值准确；
        2. WallWalk 任务默认关闭 path_allow_backtracking，且默认不带怠速惩罚。
    """
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
