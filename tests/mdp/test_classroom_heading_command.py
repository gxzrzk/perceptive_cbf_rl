"""教室导航朝向指令与威胁门控响应测试 (Classroom Heading Command Tests)。

测试功能：
- 验证机器人在目标朝向大角度偏差下的“先转向后前行” (turn-before-walk) 行为；
- 验证偏差角在不同区间时前向速度 (vx) 的平滑过渡策略（大角度纯自转、中等角度平滑加速、小角度全速前行）；
- 验证自转角速度 (wz) 的方向选择（朝向目标的最短旋转方向）；
- 验证静止惩罚 (walk_path_stillness_when_safe) 与前向速度门控的严格对齐；
- 验证来球威胁 (dodge_threat) 对行走门控的即时释放 (release)：面临威胁时立刻允许全向运动（可后退或侧滑躲避），威胁解除后自动恢复转向门控；
- 验证目标到达 (arrived)、原地站立 (standing) 与定点抛球模式下的速度归零；
- 验证任务间配置隔离性：教室任务默认开启 turn_before_walk，而常规走廊避障 (WallWalk) 保持全向移动。

关键断言条件：
- 角度 > 75° 时：前向速度 vx == 0，执行纯原地旋转；
- 角度在 30°~75° 时：vx 单调递增，60° 时缩放因子为 0.5 (vx = 0.3 m/s)；
- 角度 <= 30° 时：达到全速 0.6 m/s；
- 突发来球威胁时：_dodge_threat 为 True，原先受限的倒退/侧移指令被瞬时放行；
- 威胁消失后：指令重新收拢归零。

覆盖边界情况：
- 正负对称大角度（如 ±120°、±180°）；
- 临界过渡角（75°、60°、30°、0°）；
- 启用与关闭 CBF 滤波器两种模式；
- 多环境不同威胁状态并发测试。
"""
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
    """构造包含指定朝向偏差角的模拟导航指令环境。

    参数：
        angles: 列表形式的一组朝向偏差角（单位：度）
    返回：
        (command, env): DodgeGoToGoalCommand 实例及配套的精简 Mock 环境对象
    """
    n = len(angles)
    heading = -torch.deg2rad(torch.tensor(angles, dtype=torch.float32))
    quat = torch.zeros(n, 4)
    quat[:, 0], quat[:, 3] = torch.cos(heading / 2), torch.sin(heading / 2)
    scene = Scene(
        robot=NS(data=NS(root_link_pos_w=torch.zeros(n, 3),
                         root_link_quat_w=quat, heading_w=heading,
                         root_link_ang_vel_w=torch.zeros(n, 3))),
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
    """测试场景：机器人朝向与目标存在不同角度偏差时的“先转后走”速度指令曲线与静止惩罚。

    输入假数据：
        覆盖 [180°, 120°, 90°, 75°, 60°, 45°, 30°, 0°, -60°, -120°] 的机器人朝向角。
    预期输出及验证逻辑：
        1. 偏差角 >= 75°（前3个角度 180°, 120°, 90°）时，前向速度 vx 严格为 0；
        2. 偏差角在 75° 到 30° 之间时，vx 随角度减小而单调平滑增加；
        3. 偏差角为 60° 时，vx 恰为 0.3 m/s（最大前向速度 0.6 m/s * 缩放因子 0.5）；
        4. 偏差角 <= 30°（30° 与 0°）时，vx 达到最大限制 0.6 m/s；
        5. 转向角速度 wz 符号与旋转方向严格匹配：目标在左侧则 wz > 0，在右侧则 wz < 0；
        6. 静止状态惩罚代价 walk_path_stillness_when_safe 与 vx 缩放比例精准同步。
    """
    command, env = make_command([180., 120., 90., 75., 60., 45., 30., 0., -60., -120.])
    command.cfg.cbf_enabled = cbf_enabled
    command._update_command()
    vx, vy, wz = command.command.unbind(-1)
    assert torch.allclose(vx[:3], torch.zeros(3), atol=1e-6)
    assert torch.all(vx[3:7].diff() > 0)
    assert vx[4].item() == pytest.approx(.3, abs=1e-6)  # max 0.6 m/s x scale(60deg)=0.5
    assert torch.allclose(vx[6:8], torch.full((2,), .6))
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
    """测试场景：来球威胁即时释放行走门控，威胁消除后自动恢复限制。

    输入假数据与测试流程：
        1. 初始状态：两个环境的机器人朝向与目标偏差 ±120°，均无威胁，前向指令被锁死；
        2. 环境 0 注入威胁：球位于 [2, 0, 0.8] 并以 [-4, 0, 0] 飞来；环境 1 保持安全；
        3. 更新指令，验证环境 0 瞬时放行全向速度（vx < 0 倒退躲避，vy > 0 侧滑躲避），环境 1 保持锁死；
        4. 移除威胁球，重新更新指令，验证环境 0 门控重新生效，速度指令再度归零。
    """
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
    """测试场景：机器人到达目标点、原地等待或原地抛球模式下的指令零值保持。

    输入假数据：
        3 个环境：环境 0 目标距离极小 (0.1m) 触发到达；环境 1 标记为 standing；环境 2 标记为 inplace。
    预期输出及验证逻辑：
        所有环境的 command 输出向量均为 0。
    """
    command, _ = make_command([0., 0., 0.])
    command.goal_pos_w[0, 0] = .1
    command.is_standing_env[1] = True
    command.is_inplace_env[2] = True
    command._update_command()
    assert torch.all(command.command == 0)


def test_other_tasks_retain_backward_commands_and_stillness():
    """测试场景：配置检查与跨任务回归测试。

    验证逻辑：
        1. WallWalk 走廊避障任务默认关闭 turn_before_walk，允许直接产生后退/侧向速度指令；
        2. 教室避障任务在训练与演示 (play) 模式下均必须启用 turn_before_walk。
    """
    assert not g1_amp_dodge_mimickit_wallwalk_flat_env_cfg().commands['twist'].turn_before_walk
    command, env = make_command([120.])
    command.cfg.turn_before_walk = False
    command._update_command()
    assert command.command[0, 0] < 0 and command.command[0, 1] > 0
    assert walk_path_stillness_when_safe(env).item() == 1.
    for play in (False, True):
        assert g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=play).commands['twist'].turn_before_walk
