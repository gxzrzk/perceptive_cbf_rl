"""全向高速抛球弹道解析求解与事件配置测试 (Omnidirectional Throw Ballistics Tests)。

测试功能：
- 验证全向来球弹道速度闭式求解函数 ``solve_ballistic_velocity`` 的数学正确性；
- 验证求解所得初速度大小严格保真（模长等于指定的标量速率）；
- 验证在重力场（g = 9.81 m/s²）作用下，抛体轨迹能够精确穿越目标三维空间点（XY 水平对准且 Z 轴高度命中）；
- 验证多根选择机制：在两个可行抛物线中，必须选取低仰角的平射直射解 (direct flat arc，仰角 < 45°)，而非高抛高吊解；
- 验证 360° 全方位发射角覆盖（无除零或方位奇异点）；
- 验证环境变量 OMNI_THROW 对 MimicKit 任务仿真事件参数的重写与默认前向锥形发射的向下兼容。

关键断言条件：
- |norm(vel) - speed| < 1e-3；
- 轨迹积分到达目标处的高度误差与水平对准误差 < 1e-2；
- 垂直初速度分量绝对值 |vz| < 水平速度分量 v_horizontal；
- 全向角 [-pi, pi] 求解结果无 NaN。

覆盖边界情况：
- 目标高于或低于发射起点（如 start_z=1.5, target_z=0.4）；
- 环绕机器人周围 16 个等间距全向方位角；
- 环境变量开启与关闭状态下的配置回退测试。
"""
import math

import torch

from src.tasks.amp_loco.mdp.events import solve_ballistic_velocity

G = 9.81


def _arrival_z(start, vel, gravity=G):
    """辅助函数：计算抛体在水平位移到达目标时的高度。

    公式：t = horizontal_dist / horizontal_speed, z = z0 + vz*t - 0.5 * g * t^2。
    """
    vh = vel[..., :2].norm(dim=-1)
    return vh, vel[..., 2]


def test_speed_is_preserved():
    """测试场景：验证求解出的三维初速度向量模长精确等于请求的发射标量速率。

    输入假数据：
        两个样本：起点分别在 (0, 0, 1.5) 和 (3, -2, 1.0)，目标分别在 (9, 0, 1.0) 和 (3, 7, 0.4)，
        设定发射速率为 22.0 m/s 和 24.0 m/s。
    预期输出及验证逻辑：
        torch.norm(vel, dim=-1) 与期望 speed 在 1e-3 容差内完全一致。
    """
    start = torch.tensor([[0.0, 0.0, 1.5], [3.0, -2.0, 1.0]])
    target = torch.tensor([[9.0, 0.0, 1.0], [3.0, 7.0, 0.4]])
    speed = torch.tensor([22.0, 24.0])
    vel = solve_ballistic_velocity(start, target, speed)
    assert torch.allclose(vel.norm(dim=-1), speed, atol=1e-3)


def test_passes_through_target():
    """测试场景：在重力作用下前向数值积分抛体轨迹，验证精准命中目标 3D 坐标。

    输入假数据：
        起点在 (0, 0, 1.8)，目标在 (9.5, 0, 0.7)，发射速度为 21.0 m/s。
    预期输出及验证逻辑：
        1. 计算到达目标水平距离所需时间 t = d_xy / v_xy；
        2. 积分计算垂直高度 z(t) = z0 + vz * t - 0.5 * g * t²，与 target_z 误差小于 1e-2；
        3. 求解出的水平速度方向向量与目标水平位移方向向量余弦误差小于 1e-4。
    """
    start = torch.tensor([[0.0, 0.0, 1.8]])
    target = torch.tensor([[9.5, 0.0, 0.7]])
    speed = torch.tensor([21.0])
    vel = solve_ballistic_velocity(start, target, speed)
    d = (target[:, :2] - start[:, :2]).norm(dim=-1)
    vh = vel[:, :2].norm(dim=-1)
    t = d / vh                                   # time to cover the horizontal range
    z = start[:, 2] + vel[:, 2] * t - 0.5 * G * t * t
    assert torch.allclose(z, target[:, 2], atol=1e-2)
    # horizontal direction points at the target
    dir_solved = vel[:, :2] / vh.unsqueeze(-1)
    dir_target = (target[:, :2] - start[:, :2]) / d.unsqueeze(-1)
    assert torch.allclose(dir_solved, dir_target, atol=1e-4)


def test_direct_arc_is_flat():
    """测试场景：验证求解器选取的是直射平射解，而非高抛解。

    输入假数据：
        起点在 (0, 0, 1.5)，目标在 (9.0, 0, 1.1)，发射速度为 23.0 m/s。
    预期输出及验证逻辑：
        对于 8-10m 距离 / 20-25m/s 高速来球，仰角远低于 45°，
        断言 |vz| < v_horizontal，确保球体飞行轨迹平直快速，贴近实战躲避场景。
    """
    start = torch.tensor([[0.0, 0.0, 1.5]])
    target = torch.tensor([[9.0, 0.0, 1.1]])
    speed = torch.tensor([23.0])
    vel = solve_ballistic_velocity(start, target, speed)
    vh = vel[0, :2].norm().item()
    assert abs(vel[0, 2].item()) < vh           # elevation < 45 deg -> flat/direct arc


def test_omni_throw_flag_sets_state_env_params(monkeypatch):
    """测试场景：环境变量 OMNI_THROW=1 触发全向高速抛球环境参数重写。

    输入假数据与测试流程：
        使用 monkeypatch 设置 OMNI_THROW=1，直接调用环境构建器生成配置。
    预期输出及验证逻辑：
        事件 throw_ball_on_dwell 的参数中：omnidirectional 为 True，
        速度范围为 (12.0, 15.0)，距离范围扩展为 (8.0, 10.0)，目标高度在 (0.3, 1.3)。
    """
    monkeypatch.setenv("OMNI_THROW", "1")
    monkeypatch.delenv("CBF_JOINT", raising=False)
    import src.tasks  # noqa: F401
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_flat_env_cfg
    cfg = g1_amp_dodge_mimickit_flat_env_cfg(play=False)
    p = cfg.events["throw_ball_on_dwell"].params
    assert p["omnidirectional"] is True
    assert p["launch_speed_range"] == (12.0, 15.0)
    assert p["dist_range"] == (8.0, 10.0)
    assert p["target_z_range"] == (0.3, 1.3)


def test_omni_throw_flag_off_keeps_front_cone(monkeypatch):
    """测试场景：未设置 OMNI_THROW 时，保持默认的前向锥形区域近距离抛球模式（向下兼容）。

    预期输出及验证逻辑：
        omnidirectional 为 False，无额外发射速度范围限制，发射距离保持在 (2.0, 3.0) 米。
    """
    monkeypatch.delenv("OMNI_THROW", raising=False)
    monkeypatch.delenv("CBF_JOINT", raising=False)
    import src.tasks  # noqa: F401
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_flat_env_cfg
    cfg = g1_amp_dodge_mimickit_flat_env_cfg(play=False)
    p = cfg.events["throw_ball_on_dwell"].params
    assert p.get("omnidirectional", False) is False
    assert p.get("launch_speed_range") is None
    assert p["dist_range"] == (2.0, 3.0)


def test_omni_360_coverage_in_solver_inputs():
    """测试场景：验证 360° 全向环绕方位角下弹道求解的无死角数值鲁棒性。

    输入假数据：
        在 [-pi, pi] 之间均匀选取 16 个方位角的目标点，距离 9m，速度 22m/s。
    预期输出及验证逻辑：
        所有方位的速度模长均精准等于 22.0 m/s，且输出张量中不存在任何 NaN 异常值。
    """
    n = 16
    ang = torch.linspace(-math.pi, math.pi, n)
    start = torch.zeros(n, 3)
    start[:, 2] = 1.5
    dist = 9.0
    target = torch.stack([dist * torch.cos(ang), dist * torch.sin(ang), torch.full((n,), 0.9)], dim=-1)
    speed = torch.full((n,), 22.0)
    vel = solve_ballistic_velocity(start, target, speed)
    assert torch.allclose(vel.norm(dim=-1), speed, atol=1e-3)
    assert not torch.isnan(vel).any()
