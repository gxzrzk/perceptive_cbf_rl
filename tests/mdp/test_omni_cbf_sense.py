"""全向投球下控制障碍函数 (CBF) 感知半径与威胁边界测试 (Omni CBF Sense Radius Tests)。

测试功能：
- 验证 CBF 感知半径在全向高速来球场景 (OMNI_THROW=1) 下从 4m 扩展至 12m 的自适应缩放机制；
- 证明该修改的核心必要性：
  全向抛球模式下，小球生成在距离机器人 8-10m 的远端，初速度达 20-25 m/s；
  1. 当 sense_radius = 12m 时，小球刚发射即可落入感知球内（dist < 12m），CBF 威胁门控 (threat gate)
     在发射瞬间即可立刻触发，为机器人提供足够的躲避前置反应时间（约 0.4s）；
  2. 当 sense_radius = 4m（旧版默认值）时，发射瞬间小球处于感知盲区（dist > 4m），CBF 门控无法触发，
     直到小球飞至 4m 以内（仅剩约 0.18s）才响应，导致人形机器人无法来得及避让高速球。

关键断言条件：
- sense_radius == 12.0m 时：发射瞬时受威胁环境占比 >= 80%；
- sense_radius == 4.0m 时：发射瞬时受威胁环境占比 <= 5%。

覆盖边界情况：
- 全方位 360 度随机发射方位角 (bearing in [-pi, pi])；
- 8-10m 随机远距离与 20-25m/s 高速弹道求解；
- MuJoCo Warp GPU 动力学单步前向传播与 CBF 动作处理流程同步。
"""
import math
import os

import torch
import pytest


# ---------------------------------------------------------------------------
# 辅助构建与场景状态注入函数
# ---------------------------------------------------------------------------

def _build_state_cbf_env(num_envs: int, sense_radius_override: str | None = None):
    """构建启用 OMNI_THROW=1 和 CBF_JOINT=1 的 MimicKit 状态环境。

    说明：
        直接调用配置构建函数以在调用时动态获取环境变量，支持覆写 CBF_SENSE_RADIUS。
    """
    if sense_radius_override is not None:
        os.environ["CBF_SENSE_RADIUS"] = sense_radius_override
    else:
        os.environ.pop("CBF_SENSE_RADIUS", None)
    os.environ.update({
        "OMNI_THROW": "1",
        "CBF_JOINT": "1",
        "CBF_JOINT_APPLY_FILTER": "0",   # 仅用于奖励塑形，不直接覆盖动作
        "CBF_JOINT_LAMBDA_CORR": "0",
        "CBF_JOINT_LAMBDA_BUF": "0",
    })

    import src.tasks  # noqa: F401 -- 触发任务注册副作用
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_flat_env_cfg

    cfg = g1_amp_dodge_mimickit_flat_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    env.reset()
    return env


def _place_omni_ball_and_run_cbf(env):
    """写入全向抛球状态（8-10m 距离、20-25m/s 速度、全向方位），前向物理动力学并运行 CBF 处理。

    测试数据构造：
        1. 机器人当前位置：rp；
        2. 球体发射方位角：在 [-pi, pi] 均匀采样；
        3. 发射距离：dist 采样自 U(8, 10) 米；
        4. 发射高度：采样自 U(1.5, 2.3) 米；
        5. 球速：采样自 U(20, 25) m/s；
        6. 调用 solve_ballistic_velocity 求解命中机器人的抛物线弹道速度向量；
        7. 同步物理缓冲区并调用 process_action，返回 env._cbf_joint_threat 布尔张量。
    """
    from src.tasks.amp_loco.mdp.events import solve_ballistic_velocity
    import mujoco_warp as mjw

    G = 9.81
    n = env.num_envs
    dev = env.device

    g = torch.Generator(device=dev).manual_seed(42)
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    rp = robot.data.root_link_pos_w  # 机器人基座世界坐标，形状为 (N, 3) (robot root position in world frame)

    bearing = torch.rand(n, generator=g, device=dev) * (2 * math.pi) - math.pi
    dist = torch.rand(n, generator=g, device=dev) * 2.0 + 8.0  # 均匀分布采样发射距离 U(8, 10) 米
    start = torch.empty(n, 3, device=dev)
    start[:, 0] = rp[:, 0] + dist * torch.cos(bearing)
    start[:, 1] = rp[:, 1] + dist * torch.sin(bearing)
    start[:, 2] = torch.rand(n, generator=g, device=dev) * 0.8 + 1.5  # 均匀分布采样发射点起始高度 U(1.5, 2.3) 米

    speed = torch.rand(n, generator=g, device=dev) * 5.0 + 20.0  # 均匀分布采样发射标量速率 U(20, 25) m/s
    target = rp.clone()
    target[:, 2] = torch.rand(n, generator=g, device=dev) * 1.0 + 0.3  # 均匀分布采样命中目标高度 U(0.3, 1.3) 米

    vel = solve_ballistic_velocity(start, target, speed, gravity=G)

    quat = torch.zeros(n, 4, device=dev)
    quat[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(torch.cat([start, quat], dim=-1))
    rv = torch.zeros(n, 6, device=dev)
    rv[:, 0:3] = vel
    ball.write_root_link_velocity_to_sim(rv)

    # 将写入的数据同步至物理引擎缓冲区，刷新 ball.data
    env.scene.write_data_to_sim()
    mjw.forward(env.sim.wp_model, env.sim.wp_data)
    ball.update(env.step_dt)

    # 执行 CBF 动作项处理（读取当前球体世界坐标与速度）
    zero_action = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    env.action_manager.process_action(zero_action)

    return env._cbf_joint_threat


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_sense_radius_12_fires_at_omni_launch():
    """测试场景：在扩展后的感知半径 (sense_radius=12m) 下，全向来球在 8-10m 远端发射瞬间即触发威胁。

    预期输出及验证逻辑：
        1. 验证环境正确加载了 OMNI_THROW=1 默认的 sense_radius=12.0；
        2. 发射 8-10m 远端小球后，由于距离小于 12m，至少 80% 的环境立即判定为受威胁 (threat=True)。
    """
    env = _build_state_cbf_env(num_envs=64)
    term = env.action_manager.get_term("joint_pos")

    # 确认环境正确加载了 sense_radius=12.0（OMNI_THROW=1 默认值，无覆盖时） (Confirm the env picked up sense_radius=12).
    assert abs(term.cfg.sense_radius - 12.0) < 1e-9, (
        f"Expected sense_radius=12.0 under OMNI_THROW=1, got {term.cfg.sense_radius}"
    )

    threat = _place_omni_ball_and_run_cbf(env)
    frac = threat.float().mean().item()

    # 在 8-10m 发射距离与 sense_radius=12 条件下，所有环境的距离均严格小于 12m -> 均判定为受威胁
    # （除了极个别相对法向速度 nu<=0 的边缘情况，例如首次前向时球体相对法向速度尚未完全收敛；
    # 实际上弹道求解始终产生逼近轨迹 -> 接近 100% 触发）。此处要求 >=80% 以留出稳健的安全边界 (At 8-10 m launch distance and sense_radius=12, require >=80% threatened).
    assert frac >= 0.80, (
        f"Expected >=80% of envs threatened at launch with sense_radius=12, got {frac:.1%}. "
        f"This means the 12 m sense_radius is NOT engaging at the 8-10 m omni launch range."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_sense_radius_4_does_not_fire_at_omni_launch():
    """测试场景：对照实验——在旧版感知半径 (sense_radius=4m) 下，CBF 在发射瞬间处于盲区。

    预期输出及验证逻辑：
        1. 强制设定 CBF_SENSE_RADIUS=4.0；
        2. 小球在 8-10m 远端发射时，超出 4m 感知边界，受威胁环境占比 <= 5%（完全不触发），
           证明旧版配置在全向高速抛球下存在严重的感知盲区与反应滞后。
    """
    env = _build_state_cbf_env(num_envs=64, sense_radius_override="4.0")
    term = env.action_manager.get_term("joint_pos")

    assert abs(term.cfg.sense_radius - 4.0) < 1e-9, (
        f"Expected sense_radius=4.0 via CBF_SENSE_RADIUS=4, got {term.cfg.sense_radius}"
    )

    threat = _place_omni_ball_and_run_cbf(env)
    frac = threat.float().mean().item()

    # 球体距离机器人各刚体点 8-10m；当 sense_radius=4 时，对于所有环境均满足 dist > sense_radius（位于感知区外）。
    # 允许 <=5% 的微量数值边缘误差，验证旧版小感知半径在此全向距离下不会误激活 (Ball is 8-10 m, sense_radius=4 -> dist > sense_radius, allow <=5%).
    assert frac <= 0.05, (
        f"Expected <5% of envs threatened at launch with sense_radius=4 (ball at 8-10m), "
        f"got {frac:.1%}.  The sense_radius=4 gate should not engage at the omni launch range."
    )
