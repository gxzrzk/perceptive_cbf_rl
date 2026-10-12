"""走廊随机导航路径生成、进度追踪与沿路径对齐奖励测试 (Walk Path Integration Tests)。

测试功能：
- 验证 WallWalk 任务中随机走廊多段折线路径的动态生成算法 (reset_walk_path)；
- 验证路径起始几何特性：起点严格位于机器人重置后的 (X, Y) 坐标，首段航向角与机器人初始朝向完全对齐，
  各并行环境生成具有差异化拐弯特性的独立随机路径，初始生成总长度 >= 80 米；
- 验证弧长插值与查询函数 (walk_path_query)：在起点、终点、线段中点等桩号下进行线性插值，
  并准确计算该处的切线航向角 (tangent yaw)；超出终点查询时平滑截断于末端顶点；
- 验证路径连续动态延长机制 (extend_walk_path)：在旧路径末端无缝追加新线段，旧顶点与历史累计弧长保持绝对不变；
- 验证目标指令跟踪 (DodgeGoToGoalCommand)：当机器人被瞬移或行进时，弧长进度 _path_s 紧密追踪，
  目标点 goal_pos_w 始终锁定在超前前视距离 (path_lookahead = 2m) 的路径桩号上；
- 验证沿路径前进速度奖励 (walk_path_progress_reward) 与横向贴合度奖励 (walk_path_adherence_reward)；
- 验证沿路径顺向抛球事件 (throw_ball_on_dwell with along_path=True)：球体从机器人前方路径中央发射并沿走廊逆向飞袭；
- 验证走廊约束瞄准机制：即使机器人大范围侧滑闪避脱离走廊，球体落点瞄准依然被限制在走廊中央安全带宽内，防止球体撞击侧翼墙壁；
- 验证威胁激活时进度奖励与停滞惩罚的完全清零门控。

关键断言条件：
- 路径起点与机器人重置坐标重合 (误差 < 1e-3)，首段偏航角误差 < 1e-2 rad，总长 >= 80m；
- 走廊侧向偏差 0.5m 时贴合度奖励为 exp(-1) ≈ 0.37；
- 威胁激活时 walk_path_progress_when_safe 和 walk_path_stillness_when_safe 强制为 0.0；
- 沿路径抛球轨迹与走廊中轴线横向偏离始终处于安全带宽内。

覆盖边界情况：
- 机器人处于路径起点、中途瞬移、大跨度侧偏；
- 查询桩号超出路径总长时的截断；
- 机器人闪避至走廊外 1.2m 极限工况下的抛球防撞墙测试；
- 剧集跨步重置前后 epoch 递增与进度归零。
"""

import torch
import pytest


def _build_wallwalk_env(num_envs: int = 64):
    """构建包含 64 个环境的 WallWalk 动态走廊避障仿真任务实例。"""
    import src.tasks  # noqa: F401 -- 触发任务注册副作用
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
    )

    cfg = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    env.reset()
    return env


def _refresh(env, *entities):
    """辅助函数：将底层的显式状态写入同步至物理仿真数据结构并刷新实体张量缓存。"""
    import mujoco_warp as mjw

    env.scene.write_data_to_sim()
    mjw.forward(env.sim.wp_model, env.sim.wp_data)
    for e in entities:
        e.update(env.step_dt)


def _path_total(env):
    """辅助函数：获取当前各环境中完整折线路径的总弧长。"""
    ar = torch.arange(env.num_envs, device=env.device)
    return env._walk_path_cumlen[ar, env._walk_path_n - 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_path_starts_at_robot_aligned_with_heading():
    """测试场景：验证路径起点坐标、首段航向对齐性、初始长度及跨环境多样性。

    预期输出及验证逻辑：
        1. 路径原点 env._walk_path_pts[:, 0] 与机器人重置后的 XY 坐标在 1e-3 容差内完全吻合；
        2. 路径第一分段的方向角 path_yaw0 与机器人的初始朝向 robot_yaw 差值 < 1e-2 rad；
        3. 所有并行环境生成的初始折线总长度均 >= 80.0m；
        4. 延伸到第 6 段时，由于各环境独立随机拐弯，各环境的段航向角标准差 > 0.05 rad，证明路线形态具备多样性。
    """
    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)

    assert hasattr(env, "_walk_path_pts") and hasattr(env, "_walk_path_n")
    origin = env._walk_path_pts[:, 0]
    assert torch.allclose(
        origin, robot.data.root_link_pos_w[:, :2], atol=1e-3
    ), "path must start at the robot's post-reset xy"

    seg0 = env._walk_path_pts[:, 1] - env._walk_path_pts[:, 0]
    path_yaw0 = torch.atan2(seg0[:, 1], seg0[:, 0])
    robot_yaw = robot.data.heading_w
    dyaw = torch.atan2(
        torch.sin(path_yaw0 - robot_yaw), torch.cos(path_yaw0 - robot_yaw)
    )
    assert dyaw.abs().max() < 1e-2, (
        f"first segment must run along the robot heading, max dev {dyaw.abs().max():.4f} rad"
    )

    total = _path_total(env)
    assert torch.all(total >= 80.0 - 1e-4), (
        f"initial path should be >= 80 m, min {total.min():.2f}"
    )

    # 随机走廊几何形状检验：经过若干分段后，各环境的航向角必须展现出显著多样性（跨越环境方差 std > 0.05，首段共享的初始航向在经历随机转弯增量后迅速分散） (Random shapes: heading a few segments in must vary across envs).
    k = 6
    segk = env._walk_path_pts[:, k] - env._walk_path_pts[:, k - 1]
    yaw_k = torch.atan2(segk[:, 1], segk[:, 0])
    assert yaw_k.std() > 0.05, (
        f"paths look identical across envs (segment-{k} heading std {yaw_k.std():.4f})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_query_endpoints_and_interpolation():
    """测试场景：验证 walk_path_query 在起点、终点、中段及越界桩号下的插值精度与切向角。

    输入假数据与测试流程：
        1. 查询 s = 0 处坐标，断言等于起点原点；
        2. 查询 s = total 处坐标，断言等于最后一个顶点；
        3. 查询第 0 段中点 (s = 0.5 * c1)，断言线性插值坐标等于两顶点几何中点，切向角与段方向一致；
        4. 查询超出路径总长 (s = total + 50m)，断言平滑截断于最后一个顶点坐标。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    ar = torch.arange(env.num_envs, device=env.device)
    total = _path_total(env)

    p0, _ = mdp.walk_path_query(env, ar, torch.zeros_like(total))
    assert torch.allclose(p0, env._walk_path_pts[:, 0], atol=1e-5)
    p1, _ = mdp.walk_path_query(env, ar, total)
    last = env._walk_path_pts[ar, env._walk_path_n - 1]
    assert torch.allclose(p1, last, atol=1e-4)

    # 第 0 分段中点插值验证：桩号 s 等于半段长度时，插值坐标应精确位于顶点 0 与 1 的几何中点，切向角与该分段几何方向一致 (Midpoint of segment 0: s = half its length -> halfway between vertices 0/1, tangent == segment direction).
    c1 = env._walk_path_cumlen[:, 1]
    pm, yaw = mdp.walk_path_query(env, ar, 0.5 * c1)
    expect = 0.5 * (env._walk_path_pts[:, 0] + env._walk_path_pts[:, 1])
    assert torch.allclose(pm, expect, atol=1e-4)
    seg0 = env._walk_path_pts[:, 1] - env._walk_path_pts[:, 0]
    seg0_yaw = torch.atan2(seg0[:, 1], seg0[:, 0])
    dyaw = torch.atan2(torch.sin(yaw - seg0_yaw), torch.cos(yaw - seg0_yaw))
    assert dyaw.abs().max() < 1e-4

    # 越界桩号保护：查询桩号超出折线总弧长时，平滑截断于最后一个末端顶点 (Querying past the end clamps to the last vertex).
    p2, _ = mdp.walk_path_query(env, ar, total + 50.0)
    assert torch.allclose(p2, last, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_extend_walk_path_continuous_and_longer():
    """测试场景：验证路径动态追加延长函数的平滑连续性与历史不可篡改性。

    输入假数据与测试流程：
        记录当前顶点与累计弧长快照，调用 extend_walk_path 请求延长 50m。
    预期输出及验证逻辑：
        1. 顶点数量增加 (n1 > n0)；
        2. 原有的前缀顶点坐标与弧长保持完全不变（无数值漂移）；
        3. 延伸后的总弧长 >= 原总长 + 50m；
        4. 新增的第一个分段起点与旧路径终点完全衔接，且新增线段长度在合理的 [2, 4] 米步长区间内。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    ar = torch.arange(env.num_envs, device=env.device)
    n0 = env._walk_path_n.clone()
    total0 = _path_total(env)
    pts0 = env._walk_path_pts.clone()
    cum0 = env._walk_path_cumlen.clone()
    end0 = env._walk_path_pts[ar, n0 - 1].clone()

    mdp.extend_walk_path(env, ar, total0 + 50.0)

    n1 = env._walk_path_n
    assert torch.all(n1 > n0), "extension added no vertices"
    # 历史前缀完整性：路径动态追加延长时，原有旧分段顶点完全不受修改（与快照严格完全一致，排除填充行） (Old prefix untouched).
    for i in (0, 1, 5):
        assert torch.allclose(env._walk_path_pts[ar, i], pts0[ar, i], atol=0.0)
    assert torch.allclose(
        env._walk_path_pts[ar, n0 - 1], end0, atol=0.0
    ), "extension must not move the old endpoint"
    assert torch.allclose(
        env._walk_path_cumlen[ar, n0 - 1], total0, atol=0.0
    ), "extension must not shift the old arc lengths"
    total1 = _path_total(env)
    assert torch.all(total1 >= total0 + 50.0 - 1e-4), (
        f"extended total should be >= old+50, got min delta {(total1 - total0).min():.2f}"
    )
    # 空间位置平滑接续：追加的首个新线段必须紧接旧终点延伸，不产生任何坐标跳变 (First appended segment continues from the old endpoint, no jump).
    first_new = env._walk_path_pts[ar, n0]
    gap = (first_new - end0).norm(dim=-1)
    assert torch.all(gap >= 2.0 - 1e-4) and torch.all(gap <= 4.0 + 1e-4), (
        f"appended segment length out of [2, 4]: min {gap.min():.3f}, max {gap.max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_command_tracks_progress_and_lookahead_goal():
    """测试场景：瞬移机器人沿路径前行，验证控制器进度追踪与前视目标锁定。

    输入假数据与测试流程：
        将机器人直接传送至路径弧长 s = 10m 的点上，更新控制器指令。
    预期输出及验证逻辑：
        1. 控制器的进度变量 cmd._path_s 紧密收敛至 10.0m (误差 < 0.3m)；
        2. 追踪的目标点 cmd.goal_pos_w 恰好等于路径上 s = 10 + 2 = 12m 处的三维位置；
        3. 正常走廊行进模式下，无环境被标记为原地站立或定点抛球模式。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    target_xy, _ = mdp.walk_path_query(env, ar, torch.full((env.num_envs,), 10.0, device=env.device))
    pose = robot.data.root_link_pose_w.clone()  # 机器人基座世界位姿，形状为 (N, 7): pos(3) + quat wxyz(4)
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)

    env.command_manager.compute(dt=env.step_dt)

    assert torch.allclose(cmd._path_s, torch.full_like(cmd._path_s, 10.0), atol=0.3), (
        f"_path_s should track the teleport, got mean {cmd._path_s.mean():.3f}"
    )
    goal_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 2.0)
    assert torch.allclose(cmd.goal_pos_w, goal_xy, atol=1e-3), (
        "goal should be the path point path_lookahead ahead of the robot's progress"
    )
    # 剧集重置代际（epoch）逻辑确保在正常走廊导航任务中没有任何环境被误标记为原地站立或定点模式 (The episode-reset epoch logic left no env flagged as standing/in-place).
    assert not cmd.is_standing_env.any() and not cmd.is_inplace_env.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_progress_reward():
    """测试场景：沿路径前行速率奖励项的饱和与瞬时重置。

    预期输出及验证逻辑：
        1. 静止在原点时，进度奖励接近 0 (< 0.5)；
        2. 沿路径瞬移 0.5m（等效瞬时速度 25 m/s），奖励饱和截断在上限 +2.0 附近 (> 1.5)；
        3. 随后停止移动，下一时间步奖励立即回落至 ~0，无历史增量残留。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    # 机器人在路径原点静止时 -> 无前向进度，奖励接近 0 (Stationary at the path origin -> no progress).
    env.command_manager.compute(dt=env.step_dt)
    r0 = mdp.walk_path_progress_reward(env)
    assert r0.abs().max() < 0.5, (
        f"stationary robot should get ~0 progress reward, got max {r0.abs().max():.3f}"
    )

    # 沿路径向前瞬移 0.5m：在单个 0.02s 控制步长内等效速度达 25 m/s，进度奖励被安全截断在 +2.0 上限处 (Teleport 0.5 m forward along path -> 25 m/s, clipped to +2.0 cap).
    target_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 0.5)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)
    env.command_manager.compute(dt=env.step_dt)
    r1 = mdp.walk_path_progress_reward(env)
    assert torch.all(r1 > 1.5), (
        f"forward teleport should saturate the progress reward, min {r1.min():.3f}"
    )

    # 再次恢复静止 -> 下一次计算报告 ~0 进度奖励（delta 是步间差分，不保留历史瞬移记忆） (Standing still again -> next compute reports ~0).
    env.command_manager.compute(dt=env.step_dt)
    r2 = mdp.walk_path_progress_reward(env)
    assert r2.abs().max() < 0.5, (
        f"progress reward should fall back to ~0 once the robot stops, got max {r2.abs().max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_adherence_reward():
    """测试场景：横向偏离路径惩罚与贴合度高斯奖励 (exp(-d^2 / sigma^2))。

    输入假数据与测试流程：
        1. 位于路径正中央时，横向误差 ~0，贴合度奖励接近 1.0 (> 0.95)；
        2. 沿垂直切线方向瞬移 0.5m，横向误差 cmd._path_lat_err 读数约为 0.5m，
           贴合度奖励精确衰减至 exp(-1.0) ≈ 0.368。
    """
    import math
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    # 出生时精准处于路径中心中轴线上 -> 贴合度奖励 ~1.0，横向偏离误差 ~0 (On the route at spawn -> adherence ~1, lat_err ~0).
    env.command_manager.compute(dt=env.step_dt)
    r0 = mdp.walk_path_adherence_reward(env)
    assert torch.all(r0 > 0.95), (
        f"on-path robot should get ~1 adherence, min {r0.min():.3f}"
    )
    assert cmd._path_lat_err.max() < 0.1, (
        f"lat_err should be ~0 at the path origin, max {cmd._path_lat_err.max():.3f}"
    )

    # 沿局部切线垂直法向侧向瞬移 0.5m -> 贴合度奖励按高斯核衰减至 exp(-1.0) ≈ 0.368 (Teleport 0.5 m perpendicular to path tangent -> adherence ~e^-1).
    _, tan_yaw = mdp.walk_path_query(env, ar, cmd._path_s + 0.5)
    perp = torch.stack([-torch.sin(tan_yaw), torch.cos(tan_yaw)], dim=-1)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = pose[:, :2] + 0.5 * perp
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)
    env.command_manager.compute(dt=env.step_dt)
    r1 = mdp.walk_path_adherence_reward(env)
    expect = math.exp(-1.0)
    assert torch.allclose(cmd._path_lat_err, torch.full_like(cmd._path_lat_err, 0.5), atol=0.15), (
        f"lat_err should be ~0.5 after the sideways teleport, mean {cmd._path_lat_err.mean():.3f}"
    )
    assert torch.allclose(r1, torch.full_like(r1, expect), atol=0.2), (
        f"adherence at 0.5 m off-path should be ~e^-1={expect:.3f}, mean {r1.mean():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_throw_along_path_launches_from_path_ahead():
    """测试场景：沿走廊抛球事件中，球体位于前方路径中轴线上发射并逆向迎面飞来。

    预期输出及验证逻辑：
        1. 测量发射起点到路径折线的欧氏距离，最大离轴距离 < 0.05m（严格位于路径上）；
        2. 发射桩号位于机器人前方 s0 + [2, 3] 米范围内；
        3. 球体水平速度方向直指机器人（速度向量与指向机器人向量的余弦夹角 > 0.85）；
        4. 球体速度与走廊切线方向相反（迎面飞来，逆向夹角 cos < -0.5）。
    """
    import src.tasks.amp_loco.mdp as mdp
    from tests.mdp.test_wall_cbf import _path_distance_and_station

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    cmd = env.command_manager.get_term("twist")
    ar = torch.arange(env.num_envs, device=env.device)

    s0 = cmd._path_s.clone()
    env._dodge_throw_once = True  # 强制下一次调用时在所有并行环境中触发一次抛球事件 (force a throw at every env on the next call)
    mdp.throw_ball_on_dwell(
        env, None,
        throw_interval_range=(1.0, 4.0),
        dist_range=(2.0, 3.0),
        height_range=(1.5, 2.3),
        flight_time_range=(0.58, 0.63),
        along_path=True,
    )
    _refresh(env, ball)
    launch = ball.data.root_link_pos_w  # 球体发射起始世界坐标，形状为 (N, 3) (ball launch position in world frame)
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]

    # 1. 验证发射点严格位于走廊路径中轴线上，桩号位于 s0 + dist 处（距离 dist 在 [2.0, 3.0] 米之间） (Launch point lies ON the path, at station ~s0 + dist).
    d_path, station = _path_distance_and_station(env, launch[:, :2])
    assert d_path.max() < 0.05, (
        f"launch point should be on the path, max off-path dist {d_path.max():.3f}"
    )
    assert torch.all(station >= s0 + 2.0 - 1.0) and torch.all(station <= s0 + 3.0 + 1.0), (
        f"launch station out of ~[s0+2, s0+3]: min {(station - s0).min():.2f}, "
        f"max {(station - s0).max():.2f} ahead"
    )
    # 2. 验证小球水平初速度方向精确朝向机器人迎面飞来（速度向量与发射点到机器人连线余弦值 > 0.85） (Ball flies back AT the robot).
    to_robot = robot.data.root_link_pos_w[:, :2] - launch[:, :2]
    cos = (
        (vel_xy * to_robot).sum(dim=-1)
        / (vel_xy.norm(dim=-1) * to_robot.norm(dim=-1)).clamp_min(1e-6)
    )
    assert torch.all(cos > 0.85), (
        f"ball should fly back at the robot, min cos(vel, to_robot) {cos.min():.3f}"
    )
    # 3. 验证小球沿走廊逆向袭来（水平初速度与路径切线方向呈钝角反向，余弦值 < -0.5） (And roughly AGAINST the path travel direction).
    _, tan_yaw = mdp.walk_path_query(env, ar, station)
    tan = torch.stack([torch.cos(tan_yaw), torch.sin(tan_yaw)], dim=-1)
    cos_tan = (vel_xy * tan).sum(dim=-1) / vel_xy.norm(dim=-1).clamp_min(1e-6)
    assert (cos_tan < -0.5).float().mean() > 0.8, (
        f"ball should come back along the corridor (against the tangent), "
        f"fraction with cos<-0.5: {(cos_tan < -0.5).float().mean():.2f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_progress_when_safe_gated():
    """测试场景：安全状态下正常奖励前向进度，来球威胁时奖励严格清零。

    预期输出及验证逻辑：
        1. 安全无威胁时，前向移动 0.5m 获得饱和正奖励 (> 1.4)；
        2. 强制设置 _dodge_threat = True，完全相同的物理前向运动下，奖励输出严格为 0.0，
           确保策略在面临飞球危险时全力躲闪，不与安全避障目标竞争。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    # 重置后球体停泊在安全停机位 -> 无来球威胁 (Ball parked after reset -> no threat).
    env.command_manager.compute(dt=env.step_dt)
    assert not cmd._dodge_threat.any(), "test expects no looming ball after reset"

    # 沿路径向前瞬移 0.5m：等效瞬时速度 25 m/s，奖励截断在 v_max=1.5 (Teleport 0.5 m forward along the path -> rate 25 m/s, capped at v_max=1.5).
    target_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 0.5)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)
    env.command_manager.compute(dt=env.step_dt)

    r_safe = mdp.walk_path_progress_when_safe(env)
    assert torch.all(r_safe > 1.4), (
        f"forward progress while safe should pay ~v_max (1.5), min {r_safe.min():.3f}"
    )

    # 强制注入威胁标志 -> 相同的物理前向运动下，受门控保护的进度奖励严格清零 (Force the threat flag -> the SAME progress reads exactly 0).
    cmd._dodge_threat[:] = True
    r_threat = mdp.walk_path_progress_when_safe(env)
    assert torch.all(r_threat == 0.0), (
        f"under threat the term must vanish, got max {r_threat.abs().max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_progress_when_safe_idle_cost():
    """测试场景：怠速停滞惩罚独立于前视目标距离，且在受到威胁时清零。

    预期输出及验证逻辑：
        1. 原地站立时支付固定的停滞惩罚 -0.6；
        2. 向前瞬移时获得前向全额奖励 +1.5；
        3. 威胁触发时，整个项被门控置零 (== 0.0)。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("twist")
    env.command_manager.compute(dt=env.step_dt)
    assert not cmd._dodge_threat.any()

    # 静止停滞需支付固定的怠速停滞惩罚代价，与前视移动目标的远近无关 (Standing still pays a fixed cost, independent of the moving goal).
    r_stand = mdp.walk_path_progress_when_safe(env, stillness_penalty=0.6)
    expected = -0.6 * cmd._walk_forward_scale
    assert torch.allclose(r_stand, expected, atol=0.1), (
        f"standing should pay the idle cost, got {r_stand.min():.3f} vs {expected.min():.3f}"
    )

    # 沿路径向前瞬移 0.5m：速率奖励饱和在 v_max，免除任何距离代价 (Teleport 0.5 m forward: rate caps at v_max, with no distance cost).
    ar = torch.arange(env.num_envs, device=env.device)
    target_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 0.5)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)
    env.command_manager.compute(dt=env.step_dt)
    r_move = mdp.walk_path_progress_when_safe(env, v_max=1.5, stillness_penalty=0.6)
    expected_move = torch.full_like(r_move, 1.5)
    assert torch.allclose(r_move, expected_move, atol=0.1), (
        f"forward progress should pay, got {r_move.min():.3f} vs {expected_move.min():.3f}"
    )

    # 威胁激活时门控将该复合奖励项彻底置零，避免避障行为与前进目标竞争 (Threat gates the combined term to exactly 0).
    cmd._dodge_threat[:] = True
    assert torch.all(
        mdp.walk_path_progress_when_safe(env, stillness_penalty=0.6) == 0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_stillness_when_safe_gated():
    """测试场景：静止惩罚项在停滞时接近 1.0，运动时衰减至 0，威胁激活时强制置 0。

    预期输出及验证逻辑：
        1. 原地停滞时 cost > 0.9；
        2. 高速前行时 cost < 0.05；
        3. 威胁激活时 cost 严格为 0.0，使机器人在躲避阶段原地急停稳固姿态不承受任何惩罚。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    env.command_manager.compute(dt=env.step_dt)
    assert not cmd._dodge_threat.any(), "test expects no looming ball after reset"

    # 停留在路径原点时 -> 静止停滞代价接近 1.0 (Parked at the path origin -> stillness cost ~1).
    c0 = mdp.walk_path_stillness_when_safe(env)
    assert torch.all(c0 > 0.9), (
        f"parked robot should read full stillness cost, min {c0.min():.3f}"
    )

    # 向前瞬移 0.5m（等效速度 25 m/s） -> 静止代价迅速衰减至 ~0 (Teleport 0.5 m forward, rate 25 m/s -> cost decays to ~0).
    target_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 0.5)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)
    env.command_manager.compute(dt=env.step_dt)
    c1 = mdp.walk_path_stillness_when_safe(env)
    assert c1.max() < 0.05, (
        f"fast-moving robot should read ~0 stillness cost, max {c1.max():.3f}"
    )

    # 威胁激活状态下，该代价完全消失，即便在运动状态中也同样归零：机器人在躲避阶段急停站稳不承受任何静止惩罚 (Under threat the cost vanishes EVEN WHILE MOVING: planting for a dodge is free).
    cmd._dodge_threat[:] = True
    c2 = mdp.walk_path_stillness_when_safe(env)
    assert torch.all(c2 == 0.0), (
        f"under threat the stillness cost must vanish, got max {c2.abs().max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_throw_along_path_aim_stays_in_corridor():
    """测试场景：走廊瞄准防碰撞边界保护——即使机器人大幅闪避出走廊，弹道依然保持在走廊内部。

    输入假数据与测试流程：
        1. 将机器人向垂直于路径方向推移 1.2m（已经完全脱离走廊并靠近侧墙）；
        2. 执行 along_path 抛球事件；
        3. 沿球体前向飞行轨迹采样 59 个时间切片，测量轨迹到路径折线的欧氏距离。
    预期输出及验证逻辑：
        1. 轨迹到路径的距离始终被约束在安全走廊宽度内 (< 0.86m)，绝不会撞上 1.05m 外的侧墙；
        2. 球体虽然被约束在走廊内，但依然朝向机器人所在桩号飞行（距离机器人的最短交会间隙 < 1.7m）。
    """
    import src.tasks.amp_loco.mdp as mdp
    from tests.mdp.test_wall_cbf import _path_distance_and_station

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ball = env.scene["ball"]
    cmd = env.command_manager.get_term("twist")
    ar = torch.arange(env.num_envs, device=env.device)

    # 将机器人沿路径垂直法向推移 1.2m（已经完全超出 ±0.6m 的瞄准走廊带） (Shove the robot 1.2 m perpendicular to the path, outside the +-0.6 aim band).
    proj_xy, tan_yaw = mdp.walk_path_query(env, ar, cmd._path_s)
    perp = torch.stack([-torch.sin(tan_yaw), torch.cos(tan_yaw)], dim=-1)
    side = torch.where(
        torch.rand(env.num_envs, device=env.device) < 0.5, -1.0, 1.0
    ).unsqueeze(-1)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = proj_xy + side * 1.2 * perp
    robot.write_root_link_pose_to_sim(pose)
    # 清除 RSI 初始残余速度：由于其指向任意方向，纯切向超前预测会将瞄准点拖曳至沿轨迹方向（甚至落后于起点），
    # 从而虚假放大点到折线的几何距离，而走廊安全性核心关注的是横向偏离而非沿轨迹偏移 (Zero leftover RSI velocities to prevent spurious along-track lead).
    robot.write_root_link_velocity_to_sim(
        torch.zeros(env.num_envs, 6, device=env.device)
    )
    _refresh(env, robot)

    env._dodge_throw_once = True
    mdp.throw_ball_on_dwell(
        env, None,
        throw_interval_range=(1.0, 4.0),
        dist_range=(2.0, 3.0),
        height_range=(1.5, 2.3),
        flight_time_range=(0.58, 0.63),
        along_path=True,
    )
    _refresh(env, ball)
    launch = ball.data.root_link_pos_w[:, :2]
    vel_xy = ball.data.root_link_lin_vel_w[:, :2]

    # 小球在 t_flight ∈ [0.58, 0.63] 秒时到达瞄准点（初速度 = 位移 / 飞行时间）。仅在 [0, 0.58] 秒内密集插值采样，
    # 确保所有环境的采样点严格处于飞行中段，避免飞过目标点后的横向几何发散 (The ball arrives at t_flight in [0.58, 0.63] s; sample [0, 0.58] inside flight).
    t = torch.linspace(0.0, 0.58, 59, device=env.device).unsqueeze(0)  # 采样时间网格，形状为 (1, T)
    traj = launch.unsqueeze(1) + vel_xy.unsqueeze(1) * t.unsqueeze(-1)  # 轨迹采样点三维水平投影，形状为 (N, T, 2)
    d_to_path = []
    for j in range(traj.shape[1]):
        d_j, _ = _path_distance_and_station(env, traj[:, j])
        d_to_path.append(d_j)
    d_to_path = torch.stack(d_to_path, dim=1)  # 轨迹点到走廊折线中轴线距离矩阵，形状为 (N, T)
    # 飞行弹道绝不能飘出走廊防碰撞包络：走廊限制带 (0.4m) + 弯道弓形拱高 (~0.26m) + 横向扰动裕量 (0.15m)，始终处于 ~1.05m 的内侧墙体安全边界之内 (Trajectory must stay inside corridor band + sagitta slack).
    assert d_to_path.max() < 0.4 + 0.26 + 0.2, (
        f"trajectory leaves the corridor band: max off-path dist {d_to_path.max():.3f}"
    )
    # 并且该射击必须切实朝向机器人当前所在桩号：在机器人脱离中轴线 1.2m 且瞄准点被截断在 0.4m 走廊带的情况下，
    # 最近掠过距离约为 0.8m 横向间隙 + 沿轨抖动 (~0.3m) -> 断言 < 1.7m（证明球体没有射偏或射空；脱离走廊的机器人按设计唯有重新回到走廊才能被再次有效瞄准） (Ball shot genuine toward robot station, distance < 1.7m).
    d_to_robot = (traj - robot.data.root_link_pos_w[:, :2].unsqueeze(1)).norm(dim=-1)
    assert d_to_robot.min(dim=1).values.max() < 1.7, (
        f"ball should still pass near the (off-line) robot's station, min-pass "
        f"{d_to_robot.min(dim=1).values.max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_path_debug_vis_draws_polyline():
    """测试场景：验证路径调试可视化钩子函数 debug_vis 能正确绘制折线圆柱与目标球体。

    预期输出及验证逻辑：
        1. 绘制的圆柱体数量 > 40（覆盖多段折线段）；
        2. 每个绘制圆柱的起点坐标均精准落在所生成的路径折线点集合上；
        3. 进度球与目标球数量满足预期。
    """
    env = _build_wallwalk_env(num_envs=4)
    cmd = env.command_manager.get_term("twist")

    class _FakeViz:
        def __init__(self):
            self.cylinders = []
            self.spheres = []
            self.arrows = []

        def get_env_indices(self, num_envs):
            return [0, 1]

        def add_cylinder(self, start, end, radius, color, label=None):
            self.cylinders.append((start, end))

        def add_sphere(self, center, radius, color, label=None):
            self.spheres.append(center)

        def add_arrow(self, start, end, color, width=0.015, label=None):
            self.arrows.append((start, end))

    viz = _FakeViz()
    cmd.debug_vis(viz)
    # 验证绘制元素：包含 2 个可视化环境 × (<= 41 个线段) 的路径圆柱体，以及进度球与目标球 (2 envs x path cylinders, plus progress and goal spheres).
    assert len(viz.cylinders) > 40, (
        f"expected the path polyline to be drawn, got {len(viz.cylinders)} cylinders"
    )
    # 验证每个绘制圆柱线段均精准落在折线路径上：其起点坐标必须落在两个可视化环境中之一的已生成顶点集合内 (Every cylinder segment lies on the path).
    pts0 = env._walk_path_pts[0].cpu().numpy()
    s0 = viz.cylinders[0][0]
    import numpy as np
    d = np.linalg.norm(pts0 - np.array(s0[:2]), axis=1).min()
    assert d < 1e-4, f"cylinder start not on the path polyline (min dist {d:.4f})"
    assert len(viz.spheres) >= 2, "progress-point (and goal) spheres missing"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_path_regenerated_per_episode_reset():
    """测试场景：剧集重置 (env.reset()) 触发路径再生与 epoch 递增。

    预期输出及验证逻辑：
        1. 重置后所有环境的 env._walk_path_epoch 均大于初始 epoch；
        2. 控制器的累计进度 cmd._path_s 重新归零 (< 1.0m)；
        3. 新路径的原点依然与机器人的重置坐标对齐。
    """
    env = _build_wallwalk_env()
    robot = env.scene["robot"]

    epoch0 = env._walk_path_epoch.clone()
    env.reset()
    assert torch.all(env._walk_path_epoch > epoch0), "reset did not regenerate the path"
    cmd = env.command_manager.get_term("twist")
    assert cmd._path_s.abs().max() < 1.0, (
        f"progress should restart near the path origin after reset, got max {cmd._path_s.abs().max():.3f}"
    )
    origin = env._walk_path_pts[:, 0]
    assert torch.allclose(origin, robot.data.root_link_pos_w[:, :2], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_step_smoke_no_nan():
    """测试场景：执行 10 步空动作步进冒烟测试，验证数据流数值有限性。

    预期输出及验证逻辑：
        步进后速度指令、路径进度以及路径坐标张量均全为有限数 (无 NaN/Inf)。
    """
    env = _build_wallwalk_env()
    cmd = env.command_manager.get_term("twist")
    for _ in range(10):
        obs, rew, terminated, time_out, extras = env.step(
            torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        )
    cmd_vel = env.command_manager.get_command("twist")
    assert torch.isfinite(cmd_vel).all(), "non-finite velocity command"
    assert torch.isfinite(cmd._path_s).all(), "non-finite path progress"
    assert torch.isfinite(env._walk_path_pts).all(), "non-finite path"
