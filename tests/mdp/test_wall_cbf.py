"""控制障碍函数 (CBF) 静态墙体安全边界、距离惩罚与走廊循环摆放测试 (Wall CBF Integration Tests)。

测试功能：
- 验证针对静态墙体障碍物的控制障碍函数 (CBF) 状态与安全约束；
- 验证墙体相对机器人初始朝向的自适应侧向随机摆放算法 (reset_wall_pose)；
- 验证被碰撞击飞后的运动学复位锁止机制 (pin_wall)；
- 验证 CBF 障碍物安全距离与逼近惩罚函数 ``wall_link_cbf_reward`` 的屏障条件：
  1. 机器人静止且距离墙体足够远时，屏障条件成立，CBF 惩罚接近 0；
  2. 机器人高速向墙体靠近时，由于距离变化率 h_dot < 0 导致约束违反，CBF 产生显著负奖励惩罚；
- 验证防误触机制：在有墙任务中关闭基于球速不连续的击中兜底判定 (delta_v_fallback)，防止球体砸墙反弹被误判为击中机器人；
- 验证走廊多墙体任务 (WallWalk) 中，多堵墙沿路径随桩号散布放置；
- 验证越过墙体的动态回收推进算法 (recycle_walls_ahead)：当机器人走过某堵墙后，该墙体自动传送到前方 7-10m 的新桩号，维持连续避障环境；
- 验证距离排序观测函数 (dodge_wall_state_b)：返回按中心欧氏距离严格升序排列的 k 个最近墙体。

关键断言条件：
- 墙体相对初始朝向的侧向间隙在 [0.6, 1.0] 米之间，且双侧均有分布；
- 静止安全状态下 CBF 惩罚 min > -0.05；逼近墙体时 > 90% 的环境触发强烈负奖励惩罚 (r < -0.1)；
- 机器人后方 > 1m 的墙体成功回收并重新布置在前方 7-10m；
- 墙体观测项向量严格按照近到远的顺序排列。

覆盖边界情况：
- 机器人在世界系具有不同随机航向（-90° 附近）；
- 机器人全向各连杆向侧墙以 1 m/s 整体冲刺；
- 墙体被外力移动后的运动学复位与速度归零；
- 沿曲折随机路径的投影距离与弧长桩号计算。
"""

import torch
import pytest


def _build_wall_env(num_envs: int = 64):
    """构建包含单面侧墙的 MimicKit-Wall 仿真任务环境。"""
    import src.tasks  # noqa: F401 -- 触发任务注册副作用
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_mimickit_wall_flat_env_cfg,
    )

    cfg = g1_amp_dodge_mimickit_wall_flat_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    env.reset()
    return env


def _build_wallwalk_env(num_envs: int = 64):
    """构建包含 3 堵连续路径墙体的 MimicKit-WallWalk 走廊避障仿真任务环境。"""
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
    """辅助函数：将显式写入的状态同步至物理仿真数据结构并刷新实体张量缓存。"""
    import mujoco_warp as mjw

    env.scene.write_data_to_sim()
    mjw.forward(env.sim.wp_model, env.sim.wp_data)
    for e in entities:
        e.update(env.step_dt)


def _heading_frame_from_yaw(yaw: torch.Tensor):
    """根据偏航角计算前进方向前向向量 fwd 与垂直向左侧向向量 perp。"""
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    perp = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
    return fwd, perp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_placement_randomized_beside_spawn_heading_relative():
    """测试场景：验证墙体相对机器人初始朝向的侧方随机摆放（非世界坐标系绝对固定）。

    预期输出及验证逻辑：
        1. 检查各环境暂存的墙体坐标 env._wall_pos_w；
        2. 将墙体与机器人的相对位移投影到机器人航向局部系：
           - 侧向距离绝对值 |lat| 严格位于 [0.6, 1.0] 米之间；
           - 纵向距离绝对值 |along| 位于 [-0.5, 0.5] 米之间；
        3. 双侧双峰分布：在 64 个环境中，机器人左侧 (lat > 0) 与右侧 (lat < 0) 均有墙体分布；
        4. 墙体高度中心恰好在地面上方半高处（Z = 1.0m）。
    """
    env = _build_wall_env()
    wall = env.scene["wall"]
    robot = env.scene["robot"]

    assert hasattr(env, "_wall_pos_w"), "reset_wall_pose did not stash the wall pose"
    assert env._wall_pos_w.shape == (env.num_envs, 1, 3)
    # 墙体航向角等于机器人重置时的朝向：将其作为横向与沿向摆放的基准局部坐标系 (Wall yaw == the robot's reset heading: use it as the placement frame).
    fwd, perp = _heading_frame_from_yaw(env._wall_yaw_w[:, 0])
    off = env._wall_pos_w[:, 0, :2] - robot.data.root_link_pos_w[:, :2]  # 墙体相对机器人的水平偏移向量，形状为 (N, 2)

    lat = (off * perp).sum(dim=-1)
    along = (off * fwd).sum(dim=-1)
    assert torch.all(lat.abs() >= 0.6 - 1e-5) and torch.all(lat.abs() <= 1.0 + 1e-5), (
        f"lateral distance out of [0.6, 1.0]: min {lat.abs().min():.3f}, max {lat.abs().max():.3f}"
    )
    assert torch.all(along.abs() <= 0.5 + 1e-5), (
        f"along-heading offset out of [-0.5, 0.5]: max {along.abs().max():.3f}"
    )
    # 双峰左右双侧采样：在 64 个并行环境中，左右两侧必须同时出现（全在同一侧的概率极低仅为 2^-63） (Bimodal side sampling: with 64 envs both sides must appear).
    assert (lat > 0).any() and (lat < 0).any(), "wall only ever on one side"
    # 墙体几何中心位于半高处（确保墙体底面与地面相切贴合），且实体实际位姿与暂存数据完全匹配 (Centre at half height, wall base on the ground, and entity pose matches the stash).
    assert torch.allclose(
        env._wall_pos_w[:, 0, 2], torch.ones_like(lat), atol=1e-4
    )
    assert torch.allclose(wall.data.root_link_pos_w, env._wall_pos_w[:, 0], atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_pin_wall_restores_pose():
    """测试场景：墙体在外力冲击或被挪开后，pin_wall 事件函数能将其牢固锁回原位。

    输入假数据与测试流程：
        1. 手动将墙体坐标在 Y 轴方向平移推开 2.0m；
        2. 调用 mdp.pin_wall；
        3. 验证墙体坐标精准恢复至初始暂存的位姿，且线速度彻底归零。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    wall = env.scene["wall"]
    n = env.num_envs

    away = env._wall_pos_w[:, 0].clone()
    away[:, 1] += 2.0  # 将墙体坐标沿 Y 轴平移推远 2m 以构造受扰动状态 (shove the wall 2 m further out)
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0
    wall.write_root_link_pose_to_sim(torch.cat([away, quat], dim=-1))
    _refresh(env, wall)
    assert not torch.allclose(wall.data.root_link_pos_w, env._wall_pos_w[:, 0], atol=1e-3)

    mdp.pin_wall(env, None, wall_names=("wall",))
    _refresh(env, wall)
    assert torch.allclose(wall.data.root_link_pos_w, env._wall_pos_w[:, 0], atol=1e-4), (
        "pin_wall did not restore the sampled wall pose"
    )
    assert wall.data.root_link_lin_vel_w.abs().max() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_link_cbf_zero_when_standing_far():
    """测试场景：静止且远离墙体时，控制障碍函数 (CBF) 约束自然满足，惩罚接近 0。

    输入假数据与测试流程：
        将机器人关节速度和根速度全部清零，保持在原点静止（距离墙体 >= 0.6m）。
    预期输出及验证逻辑：
        调用 wall_link_cbf_reward，计算各连杆 barrier 条件 h_dot + alpha * h >= 0；
        惩罚值 r.max() <= 0.0 且 r.min() > -0.05（几乎完全无惩罚）。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    robot = env.scene["robot"]
    n = env.num_envs

    # RSI 重置会写入动作片段的初始速度；此处强制清零以确保机器人处于绝对静止状态 (RSI reset writes the motion clip's velocities; zero them so the robot is truly still).
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device))
    robot.write_joint_state_to_sim(
        robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
    )
    _refresh(env, robot)

    r = mdp.wall_link_cbf_reward(
        env, robot_name="robot", wall_names=("wall",),
        alpha=1.0, margin=0.05, constraint_clip=2.0,
    )
    # 各连杆距离墙面 >= 0.5m 且处于静止状态：法向接近速度 h_dot ~ 0，屏障值 alpha*h > 0，故 CBF 惩罚截断为 0 (Links are >= ~0.5 m from the surface and ~stationary: h_dot ~ 0, alpha*h > 0 -> 0).
    assert r.max() <= 0.0, "CBF reward must be <= 0"
    assert r.min() > -0.05, (
        f"standing far from the wall should be (near) penalty-free, got min {r.min():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_link_cbf_penalizes_approach():
    """测试场景：机器人以高速向墙体冲去时，CBF 约束被破坏，产生显著负奖励惩罚。

    输入假数据与测试流程：
        驱动整个机器人以 1.0 m/s 的线速度直接朝向墙体侧向平移。
    预期输出及验证逻辑：
        由于向墙体逼近，距离导数 h_dot < 0 导致 h_dot + alpha * h < 0，
        超过 90% 的环境触发明显惩罚 (r < -0.1)，起到安全防护引导作用。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    robot = env.scene["robot"]
    n = env.num_envs

    # 清除 RSI 初始速度，随后驱动整个机器人以 1.0 m/s 侧向线速度直接冲向墙体 (Zero the RSI clip velocities, then drive the whole robot laterally toward its wall at 1 m/s).
    robot.write_joint_state_to_sim(
        robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
    )
    robot_xy = robot.data.root_link_pos_w[:, :2]
    direction = torch.sign(env._wall_pos_w[:, 0, :2] - robot_xy)  # 机器人朝向墙体的单位符号方向向量，形状为 (N, 2)
    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 0:2] = direction * 1.0
    robot.write_root_link_velocity_to_sim(vel)
    _refresh(env, robot)

    r = mdp.wall_link_cbf_reward(
        env, robot_name="robot", wall_names=("wall",),
        alpha=1.0, margin=0.05, constraint_clip=2.0,
    )
    # 此时安全间距 h ~ 0.5m，接近率 h_dot ~ -1.0 m/s 冲向墙面 -> 朝向墙面的各连杆满足 h_dot + alpha*h ~ -0.5 < 0，从而触发惩罚 (h_dot + alpha*h < 0 triggers penalty for links facing wall).
    assert (r < -0.1).float().mean() > 0.9, (
        f"approaching the wall should penalize nearly all envs; rewards: "
        f"min {r.min():.3f}, mean {r.mean():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_ball_hit_dv_fallback_disabled():
    """测试场景：验证有墙场景下禁用了球体速度跳变 (delta_v) 兜底判定，防止弹墙误判。

    预期输出及验证逻辑：
        ball_hit 终止项配置中的 delta_v_threshold 必须为 0.0。
    """
    env = _build_wall_env()
    params = env.termination_manager.get_term_cfg("ball_hit").params
    assert params.get("delta_v_threshold", 0.0) == 0.0, (
        f"delta_v_fallback should be off on the wall task, got {params}"
    )


# ---------------------------------------------------------------------------
# WallWalk 走廊避障任务测试：前向指令、多墙体分布、动态循环回收与排序观测
# ---------------------------------------------------------------------------

_WALLWALK_NAMES = ("wall_0", "wall_1", "wall_2")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_forward_command_constant():
    """测试场景：走廊路径追踪在初始对齐状态下输出稳定的前向巡航速度 (1.3 m/s)。

    预期输出及验证逻辑：
        重置后首段路径与朝向完全对齐，纯追踪目标锁定在前方 2m 处；
        前向速度 vx 稳定饱和在最大线速度 1.3 m/s，侧向速度 vy 与角速度 wz 均为 0。
    """
    env = _build_wallwalk_env()
    cmd = env.command_manager.get_command("twist")  # 获取速度跟踪指令，形状为 (N, 3): vx, vy, wz
    assert cmd.shape == (env.num_envs, 3)
    assert torch.allclose(
        cmd[:, 0], torch.full_like(cmd[:, 0], 1.3), atol=1e-4
    ), f"forward command should be clamped to 1.3 m/s, got {cmd[:, 0].unique()}"
    assert cmd[:, 1].abs().max() < 1e-4 and cmd[:, 2].abs().max() < 1e-4, (
        "no lateral / yaw command expected on the heading-aligned first segment "
        "(tiny float residue from the path geometry is fine)"
    )


def _path_distance_and_station(env, points_xy: torch.Tensor):
    """辅助向量化函数：计算平面上一组点到各环境折线路径的最短距离及对应的弧长桩号。"""
    pts = env._walk_path_pts  # 折线路径顶点坐标，形状为 (N, M, 2)
    cum = env._walk_path_cumlen  # 累计弧长桩号，形状为 (N, M)
    count = env._walk_path_n  # 各环境有效顶点数，形状为 (N,)
    p0 = pts[:, :-1]  # 线段起始端点集合，形状为 (N, M-1, 2)
    p1 = pts[:, 1:]
    seg = p1 - p0
    l2 = (seg * seg).sum(dim=-1).clamp_min(1e-9)
    u = (((points_xy.unsqueeze(1) - p0) * seg).sum(dim=-1) / l2).clamp(0.0, 1.0)
    closest = p0 + u.unsqueeze(-1) * seg
    dist = (closest - points_xy.unsqueeze(1)).norm(dim=-1)  # 点到各分段的垂直/最近距离，形状为 (N, M-1)
    station = cum[:, :-1] + u * seg.norm(dim=-1)
    # 忽略超出各环境真实顶点有效数量的尾部填充线段 (Ignore padding segments beyond each env's vertex count).
    valid = (
        torch.arange(pts.shape[1] - 1, device=env.device).unsqueeze(0)
        < (count - 1).unsqueeze(1)
    )
    dist = torch.where(valid, dist, torch.full_like(dist, 1e9))
    j = dist.argmin(dim=1)
    ar = torch.arange(env.num_envs, device=env.device)
    return dist[ar, j], station[ar, j]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_walls_scattered_ahead_on_path():
    """测试场景：WallWalk 任务中 3 堵侧墙沿随机走廊路径的桩号与横向间隙排布。

    预期输出及验证逻辑：
        遍历 3 堵墙体：
        1. 距离路径中轴线的横向距离位于 [1.1, 1.5] 米区间；
        2. 沿路径的弧长桩号随序号递进，第 i 堵墙位于 2.5 + 3i 到 4.0 + 3i 附近。
    """
    env = _build_wallwalk_env()

    assert env._wall_pos_w.shape == (env.num_envs, 3, 3)
    for i in range(3):
        dist, station = _path_distance_and_station(env, env._wall_pos_w[:, i, :2])
        assert torch.all(dist >= 1.1 - 0.15) and torch.all(dist <= 1.5 + 0.05), (
            f"wall_{i} distance-to-path out of ~[1.1, 1.5]: "
            f"min {dist.min():.3f}, max {dist.max():.3f}"
        )
        lo, hi = 2.5 + i * 3.0, 4.0 + i * 3.0
        assert torch.all(station >= lo - 1.2) and torch.all(station <= hi + 1.2), (
            f"wall_{i} path station out of ~[{lo}, {hi}]: "
            f"min {station.min():.3f}, max {station.max():.3f}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_recycle_moves_walked_past_walls_ahead():
    """测试场景：机器人走过后方墙体的动态循环利用（瞬移至前方 7-10m 新桩号）。

    输入假数据与测试流程：
        1. 将 wall_0 挪至机器人后方 3.0m 处，保持 wall_1 与 wall_2 在前方不变；
        2. 执行 recycle_walls_ahead(along_path=True, lateral_range=(1.1, 1.6))；
    预期输出及验证逻辑：
        1. wall_0 的新桩号被成功重置在当前机器人进度前方 7-10m 处；
        2. wall_0 到路径中轴线的横向距离落在 [1.1, 1.6] 米范围内；
        3. 前方的 wall_1 和 wall_2 保持位置完全未动；
        4. 物理实体的世界坐标成功同步刷新。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    fwd, _ = _heading_frame_from_yaw(env._wall_yaw_w[:, 0])
    before = env._wall_pos_w.clone()

    # 将 wall_0 瞬移拖曳至机器人身后 3m 处（保持 wall_1/2 位置不变，仅触发 wall_0 回收） (Drag wall_0 to 3 m BEHIND the robot, leave wall_1/2 untouched).
    env._wall_pos_w[:, 0, :2] = robot.data.root_link_pos_w[:, :2] - 3.0 * fwd
    # 传入任务真实的回收横向间距范围（函数默认值为旧版 0.5-1.2，WallWalk 任务实际注册为 1.1-1.6） (Pass the task's actual recycle lateral range [1.1, 1.6]).
    mdp.recycle_walls_ahead(
        env, None, wall_names=_WALLWALK_NAMES, along_path=True,
        lateral_range=(1.1, 1.6),
    )

    dist, station = _path_distance_and_station(env, env._wall_pos_w[:, 0, :2])
    s_robot = env.command_manager.get_term("twist")._path_s  # 机器人重置后的初始路径桩号（约等于 0） (~0 right after reset)
    assert torch.all(station >= s_robot + 7.0 - 1.2) and torch.all(station <= s_robot + 10.0 + 1.2), (
        f"recycled wall_0 station should be ~7-10 m ahead of s_robot, got "
        f"min {station.min():.3f}, max {station.max():.3f} (s_robot ~ {s_robot.mean():.3f})"
    )
    assert torch.all(dist >= 1.1 - 0.15) and torch.all(dist <= 1.6 + 0.05), (
        f"recycled wall_0 distance-to-path out of ~[1.1, 1.6]: "
        f"min {dist.min():.3f}, max {dist.max():.3f}"
    )
    # wall_1 和 wall_2 仍在前方未被走过 -> 保持原地不动不受影响 (wall_1/2 were still ahead -> untouched).
    assert torch.allclose(env._wall_pos_w[:, 1:], before[:, 1:]), (
        "wall_1/2 should not have been recycled"
    )
    # 实体真实物理位姿已同步瞬移（不仅更新了暂存数据）。回收逻辑直接写入了仿真器 qpos，因此读取实体数据前需调用 _refresh 刷新前向动力学缓存 (Entity pose teleported; recycle writes sim qpos, so refresh before reading).
    w0 = env.scene["wall_0"]
    _refresh(env, w0)
    assert torch.allclose(
        w0.data.root_link_pos_w[:, :2], env._wall_pos_w[:, 0, :2], atol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_obs_k_nearest_sorted():
    """测试场景：多墙体观测函数返回按中心距离严格排序的近邻特征向量。

    预期输出及验证逻辑：
        1. dodge_wall_state_b 输出维度为 (N, 9)（3 个障碍物 * 3 维特征）；
        2. 计算各墙体中心到机器人的实际几何距离并排序，与观测向量中解包出的距离顺序完全相符；
        3. 环境观测管理器中 ball_state 组总维度等于 6 (球) + 9 (墙) = 15 维。
    """
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]

    obs = mdp.dodge_wall_state_b(
        env, robot_name="robot", wall_names=_WALLWALK_NAMES, k=3
    )
    assert obs.shape == (env.num_envs, 9)

    # 重新计算每堵墙到机器人的中心欧氏距离，核验观测特征槽位是否严格按距离升序排列 (Recompute each wall's centre distance and check the obs slots are rank-ordered).
    rp = robot.data.root_link_pos_w[:, :2]
    d = torch.stack(
        [(env.scene[n].data.root_link_pos_w[:, :2] - rp).norm(dim=-1) for n in _WALLWALK_NAMES],
        dim=1,
    )  # 各墙体中心到机器人的真实距离，形状为 (N, 3)
    d_sorted, _ = torch.sort(d, dim=1)
    obs_d = obs.view(env.num_envs, 3, 3)[:, :, :2].norm(dim=-1)  # 观测特征中解包出的水平相对坐标模长 (rel_x, rel_y norms)
    assert torch.allclose(obs_d, d_sorted, atol=1e-4), (
        "obs slots are not the k walls sorted by centre distance"
    )

    # 验证观测管理器中 ball_state 特征组的总维度包含 6 维小球状态 + 9 维墙体状态 = 15 维 (ball_state group carries 6 ball + 9 walls = 15 dims).
    group = env.observation_manager.group_obs_dim["ball_state"]
    assert group == [15] or group == (15,), f"ball_state obs dim should be 15, got {group}"
