"""教室几何布局、无碰撞路径规划与任务注册配置测试 (Classroom Layout Tests)。

测试功能：
- 验证教室中 34 个静态障碍物（15 套桌椅对应 30 个实体，以及 4 堵外围边界墙）的精确空间尺寸与走廊通道净宽；
- 验证主通道、纵向侧走廊以及横向排间过道的通行余量与路径站点对齐；
- 验证教室随机巡航折线路径生成算法 (generate_classroom_route)：在纵横交错的走廊中生成不穿透任何家具的高自由度安全路径；
- 验证路径离散采样点到所有课桌、椅背碰撞几何包络的安全距离均满足阈值；
- 验证注册表中的任务配置完整性、AMP 训练配置以及朝向奖励项参数。

关键断言条件：
- 障碍物数量为 34，且无 mocap 自由度；
- 中央走道宽度约为 2.20m，侧走道宽度约为 1.84m，横向过道宽度约为 1.80m；
- 随机生成的 80 段路径折线总长度 > 200m，各段长度 > 0.79m；
- 路径任意分段上的连续采样点到 30 个家具外壳的最短欧氏距离 >= 安全阈值；
- 教室任务在任务注册表已正确注册，且配置隔离不影响其他任务。

覆盖边界情况：
- 沿 X/Y 轴边界的极限坐标与碰撞半长；
- 课桌与椅背的不同包络几何；
- 多条走廊车道 (lane) 的随机生成与边界检查；
- 目标朝向奖励在不同 Sharpness 陡峭度参数下的非线性响应。
"""
import pytest
from src.assets.objects.classroom import (
    CLASSROOM_COLUMN_Y, CLASSROOM_ROW_PITCH, CLASSROOM_ROUTE_STATIONS,
    CLASSROOM_SPAWN_LANES, CLASSROOM_X_BOUNDS, CLASSROOM_Y_BOUNDS,
)
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)


def test_classroom_clearance_and_wiring():
    """测试场景：教室静态家具排布尺寸、走廊通道通行净空及仿真事件绑定。

    输入假数据：
        解析教室任务配置中的 initialize_static_classroom 事件参数与碰撞模型。
    预期输出及验证逻辑：
        1. 障碍物命名列表长度严格为 34（30个家具几何体 + 4面外围墙体）；
        2. 验证场景中各排各列过道净宽：
           - 中央走道净宽 = COLUMN_Y[2] - COLUMN_Y[1] - 课桌宽(1.2) ≈ 2.1988m；
           - 侧走道净宽 = COLUMN_Y[1] - COLUMN_Y[0] - 课桌宽(1.2) ≈ 1.8411m；
           - 排间横向过道净宽 = 1.7988m；
        3. 教室总平面面积满足 (x_max - x_min) * (y_max - y_min) 预期；
        4. 课桌碰撞半长为 (0.4, 0.6, 0.38)，椅子碰撞半长为 (0.25, 0.25, 0.45)；
        5. 所有静态物体的碰撞体全部位于教室有效边界内部。
    """
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    params = cfg.events['initialize_static_classroom'].params
    poses = params['layout']
    names = params['wall_names']
    assert len(names) == 34
    assert cfg.sim.njmax >= 2048
    assert 'recycle_walls_ahead' not in cfg.events
    assert cfg.commands['twist'].rel_standing_envs == 0
    assert cfg.commands['twist'].rel_inplace_throw_envs == 0
    assert cfg.events['reset_walk_path'].func.__name__ == 'reset_classroom_robot'
    assert cfg.observations['ball_state'].terms['wall_state'].params['k'] == 8
    throw = cfg.events['throw_ball_on_dwell'].params
    assert throw['omnidirectional'] is True
    assert throw['along_path'] is False
    assert throw['launch_speed_range'] is None
    assert len(poses[:30:2]) == 15
    assert {p[1] for p in poses[:30:2]} == set(CLASSROOM_COLUMN_Y)
    # 内侧课桌边缘界定中央主走廊宽度（走廊净宽经桌宽 1.2m 扣除后精确满足要求） (Inner desk edges define the central aisle).
    assert CLASSROOM_COLUMN_Y[2] - CLASSROOM_COLUMN_Y[1] - 1.2 == pytest.approx(2.1988233258)
    # 相邻排课桌间距界定两条纵向侧走廊净宽 (Adjacent columns define the two additional longitudinal aisles).
    assert CLASSROOM_COLUMN_Y[1] - CLASSROOM_COLUMN_Y[0] - 1.2 == pytest.approx(1.8410524494)
    # 每对桌椅与下一排桌椅之间必须预留承诺的横向过道净空 (Every chair/desk pair must leave the promised cross aisle to the next row).
    for row in range(4):
        for col in range(3):
            desk = poses[row * 6 + col * 2]
            next_chair = poses[(row + 1) * 6 + col * 2 + 1]
            assert next_chair[0] - 0.25 - (desk[0] + 0.4) == pytest.approx(1.7988233258)
            # 路径拐点枢纽精确居中于前后排家具包络之间 (Junctions stay centered between the unchanged furniture envelopes).
            assert CLASSROOM_ROUTE_STATIONS[row + 1] == pytest.approx(
                (desk[0] + 0.4 + next_chair[0] - 0.25) / 2)
    x_min, x_max = CLASSROOM_X_BOUNDS
    y_min, y_max = CLASSROOM_Y_BOUNDS
    assert (x_max - x_min) * (y_max - y_min) == pytest.approx(25.1 * 9.6 * 0.8)
    for i, (name, pose) in enumerate(zip(names, poses, strict=True)):
        asset = cfg.scene.entities[name]
        spec = asset.spec_fn()
        assert tuple(spec.body('wall').pos) == pytest.approx(pose[:3])
        half = spec.geom('wall_collision').size
        if i < 30:
            assert tuple(half) == pytest.approx(
                (0.4, 0.6, 0.38) if i % 2 == 0 else (0.25, 0.25, 0.45))
            assert x_min + 0.08 < pose[0] - half[0] < pose[0] + half[0] < x_max - 0.08
            assert y_min + 0.08 < pose[1] - half[1] < pose[1] + half[1] < y_max - 0.08
        elif i < 32:
            assert tuple(half) == pytest.approx(((x_max - x_min) / 2, 0.08, 1.5))
        else:
            assert tuple(half) == pytest.approx((0.08, (y_max - y_min) / 2, 1.5))
        model = spec.compile()
        assert model.geom('wall_collision').contype == 1


def test_random_routes_stay_clear_of_furniture():
    """测试场景：批量生成的教室随机巡航路线对所有家具的避让安全性。

    输入假数据：
        生成 128 条随机路径，初始位置位于 X=12.0m，Y=0.0m，固定随机种子。
    预期输出及验证逻辑：
        1. 路线张量维度为 (128, 80, 2)，即 128 个环境各包含 80 个 2D 路径拐点；
        2. 各环境路径呈现多样性（不完全相同），具有前行和后退分支；
        3. 每段直线长度均 > 0.79m，总路径长度均 > 200m；
        4. 关键几何安全性校验：在路径每两相邻拐点间线性采样 17 个点，
           计算所有采样点到教室 30 个桌椅长方体碰撞包络的带符号欧氏距离，
           断言最短距离全部满足安全间隙阈值 (CLASSROOM_ROW_PITCH - 1.6) / 2。
    """
    import torch
    from src.tasks.amp_loco.mdp.classroom import generate_classroom_route
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    n = 128
    x = torch.full((n,), 12.0)
    y = torch.zeros(n)
    lane = torch.zeros(n, dtype=torch.long)
    torch.manual_seed(21)
    route = generate_classroom_route(x, y, lane)
    assert route.shape == (n, 80, 2)
    assert not torch.equal(route[0], route[1])
    assert (route[:, 1, 0] < x).any() and (route[:, 1, 0] > x).any()
    assert (route[:, :, 1] != 0).any()
    assert torch.allclose(route[:, 0], torch.stack((x, y), -1))
    lengths = (route[:, 1:] - route[:, :-1]).norm(dim=-1)
    assert (lengths > 0.79).all()
    assert (lengths.sum(-1) > 200).all()
    assert ((route[..., 0] >= 0) & (route[..., 0] <= CLASSROOM_ROUTE_STATIONS[-1])).all()
    assert (route[..., 1].abs() <= max(abs(y) for y in CLASSROOM_SPAWN_LANES)).all()
    # 对全路径各线段高密插值采样，严格核验每个采样点至所有课桌与椅子的物理包络安全间隙 (Sample all segments and check clearance to every desk and chair footprint).
    t = torch.linspace(0, 1, 17)
    samples = route[:, :-1, None] * (1 - t[None, None, :, None]) + route[:, 1:, None] * t[None, None, :, None]
    for i, pose in enumerate(cfg.events['initialize_static_classroom'].params['layout'][:30]):
        half = (0.4, 0.6) if i % 2 == 0 else (0.25, 0.25)
        outside = (samples - torch.tensor(pose[:2])).abs() - torch.tensor(half)
        distance = outside.clamp_min(0).norm(dim=-1)
        assert (distance >= (CLASSROOM_ROW_PITCH - 1.6) / 2 - 1e-5).all()


def test_classroom_registered_task_configs():
    """测试场景：验证教室避障任务在 mjlab 全局任务注册表中的注册项与配置参数。

    预期输出及验证逻辑：
        1. 任务名 'Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat' 正确存在于全局注册表中；
        2. Runner 训练类为 AMPOnPolicyRunner；
        3. 训练与演示配置中重置路径事件函数绑定为 reset_classroom_robot，抛球模式为全向抛球；
        4. RL 运行配置中实验名称与动作库文件路径一致。
    """
    import src.tasks  # noqa: F401 -- 注册项目中所有任务配置环境 (register all project tasks via side-effect import)
    from mjlab.tasks.registry import (
        list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls,
    )
    from src.tasks.amp_loco.rl import AMPOnPolicyRunner
    from src.tasks.amp_loco.config.g1.goto_rl_cfg import (
        g1_amp_dodge_mimickit_classroom_ppo_runner_cfg,
    )
    task = 'Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat'
    assert task in list_tasks()
    assert load_runner_cls(task) is AMPOnPolicyRunner
    for play in (False, True):
        cfg = load_env_cfg(task, play=play)
        assert cfg.events['reset_walk_path'].func.__name__ == 'reset_classroom_robot'
        assert cfg.events['throw_ball_on_dwell'].params['omnidirectional'] is True
    runner = load_rl_cfg(task)
    assert runner.experiment_name == 'g1_amp_dodge_mimickit_classroom'
    assert runner.amp_motion_files == g1_amp_dodge_mimickit_classroom_ppo_runner_cfg().amp_motion_files
    runner.experiment_name = 'changed'
    assert load_rl_cfg(task).experiment_name == 'g1_amp_dodge_mimickit_classroom'


def test_goal_heading_reward_front_side_back_and_threat():
    """测试场景：多朝向工况与来袭威胁状态下的目标朝向奖励响应。

    输入假数据：
        7 种典型状态：正前 (0°)、侧向 (90°)、正后 (180°)、反向侧向 (-90°)、
        存在威胁时的正前、存在威胁时的正后、目标点与机器人重合。
    预期输出及验证逻辑：
        1. 正前方对齐获得满分 1.0；
        2. 侧向与背向未转向时获得停止惩罚 -1.0；
        3. 威胁激活时奖励项输出强制为 0.0；
        4. 目标重合时同样输出 0.0；
        5. 空间旋转不变性：同时将机器人与目标沿 Z 轴旋转 90°，输出奖励值保持不变。
    """
    from types import SimpleNamespace
    import math
    import torch
    from src.tasks.amp_loco.mdp.rewards import walk_goal_heading_when_safe
    headings = torch.tensor([0., torch.pi / 2, torch.pi, -torch.pi / 2, 0., math.pi, 0.])
    origins = torch.tensor([[5., -2., 0.8]]).repeat(7, 1)
    goals = origins[:, :2] + torch.tensor([1., 0.])
    goals[-1] = origins[-1, :2]
    command = SimpleNamespace(goal_pos_w=goals,
                              _dodge_threat=torch.tensor([0, 0, 0, 0, 1, 1, 0]),
                              cfg=SimpleNamespace(kp_yaw=1., max_ang_vel_z=.5))
    env = SimpleNamespace(
        scene={'robot': SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=origins, heading_w=headings,
            root_link_ang_vel_w=torch.zeros(7, 3)))},
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    value = walk_goal_heading_when_safe(env)
    stopped_turn_score = -1.
    assert torch.allclose(value, torch.tensor([
        1., stopped_turn_score, stopped_turn_score, stopped_turn_score, 0., 0., 0.
    ]), atol=1e-6)
    # 将机器人朝向与目标点同时旋转 90 度（保持相对夹角不变），奖励值保持完全不变，验证 SO(2) 旋转等变性 (Turning both robot and target by 90 degrees preserves the reward).
    command.goal_pos_w[:6] = origins[:6, :2] + torch.tensor([0., 1.])
    env.scene['robot'].data.heading_w += torch.pi / 2
    assert torch.allclose(walk_goal_heading_when_safe(env), value, atol=1e-6)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    assert cfg.rewards['walk_goal_heading_when_safe'].weight == 2.0


def test_goal_heading_reward_sharpness():
    """测试场景：朝向对齐奖励中的非线性指数锐化参数 (sharpness) 作用机理。

    输入假数据：
        机器人的朝向偏差为 ±20°（在 30° 容忍度以内）。
    预期输出及验证逻辑：
        1. 当 sharpness=4.0 时，奖励值为 cos(20°)^4；
        2. 当 sharpness=1.0 时，奖励值为 cos(20°)；
        3. 超出 30° 阈值（如偏差 180°）时，无论 sharpness 取何值，静止不转向的惩罚始终锁定为 -1.0。
    """
    from types import SimpleNamespace
    import math
    import torch
    from src.tasks.amp_loco.mdp.rewards import walk_goal_heading_when_safe
    headings = torch.tensor([math.pi / 9, -math.pi / 9])  # 机器人朝向角偏差为 ±20° (+/-20deg)
    origins = torch.tensor([[5., -2., 0.8]]).repeat(2, 1)
    goals = origins[:, :2] + torch.tensor([1., 0.])
    command = SimpleNamespace(goal_pos_w=goals,
                              _dodge_threat=torch.zeros(2),
                              cfg=SimpleNamespace(kp_yaw=1., max_ang_vel_z=.5))
    env = SimpleNamespace(
        scene={'robot': SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=origins, heading_w=headings,
            root_link_ang_vel_w=torch.zeros(2, 3)))},
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    cos20 = math.cos(math.pi / 9)
    assert torch.allclose(
        walk_goal_heading_when_safe(env, sharpness=4.0),
        torch.full((2,), cos20 ** 4), atol=1e-6)
    assert torch.allclose(
        walk_goal_heading_when_safe(env, sharpness=1.0),
        torch.full((2,), cos20), atol=1e-6)
    # 当朝向偏差超出 30° 阈值（如掉头 180°）时，无论锐化指数 sharpness 为何值，静止不转向的惩罚均固定为负分 -1.0 (Outside the threshold, stopped turning is negative regardless of sharpness).
    env.scene['robot'].data.heading_w += torch.pi
    assert torch.allclose(
        walk_goal_heading_when_safe(env, sharpness=4.0),
        torch.full((2,), -1.), atol=1e-6)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    assert cfg.rewards['walk_goal_heading_when_safe'].params['sharpness'] == 4.0
