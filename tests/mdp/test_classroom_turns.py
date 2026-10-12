"""教室路径拐弯过渡、圆弧倒角与避障安全性测试 (Classroom Turns Tests)。

测试功能：
- 验证机器人在折线路径拐角处的平滑分段推进判定逻辑；
- 验证拐角过渡区无缝切换：无需严格越过旧线段的正交末端垂面，只要进入转弯切换半径 (path_switch_radius) 且贴近新线段，分段索引 (_path_seg_idx) 即可自然递增；
- 验证在未达到转角过渡区时，分段索引保持锁定在当前分段；
- 验证回环路径在交叉点附近不会误跳到未来的圈数 (later lap)；
- 验证圆弧倒角平滑算法 (round_classroom_route)：在直角或锐角拐弯处用相切圆弧替代生硬拐点，将大角度偏转分散为小角度微段（相邻段航向差 <= 15°）；
- 验证平滑圆弧路径对教室全部 34 处家具与墙体障碍物的安全距离均 >= CLASSROOM_ROUTE_CLEARANCE；
- 验证动态扩容缓存 (_ensure_stash) 时，未重置环境的路径历史与累积弧长保持完好无损。

关键断言条件：
- 处于拐角切换区域时：_path_seg_idx 顺利从 0 推进到 1；
- 处于非拐角区域时：_path_seg_idx 保持为 0；
- 圆弧倒角离散点位于相切圆上（半径为 0.6m），相邻切段航向角差值 <= pi/12 (15°)；
- 64 个环境生成的随机平滑路线总长 > 200m，且对所有障碍物的采样最短距离均 >= 通行间隙；
- 扩容缓存前后，现有环境数据完全一致。

覆盖边界情况：
- 拐角过渡附近的微小扰动点；
- 自相交环形路线的重叠节点；
- 各种随机起始走廊车道与随机横向偏移；
- 路径缓存跨环境批量扩容与状态保护。
"""
from types import SimpleNamespace as NS

import pytest
import torch

from src.assets.objects.classroom import (
    CLASSROOM_ROUTE_CLEARANCE, CLASSROOM_ROUTE_STATIONS, CLASSROOM_SPAWN_LANES,
)
from src.tasks.amp_loco.mdp.classroom import generate_classroom_route, round_classroom_route
from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommand
from src.tasks.amp_loco.mdp.walk_path import _ensure_stash
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg


def make_command(points, robot_xy):
    """构造用于测试路径转弯与分段切换的模拟指令控制器。

    参数：
        points: 路径折线点序列
        robot_xy: 机器人当前的 2D 平面坐标
    """
    points = torch.tensor([points], dtype=torch.float32)
    cumulative = torch.cat((torch.zeros(1, 1), (points[:, 1:] - points[:, :-1]).norm(dim=-1).cumsum(-1)), -1)
    env = NS(device='cpu', _walk_path_pts=points, _walk_path_cumlen=cumulative,
             _walk_path_n=torch.tensor([points.shape[1]]), _walk_path_epoch=torch.tensor([1]))
    return NS(_env=env, num_envs=1, device='cpu', _path_epoch_seen=torch.tensor([1]),
              _path_seg_idx=torch.tensor([0]), _path_s=torch.tensor([0.]),
              robot=NS(data=NS(root_link_pos_w=torch.tensor([[*robot_xy, .8]]))),
              cfg=NS(path_lookahead=.8, path_wall_ahead=0., path_switch_radius=.8),
              goal_pos_w=torch.zeros(1, 2), is_standing_env=torch.zeros(1, dtype=torch.bool),
              is_inplace_env=torch.zeros(1, dtype=torch.bool))


@pytest.mark.parametrize('xy', [(5., .7), (4.9, .7), (5., 0.)])
def test_turn_advances_without_crossing_old_end_plane(xy):
    """测试场景：在直角拐角处，当机器人进入转角半径并靠近下一段时，分段索引向前推移。

    输入假数据：
        由 [(0,0), (5,0), (5,4)] 构成的 90° 拐角，机器人位于拐弯过渡区附近的 xy 点。
    预期输出及验证逻辑：
        1. 机器人无需严格越过旧线段端面，分段索引 _path_seg_idx 即推进至 1（转入新线段）；
        2. 追踪的前方目标点与机器人距离保持在 0.25m 以上；
        3. 累计进度 _path_s 正确累加了拐弯后的位移。
    """
    command = make_command([(0., 0.), (5., 0.), (5., 4.)], xy)
    DodgeGoToGoalCommand._update_path_goal(command)
    assert command._path_seg_idx.item() == 1
    assert (command.goal_pos_w - command.robot.data.root_link_pos_w[:, :2]).norm() > .25
    if xy[1] > 0:
        assert command._path_s.item() > 5


@pytest.mark.parametrize('xy', [(2., 1.), (4.8, .1), (4.9, -.4)])
def test_turn_does_not_advance_before_entering_next_segment(xy):
    """测试场景：当机器人尚未到达拐角或偏离有效拐角过渡区时，禁止提前推进分段。

    输入假数据：
        处于第 0 段中途或远离过渡区的三个代表点。
    预期输出及验证逻辑：
        分段索引 _path_seg_idx 严格维持为 0。
    """
    command = make_command([(0., 0.), (5., 0.), (5., 4.)], xy)
    DodgeGoToGoalCommand._update_path_goal(command)
    assert command._path_seg_idx.item() == 0


def test_intersection_does_not_jump_to_later_lap():
    """测试场景：自相交的循环回环路径，在起点重合区域不发生圈数跳跃。

    输入假数据：
        一条环形回绕后重回起点的 6 节点路径，机器人位于第一圈原点附近 (1, 0)。
    预期输出及验证逻辑：
        分段索引保持为 0，累计进度保持在 1.0m，不跳入第 5 段或后续圈数。
    """
    command = make_command([(0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.), (5., 0.)], (1., 0.))
    DodgeGoToGoalCommand._update_path_goal(command)
    assert command._path_seg_idx.item() == 0
    assert command._path_s.item() == pytest.approx(1.)


def test_arc_geometry_and_progress():
    """测试场景：圆弧倒角几何性质校验及沿平滑弧线前行的单调追踪表现。

    输入假数据：
        对 [(0,0), (5,0), (5,4)] 进行圆弧倒角处理。
    预期输出及验证逻辑：
        1. 插入的 7 个平滑采样点精确位于圆心为 (4.4, 0.6)、半径为 0.6m 的外切相切圆弧上；
        2. 相邻采样段之间的航向角差值均 <= 15° (pi / 12)，消除直角急转；
        3. 模拟机器人在圆弧各分段上以 20%、50%、80%、100% 步长移动，
           验证累计进度 _path_s 单调递增，且始终保持有效前视距离。
    """
    points, counts = round_classroom_route(torch.tensor([[[0., 0.], [5., 0.], [5., 4.]]]))
    route = points[0, :counts[0]]
    # 圆弧内插点验证：所生成的 7 个插值点精确落在以 (4.4, 0.6) 为圆心、半径为 0.6m 的切向圆弧上 (The seven sampled points lie on the tangent circle centered at (4.4, .6)).
    assert torch.allclose((route[1:-1] - torch.tensor([4.4, .6])).norm(dim=-1), torch.full((7,), .6), atol=1e-5)
    segments = route[1:] - route[:-1]
    headings = torch.atan2(segments[:, 1], segments[:, 0])
    assert (headings.diff().abs() <= torch.pi / 12 + 1e-5).all()
    # 在拐弯后追加一段直线尾段 [5., 10.]，使得测试专注于检验转弯路径跟踪，而非提前触发到达终点时的动态路线延展逻辑 (Add a straight tail so the test exercises following, not route extension).
    command = make_command(route.tolist() + [[5., 10.]], (0., 0.))
    last_progress = 0.
    for i in range(len(route) - 1):
        for fraction in (.2, .5, .8, 1.):
            point = route[i] + fraction * (route[i + 1] - route[i])
            command.robot.data.root_link_pos_w[0, :2] = point
            DodgeGoToGoalCommand._update_path_goal(command)
            assert command._path_s.item() >= last_progress - 1e-5
            last_progress = command._path_s.item()
            remaining = command._env._walk_path_cumlen[0, -1] - last_progress
            if remaining > 1.:
                assert (command.goal_pos_w[0] - point).norm() > .25
    assert command._path_seg_idx.item() == len(route) - 1


def test_rounded_random_routes_clear_all_obstacles():
    """测试场景：大规模随机生成的圆弧平滑路线对所有教室障碍物的无碰撞通行验证。

    输入假数据：
        生成 64 条随机教室路线，随机采样出生车道与横向偏差，并执行圆弧平滑倒角。
    预期输出及验证逻辑：
        1. 路线起点与初始 (x, y) 吻合；
        2. 有效分段长度 > 1e-5，有效路线总长度均 > 200m；
        3. 对所有线段采样 9 个点，计算采样点到场景中 34 个静态障碍物边界框的最短欧氏距离；
        4. 断言所有采样点的最短距离均 >= CLASSROOM_ROUTE_CLEARANCE，保证平滑后无碰撞。
    """
    torch.manual_seed(24)
    n = 64
    lane = torch.randint(2, (n,))
    x = torch.rand(n) * CLASSROOM_ROUTE_STATIONS[-1]
    y = torch.tensor(CLASSROOM_SPAWN_LANES)[lane] + (torch.rand(n) * 2 - 1) * .15
    route, counts = round_classroom_route(generate_classroom_route(x, y, lane))
    assert torch.allclose(route[:, 0], torch.stack((x, y), -1))
    lengths = (route[:, 1:] - route[:, :-1]).norm(dim=-1)
    valid = torch.arange(lengths.shape[1])[None] < counts[:, None] - 1
    assert (lengths[valid] > 1e-5).all()
    assert ((lengths * valid).sum(-1) > 200).all()
    t = torch.linspace(0, 1, 9)
    samples = route[:, :-1, None] * (1 - t[None, None, :, None]) + route[:, 1:, None] * t[None, None, :, None]
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    params = cfg.events['initialize_static_classroom'].params
    for name, pose in zip(params['wall_names'], params['layout'], strict=True):
        size = cfg.scene.entities[name].spec_fn().geom('wall_collision').size[:2]
        outside = (samples - torch.tensor(pose[:2])).abs() - torch.tensor(size.copy())
        assert (outside.clamp_min(0).norm(dim=-1) >= CLASSROOM_ROUTE_CLEARANCE - 1e-5).all()


def test_larger_stash_preserves_nonreset_environments():
    """测试场景：当路径点数超过初始缓冲区大小时，触发动态扩容，保护未重置环境数据。

    输入假数据：
        2 个环境，向环境 1 写入特定的路径顶点、累计长度、分段数及 epoch。
    预期输出及验证逻辑：
        调用 _ensure_stash(max_points=548) 执行扩容后，环境 1 的历史数据严格保持不变。
    """
    env = NS(device='cpu', num_envs=2)
    _ensure_stash(env)
    env._walk_path_pts[1, :3] = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    env._walk_path_cumlen[1, :3] = torch.tensor([0., 2., 4.])
    env._walk_path_n[1] = 3
    env._walk_path_epoch[1] = 7
    original = env._walk_path_pts[1].clone()
    _ensure_stash(env, max_points=548)
    assert torch.equal(env._walk_path_pts[1, :len(original)], original)
    assert env._walk_path_n[1] == 3 and env._walk_path_epoch[1] == 7
    assert torch.equal(env._walk_path_cumlen[1, :3], torch.tensor([0., 2., 4.]))
