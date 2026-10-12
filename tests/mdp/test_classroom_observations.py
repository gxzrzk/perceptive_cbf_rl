"""教室场景障碍物几何感知观测与网络前向传播测试 (Classroom Observations Tests)。

测试功能：
- 验证 CBF 与策略所依赖的障碍物三维几何特征观测函数 ``dodge_obstacle_geometry_b``；
- 验证障碍物排序依据：基于外表面最短欧氏距离 (surface distance) 排序，而非简单的几何中心距离；
- 验证具有较大尺寸的狭长墙体（中心距离较远但表面极近）能够被正确置于最高优先级；
- 验证障碍物在机器人自身基座坐标系 (body frame) 下的相对中心坐标 (rel_x, rel_y, rel_z)、尺寸 (half_x, half_y, half_z)、偏航角 (cos_yaw, sin_yaw) 以及表面距离计算；
- 验证观测对世界坐标系平移与旋转的绝对不变性 (SE(2) / SE(3) 不变性)；
- 验证机器人穿入障碍物内部时，表面距离被平滑截断为 0.0（数值保持有限，无 NaN）；
- 验证并行 GPU 环境下，Actor-Critic 网络结构对包含 8 个最近障碍物几何特征（共 72 维）的观测进行完整推理与步进。

关键断言条件：
- 狭长障碍物因表面距离更近 (0.9m < 0.95m)，排在几何中心更近 (1.0m < 5.1m) 的小物体前面；
- 改变障碍物尺寸后，排序与几何特征实时更新（验证无不合法的静态尺寸缓存）；
- 两个在世界坐标系下具有不同平移和旋转但在机器人局部系下相对姿态一致的环境，观测特征完全相等；
- GPU 仿真前向传播中，actor 动作输出、critic 价值输出以及 step() 返回的奖励和观测全为有限数值 (torch.isfinite)。

覆盖边界情况：
- 远中心但近表面的各向异性长条形障碍物；
- 具有任意偏航角旋转的长方体障碍物；
- 机器人处于障碍物内部或边界表面；
- 并行环境下的张量批量运算与网络输入维度对齐。
"""
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch

from src.tasks.amp_loco.mdp.observations import dodge_obstacle_geometry_b, dodge_wall_state_b
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)


def make_env(centres, half, yaw=None, heading=None, root=None):
    """构造用于测试几何体观测计算的精简仿真环境 Mock。

    参数：
        centres: (n, w, 3) 各环境中障碍物的世界中心坐标
        half: (n, w, 3) 障碍物长方体沿各轴的半长
        yaw: (n, w) 障碍物绕 Z 轴的旋转角
        heading: (n,) 机器人自身的航向角
        root: (n, 3) 机器人根连杆的世界坐标
    """
    centres = torch.tensor(centres, dtype=torch.float32)
    n, w, _ = centres.shape
    yaw = torch.zeros(n, w) if yaw is None else torch.tensor(yaw)
    heading = torch.zeros(n) if heading is None else torch.tensor(heading)
    root = torch.zeros(n, 3) if root is None else torch.tensor(root)
    rotation = torch.zeros(n, w, 3, 3)
    rotation[..., 0, 0] = rotation[..., 1, 1] = yaw.cos()
    rotation[..., 1, 0], rotation[..., 0, 1] = yaw.sin(), -yaw.sin()
    rotation[..., 2, 2] = 1
    names = tuple(f'wall_{i}' for i in range(w))
    ids = {f'{name}/wall_collision': i for i, name in enumerate(names)}
    env = NS(num_envs=n, device='cpu',
             scene={'robot': NS(data=NS(root_link_pos_w=root, heading_w=heading))},
             sim=NS(data=NS(geom_xpos=centres, geom_xmat=rotation),
                    model=NS(geom_size=torch.tensor(half, dtype=torch.float32)),
                    mj_model=NS(geom=lambda name: NS(id=ids[name]))))
    return env, names


def test_near_long_wall_beats_nearer_centre_and_updates_geometry():
    """测试场景：表面距离优先于中心距离排序，且几何体动态形变能实时刷新。

    输入假数据：
        障碍物 0：中心在 (5, 1, 1.5)，尺寸半长 (6, 0.1, 1.5)，是一面巨幅狭长墙体；
        障碍物 1：中心在 (1, 0, 0.38)，尺寸半长 (0.05, 0.05, 0.38)，是一个小方块。
        机器人位于原点 (0, 0, 0)。
    预期输出及验证逻辑：
        1. 障碍物 0 的表面最近距离 = 1.0 - 0.1 = 0.9m；
        2. 障碍物 1 的表面最近距离 = 1.0 - 0.05 = 0.95m；
        3. 尽管障碍物 1 的中心距离 (1.0m) 远小于障碍物 0 的中心距离 (~5.1m)，
           但根据表面距离排序，障碍物 0 必须排在第 1 位 (obs[0, 0] == 5, obs[0, 8] == 0.9)；
        4. 动态增大障碍物 1 的 X 轴半长至 0.5m，其表面距离缩短为 0.5m，
           重新计算观测，验证障碍物 1 成功升至第 1 位，证明排序没有死锁旧缓存。
    """
    env, names = make_env([[[5., 1., 1.5], [1., 0., .38]]],
                          [[[6., .1, 1.5], [.05, .05, .38]]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)
    assert obs.shape == (1, 9)
    assert obs[0, 0] == 5  # 虽然中心距离更远（5.1m vs 1.0m），但其表面距离更近（0.9m vs 0.95m），故排在首位 (farther centre, but surface .9 m vs .95 m)
    assert obs[0, 8].item() == pytest.approx(.9)
    assert torch.allclose(obs[0, 3:6], torch.tensor([6., .1, 1.5]))
    # 缓存仅索引几何体 ID，从不静态缓存位姿/尺寸：动态改变真实几何尺寸后，排序位次立即相应更新 (Cache only IDs, never poses/sizes: changing the actual geometry changes rank).
    env.sim.model.geom_size[0, 1, 0] = .5
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)
    assert obs[0, 0] == 1 and obs[0, 8] == .5


def test_yaw_rotated_box_distance_and_robot_frame():
    """测试场景：带有偏航角旋转的障碍物在机器人机体坐标系下的相对位姿投影。

    输入假数据：
        机器人位于 (0, 0, 1)，航向角为 90°；障碍物中心位于世界坐标 (0, 2, 0.5)，自身旋转 90°。
    预期输出及验证逻辑：
        1. 机器人系下的相对中心位置应旋转变换为 (2.0, 0.0, -0.5)；
        2. 障碍物相对航向角的 (cos, sin) 表示为 (0.0, 1.0)；
        3. 表面距离经旋转投影计算精确为 0.0m。
    """
    env, names = make_env([[[0., 2., .5]]], [[[2., .2, .5]]],
                          yaw=[[torch.pi / 2]], heading=[torch.pi / 2], root=[[0., 0., 1.]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)[0]
    assert torch.allclose(obs[:3], torch.tensor([2., 0., -.5]), atol=1e-6)
    assert torch.allclose(obs[6:8], torch.tensor([0., 1.]), atol=1e-6)
    assert obs[8].item() == pytest.approx(0., abs=1e-6)


def test_world_translation_rotation_invariance_across_envs():
    """测试场景：验证机体观测特征对全局世界坐标系刚体变换的不变性。

    输入假数据：
        环境 0 位于原点，航向 0°；环境 1 整体平移至 (10, 20, 4) 且旋转 90°，但两者的内部相对几何完全相同。
    预期输出及验证逻辑：
        两环境在机体坐标系下生成的几何观测特征向量 obs[0] 与 obs[1] 严格相同 (atol=1e-6)。
    """
    # 构造相同相对几何位姿：环境 1 整体刚体平移至 (9, 22, 3.5) 并旋转 90°，验证机体观测特征的刚体不变性 (Same geometry and relative pose, second environment translated and rotated 90 deg).
    env, names = make_env([[[2., 1., .5]], [[9., 22., 3.5]]],
                          [[[.4, .6, .5]], [[.4, .6, .5]]],
                          yaw=[[0.], [torch.pi / 2]], heading=[0., torch.pi / 2],
                          root=[[0., 0., 1.], [10., 20., 4.]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)
    assert torch.allclose(obs[0], obs[1], atol=1e-6)


def test_inside_distance_is_finite_and_equal_distances_keep_scene_order():
    """测试场景：穿透内部时的平滑截断与等距稳定排序。

    输入假数据：
        机器人中心位于两个重叠障碍物的包络内部。
    预期输出及验证逻辑：
        1. 表面距离 clamped 为 0.0，整个张量无 NaN 或 Inf；
        2. 当距离完全相等时，排序保持稳定的初始场景顺序。
    """
    env, names = make_env([[[0., 0., .4], [.1, 0., .5]]],
                          [[[.4, .6, .4], [.5, .5, .5]]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=2).reshape(2, 9)
    assert torch.isfinite(obs).all() and torch.all(obs[:, 8] == 0)
    assert torch.allclose(obs[:, 0], torch.tensor([0., .1]))


def test_classroom_wiring_leaves_wallwalk_unchanged():
    """测试场景：任务配置项隔离性核查。

    验证逻辑：
        1. 教室任务（训练与 play 模式）绑定 dodge_obstacle_geometry_b，追踪 k=8 个障碍物（34 堵墙中筛选）；
        2. WallWalk 走廊任务仍绑定旧的 dodge_wall_state_b，不受教室新感知项影响。
    """
    for play in (False, True):
        cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=play)
        term = cfg.observations['ball_state'].terms['wall_state']
        assert term.func is dodge_obstacle_geometry_b and term.params['k'] == 8
        assert len(term.params['wall_names']) == 34
    cfg = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    assert cfg.observations['ball_state'].terms['wall_state'].func is dodge_wall_state_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU sim')
def test_gpu_observation_and_actor_critic_forward():
    """测试场景：GPU 上包含 16 个环境的完整强化学习模型前向推理与仿真步进链路。

    输入假数据与测试流程：
        1. 构建 16 环境的真实 GPU 仿真；
        2. 计算几何观测项，断言维度为 (16, 72)（8 个物体 * 9 维特征）；
        3. 距离字段必须按升序排列（即 distances.diff >= 0）；
        4. 初始化 AMPOnPolicyRunner，提取 Actor 与 Critic 网络；
        5. 在推理模式下输入观测字典，前向计算 action 与 value，执行 3 步 wrapper.step()。
    预期输出及验证逻辑：
        1. 动作形状为 (16, num_actions)，价值形状为 (16, 1)；
        2. 动作、价值、奖励与下一步观测全部有限 (torch.isfinite)。
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from src.tasks.amp_loco.rl import AMPOnPolicyRunner
    from src.tasks.amp_loco.config.g1.goto_rl_cfg import g1_amp_dodge_mimickit_classroom_ppo_runner_cfg

    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    cfg.scene.num_envs = 16
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda')
    try:
        wrapper = RslRlVecEnvWrapper(env)
        env.reset()
        term = cfg.observations['ball_state'].terms['wall_state']
        geometry = term.func(env, **term.params)
        assert geometry.shape == (16, 72) and torch.isfinite(geometry).all()
        distances = geometry.reshape(16, 8, 9)[..., 8]
        assert (distances.diff(dim=1) >= 0).all()
        obs = wrapper.get_observations()
        runner_cfg = asdict(g1_amp_dodge_mimickit_classroom_ppo_runner_cfg())
        # 采用实际训练运行器（Runner）的网络模型配置转换与观测维度自动推断逻辑 (Use the actual runner's model-config conversion and dimension inference).
        runner_cfg.update(logger='tensorboard', num_steps_per_env=4,
                          amp_replay_buffer_size=1024)
        runner = AMPOnPolicyRunner(wrapper, runner_cfg, log_dir=None, device='cuda')
        models = {'actor': runner.alg.actor, 'critic': runner.alg.critic}
        for model in models.values():
            assert 'ball_state' in model.obs_groups
        with torch.inference_mode():
            for _ in range(3):
                actions = models['actor'](obs)
                values = models['critic'](obs)
                assert actions.shape == (16, wrapper.num_actions)
                assert values.shape == (16, 1)
                assert torch.isfinite(actions).all() and torch.isfinite(values).all()
                obs, rewards, _, _ = wrapper.step(actions)
                assert torch.isfinite(rewards).all()
                assert torch.isfinite(obs['ball_state']).all()
    finally:
        env.close()
