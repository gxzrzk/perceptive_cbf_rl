"""教室静态障碍物刚性焊接、坐标固化与机器人碰撞检测测试 (Classroom Static Obstacles Tests)。

测试功能：
- 验证教室中全部 34 个障碍物（桌、椅、外墙）均通过静态刚性焊接 (world-welded) 固定于世界基座，而非使用 mocap 动态体；
- 验证模型自由度优化：彻底剔除 34 个物体的自由浮动关节，减少 204 个速度自由度 (DOF)，大幅提升多环境并行仿真物理步进效率；
- 验证多环境并行执行局部环境重置 (partial reset) 时，静态障碍物位姿不受干扰且保持几何恒定；
- 验证机器人与各静态障碍物碰撞时，接触传感器与 ``wall_contact`` 碰撞终止判定函数能准确触发。

关键断言条件：
- 所有障碍物 entity.is_fixed_base 为 True，且 is_mocap 为 False；
- 编译后模型的自由度满足 model.nv == robot.nv + ball.nv（无多余障碍物 DOF）；
- 仿真步进与物理碰撞发生后，障碍物的位置 geom_xpos 保持绝对静止（误差 < 1e-5）；
- 机器人传送至障碍物位置触发穿透碰撞时，wall_contact 终止信号为 True。

覆盖边界情况：
- 仅重置特定单个子环境 (env_ids=[0])，验证其他并行环境状态的不可侵犯性；
- 机器人与多堵不同墙体（wall_0, wall_1, wall_30）的强制碰撞判定；
- 连续 5 步空动作用于验证长时间物理演化下静态几何体的数值稳定性。
"""
import pytest
import torch

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.terminations import wall_contact


def test_all_classroom_obstacles_are_welded_without_mocap():
    """测试场景：验证教室所有障碍物均无运动学自由度，完全焊接至世界原点。

    预期输出及验证逻辑：
        1. 教室事件中不包含动态移墙的 'pin_wall' 或 'reset_wall_pose' 事件；
        2. initialize_static_classroom 事件仅在仿真启动 'startup' 阶段执行一次；
        3. 34 个物体实体全部满足 is_fixed_base=True 且 is_mocap=False；
        4. 编译后的模型广义坐标数 nq=0、速度自由度 nv=0、mocap数 nmocap=0；
        5. 对比 WallWalk 任务，确认 WallWalk 仍保留动态移动墙体机制，未受影响。
    """
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    assert 'pin_wall' not in cfg.events and 'reset_wall_pose' not in cfg.events
    event = cfg.events['initialize_static_classroom']
    assert event.mode == 'startup'
    assert cfg.scene.env_spacing == 0.
    for name in event.params['wall_names']:
        entity = cfg.scene.entities[name].build()
        assert entity.is_fixed_base and not entity.is_mocap
        model = entity.spec.compile()
        assert model.nq == model.nv == model.nmocap == 0
        assert model.body_weldid[model.body('wall').id] == 0
    original = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    assert 'pin_wall' in original.events and 'reset_wall_pose' in original.events
    entity = original.scene.entities['wall_0'].build()
    assert not entity.is_fixed_base


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU sim')
def test_static_layout_survives_contacts_steps_and_partial_reset():
    """测试场景：实际 GPU 仿真中，静态障碍物在碰撞冲击、仿真步进与单环境重置下保持位置绝对固定。

    输入假数据与测试流程：
        1. 启动 4 个并行环境，核验 34 处几何体初始空间位置 geom_xpos 与 layout 理论坐标一致；
        2. 确认系统总自由度 nv 严格等于机器人的自由度与球的自由度之和（无障碍物自由度开销）；
        3. 仅对环境 0 执行 reset(env_ids=[0])，验证环境 1~3 的 qpos 保持绝对不变，且障碍物位置未移位；
        4. 将机器人强行传送至 wall_0, wall_1, wall_30 障碍物中心触发穿透碰撞，
           验证接触传感器 found > 0，且 wall_contact 判定为 True；
        5. 推进物理步进后，障碍物坐标依然稳定在原位；
        6. 执行 5 步正常控制步进，验证奖励和观测值均为有限实数 (torch.isfinite)。
    """
    from mjlab.envs import ManagerBasedRlEnv
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda')
    try:
        env.reset()
        params = cfg.events['initialize_static_classroom'].params
        names = params['wall_names']
        model = env.sim.mj_model
        body_ids = torch.tensor([model.body(f'{n}/wall').id for n in names], device='cuda')
        geom_ids = torch.tensor([model.geom(f'{n}/wall_collision').id for n in names], device='cuda')
        expected = env.scene.env_origins[:, None, :] + torch.tensor(
            params['layout'], device='cuda')[None, :, :3]
        assert torch.all(env.scene.env_origins == 0)
        assert torch.allclose(env.sim.data.geom_xpos[:, geom_ids], expected, atol=1e-5)
        assert (model.body_weldid[body_ids.cpu().numpy()] == 0).all()
        assert (model.body_mocapid[body_ids.cpu().numpy()] == -1).all()
        # All 34 obstacle free joints (204 velocity DOFs) have been removed.
        robot_model = cfg.scene.entities['robot'].build().spec.compile()
        ball_model = cfg.scene.entities['ball'].build().spec.compile()
        assert model.nv == robot_model.nv + ball_model.nv
        before = env.sim.data.qpos.clone()
        env.reset(env_ids=torch.tensor([0], device='cuda'))
        assert torch.equal(env.sim.data.qpos[1:], before[1:])
        assert not torch.equal(env.sim.data.qpos[0], before[0])
        assert torch.allclose(env.sim.data.geom_xpos[:, geom_ids], expected, atol=1e-5)

        robot = env.scene['robot']
        for name in ('wall_0', 'wall_1', 'wall_30'):
            env.reset()
            gid = model.geom(f'{name}/wall_collision').id
            pose = robot.data.root_link_pose_w.clone()
            pose[:, :3] = env.sim.data.geom_xpos[:, gid]
            robot.write_root_link_pose_to_sim(pose)
            robot.write_root_link_velocity_to_sim(torch.zeros(4, 6, device='cuda'))
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.step_dt)
            found = env.scene[f'{name}_robot_contact'].data.found
            assert found is not None and (found > 0).all(), name
            assert wall_contact(env, **cfg.terminations['wall_hit'].params).all()
            env.sim.step()
            env.sim.forward()
            assert torch.allclose(env.sim.data.geom_xpos[:, geom_ids], expected, atol=1e-5)

        env.reset()
        for _ in range(5):
            obs, reward, _, _, _ = env.step(torch.zeros(4, env.action_manager.total_action_dim, device='cuda'))
            assert torch.isfinite(reward).all() and torch.isfinite(obs['ball_state']).all()
            assert torch.allclose(env.sim.data.geom_xpos[:, geom_ids], expected, atol=1e-5)
    finally:
        env.close()
