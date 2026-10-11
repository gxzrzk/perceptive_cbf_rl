"""教室避障环境中球体碰撞过滤与接触终止测试 (Classroom Ball Collision Tests)。

测试功能：
- 验证球体 (ball) 仅与机器人的碰撞几何体 (robot collision geoms) 发生物理碰撞接触；
- 验证球体与教室中的墙体、桌椅等障碍物 (scenery/walls) 以及地面完全无碰撞（穿透），防止球体被教室静态家具反弹或阻挡；
- 验证机器人与教室障碍物之间的碰撞属性保持激活；
- 验证教室任务的碰撞配置隔离性，确保不污染其他任务（如 WallWalk）的全局机器人碰撞预设；
- 验证球体穿过障碍物时不触发接触，但在击中机器人骨盆 (pelvis) 时正常触发终止条件 (ball_contact termination)。

关键断言条件：
- can_collide(ball, robot_collision_geoms) 全部为 True；
- can_collide(ball, scenery_geoms) 全部为 False；
- can_collide(robot, scenery_geoms) 全部为 True；
- 放置球体穿透墙壁或地面时，MuJoCo contact 接触列表中完全不包含 ball_gid；
- 将球体置于机器人骨盆内部时，ball_contact 终止判定对于所有环境均返回 True。

覆盖边界情况：
- 球体重叠于多堵墙体（如 wall_0, wall_1, wall_30）；
- 球体穿透地面 (z = -0.03m)；
- 球体与机器人躯干的精确相交判定；
- 跨任务场景创建时的预设污染防范。
"""
import pytest
import torch

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)


def can_collide(model_a, a, model_b, b):
    """判断两个 MuJoCo 几何体根据 contype 和 conaffinity 位掩码是否允许发生碰撞。

    判定规则：(contype_a & conaffinity_b) != 0 或 (contype_b & conaffinity_a) != 0。
    """
    return bool((model_a.geom_contype[a] & model_b.geom_conaffinity[b]) |
                (model_b.geom_contype[b] & model_a.geom_conaffinity[a]))


def test_collision_filters_preserve_robot_contacts_and_other_tasks():
    """测试场景：验证教室任务中的碰撞掩码规则以及对其他任务的无副作用性。

    输入假数据：
        从教室任务与 WallWalk 任务分别编译 robot, ball 及各静态墙体的 MuJoCo spec。
    预期输出及验证逻辑：
        1. 机器人必须存在以 '_collision' 结尾的碰撞几何体；
        2. 球体碰撞几何体 (ball_collision) 与机器人的每个碰撞几何体均可发生碰撞；
        3. 遍历教室中所有静态障碍物（34 堵桌椅与墙体），球体与障碍物的任何几何体均不可碰撞；
        4. 机器人与障碍物的碰撞几何体之间必须保持可碰撞；
        5. 检查 WallWalk 任务原始配置，验证教室环境的自定义掩码未修改通用的机器人和球体碰撞层。
    """
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    ball = cfg.scene.entities['ball'].build().spec.compile()
    robot = cfg.scene.entities['robot'].build().spec.compile()
    ball_id = ball.geom('ball_collision').id
    collision_ids = [i for i in range(robot.ngeom) if robot.geom(i).name.endswith('_collision')]
    assert collision_ids
    assert all(can_collide(ball, ball_id, robot, i) for i in collision_ids)
    for name in cfg.events['initialize_static_classroom'].params['wall_names']:
        scenery = cfg.scene.entities[name].build().spec.compile()
        assert all(not can_collide(ball, ball_id, scenery, i) for i in range(scenery.ngeom))
        gid = scenery.geom('wall_collision').id
        assert all(can_collide(robot, i, scenery, gid) for i in collision_ids)
    # Creating the classroom must not mutate shared robot collision presets.
    original = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    original_robot = original.scene.entities['robot'].build().spec.compile()
    original_ball = original.scene.entities['ball'].build().spec.compile()
    assert (original_robot.geom_conaffinity[collision_ids] == 1).all()
    assert original_ball.geom_contype[0] == 1 and original_ball.geom_conaffinity[0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU sim')
def test_ball_passes_scene_but_robot_contact_terminates():
    """测试场景：实际仿真步进中，球体穿透静态物体不产生碰撞点，但接触机器人必然触发终止。

    输入假数据与测试场景：
        1. 启动 2 个并行环境的教室仿真；
        2. 将球体位置分别强制移动到 wall_0, wall_1, wall_30 几何体中心以及地面下方 (z = -0.03m)；
        3. 将球体放置在机器人的根连杆（骨盆）中心。
    预期输出及验证逻辑：
        1. 模型级几何体过滤检查：除机器人外，其他几何体与球体不可碰撞；
        2. 执行 mj_forward，检查 contact 接触列表，验证球体与墙体、地面重叠时 contact 列表中无球体几何体；
        3. 将球体写入机器人骨盆并在 GPU 上推进前向动力学，调用 ball_contact() 终止判定函数，必须返回 True。
    """
    import mujoco
    import mujoco_warp as mjw
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.amp_loco.mdp.terminations import ball_contact

    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda')
    try:
        env.reset()
        model = env.sim.mj_model
        ball = env.scene['ball']
        ball_gid = model.geom('ball/ball_collision').id
        robot_ids = {i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/')}
        for i in range(model.ngeom):
            if i != ball_gid and i not in robot_ids:
                assert not can_collide(model, ball_gid, model, i), model.geom(i).name
        # CPU forward on the same compiled scene checks actual overlaps, not only masks.
        data = mujoco.MjData(model)
        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        ball_adr = model.jnt_qposadr[model.body_jntadr[model.geom_bodyid[ball_gid]]]
        for name in ('wall_0/wall_collision', 'wall_1/wall_collision', 'wall_30/wall_collision'):
            mujoco.mj_forward(model, data)
            data.qpos[ball_adr:ball_adr + 3] = data.geom_xpos[model.geom(name).id]
            mujoco.mj_forward(model, data)
            assert all(ball_gid not in data.contact[j].geom for j in range(data.ncon))
        data.qpos[ball_adr:ball_adr + 3] = (0., 0., -.03)
        mujoco.mj_forward(model, data)
        assert all(ball_gid not in data.contact[j].geom for j in range(data.ncon))

        # Place the ball inside the robot pelvis in both GPU worlds. The existing
        # ball_robot_contact sensor and termination must still recognize the hit.
        pose = torch.zeros(2, 7, device='cuda')
        pose[:, :3] = env.scene['robot'].data.root_link_pos_w
        pose[:, 3] = 1
        ball.write_root_link_pose_to_sim(pose)
        ball.write_root_link_velocity_to_sim(torch.zeros(2, 6, device='cuda'))
        env.scene.write_data_to_sim()
        mjw.forward(env.sim.wp_model, env.sim.wp_data)
        env.scene.update(env.step_dt)
        assert ball_contact(env, **cfg.terminations['ball_hit'].params).all()
    finally:
        env.close()
