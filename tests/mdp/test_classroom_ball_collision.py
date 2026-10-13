"""The classroom ball collides only with robot collision geoms."""
import pytest
import torch

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)


def can_collide(model_a, a, model_b, b):
    return bool((model_a.geom_contype[a] & model_b.geom_conaffinity[b]) |
                (model_b.geom_contype[b] & model_a.geom_conaffinity[a]))


def test_collision_filters_preserve_robot_contacts_and_other_tasks():
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

        # Exercise the real freejoint/event/observation path: an out-of-room ball
        # is physically parked and absent from every robot signal, even before a
        # new forward pass refreshes derived quantities.
        from src.assets.objects.classroom import CLASSROOM_BALL_BOUNDS, CLASSROOM_BALL_PARK_POS
        from src.tasks.amp_loco.mdp.ball_region import ball_in_play, hide_outside_classroom_ball
        from src.tasks.amp_loco.mdp.observations import dodge_ball_state_b
        from src.tasks.amp_loco.mdp.rewards import mimickit_dodge_reward
        pose[:, :3] = pose.new_tensor(CLASSROOM_BALL_BOUNDS[1]) + 1.
        ball.write_root_link_pose_to_sim(pose)
        hide_outside_classroom_ball(env)
        assert not ball_in_play(env).any()
        assert not dodge_ball_state_b(env).any()
        assert not mimickit_dodge_reward(env).any()
        assert not ball_contact(env, **cfg.terminations['ball_hit'].params).any()
        assert torch.allclose(
            env.sim.data.qpos[:, ball.data.indexing.free_joint_q_adr[:3]],
            pose.new_tensor(CLASSROOM_BALL_PARK_POS).expand(2, 3))
        assert not env.sim.data.qvel[:, ball.data.indexing.free_joint_v_adr].any()
        # A full step keeps hidden balls bounded, with unchanged throw timing.
        env._dodge_throw_paused = True
        env.step(torch.zeros(2, env.action_manager.total_action_dim, device='cuda'))
        assert not ball_in_play(env).any()
        assert not dodge_ball_state_b(env).any()
        assert not env.sim.data.qvel[:, ball.data.indexing.free_joint_v_adr].any()
        # The existing throw event reuses the hidden ball; a short launch radius
        # guarantees the launch points stay inside the room for both robots.
        env._dodge_throw_once = True
        throw = cfg.events['throw_ball_on_dwell']
        throw.func(env, None, **{**throw.params, 'dist_range': (.1, .1)})
        hide_outside_classroom_ball(env)
        assert ball_in_play(env).all()
        assert dodge_ball_state_b(env).abs().sum(dim=1).gt(0).all()
    finally:
        env.close()
