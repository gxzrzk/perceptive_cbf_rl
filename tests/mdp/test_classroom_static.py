"""World-welded classroom scenery, per-world placement and robot contacts."""
import pytest
import torch

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.terminations import wall_contact


def test_all_classroom_obstacles_are_welded_without_mocap():
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
