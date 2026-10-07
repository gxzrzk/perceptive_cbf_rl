"""Collision geometry observations: selection, coordinates and policy integration."""
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
    env, names = make_env([[[5., 1., 1.5], [1., 0., .38]]],
                          [[[6., .1, 1.5], [.05, .05, .38]]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)
    assert obs.shape == (1, 9)
    assert obs[0, 0] == 5  # farther centre, but surface .9 m vs .95 m
    assert obs[0, 8].item() == pytest.approx(.9)
    assert torch.allclose(obs[0, 3:6], torch.tensor([6., .1, 1.5]))
    # Cache only IDs, never poses/sizes: changing the actual geometry changes rank.
    env.sim.model.geom_size[0, 1, 0] = .5
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)
    assert obs[0, 0] == 1 and obs[0, 8] == .5


def test_yaw_rotated_box_distance_and_robot_frame():
    env, names = make_env([[[0., 2., .5]]], [[[2., .2, .5]]],
                          yaw=[[torch.pi / 2]], heading=[torch.pi / 2], root=[[0., 0., 1.]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)[0]
    assert torch.allclose(obs[:3], torch.tensor([2., 0., -.5]), atol=1e-6)
    assert torch.allclose(obs[6:8], torch.tensor([0., 1.]), atol=1e-6)
    assert obs[8].item() == pytest.approx(0., abs=1e-6)


def test_world_translation_rotation_invariance_across_envs():
    # Same geometry and relative pose, second environment translated and rotated 90 deg.
    env, names = make_env([[[2., 1., .5]], [[9., 22., 3.5]]],
                          [[[.4, .6, .5]], [[.4, .6, .5]]],
                          yaw=[[0.], [torch.pi / 2]], heading=[0., torch.pi / 2],
                          root=[[0., 0., 1.], [10., 20., 4.]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=1)
    assert torch.allclose(obs[0], obs[1], atol=1e-6)


def test_inside_distance_is_finite_and_equal_distances_keep_scene_order():
    env, names = make_env([[[0., 0., .4], [.1, 0., .5]]],
                          [[[.4, .6, .4], [.5, .5, .5]]])
    obs = dodge_obstacle_geometry_b(env, wall_names=names, k=2).reshape(2, 9)
    assert torch.isfinite(obs).all() and torch.all(obs[:, 8] == 0)
    assert torch.allclose(obs[:, 0], torch.tensor([0., .1]))


def test_classroom_wiring_leaves_wallwalk_unchanged():
    for play in (False, True):
        cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=play)
        term = cfg.observations['ball_state'].terms['wall_state']
        assert term.func is dodge_obstacle_geometry_b and term.params['k'] == 8
        assert len(term.params['wall_names']) == 34
    cfg = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    assert cfg.observations['ball_state'].terms['wall_state'].func is dodge_wall_state_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU sim')
def test_gpu_observation_and_actor_critic_forward():
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
        # Use the actual runner's model-config conversion and dimension inference.
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
