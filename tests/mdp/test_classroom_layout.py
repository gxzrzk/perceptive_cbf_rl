"""Check navigable clearance using collision envelopes, including chair backs."""
import pytest
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)


def test_classroom_clearance_and_wiring():
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
    assert {p[1] for p in poses[:30:2]} == {-5.3, -1.9, 1.9}
    # Inner desk edges define the central aisle.
    assert (1.9 - 0.6) * 2 == pytest.approx(2.6)
    # Adjacent columns define the two additional longitudinal aisles.
    assert (5.3 - 0.6) - (1.9 + 0.6) == pytest.approx(2.2)
    # Every chair/desk pair must leave the promised cross aisle to the next row.
    for row in range(4):
        for col in range(3):
            desk = poses[row * 6 + col * 2]
            next_chair = poses[(row + 1) * 6 + col * 2 + 1]
            assert next_chair[0] - 0.25 - (desk[0] + 0.4) == pytest.approx(2.2)
    for name, pose in zip(names, poses, strict=True):
        asset = cfg.scene.entities[name]
        spec = asset.spec_fn()
        assert tuple(spec.body('wall').pos) == pytest.approx(pose[:3])
        model = spec.compile()
        assert model.geom('wall_collision').contype == 1


def test_random_routes_stay_clear_of_furniture():
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
    assert ((route[..., 0] >= 0) & (route[..., 0] <= 20.9)).all()
    assert (route[..., 1].abs() <= 3.6).all()
    # Sample all segments and check clearance to every desk and chair footprint.
    t = torch.linspace(0, 1, 17)
    samples = route[:, :-1, None] * (1 - t[None, None, :, None]) + route[:, 1:, None] * t[None, None, :, None]
    for i, pose in enumerate(cfg.events['initialize_static_classroom'].params['layout'][:30]):
        half = (0.4, 0.6) if i % 2 == 0 else (0.25, 0.25)
        outside = (samples - torch.tensor(pose[:2])).abs() - torch.tensor(half)
        distance = outside.clamp_min(0).norm(dim=-1)
        assert (distance >= 0.9).all()


def test_classroom_registered_task_configs():
    import src.tasks  # noqa: F401 -- register all project tasks
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
    from types import SimpleNamespace
    import torch
    from src.tasks.amp_loco.mdp.rewards import walk_goal_heading_when_safe
    headings = torch.tensor([0., torch.pi / 2, torch.pi, -torch.pi / 2, 0., torch.pi, 0.])
    origins = torch.tensor([[5., -2., 0.8]]).repeat(7, 1)
    goals = origins[:, :2] + torch.tensor([1., 0.])
    goals[-1] = origins[-1, :2]
    command = SimpleNamespace(goal_pos_w=goals,
                              _dodge_threat=torch.tensor([0, 0, 0, 0, 1, 1, 0]))
    env = SimpleNamespace(
        scene={'robot': SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=origins, heading_w=headings))},
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    value = walk_goal_heading_when_safe(env)
    assert torch.allclose(value, torch.tensor([1., 0., -1., 0., 0., 0., 0.]), atol=1e-6)
    # Turning both robot and target by 90 degrees preserves the reward.
    command.goal_pos_w[:6] = origins[:6, :2] + torch.tensor([0., 1.])
    env.scene['robot'].data.heading_w += torch.pi / 2
    assert torch.allclose(walk_goal_heading_when_safe(env), value, atol=1e-6)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    assert cfg.rewards['walk_goal_heading_when_safe'].weight == 2.0
