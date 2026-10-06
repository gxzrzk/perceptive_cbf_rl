"""Check navigable clearance using collision envelopes, including chair backs."""
import pytest
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)


def test_classroom_clearance_and_wiring():
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    params = cfg.events['reset_wall_pose'].params
    poses = params['layout']
    names = params['wall_names']
    assert len(names) == 52
    assert 'recycle_walls_ahead' not in cfg.events
    assert cfg.commands['twist'].rel_standing_envs == 0
    assert cfg.commands['twist'].rel_inplace_throw_envs == 0
    assert cfg.events['reset_walk_path'].func.__name__ == 'reset_classroom_robot'
    assert cfg.observations['ball_state'].terms['wall_state'].params['k'] == 6
    throw = cfg.events['throw_ball_on_dwell'].params
    assert throw['omnidirectional'] is True
    assert throw['along_path'] is False
    assert throw['launch_speed_range'] is None
    # Inner desk edges define the central aisle.
    assert (2.1 - 0.6) * 2 == pytest.approx(3.0)
    # Adjacent columns define the two additional longitudinal aisles.
    assert (5.8 - 0.6) - (2.1 + 0.6) == pytest.approx(2.5)
    # Every chair/desk pair must leave the promised cross aisle to the next row.
    for row in range(5):
        for col in range(4):
            desk = poses[row * 8 + col * 2]
            next_chair = poses[(row + 1) * 8 + col * 2 + 1]
            assert next_chair[0] - 0.25 - (desk[0] + 0.4) == pytest.approx(2.5)
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
    assert ((route[..., 0] >= 0) & (route[..., 0] <= 27)).all()
    assert (route[..., 1].abs() <= 3.95).all()
    # Sample all segments and check clearance to every desk and chair footprint.
    t = torch.linspace(0, 1, 17)
    samples = route[:, :-1, None] * (1 - t[None, None, :, None]) + route[:, 1:, None] * t[None, None, :, None]
    for i, pose in enumerate(cfg.events['reset_wall_pose'].params['layout'][:48]):
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
