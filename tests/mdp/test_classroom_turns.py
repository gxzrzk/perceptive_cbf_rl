"""Rounded route clearance and regressions for local path-segment transitions."""
from types import SimpleNamespace as NS

import pytest
import torch

from src.tasks.amp_loco.mdp.classroom import generate_classroom_route, round_classroom_route
from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommand
from src.tasks.amp_loco.mdp.walk_path import _ensure_stash
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg


def make_command(points, robot_xy):
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
    command = make_command([(0., 0.), (5., 0.), (5., 4.)], xy)
    DodgeGoToGoalCommand._update_path_goal(command)
    assert command._path_seg_idx.item() == 1
    assert (command.goal_pos_w - command.robot.data.root_link_pos_w[:, :2]).norm() > .25
    if xy[1] > 0:
        assert command._path_s.item() > 5


@pytest.mark.parametrize('xy', [(2., 1.), (4.8, .1), (4.9, -.4)])
def test_turn_does_not_advance_before_entering_next_segment(xy):
    command = make_command([(0., 0.), (5., 0.), (5., 4.)], xy)
    DodgeGoToGoalCommand._update_path_goal(command)
    assert command._path_seg_idx.item() == 0


def test_intersection_does_not_jump_to_later_lap():
    command = make_command([(0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.), (5., 0.)], (1., 0.))
    DodgeGoToGoalCommand._update_path_goal(command)
    assert command._path_seg_idx.item() == 0
    assert command._path_s.item() == pytest.approx(1.)


def test_arc_geometry_and_progress():
    points, counts = round_classroom_route(torch.tensor([[[0., 0.], [5., 0.], [5., 4.]]]))
    route = points[0, :counts[0]]
    # The seven sampled points lie on the tangent circle centered at (4.4, .6).
    assert torch.allclose((route[1:-1] - torch.tensor([4.4, .6])).norm(dim=-1), torch.full((7,), .6), atol=1e-5)
    segments = route[1:] - route[:-1]
    headings = torch.atan2(segments[:, 1], segments[:, 0])
    assert (headings.diff().abs() <= torch.pi / 12 + 1e-5).all()
    # Add a straight tail so the test exercises following, not route extension.
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
    torch.manual_seed(24)
    n = 64
    lane = torch.randint(2, (n,))
    x = torch.rand(n) * 20.9
    y = torch.tensor([0., -3.6])[lane] + (torch.rand(n) * 2 - 1) * .15
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
        assert (outside.clamp_min(0).norm(dim=-1) >= .9 - 1e-5).all()


def test_larger_stash_preserves_nonreset_environments():
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
