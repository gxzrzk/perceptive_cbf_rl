"""Continuous aisle routes and one-shot safe episode completion."""
import math
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from src.assets.objects.classroom import CLASSROOM_ROUTE_STATIONS, CLASSROOM_SPAWN_LANES
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
from src.tasks.amp_loco.mdp.classroom import (
    _write_classroom_route, extend_classroom_route, generate_classroom_route,
)
from src.tasks.amp_loco.mdp.path_navigation import classroom_follow_state
from src.tasks.amp_loco.mdp.walk_path import walk_path_query
from tests.mdp.test_classroom_follow import advance, follower, reward


@pytest.mark.parametrize('dt', [.01, .02, .1])
def test_safe_completion_only_at_timeout_and_never_on_simultaneous_failure(dt):
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    env = NS(num_envs=5, device='cpu', step_dt=dt, elapsed=torch.zeros(5),
             ball=torch.zeros(5, dtype=torch.bool), wall=torch.zeros(5, dtype=torch.bool),
             fall=torch.zeros(5, dtype=torch.bool))
    terms = {
        'time_out': TerminationTermCfg(func=lambda e: e.elapsed >= 25., time_out=True),
        'ball_hit': TerminationTermCfg(func=lambda e: e.ball),
        'wall_hit': TerminationTermCfg(func=lambda e: e.wall),
        'bad_base_height': TerminationTermCfg(func=lambda e: e.fall),
    }
    env.termination_manager = TerminationManager(terms, env)
    manager = RewardManager({'completion': cfg.rewards['classroom_safe_completion']}, env, scale_by_dt=True)
    env.elapsed[:] = 25. - dt
    env.termination_manager.compute()
    assert torch.equal(manager.compute(dt=dt), torch.zeros(5))
    env.elapsed[:4] = 25.
    env.ball[1], env.wall[2], env.fall[3] = True, True, True
    env.termination_manager.compute()
    assert manager.compute(dt=dt).tolist() == pytest.approx([10., 0., 0., 0., 0.])
    assert env.termination_manager.terminated.tolist() == [False, True, True, True, False]
    # Earlier collisions cannot survive to a timeout: all are immediate failures.
    env.elapsed[:] = 0.
    env.ball[:], env.wall[:], env.fall[:] = False, False, False
    env.termination_manager.compute()
    assert torch.equal(manager.compute(dt=dt), torch.zeros(5))


def test_extension_preserves_prefix_epoch_other_env_and_graph_clearance():
    torch.manual_seed(17)
    env = NS(num_envs=2, device='cpu', scene=NS(env_origins=torch.zeros(2, 3)),
             _classroom_route_extension_nodes=8)
    lane = torch.tensor([0, 1])
    raw = generate_classroom_route(torch.tensor([2., 2.]),
                                  torch.tensor(CLASSROOM_SPAWN_LANES)[lane], lane, count=4)
    _write_classroom_route(env, torch.arange(2), raw)
    env._walk_path_epoch[:] = torch.tensor([3, 7])
    old_points, old_cum, old_counts = (env._walk_path_pts.clone(), env._walk_path_cumlen.clone(),
                                     env._walk_path_n.clone())
    extend_classroom_route(env, torch.tensor([0]), torch.tensor([60.]))
    assert env._classroom_route_n[0] > 4 and env._classroom_route_n[1] == 4
    assert torch.equal(env._walk_path_epoch, torch.tensor([3, 7]))
    prefix = old_counts[0].item() - 1  # Only the former final junction is rounded.
    assert torch.equal(env._walk_path_pts[0, :prefix], old_points[0, :prefix])
    assert torch.equal(env._walk_path_cumlen[0, :prefix], old_cum[0, :prefix])
    assert torch.equal(env._walk_path_pts[1, :old_points.shape[1]], old_points[1])
    assert torch.equal(env._walk_path_cumlen[1, :old_cum.shape[1]], old_cum[1])
    assert env._walk_path_n[1] == old_counts[1]
    total = env._walk_path_cumlen[0, env._walk_path_n[0] - 1]
    assert total >= 60.
    nodes = env._classroom_route_pts[0, :env._classroom_route_n[0]]
    assert not torch.any((nodes[2:] - nodes[:-2]).norm(dim=-1) < 1e-5)
    assert nodes[:, 0].min() >= 0 and nodes[:, 0].max() <= CLASSROOM_ROUTE_STATIONS[-1]
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    points = env._walk_path_pts[0, :env._walk_path_n[0]]
    t = torch.linspace(0, 1, 9)[None, :, None]
    samples = points[:-1, None] * (1 - t) + points[1:, None] * t
    for name, pose in zip(cfg.events['initialize_static_classroom'].params['wall_names'],
                          cfg.events['initialize_static_classroom'].params['layout'], strict=True):
        size = cfg.scene.entities[name].spec_fn().geom('wall_collision').size[:2]
        distance = ((samples - torch.tensor(pose[:2])).abs() - torch.tensor(size.copy())).clamp_min(0).norm(dim=-1)
        assert distance.min() >= .7 - 1e-5
    # Resetting only env 1 discards its old raw route without touching env 0.
    preserved = env._classroom_route_pts[0].clone()
    _write_classroom_route(env, torch.tensor([1]), raw[1:, :3])
    assert env._classroom_route_n[1] == 3
    assert torch.equal(env._classroom_route_pts[0], preserved)


def test_following_crosses_original_tail_with_continuous_progress_and_no_bonus():
    torch.manual_seed(23)
    command, env = follower(dt=.1)
    env._classroom_route_extension_nodes = 3
    raw = generate_classroom_route(torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0]), count=3)
    _write_classroom_route(env, torch.tensor([0]), raw)
    original_length = env._walk_path_cumlen[0, env._walk_path_n[0] - 1].item()
    env._walk_path_epoch += 1
    command.cfg.path_wall_ahead = 0.
    command._update_command()
    epoch = env._walk_path_epoch.clone()
    last_s = 0.
    for index in range(math.ceil((original_length + 4.) / .03)):
        point, yaw = walk_path_query(env, torch.tensor([0]), torch.tensor([(index + 1) * .03]))
        advance(command, env, xy=point.tolist(), angles=[-math.degrees(yaw.item())])
        value = reward(env)
        state = classroom_follow_state(command)
        assert torch.isfinite(value).all()
        assert command._path_s.item() == pytest.approx((index + 1) * .03, abs=1e-4)
        assert state.speed.item() == pytest.approx(.3, abs=1e-3)
        assert command._path_s.item() > last_s
        assert 'success' not in state.reward_components
        last_s = command._path_s.item()
    assert last_s > original_length
    assert env._classroom_route_n.item() > 3
    assert torch.equal(env._walk_path_epoch, epoch)
    before_frontier, before_age = state.s_frontier.clone(), state.age.clone()
    # Force another extension while stationary. Geometry updates pay no reward.
    command.cfg.path_wall_ahead = 40.
    advance(command, env)
    reward(env)
    state = classroom_follow_state(command)
    assert state.speed.item() == pytest.approx(0., abs=1e-6)
    assert state.turn_delta.item() == pytest.approx(0., abs=1e-6)
    assert state.reward_components['forward'].item() == 0.
    assert torch.equal(state.s_frontier, before_frontier)
    assert state.age.item() == pytest.approx(before_age.item() + .1)
