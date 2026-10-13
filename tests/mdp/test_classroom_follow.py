"""Numerical and manager-order regressions for the classroom follower."""
import math
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.classroom import (
    classroom_follow_state, classroom_path_follow_reward, classroom_navigation_update,
    generate_classroom_route, round_classroom_route,
)
from src.tasks.amp_loco.mdp.rewards import is_terminated_except
from src.tasks.amp_loco.mdp.walk_path import walk_path_query
from tests.mdp.test_classroom_heading_command import make_command


def follower(angles=(0.,), points=((0., 0.), (10., 0.)), dt=.02, xy=None):
    command, env = make_command(angles)
    env.step_dt = dt
    command.cfg.path_follow = True
    route = torch.tensor(points, dtype=torch.float32)[None].repeat(len(angles), 1, 1)
    env._walk_path_pts = route
    env._walk_path_cumlen = torch.cat((torch.zeros(len(angles), 1),
        (route[:, 1:] - route[:, :-1]).norm(dim=-1).cumsum(-1)), -1)
    env._walk_path_n = torch.full((len(angles),), len(points), dtype=torch.long)
    env._walk_path_epoch = torch.ones(len(angles), dtype=torch.long)
    command.robot.data.root_link_lin_vel_b = torch.zeros(len(angles), 3)
    command.cfg.cbf_enabled = False
    if xy is not None:
        command.robot.data.root_link_pos_w[:, :2] = torch.tensor(xy)
    command._update_command()  # reference at the start of the first physics step
    return command, env


def advance(command, env, xy=None, angles=None, velocity=None):
    env.common_step_counter += 1
    if xy is not None:
        command.robot.data.root_link_pos_w[:, :2] = torch.tensor(xy)
    if angles is not None:
        # Angle is heading error to world +x, as in make_command.
        yaw = -torch.deg2rad(torch.tensor(angles, dtype=torch.float32))
        command.robot.data.heading_w[:] = yaw
        command.robot.data.root_link_quat_w[:, 0] = (yaw / 2).cos()
        command.robot.data.root_link_quat_w[:, 3] = (yaw / 2).sin()
    if velocity is not None:
        command.robot.data.root_link_lin_vel_b[:] = torch.tensor(velocity)
        body = torch.tensor(velocity)
        yaw = command.robot.data.heading_w
        command.robot.data.root_com_lin_vel_w[:] = torch.stack((
            body[:, 0] * yaw.cos() - body[:, 1] * yaw.sin(),
            body[:, 0] * yaw.sin() + body[:, 1] * yaw.cos(), body[:, 2]), -1)


def reward(env, **params):
    # Actual per-step task reward, including the framework's dt scaling.
    return env.step_dt * classroom_path_follow_reward(env, **params)


@pytest.mark.parametrize("angle,expected", [(0., 1.), (45., .5), (75., math.cos(math.radians(75)) ** 2),
                                            (90., 0.), (120., 0.), (180., 0.)])
def test_forward_alignment_coefficient_and_threat_freezes_retreat(angle, expected):
    command, env = follower((angle,))
    advance(command, env, xy=[[.006, 0.]])  # .3 m/s: half reference speed
    assert reward(env).item() == pytest.approx(2 * expected * .5 * .02, abs=1e-6)
    command._dodge_threat[:] = True
    advance(command, env, xy=[[0., 0.]])
    assert reward(env).item() == 0.
    command._dodge_threat[:] = False
    advance(command, env, xy=[[-.006, 0.]])
    assert reward(env, deviation_weight=0.).item() == pytest.approx(-.02, abs=1e-6)


@pytest.mark.parametrize('mode', ['forward', 'retreat', 'turn', 'join', 'stall'])
def test_whole_navigation_reward_and_component_metrics_freeze_per_threat_env(mode):
    command, env = follower((120., 120.) if mode == 'turn' else (0., 0.))
    if mode == 'join':
        advance(command, env, xy=[[0., .5]] * 2)
        reward(env)
    if mode == 'stall':
        for _ in range(60):
            advance(command, env)
            reward(env)
    command._dodge_threat[:] = torch.tensor([True, False])
    if mode == 'forward':
        advance(command, env, xy=[[.012, 0.]] * 2)
    elif mode == 'retreat':
        advance(command, env, xy=[[-.012, .1]] * 2, velocity=[[-.6, .6, 0.]] * 2)
    elif mode == 'turn':
        advance(command, env, angles=[110., 110.])
    elif mode == 'join':
        advance(command, env, xy=[[0., .45]] * 2)
    else:
        advance(command, env)
    value = reward(env)
    assert value[0] == 0. and value[1].abs() > 0.
    parts = classroom_follow_state(command).reward_components
    assert all(component[0] == 0. for component in parts.values())
    assert any(component[1] != 0. for component in parts.values())


@pytest.mark.parametrize('mode', ['forward', 'turn', 'join'])
def test_threat_motion_is_not_paid_after_navigation_reward_resumes(mode):
    command, env = follower((120.,) if mode == 'turn' else (0.,),
                            xy=[0., .5] if mode == 'join' else None)
    command._dodge_threat[:] = True
    for step in range(2):
        if mode == 'forward':
            advance(command, env, xy=[[(step + 1) * .2, 0.]])
        elif mode == 'turn':
            advance(command, env, angles=[120. - (step + 1) * 20.])
        else:
            advance(command, env, xy=[[0., .5 - (step + 1) * .05]])
        assert reward(env, deviation_weight=0.).item() == 0.
    command._dodge_threat[:] = False
    advance(command, env)
    assert reward(env, deviation_weight=0.).item() == 0.
    if mode == 'forward':
        advance(command, env, xy=[[.412, 0.]])
    elif mode == 'turn':
        advance(command, env, angles=[70.])
    else:
        advance(command, env, xy=[[0., .35]])
    assert reward(env, deviation_weight=0.).item() > 0.


def test_hidden_outside_ball_cannot_freeze_navigation_with_a_stale_threat_flag():
    from tests.mdp.test_classroom_ball_region import make_env
    command, env = follower((0., 0.))
    ball_env = make_env([[3., 0., 1.], [0., 1000., 1.]])
    env._classroom_ball_bounds = ball_env._classroom_ball_bounds
    env.scene['ball'] = ball_env.scene['ball']
    env.sim.data = ball_env.sim.data
    command._dodge_threat[:] = True
    advance(command, env, xy=[[.012, 0.]] * 2)
    assert reward(env).tolist() == pytest.approx([0., .04], abs=1e-6)


@pytest.mark.parametrize("dt", [.01, .02, .05])
def test_turn_difference_is_signed_and_not_dt_scaled(dt):
    command, env = follower((120.,), dt=dt)
    advance(command, env, angles=[110.])
    assert reward(env).item() == pytest.approx(2 * 10 / 180, abs=1e-6)
    advance(command, env, angles=[120.])
    assert reward(env).item() == pytest.approx(-2 * 10 / 180, abs=1e-6)


def test_turn_uses_one_world_bearing_and_wraps_pi():
    command, env = follower((120.,))
    # Translation changes the new lookahead bearing, but without rotating
    # there must be exactly zero turn difference.
    advance(command, env, xy=[[0., .2]])
    state = classroom_follow_state(command)
    assert state.theta.item() != pytest.approx(state.step_theta.item())
    assert state.turn_delta.item() == 0.
    command, env = follower((-179.,), points=((0., 0.), (-10., -.17455)))
    advance(command, env, angles=[179.])
    state = classroom_follow_state(command)
    assert abs(state.turn_delta.item()) <= math.radians(2.001)


def test_hysteresis_keeps_phase_and_pays_exit_step_turn():
    command, env = follower((75.,))
    assert not classroom_follow_state(command).aligning.item()
    for angle, aligning in [(76., True), (60., True), (25., True), (24., False), (60., False)]:
        advance(command, env, angles=[angle])
        command._update_command()
        state = classroom_follow_state(command)
        assert state.aligning.item() is aligning
        assert (command.command[0, 0] == 0).item() is aligning
        if angle == 24.:
            assert state.step_aligning.item() and reward(env).item() > 0


@pytest.mark.parametrize("dt", [.01, .02, .1])
def test_stalls_have_seconds_grace_and_no_static_positive_reward(dt):
    command, env = follower((120., 0.), dt=dt)
    for _ in range(round(1 / dt) - 1):
        advance(command, env)
        assert torch.equal(reward(env), torch.zeros(2))
    advance(command, env)
    assert torch.allclose(reward(env), torch.full((2,), -1.2 * dt), atol=1e-6)
    # Reverse then forward turning doesn't repeatedly reset the stall clock.
    advance(command, env, angles=[121., 0.])
    assert reward(env)[0] < -1.2 * dt
    advance(command, env, angles=[120., 0.])
    assert classroom_follow_state(command).turn_stall[0] >= 1.


def test_curve_speed_and_slow_following_avoid_false_stall():
    rounded, counts = round_classroom_route(torch.tensor([[[0., 0.], [5., 0.], [5., 4.]]]))
    command, env = follower(points=rounded[0, :counts[0]].tolist())
    station = torch.tensor([4.6])
    for _ in range(70):
        point, yaw = walk_path_query(env, torch.tensor([0]), station)
        advance(command, env, xy=point.tolist(), angles=[-math.degrees(yaw.item())])
        command._update_command()
        state = classroom_follow_state(command)
        assert state.follow_stall.item() < command.cfg.stall_duration
        if station.item() < 5.1:
            assert state.curvature.abs().item() > 1.
            expected = .6 * state.theta.cos().clamp_min(0) / (1 + .6 * state.curvature.abs())
            assert torch.allclose(command.command[:, 0], expected, atol=1e-6)
        station += .003  # .15 m/s, enough at this curve's reduced target


def test_body_backward_lateral_and_deviation_are_costs():
    command, env = follower((0., 120.))
    advance(command, env, velocity=[[-.3, .45, 0.], [0., .45, 0.]])
    value = reward(env, backward_limit=0., lateral_limit=.15)
    assert value[0].item() == pytest.approx(-(.25 + .5 * .5 ** 2) * .02)
    assert value[1].item() == pytest.approx(-.5 * .5 ** 2 * .2 * .02)
    advance(command, env, xy=[[0., .5], [0., .5]], velocity=[[0., 0., 0.]] * 2)
    assert reward(env, k_theta=0., k_join=0.).tolist() == pytest.approx([-.01, -.01])


@pytest.mark.parametrize('route_version', [None, 1])
def test_reward_and_command_reads_are_idempotent_and_resets_are_per_env(route_version):
    command, env = follower((0., 120.))
    if route_version is not None:
        env._classroom_route_version = route_version
    advance(command, env, xy=[[.012, 0.], [0., 0.]], angles=[0., 110.])
    first = reward(env).clone()
    command._update_command()
    assert torch.equal(reward(env), first)
    env._walk_path_epoch[1] += 1
    if route_version is not None:
        env._classroom_route_version += 1
    command.robot.data.root_link_pos_w[1, 0] = 10.
    command.robot.data.heading_w[1] = 0.
    env._walk_path_pts[1] += torch.tensor([10., 0.])
    command._update_command()
    state = classroom_follow_state(command)
    assert reward(env)[0].item() == first[0].item()
    assert command._path_s_delta[1] == 0 and state.turn_delta[1] == 0
    assert state.turn_stall[1] == 0 and state.follow_stall[1] == 0


def test_route_end_does_not_terminate_stop_or_pay_an_endpoint_bonus():
    command, env = follower(points=((0., 0.), (1., 0.)))
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    env.fail = torch.zeros(1, dtype=torch.bool)
    terms = {
        "collision": TerminationTermCfg(func=lambda env: env.fail),
        "navigation_update": cfg.terminations["navigation_update"],
    }
    env.termination_manager = TerminationManager(terms, env)
    manager = RewardManager({"follow": cfg.rewards["classroom_path_follow"]}, env, scale_by_dt=True)
    advance(command, env, xy=[[.9, 0.]])
    assert not env.termination_manager.compute().item()
    assert manager.compute(dt=.02).item() == pytest.approx(.12)
    assert classroom_follow_state(command).command_speed.item() > 0
    env.fail[:] = True
    advance(command, env)
    assert env.termination_manager.compute().item()
    assert not env.termination_manager.get_term("navigation_update").item()
    assert manager.compute(dt=.02).item() == 0
    assert is_terminated_except(env).item() == 1


def test_same_endpoint_on_earlier_lap_never_terminates():
    command, env = follower(points=((0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.)))
    assert not classroom_navigation_update(env).item()
    advance(command, env, xy=[[0., .1]])
    assert not classroom_navigation_update(env).item()


def test_continuous_terms_scale_with_dt_and_round_trip_cannot_pay():
    totals = []
    for dt in (.01, .02, .1):
        command, env = follower(dt=dt)
        total = 0.
        for step in range(round(1 / dt)):
            advance(command, env, xy=[[(step + 1) * .3 * dt, 0.]])
            total += reward(env).item()
        totals.append(total)
        # Same rate retracing retains every negative step.
        for step in range(round(1 / dt)):
            advance(command, env, xy=[[.3 - (step + 1) * .3 * dt, 0.]])
            total += reward(env).item()
        assert total <= 1e-5
    assert totals == pytest.approx([1., 1., 1.], abs=1e-5)


def test_classroom_config_and_short_routes_leave_wallwalk_unchanged():
    for play in (False, True):
        cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=play)
        assert cfg.scale_rewards_by_dt
        assert cfg.events['reset_walk_path'].params['route_nodes'] == 4
        assert 'path_success' not in cfg.terminations
        assert 'navigation_update' in cfg.terminations
        assert cfg.rewards['classroom_safe_completion'].params['bonus'] == 10.
        assert 'success_bonus' not in cfg.rewards['classroom_path_follow'].params
        assert cfg.rewards['is_terminated'].func.__name__ == 'classroom_failure_cost'
        for name in ('walk_goal_heading_when_safe', 'walk_path_progress_when_safe', 'walk_path_adherence'):
            assert name not in cfg.rewards
        for name in ('track_anchor_linear_velocity', 'track_anchor_angular_velocity'):
            if name in cfg.rewards:
                assert cfg.rewards[name].params['as_cost']
    wall = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg()
    assert not wall.commands['twist'].classroom_follow
    assert 'path_success' not in wall.terminations
    assert wall.rewards['walk_path_progress_when_safe'].params == {'command_name': 'twist'}
    torch.manual_seed(5)
    from src.assets.objects.classroom import CLASSROOM_ROUTE_STATIONS
    x = torch.rand(128) * CLASSROOM_ROUTE_STATIONS[-1]
    lane = torch.randint(2, (128,))
    from src.assets.objects.classroom import CLASSROOM_SPAWN_LANES
    y = torch.tensor(CLASSROOM_SPAWN_LANES)[lane]
    route, counts = round_classroom_route(generate_classroom_route(x, y, lane, count=4))
    lengths = (route[:, 1:] - route[:, :-1]).norm(dim=-1).sum(-1)
    assert (counts <= 16).all() and (lengths < 15.).all()


def test_invalid_hysteresis_and_reward_weights_are_rejected():
    command, env = follower()
    with pytest.raises(ValueError, match='k_b >= k_p'):
        reward(env, k_b=1.)
    from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommand
    command.cfg.align_exit_angle = command.cfg.align_enter_angle
    with pytest.raises(ValueError, match='alignment angles'):
        DodgeGoToGoalCommand(command.cfg, env)


def test_tracking_cost_is_neutral_at_rest_and_negative_for_error():
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from src.tasks.amp_loco.mdp.rewards import (
        track_anchor_linear_velocity, track_anchor_angular_velocity,
    )
    command, env = follower((0., 0.))
    data = command.robot.data
    data.body_link_quat_w = data.root_link_quat_w[:, None]
    data.body_link_lin_vel_w = torch.tensor([[[0., 0., 0.]], [[.3, 0., 0.]]])
    data.body_link_ang_vel_w = torch.tensor([[[0., 0., 0.]], [[0., 0., .3]]])
    env.command_manager.get_command = lambda _: torch.zeros(2, 3)
    anchor = SceneEntityCfg('robot')
    anchor.body_ids = [0]
    for func in (track_anchor_linear_velocity, track_anchor_angular_velocity):
        params = dict(std=.3, command_name='twist', anchor_cfg=anchor)
        assert torch.allclose(func(env, **params), torch.tensor([1., math.exp(-1)]))
        assert torch.allclose(func(env, **params, as_cost=True), torch.tensor([0., math.exp(-1) - 1]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs GPU sim')
def test_gpu_safe_timeout_reward_and_partial_auto_reset():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda')
    try:
        env.reset()
        command = env.command_manager.get_term('twist')
        epoch = env._walk_path_epoch.clone()
        env.episode_length_buf[0] = env.max_episode_length - 1
        _, value, terminated, truncated, _ = env.step(torch.zeros(2, env.action_manager.total_action_dim, device='cuda'))
        assert not terminated[0] and truncated[0]
        assert value[0] > 5.  # one-shot +10 survives the other task costs
        terms = dict(env.reward_manager.get_active_iterable_terms(0))
        assert terms['classroom_safe_completion'][0] * env.step_dt == pytest.approx(10.)
        assert env._walk_path_epoch[0] == epoch[0] + 1
        assert env._walk_path_epoch[1] == epoch[1]
        state = classroom_follow_state(command)
        assert state.turn_delta[0] == 0 and command._path_s_delta[0] == 0
        # The next physics step must not pay the previous episode's bonus again.
        env.step(torch.zeros(2, env.action_manager.total_action_dim, device='cuda'))
        terms = dict(env.reward_manager.get_active_iterable_terms(0))
        assert terms['classroom_safe_completion'][0] == 0.
    finally:
        env.close()
