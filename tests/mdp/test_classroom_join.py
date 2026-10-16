"""Off-path recovery, windows, gait tolerance and reward-abuse regressions."""
import math

import pytest
import torch

from src.tasks.amp_loco.mdp.path_navigation import (
    classroom_follow_state, classroom_navigation_observation, classroom_failure_cost,
)
from tests.mdp.test_classroom_follow import follower, advance, reward


def off_path(reverse=False, distance=.5):
    angle = math.degrees(math.atan2(distance, .8)) + (180. if reverse else 0.)
    command, env = follower((angle,))
    command.robot.data.root_link_pos_w[0, 1] = distance
    env._walk_path_epoch += 1
    command._update_command()
    return command, env


@pytest.mark.parametrize('reverse', [False, True])
def test_off_path_spawn_selects_stable_forward_join_target(reverse):
    command, env = off_path(reverse)
    state = classroom_follow_state(command)
    assert state.joining.item() and state.aligning.item() is reverse
    target = command.goal_pos_w.clone()
    assert target[0].tolist() == pytest.approx([.8, 0.])
    advance(command, env, xy=[[.08, .45]])
    value = reward(env)
    assert torch.equal(command.goal_pos_w, target)
    parts = classroom_follow_state(command).reward_components
    assert parts['join'].item() == pytest.approx(0. if reverse else .2, abs=1e-6)
    assert (value.item() <= 0) if reverse else (value.item() > 0)
    advance(command, env, xy=[[.08, .55]])
    reward(env)
    assert classroom_follow_state(command).reward_components['join'].item() < 0


def test_join_point_produces_distance_improvement_along_command_direction():
    command, env = off_path()
    state = classroom_follow_state(command)
    before = state.distance.clone()
    heading = command.robot.data.heading_w
    xy = state.xy + .1 * torch.stack((heading.cos(), heading.sin()), -1)
    advance(command, env, xy=xy.tolist())
    state = classroom_follow_state(command)
    assert state.distance < before and state.join_delta > 0
    assert state.target_distance > 0 and state.target_reachable


def test_join_follow_hysteresis_and_no_arrival_deadlock():
    command, env = off_path(distance=.21)
    # Place just below the parent's arrival radius at the frozen merge point:
    # navigation must continue until d_in, rather than stopping at d=.21.
    target_x = classroom_follow_state(command).join_target[0, 0].item()
    advance(command, env, xy=[[target_x, .21]], angles=[90.])
    command._update_command()
    assert classroom_follow_state(command).joining
    assert command.command[0, 0] > 0
    for y, joining in [(.19, False), (.3, False), (.41, True), (.3, True), (.19, False)]:
        advance(command, env, xy=[[target_x, y]])
        state = classroom_follow_state(command)
        assert state.joining.item() is joining


def test_observation_features_are_scaled_finite_and_actor_critic_share_group():
    command, env = off_path()
    features = classroom_navigation_observation(env)
    assert features.shape == (1, 14) and torch.isfinite(features).all()
    assert features[0, 0].item() == pytest.approx(1., abs=1e-6)
    assert features[0, 2].item() == pytest.approx(1.)
    assert features[0, 3].item() == pytest.approx(1.)
    assert features[0, 6].item() == 1. and features[0, 7].item() == 0.
    from src.tasks.amp_loco.config.g1.goto_rl_cfg import g1_amp_dodge_mimickit_classroom_ppo_runner_cfg
    runner = g1_amp_dodge_mimickit_classroom_ppo_runner_cfg()
    assert 'path_navigation' in runner.obs_groups['actor']
    assert 'path_navigation' in runner.obs_groups['critic']
    assert runner.actor.obs_normalization and runner.critic.obs_normalization


def test_normal_turns_and_command_ramp_have_no_premature_stall():
    command, env = follower((180.,))
    command.cfg.command_acceleration = .8
    for step in range(120):
        angle = 180 - math.degrees((step + 1) * .5 * env.step_dt)
        advance(command, env, angles=[angle])
        reward(env)
        state = classroom_follow_state(command)
        assert not state.turn_stall and not state.follow_stall
        assert state.command_speed == 0
    command, env = follower()
    command.cfg.command_acceleration = .8
    command._classroom_follow_state.command_speed.zero_()
    previous = 0.
    for _ in range(35):
        advance(command, env)
        state = classroom_follow_state(command)
        assert 0 <= state.command_speed - previous <= .8 * .02 + 1e-6
        previous = state.command_speed.item()
        assert not state.follow_stall


def test_com_velocity_ignores_link_rotation_and_filter_allows_gait_sway():
    command, env = follower()
    command.cfg.velocity_filter_time = .15
    command.robot.data.root_link_lin_vel_b[0, 1] = 2.  # rotating offset point
    advance(command, env, angles=[20.])
    reward(env)
    assert classroom_follow_state(command).reward_components['lateral'] == 0
    command, env = follower()
    command.cfg.velocity_filter_time = .15
    for step in range(200):
        lateral = .3 * math.sin(2 * math.pi * 2 * step * env.step_dt)
        advance(command, env, velocity=[[0., lateral, 0.]])
        reward(env)
        assert classroom_follow_state(command).reward_components['lateral'] == 0
    for _ in range(30):
        advance(command, env, velocity=[[0., .45, 0.]])
        reward(env)
    assert classroom_follow_state(command).reward_components['lateral'] < 0


def configured_reward_params(configured=True):
    if not configured:
        return {}
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
    return g1_amp_dodge_mimickit_classroom_flat_env_cfg().rewards['classroom_path_follow'].params.copy()


def test_configured_navigation_keeps_correct_gains_and_halves_wrong_motion():
    params = configured_reward_params()
    command, env = follower((120.,))
    advance(command, env, angles=[110.])
    reward(env, **params)
    assert classroom_follow_state(command).reward_components['turn'].item() == pytest.approx(6 * 10 / 180)
    advance(command, env, angles=[120.])
    reward(env, **params)
    assert classroom_follow_state(command).reward_components['turn'].item() == pytest.approx(-3 * 10 / 180)

    command, env = off_path()
    advance(command, env, xy=[[.08, .45]])
    reward(env, **params)
    assert classroom_follow_state(command).reward_components['join'].item() == pytest.approx(.4, abs=1e-6)
    advance(command, env, xy=[[.08, .5]])
    reward(env, **params)
    assert classroom_follow_state(command).reward_components['join'].item() == pytest.approx(-.2, abs=1e-6)

    command, env = follower()
    advance(command, env, xy=[[.012, 0.]])
    assert reward(env, **params).item() == pytest.approx(.12)
    advance(command, env, xy=[[0., 0.]])
    assert reward(env, **params).item() == pytest.approx(-.06)


@pytest.mark.parametrize('mode', ['forward', 'retreat', 'turn', 'join_in', 'join_out', 'stall'])
def test_progress_discount_changes_only_negative_progress_components(mode):
    params = configured_reward_params()
    command, env = off_path() if mode.startswith('join_') else follower((120.,) if mode == 'turn' else (0.,))
    if mode == 'stall':
        for _ in range(60):
            advance(command, env)
            reward(env, **params)
    if mode == 'forward':
        advance(command, env, xy=[[.012, .1]], velocity=[[0., .6, 0.]])
    elif mode == 'retreat':
        advance(command, env, xy=[[-.012, .1]], velocity=[[-.6, .6, 0.]])
    elif mode == 'turn':
        advance(command, env, angles=[130.], velocity=[[0., .6, 0.]])
    elif mode.startswith('join_'):
        advance(command, env, xy=[[.08, .45 if mode == 'join_in' else .55]])
    else:
        advance(command, env)
    reward(env, **dict(params, progress_penalty_scale=1.))
    original = {name: value.clone() for name, value in classroom_follow_state(command).reward_components.items()}
    reward(env, **params)
    for name, value in classroom_follow_state(command).reward_components.items():
        expected = original[name]
        if name in ('turn', 'join', 'retreat'):
            expected = torch.where(expected < 0, expected * .5, expected)
        assert torch.allclose(value, expected), name


def test_progress_penalty_scale_can_restore_original_costs(monkeypatch):
    monkeypatch.setenv('WALK_PROGRESS_PENALTY_SCALE', '1.0')
    params = configured_reward_params()
    command, env = follower((120.,))
    advance(command, env, angles=[130.])
    assert reward(env, **params).item() == pytest.approx(-6 * 10 / 180)


@pytest.mark.parametrize('configured', [False, True])
def test_phase_cycling_keeps_watchdog_and_reflects_join_penalty_discount(configured):
    command, env = follower()
    params = configured_reward_params(configured)
    params['k_theta'] = 0.
    total = 0.
    for cycle in range(30):
        for y in (.5, .19):
            advance(command, env, xy=[[0., y]], angles=[0.])
            reward(env, **params)
            total += classroom_follow_state(command).reward_components['join'].item()
    state = classroom_follow_state(command)
    assert state.age >= 1. and state.recovery_age >= 1.
    # Closings repay self-caused openings even with discounted penalties;
    # phase changes cannot restart grace or the recovery watchdog.
    q = params.get('progress_penalty_scale', 1.)
    k_join = params.get('k_join', 2.)
    assert total < 0.
    assert total == pytest.approx(-30 * q * k_join * .3 / command.cfg.navigation_d_ref, abs=1e-4)


@pytest.mark.parametrize('configured', [False, True])
def test_fast_retreat_and_repeated_advance_cannot_farm_same_arc_length(configured):
    command, env = follower()
    params = configured_reward_params(configured)
    total = 0.
    for _ in range(8):
        for step in range(30):
            advance(command, env, xy=[[(step + 1) * .006, 0.]])
            total += reward(env, **params).item()
        advance(command, env, xy=[[0., 0.]])  # faster than capped forward speed
        total += reward(env, **params).item()
    assert total <= 1e-5


@pytest.mark.parametrize('configured', [False, True])
def test_repeated_turns_and_phase_switches_eventually_pay_stall(configured):
    command, env = follower((80.,))
    params = configured_reward_params(configured)
    turn_sum = 0.
    for step in range(600):
        advance(command, env, angles=[24. if step % 2 == 0 else 80.])
        reward(env, **params)
        turn_sum += classroom_follow_state(command).reward_components['turn'].item()
    state = classroom_follow_state(command)
    q = params.get('progress_penalty_scale', 1.)
    k_theta = params.get('k_theta', 2.)
    expected_turn_sum = k_theta * 56 / 180 * (1 - 300 * q)
    assert turn_sum < 0.
    assert turn_sum == pytest.approx(expected_turn_sum, abs=1e-4)
    assert state.recovery_stall.item()
    assert state.reward_components['recovery_stall'].item() < 0


def test_projection_search_stays_on_current_visit_and_no_motion_has_no_credit():
    command, env = follower(points=((0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.), (5., 0.)))
    advance(command, env, xy=[[1., 0.]])
    reward(env)
    assert command._path_seg_idx == 0 and command._path_s == 1.
    # Exercise an internal reference correction without physically moving.
    command._path_s[:] = .5
    advance(command, env)
    reward(env)
    state = classroom_follow_state(command)
    assert state.speed == 0 and state.reward_components['forward'] == 0


def test_endpoint_proximity_keeps_navigation_active():
    command, env = follower(points=((0., 0.), (1., 0.)))
    advance(command, env, xy=[[.95, .21]])  # inside goal radius, outside d_in
    state = classroom_follow_state(command)
    assert state.aligning  # Turn toward the route instead of declaring arrival.
    from src.tasks.amp_loco.mdp.path_navigation import classroom_navigation_update
    assert not classroom_navigation_update(env).item()


def test_failure_budget_is_dt_independent_and_simultaneous_ball_hit_does_not_hide_fall():
    from types import SimpleNamespace as NS
    for dt in (.01, .02, .1):
        terms = {'ball_hit': torch.tensor([True, True]), 'bad_base_height': torch.tensor([False, True])}
        env = NS(num_envs=2, device='cpu', step_dt=dt, max_episode_length_s=25.,
            episode_length_buf=torch.zeros(2),
            termination_manager=NS(active_terms=list(terms), get_term=lambda key: terms[key]))
        cost = dt * classroom_failure_cost(env, base_cost=10., remaining_rate=12.4)
        assert cost.tolist() == pytest.approx([0., 320.])


def test_blocked_local_connector_is_reported_without_walking_through_furniture():
    command, env = off_path()
    from types import SimpleNamespace as NS
    # A large box lies across every connector to the currently valid segment.
    command.cfg.navigation_obstacles = ('box',)
    env.sim.data = NS(geom_xpos=torch.tensor([[[.4, .2, .5]]]),
                      geom_xmat=torch.eye(3).reshape(1, 1, 9))
    env.sim.model.geom_size[:] = torch.tensor([.6, .1, .5])
    env._walk_path_epoch += 1
    command._update_command()
    state = classroom_follow_state(command)
    assert not state.target_reachable
    assert state.command_speed == 0 and command.command[0, 0] == 0


def test_sharp_arc_respects_yaw_rate_and_linear_speed_limits():
    from src.tasks.amp_loco.mdp.classroom import round_classroom_route
    from src.tasks.amp_loco.mdp.walk_path import walk_path_query
    route, count = round_classroom_route(torch.tensor([[[0., 0.], [5., 0.], [5., 4.]]]), radius=.2)
    command, env = follower(points=route[0, :count[0]].tolist())
    # Keep the yaw-rate cap active at 0.8 rad/s rather than curvature slowdown.
    # Pin the cap explicitly: the scenario geometry assumes 0.8, not the task default.
    command.cfg.max_ang_vel_z = .8
    command.cfg.curvature_alpha = .3
    point, yaw = walk_path_query(env, torch.tensor([0]), torch.tensor([4.8 + math.pi * .2 / 4]))
    advance(command, env, xy=point.tolist(), angles=[-math.degrees(yaw.item())])
    state = classroom_follow_state(command)
    assert state.curvature.abs().item() > 4.9
    nominal = .6 * state.theta.cos().clamp_min(0) / (1 + .3 * state.curvature.abs())
    assert nominal > command.cfg.max_ang_vel_z / state.curvature.abs()
    assert state.v_cmd * state.curvature.abs() <= command.cfg.max_ang_vel_z + 1e-6
    command.cfg.max_lin_vel_x = .05
    advance(command, env)
    assert classroom_follow_state(command).v_cmd <= .05


def test_join_threshold_overshoot_cannot_earn_credit_inside_gait_band():
    command, env = off_path(distance=.21)
    advance(command, env, xy=[[0., 0.]])
    reward(env)
    state = classroom_follow_state(command)
    # Only the final .01 m outside d_in is credited, not the .20 m tolerance band.
    assert state.reward_components['join'].item() == pytest.approx(.04, abs=1e-6)


def test_gait_can_start_after_long_alignment_without_immediate_follow_stall():
    command, env = follower((120.,))
    for step in range(100):
        advance(command, env, angles=[120. - math.degrees((step + 1) * .5 * env.step_dt)])
        reward(env)
    advance(command, env, angles=[24.])
    reward(env)
    initial_age = classroom_follow_state(command).age.clone()
    for _ in range(14):  # first .28 seconds after the transition
        advance(command, env, angles=[24.])
        reward(env)
        state = classroom_follow_state(command)
        assert not state.follow_stall
        assert state.age > initial_age
    advance(command, env, angles=[24.])
    reward(env)
    assert classroom_follow_state(command).follow_stall
