"""Room bounds must isolate hidden balls from robot observations and signals."""
from types import SimpleNamespace as NS

import pytest
import torch

from src.assets.objects.classroom import CLASSROOM_BALL_BOUNDS, CLASSROOM_BALL_PARK_POS
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
)
from src.tasks.amp_loco.mdp.ball_region import (
    ball_in_play, hide_outside_classroom_ball, set_ball_throw_target,
)
from src.tasks.amp_loco.mdp.events import reset_dodge_state, throw_ball_on_dwell
from src.tasks.amp_loco.mdp.observations import dodge_ball_state_b
from src.tasks.amp_loco.mdp.rewards import airborne, mimickit_dodge_reward
from src.tasks.amp_loco.mdp.terminations import ball_contact


def make_env(positions, origins=None):
    pos = torch.tensor(positions, dtype=torch.float32)
    n = len(pos)
    origins = torch.zeros_like(pos) if origins is None else torch.tensor(origins, dtype=pos.dtype)
    qpos = torch.zeros(n, 7)
    qpos[:, :3], qpos[:, 3] = pos + origins, 1.
    qvel = torch.ones(n, 6)
    # Deliberately stale derived state to exercise event ordering after a launch.
    ball = NS(data=NS(root_link_pos_w=pos + origins, root_link_lin_vel_w=torch.ones(n, 3),
                     indexing=NS(free_joint_q_adr=torch.arange(7), free_joint_v_adr=torch.arange(6))))
    ball.write_root_link_pose_to_sim = lambda pose, env_ids: qpos.__setitem__(env_ids, pose)
    ball.write_root_link_velocity_to_sim = lambda vel, env_ids: qvel.__setitem__(env_ids, vel)
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.
    robot = NS(data=NS(root_link_pos_w=origins + torch.tensor([2., 0., 1.]),
                      root_link_lin_vel_w=torch.zeros(n, 3), root_link_quat_w=quat))
    class Scene(dict):
        env_origins = origins
    scene = Scene(ball=ball, robot=robot, ball_robot_contact=NS(data=NS(found=torch.ones(n, 1))))
    command = NS(_dodge_threat=torch.ones(n, dtype=torch.bool), _dodge_h=-torch.ones(n),
                 _dodge_tti=torch.ones(n),
                 metrics={'cbf_active_frac': torch.ones(n), 'cbf_min_h': -torch.ones(n)})
    env = NS(num_envs=n, device='cpu', scene=scene,
             sim=NS(data=NS(qpos=qpos, qvel=qvel),
                    model=NS(geom_size=torch.full((n, 1, 3), .1))),
             _classroom_ball_bounds=torch.tensor(CLASSROOM_BALL_BOUNDS),
             _classroom_ball_park_pos=torch.tensor(CLASSROOM_BALL_PARK_POS),
             _airborne_ball_geom_id=0,
             command_manager=NS(get_term=lambda _: command))
    return env


def test_room_faces_are_inclusive_and_all_six_exterior_directions_are_invalid():
    lo, hi = torch.tensor(CLASSROOM_BALL_BOUNDS)
    center = (lo + hi) / 2
    positions = [center.tolist()]
    expected = [True]
    for axis in range(3):
        for face, sign in [(lo, -1), (hi, 1)]:
            on_face = center.clone()
            on_face[axis] = face[axis]
            outside = on_face.clone()
            outside[axis] += sign * .01
            positions.extend([on_face.tolist(), outside.tolist()])
            expected.extend([True, False])
    positions.append([float('nan'), 0., 1.])
    expected.append(False)
    env = make_env(positions, origins=[[30., -20., 5.]] * len(positions))
    assert ball_in_play(env).tolist() == expected


def test_hidden_ball_has_no_observation_reward_threat_or_hit_signal():
    env = make_env([[3., 0., 1.], [0., 1000., 1.], [3., 0., -1.]])
    obs = dodge_ball_state_b(env)
    assert obs[0].abs().sum() > 0
    assert torch.equal(obs[1:], torch.zeros(2, 6))
    reward = mimickit_dodge_reward(env)
    assert reward[0] > 0 and torch.equal(reward[1:], torch.zeros(2))
    assert airborne(env, torch.ones(3)).tolist() == [True, False, False]
    # Even a stale positive contact sensor cannot report a hit outside the room.
    assert ball_contact(env, delta_v_threshold=0.).tolist() == [True, False, False]
    hide_outside_classroom_ball(env)
    assert torch.equal(env.sim.data.qpos[1:, :3], env._classroom_ball_park_pos.expand(2, 3))
    assert torch.equal(env.sim.data.qvel[1:], torch.zeros(2, 6))
    command = env.command_manager.get_term('twist')
    assert command._dodge_threat.tolist() == [True, False, False]
    assert torch.equal(command.metrics['cbf_min_h'][1:], torch.zeros(2))
    assert torch.equal(dodge_ball_state_b(env)[1:], torch.zeros(2, 6))


def test_new_in_room_launch_is_visible_without_forward_and_outside_launch_is_hidden():
    env = make_env([CLASSROOM_BALL_PARK_POS, CLASSROOM_BALL_PARK_POS])
    env.sim.data.qpos[:, :3] = torch.tensor([[4., 1., 1.], [0., 1000., 1.]])
    hide_outside_classroom_ball(env)
    assert ball_in_play(env).tolist() == [True, False]
    assert torch.equal(env.sim.data.qpos[0, :3], torch.tensor([4., 1., 1.]))
    assert torch.equal(dodge_ball_state_b(env)[0], torch.tensor([2., 1., 0., 1., 1., 1.]))
    # Repeated hidden steps keep the parking state bounded even under gravity.
    env.sim.data.qpos[1, 2] -= .1
    env.sim.data.qvel[1, 2] = -2.
    hide_outside_classroom_ball(env)
    assert torch.equal(env.sim.data.qpos[1, :3], env._classroom_ball_park_pos)
    assert torch.equal(env.sim.data.qvel[1], torch.zeros(6))


def test_outside_launches_remain_in_play_until_the_saved_target_is_reached():
    starts = torch.tensor([[-4., 0., 1.], [22., 0., 1.], [2., -10., 1.], [2., 8., 1.]])
    env = make_env(starts.tolist(), origins=[[30., -20., 5.]] * 4)
    ids = torch.arange(4)
    start_xy = env.sim.data.qpos[:, :2].clone()
    targets = env.scene.env_origins[:, :2] + torch.tensor([3., 1.])
    set_ball_throw_target(env, ids, start_xy, targets)
    hide_outside_classroom_ball(env)
    assert ball_in_play(env).all()
    assert dodge_ball_state_b(env).abs().sum(dim=1).gt(0).all()
    assert mimickit_dodge_reward(env).gt(0).all()
    assert ball_contact(env, delta_v_threshold=0.).all()
    assert torch.equal(env.sim.data.qpos[:, :2], start_xy)
    assert not env._classroom_ball_target_reached.any()
    # Neither roof nor floor bounds apply during the inbound leg.
    env.sim.data.qpos[:, 2] = env.scene.env_origins[:, 2] + torch.tensor([4., -1., 4., -1.])
    hide_outside_classroom_ball(env)
    assert ball_in_play(env).all()
    # Overshoot the target within a single step, while still outside vertically.
    env.sim.data.qpos[:, :2] = targets + .1 * (targets - start_xy)
    hide_outside_classroom_ball(env)
    assert env._classroom_ball_target_reached.all()
    assert not ball_in_play(env).any()
    assert not dodge_ball_state_b(env).any()
    assert not ball_contact(env, delta_v_threshold=0.).any()
    assert torch.equal(env.sim.data.qpos[:, :3],
                       env.scene.env_origins + env._classroom_ball_park_pos)
    # Parking behind the target must not reopen the gate.
    hide_outside_classroom_ball(env)
    assert not ball_in_play(env).any()
    assert not env.sim.data.qvel.any()


def test_arrival_inside_room_latches_bounds_for_later_exit_and_resets_per_env():
    env = make_env([[6., 0., 1.], [6., 0., 1.]])
    env.step_dt = .02
    ids = torch.arange(2)
    targets = torch.tensor([[3., 0.], [3., 0.]])
    set_ball_throw_target(env, ids, env.sim.data.qpos[:, :2], targets)
    env.sim.data.qpos[0, :2] = targets[0]  # exact arrival; the other ball is still inbound
    hide_outside_classroom_ball(env, torch.tensor([0]))
    assert env._classroom_ball_target_reached.tolist() == [True, False]
    assert ball_in_play(env).all()
    env.sim.data.qpos[:, 2] = -1.
    hide_outside_classroom_ball(env)
    assert ball_in_play(env).tolist() == [False, True]
    reset_dodge_state(env, torch.tensor([1]), park_offset=CLASSROOM_BALL_PARK_POS)
    assert env._classroom_ball_target_reached.all()
    assert not ball_in_play(env).any()
    # A new throw clears arrival only for the selected world.
    set_ball_throw_target(env, torch.tensor([0]), torch.tensor([[22., 0.]]), targets[:1])
    env.sim.data.qpos[0, :3] = torch.tensor([22., 0., 1.])
    hide_outside_classroom_ball(env)
    assert env._classroom_ball_target_reached.tolist() == [False, True]
    assert ball_in_play(env).tolist() == [True, False]


@pytest.mark.parametrize('launch_speed_range', [None, (10., 10.)])
def test_throw_saves_final_led_and_noisy_target_in_both_trajectory_modes(monkeypatch, launch_speed_range):
    env = make_env([CLASSROOM_BALL_PARK_POS] * 3)
    env.step_dt = .02
    robot = env.scene['robot'].data
    robot.root_link_pos_w[:, 0] = 18.
    robot.root_link_lin_vel_w[:, :2] = torch.tensor([-1., -.5])
    env._dodge_throw_countdown = torch.tensor([1, 100, 1])
    monkeypatch.setattr(torch, 'randn_like', lambda x: torch.full_like(x, .5))
    throw_ball_on_dwell(env, None, throw_interval_range=(2., 5.),
                        dist_range=(4., 4.), angle_deg=0., height_range=(2., 2.),
                        flight_time_range=(.4, .4), lead_target=True, aim_noise_scale=.2,
                        launch_speed_range=launch_speed_range)
    ids = torch.tensor([0, 2])
    expected = robot.root_link_pos_w[ids, :2] + torch.tensor([-.3, -.1])
    assert torch.allclose(env._classroom_ball_target_xy[ids], expected)
    assert env._classroom_ball_target_reached.tolist() == [False, True, False]
    start_xy = env.sim.data.qpos[ids, :2].clone()
    assert torch.allclose(env._classroom_ball_target_direction[ids], expected - start_xy)
    # Launch x=22 is outside; it stays visible while passing the robot's old x=18.
    hide_outside_classroom_ball(env)
    assert ball_in_play(env).tolist() == [True, False, True]
    robot.root_link_pos_w[:, :2] = -100.  # later robot motion cannot move the saved aim
    env.sim.data.qpos[ids, :2] = start_xy + .95 * (expected - start_xy)
    hide_outside_classroom_ball(env)
    assert not env._classroom_ball_target_reached[ids].any()
    env.sim.data.qpos[ids, :2] = start_xy + 1.01 * (expected - start_xy)
    hide_outside_classroom_ball(env)
    assert env._classroom_ball_target_reached[ids].all()
    assert ball_in_play(env).tolist() == [True, False, True]


@pytest.mark.parametrize('play', [False, True])
def test_classroom_event_runs_after_throw_and_reset_parks_below_floor(play):
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=play)
    keys = list(cfg.events)
    assert keys.index('throw_ball_on_dwell') < keys.index('hide_outside_classroom_ball')
    assert cfg.events['hide_outside_classroom_ball'].mode == 'step'
    assert cfg.events['reset_dodge_state'].params['park_offset'] == CLASSROOM_BALL_PARK_POS
    assert cfg.scene.entities['ball'].init_state.pos == CLASSROOM_BALL_PARK_POS
    original = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(play=play)
    assert 'hide_outside_classroom_ball' not in original.events


def test_other_dodge_tasks_keep_their_ball_signals():
    env = make_env([[0., 1000., 1.]])
    del env._classroom_ball_bounds
    assert ball_in_play(env).all()
    assert dodge_ball_state_b(env).abs().sum() > 0
    assert mimickit_dodge_reward(env).all()
    assert ball_contact(env, delta_v_threshold=0.).all()
