"""Integration tests: static-wall obstacles on the MimicKit Wall / WallWalk dodge tasks.

Proves the wall wiring is load-bearing:
  - reset_wall_pose places the wall BESIDE the robot relative to its post-RSI
    heading (both sides across envs), clear of the RSI reset poses.
  - pin_wall restores the sampled pose after the wall is knocked away (kinematic
    static obstacle).
  - wall_link_cbf_reward is ~0 when the robot stands still far from the wall, and
    negative when the robot moves TOWARD the wall (the CBF constraint bites).
  - The WallWalk task drives a constant forward command, scatters multiple walls
    along the path, recycles walked-past walls back ahead, and observes the k
    nearest walls sorted by distance.

GPU required (builds full sim envs).  Kept at 64 envs / a handful of forwards, so
GPU time is negligible (same pattern as test_omni_cbf_sense.py).
"""

import torch
import pytest


def _build_wall_env(num_envs: int = 64):
    """Build the MimicKit-Wall state env (env-vars read at call time by the builder)."""
    import src.tasks  # noqa: F401 -- triggers task registration side-effect
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_mimickit_wall_flat_env_cfg,
    )

    cfg = g1_amp_dodge_mimickit_wall_flat_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    env.reset()
    return env


def _build_wallwalk_env(num_envs: int = 64):
    """Build the MimicKit-WallWalk state env (3 walls, forward-walk command)."""
    import src.tasks  # noqa: F401 -- triggers task registration side-effect
    from mjlab.envs import ManagerBasedRlEnv
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
    )

    cfg = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    env.reset()
    return env


def _refresh(env, *entities):
    """Propagate direct state writes into the entity data tensors."""
    import mujoco_warp as mjw

    env.scene.write_data_to_sim()
    mjw.forward(env.sim.wp_model, env.sim.wp_data)
    for e in entities:
        e.update(env.step_dt)


def _heading_frame_from_yaw(yaw: torch.Tensor):
    """(fwd_xy, perp_xy) for a batch of yaws; perp = 90deg left of fwd."""
    fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    perp = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=-1)
    return fwd, perp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_placement_randomized_beside_spawn_heading_relative():
    """Wall sits BESIDE the robot relative to its reset heading: |lateral| in
    [0.6, 1.0], |along| <= 0.5, both sides present (RSI yaws are ~-90deg, so this
    must NOT be checked in world axes)."""
    env = _build_wall_env()
    wall = env.scene["wall"]
    robot = env.scene["robot"]

    assert hasattr(env, "_wall_pos_w"), "reset_wall_pose did not stash the wall pose"
    assert env._wall_pos_w.shape == (env.num_envs, 1, 3)
    # Wall yaw == the robot's reset heading: use it as the placement frame.
    fwd, perp = _heading_frame_from_yaw(env._wall_yaw_w[:, 0])
    off = env._wall_pos_w[:, 0, :2] - robot.data.root_link_pos_w[:, :2]  # (N, 2)

    lat = (off * perp).sum(dim=-1)
    along = (off * fwd).sum(dim=-1)
    assert torch.all(lat.abs() >= 0.6 - 1e-5) and torch.all(lat.abs() <= 1.0 + 1e-5), (
        f"lateral distance out of [0.6, 1.0]: min {lat.abs().min():.3f}, max {lat.abs().max():.3f}"
    )
    assert torch.all(along.abs() <= 0.5 + 1e-5), (
        f"along-heading offset out of [-0.5, 0.5]: max {along.abs().max():.3f}"
    )
    # Bimodal side sampling: with 64 envs both sides must appear (p(all one side) = 2^-63).
    assert (lat > 0).any() and (lat < 0).any(), "wall only ever on one side"
    # Centre at half height (wall base on the ground), and entity pose matches the stash.
    assert torch.allclose(
        env._wall_pos_w[:, 0, 2], torch.ones_like(lat), atol=1e-4
    )
    assert torch.allclose(wall.data.root_link_pos_w, env._wall_pos_w[:, 0], atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_pin_wall_restores_pose():
    """After teleporting the wall away, pin_wall puts it back at the sampled pose."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    wall = env.scene["wall"]
    n = env.num_envs

    away = env._wall_pos_w[:, 0].clone()
    away[:, 1] += 2.0  # shove the wall 2 m further out
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0
    wall.write_root_link_pose_to_sim(torch.cat([away, quat], dim=-1))
    _refresh(env, wall)
    assert not torch.allclose(wall.data.root_link_pos_w, env._wall_pos_w[:, 0], atol=1e-3)

    mdp.pin_wall(env, None, wall_names=("wall",))
    _refresh(env, wall)
    assert torch.allclose(wall.data.root_link_pos_w, env._wall_pos_w[:, 0], atol=1e-4), (
        "pin_wall did not restore the sampled wall pose"
    )
    assert wall.data.root_link_lin_vel_w.abs().max() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_link_cbf_zero_when_standing_far():
    """Standing still with the wall >= 0.6 m away -> barrier satisfied -> ~0 penalty."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    robot = env.scene["robot"]
    n = env.num_envs

    # RSI reset writes the motion clip's velocities; zero them so the robot is truly still.
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device))
    robot.write_joint_state_to_sim(
        robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
    )
    _refresh(env, robot)

    r = mdp.wall_link_cbf_reward(
        env, robot_name="robot", wall_names=("wall",),
        alpha=1.0, margin=0.05, constraint_clip=2.0,
    )
    # Links are >= ~0.5 m from the surface and ~stationary: h_dot ~ 0, alpha*h > 0 -> 0.
    assert r.max() <= 0.0, "CBF reward must be <= 0"
    assert r.min() > -0.05, (
        f"standing far from the wall should be (near) penalty-free, got min {r.min():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_link_cbf_penalizes_approach():
    """Moving the robot TOWARD the wall -> h_dot < 0 -> negative CBF reward."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    robot = env.scene["robot"]
    n = env.num_envs

    # Zero the RSI clip velocities, then drive the whole robot laterally toward its
    # wall at 1 m/s.
    robot.write_joint_state_to_sim(
        robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
    )
    robot_xy = robot.data.root_link_pos_w[:, :2]
    direction = torch.sign(env._wall_pos_w[:, 0, :2] - robot_xy)  # (N, 2)
    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 0:2] = direction * 1.0
    robot.write_root_link_velocity_to_sim(vel)
    _refresh(env, robot)

    r = mdp.wall_link_cbf_reward(
        env, robot_name="robot", wall_names=("wall",),
        alpha=1.0, margin=0.05, constraint_clip=2.0,
    )
    # h ~ 0.5 m (wall at 0.6-1.0 m, minus thickness/radius), h_dot ~ -1 m/s toward the
    # wall -> h_dot + alpha*h ~ -0.5 < 0 for the links facing the wall.
    assert (r < -0.1).float().mean() > 0.9, (
        f"approaching the wall should penalize nearly all envs; rewards: "
        f"min {r.min():.3f}, mean {r.mean():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_ball_hit_dv_fallback_disabled():
    """The wall task must disable the ball-hit velocity-discontinuity fallback (a ball
    bouncing off the wall near the robot would otherwise falsely register as a body hit)."""
    env = _build_wall_env()
    params = env.termination_manager.get_term_cfg("ball_hit").params
    assert params.get("delta_v_threshold", 0.0) == 0.0, (
        f"delta_v_fallback should be off on the wall task, got {params}"
    )


# ---------------------------------------------------------------------------
# WallWalk task: forward-walk command, multi-wall corridor, recycling, obs.
# ---------------------------------------------------------------------------

_WALLWALK_NAMES = ("wall_0", "wall_1", "wall_2")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_forward_command_constant():
    """Path-follow pure pursuit pins the goal 2 m ahead ON THE PATH -> at reset the
    path's first segment is aligned with the robot heading (and >= the lookahead),
    so the command is the same constant clamped forward walk as before
    (kp * 2.0 = 3.0 > max_lin_vel_x = 1.3 -> vx = 1.3 m/s for every env)."""
    env = _build_wallwalk_env()
    cmd = env.command_manager.get_command("twist")  # (N, 3): vx, vy, wz
    assert cmd.shape == (env.num_envs, 3)
    assert torch.allclose(
        cmd[:, 0], torch.full_like(cmd[:, 0], 1.3), atol=1e-4
    ), f"forward command should be clamped to 1.3 m/s, got {cmd[:, 0].unique()}"
    assert cmd[:, 1].abs().max() < 1e-4 and cmd[:, 2].abs().max() < 1e-4, (
        "no lateral / yaw command expected on the heading-aligned first segment "
        "(tiny float residue from the path geometry is fine)"
    )


def _path_distance_and_station(env, points_xy: torch.Tensor):
    """(min distance, arc-length station of the closest point) from each (N, 2) xy
    point to its env's walk-path polyline. Vectorized over all segments."""
    pts = env._walk_path_pts  # (N, M, 2)
    cum = env._walk_path_cumlen  # (N, M)
    count = env._walk_path_n  # (N,)
    p0 = pts[:, :-1]  # (N, M-1, 2)
    p1 = pts[:, 1:]
    seg = p1 - p0
    l2 = (seg * seg).sum(dim=-1).clamp_min(1e-9)
    u = (((points_xy.unsqueeze(1) - p0) * seg).sum(dim=-1) / l2).clamp(0.0, 1.0)
    closest = p0 + u.unsqueeze(-1) * seg
    dist = (closest - points_xy.unsqueeze(1)).norm(dim=-1)  # (N, M-1)
    station = cum[:, :-1] + u * seg.norm(dim=-1)
    # Ignore padding segments beyond each env's vertex count.
    valid = (
        torch.arange(pts.shape[1] - 1, device=env.device).unsqueeze(0)
        < (count - 1).unsqueeze(1)
    )
    dist = torch.where(valid, dist, torch.full_like(dist, 1e9))
    j = dist.argmin(dim=1)
    ar = torch.arange(env.num_envs, device=env.device)
    return dist[ar, j], station[ar, j]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_walls_scattered_ahead_on_path():
    """Walk-path placement: wall i sits at arc-length station U(2.5,4)+3i along the
    random path, offset 0.6-1.0 m perpendicular to the local tangent (station/distance
    measured against the polyline, toleranced for curve geometry)."""
    env = _build_wallwalk_env()

    assert env._wall_pos_w.shape == (env.num_envs, 3, 3)
    for i in range(3):
        dist, station = _path_distance_and_station(env, env._wall_pos_w[:, i, :2])
        assert torch.all(dist >= 0.6 - 0.15) and torch.all(dist <= 1.0 + 0.05), (
            f"wall_{i} distance-to-path out of ~[0.6, 1.0]: "
            f"min {dist.min():.3f}, max {dist.max():.3f}"
        )
        lo, hi = 2.5 + i * 3.0, 4.0 + i * 3.0
        assert torch.all(station >= lo - 1.2) and torch.all(station <= hi + 1.2), (
            f"wall_{i} path station out of ~[{lo}, {hi}]: "
            f"min {station.min():.3f}, max {station.max():.3f}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_recycle_moves_walked_past_walls_ahead():
    """A wall > 1 m behind the robot teleports to a path station 7-10 m ahead of the
    robot's path progress, offset 0.5-1.2 m from the path (along_path mode)."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    fwd, _ = _heading_frame_from_yaw(env._wall_yaw_w[:, 0])
    before = env._wall_pos_w.clone()

    # Drag wall_0 to 3 m BEHIND the robot (leave wall_1/2 untouched).
    env._wall_pos_w[:, 0, :2] = robot.data.root_link_pos_w[:, :2] - 3.0 * fwd
    mdp.recycle_walls_ahead(env, None, wall_names=_WALLWALK_NAMES, along_path=True)

    dist, station = _path_distance_and_station(env, env._wall_pos_w[:, 0, :2])
    s_robot = env.command_manager.get_term("twist")._path_s  # ~0 right after reset
    assert torch.all(station >= s_robot + 7.0 - 1.2) and torch.all(station <= s_robot + 10.0 + 1.2), (
        f"recycled wall_0 station should be ~7-10 m ahead of s_robot, got "
        f"min {station.min():.3f}, max {station.max():.3f} (s_robot ~ {s_robot.mean():.3f})"
    )
    assert torch.all(dist >= 0.5 - 0.15) and torch.all(dist <= 1.2 + 0.05), (
        f"recycled wall_0 distance-to-path out of ~[0.5, 1.2]: "
        f"min {dist.min():.3f}, max {dist.max():.3f}"
    )
    # wall_1/2 were still ahead -> untouched.
    assert torch.allclose(env._wall_pos_w[:, 1:], before[:, 1:]), (
        "wall_1/2 should not have been recycled"
    )
    # The entity pose was teleported too (not just the stash). recycle writes sim
    # qpos directly, so refresh (forward + update) before reading entity data.
    w0 = env.scene["wall_0"]
    _refresh(env, w0)
    assert torch.allclose(
        w0.data.root_link_pos_w[:, :2], env._wall_pos_w[:, 0, :2], atol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wallwalk_obs_k_nearest_sorted():
    """dodge_wall_state_b returns (N, 3k) with the k walls sorted by centre distance."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]

    obs = mdp.dodge_wall_state_b(
        env, robot_name="robot", wall_names=_WALLWALK_NAMES, k=3
    )
    assert obs.shape == (env.num_envs, 9)

    # Recompute each wall's centre distance and check the obs slots are rank-ordered.
    rp = robot.data.root_link_pos_w[:, :2]
    d = torch.stack(
        [(env.scene[n].data.root_link_pos_w[:, :2] - rp).norm(dim=-1) for n in _WALLWALK_NAMES],
        dim=1,
    )  # (N, 3)
    d_sorted, _ = torch.sort(d, dim=1)
    obs_d = obs.view(env.num_envs, 3, 3)[:, :, :2].norm(dim=-1)  # rel_x, rel_y norms
    assert torch.allclose(obs_d, d_sorted, atol=1e-4), (
        "obs slots are not the k walls sorted by centre distance"
    )

    # And the env's ball_state group carries 6 (ball) + 9 (walls) = 15 dims.
    group = env.observation_manager.group_obs_dim["ball_state"]
    assert group == [15] or group == (15,), f"ball_state obs dim should be 15, got {group}"
