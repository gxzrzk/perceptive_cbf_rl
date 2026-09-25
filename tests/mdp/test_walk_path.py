"""Integration tests: random walk paths on the MimicKit WallWalk dodge task.

Proves the random-path wiring is load-bearing:
  - reset_walk_path generates a per-env polyline that starts at the robot's
    post-reset pose, aligned with its heading, with a DIFFERENT shape per env
    (random turns), and >= initial_len metres long.
  - walk_path_query interpolates arc-length stations correctly (endpoints + a
    mid-segment point) and extends past the end.
  - extend_walk_path lengthens the path continuously (old vertices untouched).
  - The path-follow command pins the goal to the lookahead point and tracks the
    robot's arc-length progress (teleport the robot along the path -> _path_s
    follows, goal stays lookahead metres ahead).

GPU required (builds full sim envs). Same pattern as test_wall_cbf.py.
"""

import torch
import pytest


def _build_wallwalk_env(num_envs: int = 64):
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


def _path_total(env):
    ar = torch.arange(env.num_envs, device=env.device)
    return env._walk_path_cumlen[ar, env._walk_path_n - 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_path_starts_at_robot_aligned_with_heading():
    """Path origin == robot post-reset xy; first segment runs along the robot's
    heading; total length >= initial_len (80 m); shapes differ across envs."""
    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)

    assert hasattr(env, "_walk_path_pts") and hasattr(env, "_walk_path_n")
    origin = env._walk_path_pts[:, 0]
    assert torch.allclose(
        origin, robot.data.root_link_pos_w[:, :2], atol=1e-3
    ), "path must start at the robot's post-reset xy"

    seg0 = env._walk_path_pts[:, 1] - env._walk_path_pts[:, 0]
    path_yaw0 = torch.atan2(seg0[:, 1], seg0[:, 0])
    robot_yaw = robot.data.heading_w
    dyaw = torch.atan2(
        torch.sin(path_yaw0 - robot_yaw), torch.cos(path_yaw0 - robot_yaw)
    )
    assert dyaw.abs().max() < 1e-2, (
        f"first segment must run along the robot heading, max dev {dyaw.abs().max():.4f} rad"
    )

    total = _path_total(env)
    assert torch.all(total >= 80.0 - 1e-4), (
        f"initial path should be >= 80 m, min {total.min():.2f}"
    )

    # Random shapes: heading a few segments in must vary across envs (the shared
    # start heading washes out after the first random turn increments).
    k = 6
    segk = env._walk_path_pts[:, k] - env._walk_path_pts[:, k - 1]
    yaw_k = torch.atan2(segk[:, 1], segk[:, 0])
    assert yaw_k.std() > 0.05, (
        f"paths look identical across envs (segment-{k} heading std {yaw_k.std():.4f})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_query_endpoints_and_interpolation():
    """s=0 -> origin; s=total -> last vertex; a mid-segment s interpolates linearly."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    ar = torch.arange(env.num_envs, device=env.device)
    total = _path_total(env)

    p0, _ = mdp.walk_path_query(env, ar, torch.zeros_like(total))
    assert torch.allclose(p0, env._walk_path_pts[:, 0], atol=1e-5)
    p1, _ = mdp.walk_path_query(env, ar, total)
    last = env._walk_path_pts[ar, env._walk_path_n - 1]
    assert torch.allclose(p1, last, atol=1e-4)

    # Midpoint of segment 0: s = half its length -> halfway between vertices 0/1,
    # tangent == the segment direction.
    c1 = env._walk_path_cumlen[:, 1]
    pm, yaw = mdp.walk_path_query(env, ar, 0.5 * c1)
    expect = 0.5 * (env._walk_path_pts[:, 0] + env._walk_path_pts[:, 1])
    assert torch.allclose(pm, expect, atol=1e-4)
    seg0 = env._walk_path_pts[:, 1] - env._walk_path_pts[:, 0]
    seg0_yaw = torch.atan2(seg0[:, 1], seg0[:, 0])
    dyaw = torch.atan2(torch.sin(yaw - seg0_yaw), torch.cos(yaw - seg0_yaw))
    assert dyaw.abs().max() < 1e-4

    # Querying past the end clamps to the last vertex.
    p2, _ = mdp.walk_path_query(env, ar, total + 50.0)
    assert torch.allclose(p2, last, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_extend_walk_path_continuous_and_longer():
    """extend_walk_path appends segments: old vertices/cumlen untouched, the new
    total >= requested, and the appended chain continues from the old endpoint."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    ar = torch.arange(env.num_envs, device=env.device)
    n0 = env._walk_path_n.clone()
    total0 = _path_total(env)
    pts0 = env._walk_path_pts.clone()
    cum0 = env._walk_path_cumlen.clone()
    end0 = env._walk_path_pts[ar, n0 - 1].clone()

    mdp.extend_walk_path(env, ar, total0 + 50.0)

    n1 = env._walk_path_n
    assert torch.all(n1 > n0), "extension added no vertices"
    # Old prefix untouched (compare against the snapshot, padded rows excluded).
    for i in (0, 1, 5):
        assert torch.allclose(env._walk_path_pts[ar, i], pts0[ar, i], atol=0.0)
    assert torch.allclose(
        env._walk_path_pts[ar, n0 - 1], end0, atol=0.0
    ), "extension must not move the old endpoint"
    assert torch.allclose(
        env._walk_path_cumlen[ar, n0 - 1], total0, atol=0.0
    ), "extension must not shift the old arc lengths"
    total1 = _path_total(env)
    assert torch.all(total1 >= total0 + 50.0 - 1e-4), (
        f"extended total should be >= old+50, got min delta {(total1 - total0).min():.2f}"
    )
    # First appended segment continues from the old endpoint (no jump).
    first_new = env._walk_path_pts[ar, n0]
    gap = (first_new - end0).norm(dim=-1)
    assert torch.all(gap >= 2.0 - 1e-4) and torch.all(gap <= 4.0 + 1e-4), (
        f"appended segment length out of [2, 4]: min {gap.min():.3f}, max {gap.max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_command_tracks_progress_and_lookahead_goal():
    """Teleport the robot to the path point at s=10 -> the command's _path_s follows
    and the goal sits path_lookahead (2 m) further along the path."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    target_xy, _ = mdp.walk_path_query(env, ar, torch.full((env.num_envs,), 10.0, device=env.device))
    pose = robot.data.root_link_pose_w.clone()  # (N, 7): pos(3) + quat wxyz(4)
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)

    env.command_manager.compute(dt=env.step_dt)

    assert torch.allclose(cmd._path_s, torch.full_like(cmd._path_s, 10.0), atol=0.3), (
        f"_path_s should track the teleport, got mean {cmd._path_s.mean():.3f}"
    )
    goal_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 2.0)
    assert torch.allclose(cmd.goal_pos_w, goal_xy, atol=1e-3), (
        "goal should be the path point path_lookahead ahead of the robot's progress"
    )
    # And the episode-reset epoch logic left no env flagged as standing/in-place.
    assert not cmd.is_standing_env.any() and not cmd.is_inplace_env.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_walk_path_progress_reward():
    """walk_path_progress_reward = arc-length rate along the path (m/s, clipped):
    ~0 when stationary, strongly positive after a forward teleport (clipped at the
    2.0 m/s cap), ~0 again right after (no stale delta)."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wallwalk_env()
    robot = env.scene["robot"]
    ar = torch.arange(env.num_envs, device=env.device)
    cmd = env.command_manager.get_term("twist")

    # Stationary at the path origin -> no progress.
    env.command_manager.compute(dt=env.step_dt)
    r0 = mdp.walk_path_progress_reward(env)
    assert r0.abs().max() < 0.5, (
        f"stationary robot should get ~0 progress reward, got max {r0.abs().max():.3f}"
    )

    # Teleport 0.5 m forward along the path -> 0.5 m in one 0.02 s step = 25 m/s,
    # clipped to the +2.0 m/s cap.
    target_xy, _ = mdp.walk_path_query(env, ar, cmd._path_s + 0.5)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = target_xy
    robot.write_root_link_pose_to_sim(pose)
    _refresh(env, robot)
    env.command_manager.compute(dt=env.step_dt)
    r1 = mdp.walk_path_progress_reward(env)
    assert torch.all(r1 > 1.5), (
        f"forward teleport should saturate the progress reward, min {r1.min():.3f}"
    )

    # Standing still again -> the next compute reports ~0 (delta is per-step, no
    # memory of the teleport).
    env.command_manager.compute(dt=env.step_dt)
    r2 = mdp.walk_path_progress_reward(env)
    assert r2.abs().max() < 0.5, (
        f"progress reward should fall back to ~0 once the robot stops, got max {r2.abs().max():.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_path_regenerated_per_episode_reset():
    """A fresh env.reset() regenerates every env's path: epoch bumps, command
    progress restarts at 0, and the new path still starts at the robot."""
    env = _build_wallwalk_env()
    robot = env.scene["robot"]

    epoch0 = env._walk_path_epoch.clone()
    env.reset()
    assert torch.all(env._walk_path_epoch > epoch0), "reset did not regenerate the path"
    cmd = env.command_manager.get_term("twist")
    assert cmd._path_s.abs().max() < 1.0, (
        f"progress should restart near the path origin after reset, got max {cmd._path_s.abs().max():.3f}"
    )
    origin = env._walk_path_pts[:, 0]
    assert torch.allclose(origin, robot.data.root_link_pos_w[:, :2], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_step_smoke_no_nan():
    """Step the env a few times with zero actions: obs/command/path stay finite and
    the goal keeps tracking the lookahead (the command recomputes every step)."""
    env = _build_wallwalk_env()
    cmd = env.command_manager.get_term("twist")
    for _ in range(10):
        obs, rew, terminated, time_out, extras = env.step(
            torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        )
    cmd_vel = env.command_manager.get_command("twist")
    assert torch.isfinite(cmd_vel).all(), "non-finite velocity command"
    assert torch.isfinite(cmd._path_s).all(), "non-finite path progress"
    assert torch.isfinite(env._walk_path_pts).all(), "non-finite path"
