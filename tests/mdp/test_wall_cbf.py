"""Integration test: static-wall obstacle on the MimicKit-Wall dodge task.

Proves the wall wiring is load-bearing:
  - reset_wall_pose places the wall BESIDE each spawn point at the configured
    lateral distance (both sides across envs), clear of the RSI reset poses.
  - pin_wall restores the sampled pose after the wall is knocked away (kinematic
    static obstacle).
  - wall_link_cbf_reward is ~0 when the robot stands still far from the wall, and
    negative when the robot moves TOWARD the wall (the CBF constraint bites).

GPU required (builds a full sim env).  Kept at 64 envs / a handful of forwards, so
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


def _refresh(env, *entities):
    """Propagate direct state writes into the entity data tensors."""
    import mujoco_warp as mjw

    env.scene.write_data_to_sim()
    mjw.forward(env.sim.wp_model, env.sim.wp_data)
    for e in entities:
        e.update(env.step_dt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_wall_placement_randomized_beside_spawn():
    """Wall sits beside each env origin: |y| in [0.6, 1.0], |x| <= 0.5, both sides present."""
    env = _build_wall_env()
    wall = env.scene["wall"]

    assert hasattr(env, "_wall_pose_w"), "reset_wall_pose did not stash the wall pose"
    off = env._wall_pose_w - env.scene.env_origins  # (N, 3)

    lat = off[:, 1].abs()
    assert torch.all(lat >= 0.6 - 1e-5) and torch.all(lat <= 1.0 + 1e-5), (
        f"lateral distance out of [0.6, 1.0]: min {lat.min():.3f}, max {lat.max():.3f}"
    )
    assert torch.all(off[:, 0].abs() <= 0.5 + 1e-5), (
        f"x offset out of [-0.5, 0.5]: max |x| {off[:, 0].abs().max():.3f}"
    )
    # Bimodal side sampling: with 64 envs both sides must appear (p(all one side) = 2^-63).
    assert (off[:, 1] > 0).any() and (off[:, 1] < 0).any(), "wall only ever on one side"
    # Centre at half height (wall base on the ground), and entity pose matches the stash.
    assert torch.allclose(env._wall_pose_w[:, 2], torch.ones_like(lat), atol=1e-4)
    assert torch.allclose(wall.data.root_link_pos_w, env._wall_pose_w, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_pin_wall_restores_pose():
    """After teleporting the wall away, pin_wall puts it back at the sampled pose."""
    import src.tasks.amp_loco.mdp as mdp

    env = _build_wall_env()
    wall = env.scene["wall"]
    n = env.num_envs

    away = env._wall_pose_w.clone()
    away[:, 1] += 2.0  # shove the wall 2 m further out
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0
    wall.write_root_link_pose_to_sim(torch.cat([away, quat], dim=-1))
    _refresh(env, wall)
    assert not torch.allclose(wall.data.root_link_pos_w, env._wall_pose_w, atol=1e-3)

    mdp.pin_wall(env, None)
    _refresh(env, wall)
    assert torch.allclose(wall.data.root_link_pos_w, env._wall_pose_w, atol=1e-4), (
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

    r = mdp.wall_link_cbf_reward(env, alpha=1.0, margin=0.05, constraint_clip=2.0)
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
    direction = torch.sign(env._wall_pose_w[:, :2] - robot_xy)  # (N, 2)
    vel = torch.zeros(n, 6, device=env.device)
    vel[:, 0:2] = direction * 1.0
    robot.write_root_link_velocity_to_sim(vel)
    _refresh(env, robot)

    r = mdp.wall_link_cbf_reward(env, alpha=1.0, margin=0.05, constraint_clip=2.0)
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
