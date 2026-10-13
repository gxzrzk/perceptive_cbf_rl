"""Mask balls outside a configured room without changing other dodge tasks."""
import torch


def ball_world_state(env, ball_name="ball"):
  ball = env.scene[ball_name]
  if hasattr(env, "_classroom_ball_bounds"):
    # Step events run after forward(): read freejoint state so a newly launched
    # or parked ball does not use the previous frame's derived position/velocity.
    indexing = ball.data.indexing
    return (env.sim.data.qpos[:, indexing.free_joint_q_adr[:3]],
            env.sim.data.qvel[:, indexing.free_joint_v_adr[:3]])
  return ball.data.root_link_pos_w, ball.data.root_link_lin_vel_w


def ball_in_play(env, ball_name="ball"):
  bounds = getattr(env, "_classroom_ball_bounds", None)
  if bounds is None:
    return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
  position, _ = ball_world_state(env, ball_name)
  origins = env.scene.env_origins
  return ((position >= origins + bounds[0]) & (position <= origins + bounds[1])).all(dim=-1)


def hide_outside_classroom_ball(env, env_ids=None, ball_name="ball"):
  """Park invalid balls below the floor, keeping their state bounded until reuse."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  hidden = env_ids[~ball_in_play(env, ball_name)[env_ids]]
  if hidden.numel() == 0:
    return
  ball = env.scene[ball_name]
  pose = env.scene.env_origins.new_zeros(hidden.numel(), 7)
  pose[:, :3] = env.scene.env_origins[hidden] + env._classroom_ball_park_pos
  pose[:, 3] = 1.0
  ball.write_root_link_pose_to_sim(pose, env_ids=hidden)
  ball.write_root_link_velocity_to_sim(pose.new_zeros(hidden.numel(), 6), env_ids=hidden)
  # Commands were computed before the step event. Clear any stale threat so the
  # next navigation reward treats the hidden ball as absent as well.
  command = env.command_manager.get_term("twist")
  command._dodge_threat[hidden] = False
  command._dodge_h[hidden] = 0.0
  command._dodge_tti[hidden] = 0.0
  command.metrics["cbf_active_frac"][hidden] = 0.0
  command.metrics["cbf_min_h"][hidden] = 0.0
