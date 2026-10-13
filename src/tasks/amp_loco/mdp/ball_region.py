"""Apply room bounds to thrown balls only after they reach their saved aim point."""
import torch


def reset_ball_target_gate(env, env_ids):
  """Parked balls use room bounds; each new throw starts a fresh target gate."""
  if not hasattr(env, "_classroom_ball_bounds"):
    return
  if not hasattr(env, "_classroom_ball_target_reached"):
    xy = env.scene.env_origins[:, :2]
    env._classroom_ball_target_xy = torch.zeros_like(xy)
    env._classroom_ball_target_direction = torch.zeros_like(xy)
    env._classroom_ball_target_reached = torch.ones(
      env.num_envs, device=env.device, dtype=torch.bool)
  env._classroom_ball_target_xy[env_ids] = 0.0
  env._classroom_ball_target_direction[env_ids] = 0.0
  env._classroom_ball_target_reached[env_ids] = True


def set_ball_throw_target(env, env_ids, start_xy, target_xy):
  """Save the final world-frame aim, including lead, noise and aim offsets."""
  if not hasattr(env, "_classroom_ball_bounds"):
    return
  reset_ball_target_gate(env, env_ids)
  env._classroom_ball_target_xy[env_ids] = target_xy
  env._classroom_ball_target_direction[env_ids] = target_xy - start_xy
  env._classroom_ball_target_reached[env_ids] = False


def _ball_target_reached(env, position):
  # Crossing the plane through target_xy normal to the initial flight direction
  # detects arrival even when one simulation step skips over the exact point.
  offset = position[:, :2] - env._classroom_ball_target_xy
  crossed = (offset * env._classroom_ball_target_direction).sum(dim=-1) >= 0.0
  return env._classroom_ball_target_reached | crossed


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
  inside = ((position >= origins + bounds[0]) & (position <= origins + bounds[1])).all(dim=-1)
  if hasattr(env, "_classroom_ball_target_reached"):
    inside = inside | ~_ball_target_reached(env, position)
  return inside & torch.isfinite(position).all(dim=-1)


def hide_outside_classroom_ball(env, env_ids=None, ball_name="ball"):
  """Latch target arrival, then park out-of-room balls until the next throw."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if hasattr(env, "_classroom_ball_target_reached"):
    position, _ = ball_world_state(env, ball_name)
    env._classroom_ball_target_reached[env_ids] = _ball_target_reached(env, position)[env_ids]
  hidden = env_ids[~ball_in_play(env, ball_name)[env_ids]]
  if hidden.numel() == 0:
    return
  if hasattr(env, "_classroom_ball_target_reached"):
    env._classroom_ball_target_reached[hidden] = True
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
