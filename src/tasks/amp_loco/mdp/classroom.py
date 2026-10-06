"""Fixed classroom layout relative to each environment's fresh RSI pose."""
import torch
from .walk_path import _fresh_root_yaw


def reset_classroom(env, env_ids, wall_names, layout):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  xy, heading = _fresh_root_yaw(env, env.scene["robot"], env_ids)
  count = len(wall_names)
  if not hasattr(env, "_wall_pos_w") or env._wall_pos_w.shape[1] != count:
    env._wall_pos_w = torch.zeros(env.num_envs, count, 3, device=env.device)
    env._wall_yaw_w = torch.zeros(env.num_envs, count, device=env.device)
  fwd = torch.stack((heading.cos(), heading.sin()), dim=-1)
  left = torch.stack((-heading.sin(), heading.cos()), dim=-1)
  for i, (name, (x, y, z, yaw)) in enumerate(zip(wall_names, layout, strict=True)):
    pos = torch.zeros(len(env_ids), 3, device=env.device)
    pos[:, :2] = xy + x * fwd + y * left
    pos[:, 2] = z
    angle = heading + yaw
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0], quat[:, 3] = (angle / 2).cos(), (angle / 2).sin()
    env._wall_pos_w[env_ids, i] = pos
    env._wall_yaw_w[env_ids, i] = angle
    env.scene[name].write_root_link_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
    env.scene[name].write_root_link_velocity_to_sim(
      torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
