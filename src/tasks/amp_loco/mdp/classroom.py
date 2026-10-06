"""Fixed classroom layout and safe randomized robot spawns."""
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from .walk_path import _ensure_stash


def reset_classroom(env, env_ids, wall_names, layout):
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  xy = env.scene.env_origins[env_ids, :2]
  heading = torch.zeros(len(env_ids), device=env.device)
  count = len(wall_names)
  if not hasattr(env, "_wall_pos_w") or env._wall_pos_w.shape[1] != count:
    env._wall_pos_w = torch.zeros(env.num_envs, count, 3, device=env.device)
    env._wall_yaw_w = torch.zeros(env.num_envs, count, device=env.device)
  fwd = torch.stack((heading.cos(), heading.sin()), dim=-1)
  left = torch.stack((-heading.sin(), heading.cos()), dim=-1)
  for i, (name, (x, y, z, yaw)) in enumerate(zip(wall_names, layout, strict=True)):
    pos = torch.zeros(len(env_ids), 3, device=env.device)
    pos[:, :2] = xy + x * fwd + y * left
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + z
    angle = heading + yaw
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0], quat[:, 3] = (angle / 2).cos(), (angle / 2).sin()
    env._wall_pos_w[env_ids, i] = pos
    env._wall_yaw_w[env_ids, i] = angle
    env.scene[name].write_root_link_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
    env.scene[name].write_root_link_velocity_to_sim(
      torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def reset_classroom_robot(env, env_ids):
  """Randomize planar pose after RSI and build a route inside the clear aisles.

  Preserve RSI height, tilt and joints; rotate world velocities with the yaw.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  n, device = len(env_ids), env.device
  robot = env.scene["robot"]
  indexing = robot.data.indexing
  pose = env.sim.data.qpos[env_ids][:, indexing.free_joint_q_adr].clone()
  velocity = env.sim.data.qvel[env_ids][:, indexing.free_joint_v_adr].clone()
  lane = torch.randint(0, 3, (n,), device=device)
  lane_y = torch.tensor((0.0, -3.95, 3.95), device=device)[lane]
  x = torch.rand(n, device=device) * 27.0
  y = lane_y + (torch.rand(n, device=device) * 2 - 1) * 0.15
  origins = env.scene.env_origins[env_ids]
  pose[:, :2] = origins[:, :2] + torch.stack((x, y), -1)
  q = pose[:, 3:7]
  old_yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                        1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  yaw = (torch.rand(n, device=device) * 2 - 1) * torch.pi
  delta = torch.zeros(n, 4, device=device)
  delta[:, 0] = ((yaw - old_yaw) / 2).cos()
  delta[:, 3] = ((yaw - old_yaw) / 2).sin()
  pose[:, 3:7] = quat_mul(delta, q)
  # Root link velocity is world-frame (MuJoCo freejoint angular qvel is local).
  velocity[:, :3] = quat_apply(delta, velocity[:, :3])
  velocity[:, 3:] = quat_apply(pose[:, 3:7], velocity[:, 3:])
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  # Head toward the farther end, then loop through the clear front/rear areas.
  # The long repeated route prevents generic path extension leaving the room.
  _ensure_stash(env)
  pts = env._walk_path_pts
  local = torch.zeros(n, 26, 2, device=device)
  local[:, 0] = torch.stack((x, y), -1)
  end_x = torch.where(x < 13.5, 27.0, 0.0)
  side_y = torch.where(lane_y != 0, -lane_y,
                       torch.where(torch.rand(n, device=device) < 0.5, -3.95, 3.95))
  local[:, 1] = torch.stack((end_x, y), -1)
  for i in range(2, 26):
    if i % 2 == 0:
      next_x, next_y = local[:, i - 1, 0], side_y
      side_y = -side_y
    else:
      next_x, next_y = 27.0 - local[:, i - 1, 0], local[:, i - 1, 1]
    local[:, i] = torch.stack((next_x, next_y), -1)
  pts[env_ids, :26] = local + origins[:, None, :2]
  lengths = torch.linalg.vector_norm(local[:, 1:] - local[:, :-1], dim=-1)
  env._walk_path_cumlen[env_ids, 0] = 0
  env._walk_path_cumlen[env_ids, 1:26] = lengths.cumsum(-1)
  env._walk_path_n[env_ids] = 26
  env._walk_path_epoch[env_ids] += 1
