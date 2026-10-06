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

  _ensure_stash(env)
  local = generate_classroom_route(x, y, lane)
  count = local.shape[1]
  env._walk_path_pts[env_ids, :count] = local + origins[:, None, :2]
  lengths = torch.linalg.vector_norm(local[:, 1:] - local[:, :-1], dim=-1)
  env._walk_path_cumlen[env_ids, 0] = 0
  env._walk_path_cumlen[env_ids, 1:count] = lengths.cumsum(-1)
  env._walk_path_n[env_ids] = count
  env._walk_path_epoch[env_ids] += 1


def generate_classroom_route(x, y, lane, count=80):
  """Random walk on the three longitudinal aisles and seven cross aisles.

  Select only adjacent junctions, excluding immediate reversal. Each centerline
  has at least 1.25 m clearance to furniture; the initial leg stays in its lane.
  """
  n, device = len(x), x.device
  # Cross-aisle centers between the desk front and the next row's chair back.
  stations = torch.tensor((0.0, 5.65, 9.75, 13.85, 17.95, 22.05, 27.0), device=device)
  # Spawn lane indices are 0=center, 1=left, 2=right; graph lanes are sorted.
  lanes = torch.tensor((-3.95, 0.0, 3.95), device=device)
  lane_idx = torch.tensor((1, 0, 2), device=device)[lane]
  difference = stations[None] - x[:, None]
  # Randomize forward/backward first travel; keep the first leg >=0.8 m long.
  forward = torch.rand(n, device=device) < 0.5
  valid = torch.where(forward[:, None], difference >= 0.8, difference <= -0.8)
  fallback = difference.abs() >= 0.8
  valid = torch.where(valid.any(-1, keepdim=True), valid, fallback)
  station_idx = difference.abs().masked_fill(~valid, float('inf')).argmin(-1)
  local = torch.zeros(n, count, 2, device=device)
  local[:, 0] = torch.stack((x, y), -1)
  local[:, 1] = torch.stack((stations[station_idx], lanes[lane_idx]), -1)
  previous = torch.full((n,), -1, device=device, dtype=torch.long)
  for i in range(2, count):
    # Neighbor choices: backward/forward in this lane, or adjacent lane at this junction.
    sx = torch.stack((station_idx - 1, station_idx + 1, station_idx, station_idx), -1)
    ly = torch.stack((lane_idx, lane_idx, lane_idx - 1, lane_idx + 1), -1)
    legal = (sx >= 0) & (sx < len(stations)) & (ly >= 0) & (ly < len(lanes))
    node = sx * len(lanes) + ly
    legal &= node != previous[:, None]
    if i == 2:
      first_forward = stations[station_idx] > x
      legal[:, 0] &= ~first_forward
      legal[:, 1] &= first_forward
    choice = torch.rand(n, 4, device=device).masked_fill(~legal, -1).argmax(-1)
    previous = station_idx * len(lanes) + lane_idx
    station_idx = sx.gather(1, choice[:, None]).squeeze(-1)
    lane_idx = ly.gather(1, choice[:, None]).squeeze(-1)
    local[:, i] = torch.stack((stations[station_idx], lanes[lane_idx]), -1)
  return local
