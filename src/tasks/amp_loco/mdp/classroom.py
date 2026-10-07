"""Fixed classroom layout and safe randomized robot spawns."""
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from .walk_path import _ensure_stash
from src.assets.objects.classroom import (
  CLASSROOM_SPAWN_LANES, CLASSROOM_ROUTE_STATIONS,
  CLASSROOM_ROW_PITCH, CLASSROOM_COLUMN_Y,
)


def initialize_static_classroom(env, env_ids, wall_names, layout):
  """Cache fixed layout metadata once; geometry is already placed at compile time.

  Warp caches static geom transforms/BVH, so independent physics worlds use the
  same local origin rather than translating static bodies after compilation.
  """
  if torch.any(env.scene.env_origins != 0):
    raise ValueError("Static classroom requires zero environment origins (env_spacing=0)")
  poses = torch.tensor(layout, device=env.device, dtype=torch.float32)
  if len(wall_names) != len(layout):
    raise ValueError("Classroom obstacle names and layout must have matching lengths")
  env._wall_pos_w = poses[None, :, :3].expand(env.num_envs, -1, -1)
  env._wall_yaw_w = poses[None, :, 3].expand(env.num_envs, -1)


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
  lane = torch.randint(0, len(CLASSROOM_SPAWN_LANES), (n,), device=device)
  lane_y = torch.tensor(CLASSROOM_SPAWN_LANES, device=device)[lane]
  x = torch.rand(n, device=device) * CLASSROOM_ROUTE_STATIONS[-1]
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

  local, counts = round_classroom_route(generate_classroom_route(x, y, lane))
  count = local.shape[1]
  _ensure_stash(env, max_points=count)
  env._walk_path_pts[env_ids, :count] = local + origins[:, None, :2]
  lengths = torch.linalg.vector_norm(local[:, 1:] - local[:, :-1], dim=-1)
  env._walk_path_cumlen[env_ids, 0] = 0
  env._walk_path_cumlen[env_ids, 1:count] = lengths.cumsum(-1)
  env._walk_path_n[env_ids] = counts
  env._walk_path_epoch[env_ids] += 1


def generate_classroom_route(x, y, lane, count=80):
  """Random walk on the two longitudinal aisles and six cross aisles.

  Select only adjacent junctions, excluding immediate reversal. Each centerline
  has at least 1.1 m clearance to furniture; the initial leg stays in its lane.
  """
  n, device = len(x), x.device
  # Cross-aisle centers between the desk front and the next row's chair back.
  stations = torch.tensor(CLASSROOM_ROUTE_STATIONS, device=device)
  # The same lane order is used by spawning and the junction graph.
  lanes = torch.tensor(CLASSROOM_SPAWN_LANES, device=device)
  lane_idx = lane.clone()
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


def round_classroom_route(points, radius=0.6, arc_segments=6, clearance=0.9):
  """Replace junctions with tangent circular arcs; return packed points/counts.

  The radius is limited by both adjacent leg lengths and available aisle space.
  At the midpoint of a turn, the departure from the nearer original leg is
  R * (1 - cos(turn_angle / 2)); bounding this preserves the clearance corridor.
  Straight junctions contribute one vertex, without zero-length arc segments.
  """
  if radius <= 0 or arc_segments < 2 or points.shape[1] < 3:
    raise ValueError("Need a positive radius, >=2 arc segments and >=3 route points")
  column_gaps = [b - a - 1.2 for a, b in zip(CLASSROOM_COLUMN_Y, CLASSROOM_COLUMN_Y[1:])]
  half_aisle = min(CLASSROOM_ROW_PITCH - 1.6, *column_gaps) / 2
  if not 0 < clearance < half_aisle:
    raise ValueError("Clearance must be smaller than the narrowest aisle half-width")
  before, corner, after = points[:, :-2], points[:, 1:-1], points[:, 2:]
  incoming, outgoing = corner - before, after - corner
  len_in, len_out = incoming.norm(dim=-1), outgoing.norm(dim=-1)
  u = incoming / len_in.clamp_min(1e-8)[..., None]
  v = outgoing / len_out.clamp_min(1e-8)[..., None]
  cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
  dot = (u * v).sum(-1).clamp(-1, 1)
  angle = torch.atan2(cross, dot)
  turning = (angle.abs() > 1e-3) & (angle.abs() < torch.pi - 1e-3)
  half = angle.abs() / 2
  tan_half = torch.tan(half).clamp_min(1e-6)
  safe_radius = (half_aisle - clearance) / (1 - half.cos()).clamp_min(1e-6)
  r = torch.minimum(torch.full_like(angle, radius), safe_radius)
  trim = torch.minimum(r * tan_half, 0.45 * torch.minimum(len_in, len_out))
  r = trim / tan_half
  entry = corner - trim[..., None] * u
  left = torch.stack((-u[..., 1], u[..., 0]), -1)
  center = entry + (angle.sign() * r)[..., None] * left
  radial = entry - center
  t = torch.linspace(0, 1, arc_segments + 1, device=points.device, dtype=points.dtype)
  theta = angle[..., None] * t
  rx = radial[..., 0, None] * theta.cos() - radial[..., 1, None] * theta.sin()
  ry = radial[..., 0, None] * theta.sin() + radial[..., 1, None] * theta.cos()
  arc = center[..., None, :] + torch.stack((rx, ry), -1)
  arc = torch.where(turning[..., None, None], arc, corner[..., None, :])
  valid = turning[..., None].expand(-1, -1, arc_segments + 1).clone()
  valid[..., 0] = True
  candidates = torch.cat((points[:, :1], arc.flatten(1, 2), points[:, -1:]), dim=1)
  end_mask = torch.ones(len(points), 1, device=points.device, dtype=torch.bool)
  valid = torch.cat((end_mask, valid.flatten(1), end_mask), dim=1)
  counts = valid.sum(-1)
  slots = valid.long().cumsum(-1) - 1
  batch = torch.arange(len(points), device=points.device)[:, None].expand_as(valid)
  # Pad with the final point so all padded segment lengths are zero.
  packed = points[:, -1:, :].expand_as(candidates).clone()
  packed[batch[valid], slots[valid]] = candidates[valid]
  return packed, counts
