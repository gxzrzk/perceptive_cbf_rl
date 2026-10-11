"""静态教室环境布局与走廊路线规划模块 (Classroom Obstacle Navigation).

【背景与物理机制】
在复杂室内走廊导航与避障任务（如 G1 机器人在布置有桌椅排布的教室环境中穿行）中，
环境包含纵向主通道 (Longitudinal Aisles) 与横向交叉走道 (Cross Aisles)。
由于 Warp 会在编译期缓存静态几何体变换及其层次包围盒 (BVH)，如果运行时平移静态刚体，
会导致物理碰撞计算失效。因此，所有障碍物桌椅在编译期直接放置在世界固定位置，
各并行子环境共享该全局静态几何。

【核心算法实现】
1. 静态教室元数据初始化 (`initialize_static_classroom`)：
   缓存预定义的家具障碍物位姿列表 `layout` 到环境张量 `_wall_pos_w` 与 `_wall_yaw_w` 中。
2. 机器人安全出生点随机化 (`reset_classroom_robot`)：
   在参考状态初始化 (RSI) 之后，将机器人的水平面 (x, y, yaw) 投影重置到走廊安全走道内，
   同时保持 RSI 采样得到的关节姿态、躯干相对高度和倾角，并同步旋转线速度与角速度。
3. 走廊图随机游走路由生成 (`generate_classroom_route`)：
   在纵向与横向走道交叉网格图上执行带约束的随机游走 (Random Walk)，
   禁止立即掉头反向走，保证每次转弯的通道具有充足的安全通行净空 (~0.9m)。
4. 圆角切线平滑处理 (`round_classroom_route`)：
   机器人无法做到零半径瞬间折角转向。本算法将直角折线拐点替换为连续可导的圆弧切线切角，
   圆弧半径 $R$ 根据折角 $\\theta$ 与通道净宽约束 $R (1 - \\cos(\\theta/2)) \\le \\Delta_{\\text{clearance}}$ 动态受限，
   生成平滑、无碰撞且符合动力学跟踪极限的走廊巡航参考路径。
"""

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul
from .walk_path import _ensure_stash
from src.assets.objects.classroom import (
  CLASSROOM_SPAWN_LANES, CLASSROOM_ROUTE_STATIONS,
  CLASSROOM_ROW_PITCH, CLASSROOM_COLUMN_Y, CLASSROOM_ROUTE_CLEARANCE,
)


def initialize_static_classroom(env, env_ids, wall_names, layout):
  """在环境初始化时缓存静态教室障碍物的布局元数据.

  注意：静态碰撞几何体在编译时已固化在物理引擎中，Warp 缓存了其 BVH 层次包围树，
  因此所有并行环境必须使用原点零偏移 (env_spacing=0)。

  参数:
    env: 管理型强化学习环境
    env_ids: 待初始化的环境索引
    wall_names: 障碍物实体名称列表
    layout: 障碍物世界位姿列表 [(x, y, z, yaw), ...]
  """
  if torch.any(env.scene.env_origins != 0):
    raise ValueError("静态教室要求环境原点偏移全为 0 (env_spacing=0)")
  poses = torch.tensor(layout, device=env.device, dtype=torch.float32)
  if len(wall_names) != len(layout):
    raise ValueError("教室障碍物名称数量与布局位姿数量不一致")
  # 扩展并缓存障碍物的全局世界三维坐标与航向角
  env._wall_pos_w = poses[None, :, :3].expand(env.num_envs, -1, -1)
  env._wall_yaw_w = poses[None, :, 3].expand(env.num_envs, -1)


def reset_classroom_robot(env, env_ids):
  """在走廊安全通道内随机初始化机器人平面位姿并构建平滑巡航路径.

  保留动作捕捉参考状态 (RSI) 的身体高度、俯仰倾角和关节角度；
  将世界系线速度与角速度按照采样的航向增量进行严格旋转对齐。

  参数:
    env: 管理型强化学习环境
    env_ids: 触发重置的环境索引集合
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if len(env_ids) == 0:
    return
  n, device = len(env_ids), env.device
  robot = env.scene["robot"]
  indexing = robot.data.indexing
  # 从底层自由度切片读取根连杆自由浮动基座的 qpos (7维: pos(3)+quat(4)) 与 qvel (6维: lin(3)+ang(3))
  pose = env.sim.data.qpos[env_ids][:, indexing.free_joint_q_adr].clone()
  velocity = env.sim.data.qvel[env_ids][:, indexing.free_joint_v_adr].clone()
  
  # 随机选择走廊通道 (Spawn Lanes)
  lane = torch.randint(0, len(CLASSROOM_SPAWN_LANES), (n,), device=device)
  lane_y = torch.tensor(CLASSROOM_SPAWN_LANES, device=device)[lane]
  # 在可用通道长度范围内均匀随机采样 x 坐标
  x = torch.rand(n, device=device) * CLASSROOM_ROUTE_STATIONS[-1]
  # 在通道中心线附近添加 ±0.15m 的横向扰动
  y = lane_y + (torch.rand(n, device=device) * 2 - 1) * 0.15
  origins = env.scene.env_origins[env_ids]
  pose[:, :2] = origins[:, :2] + torch.stack((x, y), -1)
  
  # 计算并随机化偏航角 yaw
  q = pose[:, 3:7]
  old_yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                        1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  yaw = (torch.rand(n, device=device) * 2 - 1) * torch.pi
  # 构造绕 z 轴旋转的相对四元数增量
  delta = torch.zeros(n, 4, device=device)
  delta[:, 0] = ((yaw - old_yaw) / 2).cos()
  delta[:, 3] = ((yaw - old_yaw) / 2).sin()
  pose[:, 3:7] = quat_mul(delta, q)
  
  # 对世界系线速度与角速度进行四元数旋转变换
  # 根节点线速度为世界坐标系；自由关节角速度在局部坐标系，先由四元数变换同步
  velocity[:, :3] = quat_apply(delta, velocity[:, :3])
  velocity[:, 3:] = quat_apply(pose[:, 3:7], velocity[:, 3:])
  # 将重置后的位姿与速度直接写入物理模拟器底层
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  # 基于当前坐标生成带圆角平滑的教室走廊路径点序列
  local, counts = round_classroom_route(generate_classroom_route(x, y, lane))
  count = local.shape[1]
  # 确保路径缓存容量充足
  _ensure_stash(env, max_points=count)
  # 写入世界坐标路径点与累积弧长
  env._walk_path_pts[env_ids, :count] = local + origins[:, None, :2]
  lengths = torch.linalg.vector_norm(local[:, 1:] - local[:, :-1], dim=-1)
  env._walk_path_cumlen[env_ids, 0] = 0
  env._walk_path_cumlen[env_ids, 1:count] = lengths.cumsum(-1)
  env._walk_path_n[env_ids] = counts
  # 递增路径轮次代号，触发上层导航指令管理器同步重置跟踪状态
  env._walk_path_epoch[env_ids] += 1


def generate_classroom_route(x, y, lane, count=80):
  """在教室的纵向主通道与横向交叉过道构成的拓扑图上生成随机游走路径.

  仅向相邻交叉路口前进，严禁立刻原地回头。
  每个通道中心线距离两侧桌椅均有约 0.9m 的充足安全间距。

  参数:
    x: (n,) 机器人当前 x 坐标
    y: (n,) 机器人当前 y 坐标
    lane: (n,) 当前通道索引
    count: 采样的总路径关键点数量 (默认 80)

  返回:
    local: (n, count, 2) 本地折线路径点坐标序列
  """
  n, device = len(x), x.device
  # 横向交叉过道中心 x 坐标列表 (位于前排课桌与后排座椅背之间)
  stations = torch.tensor(CLASSROOM_ROUTE_STATIONS, device=device)
  # 纵向通道 y 坐标列表
  lanes = torch.tensor(CLASSROOM_SPAWN_LANES, device=device)
  lane_idx = lane.clone()
  difference = stations[None] - x[:, None]
  
  # 随机选择第一段是向前走还是向后走，并要求第一段移动距离至少 >= 0.8m
  forward = torch.rand(n, device=device) < 0.5
  valid = torch.where(forward[:, None], difference >= 0.8, difference <= -0.8)
  fallback = difference.abs() >= 0.8
  valid = torch.where(valid.any(-1, keepdim=True), valid, fallback)
  station_idx = difference.abs().masked_fill(~valid, float('inf')).argmin(-1)
  
  local = torch.zeros(n, count, 2, device=device)
  local[:, 0] = torch.stack((x, y), -1)
  local[:, 1] = torch.stack((stations[station_idx], lanes[lane_idx]), -1)
  previous = torch.full((n,), -1, device=device, dtype=torch.long)
  
  # 逐步随机扩展路径拓扑节点
  for i in range(2, count):
    # 可选邻居：在当前通道前后移动，或在当前路口向左/右相邻通道变道
    sx = torch.stack((station_idx - 1, station_idx + 1, station_idx, station_idx), -1)
    ly = torch.stack((lane_idx, lane_idx, lane_idx - 1, lane_idx + 1), -1)
    # 筛选合法的节点编号（必须在走廊范围内且不能是上一跳的原路折返）
    legal = (sx >= 0) & (sx < len(stations)) & (ly >= 0) & (ly < len(lanes))
    node = sx * len(lanes) + ly
    legal &= node != previous[:, None]
    if i == 2:
      first_forward = stations[station_idx] > x
      legal[:, 0] &= ~first_forward
      legal[:, 1] &= first_forward
    # 从所有合法邻居中均匀随机抽取下一个路口
    choice = torch.rand(n, 4, device=device).masked_fill(~legal, -1).argmax(-1)
    previous = station_idx * len(lanes) + lane_idx
    station_idx = sx.gather(1, choice[:, None]).squeeze(-1)
    lane_idx = ly.gather(1, choice[:, None]).squeeze(-1)
    local[:, i] = torch.stack((stations[station_idx], lanes[lane_idx]), -1)
  return local


def round_classroom_route(points, radius=0.6, arc_segments=6,
                          clearance=CLASSROOM_ROUTE_CLEARANCE):
  """将折线路径中的尖锐拐角替换为连续相切的圆弧线段，返回打包装箱后的平滑路径点集.

  物理与几何机制：
  - 转向圆弧半径受到两段相邻线段长度以及通道可用净宽的严格几何限制；
  - 在转弯顶点处，圆弧偏离原折线顶点的最大距离为 R * (1 - cos(theta / 2))；
  - 严格限制该距离，确保圆弧拐弯轨迹全程保持在无碰撞的安全走廊包络线内。

  参数:
    points: (n, count, 2) 原始折线航路点
    radius: 期望转弯圆弧半径 (米，默认 0.6m)
    arc_segments: 每个圆角离散化为的小线段数量 (默认 6 段)
    clearance: 避障走廊净空边距要求

  返回:
    (packed, counts):
      - packed: (n, M, 2) 平滑化后的密集路径点张量
      - counts: (n,) 每个环境实际有效的有效路径点总数
  """
  if radius <= 0 or arc_segments < 2 or points.shape[1] < 3:
    raise ValueError("需要正的圆角半径、至少 2 个弧段以及至少 3 个折线点")
  # 计算通道柱间空隙净宽
  column_gaps = [b - a - 1.2 for a, b in zip(CLASSROOM_COLUMN_Y, CLASSROOM_COLUMN_Y[1:])]
  half_aisle = min(CLASSROOM_ROW_PITCH - 1.6, *column_gaps) / 2
  if not 0 < clearance < half_aisle:
    raise ValueError("安全通道净空必须严格小于最窄走道的半宽")
    
  # 提取三联拐点 (before, corner, after)
  before, corner, after = points[:, :-2], points[:, 1:-1], points[:, 2:]
  incoming, outgoing = corner - before, after - corner
  len_in, len_out = incoming.norm(dim=-1), outgoing.norm(dim=-1)
  u = incoming / len_in.clamp_min(1e-8)[..., None]
  v = outgoing / len_out.clamp_min(1e-8)[..., None]
  
  # 计算拐弯转向夹角 theta
  cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
  dot = (u * v).sum(-1).clamp(-1, 1)
  angle = torch.atan2(cross, dot)
  turning = (angle.abs() > 1e-3) & (angle.abs() < torch.pi - 1e-3)
  half = angle.abs() / 2
  tan_half = torch.tan(half).clamp_min(1e-6)
  
  # 根据通道净宽几何边界求解允许的最大安全半径
  safe_radius = (half_aisle - clearance) / (1 - half.cos()).clamp_min(1e-6)
  r = torch.minimum(torch.full_like(angle, radius), safe_radius)
  # 切线切入长度不能超过两侧线段总长的 45%
  trim = torch.minimum(r * tan_half, 0.45 * torch.minimum(len_in, len_out))
  r = trim / tan_half
  
  # 求解圆弧圆心与切点位置
  entry = corner - trim[..., None] * u
  left = torch.stack((-u[..., 1], u[..., 0]), -1)
  center = entry + (angle.sign() * r)[..., None] * left
  radial = entry - center
  
  # 生成离散圆弧插值点
  t = torch.linspace(0, 1, arc_segments + 1, device=points.device, dtype=points.dtype)
  theta = angle[..., None] * t
  rx = radial[..., 0, None] * theta.cos() - radial[..., 1, None] * theta.sin()
  ry = radial[..., 0, None] * theta.sin() + radial[..., 1, None] * theta.cos()
  arc = center[..., None, :] + torch.stack((rx, ry), -1)
  arc = torch.where(turning[..., None, None], arc, corner[..., None, :])
  
  # 打包装箱输出紧凑点阵张量
  valid = turning[..., None].expand(-1, -1, arc_segments + 1).clone()
  valid[..., 0] = True
  candidates = torch.cat((points[:, :1], arc.flatten(1, 2), points[:, -1:]), dim=1)
  end_mask = torch.ones(len(points), 1, device=points.device, dtype=torch.bool)
  valid = torch.cat((end_mask, valid.flatten(1), end_mask), dim=1)
  counts = valid.sum(-1)
  slots = valid.long().cumsum(-1) - 1
  batch = torch.arange(len(points), device=points.device)[:, None].expand_as(valid)
  # 末端填充最后一个有效点，使填充段的相对长度为 0
  packed = points[:, -1:, :].expand_as(candidates).clone()
  packed[batch[valid], slots[valid]] = candidates[valid]
  return packed, counts
