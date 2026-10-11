"""随机巡航漫游路径 (Random Walk Path) 生成、动态延长与弧长查询模块 (WallWalk 避障任务).

【背景与架构设计】
在墙体避障穿行任务 (WallWalk) 中，每个并行环境在每个回合拥有**各自独立的随机折线参考轨迹**：
这是一条由 2D 平面多段折线组成的路径。它起始于机器人重置后的 (x, y) 位置，初始切向与机器人的偏航角对齐，
随后以温和的随机偏转角向前蜿蜒延展（每段长度 $L \\sim U(\\text{seg\\_len\\_range})$，每段偏航角增量 $\\Delta \\psi \\sim U(\\pm \\text{turn\\_max})$）。

【控制与感知交互机制】
1. 路径并不直接输入策略的观测空间！
   而是由导航指令项 (`DodgeGoToGoalCommand.path_follow`) 采用**纯追踪 (Pure Pursuit)** 机制，
   提取沿路径前方特定前视距离 (`path_lookahead`) 处的几何点作为局部“目标”，
   上层 P 控制器将其转化为机器人躯干速度指令 ($v_x, v_y, \\omega_z$)。
2. 墙体事件 (`reset_wall_pose` 与 `recycle_walls_ahead`) 沿该路径的弧长站点 (Arc-length Stations)
   动态放置或回收循环障碍墙，使走廊障碍物始终与路径贴合。
3. 共享环境缓存张量 (Shared Environment Stash)：
   - `_walk_path_pts`: (N, MAX_POINTS, 2) 顶点全局世界二维坐标；
   - `_walk_path_cumlen`: (N, MAX_POINTS) 各顶点的累积弧长 (Cumulative Arc-Length)；
   - `_walk_path_n`: (N,) 当前有效顶点数量；
   - `_walk_path_epoch`: (N,) 路径更新代号（每次重置路径 +1，上层指令管理器据此清空历史跟踪进度）；
   - `_walk_path_params`: 记录路径生成参数。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

WALK_PATH_MAX_POINTS = 160
"""每个环境最大的路径顶点存储容量。
以每段平均 3m 计算，总容量达 ~480m，远超单个训练回合机器人的移动总程 (~26m)。
设置容量上限是为了防止长时间运行导致显存失控。"""

_WALK_PATH_PTS_ATTR = "_walk_path_pts"
_WALK_PATH_CUMLEN_ATTR = "_walk_path_cumlen"
_WALK_PATH_N_ATTR = "_walk_path_n"
_WALK_PATH_EPOCH_ATTR = "_walk_path_epoch"
_WALK_PATH_PARAMS_ATTR = "_walk_path_params"

__all__ = [
  "WALK_PATH_MAX_POINTS",
  "generate_walk_path",
  "extend_walk_path",
  "walk_path_query",
]


def _ensure_stash(env: "ManagerBasedRlEnv", max_points: int = WALK_PATH_MAX_POINTS) -> None:
  """确保环境对象上已分配充足维度的路径张量缓存，并在需要时扩容."""
  pts = getattr(env, _WALK_PATH_PTS_ATTR, None)
  if pts is None or pts.shape[0] != env.num_envs:
    device = env.device
    capacity = max(WALK_PATH_MAX_POINTS, max_points)
    setattr(env, _WALK_PATH_PTS_ATTR, torch.zeros(env.num_envs, capacity, 2, device=device))
    setattr(env, _WALK_PATH_CUMLEN_ATTR, torch.zeros(env.num_envs, capacity, device=device))
    setattr(env, _WALK_PATH_N_ATTR, torch.zeros(env.num_envs, dtype=torch.long, device=device))
    setattr(env, _WALK_PATH_EPOCH_ATTR, torch.zeros(env.num_envs, dtype=torch.long, device=device))
  elif pts.shape[1] < max_points:
    # 当教室环境需要更密集的平滑圆角点阵时，动态扩容并保留已有数据
    old_capacity = pts.shape[1]
    expanded = pts.new_zeros(env.num_envs, max_points, 2)
    expanded[:, :old_capacity] = pts
    cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)
    expanded_cum = cum.new_zeros(env.num_envs, max_points)
    expanded_cum[:, :old_capacity] = cum
    setattr(env, _WALK_PATH_PTS_ATTR, expanded)
    setattr(env, _WALK_PATH_CUMLEN_ATTR, expanded_cum)


def _fresh_root_yaw(env: "ManagerBasedRlEnv", robot: "Entity", env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """直接从底层物理引擎 qpos 读取最新的机器人根节点水平位置与偏航角 yaw.

  注意：在环境 reset 事件连续触发期间，`robot.data` 缓存尚未刷新，
  因此必须直接穿透读取 `env.sim.data.qpos`，确保获取刚重置的瞬时位姿。
  """
  adr = robot.data.indexing.free_joint_q_adr
  root_pose = env.sim.data.qpos[env_ids][:, adr]  # (n, 7): [x, y, z, qw, qx, qy, qz]
  q = root_pose[:, 3:7]
  # 由四元数计算偏航角 yaw
  yaw = torch.atan2(
    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
  )
  return root_pose[:, :2], yaw


def _append_segments(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  start_xy: torch.Tensor,
  start_yaw: torch.Tensor,
  num_segments: int,
  seg_len_range: tuple[float, float],
  turn_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """从给定起点和初始朝向延伸生成 num_segments 段随机折线.

  返回:
    pts: (n, num_segments, 2) 各段末端顶点的二维世界坐标
    seg_len: (n, num_segments) 每小段的实际空间欧式长度
  """
  n = len(env_ids)
  device = env.device
  # 均匀随机采样每段线段长度
  seg_len = torch.rand(n, num_segments, device=device) * (
    seg_len_range[1] - seg_len_range[0]
  ) + seg_len_range[0]
  # 均匀随机采样偏航转角扰动
  dyaw = (torch.rand(n, num_segments, device=device) * 2.0 - 1.0) * turn_max
  # 排除式累加：第 0 段严格保持初始朝向 start_yaw（确保轨迹与出生位姿正交/切向平滑衔接），
  # 从第 1 段起才逐段累加转向
  yaw = start_yaw.unsqueeze(1) + (torch.cumsum(dyaw, dim=1) - dyaw)
  # 计算每小段的平面位移增量向量
  step = torch.stack([seg_len * torch.cos(yaw), seg_len * torch.sin(yaw)], dim=-1)
  # 累加得到绝对世界坐标
  pts = start_xy.unsqueeze(1) + torch.cumsum(step, dim=1)
  return pts, seg_len


def generate_walk_path(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  robot_name: str = "robot",
  seg_len_range: tuple[float, float] = (2.0, 4.0),
  turn_max: float = 0.45,
  initial_len: float = 80.0,
) -> None:
  """在环境重置 (Episode Reset) 时，为指定环境重新生成初始随机折线路径.

  路径从机器人的当前重置位置和偏航角出发，延伸至少 initial_len 米；
  单段偏航转角 turn_max=0.45 rad / 2~4m 对应约 4.4~8.9m 的转弯半径，
  符合机器人在 1.3m/s 航速和 1.0 rad/s 偏航角速度限制下的动力学跟踪包络。
  """
  if len(env_ids) == 0:
    return
  _ensure_stash(env)
  robot: Entity = env.scene[robot_name]
  n = len(env_ids)
  device = env.device

  # 计算所需线段数
  num_segments = min(
    WALK_PATH_MAX_POINTS - 1, math.ceil(initial_len / seg_len_range[0])
  )
  start_xy, start_yaw = _fresh_root_yaw(env, robot, env_ids)
  pts_new, seg_len = _append_segments(
    env, env_ids, start_xy, start_yaw, num_segments, seg_len_range, turn_max
  )

  pts = getattr(env, _WALK_PATH_PTS_ATTR)
  cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)
  count = getattr(env, _WALK_PATH_N_ATTR)

  # 起点为机器人当前出生点，弧长为 0
  pts[env_ids, 0] = start_xy
  pts[env_ids, 1 : 1 + num_segments] = pts_new
  cum[env_ids, 0] = 0.0
  cum[env_ids, 1 : 1 + num_segments] = torch.cumsum(seg_len, dim=1)
  count[env_ids] = 1 + num_segments
  # 递增 epoch 告知指令管理器重置累积进度
  getattr(env, _WALK_PATH_EPOCH_ATTR)[env_ids] += 1
  setattr(env, _WALK_PATH_PARAMS_ATTR, (seg_len_range[0], seg_len_range[1], turn_max))


def extend_walk_path(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  min_total_len: torch.Tensor,
) -> None:
  """当机器人的前视目标或墙体放置范围超出当前路径总长时，沿末端航向动态追加折线段.

  保持与初始生成相同的曲率与线段长度分布。
  注意：这属于同一回合内的路径追加，不改变 epoch 代号。
  """
  if len(env_ids) == 0:
    return
  _ensure_stash(env)
  params = getattr(env, _WALK_PATH_PARAMS_ATTR, (2.0, 4.0, 0.45))
  seg_lo, seg_hi, turn_max = params
  pts = getattr(env, _WALK_PATH_PTS_ATTR)
  cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)
  count = getattr(env, _WALK_PATH_N_ATTR)

  # 循环批量追加，直到满足 min_total_len 或达到容量上限
  for _ in range(8):
    total = cum[env_ids, count[env_ids] - 1]
    need = (total < min_total_len) & (count[env_ids] <= WALK_PATH_MAX_POINTS - 17)
    if not need.any():
      return
    ids = env_ids[need]
    n = len(ids)
    device = env.device
    K = 16  # 每次批量追加 16 个线段
    c = count[ids]
    last = pts[ids, c - 1]
    prev = pts[ids, c - 2]
    last_yaw = torch.atan2(last[:, 1] - prev[:, 1], last[:, 0] - prev[:, 0])
    pts_new, seg_len = _append_segments(
      env, ids, last, last_yaw, K, (seg_lo, seg_hi), turn_max
    )
    # 将新生成的顶点追加写入对应环境的张量切片中
    slots = c.unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0)
    pts[ids.unsqueeze(1), slots] = pts_new
    cum_new = total[need].unsqueeze(1) + torch.cumsum(seg_len, dim=1)
    cum[ids.unsqueeze(1), slots] = cum_new
    count[ids] = c + K


def walk_path_query(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """根据给定的沿路径累积弧长 s 查询对应的 2D 坐标与切线航向角.

  参数:
    env: 管理型强化学习环境
    env_ids: (n,) 查询的环境编号
    s: (n,) 目标弧长坐标 (米)

  返回:
    pos: (n, 2) 路径上对应位置的 (x, y) 坐标
    yaw: (n,) 路径在该点所在线段的切向航向角 (rad)
  """
  pts = getattr(env, _WALK_PATH_PTS_ATTR)[env_ids]
  cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)[env_ids]
  count = getattr(env, _WALK_PATH_N_ATTR)[env_ids]
  n = len(env_ids)
  ar = torch.arange(n, device=env.device)

  total = cum[ar, count - 1]
  # 将查询弧长截断至 [0, 总长度]
  s = s.clamp(min=torch.zeros_like(total), max=total)
  
  # 二分查找包含弧长 s 的线段索引
  valid = (
    torch.arange(cum.shape[1], device=env.device).unsqueeze(0) < count.unsqueeze(1)
  )
  cum_search = torch.where(valid, cum, torch.full_like(cum, float("inf")))
  idx = torch.searchsorted(cum_search, s.unsqueeze(1), right=True).squeeze(1) - 1
  idx = idx.clamp(min=torch.zeros_like(count), max=(count - 2).clamp(min=0))

  p0 = pts[ar, idx]
  p1 = pts[ar, idx + 1]
  c0 = cum[ar, idx]
  c1 = cum[ar, idx + 1]
  # 在该线段内部进行线性插值
  t = ((s - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0.0, 1.0)
  pos = p0 + t.unsqueeze(-1) * (p1 - p0)
  # 计算切线方向角
  yaw = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
  return pos, yaw
