"""目标驱动的导航速度指令管理器模块 (Goal-Conditioned Navigation Commands).

【架构定位与理论机制】
在腿足机器人的分层控制架构中，导航决策与底层步态运动控制通常进行解耦：
1. 目标对策略的隐式呈现 (Goal Absent from Observations)：
   本模块实现的指令项并不直接将目标三维坐标放入策略网络的观测空间，而是通过航向与位置误差计算，
   实时转化为机器人机体坐标系下的名义速度指令 $[v_x, v_y, \\omega_z]$。策略网络只需像响应遥控手柄 (Joystick) 一样
   执行速度跟踪。
2. 比例衰减机制 (Proportional Decay Cue)：
   速度指令大小正比于机器人距目标的剩余欧氏距离：
       $v_{xy, b} = \\text{clip}(k_p \\cdot R_{\\text{yaw}}^{-1} (p_{\\text{goal}} - p_{\\text{base}}), v_{\\max})$
       $\\omega_z = \\text{clip}(k_{p, \\text{yaw}} \\cdot \\text{wrap}(\\text{atan2}(p_{\\text{goal}} - p_{\\text{base}}) - \\psi), \\omega_{\\max})$
   当机器人接近目标时，速度指令自动衰减归零，从而使无记忆的马尔可夫策略能够自然减速并精确悬停在目标点。
3. 预测性控制屏障函数 (CBF) 躲避滤波 (`DodgeGoToGoalCommand`)：
   在躲避球任务中，名义导航速度在下发前通过预测性 CBF 安全滤波器 (`cbf.py`) 处理。
   若来球处于威胁航向，滤波器自动将速度指令沿来球航向的法向叠加剧烈侧移量，迫使机器人飞跃避让。
   在纯 CBF-RL 训练模式下，滤波结果可仅作为特权安全奖励指导（不直接改写指令），强迫策略依靠自感知视觉学习躲避。
4. 路径纯追踪 (Path Follow Pure Pursuit)：
   在走廊避障任务中，指令器提取前方前视点作为动态目标，引导机器人沿着随机折线路径平滑巡航。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  wrap_to_pi,
  yaw_quat,
)

from src.assets.objects.ball import DEFAULT_BALL_RADIUS
from src.tasks.amp_loco.mdp.cbf import predictive_dodge_filter
from src.tasks.amp_loco.mdp.walk_path import extend_walk_path, walk_path_query

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class GoToGoalCommand(CommandTerm):
  """将采样的平面目标点转化为机体系速度指令的基础指令管理器 (目标点对策略隐式隐藏)."""

  cfg: GoToGoalCommandCfg

  def __init__(self, cfg: GoToGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # 世界坐标系下的目标位置 (xy) 与计算输出的机体系速度指令 [vx, vy, wz]
    self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.distance_to_goal = torch.zeros(self.num_envs, device=self.device)
    # 单步向目标靠近的距离差值 (米)，正值表示靠近，供 goal_progress_reward 奖励函数使用
    self.prev_distance = torch.zeros(self.num_envs, device=self.device)
    self.distance_delta = torch.zeros(self.num_envs, device=self.device)
    # 纯静止环境掩码 (强制速度指令全为 0 且不扔球，作为静止站立步态锚点)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # 原地等待来球环境掩码 (速度指令为 0，但依然会向其扔球，强迫策略学习从完全静止状态快速启动侧跃避让)
    self.is_inplace_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # 驻留与重采样计数器：当机器人在目标点停驻达到随机时长后，重新采样新目标
    self.dwell_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.dwell_steps = torch.full(
      (self.num_envs,),
      max(1, round(sum(self.cfg.dwell_time_range) / 2 / self._env.step_dt)),
      dtype=torch.long,
      device=self.device,
    )

    self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["fraction_at_goal"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """返回策略网络观测到的机体系线速度与角速度指令 [vx, vy, wz]."""
    return self.vel_command_b

  def _sample_stand_inplace(self, env_ids: torch.Tensor) -> torch.Tensor:
    """按比例将重置环境划分为：永久站立环境、原地避让环境和普通移动目标跟踪环境."""
    r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
    stand_p = self.cfg.rel_standing_envs
    inplace_p = self.cfg.rel_inplace_throw_envs
    standing = r < stand_p
    inplace = (r >= stand_p) & (r < stand_p + inplace_p)
    self.is_standing_env[env_ids] = standing
    self.is_inplace_env[env_ids] = inplace
    return standing

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """在指定环境集合中重新采样随机目标点."""
    n = len(env_ids)
    if n == 0:
      return
    r = torch.empty(n, device=self.device)
    radius = r.uniform_(*self.cfg.radius)
    angle = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    base_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    # 在以机器人当前位置为圆心、在给定半径区间内随机采样新目标位置
    self.goal_pos_w[env_ids, 0] = base_xy[:, 0] + radius * torch.cos(angle)
    self.goal_pos_w[env_ids, 1] = base_xy[:, 1] + radius * torch.sin(angle)
    # 对于纯站立环境，将目标点固定为当前位置
    standing = self._sample_stand_inplace(env_ids)
    self.goal_pos_w[env_ids[standing]] = base_xy[standing]
    # 初始化到新目标的距离并清空停驻计时器
    self.prev_distance[env_ids] = torch.norm(
      self.goal_pos_w[env_ids] - base_xy, dim=-1
    )
    self.dwell_counter[env_ids] = 0
    dwell_s = torch.empty(n, device=self.device).uniform_(*self.cfg.dwell_time_range)
    self.dwell_steps[env_ids] = (dwell_s / self._env.step_dt).round().long().clamp(min=1)

  def _update_command(self) -> None:
    """每控制步更新目标距离、计算 P 控制机体系速度指令并处理停驻重采样."""
    # 判定是否已进入到达圆域
    reached = (
      (self.distance_to_goal < self.cfg.arrive_radius)
      & ~self.is_standing_env
      & ~self.is_inplace_env
    )
    self.dwell_counter = torch.where(
      reached, self.dwell_counter + 1, torch.zeros_like(self.dwell_counter)
    )
    # 达到停驻步数阈值后触发重采样
    resample_ids = (self.dwell_counter >= self.dwell_steps).nonzero(as_tuple=False).flatten()
    if len(resample_ids) > 0:
      self._resample(resample_ids)

    base_pos_w = self.robot.data.root_link_pos_w[:, :2]
    err_w = self.goal_pos_w - base_pos_w  # (N, 2) 世界系位置误差
    self.distance_to_goal = torch.norm(err_w, dim=-1)

    # 将世界系水平位置误差旋转到机器人当前偏航系 (Base Yaw Frame)
    err_w3 = torch.cat([err_w, torch.zeros_like(err_w[:, :1])], dim=-1)
    err_b = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w), err_w3
    )[:, :2]

    # 计算比例速度指令并进行轴向限幅截断
    v = self.cfg.kp * err_b
    v[:, 0] = v[:, 0].clamp(-self.cfg.max_lin_vel_x, self.cfg.max_lin_vel_x)
    v[:, 1] = v[:, 1].clamp(-self.cfg.max_lin_vel_y, self.cfg.max_lin_vel_y)

    # 偏航角速度：计算朝向目标所需的旋转角速度
    if self.cfg.simple_heading:
      heading_target = torch.atan2(err_w[:, 1], err_w[:, 0])
      yaw_err = wrap_to_pi(heading_target - self.robot.data.heading_w)
      wz = (self.cfg.kp_yaw * yaw_err).clamp(
        -self.cfg.max_ang_vel_z, self.cfg.max_ang_vel_z
      )
    else:
      wz = torch.zeros_like(self.distance_to_goal)

    # 到达目标点判定：进入 arrive_radius 范围内强制置零速度指令，促使原地平稳站立
    arrived = self.distance_to_goal < self.cfg.arrive_radius
    v[arrived] = 0.0
    wz[arrived] = 0.0

    self.vel_command_b[:, :2] = v
    self.vel_command_b[:, 2] = wz
    # 对静止环境和原地避让环境强制设为 0 速度
    self.vel_command_b[self.is_standing_env] = 0.0
    self.vel_command_b[self.is_inplace_env] = 0.0

    # 记录本步向目标靠近的净位移量
    self.distance_delta = self.prev_distance - self.distance_to_goal
    self.prev_distance = self.distance_to_goal.clone()

  def _update_metrics(self) -> None:
    """更新评估指标."""
    self.metrics["distance_to_goal"] = self.distance_to_goal
    self.metrics["fraction_at_goal"] = (
      self.distance_to_goal < self.cfg.arrive_radius
    ).float()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """在可视化查看器 (Viser/Viewer) 中绘制目标点橙色球体与指向箭头."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    goals = self.goal_pos_w.cpu().numpy()
    base = self.robot.data.root_link_pos_w.cpu().numpy()
    for batch in env_indices:
      if np.linalg.norm(base[batch]) < 1e-6:
        continue
      goal_xyz = np.array([goals[batch, 0], goals[batch, 1], 0.05])
      visualizer.add_sphere(goal_xyz, 0.12, color=(0.9, 0.3, 0.1, 0.8))
      visualizer.add_arrow(
        base[batch] + np.array([0, 0, 0.1]),
        goal_xyz + np.array([0, 0, 0.1]),
        color=(0.9, 0.6, 0.1, 0.5),
        width=0.01,
      )


@dataclass(kw_only=True)
class GoToGoalCommandCfg(CommandTermCfg):
  """GoToGoalCommand 参数配置项."""

  entity_name: str
  radius: tuple[float, float] = (1.0, 3.0)
  """新目标距离机器人根节点的最小/最大采样半径 (米)。"""
  rel_standing_envs: float = 0.05
  """纯静止环境比例 (无速度指令且不投球)。"""
  rel_inplace_throw_envs: float = 0.0
  """原地避球环境比例 (无速度指令但依然投球，训练从静止起步避碰)。"""
  kp: float = 1.5
  """平面位置误差转换为线速度指令的比例增益 (m/s / m)。"""
  kp_yaw: float = 1.0
  """偏航角误差转换为偏航角速度指令的比例增益 (rad/s / rad)。"""
  max_lin_vel_x: float = 2.0
  max_lin_vel_y: float = 1.0
  max_ang_vel_z: float = 1.0
  arrive_radius: float = 0.25
  """到达判定半径 (米)。进入该距离后速度指令置零。"""
  dwell_time_range: tuple[float, float] = (1.0, 3.0)
  """到达目标后停留驻留时间范围 (秒)。"""
  simple_heading: bool = True
  """是否下发偏航角速度指令以转向对齐目标方向。"""

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoToGoalCommand:
    return GoToGoalCommand(self, env)


class DodgeGoToGoalCommand(GoToGoalCommand):
  """集成控制屏障函数 (CBF) 轨迹预测避险与折线路径纯追踪的高级指令管理器."""

  cfg: DodgeGoToGoalCommandCfg

  def __init__(self, cfg: DodgeGoToGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if cfg.turn_before_walk:
      if not cfg.simple_heading:
        raise ValueError("turn_before_walk 需要启用 simple_heading")
      if not 0 <= cfg.walk_full_speed_angle < cfg.walk_stop_angle <= math.pi / 2:
        raise ValueError("行走角度阈值必须满足 0 <= full < stop <= pi/2")
    self.ball: Entity = env.scene[cfg.ball_name]
    self._walk_forward_scale = torch.ones(self.num_envs, device=self.device)
    # 获取球体碰撞几何体 ID，用于读取每回合随机化后的真实球半径
    self._ball_geom_id = env.sim.mj_model.geom(f"{cfg.ball_name}/ball_collision").id
    # 固定母港目标点 (Home Goal): 记录各环境的原点出生坐标
    self._home_xy = env.scene.env_origins[:, :2].clone()
    # 机体系前向基准单位向量 [1, 0, 0]
    self._forward_b = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
      self.num_envs, 1
    )
    # 名义未滤波机体系速度指令暂存
    self.vel_command_nominal_b = torch.zeros(self.num_envs, 3, device=self.device)
    # 锁定的逃避方向与威胁标志
    self._dodge_side = torch.ones(self.num_envs, device=self.device)
    self._dodge_threat = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    # CBF-RL 奖励所需的状态暂存变量 (供 rewards.py 读取):
    # - _dodge_e_w: 世界坐标系逃逸方向 (屏障函数关于速度的雅可比梯度 nabla_v h)
    # - _dodge_h: 屏障函数当前安全裕度 s - D
    # - _dodge_u_safe_w: CBF 截断后的世界系理论安全速度
    # - _dodge_tti: 预估撞击剩余时间 TTI (秒)
    self._dodge_e_w = torch.zeros(self.num_envs, 2, device=self.device)
    self._dodge_h = torch.zeros(self.num_envs, device=self.device)
    self._dodge_u_safe_w = torch.zeros(self.num_envs, 2, device=self.device)
    self._dodge_tti = torch.zeros(self.num_envs, device=self.device)
    self._dodge_ball_radius = torch.full(
      (self.num_envs,), DEFAULT_BALL_RADIUS, device=self.device
    )

    # 路径纯追踪状态变量 (WallWalk 任务):
    self._path_seg_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._path_s = torch.zeros(self.num_envs, device=self.device)
    self._path_s_delta = torch.zeros(self.num_envs, device=self.device)
    self._path_lat_err = torch.zeros(self.num_envs, device=self.device)
    self._path_epoch_seen = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    self._PATH_VIZ_SEGMENTS = 40
    self.metrics["cbf_active_frac"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["cbf_min_h"] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """重采样指令：支持固定母港点 (home_goal) 模式."""
    if not self.cfg.home_goal:
      super()._resample_command(env_ids)
      return
    if len(env_ids) == 0:
      return
    # 固定母港模式：将目标点钉死在出生原点，闪避飞跃后机器人会自动返航
    self.goal_pos_w[env_ids] = self._home_xy[env_ids]
    self._sample_stand_inplace(env_ids)
    base_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    self.prev_distance[env_ids] = torch.norm(self.goal_pos_w[env_ids] - base_xy, dim=-1)
    self.dwell_counter[env_ids] = 0

  def _update_path_goal(self) -> None:
    """纯追踪 (Pure Pursuit) 更新：沿随机折线路径前向投射目标点."""
    env = self._env
    pts = env._walk_path_pts
    cum = env._walk_path_cumlen
    count = env._walk_path_n
    ar = torch.arange(self.num_envs, device=self.device)

    # 检测到新路径代号时重置跟踪进度
    new_path = env._walk_path_epoch != self._path_epoch_seen
    self._path_seg_idx = torch.where(
      new_path, torch.zeros_like(self._path_seg_idx), self._path_seg_idx
    )
    self._path_s = torch.where(new_path, torch.zeros_like(self._path_s), self._path_s)
    self._path_epoch_seen = env._walk_path_epoch.clone()

    robot_xy = self.robot.data.root_link_pos_w[:, :2]
    allow_backtracking = getattr(self.cfg, "path_allow_backtracking", False)

    # 局部更新线段索引，支持拐角自适应切换与倒退检测
    for _ in range(8):
      i = torch.minimum(self._path_seg_idx, (count - 2).clamp(min=0))
      p0 = pts[ar, i]
      p1 = pts[ar, i + 1]
      seg = p1 - p0
      l2 = (seg * seg).sum(dim=-1).clamp_min(1e-9)
      u = ((robot_xy - p0) * seg).sum(dim=-1) / l2
      
      # 转角平滑衔接判定
      has_next = i < count - 2
      p2 = pts[ar, torch.minimum(i + 2, count - 1)]
      next_seg = p2 - p1
      next_u = ((robot_xy - p1) * next_seg).sum(-1) / (
        next_seg.square().sum(-1).clamp_min(1e-9)
      )
      current_closest = p0 + u.clamp(0, 1)[:, None] * seg
      next_closest = p1 + next_u.clamp(0, 1)[:, None] * next_seg
      current_dist = (robot_xy - current_closest).norm(dim=-1)
      next_dist = (robot_xy - next_closest).norm(dim=-1)
      near_corner = (robot_xy - p1).norm(dim=-1) <= self.cfg.path_switch_radius
      entered_next = (next_u > 0.0) & (next_dist + 1e-4 < current_dist)
      adv = ((u >= 1.0) | (near_corner & entered_next)) & has_next
      back = torch.zeros_like(adv)
      if allow_backtracking:
        prev0 = pts[ar, (i - 1).clamp(min=0)]
        prev_seg = p0 - prev0
        prev_u = ((robot_xy - prev0) * prev_seg).sum(-1) / (
          prev_seg.square().sum(-1).clamp_min(1e-9)
        )
        prev_closest = prev0 + prev_u.clamp(0, 1)[:, None] * prev_seg
        prev_dist = (robot_xy - prev_closest).norm(dim=-1)
        near_start = (robot_xy - p0).norm(dim=-1) <= self.cfg.path_switch_radius
        back = (i > 0) & (prev_u < 1.0) & (
          (u < 0.0) | (near_start & (prev_dist + 1e-4 < current_dist))
        ) & ~adv
      if not (adv | back).any():
        break
      self._path_seg_idx = self._path_seg_idx + adv.long() - back.long()

    # 计算沿路径的弧长位置 s
    i = torch.minimum(self._path_seg_idx, (count - 2).clamp(min=0))
    p0 = pts[ar, i]
    p1 = pts[ar, i + 1]
    seg = p1 - p0
    l2 = (seg * seg).sum(dim=-1).clamp_min(1e-9)
    projection = ((robot_xy - p0) * seg).sum(dim=-1) / l2
    u = projection.clamp(0.0, 1.0)
    progress_u = torch.where(
      (i == 0) & allow_backtracking, projection.clamp(max=1.0), u
    )
    s_new = cum[ar, i] + progress_u * seg.norm(dim=-1)
    closest = p0 + u.unsqueeze(-1) * seg
    self._path_lat_err = (robot_xy - closest).norm(dim=-1)
    self._path_s_delta = torch.where(new_path, 0.0, s_new - self._path_s)
    self._path_s = s_new

    # 动态延长路径以保证前视与墙体放置视距
    total = cum[ar, count - 1]
    horizon = self._path_s + self.cfg.path_lookahead + self.cfg.path_wall_ahead
    need = total < horizon
    if need.any():
      extend_walk_path(env, ar[need], horizon[need] + 20.0)

    # 在路径前方 path_lookahead 距离处提取前视引导目标点
    goal_xy, _ = walk_path_query(env, ar, self._path_s + self.cfg.path_lookahead)
    self.goal_pos_w[:] = goal_xy
    self.is_standing_env[:] = False
    self.is_inplace_env[:] = False

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """可视化折线路径与当前追踪点."""
    super()._debug_vis_impl(visualizer)
    if not self.cfg.path_follow or not hasattr(self._env, "_walk_path_pts"):
      return
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    pts = self._env._walk_path_pts.cpu().numpy()
    count = self._env._walk_path_n.cpu().numpy()
    seg_idx = self._path_seg_idx.cpu().numpy()
    base = self.robot.data.root_link_pos_w.cpu().numpy()
    for batch in env_indices:
      if np.linalg.norm(base[batch]) < 1e-6:
        continue
      i0 = max(0, int(seg_idx[batch]) - 2)
      i1 = min(int(count[batch]), i0 + self._PATH_VIZ_SEGMENTS + 3)
      poly = pts[batch, i0:i1]
      z = 0.03
      for k in range(len(poly) - 1):
        visualizer.add_cylinder(
          start=np.array([poly[k, 0], poly[k, 1], z]),
          end=np.array([poly[k + 1, 0], poly[k + 1, 1], z]),
          radius=0.02,
          color=(0.2, 0.8, 0.3, 0.9),
        )
      prog, _ = walk_path_query(
        self._env,
        torch.tensor([batch], device=self.device),
        self._path_s[batch : batch + 1],
      )
      prog = prog[0].cpu().numpy()
      visualizer.add_sphere(
        np.array([prog[0], prog[1], z]), 0.06, color=(0.2, 0.5, 0.9, 0.9)
      )

  def _update_command(self) -> None:
    """综合目标模式、执行 CBF 速度滤波与步态平滑门控."""
    # 1. 后退偏移模式 (Back-Offset Mode)
    if self.cfg.back_offset != 0.0:
      fwd_xy = quat_apply(
        yaw_quat(self.robot.data.root_link_quat_w), self._forward_b
      )[:, :2]
      self.goal_pos_w[:] = self.robot.data.root_link_pos_w[:, :2] - (
        self.cfg.back_offset * fwd_xy
      )
      self.is_standing_env[:] = False
      self.is_inplace_env[:] = False

    # 2. 路径跟踪模式 (Path-Follow Mode)
    path_active = False
    if self.cfg.path_follow and hasattr(self._env, "_walk_path_pts"):
      self._update_path_goal()
      path_active = True

    # 3. 前进偏移模式 (Forward-Offset Mode)
    if self.cfg.forward_offset != 0.0 and not path_active:
      fwd_xy = quat_apply(
        yaw_quat(self.robot.data.root_link_quat_w), self._forward_b
      )[:, :2]
      self.goal_pos_w[:] = self.robot.data.root_link_pos_w[:, :2] + (
        self.cfg.forward_offset * fwd_xy
      )
      self.is_standing_env[:] = False
      self.is_inplace_env[:] = False

    # 计算父类名义速度指令
    super()._update_command()
    self.vel_command_nominal_b = self.vel_command_b.clone()
    if not self.cfg.cbf_enabled:
      self._dodge_threat.zero_()
      self._apply_walk_heading_gate()
      return

    yq = yaw_quat(self.robot.data.root_link_quat_w)

    # 将机体系名义线速度转换至世界坐标系
    v_nom_b = self.vel_command_b[:, :2]
    v_nom_b3 = torch.cat([v_nom_b, torch.zeros_like(v_nom_b[:, :1])], dim=-1)
    u_nom_w = quat_apply(yq, v_nom_b3)[:, :2]

    # 根据当前球体的随机化真实半径，动态扩展安全防范距离
    self._dodge_ball_radius = self._env.sim.model.geom_size[:, self._ball_geom_id, 0]
    safe_radius = self.cfg.safe_radius + (self._dodge_ball_radius - DEFAULT_BALL_RADIUS)

    # 调用预测性控制屏障函数 (CBF) 2D 滤波核
    u_safe_w, threat, perp_signed, h, e, tti = predictive_dodge_filter(
      u_nom_w,
      self.robot.data.root_link_pos_w,
      self.ball.data.root_link_pos_w,
      self.ball.data.root_link_lin_vel_w,
      self._dodge_side,
      safe_radius=safe_radius,
      alpha=self.cfg.cbf_alpha,
      sense_radius=self.cfg.sense_radius,
      ball_radius=self._dodge_ball_radius,
      z_margin=0.05,
      momentum_time=self.cfg.momentum_time,
      comfort_buffer=self.cfg.comfort_buffer,
      min_ball_speed=self.cfg.min_ball_speed,
    )

    # 在威胁初次出现时锁定逃逸侧向符号 (+1 或 -1)，防止来回震荡
    new_threat = threat & ~self._dodge_threat
    default_side = torch.where(
      perp_signed >= 0.0, torch.ones_like(perp_signed), -torch.ones_like(perp_signed)
    )
    self._dodge_side = torch.where(new_threat, default_side, self._dodge_side)
    self._dodge_threat = threat

    # 将安全速度投影回机体坐标系，并截断在动力学包络内
    u_safe_w3 = torch.cat([u_safe_w, torch.zeros_like(u_safe_w[:, :1])], dim=-1)
    u_safe_b = quat_apply_inverse(yq, u_safe_w3)[:, :2]
    u_safe_b[:, 0] = u_safe_b[:, 0].clamp(-self.cfg.max_lin_vel_x, self.cfg.max_lin_vel_x)
    u_safe_b[:, 1] = u_safe_b[:, 1].clamp(-self.cfg.max_lin_vel_y, self.cfg.max_lin_vel_y)

    # 暂存 CBF-RL 奖励所需状态
    u_safe_clamped_b3 = torch.cat([u_safe_b, torch.zeros_like(u_safe_b[:, :1])], dim=-1)
    self._dodge_u_safe_w = quat_apply(yq, u_safe_clamped_b3)[:, :2]
    self._dodge_e_w = e
    self._dodge_h = h
    self._dodge_tti = tti

    # 决定是否改写策略网络实际观测到的速度指令
    if self.cfg.cbf_filter_command:
      self.vel_command_b[:, :2] = u_safe_b

    self._apply_walk_heading_gate()
    self.metrics["cbf_active_frac"] = threat.float()
    self.metrics["cbf_min_h"] = h

  def _apply_walk_heading_gate(self) -> None:
    """航向对齐平滑门控：在安全状态下先原地转弯，对准目标后再向前平稳迈步."""
    self._walk_forward_scale.fill_(1.0)
    if not self.cfg.turn_before_walk:
      return
    direction = self.goal_pos_w - self.robot.data.root_link_pos_w[:, :2]
    yaw_error = wrap_to_pi(
      torch.atan2(direction[:, 1], direction[:, 0]) - self.robot.data.heading_w
    ).abs()
    fraction = (
      (self.cfg.walk_stop_angle - yaw_error)
      / (self.cfg.walk_stop_angle - self.cfg.walk_full_speed_angle)
    ).clamp(0.0, 1.0)
    # 平滑三次 S 曲线缩放
    scale = fraction.square() * (3.0 - 2.0 * fraction)
    moving = (
      (self.distance_to_goal >= self.cfg.arrive_radius)
      & ~self.is_standing_env & ~self.is_inplace_env
    )
    forward = (self.cfg.kp * self.distance_to_goal).clamp(
      min=0.0, max=self.cfg.max_lin_vel_x
    ) * scale * moving
    safe = ~self._dodge_threat
    for command in (self.vel_command_b, self.vel_command_nominal_b):
      command[:, 0] = torch.where(safe, forward, command[:, 0])
      command[:, 1] = torch.where(safe, 0.0, command[:, 1])
    self._walk_forward_scale = torch.where(safe & moving, scale, 1.0)


@dataclass(kw_only=True)
class DodgeGoToGoalCommandCfg(GoToGoalCommandCfg):
  """DodgeGoToGoalCommand 参数配置项."""

  ball_name: str = "ball"
  home_goal: bool = False
  back_offset: float = 0.0
  forward_offset: float = 0.0
  turn_before_walk: bool = False
  walk_full_speed_angle: float = math.pi / 6
  walk_stop_angle: float = math.pi / 2
  path_follow: bool = False
  path_lookahead: float = 2.0
  path_switch_radius: float = 0.0
  path_allow_backtracking: bool = False
  path_wall_ahead: float = 11.0
  cbf_enabled: bool = True
  cbf_filter_command: bool = True
  cbf_alpha: float = 4.0
  safe_radius: float = 0.6
  sense_radius: float = 3.0
  z_active: float = 0.25
  momentum_time: float = 0.5
  comfort_buffer: float = 0.3
  min_ball_speed: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> DodgeGoToGoalCommand:
    return DodgeGoToGoalCommand(self, env)
