"""经过关节级控制屏障函数 (Joint-Level CBF) 滤波的位置动作管理器 (CBF-RL Phase 1).

【功能与架构定位】
该动作项继承并扩展了标准的关节位置动作项 `JointPositionAction`：
在标准流程中，网络策略输出的动作动作经过 `scale * a + default_pose` 得到期望的关节位置目标 $q_{\\text{des}}$。
但在安全强化学习 (CBF-RL) 框架下：
1. 隐式关节速度计算：将一步内的运动请求视为离散一阶微分得到的隐式期望关节速度：
       $v_{\\text{des}} = (q_{\\text{des}} - q_{\\text{cur}}) / \\Delta t$
2. 单活跃约束关节 CBF 投影：调用 `cbf_joint.py` 对最濒临碰撞威胁的刚体点进行单约束闭式投影，
   得到满足安全边界的关节速度 $v^*$。
3. 目标位置重积分：将安全速度重新积分得到安全目标关节位置：
       $q_{\\text{des}}^* = q_{\\text{cur}} + v^* \\cdot \\Delta t$
4. 状态暂存与奖励引导：
   计算单步修正量范数平方 $\\|v^* - v_{\\text{des}}\\|^2$ 和最小线性几何间隙 $h_{\\text{linear}} = \\text{dist} - D$，
   存入环境对象供后续奖励项（如 `dodge_cbf_joint_correction` 与 `dodge_cbf_joint_buffer`）读取。

【只用于训练与纯奖励塑形模式 (Train-Only & Pure Reward Shaping)】
- 运行期硬滤波 (`filter_action=True`): 修正量不仅指导奖励，还直接重写下发给底层 PD 控制器的实际目标位置。
- 纯奖励引导 (`filter_action=False`): 计算安全修正量并施加惩罚，但**不直接拦截执行器动作**（下发原始动作）。
  此时策略完全依靠奖励信号迫使其在没有保姆式硬拦截的情况下自发学会躲避，彻底杜绝 Sim2Real 训练与部署的不一致 (Train/Deploy Mismatch)。
- 导出部署时由于硬件无法提供特权球体状态与实时雅可比矩阵，该动作项仅在训练阶段启用，策略网络以纯自感知形式独立工作。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction

from src.tasks.amp_loco.mdp.cbf_joint import project_action, select_active_point

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# 默认受保护的候选刚体列表：涵盖骨盆和四肢全部主要连杆（与 rl_cfg.py 中的 amp_body_names 保持一致）
# 确保控制屏障函数能够对整个全身骨骼进行立体防御，而不仅局限于躯干底盘
_DEFAULT_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)


@dataclass(kw_only=True)
class CbfJointPositionActionCfg(JointPositionActionCfg):
  """附带关节级 CBF 避球安全投影的位置动作配置项."""

  cbf_enabled: bool = True
  """总开关。为 False 时退化为标准 JointPositionAction 透传模式。"""
  
  filter_action: bool = True
  """是否直接改写下发的动作。
  若为 True（默认），投影得到的 v* 将重写下发目标；
  若为 False，仍然计算 v* 和 h_min 并存入环境供奖励函数惩罚，但执行器依然执行策略生成的原始动作。
  这是纯 CBF-RL 奖励塑形，完全消除了训练与实机部署的差异。"""
  
  ball_name: str = "ball"
  """场景中球体实体对象的名称。"""
  
  body_names: tuple[str, ...] = ()
  """受 CBF 保护的候选刚体名称元组。
  若为空（默认），则自动匹配除头部云台相机之外的所有机器人连杆。
  每个连杆均从 MuJoCo 模型几何体自动提取其真实的物理碰撞半径。"""
  
  body_buffer: float = 0.05
  """添加到每个刚体安全边界上的几何离散化缓冲余量 (米)。
  最终每个刚体的防范包络半径为 D_i = r_ball + r_body_i + body_buffer。"""
  
  alpha: float = 1.5
  """Class-K 控制屏障函数衰减增益系数。"""
  
  delta_max: float = 2.0
  """最大 TTC 紧迫度惩罚裕量 (对应 h-dot 的量纲)，用于在快撞击时激发更激进的规避动作。"""
  
  t_alert: float = 0.4
  """触发最大紧迫度惩罚的预计碰撞时间阈值 (秒)。"""
  
  z_margin: float = 0.05
  """判定球体处于空中的相对高度偏置 (当球心高度大于 ball_radius + z_margin 时视为滞空)。"""
  
  sense_radius: float = 4.0
  """球体与刚体之间的最大预警距离阈值 (米)。"""
  
  min_ball_speed: float = 0.5
  """判定来球具有威胁的最小航速阈值 (米/秒)。"""

  def build(self, env: "ManagerBasedRlEnv") -> "CbfJointPositionAction":
    """构建对应的动作项实例."""
    return CbfJointPositionAction(self, env)


class CbfJointPositionAction(JointPositionAction):
  """集成关节级 CBF 安全投影的关节位置动作项实现."""

  cfg: CbfJointPositionActionCfg

  def __init__(self, cfg: CbfJointPositionActionCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self._enabled = bool(cfg.cbf_enabled)
    self._filter_action = bool(cfg.filter_action)
    # 初始化环境级暂存张量，防止第一步前未定义
    self._init_stash()
    if not self._enabled:
      return

    robot = self._entity
    idx = robot.indexing
    # 获取受控关节在全局广义速度 (qvel/nv) 中的列索引，用于雅可比切片
    self._act_vadr = idx.joint_v_adr[self._target_ids].long()  # (ndof,)
    
    # 确定受保护的候选刚体集合：默认匹配除相机俯仰连杆外的全部机器人刚体
    names = list(cfg.body_names) if cfg.body_names else ["^(?!camera_pitch_link$).*$"]
    body_local, _ = robot.find_bodies(names)
    self._body_local = torch.as_tensor(body_local, device=self.device, dtype=torch.long)
    self._body_global = idx.body_ids[self._body_local].to(torch.int32).contiguous()  # (K,)
    
    # 解析 MuJoCo 几何模型，提取每个候选刚体的真实最大碰撞横截面半径 (球体、胶囊体、圆柱体)
    mjm = env.sim.mj_model
    _round = {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
              int(mujoco.mjtGeom.mjGEOM_CYLINDER)}
    _br: dict[int, float] = {}
    for g in range(mjm.ngeom):
      if int(mjm.geom_type[g]) in _round:
        b = int(mjm.geom_bodyid[g])
        _br[b] = max(_br.get(b, 0.0), float(mjm.geom_size[g, 0]))
    self._body_radius = torch.tensor(
      [_br.get(int(gid), 0.05) for gid in self._body_global.tolist()], device=self.device
    )  # (K,) 每个候选刚体的真实碰撞几何半径
    self._dt = float(env.step_dt)
    self._ball = env.scene[cfg.ball_name]
    self._ball_geom_id = env.sim.mj_model.geom(f"{cfg.ball_name}/ball_collision").id
    self._nv = int(env.sim.wp_model.nv)

    import mujoco_warp as mjw
    import warp as wp

    self._wp = wp
    self._mjw = mjw

  def _init_stash(self) -> None:
    """在环境对象上分配用于奖励函数计算的状态张量 (供 step 1 之前安全访问)."""
    z = torch.zeros(self.num_envs, device=self.device)
    self._env._cbf_joint_corr_sq = z.clone()  # (N,) 关节速度修正量的均方值 mean_j (v* - v_des)_j^2
    self._env._cbf_joint_hmin = torch.full((self.num_envs,), 1e3, device=self.device)  # (N,) 最小距离裕度
    self._env._cbf_joint_threat = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _ball_estimate(self):
    """返回球体的真实世界位置与线速度 (特权信息)."""
    return self._ball.data.root_link_pos_w, self._ball.data.root_link_lin_vel_w

  def process_actions(self, actions: torch.Tensor) -> None:
    """处理策略输出的动作张量，执行隐式速度转换、CBF 投影滤波与位置更新."""
    # 父类方法负责将网络原始输出变换为期望目标位置 q_des 存入 self._processed_actions
    super().process_actions(actions)
    if not self._enabled:
      return

    q_des = self._processed_actions  # (N, ndof) 期望关节位置
    q_cur = self._entity.data.joint_pos[:, self._target_ids]  # (N, ndof) 当前实际关节位置
    v_des = (q_des - q_cur) / self._dt  # 离散一阶微分得到的期望关节速度 (隐式 q_dot)

    # 提取所有受保护候选刚体的世界坐标
    body_pos = self._entity.data.body_link_pos_w[:, self._body_local, :]  # (N, K, 3)
    ball_pos, ball_vel = self._ball_estimate()
    # 提取每个环境中当前球体的实际半径 (支持训练期球体大小随机化)
    ball_r = self._env.sim.model.geom_size[:, self._ball_geom_id, 0]  # (N,)
    # 计算每个刚体的专属安全防御距离矩阵 D_i = r_ball + r_body_i + buffer (N, K)
    clearance = ball_r.unsqueeze(1) + self._body_radius.unsqueeze(0) + self.cfg.body_buffer  # (N, K)

    # 1. 纯几何级选出每个环境中处于最危险状态的活跃刚体 (完全免雅可比计算)
    sel = select_active_point(
      body_pos,
      ball_pos,
      ball_vel,
      clearance,
      ball_radius=ball_r,
      z_margin=self.cfg.z_margin,
      sense_radius=self.cfg.sense_radius,
      min_ball_speed=self.cfg.min_ball_speed,
    )
    # 2. 仅对每个环境中最危险的一个刚体点，调用 GPU Warp 核函数计算其平移雅可比
    jac_act = self._active_jacobian(sel["idx"], body_pos)  # (N, 3, ndof)

    # 3. 闭式二次规划求解安全关节速度指令 v* 及修正差量 corr
    v_star, corr = project_action(
      v_des,
      jac_act,
      sel["r_act"],
      ball_vel,
      sel["h_act"],
      sel["dist"],
      sel["nu"],
      sel["D_act"],
      sel["threat"],
      alpha=self.cfg.alpha,
      delta_max=self.cfg.delta_max,
      t_alert=self.cfg.t_alert,
    )

    # 4. 根据配置决定是否直接重写下发执行的动作
    if self._filter_action:
      q_des_star = q_cur + v_star * self._dt
      # 仅在受到威胁的环境采用投影后的安全位置，其余环境保持原始 q_des，避免浮点数漂移
      self._processed_actions = torch.where(sel["threat"].unsqueeze(-1), q_des_star, q_des)
    # 若 filter_action 为 False，保持 _processed_actions 不变，以纯奖励惩罚驱动策略自主学习

    # 5. 将 CBF 指标暂存至环境属性，供 CBF-RL 安全惩罚与缓冲奖励项使用
    self._env._cbf_joint_corr_sq = (corr * corr).mean(dim=-1)  # (N,) 修正量均方
    # 暂存线性空间净裕度 (dist - D，单位米)，相比平方后的 h 具有物理可解释性
    self._env._cbf_joint_hmin = sel["dist"] - sel["D_act"]  # (N,)
    self._env._cbf_joint_threat = sel["threat"]  # (N,)

  def _active_jacobian(
    self, idx: torch.Tensor, body_pos: torch.Tensor
  ) -> torch.Tensor:
    """使用 Warp GPU 核函数批量计算每个环境中活跃刚体点的平移雅可比矩阵 (已切片受控关节列)."""
    wp = self._wp
    ar = torch.arange(self.num_envs, device=self.device)
    # 提取活跃刚体在世界系下的空间点坐标 (N, 3)
    point_t = body_pos[ar, idx].contiguous().to(torch.float32)
    # 提取对应的全局刚体 ID (N,)
    body_t = self._body_global[idx].contiguous()
    # 分配雅可比输出缓冲区 (N, 3, nv)
    jacp_t = torch.zeros(
      (self.num_envs, 3, self._nv), device=self.device, dtype=torch.float32
    )
    # 调用 MuJoCo Warp 的并行几何雅可比算子
    self._mjw.jac(
      self._env.sim.wp_model,
      self._env.sim.wp_data,
      wp.from_torch(jacp_t, dtype=wp.float32),
      None,
      wp.from_torch(point_t, dtype=wp.vec3),
      wp.from_torch(body_t, dtype=wp.int32),
    )
    # 切片出受控自由度列返回，形状为 (N, 3, ndof)
    return jacp_t[:, :, self._act_vadr]
