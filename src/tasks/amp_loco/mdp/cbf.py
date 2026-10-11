"""2D 质心速度级预测性避球控制屏障函数 (Predictive Ball-Avoidance CBF) 滤波模块.

【模块定位与背景】
在敏捷躲避高速来球 (Dodgeball) 任务中，导航层 (`GoToGoalCommand`) 生成机体系名义速度指令 $u_{\\text{nom}}$。
若直接将球体当前瞬时位置作为避障点，会导致两个严重问题：
1. 球体是以极高速度抛物线砸向躯干中心，在飞行阶段其在机器人位置的法向梯度接近奇异（正对来球时无法确定往哪侧躲）；
2. 球体落地后在地面滚动，如果基于当前位置滤波，机器人会像追逐足球一样跟随滚动的球，破坏躲避行为。

【核心物理机制：沿预测直线轨迹的法向侧移 (Perpendicular Sidestep)】
由于重力仅在竖直 z 轴方向产生加速度，球体在水平 xy 平面内的运动为匀速直线运动。
因此，真实威胁是机器人当前位置到球体**预测未来直线轨迹**的垂直距离。
最有效的避险动作是**正交于来球航向的瞬时侧跃 (Sidestep)**。

【CBF 数学推导与反应时间补偿 (Reaction-Time Aware CBF)】
1. 几何定义：
   设球体水平速度方向为 $v_{\\text{dir}} = v_{\\text{ball}, xy} / \\|v_{\\text{ball}, xy}\\|$；
   垂直于来向的法向量 $n = [-v_{y}, v_{x}]^T$；
   选定逃逸方向 $e = \\text{side} \\cdot n$（其中 $\\text{side} \\in \\{+1, -1\\}$ 在威胁初次出现时锁定，避免震荡）；
   机器人相对于球体在逃逸方向上的投影坐标为 $s = (p_{\\text{robot}} - p_{\\text{ball}}) \\cdot e$。

2. 相对阶 1 屏障函数 (Relative-degree 1 Barrier Function)：
   定义安全裕度函数：
       $h(x) = s - D$
   其中 $D$ 为安全距离阈值 (`safe_radius`，综合球半径与机器人躯干半宽)。当 $h \\ge 0$ 时处于安全集内部。
   由于球体速度与逃逸方向 $e$ 正交，因此：
       $\\dot{h} = u \\cdot e$
   标准扩展类-K (Class-K) CBF 约束条件为：
       $\\dot{h} \\ge -\\alpha h \\iff u \\cdot e \\ge \\alpha (D - s)$

3. 动力学时延与反应时间补偿 (Time-to-Impact Awareness)：
   真实人形机器人并非理想单积分器，实测从下发指令到建立横向速度有约 0.5s 的动量加速延迟 (`momentum_time`)，
   且速度跟踪率约为 54%。若仅依赖瞬态几何度量，在球速极快时往往因来不及加速而被击中。
   计算来球到达正对位置的剩余时间 (Time-to-Impact, TTI)：
       $TTI = \\text{along} / v_{\\text{speed}}$
   扣除动量延迟与舒适安全裕度后，实际可用避让时间为：
       $t_{\\text{usable}} = \\max(TTI - t_{\\text{momentum}} - t_{\\text{comfort}}, t_{\\text{floor}})$
   在有限可用时间内逃出危险区所需的保底侧移速度为：
       $v_{\\text{urgent}} = (D - s) / t_{\\text{usable}}$

4. 综合约束与闭式二次规划 (QP) 投影：
   速度指令沿逃逸方向的下限取二者最大值：
       $u \\cdot e \\ge \\text{rhs} = \\max\\left( \\alpha (D - s), \\frac{D - s}{t_{\\text{usable}}} \\right)$
   通过最小范数投影 (Minimal-norm Projection)，闭式求解得到滤波后的安全速度：
       $u^* = u_{\\text{nom}} + \\max(0, \\text{rhs} - u_{\\text{nom}} \\cdot e) \\cdot e$
   当机器人正处于来球轨迹上 ($s \\approx 0$) 时，全速向侧面弹射；一旦逃至安全距离 ($s \\ge D$)，附加推力平滑归零。
"""

from __future__ import annotations

import torch


def predictive_dodge_filter(
  u_nom_w: torch.Tensor,  # (N, 2) 世界坐标系下的名义水平速度指令 (vx, vy)
  robot_pos_w: torch.Tensor,  # (N, 3) 机器人世界坐标系位置
  ball_pos_w: torch.Tensor,  # (N, 3) 球体世界坐标系位置
  ball_vel_w: torch.Tensor,  # (N, 3) 球体世界坐标系线速度
  latched_side: torch.Tensor,  # (N,) 逃逸侧向锁定标志 (+1 或 -1)
  *,
  safe_radius: float | torch.Tensor,  # 安全距离阈值 D (标量或逐环境张量，随球尺寸自适应)
  alpha: float,  # Class-K CBF 衰减增益系数 (控制靠近边界时的排斥刚度)
  sense_radius: float,  # CBF 感知激活范围 (沿来球方向的最大预警距离)
  ball_radius: float | torch.Tensor,  # (N,) 逐环境球体碰撞半径
  z_margin: float = 0.05,  # 判定球体在空中的高度裕度 (球心 z > ball_radius + z_margin 时视为滞空)
  momentum_time: float = 0.5,  # 机器人速度建立动量时间常数 (秒)
  comfort_buffer: float = 0.3,  # 预留的安全缓冲时间裕度 (秒)
  min_ball_speed: float = 0.5,  # 判定球体有效飞行的最小水平航速阈值 (m/s)
  t_floor: float = 0.1,  # 最短可用时间下限 (秒)，防止除零与速度指令无限爆炸
  eps: float = 1e-6,  # 极小值数值稳定常数
) -> tuple[
  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
  """基于预测轨迹的 2D 质心速度 CBF 安全投影滤波函数.

  参数:
    u_nom_w: 世界系下的名义速度指令 (N, 2)
    robot_pos_w: 机器人根节点世界坐标 (N, 3)
    ball_pos_w: 球体根节点世界坐标 (N, 3)
    ball_vel_w: 球体世界系线速度 (N, 3)
    latched_side: 锁定的逃避方向符号 (+1/-1, 形状 N)
    safe_radius: 安全隔离半径 D
    alpha: 屏障函数松弛系数
    sense_radius: 纵向感知警戒距离
    ball_radius: 球半径
    z_margin: 滞空高度判据裕量
    momentum_time: 加速起步迟滞时间
    comfort_buffer: 提前规避时间余量
    min_ball_speed: 最小球速
    t_floor: 避碰可用时间下限
    eps: 防除零微小偏置

  返回:
    u_out: (N, 2) 滤波后的世界系安全速度指令；无威胁环境下等于 u_nom_w
    threat: (N,) 布尔掩码，指示当前环境是否触发来球威胁 (在空中 + 正向飞来 + 射程内)
    perp_signed: (N,) 机器人相对于来球轨迹线的带符号横向偏差 (供调用者在初次受到威胁时锁定逃逸方向)
    h: (N,) 屏障函数当前取值 s - D (>=0 表示位于安全集)
    e: (N, 2) 归一化逃逸方向向量 (同时也是屏障函数关于机器人速度的雅可比梯度 nabla_v h)
    tti: (N,) 预计到达机器人横向平齐点的时间 (Time-to-Impact, 秒)，供特权 Critic 网络观测
  """
  # 提取机器人与球体在水平 xy 平面的位置与速度分量
  rp = robot_pos_w[:, :2]
  bp = ball_pos_w[:, :2]
  vb = ball_vel_w[:, :2]

  # 计算球体在水平面上的航速与单位航向向量
  speed = vb.norm(dim=-1)
  vdir = vb / (speed.unsqueeze(-1) + eps)  # 球体水平航向单位向量
  # 构造与航向逆时针正交 90 度的法向量 n = [-vdir_y, vdir_x]
  n = torch.stack([-vdir[:, 1], vdir[:, 0]], dim=-1)

  # 计算球心指向机器人重心的相对位移向量
  to_robot = rp - bp
  # along > 0: 机器人依然处于球体飞行轨迹的前方（即球正在逼近）
  along = (to_robot * vdir).sum(dim=-1)
  # 计算机器人相对于球体飞行轨迹线在法向 n 上的带符号垂直偏移
  perp_signed = (to_robot * n).sum(dim=-1)

  # 威胁判定条件 (Threat Gating):
  # 1. 球体处于空中 (z 高于地面一定安全高度，排除在地面静止或滚动的球)
  # 2. 球速大于最小运动阈值 (处于受控飞行抛射状态)
  # 3. 球体仍在接近机器人 (along > 0，球已飞过时不触发)
  # 4. 处于感知距离以内 (along < sense_radius)
  threat = (
    (ball_pos_w[:, 2] > ball_radius + z_margin)
    & (speed > min_ball_speed)
    & (along > 0.0)
    & (along < sense_radius)
  )

  # 确定逃逸方向向量 e (通过 latched_side 保持单次躲避中方向不翻转)
  e = latched_side.unsqueeze(-1) * n  # 选定的侧移单位方向 (N, 2)
  # 机器人在逃逸方向上的投影距离 s
  s = (to_robot * e).sum(dim=-1)
  # 屏障函数 h = s - D (h >= 0 表示已走出危险包络)
  h = s - safe_radius
  # 距安全边界还差的侧移距离 clearance = D - s = -h
  clearance = safe_radius - s

  # 计算到达平齐线的碰撞剩余时间 (TTI = 纵向距离 / 球航速)
  tti = along / (speed + eps)
  # 计算扣除加速度建立时间与安全舒适裕度后的可用有效反应时间
  t_usable = torch.clamp(tti - momentum_time - comfort_buffer, min=t_floor)

  # 所需的最小侧向速度边界 rhs = max(Class-K 项, 时间紧迫性倒数项)
  rhs = torch.maximum(alpha * clearance, clearance / t_usable)
  # 计算名义指令在逃逸方向上的当前投影速度
  u_dot_e = (u_nom_w * e).sum(dim=-1)
  # 计算不足以满足 CBF 约束的安全修正量
  add = torch.clamp(rhs - u_dot_e, min=0.0)
  # 进行半空间正交投影补偿
  u_safe = u_nom_w + add.unsqueeze(-1) * e

  # 仅在存在威胁的环境应用 CBF 滤波，其余环境维持名义速度
  u_out = torch.where(threat.unsqueeze(-1), u_safe, u_nom_w)
  return u_out, threat, perp_signed, h, e, tti
