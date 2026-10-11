"""关节级控制屏障函数 (Joint-Level CBF) 避球安全投影模块.

【理论背景与参考文献】
本模块基于 CBF-RL (arXiv:2412.04658 / docs/ball_cbf_rl_summary.pdf) 理论，
区别于 `cbf.py` 中针对质心/底盘 2D 平面线速度滤波，本模块直接作用于机器人的**全身各关节速度** $\\dot{q}$。
它允许人形机器人不仅通过整体平移跳跃躲避，还能通过全身协同（如收腿、缩颈、下蹲、侧身扭转等精细动作）规避来球。

【数学模型与单约束 GPU 闭式投影推导】
1. 几何定义与离散候选刚体点：
   设机器人由 $K$ 个关键骨骼刚体组成，其在关节广义坐标 $q$ 下的世界坐标为 $p_i(q), i=1,\\dots,K$。
   针对每个刚体点，定义其与球心 $p_{\\text{ball}}$ 的安全屏障函数：
       $h_i(q) = \\|p_{\\text{ball}} - p_i(q)\\|^2 - D_i^2$
   其中 $D_i = r_{\\text{ball}} + r_i + \\text{buffer}$ 为球半径、刚体自身包围球半径及安全缓冲距离之和。
   当 $h_i(q) \\ge 0$ 时，该刚体处于球体安全边界之外。

2. 关节速度级仿射控制屏障条件 (Affine CBF on Joint Velocity)：
   对时间求全导数：
       $\\dot{h}_i = 2 (p_{\\text{ball}} - p_i)^T (\\dot{p}_{\\text{ball}} - \\dot{p}_i)$
   根据运动学雅可比关系 $\\dot{p}_i = J_i(q) v$，其中 $v = \\dot{q}$ 为关节广义速度，代入得：
       $\\dot{h}_i = 2 r_i^T \\dot{p}_{\\text{ball}} - 2 r_i^T J_i v$   (记 $r_i = p_{\\text{ball}} - p_i$)
   标准 Class-K CBF 条件要求 $\\dot{h}_i \\ge -\\alpha h_i + \\delta$（其中 $\\delta$ 为时间紧迫度安全松弛量）：
       $2 r_i^T \\dot{p}_{\\text{ball}} - 2 r_i^T J_i v \\ge -\\alpha h_i + \\delta$
       $\\iff 2 r_i^T J_i v \\le \\alpha h_i + 2 r_i^T \\dot{p}_{\\text{ball}} - \\delta$
   整理为对关节速度 $v$ 的标准线性不等式约束：
       $a_i v \\le b_i$
   其中：
       $a_i = 2 r_i^T J_i \\in \\mathbb{R}^{1 \\times n_{\\text{dof}}}$
       $c_i = 2 r_i^T \\dot{p}_{\\text{ball}} \\in \\mathbb{R}$
       $b_i = \\alpha h_i + c_i - \\delta \\in \\mathbb{R}$

3. 活跃碰撞点选择 (Active Point Selection)：
   为适应 GPU 并行高吞吐采样，采用最危险单约束近似 (Single Active Constraint Approximation)：
       $i^* = \\arg\\min_{i \\in \\{1,\\dots,K\\}} h_i(q)$
   该选择过程**完全无需计算雅可比矩阵**，仅依赖正向运动学位置。因此，在数千个并行环境中，
   每个控制步仅需对选出的最危险刚体 $i^*$ 调用一次 MuJoCo/Warp GPU 雅可比核函数，极大地节省了计算开销。

4. 碰撞时间紧迫度裕度 (TTC Urgency Margin $\\delta$)：
   定义相对法向接近速率：
       $\\nu = \\max\\left( 0, -\\frac{r_{i^*}^T \\dot{p}_{\\text{ball}}}{\\|r_{i^*}\\|} \\right)$
   计算预计碰撞时间 TTC (Time-to-Contact)：
       $TTC = \\frac{\\max(0, \\text{dist} - D_{i^*})}{\\nu + \\epsilon}$
   紧迫度裕量 $\\delta$ 随 TTC 缩短而线性增强：
       $\\delta = \\delta_{\\max} \\max\\left(0, 1 - \\frac{TTC}{T_{\\text{alert}}}\\right)$
   当球体极度接近时，$\\delta$ 会收紧约束上限 $b$，迫使各关节以更大加速度紧急回避。

5. 闭式一秩二次规划 (QP) 投影求解：
   求解最小扰动优化问题 $\\min_v \\frac{1}{2}\\|v - v_{\\text{des}}\\|^2$ s.t. $a_{i^*} v \\le b_{i^*}$。
   约束违反量标量为 $g = a_{i^*} v_{\\text{des}} - b_{i^*}$。若 $g > 0$，则沿约束梯度方向进行正交修正：
       $v^* = v_{\\text{des}} - \\frac{\\max(0, g)}{\\|a_{i^*}\\|^2 + \\epsilon} a_{i^*}^T$
   输出安全关节速度 $v^*$ 及其修正向量 $\\Delta v = v^* - v_{\\text{des}}$。
"""

from __future__ import annotations

import torch


def select_active_point(
  body_pos_w: torch.Tensor,  # (N, K, 3) 候选刚体中心的世界坐标 (N 个环境, K 个候选刚体)
  ball_pos_w: torch.Tensor,  # (N, 3) 球体世界坐标
  ball_vel_w: torch.Tensor,  # (N, 3) 球体世界线速度
  clearance: torch.Tensor,  # (N, K) 每个刚体的安全防范距离 D_i = r_ball + r_body_i + buffer
  *,
  ball_radius: torch.Tensor,  # (N,) 逐环境球体半径
  z_margin: float,  # 判定球体在空中的高度偏置 (ball_z > ball_radius + z_margin)
  sense_radius: float,  # 感知报警距离
  min_ball_speed: float,  # 判定来球有效的最小航速
  eps: float = 1e-6,  # 防除零极小常数
) -> dict[str, torch.Tensor]:
  """无需计算雅可比，纯几何级选出每个环境中安全屏障值最小（最危险）的活跃刚体点.

  返回字典包含:
    - ``idx``: (N,) 选出的活跃刚体在候选集合 K 中的索引
    - ``r_act``: (N, 3) 相对位移矢量 ball - p_active
    - ``h_act``: (N,) 活跃刚体当前的屏障值 h_min
    - ``dist``: (N,) 活跃刚体与球心的空间几何欧氏距离 ||r_act||
    - ``nu``: (N,) 沿相对连线方向的径向逼近速率
    - ``D_act``: (N,) 活跃刚体对应的安全距离阈值
    - ``threat``: (N,) 布尔掩码，指示球体是否滞空、正向逼近且在警报距离内
  """
  n, k = body_pos_w.shape[0], body_pos_w.shape[1]
  # 计算球心到所有刚体的位移矢量 r_all = p_ball - p_i (N, K, 3)
  r_all = ball_pos_w.unsqueeze(1) - body_pos_w
  # 计算各刚体与球心距离的平方 ||r_all||^2 (N, K)
  d2 = (r_all * r_all).sum(-1)
  # 扩展或广播逐刚体安全隔离距离矩阵 D
  D = clearance if clearance.dim() == 2 else clearance.unsqueeze(1).expand(n, k)
  # 计算各刚体的连续屏障函数取值 h_i = ||r_i||^2 - D_i^2 (N, K)
  h_all = d2 - D * D

  # 寻找最容易破界违规的最小屏障值刚体索引 idx (N,)
  idx = torch.argmin(h_all, dim=1)
  ar = torch.arange(n, device=body_pos_w.device)
  # 提取该最危险活跃刚体的各项状态
  r_act = r_all[ar, idx]  # 活跃刚体相对位移 (N, 3)
  h_act = h_all[ar, idx]  # 最小屏障值 (N,)
  dist = torch.sqrt(d2[ar, idx].clamp(min=0.0) + eps)  # 实际距离 (N,)
  D_act = D[ar, idx]  # 活跃刚体安全防线半径 (N,)

  # 计算球速
  speed = ball_vel_w.norm(dim=-1)
  # 径向接近速度 nu: 当球向刚体靠近时，r_act . ball_vel < 0，取反后得到正的向心接近率
  nu = torch.clamp(-(r_act * ball_vel_w).sum(-1) / dist, min=0.0)
  # 威胁判定掩码：滞空且速度有效、正向接近、距离小于探测阈值
  threat = (
    (ball_pos_w[:, 2] > ball_radius + z_margin)
    & (speed > min_ball_speed)
    & (nu > 0.0)
    & (dist < sense_radius)
  )
  return {
    "idx": idx,
    "r_act": r_act,
    "h_act": h_act,
    "dist": dist,
    "nu": nu,
    "D_act": D_act,
    "threat": threat,
  }


def project_action(
  v_des: torch.Tensor,  # (N, ndof) 策略期望输出的未滤波关节速度指令
  jac_act: torch.Tensor,  # (N, 3, ndof) 活跃刚体对应的平移雅可比矩阵 (已切片为受控关节列)
  r_act: torch.Tensor,  # (N, 3) 相对位移向量 p_ball - p_active
  ball_vel_w: torch.Tensor,  # (N, 3) 球体世界线速度
  h_act: torch.Tensor,  # (N,) 活跃刚体当前屏障函数值
  dist: torch.Tensor,  # (N,) 活跃刚体距球心几何距离
  nu: torch.Tensor,  # (N,) 径向逼近速度
  D_act: torch.Tensor,  # (N,) 活跃刚体安全包络半径
  threat: torch.Tensor,  # (N,) 威胁激活布尔掩码
  *,
  alpha: float,  # Class-K CBF 衰减系数
  delta_max: float,  # 最大时间紧迫度惩罚裕量
  t_alert: float,  # 紧迫度达到饱和预警时间阈值 (秒)
  eps: float = 1e-6,  # 极小数值稳定常量
) -> tuple[torch.Tensor, torch.Tensor]:
  """针对期望关节速度 v_des 进行单约束闭式二次规划 (QP) 安全投影.

  优化目标: min_{v} ||v - v_des||^2  s.t.  a * v <= b
  求解公式: v* = v_des - a^T * [a * v_des - b]_+ / (||a||^2 + eps)

  参数:
    v_des: (N, ndof) 策略生成的期望关节速度
    jac_act: (N, 3, ndof) 活跃刚体平移雅可比
    r_act: (N, 3) 相对位置向量
    ball_vel_w: (N, 3) 球速
    h_act: (N,) 屏障函数值
    dist: (N,) 空间间距
    nu: (N,) 相对逼近速率
    D_act: (N,) 刚体防范距离
    threat: (N,) 是否存在碰撞威胁掩码
    alpha: 屏障衰减系数
    delta_max: 最大松弛增量
    t_alert: 紧急警报时间窗
    eps: 极小稳定因子

  返回:
    (v_star, correction):
      - v_star: (N, ndof) 经过 CBF 安全投影后的关节速度指令
      - correction: (N, ndof) 修正补偿量 v* - v_des (无威胁时严格全零)
  """
  # 计算约束梯度向量 a = 2 * r^T * J (形状: (N, ndof))
  a = 2.0 * torch.einsum("nj,njd->nd", r_act, jac_act)
  # 计算球体自速度产生的偏置项 c = 2 * r^T * ball_vel (形状: (N,))
  c = 2.0 * (r_act * ball_vel_w).sum(-1)

  # 计算碰撞剩余时间 TTC (Time-to-Contact)
  ttc = torch.clamp(dist - D_act, min=0.0) / (nu + eps)
  # 紧迫度惩罚变量 delta：当 TTC 趋于 0 时向 delta_max 线性饱和提升
  delta = delta_max * torch.clamp(1.0 - ttc / t_alert, min=0.0)

  # 仿射不等式约束上界 b = alpha * h + c - delta (形状: (N,))
  b = alpha * h_act + c - delta
  # 计算当前期望速度对安全边界的违背量 g = a * v_des - b (形状: (N,))
  g = (a * v_des).sum(-1) - b
  # 计算投影步长比例 scale = max(0, g) / (||a||^2 + eps)
  scale = torch.clamp(g, min=0.0) / ((a * a).sum(-1) + eps)
  # 一阶闭式投影修正：若发生违背，沿梯度 a 的反方向投影推回可行解域内
  v_star = v_des - a * scale.unsqueeze(-1)

  # 仅在确认受到威胁的环境中采用安全速度，其余环境严格保留原始期望速度
  v_star = torch.where(threat.unsqueeze(-1), v_star, v_des)
  return v_star, v_star - v_des
