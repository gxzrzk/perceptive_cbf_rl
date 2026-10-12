"""强化学习奖励函数 (Reward Functions) 与惩罚项模块.

【理论架构与奖励设计机制】
在腿足机器人基于对抗运动先验 (Adversarial Motion Priors, AMP) 与控制屏障函数 (CBF-RL) 的控制架构中，
奖励系统由三大正交/协同的分支构成：

1. 运动先验与物理拟真步态奖励 (Locomotion & Gait Naturalness):
   - AMP 判别器风格奖励：由单独的网络评估机器人轨迹与动作捕捉真实数据集的分布相似度；
   - 速度与航向跟踪 (`track_anchor_linear_velocity`, `track_anchor_angular_velocity`): 指导质心跟踪指定速度；
   - 接触动力学柔顺性 (`feet_slip`, `soft_landing`): 惩罚着地期间的足底打滑与触地瞬间的剧烈冲击力，促使形成平稳迈步；
   - 腾空时间奖励 (`leap_flight_time`): 检测双足同时离地的连续腾空时长，提供粗粒度的跳跃诱导梯度。

2. 导航目标与走廊巡航奖励 (Goal Navigation & Corridor Adherence):
   - 势能塑形目标逼近奖励 (`goal_progress_reward`, `goal_distance_reward`): 沿目标位移方向给予连续势能递增；
   - 路径纯追踪与走廊贴合 (`walk_path_progress_reward`, `walk_path_adherence_reward`):
     奖励沿随机折线走廊的弧长推进，同时严格惩罚横向偏离走廊中心线的偏移距离；
   - 威胁门控步态与停止奖励 (`walk_path_progress_when_safe`, `walk_path_stillness_when_safe`):
     在安全状态下督促前进，在来球威胁逼近时瞬时解除前进要求，允许全力侧跃。

3. 控制屏障函数 (CBF) 避险与安全奖励 (Safety Barrier Rewards - CBF-RL):
   - 2D 质心速度级 CBF 安全奖励 (`dodge_cbf_reward`):
     基于相对阶 1 屏障函数 $h = s - D$，定义安全价值惩罚项与仿射动作模仿项：
         $r_{\\text{cbf}} = \\min(\\dot{h} + \\alpha h, 0) + (\\exp(-\\|v - v_{\\text{safe}}\\|^2 / \\sigma^2) - 1)$
     当策略产生不安全速度时产生严厉的一阶导数违规惩罚；
   - 3D 全身连杆级 Link-CBF 奖励 (`dodge_link_cbf_reward`, `wall_link_cbf_reward`):
     为每个单独连杆建立空间球体/胶囊体与来球或静态障碍墙表面的真实欧氏几何安全包络 $h_i = d_i - (r_{\\text{ball}} + r_{\\text{link}})$，
     并结合相对逼近速率 $\\dot{h}_i$，施加 $\\min(\\dot{h}_i + \\alpha h_i, 0)$ 的全身关节规避惩罚；
   - 关节级 Joint-CBF 修正与缓冲奖励 (`dodge_cbf_joint_correction`, `dodge_cbf_joint_buffer`):
     对动作管理器中二次规划 (QP) 投影产生的关节速度修正量均方 $-\\|v^* - v_{\\text{des}}\\|^2$ 施加惩罚，
     强迫策略自发输出符合安全约束的原生动作，进而实现实机无滤波器的裸网络部署。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse, 
  yaw_quat, 
  quat_apply,
  wrap_to_pi,
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def airborne(env, ball_z: torch.Tensor, ball_name: str = "ball", margin: float = 0.05) -> torch.Tensor:
  """半径自适应的球体滞空检测 (所有威胁门控共享).

  物理机制：
  静止在地面上的球体，其球心高度为 z = ball_radius。如果使用固定的绝对高度阈值，
  会错误排除在贴地高度飞行、直奔小腿和脚踝的低空危险来球。
  本函数通过检测 `z > ball_radius + margin`，既能准确捕捉低空来球，又能完全过滤在地面滚动的死球。
  """
  if not hasattr(env, "_airborne_ball_geom_id"):
    env._airborne_ball_geom_id = env.sim.mj_model.geom(f"{ball_name}/ball_collision").id
  # 提取每个环境中当前随机化后的球体真实半径
  r = env.sim.model.geom_size[:, env._airborne_ball_geom_id, 0]  # 形状: (N,)，各并行环境中飞球的当前几何半径
  return ball_z > r + margin


def _get_delay_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """获取延迟重置环境的布尔掩码（若启用了 DelayedTerminationManager）."""
  tm = env.termination_manager
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  delay_counters = getattr(tm, "_delay_counters", None)
  if isinstance(delay_env_mask, torch.Tensor) and isinstance(delay_counters, torch.Tensor):
    return delay_env_mask & (delay_counters > 0)
  return None


def _apply_delay_env_reward_scaling(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  """对处于延迟重置阶段的环境缩放奖励数值."""
  if not mask_delay:
    return reward

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return reward

  scaled_reward = reward * delay_env_rew_ratio
  return torch.where(delay_env_mask, scaled_reward, reward)


def _apply_delay_env_reward_mask_only(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  """仅在延迟重置阶段保留奖励，其余正常阶段掩膜置零."""
  if not mask_delay:
    return torch.zeros_like(reward)

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return torch.zeros_like(reward)

  scaled_reward = reward * delay_env_rew_ratio
  masked_reward = torch.where(delay_env_mask, scaled_reward, torch.zeros_like(reward))
  return masked_reward


def track_anchor_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
  track_dodge_safe: bool = False,
  mask_inplace: bool = False,
) -> torch.Tensor:
  """基准锚点刚体 (躯干/骨盆) 水平线速度跟踪奖励.

  奖励公式: r = exp(- ||v_xy - v_target_xy||^2 / std^2)

  参数说明:
    track_dodge_safe: 若为 True，则跟踪经过 CBF 安全滤波后的速度指令 u_safe_w（在来球威胁时包含侧移分量）。
      由于策略观测到的依然是名义速度，该机制使策略在不直接知道安全指令的情况下，通过奖励信号学会自发侧跃。
    mask_inplace: 对原地待命扔球的环境屏蔽此项奖励，防止静止跟踪目标阻碍策略做出躲避动作。
  """
  asset: Entity = env.scene[asset_cfg.name]

  if track_dodge_safe:
    # 提取由 DodgeGoToGoalCommand 暂存的世界系安全速度指令
    cmd_term = env.command_manager.get_term(command_name)
    u_safe_w = cmd_term._dodge_u_safe_w  # 形状: (N, 2)，世界坐标系下经 CBF 过滤后的二维安全期望速度 (vx, vy)
    command_xyz_w = torch.cat((u_safe_w, torch.zeros_like(u_safe_w[:, :1])), dim=-1)
  else:
    # 提取标准指令并旋转至世界坐标系
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"找不到指令项 '{command_name}'。"
    command_xyz_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    command_xyz_w = quat_apply(
      yaw_quat(asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]),
      command_xyz_b,
    )
  # 计算线速度跟踪误差平方
  lin_vel_error = torch.sum(torch.square(command_xyz_w[:,:3] - asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]), dim=1)
  reward = torch.exp(-lin_vel_error / std**2)
  if mask_inplace and not track_dodge_safe:
    cmd_term = env.command_manager.get_term(command_name)
    inplace = getattr(cmd_term, "is_inplace_env", None)
    if inplace is not None:
      reward = torch.where(inplace, torch.zeros_like(reward), reward)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """跟踪偏航角速度指令 wz，同时惩罚横滚与俯仰角速度 (wx, wy)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"找不到指令项 '{command_name}'。"

  anchor_ang_vel_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
  anchor_ang_z_vel_w = anchor_ang_vel_w[:, 2]
  command_ang_vel_w = command[:, 2]
  ang_vel_z_error = torch.square(command_ang_vel_w - anchor_ang_z_vel_w)

  anchor_ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]],
    anchor_ang_vel_w,
  )
  ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)

  total_error = ang_vel_z_error + ang_vel_xy_error

  reward = torch.exp(-total_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def body_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """惩罚刚体在机体系水平横滚与俯仰轴上的寄生角速度."""
  asset: Entity = env.scene[body_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids[0]]
  body_ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, body_cfg.body_ids[0]],
    body_ang_vel_w,
  )
  body_ang_vel_xy_b = body_ang_vel_b[:, :2]
  ang_vel_xy_error = torch.sum(torch.square(body_ang_vel_xy_b), dim=-1)

  reward = torch.exp(-ang_vel_xy_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_root_height(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """跟踪机器人根节点的默认站立高度."""
  asset: Entity = env.scene[asset_cfg.name]

  desired_height = asset.data.default_root_state[:, 2]
  cur_root_height = asset.data.body_link_pos_w[:, 0, 2]
  height_error = torch.square(desired_height - cur_root_height)
  reward = torch.exp(-height_error / std**2)
  return _apply_delay_env_reward_mask_only(env, reward, mask_delay, delay_env_rew_ratio)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """惩罚足底着地支撑期间的水平滑动线速度 (Penalize Foot Slipping)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  # 足底着地布尔掩码
  in_contact = (contact_sensor.data.found > 0).float()  # 形状: [B, N]，环境批次与足底接触点触地指示
  # 提取足底位姿点的世界水平线速度
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # 形状: [B, N, 2]，足底各接触点世界水平线速度
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # 形状: [B, N]，足底水平速度模长大小
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # 形状: [B, N]，足底水平速度模长平方
  # 仅在足底着地时计算滑动速度平方惩罚
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """惩罚足底触地瞬间的峰值冲击力，引导机器人形成轻柔吸能着陆的自然步态."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # 形状: [B, N, 3]，各接触点三维接触力矢量
  force_magnitude = torch.norm(forces, dim=-1)  # 形状: [B, N]，接触力大小模长
  # 计算第一拍触地检测事件
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # 形状: [B, N]，第一拍着地事件掩码
  landing_impact = force_magnitude * first_contact.float()  # 形状: [B, N]，着地瞬时冲击力大小
  cost = torch.sum(landing_impact, dim=1)  # 形状: [B]，每个环境汇总的着地冲击惩罚值
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def leap_flight_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  min_flight_s: float = 0.05,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """奖励双足同时腾空的滞空飞行时间 (跳跃动作特征).

  取双足当前滞空时间的最小值：min(air_time_left, air_time_right)。
  只有当两只脚均完全离地时才为正值，行走或拖步时至少一足触地从而严格得 0 分，杜绝作弊。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time  # 形状: [B, N_feet]，双足各自当前空中滞空时间 (秒)
  assert air_time is not None, (
    f"传感器 '{sensor_name}' 必须设置 track_air_time=True。"
  )
  # 双足必须同时处于空中
  flight_t = air_time.min(dim=1).values
  reward = (flight_t - min_flight_s).clamp(min=0.0)

  command = env.command_manager.get_command(command_name)
  active = (torch.norm(command[:, :2], dim=1) > command_threshold).float()

  env.extras["log"]["Metrics/flight_time_mean"] = flight_t.mean()
  return reward * active


def goal_distance_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.0,
) -> torch.Tensor:
  """基于高斯核的目标距离密集奖励: exp(-d^2 / std^2)."""
  command = env.command_manager.get_term(command_name)
  dist = command.distance_to_goal
  return torch.exp(-torch.square(dist) / std**2)


def goal_progress_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  clip: float = 2.0,
) -> torch.Tensor:
  """基于势能塑形的目标逼近速度奖励: (prev_dist - cur_dist) / dt.

  远离目标时产生对称惩罚，有效防止策略反复震荡。
  """
  command = env.command_manager.get_term(command_name)
  closing_speed = command.distance_delta / env.step_dt
  return closing_speed.clamp(-clip, clip)


def walk_path_progress_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  clip: float = 2.0,
) -> torch.Tensor:
  """沿随机巡航折线路径的沿线弧长推进速度奖励 (WallWalk 任务): delta_s / dt."""
  command = env.command_manager.get_term(command_name)
  rate = command._path_s_delta / env.step_dt
  return rate.clamp(-clip, clip)


def walk_path_progress_when_safe(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  v_max: float = 1.5,
  clip_lo: float = 2.0,
  stillness_penalty: float = 0.0,
  stillness_threshold: float = 0.02,
) -> torch.Tensor:
  """安全状态下的路径沿线推进奖励（在受到来球威胁时自动被门控清零，允许侧跃回避）."""
  command = env.command_manager.get_term(command_name)
  threat = command._dodge_threat.float()  # 来球滞空且逼近时为 1
  rate = command._path_s_delta / env.step_dt
  pay = rate.clamp(min=-clip_lo, max=v_max)
  if stillness_penalty > 0.0:
    forward_scale = getattr(command, "_walk_forward_scale", 1.0)
    pay = torch.where(
      rate.abs() <= stillness_threshold, -stillness_penalty * forward_scale, pay
    )
  return (1.0 - threat) * pay


def walk_path_stillness_when_safe(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  rate_scale: float = 0.4,
) -> torch.Tensor:
  """安全状态下的停滞惩罚代价: 惩罚在没有来球威胁时停在原地不往前走的偷懒行为."""
  command = env.command_manager.get_term(command_name)
  threat = command._dodge_threat.float()
  rate = command._path_s_delta / env.step_dt
  forward_scale = getattr(command, "_walk_forward_scale", 1.0)
  return (1.0 - threat) * forward_scale * torch.exp(-torch.square(rate) / rate_scale**2)


def walk_goal_heading_when_safe(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  robot_name: str = "robot",
  sharpness: float = 1.0,
  turn_angle_threshold: float = math.pi / 6,
  turn_velocity_std: float = 0.25,
) -> torch.Tensor:
  """在安全状态下奖励机器人面向目标方向或按期望角速度主动转弯对齐."""
  if not 0.0 <= turn_angle_threshold <= math.pi:
    raise ValueError("turn_angle_threshold 必须在 0 到 pi 之间")
  if turn_velocity_std <= 0.0:
    raise ValueError("turn_velocity_std 必须为正数")
  command = env.command_manager.get_term(command_name)
  robot = env.scene[robot_name]
  direction = command.goal_pos_w - robot.data.root_link_pos_w[:, :2]
  distance = torch.linalg.vector_norm(direction, dim=-1)
  heading = robot.data.heading_w
  forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
  alignment = (forward * direction).sum(-1) / distance.clamp_min(1e-6)
  alignment = alignment.clamp(-1.0, 1.0)
  signed = torch.sign(alignment) * alignment.abs().pow(sharpness)
  yaw_error = wrap_to_pi(torch.atan2(direction[:, 1], direction[:, 0]) - heading)
  target_yaw_rate = (command.cfg.kp_yaw * yaw_error).clamp(
    min=-command.cfg.max_ang_vel_z, max=command.cfg.max_ang_vel_z
  )
  yaw_rate = robot.data.root_link_ang_vel_w[:, 2]
  turn_score = 2.0 * torch.exp(
    -torch.square((yaw_rate - target_yaw_rate) / turn_velocity_std)
  ) - 1.0
  turn_score = torch.where(
    yaw_rate * yaw_error > 0.0, turn_score, -torch.ones_like(turn_score)
  )
  reward = torch.where(
    yaw_error.abs() > turn_angle_threshold + 1e-6, turn_score, signed
  )
  reward = torch.where(distance > 1e-6, reward, torch.zeros_like(reward))
  return (1.0 - command._dodge_threat.float()) * reward


def walk_path_adherence_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  std: float = 0.5,
) -> torch.Tensor:
  """走廊路径横向贴合度奖励: 惩罚垂直偏离折线中心线的横向漂移距离."""
  command = env.command_manager.get_term(command_name)
  d = command._path_lat_err
  return torch.exp(-torch.square(d) / std**2)


def goal_reached_bonus(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.25,
  speed_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """到达目标稀疏奖励: 要求同时进入目标圆域且线速度充分降低（平稳停驻）."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  dist = command.distance_to_goal
  speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1)
  return ((dist < threshold) & (speed < speed_threshold)).float()


def stop_at_goal_cost(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """进入目标区域后的残余速度惩罚代价."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  dist = command.distance_to_goal
  speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1)
  return speed * (dist < threshold).float()


def is_terminated_except(
  env: ManagerBasedRlEnv,
  exclude_terms: tuple[str, ...] = (),
) -> torch.Tensor:
  """常规非超时异常终止惩罚 (-200)，但排除指定的终止条件 (如 ball_contact).

  MimicKit 避球任务标准设定：被球击中仅提前结束回合以损失未来期望回报作为代价，
  而不施加巨大的即时负数尖峰，避免负梯度淹没动作模仿与跳跃探索。
  """
  tm = env.termination_manager
  terminated = tm.terminated.clone()
  for name in exclude_terms:
    if name in tm.active_terms:
      terminated = terminated & ~tm.get_term(name)
  return terminated.float()


def dodge_cbf_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.5,
  constraint_clip: float = 2.0,
  include_imitation: bool = True,
) -> torch.Tensor:
  """2D 质心控制屏障函数 (CBF-RL) 安全与模仿奖励项.

  理论公式 (Yang et al., CBF-RL Eqs. 22-23):
      r_cbf = min(dh/dt + alpha * h, 0) + (exp(-||v - v_safe||^2 / sigma^2) - 1)

  包含两个互补部分:
  1. 安全约束项 min(v . e + alpha * h, 0):
     若机器人自身当前移动速度未能满足 CBF 要求的最小逃逸速度，则产生负值惩罚；
  2. 安全动作模仿项 exp(-||v - v_safe||^2 / sigma^2) - 1:
     奖励真实线速度 v 贴合 CBF 滤波器推荐的安全速度 v_safe。
  """
  command = env.command_manager.get_term(command_name)
  threat = command._dodge_threat  # 形状: (N,)，威胁激活度标量 (0: 无威胁, 1: 威胁逼近)
  e = command._dodge_e_w  # 形状: (N, 2)，世界坐标系下垂直于小球速度的法向侧移单位矢量
  h = command._dodge_h  # 形状: (N,)，当前屏障函数代数安全裕度 (米)
  v_safe = command._dodge_u_safe_w  # 形状: (N, 2)，CBF 推荐执行的二维世界系避碰安全速度
  alpha = command.cfg.cbf_alpha

  v = command.robot.data.root_link_lin_vel_w[:, :2]

  # 安全屏障一阶导数: dh/dt = v . e (因垂直于航向，球速在逃逸轴投影为 0)
  h_dot = (v * e).sum(dim=-1)
  constraint = (h_dot + alpha * h).clamp(min=-constraint_clip, max=0.0)
  r = constraint
  if include_imitation:
    r = r + (torch.exp(-torch.square(v - v_safe).sum(dim=-1) / sigma**2) - 1.0)
  return torch.where(threat, r, torch.zeros_like(r))


def dodge_link_cbf_reward(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  ball_name: str = "ball",
  alpha: float = 2.0,
  margin: float = 0.05,
  constraint_clip: float = 2.0,
  z_active: float = 0.25,
  min_ball_speed: float = 0.5,
  reduce: str = "min",
  danger_band: float | None = None,
) -> torch.Tensor:
  """3D 全身各连杆刚体 (Link-CBF) 避球控制屏障函数安全惩罚 (<= 0).

  物理机制：
  针对机器人全身每一个独立连杆建立三维线性距离屏障：
      h_i = ||p_ball - p_link_i|| - (r_ball + r_link_i + margin)
  计算相对逼近速率：
      h_dot_i = (p_ball - p_link_i) . (v_ball - v_link_i) / ||p_ball - p_link_i||
  计算离散违约项：
      cbf_i = min(h_dot_i + alpha * h_i, 0)
  通过 reduce 参数将各肢体的违约惩罚聚合（min 惩罚最危险肢体，sum 督促四肢同时收缩躲避）。
  """
  robot: Entity = env.scene[robot_name]
  ball: Entity = env.scene[ball_name]

  # 首次调用时缓存各刚体的碰撞半径
  if not hasattr(env, "_dodge_link_radii"):
    mjm = env.sim.mj_model
    round_types = {
      int(mujoco.mjtGeom.mjGEOM_SPHERE),
      int(mujoco.mjtGeom.mjGEOM_CAPSULE),
      int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }
    body_r: dict[int, float] = {}
    for g in range(mjm.ngeom):
      if int(mjm.geom_type[g]) in round_types:
        b = int(mjm.geom_bodyid[g])
        body_r[b] = max(body_r.get(b, 0.0), float(mjm.geom_size[g, 0]))
    body_ids = robot.indexing.body_ids.tolist()
    radii = [body_r.get(int(b), 0.05) for b in body_ids]
    env._dodge_link_radii = torch.tensor(
      radii, device=env.device, dtype=torch.float32
    ) + margin  # 形状: (L,)，机器人 L 个被监视连杆各自的安全包络球半径
    env._dodge_ball_geom_id = int(
      mjm.geom(f"{ball_name}/ball_collision").id
    )

  r_link = env._dodge_link_radii  # 形状: (L,)，各连杆安全半径
  r_ball = env.sim.model.geom_size[:, env._dodge_ball_geom_id, 0]  # 形状: (N,)，各环境中飞球几何半径

  ball_p = ball.data.root_link_pos_w  # 形状: (N, 3)，飞球在世界系下的空间位置
  ball_v = ball.data.root_link_lin_vel_w  # 形状: (N, 3)，飞球在世界系下的线速度矢量
  link_p = robot.data.body_link_pos_w  # 形状: (N, L, 3)，机器人各连杆在世界系下的空间位置
  link_v = robot.data.body_link_lin_vel_w  # 形状: (N, L, 3)，机器人各连杆在世界系下的线速度矢量

  rel = ball_p.unsqueeze(1) - link_p  # 形状: (N, L, 3)，飞球相对各连杆中心的相对位移
  d = rel.norm(dim=-1).clamp_min(1e-6)  # 形状: (N, L)，飞球中心到各连杆中心的欧氏距离
  h = d - (r_ball.unsqueeze(1) + r_link.unsqueeze(0))  # 形状: (N, L)，连杆与球体的线性安全净距
  h_dot = (rel * (ball_v.unsqueeze(1) - link_v)).sum(dim=-1) / d  # 形状: (N, L)，安全净距的时间导数（接近率）
  cbf = (h_dot + alpha * h).clamp(min=-constraint_clip, max=0.0)

  if danger_band is not None:
    cbf = torch.where(h < danger_band, cbf, torch.zeros_like(cbf))
  if reduce == "sum":
    agg = cbf.sum(dim=1)
  elif reduce == "mean":
    agg = cbf.mean(dim=1)
  else:
    agg = cbf.min(dim=1).values

  threat = airborne(env, ball_p[:, 2]) & (ball_v[:, :2].norm(dim=-1) > min_ball_speed)
  return torch.where(threat, agg, torch.zeros_like(agg))


def wall_link_cbf_reward(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  wall_names: tuple[str, ...] = ("wall",),
  alpha: float = 1.0,
  margin: float = 0.05,
  constraint_clip: float = 2.0,
  reduce: str = "min",
  danger_band: float | None = None,
) -> torch.Tensor:
  """针对静态障碍墙的全身各刚体连杆控制屏障函数安全惩罚 (<= 0)."""
  robot: Entity = env.scene[robot_name]
  if not hasattr(env, "_wall_link_radii"):
    mjm = env.sim.mj_model
    round_types = {
      int(mujoco.mjtGeom.mjGEOM_SPHERE),
      int(mujoco.mjtGeom.mjGEOM_CAPSULE),
      int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }
    body_r: dict[int, float] = {}
    for g in range(mjm.ngeom):
      if int(mjm.geom_type[g]) in round_types:
        b = int(mjm.geom_bodyid[g])
        body_r[b] = max(body_r.get(b, 0.0), float(mjm.geom_size[g, 0]))
    body_ids = robot.indexing.body_ids.tolist()
    radii = [body_r.get(int(b), 0.05) for b in body_ids]
    env._wall_link_radii = torch.tensor(
      radii, device=env.device, dtype=torch.float32
    ) + margin
    env._wall_geom_ids = [
      int(mjm.geom(f"{w}/wall_collision").id) for w in wall_names
    ]

  r_link = env._wall_link_radii  # 形状: (L,)，各身体连杆安全包络球半径
  link_p = robot.data.body_link_pos_w  # 形状: (N, L, 3)，机器人各连杆在世界系下的空间位置
  link_v = robot.data.body_link_lin_vel_w  # 形状: (N, L, 3)，机器人各连杆在世界系下的线速度矢量

  cbf_per_wall = []
  wyaw = getattr(env, "_wall_yaw_w", None)

  for wi, wname in enumerate(wall_names):
    gid = env._wall_geom_ids[wi]
    wall: Entity = env.scene[wname]
    wp = wall.data.root_link_pos_w  # 形状: (N, 3)，静态障碍墙体在世界系下的中心位置
    hl = env.sim.model.geom_size[:, gid, 0]  # 形状: (N,)，障碍墙体在局部 X 方向的半长 (half-length)
    ht = env.sim.model.geom_size[:, gid, 1]  # 形状: (N,)，障碍墙体在局部 Y 方向的半厚度 (half-thickness)

    if wyaw is not None:
      w_yaw = wyaw[:, wi]
    else:
      w_yaw = torch.zeros(env.num_envs, device=env.device)
    c, s = torch.cos(-w_yaw), torch.sin(-w_yaw)

    rel_w = link_p - wp.unsqueeze(1)  # 形状: (N, L, 3)，连杆相对于障碍墙中心的空间位移矢量
    lx = c.unsqueeze(1) * rel_w[:, :, 0] - s.unsqueeze(1) * rel_w[:, :, 1]
    ly = s.unsqueeze(1) * rel_w[:, :, 0] + c.unsqueeze(1) * rel_w[:, :, 1]

    ex = (lx.abs() - hl.unsqueeze(1)).clamp(min=0.0)
    ey = (ly.abs() - ht.unsqueeze(1)).clamp(min=0.0)
    d = torch.sqrt(ex * ex + ey * ey + 1e-12)  # 形状: (N, L)，连杆中心到墙体平截面的欧氏距离
    h = d - r_link.unsqueeze(0)  # 形状: (N, L)，连杆外包络到墙体外表面的几何安全净距

    gx = torch.where(ex > 0, lx.sign() * (ex / d), torch.zeros_like(d))
    gy = torch.where(ey > 0, ly.sign() * (ey / d), torch.zeros_like(d))
    c_f, s_f = torch.cos(w_yaw), torch.sin(w_yaw)
    grad_wx = c_f.unsqueeze(1) * gx - s_f.unsqueeze(1) * gy
    grad_wy = s_f.unsqueeze(1) * gx + c_f.unsqueeze(1) * gy

    # 障碍墙静止，h_dot 完全取决于肢体自身向墙体移动的速度分量
    h_dot = -(grad_wx * link_v[:, :, 0] + grad_wy * link_v[:, :, 1])
    cbf_w = (h_dot + alpha * h).clamp(min=-constraint_clip, max=0.0)
    if danger_band is not None:
      cbf_w = torch.where(h < danger_band, cbf_w, torch.zeros_like(cbf_w))
    cbf_per_wall.append(cbf_w)

  stacked = torch.stack(cbf_per_wall, dim=-1)  # 形状: (N, L, W)，汇集所有 W 堵静态墙体对各连杆的 CBF 屏障违例值
  cbf_link = stacked.min(dim=-1).values  # 形状: (N, L)，各连杆面对所有墙体中最紧迫的屏障值

  if reduce == "sum":
    agg = cbf_link.sum(dim=1)
  elif reduce == "mean":
    agg = cbf_link.mean(dim=1)
  else:
    agg = cbf_link.min(dim=1).values
  return agg


def dodge_sidestep_reward(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  ball_name: str = "ball",
  alpha: float = 1.0,
  safe_radius: float = 0.4,
  constraint_clip: float = 2.0,
  z_active: float = 0.25,
  min_ball_speed: float = 0.5,
  strike_z: tuple[float, float] = (0.0, 0.7),
  gravity: float = 9.81,
) -> torch.Tensor:
  """针对贴地/低空来球的预测性全身侧向避让 CBF 奖励 (<= 0).

  物理机制：
  对于低于 0.7m 的低空球，机器人通过下蹲无法躲避（反而会把身体送入弹道）。
  此时唯一的逃生手段是向轨迹法向全力侧跃。本奖励在预测到来球弹道交点处于下半身区域时，
  强制要求底盘沿法向加速脱离直线轨迹。
  """
  robot: Entity = env.scene[robot_name]
  ball: Entity = env.scene[ball_name]
  rp = robot.data.root_link_pos_w[:, :2]
  vbase = robot.data.root_link_lin_vel_w[:, :2]
  bp = ball.data.root_link_pos_w[:, :2]
  bz = ball.data.root_link_pos_w[:, 2]
  vb = ball.data.root_link_lin_vel_w[:, :2]
  speed = vb.norm(dim=-1)
  vdir = vb / speed.clamp_min(1e-6).unsqueeze(-1)
  n = torch.stack([-vdir[:, 1], vdir[:, 0]], dim=-1)
  to_robot = rp - bp
  along = (to_robot * vdir).sum(dim=-1)
  perp_signed = (to_robot * n).sum(dim=-1)
  h = perp_signed.abs() - safe_radius
  h_dot = torch.sign(perp_signed) * (vbase * n).sum(dim=-1)
  cbf = (h_dot + alpha * h).clamp(min=-constraint_clip, max=0.0)

  # 弹道抛物线下坠预测交点高度 z_cross
  t_star = (along / speed.clamp_min(min_ball_speed)).clamp(min=0.0)
  vz = ball.data.root_link_lin_vel_w[:, 2]
  z_cross = bz + vz * t_star - 0.5 * gravity * t_star * t_star
  threat = (
    airborne(env, bz)
    & (speed > min_ball_speed)
    & (along > 0.0)
    & (z_cross > strike_z[0])
    & (z_cross < strike_z[1])
  )
  return torch.where(threat, cbf, torch.zeros_like(cbf))


def mimickit_dodge_reward(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  ball_name: str = "ball",
  pos_w: float = 0.9,
  vel_w: float = 0.1,
  pos_scale: float = 0.3,
  vel_scale: float = 1.0,
) -> torch.Tensor:
  """MimicKit / SMP 风格的躲避球经典综合奖励 (arXiv:2512.03028).

  公式: r = pos_w * (1 - exp(-pos_scale * dist_to_ball)) + vel_w * exp(-vel_scale * ||v_xy||^2)
  综合远离球体意图与非受威胁时的静止偏好。
  """
  robot: Entity = env.scene[robot_name]
  ball: Entity = env.scene[ball_name]
  pos_diff = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  pos_err = torch.linalg.norm(pos_diff, dim=-1)
  pos_reward = 1.0 - torch.exp(-pos_scale * pos_err)
  vel_err = torch.sum(torch.square(robot.data.root_link_lin_vel_w[:, :2]), dim=-1)
  vel_reward = torch.exp(-vel_scale * vel_err)
  return pos_w * pos_reward + vel_w * vel_reward


def dodge_stillness_when_safe(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  robot_name: str = "robot",
  vel_scale: float = 2.0,
) -> torch.Tensor:
  """仅在没有受到来球威胁时奖励保持完全静止 (消除由于视觉噪斑引发的频繁误躲抽搐)."""
  command = env.command_manager.get_term(command_name)
  threat = command._dodge_threat.float()
  robot: Entity = env.scene[robot_name]
  v_xy = robot.data.root_link_lin_vel_w[:, :2]
  still = torch.exp(-vel_scale * torch.sum(v_xy * v_xy, dim=-1))
  return (1.0 - threat) * still


def dodge_action_rate_when_safe(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
) -> torch.Tensor:
  """仅在安全状态下惩罚动作变化率 ||a_t - a_{t-1}||^2，压制无威胁时的关节抖动."""
  command = env.command_manager.get_term(command_name)
  threat = command._dodge_threat.float()
  rate = torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )
  return (1.0 - threat) * rate


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """机器人自身肢体相互干涉碰撞惩罚 (Self-Collision Cost)."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)


def dodge_cbf_joint_correction(env: ManagerBasedRlEnv) -> torch.Tensor:
  """关节级 CBF-RL 滤波器动作修正量惩罚: - mean_j (v* - v_des)_j^2.

  这是关节级 CBF-RL 训练的核心驱动杠杆：
  通过惩罚底层安全滤波器对策略期望速度所施加的修正量，
  迫使策略网络自主学会输出无需滤波介入就能完全满足安全边界的全身规避动作。
  """
  corr_sq = getattr(env, "_cbf_joint_corr_sq", None)
  if corr_sq is None:
    return torch.zeros(env.num_envs, device=env.device)
  return -corr_sq


def dodge_cbf_joint_buffer(
  env: ManagerBasedRlEnv,
  h_buf: float = 0.1,
) -> torch.Tensor:
  """关节级 CBF-RL 安全净距缓冲惩罚: - max(0, h_buf - h_min)^2.

  惩罚最危险肢体刚体侵入到距碰撞边界 h_buf 缓冲区以内的行为，
  确保策略在规避时保持从容的安全裕量，而非擦边越界。
  """
  hmin = getattr(env, "_cbf_joint_hmin", None)
  threat = getattr(env, "_cbf_joint_threat", None)
  if hmin is None or threat is None:
    return torch.zeros(env.num_envs, device=env.device)
  viol = torch.clamp(h_buf - hmin, min=0.0)
  return torch.where(threat, -(viol * viol), torch.zeros_like(viol))
