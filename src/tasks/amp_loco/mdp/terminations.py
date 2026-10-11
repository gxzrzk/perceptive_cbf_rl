"""回合终止条件 (Terminations) 与延迟重置管理器模块.

【物理机制与任务终止判定】
在强化学习训练中，合理的终止条件设计直接决定策略能否收敛到符合人类物理直觉的自然姿态，
并提供及时、精准的失败惩罚信号。本模块涵盖三项核心终止逻辑与一项延迟重置调度器：

1. 持续过低高度倒地检测 (`root_height_below_minimum_sustained`)：
   不同于传统瞬间高度判定，本判定通过累积计数器检测躯干是否**持续**低于阈值。
   这避免了跳跃着陆瞬间正常的蹲伏缓冲被误判为摔倒，同时又能精准识别策略为了偷懒而退化成的持续低姿态跪爬。

2. 高速飞球击中检测 (`ball_contact`)：
   包含两套互补的击中判定机制：
   - 接触传感器检测 (`ContactSensor`)：检测球体与机器人骨盆子树（即全身各刚体）的实际物理碰撞。
   - 速度不连续性穿模检测 (Velocity-Discontinuity Fallback)：
     当球速高达 12~15 m/s 时，薄壁肢体可能在单控制步内发生物理引擎穿模 (Tunneling)，导致接触传感器漏报。
     但在 MuJoCo 中无空气阻力，空中自由落体仅受重力作用：理论速度应满足 $v = v_{\\text{prev}} + g \\Delta t$。
     一旦球体实际速度偏离重力自由落体预测值，且发生在机器人躯干附近，即可 100% 断定球体撞击了机器人。
   - 课程学习门控 (`enable_after_steps`)：在训练初期抑制强行终止，允许策略先建立奔跑跳跃基础能力。

3. 墙体撞击检测 (`wall_contact`)：
   检测机器人肢体是否与场景中的静态或动态障碍墙发生触碰。

4. 延迟重置管理器 (`DelayedTerminationManager`)：
   为部分环境提供重置缓冲延迟，使机器人在倒地后继续仿真数步，为 Critic 网络提供更充分的惩罚价值衰减信号。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationManager
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def root_height_below_minimum_sustained(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  duration_s: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """当机器人根节点高度持续低于 minimum_height 超过 duration_s 秒时终止回合.

  设计意图：
  - 瞬时下蹲（如跳跃腾空落地时的深蹲卸力）不会触发终止；
  - 只有当机器人彻底倒地或陷入长时间“瘫地蠕行”的退化解时才会触发终止并施加跌倒惩罚。
  """
  asset: Entity = env.scene[asset_cfg.name]
  # 判定当前步高度是否低于阈值
  below = asset.data.root_link_pos_w[:, 2] < minimum_height

  # 获取或初始化持续过低步数计数器
  counter = getattr(env, "_low_crouch_counter", None)
  if counter is None or counter.shape[0] != env.num_envs:
    counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  # 若低于阈值则计数器累加 1，否则重置为 0
  counter = torch.where(below, counter + 1, torch.zeros_like(counter))
  env._low_crouch_counter = counter

  # 计算持续时间对应的离散仿真步数
  steps = max(1, round(duration_s / env.step_dt))
  return counter >= steps


def ball_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "ball_robot_contact",
  enable_after_steps: float = 0.0,
  ball_name: str = "ball",
  robot_name: str = "robot",
  delta_v_threshold: float = 1.5,
  hit_dist: float = 1.0,
  hit_z_min: float = 0.3,
  gravity: float = 9.81,
) -> torch.Tensor:
  """当飞来的球击中机器人身体任何部位时终止回合.

  双重检测机制：
  1. 读取接触传感器 `ball_robot_contact`：检测球体与骨盆子树的直接接触；
  2. 速度不连续性穿模检测：应对 12~15 m/s 高速球在薄连杆上的离散穿模问题，
     若球体速度相比单步重力自由落体预测发生突变（且位于身体一定高度范围内），同样判定为命中。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found  # [B, num_slots]
  assert found is not None, f"传感器 '{sensor_name}' 必须包含 'found' 字段。"
  # 传感器判定：接触点数量大于 0
  hit = found.squeeze(-1) > 0.0

  # 速度突变穿模检测补充逻辑
  if delta_v_threshold > 0.0:
    ball: Entity = env.scene[ball_name]
    robot: Entity = env.scene[robot_name]
    v = ball.data.root_link_lin_vel_w  # 当前球体线速度 [B, 3]
    prev = getattr(env, "_dodge_prev_ball_vel", None)
    if prev is None or prev.shape != v.shape:
      prev = v.clone()
    
    # 纯重力作用下一控制步后的理论速度预测 (仅 z 轴由于重力加速度减少 g * dt)
    expected = prev.clone()
    expected[:, 2] -= gravity * env.step_dt
    # 计算实际速度与理论自由落体速度的差值范数
    delta_v = (v - expected).norm(dim=-1)
    
    # 几何位置门控：必须在机器人根节点周围 hit_dist 范围内，且高度高于 hit_z_min（排除在脚边砸向地面的反弹）
    ball_pos = ball.data.root_link_pos_w  # [B, 3]
    dist = (ball_pos - robot.data.root_link_pos_w).norm(dim=-1)
    near_body = (dist < hit_dist) & (ball_pos[:, 2] > hit_z_min)
    dv_hit = (delta_v > delta_v_threshold) & near_body
    env._dodge_prev_ball_vel = v.clone()
    # 逻辑或合并两种击中判定
    hit = hit | dv_hit

  # 课程学习开关：在全局环境步数达到 enable_after_steps 之前屏蔽终止信号
  if enable_after_steps > 0.0 and env.common_step_counter < enable_after_steps:
    return torch.zeros_like(hit)
  return hit


def wall_contact(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...] = ("wall_robot_contact",),
) -> torch.Tensor:
  """当机器人的任何肢体与环境中的障碍墙发生碰撞时终止回合."""
  hit = None
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    found = sensor.data.found  # [B, num_slots]
    assert found is not None, f"传感器 '{name}' 必须包含 'found' 字段。"
    h = found.squeeze(-1) > 0.0
    hit = h if hit is None else (hit | h)
  return hit


class DelayedTerminationManager(TerminationManager):
  """对特定子集环境延迟触发物理重置的终止管理器.

  当延迟环境触发终止条件时，并不立刻执行 reset，而是启动计数器。
  在累积达到 max_delay_steps 之前保持跌倒/撞击后的物理演化，之后再执行真正的重置。
  """

  def __init__(
    self,
    base: TerminationManager,
    delay_env_mask: torch.Tensor,
    max_delay_steps: int,
  ) -> None:
    # 复制基础终止管理器的全部内部状态字典
    self.__dict__.update(base.__dict__)
    self._delay_env_mask = delay_env_mask          # (num_envs,) 延迟掩码
    self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
    self._max_delay_steps = max_delay_steps

  def compute(self) -> torch.Tensor:
    """计算终止与截断信号，并对延迟环境拦截重置."""
    dones = super().compute()  # 填充 _truncated_buf 与 _terminated_buf

    if self._max_delay_steps <= 0:
      return dones

    # 针对收到 done 信号的延迟环境，递增计数器
    delay_and_done = self._delay_env_mask & dones
    self._delay_counters[delay_and_done] += 1

    # 未达到最大延迟时长的环境：压制终止标志，阻止立即重置
    not_ready = delay_and_done & (self._delay_counters < self._max_delay_steps)
    self._terminated_buf[not_ready] = False

    # 达到延迟时长的环境：释放终止标志，清空计数器，允许重置
    ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
    self._delay_counters[ready] = 0

    # 对于在延迟期间自主恢复的环境，清除其计数器
    self._delay_counters[self._delay_env_mask & ~dones] = 0

    return self._truncated_buf | self._terminated_buf