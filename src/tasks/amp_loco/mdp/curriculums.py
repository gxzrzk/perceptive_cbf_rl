"""AMP 强化学习任务的课程学习 (Curriculum) 调度项模块.

在复杂的腿足机器人运动任务中，直接在大范围空间内训练往往导致探索难度过大、策略崩塌。
课程学习通过在训练过程中逐步调节环境难度参数（例如逐步扩大目标点的采样半径），
使得智能体先在近距离学会基础站立、踏步和短距离位移，再平滑过渡到长距离敏捷奔跑与跳跃。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def goal_radius_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  start_radius: float = 1.0,
  final_radius: float = 4.0,
  steps_to_full: float = 40_000.0,
) -> float:
  """目标采样半径课程学习：随训练推进将目标距离从 `start_radius` 逐步线性放大到 `final_radius`.

  机制说明：
  - 进度依赖于全局环境步数计数器 `env.common_step_counter`（每次环境 step 递增 1，
    每个训练 iteration 约增加 `num_steps_per_env`，默认 24 步）。
    关键是该计数器在单回合 reset 时**不会被清零**，能稳定反映全局训练进程。
  - 在前 `steps_to_full` 步（例如 40k 步 / 24 步/iter ≈ 1.7k iterations）内，
    目标采样最大半径随步数线性增加至 `final_radius`，之后维持最大半径。
  - 返回当前计算得到的最大采样半径，便于在 WandB / TensorBoard 中记录监控。

  参数:
    env: 管理型强化学习环境实例
    env_ids: 触发更新的环境索引
    command_name: 目标指令项名称（如 "goal_command"）
    start_radius: 初始最大采样半径（米，默认 1.0m）
    final_radius: 最终最大采样半径（米，默认 4.0m）
    steps_to_full: 达到最终最大半径所需的全局环境步数（默认 40,000）

  返回:
    当前设定的最大目标距离标量值
  """
  # 获取环境中的目标指令管理器项
  command = env.command_manager.get_term(command_name)
  # 计算当前全局进度的比例系数 [0.0, 1.0]
  frac = min(env.common_step_counter / steps_to_full, 1.0)
  # 线性插值计算新的最大采样半径
  new_max = start_radius + frac * (final_radius - start_radius)
  # 动态更新指令项配置中的半径范围 (最小半径保持不变，更新最大半径)
  command.cfg.radius = (command.cfg.radius[0], new_max)
  return new_max
