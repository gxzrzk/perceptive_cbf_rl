"""评估指标 (Metrics) 计算模块.

用于在强化学习训练和评估过程中，提取和监控策略的运行统计信息（如延迟终止的平均等待步数）。
指标由指标管理器 (Metrics Manager) 周期性聚合记录并上传至日志系统 (如 WandB)。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mean_delay_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
  """统计处于延迟终止 (Delayed Termination) 状态的环境的平均延迟步数.

  当启用延迟终止（智能体在触发失败后不立即重置，而是等待数步以观察倒地或碰撞后果）时，
  该指标用于监控所有延迟环境当前的平均累积等待时长。

  参数:
    env: 管理型强化学习环境实例

  返回:
    形状为 `(num_envs,)` 的张量，每个环境均填充相同的全局平均延迟步数，便于统一日志记录。
  """
  tm = env.termination_manager
  # 获取延迟终止管理器内部维护的延迟计数器和延迟掩码
  delay_counters = getattr(tm, "_delay_counters", None)
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  
  if delay_env_mask is not None and delay_counters is not None:
    # 统计所有延迟环境的总延迟步数与环境总数
    total_delay_steps = torch.sum(delay_counters.float())
    total_delay_envs = torch.sum(delay_env_mask.float())
    # 计算均值（加 1e-8 防止除零）
    mean_delay = total_delay_steps / (total_delay_envs + 1e-8)
    # 扩展为每个环境相同的标量张量
    return mean_delay.expand(env.num_envs)
  else:
    # 若未启用延迟终止，则返回全零张量
    return torch.zeros(env.num_envs, device=env.device)