"""AMP 判别器策略样本经验回放缓冲区（Replay Buffer for AMP Discriminator）。

功能与架构定位：
在对抗运动先验（AMP）训练中，判别器需要区分真实动捕专家样本（Expert Transitions）
与当前强化学习策略生成的假样本（Policy Transitions）。
若判别器仅使用当前最新一个 rollout 迭代中的策略样本，会导致判别器遗忘早期的行为模式或训练剧烈震荡。

本模块提供了一个固定容量的循环经验缓冲区（Circular FIFO Replay Buffer）：
1. 存储近期策略生成的 `(state, next_state)` 状态转移对；
2. 环形覆盖机制：当写入数据超过容量上限时自动从头部回环覆盖最早的历史样本；
3. 全 GPU 端随机采样：使用 `torch.randint` 直接在显存中生成均匀采样索引并收集小批次，
   避免了 CPU 索引数组传输到 GPU 的额外开销。
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    """存储策略历史 AMP 观测转移 (s_t, s_{t+1}) 的循环回放缓冲区。"""

    def __init__(self, obs_dim: int, buffer_size: int, device: str | torch.device) -> None:
        """初始化经验回放缓冲区。

        Args:
            obs_dim: AMP 观测向量的维度大小。
            buffer_size: 缓冲区能够容纳的最大样本对总数。
            device: 显存分配的目标设备（如 "cuda:0"）。
        """
        # 在显存中预先分配整块连续张量空间，存储状态与下一状态
        self.states = torch.zeros(buffer_size, obs_dim, device=device)
        self.next_states = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device
        # 当前环形写指针位置
        self.step = 0
        # 当前已有效填充的样本总数（不超过 buffer_size）
        self.num_samples = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        """向缓冲区中批量插入一组新的状态转移数据。

        处理环形写指针越界回环：若写入长度跨越缓冲区末尾，则自动分两段切片分别存入末尾与缓冲区起始位置。

        Args:
            states: 当前状态批次张量，形状为 (num, obs_dim)。
            next_states: 下一状态批次张量，形状为 (num, obs_dim)。
        """
        num = states.shape[0]
        start = self.step
        end = self.step + num

        if end > self.buffer_size:
            # 第一段：填充从当前指针到缓冲区末尾的空间
            first_chunk = self.buffer_size - self.step
            self.states[self.step : self.buffer_size] = states[:first_chunk]
            self.next_states[self.step : self.buffer_size] = next_states[:first_chunk]
            # 第二段：回环从索引 0 开始填充剩余部分
            remainder = end - self.buffer_size
            self.states[:remainder] = states[first_chunk:]
            self.next_states[:remainder] = next_states[first_chunk:]
        else:
            self.states[start:end] = states
            self.next_states[start:end] = next_states

        # 更新已存样本总量与指针位置
        self.num_samples = min(self.buffer_size, max(end, self.num_samples))
        self.step = (self.step + num) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """生成用于前向传播与反向传播的策略状态转移小批次生成器。

        直接在 GPU 端调用 `torch.randint` 生成随机索引进行切片采样（有放回均匀采样），
        确保极高的采样吞吐量。

        Args:
            num_mini_batch: 需要生成的小批次总数量。
            mini_batch_size: 每个小批次的样本数量。

        Yields:
            tuple[torch.Tensor, torch.Tensor]: 采样的 (states_mb, next_states_mb)，
            形状均为 (mini_batch_size, obs_dim)。
        """
        for _ in range(num_mini_batch):
            # 在 GPU 上直接采样索引，避免 CPU->GPU 主机拷贝
            idx = torch.randint(
                0, self.num_samples, (mini_batch_size,), device=self.states.device
            )
            yield self.states[idx], self.next_states[idx]
