"""AMP 观测特征的在线滑动均值与方差归一化器（Running Mean/Std Normalizer）。

核心设计与性能优化：
在对抗运动先验（AMP）中，判别器需要接收机器人当前状态与下一状态拼接后的转移特征 (s_t, s_{t+1})。
由于各关节角度、线速度、角速度的物理量纲和数值尺度差异极大，必须对输入特征进行在线动态归一化。

本模块直接基于 PyTorch 张量实现，并全程驻留在 GPU 端（利用 float64 累加器保证长时间训练的数值稳定性）：
1. 避免 GPU-CPU 阻塞同步：传统 NumPy 实现会在每个 mini-batch 强制执行 `.cpu().numpy()` 拷贝并在 CPU 端
   计算 `np.mean`/`np.var`，造成严重的流水线气泡与 GPU 闲置等待；本实现所有统计计算均在设备端并行完成；
2. 数学原理：严格采用 Welford 并行流式方差更新算法（Welford's algorithm for parallel variance calculation）；
3. 懒加载缓存：内部维护 float32 类型的均值与标准差张量缓存，在无统计量更新时避免重复类型转换与开方运算。
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """基于 Welford 并行算法的流式均值与方差统计器（全 GPU 张量实现）。

    参考资料：
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        device: str | torch.device = "cuda:0",
    ) -> None:
        """初始化流式统计器。

        Args:
            epsilon: 平滑小项，防止初始阶段方差为零或除以零。
            shape: 统计量的维度形状（通常对应特征维度，如 (obs_dim,)）。
            device: 统计张量存放的计算设备。
        """
        self.device = torch.device(device)
        # 使用 float64 双精度累加器，确保超长训练周期（数千万步）下的数值稳定性与抗舍入误差能力
        self.mean = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.var = torch.ones(shape, dtype=torch.float64, device=self.device)
        self.count = float(epsilon)

    @torch.no_grad()
    def update(self, arr: torch.Tensor) -> None:
        """使用当前批次的输入数据更新全局滑动均值与方差。

        Args:
            arr: 当前批次的输入张量，形状为 (batch_size, *shape)。
        """
        arr = arr.to(self.mean.device, torch.float64)
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0, unbiased=False)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: int
    ) -> None:
        """根据当前小批次的样本统计矩（均值、方差、样本数）融合更新全局滑动统计量。

        采用 Chan 等人的并行 Welford 增量合并公式：
        delta = mean_b - mean_a
        M2_total = M2_a + M2_b + delta^2 * (count_a * count_b / (count_a + count_b))
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        )
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class Normalizer(RunningMeanStd):
    """集成在线标准化（Z-Score）与截断（Clipping）功能的 AMP 观测归一化器。"""

    def __init__(
        self,
        input_dim: int | tuple[int, ...],
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
        device: str | torch.device = "cuda:0",
    ) -> None:
        """初始化归一化器。

        Args:
            input_dim: 观测向量维度。
            epsilon: 数值稳定性常数，防止标准差为 0。
            clip_obs: 归一化后的极大值截断门限 [-clip_obs, clip_obs]，抑制极端异常值。
            device: 目标设备。
        """
        super().__init__(
            shape=input_dim if isinstance(input_dim, tuple) else (input_dim,),
            device=device,
        )
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        # 用于快速前向推理的 float32 缓存
        self._mean_f32: torch.Tensor | None = None
        self._std_f32: torch.Tensor | None = None

    def _refresh_cache(self, device: torch.device) -> None:
        """在目标计算设备上刷新 float32 类型的均值与标准差张量缓存。"""
        self._mean_f32 = self.mean.to(device, torch.float32)
        self._std_f32 = torch.sqrt((self.var + self.epsilon).to(device, torch.float32))

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:  # type: ignore[override]
        """更新统计矩并使先前的 float32 缓存失效，以便在下次归一化时懒加载刷新。"""
        super().update_from_moments(batch_mean, batch_var, batch_count)
        self._mean_f32 = None  # 标记缓存失效

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """针对 NumPy 数组进行标准化与截断（主要用于离线评估或兼容旧接口）。"""
        mean = self.mean.detach().cpu().numpy()
        var = self.var.detach().cpu().numpy()
        return np.clip(
            (x - mean) / np.sqrt(var + self.epsilon), -self.clip_obs, self.clip_obs
        )

    def normalize_torch(self, x: torch.Tensor, device: str | torch.device) -> torch.Tensor:
        """对 PyTorch 张量进行纯 GPU 端无同步延迟的归一化与截断。

        公式：
            x_norm = clamp((x - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)

        Args:
            x: 待归一化的原始观测张量。
            device: 输出张量所在设备。

        Returns:
            归一化并截断后的观测张量。
        """
        dev = torch.device(device)
        if self._mean_f32 is None or self._mean_f32.device != dev:
            self._refresh_cache(dev)
        return torch.clamp(
            (x - self._mean_f32) / self._std_f32, -self.clip_obs, self.clip_obs
        )
