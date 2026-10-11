"""AMP 判别器网络与风格奖励计算模块（AMP Discriminator & Style Reward）。

数学原理与算法架构：
对抗运动先验（Adversarial Motion Priors, AMP; Peng et al., 2021）采用对抗学习思想，
将模仿学习形式化为一个类似生成对抗网络（GAN）的极小极大优化博弈：
- 生成器（Generator）：即强化学习策略网络 $\pi_\theta(a|s)$，负责控制机器人在环境中运动；
- 判别器（Discriminator）：即本模块定义的 MLP 网络 $D_\phi(s, s')$，用于判断连续状态转移对 $(s, s')$
  是来自于真实动捕专家数据集（Expert），还是来自于强化学习策略的采样（Policy）。

关键机制解析：
1. 最小二乘损失函数（Least-Squares GAN Formulation, LSGAN）：
   相比于标准交叉熵损失，最小二乘损失能提供更平滑的梯度并有效避免梯度消失：
   $L_D = \frac{1}{2} \mathbb{E}_{d \sim \mathcal{M}} [(D(d) - 1)^2] + \frac{1}{2} \mathbb{E}_{d \sim \pi} [(D(d) + 1)^2]$
   即专家数据目标预测值为 +1，策略假样本目标预测值为 -1。

2. 梯度惩罚机制（Gradient Penalty, R1 Regularization / WGAN-GP 变体）：
   为使判别器满足 Lipschitz 连续性条件并稳定对抗训练过程，在专家数据样本流形上计算梯度二范数惩罚项：
   $L_{gp} = \frac{\lambda}{2} \mathbb{E}_{d \sim \mathcal{M}} [\|\nabla_d D(d)\|_2^2]$

3. AMP 风格奖励函数（AMP Reward Mapping）：
   判别器的输出评分 $d = D(s, s')$ 通过抛物线截断函数映射为非负的步态风格奖励：
   $r^{AMP}(s, s') = w_{amp} \cdot \max\left(0, 1 - \frac{1}{4} (d - 1)^2\right)$
   当策略生成的运动与动捕完全一致时（$d \approx 1$），风格奖励达到最大值 $w_{amp}$；
   当运动严重偏离专家模式时（$d \le -1$ 或 $d \ge 3$），奖励归零。

4. 风格奖励与任务奖励线性插值（Task-Reward LERP）：
   综合步态质量与目标完成度，最终传递给强化学习价值网络的步进奖励为两者的线性加权：
   $r = (1 - \alpha) \cdot r^{AMP} + \alpha \cdot r^{task}, \quad \text{其中 } \alpha = \text{task\_reward\_lerp}$
"""

from __future__ import annotations

import torch
from torch import nn


class Discriminator(nn.Module):
    """基于多层感知机（MLP）的 AMP 运动判别器网络。

    输入连续两帧的机体观测拼接向量 (s_t, s_{t+1})，输出用于判别真伪的无界标量得分。
    """

    def __init__(
        self,
        input_dim: int,
        amp_reward_coef: float = 0.1,
        hidden_layer_sizes: tuple[int, ...] = (1024, 512, 256),
        device: str = "cpu",
        task_reward_lerp: float = 0.75,
    ) -> None:
        """初始化判别器网络。

        Args:
            input_dim: 判别器输入维度，对应单步 AMP 观测维度的两倍 (2 * amp_obs_dim)。
            amp_reward_coef: AMP 风格奖励的基准缩放系数 (w_amp)。
            hidden_layer_sizes: 主干 MLP 隐藏层的神经元维度序列（默认 [1024, 512, 256]）。
            device: 运算设备。
            task_reward_lerp: 任务奖励插值权重 alpha，范围在 [0, 1]。
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

        # 构建主干全连接特征提取层（Linear + ReLU）
        layers: list[nn.Module] = []
        prev = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            prev = size
        self.trunk = nn.Sequential(*layers).to(device)
        # 输出层：将高维特征投影为单维标量预测得分
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播计算判别器预测得分。

        Args:
            x: 拼接的状态转移对 (s_t, s_{t+1})，形状为 (batch_size, input_dim)。

        Returns:
            形状为 (batch_size, 1) 的预测标量张量。
        """
        h = self.trunk(x)
        return self.amp_linear(h)

    def compute_grad_pen(
        self,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """在专家动捕数据流形上计算梯度惩罚项（R1 正则化）。

        计算公式：
            grad_pen = lambda * mean(||nabla_{expert_data} D(expert_data)||_2^2)

        通过惩罚专家点附近的判别器梯度范数，促使判别函数满足局部 1-Lipschitz 平滑性，
        有效消除对抗训练中的高频振荡，并防止判别器对微小扰动过于敏感。

        Args:
            expert_state: 专家当前状态张量。
            expert_next_state: 专家下一状态张量。
            lambda_: 梯度惩罚系数权重。

        Returns:
            标量形式的梯度惩罚损失。
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        # 对输入专家数据求一阶偏导数
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # 计算梯度二范数的平方均值
        grad_pen = lambda_ * (grad.norm(2, dim=1).pow(2)).mean()
        return grad_pen

    def predict_amp_reward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        normalizer=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据当前策略采样的环境转移对计算 AMP 风格奖励并与任务奖励混合。

        Args:
            state: 采样当前步的 AMP 观测 s_t。
            next_state: 采样下一步的 AMP 观测 s_{t+1}（若遇环境重置则自动使用终止前截断）。
            task_reward: 环境原生的外在任务奖励（Task Reward）。
            normalizer: 可选的在线均值方差归一化器。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - blended_reward: 混合了风格先验与任务目标的最终奖励标量；
                - disc_score: 判别器原始输出得分 d。
        """
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            # AMP 核心奖励映射：r_amp = w * max(0, 1 - 0.25 * (d - 1)^2)
            reward = self.amp_reward_coef * torch.clamp(
                1 - (1 / 4) * torch.square(d - 1), min=0
            )
            # 若启用了任务奖励插值，则进行混合加权
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward.squeeze(), task_reward)
            self.train()
        return reward.squeeze(), d

    def _lerp_reward(
        self, disc_r: torch.Tensor, task_r: torch.Tensor
    ) -> torch.Tensor:
        """执行风格奖励与任务奖励的线性插值（Linear Interpolation）。

        公式：
            r_total = (1 - alpha) * r_disc + alpha * r_task
        """
        return (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
