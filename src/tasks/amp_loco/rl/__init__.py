"""AMP 强化学习核心算法与运行器模块（AMP RL Core Package）。

本包封装了结合对抗运动先验（AMP）的强化学习算法组件：
1. `AmpPPO`：继承自上游 `rsl_rl.algorithms.PPO`，内嵌 AMP 判别器、专家经验数据加载器（AMPLoader）、
   在线运动观测归一化器（Normalizer）以及策略状态转移经验回放池（ReplayBuffer）；
   在环境采样阶段实时预测 AMP 风格奖励并注入 PPO 回合，并在优化阶段将策略与判别器联合迭代更新。
2. `AMPOnPolicyRunner`：继承自 `MjlabOnPolicyRunner`，支持模型检查点保存、Wandb 同步与 ONNX 模型导出。
"""

from .amp_algorithm import AmpPPO as AmpPPO
from .runner import (
  AMPOnPolicyRunner as AMPOnPolicyRunner,
)
