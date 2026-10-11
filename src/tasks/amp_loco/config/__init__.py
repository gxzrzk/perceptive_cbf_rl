"""AMP 任务配置模块（Task Configurations for AMP Locomotion）。

本包集中管理基于对抗运动先验（AMP）的各种人形机器人强化学习环境与算法超参数配置。
子目录按机器人硬件型号进行组织（如 `g1` 对应宇树 Unitree G1 人形机器人）。

配置内容涵盖：
1. 环境配置（Environment Configurations）：定义仿真步长、刚体与关节参数、观测组（Observations）、
   动作空间（Actions）、速度/目标指令（Commands）、随机化与重置事件（Events）、奖励函数（Rewards）
   与终止条件（Terminations）；
2. 强化学习运行器配置（RL Runner Configurations）：定义 Actor/Critic 网络结构、PPO 超参数、
   AMP 判别器（Discriminator）结构与损失权重、专家运动数据集路径等。
"""
