"""AMP 强化学习环境包装器模块（RL Environment Wrapper Module）。

本模块预留用于存放针对 AMP 强化学习任务的定制化环境包装器（Environment Wrappers）。
在标准训练流程中，环境通过 `mjlab.rl.RslRlVecEnvWrapper` 进行标准向量化封装；
该模块可用于扩展自定义的观测预处理、动作后处理、辅助度量监控或多智能体交互逻辑。
"""
