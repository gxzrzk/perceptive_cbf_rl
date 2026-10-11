"""Unitree G1 人形机器人敏捷跳跃（Leap）任务强化学习算法配置。

任务定位与先验数据集设定：
本配置继承自基础平地行走运行器配置（`g1_amp_ppo_runner_cfg`），关键差异在于：
1. 动捕数据集替换：将 AMP 判别器的输入动捕数据集重定向至 `src/assets/motions/g1/amp_leap`。
   该数据集专门包含 G1 人形机器人的各方向原地跳跃与飞扑动作片段（前跳、后跳、侧跳、对角跳跃等）；
2. 动作风格引导：因为判别器正样本全部为跳跃轨迹，强化学习策略被强力引导通过“离地腾空跳跃”
   而非平地行走来实现上层下达的速度/位移指令；
3. 独立的实验记录空间（experiment_name = "g1_amp_leap"），便于与行走任务分开记录曲线与检查点。
"""

import os

from .rl_cfg import RslRlAmpRunnerCfg, g1_amp_ppo_runner_cfg

# 跳跃动作专家动捕数据集所在路径（递归加载 Leap/ 与相关子目录）
_LEAP_MOTION_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
        "src", "assets", "motions", "g1", "amp_leap",
    )
)


def g1_amp_leap_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建 Unitree G1 敏捷跳跃任务的完整强化学习运行器配置。

    Returns:
        指向跳跃动捕数据集的 RslRlAmpRunnerCfg 配置实例。
    """
    cfg = g1_amp_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_leap"
    cfg.amp_motion_files = _LEAP_MOTION_DIR
    return cfg
