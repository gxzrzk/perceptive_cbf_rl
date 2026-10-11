"""Unitree G1 人形机器人基础 AMP 运动控制强化学习算法配置（RL Configuration）。

配置架构与超参数定位：
本模块定义了针对 Unitree G1 双足机器人的基础速度跟踪任务强化学习运行器与算法超参数。
算法基于 `rsl_rl`（版本 5.2.0）框架并扩展了 AMP 核心组件：
1. `AmpAlgorithmCfg`：在 PPO 算法配置基础上，注入判别器损失权重（amp_loss_coef）、
   梯度惩罚系数（amp_grad_pen_lambda）以及判别器网络主干与输出头的独立权重衰减；
2. `RslRlAmpRunnerCfg`：定义运行器顶层的 AMP 超参数，包括动捕数据文件路径、跟踪的关键连杆列表、
   局部坐标系锚点刚体（torso_link）、判别器隐藏层架构（[1024, 512, 256]）与策略状态转移经验池大小；
3. `g1_amp_ppo_runner_cfg()`：实例化具体的 G1 机器人基础步态训练配置。
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

# 默认 AMP 动捕专家数据目录（存放 G1 的 WalkandRun 基础行走与奔跑 npz 文件）
_MOTION_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
    "src", "assets", "motions", "g1", "amp",
)


@dataclass
class AmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """AMP-PPO 算法配置类：继承自标准 PPO 配置，新增对抗判别器损失项与超参数。"""

    amp_loss_coef: float = 1.0
    """判别器最小二乘损失与梯度惩罚在总损失中的加权系数。"""
    amp_grad_pen_lambda: float = 10.0
    """WGAN-GP 风格的专家样本 R1 梯度惩罚系数（稳定判别器训练，满足 Lipschitz 平滑性）。"""
    amp_trunk_weight_decay: float = 1.0e-3
    """判别器主干全连接层（Trunk）的 L2 权重衰减系数。"""
    amp_head_weight_decay: float = 1.0e-2
    """判别器最终标量输出层（Head）的 L2 权重衰减系数。"""
    class_name: str = "src.tasks.amp_loco.rl.amp_algorithm:AmpPPO"
    """动态解析并实例化的算法类入口路径。"""


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
    """AMP 任务运行器顶层配置类。

    声明在顶层的 `amp_*` 字段在初始化阶段会被 `AmpPPO.construct_algorithm` 提取并用于
    构建专家数据加载器、判别器网络与经验回放池。
    """

    amp_reward_coef: float = 0.1
    """AMP 判别器风格奖励的缩放系数 w_amp。"""
    amp_motion_files: str = ""
    """动捕专家数据文件路径（支持单个 .npz 文件或包含子目录的根文件夹路径）。"""
    amp_task_reward_lerp: float = 0.75
    """风格奖励与任务奖励的线性插值权重：r = (1 - l) * r_amp + l * r_task。"""
    amp_discr_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512, 256])
    """判别器 MLP 主干的隐藏层神经元数量序列。"""
    amp_replay_buffer_size: int = 100_000
    """策略状态转移经验回放缓冲区容量上限（最大存储样本对数量）。"""
    amp_body_names: Tuple[str, ...] = ()
    """用于构建 AMP 观测特征的关键跟踪连杆名称元组。"""
    amp_anchor_name: str = ""
    """用于构建身体局部相对坐标系的基准锚点刚体名称（通常为 "torso_link"）。"""

    algorithm: AmpAlgorithmCfg = field(default_factory=AmpAlgorithmCfg)
    """内部算法超参数配置对象。"""


def g1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """创建 Unitree G1 基础 AMP 平地运动控制任务的完整强化学习运行器配置。

    超参数设定：
    - Actor/Critic 架构：3 层 MLP [512, 256, 128]，ELU 激活函数，内置在线观测标准化；
    - 高斯策略动作分布：标量方差，初始标准差 1.0，方差下限截断在 0.05（防止策略退化为确定性）；
    - PPO 参数：步长 24 步/环境，4 个 mini-batches，5 个更新 epochs，学习率 1e-3（自适应 KL 调整）；
    - 奖励混合：amp_reward_coef=0.3, amp_task_reward_lerp=0.6，赋予动作风格足够影响力。
    """
    return RslRlAmpRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
                # 将动作探索噪声标准差下限钳位在 0.05，确保在长时间训练后依然维持适度探索，避免步态僵死
                "std_range": (0.05, 1.0e6),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=AmpAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",  # 依据实际 KL 散度动态缩放学习率
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_amp_locomotion",
        logger="wandb",
        wandb_project="humanoid_dodgeball",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=100001,
        # AMP 超参数设置：提高风格奖励权重（coef=0.3, lerp=0.6），有效惩罚不合规范的擦地或深蹲动作
        amp_reward_coef=0.3,
        amp_motion_files=os.path.normpath(_MOTION_DATA_DIR),
        amp_task_reward_lerp=0.6,
        amp_discr_hidden_dims=[1024, 512, 256],
        amp_replay_buffer_size=100_000,
        # G1 机器人的 13 个核心运动判别连杆（涵盖骨盆、双腿三关节、双臂肩肘腕）
        amp_body_names=(
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ),
        amp_anchor_name="torso_link",  # 以躯干作为局部相对坐标系原点
    )
