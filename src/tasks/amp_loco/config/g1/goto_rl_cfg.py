"""Unitree G1 人形机器人跳跃目标导航（Leap-to-Goal）与躲避球任务系列强化学习算法配置。

本模块汇总定义了论文中各种进阶感知模式与环境变体的 RL 运行器配置：
1. 跳跃目标导航（Leap Go-To-Goal）：保持跳跃运动先验，将速度指令改为目标位移向量驱动；
2. 基础躲避球任务（Base Dodge）：融合敏捷跳跃与站立闪避动捕数据集（`amp_dodge`）；
3. 状态真值环境（State Oracle / MimicKit）：
   - 对称 Actor-Critic：Actor 与 Critic 均观测小球在航向坐标系下的真实 3D 位置与速度（`ball_state`）；
   - 保持 50/50 的风格与任务奖励平衡（amp_reward_coef=0.5, amp_task_reward_lerp=0.5），防止动作退化；
4. 走廊巡航避障（WallWalk）：
   - 采用行走增强运动先验（`amp_dodge_walk`）：在跳跃与闪避基础上并入 17 个行走/跑步动捕片段，
     使得策略在维持 1.3 m/s 巡航速度的同时认可行走步态，避免风格判别器抵触前进动作；
5. 教室导航避障（Classroom）：
   - 在复杂课桌椅阵列中穿梭，采用与 WallWalk 相同的增强运动集；
6. 深度视觉感知（Depth Vision & Single Level Camera）：
   - 非对称 Actor-Critic：Actor 仅观测本体感觉 + 展平的单目深度图（`depth`），Critic 额外观测特权真值（`critic_dodge`）。
"""

import os

from .leap_rl_cfg import g1_amp_leap_ppo_runner_cfg
from .rl_cfg import RslRlAmpRunnerCfg

# 躲避球基础动捕数据集路径：包含 Leap/ 跳跃片段与 Dodge/ 站立敏捷回避片段
_DODGE_MOTION_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
        "src", "assets", "motions", "g1", "amp_dodge",
    )
)

# WallWalk 与 Classroom 巡航任务动捕数据集路径：
# 在 amp_dodge 基础上软链接并合入了全部 17 个 WalkandRun 行走/慢跑/侧移片段（约占 45% 样本帧数），
# 使得判别器不仅接纳紧急躲避姿态，也将 1.3 m/s 的稳定巡航步态判定为正样本。
_WALLWALK_MOTION_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
        "src", "assets", "motions", "g1", "amp_dodge_walk",
    )
)


def g1_amp_leap_goto_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建 G1 跳跃目标点导航（Leap GoTo）任务的运行器配置。"""
    cfg = g1_amp_leap_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_leap_goto"
    cfg.save_interval = 1000  # 每 1000 轮保存一次检查点并导出 ONNX
    return cfg


def g1_amp_dodge_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建基础躲避球任务运行器配置：将判别器输入切换为跳跃+闪避混合动捕集（amp_dodge）。"""
    cfg = g1_amp_leap_goto_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge"
    cfg.amp_motion_files = _DODGE_MOTION_DIR
    return cfg


def g1_amp_dodge_depth_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建单目深度相机视觉感知躲避球任务运行器配置。

    核心特性：
    1. 观测空间配置（非对称 Actor-Critic）：
       - Actor 观测：本体感觉历史 ('actor') + 归一化展平的深度图 ('depth')，直接输入 MLP，不经过 CNN，
         因为头部固定相机在特定像素位置直接提供了目标方位的绝对视线信息；
       - Critic 观测：Actor 观测 + 特权状态 ('critic_dodge'，包含小球位置速度真值与 CBF 障碍物距离)；
    2. 动作风格保护（50/50 奖励平衡）：
       - amp_reward_coef=0.5, amp_task_reward_lerp=0.5，确保躲避球任务奖励不会冲淡跳跃与类人风格先验。
    """
    cfg = g1_amp_dodge_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge_depth"
    cfg.amp_reward_coef = 0.5
    cfg.amp_task_reward_lerp = 0.5
    cfg.obs_groups = {
        "actor": ("actor", "depth"),
        "critic": ("critic", "depth", "critic_dodge"),
    }
    return cfg


def g1_amp_dodge_mimickit_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建状态真值环境（State Oracle / MimicKit 风格）的躲避球运行器配置。

    特性：
    - 对称 Actor-Critic 架构：Actor 与 Critic 均观测小球在航向坐标系下的相对位置与速度（'ball_state'）；
    - 保留 50/50 的风格保护插值加权；
    - 全 MLP 策略模型（输入为 6 维小球状态，无深度图像）。
    """
    cfg = g1_amp_dodge_depth_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge_mimickit"
    cfg.obs_groups = {
        "actor": ("actor", "ball_state"),
        "critic": ("critic", "ball_state"),
    }
    return cfg


def g1_amp_dodge_mimickit_wallwalk_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建走廊巡航避障（WallWalk）任务的强化学习运行器配置。

    特性：
    - 对称观测小球与墙体状态；
    - 判别器采用融合了行走动捕的增强数据集 `amp_dodge_walk`，使模型在前进巡航和躲避飞球间流畅自如。
    """
    cfg = g1_amp_dodge_mimickit_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge_mimickit_wallwalk"
    cfg.amp_motion_files = _WALLWALK_MOTION_DIR
    return cfg


def g1_amp_dodge_mimickit_classroom_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建真实教室课桌椅穿梭避障（Classroom）任务的强化学习运行器配置。"""
    cfg = g1_amp_dodge_mimickit_wallwalk_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge_mimickit_classroom"
    cfg.amp_reward_coef = 0.5
    cfg.amp_task_reward_lerp = 0.5
    return cfg


def g1_amp_dodge_depth_mimickit_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建结合单目深度视觉观测与 MimicKit 任务奖励的躲避球运行器配置。"""
    cfg = g1_amp_dodge_depth_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge_depth_mimickit"
    return cfg


def g1_amp_dodge_depth_single_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
    """构建搭载单目水平相机的深度视觉躲避球任务运行器配置。

    具有独立的实验名称（g1_amp_dodge_depth_single），便于与其它深度实验独立对照与记录。
    """
    cfg = g1_amp_dodge_depth_mimickit_ppo_runner_cfg()
    cfg.experiment_name = "g1_amp_dodge_depth_single"
    return cfg
