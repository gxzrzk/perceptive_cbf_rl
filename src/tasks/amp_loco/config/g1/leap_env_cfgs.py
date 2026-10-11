"""Unitree G1 人形机器人敏捷跳跃（Leap）环境配置。

环境设计与跳跃动力学设定：
本环境旨在训练 G1 机器人在接收到水平速度指令时，通过大幅度离地腾空的敏捷跳跃（Jumping）
来实现位移，而不是通过常规的双足交替行走步态。

关键配置与物理约束修改：
1. 动捕数据集与重置（RSI）：
   - 采用 `src/assets/motions/g1/amp_leap/Leap` 数据集（包含基于真实人类动作捕捉的前、后、左、右与对角跳跃）；
   - 禁用跌倒恢复延迟重置（`delay_reset_env_ratio=0`），因为跳跃任务不包含匍匐起立阶段；
2. 奖励塑形裁剪：
   - 移除 `track_root_height` 奖励：传统站立高度跟踪会惩罚腾空跳跃的高度变化，移除后消除对跳跃飞行的抑制；
3. 速度指令范围校准（Command Envelope）：
   - 根据真实跳跃动捕数据的速度统计（scripts/clip_velocity_stats.py），设置水平速度范围：
     前进/后退 vx 范围 `[-1.1, 1.3] m/s`，横向跳跃 vy 范围 `[-1.4, 1.4] m/s`；
4. 深度下蹲与擦地防作弊终止（Anti-Scraping / Collapsed Crouch Termination）：
   - 动捕中真实跳跃的根节点高度最低不低于 0.65m；为防止策略通过深蹲贴地滑行（Scraping）逃避跳跃腾空带来的不稳定，
     设定持续下蹲终止判定：若基座高度低于 0.55m 持续超过 0.3s，则判定为失败跌倒并处以 -200 终止惩罚。
"""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

import src.tasks.amp_loco.mdp as mdp

from .env_cfgs import g1_amp_flat_env_cfg, g1_amp_rough_env_cfg

# 跳跃动捕根路径
_MOTION_BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "g1", "amp_leap"
    )
)
# 判别器加载包含所有跳跃子目录的数据集根目录
_LEAP_DISCR_DIR = _MOTION_BASE
# 环境重置采样安全片段目录（仅从 Leap/ 目录重置，确保初始帧在空中或站立平稳状态）
_LEAP_RESET_DIR = os.path.join(_MOTION_BASE, "Leap")


def _apply_leap_overrides(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
    """就地修改传入的基础 AMP 环境配置，应用敏捷跳跃专用覆盖项。

    Args:
        cfg: 基础平地或崎岖地形环境配置。
        play: 是否为可视化推理/演示模式。

    Returns:
        修改后的环境配置对象。
    """
    # 1. 动捕数据集与重置配置：纯跳跃片段，禁用跌倒恢复逻辑
    cfg.events["init_motion_loader"].params["motion_dir"] = _LEAP_RESET_DIR
    cfg.events["init_motion_loader"].params["recovery_dir"] = None
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
    cfg.events["init_motion_loader"].params["max_delay_steps"] = 0
    cfg.events["reset_from_motion"].params["motion_dir"] = _LEAP_RESET_DIR

    # 2. 移除站立高度跟踪奖励：避免该奖励对抗跳跃腾空高度变化
    cfg.rewards.pop("track_root_height", None)

    # 3. 速度指令生成器配置：支持大范围水平跳跃与航向保持
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.resampling_time_range = (3.0, 6.0)
    twist.rel_standing_envs = 0.1  # 10% 概率静止站立
    twist.rel_heading_envs = 1.0   # 100% 启用航向跟踪
    twist.heading_command = True
    # 速度指令范围严格匹配真实跳跃动力学极限
    twist.ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.1, 1.3),
        lin_vel_y=(-1.4, 1.4),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi / 2, math.pi / 2),
    )

    # 4. 终止判定条件：抑制异常深蹲与擦地作弊
    cfg.terminations["bad_base_height"].params["minimum_height"] = 0.45
    cfg.terminations["bad_orientation"].params["limit_angle"] = math.radians(80.0)
    # 若基座高度低于 0.55m 持续超过 0.3s，则判定为策略瘫塌失效触发终止
    cfg.terminations["collapsed_crouch"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum_sustained,
        params={"minimum_height": 0.55, "duration_s": 0.3},
    )

    if play:
        twist.ranges.lin_vel_x = (-1.1, 1.3)
        twist.ranges.lin_vel_y = (-1.4, 1.4)
        twist.ranges.ang_vel_z = (-0.5, 0.5)

    return cfg


def g1_amp_leap_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Unitree G1 平地敏捷跳跃任务环境配置。"""
    cfg = g1_amp_flat_env_cfg(play=play)
    return _apply_leap_overrides(cfg, play)


def g1_amp_leap_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Unitree G1 崎岖地形敏捷跳跃任务环境配置。"""
    cfg = g1_amp_rough_env_cfg(play=play)
    return _apply_leap_overrides(cfg, play)
