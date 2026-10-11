"""Unitree G1 人形机器人跳跃目标导航（Leap-to-Goal）环境配置。

任务架构与控制分层：
本任务是跳跃任务的目标条件化（Goal-Conditioned）变体：
将原本随机采样的速度指令替换为基于目标点的闭环速度生成器 `GoToGoalCommand`：
1. 分层控制接口（Hierarchical Control Decomposition）：
   - 上层导航项随机采样平面目标点位置，并在每个仿真步计算机器人当前基座到目标点的误差向量，
     通过比例控制器（P 控制器）将其转换为水平期望速度指令（`twist`）；
   - 下层策略的观测空间完全不包含绝对全局目标坐标，策略仅仅接收该衰减的速度指令。
     这种架构严格契合真机部署分层（全局路径规划器 -> 局部速度跟踪策略）；
2. 全向闪避特性（Dodge-style Omnidirectional Leaps）：
   - 设置 `simple_heading=False`：机器人前往目标点时“不主动旋转身体面朝目标”，
     而是保持躯干面朝前方（面向潜在对手或来球方向），通过向侧方、后方或斜向的全向跳跃快速移动至目标点；
3. 密集奖励塑形与精准停驻（Goal Rewards & Precise Arrival）：
   - 包含目标距离高斯衰减奖励（`goal_distance`）、朝向目标前进进度奖励（`goal_progress`）、
     到达目标奖励（`goal_reached`）、目标点停驻制动惩罚（`stop_at_goal`）以及双足真正离地腾空奖励（`leap_flight`）；
4. 目标距离课程学习（Curriculum Learning）：
   - 目标采样半径随训练步数从 0.5 米逐步扩展至 2.0 米，引导策略循序渐进地掌握远距离跳跃与近距离微调。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.amp_loco.mdp as mdp
from src.tasks.amp_loco.mdp.goal_command import GoToGoalCommandCfg

from .leap_env_cfgs import g1_amp_leap_flat_env_cfg, g1_amp_leap_rough_env_cfg

_ANCHOR = "torso_link"

# 强制静止站立的环境比例：10% 的环境目标点重合在原地（指令为 0），促使策略维持平稳站立
_STAND_RATIO = 0.1


def _apply_goto_overrides(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
    """就地修改传入的跳跃环境配置，应用目标点导航指令与奖励函数。

    Args:
        cfg: 基础跳跃环境配置。
        play: 是否处于可视化演示模式。

    Returns:
        修改后的目标导航跳跃环境配置。
    """
    # 1. 替换原生速度指令生成器为基于目标点偏差的速度生成器
    cfg.commands["twist"] = GoToGoalCommandCfg(
        entity_name="robot",
        resampling_time_range=(6.0, 10.0),
        radius=(0.5, 2.0),           # 目标点采样半径范围 [m]（训练初期由课程控制）
        rel_standing_envs=_STAND_RATIO,
        kp=1.5,                      # 位置偏差转速度指令的比例增益
        kp_yaw=1.0,                  # 偏航角增益
        max_lin_vel_x=1.3,           # 最大前进/后退速度截断 [m/s]
        max_lin_vel_y=1.5,           # 最大侧跳速度截断 [m/s]
        max_ang_vel_z=0.5,           # 最大转向角速度截断 [rad/s]
        arrive_radius=0.25,          # 判定到达目标点的容差半径 [m]
        dwell_time_range=(1.0, 3.0), # 到达目标点后的驻留等待时间 [s]
        # 设为 False 表示不转向面朝目标点，强制保持机体航向角，锻炼全向侧跃与后跃回避能力
        simple_heading=False,
        debug_vis=True,
    )

    # 2. 目标点导航与跳跃飞行奖励配置
    # 目标距离指数衰减奖励（距离越近奖励越高）
    cfg.rewards["goal_distance"] = RewardTermCfg(
        func=mdp.goal_distance_reward,
        weight=1.0,
        params={"command_name": "twist", "std": 1.5},
    )
    # 单步朝向目标的位移投影进展奖励
    cfg.rewards["goal_progress"] = RewardTermCfg(
        func=mdp.goal_progress_reward,
        weight=0.5,
        params={"command_name": "twist", "clip": 2.0},
    )
    # 成功进入目标半径且速度减缓的到达奖励
    cfg.rewards["goal_reached"] = RewardTermCfg(
        func=mdp.goal_reached_bonus,
        weight=2.0,
        params={
            "command_name": "twist",
            "threshold": 0.3,
            "speed_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 惩罚在目标点附近依然保持高速运动（鼓励停稳制动）
    cfg.rewards["stop_at_goal"] = RewardTermCfg(
        func=mdp.stop_at_goal_cost,
        weight=-0.5,
        params={
            "command_name": "twist",
            "threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 腾空飞行相奖励：奖赏双足真正离地的阶段（腾空时间超过 50ms 触发），避免脚底粘地滑行作弊
    cfg.rewards["leap_flight"] = RewardTermCfg(
        func=mdp.leap_flight_time,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "min_flight_s": 0.05,
            "command_threshold": 0.1,
        },
    )

    # 3. 课程学习：随着训练步数线性增加目标采样半径（从 0.5m 扩展到 2.0m）
    if not play:
        cfg.curriculum = dict(cfg.curriculum) if cfg.curriculum else {}
        cfg.curriculum["goal_radius"] = CurriculumTermCfg(
            func=mdp.goal_radius_curriculum,
            params={
                "command_name": "twist",
                "start_radius": 0.5,
                "final_radius": 2.0,
                # 经历 40,000 个环境交互步（约 1700 次训练迭代）完全扩展至最大距离
                "steps_to_full": 40_000.0,
            },
        )

    return cfg


def g1_amp_leap_goto_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Unitree G1 平地跳跃目标导航任务环境配置。"""
    cfg = g1_amp_leap_flat_env_cfg(play=play)
    return _apply_goto_overrides(cfg, play)


def g1_amp_leap_goto_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Unitree G1 崎岖地形跳跃目标导航任务环境配置。"""
    cfg = g1_amp_leap_rough_env_cfg(play=play)
    return _apply_goto_overrides(cfg, play)
