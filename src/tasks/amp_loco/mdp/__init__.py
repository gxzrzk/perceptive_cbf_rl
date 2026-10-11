"""AMP 运动控制与控制屏障函数 (CBF) 强化学习马尔可夫决策过程 (MDP) 模块.

本模块定义了人形机器人 (Unitree G1 等) 在敏捷运动 (Locomotion/Leap)、
躲避高速来球 (Dodgeball) 以及障碍物走廊/教室避障 (Wall/Classroom Obstacle Avoidance) 任务中
所需的全部 MDP 核心组件，包括：
1. 控制屏障函数与安全动作滤波:
   - 2D 质心速度级预测滤波 (`cbf.py`)
   - 全身关节速度级闭式投影 (`cbf_joint.py`, `actions_cbf.py`)
2. 导航目标与指令生成:
   - 纯速度观测的无状态目标跟踪 (`command.py`, `goal_command.py`)
   - 随机折线走廊与纯追踪引导 (`walk_path.py`, `classroom.py`)
3. 观测空间 (Observations):
   - 机器人本体感觉、历史观测拼接、特权球体/障碍物几何状态
   - 头戴深度相机感知 (Ball-only masked depth 16x9、Sim2Real 域随机化)
4. 奖励设计 (Rewards):
   - AMP 判别器对抗模仿奖励、速度跟踪与足底软着陆/防滑奖励
   - Link-CBF 与 Joint-CBF 惩罚/安全缓冲奖励
5. 事件与课程学习 (Events & Curriculums):
   - 基于动作捕捉的参考状态重置 (AMP RSI: `MotionResetManager`)
   - 弹道抛物线投球 (`throw_ball_on_dwell`)、静态障碍固定 (`pin_wall`) 与动态循环 (`recycle_walls_ahead`)
6. 终止条件 (Terminations) 与监控指标 (Metrics):
   - 持续过低高度倒地、击中检测 (瞬时接触+速度不连续穿模检测)、撞墙检测
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

# 控制屏障函数 (CBF) 核心算法与动作滤波
from .cbf import *  # noqa: F403
from .cbf_joint import *  # noqa: F403
from .actions_cbf import (  # noqa: F401
  CbfJointPositionAction,
  CbfJointPositionActionCfg,
)

# 课程学习调度
from .curriculums import *  # noqa: F403

# 重置与域随机化事件 (RSI、抛球、墙体放置等)
from .events import *  # noqa: F403

# 观测空间定义 (本体感觉、特权状态、深度图等)
from .observations import *  # noqa: F403

# 奖励函数计算 (AMP风格模仿、任务跟踪、CBF安全约束奖励)
from .rewards import *  # noqa: F403

# 回合终止条件检测 (倒地、被球击中、撞墙等)
from .terminations import *  # noqa: F403

# 指令管理器项 (基础与目标驱动指令)
from .command import *  # noqa: F403
from .goal_command import *  # noqa: F403

# 随机路径生成与教室布局
from .walk_path import *  # noqa: F403
from .classroom import *  # noqa: F403

# 地形生成与评估指标
from .terrain import *  # noqa: F403
from .metrics import *  # noqa: F403
