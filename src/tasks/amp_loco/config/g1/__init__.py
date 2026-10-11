"""宇树 Unitree G1 人形机器人 AMP 任务注册入口模块。

本模块集中注册了基于 Unitree G1 人形机器人的全套强化学习任务，包含：
1. 基础运动任务（Base AMP Locomotion）：纯本体感觉的平地双足行走与速度跟踪；
2. 躲避球基准任务系列（Dodgeball Benchmarks / Perception Regimes）：
   - 状态真值环境（State Oracle）：Actor/Critic 均观测小球的位置与速度真值；
   - 单墙体躲避环境（State Oracle + Wall）：在躲避小球的同时，利用控制屏障函数（CBF）与感知避免碰撞静态墙面；
   - 走廊连续巡航避障（State Oracle + WallWalk）：沿连续弯曲路径巡航行走，同时回避两侧障碍墙壁与迎面飞来的飞球；
   - 复杂教室环境（State Oracle + Classroom）：在真实教室课桌椅障碍物布局中进行走廊穿梭导航与全方位（360度）躲避飞球；
   - 视觉感知环境（Fixed / Gimbal Camera）：使用机载单目深度相机（支持选配头部 1 自由度俯仰云台），
     配合球体掩膜深度图（Ball-Only Masked Depth），直接从视觉观测中端到端学习躲避行为。
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import g1_amp_flat_env_cfg
from .dodge_env_cfgs import (
  g1_amp_dodge_depth_single_ballonly_flat_env_cfg,
  g1_amp_dodge_mimickit_flat_env_cfg,
  g1_amp_dodge_mimickit_wall_flat_env_cfg,
  g1_amp_dodge_mimickit_wallwalk_flat_env_cfg,
  g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)
from .goto_rl_cfg import (
  g1_amp_dodge_depth_single_ppo_runner_cfg,
  g1_amp_dodge_mimickit_ppo_runner_cfg,
  g1_amp_dodge_mimickit_wallwalk_ppo_runner_cfg,
  g1_amp_dodge_mimickit_classroom_ppo_runner_cfg,
)
from .rl_cfg import g1_amp_ppo_runner_cfg

# ---------------------------------------------------------------------------
# 1. 基础 AMP 步态控制任务 (Base AMP Locomotion)
# 训练仅依赖本体感觉（Proprioception-only）的平地行走策略。
# 该基础策略被广泛用于真机部署与评测中的恢复步态（--walk-recover），
# 以及真机在日常行走与紧急躲避模式（walk <-> dodge）之间的动态模式切换。
# ---------------------------------------------------------------------------
register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat",
  env_cfg=g1_amp_flat_env_cfg(),
  play_env_cfg=g1_amp_flat_env_cfg(play=True),
  rl_cfg=g1_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# ---------------------------------------------------------------------------
# 2. 躲避球任务族（Dodgeball Tasks - 论文中的各种感知与避障场景）
# ---------------------------------------------------------------------------

# 【状态真值基线 - STATE ORACLE】
# Actor 直接观测小球在机体航向坐标系下的真实 3D 位置与 3D 速度，不依赖深度相机。
# 对应论文中的“状态先知”（State oracle）感知机制（Actor 与 Critic 采用对称观测）。
register_mjlab_task(
  task_id="Unitree-G1-AMP-Dodge-MimicKit-Flat",
  env_cfg=g1_amp_dodge_mimickit_flat_env_cfg(),
  play_env_cfg=g1_amp_dodge_mimickit_flat_env_cfg(play=True),
  rl_cfg=g1_amp_dodge_mimickit_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# 【状态真值 + 静态单墙避碰 - STATE ORACLE + WALL】
# 感知机制同上，但在机器人出生点旁随机生成一面静态墙壁（每轮随机化所在侧与相对距离）。
# 机器人必须在躲避飞球的同时避免撞墙：
# - 墙体状态加入到对称的 ball_state 观测组中；
# - 为各连杆引入静态障碍物安全屏障奖励（CBF barrier reward）塑造防撞安全裕度；
# - 一旦与墙面发生碰撞，立即触发终止并处以 -200 的重度跌倒/碰撞惩罚。
register_mjlab_task(
  task_id="Unitree-G1-AMP-Dodge-MimicKit-Wall-Flat",
  env_cfg=g1_amp_dodge_mimickit_wall_flat_env_cfg(),
  play_env_cfg=g1_amp_dodge_mimickit_wall_flat_env_cfg(play=True),
  rl_cfg=g1_amp_dodge_mimickit_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# 【状态真值 + 走廊连续巡航避障 - STATE ORACLE + WALLS WHILE WALKING A RANDOM PATH】
# 机器人在每轮幕中需跟随随机生成的柔和弯曲路径巡航前进（纯追踪算法，巡航速度约 1.3 m/s），
# 穿过由多面静态墙壁（默认 3 面）构成的无尽走廊（走过的墙壁会被循环回收并重新放置在前方）。
# 机器人必须在前进巡航、绕过障碍物以及躲避飞球之间达成动态平衡：
# - 移除静止站立奖励，引入沿路径前进奖励（walk_path_progress）；
# - AMP 判别器与 RSI（参考状态初始化）均采用融合了行走动捕数据的混合运动集（amp_dodge_walk），
#   确保判别器认可高速巡航步态为正样本。
register_mjlab_task(
  task_id="Unitree-G1-AMP-Dodge-MimicKit-WallWalk-Flat",
  env_cfg=g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(),
  play_env_cfg=g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(play=True),
  rl_cfg=g1_amp_dodge_mimickit_wallwalk_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# 【状态真值 + 真实教室环境穿梭 - STATE ORACLE + CLASSROOM】
# 场景中布置了固定几何尺寸与间距的真实课桌椅阵列，机器人随机在安全走廊区域出生，
# 沿纵向或横向走道生成的路径穿梭导航，同时应对 360 度全向随机发射的飞球：
# - 课桌椅参与碰撞终止判定以及各机体连杆的 CBF 安全屏障计算；
# - Actor 与 Critic 均观测距离机器人最近的 6 个障碍物的几何状态与相对位置；
# - AMP 判别器与初始化同样使用行走增强运动集。
register_mjlab_task(
  task_id="Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat",
  env_cfg=g1_amp_dodge_mimickit_classroom_flat_env_cfg(),
  play_env_cfg=g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=True),
  rl_cfg=g1_amp_dodge_mimickit_classroom_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

# 【视觉感知基准：固定相机 / 俯仰云台相机 - FIXED / GIMBAL CAMERA】
# 头部搭载单目深度相机，采用仅针对小球分割的掩膜深度图（BallOnlyDepthObs）：
# 小球区域显示真实测距，背景全部置为远距离截断值。
# 该表征与真机上使用 EfficientTAM 分割网络提取小球后的表征完全一致，大幅缩小了 Sim2Real 差距。
# - 默认配置为固定视角深度相机；
# - 当指定环境变量 CAMERA_GIMBAL=1 CAMERA_PROPRIO=1 时，自动激活头部 1-DoF 俯仰伺服云台，
#   由先知视线自动瞄准小球，大幅扩展视场（FOV）跟踪范围。
register_mjlab_task(
  task_id="Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat",
  env_cfg=g1_amp_dodge_depth_single_ballonly_flat_env_cfg(),
  play_env_cfg=g1_amp_dodge_depth_single_ballonly_flat_env_cfg(play=True),
  rl_cfg=g1_amp_dodge_depth_single_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
