"""基于对抗运动先验（Adversarial Motion Priors, AMP）的人形机器人运动控制任务包。

本任务包结合了深度强化学习（PPO）与对抗运动先验（AMP）算法，为人造人形机器人（如 Unitree G1）
提供基于真实动捕数据（Mocap Dataset）的高质量运动步态生成与高级运动控制任务支持。

核心架构定位：
1. 基础运动学习（Base AMP Locomotion）：通过判别器促使策略生成类人、自然且鲁棒的速度跟踪步态；
2. 敏捷跳跃与奔跑（Leap / Sprint）：支持方向性跳跃与敏捷回避；
3. 动态避障与避球（Dodgeball / Perception Regimes）：结合感知模态（状态真值 / 单目深度相机 / 云台相机）
   与控制障碍函数（Control Barrier Functions, CBF），实现兼顾自然运动与安全避障/躲避球的策略。

初始化说明：
在构建任何声明了 ``history_ordering="time"`` 的任务配置之前，在此处预先导入并安装
时间优先（time-major）观测历史重排补丁（见 ``observation_history_ordering.py``）。
"""

# 在构建任何任务配置前，安装时间优先的观测历史排序重载补丁。
# 该操作替代了旧版本需要手动覆盖 mjlab 安装包文件的补丁方式，改为在运行时动态 Monkey-Patch。
from src.tasks.amp_loco import observation_history_ordering as observation_history_ordering  # noqa: F401
