"""机器人资产顶层统一导出接口 (Robot Assets Top-level Export Module)。

本模块汇集并对外暴露 Unitree G1 仿人机器人两种主要构型的配置与控制接口：
1. **G1 全自由度构型 (29/30 DOF)**:
   - `get_g1_robot_cfg`: 获取包含精细电驱非线性 T-N 执行器模型及完整全身自由度的配置对象。
   - `G1_ACTION_SCALE`: 全自由度构型中强化学习策略动作 [-1, 1] 映射到关节目标位置增量的缩放字典。
2. **G1 精简自由度构型 (23 DOF)**:
   - `get_g1_23dof_robot_cfg`: 获取锁定部分末端腕部/手部自由度的 23 自由度精简机器人配置对象。
   - `G1_23DOF_ACTION_SCALE`: 23 自由度构型对应的策略动作缩放字典。
"""

from .unitree_g1.g1_constants import (
  G1_ACTION_SCALE as G1_ACTION_SCALE,        # 全自由度 G1 策略动作缩放字典 (action scale dict)
  get_g1_robot_cfg as get_g1_robot_cfg,      # 获取全自由度 G1 机器人实体配置函数
)

from .unitree_g1.g1_23dof_constants import (
  G1_23DOF_ACTION_SCALE as G1_23DOF_ACTION_SCALE,    # 23 自由度 G1 策略动作缩放字典
  get_g1_23dof_robot_cfg as get_g1_23dof_robot_cfg,  # 获取 23 自由度 G1 机器人实体配置函数
)
