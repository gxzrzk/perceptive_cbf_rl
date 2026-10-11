"""环境物体与障碍物资产导出模块 (Environmental Objects and Obstacles Module)。

本模块统一对外暴露仿真环境中交互物体的配置接口与核心几何常量：
1. 躲避球实体 (ball): 动态抛掷物体的半径常量 `DEFAULT_BALL_RADIUS` 与配置生成函数 `get_ball_cfg`。
2. 静态/固化障碍墙体 (wall): 空间约束障碍物的几何半尺寸常量与配置生成函数 `get_wall_cfg`。
"""

from .ball import (
  DEFAULT_BALL_RADIUS as DEFAULT_BALL_RADIUS,  # 默认躲避球半径 (m)
  get_ball_cfg as get_ball_cfg,                # 获取动态躲避球实体配置函数
)
from .wall import (
  WALL_HALF_HEIGHT as WALL_HALF_HEIGHT,        # 障碍墙体半高度 (m)
  WALL_HALF_LENGTH as WALL_HALF_LENGTH,        # 障碍墙体半长度 (m)
  WALL_HALF_THICKNESS as WALL_HALF_THICKNESS,  # 障碍墙体半厚度 (m)
  get_wall_cfg as get_wall_cfg,                # 获取静态障碍墙体实体配置函数
)
