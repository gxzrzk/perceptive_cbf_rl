"""地形生成与配置模块 (Terrain Configurations).

用于在强化学习训练环境中构建非平整复杂地形，提升腿足机器人的足底适应性、鲁棒性与 Sim2Real 泛化能力。
配置包括平整地面 (Flat) 与随机粗糙高程图地形 (Heightfield Random Uniform Rough) 的网格阵列组合。
"""

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

# 随机粗糙混合地形配置：用于训练机器人在轻微起伏路面上的行走与抗扰能力
RANDOM_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),            # 每个子地形网格的尺寸 (长 8m, 宽 8m)
  border_width=20.0,          # 外围边界平坦保护区宽度 (20m)
  num_rows=10,                # 地形阵列的行数 (10行)
  num_cols=20,                # 地形阵列的列数 (20列，共 200 个子地形)
  sub_terrains={
    # 基础平整地面占比 40% (便于学习基础步态与直线快速奔跑)
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.4),
    # 随机均匀粗糙高程图占比 60% (高程噪声范围 2cm~5cm，提升落足扰动适应性)
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.6,
      noise_range=(0.02, 0.05), # 随机高程起伏范围 [0.02m, 0.05m]
      noise_step=0.02,          # 采样离散步长 2cm
      border_width=0.25,        # 子地形边缘过渡带宽度
    ),
    # 可选波动地形（波浪起伏地貌，备用）
    # "wave_terrain": terrain_gen.HfWaveTerrainCfg(
    #   proportion=0.1,
    #   amplitude_range=(0.0, 0.2),
    #   num_waves=4,
    #   border_width=0.25,
    # ),
  },
  add_lights=True,            # 在地形上方添加渲染光源
)
