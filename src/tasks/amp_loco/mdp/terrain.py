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
    # 可选连续起伏波动地形（正弦波浪地貌，可作为进阶泛化测试备用）：
    # "wave_terrain": terrain_gen.HfWaveTerrainCfg(  # 正弦波动起伏地形配置项
    #   proportion=0.1,             # 地形占比：占全部子地形网格的 10%
    #   amplitude_range=(0.0, 0.2), # 波峰波谷振幅范围：起伏高度在 0.0m 至 0.2m 之间随机采样
    #   num_waves=4,                # 波动周期数：每个 8x8m 子网格内包含 4 个连续起伏的正弦波峰
    #   border_width=0.25,          # 边缘平滑过渡区宽度：0.25m，避免子地形拼接处发生几何高度跳变
    # ),                            # 结束 wave_terrain 起伏配置
  },
  add_lights=True,            # 在地形上方添加渲染光源
)
