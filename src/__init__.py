"""Perceptive CBF-RL 源码根包初始化模块 (Source root package initialization module)。

该模块定义了项目源码目录的全局绝对路径常量 `SRC_PATH`，作为整个项目定位模型资产文件
(如 MuJoCo MJCF XML、网格模型)、动作捕捉先验数据 (motion dataset) 及配置文件的基准路径。
"""

from pathlib import Path

# 项目源码根目录路径（指向 /home/.../src 目录）
# 用于通过相对层级在各子模块中解析 assets/xmls 等物理资源路径
SRC_PATH: Path = Path(__file__).parent