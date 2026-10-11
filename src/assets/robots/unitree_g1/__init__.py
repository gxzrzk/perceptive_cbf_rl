"""宇树 Unitree G1 仿人双足机器人模型资产包 (Unitree G1 Humanoid Robot Assets Package)。

本包提供了 Unitree G1 仿人机器人在 MuJoCo / mjlab 仿真环境中的全套模型资产与控制常量：
- `unitree_actuators.py`: Unitree 电驱执行器非线性动力学模型，包含 T-N 扭矩-转速包络截断、
  正反向旋转峰值转矩非对称性、基于 tanh 的静/动摩擦力模型、以及等效位置控制律转换。
- `g1_constants.py`: G1 全自由度 (29 DOF) 物理常量与配置，包含电机两级行星减速器等效电枢转动惯量、
  二阶系统固有频率/阻尼比 PD 增益推导、四连杆并联关节（踝部/腰部）折算、碰撞过滤掩码与动作缩放。
- `g1_23dof_constants.py`: G1 23 自由度精简构型配置，锁定末端手腕自由度以加速强化学习步态策略收敛。
- `g1_constants_bp.py`: 使用 MuJoCo 原生线性位置执行器 (Builtin Position) 的基准对比版本。
- `xmls/`: 机器人统一模型描述文件 (MJCF XML) 与 3D 表面几何网格 (meshes)。
"""
