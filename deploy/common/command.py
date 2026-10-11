"""躲避策略的名义速度指令与无线遥控手柄摇杆映射模块。

在部署架构中的角色：
-------------------
躲避策略网络 (Actor) 观测包含一个 3 维速度指令向量 ``[vx, vy, yaw_rate]``（对应训练时的 twist 观测项）。
- 在躲避模式 (Dodge Mode) 下，默认指令为 ``[0, 0, 0]``（原地站立躲避：保持平衡并对飞来的球体做出反应）。
  操作人员可通过手柄轻推机器人调整站位（例如两次抛球之间微调位置）。
- 在行走模式 (Walk Mode) 下，该指令直接驱动双足移动策略的行走速度。

本文件为纯函数实现（不依赖任何 SDK），方便在任意节点或测试中被导入。
"""

import numpy as np

# 指令向量索引定义
CMD_VX = 0         # 前向线速度 (vx, 单位: m/s)
CMD_VY = 1         # 横向线速度 (vy, 单位: m/s，左正右负)
CMD_YAW_RATE = 2   # 航向角速度 (yaw_rate, 单位: rad/s，逆时针为正)
CMD_SIZE = 3       # 速度指令向量维度

# 站立躲避的默认名义指令：零速度 [0.0, 0.0, 0.0]
NOMINAL_COMMAND = np.zeros(CMD_SIZE, dtype=np.float32)

# 摇杆推满时映射到速度指令的增益比例 (vx: m/s, vy: m/s, yaw_rate: rad/s)。
# 其数值与 AMP-Flat 行走策略训练时的速度范围匹配 (amp_env_cfg.py: vx ∈ [-1.5, 3.0], vy ∈ [-1.0, 1.0], yaw ∈ [-pi/2, pi/2])，
# 确保遥控器推满时仍处于策略分布内。
#
# 注意事项（躲避模式共享此增益）：
# 躲避策略在训练时的角速度范围 max_ang_vel_z=1.0，且名义偏航指令接近 0，因此 YAW_SCALE=1.57 主要用于行走模式。
# 在躲避模式下建议尽量让偏航轴保持中立，主要依赖视觉感知进行躲避。
VX_SCALE = 1.5     # 前向速度缩放系数 (最大 1.5 m/s)
VY_SCALE = 1.0     # 横向速度缩放系数 (最大 1.0 m/s)
YAW_SCALE = 1.57   # 偏航角速度缩放系数 (最大 pi/2 ≈ 1.57 rad/s)
DEADZONE = 0.1     # 摇杆死区阈值（低于此绝对值的微小漂移将被置零）


def make_command() -> np.ndarray:
    """创建并返回名义零速度指令数组的全新副本。"""
    return NOMINAL_COMMAND.copy()


def _dz(v: float) -> float:
    """死区过滤辅助函数：若输入幅值小于 DEADZONE，则归零；否则返回原值。"""
    return 0.0 if abs(v) < DEADZONE else float(v)


def command_from_sticks(left_stick, right_stick) -> np.ndarray:
    """将宇树无线手柄的双摇杆模拟量映射为速度指令向量 ``[vx, vy, yaw_rate]`` 并应用死区过滤。

    参数:
        left_stick: 左摇杆输入 [x, y]，控制机身平移速度 (vy, vx)。
        right_stick: 右摇杆输入 [x, y]，通过 x 轴控制偏航旋转速度 yaw_rate。

    返回:
        np.ndarray: 包含 [vx, vy, yaw_rate] 的 float32 数组，维度为 (3,)。

    说明:
        在 Unitree G1 遥控手柄硬件上，双摇杆的 x 轴物理读数呈反向，因此代码中对 x 轴取负号：
        - 左摇杆 x 轴取负 -> 向左推输出正的 vy（向左平移）。
        - 右摇杆 x 轴取负 -> 向左推输出正的 yaw_rate（逆时针偏航）。
        左摇杆 y 轴向上推输出正的 vx（向前平移）。
    """
    lx, ly = -float(left_stick[0]), float(left_stick[1])  # x 轴取反（修正硬件摇杆反相）
    rx = -float(right_stick[0])
    cmd = np.zeros(CMD_SIZE, dtype=np.float32)
    cmd[CMD_VX] = _dz(ly) * VX_SCALE
    cmd[CMD_VY] = _dz(lx) * VY_SCALE
    cmd[CMD_YAW_RATE] = _dz(rx) * YAW_SCALE
    return cmd
