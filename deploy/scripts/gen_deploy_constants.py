"""G1 实机部署常量一键生成脚本（开发机专用）。

在部署架构中的角色：
-------------------
本脚本用于从仿真真理来源 (``src.assets.robots.unitree_g1.g1_constants``) 提取
G1 机器人的各项物理与控制常量，并将其生成为纯字面量形式的 Python 代码数组。
生成的代码用于直接粘贴更新到 ``deploy/common/g1_deploy_constants.py`` 中，
使得实机 Jetson 边缘计算平台无需安装或导入庞大的 mjlab/Warp 依赖即可运行。

工作流程：
---------
1. 在开发机 (x86_64) 环境下运行本脚本：
       uv run python deploy/scripts/gen_deploy_constants.py
2. 脚本自动提取并计算：
   - 29 个关节的标称微屈膝站立角度 (DEFAULT_POS)；
   - 各关节执行器刚度增益 (KP) 与阻尼增益 (KD)；
   - 策略动作缩放比例系数 (ACTION_SCALE = 0.25 * effort_limit / stiffness)；
3. 将终端打印的代码块复制并覆盖到 ``deploy/common/g1_deploy_constants.py``；
4. 运行单元测试进行 1e-6 精度漂移验证：
       uv run python deploy/tests/test_deploy_constants.py
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 策略训练与实机控制所依赖的标准 29 自由度关节顺序（左腿6、右腿6、腰部3、左臂7、右臂7）
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def main():
    """从 G1 仿真资产配置中解析关节参数并输出格式化 Python 代码。"""
    from src.assets.robots.unitree_g1.g1_constants import (
        G1_ACTION_SCALE,
        G1_ARTICULATION,
        KNEES_BENT_KEYFRAME,
    )

    names = POLICY_JOINT_NAMES
    n = len(names)

    # 1. 提取微屈膝关键帧 (KNEES_BENT_KEYFRAME) 的关节标称角度
    default_pos = np.zeros(n)
    for i, name in enumerate(names):
        for pattern, value in KNEES_BENT_KEYFRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                default_pos[i] = value
                break

    # 2. 从执行器配置 (G1_ARTICULATION.actuators) 中提取各关节的刚度 (KP) 与阻尼 (KD)
    kp = np.full(n, np.nan)
    kd = np.full(n, np.nan)
    for act in G1_ARTICULATION.actuators:
        for pattern in act.target_names_expr:
            for i, name in enumerate(names):
                if re.fullmatch(pattern, name):
                    kp[i] = act.stiffness
                    kd[i] = act.damping

    # 3. 提取各关节的动作缩放比例 (G1_ACTION_SCALE)
    action_scale = np.full(n, np.nan)
    for i, name in enumerate(names):
        for pattern, scale in G1_ACTION_SCALE.items():
            if re.match(pattern, name):
                action_scale[i] = scale
                break

    # 完整性校验：确保 29 个关节每一个都有明确定义的参数
    assert not np.isnan(kp).any() and not np.isnan(kd).any(), "存在未匹配到刚度/阻尼的关节！"
    assert not np.isnan(action_scale).any(), "存在未匹配到动作缩放系数的关节！"

    def fmt(arr, name):
        """格式化数组为带缩进的 Python 代码文本。"""
        body = ",\n    ".join(repr(float(x)) for x in arr)
        return f"{name} = np.array([\n    {body},\n], dtype=np.float32)\n"

    print("# --- 请将以下内容复制并粘贴到 deploy/common/g1_deploy_constants.py 中 ---")
    print(fmt(default_pos, "DEFAULT_POS"))
    print(fmt(kp, "KP"))
    print(fmt(kd, "KD"))
    print(fmt(action_scale, "ACTION_SCALE"))


if __name__ == "__main__":
    main()
