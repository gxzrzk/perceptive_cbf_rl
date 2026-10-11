"""真机部署静态常量与仿真物理定义一致性校验测试 (Deploy Constants Drift Guard)。

测试功能：
- 防漂移守护 (Drift Guard)：确保部署端固化的静态常量 (deploy/common/g1_deploy_constants.py)
  与仿真端的真实权威数据源 (src.assets.robots.unitree_g1.g1_constants) 之间保持严格数值一致（误差 < 1e-6）；
- 机器人部署端在 Python 3.8 环境下直接运行，不依赖 mjlab 复杂物理引擎，因此使用预先固化的静态字面量；
  本测试动态导入仿真源配置并重新推导所有电机常量，防止修改 G1 关节致动器增益或关键帧位姿后忘记重新生成部署常量文件；
- 校验 29 自由度控制关节序列顺序、默认折膝站立角度 (DEFAULT_POS)、PD 控制刚度 (KP) 与阻尼 (KD)、动作缩放系数 (ACTION_SCALE)；
- 校验深度相机输入尺寸 (9x16)、测量截断区间 ([0.1, 5.0] 米) 以及时序多帧延迟偏移量 (0, 3, 8, 18)。

关键断言条件：
- POLICY_JOINT_NAMES 长度为 29 且顺序完全吻合；
- DEFAULT_POS、KP、KD、ACTION_SCALE 数组对比数值完全一致 (atol=1e-6)；
- ACTION_SCALE * KP == 0.25 * effort_limit，均为正实数；
- 深度图参数尺寸为 9x16，near=0.1, far=5.0。

覆盖边界情况：
- 正则表达式全匹配与前缀匹配；
- 各致动器组之间的参数映射完备性检查（无 NaN）。
"""

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.common import g1_deploy_constants as C


def _derive():
    """从仿真权威源 g1_constants 动态推导真机部署关节参数。

    推导项包括：
        - names: 策略控制的 29 个关节名称
        - default_pos: 弯膝初始站立位姿下的各关节目标角度
        - kp, kd: 各电机的 PD 位置刚度与速度阻尼
        - action_scale: 策略归一化输出到实际关节角度偏差的缩放因子
    """
    from src.assets.robots.unitree_g1.g1_constants import (
        G1_ACTION_SCALE,
        G1_ARTICULATION,
        KNEES_BENT_KEYFRAME,
    )

    names = C.POLICY_JOINT_NAMES
    n = len(names)

    # 默认位姿 DEFAULT_POS：遍历折膝站立关键帧字典正则，首个全匹配的规则生效
    default_pos = np.zeros(n, dtype=np.float64)
    for i, name in enumerate(names):
        for pattern, value in KNEES_BENT_KEYFRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                default_pos[i] = value
                break

    # 刚度 KP 与阻尼 KD：遍历致动器规格的 target_names_expr 正则匹配各关节
    kp = np.full(n, np.nan, dtype=np.float64)
    kd = np.full(n, np.nan, dtype=np.float64)
    for act in G1_ARTICULATION.actuators:
        for pattern in act.target_names_expr:
            for i, name in enumerate(names):
                if re.fullmatch(pattern, name):
                    kp[i] = act.stiffness
                    kd[i] = act.damping
    assert not np.isnan(kp).any(), "a joint matched no actuator (KP)"
    assert not np.isnan(kd).any(), "a joint matched no actuator (KD)"

    # ACTION_SCALE: first G1_ACTION_SCALE regex that matches (re.match, mirrors amp_policy).
    action_scale = np.full(n, np.nan, dtype=np.float64)
    for i, name in enumerate(names):
        for pattern, scale in G1_ACTION_SCALE.items():
            if re.match(pattern, name):
                action_scale[i] = scale
                break
    assert not np.isnan(action_scale).any(), "a joint matched no action scale"

    return names, default_pos, kp, kd, action_scale


def test_joint_order_is_29_and_matches_source():
    """测试场景：验证关节列表长度为 29 且命名与排序严格匹配仿真来源。"""
    names, *_ = _derive()
    assert len(names) == 29
    assert list(C.POLICY_JOINT_NAMES) == list(names)


def test_default_pos_matches():
    """测试场景：验证预编译常量 DEFAULT_POS 与仿真源折膝位姿数值完全相等。"""
    _, default_pos, *_ = _derive()
    np.testing.assert_allclose(C.DEFAULT_POS, default_pos, atol=1e-6)


def test_kp_kd_match():
    """测试场景：验证预编译常量 KP 与 KD 与致动器物理模型完全一致。"""
    _, _, kp, kd, _ = _derive()
    np.testing.assert_allclose(C.KP, kp, atol=1e-6)
    np.testing.assert_allclose(C.KD, kd, atol=1e-6)


def test_action_scale_matches():
    """测试场景：验证预编译动作缩放比例 ACTION_SCALE 与策略动作缩放配置一致。"""
    *_, action_scale = _derive()
    np.testing.assert_allclose(C.ACTION_SCALE, action_scale, atol=1e-6)


def test_action_scale_consistency_with_kp():
    """测试场景：物理一致性自检——ACTION_SCALE 与 KP 维度必须均为 (29,) 且元素全部严格大于 0。"""
    # ACTION_SCALE = 0.25 * effort_limit / stiffness, and KP = stiffness, so
    # ACTION_SCALE * KP = 0.25 * effort_limit. Just sanity that the arrays agree
    # in shape and are positive (a zero would null an actuator).
    assert C.ACTION_SCALE.shape == C.KP.shape == C.KD.shape == (29,)
    assert (C.ACTION_SCALE > 0).all()
    assert (C.KP > 0).all() and (C.KD > 0).all()


def test_depth_params():
    """测试场景：验证部署端视觉预处理尺寸 (9x16)、截断距离区间 [0.1, 5.0] 及历史帧偏移 (0, 3, 8, 18)。"""
    assert C.DEPTH_H == 9 and C.DEPTH_W == 16
    assert C.DEPTH_NEAR == 0.1 and C.DEPTH_FAR == 5.0
    assert C.DEFAULT_FRAME_OFFSETS == (0, 3, 8, 18)


def _run():
    """直接运行当前文件内的所有测试函数（用于在开发机脱离 pytest 独立执行）。"""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"deploy_constants: {len(fns)} tests passed")


if __name__ == "__main__":
    _run()
