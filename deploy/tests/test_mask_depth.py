"""部署端深度图掩码有效性与池化降采样测试 (Mask Depth to Far Tests)。

测试功能：
- 验证基于二值掩码 (binary mask) 的深度图前景提取与背景剔除函数 ``mask_depth_to_far``；
- 验证掩码有效性：仅保留掩码为 True 的前景像素（如视觉检测到的小球区域）真实深度，
  其余非掩码像素（杂乱背景、地面或墙面）强制填充为远端距离 (far = 5.0m)；
- 验证空掩码（未检测到物体或跟踪丢失）时的安全回退：整张图全变为 5.0m，避免策略误触避碰；
- 验证形状不匹配时的异常防御机制；
- 验证掩码过滤后与分位数下采样池化 (percentile_pool_depth) 的端到端流水线：
  成功保留小球深度，彻底滤除前景周围极近的背景噪点。

关键断言条件：
- mask 为 True 的位置深度精确保留，mask 为 False 的位置强制为 5.0；
- 原输入数组未被就地修改 (immutable)；
- 形状不一致抛出 ValueError；
- 经过 (9, 16) 池化后，仅球体所在网格保留近处深度 1.0m，其余网格全为 5.0m。

覆盖边界情况：
- 全空掩码 (all False)；
- 各种尺寸数组与维度不匹配；
- 包含极近地面干扰杂讯 (0.3m) 的深度图过滤。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.common.zed_depth_source import mask_depth_to_far, percentile_pool_depth


def test_non_masked_pixels_become_far():
    """测试场景：掩码区域保留原始深度，非掩码背景像素被重置为远端极限值 (far=5.0m)。

    输入假数据：
        (4, 6) 的深度图，初始深度全为 1.5m；中间 (1:3, 2:4) 区域标记为 True。
    预期输出及验证逻辑：
        1. 输出数组中，掩码区域依然等于 1.5m；
        2. 非掩码区域全部被替换为 5.0m；
        3. 输出类型为 float32；
        4. 输入的原始 depth 数组未被原地更改。
    """
    depth = np.full((4, 6), 1.5, dtype=np.float32)  # everything near
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:4] = True                            # a small "ball" region
    out = mask_depth_to_far(depth, mask, far=5.0)
    assert np.all(out[mask] == 1.5)                  # ball depth kept
    assert np.all(out[~mask] == 5.0)                 # everything else forced far
    assert out.dtype == np.float32
    assert depth[0, 0] == 1.5                         # input not mutated


def test_empty_mask_is_all_far():
    """测试场景：视觉跟踪丢失或初始未检出时（空掩码），整幅深度图全置为 5.0m。

    输入假数据：
        随机近距离深度图 [0.3, 1.0] 米，传入全 False 的空掩码。
    预期输出及验证逻辑：
        输出深度图全部等于 5.0m，使神经网络策略判定前方空旷无障碍，防止误触发摔倒或侧扑。
    """
    # Track lost / not started -> empty mask -> everything far (policy sees no near object).
    depth = np.random.uniform(0.3, 1.0, (8, 8)).astype(np.float32)
    out = mask_depth_to_far(depth, np.zeros((8, 8), dtype=bool), far=5.0)
    assert np.all(out == 5.0)


def test_shape_mismatch_raises():
    """测试场景：异常防御——当深度图与掩码尺寸不一致时必须抛出异常。

    输入假数据：
        深度图尺寸 (4, 6)，掩码尺寸 (4, 5)。
    预期输出及验证逻辑：
        函数拦截并抛出 ValueError。
    """
    depth = np.zeros((4, 6), dtype=np.float32)
    try:
        mask_depth_to_far(depth, np.zeros((4, 5), dtype=bool), far=5.0)
        assert False, "expected ValueError on shape mismatch"
    except ValueError:
        pass


def test_masked_then_pooled_keeps_ball_drops_background():
    """测试场景：掩码过滤配合分位数池化 (percentile pooling) 的完整预处理流水线。

    输入假数据：
        (18, 32) 深度图，背景充斥 0.3m 的地面近距噪点，仅左上角 (0:2, 0:2) 处存在 1.0m 的真实球体并赋予掩码。
    预期输出及验证逻辑：
        1. 先执行 mask_depth_to_far 将 0.3m 的背景置为 5.0m；
        2. 再执行 percentile_pool_depth 降采样到 (9, 16)；
        3. 左上角网格正确保留球体 1.0m 读数，其余区域全部为 5.0m，验证背景噪点被彻底过滤。
    """
    # Ball pixels near (1.0 m) in one corner; rest is junk-near (0.3 m) that masking removes.
    depth = np.full((18, 32), 0.3, dtype=np.float32)
    mask = np.zeros((18, 32), dtype=bool)
    mask[0:2, 0:2] = True
    depth[0:2, 0:2] = 1.0
    masked = mask_depth_to_far(depth, mask, far=5.0)
    pooled = percentile_pool_depth(masked, 9, 16, percentile=3.0, fill_invalid=5.0)  # [9,16]
    assert pooled.shape == (9, 16)
    assert pooled[0, 0] == 1.0           # the ball cell reads near
    assert np.all(pooled[1:, :] == 5.0)  # everywhere the ball isn't reads far (background gone)
