"""部署端基于视线垂直高度与距离的掩码尺寸过滤测试 (Mask Too Tall Tests)。

测试功能：
- 验证掩码几何尺寸过滤函数 ``mask_too_tall``；
- 验证基于绝对画面高度比例的初级过滤：过滤站立人员、门框等大垂直跨度连通域（防止将实验人员误识别为来袭球）；
- 验证结合真实物理距离 (dist) 的自适应视网膜成像几何尺寸过滤：
  在远距离（如 2.46m）处，直径 <= 0.45m 的小球在像平面上所占垂直高度绝不会超过画面高度的 ~18%；
  若某连通域占比达 30%，虽未达到绝对上限 0.5，但结合测距明显超出球体物理极限，必须果断剔除；
  相反，如果该 30% 高度连通域处于近距离 (0.6m)，则符合近大远小的物理透视规律，予以保留；
- 验证无距离信息 (dist=None / NaN / 0) 时的安全回退降级（仅应用绝对阈值）。

关键断言条件：
- 紧凑小球 (高度比 0.11)：not mask_too_tall()；
- 人体高度连通域 (高度比 0.90)：mask_too_tall() 为 True；
- 2.46m 远处的 30% 高度假目标：绝对规则保留，但距离感知规则剔除；
- 0.6m 近处的 30% 高度真目标：予以保留；
- 空掩码或测距无效时：正常回退不崩溃。

覆盖边界情况：
- 全空掩码；
- 自定义 max_height_frac 截断分位数；
- 极端/异常测距输入 (None, NaN, 0.0)。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.tools.etam_track_live import mask_too_tall


def test_compact_ball_kept():
    """测试场景：形态紧凑的小球掩码（高度约 40 像素，占画幅 11%）应正常保留。"""
    m = np.zeros((376, 672), dtype=bool)
    m[170:210, 320:360] = True            # 垂直高度约 40 像素的小球 -> 占画面总高度 11% (紧凑球体正常放行: ~40 px tall ball -> 0.11 of height)
    assert not mask_too_tall(m)


def test_person_height_rejected():
    """测试场景：人体高细连通域（高度约 340 像素，占画幅 90%）被识别为过高并剔除。"""
    m = np.zeros((376, 672), dtype=bool)
    m[20:360, 300:340] = True             # 垂直高度约 340 像素的人体连通域 -> 占画面总高度 90% (非球大目标剔除: ~340 px tall -> 0.90 of height)
    assert mask_too_tall(m)


def test_empty_mask_not_too_tall():
    """测试场景：空掩码（未检出物体）返回 False，不误判为过高。"""
    assert not mask_too_tall(np.zeros((376, 672), dtype=bool))


def test_threshold_is_fraction_of_height():
    """测试场景：自定义高度比例阈值对过滤判定的动态影响。

    输入假数据：
        高度占比为 60% 的掩码。
    预期输出及验证逻辑：
        当阈值上限设为 0.5 时判定为过高 (True)；当阈值上限提升至 0.7 时判定为未超限 (False)。
    """
    m = np.zeros((100, 100), dtype=bool)
    m[0:60, 0:10] = True                  # 构造占画幅高度 60% 的垂直条状掩码 (60% tall)
    assert mask_too_tall(m, max_height_frac=0.5)
    assert not mask_too_tall(m, max_height_frac=0.7)


def _mask(frac, h=376, w=672):
    """辅助函数：在 (h, w) 分辨率画幅中央构造指定高度占比 frac 的连通域掩码。"""
    m = np.zeros((h, w), dtype=bool)
    n = int(frac * h)
    m[h // 2 : h // 2 + n, w // 2 - 5 : w // 2 + 5] = True
    return m


def test_far_blob_too_tall_for_distance_rejected():
    """测试场景：远距离物体结合成像几何测距判定——排除远处的过大物体。

    测试原理：
        画面占比 30% 虽然低于默认的 0.5 绝对上限，但若物体测距为 2.46m，
        标准球体（直径 <= 0.45m）在透视成像下的理论最大像素占比仅为 ~18%；
        故该物体必然是大型背景或站立的人，距离感知规则必须将其标记为过高 (mask_too_tall=True)。
    """
    # 结合真实物理测距与成像几何的自适应尺寸检验机制：
    # 占画幅高度 30% 的连通域低于 0.5 绝对比例门槛（旧版固定阈值会误放行），但在 2.46 米测距下，
    # 真实小球（直径 <= 0.45 米）在透视成像下的理论最大像素占比仅为 ~18% -> 距离感知规则将其精准剔除。
    # (原英文推导: A 30%-tall blob is UNDER the absolute 0.5 cap, but at 2.46 m a real ball (<=0.45 m diam))
    # (原英文结论: can only span ~18% of the frame -> distance-aware reject.)
    m = _mask(0.30)
    assert not mask_too_tall(m)                 # 旧版绝对比例规则：低于 0.5 予以保留 (old absolute rule: kept)
    assert mask_too_tall(m, dist=2.46)          # 距离自适应规则：超出该距离下的球体几何极限，判定过高剔除 (distance-aware: rejected)


def test_real_ball_far_kept():
    """测试场景：远距离处的真实尺寸小球被正常保留。

    测试原理：
        在 2.46m 距离处，直径 0.22m 的真实小球高度占比约为 9%，
        同时符合绝对上限与距离自适应几何上限，予以保留 (False)。
    """
    # 真实小球场景：在 2.46 米距离处，直径约 0.22 米的真实球体仅占画幅高度约 9% -> 同时满足绝对阈值与距离自适应几何阈值，予以保留
    # (原英文注释: A real ~0.22 m ball at 2.46 m subtends ~9% of the frame -> kept under both caps.)
    m = _mask(0.09)
    assert not mask_too_tall(m, dist=2.46)


def test_large_blob_close_kept():
    """测试场景：近距离处大占比掩码的正常保留。

    测试原理：
        相同的 30% 画面占比物体，若距离只有 0.6m，其物理投影尺寸与近处小球完全吻合，予以保留。
    """
    # 近距离透视放大效应：同样的 30% 画幅高度连通域，若出现在近处 (0.6m)，与近距离真实球体几何尺度完全吻合 -> 正常保留放行
    # (原英文注释: The same 30%-tall blob up close (0.6 m) is consistent with a ball -> kept.)
    assert not mask_too_tall(_mask(0.30), dist=0.6)


def test_no_distance_falls_back_to_absolute_cap():
    """测试场景：测距缺失或非实数时，平滑降级回退至绝对比例阈值。

    输入假数据：
        占比 30% 的连通域，分别传入 dist=None、dist=NaN、dist=0.0。
    预期输出及验证逻辑：
        均因小于 0.5 绝对门槛而被保留 (False)，不发生除零或崩溃。
    """
    # 测距无效回退机制：当 dist 为 None 或非有限数值 (NaN/Inf) 时，自动回退到仅依赖绝对比例阈值，保证向后兼容性与系统稳健性
    # (原英文注释: dist=None / non-finite -> only the absolute cap applies (back-compat).)
    m = _mask(0.30)
    assert not mask_too_tall(m, dist=None)
    assert not mask_too_tall(m, dist=float("nan"))
    assert not mask_too_tall(m, dist=0.0)
