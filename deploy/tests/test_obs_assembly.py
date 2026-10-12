"""部署端策略观测向量装配契约测试 (Obs-Assembly Contract Tests)。

测试功能：
- 验证真机部署端与仿真端完全等价的策略输入观测向量拼接协议；
- 策略输入总结构：actor input = actor_proprio (384维) ++ depth (144 * 4 = 576维) = 960 维；
- 验证本体感知历史缓存 (ProprioHistory)：
  采用“项优先” (TERM-MAJOR) 时序排列：先排同一物理量的所有历史帧（由旧到新），
  再排下一物理量；物理项顺序依次为：基座角速度 (3)、投影重力 (3)、速度指令 (3)、
  关节相对位置 (29)、关节相对速度 (29)、上一动作 (29)；
  在 4 帧历史下总维度为 (3+3+3+29+29+29) * 4 = 384 维；
- 验证深度图归一化 (depth_metres_to_obs)：将物理距离 [0.1m, 5.0m] 线性映射归一化至 [0.0, 1.0]，
  盲区或越界近景安全映射至远端 (1.0)；
- 验证深度图环形缓冲区 (DepthRing)：
  支持按照 (0, 3, 8, 18) 等指定帧偏移量重采样，采用“由新到旧” (NEWEST->OLDEST) 顺序堆叠；
  在初始帧数不足时自动以最旧帧安全回退填充，重置后隔绝前一剧集历史；
- 验证视觉数据新鲜度门控 (DepthGate)：当相机帧率掉帧或超时卡顿时，安全保持最后一帧且标记陈旧，
  首帧未到达前返回全远端占位，避免人形机器人因瞬态异常发生姿态失控或摔倒。

关键断言条件：
- 归一化后深度：0.05m -> 1.0 (盲区), 0.1m -> 0.0, 5.0m -> 1.0, 2.55m -> 0.5；
- 本体感知输出形状严格为 (384,)，排列严格满足 term-major 顺序；
- 拼接后完整观测矩阵形状为 (1, 960)，前 384 维为本体，后 576 维为深度图；
- 深度门控在超时后返回 is_stale=True 且不阻塞动作输出。

覆盖边界情况：
- 深度图极小盲区与超远距离截断；
- 环形缓冲区初启单帧、半填充及跨剧集清空；
- 相机首帧未接收前的空包回退。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.policy import dodge_policy as P


# ---------------------------------------------------------------------------
# 深度米制距离 -> 归一化观测值映射 [0, 1]
# ---------------------------------------------------------------------------
def test_depth_metres_to_obs():
    """测试场景：米制深度图到 [0, 1] 归一化空间的无量纲化映射。

    输入假数据：
        [0.05 (近于盲区 0.1m), 0.1 (最近有效距离), 5.0 (远端截断距离), 2.55 (正中距离)]。
    预期输出及验证逻辑：
        1. 盲区距离 (0.05m) 映射为远端 (1.0)；
        2. 0.1m 映射为 0.0；
        3. 5.0m 映射为 1.0；
        4. 中间值 2.55m 映射为 (2.55 - 0.1) / (5.0 - 0.1) = 0.5。
    """
    near, far = 0.1, 5.0
    frame = np.array([0.05, 0.1, 5.0, 2.55], dtype=np.float32)  # 构造典型测试点：盲区近距、最近边界、最远边界、正中距离 (below-near, near, far, mid)
    out = P.depth_metres_to_obs(frame, near, far)
    # 归一化映射计算验证：盲区 (0.05m) 映射为远端 1.0；0.1m 映射为 0.0；5.0m 映射为 1.0；2.55m 映射为 (2.55-0.1)/(5.0-0.1) = 0.5
    # (原英文公式: below-near -> far -> 1.0; near -> 0.0; far -> 1.0; mid 2.55 -> (2.55-0.1)/4.9 = 0.5)
    np.testing.assert_allclose(out, [1.0, 0.0, 1.0, 0.5], atol=1e-6)


def test_depth_metres_to_obs_clamps_above_far():
    """测试场景：远距离超出 far 上限的像素被平滑截断至 1.0。

    输入假数据：
        1000.0 米的超远距离。
    预期输出及验证逻辑：
        输出值截断在 1.0。
    """
    near, far = 0.1, 5.0
    out = P.depth_metres_to_obs(np.array([1000.0], dtype=np.float32), near, far)
    np.testing.assert_allclose(out, [1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# 本体感知 TERM-MAJOR 历史时序排列测试
# ---------------------------------------------------------------------------
def test_proprio_dim_is_384_at_history_4():
    """测试场景：验证 4 帧本体感知历史向量的总维度为 384 维。

    输入假数据：
        全零的角速度 (3)、投影重力 (3)、指令 (3)、关节位置 (29)、关节速度 (29)、上一动作 (29)。
    预期输出及验证逻辑：
        单帧 96 维 * 4 帧历史 = 384 维。
    """
    h = P.ProprioHistory(history_length=4)
    h.append(
        base_ang_vel=np.zeros(3), proj_grav=np.zeros(3), command=np.zeros(3),
        joint_pos_rel=np.zeros(29), joint_vel_rel=np.zeros(29), last_action=np.zeros(29),
    )
    assert h.vector().shape == (384,)


def test_proprio_term_major_layout():
    """测试场景：严格校验 term-major（项优先：各物理项内部按由旧到新展开）的时序内存布局。

    输入假数据：
        设置 history_length=2，第 0 帧与第 1 帧赋予特征分明的常数哨兵值：
        角速度 (10->11), 重力 (20->21), 指令 (30->31), 关节位置 (40->41), 速度 (50->51), 动作 (60->61)。
    预期输出及验证逻辑：
        向量各切片必须严格按照 [旧帧, 新帧] 连续排布，与仿真环境 mjlab 的默认格式完全对齐。
    """
    # 历史长度设定为 2 (history_length=2)，为各物理项指定独特的哨兵常数值以严格校验内存时序布局
    # (原英文注释: history_length=2, distinct per-term sentinel values so the layout is checkable.)
    h = P.ProprioHistory(history_length=2)
    # 推入第 0 帧历史观测数据（较旧历史帧: frame 0, oldest）
    h.append(
        base_ang_vel=np.full(3, 10.0), proj_grav=np.full(3, 20.0), command=np.full(3, 30.0),
        joint_pos_rel=np.full(29, 40.0), joint_vel_rel=np.full(29, 50.0), last_action=np.full(29, 60.0),
    )
    # 推入第 1 帧历史观测数据（当前最新帧: frame 1, newest）
    h.append(
        base_ang_vel=np.full(3, 11.0), proj_grav=np.full(3, 21.0), command=np.full(3, 31.0),
        joint_pos_rel=np.full(29, 41.0), joint_vel_rel=np.full(29, 51.0), last_action=np.full(29, 61.0),
    )
    v = h.vector()
    # 项主序排布契约 (term-major): 各物理项内部按由旧到新 [oldest(frame0) .. newest(frame1)] 连续拼接，严格对应训练端 actor_terms 顺序
    # (原英文注释: term-major: each term is [oldest(frame0) .. newest(frame1)] concatenated, in term order.)
    expected = np.concatenate([
        np.full(3, 10.0), np.full(3, 11.0),    # 机身角速度 base_ang_vel: 最旧帧 (oldest), 最新帧 (newest)
        np.full(3, 20.0), np.full(3, 21.0),    # 投影重力向量 projected_gravity: 最旧帧, 最新帧
        np.full(3, 30.0), np.full(3, 31.0),    # 速度指令 command: 最旧帧, 最新帧
        np.full(29, 40.0), np.full(29, 41.0),  # 29 关节相对角度 joint_pos: 最旧帧, 最新帧
        np.full(29, 50.0), np.full(29, 51.0),  # 29 关节角速度 joint_vel: 最旧帧, 最新帧
        np.full(29, 60.0), np.full(29, 61.0),  # 上拍控制动作 actions: 最旧帧, 最新帧
    ])
    np.testing.assert_array_equal(v, expected)


def test_proprio_reset_zeros_history():
    """测试场景：本体感知历史重置清零。

    预期输出及验证逻辑：
        调用 reset() 后，返回的 384 维向量全部恢复为 0。
    """
    h = P.ProprioHistory(history_length=4)
    h.append(
        base_ang_vel=np.full(3, 1.0), proj_grav=np.full(3, 1.0), command=np.full(3, 1.0),
        joint_pos_rel=np.full(29, 1.0), joint_vel_rel=np.full(29, 1.0), last_action=np.full(29, 1.0),
    )
    h.reset()
    np.testing.assert_array_equal(h.vector(), np.zeros(384))


# ---------------------------------------------------------------------------
# 深度图环形缓冲区：帧偏移堆叠由新到旧 (newest -> oldest)
# ---------------------------------------------------------------------------
def test_depth_ring_first_frame_fills_all_offsets():
    """测试场景：深度图缓冲区刚接收首帧时，所有历史延迟分头均填充该首帧数据。

    输入假数据：
        指定偏移量 (0, 1, 3)，推入单帧 144 维常数数据 (0.7)。
    预期输出及验证逻辑：
        输出维度为 3 * 144 = 432，且 3 个槽位全部等于该首帧数据。
    """
    ring = P.DepthRing(frame_offsets=(0, 1, 3))
    f0 = np.full(144, 0.7, dtype=np.float32)
    out = ring.push(f0)
    # 单帧初启安全回退机制：当仅收到首帧图像时，所有时间偏移槽位均安全填充该首帧数据 (only one frame seen -> every offset reads it)
    assert out.shape == (3 * 144,)
    np.testing.assert_array_equal(out, np.tile(f0, 3))


def test_depth_ring_newest_first_ordering():
    """测试场景：多帧推入后，深度图切片按照最新帧优先（newest first）排列。

    输入假数据：
        连续推入序号为 0 到 5 的 6 帧图像，指定偏移量为 (0, 1, 3)。
    预期输出及验证逻辑：
        当前最新帧为 5，偏移 0 对应帧 5、偏移 1 对应帧 4、偏移 3 对应帧 2；
        输出拼接向量依次为 [5.0, 4.0, 2.0] 各自平铺 144 维。
    """
    ring = P.DepthRing(frame_offsets=(0, 1, 3))
    frames = [np.full(144, float(i), dtype=np.float32) for i in range(6)]
    out = None
    for f in frames:
        out = ring.push(f)
    # 由新到旧 (newest->oldest) 时序切片验证：连续推入 0..5 帧后最新帧为 5，偏移量 (0, 1, 3) 分别对应提取第 5、4、2 帧
    # (原英文注释: after pushing 0..5, newest=5. offsets (0,1,3) -> frames 5, 4, 2.)
    expected = np.concatenate([np.full(144, 5.0), np.full(144, 4.0), np.full(144, 2.0)])
    np.testing.assert_array_equal(out, expected)


def test_depth_ring_partial_history_clamps_to_oldest():
    """测试场景：当历史帧数不足以满足深层延迟偏移量时，自动截断回退到最旧的可用帧。

    输入假数据：
        定义偏移量为 (0, 3, 8, 18)，先后仅推入两帧 a (1.0) 和 b (2.0)。
    预期输出及验证逻辑：
        最新偏移 0 读取 b (2.0)，更深层的偏移 3、8、18 均安全回退读取最早的帧 a (1.0)。
    """
    ring = P.DepthRing(frame_offsets=(0, 3, 8, 18))
    a = np.full(144, 1.0, dtype=np.float32)
    b = np.full(144, 2.0, dtype=np.float32)
    ring.push(a)
    out = ring.push(b)  # 仅推入两帧数据；深层偏移量 3/8/18 尚未积累足够步数 (only two frames pushed; offset 3/8/18 not available yet)
    # 最新槽位 (offset 0) 为最新帧 b；深层槽位自动安全回退填充最早帧 a，防止越界或未初始化脏数据 (newest offset 0 = b; deeper offsets fall back to oldest fill a)
    expected = np.concatenate([np.full(144, 2.0), np.full(144, 1.0), np.full(144, 1.0), np.full(144, 1.0)])
    np.testing.assert_array_equal(out, expected)


def test_depth_ring_reset_reinits_to_next_frame():
    """测试场景：调用 reset() 后彻底隔离历史帧污染。

    输入假数据与测试流程：
        推入旧数据后执行 reset()，再推入新帧 (9.0)。
    预期输出及验证逻辑：
        所有偏移量槽位全被新帧 9.0 填充，无旧剧集历史残留。
    """
    ring = P.DepthRing(frame_offsets=(0, 1))
    ring.push(np.full(144, 1.0, dtype=np.float32))
    ring.push(np.full(144, 2.0, dtype=np.float32))
    ring.reset()
    out = ring.push(np.full(144, 9.0, dtype=np.float32))
    # 复位重置隔离性验证：reset 后推入的新帧填满所有历史偏移量槽位，彻底消除旧回合历史的污染渗透 (post-reset new frame fills every offset; no bleed from old episode)
    np.testing.assert_array_equal(out, np.tile(np.full(144, 9.0), 2))


# ---------------------------------------------------------------------------
# 完整装配测试：本体感知 ++ 深度时序图，维度必须严格等于 ONNX 输入
# ---------------------------------------------------------------------------
def test_assemble_obs_concatenates_proprio_then_depth():
    """测试场景：验证本体感知 (384) 与深度观测 (576) 的总装配，输出形状为 (1, 960)。

    预期输出及验证逻辑：
        前 384 维完全等于 proprio，后 576 维完全等于 depth。
    """
    proprio = np.arange(384, dtype=np.float32)
    depth = np.arange(384, 384 + 576, dtype=np.float32)  # 模拟堆叠深度图特征: 4 个历史时刻偏移 * 144 维/帧 = 576 维 (4 offsets * 144)
    obs = P.assemble_obs(proprio, depth)
    assert obs.shape == (1, 960)
    np.testing.assert_array_equal(obs[0, :384], proprio)
    np.testing.assert_array_equal(obs[0, 384:], depth)


# ---------------------------------------------------------------------------
# 深度数据新鲜度门控 (Depth Freshness Gate)
# ---------------------------------------------------------------------------
def test_freshness_gate_holds_last_frame_when_stale():
    """测试场景：相机数据传输超时后，门控报警但依然保持最后一帧可用图像，防止策略无数据崩溃。

    输入假数据与测试流程：
        在 t=0.0 时传入图像帧，超时阈值设为 0.1s。
    预期输出及验证逻辑：
        1. 在 t=0.05s 时，is_stale 为 False（新鲜）；
        2. 在 t=0.25s 时，is_stale 为 True（陈旧报警）；
        3. gate.frame() 依然完整返回上一有效帧数据，维持控制回路持续运转。
    """
    gate = P.DepthGate(n_cam=1)
    f = np.full(144, 0.3, dtype=np.float32)
    gate.update(f, t=0.0)
    # 到达即时验证：刚接收到新数据报时，时间间隔在超时阈值内，判定为新鲜 (fresh just after arrival)
    assert not gate.is_stale(now=0.05, stale_s=0.1)
    # 超时保持机制：超过设定窗口后判定为陈旧 (stale)；但仍稳定返回锁存的上一有效帧，保证控制主循环不阻塞崩溃 (held frame still returned, no stall)
    assert gate.is_stale(now=0.25, stale_s=0.1)
    np.testing.assert_array_equal(gate.frame(), f)


def test_freshness_gate_stale_before_first_frame():
    """测试场景：尚未收到首帧图像时的初始待机状态。

    预期输出及验证逻辑：
        is_stale 为 True，且 frame() 默认返回全 1.0（远端空旷）深度图，
        保证机器人在视觉模块启动就绪前保持平稳站立，不会误触发剧烈躲避动作。
    """
    gate = P.DepthGate(n_cam=1)
    # 首帧未达保底机制：尚未收到任何 UDP 数据报时判定为陈旧，且 frame() 安全回退返回全 1.0 全远平面空旷图，
    # 确保机器人在视觉节点启动阶段平稳站立，避免误动作失控摔倒 (no datagram yet -> stale; returns far empty image so robot stands)
    assert gate.is_stale(now=1.0, stale_s=0.1)
    np.testing.assert_array_equal(gate.frame(), np.ones(144, dtype=np.float32))


def _run():
    """直接执行测试（开发机无需 pytest 支持）。"""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"obs_assembly: {len(fns)} tests passed")


if __name__ == "__main__":
    _run()
