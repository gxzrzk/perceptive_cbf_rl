"""真机部署网络通信 UDP 数据包打包、解包与时序序列号同步测试 (UDP Sync Tests)。

测试功能：
- 验证真机底层硬件控制进程、上层深度相机采集进程与策略推理进程之间的 UDP 二进制网络数据契约；
- 机器人状态数据包 (State Datagram) 序列化与反序列化：
  数据包总长度固定为 74 个 float32 (296 字节)，严格对齐以下字段：
  1 step_id + 4 quat + 3 ang_vel + 3 proj_grav + 29 q + 29 dq + 3 cmd + 1 depth_far_hold + 1 mode；
- 机器人控制动作包 (Action Datagram) 序列化与反序列化：
  29 个关节目标角度目标 (29 个 float32 = 116 字节)；
- 深度图图像包 (Depth Datagram) 序列化与序列号 (seq) 同步：
  报头包含 3 个浮点数字段（帧序列号 seq、时间戳 stamp、相机数 n_cam）+ 载荷数据（每个相机 144 个浮点数）；
  支持单目 (n_cam=1) 与双目 (n_cam=2) 传输；
- 校验数据包报文长度、内存对齐与声明相机数量不一致时的异常防御；
- 校验部署所用的硬件端口 (UDP_HW_PORT)、相机端口 (UDP_DEPTH_PORT) 与策略端口 (UDP_POLICY_PORT) 互相独立不冲突。

关键断言条件：
- STATE_FLOATS == 74，STATE_BYTES == 296；
- 打包再解包后的 step_id、quat、ang_vel、proj_grav、q、dq、cmd 等逐元素完全相等；
- 动作包长度严格为 116 字节；
- 深度包报头 seq、stamp、n_cam 及 depth 载荷完全对齐；
- 3 个通信端口两两互斥。

覆盖边界情况：
- depth_far_hold 布尔开关（True/False）及其默认状态；
- 控制模式 mode (0 为躲避，1 为其他工况)；
- 相机数量 n_cam 声明与实际图像数组长度不一致时的异常拦截；
- 单相机 (147 floats) 与双相机 (291 floats) 不同的数据包规格。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.common import udp_sync


def test_state_sizes_match_layout():
    """测试场景：校验机器人状态包的固定浮点数个数与字节总大小。

    数据结构：
        1 步进ID + 4 姿态四元数 + 3 基座角速度 + 3 投影重力 + 29 关节角度 + 29 关节角速度
        + 3 导航速度指令 + 1 远端保持开关 + 1 运行模式 = 74 个 float32 = 296 字节。
    """
    # 1 step_id + 4 quat + 3 ang_vel + 3 proj_grav + 29 q + 29 dq + 3 cmd
    # + 1 depth_far_hold + 1 mode = 74 floats.
    assert udp_sync.STATE_FLOATS == 74
    assert udp_sync.STATE_BYTES == 74 * 4


def test_state_round_trip():
    """测试场景：机器人状态数据包打包与解包往返一致性校验。

    输入假数据：
        构造包含随机浮点数的四元数、角速度、重力、29关节位姿、29关节速度及速度指令，
        分别测试 depth_far_hold 为 False 与 True。
    预期输出及验证逻辑：
        1. 二进制数据流长度严格等于 STATE_BYTES (296)；
        2. 解包出的 step_id、四元数、角速度、重力、关节位姿与速度、指令及 hold 状态与原值严格相等。
    """
    rng = np.random.default_rng(0)
    step_id = 12345
    quat = rng.standard_normal(4).astype(np.float32)
    ang_vel = rng.standard_normal(3).astype(np.float32)
    proj_grav = rng.standard_normal(3).astype(np.float32)
    q = rng.standard_normal(29).astype(np.float32)
    dq = rng.standard_normal(29).astype(np.float32)
    cmd = rng.standard_normal(3).astype(np.float32)

    for hold in (False, True):
        buf = udp_sync.pack_state(step_id, quat, ang_vel, proj_grav, q, dq, cmd,
                                  depth_far_hold=hold)
        assert len(buf) == udp_sync.STATE_BYTES

        o_step, o_quat, o_ang, o_grav, o_q, o_dq, o_cmd, o_hold, o_mode = udp_sync.unpack_state(buf)
        assert o_step == step_id
        np.testing.assert_array_equal(o_quat, quat)
        np.testing.assert_array_equal(o_ang, ang_vel)
        np.testing.assert_array_equal(o_grav, proj_grav)
        np.testing.assert_array_equal(o_q, q)
        np.testing.assert_array_equal(o_dq, dq)
        np.testing.assert_array_equal(o_cmd, cmd)
        assert o_hold is hold
        assert o_mode == 0

    # Default (omitted) -> hold off, mode dodge(0).
    *_, o_hold, o_mode = udp_sync.unpack_state(
        udp_sync.pack_state(step_id, quat, ang_vel, proj_grav, q, dq, cmd))
    assert o_hold is False
    assert o_mode == 0


def test_state_mode_round_trip():
    """测试场景：状态包中运行控制模式字段 (mode) 的打包与解包。

    输入假数据：
        mode 分别为 0（躲避模式）与 1（其他模式）。
    预期输出及验证逻辑：
        解包后 o_mode 精确恢复传入的整数模式编号。
    """
    rng = np.random.default_rng(9)
    args = (3, rng.standard_normal(4).astype(np.float32),
            rng.standard_normal(3).astype(np.float32),
            rng.standard_normal(3).astype(np.float32),
            rng.standard_normal(29).astype(np.float32),
            rng.standard_normal(29).astype(np.float32),
            rng.standard_normal(3).astype(np.float32))
    for mode in (0, 1):
        buf = udp_sync.pack_state(*args, depth_far_hold=False, mode=mode)
        assert len(buf) == udp_sync.STATE_BYTES
        *_, o_hold, o_mode = udp_sync.unpack_state(buf)
        assert o_mode == mode


def test_action_round_trip():
    """测试场景：策略动作包打包与解包往返一致性校验。

    输入假数据：
        29 维关节目标角度随机向量 q_target。
    预期输出及验证逻辑：
        打包长度严格为 29 * 4 = 116 字节，解包出的动作向量与原数组完全相等。
    """
    rng = np.random.default_rng(1)
    q_target = rng.standard_normal(29).astype(np.float32)
    buf = udp_sync.pack_action(q_target)
    assert len(buf) == udp_sync.ACTION_BYTES == 29 * 4
    np.testing.assert_array_equal(udp_sync.unpack_action(buf), q_target)


def test_depth_round_trip_one_cam():
    """测试场景：单目深度图数据包 (n_cam=1) 序列号同步与往返解包。

    输入假数据：
        帧序列号 seq=7，时间戳 stamp=1.5，相机数 1，144 维单目深度数据。
    预期输出及验证逻辑：
        1. 解包后的 seq 精确恢复为 7，支持网络丢包与乱序校验；
        2. 时间戳 stamp 误差 < 1e-3；
        3. 深度数据数组完全一致；
        4. 数据包二进制长度 = (3 个报头浮点数 + 144) * 4 = 588 字节。
    """
    rng = np.random.default_rng(2)
    seq = 7
    stamp = 1.5  # informational; float32-representable
    depth = rng.standard_normal(1 * udp_sync.DEPTH_FRAME_FLOATS).astype(np.float32)
    buf = udp_sync.pack_depth(seq, stamp, 1, depth)
    o_seq, o_stamp, o_ncam, o_depth = udp_sync.unpack_depth(buf)
    assert o_seq == seq
    assert o_ncam == 1
    np.testing.assert_allclose(o_stamp, stamp, rtol=0, atol=1e-3)
    np.testing.assert_array_equal(o_depth, depth)
    # n_cam=1 -> 3 header floats + 144 depth = 147 floats.
    assert len(buf) == (3 + udp_sync.DEPTH_FRAME_FLOATS) * 4


def test_depth_round_trip_two_cam():
    """测试场景：双目深度图数据包 (n_cam=2) 的序列化与反序列化。

    输入假数据：
        帧序列号 seq=42，时间戳 stamp=2.0，相机数 2，288 维双目拼接深度数据。
    预期输出及验证逻辑：
        解包出的 seq=42、n_cam=2，数据载荷数组严格一致，数据包长度为 (3 + 288) * 4 = 1164 字节。
    """
    rng = np.random.default_rng(3)
    depth = rng.standard_normal(2 * udp_sync.DEPTH_FRAME_FLOATS).astype(np.float32)
    buf = udp_sync.pack_depth(42, 2.0, 2, depth)
    o_seq, _o_stamp, o_ncam, o_depth = udp_sync.unpack_depth(buf)
    assert o_seq == 42
    assert o_ncam == 2
    np.testing.assert_array_equal(o_depth, depth)
    assert len(buf) == (3 + 2 * udp_sync.DEPTH_FRAME_FLOATS) * 4


def test_depth_n_cam_mismatch_raises():
    """测试场景：异常防御——声明的相机数量与实际深度数据长度冲突时抛出异常。

    输入假数据：
        仅有 144 个浮点数（单相机），但在 pack_depth 中声称 n_cam=2。
    预期输出及验证逻辑：
        抛出 ValueError 异常，防止下位机或策略解析出越界破坏数据。
    """
    # Guard the camera-count contract: declared n_cam must match the payload length.
    bad = np.zeros(udp_sync.DEPTH_FRAME_FLOATS, dtype=np.float32)  # one cam's worth
    try:
        udp_sync.pack_depth(0, 0.0, 2, bad)  # claim two cams
    except ValueError:
        return
    raise AssertionError("pack_depth must reject an n_cam/payload-length mismatch")


def test_distinct_ports():
    """测试场景：硬件、相机与策略三方通信使用的 UDP 端口必须两两互斥。"""
    ports = {udp_sync.UDP_HW_PORT, udp_sync.UDP_DEPTH_PORT, udp_sync.UDP_POLICY_PORT}
    assert len(ports) == 3, "the three UDP ports must be distinct"


def _run():
    """直接执行测试（开发机无需 pytest 支持）。"""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"udp_sync: {len(fns)} tests passed")


if __name__ == "__main__":
    _run()
