"""单目深度躲避与行走双模态策略推理节点（纯 Python / ONNXRuntime CPU 推理，无 ROS 依赖）。

在部署架构中的角色：
-------------------
本节点是整个部署系统的大脑，掌控 ``Unitree-G1-AMP-Dodge-Depth-Single-Flat`` 的观测输入契约。
以 50 Hz 循环频率运行，核心逻辑包括：
1. 双端口非阻塞清空接收 (Drain)：
   - 从 UDP_POLICY_PORT (9871) 获取来自 hardware_node（或 sim_node）的机器人状态；
   - 从 UDP_DEPTH_PORT (9872) 获取来自 camera_node（或 camera_node_etam）的米制深度图；
2. 观测组装 (Observation Assembly):
   - 严格按照 MuJoCo 仿真训练时的排布规则，组装本体感受 4 帧项主序历史与深度图时间堆叠帧；
3. ONNX CPU 推理与模式切换：
   - 运行 ONNX 模型得到 [-1, 1] 的无量纲动作；
   - 支持在躲避 (Dodge) 与纯双足行走 (Walk) 双策略之间平滑线性插值切换；
4. 动作反归一化与下发：
   - 将动作反归一化为 29 个关节物理目标弧度，通过 UDP_HW_PORT (9870) 发送给硬件控制节点。

观测数据契约规范 (Observation Contract)：
----------------------------------------
Actor 输入向量 = actor_proprio (384 维) ++ depth (144 * n_cam * len(offsets) 维)

* 本体感受项 (proprio, 384 维)：
  - 采用 **项主序 (TERM-MAJOR)** 排布，每项内部按 **由旧到新 (oldest -> newest)** 排布 4 帧历史；
  - 项排列顺序必须严格匹配 ``src/tasks/amp_loco/amp_env_cfg.py`` 中 actor_terms 的定义顺序：
    1. base_ang_vel: 3 维 * 4 帧 = 12
    2. projected_gravity: 3 维 * 4 帧 = 12
    3. command: 3 维 * 4 帧 = 12
    4. joint_pos (相对默认位置的偏移): 29 维 * 4 帧 = 116
    5. joint_vel: 29 维 * 4 帧 = 116
    6. actions (上一步动作): 29 维 * 4 帧 = 116
    总计: 12 + 12 + 12 + 116 + 116 + 116 = 384 维。

* 深度图观测项 (depth):
  - 单帧尺寸为 9 行 x 16 列 = 144 维，经 ``DepthImageObs`` 规则归一化到 [0, 1] 区间（0=最近, 1=最远）；
  - 沿时间轴按 ``frame_offsets`` **由新到旧 (NEWEST -> OLDEST)** 堆叠！即 offset 0 居首，后续为历史步数。
  - 单相机默认 offsets=(0, 3, 8, 18)，对应深度输入维度为 144 * 1 * 4 = 576 维。
  - 总输入维度为 384 + 576 = 960 维。

安全与容错机制：
---------------
- 维度强校验：节点启动时比对拼装观测维度与 ONNX 模型输入的输入维度，不匹配立即报错退出，绝不盲目运行导致摔倒；
- 硬件复位重置：监测到 ``step_id`` 反向跳变时，立即重置本体感受历史和动作缓存；
- 深度新鲜度守门 (DepthGate): 若深度图接收间隔超时（> depth_stale_s），自动保持最后一帧并报警，机器人停留在当前平稳状态；
- 双模式平滑插值切换 (Mode Switch Blend): 在躲避与行走切换时，对关节目标角度进行 25 步（约 0.5 秒）线性插值过渡，消除跳变冲击。
"""

import argparse
import ast
import os
import time
from collections import deque

# 抑制 ONNXRuntime 冗余日志输出
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.common.g1_deploy_constants import (
    ACTION_SCALE,
    DEFAULT_FRAME_OFFSETS,
    DEFAULT_POS,
    DEPTH_FAR,
    DEPTH_NEAR,
    NUM_JOINTS,
)
from deploy.common.udp_sync import (
    DEPTH_FRAME_FLOATS,
    UDP_DEPTH_PORT,
    UDP_HOST,
    UDP_HW_PORT,
    UDP_POLICY_PORT,
    STATE_BYTES,
    create_udp_socket,
    pack_action,
    unpack_depth,
    unpack_state,
)

# 本体感受历史帧数与总维度常量
HISTORY_LENGTH = 4
PROPRIO_DIM = 384

# 单帧本体感受各项排布顺序及其各自维度。
# 必须与 amp_env_cfg.py 中 actor_terms 的插入顺序及 mjlab 默认项主序展平完全一致！
_TERM_ORDER = (
    "base_ang_vel", "projected_gravity", "command",
    "joint_pos", "joint_vel", "actions",
)
_TERM_DIMS = {
    "base_ang_vel": 3, "projected_gravity": 3, "command": 3,
    "joint_pos": NUM_JOINTS, "joint_vel": NUM_JOINTS, "actions": NUM_JOINTS,
}
# 断言总维度必须等于 384
assert sum(_TERM_DIMS[k] for k in _TERM_ORDER) * HISTORY_LENGTH == PROPRIO_DIM


# ---------------------------------------------------------------------------
# 深度米制数据 -> 归一化策略输入 [0, 1]
# ---------------------------------------------------------------------------
def depth_metres_to_obs(frame_m: np.ndarray, near: float, far: float) -> np.ndarray:
    """将池化后的米制深度数据转换为训练策略期望的 [0, 1] 归一化输入向量。

    复现仿真中 DepthImageObs._one 的核心转换逻辑：
    1. 未击中/超出有效测量范围 (raw < near) 映射为远距离 far；
    2. 距离夹紧到 [near, far] 区间；
    3. 线性归一化: (depth - near) / (far - near)，输出 float32 数组。
    """
    x = np.asarray(frame_m, dtype=np.float32)
    x = np.where(x < near, far, x)
    x = np.clip(x, near, far)
    return ((x - near) / (far - near)).astype(np.float32)


# ---------------------------------------------------------------------------
# 本体感受项主序历史管理器 (ProprioHistory)
# ---------------------------------------------------------------------------
class ProprioHistory:
    """管理 4 帧本体感受历史并按照项主序 (Term-Major) 展平为 384 维向量。

    排布逻辑：
    按 _TERM_ORDER 中的顺序依次排列各物理项；
    每个物理项内部保存一个长度为 HISTORY_LENGTH 的双端队列，元素按时间由旧到新 (oldest -> newest) 排列。
    """

    def __init__(self, history_length: int = HISTORY_LENGTH):
        self._L = int(history_length)
        self._hist = {
            k: deque([np.zeros(_TERM_DIMS[k], np.float32)] * self._L, maxlen=self._L)
            for k in _TERM_ORDER
        }

    def reset(self) -> None:
        """重置所有历史状态为零向量（在检测到硬件重启或复位时调用）。"""
        for k in _TERM_ORDER:
            self._hist[k].clear()
            for _ in range(self._L):
                self._hist[k].append(np.zeros(_TERM_DIMS[k], np.float32))

    def append(self, base_ang_vel, proj_grav, command,
               joint_pos_rel, joint_vel_rel, last_action) -> None:
        """将当前控制步最新的本体感受数据推入各自的队列末端。"""
        self._hist["base_ang_vel"].append(np.asarray(base_ang_vel, np.float32))
        self._hist["projected_gravity"].append(np.asarray(proj_grav, np.float32))
        self._hist["command"].append(np.asarray(command, np.float32))
        self._hist["joint_pos"].append(np.asarray(joint_pos_rel, np.float32))
        self._hist["joint_vel"].append(np.asarray(joint_vel_rel, np.float32))
        self._hist["actions"].append(np.asarray(last_action, np.float32))

    def vector(self) -> np.ndarray:
        """按项主序拼接所有历史帧，输出展平后的 384 维 float32 数组。"""
        parts = [np.concatenate(list(self._hist[k])) for k in _TERM_ORDER]
        return np.concatenate(parts, dtype=np.float32)


# ---------------------------------------------------------------------------
# 深度时间环形缓冲区 (DepthRing): 按 frame_offsets 由新到旧堆叠
# ---------------------------------------------------------------------------
class DepthRing:
    """深度图时间环形缓冲区。
    每次调用 ``push(frame)`` 时将最新帧存入环形队列，并返回按 ``frame_offsets``
    从最新到最旧 (**NEWEST -> OLDEST**, offset 0 居首) 堆叠抽取的展平向量。
    首帧到达时会填满整个缓冲区；复位时重新初始化，避免跨硬件复位发生旧数据泄露。
    """

    def __init__(self, frame_offsets):
        self._offsets = tuple(sorted(int(o) for o in frame_offsets))
        self._L = max(self._offsets) + 1  # 缓冲区长度取决于最大历史回溯步数 + 1
        self._buf = None                  # 内部环形数组 [L, frame_dim]
        self._head = 0                    # 当前写入头部游标
        self._pending_reset = False       # 待复位标志

    def reset(self) -> None:
        """标记复位，下一帧写入时将重新填满整个环形缓冲区。"""
        self._pending_reset = True

    def push(self, frame: np.ndarray) -> np.ndarray:
        """推入最新归一化深度帧，并按指定的帧偏移量提取堆叠深度特征。"""
        cur = np.asarray(frame, dtype=np.float32)
        if self._buf is None or self._pending_reset:
            # 初始化或复位：使用当前帧填满整个环形队列
            self._buf = np.broadcast_to(cur, (self._L, cur.size)).copy()
            self._head = 0
            self._pending_reset = False
        else:
            self._head = (self._head + 1) % self._L
            self._buf[self._head] = cur
        # 按 frame_offsets 从新到旧提取帧索引: head - k
        idx = [(self._head - k) % self._L for k in self._offsets]
        return self._buf[idx].reshape(-1).copy()


# ---------------------------------------------------------------------------
# 深度时效守门员 (DepthGate): 基于到达时间的保活与兜底检测
# ---------------------------------------------------------------------------
class DepthGate:
    """维护最新接收到的归一化深度帧。
    若在启动阶段尚未收到任何相机报文，``frame()`` 默认返回全 1.0 (全远平面/空旷空间) 数组，
    使机器人保持安全站立而不至于崩溃。
    通过本地数据报到达时间戳评估新鲜度 (is_stale)。
    """

    def __init__(self, n_cam: int = 1):
        self._dim = int(n_cam) * DEPTH_FRAME_FLOATS
        self._frame = np.ones(self._dim, dtype=np.float32)  # 默认全远平面 (1.0 代表空旷)
        self._last_arrival = None

    def update(self, frame: np.ndarray, t: float) -> None:
        """接收新深度数据并刷新到达时间戳。"""
        f = np.asarray(frame, dtype=np.float32).reshape(-1)
        if f.size != self._dim:
            raise ValueError(f"深度帧大小 {f.size} != 期望维度 {self._dim}")
        self._frame = f
        self._last_arrival = float(t)

    def frame(self) -> np.ndarray:
        """获取当前缓存的深度图。"""
        return self._frame

    def is_stale(self, now: float, stale_s: float) -> bool:
        """检查深度图是否超时陈旧（超过 stale_s 秒未收到新包）。"""
        return self._last_arrival is None or (now - self._last_arrival) > stale_s


# ---------------------------------------------------------------------------
# 全观测组装函数 (Observation Assembly)
# ---------------------------------------------------------------------------
def assemble_obs(proprio: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """拼接本体感受 (384维) 与时间堆叠深度特征，输出形状为 [1, D] 的 ONNX 推理张量。"""
    obs = np.concatenate(
        [np.asarray(proprio, np.float32), np.asarray(depth, np.float32)]
    ).astype(np.float32)
    return obs.reshape(1, -1)


# ---------------------------------------------------------------------------
# 主循环与控制流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="单目深度躲避策略推理节点 (Unitree G1)")
    parser.add_argument("onnx_path", help="导出的躲避策略 ONNX 模型路径 (.onnx)")
    parser.add_argument(
        "--frame-offsets", default=str(DEFAULT_FRAME_OFFSETS),
        help=f"深度图历史堆叠步数偏移元组，必须与模型导出时严格一致 (默认 {DEFAULT_FRAME_OFFSETS})",
    )
    parser.add_argument("--n-cam", type=int, default=1, help="相机数量 (单目头部相机设为 1)")
    parser.add_argument("--depth-stale-s", type=float, default=0.3,
                        help="深度帧超时阈值(秒)。若超过此时长未收到新帧，将保持末帧并打印警告，机器人保持站立")
    parser.add_argument("--hz", type=float, default=50.0, help="策略推理与控制主循环频率 (Hz)")
    parser.add_argument("--walk-onnx", default=None,
                        help="纯本体感受行走策略 ONNX 模型路径 (384->29)。"
                             "若指定，则启用双模态切换 (通过手柄 Y 键或状态包中的 mode=1 激活)。"
                             "与躲避策略共享相同的本体感受历史与动作缩放，无需切换 PD 增益。")
    parser.add_argument("--blend-ticks", type=int, default=25,
                        help="在躲避模式与行走模式切换时的线性目标过渡步数 (ticks; 50 Hz 下 25 步约 0.5 秒)，"
                             "避免关节目标位置突变产生机械冲击。")
    args = parser.parse_args()

    import onnxruntime as ort

    offsets = tuple(ast.literal_eval(args.frame_offsets))
    n_cam = int(args.n_cam)

    # 初始化躲避策略 ONNX 推理会话 (CPU 执行)
    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    expected_dim = int(session.get_inputs()[0].shape[-1])
    depth_dim = n_cam * DEPTH_FRAME_FLOATS * len(offsets)
    assembled_dim = PROPRIO_DIM + depth_dim
    # 严格校验观测输入维度是否匹配
    if assembled_dim != expected_dim:
        raise ValueError(
            f"组装的观测维度 {assembled_dim} (本体感受 {PROPRIO_DIM} + 深度 {depth_dim} = "
            f"{n_cam}相机*144*{len(offsets)}偏移) != ONNX 期望输入维度 {expected_dim}。"
            f"是否指定了错误的 --frame-offsets 或 --n-cam？"
        )
    print(f"[dodge_policy] 观测维度校验通过: {assembled_dim} 维 (offsets={offsets}, n_cam={n_cam})")

    # 可选行走策略加载 (纯本体感受 384 -> 29)
    walk_session = walk_inp = None
    if args.walk_onnx:
        walk_session = ort.InferenceSession(args.walk_onnx, providers=["CPUExecutionProvider"])
        walk_inp = walk_session.get_inputs()[0].name
        walk_dim = int(walk_session.get_inputs()[0].shape[-1])
        if walk_dim != PROPRIO_DIM:
            raise ValueError(f"行走策略 ONNX 输入维度 {walk_dim} != 本体感受维度 {PROPRIO_DIM}")
        print(f"[dodge_policy] 行走模式已启用: {args.walk_onnx} ({PROPRIO_DIM}->{NUM_JOINTS})")

    # 初始化状态机内部结构与缓存
    proprio = ProprioHistory(HISTORY_LENGTH)
    ring = DepthRing(offsets)
    gate = DepthGate(n_cam)
    far_frame = np.ones(n_cam * DEPTH_FRAME_FLOATS, dtype=np.float32)  # 全远平面深度
    prev_depth_far_hold = False
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    prev_mode = 0
    last_target = DEFAULT_POS.copy()   # 上一步发送的实际关节目标（用作模式切换插值的起始点）
    blend_from = DEFAULT_POS.copy()
    blend_i = args.blend_ticks         # 初始设为完成状态（无需插值）

    # 创建非阻塞 UDP 套接字
    state_sock = create_udp_socket(UDP_HOST, UDP_POLICY_PORT)
    depth_sock = create_udp_socket(UDP_HOST, UDP_DEPTH_PORT)
    state_sock.setblocking(False)
    depth_sock.setblocking(False)
    hw_addr = (UDP_HOST, UDP_HW_PORT)
    print(f"[dodge_policy] UDP端口监听: 状态<-{UDP_POLICY_PORT}, 深度<-{UDP_DEPTH_PORT}, 动作下发->{UDP_HW_PORT}")

    dt = 1.0 / float(args.hz)
    next_tick = time.perf_counter()
    prev_step_id = -1
    last_stale_warn = 0.0

    try:
        while True:
            # 严格控制 50 Hz 主循环周期
            now = time.perf_counter()
            sleep_t = next_tick - now
            if sleep_t > 0:
                time.sleep(sleep_t)
            next_tick += dt

            # 非阻塞清空深度套接字缓冲区，仅保留最新深度数据报；执行归一化并更新到守门员
            depth_data = _drain(depth_sock, (3 + n_cam * DEPTH_FRAME_FLOATS) * 4 + 64)
            if depth_data is not None:
                _seq, _stamp, dn, depth_m = unpack_depth(depth_data)
                if dn == n_cam:
                    gate.update(depth_metres_to_obs(depth_m, DEPTH_NEAR, DEPTH_FAR),
                                time.perf_counter())

            # 非阻塞清空状态套接字缓冲区，获取最新硬件状态
            state_data = _drain(state_sock, STATE_BYTES + 64)
            if state_data is None:
                continue
            step_id, _quat, ang_vel, proj_grav, q, dq, cmd, depth_far_hold, mode = unpack_state(state_data)

            # 若请求了行走模式但未加载行走模型，则降级保持在躲避模式并输出告警
            if mode == 1 and walk_session is None:
                if (time.perf_counter() - last_stale_warn) > 1.0:
                    print("[dodge_policy] 警告: 请求了行走模式但未指定 --walk-onnx，保持在躲避模式",
                          flush=True)
                    last_stale_warn = time.perf_counter()
                mode = 0

            # 躲避模式下为原地站立躲避：忽略摇杆推力速度指令（主要由相机视觉引导躲避）；
            # 仅在行走模式下由手柄摇杆速度指令驱动
            if mode == 0:
                cmd = np.zeros(3, dtype=np.float32)

            # 检测硬件复位（step_id 发生倒退）：清空历史状态与环形缓冲区
            if step_id < prev_step_id:
                proprio.reset()
                ring.reset()
                last_action[:] = 0.0
                print(f"[dodge_policy] 检测到硬件复位 (step_id {prev_step_id}->{step_id})；清空历史状态",
                      flush=True)
            prev_step_id = step_id

            # 模式切换边沿检测：触发关节目标平滑插值过渡 (Target Blend)
            if mode != prev_mode:
                blend_from = last_target.copy()
                blend_i = 0
                print(f"[dodge_policy] 策略模式切换 -> {'WALK (行走)' if mode == 1 else 'DODGE (躲避)'}", flush=True)
                prev_mode = mode

            # 检查深度数据新鲜度
            tnow = time.perf_counter()
            if mode == 0 and gate.is_stale(tnow, args.depth_stale_s) and (tnow - last_stale_warn) > 1.0:
                print("[dodge_policy] 警告: 深度帧超时；保持末帧（机器人暂停主动反应，保持站立）",
                      flush=True)
                last_stale_warn = tnow

            # 深度环形缓冲区每步均推入数据（开销极低），确保在行走切回躲避时视觉历史无感知滞后
            # X 键触发全远平面深度输入 (depth_far_hold)
            if depth_far_hold != prev_depth_far_hold:
                print(f"[dodge_policy] 深度置远锁定 (depth-far hold): {'开启 (忽略相机输入)' if depth_far_hold else '关闭'}",
                      flush=True)
                prev_depth_far_hold = depth_far_hold
            frame = far_frame if depth_far_hold else gate.frame()
            depth_stacked = ring.push(frame)

            # 更新当前控制步本体感受观测（双策略共享）
            proprio.append(
                base_ang_vel=ang_vel,
                proj_grav=proj_grav,
                command=cmd,
                joint_pos_rel=(q - DEFAULT_POS),
                joint_vel_rel=dq,
                last_action=last_action,
            )

            # 执行 ONNX 模型推理
            if mode == 1:  # 行走模式：仅输入本体感受 (384 维)
                obs = proprio.vector().reshape(1, -1)
                action = walk_session.run(None, {walk_inp: obs})[0][0].astype(np.float32)
            else:          # 躲避模式：输入本体感受 ++ 堆叠深度 (960 维)
                obs = assemble_obs(proprio.vector(), depth_stacked)
                action = session.run(None, {inp_name: obs})[0][0].astype(np.float32)
            last_action = action.copy()

            # 动作反归一化: target = DEFAULT_POS + action * ACTION_SCALE
            active_target = DEFAULT_POS + action * ACTION_SCALE

            # 模式切换过渡插值：在 blend_ticks 步内平滑过渡，避免目标位置阶跃
            if blend_i < args.blend_ticks:
                blend_i += 1
                alpha = blend_i / args.blend_ticks
                target = (1.0 - alpha) * blend_from + alpha * active_target
            else:
                target = active_target
            last_target = target.copy()

            # 将目标关节角度下发给硬件控制节点
            state_sock.sendto(pack_action(target.astype(np.float32)), hw_addr)
    except KeyboardInterrupt:
        pass
    finally:
        state_sock.close()
        depth_sock.close()
        print("dodge_policy 策略节点已停止。")


def _drain(sock, bufsize):
    """非阻塞循环读取套接字，清空累积数据报并仅返回最新的一包（若无数据则返回 None）。
    对于 UDP 实时控制极其关键，彻底杜绝历史积压包导致的控制延迟。"""
    latest = None
    try:
        while True:
            latest, _ = sock.recvfrom(bufsize)
    except BlockingIOError:
        pass
    return latest


if __name__ == "__main__":
    main()
