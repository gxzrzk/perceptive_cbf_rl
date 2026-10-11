"""单目深度躲避部署栈的轻量级 UDP 数据包封包与解包同步模块。

在部署架构中的角色：
-------------------
为了在机器人实机实现毫秒级超低延迟通信，本模块替代了沉重的 ROS2/DDS 中间件，
实现相机节点 (camera_node)、策略节点 (dodge_policy) 与硬件节点 (hardware_node)
三个独立进程之间的无锁高效数据交互。

所有 UDP 数据包载荷均采用小端格式 (Little-Endian) float32 二进制编码。
定义了三种固定格式的数据报文与对应通信端口：

1. 状态报文 State (hardware_node -> dodge_policy, 频率 50 Hz):
   - 通信通道: 从 UDP_HW_PORT 发送到 UDP_POLICY_PORT (9871)
   - 载荷构成: 74 个 float32 (共 296 字节)
     [ step_id(1), quat_wxyz(4), base_ang_vel(3), proj_grav(3),
       q[29], dq[29], cmd[3], depth_far_hold(1), mode(1) ]
   - 机制说明:
     * ``step_id``: 硬件步数计数器。若策略检测到 step_id 发生反向跳变（例如从 1500 变回 0），
       即识别到硬件发生重启/复位，策略会立即清空本体感受历史与上一次动作缓存（冷重启）。
     * ``depth_far_hold``: 操作手柄 X 键触发的布尔标记（0.0 或 1.0）。若为 1，
       策略将忽略相机真实输入，强制注入全远平面（空旷）深度图，使机器人停止激进躲避，仅保持平稳站立。
     * ``mode``: 策略运行模式标志位。0 = 视觉躲避策略 (Dodge, 融合本体感受与深度图)；
       1 = 纯双足行走策略 (Walk, 仅输入本体感受，由手柄摇杆速度指令驱动)。

2. 深度报文 Depth (camera_node -> dodge_policy, 频率约 50 Hz，随相机而定):
   - 通信通道: 发送至 UDP_DEPTH_PORT (9872)
   - 载荷构成: (3 + n_cam * 144) 个 float32
     [ seq(1), stamp(1), n_cam(1), depth[n_cam * 144] ]
   - 机制说明:
     * 深度数据格式: 每台相机池化为 9 行 x 16 列 = 144 个浮点数（行优先，单位为米）。
     * ``stamp``: 发送端墙上时间戳（仅供信息参考）。由于两个独立进程的单调时钟并不严格同步，
       策略端在本地测量数据包到达时间来判断深度帧新鲜度（Freshness Gate）。

3. 动作报文 Action (dodge_policy -> hardware_node, 频率 50 Hz):
   - 通信通道: 发送至 UDP_HW_PORT (9870)
   - 载荷构成: 29 个 float32 (共 116 字节)
     [ q_target[29] ] -> 29 个关节的目标角度（弧度制）。
"""

import errno
import os
import signal
import socket
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------------------
# 通信端口与主机地址配置
# ---------------------------------------------------------------------------
UDP_HW_PORT = 9870       # hardware_node 硬件节点监听端口（接收来自策略的 Action 目标角度）
UDP_POLICY_PORT = 9871   # dodge_policy 策略节点监听端口（接收来自硬件节点的 State 状态包）
UDP_DEPTH_PORT = 9872    # dodge_policy 策略节点监听端口（接收来自相机节点的 Depth 深度图）
UDP_HOST = "127.0.0.1"   # 本地回路地址（跨进程低延迟通信）

# ---------------------------------------------------------------------------
# 数据包结构维度与字节大小定义
# ---------------------------------------------------------------------------
NUM_JOINTS = 29          # G1 机器人的控制关节数量
CMD_SIZE = 3             # 速度指令向量维度 [vx, vy, yaw_rate]
DEPTH_H = 9              # 池化后深度图行高
DEPTH_W = 16             # 池化后深度图列宽
DEPTH_FRAME_FLOATS = DEPTH_H * DEPTH_W  # 单相机每帧深度浮点数数量: 144

# 状态报文包含: step_id(1) + 四元数(4) + 机身角速度(3) + 投影重力(3) + 关节位置(29) + 关节速度(29) + 速度指令(3) + 深度挂起(1) + 模式(1) = 74
STATE_FLOATS = 1 + 4 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + CMD_SIZE + 1 + 1  # 74 个 float32
ACTION_FLOATS = NUM_JOINTS  # 动作报文包含 29 个关节目标角度 float32
STATE_BYTES = STATE_FLOATS * 4   # 状态包字节大小: 296 字节
ACTION_BYTES = ACTION_FLOATS * 4  # 动作包字节大小: 116 字节
DEPTH_HEADER_FLOATS = 3  # 深度包包头浮点数数量: [seq, stamp, n_cam]


# ---------------------------------------------------------------------------
# 套接字管理与清理辅助函数
# ---------------------------------------------------------------------------
def _kill_udp_port_holders(port: int) -> None:
    """自动清理并杀死非法占用指定 UDP 端口的僵尸进程（防止前一次异常退出遗留进程阻塞新进程启动）。"""
    my_pid = os.getpid()
    try:
        result = subprocess.run(
            ["fuser", f"{port}/udp"], capture_output=True, text=True, timeout=5
        )
        pids_str = result.stdout.strip()
        if not pids_str:
            return
        for tok in pids_str.split():
            try:
                pid = int(tok)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            print(f"正在杀死占用 UDP 端口 {port} 的残留进程 PID {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.2)
    except FileNotFoundError:
        print(
            f"警告: 系统中未找到 'fuser' 命令 - 无法自动终止端口 {port} 的占用进程。"
            "请安装 psmisc 或手动杀死占用端口的进程。"
        )
    except subprocess.TimeoutExpired:
        pass


def create_udp_socket(host: str, port: int) -> socket.socket:
    """创建并绑定一个 UDP 套接字，若端口已被占用则自动清理僵尸进程后重新绑定。

    参数:
        host: 绑定的 IP 地址（如 '127.0.0.1'）。
        port: 绑定的端口号（若为 0 则由系统自动分配临时发送端口）。

    返回:
        socket.socket: 已绑定并设置 SO_REUSEADDR 的 UDP 套接字对象。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        print(f"UDP 端口 {port} 被占用 - 正在尝试清理占用进程...")
        _kill_udp_port_holders(port)
        sock.bind((host, port))
    return sock


# ---------------------------------------------------------------------------
# 状态报文封包与解包 (硬件节点 -> 策略节点)
# ---------------------------------------------------------------------------
def pack_state(
    step_id: int,
    quat: np.ndarray,
    base_ang_vel: np.ndarray,
    proj_grav: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    cmd: np.ndarray,
    depth_far_hold: bool = False,
    mode: int = 0,
) -> bytes:
    """将硬件节点采集到的机器人全状态打包为二进制字节流 (74 个 float32)。

    参数:
        step_id: 控制步计数序号（用于检测复位重置）。
        quat: 浮动基座姿态四元数 [w, x, y, z]。
        base_ang_vel: 机身坐标系下的角速度 [wx, wy, wz] (rad/s)。
        proj_grav: 投影重力向量 [gx, gy, gz] (通过四元数反向旋转计算)。
        q: 29 个关节的当前测量位置角度 (rad)。
        dq: 29 个关节的当前测量速度 (rad/s)。
        cmd: 遥控器映射的目标速度指令 [vx, vy, yaw_rate]。
        depth_far_hold: 是否锁定全远深度观测（忽略视觉，保底站立）。
        mode: 运行模式 (0=Dodge 视觉躲避, 1=Walk 纯双足行走)。

    返回:
        bytes: 长度为 296 字节的二进制数据流。
    """
    buf = np.empty(STATE_FLOATS, dtype=np.float32)
    buf[0] = float(step_id)
    buf[1:5] = quat
    buf[5:8] = base_ang_vel
    buf[8:11] = proj_grav
    buf[11:40] = q
    buf[40:69] = dq
    buf[69:72] = cmd
    buf[72] = 1.0 if depth_far_hold else 0.0
    buf[73] = float(int(mode))
    return buf.tobytes()


def unpack_state(data: bytes):
    """解包二进制状态报文，还原为各状态物理量。

    参数:
        data: 接收到的 296 字节二进制数据。

    返回:
        tuple: (step_id, quat, base_ang_vel, proj_grav, q, dq, cmd, depth_far_hold, mode)
    """
    buf = np.frombuffer(data, dtype=np.float32)
    step_id = int(buf[0])
    quat = buf[1:5].copy()
    base_ang_vel = buf[5:8].copy()
    proj_grav = buf[8:11].copy()
    q = buf[11:40].copy()
    dq = buf[40:69].copy()
    cmd = buf[69:72].copy()
    depth_far_hold = bool(buf[72] >= 0.5)
    mode = int(round(float(buf[73])))
    return step_id, quat, base_ang_vel, proj_grav, q, dq, cmd, depth_far_hold, mode


# ---------------------------------------------------------------------------
# 动作报文封包与解包 (策略节点 -> 硬件节点)
# ---------------------------------------------------------------------------
def pack_action(q_target: np.ndarray) -> bytes:
    """将策略网络计算并反归一化得到的 29 关节目标位置打包为二进制数据 (116 字节)。

    参数:
        q_target: 长度为 29 的目标关节位置数组 (rad)。

    返回:
        bytes: 长度为 116 字节的二进制数据流。
    """
    return np.asarray(q_target, dtype=np.float32).reshape(ACTION_FLOATS).tobytes()


def unpack_action(data: bytes) -> np.ndarray:
    """解包二进制动作报文，还原为 29 关节目标位置数组。

    参数:
        data: 接收到的 116 字节动作二进制数据。

    返回:
        np.ndarray: 包含 29 个目标位置的 float32 数组。
    """
    return np.frombuffer(data, dtype=np.float32)[:ACTION_FLOATS].copy()


# ---------------------------------------------------------------------------
# 深度报文封包与解包 (相机节点 -> 策略节点)
# ---------------------------------------------------------------------------
def pack_depth(seq: int, stamp: float, n_cam: int, depth: np.ndarray) -> bytes:
    """将池化后的深度图数据打包为二进制报文。

    参数:
        seq: 相机帧递增序列号。
        stamp: 相机节点发送时的墙上时间戳。
        n_cam: 相机数量 (单目为 1)。
        depth: 展平后的深度数据（单位: 米，长度必须严格等于 n_cam * 144）。

    返回:
        bytes: 打包后的二进制深度报文。
    """
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    if depth.size != n_cam * DEPTH_FRAME_FLOATS:
        raise ValueError(
            f"深度数据大小 {depth.size} 不匹配期望大小: n_cam({n_cam}) * {DEPTH_FRAME_FLOATS}"
        )
    buf = np.empty(DEPTH_HEADER_FLOATS + depth.size, dtype=np.float32)
    buf[0] = float(seq)
    buf[1] = float(stamp)
    buf[2] = float(n_cam)
    buf[3:] = depth
    return buf.tobytes()


def unpack_depth(data: bytes):
    """解包二进制深度报文。

    参数:
        data: 接收到的深度报文字节流。

    返回:
        tuple: (seq, stamp, n_cam, depth)
            - seq (int): 序列号
            - stamp (float): 发送时间戳
            - n_cam (int): 相机数量
            - depth (np.ndarray): 长度为 n_cam * 144 的深度数据数组（单位: 米）
    """
    buf = np.frombuffer(data, dtype=np.float32)
    seq = int(buf[0])
    stamp = float(buf[1])
    n_cam = int(buf[2])
    depth = buf[3 : 3 + n_cam * DEPTH_FRAME_FLOATS].copy()
    return seq, stamp, n_cam, depth
