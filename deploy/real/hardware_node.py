"""Unitree G1 机器人实机硬件接口与安全状态机控制节点 (无 ROS 依赖)。

在部署架构中的角色与最高安全准则：
---------------------------------
本节点是整个部署系统的**核心底座与安全屏障**，也是唯一一个绝不允许崩溃退出的进程
（视觉相机节点放在独立进程即是为了保护本节点不受视觉崩溃影响）：
1. 硬件接口管理 (Unitree SDK2 Bridge):
   - 持有 ``unitree_interface`` 动态库接口，通过低级控制 (LowCmd/LowState) 以 50 Hz 循环与 G1 物理电机及板载 IMU 交互；
   - 采集 29 个电机的真实角度 q、角速度 dq、机身姿态四元数 quat 与角速度 omega；
2. 动力学与指令解析：
   - 利用四元数逆旋转实时计算重力在机身坐标系的投影向量 projected_gravity；
   - 解析无线手柄左右摇杆推力，生成速度指令向量 cmd；
3. UDP 通信网关：
   - 打包 State 报文发送给策略节点 (:9871)，接收来自策略节点的 Action 关节目标 (:9870) 并转化为底层电机 PD 力矩；
4. 物理无线遥控器安全状态机 (Safety State Machine)：
   - **START 键**：从当前任意初始姿态，在 2 秒内平滑线性插值过渡到默认标称站立姿态 (DEFAULT_POS)；
   - **A 键**：激活 50 Hz 策略闭环控制循环，机器人开始根据策略网络自主平衡、躲避或行走；
   - **B 键**：平滑阻尼安全停机（将刚度 KP 置零，进入纯阻尼 KD 耗能停机状态；随时可再次按 START 重新起立）；
   - **SELECT 键**：**一键紧急停止 (Emergency Stop)**，在当前关节位置就地执行阻尼刹车并彻底退出程序。
     本节点退出会级联触发启动脚本的清理 trap，联动关闭相机节点与策略节点；
   - **X 键 (上升沿自锁切换)**：切换深度置远锁定 (depth_far_hold)，使策略接收全远深度，用于诊断原地纯站立稳定性；
   - **Y 键 (上升沿自锁切换)**：切换行走模式 (Walk Mode)，通知策略节点切换为双足移动步态，由左摇杆直接操控移速。
"""

import argparse
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unitree_interface

from deploy.common.command import command_from_sticks, make_command
from deploy.common.g1_deploy_constants import DEFAULT_POS, KD, KP, NUM_JOINTS
from deploy.common.udp_sync import (
    ACTION_BYTES,
    UDP_HOST,
    UDP_HW_PORT,
    UDP_POLICY_PORT,
    create_udp_socket,
    pack_state,
    unpack_action,
)

CONTROL_DT = 0.02  # 控制周期 20ms (对应 50 Hz 控制频率)

# ---------------------------------------------------------------------------
# 遥控手柄按键语义绑定 (匹配 Unitree SDK2 ControllerMapping 位掩码)
# ---------------------------------------------------------------------------
STAND_BUTTON = "start"        # START 键: 插值起立到默认微屈膝站立位
GO_BUTTON = "A"               # A 键: 启动策略闭环控制循环
DAMP_BUTTON = "B"             # B 键: 阻尼缓冲停机 (按 START 可重新起立)
KILL_BUTTON = "select"        # SELECT 键: 一键急停 (当前姿态阻尼并终止全部进程)
DEPTH_HOLD_BUTTON = "X"       # X 键: 自锁切换“深度置远锁定” (强制策略视为空旷空间)
WALK_MODE_BUTTON = "Y"        # Y 键: 自锁切换“行走模式 / 躲避模式” (通过摇杆操控行走)

# 宇树无线遥控手柄按键位掩码定义 (TWIST2 g1_wrapper ControllerMapping)
_KEYS_BITMASK = {
    "R1": 0x0001, "L1": 0x0002, "start": 0x0004, "select": 0x0008,
    "R2": 0x0010, "L2": 0x0020, "F1": 0x0040, "F2": 0x0080,
    "A": 0x0100, "B": 0x0200, "X": 0x0400, "Y": 0x0800,
    "up": 0x1000, "right": 0x2000, "down": 0x4000, "left": 0x8000,
}


def btn(ctrl, name: str) -> bool:
    """检查指定名称的手柄按键当前是否被按下。"""
    keys = getattr(ctrl, "keys", None)
    if keys is not None and name in _KEYS_BITMASK:
        return bool(keys & _KEYS_BITMASK[name])
    v = getattr(ctrl, name, None)
    return bool(v) if isinstance(v, bool) else False


def quat_rotate_inverse(q, v):
    """利用四元数 q (wxyz 顺序) 的逆旋转（共轭旋转）对三维向量 v 进行变换。
    在部署中用于将世界坐标系下的重力加速度 [0, 0, -1] 转换至机器人机身局部坐标系，
    得到机身局部的投影重力向量 projected_gravity。"""
    w, x, y, z = (float(c) for c in q)
    q_vec = np.array([x, y, z], dtype=np.float32)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * w * 2.0
    c = q_vec * float(np.dot(q_vec, v)) * 2.0
    return (a - b + c).astype(np.float32)


def _send_pd(robot, q_target, kp, kd):
    """底层电机力矩指令发送函数：向 29 个关节下发包含目标位置、零目标速度及 KP/KD 的 LowCmd。"""
    cmd = robot.create_zero_command()
    cmd.q_target = list(np.asarray(q_target, dtype=float))
    cmd.dq_target = [0.0] * NUM_JOINTS
    cmd.kp = list(np.asarray(kp, dtype=float))
    cmd.kd = list(np.asarray(kd, dtype=float))
    cmd.tau_ff = [0.0] * NUM_JOINTS
    robot.write_low_command(cmd)


def _wait_for(robot, button, hold_fn):
    """阻塞等待遥控手柄上的特定按键按下，在轮询等待的每个控制步周期调用 hold_fn 维持电机受控。"""
    while True:
        ctrl = robot.read_wireless_controller()
        if btn(ctrl, button):
            return
        hold_fn()
        time.sleep(CONTROL_DT)


def startup(robot, kp):
    """启动与平滑起立状态机：
    1. 阻尼等待用户按下 START 键；
    2. 在 2.0 秒内从当前实际关节位置线性插值平滑移动至标称默认站立姿态 DEFAULT_POS；
    3. 姿态到位后，保持默认站立姿态并等待用户按下 A 键启动策略控制主循环。
    """
    print(f"请按下手柄上的 {STAND_BUTTON} 键以插值移动至默认站立姿态...")
    _wait_for(
        robot, STAND_BUTTON,
        lambda: _send_pd(robot, np.array(robot.read_low_state().motor.q[:NUM_JOINTS]),
                         np.zeros(NUM_JOINTS), KD),
    )

    print("正在平滑移动至默认站立姿态 (2秒插值)...")
    q_start = np.array(robot.read_low_state().motor.q[:NUM_JOINTS], dtype=np.float32)
    n_steps = int(2.0 / CONTROL_DT)
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        _send_pd(robot, (1.0 - alpha) * q_start + alpha * DEFAULT_POS, kp, KD)
        time.sleep(CONTROL_DT)
    print("已成功到达默认站立姿态。")

    print(f"请按下手柄上的 {GO_BUTTON} 键以启动策略闭环控制循环...")
    _wait_for(robot, GO_BUTTON, lambda: _send_pd(robot, DEFAULT_POS, kp, KD))
    print("策略闭环控制已启动！")


def damp(robot, n_ticks=100):
    """平滑阻尼缓冲停机：将位置刚度 KP 归零，仅施加纯阻尼 KD，并保持在当前电机角度缓慢耗能制动。"""
    print("正在执行阻尼保护 (Damping)...")
    for _ in range(n_ticks):
        q = np.array(robot.read_low_state().motor.q[:NUM_JOINTS], dtype=np.float32)
        _send_pd(robot, q, np.zeros(NUM_JOINTS), KD)
        time.sleep(CONTROL_DT)


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 单目深度躲避硬件控制节点")
    parser.add_argument("--net", default="eth0", help="机器人底层通讯网络物理网卡接口名 (如 eth0)")
    parser.add_argument("--policy-ip", default=UDP_HOST, help="策略推理节点所在 IP 地址")
    parser.add_argument("--kp-scale", type=float, default=1.0,
                        help="位置刚度比例系数缩放因子 (默认 1.0 = 标准刚度)。"
                             "KD 阻尼和急停阻尼不受影响。在首次实机试跑时可设为 0.7~0.8 使关节更柔顺安全。")
    args = parser.parse_args()

    kp = (KP * args.kp_scale).astype(np.float32)
    print(f"[hardware] 刚度 KP 缩放因子 {args.kp_scale} -> 实际生效 KP 范围: {kp.min():.2f}..{kp.max():.2f} "
          f"(KD 阻尼保持标称值不变)")

    # 优雅停机信号处理（避免二次信号打断正在进行的阻尼保护）
    _shutting_down = [False]

    def _on_sigterm(signum, frame):
        if _shutting_down[0]:
            return
        _shutting_down[0] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)

    # 初始化 Unitree SDK2 硬件驱动
    robot = unitree_interface.UnitreeInterface.create_g1(args.net)
    if not robot.release_motion_control():
        print("[警告] 释放高层运动控制权失败；底层 LowCmd 写入可能被固件忽略。")
    robot.set_control_mode(unitree_interface.ControlMode.PR)

    # 创建 UDP 套接字：监听动作接收端口，绑定策略状态发送地址
    sock = create_udp_socket(UDP_HOST, UDP_HW_PORT)
    sock.setblocking(False)
    policy_addr = (args.policy_ip, UDP_POLICY_PORT)
    print(f"UDP通信就绪: 硬件端={UDP_HOST}:{UDP_HW_PORT}  策略端={args.policy_ip}:{UDP_POLICY_PORT}")

    gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    last_target = DEFAULT_POS.copy()
    depth_far_hold = False        # 手柄 X 键状态锁存
    prev_depth_hold_btn = False   # X 键上升沿检测
    walk_mode = False             # 手柄 Y 键行走模式锁存
    prev_walk_btn = False         # Y 键上升沿检测

    try:
        while True:
            # 状态机：起立插值流程
            startup(robot, kp)
            last_target = DEFAULT_POS.copy()
            step_id = 0  # 每次重新起立时计数器清零，通知策略重置历史

            # 50 Hz 策略控制主循环，直到收到按键 B 触发阻尼停机
            next_tick = time.perf_counter()
            damp_requested = False
            while not damp_requested:
                now = time.perf_counter()
                sleep_t = next_tick - now
                if sleep_t > 0:
                    time.sleep(sleep_t)
                next_tick += CONTROL_DT

                # 读取无线遥控手柄输入
                ctrl = robot.read_wireless_controller()
                if btn(ctrl, KILL_BUTTON):
                    # SELECT 键: 一键紧急停止
                    print("SELECT 按下 -- 一键紧急停止 (阻尼刹车并退出程序)。")
                    damp(robot, n_ticks=50)
                    return
                if btn(ctrl, DAMP_BUTTON):
                    # B 键: 平滑阻尼缓冲
                    damp_requested = True
                    break

                # X 键上升沿边沿触发: 切换深度置远锁定
                depth_hold_btn = btn(ctrl, DEPTH_HOLD_BUTTON)
                if depth_hold_btn and not prev_depth_hold_btn:
                    depth_far_hold = not depth_far_hold
                    print(f"[hardware] 深度置远锁定状态: {'开启 (忽略视觉输入，保底站立)' if depth_far_hold else '关闭 (正常对球反应)'}",
                          flush=True)
                prev_depth_hold_btn = depth_hold_btn

                # Y 键上升沿边沿触发: 切换行走与躲避模式
                walk_btn = btn(ctrl, WALK_MODE_BUTTON)
                if walk_btn and not prev_walk_btn:
                    walk_mode = not walk_mode
                    print(f"[hardware] 策略模式切换: {'WALK (手柄摇杆控制行走)' if walk_mode else 'DODGE (视觉主动躲避球体)'}",
                          flush=True)
                prev_walk_btn = walk_btn

                # 读取电机与 IMU 传感器硬件测量值
                state = robot.read_low_state()
                quat = np.array(state.imu.quat, dtype=np.float32)      # 四元数 [w, x, y, z]
                ang_vel = np.array(state.imu.omega, dtype=np.float32)  # 机身局部角速度
                q = np.array(state.motor.q[:NUM_JOINTS], dtype=np.float32)
                dq = np.array(state.motor.dq[:NUM_JOINTS], dtype=np.float32)
                proj_grav = quat_rotate_inverse(quat, gravity)
                cmd = command_from_sticks((ctrl.lx, ctrl.ly), (ctrl.rx, ctrl.ry))

                # 打包并通过 UDP 下发给策略节点
                sock.sendto(
                    pack_state(step_id, quat, ang_vel, proj_grav, q, dq, cmd,
                               depth_far_hold=depth_far_hold, mode=(1 if walk_mode else 0)),
                    policy_addr,
                )
                step_id += 1

                # 非阻塞获取策略节点最新计算的动作目标（若暂无新包则沿用上一步目标）
                action_data = None
                try:
                    while True:
                        action_data, _ = sock.recvfrom(ACTION_BYTES + 64)
                except BlockingIOError:
                    pass
                if action_data is not None:
                    last_target = unpack_action(action_data)

                # 将目标角度写入物理电机
                _send_pd(robot, last_target, kp, KD)

            damp(robot)
            print("已进入阻尼保护状态 (B)。按 START 键可重新起立；按 SELECT 或 Ctrl-C 退出。")
    except KeyboardInterrupt:
        pass
    finally:
        # 无论发生何种异常，首要执行电机阻尼保护（硬件安全第一）
        damp(robot, n_ticks=50)
        sock.close()
        print("hardware_node 硬件节点已安全退出。")


if __name__ == "__main__":
    main()
