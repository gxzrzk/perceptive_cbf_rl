"""Sim2sim 仿真节点：通过实机部署 UDP 协议驱动 MuJoCo 躲避仿真环境。

在部署架构中的角色与意义：
-------------------------
本节点用于在**零硬件风险**的前提下，端到端完整验证实机部署软件栈的全部逻辑与通信链路。
在架构中，它一人分饰两角，同时充当实机上的 ``hardware_node`` 与 ``camera_node``：
1. 构建真实的 ``Unitree-G1-AMP-Dodge-Depth-Single-Flat`` 仿真环境（包含机器人刚体动力学、
   投掷球体以及头部深度相机传感器渲染）；
2. 提取硬件级原始观测物理量（机身四元数、局部角速度、29 关节 q/dq、米制原始深度图）；
3. 将上述信号按照与实机完全相同的 UDP 数据包格式发送给后台的 ``dodge_policy`` 策略节点；
4. 接收策略网络算出的 29 关节目标角度，逆向反求出动作值并作用于仿真步进 (env.step)。

环境解耦与架构验证：
-------------------
- ``sim_node`` 运行在开发机环境 (DEV env, 依赖 mjlab/MuJoCo/PyTorch)；
- ``dodge_policy`` 运行在独立的部署环境 (DEPLOY env, 仅需 onnxruntime + numpy)；
- 两个独立进程仅通过本地回路 UDP (:9870, :9871, :9872) 连接。
这种进程间彻底解耦能够全面覆盖并检验实机部署中的：
观测拼接排布、深度时间堆叠环形缓冲区、归一化算法、动作反归一化缩放以及复位重置信号。

关键安全防线：
-------------
节点启动时会执行**致命关节点顺序强校验**：断言 ``POLICY_JOINT_NAMES`` 必须与 MuJoCo 环境中的
关节索引顺序完全一致（这是实机部署中最隐蔽、最致命的静默 Bug，若顺序错位会导致机器人起立瞬间摔倒）。
"""

import argparse
import ast
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deploy.common.g1_deploy_constants import (
    ACTION_SCALE,
    DEFAULT_POS,
    DEPTH_H,
    DEPTH_W,
    NUM_JOINTS,
    POLICY_JOINT_NAMES,
)
from deploy.common.udp_sync import (
    ACTION_BYTES,
    UDP_DEPTH_PORT,
    UDP_HOST,
    UDP_HW_PORT,
    UDP_POLICY_PORT,
    create_udp_socket,
    pack_depth,
    pack_state,
    unpack_action,
)


class UdpDodgeBridge:
    """策略桥接类：将 MuJoCo 仿真环境与独立的 deploy dodge_policy 策略节点双向连通。

    在每个控制步周期：
    1. 提取仿真物理量并封装 State 与 Depth 数据包发送给策略节点；
    2. 等待并接收策略节点输出的目标关节角度，反向计算无量纲动作供仿真 step 使用；
    3. 在回合重置时将 step_id 归零，通知策略节点重置本体感受历史。
    """

    def __init__(self, env, torch_mod, device, gravity_dir, depth_key,
                 state_sock, depth_sock, action_sock, depth_dim, depth_aug):
        self.env = env
        self.torch = torch_mod
        self.device = device
        self.depth_key = depth_key
        self.depth_aug = depth_aug
        self.external_depth = False  # 若为 True，则由真实相机节点 (camera_node) 提供深度，本节点不发送仿真深度
        self.zero_command = False    # 若为 True，强制下发 0 速度指令（原地站立躲避）
        self.mode = 0                # 0 = 躲避模式 (Dodge), 1 = 行走模式 (Walk)
        # Viser 交互界面手动速度调节滑条 (vx, vy, wz)
        self.manual_cmd = np.zeros(3, dtype=np.float32)
        self.use_manual_cmd = False
        self.depth_decim = 1         # 深度降采样间隔（每 N 步发送一帧深度，模拟低帧率相机）
        self._warned_no_stash = False
        self.state_sock = state_sock
        self.depth_sock = depth_sock
        self.action_sock = action_sock
        self.depth_dim = depth_dim
        self._gravity = gravity_dir
        self.last_target = DEFAULT_POS.copy()
        self.seq = 0
        self._prev_len = -1
        # 存活表现统计
        self.n_steps = 0
        self.episodes = 0
        self.ep_lens = []
        self._t_log = time.perf_counter()
        self._dt = float(env.step_dt)

    def reset(self):
        """重置内部关节目标为默认站立姿态。"""
        self.last_target = DEFAULT_POS.copy()

    def _depth_metres(self):
        """提取仿真中头部相机的米制池化深度图。
        若启用了 depth_aug，提取经过域随机化加噪后的米制深度（贴合训练分布）；
        否则提取未经加噪的纯净传感器渲染深度。"""
        env = self.env
        if self.depth_aug:
            store = getattr(env, "_depth_aug_display", None)
            frame = store.get(self.depth_key) if store else None
            if frame is not None:
                return frame[0].reshape(DEPTH_H * DEPTH_W).detach().cpu().numpy().astype(np.float32)
            if not self._warned_no_stash:
                print("[sim_node] 警告: 请求了深度加噪但暂无暂存数据；当前步回退至纯净传感器深度。", flush=True)
                self._warned_no_stash = True
        return (env.scene[self.depth_key].data.depth[0].reshape(DEPTH_H * DEPTH_W)
                .detach().cpu().numpy().astype(np.float32))

    def __call__(self, obs=None):
        """主控制步回调：提取状态 -> UDP发送 -> UDP接收动作 -> 反归一化返回给仿真环境。"""
        env = self.env
        step_id = int(env.episode_length_buf[0].item())
        # 检测回合重置：若步数反向跳变，记录存活统计
        if 0 <= step_id < self._prev_len:
            self.episodes += 1
            self.ep_lens.append(self._prev_len)
        self._prev_len = step_id

        # 提取仿真物理量
        q = env.scene["robot"].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        dq = env.scene["robot"].data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
        quat = env.scene["robot"].data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
        ang_vel = env.scene["robot"].data.root_link_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        proj_grav = env.scene["robot"].data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
        cmd = env.command_manager.get_command("twist")[0, :3].detach().cpu().numpy().astype(np.float32)
        if self.use_manual_cmd:
            cmd = self.manual_cmd.copy()
        elif self.zero_command:
            cmd[:] = 0.0

        # 向策略节点发送 State 状态报文
        self.state_sock.sendto(
            pack_state(step_id, quat, ang_vel, proj_grav, q, dq, cmd, mode=self.mode),
            (UDP_HOST, UDP_POLICY_PORT))

        # 按设定的降采样间隔发送 Depth 深度报文（外部相机模式下跳过）
        if not self.external_depth and (self.seq % self.depth_decim == 0):
            depth_m = self._depth_metres()
            self.depth_sock.sendto(pack_depth(self.seq, time.time(), 1, depth_m),
                                   (UDP_HOST, UDP_DEPTH_PORT))
        self.seq += 1

        # 接收策略节点发回的 Action 关节目标（清空缓冲区至最新帧，超时则保持上一步目标）
        data = None
        self.action_sock.settimeout(0.1)
        try:
            data, _ = self.action_sock.recvfrom(ACTION_BYTES + 64)
        except (OSError,):
            pass
        if data is not None:
            self.action_sock.setblocking(False)
            try:
                while True:
                    data, _ = self.action_sock.recvfrom(ACTION_BYTES + 64)
            except (BlockingIOError, OSError):
                pass
            self.action_sock.setblocking(True)
            self.last_target = unpack_action(data)

        # 反归一化逆变换：action = (target_pos - DEFAULT_POS) / ACTION_SCALE
        action_np = (self.last_target - DEFAULT_POS) / ACTION_SCALE
        self.n_steps += 1
        now = time.perf_counter()
        if now - self._t_log >= 2.0:
            recent = self.ep_lens[-20:]
            mean_ep = (sum(recent) / len(recent)) if recent else float("nan")
            print(f"[sim_node] 仿真步数={self.n_steps} 回合数={self.episodes} "
                  f"近期平均存活={mean_ep:.0f} 步 ({mean_ep * self._dt:.1f} 秒) "
                  f"当前回合步数={step_id}", flush=True)
            self._t_log = now
        return self.torch.as_tensor(action_np, device=self.device).reshape(1, NUM_JOINTS)


def main():
    parser = argparse.ArgumentParser(description="Sim2sim 仿真节点 (通过部署 UDP 协议驱动 mjlab)")
    parser.add_argument("task_id", nargs="?", default="Unitree-G1-AMP-Dodge-Depth-Single-Flat")
    parser.add_argument("--frame-offsets", default="(0,3,8,18)",
                        help="深度历史时间堆叠步数偏移，必须与策略节点及 ONNX 严格一致")
    parser.add_argument("--device", default=None, help="运行设备 cuda:0 或 cpu (默认优先 cuda)")
    parser.add_argument("--viewer", default="none", choices=["none", "viser", "native"],
                        help="可视化模式: none (无头纯统计); viser (网页端可视化); native (本地 MuJoCo 窗口)")
    parser.add_argument("--reset-stand", action=argparse.BooleanOptionalAction, default=True,
                        help="是否在回合重置时回到默认标称站立位 (默认开启，对齐实机起步状态)")
    parser.add_argument("--depth-aug", action=argparse.BooleanOptionalAction, default=True,
                        help="是否向策略发送带有 Sim2Real 域随机化加噪的深度图 (默认开启，贴合训练策略)")
    parser.add_argument("--external-depth", action="store_true",
                        help="不发送仿真深度数据：由外部运行的真实相机节点 (如 camera_node_etam) 负责向策略提供深度。"
                             "用于测试真实相机与仿真机器人的混合在环验证")
    parser.add_argument("--pause-throws", action="store_true",
                        help="暂停仿真中的自动投掷球体（与 --external-depth 配合使用，允许向真实相机扔球）")
    parser.add_argument("--zero-command", action="store_true",
                        help="强制将速度指令置零 (原地站立)，去除回退偏置，纯粹检验策略对飞球的躲避响应")
    parser.add_argument("--depth-decimation", type=int, default=1,
                        help="深度发布间隔（每 N 步发送一帧深度，1=50Hz, 5=10Hz 模拟慢速相机）")
    parser.add_argument("--max-steps", type=int, default=0, help="无头模式下最大运行步数 (0 表示无限运行)")
    args = parser.parse_args()

    import torch

    import src.tasks.amp_loco.config.g1 as _g1  # noqa: F401 (注册 G1 躲避任务)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    offsets = tuple(ast.literal_eval(args.frame_offsets))

    env_cfg = load_env_cfg(args.task_id, play=True)
    env_cfg.depth_frame_offsets = offsets
    env_cfg.scene.num_envs = 1

    # 替换重置事件：重置到默认标称站立姿态
    if args.reset_stand and "reset_from_motion" in env_cfg.events:
        from src.tasks.amp_loco.mdp.events import reset_to_default_stand

        stand_event = env_cfg.events["reset_from_motion"]
        stand_event.func = reset_to_default_stand
        stand_event.params = {}
        print("[sim_node] reset_stand: 回合重置将回到默认标称站立姿态")

    # 开启深度图加噪预览
    if args.depth_aug:
        from src.tasks.amp_loco.config.g1.dodge_env_cfgs import enable_depth_aug_preview

        enable_depth_aug_preview(env_cfg)
        print("[sim_node] depth_aug: 已开启训练级深度域随机化加噪数据流")

    print(f"[sim_node] 正在构建仿真环境 {args.task_id} (设备: {device}, offsets: {offsets}, viewer: {args.viewer})...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if args.depth_aug:
        env._depth_aug_display_enabled = True
    env.reset()

    # 核心安全校验：强行断言环境关节顺序与策略关节顺序完全一致！
    env_joint_names = list(env.scene["robot"].joint_names)
    if env_joint_names != list(POLICY_JOINT_NAMES):
        print("[sim_node] 致命错误: 仿真环境关节顺序 != POLICY_JOINT_NAMES！\n"
              f"  仿真环境关节: {env_joint_names}\n  策略期望关节: {list(POLICY_JOINT_NAMES)}", file=sys.stderr)
        sys.exit(1)
    print(f"[sim_node] 关节顺序校验通过: {NUM_JOINTS} 个关节顺序严格吻合 POLICY_JOINT_NAMES")

    depth_key = "head_depth_single"
    try:
        _ = env.scene[depth_key]
    except KeyError:
        print(f"[sim_node] 致命错误: 场景中未找到深度传感器 '{depth_key}' (是否为单目任务？)", file=sys.stderr)
        sys.exit(1)

    state_sock = create_udp_socket(UDP_HOST, 0)
    depth_sock = create_udp_socket(UDP_HOST, 0)
    action_sock = create_udp_socket(UDP_HOST, UDP_HW_PORT)
    if args.external_depth:
        print(f"[sim_node] 状态发送->{UDP_POLICY_PORT}, 深度模式: 外部输入(真实相机), 动作接收<-{UDP_HW_PORT}")
    else:
        print(f"[sim_node] 状态发送->{UDP_POLICY_PORT}, 深度发送->{UDP_DEPTH_PORT}, 动作接收<-{UDP_HW_PORT}")
    print("[sim_node] 请在另一个终端启动策略节点 (或运行 play_sim_dodge.sh 自动化脚本)。")

    if args.pause_throws:
        env._dodge_throw_paused = True
        print("[sim_node] 仿真中自动抛球已暂停")

    bridge = UdpDodgeBridge(
        env, torch, device, np.array([0.0, 0.0, -1.0], dtype=np.float32), depth_key,
        state_sock, depth_sock, action_sock, DEPTH_H * DEPTH_W, args.depth_aug,
    )
    bridge.external_depth = bool(args.external_depth)
    bridge.zero_command = bool(args.zero_command)
    bridge.depth_decim = max(1, int(args.depth_decimation))
    if args.zero_command:
        print("[sim_node] 速度指令已强制置零 (原地躲避)")

    try:
        if args.viewer == "viser":
            from mjlab.viewer import ViserPlayViewer

            class _DodgeWalkViewer(ViserPlayViewer):
                """定制 Viser 网页交互界面：包含双模式切换开关、速度摇杆滑条以及抛球调试控制箱。"""
                def setup(self):
                    super().setup()
                    gui = self._server.gui

                    # --- 行走/躲避模式切换控件 ---
                    cb = gui.add_checkbox("行走模式 (Walk mode)", initial_value=False)
                    def _on_mode(_=None, _cb=cb):
                        bridge.mode = 1 if _cb.value else 0
                        print(f"[sim_node] 模式切换 -> {'WALK (行走)' if bridge.mode else 'DODGE (躲避)'}", flush=True)
                    cb.on_update(_on_mode)

                    # --- 手动速度指令 (vx, vy, wz) 滑条控件 ---
                    with gui.add_folder("手动速度指令 (vx, vy, wz)"):
                        man = gui.add_checkbox("启用手动指令 (Manual cmd)", initial_value=False)
                        s_vx = gui.add_slider("vx (前向速度)", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
                        s_vy = gui.add_slider("vy (横向速度)", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
                        s_wz = gui.add_slider("wz (偏航角速度)", min=-1.0, max=1.0, step=0.05, initial_value=0.0)
                        stop = gui.add_button("立即归零 (Stop)")

                    def _push_cmd(_=None):
                        bridge.use_manual_cmd = man.value
                        bridge.manual_cmd = np.array(
                            [s_vx.value, s_vy.value, s_wz.value], dtype=np.float32)
                    for h in (man, s_vx, s_vy, s_wz):
                        h.on_update(_push_cmd)
                    def _on_stop(_=None):
                        s_vx.value = s_vy.value = s_wz.value = 0.0
                        _push_cmd()
                    stop.on_click(_on_stop)

                    # --- 抛球控制控件 ---
                    with gui.add_folder("抛球控制 (Ball throws)"):
                        pause = gui.add_checkbox("暂停自动抛球", initial_value=False)
                        b_hi = gui.add_button("高球 (头部上方)")
                        b_lo = gui.add_button("低球 (下肢部位)")
                        b_rand = gui.add_button("随机抛球")
                    pause.on_update(lambda _=None: setattr(env, "_dodge_throw_paused", pause.value))
                    def _throw(force_high):
                        env._dodge_throw_force_high = force_high
                        env._dodge_throw_once = True
                    b_hi.on_click(lambda _=None: _throw(True))
                    b_lo.on_click(lambda _=None: _throw(False))
                    b_rand.on_click(lambda _=None: _throw(None))

            _DodgeWalkViewer(env, bridge).run()
        elif args.viewer == "native":
            from mjlab.viewer import NativeMujocoViewer
            NativeMujocoViewer(env, bridge).run()
        else:
            # 无头纯命令行压测模式
            dt = float(env.step_dt)
            next_tick = time.perf_counter()
            while True:
                action = bridge(None)
                env.step(action)
                if args.max_steps and bridge.n_steps >= args.max_steps:
                    break
                next_tick += dt
                sleep_t = next_tick - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        state_sock.close()
        depth_sock.close()
        action_sock.close()
        env.close()
        print(f"[sim_node] 仿真已停止。共执行 {bridge.episodes} 回合，累计 {bridge.n_steps} 步。")


if __name__ == "__main__":
    main()
