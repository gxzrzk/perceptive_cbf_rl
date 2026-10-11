#!/usr/bin/env python3
"""基于 EfficientTAM 目标分割的高精度掩码深度相机节点 (Masked-Depth Camera Node)。

在部署架构中的角色与工作原理：
-----------------------------
本节点是实机与高质量感知仿真环境下的**生产级感知节点**，作为 ``camera_node.py`` 的即插即用替代。
它向策略节点发布完全相同协议的 UDP 深度报文，但包含关键的**前景目标分割与背景剔除**：
1. 图像采集与分割：
   - 从 ZED Mini 实时抓取全分辨率 BGR 彩色图像与原始深度图；
   - 利用轻量级视频目标分割模型 EfficientTAM（在打过 py3.8 兼容补丁的子模块中运行，无需 TensorRT）
     对飞掷的球体进行逐帧追踪；
2. 纯净深度图生成 (Ball-only Masked Depth):
   - 对应仿真中 ``BallOnlyDepthObs`` 观测模式：
     只保留分割掩码内的球体真实深度，画面中其余所有背景区域一律强制填充为 DEPTH_FAR (5.0m)；
   - 彻底解决了普通深度相机将地面、操作人员或周围杂物误识别为近处障碍物而引发误躲避的痛点；
3. 动态逼近门控 (Looming Gate / Static Mask):
   - 传统目标检测只要画面有球就会触发躲避，导致手持球或静止摆放的球也会让机器人剧烈反应；
   - 本节点引入逼近门控：利用最小二乘法拟合窗口内球体距离的变化率（逼近速度），
     若球体在世界坐标系下相对静止（距离变化率 < min_closing_speed），则自动掩码剔除（视为空旷）；
     唯有向机器人高速飞来的抛球才会放行深度，触发躲避动作！

交互操作指南：
-------------
启动后程序弹出显示窗口，在画面中**用鼠标左键单击球体一次**即可启动跟踪；
之后 EfficientTAM 利用时序记忆机制自主跨帧连续追踪，并在短暂出画重入时重新捕获。
控制热键：
- 'r' 或 'c': 重置跟踪状态，允许重新点击目标；
- 'f': 重新激活首帧冻结诊断模式；
- 'q': 安全退出程序。
"""

import argparse
import os
import signal
import sys
import time
from collections import deque

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RT = os.path.join(_REPO_ROOT, "deploy", "common", "RealtimeEfficientTAM")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _RT)

from deploy.common.g1_deploy_constants import DEPTH_FAR, DEPTH_H, DEPTH_NEAR, DEPTH_W
from deploy.common.udp_sync import (UDP_DEPTH_PORT, UDP_HOST,
                                    create_udp_socket, pack_depth)
from deploy.common.zed_depth_source import ZedDepthSource, mask_depth_to_far, percentile_pool_depth
from deploy.tools.etam_track_live import _quiet, depth_panel, mask_too_tall, overlay_mask

WINDOW = "camera_node_etam (鼠标点击球体启动跟踪; r=重置, f=重置冻结, q=退出)"


def main():
    parser = argparse.ArgumentParser(description="ZED -> EfficientTAM 纯净球体掩码深度图 -> UDP 节点")
    parser.add_argument("--hz", type=float, default=50.0, help="UDP 深度数据发布频率 (Hz，匹配控制周期)")
    parser.add_argument("--policy-ip", default="127.0.0.1", help="策略节点目标 IP")
    parser.add_argument("--serial", type=int, default=None, help="绑定的 ZED 相机硬件序列号")
    parser.add_argument("--pool-percentile", type=float, default=3.0, help="软最小值降采样百分位数")
    parser.add_argument("--checkpoint", default=os.path.join(_RT, "checkpoints", "efficienttam_s.pt"),
                        help="EfficientTAM 模型权重文件路径")
    parser.add_argument("--config", default="configs/efficienttam/efficienttam_s.yaml",
                        help="Hydra 配置文件路径")
    parser.add_argument("--tiny", action="store_true",
                        help="使用 efficienttam_ti_512x512 轻量模型 (在 Orin 上运行提速约 2 倍，推荐实机开启)")
    parser.add_argument("--conf", type=float, default=0.5, help="EfficientTAM 追踪置信度阈值")
    parser.add_argument("--no-view", action="store_true",
                        help="无图形界面模式（注意：此模式下无法鼠标点击目标，将始终发送全远平面）")
    parser.add_argument("--freeze-first", action="store_true",
                        help="诊断功能：锁存捕获到的第一帧有效（包含球体）池化深度图，后续始终发送该静态帧，"
                             "用于测试机器人对固定静态球体输入的姿态响应。按 'f' 键可重新拾取锁定")
    parser.add_argument("--static-mask", action="store_true",
                        help="逼近门控 (Looming Gate)：自动过滤静态物体。除非跟踪目标的平均距离正在向相机逼近，"
                             "否则将其掩码剔除（输出全远平面），避免手持静止球或背景误检触发机器人躲避")
    parser.add_argument("--min-closing-speed", type=float, default=1.0,
                        help="逼近速度阈值 (m/s)：目标平均距离减小的速度必须达到此阈值才被判定为向机器人飞来的威胁。"
                             "1.0 m/s 高于双目测距噪声抖动，且低于真实扔球速度 (~3-5 m/s)")
    parser.add_argument("--static-window", type=int, default=12,
                        help="最小二乘拟合逼近速度的历史帧窗口长度（较长窗口抗噪性能更好，且不增加飞球响应延迟）")
    args = parser.parse_args()
    if args.tiny:
        args.checkpoint = os.path.join(_RT, "checkpoints", "efficienttam_ti_512x512.pt")
        args.config = "configs/efficienttam/efficienttam_ti_512x512.yaml"

    import cv2
    import torch
    from efficient_track_anything.realtime_tam import build_predictor, start, track

    _shutting_down = [False]

    def _on_signal(signum, frame):
        if _shutting_down[0]:
            return
        _shutting_down[0] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # 初始化 EfficientTAM 推理预测器 (优先使用 GPU/CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with _quiet():
        state = build_predictor(args.config, args.checkpoint, device=device)
    # 初始化 ZED 相机源，keep_full=True 保留全分辨率原始深度图与彩色帧供分割使用
    src = ZedDepthSource(
        height=DEPTH_H, width=DEPTH_W, min_depth_m=DEPTH_NEAR, max_depth_m=DEPTH_FAR,
        pool_percentile=args.pool_percentile, fill_invalid=DEPTH_FAR,
        serial_number=args.serial, keep_full=True,
    )

    sock = create_udp_socket(UDP_HOST, 0)
    addr = (args.policy_ip, UDP_DEPTH_PORT)
    print(f"[camera_node_etam] 纯净掩码深度发布至 -> {args.policy_ip}:{UDP_DEPTH_PORT} @ {args.hz} Hz "
          f"(模型={'tiny 轻量' if args.tiny else 's 标准'})")

    # 鼠标点击交互回调
    pending = {"pt": None}
    if not args.no_view:
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                pending["pt"] = (int(x), int(y))
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, on_mouse)

    far_pooled = np.full((DEPTH_H, DEPTH_W), DEPTH_FAR, dtype=np.float32)
    tracking = False
    frozen = None  # --freeze-first 模式下的锁存帧
    depth_hist = deque(maxlen=max(2, int(args.static_window)))  # 逼近门控：近期球体中值距离历史队列 (m)
    seq = 0
    dt = 1.0 / float(args.hz)
    next_tick = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if next_tick - now > 0:
                time.sleep(next_tick - now)
            next_tick += dt

            depth, bgr = src.latest_full()
            m = None
            if bgr is not None:
                with _quiet():  # 抑制 RealtimeEfficientTAM 的控制台打印
                    if pending["pt"] is not None:
                        # 用户在画面上点击了目标：初始化视频序列与提示点
                        x, y = pending["pt"]
                        pending["pt"] = None
                        state.initialized = False
                        state.predictor.load_first_frame(bgr)
                        _ids, ml = start(state, points=np.array([[x, y]], np.float32),
                                         labels=np.array([1], np.int32), obj_id=0)
                        m = _mask_hw(ml, bgr.shape[0], bgr.shape[1], cv2)
                        tracking = True
                    elif tracking:
                        # 持续逐帧跟踪
                        _ids, ml = track(state, bgr, confidence_threshold=args.conf)
                        m = _mask_hw(ml, bgr.shape[0], bgr.shape[1], cv2)

            # 计算分割掩码覆盖下的代表性中值距离（米）
            m_dist = None
            if m is not None and bool(m.any()) and depth is not None:
                mpx = np.asarray(depth, dtype=np.float32)[m]
                mpx = mpx[np.isfinite(mpx)]
                if mpx.size:
                    m_dist = float(np.median(mpx))

            # 目标合法性校验：
            # 1. 掩码有效性 (m.any())；
            # 2. 距离感知几何高度检查 mask_too_tall：
            #    排除分割器误识别出的人体、墙体或远方误检，若掩码几何尺寸不符合该距离下的球体，则视为空旷
            too_tall = m is not None and bool(m.any()) and mask_too_tall(m, dist=m_dist)
            valid_ball = m is not None and bool(m.any()) and not too_tall
            ball_dist = m_dist if valid_ball else None

            # 逼近门控 (Looming Gate / Static Mask):
            # 若开启了 --static-mask，利用一阶多项式拟合距离与时间的变化斜率，
            # 若逼近速度低于设定阈值，则判定为静止物体，标记为 static_masked 并强制输出全远平面
            static_masked = False
            if args.static_mask:
                if valid_ball and m_dist is not None:
                    depth_hist.append(m_dist)
                    if len(depth_hist) >= depth_hist.maxlen:
                        # 最小二乘法拟合 slope = d(depth)/dt (单位: m/s)
                        d = np.asarray(depth_hist, dtype=np.float64)
                        t = np.arange(d.size, dtype=np.float64) * dt
                        slope = np.polyfit(t, d, 1)[0]
                        closing = -slope  # 正数表示正在逼近 (距离变小)
                        if closing < args.min_closing_speed:
                            static_masked = True  # 相对静止或远离中，不构成撞击威胁
                else:
                    depth_hist.clear()

            send_ball = valid_ball and not static_masked
            if depth is not None:
                sel = m if send_ball else np.zeros(depth.shape[:2], dtype=bool)
                # 将背景覆写为远平面并执行软最小值池化
                pooled = percentile_pool_depth(
                    mask_depth_to_far(depth, sel, DEPTH_FAR),
                    DEPTH_H, DEPTH_W, args.pool_percentile, fill_invalid=DEPTH_FAR,
                )
            else:
                pooled = far_pooled

            # 首帧冻结诊断模式处理
            if args.freeze_first:
                if frozen is None and send_ball:
                    frozen = pooled.copy()
                    print("[camera_node_etam] 已锁定第一帧有效分割深度 -> 持续静态发送", flush=True)
                sent = frozen if frozen is not None else pooled
            else:
                sent = pooled

            # 打包并通过 UDP 下发深度数据
            sock.sendto(pack_depth(seq, time.time(), 1, sent.reshape(-1)), addr)
            seq += 1

            # 渲染图形调试窗口
            if not args.no_view and bgr is not None:
                disp = overlay_mask(bgr, m) if valid_ball else bgr.copy()
                label = ("静态物体已屏蔽 (全远深度)" if static_masked else "跟踪中") if valid_ball else (
                    "已剔除: 高度过大非球体" if too_tall else
                    "请用鼠标点击球体" if not tracking else "目标丢失 (全远深度) - 请重新点击")
                cv2.putText(disp, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                dist_txt = f"球体距离: {ball_dist:.2f} m" if ball_dist is not None else "球体距离: -- m"
                cv2.putText(disp, dist_txt, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                if args.freeze_first:
                    froze = frozen is not None
                    cv2.putText(disp, "深度已冻结 (锁定帧)" if froze else "冻结就绪 (等待目标)",
                                (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
                panel_label = "策略深度 (已冻结)" if (args.freeze_first and frozen is not None) else "策略深度输入"
                disp = np.hstack([disp, depth_panel(sent, disp.shape[1] // 2, disp.shape[0], panel_label)])
                cv2.imshow(WINDOW, disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k in (ord("r"), ord("c")):
                    state.initialized = False
                    tracking = False
                    frozen = None
                if k == ord("f"):
                    frozen = None
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        sock.close()
        if not args.no_view:
            cv2.destroyAllWindows()


def _mask_hw(mask_logits, h, w, cv2):
    """将 EfficientTAM 输出的 logits 张量转换为布尔掩码，并重采样至输入画幅 (h, w)。"""
    m = (mask_logits[0, 0] > 0.0).detach().cpu().numpy()
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return m


if __name__ == "__main__":
    main()
