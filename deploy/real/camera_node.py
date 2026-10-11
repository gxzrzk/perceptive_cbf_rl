"""单目/多目原始深度图采集与 UDP 发布相机节点。

在部署架构中的角色与故障隔离：
-----------------------------
本节点作为独立的感知进程运行，核心设计目标为 **故障隔离 (Fault Isolation)**：
双目深度相机驱动 (pyzed / ZED-SDK) 可能因 USB 拔插抖动、底层驱动段错误或固件阻塞出现异常。
将其与电机控制进程 (hardware_node) 彻底物理隔离，确保无论视觉感知出现何种异常，
负责机器人平衡保持、阻尼制动和安全急停的硬件控制节点绝对不会被连带崩溃。

运行流程：
---------
1. 启动并管理一个（或多个）``ZedDepthSource`` 实例；
2. 在主循环中以 ``--hz`` (默认 50 Hz，对应策略控制频率) 读取池化后的 ``[9, 16]`` 米制深度帧；
3. 打包为二进制 Depth 深度报文，发送至本地 ``UDP_DEPTH_PORT`` (9872)；
4. 自愈防护机制：若相机采集线程超过 2 秒未产生新帧，自动触发相机重连重启，在此期间持续发送
   最后有效帧缓存，保障策略节点感知不中断；
5. 提供 ``--dry-run`` 虚拟模式（无需物理相机，发送全远平面深度，用于实机上电前的全链路联调）。
"""

import argparse
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.common.g1_deploy_constants import DEPTH_FAR, DEPTH_H, DEPTH_NEAR, DEPTH_W
from deploy.common.udp_sync import (
    UDP_DEPTH_PORT,
    UDP_HOST,
    create_udp_socket,
    pack_depth,
)


def _depth_panel(img2d, scale, label, cv2, interp):
    """生成单路相机的 JET 伪彩色深度显示面板（红=近处, 蓝=远处）。
    尺寸放大至 DEPTH_W*scale x DEPTH_H*scale，并标记最近点坐标十字准星及距离统计信息。"""
    h, w = img2d.shape
    tw, th = DEPTH_W * scale, DEPTH_H * scale
    norm = np.clip((img2d - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR), 0.0, 1.0)
    u8 = (255 - norm * 255).astype(np.uint8)  # 红色代表近处物体（危险告警色）
    big = cv2.resize(u8, (tw, th), interpolation=interp)
    color = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    near_frac = float((norm < 0.5).mean()) * 100.0
    # 查找最近的有效像素点坐标并在面板上绘制十字标记
    masked = np.where(np.isfinite(img2d), img2d, np.inf)
    r, c = np.unravel_index(int(np.argmin(masked)), masked.shape)
    mn = float(masked[r, c])
    loc = ""
    if np.isfinite(mn):
        mx, my = int((c + 0.5) / w * tw), int((r + 0.5) / h * th)
        cv2.circle(color, (mx, my), 9, (0, 0, 0), 2)            # 黑色外圈增强对比度
        cv2.drawMarker(color, (mx, my), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        loc = f"@({r},{c})"
    else:
        mn = DEPTH_FAR
    cv2.putText(color, f"{label}  min={mn:.2f}m{loc}  <2.5m:{near_frac:.0f}%",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return color


def _colorbar(width, cv2, height=28):
    """绘制水平色条图例：左侧红色对应 DEPTH_NEAR (0.1m)，右侧蓝色对应 DEPTH_FAR (5.0m)。"""
    grad = np.linspace(255, 0, max(width, 2)).astype(np.uint8)[None, :]
    bar = cv2.applyColorMap(np.repeat(grad, height, axis=0), cv2.COLORMAP_JET)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = DEPTH_NEAR + frac * (DEPTH_FAR - DEPTH_NEAR)
        x = min(max(int(frac * (width - 1)) - 12, 2), width - 38)
        cv2.putText(bar, f"{d:.1f}m", (x, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def _show_view(raw_full_frames, pooled_frames, scale):
    """OpenCV 调试窗口：显示原始全分辨率深度图（用于观测噪点与伪影）与输入策略的池化深度图对比。"""
    import cv2

    sep = np.full((DEPTH_H * scale, 2, 3), 255, dtype=np.uint8)
    rows = []
    for raw, pooled in zip(raw_full_frames, pooled_frames):
        panels = []
        if raw is not None:
            raw_img = np.where(np.isfinite(raw) & (raw > 0.0), raw, DEPTH_FAR).astype(np.float32)
            raw_img = np.clip(raw_img, DEPTH_NEAR, DEPTH_FAR)
            panels += [_depth_panel(raw_img, scale, "原始深度 (RAW)", cv2, cv2.INTER_AREA), sep]
        panels.append(
            _depth_panel(pooled.reshape(DEPTH_H, DEPTH_W), scale, "池化->策略输入",
                         cv2, cv2.INTER_NEAREST)
        )
        rows.append(np.hstack(panels))
    grid = np.vstack(rows)
    out = np.vstack([grid, _colorbar(grid.shape[1], cv2)])
    cv2.imshow("camera_node 深度监控 (原始 | 池化策略输入)", out)
    cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description="ZED 深度采集与 UDP 发送节点")
    parser.add_argument("--hz", type=float, default=50.0,
                        help="UDP 发送频率 (Hz)，默认 50 Hz 匹配策略控制周期")
    parser.add_argument("--camera-fps", type=int, default=60,
                        help="ZED 相机硬件采集帧率 (VGA 支持 15/30/60/100，建议 >= --hz)")
    parser.add_argument("--depth-mode", default="PERFORMANCE",
                        help="ZED 深度模式: PERFORMANCE (默认最快) / QUALITY / NEURAL / ULTRA")
    parser.add_argument("--pool-percentile", type=float, default=3.0,
                        help="降采样软最小值百分位 (默认 3.0%%)")
    parser.add_argument("--zed-confidence", type=int, default=50,
                        help="ZED 深度置信度阈值 (0-100，越小越严格，SDK 默认 95 太宽松，设为 50 剔除飞点)")
    parser.add_argument("--zed-texture-conf", type=int, default=100,
                        help="ZED 纹理置信度阈值 (0-100，越小越严格，剔除白墙/空旷区域弱纹理噪点)")
    parser.add_argument("--fill-holes", action="store_true",
                        help="是否开启 ZED FILL 填洞模式 (默认关闭，避免在盲区产生虚假近处障碍物)")
    parser.add_argument("--edge-crop", type=int, default=30,
                        help="外圈裁剪黑边像素数 (默认 30，消除双目视差边缘伪影)")
    parser.add_argument(
        "--serials", default="",
        help="相机序列号列表，以逗号分隔（多相机时使用）。为空则打开默认单台相机",
    )
    parser.add_argument("--policy-ip", default=UDP_HOST, help="策略节点 IP 地址")
    parser.add_argument("--dry-run", action="store_true",
                        help="虚拟运行模式：不打开物理相机，持续发送全远平面虚拟深度图用于全链路网络测试")
    parser.add_argument("--obs-far", action="store_true",
                        help="诊断模式：物理相机正常运行，但对外强制发布全远平面深度图（使策略保持平静站立）")
    parser.add_argument("--view", action="store_true",
                        help="是否开启 OpenCV 图像监控窗口（需要显示环境支持）")
    parser.add_argument("--view-scale", type=int, default=20, help="监控窗口缩放比例")
    args = parser.parse_args()

    # 信号处理器配置：确保收到 SIGINT (Ctrl+C) 或 SIGTERM 时能安全释放相机硬件句柄
    _shutting_down = [False]

    def _on_signal(signum, frame):
        if _shutting_down[0]:
            return
        _shutting_down[0] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    serials = [int(s) for s in args.serials.split(",") if s.strip()]
    n_cam = max(1, len(serials)) if not args.dry_run else max(1, len(serials) or 1)

    # 卡死重连超时时长：若超过 2.0 秒未收到新帧，自动触发重新打开相机
    STALE_RESET_S = 2.0

    def make_source(sn):
        from deploy.common.zed_depth_source import ZedDepthSource
        return ZedDepthSource(
            height=DEPTH_H, width=DEPTH_W, depth_mode=args.depth_mode,
            fps=args.camera_fps,
            min_depth_m=DEPTH_NEAR, max_depth_m=DEPTH_FAR,
            pool_percentile=args.pool_percentile, fill_invalid=DEPTH_FAR,
            fill_holes=args.fill_holes,
            confidence_threshold=args.zed_confidence,
            texture_confidence_threshold=args.zed_texture_conf,
            edge_crop_px=args.edge_crop,
            serial_number=sn,
            keep_full=bool(args.view),
        )

    serial_list = serials or [None]
    sources = []
    if not args.dry_run:
        for sn in serial_list:
            sources.append(make_source(sn))
        n_cam = len(sources)
        print(f"[camera_node] 成功打开 {n_cam} 台 ZED 相机")
    else:
        print(f"[camera_node] 虚拟运行模式 (DRY RUN): 模拟发送 {n_cam} 台相机的全远平面深度帧")

    # 创建发送 UDP 套接字
    sock = create_udp_socket(UDP_HOST, 0)
    policy_addr = (args.policy_ip, UDP_DEPTH_PORT)
    print(f"[camera_node] 深度数据发布目标: {args.policy_ip}:{UDP_DEPTH_PORT} @ {args.hz} Hz")

    far_frame = np.full(DEPTH_H * DEPTH_W, DEPTH_FAR, dtype=np.float32)
    far_depth = np.full(DEPTH_H * DEPTH_W * n_cam, DEPTH_FAR, dtype=np.float32)
    if args.obs_far:
        print("[camera_node] --obs-far 模式激活: 物理相机正常运行，但下发全远平面深度图")
    dt = 1.0 / float(args.hz)
    seq = 0
    # 相机健康状态与最后有效帧缓存（在相机重连期间维系数据发送）
    last_good = [far_frame.copy() for _ in range(n_cam)]
    last_count = [s.grab_count for s in sources]
    last_progress = [time.monotonic() for _ in sources]
    next_tick = time.perf_counter()
    # 统计日志记录
    t_log = time.perf_counter()
    seq_at_log = 0
    grabs_at_log = sum(s.grab_count for s in sources)
    try:
        while True:
            now = time.perf_counter()
            sleep_t = next_tick - now
            if sleep_t > 0:
                time.sleep(sleep_t)
            next_tick += dt

            if args.dry_run:
                frames = [far_frame for _ in range(n_cam)]
            else:
                now_m = time.monotonic()
                frames = []
                for i, s in enumerate(sources):
                    gc = s.grab_count
                    if gc != last_count[i]:
                        # 接收到新帧：更新计数并重置超时计时器
                        last_count[i] = gc
                        last_progress[i] = now_m
                        try:
                            last_good[i] = s.latest_frame().reshape(-1).astype(np.float32)
                        except RuntimeError:
                            pass
                    elif now_m - last_progress[i] > STALE_RESET_S:
                        # 抓取线程卡死超时 -> 自动重启相机自愈
                        print(f"[camera_node] 相机 {i} 卡死停顿 (超过 {STALE_RESET_S:.0f} 秒无新帧)；正在尝试重新打开...",
                              flush=True)
                        try:
                            s.close()
                        except Exception:
                            pass
                        try:
                            sources[i] = make_source(serial_list[i])
                            last_count[i] = sources[i].grab_count
                            print(f"[camera_node] 相机 {i} 成功重新打开恢复。", flush=True)
                        except Exception as exc:
                            print(f"[camera_node] 相机 {i} 重新打开失败: {exc}；将在后续循环重试。",
                                  flush=True)
                        last_progress[i] = time.monotonic()
                    frames.append(last_good[i])

            real_depth = np.concatenate(frames).astype(np.float32)
            depth = far_depth if args.obs_far else real_depth
            sock.sendto(pack_depth(seq, time.time(), n_cam, depth), policy_addr)
            seq += 1

            if args.view:
                raw_full = ([sources[i].latest_full()[0] for i in range(n_cam)]
                            if sources else [None] * n_cam)
                _show_view(raw_full, frames, args.view_scale)

            # 周期性日志打印实际发布频率与相机抓取帧率
            now = time.perf_counter()
            if now - t_log >= 3.0:
                pub_fps = (seq - seq_at_log) / (now - t_log)
                if sources:
                    grabs = sum(s.grab_count for s in sources)
                    grab_fps = max(0.0, grabs - grabs_at_log) / (now - t_log) / max(1, len(sources))
                    print(f"[camera_node] 发布频率: {pub_fps:.0f} Hz | ZED抓取帧率: {grab_fps:.0f} FPS "
                          f"({args.depth_mode} 模式, 设定目标 {args.camera_fps} FPS)", flush=True)
                    grabs_at_log = grabs
                else:
                    print(f"[camera_node] 发布频率: {pub_fps:.0f} Hz (虚拟运行模式)", flush=True)
                t_log, seq_at_log = now, seq
    except KeyboardInterrupt:
        pass
    finally:
        for s in sources:
            s.close()
        sock.close()
        print("camera_node 已安全停止。")


if __name__ == "__main__":
    main()
