#!/usr/bin/env python3
"""基于 ZED Mini 与 RealtimeEfficientTAM 的实时球体视觉跟踪与深度掩码调试工具。

在部署架构中的角色：
-------------------
本脚本用于在真实硬件或开发机上独立调试与验证 EfficientTAM 目标分割管道：
1. 运行环境：
   - 运行于 Python 3.8 部署虚拟环境（PyTorch 2.4.1）；
   - 在执行过 ``deploy/tools/patch_efficienttam_py38.py`` 补丁后，RealtimeEfficientTAM
     无需迁移到 Python 3.10 或依赖复杂的 TensorRT，直接在 PyTorch 下实时运行；
2. 跟踪与时序记忆：
   - 捕获 ZED 左目 RGB 彩色视频流送入 EfficientTAM；
   - 用户在画面中**鼠标单击球体**即可初始化跟踪序列；
   - EfficientTAM 利用其内置的记忆注意力机制（Memory Attention）实现跨帧连续跟踪，
     相比传统的逐帧检测或中心点重提示，其在球体快速移动、短暂被遮挡或飞出视场后重新飞入时，
     具有卓越的重捕获能力；
3. 纯净深度可视化 (`--depth` 参数)：
   - 同步提取原始深度图，结合当前分割掩码执行 ``mask_depth_to_far`` 与百分位数池化，
     在屏幕右侧实时展示策略网络实际看到的 ``[9, 16]`` 伪彩深度图，直观检验背景剔除效果。

交互快捷键：
-----------
- 鼠标左键单击：在当前帧以点击位置为正样本提示点，初始化或重设跟踪目标；
- 'r' 或 'c': 重置跟踪器状态；
- 'q': 退出程序。
"""

import argparse
import contextlib
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RT = os.path.join(_REPO_ROOT, "deploy", "common", "RealtimeEfficientTAM")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _RT)  # 注入 efficient_track_anything 命名空间及 Hydra 配置搜索路径

from deploy.common.g1_deploy_constants import DEPTH_FAR, DEPTH_H, DEPTH_NEAR, DEPTH_W
from deploy.common.zed_depth_source import ZedDepthSource, mask_depth_to_far, percentile_pool_depth

WINDOW = "EfficientTAM 实时跟踪 (鼠标单击=启动追踪, r=重置, q=退出)"


def overlay_mask(bgr, mask, color=(0, 0, 255), alpha=0.5):
    """将形状为 (H, W) 的布尔掩码以半透明颜色（默认红色）叠加渲染在 BGR 彩色图像上。"""
    out = bgr.copy()
    sel = np.asarray(mask, dtype=bool)
    out[sel] = ((1.0 - alpha) * out[sel] + alpha * np.array(color, dtype=np.float32)).astype(np.uint8)
    return out


# 几何防误检阈值定义
MASK_MAX_HEIGHT_FRAC = 0.5  # 掩码高度占画面垂直比例绝对上限（超过 50% 必非球体，直接剔除）
BALL_CAM_VFOV_DEG = 54.0    # 头部深度相机垂直视场角 FOV (54度，匹配 ZED Mini VGA 模式)
BALL_DIAM_MAX_M = 0.65      # 躲避球物理直径绝对上限（米，包含分割溢出容差；放宽至 0.65m 避免过度敏感拒识）


def mask_too_tall(mask, max_height_frac=MASK_MAX_HEIGHT_FRAC, dist=None,
                  vfov_deg=BALL_CAM_VFOV_DEG, ball_diam_max=BALL_DIAM_MAX_M):
    """几何高度综合判定：若掩码包围盒垂直高度过大，则判定其不是球体（很可能是误抓取的人体或墙体），
    返回 True，从而指示上层感知管道放弃该掩码并输出全远平面深度，防止策略被误导。

    双重几何过滤机制（取两者中最严格的阈值）：
    -------------------------------------------
    1. 绝对画幅比例上限：
       若掩码高度超过图像总高度的 ``max_height_frac`` (默认 0.5)，直接判定为非球体（近处的墙体或全身人体）；
    2. 距离自适应物理尺寸检验 (Distance-Aware Physical Size Gate):
       在测距为 ``dist`` (单位：米) 的距离下，直径不超过 ``ball_diam_max`` 的真实球体在相机垂直方向
       所占画幅的最大理论高度比例为：
           cap_ratio = ball_diam_max / (2 * dist * tan(vfov / 2))
       若当前掩码的高度比例超过了该距离所对应的物理几何上限，说明该物体绝不可能是球体！
       （例如：远处站着一个人，虽然因距离远在画面中像素占比不大，未触发绝对比例上限，
       但相对于其测量的数米距离而言，其实际物理尺寸远大于球体，因此被该自适应门控精准剔除）。
    """
    m = np.asarray(mask, dtype=bool)
    rows = np.nonzero(m.any(axis=1))[0]
    if rows.size == 0:
        return False
    frac = (rows[-1] - rows[0] + 1) / m.shape[0]
    cap = max_height_frac
    if dist is not None and np.isfinite(dist) and dist > 0.0:
        dist_cap = ball_diam_max / (2.0 * float(dist) * math.tan(math.radians(vfov_deg) * 0.5))
        cap = min(cap, dist_cap)
    return frac > cap


def depth_panel(pooled, out_w, out_h, label):
    """将 [DEPTH_H, DEPTH_W] (9x16) 的池化深度图以伪彩色 (JET, 红近蓝远) 渲染在指定的窗口区域内。
    采用居中加黑边 (Letterbox) 保持 16:9 原始画幅比例，防止画面被上下拉伸变形。"""
    import cv2

    norm = np.clip((pooled - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR), 0.0, 1.0)
    u8 = (255 - norm * 255).astype(np.uint8)  # 红色=近处
    h, w = u8.shape[:2]  # 获取池化后的深度网格分辨率 (高 H=9, 宽 W=16) (9, 16)
    scale = min(out_w / w, out_h / h)
    fit_w, fit_h = max(1, int(w * scale)), max(1, int(h * scale))
    big = cv2.resize(u8, (fit_w, fit_h), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    y0, x0 = (out_h - fit_h) // 2, (out_w - fit_w) // 2
    canvas[y0:y0 + fit_h, x0:x0 + fit_w] = color
    cv2.putText(canvas, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


@contextlib.contextmanager
def _quiet():
    """重定向标准输出以静音 RealtimeEfficientTAM 的逐帧控制台打印。"""
    with open(os.devnull, "w") as fn, contextlib.redirect_stdout(fn):
        yield


def _mask_hw(mask_logits, h, w):
    """将 EfficientTAM 的 logits 预测张量提取为 (h, w) 布尔掩码数组。"""
    import cv2

    m = (mask_logits[0, 0] > 0.0).detach().cpu().numpy()
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return m


def main():
    parser = argparse.ArgumentParser(description="ZED + RealtimeEfficientTAM 实时球体跟踪交互工具")
    parser.add_argument("--checkpoint", default=os.path.join(_RT, "checkpoints", "efficienttam_s.pt"),
                        help="EfficientTAM 模型权重路径")
    parser.add_argument("--config", default="configs/efficienttam/efficienttam_s.yaml",
                        help="Hydra 配置文件名")
    parser.add_argument("--serial", type=int, default=None, help="ZED 相机硬件序列号")
    parser.add_argument("--max-depth-m", type=float, default=5.0, help="最大测距距离 (米)")
    parser.add_argument("--conf", type=float, default=0.5, help="追踪置信度阈值")
    parser.add_argument("--tiny", action="store_true",
                        help="使用 efficienttam_ti_512x512 轻量模型 (提速约 2 倍)")
    parser.add_argument("--depth", action="store_true",
                        help="在右侧并排显示经掩码过滤并池化到 [9, 16] 的策略输入深度图")
    args = parser.parse_args()
    if args.tiny:
        args.checkpoint = os.path.join(_RT, "checkpoints", "efficienttam_ti_512x512.pt")
        args.config = "configs/efficienttam/efficienttam_ti_512x512.yaml"

    import cv2
    import torch
    from efficient_track_anything.realtime_tam import build_predictor, start, track

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"模型权重未找到: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = build_predictor(args.config, args.checkpoint, device=device)
    src = ZedDepthSource(keep_full=True, serial_number=args.serial, max_depth_m=args.max_depth_m)

    pending = {"pt": None}  # 记录鼠标点击坐标

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["pt"] = (int(x), int(y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    tracking = False
    try:
        while True:
            _depth, bgr = src.latest_full()
            if bgr is None:
                time.sleep(0.005)
                continue
            h, w = bgr.shape[:2]
            disp, m, dt = bgr, None, None

            if pending["pt"] is not None:
                x, y = pending["pt"]
                pending["pt"] = None
                state.initialized = False
                with _quiet():
                    state.predictor.load_first_frame(bgr)
                    t0 = time.perf_counter()
                    _ids, ml = start(state, points=np.array([[x, y]], np.float32),
                                     labels=np.array([1], np.int32), obj_id=0)
                dt = (time.perf_counter() - t0) * 1e3
                m = _mask_hw(ml, h, w)
                tracking = True
            elif tracking:
                with _quiet():
                    t0 = time.perf_counter()
                    _ids, ml = track(state, bgr, confidence_threshold=args.conf)
                dt = (time.perf_counter() - t0) * 1e3
                m = _mask_hw(ml, h, w)

            if m is not None and m.any():
                disp = overlay_mask(bgr, m)
                ys, xs = np.nonzero(m)
                cv2.circle(disp, (int(np.median(xs)), int(np.median(ys))), 5, (0, 255, 0), -1)
                label = f"追踪耗时 {dt:.0f}ms  ({int(m.sum())} 像素)"
            elif tracking:
                label = f"追踪耗时 {dt:.0f}ms  (无有效掩码)" if dt is not None else "正在跟踪..."
            else:
                label = "请点击画面中的球体启动跟踪"

            cv2.putText(disp, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if args.depth and _depth is not None:
                # 掩码深度图生成与池化
                _mpx = np.asarray(_depth, dtype=np.float32)[m] if (m is not None and bool(m.any())) else None
                _mpx = _mpx[np.isfinite(_mpx)] if _mpx is not None else None
                _mdist = float(np.median(_mpx)) if (_mpx is not None and _mpx.size) else None
                too_tall = m is not None and bool(m.any()) and mask_too_tall(m, dist=_mdist)
                have_mask = m is not None and bool(m.any()) and not too_tall
                sel = m if have_mask else np.zeros(np.asarray(_depth).shape[:2], dtype=bool)
                md = mask_depth_to_far(_depth, sel, DEPTH_FAR)
                dlabel = ("策略深度输入 (已掩码)" if have_mask else
                          "策略深度输入 (尺寸过大-全远)" if too_tall else
                          "策略深度输入 (未跟踪-全远)")
                pooled = percentile_pool_depth(md, DEPTH_H, DEPTH_W, percentile=3.0, fill_invalid=DEPTH_FAR)
                disp = np.hstack([disp, depth_panel(pooled, disp.shape[1] // 2, disp.shape[0], dlabel)])

            cv2.imshow(WINDOW, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k in (ord("r"), ord("c")):
                state.initialized = False
                tracking = False
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
