"""36 列动作捕捉 CSV 数据批量前向运动学可视化与视频导出工具 (Visualize Motion CSVs)

本脚本用于直接读取 `motion_data_csv` 目录下的 36 列 AMP 格式动作捕捉 CSV 片段，在 Unitree G1 人形机器人上
执行纯前向运动学姿态回放，并在画面左上角自动绘制片段名称水印标签，最终拼接导出为单份高清 MP4 视频或在本地窗口中实时播放。
无需预先将 CSV 转换为 NPZ 格式即可快速浏览检查原始动作数据质量。

1. 核心流程与数据加载 (`_load_clip_csv`):
   - 数据解析: 直接加载无表头 36 列数据 (`x, y, z` 平移(米), `qx, qy, qz, qw` 四元数, 29 个关节角(弧度))。
   - 帧率降采样: 按 `round(src_fps / target_fps)` 步长对动作序列进行降采样，以匹配目标视频帧率。
   - 姿态转换: 将根节点四元数从 CSV 的 xyzw 格式重排拼接为 MuJoCo 所要求的 wxyz 格式。
   - 前向运动学渲染: 由于纯前向运动学仅依赖 qpos (基座与关节位置)，速度项自动填充零，随后送入仿真渲染器。
   - 画面合成与导出: 复用 `visualize_leap_motions.py` 中的画面标注器与视频写出器，按名称顺序无缝衔接各动作片段。

2. 命令行参数解析机制 (tyro CLI):
   - `--motion-dir`: 包含待回放 36 列 CSV 片段的目录路径（默认 `motion_data_csv/amp_dodge`）。
   - `--video-output`: 离屏渲染输出的 MP4 视频文件路径（默认 `amp_dodge_clips.mp4`）。
   - `--src-fps`: 原始 CSV 文件的采样率（例如 SEED 动作库数据通常为 120.0 fps）。
   - `--target-fps`: 输出视频与重采样后的目标帧率（默认 30.0 fps）。
   - `--realtime-scale`: 视频回放速度缩放比例（1.0 表示原速，0.5 表示 2 倍慢动作）。
   - `--render-backend`: 渲染后端，可选 `offscreen` (离屏录制 MP4 视频) 或 `window` (原生窗口实时交互查看)。
   - `--device`: 仿真与张量计算设备（默认 `cuda:0`）。

使用示例:
    uv run python scripts/visualize_motion_csvs.py \\
        --motion-dir motion_data_csv/amp_dodge --video-output amp_dodge_clips.mp4
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# 复用 leap viz 模块中的渲染器（文本水印绘制、视频帧编码、离屏/窗口主循环）
from visualize_leap_motions import _run_offscreen, _run_window  # noqa: E402


def _load_clip_csv(path: Path, device: str, src_fps: float, target_fps: float) -> dict[str, Any]:
    """Load a 36-col AMP CSV, subsample to target_fps, into the leap-viz clip dict.

    CSV quaternion is XYZW; the sim root state wants WXYZ. Velocities aren't in the CSV
    (and don't affect the rendered pose -- it's pure forward-kinematics from qpos), so they
    are zero-filled.
    """
    a = np.loadtxt(path, delimiter=",")
    if a.ndim == 1:
        a = a[None, :]
    stride = max(1, round(src_fps / target_fps))
    a = a[::stride]
    T = a.shape[0]
    q_xyzw = a[:, 3:7]
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, 0:3]], axis=1)
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
    z = lambda *s: torch.zeros(s, dtype=torch.float32, device=device)  # noqa: E731
    return {
        "name": path.stem,
        "fps": src_fps / stride,
        "num_frames": T,
        "root_pos": t(a[:, 0:3]),
        "root_quat": t(q_wxyz),
        "root_lin_vel": z(T, 3),
        "root_ang_vel": z(T, 3),
        "joint_pos": t(a[:, 7:36]),
        "joint_vel": z(T, 29),
    }


def main(
    motion_dir: str = "motion_data_csv/amp_dodge",
    video_output: str = "amp_dodge_clips.mp4",
    src_fps: float = 120.0,
    target_fps: float = 30.0,
    realtime_scale: float = 1.0,
    render_backend: str = "offscreen",
    device: str = "cuda:0",
) -> None:
    """Replay every *.csv under motion_dir on the G1, labeled with the clip name.

    Args:
      motion_dir: dir of 36-col AMP CSVs (sorted by name; alternates land adjacent).
      video_output: output mp4 (offscreen backend).
      src_fps: source CSV rate (seed-derived clips are 120 fps).
      target_fps: subsample to this for a standard, viewable video.
      realtime_scale: 1.0 = real time; 0.5 = half speed.
      render_backend: "offscreen" (mp4) or "window" (live viewer).
    """
    paths = sorted(Path(motion_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No .csv clips under {motion_dir}")
    print(f"Found {len(paths)} CSV clip(s) under {motion_dir}")

    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot: Entity = scene["robot"]
    scene.reset()

    clips = [_load_clip_csv(p, device, src_fps, target_fps) for p in paths]
    total = sum(c["num_frames"] for c in clips)
    print(f"{total} frames @ {target_fps:.0f} fps -> ~{total/target_fps:.0f}s video")

    if render_backend == "offscreen":
        _run_offscreen(sim, scene, robot, clips, video_output, realtime_scale)
    else:
        _run_window(sim, scene, robot, clips, False, realtime_scale, 0.5)


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)
