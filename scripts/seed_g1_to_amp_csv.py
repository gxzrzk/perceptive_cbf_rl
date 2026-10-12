"""NVIDIA SEED G1 动作数据至 AMP CSV 格式转换工具 (SEED to AMP CSV Converter)

本脚本用于将 NVIDIA SEED 重定向的人形机器人 G1 动作捕捉 CSV 数据转换为当前代码库 AMP (Adversarial Motion Priors)
强化学习训练管道所要求的 36 列标准 CSV 格式。

1. 数据格式对比与转换映射 (Data Format Specification):
   - 输入数据格式 (SEED G1 CSV, 通常位于 `seed/g1/csv/<date>/<clip>.csv`):
     * 含有第 1 行表头说明，每行包含 36 列数据：
       - 列 0: 帧序号 (`Frame`)
       - 列 1-3: 根节点世界平移坐标 (`root_translateX/Y/Z`)，单位为厘米 (cm)
       - 列 4-6: 根节点姿态外旋欧拉角 (`root_rotateX/Y/Z`)，'xyz' 外旋顺序，单位为角度 (degrees)
       - 列 7-35: 29 个关节自由度角度 (`*_dof`)，单位为角度 (degrees)
   - 目标数据格式 (AMP 36 列 CSV, 供 `scripts/csv_to_npz.py` 与 `MotionLoader` 加载):
     * 无表头 (Headerless)，每行包含 36 列双精度浮点数：
       - 列 0-2: 根节点平移 `x, y, z`，单位为米 (m，由 cm / 100 换算)
       - 列 3-6: 根节点旋转四元数 `qx, qy, qz, qw` (xyzw 规范，由欧拉角转换生成)
       - 列 7-35: 29 个关节角弧度 (radians，由 degrees 转换为弧度)
   - 关节顺序映射：
     SEED 文件中的关节排列顺序与 `csv_to_npz.py` 内部固定的 29 个 G1 关节顺序完全一致，转换时无需调整列顺序。
     经数值验证，转换精度与代码库现存 `motion_data_csv/amp/*.csv` 完全一致（四元数 L1 误差为 0，关节误差 < 1e-6）。

2. 命令行参数解析机制 (tyro CLI):
   - `--output-dir`: 转换后 AMP CSV 文件的输出目录路径。
   - `--input-file`: 单个输入的 SEED G1 CSV 文件绝对或相对路径。
   - `--manifest`: 文本清单文件路径（每行包含一个 SEED CSV 路径，支持以 '#' 开头的注释行）。
     支持在路径后附加空格分隔的截取帧区间 `start:end`（1-based 闭区间，如 `/path/clip.csv 1:246` 用于裁剪尾部失稳晃动）。

使用示例:
    # 单文件转换模式:
    python scripts/seed_g1_to_amp_csv.py \\
        --input-file seed/g1/csv/230509/broad_jump_002__A359.csv \\
        --output-dir motion_data_csv/amp_leap

    # 清单批量转换模式:
    python scripts/seed_g1_to_amp_csv.py \\
        --manifest motion_data_csv/amp_leap_manifest.txt \\
        --output-dir motion_data_csv/amp_leap
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

# SEED G1 CSV 数据列布局（去除首行表头后）(SEED g1 csv layout after the header row):
_FRAME_COL = 0             # 帧序号列 (Frame number)
_TRANS_COLS = slice(1, 4)  # 根节点世界坐标平移 (root_translateX/Y/Z)，单位：厘米 (cm)
_ROT_COLS = slice(4, 7)    # 根节点姿态外旋欧拉角 (root_rotateX/Y/Z)，'xyz' 顺序，单位：度 (degrees)
_DOF_COLS = slice(7, 36)   # 29 个关节自由度角度 (joint angles)，单位：度 (degrees)
_N_EXPECTED_COLS = 36      # 预期总列数：1 帧号 + 3 平移 + 3 旋转 + 29 关节 = 36 列


def convert_array(seed: np.ndarray) -> np.ndarray:
  """将单个 SEED G1 运动数据数组 (T, 36) 转换为 AMP 标准格式 (T, 36)。
  (Convert one SEED g1 motion array (T, 36) to AMP format (T, 36).)

  转换细节 (Conversion Details):
  - 坐标单位换算: 平移列除以 100.0，从厘米 (cm) 转换为国际标准单位米 (m)。
  - 姿态表达转换: 通过 SciPy 将 'xyz' 外旋欧拉角转换为四元数 [qx, qy, qz, qw] (xyzw 格式)。
  - 角度单位换算: 将 29 个关节角度从角度 (degrees) 转换为弧度 (radians)。
  - 格式拼接: 拼接为 [3维平移(m), 4维四元数(xyzw), 29维关节角(rad)] 的标准 36 列 float64 矩阵。
  """
  if seed.ndim != 2 or seed.shape[1] != _N_EXPECTED_COLS:
    raise ValueError(
      f"Expected SEED array of shape (T, {_N_EXPECTED_COLS}), got {seed.shape}"
    )
  pos_m = seed[:, _TRANS_COLS] / 100.0  # 厘米转换为米 (cm -> m)
  quat_xyzw = R.from_euler("xyz", seed[:, _ROT_COLS], degrees=True).as_quat()
  dof_rad = np.deg2rad(seed[:, _DOF_COLS])
  return np.concatenate([pos_m, quat_xyzw, dof_rad], axis=1).astype(np.float64)


def convert_file(
  input_file: Path,
  output_dir: Path,
  frame_range: tuple[int, int] | None = None,
) -> Path:
  """转换单个 SEED G1 CSV 文件并保存为标准 36 列 AMP CSV 格式。
  
  参数:
    input_file: 源 SEED CSV 文件的路径。
    output_dir: 输出目录路径。
    frame_range: 可选的裁剪帧区间元组 (start, end)，基于 1-indexed 包含两端的闭区间。
  """
  seed = np.genfromtxt(str(input_file), delimiter=",", skip_header=1)
  total = seed.shape[0]
  if frame_range is not None:
    start, end = frame_range  # 基于 1 的双闭区间索引 [start, end] (1-indexed, inclusive, over data frames)
    seed = seed[start - 1 : end]
  amp = convert_array(seed)
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / input_file.name
  # 保存为浮点数保留 6 位小数，与代码库中既有的 AMP CSV 精度规范一致 (6 decimals matches the repo's existing amp csv precision)
  np.savetxt(str(out_path), amp, delimiter=",", fmt="%.6f")
  range_note = f" (kept {frame_range[0]}:{frame_range[1]} of {total})" if frame_range else ""
  print(f"  {input_file.name}: {seed.shape[0]} frames{range_note} -> {out_path}")
  return out_path


def main(
  output_dir: str,
  input_file: str | None = None,
  manifest: str | None = None,
) -> None:
  """将 SEED G1 格式的动作 CSV 转换为 AMP 格式的 CSV。
  (Convert SEED g1 CSV(s) to AMP CSV format.)

  参数说明 (Args):
    output_dir: 转换后 AMP CSV 文件的输出目录路径 (Directory to write converted AMP csvs into)。
    input_file: 单个待转换的 SEED G1 CSV 文件路径 (Path to a single SEED g1 csv)。
    manifest: 文本清单文件路径，内部列出若干 SEED G1 CSV 文件路径（每行一个）(Path to a text file listing SEED g1 csv paths)。
  """
  if input_file is None and manifest is None:
    raise ValueError("Provide either --input-file or --manifest.")

  out_dir = Path(output_dir)
  # 每个待处理条目记录为：(文件路径, 可选的 (start, end) 帧范围，1-indexed 包含边界) [原英文: Each entry: path, optional frame range]
  entries: list[tuple[Path, tuple[int, int] | None]] = []
  if input_file is not None:
    entries.append((Path(input_file), None))
  if manifest is not None:
    for line in Path(manifest).read_text().splitlines():
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      # 行尾可选带有由空格分隔的 "start:end" 帧裁剪区间。
      # 例如 "/path/clip.csv  1:246" 仅保留第 1 至 246 帧（可剔除采集末尾站立不稳/摔倒的尾部数据，对应原英文 stumbly tail 截断）。
      parts = line.split()
      path = Path(parts[0])
      frame_range: tuple[int, int] | None = None
      if len(parts) > 1 and ":" in parts[1]:
        a, b = parts[1].split(":")
        frame_range = (int(a), int(b))
      entries.append((path, frame_range))

  print(f"Converting {len(entries)} SEED g1 clip(s) -> {out_dir}")
  missing = [p for p, _ in entries if not p.exists()]
  if missing:
    raise FileNotFoundError(
      "Missing SEED csv files:\n" + "\n".join(f"  {p}" for p in missing)
    )
  for path, frame_range in entries:
    convert_file(path, out_dir, frame_range)
  print(f"Done. {len(entries)} clip(s) written to {out_dir}")


if __name__ == "__main__":
  tyro.cli(main)
