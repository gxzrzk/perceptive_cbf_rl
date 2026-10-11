"""用于实机/硬件在环 (HIL) 躲避任务的实时 ZED Mini 深度感知源模块 (Live ZED Mini depth source for HIL dodge play)。

【系统定位与模块作用】
本模块负责驱动连接在机器人头部的 Stereolabs ZED Mini 双目深度相机，在独立的后台守护线程中
实时抓取原始公制度量深度图 (DEPTH measure, 单位: 米)，并通过创新的“软最小值分位数池化”
(Soft-min / Low-percentile Pooling) 算法，将高分辨率深度图下采样为与仿真环境头部相机
(`head_camera_single`, 默认 16x9 极低分辨率) 完全对齐的深度张量，实现 Sim-to-Real 的直接零样本迁移。

【核心设计与关键算法细节】
1. **为什么必须采用软最小值（低分位数）池化，而坚决不能使用平均池化或双线性插值？**
   - **平均池化/双线性插值的致命缺陷**: 快速迎面飞来的躲避球是一个较小的前景物体，在 2~5 米距离时
     投影在相机图像中仅占据对应网格块 (block) 约 5%~18% 的像素。若采用平均池化，球体的近距离深度
     会被占多数的远景/地面深度剧烈平滑混叠，导致球体信号被完全抹除，并凭空捏造出一个仿真中从未出现过的
     “中间假距离”曲面。
   - **严格最小值的局限**: 仿真中的光线投射相机对每个像素采样的是该视线方向最近击中物体的深度，
     但真实双目相机在视差匹配中不可避免会产生少量离群噪点（如飞点、高反光噪点）。严格最小值 (min, percentile=0)
     会导致单个 0.2m 的噪点将整个 16x9 网格单元全部拉近变暗，产生严重的假阳性避障误动作。
   - **软最小值 (Low-Percentile, 如 2%~5% 分位数)**:
     在各个池化网格块内仅对有效像素计算低分位数。既能稳健捕捉占据 5% 以上像素的微小球体，
     又能天然剔除 1~2 个孤立的噪点飞点，完美贴合仿真光线采样的物理统计特性。
2. **无效深度 (Invalid Pixels) 的规范化处理**:
   - ZED 相机的无效码包括 NaN、+/-Inf 以及 <=0（超出最近/最远盲区、视差未匹配空洞）。
   - 算法将无效像素转为 NaN，使其在分位数统计中自动被排除；
   - 若整个网格块内全部为无效像素，则填充为 `fill_invalid=far`（远距离极限值），
     与策略端 `DepthImageObs` 预处理管道保持完全一致的归一化映射 (far -> 1.0)。
3. **双目视差边缘裁切与置信度过滤**:
   - 双目相机由于基线视差盲区，视野最外边缘往往存在伪影，通过 `edge_crop_px` 在池化前将边缘置为 NaN；
   - 关闭 ZED SDK 的空洞填充模式 (`enable_fill_mode=False`)，杜绝 SDK 在低纹理空白区域盲目插值出假近景障碍。
4. **异步零等待架构 (Non-blocking Daemon Thread)**:
   - 图像抓取与池化在后台守护线程以相机自身帧率运行，主控循环通过 `latest_frame()` 零等待读取最新缓存帧，
     彻底杜绝相机 I/O 阻塞强化学习控制主循环。
"""

from __future__ import annotations

import threading
import time
import warnings

import numpy as np
import torch


def percentile_pool_depth(
  depth_full: np.ndarray,
  height: int,
  width: int,
  percentile: float = 0.0,
  fill_invalid: float = 0.0,
) -> np.ndarray:
  """通过局部网格有效像素的低分位数（软最小值）将全分辨率深度图下采样至 target_height x target_width。

  算法原理：
  1. 将输入全分辨率深度图中的非有限值（NaN, +/-Inf）以及非正值（<=0，相机近/远盲区或未匹配点）
     标记为无效，并统一替换为 `np.nan`；
  2. 对深度图进行中心裁切 (Center Crop)，使其尺寸能够被目标网格 `height x width` 整除；
  3. 将每个网格块 (block) 内的像素展平成一维向量；
  4. 沿块内像素维度利用 `np.nanpercentile` 计算指定分位数深度（忽略 NaN）；
  5. 若某个网格块内全部为无效像素（结果为 NaN），则使用 `fill_invalid`（通常设为 far 远景值）填充。

  参数:
      depth_full: 输入全分辨率 2D 深度图数组 (shape: [H0, W0]，单位: 米)。
      height: 下采样后的目标高度（如 9）。
      width: 下采样后的目标宽度（如 16）。
      percentile: 各块有效像素的分位数百分比，范围 [0.0, 100.0]；
                  0.0 对应严格最小值 (strict min)；建议设为 2.0~5.0 用于滤除孤立噪点。
      fill_invalid: 当整块均为无效点时的替补深度值（单位: 米），默认 0.0（或传入 far 距离）。

  返回:
      np.ndarray: 下采样后的低分辨率深度图，shape 为 [height, width]，类型为 float32。
  """
  if depth_full.ndim != 2:
    raise ValueError(f"expected a 2-D depth map, got shape {depth_full.shape} (期望二维深度图，收到形状: {depth_full.shape})")
  if not 0.0 <= percentile <= 100.0:
    raise ValueError(f"percentile must be in [0, 100], got {percentile} (分位数必须在 [0, 100] 范围内)")
  h0, w0 = depth_full.shape
  if h0 < height or w0 < width:
    raise ValueError(f"source {h0}x{w0} smaller than target {height}x{width} (输入尺寸小于目标尺寸)")

  # 1. 过滤无效像素：NaN、Inf、非正数值 -> 替换为 np.nan，使 np.nanpercentile 能直接跳过这些无效点
  invalid = ~np.isfinite(depth_full) | (depth_full <= 0.0)
  d = np.where(invalid, np.nan, depth_full).astype(np.float32, copy=False)

  # 2. 计算每个网格块的尺寸，并对原图进行居中裁剪 (Center-crop)，确保能够被目标网格均匀整除
  bh, bw = h0 // height, w0 // width          # 单个网格块的高与宽（像素数）
  ch, cw = bh * height, bw * width            # 整除后可利用的中心区域总高宽
  top, left = (h0 - ch) // 2, (w0 - cw) // 2  # 居中裁剪偏移量

  # 3. 维度变换与重塑：将每个网格块内部的所有像素收集到最后一个轴 (axis=2)
  # [H, bh, W, bw] -> transpose -> [H, W, bh, bw] -> reshape -> [H, W, bh * bw]
  blocks = (
    d[top : top + ch, left : left + cw]
    .reshape(height, bh, width, bw)
    .transpose(0, 2, 1, 3)
    .reshape(height, width, bh * bw)
  )

  # 4. 沿轴 2 计算低分位数；若某网格块内全部为 NaN，numpy 会抛出 RuntimeWarning，此处安全捕获忽略
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)  # 忽略全 NaN 块的统计警告
    pooled = np.nanpercentile(blocks, percentile, axis=2)

  # 5. 将全无效网格块（计算结果为 NaN 或非有限值）回填为 fill_invalid（例如 far 远距离）
  return np.where(np.isfinite(pooled), pooled, fill_invalid).astype(np.float32)


def min_pool_depth(
  depth_full: np.ndarray, height: int, width: int, fill_invalid: float = 0.0
) -> np.ndarray:
  """严格最小值池化函数（分位数 percentile=0 的快捷别名封装）。

  等价于调用 `percentile_pool_depth(depth_full, height, width, 0.0, fill_invalid)`。
  """
  return percentile_pool_depth(depth_full, height, width, 0.0, fill_invalid)


class ZedDepthSource:
  """后台异步 ZED Mini 深度图像抓取与下采样适配器 (Background ZED Mini depth grabber)。

  【工作机制】
  在后台独立启动一个守护线程持续以指定帧率从 ZED Mini 相机捕获原生深度和 RGB 图像。
  仿真主循环调用 `latest_frame()` 可以无等待、非阻塞地立即获取当前最新的
  `[height, width]` 尺寸 float32 深度张量（单位: 米，在指定 PyTorch 计算设备上），
  从而使硬件深度输入无缝平替仿真环境中的头部相机观察项。
  """

  @staticmethod
  def list_devices() -> list[tuple[int, str]]:
    """扫描并列出所有当前已连接的 ZED 相机硬件设备列表。

    返回:
        list[tuple[int, str]]: 包含 `[(序列号 serial_number, 相机型号 model), ...]` 的列表；
        若系统中未安装 pyzed SDK，则优雅返回空列表 `[]`。
    """
    try:
      import pyzed.sl as sl
    except ImportError:
      return []
    out: list[tuple[int, str]] = []
    for dev in sl.Camera.get_device_list():
      out.append((int(dev.serial_number), str(getattr(dev, "camera_model", "?"))))
    return out

  def __init__(
    self,
    height: int = 9,
    width: int = 16,
    depth_mode: str = "PERFORMANCE",
    flip: bool = False,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    pool_percentile: float = 3.0,
    fill_holes: bool = False,
    confidence_threshold: int = 50,
    texture_confidence_threshold: int = 100,
    edge_crop_px: int = 0,
    obs_mask: str = "none",
    device: str = "cpu",
    first_frame_timeout_s: float = 10.0,
    serial_number: int | None = None,
  ) -> None:
    """初始化 ZED 相机参数并启动后台图像捕获线程。

    参数:
        height: 策略端目标输入高度，默认 9。
        width: 策略端目标输入宽度，默认 16。
        depth_mode: ZED 深度感知算法模式，支持 "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL" 等。
        flip: 是否 180 度翻转图像（倒置安装相机时开启）。
        min_depth_m: 测量最近有效距离（米），默认 0.1 m。
        max_depth_m: 测量最远有效距离（米），默认 5.0 m。
        pool_percentile: 下采样软最小值池化分位数百分比，默认 3.0 (3%)。
        fill_holes: 是否开启 SDK 的空洞插值填充（默认 False，禁止盲目填补假近景表面）。
        confidence_threshold: 深度置信度阈值 [1, 100]，低于该值的边缘/离群噪点将被标记为无效。
        texture_confidence_threshold: 纹理置信度阈值 [1, 100]，纯色弱纹理区域的过滤阈值。
        edge_crop_px: 外边缘裁切像素宽度（剔除双目边缘视差遮挡盲区的伪影），默认 0。
        obs_mask: 策略观察遮罩诊断模式，"none" 输出真实深度，"far" 强制输出全远景图像（用于对比实验）。
        device: PyTorch 张量设备（如 "cuda:0" 或 "cpu"）。
        first_frame_timeout_s: 打开相机后等待首帧的最长超时时间（秒）。
        serial_number: 指定打开的特定相机硬件序列号（多相机挂载时用于区分）。
    """
    try:
      import pyzed.sl as sl
    except ImportError as e:  # pragma: no cover - 仅在未安装 SDK 时触发
      raise RuntimeError(
        "pyzed is not installed. Install it into the venv with:\n"
        "  uv run python /usr/local/zed/get_python_api.py   # downloads the cp311 wheel\n"
        "  uv pip install ./pyzed-5.1-cp311-cp311-linux_x86_64.whl"
      ) from e

    self._sl = sl
    self.height = int(height)
    self.width = int(width)
    self._device = device
    self._near = float(min_depth_m)
    self._far = float(max_depth_m)
    self._pool_percentile = float(pool_percentile)
    self._edge_crop = max(0, int(edge_crop_px))

    # 策略观察掩码诊断模式：
    # "none" -> 真实池化深度图
    # "far"  -> 全填 far 距离（用于评估机器人失去视觉感知时的盲操退化表现）
    if obs_mask not in ("none", "far"):
      raise ValueError(f"obs_mask must be 'none' or 'far', got {obs_mask!r}")
    self._obs_mask = obs_mask

    # 全分辨率（池化前）深度图尺寸，将在收到首帧时自动获取
    self.raw_height: int = 0
    self.raw_width: int = 0

    # 配置 ZED 相机硬件初始化参数
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.VGA  # WVGA 分辨率 672x376 (~16:9，与仿真相机视场角匹配)
    init.coordinate_units = sl.UNIT.METER       # 深度公制单位：米
    try:
      init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    except AttributeError as e:
      raise ValueError(f"unknown ZED depth_mode {depth_mode!r}") from e
    init.depth_minimum_distance = float(min_depth_m)
    init.depth_maximum_distance = float(max_depth_m)
    if flip:
      init.camera_image_flip = sl.FLIP_MODE.ON

    # 通过序列号精准绑定相机实例（避免双相机冲突）
    self.serial_number = None if serial_number is None else int(serial_number)
    if self.serial_number is not None:
      init.set_from_serial_number(self.serial_number)

    # 打开相机硬件设备
    self._cam = sl.Camera()
    status = self._cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
      _which = f" (serial {self.serial_number})" if self.serial_number is not None else ""
      raise RuntimeError(
        f"Failed to open ZED camera{_which} ({status}). Is the ZED plugged in, the SDK "
        f"installed, and the serial correct? Connected: {ZedDepthSource.list_devices()}"
      )

    # 运行时滤波参数设置
    self._runtime = sl.RuntimeParameters()
    # 空洞填充模式：关闭后低纹理空洞保持为无效点，交由软最小值处理为 far，防止出现近距离幽灵假目标
    self._runtime.enable_fill_mode = bool(fill_holes)
    # 置信度阈值过滤：置信度低于阈值的点提前标记为无效
    self._runtime.confidence_threshold = int(confidence_threshold)
    self._runtime.texture_confidence_threshold = int(texture_confidence_threshold)

    # 图像抓取缓冲区与线程锁
    self._mat = sl.Mat()      # 原生深度数据容器
    self._img_mat = sl.Mat()  # 左目彩色 RGB 图像容器 (供可视化监控)
    self._latest: np.ndarray | None = None      # 最新池化后深度 [height, width] (米)
    self._latest_raw: np.ndarray | None = None  # 最新原始全分辨率深度 [H0, W0] (米)
    self._latest_rgb: np.ndarray | None = None  # 最新彩色图 [H0, W0, 3] (uint8 RGB)
    self._lock = threading.Lock()
    self._stop = threading.Event()

    # 启动后台守护采集线程
    self._thread = threading.Thread(target=self._loop, name="zed-depth", daemon=True)
    self._thread.start()

    # 同步阻塞直到成功抓取到首帧图像，保证上层控制器在第一步即可读取有效数据
    deadline = first_frame_timeout_s
    waited = 0.0
    while self._latest is None and waited < deadline:
      time.sleep(0.05)
      waited += 0.05
    if self._latest is None:
      self.close()
      raise RuntimeError(
        f"No ZED depth frame within {first_frame_timeout_s}s of opening the camera (打开相机后在超时时间内未收到首帧图像)."
      )
    self.raw_height, self.raw_width = self._latest_raw.shape  # type: ignore[union-attr]

  def _loop(self) -> None:
    """后台采集守护线程执行循环：持续读取帧、执行软最小值池化并更新缓存。"""
    sl = self._sl
    while not self._stop.is_set():
      # 尝试抓取新的一帧
      if self._cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
        continue

      # 获取深度浮点数据矩阵 (float32, 单位: 米)
      self._cam.retrieve_measure(self._mat, sl.MEASURE.DEPTH)
      depth_np = np.array(self._mat.get_data(), copy=True)  # [H, W] float32 metres

      # 获取左目 RGB 图像（用于可视化显示，ZED 原生返回 BGRA，需切片转为 RGB）
      self._cam.retrieve_image(self._img_mat, sl.VIEW.LEFT)
      rgb_np = np.ascontiguousarray(self._img_mat.get_data()[:, :, 2::-1])  # [H, W, 3] uint8

      # 执行低分位数（软最小值）池化
      if self._obs_mask == "far":
        # 诊断模式：将策略输入强制覆盖为全远距离无障碍图像
        small = np.full((self.height, self.width), self._far, dtype=np.float32)
      else:
        pool_in = depth_np
        # 若指定了边缘裁切，则将四周边缘置为 NaN，消除双目外边缘遮挡匹配噪点
        if self._edge_crop:
          n = self._edge_crop
          pool_in = depth_np.copy()
          pool_in[:n, :] = np.nan
          pool_in[-n:, :] = np.nan
          pool_in[:, :n] = np.nan
          pool_in[:, -n:] = np.nan
        small = percentile_pool_depth(
          pool_in, self.height, self.width, self._pool_percentile, fill_invalid=self._far
        )

      # 处理用于面板监控显示的全分辨率深度：无效点设为 far，截断在 [near, far] 区间
      finite = np.isfinite(depth_np) & (depth_np > 0.0)
      raw_disp = np.where(finite, depth_np, self._far).astype(np.float32)
      np.clip(raw_disp, self._near, self._far, out=raw_disp)

      # 加互斥锁更新最新帧张量
      with self._lock:
        self._latest = small
        self._latest_raw = raw_disp
        self._latest_rgb = rgb_np

  def latest_frame(self) -> torch.Tensor:
    """获取最新池化后的策略输入深度帧 (Most recent POOLED depth frame)。

    返回:
        torch.Tensor: 形状为 `[height, width]` 的 float32 深度张量（单位: 米），位于指定 `device` 上。
        该分辨率与策略网络观测输入（如 16x9）严格对齐。
    """
    with self._lock:
      frame = self._latest
    if frame is None:
      raise RuntimeError("ZED depth source has no frame yet (ZED 深度数据源尚未就绪).")
    return torch.from_numpy(frame).to(self._device)

  def latest_raw(self) -> torch.Tensor:
    """获取最新未经池化的全分辨率深度图（仅供可视化监控显示使用）。

    返回:
        torch.Tensor: 形状为 `[raw_height, raw_width]` 的 float32 深度张量（位于 CPU 上）。
    """
    with self._lock:
      frame = self._latest_raw
    if frame is None:
      raise RuntimeError("ZED depth source has no frame yet.")
    return torch.from_numpy(frame)

  def latest_rgb(self) -> torch.Tensor:
    """获取最新左目彩色 RGB 图像（仅供可视化监控显示使用）。

    返回:
        torch.Tensor: 形状为 `[raw_height, raw_width, 3]` 的 uint8 彩色张量（位于 CPU 上）。
    """
    with self._lock:
      frame = self._latest_rgb
    if frame is None:
      raise RuntimeError("ZED depth source has no frame yet.")
    return torch.from_numpy(frame)

  def close(self) -> None:
    """关闭后台抓取线程并安全释放 ZED 相机硬件设备连接。"""
    self._stop.set()
    thread = getattr(self, "_thread", None)
    if thread is not None and thread.is_alive():
      thread.join(timeout=2.0)
    cam = getattr(self, "_cam", None)
    if cam is not None:
      cam.close()

  def __enter__(self) -> "ZedDepthSource":
    """支持上下文管理器协议 (with 语法)。"""
    return self

  def __exit__(self, *exc) -> None:
    """退出上下文管理器时自动调用 close 释放资源。"""
    self.close()


def _self_test() -> None:
  """无需真实相机的纯数学单元自测：校验分位数池化、无效点填充与抗噪性能。"""
  # 测试用例 1: 4x6 矩阵下采样为 2x3 (每块 2x2)。验证严格最小值计算正确性
  src = np.array(
    [
      [5.0, 5.0, 4.0, 4.0, 3.0, 3.0],
      [5.0, 2.0, 4.0, 4.0, 3.0, 3.0],  # 块 (0,0) 内包含一个近距离 2.0
      [9.0, 9.0, 8.0, 8.0, 7.0, 7.0],
      [9.0, 9.0, 8.0, 8.0, 7.0, 7.0],
    ],
    dtype=np.float32,
  )
  out = min_pool_depth(src, 2, 3)
  expected = np.array([[2.0, 4.0, 3.0], [9.0, 8.0, 7.0]], dtype=np.float32)
  assert np.array_equal(out, expected), f"池化计算结果不匹配: {out} vs {expected}"

  # 测试用例 2: 无效值处理 (NaN / Inf / <=0 必须被剔除；全无效块填充默认 0.0)
  src2 = np.array(
    [
      [np.nan, 1.5, np.inf, 0.0],
      [-1.0, 1.5, np.nan, 0.0],
    ],
    dtype=np.float32,
  )
  out2 = min_pool_depth(src2, 1, 2)
  # 块 0 = {nan, 1.5, -1, 1.5} -> 有效值为 1.5; 块 1 = {inf, 0, nan, 0} 全无效 -> 填充 0.0
  assert np.array_equal(out2, np.array([[1.5, 0.0]], dtype=np.float32)), out2

  # 测试用例 3: 自定义 fill_invalid=5.0 (全无效块填充为 far 距离)
  out3 = min_pool_depth(src2, 1, 2, fill_invalid=5.0)
  assert np.array_equal(out3, np.array([[1.5, 5.0]], dtype=np.float32)), out3

  # 测试用例 4: 分位数抗噪测试 (单个孤立近距离离群噪点应被分位数过滤，但被严格 min 误保留)
  # 1 个 3x3 图像块下采样到 1x1：包含 8 个远距离 5.0 和 1 个噪点 1.0
  src3 = np.full((3, 3), 5.0, dtype=np.float32)
  src3[1, 1] = 1.0  # 孤立近距离噪点
  assert min_pool_depth(src3, 1, 1)[0, 0] == 1.0  # 严格 min 误保留了噪点
  # 50 分位数（中位数）成功忽略孤立噪点 -> 5.0
  assert percentile_pool_depth(src3, 1, 1, 50.0)[0, 0] == 5.0
  # percentile=0 等价于严格 min
  assert percentile_pool_depth(src3, 1, 1, 0.0)[0, 0] == 1.0

  print("zed_depth_source self-test OK (ZED 深度源数学自测通过)")


if __name__ == "__main__":
  _self_test()
