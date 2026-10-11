"""ZED Mini 深度相机实时采集与百分位数池化处理模块（纯 NumPy 实现）。

在部署架构中的角色：
-------------------
本模块提供从真实 ZED Mini 相机获取深度数据并在后台多线程下采样的一体化驱动。
完全移除了 PyTorch 依赖，仅使用 NumPy：
- ``latest_frame()`` 返回形状为 ``[H, W]`` (默认 9x16) 的 float32 米制深度数组；
  无效像素（盲区/超量程）填充为 ``fill_invalid``（默认设为 DEPTH_FAR=5.0m，与仿真评估保持一致）。
- 策略节点 (dodge_policy) 接收该深度图后，再执行标准 DepthImageObs 的
  “未击中置远 (no-hit->far) / 范围裁剪 (clamp) / 归一化 (normalise)”流水线，
  保证实机策略接收的表征与强化学习训练环境完全一致。

多线程与容错设计：
-----------------
``pyzed`` 采用延迟动态导入，使得无需连接相机或未安装 ZED SDK 的开发机也能正常导入本模块
进行百分位数池化算法测试与离线单元测试。
相机图像采集运行在独立的后台守护线程中，策略或相机发布节点读取最新帧耗时仅数微秒，
绝不阻塞 50 Hz 的电机控制主循环。
"""

from __future__ import annotations

import threading
import time

import numpy as np


def percentile_pool_depth(
    depth_full: np.ndarray,
    height: int,
    width: int,
    percentile: float = 0.0,
    fill_invalid: float = 0.0,
) -> np.ndarray:
    """对全分辨率深度图（单位：米）进行分块低百分位数（软最小值 Soft-Min）池化降采样至 ``height x width``。

    为什么采用低百分位数（Soft-Min）池化，而不是严格最小值或均值池化？
    -------------------------------------------------------------
    1. 相比严格最小值池化 (Strict-Min, percentile=0)：
       真实双目相机常存在 1~2 个测量噪点像素（飞点/闪烁孤立点）。严格最小值会让孤立噪点直接污染
       整个池化网格单元，使远处空旷空间误报存在近处障碍物；而设定极低的百分位（如 2%~5%），
       可以稳健剔除这 1~2 个孤立近噪点。
    2. 相比均值池化 (Average Pooling)：
       均值池化是完全错误的！飞来的小球在相机网格块中占比很小（在 2~5 米距离下，球在局部块中的
       像素覆盖率仅约 5%~18%）。均值池化会将近处小球的深度完全冲淡并融合进广阔的背景距离中，
       甚至在空间中凭空捏造出一个仿真中从未出现过的“中等距离虚假表面”，导致策略失察被砸。
    3. 百分位取值必须保持极小（建议 2.0%~5.0%），使其始终低于球体在块内的像素占比，
       从而确保小球在进入视场时能可靠被检测到。

    无效值与边界处理：
    -----------------
    - 无效像素（NaN、+/-Inf 或 <= 0.0，即 ZED 盲区、视差失败或超出量程）在排序时被推至 +Inf；
    - 若整个分块内全部为无效像素，或有效像素数量不足以支撑百分位选择，则返回 ``fill_invalid``
      （通常设为 far=5.0m，使空闲网格在归一化后映射为 1.0，代表空旷远景）；
    - 图像分块前先进行中心裁剪 (center-crop)，使其尺寸能被 target ``height x width`` 整除。

    性能优化（O(N) 快速选择算法）：
    -----------------------------
    本函数采用 ``np.partition``（内省快速选择算法 Introselect，线性复杂度 O(N)），
    而不是常规的 ``np.nanpercentile``（全排序加 NaN 簿记）。处理单帧全分辨率深度图
    速度提升约 10 倍（单帧耗时从 ~15ms 降低至 ~1.5ms），确保抓取线程跑满 60 FPS 相机帧率。

    参数:
        depth_full: 形状为 (H, W) 的 2D 原始深度图，float32 格式，单位为米。
        height: 目标池化网格高度 (行数，单目躲避为 9)。
        width: 目标池化网格宽度 (列数，单目躲避为 16)。
        percentile: 选取的百分位数 [0, 100]，0 代表严格最小值，推荐 2.0~5.0。
        fill_invalid: 当分块全部为无效值时填充的距离数值（通常为 DEPTH_FAR=5.0m）。

    返回:
        np.ndarray: 形状为 (height, width) 的池化后深度数组，float32。
    """
    if depth_full.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {depth_full.shape}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")
    h0, w0 = depth_full.shape
    if h0 < height or w0 < width:
        raise ValueError(f"source {h0}x{w0} smaller than target {height}x{width}")

    # 将所有非有限值 (NaN, +/-Inf) 及非正深度值（盲区/视差失效）设为 +inf，
    # 这样在升序快速选择中它们会排在最后，低百分位数挑出的即为块内第 k 个最小的有效数值。
    invalid = ~np.isfinite(depth_full) | (depth_full <= 0.0)
    d = np.where(invalid, np.inf, depth_full).astype(np.float32, copy=False)

    # 计算整除网格块尺寸，并进行图像中心裁剪
    bh, bw = h0 // height, w0 // width
    ch, cw = bh * height, bw * width
    top, left = (h0 - ch) // 2, (w0 - cw) // 2
    blocks = (
        d[top : top + ch, left : left + cw]
        .reshape(height, bh, width, bw)
        .transpose(0, 2, 1, 3)
        .reshape(height, width, bh * bw)
    )

    # 使用 np.partition 沿第 2 轴进行 O(N) 原地快速划分，选取索引为 k 的元素。
    # 相比 np.nanpercentile 的完全排序，提速近 10 倍。
    n = blocks.shape[2]
    k = int(round((n - 1) * (percentile / 100.0)))
    pooled = np.partition(blocks, k, axis=2)[:, :, k]

    # 若该块第 k 个值依然是 +inf（说明该块内有效像素不足 k+1 个），则回退填充为 fill_invalid
    return np.where(np.isfinite(pooled), pooled, fill_invalid).astype(np.float32)


def min_pool_depth(
    depth_full: np.ndarray, height: int, width: int, fill_invalid: float = 0.0
) -> np.ndarray:
    """严格最小值池化（percentile=0.0）。为 :func:`percentile_pool_depth` 的精简封装。"""
    return percentile_pool_depth(depth_full, height, width, 0.0, fill_invalid)


def mask_depth_to_far(
    depth_full: np.ndarray, mask: np.ndarray, far: float
) -> np.ndarray:
    """目标球体掩码远端填充：仅保留 ``mask`` 为 True 的前景真实深度，其他所有背景像素强制设为 ``far`` 米。

    在感知与策略中的核心原理 (Ball-only Masked Depth):
    --------------------------------------------------
    - 本函数是实机 EfficientTAM 感知链条的核心。经由目标分割网络提取出球体的布尔掩码 mask；
    - 通过本函数处理后，画面中除了球体具有真实的近处距离外，所有杂乱的背景环境（地面、墙壁、人体等）
      均被强制涂覆为远平面距离 (far=5.0m)；
    - 策略网络接收到的是一张极其“纯净”的深度图——完全消除了非躲避障碍物（比如操作人员、实验室背景）
      对策略网络的视觉误导，使网络专注于追踪逼近的球体；
    - 该操作也天然消除了 ZED 双目相机的边缘视差伪影（因为边缘非球体像素全部被覆写为 far）。

    参数:
        depth_full: 原始输入全分辨率深度图 (H, W)。
        mask: 与 depth_full 同尺寸的布尔掩码数组 (H, W)，True 表示目标球体前景。
        far: 背景填充距离（米），一般取 DEPTH_FAR=5.0m。

    返回:
        np.ndarray: 背景被置为 far 的纯净深度图。
    """
    depth = np.asarray(depth_full, dtype=np.float32)
    sel = np.asarray(mask, dtype=bool)
    if sel.shape != depth.shape:
        raise ValueError(f"mask shape {sel.shape} does not match depth {depth.shape}")
    return np.where(sel, depth, np.float32(far))


class ZedDepthSource:
    """ZED Mini 深度相机后台守护抓取器，持续输出与仿真格式对齐的 NumPy 深度数组。

    功能与架构特性：
    ---------------
    - ``latest_frame()``: 以极小开销（互斥锁读取缓存拷贝）返回最新的 ``[height, width]``
      float32 米制深度帧，避免相机 I/O 阻塞电机控制循环；
    - 后台采集线程持续以相机硬件帧率 (60 FPS) 抓取、滤波并执行百分位数池化；
    - 内置设备重试、双目视差边缘裁剪、置信度阈值过滤及多相机串号绑定等全套实机防护特性。
    """

    @staticmethod
    def list_devices() -> list[tuple[int, str]]:
        """枚举当前系统连接的所有 ZED 相机，返回 ``[(serial_number, model), ...]`` 列表。
        若系统未安装 pyzed，则返回空列表。用于多相机部署时绑定序列号。"""
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
        fps: int = 60,
        flip: bool = False,
        min_depth_m: float = 0.1,
        max_depth_m: float = 5.0,
        pool_percentile: float = 3.0,
        fill_holes: bool = False,
        confidence_threshold: int = 50,
        texture_confidence_threshold: int = 100,
        edge_crop_px: int = 0,
        fill_invalid: float = 5.0,
        first_frame_timeout_s: float = 10.0,
        serial_number: int | None = None,
        open_max_attempts: int = 5,
        open_retry_delay_s: float = 2.0,
        keep_full: bool = False,
    ) -> None:
        """初始化 ZED 深度采集源并启动后台抓取线程。

        参数:
            height, width: 池化后的深度图分辨率（默认 9x16）。
            depth_mode: ZED SDK 深度计算模式。默认为 'PERFORMANCE'（最快速且带真实立体视觉噪声，
                        最贴合仿真中 DR 域随机化噪声；若选择 'NEURAL' 在 Orin 上帧率通常难以维持 60 FPS）。
            fps: 相机硬件抓取帧率（VGA 分辨率下支持 15/30/60/100，固定设为 60 FPS，高于 50 Hz 控制率）。
            flip: 图像是否 180 度翻转（用于倒置安装）。
            min_depth_m, max_depth_m: 深度裁剪范围 [0.1m, 5.0m]。
            pool_percentile: 降采样百分位数（默认 3.0% 软最小值）。
            fill_holes: 是否开启 ZED FILL 孔洞插值模式（强烈建议关闭！FILL 模式会在无纹理或空旷盲区
                        脑补近距离虚假表面，被软最小值池化捕捉后会引发误躲避）。
            confidence_threshold: ZED 深度置信度阈值 (0-100，越小越严格，默认 50，SDK 默认 95 太宽松)。
            texture_confidence_threshold: 纹理置信度阈值 (0-100，越小越严格，白墙空旷区可设为 50)。
            edge_crop_px: 全分辨率图像外圈黑边裁剪像素数（剔除双目立体视差边界盲区的伪影）。
            fill_invalid: 无效值填充距离（米，默认 5.0m）。
            first_frame_timeout_s: 相机初始化获取首帧的超时时间。
            serial_number: 绑定的特定 ZED 相机硬件序列号（多相机时使用）。
            open_max_attempts, open_retry_delay_s: 相机初次打开失败时的重试机制参数。
            keep_full: 是否在内存中额外保留全分辨率原始深度图与 BGR 彩色图像（供 UI 显示或 EfficientTAM 使用）。
        """
        try:
            import pyzed.sl as sl
        except ImportError as e:  # 仅在未安装 SDK 时触发
            raise RuntimeError(
                "pyzed 未安装。在 Jetson 上，请安装系统 Python 3.8 适配的 wheel 包:\n"
                "  uv pip install /usr/local/zed/pyzed-*-cp38-*.whl"
            ) from e

        self._sl = sl
        self.height = int(height)
        self.width = int(width)
        self._near = float(min_depth_m)
        self._far = float(max_depth_m)
        self._pool_percentile = float(pool_percentile)
        self._edge_crop = max(0, int(edge_crop_px))
        self._fill_invalid = float(fill_invalid)
        self.raw_height: int = 0
        self.raw_width: int = 0

        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.VGA  # WVGA 672x376 (~16:9，完美匹配仿真相机 FOV)
        # 固定采集帧率为 60 FPS（高于 50 Hz 控制率，确保策略推入环形缓冲区的历史帧基本为最新帧）
        init.camera_fps = int(fps)
        init.coordinate_units = sl.UNIT.METER
        try:
            init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        except AttributeError as e:
            raise ValueError(f"未知的 ZED depth_mode: {depth_mode!r}") from e
        init.depth_minimum_distance = float(min_depth_m)
        init.depth_maximum_distance = float(max_depth_m)
        if flip:
            init.camera_image_flip = sl.FLIP_MODE.ON
        self.serial_number = None if serial_number is None else int(serial_number)
        if self.serial_number is not None:
            init.set_from_serial_number(self.serial_number)

        self._cam = sl.Camera()
        # 打开相机重试逻辑：ZED Mini 在冷启动首次打开时容易报 CAMERA STREAM FAILED TO START，
        # 等待数秒重试往往能够成功打开，避免启动脚本因单次握手失败崩溃。
        _which = f" (序列号 {self.serial_number})" if self.serial_number is not None else ""
        status = None
        for attempt in range(1, int(open_max_attempts) + 1):
            status = self._cam.open(init)
            if status == sl.ERROR_CODE.SUCCESS:
                break
            print(
                f"[zed] 打开相机尝试 {attempt}/{open_max_attempts} 失败{_which}: {status}"
                + (f"; 将在 {open_retry_delay_s:.0f} 秒后重试..." if attempt < open_max_attempts else ""),
                flush=True,
            )
            if attempt < open_max_attempts:
                time.sleep(float(open_retry_delay_s))
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"在尝试 {open_max_attempts} 次后仍未能打开 ZED 相机{_which} ({status})。"
                f"请检查 USB 连线、ZED SDK 安装状态或序列号是否正确。当前已连接设备: {ZedDepthSource.list_devices()}"
            )

        self._runtime = sl.RuntimeParameters()
        # 禁用 FILL 填洞模式：FILL 会脑补虚假近处距离，导致软最小值池化误报近距离障碍
        self._runtime.enable_fill_mode = bool(fill_holes)
        # 置信度滤波：在池化前剔除低置信度和低纹理像素，使盲区保持无效值，避免拉低整块深度
        self._runtime.confidence_threshold = int(confidence_threshold)
        self._runtime.texture_confidence_threshold = int(texture_confidence_threshold)
        self._mat = sl.Mat()
        self._latest: np.ndarray | None = None  # 池化后的最新帧 [H, W]，单位米
        # 全分辨率图像缓存（仅在 keep_full=True 时启用）
        self._keep_full = bool(keep_full)
        self._img_mat = sl.Mat() if self._keep_full else None
        self._latest_full: np.ndarray | None = None
        self._latest_bgr: np.ndarray | None = None
        self.grab_count: int = 0  # 成功抓取帧计数器（用于实机吞吐量监控与卡死自愈）
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="zed-depth", daemon=True)
        self._thread.start()

        # 等待首帧到达
        deadline, waited = first_frame_timeout_s, 0.0
        while self._latest is None and waited < deadline:
            time.sleep(0.05)
            waited += 0.05
        if self._latest is None:
            self.close()
            raise RuntimeError(
                f"在相机成功打开后 {first_frame_timeout_s} 秒内未接收到有效的 ZED 深度帧。"
            )

    def _loop(self) -> None:
        """后台独立抓取线程循环：持续从相机拉取深度帧并执行百分位数池化。"""
        sl = self._sl
        while not self._stop.is_set():
            if self._cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            self._cam.retrieve_measure(self._mat, sl.MEASURE.DEPTH)
            depth_np = np.array(self._mat.get_data(), copy=True)  # [H0, W0] float32 米制深度
            # 裁剪双目视差边缘（左相机左侧及顶端盲区），去除边缘视差噪点
            pool_in = depth_np
            if self._edge_crop:
                n = self._edge_crop
                pool_in = depth_np.copy()
                pool_in[:n, :] = np.nan
                pool_in[-n:, :] = np.nan
                pool_in[:, :n] = np.nan
                pool_in[:, -n:] = np.nan
            # 执行高性能软最小值池化
            small = percentile_pool_depth(
                pool_in, self.height, self.width, self._pool_percentile,
                fill_invalid=self._fill_invalid,
            )
            bgr_np = None
            if self._keep_full:
                self._cam.retrieve_image(self._img_mat, sl.VIEW.LEFT)
                bgr_np = np.ascontiguousarray(self._img_mat.get_data()[:, :, :3])  # BGRA 转换为 BGR
            with self._lock:
                self._latest = small
                if self._keep_full:
                    self._latest_full = depth_np
                    self._latest_bgr = bgr_np
                self.grab_count += 1
                if not self.raw_height:
                    self.raw_height, self.raw_width = depth_np.shape

    def latest_frame(self) -> np.ndarray:
        """获取最新的池化深度帧: ``[height, width]`` float32（单位：米）。
        无效值填充为 ``fill_invalid``。返回深拷贝，调用方可随意修改。"""
        with self._lock:
            frame = self._latest
        if frame is None:
            raise RuntimeError("ZED 深度采集器尚未就绪。")
        return frame.copy()

    def latest_full(self):
        """获取最新全分辨率数据: (raw_depth [H0, W0], left_bgr [H0, W0, 3])。
        仅在初始化传入 ``keep_full=True`` 时有效，未启用时返回 (None, None)。"""
        with self._lock:
            d, b = self._latest_full, self._latest_bgr
        if d is None:
            return None, None
        return d.copy(), (b.copy() if b is not None else None)

    def close(self) -> None:
        """安全停止后台抓取线程并释放 ZED 相机硬件句柄。"""
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        cam = getattr(self, "_cam", None)
        if cam is not None:
            cam.close()

    def __enter__(self) -> "ZedDepthSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _self_test() -> None:
    """脱离硬件环境的池化数学逻辑自测试。"""
    src = np.array(
        [
            [5.0, 5.0, 4.0, 4.0, 3.0, 3.0],
            [5.0, 2.0, 4.0, 4.0, 3.0, 3.0],  # 块 (0,0) 内包含一个最近距离 2.0
            [9.0, 9.0, 8.0, 8.0, 7.0, 7.0],
            [9.0, 9.0, 8.0, 8.0, 7.0, 7.0],
        ],
        dtype=np.float32,
    )
    out = min_pool_depth(src, 2, 3)
    expected = np.array([[2.0, 4.0, 3.0], [9.0, 8.0, 7.0]], dtype=np.float32)
    assert np.array_equal(out, expected), (out, expected)

    src2 = np.array([[np.nan, 1.5, np.inf, 0.0], [-1.0, 1.5, np.nan, 0.0]], dtype=np.float32)
    out2 = min_pool_depth(src2, 1, 2)
    assert np.array_equal(out2, np.array([[1.5, 0.0]], dtype=np.float32)), out2
    out3 = min_pool_depth(src2, 1, 2, fill_invalid=5.0)
    assert np.array_equal(out3, np.array([[1.5, 5.0]], dtype=np.float32)), out3

    src3 = np.full((3, 3), 5.0, dtype=np.float32)
    src3[1, 1] = 1.0
    assert min_pool_depth(src3, 1, 1)[0, 0] == 1.0
    assert percentile_pool_depth(src3, 1, 1, 50.0)[0, 0] == 5.0
    assert percentile_pool_depth(src3, 1, 1, 0.0)[0, 0] == 1.0
    print("zed_depth_source 百分位数池化自测试通过 (OK)")


if __name__ == "__main__":
    _self_test()
