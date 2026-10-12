"""强化学习观测空间 (Observations) 管理模块.

【模块定位与观测体系分层】
本模块负责从物理仿真器 (MuJoCo/Warp) 中提取、转换、拼接并归一化智能体所需的全部多模态状态信息。
在复杂的敏捷运动控制、高速来球躲避以及走廊避障任务中，观测空间分为三大核心层次：

1. 本体感觉观测 (Proprioception Observations - Actor 可见):
   - 机器人关键刚体（骨盆、四肢末端等）在基座偏航坐标系下的相对空间位置、姿态矩阵、线速度与角速度；
   - 关节位置误差 $q - q_{\\text{default}}$、关节速度 $\\dot{q}$ 以及上一拍历史动作。

2. 视觉深度感知与 Sim2Real 域随机化 (Visual Depth Perception - Actor 可见):
   - `DepthImageObs`: 读取头戴深度相机 (16x9 宽高比，模拟 ZED Mini WVGA 相机) 真实几何测距深度；
   - 观测时序拼接 (Temporal Frame-Stacking): 通过 GPU 循环缓冲区 (Ring Buffer)，以膨胀步长抽样
     历史多帧深度（如 `(0, 1)` 或稀疏帧 `(0, 5, 10, 20, 40)`），使前馈网络具有捕捉来球 loom（视尺寸快速膨胀）
     与纵向速度特征的时序感知野；
   - `BallOnlyDepthObs`: 球体专属掩码深度图 (16x9)。模拟实机边缘计算芯片（如 Jetson Orin）上运行
     EfficientTAM / YOLO 分割模型后的输出——仅保留球体像素的测距值，其余背景全被掩膜为 `far`。
     配合掩码抖动、丢球率随机化以及边缘斑块闪烁，直接消除复杂室内背景的 Sim2Real 视觉鸿沟；
   - 深度域随机化 (Depth Domain Randomization):
     * 房间背景渐变填充 (`_bg_fill`): 随机生成假想天花板与墙面深度渐变，防止策略将空旷训练场捷径硬编码；
     * 双目立体视差失效斑点与空洞 (`_augment`, `_blink_aug`): 在深度不连续边缘与远景生成闪烁噪斑与孔洞；
     * 慢速漂移假目标物体 (`_advance_clutter`): 生成带有自身微小抖动的大尺寸矩形物体，迫使策略区分“静止环境”与“高速来球”。

3. 特权几何与 CBF 安全状态 (Privileged State - 仅 Critic 可见，不对 Actor 暴露):
   - `dodge_ball_state_b`: 球体相对于机器人的精确 3D 相对位置与相对线速度；
   - `dodge_wall_state_b` 与 `dodge_obstacle_geometry_b`: 最近障碍墙的中心坐标、局部包围盒尺寸、相对航向角与表面几何净间距；
   - `dodge_cbf_state_b`: 控制屏障函数 (CBF) 计算出的安全滤波特权状态，包括威胁标志、屏障值 $h$、碰撞时间 TTI、
     逃逸梯度方向 $e_b$、理论安全速度修正量 $\\Delta u$ 以及真实球半径；
   - `BallObservableGate`: 视锥内可见性门控。检测来球是否曾穿过相机视锥，用于在自编码器/信念网络中屏蔽不可见时刻的预测损失。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
  quat_apply_inverse,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """计算指定身体刚体相对于基座锚点 (Anchor) 局部坐标系的 3D 相对空间位置.

    返回:
      形状为 `(num_envs, num_bodies * 3)` 的一维展平张量。
    """
    asset: Entity = env.scene[anchor_cfg.name]
    
    # 提取基准锚点刚体（通常为骨盆/躯干）的世界位置与四元数
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # 形状: (num_envs, 3)，锚点刚体世界系 3D 位置 (x, y, z)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # 形状: (num_envs, 4)，锚点刚体世界系旋转四元数 (w, x, y, z)
    
    # 提取目标刚体集合的世界位置与四元数
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # 形状: (num_envs, num_bodies, 3)，各跟踪刚体世界系 3D 位置
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # 形状: (num_envs, num_bodies, 4)，各跟踪刚体世界系旋转四元数

    num_bodies = body_pos_w.shape[1]
    # 执行坐标系刚体逆变换，将世界坐标转入锚点局部坐标系
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """计算指定身体刚体相对于基座锚点局部坐标系的旋转姿态矩阵 (前两列，用于连续无奇点表示).

    返回:
      形状为 `(num_envs, num_bodies * 6)` 的一维展平张量。
    """
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # 形状: (num_envs, 3)，锚点刚体世界系位置 (x, y, z)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # 形状: (num_envs, 4)，锚点刚体世界系四元数 (w, x, y, z)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # 形状: (num_envs, num_bodies, 3)，各身体刚体世界系位置
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # 形状: (num_envs, num_bodies, 4)，各身体刚体世界系四元数

    num_bodies = body_pos_w.shape[1]
    # 计算相对旋转四元数
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    # 将四元数转化为 3x3 旋转矩阵，并取前两列 (6D 连续旋转表示)
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """计算指定身体刚体在自身局部坐标系下的线速度矢量.

    返回:
      形状为 `(num_envs, num_bodies * 3)` 的一维展平张量。
    """
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]   # 形状: (num_envs, num_bodies, 3)，各刚体世界系线速度 (vx, vy, vz)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # 形状: (num_envs, num_bodies, 4)，各刚体世界系姿态四元数

    num_bodies = body_lin_vel_w.shape[1]

    # 将世界系线速度应用自身四元数逆变换旋转至局部刚体系
    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    """计算指定身体刚体在自身局部坐标系下的角速度矢量.

    返回:
      形状为 `(num_envs, num_bodies * 3)` 的一维展平张量。
    """
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]   # 形状: (num_envs, num_bodies, 3)，各刚体世界系角速度 (wx, wy, wz)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # 形状: (num_envs, num_bodies, 4)，各刚体世界系姿态四元数

    num_bodies = body_ang_vel_w.shape[1]

    # 将世界系角速度旋转至局部刚体系
    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)


class DepthImageObs(ManagerTermBase):
  """头戴深度相机观测项：读取原始真实几何距离并输出归一化深度图或展平特征.

  数据处理流水线：
  1. 无碰撞/天际线像素映射：原始测距小于 `near`（如天空未命中点）强制映射为 `far`；
  2. 距离截断：裁剪至 `[near, far]`（例如 0.1m ~ 5.0m 探测窗）；
  3. Sim2Real 域随机化 (训练阶段)：
     - 背景梯度生成（模拟室内房间天花板与墙面）；
     - 深度相关的双目匹配噪点、边缘视差跳变噪声与无效孔洞 (Dropout)；
     - 带有自运动抖动的静态杂物斑块 (Clutter Blobs)；
  4. 归一化至 `[0, 1]` 范围：`(d - near) / (far - near)`，近处目标接近 0，远处空旷接近 1；
  5. 时序环形缓冲区历史抽帧拼接 (Dilated Frame Offsets Stacking)。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    p = cfg.params
    # 绑定的深度传感器名称列表
    self._sensors: list[str] = list(p.get("sensor_names") or [p["sensor_name"]])
    # 融合模式："channel" (按通道堆叠) 或 "vertical" (沿图像垂直高度拼合)
    self._fuse: str = str(p.get("fuse", "channel"))
    self._near: float = float(p.get("near", 0.1))
    self._far: float = float(p.get("far", 5.0))
    self._flatten: bool = bool(p.get("flatten", False))
    # 感知高斯噪声参数
    self._noise_sigma_base: float = float(p.get("noise_sigma_base", 0.0))
    self._noise_sigma_per_m: float = float(p.get("noise_sigma_per_m", 0.0))
    self._dropout_rate: float = float(p.get("dropout_rate", 0.0))
    
    # Sim2Real 房间背景填充 (Background Fill)：以 bg_fill_prob 概率将远景空旷像素替换为斜率渐变墙面
    self._bg_fill_prob: float = float(p.get("bg_fill_prob", 0.0))
    self._bg_depth_range: tuple = tuple(p.get("bg_depth_range", (2.0, 5.0)))
    self._bg_state: dict = {}
    
    # 仿 ZED 双目视差失效噪斑 (Speckle) 与孔洞 (Hole) 随机化参数
    self._speckle_rate_max: float = float(p.get("speckle_rate_max", 0.0))
    self._speckle_edge_bias: float = float(p.get("speckle_edge_bias", 0.0))
    self._speckle_border_bias: float = float(p.get("speckle_border_bias", 0.0))
    self._speckle_far_bias: float = float(p.get("speckle_far_bias", 0.0))
    self._speckle_keep_min: float = float(p.get("speckle_keep_min", -1.0))
    self._haze_sigma_max: float = float(p.get("haze_sigma_max", 0.0))
    self._hole_rate_max: float = float(p.get("hole_rate_max", 0.0))
    self._border_w: torch.Tensor | None = None
    
    # 时序一致的杂物斑块 (Clutter Blobs)：模拟房间内的固定家具/墙柱
    self._clutter_n: int = int(p.get("clutter_blobs", 0))
    self._clutter_prob: float = float(p.get("clutter_prob", 0.5))
    self._clutter_edge_bias: float = float(p.get("clutter_edge_bias", 1.5))
    self._clutter_size_min: float = float(p.get("clutter_size_min", 0.04))
    self._clutter_size_max: float = float(p.get("clutter_size_max", 0.14))
    self._clutter_depth_min: float = float(p.get("clutter_depth_min", 0.1))
    self._clutter_depth_max: float = float(p.get("clutter_depth_max", 0.95))
    self._clutter_drift: float = float(p.get("clutter_drift", 0.02))
    self._clutter_jitter: float = float(p.get("clutter_jitter", 0.0))
    self._clutter_jitter_max: float = float(p.get("clutter_jitter_max", 1.0))
    self._clutter_cam_vfov_deg: float = float(p.get("clutter_cam_vfov_deg", 54.0))
    self._clutter_depth_jitter: float = float(p.get("clutter_depth_jitter", 0.0))
    self._clutter_depth_jitter_max: float = float(p.get("clutter_depth_jitter_max", 0.2))
    self._clutter_blob_prob: float = float(p.get("clutter_blob_prob", 0.8))
    self._cl_cy = self._cl_cx = self._cl_hy = self._cl_hx = None
    self._cl_d = self._cl_vy = self._cl_vx = self._cl_on = None
    self._cl_oy = self._cl_ox = None
    self._cl_od = None
    
    # 地面纹理缺失模拟 (Ground Far Out)：将地面网格像素掩膜至远端
    self._ground_far_prob: float = float(p.get("ground_far_prob", 0.0))
    self._plane_ids: torch.Tensor | None = None
    
    # 时序历史帧拼接抽样偏置 (Frame Offsets)
    fo = getattr(getattr(env, "cfg", None), "depth_frame_offsets", None)
    if fo is None:
      fo = p.get("frame_offsets", None)
    if fo is None:
      fo = (0, 1)
    self._frame_offsets: tuple[int, ...] = tuple(sorted(int(o) for o in fo))
    self._L: int = max(self._frame_offsets) + 1
    self._buf: torch.Tensor | None = None  # [N, L, *frame_shape] GPU 环形缓冲区
    self._head: int = 0
    self._pending_reset: torch.Tensor | None = None
    
    # 相机帧率采样周期控制 (Update Period)
    self._update_period: int = max(1, int(p.get("update_period", 1)))
    self._held: torch.Tensor | None = None
    self._step: int = 0
    self._aug_update_period: int = max(1, int(p.get("aug_update_period", 2)))
    self._flicker_keep: float = float(p.get("flicker_keep", 0.5))
    self._aug_state: dict[str, dict] = {}
    self._aug_step: int = 0
    self._aug_do_refresh: bool = True

  def _one(self, raw: torch.Tensor, env=None, sensor_name: str | None = None) -> torch.Tensor:
    """处理单张深度图像，执行测距裁剪、背景填充、噪声注入与归一化."""
    far, near = self._far, self._near
    raw = self._ground_far_out(raw, env, sensor_name)
    # 无碰撞天空像素设为远端
    x = torch.where(raw < near, torch.full_like(raw, far), raw)
    x = x.clamp(min=near, max=far)
    x = self._bg_fill(x, sensor_name)
    if self._noise_sigma_base > 0.0 or self._noise_sigma_per_m > 0.0:
      sigma = self._noise_sigma_base + self._noise_sigma_per_m * x
      x = (x + torch.randn_like(x) * sigma).clamp(min=near, max=far)
    if self._dropout_rate > 0.0:
      hole = torch.rand_like(x) < self._dropout_rate
      x = torch.where(hole, torch.full_like(x, far), x)
    x = self._augment(x, sensor_name)
    if env is not None and sensor_name is not None and getattr(env, "_depth_aug_display_enabled", False):
      store = getattr(env, "_depth_aug_display", None)
      if store is None:
        store = {}
        env._depth_aug_display = store
      store[sensor_name] = x.detach()
    # 归一化到 [0, 1] 区间 (0 表示近处/球体，1 表示远端/空旷)
    return (x - near) / (far - near)

  def _bg_fill(self, x: torch.Tensor, sensor_name: str | None) -> torch.Tensor:
    """将远端空白背景填充为带有随机坡度的室内房间梯度背景."""
    if self._bg_fill_prob <= 0.0:
      return x
    n, h = x.shape[0], x.shape[1]
    key = sensor_name or "_"
    st = self._bg_state.get(key)
    if st is None or st["b"].shape != (n, h):
      st = {"b": x.new_full((n, h), self._far)}
      self._bg_state[key] = st
      self._sample_bg(st, x, torch.ones(n, dtype=torch.bool, device=x.device))
    elif self._pending_reset is not None and bool(self._pending_reset.any()):
      self._sample_bg(st, x, self._pending_reset)
    bg = st["b"][:, :, None, None].expand_as(x)
    # 取较小值进行遮挡合成：处于背景后方的物体不可见
    return torch.minimum(x, bg)

  def _sample_bg(self, st: dict, x: torch.Tensor, mask: torch.Tensor) -> None:
    """在重置时为各个环境随机采样背景深度的上下边界梯度."""
    n, h = st["b"].shape
    lo, hi = self._bg_depth_range
    m = int(mask.sum())
    if m == 0:
      return
    ends = lo + (hi - lo) * torch.rand(m, 2, device=x.device, dtype=x.dtype)
    rows = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype)
    b = ends[:, 0:1] * (1.0 - rows) + ends[:, 1:2] * rows
    off = torch.rand(m, 1, device=x.device, dtype=x.dtype) >= self._bg_fill_prob
    st["b"][mask] = torch.where(off, x.new_full((m, h), self._far), b)

  def _ground_far_out(self, raw: torch.Tensor, env, sensor_name: str | None) -> torch.Tensor:
    """根据物理几何体的平面类型，将平坦地面的像素随机置为远端 far."""
    if self._ground_far_prob <= 0.0 or env is None or sensor_name is None:
      return raw
    seg = getattr(env.scene[sensor_name].data, "segmentation", None)
    if seg is None:
      return raw
    if self._plane_ids is None:
      gtype = env.sim.mj_model.geom_type
      plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
      ids = [i for i in range(len(gtype)) if int(gtype[i]) == plane]
      self._plane_ids = torch.tensor(ids, device=seg.device, dtype=seg.dtype)
    geom_obj = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    is_floor = geom_obj & torch.isin(seg[..., 0], self._plane_ids)
    blind = is_floor & (torch.rand_like(raw[..., 0]) < self._ground_far_prob)
    return torch.where(blind.unsqueeze(-1), torch.full_like(raw, self._far), raw)

  def _border_weight(self, h: int, w: int, device, dtype) -> torch.Tensor:
    """计算图像边界权重掩码，越靠近视野边缘权重越大 (模拟双目重叠区减少导致的视差失效)."""
    if self._border_w is not None and self._border_w.shape == (h, w):
      return self._border_w
    ii = torch.arange(h, device=device, dtype=dtype).view(h, 1).expand(h, w)
    jj = torch.arange(w, device=device, dtype=dtype).view(1, w).expand(h, w)
    di = torch.minimum(ii, (h - 1) - ii)
    dj = torch.minimum(jj, (w - 1) - jj)
    edge_dist = torch.minimum(di / max(h - 1, 1), dj / max(w - 1, 1))
    bw = 1.0 - edge_dist
    bw = bw / (bw.mean() + 1e-6)
    self._border_w = bw
    return bw

  def _spawn_clutter(self, mask: torch.Tensor) -> None:
    """在环境重置时生成室内静态大尺寸杂物矩形斑块 (Clutter Blobs)."""
    idx = mask.nonzero(as_tuple=True)[0]
    m = int(idx.numel())
    if m == 0:
      return
    k, dev = self._clutter_n, self._cl_cy.device

    def U(a: float, b: float) -> torch.Tensor:
      return a + (b - a) * torch.rand(m, k, device=dev)

    def edge_coord() -> torch.Tensor:
      d = (torch.rand(m, k, device=dev) ** (1.0 + self._clutter_edge_bias)) * 0.5
      side = torch.rand(m, k, device=dev) < 0.5
      return torch.where(side, d, 1.0 - d)

    self._cl_cy[idx] = edge_coord()
    self._cl_cx[idx] = edge_coord()
    self._cl_oy[idx] = 0.0
    self._cl_ox[idx] = 0.0
    self._cl_od[idx] = 0.0
    self._cl_hy[idx] = U(self._clutter_size_min, self._clutter_size_max)
    self._cl_hx[idx] = U(self._clutter_size_min, self._clutter_size_max)
    self._cl_d[idx] = U(self._clutter_depth_min, self._clutter_depth_max)
    self._cl_vy[idx] = U(-self._clutter_drift, self._clutter_drift)
    self._cl_vx[idx] = U(-self._clutter_drift, self._clutter_drift)
    has = (torch.rand(m, 1, device=dev) < self._clutter_prob).float()
    self._cl_on[idx] = has * (torch.rand(m, k, device=dev) < self._clutter_blob_prob).float()

  def _advance_clutter(self) -> None:
    """单步更新静态杂物的微小自运动扰动 (Jitter)."""
    k = self._clutter_n
    if k <= 0:
      return
    n, dev = self.num_envs, self.device
    if self._cl_cy is None:
      z = torch.zeros(n, k, device=dev)
      self._cl_cy, self._cl_cx = z.clone(), z.clone()
      self._cl_hy, self._cl_hx = z.clone(), z.clone()
      self._cl_d, self._cl_vy, self._cl_vx, self._cl_on = z.clone(), z.clone(), z.clone(), z.clone()
      self._cl_oy, self._cl_ox, self._cl_od = z.clone(), z.clone(), z.clone()
      self._spawn_clutter(torch.ones(n, dtype=torch.bool, device=dev))
      return
    if self._pending_reset is not None and bool(self._pending_reset.any()):
      self._spawn_clutter(self._pending_reset)
    if self._clutter_jitter > 0.0:
      mx = self._clutter_jitter_max
      self._cl_oy = (self._cl_oy + torch.randn_like(self._cl_oy) * self._clutter_jitter).clamp(-mx, mx)
      self._cl_ox = (self._cl_ox + torch.randn_like(self._cl_ox) * self._clutter_jitter).clamp(-mx, mx)
    if self._clutter_depth_jitter > 0.0:
      dmx = self._clutter_depth_jitter_max
      self._cl_od = (self._cl_od + torch.randn_like(self._cl_od) * self._clutter_depth_jitter).clamp(-dmx, dmx)

  def _augment(self, x: torch.Tensor, sensor_name: str | None = None) -> torch.Tensor:
    """渲染并叠加斑点噪声、孔洞、静态杂物及薄雾 (Haze)."""
    if not (
      self._speckle_rate_max > 0.0
      or self._haze_sigma_max > 0.0
      or self._hole_rate_max > 0.0
      or self._clutter_n > 0
    ):
      return x
    near, far = self._near, self._far
    need = self._speckle_rate_max > 0.0 or self._haze_sigma_max > 0.0 or self._hole_rate_max > 0.0
    st = None
    if need:
      key = sensor_name or "_"
      st = self._aug_state.get(key)
      if st is None or st.get("shape") != tuple(x.shape):
        st = {"shape": tuple(x.shape)}
        self._aug_state[key] = st
        self._sample_aug_candidates(
          st, x, torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        )
        self._blink_aug(st)
      else:
        if self._pending_reset is not None and bool(self._pending_reset.any()):
          self._sample_aug_candidates(st, x, self._pending_reset)
        if self._aug_do_refresh:
          self._blink_aug(st)
      if "spk_on" in st:
        if "f_spk" in st:
          x = torch.where(st["spk_on"], x * st["f_spk"], x)
        else:
          x = torch.where(st["spk_on"], torch.minimum(x, st["d_spk"]), x)
    if self._clutter_n > 0 and self._cl_cy is not None:
      h, w = x.shape[1], x.shape[2]
      yg = ((torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / h).view(1, 1, h, 1)
      xg = ((torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) / w).view(1, 1, 1, w)
      depth_m = (near + (far - near) * self._cl_d + self._cl_od).clamp(near, far)
      tan_v = math.tan(math.radians(self._clutter_cam_vfov_deg) * 0.5)
      tan_h = tan_v * (w / h)
      ny = self._cl_oy / (2.0 * depth_m * tan_v)
      nx = self._cl_ox / (2.0 * depth_m * tan_h)
      cy = (self._cl_cy + ny).clamp(0.0, 1.0)[..., None, None]
      cx = (self._cl_cx + nx).clamp(0.0, 1.0)[..., None, None]
      hy, hx = self._cl_hy[..., None, None], self._cl_hx[..., None, None]
      inside = (
        ((yg - cy).abs() <= hy) & ((xg - cx).abs() <= hx) & (self._cl_on[..., None, None] > 0.5)
      )
      d_m = depth_m[..., None, None]
      blob = torch.where(inside, d_m.expand_as(inside.float()), x.new_full((), far)).amin(dim=1)
      x = torch.minimum(x, blob.unsqueeze(-1))
    if st is not None:
      if "haze" in st:
        x = (x + st["haze"] * (x / far)).clamp(min=near, max=far)
      if "hole_on" in st:
        x = torch.where(st["hole_on"], torch.full_like(x, far), x)
    return x

  def _sample_aug_candidates(self, st: dict, x: torch.Tensor, mask: torch.Tensor) -> None:
    """采样固定的噪声像素候选坐标及深度偏差."""
    near, far = self._near, self._far
    b, h, w = x.shape[0], x.shape[1], x.shape[2]
    dev, dt = x.device, x.dtype
    if "cand" not in st:
      st["cand"] = torch.zeros(b, h, w, 1, dtype=torch.bool, device=dev)
      if self._speckle_keep_min >= 0.0:
        st["f_spk"] = torch.zeros(b, h, w, 1, device=dev, dtype=dt)
      else:
        st["d_spk"] = torch.zeros(b, h, w, 1, device=dev, dtype=dt)
      st["hole_cand"] = torch.zeros(b, h, w, 1, dtype=torch.bool, device=dev)
      st["haze"] = torch.zeros(b, h, w, 1, device=dev, dtype=dt)
    idx = mask.nonzero(as_tuple=True)[0]
    m = int(idx.numel())
    if m == 0:
      return
    xs = x[idx]
    if self._speckle_rate_max > 0.0:
      rate = torch.rand(m, 1, 1, 1, device=dev, dtype=dt) * self._speckle_rate_max
      xm = xs[..., 0]
      wgt = torch.ones_like(xm)
      if self._speckle_edge_bias > 0.0:
        gx = torch.zeros_like(xm)
        gx[:, :, 1:] = (xm[:, :, 1:] - xm[:, :, :-1]).abs()
        gy = torch.zeros_like(xm)
        gy[:, 1:, :] = (xm[:, 1:, :] - xm[:, :-1, :]).abs()
        edge = gx + gy
        edge = edge / (edge.mean(dim=(1, 2), keepdim=True) + 1e-6)
        wgt = wgt + self._speckle_edge_bias * (edge - 1.0)
      if self._speckle_border_bias > 0.0:
        bw = self._border_weight(h, w, dev, dt)
        wgt = wgt + self._speckle_border_bias * (bw.unsqueeze(0) - 1.0)
      if self._speckle_far_bias > 0.0:
        fw = ((xm - near) / (far - near)).clamp(0.0, 1.0)
        fw = fw / (fw.mean(dim=(1, 2), keepdim=True) + 1e-6)
        wgt = wgt + self._speckle_far_bias * (fw - 1.0)
      wgt = wgt.clamp_min(0.0)
      wgt = wgt / (wgt.mean(dim=(1, 2), keepdim=True) + 1e-6)
      prob = (rate * wgt.unsqueeze(-1)).clamp(0.0, 1.0)
      st["cand"][idx] = torch.rand_like(xs) < prob
      if self._speckle_keep_min >= 0.0:
        km = self._speckle_keep_min
        st["f_spk"][idx] = km + (1.0 - km) * torch.rand_like(xs)
      else:
        st["d_spk"][idx] = near + (far - near) * torch.rand_like(xs)
    if self._hole_rate_max > 0.0:
      hr = torch.rand(m, 1, 1, 1, device=dev, dtype=dt) * self._hole_rate_max
      st["hole_cand"][idx] = torch.rand_like(xs) < hr
    if self._haze_sigma_max > 0.0:
      s = torch.rand(m, 1, 1, 1, device=dev, dtype=dt) * self._haze_sigma_max
      st["haze"][idx] = torch.randn_like(xs) * s

  def _blink_aug(self, st: dict) -> None:
    """刷新哪些固定的候选噪点在当前周期被激活（形成闪烁效应而非位置跳变）."""
    keep = self._flicker_keep
    if "cand" in st:
      ref = st["f_spk"] if "f_spk" in st else st["d_spk"]
      st["spk_on"] = st["cand"] & (torch.rand_like(ref) < keep)
    if "hole_cand" in st:
      st["hole_on"] = st["hole_cand"] & (torch.rand_like(st["haze"]) < keep)

  def _raw_depth(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """提取传感器原始测距数据 (基类直接透传)."""
    return env.scene[sensor_name].data.depth

  def _frame(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """处理当前时间步的相机图像，支持多传感器垂直拼合或通道融合."""
    frames = [self._one(self._raw_depth(env, s), env, s) for s in self._sensors]
    if self._fuse == "vertical" and len(frames) > 1:
      img = torch.cat([f[..., 0] for f in frames], dim=1)
      if self._flatten:
        return img.reshape(img.shape[0], -1).contiguous()
      return img.unsqueeze(1).contiguous()
    if self._flatten:
      return torch.cat([f.reshape(f.shape[0], -1) for f in frames], dim=-1).contiguous()
    return torch.stack([f[..., 0] for f in frames], dim=1).contiguous()

  def reset(self, env_ids=None):
    """在环境重置时标记需要刷新的历史缓存."""
    if self._buf is None:
      return {}
    if self._pending_reset is None:
      self._pending_reset = torch.zeros(
        self._buf.shape[0], dtype=torch.bool, device=self._buf.device
      )
    if env_ids is None:
      self._pending_reset[:] = True
    else:
      self._pending_reset[env_ids] = True
    return {}

  def __call__(self, env: ManagerBasedRlEnv, **_kwargs) -> torch.Tensor:
    """单步调用：更新噪点闪烁周期、推进杂物、推入 GPU 环形缓冲区并按指定时延切片拼接."""
    self._aug_do_refresh = (self._aug_step % self._aug_update_period) == 0
    self._aug_step += 1
    self._advance_clutter()
    frame = self._frame(env)
    n = frame.shape[0]
    
    # 模拟相机帧率低于控制频率的保持特性
    if self._update_period > 1:
      if self._held is None:
        self._held = frame.clone()
      if self._step % self._update_period == 0:
        self._held.copy_(frame)
      if self._pending_reset is not None and bool(self._pending_reset.any()):
        rm = self._pending_reset
        self._held[rm] = frame[rm]
      self._step += 1
      cur = self._held
    else:
      cur = frame

    # 维护 GPU 环形缓冲区，拼接历史多帧观测
    if self._buf is None:
      self._buf = cur.new_zeros(n, self._L, *cur.shape[1:])
      self._buf[:] = cur.unsqueeze(1)
      self._head = 0
    else:
      if self._pending_reset is not None and bool(self._pending_reset.any()):
        m = self._pending_reset
        self._buf[m] = cur[m].unsqueeze(1)
        self._pending_reset[:] = False
      self._head = (self._head + 1) % self._L
      self._buf[:, self._head] = cur
    idx = [(self._head - k) % self._L for k in self._frame_offsets]
    sel = self._buf[:, idx]
    if self._flatten:
      return sel.reshape(n, -1).contiguous()
    return sel.reshape(n, -1, sel.shape[-2], sel.shape[-1]).contiguous()


class BallOnlyDepthObs(DepthImageObs):
  """专属纯球体掩码深度观测项 (Ball-Only Masked Depth 16x9).

  【Sim2Real 核心桥梁】
  在实机部署中，由机载端侧视觉模型（如 EfficientTAM 或 YOLO-World）提取球体 Mask，
  非球体区域全部置为 far（远端空旷）。这样策略网络看到的仅仅是纯净的“黑夜中逼近的球”，
  彻底隔绝了现实复杂背景环境对策略的视觉干扰。

  特性与增强：
  - 读取 MuJoCo 传感器语义分割通道 (`segmentation`)；
  - 仅保留球体几何体像素 (`ball_geom_name`) 的测距，其余像素强置为 `far`；
  - 训练期领域随机化：
    * `ball_full_dropout_prob`: 整球丢失概率（模拟跟踪算法暂时丢帧）；
    * `ball_pixel_dropout`: 像素级空洞丢失；
    * `ball_depth_jitter`: 整球测距常数级偏移误差；
    * `edge_tile_flicker`: 球体外缘一圈网格像素的边缘闪烁（模拟低分辨率下池化离散抖动）。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    p = cfg.params
    self._ball_geom_name: str = str(p.get("ball_geom_name", "ball/ball_collision"))
    self._ball_full_dropout_prob: float = float(p.get("ball_full_dropout_prob", 0.0))
    self._ball_pixel_dropout: float = float(p.get("ball_pixel_dropout", 0.0))
    self._ball_depth_jitter: float = float(p.get("ball_depth_jitter", 0.0))
    self._ball_noise_sigma: float = float(p.get("ball_noise_sigma", 0.0))
    self._edge_tile_flicker: float = float(p.get("edge_tile_flicker", 0.0))
    self._false_positive_rate: float = float(p.get("false_positive_rate", 0.0))
    self._ball_gid: int | None = None

  def _raw_depth(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """提取原始深度并通过语义分割图将非球体区域全部掩膜为 far."""
    sensor = env.scene[sensor_name]
    raw = sensor.data.depth  # [B, H, W, 1] 米
    seg = sensor.data.segmentation  # [B, H, W, 2] 语义分割图 (第 0 通道为 geom_id)
    if seg is None:
      raise RuntimeError(
        f"BallOnlyDepthObs 需要启用分割通道；传感器 '{sensor_name}' 必须渲染分割数据。"
      )
    if self._ball_gid is None:
      self._ball_gid = int(env.sim.mj_model.geom(self._ball_geom_name).id)

    far = torch.full_like(raw, self._far)
    # 提取球体像素掩码
    ball = seg[..., 0:1] == self._ball_gid

    # 随机化：模拟整个球体被视觉追踪器丢失
    if self._ball_full_dropout_prob > 0.0:
      keep_env = torch.rand(raw.shape[0], 1, 1, 1, device=raw.device) >= self._ball_full_dropout_prob
      ball = ball & keep_env
    # 随机化：像素级残缺
    if self._ball_pixel_dropout > 0.0:
      ball = ball & (torch.rand_like(raw) >= self._ball_pixel_dropout)

    # 核心掩膜：球体位置填入真实几何距离，其余全填 far
    d = torch.where(ball, raw, far)

    # 随机化：球体整体系数偏移
    if self._ball_depth_jitter > 0.0:
      off = (torch.rand(raw.shape[0], 1, 1, 1, device=raw.device) * 2.0 - 1.0) * self._ball_depth_jitter
      d = torch.where(ball, d + off, d)

    # 随机化：仅在球体表面添加测距高斯噪声
    if self._ball_noise_sigma > 0.0:
      d = torch.where(ball, d + torch.randn_like(raw) * self._ball_noise_sigma, d)

    # 随机化：模拟球体边界网格在外圈 1 像素膨胀环上的闪烁跳变
    if self._edge_tile_flicker > 0.0:
      ballf = ball.float().permute(0, 3, 1, 2)
      dilated = (F.max_pool2d(ballf, kernel_size=3, stride=1, padding=1) > 0.5).permute(0, 2, 3, 1)
      ring = dilated & (~ball)
      ball_dist = torch.where(ball, raw, torch.full_like(raw, float("inf"))).amin(dim=(1, 2), keepdim=True)
      ball_dist = torch.where(torch.isfinite(ball_dist), ball_dist, far[:, :1, :1, :])
      flick = ring & (torch.rand_like(raw) < self._edge_tile_flicker)
      d = torch.where(flick, ball_dist.expand_as(d), d)

    # 随机化：背景杂散假阳性噪点
    if self._false_positive_rate > 0.0:
      fp = (~ball) & (torch.rand_like(raw) < self._false_positive_rate)
      near = self._near + torch.rand_like(raw) * (self._far - self._near)
      d = torch.where(fp, near, d)
    return d


def dodge_ball_state_b(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  ball_name: str = "ball",
) -> torch.Tensor:
  """特权观测 (仅 Critic 可见): 机器人体坐标系下的球体真实 3D 相对位置与相对线速度 [B, 6].

  包含 [rel_pos_b (3), rel_vel_b (3)]。
  Actor 必须在真实部署时从噪声深度图中隐式估算这些状态；而 Critic 在训练时直接观测真值，
  能够极大提升值函数评估的准确性，进而提供更高信噪比的优势估计 (Advantage)。
  """
  robot: Entity = env.scene[robot_name]
  ball: Entity = env.scene[ball_name]
  yq = yaw_quat(robot.data.root_link_quat_w)
  rel_pos_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  rel_vel_w = ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
  pos_b = quat_apply_inverse(yq, rel_pos_w)
  vel_b = quat_apply_inverse(yq, rel_vel_w)
  return torch.cat([pos_b, vel_b], dim=-1)


def dodge_wall_state_b(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  wall_names: tuple[str, ...] = ("wall",),
  k: int = 1,
) -> torch.Tensor:
  """特权/环境状态观测: 机器人偏航坐标系下最近的 k 个障碍墙的相对位置与表面间距 [B, 3*k].

  针对距离最近的 k 个障碍墙，输出：
  - rel_x_b, rel_y_b: 墙体中心在机器人偏航系的相对水平坐标；
  - surf_dist: 机器人根节点到墙体 2D 矩形几何边界的最短空间外距 (>0 表示安全未碰)。
  """
  robot: Entity = env.scene[robot_name]
  yq = yaw_quat(robot.data.root_link_quat_w)
  rp = robot.data.root_link_pos_w

  if not hasattr(env, "_wall_geom_ids"):
    env._wall_geom_ids = {}
  feats = []
  dists = []
  for name in wall_names:
    if name not in env._wall_geom_ids:
      env._wall_geom_ids[name] = env.sim.mj_model.geom(f"{name}/wall_collision").id
    gid = env._wall_geom_ids[name]
    wall: Entity = env.scene[name]
    rel_w = wall.data.root_link_pos_w - rp
    pos_b = quat_apply_inverse(yq, rel_w)
    wyaw = getattr(env, "_wall_yaw_w", None)
    wi = wall_names.index(name)
    if wyaw is not None:
      w_yaw = wyaw[:, wi]
    else:
      w_yaw = torch.zeros(env.num_envs, device=env.device)
    c, s = torch.cos(-w_yaw), torch.sin(-w_yaw)
    lx = c * rel_w[:, 0] - s * rel_w[:, 1]
    ly = s * rel_w[:, 0] + c * rel_w[:, 1]
    hl = env.sim.model.geom_size[:, gid, 0]
    ht = env.sim.model.geom_size[:, gid, 1]
    ex = (lx.abs() - hl).clamp(min=0.0)
    ey = (ly.abs() - ht).clamp(min=0.0)
    surf = torch.sqrt(ex * ex + ey * ey)
    feats.append(torch.stack([pos_b[:, 0], pos_b[:, 1], surf], dim=-1))
    dists.append(rel_w[:, :2].norm(dim=-1))

  all_feats = torch.stack(feats, dim=1)
  if len(wall_names) > 1:
    order = torch.argsort(torch.stack(dists, dim=1), dim=1)
    all_feats = torch.gather(
        all_feats, 1, order.unsqueeze(-1).expand(-1, -1, 3)
    )
  kk = min(k, len(wall_names))
  return all_feats[:, :kk].reshape(env.num_envs, 3 * kk)


def dodge_obstacle_geometry_b(
  env: ManagerBasedRlEnv,
  robot_name: str = "robot",
  wall_names: tuple[str, ...] = ("wall",),
  k: int = 1,
) -> torch.Tensor:
  """机器人偏航系下最近的 k 个障碍物几何包围盒完整参数 [B, 9*k].

  每个障碍物包含 9 维几何特征：
  中心三维坐标 (3), 半长宽高尺寸 (3), 相对航向角正弦与余弦 (2), 到底面轮廓的水平间距 (1)。
  """
  if not 1 <= k <= len(wall_names):
    raise ValueError("k 必须介于 1 和障碍物总数之间")
  if not hasattr(env, "_obstacle_geometry_ids"):
    env._obstacle_geometry_ids = {}
  key = tuple(wall_names)
  if key not in env._obstacle_geometry_ids:
    env._obstacle_geometry_ids[key] = torch.tensor(
      [env.sim.mj_model.geom(f"{name}/wall_collision").id for name in key],
      device=env.device, dtype=torch.long,
    )
  ids = env._obstacle_geometry_ids[key]
  robot = env.scene[robot_name]
  rel_w = env.sim.data.geom_xpos[:, ids] - robot.data.root_link_pos_w[:, None, :]
  half = env.sim.model.geom_size[:, ids, :]
  rotation = env.sim.data.geom_xmat[:, ids].reshape(env.num_envs, len(key), 3, 3)
  obstacle_yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
  c, s = obstacle_yaw.cos(), obstacle_yaw.sin()
  local_xy = torch.stack((
    c * rel_w[..., 0] + s * rel_w[..., 1],
    -s * rel_w[..., 0] + c * rel_w[..., 1],
  ), dim=-1)
  surface_distance = (local_xy.abs() - half[..., :2]).clamp_min(0).norm(dim=-1)

  heading = robot.data.heading_w[:, None]
  c, s = heading.cos(), heading.sin()
  rel_b = torch.stack((
    c * rel_w[..., 0] + s * rel_w[..., 1],
    -s * rel_w[..., 0] + c * rel_w[..., 1],
    rel_w[..., 2],
  ), dim=-1)
  relative_yaw = obstacle_yaw - heading
  features = torch.cat((
    rel_b, half, relative_yaw.sin().unsqueeze(-1),
    relative_yaw.cos().unsqueeze(-1), surface_distance.unsqueeze(-1),
  ), dim=-1)
  order = surface_distance.argsort(dim=1, stable=True)[:, :k]
  nearest = features.gather(1, order.unsqueeze(-1).expand(-1, -1, 9))
  return nearest.reshape(env.num_envs, 9 * k)


def dodge_cbf_state_b(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
) -> torch.Tensor:
  """特权观测 (仅 Critic 可见): CBF 控制屏障函数与安全滤波器的内部状态 [B, 8].

  返回包含 8 维状态向量：
  `[ threat (1), h (1), tti (1), e_b (2), dodge_correction_b (2), ball_radius (1) ]`
  - threat: 是否正遭受来球逼近威胁 (0/1)；
  - h: 当前安全裕度 s - D（无威胁时截断为 0）；
  - tti: 距离平齐碰撞的剩余时间 Time-to-Impact (秒)；
  - e_b: 机器人机体系逃逸侧向单位方向 (nabla_v h)；
  - dodge_correction_b: CBF 期望的安全速度与名义速度之差 u_safe - u_nom；
  - ball_radius: 当前环境采样的球体真实物理半径。
  """
  cmd = env.command_manager.get_term(command_name)
  robot: Entity = cmd.robot
  yq = yaw_quat(robot.data.root_link_quat_w)
  threat = cmd._dodge_threat.float().unsqueeze(-1)

  def _to_b(v2: torch.Tensor) -> torch.Tensor:
    v3 = torch.cat([v2, torch.zeros_like(v2[:, :1])], dim=-1)
    return quat_apply_inverse(yq, v3)[:, :2]

  h = cmd._dodge_h.unsqueeze(-1) * threat
  tti = cmd._dodge_tti.unsqueeze(-1) * threat
  e_b = _to_b(cmd._dodge_e_w) * threat
  corr_b = _to_b(cmd._dodge_u_safe_w) - cmd.vel_command_nominal_b[:, :2]
  radius = cmd._dodge_ball_radius.unsqueeze(-1)
  return torch.cat([threat, h, tti, e_b, corr_b, radius], dim=-1)


def dodge_ball_radius_b(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
) -> torch.Tensor:
  """特权观测 (仅 Critic 可见): 当前环境采样的球体真实物理半径 (米) [B, 1]."""
  cmd = env.command_manager.get_term(command_name)
  return cmd._dodge_ball_radius.unsqueeze(-1)


class BallObservableGate(ManagerTermBase):
  """特权门控观测项：指示来球在本回合内是否曾进入过相机视野锥 [B, 1] ({0, 1}).

  设计意图：
  1. 用于辅助信念网络 (Belief Decoder) 监督损失门控：
     只有当球曾进入过视野后，才监督其对球体位置的回归推断；
     一旦球飞出视野，利用循环时序机制继续监督外推；但在球首次进入视野前不进行预测监督。
  2. 结合相机真实世界位姿与视锥几何（包含俯仰下蹲联动），实时判定视锥投影。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    p = cfg.params
    self._ball: str = p.get("ball_name", "ball")
    self._up_sensor: str = p.get("up_sensor", "head_depth_single")
    self._down_sensor: str = p.get("down_sensor", "head_depth_single")
    self._max_range: float = float(p.get("max_range", 5.0))
    self._near: float = float(p.get("near", 0.05))
    fovy = math.radians(float(p.get("fovy_deg", 54.0)))
    aspect = float(p.get("aspect", 16.0 / 9.0))
    self._tan_v = math.tan(0.5 * fovy)
    self._tan_h = self._tan_v * aspect
    self._latch = torch.zeros(self.num_envs, device=self.device)
    self._prev_p = torch.zeros(self.num_envs, 3, device=self.device)
    self._cam_ids: list[int] | None = None
    self._init = False

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    ids = slice(None) if env_ids is None else env_ids
    self._latch[ids] = 0.0
    self._prev_p[ids] = self._env.scene[self._ball].data.root_link_pos_w[ids]
    return {}

  def _in_view(self, env: ManagerBasedRlEnv, p: torch.Tensor) -> torch.Tensor:
    """计算球心世界点 p 是否位于相机真实三维视锥平截头体内部."""
    data = env.sim.data
    cam_xpos = data.cam_xpos
    cam_xmat = data.cam_xmat
    out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    for cid in self._cam_ids:
      R = cam_xmat[:, cid].reshape(-1, 3, 3)
      rel = p - cam_xpos[:, cid]
      pc = torch.einsum("bij,bj->bi", R.transpose(1, 2), rel)
      d = -pc[:, 2]
      inv = d.clamp_min(1e-6)
      nx = pc[:, 0] / (inv * self._tan_h)
      ny = pc[:, 1] / (inv * self._tan_v)
      out |= (d > self._near) & (d < self._max_range) & (nx.abs() <= 1.0) & (ny.abs() <= 1.0)
    return out

  def __call__(self, env: ManagerBasedRlEnv, **_kwargs) -> torch.Tensor:
    """每步检测视锥并维护锁定标志（新抛球瞬移时重置门控）."""
    if self._cam_ids is None:
      self._cam_ids = [
        int(env.scene[self._up_sensor].camera_idx),
        int(env.scene[self._down_sensor].camera_idx),
      ]
    p = env.scene[self._ball].data.root_link_pos_w
    if not self._init:
      self._prev_p[:] = p
      self._init = True
    jumped = torch.norm(p - self._prev_p, dim=-1) > 1.0
    self._latch = torch.where(jumped, torch.zeros_like(self._latch), self._latch)
    self._latch = torch.maximum(self._latch, self._in_view(env, p).float())
    self._prev_p = p.clone()
    return self._latch.unsqueeze(-1)