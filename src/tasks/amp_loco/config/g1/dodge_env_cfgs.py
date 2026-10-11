"""Unitree G1 人形机器人敏捷躲避球（Dodgeball）环境配置族。

任务背景与研究体系（Perception Regimes & Benchmark）：
本模块是强化学习躲避球控制器的核心环境构建集，将跳跃目标导航（Leap-to-Goal）运动控制器与
动态投掷飞球（Thrown Ball）相融合，全面评估人形机器人在极端动态威胁下的感知与全身协同躲避能力。

核心任务变体与感知机制：
1. 状态真值基准（State Oracle / MimicKit Benchmark）：
   - Actor 与 Critic 对称直接观测小球在航向坐标系下的相对 3D 位置与 3D 速度，不依赖图像感知；
   - 包含平地躲避（MimicKit-Flat）、单静态墙体避障（Wall-Flat）、无尽走廊巡航避障（WallWalk-Flat）
     以及复杂课桌椅教室穿梭避障（Classroom-Flat）；
2. 视觉深度感知基准（Depth Camera & Gimbal Camera）：
   - 头部搭载单目深度相机（分辨率 16x9 对应真实 ZED Mini 相机视角）；
   - 掩膜深度表征（Ball-Only Masked Depth）：仅保留小球区域的真实测距，背景全部截断，直接对齐真机
     端侧 EfficientTAM 实时分割模型的输出，消除视觉 Sim2Real 域间隙；
   - 云台相机（Camera Gimbal）：支持通过 `CAMERA_GIMBAL=1` 动态挂载 1-DoF 俯仰伺服云台，
     先知视线自动瞄准小球，大幅扩展视场角；
3. 动力学安全屏障（Control Barrier Functions, CBF）：
   - 支持连杆级障碍物排斥奖励（CBF-Link barrier shaping）；
   - 支持关节层级 CBF 动作安全滤波器（`CBF_JOINT=1`），在策略输出后执行二阶安全投影修正；
4. 飞球动力学抛射机制：
   - 当机器人在目标点驻留等待（Dwell in standing）或在原地站立模式下，在机器人前方扇形区域内
     随机生成小球，计算经重力补偿的抛物线初速度，使小球在设定反应窗口内精确飞向机器人躯干/骨盆区域。
"""

import os
from dataclasses import dataclass, fields, replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.utils.spec_config import CollisionCfg

import src.tasks.amp_loco.mdp as mdp
from src.assets.objects import get_ball_cfg, get_wall_cfg
from src.assets.robots import G1_ACTION_SCALE
import src.assets.robots.unitree_g1.g1_constants as gc
from src.tasks.amp_loco.mdp.goal_command import (
  DodgeGoToGoalCommandCfg,
  GoToGoalCommandCfg,
)

from .goto_env_cfgs import (
  g1_amp_leap_goto_flat_env_cfg,
  g1_amp_leap_goto_rough_env_cfg,
)

# 永久静止站立环境的比例（0.2）：20% 的环境始终被要求站立原地（速度指令为 0 且无小球投掷）
_STAND_RATIO = 0.2

# 原地投掷环境比例（0.1）：10% 的环境强制期望速度为 0，但依然会被密集投掷飞球，
# 专门训练机器人在静止站立状态下的原地下蹲、侧闪或原地跳跃闪避能力。
_INPLACE_THROW_RATIO = 0.1

# 头部深度相机的数据裁剪范围 [米]：近端 0.1m，远端 5.0m
DEPTH_NEAR = 0.1
DEPTH_FAR = 5.0

# 击中终止课程学习步数阈值：在前 200,000 个环境步内禁用小球碰撞终止，
# 让策略先平稳学习运动与跳跃平衡；超过该步数后开启碰撞终止并施加 -200 严惩
HIT_TERM_ENABLE_STEPS = 200_000.0

# 深度图分辨率设置：16x9（宽 16，高 9，展平后为 144 维向量）。
# 对应实机 ZED Mini 在 WVGA 模式下的水平视场角（约 85° H x 54° V）。
DEPTH_WIDTH = 16
DEPTH_HEIGHT = 9

# 单相机俯仰倾角（相对于水平向上仰起的度数）：默认 20.0 度
CAMERA_TILT_DEG = float(os.environ.get("CAMERA_TILT_DEG", "20.0"))

# 深度图像帧率降采样周期（控制步长单位）：
# 1 表示每步更新（50Hz）；5 表示每 5 步更新一次并保持（对应 10Hz 相机）
DEPTH_UPDATE_PERIOD = int(os.environ.get("DEPTH_UPDATE_PERIOD", "1"))

# 小球掩膜深度图的域随机化强度缩放系数
BALLONLY_DR_SCALE = float(os.environ.get("BALLONLY_DR_SCALE", "1.0"))
BALLONLY_CLUTTER_SCALE = float(os.environ.get("BALLONLY_CLUTTER_SCALE", "1.0"))
BALLONLY_CLUTTER_JITTER = float(
  os.environ.get("BALLONLY_CLUTTER_JITTER", str(0.1 * BALLONLY_CLUTTER_SCALE))
)
BALLONLY_CLUTTER_DEPTH_JITTER = float(
  os.environ.get("BALLONLY_CLUTTER_DEPTH_JITTER", str(0.03 * BALLONLY_CLUTTER_SCALE))
)

# 关节级 CBF 动作安全滤波器开关（默认关闭）
CBF_JOINT = os.environ.get("CBF_JOINT", "0") == "1"
# CBF 奖励权重与安全裕度参数
CBF_JOINT_LAMBDA_CORR = float(os.environ.get("CBF_JOINT_LAMBDA_CORR", "0.1"))
CBF_JOINT_LAMBDA_BUF = float(os.environ.get("CBF_JOINT_LAMBDA_BUF", "1.0"))
CBF_JOINT_BUFFER = float(os.environ.get("CBF_JOINT_BUFFER", "0.1"))
CBF_JOINT_BODY_BUFFER = float(os.environ.get("CBF_JOINT_BODY_BUFFER", "0.25"))
CBF_JOINT_ALPHA = float(os.environ.get("CBF_JOINT_ALPHA", "1.5"))
CBF_JOINT_DELTA_MAX = float(os.environ.get("CBF_JOINT_DELTA_MAX", "2.0"))
CBF_JOINT_T_ALERT = float(os.environ.get("CBF_JOINT_T_ALERT", "0.4"))

# 躲避球任务参考状态初始化（RSI）动捕目录：包含跳跃与敏捷回避片段（amp_dodge）
_DODGE_RESET_DIR = os.path.normpath(
  os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "assets", "motions", "g1", "amp_dodge",
  )
)

# 走廊连续巡航任务（WallWalk）RSI 动捕目录：包含跳跃、闪避以及 17 个行走/跑步片段（amp_dodge_walk）
_WALLWALK_RESET_DIR = os.path.normpath(
  os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "assets", "motions", "g1", "amp_dodge_walk",
  )
)


def _head_depth_camera_cfg(
  name: str = "head_depth",
  camera: str = "head_camera",
  width: int = DEPTH_WIDTH,
  height: int = DEPTH_HEIGHT,
  data_types: tuple[str, ...] = ("depth",),
) -> CameraSensorCfg:
  """构建头部前视深度相机传感器配置对象。

  Args:
      name: 传感器注册名称。
      camera: 机器人模型 XML 中定义的相机标签名称。
      width: 渲染深度图宽度（像素）。
      height: 渲染深度图高度（像素）。
      data_types: 输出数据类型（默认为深度通道 ("depth",)）。

  Returns:
      CameraSensorCfg: 深度相机传感器配置。
  """
  return CameraSensorCfg(
    name=name,
    camera_name=f"robot/{camera}",
    width=width,
    height=height,
    data_types=data_types,
  )


def _apply_dodge_overrides(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
  """为基础跳跃目标导航环境应用飞球生成、碰撞检测、CBF 避障指令与投掷事件。

  Args:
      cfg: 基础平地跳跃导航环境配置。
      play: 是否处于可视化演示模式。

  Returns:
      应用修改后的环境配置。
  """
  # 1. 向物理场景中动态添加小球实体（固定半径或域随机化）
  cfg.scene.entities = {**cfg.scene.entities, "ball": get_ball_cfg()}

  # 2. 将回合重置（RSI）动捕目录切换至完整 amp_dodge 数据集（含跳跃与敏捷回避姿态）
  cfg.events["init_motion_loader"].params["motion_dir"] = _DODGE_RESET_DIR
  cfg.events["reset_from_motion"].params["motion_dir"] = _DODGE_RESET_DIR

  # 3. 部署小球与机器人的接触传感器：监测小球是否碰撞到机器人骨盆子树下的任意连杆
  ball_hit_cfg = ContactSensorCfg(
    name="ball_robot_contact",
    primary=ContactMatch(mode="body", pattern="ball", entity="ball"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (ball_hit_cfg,)

  # 4. 被小球击中时的非正常终止条件：
  # 包含接触传感器触发与速度突变回退检测（防止高速飞球发生单步穿模 Tunneling 现象）
  cfg.terminations["ball_hit"] = TerminationTermCfg(
    func=mdp.ball_contact,
    params={
      "sensor_name": "ball_robot_contact",
      # 速度不连续双重保护：在 MuJoCo 理想刚体物理中除碰撞外无空气阻力，
      # 若小球在躯干 1.0m 范围内速度与纯重力自由落体产生显著偏离（>1.5 m/s），判定发生穿模碰撞
      "delta_v_threshold": 1.5,
      "hit_dist": 1.0,
      "hit_z_min": 0.3,
    },
  )

  # 5. 替换上层目标指令为具备小球 CBF 避碰过滤能力的 DodgeGoToGoalCommandCfg
  twist = cfg.commands["twist"]
  assert isinstance(twist, GoToGoalCommandCfg)
  # 演示模式下目标点固定在初始位置后方 0.5m，促使机器人保持缓慢后撤巡航步态（躲避成功率更高）；
  # 训练模式下保留随机目标点与 20% 原地站立混合比
  twist.rel_standing_envs = 0.0 if play else _STAND_RATIO
  twist.rel_inplace_throw_envs = 0.0 if play else _INPLACE_THROW_RATIO
  goto_params = {f.name: getattr(twist, f.name) for f in fields(GoToGoalCommandCfg)}
  cfg.commands["twist"] = DodgeGoToGoalCommandCfg(
    **goto_params,
    ball_name="ball",
    back_offset=0.5 if play else 0.0,
    cbf_enabled=True,
    # CBF 避障参数：基于小球预测直线轨迹输出垂直于轨迹切线的侧移躲避速度
    cbf_alpha=1.5,     # Class-K 增益系数，控制侧移响应烈度
    safe_radius=0.9,   # 横向安全期望裕度（米）
    sense_radius=4.0,  # 沿小球飞行路径的感知预警前瞻距离（米）
    z_active=0.25,     # CBF 生效的离地高度门限
  )

  # 6. 小球生成与投掷事件配置
  # 重置时将小球暂时隐藏/停泊在远离战场的安全位置
  cfg.events["reset_dodge_state"] = EventTermCfg(
    func=mdp.reset_dodge_state,
    mode="reset",
    params={"ball_name": "ball", "park_offset": (0.0, 3.0, 0.1)},
  )
  # 域随机化：小球半径在 7.5cm 至 12.5cm（直径 15-25cm）之间随机化
  cfg.events["randomize_ball_size"] = EventTermCfg(
    func=dr.geom_size,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("ball", geom_names=("ball_collision",)),
      "operation": "abs",
      "ranges": (0.075, 0.125),
      "axes": [0],
    },
  )
  # 周期性或状态触发投掷飞球事件：在机器人驻留站立稳定后发射
  play_kw = dict(rethrow_ground_height=0.15)
  cfg.events["throw_ball_on_dwell"] = EventTermCfg(
    func=mdp.throw_ball_on_dwell,
    mode="step",
    params={
      "ball_name": "ball",
      "robot_name": "robot",
      "command_name": "twist",
      "dwell_time_s": 0.5,         # 静止驻留时间达到 0.5s 触发发射
      "command_threshold": 0.1,
      "speed_threshold": 0.3,
      **play_kw,
      # 抛射起点分布在前方扇形区域（距离 2-3 米，发射高度 1.5-2.3 米，偏角 +/-25 度）
      "dist_range": (2.0, 3.0),
      "height_range": (1.5, 2.3),
      "angle_deg": 25.0,
      "flight_time_range": (0.58, 0.63), # 飞行时间反应窗口约 0.6s
      "lead_target": True,         # 提前量瞄准：考虑机器人当前运动速度，瞄准未来拦截点
      "aim_noise_scale": 0.1,      # 瞄准点高斯抖动噪声尺度
      # 混合投掷模式：50% 概率发射平抛/下落球，50% 概率发射低位起抛的高抛球（迫使机器人下蹲闪避）
      "high_throw_fraction": 0.5,
      "high_launch_height_range": (0.4, 0.9),
      "high_target_z_range": (0.9, 1.3),
      "skip_standing_envs": True,  # 跳过永久站立的参考环境
    },
  )

  # 演示模式下小幅收紧高抛到达高度以保证更清晰的下蹲展示效果
  if play:
    cfg.events["throw_ball_on_dwell"].params["high_target_z_range"] = (0.9, 1.1)

  return cfg


def g1_amp_dodge_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构建 Unitree G1 平地基础躲避球任务环境配置。"""
  cfg = g1_amp_leap_goto_flat_env_cfg(play=play)
  return _apply_dodge_overrides(cfg, play)




# Full-strength (scale=1.0) single-camera depth sim2real augmentation. Used by
# `_add_depth_obs_single` (training) and `enable_depth_aug_preview` (the play preview that lets you
# eyeball the augmentation live). See `DepthImageObs._augment` / `_ground_far_out` for the semantics.
_DEPTH_AUG_FULL: dict = {
  # Base D435/ZED gaussian + dropout.
  "noise_sigma_base": 0.01,
  "noise_sigma_per_m": 0.01,
  "dropout_rate": 0.01,
  # Per-frame flickering speckle, far-field haze, invalid->far holes. The real ZED speckles in
  # low-texture FREE SPACE (open air), not on the close ground, and with moderate values -- so we bias
  # speckle toward FAR/empty pixels (speckle_far_bias) instead of depth edges (edge_bias=0; the edge
  # bias actually starved the uniform far field of speckle and piled it on the ground), and make the
  # spurious depth a RELATIVE perturbation depth*U(keep_min,1) (speckle_keep_min) so a free-space
  # fleck reads a moderate fraction of its true range rather than a phantom slammed to `near`.
  "speckle_rate_max": 0.15,
  "speckle_edge_bias": 0.0,
  "speckle_border_bias": 0.2,
  "speckle_far_bias": 1.0,
  "speckle_keep_min": 0.6,
  "haze_sigma_max": 0.05,
  "hole_rate_max": 0.05,
  # Coherent clutter: SMALL persistent+drifting blobs pinned to the frame EDGES/corners (matching the
  # real ZED feed -- far field clean, junk only at the periphery). Persist across the frame stack so
  # the policy must use the ball's looming, not proximity. Half the envs stay fully clean.
  "clutter_blobs": 3,
  "clutter_prob": 0.5,
  "clutter_edge_bias": 1.5,
  "clutter_size_min": 0.04,
  "clutter_size_max": 0.14,
  "clutter_depth_min": 0.1,
  "clutter_depth_max": 0.95,
  "clutter_drift": 0.02,
  # Ground blindness: far-out floor pixels (read from the segmentation map: plane-type geoms) so the
  # policy never leans on a ground cue a real ZED can't deliver at a grazing angle.
  "ground_far_prob": 0.9,
  # BACKGROUND FILL (the room sim2real fix; hardware forensics 2026-06-09): with prob 0.8 an env's
  # far/no-hit pixels read a per-env random vertical-gradient "room shell" (b_top, b_bottom ~
  # U(2, 5) m, lerped over rows, held all episode) instead of far -- a real room's ceiling/walls
  # sit inside the 5 m clip and an empty-arena-trained policy reads them as a wall-sized incoming
  # ball. Static + non-looming, so only the ball's depth-collapse remains a dodge cue. The other
  # 20% of envs keep the empty-arena background as an anchor. NOT scaled by DEPTH_DR_SCALE
  # (scene content, not noise intensity). See DepthImageObs._bg_fill.
  # THIRD attempt (2026-06-10): (2,5) coverage restored -- the deployment room's
  # content sits at 1.6-3.5 m (mid rows median 2.38 m), so a (4,5) fill trains
  # nothing the room actually shows. What changed vs the failed (2,5)@0.8 run:
  # the fill now composites OCCLUDING (min(b, x), see DepthImageObs._bg_fill) --
  # the ball pops into view exactly when nearer than its background, deployment
  # optics -- and the camera is back at +20 deg, restoring the close/low late
  # window where the ball always out-contrasts a >=2 m background. Benchmark
  # attribution of the failure: bench A/B 39/40% with DOWN 0% both ways = the
  # DOWN skill was never learned (early window contrast-erased by the fill, late
  # window cut by the +25 deg tilt). prob 0.5 keeps a strong clean-arena anchor.
  "bg_fill_prob": 0.0,
  "bg_depth_range": (2.0, 5.0),
}

# 全强度单相机深度图 Sim2Real 域随机化参数字典。
# 包含高斯噪声、随机丢弃（Dropout）、离散斑点噪声（Speckle）、远场薄雾（Haze）、无效孔洞（Holes）、
# 边缘假阳性静态聚类（Clutter Blobs）以及地面远距离截断（Ground Far-out）。
_DEPTH_AUG_FULL: dict = {
  # 基础高斯深度测量噪声标准差
  "noise_sigma_base": 0.01,
  "noise_sigma_per_m": 0.01,
  "dropout_rate": 0.01,
  # 真实 ZED Mini 相机在低纹理开阔空间中出现的随机斑点噪声与孔洞
  "speckle_rate_max": 0.15,
  "speckle_edge_bias": 0.0,
  "speckle_border_bias": 0.2,
  "speckle_far_bias": 1.0,
  "speckle_keep_min": 0.6,
  "haze_sigma_max": 0.05,
  "hole_rate_max": 0.05,
  # 连贯杂波/静态斑块：模拟视野边缘持久存在且轻微漂移的近距离伪影，迫使策略学会依赖小球的
  # “快速逼近扩散”（Looming）而非单纯凭借静态近距离深度触发躲避动作
  "clutter_blobs": 3,
  "clutter_prob": 0.5,
  "clutter_edge_bias": 1.5,
  "clutter_size_min": 0.04,
  "clutter_size_max": 0.14,
  "clutter_depth_min": 0.1,
  "clutter_depth_max": 0.95,
  "clutter_drift": 0.02,
  # 地面盲区模拟（Ground Blindness）：掠射角下的地面无法有效反射红外结构光，自动填充为远端截断值
  "ground_far_prob": 0.9,
  # 背景房间墙面模拟填充
  "bg_fill_prob": 0.0,
  "bg_depth_range": (2.0, 5.0),
}

# 允许通过环境变量 DEPTH_DR_SCALE 进行线性缩放的噪声项键名
_DEPTH_AUG_SCALABLE_KEYS = (
  "noise_sigma_base", "noise_sigma_per_m", "dropout_rate",
  "speckle_rate_max", "haze_sigma_max", "hole_rate_max",
  "clutter_prob", "clutter_blobs",
)


def _scaled_depth_aug(scale: float) -> dict:
  """根据给定的缩放系数生成调整后的深度域随机化参数字典。

  Args:
      scale: 噪声缩放因子（0 为无噪声纯净图像，1 为全强度随机化）。
  """
  out = dict(_DEPTH_AUG_FULL)
  for k in _DEPTH_AUG_SCALABLE_KEYS:
    scaled = out[k] * scale
    out[k] = max(0, round(scaled)) if k == "clutter_blobs" else scaled
  return out


DEPTH_DR_SCALE = float(os.environ.get("DEPTH_DR_SCALE", "1.0"))
# 单目头部相机深度随机化最终实际采用的关键字参数
_DEPTH_SINGLE_AUG_KWARGS: dict = _scaled_depth_aug(DEPTH_DR_SCALE)


def enable_depth_aug_preview(cfg: ManagerBasedRlEnvCfg) -> None:
  """在交互式演示/评测（Play）环境中启用单目深度相机的 Sim2Real 域随机化。

  使得可视化工具（如 Viser）中显示的深度图像与训练时策略所见带有噪声的观测完全一致。
  """
  if cfg.observations["depth"].terms["head_depth"].func is mdp.BallOnlyDepthObs:
    return
  sensors = []
  for s in cfg.scene.sensors or ():
    if s.name == "head_depth_single" and "segmentation" not in s.data_types:
      s.data_types = tuple(s.data_types) + ("segmentation",)
    sensors.append(s)
  cfg.scene.sensors = tuple(sensors)
  term = cfg.observations["depth"].terms["head_depth"]
  term.params = {**term.params, **_DEPTH_SINGLE_AUG_KWARGS}


def _set_single_camera_tilt(cfg: ManagerBasedRlEnvCfg, tilt_deg: float) -> None:
  """在模型编译前，通过包装机器人实体的 spec_fn 动态重写单目相机的俯仰安装倾角。

  Args:
      cfg: 环境配置对象。
      tilt_deg: 向上仰起的目标角度（度）。
  """
  import math

  if abs(tilt_deg - 20.0) < 1e-9:
    return  # 若为默认的 20 度则不执行修改
  robot_cfg = cfg.scene.entities["robot"]
  orig_spec_fn = robot_cfg.spec_fn
  s, c = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))

  def spec_fn():
    spec = orig_spec_fn()
    for cam in spec.cameras:
      if cam.name == "head_camera_single":
        cam.alt.xyaxes = [0.0, -1.0, 0.0, -s, 0.0, c]
    return spec

  robot_cfg.spec_fn = spec_fn
  print(f"[dodge] 头部相机俯仰倾角已覆盖重设为: +{tilt_deg:g} 度 "
        f"(垂直视场范围覆盖: {tilt_deg-27:g} ~ {tilt_deg+27:g} 度)。请确保与真机物理安装一致！")


def _add_depth_obs_single(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """将单目头部深度相机（head_camera_single）接入展平的 `depth` 观测组。

  分辨率 16x9 = 144 维向量，直接作为一维特征拼接输入策略 MLP 网络。
  """
  _set_single_camera_tilt(cfg, CAMERA_TILT_DEG)
  sensors = tuple(
    s for s in (cfg.scene.sensors or ())
    if s.name not in ("head_depth", "head_depth_single")
  )
  _single_data_types = ("depth",) if play else ("depth", "segmentation")
  cfg.scene.sensors = sensors + (
    _head_depth_camera_cfg(
      name="head_depth_single", camera="head_camera_single", data_types=_single_data_types
    ),
  )
  depth_noise = {} if play else dict(_DEPTH_SINGLE_AUG_KWARGS)
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "head_depth": ObservationTermCfg(
        func=mdp.DepthImageObs,
        params={
          "sensor_name": "head_depth_single",
          "near": DEPTH_NEAR,
          "far": DEPTH_FAR,
          "flatten": True,
          "update_period": DEPTH_UPDATE_PERIOD,
          **depth_noise,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )


def _add_depth_obs_single_ballonly(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """配置专用的“球体掩膜深度观测”（BallOnlyDepthObs）。

  表征设计（直接对接 EfficientTAM 分割网络）：
  除了被语义分割判定为小球的像素保留真实测距外，视野中其余所有背景像素全部截断为远距离常数 `far`。
  该表征具有极佳的 Sim2Real 迁移特性，将现实世界复杂的视觉背景（光照变化、人群走动、复杂家具纹理）
  在感知层彻底过滤，消除端到端视觉策略在背景泛化上的脆弱性。

  专属域随机化（Ball-specific DR）：
  - 小球偶发性整体丢失（Whole-ball Dropout）；
  - 小球边缘毛刺与部分像素缺失（Ragged Mask & Pixel Dropout）；
  - 小球深度整体抖动与单像素噪声（Ball Depth Jitter & Noise）；
  - 静态背景假阳性虚假斑块（Static False-Positive Blobs）：训练策略仅对“急速靠近扩张”（Looming）的小球
    做出躲避反应，避免因背景中静态近距离物体产生惊跳抖动。
  """
  _set_single_camera_tilt(cfg, CAMERA_TILT_DEG)
  sensors = tuple(
    s for s in (cfg.scene.sensors or ())
    if s.name not in ("head_depth", "head_depth_single")
  )
  cfg.scene.sensors = sensors + (
    _head_depth_camera_cfg(
      name="head_depth_single", camera="head_camera_single",
      data_types=("depth", "segmentation")
    ),
  )
  _aug_on = (not play) or os.environ.get("BALLONLY_AUG", "0") == "1"
  _s = BALLONLY_DR_SCALE
  ball_dr = {
    "ball_full_dropout_prob": 0.02 * _s,   # 小球整体短暂丢失概率（模拟跟踪丢帧）
    "ball_pixel_dropout": 0.10 * _s,       # 掩膜边缘像素缺失率
    "ball_depth_jitter": 0.15 * _s,        # 小球整体测距偏移抖动（+/- 米）
    "ball_noise_sigma": 0.03 * _s,         # 小球自身区域的高斯深度噪声
    "edge_tile_flicker": 0.15 * _s,        # 边缘分块闪烁率
    # 静态假阳性斑块：模拟分割器误将某个静态物体识别为小球，持续整个回合且不发生 Looming 逼近
    "clutter_blobs": 3,
    "clutter_prob": 0.6,
    "clutter_blob_prob": 0.5,
    "clutter_drift": 0.0,
    "clutter_edge_bias": 0.0,
    "clutter_size_min": 0.04,
    "clutter_size_max": 0.12,
    "clutter_depth_min": 0.1 + 0.4 * (1.0 - BALLONLY_CLUTTER_SCALE),
    "clutter_depth_max": 0.9,
    # 机器人自身运动引发的假阳性物体相对视差抖动
    "clutter_jitter": BALLONLY_CLUTTER_JITTER,
    "clutter_jitter_max": 1.0 * BALLONLY_CLUTTER_SCALE,
    "clutter_depth_jitter": BALLONLY_CLUTTER_DEPTH_JITTER,
    "clutter_depth_jitter_max": 0.2 * BALLONLY_CLUTTER_SCALE,
  } if _aug_on else {}
  depth_noise = {}
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "head_depth": ObservationTermCfg(
        func=mdp.BallOnlyDepthObs,
        params={
          "sensor_name": "head_depth_single",
          "near": DEPTH_NEAR,
          "far": DEPTH_FAR,
          "flatten": True,
          "ball_geom_name": "ball/ball_collision",
          "update_period": DEPTH_UPDATE_PERIOD,
          **ball_dr,
          **depth_noise,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )


def _add_ball_state_obs(cfg: ManagerBasedRlEnvCfg) -> None:
  """Wire MimicKit's dodgeball observation into an actor-visible ``ball_state`` group.

  MimicKit's ``compute_dodgeball_observations`` is the ball's position + velocity relative
  to the character, expressed in its heading (yaw) frame -- exactly ``dodge_ball_state_b``
  ([B, 6]). For the MimicKit-observation task this group is given to BOTH the actor and the
  critic (MimicKit's PPO is symmetric: actor and critic share the obs), so the actor is
  handed clean ground-truth ball state instead of inferring it from depth. ``history_length
  =0`` -- pos+vel already encode the ball's motion (MimicKit uses a single frame too).
  """
  cfg.observations["ball_state"] = ObservationGroupCfg(
    terms={
      "ball_state": ObservationTermCfg(
        func=mdp.dodge_ball_state_b,
        params={"robot_name": "robot", "ball_name": "ball"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )


def _apply_mimickit_reward(cfg: ManagerBasedRlEnvCfg, play: bool) -> ManagerBasedRlEnvCfg:
  """配置对齐 MimicKit/SMP 标准基准的躲避球奖励函数体系与课程机制。

  三项核心对齐设定：
  1. 奖励塑形（Reward Shaping）：
     - 采用 MimicKit 原生躲避奖励 `mimickit_dodge`（远离小球奖励 + 稳定静止奖励）；
     - 移除跳跃目标导航项，将速度跟踪奖励缩减为轻量平稳支撑（权重 0.5）；
     - 引入威胁门控（Threat-gated）静止与防抖奖励（`dodge_stillness_when_safe`, `dodge_action_rate_when_safe`）：
       在没有小球来袭的和平期严惩不必要的虚假惊跳，来球时门控归零不干扰真实躲避动作；
     - 连杆级控制屏障函数奖励（`dodge_link_cbf`）：针对手臂、小腿等外围肢体施加排斥屏障，
       彻底解决仅以骨盆为中心的避碰导致的“肢体挂彩”遗留击中问题；
  2. 终止条件判定（Immediate Hit Termination）：
     - 第 0 轮起立即开启被球击中终止判定，消除阶跃式课程门槛；
  3. 投掷节律（Intermittent Timed Throws）：
     - 采用 1 至 4 秒的随机间歇投掷机制，给机器人躲避后的平衡恢复留出时间窗口。
  """
  twist = cfg.commands["twist"]
  assert isinstance(twist, DodgeGoToGoalCommandCfg)
  # MimicKit 任务不使用指令级硬 CBF 滤波
  twist.cbf_filter_command = False

  # 原地站立等待来球，不设定远距离导航目标点
  twist.home_goal = True
  twist.back_offset = 0.0
  for term in ("goal_distance", "goal_progress", "goal_reached", "stop_at_goal", "leap_flight"):
    cfg.rewards.pop(term, None)
  for term in ("track_anchor_linear_velocity", "track_anchor_angular_velocity"):
    if term in cfg.rewards:
      cfg.rewards[term].weight = 0.5
  if "track_anchor_linear_velocity" in cfg.rewards:
    cfg.rewards["track_anchor_linear_velocity"].params["mask_inplace"] = True

  # 施加微弱的复位软拉力，抑制长时间运行下的微小前移漂移
  cfg.rewards["goal_distance"] = RewardTermCfg(
    func=mdp.goal_distance_reward,
    weight=0.3,
    params={"command_name": "twist", "std": 1.5},
  )

  # MimicKit 核心躲避奖励（小球与骨盆间距排斥项 + 速度抑制项）
  cfg.rewards.pop("dodge_cbf", None)
  cfg.rewards["mimickit_dodge"] = RewardTermCfg(
    func=mdp.mimickit_dodge_reward,
    weight=1.0,
    params={"robot_name": "robot", "ball_name": "ball"},
  )
  # 和平期基座静止奖励（小球未逼近时鼓励站定）
  cfg.rewards["dodge_stillness_when_safe"] = RewardTermCfg(
    func=mdp.dodge_stillness_when_safe,
    weight=0.5,
    params={"command_name": "twist", "robot_name": "robot", "vel_scale": 2.0},
  )
  # 和平期动作平滑性惩罚（抑制因传感器噪声引起的关节高频微小抽动）
  cfg.rewards["dodge_action_rate_when_safe"] = RewardTermCfg(
    func=mdp.dodge_action_rate_when_safe,
    weight=-0.05,
    params={"command_name": "twist"},
  )
  # 全身各连杆层级的 CBF 障碍物安全屏障奖励（约束所有肢体与小球的相对速度和距离）
  cfg.rewards["dodge_link_cbf"] = RewardTermCfg(
    func=mdp.dodge_link_cbf_reward,
    weight=0.27,
    params={
      "robot_name": "robot",
      "ball_name": "ball",
      "alpha": 1.0,
      "margin": 0.05,
      "constraint_clip": 2.0,
    },
  )
  # 全局机体侧移 CBF 惩罚项（默认置 0 禁用）
  cfg.rewards["dodge_sidestep"] = RewardTermCfg(
    func=mdp.dodge_sidestep_reward,
    weight=0.0,
    params={"robot_name": "robot", "ball_name": "ball", "alpha": 1.0, "safe_radius": 0.4},
  )

  # 若显式设定 NO_LINK_CBF=1，则关闭连杆级 CBF，还原为纯骨盆距离基线
  if os.environ.get("NO_LINK_CBF", "0") == "1":
    cfg.rewards["dodge_link_cbf"].weight = 0.0

  # 击中即刻触发终止（第 0 步开始生效）
  cfg.rewards["is_terminated"] = RewardTermCfg(
    func=mdp.is_terminated_except,
    weight=-200.0,
    params={"exclude_terms": ("ball_hit",)},
  )
  cfg.terminations["ball_hit"].params.pop("enable_after_steps", None)

  # 间歇性发射节律（随机 1.0 ~ 4.0 秒间隔）
  interval = (1.0, 4.0)
  throw = cfg.events["throw_ball_on_dwell"]
  throw.params.pop("rethrow_ground_height", None)
  throw.params["throw_interval_range"] = interval
  cfg.events["reset_dodge_state"].params["throw_interval_range"] = interval

  return cfg


def _set_slim_critic_dodge(
  cfg: ManagerBasedRlEnvCfg,
  gate_up: str = "head_depth_single",
  gate_down: str = "head_depth_single",
) -> None:
  """构建针对深度视觉任务的精简特权 Critic 观测组（critic_dodge）。

  包含：
  - `ball_state`: 小球在机体航向系下的相对 3D 位置与速度（6 维）；
  - `ball_radius`: 当前环境实例下小球的实际真实半径（1 维）；
  - `belief_gate`: 视觉置信度门控（判断小球在此次投掷中是否曾出现在相机视野内，1 维）。
  """
  cfg.observations["critic_dodge"] = ObservationGroupCfg(
    terms={
      "ball_state": ObservationTermCfg(
        func=mdp.dodge_ball_state_b,
        params={"robot_name": "robot", "ball_name": "ball"},
      ),
      "ball_radius": ObservationTermCfg(
        func=mdp.dodge_ball_radius_b,
        params={"command_name": "twist"},
      ),
      "belief_gate": ObservationTermCfg(
        func=mdp.BallObservableGate,
        params={
          "ball_name": "ball",
          "up_sensor": gate_up,
          "down_sensor": gate_down,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )


def g1_amp_dodge_mimickit_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构建 Unitree G1 状态真值（State Oracle）平地躲避球基准环境配置。

  特征：
  - Actor 与 Critic 均观测 6 维小球状态真值（相对位置与速度）；
  - 任务奖励采用 MimicKit 风格体系；
  - 支持 OMNI_THROW=1 开启 360 度全方位超高速投掷模式；
  - 支持 CBF_JOINT=1 开启关节级动作投影安全滤波器。
  """
  cfg = g1_amp_dodge_flat_env_cfg(play=play)
  cfg = _apply_mimickit_reward(cfg, play)
  _add_ball_state_obs(cfg)
  # OMNI_THROW=1 选项：支持从小球以 12-15 m/s 速度从 8-10m 外全向（360度）投掷
  if os.environ.get("OMNI_THROW", "0") == "1":
    _p = cfg.events["throw_ball_on_dwell"].params
    _p["omnidirectional"] = True
    _p["launch_speed_range"] = (
      float(os.environ.get("OMNI_SPEED_MIN", "12.0")),
      float(os.environ.get("OMNI_SPEED_MAX", "15.0")),
    )
    _p["dist_range"] = (
      float(os.environ.get("OMNI_DIST_MIN", "8.0")),
      float(os.environ.get("OMNI_DIST_MAX", "10.0")),
    )
    _p["target_z_range"] = (0.3, 1.3)
  if os.environ.get("CBF_JOINT", "0") == "1":
    _apply_cbf_joint_filter(cfg)
  return cfg


def _apply_wall_overrides(
  cfg: ManagerBasedRlEnvCfg,
  num_walls: int = 1,
  placement: str = "beside",
  recycle_ahead: bool = False,
  obs_k: int = 1,
) -> ManagerBasedRlEnvCfg:
  """向构建好的闪避环境追加静态障碍墙体（Static Wall Obstacles）。

  【环境设计与任务目标】
  机器人在规避飞来的球体时，不仅要在有限时间内躲开弹道，还必须避免在后撤或侧步时与身侧或路径周围的墙体发生碰撞：
  - 墙体侧向位置与距离随机化（Heading-Relative）：每个回合随机决定墙体位于机器人左侧或右侧（50%/50%）及横向距离。
    由于 AMP 参考动作初始化（RSI）重置时的机器人朝向并非固定的世界坐标轴（往往存在 ~-90° 的平均偏航角），
    因此墙体生成必须基于机器人重置后的机体航向坐标系（Heading-Relative），而非世界全局轴。
  - 墙体状态观测（Wall State Observation）：Actor 和 Critic 共同观测墙体状态（通过 `ball_state` 观测组中的
    `dodge_wall_state_b` 项，MimicKit Runner 会自动将其分发给 Actor 和 Critic，无需修改 Runner 配置）。
  - 静态控制屏障函数奖励（Wall-Link CBF Reward）：通过 `wall_link_cbf` 奖励项对机器人各个刚体连杆与墙体的间距进行塑形，
    在处于安全界限外时奖励为 0，一旦突破安全边界即施加屏障违例惩罚。
  - 撞墙终止与硬惩罚（Wall-Hit Termination）：与飞球碰撞（允许软接触或轻度惩罚）不同，撞墙属于不可挽回的物理事故，
    触发环境终止并施加完整的 -200 分 `is_terminated` 严厉惩罚。

  【核心参数说明】
  - num_walls: 障碍墙体数量（1 堵时命名为 "wall"，多堵时依次命名为 "wall_0", "wall_1" ...）。
  - placement: 摆放模式。
    * "beside": 放置于机器人出生点侧方（适用于原地站立闪避任务）。
    * "walk_path": 沿漫游行走走廊分散布置（适用于行进步态穿行闪避任务）。
  - recycle_ahead: 是否在仿真单步中将已经走过的墙体瞬移循环至前方（构成无尽障碍走廊）。
  - obs_k: 观测向量中包含的最近墙体数量（每堵墙体占据 3 维相对空间坐标）。

  所有超参数均支持在运行时通过环境变量覆盖（如 OMNI_THROW/CBF_JOINT），便于 Benchmark 基准评测动态调整。
  """
  ev = lambda k, d: float(os.environ.get(k, d))
  wall_names = (
    ("wall",) if num_walls == 1 else tuple(f"wall_{i}" for i in range(num_walls))
  )

  # --- 墙体实体定义（自由关节刚体盒，每步由 pin_wall 重新锁定 -> 动力学上表现为完全静态） ---
  for name in wall_names:
    cfg.scene.entities = {**cfg.scene.entities, name: get_wall_cfg()}

  # --- 墙体与机器人碰撞接触传感器（Wall<->Robot Contact Sensors，结构与 ball_robot_contact 类似） ---
  sensor_names = []
  for name in wall_names:
    sensor = f"{name}_robot_contact"
    sensor_names.append(sensor)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
      ContactSensorCfg(
        name=sensor,
        primary=ContactMatch(mode="body", pattern="wall", entity=name),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
      ),
    )

  # --- 撞墙终止条件：发生墙体接触立即终止回合，施加完整的 -200 失败惩罚 ---
  cfg.terminations["wall_hit"] = TerminationTermCfg(
    func=mdp.wall_contact,
    params={"sensor_names": tuple(sensor_names)},
  )

  # --- 仿真事件：在回合重置时随机化墙体位姿；在每步物理仿真时锁定墙体位置 ---
  # 横向（垂直于路径）墙体距离设定：
  # 行走任务走廊（walk_path）宽于原地闪避任务侧方墙体。原地任务保留 0.6-1.0m（RSI 摆臂空间下限）；
  # 行走任务放宽至 1.1-1.5m，保证相对侧墙间距 >= 2.2m，使机器人在两墙间躲避飞球时不会被迫贴墙擦撞。
  if placement == "walk_path":
    reset_lat = (ev("WALL_PATH_DIST_MIN", 1.1), ev("WALL_PATH_DIST_MAX", 1.5))
    recycle_lat = (ev("WALL_PATH_DIST_MIN", 1.1), ev("WALL_PATH_DIST_MAX", 1.6))
  else:
    reset_lat = (ev("WALL_DIST_MIN", 0.6), ev("WALL_DIST_MAX", 1.0))
    recycle_lat = (ev("WALL_DIST_MIN", 0.6), ev("WALL_DIST_MAX", 1.2))
  if placement == "walk_path":
    # 优先生成随机漫游路径（字典插入顺序 = reset 事件执行顺序），以便 reset_wall_pose 沿其放置墙体
    cfg.events["reset_walk_path"] = EventTermCfg(
      func=mdp.reset_walk_path,
      mode="reset",
      params={
        "robot_name": "robot",
        "seg_len_range": (
          ev("WALK_PATH_SEG_MIN", 2.0),
          ev("WALK_PATH_SEG_MAX", 4.0),
        ),
        "turn_max": ev("WALK_PATH_TURN_MAX", 0.45),
        "initial_len": ev("WALK_PATH_INIT_LEN", 80.0),
      },
    )
  cfg.events["reset_wall_pose"] = EventTermCfg(
    func=mdp.reset_wall_pose,
    mode="reset",
    params={
      "wall_names": wall_names,
      "robot_name": "robot",
      # 横向距离范围：既能有效约束躲避空间，又确保 RSI 重置姿态（如手臂摆动）不会直接穿模触墙
      "lateral_dist_range": reset_lat,
      "x_offset_range": (
        ev("WALL_X_OFF_MIN", -0.5),
        ev("WALL_X_OFF_MAX", 0.5),
      ),
      "placement": placement,
      # 路径模式（行走任务）：第一堵墙位于前方 2.5-4m，随后每隔 3m 放置一堵
      "path_ahead_range": (
        ev("WALL_PATH_AHEAD_MIN", 2.5),
        ev("WALL_PATH_AHEAD_MAX", 4.0),
      ),
      "path_spacing": ev("WALL_PATH_SPACING", 3.0),
    },
  )
  cfg.events["pin_wall"] = EventTermCfg(
    func=mdp.pin_wall,
    mode="step",
    params={"wall_names": wall_names},
  )
  if recycle_ahead:
    # 行走任务动态循环回收机制：当机器人走过某堵墙后，在 step 循环中将其瞬移回前方，构成无尽走廊
    cfg.events["recycle_walls_ahead"] = EventTermCfg(
      func=mdp.recycle_walls_ahead,
      mode="step",
      params={
        "wall_names": wall_names,
        "robot_name": "robot",
        "ahead_range": (
          ev("WALL_RECYCLE_AHEAD_MIN", 7.0),
          ev("WALL_RECYCLE_AHEAD_MAX", 10.0),
        ),
        "lateral_range": recycle_lat,
        "behind_margin": 1.0,
        "along_path": placement == "walk_path",
      },
    )

  # --- Actor 与 Critic 墙体感知观测：将最近墙体的相对状态追加至 ball_state 观测组 ---
  cfg.observations["ball_state"].terms["wall_state"] = ObservationTermCfg(
    func=mdp.dodge_wall_state_b,
    params={"robot_name": "robot", "wall_names": wall_names, "k": obs_k},
  )

  # --- 奖励函数：机器人各连杆对静态墙体的控制屏障函数奖励（安全区为 0，靠近时施加惩罚） ---
  cfg.rewards["wall_link_cbf"] = RewardTermCfg(
    func=mdp.wall_link_cbf_reward,
    weight=ev("WALL_CBF_WEIGHT", 0.3),
    params={
      "robot_name": "robot",
      "wall_names": wall_names,
      "alpha": ev("WALL_CBF_ALPHA", 1.0),
      "margin": 0.05,
      "constraint_clip": 2.0,
    },
  )

  # --- 禁用球体撞墙速度突变兜底检测：
  # 当飞球弹到墙上时，在盆骨附近 hit_dist（1m）内会产生很大的速度变化量 delta_v，
  # 容易引起误判判定为命中身体。此处投掷速度适中（~3-5 m/s），物理接触传感器已足够精确，
  # 因此将速度突变阈值设为 0 以关闭该误检机制。 ---
  cfg.terminations["ball_hit"].params["delta_v_threshold"] = 0.0

  return cfg


def g1_amp_dodge_mimickit_wall_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 AMP 原地闪避（状态真值 Oracle）+ 出生点侧方静态障碍墙环境配置。

  基于 ``Unitree-G1-AMP-Dodge-MimicKit-Flat`` 基础环境（包含球体状态 Actor 观测与 MimicKit 任务奖励），
  并在机器人侧方追加一面静态障碍墙体：
  - 墙体相对机体坐标系随机生成在左侧或右侧（0.6-1.0m）；
  - 墙体相对位置拼接入 Actor 和 Critic 观测中的 ``ball_state`` 组；
  - 引入各刚体连杆的静态 CBF 避障奖励（``wall_link_cbf``）；
  - 触碰墙体触发硬终止，施加完整的 -200 惩罚；
  - 旨在训练机器人“在成功规避来球的同时，绝不撞上身边的障碍物”。
  """
  cfg = g1_amp_dodge_mimickit_flat_env_cfg(play=play)
  return _apply_wall_overrides(cfg, num_walls=1, placement="beside", obs_k=1)


def g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 AMP 动态行走闪避（状态真值 Oracle）+ 沿随机曲线走廊穿行避障环境配置。

  【任务背景与架构】
  本环境继承自静态墙体闪避任务的感知与奖励基底，但核心目标从“原地站立闪避”转变为“在连续行进中穿行障碍走廊并闪避迎面飞球”：
  1. 随机漫游路径生成（Random Polyline Walk Path）：
     每个训练回合各并行环境均独立生成一条平缓弯曲的折线漫游路径（参见 mdp/walk_path.py）。
     机器人被指令驱动沿着该路径持续前进（`path_follow` 模式：基于路径纯追踪 Lookahead 目标点，
     以恒定 ~1.3 m/s 的巡航速度自主转弯导航），彻底消除了原地驻留环境（无静态站立环境，全员行进）。
  2. 障碍墙体动态走廊（Flanking Walls Recycling）：
     在回合开始时沿路径两侧分散布置 `WALLWALK_NUM_WALLS` 堵墙体（默认 3 堵，偏航角切齐路径切线，横向间距 0.9-1.3m）。
     当机器人走过某一堵墙后，单步事件自动将其回收并瞬移至前方（7.0-10.0m）的路径两侧，形成无尽连续障碍走廊。
  3. 沿路径逆向来袭飞球（Along-Path Ball Throws）：
     飞球在定时触发器（4-9s）控制下，从机器人前方走廊路径点逆向发射，沿路径走廊扑向行进中的机器人，
     考验其在行进动态步态中横向让位避让或俯仰规避后迅速回归路径的能力。
  4. 奖励机制重塑（防止策略退化为原地站立）：
     在行走任务中，机器人绝不能因静止而获得奖励。因此完全移除原地防抖静止项，并引入四大行进驱动项：
     - 全权重巡航速度锚定（`track_anchor_linear_velocity`，权重 1.0）；
     - 路径弧长推进速率（`walk_path_progress`，权重 2.0，核心驱动力，巡航时提供 ~2.6 分/步）；
     - 安全期额外推进激励与原地驻留惩罚（`walk_path_progress_when_safe` 与 `walk_path_stillness_when_safe`）；
     - 路径横向贴合度（`walk_path_adherence`，惩罚偏离走廊中线过远，防止过度横移撞墙）。
  5. 行进步态参考动作重置（Walk-Augmented RSI）：
     AMP 原生闪避动作库中无行进步态，因此切换至增强版行进动作集（`_WALLWALK_RESET_DIR`），
     确保回合重置时机器人直接处于步态行进过程中，避免在原地起步阶段失衡。
  """
  ev = lambda k, d: float(os.environ.get(k, d))
  cfg = g1_amp_dodge_mimickit_flat_env_cfg(play=play)

  # --- 回合时长设置：相比原地任务（20s）延长至 25s，适配更长的投掷间隔（4-9s），
  # 单个回合大约经历 3-6 次投掷，赋予机器人单回合约 32m 的路线行进评估距离 ---
  cfg.episode_length_s = ev("WALK_EPISODE_LENGTH_S", 25.0)

  # --- 路径跟踪行进指令配置（Path-Follow Walk Command）：
  # 每步在随机漫游路径上前瞻 lookahead（2.0m）选取纯追踪目标点，通过去往目标（Go-to-goal）
  # 航向控制器驱动机器人以恒定巡航速度（kp 1.5 * 2.0m = 3.0，截断至 max_lin_vel_x = 1.3 m/s）自主过弯。
  # 完全关闭原地/站立环境比例，强制所有机器人持续行进。 ---
  twist = cfg.commands["twist"]
  twist.home_goal = False
  twist.back_offset = 0.0
  twist.forward_offset = 0.0
  twist.path_follow = True
  twist.path_lookahead = ev("WALK_PATH_LOOKAHEAD", 2.0)
  # path_wall_ahead 必须超过墙体循环回收的前瞻上限（10m），确保回收墙体始终落在已生成的有效路径上
  twist.path_wall_ahead = ev("WALK_PATH_WALL_AHEAD", 11.0)
  twist.max_lin_vel_x = ev("WALK_MAX_VEL_X", 1.3)
  twist.rel_standing_envs = 0.0
  twist.rel_inplace_throw_envs = 0.0

  # --- 移除与行进相冲突的原地奖励项：消除对站立静止的奖励偏置 ---
  cfg.rewards.pop("dodge_stillness_when_safe", None)
  cfg.rewards.pop("dodge_action_rate_when_safe", None)
  if "mimickit_dodge" in cfg.rewards:
    cfg.rewards["mimickit_dodge"].params["vel_w"] = 0.0

  # --- 行进动力学激励组合（Walk Incentives）：
  # 早期训练曾出现策略退化为原地站立规避（进度仅 0.07 m/s 对比指令 1.3 m/s），因为原地站立可白拿 ~2.0/步
  # 而行进仅得 ~1.15 且增加撞墙风险。以下四项奖励彻底改变收益结构，使行进路线的收益远超原地驻留： ---
  # 1) 全权重速度锚定项：提供线速度跟踪基准轮廓
  cfg.rewards["track_anchor_linear_velocity"].weight = ev("WALK_TRACK_LIN_WEIGHT", 1.0)
  # 2) 路径推进奖励（替换 goal_distance，后者在纯追踪中前瞻距离恒定导致无梯度）：
  # 沿路径的弧长速率（m/s，具有符号对称性，倒退将被扣分）——驱动机器人向路径终点推进。
  # 权重 2.0：在 1.3 m/s 巡航速度下提供 ~2.6 分/步，是最大的行进奖励项。
  cfg.rewards.pop("goal_distance", None)
  cfg.rewards["walk_path_progress"] = RewardTermCfg(
    func=mdp.walk_path_progress_reward,
    weight=ev("WALK_PATH_PROGRESS_WEIGHT", 2.0),
    params={"command_name": "twist"},
  )
  # 2b) 威胁门控的原地驻留惩罚项（行进的惩罚型约束）：
  # 两次来球之间的安全期间，机器人若原地停滞将主动扣除奖励；而在威胁期间惩罚归零，允许机器人急停闪避。
  cfg.rewards["walk_path_stillness_when_safe"] = RewardTermCfg(
    func=mdp.walk_path_stillness_when_safe,
    weight=ev("WALK_PATH_STILLNESS_WEIGHT", -1.0),
    params={"command_name": "twist"},
  )
  # 3) 威胁门控的额外推进奖励项：
  # 在无球来袭的安全期间给予额外的前向推进加分；一旦飞球升空逼近，该项奖励立即归零，绝不拖累躲避动作。
  cfg.rewards["walk_path_progress_when_safe"] = RewardTermCfg(
    func=mdp.walk_path_progress_when_safe,
    weight=ev("WALK_PATH_PROGRESS_SAFE_WEIGHT", 1.0),
    params={"command_name": "twist"},
  )
  # 4) 路径贴合度奖励项：exp(-d^2 / 0.5^2)，对偏离路径中线的横向距离施加指数衰减奖励。
  # 引导机器人在闪避之后主动回归路线中心，避免侧移过大擦碰走廊外侧墙体（墙体距离 0.6-1.0m）。
  cfg.rewards["walk_path_adherence"] = RewardTermCfg(
    func=mdp.walk_path_adherence_reward,
    weight=ev("WALK_PATH_ADHERENCE_WEIGHT", 0.5),
    params={"command_name": "twist", "std": ev("WALK_PATH_ADHERENCE_STD", 0.5)},
  )

  # --- 动作库初始化与重置（RSI）：切换至包含行进步态的动作集合 ---
  cfg.events["init_motion_loader"].params["motion_dir"] = _WALLWALK_RESET_DIR
  cfg.events["reset_from_motion"].params["motion_dir"] = _WALLWALK_RESET_DIR

  # --- 沿路径走廊迎面投掷飞球（Along-Path Corridor Throws）：
  # 从机器人前方沿路径弧长 dist_range 处发射，使球体顺着走廊迎面冲来，而非简单的朝向正前方发射。
  cfg.events["throw_ball_on_dwell"].params["along_path"] = True
  # 走廊横向束缚瞄准：将投掷横向偏移截断在 +-0.4m 范围内，确保飞球绝不会被两侧墙体挡住
  cfg.events["throw_ball_on_dwell"].params["corridor_max_lateral"] = ev(
    "WALK_THROW_CORRIDOR_LAT", 0.4
  )
  cfg.events["throw_ball_on_dwell"].params["aim_noise_lateral_scale"] = ev(
    "WALK_THROW_LAT_NOISE", 0.05
  )
  # --- 投掷时间间隔拉长（4-9s，均值 6.5s）：
  # 降低威胁期占空比至 ~17%，留下约 5.4s 的安全窗口，充分保证机器人能够平稳完成起步加速、稳定巡航与闪避应对。
  walk_interval = (
    ev("WALK_THROW_INTERVAL_MIN", 4.0),
    ev("WALK_THROW_INTERVAL_MAX", 9.0),
  )
  cfg.events["throw_ball_on_dwell"].params["throw_interval_range"] = walk_interval
  cfg.events["reset_dodge_state"].params["throw_interval_range"] = walk_interval

  num_walls = int(ev("WALLWALK_NUM_WALLS", 3))
  cfg = _apply_wall_overrides(
    cfg, num_walls=num_walls, placement="walk_path", recycle_ahead=True, obs_k=num_walls
  )
  return cfg


def g1_amp_dodge_mimickit_classroom_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """22.45m x 8.59m 完整教室室内避障漫游闪避环境配置。

  【场景架构与布局】
  在真实室内教室环境中部署机器人，环境包含 15 套固定桌椅组合与 4 堵外围静态墙体：
  1. 结构化课桌椅布局（Structured Classroom Layout）：
     - 由 `CLASSROOM_ROWS`（行数）与 `CLASSROOM_COLUMN_Y`（各列 Y 坐标）排列 15 套课桌与座椅组合；
     - 每套桌椅沿 X 轴延伸（从椅背 x-1.2 到桌面前沿 x+0.4），行距约为 3.40m，留出约 1.80m 的通行净跨过道；
     - 4 面边界静态墙体封顶四周，界定出 22.45m × 8.59m 的室内受限活动空间。
  2. 障碍物高维几何观测（Obstacle Geometry Observation）：
     - 障碍物观测从简单的相对坐标升级为 `dodge_obstacle_geometry_b`，提取距离机器人最近的 k=8 个
       障碍物（课桌、椅子或边界墙）的完整 3D 边界框与相对姿态信息。
  3. 碰撞掩码过滤机制（Collision Bitmask Filtering）：
     - 解决复杂场景中抛球被家具遮挡或卡住的问题：飞球碰撞几何体设定 `contype = 2, conaffinity = 0`，
       机器人各几何体亲和掩码中注入 `ball_bit = 2`；
     - 场景中的课桌、椅子和四周外墙未分配 bit 2 掩码，因而飞球可物理穿透桌椅家具无阻碍飞向机器人，
       专门检验机器人在受限走道内的敏捷躲避反应，而机器人与家具、墙体之间则维持正常的实体碰撞。
  4. 漫游导航与运动控制适配：
     - 适应狭窄过道环境：缩小纯追踪前瞻距离（`path_lookahead = 0.8`）、允许倒退（`path_allow_backtracking = True`）、
       启用行前原地转弯（`turn_before_walk = True`），并将最大巡航速度降至 0.6 m/s。
     - 引入过道航向对齐奖励（`walk_goal_heading_when_safe`）与精细化的安全期过道推进奖励（`walk_path_progress_when_safe`），
       在安全期严厉惩罚停滞驻留，在来球威胁期间惩罚归零以优先保证规避动作。
  5. 隐藏球与高机位训练视口：
     - 未发射的飞球在重置时被“停泊”在教室外超远距离（y=1000m），仅在发射事件触发时瞬移至发射点；
     - 训练录制相机采用 -80° 高仰角俯视角（距离 4m），避免水平视线被教室外墙或密集课桌遮挡。
  """
  from src.assets.objects.classroom import (
    get_furniture_cfg, get_static_wall_cfg, CLASSROOM_ROWS, CLASSROOM_COLUMN_Y,
    CLASSROOM_ROW_START, CLASSROOM_ROW_PITCH,
    CLASSROOM_X_BOUNDS, CLASSROOM_Y_BOUNDS,
  )
  from src.tasks.amp_loco.mdp.classroom import initialize_static_classroom, reset_classroom_robot

  cfg = g1_amp_dodge_mimickit_wallwalk_flat_env_cfg(play=play)
  # 奖励缩放匹配：在 0.02s 控制步长下，奖励 = 0.01 * sum(weight_i * term_i) + 0.25 * style_score
  cfg.scale_rewards_by_dt = True
  # 剔除原本 WallWalk 的动态走廊墙体实体与传感器，替换为教室静态家具与外墙
  old_names = cfg.events["pin_wall"].params["wall_names"]
  cfg.scene.entities = {k: v for k, v in cfg.scene.entities.items() if k not in old_names}
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in
                            {f"{n}_robot_contact" for n in old_names})
  cfg.events.pop("recycle_walls_ahead", None)
  layout, assets = [], []
  # 每套桌椅从 x-1.2（椅背）延伸到 x+0.4（桌前沿），~3.40m 行间距提供 ~1.80m 净跨过道
  for row in range(CLASSROOM_ROWS):
    for y in CLASSROOM_COLUMN_Y:
      x = CLASSROOM_ROW_START + row * CLASSROOM_ROW_PITCH
      layout.extend(((x, y, 0.38, 0.0), (x - 0.95, y, 0.45, 0.0)))
      assets.extend((get_furniture_cfg("desk"), get_furniture_cfg("chair")))
  x_min, x_max = CLASSROOM_X_BOUNDS
  y_min, y_max = CLASSROOM_Y_BOUNDS
  cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
  hx, hy = (x_max - x_min) / 2, (y_max - y_min) / 2
  # 教室四周边墙定义
  for half, pose in (
    ((hx, 0.08, 1.5), (cx, y_min, 1.5, 0.0)),
    ((hx, 0.08, 1.5), (cx, y_max, 1.5, 0.0)),
    ((0.08, hy, 1.5), (x_min, cy, 1.5, 0.0)),
    ((0.08, hy, 1.5), (x_max, cy, 1.5, 0.0)),
  ):
    layout.append(pose)
    assets.append(get_static_wall_cfg(half_extents=half, rgba=(0.82, 0.85, 0.88, 1.0)))
  # 应用墙体与家具覆盖（接触传感器与 CBF 奖励）
  cfg = _apply_wall_overrides(cfg, num_walls=len(assets), obs_k=8)
  names = cfg.events["pin_wall"].params["wall_names"]
  # 将最近障碍物观测替换为高维障碍几何体观测（k=8）
  cfg.observations["ball_state"].terms["wall_state"] = ObservationTermCfg(
    func=mdp.dodge_obstacle_geometry_b,
    params={"robot_name": "robot", "wall_names": names, "k": 8},
  )
  for name, asset, pose in zip(names, assets, layout, strict=True):
    # 在模型编译期预分离家具实体位置，避免在 reset 事件执行前重叠
    asset.init_state.pos = pose[:3]
    def placed_spec(spec_fn=asset.spec_fn, pos=pose[:3]):
      spec = spec_fn()
      spec.body("wall").pos = pos
      return spec
    asset.spec_fn = placed_spec
    cfg.scene.entities[name] = asset
  # 移除动态墙体锁定与重置事件，替换为静态教室启动与重置事件
  cfg.events.pop("pin_wall")
  cfg.events.pop("reset_wall_pose")
  cfg.events["initialize_static_classroom"] = EventTermCfg(
    func=initialize_static_classroom, mode="startup",
    params={"wall_names": names, "layout": tuple(layout)},
  )
  # 机器人重置槽位紧随 RSI，保留行进步态状态
  cfg.events["reset_walk_path"] = EventTermCfg(
    func=reset_classroom_robot, mode="reset", params={},
  )
  # 狭窄教室过道运动控制参数微调：减小纯追踪前瞻距离以限制路口切弯幅度
  cfg.commands["twist"].path_lookahead = 0.8
  cfg.commands["twist"].path_switch_radius = 0.8
  cfg.commands["twist"].path_allow_backtracking = True
  cfg.commands["twist"].turn_before_walk = True
  cfg.commands["twist"].simple_heading = True
  cfg.commands["twist"].max_lin_vel_x = 0.6
  # 奖励项调整：移除原有的 WallWalk 独立静止项
  cfg.rewards.pop("walk_path_stillness_when_safe", None)
  # 过道航向对齐奖励：在 30deg 以内奖励朝向对准，超出时对偏航角速率跟踪进行奖惩
  cfg.rewards["walk_goal_heading_when_safe"] = RewardTermCfg(
    func=mdp.walk_goal_heading_when_safe,
    weight=float(os.environ.get("WALK_GOAL_HEADING_WEIGHT", 2.0)),
    params={
      "command_name": "twist",
      "robot_name": "robot",
      "sharpness": float(os.environ.get("WALK_GOAL_HEADING_SHARP", 4.0)),
      "turn_angle_threshold": cfg.commands["twist"].walk_full_speed_angle,
      "turn_velocity_std": float(os.environ.get("WALK_GOAL_TURN_STD", 0.25)),
    },
  )
  # 安全期过道前向推进奖励：无球威胁时奖励顺畅推进（上限 0.75m/s），严厉惩罚停滞，来球威胁时自动归零
  cfg.rewards.pop("walk_path_progress", None)
  cfg.rewards["walk_path_progress_when_safe"].weight = float(
    os.environ.get("WALK_PATH_PROGRESS_SAFE_WEIGHT", 2.0)
  )
  cfg.rewards["walk_path_progress_when_safe"].params["v_max"] = float(
    os.environ.get("WALK_PROGRESS_SAFE_VMAX", 0.75)
  )
  cfg.rewards["walk_path_progress_when_safe"].params["stillness_penalty"] = float(
    os.environ.get("WALK_PROGRESS_STILLNESS_PENALTY", 0.6)
  )
  cfg.rewards["walk_path_progress_when_safe"].params["stillness_threshold"] = float(
    os.environ.get("WALK_PROGRESS_STILLNESS_THRESHOLD", 0.02)
  )
  cfg.rewards["track_anchor_linear_velocity"].params["std"] = float(
    os.environ.get("WALK_TRACK_LIN_STD", 0.3)
  )
  cfg.episode_length_s = 25.0
  classroom_interval = (
    float(os.environ.get("WALK_THROW_INTERVAL_MIN", 3.0)),
    float(os.environ.get("WALK_THROW_INTERVAL_MAX", 7.0)),
  )
  # 首球与后续投掷使用统一倒计时区间
  cfg.events["reset_dodge_state"].params["throw_interval_range"] = classroom_interval
  cfg.events["throw_ball_on_dwell"].params.update(
    omnidirectional=True, along_path=False, launch_speed_range=None,
    dist_range=(3.0, 5.0), throw_interval_range=classroom_interval)
  # 碰撞掩码过滤：Bit 1 维持常规机器人与静态场景碰撞；仅机器人接受飞球 Bit 2 掩码，
  # 家具与外墙不检测 Bit 2，确保飞球可穿透家具平滑袭向机器人
  ball_bit = 2
  cfg.scene.entities["ball"].collisions = (
    CollisionCfg(geom_names_expr=("ball_collision",), contype=ball_bit, conaffinity=0),
  )
  cfg.scene.entities["robot"].collisions = tuple(
    replace(
      collision,
      conaffinity=(
        {pattern: value | ball_bit for pattern, value in collision.conaffinity.items()}
        if isinstance(collision.conaffinity, dict)
        else collision.conaffinity | ball_bit
      ),
    )
    for collision in cfg.scene.entities["robot"].collisions
  )
  # 隐藏球机制：在发射前将飞球停泊在教室外远端 (0, 1000, 0.15)，发射时瞬移至预定航向点
  hidden_ball_pos = (0.0, 1000.0, 0.15)
  cfg.scene.entities["ball"].init_state.pos = hidden_ball_pos
  cfg.events["reset_dodge_state"].params["park_offset"] = hidden_ball_pos
  # 训练录制机位：采用 -80° 高仰角俯视跟踪（距离 4m），避免教室密集外墙与桌椅遮挡录制画面
  if not play:
    cfg.viewer.elevation = -80.0
    cfg.viewer.distance = 4.0
  # 扩增最大约束与接触求解容量，防止密集接触点溢出
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 2048)
  # 环境间距设为 0，共享局部原点以保证编译期静态几何 BVH 有效
  cfg.scene.env_spacing = 0.0
  return cfg


@dataclass(kw_only=True)
class G1DodgeDepthEnvCfg(ManagerBasedRlEnvCfg):
  """基于深度图观测的 G1 AMP 闪避环境配置数据类。

  扩展提供支持 CLI 命令行覆盖的深度图时序采样参数（Temporal-Sampling Knob）：
  - ``depth_frame_offsets``: 仿真步长偏移元组（0 表示当前仿真步帧，k 表示过去第 k 步的历史帧）。
    控制深度图观测在时间轴上的堆叠方式；``DepthImageObs`` 在环境实例化时读取该字段，
    因此在任务启动时可直接通过 ``--env.depth_frame_offsets 0 5 10 20`` 进行灵活配置。
    * 默认值 ``(0, 1)``：密集 2 帧堆叠（当前帧 + 上一步历史帧，作为基础对照）；
    * 稀疏采样集如 ``(0, 5, 10, 20, 40, 80)``：为原始视觉深度 MLP 提供超长时序感受野，
      使策略在飞球暂时离开相机视场角（FOV Exit）时仍能维持对弹道的时序感知；
    * 模型网络会自动根据最终计算出的深度观测维度自适应调整输入层结构；
    * 对于 RNN/LSTM 信念任务，因循环网络本身维护隐藏状态记忆，帧堆叠属于冗余项，建议保持默认 (0, 1)。
  """

  depth_frame_offsets: tuple[int, ...] = (0, 1)


def _promote_to_depth_cfg(
  cfg: ManagerBasedRlEnvCfg, depth_frame_offsets: tuple[int, ...]
) -> G1DodgeDepthEnvCfg:
  """将已构建的基础环境配置对象直接重构/升格为 ``G1DodgeDepthEnvCfg`` 实例。

  通过复制所有内部属性字典（``__dict__``），使环境配置对象具备扩展的 ``depth_frame_offsets`` 字段，
  从而支持 tyro 命令行解析器将其暴露于 ``--env`` 下属参数中。绕过常规带命名关键字检查的 ``__init__`` 调用。

  Args:
    cfg: 已完成各模块装配的基础 RL 环境配置实例。
    depth_frame_offsets: 深度图时序帧步长偏移量元组。

  Returns:
    拥有 depth_frame_offsets 属性的 G1DodgeDepthEnvCfg 实例。
  """
  sub = G1DodgeDepthEnvCfg.__new__(G1DodgeDepthEnvCfg)
  sub.__dict__.update(cfg.__dict__)
  sub.depth_frame_offsets = depth_frame_offsets
  return sub



def g1_amp_dodge_depth_single_flat_env_cfg(
  play: bool = False, depth_frame_offsets: tuple[int, ...] = (0, 1)
) -> ManagerBasedRlEnvCfg:
  """G1 AMP 闪避任务：搭载单头部深度相机（+20° 上仰角）+ MimicKit 任务奖励配置。

  【视觉感知与网络架构】
  - 硬件机位：搭载单个头部深度相机（``head_camera_single``），固定向上仰角 +20°，垂直视野覆盖 -7° 至 +47°。
  - 深度图观测空间：16×9 像素低分辨率深度图，单帧 144 维特征。
  - 任务奖励与 Critic 设计：结合 MimicKit 风格奖励以及精简版 CBF-free Critic；
    Critic 的信念门控（Belief Gate）直接绑定至单相机观测组。
  - 策略兼容性：深度图 MimicKit MLP Runner 配置无需改动即可无缝对接（Actor MLP 会自动根据深度维度重塑输入层）。
  """
  cfg = g1_amp_dodge_flat_env_cfg(play=play)
  cfg = _apply_mimickit_reward(cfg, play)
  _add_depth_obs_single(cfg, play)
  _set_slim_critic_dodge(cfg, gate_up="head_depth_single", gate_down="head_depth_single")
  return _promote_to_depth_cfg(cfg, depth_frame_offsets)


def _apply_cbf_joint_filter(cfg: ManagerBasedRlEnvCfg) -> None:
  """将关节位置动作项替换为 CBF 安全控制滤波动作项，并追加两项 CBF-RL 强化学习惩罚项。

  【CBF-RL 阶段一：关节级飞球安全避障动作滤波（docs/cbf_joint_filter_spec.md）】
  - 滤波机理（Projection Filter）：
    安全滤波器将策略网络输出的隐式关节速度指令投影到飞球碰撞圆锥安全边界之外（计算得到最优安全动作 $v^*$）。
  - 软约束驱动自学习（Reward as Training Lever）：
    虽然滤波器在训练时介入修正，但其核心价值在于“通过奖励惩罚促使策略自主进化”：
    对滤波修正量范数 $\Vert v^* - v_{des} \Vert^2$ 进行惩罚，驱使策略网络本身学会主动规避，
    从而在真机部署时可以安全卸载该滤波拐杖，消除训练-实机之间的动作偏差。
  - 环境变量门控：受 `CBF_JOINT=1` 环境变量控制。

  所有控制阈值与惩罚系数均在函数调用时动态从环境变量读取，支持基准评测针对不同构建单独微调。
  """
  ev = lambda k, d: float(os.environ.get(k, d))
  old = cfg.actions["joint_pos"]
  base = {f.name: getattr(old, f.name) for f in fields(old)}
  cfg.actions["joint_pos"] = mdp.CbfJointPositionActionCfg(
    **base,
    cbf_enabled=True,
    # CBF_JOINT_APPLY_FILTER=0 -> 仅计算 v*/h_min 用于奖励惩罚计算，但不实际覆写动作（纯奖励塑形，无动作偏差）
    # 默认 1（动作滤波生效）
    filter_action=os.environ.get("CBF_JOINT_APPLY_FILTER", "1") == "1",
    ball_name="ball",
    body_buffer=ev("CBF_JOINT_BODY_BUFFER", CBF_JOINT_BODY_BUFFER),
    alpha=ev("CBF_JOINT_ALPHA", CBF_JOINT_ALPHA),
    delta_max=ev("CBF_JOINT_DELTA_MAX", CBF_JOINT_DELTA_MAX),
    t_alert=ev("CBF_JOINT_T_ALERT", CBF_JOINT_T_ALERT),
    # 威胁感知半径（距离门控）：
    # 全向高速投掷（OMNI_THROW，速度 20-25 m/s，距离 8-10 m）在 ~0.18s 内即可穿过默认 4m 距离，
    # 若采用 4m 感知半径将导致 CBF 介入过晚。在 OMNI_THROW 模式下感知距离扩大至 12m，
    # 确保飞球发射瞬间即进入 CBF 监视范围；t_alert 仍按碰撞剩余时间动态缩放紧迫度。
    sense_radius=ev("CBF_SENSE_RADIUS", 12.0 if os.environ.get("OMNI_THROW", "0") == "1" else 4.0),
  )
  # 动作修正量范数惩罚：-lambda_corr * ||v* - v_des||^2（计算项本身 <= 0，因此权重为正向 lambda）
  cfg.rewards["dodge_cbf_joint_correction"] = RewardTermCfg(
    func=mdp.dodge_cbf_joint_correction,
    weight=ev("CBF_JOINT_LAMBDA_CORR", CBF_JOINT_LAMBDA_CORR),
    params={},
  )
  # 屏障函数安全裕度违例惩罚：-lambda_h * [h_buf - h_min]_+^2（计算项 <= 0，正权重）
  cfg.rewards["dodge_cbf_joint_buffer"] = RewardTermCfg(
    func=mdp.dodge_cbf_joint_buffer,
    weight=ev("CBF_JOINT_LAMBDA_BUF", CBF_JOINT_LAMBDA_BUF),
    params={"h_buf": ev("CBF_JOINT_BUFFER", CBF_JOINT_BUFFER)},
  )


def g1_amp_dodge_depth_single_ballonly_flat_env_cfg(
  play: bool = False, depth_frame_offsets: tuple[int, ...] = (0, 1)
) -> ManagerBasedRlEnvCfg:
  """G1 AMP 闪避任务：搭载单头部深度相机 + 仅保留飞球的掩码深度观测（Ball-Only Depth）环境配置。

  【视觉架构与实机对齐】
  本环境配置与 ``g1_amp_dodge_depth_single_flat_env_cfg`` 结构一致，但深度图观测采用 ``BallOnlyDepthObs``：
  - 硬件实机镜像（Hardware Mirroring）：将除飞球外的所有背景像素全部截断为远平面（far），
    完美对齐实机部署时利用轻量 EfficientTAM 视频分割网络实时提取的目标球体深度流。
  - 多模态渲染：仿真传感器同时渲染深度图与语义分割标签，以便逐像素准确识别并掩码飞球几何体。
  - 针对性域随机化（Domain Randomization）：
    针对真实分割深度流模拟定制化噪声——整球突发丢帧、边缘毛刺像素丢弃、球体深度整体抖动、网格切片边缘闪烁；
    不添加全屏雾化或背景杂乱填充（背景始终保持纯净 far 远平面）。
  - 继承相同的 MimicKit 风格任务奖励、精简版 CBF-free Critic 以及固定 +20° 头部相机仰角。

  【一阶段主动追球云台（Stage-1 Camera Gimbal，当 CAMERA_GIMBAL=1）】
  - 机器人模型增强：在深度传感器与场景初始化前，通过 `add_camera_gimbal` 向 G1 头部插入 1-DoF 俯仰云台关节。
  - 动作空间隔离：通过正则表达式 ``^(?!camera_pitch_joint$).*$`` 将云台关节从策略动作项中排除，
    保证策略动作维度保持标准的 29 自由度（与基线检查点保持完全兼容）。
  - RSI 重置隔离：在参考动作状态重置中排除云台关节，避免动作库维度不匹配，使云台在重置时保持默认的 0 rad 水平姿态。
  - 本体感知隔离：默认（`CAMERA_PROPRIO=0`）从本体感知观察组中过滤掉云台关节的 pos 与 vel，
    防止输入维度扩张破坏现有 29 关节模型检查点；`CAMERA_PROPRIO=1` 时开放给策略用于全自主感知训练。
  - 真值预言机瞄准事件（Oracle Gimbal Tracking）：
    单步注册 `aim_camera` 事件（`aim_camera_at_ball`），在每个物理步根据特权球体状态计算目标俯仰角，
    驱动云台在飞球飞行全程持续稳定锁定并直视球体。

  【CBF 动作滤波集成】
  若设置环境变量 `CBF_JOINT=1`，则调用 `_apply_cbf_joint_filter` 注入关节级安全滤波与惩罚项。
  """
  cfg = g1_amp_dodge_flat_env_cfg(play=play)
  # 一阶段追球云台集成：在深度/场景配置前修改机器人配置（spec_fn + 执行器），确保重新绑定的相机与新关节到位
  if os.environ.get("CAMERA_GIMBAL", "0") == "1":
    # 云台相机的机械零位为水平方向（0°）。_set_single_camera_tilt 仅在默认 20° 仰角下为空操作，
    # 任何非 20° 的 CAMERA_TILT_DEG 会导致云台相机偏离零位，因此强制断言校验
    assert abs(CAMERA_TILT_DEG - 20.0) < 1e-9, (
      "CAMERA_GIMBAL=1 is only supported at the default CAMERA_TILT_DEG=20 "
      f"(got {CAMERA_TILT_DEG}); a non-default tilt would re-tilt the gimbal camera off its level neutral."
    )
    from src.tasks.amp_loco.config.g1.camera_gimbal import add_camera_gimbal

    add_camera_gimbal(cfg.scene.entities["robot"])
    # 预言机驱动的云台：从策略动作项中排除 camera_pitch_joint，保持策略动作维度为 29（与基线一致）
    cfg.actions["joint_pos"].actuator_names = ("^(?!camera_pitch_joint$).*$",)
    # 从 RSI 动作库重置中排除 camera_pitch_joint（动作剪辑仅包含 29 个身体关节），重置时维持 0 rad 水平朝向
    cfg.events["reset_from_motion"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=("^(?!camera_pitch_joint$).*$",)
    )
    # 从本体感知项（joint_pos 与 joint_vel）中过滤掉 camera_pitch_joint：
    # 默认包含所有关节，若不加以过滤，云台关节将导致历史堆叠后多出 8 维输入，破坏基线检查点。
    # CAMERA_PROPRIO=1 时跳过此过滤以将云台关节纳入本体感知（用于感知训练）；默认 0 时过滤以维持 29 关节维度。
    camera_proprio = os.environ.get("CAMERA_PROPRIO", "0")
    if camera_proprio != "1":
      _body_joint_cfg = SceneEntityCfg("robot", joint_names=("^(?!camera_pitch_joint$).*$",))
      for _group_name in ("actor", "critic"):
        _group = cfg.observations[_group_name]
        for _term_name in ("joint_pos", "joint_vel"):
          if _term_name in _group.terms:
            _group.terms[_term_name].params = {
              **_group.terms[_term_name].params,
              "asset_cfg": _body_joint_cfg,
            }
    # 单步真值预言机驱动：利用特权球体状态计算俯仰角，驱动 camera_pitch_joint 直视飞球。
    # 该关节不由策略控制，因此本事件为其唯一驱动源，无控制权冲突。
    cfg.events["aim_camera"] = EventTermCfg(
      func=mdp.aim_camera_at_ball,
      mode="step",
      params={"robot_name": "robot", "ball_name": "ball"},
    )
  cfg = _apply_mimickit_reward(cfg, play)
  _add_depth_obs_single_ballonly(cfg, play)
  _set_slim_critic_dodge(cfg, gate_up="head_depth_single", gate_down="head_depth_single")
  # 在函数调用时动态读取 CBF_JOINT 环境变量，便于 Benchmark 基准评测动态切换动作滤波器
  if os.environ.get("CBF_JOINT", "0") == "1":
    _apply_cbf_joint_filter(cfg)
  return _promote_to_depth_cfg(cfg, depth_frame_offsets)



