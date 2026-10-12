"""强化学习环境事件 (Events) 与重置调度器模块.

【模块定位与物理机制】
本模块负责仿真环境中智能体状态重置、动态投射物弹道发射、障碍墙空间布局与相机云台主动跟踪等关键物理事件：

1. 动作捕捉参考状态重置 (AMP Reference State Initialization, RSI):
   - `MotionResetManager`: 单例管理器，负责从真实人体动作捕捉数据库 (`ampmotion_loader`) 中加载奔跑、行走、
     跌倒后自恢复等动作切片。在回合开始时随机采样某一帧姿态作为初始状态 (RSI)，使机器人直接在富有动态动量的相空间中开始探索，
     极大加快敏捷奔跑与跳跃步态的学习效率；
   - 支持延迟重置环境过滤，为跌倒恢复训练提供专用数据重置分支。

2. 弹道抛物线投球事件 (Ballistic Ball Throwing):
   - `solve_ballistic_velocity`: 闭式求解固定发射初速度下的平射直瞄弹道 (Direct/Flat Arc)。
     解二次抛体方程 $k (1+u^2) - d \\cdot u + \\Delta z = 0$（其中 $u = \\tan\\theta, k = \\frac{gd^2}{2v^2}$），
     选取低仰角实根，使高速来球以极快速度直奔机器人；
   - `throw_ball_on_dwell`: 反应式多模式抛球触发器。
     * 触发模式：支持站立稳定后抛球 (Grounded Dwell)、基于 MimicKit 的固定时间窗随机抛球 (Timed Throw)、
       球体落地即抛 (Rethrow on Ground) 以及针对演示的边跑边扔 (Throw while Moving)；
     * 弹道生成：
       - 水平下坠球 (Descending Throw): $v_{z0} = 0$，自 ~2m 高处平抛，下落至腿部/躯干，迫使机器人进行 2D 侧跃避碰；
       - 低仰角上升球 (Low-Arc High Throw): 计算纵向上挑初速度 $v_{z0} = \\frac{z_{\\text{tgt}} - z_0}{t_{\\text{flight}}} + \\frac{1}{2} g t_{\\text{flight}}$，
         抛向胸口或头部，促使策略学习深蹲下潜 (Duck) 规避；
     * 速度前瞻导引 (Velocity Lead) 与走廊系泊 (Corridor-Tethered Aim):
       根据机器人当前运动速度进行超前拦截瞄准，并在走廊避障任务中将瞄准点投影在路径切线走廊内部，防止来球被侧面墙体遮挡。

3. 障碍墙定位、固定与动态循环 (Obstacle Wall Management):
   - 相对航向定位 (`reset_wall_pose`): 沿机器人偏航角坐标系或折线巡航路径两侧放置障碍墙，强制双侧随机分布以训练感知能力；
   - 运动学刚性钉死 (`pin_wall`): 在 MuJoCo 中通过每步重写根节点位姿并清零速度，以自由浮动刚体实现无限大质量的虚拟刚性障碍墙；
   - 动态前置循环 (`recycle_walls_ahead`): 将机器人已经走过的后方障碍墙，实时瞬移回收并重新布置在前方视野内，
     以有限数量的刚体对象构建出无限延伸的障碍物走廊。

4. 头部云台主动注视预警 (Gimbal Camera Aiming):
   - `optimal_camera_pitch` 与 `aim_camera_at_ball`: 根据球体相对相机的空间视线夹角，计算躯干坐标系下的俯仰偏角，
     自动控制头部俯仰舵机对准来球，使球体始终处于 16x9 深度相机的最佳视野中心。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from src.tasks.amp_loco.ampmotion_loader import MotionLoader
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager
from src.tasks.amp_loco.mdp.walk_path import generate_walk_path, walk_path_query

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MotionResetManager:
    """管理 AMP 动作捕捉数据库加载与参考状态重置 (RSI) 的单例管理器."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self.walk_run_frames: dict[str, dict[str, torch.Tensor]] = {}
        self.recovery_frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        """获取全局唯一单例实例."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def init(
        self,
        env: ManagerBasedRlEnv,
        motion_dir: str,
        recovery_dir: str | None = None,
    ) -> None:
        """从指定目录解析加载动捕动作片段 (Walk/Run 与 Recovery)."""
        if motion_dir in self.walk_run_frames:
            return

        loader = MotionLoader(
            motion_dir=motion_dir,
            tgt_body_indexes=[],
            tgt_anchor_indexes=0,
            feet_indexes=0,
            device=str(env.device),
            recovery_dir=recovery_dir,
        )

        self.walk_run_frames[motion_dir] = self._concat_frames(loader.motion_data)
        motion_count = self.walk_run_frames[motion_dir]["root_pos"].shape[0]
        print(f"[MotionResetManager] 已从 {motion_dir} 加载 {len(loader.motion_data)} 个片段，共 {motion_count} 帧动捕姿态")

        if loader.motion_data_recovery:
            self.recovery_frames[motion_dir] = self._concat_frames(loader.motion_data_recovery)
            recovery_count = self.recovery_frames[motion_dir]["root_pos"].shape[0]
            print(f"[MotionResetManager] 已从 {recovery_dir} 加载 {len(loader.motion_data_recovery)} 个自恢复片段，共 {recovery_count} 帧姿态")

    # ------------------------------------------------------------------
    # 重置执行
    # ------------------------------------------------------------------

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        """对指定环境执行动捕参考状态初始化 (RSI)."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        if len(env_ids) == 0:
            return

        # 区分正常重置环境与延迟重置环境
        delay_mask = self._get_delay_env_mask(env)
        if delay_mask is not None:
            is_delay = delay_mask[env_ids]
            delay_ids = env_ids[is_delay]
            normal_ids = env_ids[~is_delay]
        else:
            delay_ids = env_ids[:0]
            normal_ids = env_ids

        # 正常环境从奔跑/行走数据集中均匀抽取重置帧
        if len(normal_ids) > 0:
            self._write_reset_state(env, normal_ids, self.walk_run_frames[motion_dir], asset_cfg)

        # 延迟环境优先从跌倒自恢复数据集中抽取重置帧
        if len(delay_ids) > 0:
            recovery = self.recovery_frames.get(motion_dir)
            frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
            self._write_reset_state(env, delay_ids, frames, asset_cfg)

    def _get_delay_env_mask(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
        """提取延迟终止掩码."""
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            return tm._delay_env_mask
        return None

    def _write_reset_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """将动捕帧的根节点位姿、速度以及各关节状态写入物理模拟器."""
        total_frames = frames["root_pos"].shape[0]
        num_reset = len(env_ids)
        idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset: Entity = env.scene[asset_cfg.name]

        # --- 根节点位姿 (保持动捕高度与四元数，水平坐标贴合环境原点) ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        positions = env.scene.env_origins[env_ids].clone()
        positions[:, 2] = root_pos[:, 2]

        root_pose = torch.cat([positions, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # --- 根节点速度 (线速度与角速度) ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # --- 关节角度与速度 (根据模型软限位进行安全截断) ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_frames(motions: list[dict]) -> dict[str, torch.Tensor]:
        """合并多个动捕片段为统一的字典张量."""
        root_pos_list = []
        root_quat_list = []
        root_lin_vel_list = []
        root_ang_vel_list = []
        joint_pos_list = []
        joint_vel_list = []
        for motion in motions:
            root_pos_list.append(motion["body_pos_w"][:, 0, :])
            root_quat_list.append(motion["body_quat_w"][:, 0, :])
            root_lin_vel_list.append(motion["body_lin_vel_w"][:, 0, :])
            root_ang_vel_list.append(motion["body_ang_vel_w"][:, 0, :])
            joint_pos_list.append(motion["dof_pos"])
            joint_vel_list.append(motion["dof_vel"])
        return {
            "root_pos": torch.cat(root_pos_list, dim=0),
            "root_quat": torch.cat(root_quat_list, dim=0),
            "root_lin_vel": torch.cat(root_lin_vel_list, dim=0),
            "root_ang_vel": torch.cat(root_ang_vel_list, dim=0),
            "joint_pos": torch.cat(joint_pos_list, dim=0),
            "joint_vel": torch.cat(joint_vel_list, dim=0),
        }


# ------------------------------------------------------------------
# 事件回调函数 (委托给单例)
# ------------------------------------------------------------------

def init_motion_loader(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None = None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
) -> None:
    """启动事件: 初始化加载动捕数据集，并可选安装延迟重置管理器."""
    MotionResetManager.get().init(
        env=env,
        motion_dir=motion_dir,
        recovery_dir=recovery_dir,
    )

    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay > 0 and max_delay_steps > 0:
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
        delay_mask[delay_indices] = True
        env.termination_manager = DelayedTerminationManager(
            base=env.termination_manager,
            delay_env_mask=delay_mask,
            max_delay_steps=max_delay_steps,
        )
        print(
            "[init_motion_loader] 延迟终止管理器安装完毕: "
            f"{num_delay}/{env.num_envs} 个环境, 最大延迟步数={max_delay_steps}"
        )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """环境重置事件: 从动捕数据集中随机抽取状态执行 RSI 初始化."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        asset_cfg=asset_cfg,
    )


def reset_to_default_stand(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """环境重置事件: 将机器人初始化为默认的经典站立平衡位姿 (用于实机演练与展示)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if len(env_ids) == 0:
        return

    asset: Entity = env.scene[asset_cfg.name]

    # 根节点默认高度与朝向，线速度与角速度清零
    default_root = asset.data.default_root_state[env_ids]
    positions = env.scene.env_origins[env_ids].clone()
    positions[:, 2] = default_root[:, 2]
    root_pose = torch.cat([positions, default_root[:, 3:7]], dim=-1)
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(
        torch.zeros_like(default_root[:, 7:13]), env_ids=env_ids
    )

    # 关节角度与速度恢复默认值
    asset.write_joint_state_to_sim(
        asset.data.default_joint_pos[env_ids].clone(),
        torch.zeros_like(asset.data.default_joint_vel[env_ids]),
        env_ids=env_ids,
    )


# ------------------------------------------------------------------
# 避球任务：多模式抛物线动态投球事件
# ------------------------------------------------------------------

_DODGE_DWELL_ATTR = "_dodge_dwell_counter"
_DODGE_THROW_CD_ATTR = "_dodge_throw_countdown"


def _sample_interval_steps(
    env: "ManagerBasedRlEnv", n: int, interval_s: tuple[float, float]
) -> torch.Tensor:
    """在指定秒数区间内均匀随机采样离散抛球倒计时步数."""
    lo = max(1, round(interval_s[0] / env.step_dt))
    hi = max(lo, round(interval_s[1] / env.step_dt))
    return torch.randint(lo, hi + 1, (n,), device=env.device)


def _park_offset_world(env: "ManagerBasedRlEnv", env_ids: torch.Tensor, park_offset):
    """计算球体在停靠等待状态下的世界坐标 (默认放置在侧面地面)."""
    origins = env.scene.env_origins[env_ids]
    off = torch.tensor(park_offset, device=env.device, dtype=origins.dtype)
    return origins + off


def reset_dodge_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ball_name: str = "ball",
    park_offset: tuple[float, float, float] = (0.0, 3.0, 0.1),
    throw_interval_range: tuple[float, float] | None = None,
) -> None:
    """重置事件: 清空停驻计时器，并将球体停靠在远离机器人的侧面地面，防止干扰新回合."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    counter = getattr(env, _DODGE_DWELL_ATTR, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        setattr(env, _DODGE_DWELL_ATTR, counter)
    counter[env_ids] = 0

    if throw_interval_range is not None:
        cd = getattr(env, _DODGE_THROW_CD_ATTR, None)
        if cd is None or cd.shape[0] != env.num_envs:
            cd = _sample_interval_steps(env, env.num_envs, throw_interval_range)
            setattr(env, _DODGE_THROW_CD_ATTR, cd)
        cd[env_ids] = _sample_interval_steps(env, len(env_ids), throw_interval_range)

    ball: Entity = env.scene[ball_name]
    n = len(env_ids)
    park_pos = _park_offset_world(env, env_ids, park_offset)
    quat_identity = torch.zeros(n, 4, device=env.device)
    quat_identity[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([park_pos, quat_identity], dim=-1), env_ids=env_ids
    )
    ball.write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=env_ids
    )


def solve_ballistic_velocity(
    start: torch.Tensor,
    target: torch.Tensor,
    speed: torch.Tensor,
    gravity: float = 9.81,
) -> torch.Tensor:
    """闭式求解定初速重力抛体运动方程，返回直瞄低仰角发射速度矢量 (Direct Flat Arc).

    数学公式推导：
      设水平位移为 d，竖直落差为 dz，发射仰角为 theta，u = tan(theta)。
      由物理抛体公式: dz = d * u - (g * d^2 / (2 * speed^2)) * (1 + u^2)
      记 k = g * d^2 / (2 * speed^2)，化为关于 u 的一元二次方程:
          k * u^2 - d * u + (dz + k) = 0
      判别式: disc = d^2 - 4 * k * (dz + k)
      选取低俯仰平直弹道解: u = (d - sqrt(disc)) / (2 * k)
    """
    d_xy = target[..., :2] - start[..., :2]
    d = torch.norm(d_xy, dim=-1).clamp(min=1e-6)            # 水平距离
    dir_xy = d_xy / d.unsqueeze(-1)                          # 水平飞行单位方向
    dz = target[..., 2] - start[..., 2]                     # 竖直落差
    k = gravity * d * d / (2.0 * speed * speed)
    disc = (d * d - 4.0 * k * (dz + k)).clamp(min=0.0)
    u = (d - torch.sqrt(disc)) / (2.0 * k)                  # 抛射仰角正切值 u = tan(theta)
    theta = torch.atan(u)
    vh = speed * torch.cos(theta)                           # 水平初速度大小
    vz = speed * torch.sin(theta)                           # 竖直初速度大小 (带符号)
    vel = torch.zeros_like(start)
    vel[..., :2] = dir_xy * vh.unsqueeze(-1)
    vel[..., 2] = vz
    return vel


def throw_ball_on_dwell(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ball_name: str = "ball",
    robot_name: str = "robot",
    command_name: str = "twist",
    dwell_time_s: float = 0.5,
    command_threshold: float = 0.1,
    speed_threshold: float = 0.3,
    home_radius: float | None = None,
    min_move_speed: float | None = None,
    cooldown_s: float = 1.5,
    throw_interval_range: tuple[float, float] | None = None,
    rethrow_ground_height: float | None = None,
    require_feet_grounded: bool = False,
    feet_sensor_name: str = "feet_ground_contact",
    dist_range: tuple[float, float] = (2.0, 3.0),
    height_range: tuple[float, float] = (1.8, 2.0),
    angle_deg: float = 25.0,
    flight_time_range: tuple[float, float] = (0.55, 0.62),
    lead_target: bool = True,
    aim_noise_scale: float = 0.1,
    gravity: float = 9.81,
    high_throw_fraction: float = 0.0,
    high_launch_height_range: tuple[float, float] = (0.4, 0.9),
    high_target_z_range: tuple[float, float] = (0.9, 1.3),
    skip_standing_envs: bool = False,
    omnidirectional: bool = False,
    launch_speed_range: tuple[float, float] | None = None,
    target_z_range: tuple[float, float] = (0.3, 1.3),
    along_path: bool = False,
    corridor_max_lateral: float = 0.4,
    aim_noise_lateral_scale: float = 0.05,
) -> None:
    """单步执行事件: 判定机器人准备就绪并执行向机器人精准投球."""
    robot: Entity = env.scene[robot_name]
    ball: Entity = env.scene[ball_name]
    device = env.device

    # --- 站立准备计数器 ---
    counter = getattr(env, _DODGE_DWELL_ATTR, None)
    if counter is None or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        setattr(env, _DODGE_DWELL_ATTR, counter)

    base_speed = torch.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    near_home = torch.ones(env.num_envs, dtype=torch.bool, device=device)
    if home_radius is not None:
        home_xy = env.scene.env_origins[:, :2]
        home_dist = torch.norm(robot.data.root_link_pos_w[:, :2] - home_xy, dim=1)
        near_home = home_dist < home_radius

    # 1. 定时触发模式 (Timed Throw - MimicKit 经典模式)
    if throw_interval_range is not None:
        cd = getattr(env, _DODGE_THROW_CD_ATTR, None)
        if cd is None or cd.shape[0] != env.num_envs:
            cd = _sample_interval_steps(env, env.num_envs, throw_interval_range)
        cd = cd - 1
        fire = cd <= 0
        if fire.any():
            new = _sample_interval_steps(env, env.num_envs, throw_interval_range)
            cd = torch.where(fire, new, cd)
        setattr(env, _DODGE_THROW_CD_ATTR, cd)
    # 2. 落地连续重掷模式 (Rethrow on Ground)
    elif rethrow_ground_height is not None:
        landed = ball.data.root_link_pos_w[:, 2] < rethrow_ground_height
        if require_feet_grounded:
            feet: ContactSensor = env.scene[feet_sensor_name]
            contact_t = feet.data.current_contact_time
            assert contact_t is not None, f"传感器 '{feet_sensor_name}' 必须启用 track_air_time。"
            landed = landed & (contact_t > 0.0).all(dim=1)
        fire = landed & near_home
    # 3. 边跑边扔模式 (Play 演示模式)
    elif min_move_speed is not None:
        counter = counter + 1
        cooldown = max(1, round(cooldown_s / env.step_dt))
        moving = base_speed > min_move_speed
        fire = near_home & (
            ((counter >= cooldown) & moving) | (counter >= 2 * cooldown)
        )
    # 4. 着地稳定停驻模式 (Grounded Dwell 训练默认)
    else:
        feet: ContactSensor = env.scene[feet_sensor_name]
        contact_t = feet.data.current_contact_time
        assert contact_t is not None, f"传感器 '{feet_sensor_name}' 必须启用 track_air_time。"
        both_feet_grounded = (contact_t > 0.0).all(dim=1)
        standing = both_feet_grounded & near_home
        counter = torch.where(standing, counter + 1, torch.zeros_like(counter))
        dwell_steps = max(1, round(dwell_time_s / env.step_dt))
        fire = counter >= dwell_steps

    # 查看器手动控制与暂停
    force_high: bool | None = None
    if getattr(env, "_dodge_throw_paused", False):
        fire = torch.zeros_like(fire)
    if getattr(env, "_dodge_throw_once", False):
        fire = torch.ones_like(fire)
        env._dodge_throw_once = False
        force_high = getattr(env, "_dodge_throw_force_high", None)
        env._dodge_throw_force_high = None

    # 对永久站立锚点环境跳过投球
    if skip_standing_envs:
        cmd_term = env.command_manager.get_term(command_name)
        standing_mask = getattr(cmd_term, "is_standing_env", None)
        if standing_mask is not None:
            fire = fire & ~standing_mask

    counter = torch.where(fire, torch.zeros_like(counter), counter)
    setattr(env, _DODGE_DWELL_ATTR, counter)

    throw_ids = fire.nonzero(as_tuple=False).squeeze(-1)
    if len(throw_ids) == 0:
        return

    n = len(throw_ids)
    root_pos = robot.data.root_link_pos_w[throw_ids]
    root_quat = robot.data.root_link_quat_w[throw_ids]
    yq = yaw_quat(root_quat)

    def _uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=device) * (hi - lo) + lo

    dist = _uniform(*dist_range)
    if along_path:
        cmd_term = env.command_manager.get_term(command_name)
        s_launch = cmd_term._path_s[throw_ids] + dist
        path_start_xy, _path_tan = walk_path_query(env, throw_ids, s_launch)

    # 固定发射速度弹道求解分支
    if launch_speed_range is not None:
        if along_path:
            start_xy = path_start_xy
        elif omnidirectional:
            bearing = _uniform(-math.pi, math.pi)
            off_xy = torch.stack([dist * torch.cos(bearing), dist * torch.sin(bearing)], dim=-1)
            start_xy = root_pos[:, 0:2] + off_xy
        else:
            bearing = _uniform(-angle_deg, angle_deg) * (math.pi / 180.0)
            offset_b = torch.stack([dist, dist * torch.tan(bearing), torch.zeros_like(dist)], dim=-1)
            start_xy = root_pos[:, 0:2] + quat_apply(yq, offset_b)[:, 0:2]
        start = torch.empty(n, 3, device=device)
        start[:, 0:2] = start_xy
        start[:, 2] = _uniform(*height_range)
        speed = _uniform(*launch_speed_range)
        target_xy = root_pos[:, 0:2].clone()
        if lead_target:
            t_lead = (dist / speed).unsqueeze(-1)
            target_xy = target_xy + robot.data.root_link_lin_vel_w[throw_ids, :2] * t_lead
        if aim_noise_scale > 0.0:
            target_xy = target_xy + aim_noise_scale * torch.randn_like(target_xy)
        target = torch.cat([target_xy, _uniform(*target_z_range).unsqueeze(-1)], dim=-1)
        vel = solve_ballistic_velocity(start, target, speed, gravity=gravity)
        quat_identity = torch.zeros(n, 4, device=device)
        quat_identity[:, 0] = 1.0
        ball.write_root_link_pose_to_sim(
            torch.cat([start, quat_identity], dim=-1), env_ids=throw_ids
        )
        root_vel = torch.zeros(n, 6, device=device)
        root_vel[:, 0:3] = vel
        ball.write_root_link_velocity_to_sim(root_vel, env_ids=throw_ids)
        return

    # 基于飞行时间的恒定反应时间弹道生成分支
    if omnidirectional:
        bearing = _uniform(-math.pi, math.pi)
        offset_w = torch.stack(
            [dist * torch.cos(bearing), dist * torch.sin(bearing), torch.zeros_like(dist)],
            dim=-1,
        )
    else:
        angle = _uniform(-angle_deg, angle_deg) * (math.pi / 180.0)
        lateral = dist * torch.tan(angle)
        offset_b = torch.stack([dist, lateral, torch.zeros_like(dist)], dim=-1)
        offset_w = quat_apply(yq, offset_b)

    # 混合两种弹道威胁类型：下蹲球 (High) 与平抛侧跃球 (Descending)
    if force_high is None:
        high = torch.rand(n, device=device) < high_throw_fraction
    else:
        high = torch.full((n,), bool(force_high), dtype=torch.bool, device=device)
    start = torch.empty(n, 3, device=device)
    if along_path:
        start[:, 0:2] = path_start_xy
    else:
        start[:, 0:2] = root_pos[:, 0:2] + offset_w[:, 0:2]
    start[:, 2] = torch.where(high, _uniform(*high_launch_height_range), _uniform(*height_range))

    # 计算预期的飞行时间 (反应窗口)
    t_req = _uniform(*flight_time_range)
    t_max = torch.sqrt(2.0 * (start[:, 2] - 0.05).clamp(min=1e-3) / gravity)
    t_flight = torch.where(high, t_req, torch.minimum(t_req, t_max))

    target_xy = root_pos[:, 0:2].clone()
    if along_path:
        # 沿走廊系泊瞄准：沿切线超前前瞻，横向偏移严格截断在走廊内
        proj_xy, tan_yaw = walk_path_query(env, throw_ids, cmd_term._path_s[throw_ids])
        tan = torch.stack([torch.cos(tan_yaw), torch.sin(tan_yaw)], dim=-1)
        perp = torch.stack([-tan[:, 1], tan[:, 0]], dim=-1)
        off = target_xy - proj_xy
        along = (off * tan).sum(dim=-1)
        lat = (off * perp).sum(dim=-1)
        if lead_target:
            v_xy = robot.data.root_link_lin_vel_w[throw_ids, :2]
            v_along = (v_xy * tan).sum(dim=-1)
            along = along + v_along * t_flight
        lat = lat.clamp(-corridor_max_lateral, corridor_max_lateral)
        target_xy = (
            proj_xy + tan * along.unsqueeze(-1) + perp * lat.unsqueeze(-1)
        )
        if aim_noise_scale > 0.0:
            target_xy = target_xy + tan * (aim_noise_scale * torch.randn(n, device=device)).unsqueeze(-1)
            target_xy = target_xy + perp * (aim_noise_lateral_scale * torch.randn(n, device=device)).unsqueeze(-1)
    else:
        if lead_target:
            root_vel_xy = robot.data.root_link_lin_vel_w[throw_ids, :2]
            target_xy = target_xy + root_vel_xy * t_flight.unsqueeze(-1)
        if aim_noise_scale > 0.0:
            target_xy = target_xy + aim_noise_scale * torch.randn_like(target_xy)

    aim_off_b = getattr(env, "_dodge_throw_aim_offset_b", None)
    if aim_off_b is not None:
        off_b = torch.tensor(
            [float(aim_off_b[0]), float(aim_off_b[1]), 0.0], device=device
        ).expand(n, 3)
        target_xy = target_xy + quat_apply(yq, off_b)[:, 0:2]

    # 计算水平初速度
    disp_xy = target_xy - start[:, 0:2]
    vel = torch.zeros(n, 3, device=device)
    vel[:, 0] = disp_xy[:, 0] / t_flight
    vel[:, 1] = disp_xy[:, 1] / t_flight
    # 针对下潜球计算上抛初速度 vz0
    z_target = _uniform(*high_target_z_range)
    vz_high = (z_target - start[:, 2]) / t_flight + 0.5 * gravity * t_flight
    vel[:, 2] = torch.where(high, vz_high, torch.zeros_like(vz_high))

    quat_identity = torch.zeros(n, 4, device=device)
    quat_identity[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(
        torch.cat([start, quat_identity], dim=-1), env_ids=throw_ids
    )
    root_vel = torch.zeros(n, 6, device=device)
    root_vel[:, 0:3] = vel
    ball.write_root_link_velocity_to_sim(root_vel, env_ids=throw_ids)


# ------------------------------------------------------------------
# 静态障碍墙事件：重置定位、刚性固定与前置循环
# ------------------------------------------------------------------

_WALL_POS_ATTR = "_wall_pos_w"
_WALL_YAW_ATTR = "_wall_yaw_w"


def reset_walk_path(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    robot_name: str = "robot",
    seg_len_range: tuple[float, float] = (2.0, 4.0),
    turn_max: float = 0.45,
    initial_len: float = 80.0,
) -> None:
    """环境重置事件: 为指定环境生成随机折线漫游路径 (参见 walk_path.py)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return
    generate_walk_path(
        env,
        env_ids,
        robot_name=robot_name,
        seg_len_range=seg_len_range,
        turn_max=turn_max,
        initial_len=initial_len,
    )


def _wall_geom_id(env: "ManagerBasedRlEnv", wall_name: str) -> int:
    """解析障碍墙碰撞几何体的全局 ID."""
    return env.sim.mj_model.geom(f"{wall_name}/wall_collision").id


def _heading_frame(robot_quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """提取机器人的航向单位向量 fwd 与垂直法向量 perp (世界 xy 平面)."""
    fwd_b = torch.tensor([1.0, 0.0, 0.0], device=robot_quat_w.device).expand(
        robot_quat_w.shape[0], 3
    )
    yq = yaw_quat(robot_quat_w)
    fwd = quat_apply(yq, fwd_b)[:, :2]
    perp = torch.stack([-fwd[:, 1], fwd[:, 0]], dim=-1)
    return fwd, perp


def reset_wall_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  wall_names: tuple[str, ...] = ("wall",),
  robot_name: str = "robot",
  lateral_dist_range: tuple[float, float] = (0.6, 1.0),
  x_offset_range: tuple[float, float] = (-0.5, 0.5),
  placement: str = "beside",
  path_ahead_range: tuple[float, float] = (2.5, 4.0),
  path_spacing: float = 3.0,
) -> None:
    """环境重置事件: 根据机器人当前重置朝向或折线路径，随机布置障碍墙的位置与朝向."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    robot: Entity = env.scene[robot_name]
    n = len(env_ids)
    W = len(wall_names)
    device = env.device

    def _uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=device) * (hi - lo) + lo

    pos = getattr(env, _WALL_POS_ATTR, None)
    yaw = getattr(env, _WALL_YAW_ATTR, None)
    if pos is None or pos.shape[0] != env.num_envs or pos.shape[1] != W:
        pos = torch.zeros(env.num_envs, W, 3, device=device)
        yaw = torch.zeros(env.num_envs, W, device=device)
        setattr(env, _WALL_POS_ATTR, pos)
        setattr(env, _WALL_YAW_ATTR, yaw)

    adr = robot.data.indexing.free_joint_q_adr
    root_pose = env.sim.data.qpos[env_ids][:, adr]
    root_pos = root_pose[:, :3]
    root_quat = root_pose[:, 3:7]
    robot_yaw = torch.atan2(
        2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
        1.0 - 2.0 * (root_quat[:, 2] ** 2 + root_quat[:, 3] ** 2),
    )
    fwd, perp = _heading_frame(root_quat)

    for i, name in enumerate(wall_names):
        side = torch.where(
            torch.rand(n, device=device) < 0.5,
            -torch.ones(n, device=device),
            torch.ones(n, device=device),
        )
        if placement == "walk_path":
            station = _uniform(*path_ahead_range) + i * path_spacing
            base_xy, wall_yaw = walk_path_query(env, env_ids, station)
            perp_i = torch.stack(
                [-torch.sin(wall_yaw), torch.cos(wall_yaw)], dim=-1
            )
            wall_xy = base_xy + perp_i * (side * _uniform(*lateral_dist_range)).unsqueeze(-1)
        else:
            if placement == "path":
                ahead = _uniform(*path_ahead_range) + i * path_spacing
                lateral = side * _uniform(*lateral_dist_range)
            else:
                ahead = _uniform(*x_offset_range)
                lateral = side * _uniform(*lateral_dist_range)
            wall_xy = root_pos[:, :2] + fwd * ahead.unsqueeze(-1) + perp * lateral.unsqueeze(-1)
            wall_yaw = robot_yaw
        gid = _wall_geom_id(env, name)
        half_h = env.sim.model.geom_size[env_ids, gid, 2]
        wall_pos = torch.cat([wall_xy, half_h.unsqueeze(-1)], dim=-1)
        pos[env_ids, i] = wall_pos
        yaw[env_ids, i] = wall_yaw

        quat = torch.zeros(n, 4, device=device)
        quat[:, 0] = torch.cos(wall_yaw / 2.0)
        quat[:, 3] = torch.sin(wall_yaw / 2.0)
        wall: Entity = env.scene[name]
        wall.write_root_link_pose_to_sim(
            torch.cat([wall_pos, quat], dim=-1), env_ids=env_ids
        )
        wall.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=device), env_ids=env_ids
        )


def pin_wall(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  wall_names: tuple[str, ...] = ("wall",),
) -> None:
    """单步执行事件: 每控制步重新钉死障碍墙至其暂存位姿并置零速度 (实现无限大质量虚拟刚性体)."""
    pos = getattr(env, _WALL_POS_ATTR, None)
    yaw = getattr(env, _WALL_YAW_ATTR, None)
    if pos is None or yaw is None:
        return
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    for i, name in enumerate(wall_names):
        wall: Entity = env.scene[name]
        quat = torch.zeros(env.num_envs, 4, device=env.device)
        quat[:, 0] = torch.cos(yaw[:, i] / 2.0)
        quat[:, 3] = torch.sin(yaw[:, i] / 2.0)
        wall.write_root_link_pose_to_sim(
            torch.cat([pos[:, i], quat], dim=-1), env_ids=env_ids
        )
        wall.write_root_link_velocity_to_sim(
            torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids
        )


def recycle_walls_ahead(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  wall_names: tuple[str, ...] = ("wall",),
  robot_name: str = "robot",
  ahead_range: tuple[float, float] = (7.0, 10.0),
  lateral_range: tuple[float, float] = (0.5, 1.2),
  behind_margin: float = 1.0,
  along_path: bool = False,
) -> None:
    """单步执行事件: 将机器人已经走过的后方障碍墙，实时瞬移到前方走廊形成无尽循环障碍物."""
    robot: Entity = env.scene[robot_name]
    pos = getattr(env, _WALL_POS_ATTR, None)
    yaw = getattr(env, _WALL_YAW_ATTR, None)
    if pos is None or yaw is None:
        return

    n = env.num_envs
    device = env.device
    root_pos = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    fwd, perp = _heading_frame(root_quat)

    def _uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, device=device) * (hi - lo) + lo

    path_s = None
    if along_path:
        path_s = env.command_manager.get_term("twist")._path_s

    for i, name in enumerate(wall_names):
        rel_xy = pos[:, i, :2] - root_pos[:, :2]
        along = (rel_xy * fwd).sum(dim=-1)
        # 当墙体落后于机器人身后超过 behind_margin 时触发回收
        recycle = along < -behind_margin
        if not recycle.any():
            continue
        side = torch.where(
            torch.rand(n, device=device) < 0.5,
            -torch.ones(n, device=device),
            torch.ones(n, device=device),
        )
        if along_path:
            ids = recycle.nonzero(as_tuple=False).squeeze(-1)
            station = path_s[ids] + _uniform(*ahead_range)[ids]
            base_xy, base_yaw = walk_path_query(env, ids, station)
            perp_i = torch.stack(
                [-torch.sin(base_yaw), torch.cos(base_yaw)], dim=-1
            )
            off = (side[ids] * _uniform(*lateral_range)[ids]).unsqueeze(-1)
            pos[ids, i, :2] = base_xy + perp_i * off
            yaw[ids, i] = base_yaw
        else:
            m = recycle.float().unsqueeze(-1)
            new_xy = (
                root_pos[:, :2]
                + fwd * _uniform(*ahead_range).unsqueeze(-1)
                + perp * (side * _uniform(*lateral_range)).unsqueeze(-1)
            )
            pos[:, i, :2] = m * new_xy + (1.0 - m) * pos[:, i, :2]
            robot_yaw = torch.atan2(
                2.0 * (root_quat[:, 0] * root_quat[:, 3] + root_quat[:, 1] * root_quat[:, 2]),
                1.0 - 2.0 * (root_quat[:, 2] ** 2 + root_quat[:, 3] ** 2),
            )
            yaw[:, i] = torch.where(recycle, robot_yaw, yaw[:, i])

        wall: Entity = env.scene[name]
        ids = recycle.nonzero(as_tuple=False).squeeze(-1)
        quat = torch.zeros(len(ids), 4, device=device)
        quat[:, 0] = torch.cos(yaw[ids, i] / 2.0)
        quat[:, 3] = torch.sin(yaw[ids, i] / 2.0)
        wall.write_root_link_pose_to_sim(
            torch.cat([pos[ids, i], quat], dim=-1), env_ids=ids
        )
        wall.write_root_link_velocity_to_sim(
            torch.zeros(len(ids), 6, device=device), env_ids=ids
        )


# ------------------------------------------------------------------
# 相机俯仰注视预警：自动将头部云台对准来球
# ------------------------------------------------------------------

def optimal_camera_pitch(
    env: "ManagerBasedRlEnv",
    robot,
    ball,
    *,
    z_margin: float = 0.05,
    min_ball_speed: float = 0.5,
    sense_radius: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算头部云台相机朝向来球的最优俯仰角与威胁状态.

    返回:
      target_pitch: (N,) 在躯干局部系下的期望俯仰角 (截断在 ±30 度以内)
      threat: (N,) 是否存在滞空、逼近且在感知范围内的来球威胁
    """
    if not hasattr(env, "_gimbal_ids"):
        jid, _ = robot.find_joints("camera_pitch_joint")
        bid, _ = robot.find_bodies("camera_pitch_link")
        tid, _ = robot.find_bodies("torso_link")
        ball_scene_name = next(
            k for k, v in env.scene.entities.items() if v is ball
        )
        ball_geom_id = env.sim.mj_model.geom(f"{ball_scene_name}/ball_collision").id
        env._gimbal_ids = (
            torch.tensor(jid, device=env.device), int(bid[0]), int(ball_geom_id), int(tid[0])
        )
    jid, bid, ball_geom_id, tid = env._gimbal_ids

    cam = robot.data.body_link_pos_w[:, bid, :]
    bp = ball.data.root_link_pos_w
    bv = ball.data.root_link_lin_vel_w
    rp = robot.data.root_link_pos_w

    # 在躯干坐标系下度量球体的仰角，不受机器人整体俯仰倾斜影响
    torso_quat = robot.data.body_link_quat_w[:, tid, :]
    d_torso = quat_apply_inverse(torso_quat, bp - cam)
    horiz = torch.linalg.norm(d_torso[:, :2], dim=-1).clamp(min=1e-6)

    # 俯仰角：由于关节正旋转对应向下低头，取仰角的相反数
    elev = torch.atan2(d_torso[:, 2], horiz)
    target_pitch = (-elev).clamp(-math.radians(30), math.radians(30))

    ball_r = env.sim.model.geom_size[:, ball_geom_id, 0]
    speed = torch.linalg.norm(bv, dim=-1)
    closing = ((rp - bp) * bv).sum(-1) > 0.0
    in_range = torch.linalg.norm(d_torso, dim=-1) < sense_radius
    threat = (bp[:, 2] > ball_r + z_margin) & (speed > min_ball_speed) & closing & in_range

    return target_pitch, threat


def aim_camera_at_ball(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    robot_name: str = "robot",
    ball_name: str = "ball",
    z_margin: float = 0.05,
    min_ball_speed: float = 0.5,
    sense_radius: float = 4.0,
) -> None:
    """单步执行事件: 驱动头部 camera_pitch_joint 关节自动瞄准飞来的球体."""
    robot = env.scene[robot_name]
    ball = env.scene[ball_name]

    target_pitch, threat = optimal_camera_pitch(
        env, robot, ball,
        z_margin=z_margin,
        min_ball_speed=min_ball_speed,
        sense_radius=sense_radius,
    )

    # 有威胁时瞄准来球，无威胁时回平 (0 rad)
    target = torch.where(threat, target_pitch, torch.zeros_like(target_pitch))

    jid = env._gimbal_ids[0]
    robot.set_joint_position_target(target.unsqueeze(-1), joint_ids=jid)
