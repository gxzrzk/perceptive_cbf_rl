"""基于对抗运动先验（AMP）的速度跟踪任务通用基础环境配置。

本模块提供了一个工厂函数 `make_amp_env_cfg()`，用于构建基于 AMP 的人形机器人
速度跟踪基础强化学习环境（ManagerBasedRlEnvCfg）。
针对具体机器人（如宇树 Unitree G1）的配置脚本会调用该工厂函数，并根据特定硬件的刚体名称、
几何形状、执行器参数与动捕数据集路径进行定制。

核心架构与设计特点：
1. 传感器配置（Sensors）：包含用于地形高度感知的 RayCast 高度扫描雷达（地形高度场网格）；
2. 观测空间（Observations）：
   - `actor`: 纯本体感觉观测（IMU 基座角速度、重力投影矢量、速度跟踪指令、相对关节角度与角速度、历史动作），
     并叠加了高斯/均匀传感器噪声与 4 步历史堆叠（History Stacking），用于部署到真机；
   - `critic`: 非对称 Critic 观测，在 Actor 观测基础上补充了基座真实线速度以及各跟踪连杆在锚点机体（torso）
     坐标系下的相对位置与 6D 连续旋转姿态（特权信息）；
   - `amp`: 专门提供给 AMP 判别器的局部机体特征（各跟踪连杆相对躯干的位置、6D 旋转姿态、机体系线速度与角速度），
     严格与离线动捕数据格式对齐；
3. 动作空间（Actions）：关节位置残差控制（JointPositionActionCfg），频率 50Hz（仿真步长 0.005s，decimation=4）；
4. 指令系统（Commands）：全向平面速度指令（线速度 vx, vy 与偏航角速度 yaw）；
5. 随机化与事件（Events）：动捕数据状态初始化（RSI）、速度外力扰动、足部摩擦力随机化、编码器零偏与质心偏移；
6. 奖励函数（Rewards）：速度指令跟踪、躯干部高度维持、防止侧倾侧滚、足部打滑惩罚、自碰撞惩罚及动作平滑性约束；
7. 终止条件（Terminations）：超时（20s）、倾倒摔倒判定（倾角超过 70 度）以及躯干高度跌落惩罚。
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.amp_loco.mdp as mdp
from src.tasks.amp_loco.mdp.terrain import RANDOM_ROUGH_TERRAINS_CFG


def make_amp_env_cfg() -> ManagerBasedRlEnvCfg:
  """构建并返回通用的 AMP 双足/人形速度跟踪基础环境配置。"""

  ##
  # 1. 传感器配置 (Sensors)
  ##

  # 地形高度扫描光线传感器：从机器人基座向下发射网格光线探测前方高低起伏，支持崎岖地形行走
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # 在具体机器人配置中指定为 pelvis
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),  # 1.6m x 1.0m 探测区域，0.1m 分辨率
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # 2. 观测空间配置 (Observations)
  ##

  # 【Actor 策略网络观测组】：真机部署可获取的本体感觉（Proprioception）+ 目标速度指令，带噪声注入
  actor_terms = {
    # IMU 测量的机体基座角速度（添加均匀噪声模拟真实传感器误差）
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # 投影到机体局部坐标系下的重力向量（用于感知机体俯仰和滚转倾斜度）
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # 上层导航下发的三维期望速度指令 [vx, vy, yaw_rate]
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # 各自由度关节相对默认站立零位的角度偏差
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # 各自由度关节角速度
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # 上一时间步实际执行的控制动作向量
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # 【Critic 评价网络观测组】：非对称 Actor-Critic 架构，包含 Actor 观测及特权状态（Privileged Information）
  critic_terms = {
    **actor_terms,
    # 基座局部坐标系下的真实线速度（真机难以直接无漂移测量，仅供训练时 Critic 评估状态价值）
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    # 各跟踪连杆在锚点机体（torso）局部坐标系下的 3D 相对位置
    "body_pos_b": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    # 各跟踪连杆在锚点机体局部坐标系下的 6D 连续旋转矩阵前两列
    "body_ori_b": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  # 【AMP 判别器专用观测组】：提取用于与动捕数据计算判别相似度的局部身体构型特征
  amp_terms = {
    # 跟踪刚体相对躯干锚点的 3D 位置
    "body_pos_b": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    # 跟踪刚体相对躯干锚点的 6D 姿态表示
    "body_ori_b": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    # 跟踪刚体在机体自身局部坐标系下的线速度
    "body_lin_vel_b": ObservationTermCfg(
      func=mdp.robot_body_lin_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    # 跟踪刚体在机体自身局部坐标系下的角速度
    "body_ang_vel_b": ObservationTermCfg(
      func=mdp.robot_body_ang_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  # 汇总定义三个观测组
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=4,  # 堆叠最近 4 帧历史观测，增强策略的隐式系统辨识与动态估计能力
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=4,
    ),
    "amp": ObservationGroupCfg(
      terms=amp_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,  # AMP 观测以单步状态转移 (s_t, s_{t+1}) 形式成对输入判别器
    ),
  }

  ##
  # 3. 统计度量指标 (Metrics)
  ##

  metrics = {
    # 动作平均加速度度量（监控控制指令平滑程度）
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
    # 延迟步数统计
    "mean_delay_steps": MetricsTermCfg(
      func=mdp.mean_delay_steps,
    ),
  }

  ##
  # 4. 动作空间配置 (Actions)
  ##

  actions: dict[str, ActionTermCfg] = {
    # 目标关节位置控制：策略输出关节位置偏差，缩放后加到默认初始关节姿态上
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # 缩放系数（具体机器人配置中可能会进行覆盖）
      use_default_offset=True,
    )
  }

  ##
  # 5. 目标指令生成 (Commands)
  ##

  commands: dict[str, CommandTermCfg] = {
    # 均匀采样平面速度与航向角指令
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),  # 每 3 至 8 秒重新采样一次新指令
      rel_standing_envs=0.05,            # 5% 的环境采样零速度指令以学习静止原地站立
      rel_heading_envs=0.25,             # 25% 的环境启用航向角跟踪模式
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 3.0),          # 前进/后退线速度范围 [m/s]
        lin_vel_y=(-1.0, 1.0),          # 横向侧移线速度范围 [m/s]
        ang_vel_z=(-3.14 / 2, 3.14 / 2),# 转向角速度范围 [rad/s]
        heading=(-math.pi / 2, math.pi / 2), # 航向角范围 [rad]
      ),
    )
  }

  ##
  # 6. 环境重置与随机化事件 (Events)
  ##

  events = {
    # 启动时初始化动捕数据加载器
    "init_motion_loader": EventTermCfg(
      func=mdp.init_motion_loader,
      mode="startup",
      params={
        "motion_dir": "",  # 具体机器人在子配置中指定动捕文件路径
        "recovery_dir": None,
        "delay_reset_env_ratio": 0.0,
        "max_delay_steps": 0,
      },
    ),
    # 回合重置时采用参考状态初始化（RSI）：从动捕库随机采样姿态作为机器人初始状态
    "reset_from_motion": EventTermCfg(
      func=mdp.reset_from_motion_data,
      mode="reset",
      params={
        "motion_dir": "",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # 周期性对机器人基座施加外力速度突变扰动，增强抗推与鲁棒性
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),  # 每 1 至 3 秒施加一次推力
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    # 域随机化：足部与地面接触几何体摩擦系数随机化 (0.3 ~ 1.2)
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # 双足使用相同的随机摩擦系数
      },
    ),
    # 域随机化：关节编码器零偏随机化（模拟硬件安装误差）
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    # 域随机化：躯干质心偏移（COM offset）随机化（模拟负载与质量分布变化）
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),  # X 轴偏移
          1: (-0.025, 0.025),  # Y 轴偏移
          2: (-0.03, 0.03),    # Z 轴偏移
        },
      },
    ),
  }

  ##
  # 7. 任务奖励函数 (Rewards)
  ##

  rewards = {
    # 跟踪期望线速度（高斯核指数衰减形式）
    "track_anchor_linear_velocity": RewardTermCfg(
      func=mdp.track_anchor_linear_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 1.0,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    # 跟踪期望角速度/偏航转向
    "track_anchor_angular_velocity": RewardTermCfg(
      func=mdp.track_anchor_angular_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 3.14,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    # 跟踪基座目标高度（维持正常站立行走姿态高度）
    "track_root_height": RewardTermCfg(
      func=mdp.track_root_height,
      weight=1.0,
      params={
        "std": 0.3,
        "mask_delay": True,
        "delay_env_rew_ratio": 3.5,
      },
    ),
    # 惩罚骨盆横向与纵向角速度（抑制行走中的晃动与倾翻）
    "body_ang_vel_xy_l2": RewardTermCfg(
      func=mdp.body_ang_vel_xy_l2,
      weight=0.5,
      params={
        "std": 3.14,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "body_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      },
    ),

    # 摔倒非正常终止严重惩罚 (-200)
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # 惩罚过大关节加速度（提升电机寿命并抑制高频振动）
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    # 惩罚超出关节软限位范围的动作
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # 惩罚动作变化率（一阶差分，确保输出力矩平滑）
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),

    # 惩罚足部在触地支撑相时的打滑速度
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    # 惩罚机器人各肢体之间的自碰撞（防止左右腿相互绊倒）
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # 8. 终止条件判定 (Terminations)
  ##

  terminations = {
    # 达到最大时长后触发自然超时重置（不给负奖励）
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # 姿态严重偏离：身体姿态偏离竖直方向超过 70 度判定为摔倒
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    # 基座高度过低：骨盆低于 0.5 米判定为跪地或摔倒
    "bad_base_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.5},
    ),
  }

  ##
  # 9. 组装并返回标准 ManagerBasedRlEnvCfg 环境配置对象
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # 在各机器人具体配置中指定跟踪视角中心
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,  # 物理仿真步长 5ms
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,         # 控制策略每 4 个物理步执行一次（控制频率 50Hz）
    episode_length_s=20.0,# 单幕最大时长 20 秒
  )
