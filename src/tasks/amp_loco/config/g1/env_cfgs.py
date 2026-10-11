"""Unitree G1 人形机器人基础 AMP 运动控制环境配置（平地与崎岖地形）。

功能与架构定位：
本模块将通用 AMP 基础环境模板（`make_amp_env_cfg`）与宇树 Unitree G1 人形机器人的具体物理规格相绑定：
1. 机器人模型规格集成：
   - 载入 G1 的 MJCF 结构定义（`get_g1_robot_cfg`）与默认动作缩放比例（`G1_ACTION_SCALE`）；
   - 绑定 G1 的 13 个关键跟踪连杆（骨盆、左右髋膝踝、左右肩肘腕）与锚点刚体（`torso_link`）；
   - 映射左右足底 14 个碰撞网格几何体（每足 7 个）及触地接触传感器（`feet_ground_contact`）；
   - 部署机器人肢体间的自碰撞检测传感器（`self_collision`）；
2. 动捕数据集与初始重置（RSI）：
   - 绑定 G1 官方动捕数据集中的 `WalkandRun` 行走与奔跑片段；
   - 配置重置采样与延迟重置比例；
3. 地形配置划分：
   - `g1_amp_rough_env_cfg`：包含高度起伏的生成式崎岖地形（坡道、楼梯、粗糙地面），启用 RayCast 高度扫描雷达；
   - `g1_amp_flat_env_cfg`：无限平坦地面环境，精简计算资源并移除高度扫描雷达；
4. 评测演示模式（Play Mode）：
   - 支持可视化（Viser / MuJoCo Viewer）交互式速度操控，关闭观测域随机化，扩展单轮幕时长至近乎无穷大。
"""

import os

from src.assets.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg


def g1_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Unitree G1 崎岖地形（Rough Terrain）速度跟踪基础环境配置。

    Args:
        play: 是否为评估演示模式（若为 True，关闭推力扰动与观测噪声，延长回合时间）。

    Returns:
        ManagerBasedRlEnvCfg: 适配 G1 机器人的崎岖地形环境配置。
    """
    cfg = make_amp_env_cfg()

    # 仿真求解器数值稳定性设置：启用连续碰撞检测（CCD）并限制接触缓冲区大小，防止显存溢出
    cfg.sim.mujoco.ccd_iterations = 128
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = 48

    # 导入 G1 机器人模型配置实体
    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    # 将地形扫描传感器的基准参考点绑定至机器人骨盆（pelvis）
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "pelvis"

    # 足部点位与碰撞几何体定义
    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )
    # AMP 判别器跟踪的 13 个核心连杆名称
    body_names = (
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    )
    anchor_name = "torso_link"
    root_name = "pelvis"

    # 双足与地形地面触地接触力传感器
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # 机器人各连杆肢体间自碰撞检测传感器（主副匹配均为 robot 骨盆子树）
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    # 启用崎岖地形难度递进课程
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    # 动作输出缩放比例对齐 G1 规格
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    # 可视化相机跟踪视角绑定到机器人躯干
    cfg.viewer.body_name = "torso_link"

    # 速度指令可视化箭头高度偏移
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    # 绑定域随机化几何体与刚体名称
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    # 配置动捕重置采样与延迟重置（RSI）
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
    cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

    # 指定行走与奔跑动捕数据路径
    _motion_base = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "g1", "amp"
    )
    _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))

    cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
    cfg.events["init_motion_loader"].params["recovery_dir"] = None
    cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

    # 绑定各项奖励函数的锚点与连杆实体
    cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )
    cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

    # 绑定 Critic 观测项的刚体与锚点配置
    cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
    cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

    # 绑定 AMP 判别器专用观测项的刚体与锚点配置
    cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names
    cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names
    cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names
    cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

    # 评测演示模式覆盖项
    if play:
        cfg.episode_length_s = int(1e9)  # 事实上无限长回合
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

    return cfg


def g1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Unitree G1 平坦地面（Flat Terrain）速度跟踪基础环境配置。

    在崎岖地形配置基础上：
    1. 地形类型切换为无限大平坦平面（plane），移除地形生成器；
    2. 移除地形高度扫描传感器（terrain_scan），减轻物理引擎射线计算负担；
    3. 优化求解器参数（ccd_iterations=50, njmax=640），提升仿真吞吐量。
    """
    cfg = g1_amp_rough_env_cfg(play=play)

    cfg.sim.njmax = 640
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 256
    cfg.sim.nconmax = None

    # 地形切换为平面
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # 移除光线雷达传感器
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
        twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
        twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

    return cfg
