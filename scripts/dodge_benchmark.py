"""受控躲避球基准测试套件 (Controlled Dodge Benchmark)

本脚本用于对不同感知模态 (Perception) 与 CBF 安全屏障模式 (Safety Mode) 的策略进行标准化、可重复的横向对比评测。
每个策略在相同固定随机种子下接受 100 次受控抛球测试（50 次高抛 UP/需低头下潜 + 50 次低抛 DOWN/需侧移闪避）。
成功判定标准：在测试时间窗口内，机器人既未被球击中 (HIT) 也未失稳摔倒 (FELL)。

1. 策略矩阵体系与任务键值 (Perception x Safety Grid, 见 TASKS):
   - 感知模态 (Perception):
     * `state_*`: 状态真值 Oracle 策略（直接获取球体在自身航向坐标系下的 6 维相对位置与速度，无相机，性能上限标尺）。
     * `vision_*`: 固定俯仰角头部相机 (+20°)，输入经 EfficientTAM 分割后的 BallOnly 掩码深度图。
     * `gimbal_*`: 在 BallOnly 深度相机基础上搭载真值引导的动态俯仰云台 (+/-30°)，测量主动瞄准带来的视场收益。
   - 安全屏障模式 (Safety Mode):
     * `*_none`: 无 CBF 屏障基线，仅采用基于距离的球体躲避惩罚奖励 (`NO_LINK_CBF=1`)。
     * `*_link`: 带有各连杆碰撞间隙 CBF 屏障奖励（论文核心贡献，真机物理部署配置）。
     * `*_joint`: 带有关节空间 CBF 导向奖励 (`CBF_JOINT=1`)。
     * `+filter`: 对 `*_joint` 模型在评测时指定 `BENCH_CBF_FILTER=1`，激活运行期特权真值 CBF 投影安全滤波器。
     * `omni`: 360° 全向高速抛球校准组 (`state_link_omni`, `state_joint_omni`，训练配置 `OMNI_THROW=1`)，
       需通过 `--omni` 参数触发评测。
     * `statue`: 原地完全静止不动的“雕像”对照基线（无需模型权重，测量不闪避时的天然未命中概率底线）。

2. 抛球类型与动力学特性 (Throw Types):
   - UP (高抛球, force_high=True):
     从较低高度（约腰部）起抛，轨迹呈向上抛物线爬升至躯干与头部高度 -> 诱导机器人下潜、弯腰或低头躲避 (Duck)。
   - DOWN (低抛球, force_high=False):
     从约 2m 高度水平射出，在重力作用下下落穿过下半身与腿部区域 -> 诱导机器人横向侧跨步闪避 (Sidestep)。

3. 三种评估体制 (Evaluation Regimes):
   - 默认模式 (Reset regime):
     每个环境单次抛球，机器人从标准的初始平稳站立姿态出发（对应论文中的 "Reset" 评测）。
   - 部署恢复模式 (--walk-recover, Deployment regime):
     镜像真实真机部署运行回路：在抛球间隙，自动调用基础步态行走网络 (`Unitree-G1-AMP-Flat`，由 `train_walk.sh` 训练)，
     引导机器人自动走回并对齐初始站立工位，待完全平稳站立后再触发下一次抛球。
   - 连续模式 (--continuous):
     连续多轮次背靠背抛球（每轮交替 UP/DOWN），期间不重置机器人姿态，测试策略在连续动态冲击下的自稳定与流转能力。

4. 核心评测判定与统计逻辑 (Metrics & Statistics):
   - 成功条件 (Success): `not hit and not fell`。
   - 严谨的物理接触判定 (HIT):
     必须采用 MuJoCo 物理引擎在当前步产生的实际几何接触项 (`ball_hit`)，严禁使用粗糙的空间欧氏距离阈值（避免贴身擦过被误判）。
   - 跌倒判定 (FELL):
     综合环境中的姿态失稳 (`bad_orientation`)、质心过度跌落 (`bad_base_height`) 以及跪地塌陷 (`collapsed_crouch`) 终止项。
   - 深度域随机化规范:
     对所有深度视觉策略，评测时必须默认应用训练时的 sim2real 域随机化增广 (`--depth-aug True`)，避免分布外 (OOD) 评估失真。
   - 统计输出: 打印包含总成功率、高球成功率、低球成功率、命中率、摔倒率以及受击身体部位分布 (`hit_regions`) 的完整汇总表。

常用运行命令：
  uv run python scripts/dodge_benchmark.py                    # 运行雕像基线与所有标准策略 (Reset 模式)
  uv run python scripts/dodge_benchmark.py --walk-recover     # 运行真机部署模式 (带走回恢复)
  uv run python scripts/dodge_benchmark.py --only vision_link # 仅评测真机部署的 vision_link 策略
  BENCH_CBF_FILTER=1 uv run python scripts/dodge_benchmark.py --only state_joint   # 运行带真值 CBF 滤波的 state_joint
  uv run python scripts/dodge_benchmark.py --omni             # 评测 360° 全向抛球策略
"""
from __future__ import annotations
import argparse, math, os
from collections import Counter
from dataclasses import asdict

import torch

# 基础运动行走策略网络 (Unitree-G1-AMP-Flat) 所期望的本体感觉维度：29 关节 G1，历史步长 4，
# 按 term-major 展开 [角速度(3) 重力向量(3) 速度指令(3) 关节位置(29) 关节速度(29) 上步动作(29)] * 4 = 384 维。
# [英文对照: Base proprio dim the loco walk actor expects: 29-joint G1, history 4, total 384]
PROPRIO_BASE = 384


def _actor_cmd_slice(env):
    """计算 actor 观测向量中 'command' (速度控制指令) 项的 (起始偏移 offset, 特征宽度 width)。
    (offset, width of the 'command' term within the actor obs vector.)

    排布与切片细节:
    在 term-major 排布结构中，command 项位于 joint_pos / joint_vel 之前，
    因此无论是完整的 actor 观测向量，还是经过云台关节剔除后的切片观测向量，command 项的起始偏移量均完全相同。
    其特征宽度恒为 3 * 历史帧步长 (history_length)。
    """
    om = env.observation_manager
    names = list(om._group_obs_term_names["actor"])
    dims = [int(d[0]) for d in om._group_obs_term_dim["actor"]]
    off = 0
    for n, d in zip(names, dims):
        if n == "command":
            return off, d
        off += d
    raise RuntimeError("no 'command' term in the actor obs group")


def _home_recover_cmd(env, vel_deadband=0.5, omega_floor=1.0, pos_tol=0.35, yaw_tol=0.15,
                      turn_first_thresh=4.0, speed_tol=0.3, ang_speed_tol=1.0, height_tol=0.1):
    """计算机体坐标系下的速度控制指令 (N,3) = [vx, vy, wz] 以及各环境机器人的稳定就位掩码 settled，
    以“先对齐朝向后直线行进 (ROTATE-THEN-TRANSLATE)”控制逻辑引导机器人自主走回发球工位（初始原点与初始朝向）。
    (Body-frame velocity command [vx, vy, wz] + settled mask driving robot back home.)

    控制机制与动力学背景说明:
    - 用于 walk-recover 部署闭环：在每次避障抛球间隙，基础步态网络接管控制，使机器人自动走回并对齐测试原点，
      避免机器人在上一次避障留下的边缘或偏航状态下被动开始下一轮，忠实还原真机部署运行流程。
    - 边走边转逻辑: 本步态策略在原地自转 (纯旋转零线速) 时表现欠佳，但在行进移动过程中具有优秀的航向修正能力，
      因此默认允许平移与旋转同时进行；当航向误差需要修正时，施加保底角速度 omega_floor 保证快速对齐。
    - 速度死区与 P 控制: 距离原点较远时按比例控制器生成目标速度；当指令速度低于死区阈值 vel_deadband 时截断归零，
      避免在极小残余误差下引发高频原地踏步抖动。
    - 稳定站立就位判定: 必须同时满足空间位置误差 < pos_tol、航向偏差 < yaw_tol，且机器人处于低动量平稳直立状态
      （线速度、角速度接近零，机体达到标称站立高度，排除迈步中或跪倒失稳状态）。
    """
    from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat, wrap_to_pi
    robot = env.scene["robot"]
    tc = env.cfg.commands["twist"]  # 复用环境中的 twist 跟踪指令配置（GoToGoalCommandCfg 的增益与限幅参数）
    base_xy = robot.data.root_link_pos_w[:, :2]
    home_xy = env.scene.env_origins[:, :2]
    err_w = home_xy - base_xy
    dist = err_w.norm(dim=-1)
    err_w3 = torch.cat([err_w, torch.zeros_like(err_w[:, :1])], dim=-1)
    err_b = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), err_w3)[:, :2]
    # 原点标称朝向 = 机器人重置时的默认根节点四元数的偏航角 yaw (Home heading = yaw of spawn root quat)
    hq = robot.data.default_root_state[:, 3:7]  # (w, x, y, z)
    w, x, y, z = hq.unbind(-1)
    home_heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw_err = wrap_to_pi(home_heading - robot.data.heading_w)
    need_turn = yaw_err.abs() > yaw_tol

    # 转向控制：当朝向偏差超出容差时，向标称朝向施加强力旋转 (>= omega_floor rad/s) (Rotation P-control)
    omega_cap = max(tc.max_ang_vel_z, omega_floor)
    wz_mag = (tc.kp_yaw * yaw_err.abs()).clamp(min=omega_floor, max=omega_cap)
    wz = torch.where(need_turn, torch.sign(yaw_err) * wz_mag, torch.zeros_like(yaw_err))

    # 平移控制：朝向原点行进（P 比例控制，限幅截断，线性死区过滤）(Translation toward origin)
    # 先转后移策略：当偏航角偏差剧烈时压制前向线速度，优先转动纠正朝向
    v = tc.kp * err_b
    v[:, 0] = v[:, 0].clamp(-tc.max_lin_vel_x, tc.max_lin_vel_x)
    v[:, 1] = v[:, 1].clamp(-tc.max_lin_vel_y, tc.max_lin_vel_y)
    v[v.norm(dim=-1) < vel_deadband] = 0.0
    v[yaw_err.abs() > turn_first_thresh] = 0.0  # 仅当严重偏离航向时才完全静止原地先转向 (rotate-first only when grossly mis-facing)

    cmd = torch.zeros(env.num_envs, 3, device=err_w.device)
    cmd[:, :2] = v
    cmd[:, 2] = wz
    # 稳定站立就位判定 (STABLE STAND at home)：到达原点坐标，对齐标称朝向，且处于真实的低动量直立站姿
    # （线速度与角速度均接近零，机体高度达到站立标称高度，非奔跑迈步中或上一次避障残留的跪地失稳状态）。
    # 评测主循环仅在环境完全就位后才触发下一轮投球，确保机器人绝不会在恢复平衡途中或失衡踉跄时被动挨打。
    base_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
    ang_speed = robot.data.root_link_ang_vel_w.norm(dim=-1)
    stand_h = robot.data.default_root_state[:, 2]
    root_h = robot.data.root_link_pos_w[:, 2]
    standing = (base_speed < speed_tol) & (ang_speed < ang_speed_tol) & (root_h > stand_h - height_tol)
    settled = (dist < pos_tol) & (yaw_err.abs() < yaw_tol) & standing
    return cmd, settled


def region_of(name: str) -> str:
    n = name.lower()
    if "head" in n or "neck" in n:
        return "head"
    if any(k in n for k in ("shoulder", "elbow", "wrist", "hand", "arm")):
        return "arms"
    if any(k in n for k in ("hip", "knee", "ankle", "foot", "leg", "thigh", "calf")):
        return "legs"
    if any(k in n for k in ("pelvis", "torso", "waist", "spine", "trunk", "chest", "base")):
        return "torso"
    return "other"


_STATE_TASK = "Unitree-G1-AMP-Dodge-MimicKit-Flat"
_VISION_TASK = "Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat"

# 评测策略矩阵字典：key -> (任务注册 ID task_id, 实验目录名 experiment_name)。
# 实验目录名与 train_runs/<key>.sh 中的 EXP_NAME 严格对齐，通过 newest_ckpt(exp) 自动发现对应训练结果。
# gimbal_* 系列键值运行相同的 BallOnly 任务但搭载由真值动态瞄准的相机俯仰云台，
# 基准测试脚本在环境构建前自动设置环境变量 CAMERA_GIMBAL=1 与 CAMERA_PROPRIO=1 (见 _set_gimbal_env)。
TASKS = {
    # 状态真值基准组 (State oracle: ground-truth ball obs)，包含三种安全屏障模式 (none / link / joint):
    "state_none":   (_STATE_TASK, "state_none"),
    "state_link":   (_STATE_TASK, "state_link"),
    "state_joint":  (_STATE_TASK, "state_joint"),
    # 头部固定相机深度视觉组 (Fixed head camera, ball-only masked depth):
    "vision_none":  (_VISION_TASK, "vision_none"),
    "vision_link":  (_VISION_TASK, "vision_link"),
    "vision_joint": (_VISION_TASK, "vision_joint"),
    # 带有真值主动瞄准的俯仰云台相机组 (Oracle-aimed gimbal camera):
    "gimbal_none":  (_VISION_TASK, "gimbal_none"),
    "gimbal_link":  (_VISION_TASK, "gimbal_link"),
    "gimbal_joint": (_VISION_TASK, "gimbal_joint"),
    # 全向 360° 高速抛球校准组 (Omni-throw oracle calibration, 训练带 OMNI_THROW=1；必须搭配 --omni 进行评测):
    "state_link_omni":  (_STATE_TASK, "state_link_omni"),
    "state_joint_omni": (_STATE_TASK, "state_joint_omni"),
}


def _set_gimbal_env(key: str) -> None:
    """为指定任务键值设置或清除云台环境变量门控（在构建环境时读取生效）。
    (Set/clear the gimbal env-var gates for this key read at env-BUILD time.)
    """
    if key.startswith("gimbal"):
        os.environ["CAMERA_GIMBAL"] = "1"
        os.environ["CAMERA_PROPRIO"] = "1"
    else:
        os.environ.pop("CAMERA_GIMBAL", None)
        os.environ.pop("CAMERA_PROPRIO", None)
# 投掷动力学参数配置（严格镜像 dodge_env_cfgs.py 中 throw_ball_on_dwell 的注册参数）：
DIST = (2.0, 3.0); ANGLE_DEG = 25.0; FLIGHT = (0.58, 0.63)
H_DESC = (1.5, 2.3)                 # 下沉平射低球 (DOWN) 的初始出射高度区间 (descending launch height)
H_LOW = (0.4, 0.9)                  # 仰角高弧线球 (UP) 的初始起抛高度区间 (low-arc launch height)
# UP 仰角高球到达目标高度区间 (0.9, 1.3) —— 与训练环境完全对齐 (dodge_env_cfgs.py 采纳的规范)：
# 当目标高度 z_tgt > 1.4m 时小球会直接从机器人头顶越过（头部顶端约 1.3m + 球半径 0.1m），叠加 0.1m 瞄准噪声后，
# 旧版的 (1.0, 1.5) 区间会有约 26% 的抛球属于天生打不中的废球。(0.9, 1.3) 确保每次抛球都构成致命威胁（雕像中弹率约 97%），
# 且落在胸口/头部高度（必须执行下潜 duck 动作闪避）。--up-z 命令行参数可进行灵活覆盖。
H_TGT = (0.9, 1.3)
AIM_NOISE = 0.1; G = 9.81
# OMNI_* 全向高速抛球评测规范（镜像 OMNI_THROW=1 训练环境）：360° 世界坐标系方位角，距离 OMNI_DIST (8~10m)，
# 固定初速度 OMNI_SPEED (12~15m/s)，直接解析求解弹道命中机器人机体。
# 速度参数与 MimicKit 开源代码规范对齐 (12~15 m/s)；论文正文写作 20~25 m/s；可通过 --omni-speed 命令行进行覆盖。
OMNI_DIST = (8.0, 10.0); OMNI_SPEED = (12.0, 15.0)
OMNI_TZ_HIGH = (0.9, 1.3); OMNI_TZ_LOW = (0.3, 0.7)


def newest_ckpt(exp: str) -> str | None:
    d = f"logs/rsl_rl/{exp}"
    best, bt = None, -1.0
    for root, _, files in os.walk(d):
        for f in files:
            if f.startswith("model_") and f.endswith(".pt"):
                p = os.path.join(root, f); t = os.path.getmtime(p)
                if t > bt: best, bt = p, t
    return best


def yaw_quat(q):  # 将维度为 (..., 4) 的 wxyz 完整四元数提取为仅含航向角偏航的 yaw-only 四元数 (wxyz -> yaw-only quat)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    hz = yaw * 0.5
    out = torch.zeros_like(q); out[..., 0] = torch.cos(hz); out[..., 3] = torch.sin(hz)
    return out


def quat_apply(q, v):  # 使用四元数 q 对三维向量 v 进行空间旋转变换 (Rotate vector v by quaternion q using Rodrigues formula)
    w, x, y, z = q.unbind(-1)
    t = 2 * torch.cross(torch.stack([x, y, z], -1), v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(torch.stack([x, y, z], -1), t, dim=-1)


def inject_throw(env, up_mask, seed, h_tgt=None, omni=False):
    """为所有环境并行生成并注入受控抛球物理状态；up_mask[i]=True 对应可下潜闪避的高弧线球 (duckable arc)。
    (Replicate throw_ball_on_dwell's launch math for ALL envs; up_mask picks high duck vs low arc.)

    参数与动力学求解原理:
    - 统一随机种子: 使用指定的 seed 初始化发生器，确保不同策略评估时所承受的测试球序列物理参数完全相同。
    - omni=True: 激活 360° 全向高速投球模式（围绕机器人机体 360 度随机选取发射方位角，发射距离 OMNI_DIST，
      初速度 OMNI_SPEED，通过 solve_ballistic_velocity 逆向求解弹道使球击中机体目标高程）。
    - omni=False (正面投球): 在机器人正面 +/-25 度扇区内发射，飞行时间 FLIGHT (0.58~0.63s)；
      up_mask[i]=True 时从低位 (0.4~0.9m) 仰角高抛至胸颈高度 (0.9~1.3m)；False 时从高位 (1.5~2.3m) 平射下落穿过腿部。
    """
    h_tgt = h_tgt or H_TGT
    g = torch.Generator(device=env.device).manual_seed(seed)
    robot, ball = env.scene["robot"], env.scene["ball"]
    n = env.num_envs; dev = env.device
    rp = robot.data.root_link_pos_w; rq = robot.data.root_link_quat_w
    yq = yaw_quat(rq)
    U = lambda a, b: torch.rand(n, generator=g, device=dev) * (b - a) + a
    if omni:
        from src.tasks.amp_loco.mdp.events import solve_ballistic_velocity
        bearing = U(-math.pi, math.pi)                       # 在世界坐标系下 360 度随机选取发射方位角 (world-frame 360deg)
        dist = U(*OMNI_DIST)
        start = torch.empty(n, 3, device=dev)
        start[:, 0] = rp[:, 0] + dist * torch.cos(bearing)
        start[:, 1] = rp[:, 1] + dist * torch.sin(bearing)
        start[:, 2] = U(*H_DESC)                             # 随机出球高度 (release height)
        speed = U(*OMNI_SPEED)
        target = torch.empty(n, 3, device=dev)
        target[:, 0:2] = rp[:, 0:2] + AIM_NOISE * torch.randn(n, 2, generator=g, device=dev)
        target[:, 2] = torch.where(up_mask, U(*OMNI_TZ_HIGH), U(*OMNI_TZ_LOW))
        vel = solve_ballistic_velocity(start, target, speed, gravity=G)
        quat = torch.zeros(n, 4, device=dev); quat[:, 0] = 1.0
        ball.write_root_link_pose_to_sim(torch.cat([start, quat], -1))
        rv = torch.zeros(n, 6, device=dev); rv[:, 0:3] = vel
        ball.write_root_link_velocity_to_sim(rv)
        return
    high = up_mask
    dist = U(*DIST); angle = (U(-ANGLE_DEG, ANGLE_DEG)) * math.pi / 180.0
    lateral = dist * torch.tan(angle)
    offb = torch.stack([dist, lateral, torch.zeros_like(dist)], -1)
    offw = quat_apply(yq, offb)
    start = torch.empty(n, 3, device=dev)
    start[:, 0:2] = rp[:, 0:2] + offw[:, 0:2]
    start[:, 2] = torch.where(high, U(*H_LOW), U(*H_DESC))
    t_req = U(*FLIGHT)
    t_max = torch.sqrt((2.0 * (start[:, 2] - 0.05).clamp(min=1e-3) / G))
    t_flight = torch.where(high, t_req, torch.minimum(t_req, t_max))
    target_xy = rp[:, 0:2].clone() + AIM_NOISE * torch.randn(n, 2, generator=g, device=dev)
    disp = target_xy - start[:, 0:2]
    vel = torch.zeros(n, 3, device=dev)
    vel[:, 0] = disp[:, 0] / t_flight; vel[:, 1] = disp[:, 1] / t_flight
    z_tgt = U(*h_tgt)
    vz_high = (z_tgt - start[:, 2]) / t_flight + 0.5 * G * t_flight
    vel[:, 2] = torch.where(high, vz_high, torch.zeros_like(vz_high))
    quat = torch.zeros(n, 4, device=dev); quat[:, 0] = 1.0
    ball.write_root_link_pose_to_sim(torch.cat([start, quat], -1))
    rv = torch.zeros(n, 6, device=dev); rv[:, 0:3] = vel
    ball.write_root_link_velocity_to_sim(rv)


def run_task(key, task, exp, ckpt, args, dev, walk_recover=False):
    import src.tasks  # noqa: F401 (导入自定义任务模块以触发注册表装载)
    from src.tasks.amp_loco.config.g1 import dodge_env_cfgs as dec
    from src.tasks.amp_loco.mdp.events import reset_to_default_stand
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    builders = {
        _STATE_TASK:
            lambda play=True, depth_frame_offsets=None: dec.g1_amp_dodge_mimickit_flat_env_cfg(play=play),
        _VISION_TASK: dec.g1_amp_dodge_depth_single_ballonly_flat_env_cfg,
    }
    offs = tuple(int(x) for x in args.frame_offsets.split(","))
    ballonly = task == _VISION_TASK
    # gimbal_* 云台系列任务需要在调用配置构造器之前设定环境变量门控
    _set_gimbal_env(key)
    # BallOnly 深度视觉任务的训练期域随机化由 BALLONLY_AUG 门控（针对分割后的深度图注入噪声与轮廓形变）。
    # 在构建环境之前启用该环境变量，确保评测观测分布与模型训练时完全一致。
    # 关节 CBF 滤波器仅用于训练期的探索约束，在评测期默认关闭以评估纯裸 Actor 网络（与真机物理部署场景一致）。
    if ballonly and args.depth_aug:
        os.environ["BALLONLY_AUG"] = "1"
    # CBF 滤波器在评估时默认关闭（测试裸策略部署表现）；若显式设置 BENCH_CBF_FILTER=1，
    # 则在评估期保持关节空间 CBF 滤波器处于激活状态（对应论文中的 "+filter" 单元格，需要特权真值小球状态）。
    os.environ["CBF_JOINT"] = "1" if os.environ.get("BENCH_CBF_FILTER") == "1" else "0"
    cfg = builders[task](play=True, depth_frame_offsets=offs)
    cfg.scene.num_envs = args.n
    # 为所有并行环境设置全量原地驻留避障速度指令 (zero-velocity command, stand & dodge in place)
    cfg.commands["twist"].rel_inplace_throw_envs = 1.0
    cfg.commands["twist"].rel_standing_envs = 0.0
    # 统一使用标准默认平稳站立姿态重置（而非 RSI 动态位姿重置），消除初始状态扰动差异
    if "reset_from_motion" in cfg.events:
        cfg.events["reset_from_motion"].func = reset_to_default_stand
        cfg.events["reset_from_motion"].params = {}
    # 移除环境内部自动的驻留投球事件（由评测套件通过 inject_throw 自行注入完全受控的测试球序列）
    cfg.events.pop("throw_ball_on_dwell", None)
    # 严格保留环境的真实终止条件 (terminations)：我们捕获环境底层精确的单项触发（包括持续跪地的 collapsed_crouch），
    # 并对每个环境的首次事件进行锁存记录。终止后的自动重置是安全的 —— 已锁存的环境在后续步骤中不再参与统计。
    # 环境的 1000 步超时判定在约 100 步的评测窗口内绝不会触发，因此窗口期内仅会记录真实的被击中 (hit) 或摔倒 (fall)。

    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    envw = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(envw, asdict(agent_cfg), device=dev)
    runner.load(ckpt); policy = runner.get_inference_policy(device=dev)

    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    n = args.n
    up_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    up_mask[: n // 2] = True  # 前半部分环境分配为高抛球 UP（需低头下潜），后半部分为低抛球 DOWN（需侧移跨步）
    reg_up, reg_dn = Counter(), Counter()  # 按抛球类型统计被击中的身体部位分布 (region tally)
    link_ct = Counter()                    # 按具体连杆名称统计被击中次数 (link tally)
    ballz = []                             # 记录每次发生碰撞时小球的高度 (ball height at hit)

    # 提取环境中实际激活的物理终止判定项 (categorize the env's real termination terms)
    tm = env.termination_manager
    active = list(tm.active_terms)
    HIT_TERMS = [t for t in active if t == "ball_hit"]
    FALL_TERMS = [t for t in active if t in ("bad_orientation", "bad_base_height", "collapsed_crouch")]
    print(f"  termination terms: hit={HIT_TERMS} fall={FALL_TERMS} (all active={active})")

    obs, _ = envw.reset()
    cmd_norm = float(env.command_manager.get_term("twist").command[:, :2].norm(dim=-1).mean())
    hit = torch.zeros(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    trial_done = torch.zeros(n, dtype=torch.bool, device=dev)  # 单轮事件锁存掩码：当环境发生首次 hit 或 fall 终止事件后对其锁存并忽略后续状态

    def latch_step(up_now, pb, pbody):
        # 读取本仿真步中触发的具体终止项（记录每个环境的首次事件）。pb 与 pbody 为在步进前快照记录的小球与连杆世界坐标，
        # 确保记录的是接触发生瞬间的准确空间构型，避免读取步进后环境自动 reset 导致的球体复位漂移。
        # [英文对照: read env's exact per-term firings from THIS step; pb/pbody captured BEFORE step]
        live = ~trial_done
        hb = torch.zeros(n, dtype=torch.bool, device=dev)
        fb = torch.zeros(n, dtype=torch.bool, device=dev)
        for t in HIT_TERMS:
            hb |= tm.get_term(t).bool()
        for t in FALL_TERMS:
            fb |= tm.get_term(t).bool()
        new_hit = hb & live
        if bool(new_hit.any()):
            d = (pbody - pb.unsqueeze(1)).norm(dim=-1)  # 接触瞬间小球到机器人各连杆的距离矩阵 (n,L) (ball-to-link distance at contact)
            nearest = d.argmin(dim=1)
            for i in new_hit.nonzero(as_tuple=False).flatten().tolist():
                ln = body_names[int(nearest[i])]
                (reg_up if bool(up_now[i]) else reg_dn)[region_of(ln)] += 1
                link_ct[ln] += 1
                ballz.append(float(pb[i, 2]))
        hit.logical_or_(new_hit)
        fell.logical_or_(fb & live)
        trial_done.logical_or_(hb | fb)

    def snap():  # 在步进前快照小球与机器人连杆的世界坐标（用于接触瞬间的物理归因分析）
        return (env.scene["ball"].data.root_link_pos_w.clone(),
                robot.data.body_link_pos_w.clone())

    # DEPTH_HOLD_STEPS=N: 将深度观测图保持恒定 N 个策略步后再刷新，
    # 建模真实机载感知硬件更新率（ZED 相机 + EfficientTAM 语义分割推理约 10Hz，而底层控制策略运行在 50Hz，
    # 即 N=5 步刷新一次深度图，更忠实模拟机载延迟）。默认 1 表示每步均刷新。
    _hold_n = int(os.environ.get("DEPTH_HOLD_STEPS", "1"))
    if _hold_n > 1:
        _dkeys = [k for k in obs.keys() if "depth" in str(k).lower()]
        assert _dkeys, f"DEPTH_HOLD_STEPS set but no depth obs group in {list(obs.keys())}"
        _raw_policy = policy
        _hold = {"t": 0, "buf": None}

        def policy(o):  # noqa: F811 (重定义 policy 包装函数以实施深度观测保持；本体感觉仍以 50Hz 实时更新)
            if _hold["t"] % _hold_n == 0:
                _hold["buf"] = {k: o[k].clone() for k in _dkeys}
            else:
                for k in _dkeys:
                    o[k] = _hold["buf"][k]
            _hold["t"] += 1
            return _raw_policy(o)
        print(f"  [depth-hold] holding {_dkeys} for {_hold_n} policy steps (~{50 // _hold_n} fps perception)")

    # SAFETY_PROBE=1: 探针模式，在每个威胁时间步 (THREAT step) 记录 CBF 屏障旨在优化但命中率指标忽略的细粒度物理量：
    # 最近肢体与小球的表面间隙 clearance（即 CBF 安全裕度 h）以及机体动作剧烈程度 (关节角速度与动作变化率)。
    # 排除云台俯仰自由度以保证不同机体构型公平对比。仅用于诊断分析，不改变仿真推演逻辑。
    if os.environ.get("SAFETY_PROBE", "0") == "1":
        _cmd = env.command_manager.get_term("twist")
        _ball_s = env.scene["ball"]
        _bgid = env.sim.mj_model.geom(next(k for k, v in env.scene.entities.items()
                                            if v is _ball_s) + "/ball_collision").id
        # 机体关节索引掩码（剔除 camera_pitch 云台关节）以公平统计关节角速度与动作剧烈度
        _jn = list(robot.joint_names)
        _bjids = torch.tensor([i for i, n in enumerate(_jn) if n != "camera_pitch_joint"],
                              device=dev, dtype=torch.long)
        _atrm = env.action_manager.get_term("joint_pos")
        _an = list(_atrm._target_names)
        _baids = torch.tensor([i for i, n in enumerate(_an) if n != "camera_pitch_joint"],
                              device=dev, dtype=torch.long)
        _S = {"clear": [], "jvel": [], "arate": []}
        _sp_policy = policy

        def policy(o):  # noqa: F811 (重定义 policy 包装函数以在威胁步记录安全裕度与动作剧烈度)
            a = _sp_policy(o)
            th = _cmd._dodge_threat.bool()
            if bool(th.any()):
                bp = _ball_s.data.root_link_pos_w
                lp = robot.data.body_link_pos_w
                d = (lp - bp.unsqueeze(1)).norm(dim=-1)  # (N,L) 小球中心与全身连杆几何中心的距离矩阵
                r_ball = env.sim.model.geom_size[:, _bgid, 0]
                clear = d.min(dim=1).values - r_ball      # (N,) 小球到最近连杆表面的最小安全间隙 (表面间隙)
                jvel = robot.data.joint_vel[:, _bjids].norm(dim=-1)          # (N,) 机器人全身本体关节角速度范数
                arate = (a - env.action_manager.action)[:, _baids].norm(dim=-1)  # (N,) 策略动作一阶变化率
                _S["clear"].append(clear[th].detach().cpu())
                _S["jvel"].append(jvel[th].detach().cpu())
                _S["arate"].append(arate[th].detach().cpu())
            return a
        env._safety_stats = _S
        print("  [safety-probe] recording clearance / joint-vel / action-rate on threat steps")

    # REACTION_PROBE=1: 记录全时序轨迹（威胁标志、关节转速、小球最近距离），
    # 以测量策略的反应潜伏期 (Reaction Latency: 从威胁触发到产生规避动作的时差) 和反应提前量 (Lead: 躲避开始到球体最近点的时间差)。
    # 威胁状态来自特权 command._dodge_threat（所有机型统一物理基准），因此深度视觉策略测得的潜伏期真实包含了视觉感知处理流水线的延迟。
    if os.environ.get("REACTION_PROBE", "0") == "1":
        _cmdr = env.command_manager.get_term("twist")
        _ballr = env.scene["ball"]
        _jnr = list(robot.joint_names)
        _bjr = torch.tensor([i for i, n in enumerate(_jnr) if n != "camera_pitch_joint"],
                            device=dev, dtype=torch.long)
        _R = {"threat": [], "jvel": [], "dist": []}
        _rp_policy = policy

        def policy(o):  # noqa: F811 (重定义 policy 包装函数以记录完整反应时间序列分析数据)
            a = _rp_policy(o)
            bp = _ballr.data.root_link_pos_w
            lp = robot.data.body_link_pos_w
            dist = (lp - bp.unsqueeze(1)).norm(dim=-1).min(dim=1).values  # (N,) 小球中心到全身各连杆的最近距离 (closest-limb dist)
            jvel = robot.data.joint_vel[:, _bjr].norm(dim=-1)             # (N,) 机器人全身本体关节转速范数 (body joint speed)
            _R["threat"].append(_cmdr._dodge_threat.bool().detach().cpu())
            _R["jvel"].append(jvel.detach().cpu())
            _R["dist"].append(dist.detach().cpu())
            return a
        env._react_stats = _R
        env._react_dt = float(env.step_dt)
        print("  [reaction-probe] recording threat / jvel / ball-distance time series")

    # 在首个测试球发射前，让机器人原地静止稳定约 settle_seconds (默认 3 秒)，消除初始位姿重置带来的瞬态晃动；
    # 将时间秒数除以仿真步长 env.step_dt 换算为仿真步数 n_settle。
    # [英文对照说明: Settle for ~settle_seconds so robot has fully stabilized from reset pose before first throw]
    n_settle = round(args.settle_seconds / env.step_dt)

    def _settled_speed():
        return float(robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1).mean())

    if walk_recover:
        # ---- WALK-RECOVER: 真实还原物理部署闭环 (Deploy-faithful walk<->dodge switch) ----
        # 基础步态行走网络在抛球间歇负责走回发球站位并重新站稳；避障躲球网络负责处理来袭抛球。
        # 行走策略 = Unitree-G1-AMP-Flat (384 维本体感觉 -> 29 关节控制 MLP)，
        # 共享避障任务的 actor 观测组与 ACTION_SCALE（无增益切换，在当前分支下执行机构配置完全一致）。
        from src.tasks.amp_loco.rl.frozen_actor import load_frozen_actor
        from tensordict import TensorDict as TD
        import rsl_rl.utils.utils as _rsl_utils

        # 旧版本 rsl_rl 保存的检查点依赖 rsl_rl.utils.utils.Normalizer 类。
        # 在此注入占位桩类 (Stub) 确保 torch.load 能够无误反序列化加载旧检查点；随后 load_frozen_actor 仅提取权重字典。
        # [英文对照说明: Inject stub so torch.load can unpickle older rsl_rl checkpoints]
        if not hasattr(_rsl_utils, "Normalizer"):
            class _LegacyNormalizerStub:
                def __init__(self, *a, **kw): pass
                def __setstate__(self, d): self.__dict__.update(d)
            _rsl_utils.Normalizer = _LegacyNormalizerStub

        # 修补 load_frozen_actor 内部调用的 torch.load：旧版步态检查点采用 'model_state_dict' (旧格式) 而非 'actor_state_dict'。
        # 包装 torch.load 实现自动键名重构映射（严格对齐 MjlabOnPolicyRunner.load 的迁移逻辑），
        # 将旧版单独存储的 obs_norm_state_dict 自动转录为 obs_normalizer.*，确保 MLPModel 顺利完成权重加载。
        # [英文对照说明: Patch load_frozen_actor's torch.load: migrate legacy model_state_dict -> actor_state_dict]
        _orig_torch_load = torch.load

        def _patched_load(path, *a, **kw):
            ck = _orig_torch_load(path, *a, **kw)
            if "model_state_dict" in ck and "actor_state_dict" not in ck:
                # 旧格式检查点状态字典键名迁移 (Legacy format migration)
                msd = ck.pop("model_state_dict")
                asd = {}
                for k, v in msd.items():
                    if k.startswith("actor."):
                        asd[k.replace("actor.", "mlp.")] = v
                    elif k.startswith("actor_obs_normalizer."):
                        asd[k.replace("actor_obs_normalizer.", "obs_normalizer.")] = v
                    elif k in ("std", "log_std"):
                        asd[k] = v
                if "std" in asd:
                    asd["distribution.std_param"] = asd.pop("std")
                if "log_std" in asd:
                    asd["distribution.log_std_param"] = asd.pop("log_std")
                # 旧检查点将观测归一化统计量独立保存在 obs_norm_state_dict 中；
                # 在此将其重命名为 obs_normalizer.* 使得 MLPModel.load_state_dict 能够直接识别匹配
                obs_norm = ck.get("obs_norm_state_dict", {})
                for k, v in obs_norm.items():
                    asd[f"obs_normalizer.{k}"] = v
                ck["actor_state_dict"] = asd
            return ck

        torch.load = _patched_load
        _walk_ckpt = args.walk_ckpt or newest_ckpt("walk")
        if not _walk_ckpt:
            raise SystemExit(
                "--walk-recover needs a walk checkpoint: train one with ./train_walk.sh "
                "(experiment name 'walk') or pass --walk-ckpt <path to Unitree-G1-AMP-Flat model_N.pt>."
            )
        n_recover = round(args.recover_seconds / env.step_dt)
        n_dwell = round(args.prethrow_settle / env.step_dt)
        tot = {"up": 0, "dn": 0}; hc = {"up": 0, "dn": 0}; fc = {"up": 0, "dn": 0}
        with torch.inference_mode():
            # 使用避障策略执行初始站立稳定，使视觉深度环形缓存 (depth ring) 与观测归一化层预热稳定
            for _ in range(n_settle):
                obs, _, _, _ = envw.step(policy(obs))
            print(f"  settled {n_settle} steps (~{args.settle_seconds:.1f}s) with dodge policy; "
                  f"base speed {_settled_speed():.3f} m/s")

            # ---- GIMBAL 云台支持与本体感觉特征切片 ----
            # 当环境启用 CAMERA_PROPRIO=1 (云台) 时，其 "actor" 观测项在 joint_pos 与 joint_vel 中均包含了 camera_pitch_joint 自由度，
            # 使得 actor 观测组总维度达到 392 维，而基础步态行走网络期望的输入维度为标准的 384 维。
            # 构建列保留索引掩码 walk_keep，从历史步长 H 的每一个时间步中精准剔除 camera_pitch 对应的关节位置与速度分量，
            # 将输入截断对齐为 384 维送入行走网络；而避障网络继续接收完整的 392 维全量观测（不受任何影响）。
            #
            # 核心细节 -- MjLab "actor" 观测组采用 TERM-MAJOR (特征优先) 排布，而非按帧展平：
            # 每个特征项的历史序列紧密连续平铺 (term.buffer.reshape(N, -1) 生成 [t0, t1, ..., tH-1])，随后在项级别完成拼接：
            #   特征项排列: [ang_vel(3*H), grav(3*H), cmd(3*H), joint_pos(J*H), joint_vel(J*H), actions(29*H)]
            # 因此在 joint_pos / joint_vel 内部的数据排布维度为 (history, joint)：必须在 H 个历史帧副本中分别剔除索引 cam_idx。
            # （实测数值校验：J=30, cam_idx=29, 各项平铺维度依次为 12, 12, 12, 120, 120, 116，总计 392 维）。
            _actor_dim = int(obs["actor"].shape[-1])
            _robot = env.scene["robot"]
            _has_gimbal_joint = "camera_pitch_joint" in list(_robot.joint_names)
            walk_keep = None  # walk_keep 为 None 时表示非云台任务，直接透传标准的 384 维本体感觉 (non-gimbal 384 path)
            if _has_gimbal_joint and _actor_dim != PROPRIO_BASE:
                cam_idx = list(_robot.joint_names).index("camera_pitch_joint")
                om = env.observation_manager
                term_names = list(om._group_obs_term_names["actor"])
                term_flat_dims = [int(d[0]) for d in om._group_obs_term_dim["actor"]]
                H = int(env.cfg.observations["actor"].history_length)
                keep_idx = []
                off = 0
                for tname, fdim in zip(term_names, term_flat_dims):
                    base = fdim // H  # 本特征项在单帧历史副本中的维度 (per-history-copy width)
                    # 在每一个历史帧副本中，剔除 joint_pos 与 joint_vel 中的 camera_pitch 关节 (末位索引 cam_idx)。
                    # 云台环境中的动作空间保持为 29 维 (相机由真值闭环驱动，不包含在动作输出中)，因此动作历史项无需切片。
                    if tname in ("joint_pos", "joint_vel"):
                        for h in range(H):
                            for k in range(base):
                                if k != cam_idx:
                                    keep_idx.append(off + h * base + k)
                    else:
                        keep_idx.extend(range(off, off + fdim))
                    off += fdim
                walk_keep = torch.tensor(keep_idx, device=dev, dtype=torch.long)
                assert walk_keep.numel() == PROPRIO_BASE, (
                    f"gimbal proprio slice produced {walk_keep.numel()} != {PROPRIO_BASE}"
                )
                print(f"  [gimbal] slicing walk proprio {_actor_dim} -> {PROPRIO_BASE} "
                      f"(history={H}, camera_pitch_joint idx={cam_idx}, dropped from "
                      f"joint_pos+joint_vel x{H} history copies)")

            def _walk_obs(o):
                """从环境观测构建行走网络所需的 384 维本体感觉 TensorDict。
                (Build the 384-D proprio TensorDict the walk actor consumes from the env obs.)

                始终返回全新独立的克隆张量 (clone / index_select copy)，
                确保走回恢复循环在覆写速度指令切片时绝不污染环境底层的实时观测。
                """
                a = o["actor"]
                a = a.index_select(-1, walk_keep) if walk_keep is not None else a.clone()
                return TD({"actor": a}, batch_size=o.batch_size)

            cmd_off, cmd_w = _actor_cmd_slice(env)  # command 速度指令切片范围 (起始偏移 offset, 总宽度 width = 3 * H)

            # 在初始站立稳定之后再加载行走 Actor，确保观测 TensorDict 已完全构建填充（用于解析 obs_groups 并构建观测归一化层）。
            # 传入经过切片的 384 维观测，确保行走策略网络及其观测归一化层按标准的 384 维正确初始化。
            walk_actor, walk_groups = load_frozen_actor(
                teacher_task="Unitree-G1-AMP-Flat",
                checkpoint=_walk_ckpt,
                obs=_walk_obs(obs),
                num_actions=envw.num_actions,
                device=dev,
                obs_set="actor",
            )
            torch.load = _orig_torch_load  # 行走模型加载完毕后立即恢复原生 torch.load，消除全局猴子补丁副作用
            print(f"  walk-recover mode: n_recover={n_recover} steps (~{args.recover_seconds:.1f}s), "
                  f"n_dodge={args.window} steps per throw; walk_groups={walk_groups}")
            for r in range(args.rounds):
                # -- 恢复阶段 (Recovery phase): 行走策略驱动机器人走回发球站位（原点坐标与初始朝向），此阶段不投球且不计入评测。
                # 在每一步中将行走观测中的 command 切片覆写为“先转后移”对准原点的导航指令；
                # 当所有并行环境全部稳定就位 (settled) 时提前退出恢复循环并立即触发投球。 --
                for _ in range(n_recover):
                    cmd, settled = _home_recover_cmd(env)
                    if bool(settled.all()):
                        break
                    wo = _walk_obs(obs)
                    wo["actor"][:, cmd_off:cmd_off + cmd_w] = cmd.repeat(1, cmd_w // 3)
                    obs, _, _, _ = envw.step(walk_actor(wo))  # 采用确定性均值动作执行恢复行走 (deterministic mean)
                # 投球前静止稳定 (Pre-throw settle): 施加零速度指令保持 n_dwell 步，消除行走残余动量，确保机器人在完全站定后再接受抛球。
                for _ in range(n_dwell):
                    wo = _walk_obs(obs)
                    wo["actor"][:, cmd_off:cmd_off + cmd_w] = 0.0
                    obs, _, _, _ = envw.step(walk_actor(wo))
                # 恢复阶段结束清零锁存器（走回过程中的任何失足均不计入避障成绩）(Clear latches after recovery)
                hit.zero_(); fell.zero_(); trial_done.zero_()
                # -- 避障阶段 (Dodge phase): 注入受控抛球测试，切换为避障策略执行闪避并实时统计碰撞与跌倒 --
                up = (r % 2 == 0)
                up_now = torch.full((n,), up, dtype=torch.bool, device=dev)
                inject_throw(env, up_now, seed=args.seed + r, h_tgt=args.up_z, omni=getattr(args, 'omni', False))
                dodge_steps = 0
                for _ in range(args.window):
                    pb, pbody = snap()
                    obs, _, _, _ = envw.step(policy(obs))
                    latch_step(up_now, pb, pbody)
                    dodge_steps += 1
                kk = "up" if up else "dn"
                tot[kk] += n; hc[kk] += int(hit.sum()); fc[kk] += int(fell.sum())
                if r == 0:
                    print(f"  round 0: recover={n_recover} steps, dodge={dodge_steps} steps "
                          f"(up={up}); hits={int(hit.sum())}/{n} fell={int(fell.sum())}/{n}")
        def rate(d):  # 计算单次抛球维度的百分比比率 (per-throw fraction)
            return d["up"] / max(tot["up"], 1), d["dn"] / max(tot["dn"], 1), (d["up"] + d["dn"]) / max(tot["up"] + tot["dn"], 1)
        hu, hd, ha = rate(hc); fu, fd, fa = rate(fc)
        su, sd_, sa = 1 - hu - fu, 1 - hd - fd, 1 - ha - fa
        mode = (f"walk-recover ({args.rounds} rounds x {n} envs = {args.rounds*n} throws; "
                f"recover={n_recover}steps/{args.recover_seconds:.1f}s, dodge={args.window}steps)")
    elif not args.continuous:
        # ---- CONTROLLED 受控基准测试模式: 每个环境单次抛球，从平稳初始站姿出发；50 次 UP / 50 次 DOWN ----
        # TRAJ_DUMP=path.npz: 环境变量，用于完整记录单次受控抛球测试中全部环境的逐时间步时空轨迹
        # （根节点位姿、关节位置、小球状态、前置 0.5 秒预热导引帧、各环境测试结果与首次碰撞时间步），
        # 便于离线重放可视化或绘制高精度轨迹示意图，无需重新运行策略。
        _dump = os.environ.get("TRAJ_DUMP")
        _tr = None
        if _dump:
            _tr = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "ball_pos", "ball_vel",
                                   "depth_obs")}
            _ball_e = env.scene["ball"]
            _hit_step = torch.full((n,), -1, dtype=torch.long, device=dev)
            _bgid_d = env.sim.mj_model.geom(
                next(k for k, v in env.scene.entities.items() if v is _ball_e) + "/ball_collision").id
            _dkey = next((k for k in obs.keys() if "depth" in str(k).lower()), None)

            def _rec():
                _tr["root_pos"].append(robot.data.root_link_pos_w.cpu().numpy().copy())
                _tr["root_quat"].append(robot.data.root_link_quat_w.cpu().numpy().copy())
                _tr["joint_pos"].append(robot.data.joint_pos.cpu().numpy().copy())
                _tr["ball_pos"].append(_ball_e.data.root_link_pos_w.cpu().numpy().copy())
                _tr["ball_vel"].append(_ball_e.data.root_link_lin_vel_w.cpu().numpy().copy())
                if _dkey is not None:
                    _tr["depth_obs"].append(
                        obs[_dkey].cpu().numpy().astype("float16").copy())
        _lead = 25  # 抛球前记录的预热稳定时间步数（约 0.5 秒前置导引帧）(settle steps recorded before throw, ~0.5s lead-in)
        with torch.inference_mode():
            for _si in range(n_settle):        # 初始站立稳定并填充深度观测环形缓存（抛球前阶段，不进行事件锁存）
                obs, _, _, _ = envw.step(policy(obs))
                if _tr is not None and _si >= n_settle - _lead:
                    _rec()
            print(f"  settled {n_settle} steps (~{args.settle_seconds:.1f}s); base speed {_settled_speed():.3f} m/s")
            hit.zero_(); fell.zero_(); trial_done.zero_()   # 预热稳定阶段的终止事件不计入评测成绩，在此清零锁存器
            _throw_step = len(_tr["root_pos"]) if _tr is not None else 0
            inject_throw(env, up_mask, seed=args.seed, h_tgt=args.up_z, omni=getattr(args, 'omni', False))
            for _wi in range(args.window):
                pb, pbody = snap()
                obs, _, _, _ = envw.step(policy(obs))
                latch_step(up_mask, pb, pbody)
                if _tr is not None:
                    _rec()
                    _hit_step[(_hit_step < 0) & hit] = _throw_step + _wi
        if _tr is not None:
            import numpy as np
            if not _tr["depth_obs"]:
                _tr.pop("depth_obs")
            np.savez_compressed(
                _dump,
                **{k: np.stack(v) for k, v in _tr.items()},
                depth_key=str(_dkey),
                hit=hit.cpu().numpy(), fell=fell.cpu().numpy(),
                up_mask=up_mask.cpu().numpy(), hit_step=_hit_step.cpu().numpy(),
                ball_radius=env.sim.model.geom_size[:, _bgid_d, 0].cpu().numpy(),
                joint_names=np.array(list(robot.joint_names)),
                step_dt=float(env.step_dt), throw_step=_throw_step,
                seed=args.seed, window=args.window)
            print(f"  [traj-dump] {len(_tr['root_pos'])} steps x {n} envs -> {_dump}", flush=True)
        success = ~(hit | fell)
        def split(m):
            return (float(m[up_mask].float().mean()), float(m[~up_mask].float().mean()), float(m.float().mean()))
        su, sd_, sa = split(success); hu, hd, ha = split(hit); fu, fd, fa = split(fell)
        mode = "controlled (1 throw/env, stand-reset)"
    else:
        # ---- CONTINUOUS 连续流转基准测试模式: 背靠背连续多轮抛球，轮次之间不重置站立姿态（仅触发终止条件时由环境内部自动 reset）；
        # 每轮次交替投掷 UP/DOWN。每轮次记录锁存每个环境的首次事件并计入总成绩。 ----
        tot = {"up": 0, "dn": 0}; hc = {"up": 0, "dn": 0}; fc = {"up": 0, "dn": 0}
        with torch.inference_mode():
            for _ in range(n_settle):          # 仅在全部测试开始前执行一次初始站立稳定（轮次之间无重置与额外稳定时间）
                obs, _, _, _ = envw.step(policy(obs))
            print(f"  settled {n_settle} steps (~{args.settle_seconds:.1f}s); base speed {_settled_speed():.3f} m/s")
            # 节奏控制 (Cadence): 仅当前一个测试球在所有环境中均已结束结算 (RESOLVED，即落地或已越过机器人飞离、不再处于空中接近状态)
            # 时才触发下一球发射；随后给予固定时长的着陆后恢复时间 (post-resolve settle)，给予机器人真实的动态平衡恢复窗口而非机械节拍器。
            # CONT_POST_LAND_SETTLE (秒，默认 0.5s) = 球体失效后的暂停恢复时间；
            # CONT_MAX_FLY_STEPS 设定最大飞行等待步数 (默认 150 步 ≈ 3 秒)，防止卡住的小球导致主循环永久挂起。
            # 设置 CONT_FIXED_CADENCE=1 可切回旧版的固定 throw_every 步长节拍器模式。
            _fixed = os.environ.get("CONT_FIXED_CADENCE", "0") == "1"
            _post_land = round(float(os.environ.get("CONT_POST_LAND_SETTLE", "0.5")) / env.step_dt)
            _max_fly = int(os.environ.get("CONT_MAX_FLY_STEPS", "150"))
            _min_fly = 8  # 最小飞行步数：让物理引擎在检查 "resolved" 之前先完全施加发射初速度 (min fly steps)
            ball_e = env.scene["ball"]; robot_e = env.scene["robot"]
            for r in range(args.rounds):
                up = (r % 2 == 0)
                up_now = torch.full((n,), up, dtype=torch.bool, device=dev)
                inject_throw(env, up_now, seed=args.seed + r, h_tgt=args.up_z, omni=getattr(args, 'omni', False))
                hit.zero_(); fell.zero_(); trial_done.zero_()   # 每轮开始前清空当前轮次的事件锁存器 (per-round latch)
                if _fixed:
                    for _ in range(args.throw_every):
                        pb, pbody = snap()
                        obs, _, _, _ = envw.step(policy(obs))
                        latch_step(up_now, pb, pbody)
                else:
                    resolved = torch.zeros(n, dtype=torch.bool, device=dev)
                    for t in range(_max_fly):    # 阶段 A: 小球飞行阶段，持续步进直到所有环境的小球均已落地或越过机器人 (fly until every ball resolves)
                        pb, pbody = snap()
                        obs, _, _, _ = envw.step(policy(obs))
                        latch_step(up_now, pb, pbody)
                        bp = ball_e.data.root_link_pos_w; bv = ball_e.data.root_link_lin_vel_w
                        rp = robot_e.data.root_link_pos_w
                        closing = ((rp - bp) * bv).sum(-1) > 0.0
                        active = (bp[:, 2] > 0.15) & closing   # 判定小球处于空中并且仍在朝向机器人接近 (airborne AND still approaching)
                        if t >= _min_fly:
                            resolved |= ~active
                        if bool(resolved.all()):
                            break
                    for _ in range(_post_land):   # 阶段 B: 球体失效后的动态平衡恢复稳定阶段 (phase B: post-resolve recovery settle)
                        pb, pbody = snap()
                        obs, _, _, _ = envw.step(policy(obs))
                        latch_step(up_now, pb, pbody)
                kk = "up" if up else "dn"
                tot[kk] += n; hc[kk] += int(hit.sum()); fc[kk] += int(fell.sum())
        def rate(d):  # 计算单次抛球维度的百分比比率 (per-throw fraction)
            return d["up"] / max(tot["up"], 1), d["dn"] / max(tot["dn"], 1), (d["up"] + d["dn"]) / max(tot["up"] + tot["dn"], 1)
        hu, hd, ha = rate(hc); fu, fd, fa = rate(fc)
        su, sd_, sa = 1 - hu - fu, 1 - hd - fd, 1 - ha - fa
        _cad = "fixed throw_every" if _fixed else f"land+{os.environ.get('CONT_POST_LAND_SETTLE','0.5')}s"
        mode = f"continuous ({args.rounds} rounds x {n} envs = {args.rounds*n} throws, {_cad}, no stand-reset)"

    print(f"\n===== {key:13s} ({exp}) =====")
    print(f"  ckpt={os.path.basename(ckpt)}  cmd_norm={cmd_norm:.3f}  mode={mode}")
    print(f"  {'':10s} {'UP(duck)':>10s} {'DOWN(low)':>10s} {'ALL':>8s}")
    print(f"  {'success':10s} {su*100:9.0f}% {sd_*100:9.0f}% {sa*100:7.0f}%")
    print(f"  {'  hit':10s} {hu*100:9.0f}% {hd*100:9.0f}% {ha*100:7.0f}%")
    print(f"  {'  fell':10s} {fu*100:9.0f}% {fd*100:9.0f}% {fa*100:7.0f}%")
    # 被击中身体部位统计分析 (hit-location breakdown: 触发 ball_hit 时小球距离最近的身体区域)
    reg_all = reg_up + reg_dn
    order = ["legs", "torso", "arms", "head", "other"]
    tot_h = sum(reg_all.values())
    if tot_h:
        cells = "  ".join(f"{r}={100*reg_all[r]/tot_h:.0f}%" for r in order if reg_all[r])
        print(f"  hit regions ({tot_h} hits): {cells}")
        nu, nd = sum(reg_up.values()), sum(reg_dn.values())
        if nu:
            print(f"     UP   hits: " + "  ".join(f"{r}={100*reg_up[r]/nu:.0f}%" for r in order if reg_up[r]))
        if nd:
            print(f"     DOWN hits: " + "  ".join(f"{r}={100*reg_dn[r]/nd:.0f}%" for r in order if reg_dn[r]))
        top = "  ".join(f"{ln}={c}" for ln, c in link_ct.most_common(6))
        print(f"  nearest link: {top}")
        bz = torch.tensor(ballz)
        print(f"  ball height at hit (m): median={float(bz.median()):.2f}  range=[{float(bz.min()):.2f},{float(bz.max()):.2f}]")
    if hasattr(env, "_react_stats") and env._react_stats["threat"]:
        _th = torch.stack(env._react_stats["threat"])   # (T, N)
        _jv = torch.stack(env._react_stats["jvel"])      # (T, N)
        _ds = torch.stack(env._react_stats["dist"])      # (T, N)
        _dt = env._react_dt
        _dd = os.environ.get("REACTION_DUMP_DIR")
        if _dd:
            torch.save({"threat": _th, "jvel": _jv, "dist": _ds, "dt": _dt, "key": key},
                       os.path.join(_dd, f"react_{key}.pt"))
            print(f"  [reaction] dumped series to {_dd}/react_{key}.pt")
        T_, N_ = _th.shape
        _JT = 2.0  # 机体关节转速阈值 (rad/s)：用于区分明显的规避机动与静止微小晃动 (clearly-evasive vs idle)
        lat, lead = [], []
        for i in range(N_):
            thr = _th[:, i]
            if not bool(thr.any()):
                continue
            t0 = int(thr.float().argmax())               # 首次判定威胁的时间步 (first threat step)
            react = (_jv[t0:, i] > _JT).nonzero(as_tuple=False)
            if react.numel() == 0:
                continue
            tr = t0 + int(react[0])                       # 策略产生规避动作的首次时间步 (first evasive-motion step)
            tc = t0 + int(_ds[t0:, i].argmin())           # 小球距离机体最近的时间步 (closest-approach step)
            lat.append((tr - t0) * _dt)
            lead.append((tc - tr) * _dt)
        if lat:
            lat_t = torch.tensor(lat); lead_t = torch.tensor(lead)
            print(f"  [reaction] n={lat_t.numel()} throws (jvel>{_JT} rad/s) | "
                  f"latency(s) med={lat_t.median():.3f} mean={lat_t.mean():.3f} p90={torch.quantile(lat_t, 0.9):.3f} | "
                  f"lead-before-closest(s) med={lead_t.median():.3f} mean={lead_t.mean():.3f} | "
                  f"reacted-before-closest={100 * (lead_t > 0).float().mean():.0f}%")
    if hasattr(env, "_safety_stats") and env._safety_stats["clear"]:
        _S = {k: torch.cat(v) for k, v in env._safety_stats.items()}
        cl, jv, ar = _S["clear"], _S["jvel"], _S["arate"]
        clq = torch.quantile(cl, torch.tensor([0.05, 0.10, 0.50]))
        print(f"  [safety] n={cl.numel()} threat-steps | "
              f"clearance(m) p05={clq[0]:.3f} p10={clq[1]:.3f} med={clq[2]:.3f} "
              f"frac<0.15m={(cl < 0.15).float().mean() * 100:.0f}% | "
              f"jointvel(rad/s) mean={jv.mean():.2f} p95={torch.quantile(jv, 0.95):.2f} | "
              f"actionrate mean={ar.mean():.3f} p95={torch.quantile(ar, 0.95):.3f}")
    env.close() if hasattr(env, "close") else None
    del env, envw, runner, policy
    torch.cuda.empty_cache()
    return dict(key=key, success=sa, success_up=su, success_down=sd_, hit=ha, fell=fa,
                cmd_norm=cmd_norm, hit_regions={r: reg_all[r] for r in order if reg_all[r]})


def run_statue(args, dev):
    """“原地静止雕像”对照基线评测 (Frozen-statue CONTROL):
    机器人在每一步都被运动学固定在默认直立站姿（不能做任何躲避动作，也绝不会倒地），向其投掷完全相同的测试球序列。
    该模式下的被命中率反映了测试球序列在几何上真正“命中靶心”的天然比例 —— 亦即任何策略天然享有的“免费未命中底线 (free-miss floor)”。
    在状态环境（无相机）下运行：抛球物理轨迹与策略无关。
    """
    import src.tasks  # noqa: F401 (导入以触发自定义任务注册)
    from src.tasks.amp_loco.config.g1 import dodge_env_cfgs as dec
    from src.tasks.amp_loco.mdp.events import reset_to_default_stand
    from mjlab.envs import ManagerBasedRlEnv

    n = args.n
    cfg = dec.g1_amp_dodge_mimickit_flat_env_cfg(play=True)
    cfg.scene.num_envs = n
    cfg.commands["twist"].rel_inplace_throw_envs = 1.0
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.events["reset_from_motion"].func = reset_to_default_stand
    cfg.events["reset_from_motion"].params = {}
    cfg.events.pop("throw_ball_on_dwell", None)
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    robot, tm = env.scene["robot"], env.termination_manager
    acts = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    up_mask = torch.zeros(n, dtype=torch.bool, device=dev); up_mask[: n // 2] = True

    env.reset()
    pos0 = robot.data.root_link_pos_w.clone(); quat0 = robot.data.root_link_quat_w.clone()
    jp0 = robot.data.joint_pos.clone(); jv0 = torch.zeros_like(robot.data.joint_vel)
    zero6 = torch.zeros(n, 6, device=dev)

    def freeze():
        robot.write_root_link_pose_to_sim(torch.cat([pos0, quat0], dim=-1))
        robot.write_root_link_velocity_to_sim(zero6)
        robot.write_joint_state_to_sim(jp0, jv0)

    def play_round(up_now, seed, steps):
        inject_throw(env, up_now, seed=seed, h_tgt=args.up_z, omni=getattr(args, 'omni', False))
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        for _ in range(steps):
            freeze(); env.step(acts)
            done |= tm.get_term("ball_hit").bool()
        return done

    with torch.inference_mode():
        for _ in range(10):
            freeze(); env.step(acts)
        if not args.continuous:
            hit = play_round(up_mask, args.seed, args.window)
            hu, hd = float(hit[up_mask].float().mean()), float(hit[~up_mask].float().mean())
            ha = float(hit.float().mean())
        else:
            tot = {"up": 0, "dn": 0}; hc = {"up": 0, "dn": 0}
            for r in range(args.rounds):
                up = (r % 2 == 0)
                done = play_round(torch.full((n,), up, dtype=torch.bool, device=dev),
                                  args.seed + r, args.throw_every)
                kk = "up" if up else "dn"
                tot[kk] += n; hc[kk] += int(done.sum())
            hu, hd = hc["up"] / max(tot["up"], 1), hc["dn"] / max(tot["dn"], 1)
            ha = (hc["up"] + hc["dn"]) / max(tot["up"] + tot["dn"], 1)
    su, sd_, sa = 1 - hu, 1 - hd, 1 - ha
    print(f"\n===== {'statue':13s} (frozen control -- hit rate = fraction of throws ON-TARGET) =====")
    print(f"  {'':10s} {'UP(duck)':>10s} {'DOWN(low)':>10s} {'ALL':>8s}")
    print(f"  {'success':10s} {su*100:9.0f}% {sd_*100:9.0f}% {sa*100:7.0f}%")
    print(f"  {'  hit':10s} {hu*100:9.0f}% {hd*100:9.0f}% {ha*100:7.0f}%")
    env.close() if hasattr(env, "close") else None
    del env
    torch.cuda.empty_cache()
    return dict(key="statue", success=sa, success_up=su, success_down=sd_, hit=ha, fell=0.0,
                cmd_norm=0.0, hit_regions={})


def render_task(key, task, exp, ckpt, args, dev):
    """为指定策略录制并渲染 walk-recover (走回恢复 + 躲球) 高清评估视频 (单环境，render_mode='rgb_array')。
    (Render a walk-recover video for one policy: num_envs=1, render_mode='rgb_array'.)

    核心逻辑与可视化规格:
    - 第三人称跟拍视角: 输出追踪机器人机体的高清 MP4 视频，记录连续多轮次从站立/闪避 -> 步态走回对齐原点 -> 再次站立稳定 -> 闪避抛球的全流程。
    - 抛球序列组合: 支持 50 次 UP + 50 次 DOWN 抛球，可通过伪随机数种子随机交错混洗打乱或按固定交替轮次进行。
    - 逻辑严格复用: 渲染路径完整复用 run_task(..., walk_recover=True) 的真实控制流，并在每个仿真步通过 env.render()
      提取画面，最后由 imageio 编码写出为高质量 MP4 视频文件。保留完整的指标统计结算逻辑，运行完毕后打印表格。

    参数说明:
        key: 策略任务标识键名 (例如 "vision_link" 或 "gimbal_joint")
        task: 任务注册 ID 字符串
        exp: 实验目录名称
        ckpt: 待评估策略的模型权重检查点文件路径
        args: 命令行解析参数命名空间
        dev: 计算与渲染设备字符串 (例如 "cuda:0" 或 "cpu")
    """
    import random as _random
    import numpy as np
    import imageio
    import src.tasks  # noqa: F401 (导入自定义任务模块以触发注册表填充)
    from src.tasks.amp_loco.config.g1 import dodge_env_cfgs as dec
    from src.tasks.amp_loco.mdp.events import reset_to_default_stand
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.viewer.viewer_config import ViewerConfig

    out_path = args.render_video
    n_throws = args.render_throws
    randomized = args.render_randomized

    builders = {
        _STATE_TASK:
            lambda play=True, depth_frame_offsets=None: dec.g1_amp_dodge_mimickit_flat_env_cfg(play=play),
        _VISION_TASK: dec.g1_amp_dodge_depth_single_ballonly_flat_env_cfg,
    }
    offs = tuple(int(x) for x in args.frame_offsets.split(","))
    ballonly = task == _VISION_TASK
    _set_gimbal_env(key)
    if ballonly and args.depth_aug:
        os.environ["BALLONLY_AUG"] = "1"
    os.environ["CBF_JOINT"] = "1" if os.environ.get("BENCH_CBF_FILTER") == "1" else "0"

    cfg = builders[task](play=True, depth_frame_offsets=offs)
    # 视频录制采用单个环境实例 (Single environment for rendering)
    cfg.scene.num_envs = 1
    cfg.commands["twist"].rel_inplace_throw_envs = 1.0
    cfg.commands["twist"].rel_standing_envs = 0.0
    # 在恢复阶段导出的视频中隐藏画面调试标记：关闭原点目标球与速度箭头显示
    # (GoToGoalCommand 仅在其 command 项的 debug_vis 开启时才在画面上绘制这两者)
    cfg.commands["twist"].debug_vis = False
    if "reset_from_motion" in cfg.events:
        cfg.events["reset_from_motion"].func = reset_to_default_stand
        cfg.events["reset_from_motion"].params = {}
    cfg.events.pop("throw_ball_on_dwell", None)

    # 配置离屏渲染相机视角：相机跟随机器人根节点刚体 (robot root body)，设置为第三人称斜后方视角
    # 方位角 Azimuth 135°：能够清晰俯瞰机器人全身以及小球从左前方袭来的完整飞行弧线；跟拍距离 4.0 米
    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=4.0,
        elevation=-20.0,
        azimuth=135.0,
        lookat=(0.0, 0.0, 0.8),
        height=720,
        width=1280,
        max_extra_envs=0,
        enable_shadows=False,
        enable_reflections=False,
    )

    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array")
    envw = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(envw, asdict(agent_cfg), device=dev)
    runner.load(ckpt); policy = runner.get_inference_policy(device=dev)

    n = 1  # 单个并行环境 (single env)
    tm = env.termination_manager
    active = list(tm.active_terms)
    HIT_TERMS = [t for t in active if t == "ball_hit"]
    FALL_TERMS = [t for t in active if t in ("bad_orientation", "bad_base_height", "collapsed_crouch")]
    print(f"  [render] termination terms: hit={HIT_TERMS} fall={FALL_TERMS}")

    # 构造带固定随机种子的高低抛球 UP/DOWN 序列 (Build seeded UP/DOWN sequence)
    if randomized:
        n_up = n_throws // 2; n_dn = n_throws - n_up
        up_seq = [True] * n_up + [False] * n_dn
        _random.Random(args.seed).shuffle(up_seq)
    else:
        up_seq = [(r % 2 == 0) for r in range(n_throws)]

    # 走回恢复模式配置与策略加载（镜像 run_task 中的 walk_recover 逻辑）
    from src.tasks.amp_loco.rl.frozen_actor import load_frozen_actor
    from tensordict import TensorDict as TD
    import rsl_rl.utils.utils as _rsl_utils

    if not hasattr(_rsl_utils, "Normalizer"):
        class _LegacyNormalizerStub:
            def __init__(self, *a, **kw): pass
            def __setstate__(self, d): self.__dict__.update(d)
        _rsl_utils.Normalizer = _LegacyNormalizerStub

    _orig_torch_load = torch.load
    def _patched_load(path, *a, **kw):
        ck = _orig_torch_load(path, *a, **kw)
        if "model_state_dict" in ck and "actor_state_dict" not in ck:
            msd = ck.pop("model_state_dict")
            asd = {}
            for k, v in msd.items():
                if k.startswith("actor."):
                    asd[k.replace("actor.", "mlp.")] = v
                elif k.startswith("actor_obs_normalizer."):
                    asd[k.replace("actor_obs_normalizer.", "obs_normalizer.")] = v
                elif k in ("std", "log_std"):
                    asd[k] = v
            if "std" in asd:
                asd["distribution.std_param"] = asd.pop("std")
            if "log_std" in asd:
                asd["distribution.log_std_param"] = asd.pop("log_std")
            obs_norm = ck.get("obs_norm_state_dict", {})
            for k, v in obs_norm.items():
                asd[f"obs_normalizer.{k}"] = v
            ck["actor_state_dict"] = asd
        return ck
    torch.load = _patched_load

    _walk_ckpt = args.walk_ckpt or newest_ckpt("walk")
    if not _walk_ckpt:
        raise SystemExit(
            "the render path needs a walk checkpoint: train one with ./train_walk.sh "
            "(experiment name 'walk') or pass --walk-ckpt <path to Unitree-G1-AMP-Flat model_N.pt>."
        )
    n_settle = round(args.settle_seconds / env.step_dt)
    n_recover = round(args.recover_seconds / env.step_dt)
    n_dwell = round(args.prethrow_settle / env.step_dt)

    # 逐次抛球结果统计容器 (Per-throw tally: list of dicts per throw)
    results = []
    frames = []

    up_dev = torch.zeros(1, dtype=torch.bool, device=dev)

    obs, _ = envw.reset()

    with torch.inference_mode():
        # 使用避障策略执行初始站立稳定并录制帧 (Initial settle with dodge policy)
        print(f"  [render] settling {n_settle} steps with dodge policy ...")
        for _ in range(n_settle):
            obs, _, _, _ = envw.step(policy(obs))
            frames.append(env.render())

        # 针对行走策略 Actor 的云台观测切片逻辑（严格镜像 run_task 逻辑）(Gimbal slice logic for walk actor)
        _actor_dim = int(obs["actor"].shape[-1])
        _robot = env.scene["robot"]
        _has_gimbal_joint = "camera_pitch_joint" in list(_robot.joint_names)
        walk_keep = None
        if _has_gimbal_joint and _actor_dim != PROPRIO_BASE:
            cam_idx = list(_robot.joint_names).index("camera_pitch_joint")
            om = env.observation_manager
            term_names = list(om._group_obs_term_names["actor"])
            term_flat_dims = [int(d[0]) for d in om._group_obs_term_dim["actor"]]
            H = int(env.cfg.observations["actor"].history_length)
            keep_idx = []
            off = 0
            for tname, fdim in zip(term_names, term_flat_dims):
                base = fdim // H
                if tname in ("joint_pos", "joint_vel"):
                    for h in range(H):
                        for ki in range(base):
                            if ki != cam_idx:
                                keep_idx.append(off + h * base + ki)
                else:
                    keep_idx.extend(range(off, off + fdim))
                off += fdim
            walk_keep = torch.tensor(keep_idx, device=dev, dtype=torch.long)
            assert walk_keep.numel() == PROPRIO_BASE
            print(f"  [render] gimbal: slicing walk proprio {_actor_dim} -> {PROPRIO_BASE}")

        def _walk_obs(o):
            # 返回全新的独立张量，确保恢复循环覆写速度指令切片时安全无副作用 (Fresh tensor for safe overwrite)
            a = o["actor"]
            a = a.index_select(-1, walk_keep) if walk_keep is not None else a.clone()
            return TD({"actor": a}, batch_size=o.batch_size)

        cmd_off, cmd_w = _actor_cmd_slice(env)  # 速度指令切片范围 (起始偏移 offset, 总宽度 width = 3 * H)

        walk_actor, walk_groups = load_frozen_actor(
            teacher_task="Unitree-G1-AMP-Flat",
            checkpoint=_walk_ckpt,
            obs=_walk_obs(obs),
            num_actions=envw.num_actions,
            device=dev,
            obs_set="actor",
        )
        torch.load = _orig_torch_load
        print(f"  [render] walk actor loaded; walk_groups={walk_groups}")
        print(f"  [render] starting {n_throws} throws (recover={n_recover}steps, dodge={args.window}steps)")

        _dbg_home = os.environ.get("BENCH_DEBUG_HOME") == "1"

        def _dist_home():
            rb = env.scene["robot"].data.root_link_pos_w[:, :2]
            return float((rb - env.scene.env_origins[:, :2]).norm(dim=-1).mean())

        def _yaw_err_home():
            from mjlab.utils.lab_api.math import wrap_to_pi
            rob = env.scene["robot"]
            hq = rob.data.default_root_state[:, 3:7]
            w, x, y, z = hq.unbind(-1)
            hh = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return float(wrap_to_pi(hh - rob.data.heading_w).abs().mean())

        def _heading_info():
            import math as _m
            from mjlab.utils.lab_api.math import wrap_to_pi
            rob = env.scene["robot"]
            hq = rob.data.default_root_state[:, 3:7]
            w, x, y, z = hq.unbind(-1)
            hh = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            hw = rob.data.heading_w
            err = wrap_to_pi(hh - hw)
            return (_m.degrees(float(hh.mean())), _m.degrees(float(hw.mean())),
                    _m.degrees(float(err.mean())))

        for r in range(n_throws):
            # 走回恢复阶段：走回环境原点工位，设置 0.5 m/s 速度死区 (Recovery phase: walk back to station)
            if _dbg_home:
                d0, a0 = _dist_home(), _yaw_err_home()
            if _dbg_home:
                hh0, hw0, se0 = _heading_info()
            used = 0
            for _ in range(n_recover):
                cmd, settled = _home_recover_cmd(env)
                if bool(settled.all()):
                    break
                wo = _walk_obs(obs)
                wo["actor"][:, cmd_off:cmd_off + cmd_w] = cmd.repeat(1, cmd_w // 3)
                obs, _, _, _ = envw.step(walk_actor(wo))
                frames.append(env.render())
                used += 1
            # 投球前静止稳定：保持零速度指令使残余动量阻尼衰减 (Pre-throw settle: hold zero velocity command)
            for _ in range(n_dwell):
                wo = _walk_obs(obs)
                wo["actor"][:, cmd_off:cmd_off + cmd_w] = 0.0
                obs, _, _, _ = envw.step(walk_actor(wo))
                frames.append(env.render())
            if _dbg_home:
                hh1, hw1, se1 = _heading_info()
                gated = "settled" if used < n_recover else "MAX (not settled)"
                print(f"  [home] throw {r+1}: dist {d0:.2f}->{_dist_home():.2f} m, "
                      f"head {hw0:+.0f}->{hw1:+.0f} deg (home {hh0:+.0f}), "
                      f"recover {used}/{n_recover} steps [{gated}] + {n_dwell} dwell", flush=True)

            # 注入测试球并进入避障阶段 (Inject throw + dodge phase)
            up = up_seq[r]
            up_dev.fill_(up)
            inject_throw(env, up_dev, seed=args.seed + r, h_tgt=args.up_z, omni=getattr(args, 'omni', False))

            hit_r = False; fell_r = False
            for _ in range(args.window):
                obs, _, _, _ = envw.step(policy(obs))
                frames.append(env.render())
                # 结算：读取环境各终止项触发状态 (Tally: read terminations)
                hb = any(tm.get_term(t).bool()[0].item() for t in HIT_TERMS)
                fb = any(tm.get_term(t).bool()[0].item() for t in FALL_TERMS)
                if hb and not hit_r: hit_r = True
                if fb and not fell_r: fell_r = True
            results.append(dict(round=r, up=up, hit=hit_r, fell=fell_r,
                                success=not hit_r and not fell_r))
            if r % 10 == 0 or r == n_throws - 1:
                suc = sum(x["success"] for x in results)
                print(f"  [render] throw {r+1}/{n_throws}: up={up} hit={hit_r} fell={fell_r} | "
                      f"running success {suc}/{len(results)}")

    # 编码写出 MP4 视频文件 (Write MP4)
    print(f"  [render] writing {len(frames)} frames to {out_path} ...")
    valid_frames = [f for f in frames if f is not None]
    print(f"  [render] {len(valid_frames)}/{len(frames)} frames are non-None")
    if valid_frames:
        imageio.mimwrite(out_path, valid_frames, fps=30, macro_block_size=1)
        sz = os.path.getsize(out_path)
        print(f"  [render] wrote {out_path} ({sz/1e6:.1f} MB, {len(valid_frames)} frames, "
              f"{len(valid_frames)/30:.1f}s @ 30fps)")
    else:
        print("  [render] ERROR: no valid frames captured!")

    # 按投球类型 (UP/DOWN/ALL) 统计成功率、被击中率与摔倒率 (Per-type tally)
    up_res = [x for x in results if x["up"]]
    dn_res = [x for x in results if not x["up"]]
    def _pct(lst, key): return 100.0 * sum(x[key] for x in lst) / max(len(lst), 1)
    print(f"\n===== {key} render tally ({n_throws} throws) =====")
    print(f"  {'':10s} {'UP(duck)':>10s} {'DOWN(low)':>10s} {'ALL':>8s}")
    all_res = results
    for label, lst in [("success", None), ("hit", None), ("fell", None)]:
        lu = _pct(up_res, label); ld = _pct(dn_res, label); la = _pct(all_res, label)
        print(f"  {label:10s} {lu:9.0f}% {ld:9.0f}% {la:7.0f}%")

    env.close() if hasattr(env, "close") else None
    del env, envw, runner, policy
    torch.cuda.empty_cache()
    return dict(key=key, frames=len(valid_frames), out_path=out_path,
                results=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--settle", type=int, default=30)
    ap.add_argument("--settle-seconds", type=float, default=0.1,
                    help="settle this many SECONDS (robot stands + stabilizes) before the first "
                         "throw; converted to steps via env.step_dt. Overrides --settle. NOTE: kept "
                         "SHORT (0.1 s) on purpose -- letting the dodge policy stand idle longer drifts "
                         "the robot off its clean ready stand into a worse dodging posture "
                         "(reset: 0.1s=94 -> 1s=87 -> 3s=79 for vision_baseline), which is a settle "
                         "artifact, not real perception difficulty. Short settle = the deployment-"
                         "realistic 'ready robot' condition and agrees with the walk-recover regime.")
    # 评估窗口设为 120 步 (小球到达约第 30 步后持续 ~1.8 秒)：零动作站立约需 1.3 秒失稳倒地，
    # 因此 120 步窗口足够充分捕获由躲球机动导致的失稳迟发摔倒（原 70 步窗口过短无法完全捕获）。
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--up-z", type=lambda s: tuple(float(x) for x in s.split(",")), default=H_TGT,
                    help="UP-throw arrival height range, e.g. '0.9,1.3' (default; matches training "
                         "-- the pre-2026-06-09 (1.0,1.5) range sailed 1.4+ balls over the head; "
                         "see H_TGT comment)")
    ap.add_argument("--frame-offsets", default="0,3,8,18")
    ap.add_argument("--omni", action="store_true",
                    default=os.environ.get("OMNI_THROW", "0") == "1",
                    help="omnidirectional fast-throw eval (360deg, 8-10m, 12-15 m/s); matches released MimicKit config + OMNI_THROW=1 training")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--only", default=None, help="comma list of keys to run (default all)")
    ap.add_argument("--depth-aug", type=lambda s: s.lower() != "false", default=True,
                    help="apply the training sim2real depth aug to depth tasks (default True; "
                         "clean depth is OOD for the depth policies). --depth-aug False for clean.")
    ap.add_argument("--continuous", action="store_true",
                    help="continuous mode: back-to-back throws (alternating UP/DOWN per round) with "
                         "NO stand-reset between -- the robot flows (only termination auto-resets). "
                         "Harder, realistic regime vs the default one-throw-from-stand.")
    ap.add_argument("--rounds", type=int, default=12, help="continuous/walk-recover: throws per env")
    ap.add_argument("--throw-every", type=int, default=45, help="continuous mode: steps between throws")
    ap.add_argument("--walk-recover", action="store_true",
                    help="deploy-faithful mode: walk (loco) policy recovers between throws, then "
                         "dodge policy handles each throw.  Mirrors real deployment (walk<->dodge "
                         "switch). Mutually exclusive with --continuous.")
    ap.add_argument("--recover-seconds", type=float, default=5.0,
                    help="walk-recover: MAX seconds of walk-policy recovery before each throw "
                         "(default 5.0, ~250 steps at 50 Hz). Recovery early-exits as soon as the "
                         "robot is within tolerance of home in both position and heading, so this "
                         "is a ceiling, not a fixed dwell.")
    ap.add_argument("--prethrow-settle", type=float, default=1.0,
                    help="walk-recover: seconds to hold a ZERO velocity command (walk policy) "
                         "AFTER the recovery gate settles, so residual motion damps out and the "
                         "robot is fully standing still before the next throw (default 1.0).")
    ap.add_argument("--walk-ckpt", default=None,
                    help="walk-recover: path to the Unitree-G1-AMP-Flat checkpoint to use as the "
                         "walk actor (default: newest checkpoint of the 'walk' experiment, "
                         "trained via ./train_walk.sh).")
    ap.add_argument("--ckpt", default=None,
                    help="explicit dodge checkpoint path to load (overrides newest_ckpt(exp)). "
                         "Only valid with a single --only key.")
    ap.add_argument("--render-video", default=None,
                    help="[render path] write a walk-recover dodge video to this mp4 path "
                         "(requires --only with exactly one key). Builds num_envs=1, "
                         "render_mode='rgb_array', third-person robot-tracking camera.")
    ap.add_argument("--render-throws", type=int, default=100,
                    help="[render path] number of throws to render (default 100).")
    ap.add_argument("--render-randomized", action="store_true",
                    help="[render path] shuffle UP/DOWN sequence with seeded RNG "
                         "(50 UP + 50 DOWN, shuffled) instead of alternating UP/DOWN.")
    args = ap.parse_args()
    if args.walk_recover and args.continuous:
        raise SystemExit("--walk-recover and --continuous are mutually exclusive.")
    import mjlab.tasks  # noqa: F401 (导入以触发基础任务模块注册)
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---- 渲染视频分支路径: 单策略、单环境并发 (num_envs=1)、捕获每步图像帧 ----
    if args.render_video:
        if not args.only or len(args.only.split(",")) != 1:
            raise SystemExit("--render-video requires exactly one --only key.")
        k = args.only.strip()
        if k not in TASKS:
            raise SystemExit(f"--render-video key '{k}' not in TASKS: {list(TASKS)}")
        task, exp = TASKS[k]
        ckpt = args.ckpt or newest_ckpt(exp)
        if not ckpt:
            raise SystemExit(f"[render] no checkpoint found under {exp}")
        print(f"[render] policy={k}  task={task}  ckpt={ckpt}")
        print(f"[render] out={args.render_video}  throws={args.render_throws}  "
              f"randomized={args.render_randomized}")
        render_task(k, task, exp, ckpt, args, dev)
        return

    # 默认策略键值集合选择逻辑：全向训练策略仅能全向抛球评测，正面训练策略仅能正面抛球评测 —— 默认列表遵循 --omni 参数。
    # [英文对照说明: Default key set: omni policies only against omni throws, front policies only against front throws]
    if args.only:
        keys = args.only.split(",")
    elif args.omni:
        keys = [k for k in TASKS if k.endswith("_omni")]
    else:
        keys = ["statue"] + [k for k in TASKS if not k.endswith("_omni")]
    rows = []
    for k in keys:
        if k == "statue":   # 运行静止雕像基准：测量天然未命中底线 (无需加载模型检查点) (frozen control: free-miss floor)
            rows.append(run_statue(args, dev))
            continue
        task, exp = TASKS[k]
        ckpt = args.ckpt if (args.ckpt and len(keys) == 1) else newest_ckpt(exp)
        if not ckpt:
            print(f"[bench] {k}: no checkpoint under {exp}, skipping"); continue
        rows.append(run_task(k, task, exp, ckpt, args, dev, walk_recover=args.walk_recover))
    print("\n================= SUMMARY (success = not hit & not fell) =================")
    print(f"{'policy':14s} {'success':>8s} {'UP':>6s} {'DOWN':>6s} {'hit':>6s} {'fell':>6s}   hit-links (of hits)")
    for r in rows:
        regs = r.get("hit_regions", {})
        rtot = sum(regs.values()) or 1
        rs = " ".join(f"{k}={100*v/rtot:.0f}%" for k, v in sorted(regs.items(), key=lambda x: -x[1]))
        print(f"{r['key']:14s} {r['success']*100:7.0f}% {r['success_up']*100:5.0f}% "
              f"{r['success_down']*100:5.0f}% {r['hit']*100:5.0f}% {r['fell']*100:5.0f}%   {rs}")


if __name__ == "__main__":
    main()
