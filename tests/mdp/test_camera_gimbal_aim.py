"""云台相机追踪方向与最优俯仰角计算测试 (Camera Gimbal Aim Tests)。

测试功能：
- 验证从事件中抽离的核心几何计算函数 ``optimal_camera_pitch`` 的正确性；
- 验证视线威胁判断门控逻辑 (threat flag)：根据球体是否在空中以及是否朝向机器人躯干逼近动态判定；
- 验证在躯干局部坐标系 (torso frame) 下计算相机最佳俯仰角度 (target pitch)，并限制在安全机械限位 ±30° (±0.5236 rad) 范围内；
- 验证重构后的仿真事件 ``aim_camera_at_ball`` 与几何辅助函数的计算结果完全一致：在有威胁时自动将云台关节目标位置设置为计算出的 target_pitch，无威胁时回正为 0。

关键断言条件：
- Case A (空中来球逼近): threat 掩码全为 True，target_pitch 无 NaN 且被严格截断在 ±30° 范围内，且数值与躯干系仰角数学推导值 (-elev) 绝对误差小于 1e-5；
- Case B (地面静止小球): threat 掩码全为 False，避免地面非威胁球误触发云台抬起/低头；
- 关节目标输出一致性: robot.data.joint_pos_target 中云台俯仰关节的指令值严格等于 where(threat, pitch, 0)。

覆盖边界情况：
- 空中高速向机器人飞来的来袭球（逼近速度 vx < 0）；
- 落在地面且速度归零的静态小球（高度为球体半径，vx = 0）；
- 目标俯仰角超出 ±30° 限位时的截断保护；
- 并行多环境 (num_envs=4) 下的 GPU 张量批处理计算。
"""
from __future__ import annotations

import math
import os

import pytest
import torch


# ---------------------------------------------------------------------------
# 测试共享的辅助构建与场景放置函数
# ---------------------------------------------------------------------------

def _build_gimbal_env():
    """设置环境变量 CAMERA_GIMBAL=1 并直接调用配置构建器以初始化带有云台关节的仿真环境。

    说明：
        绕过全局环境注册表 load_env_cfg（注册表在初次导入时会固化配置缓存），
        直接调用 g1_amp_dodge_depth_single_ballonly_flat_env_cfg()。该函数在调用时
        即时读取环境变量，确保不受测试执行顺序影响，始终构造包含完整云台关节铰链的环境。
    """
    os.environ.update({
        "CAMERA_GIMBAL": "1",
        "CAMERA_TILT_DEG": "20",
        "BALLONLY_AUG": "0",
        # 清除可能引入其他无关特性的环境变量
        "CBF_JOINT": "0",
        "DAGGER_TEACHER_OBS": "0",
    })
    # 导入 src.tasks 以触发任务模块注册（副作用导入）
    import src.tasks  # noqa: F401 -- 触发任务注册

    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_mimickit_classroom_flat_env_cfg,  # noqa: F401 -- 导入教室躲避任务配置备用
        g1_amp_dodge_depth_single_ballonly_flat_env_cfg,
    )
    from mjlab.envs import ManagerBasedRlEnv

    cfg = g1_amp_dodge_depth_single_ballonly_flat_env_cfg(play=False)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    env.reset()
    return env


def _place_ball_airborne(env, robot, ball, x_offset=2.0, z_world=1.8, vx=-3.0):
    """将球放置在空中并赋予逼近机器人的线速度。

    测试数据构造：
        位置：(robot_x + x_offset, robot_y, z_world)，位于机器人前方空中；
        线速度：(vx, 0, 0)，世界坐标系下沿 X 轴负方向高速逼近机器人。
    通过 mjlab API 直接写入仿真器并单步推进 step() 以刷新传感器缓存。
    """
    N = env.num_envs
    dev = env.device

    robot_base_pos = robot.data.root_link_pos_w.clone()  # 机器人基座世界坐标，形状为 (N, 3) (robot root position in world frame)
    quat_identity = torch.zeros(N, 4, device=dev)
    quat_identity[:, 0] = 1.0  # w=1, x=y=z=0 单位四元数

    ball_pos = robot_base_pos.clone()
    ball_pos[:, 0] += x_offset
    ball_pos[:, 2] = z_world

    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, quat_identity], dim=-1)
    )
    vel = torch.zeros(N, 6, device=dev)
    vel[:, 0] = vx  # 世界系 X 轴负向速度（接近机器人）
    ball.write_root_link_velocity_to_sim(vel)

    env.sim.step()  # 推进仿真以刷新数据缓存


def _place_ball_grounded(env, ball, ball_geom_id, x_offset=1.0):
    """将球放置在地面静止状态（z = ball_radius，速度为 0）。

    测试数据构造：
        位置：机器人前方 1.0m，高度等于球体几何半径（恰好接触地面）；
        速度：全零。
    """
    N = env.num_envs
    dev = env.device

    robot = env.scene["robot"]
    robot_base_pos = robot.data.root_link_pos_w.clone()
    ball_r = env.sim.model.geom_size[:, ball_geom_id, 0]  # 球体几何半径标量，形状为 (N,) (ball geometry radius scalar)

    quat_identity = torch.zeros(N, 4, device=dev)
    quat_identity[:, 0] = 1.0

    ball_pos = robot_base_pos.clone()
    ball_pos[:, 0] += x_offset
    ball_pos[:, 2] = ball_r  # 球心高度等于半径，即静止放置在地面上

    ball.write_root_link_pose_to_sim(
        torch.cat([ball_pos, quat_identity], dim=-1)
    )
    ball.write_root_link_velocity_to_sim(torch.zeros(N, 6, device=dev))

    env.sim.step()  # 刷新仿真缓存


# ---------------------------------------------------------------------------
# 测试 1: optimal_camera_pitch 计算目标俯仰角与来袭威胁标识
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_optimal_camera_pitch_matches_legacy():
    """测试场景：验证 optimal_camera_pitch 函数在来球威胁及安全两种场景下的计算准确性。

    输入假数据与测试场景：
      Case A – 空中逼近来球：球位于机器人前方 2m、高度 1.8m、以 -3.0m/s 速度逼近机器人；
      Case B – 地面静止球：球位于机器人前方 1m、处于地面 (z = ball_radius)、速度为 0。

    预期输出及验证逻辑：
      Case A：
        - threat 威胁标识必须对所有并行环境均为 True；
        - target_pitch 必须无 NaN，且在 [-30°, +30°] 物理极限内；
        - 数值精度核对：通过 quat_apply_inverse 将相对位移变换至躯干坐标系，
          计算几何仰角 elev = atan2(dz, d_horiz)，预期 pitch 为 clamp(-elev, -30°, +30°)，
          断言两者在 1e-5 容差内完全吻合。
      Case B：
        - threat 威胁标识必须对所有并行环境均为 False，避免无威胁时误转云台。
    """
    from src.tasks.amp_loco.mdp.events import optimal_camera_pitch

    # 直接构建包含云台关节的仿真环境
    env = _build_gimbal_env()
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    # 预热并读取环境中的云台关节与几何体 ID 缓存
    optimal_camera_pitch(env, robot, ball)
    _, _, ball_geom_id, _ = env._gimbal_ids

    # ------------------------------------------------------------------
    # Case A: 球在机器人前方空中，且以负速度向机器人飞来
    # ------------------------------------------------------------------
    _place_ball_airborne(env, robot, ball, x_offset=2.0, z_world=1.8, vx=-3.0)

    target_pitch, threat = optimal_camera_pitch(env, robot, ball)

    # 验证威胁判断：空中飞来球必须被判定为有威胁
    assert threat.all(), f"Expected all threats (airborne+closing), got threat={threat}"

    # 验证俯仰角计算：不包含 NaN 且在 ±30° 截断范围内
    assert not torch.isnan(target_pitch).any(), "target_pitch contains NaN"
    lim = math.radians(30)
    assert (target_pitch.abs() <= lim + 1e-5).all(), \
        f"target_pitch out of ±30° clamp: {target_pitch}"

    # 独立几何推导交叉验证：将相机到球的向量通过躯干姿态四元数的逆旋转投影到躯干坐标系
    from mjlab.utils.lab_api.math import quat_apply_inverse

    jid, bid, _, tid = env._gimbal_ids
    cam = robot.data.body_link_pos_w[:, bid, :]
    bp = ball.data.root_link_pos_w
    torso_quat = robot.data.body_link_quat_w[:, tid, :]
    d_torso = quat_apply_inverse(torso_quat, bp - cam)
    horiz = torch.linalg.norm(d_torso[:, :2], dim=-1).clamp(min=1e-6)
    elev = torch.atan2(d_torso[:, 2], horiz)
    expected_pitch = (-elev).clamp(-lim, lim)

    assert torch.allclose(target_pitch, expected_pitch, atol=1e-5), \
        f"Mismatch: target_pitch={target_pitch}, expected={expected_pitch}"

    # ------------------------------------------------------------------
    # Case B: 球静止在地面上 -> 判定为无威胁 (threat == False)
    # ------------------------------------------------------------------
    _place_ball_grounded(env, ball, ball_geom_id, x_offset=1.0)

    _, threat2 = optimal_camera_pitch(env, robot, ball)
    assert not threat2.any(), \
        f"Expected no threats for grounded ball (z=ball_r, vel=0), got threat2={threat2}"


# ---------------------------------------------------------------------------
# 测试 2: aim_camera_at_ball 事件与几何辅助函数设置相同的目标位置
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_aim_event_sets_same_target():
    """测试场景：验证 aim_camera_at_ball 事件函数正确将计算出的最优俯仰角写入机器人关节控制目标。

    输入假数据：
        构造前方 1.5m、高度 1.5m、vx = -4.0m/s 的飞来球，形成即时威胁。
    预期输出及验证逻辑：
        1. optimal_camera_pitch 计算出 expected_target = where(threat, target_pitch, 0)；
        2. 调用 aim_camera_at_ball 事件函数（步进模式，作用于所有环境）；
        3. 读取机器人关节位置控制目标 robot.data.joint_pos_target[:, jid]；
        4. 断言实际写入的关节目标 actual_target 与 expected_target 严格相等 (atol=1e-5)。
    """
    from src.tasks.amp_loco.mdp.events import aim_camera_at_ball, optimal_camera_pitch

    # 直接构建包含云台关节的环境
    env = _build_gimbal_env()
    robot = env.scene["robot"]
    ball = env.scene["ball"]

    # 预热云台 ID 缓存
    optimal_camera_pitch(env, robot, ball)

    # 放置空中高速飞向机器人的球
    _place_ball_airborne(env, robot, ball, x_offset=1.5, z_world=1.5, vx=-4.0)

    # 通过几何辅助函数计算期望的关节目标
    target_pitch, threat = optimal_camera_pitch(env, robot, ball)
    expected_target = torch.where(threat, target_pitch, torch.zeros_like(target_pitch))

    # 执行重构后的云台瞄准事件
    aim_camera_at_ball(env, env_ids=None, robot_name="robot", ball_name="ball")

    # 读回关节目标并进行断言比对
    jid = env._gimbal_ids[0]
    actual_target = robot.data.joint_pos_target[:, jid].squeeze(-1)   # 实际下发的关节目标位置张量，形状为 (N,) (actual joint position target)

    assert torch.allclose(actual_target, expected_target, atol=1e-5), (
        f"aim_camera_at_ball joint target mismatch.\n"
        f"  expected = {expected_target}\n"
        f"  actual   = {actual_target}"
    )
