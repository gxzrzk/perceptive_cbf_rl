"""云台相机与关节级控制障碍函数 (CBF) 组合配置测试 (Gimbal + CBF Composition Tests)。

测试功能：
- 验证当云台相机铰链 (CAMERA_GIMBAL=1) 与关节级控制障碍函数 (CBF_JOINT=1) 同时启用时的组合交互行为；
- 验证动作维度隔离：云台关节由专门的高层视线追踪控制器或策略独立控制，CBF 滤波动作项仅作用于机器人的 29 个全身运动关节，因此总动作维度必须严格保持为 29（而非 30）；
- 验证 CBF 障碍物安全距离约束的连杆保护集合中，排除了相机机身连杆 (camera_pitch_link)，防止相机头部的自由俯仰运动与全身避碰约束产生耦合干涉。

关键断言条件：
- env.action_manager.total_action_dim == 29；
- "camera_pitch_link" 不在 CBF 项过滤的刚体名称列表中。

覆盖边界情况：
- 跨模块环境变量在函数调用时的即时解析；
- 机器人关节拓扑扩充与动作空间维度的正确映射。
"""
import os
import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU sim")
def test_gimbal_plus_cbf_keeps_29_action_dim_and_excludes_camera():
    """测试场景：同时启用云台相机与关节级 CBF 时，动作空间维持 29 维且安全连杆排除相机连杆。

    输入假数据与环境配置：
        设置 CAMERA_GIMBAL=1, CAMERA_PROPRIO=1, CBF_JOINT=1 等环境变量，
        构建包含 4 个并行环境的 G1 躲避仿真任务。
    预期输出及验证逻辑：
        1. 读取 action_manager 中的 'joint_pos' 项，确认动作维度 total_action_dim 严格为 29，
           验证云台俯仰自由度未被错误混入策略全身控制动作空间；
        2. 获取 CBF 约束所保护的所有连杆名称，验证 'camera_pitch_link' 未包含其中，
           保证云台相机独立对准小球而不受 CBF 障碍物碰撞距离惩罚限制。
    """
    os.environ.update({
        "CAMERA_GIMBAL": "1", "CAMERA_PROPRIO": "1", "CBF_JOINT": "1",
        "CAMERA_TILT_DEG": "20", "BALLONLY_AUG": "0",
    })
    import src.tasks  # noqa: F401 -- 触发任务注册（通过副作用导入注册所有任务配置环境） (triggers task registration via side-effect import)
    from mjlab.envs import ManagerBasedRlEnv
    # 直接调用构建器函数：云台与 CBF 相关的环境变量在调用函数时（CALL time）动态解析，
    # 而注册表中的 cfg 在首次模块导入时便已固化（测试无法直接控制其环境变量） (Call the builder directly: the gimbal/CBF env vars are read at CALL time, whereas the registered cfg is frozen at first import).
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_depth_single_ballonly_flat_env_cfg,
    )
    cfg = g1_amp_dodge_depth_single_ballonly_flat_env_cfg(play=False)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    term = env.action_manager.get_term("joint_pos")
    # 云台关节独立驱动被排除：CBF 动作项仅作用于 G1 机器人的 29 个全身运动学自由度，而非包含相机的 30 维 (gimbal joint excluded -> 29 body DOF actuated by the CBF term, not 30).
    assert env.action_manager.total_action_dim == 29
    # CBF 屏障保护刚体集合过滤：CBF 碰撞屏障保护的刚体列表中绝不能包含云台相机刚体 camera_pitch_link (CBF filter bodies must not include the gimbal camera body).
    body_names = [env.scene["robot"].body_names[i] for i in term._body_local.tolist()]
    assert "camera_pitch_link" not in body_names
