"""环境级参考运动数据加载器（Environment Reference Motion Loader）。

本模块定义了 `MotionLoader` 类，负责从指定目录（包括子目录）递归加载 `.npz` 格式的专家动捕运动数据，
在仿真环境启动及重置时提供参考姿态。

主要用途：
1. 参考状态初始化（Reference State Initialization, RSI）：在回合开始时随机从动捕数据中抽取真实姿态
   和速度来初始化机器人刚体与关节状态，显著提高强化学习探索效率并避免陷入局部极小；
2. 跌倒恢复与运动追踪：提供根节点、关键刚体、锚点刚体及双足在世界坐标系下的位置、四元数姿态与线/角速度。
"""

from __future__ import annotations
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING


class MotionLoader:
    """环境级动捕参考运动数据加载与采样器。

    支持加载正常运动片段与恢复运动片段，按索引或随机抽取帧，输出用于环境初始化的物理状态。
    """

    def __init__(
        self,
        motion_dir: str,
        tgt_body_indexes: Sequence[int],
        tgt_anchor_indexes: int,
        feet_indexes: int,
        device: str = "cpu",
        recovery_dir: str | None = None,
    ):
        """初始化运动数据加载器。

        Args:
            motion_dir: 正常运动 .npz 文件所在目录（支持递归子目录）。
            tgt_body_indexes: 跟踪的关键身体部件在模型中的索引序列。
            tgt_anchor_indexes: 锚点身体部件（如躯干 torso）在模型中的索引。
            feet_indexes: 双足接触刚体在模型中的索引。
            device: 张量存放的目标计算设备（如 "cpu" 或 "cuda:0"）。
            recovery_dir: 可选的跌倒恢复运动 .npz 文件所在目录。
        """
        # 存储所有常规运动数据的字典列表
        self.motion_data: list[dict] = []
        # 存储所有跌倒恢复运动数据的字典列表
        self.motion_data_recovery: list[dict] = []

        # 递归扫描并加载正常运动数据
        self.motion_data = self._load_dir(motion_dir, device)
        assert len(self.motion_data) > 0, f"在目录 {motion_dir} 中未找到任何 .npz 文件！"

        # 若提供了恢复运动目录，则递归扫描并加载恢复数据
        if recovery_dir is not None and os.path.isdir(recovery_dir):
            self.motion_data_recovery = self._load_dir(recovery_dir, device)

        # 记录所有加载的运动片段名称
        self.motion_names = [m["motion_name"] for m in self.motion_data + self.motion_data_recovery]

        if not self.motion_data and not self.motion_data_recovery:
            raise ValueError(f"未能从目录加载到任何运动数据: {motion_dir}")

        # 取第一个运动片段作为默认参考
        default_motion = self.motion_data[0] if self.motion_data else self.motion_data_recovery[0]

        # 运动数据的帧率（FPS）及各项物理量缓存
        self.fps = default_motion["fps"]
        self._dof_pos = default_motion["dof_pos"]
        self._dof_vel = default_motion["dof_vel"]
        self._body_pos_w = default_motion["body_pos_w"]
        self._body_quat_w = default_motion["body_quat_w"]
        self._body_lin_vel_w = default_motion["body_lin_vel_w"]
        self._body_ang_vel_w = default_motion["body_ang_vel_w"]

        self._body_indexes = tgt_body_indexes
        self._anchor_indexes = tgt_anchor_indexes
        self._feet_indexes = feet_indexes
        # 默认片段的总帧数与总持续时间（秒）
        self.time_step_total = self._dof_pos.shape[0]
        self.motion_total_time = self.time_step_total / self.fps

    @staticmethod
    def _load_dir(dir_path: str, device: str) -> list[dict]:
        """从目录中递归加载所有 .npz 动捕数据文件并解析为张量字典列表。

        采用 os.walk 递归遍历，确保若目录包含子目录（如 amp_dodge 下的 Leap/ 与 Dodge/），
        能够完整加载所有动作片段，与判别器的加载行为保持一致。

        Args:
            dir_path: 动捕文件所在的目录路径。
            device: 目标设备。

        Returns:
            包含各个动作片段物理特征字典的列表。
        """
        assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"
        npz_paths = []
        for root, _dirs, files in os.walk(dir_path):
            for filename in files:
                if filename.endswith(".npz"):
                    npz_paths.append(os.path.join(root, filename))
        result = []
        for path in sorted(npz_paths):
            motion_name = os.path.splitext(os.path.basename(path))[0]
            data = np.load(path)
            result.append({
                "motion_name": motion_name,
                "fps": data["fps"],
                "dof_pos": torch.tensor(data["joint_pos"], dtype=torch.float32, device=device),
                "dof_vel": torch.tensor(data["joint_vel"], dtype=torch.float32, device=device),
                "body_pos_w": torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device),
                "body_quat_w": torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device),
                "body_lin_vel_w": torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device),
                "body_ang_vel_w": torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device),
            })
        return result

    def _get_motion_data(self, motion_index: int = None) -> dict:
        """获取指定动作片段的数据字典；若 motion_index 为 None，则返回默认初始动作数据。

        Args:
            motion_index: 动作片段索引，范围为 [0, len(motion_data))。

        Returns:
            包含各刚体与关节物理信息的字典。
        """
        if motion_index is None:
            return {
                "body_pos_w": self._body_pos_w,
                "body_quat_w": self._body_quat_w,
                "body_lin_vel_w": self._body_lin_vel_w,
                "body_ang_vel_w": self._body_ang_vel_w,
                "dof_pos": self._dof_pos,
                "dof_vel": self._dof_vel,
            }
        else:
            assert 0 <= motion_index < len(self.motion_data), f"动作索引 {motion_index} 超出有效范围 [0, {len(self.motion_data)})"
            return self.motion_data[motion_index]

    def tgt_body_pos_w(self, motion_index: int = None) -> torch.Tensor:
        """获取目标跟踪刚体在世界坐标系下的位置序列 [T, num_bodies, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._body_indexes, :]

    def tgt_body_quat_w(self, motion_index: int = None) -> torch.Tensor:
        """获取目标跟踪刚体在世界坐标系下的四元数姿态序列 [T, num_bodies, 4]。"""
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, self._body_indexes, :]

    def tgt_body_lin_vel_w(self, motion_index: int = None) -> torch.Tensor:
        """获取目标跟踪刚体在世界坐标系下的线速度序列 [T, num_bodies, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, self._body_indexes, :]

    def tgt_body_ang_vel_w(self, motion_index: int = None) -> torch.Tensor:
        """获取目标跟踪刚体在世界坐标系下的角速度序列 [T, num_bodies, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, self._body_indexes, :]

    def tgt_anchor_pos_w(self, motion_index: int = None) -> torch.Tensor:
        """获取参考锚点刚体（通常为躯干 torso）在世界坐标系下的位置序列 [T, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._anchor_indexes]

    def tgt_anchor_quat_w(self, motion_index: int = None) -> torch.Tensor:
        """获取参考锚点刚体在世界坐标系下的四元数姿态序列 [T, 4]。"""
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, self._anchor_indexes]

    def tgt_anchor_lin_vel_w(self, motion_index: int = None) -> torch.Tensor:
        """获取参考锚点刚体在世界坐标系下的线速度序列 [T, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, self._anchor_indexes]

    def tgt_anchor_ang_vel_w(self, motion_index: int = None) -> torch.Tensor:
        """获取参考锚点刚体在世界坐标系下的角速度序列 [T, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, self._anchor_indexes]

    def tgt_dof_pos(self, motion_index: int = None) -> torch.Tensor:
        """获取各自由度关节角度序列 [T, num_dof]。"""
        data = self._get_motion_data(motion_index)
        return data["dof_pos"]

    def tgt_dof_vel(self, motion_index: int = None) -> torch.Tensor:
        """获取各自由度关节角速度序列 [T, num_dof]。"""
        data = self._get_motion_data(motion_index)
        return data["dof_vel"]

    def tgt_feet_pos_w(self, motion_index: int = None) -> torch.Tensor:
        """获取双足在世界坐标系下的位置序列 [T, num_feet, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, self._feet_indexes]

    def tgt_root_pos(self, motion_index: int = None) -> torch.Tensor:
        """获取根节点（基座 pelvis）在世界坐标系下的位置序列 [T, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_pos_w"][:, 0, :]

    def tgt_root_quat(self, motion_index: int = None) -> torch.Tensor:
        """获取根节点在世界坐标系下的四元数姿态序列 [T, 4]。"""
        data = self._get_motion_data(motion_index)
        return data["body_quat_w"][:, 0, :]

    def tgt_root_lin_vel(self, motion_index: int = None) -> torch.Tensor:
        """获取根节点在世界坐标系下的线速度序列 [T, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_lin_vel_w"][:, 0, :]

    def tgt_root_ang_vel(self, motion_index: int = None) -> torch.Tensor:
        """获取根节点在世界坐标系下的角速度序列 [T, 3]。"""
        data = self._get_motion_data(motion_index)
        return data["body_ang_vel_w"][:, 0, :]

    def sample_random_frames(self, num_samples: int) -> dict[str, torch.Tensor]:
        """从所有已加载的动捕动作（包含正常运动与跌倒恢复）中随机抽取帧，用于环境状态重置（RSI）。

        Args:
            num_samples: 待采样的样本数量（通常对应当前需要 reset 的环境数量 num_resets）。

        Returns:
            dict with keys:
                root_pos: (num_samples, 3) - 根节点世界位置
                root_quat: (num_samples, 4) - 根节点朝向四元数
                root_lin_vel: (num_samples, 3) - 根节点世界线速度
                root_ang_vel: (num_samples, 3) - 根节点世界角速度
                joint_pos: (num_samples, num_joints) - 各关节位置
                joint_vel: (num_samples, num_joints) - 各关节速度
        """
        all_motions = self.motion_data + self.motion_data_recovery
        # 随机挑选动作片段索引
        motion_indices = torch.randint(0, len(all_motions), (num_samples,))
        # 对每个选中的动作片段随机采样一帧
        result_root_pos = []
        result_root_quat = []
        result_root_lin_vel = []
        result_root_ang_vel = []
        result_joint_pos = []
        result_joint_vel = []

        for i in range(num_samples):
            motion = all_motions[motion_indices[i].item()]
            num_frames = motion["dof_pos"].shape[0]
            frame_idx = torch.randint(0, num_frames, (1,)).item()

            result_root_pos.append(motion["body_pos_w"][frame_idx, 0, :])
            result_root_quat.append(motion["body_quat_w"][frame_idx, 0, :])
            result_root_lin_vel.append(motion["body_lin_vel_w"][frame_idx, 0, :])
            result_root_ang_vel.append(motion["body_ang_vel_w"][frame_idx, 0, :])
            result_joint_pos.append(motion["dof_pos"][frame_idx])
            result_joint_vel.append(motion["dof_vel"][frame_idx])

        return {
            "root_pos": torch.stack(result_root_pos),
            "root_quat": torch.stack(result_root_quat),
            "root_lin_vel": torch.stack(result_root_lin_vel),
            "root_ang_vel": torch.stack(result_root_ang_vel),
            "joint_pos": torch.stack(result_joint_pos),
            "joint_vel": torch.stack(result_joint_vel),
        }
