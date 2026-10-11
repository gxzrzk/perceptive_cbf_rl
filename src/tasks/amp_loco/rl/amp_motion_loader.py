"""AMP 专家动捕数据加载与特征预处理器（AMP Expert Motion Loader for Discriminator）。

功能与架构定位：
本模块负责从单个 `.npz` 文件或包含多级子目录的动捕数据集目录中，解析并预计算供判别器训练使用的
专家运动状态转移对 `(s_t, s_{t+1})`。

核心特征处理流程：
为确保对抗学习中判别器比较的一致性与公平性，此处计算的专家特征与仿真环境对外输出的
`amp` 观测组在数学定义上完全一致：
1. 相对锚点位置（Anchor-relative Position）：各关键连杆在锚点刚体（通常为躯干 torso）局部坐标系下的 3D 位置；
2. 6D 连续旋转表示（6D Continuous Orientation）：根据 Zhou 等人的《On the Continuity of Rotation Representations in Neural Networks》，
   将相对旋转矩阵的前两列取出构成 6 维向量，彻底消除四元数符号双重覆盖（q 与 -q 表示同一姿态）和欧拉角奇异点问题；
3. 机体局部线速度与角速度（Body-frame Velocities）：利用世界坐标四元数的共轭/逆变换，将连杆线速度和角速度旋转至连杆自身局部坐标系下。
4. 全 GPU 预打包缓存（Packed Sampling Cache）：在初始化阶段完成全部几何变换并拼接为显存中的紧凑二维张量，
   在训练迭代中仅需通过索引切片即可瞬间生成数万个专家转移对，杜绝 CPU-GPU 数据瓶颈。
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

import mjlab.utils.lab_api.math as math_utils


class AMPLoader:
    """判别器专家运动数据管理与相邻帧采样器。"""

    def __init__(
        self,
        motion_file: str,
        body_names: Sequence[str],
        anchor_name: str,
        all_body_names: Sequence[str],
        device: str = "cuda:0",
    ) -> None:
        """初始化 AMP 专家动捕加载器。

        Args:
            motion_file: 单个 .npz 文件路径或包含动捕数据的目录路径（支持递归子目录）。
            body_names: 参与 AMP 观测判别的目标关键刚体名称列表（如双腿骨骼、双臂骨骼、骨盆等）。
            anchor_name: 用于构建局部参考坐标系的锚点刚体名称（通常为 "torso_link"）。
            all_body_names: 机器人完整模型中所有刚体名称的有序列表，用于在动捕文件中对齐刚体索引。
            device: 预处理张量存放的目标 PyTorch 计算设备。
        """
        assert os.path.exists(motion_file), f"动捕数据路径不存在: {motion_file}"

        all_names = list(all_body_names)
        # 获取跟踪刚体和锚点刚体在全局模型中的索引位置
        self._body_indexes = [all_names.index(n) for n in body_names]
        self._anchor_indexes = all_names.index(anchor_name)
        self._num_bodies = len(self._body_indexes)

        # 扫描并解析指定路径下的所有 .npz 动捕文件
        if os.path.isfile(motion_file):
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file):
                for filename in sorted(files):
                    if filename.endswith(".npz"):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            if motion_files:
                motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names)))
                motion_files = list(motion_files)
                motion_names = list(motion_names)
            assert motion_files, f"在目录 {motion_file} 中未检索到任何 .npz 文件！"
        else:
            raise ValueError(f"指定路径既非文件亦非目录: {motion_file}")

        self.motion_names = motion_names
        # 各个片段转换后的局部特征缓存列表
        self._body_pos_b_list: list[torch.Tensor] = []
        self._body_quat_b_list: list[torch.Tensor] = []
        self._body_ori_b_list: list[torch.Tensor] = []
        self._body_lin_vel_b_list: list[torch.Tensor] = []
        self._body_ang_vel_b_list: list[torch.Tensor] = []

        # 遍历所有动作片段并执行刚体坐标系几何变换
        for motion_idx, (motion_name, motion_path) in enumerate(
            zip(motion_names, motion_files)
        ):
            print(f"正在预处理 AMP 动作片段 [{motion_idx + 1}/{len(motion_files)}]: {motion_name}")
            data = np.load(motion_path)
            if motion_idx == 0:
                self.fps = float(np.asarray(data["fps"]).item())

            # 加载世界坐标系下的原始位姿与速度并移入 GPU
            body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            body_quat_w = torch.tensor(
                data["body_quat_w"], dtype=torch.float32, device=device
            )
            body_lin_vel_w = torch.tensor(
                data["body_lin_vel_w"], dtype=torch.float32, device=device
            )
            body_ang_vel_w = torch.tensor(
                data["body_ang_vel_w"], dtype=torch.float32, device=device
            )

            n_steps = body_pos_w.shape[0]
            if n_steps < 2:
                raise ValueError(f"动作片段 {motion_path} 帧数不足 2 帧，无法构建连续状态转移对！")

            # 初始化当前动作片段各帧的局部特征张量
            pos_b = torch.zeros(
                (n_steps, self._num_bodies, 3), dtype=torch.float32, device=device
            )
            quat_b = torch.zeros(
                (n_steps, self._num_bodies, 4), dtype=torch.float32, device=device
            )
            ori_b = torch.zeros(
                (n_steps, self._num_bodies, 6), dtype=torch.float32, device=device
            )
            lin_b = torch.zeros(
                (n_steps, self._num_bodies, 3), dtype=torch.float32, device=device
            )
            ang_b = torch.zeros(
                (n_steps, self._num_bodies, 3), dtype=torch.float32, device=device
            )

            # 逐帧进行刚体几何变换
            for frame_idx in tqdm(
                range(n_steps), desc=f"预计算并缓存: {motion_name}"
            ):
                # 提取当前帧锚点的位置与四元数朝向
                anchor_pos_w = (
                    body_pos_w[frame_idx, self._anchor_indexes]
                    .unsqueeze(0)
                    .repeat(self._num_bodies, 1)
                )
                anchor_quat_w = (
                    body_quat_w[frame_idx, self._anchor_indexes]
                    .unsqueeze(0)
                    .repeat(self._num_bodies, 1)
                )
                bp_w = body_pos_w[frame_idx, self._body_indexes]
                bq_w = body_quat_w[frame_idx, self._body_indexes]
                blv_w = body_lin_vel_w[frame_idx, self._body_indexes]
                bav_w = body_ang_vel_w[frame_idx, self._body_indexes]

                # 将跟踪刚体的全局位置与姿态转换至锚点局部坐标系下
                bp_b, bq_b = math_utils.subtract_frame_transforms(
                    anchor_pos_w, anchor_quat_w, bp_w, bq_w
                )
                # 计算相对旋转矩阵
                mat = math_utils.matrix_from_quat(bq_b)
                # 提取旋转矩阵前两列构建 6D 连续姿态表示
                ori_6 = mat[..., :, :2].reshape(self._num_bodies, 6)
                # 将线速度与角速度旋转至刚体自身局部坐标系
                lin_b_frame = math_utils.quat_apply_inverse(bq_w, blv_w)
                ang_b_frame = math_utils.quat_apply_inverse(bq_w, bav_w)

                pos_b[frame_idx] = bp_b
                quat_b[frame_idx] = bq_b
                ori_b[frame_idx] = ori_6
                lin_b[frame_idx] = lin_b_frame
                ang_b[frame_idx] = ang_b_frame

            self._body_pos_b_list.append(pos_b)
            self._body_quat_b_list.append(quat_b)
            self._body_ori_b_list.append(ori_b)
            self._body_lin_vel_b_list.append(lin_b)
            self._body_ang_vel_b_list.append(ang_b)

        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        # 打包紧凑缓存张量以实现极速 GPU 批量采样
        self._build_sampling_cache()

    @property
    def observation_dim(self) -> int:
        """返回单帧 AMP 观测特征的总维度大小。

        计算方式：每个跟踪刚体包含 3 (相对位置) + 6 (6D姿态) + 3 (局部线速度) + 3 (局部角速度) = 15 维，
        总维度为 15 * num_bodies。
        """
        return (3 + 6 + 3 + 3) * self._num_bodies

    def _build_sampling_cache(self) -> None:
        """将所有动作片段一次性打包压平为一个连续的 GPU 大张量，使后续采样仅需显存索引聚集。"""
        lengths = [pos.shape[0] for pos in self._body_pos_b_list]
        if not lengths or min(lengths) < 2:
            raise ValueError("AMP 数据集要求必须包含至少一个长度 >= 2 帧的非空动作片段！")
        device = self._body_pos_b_list[0].device
        self._motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self._motion_offsets = self._motion_lengths.cumsum(0) - self._motion_lengths
        # 将每个片段的 (pos, ori, lin, ang) 拼接并在时间维拼合为单个显存连续张量
        self._motion_frames = torch.cat([
            torch.cat([
                pos.flatten(1), ori.flatten(1), lin.flatten(1), ang.flatten(1),
            ], dim=-1)
            for pos, ori, lin, ang in zip(
                self._body_pos_b_list, self._body_ori_b_list,
                self._body_lin_vel_b_list, self._body_ang_vel_b_list, strict=True,
            )
        ], dim=0)

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        """均匀混合并采样专家相邻状态对 `(s_t, s_{t+1})` 的批次生成器。

        采样特性：
        1. 动作平衡采样：小批次内的采样槽位在所有动捕片段之间均匀循环分配，确保不同动作均衡参与判别；
        2. 严格时序有效性：在选中的片段内部，仅在有效区间 [0, T-2] 随机选择起始帧，并取其相邻的下一帧组成状态对，
           绝不跨越片段边界，也不引入自环伪数据；
        3. 纯 GPU 向量化索引，具备极高的采样速度。

        Args:
            num_mini_batch: 生成的 mini-batch 批次数。
            mini_batch_size: 每个 mini-batch 包含的状态转移对数量。

        Yields:
            tuple[torch.Tensor, torch.Tensor]: 采样的专家状态对 (expert_state, expert_next_state)，
            形状均为 (mini_batch_size, observation_dim)。
        """
        if mini_batch_size < 1 or num_mini_batch < 0:
            raise ValueError("mini_batch_size 必须为正整数，且 num_mini_batch 必须为非负整数！")
        num_motions = self._motion_lengths.numel()
        device = self._motion_frames.device
        slots = torch.arange(mini_batch_size, device=device) % num_motions
        for _ in range(num_mini_batch):
            # 当批次大小不能被片段数量整除时，随机打乱多余槽位的归属
            motion_order = torch.randperm(num_motions, device=device)
            motion_ids = motion_order[slots]
            motion_ids = motion_ids[torch.randperm(mini_batch_size, device=device)]
            pair_counts = self._motion_lengths[motion_ids] - 1
            # 在片段有效长度范围内均匀抽取帧索引
            local_frame = (torch.rand(mini_batch_size, device=device) * pair_counts).long()
            frame_ids = self._motion_offsets[motion_ids] + local_frame
            # 返回当前帧与其紧接着的下一帧作为专家转移对
            yield self._motion_frames[frame_ids], self._motion_frames[frame_ids + 1]
