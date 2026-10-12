"""AMP 运动数据加载器 (AMPLoader) 单元测试。

测试功能：
- 验证混合动作片段 (multi-clip) 状态转移采样在批量生成中的负载均衡与随机覆盖性；
- 验证批次内时序连续帧转换配对（当前帧 state 与下一时序帧 next_state）；
- 验证不同 batch_size 下片段采样的整除性、均匀分布与边界情况；
- 验证从真实 .npz 文件解析、解包运动特征与特征维度构造；
- 验证非法输入（如单帧片段导致无法构建时序对）的异常拦截机制。

关键断言条件：
- 每个 batch 必须均匀覆盖所有 clip（各 clip 采样频次之差不超过 1）；
- next_state 必须与 state 来自同一 clip 且时间步严格满足 t+1；
- 观测维度与各刚体在基座坐标系下的位姿、线速度、角速度等特征正确对应。

覆盖边界情况：
- 极短片段（仅两帧 clip）；
- 极小批次 (batch_size=1) 与非整除批次；
- 非法单帧片段（抛出 ValueError）；
- 随机种子的一致性与多轮采样随机性验证。
"""
import numpy as np
import pytest
import torch

from src.tasks.amp_loco.rl.amp_motion_loader import AMPLoader


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def make_loader(num_clips=54, device='cpu'):
    """构造包含指定数量模拟动作片段的 AMPLoader 实例。

    参数：
        num_clips: 片段总数，默认 54
        device: 数据张量存放的目标设备 ('cpu' 或 'cuda')

    返回：
        已构建并完成采样缓存 (_build_sampling_cache) 的 AMPLoader 对象
    """
    loader = AMPLoader.__new__(AMPLoader)
    loader._num_bodies = 2
    loader._body_pos_b_list = []
    loader._body_ori_b_list = []
    loader._body_lin_vel_b_list = []
    loader._body_ang_vel_b_list = []
    for clip in range(num_clips):
        n = 2 + clip % 7
        pos = torch.zeros(n, 2, 3, device=device)
        pos[:, :, 0] = clip
        pos[:, :, 1] = torch.arange(n, device=device)[:, None]
        pos[:, :, 2] = 11.
        loader._body_pos_b_list.append(pos)
        loader._body_ori_b_list.append(torch.full((n, 2, 6), 22., device=device))
        loader._body_lin_vel_b_list.append(torch.full((n, 2, 3), 33., device=device))
        loader._body_ang_vel_b_list.append(torch.full((n, 2, 3), 44., device=device))
    loader._build_sampling_cache()
    return loader


@pytest.mark.parametrize('device', DEVICES)
def test_every_default_batch_covers_all_clips_on_repeated_updates(device):
    """测试场景：多轮次反复抽取大批量数据时，样本在所有 clip 间的均匀覆盖性与有效时序对。

    输入假数据：
        54 个片段（每个片段长度为 2 到 8 帧），批次大小为 6144，迭代 20 个 batch，运行 2 轮。
    预期输出及验证逻辑：
        1. state 与 next_state 的形状均为 (6144, 30)，张量处于指定 device；
        2. 6144 / 54 = 113 余 42，因此每个 clip 采样次数必然为 113 或 114 次（差值 <= 1）；
        3. 每轮获得额外分配（114次）的 clip 集合随时间动态改变，避免固定偏向某一部分片段；
        4. next_state 与 state 属于同一 clip，且帧索引精确满足 index(next) == index(curr) + 1；
        5. 当前帧 index 严格小于当前 clip 长度 - 1（确保有合法的后继帧）；
        6. 各刚体观测特征（旋转矩阵 22.0、线速度 33.0、角速度 44.0）保持完整。
    """
    torch.manual_seed(24)
    loader = make_loader(device=device)
    extras = []
    previous = None
    for _ in range(2):
        batches = 0
        for state, next_state in loader.feed_forward_generator(20, 6144):
            batches += 1
            assert state.shape == next_state.shape == (6144, 30)
            assert state.device.type == device
            ids = state[:, 0].long()
            counts = ids.bincount(minlength=54)
            assert counts.sum() == 6144
            assert counts.min() == 113 and counts.max() == 114
            extras.append(counts == 114)
            assert not torch.equal(ids, ids.sort().values)
            if previous is not None:
                assert not torch.equal(state, previous)
            previous = state
            # 同一片段的下一自然帧验证（包含仅有两帧的最短动作片段）：确保 clip_id 保持一致，帧索引恰好严格加 1 (Same clip, next chronological frame, including two-frame clips).
            assert torch.equal(next_state[:, 0], state[:, 0])
            assert torch.equal(next_state[:, 1], state[:, 1] + 1)
            assert (state[:, 1] < loader._motion_lengths[ids] - 1).all()
            assert (state[:, 6:18] == 22.).all()
            assert (state[:, 18:24] == 33.).all()
            assert (state[:, 24:] == 44.).all()
        assert batches == 20
    # 验证余数插槽动态轮转：除不尽时的多余采样名额不会永久分配给同一子集片段，批次间具有随机轮换特性 (Extra slots are not permanently assigned to the same clip subset).
    assert not torch.equal(extras[0], extras[1])


@pytest.mark.parametrize('batch_size', [1, 20, 54, 108])
def test_small_and_divisible_batch_sizes(batch_size):
    """测试场景：边界批次大小（包括极小批次 1、非整除批次 20、正好整除批次 54 和 108）。

    输入假数据：
        54 个片段，batch_size 分别取 1, 20, 54, 108。
    预期输出及验证逻辑：
        1. 采样的样本数严格等于要求的 batch_size；
        2. 各片段采样计数最大值与最小值之差不超过 1（实现极致平摊）；
        3. 覆盖的不同 clip 数量等于 min(batch_size, 54)；
        4. 时序后继关系严格满足 next_frame == curr_frame + 1。
    """
    loader = make_loader()
    for state, next_state in loader.feed_forward_generator(3, batch_size):
        counts = state[:, 0].long().bincount(minlength=54)
        assert len(state) == batch_size
        assert counts.max() - counts.min() <= 1
        assert (counts > 0).sum() == min(batch_size, 54)
        assert torch.equal(next_state[:, 1], state[:, 1] + 1)


def test_single_two_frame_clip_and_repeatable_seed():
    """测试场景：单一两帧极短片段采样与固定随机种子的可复现性。

    输入假数据：
        1. 仅包含 1 个 clip 且长度为 2 帧的 loader；
        2. 包含 54 个 clip 的标准 loader，两次设置相同的 torch 随机种子 (seed=42)。
    预期输出及验证逻辑：
        1. 单一两帧 clip 采样时，当前帧必为 0，下一帧必为 1；
        2. 固定随机种子后，两次调用 generator 产生的 (state, next_state) 张量完全一致。
    """
    loader = make_loader(num_clips=1)
    state, next_state = next(loader.feed_forward_generator(1, 16))
    assert (state[:, 1] == 0).all() and (next_state[:, 1] == 1).all()
    loader = make_loader()
    torch.manual_seed(42)
    first = next(loader.feed_forward_generator(1, 100))
    torch.manual_seed(42)
    second = next(loader.feed_forward_generator(1, 100))
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def write_motion(path, n):
    """辅助函数：向指定路径生成包含 n 帧模拟动作数据的 .npz 文件。

    包含 body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w 等关键刚体运动字段。
    """
    pos = np.zeros((n, 2, 3), dtype=np.float32)
    pos[:, 1, 0] = np.arange(n)
    quat = np.zeros((n, 2, 4), dtype=np.float32)
    quat[:, :, 0] = 1.
    np.savez(path, fps=50., body_pos_w=pos, body_quat_w=quat,
             body_lin_vel_w=np.full_like(pos, 3.), body_ang_vel_w=np.full_like(pos, 4.))


def test_constructor_packs_real_npz_features(tmp_path):
    """测试场景：AMPLoader 构造函数读取实际 .npz 文件并正确打包特征维度。

    输入假数据：
        包含 4 帧的临时 motion.npz 文件，定义躯干 (torso) 和足部 (foot) 2 个刚体。
    预期输出及验证逻辑：
        1. 观测维度 observation_dim 为 15（基座相对位姿与刚体速度拼接维度）；
        2. state 的 shape 为 (64, 15)；
        3. 时序后继帧序号满足 next == curr + 1；
        4. 线速度切片 (3.0) 与角速度切片 (44.0 对应的 4.0) 数值正确；
        5. 所有采样特征张量数值有限（无 NaN/Inf）。
    """
    path = tmp_path / 'motion.npz'
    write_motion(path, 4)
    loader = AMPLoader(str(path), ['foot'], 'torso', ['torso', 'foot'], device='cpu')
    state, next_state = next(loader.feed_forward_generator(1, 64))
    assert loader.observation_dim == 15 and state.shape == (64, 15)
    assert torch.equal(next_state[:, 0], state[:, 0] + 1)
    assert (state[:, 9:12] == 3.).all() and (state[:, 12:] == 4.).all()
    assert torch.isfinite(state).all()


def test_single_frame_clip_rejected(tmp_path):
    """测试场景：异常输入校验——当动作用于训练时如果只有 1 帧，无法构建转移对。

    输入假数据：
        仅包含 1 帧动作数据的临时 NPZ 文件。
    预期输出及验证逻辑：
        AMPLoader 初始化时抛出 ValueError，异常信息匹配 'at least two frames'。
    """
    path = tmp_path / 'single.npz'
    write_motion(path, 1)
    with pytest.raises(ValueError, match='at least two frames'):
        AMPLoader(str(path), ['foot'], 'torso', ['torso', 'foot'], device='cpu')
