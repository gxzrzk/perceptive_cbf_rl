"""Balanced mixed-clip AMP batches and valid within-clip transition sampling."""
import numpy as np
import pytest
import torch

from src.tasks.amp_loco.rl.amp_motion_loader import AMPLoader


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def make_loader(num_clips=54, device='cpu'):
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
            # Same clip, next chronological frame, including two-frame clips.
            assert torch.equal(next_state[:, 0], state[:, 0])
            assert torch.equal(next_state[:, 1], state[:, 1] + 1)
            assert (state[:, 1] < loader._motion_lengths[ids] - 1).all()
            assert (state[:, 6:18] == 22.).all()
            assert (state[:, 18:24] == 33.).all()
            assert (state[:, 24:] == 44.).all()
        assert batches == 20
    # Extra slots are not permanently assigned to the same clip subset.
    assert not torch.equal(extras[0], extras[1])


@pytest.mark.parametrize('batch_size', [1, 20, 54, 108])
def test_small_and_divisible_batch_sizes(batch_size):
    loader = make_loader()
    for state, next_state in loader.feed_forward_generator(3, batch_size):
        counts = state[:, 0].long().bincount(minlength=54)
        assert len(state) == batch_size
        assert counts.max() - counts.min() <= 1
        assert (counts > 0).sum() == min(batch_size, 54)
        assert torch.equal(next_state[:, 1], state[:, 1] + 1)


def test_single_two_frame_clip_and_repeatable_seed():
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
    pos = np.zeros((n, 2, 3), dtype=np.float32)
    pos[:, 1, 0] = np.arange(n)
    quat = np.zeros((n, 2, 4), dtype=np.float32)
    quat[:, :, 0] = 1.
    np.savez(path, fps=50., body_pos_w=pos, body_quat_w=quat,
             body_lin_vel_w=np.full_like(pos, 3.), body_ang_vel_w=np.full_like(pos, 4.))


def test_constructor_packs_real_npz_features(tmp_path):
    path = tmp_path / 'motion.npz'
    write_motion(path, 4)
    loader = AMPLoader(str(path), ['foot'], 'torso', ['torso', 'foot'], device='cpu')
    state, next_state = next(loader.feed_forward_generator(1, 64))
    assert loader.observation_dim == 15 and state.shape == (64, 15)
    assert torch.equal(next_state[:, 0], state[:, 0] + 1)
    assert (state[:, 9:12] == 3.).all() and (state[:, 12:] == 4.).all()
    assert torch.isfinite(state).all()


def test_single_frame_clip_rejected(tmp_path):
    path = tmp_path / 'single.npz'
    write_motion(path, 1)
    with pytest.raises(ValueError, match='at least two frames'):
        AMPLoader(str(path), ['foot'], 'torso', ['torso', 'foot'], device='cpu')
