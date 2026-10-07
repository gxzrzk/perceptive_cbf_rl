"""AMP NPZ motion loader.

Pre-computes per-frame anchor-relative body pose/velocity tensors from a single
NPZ or a directory of NPZs, then yields (state, next_state) batches for
discriminator training.

Ported from the previous vendored rsl_rl fork (rsl_rl/utils/motion_loader.py).
The per-frame features (anchor-relative position, 6D orientation, body-frame
linear/angular velocity) match the env's ``amp`` observation group exactly, so
the discriminator compares like-with-like.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

import mjlab.utils.lab_api.math as math_utils


class AMPLoader:
  """Holds expert motion clips in body-local form for AMP discriminator training."""

  def __init__(
    self,
    motion_file: str,
    body_names: Sequence[str],
    anchor_name: str,
    all_body_names: Sequence[str],
    device: str = "cuda:0",
  ) -> None:
    """Args:
      motion_file: Path to a single .npz or a directory of .npz files.
      body_names: Tracked body names (used for AMP obs).
      anchor_name: Anchor body (typically root/torso) for the body-local frame.
      all_body_names: Ordered list of *all* body names in the model. Indexes
        each body in the NPZs by its position here.
      device: Torch device.
    """
    assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

    all_names = list(all_body_names)
    self._body_indexes = [all_names.index(n) for n in body_names]
    self._anchor_indexes = all_names.index(anchor_name)
    self._num_bodies = len(self._body_indexes)

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
      assert motion_files, f"No npz files found in directory: {motion_file}"
    else:
      raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

    self.motion_names = motion_names
    self._body_pos_b_list: list[torch.Tensor] = []
    self._body_quat_b_list: list[torch.Tensor] = []
    self._body_ori_b_list: list[torch.Tensor] = []
    self._body_lin_vel_b_list: list[torch.Tensor] = []
    self._body_ang_vel_b_list: list[torch.Tensor] = []

    for motion_idx, (motion_name, motion_path) in enumerate(
      zip(motion_names, motion_files)
    ):
      print(f"Processing motion {motion_idx + 1}/{len(motion_files)}: {motion_name}")
      data = np.load(motion_path)
      if motion_idx == 0:
        self.fps = float(np.asarray(data["fps"]).item())

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
        raise ValueError(f"AMP motion {motion_path} needs at least two frames")
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

      for frame_idx in tqdm(
        range(n_steps), desc=f"Preloading AMP data for {motion_name}"
      ):
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

        bp_b, bq_b = math_utils.subtract_frame_transforms(
          anchor_pos_w, anchor_quat_w, bp_w, bq_w
        )
        mat = math_utils.matrix_from_quat(bq_b)
        # First two columns of the rotation matrix -> 6D continuous rotation.
        ori_6 = mat[..., :, :2].reshape(self._num_bodies, 6)
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
    self._build_sampling_cache()

  @property
  def observation_dim(self) -> int:
    return (3 + 6 + 3 + 3) * self._num_bodies

  def _build_sampling_cache(self) -> None:
    """Pack clips once so each training batch needs only batched GPU gathers."""
    lengths = [pos.shape[0] for pos in self._body_pos_b_list]
    if not lengths or min(lengths) < 2:
      raise ValueError("AMP sampling requires nonempty clips with at least two frames")
    device = self._body_pos_b_list[0].device
    self._motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self._motion_offsets = self._motion_lengths.cumsum(0) - self._motion_lengths
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
    """Mix all clips in every batch, with counts differing by at most one.

    Remainder slots go to randomly chosen clips, and sample order is shuffled.
    Frames are drawn uniformly from valid adjacent pairs within each chosen clip;
    neither clip boundaries nor artificial final-frame self-pairs are sampled.
    When a batch is smaller than the clip count, sample a random subset without
    replacement rather than increasing the requested training batch size.
    """
    if mini_batch_size < 1 or num_mini_batch < 0:
      raise ValueError("mini_batch_size must be positive and num_mini_batch nonnegative")
    num_motions = self._motion_lengths.numel()
    device = self._motion_frames.device
    slots = torch.arange(mini_batch_size, device=device) % num_motions
    for _ in range(num_mini_batch):
      # Randomize which clips get the extra slots when the batch is indivisible.
      motion_order = torch.randperm(num_motions, device=device)
      motion_ids = motion_order[slots]
      motion_ids = motion_ids[torch.randperm(mini_batch_size, device=device)]
      pair_counts = self._motion_lengths[motion_ids] - 1
      local_frame = (torch.rand(mini_batch_size, device=device) * pair_counts).long()
      frame_ids = self._motion_offsets[motion_ids] + local_frame
      yield self._motion_frames[frame_ids], self._motion_frames[frame_ids + 1]
