"""Tests for raycast observation sensor-model behavior."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.observations import (
  _vl53l8cx_false_max_probability,
  raycast_distances,
)


def test_vl53l8cx_false_max_probability_prefers_shallow_hits() -> None:
  distances_m = torch.full((1, 2), 1.0)
  hit_mask = torch.ones((1, 2), dtype=torch.bool)
  corner_mask = torch.zeros(2, dtype=torch.bool)
  ray_dirs_w = torch.tensor([[[1.0, 0.0, -0.05], [1.0, 0.0, 0.0]]])
  normals_w = torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]])

  probs = _vl53l8cx_false_max_probability(
    distances_m=distances_m,
    max_distance=4.0,
    hit_mask=hit_mask,
    corner_mask=corner_mask,
    ray_dirs_w=ray_dirs_w,
    normals_w=normals_w,
    shallow_incidence_invalid_prob=0.75,
    shallow_incidence_cos_start=0.35,
    shallow_incidence_cos_full=0.08,
  )

  assert probs.shape == (1, 2)
  assert torch.all((0.0 <= probs) & (probs <= 1.0))
  assert probs[0, 0] > probs[0, 1]


def test_vl53l8cx_false_max_probability_uses_distance_and_corners() -> None:
  distances_m = torch.tensor([[1.0, 4.0]])
  hit_mask = torch.ones((1, 2), dtype=torch.bool)
  corner_mask = torch.tensor([False, True])

  probs = _vl53l8cx_false_max_probability(
    distances_m=distances_m,
    max_distance=4.0,
    hit_mask=hit_mask,
    corner_mask=corner_mask,
    invalid_distance_start_m=2.0,
    invalid_prob_at_max=0.2,
    corner_invalid_prob_extra=0.1,
  )

  assert probs.shape == (1, 2)
  assert torch.all((0.0 <= probs) & (probs <= 1.0))
  assert probs[0, 0] == 0.0
  assert torch.isclose(probs[0, 1], torch.tensor(0.3))


def test_raycast_distances_shape_range_and_shallow_invalid() -> None:
  distances = torch.full((1, 64), 2.0)
  hit_pos_w = torch.zeros(1, 64, 3)
  hit_pos_w[:, :, 0] = 2.0
  normals_w = torch.zeros(1, 64, 3)
  normals_w[:, :, 0] = -1.0

  hit_pos_w[:, 0, :] = torch.tensor([2.0, 0.0, -0.1])
  normals_w[:, 0, :] = torch.tensor([0.0, 0.0, 1.0])

  sensor = SimpleNamespace(
    cfg=SimpleNamespace(max_distance=4.0),
    data=SimpleNamespace(
      distances=distances,
      hit_pos_w=hit_pos_w,
      pos_w=torch.zeros(1, 3),
      normals_w=normals_w,
    ),
  )
  env = SimpleNamespace(scene={"raycast": sensor})

  obs = raycast_distances(
    env,  # type: ignore[arg-type]
    sensor_name="raycast",
    sensor_width=8,
    vl53l8cx_shallow_incidence_invalid_prob=1.0,
    vl53l8cx_shallow_incidence_cos_start=0.35,
    vl53l8cx_shallow_incidence_cos_full=0.08,
  )

  assert obs.shape == (1, 64)
  assert torch.all((0.0 <= obs) & (obs <= 1.0))
  assert obs[0, 0] == 1.0
  assert obs[0, 1] == 0.5
