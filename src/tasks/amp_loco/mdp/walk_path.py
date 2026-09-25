"""Random walk-path generation, extension, and arc-length queries (WallWalk task).

Each episode every env gets its OWN randomly shaped walking path: a 2-D polyline
that starts at the robot's post-reset position, aligned with its post-reset
heading, then wanders with gentle random turns (segment length ~ U(seg_len_range),
heading increment ~ U(±turn_max) per segment). The path never enters the policy's
observation -- the command term (``DodgeGoToGoalCommand.path_follow``) converts it
into a pure-pursuit velocity command (a lookahead point on the path becomes the
"goal", which the existing go-to-goal P-controller turns into vx/vy/wz), and the
wall events place/recycle the obstacle walls at arc-length stations along it.

The path data is stashed on the env (same pattern as ``_wall_pos_w`` /
``_wall_yaw_w`` in events.py) so the three consumers -- the reset event
(``reset_walk_path``), the command term (progress tracking + lookahead goal +
extension), and the wall events (placement / recycling) -- share one source of
truth with no cross-manager ordering dependency:

  * ``_walk_path_pts``    (N, MAX_POINTS, 2)  polyline vertices, world frame
  * ``_walk_path_cumlen`` (N, MAX_POINTS)     cumulative arc length at each vertex
  * ``_walk_path_n``      (N,)                number of valid vertices
  * ``_walk_path_epoch``  (N,)                +1 per (re)generation; the command
                                              term watches this to detect a new
                                              episode's path and restart progress
  * ``_walk_path_params`` tuple               (seg_lo, seg_hi, turn_max) used at
                                              generation, reused by extension
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

WALK_PATH_MAX_POINTS = 160
"""Vertex capacity per env. At ~3 m/segment this is ~480 m of path -- far beyond
any episode (~26 m at 1.3 m/s x 20 s), so the capacity guard in
:func:`extend_walk_path` effectively never binds; it exists so a runaway play
session degrades gracefully (the goal clamps at the path end) instead of
corrupting memory."""

_WALK_PATH_PTS_ATTR = "_walk_path_pts"
_WALK_PATH_CUMLEN_ATTR = "_walk_path_cumlen"
_WALK_PATH_N_ATTR = "_walk_path_n"
_WALK_PATH_EPOCH_ATTR = "_walk_path_epoch"
_WALK_PATH_PARAMS_ATTR = "_walk_path_params"

__all__ = [
  "WALK_PATH_MAX_POINTS",
  "generate_walk_path",
  "extend_walk_path",
  "walk_path_query",
]


def _ensure_stash(env: "ManagerBasedRlEnv") -> None:
  pts = getattr(env, _WALK_PATH_PTS_ATTR, None)
  if pts is None or pts.shape[0] != env.num_envs:
    device = env.device
    setattr(env, _WALK_PATH_PTS_ATTR, torch.zeros(env.num_envs, WALK_PATH_MAX_POINTS, 2, device=device))
    setattr(env, _WALK_PATH_CUMLEN_ATTR, torch.zeros(env.num_envs, WALK_PATH_MAX_POINTS, device=device))
    setattr(env, _WALK_PATH_N_ATTR, torch.zeros(env.num_envs, dtype=torch.long, device=device))
    setattr(env, _WALK_PATH_EPOCH_ATTR, torch.zeros(env.num_envs, dtype=torch.long, device=device))


def _fresh_root_yaw(env: "ManagerBasedRlEnv", robot: "Entity", env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """(root_xy, root_yaw) read back from sim qpos (freejoint layout pos(3)+quat wxyz(4)).

  ``robot.data`` is STALE at reset-event time (reset events fire back-to-back with
  no entity-data refresh; reset_from_motion writes the RSI pose straight into sim
  qpos). Same trick as ``reset_wall_pose`` in events.py.
  """
  adr = robot.data.indexing.free_joint_q_adr
  root_pose = env.sim.data.qpos[env_ids][:, adr]  # (n, 7)
  q = root_pose[:, 3:7]
  yaw = torch.atan2(
    2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
    1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2),
  )
  return root_pose[:, :2], yaw


def _append_segments(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  start_xy: torch.Tensor,
  start_yaw: torch.Tensor,
  num_segments: int,
  seg_len_range: tuple[float, float],
  turn_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample ``num_segments`` random polyline segments continuing from
  ``(start_xy, start_yaw)``; returns (points (n, num_segments, 2), seg_len (n, num_segments))
  where points[:, k] is the END vertex of segment k."""
  n = len(env_ids)
  device = env.device
  seg_len = torch.rand(n, num_segments, device=device) * (
    seg_len_range[1] - seg_len_range[0]
  ) + seg_len_range[0]
  dyaw = (torch.rand(n, num_segments, device=device) * 2.0 - 1.0) * turn_max
  # EXCLUSIVE cumsum: segment 0 keeps the start heading (so the path leaves the
  # spawn aligned with the robot / the last heading of the previous chain), the
  # random increments turn it from segment 1 on.
  yaw = start_yaw.unsqueeze(1) + (torch.cumsum(dyaw, dim=1) - dyaw)  # heading per segment
  step = torch.stack([seg_len * torch.cos(yaw), seg_len * torch.sin(yaw)], dim=-1)  # (n, K, 2)
  pts = start_xy.unsqueeze(1) + torch.cumsum(step, dim=1)  # (n, K, 2)
  return pts, seg_len


def generate_walk_path(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  robot_name: str = "robot",
  seg_len_range: tuple[float, float] = (2.0, 4.0),
  turn_max: float = 0.45,
  initial_len: float = 80.0,
) -> None:
  """(Re)generate the random walk path for ``env_ids`` (called at episode reset).

  The path starts at the robot's post-reset xy with its post-reset yaw, then
  extends for at least ``initial_len`` metres of gentle random turns
  (``turn_max`` rad per ``seg_len_range`` m -- 0.45 rad / 2-4 m is a ~4.4-8.9 m
  turning radius, comfortably trackable at 1.3 m/s with a 1 rad/s yaw limit).
  Bumps ``_walk_path_epoch`` so the command term restarts its progress tracking.
  """
  if len(env_ids) == 0:
    return
  _ensure_stash(env)
  robot: Entity = env.scene[robot_name]
  n = len(env_ids)
  device = env.device

  num_segments = min(
    WALK_PATH_MAX_POINTS - 1, math.ceil(initial_len / seg_len_range[0])
  )
  start_xy, start_yaw = _fresh_root_yaw(env, robot, env_ids)
  pts_new, seg_len = _append_segments(
    env, env_ids, start_xy, start_yaw, num_segments, seg_len_range, turn_max
  )  # (n, K, 2), (n, K)

  pts = getattr(env, _WALK_PATH_PTS_ATTR)
  cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)
  count = getattr(env, _WALK_PATH_N_ATTR)

  pts[env_ids, 0] = start_xy
  pts[env_ids, 1 : 1 + num_segments] = pts_new
  cum[env_ids, 0] = 0.0
  cum[env_ids, 1 : 1 + num_segments] = torch.cumsum(seg_len, dim=1)
  count[env_ids] = 1 + num_segments
  getattr(env, _WALK_PATH_EPOCH_ATTR)[env_ids] += 1
  setattr(env, _WALK_PATH_PARAMS_ATTR, (seg_len_range[0], seg_len_range[1], turn_max))


def extend_walk_path(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  min_total_len: torch.Tensor,
) -> None:
  """Append random segments until each env's total path length >= ``min_total_len`` (n,).

  Continues from the last segment's heading, reusing the generation-time
  (seg_len_range, turn_max). Envs at the vertex capacity are skipped (their goal
  clamps at the path end -- see ``WALK_PATH_MAX_POINTS``). Does NOT bump the
  epoch: this is the same episode's path, just longer.
  """
  if len(env_ids) == 0:
    return
  _ensure_stash(env)
  params = getattr(env, _WALK_PATH_PARAMS_ATTR, (2.0, 4.0, 0.45))
  seg_lo, seg_hi, turn_max = params
  pts = getattr(env, _WALK_PATH_PTS_ATTR)
  cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)
  count = getattr(env, _WALK_PATH_N_ATTR)

  for _ in range(8):  # 8 x 16 segments x >=2 m >= 256 m per call; plenty
    total = cum[env_ids, count[env_ids] - 1]
    need = (total < min_total_len) & (count[env_ids] <= WALK_PATH_MAX_POINTS - 17)
    if not need.any():
      return
    ids = env_ids[need]
    n = len(ids)
    device = env.device
    K = 16
    c = count[ids]
    last = pts[ids, c - 1]
    prev = pts[ids, c - 2]
    last_yaw = torch.atan2(last[:, 1] - prev[:, 1], last[:, 0] - prev[:, 0])
    pts_new, seg_len = _append_segments(
      env, ids, last, last_yaw, K, (seg_lo, seg_hi), turn_max
    )
    # Scatter the new vertices in at per-env positions c .. c+K-1.
    slots = c.unsqueeze(1) + torch.arange(K, device=device).unsqueeze(0)  # (n, K)
    pts[ids.unsqueeze(1), slots] = pts_new
    cum_new = total[need].unsqueeze(1) + torch.cumsum(seg_len, dim=1)
    cum[ids.unsqueeze(1), slots] = cum_new
    count[ids] = c + K


def walk_path_query(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Point + tangent on the path at arc length ``s`` (n,) -> (pos_xy (n,2), yaw (n,)).

  ``s`` is clamped into [0, total_length]: querying past the end pins to the last
  vertex (the command extends the path long before this can matter).
  """
  pts = getattr(env, _WALK_PATH_PTS_ATTR)[env_ids]  # (n, M, 2)
  cum = getattr(env, _WALK_PATH_CUMLEN_ATTR)[env_ids]  # (n, M)
  count = getattr(env, _WALK_PATH_N_ATTR)[env_ids]  # (n,)
  n = len(env_ids)
  ar = torch.arange(n, device=env.device)

  total = cum[ar, count - 1]
  s = s.clamp(min=torch.zeros_like(total), max=total)
  # First vertex with cum > s, minus one -> the segment containing s. The padded
  # tail of cum is 0 (not monotonic), which would corrupt searchsorted -- mask it
  # to +inf so the search only sees the valid prefix.
  valid = (
    torch.arange(cum.shape[1], device=env.device).unsqueeze(0) < count.unsqueeze(1)
  )
  cum_search = torch.where(valid, cum, torch.full_like(cum, float("inf")))
  idx = torch.searchsorted(cum_search, s.unsqueeze(1), right=True).squeeze(1) - 1
  idx = idx.clamp(min=torch.zeros_like(count), max=(count - 2).clamp(min=0))

  p0 = pts[ar, idx]  # (n, 2)
  p1 = pts[ar, idx + 1]
  c0 = cum[ar, idx]
  c1 = cum[ar, idx + 1]
  t = ((s - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0.0, 1.0)
  pos = p0 + t.unsqueeze(-1) * (p1 - p0)
  yaw = torch.atan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
  return pos, yaw
