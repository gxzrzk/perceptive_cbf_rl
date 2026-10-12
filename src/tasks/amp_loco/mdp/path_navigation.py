"""Finite-route navigation for G1: align, join a local segment, then follow.

Only the classroom config enables this layer. Refresh before terminations and
rewards; command/observation reads in the same step are idempotent. Differences
use the world bearing/segment saved before the action, never the new target.
"""
import math
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from .walk_path import walk_path_query


def _segment_distance(xy, start, end):
  segment = end - start
  fraction = ((xy - start) * segment).sum(-1) / segment.square().sum(-1).clamp_min(1e-9)
  point = start + fraction.clamp(0, 1)[:, None] * segment
  return (xy - point).norm(dim=-1)


def _center_velocity(command):
  """Pelvis COM velocity, expressed along its horizontal forward/left axes.

  COM avoids the omega-cross-offset velocity of a torso or link-frame origin.
  Heading-only rotation keeps pelvis roll/pitch from mixing vertical gait speed
  into the horizontal backward/lateral penalties.
  """
  data = command.robot.data
  velocity = data.root_com_lin_vel_w[:, :2]
  yaw = data.heading_w
  return torch.stack((velocity[:, 0] * yaw.cos() + velocity[:, 1] * yaw.sin(),
                      -velocity[:, 0] * yaw.sin() + velocity[:, 1] * yaw.cos()), -1)


def _join_target(command, xy, station, segment_end_s):
  """Choose a visible forward point on the current segment, then freeze it.

  Check the pelvis-center connector against inflated oriented obstacle boxes.
  If all local connectors are blocked, report that instead of commanding a walk
  through furniture. This is a local join planner, not a global obstacle detour.
  """
  env, cfg = command._env, command.cfg
  ar = torch.arange(command.num_envs, device=command.device)
  candidates = []
  for fraction in (1., .5, 0.):
    query = torch.minimum(station.clamp_min(0) + cfg.join_lookahead * fraction, segment_end_s)
    point, _ = walk_path_query(env, ar, query)
    candidates.append(point)
  candidates = torch.stack(candidates, 1)
  visible = torch.ones(candidates.shape[:2], dtype=torch.bool, device=command.device)
  if cfg.navigation_obstacles:
    ids = getattr(command, "_navigation_obstacle_ids", None)
    if ids is None:
      ids = torch.tensor([env.sim.mj_model.geom(f"{name}/wall_collision").id
                          for name in cfg.navigation_obstacles], device=command.device)
      command._navigation_obstacle_ids = ids
    center = env.sim.data.geom_xpos[:, ids, :2]
    rotation = env.sim.data.geom_xmat[:, ids].reshape(command.num_envs, len(ids), 3, 3)
    # Transform each connector into each obstacle's planar local frame.
    local_start = torch.einsum("nwij,nwi->nwj", rotation[:, :, :2, :2], xy[:, None] - center)
    local_delta = torch.einsum("nwij,ncwi->ncwj", rotation[:, :, :2, :2],
                              candidates[:, :, None] - xy[:, None, None])
    half = env.sim.model.geom_size[:, ids, :2] + cfg.join_obstacle_margin
    p = local_start[:, None]
    parallel = local_delta.abs() < 1e-8
    denominator = torch.where(parallel, 1., local_delta)
    lo = (-half[:, None] - p) / denominator
    hi = (half[:, None] - p) / denominator
    entry = torch.minimum(lo, hi)
    exit = torch.maximum(lo, hi)
    outside = parallel & (p.abs() > half[:, None])
    entry = torch.where(parallel, -torch.inf, entry).amax(-1).clamp_min(0)
    exit = torch.where(parallel, torch.inf, exit).amin(-1).clamp_max(1)
    intersects = (entry <= exit) & ~outside.any(-1)
    visible &= ~intersects.any(-1)
  chosen = visible.long().argmax(-1)
  return candidates[ar, chosen], visible.any(-1)


def classroom_follow_state(command):
  env, cfg = command._env, command.cfg
  heading = command.robot.data.heading_w
  xy = command.robot.data.root_link_pos_w[:, :2]
  epoch = getattr(env, "_walk_path_epoch", torch.zeros_like(heading, dtype=torch.long))
  step, dt = env.common_step_counter, env.step_dt
  state = getattr(command, "_classroom_follow_state", None)
  if state is None:
    zero = torch.zeros_like(heading)
    window = max(1, math.ceil(cfg.stall_duration / dt))
    state = SimpleNamespace(
      step=-1, epoch=torch.full_like(epoch, -1), xy=xy.clone(), heading=heading.clone(),
      bearing=heading.clone(), joining=torch.zeros_like(heading, dtype=torch.bool),
      aligning=torch.zeros_like(heading, dtype=torch.bool), join_target=xy.clone(),
      segment_start=xy.clone(), segment_end=xy.clone(), age=zero.clone(),
      recovery_age=zero.clone(), v_cmd=zero.clone(), command_speed=zero.clone(),
      target_reachable=torch.ones_like(heading, dtype=torch.bool),
      filtered_velocity=xy.new_zeros(len(xy), 2), s_frontier=zero.clone(),
      history=xy.new_zeros(len(xy), window, 9), cumulative=xy.new_zeros(len(xy), 9),
    )
    command._classroom_follow_state = state
  new = state.epoch != epoch
  update = torch.ones_like(new) if state.step != step else new
  if not update.any():
    return state

  path_active = cfg.path_follow and hasattr(env, "_walk_path_pts")
  if path_active:
    # Keep non-reset environments' rewards/references when only a subset resets.
    saved = {name: getattr(command, name).clone() for name in (
      "_path_s", "_path_s_delta", "_path_seg_idx", "_path_lat_err", "goal_pos_w")}
    command._update_path_goal()
    for name, old in saved.items():
      value = getattr(command, name)
      mask = update[:, None] if value.ndim == 2 else update
      setattr(command, name, torch.where(mask, value, old))
  distance = command._path_lat_err
  joining = torch.where(new, distance > cfg.join_d_in,
    torch.where(state.joining, distance >= cfg.join_d_in, distance > cfg.join_d_out))
  joining &= path_active
  step_joining = torch.where(new, joining, state.joining)
  join_target = state.join_target.clone()
  target_reachable = state.target_reachable.clone()
  start, end = xy, xy
  curvature = torch.zeros_like(heading)
  success = torch.zeros_like(new)
  signed_distance = distance.clone()
  if path_active:
    ar = torch.arange(command.num_envs, device=command.device)
    segment_idx = torch.minimum(command._path_seg_idx, env._walk_path_n - 2)
    start = env._walk_path_pts[ar, segment_idx]
    end = env._walk_path_pts[ar, segment_idx + 1]
    total = env._walk_path_cumlen[ar, env._walk_path_n - 1]
    # Stay inside the active segment; don't cut across a bend/furniture to an
    # arbitrary later lap. Freeze this world point until the join is complete.
    latch = new | (joining & ~state.joining) | (joining & ~state.target_reachable)
    selected, reachable = _join_target(command, xy, command._path_s,
                                       env._walk_path_cumlen[ar, segment_idx + 1])
    join_target = torch.where(latch[:, None], selected, join_target)
    target_reachable = torch.where(latch, reachable, target_reachable)
    command.goal_pos_w[:] = torch.where(joining[:, None], join_target, command.goal_pos_w)
    s0 = torch.minimum((command._path_s - cfg.curvature_window).clamp_min(0), total)
    s1 = torch.minimum((command._path_s + cfg.curvature_window).clamp_min(0), total)
    _, t0 = walk_path_query(env, ar, s0)
    _, t1 = walk_path_query(env, ar, s1)
    curvature = wrap_to_pi(t1 - t0) / (s1 - s0).clamp_min(1e-6)
    # A short sharp arc can be diluted by the forward averaging window. Include
    # adjacent-segment curvature so commanded v/R respects G1's yaw-rate limit.
    points = env._walk_path_pts
    previous = points[ar, (segment_idx - 1).clamp_min(0)]
    following = points[ar, torch.minimum(segment_idx + 2, env._walk_path_n - 1)]
    before, current, after = start - previous, end - start, following - end
    yaw_current = torch.atan2(current[:, 1], current[:, 0])
    for neighbor, valid, incoming in ((before, segment_idx > 0, True),
        (after, segment_idx < env._walk_path_n - 2, False)):
      neighbor_yaw = torch.atan2(neighbor[:, 1], neighbor[:, 0])
      angle = wrap_to_pi(yaw_current - neighbor_yaw if incoming else neighbor_yaw - yaw_current)
      local = 2 * angle / (neighbor.norm(dim=-1) + current.norm(dim=-1)).clamp_min(1e-6)
      curvature = torch.where(valid & (local.abs() > curvature.abs()), local, curvature)
    # See the peak curvature inside the lookahead early enough to decelerate,
    # instead of discovering a small-radius arc only after entering it.
    if points.shape[1] > 2:
      all_segments = points[:, 1:] - points[:, :-1]
      all_yaws = torch.atan2(all_segments[..., 1], all_segments[..., 0])
      vertex_curve = 2 * wrap_to_pi(all_yaws[:, 1:] - all_yaws[:, :-1]) / (
        all_segments[:, 1:].norm(dim=-1) + all_segments[:, :-1].norm(dim=-1)).clamp_min(1e-6)
      vertices_s = env._walk_path_cumlen[:, 1:-1]
      vertices_valid = torch.arange(1, points.shape[1] - 1, device=command.device)[None] < env._walk_path_n[:, None] - 1
      ahead = vertices_valid & (vertices_s >= command._path_s[:, None]) & (
        vertices_s <= command._path_s[:, None] + cfg.path_lookahead)
      future_curve = torch.where(ahead, vertex_curve, 0.)
      peak = future_curve.gather(1, future_curve.abs().argmax(-1)[:, None]).squeeze(-1)
      curvature = torch.where(peak.abs() > curvature.abs(), peak, curvature)
    segment = end - start
    signed_distance = ((segment[:, 0] * (xy - start)[:, 1]
                        - segment[:, 1] * (xy - start)[:, 0]) / segment.norm(dim=-1).clamp_min(1e-6))
    endpoint = env._walk_path_pts[ar, env._walk_path_n - 1]
    success = ((segment_idx == env._walk_path_n - 2)
               & (total - command._path_s <= cfg.arrive_radius)
               & (distance < cfg.join_d_in)
               & ((endpoint - xy).norm(dim=-1) < cfg.arrive_radius))

  # Score the executed interval against its OLD reference. Changing stages or
  # moving the lookahead only changes references for the NEXT physics step.
  old_distance = _segment_distance(state.xy, state.segment_start, state.segment_end)
  now_distance = _segment_distance(xy, state.segment_start, state.segment_end)
  join_delta = torch.where(new | (not path_active), 0., old_distance - now_distance)
  # Outward motion while following pays back join credit, even before d_out is
  # crossed. The d_in band exempts ordinary gait sway on the path.
  outward_delta = (now_distance - cfg.join_d_in).clamp_min(0) - (
    old_distance - cfg.join_d_in).clamp_min(0)
  join_credit_delta = torch.where(new, 0., -outward_delta)
  outward_delta = torch.where(new, 0., outward_delta.clamp_min(0))
  direction = command.goal_pos_w - xy
  target_distance = direction.norm(dim=-1)
  bearing = torch.where(target_distance > 1e-6, torch.atan2(direction[:, 1], direction[:, 0]), heading)
  theta = wrap_to_pi(bearing - heading)
  step_theta = wrap_to_pi(state.bearing - heading)
  turn_delta = wrap_to_pi(state.bearing - state.heading).abs() - step_theta.abs()
  turn_delta = torch.where(new, 0., turn_delta)
  step_theta = torch.where(new, theta, step_theta)
  aligning = torch.where(new, theta.abs() > cfg.align_enter_angle + 1e-6,
    torch.where(state.aligning, theta.abs() >= cfg.align_exit_angle - 1e-6,
                theta.abs() > cfg.align_enter_angle + 1e-6))
  step_aligning = torch.where(new, aligning, state.aligning)
  v_cmd = cfg.path_v_ref * theta.cos().clamp_min(0) / (
    1 + cfg.curvature_alpha * curvature.abs() * (~joining))
  v_cmd = v_cmd.clamp(max=cfg.max_lin_vel_x)
  turn_speed_limit = cfg.max_ang_vel_z / curvature.abs().clamp_min(1e-6)
  v_cmd = torch.where(joining, v_cmd, torch.minimum(v_cmd, turn_speed_limit))
  desired = torch.where(aligning | success | (joining & ~target_reachable), 0., v_cmd)
  command_speed = state.command_speed + (desired - state.command_speed).clamp(
    -cfg.command_acceleration * dt, cfg.command_acceleration * dt)
  command_speed = torch.where(new, desired.clamp(max=cfg.command_acceleration * dt), command_speed)
  command_speed = torch.where(aligning | success, 0., command_speed)
  # A safety cap also covers a reset/teleport directly into a sharp arc. In
  # ordinary motion the lookahead curvature permits the smooth ramp first.
  command_speed = torch.where(joining, command_speed,
    torch.minimum(command_speed, torch.minimum(turn_speed_limit, torch.full_like(command_speed, cfg.max_lin_vel_x))))
  raw_velocity = _center_velocity(command)
  filter_gain = 1. if cfg.velocity_filter_time == 0 else 1 - math.exp(-dt / cfg.velocity_filter_time)
  filtered = state.filtered_velocity + filter_gain * (raw_velocity - state.filtered_velocity)
  filtered = torch.where(new[:, None], 0., filtered)
  delta_s = command._path_s_delta if path_active else torch.zeros_like(heading)
  travel_bound = (xy - state.xy).norm(dim=-1) * cfg.projection_progress_ratio
  delta_s = delta_s.clamp(min=-travel_bound, max=travel_bound)
  delta_s = torch.where(new, 0., delta_s)
  speed = delta_s / dt
  # Never pay repeatedly for the same forward arc length. This also blocks slow
  # advance / fast retreat exploiting the asymmetric time spent at clipped rates.
  frontier = torch.where(new, command._path_s, state.s_frontier)
  novel_s = (command._path_s - frontier).clamp_min(0)
  novel_s = torch.minimum(novel_s, delta_s.clamp_min(0))
  frontier = torch.maximum(frontier, command._path_s)
  safe = ~command._dodge_threat
  eligible = safe & ~new & ~success
  active_turn = eligible & step_aligning
  active_join = eligible & ~step_aligning & step_joining
  active_follow = eligible & ~step_aligning & ~step_joining
  increments = torch.stack((turn_delta * active_turn, join_delta * active_join,
    delta_s * active_follow, cfg.turn_progress_threshold * dt * active_turn,
    cfg.join_progress_rate * dt * active_join,
    cfg.follow_progress_fraction * state.command_speed * dt * active_follow,
    dt * active_turn, dt * active_join, dt * active_follow), -1)
  cumulative = torch.where(new[:, None], 0., state.cumulative) + increments
  state.history[new] = 0.
  index = step % state.history.shape[1]
  window_progress = cumulative - state.history[:, index]
  state.history[:, index] = torch.where(update[:, None], cumulative, state.history[:, index])
  age = torch.where(new, 0., state.age + dt)
  ready = age >= max(cfg.startup_grace, cfg.stall_duration) - 1e-6
  # Require enough evidence from this phase inside the same rolling window.
  # This permits gait startup after a long turn without granting a new episode
  # grace on every switch; alternating phases accumulates evidence in both.
  observed = window_progress[:, 6:] >= cfg.phase_min_window_time - 1e-6
  turn_stall = ready & active_turn & observed[:, 0] & (window_progress[:, 0] <= window_progress[:, 3] + 1e-7)
  join_stall = ready & active_join & observed[:, 1] & (window_progress[:, 1] <= window_progress[:, 4] + 1e-7)
  follow_stall = ready & active_follow & observed[:, 2] & (window_progress[:, 2] <= window_progress[:, 5] + 1e-7)
  # A phase switch grants no new startup grace and cannot restart this watchdog.
  # Recovery ends only with actual novel forward motion while on the path.
  recovered = active_follow & (novel_s > cfg.join_progress_rate * dt)
  recovery_age = torch.where(new | recovered | success, 0., state.recovery_age + dt * safe)
  recovery_stall = eligible & ready & (recovery_age >= cfg.recovery_timeout)
  values = dict(xy=xy, heading=heading, bearing=bearing, theta=theta, step_theta=step_theta,
    target_distance=target_distance, joining=joining, step_joining=step_joining,
    join_target=join_target, target_reachable=target_reachable,
    aligning=aligning, step_aligning=step_aligning,
    segment_start=start, segment_end=end, distance=distance, signed_distance=signed_distance,
    turn_delta=turn_delta, join_delta=join_delta, join_credit_delta=join_credit_delta,
    outward_delta=outward_delta,
    curvature=curvature, v_cmd=v_cmd, command_speed=command_speed,
    speed=speed, novel_speed=novel_s / dt, s_frontier=frontier,
    filtered_velocity=filtered, age=age, cumulative=cumulative, recovery_age=recovery_age,
    turn_stall=turn_stall, join_stall=join_stall, follow_stall=follow_stall,
    recovery_stall=recovery_stall, success=success)
  for name, value in values.items():
    old = getattr(state, name, torch.zeros_like(value))
    mask = update[:, None] if value.ndim == 2 else update
    setattr(state, name, torch.where(mask, value, old).clone())
  state.epoch, state.step = epoch.clone(), step
  return state


def classroom_path_success(env, command_name="twist"):
  """Register last: falls, collisions and ball hits take precedence."""
  state = classroom_follow_state(env.command_manager.get_term(command_name))
  success = state.success.clone()
  manager = getattr(env, "termination_manager", None)
  if manager is not None:
    for name in manager.active_terms:
      if name not in ("path_success", "time_out"):
        success &= ~manager.get_term(name)
  return success


def classroom_path_follow_reward(
  env, command_name="twist", k_theta=2., k_join=2., k_p=2., k_b=2.,
  backward_weight=1., backward_limit=.05, lateral_weight=.5, lateral_limit=.2,
  align_lateral_scale=.2, deviation_weight=.5, deviation_std=.5,
  join_deviation_scale=.2, movement_cost_cap=4., deviation_cost_cap=4.,
  stall_weight=1.2, turn_stall_weight=1.2, join_stall_weight=1.2,
  recovery_stall_weight=1.2, success_bonus=10.,
):
  """Reward rate: manager integrates rates once; differences/events divide by dt."""
  if (min(k_theta, k_join, k_p, backward_weight, lateral_weight, deviation_weight,
          stall_weight, turn_stall_weight, join_stall_weight, recovery_stall_weight,
          success_bonus) < 0 or k_b < k_p or deviation_std <= 0 or lateral_limit < 0
      or backward_limit < 0 or min(movement_cost_cap, deviation_cost_cap) <= 0
      or not 0 <= align_lateral_scale <= 1 or not 0 <= join_deviation_scale <= 1):
    raise ValueError("Invalid classroom reward parameters (require k_b >= k_p)")
  command = env.command_manager.get_term(command_name)
  state, cfg = classroom_follow_state(command), command.cfg
  dt = env.step_dt
  g = state.step_theta.cos().clamp_min(0).square()
  follow = (~state.step_aligning & ~state.step_joining).float()
  join = (state.step_joining & ~state.step_aligning).float()
  safe = (~command._dodge_threat).float()
  # Stop both sides of join accounting at d_in. Otherwise an inward step that
  # overshoots the join threshold could earn unrepaid credit inside the gait band.
  positive_join = g * state.join_credit_delta.clamp_min(0) * join
  negative_join = (-state.join_credit_delta).clamp_min(0)
  velocity = state.filtered_velocity
  backward = ((-velocity[:, 0] - backward_limit).clamp_min(0) / cfg.path_v_ref).square().clamp(max=movement_cost_cap)
  lateral = ((velocity[:, 1].abs() - lateral_limit).clamp_min(0) / cfg.path_v_ref).square().clamp(max=movement_cost_cap)
  lateral *= torch.where(state.step_aligning, align_lateral_scale, 1.)
  deviation = (state.distance / deviation_std).square().clamp(max=deviation_cost_cap)
  deviation *= torch.where(state.step_joining, join_deviation_scale, 1.)
  manager = getattr(env, "termination_manager", None)
  success = manager.get_term("path_success") if manager is not None else state.success
  parts = {
    "turn": k_theta * state.turn_delta / torch.pi,
    "join": k_join * (positive_join - negative_join) / cfg.navigation_d_ref,
    "forward": dt * k_p * g * (state.novel_speed / cfg.path_v_ref).clamp(0, 1) * follow,
    "retreat": -dt * k_b * (-state.speed / cfg.path_v_ref).clamp(0, 1),
    "body_backward": -dt * safe * backward_weight * backward,
    "lateral": -dt * safe * lateral_weight * lateral,
    "deviation": -dt * deviation_weight * deviation,
    "turn_stall": -dt * turn_stall_weight * state.turn_stall.float(),
    "join_stall": -dt * join_stall_weight * state.join_stall.float(),
    "follow_stall": -dt * stall_weight * state.follow_stall.float(),
    "recovery_stall": -dt * recovery_stall_weight * state.recovery_stall.float(),
    "success": success_bonus * success.float(),
  }
  state.reward_components = parts
  return sum(parts.values()) / dt


def classroom_navigation_observation(env, command_name="twist"):
  """14 scaled planner/phase features; model also uses running normalization."""
  command = env.command_manager.get_term(command_name)
  # ObservationManager probes dimensions before startup/reset generates a route.
  if not hasattr(env, "_walk_path_pts"):
    return torch.zeros(env.num_envs, 14, device=env.device)
  state, cfg = classroom_follow_state(command), command.cfg
  return torch.stack((state.theta.cos(), state.theta.sin(),
    state.signed_distance / cfg.navigation_d_ref, state.distance / cfg.navigation_d_ref,
    state.target_distance / cfg.path_lookahead, state.command_speed / cfg.path_v_ref,
    state.joining.float(), state.aligning.float(),
    state.filtered_velocity[:, 0] / cfg.path_v_ref, state.filtered_velocity[:, 1] / cfg.path_v_ref,
    (state.age / max(cfg.startup_grace, cfg.stall_duration)).clamp(max=1),
    (state.recovery_age / cfg.recovery_timeout).clamp(max=1), state.target_reachable.float(),
    (state.s_frontier - command._path_s).clamp_min(0) / cfg.navigation_d_ref), -1).clamp(
      -cfg.navigation_obs_clip, cfg.navigation_obs_clip)


def classroom_navigation_metric(env, key, command_name="twist"):
  state = classroom_follow_state(env.command_manager.get_term(command_name))
  if key.startswith("reward_"):
    return state.reward_components[key.removeprefix("reward_")]
  if key == "phase":
    return torch.where(state.aligning, 2., state.joining.float())
  return getattr(state, key).float()


def classroom_failure_cost(env, base_cost=10., remaining_rate=4.):
  """One-shot failure cost plus remaining navigation-cost budget.

  Makes deliberately falling to avoid bounded ongoing path penalties costly.
  Ball hits retain the existing no-spike treatment; simultaneous falls still pay.
  Like success, this event compensates the manager's dt scaling.
  """
  manager = env.termination_manager
  failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in manager.active_terms:
    if name not in ("time_out", "ball_hit", "path_success"):
      failure |= manager.get_term(name)
  remaining = (env.max_episode_length_s - env.episode_length_buf * env.step_dt).clamp_min(0)
  return failure.float() * (base_cost + remaining_rate * remaining) / env.step_dt
