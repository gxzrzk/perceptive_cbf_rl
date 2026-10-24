#!/usr/bin/env python3
"""Paired PAC-MAN / LP-NavOA classroom evaluation on fixed turning routes.

The parent process creates one JSON route manifest. Each backend runs in its own
Python environment and evaluates those exact routes in its native simulator.
Coordinates in the manifest use the PAC-MAN classroom frame (x forward, y left).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LP_ROOT = ROOT / "others" / "LP-NavOA-code"
DEFAULT_PAC = ROOT / "logs/rsl_rl/classroom_vs_lp/2026-10-22_15-42-09/model_24999.pt"
DEFAULT_BACKBONE = LP_ROOT / "logs/rsl_rl/g1_velocity_raycast_h1/2026-10-22_00-33-49/model_12497.pt"
DEFAULT_PLANNER = LP_ROOT / "logs/steering/local_planner_classroom_v6_511_final.pt"
STATIONS = (0.0, 3.1, 6.2, 9.3, 12.4, 15.5)
LANES = (-5.4, -2.7, 0.0, 2.7)
SPEED = 0.6
ARRIVAL_RADIUS_M = 0.8
HARD_PENETRATION_M = 0.03


def make_routes(count: int, seed: int, margin_s: float) -> list[dict]:
    rng = random.Random(seed)
    routes = []
    for case_id in range(count):
        lane_index = rng.choice((1, 2))
        first_station = rng.choice((1, 2))
        turn = rng.choice((-1, 1))
        lane = LANES[lane_index]
        next_lane = LANES[lane_index + turn]
        points = [[0.2, lane], [STATIONS[first_station], lane],
                  [STATIONS[first_station], next_lane]]
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        routes.append({"case_id": case_id, "points_pac_xy_m": points,
                       "turn": "left" if turn > 0 else "right",
                       "L_obj_m": length, "T_obj_s": length / SPEED + margin_s})
    return routes


def aggregate(rows: list[dict], slow_brush_definition: str) -> dict:
    out = {}
    for name in sorted({r["model"] for r in rows}):
        group = [r for r in rows if r["model"] == name]
        n = len(group)
        if not n:
            continue
        if slow_brush_definition == "late":
            slow_brush = lambda r: r["arrived"] and not r["on_time"] and r["any_brush"]
        elif slow_brush_definition == "on-time":
            slow_brush = lambda r: r["arrived"] and r["on_time"] and r["any_brush"]
        else:
            slow_brush = lambda r: r["arrived"] and r["any_brush"]
        rate = lambda predicate: sum(bool(predicate(r)) for r in group) / n
        out[name] = {
            "episodes": n,
            "success_rate": rate(lambda r: r["arrived"]),
            "on_time_arrival_rate": rate(lambda r: r["on_time"]),
            "L_obj_m": sum(r["L_obj_m"] for r in group) / n,
            "T_obj_s": sum(r["T_obj_s"] for r in group) / n,
            "episode_duration_s": sum(r["episode_duration_s"] for r in group) / n,
            "any_brush_rate": rate(lambda r: r["any_brush"]),
            "slow_brush_arrival_rate": rate(slow_brush),
            "timeout_rate": rate(lambda r: r["outcome"] == "timeout"),
            "hard_collision_rate": rate(lambda r: r["outcome"] == "hard_collision"),
            "fall_rate": rate(lambda r: r["outcome"] == "fall"),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--budget-margin-s", type=float, default=4.0)
    parser.add_argument("--max-episode-s", type=float, default=40.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/classroom_comparison")
    parser.add_argument("--pac-checkpoint", type=Path, default=DEFAULT_PAC)
    parser.add_argument("--lp-backbone", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--lp-planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--pac-python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--lp-python", type=Path, default=LP_ROOT / ".venv-stage67/bin/python")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", choices=("both", "pac", "lp"), default="both")
    parser.add_argument("--routes-only", action="store_true")
    parser.add_argument("--slow-brush-definition", choices=("late", "on-time", "any"), default="late")
    parser.add_argument("--worker", choices=("pac", "lp"), help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.episodes < 1 or args.batch_size < 1 or args.budget_margin_s < 0 or args.max_episode_s <= 0:
        parser.error("episodes, batch-size and max-episode-s must be positive; margin must be nonnegative")
    if args.worker and (args.manifest is None or args.rows is None):
        parser.error("worker requires --manifest and --rows")
    return args


def run_parent(args: argparse.Namespace) -> None:
    for key in ("output_dir", "pac_checkpoint", "lp_backbone", "lp_planner"):
        setattr(args, key, getattr(args, key).resolve())
    # A venv's bin/python is a symlink; resolving it would escape the venv.
    for key in ("pac_python", "lp_python"):
        setattr(args, key, Path(os.path.abspath(getattr(args, key))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routes = make_routes(args.episodes, args.seed, args.budget_margin_s)
    manifest = args.output_dir / "routes.json"
    manifest.write_text(json.dumps({"seed": args.seed, "speed_m_s": SPEED,
                                    "budget_margin_s": args.budget_margin_s,
                                    "routes": routes}, indent=2) + "\n")
    print(f"Shared routes: {manifest} ({len(routes)} cases)", flush=True)
    if args.routes_only:
        return
    selected = ("pac", "lp") if args.only == "both" else (args.only,)
    for name in selected:
        interpreter = args.pac_python if name == "pac" else args.lp_python
        checkpoints = [args.pac_checkpoint] if name == "pac" else [args.lp_backbone, args.lp_planner]
        for path in (interpreter, *checkpoints):
            if not path.is_file():
                raise FileNotFoundError(f"{name} input missing: {path}")
        rows_path = args.output_dir / f"{name}_episodes.json"
        command = [str(interpreter), str(Path(__file__).resolve()), "--worker", name,
                   "--manifest", str(manifest), "--rows", str(rows_path),
                   "--batch-size", str(args.batch_size), "--max-episode-s", str(args.max_episode_s),
                   "--device", args.device, "--pac-checkpoint", str(args.pac_checkpoint),
                   "--lp-backbone", str(args.lp_backbone), "--lp-planner", str(args.lp_planner)]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT if name == "pac" else LP_ROOT / "src")
        environment["WALK_THROW_ENABLED"] = "0"
        environment["WALK_V_REF"] = str(SPEED)
        print(f"Running {name} in {interpreter}", flush=True)
        subprocess.run(command, cwd=ROOT if name == "pac" else LP_ROOT, env=environment, check=True)
    rows = []
    for name in selected:
        model_rows = json.loads((args.output_dir / f"{name}_episodes.json").read_text())
        by_id = {row["case_id"]: row for row in model_rows}
        if len(by_id) != len(routes) or set(by_id) != set(range(len(routes))):
            raise RuntimeError(f"{name} did not return exactly one result per shared route")
        for route in routes:
            row = by_id[route["case_id"]]
            if (row["model"] != name or row["turn"] != route["turn"] or
                row["L_obj_m"] != route["L_obj_m"] or row["T_obj_s"] != route["T_obj_s"]):
                raise RuntimeError(f"{name} result differs from shared route {route['case_id']}")
        rows.extend(model_rows)
    summary = aggregate(rows, args.slow_brush_definition)
    report = {"inputs": {
        "seed": args.seed, "episodes_per_model": len(routes),
        "max_episode_s": args.max_episode_s, "speed_m_s": SPEED,
        "pac_checkpoint": str(args.pac_checkpoint),
        "lp_backbone": str(args.lp_backbone),
        "lp_planner": str(args.lp_planner),
    }, "definitions": {
        "success": "reached junction then final goal before max episode time, without hard collision",
        "on_time": "successful arrival at t <= T_obj",
        "L_obj": "raw shared route polyline length",
        "T_obj": f"L_obj / {SPEED} + {args.budget_margin_s} seconds",
        "any_brush": "any robot/scenery contact with distance <= 0",
        "hard_collision": f"robot/scenery contact penetration >= {HARD_PENETRATION_M} m",
        "slow_brush_arrival": args.slow_brush_definition,
        "timeout": "no arrival or collision before max episode time",
        "rates_denominator": "all cases for each model",
    }, "summary": summary}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, values in summary.items():
        print(f"{name}: " + " ".join(f"{k}={v:.3f}" for k, v in values.items() if k != "episodes"))
    print(f"Saved: {args.output_dir / 'summary.json'}", flush=True)


def _contact_reader(env, model_name: str):
    """Return physical robot/scenery contact penetration for each world."""
    import mujoco
    import torch

    model = env.sim.mj_model
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
             for i in range(model.ngeom)]
    robot = torch.tensor([name.startswith("robot/") for name in names], device=env.device)
    if model_name == "pac":
        obstacle = torch.tensor([name.endswith("/wall_collision") for name in names], device=env.device)
    else:
        terrain_bodies = {i for i in range(model.nbody)
                          if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("terrain")}
        obstacle = torch.tensor([int(model.geom_bodyid[i]) in terrain_bodies and model.geom_size[i, 2] > 0.05
                                 for i in range(model.ngeom)], device=env.device)
    if not robot.any() or not obstacle.any():
        raise RuntimeError(f"Could not identify robot/scenery geoms: robot={robot.sum()}, scenery={obstacle.sum()}")

    def read():
        data = env.sim.data
        count = int(data.nacon[0].item())
        distance = torch.full((env.num_envs,), float("inf"), device=env.device)
        if count == 0:
            return distance
        pair = data.contact.geom[:count].long()
        world = data.contact.worldid[:count].long()
        dist = data.contact.dist[:count]
        valid = ((robot[pair[:, 0]] & obstacle[pair[:, 1]]) |
                 (obstacle[pair[:, 0]] & robot[pair[:, 1]])) & (world >= 0) & (world < env.num_envs)
        if valid.any():
            distance.scatter_reduce_(0, world[valid], dist[valid], reduce="amin", include_self=True)
        return distance

    return read


def _install_reset_capture(env, read_contact):
    """Keep terminal contact/pose before mjlab auto-resets an environment."""
    captured = {}
    original = env._reset_idx

    def capture(env_ids):
        if env_ids is not None and len(env_ids) and capture.enabled:
            distances = read_contact().detach().cpu().tolist()
            positions = env.scene["robot"].data.root_link_pos_w.detach().cpu().tolist()
            terminated = env.termination_manager.terminated.detach().cpu().tolist()
            terms = {name: env.termination_manager.get_term(name).detach().cpu().tolist()
                     for name in env.termination_manager.active_terms}
            for index in env_ids.detach().cpu().tolist():
                triggered = [name for name, flags in terms.items() if flags[index]]
                captured[index] = (distances[index], positions[index], bool(terminated[index]), triggered)
        return original(env_ids)

    capture.enabled = False
    env._reset_idx = capture
    return captured, capture


def _place_robot(env, xy, yaw):
    import torch
    robot = env.scene["robot"]
    ids = torch.arange(env.num_envs, device=env.device)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :2] = xy
    pose[:, 3:] = 0.0
    pose[:, 3] = torch.cos(yaw / 2)
    pose[:, 6] = torch.sin(yaw / 2)
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()


def _lp_world_points(env, routes):
    """Map PAC routes into each LP cell, preserving left/right via room symmetry."""
    import mujoco
    import torch
    model = env.sim.mj_model
    floors = [i for i in range(model.ngeom)
              if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
              and abs(model.geom_size[i, 0] - 6.0) < 1e-3
              and abs(model.geom_size[i, 1] - 10.0) < 1e-3
              and abs(model.geom_size[i, 2] - 0.02) < 1e-3]
    if not floors:
        raise RuntimeError("LP classroom terrain floor was not found")
    centers = env.sim.data.geom_xpos[0, floors, :2]
    origins = env.scene.env_origins[:, :2]
    center = centers[torch.cdist(origins, centers).argmin(dim=1)]
    if torch.any((origins - center).abs() >= torch.tensor((6.0, 10.0), device=env.device)):
        raise RuntimeError("LP spawn origin did not map to a classroom terrain cell")
    pac = torch.tensor([r["points_pac_xy_m"] for r in routes], device=env.device)
    out = torch.empty_like(pac)
    # Both classrooms are symmetric about lateral -1.35 m. Reflecting there
    # compensates for the axis swap, so PAC left remains LP left at +Y heading.
    out[..., 0] = center[:, None, 0] - pac[..., 1] - 1.35
    out[..., 1] = center[:, None, 1] + pac[..., 0] - 7.75
    return out


def _prepare_pac(env, routes):
    import torch
    from src.tasks.amp_loco.mdp.classroom import _write_classroom_route
    points = torch.tensor([r["points_pac_xy_m"] for r in routes], device=env.device)
    _place_robot(env, points[:, 0], torch.zeros(env.num_envs, device=env.device))
    ids = torch.arange(env.num_envs, device=env.device)
    _write_classroom_route(env, ids, points)
    env._walk_path_epoch[ids] += 1
    if not torch.allclose(env._classroom_route_pts[:, :3], points, atol=1e-4):
        raise RuntimeError("PAC route injection failed")
    return points


def _prepare_lp(env, routes):
    import torch
    points = _lp_world_points(env, routes)
    _place_robot(env, points[:, 0], torch.full((env.num_envs,), math.pi / 2, device=env.device))
    ids = torch.arange(env.num_envs, device=env.device)
    term = env.command_manager.get_term("goal")
    if term.num_goals != 2:
        raise RuntimeError(f"Expected two NoPP key goals, found {term.num_goals}")
    term._spawn_pos_w[:] = points[:, 0]
    term._env_goal_offsets[:] = points[:, 1:] - points[:, :1]
    term.current_goal_idx[:] = 0
    term.current_key_goal_idx[:] = 0
    term.goal_reached[:] = False
    term.sequence_complete[:] = False
    term._base_target_velocity_cmd[:] = SPEED
    term._target_velocity_cmd[:] = SPEED
    term.time_left[:] = 1e6
    term._update_goal_position(ids)
    term._update_key_goal_position(ids)
    if not torch.allclose(term.goal_pos_w[:, :2], points[:, 1], atol=1e-4):
        raise RuntimeError("LP waypoint injection failed")
    return points


def _fresh_observation(env):
    env.command_manager.compute(dt=0.0)
    env.sim.sense()
    return env.observation_manager.compute(update_history=True)


def _make_pac_backend(args, n):
    import torch
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
    from src.tasks.amp_loco.config.g1.goto_rl_cfg import g1_amp_dodge_mimickit_classroom_ppo_runner_cfg
    from src.tasks.amp_loco.rl import AMPOnPolicyRunner
    import src.tasks.amp_loco.mdp.classroom as classroom

    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=True)
    cfg.scene.num_envs = n
    cfg.seed = 42
    cfg.auto_reset = True
    cfg.episode_length_s = args.max_episode_s
    cfg.terminations.pop("wall_hit", None)  # common hard-contact rule below
    classroom.extend_classroom_route = lambda env, env_ids, min_total_len: None
    env = ManagerBasedRlEnv(cfg, device=args.device)
    runner_cfg = asdict(g1_amp_dodge_mimickit_classroom_ppo_runner_cfg())
    wrapper = RslRlVecEnvWrapper(env, clip_actions=runner_cfg["clip_actions"])
    runner = AMPOnPolicyRunner(wrapper, runner_cfg,
                               log_dir=None, device=args.device)
    runner.load(str(args.pac_checkpoint.resolve()))
    policy = runner.get_inference_policy(device=args.device)
    return env, policy, lambda obs, action: wrapper.step(action)


def _make_lp_backend(args, n):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.tasks.steering.train_local_planner import load_policy
    from mjlab.tool.experiments.eval_utils import load_local_planner_inference, make_local_planner_policy

    task = "Mjlab-Classroom-NoPP-G1-v6"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = n
    cfg.seed = 42
    cfg.auto_reset = True
    cfg.episode_length_s = args.max_episode_s
    cfg.terminations.pop("goal_sequence_complete", None)  # shared arrival radius below
    env = ManagerBasedRlEnv(cfg, device=args.device)
    base_policy = load_policy(str(args.lp_backbone.resolve()), env, load_rl_cfg(task), torch.device(args.device))
    inference = load_local_planner_inference(
        checkpoint=str(args.lp_planner.resolve()),
        raycast_obs_names=("raycast", "raycast_left", "raycast_right"),
        device=torch.device(args.device), num_envs=n, task=task,
        locomotion_checkpoint=str(args.lp_backbone.resolve()))
    correction = torch.zeros(n, device=env.device)
    policy = make_local_planner_policy(base_policy=base_policy, local_planner_inference=inference,
                                       env=env, correction_gain=1.0, last_corrections=correction)
    return env, policy, lambda obs, action: env.step(action), inference


def run_worker(args: argparse.Namespace) -> None:
    import torch
    torch.manual_seed(42)
    routes = json.loads(args.manifest.read_text())["routes"]
    rows = []
    for start in range(0, len(routes), args.batch_size):
        batch = routes[start:start + args.batch_size]
        n = len(batch)
        if args.worker == "pac":
            env, policy, step = _make_pac_backend(args, n)
            inference = None
        else:
            env, policy, step, inference = _make_lp_backend(args, n)
        try:
            read_contact = _contact_reader(env, args.worker)
            captured, capture = _install_reset_capture(env, read_contact)
            env.reset()
            capture.enabled = True
            points = _prepare_pac(env, batch) if args.worker == "pac" else _prepare_lp(env, batch)
            if inference is not None:
                inference.obs_builder.update_initial_goal(env, torch.ones(n, dtype=torch.bool, device=env.device))
            obs = _fresh_observation(env)
            active = [True] * n
            via = [False] * n
            brush = [False] * n
            min_contact = [float("inf")] * n
            steps = [0] * n
            max_steps = math.ceil(args.max_episode_s / env.step_dt)
            for _ in range(max_steps):
                if not any(active):
                    break
                captured.clear()
                with torch.inference_mode():
                    action = policy(obs)
                    result = step(obs, action)
                    if len(result) == 5:
                        obs, _, terminated, timed_out, _ = result
                        dones = terminated | timed_out
                    else:
                        obs, _, dones, _ = result
                distances = read_contact().detach().cpu().tolist()
                positions = env.scene["robot"].data.root_link_pos_w.detach().cpu().tolist()
                done_flags = dones.detach().cpu().tolist()
                for i in range(n):
                    if not active[i]:
                        continue
                    steps[i] += 1
                    native_terminated = False
                    native_terms = []
                    if done_flags[i] and i in captured:
                        distance, position, native_terminated, native_terms = captured[i]
                    else:
                        distance, position = distances[i], positions[i]
                    min_contact[i] = min(min_contact[i], distance)
                    brush[i] |= distance <= 0.0
                    hard = distance <= -HARD_PENETRATION_M
                    if math.dist(position[:2], points[i, 1].tolist()) <= ARRIVAL_RADIUS_M:
                        via[i] = True
                    arrived = via[i] and math.dist(position[:2], points[i, 2].tolist()) <= ARRIVAL_RADIUS_M
                    elapsed = steps[i] * env.step_dt
                    if not (hard or arrived or done_flags[i] or steps[i] >= max_steps):
                        continue
                    outcome = ("hard_collision" if hard else "fall" if native_terminated else
                               "arrived" if arrived else "timeout")
                    on_time = outcome == "arrived" and elapsed <= batch[i]["T_obj_s"]
                    rows.append({
                        "model": args.worker, "case_id": batch[i]["case_id"],
                        "turn": batch[i]["turn"], "L_obj_m": batch[i]["L_obj_m"],
                        "T_obj_s": batch[i]["T_obj_s"], "episode_duration_s": elapsed,
                        "outcome": outcome, "arrived": outcome == "arrived",
                        "on_time": on_time, "any_brush": brush[i],
                        "max_penetration_m": max(0.0, -min_contact[i]),
                        "native_termination": ",".join(native_terms),
                        "terminal_root_height_m": position[2],
                        "passed_junction": via[i],
                        "terminal_xy_m": position[:2],
                    })
                    active[i] = False
            if any(active):
                raise RuntimeError(f"Worker failed to finish {sum(active)} cases")
            print(f"{args.worker}: {start + n}/{len(routes)} cases", flush=True)
        finally:
            env.close()
    args.rows.write_text(json.dumps(sorted(rows, key=lambda r: r["case_id"]), indent=2) + "\n")


if __name__ == "__main__":
    cli = parse_args()
    if cli.worker:
        run_worker(cli)
    else:
        run_parent(cli)
