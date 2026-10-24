#!/usr/bin/env python3
"""Replay one PAC classroom comparison case in the interactive Viser viewer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import ViserPlayViewer
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
from src.tasks.amp_loco.config.g1.goto_rl_cfg import g1_amp_dodge_mimickit_classroom_ppo_runner_cfg
from src.tasks.amp_loco.rl import AMPOnPolicyRunner
import src.tasks.amp_loco.mdp.classroom as classroom

from others.compare_classroom import _prepare_pac


class PacCaseViewer(ViserPlayViewer):
    def __init__(self, env, policy, route):
        super().__init__(env, policy)
        self.route = route

    def setup(self):
        super().setup()
        with self._server.gui.add_folder("PAC comparison case"):
            self._case_info = self._server.gui.add_html(
                f"Case {self.route['case_id']} · {self.route['turn']} turn · "
                f"L_obj {self.route['L_obj_m']:.2f} m · T_obj {self.route['T_obj_s']:.2f} s"
            )
            self._case_telemetry = self._server.gui.add_html("")

    def sync_env_to_viewer(self):
        super().sync_env_to_viewer()
        if self._counter % 10:
            return
        env = self.env.unwrapped
        robot = env.scene["robot"]
        command = env.command_manager.get_term("twist")
        xy = robot.data.root_link_pos_w[0, :2].detach().cpu().tolist()
        speed = robot.data.root_link_lin_vel_w[0, :2].detach().cpu().tolist()
        cmd = command.vel_command_b[0].detach().cpu().tolist()
        junction = self.route["points_pac_xy_m"][1]
        goal = self.route["points_pac_xy_m"][2]
        distance = lambda p: ((xy[0] - p[0]) ** 2 + (xy[1] - p[1]) ** 2) ** 0.5
        elapsed = float(env.episode_length_buf[0]) * env.step_dt
        self._case_telemetry.content = (
            f"t={elapsed:.1f} s · xy=({xy[0]:.2f}, {xy[1]:.2f}) m<br>"
            f"junction={distance(junction):.2f} m · final={distance(goal):.2f} m<br>"
            f"command body=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}) · "
            f"actual world=({speed[0]:+.2f}, {speed[1]:+.2f}) m/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--max-episode-s", type=float, default=40.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_episode_s <= 0:
        parser.error("--max-episode-s must be positive")
    routes = json.loads(args.manifest.read_text())["routes"]
    matches = [route for route in routes if route["case_id"] == args.case_id]
    if len(matches) != 1:
        parser.error(f"case_id {args.case_id} must occur exactly once in {args.manifest}")
    route = matches[0]
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    # Match the PAC comparison worker: seed, one classroom cell, 40 s episodes,
    # native training action clipping (None), no path extension or wall-hit end.
    torch.manual_seed(42)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg(play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.auto_reset = True
    cfg.episode_length_s = args.max_episode_s
    cfg.terminations.pop("wall_hit", None)
    cfg.commands["twist"].debug_vis = True
    classroom.extend_classroom_route = lambda env, env_ids, min_total_len: None
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        runner_cfg = asdict(g1_amp_dodge_mimickit_classroom_ppo_runner_cfg())
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg["clip_actions"])
        runner = AMPOnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args.device)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=args.device)

        # The viewer reset button and the environment's automatic timeout reset
        # both call _reset_idx. Install the route after native reset events so
        # every replay starts at the same point and faces the same first aisle.
        native_reset = env._reset_idx

        def reset_fixed_case(env_ids):
            native_reset(env_ids)
            _prepare_pac(env, [route])

        env._reset_idx = reset_fixed_case
        wrapped.reset()
        print(f"[PAC viewer] Replaying case {args.case_id}; press Reset to restart it.", flush=True)
        PacCaseViewer(wrapped, policy, route).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
