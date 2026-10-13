"""Record prescribed CPU navigation traces and optional untrained GPU steps.

Run from the repo root:
  .venv/bin/python scripts/validate_classroom_navigation.py --gpu-steps 20

CPU traces prescribe pose/COM velocity to validate reward logic. GPU steps use
zero joint actions and validate integration, not a learned navigation policy.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
from src.tasks.amp_loco.mdp.classroom import (
    round_classroom_route, generate_classroom_route, _write_classroom_route,
)
from src.tasks.amp_loco.mdp.goal_command import DodgeGoToGoalCommand
from src.tasks.amp_loco.mdp.path_navigation import (
    classroom_follow_state, classroom_path_follow_reward, classroom_navigation_observation,
    classroom_safe_completion_reward,
)
from src.tasks.amp_loco.mdp.walk_path import walk_path_query


class Scene(dict):
    pass


def make_command(points, xy=(0., 0.), yaw=0.):
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    command_cfg = cfg.commands['twist']
    command_cfg.cbf_enabled = False
    command_cfg.navigation_obstacles = ()  # geometry tested in actual GPU world
    data = NS(root_link_pos_w=torch.tensor([[*xy, .8]]), heading_w=torch.tensor([yaw]),
              root_link_quat_w=torch.tensor([[math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]]),
              root_link_ang_vel_w=torch.zeros(1, 3), root_com_lin_vel_w=torch.zeros(1, 3))
    scene = Scene(robot=NS(data=data), ball=NS(data=NS(root_link_pos_w=torch.tensor([[0., 1000., .8]]))))
    scene.env_origins = torch.zeros(1, 3)
    route = torch.tensor([points], dtype=torch.float32)
    env = NS(scene=scene, num_envs=1, device='cpu', common_step_counter=0,
        step_dt=cfg.sim.mujoco.timestep * cfg.decimation,
        sim=NS(mj_model=NS(geom=lambda _: NS(id=0))),
        _walk_path_pts=route,
        _walk_path_cumlen=torch.cat((torch.zeros(1, 1), (route[:, 1:] - route[:, :-1]).norm(dim=-1).cumsum(-1)), -1),
        _walk_path_n=torch.tensor([len(points)]), _walk_path_epoch=torch.ones(1, dtype=torch.long))
    command = DodgeGoToGoalCommand(command_cfg, env)
    env.command_manager = NS(get_term=lambda _: command)
    command._update_command()
    return command, env, cfg.rewards['classroom_path_follow'].params


def update_pose(command, env, xy, yaw):
    data = command.robot.data
    before = data.root_link_pos_w[0, :2].clone()
    data.root_link_pos_w[0, :2] = torch.tensor(xy)
    data.root_com_lin_vel_w[0, :2] = (data.root_link_pos_w[0, :2] - before) / env.step_dt
    data.heading_w[0] = yaw
    data.root_link_quat_w[0] = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])
    env.common_step_counter += 1


def record(command, env, params, name, rows, source='prescribed_cpu'):
    total = env.step_dt * classroom_path_follow_reward(env, **params)
    state = classroom_follow_state(command)
    observation = classroom_navigation_observation(env)
    assert torch.isfinite(total).all() and torch.isfinite(observation).all()
    completion = (env.step_dt * classroom_safe_completion_reward(env)
                  if hasattr(env, 'termination_manager') else torch.zeros_like(total))
    for index in range(env.num_envs):
        row = dict(source=source, scenario=name, env=index, step=env.common_step_counter,
            dt=env.step_dt, phase=2 if state.aligning[index] else int(state.joining[index]),
            executed_phase=2 if state.step_aligning[index] else int(state.step_joining[index]),
            theta_rad=state.theta[index].item(), step_theta_rad=state.step_theta[index].item(),
            path_distance_m=state.distance[index].item(), s_m=command._path_s[index].item(),
            v_s_m_s=state.speed[index].item(), v_cmd_m_s=state.command_speed[index].item(),
            curvature=state.curvature[index].item(), recovery_age_s=state.recovery_age[index].item(),
            target_x=command.goal_pos_w[index, 0].item(), target_y=command.goal_pos_w[index, 1].item(),
            total_task_navigation=total[index].item())
        for key, value in state.reward_components.items():
            assert torch.isfinite(value).all()
            row['reward_' + key] = value[index].item()
        row['reward_safe_completion'] = completion[index].item()
        rows.append(row)


def cpu_traces(rows):
    straight = [(0., 0.), (10., 0.)]
    for reverse in (False, True):
        yaw = math.atan2(-.5, .8) + (math.pi if reverse else 0.)
        command, env, params = make_command(straight, (0., .5), yaw)
        name = 'off_path_reverse' if reverse else 'off_path_aligned'
        for _ in range(420):
            state = classroom_follow_state(command)
            yaw += max(-.5 * env.step_dt, min(.5 * env.step_dt, state.theta.item()))
            speed = 0. if state.aligning else min(.3, state.command_speed.item())
            xy = state.xy[0].tolist()
            xy[0] += env.step_dt * speed * math.cos(yaw)
            xy[1] += env.step_dt * speed * math.sin(yaw)
            update_pose(command, env, xy, yaw)
            record(command, env, params, name, rows)
            command._update_command()
    for degrees in (90., 180.):
        command, env, params = make_command(straight, yaw=math.radians(degrees))
        yaw = math.radians(degrees)
        for _ in range(400):
            state = classroom_follow_state(command)
            yaw += max(-.5 * env.step_dt, min(.5 * env.step_dt, state.theta.item()))
            xy = state.xy[0].tolist()
            speed = 0. if state.aligning else min(.3, state.command_speed.item())
            xy[0] += env.step_dt * speed * math.cos(yaw)
            xy[1] += env.step_dt * speed * math.sin(yaw)
            update_pose(command, env, xy, yaw)
            record(command, env, params, f'on_path_{int(degrees)}deg', rows)
    for name, vx, vy in [('straight', .3, 0.), ('retreat', -.3, 0.), ('sideways', 0., .3), ('still', 0., 0.)]:
        command, env, params = make_command(straight)
        for i in range(120):
            update_pose(command, env, ((i + 1) * vx * env.step_dt, (i + 1) * vy * env.step_dt), 0.)
            record(command, env, params, name, rows)
    for name, radius in [('bend', .6), ('sharp_bend', .2)]:
        route, count = round_classroom_route(torch.tensor([[[0., 0.], [3., 0.], [3., 3.]]]), radius=radius)
        command, env, params = make_command(route[0, :count[0]].tolist())
        for i in range(800):
            station = torch.tensor([(i + 1) * .3 * env.step_dt])
            point, yaw = walk_path_query(env, torch.tensor([0]), station)
            update_pose(command, env, point[0].tolist(), yaw.item())
            record(command, env, params, name, rows)
    command, env, params = make_command(straight)
    for cycle in range(8):
        for i in range(30):
            update_pose(command, env, ((i + 1) * .006, 0.), 0.)
            record(command, env, params, 'round_trip', rows)
        update_pose(command, env, (0., 0.), 0.)
        record(command, env, params, 'round_trip', rows)
    command, env, params = make_command(straight, yaw=math.radians(120.))
    for i in range(600):
        update_pose(command, env, (0., 0.), math.radians(110. if i % 2 == 0 else 120.))
        record(command, env, params, 'repeated_turn', rows)
    command, env, params = make_command(straight)
    for i in range(600):
        update_pose(command, env, (0., .5 if i % 2 == 0 else .19), 0.)
        record(command, env, params, 'phase_cycle', rows)
    env._walk_path_epoch += 1
    update_pose(command, env, (0., .5), math.pi)
    record(command, env, params, 'reset', rows)
    state = classroom_follow_state(command)
    assert state.turn_delta == 0 and state.join_delta == 0 and state.speed == 0
    command, env, params = make_command([(0., 0.), (5., 0.), (5., 4.), (0., 4.), (0., 0.), (5., 0.)])
    for i in range(120):
        update_pose(command, env, ((i + 1) * .006, 0.), 0.)
        record(command, env, params, 'intersection', rows)
        assert command._path_seg_idx == 0

    # Cross the original route tail, keeping the same episode and station.
    torch.manual_seed(23)
    command, env, params = make_command(straight)
    env._classroom_route_extension_nodes = 3
    raw = generate_classroom_route(torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0]), count=3)
    _write_classroom_route(env, torch.tensor([0]), raw)
    original_length = env._walk_path_cumlen[0, env._walk_path_n[0] - 1].item()
    env._walk_path_epoch += 1
    command.cfg.path_wall_ahead = 0.
    command._update_command()
    epoch = env._walk_path_epoch.clone()
    for i in range(math.ceil((original_length + 2.) / (.3 * env.step_dt))):
        station = torch.tensor([(i + 1) * .3 * env.step_dt])
        point, yaw = walk_path_query(env, torch.tensor([0]), station)
        update_pose(command, env, point[0].tolist(), yaw.item())
        record(command, env, params, 'continuous_extension', rows)
        assert abs(command._path_s.item() - station.item()) < 1e-4
        assert torch.equal(env._walk_path_epoch, epoch)
    assert command._path_s.item() > original_length
    assert env._classroom_route_n.item() > 3

    command, env, params = make_command(straight)
    flags = dict(timeout=torch.tensor([False]), failed=torch.tensor([False]))
    env.termination_manager = NS(get_term=lambda _: flags['timeout'], terminated=flags['failed'])
    for timeout, failed in ((False, False), (True, False), (True, True)):
        flags['timeout'][:], flags['failed'][:] = timeout, failed
        update_pose(command, env, (0., 0.), 0.)
        record(command, env, params, 'safe_timeout', rows)


def gpu_steps(rows, steps):
    if steps == 0:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('--gpu-steps requires CUDA')
    from mjlab.envs import ManagerBasedRlEnv
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    cfg.scene.num_envs = 4
    # Capture before automatic reset; explicit reset follows recording.
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda')
    try:
        env.reset()
        command = env.command_manager.get_term('twist')
        for _ in range(steps):
            obs, values, terminated, truncated, _ = env.step(torch.zeros(4, env.action_manager.total_action_dim, device='cuda'))
            assert torch.isfinite(values).all() and torch.isfinite(obs['path_navigation']).all()
            record(command, env, cfg.rewards['classroom_path_follow'].params, 'zero_actions', rows, 'untrained_gpu')
            done = (terminated | truncated).nonzero().flatten()
            if len(done):
                env.reset(env_ids=done)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('logs/validation/classroom_navigation'))
    parser.add_argument('--gpu-steps', type=int, default=0)
    args = parser.parse_args()
    rows = []
    cpu_traces(rows)
    gpu_steps(rows, args.gpu_steps)
    summary = {}
    for row in rows:
        item = summary.setdefault(row['scenario'], dict(steps=0, total=0., components={}, phases={}))
        item['steps'] += 1
        item['total'] += row['total_task_navigation']
        phase = str(row['phase'])
        item['phases'][phase] = item['phases'].get(phase, 0) + 1
        for key, value in row.items():
            if key.startswith('reward_'):
                item['components'][key] = item['components'].get(key, 0.) + value
    assert summary['straight']['components']['reward_forward'] > 0
    assert summary['retreat']['components']['reward_retreat'] < 0
    assert summary['off_path_aligned']['components']['reward_join'] > 0
    assert summary['still']['total'] < 0
    assert summary['round_trip']['total'] <= 1e-5
    assert summary['repeated_turn']['components']['reward_turn'] < 0
    assert summary['repeated_turn']['components']['reward_recovery_stall'] < 0
    assert summary['phase_cycle']['total'] < 0
    assert summary['phase_cycle']['components']['reward_join'] < 0
    assert summary['continuous_extension']['components']['reward_forward'] > 0
    assert summary['safe_timeout']['components']['reward_safe_completion'] == 10.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'traces.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(dt=.02, finite=True, interpretation='Prescribed poses and untrained zero actions; no learned-policy claim.', scenarios=summary)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), scenarios=len(summary), finite=True, output=str(args.output_dir))))


if __name__ == '__main__':
    main()
