"""Profile classroom rollouts without learning or changing training checkpoints.

Example: CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/profile_classroom_sampling.py
  --checkpoint logs/rsl_rl/state_classroom/RUN/model_3000.pt --output .cache/sampling.json
Wall-clock benchmarks run separately from the optional, intrusive torch profiler.
"""
import argparse
import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import pstats
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.utils import check_nan
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
from src.tasks.amp_loco.config.g1.goto_rl_cfg import g1_amp_dodge_mimickit_classroom_ppo_runner_cfg
from src.tasks.amp_loco.rl import AMPOnPolicyRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=512)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--warmup', type=int, default=48)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=Path('.cache/classroom_sampling.json'))
    parser.add_argument('--torch-profile-steps', type=int, default=0)
    parser.add_argument('--no-nan-check', action='store_true')
    parser.add_argument('--no-episode-log', action='store_true')
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.warmup < 0:
        parser.error('num-envs/steps must be positive and warmup must be nonnegative')
    configure_torch_backends()
    torch.cuda.set_device(0)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
    try:
        wrapper = RslRlVecEnvWrapper(env)
        runner = AMPOnPolicyRunner(wrapper, asdict(g1_amp_dodge_mimickit_classroom_ppo_runner_cfg()),
                                   log_dir=None, device='cuda:0')
        if args.checkpoint:
            runner.load(str(args.checkpoint.resolve()))
        runner.alg.train_mode()
        # Exercise rank-0 per-step bookkeeping without opening writers or W&B.
        runner.logger.writer = None if args.no_episode_log else object()
        obs = wrapper.get_observations()
        terminated_count = torch.zeros((), device=wrapper.device, dtype=torch.int64)

        def sample():
            nonlocal obs
            if runner.alg.storage.step == runner.cfg['num_steps_per_env']:
                runner.alg.storage.clear()
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = wrapper.step(actions)
            if not args.no_nan_check:
                check_nan(obs, rewards, dones)
            runner.alg.process_env_step(obs, rewards, dones, extras)
            runner.logger.process_env_step(rewards, dones, extras)
            terminated_count.add_(dones.sum())

        def run_steps(count):
            with torch.inference_mode():
                for _ in range(count):
                    sample()

        run_steps(args.warmup)
        terminated_count.zero_()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()
        run_steps(args.steps)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - elapsed
        summary = {
            'num_envs': args.num_envs, 'steps': args.steps, 'warmup': args.warmup,
            'checkpoint': str(args.checkpoint) if args.checkpoint else None,
            'gpu': torch.cuda.get_device_name(), 'video': False,
            'nan_check': not args.no_nan_check, 'episode_log': not args.no_episode_log,
            'seconds': elapsed, 'ms_per_step': elapsed * 1000 / args.steps,
            'env_steps_per_second': args.num_envs * args.steps / elapsed,
            'episode_ends': terminated_count.item(),
        }
        print('BENCHMARK ' + json.dumps(summary), flush=True)

        # Keep tracing overhead out of the wall-clock result above.
        profiler = cProfile.Profile()
        profiler.enable()
        run_steps(min(args.steps, 96))
        torch.cuda.synchronize()
        profiler.disable()
        stats = pstats.Stats(profiler)
        summary['python_profile'] = [
            {'file': filename, 'line': line, 'function': function, 'calls': calls,
             'self_seconds': self_time, 'cumulative_seconds': cumulative}
            for (filename, line, function), (_, calls, self_time, cumulative, _) in
            sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:100]
        ]
        stats.strip_dirs().sort_stats('cumulative').print_stats(35)

        if args.torch_profile_steps:
            originals = []
            for owner, methods in (
                (env.sim, ('step', 'forward', 'sense')),
                (env.scene, ('update', 'write_data_to_sim')),
                (env, ('_reset_idx',)),
                (env.termination_manager, ('compute',)),
                (env.reward_manager, ('compute',)),
                (env.metrics_manager, ('compute', 'compute_substep')),
                (env.command_manager, ('compute',)),
                (env.event_manager, ('apply',)),
                (env.observation_manager, ('compute',)),
            ):
                for name in methods:
                    original = getattr(owner, name)
                    label = f'{type(owner).__name__}.{name}'
                    def recorded(*call_args, _fn=original, _label=label, **kwargs):
                        with torch.profiler.record_function(_label):
                            return _fn(*call_args, **kwargs)
                    originals.append((owner, name, original))
                    setattr(owner, name, recorded)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                         torch.profiler.ProfilerActivity.CUDA]) as trace:
                    run_steps(args.torch_profile_steps)
                    torch.cuda.synchronize()
                print(trace.key_averages().table(sort_by='self_cpu_time_total', row_limit=25))
                summary['torch_profile'] = [
                    {'name': event.key, 'calls': event.count,
                     'cpu_ms': event.cpu_time_total / 1000,
                     'self_cpu_ms': event.self_cpu_time_total / 1000,
                     'device_ms': event.device_time_total / 1000}
                    for event in trace.key_averages()
                ]
            finally:
                for owner, name, original in originals:
                    setattr(owner, name, original)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + '\n')
        print(f'Profile saved to {args.output}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
