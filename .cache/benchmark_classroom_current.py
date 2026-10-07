import json
import sys
import time
from dataclasses import asdict
from statistics import mean
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_classroom_flat_env_cfg
from src.tasks.amp_loco.config.g1.goto_rl_cfg import g1_amp_dodge_mimickit_classroom_ppo_runner_cfg
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

configure_torch_backends()
n = int(sys.argv[1])
cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
cfg.scene.num_envs = n
cfg.seed = 42
agent = g1_amp_dodge_mimickit_classroom_ppo_runner_cfg()
agent.logger = 'tensorboard'
start = time.perf_counter()
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    env_ready = time.perf_counter()
    runner = AMPOnPolicyRunner(wrapped, asdict(agent), log_dir=None, device='cuda:0')
    torch.cuda.synchronize()
    print('STARTUP',json.dumps({'env_s':env_ready-start,'runner_s':time.perf_counter()-env_ready}),flush=True)
    wrapped.episode_length_buf = torch.randint_like(wrapped.episode_length_buf, high=wrapped.max_episode_length)
    obs = wrapped.get_observations()
    runner.alg.train_mode()
    rows = []
    for i in range(8):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(agent.num_steps_per_env):
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(actions)
                runner.alg.process_env_step(obs, rewards, dones, extras)
            torch.cuda.synchronize()
            collected = time.perf_counter()
            runner.alg.compute_returns(obs)
        losses = runner.alg.update()
        torch.cuda.synchronize()
        end = time.perf_counter()
        row={'iteration':i,'collection_s':collected-start,'learning_s':end-collected,'total_s':end-start}
        rows.append(row)
        print('ITER',json.dumps(row),flush=True)
    steady = rows[3:]
    total = mean(x['total_s'] for x in steady)
    print('RESULT',json.dumps({'num_envs':n,'steps_per_env':agent.num_steps_per_env,
        'collection_s':mean(x['collection_s'] for x in steady),
        'learning_s':mean(x['learning_s'] for x in steady),'iteration_s':total,
        'samples_per_second':n*agent.num_steps_per_env/total,
        'hours_25000_iters':25000*total/3600,
        'hours_614400000_samples':614400000/(n*agent.num_steps_per_env/total)/3600,
        'gpu_peak_gb':torch.cuda.max_memory_allocated()/1e9}),flush=True)
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    actions = torch.zeros(n, wrapped.num_actions, device='cuda:0')
    with torch.inference_mode():
        profiler.enable()
        for _ in range(5):
            wrapped.step(actions)
        torch.cuda.synchronize()
        profiler.disable()
    pstats.Stats(profiler).sort_stats('cumulative').print_stats(30)
finally:
    env.close()
