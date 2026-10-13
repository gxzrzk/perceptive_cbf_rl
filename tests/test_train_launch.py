"""Check worker device routing and configuration persistence without simulation."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

import src.tasks  # noqa: F401
from scripts import train


@pytest.mark.parametrize('rank', [0, 1])
def test_only_rank_zero_renders_and_logs_complete_amp_config(monkeypatch, tmp_path, rank):
    cfg = replace(train.TrainConfig.from_task(
        'Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat'), video=True)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,5')
    monkeypatch.setenv('LOCAL_RANK', str(rank))
    monkeypatch.setenv('RANK', str(rank))
    devices, environments, videos, logged = [], [], [], {}
    monkeypatch.setattr(train.torch.cuda, 'set_device', devices.append)
    monkeypatch.setattr(train, 'configure_torch_backends', lambda: None)

    def environment(**kwargs):
        environments.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    def recorder(env, **kwargs):
        videos.append(kwargs)
        return env

    class Runner:
        def __init__(self, env, agent_cfg, *args, **kwargs):
            self.cfg = agent_cfg
            self.cfg.pop('amp_motion_files')
            self.cfg['algorithm'].pop('class_name')
            self.cfg['multi_gpu'] = {'local_rank': rank, 'world_size': 2}

        def add_git_repo_to_log(self, path):
            pass

        def learn(self, **kwargs):
            pass

    monkeypatch.setattr(train, 'ManagerBasedRlEnv', environment)
    monkeypatch.setattr(train, 'VideoRecorder', recorder)
    monkeypatch.setattr(train, 'RslRlVecEnvWrapper', lambda env, **kwargs: env)
    monkeypatch.setattr(train, 'load_runner_cls', lambda task: Runner)
    monkeypatch.setattr(train, 'dump_yaml', lambda path, value: logged.update({path.name: value}))
    train.run_train('classroom', cfg, tmp_path)
    assert devices == [rank]
    assert train.os.environ['MUJOCO_EGL_DEVICE_ID'] == ['2', '5'][rank]
    assert environments[0]['device'] == f'cuda:{rank}'
    assert environments[0]['render_mode'] == ('rgb_array' if rank == 0 else None)
    assert len(videos) == (1 if rank == 0 else 0)
    if rank == 0:
        assert logged['agent.yaml']['amp_motion_files'] == cfg.agent.amp_motion_files
        assert logged['agent.yaml']['algorithm']['class_name'] == cfg.agent.algorithm.class_name
        assert logged['agent.yaml']['multi_gpu']['world_size'] == 2
    else:
        assert not logged


def test_launcher_copies_worker_cache_logging_and_thread_settings(monkeypatch):
    import torchrunx

    cfg = train.TrainConfig.from_task('Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat')
    monkeypatch.setattr(train, 'select_gpus', lambda ids: ([1, 2], 2))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,2')
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    monkeypatch.setenv('TORCHRUNX_LOG_DIR', '')
    calls = []

    def launcher(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run=lambda *args: None)

    monkeypatch.setattr(torchrunx, 'Launcher', launcher)
    train.launch_training('classroom', cfg)
    assert calls[0]['workers_per_host'] == 2
    assert calls[0]['backend'] is None
    assert set(('TMPDIR', 'WARP*', 'MUJOCO*', 'WANDB*', 'OMP*', 'MKL*',
                'OPENBLAS*', 'LD_LIBRARY_PATH')).issubset(calls[0]['copy_env_vars'])
