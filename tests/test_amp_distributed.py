"""Exercise AMP synchronization through real two-process PPO updates on CPU."""
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from src.tasks.amp_loco.rl.amp_algorithm import AmpPPO
from src.tasks.amp_loco.rl.amp_discriminator import Discriminator
from src.tasks.amp_loco.rl.amp_normalizer import Normalizer
from src.tasks.amp_loco.rl.amp_replay_buffer import ReplayBuffer


def make_algorithm(rank=0, world_size=1, device='cpu'):
    torch.manual_seed(42 + rank)
    obs = TensorDict({'actor': torch.randn(2, 3, device=device),
                      'amp': torch.randn(2, 2, device=device)}, batch_size=[2])
    groups = {'actor': ['actor'], 'critic': ['actor']}
    actor = MLPModel(obs, groups, 'actor', 2, hidden_dims=(8,),
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .5})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=(8,))
    raw_policy = torch.tensor([[10., 20.], [12., 24.], [14., 28.], [16., 32.]], device=device) + rank
    raw_expert = raw_policy + 20.

    def samples(states):
        def generate(num_mini_batch, mini_batch_size):
            assert mini_batch_size == len(states)
            for _ in range(num_mini_batch):
                yield states, states + 1.
        return generate

    replay = ReplayBuffer(2, 8, device)
    replay.feed_forward_generator = samples(raw_policy)
    normalizer = Normalizer(2, device=device)
    normalizer.update(torch.tensor([[100., 200.], [110., 220.]], device=device))
    alg = AmpPPO(
        actor, critic, RolloutStorage('rl', 2, 2, obs, [2], device),
        Discriminator(4, hidden_layer_sizes=(8,), device=device),
        SimpleNamespace(feed_forward_generator=samples(raw_expert)), normalizer, replay,
        device=device, num_learning_epochs=1, num_mini_batches=1,
        multi_gpu_cfg=({'global_rank': rank, 'local_rank': rank, 'world_size': world_size}
                       if world_size > 1 else None),
    )
    return alg, obs, torch.cat((raw_policy, raw_expert))


def rollout_and_update(alg, obs):
    with torch.inference_mode():
        for _ in range(2):
            alg.act(obs)
            alg.process_env_step(obs, torch.tensor([1., 2.], device=alg.device),
                                 torch.zeros(2, device=alg.device), {})
        alg.compute_returns(obs)
    return alg.update()


def test_update_fits_raw_observations_and_load_drops_cached_stats():
    alg, obs, raw = make_algorithm()
    expected = Normalizer(2, device='cpu')
    expected.update(torch.tensor([[100., 200.], [110., 220.]]))
    expected.update(raw)
    losses = rollout_and_update(alg, obs)
    torch.testing.assert_close(alg.amp_normalizer.mean, expected.mean)
    torch.testing.assert_close(alg.amp_normalizer.var, expected.var)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    saved = alg.save()
    alg.amp_normalizer.normalize_torch(raw, 'cpu')
    assert alg.amp_normalizer._mean_f32 is not None
    alg.load(saved, None, strict=True)
    assert alg.amp_normalizer._mean_f32 is None


def test_terminal_amp_transition_preserves_previous_frame():
    alg, obs, _ = make_algorithm()
    next_obs = obs.clone()
    next_obs['amp'] = obs['amp'] + 100.
    with torch.inference_mode():
        alg.act(obs)
        alg.process_env_step(next_obs, torch.ones(2), torch.tensor([1., 0.]), {})
    torch.testing.assert_close(alg.amp_storage.states[:2], obs['amp'])
    torch.testing.assert_close(alg.amp_storage.next_states[0], obs['amp'][0])
    torch.testing.assert_close(alg.amp_storage.next_states[1], next_obs['amp'][1])


def assert_identical(tensor):
    values = [torch.empty_like(tensor) for _ in range(2)]
    dist.all_gather(values, tensor)
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)


def distributed_worker(rank, rendezvous, device_ids=None):
    device = f'cuda:{device_ids[rank]}' if device_ids is not None else 'cpu'
    if device_ids is not None:
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device_ids is not None else 'gloo',
                            init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        alg, obs, raw = make_algorithm(rank, 2, device)
        alg.broadcast_parameters()
        for model in (alg.actor, alg.critic, alg.discriminator):
            for parameter in model.parameters():
                assert_identical(parameter.detach())
                parameter.grad = torch.full_like(parameter, rank + 1.)
        alg.reduce_parameters()
        for group in alg.optimizer.param_groups:
            for parameter in group['params']:
                torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 1.5))

        # Reference is a single normalizer fitted to the union of both ranks'
        # raw observations, including the shared initialization samples once.
        expected = Normalizer(2, device=device)
        expected.update(torch.tensor([[100., 200.], [110., 220.]], device=device))
        expected.update(torch.cat((raw - rank, raw - rank + 1.)))
        for _ in range(2):
            rollout_and_update(alg, obs)
            for model in (alg.actor, alg.critic, alg.discriminator):
                for parameter in model.parameters():
                    assert_identical(parameter.detach())
            for state in alg.optimizer.state.values():
                assert_identical(state['exp_avg'])
            torch.testing.assert_close(alg.amp_normalizer.mean, expected.mean)
            torch.testing.assert_close(alg.amp_normalizer.var, expected.var)
            assert alg.amp_normalizer.count == expected.count
            expected.update(torch.cat((raw - rank, raw - rank + 1.)))
    finally:
        dist.destroy_process_group()


def test_two_workers_share_discriminator_gradients_and_raw_amp_stats(tmp_path):
    mp.spawn(distributed_worker,
             args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
