"""Opening ledgers prevent turn/join oscillations from earning progress twice."""
import math

import pytest
import torch

from src.tasks.amp_loco.mdp.path_navigation import classroom_follow_state
from tests.mdp.test_classroom_follow import advance, follower, reward


def components(command, env, q=.5):
    reward(env, progress_penalty_scale=q, turn_stall_weight=0.,
           join_stall_weight=0., stall_weight=0., recovery_stall_weight=0.,
           deviation_weight=0.)
    return classroom_follow_state(command).reward_components


@pytest.mark.parametrize('q', [.5, 1.])
def test_repeated_turn_closes_only_pay_initial_improvement(q):
    command, env = follower((120.,))
    total = 0.
    cycles = 6
    for _ in range(cycles):
        for angle in (110., 120.):
            advance(command, env, angles=[angle])
            total += components(command, env, q)['turn'].item()
    assert total < 0.
    assert total == pytest.approx(2 * 10 / 180 - q * cycles * 2 * 10 / 180, abs=1e-6)


@pytest.mark.parametrize('q', [0., .5, 1.])
def test_self_caused_turn_round_trip_loses_exact_opening_fee(q):
    command, env = follower((110.,))
    advance(command, env, angles=[120.])
    total = components(command, env, q)['turn'].item()
    advance(command, env, angles=[110.])
    total += components(command, env, q)['turn'].item()
    assert total == pytest.approx(-q * 2 * 10 / 180, abs=1e-6)
    assert classroom_follow_state(command).turn_bank.item() == pytest.approx(0., abs=1e-6)


def test_monotonic_turn_is_paid_in_full():
    command, env = follower((120.,))
    total = 0.
    for angle in range(110, -1, -10):
        advance(command, env, angles=[angle])
        total += components(command, env)['turn'].item()
        assert classroom_follow_state(command).turn_bank.item() == 0.
    assert total == pytest.approx(2 * 120 / 180, abs=1e-6)


@pytest.mark.parametrize('deltas', [(.3, -.1, .3, -.1, .4),
                                    (-.1, -.1, .3, -.1, .7)])
def test_turn_total_depends_on_net_improvement_and_total_openings(deltas):
    angle = 1.5
    command, env = follower((math.degrees(angle),))
    total = 0.
    for delta in deltas:
        angle -= delta
        advance(command, env, angles=[math.degrees(angle)])
        total += components(command, env)['turn'].item()
    state = classroom_follow_state(command)
    assert state.turn_bank.item() == pytest.approx(0., abs=1e-6)
    total_open = sum(max(-delta, 0.) for delta in deltas)
    expected = 2 / math.pi * (sum(deltas) - .5 * total_open)
    assert total == pytest.approx(expected, abs=1e-6)


def test_join_phase_cycles_repay_all_self_caused_openings():
    command, env = follower()
    total = 0.
    cycles = 6
    for _ in range(cycles):
        for y in (.5, .19):
            advance(command, env, xy=[[0., y]])
            total += components(command, env)['join'].item()
    assert total < 0.
    assert total == pytest.approx(-.5 * 2 / .5 * .3 * cycles, abs=1e-6)
    assert classroom_follow_state(command).join_bank.item() == pytest.approx(0., abs=1e-6)


def test_threat_time_monotonic_turn_is_not_saved_and_recovery_is_paid():
    command, env = follower((120.,))
    command._dodge_threat[:] = True
    advance(command, env, angles=[80.])
    assert components(command, env)['turn'].item() == 0.
    command._dodge_threat[:] = False
    advance(command, env, angles=[70.])
    assert components(command, env)['turn'].item() == pytest.approx(2 * 10 / 180, abs=1e-6)


@pytest.mark.parametrize('mode', ['turn', 'join'])
def test_threat_closings_repay_debt_and_openings_do_not_add_debt(mode):
    command, env = follower((120.,) if mode == 'turn' else (0.,),
                            xy=None if mode == 'turn' else [0., .5])

    def move(value):
        if mode == 'turn':
            advance(command, env, angles=[value])
        else:
            advance(command, env, xy=[[0., value]])
        parts = components(command, env)
        return classroom_follow_state(command), parts[mode].item()

    state, value = move(130. if mode == 'turn' else .6)
    assert value < 0.
    assert getattr(state, mode + '_bank').item() > 0.
    command._dodge_threat[:] = True
    state, value = move(125. if mode == 'turn' else .55)
    assert value == 0.
    remaining = math.radians(5.) if mode == 'turn' else .05
    assert getattr(state, mode + '_bank').item() == pytest.approx(remaining, abs=1e-6)
    state, value = move(120. if mode == 'turn' else .5)
    assert value == 0.
    assert getattr(state, mode + '_bank').item() == pytest.approx(0., abs=1e-6)
    state, value = move(140. if mode == 'turn' else .7)
    assert value == 0.
    assert getattr(state, mode + '_bank').item() == pytest.approx(0., abs=1e-6)
    command._dodge_threat[:] = False
    state, value = move(130. if mode == 'turn' else .6)
    expected = (2 * 10 / 180 if mode == 'turn' else
                2 / .5 * .1 * state.step_theta.cos().clamp_min(0).square().item())
    assert value == pytest.approx(expected, abs=1e-6)


def test_partial_same_step_reset_clears_only_reset_env_ledgers_and_payments():
    command, env = follower((0., 0.))
    advance(command, env, angles=[10., 10.], xy=[[0., .5]] * 2)
    parts = components(command, env)
    state = classroom_follow_state(command)
    before = {name: getattr(state, name).clone()
              for name in ('turn_bank', 'join_bank', 'turn_pay', 'join_pay')}
    before_parts = {name: value.clone() for name, value in parts.items()}
    assert torch.all(state.turn_bank > 0.) and torch.all(state.join_bank > 0.)
    env._walk_path_epoch[0] += 1
    command._update_command()
    parts = components(command, env)
    for name, old in before.items():
        assert getattr(state, name)[0].item() == 0.
        assert getattr(state, name)[1].item() == old[1].item()
    assert parts['turn'][0].item() == parts['join'][0].item() == 0.
    for name, old in before_parts.items():
        assert parts[name][1].item() == old[1].item()
    # Re-reading reward and commands must not charge/repay a ledger twice.
    advance(command, env, angles=[5., 5.], xy=[[0., .45]] * 2)
    parts = components(command, env)
    snapshot = {name: getattr(state, name).clone() for name in before}
    paid = {name: value.clone() for name, value in parts.items()}
    command._update_command()
    parts = components(command, env)
    for name, old in snapshot.items():
        assert torch.equal(getattr(state, name), old)
    for name, old in paid.items():
        assert torch.equal(parts[name], old)


def test_bank_metrics_are_registered_in_classroom_config():
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
        g1_amp_dodge_mimickit_classroom_flat_env_cfg,
    )
    command, env = follower()
    advance(command, env, angles=[10.], xy=[[0., .5]])
    components(command, env)
    state = classroom_follow_state(command)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    for name in ('turn_bank', 'join_bank'):
        metric = cfg.metrics['navigation/' + name]
        assert torch.equal(metric.func(env, **metric.params), getattr(state, name))
