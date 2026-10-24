"""WALK_THROW_ENABLED=0: pure-navigation classroom config (LP-NavOA comparison)."""
import pytest

from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)


@pytest.fixture
def no_throw(monkeypatch):
    monkeypatch.setenv("WALK_THROW_ENABLED", "0")
    return g1_amp_dodge_mimickit_classroom_flat_env_cfg


def test_throw_events_and_curriculum_removed(no_throw):
    cfg = no_throw()
    assert "throw_ball_on_dwell" not in cfg.events
    assert "hide_outside_classroom_ball" not in cfg.events
    assert "classroom_throw_interval_update" not in cfg.events
    assert "classroom_throw_interval" not in cfg.curriculum
    # The ball-parking reset event stays so the ball remains below the floor.
    assert "reset_dodge_state" in cfg.events


def test_dodge_rewards_removed(no_throw):
    cfg = no_throw()
    for name in ("mimickit_dodge", "dodge_stillness_when_safe",
                 "dodge_action_rate_when_safe", "dodge_link_cbf", "dodge_sidestep"):
        assert name not in cfg.rewards


def test_ball_obs_dropped_but_wall_state_kept(no_throw):
    cfg = no_throw()
    terms = cfg.observations["ball_state"].terms
    assert "ball_state" not in terms
    # Furniture geometry (k=8) is a navigation input and must survive.
    assert "wall_state" in terms


def test_play_cfg_also_no_throw(no_throw):
    cfg = no_throw(play=True)
    assert "throw_ball_on_dwell" not in cfg.events


def test_default_config_unchanged(monkeypatch):
    monkeypatch.delenv("WALK_THROW_ENABLED", raising=False)
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    assert "throw_ball_on_dwell" in cfg.events
    assert "hide_outside_classroom_ball" in cfg.events
    assert "classroom_throw_interval" in cfg.curriculum
    assert "mimickit_dodge" in cfg.rewards
    assert "ball_state" in cfg.observations["ball_state"].terms
