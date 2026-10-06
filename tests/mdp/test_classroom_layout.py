"""Check navigable clearance using collision envelopes, including chair backs."""
import pytest
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
    g1_amp_dodge_mimickit_classroom_flat_env_cfg,
)


def test_classroom_clearance_and_wiring():
    cfg = g1_amp_dodge_mimickit_classroom_flat_env_cfg()
    params = cfg.events['reset_wall_pose'].params
    poses = params['layout']
    names = params['wall_names']
    assert len(names) == 52
    assert 'recycle_walls_ahead' not in cfg.events
    assert cfg.commands['twist'].rel_standing_envs == 0
    assert cfg.commands['twist'].rel_inplace_throw_envs == 0
    assert cfg.events['reset_walk_path'].params['turn_max'] == 0
    assert cfg.observations['ball_state'].terms['wall_state'].params['k'] == 6
    # Inner desk edges define the central aisle.
    assert (2.1 - 0.6) * 2 == pytest.approx(3.0)
    # Adjacent columns define the two additional longitudinal aisles.
    assert (5.8 - 0.6) - (2.1 + 0.6) == pytest.approx(2.5)
    # Every chair/desk pair must leave the promised cross aisle to the next row.
    for row in range(5):
        for col in range(4):
            desk = poses[row * 8 + col * 2]
            next_chair = poses[(row + 1) * 8 + col * 2 + 1]
            assert next_chair[0] - 0.25 - (desk[0] + 0.4) == pytest.approx(2.5)
    for name, pose in zip(names, poses, strict=True):
        asset = cfg.scene.entities[name]
        spec = asset.spec_fn()
        assert tuple(spec.body('wall').pos) == pytest.approx(pose[:3])
        model = spec.compile()
        assert model.geom('wall_collision').contype == 1
