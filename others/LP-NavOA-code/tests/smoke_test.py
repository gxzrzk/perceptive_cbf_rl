"""Smoke test for the mjlab package."""

import io
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

try:
  import pytest
except ModuleNotFoundError:
  pytest = None  # type: ignore[assignment]


@pytest.mark.slow if pytest else lambda f: f
def test_basic_functionality() -> None:
  """Test that mjlab can create and close an environment."""
  import torch

  import mjlab.tasks  # noqa: F401
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Mjlab-CircularGoal-Flat-G1Rw-Pure-v6")
  cfg.scene.num_envs = 4

  # Suppress env spam.
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      env = ManagerBasedRlEnv(cfg, device="cpu")
      assert bool(torch.all(env.sim.data.time == 0.0))
      env.close()


if __name__ == "__main__":
  try:
    test_basic_functionality()
    print("✓ Smoke test passed!")
    sys.exit(0)
  except Exception as e:
    print(f"✗ Smoke test failed: {e}")
    sys.exit(1)
