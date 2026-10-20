"""RL configuration for Unitree G1 velocity task with multi-direction circular raycast."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlRaycastModelCfg(RslRlModelCfg):
  """Config for a model with raycast temporal encoding."""

  class_name: str = "mjlab.tasks.velocity.rl:ActorCriticRaycast"
  raycast_obs_name: str | tuple[str, ...] = "raycast"
  raycast_history_length: int = 3
  raycast_grid_size: tuple[int, int] = (8, 8)
  raycast_frame_output_dim: int = 32
  raycast_conv_out_channels: int = 16
  raycast_conv_kernel_size: int = 2
  raycast_latent_dim: int = 32
  share_raycast_encoder: bool = True


def unitree_g1_ppo_raycast_h1_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration with multi-direction single-frame raycast.

  This configuration uses three raycast sensors (front, left, right) to provide
  wider field-of-view obstacle detection. The critic uses height scan for
  privileged terrain information during training.
  """
  _raycast_cfg: dict = dict(
    raycast_obs_name=("raycast", "raycast_left", "raycast_right"),
    raycast_history_length=1,
    raycast_grid_size=(8, 8),
    raycast_frame_output_dim=32,
    raycast_conv_out_channels=16,
    raycast_conv_kernel_size=1,
    raycast_latent_dim=32,
    share_raycast_encoder=False,
  )
  return RslRlOnPolicyRunnerCfg(
    obs_groups={
      "actor": ("actor", "raycast", "raycast_left", "raycast_right"),
      "critic": ("critic",),
    },
    actor=RslRlRaycastModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      **_raycast_cfg,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity_raycast_h1",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=7_500,
  )
