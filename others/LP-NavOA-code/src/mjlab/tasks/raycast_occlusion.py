"""Helpers for task-configured raycast near-occlusion DR."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

RAYCAST_NEAR_OCCLUSION_PARAM_KEYS = (
  "near_occlusion_prob",
  "near_occlusion_distance_range",
  "near_occlusion_ray_prob",
  "near_occlusion_full_frame_prob",
)

DEFAULT_RAYCAST_OBSERVATION_NAMES = (
  "raycast",
  "raycast_left",
  "raycast_right",
)


def disable_raycast_near_occlusion(
  env_cfg: ManagerBasedRlEnvCfg,
  observation_names: tuple[str, ...] = DEFAULT_RAYCAST_OBSERVATION_NAMES,
) -> tuple[str, ...]:
  """Remove task-configured near-field raycast occlusion parameters.

  Tasks enable observation-level near-field occlusion by injecting optional
  keyword arguments into the raycast observation term params. Removing those
  keys restores the default behavior of ``raycast_distances`` without changing
  the underlying sensor or observation function wiring.

  Args:
    env_cfg: Environment configuration to mutate in place.
    observation_names: Raycast observation group names to inspect.

  Returns:
    Tuple of observation group names whose term params were updated.
  """
  disabled_groups: list[str] = []

  for obs_name in observation_names:
    if obs_name not in env_cfg.observations:
      continue

    obs_group = env_cfg.observations[obs_name]
    if obs_name not in obs_group.terms:
      continue

    obs_term = obs_group.terms[obs_name]
    if not obs_term.params:
      continue

    if not any(key in obs_term.params for key in RAYCAST_NEAR_OCCLUSION_PARAM_KEYS):
      continue

    params = dict(obs_term.params)
    for key in RAYCAST_NEAR_OCCLUSION_PARAM_KEYS:
      params.pop(key, None)
    obs_term.params = params
    disabled_groups.append(obs_name)

  return tuple(disabled_groups)
