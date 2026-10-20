"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.raycast_occlusion import disable_raycast_near_occlusion
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing with dummy agents)."""
  disable_raycast_near_occlusion: bool = False
  """Disable task-configured near-field raycast occlusion randomization."""


_MANUAL_PLAY_OBSTACLE_ENTITY_PREFIX = "manual_obstacle"


def _compose_overlay_callbacks(
  *callbacks: Callable[[], tuple[str, str] | None] | None,
) -> Callable[[], tuple[str, str] | None] | None:
  valid_callbacks = [callback for callback in callbacks if callback is not None]
  if not valid_callbacks:
    return None

  def composed() -> tuple[str, str] | None:
    sections: list[tuple[str, str]] = []
    for callback in valid_callbacks:
      section = callback()
      if section is not None:
        sections.append(section)

    if not sections:
      return None
    if len(sections) == 1:
      return sections[0]

    labels = "\n\n".join(section[0] for section in sections)
    values = "\n\n".join(section[1] for section in sections)
    return labels, values

  return composed


def _build_manual_obstacle_overlay_callback(
  env: RslRlVecEnvWrapper,
) -> Callable[[], tuple[str, str] | None] | None:
  real_env = env.unwrapped
  obstacle_names = sorted(
    name
    for name in real_env.scene.entities
    if name.startswith(_MANUAL_PLAY_OBSTACLE_ENTITY_PREFIX)
  )
  if not obstacle_names:
    return None

  def overlay() -> tuple[str, str]:
    obstacle_positions = []
    for obstacle_name in obstacle_names:
      obstacle = real_env.scene[obstacle_name]
      obstacle_pos = obstacle.data.root_link_pos_w[0].detach().cpu().tolist()
      display_name = obstacle_name.removeprefix(
        f"{_MANUAL_PLAY_OBSTACLE_ENTITY_PREFIX}_"
      )
      obstacle_positions.append(
        f"{display_name.upper()}: {obstacle_pos[0]:+.2f}, {obstacle_pos[1]:+.2f}, {obstacle_pos[2]:+.2f}"
      )

    labels = (
      "Manual Obstacles\n"
      "----------------\n"
      "Mode\n"
      "Draggable\n"
      "Static\n"
      "Select\n"
      "Move XY\n"
      "Move Plane\n"
      "Rotate\n"
      "Reset\n"
      "Markers\n"
      "Obstacle XYZ"
    )
    values = (
      "\n"
      "\n"
      "KEYBOARD_CONTROL=2\n"
      "A and B only\n"
      "Terrain pillars/walls/rough tiles\n"
      "Left double-click obstacle\n"
      "Ctrl+Shift+Right drag\n"
      "Ctrl+Right drag\n"
      "Ctrl+Left drag\n"
      "Enter\n"
      "A=yellow top marker, B=magenta top marker\n" + "\n".join(obstacle_positions)
    )
    return labels, values

  return overlay


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.disable_raycast_near_occlusion:
    disabled_groups = disable_raycast_near_occlusion(env_cfg)
    if disabled_groups:
      disabled_names = ", ".join(disabled_groups)
      print(
        "[INFO] Disabled raycast near-occlusion DR for "
        f"observation groups: {disabled_names}"
      )
    else:
      print(
        "[WARN] --disable-raycast-near-occlusion was requested, "
        "but no task-configured near-occlusion params were found."
      )

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested.
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  # Check if keyboard control is enabled and get the keyboard command term.
  key_callback = None
  overlay_callback = None
  manual_obstacle_overlay = _build_manual_obstacle_overlay_callback(env)
  from mjlab.tasks.velocity.mdp.keyboard_command import get_keyboard_control_mode

  keyboard_control_mode = get_keyboard_control_mode()
  if keyboard_control_mode is not None:
    # Try to get the keyboard command from command manager.
    # Supports: KeyboardVelocityCommand, KeyboardGoalTrackingCommand,
    # KeyboardCircularGoalTrackingCommand
    from mjlab.tasks.velocity.mdp import KeyboardVelocityCommand
    from mjlab.tasks.velocity.mdp.keyboard_command import (
      KeyboardCircularGoalTrackingCommand,
      KeyboardGoalTrackingCommand,
    )

    real_env = env.unwrapped
    if hasattr(real_env, "command_manager"):
      for term in real_env.command_manager._terms.values():
        if isinstance(
          term,
          (
            KeyboardVelocityCommand,
            KeyboardGoalTrackingCommand,
            KeyboardCircularGoalTrackingCommand,
          ),
        ):
          key_callback = term.handle_key
          print(
            f"[INFO] Keyboard control callback registered with viewer "
            f"({type(term).__name__})."
          )
          break
      if key_callback is None:
        print("[WARN] KEYBOARD_CONTROL enabled but no keyboard command term found.")

  if keyboard_control_mode == 2 and manual_obstacle_overlay is None:
    print(
      "[WARN] KEYBOARD_CONTROL=2 requested, but this task does not expose a "
      "manual play obstacle. Falling back to keyboard goal control only."
    )

  if manual_obstacle_overlay is not None:
    print(
      "[INFO] Manual obstacles enabled. Only the dedicated A/B draggable "
      "obstacles can be moved; terrain-generated obstacles remain static. "
      "Left-double-click an obstacle, then use Ctrl+Shift+right-drag to move "
      "it in the ground plane."
    )

  # Check for FixedSequentialGoalCommand to enable overlay display
  from mjlab.tasks.velocity.mdp.reachable_goal_command import FixedSequentialGoalCommand

  real_env = env.unwrapped
  if hasattr(real_env, "command_manager"):
    for term in real_env.command_manager._terms.values():
      if isinstance(term, FixedSequentialGoalCommand):
        # Create overlay callback using the command term's method
        overlay_callback = term.get_overlay_info
        print("[INFO] Goal tracking overlay enabled for FixedSequentialGoalCommand.")
        break

  overlay_callback = _compose_overlay_callbacks(
    overlay_callback,
    manual_obstacle_overlay,
  )

  if manual_obstacle_overlay is not None and resolved_viewer != "native":
    raise RuntimeError(
      "KEYBOARD_CONTROL=2 manual obstacle dragging requires the native MuJoCo "
      "viewer. Re-run with a local display or pass --viewer native."
    )

  if resolved_viewer == "native":
    NativeMujocoViewer(
      env, policy, key_callback=key_callback, overlay_callback=overlay_callback
    ).run()
  elif resolved_viewer == "viser":
    if key_callback is not None:
      print("[WARN] Keyboard control is not supported with Viser viewer.")
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
