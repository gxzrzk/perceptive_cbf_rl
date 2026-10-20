"""Standalone MuJoCo viewer for visualizing the G1 RayCast sensor.

This script avoids the RL "play" pipeline and drives MuJoCo directly:

- Builds a minimal scene with the Unitree G1, a flat ground plane, and
  three 8x8 pinhole RayCast sensors (front + left + right on pelvis).
- Runs the physics loop manually (no policy/env manager) so viewer hotkeys like
  pause or contact visualization stay responsive.
- Draws ray debug geometry straight into the MuJoCo scene each frame and shows
  the live 8x8 distance grid in the top-left overlay.

Controls inside the viewer:
  * SPACE  - Pause/Resume simulation
  * '.'    - Single simulation step while paused
  * ENTER  - Reset to the default standing pose
  * MuJoCo native keys (c, tab, etc.) continue to work for diagnostics
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_active_g1_n3_robot_cfg
from mjlab.scene.scene import Scene, SceneCfg
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.native import keys as keycodes
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

RAY_RES = (8, 8)
# 3 raycast sensors: front, left, right (matching velocity task config)
RAY_NAMES = ("raycast_front", "raycast_left", "raycast_right")
DIAGONAL_FOV_DEG = 65.0


def _square_fovy(diagonal_deg: float) -> float:
  diag_rad = math.radians(diagonal_deg)
  return math.degrees(2.0 * math.atan(math.tan(diag_rad / 2.0) / math.sqrt(2.0)))


def _add_ground(spec: mujoco.MjSpec) -> None:
  # Add simple lighting
  light = spec.worldbody.add_light()
  light.pos[:] = (0, 0, 4)  # type: ignore[index]
  light.dir[:] = (0, 0, -1)  # type: ignore[index]
  light.diffuse[:] = (0.6, 0.6, 0.6)  # type: ignore[index]
  light.ambient[:] = (0.2, 0.2, 0.2)  # type: ignore[index]

  # Add checkerboard texture (Skipped to avoid Spec API issues, using solid color with light)
  # tex = spec.add_texture()
  # tex.type = mujoco.mjtTexture.mjTEXTURE_2D
  # ...

  geom = spec.worldbody.add_geom()
  geom.name = "ground"
  geom.type = mujoco.mjtGeom.mjGEOM_PLANE
  geom.size[:] = (20.0, 20.0, 0.1)  # type: ignore[index]
  # geom.material = "groundplane"  <-- Removed
  geom.rgba[:] = (0.5, 0.5, 0.6, 1.0)  # type: ignore[index] Light blue-ish grey
  geom.friction[:] = (1.0, 0.01, 0.001)  # type: ignore[index]
  geom.contype = 1
  geom.conaffinity = 1
  geom.group = 0

  # Add some obstacles
  # Sphere
  body = spec.worldbody.add_body()
  body.name = "obstacle_sphere"
  body.pos[:] = (1.5, 0.5, 0.5)  # type: ignore[index]
  geom = body.add_geom()
  geom.name = "obstacle_sphere_geom"
  geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
  geom.size[:] = (0.4, 0.0, 0.0)  # type: ignore[index]
  geom.rgba[:] = (0.8, 0.2, 0.2, 1.0)  # type: ignore[index]
  geom.contype = 1
  geom.conaffinity = 1

  # Box
  body2 = spec.worldbody.add_body()
  body2.name = "obstacle_box"
  body2.pos[:] = (1.5, -0.6, 0.4)  # type: ignore[index]
  geom2 = body2.add_geom()
  geom2.name = "obstacle_box_geom"
  geom2.type = mujoco.mjtGeom.mjGEOM_BOX
  geom2.size[:] = (0.4, 0.4, 0.4)  # type: ignore[index]
  geom2.rgba[:] = (0.2, 0.2, 0.8, 1.0)  # type: ignore[index]
  geom2.contype = 1
  geom2.conaffinity = 1


def _build_scene_cfg() -> SceneCfg:
  square_fovy = _square_fovy(DIAGONAL_FOV_DEG)

  # Common raycast config parameters
  def make_raycast_cfg(
    name: str, site_name: str, hit_color: tuple, miss_color: tuple
  ) -> RayCastSensorCfg:
    return RayCastSensorCfg(
      name=name,
      frame=ObjRef(type="site", name=site_name, entity="robot"),
      pattern=PinholeCameraPatternCfg(
        width=RAY_RES[0], height=RAY_RES[1], fovy=square_fovy
      ),
      ray_alignment="base",
      max_distance=4.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=True,
      viz=RayCastSensorCfg.VizCfg(
        show_rays=True,
        show_normals=False,
        hit_color=hit_color,
        miss_color=miss_color,
      ),
    )

  # Create 3 raycast sensors with different colors for visibility
  ray_front_cfg = make_raycast_cfg(
    "raycast_front",
    "forward_camera",
    hit_color=(0.1, 0.8, 0.2, 0.9),  # Green = front
    miss_color=(0.8, 0.2, 0.2, 0.5),
  )
  ray_left_cfg = make_raycast_cfg(
    "raycast_left",
    "left_camera",
    hit_color=(0.8, 0.8, 0.1, 0.9),  # Yellow = left
    miss_color=(0.5, 0.1, 0.5, 0.5),
  )
  ray_right_cfg = make_raycast_cfg(
    "raycast_right",
    "right_camera",
    hit_color=(0.2, 0.6, 0.8, 0.9),  # Cyan = right
    miss_color=(0.6, 0.3, 0.1, 0.5),
  )

  return SceneCfg(
    num_envs=1,
    env_spacing=0.0,
    extent=4.0,
    entities={"robot": get_active_g1_n3_robot_cfg()},
    sensors=(ray_front_cfg, ray_left_cfg, ray_right_cfg),
    spec_fn=_add_ground,
  )


def _build_sim_cfg() -> SimulationCfg:
  return SimulationCfg(
    nconmax=128,
    njmax=256,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=20,
      ls_iterations=40,
      gravity=(0.0, 0.0, -9.81),
    ),
  )


class RaycastSimulation:
  """Lightweight simulation wrapper around Scene + Simulation."""

  def __init__(self, device: str, substeps: int = 4):
    scene_cfg = _build_scene_cfg()
    self.scene = Scene(scene_cfg, device=device)
    self.sim = Simulation(
      num_envs=scene_cfg.num_envs,
      cfg=_build_sim_cfg(),
      model=self.scene.compile(),
      device=device,
    )
    self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
    # Wire SensorContext so sim.sense() triggers prepare → kernel → finalize.
    if self.scene.sensor_context is not None:
      self.sim.set_sensor_context(self.scene.sensor_context)
    self.scene.reset(slice(None))
    self.sim.reset()
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.sim.sense()  # Compute initial raycast data.
    self.substeps = max(1, substeps)
    self.physics_dt = self.sim.cfg.mujoco.timestep

  @property
  def ray_sensors(self):  # noqa: ANN101
    """Return all 4 raycast sensors."""
    return {name: self.scene.sensors[name] for name in RAY_NAMES}

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self.sim.mj_model

  @property
  def sim_data(self):  # noqa: ANN101
    return self.sim.data

  def reset(self) -> None:
    self.scene.reset(slice(None))
    self.sim.reset()
    self.scene.write_data_to_sim()
    self.sim.forward()

  def step_frame(self) -> None:
    for _ in range(self.substeps):
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)
    self.sim.sense()  # Recompute raycast after physics substeps.

  def forward_only(self) -> None:
    self.scene.write_data_to_sim()
    self.sim.forward()
    self.sim.sense()  # Recompute raycast after forward.

  def draw_debug(self, visualizer: MujocoNativeDebugVisualizer) -> None:
    for sensor in self.ray_sensors.values():
      sensor.debug_vis(visualizer)

  def distance_grids(self) -> dict[str, torch.Tensor]:
    """Return distance grids for all 4 raycast sensors."""
    grids = {}
    for name, sensor in self.ray_sensors.items():
      distances = sensor.data.distances[0]
      grids[name] = distances.view(RAY_RES[1], RAY_RES[0])
    return grids

  def copy_state_to(self, mj_data: mujoco.MjData) -> None:
    sim_data = self.sim_data
    mj_data.qpos[:] = sim_data.qpos[0].detach().cpu().numpy()
    mj_data.qvel[:] = sim_data.qvel[0].detach().cpu().numpy()
    if self.mj_model.nmocap > 0:
      mj_data.mocap_pos[:] = sim_data.mocap_pos[0].detach().cpu().numpy()
      mj_data.mocap_quat[:] = sim_data.mocap_quat[0].detach().cpu().numpy()
    mujoco.mj_forward(self.mj_model, mj_data)

  def apply_perturbations(self, mj_data: mujoco.MjData) -> None:
    """Best-effort transfer of perturbations from viewer (mj_data) to simulation."""
    if not hasattr(mj_data, "qfrc_applied"):
      return

    try:
      qfrc_cpu = mj_data.qfrc_applied
      # Just return if no force is applied to save bandwidth
      if (qfrc_cpu == 0).all():
        return

      target = self.sim_data.qfrc_applied
      # Handle Tensor (GPU) vs Numpy (CPU)
      if isinstance(target, torch.Tensor) or (
        hasattr(target, "device") and hasattr(target, "copy_")
      ):
        target[0].copy_(
          torch.as_tensor(qfrc_cpu, device=target.device, dtype=target.dtype)
        )
      else:
        target[0][:] = qfrc_cpu
    except Exception:
      # Ignore perturbation transfer errors to keep viewer alive
      pass


@dataclass
class ViewerState:
  paused: bool = False
  single_step: bool = False
  reset_requested: bool = False

  def handle_key(self, key: int) -> None:
    if key == keycodes.KEY_SPACE:
      self.paused = not self.paused
    elif key == keycodes.KEY_PERIOD:
      self.single_step = True
    elif key == keycodes.KEY_ENTER or key == keycodes.KEY_BACKSPACE:
      self.reset_requested = True


def _format_overlay(distance_grids: dict[str, torch.Tensor]) -> tuple[str, str]:
  """Format overlay text for 4 raycast sensors."""
  lines = []
  short_names = {
    "raycast_front": "FRONT",
    "raycast_left": "LEFT",
    "raycast_right": "RIGHT",
  }
  for name in RAY_NAMES:
    grid_cpu = distance_grids[name].detach().cpu()
    short_name = short_names.get(name, name)
    lines.append(f"=== {short_name} ===")
    for row in grid_cpu:
      lines.append(" ".join(f"{float(v):.1f}" for v in row))
    lines.append("")  # blank line between grids

  matrix_text = "\n".join(lines)
  title = "Enter/BackSP: reset | Space: pause | .: step"
  return title, matrix_text


def run_viewer(sim: RaycastSimulation, target_fps: float = 60.0) -> None:
  viewer_state = ViewerState()
  mj_data = mujoco.MjData(sim.mj_model)
  handle = mujoco.viewer.launch_passive(  # type: ignore[attr-defined]
    sim.mj_model,
    mj_data,
    key_callback=viewer_state.handle_key,
    show_left_ui=False,
    show_right_ui=False,
  )
  if handle is None:
    raise RuntimeError("Failed to launch MuJoCo viewer")

  dt = 1.0 / target_fps
  try:
    while handle.is_running():
      step_start = time.time()

      if viewer_state.reset_requested:
        sim.reset()
        viewer_state.reset_requested = False

      if not viewer_state.paused or viewer_state.single_step:
        # Transfer Mouse Perturbations before stepping
        sim.apply_perturbations(mj_data)

        sim.step_frame()
        viewer_state.single_step = False
      else:
        sim.forward_only()

      sim.copy_state_to(mj_data)

      handle.user_scn.ngeom = 0
      visualizer = MujocoNativeDebugVisualizer(handle.user_scn, sim.mj_model, env_idx=0)
      sim.draw_debug(visualizer)

      text_left, text_right = _format_overlay(sim.distance_grids())
      # Attempt to display overlay
      if hasattr(handle, "set_texts"):
        overlay = (
          mujoco.mjtFontScale.mjFONTSCALE_150.value,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
          text_left,
          text_right,
        )
        # Some bindings expect a list of overlays
        try:
          handle.set_texts((overlay,))  # Try list first
        except Exception:
          try:
            handle.set_texts(overlay)  # Try single
          except Exception:
            pass  # Ignore overlay errors for now

      handle.sync()

      elapsed = time.time() - step_start
      if elapsed < dt:
        time.sleep(dt - elapsed)
  finally:
    if handle.is_running():
      handle.close()


def print_guide() -> None:
  print("\n" + "=" * 60)
  print("MuJoCo Viewer Operation Guide (G1 RayCast x3)")
  print("=" * 60)
  print("Camera Controls:")
  print("  Rotate           : Left Mouse Drag")
  print("  Pan              : Right Mouse Drag")
  print("  Zoom             : Scroll or Middle Mouse Drag")
  print("  Focus            : Double-click on object")
  print("")
  print("Simulation Controls:")
  print("  Space            : Pause / Resume")
  print("  . (Period)       : Single Step (when paused)")
  print("  Enter/Backspace  : Reset Simulation")
  print("  Ctrl + Drag      : Apply External Force (Perturbation)")
  print("")
  print("Native Visualization Options (Toggle on/off):")
  print("  1: Contact Point | 2: Contact Force | 3: Geom Group 0")
  print("  4: Geom Group 1  | 5: Geom Group 2  | H: Toggle HUD")
  print("  C: Contact Info  | J: Joint Info    | F: Frames")
  print("=" * 60 + "\n")


def main() -> None:
  configure_torch_backends()
  print_guide()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  sim = RaycastSimulation(device=device, substeps=4)
  print("Loaded standalone G1 raycast viewer (3 sensors: front, left, right).")
  print("Press Space to pause, '.' to step, Enter to reset.")
  run_viewer(sim)


if __name__ == "__main__":
  main()
