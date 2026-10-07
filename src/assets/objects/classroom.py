"""Furniture visuals and conservative collision boxes used by the wall CBF."""
import mujoco
from mjlab.entity import Entity, EntityCfg

# Layout shared by scene creation, safe spawns and random route generation.
CLASSROOM_ROWS = 5
CLASSROOM_COLUMN_Y = (-5.3, -1.9, 1.9)
CLASSROOM_ROW_START = 3.5
CLASSROOM_ROW_PITCH = 3.8
CLASSROOM_X_BOUNDS = (-1.7, 23.4)
CLASSROOM_Y_BOUNDS = (-6.5, 3.1)
CLASSROOM_SPAWN_LANES = (0.0, -3.6)
CLASSROOM_ROUTE_STATIONS = (0.0, 5.0, 8.8, 12.6, 16.4, 20.9)


class StaticClassroomEntity(Entity):
  """World-welded scenery; bypass mjlab's automatic fixed-base mocap wrapper."""

  def _build_spec(self) -> None:
    self._spec = self.cfg.spec_fn()
    if self._spec.joints or any(body.mocap for body in self._spec.bodies):
      raise ValueError("Static classroom scenery must have no joints or mocap bodies")


class StaticClassroomEntityCfg(EntityCfg):
  def build(self) -> StaticClassroomEntity:
    return StaticClassroomEntity(self)


def get_static_wall_cfg(half_extents, rgba):
  """A world-welded box with the same collision naming as the wall tasks."""
  def spec_fn():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    body.add_geom(name="wall_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=half_extents, rgba=rgba)
    return spec

  return StaticClassroomEntityCfg(spec_fn=spec_fn)


def get_furniture_cfg(kind):
  if kind not in ("desk", "chair"):
    raise ValueError(kind)
  half = (0.4, 0.6, 0.38) if kind == "desk" else (0.25, 0.25, 0.45)

  def spec_fn():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    body.add_geom(name="wall_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=half, rgba=(0, 0, 0, 0))

    def box(name, size, pos, color):
      body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=size, pos=pos, rgba=color, contype=0,
                    conaffinity=0, mass=0)

    wood, metal = (0.65, 0.42, 0.22, 1), (0.22, 0.25, 0.28, 1)
    if kind == "desk":
      box("top", (0.4, 0.6, 0.025), (0, 0, 0.355), wood)
      for i, x in enumerate((-0.33, 0.33)):
        for j, y in enumerate((-0.53, 0.53)):
          box(f"leg_{i}_{j}", (0.025, 0.025, 0.355), (x, y, -0.025), metal)
    else:
      box("seat", (0.25, 0.25, 0.025), (0, 0, 0), wood)
      box("back", (0.025, 0.25, 0.21), (-0.225, 0, 0.24), wood)
      for i, x in enumerate((-0.20, 0.20)):
        for j, y in enumerate((-0.20, 0.20)):
          box(f"leg_{i}_{j}", (0.02, 0.02, 0.2125), (x, y, -0.2375), metal)
    return spec

  return StaticClassroomEntityCfg(spec_fn=spec_fn,
                   init_state=EntityCfg.InitialStateCfg(pos=(0, 0, half[2])))
