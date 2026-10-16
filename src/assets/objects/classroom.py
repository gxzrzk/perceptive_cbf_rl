"""Furniture visuals and conservative collision boxes used by the wall CBF."""
import mujoco
from mjlab.entity import Entity, EntityCfg

# Layout shared by scene creation, safe spawns and random route generation.
# Measure aisles between furniture collision envelopes and wall INNER faces.
# Uniform gap: desk<->desk and desk<->wall passages are all this wide, so the
# 15 pairs tile the room evenly with no larger open areas at the front/back.
CLASSROOM_AISLE_WIDTH = 1.5
CLASSROOM_WALL_HALF_THICKNESS = 0.08
CLASSROOM_DESK_HALF_SIZE = (0.4, 0.6, 0.38)
CLASSROOM_CHAIR_HALF_SIZE = (0.25, 0.25, 0.45)
CLASSROOM_CHAIR_OFFSET = 0.95
CLASSROOM_PAIR_REAR = CLASSROOM_CHAIR_OFFSET + CLASSROOM_CHAIR_HALF_SIZE[0]
CLASSROOM_PAIR_FRONT = CLASSROOM_DESK_HALF_SIZE[0]
CLASSROOM_PAIR_LENGTH = CLASSROOM_PAIR_REAR + CLASSROOM_PAIR_FRONT
CLASSROOM_ROWS = 5
CLASSROOM_COLUMN_PITCH = 2 * CLASSROOM_DESK_HALF_SIZE[1] + CLASSROOM_AISLE_WIDTH
CLASSROOM_COLUMN_Y = tuple(index * CLASSROOM_COLUMN_PITCH for index in (-1.5, -0.5, 0.5))
# Keep the front cross-aisle center at x=0 and the central internal aisle at y=0.
CLASSROOM_ROW_START = CLASSROOM_PAIR_REAR + CLASSROOM_AISLE_WIDTH / 2
CLASSROOM_ROW_PITCH = CLASSROOM_PAIR_LENGTH + CLASSROOM_AISLE_WIDTH
CLASSROOM_FURNITURE_X_MIN = CLASSROOM_ROW_START - CLASSROOM_PAIR_REAR
CLASSROOM_FURNITURE_X_MAX = (CLASSROOM_ROW_START + (CLASSROOM_ROWS - 1) * CLASSROOM_ROW_PITCH
                           + CLASSROOM_PAIR_FRONT)
CLASSROOM_X_BOUNDS = (
  CLASSROOM_FURNITURE_X_MIN - CLASSROOM_AISLE_WIDTH - CLASSROOM_WALL_HALF_THICKNESS,
  CLASSROOM_FURNITURE_X_MAX + CLASSROOM_AISLE_WIDTH + CLASSROOM_WALL_HALF_THICKNESS,
)
CLASSROOM_Y_BOUNDS = (
  CLASSROOM_COLUMN_Y[0] - CLASSROOM_DESK_HALF_SIZE[1] - CLASSROOM_AISLE_WIDTH - CLASSROOM_WALL_HALF_THICKNESS,
  CLASSROOM_COLUMN_Y[-1] + CLASSROOM_DESK_HALF_SIZE[1] + CLASSROOM_AISLE_WIDTH + CLASSROOM_WALL_HALF_THICKNESS,
)
# Whole room, including the perimeter walls; ball centers on a face are inside.
CLASSROOM_WALL_HEIGHT = 3.0
CLASSROOM_BALL_BOUNDS = (
  (CLASSROOM_X_BOUNDS[0] - CLASSROOM_WALL_HALF_THICKNESS,
   CLASSROOM_Y_BOUNDS[0] - CLASSROOM_WALL_HALF_THICKNESS, 0.0),
  (CLASSROOM_X_BOUNDS[1] + CLASSROOM_WALL_HALF_THICKNESS,
   CLASSROOM_Y_BOUNDS[1] + CLASSROOM_WALL_HALF_THICKNESS, CLASSROOM_WALL_HEIGHT),
)
CLASSROOM_BALL_PARK_POS = (
  sum(CLASSROOM_X_BOUNDS) / 2, sum(CLASSROOM_Y_BOUNDS) / 2, -10.0,
)
CLASSROOM_SPAWN_LANES = (
  (CLASSROOM_COLUMN_Y[1] + CLASSROOM_COLUMN_Y[2]) / 2,
  (CLASSROOM_COLUMN_Y[0] + CLASSROOM_COLUMN_Y[1]) / 2,
)
# Center every cross aisle, including the uniform front/back perimeter passages.
CLASSROOM_ROUTE_STATIONS = (
  CLASSROOM_FURNITURE_X_MIN - CLASSROOM_AISLE_WIDTH / 2,
  *(CLASSROOM_ROW_START + row * CLASSROOM_ROW_PITCH + CLASSROOM_PAIR_FRONT + CLASSROOM_AISLE_WIDTH / 2
    for row in range(CLASSROOM_ROWS - 1)),
  CLASSROOM_FURNITURE_X_MAX + CLASSROOM_AISLE_WIDTH / 2,
)
# Minimum root-point clearance on rounded corners of the 1.5 m aisles. The
# 0.25 m margin to the 0.75 m half-aisle still allows the default 0.6 m corner
# radius at a 90-degree junction (needs (0.75 - clearance) >= 0.6*(1-cos45)).
CLASSROOM_ROUTE_CLEARANCE = 0.5


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
  half = CLASSROOM_DESK_HALF_SIZE if kind == "desk" else CLASSROOM_CHAIR_HALF_SIZE

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
