"""Furniture visuals and conservative collision boxes used by the wall CBF."""
import mujoco
from mjlab.entity import EntityCfg


def get_furniture_cfg(kind):
  if kind not in ("desk", "chair"):
    raise ValueError(kind)
  half = (0.4, 0.6, 0.38) if kind == "desk" else (0.25, 0.25, 0.45)

  def spec_fn():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    body.add_freejoint()
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

  return EntityCfg(spec_fn=spec_fn,
                   init_state=EntityCfg.InitialStateCfg(pos=(0, 0, half[2])))
