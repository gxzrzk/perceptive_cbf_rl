"""A static wall (obstacle) entity for the G1 dodge-with-obstacle tasks.

The wall constrains the robot's dodge space: it stands beside the spawn point so
the robot must evade the thrown ball without backing/sidestepping into it.

Implemented as a *floating non-articulated* entity (single box body + freejoint),
exactly like the ball (``assets/objects/ball``), NOT as a welded static body: mjlab
compiles welded geoms into the shared model at global coordinates, so they cannot be
placed per env. A freejoint entity's root pose is written per env by
``reset_wall_pose`` (randomized side/distance each episode) and re-pinned every step
by ``pin_wall`` (see ``amp_loco.mdp.events``), which makes it kinematically static --
robot and ball both bounce off without moving it.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg

# Wall half-extents (m): (half_length, half_thickness, half_height). Full size is
# 3.0 m long (spans well past the robot's dodge range along its spawn axis) x
# 0.1 m thick x 2.0 m tall (cannot be ducked under or leapt over). The wall's local
# x runs ALONG the wall; it is placed yaw=0 so the long axis is the env's x axis and
# the thin dimension is lateral (env y) -- a wall "beside" the robot.
WALL_HALF_LENGTH = 1.5
WALL_HALF_THICKNESS = 0.05
WALL_HALF_HEIGHT = 1.0
# Grey-blue so it reads as scenery, distinct from the red ball and the robot.
_WALL_RGBA = (0.4, 0.45, 0.55, 1.0)


def _make_wall_spec(
  half_extents: tuple[float, float, float], rgba: tuple[float, ...]
):
  """Build an ``MjSpec`` for a single free-floating box named ``wall``."""

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    # Freejoint -> floating base (6-DOF), so the reset event can write its pose per
    # env. It is re-pinned every step (pin_wall), so it never actually moves.
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = list(half_extents)
    # contype/conaffinity keep MuJoCo defaults (=1): collides with the robot, the
    # ball, and the terrain.
    geom.rgba = list(rgba)
    geom.name = "wall_collision"
    return spec

  return spec_fn


def get_wall_cfg(
  half_extents: tuple[float, float, float] = (
    WALL_HALF_LENGTH,
    WALL_HALF_THICKNESS,
    WALL_HALF_HEIGHT,
  ),
  rgba: tuple[float, ...] = _WALL_RGBA,
) -> EntityCfg:
  """Return an ``EntityCfg`` for a static wall.

  Returns a fresh cfg each call. The ``init_state`` is a placeholder parked out of
  the way (to the side, base on the ground); ``reset_wall_pose`` overwrites the pose
  every episode, so the initial state is never actually trained from.
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 3.0, half_extents[2]),
      rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spec_fn=_make_wall_spec(half_extents, rgba),
  )
