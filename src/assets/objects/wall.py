"""用于 G1 机器人带障碍闪避任务的静态墙壁实体 (A static wall entity for G1 dodge-with-obstacle tasks)。

墙体用于在物理空间上对机器人的闪避机动施加严格几何边界约束：
- 墙体通常放置在机器人初始站立位置的侧方，迫使机器人在躲避迎面高速飞来的躲避球时，
  不能盲目侧移或后撤而撞击墙面，从而训练具备环境感知与控制屏障函数 (CBF) 约束的全身协调闪避动作。

核心架构与动力学机制：
- **为何采用浮动实体 (floating entity + freejoint) 而非世界焊接刚体 (welded static body)？**
  `mjlab` 在并行向量化仿真环境中，会将所有焊接在世界坐标系 (worldbody) 上的静态几何体编译入全局共享模型，
  导致无法针对每一个独立的子环境 (env) 随机化设置墙体的位置和朝向。
  为了支持环境级别的域随机化（如每局随机决定墙在机器人的左侧还是右侧、随机化距离）：
  1. 将墙壁构建为带有 6-DOF 自由关节 (freejoint) 的独立浮动刚体；
  2. 在每个回合重置时，通过 `reset_wall_pose` 事件为每个环境独立写入墙体的随机根位姿；
  3. 在仿真的每一步计算中，通过 `pin_wall` 事件强制将墙体的线/角速度清零并锁定位置，
     从而在运动学上实现严格的“绝对刚性静止”——机器人和球体与其碰撞时会产生弹性反弹，但墙体本身绝不产生任何位移。
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg

# 墙体半尺寸参数 (half-extents, 单位: 米): (半长, 半厚度, 半高)。
# 实际全尺寸为：
# - 长度 3.0 m (半长 1.5 m)：在机器人生成轴向前后充分延展，覆盖机器人的主要闪避机动范围；
# - 厚度 0.1 m (半厚 0.05 m)：纤薄但具有足够碰撞刚度；
# - 高度 2.0 m (半高 1.0 m)：高于机器人身高，杜绝跳跃翻越或下蹲钻过的投机解。
# 墙体的局部坐标系 X 轴沿其长度方向；当 yaw=0 时长轴平行于环境 X 轴，薄厚度方向沿环境 Y 轴（即位于机器人侧面）。
WALL_HALF_LENGTH = 1.5
WALL_HALF_THICKNESS = 0.05
WALL_HALF_HEIGHT = 1.0

# 墙体渲染颜色 RGBA：灰蓝色调 (0.4, 0.45, 0.55, 1.0)，在视觉上明确呈现为背景静态障碍，与红色的躲避球及机器人本体形成鲜明反差。
_WALL_RGBA = (0.4, 0.45, 0.55, 1.0)


def _make_wall_spec(
  half_extents: tuple[float, float, float], rgba: tuple[float, ...]
):
  """构建单个带有自由关节的长方体墙壁 MuJoCo 规范闭包 (Build MjSpec closure for a free-floating box wall)。

  参数:
      half_extents: 墙体在局部 x, y, z 轴的半长、半宽、半高三元组 (half_length, half_thickness, half_height)。
      rgba: 视觉渲染的 RGBA 四元组。

  返回:
      spec_fn: 无参可调用对象，调用时返回配置好的 `mujoco.MjSpec` 实例。
  """

  def spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    # 添加自由关节 (freejoint) -> 赋予 6-DOF 自由度，使各并行环境能独立通过重置事件设置其位姿。
    # 结合每步物理循环中的 pin_wall 事件清零速度，使其表现为绝对刚性静止障碍。
    body.add_freejoint()
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = list(half_extents)
    # 碰撞参数 contype/conaffinity 保持 MuJoCo 默认值 (=1)，使墙壁能与机器人连杆、球体及地面正常发生碰撞。
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
  """获取静态障碍墙体的实体配置对象 `EntityCfg` (Return an EntityCfg for a static wall)。

  每次调用均返回独立的 `EntityCfg` 新实例。配置中的 `init_state` 仅为默认停放占位符
  （放置在侧方 y=3.0m，底部贴地 z=half_height）；在每个训练回合开始时，
  `reset_wall_pose` 事件均会根据任务配置重写其位姿，因此占位状态不会直接作为训练初始位姿。

  参数:
      half_extents: 墙体半尺寸，默认为 (1.5, 0.05, 1.0) 米。
      rgba: 材质颜色四元组，默认为灰蓝色 `_WALL_RGBA`。

  返回:
      EntityCfg: 包含初始位姿占位符与模型规范构建闭包的实体配置对象。
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 3.0, half_extents[2]),  # 占位位置：x=0, y=3m, 底部着地 z=half_extents[2]
      rot=(1.0, 0.0, 0.0, 0.0),          # 占位朝向：四元数无旋转
    ),
    spec_fn=_make_wall_spec(half_extents, rgba),
  )
