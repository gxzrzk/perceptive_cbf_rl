"""用于 G1 机器人闪避任务的被投掷球体（躲避球）实体定义 (A thrown ball entity for G1 dodge tasks)。

该球体是一个**无铰接的浮动基座实体** (floating non-articulated entity)：
- 结构上由单个球形几何体构成，以 6 自由度自由关节 (freejoint) 作为根节点，不含任何主动执行器。
- 运动控制机制：在每个强化学习回合重置时，由环境重置事件 `throw_ball`（详见 `amp_loco.mdp.events`）
  直接写入其根节点位姿 (root pose) 和线速度/角速度 (velocity)，使其沿着考虑重力加速度的抛物线
  高速飞向机器人的躯干或骨盆区域。
- 几何与动力学参数：球体半径在构建时确定，但对外暴露为形参，以便支持未来的课程学习与域随机化 (Domain Randomization)；
  默认质量设定为真实海绵躲避球的约 136g。由于重力加速度恒定，飞行动轨迹在无空气阻力下与质量无关，
  但该质量决定了与机器人发生碰撞时的冲量传递 (collision impulse)。
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg

# 默认躲避球半径（单位：米）：6 英寸直径的球体 -> 半径约为 0.0762 m。
# 暴露为配置项以支持后续多环境下的球体尺寸域随机化 (Domain Randomization)。
DEFAULT_BALL_RADIUS = 0.0762

# 默认球体固定质量（单位：千克）：真实海绵躲避球约重 136 g (0.136 kg)。
# 直接显式指定质量（而非通过密度反算），使得各尺寸球体均保持该质量基准。
# 抛物线轨迹由初速度与重力加速度直接决定，与质量无关；质量仅影响碰撞冲量及漂浮重力补偿力。
DEFAULT_BALL_MASS = 0.136

# 球体渲染颜色 RGBA：偏红色调 (0.9, 0.2, 0.2, 1.0)，便于在点云、相机图像及 MuJoCo 视口中与机器人本体和地面区分。
_BALL_RGBA = (0.9, 0.2, 0.2, 1.0)


def _make_ball_spec(radius: float, mass: float, rgba: tuple[float, ...]):
  """构建单个名为 `ball` 的自由漂浮球体 MuJoCo 模型规范闭包 (Build MjSpec closure for a free-floating sphere)。

  参数:
      radius: 球体半径 (单位: 米)。
      mass: 球体物理质量 (单位: 千克)，MuJoCo 将自动推导其对角转动惯量。
      rgba: 视觉渲染的 RGBA 四元组，各通道范围 [0.0, 1.0]。

  返回:
      spec_fn: 无参可调用对象，调用时返回构建完成的 `mujoco.MjSpec` 实例。
  """

  def spec_fn() -> mujoco.MjSpec:
    # 创建 MuJoCo 声明式规范对象
    spec = mujoco.MjSpec()
    # 在世界坐标系下新增名为 "ball" 的独立刚体
    body = spec.worldbody.add_body(name="ball")
    # 添加自由关节 (freejoint) -> 赋予其 6-DOF 自由浮动基座能力，使得能够在运行时直接写入根位姿和线/角速度
    body.add_freejoint()
    # 为刚体添加球形几何体 (geom)
    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    # geom.size 对于球体：[半径, 0, 0]
    geom.size = [radius, 0.0, 0.0]
    # 固定质量，MuJoCo 会根据球体几何形状与质量自动计算球体的转动惯量 (inertia)
    geom.mass = mass
    # 设置外观颜色
    geom.rgba = list(rgba)
    # 几何体碰撞名称，用于在碰撞回调与感知过滤中定位
    geom.name = "ball_collision"
    # contype 与 conaffinity 保持 MuJoCo 默认值 (=1)，使球体能够与机器人各连杆及地面正常发生弹性碰撞
    return spec

  return spec_fn


def get_ball_cfg(
  radius: float = DEFAULT_BALL_RADIUS,
  mass: float = DEFAULT_BALL_MASS,
  rgba: tuple[float, ...] = _BALL_RGBA,
) -> EntityCfg:
  """获取抛掷球实体的配置对象 `EntityCfg` (Return an EntityCfg for a thrown ball)。

  每次调用均返回全新的 `EntityCfg` 实例，以避免在多个环境之间共享配置导致的状态污染。
  初始状态 `init_state` 仅作为占位符放置在世界原点前方；每个回合重置时，环境中的
  `throw_ball` 重置事件都会重写其位置与初速度，因此该初始状态不会直接参与策略训练。

  参数:
      radius: 球体半径 (m)，默认为 `DEFAULT_BALL_RADIUS` (0.0762 m)。
      mass: 球体质量 (kg)，默认为 `DEFAULT_BALL_MASS` (0.136 kg)。
      rgba: 渲染颜色四元组，默认为红色 `_BALL_RGBA`。

  返回:
      EntityCfg: 包含初始位姿占位符与模型构建规范闭包的实体配置。
  """
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(1.0, 0.0, 1.0),      # 初始位置占位：(x=1.0m, y=0.0m, z=1.0m)
      rot=(1.0, 0.0, 0.0, 0.0),  # 四元数朝向 (w=1, x=0, y=0, z=0)
    ),
    spec_fn=_make_ball_spec(radius, mass, rgba),
  )
