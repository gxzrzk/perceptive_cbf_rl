"""教室静态环境场景、家具视觉模型与保守碰撞盒定义 (Classroom scenery, furniture visuals, and conservative collision boxes)。

该模块主要用于构建结构化室内教室/走廊环境，用于测试 Unitree G1 仿人机器人在静态障碍物复杂场景下的
导航感知策略与基于控制屏障函数 (Control Barrier Function, CBF) 的安全避障能力。

核心设计与机制：
1. **世界固定布景 (World-welded scenery)**:
   通过自定义 `StaticClassroomEntity` 绕过 `mjlab` 默认针对固定基座自动生成的 mocap 包装器，
   杜绝不必要的自由度或 mocap 刚体开销。
2. **场景布局与缩放参数 (Layout & Scaling)**:
   教室场景采用规则的多行多列桌椅分布。通过 `CLASSROOM_LAYOUT_SCALE` 将布局中心占地面积缩小 20%，
   在保持家具自身真实尺寸不变的同时压缩过道宽度（最窄走廊约 1.80 m），提升避障控制难度。
3. **视觉与碰撞分离的复合几何体 (Visuals & Collision Separation)**:
   桌子和椅子由木质桌面/椅面、金属椅腿等轻量视觉几何体组装（contype=0, conaffinity=0，不参与物理碰撞），
   同时外包一层透明的保守长方体包围盒 (`name="wall_collision"`)，与墙体任务共用碰撞命名规范，
   使得 Wall CBF 安全评估算法能够直接无缝复用。
"""

import mujoco
from mjlab.entity import Entity, EntityCfg

# ==============================================================================
# 教室布局拓扑与航路规划常量 (Classroom layout, safe spawns and route generation)
# ==============================================================================

# 布局缩放因子：将基于墙中心的占地面积缩小 20% (即面积变为 0.8 倍，坐标线性缩放 sqrt(0.8))
# 保持桌椅本体大小不变，从而使通道更加狭窄，增加导航避障挑战
CLASSROOM_LAYOUT_SCALE = 0.8 ** 0.5

# 教室排布总行数 (桌椅排数)
CLASSROOM_ROWS = 5

# 教室三列桌椅在 Y 轴（横向）的中心坐标 (m)，经过比例缩放
CLASSROOM_COLUMN_Y = tuple(y * CLASSROOM_LAYOUT_SCALE for y in (-5.3, -1.9, 1.9))

# 教室第一排行走起始基准 X 坐标 (m)
CLASSROOM_ROW_START = 3.5 * CLASSROOM_LAYOUT_SCALE

# 行与行之间的中心轴间距 (Row pitch, m)
CLASSROOM_ROW_PITCH = 3.8 * CLASSROOM_LAYOUT_SCALE

# 教室在 X 轴（纵向）的有效安全边界 [x_min, x_max] (m)
CLASSROOM_X_BOUNDS = tuple(x * CLASSROOM_LAYOUT_SCALE for x in (-1.7, 23.4))

# 教室在 Y 轴（横向）的有效安全边界 [y_min, y_max] (m)
CLASSROOM_Y_BOUNDS = tuple(y * CLASSROOM_LAYOUT_SCALE for y in (-6.5, 3.1))

# 机器人安全生成车道 (Spawn lanes)：位于相邻两列桌椅之间的两条主纵向过道中心 Y 坐标 (m)
CLASSROOM_SPAWN_LANES = (
  (CLASSROOM_COLUMN_Y[1] + CLASSROOM_COLUMN_Y[2]) / 2,  # 第一通道 Y 坐标
  (CLASSROOM_COLUMN_Y[0] + CLASSROOM_COLUMN_Y[1]) / 2,  # 第二通道 Y 坐标
)

# 教室巡航航路的关键路标站点 X 坐标集合 (Route stations)
# 包含起始点 (0.0)、缩放后各行过道横向交汇点、以及末端终点 (20.9 * scale)
CLASSROOM_ROUTE_STATIONS = (
  0.0,
  *(CLASSROOM_ROW_START + (row + 0.5) * CLASSROOM_ROW_PITCH - 0.4
    for row in range(CLASSROOM_ROWS - 1)),
  20.9 * CLASSROOM_LAYOUT_SCALE,
)

# 机器人根节点路径规划的安全转弯净空半径 (Clearance, m)；缩小后最窄过道约 1.80 m，设置 0.7 m 保留充裕安全裕度
CLASSROOM_ROUTE_CLEARANCE = 0.7


class StaticClassroomEntity(Entity):
  """世界坐标固定的教室布景实体 (World-welded scenery entity)。

  绕过 mjlab 对固定基座实体自动包装为 mocap body 的默认逻辑，
  严格确保模型中不含任何动关节 (joints) 或 mocap 刚体，仅作为纯静态刚体存在于物理世界中。
  """

  def _build_spec(self) -> None:
    """构建并校验 MuJoCo 规范对象，断言其无关节与 mocap 刚体。"""
    self._spec = self.cfg.spec_fn()
    if self._spec.joints or any(body.mocap for body in self._spec.bodies):
      raise ValueError("Static classroom scenery must have no joints or mocap bodies (静态教室布景严禁包含关节或 mocap 刚体)")


class StaticClassroomEntityCfg(EntityCfg):
  """静态教室布景实体的配置类 (EntityCfg subclass for StaticClassroomEntity)。"""

  def build(self) -> StaticClassroomEntity:
    """实例化并返回 `StaticClassroomEntity`。"""
    return StaticClassroomEntity(self)


def get_static_wall_cfg(half_extents, rgba):
  """构建具有与墙体任务相同碰撞体命名的世界固定长方体配置 (World-welded box with wall collision naming)。

  参数:
      half_extents: 长方体在局部 x, y, z 轴的半长尺寸 (half_length, half_width, half_height)。
      rgba: 几何体材质颜色四元组。

  返回:
      StaticClassroomEntityCfg: 静态刚体配置对象。
  """
  def spec_fn():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    # 命名为 wall_collision，使得基于名称匹配的 CBF 屏障约束和射线感知逻辑能自动识别该障碍
    body.add_geom(
      name="wall_collision",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half_extents,
      rgba=rgba,
    )
    return spec

  return StaticClassroomEntityCfg(spec_fn=spec_fn)


def get_furniture_cfg(kind):
  """获取特定类型教室家具（课桌 desk 或椅子 chair）的实体配置。

  家具采用“保守包围盒 + 精细视觉部件”的设计：
  - 外层包围盒命名为 `wall_collision`，透明不可见 (alpha=0)，负责物理碰撞检测并被 Wall CBF 识别；
  - 内部添加桌面、椅面、椅背及 4 根金属细腿等精美视觉元素 (木纹/金属色)，不参与碰撞计算 (contype=0, conaffinity=0, mass=0)。

  参数:
      kind: 家具类型字符串，仅支持 "desk"（课桌）或 "chair"（椅子）。

  返回:
      StaticClassroomEntityCfg: 包含分层几何体与合适离地初始高度的配置对象。
  """
  if kind not in ("desk", "chair"):
    raise ValueError(f"Unknown furniture kind: {kind}, must be 'desk' or 'chair'")

  # 保守碰撞盒的半长半宽半高 (half-extents, m)
  # 课桌: 0.8m x 1.2m x 0.76m (半尺寸 0.4, 0.6, 0.38)
  # 椅子: 0.5m x 0.5m x 0.90m (半尺寸 0.25, 0.25, 0.45)
  half = (0.4, 0.6, 0.38) if kind == "desk" else (0.25, 0.25, 0.45)

  def spec_fn():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="wall")
    # 添加用于物理碰撞和 CBF 安全距离计算的透明外包络碰撞盒
    body.add_geom(
      name="wall_collision",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=half,
      rgba=(0, 0, 0, 0),  # 完全透明
    )

    def box(name, size, pos, color):
      """辅助函数：添加不参与碰撞与惯量计算的纯视觉长方体组件。"""
      body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        pos=pos,
        rgba=color,
        contype=0,      # 关闭物理碰撞发射
        conaffinity=0,  # 关闭物理碰撞接收
        mass=0,         # 不计入物理刚体质量
      )

    # 视觉材质颜色：木质褐色与金属深灰
    wood, metal = (0.65, 0.42, 0.22, 1), (0.22, 0.25, 0.28, 1)

    if kind == "desk":
      # 课桌：桌面面板 + 四条金属桌腿
      box("top", (0.4, 0.6, 0.025), (0, 0, 0.355), wood)
      for i, x in enumerate((-0.33, 0.33)):
        for j, y in enumerate((-0.53, 0.53)):
          box(f"leg_{i}_{j}", (0.025, 0.025, 0.355), (x, y, -0.025), metal)
    else:
      # 椅子：座垫 + 靠背 + 四条金属椅腿
      box("seat", (0.25, 0.25, 0.025), (0, 0, 0), wood)
      box("back", (0.025, 0.25, 0.21), (-0.225, 0, 0.24), wood)
      for i, x in enumerate((-0.20, 0.20)):
        for j, y in enumerate((-0.20, 0.20)):
          box(f"leg_{i}_{j}", (0.02, 0.02, 0.2125), (x, y, -0.2375), metal)
    return spec

  # 初始位置的 Z 轴偏置设为 half[2]，使家具底面恰好贴合地面 (z=0)
  return StaticClassroomEntityCfg(
    spec_fn=spec_fn,
    init_state=EntityCfg.InitialStateCfg(pos=(0, 0, half[2])),
  )
