"""动态为 mjlab 注入“时间优先”（time-major）观测历史排列机制的运行时重载模块。

背景与设计动机：
在 AMP 及复杂的强化学习运动控制任务中，通常需要将过去多个时间步（H 步）的观测特征进行堆叠（Observation Stacking）。
mjlab 默认采用“特征项优先”（term-major）的排列方式：
    [A_t0, A_t1, ..., A_tH, B_t0, B_t1, ..., B_tH]
然而在序列建模、时序卷积或标准控制策略中，更自然的排列方式通常是“时间优先”（time-major）：
    [A_t0, B_t0, ..., A_t1, B_t1, ..., A_tH, B_tH]
这使得每个时间步内部的所有物理特征紧密相邻。

上游 mjlab 官方库原生并不支持该选项。本项目早期通过在 `mjlab_patch/` 下提供文件补丁并覆盖安装包的方式实现，
但存在侵入性强、依赖环境管理流程的问题。

本模块通过在内存中执行无侵入式的运行时重载（Runtime Overload / Monkey-Patch）：
1. 继承并扩展 `ObservationGroupCfg`，为其新增 `history_ordering` 字段（可选 `"term"` 或 `"time"`）；
2. 继承并重写 `ObservationManager`，当观测组标记为 `"time"` 时，保留每个特征项未展平的时序维度 (B, H, D)，
   在拼接为 (B, H, sum_D) 后重塑为 (B, H * sum_D)，使得各时间步特征交错排列；
3. 将增强后的类重新绑定回所有引用了它们的 mjlab 内部模块中（包括 `manager_based_rl_env`）。
该过程是幂等的（多次导入不重复执行），且不修改磁盘上的任何安装包文件。
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch
from mjlab import managers as _managers_pkg
from mjlab.envs import manager_based_rl_env as _mbenv
from mjlab.managers import observation_manager as _om

# 注入到 ObservationGroupCfg 的属性名称
_HISTORY_ORDERING_ATTR = "history_ordering"


def _already_installed() -> bool:
  """检查时间优先重载补丁是否已经被安装过。

  Returns:
      bool: 若 ObservationGroupCfg 中已包含 history_ordering 字段则返回 True，否则返回 False。
  """
  return _HISTORY_ORDERING_ATTR in {
    f.name for f in dataclasses.fields(_om.ObservationGroupCfg)
  }


def _install() -> None:
  """执行运行时类替换，为 mjlab 注入时间优先观测历史排序支持。"""
  if _already_installed():
    return

  base_group_cfg = _om.ObservationGroupCfg

  @dataclasses.dataclass
  class ObservationGroupCfg(base_group_cfg):  # type: ignore[valid-type, misc]
    """扩展后的观测组配置类，新增历史维度排列选项。"""
    history_ordering: Literal["term", "time"] = "term"
    """当设置了 ``concatenate_terms`` 与 ``history_length`` 时的历史特征排列布局：

    - ``"term"`` (默认): 特征项优先排列 ``[A_t0, A_t1, ..., B_t0, B_t1, ...]``。
    - ``"time"``: 时间优先排列 ``[A_t0, B_t0, ..., A_t1, B_t1, ...]``。
    """

  class TimeOrderingObservationManager(_om.ObservationManager):
    """支持 ``history_ordering="time"`` 观测组的观测管理器子类。"""

    def __init__(self, cfg, env):  # noqa: ANN001
      """初始化观测管理器，并根据时间优先配置调整特征项展平标志及总观测维度。

      Args:
          cfg: 观测组配置字典。
          env: 基础强化学习环境实例。
      """
      # 时间优先排序需要各特征项的历史保留未展平的三维张量 (B, H, D)，
      # 从而在后续拼接时能够交错组织各个时间步。
      # mjlab 的 _prepare_terms 会把组级别的 flatten 标志透传给各个项，
      # 因此在此处将 time 模式的组预先将 flatten_history_dim 置为 False。
      for group_cfg in cfg.values():
        if group_cfg is None:
          continue
        if getattr(group_cfg, _HISTORY_ORDERING_ATTR, "term") == "time":
          group_cfg.flatten_history_dim = False

      super().__init__(cfg, env)

      # 纠正向外报告的组展平观测维度 (H * sum_D,)，以与 compute_group 实际运行时的展平输出匹配
      # （基类针对未展平历史默认计算的是二维维度 (H, sum_D)）。
      for name, group_cfg in self.cfg.items():
        if group_cfg is None:
          continue
        if getattr(group_cfg, _HISTORY_ORDERING_ATTR, "term") != "time":
          continue
        if not self._group_obs_concatenate.get(name, False):
          continue
        term_dims = self._group_obs_term_dim[name]
        if term_dims and len(term_dims[0]) > 1:
          history_len = int(term_dims[0][0])
          feature_sum = sum(int(d[1]) for d in term_dims)
          self._group_obs_dim[name] = (history_len * feature_sum,)

    def compute_group(self, group_name, update_history=False):  # noqa: ANN001
      """计算并返回指定观测组的当前观测张量。

      当观测组为 time-major 模式且返回结果为 3 维张量 (B, H, sum_D) 时，
      将其重塑为 (B, H * sum_D)，完成时序交错排列。
      """
      result = super().compute_group(group_name, update_history)
      group_cfg = self.cfg[group_name]
      if (
        isinstance(result, torch.Tensor)
        and result.dim() == 3
        and getattr(group_cfg, _HISTORY_ORDERING_ATTR, "term") == "time"
      ):
        # 将 (B, H, sum_D) 展平为 (B, H * sum_D)，实现每个时间步内各项特征的交织排列
        result = result.reshape(result.shape[0], -1)
      return result

  # 在所有持有该类引用的模块中重新绑定（Monkey-Patch）。
  # manager_based_rl_env 按值导入了这两个符号并实例化了 ObservationManager，
  # 因此必须对其实施显式重定向。
  for module in (_om, _managers_pkg, _mbenv):
    setattr(module, "ObservationGroupCfg", ObservationGroupCfg)
    setattr(module, "ObservationManager", TimeOrderingObservationManager)


# 模块导入时自动执行安装
_install()
