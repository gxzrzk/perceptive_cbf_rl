"""导航与速度指令基类模块 (Command Terms).

在强化学习任务架构中，指令管理器 (Command Manager) 负责为智能体生成任务目标。
该模块提供速度指令和目标生成器的基础类型定义与几何工具支持，
其具体实现衍生于 `goal_command.py`（如 `GoToGoalCommand` 与 `DodgeGoToGoalCommand`）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer