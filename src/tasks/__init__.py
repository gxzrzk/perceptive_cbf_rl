"""任务模块初始化入口（Tasks Package Initialization）。

本模块作为整个 `src/tasks` 命名空间的顶层初始化文件，负责通过 `mjlab` 的任务导入工具
自动扫描并导入当前包下的所有子任务包（如 `amp_loco` 等），使得各个子任务中定义的强化学习环境
与配置能够自动注册到全局任务注册表（Task Registry）中。
"""

from mjlab.utils.lab_api.tasks.importer import import_packages

# 排除不需要作为独立任务包导入的工具库与底层 MDP（马尔可夫决策过程）定义包
_BLACKLIST_PKGS = ["utils", ".mdp"]

# 自动递归导入当前目录下的所有子包，忽略黑名单中的包
import_packages(__name__, _BLACKLIST_PKGS)
