"""列出已注册强化学习环境任务清单的工具脚本 (List Registered Environments)

本脚本用于查询并展示当前工程与 MjLab 任务注册表 (`mjlab.tasks` 及 `src.tasks`) 中注册的全部 Task ID。

1. 命令行参数解析机制 (tyro CLI):
   - `--keyword`: 可选过滤关键字字符串（不区分大小写），仅展示名称中包含该关键字的任务。

2. 执行流程:
   - 动态导入任务模块，触发注册表注册机制。
   - 调用 `list_tasks()` 获取所有已注册的任务 ID。
   - 使用 `PrettyTable` 进行表格化对齐排版并在终端输出。
"""

import tyro
from prettytable import PrettyTable

import mjlab
import mjlab.tasks  # noqa: F401  # 导入基础任务模块以触发任务注册（未直接调用因此忽略 F401 告警）
import src.tasks
from mjlab.tasks.registry import list_tasks


def list_environments(keyword: str | None = None):
  """列出所有已注册的环境任务
  
  参数:
    keyword: 可选的过滤关键字，用于仅显示匹配特定子串的任务名称。
  """
  table = PrettyTable(["#", "Task ID"])
  table.title = "Available Environments in mjlab"
  table.align["Task ID"] = "l"

  all_tasks = list_tasks()
  idx = 0
  for task_id in all_tasks:
    try:
      # 可选按关键字过滤任务 (Optionally filter by keyword)：
      # 将任务名称与用户输入的关键字均转换为小写比对，支持模糊匹配子串
      if keyword and keyword.lower() not in task_id.lower():
        continue

      table.add_row([idx + 1, task_id])
      idx += 1
    except Exception:
      continue

  print(table)
  if idx == 0:
    msg = "[INFO] No tasks matched"
    if keyword:
      msg += f" keyword '{keyword}'"
    print(msg)
  return idx


def main():
  """命令行主入口，通过 tyro 解析 CLI 参数并调用 list_environments 函数"""
  return tyro.cli(list_environments, config=mjlab.TYRO_FLAGS)


if __name__ == "__main__":
  main()
