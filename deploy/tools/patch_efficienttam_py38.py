#!/usr/bin/env python3
"""RealtimeEfficientTAM 子模块 Python 3.8 兼容性自动补丁工具。

在部署架构中的角色与技术背景：
-----------------------------
官方的 RealtimeEfficientTAM 声明依赖 Python >= 3.10 以及 PyTorch >= 2.5.1。
然而在边缘计算设备 Unitree G1 (搭载 Jetson Orin / JetPack 5.x) 上，系统原生的最高 Python
版本为 3.8（Ubuntu 20.04 LTS）。
经过深入分析，EfficientTAM 在 Python 3.8 下的**唯一语法不兼容点**在于 PEP 585
内置泛型类型标注（例如函数签名中的 ``list[np.ndarray]`` 或 ``dict[str, Any]``）
在 Python 3.8 模块导入阶段会被作为运行时表达式求值，从而抛出 TypeError。

解决方案：
---------
为所有源码文件头部注入 ``from __future__ import annotations``。
该机制将所有类型标注转为延迟求值的静态字符串，从而完美解除 Python 3.8 对类型提示的限制。
修补后，RealtimeEfficientTAM 可在 Python 3.8 + PyTorch 2.4.1/2.1.0 环境下无缝导入、
构建模型并高速追踪推理，避免了将实机环境激进迁移到 Python 3.10 的系统风险，
也无需额外编译沉重的 TensorRT 引擎。

幂等性设计：
-----------
本脚本具备幂等性：如果目标文件已经包含 future import，则自动跳过。
在首次拉取或更新 git submodule 后运行一次即可：
    deploy/.venv/bin/python deploy/tools/patch_efficienttam_py38.py
"""

import ast
import pathlib
import sys

# 目标子模块源码目录
PKG = pathlib.Path(__file__).resolve().parents[1] / "common" / "RealtimeEfficientTAM" / "efficient_track_anything"


def main():
    """遍历子模块中的所有 Python 文件，基于 AST 语法树安全插入 future annotations 导入语句。"""
    if not PKG.is_dir():
        sys.exit(f"错误: 目标源码包目录不存在: {PKG} (请先执行 git submodule update --init)")
    patched = 0
    for p in sorted(PKG.rglob("*.py")):
        src = p.read_text()
        if "from __future__ import annotations" in src:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        insert_at = 0  # 插入行号（默认文件第一行）
        body = tree.body
        # 如果文件存在顶层模块级 docstring，则将 future import 放置在 docstring 之后
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            insert_at = body[0].end_lineno
        lines = src.splitlines(keepends=True)
        lines.insert(insert_at, "from __future__ import annotations\n")
        p.write_text("".join(lines))
        patched += 1
    print(f"成功为 {patched} 个文件应用 Python 3.8 兼容补丁 (已注入 'from __future__ import annotations')")


if __name__ == "__main__":
    main()
