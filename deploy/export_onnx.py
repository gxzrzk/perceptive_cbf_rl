"""开发机专用的 ONNX 策略导出工具（将强化学习训练的 .pt 权重转换为 .onnx 模型）。

在部署架构中的角色：
-------------------
本脚本仅在开发机 (Dev Box, x86_64 + GPU) 上执行，**严禁在实机 Jetson 上运行**！
因为本脚本会导入完整 mjlab、Warp 和 MuJoCo 环境依赖，而 Jetson 部署环境完全不包含这些大依赖。

导出机制与核心逻辑：
-------------------
1. 封装了 ``runner.export_policy_to_onnx`` 导出机制：
   构建一个单环境 (num_envs=1) 的 CPU 仿真环境，将用户指定的深度堆叠帧参数 ``--frame-offsets``
   注入环境配置中，确保 Actor 网络输入维度与检查点完全吻合。
2. 固化观测归一化器 (Observation Normalizer)：
   ``runner.export_policy_to_onnx`` 会将训练过程中动态统计的经验归一化器参数
   （均值、方差等 Running Mean/Std）作为常量层直接烘焙进导出的 ONNX 计算图中。
   因此实机部署推理时，部署代码只需直接输入原始物理尺度的观测，而无需在 Python 端重复计算复杂的观测归一化。
3. 输出文件保存：
   导出的模型自动以 ``<checkpoint_stem>.onnx`` 命名并存放在检查点同级目录下，随后可复制到实机 ``deploy/ckpts/``。

用法示例：
---------
    uv run python deploy/export_onnx.py <task_id> <checkpoint.pt> [--frame-offsets '(0,3,8,18)']

例如：
    uv run python deploy/export_onnx.py Unitree-G1-AMP-Dodge-Depth-Single-Flat \\
        logs/rsl_rl/g1_amp_dodge_depth_single/<run>/model_24999.pt
"""

import argparse
import ast
import os
import sys
from dataclasses import asdict
from pathlib import Path

# 将仓库根目录添加至 Python 模块搜索路径
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    """解析命令行参数，构建单智能体环境，加载 PyTorch 权重并导出固化归一化器的 ONNX 模型。"""
    parser = argparse.ArgumentParser(description="将训练好的躲避策略检查点导出为 ONNX 模型 (开发机专用)")
    parser.add_argument("task_id", help="注册的任务 ID (如 Unitree-G1-AMP-Dodge-Depth-Single-Flat)")
    parser.add_argument("checkpoint", help="待导出的 PyTorch 检查点文件路径 (.pt)")
    parser.add_argument(
        "--frame-offsets", default="(0,3,8,18)",
        help="训练该检查点时采用的深度图时间堆叠步数偏移元组 (Python 字面量元组形式)。"
             "用于配置 env.depth_frame_offsets，使导出的 Actor 输入维度完全匹配。",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"错误: 检查点文件不存在: {args.checkpoint}")
        sys.exit(1)

    # 导入任务配置，自动将 G1 躲避任务注册进 mjlab 任务注册表中 (触发任务注册 register tasks)
    import src.tasks.amp_loco.config.g1 as _g1  # noqa: F401 # 模块导入副作用：执行任务装饰器注册
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = "cpu"  # 导出过程在 CPU 上进行，保证跨平台兼容性
    offsets = tuple(ast.literal_eval(args.frame_offsets))

    print(f"正在加载任务配置: {args.task_id}  (frame_offsets={offsets})")
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner

    # DepthImageObs 观测项在实例化时读取 env_cfg.depth_frame_offsets，故必须在 build 前配置
    env_cfg.depth_frame_offsets = offsets
    env_cfg.scene.num_envs = 1  # 导出仅需单环境

    # 构建 MuJoCo 环境并用 RslRl 向量化包装器封装
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 初始化算法 Runner 并加载训练好的模型权重
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    print(f"正在加载检查点: {args.checkpoint}")
    runner.load(args.checkpoint)

    ckpt = Path(args.checkpoint).resolve()
    onnx_path = ckpt.with_suffix(".onnx")
    # runner.export_policy_to_onnx 会将经验观测归一化器作为算子直接烘焙进 ONNX 模型内
    # 实机部署时直接喂入未归一化的物理观测即可
    runner.export_policy_to_onnx(str(onnx_path.parent), onnx_path.name)
    print(f"成功导出 ONNX 模型至: {onnx_path}")


if __name__ == "__main__":
    main()
