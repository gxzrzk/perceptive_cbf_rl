"""冻结策略网络加载工具（Frozen Actor Loader）。

功能与应用场景：
在复合控制任务（如躲避球任务中的模式切换与跌倒恢复评测）中，机器人需要在特定场景下调用预训练完成的
基础运动策略（如平地行走策略 `Unitree-G1-AMP-Flat`）：
1. 状态机模式切换（Walk <-> Dodge Mode Switch）：在没有飞球威胁的静止/巡航间歇期，使用冻结的行走 Actor
   执行常规移动；当感知到飞球威胁时无缝切入敏捷躲避策略；
2. 真机部署与恢复基准（Deployment & Recovery Benchmark）：真机或仿真评测中（`--walk-recover` 模式），
   在躲避动作完成后切换回冻结的基础行走策略以使机器人恢复平衡步态；
3. 本模块负责跨任务读取目标策略的强化学习配置参数、动态实例化网络、加载权重检查点，
   并将其锁定为评估模式（eval-mode）和梯度禁用状态（requires_grad=False）。
"""

from __future__ import annotations

import inspect
from dataclasses import asdict

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.utils import resolve_callable, resolve_obs_groups


def load_frozen_actor(
    teacher_task: str,
    checkpoint: str,
    obs: TensorDict,
    num_actions: int,
    device: str,
    obs_set: str = "actor",
) -> tuple[nn.Module, list[str]]:
    """从教师任务（Teacher Task）的配置与权重检查点构建并加载冻结的 Actor 策略网络。

    Args:
        teacher_task: 教师任务注册 ID（例如 "Unitree-G1-AMP-Flat"）。
        checkpoint: 训练好的模型检查点权重文件路径 (.pt)。
        obs: 当前环境输出的示例 TensorDict（必须包含教师 Actor 所需的观测组，如 "actor"）。
        num_actions: 动作输出维度。
        device: 目标加载设备（如 "cuda:0" 或 "cpu"）。
        obs_set: 使用的观测组键名（默认为 "actor"）。

    Returns:
        tuple[nn.Module, list[str]]:
            - actor: 已加载权重、处于 eval 模式且禁用梯度的 PyTorch 神经网络模块；
            - groups: 该 Actor 所消费的观测组名称列表。
    """
    from mjlab.tasks.registry import load_rl_cfg

    # 加载教师任务的强化学习配置
    tcfg = load_rl_cfg(teacher_task)
    actor_cfg = asdict(tcfg.actor)
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    # 仅保留目标网络构造函数显式接受的关键字参数（过滤掉 cnn_cfg、rnn_* 等无关项）
    accepted = set(inspect.signature(actor_cls.__init__).parameters)
    actor_cfg = {k: v for k, v in actor_cfg.items() if k in accepted}

    groups = list(tcfg.obs_groups[obs_set])
    sample = TensorDict(
        {g: obs[g] for g in groups}, batch_size=obs.batch_size
    )
    resolved = resolve_obs_groups(sample, {obs_set: tuple(groups)}, [obs_set])
    # 实例化网络并移至指定设备
    actor = actor_cls(sample, resolved, obs_set, num_actions, **actor_cfg).to(device)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(ck["actor_state_dict"])
    actor.eval()
    # 冻结网络参数，确保推理过程不占用计算图和梯度内存
    for p in actor.parameters():
        p.requires_grad_(False)
    print(
        f"[frozen-actor] 成功从 '{checkpoint}' 加载教师任务 '{teacher_task}' "
        f"(训练迭代步: {ck.get('iter', '?')}); 观测组={groups}"
    )
    return actor, groups
