"""AMP 任务强化学习运行器模块（AMP On-Policy Runner）。

架构与功能定位：
`AMPOnPolicyRunner` 继承自 `MjlabOnPolicyRunner`（其进一步封装自上游 `rsl_rl.runners.OnPolicyRunner`），
专为对抗运动先验（AMP）环境的训练与部署提供以下扩展：

1. 零侵入式的采样流水线对齐：
   仿真环境在单步推进时直接将包含机体运动特征的 `"amp"` 观测组发布在 TensorDict 中。
   由于 `AmpPPO` 在底层重写了 `act()` 与 `process_env_step()`，上层的轨迹采样与环境交互循环完全无需修改，
   保持了与标准 PPO 运行器的一致性。

2. 状态保存与断点续训（Save & Load）：
   在调用 `save()` 时，`AmpPPO` 会自动将其内部的判别器参数 `discriminator_state_dict` 与
   在线归一化器 `amp_normalizer` 打包至模型权重字典中，保证断点恢复时判别器状态丝毫不丢。

3. Wandb 云端同步瘦身优化：
   本地磁盘完整保留所有周期的检查点（`model_{iter}.pt`），但向 Wandb 云端仅覆盖上传一个固定的
   `latest_model.pt`，防止因长时间训练产生数千个历史模型填满云端存储配额。

4. 部署级 ONNX 策略模型导出：
   在保存检查点的同时，自动将前馈策略网络导出为工业标准的 `policy.onnx` 模型，
   并调用 `attach_metadata_to_onnx` 写入环境观测组映射等元数据，便于直接在真机（Unitree G1 SDK）
   或 C++ 推理引擎中脱机加载运行。
"""

import os
import shutil

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

# 导入 AmpPPO 算法类，使得运行器配置中的 class_name 字符串能够被动态解析
from src.tasks.amp_loco.rl.amp_algorithm import AmpPPO as AmpPPO


class AMPOnPolicyRunner(MjlabOnPolicyRunner):
    """专用于 AMP 强化学习训练与策略导出的 On-Policy 运行器。"""

    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        """保存当前学习迭代的模型检查点，上传最新权重至 Wandb，并导出部署用 ONNX 模型。

        Args:
            path: 本地权重文件保存路径（形如 `.../model_1000.pt`）。
            infos: 可选的附加训练统计信息字典。
        """
        # 保存本地包含策略网络、价值网络、判别器以及归一化器的完整检查点
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)

        # 若启用了 Wandb 日志，则仅向云端上传单份 latest_model.pt，节约存储带宽
        if (
            self.logger.logger_type == "wandb"
            and wandb.run is not None
            and self.cfg.get("upload_model", True)
        ):
            latest = os.path.join(os.path.dirname(path), "latest_model.pt")
            shutil.copyfile(path, latest)
            wandb.save(latest, base_path=os.path.dirname(latest))

        # 循环网络或分支深度置信度模型无法直接通过单输入 ONNX 格式导出，跳过导出
        if getattr(self.alg.actor, "is_recurrent", False):
            return

        # 导出前馈 Actor 为 ONNX 格式以供真机或 C++ 仿真端部署
        policy_path = path.split("model")[0]
        filename = "policy.onnx"
        try:
            self.export_policy_to_onnx(policy_path, filename)
            run_name: str = (
                wandb.run.name
                if self.logger.logger_type == "wandb" and wandb.run
                else "local"
            )  # type: ignore[assignment]
            onnx_path = os.path.join(policy_path, filename)
            # 提取环境基础元数据（观测项名称、输入输出维度等）并固化至 ONNX 文件中
            metadata = get_base_metadata(self.env.unwrapped, run_name)
            attach_metadata_to_onnx(onnx_path, metadata)
            if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
        except Exception as e:
            print(f"[WARN] 导出 ONNX 策略模型失败（训练仍将继续）: {e}")
