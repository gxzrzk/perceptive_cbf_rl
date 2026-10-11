"""对抗运动先验（AMP）PPO 强化学习核心算法模块（AmpPPO Algorithm）。

算法架构与工作原理：
`AmpPPO` 继承自现代标准的 `rsl_rl.algorithms.PPO` 算法类，并在保留 PPO 截断代理损失（Clipped Surrogate Loss）、
广义优势估计（GAE）与非对称 Actor-Critic 架构的同时，深度融合了对抗运动先验（AMP）机制：

1. 采样阶段的实时风格奖励注入（Real-time AMP Reward Injection in Rollout）：
   - 在 `act()` 阶段，从环境的 TensorDict 中截取当前时间步的 AMP 观测特征 $s_t$；
   - 在 `process_env_step()` 阶段，提取环境执行动作后的下一时间步 AMP 观测 $s_{t+1}$，并妥善处理环境重置截断；
   - 调用判别器计算当前状态转移对 $(s_t, s_{t+1})$ 的真实感得分并映射为风格奖励 $r^{AMP}$；
   - 将风格奖励与环境外在任务奖励按比例插值融合，替换原生奖励传递给 PPO 价值计算流水线；
   - 将策略生成的状态转移对 $(s_t, s_{t+1})$ 存入经验回放缓冲区（ReplayBuffer）。

2. 训练阶段的联合对抗优化（Joint Adversarial Optimization in Update）：
   - 将策略参数（Actor/Critic）与判别器参数（Trunk/Head）整合至同一个 Adam 优化器中，确保两者在小批次迭代中同频更新；
   - 每个训练批次同步从环境轨迹池、策略经验回放池和专家动捕数据集中采样数据；
   - 判别器采用最小二乘损失（LSGAN）：专家样本目标设为 +1，策略样本目标设为 -1；
   - 在专家样本流形上计算 R1 梯度二范数惩罚（Gradient Penalty），严格约束判别器函数的平滑性；
   - 总损失函数包含：PPO 代理损失 + 状态价值损失 - 策略熵奖励 + AMP 判别损失 + 梯度惩罚损失。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from src.tasks.amp_loco.rl.amp_discriminator import Discriminator
from src.tasks.amp_loco.rl.amp_motion_loader import AMPLoader
from src.tasks.amp_loco.rl.amp_normalizer import Normalizer
from src.tasks.amp_loco.rl.amp_replay_buffer import ReplayBuffer


class AmpPPO(PPO):
    """结合对抗运动先验（AMP）判别器的 PPO 强化学习算法。"""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_loader: AMPLoader,
        amp_normalizer: Normalizer | None,
        amp_storage: ReplayBuffer,
        *,
        amp_loss_coef: float = 1.0,
        amp_grad_pen_lambda: float = 10.0,
        amp_trunk_weight_decay: float = 1.0e-3,
        amp_head_weight_decay: float = 1.0e-2,
        learning_rate: float = 1.0e-3,
        optimizer: str = "adam",
        multi_gpu_cfg: dict | None = None,
        **ppo_kwargs,
    ) -> None:
        """初始化 AmpPPO 算法。

        Args:
            actor: Actor 策略网络模型。
            critic: Critic 状态价值评估网络模型。
            storage: PPO 经验轨迹存储器（RolloutStorage）。
            discriminator: AMP 判别器神经网络。
            amp_loader: 专家动捕数据加载器。
            amp_normalizer: AMP 观测在线均值方差归一化器。
            amp_storage: 策略状态转移经验回放池。
            amp_loss_coef: 判别器损失与梯度惩罚在总损失中的权重系数。
            amp_grad_pen_lambda: 专家样本梯度惩罚系数 lambda。
            amp_trunk_weight_decay: 判别器主干 MLP 的权重衰减系数（L2 正则化）。
            amp_head_weight_decay: 判别器线性输出头的权重衰减系数。
            learning_rate: 初始学习率。
            optimizer: 优化器类型（默认 "adam"）。
            multi_gpu_cfg: 多 GPU 分布式训练配置。
            **ppo_kwargs: 传递给基类 PPO 的其余超参数。
        """
        # 移除上游 rsl_rl 不支持的可选模块配置
        ppo_kwargs.pop("rnd_cfg", None)
        ppo_kwargs.pop("symmetry_cfg", None)
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            learning_rate=learning_rate,
            optimizer=optimizer,
            rnd_cfg=None,
            symmetry_cfg=None,
            multi_gpu_cfg=multi_gpu_cfg,
            **ppo_kwargs,
        )

        self.discriminator = discriminator.to(self.device)
        self.amp_loader = amp_loader
        self.amp_normalizer = amp_normalizer
        self.amp_storage = amp_storage
        self.amp_loss_coef = float(amp_loss_coef)
        self.amp_grad_pen_lambda = float(amp_grad_pen_lambda)

        # 重新创建统一的 Adam 优化器，将判别器参数与策略网络参数合并，
        # 并针对判别器主干与输出头设置独立的权重衰减，保证策略与判别器梯度同频更新。
        policy_params = list(self.actor.parameters()) + list(self.critic.parameters())
        param_groups = [
            {"params": policy_params, "name": "policy"},
            {
                "params": list(self.discriminator.trunk.parameters()),
                "weight_decay": float(amp_trunk_weight_decay),
                "name": "amp_trunk",
            },
            {
                "params": list(self.discriminator.amp_linear.parameters()),
                "weight_decay": float(amp_head_weight_decay),
                "name": "amp_head",
            },
        ]
        self.optimizer = optim.Adam(param_groups, lr=learning_rate)

        # 缓存当前时间步的 AMP 观测，用于与下一时间步拼接成 (s_t, s_{t+1}) 状态对
        self._prev_amp_obs: torch.Tensor | None = None

    # ---------------------------------------------------------------------------
    # 环境交互收集钩子函数 (Rollout Hooks)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """根据当前观测执行推理并输出动作，同时截获并保存当前步的 AMP 观测。

        Args:
            obs: 环境返回的多组观测构成的 TensorDict（必须包含 'amp' 组）。

        Returns:
            策略采样得到的控制动作张量。
        """
        if "amp" not in obs.keys():
            raise KeyError("AmpPPO 要求环境 TensorDict 中必须包含名为 'amp' 的观测组！")
        # 缓存上一时间步的 AMP 特征张量
        self._prev_amp_obs = obs["amp"].detach().clone().to(self.device)
        return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """处理环境单步推进后的反馈信息：计算判别器风格奖励，并将转移对写入缓冲区。

        边界条件特别处理：
        mjlab 环境在触发终止（dones > 0）时会自动在内部执行 reset，导致返回的 `obs["amp"]`
        实际上是新一轮首帧（来自新 episode）而非终止瞬间的真实状态。
        对于发生重置的环境，此处将终止帧近似退化为 `_prev_amp_obs`，防止跨 episode 的伪跳跃转移污染判别器。

        Args:
            obs: 执行动作后环境返回的新观测 TensorDict。
            rewards: 环境计算的原生任务奖励张量。
            dones: 终止标志张量（包含环境超时与摔倒终止）。
            extras: 环境附加信息字典。
        """
        next_amp_obs = obs["amp"].detach().clone().to(self.device)
        if self._prev_amp_obs is None:
            raise RuntimeError("process_env_step 在 act() 捕获 amp 观测之前被调用！")

        next_amp_with_term = next_amp_obs.clone()
        reset_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if reset_ids.numel():
            # 对重置的环境用上一帧状态修正截断，避免跨轮次样本污染
            next_amp_with_term[reset_ids] = self._prev_amp_obs[reset_ids]

        # 计算判别器风格奖励并与原生任务奖励融合（若设置了 task_reward_lerp）
        rewards_dev = rewards.to(self.device)
        amp_reward, _ = self.discriminator.predict_amp_reward(
            self._prev_amp_obs,
            next_amp_with_term,
            rewards_dev,
            normalizer=self.amp_normalizer,
        )
        blended_rewards = amp_reward.to(rewards.device)

        # 将生成的策略样本对 (s_t, s_{t+1}) 存入判别器回放缓冲区
        self.amp_storage.insert(self._prev_amp_obs, next_amp_with_term)
        # 将混合后的最终奖励与观测递交给基类 PPO，完成价值更新与 GAE 优势计算准备
        super().process_env_step(obs, blended_rewards, dones, extras)

    # ---------------------------------------------------------------------------
    # 模式切换与权重保存/加载 (Train / Eval / Save / Load)

    def train_mode(self) -> None:
        """切换模型为训练模式（包括 Actor、Critic 与 Discriminator）。"""
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        """切换模型为评估模式。"""
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        """将算法各模块的权重和状态打包保存为字典，纳入判别器与归一化器状态。"""
        saved = super().save()
        saved["discriminator_state_dict"] = self.discriminator.state_dict()
        saved["amp_normalizer"] = self.amp_normalizer
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """从字典检查点加载策略网络、判别器网络以及在线归一化器的状态。"""
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(
                loaded_dict["discriminator_state_dict"], strict=strict
            )
        if "amp_normalizer" in loaded_dict and loaded_dict["amp_normalizer"] is not None:
            self.amp_normalizer = loaded_dict["amp_normalizer"]
        return load_iteration

    # ---------------------------------------------------------------------------
    # 参数更新主循环 (Update Loop)

    def update(self) -> dict[str, float]:  # noqa: C901
        """执行一个 PPO 学习周期的参数更新，联合优化策略网络与判别器网络。

        优化流程：
        1. 循环遍历若干个学习轮次（epochs）与小批次（mini-batches）；
        2. 同步提取三部分数据：PPO 策略轨迹样本、策略状态转移样本（负样本）和动捕专家转移样本（正样本）；
        3. 计算 PPO 裁剪代理损失（Surrogate Loss）与价值函数均方误差损失（Value Loss）；
        4. 计算判别器最小二乘损失（LSGAN Loss）与专家样本梯度惩罚（Gradient Penalty）；
        5. 联合反向传播计算梯度并裁剪梯度范数，执行优化器参数更新；
        6. 在线更新归一化器的均值与方差统计量；
        7. 在 GPU 端完成标量指标累加，仅在最后通过一次同步返回度量字典，最大化 GPU 利用率。

        Returns:
            包含各项损失值与预测均值的度量指标字典。
        """
        # 在 GPU 设备端累加各项标量损失，避免每次 minibatch 产生 GPU->CPU 阻塞式同步
        mean_value_loss = torch.zeros((), device=self.device)
        mean_surrogate_loss = torch.zeros((), device=self.device)
        mean_entropy = torch.zeros((), device=self.device)
        mean_amp_loss = torch.zeros((), device=self.device)
        mean_grad_pen_loss = torch.zeros((), device=self.device)
        mean_policy_pred = torch.zeros((), device=self.device)
        mean_expert_pred = torch.zeros((), device=self.device)

        # 根据网络类型（前馈/循环）构建 PPO mini-batch 生成器
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        num_updates_total = self.num_learning_epochs * self.num_mini_batches
        # 计算 AMP 小批次容量 = 总采样样本数 / mini-batch 数量
        amp_mb_size = (
            self.storage.num_envs * self.storage.num_transitions_per_env
        ) // self.num_mini_batches

        # 构建策略样本生成器与专家动捕样本生成器
        amp_policy_gen = self.amp_storage.feed_forward_generator(
            num_updates_total, amp_mb_size
        )
        amp_expert_gen = self.amp_loader.feed_forward_generator(
            num_updates_total, amp_mb_size
        )

        for batch, sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_gen, amp_expert_gen
        ):
            original_batch_size = batch.observations.batch_size[0]

            # 优势函数按 mini-batch 进行标准化
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        batch.advantages - batch.advantages.mean()
                    ) / (batch.advantages.std() + 1e-8)

            # 重新计算当前网络权重下的动作对数概率、状态价值和策略分布熵
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
            )
            distribution_params = tuple(
                p[:original_batch_size] for p in self.actor.output_distribution_params
            )
            entropy = self.actor.output_entropy[:original_batch_size]

            # 基于 KL 散度的自适应学习率调节
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # 1. PPO 裁剪代理损失（Surrogate Loss）
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # 2. 状态价值损失（Value Loss，支持带裁剪的价值损失）
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # PPO 基础总损失
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            # 3. AMP 判别器损失计算 (Least-Squares GAN)
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert

            # 在线归一化判别器特征输入
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(
                        policy_next_state, self.device
                    )
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(
                        expert_next_state, self.device
                    )

            # 判别器前向预测：专家目标为 1，策略目标为 -1
            policy_d = self.discriminator(
                torch.cat([policy_state, policy_next_state], dim=-1)
            )
            expert_d = self.discriminator(
                torch.cat([expert_state, expert_next_state], dim=-1)
            )
            expert_loss = nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device)
            )
            policy_loss = nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)

            # 4. 专家样本梯度惩罚（R1 梯度惩罚）
            grad_pen_loss = self.discriminator.compute_grad_pen(
                *sample_amp_expert, lambda_=self.amp_grad_pen_lambda
            )

            # 5. 合并 PPO 损失与 AMP 判别损失
            loss = loss + self.amp_loss_coef * (amp_loss + grad_pen_loss)

            # 反向传播与参数优化
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # 6. 更新在线统计归一化器（纯 GPU 端执行，无主机同步开销）
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.detach())
                self.amp_normalizer.update(expert_state.detach())

            # 累加统计标量
            mean_value_loss += value_loss.detach()
            mean_surrogate_loss += surrogate_loss.detach()
            mean_entropy += entropy.mean().detach()
            mean_amp_loss += amp_loss.detach()
            mean_grad_pen_loss += grad_pen_loss.detach()
            mean_policy_pred += policy_d.mean().detach()
            mean_expert_pred += expert_d.mean().detach()

        n = max(num_updates_total, 1)
        self.storage.clear()

        # 仅在整个更新完成时通过一次 .item() 同步回 CPU
        return {
            "value": (mean_value_loss / n).item(),
            "surrogate": (mean_surrogate_loss / n).item(),
            "entropy": (mean_entropy / n).item(),
            "amp": (mean_amp_loss / n).item(),
            "amp_grad_pen": (mean_grad_pen_loss / n).item(),
            "amp_policy_pred": (mean_policy_pred / n).item(),
            "amp_expert_pred": (mean_expert_pred / n).item(),
        }

    # ---------------------------------------------------------------------------
    # 算法实例工厂函数 (Construct Algorithm)

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "AmpPPO":
        """根据配置字典与环境实例构造完整的 AmpPPO 算法组件。

        实例化流程：
        1. 解析 Actor 与 Critic 类并根据观测空间构建神经网络实例；
        2. 读取 AMP 专用顶层参数（动捕文件路径、跟踪刚体名称、锚点名称、判别器隐藏层配置等）；
        3. 构建专家动捕加载器 AMPLoader、判别器网络 Discriminator、经验回放池 ReplayBuffer 与归一化器 Normalizer；
        4. 实例化 AmpPPO 算法对象并执行可选的 torch.compile 加速编译。
        """
        alg_class: type[AmpPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        cfg["obs_groups"] = resolve_obs_groups(
            obs, cfg["obs_groups"], ["actor", "critic"]
        )
        if "amp" not in obs.keys():
            raise ValueError(
                "AmpPPO 要求环境必须输出 'amp' 观测组！"
            )
        amp_dim = obs["amp"].shape[-1]

        # 提取 AMP 相关的顶层运行配置
        amp_motion_files = cfg.pop("amp_motion_files")
        amp_body_names = tuple(cfg.pop("amp_body_names"))
        amp_anchor_name = cfg.pop("amp_anchor_name")
        amp_reward_coef = float(cfg.pop("amp_reward_coef", 0.1))
        amp_task_reward_lerp = float(cfg.pop("amp_task_reward_lerp", 0.75))
        amp_discr_hidden_dims = tuple(
            cfg.pop("amp_discr_hidden_dims", (1024, 512, 256))
        )
        amp_replay_buffer_size = int(cfg.pop("amp_replay_buffer_size", 100_000))

        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        cfg["algorithm"].pop("share_cnn_encoders", None)

        # 实例化策略网络与价值网络
        actor = actor_class(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            **copy.deepcopy(cfg["actor"]),
        ).to(device)
        print(f"Actor Model: {actor}")
        critic = critic_class(
            obs,
            cfg["obs_groups"],
            "critic",
            1,
            **copy.deepcopy(cfg["critic"]),
        ).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        # 构建 AMP 专用组件：动捕数据加载器、判别器网络、经验回放池与在线归一化器
        robot_entity = env.unwrapped.scene["robot"]
        all_body_names = tuple(robot_entity.body_names)
        amp_loader = AMPLoader(
            motion_file=amp_motion_files,
            body_names=amp_body_names,
            anchor_name=amp_anchor_name,
            all_body_names=all_body_names,
            device=device,
        )
        if amp_loader.observation_dim != amp_dim:
            raise ValueError(
                f"AMP 数据加载器特征维度 ({amp_loader.observation_dim}) 与环境 'amp' 组观测维度 ({amp_dim}) 不匹配！"
                f"请核对 env 配置与 runner 配置中的 amp_body_names 及 amp_anchor_name。"
            )
        discriminator = Discriminator(
            input_dim=amp_dim * 2,
            amp_reward_coef=amp_reward_coef,
            hidden_layer_sizes=amp_discr_hidden_dims,
            device=device,
            task_reward_lerp=amp_task_reward_lerp,
        ).to(device)
        amp_normalizer = Normalizer(amp_dim, device=device)
        amp_storage = ReplayBuffer(amp_dim, amp_replay_buffer_size, device)

        alg = alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=discriminator,
            amp_loader=amp_loader,
            amp_normalizer=amp_normalizer,
            amp_storage=amp_storage,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **cfg["algorithm"],
        )
        # 执行可选的 PyTorch 编译加速（torch.compile）
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
