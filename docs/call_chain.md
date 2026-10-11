# PAC-MAN 启动流程与核心调用链路分析

> 本文档基于对仓库源码（`src/`、`scripts/`、`deploy/`）与依赖库（mjlab v1.5.3、rsl_rl 5.4.0，位于 `.venv/`）的实际阅读梳理而成，描述训练 / 播放 / 部署三条启动链路、`env.step()` 的完整调用链，以及核心数据对象在各模块间的读写职责。

---

## 0. 总体架构：四层结构

```
┌─ 入口层   train_dodge_*.sh / play_*.sh / deploy/play_real_dodge.sh
├─ 脚本层   scripts/train.py, scripts/play.py, scripts/dodge_benchmark.py
│            （tyro CLI 解析 → 从任务注册表取配置）
├─ 环境层   mjlab.ManagerBasedRlEnv（8 个 Manager 管线）
│            + 本仓库 mdp/ 模块（CBF、投球事件、躲避奖励、深度观测）
└─ RL 层    rsl_rl.OnPolicyRunner → MjlabOnPolicyRunner → AMPOnPolicyRunner
             → AmpPPO（PPO + AMP 判别器 + 专家动作加载器 + 回放缓冲）
```

任务注册发生在 `src/tasks/amp_loco/config/g1/__init__.py`：`register_mjlab_task(task_id, env_cfg, rl_cfg, runner_cls=AMPOnPolicyRunner)`，共注册 5 个任务（`Unitree-G1-AMP-Flat`、3 个 MimicKit 状态基线、1 个 BallOnly 深度任务）。`import src.tasks` 即完成注册。

另一个启动期的隐藏动作：`src/tasks/amp_loco/__init__.py` 导入 `observation_history_ordering`，它在**运行时给 mjlab 打猴子补丁**（子类化 `ObservationGroupCfg`/`ObservationManager` 并重新绑定到所有引用处），为观测历史增加 time-major 排列 `[A_t0, B_t0, A_t1, ...]`。这相当于环境层的"中间件注入"，必须在任何任务 config 构造之前执行。

---

## 1. 训练启动流程（以 `./train_dodge_state.sh` 为例）

1. **`train_dodge_state.sh`（shell 入口）**
   1.1. 设置环境变量与默认值：`TASK`、`NUM_ENVS=8192`、`STAND_RATIO=0.2`、`INPLACE_RATIO=0.4`、`EXP_NAME`。
   1.2. `PYTHONPATH=$REPO_ROOT uv run python scripts/train.py $TASK --env.scene.num-envs=... --agent...`（仓库从源码运行，`pyproject.toml` 中 `package = false`，uv 只装依赖）。
2. **`scripts/train.py::main()`**
   2.1. `import mjlab.tasks` + `import src.tasks` → 触发历史排序补丁安装 + 任务注册（见第 0 节）。
   2.2. `tyro.cli` 两段解析：第一个位置参数选任务，剩余参数按 `TrainConfig`（`env: ManagerBasedRlEnvCfg` + `agent: RslRlBaseRunnerCfg`）覆盖。
3. **`launch_training()`**
   3.1. 创建日志目录 `logs/rsl_rl/<experiment_name>/<时间戳>[_<run_name>]`。
   3.2. `select_gpus()` → 写 `CUDA_VISIBLE_DEVICES`、`MUJOCO_GL=egl`、`MUJOCO_EGL_DEVICE_ID`。
   3.3. 单 GPU/CPU：直接调 `run_train()`；多 GPU：`torchrunx.Launcher` 为每卡起一个 worker 进程，各自执行 `run_train()`。
4. **`run_train()`（单进程训练体）**
   4.1. `configure_torch_backends()`；按 `LOCAL_RANK` 设置 device 与 seed。
   4.2. `ManagerBasedRlEnv(cfg=cfg.env, device, render_mode)` —— **环境实例化（核心）**：
       4.2.1. `Scene(cfg.scene)`：合并 MJCF（G1 机器人 + `ball` 实体 + 墙 + 地形），收集传感器（IMU、足底接触、`ball_robot_contact` 接触传感、`head_depth_single` 相机），生成 warp 化模型 spec。
       4.2.2. `Simulation(...)`：构建 MuJoCo-Warp 并行仿真（`num_envs` 个环境批量实例）。
       4.2.3. `scene.initialize(...)`：把 model/data 句柄回填给各 entity 和 sensor。
       4.2.4. `load_managers()`，**加载顺序有硬性依赖**：
             - `EventManager`（先于一切，域随机化需要）→ 立即执行 `startup` 模式事件：
               - `init_motion_loader` → `MotionResetManager.init()`：加载 `src/assets/motions/g1/amp_dodge*` 下的 `.npz` 专家动作（RSI 复位分布）；
               - `dr.geom_friction / encoder_bias / body_com_offset`：一次性物理参数随机化。
             - `CommandManager`（先于观测，因为观测项引用命令）→ 构造 `DodgeGoToGoalCommand`（含 CBF 状态暂存区，见第 4 节 C）。
             - `ActionManager`（`JointPositionAction`，或 `CBF_JOINT=1` 时 `CbfJointPositionAction`）。
             - `ObservationManager`（actor / critic / amp / depth / ball_state / critic_dodge 各组，惰性实例化 `ManagerTermBase` 类如 `BallOnlyDepthObs`）。
             - `TerminationManager`、`RewardManager`、`CurriculumManager`、`MetricsManager`、`RecorderManager`。
       4.2.5. `_configure_gym_env_spaces()`：推导 action/obs 空间维度。
   4.3. （可选）`VideoRecorder` 包装 env（rank 0 才录）。
   4.4. `RslRlVecEnvWrapper(env, clip_actions)`：适配 rsl_rl 的 `VecEnv` 接口，把 obs dict 包成 `TensorDict`、clip 动作、合并 terminated/truncated 为 `dones`。
   4.5. `runner = AMPOnPolicyRunner(env, agent_cfg, log_dir, device)`：
       4.5.1. 父类 `OnPolicyRunner.__init__`：先 `env.get_observations()` 拿一份 obs 用于推导网络输入维度。
       4.5.2. `resolve_callable(cfg.algorithm.class_name)` → 字符串 `"src.tasks.amp_loco.rl.amp_algorithm:AmpPPO"` 解析成类 → `AmpPPO.construct_algorithm(obs, env, cfg, device)`，按顺序实例化：
             - `actor` / `critic`：`MLPModel`（512,256,128，actor/critic 的 `obs_groups` 决定各自输入拼接，深度任务 actor=proprio+depth，critic 再加特权组 `critic_dodge`）；
             - `RolloutStorage`（on-policy 轨迹缓冲）；
             - `AMPLoader`：读取 runner cfg 顶层 `amp_motion_files` 指向的专家 `.npz`（深度/躲避任务用 `amp_dodge` = Leap+Dodge 合集）；
             - `Discriminator`（1024,512,256 trunk + amp_linear 头）；
             - `Normalizer`（AMP 观测在线归一化）、`ReplayBuffer`（AMP 转移回放，容量 100k）；
             - 重建 `optim.Adam`：三组参数（policy / amp_trunk / amp_head，后两组带 weight decay）——策略与判别器同一步锁步更新。
       4.5.3. `Logger`（wandb / tensorboard）。
   4.6. rank 0 落盘 `params/env.yaml`、`params/agent.yaml`；如有 resume 则 `runner.load(checkpoint)`。
   4.7. `runner.learn(num_learning_iterations=25000, init_at_random_ep_len=True)` → 进入第 2 节主循环。

---

## 2. 训练主循环（`OnPolicyRunner.learn`，AmpPPO 各钩子插入点）

每轮 iteration（`inference_mode` 下）：

1. **Rollout**：循环 `num_steps_per_env`（默认 24）步：
   1.1. `actions = alg.act(obs)`
       1.1.1. `AmpPPO.act`：先把 `obs["amp"]` 克隆到 `self._prev_amp_obs`（用于配对 (s,s') 喂判别器），再 `super().act(obs)` → actor MLP 采样高斯动作。
   1.2. `obs, rewards, dones, extras = env.step(actions)`（wrapper clip 动作后进入第 3 节的 env.step 链）。
   1.3. `alg.process_env_step(obs, rewards, dones, extras)`
       1.3.1. `AmpPPO`：取 `obs["amp"]` 作 next_amp_obs（done 的 env 用 pre-step obs 顶替终止帧）；`discriminator.predict_amp_reward(prev, next, task_reward)` → 按公式 `reward = (1-lerp)*coef*disc + lerp*task` 混合出 blended reward（lerp=0.5, coef=0.5，保持风格奖励约占 50%）；`amp_storage.insert(prev, next)`；再把 blended reward 交给 `super().process_env_step`（推入 RolloutStorage、记录 value/entropy 等）。
2. `alg.compute_returns(obs)`：GAE 广义优势估计（含 time-out bootstrap）。
3. `alg.update()`：`AmpPPO.update` 复刻 PPO minibatch 循环，额外叠加 AMP 损失：
   3.1. 策略损失（clipped surrogate）+ value 损失 + 熵。
   3.2. AMP 损失：从 `amp_storage` 采样 policy (s,s') 对、从 `amp_loader` 采样专家 (s,s') 对 → 判别器 BCE + WGAN 风格梯度惩罚（`amp_grad_pen_lambda=10`）。
   3.3. 一次 Adam step 同步更新 actor/critic/discriminator（锁步）。
4. `logger.log(...)`；每 `save_interval`（1000）轮 `runner.save()`：
   4.1. `AmpPPO.save()`：在 PPO 状态基础上追加 `discriminator_state_dict`、`amp_normalizer`。
   4.2. `AMPOnPolicyRunner.save()`：本地保留全部 `model_*.pt`，wandb 只上传一份覆盖式 `latest_model.pt`；非 recurrent actor 额外导出 `policy.onnx`（挂 metadata），供 deploy 栈使用。

---

## 3. `env.step(action)` 完整调用链（每 50Hz 控制步）

`ManagerBasedRlEnv.step()` 是所有模块的汇合点。mjlab 没有传统 Web 式"中间件"，但 **Manager 管线就是等价物**：动作先经过 ActionManager 变换写入仿真，仿真结果再按固定顺序流经 Termination → Reward → Reset → Command → Event → Sense → Observation 六道"中间件"，最后产出 obs/reward/dones。

1. `action_manager.process_action(action)` —— `JointPositionAction`：暂存原始动作，计算关节位置目标（`default_offset + scale * action`，scale=0.25）。
2. **decimation 循环（4 × 5ms 物理子步）**：
   2.1. `action_manager.apply_action()`：把关节目标写入 robot entity。
   2.2. `scene.write_data_to_sim()`：所有 entity 的写操作（含上一步投球写入的球状态）落到 warp data。
   2.3. `sim.step()`：MuJoCo-Warp `mj_step` 并行积分（重力、接触、关节动力学）。
   2.4. `scene.update(dt)`：把 qpos/qvel 等回读到 `entity.data`（球的 `root_link_pos_w / lin_vel_w` 在这里刷新），传感器缓存失效。
   2.5. `metrics_manager.compute_substep()`。
3. `episode_length_buf += 1`，`common_step_counter += 1`。
4. `termination_manager.compute()`：逐项调用（短路 or 合并）：
   4.1. `time_out`（`time_out=True`，不计失败）。
   4.2. `bad_orientation`（>70°）、`bad_base_height`（<0.5m）、`collapsed_crouch`。
   4.3. `ball_contact`（`ball_hit`）：读 `ball_robot_contact` 接触传感；附带速度突变回退（球速偏离纯重力抛物线 > 1.5m/s 且在 1m 内 → 判穿隧命中）。训练中经课程 `enable_after_steps=200k` 后才启用 -200 终止惩罚。
   4.4. （wall 任务）`wall_contact`。
5. `reward_manager.compute(dt)`：按 cfg 权重加权求和，核心项：
   5.1. `track_anchor_linear/angular_velocity`：跟踪 `twist` 命令（可能已是 CBF 过滤后的安全命令）。
   5.2. `dodge_link_cbf_reward`（**部署配置的核心安全奖励**）：对每个机器人 link 求 `h = ‖p_ball − p_link‖ − (r_ball + r_link + margin)`、闭速率 `ḣ`，取 `min(ḣ + αh, 0)`（仅负值=违约罚），`reduce="min"` 只罚最紧 link；门控在球飞行中。特权量（真值位置），actor 只看深度。
   5.3. `mimickit_dodge_reward`：MimicKit 式骨盆躲避奖励。
   5.4. `dodge_cbf_reward`（基座级 2D CBF，读取命令项暂存状态，见 4C）：`min(v·e + αh, 0) + (exp(−‖v−v_safe‖²/σ²) − 1)`。
   5.5. 正则项：`is_terminated(-200)`、`action_rate_l2`、`joint_acc_l2`、`joint_pos_limits`、`feet_slip`、`self_collision_cost`、`leap_flight` 等。
6. **auto-reset**：对 done 的 env → `recorder_manager.record_pre_reset` → `_reset_idx(env_ids)`（第 3.1 节）→ `scene.write_data_to_sim()`。
7. `sim.forward()`：**单次**重算所有派生量（xpos/xquat/cvel/sensordata），同时覆盖非复位 env 的物理结果与复位 env 的新状态（代码注释明确说明：用一次 forward 换取终止/奖励读到的"滞后一个子步"的一致性）。
8. `command_manager.compute(dt)` → `DodgeGoToGoalCommand._update_command()`：
   8.1. `super()._update_command()`（`GoToGoalCommand`）：P 控制器把 goal 误差转成 body 系速度命令 `vel_command_b`；dwell 到期重采样 goal；standing/inplace env 强制零速。
   8.2. `predictive_dodge_filter()`（`mdp/cbf.py`，纯函数）：预测球的 xy 直线轨迹（重力不影响 xy），构造相对阶 1 的 barrier `h = s − D`（s = 机器人对球航线的垂直偏移），按 `u·e ≥ max(α(D−s), (D−s)/t_usable)` 做闭式半平面投影，叠加**反应时间感知**（TTI − 0.5s 动量起步 − 0.3s 舒适缓冲）；返回 `(u_safe_w, threat, perp_signed, h, e, tti)`。
   8.3. 威胁出现瞬间 latch 逃离方向 `_dodge_side`（±1，防抖动）；安全命令 clamp 回速度包络。
   8.4. **暂存 CBF 奖励状态** `_dodge_e_w/_dodge_h/_dodge_u_safe_w/_dodge_tti/_dodge_threat`（供奖励与 critic 观测读取，见第 4 节 C）。
   8.5. `cbf_filter_command=True` 时用 `u_safe_b` 覆盖 `vel_command_b`（play 与非深度任务）；`=False`（CBF-RL 训练模式）时命令保持 nominal，CBF 只出奖励信号，部署时可整个丢弃。
9. `event_manager.apply("step")`：
   9.1. `throw_ball_on_dwell`：按触发器（训练=球落地即重投 + 站立 dwell；MimicKit 基线=1–4s 随机倒计时；play=落地即重投）决定 `fire` 掩码 → `solve_ballistic_velocity()` 解出抛物速度（平抛 vz0=0、飞行时间 0.58–0.63s 即反应窗口、lead_target 前置瞄准 + aim noise；50% 低弧高抛到 0.9–1.3m 触发 duck）→ 写球的 root pos/vel（经 entity → 下一个子步的 `write_data_to_sim` 生效）。
   9.2. （gimbal 任务）`aim_camera_at_ball`：oracle 瞄准相机俯仰。
   `event_manager.apply("interval")`：`push_robot` 等周期扰动。
10. `sim.sense()`：渲染传感器——含 `head_depth_single` 相机的 depth+segmentation 离屏渲染（EGL），IMU/接触传感读取。
11. `observation_manager.compute(update_history=True)`：每组过一遍管线 **compute → noise → clip → scale → delay → history**（time-major 排列由补丁实现）：
    11.1. `actor` 组：proprio 6 项（base_ang_vel、projected_gravity、command、joint_pos/vel、actions）×4 帧历史 + 噪声（enable_corruption=True 模拟感知噪声）。
    11.2. `critic` 组：actor 全部 + 线速度 + 全身 body pos/ori（无噪声，特权）。
    11.3. `amp` 组：body_pos/ori/lin_vel/ang_vel（判别器输入）。
    11.4. `depth` 组：`BallOnlyDepthObs._raw_depth()`：读相机 depth + segmentation → 只保留球 geom 像素、其余置 far → DR（整球丢弃 2%、像素丢弃 10%、深度抖动 ±0.15m、边缘 tile 闪烁、静态 clutter blob）→ 父类 `DepthImageObs` 做 clamp[0.1,5]/归一化/按 `update_period` 保持/帧偏移堆叠 (0,3,8,18) → flatten 144 维。
    11.5. （状态基线）`ball_state`：`dodge_ball_state_b` 真值球相对 pos+vel；`critic_dodge`：`dodge_cbf_state_b`（CBF 特权状态）。
12. 返回 `(obs_buf, reward_buf, terminated, time_outs, extras)`；`RslRlVecEnvWrapper` 把 obs dict 包成 `TensorDict`、合并 dones → 回到第 2 节的 1.3。

### 3.1 复位链（`_reset_idx(env_ids)`，被 step 第 6 步或 `env.reset()` 调用）

1. `curriculum_manager.compute(env_ids)`：`goal_radius_curriculum` 等收窄采样范围。
2. `sim.reset(env_ids)` / `scene.reset(env_ids)`：清物理状态。
3. `event_manager.apply("reset")`：
   3.1. `reset_scene_to_default`：回 XML 默认位。
   3.2. `reset_from_motion_data` → `MotionResetManager.reset()`：从 amp_dodge 动作库随机帧做 RSI 复位（关节角+速度，~46% 单脚/腾空起姿）。
   3.3. `reset_dodge_state`：球停到 `(0, 3, 0.1)` 停车位，清 dwell 计数与投球冷却。
   3.4. `randomize_ball_size`：`dr.geom_size` 球半径 7.5–12.5cm（CBF 与 critic 都会读到）。
   3.5. （wall 任务）`reset_wall_pose`、`reset_walk_path`。
4. 依次 `observation/action/reward/metrics/curriculum/command/event/termination_manager.reset(env_ids)`：清观测历史环、episode 累计量、命令计时器。
5. `episode_length_buf[env_ids] = 0`。

---

## 4. 播放与部署启动链

### 4.1 Sim 播放（`play_vision.sh` → `scripts/play.py::run_play`）

1. shell 设 `CAMERA_TILT_DEG`（必须与 ckpt 训练一致，从路径自动推断）、`BALLONLY_AUG`、帧偏移；无 ckpt 参数则 `--agent zero` 站桩。
2. `load_env_cfg(task, play=True)` 拿 **play_env_cfg**（与训练的差异：stand_ratio=0、`back_offset=0.5` 即持续倒退保持运动模式、hit 终止立即启用、DR 关闭）。
3. cfg 变换：`--depth-frame-offsets` 对齐网络输入维度；`--reset-stand` 把 RSI 复位换成默认站立；`--depth-aug` 重开训练 DR；`--zed-camera` 硬件在环——**猴替换 `zed_sensor._compute_data`**，用真实 ZED 深度（3% soft-min 池化到 16×9）替换仿真相机，可把仿真球 min-pool 叠加进真实画面。
4. `ManagerBasedRlEnv` → `RslRlVecEnvWrapper` → `runner_cls(env, cfg)` → `runner.load(ckpt)` → `policy = runner.get_inference_policy(device)`（eval 模式 + obs 归一化器）。
5. （默认）导出 ONNX；`_DodgePlayViewer(ViserPlayViewer).run()`：每步 `policy(obs) → env.step`，网页面板含深度图与 Dodgeball 控制夹（Pause throws / Throw overhead / Throw underbody，本质是直接改 `env._dodge_throw_paused/_dodge_throw_once` 两个裸标志，`throw_ball_on_dwell` 每步读取）。

### 4.2 硬件部署（`deploy/play_real_dodge.sh`，独立 Python 3.8 uv 项目）

三进程经 localhost UDP 同步：ZED+EfficientTAM 球分割节点 → ONNX 策略进程（自动 walk↔dodge 模式切换，dodge_link_cbf.onnx + walk_policy.onnx）→ Unitree DDS 硬件桥。策略输入即训练时 `BallOnlyDepthObs` 所模仿的"球掩码深度"表征，sim2real 差距被压缩到感知层。

---

## 5. 核心数据对象的读写职责分析

本仓库类间频繁传递的对象主要有五个，按执行顺序逐一说明谁初始化、谁只读、谁改写。

### A. `obs: TensorDict`（含 actor/critic/amp/depth/… 组）

| 环节 | 操作 |
|---|---|
| `ObservationManager.compute()` | **唯一写入者**：每步（及每次 reset）从零计算各组，做噪声/裁剪/历史堆叠后写入 `env.obs_buf`；`RslRlVecEnvWrapper` 再包成 TensorDict |
| `AmpPPO.act()` | 只读：取 actor 组喂策略、克隆 amp 组存 `_prev_amp_obs` |
| `AmpPPO.process_env_step()` | 只读：取 amp 组作 next_obs 配对；**修改的是 rewards 张量**（判别器混合替换），不改 obs |
| `RolloutStorage` | 只读：按引用存储各步 obs |
| `play` 的 `policy(obs)` | 只读：推理 |
| 训练循环 `learn()` | 只读：跨步传递 |

结论：obs 是**一次性快照**，生成后无人原地修改；所有"变化"都来自下一轮 compute。这使 AMP 的 (s, s') 配对语义干净。

### B. `env: ManagerBasedRlEnv`（god-object，所有 mdp 项的第一个参数）

| 模块 | 对 env 的操作 |
|---|---|
| `run_train/run_play` | **初始化者** |
| 全部 obs/reward/termination 项函数 | 只读（`env.scene[...]`、`env.sim.mj_model`、`env.command_manager`） |
| `throw_ball_on_dwell`、`reset_*`、`dr.*` 事件 | **写入**：经 entity API 写球/机器人 root state、joint state、geom 参数（缓存为 `_dodge_link_radii`、`_dodge_throw_cd` 等裸属性挂在 env 上） |
| `DodgeGoToGoalCommand` | 写自己的 `vel_command_b`（挂在命令项上，不在 env 上） |
| `play.py` / viser viewer | 写控制标志（`_dodge_throw_paused` 等）并猴替换传感器 `_compute_data` |

它是典型的"上下文总线"：读多写少，写集中在事件（reset/step 模式）与 play 注入点。

### C. CBF 暂存状态（`DodgeGoToGoalCommand` 上的 `_dodge_threat/_dodge_e_w/_dodge_h/_dodge_u_safe_w/_dodge_tti/_dodge_ball_radius`）

这是**跨 Manager 传递最频繁的数据**：CommandManager 每步计算，RewardManager 与 ObservationManager 消费。

| 沿执行顺序 | 模块 → 操作 |
|---|---|
| ① 初始化 | `DodgeGoToGoalCommand.__init__()`：全部置零/默认 |
| ② 每步写入 | `_update_command()`（step 第 8 步）：调用 `predictive_dodge_filter` 后刷新全部六个字段；`_dodge_ball_radius` 从 `sim.model.geom_size` 读每 env 真值 |
| ③ 只读（奖励） | `dodge_cbf_reward()`：读 threat/e/h/u_safe 构造 CBF-RL 奖励（式 22-23）；`track_anchor_linear_velocity(track_dodge_safe=True)` 可改读安全命令 |
| ④ 只读（观测） | `dodge_cbf_state_b()`：把 e/h/tti/radius 组成特权 critic 观测 `critic_dodge` 组 |
| ⑤ 只读（监控） | `metrics["cbf_active_frac"/"cbf_min_h"]`、viser 调试可视化 |

谁都不在 ②③④ 之外改它；`cbf_filter_command=False` 时它**纯粹是奖励/观测的数据源**，不影响策略可见的命令——这正是论文"训练期安全塑形、部署期零开销丢弃"的实现机制。

### D. 球状态（`ball.data.root_link_pos_w / root_link_lin_vel_w` + per-env `geom_size`）

| 模块 | 操作 |
|---|---|
| `throw_ball_on_dwell`（事件） | **写入起点**：设球的初始位置与抛物速度（`solve_ballistic_velocity`） |
| 物理仿真 `sim.step → scene.update` | **持续写入**：每子步积分刷新 |
| `randomize_ball_size`（reset 事件） | 写半径 `geom_size` |
| `predictive_dodge_filter`（命令） | 只读：pos/vel 预测航线 |
| `dodge_link_cbf_reward` / `mimickit_dodge_reward`（奖励） | 只读：全 link 距离 |
| `dodge_ball_state_b`（obs，状态基线） | 只读 |
| `ball_contact`（终止） | 只读（接触传感 + 速度突变） |
| `BallOnlyDepthObs`（obs，深度任务） | 间接只读：经相机 segmentation 找球像素 |

### E. 专家动作数据（`.npz` → `AMPLoader` / `MotionResetManager`）

初始化者有两处：startup 事件 `init_motion_loader`（供 RSI 复位）与 `AmpPPO.construct_algorithm`（供判别器训练），二者指向同一目录但实例独立。此后只读：`MotionResetManager.reset()` 抽帧复位，`AmpPPO.update()` 抽专家 (s,s') 对算判别器损失。全生命周期只读。

---

## 6. 关键文件速查

| 文件 | 角色 |
|---|---|
| `scripts/train.py` / `play.py` | 训练/播放入口（tyro CLI → registry → env+runner 装配） |
| `src/tasks/amp_loco/config/g1/__init__.py` | 任务注册表（5 个 task_id → env_cfg + rl_cfg + AMPOnPolicyRunner） |
| `config/g1/dodge_env_cfgs.py` | 躲避任务 env 工厂：加球/投球事件/CBF 命令/深度 obs/奖励组装 |
| `config/g1/rl_cfg.py`, `goto_rl_cfg.py` | PPO+AMP 超参、obs_groups、动作数据目录 |
| `mdp/cbf.py` | `predictive_dodge_filter`：预测式 sidestep CBF（纯函数） |
| `mdp/goal_command.py` | `GoToGoalCommand`/`DodgeGoToGoalCommand`：导航层 + CBF 滤波 + 奖励状态暂存 |
| `mdp/rewards.py` | `dodge_cbf_reward`、`dodge_link_cbf_reward`（部署核心）、MimicKit 奖励 |
| `mdp/events.py` | 投球 `throw_ball_on_dwell`、RSI 复位、球尺寸随机化、墙/路径 |
| `mdp/observations.py` | `DepthImageObs`/`BallOnlyDepthObs`（sim2real 桥）、球状态/特权 obs |
| `mdp/cbf_joint.py`, `actions_cbf.py` | Joint-CBF 动作滤波（CBF_JOINT=1 备选配置） |
| `rl/amp_algorithm.py` | `AmpPPO`：act/process_env_step/update 三钩子 + 判别器锁步训练 |
| `rl/runner.py` | `AMPOnPolicyRunner`：save 时导出 ONNX |
| `observation_history_ordering.py` | 运行时给 mjlab 打的 time-major 历史排列补丁 |
| `deploy/` | 硬件栈：ZED+EfficientTAM / ONNX 策略 / Unitree DDS 三进程 |

---

## 7. 一句话总结

**启动**：shell → `scripts/train.py` → 注册表取 cfg → `ManagerBasedRlEnv`（Scene→Simulation→按依赖序装 8 个 Manager，startup 事件加载 AMP 动作）→ `RslRlVecEnvWrapper` → `AMPOnPolicyRunner`（`AmpPPO.construct_algorithm` 建 actor/critic/判别器/专家加载器/回放）→ `learn()`。
**每步**：action 经 ActionManager 写入仿真 → 4×`mj_step` → Termination→Reward→auto-reset→`sim.forward`→CommandManager（`DodgeGoToGoalCommand` 内跑 `predictive_dodge_filter` 并暂存 CBF 状态）→EventManager（投球）→`sim.sense`（含深度+分割渲染）→ObservationManager（compute→noise→clip→delay→history）产出 TensorDict obs → `AmpPPO` 用判别器混合奖励并配对 (s,s')。
**数据流**：obs 是只读快照；env 是读多写少的上下文总线（写集中在事件）；CBF 暂存状态由命令项每步唯一写入、奖励与特权观测只读；球状态由投球事件+物理写入、被 CBF/奖励/观测/终止四方只读；专家动作数据仅初始化后只读。
