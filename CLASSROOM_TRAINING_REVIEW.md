# Classroom 多卡训练检查（2026-10-14）

当前多卡启动链路可用，但原 AMP 实现的分布式同步不完整。本次已补齐相关同步；正在运行的进程保持原代码，修改在下次启动时生效。

## 当前运行和速度

检查对象：`logs/rsl_rl/state_classroom/2026-10-14_00-18-58`。

| 项目 | 实际配置 / 观测 |
| --- | --- |
| GPU | 物理 GPU 1、2，RTX PRO 6000 Blackwell，单卡约 96 GB |
| 多卡模式 | torchrunx 两个 worker，RSL-RL 初始化 NCCL |
| 环境数 | 每卡 512，全局 1024；`NUM_ENVS` 是每卡值 |
| Rollout | 每环境 24 步，全局每轮 24,576 个样本 |
| PPO | 5 epochs × 4 minibatches；每卡 minibatch 3072 |
| 控制周期 | timestep 0.005 s × decimation 4 = 0.02 s |
| AMP 数据 | 两个 worker 都成功加载 64 个动作片段 |
| 训练计划 | 40,000 轮；6000 轮投球预热 + 4000 轮递增 |
| GPU 瞬时利用率 | 检查时两卡约 43%，显存约 2–3 GB/卡 |
| 每轮采样耗时 | 最近 100 轮中位数约 1.236 s |
| 每轮学习耗时 | 最近 100 轮中位数约 0.141 s |
| 全局吞吐 | 最近 100 轮中位数约 17,715 环境步/s |

时间统计来自 TensorBoard 的 `Perf/collection_time`、`Perf/learning_time`、`Perf/total_fps`，截至 iteration 2777。采样约占总时间 90%，应优先优化环境步进及 Python/GPU 同步；这些是原运行数据，本次没有测量修复后的长训练吞吐或收敛。

## 已修复

- 判别器原来使用每卡不同的随机初始化，且上游梯度同步只覆盖 actor/critic。现在启动时广播判别器权重，每个 minibatch 将 actor、critic、判别器的梯度一起平均。
- AMP 归一化器原来每卡独立更新，而且误用了已经归一化、裁剪的数据。现在从原始 policy/expert 观测计算全局统计，一个 minibatch 使用一次统计通信；恢复 checkpoint 时将统计量移到当前 worker 的设备并清除缓存。
- AMP 终止帧替换使用 GPU 上的 `torch.where`，去掉每步 `nonzero` 导致的主机同步。
- torchrunx 子进程现在继承 `TMPDIR`、`WARP*`、`WANDB*`、CPU 线程设置和动态库路径。原命令没有转发项目磁盘上的临时目录/Warp 缓存设置。
- 仅 rank 0 初始化录像 renderer；按 `CUDA_VISIBLE_DEVICES` 映射 EGL 物理 GPU，提前设置默认 CUDA 设备。原实现会在其他 worker 初始化无用 renderer，且 EGL local rank 会选到不同的物理卡。
- 保存完整输入 `agent.yaml`。原 runner 构造会消耗 AMP 字段和模型类名，导致日志缺失实际训练设置。
- 训练脚本支持 `VIDEO`、`VIDEO_INTERVAL`、`VIDEO_LENGTH`；公开 `--agent.check-for-nan`，默认仍为 True。

## 建议的加速顺序

1. **先关闭录像并保留其余参数做对照。** `VIDEO=False` 会同时去掉录像期间的渲染以及包装器每一步读取 GPU 结束标志的同步。多卡更新需要等最慢 worker，rank 0 录视频会拖住其他卡。
2. **每卡环境数按 512 → 1024 → 2048 做短跑。** 每个配置预热后统计至少 100 轮，比较全局 FPS、采样时间、学习时间和显存。当前显存余量很大，较大的并行 batch 可能提高 GPU 利用率；收益需实测。
3. **运行稳定后可关闭 rollout NaN 检查。** 原上游每步对每组观测、reward、done 分别检查，会触发多次 GPU 同步。使用 `--agent.check-for-nan False`；它关闭的是输出异常检测，新代码/新 reward 调试阶段保留 True。
4. **继续定位课堂导航的同步和小算子。** `classroom_follow_state()` 的 `update.any()`、路线扩展的 `need.any()`/`unique().tolist()`、路径查询、每个障碍的接触传感器都是可分析对象。先 profile，再改变刷新条件、批量几何计算或内核实现，避免改变重置和路径状态语义。
5. **保持同 NUMA 节点的卡组合。** 当前 GPU 1、2 属于同一 NUMA 节点，没有 NVLink；比跨 NUMA 的组合更合适。无需先增加卡数，先确认每卡负载和吞吐扩展性。

TF32、非确定性 cuDNN 配置已启用；仿真已有 CUDA graph 路径。当前 timestep、decimation、solver iterations、接触容量不建议直接降低：这些会改变控制/碰撞结果，需要额外验证训练质量。单纯增加模型编译也不是优先项，学习更新仅约占 10%；上游不支持 `reduce-overhead` / `max-autotune` CUDA graph 编译模式。

多卡或环境数增加会改变全局 batch：两卡每卡 1024 时，每轮为 49,152 个样本；每卡 2048 时为 98,304。课程按 iteration 推进，同样的 6000 轮预热将对应更多样本。评估加速时同时比较 wall time、全局样本数及任务成功率，不能只比较 iteration/s，也不能假设多卡线性提速。

保留原规模、关闭录像的对照命令：

```bash
CUDA_VISIBLE_DEVICES=1,2 GPU_IDS=all NUM_ENVS=512 VIDEO=False \
  EXP_NAME=state_classroom_fixed_512 ./train_classroom.sh \
  --agent.max-iterations=40000
```

后续将 `NUM_ENVS` 改为 1024、2048，并给各运行不同的 `EXP_NAME`。稳定运行后可追加 `--agent.check-for-nan False`。脚本默认 `GPU_IDS=0`，只用一张卡；显式设置 `GPU_IDS=all` 才使用全部可见卡。

## 验证与限制

- CPU AMP/MDP 回归：252 项通过，31 项跳过（包含 GPU 专用测试）；另有 3 项训练启动回归通过。
- 实际 GPU 1、2 的微型双进程 NCCL 测试通过：不同 seed 初始化后广播一致，梯度平均正确，连续两次完整 PPO update 后 actor/critic/discriminator 参数和 Adam 一阶状态逐项一致，AMP 统计与两卡原始观测的联合参考一致。
- Bash 语法、启动脚本环境变量传参、Tyro 命令行解析、diff 空白检查通过。
- 未中断或重启已有训练，未启动新的长训练；当前 checkpoint 来自旧 AMP 行为。用于比较修复前后的训练效果时，建议新开实验，恢复旧 checkpoint 不会自动纠正已累计的旧归一化统计。
