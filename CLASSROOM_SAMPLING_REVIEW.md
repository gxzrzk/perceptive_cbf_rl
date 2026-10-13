# Classroom 采样性能检查（2026-10-14）

主要瓶颈是 Python 调度、大量小 GPU 算子和主机同步。当前最有收益的配置调整是增大每卡环境数；本轮同时落实了执行器和课堂导航的冗余计算优化。

## 测量方式

在空闲的物理 GPU 4（RTX PRO 6000 Blackwell）上使用同一份 `model_3000.pt`，固定 timestep/decimation/solver、开启 NaN 检查和 rank-0 episode bookkeeping，关闭录像。预热 48 步，计时 240 步；另做 cProfile 96 步和可选 PyTorch profiler 4 步。计时与 profiler 分开，表格时间不包含初始化、加载动作数据或 profiler 开销。

这是单卡采样基准，包括 actor/critic forward、环境步进、AMP reward/transition 和 episode bookkeeping，不进行 PPO update 或 NCCL 通信，不修改 checkpoint。物理仿真和浮点运算存在短跑波动；不同运行的 episode 结束数略有差异，结果不等同于长期收敛或完整多卡训练提速。

## 实测结果

| 配置 | 每个向量环境步耗时 | 采样吞吐（环境步/s） |
| --- | ---: | ---: |
| 本轮优化前，512 环境 | 39.28 ms | 13,033 |
| 最终优化后，512 环境 | 35.57 ms | 14,396 |
| 最终优化后，512 环境，关闭 NaN 检查 | 35.02 ms | 14,620 |
| 最终优化后，1024 环境 | 35.18 ms | 29,103 |
| 最终优化后，2048 环境 | 41.11 ms | 49,818 |

在 512 环境的这次前后对照中，最终代码每步时间减少约 9.5%，吞吐提高约 10.5%。中间版本测到 33.61 ms，说明单次速度有波动，不应把最佳单次数字当作稳定收益。

优化后从 512 增加到 1024，采样吞吐约为 2.02 倍；2048 约为 3.46 倍。1024 环境与 512 的步耗时几乎一样，说明扩大 batch 能摊薄相当一部分固定调度开销。

4 步 PyTorch profiler 的对照：

| CUDA 调用 | 优化前 | 最终优化后 |
| --- | ---: | ---: |
| kernel 启动 | 25,284 | 23,314 |
| 流同步 | 1268 | 807 |
| 异步内存复制 | 2176 | 1551 |

流同步减少约 36%。原 cProfile 中，导航读取 2304 次/96 步，即每步约 24 次；执行器控制与导航是较明显的 Python 热点。Profiler 耗时包含工具本身的额外成本，嵌套函数的累计时间不能相加。

## 已落实的代码优化

- **执行器固定参数不再每个子步创建 CUDA 标量。** 刚度、阻尼和 effort limit 改用等价的标量算术/裁剪，去掉每个 actuator group 每个 physics substep 的重复张量创建、复制与同步。
- **执行器直接读取自己控制的关节。** 缓存该组关节的全局 qpos/qvel 地址，避免先收集整个机器人所有关节，再做第二次子集索引。读取的仍是实时仿真状态；entity indexing 更换时刷新缓存。
- **导航同一步重复读取快速返回。** 路线写入/扩展维护 CPU 版本号，奖励、指标、命令、观测读到未变化的导航状态时无需重新读取 heading 或将 GPU epoch 比较同步到 CPU。保留部分重置后的逐环境刷新；无版本号的手工路径保留原 epoch 检查。
- **导航全环境刷新不构造无用掩码和零张量。** 全量更新直接快照状态；只有同一步部分重置才保存其他环境并做 masked merge。历史清零改用广播 `masked_fill_`。
- **先判断课堂模式再做 GPU 条件判断。** 课堂已经有自己的路线扩展，跳过普通 walk-path 分支中无用的 `need.any()` 同步。

物理 timestep、decimation、solver、接触容量、奖励公式和重置规则保持原配置。执行器数值等价测试使用 `1e-6` 容差；没有要求仿真轨迹逐 bit 一致。

## 配置建议与后续优化

1. **先用每卡 1024、关闭录像做完整多卡对照。** 本机单卡采样几乎无需增加步时间即可处理两倍环境。每卡 2048 的采样收益也明确，但 PPO minibatch、全局 batch 和课程对应的样本量都会更大，需要同时看学习时间和成功率。
2. **优先保留 NaN 检查。** 单独关闭它，512 环境测到 35.02 ms/步、14,620 步/s，相对开启只提高约 1.6%，处于短跑波动范围。它不是当前主要瓶颈，当前没有充分性能理由优先关闭。
3. **下一步可合并纯张量计算的 GPU kernel。** 执行器扭矩曲线/摩擦、课堂路径投影的相邻段循环、局部连接障碍检测都是候选。目前即使优化后仍有约 5800 次 kernel 启动/步；需要用等价测试覆盖转弯、回退、部分 reset 和索引顺序。
4. **批量查询与缓存静态路径几何。** `walk_path_query()` 每次整理 padding 和搜索数据；同一轮导航会查询 lookahead、3 个 join 候选和 2 个曲率窗口端点。可以合成批量查询，把固定曲率等几何量只在路线写入/扩展时更新。需覆盖扩容、部分重置和路线延长的缓存失效。
5. **减少按步日志复制。** 上游 logger 每步执行 done `nonzero()`，并将完成 episode 的 reward/length 复制到 CPU；可延迟到 rollout 末尾批量处理。命令 reset 也逐项 `.item()`，可合并复制。应保留 episode 顺序、窗口长度和报告指标。
6. **投球事件可以推迟无用计算。** timed-throw 模式下每步计算的 `base_speed` 没有参与触发条件，可移到需要它的模式；计时器分支和末尾的 `nonzero()` 可进一步合并检查。不能无条件重新采样 countdown，否则会改变随机数序列和投球行为。

两卡启动示例（不影响已有运行）：

```bash
CUDA_VISIBLE_DEVICES=1,2 GPU_IDS=all NUM_ENVS=1024 VIDEO=False \
  EXP_NAME=state_classroom_sampling_1024 ./train_classroom.sh \
  --agent.max-iterations=40000
```

`NUM_ENVS` 是每卡值。两卡每卡 1024 意味着总计 2048 环境、每轮 49,152 个样本；2048 则为总计 4096 环境、每轮 98,304 个样本。以上命令保留 NaN 检查；课程仍按 iteration 推进。当前正在运行的旧进程不会加载这些代码修改。

## 验证和复现

CPU 回归 253 项通过、31 项跳过；真实 GPU 回归 52 项通过，覆盖执行器控制、关节读取、课堂静态碰撞、观测与 actor/critic forward、导航奖励和部分重置。新增执行器测试与已有导航测试共同验证行为；原运行没有被中断。

复现基准：

```bash
CUDA_VISIBLE_DEVICES=4 TMPDIR="$PWD/.cache/tmp" WARP_CACHE_PATH="$PWD/.cache/warp" \
  .venv/bin/python scripts/profile_classroom_sampling.py \
  --num-envs 1024 --steps 240 --torch-profile-steps 4 \
  --checkpoint logs/rsl_rl/state_classroom/2026-10-14_00-18-58/model_3000.pt \
  --output .cache/classroom_sampling.json
```

使用当前可用 GPU 替换示例 GPU 4。原始数据保存在 `.cache/sampling_before_512.json`、`.cache/sampling_after_512.json`、`.cache/sampling_final_512.json`、`.cache/sampling_final_512_no_nan.json`、`.cache/sampling_final_1024.json`、`.cache/sampling_final_2048.json`，对应 `.log` 保存详细 profiler 表。
