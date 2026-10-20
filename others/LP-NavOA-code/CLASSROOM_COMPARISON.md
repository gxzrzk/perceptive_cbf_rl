# LP-NavOA × PAC-MAN classroom 对比实验

目的：在**同一个 classroom 场景**下比较 LP-NavOA 与 PAC-MAN（perceptive_cbf_rl）
两个模型的导航表现。本文档记录框架搭建的全部信息，训练启动前请先读「待决事项」。

## 组成

### 场景移植（本仓库新增）

- `src/mjlab/tasks/velocity/config/obstacle_avoidance/g1_raycast_circular_n3_v3/classroom_env_cfgs.py`
  - `ClassroomTerrainCfg`：procgen 地形，几何与 PAC-MAN 的
    `src/assets/objects/classroom.py` 完全一致（4 围墙 + 5 行×3 列课桌椅，
    1.5 m 过道，房间 9.76×17.16 m）。坐标系旋转 90°（LP 前向 v = PAC x）。
  - 命令速度 0.6 m/s 巡航（0.4–0.8 采样），与 PAC-MAN classroom 对齐。
  - **禁用了速度自适应 waypoint 生成**（`velocity_adaptive_offsets=False`）——
    其墙体检测会把课桌误判为墙；waypoint 为显式直线车道序列。
- 注册任务：`Mjlab-Classroom-G1-v6`（教师，蒸馏用）、
  `Mjlab-Classroom-NoPP-G1-v6`（无 planner waypoint，部署形态评测用）。
- 模块 docstring 里有升级路径和注意事项，改动前先读。

### 训练流水线

`run_classroom_curriculum.sh`（仓库根）：完整复刻官方
`src/mjlab/tool/curriculum_train_v6.sh` 的 stage 1/2/3/5/6（官方 env 数与迭代数），
追加 **stage 7 = classroom 蒸馏**（1024 envs / 20000 迭代，从 stage 6 的 indoor
规划器热启动）。

启动（在本目录下）：

```bash
bash run_classroom_curriculum.sh
```

产物：backbone → `logs/rsl_rl/g1_velocity_raycast_h1/<ts>/model_*.pt`；
各规划器 → `logs/steering/local_planner_<name>_final.pt`。

### 运行纪律（重要）

本拷贝与原项目 `/home/zxg/LP-NavOA-code` **共享 venv**（`.venv` 是符号链接），
且 venv 内 `mjlab.pth` 指向原项目源码。因此在本目录下必须：

```bash
export PYTHONPATH=$(pwd)/src
uv run --no-sync ...    # 绝不使用裸 uv run / uv sync（会改写共享 venv 的 mjlab.pth）
```

驱动脚本已内置这两条，手跑单条命令时自己加上。

## 状态

- [x] 场景移植 + 几何校验（导出 XML 逐项核对桌椅/墙坐标尺寸）
- [x] 任务注册（`list_envs` 可见两个 classroom 任务）
- [x] smoke test 全链路通过（64 envs × 30 迭代 backbone → 200 迭代蒸馏 →
      NoPP 测试模式；产物在 `logs_smoketest/`，仅作流程验证，策略无效）
- [ ] 完整训练（未启动，见待决事项）

## 硬件实测（本机 RTX 3060 Laptop 6GB 显存 + 7.4GB 内存 WSL2）

| 规模 | 结果 |
|---|---|
| 1024 envs | ✅ 正常 |
| 4096 envs | ✅ 正常（stage 2/3/5/6/7 官方规模均为此值或以下） |
| 8192 envs（stage 1 官方值） | ❌ 启动即被系统 OOM kill（exit 137） |

结论：**只有 stage 1 的 8192 超出本机上限**。若在本机训练，把脚本里
`STAGE1_NUM_ENVS` 改为 4096（或换显存/内存更大的机器后保持官方值）。

## PAC-MAN 侧（对照组，已训好）

- 任务：`Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat`（主仓库）
- 最新 checkpoint：`/home/zxg/perceptive_cbf_rl/logs/rsl_rl/state_classroom/2026-10-16_01-19-30/model_75000.pt`
- 巡航速度 0.6 m/s，训练路线含随机换道转弯（横过道站点 v ≈ 0/3.1/6.2/9.3/12.4/15.5）

## 已拍板的设计原则

**忠于 LP-NavOA 原生训练方法，场景替换（classroom）是唯一变量**
（用户 2026-10-21 决定）。含义：

- stage 1–6 与官方 `curriculum_train_v6.sh` 完全一致，不加任何定制。
- classroom 阶段不做定制的转弯路线（否决了"S 形换道路线"提案）。
  原生的速度自适应 waypoint 生成在 classroom 跑不起来（墙体检测会
  把课桌误判为墙），这是场景替换被迫的最小适配：显式直线车道 waypoint。
- 转弯能力由原生 stage 5/6 提供（classroom 规划器从 indoor 规划器热启动），
  规划器能否把 indoor 的转弯经验泛化到 classroom 属于实验结论本身。

## 待决事项（训练前需拍板）

1. **stage 1 env 数**：本机 4096（实测可行）或换机器保持官方 8192。
2. **训练机器**：官方总规模（4000+4000+4500+80000+20000+20000 迭代）在
   3060 上预计需要数天；若有服务器建议迁移（整个 `others/LP-NavOA-code/`
   目录 + 独立 `uv sync --extra cu128` 即可，不再需要共享 venv 纪律）。
3. 评测协议：建议双方都在 `Mjlab-Classroom-NoPP-G1-v6` 等价场景下测
   成功率/碰撞率/平均速度，路线集合一致。
