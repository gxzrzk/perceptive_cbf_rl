# train_runs: 论文实验训练集群脚本 (Paper Training Cohort)

本目录为论文核心的 **感知模态 × 安全屏障 (Perception × Safety)** 网格实验中的每个单元格提供了一键启动的封装脚本。
每个脚本均固化了统一的基准超参数（`NUM_ENVS=8192`, `STAND_RATIO=0.2`, `INPLACE_RATIO=0.4`，最多 25k 迭代），
并自动导出环境变量 `EXP_NAME=<key>`，以便 `scripts/dodge_benchmark.py` 能够自动发现并加载 `logs/rsl_rl/<key>/` 下的最新权重。

论文中的所有汇报数据均基于统一对齐的 20k 迭代检查点 (`model_20000.pt`)；在运行基准测试脚本时，可通过
`--ckpt logs/rsl_rl/<key>/<run>/model_20000.pt` 显式锁定该检查点，以避免受后续未收敛权重影响。

可通过环境变量 `NUM_ENVS=<n>` 覆盖默认环境数（非默认值会自动在实验名后添加后缀，如 `state_link_4096`）。
其中状态真值任务无需渲染深度图，训练速度极快；视觉和云台任务则受限于 GPU 深度渲染吞吐量。

---

## 实验矩阵网格 (The Grid)

| 感知模态 (Perception) | 无安全屏障 (No barrier) | 连杆级 CBF (Link-CBF) | 关节级 CBF (Joint-CBF / +filter) |
|---|---|---|---|
| **状态真值 (State oracle)** | `state_none.sh` | `state_link.sh` | `state_joint.sh` |
| **固定深度相机 (Fixed camera)** | `vision_none.sh` | `vision_link.sh` (**实机部署配置**) | `vision_joint.sh` |
| **云台俯仰相机 (Gimbal camera)** | `gimbal_none.sh` | `gimbal_link.sh` | `gimbal_joint.sh` |

### 核心安全模式与机制说明:
- **No barrier (无屏障)**: 仅包含基于质心距离的球体排斥奖励，剔除连杆级屏障项 (`NO_LINK_CBF=1`)。
- **Link-CBF (连杆级 CBF)**: 在基础奖励上加入机器人各身体连杆碰撞间隙的 CBF 屏障函数惩罚。其中 `vision_link` 是实机物理部署的最终配置 (`deploy/ckpts/dodge_link_cbf.onnx` 即由此生成)。
- **Joint-CBF (关节级 CBF)**: 在训练期引入关节空间 CBF 求解模块作为引导奖励 (`CBF_JOINT=1`)。默认以裸策略评测；论文中的 **"+filter"** 单元格则使用同一模型权重，在推理时附加 `BENCH_CBF_FILTER=1` 开启特权真值球状态的运行期安全投影滤波（注：依赖真值，无法在实机纯视觉部署）。
- **Gimbal (云台相机)**: 在 BallOnly 深度任务基础上，安装了带真值动态跟踪的 +/-30° 俯仰云台 (`CAMERA_GIMBAL=1 CAMERA_PROPRIO=1`)，代表主动视野追踪的性能上限边界。

---

## 全向高速抛球校准 (Omni-throw calibration)

- `state_link_omni.sh` / `state_joint_omni.sh`：针对 360° 全向高速抛球场景进行专门训练 (`OMNI_THROW=1`，起抛距离 8-10m，球速 12-15 m/s)。
- 评估时必须搭配 `--omni` 选项；benchmark 脚本仅在传入 `--omni` 时才会默认执行此类任务。

---

## 行走恢复策略 (Walk policy / Deployment regime)

- `../train_walk.sh`：训练纯本体感知的行走策略 (`Unitree-G1-AMP-Flat`，实验名为 `walk`)。
- 该策略供基准评测的 `--walk-recover` 模式以及硬件部署时的 walk <-> dodge 模式切换调用。

---

## 基准测试评测命令 (Benchmarking a trained cohort)

```bash
# 1. 默认 Reset 模式 (每次单球测试，包含纯站立雕像基线):
uv run python scripts/dodge_benchmark.py

# 2. 真机部署模式 (Deployment regime: 躲避后由 walk 策略自动走回工位):
uv run python scripts/dodge_benchmark.py --walk-recover

# 3. 指定单一单元格 / "+filter" 滤波模式 / 全向 360° 评测:
uv run python scripts/dodge_benchmark.py --only vision_link
BENCH_CBF_FILTER=1 uv run python scripts/dodge_benchmark.py --only state_joint
uv run python scripts/dodge_benchmark.py --omni
```

