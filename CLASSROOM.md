# WallWalk 教室场景

任务：`Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat`。

与 WallWalk 等任务使用相同注册机制：`config/g1/__init__.py` 中注册训练环境、播放环境、独立 PPO 配置函数和 `AMPOnPolicyRunner`；PPO 配置函数位于 `goto_rl_cfg.py`。

可直接使用统一入口：

```bash
uv run python scripts/list_envs.py
uv run python scripts/train.py Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat --env.scene.num-envs=1024
uv run python scripts/play.py Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat --agent zero --num-envs 1 --viewer viser
```

统一入口默认日志目录为 `logs/rsl_rl/g1_amp_dodge_mimickit_classroom/`；便利脚本默认使用 `logs/rsl_rl/state_classroom/`，与其他任务的脚本约定一致。`./play_classroom.sh` 默认查找训练模型；无模型预览使用 `--agent zero`。

教室占地 32 × 14 米，6 排 × 4 列，共 24 套课桌椅。中央纵向通道净宽 3 米；相邻两排之间横向净空 2.5 米；中央通道两侧各有一条净宽 2.5 米的内部纵向通道。横向通道连通三条纵向通道，可以用于侧移和闪避。排距 4.1 米，已扣除课桌与椅子的总占地长度 1.6 米。

机器人每次重置时从中央或两侧内部纵向通道随机出生，位置在 x=0–27 米、通道中心线左右 0.15 米范围采样，朝向在 360° 范围均匀采样，以最高 0.8 米/秒沿通道前进，每回合 25 秒，保留 WallWalk 的走路、躲球和 AMP 训练逻辑。教室固定在各环境的原点坐标系内，家具在回合中固定。球在每次初始化或回合重置时停放到教室外的隐藏位置，投掷触发时才出现在发射点。投球方位在机器人周围 360° 均匀随机采样，发射距离为 3–5 米，保留原有投球间隔和高低弹道混合；桌椅及墙体仍会与球发生碰撞。每次重置为各环境独立生成随机路线：先随机选择前进方向，再在三条纵向通道与七条横向通道（五条排间通道及前后空地）的交叉点随机选择下一段，避免立即折返。每条路线预生成 80 个点，保持在教室内；跟随前视距离为 0.8 米，减少拐弯时切入桌椅区。

家具使用可见桌面、椅座、椅背和腿部模型；碰撞与 CBF 使用覆盖整件家具的保守包围盒，因此不能钻桌底或穿过椅腿之间。碰撞终止覆盖全部 52 件障碍物，策略观察最近 6 件障碍物。相比默认 WallWalk 的 3 件障碍物观察，网络输入尺寸不同，需要单独训练。

预览（无训练模型）：

```bash
./play_classroom.sh --agent zero
```

训练（默认 1024 个环境，站立比例和原地躲球比例均为 0）：

```bash
./train_classroom.sh
# 调整并行环境数量
NUM_ENVS=512 ./train_classroom.sh
```

播放教室模型：

```bash
./play_classroom.sh logs/rsl_rl/state_classroom/<run>/model_25000.pt
```

布局位于 `src/tasks/amp_loco/config/g1/dodge_env_cfgs.py` 的 `g1_amp_dodge_mimickit_classroom_flat_env_cfg`。修改速度或回合长度时，应同时留足前墙之前的行走和投球空间。
