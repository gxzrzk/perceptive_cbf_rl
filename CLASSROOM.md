# WallWalk 教室场景

任务：`Unitree-G1-AMP-Dodge-MimicKit-Classroom-Flat`。

教室占地 32 × 14 米，6 排 × 4 列，共 24 套课桌椅。中央纵向通道净宽 3 米；相邻两排之间横向净空 2.5 米；中央通道两侧各有一条净宽 2.5 米的内部纵向通道。横向通道连通三条纵向通道，可以用于侧移和闪避。排距 4.1 米，已扣除课桌与椅子的总占地长度 1.6 米。

机器人从后方空地开始，以最高 0.8 米/秒沿中央通道前进，每回合 25 秒，保留 WallWalk 的走路、躲球和 AMP 训练逻辑。教室随每个环境的初始朝向整体放置，家具在回合中固定。当前目标路线沿中央通道；横向空间可供闪避，但任务不会主动命令机器人横穿排间。

家具使用可见桌面、椅座、椅背和腿部模型；碰撞与 CBF 使用覆盖整件家具的保守包围盒，因此不能钻桌底或穿过椅腿之间。碰撞终止覆盖全部 52 件障碍物，策略观察最近 6 件障碍物。相比默认 WallWalk 的 3 件障碍物观察，网络输入尺寸不同，需要单独训练。

预览（无训练模型）：

```bash
./play_classroom.sh
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
