"""基于 RSL-RL 算法框架的强化学习训练入口脚本 (Training Script with RSL-RL)

本脚本是整个代码库强化学习策略训练的通用总入口，支持单卡、多卡分布式训练以及基于跟踪任务的参考轨迹加载。

1. 命令行参数解析机制 (tyro CLI):
   - 第一个参数用于从注册表中选择任务 ID (Task ID)，可选任务由 `mjlab.tasks` 与 `src.tasks` 动态注册。
   - 后续命令行参数映射并覆盖 `TrainConfig` 数据类中的配置字段：
     * `--env.*`: 环境层级超参数（例如 `--env.scene.num-envs 8192` 覆盖并行环境数）。
     * `--agent.*`: 算法与 Runner 超参数（例如 `--agent.max-iterations 25000`, `--agent.experiment-name state_link`）。
     * `--motion-file`: 运动跟踪任务所需加载的参考轨迹文件路径 (.npz)。
     * `--video`: 是否在训练过程中周期性录制策略评估视频（默认 False）。
     * `--video-interval`: 视频录制迭代间隔（以环境步数为单位）。
     * `--video-length`: 每次录制视频的最大步长（帧数）。
     * `--enable-nan-guard`: 是否启用 MuJoCo 仿真的 NaN 状态崩溃防护守卫。
     * `--gpu-ids`: 指定参与训练的 GPU 编号列表（如 `[0]` 或 `'all'`），支持多卡调度。

2. 训练执行核心流程 (Training Pipeline):
   - 任务选择与参数绑定 (`main`):
     解析 Task ID 与剩余覆盖参数，生成 `TrainConfig` 配置实例后调用 `launch_training`。
   - 进程与 GPU 调度 (`launch_training`):
     * 创建带时间戳的标准实验日志路径 (`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>`)。
     * 自动配置 `CUDA_VISIBLE_DEVICES` 与无头渲染引擎环境变量 (`MUJOCO_GL=egl`)。
     * 若为单卡或 CPU 训练，直接在当前进程执行 `run_train`。
     * 若为多卡训练，使用 `torchrunx` 自动拉起分布式多进程，分发至各 GPU 节点。
   - 训练主循环 (`run_train`):
     * 多进程随机种子自增设定（`cfg.agent.seed + local_rank`），避免各 worker 随机序列重合。
     * 绑定 MuJoCo EGL 渲染卡与 PyTorch 计算卡对齐。
     * 针对动作跟踪 (Tracking) 任务校验并绑定 motion command 文件。
     * 实例化仿真环境 `ManagerBasedRlEnv`。
     * Rank 0 进程装配 `VideoRecorder` 用于生成可视化训练效果片段。
     * 使用 `RslRlVecEnvWrapper` 进行强化学习向量化接口对齐与动作截断 (Action Clipping)。
     * 动态加载对应的算法执行器（Runner，如 `MjlabOnPolicyRunner`）。
     * 检查点恢复处理：若配置中指定 `--agent.resume`，自动加载旧权重。
     * Rank 0 进程持久化保存环境与算法配置至 `params/env.yaml` 和 `params/agent.yaml`。
     * 调用 `runner.learn()` 进入 PPO 迭代优化循环。
"""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  """训练总体配置类 (Train Configuration)
  
  封装环境配置、强化学习算法配置以及运行时相关的通用参数选项。
  """
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  motion_file: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    """根据任务 ID 自动加载默认的环境配置与 RL 算法配置"""
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  """单卡/单进程训练主体流程 (Single-worker Training Core)
  
  负责在分配的具体设备上初始化环境、包装器、算法 Runner 并执行策略训练迭代。
  """
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    if not cfg.motion_file:
      raise ValueError("For tracking tasks, --motion-file must be set ...")
    motion_path = Path(cfg.motion_file).expanduser().resolve()
    if not motion_path.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner   # 

  runner_kwargs = {}
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  """启动训练作业（支持单卡直接运行与多卡 torchrunx 分布式启动）"""
  args = args or TrainConfig.from_task(task_id)

  # 创建带时间戳和实验名称的实验日志目录 (logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>)
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # 根据命令行与 CUDA_VISIBLE_DEVICES 环境变量筛选可用的 GPU 设备列表
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # 配置统一的环境变量（指定 GPU 并在 Linux 上强制开启 EGL 无头渲染加速）
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU 或单 GPU 环境：直接在主进程中执行训练，无需多进程编排
    run_train(task_id, args, log_dir)
  else:
    # 多 GPU 分布式训练：使用 torchrunx 自动拉起多进程 workers
    import torchrunx

    # torchrunx 将标准输出重定向到 Python logging 系统
    logging.basicConfig(level=logging.INFO)

    # 配置 torchrunx 的日志输出目录（优先级：环境变量 > 命令行选项 > 默认在 log_dir/torchrunx）
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # 用户通过命令行显式指定路径
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # 默认：存放在本次训练 run 对应的目录下
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # 允许 rsl_rl 自行初始化分布式进程组 (process group)
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  """主函数入口：两阶段命令行解析 (任务 ID 选取 -> 覆盖参数绑定 -> 启动训练)"""
  # 导入任务模块以触发任务注册表 (Registry) 的自动填充
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  # 第一阶段：解析命令行第一个参数（Task ID），并捕获未解析的剩余参数 (remaining_args)
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # 第二阶段：以选定任务的默认配置为基础，将剩余参数绑定并覆盖至 TrainConfig
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  # 启动训练流程
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
