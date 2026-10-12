"""基于 RSL-RL 策略模型的推理回放、可视化交互、ONNX 导出与硬件在环评测脚本 (Play & Evaluation Script)

本脚本用于对训练完毕的强化学习策略进行全面评估、可视化回放 (Play/Render)、单文件 ONNX 模型导出
以及物理双目相机硬件在环 (HIL, Hardware-In-The-Loop) 联调测试。

1. 命令行参数解析机制 (PlayConfig 与 tyro CLI):
   - 任务选择: 第一个位置参数为 Task ID（如 `Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat`）。
   - 智能体与模型参数:
     * `--agent`: 策略执行模式，可选：
       - `trained` (默认): 加载训练好的神经网络权重模型执行推理。
       - `zero`: 恒输出零动作的 Dummy Agent（用于测试纯物理站立稳定性或观察空场景）。
       - `random`: 均匀随机动作 Dummy Agent。
     * `--checkpoint-file`: 本地训练权重检查点文件路径 (`model_*.pt`)。
     * `--num-envs`: 并行可视化环境数量（交互式查看一般设为 1，多环境可同屏观察）。
     * `--device`: 推理运行设备 (`cuda:0` 或 `cpu`)。
     * `--no-terminations`: 禁用所有回合终止条件（摔倒不重置，方便连续观察动作轨迹）。
   - 可视化与交互参数:
     * `--viewer`: 可视化窗口后端，可选 `auto` (默认)、`viser`、`native`:
       - `auto`: 检测系统桌面环境变量，有图形界面时优先使用原生 GLFW 窗口，无头服务器自动启动 Web 端 Viser。
       - `viser`: 启动 Web 交互服务 (默认 http://localhost:8080)，在网页端提供 Dodgeball 交互控制面板
         （支持一键暂停抛球、强制投掷高球 Overhead / 低球 Underbody 等实时控制）。
       - `native`: 启动 MuJoCo 原生交互式渲染窗口。
     * `--video`: 是否录制推理回放过程并保存为 MP4 格式视频。
     * `--video-length`: 录制视频的持续步数。
   - 视觉与深度感知参数:
     * `--depth-frame-offsets`: 深度图多帧历史时间偏移量（如 `(0,3,8,18)`），必须严格匹配模型训练时
       的帧历史堆叠设置，否则 Actor 输入特征维度不匹配。
     * `--depth-aug`: 仿真回放中是否开启与训练一致的深度图域随机化 (DR) 噪声，保证评估处于分布内。
     * `--depth-aug-preview`: 在 Viser 中额外打开一个 `head_depth_single_aug` 实时窗口，
       用于并排对比原始干净深度图与送入神经网络的带噪/随机化深度图。
   - 硬件在环 (HIL) 实机相机参数:
     * `--zed-camera`: 启用实机硬件在环模式，使用物理连接的 ZED Mini 双目深度相机替代仿真中的头部相机。
     * `--zed-depth-mode`: ZED SDK 深度计算模式（如 `PERFORMANCE`，兼顾低延迟与高帧率）。
     * `--zed-pool-percentile`: 深度图下采样至策略输入分辨率 (如 16x9) 时的分位数滤波（采用 Soft-min 保留近处小球）。
     * `--zed-sim-ball`: 在真实 ZED 深度画面中动态叠加由仿真物理引擎抛出的虚拟小球（语义分割 Mask + Min-pooling），
       实现在真实物理背景下安全受控地评测避障策略。
   - 部署导出与状态复位:
     * `--export-onnx`: 模型加载成功后，自动将 Actor 网络与观测归一化器导出为紧凑的单文件 ONNX 模型
       （保存于 `<log_dir>/export/<task>_<checkpoint>.onnx`）。
     * `--reset-stand`: 每次重置时强制恢复为标准平静站立姿态（XML Init Keyframe），而非激进的动态躲避初始姿态，
       专为实机部署演示与 HIL 调试优化。

2. 核心评估流程 (Evaluation Pipeline):
   - 环境与网络配置装载: 根据任务 ID 加载环境配置 `env_cfg` 与算法配置 `agent_cfg`。
   - 传感器与显示重写: 覆写相机的 `_compute_data` 钩子函数，统一近远距离截断 (Clamp to [0.1, 5.0]m)，
     使 Viser 显示画面与策略网络输入在数值和视觉呈现上严格对齐。
   - 硬件在环接入 (可选): 初始化 `ZedDepthSource`，启动异步相机拉流线程并处理球体图层合成。
   - 策略加载与 ONNX 导出: 实例化 Runner，恢复权重检查点，自动提取 `act_inference` 并导出为标准单文件 ONNX。
   - 交互式运行循环: 启动选定的 Viewer，持续驱动物理步进、接收动作输入并渲染更新画面。
"""

import os
import inspect
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  """策略推理与回放配置类 (Play Configuration)"""
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  depth_frame_offsets: tuple[int, ...] | None = None
  """Override the depth obs frame offsets to MATCH the trained checkpoint (e.g. '(0,3,8,18)').
  The play env defaults to the registered offsets (0,1); a checkpoint trained with more frames
  has a larger actor input, so playing it requires the same offsets or load_state_dict mismatches."""
  depth_aug: bool = True
  """Apply the training depth augmentation (speckle/clutter/haze/holes/ground far-out) to the POLICY
  obs during SIM play, so play matches what the policy trained on. Single-camera dodge-depth task
  only; automatically skipped on the ZED path (the real camera supplies real noise -- augmenting it
  would double the noise). Honors DEPTH_DR_SCALE. Set False for a clean-depth sim eval."""
  depth_aug_preview: bool = False
  """Like --depth-aug but ALSO adds a 'head_depth_single_aug' panel showing the exact augmented depth
  the policy consumes -- next to the clean 'head_depth_single' panel -- so you can compare them live.
  Single-camera dodge-depth task only. Sim balls keep throwing so you can watch the DR react."""
  zed_camera: bool = False
  """Hardware-in-the-loop: feed LIVE depth from a real ZED Mini into the head depth sensor
  instead of the simulated camera (the sim still supplies proprioception and is what you watch
  dodge). Single-camera dodge-depth task only (head_depth_single). Requires pyzed + a plugged-in
  ZED Mini. Pairs with num_envs=1; pauses sim ball throws so the real camera is the only stimulus."""
  zed_depth_mode: str = "PERFORMANCE"
  """ZED SDK depth mode (PERFORMANCE/QUALITY/NEURAL/...). PERFORMANCE = lowest latency for HIL."""
  zed_flip: bool = False
  """Flip the ZED image 180 deg (set if the camera is mounted upside-down)."""
  zed_head_serial: int | None = None
  """Serial number of the ZED to use as the HEAD (+20 deg up) camera that feeds the policy. Open the
  specific device by serial on a multi-camera rig; None = the first available ZED (single-cam, as
  before). The connected serials are printed on startup."""
  zed_pool_percentile: float = 3.0
  """Per-block percentile when downsampling ZED WVGA depth to the policy 16x9 (0 = strict min,
  the nearest pixel). A low value ("soft-min") preserves the small near ball while rejecting 1-2
  spurious near pixels; keep it below the ball's per-block pixel fraction (~5%). Never use a high
  value / averaging -- that blends a small ball into the far background and erases it."""
  zed_fill_holes: bool = False
  """Enable the ZED SDK dense FILL depth mode. OFF by default: FILL interpolates/extrapolates into
  no-measure holes and invents a phantom near surface in empty/low-texture space that the low-
  percentile pool reports as a near object. Leave off unless you specifically want a hole-free panel."""
  zed_confidence: int = 50
  """ZED confidence_threshold (0-100, lower=stricter; SDK default 95). Culls low-confidence
  (flying/mixed) depth pixels before pooling so empty space stops reading as a near object."""
  zed_texture_conf: int = 100
  """ZED texture_confidence_threshold (0-100, lower=stricter; SDK default 100=keep all). Lower
  (e.g. 50) to also cull depth from low-texture regions (blank walls / empty space)."""
  zed_edge_crop: int = 10
  """Blank the outer N px of the full-res depth before pooling (default 10). Drops the ZED stereo
  occlusion border (top/left edge emits confident-but-bogus near depth). 0 disables."""
  zed_obs_mask: Literal["none", "far"] = "none"
  """Mask applied to the POLICY depth obs (the RGB/depth display panels always show the real scene).
  'none' = real pooled depth. 'far' = force the whole obs to far (an empty image) -- diagnostic."""
  zed_sim_ball: bool = True
  """SUPERIMPOSE the simulated ball onto the live ZED depth (ZED path only). Renders the sim head
  camera's segmentation, masks the ball geom, and min-pools just those pixels into the real ZED frame
  -- so the policy sees a controllable sim ball against the real-camera background. Faithful to
  training (same sub-pixel limit: a far ball is <1px until ~1 m). Throws still start paused (use the
  viser 'Pause ball throws' checkbox / throw-once to launch one). Set False for a pure real-camera feed."""
  viewer: Literal["auto", "native", "viser"] = "auto"
  reset_stand: bool = False
  """Reset every episode to the robot's nominal default standing pose (XML init_state keyframe)
  instead of the dynamic dodge RSI poses (one-foot/airborne ~46%). For HIL/demo play so the robot
  starts and re-starts calm. Dodge tasks only (needs a 'reset_from_motion' event)."""
  no_terminations: bool = False
  export_onnx: bool = True
  """Export loaded trained policy to ONNX under current run directory/export."""
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # 内部标志位，供 demo 演示脚本直接调用控制 (Internal flag used by demo script)
  _demo_mode: tyro.conf.Suppress[bool] = False


class _DodgePlayViewer(ViserPlayViewer):
  """专为 Dodgeball 避障任务定制的 Viser 网页端交互可视化器

  在通用 ViserPlayViewer 基础上扩展了 "Dodgeball" 控件面板：
  - "Pause ball throws" 复选框：一键暂停自动周期性抛球，方便静态观察机器人站立姿态。
  - "Throw overhead" 按钮：手动触发投掷高球（高弧线到达躯干/头部高度，测试策略低头/弯腰/下潜躲避）。
  - "Throw underbody" 按钮：手动触发投掷低球（平射直接掠过下半身/腿部高度，测试策略侧跨步闪避）。
  通过在环境实例上修改 `_dodge_throw_paused` / `_dodge_throw_once` / `_dodge_throw_force_high` 标志位实现通信。
  """

  def setup(self) -> None:
    super().setup()
    import viser

    env = self.env.unwrapped
    throws_balls = any(
      "throw_ball_on_dwell" in names
      for names in env.event_manager.active_terms.values()
    )
    if not throws_balls:
      return

    # 保留在可视化器启动前设定的抛球暂停状态（在 HIL/ZED 硬件在环模式下暂停投球，使真实相机成为唯一刺激源；原英文: Preserve a pause set before the viewer started）
    initial_paused = bool(getattr(env, "_dodge_throw_paused", False))
    env._dodge_throw_paused = initial_paused
    env._dodge_throw_once = False

    with self._server.gui.add_folder("Dodgeball"):
      pause_cb = self._server.gui.add_checkbox(
        "Pause ball throws", initial_value=initial_paused
      )
      overhead_btn = self._server.gui.add_button(
        "Throw overhead", icon=viser.Icon.BALL_BASKETBALL
      )
      underbody_btn = self._server.gui.add_button(
        "Throw underbody", icon=viser.Icon.BALL_BASKETBALL
      )

    def _on_pause(_ev) -> None:
      env._dodge_throw_paused = pause_cb.value

    # Overhead = 投掷高球 (high_throw 分支)：球到达躯干/头部高度 -> 诱导策略下蹲/俯身下潜闪避 (duck / lean)
    def _on_throw_overhead(_ev) -> None:
      env._dodge_throw_force_high = True
      env._dodge_throw_once = True

    # Underbody = 投掷低球 (descending 分支)：球瞄准下半身/腿部高度 -> 诱导策略侧跨步闪避 (sidestep / step over)
    def _on_throw_underbody(_ev) -> None:
      env._dodge_throw_force_high = False
      env._dodge_throw_once = True

    pause_cb.on_update(_on_pause)
    overhead_btn.on_click(_on_throw_overhead)
    underbody_btn.on_click(_on_throw_underbody)


def run_play(task_id: str, cfg: PlayConfig):
  """执行策略评估推理与可视化主循环 (Play Core Function)"""
  def onnx_export_kwargs_single_file() -> dict:
    """构建用于跨不同 PyTorch 版本导出单文件 ONNX 的参数字典"""
    try:
      params = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
      return {}

    if "external_data" in params:
      return {"external_data": False}
    if "use_external_data_format" in params:
      return {"use_external_data_format": False}
    return {}

  def inline_external_onnx_data(onnx_path: Path) -> None:
    """若存在外部权重数据文件 (.data)，将其重新合并内嵌回单个自包含的 ONNX 模型文件中。
    (Merge external tensor data back into a single ONNX file if needed.)
    """
    data_path = Path(str(onnx_path) + ".data")
    if not data_path.exists():
      return

    try:
      import onnx

      model = onnx.load(str(onnx_path), load_external_data=True)
      onnx.save_model(model, str(onnx_path), save_as_external_data=False)
      if data_path.exists():
        data_path.unlink()
      print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
    except Exception as exc:
      print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")

  class _OnnxPolicyWrapper(torch.nn.Module):
    """将强化学习策略网络封装为标准 PyTorch Module，将 act_inference 暴露为 forward，并可选挂载观测归一化层。
    (Expose act_inference as forward and optionally include obs normalizer.)
    """

    def __init__(self, actor_critic: torch.nn.Module, obs_normalizer: Any = None):
      super().__init__()
      self.actor_critic = actor_critic
      self.obs_normalizer = obs_normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
      if self.obs_normalizer is not None:
        obs = self.obs_normalizer(obs)
      return self.actor_critic.act_inference(obs)

  def export_runner_policy_to_onnx(runner: Any, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 优先使用 Runner 自带的导出接口，确保模型图导出逻辑与算法训练配置完全一致 (Prefer runner-provided exporters)
    if hasattr(runner, "export_policy_to_onnx"):
      runner.export_policy_to_onnx(str(output_path.parent), output_path.name)
      inline_external_onnx_data(output_path)
      return
    if hasattr(runner, "_export_policy_to_onnx"):
      runner._export_policy_to_onnx(str(output_path.parent), output_path.name)
      inline_external_onnx_data(output_path)
      return

    # 针对未显式提供 ONNX 导出方法的 Runner 采用通用后备导出逻辑 (Fallback exporter for runners without explicit ONNX export helper)
    policy = runner.alg.policy
    obs_normalizer = None
    if getattr(runner, "empirical_normalization", False) and hasattr(
      runner, "obs_normalizer"
    ):
      obs_normalizer = runner.obs_normalizer
      obs_normalizer.to("cpu")
      obs_normalizer.eval()

    wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
    wrapper.to("cpu")
    wrapper.eval()
    num_obs = policy.actor[0].in_features
    dummy_input = torch.zeros(1, num_obs)
    torch.onnx.export(
      wrapper,
      dummy_input,
      str(output_path),
      export_params=True,
      opset_version=18,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      **onnx_export_kwargs_single_file(),
    )
    inline_external_onnx_data(output_path)

    runner_device = getattr(runner, "device", None)
    if runner_device is not None:
      policy.to(runner_device)
      if obs_normalizer is not None:
        obs_normalizer.to(runner_device)

  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # 若命令行请求禁用终止条件（在纯前向观察动作库或调试特定步态行为时非常实用）(Disable terminations if requested)
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # 通过检查环境指令集合中是否存在 "motion" 命令来判定当前是否属于全身跟踪任务 (Check if this is a tracking task)
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # 演示模式：当并发环境数量 num_envs > 1 时启用均匀随机采样，以便同时展现更多姿态多样性 (Demo mode: use uniform sampling)
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # 优先检查本地是否存在指定的参考动作文件（无论使用 dummy 策略还是训练策略均生效）(Check for local motion file first)
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # 从文件路径中提取 run_id 与 checkpoint 检查点文件名用于界面打印展示 (Extract run_id and checkpoint name)
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.depth_frame_offsets is not None:
    # DepthImageObs 观测项在类初始化时即会读取 env.cfg.depth_frame_offsets；
    # 因此必须在构建环境之前完成配置，确保 Actor 网络的输入特征维度与对应时间步偏移训练出的 Checkpoint 严格对齐匹配。
    # [英文对照: DepthImageObs reads env.cfg.depth_frame_offsets at instantiation; set before build to match actor dim]
    env_cfg.depth_frame_offsets = cfg.depth_frame_offsets
  if cfg.reset_stand:
    # 将闪避任务中的 RSI 动态动作重置替换为机器人标称站立默认姿态重置（供 HIL 硬件在环或 Demo 演示使用，确保启动平稳）
    # [英文对照: Swap the dodge RSI motion-reset for a nominal default-stand reset; default asset_cfg, drop unused motion_dir]
    if "reset_from_motion" in env_cfg.events:
      from src.tasks.amp_loco.mdp.events import reset_to_default_stand

      stand_event = env_cfg.events["reset_from_motion"]
      stand_event.func = reset_to_default_stand
      stand_event.params = {}  # 采用默认 asset_cfg；移除未使用的 motion_dir 参数
      print("[play] reset_stand: episodes reset to the nominal default standing pose")
    else:
      print("[play] reset_stand requested but no 'reset_from_motion' event; ignoring")
  # 在仿真回放评估中，将训练期的深度图数据增强 (Depth Augmentation) 应用于策略输入的观测中，
  # 使得评估时的策略输入分布与训练阶段严格一致 (--depth-aug 默认开启；--depth-aug-preview 隐式开启并添加对比面板)。
  # 注意：在真实 ZED 相机硬件在环 (HIL) 路径下跳过该合成噪声增强，因为真实传感器自身已带有物理噪声，叠加会导致噪声翻倍。
  # 仅限于单深度任务 (head_depth_single)。
  _want_depth_aug = (cfg.depth_aug or cfg.depth_aug_preview) and not cfg.zed_camera
  _is_single_depth = (
    "depth" in env_cfg.observations
    and "head_depth" in env_cfg.observations["depth"].terms
    and any(getattr(s, "name", None) == "head_depth_single" for s in (env_cfg.scene.sensors or ()))
  )
  if _want_depth_aug and _is_single_depth:
    from src.tasks.amp_loco.config.g1.dodge_env_cfgs import (
      DEPTH_DR_SCALE,
      enable_depth_aug_preview,
    )

    enable_depth_aug_preview(env_cfg)
    print(
      f"[play] depth augmentation ENABLED on policy obs (DEPTH_DR_SCALE={DEPTH_DR_SCALE}; "
      "ground_far fixed 0.9; segmentation rendered)"
    )
  elif cfg.depth_aug_preview and not _is_single_depth:
    print("[play] --depth-aug-preview ignored: not the single-depth task (head_depth_single)")
  if cfg.zed_camera and cfg.zed_sim_ball:
    # 在实拍画面上叠加仿真小球需要仿真头部相机渲染语义分割图 (SEGMENTATION) 以精确提取小球网格掩码。
    # 在构建环境之前将 "segmentation" 添加入观测传感器的 data_types (ZED 模式下仿真器仍渲染该相机，我们提取小球后叠加进实拍流)。
    for s in env_cfg.scene.sensors or ():
      if getattr(s, "name", None) == "head_depth_single" and "segmentation" not in s.data_types:
        s.data_types = tuple(s.data_types) + ("segmentation",)
        print("[play] zed-sim-ball: segmentation enabled on head_depth_single for ball masking")
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  # 仅在回放可视化时生效：使 Viser 深度图显示面板呈现有用且对比度适宜的固定深度视窗。
  # MuJoCo Warp (mjwarp) 原始深度图输出为真实物理距离（范围可延伸至无穷远地平线），而 Viser 查看器默认仅将其除以滑块缩放值，
  # 导致远景地面/天空主导数值范围，深度面板显示为苍白一片。我们将每个深度 CameraSensor 的 _compute_data 进行包装：
  # 截断裁剪至 [DEPTH_NEAR, DEPTH_FAR]（未命中/天空 -> 映射为远端 far），并根据任务类型决定是否归一化到 [0, 1]，
  # 从而在默认滑块设置下渲染出层次清晰的近黑远白对比度梯度。近处的小球呈现深黑色，凸显于较浅色的远端背景前。
  #
  # 关键机制：由于此操作修改的是共享传感器实例，而策略网络的 DepthImageObs 同样读取该 sensor.data.depth，
  # 因此所施加的数据变换必须保证策略观测值绝对不变 (MUST leave the policy obs unchanged)。分为两种情形：
  #  - 无深度观测组（纯本体感觉 base-dodge 闪避任务的可视化预览）：Checkpoint 从不读取该传感器，因此可以全量
  #    裁剪并归一化到 [0, 1]，使默认滑块下的面板获得最佳灰度梯度。
  #  - 深度感知任务（存在 'depth' 观测组）：仅应用“未命中/天空->far + [near, far] 范围截断”两步（严禁执行归一化）。
  #    这一操作与 DepthImageObs._one 的前一步完全重合，具备幂等性 (idempotent) —— 传入已清洗的物理米制深度将
  #    产生完全相同的归一化观测，确保策略输入在字节级别完全不变；同时面板呈现策略所见的视界：
  #    天空/空旷区域 -> far (白色)，近处/小球 -> dark (深色)，消除了原始米制数据中未命中 (0m) 显示为全黑的严重视觉误导。
  #    （严禁在此处除以 far 进行归一化：若传入 [0, 1] 再次经由 _one 计算，会将小于 near 的小球距离映射为 far，导致图像黑白反转）。
  #    在面板中将 Depth Scale 滑块调至 ~5 (= far) 即可直接以 [0, 1] 形式直观对应策略输入的真实灰度。
  from mjlab.sensor import CameraSensor as _CameraSensor
  from mjlab.sensor.camera_sensor import CameraSensorData as _CamData

  _has_depth_obs = "depth" in env.observation_manager.active_terms
  _DEPTH_NEAR, _DEPTH_FAR = 0.1, 5.0

  for _sns in env.scene.sensors.values():
    if not (isinstance(_sns, _CameraSensor) and "depth" in _sns.cfg.data_types):
      continue
    _orig_compute = _sns._compute_data

    def _display_compute(_orig=_orig_compute, _normalize=not _has_depth_obs):
      data = _orig()
      if data.depth is not None:
        d = data.depth
        # 未命中（天空，深度接近 0）映射为远端 far；数值截断在 [near, far] 范围内。
        # 该前半部分截断操作在 DepthImageObs._one 处理下是完全幂等的，确保深度任务策略观测输入绝对无损。
        # [英文对照说明: No-hit/sky -> far; clamp to [near, far]. This half is idempotent under DepthImageObs._one]
        d = torch.where(d < _DEPTH_NEAR, torch.full_like(d, _DEPTH_FAR), d)
        d = d.clamp(_DEPTH_NEAR, _DEPTH_FAR)
        if _normalize:  # 仅当没有策略观测项后续处理该传感器数据时（纯本体感觉任务预览），才在此处归一化到 [0, 1]
          d = (d - _DEPTH_NEAR) / (_DEPTH_FAR - _DEPTH_NEAR)
        data = _CamData(rgb=data.rgb, depth=d, segmentation=data.segmentation)
      return data

    _sns._compute_data = _display_compute
    _mode = "clamp+normalize" if not _has_depth_obs else "clamp-only (policy view, obs unchanged)"
    print(f"[play] depth panel '{_sns.cfg.name}': {_mode} to [{_DEPTH_NEAR}, {_DEPTH_FAR}] m")

  # 添加一个仅用于可视化对比的独立面板，实时展示策略网络实际接收到的经增强处理后的深度图 (AUGMENTED depth)
  # （观测项将增强后的米制深度图暂存在环境实例上的 _depth_aug_display 中）。
  # 该裸传感器无需接入 MuJoCo 仿真上下文；其 _compute_data 直接返回暂存的增强帧并截断供面板渲染。
  # 紧邻原生 'head_depth_single' 面板放置，可直观实时对比原始无损深度与增强噪声深度。
  # 只要启用了数据增强即会展示（默认仿真 play 及 --depth-aug-preview），但在真实 ZED 路径或非单相机任务中不创建。
  if _want_depth_aug and _is_single_depth:
    from mjlab.sensor.camera_sensor import CameraSensorCfg as _CameraSensorCfg

    env._depth_aug_display_enabled = True
    _obs_sns = env.scene.sensors["head_depth_single"] if "head_depth_single" in env.scene.sensors else None
    if not isinstance(_obs_sns, _CameraSensor):
      raise RuntimeError(
        "depth augmentation panel needs the Unitree-G1-AMP-Dodge-Depth-Single-Flat task (head_depth_single)."
      )
    _ah, _aw = _obs_sns.cfg.height, _obs_sns.cfg.width
    _aug_sensor = _CameraSensor(
      _CameraSensorCfg(
        name="head_depth_single_aug",
        camera_name="head_camera_single_aug",
        width=_aw,
        height=_ah,
        data_types=("depth",),
      )
    )
    _aug_sensor._camera_idx = _obs_sns.camera_idx
    _aug_env = env

    def _aug_display_compute(_e=_aug_env, _name="head_depth_single", _h=_ah, _w=_aw):
      store = getattr(_e, "_depth_aug_display", None)
      frame = None if store is None else store.get(_name)
      if frame is None:  # 在首个观测计算产生之前，先展示截断处理后的干净原始深度帧 (before first obs is computed)
        frame = _e.scene.sensors[_name].data.depth
      d = frame.clamp(_DEPTH_NEAR, _DEPTH_FAR)
      return _CamData(rgb=None, depth=d, segmentation=None)

    _aug_sensor._compute_data = _aug_display_compute
    env.scene.sensors["head_depth_single_aug"] = _aug_sensor
    print("[play] panel 'head_depth_single_aug' shows the AUGMENTED depth the policy consumes")

  # 硬件在环测试 (Hardware-in-the-loop, HIL)：将仿真的头部深度相机替换为外接的真实物理 ZED Mini 双目深度相机。
  # 仿真器继续驱动机器人本体感觉与物理动力学（通过屏幕观察其闪避动作）；仅将深度相机的 _compute_data 替换为
  # 返回实时采集的真实物理 ZED 深度图（原始米制单位，维度 [N,H,W,1]），使得 DepthImageObs 能够无缝执行既有的截断/归一化/时序堆叠。
  # 仅适用于单深度相机避障任务 (Single-camera dodge-depth task only)。
  zed_source = None
  if cfg.zed_camera:
    from src.deploy.zed_depth_source import ZedDepthSource

    print(f"[play] connected ZED cameras (serial, model): {ZedDepthSource.list_devices()}")

    zed_sensor = None
    zed_sensor_name = None
    for _name in ("head_depth_single", "head_depth"):
      _s = env.scene.sensors.get(_name) if hasattr(env.scene.sensors, "get") else None
      if isinstance(_s, _CameraSensor) and "depth" in _s.cfg.data_types:
        zed_sensor, zed_sensor_name = _s, _name
        break
    if zed_sensor is None:
      raise RuntimeError(
        "--zed-camera: no head depth sensor found (expected 'head_depth_single'). "
        "Use the Unitree-G1-AMP-Dodge-Depth-Single-Flat task."
      )

    _zh, _zw = zed_sensor.cfg.height, zed_sensor.cfg.width
    zed_source = ZedDepthSource(
      height=_zh,
      width=_zw,
      depth_mode=cfg.zed_depth_mode,
      flip=cfg.zed_flip,
      min_depth_m=_DEPTH_NEAR,
      max_depth_m=_DEPTH_FAR,
      pool_percentile=cfg.zed_pool_percentile,
      fill_holes=cfg.zed_fill_holes,
      confidence_threshold=cfg.zed_confidence,
      texture_confidence_threshold=cfg.zed_texture_conf,
      edge_crop_px=cfg.zed_edge_crop,
      obs_mask=cfg.zed_obs_mask,
      device=device,
      serial_number=cfg.zed_head_serial,
    )
    print(f"[play] head ZED open (serial={zed_source.serial_number})")
    _n_env = env.num_envs

    # 叠加仿真小球机制 (Superimpose the sim ball)：在覆写相机方法前，先捕获仿真环境渲染的头部相机画面
    # （即经过显示包装的 _compute_data，其返回截断深度与语义分割），并解析小球碰撞体的 geom_id。
    # 随后 _zed_compute 通过最小池化 (min-pooling) 算法，将纯小球的仿真深度无缝叠加融入实拍 ZED 深度图帧中。
    import mujoco as _mj

    _sim_compute = zed_sensor._compute_data
    _composite_ball = bool(cfg.zed_sim_ball)
    _ball_geom_id = -1
    _mj_geom = int(_mj.mjtObj.mjOBJ_GEOM)
    if _composite_ball:
      try:
        _ball_geom_id = int(env.sim.mj_model.geom("ball/ball_collision").id)
      except Exception as _e:  # noqa: BLE001 (小球几何体解析异常忽略，避免回放流程崩溃)
        print(f"[play] zed-sim-ball: could not resolve ball geom ({_e}); superimpose disabled")
        _composite_ball = False

    def _zed_compute(
      _src=zed_source, _n=_n_env, _h=_zh, _w=_zw,
      _sim=_sim_compute, _comp=_composite_ball, _bid=_ball_geom_id, _gtype=_mj_geom,
    ):
      frame = _src.latest_frame()  # 获取实拍或全远端深度的帧数据，单位：米 (real pooled depth, or all-far in 'far' mode)
      depth = frame.view(1, _h, _w, 1).expand(_n, _h, _w, 1).contiguous()
      if _comp:
        sim = _sim()  # 仿真头部相机渲染结果：包含截断深度 [N,H,W,1] 和几何体语义分割 [N,H,W,2] (sim head-camera render)
        seg, sim_d = sim.segmentation, sim.depth
        if seg is not None and sim_d is not None:
          ball = ((seg[..., 1] == _gtype) & (seg[..., 0] == _bid)).unsqueeze(-1)  # [N,H,W,1]
          # 最小池化融合：在仿真小球像素区域，若小球在实景前方（深度更小）则保留小球深度，其余区域完全保留真实 ZED 深度。
          # [英文对照说明: Ball wins where it is in front via min-pool; real ZED depth everywhere else]
          depth = torch.where(ball, torch.minimum(depth, sim_d), depth)
      return _CamData(rgb=None, depth=depth, segmentation=None)

    zed_sensor._compute_data = _zed_compute

    # 增加第二个仅供观察的显示面板：展示未经池化融合的原始分辨率真实 ZED RGB 彩色流与深度流，
    # 便于开发者将实拍原始画面与策略实际接收的 16x9 降采样深度图进行并排同屏比对。
    # 这是一个裸 CameraSensor（无需物理仿真渲染上下文，其 _compute_data 直接返回实测原始图像），
    # 挂载到 scene.sensors 之后 Viser 查看器会自动为其构建专属显示面板；
    # scene.update() 在每个仿真步都会清除其缓存从而保证画面持续实时刷新。
    # camera_idx 继承自观测传感器以保证视锥体与宽高比正确；独立的 camera_name 避免视锥体命名冲突。
    from mjlab.sensor.camera_sensor import CameraSensorCfg as _CameraSensorCfg

    # 可复用的全分辨率真实 ZED RGB+深度画面监视面板创建辅助函数 (Reusable full-res RGB+depth display panel)
    # Viser 默认会为每个注册传感器自动生成一个 RGB 面板和一个深度面板，因此将传感器命名为 'head' 即可获得直观对应的监控流。
    def _add_zed_feed(name, src, _idx=zed_sensor.camera_idx, _n=_n_env):
      _h, _w = src.raw_height, src.raw_width
      _s = _CameraSensor(
        _CameraSensorCfg(
          name=name, camera_name=f"{name}_feed", width=_w, height=_h, data_types=("rgb", "depth")
        )
      )
      _s._camera_idx = _idx

      def _compute(_src=src, _h=_h, _w=_w):
        depth = _src.latest_raw().view(1, _h, _w, 1).expand(_n, _h, _w, 1).contiguous()
        rgb = _src.latest_rgb().view(1, _h, _w, 3).expand(_n, _h, _w, 3).contiguous()
        return _CamData(rgb=rgb, depth=depth, segmentation=None)

      _s._compute_data = _compute
      env.scene.sensors[name] = _s

    _add_zed_feed("head", zed_source)  # 接入真实头部 ZED Mini：在 Viser 中创建 'head' RGB 与 'head' 深度显示面板

    # 硬件在环测试规范 (HIL)：真实物理相机应为唯一合法的输入刺激源 —— 默认暂停仿真投球逻辑
    # （该状态由 Viser 界面中的 "Pause ball throws" 复选框双向绑定，可在评估过程中随时手动重新启用）
    env._dodge_throw_paused = True
    _ball_msg = (
      "sim ball superimposed (min-pooled into ZED depth); throws start paused -- launch via the "
      "viser checkbox" if _composite_ball else "sim camera render ignored"
    )
    print(
      f"[play] ZED Mini LIVE depth -> sensor '{zed_sensor_name}' ({_zh}x{_zw}); "
      f"'head' RGB+depth panels shown; {_ball_msg}; sim ball throws paused"
    )

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # 确保日志输出目录已在 TRAINED_MODE 分支中正确初始化 (log_dir is set in TRAINED_MODE block)
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path))
    policy = runner.get_inference_policy(device=device)

    if cfg.export_onnx:
      safe_task_name = task_id.replace("/", "_").replace(":", "_")
      checkpoint_stem = resume_path.stem if resume_path is not None else "policy"
      export_root = log_dir if log_dir is not None else Path("logs")
      onnx_path = (export_root / "export" / f"{safe_task_name}_{checkpoint_stem}.onnx").resolve()
      try:
        export_runner_policy_to_onnx(runner, onnx_path)
        print(f"[INFO]: Exported ONNX policy to: {onnx_path}")
      except Exception as exc:
        print(f"[WARN]: Failed to export ONNX policy: {exc}")
  if DUMMY_MODE and cfg.export_onnx:
    print("[WARN]: ONNX export is only available for trained agents.")

  # 处理 "auto" 可视化查看器模式自动选择逻辑（若检测到图形环境变量则使用原生 GLFW 窗口，无屏环境下回退为 Viser 网页查看器）
  # [英文对照说明: Handle "auto" viewer selection]
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      _DodgePlayViewer(env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    if zed_source is not None:
      zed_source.close()
    env.close()


def main():
  """主函数入口：两阶段命令行解析 (任务 ID 选取 -> 覆盖参数绑定 -> 启动策略评估回放)"""
  # 第一阶段：解析命令行首个参数以确定 Task ID (Parse first argument to choose the task)
  # 导入任务模块以触发任务注册表自动注册 (Import tasks to populate the registry)
  import mjlab.tasks  # noqa: F401 (导入以触发注册机制，忽略 F401 未直接调用告警)
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # 第二阶段：解析剩余命令行参数，并允许灵活覆盖 env_cfg 与 agent_cfg (Parse the rest of the arguments)
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
