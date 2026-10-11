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

  # Internal flag used by demo script.
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

    # Preserve a pause set before the viewer started (HIL/ZED mode pauses throws so the real
    # camera is the only stimulus); reflect it in the checkbox rather than clobbering it.
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

    # Overhead = HIGH throw (the high_throw branch): arrives at torso/HEAD height -> duck / lean.
    def _on_throw_overhead(_ev) -> None:
      env._dodge_throw_force_high = True
      env._dodge_throw_once = True

    # Underbody = LOW throw (the descending branch): hits the LOWER body / legs -> sidestep / step over.
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
    """Merge external tensor data back into a single ONNX file if needed."""
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
    """Expose act_inference as forward and optionally include obs normalizer."""

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

    # Prefer runner-provided exporters to keep behavior consistent with training.
    if hasattr(runner, "export_policy_to_onnx"):
      runner.export_policy_to_onnx(str(output_path.parent), output_path.name)
      inline_external_onnx_data(output_path)
      return
    if hasattr(runner, "_export_policy_to_onnx"):
      runner._export_policy_to_onnx(str(output_path.parent), output_path.name)
      inline_external_onnx_data(output_path)
      return

    # Fallback exporter for runners without explicit ONNX export helper.
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

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
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
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.depth_frame_offsets is not None:
    # DepthImageObs reads env.cfg.depth_frame_offsets at instantiation; set it before the env is
    # built so the actor input dim matches the checkpoint trained with these offsets.
    env_cfg.depth_frame_offsets = cfg.depth_frame_offsets
  if cfg.reset_stand:
    # Swap the dodge RSI motion-reset for a nominal default-stand reset (HIL/demo play).
    if "reset_from_motion" in env_cfg.events:
      from src.tasks.amp_loco.mdp.events import reset_to_default_stand

      stand_event = env_cfg.events["reset_from_motion"]
      stand_event.func = reset_to_default_stand
      stand_event.params = {}  # default asset_cfg; drop motion_dir (unused by the stand reset)
      print("[play] reset_stand: episodes reset to the nominal default standing pose")
    else:
      print("[play] reset_stand requested but no 'reset_from_motion' event; ignoring")
  # Apply the training depth augmentation to the POLICY obs in SIM play so play matches what the
  # policy trained on (--depth-aug, on by default; --depth-aug-preview implies it and also adds the
  # comparison panel further below). SKIPPED on the ZED path: the real camera already carries real
  # noise, so augmenting it would double up. Guarded to the single-depth task (head_depth_single).
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
    # Superimposing the sim ball needs the sim head camera to render SEGMENTATION (so we can mask the
    # ball geom). Add it to the obs sensor's data_types before the env is built (the sim still renders
    # this camera in ZED mode; we just read its segmentation to extract the ball).
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

  # Play-only: make the viser depth panel show a useful, fixed window. The raw mjwarp depth
  # is true distance (spans to the ~horizon) and the viewer only divides by its slider, so
  # distant ground/sky dominate and the panel looks blank. We wrap each depth CameraSensor's
  # _compute_data to clamp to [DEPTH_NEAR, DEPTH_FAR] (no-hit/sky -> far) and normalize to
  # [0,1], so the panel renders a proper near->far gradient at the default slider. The ball
  # (near) shows dark against the lighter far background.
  #
  # This mutates the SHARED sensor, and `DepthImageObs` reads the same `sensor.data.depth`, so the
  # transform we apply MUST leave the policy obs unchanged. Two cases:
  #
  #  - NO depth obs group (base-dodge play preview): the checkpoint never reads this sensor, so we
  #    fully clamp+NORMALIZE to [0,1] for a nice panel gradient at the default slider.
  #
  #  - Depth TASK (a `depth` obs group exists): we apply ONLY the no-hit->far + clamp step (NOT the
  #    normalize). That is exactly the first half of `DepthImageObs._one`, which is idempotent --
  #    feeding it already-cleaned metres yields the identical normalized obs -- so the policy input
  #    is byte-for-byte unchanged, while the panel now shows what the policy sees: empty/sky -> far
  #    (white), near/ball -> dark, instead of raw metres where sim no-hit (0 m) renders misleadingly
  #    black. (We must NOT normalize here: feeding [0,1] back through `_one` would map the near/ball
  #    values < near -> far and invert the image -- the bug the previous code avoided by leaving it
  #    raw.) Set the panel's Depth Scale to ~5 (= far) to read it as the literal [0,1] policy input.
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
        # No-hit (sky, depth ~0) -> far; clamp to [near, far]. This half is idempotent under
        # DepthImageObs._one, so the depth-task obs is unchanged.
        d = torch.where(d < _DEPTH_NEAR, torch.full_like(d, _DEPTH_FAR), d)
        d = d.clamp(_DEPTH_NEAR, _DEPTH_FAR)
        if _normalize:  # only when no obs term will re-process this sensor
          d = (d - _DEPTH_NEAR) / (_DEPTH_FAR - _DEPTH_NEAR)
        data = _CamData(rgb=data.rgb, depth=d, segmentation=data.segmentation)
      return data

    _sns._compute_data = _display_compute
    _mode = "clamp+normalize" if not _has_depth_obs else "clamp-only (policy view, obs unchanged)"
    print(f"[play] depth panel '{_sns.cfg.name}': {_mode} to [{_DEPTH_NEAR}, {_DEPTH_FAR}] m")

  # Add a display-only panel showing the AUGMENTED depth the policy consumes (the obs term stashes its
  # augmented metric frame on the env via _depth_aug_display). The bare sensor is never given a sim
  # context; its _compute_data returns the stash, clamped for the panel. Sits next to the clean
  # 'head_depth_single' panel so you can compare clean vs augmented live. Shown whenever the augmentation
  # is actually active -- i.e. default sim play (--depth-aug) as well as --depth-aug-preview, but not on
  # the ZED path (no synthetic aug there) or a non-single task.
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
      if frame is None:  # before the first obs is computed -> show clean clamped depth
        frame = _e.scene.sensors[_name].data.depth
      d = frame.clamp(_DEPTH_NEAR, _DEPTH_FAR)
      return _CamData(rgb=None, depth=d, segmentation=None)

    _aug_sensor._compute_data = _aug_display_compute
    env.scene.sensors["head_depth_single_aug"] = _aug_sensor
    print("[play] panel 'head_depth_single_aug' shows the AUGMENTED depth the policy consumes")

  # Hardware-in-the-loop: replace the simulated head depth camera with a live ZED Mini. The sim
  # keeps running the robot (proprioception) and is what we watch dodge; only the depth sensor's
  # `_compute_data` is overridden to return live ZED depth (raw metres, [N,H,W,1]) so `DepthImageObs`
  # does its usual clamp/normalize/stack unchanged. Single-camera dodge-depth task only.
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

    # Superimpose the sim ball: capture the sim head-camera render (the display-wrapped _compute_data,
    # which returns clamped sim depth + segmentation) BEFORE we override it, and resolve the ball geom
    # id. _zed_compute then min-pools the ball-only sim depth into the live ZED frame.
    import mujoco as _mj

    _sim_compute = zed_sensor._compute_data
    _composite_ball = bool(cfg.zed_sim_ball)
    _ball_geom_id = -1
    _mj_geom = int(_mj.mjtObj.mjOBJ_GEOM)
    if _composite_ball:
      try:
        _ball_geom_id = int(env.sim.mj_model.geom("ball/ball_collision").id)
      except Exception as _e:  # noqa: BLE001 - resolution failure shouldn't crash play
        print(f"[play] zed-sim-ball: could not resolve ball geom ({_e}); superimpose disabled")
        _composite_ball = False

    def _zed_compute(
      _src=zed_source, _n=_n_env, _h=_zh, _w=_zw,
      _sim=_sim_compute, _comp=_composite_ball, _bid=_ball_geom_id, _gtype=_mj_geom,
    ):
      frame = _src.latest_frame()  # [H, W] metres (real pooled depth, or all-far in 'far' mode)
      depth = frame.view(1, _h, _w, 1).expand(_n, _h, _w, 1).contiguous()
      if _comp:
        sim = _sim()  # sim head-camera render: clamped depth [N,H,W,1] + segmentation [N,H,W,2]
        seg, sim_d = sim.segmentation, sim.depth
        if seg is not None and sim_d is not None:
          ball = ((seg[..., 1] == _gtype) & (seg[..., 0] == _bid)).unsqueeze(-1)  # [N,H,W,1]
          # Ball wins where it is in front (min-pool); real ZED depth everywhere else.
          depth = torch.where(ball, torch.minimum(depth, sim_d), depth)
      return _CamData(rgb=None, depth=depth, segmentation=None)

    zed_sensor._compute_data = _zed_compute

    # Second, display-only panel showing the FULL-RES ZED RGB + depth BEFORE min-pooling, so we can
    # see the raw camera feed alongside the 16x9 the policy actually consumes. It's a bare
    # CameraSensor (never rendered, never given a sim context -- its `_compute_data` returns the
    # live raw frame) added to scene.sensors so the viser viewer auto-creates panels for it;
    # scene.update() then invalidates its cache each step so the panels stay live. camera_idx is
    # copied from the obs sensor so the frustum/aspect are correct; a distinct camera_name avoids a
    # frustum-name clash.
    from mjlab.sensor.camera_sensor import CameraSensorCfg as _CameraSensorCfg

    # Reusable full-res RGB+depth display panel for a ZED source. viser auto-creates one RGB and one
    # depth panel per sensor, so naming the sensor 'head'/'down' yields the labeled feeds.
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

    _add_zed_feed("head", zed_source)  # real head ZED: 'head' RGB + 'head' depth panels

    # HIL: the real camera is the only legitimate stimulus -- pause sim ball throws (the viser
    # "Pause ball throws" checkbox drives the same flag, so it can be re-enabled mid-session).
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
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
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

  # Handle "auto" viewer selection.
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
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
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
