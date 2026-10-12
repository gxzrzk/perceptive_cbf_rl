"""动作数据 CSV 转 NPZ 格式转换与运动学前向求解工具 (CSV to NPZ Motion Converter)

本脚本用于将 36 列的机器人动作捕捉 CSV 文件转换为包含前向运动学全身各刚体位姿与数值微分速度的 NPZ 文件，
供强化学习中的对抗运动先验 (AMP, Adversarial Motion Priors) 或全身跟踪任务 (Tracking) 作为高质量专家参考数据集。

1. 数据转换与前向运动学计算原理 (Conversion Pipeline):
   - 数据加载与四元数重排 (`MotionLoader._load_motion`):
     * 读取 36 列 CSV（支持通过 `--line-range` 裁剪有效动作片段行数）。
     * 提取根节点平移 `(x, y, z)`、根节点姿态四元数（由 CSV 的 xyzw 格式重排为 MuJoCo 所需的 wxyz 格式）、以及 29 个关节角。
   - 时间序列插值重采样 (`MotionLoader._interpolate_motion`):
     * 将原始动作帧率（如 30Hz 或 120Hz）统一重采样至仿真控制帧率（默认 50Hz，即 dt=0.02s）。
     * 根节点位移与关节角度使用一维线性插值 (LERP)。
     * 根节点旋转四元数采用严格的球面线性插值 (SLERP)，避免姿态旋转畸变。
   - 速度数值微分计算 (`MotionLoader._compute_velocities`):
     * 根节点线速度与关节角速度采用中心有限差分梯度 (`torch.gradient`) 计算。
     * 根节点旋转角速度采用 SO(3) 李代数差分 (`_so3_derivative`)：基于相邻四元数相对旋转计算等效旋转向量与角速度。
   - 仿真运动学求解与连杆数据采集 (`run_sim`):
     * 构建 `unitree_g1_flat_tracking_env_cfg` 的 MuJoCo 仿真环境。
     * 将重采样后的根节点位姿与 29 关节角度逐帧写入机器人状态并执行 `sim.forward()`。
     * 从 MuJoCo 底层提取并记录全身每个刚体 (Body link) 在世界坐标系下的位置 (`body_pos_w`)、四元数 (`body_quat_w`)、
       线速度 (`body_lin_vel_w`) 以及角速度 (`body_ang_vel_w`)。
     * 严格校验数值微分得到的根节点线速度与角速度是否与 MuJoCo 内部计算一致。
     * 将所有状态统一打包持久化保存为 `.npz` 文件。
   - 渲染与视频导出 (可选):
     * 支持交互式窗口播放 (`--render-backend window`)，可按真实时间比例对其进行慢放或快放。
     * 支持离屏渲染 (`--render-backend offscreen`)，通过 `imageio` 输出跟随机器人机体视角的 `.mp4` 高清预览视频。

2. 命令行参数解析机制 (tyro CLI):
   - 输入输出控制:
     * `--input-file`: 单个输入 CSV 文件路径。
     * `--output-name`: 输出 NPZ 文件名称（搭配 `--input-file` 使用）。
     * `--input-dir`: 包含多个 CSV 文件的输入目录（用于批量转换目录下的全部 CSV）。
     * `--output-dir`: 转换后 NPZ 文件的输出保存目录（默认 `./motion_data_npz/amp/Recovery`）。
     * `--input-fps`: 输入 CSV 的采样帧率（例如 30.0 或 120.0）。
     * `--output-fps`: 期望输出的目标帧率（默认 50.0 Hz，对齐强化学习控制周期）。
     * `--line-range`: CSV 读取行区间元组（1-based，如 `"(408, 1953)"`）。
   - 渲染与可视化控制:
     * `--render`: 是否开启可视化渲染（默认 False）。
     * `--render-backend`: 渲染后端，可选 `window` (原生 GLFW 交互窗口) 或 `offscreen` (离屏生成视频)。
     * `--window-realtime`: 窗口播放是否锁定时钟与现实物理时间保持同步。
     * `--window-realtime-scale`: 窗口回放速度缩放系数（如 1.0 为原速，0.5 为半速慢放）。
     * `--video-output`: 离屏视频的输出路径 (.mp4)。

运行示例:
    # 单文件转换并带原生窗口可视化:
    python scripts/csv_to_npz.py \\
      --input-file motion_data_csv/lafan1/fallAndGetUp1_subject1.csv \\
      --output-name fallAndGetUp1_subject1.npz \\
      --input-fps 30 --output-fps 50 \\
      --line-range "(408,1953)" \\
      --render True --render-backend window --window-realtime True

    # 单文件转换并离屏录制视频:
    python scripts/csv_to_npz.py \\
      --input-file src/assets/motions/g1/dance1_subject2.csv \\
      --output-name dance1_subject2.npz \\
      --input-fps 30 --output-fps 50 \\
      --render True --render-backend offscreen \\
      --video-output src/assets/motions/g1/dance1_subject2.mp4

    # 批量转换整个目录:
    python scripts/csv_to_npz.py \\
      --input-dir motion_data_csv/amp \\
      --output-dir motion_data_npz/amp \\
      --input-fps 120 --output-fps 50
"""


from pathlib import Path
import time
from typing import Any, Literal

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """从 CSV 文件加载原始动作捕捉数据并解析根节点位姿与关节角度。
    (Loads the motion from the csv file.)

    解析原理与数据布局:
    - 行切片支持: 若指定 line_range，则通过 skiprows 与 max_rows 精准截取有效帧区间。
    - 根节点平移: 前 3 列为世界坐标平移 (x, y, z)。
    - 四元数顺序重组: CSV 中存储为 [qx, qy, qz, qw]，通过索引重排为 MuJoCo 要求的 [qw, qx, qy, qz] (WXYZ) 规范。
    - 关节自由度: 后 29 列为机器人 29 个可控关节的目标角度（弧度）。
    """
    if self.line_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.line_range[0] - 1,
          max_rows=self.line_range[1] - self.line_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    # 可选的高程微调偏移量（根据特定动作捕捉地面校准需求启用）: motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # 将四元数从 xyzw 重排为 MuJoCo 所需的 wxyz 格式 (convert to wxyz)
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """将动作捕捉轨迹从源采样率重采样插值至目标仿真输出帧率 output_fps。
    (Interpolates the motion to the output fps.)

    算法原理:
    - 平移坐标与关节角度在欧氏空间中连续，使用一维线性插值 (LERP)。
    - 根节点三维旋转属于 SO(3) 流形，使用球面线性插值 (SLERP)，确保角速度平滑无奇异点。
    """
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """在两个欧氏张量之间执行标准线性插值 (Linear interpolation between two tensors)。"""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """在两个单位四元数之间执行球面线性插值 (Spherical linear interpolation between two quaternions)。"""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算连续时间在离散帧之间的插值权重因子与邻近两帧索引。
    (Computes the frame blend for the motion.)
    """
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """计算动作序列的线速度与角速度等物理微分量。
    (Computes the velocities of the motion.)

    计算方法:
    - 根节点线速度与关节速度: 对时间步 output_dt 采用中心差分有限差分梯度计算 (torch.gradient)。
    - 根节点角速度: 调用 _so3_derivative 在 SO(3) 旋转李代数流形上进行数值微分。
    """
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """计算 SO(3) 旋转四元数时间序列的旋转角速度数值导数。
    (Computes the derivative of a sequence of SO3 rotations.)

    数学原理:
    - 计算两帧之间的相对旋转四元数: q_rel = q_{t+1} * q_{t-1}^{-1}
    - 提取相对旋转的轴角向量并除以双倍步长 (2*dt)，得到瞬时空间角速度矢量 omega。
    - 首尾帧边界采用最近邻复制填充以保持输出维度与输入时间步完全对齐。

    参数:
      rotations: 姿态四元数序列，维度为 (B, 4)。
      dt: 离散时间步长 (秒)。
    返回:
      在世界坐标系下的三维角速度序列，维度为 (B, 3)。
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # 相对旋转四元数张量，形状为 (B−2, 4) (shape (B−2, 4))

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # 瞬时角速度向量，形状为 (B−2, 3) (shape (B−2, 3))
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # 边界补偿：重复首尾两帧样本以对齐原始序列长度 (repeat first and last sample)
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """获取下一帧运动状态，包括根节点位姿、速度与关节位置、速度。
    (Gets the next state of the motion.)
    """
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_name,
  output_dir,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
  window_viewer: Any | None = None,
  video_output: str | None = None,
  window_realtime: bool = False,
  window_realtime_scale: float = 1.0,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    if window_viewer is not None:
      print("Rendering enabled - showing native MuJoCo window...")
    else:
      print("Rendering enabled - generating offscreen video frames...")

  # 创建终端处理进度条 (Create progress bar)
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  wall_start_time = time.perf_counter()
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())
    if render and window_viewer is not None:
      if not window_viewer.is_running():
        print("Window closed by user, stopping simulation loop.")
        pbar.close()
        break

      if sim.mj_model.nq > 0:
        sim.mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
        sim.mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
      if sim.mj_model.nmocap > 0:
        sim.mj_data.mocap_pos[:] = sim.data.mocap_pos[0].cpu().numpy()
        sim.mj_data.mocap_quat[:] = sim.data.mocap_quat[0].cpu().numpy()
      sim.mj_data.xfrc_applied[:] = sim.data.xfrc_applied[0].cpu().numpy()
      mujoco.mj_forward(sim.mj_model, sim.mj_data)
      window_viewer.sync()

      if window_realtime:
        sim_elapsed = frame_count / output_fps
        target_elapsed = sim_elapsed / max(window_realtime_scale, 1e-6)
        now_elapsed = time.perf_counter() - wall_start_time
        sleep_s = target_elapsed - now_elapsed
        if sleep_s > 0:
          time.sleep(sleep_s)

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # 每 100 帧更新一次描述信息以避免终端高频刷新 (Update every 100 frames to avoid spam)
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        np.savez(str(output_dir_path / output_name), **log)  # type: ignore[arg-type] (静态类型检查忽略字典展开类型标注)

        if render and renderer is not None and frames:
          mp4_path = Path(video_output) if video_output is not None else None
          if mp4_path is None:
            default_mp4_name = Path(output_name).with_suffix(".mp4").name
            mp4_path = output_dir_path / default_mp4_name
          mp4_path.parent.mkdir(parents=True, exist_ok=True)

          try:
            import imageio.v3 as iio
          except ImportError as exc:
            raise RuntimeError(
              "Saving mp4 requires imageio. Install with: pip install imageio[ffmpeg]"
            ) from exc

          print(f"Saving offscreen video to: {mp4_path}")
          iio.imwrite(str(mp4_path), np.stack(frames, axis=0), fps=output_fps)


def main(
  input_file: str | None = None,
  output_name: str | None = None,
  input_dir: str | None = None,
  output_dir: str = "./motion_data_npz/amp/Recovery",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  render_backend: Literal["offscreen", "window"] = "offscreen",
  window_realtime: bool = False,
  window_realtime_scale: float = 1.0,
  video_output: str | None = None,
  render_entity_name: str | None = "robot",
  line_range: tuple[int, int] | None = None,
):
  """在 MuJoCo 仿真中回放动作 CSV 轨迹并将求解得到的完整刚体运动学状态保存为 NPZ 文件。
  (Replay motion from CSV file and output to npz file.)

  参数说明 (Args):
    input_file: 单个输入 CSV 动作捕捉文件路径 (Path to a single input CSV file)。
    output_name: 输出 NPZ 文件名称 (搭配 input_file 单文件模式使用)。
    input_dir: 批量处理模式下的 CSV 文件输入目录 (Directory containing CSV files for batch conversion)。
    output_dir: 转换生成的 NPZ 文件的输出保存目录 (Directory to save output npz files)。
    input_fps: 原始 CSV 文件的采样帧率 (Frame rate of the CSV file)。
    output_fps: 期望的目标输出控制帧率 (Desired output frame rate，通常为 50Hz)。
    device: 计算与仿真设备 (Device to use，如 'cuda:0' 或 'cpu')。
    render: 是否开启仿真可视化回放 (Whether to render the simulation)。
    render_backend: 渲染引擎后端，可选 'offscreen' (离屏视频录制) 或 'window' (原生 GLFW 交互窗口)。
    window_realtime: 窗口播放时是否与真实物理墙钟时间保持同步 (Keep window playback aligned with wall-clock time)。
    window_realtime_scale: 窗口播放的时间缩放系数 (1.0 代表原速实时回放，0.5 代表慢动作)。
    video_output: 离屏渲染导出的 MP4 视频路径 (仅在 offscreen 后端有效)。
    render_entity_name: 追踪渲染的目标实体名称 (当视角为 ASSET_ROOT 时跟随该实体机体)。
    line_range: 从 CSV 文件中截取的行数区间元组 (1-based 双闭区间)。
  """
  if input_file is None and input_dir is None:
    raise ValueError("Either --input_file or --input_dir must be specified.")

  if input_dir is not None:
    csv_files = sorted(Path(input_dir).glob("*.csv"))
    if not csv_files:
      raise FileNotFoundError(f"No CSV files found in {input_dir}")
    file_pairs = [(str(f), f.with_suffix(".npz").name) for f in csv_files]
    print(f"Found {len(csv_files)} CSV files in {input_dir}")
  else:
    assert input_file is not None
    if output_name is None:
      output_name = Path(input_file).with_suffix(".npz").name
    file_pairs = [(input_file, output_name)]
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render and render_backend == "offscreen":
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )

    # 在多实体仿真场景中，当相机视角类型设为 ASSET_ROOT 时，OffscreenRenderer 必须显式指定目标 entity_name (OffscreenRenderer needs entity_name when ASSET_ROOT is used in a multi-entity scene)
    if viewer_cfg.origin_type == ViewerConfig.OriginType.ASSET_ROOT:
      if render_entity_name is not None:
        viewer_cfg.entity_name = render_entity_name
      elif len(scene.entities) == 1:
        viewer_cfg.entity_name = next(iter(scene.entities.keys()))

    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ]

  for i, (cur_input_file, cur_output_name) in enumerate(file_pairs):
    if len(file_pairs) > 1:
      print(f"\n{'='*60}")
      print(f"Processing file {i + 1}/{len(file_pairs)}: {cur_input_file}")
      print(f"{'='*60}")

    common_kwargs = dict(
      sim=sim,
      scene=scene,
      joint_names=joint_names,
      input_fps=input_fps,
      input_file=cur_input_file,
      output_fps=output_fps,
      output_name=cur_output_name,
      output_dir=output_dir,
      render=render,
      line_range=line_range,
      renderer=renderer,
      video_output=video_output,
      window_realtime=window_realtime,
      window_realtime_scale=window_realtime_scale,
    )

    if render and render_backend == "window":
      with mj_viewer.launch_passive(sim.mj_model, sim.mj_data) as window_viewer:
        run_sim(**common_kwargs, window_viewer=window_viewer)
    else:
      run_sim(**common_kwargs)


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
