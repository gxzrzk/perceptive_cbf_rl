"""Keyboard-controlled velocity command for teleoperation during play mode."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


# Standalone keyboard control script that runs in a new terminal
_KEYBOARD_CONTROL_SCRIPT = '''
#!/usr/bin/env python3
"""Keyboard control UI for robot teleoperation."""

import curses
import json
import os
import sys
import time

def main(state_file: str, max_lin_x: float, max_lin_y: float, max_ang_z: float, step_ratio: float):
    # Current velocity targets
    target_lin_x = 0.0
    target_lin_y = 0.0
    target_ang_z = 0.0

    step_x = max_lin_x * step_ratio
    step_y = max_lin_y * step_ratio
    step_z = max_ang_z * step_ratio

    def write_state():
        with open(state_file, "w") as f:
            json.dump({
                "lin_x": target_lin_x,
                "lin_y": target_lin_y,
                "ang_z": target_ang_z,
            }, f)

    def draw_bar(percent: float, width: int = 20) -> str:
        percent = max(-100, min(100, percent))
        mid = width // 2
        if percent >= 0:
            fill = int(mid * percent / 100)
            return " " * mid + "█" * fill + "░" * (mid - fill)
        else:
            fill = int(mid * (-percent) / 100)
            return "░" * (mid - fill) + "█" * fill + " " * mid

    # Initialize curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    try:
        while True:
            # Handle input
            key = stdscr.getch()
            if key != -1:
                # Forward/backward (W/S or Up/Down)
                if key in (ord("w"), ord("W"), curses.KEY_UP):
                    target_lin_x = min(max_lin_x, target_lin_x + step_x)
                elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
                    target_lin_x = max(-max_lin_x, target_lin_x - step_x)

                # Left/right strafe (A/D or Left/Right)
                elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
                    target_lin_y = min(max_lin_y, target_lin_y + step_y)
                elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
                    target_lin_y = max(-max_lin_y, target_lin_y - step_y)

                # Rotation (Q/E)
                elif key in (ord("q"), ord("Q")):
                    target_ang_z = min(max_ang_z, target_ang_z + step_z)
                elif key in (ord("e"), ord("E")):
                    target_ang_z = max(-max_ang_z, target_ang_z - step_z)

                # Space to reset
                elif key == ord(" "):
                    target_lin_x = 0.0
                    target_lin_y = 0.0
                    target_ang_z = 0.0

                # Escape to exit
                elif key == 27:  # ESC
                    break

                write_state()

            # Draw UI
            try:
                stdscr.clear()
                height, width = stdscr.getmaxyx()

                # Title
                title = "═══ Robot Keyboard Control ═══"
                stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)

                # Instructions
                stdscr.addstr(2, 2, "Controls (each press = ±10%):", curses.A_UNDERLINE)
                stdscr.addstr(3, 4, "W / ↑  : Increase forward speed")
                stdscr.addstr(4, 4, "S / ↓  : Decrease forward speed")
                stdscr.addstr(5, 4, "A / ←  : Increase left strafe")
                stdscr.addstr(6, 4, "D / →  : Decrease left strafe")
                stdscr.addstr(7, 4, "Q      : Increase left rotation")
                stdscr.addstr(8, 4, "E      : Decrease left rotation")
                stdscr.addstr(9, 4, "SPACE  : Stop all (reset to zero)")
                stdscr.addstr(10, 4, "ESC    : Exit")

                # Current values
                stdscr.addstr(12, 2, "Current Command:", curses.A_UNDERLINE)

                pct_x = (target_lin_x / max_lin_x * 100) if max_lin_x != 0 else 0
                pct_y = (target_lin_y / max_lin_y * 100) if max_lin_y != 0 else 0
                pct_z = (target_ang_z / max_ang_z * 100) if max_ang_z != 0 else 0

                bar_x = draw_bar(pct_x)
                bar_y = draw_bar(pct_y)
                bar_z = draw_bar(pct_z)

                stdscr.addstr(14, 4, f"Forward (X): {target_lin_x:+6.2f} m/s   [{bar_x}] {pct_x:+4.0f}%")
                stdscr.addstr(15, 4, f"Strafe  (Y): {target_lin_y:+6.2f} m/s   [{bar_y}] {pct_y:+4.0f}%")
                stdscr.addstr(16, 4, f"Rotate  (Z): {target_ang_z:+6.2f} rad/s [{bar_z}] {pct_z:+4.0f}%")

                stdscr.refresh()
            except curses.error:
                pass

            time.sleep(0.02)

    finally:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: keyboard_ui.py <state_file> <max_lin_x> <max_lin_y> <max_ang_z> <step_ratio>")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
'''


class KeyboardListener:
  """Keyboard listener that runs in a separate terminal window."""

  def __init__(
    self,
    max_lin_vel_x: float,
    max_lin_vel_y: float,
    max_ang_vel_z: float,
    step_ratio: float = 0.1,
  ):
    self.max_lin_vel_x = max_lin_vel_x
    self.max_lin_vel_y = max_lin_vel_y
    self.max_ang_vel_z = max_ang_vel_z
    self.step_ratio = step_ratio

    # Create temp files for script and state
    self._temp_dir = tempfile.mkdtemp(prefix="mjlab_keyboard_")
    self._script_file = Path(self._temp_dir) / "keyboard_ui.py"
    self._state_file = Path(self._temp_dir) / "state.json"

    # Write initial state
    self._write_state(0.0, 0.0, 0.0)

    # Write script
    self._script_file.write_text(_KEYBOARD_CONTROL_SCRIPT)
    self._script_file.chmod(0o755)

    self._process: subprocess.Popen | None = None
    self._running = False
    self._reader_thread: threading.Thread | None = None

    # Current targets (read from file)
    self._target_lin_x = 0.0
    self._target_lin_y = 0.0
    self._target_ang_z = 0.0
    self._lock = threading.Lock()

  def _write_state(self, lin_x: float, lin_y: float, ang_z: float) -> None:
    with open(self._state_file, "w") as f:
      json.dump({"lin_x": lin_x, "lin_y": lin_y, "ang_z": ang_z}, f)

  def _read_state(self) -> tuple[float, float, float]:
    try:
      with open(self._state_file, "r") as f:
        data = json.load(f)
        return data.get("lin_x", 0.0), data.get("lin_y", 0.0), data.get("ang_z", 0.0)
    except (FileNotFoundError, json.JSONDecodeError):
      return 0.0, 0.0, 0.0

  def start(self) -> None:
    """Start the keyboard control terminal."""
    if self._running:
      return

    self._running = True

    # Find available terminal emulator
    terminal_cmd = self._find_terminal_command()
    if terminal_cmd is None:
      print("[WARN] No terminal emulator found. Keyboard control disabled.")
      return

    # Build command to run the script
    python_exe = sys.executable
    script_args = [
      str(self._script_file),
      str(self._state_file),
      str(self.max_lin_vel_x),
      str(self.max_lin_vel_y),
      str(self.max_ang_vel_z),
      str(self.step_ratio),
    ]

    # Launch in new terminal
    try:
      if "gnome-terminal" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "--title=Robot Keyboard Control",
            "--geometry=70x20",
            "--",
            python_exe,
            *script_args,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      elif "xterm" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "-title",
            "Robot Keyboard Control",
            "-geometry",
            "70x20",
            "-e",
            python_exe,
            *script_args,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      elif "konsole" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "--title",
            "Robot Keyboard Control",
            "-e",
            python_exe,
            *script_args,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      elif "xfce4-terminal" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "--title=Robot Keyboard Control",
            "--geometry=70x20",
            "-e",
            f"{python_exe} {' '.join(script_args)}",
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      else:
        # Generic fallback
        self._process = subprocess.Popen(
          [terminal_cmd, "-e", python_exe, *script_args],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )

      print(f"[INFO] Keyboard control terminal launched ({terminal_cmd})")
    except Exception as e:
      print(f"[WARN] Failed to launch terminal: {e}")
      self._running = False
      return

    # Start reader thread
    self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
    self._reader_thread.start()

    # Register cleanup
    atexit.register(self.stop)

  def _find_terminal_command(self) -> str | None:
    """Find an available terminal emulator."""
    terminals = [
      "gnome-terminal",
      "konsole",
      "xfce4-terminal",
      "xterm",
      "lxterminal",
      "mate-terminal",
      "terminator",
    ]
    for term in terminals:
      try:
        result = subprocess.run(
          ["which", term], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
          return term
      except Exception:
        continue
    return None

  def _reader_loop(self) -> None:
    """Continuously read state from file."""
    while self._running:
      try:
        lin_x, lin_y, ang_z = self._read_state()
        with self._lock:
          self._target_lin_x = lin_x
          self._target_lin_y = lin_y
          self._target_ang_z = ang_z
      except Exception:
        pass
      time.sleep(0.02)

  def stop(self) -> None:
    """Stop the keyboard listener."""
    self._running = False

    if self._process is not None:
      try:
        self._process.terminate()
        self._process.wait(timeout=2)
      except Exception:
        pass
      self._process = None

    if self._reader_thread is not None:
      self._reader_thread.join(timeout=1)
      self._reader_thread = None

    # Cleanup temp files
    try:
      import shutil

      shutil.rmtree(self._temp_dir, ignore_errors=True)
    except Exception:
      pass

  def get_targets(self) -> tuple[float, float, float]:
    """Get current velocity targets (thread-safe)."""
    with self._lock:
      return (self._target_lin_x, self._target_lin_y, self._target_ang_z)


class KeyboardVelocityCommand(CommandTerm):
  """Velocity command controlled by keyboard input in a separate terminal.

  Key bindings (each press changes by 10% of max):
    W / UP: Increase forward speed (+10% lin_vel_x)
    S / DOWN: Decrease forward speed (-10% lin_vel_x)
    A / LEFT: Increase left strafe (+10% lin_vel_y)
    D / RIGHT: Decrease left strafe (-10% lin_vel_y)
    Q: Increase left rotation (+10% ang_vel_z)
    E: Decrease left rotation (-10% ang_vel_z)
    SPACE: Stop all (reset to zero)
  """

  cfg: KeyboardVelocityCommandCfg

  def __init__(self, cfg: KeyboardVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    # Start keyboard listener in separate terminal
    self._listener = KeyboardListener(
      max_lin_vel_x=cfg.max_lin_vel_x,
      max_lin_vel_y=cfg.max_lin_vel_y,
      max_ang_vel_z=cfg.max_ang_vel_z,
      step_ratio=cfg.step_ratio,
    )
    self._listener.start()

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  def __del__(self):
    if hasattr(self, "_listener"):
      self._listener.stop()

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def handle_key(self, key: int) -> None:
    """Handle keyboard input from viewer (unused, separate terminal is used)."""
    pass

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Don't resample - keyboard controls the command
    pass

  def _update_command(self) -> None:
    """Update velocity command based on keyboard listener state."""
    # Get current targets from keyboard listener
    target_lin_x, target_lin_y, target_ang_z = self._listener.get_targets()

    # Smooth transition to target velocity
    alpha = self.cfg.smoothing_alpha
    self.vel_command_b[:, 0] = (
      alpha * target_lin_x + (1 - alpha) * self.vel_command_b[:, 0]
    )
    self.vel_command_b[:, 1] = (
      alpha * target_lin_y + (1 - alpha) * self.vel_command_b[:, 1]
    )
    self.vel_command_b[:, 2] = (
      alpha * target_ang_z + (1 - alpha) * self.vel_command_b[:, 2]
    )

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw velocity command arrows (same as UniformVelocityCommand)."""
    import numpy as np

    from mjlab.utils.lab_api.math import matrix_from_quat

    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]

    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # Command linear velocity arrow (blue)
    cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    cmd_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.8, 0.8), width=0.02
    )

    # Command angular velocity arrow (green)
    cmd_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_ang_to, color=(0.2, 0.8, 0.2, 0.8), width=0.02
    )

    # Actual linear velocity arrow (cyan)
    act_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, act_lin_to, color=(0.0, 0.8, 1.0, 0.7), width=0.015
    )


@dataclass(kw_only=True)
class KeyboardVelocityCommandCfg(CommandTermCfg):
  """Configuration for keyboard-controlled velocity command."""

  entity_name: str
  max_lin_vel_x: float = 2.0
  max_lin_vel_y: float = 0.5
  max_ang_vel_z: float = 1.0
  step_ratio: float = 0.1  # Each key press changes by this fraction of max
  smoothing_alpha: float = 0.1  # Smoothing factor for velocity transitions

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> KeyboardVelocityCommand:
    return KeyboardVelocityCommand(self, env)


def is_keyboard_control_enabled() -> bool:
  """Check if keyboard control is enabled via environment variable."""
  return get_keyboard_control_mode() in (1, 2)


def get_keyboard_control_mode() -> int | None:
  """Parse the keyboard control mode from the environment.

  Returns:
    ``1`` for the legacy keyboard goal/velocity control mode,
    ``2`` for keyboard control plus the play-time manual obstacle mode,
    or ``None`` when keyboard control is disabled.
  """
  raw_value = os.environ.get("KEYBOARD_CONTROL", "").strip().lower()
  if raw_value in ("", "0", "false", "no", "off"):
    return None
  if raw_value in ("1", "true", "yes", "on"):
    return 1
  if raw_value in ("2", "manual-obstacle", "drag-obstacle", "obstacle"):
    return 2
  return None


def is_manual_obstacle_control_enabled() -> bool:
  """Check if the manual obstacle play mode is enabled."""
  return get_keyboard_control_mode() == 2


# ==============================================================================
# Keyboard Goal Tracking Control
# ==============================================================================

# Standalone keyboard control script for goal tracking that runs in a new terminal
_KEYBOARD_GOAL_CONTROL_SCRIPT = '''
#!/usr/bin/env python3
"""Keyboard control UI for goal tracking teleoperation.

Controls:
  W/S: Move goal forward/backward on the 3m circle
  A/D: Move goal left/right on the 3m circle
  J/K: Increase/decrease target velocity by 10%
  SPACE: Reset goal to front and velocity to default
  ESC: Exit
"""

import curses
import json
import math
import sys
import time

def main(state_file: str, goal_radius: float, max_velocity: float, default_velocity: float, angle_step_deg: float):
    # Goal angle (0 = front, positive = left, negative = right)
    goal_angle = 0.0  # radians
    target_velocity = default_velocity
    velocity_step = max_velocity * 0.1  # 10% of max

    def write_state():
        # Convert angle to x, y coordinates on the circle
        goal_x = goal_radius * math.cos(goal_angle)
        goal_y = goal_radius * math.sin(goal_angle)
        with open(state_file, "w") as f:
            json.dump({
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_angle": goal_angle,
                "target_velocity": target_velocity,
            }, f)

    def draw_bar(percent: float, width: int = 20) -> str:
        percent = max(0, min(100, percent))
        fill = int(width * percent / 100)
        return "█" * fill + "░" * (width - fill)

    def draw_circle_indicator(angle: float, width: int = 15) -> str:
        """Draw a simple circle indicator showing goal position."""
        # Normalize angle to [-pi, pi]
        angle = angle % (2 * math.pi)
        if angle > math.pi:
            angle -= 2 * math.pi
        # Convert to position on indicator (left to right)
        # -pi/2 = left, 0 = center, pi/2 = right
        # We show -90 to +90 degrees range
        if angle < -math.pi / 2:
            pos = 0
        elif angle > math.pi / 2:
            pos = width - 1
        else:
            pos = int((angle + math.pi / 2) / math.pi * (width - 1))
        line = "─" * pos + "●" + "─" * (width - 1 - pos)
        return f"[{line}]"

    # Write initial state
    write_state()

    # Initialize curses
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    stdscr.keypad(True)
    stdscr.nodelay(True)

    angle_step = math.radians(angle_step_deg)

    try:
        while True:
            # Handle input
            key = stdscr.getch()
            if key != -1:
                # W: Move goal forward (toward front of robot, angle -> 0)
                if key in (ord("w"), ord("W"), curses.KEY_UP):
                    # Move angle toward 0
                    if goal_angle > 0:
                        goal_angle = max(0, goal_angle - angle_step)
                    elif goal_angle < 0:
                        goal_angle = min(0, goal_angle + angle_step)
                    # If already at 0, do nothing

                # S: Move goal backward (toward back of robot, angle -> pi or -pi)
                elif key in (ord("s"), ord("S"), curses.KEY_DOWN):
                    if goal_angle >= 0:
                        goal_angle = min(math.pi, goal_angle + angle_step)
                    else:
                        goal_angle = max(-math.pi, goal_angle - angle_step)

                # A: Move goal left (positive angle)
                elif key in (ord("a"), ord("A"), curses.KEY_LEFT):
                    goal_angle = min(math.pi, goal_angle + angle_step)

                # D: Move goal right (negative angle)
                elif key in (ord("d"), ord("D"), curses.KEY_RIGHT):
                    goal_angle = max(-math.pi, goal_angle - angle_step)

                # J: Increase velocity by 10%
                elif key in (ord("j"), ord("J")):
                    target_velocity = min(max_velocity, target_velocity + velocity_step)

                # K: Decrease velocity by 10%
                elif key in (ord("k"), ord("K")):
                    target_velocity = max(0.0, target_velocity - velocity_step)

                # Space to reset
                elif key == ord(" "):
                    goal_angle = 0.0
                    target_velocity = default_velocity

                # Escape to exit
                elif key == 27:  # ESC
                    break

                write_state()

            # Draw UI
            try:
                stdscr.clear()
                height, width = stdscr.getmaxyx()

                # Title
                title = "═══ Goal Tracking Keyboard Control ═══"
                stdscr.addstr(0, max(0, (width - len(title)) // 2), title, curses.A_BOLD)

                # Instructions
                stdscr.addstr(2, 2, "Controls:", curses.A_UNDERLINE)
                stdscr.addstr(3, 4, "W / ↑  : Move goal toward front")
                stdscr.addstr(4, 4, "S / ↓  : Move goal toward back")
                stdscr.addstr(5, 4, "A / ←  : Move goal to left")
                stdscr.addstr(6, 4, "D / →  : Move goal to right")
                stdscr.addstr(7, 4, "J      : Increase velocity (+10%)")
                stdscr.addstr(8, 4, "K      : Decrease velocity (-10%)")
                stdscr.addstr(9, 4, "SPACE  : Reset (goal front, default vel)")
                stdscr.addstr(10, 4, "ESC    : Exit")

                # Current values
                stdscr.addstr(12, 2, "Current State:", curses.A_UNDERLINE)

                # Goal position
                goal_x = goal_radius * math.cos(goal_angle)
                goal_y = goal_radius * math.sin(goal_angle)
                angle_deg = math.degrees(goal_angle)

                circle_ind = draw_circle_indicator(goal_angle)
                stdscr.addstr(14, 4, f"Goal Position: ({goal_x:+5.2f}, {goal_y:+5.2f}) m")
                stdscr.addstr(15, 4, f"Goal Angle:    {angle_deg:+6.1f}°  {circle_ind}")
                stdscr.addstr(16, 4, f"Goal Radius:   {goal_radius:.1f} m")

                # Velocity
                vel_pct = (target_velocity / max_velocity * 100) if max_velocity > 0 else 0
                vel_bar = draw_bar(vel_pct)
                stdscr.addstr(18, 4, f"Target Velocity: {target_velocity:.2f} m/s  [{vel_bar}] {vel_pct:.0f}%")

                stdscr.refresh()
            except curses.error:
                pass

            time.sleep(0.02)

    finally:
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: keyboard_goal_ui.py <state_file> <goal_radius> <max_velocity> <default_velocity> <angle_step_deg>")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
'''


class KeyboardGoalListener:
  """Keyboard listener for goal tracking that runs in a separate terminal window."""

  def __init__(
    self,
    goal_radius: float = 3.0,
    max_velocity: float = 1.3,
    default_velocity: float = 0.5,
    angle_step_deg: float = 15.0,
  ):
    self.goal_radius = goal_radius
    self.max_velocity = max_velocity
    self.default_velocity = default_velocity
    self.angle_step_deg = angle_step_deg

    # Create temp files for script and state
    self._temp_dir = tempfile.mkdtemp(prefix="mjlab_goal_keyboard_")
    self._script_file = Path(self._temp_dir) / "keyboard_goal_ui.py"
    self._state_file = Path(self._temp_dir) / "state.json"

    # Write initial state (goal at front)
    self._write_state(goal_radius, 0.0, 0.0, default_velocity)

    # Write script
    self._script_file.write_text(_KEYBOARD_GOAL_CONTROL_SCRIPT)
    self._script_file.chmod(0o755)

    self._process: subprocess.Popen | None = None
    self._running = False
    self._reader_thread: threading.Thread | None = None

    # Current state (read from file)
    self._goal_x = goal_radius  # Front of robot
    self._goal_y = 0.0
    self._goal_angle = 0.0
    self._target_velocity = default_velocity
    self._lock = threading.Lock()

  def _write_state(
    self, goal_x: float, goal_y: float, goal_angle: float, target_velocity: float
  ) -> None:
    with open(self._state_file, "w") as f:
      json.dump(
        {
          "goal_x": goal_x,
          "goal_y": goal_y,
          "goal_angle": goal_angle,
          "target_velocity": target_velocity,
        },
        f,
      )

  def _read_state(self) -> tuple[float, float, float, float]:
    try:
      with open(self._state_file, "r") as f:
        data = json.load(f)
        return (
          data.get("goal_x", self.goal_radius),
          data.get("goal_y", 0.0),
          data.get("goal_angle", 0.0),
          data.get("target_velocity", self.default_velocity),
        )
    except (FileNotFoundError, json.JSONDecodeError):
      return self.goal_radius, 0.0, 0.0, self.default_velocity

  def start(self) -> None:
    """Start the keyboard control terminal."""
    if self._running:
      return

    self._running = True

    # Find available terminal emulator
    terminal_cmd = self._find_terminal_command()
    if terminal_cmd is None:
      print("[WARN] No terminal emulator found. Keyboard goal control disabled.")
      return

    # Build command to run the script
    python_exe = sys.executable
    script_args = [
      str(self._script_file),
      str(self._state_file),
      str(self.goal_radius),
      str(self.max_velocity),
      str(self.default_velocity),
      str(self.angle_step_deg),
    ]

    # Launch in new terminal
    try:
      if "gnome-terminal" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "--title=Goal Tracking Keyboard Control",
            "--geometry=70x24",
            "--",
            python_exe,
            *script_args,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      elif "xterm" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "-title",
            "Goal Tracking Keyboard Control",
            "-geometry",
            "70x24",
            "-e",
            python_exe,
            *script_args,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      elif "konsole" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "--title",
            "Goal Tracking Keyboard Control",
            "-e",
            python_exe,
            *script_args,
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      elif "xfce4-terminal" in terminal_cmd:
        self._process = subprocess.Popen(
          [
            terminal_cmd,
            "--title=Goal Tracking Keyboard Control",
            "--geometry=70x24",
            "-e",
            f"{python_exe} {' '.join(script_args)}",
          ],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )
      else:
        # Generic fallback
        self._process = subprocess.Popen(
          [terminal_cmd, "-e", python_exe, *script_args],
          stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL,
        )

      print(f"[INFO] Goal tracking keyboard control terminal launched ({terminal_cmd})")
    except Exception as e:
      print(f"[WARN] Failed to launch terminal: {e}")
      self._running = False
      return

    # Start reader thread
    self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
    self._reader_thread.start()

    # Register cleanup
    atexit.register(self.stop)

  def _find_terminal_command(self) -> str | None:
    """Find an available terminal emulator."""
    terminals = [
      "gnome-terminal",
      "konsole",
      "xfce4-terminal",
      "xterm",
      "lxterminal",
      "mate-terminal",
      "terminator",
    ]
    for term in terminals:
      try:
        result = subprocess.run(
          ["which", term], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
          return term
      except Exception:
        continue
    return None

  def _reader_loop(self) -> None:
    """Continuously read state from file."""
    while self._running:
      try:
        goal_x, goal_y, goal_angle, target_velocity = self._read_state()
        with self._lock:
          self._goal_x = goal_x
          self._goal_y = goal_y
          self._goal_angle = goal_angle
          self._target_velocity = target_velocity
      except Exception:
        pass
      time.sleep(0.02)

  def stop(self) -> None:
    """Stop the keyboard listener."""
    self._running = False

    if self._process is not None:
      try:
        self._process.terminate()
        self._process.wait(timeout=2)
      except Exception:
        pass
      self._process = None

    if self._reader_thread is not None:
      self._reader_thread.join(timeout=1)
      self._reader_thread = None

    # Cleanup temp files
    try:
      import shutil

      shutil.rmtree(self._temp_dir, ignore_errors=True)
    except Exception:
      pass

  def get_state(self) -> tuple[float, float, float, float]:
    """Get current goal state (thread-safe).

    Returns:
      Tuple of (goal_x, goal_y, goal_angle, target_velocity)
    """
    with self._lock:
      return (self._goal_x, self._goal_y, self._goal_angle, self._target_velocity)


class KeyboardGoalTrackingCommand(CommandTerm):
  """Goal tracking command controlled by keyboard input in a separate terminal.

  The goal point exists on a circle of fixed radius (default 3m) centered on the robot.
  Users can control:
    - Goal position on the circle via W/S/A/D keys
    - Target velocity via J/K keys (range: [0, 1.3] m/s, ±10% per press)

  Key bindings:
    W / ↑: Move goal toward front (angle toward 0)
    S / ↓: Move goal toward back (angle toward ±π)
    A / ←: Move goal to left (increase angle)
    D / →: Move goal to right (decrease angle)
    J: Increase velocity (+10%)
    K: Decrease velocity (-10%)
    SPACE: Reset goal to front and velocity to default
  """

  cfg: "KeyboardGoalTrackingCommandCfg"

  def __init__(self, cfg: "KeyboardGoalTrackingCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Goal angle in world frame (0 = +X direction in world)
    # Will be initialized to robot's heading on first update
    self.goal_angle_world = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self._initialized = False  # Flag to initialize goal angle to robot heading

    # Goal position relative to robot in world frame (for visualization)
    self.goal_pos_world_rel = torch.zeros(
      self.num_envs, 2, device=self.device, dtype=torch.float32
    )
    self.goal_pos_world_rel[:, 0] = cfg.goal_radius  # Initial: +X direction

    # Goal position in robot frame (for observation functions)
    self.goal_pos_rel = torch.zeros(
      self.num_envs, 2, device=self.device, dtype=torch.float32
    )
    self.goal_pos_rel[:, 0] = cfg.goal_radius

    # Target velocity (start at 0)
    self.target_velocity = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    # Initial velocity is 0

    # Computed values for observations
    self.delta_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    self.target_distance = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self.target_distance[:] = cfg.goal_radius

    # Command output: [delta_yaw, delta_next_yaw (same as delta_yaw), target_velocity]
    self._command = torch.zeros(
      self.num_envs, 3, device=self.device, dtype=torch.float32
    )
    self._command[:, 2] = cfg.default_velocity

    # Smoothed values for smooth transitions
    self._smoothed_goal_angle = 0.0
    self._smoothed_velocity = cfg.default_velocity

    # Start keyboard listener in separate terminal
    self._listener = KeyboardGoalListener(
      goal_radius=cfg.goal_radius,
      max_velocity=cfg.max_velocity,
      default_velocity=cfg.default_velocity,
      angle_step_deg=cfg.angle_step_deg,
    )
    self._listener.start()

    # Metrics
    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["target_velocity"] = torch.zeros(self.num_envs, device=self.device)

    # For compatibility with GoalTrackingCommand interface
    # Keyboard control never completes goals (user controls manually)
    self.all_goals_completed = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )

  def __del__(self):
    if hasattr(self, "_listener"):
      self._listener.stop()

  @property
  def command(self) -> torch.Tensor:
    """Return the command tensor [delta_yaw, delta_next_yaw, target_velocity]."""
    return self._command

  @property
  def cur_goals(self) -> torch.Tensor:
    """Get current goal positions in world frame. Shape: [num_envs, 3].

    Goal is at a fixed angle in world frame (doesn't rotate with robot).
    """
    # Get robot position
    robot_pos = self.robot.data.root_link_pos_w[:, :2]

    # Compute goal position in world frame using world angle
    cos_angle = torch.cos(self.goal_angle_world)
    sin_angle = torch.sin(self.goal_angle_world)
    goal_world_x = robot_pos[:, 0] + self.cfg.goal_radius * cos_angle
    goal_world_y = robot_pos[:, 1] + self.cfg.goal_radius * sin_angle
    goal_world_z = self.robot.data.root_link_pos_w[:, 2]

    return torch.stack([goal_world_x, goal_world_y, goal_world_z], dim=1)

  @property
  def next_goals(self) -> torch.Tensor:
    """For keyboard control, next goal is the same as current goal."""
    return self.cur_goals

  def handle_key(self, key: int) -> None:
    """Handle keyboard input from viewer (unused, separate terminal is used)."""
    pass

  def _update_metrics(self) -> None:
    """Update metrics."""
    self.metrics["goal_distance"] = self.target_distance
    self.metrics["target_velocity"] = self.target_velocity

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Don't resample - keyboard controls the goal."""
    pass

  def _update_command(self) -> None:
    """Update goal tracking command based on keyboard listener state.

    Goal angle is in world frame - doesn't rotate with robot.
    """
    import math

    from mjlab.utils.lab_api.math import wrap_to_pi

    # Get current state from keyboard listener
    goal_x, goal_y, goal_angle, target_velocity = self._listener.get_state()

    # Initialize goal angle to robot's heading on first update
    if not self._initialized:
      robot_yaw = self.robot.data.heading_w  # [num_envs]
      # Set initial world angle to robot's heading (goal starts in front)
      initial_angle = float(robot_yaw[0].cpu().item()) if robot_yaw.numel() > 0 else 0.0
      self._smoothed_goal_angle = initial_angle
      self.goal_angle_world[:] = initial_angle
      self._initialized = True

    # Smooth transitions
    alpha = self.cfg.smoothing_alpha
    self._smoothed_goal_angle = (
      alpha * goal_angle + (1 - alpha) * self._smoothed_goal_angle
    )
    self._smoothed_velocity = (
      alpha * target_velocity + (1 - alpha) * self._smoothed_velocity
    )

    # Update goal angle in world frame
    self.goal_angle_world[:] = self._smoothed_goal_angle

    # Compute goal position in world frame (relative to robot position)
    goal_x_world = self.cfg.goal_radius * math.cos(self._smoothed_goal_angle)
    goal_y_world = self.cfg.goal_radius * math.sin(self._smoothed_goal_angle)
    self.goal_pos_world_rel[:, 0] = goal_x_world
    self.goal_pos_world_rel[:, 1] = goal_y_world

    # Transform to robot frame for observation functions
    robot_yaw = self.robot.data.heading_w  # [num_envs]
    cos_yaw = torch.cos(-robot_yaw)  # Negative for world->robot transform
    sin_yaw = torch.sin(-robot_yaw)
    self.goal_pos_rel[:, 0] = goal_x_world * cos_yaw - goal_y_world * sin_yaw
    self.goal_pos_rel[:, 1] = goal_x_world * sin_yaw + goal_y_world * cos_yaw

    # Update target velocity
    self.target_velocity[:] = self._smoothed_velocity

    # Compute delta yaw: angle from robot heading to goal direction in world frame
    # delta_yaw = goal_angle_world - robot_heading
    self.delta_yaw = wrap_to_pi(self.goal_angle_world - robot_yaw)

    # Target distance is always the goal radius for keyboard control
    self.target_distance[:] = self.cfg.goal_radius

    # Update command tensor
    self._command[:, 0] = self.delta_yaw  # delta_yaw
    self._command[:, 1] = self.delta_yaw  # delta_next_yaw (same for keyboard)
    self._command[:, 2] = self.target_velocity  # target_velocity

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize goal position and heading vector."""
    import numpy as np

    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    robot_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    goal_world = self.cur_goals[batch].cpu().numpy()

    # Skip if robot appears uninitialized
    if np.linalg.norm(robot_pos) < 1e-6:
      return

    # Draw goal sphere (green)
    goal_pos = goal_world.copy()
    goal_pos[2] = robot_pos[2]  # Keep goal at robot height
    visualizer.add_sphere(goal_pos, radius=0.15, color=(0.0, 1.0, 0.0, 0.9))

    # Draw circle around robot showing goal path
    num_points = 32
    for i in range(num_points):
      angle1 = 2 * np.pi * i / num_points
      angle2 = 2 * np.pi * (i + 1) / num_points
      p1 = np.array(
        [
          robot_pos[0] + self.cfg.goal_radius * np.cos(angle1),
          robot_pos[1] + self.cfg.goal_radius * np.sin(angle1),
          robot_pos[2],
        ]
      )
      p2 = np.array(
        [
          robot_pos[0] + self.cfg.goal_radius * np.cos(angle2),
          robot_pos[1] + self.cfg.goal_radius * np.sin(angle2),
          robot_pos[2],
        ]
      )
      visualizer.add_cylinder(p1, p2, radius=0.01, color=(0.3, 0.3, 0.8, 0.3))

    # Draw heading arrow from robot to goal
    arrow_start = robot_pos.copy()
    arrow_start[2] += 0.5  # Raise arrow above robot

    direction = goal_pos[:2] - robot_pos[:2]
    dist = np.linalg.norm(direction)
    if dist > 0.1:
      direction = direction / dist
      target_vel = float(self.target_velocity[batch].cpu())
      arrow_length = max(0.3, abs(target_vel) * 0.5)
      arrow_end = arrow_start.copy()
      arrow_end[0] = arrow_start[0] + direction[0] * arrow_length
      arrow_end[1] = arrow_start[1] + direction[1] * arrow_length
    else:
      arrow_end = arrow_start.copy()
      arrow_end[0] += 0.3

    visualizer.add_arrow(arrow_start, arrow_end, color=(0.2, 0.8, 0.2, 0.8), width=0.02)


@dataclass(kw_only=True)
class KeyboardGoalTrackingCommandCfg(CommandTermCfg):
  """Configuration for keyboard-controlled goal tracking command.

  The goal point exists on a circle of fixed radius centered on the robot.
  W/S/A/D keys move the goal around the circle.
  J/K keys control the target velocity.
  """

  entity_name: str
  """Name of the robot entity in the scene."""

  goal_radius: float = 3.0
  """Radius of the goal circle around the robot [m]."""

  max_velocity: float = 1.3
  """Maximum target velocity [m/s]."""

  default_velocity: float = 0.0
  """Default target velocity [m/s]. Starts at 0."""

  angle_step_deg: float = 15.0
  """Angle step per key press [degrees]."""

  smoothing_alpha: float = 0.15
  """Smoothing factor for goal and velocity transitions."""

  @dataclass
  class VizCfg:
    """Visualization configuration."""

    z_offset: float = 0.5
    scale: float = 1.0

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> KeyboardGoalTrackingCommand:
    return KeyboardGoalTrackingCommand(self, env)


# ==============================================================================
# Keyboard Circular Goal Tracking Control
# ==============================================================================


class KeyboardCircularGoalTrackingCommand(CommandTerm):
  """Circular goal tracking command controlled by keyboard input.

  This command is compatible with CircularGoalTrackingCommand interface,
  outputting [delta_yaw, target_velocity] (2D command).

  The goal point exists on a circle of fixed radius centered on the robot.
  Users can control:
    - Goal position on the circle via W/S/A/D keys
    - Target velocity via J/K keys

  Key bindings:
    W / ↑: Move goal toward front (angle toward 0)
    S / ↓: Move goal toward back (angle toward ±π)
    A / ←: Move goal to left (increase angle)
    D / →: Move goal to right (decrease angle)
    J: Increase velocity (+10%)
    K: Decrease velocity (-10%)
    SPACE: Reset goal to front and velocity to default
  """

  cfg: "KeyboardCircularGoalTrackingCommandCfg"

  def __init__(
    self, cfg: "KeyboardCircularGoalTrackingCommandCfg", env: "ManagerBasedRlEnv"
  ):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Goal angle in world frame (0 = +X direction in world)
    self.goal_angle_w = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self._initialized = False

    # Target velocity
    self.target_velocity = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.float32
    )
    self.target_velocity[:] = cfg.default_velocity

    # Computed values for observations
    self.delta_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    self.goal_pos_w = torch.zeros(
      self.num_envs, 3, device=self.device, dtype=torch.float32
    )

    # Command output: [delta_yaw, target_velocity] - 2D like CircularGoalTrackingCommand
    self._command = torch.zeros(
      self.num_envs, 2, device=self.device, dtype=torch.float32
    )

    # Smoothed values for smooth transitions
    self._smoothed_goal_angle = 0.0
    self._smoothed_velocity = cfg.default_velocity

    # Start keyboard listener in separate terminal
    self._listener = KeyboardGoalListener(
      goal_radius=cfg.goal_radius,
      max_velocity=cfg.max_velocity,
      default_velocity=cfg.default_velocity,
      angle_step_deg=cfg.angle_step_deg,
    )
    self._listener.start()

    # Metrics
    self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)

  def __del__(self):
    if hasattr(self, "_listener"):
      self._listener.stop()

  @property
  def command(self) -> torch.Tensor:
    """Return the command tensor [delta_yaw, target_velocity]."""
    return self._command

  def handle_key(self, key: int) -> None:
    """Handle keyboard input from viewer (unused, separate terminal is used)."""
    pass

  def _update_metrics(self) -> None:
    """Update metrics."""
    # Velocity towards goal direction
    lin_vel_b = self.robot.data.root_link_lin_vel_b[:, :2]
    vel_towards_goal = lin_vel_b[:, 0] * torch.cos(self.delta_yaw) + lin_vel_b[
      :, 1
    ] * torch.sin(self.delta_yaw)
    self.metrics["heading_error"] = torch.abs(self.delta_yaw)
    self.metrics["velocity_error"] = torch.abs(vel_towards_goal - self.target_velocity)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Don't resample - keyboard controls the goal."""
    pass

  def _update_command(self) -> None:
    """Update goal tracking command based on keyboard listener state."""

    from mjlab.utils.lab_api.math import wrap_to_pi

    # Get current state from keyboard listener
    goal_x, goal_y, goal_angle, target_velocity = self._listener.get_state()

    # Initialize goal angle to robot's heading on first update
    if not self._initialized:
      robot_yaw = self.robot.data.heading_w
      initial_angle = float(robot_yaw[0].cpu().item()) if robot_yaw.numel() > 0 else 0.0
      self._smoothed_goal_angle = initial_angle
      self.goal_angle_w[:] = initial_angle
      self._initialized = True

    # Smooth transitions
    alpha = self.cfg.smoothing_alpha
    self._smoothed_goal_angle = (
      alpha * goal_angle + (1 - alpha) * self._smoothed_goal_angle
    )
    self._smoothed_velocity = (
      alpha * target_velocity + (1 - alpha) * self._smoothed_velocity
    )

    # Update goal angle in world frame
    self.goal_angle_w[:] = self._smoothed_goal_angle

    # Update target velocity
    self.target_velocity[:] = self._smoothed_velocity

    # Get robot position and heading
    robot_pos = self.robot.data.root_link_pos_w[:, :2]
    robot_yaw = self.robot.data.heading_w

    # Compute goal position in world frame
    goal_x_w = robot_pos[:, 0] + self.cfg.goal_radius * torch.cos(self.goal_angle_w)
    goal_y_w = robot_pos[:, 1] + self.cfg.goal_radius * torch.sin(self.goal_angle_w)
    self.goal_pos_w[:, 0] = goal_x_w
    self.goal_pos_w[:, 1] = goal_y_w
    self.goal_pos_w[:, 2] = self.robot.data.root_link_pos_w[:, 2]

    # Compute delta_yaw: difference between robot heading and goal direction
    self.delta_yaw = wrap_to_pi(self.goal_angle_w - robot_yaw)

    # Update command tensor [delta_yaw, target_velocity]
    self._command[:, 0] = self.delta_yaw
    self._command[:, 1] = self.target_velocity

    # Apply safety velocity clipping (overrides target_velocity near obstacles)
    if self.cfg.safety_enabled:
      self._apply_safety_velocity_clip()

  def _apply_safety_velocity_clip(self) -> None:
    """Clip target velocity using directional time-to-stop limits."""
    from .safety_velocity import (
      compute_safe_speed_limit,
      damp_blocked_delta_yaw,
      min_raycast_clearance,
    )

    cfg = self.cfg
    front_clear = min_raycast_clearance(
      self._env.scene[cfg.safety_forward_sensor].data.distances,
      skip_ground_cols=cfg.safety_skip_ground_cols,
    )
    left_clear = min_raycast_clearance(
      self._env.scene[cfg.safety_left_sensor].data.distances,
      skip_ground_cols=cfg.safety_skip_ground_cols,
    )
    right_clear = min_raycast_clearance(
      self._env.scene[cfg.safety_right_sensor].data.distances,
      skip_ground_cols=cfg.safety_skip_ground_cols,
    )

    # Front-only speed limit — side clearances handled by yaw damping.
    safe_limit = compute_safe_speed_limit(
      front_clear,
      cfg.safety_margin_m,
      cfg.safety_reaction_time_s,
      cfg.safety_braking_time_s,
    )

    clipped = torch.minimum(self.target_velocity, safe_limit)

    # Preserve turning: when front blocked + large delta_yaw + side clear,
    # keep a minimum velocity so the policy can rotate in place.
    abs_yaw = torch.abs(self.delta_yaw)
    turn_needed = abs_yaw > cfg.safety_turn_threshold_rad
    front_blocked = front_clear < (cfg.safety_margin_m + cfg.safety_stop_epsilon)
    side_clear = (left_clear > cfg.safety_hard_stop_m) | (
      right_clear > cfg.safety_hard_stop_m
    )
    allow_turn = turn_needed & front_blocked & side_clear
    min_vel = cfg.safety_min_turn_velocity
    clipped = torch.where(
      allow_turn & (clipped < min_vel),
      torch.full_like(clipped, min_vel),
      clipped,
    )
    clipped = torch.where(
      ~allow_turn & (clipped < cfg.safety_stop_epsilon),
      torch.zeros_like(clipped),
      clipped,
    )
    self.target_velocity[:] = clipped

    self.delta_yaw[:] = damp_blocked_delta_yaw(
      self.delta_yaw,
      front_clearance_m=front_clear,
      left_clearance_m=left_clear,
      right_clearance_m=right_clear,
      stop_distance_m=cfg.safety_hard_stop_m,
      blocked_turn_scale=cfg.safety_blocked_turn_scale,
    )

    self._command[:, 0] = self.delta_yaw
    self._command[:, 1] = self.target_velocity

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Visualize goal point, heading vector, and velocity information."""
    import numpy as np

    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    robot_pos = self.robot.data.root_link_pos_w[batch].cpu().numpy()
    goal_pos = self.goal_pos_w[batch].cpu().numpy()
    target_vel = float(self.target_velocity[batch].item())

    # Get actual velocity towards goal
    lin_vel_b = self.robot.data.root_link_lin_vel_b[batch, :2].cpu().numpy()
    delta_yaw = float(self.delta_yaw[batch].item())
    actual_vel = float(
      lin_vel_b[0] * np.cos(delta_yaw) + lin_vel_b[1] * np.sin(delta_yaw)
    )
    velocity_error = actual_vel - target_vel

    if np.linalg.norm(robot_pos) < 1e-6:
      return

    z_offset = self.cfg.viz.z_offset

    # Print velocity information to console
    print(
      f"\r[Velocity] Target: {target_vel:+.2f} m/s | "
      f"Actual: {actual_vel:+.2f} m/s | "
      f"Error: {velocity_error:+.2f} m/s | "
      f"Heading Error: {np.degrees(delta_yaw):+.1f}°",
      end="",
    )

    # Draw goal point as a sphere
    goal_vis_pos = goal_pos.copy()
    goal_vis_pos[2] += z_offset
    visualizer.add_sphere(
      center=goal_vis_pos,
      radius=0.15,
      color=(0.2, 0.8, 0.2, 0.8),
    )

    # Draw line from robot to goal
    robot_vis_pos = robot_pos.copy()
    robot_vis_pos[2] += z_offset
    visualizer.add_arrow(
      robot_vis_pos,
      goal_vis_pos,
      color=(0.2, 0.6, 0.8, 0.6),
      width=0.02,
    )

    # Draw robot heading direction
    robot_yaw = float(self.robot.data.heading_w[batch].item())
    heading_end = robot_pos.copy()
    heading_end[0] += 0.8 * np.cos(robot_yaw)
    heading_end[1] += 0.8 * np.sin(robot_yaw)
    heading_end[2] += z_offset
    visualizer.add_arrow(
      robot_vis_pos,
      heading_end,
      color=(1.0, 0.5, 0.0, 0.8),  # Orange - robot heading
      width=0.02,
    )

    # Draw target velocity indicator (purple arrow - command velocity)
    vel_scale = 0.3 * target_vel
    target_vel_end = robot_pos.copy()
    target_vel_end[0] += vel_scale * np.cos(robot_yaw)
    target_vel_end[1] += vel_scale * np.sin(robot_yaw)
    target_vel_end[2] += z_offset + 0.25
    visualizer.add_arrow(
      np.array([robot_vis_pos[0], robot_vis_pos[1], robot_vis_pos[2] + 0.25]),
      target_vel_end,
      color=(0.8, 0.2, 0.8, 0.7),  # Purple - target velocity
      width=0.015,
      label="target_vel",
    )

    # Draw actual velocity indicator (cyan arrow - actual velocity)
    actual_vel_scale = 0.3 * actual_vel
    actual_vel_end = robot_pos.copy()
    actual_vel_end[0] += actual_vel_scale * np.cos(robot_yaw)
    actual_vel_end[1] += actual_vel_scale * np.sin(robot_yaw)
    actual_vel_end[2] += z_offset + 0.35
    visualizer.add_arrow(
      np.array([robot_vis_pos[0], robot_vis_pos[1], robot_vis_pos[2] + 0.35]),
      actual_vel_end,
      color=(0.2, 0.8, 0.8, 0.7),  # Cyan - actual velocity
      width=0.015,
      label="actual_vel",
    )

    # Draw velocity error indicator
    # Green if error < 0.2, yellow if < 0.5, red otherwise
    abs_error = abs(velocity_error)
    if abs_error < 0.2:
      error_color = (0.2, 0.9, 0.2, 0.8)  # Green - good tracking
    elif abs_error < 0.5:
      error_color = (0.9, 0.9, 0.2, 0.8)  # Yellow - moderate error
    else:
      error_color = (0.9, 0.2, 0.2, 0.8)  # Red - large error

    # Draw a small sphere above robot indicating tracking quality
    error_indicator_pos = robot_pos.copy()
    error_indicator_pos[2] += z_offset + 0.5
    visualizer.add_sphere(
      center=error_indicator_pos,
      radius=0.08,
      color=error_color,
      label="tracking_quality",
    )

    # Draw circle showing goal path
    num_points = 32
    for i in range(num_points):
      angle1 = 2 * np.pi * i / num_points
      angle2 = 2 * np.pi * (i + 1) / num_points
      p1 = np.array(
        [
          robot_pos[0] + self.cfg.goal_radius * np.cos(angle1),
          robot_pos[1] + self.cfg.goal_radius * np.sin(angle1),
          robot_pos[2] + z_offset * 0.5,
        ]
      )
      p2 = np.array(
        [
          robot_pos[0] + self.cfg.goal_radius * np.cos(angle2),
          robot_pos[1] + self.cfg.goal_radius * np.sin(angle2),
          robot_pos[2] + z_offset * 0.5,
        ]
      )
      visualizer.add_cylinder(p1, p2, radius=0.01, color=(0.3, 0.3, 0.8, 0.3))


@dataclass(kw_only=True)
class KeyboardCircularGoalTrackingCommandCfg(CommandTermCfg):
  """Configuration for keyboard-controlled circular goal tracking command.

  This is compatible with CircularGoalTrackingCommand, outputting
  [delta_yaw, target_velocity] (2D command).
  """

  entity_name: str
  """Name of the robot entity in the scene."""

  goal_radius: float = 3.0
  """Radius of the goal circle around the robot [m]."""

  max_velocity: float = 2.0
  """Maximum target velocity [m/s]."""

  default_velocity: float = 0.5
  """Default target velocity [m/s]."""

  angle_step_deg: float = 15.0
  """Angle step per key press [degrees]."""

  smoothing_alpha: float = 0.15
  """Smoothing factor for goal and velocity transitions."""

  # ── Safety velocity envelope (optional, disabled by default) ──
  safety_enabled: bool = False
  """Enable time-to-stop safety clipping of target velocity."""

  safety_forward_sensor: str = "raycast"
  """Front raycast sensor name for frontal clearance."""

  safety_left_sensor: str = "raycast_left"
  """Left raycast sensor name."""

  safety_right_sensor: str = "raycast_right"
  """Right raycast sensor name."""

  safety_skip_ground_cols: int = 3
  """Ignore ground-facing raycast columns."""

  safety_margin_m: float = 0.25
  """Standstill clearance to preserve after braking [m]."""

  safety_reaction_time_s: float = 0.15
  """Assumed reaction delay before braking starts [s]."""

  safety_braking_time_s: float = 0.8
  """Assumed time to decelerate to zero once braking begins [s]."""

  safety_stop_epsilon: float = 0.15
  """Snap small clipped speeds to zero [m/s]."""

  safety_side_offset_m: float = 0.35
  """Forward lever arm for yaw-to-side speed conversion [m]."""

  safety_hard_stop_m: float = 0.40
  """Distance at which yaw into blocked side is damped [m]."""

  safety_blocked_turn_scale: float = 0.0
  """Multiplier for yaw into a blocked direction (0 = full block)."""

  safety_min_turn_velocity: float = 0.20
  """Minimum target velocity when turning away from obstacles [m/s]."""

  safety_turn_threshold_rad: float = 0.35
  """delta_yaw above which the robot is considered to need turning [rad]."""

  @dataclass
  class VizCfg:
    """Visualization configuration."""

    z_offset: float = 0.3
    scale: float = 1.0

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: "ManagerBasedRlEnv") -> KeyboardCircularGoalTrackingCommand:
    return KeyboardCircularGoalTrackingCommand(self, env)
