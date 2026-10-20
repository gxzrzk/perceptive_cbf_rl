"""LP-NavOA v6 tasks: locomotion backbone curriculum and the two navigation tasks.

Registered tasks, and nothing else:

Backbone curriculum (locomotion, trained in order)
  1. ``Mjlab-CircularGoal-Flat-G1Rw-Pure-v6``       circular heading-speed command
  2. ``Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6``      A* waypoint guidance, flat obstacles
  3. ``Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6`` A* guidance, obstacles + mild rough

Navigation tasks (open-wall and indoor), each with a planner-free twin
  4. ``Mjlab-StraightApproach-LowSpeed-G1-v6``       teacher reachable goals
  5. ``Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6``  planner-free inference
  6. ``Mjlab-IndoorStraightApproach-G1-v6``          teacher reachable goals
  7. ``Mjlab-IndoorStraightApproach-NoPP-G1-v6``     planner-free inference

Classroom comparison task (PAC-MAN layout), with a planner-free twin
  8. ``Mjlab-Classroom-G1-v6``                       teacher reachable goals
  9. ``Mjlab-Classroom-NoPP-G1-v6``                  planner-free inference
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl.runner_raycast import VelocityRaycastOnPolicyRunner

from .classroom_env_cfgs import (
  unitree_g1_classroom_reachable_h1_stage4_env_cfg,
  unitree_g1_classroom_reachable_nopp_h1_env_cfg,
)
from .env_cfgs import (
  unitree_g1_circular_goal_flat_g1rw_pure_h1_env_cfg,
  unitree_g1_indoor_straight_approach_reachable_h1_stage4_env_cfg,
  unitree_g1_indoor_straight_approach_reachable_nopp_h1_env_cfg,
  unitree_g1_obstacle_flat_g1rw_pure_astar_h1_env_cfg,
  unitree_g1_obstacle_mild_rough_g1rw_pure_astar_h1_env_cfg,
  unitree_g1_straight_approach_low_speed_reachable_h1_stage4_env_cfg,
  unitree_g1_straight_approach_low_speed_reachable_nopp_h1_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_raycast_h1_runner_cfg

# ==============================================================================
# Backbone curriculum
# ==============================================================================
# Stage 1: circular goal tracking on flat ground. Teaches the compact
# [heading error, target speed] command interface that every later stage reuses.
register_mjlab_task(
  task_id="Mjlab-CircularGoal-Flat-G1Rw-Pure-v6",
  env_cfg=unitree_g1_circular_goal_flat_g1rw_pure_h1_env_cfg(),
  play_env_cfg=unitree_g1_circular_goal_flat_g1rw_pure_h1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# Stage 2: flat obstacles with A* waypoint supervision. Introduces obstacle
# bypassing through intermediate waypoints while the command interface is fixed.
register_mjlab_task(
  task_id="Mjlab-ObstacleFlat-AStar-G1Rw-Pure-v6",
  env_cfg=unitree_g1_obstacle_flat_g1rw_pure_astar_h1_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_flat_g1rw_pure_astar_h1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# Stage 3: 70% obstacles + 30% mild heightfield rough terrain, still A* guided.
register_mjlab_task(
  task_id="Mjlab-ObstacleMildRough-AStar-G1Rw-Pure-v6",
  env_cfg=unitree_g1_obstacle_mild_rough_g1rw_pure_astar_h1_env_cfg(),
  play_env_cfg=unitree_g1_obstacle_mild_rough_g1rw_pure_astar_h1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# ==============================================================================
# Navigation task 1: open straight-wall, 0.5-1.2 m/s
# ==============================================================================
# The teacher stays active and supplies reachable waypoints. This is the task
# the local planner is distilled on.
register_mjlab_task(
  task_id="Mjlab-StraightApproach-LowSpeed-G1-v6",
  env_cfg=unitree_g1_straight_approach_low_speed_reachable_h1_stage4_env_cfg(),
  play_env_cfg=unitree_g1_straight_approach_low_speed_reachable_h1_stage4_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# Planner-free twin: the planner waypoints are removed, so the frozen backbone
# is steered only by the learned local planner. Used for deployment-style
# evaluation of task 1.
register_mjlab_task(
  task_id="Mjlab-StraightApproach-LowSpeed-NoPP-G1-v6",
  env_cfg=unitree_g1_straight_approach_low_speed_reachable_nopp_h1_env_cfg(),
  play_env_cfg=unitree_g1_straight_approach_low_speed_reachable_nopp_h1_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# ==============================================================================
# Navigation task 2: indoor room layout, 0.5-1.2 m/s
# ==============================================================================
register_mjlab_task(
  task_id="Mjlab-IndoorStraightApproach-G1-v6",
  env_cfg=unitree_g1_indoor_straight_approach_reachable_h1_stage4_env_cfg(),
  play_env_cfg=unitree_g1_indoor_straight_approach_reachable_h1_stage4_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-IndoorStraightApproach-NoPP-G1-v6",
  env_cfg=unitree_g1_indoor_straight_approach_reachable_nopp_h1_env_cfg(),
  play_env_cfg=unitree_g1_indoor_straight_approach_reachable_nopp_h1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# ==============================================================================
# Navigation task 3: PAC-MAN classroom layout, 0.4-0.8 m/s
# ==============================================================================
# Walled classroom with a 5x3 desk/chair grid (ported from perceptive_cbf_rl)
# for the PAC-MAN comparison. The teacher supplies straight-lane waypoints at
# the PAC-MAN cruise speed (0.6 m/s); the local planner is distilled on this.
register_mjlab_task(
  task_id="Mjlab-Classroom-G1-v6",
  env_cfg=unitree_g1_classroom_reachable_h1_stage4_env_cfg(),
  play_env_cfg=unitree_g1_classroom_reachable_h1_stage4_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)

# Planner-free twin: only the two key goals remain, so the frozen backbone is
# steered solely by the learned local planner (deployment-style evaluation).
register_mjlab_task(
  task_id="Mjlab-Classroom-NoPP-G1-v6",
  env_cfg=unitree_g1_classroom_reachable_nopp_h1_env_cfg(),
  play_env_cfg=unitree_g1_classroom_reachable_nopp_h1_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_raycast_h1_runner_cfg(),
  runner_cls=VelocityRaycastOnPolicyRunner,
)
