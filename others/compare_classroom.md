# Classroom paired comparison

Run from the repository root after the LP stage 7 final checkpoint exists:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python others/compare_classroom.py \
  --episodes 128 --batch-size 32
```

The script loads PAC-MAN's pure-navigation classroom checkpoint and LP-NavOA's
classroom local planner by default. Each model runs in its own Python environment
and native classroom task. Override the model paths with `--pac-checkpoint`,
`--lp-backbone`, and `--lp-planner`. `--only pac` or `--only lp` runs one side;
`--routes-only` creates the manifest without loading either model. The selected
GPU is relative to `CUDA_VISIBLE_DEVICES`.

Every case uses the same three-point route in the PAC-MAN frame: a spawn in one
of the two middle aisles, a forward leg to a randomly selected cross aisle,
then a random left or right turn into the adjacent lane. The route JSON is
generated once using `--seed` and replayed in both simulations. The LP runner
maps the points into its terrain cell and reflects them about the classroom's
symmetry axis, preserving left and right turns relative to each robot's heading.
PAC-MAN uses its
native rounded path follower; LP uses its native local planner with two
sequential key goals. Both use a 0.6 m/s command, no ball throws, a 0.8 m
arrival radius, and a 40 s evaluation horizon by default. PAC-MAN's comparison
checkpoint was trained at 0.8 m/s; this evaluation commands 0.6 m/s so the
two tasks and objective budget use the same speed.
The PAC-MAN RSL-RL wrapper uses the checkpoint runner's `clip_actions` setting
(`null` for this checkpoint). Adding a `[-1, 1]` action clip caused early
`bad_base_height` terminations and invalidated the first formal run.

`logs/classroom_comparison/` contains `routes.json`, each model's episode JSON,
`episodes.csv`, and `summary.json` with checkpoint paths and evaluation settings.
All rates are fractions of all cases for
that model (0–1):

| Metric | Definition |
| --- | --- |
| Success rate | Reach the turn junction and final target before the horizon, without a hard collision or fall. |
| On-time arrival rate | Successful arrival by `T_obj`. |
| `L_obj` | Length of the shared raw route polyline, in metres. |
| `T_obj` | `L_obj / 0.6 + 4` seconds; change the margin with `--budget-margin-s`. |
| Episode duration | Seconds until arrival, hard collision, fall, or timeout. |
| Any brush rate | Any physical robot–scenery contact during the episode (`contact.dist <= 0`). |
| Slow brush arrival rate | Arrival after `T_obj` with a brush. `--slow-brush-definition` also accepts `on-time` and `any`. |
| Timeout rate | No arrival, hard collision, or fall by the evaluation horizon. |
| Hard collision rate | Robot–scenery contact penetration at least 3 cm; the rollout ends at that point. |

The evaluator disables PAC-MAN's immediate `wall_hit` reset and LP's native
goal-complete reset so both sides use the same arrival and hard-collision rule.
Native fall termination remains active. A fall is reported separately in the
JSON/CSV summary, so all outcomes are accounted for.
