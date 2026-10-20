# Asset Zoo

Robot descriptions used by this release.

| Robot | Model files |
|-------|-------------|
| **Unitree G1** | [`robots/unitree_g1/`](robots/unitree_g1/) |

The G1 description ships five MuJoCo models. `g1_constants.py` selects between
them; `G1_ROBOT_VERSION` (`v5` or `v15`) chooses the variant used by the LP-NavOA
paper tasks, and `v15` is the default.

| Model | Used by |
|-------|---------|
| `xmls/g1.xml` | `get_g1_robot_cfg()` |
| `xmls/g1_v5.xml` | `get_g1_v5_robot_cfg()` |
| `xmls/g1_n3_v5.xml` | `get_g1_n3_v5_robot_cfg()` |
| `xmls/g1_n3_v15.xml` | `get_g1_n3_v15_robot_cfg()` |

The meshes under `xmls/assets/` are only the ones referenced by these models.
