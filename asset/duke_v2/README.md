# duke_v2

The Duke Humanoid V2 robot definition: the body, the two head-camera modules, and the end
effectors, each a self-contained MuJoCo component with its own creation script and meshes.

Nothing here is a complete robot on its own. The body ships with **no camera, no hand and no
actuators**; those are grafted on at build time by
`mj_envs/asset_zoo/humanoid_v21/humanoid_v21_constants.py` (`get_humanoid_v21_robot_cfg`),
which is what training, the exporter and the deploy stack actually consume. Load a component
XML directly and you get the part, not the machine.

## Components

| Directory | What it is | Shipped MJCF |
|---|---|---|
| [`humanoid_v21/`](humanoid_v21/) | **The body.** 27 DoF: waist ×1, legs 6×2, arms 7×2. Has its own README with the regeneration recipe and gotchas. | `humanoid_v21.xml` |
| `head_cam/` | The yaw-pitch camera gimbal. Two of these on the head are what the paper's design study selected. | `head_camera_dual.xml` |
| `parallel_gripper/` | The shipped end effector, ~184 mm maximum opening, one mimic-coupled jaw slide. | `parallel_gripper.xml` |
| `cartesian_hand/` | An earlier end effector, kept as a worked example. Superseded by `parallel_gripper/`. Has its own README. | `cartesian_hand.xml` |

Adding the gimbals and the grippers to the 27-DoF body gives the **31 actuated joints** the
paper reports (27 body + 2×2 camera), plus one per gripper.

## The camera-count variants

`head_cam/` carries three rigs, and they are the asset side of the paper's camera-count
ablation (Fig. 5):

| File | Layout | Role |
|---|---|---|
| `head_camera_single.xml` | K = 1 | ablation column |
| `head_camera_dual.xml` | K = 2 | **the adopted design** |
| `head_camera_triple.xml` | K = 3 | ablation column |

The matching whole-robot exports are `humanoid_v21/rig_single/` and
`humanoid_v21/rig_triple/`; the K = 2 rig is the default export and needs no subdirectory.

## Looking at it

```bash
python asset/duke_v2/humanoid_v21/view_model.py
```

Opens the body in a MuJoCo window. `inspect_partial_model.py` beside it prints the body tree
without opening a window, which is the faster way to check whether a graft landed where you
expected.

## Regenerating

Every component is emitted by a creation script next to its XML
(`humanoid_v21_creation_v3.py`, `head_camera_creation.py`,
`parallel_gripper_creation.py`), all built on the shared builder library in
[`asset/create/`](../create/). The scripts regenerate their committed XMLs **bit-identical**,
so a non-empty `git diff` after running one means something drifted.

Per-component detail, including the export commands for the URDFs that cuRobo loads, is in
[`humanoid_v21/README.md`](humanoid_v21/README.md).
