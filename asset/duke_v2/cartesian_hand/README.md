# Gripper module — `asset/duke_v2/gripper/`

A decoupled, swappable end-effector module, managed the same way as the head camera (`asset/duke_v2/head_cam/`). It attaches onto the humanoid's wrist-roll flanges at runtime — the robot model (`asset/duke_v2/humanoid_v21/humanoid_v21.xml`) is never edited.

## What this gripper is

A **4-finger parallel-jaw, rack-and-pinion hand** (imported from the `mini_gripper_old` project; provenance in [source/HANDOFF_REPORT.md](source/HANDOFF_REPORT.md)). One design is mounted on **both** wrists.

- **10 bodies:** `base` → `bridge` → {`left_up_rack`→`left_up_finger`, `right_up_rack`→`right_up_finger`}, and `base` → {`left_down_rack`→`left_down_finger`, `right_down_rack`→`right_down_finger`}.
- **9 `slide` (prismatic) joints — ranges in METERS, not radians:**
  | joint | axis | range (m) | actuator |
  |---|---|---|---|
  | `bridge_z` | z | −0.027 … 0.030 | `m_bridge_z` |
  | `left_up_y` / `right_up_y` | ±y | −0.020 … 0.035 | `m_up_pair` (1 actuator, both jaws) |
  | `left_down_y` / `right_down_y` | ±y | −0.040 … 0.017 | `m_down_pair` (1 actuator) |
  | 4× `*_finger_x` | x | 0 … 0.055 | `m_*_finger` (4 actuators) |
- **2 mimic equalities** (`polycoef="0 1 0 0 0"`, 1:1): `right_up_y = left_up_y`, `right_down_y = left_down_y` — each jaw pair is one symmetric open/close DOF.
- **7 `<position>` actuators** → after the per-arm attach, **14** total (`L_*` / `R_*`).

## Folder layout

```
gripper/
  cartesian_hand.xml                  # the module MJCF (single source of truth)
  meshes/
    *.obj                      # 10 visual meshes (full-res, render only)
    collision_pieces/*.obj     # 154 convex collision hulls (CoACD)
  cartesian_hand_fusion_info.py       # Fusion Properties -> <inertial> bridge (no mirror)
  PositionDeter/
    RELATIVE_POSITION_flange__gripper.md   # mount spec (re-root transform + checks)
  source/
    HANDOFF_REPORT.md          # full provenance from the source model
    base_*.png                 # mounting-face renders
  README.md
```

## How it attaches

`humanoid_v21_constants.get_spec(end_effector=...)` strips each baked gripper to a bare wrist-roll flange (keeps the arm's `*_wrist_3_joint` roll DOF + IK site), then grafts this module's `base` body onto the flange — once per arm, prefixed `L_`/`R_` (keeps the gripper's internal `left/right` jaw names unique). The gripper's joints, mimic equalities, and actuators all ride `attach_body` automatically (prefixed).

| `end_effector=` | result |
|---|---|
| `"builtin"` (default) | robot's original baked gripper, untouched |
| `"welded"` | this gripper, finger DOFs/mimics/actuators dropped (rigid mount) |
| `"actuated"` | this gripper with all DOFs — mjlab Entity: nu=41, nq=52, 33.355 kg |
| `"none"` | bare arms (no gripper) |

Mount transform (gripper `base` → flange): `pos=(0.0534, 0, -0.0325)`, `quat=identity` — see [PositionDeter/](PositionDeter/RELATIVE_POSITION_flange__gripper.md).

## Mass / inertia (Fusion bridge)

The inertials in `cartesian_hand.xml` are CAD uniform-density estimates. To use real values: read each of the 10 bodies' Properties in Fusion 360 (in the gripper's assembly frame, **no mirror**), paste into [cartesian_hand_fusion_info.py](cartesian_hand_fusion_info.py), then `python cartesian_hand_fusion_info.py --write`. It auto-converts units and writes MuJoCo `fullinertia` (order `ixx iyy izz ixy ixz iyz`). Blocks can be filled and written incrementally.

## Collision / FPS

MuJoCo narrow-phase needs convex shapes, so each link's concave geometry is a CoACD union of convex hulls (`group=3`, `contype=1 conaffinity=0` → collides with the world, self-collision off). For FPS the **structural** `base` (101→6) and `bridge` (35→6) hulls were re-decomposed coarse (they bolt to the wrist / lift — not grasp surfaces); the **rack/finger grasp hulls are kept** for grasp fidelity. Visual meshes are full-res (render only). To trade more accuracy for FPS, coarsen the rack hulls next; to lighten rendering, decimate the visuals (needs `fast_simplification`).

## Replacing the gripper (the pipeline)

1. Model/export the new gripper as a standalone MuJoCo subtree with a single root body (its wrist-mating face known); convex collision per link; joints in their native units.
2. Drop it in as `cartesian_hand.xml` + `meshes/` here.
3. Set the root body name + mount transform in `humanoid_v21_constants.py` (`GRIPPER_ROOT_BODY`, `GRIPPER_MOUNT_POS/QUAT`); update `PositionDeter/`.
4. Update `cartesian_hand_fusion_info.py`'s body list; read Fusion inertia → `--write`.
5. Verify: `--end-effector actuated` in the viewer; check mass/DOF and FPS.

## Verify

```bash
# standalone module
~/miniconda3/envs/mjhand/bin/python -c "import mujoco; mujoco.MjSpec.from_file('cartesian_hand.xml').compile()"
# on the robot (both wrists, full scenario)
~/miniconda3/envs/mjhand/bin/python mj_envs/asset_zoo/humanoid_v21/humanoid_v21_constants.py --end-effector actuated
```
