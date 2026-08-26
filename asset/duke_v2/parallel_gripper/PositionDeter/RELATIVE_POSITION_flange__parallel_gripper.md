# Mount geometry — wrist mount ↔ parallel_gripper

How the gripper seats on a host arm. Feeds the `asset_zoo` loader (`HAND_REACH_AXIS_IN_BASE`, `HAND_MATING_FACE_POS_IN_BASE`), which computes the attach pose via `Hand.mount_pose(flange_axis)`. Mirrors `../../cartesian_hand_v2/PositionDeter/RELATIVE_POSITION_flange__cartesian_hand_v2.md`.

All values are in the gripper **base-body frame** (every body declared at identity, so this is the one shared frame). Units: meters.

## Decoupled mounting flange

This gripper's mounting flange is **`cnc_flange`** — the silver disc with the bolt-hole ring (from `CNC.step`). Like `cartesian_hand_v2`, it is **decoupled**: mesh in `../flanges/`, mesh/parent/group config in `parallel_gripper.py:FLANGE_CONFIG`, and mass/COM/inertia as Fusion-text in `parallel_gripper_fusion_info.FLANGE_INFO` (no JSON). It is **composed onto `base` by the loader at load time** (`_attach_flanges`) so it can be edited/swapped independently. It is **visual-only** (no collider — the arm owns collision), welded (no joint, `pos=0`, world-coord mesh auto-aligns to `base`).

| flange | parent body | host arm | geom group | mass | −X mating face (m) |
|---|---|---|---|---|---|
| `cnc_flange` | `base` | (TBD — its native RS05 coupler interface) | 2 | 22.05 g (Al 6061) | `[-0.032, 0.006, -0.010]` |

## Reach axis

`HAND_REACH_AXIS_IN_BASE = [1, 0, 0]` — the jaws reach toward objects along base-frame **+X** (fingertips at x ≈ +0.12 m). The flange is at the **−X** end. The jaws open/close along **±Y** (`left_rack_y` axis `+Y`, `right_rack_y` axis `−Y`).

## Mating face

`HAND_MATING_FACE_POS_IN_BASE = [-0.032, 0.006, -0.010]` — the −X face center of `cnc_flange`, **mesh-measured** from `cnc_flange.obj` (bbox x∈[−0.032, −0.014], y center ≈ 0.006, z center ≈ −0.010). `mount_pose` lands this point on the host flange-site origin and rotates `reach_axis` onto the flange tool-out axis.

> Mesh-measured, not a verified CAD datum — confirm against the flange's CAD bolt/bore datum
> (and add a clocking rotation if it is keyed) before relying on it on hardware.

## Per-host mount

Two `Hand` variants (`mj_envs/asset_zoo/parallel_gripper.py`): **`PARALLEL_GRIPPER`** (default, composes `cnc_flange`) and **`PARALLEL_GRIPPER_BARE`** (no flange, for swapping a different one). `mount_pose(flange_axis)` works for any arm whose flange tool-out axis is supplied (panda/ur5e `+Z`, humanoid wrist-roll `+X`); add per-host flange variants via the `make_parallel_gripper(flanges=...)` factory when wiring onto a robot.
