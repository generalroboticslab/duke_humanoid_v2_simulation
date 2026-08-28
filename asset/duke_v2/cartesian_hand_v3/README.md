# cartesian_hand_v3 — mini_gripper "Cartesian Hand V1 0818"

The current Cartesian hand, imported from the standalone `mini_gripper` grasp-suite
(`grasp_suite/gripper.xml`, "Cartesian Hand V1 0818"). Same dual-double-jaw family as the
archived `cold/cartesian_hand` (v1) and `cold/cartesian_hand_v2`, at its newest, most complete
iteration.

## What it is

A wristless four-finger gripper — **9 slide joints, no revolute DoF**, coupled by **2
`<equality><joint>` constraints** down to **7 position actuators**. Total mass 0.3116 kg.

- **Bodies (10):** `base` (root) → `bridge` (Z-lift stage); the bridge carries the *upper*
  rack+finger pair, the base carries the *lower* pair.
- **9 slide joints:** `bridge_z` (Z lift); `left_up_y`/`right_up_y`, `left_down_y`/`right_down_y`
  (jaw open/close, opposed Y); `{left,right}_{up,down}_finger_x` (per-finger X extension).
- **2 equalities:** `right_up_y ← left_up_y`, `right_down_y ← left_down_y` — each jaw pair moves
  symmetrically from one driven DoF, so each rack pair needs only one motor.
- **7 position actuators** (`kp/kv` PD, `forcerange="-87 87"`): `m_bridge_z`, `m_up_pair`,
  `m_down_pair`, `m_{left,right}_{up,down}_finger`. The ±87 N follows from a FEETECH
  HLS3915M-C001 servo (stall 1.39254 N·m, rack pitch radius 8 mm, coupled two-rack reaction).
- **Collision:** group-3 `*_col_NN` geoms — 278 CoACD convex-hull pieces (base 101, bridge 35,
  racks 31/29/34/32, fingers 4×4). `contype=1 conaffinity=0` → collides with objects/floor, no
  self-collision. Visual geoms are group-2 `*_visual` (10 meshes, non-colliding).

## Layout

| Path | Contents |
|---|---|
| `cartesian_hand_v3.xml` | the hand model (graft-ready — the source's floor/lights/headlight/statistic are stripped). `meshdir="meshes"`, self-contained. |
| `meshes/` | 10 visual OBJs (`base`, `bridge`, 4 racks, 4 fingers) |
| `meshes/collision_pieces/` | 278 CoACD convex-hull collision OBJs |
| `source/` | CAD provenance: `vertical_translation_v8.step` (34 MB Fusion assembly), its solids CSV + three-view PNG, `kinematics.json` (joints/actuators/couplings/servo spec), `groups.json` (body↔CAD occurrence map), `body_inertia.json` (per-body COM + inertia, matching the `<inertial>` blocks) |

## Provenance & regeneration

This is a **direct import** of the source `gripper.xml` (only scene decoration removed), not a
regenerated artifact — the source repo's own CAD→MJCF pipeline (`mini_gripper/common/pipeline/`)
built it from `source/vertical_translation_v8.step` via `source/kinematics.json`. That pipeline
is not vendored here; the `source/` files are kept so the model stays reproducible from CAD.

## Robot wiring — DEFERRED (standalone-first)

Loaded by `mj_envs/asset_zoo/cartesian_hand_v3.py`. The **arm-mount geometry (flange + mating
face) is not set** yet: the archived v1/v2 each carry a measured `HAND_MATING_FACE_POS_IN_BASE`
against a CAD flange datum, and this hand's datum has not been confirmed. Grafting it onto an
arm (and, if desired, migrating the five robots that still use `cold/cartesian_hand`) is a
separate step — see `../cold/README.md`.
