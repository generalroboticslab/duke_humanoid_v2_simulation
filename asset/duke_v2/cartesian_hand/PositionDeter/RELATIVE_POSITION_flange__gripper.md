# Mount spec: wrist-roll flange ↔ gripper base

> For whoever mounts the gripper module onto the humanoid's wrist. Analogous to
> `head_cam/PositionDeter/RELATIVE_POSITION_top_plate__head_cameras.md`.
> Derivation source: `../source/HANDOFF_REPORT.md` §3 (renders: `../source/base_Xend.png`,
> `../source/base_views.png`). Confirmed in the MuJoCo viewer 2026-06-15.

---

## TL;DR

The gripper module's root body `base` bolts onto the arm's **wrist-roll output flange** — body `end_effector_L` / `end_effector_R` in `asset/duke_v2/humanoid_v21/humanoid_v21.xml`. The same one gripper design mounts on **both** wrists, attached once per arm with a per-arm name prefix (`L_` / `R_`).

| Item | Value |
|---|---|
| Host body (arm side) | `end_effector_L` / `end_effector_R` (keeps the wrist-roll joint + IK site) |
| Gripper root body | `base` |
| Mount translation `pos` (m) | **`0.0534  0  -0.0325`** |
| Mount rotation `quat` (wxyz) | **`1 0 0 0`** (identity — no rotation, no clocking) |
| Reach / wrist-roll axis | gripper **+X** (already aligned with the flange's local +X) |

Applied in code by `humanoid_v21_constants.py`: `GRIPPER_MOUNT_POS`, `GRIPPER_MOUNT_QUAT`, `EE_FLANGE_PREFIX`, attached in `_decouple_end_effector` via `flange.add_frame(pos, quat).attach_body(base, prefix)`.

---

## Why this transform

The gripper `base` body frame is **not** at its wrist-mating face: the circular mating flange sits at base-frame `(-0.0534, 0, 0.0325)` m (the `−X` end; the fingers reach `+X`). To land that face on the arm flange's origin we mount `base` at the negative of that offset:

```
flange_center (base frame) = (-0.0534, 0, 0.0325)
mount pos                  = -flange_center = (0.0534, 0, -0.0325)   ->  face lands at flange origin
```

The gripper's `+X` already points outward along the wrist-roll axis (away from the arm), and the flange's local `+X` is the same reach axis (its grasp site sits at flange-local `0.0919 0 0`), so **no rotation is needed** — `quat = 1 0 0 0`.

The two flanges are mirrored by the arm itself (the wrist-roll joints use `ref = ±1.5708`), so attaching the **same** gripper subtree at the **same** flange-local pose puts a correctly-oriented gripper on each wrist (verified in the viewer).

---

## Self-check (MuJoCo)

```python
import humanoid_v21_constants as h, mujoco, numpy as np
m = h.get_spec(end_effector="actuated").compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
for side, pref in (("L","L_"), ("R","R_")):
    fb = m.body(f"end_effector_{side}").id
    gb = m.body(f"{pref}base").id
    R  = d.xmat[fb].reshape(3, 3)
    prel = R.T @ (d.xpos[gb] - d.xpos[fb])          # gripper base in flange frame
    print(side, "pos(flange) =", prel.round(4))     # expect ~ (0.0534, 0, -0.0325)
```

Visual: `python mj_envs/asset_zoo/humanoid_v21/humanoid_v21_constants.py --end-effector actuated` → a gripper on each wrist, jaws facing forward, drag the `L_m_*` / `R_m_*` sliders to open/close.

---

## ⚠️ Confirm against the real flange before hardware (from HANDOFF_REPORT §3)

These come from the source model's mesh geometry, not a CAD datum — verify if it matters:

1. **X datum ±0.0089 m** — the `−X` end has an outer spigot boss (`x=-0.0534`, used here) and a recessed bolt plate ~9 mm behind it. If the flange seats on the recessed plate, add **+0.0089** to `pos.x` (→ `0.0445`).
2. **Flange center Z = +0.0325, Y = 0** derived from tessellation (±~0.5 mm) — confirm the bolt-circle/bore center.
3. **No clocking applied.** If the real flange has keyed/clocked bolt holes, add a roll (X-axis) rotation to the mount `quat`.
4. Bolt-circle diameter / hole pattern not extracted (only a ~6 mm bore + ~15–20 mm register were seen). Match the real pattern.
