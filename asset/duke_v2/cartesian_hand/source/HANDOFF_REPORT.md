# Gripper Handoff Package — REPORT

This module is a single self-contained MuJoCo gripper — a four-finger (two upper + two lower jaw), parallel-jaw, rack-and-pinion "mini" gripper — packaged for transplant onto a humanoid robot's wrist-roll flange. The gripper's root `base` body sits at the world origin at identity (`pos 0 0 0`, `quat 1 0 0 0`), so it is delivered as a clean subtree ready to be re-rooted under a wrist-roll output frame. The handoff folder (`<gripper source>/gripper_handoff/`) contains the standalone MJCF (`gripper.xml`, a verbatim copy of the source), the `meshes/` directory holding the 10 visual OBJs and 278 referenced collision-piece OBJs under `meshes/collision_pieces/`, and two reference renders of the base mounting end (`base_views.png`, `base_Xend.png`). Everything below was verified against a fresh MuJoCo compile of the bundled file and against real trimesh loads of every referenced mesh.

## 1. Locate & identify

- **MJCF path(s):** `gripper.xml` is the single self-contained model file with **NO `<include>` elements** (0 includes, verified on compile). The bundled copy at `<gripper source>/gripper_handoff/gripper.xml` recompiles standalone (`ngeom=289`). `meshdir="meshes"` (gripper.xml:3) resolves relative to the file, so the bundle is self-contained.
- **BASE body that bolts to wrist: `base`.** This is the single root of the gripper subtree: `<body name="base" pos="0 0 0">` at **gripper.xml:325**, the one child of `<worldbody>` (gripper.xml:317) that is the gripper itself. Its compiled frame is `body_pos=[0,0,0]`, `body_quat=[1,0,0,0]` (identity), `parentid=0` (world). It holds the wrist-mating face (a circular flange at the `−X` end — see §3) and is the body all other gripper bodies descend from. The other worldbody entries (`floor`, lights `overhead`/`fill`) are scene furniture outside the `base` subtree.
- **Single vs left+right: this is ONE gripper, not two.** The names "left"/"right" denote the **two jaws of a single gripper** within each pair. The gripper has an **upper jaw pair** (`left_up_*` + `right_up_*`) and a **lower jaw pair** (`left_down_*` + `right_down_*`), all four hanging off the one `base` root via `bridge`. The two jaws of each pair are coupled 1:1 by an equality constraint into a symmetric pinch driven by a single actuator. There is no second independent gripper.
- **Compiler settings:** `angle="radian"`, `meshdir="meshes"`, `autolimits="true"`, and every mesh uses `scale="0.001 0.001 0.001"` (source OBJs authored in **millimeters**, scaled to meters at load). Cited at **gripper.xml:3** (compiler block) and the mesh asset declarations from gripper.xml:13 onward. Because `angle="radian"`, no degree→radian conversion is ever needed; and because every joint is a `slide` (prismatic), all `range`/`ctrl`/`qpos` are linear displacements in **meters**, not angles.
- **Scene furniture is NOT part of the module.** The `floor` plane (gripper.xml:318), the two lights `overhead`/`fill` (gripper.xml:320–322), and the floor finish (`texture grid_tex` gripper.xml:310, `material grid` gripper.xml:313) are carried along in `worldbody` but are not referenced by any gripper part. They are safe to drop when bolting the module onto a wrist.

## 2. Compile & introspect

### 2.1 Body tree & inertials

All data verified — MJCF `<inertial>` values match the compiled `model.body_*` arrays to the printed precision. ngeom = 289 (compiled) and 289 `<geom>` tags; the totals line below reports ngeom as the compiled value.

**Totals (from compiled model, `mujoco.MjModel.from_xml_path("<gripper source>/gripper.xml")`):** `nbody = 11` (1 world + 10 gripper bodies) · `nq = 9` · `nv = 9` · `nu = 7` · `<equality> constraints (neq) = 2` · `ngeom = 289` · **TOTAL body_mass (gripper subtree, excludes world) = 0.311601 kg** (`sum(model.body_mass[1:])`; world mass = 0).

Notes that hold for **every** gripper body: in the MJCF each `<body>` is declared with `pos="0 0 0"` and no `quat` (so `body_quat` defaults to identity `1 0 0 0`). All articulation/geometry offset lives in the per-piece mesh/collision geoms and the `<inertial>` `pos`/`quat`, not in the body frame. The compiler is `angle="radian"`, so no degree→radian conversion is needed; all quaternions below are dimensionless wxyz and all `pos`/COM are in meters. Inertia is given as `diaginertia` (principal moments, kg·m²) + `iquat` (orientation of the principal axes wrt the body frame); no body uses `fullinertia`.

| body | parent | pos (rel parent) | quat wxyz (rel parent) | mass (kg) | inertial COM (local, m) | diaginertia (kg·m²) | iquat (wxyz) | def line |
|---|---|---|---|---|---|---|---|---|
| base | world | 0 0 0 | 1 0 0 0 | 0.0695138 | -0.0235692 0.00865956 0.0152749 | 4.23609e-05 1.36314e-04 1.58393e-04 | 0.922085 0.143966 0.328690 -0.144899 | gripper.xml:325 |
| bridge | base | 0 0 0 | 1 0 0 0 | 0.0811808 | -0.0152106 -0.00949439 0.0469537 | 5.69510e-05 2.23313e-04 2.56792e-04 | 0.589177 -0.0362555 -0.781758 -0.201024 | gripper.xml:433 |
| left_up_rack | bridge | 0 0 0 | 1 0 0 0 | 0.0369950 | 0.087428 -0.0550683 0.0520866 | 6.96676e-06 5.01376e-04 5.03989e-04 | 0.935380 -0.0570725 -0.223156 -0.268344 | gripper.xml:476 |
| left_up_finger | left_up_rack | 0 0 0 | 1 0 0 0 | 0.00323154 | 0.0725586 -0.0390056 0.0509872 | 9.01730e-07 3.14832e-05 3.22223e-05 | 0.826112 -0.448378 -0.140057 -0.311256 | gripper.xml:515 |
| right_up_rack | bridge | 0 0 0 | 1 0 0 0 | 0.0369950 | 0.087428 0.0550683 0.052101 | 6.96911e-06 5.01432e-04 5.04041e-04 | 0.935102 0.0612350 -0.222013 0.269340 | gripper.xml:529 |
| right_up_finger | right_up_rack | 0 0 0 | 1 0 0 0 | 0.00323154 | 0.0725586 0.0390056 0.0532003 | 9.10155e-07 3.21863e-05 3.30009e-05 | 0.823289 0.448293 -0.148770 0.314794 | gripper.xml:566 |
| left_down_rack | base | 0 0 0 | 1 0 0 0 | 0.0369950 | 0.087428 -0.0361588 -7.1681e-06 | 6.40708e-06 3.37181e-04 3.40371e-04 | 0.980837 0.00193905 -0.000352174 -0.194821 | gripper.xml:581 |
| left_down_finger | left_down_rack | 0 0 0 | 1 0 0 0 | 0.00323154 | 0.0725586 -0.0200962 -0.00110656 | 2.92597e-07 2.01226e-05 2.01748e-05 | -0.498583 0.858007 -0.110055 0.0559259 | gripper.xml:623 |
| right_down_rack | base | 0 0 0 | 1 0 0 0 | 0.0369950 | 0.087428 0.0361588 7.1681e-06 | 6.40708e-06 3.37181e-04 3.40371e-04 | 0.980837 0.00193905 0.000352174 0.194821 | gripper.xml:637 |
| right_down_finger | right_down_rack | 0 0 0 | 1 0 0 0 | 0.00323154 | 0.0725586 0.0200962 0.00110656 | 2.92597e-07 2.01226e-05 2.01748e-05 | -0.498583 0.858007 0.110055 -0.0559259 | gripper.xml:677 |

Tree shape (4-finger gripper): `base → bridge → {left_up_rack → left_up_finger, right_up_rack → right_up_finger}` and `base → {left_down_rack → left_down_finger, right_down_rack → right_down_finger}`. The two upper racks hang off `bridge` (the z-sliding stage); the two lower racks hang directly off `base`. Each `*_rack` carries one `*_y` slide joint and each `*_finger` carries one `*_finger_x` slide joint; `bridge` carries `bridge_z` — 9 slide joints total (`nq=nv=9`).

#### Inertial provenance

`grep -c "<inertial"` = **10**, exactly one per non-world body — i.e. **every** gripper body has an explicit `<inertial>` tag (not just `base`). None rely on MuJoCo's geom-derived auto-inertia. Every tag specifies a non-trivial COM `pos`, a non-axis-aligned principal-axis `quat`, and three distinct principal moments (`Ixx ≠ Iyy ≠ Izz`), which is the signature of values computed from the actual CAD mesh at uniform density rather than placeholder primitives.

- **base** (gripper.xml:326-329): REAL/CAD-like. COM offset `(-0.0236, 0.00866, 0.0153)` is plausibly off-center; `iquat` has all four components non-zero (heavily tilted principal frame); three distinct moments. Consistent with a meshed CAD inertia.
- **bridge** (gripper.xml:434-437): REAL/CAD-like. Same profile — offset COM, strongly rotated `iquat`, distinct moments. The heaviest single part (0.0812 kg).
- **left_up_rack / right_up_rack** (gripper.xml:477-480 / 530-533): REAL/CAD-like and mirror-consistent. Identical mass (0.036995), near-mirrored COM (`y = -0.0551` vs `+0.0551`) and near-mirrored `iquat`. Tiny in-plane asymmetries (e.g. z-COM 0.0520866 vs 0.052101; moments differ in the 4th–5th sig fig) indicate independently meshed left/right parts, not one part copied — consistent with genuine per-mesh computation.
- **left_up_finger / right_up_finger** (gripper.xml:516-519 / 567-570): REAL/CAD-like, mirror pair, mass 0.00323154. Large finger-frame tilt in `iquat`; distinct sub-1e-6 / 3e-5 moments typical of a small thin part.
- **left_down_rack / right_down_rack** (gripper.xml:582-585 / 638-641): REAL/CAD-like, and an *exact* mirror pair — same mass and same `diaginertia` (6.40708e-06, 3.37181e-04, 3.40371e-04), with COM y and the `iquat` y/w-sign flipped (`±0.0361588`, `±0.000352174 / ±0.194821`). These two are a clean mirror of one geometry (down racks differ from up racks: shorter z-extent, different moments), which is physically correct for a symmetric mechanism.
- **left_down_finger / right_down_finger** (gripper.xml:624-627 / 678-681): REAL/CAD-like, exact mirror pair, mass 0.00323154, identical `diaginertia` (2.92597e-07, 2.01226e-05, 2.01748e-05) with mirrored COM/`iquat`. Note the leading `iquat` component is negative (`-0.498583`) — valid (q and -q are the same rotation), just MuJoCo's chosen sign.

Verification: the compiled `model.body_mass`, `body_ipos`, `body_iquat`, and `body_inertia` reproduce the MJCF `<inertial>` mass/pos/quat/diaginertia values to the printed precision for all 10 bodies, so the documented numbers are the as-simulated values, not just the source text.

### 2.2 Joints & equality constraints

All verified against a fresh compile. Joint `pos` is unset in XML (defaults to `0 0 0`, confirmed by compile). The two equality couplings drive only the upper and lower `_y` racks; the four `_finger_x` slides are independently actuated. Note: 7 actuators but the equality means only 1 of each `_y` pair is actuated (`left_up_y`, `left_down_y`), with the right side mirrored by the constraint.

Compiled clean from a single self-contained MJCF (`<gripper source>/gripper.xml`, `gripper.xml:3` → `compiler angle="radian" meshdir="meshes" autolimits="true"`). Fresh `mujoco.MjModel.from_xml_path` reports **`njnt = 9`, `neq = 2`, `nu = 7`** — matching the established ground truth.

**All 9 joints are `type="slide"` (prismatic / linear).** Their motion is a translation along an axis, so the unit of `range` is **METERS**, not an angle. "Convert to radians" is therefore **N/A** — slide ranges are linear displacements in meters and there is nothing to convert. This is a parallel-jaw, rack-and-pinion design: each `_y` slide is a rack driven by a pinion, and the equality constraints below tie the left/right racks of a pair together so a single pinion drives a symmetric pinch.

#### Joints

`pos` is not authored on any joint element (it is omitted in the XML); the compiler defaults it to `0 0 0`, which the compile confirms for all 9. `damping`/`armature` are read from `dof_damping`/`dof_armature` after compile and equal the authored attribute values.

| Joint | Type | Parent body | Axis | pos (default) | Range (authored) | Range — units | Range in RADIANS | Damping | Armature | Line |
|---|---|---|---|---|---|---|---|---|---|---|
| `bridge_z` | slide (prismatic) | `bridge` | `0 0 1` | `0 0 0` | `-0.027 0.03` | METERS | N/A (linear, not angular) | 40 | 0.05 | gripper.xml:438 |
| `left_up_y` | slide (prismatic) | `left_up_rack` | `0 1 0` | `0 0 0` | `-0.02 0.035` | METERS | N/A (linear, not angular) | 12 | 0.02 | gripper.xml:481 |
| `left_up_finger_x` | slide (prismatic) | `left_up_finger` | `1 0 0` | `0 0 0` | `0 0.055` | METERS | N/A (linear, not angular) | 2.5 | 0.01 | gripper.xml:520 |
| `right_up_y` | slide (prismatic) | `right_up_rack` | `0 -1 0` | `0 0 0` | `-0.02 0.035` | METERS | N/A (linear, not angular) | 12 | 0.02 | gripper.xml:534 |
| `right_up_finger_x` | slide (prismatic) | `right_up_finger` | `1 0 0` | `0 0 0` | `0 0.055` | METERS | N/A (linear, not angular) | 2.5 | 0.01 | gripper.xml:571 |
| `left_down_y` | slide (prismatic) | `left_down_rack` | `0 1 0` | `0 0 0` | `-0.04 0.017` | METERS | N/A (linear, not angular) | 12 | 0.02 | gripper.xml:586 |
| `left_down_finger_x` | slide (prismatic) | `left_down_finger` | `1 0 0` | `0 0 0` | `0 0.055` | METERS | N/A (linear, not angular) | 2.5 | 0.01 | gripper.xml:628 |
| `right_down_y` | slide (prismatic) | `right_down_rack` | `0 -1 0` | `0 0 0` | `-0.04 0.017` | METERS | N/A (linear, not angular) | 12 | 0.02 | gripper.xml:642 |
| `right_down_finger_x` | slide (prismatic) | `right_down_finger` | `1 0 0` | `0 0 0` | `0 0.055` | METERS | N/A (linear, not angular) | 2.5 | 0.01 | gripper.xml:682 |

Note the deliberate axis sign flip: the right racks use `0 -1 0` while the left racks use `0 1 0` (gripper.xml:481/534 and 586/642). Because the two racks of a pair point in *opposite* world-Y directions, a positive value on both joints drives the jaws *together* (a closing pinch) — this is what makes the 1:1 equality below produce a mirrored, symmetric motion rather than both jaws sliding the same way.

#### Equality

Both are `type="joint"` couplings (verified: `eq_type = joint`, `eq_active0 = 1` for both). `eq_data` (the compiled polynomial coefficient vector) is `[0 1 0 0 0 ...]`, i.e. authored `polycoef="0 1 0 0 0"`. `solimp` is authored as 3 values `0.95 0.99 0.001`; the compiler fills the remaining defaults, yielding `solimp = [0.95 0.99 0.001 0.5 2.0]`.

| Type | joint1 (constrained) | joint2 (reference) | polycoef (authored / eq_data) | Coupling ratio & physical meaning | solref | solimp | Line |
|---|---|---|---|---|---|---|---|
| joint | `right_up_y` | `left_up_y` | `0 1 0 0 0` / `[0 1 0 0 0 ...]` | `right_up_y = 0 + 1.0·left_up_y` → **1:1 mimic**. The constant term is 0 (no offset) and the linear term is 1.0, so the two upper racks share one DOF. With the opposing axes (`+Y` left, `−Y` right), the 1:1 link mirrors the two jaws of the **upper pair** into a symmetric pinch driven by one actuator. | `0.005 1` | `0.95 0.99 0.001 0.5 2.0` | gripper.xml:696 |
| joint | `right_down_y` | `left_down_y` | `0 1 0 0 0` / `[0 1 0 0 0 ...]` | `right_down_y = 0 + 1.0·left_down_y` → **1:1 mimic** of the **lower pair**, same mechanism: one actuator on the left rack, the right rack mirrored to give a symmetric lower-jaw pinch. | `0.005 1` | `0.95 0.99 0.001 0.5 2.0` | gripper.xml:697 (block: gripper.xml:695-698) |

`polycoef` semantics: it is the polynomial `joint1 = c0 + c1·j2 + c2·j2² + c3·j2³ + c4·j2⁴`. With `c0=0, c1=1` and higher terms 0, the relationship is exactly linear 1:1 with zero offset — a pure mimic. solref `0.005 1` = time-constant 5 ms, damping-ratio 1 (critically damped, stiff constraint). solimp `0.95 0.99 …` = high min/max impedance, so the coupling holds essentially rigidly.

#### Kinematic summary (DOF structure)

Body chain (single root `base` at gripper.xml:325, the transplantable module):

- **`base` → `bridge`** (gripper.xml:433): `bridge_z` (gripper.xml:438) translates the whole jaw assembly along **+Z** (vertical), range `-0.027 … 0.03 m`. This is the bridge / overall raise-lower DOF; actuated by `m_bridge_z` (gripper.xml:702).
- **Upper pair, open/close along Y**:
  - `bridge → left_up_rack` with `left_up_y` (axis `+Y`, gripper.xml:481), range `-0.02 … 0.035 m`.
  - `bridge → right_up_rack` with `right_up_y` (axis `−Y`, gripper.xml:534), same range.
  - Coupled 1:1 by the equality at gripper.xml:696. A single actuator `m_up_pair` drives `left_up_y` (gripper.xml:703); the constraint slaves `right_up_y` to it, producing a symmetric upper-jaw pinch.
- **Lower pair, open/close along Y**:
  - `base → left_down_rack` with `left_down_y` (axis `+Y`, gripper.xml:586), range `-0.04 … 0.017 m`.
  - `base → right_down_rack` with `right_down_y` (axis `−Y`, gripper.xml:642), same range.
  - Coupled 1:1 by the equality at gripper.xml:697. Single actuator `m_down_pair` drives `left_down_y` (gripper.xml:704); `right_down_y` is slaved → symmetric lower-jaw pinch. (Note the lower pair's `_y` joints hang off `base`, not `bridge` — so the lower jaws do **not** ride the `bridge_z` lift, unlike the upper pair.)
- **Per-finger X slides (4, independent)**: `left_up_finger_x` (gripper.xml:520), `right_up_finger_x` (gripper.xml:571), `left_down_finger_x` (gripper.xml:628), `right_down_finger_x` (gripper.xml:682) — each a child of its rack body, translating its finger along **+X**, range `0 … 0.055 m`. These are **not** in any equality; each is actuated separately (`m_left_up_finger`, `m_right_up_finger`, `m_left_down_finger`, `m_right_down_finger`, gripper.xml:705–708), giving 4 independent finger-extension DOFs.

DOF accounting: 9 slide joints → 9 model DOFs, reduced by 2 equality constraints to **7 independent DOFs**, exactly matched by the **7 `<position>` actuators** (gripper.xml:702–708). Mapping: `bridge_z`←1 actuator; each `_y` pair = 2 joints driven by 1 actuator (the equality couples left↔right so one actuator drives the symmetric pinch); each of the 4 `_finger_x` = its own actuator. The two right-side `_y` joints (`right_up_y`, `right_down_y`) are **not** directly actuated — they follow their left partners through the equality constraints.

### 2.3 Actuators

All ground truth verified against a fresh compile (7 actuators, 9 slide joints, 2 equality couplings, 0 keyframes). Note: the `right_up_y` / `right_down_y` jaw joints use `axis="0 -1 0"` (mirrored), which is how a single positive `y` command on the left joint, mirrored by the `polycoef="0 1 0 0 0"` equality, produces a symmetric squeeze.

The model declares exactly **7 actuators**, all of MJCF type **`<position>`** (a `general` actuator preset: `gaintype=fixed`, `biastype=affine`), confirmed by compile (`m.nu == 7`; every `actuator_gaintype==0`, `actuator_biastype==1`). All are defined in the `<actuator>` block at **gripper.xml:700-709**. There is **no `<motor>` and no `<general>`** actuator. Each drives a single **slide** joint directly (`actuator_trntype == 0`, i.e. transmission = joint). A `<position>` actuator implements a PD law: applied force `= kp*(ctrl - qpos) - kv*qvel`, so `ctrl` is a **commanded target position in metres** along that joint's slide axis. Confirmed at compile: `gainprm[0] == kp`, `biasprm[1] == -kp`, `biasprm[2] == -kv`.

| # | Actuator | Type | Target joint | kp | kv | ctrlrange (m) | forcerange (N) | Physical motion it drives |
|---|----------|------|--------------|----|----|---------------|----------------|----------------------------|
| 0 | `m_bridge_z` | position | `bridge_z` (slide, axis `0 0 1`, gripper.xml:438) | 3000 | 15 | -0.027 … 0.03 | -87 … 87 | Vertical **Z travel of the entire finger assembly**: raises/lowers the bridge (and everything mounted on it) along the tool Z axis. Target = commanded bridge height. |
| 1 | `m_up_pair` | position | `left_up_y` (slide, axis `0 1 0`, gripper.xml:481) | 800 | 8 | -0.02 … 0.035 | -87 … 87 | **Open/close of the UPPER jaw pair.** Commands the left-upper jaw's Y slide; equality eq[0] (gripper.xml:696) slaves `right_up_y` to it 1:1. Because `right_up_y` axis is mirrored (`0 -1 0`, gripper.xml:534), one command moves both upper jaws symmetrically together/apart. |
| 2 | `m_down_pair` | position | `left_down_y` (slide, axis `0 1 0`, gripper.xml:586) | 800 | 8 | -0.04 … 0.017 | -87 … 87 | **Open/close of the LOWER jaw pair.** Commands left-lower jaw Y slide; equality eq[1] (gripper.xml:697) slaves `right_down_y` (mirrored axis `0 -1 0`, gripper.xml:642) 1:1 → symmetric lower-pair grip. |
| 3 | `m_left_up_finger` | position | `left_up_finger_x` (slide, axis `1 0 0`, gripper.xml:520) | 150 | 1 | 0 … 0.055 | -87 … 87 | **Individual extension of the left-upper finger along +X** (finger reaches forward/retracts on its rack). |
| 4 | `m_right_up_finger` | position | `right_up_finger_x` (slide, axis `1 0 0`, gripper.xml:571) | 150 | 1 | 0 … 0.055 | -87 … 87 | **Individual extension of the right-upper finger along +X.** |
| 5 | `m_left_down_finger` | position | `left_down_finger_x` (slide, axis `1 0 0`, gripper.xml:628) | 150 | 1 | 0 … 0.055 | -87 … 87 | **Individual extension of the left-lower finger along +X.** |
| 6 | `m_right_down_finger` | position | `right_down_finger_x` (slide, axis `1 0 0`, gripper.xml:682) | 150 | 1 | 0 … 0.055 | -87 … 87 | **Individual extension of the right-lower finger along +X.** |

All angles in this model are in **radians** (compiler `angle="radian"`), but note that **every actuated joint here is a `slide` joint, so all `ctrl`/`ctrlrange`/`qpos` values are linear displacements in metres** — no angular conversion applies.

#### Actuator groups and their motions

- **Bridge (Z lift) — `m_bridge_z`:** the highest-stiffness actuator (`kp=3000`, `kv=15`) because it carries the full hanging mass of the finger assembly. `ctrl` sets the bridge's vertical position; range `[-0.027, 0.03]` m (≈ 57 mm of travel) about a neutral at `qpos=0`.
- **Pair actuators — `m_up_pair`, `m_down_pair` (coupled symmetric grip):** each pair actuator drives **one jaw's Y slide only** (the *left* member). A `<joint>` equality (gripper.xml:695-698, `polycoef="0 1 0 0 0"` ⇒ `right = 0 + 1·left`) mirrors the right member to it. The right Y joints are defined with the **opposite axis sign** (`axis="0 -1 0"`, gripper.xml:534 & 642), so equal joint coordinates translate the two jaws **toward or away from each other** — i.e. a single scalar `ctrl` produces a **symmetric open/close** of that pair. Upper and lower pairs are independently commandable. Note the **asymmetric ranges**: upper pair `[-0.02, 0.035]` vs lower pair `[-0.04, 0.017]` (the two pairs close from different sides).
- **Four finger_x actuators (individual extension / "twist"):** `m_left_up_finger`, `m_right_up_finger`, `m_left_down_finger`, `m_right_down_finger` each independently extend one finger along **+X** (`axis="1 0 0"`), range `[0, 0.055]` m. Because all four are individually addressable, commanding the two fingers of a pair to **different** extensions produces the **antisymmetric "twist"** motion exploited elsewhere in the project (e.g. the screw-cap decap / unscrew demos) — one finger leads, the other trails, rotating the grasped part. Lowest stiffness (`kp=150`, `kv=1`), consistent with fine, compliant finger placement rather than load-bearing.

#### Force derivation comment (FEETECH HLS3915M-C001)

Verbatim from **gripper.xml:701**:

> FEETECH HLS3915M-C001 (a.k.a. HL-3915-C001) @ 12V nominal: stall torque 14.2 kg.cm = 1.39254 N.m. We actuate SLIDE joints directly, so forcerange is the LINEAR rack-and-pinion force: module=1, pitch diameter 16mm -> pinion pitch radius r = 8mm. The parallel jaws are a COUPLED motion (one pinion reacts against two racks), so per-finger force = stall_torque/(2*r) = 1.39254/(2*0.008) = 87 N (stall/peak; continuous load is lower). No-load speed is a runtime slew limit, not an MJCF field.

**Interpretation.** The real servo is a rotary actuator (stall torque 14.2 kg·cm = 1.39254 N·m at 12 V). In the model the joints are *linear* slides, so the rotary torque is converted to a linear rack force through the pinion pitch radius `r = 8 mm` (module 1, pitch dia 16 mm). The factor of 2 in `τ/(2r)` reflects that one pinion reacts against **two** racks in the coupled jaw mechanism, splitting the available force between the two members ⇒ **≈87 N per finger** at stall/peak. This single 87 N value is applied as `forcerange="-87 87"` to **all seven** actuators (a uniform peak-force clamp; the comment explicitly notes continuous load is lower and that no-load speed is a runtime slew limit, not an MJCF parameter). The `±` sign means the clamp is symmetric in both push and pull directions of each slide.

#### Neutral pose (ctrl = 0)

There is **no `<keyframe>`** in the file (compile: `m.nkey == 0`; `grep -c keyframe` → 0). Therefore the rest pose is `qpos == qpos0 == [0,0,0,0,0,0,0,0,0]` (all 9 slides at 0; confirmed by compile). With every actuator a `<position>` servo, **`ctrl = 0` commands each joint to `qpos = 0`**, so the model is in static equilibrium at `ctrl = 0` for all 7 actuators (commanded target equals rest position; PD error = 0).

Is `ctrl = 0` inside each ctrlrange?

| Actuator | ctrlrange | ctrl=0 in range? |
|----------|-----------|------------------|
| `m_bridge_z` | -0.027 … 0.03 | Yes |
| `m_up_pair` | -0.02 … 0.035 | Yes |
| `m_down_pair` | -0.04 … 0.017 | Yes |
| `m_left_up_finger` | 0 … 0.055 | **Boundary** — 0 is the lower limit (in range, but at the edge) |
| `m_right_up_finger` | 0 … 0.055 | **Boundary** — 0 at lower limit |
| `m_left_down_finger` | 0 … 0.055 | **Boundary** — 0 at lower limit |
| `m_right_down_finger` | 0 … 0.055 | **Boundary** — 0 at lower limit |

All seven ranges **include 0**. The three jaw/bridge actuators have 0 strictly interior; the four finger_x actuators have 0 exactly at their **lower bound** (`0 0.055`), i.e. the neutral pose is "fingers fully retracted" — fingers can only extend (positive X), never retract past neutral. No actuator excludes 0.

### 2.4 Collision geometry

All numbers below come from a fresh compile of `gripper.xml` with MuJoCo (`mujoco.MjModel.from_xml_path(...)`) and from loading every referenced collision `.obj` with `trimesh.load(path, force='mesh').faces.shape[0]`. The model compiles clean: `nbody=11`, `ngeom=289`, `njnt=9`, `nmesh=288`. The compiler is `angle="radian"` (no angular collision attributes exist here, but all angles in the file are radians), `meshdir="meshes"`, all meshes `scale="0.001 0.001 0.001"` (mm to m). There are **no `<default>` classes** (`grep` finds none) — every collision geom spells out its attributes inline.

Of the 289 geoms: 1 is the scene `floor` plane (not part of the module), 10 are top-level visual meshes (`group=2`, `contype=0`, `conaffinity=0`), and **278 are collision mesh pieces** (`group=3`) distributed across the 10 module bodies.

#### Per-body collision summary

Every collision piece in the module shares one attribute signature (verified per-geom across all 278): `type=mesh`, `contype=1`, `conaffinity=0`, `group=3`, `condim=3`, `friction="0.5 0.005 0.0001"`, `solref="0.005 1"` (representative block: `gripper.xml:332`). The CoACD convex pieces are what carry contact; the single `group=2` visual mesh per body has `contype=0 conaffinity=0` and never collides (e.g. `gripper.xml:330-331`).

| body | #collision geoms | geom type | contype | conaffinity | group | condim | friction | solref |
|---|---|---|---|---|---|---|---|---|
| base | 101 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| bridge | 35 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| left_up_rack | 31 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| left_up_finger | 4 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| right_up_rack | 29 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| right_up_finger | 4 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| left_down_rack | 34 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| left_down_finger | 4 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| right_down_rack | 32 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| right_down_finger | 4 | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |
| **TOTAL (module)** | **278** | mesh (convex) | 1 | 0 | 3 | 3 | 0.5 0.005 0.0001 | 0.005 1 |

For reference (not part of the module): the scene `floor` plane (`gripper.xml:318-319`) compiles to `contype=1 conaffinity=1 condim=3 friction="1 0.005 0.0001" solref="0.02 1"` — these are MuJoCo defaults; the floor sets neither contype nor conaffinity in the XML.

#### Per-body convex-piece count and triangle (face) totals

Faces counted by loading each referenced `collision_pieces/*.obj` with trimesh and summing across the body's pieces. All 278 files resolved on disk (0 missing).

| body | #pieces | total faces | min / max / mean per piece |
|---|---|---|---|
| base | 101 | 13,784 | 8 / 364 / 136.5 |
| bridge | 35 | 5,462 | 14 / 404 / 156.1 |
| left_up_rack | 31 | 2,996 | 12 / 330 / 96.6 |
| left_up_finger | 4 | 604 | 88 / 174 / 151.0 |
| right_up_rack | 29 | 2,906 | 26 / 346 / 100.2 |
| right_up_finger | 4 | 582 | 94 / 182 / 145.5 |
| left_down_rack | 34 | 3,020 | 12 / 334 / 88.8 |
| left_down_finger | 4 | 600 | 88 / 174 / 150.0 |
| right_down_rack | 32 | 3,162 | 12 / 334 / 98.8 |
| right_down_finger | 4 | 584 | 94 / 182 / 146.0 |
| **GRAND TOTAL** | **278** | **33,700** | 8 / 404 / 121.2 (median 108) |

#### Strategy

Each rigid link is represented for visualization by a **single high-poly mesh** (`group=2`, ~10,000 faces for base/bridge/racks, ~12,972 for the fingers) and for **contact** by a **CoACD (Approximate Convex Decomposition) of that link into many convex hull pieces** (`group=3`). MuJoCo's narrow-phase contact requires convex shapes, so the concave true geometry is approximated as a union of convex hulls — this is why the base alone needs 101 pieces and the whole gripper 278. Each piece is small (mean 121 faces, median 108), keeping each individual convex hull cheap while the union reproduces the concave envelope.

**contype/conaffinity semantics for these pieces (`contype=1`, `conaffinity=0`):** Two geoms A and B are eligible to collide only if `(contype_A & conaffinity_B) != 0` OR `(contype_B & conaffinity_A) != 0`.
- All 278 module pieces have `contype=1, conaffinity=0`.
- Two **module pieces** A and B: `(1 & 0)=0` both directions, so they **never collide with each other** — i.e. self-collision among the gripper's own collision geoms is fully disabled by construction. This is intentional: it avoids spurious self-contacts between the many decomposed hulls and across the kinematic chain, at zero cost to the collision filter.
- A module piece vs. an **external object / the environment** that exposes `conaffinity=1` (the convention for objects and the floor, which is `conaffinity=1`): `(contype_piece=1 & conaffinity_env=1)=1`, so the piece **does collide with the environment and with manipulated objects**. The gripper is "collidable-against" the world while being self-collision-free.
- Net effect: the gripper bits are active colliders against anything with `conaffinity` bit 1 set (floor, parts, objects placed in the scene), and inert against each other.

`condim=3` (tangential friction in the contact plane, no torsional/rolling friction) with `friction=0.5` tangential is used uniformly — appropriate for grasping/sliding contact without modeling spin-resistance. `solref="0.005 1"` (5 ms time-constant, damping ratio 1) makes the gripper contacts stiffer/faster than the default floor (`0.02 1`), reducing penetration during pinch grasps.

### 2.5 Visual geometry & assets

All geometry comes from a single self-contained MJCF, `<gripper source>/gripper.xml` (no `<include>`). Compiled cleanly with MuJoCo (`mujoco.MjModel.from_xml_path(...)`): `nmesh=288`, `nmat=11`, `ntex=1`, `ngeom=289`, `njnt=9`, `nu=7`, `neq=2`. The compiler block declares `angle="radian"`, `meshdir="meshes"`, `autolimits="true"`; every mesh uses `scale="0.001 0.001 0.001"`, i.e. the source OBJ files are authored in **millimeters** and scaled to meters at load. (No `<geom>` in this model carries an explicit angle attribute; all rotations are quaternions, so the radian setting is moot for the geoms but stated for completeness.)

#### Visual geoms (group=2)

There are exactly **10** visual geoms, all `type="mesh"`, all with `contype="0" conaffinity="0" group="2"` (render-only, no collision). Each carries a per-part material but **no `rgba` on the geom itself** — color is inherited from the named material (rgba listed below is the material's). Face counts are from `trimesh.load(..., process=False)` on the referenced OBJ.

| Body | Geom name | Mesh file | Faces | Material | rgba (from material) | gripper.xml |
|------|-----------|-----------|------:|----------|----------------------|-------------|
| base | `base_visual` | base.obj | 10000 | base_mat | 0.85 0.40 0.40 1 | :330 |
| bridge | `bridge_visual` | bridge.obj | 10000 | bridge_mat | 0.40 0.75 0.40 1 | :439 |
| left_up_rack | `left_up_rack_visual` | left_up_rack.obj | 10000 | left_up_rack_mat | 0.65 0.50 0.10 1 | :482 |
| left_up_finger | `left_up_finger_visual` | left_up_finger.obj | 12972 | left_up_finger_mat | 0.95 0.85 0.30 1 | :521 |
| right_up_rack | `right_up_rack_visual` | right_up_rack.obj | 10000 | right_up_rack_mat | 0.20 0.55 0.55 1 | :535 |
| right_up_finger | `right_up_finger_visual` | right_up_finger.obj | 12972 | right_up_finger_mat | 0.50 0.90 0.90 1 | :572 |
| left_down_rack | `left_down_rack_visual` | left_down_rack.obj | 10000 | left_down_rack_mat | 0.20 0.40 0.65 1 | :587 |
| left_down_finger | `left_down_finger_visual` | left_down_finger.obj | 12972 | left_down_finger_mat | 0.45 0.75 0.95 1 | :629 |
| right_down_rack | `right_down_rack_visual` | right_down_rack.obj | 10000 | right_down_rack_mat | 0.40 0.20 0.55 1 | :643 |
| right_down_finger | `right_down_finger_visual` | right_down_finger.obj | 12972 | right_down_finger_mat | 0.80 0.50 0.95 1 | :683 |

Total visual triangle budget: **111,888 faces** across the 10 render meshes. All 10 are **MODULE** assets.

#### Assets inventory

##### (a) Meshes — 288 total = 10 visual + 278 collision

All 288 `<mesh>` elements use `scale="0.001 0.001 0.001"`. Disk check: every referenced file resolves under `<gripper source>/meshes/`. (Note: the directory holds far more OBJs than are referenced — 34 top-level and 823 in `collision_pieces/` — but only the 288 listed below are wired into the model. All are MODULE assets.)

**10 visual meshes** (file · scale · faces), mesh-def line cited:

| Mesh name | File | Scale | Faces | gripper.xml |
|-----------|------|-------|------:|-------------|
| base_mesh | base.obj | 0.001³ | 10000 | :12 |
| bridge_mesh | bridge.obj | 0.001³ | 10000 | :114 |
| left_down_finger_mesh | left_down_finger.obj | 0.001³ | 12972 | :150 |
| left_down_rack_mesh | left_down_rack.obj | 0.001³ | 10000 | :155 |
| left_up_finger_mesh | left_up_finger.obj | 0.001³ | 12972 | :190 |
| left_up_rack_mesh | left_up_rack.obj | 0.001³ | 10000 | :195 |
| right_down_finger_mesh | right_down_finger.obj | 0.001³ | 12972 | :227 |
| right_down_rack_mesh | right_down_rack.obj | 0.001³ | 10000 | :232 |
| right_up_finger_mesh | right_up_finger.obj | 0.001³ | 12972 | :265 |
| right_up_rack_mesh | right_up_rack.obj | 0.001³ | 10000 | :270 |

**278 collision meshes** — convex-decomposition pieces under `collision_pieces/`, all `scale="0.001 0.001 0.001"`, named `<body>_NN.obj` (mesh names `<body>_col_NN_mesh`). The base-mesh defs begin at gripper.xml:13; the last collision def (`right_up_rack_col_28_mesh`) is at gripper.xml:299. Summary by body (collision pieces are low-poly hulls — far smaller than the visual meshes):

| Body | Pieces | Total faces | Faces/piece (min–max, mean) |
|------|------:|------------:|-----------------------------|
| base | 101 | 13784 | 8–364, ~136 |
| bridge | 35 | 5462 | 14–404, ~156 |
| left_up_rack | 31 | 2996 | 12–330, ~96 |
| left_up_finger | 4 | 604 | 88–174, ~151 |
| right_up_rack | 29 | 2906 | 26–346, ~100 |
| right_up_finger | 4 | 582 | 94–182, ~145 |
| left_down_rack | 34 | 3020 | 12–334, ~88 |
| left_down_finger | 4 | 600 | 88–174, ~150 |
| right_down_rack | 32 | 3162 | 12–334, ~98 |
| right_down_finger | 4 | 584 | 94–182, ~146 |
| **Total** | **278** | **33,700** | — |

##### (b) Materials — 11 total (10 MODULE + 1 SCENE FURNITURE)

10 gripper-part materials (color only, no texture), gripper.xml:300–309:

| Material | rgba | gripper.xml |
|----------|------|-------------|
| base_mat | 0.85 0.40 0.40 1 | :300 |
| bridge_mat | 0.40 0.75 0.40 1 | :301 |
| left_down_rack_mat | 0.20 0.40 0.65 1 | :302 |
| left_down_finger_mat | 0.45 0.75 0.95 1 | :303 |
| right_down_rack_mat | 0.40 0.20 0.55 1 | :304 |
| right_down_finger_mat | 0.80 0.50 0.95 1 | :305 |
| left_up_rack_mat | 0.65 0.50 0.10 1 | :306 |
| left_up_finger_mat | 0.95 0.85 0.30 1 | :307 |
| right_up_rack_mat | 0.20 0.55 0.55 1 | :308 |
| right_up_finger_mat | 0.50 0.90 0.90 1 | :309 |

**`grid`** (gripper.xml:313–314): `texture="grid_tex" texrepeat="2 2" texuniform="true" reflectance="0.2"`. **SCENE FURNITURE — not part of the module**; it is the floor finish only.

##### (c) Textures — 1 total (SCENE FURNITURE)

**`grid_tex`** (gripper.xml:310–312): `type="2d" builtin="checker" rgb1="0.20 0.30 0.40" rgb2="0.10 0.15 0.20" width="512" height="512"`. Procedural builtin checker — **no external image file**. **SCENE FURNITURE** (used only by the `grid` material on the floor).

#### Module vs. scene furniture (for the integrator)

- **MODULE assets (keep):** all 10 visual meshes, all 278 collision meshes, and the 10 part materials `*_mat`. These live entirely under the `base` body subtree (root `<body name="base" pos="0 0 0">` at gripper.xml:325).
- **SCENE FURNITURE (drop on integration):** `texture grid_tex` (:310), `material grid` (:313), plus the worldbody floor/lights (geom `floor`, lights `overhead`/`fill`) which are outside the `base` subtree. None of these are referenced by any gripper part — safe to delete when bolting the module onto a wrist.

## 3. Mount interface

Now the jaw kinematics are fully characterized. At ctrl=0 (neutral): UP pair gap = 72.9mm, DOWN pair gap = 35.1mm — a **partially-open / mid-travel rest pose**. Positive y-ctrl closes (gap→~0 at max), negative y-ctrl opens wide (gap→~115mm at min). The finger_x joints at qpos=0 are fully retracted (range starts at 0).

### 3.1 Module root & sites
- The model compiles cleanly: `mujoco.MjModel.from_xml_path("<gripper source>/gripper.xml")` → `nbody=11, njnt=9, nu=7, neq=2, nmesh=288`, `nkey=0`.
- The transplantable module root is `<body name="base" pos="0 0 0">` at **gripper.xml:325**, the single child of `<worldbody>` (gripper.xml:317) that is the gripper; the other worldbody entries (`floor` gripper.xml:318, lights `overhead`/`fill` gripper.xml:320‑322) are scene furniture, not part of the module.
- Compiled base body frame: `body_pos = [0,0,0]`, `body_quat = [1,0,0,0]` (identity), `parentid = 0` (world). The **base body frame coincides with the world origin at identity.**
- **There is NO mounting `<site>`.** Grep of the whole file: `<site>` count = **0**. No site of any kind exists; the integrator must author the mount pose from the geometry below.
- All angles in this file are radians (`compiler angle="radian"`, gripper.xml:3); all transforms reported here are pure translations plus an identity (zero-radian) rotation.

### 3.2 Base visual mesh geometry (`meshes/base.obj`, scaled 0.001 → m) Loaded with trimesh, 4990 vertices, scaled mm→m. The base body sits at identity, so **base-frame coordinates == the values below**.

| Quantity | Value (m, base frame) |
|---|---|
| AABB min | `(-0.053402, -0.050000, -0.012000)` |
| AABB max | `( 0.028297,  0.050000,  0.075965)` |
| Extents (X,Y,Z) | `( 0.081699,  0.100001,  0.087965)` |
| AABB center | `(-0.012552,  0.000000,  0.031982)` |
| Vertex centroid | `(-0.021537,  0.009820,  0.017604)` |
| Volume centroid (mesh COM) | `(-0.013354, -0.003539,  0.019442)` |
| `<inertial>` COM (gripper.xml:327) | `(-0.0235692, 0.00865956, 0.0152749)`, mass `0.0695138 kg` |

**Identifying the wrist-mating face.** The grasp workspace points hard in **+X**: all four rack/finger subtrees span world/base X ≈ `0.002 → 0.113 m` (exact visual AABBs: racks `x∈[0.002, 0.1135]`, fingers `x∈[0.0317, 0.1134]`), finger workspace centroid ≈ `(0.0726, 0.0, 0.0261)`. Nothing of the gripper reaches more negative than the base's own `x = -0.0534`. Therefore the arm/wrist is on the **−X** side and the mating face is the **−X end** of the base.

This is confirmed by a circular-flange signature, not just the AABB extreme. Looking down the X axis at the −X end (`x < -0.043`), the vertices form concentric rings centred at **YZ = (0.000, 0.0325)** with a central bore (inner radius ≈ 6 mm) and outer register ≈ 15.5–20 mm — i.e. a round wrist-roll **pilot/spigot boss with a bolt circle**. The competing flat faces are NOT circular: the −Z face is a plain rectangular plate (45 × 100 mm) and ±Y faces are small side walls. Only the −X end is a circular flange. (Renders saved at `<gripper source>/gripper_handoff/base_views.png` and `<gripper source>/gripper_handoff/base_Xend.png`.)

Note the −X end has **two** parallel planes: an outer raised boss at `x = -0.0534` (4.48 cm², the circular spigot that registers into the flange counterbore) and a larger flat plate ~9 mm behind it at `x ≈ -0.0445` (≈17 cm², the actual bolt-down face). I take the outermost contacting plane `x = -0.0534` as the nominal mating datum; if the real flange seats on the recessed plate, shift the X datum by +0.0089 m (see flags).

### 3.3 Mating face in base-frame coordinates
- **Mating-face center (flange/bore axis point):** `(-0.05340, 0.00000, 0.03250)` m, base frame.
- **Mating-face outward normal** (the face itself looks back toward the arm): **−X = `(-1, 0, 0)`**.
- **Wrist-roll axis / OUT-away-from-arm direction:** **+X = `(1, 0, 0)`** (the gripper-reach direction). The roll axis passes through the flange center along X.
- The base frame is **offset** from this face: it is +0.0534 m in X and +0.0325 m in Z away from the flange center (Y is centered, offset ≈ 0).

### 3.4 Re-root transform (author on the module top `<body>`) The model's +X **already** points outward (away from arm) and the flange normal is already axis-aligned to X, so **no rotation is needed** — only a translation to move the flange center onto the flange origin.

```
<body name="gripper_module" pos="0.053400 0.000000 -0.032500" quat="1 0 0 0">
```

- `pos  = (0.053400, 0.000000, -0.032500)` m  — this is `-flange_center`, shifting the flange center to the new parent origin.
- `quat = (1, 0, 0, 0)` (wxyz), identity = **0 rad** rotation.

Verification: flange_center `(-0.0534, 0, 0.0325)` + pos `(0.0534, 0, -0.0325)` = `(0,0,0)`. After this, the mating face sits at the parent (flange) origin, +X points out along the wrist-roll axis away from the arm, and the gripper reaches into +X. The base body frame does **not** coincide with the mating face at identity — the translation above is required.

### 3.5 Neutral / open default pose (ctrl = 0, no keyframe) `nkey = 0`, so the rest pose is purely `ctrl = 0 → mj_forward`. All 9 slide joints report `qpos = 0`:

| Joint (gripper.xml) | type | axis | range (m) | qpos @ ctrl0 |
|---|---|---|---|---|
| `bridge_z` (438) | slide | `0 0 1` | `[-0.027, 0.030]` | `0.000` |
| `left_up_y` / `right_up_y` (eq-coupled) | slide | `0 1 0` / `0 -1 0` | `[-0.020, 0.035]` | `0.000` |
| `left_down_y` / `right_down_y` (eq-coupled) | slide | `0 1 0` / `0 -1 0` | `[-0.040, 0.017]` | `0.000` |
| `left_up_finger_x`, `right_up_finger_x` | slide | `1 0 0` | `[0.000, 0.055]` | `0.000` |
| `left_down_finger_x`, `right_down_finger_x` | slide | `1 0 0` | `[0.000, 0.055]` | `0.000` |

Two `<equality>` joint couplings (`neq=2`): `right_up_y = left_up_y` and `right_down_y = left_down_y` (polynomial coeff `[0,1,...]`, i.e. 1:1), so the two jaws of each pair move symmetrically. 7 `<position>` actuators: `m_bridge_z, m_up_pair, m_down_pair, m_left_up_finger, m_right_up_finger, m_left_down_finger, m_right_down_finger`.

**Resulting jaw state at ctrl=0 (mid-travel, partially OPEN):**
- UP finger pair inner gap = **0.0729 m**, DOWN finger pair inner gap = **0.0351 m**.
- Finger telescoping (`*_finger_x`) is **fully retracted** at qpos=0 (range starts at 0; positive extends fingers up to +0.055 m in +X).
- Lateral jaw direction: **negative** y-ctrl opens wide (UP/DOWN gaps → ~0.113/0.115 m), **positive** y-ctrl closes (gaps → ~0.003/0.001 m). So ctrl=0 is neither fully open nor closed — it is a defined mid-spread rest pose.
- `bridge_z = 0` is mid-range (range `[-0.027, 0.030]`).

## 4. Bundled files

All 278 collision pieces attributed across 10 bodies, totaling 1,042,289 bytes — matches the disk count. Everything verified.

All paths under `<gripper source>/gripper_handoff/`. Verified against a fresh read and a real MuJoCo compile (system `python3` + `mujoco`). Angle units are **radians** (`gripper.xml:3`, `compiler angle="radian" meshdir="meshes" autolimits="true"`); model is a single self-contained MJCF with **0 `<include>`** elements; all 288 meshes use `scale="0.001 0.001 0.001"` (mm to m).

### Folder layout (tree)
```
gripper_handoff/
├── gripper.xml                         (verbatim copy of source; diff -q = identical)
└── meshes/
    ├── base.obj                        ┐
    ├── bridge.obj                      │ 10 top-level VISUAL meshes
    ├── left_up_rack.obj                │
    ├── left_up_finger.obj             │
    ├── left_down_rack.obj             │
    ├── left_down_finger.obj           │
    ├── right_up_rack.obj              │
    ├── right_up_finger.obj            │
    ├── right_down_rack.obj            │
    ├── right_down_finger.obj          ┘
    └── collision_pieces/
        └── *.obj                       278 referenced COLLISION meshes
```
`meshdir="meshes"` is relative to the copied `gripper.xml`, so it resolves inside the bundle.

### Mesh counts and bytes
| Category | Files | Bytes |
|---|---:|---:|
| Visual (top-level `.obj`) | 10 | 7,537,645 |
| Collision (`collision_pieces/*.obj`) | 278 | 1,042,289 |
| **Total copied** | **288** | **8,579,934** |

288 unique `file="..."` references parsed from `gripper.xml`; each copied exactly once (no duplicates). Source `collision_pieces/` holds 823 objs — only the 278 referenced were copied; unreferenced ones were skipped.

### 10 VISUAL meshes
| File | Bytes |
|---|---:|
| base.obj | 954,392 |
| bridge.obj | 941,658 |
| left_up_rack.obj | 950,809 |
| left_up_finger.obj | 467,802 |
| left_down_rack.obj | 948,603 |
| left_down_finger.obj | 467,610 |
| right_up_rack.obj | 946,374 |
| right_up_finger.obj | 461,314 |
| right_down_rack.obj | 944,049 |
| right_down_finger.obj | 455,034 |
| **Total** | **7,537,645** |

### 278 COLLISION pieces — per-body counts Attributed by walking the `<body>` nesting in `gripper.xml` (root subtree `<body name="base">` at `gripper.xml:325`) and assigning each `collision_pieces/` geom to its enclosing body.

| Body | Pieces | Bytes |
|---|---:|---:|
| base | 101 | 426,427 |
| bridge | 35 | 168,226 |
| left_up_rack | 31 | 93,621 |
| left_up_finger | 4 | 18,752 |
| left_down_rack | 34 | 94,405 |
| left_down_finger | 4 | 18,588 |
| right_up_rack | 29 | 89,641 |
| right_up_finger | 4 | 17,797 |
| right_down_rack | 32 | 97,091 |
| right_down_finger | 4 | 17,741 |
| **Total** | **278** | **1,042,289** |

### MISSING none — all 288 referenced mesh files resolved on disk under `<gripper source>/meshes/` and were copied.

### Standalone recompile Loaded `<gripper source>/gripper_handoff/gripper.xml` via `mujoco.MjModel.from_xml_path`; reports nmesh=288, njnt=9 (slide), nu=7 (position actuators), neq=2 (equality couplings) — consistent with the established ground truth.

Bundled gripper_handoff/gripper.xml recompiles standalone: OK, ngeom=289

(Note: ngeom=289 = 288 mesh geoms in the `base` subtree + 1 scene `floor` plane at `gripper.xml:318`. The `floor`, lights, and `grid` material/texture are scene furniture carried along in `worldbody` but are not part of the transplantable `base` module; they add no mesh files.)

## 5. Gaps / quality flags

Aggregated from every section's "Quality flags", prioritized — items that affect correct integration/dynamics first, then performance/optimization, then informational notes.

### A. Mount-interface assumptions the integrator MUST confirm in CAD before bolt-up
1. **Mating datum plane in X is ambiguous (±0.0089 m).** The −X end has two parallel planes: an outer spigot boss at `x = -0.0534 m` (used as the nominal datum) and a recessed bolt-plate ~9 mm behind it at `x ≈ -0.0445 m`. If the real flange seats on the recessed plate, add **+0.0089 m** to the module `pos.x` (→ `0.0445`). Confirm whether the −X spigot is a pilot register (seats in a counterbore) or the bolt face itself. (§3.2, §3.4)
2. **Flange center in Z = +0.0325 m, Y = 0 is derived from mesh tessellation, not a CAD datum** (±~0.5 mm). Verify the bolt-circle/bore center precisely. (§3 mount flag 2)
3. **Wrist-roll axis = base +X at identity rotation** is inferred from geometry (circular flange normal along X, fingers reach +X). Confirm the real flange register axis is exactly gripper +X, that "+X points away from arm" matches the wrist-roll output convention (no 180° flip about Z/Y), and that no clocking angle is required for keyed bolt holes. (§3 mount flag 3)
4. **Bolt-circle diameter / hole pattern / clocking NOT extracted.** Spigot shows a bore (~6 mm radius) and outer register (~15–20 mm radius) only. Match actual bolt-circle dia, hole count, and any keyed/clocked orientation; a clocking requirement would add a roll (X-axis) rotation to the mount quat. (§3 mount flag 4)
5. **No mounting `<site>` exists** (`<site>` count = 0). The re-root transform is authored from mesh geometry only; consider adding a `<site name="mount" pos=...>` at the flange center after CAD confirmation for downstream reuse. (§3.1, §3 mount flag 5)

### B. Inertials that look fake/default
6. **None found — all 10 inertials are explicit and CAD-plausible.** Every non-world body carries an explicit `<inertial>` (grep = 10); none fall back to MuJoCo geom-derived auto-inertia; none are round/aligned placeholders (offset COMs, tilted `iquat`, three distinct moments throughout). This is a positive finding, not a defect. (§2.1 bodies flag 1)
7. **Mass realism unverified against hardware.** Each of the four `*_rack` bodies is 0.036995 kg (~0.148 kg total, ~47% of the 0.311601 kg gripper) — the largest mass contributor — worth a hardware sanity check. The base mass/inertial (`0.0695 kg`, COM `(-0.0236, 0.0087, 0.0153)`, gripper.xml:327) is model-authored, not measured; verify if wrist dynamics matter. (§2.1 bodies flag 2, §3 mount flag 6)
8. **Confirm down-rack/down-finger CAD is truly mirror-symmetric.** The two down racks and the two down fingers carry byte-identical `diaginertia` (mirror copies), unlike the up racks which are independently meshed (tiny 4th–5th sig-fig differences). This is correct mirroring, not laziness — flag only to confirm the CAD geometry really is mirror-symmetric. (§2.1 bodies flag 3)

### C. Collision: raw high-poly or over/under-decomposed geometry
9. **No raw high-poly mesh is used for collision (positive finding).** Every `group=3` geom points at a `collision_pieces/*.obj` convex hull (max single piece 404 faces, on `bridge`); the 10,000–12,972-face full meshes are `group=2` visual-only. No concave full mesh silently drives contact. (§2.4 collision flag 2)
10. **`base` collision is over-fragmented (101 hulls / 13,784 faces, ~41% of all collision pieces & faces); `bridge` is also heavy (35).** CoACD over-decomposed a structurally simple, mostly-static mounting body. If the base rarely touches objects (it bolts to the wrist), replace with a handful of primitive boxes/cylinders or a coarser decomposition to cut broad-phase pair count and narrow-phase cost. (§2.4 collision flag 1)
11. **33 collision pieces exceed 200 faces** (convex hulls rarely need >~100). A hull-simplification pass could shrink the 33,700-face collision total with no loss of fidelity. (§2.4 collision flag 4)
12. **A few near-degenerate hulls (4 pieces ≤12 faces, min = 8).** Tetra/box-like slivers from decomposition; confirm non-zero volume so they don't generate unstable contacts. (§2.4 collision flag 3)
13. **Self-collision is globally OFF by construction** (`contype=1`/`conaffinity=0` on all 278 pieces), not via explicit `<contact>` exclude/pair rules. Efficient, but the gripper cannot detect fingers closing onto each other or onto the body — self-contact is structurally impossible, not merely filtered. (§2.4 collision flag 5)
14. **Uniform `condim=3` everywhere — no torsional fingertip friction.** If the use-case relies on resisting in-hand twist of a grasped part, `condim=4`/`6` plus a torsional friction term on the four `*_finger` bodies may be warranted. (§2.4 collision flag 6)

### D. Visual meshes needing decimation
15. **No single visual mesh exceeds the ~15k-face threshold, but all are decimation candidates.** Heaviest are the 4 finger meshes at **12,972 faces each**; the other 6 (base, bridge, 4 racks) are exactly **10,000 faces each** — the uniform counts strongly suggest auto-remesh/face-cap, not hand optimization, and are heavier than needed for simple rack/base geometry. Total render budget is 111,888 tris for a small gripper; decimate if render budget matters. (§2.5 assets flag 1)

### E. Actuator / control assumptions
16. **All seven forceranges are identical ±87 N, but the 87 N figure is derived only for the coupled parallel-jaw case** (`τ/(2r)`, FEETECH HLS3915M-C001, gripper.xml:701). It is physically motivated for `m_up_pair`/`m_down_pair` but reused unchanged on the bridge Z lift and the four finger_x slides, whose kinematics are not the 1-pinion-2-rack case. Likely a conservative blanket peak clamp — review if accurate per-axis force limits matter. (§2.3 actuators flag 1)
17. **Right `_y` jaw joints (`right_up_y`, `right_down_y`) are un-actuated, by design.** They are positioned only by the soft equality constraints (eq slaves right→left, gripper.xml:696-697) plus the mirrored `axis="0 -1 0"`. Jaw symmetry depends entirely on the soft solver; under heavy/asymmetric contact the 1:1 link can deflect (stiff `solref="0.005 1"` + high `solimp` mitigate but do not eliminate). If an equality is ever disabled, the right jaws become free/uncontrolled — a single-point dependency. (§2.2 joints flag 1, §2.3 actuators flag 4)
18. **Asymmetric `_y` ranges between pairs:** upper `[-0.02, 0.035]` vs lower `[-0.04, 0.017]` (gripper.xml:481/534 vs 586/642). The two pairs close from different sides / have different sign conventions for "closed" vs "open"; downstream control must not assume up and down pairs share a sign/range. Confirm the travel envelopes match intended jaw geometry/reach. (§2.2 joints flag 4, §2.3 actuators flag 3)
19. **Lower pair anchored to `base`, upper pair to `bridge`.** The upper `_y` racks ride the `bridge_z` lift; the lower racks do not. Verify this split is intended for any task assuming all four jaws move together vertically. (§2.2 joints flag 5)
20. **Finger_x neutral sits on the lower ctrl/joint limit (0 of `0 0.055`).** At `ctrl=0` the four finger actuators are pinned against their lower bound; negative finger targets (or negative-qvel overshoot) clamp at 0 — no compliance margin below neutral. Intentional ("fingers home retracted") but noted. (§2.3 actuators flag 2)
21. **Low derivative damping on `<position>` actuators** (bridge 3000/15, pairs 800/8, fingers 150/1; `kv=1` on fingers). Most velocity damping comes from the joints' own `damping` (40 / 12 / 2.5), not the actuator PD; step commands may overshoot — verify against intended controller bandwidth. (§2.3 actuators flag 5)
22. **No `<motor>`/torque-mode actuators** — the model cannot be driven in pure force/torque mode without adding actuators; all control is position-target. Flag if force-control handoff was expected. (§2.3 actuators flag 6)

### F. Constraint / authoring details (informational)
23. **`solimp` authored with only 3 of 5 values** (`0.95 0.99 0.001`) on both equalities; compiler back-fills `0.5 2.0`. The target should know the last two are defaults, not chosen. (§2.2 joints flag 2)
24. **`pos` omitted on all 9 joints** — every joint relies on default `pos="0 0 0"` (compile-confirmed). Fine for slide joints (pos only sets anchor reference), but implicit, not authored. (§2.2 joints flag 3)
25. **Equality `joint1`/`joint2` ordering and zero offset:** `joint1=right_*` (constrained), `joint2=left_*` (reference, actuated); `polycoef` offset c0=0 assumes both joints rest at 0 — correct here since all `qpos0=0`, but any nonzero rest offset would need a nonzero c0. (§2.2 joints flag 6)
26. **All body frames are identity** (`body_pos=0 0 0`, `body_quat=1 0 0 0` for every body); all spatial offset lives in geoms + `<inertial>` pos/quat. Joint axes act at the parent origin; downstream consumers must not assume the body frame sits at the part's COM or geometric center. (§2.1 bodies flag 4)
27. **`iquat` sign on the down-finger pair is negative** (`-0.498583 …`). Harmless (quaternion double cover, q and −q are the same rotation); noted so a reader comparing against a right-handed CAD export doesn't think it is flipped. (§2.1 bodies flag 5)

### G. Units / scale and packaging
28. **Units/scale are unambiguous and consistent** (positive finding): `angle="radian"` (gripper.xml:3); every mesh `scale="0.001 0.001 0.001"` (source OBJs in mm → meters). Every joint is `slide` (prismatic), so all `range`/`ctrl`/`qpos` are linear displacements in **meters**, never angles — no degree↔radian or angular conversion ever applies. (§1, §2.1, §2.2, §2.3)
29. **Color is material-driven only** (no geom-level `rgba`). Recoloring means editing the 10 `*_mat` entries (gripper.xml:300–309), not the geoms. (§2.5 assets flag 3)
30. **Disk holds many unreferenced OBJs** (34 top-level + 823 collision vs. 288 referenced). Not a model defect; the bundle already ships only the 288 referenced files — drop the ~570 dead collision OBJs from any source-tree handoff. (§2.5 assets flag 4, §4)
