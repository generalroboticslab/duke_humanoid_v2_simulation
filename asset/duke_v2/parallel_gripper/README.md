# parallel_gripper

A single-servo, rack-and-pinion **parallel-jaw** gripper, packaged as a self-contained MuJoCo module in the `duke_v2` asset family. Two jaws ride mirrored Y slides and are 1:1 coupled, so **one** FEETECH servo opens/closes both. Carries 8 AprilTag (tag36h11) fiducials per hand — 4 on the jaw plates + 4 on the base tag-holder pads — for visual servoing.

Imported from the upstream `mini_gripper` CAD→MuJoCo pipeline; current geometry source is the full-assembly re-export `ParallelGripper0710.step` (see `mini_gripper_old/ParallelGripper0710/` for the STEP, the body grouping and the probe CSV). That CAD rev added the two base tag-holder plates, their 4 pocket-seated tag pads and a `usb_c_protector` — all baked into `meshes/base.obj` as parts of the base link. The `cnc_flange` mesh (originally from `CNC.step`) is vertex-identical in the new assembly and was kept as-is.

## Layout

```
parallel_gripper/
├── parallel_gripper.xml             # GENERATED MJCF artifact — do not hand-edit (see creation script)
├── parallel_gripper_creation.py     # the GENERATOR: declarative builder -> parallel_gripper.xml
├── parallel_gripper_fusion_info.py  # ALL physics: Fusion-360 text strings (parse_fusion; physics source of truth)
├── meshes/
│   ├── base.obj / left_rack.obj / right_rack.obj   # 3 visual gripper meshes (mm; scale 0.001 in XML)
│   │                                     #   base.obj includes the tag holders + pads + usb-c protector
│   └── tags/tag_<id>.{png,obj}           # 16 AprilTag decals, both hands (PNG + flat zero-thickness quad)
├── flanges/                              # DECOUPLED mount flange mesh (composed by the loader, not in the XML)
│   └── cnc_flange.obj                    #   the silver mounting disc, visual-only
├── tag_grids.py                          # AprilTag 8x8 bit grids (TAG_GRIDS) — the configurable tag source (no JSON)
├── tag_layout.py                         # the 8 decal SLOTS + per-hand tag ids (HAND_TAGS) — placement source of truth
├── make_tags.py                          # grids + layout -> meshes/tags/tag_<id>.{png,obj}
├── PositionDeter/
│   └── RELATIVE_POSITION_flange__parallel_gripper.md   # wrist mount geometry
└── README.md
```

## Decoupled mounting flange

The silver disc with the bolt-hole ring (`cnc_flange`, from `CNC.step`) **is this gripper's mounting flange**. It is kept **out of the gripper XML** and **composed onto `base` at load time** by the loader (`parallel_gripper.py:_attach_flanges`), so it can be edited/swapped independently — same pattern as `cartesian_hand_v2`:

- `PARALLEL_GRIPPER` — gripper **+ `cnc_flange`** (default; the real mounting hardware).
- `PARALLEL_GRIPPER_BARE` — bare gripper (no flange), for swapping a different one.

The flange is visual-only (no collider — the arm owns collision), welded at `pos=0`. Its **mesh/parent/group/colour** is `FLANGE_CONFIG` in `parallel_gripper.py`; its **mass/COM/inertia** is Fusion-text in `parallel_gripper_fusion_info.FLANGE_INFO`, parsed live at load.

## Model at a glance (compiled)

Pure gripper (the generated XML): `nbody=4` (world + base + 2 racks) · `njnt=2` slide joints · `neq=1` mimic equality · `nu=0` (the position actuator is NOT in the XML — the loader re-adds it as a Builtin). **With the `cnc_flange` composed on** (the default `Hand`): `nbody=5`, total mass ≈ **346 g** (gripper 324 g + flange 22 g).

Tree: `base → {left_rack, right_rack}` (+ decoupled `cnc_flange` welded onto `base` at load). Every body is declared at identity (`pos="0 0 0"`, no quat); all geometry/articulation lives in joints, geoms and `<inertial>` (meshes are world-coord, authored in mm → every mesh is `scale="0.001"`). `base` is the mounting root: the **`cnc_flange`** (silver disc) is the −X mount end; the jaws reach **+X** (fingertips at x ≈ +0.12 m).

The two jaws are a symmetric pinch: `left_rack_y` (slide `+Y`) and `right_rack_y` (slide `−Y`), coupled 1:1 by an `<equality>` so the single actuator `m_grip` drives both. Jaw coordinate `q` (m): `q=0` is the **half-open spawn default** (the authored CAD pose, `ref=0` — mjlab resolves unlisted keyframe joints to 0, so the spawn opening comes for free), `q=−0.05` is **fully open** (~184 mm finger gap), fingers **touch** near `q=0.018`; range `[−0.05, 0.0347]`, ctrl linear **meters**. NOTE the 0710 STEP authors the jaws ~3.025 mm wider per side than this canonical pose; the export pipeline re-poses the rack meshes back (they came out vertex-identical to the 0626 meshes), so the XML, keyframes and tag slots are untouched — only the rack Fusion COMs need the ∓3.025 mm y-compensation when baking (see `parallel_gripper_fusion_info.py`).

## Properties / data management (no JSON)

Physical properties are stored the **same way as `head_cam`** and `cartesian_hand_v2` — raw Fusion-360 "Properties" **text in Python string constants**, parsed by `parse_fusion()`. There are **no `.json` files**:
- `parallel_gripper_fusion_info.py` holds `GRIPPER_INFO` (3 link strings) and `FLANGE_INFO` (1 flange string), parsed by `parse_fusion()`. The Fusion strings are the physics source of truth; the gripper-link `<inertial>` is emitted by the creation script (below), and the flange is parsed live by the loader. (The legacy `--write` path that patched `<inertial>` into a hand-authored XML is superseded by the generator.)

## Regenerate (the XML is a build artifact)

`parallel_gripper.xml` is **generated** — do not hand-edit it. Edit the declarative `parallel_gripper_creation.py` (joint ranges/dynamics, jaw capsule colliders, materials) or the Fusion physics, then:

```bash
python parallel_gripper_creation.py     # -> parallel_gripper.xml
```

Built on the same `asset/create/robot_builder.py` toolchain as `humanoid_v21`. The generated XML carries only what the loader keeps (3 bodies + 2 slide joints + coupling equality + jaw capsules); floor/lights/actuator are deliberately omitted (preview furniture / loader-owned).

## AprilTag fiducials (configurable / replaceable)

8 `tag36h11` tags per hand ride the gripper — LEFT hand: jaws `80`–`83` + base pads `84`–`87`; RIGHT hand: jaws `90`–`93` + base pads `94`–`97` (disjoint sets so the detector tells the hands apart). Each is a **zero-thickness textured quad decal** (a flat PNG image glued 0.1 mm off the plate/pad face; no thickness, no mass, no collision). The plate/pad thickness is already in the CAD, so the tag must be a picture, not geometry.

The pipeline is kept in-module so tags are **swappable without touching the gripper geometry** (no JSON):
- `tag_grids.py` — the `TAG_GRIDS` 8×8 bit grids (from `cv2.aruco DICT_APRILTAG_36h11`).
- `tag_layout.py` — the 8 physical `SLOTS` (CAD ground truth) + `HAND_TAGS` id mapping + the universal `TAG_ROT=180` (detected tag +X = +gripper X, robot forward).
- `make_tags.py` — grids + layout → `meshes/tags/tag_<id>.{png,obj}`.

The gripper XML is **tag-free**; the loader (`mj_envs/asset_zoo/parallel_gripper.py: _attach_tags`) composes one hand's 8 decals at load time. To re-skin with different tag IDs: edit `TAG_GRIDS` + `tag_layout.HAND_TAGS`, then `python make_tags.py`.

## Collision

Each jaw carries four physical capsules, `class="collision"` (group 3, `contype=1`, `conaffinity=0`, so jaws collide with grasped objects/world but not each other or the base):

- `*_rack_collision0..2` — inner grasping-face box, 18 mm thick, filled by three capsules. Its grasp-facing surface remains on visual contact plane.
- `*_rack_collision3` — finger-back capsule, behind contact plane.

`base_collision` is one fixed 52 mm × 180 mm × 52 mm base-slider capsule, centered at `(-1.7, 7.5, -10) mm`. It stays on `base` while racks translate. The `*_rack_ikproxy` capsules remain group 5 / `contype=0` IK-only guards and never make physical contact. `cnc_flange` stays collider-free. The asset_zoo loader matches physical jaw/base capsules via `HAND_COLLISION_GEOM_REGEX`. The RIGHT jaw is the LEFT **rotated 180°** about the X-axis at `(JAW_ROT_Y, JAW_ROT_Z)` (the jaws are rotated copies, **not** mirrored), via `mir()`. Edit the capsule/box layout in `parallel_gripper_creation.py` (the XML is generated — never hand-edit), then `python parallel_gripper_creation.py`.

## Quick checks

```bash
conda activate mjhand   # mujoco 3.8.0

# physics: print all bodies, or re-bake the XML link <inertial> from the Fusion-text strings
python parallel_gripper_fusion_info.py
python parallel_gripper_fusion_info.py --write

# tags: regenerate the decals (needs Pillow; the XML stays tag-free)
python make_tags.py

# standalone load
python -c "import mujoco; mujoco.MjModel.from_xml_path('parallel_gripper.xml')"
```

Loader / swappable `Hand`: `mj_envs/asset_zoo/parallel_gripper.py` exposes `PARALLEL_GRIPPER` (and `_LEFT` / `_RIGHT` per-hand variants) — pass it to an arm factory via `hand=`.

## On the humanoid

Wired into `humanoid_v21` as a selectable hand (`HAND_REGISTRY["parallel_gripper"]`): the `_LEFT` variant rides the left wrist, `_RIGHT` the right (disjoint tag ids), each EE site moved to the grasp center. Launch the humanoid with both grippers:

```bash
conda activate mjhand
python mj_envs/asset_zoo/humanoid_v21/humanoid_v21_constants.py \
    --end_effector actuated --hand parallel_gripper
```

`--end_effector welded` mounts them rigidly (no jaw DOFs); omit `--hand` (or pass `cartesian_hand`) for the default 4-finger hand.
