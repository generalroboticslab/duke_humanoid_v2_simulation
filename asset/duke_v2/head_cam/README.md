# head_cam — what runs, what is provenance

Two MJCF modules are generated here:

- `head_camera_dual.xml` — two camera columns at y = ±0.065 (robot left/right)
- `head_camera_single.xml` — one forward-facing column at the top-plate center (y = 0)

Both come out of **one** script. Regenerate with:

```bash
python head_camera_creation.py
```

Output is deterministic — re-running with no source change reproduces both XMLs byte for byte.

## Live build path (2 files)

| File | Role |
|---|---|
| `head_camera_creation.py` | builds both modules via `asset/create/robot_builder.py` (frame-on-joint emitter, same as the humanoid) |
| `cam_fusion_info.py` | the 3 left links' mass / COM / inertia as pasted Fusion 360 Properties text, plus `parse_fusion` / `mirror_180z`. Imported at build time. |

Nothing else in this directory is imported by the model.

## Provenance scripts (do not run in the build)

Kept as the audit trail for the hardcoded constants. Each one's docstring carries its own IN / OUT / DEPS / PINS block. None are importable by the build, and none run in the project env — they need CAD or hardware.

| File | Stage | Runnable today? |
|---|---|---|
| `mirror_step.py` | CAD 1: `HeadCameraV2.step` → `HeadCameraV2_dual.step`, twin rotated 180° about the plate hole | needs `build123d` (not in project env) |
| `derive_dual_articulation.py` | CAD 2: joint axes, pitch range, part→body assignment from the dual STEP | **no** — needs build123d/OCP/cadquery *and* two missing inputs |
| `extract_relative_positions.py` | measures component poses relative to the top plate, regex STEP parser | **yes** — numpy only |
| `read_d436_extrinsics.py` | reads factory RGB↔depth extrinsics off the physical camera | needs `pyrealsense2` + D436 on USB |

## Where each hardcoded number came from

| Constant (in `head_camera_creation.py` unless noted) | Source |
|---|---|
| `YAW_ANCHOR_Y` = 0.065, `YAW_Z` = 0.52, `PITCH_Z` = 0.625442 | `derive_dual_articulation.py` (rotor-flange fit on the dual STEP), cross-checked by `extract_relative_positions.py` |
| `PITCH_RANGE`, `YAW_RANGE`, joint-axis directions | `derive_dual_articulation.py` |
| `RZ180` / right column = left rotated 180° (**not** mirrored) | `mirror_step.py`; same matrix reused by `cam_fusion_info.mirror_180z` |
| `GLASS_CENTER_L` = (−0.025, −0.065, 0.662) | user's Fusion Inspect>Measure on the D436 front glass face |
| `DEPTH_FROM_GLASS_OPT` | Intel D400 datasheet, Tables 4-16 / 4-19 |
| `RGB_C2D_T_OPT` | `read_d436_extrinsics.py` on unit serial 408122071763 (per-unit factory calibration) |
| per-link mass / COM / inertia | `cam_fusion_info.py` — Fusion 360 Properties text, pasted verbatim |
| `COLLISION_L` capsules | hand-fit from the link meshes; **not** derived — re-fit if the OBJs are re-exported |

## Artifacts NOT in git

- `source/HeadCameraV2_dual.step` — regenerate with `mirror_step.py` (only `source/HeadCameraV2.step` is committed)
- `meshes/components_high_res/` — high-res per-part OBJs consumed by `derive_dual_articulation.py`; only the 3 collapsed link meshes in `meshes/links/` are committed

Both are needed *only* to re-derive the CAD constants. The model build does not touch them.

## To re-derive after a CAD change

1. Re-export `source/HeadCameraV2.step` and the `meshes/components_high_res/` parts from Fusion.
2. In a CAD env: `python mirror_step.py`, then `python derive_dual_articulation.py`.
3. Transcribe the printed axes / ranges into `head_camera_creation.py`.
4. Re-paste the 3 Fusion Properties blocks into `cam_fusion_info.py` (grouping rules are in its docstring).
5. Re-fit `COLLISION_L` if the meshes moved — nothing derives those.
6. `python head_camera_creation.py`, then diff both XMLs to confirm only intended values changed.
