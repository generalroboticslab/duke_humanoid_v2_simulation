# humanoid_v21

The **body** of the duke_v2 humanoid: a 27-DOF robot — waist (1), legs (6×2), arms (7×2) —
packaged as a MuJoCo module alongside its sibling components `head_cam/`,
`parallel_gripper/`, `cartesian_hand/` and `cartesian_hand_v2/`.

The bare model here carries **no head camera, no hand and no actuators**. Those are grafted
on at build time by `mj_envs/asset_zoo/humanoid_v21/humanoid_v21_constants.py`
(`get_humanoid_v21_robot_cfg`), which is what training and the exporter actually consume.

## Layout

| Path | What |
|---|---|
| `humanoid_v21_creation_v3.py` | **Source of truth.** Section-by-section assembly; emits both XMLs below. |
| `humanoid_v21.xml` | Shipped low-res MJCF. `HUMANOID_V21_XML` in the constants module points here. |
| `humanoid_v21_high_res.xml` | Render-quality twin. Same body/joint/site order, bit-identical kinematics/dynamics. |
| `humanoid_v21_mink_scene.xml` | `include`s the low-res XML + a floor/lighting scene, for mink IK probes. |
| `humanoid_v21_resolved.xml` | Generated. Flattened standalone snapshot of the **grafted** variant (camera + gripper). |
| `humanoid_v21_full.urdf` | Generated. Faithful twin, relative mesh paths, for viewers. |
| `humanoid_v21_curobo.urdf` | Generated. Fixed base, absolute mesh paths, for cuRobo. |
| `rig_single/`, `rig_triple/` | Generated. K=1 / K=3 head-camera rig variants for the camera-count ablation. |
| `meshes/` | Link meshes (`.obj`/`.stl`) + `source_stp/` STEP originals. |
| `view_model.py`, `inspect_partial_model.py` | Hand-run viewers. Load the XML, open a MuJoCo window / print the body tree. |

## Regenerating

The builder library (`robot_builder.py`, `builder_helpers.py`, `fusion_info.py`) lives in
`asset/create/` and is **shared** with argus / ball_circle / the other duke_v2 components.
Only this robot's own data lives here; the creation script puts `asset/create` on `sys.path`
so its bare `from builder_helpers import ...` resolves.

```bash
# MJCF: regenerates humanoid_v21.xml AND humanoid_v21_high_res.xml
python asset/duke_v2/humanoid_v21/humanoid_v21_creation_v3.py

# Grafted artifacts (resolved MJCF + both URDFs). No --output needed: the destination is
# RobotSpec.out_dir in the exporter's ROBOTS table.
python asset/create/export_mjspec_to_urdf.py --robot humanoid_v21 --format both

# Camera-count rig variants
python asset/create/export_mjspec_to_urdf.py --robot humanoid_v21 \
    --head-camera actuated_single --format both --output asset/duke_v2/humanoid_v21/rig_single
```

`humanoid_v21_creation_v3.py` regenerates the two committed XMLs **bit-identical** — a
non-empty `git diff` after running it means something drifted. `asset/robot_studio` depends
on this (`tests/test_studio.py` asserts its own export equals `humanoid_v21_high_res.xml`).

The exporter hard-asserts an exact compile snapshot `(nbody,njnt,ngeom,nmesh,nsite,nu)`,
that every resolved mesh exists with a URDF-loadable suffix, a yourdfpy FK round trip, and a
real cuRobo `RobotBuilder` load. It fails loud rather than emitting a subtly wrong model.

## Gotchas

* **Only the bare model is xz-symmetric.** `assert_mirror_symmetry()` must stay scoped to it —
  the grafted parallel gripper is two *identical* parts, not mirrored, so the full entity is
  legitimately asymmetric. Validate dynamic symmetry on the full actuated entity, never here.
* **Mesh names carry `_high_res` in both XMLs; only the `file=` differs.** `strip_high_res()`
  rewrites the *file* attribute, not the mesh name, so `humanoid_v21.xml` says
  `<mesh name="ankle_2_high_res" file="ankle_2.obj">`. Looks wrong, is not — the name is a
  handle, the file is the data. A mesh name without the suffix means the link was authored
  with a literal path instead of the `{hig_res_str}` template (`shoulder_3` was exactly that
  bug, fixed 2026-08-23).
* **`q=0` is not a pose this robot occupies** — with arms straight down the wrists sit 47.5 mm
  inside the hips. Evaluate at the home keyframe, not at zero.
