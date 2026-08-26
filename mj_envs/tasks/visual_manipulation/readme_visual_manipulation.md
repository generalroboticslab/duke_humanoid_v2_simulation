# cuRobo grasp models — humanoid_v21 & G1

How to (re)build and inspect the cuRobo motion-planning model for the two bimanual robots used in `visual_manipulation`: **humanoid_v21** and **Unitree G1**, each with the parallel gripper grafted.

Both follow the SAME 3-stage pipeline. Only the per-robot arguments and module names differ.

```
Python: python   (run everything with this)
```

---

## Pipeline overview

```
  MjSpec variant  ──(1) export──▶  <robot>_curobo.urdf  ──(2) cfg──▶  RobotCfg dict  ──(3)──▶  plan / view
   (grafted gripper)                (fixed grasp-site links)          (spheres+locks+cspace)
```

1. **Export URDF** — `asset/create/export_mjspec_to_urdf.py` compiles the resolved MjSpec variant (arms + gripper + legs + head) and writes a cuRobo-loadable URDF whose fixed links include the per-hand **grasp-center sites** used as cuRobo tool frames.
2. **Build the cuRobo cfg** — a per-robot module turns the URDF + compiled model into the `RobotCfg` dict: collision spheres (from the MuJoCo collision geoms), `self_collision_ignore`, `lock_joints` (all non-arm joints frozen at `qpos0`), and the 14-DOF arm `cspace`.
3. **Plan / view** — the phase-3 verify script plans a bimanual grasp; the sphere viewer overlays the collision model on the mesh.

**Design invariant:** the cfg's resolved variant MUST equal the exporter's variant (same `head_camera`/`end_effector`/`hand`, same `qpos0`). Otherwise `lock_joints`/`cspace` desync from the loaded URDF kinematics and grasps silently miss. The exporter enforces its variant with a compile **snapshot** `(nbody, njnt, ngeom, nmesh, nsite, nu)` that fails loud on drift.

**Coordinate invariant:** planner coordinates are `q_curobo = q_mujoco - qpos0`. Export every finite MuJoCo joint bound in that same coordinate system. Humanoid wrist-3 XML `ref=±pi/2` is folded into `qpos0`; raw `[-pi, pi]` URDF bounds would let cuRobo plan a state that playback maps beyond the physical limit. URDF defaults and limits use `q_curobo`; dynamic playback adds `qpos0`.

---

## Per-robot reference

|                    | humanoid_v21                              | G1                                        |
|--------------------|-------------------------------------------|-------------------------------------------|
| Export variant     | `--head-camera actuated`                  | `--head-camera builtin`                   |
| Base link          | `base_link`                               | `pelvis`                                  |
| Tool frames        | `end_effector_L_site`, `end_effector_R_site` | `left_hand_grasp`, `right_hand_grasp`  |
| URDF               | `asset/duke_v2/humanoid_v21/humanoid_v21_curobo.urdf`   | `asset/unitree_g1/g1_curobo.urdf`             |
| cfg module         | `curobo/ik_curobo_robot_cfg.py`           | `curobo/g1_curobo_robot_cfg.py`           |
| cfg build fn       | `build_robot_cfg_dict_from_urdf()`        | `build_robot_cfg_dict()`                  |
| verify script      | `test/curobo_reach_verify.py --robot v2 --camera --walk --dynamic --mpc` | `test/curobo_reach_verify.py --robot g1 --camera --walk --dynamic --mpc` |
| sphere viewer      | `test/curobo_view_collision_spheres.py`   | `test/curobo_g1_view_collision_spheres.py`|
| Grasp approach     | tilted, beta union (60°,45°) (8 candidates) | tilted, beta union (60°,45°) (8 candidates) |

All variants use `end_effector=actuated`, `hand=parallel_gripper`. The gripper racks are **not** planning DOFs — both robots freeze every non-arm joint (legs, waist, gripper racks) at `planning_home_joint_pos` in `lock_joints`, so cuRobo plans arm-only (14 DOF) with the gripper held at its configured open jaw state (`PARALLEL_GRIPPER_OPEN_RACK_POS_M`).

---

## 1. Export the URDF

```bash
PY=python

# humanoid_v21
$PY asset/create/export_mjspec_to_urdf.py --robot humanoid_v21 --head-camera actuated --format both

# G1
$PY asset/create/export_mjspec_to_urdf.py --robot g1 --head-camera builtin --format both
```

- `--format both` writes the resolved MJCF (`<robot>_resolved.xml`) **and** the URDFs (`<robot>_full.urdf` + `<robot>_curobo.urdf`). Use `urdf` for cuRobo-only, `mjcf` for the MJCF only.
- The exporter runs a preflight snapshot check + a yourdfpy scene-graph FK / cuRobo-coordinate joint-limit round-trip gate. Green output ⇒ URDF matches the compiled model and planner cspace.
- If you change the grafted variant (e.g. a different head camera), update that robot's registry entry in `export_mjspec_to_urdf.py` (`ROBOTS[...]`: `snapshot` + `tool_sites`) or the snapshot assert fails. Get the new snapshot by compiling the variant and reading `(nbody, njnt, ngeom, nmesh, nsite, nu)`.

Verify the grasp-site links landed:

```bash
grep -E "left_hand_grasp|right_hand_grasp" asset/unitree_g1/g1_curobo.urdf            # G1
grep -E "end_effector_L|end_effector_R"     asset/duke_v2/humanoid_v21/humanoid_v21_curobo.urdf  # humanoid
```

### TCP / grasp-site source of truth

`mj_envs/asset_zoo/parallel_gripper.py:GRASP_CENTER_IN_BASE` is the one grasp-center geometry definition. The MuJoCo graft places the humanoid `end_effector_{L,R}_site` and G1 `{left,right}_hand_grasp` sites from it. Both the kinematic attachment path and dynamic weld latch read those live MuJoCo sites, so the weld target is not hard-coded separately.

cuRobo instead reads the fixed tool-frame links in its exported URDF. Therefore, after changing `GRASP_CENTER_IN_BASE`, regenerate both planner URDFs before any evaluation. A stale export makes cuRobo plan to the old TCP while MuJoCo closes the gripper at the new TCP.

```bash
PY=python
$PY asset/create/export_mjspec_to_urdf.py --robot humanoid_v21 --head-camera actuated --end-effector actuated --hand parallel_gripper --format urdf
$PY asset/create/export_mjspec_to_urdf.py --robot g1 --head-camera builtin --end-effector actuated --hand parallel_gripper --format urdf
```

The exporter is the verification gate: it checks the compiled variant snapshot, scene-graph FK, cuRobo loading, and tool-frame existence. Current parallel-gripper geometry exports these local tool offsets: humanoid `(0.152, 0.00096, 0)` and G1 `(0.1885, 0.00096, 0)`.

## 2. Build / smoke-test the cuRobo cfg

```bash
# G1
$PY -c "from mj_envs.tasks.visual_manipulation.curobo.g1_curobo_robot_cfg import build_robot_cfg_dict; build_robot_cfg_dict()"
# humanoid
$PY -c "from mj_envs.tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import build_robot_cfg_dict_from_urdf; build_robot_cfg_dict_from_urdf()"
```

Expected: ~14 active arm DOFs, feet/below-knee subtrees excluded from the collision model, 0 arm joints leaked into `lock_joints` (asserted).

## 3. Verify a grasp (headless gate)

```bash
V=mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py
$PY $V --robot g1 left_right_close      # G1  (stands back; left_right_close is g1-reachable)
$PY $V --robot humanoid                 # humanoid (default bimanual_mixed_close)
$PY $V --robot both left_right_close    # both robots on ONE shared scene, side-by-side
```

One unified script (`--robot g1|humanoid|both`) covers bimanual (default), single-arm (`--single`), and the base-drift deploy-prep study (`--drift [--replan|--mpc|--localik]`). The scene (jittered cube poses) is robot-independent, so `--robot both` and a shared `--seed N` compare the two robots on the exact same layout.

Plans a standalone collision-free bimanual cube grasp (both hands, table + object obstacles) and asserts both grasp-site errors < 1 cm. Add `--view` to replay the trajectory on the mesh (green target spheres at each cube), `--planner-view` to overlay obstacle cuboids + swept collision spheres. Press key `"5"` in the viewer to toggle geom group 5 (planner collision spheres + inflated object cuboid overlay). Press key `"o"` to save the current view as a PNG under `test/captures/<scenario>-<robot>-kin|dyn-<timestamp>.png` (same camera + geom-group visibility on screen; kinematic capture also includes the reach-overlay triad/nav arrow). Works for `--dynamic --view` too, native viewer only -- no key hook on the viser web-viewer fallback (headless/no-DISPLAY).

`o` also prints `cam.lookat`/`cam.distance`/`cam.azimuth`/`cam.elevation` to stdout. Paste those into `_VIEW_CAMERA_POSE` (`curobo_reach_verify.py`, `--dynamic --view`'s `_run_view`) keyed by scenario name so that scenario's `--view` opens pre-framed instead of at the mjlab default free-camera view. Native viewer only, same restriction as the `o` key itself. Scenarios absent from the table keep the default view. Pass `--height-pad <meters>` (default: `0.02` m) to inflate object cuboid height in CuRobo's planner perspective and group 5 overlay while maintaining exact physical dimensions and grasp targets in MuJoCo. Needs a DISPLAY for the `--view` modes. The small RGB triad at world origin (0,0,0) is MuJoCo's built-in world-frame decoration (`opt.frame = mjFRAME_WORLD`, set by mjlab's native viewer), not a model geom. Toggle with key `F6` (cycles frame mode).

## 4. Dynamic reaching: plan once, track through physics

> **Standard Evaluation Condition**: All condition evaluations (dynamics + camera verification) MUST be run with the full dynamic pipeline flag combination: `--camera --walk --dynamic --mpc`.

### Benchmark Evaluation Protocol

Run the physics evaluation path with frozen RL locomotion policy, camera visibility gate, dynamic Warp physics, and position gravity compensation:

```bash
PY=python
$PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py \
    --robot <v2 | v2_fixed | g1> \
    --scenario <scenario_name> \
    --dynamic --walk --camera --mpc [--seed <seed>] [--view]
```

#### Evaluation Flags & Execution Specifications:
* **`--dynamic`**: Realizes the mission through full MuJoCo/Warp physics with the frozen RL locomotion policy (instead of idealized kinematic `mj_forward` joint assignment).
* **`--mpc`**: Enables the local DLS Jacobian reactive tracker with position gravity compensation pre-bending (`--grav-comp=True`). V2 bounds each raw correction against both live measured joints and its collision-checked cuRobo waypoint, preventing accumulated moving-base feedback from pinning an arm at a joint limit.
* **`--camera`**: Enforces realistic camera FOV perception gating. Objects must be framed by the head camera (fixed frustum or 2-DOF actuated gimbal scan) before a reach is committed; unseen objects cannot be reached.
* **`--walk`**: Runs the autonomous multi-target mission FSM (`ACQUIRE` $\rightarrow$ `APPROACH` $\rightarrow$ `EXTEND` $\rightarrow$ `GRASP` $\rightarrow$ `RETRACT`), driving base locomotion and head gaze scanning.
* **`--seed <seed>`**: Deterministically jitters object XY poses (default base seed 42). It does not seed dynamic base-reset or domain-randomization events.

#### Benchmark Metrics Recorded:
1. **Task Success Rate ($P$)**: Fraction of trials in which ALL required scenario objects complete their success condition and the robot remains standing ($\text{upright\_min} > 0.90$). Kinematic mode requires $\text{reach\_err} < 0.05\text{ m}$. Dynamic mode requires every assigned physical grasp latch to activate from finger contact, finish easing, and remain active through closed-grasp hold.
2. **Simulated Completion Time ($\bar{T}$ in Seconds)**: Measured strictly in simulated time through the final physical latch after its closed-jaw hold, using the 50 Hz control loop cadence. Post-grasp home retraction is safety cleanup and is not timed: $$\bar{T} = \text{completion\_control\_steps} \times 0.020\text{ seconds}$$
3. **Target Tracking Precision**: Kinematic mode records final Cartesian tool-site error $\text{visit\_min}$ in meters. Dynamic mode prints live cube-center to grasp-center error as `capture diagnostic` (typically $< 0.010\text{ m}$), but it is not a second looser success gate after physical capture.

`--mpc` is historical flag name. It does **not** run cuRobo MPC every 20 ms. cuRobo plans once; the reactive tracker is local MuJoCo damped-least-squares IK with position gravity-compensation pre-bending (`--grav-comp=True`). Always include `--mpc` during `--dynamic` evaluations — without it, uncompensated gravity droop introduces $\sim 0.055-0.075\text{ m}$ position residuals; with `--mpc`, physical contact capture is reliable and dynamic diagnostic errors are typically $\sim 0.007-0.019\text{ m}$.

### FSM keyframe capture for the per-scenario figure (`--record-keyframes`)

`--record-keyframes DIR` writes one 2K PNG per mission-FSM `walk_phase` transition of the **graded** rollout (same physics run that prints `VERDICT:`, not a replay), on the same chase cam the MP4 uses, plus a `*_keyframes.json` manifest. Offscreen renders are 2560x1440 (`RECORD_W`/`RECORD_H`) so a 60 mm cube and the jaw gap survive being cropped into a figure panel; the MP4 path shares that resolution.

#### Full-robomatrix walkthrough videos (5 robots x 6 scenarios, `--record`)

`--record DIR` records one Blender/EEVEE walkthrough MP4 per (robot, scenario), 2560x1440 @ 50 FPS, on the same chase cam used by the keyframes above. Outputs land at `mj_envs/tasks/visual_manipulation/media/blender_6scenario/<robot>_<scenario>_seed<N>_walk.mp4`.

Delivery grade is set in two places, both raised 2026-08-13 for the paper video set: the render's TAA budget (`MJ_BLENDER_SAMPLES`, default 16, sweep uses 64 -- `photoreal/blender_view.py`) and the H.264 rate control (CRF 14, `photoreal/bridge.py`). The previous fixed-quantiser encode delivered only ~2.2 Mb/s at 1440p50 and macroblocked the gripper/cube contact on motion; the current set averages 7.2 Mb/s.

The matrix is 5 robots (v2, v2_fixed, v2_single, v2_single_fixed, g1) x 6 figure scenarios (bimanual_mixed_close, bimanual_mixed_front_back_close, left_right_close, front_back_close, front_back_far, left_right_far), seed 42, 30 videos total.

**Use `test/dyn_video_sweep.py` rather than the hand-rolled loop below.** It builds exactly this 30-cell matrix, pins the right checkpoint per variant, sets both overrides described next, dispatches across the host's GPUs with a per-GPU concurrency cap, and retries a crashed cell once. The loop is kept as the single-cell reference. Full sweep on ser16 (2026-08-13: 30/30 PASS, ~55 min wall):

```bash
# GPU 3 is left OUT of the pool deliberately -- Blender ignores CUDA_VISIBLE_DEVICES and
# __EGL_DEVICE_ID and renders on GPU 3 regardless, so it is the throughput limit (measured 91% util
# there while all seven sim GPUs sat at 0%) and should not also carry a sim process.
export PATH=$HOME/bin:$PATH; unset DISPLAY     # both remote traps -- see photoreal/README.md
$PY mj_envs/tasks/visual_manipulation/test/dyn_video_sweep.py \
  --this 16 --hosts "16=0,1,2,4,5,6,7" --out ~/tmp/vid2k --cap 2 --record-trials 1
```

`--record-trials 1` gives the one-mission-per-file `seed42` naming this figure set uses; the default 10 pools ten trials into a single `seed42-51` MP4 for the metrics sweep instead. ser16 is rsync-only, so pull the result to grl1 and verify before deleting the remote copy.

Two overrides are NOT defaults and silently produce wrong output if dropped (`dyn_video_sweep` sets both; supply them yourself only when driving `curobo_reach_verify.py` directly):

- **v2_single / v2_single_fixed have a 29-dim action space**; auto-resolution walks past the single-cam parent onto a dual-cam ancestor and dies on `action_scale`. Pin the checkpoint: `--checkpoint runs/HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam/grl2_s0/model_0015000.pt`.
- **bimanual_* stand their human model exactly where the default 145 deg chase-cam sits, occluding the cubes.** 215 deg mirrors the 3/4 view to the left-front. Set via `WALK_CAM_YAW_OFFSET_DEG=215` env.

Single-host workflow (grl1 only has Blender by default):

```bash
PY=python
V=mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py
OUT=mj_envs/tasks/visual_manipulation/media/blender_6scenario
SINGLE_CKPT=runs/HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam/grl2_s0/model_0015000.pt
G1_CKPT=mj_envs/tasks/visual_manipulation/test/checkpoints/g1__G1RmaVelEstArmFlashSacStudentOnlyg1bsk2__2026-07-25_14-54-46__model_0015000.pt

cd <repo root>
for s in bimanual_mixed_close bimanual_mixed_front_back_close left_right_close front_back_close front_back_far left_right_far; do
  case "$s" in bimanual_*) yaw=215 ;; *) yaw=145 ;; esac
  for r in v2 v2_fixed v2_single v2_single_fixed g1; do
    case "$r" in v2_single|v2_single_fixed) c=(--checkpoint "$SINGLE_CKPT") ;;
                g1)                        c=(--checkpoint "$G1_CKPT") ;;
                *)                         c=() ;; esac
    WALK_CAM_YAW_OFFSET_DEG=$yaw "$PY" "$V" --robot "$r" --scenario "$s" \
      --viewer blender --dynamic --walk --camera --mpc --seed 42 "${c[@]}" --record "$OUT"
  done
done
```

Faster: fan out across grl3 + grl2_vicon. Both carry Blender 5.2.0 LTS (tarball install under `~/opt`, no sudo) and `sshfs`-mount grl1's repo (separate GPUs, ONE filesystem -- safe here because every cell runs identical code; never fan two code variants across these hosts). Two remote traps, each costs a full mission's wall time before they surface because the Blender render is the last step:

1. `ssh` forwards X11 (`DISPLAY=localhost:10.0`); Blender grabs the forwarded display, finds no GL, SIGSEGVs. **`unset DISPLAY`** in the remote command, `ssh -x` alone is not enough.
2. `blender` is not on the non-interactive PATH. `mj_envs/photoreal/bridge.py` calls `shutil.which("blender")` inline; a miss throws at MUX time. **`export PATH=$HOME/bin:$PATH`**.
3. **Piping the mission through `tail`** makes `ssh` return `tail`'s exit status, so every cell reports `rc=0` whether it passed or crashed. Redirect to a remote logfile, capture `$?` directly, then grep.

A working remote shard:

```bash
ssh -x <host> 'export PATH=$HOME/bin:$PATH; unset DISPLAY; cd ~/repo/legged_env_v2 && \
  WALK_CAM_YAW_OFFSET_DEG=<yaw> $PY <verify> --robot <r> --scenario <s> \
    --viewer blender --dynamic --walk --camera --mpc --seed 42 <ckpt> --record <out> \
    > /tmp/cell_$$.log 2>&1; rc=$?; \
  grep -E "VERDICT|\[record\] wrote" /tmp/cell_$$.log; exit $rc'
```

#### v2-family quadrant videos (2x2 grid, 5120x2880)

`build_v2_quadrants.sh` (in the same media dir as the inputs) lays the four v2 walkthroughs into one 2x2 quadrant MP4 per scenario -- pixel-identical to the inputs, no scale, no crop. Layout:

```
+-----------+-----------+
| v2_single | v2_single |   row 0: single-cam    col 0 = fixed (welded)   col 1 = actuated (gimbal)
|  _fixed   |           |
+-----------+-----------+
| v2_fixed  |    v2     |   row 1: dual-cam
+-----------+-----------+
```

Shorter cells get their last frame held (`tpad=clone`) so the grid stays temporally aligned. Both the pad and the layout are H.264 `-qp 0` (lossless), so source pixels are preserved bit-perfectly; per-cell bitrate ~2 Mb/s, output ~8 Mb/s x duration. Swap `-qp 0` for `-crf 17` for ~1/3 size at visually-lossless quality (not byte-identical).

```bash
cd mj_envs/tasks/visual_manipulation/media/blender_6scenario
./build_v2_quadrants.sh                            # all 6 scenarios, parallel
./build_v2_quadrants.sh bimanual_mixed_close       # one scenario only
```

Output: `v2_quad_<scenario>_seed42_walk.mp4`, 6 files.

For slide embeds, `<scenario>_web.mp4` sit next to each lossless master: 3840x2160 (down from 5120x2880), H.264 `-crf 18 -preset slow`, same 50fps. ~30x smaller (987M -> 33M on the largest), frame-diffed against the source with no visible loss -- this rendered content (flat shading, no camera grain) compresses far better than the `-crf 17` same-resolution estimate above. Regenerate:
```bash
ffmpeg -i IN.mp4 -vf "scale=3840:2160:flags=lanczos" -c:v libx264 -crf 18 -preset slow \
  -pix_fmt yuv420p -movflags +faststart OUT_web.mp4
```

`make_keyframe_figure.py` now drives the capture itself -- one command, not a shell loop followed by a second script. The six scenarios are independent processes, so it fans them out with `--jobs` concurrent workers (measured: 87 s wall vs 240 s serial at `--jobs 3` on one RTX 4090, 2.7x) instead of paying six cold starts back to back. `--capture missing` (default) skips any scenario that already has a `"pass": true` manifest in `--keyframe-dir`, so retuning the figure layout after a capture sweep costs zero simulation.

```bash
PY=python
OUT=mj_envs/tasks/visual_manipulation/media/keyframes_6scenario

# One shot: captures whatever's missing (parallel), then assembles the 6 x 5 figure.
$PY mj_envs/tasks/visual_manipulation/test/make_keyframe_figure.py --keyframe-dir "$OUT" --jobs 3

# Re-run after editing STAGES/camera constants without re-simulating (reuses existing manifests):
$PY mj_envs/tasks/visual_manipulation/test/make_keyframe_figure.py --keyframe-dir "$OUT" --capture none

# Force a full re-shoot (e.g. new camera calibration) for every scenario:
$PY mj_envs/tasks/visual_manipulation/test/make_keyframe_figure.py --keyframe-dir "$OUT" --capture all --jobs 3
```

Chase-cam calibration for the figure (`CAM_ENV`/`YAW_DEG` in `make_keyframe_figure.py`, passed as env vars to each capture subprocess) is three overrides on the MP4 defaults, because the shipped values frame a watchable video rather than a figure panel: `WALK_CAM_DISTANCE_M=1.35` (from 1.9 -- the 60 mm cube and jaw gap must read at panel size), `WALK_CAM_LOOKAT_Z=0.95` (from 0.68 -- at 1.35 m an aim on the table plate crops the head off), and `WALK_CAM_YAW_OFFSET_DEG` = 145 for most scenarios, 215 for `bimanual_*` (moves the lens to the robot's left-front, clear of the human model). Any of the three can still be exported by the caller before running the figure script to override the built-in default.

Distance is **phase-keyed**, so `WALK_CAM_DISTANCE_M` sets only the TIGHT end of a ramp. While the phase is in `RECORD_WALK_ESTABLISHING_PHASES` (`ACQUIRE`/`DISCOVER`/`APPROACH`) the lens sits at `RECORD_WALK_CAMERA_DISTANCE_WIDE` (2.4 m) and eases in afterwards at `RECORD_WALK_DISTANCE_LERP` (0.12/tick, converged well before the first `EXTEND` sample). Reason: the `Initial` panel's job is the SCENE -- both targets, both supports, the human -- and the manipulation panels' job is the GRASP, and no single distance serves both. At a fixed 1.35 m the `_far` rows showed only one of their two benches and the handoff rows clipped the human to an arm, losing exactly the second target that makes a two-target mission legible; at a fixed 2.4 m a 60 mm cube is a few pixels. A per-scenario distance table was tried first and dropped -- it fixed the `Initial` panel by shrinking every other panel in the same row.

- Filenames are `<robot>_<scenario>_seed<N>_k<idx>_<PHASE>_t<sim_s>.png`. `--record-trials N` writes one PNG set + manifest per seed; keep the seed whose manifest says `"pass": true`.
- Manifest rows carry `phase`, `prev_phase`, `cube` (the visit's target), `t_s` (wall-of-mission sim time) and `t_exec_s` (the `METRICS2`/`PHASES` clock, cuRobo stalls excluded), so a panel caption can quote the same clock the tables do.
- `--keyframe-every 0.5` (figure script default, also settable via `curobo_reach_verify.py --keyframe-every`) supplies mid-phase samples; phase-entry frames alone are insufficient because `EXTEND` entry is still the ready posture. The assembled main figure is 6 x 5, columns `Initial` / `Extend` / `Pre-grasp` / `Grasp` / `Retract` for the first visit. `Extend`/`Pre-grasp` are both `EXTEND` frames (30% and 70% through the phase); they replaced `Extend (early)`/`Extend (late)`, whose two-line wrap cost vertical space at column width. Manifests retain every FSM transition for a supplementary full-state sequence; a two-visit mission repeats `EXTEND..RETRACT`.
- Sized for ONE IEEE column: `--width-in` defaults to 3.5 and must equal the width the PDF is included at, or every font rescales (the old 13.0 in default was included at `\columnwidth`, shrinking 9 pt titles to ~2 pt). Panels are cropped 1:1 (`--crop` default `0.5625 1.0`), since panel width is the scarce axis and the 16:9 side margins spend it on empty floor. `--crop 1.0 1.0` restores the full 16:9 frame. (The square crop used to cost the `L/R far` Initial panel its far bench cube and clip `L/R Human`'s human to a shoulder; the phase-keyed wide distance above fixed both, so the crop is now free.)
- Row labels are the Table IV scenario names verbatim (`L/R close`, `F/B Human`, ...) in table row order, so the figure and the table can be read against each other.
- Keyframe overlays are **deliberately partial**, and the split is the point:
  - **Planned route path: always on.** Drawn from `route_path_triads` -- the same producer both live viewers consume -- via the shared `draw_triads` `mjvScene` backend, at `KEYFRAME_ROUTE_MARKER_SCALE` (2.0x). The viewer-tuned `_WP_WIDTH_M` of 1.5 mm prints below one pixel on a 1.2 in panel; scaled at the call site rather than by raising the constant, since the viewers are its primary consumer.
  - **Grasp-pose triads: off.** They are the long/thick half of `reach_overlay_triads` and land exactly on top of the gripper they annotate.
  - **FOV frustum: establishing phases only** (`RECORD_WALK_ESTABLISHING_PHASES`), i.e. the `Initial` column. The hull is a large translucent shell sharing `FOV_GEOM_GROUP` with the wireframe, so it cannot be hidden separately; at 1.35 m it washes out the whole panel -- invisible at 2560x1440, fatal at 1.2 in on the page. Known cost: it tints the `Initial` column, giving that column a different white balance from the rest of its row. Set the frozenset empty to drop it.
  - **Nav arrow / target axes: off** (MP4-only). Two independent content lists still exist here: the MP4 branch hand-rolls `draw_nav_arrow` + `draw_target_axes` and matches neither the viewer's `_draw_overlays` nor the keyframe path. Unifying them means deciding whether the MP4 should carry grasp triads.

  Recorder and viewers necessarily use different draw call sites -- `mujoco.Renderer.update_scene()` rebuilds the `mjvScene` every call so overlays cannot persist, the passive viewer owns a durable `user_scn`, and mjlab exposes only `DebugVisualizer.add_frame`. Three geometry APIs, one shared producer; only sizing and the frustum policy differ, and both are deliberate.
- Kinematic `--walk --record-keyframes` is unwired and fails loud; use `--dynamic`.
- `--jobs` is a GPU-memory budget, not a core count -- each worker holds its own cuRobo world + EGL context. 4090 with ~3.3 GB baseline usage handled `--jobs 3` cleanly; raise cautiously and watch `nvidia-smi`.

### Visible-reachable candidate ordering

The plan-0 reachability ladder keeps its legacy arm-target candidates, then ranks them by the offline visible-reachable field's worst per-target `D_reach` neighborhood. The score is evaluated at the actual emitted grasp-goalset position, including the grasp offset, and selects the largest minimum margin across assigned arms. This prefers a target that remains visible and reachable under a small base-pose error; it never fabricates a reach result.

Build the cached field once per embodiment before using this preference:

```bash
PY=python
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/visible_reachable_curator.py --robot v2
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/visible_reachable_curator.py --robot v2_fixed
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/visible_reachable_curator.py --robot g1
```

Rebuilding the underlying workspace payload (not just the cached field) is a multi-GPU shard solve — ~20 min of compute that repeatedly costs hours of wall time when preflight is assumed. Read `mj_envs/asset_zoo/reachability_study/readme_reachability.md` § **Fast shard-solve SOP** first; it is the canonical procedure. Cheapest wins, in the order they bite:

1. Skip the ~half of the grid that lies outside the arm's reach sphere (`_reach_sphere`); it is provably unsolvable, so not dispatching it is exact, not an approximation.
2. Check GPU **utilization**, not free VRAM. A training job holds 5 GB but eats 55% of the card, and the rig finishes only when its slowest chunk does.
3. Read `TOTAL_ROWS` from a 64-row probe payload — never recompute the grid arithmetic.
4. rsync **every** host, then `grep` the remote file for your edit. A host on stale code solves a different grid and every post-merge assert still passes.
5. Chunk count == usable GPU count. Two chunks on one GPU doubles the whole run's wall time, and a double-launched chunk id does the same silently (the launcher only skips *finished* outputs).
6. `pkill -f <pat>` suicides over ssh; bracket a character (`generate_workspace[_]curobo`).
7. Expect ~1 chunk per rig to die on `CUDA error: an illegal memory access`; relaunch that pair.

Live cuRobo collision/IK remains authoritative, and camera perception plus physical latch remains the success criterion. Missing sidecar, zero-score tie, or `VISIBLE_REACHABLE_ORDER=0` retains exact legacy ordering. Use the environment switch for A/B comparison:

```bash
VISIBLE_REACHABLE_ORDER=0 $PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2 --scenario left_right_close --camera --walk --dynamic --mpc --seed 42
```

### Field-derived target stance selection (opt-in A/B)

This treatment uses the visible-reachable field to choose **where the base walks**, not to declare a grasp successful. `VISIBLE_REACHABLE_TARGET_STANCE=1` preserves legacy Point-C on the first visit. Only after that target completes a closed-jaw attempt without physical latch, its one existing bounded retry may sample high-margin field points `p_b`, inverse-map each at current yaw, and keep a stance

$$b_{xy} = g_{xy} - R_{yaw} p_{b,xy}$$

that stays outside support clearance, inside support width, and is reachable through declared locomotion vectors. The current score transforms cube grasp point from world into the live root frame (`base_link` for V2/V2-fixed, `pelvis` for G1), including roll/pitch; the planned target frame remains yaw-only because cuRobo tracking cannot consume transient locomotion tilt. A gimballed head or opposed fixed cameras retains current yaw, using a dominant configured drive vector plus bounded cross-track adjustment (`1.5` ratio for V2, `0.6` for V2-fixed). V2 therefore preserves its 360-degree gaze advantage, while V2-fixed preserves front/rear camera coverage instead of adding a needless turn. G1 has one forward camera, so it samples a camera-compatible final yaw. Robustness conservatively covers the 5 cm physical margin on the 2 cm lattice (three-voxel envelope). Full collision-aware cuRobo plan-0 revalidates actual pose. A candidate must pass fast collision-blind cuRobo IK before walking. Physical latch remains dynamic success condition.

Thresholds are `v2=0.25`, `v2_fixed=0.20`, and `g1=0.10`, calibrated by `calibrate_tau.py` against fast cuRobo IK (V2 prior N=120; V2-fixed/G1 N=48 seed 17). Treatment is off by default for causal A/B isolation:

```bash
# Retry treatment: field stance only after legacy Point-C misses its physical latch.
VISIBLE_REACHABLE_TARGET_STANCE=1 $PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2 --scenario left_right_far --camera --walk --dynamic --mpc --seed 47

# Direct treatment: field stance for first visit. Isolates base-target selection; arm/grasp are unchanged.
VISIBLE_REACHABLE_TARGET_STANCE=direct $PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2 --scenario left_right_far --camera --walk --dynamic --mpc --seed 47

# Baseline A: legacy Point-C base target.
VISIBLE_REACHABLE_TARGET_STANCE=0 $PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2 --scenario left_right_far --camera --walk --dynamic --mpc --seed 47
```

`VISIBLE_REACHABLE_TARGET_TAU=<value>` overrides one threshold only for a deliberate calibration sweep. Do not enable `VISIBLE_REACHABLE_ORDER` or `VISIBLE_REACHABLE_STANCE` while comparing target-selection A/B; those are independent field effects.

Do not use a first-visit field score as a failure oracle. It is an offline static proxy, while Point-C is evaluated by live locomotion, full cuRobo collision planning, and physical latch. Retry mode constrains field use to a real capture miss. Direct mode is an explicit A/B treatment for first-visit target selection.

### Failure-triggered visible-reachable stance recovery

When Point-C reaches a bounded support stance but the live route is proven infeasible, walk recovery also scores `Point-C - 0.12 m`, `Point-C`, and `Point-C + 0.12 m` along that support's clearance contour. It moves only when one candidate has a strictly larger worst active-target visible-reachable margin. The move preserves support-normal clearance, support width, base heading, camera gating, and all success criteria. Missing/zero field follows legacy contour recovery.

Before a field-selected recovery walks, legacy collision-blind cuRobo IK compares its predicted stance: synchronous path calls `ik_feasible_route`; dynamic MPC queues equivalent `ik_feasible_scene` in its existing sole cuRobo worker. A rejected candidate falls back to legacy recovery. Full cuRobo trajectory/collision planning plus physical latch remains authority. Disable this recovery for legacy A/B with:

```bash
VISIBLE_REACHABLE_STANCE=0 $PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2 --scenario left_right_far --camera --walk --dynamic --mpc --seed 47
```

### Scenario Anchors & `far_dist` Configuration

Table and object locations are defined in `pickplace_scenarios.py`:
- `near`: near workbench edge at $0.20\text{ m}$ from robot base.
- `far`: parametrized via `far_dist` in `make_scenarios(far_dist=0.80)` (default $0.80\text{ m}$ near-edge displacement). Workbench table anchors and pick object markers shift together proportionally (`fb_far_x` and `lateral_far_y = far_dist + LATERAL_CUBE_EDGE_INSET_M`), keeping object-to-table-edge insets and surface distributions invariant when adjusting scenario distance.

### Startup: async plan-0 worker + nominal pre-warm

The sole cuRobo runtime is a spawned process (`SpawnedReachMpcWorker`, `curobo/scene.py`), eager-spawned BEFORE the parent env build so its cold session-build overlaps. plan-0 is **async**: the control loop SUBMITS the solve (`submit_plan0`) and POLLS each tick (`poll_plan0`), holding planning-home while the robot balances live — the loop never blocks on cuRobo (real-robot planner contract; the same async seam the post-grasp `goal="home"` return plan uses). `run_reach` steps physics every tick but does not advance the reach budget while `policy.plan0_pending`.

The worker **self-warms** the pose CUDA-graph at boot on a perception-free NOMINAL world (`_nominal_warm_scene_and_targets`: a stand-in table slab + reachable targets + filler boxes sizing the collision cache to `_WARM_CUBOID_COUNT=16`). This captures the one-time ~3.3 s IK+trajopt graph DURING the parent build/settle; every real plan-0 then REUSES the graph via a cheap `update_world` (~0.1 s). Reuse holds because the graph shape depends on robot cspace + goalset + cache size, not the real obstacle values (any scene with ≤ the first scene's cuboid count fits). Deployment-correct: on hardware the planner boots + warms on a generic world BEFORE perception delivers the real table/shelf/cube. Net time-to-first-reach: ~5.3 s → **~2.16 s** (2 s settle + ~0.1 s warm solve). Worker prints `[curobo] pose graph warmed …` and `[curobo] {goal} solve …s (warm graph reused)`.

**`--view` waits on `SpawnedReachMpcWorker.wait_warmed()` before opening the GL viewer (2026-07-23).** The one-time CUDA-graph capture above is a driver-global critical section with no cross-process safety contract; concurrent OpenGL driver activity in the parent (`NativeMujocoViewer.setup()` -> `mujoco.viewer.launch_passive`, shader JIT-compiling on first draw) racing the worker's capture segfaulted the parent intermittently (SIGSEGV, exit -11 -- confirmed via `journalctl -k`, no `NVRM`/Xid entries, so driver/interop-level, not a hardware fault). Single GPU, no MPS/MIG, so the fix is temporal, not isolation: `_reach_mpc_worker` sets an `mp.Event` (`warmed_event`) once the capture attempt finishes (success or failure -- capture happens either way), and `curobo_reach_verify.py`'s `--view` path blocks on `mpc_worker.wait_warmed()` right before constructing the viewer. No-op in the common case (env build + settle already ~covers the worker's cold warmup, so the event is usually already set by then). Does NOT guarantee zero crashes -- the desktop compositor's own independent GL activity on the same GPU is outside app control -- it only removes the biggest, most deterministic overlap.

**Do NOT shorten the 2 s settle to exploit the now-fast solve.** The old blocking 3.3 s solve incidentally back-filled settling via the `plan0_pending` hold; with the warm solve instant, cutting the settle makes the tracker start from a barely-settled base and reach err regresses (measured 0.052 m FAIL vs 0.008 m PASS). Warm and full settle go together.

### Mission FSM (`reach_policy.py`)

`ReachPolicy` runs ONE flat, centralized state machine for EVERY embodiment and BOTH modes (parked reach and `--walk`). State is written ONLY by `_transition`; each state's handler RETURNS its next state (so a guard and its transition never drift apart), dispatched through one `_handlers` table. States:

```
ACQUIRE       perception-first DECIDE hub (ALL robots, parked + walk): sweep the gimbals (actuated)
              or read the fixed frustum, then choose reach-in-place / walk / discover / fail
DISCOVER      body-search a hidden cube into the camera belief          (walk only)
APPROACH      locomote to point C (anchor shifted to the cube), pin the arm(s) once reachable (walk only)
EXTEND        run the reach dispatch (cuRobo route / tracker) onto the grasp
GRASP         track final object-relative arm target while jaws close: minimum 0.5 s stroke, then advance on
              completed physical latch; 1.0 s cap turns a missed latch into a retry
RETRACT_PLAN  poll the async grasp->home plan, holding the grasp
RETRACT       play the grasp->home return, jaws closed; on settle finish the target visit
TERMINATE     mission done -> explicit verdict (all cubes serviced, or FAIL(unseen/unreachable))
```

**ONE entry, capability-gated.** Both modes seed the pending pick set and enter `ACQUIRE`. The difference is capability, not path: `_can_turn`/`_can_drive` are `True` for `--walk`, `False` for parked. ACQUIRE runs the gaze + level gates, then DECIDES:

- a cube reachable from the current stance, AND this ACQUIRE entry was NOT a fresh hand-off from DISCOVER (`_from_discover`) → commit the reachable SET in place (bimanual when ≥2) → EXTEND. The `_from_discover` exclusion matters only for a single-facing robot (g1): DISCOVER's raw-anchor walk can land it inside the generous proximity gate before the base ever resolves point C, so an unconditional commit here would silently skip APPROACH's point-C correction and commit from wherever DISCOVER happened to stop.
- (**walk** only) a cube seen but out of reach, OR reachable yet freshly handed off by DISCOVER → APPROACH (drive to point C, not the anchor DISCOVER stopped at); a pending cube hidden → DISCOVER (drive to the next viewpoint) — both funnel back through ACQUIRE;
- (**parked**, caps off) nothing reachable from here → FAIL-loud (`_failed`, explicit verdict) → TERMINATE, since it cannot drive or turn to a better stance. Parked reaches its set in one visit (its EXTEND self-plans the visible scene via `_parked_poses`) and terminates; it never DISCOVER/APPROACH.

So "parked" is just "walk with `_can_drive`/`_can_turn` off" — no separate FSM. Read the current state via `policy.walk_phase`; `policy.is_reaching` is true across the four reach states.

- **APPROACH targets point C, not the shared anchor.** Every anchor (table/shelf edge, hand marker) is ONE point shared by every cube resting on it; walking straight to it leaves an off-anchor cube at a nonzero bearing. `_point_c(anchor_w, cube_w)` shifts the anchor along its own fixed world-XY tangent to the cube's live lateral position (clamped to the anchor's physical half-width when bounded — table/shelf edge; unclamped for a hand marker) and along the perpendicular to the cube's own depth — human-analogy: walk to a point on the desk edge ahead of the object, not the desk's own fixed corner. Anchor identity (`_approach_anchor_pos`/`_tangent`/`_half_w`) resolves once per target (`_nearest_anchor`); point C itself is recomputed every tick from the cached tangent so it tracks the live cube belief.
- **SPAWN-READY shortcut: APPROACH can commit on tick 0, no driving at all.** If the base is already within `geom_floor` of point C (and facing — always true, see `_faced` below) the instant APPROACH is entered, it latches and calls `_commit_reach` immediately instead of running the drive-in loop. Covers a true close spawn (e.g. `front_back_close`, ~0.25 m) so the visit doesn't waste a walk/back-off/square-up cycle to land back where it started. Gated on `geom_floor` (the SAME final-stop floor the normal drive-in below uses), not the wider `_REACH_RADIUS_M` reach envelope — see the "two competing thresholds" gotcha below for why that distinction matters.
- **Reachability is base-facing-INDEPENDENT; the base does NOT turn to face a cube for reach.** cuRobo proves reach from the actual stance in EXTEND — a lateral or rear cube is reachable in place (e.g. `left_right_close` grasps both side cubes bimanually with zero base turn). Camera visibility and locomotion are configured separately: `walk_visibility_dirs` gives fixed-camera target bearings, while `walk_drive_dirs` gives permitted target-directed translations. G1 uses `(+X,)` for both; v2-fixed uses `(+X,-X)` for both; actuated v2 has unrestricted gimbal visibility and `(+X,-X,+Y,-Y)` drive directions. The mover picks nearest permitted direction per target. Thus an exactly lateral v2 target commands `(vx,vy,wz)=(0,+0.6,0)` without a body turn. `_faced`/`_square_up_wz` ARE a separate commit-time re-square, but ONLY for a single-fixed-forward-camera robot (`_needs_reface`, g1): `not _needs_reface` short-circuits both to `True`/`0.0` for v2/v2_fixed (their all-around cameras make facing irrelevant to either reach or visibility). For g1, commit-time facing is the ONLY thing keeping an already-latched cube inside its one static frustum — see the Gotchas entry below.
- **Support clearance uses body geometry, not the selected drive vector.** A bounded table or shelf anchor supplies an outward face normal. `APPROACH` projects configured root-frame footprint corners onto that normal, requires projected extent plus 0.05 m of clearance, and starts braking one mover braking distance before that floor. Thus a V2 side approach uses its wider lateral body extent against a table face even though its preferred drive vector is `+Y` or `-Y`. Calibration rule: if a physical side approach stops too close, increase V2's lateral footprint half-width in `body_clearance_points_b` (current `0.20 m`, next conservative setting `0.25 m`). Do not add a separate velocity or facing-specific stop model for this.
- **Humanoids use one shared ready posture for cuRobo.** `_REACH_READY_ARM_JOINT_POS` is cuRobo's planning seed, reach-preparation, and post-grasp return home for V2 and V2-fixed, in both kinematic and dynamic harnesses. Walking keeps the trained locomotion policy's arm reference; overriding it with planning home destabilizes the floating-base gait. Do not add per-direction postures or inflate support clearance to compensate for an arm collision.
- **Walk-to-stop standoff tuning (`reach_policy.py`).** During `APPROACH`, `ReachPolicy` triggers `_mover.stop()` at `stop_dist = max(latch_dist - walk_appr, geom_floor)`. `geom_floor` is now hoisted to the top of `_handle_approach` and shared with the SPAWN-READY shortcut (see next gotcha) — single-camera/G1 (`_needs_reface = True`) uses `geom_floor = 0.35 m` and `walk_appr = 0.00 m` (vs `_GEOMETRIC_FLOOR_M = 0.20 m` and `_WALK_APPROACH_M = 0.15 m` for humanoid `v2`/`v2_fixed`), ensuring G1 lands at a safe $\sim 0.28-0.30\text{ m}$ final stance from the table.
- **Two competing "close enough to stop" thresholds used to silently disagree (fixed 2026-07-22).** `_handle_discover`'s walk-in brake (line ~1268) stops at the reachability-envelope radius, `_REACH_RADIUS_M = 0.35 m` — needed so a fixed forward camera frames the cube before handing off. `_handle_approach`'s SPAWN-READY shortcut (line ~1317) used to gate on that SAME `_REACH_RADIUS_M`, so it fired on ~every DISCOVER hand-off (base already sitting right at that radius) and committed on tick 0 of APPROACH — the NORMAL-approach walk-in below, and the `_GEOMETRIC_FLOOR_M`/`_WALK_APPROACH_M` constants that control it, were **dead code** for any scenario needing a DISCOVER phase first (i.e. most of them). Symptom: v2 visibly stopped far from the target no matter how tight `_GEOMETRIC_FLOOR_M` was tuned — headless grading still PASSed (reach_err only measures grasp accuracy, not stopping distance, since the arm compensates by extending further regardless of stance) so the regression was invisible to the automated verdict. Fix: gate the SPAWN-READY shortcut on `geom_floor` instead of `_REACH_RADIUS_M`, so it only fires for a TRUE close spawn; a DISCOVER hand-off now falls through into the existing NORMAL-approach drive-in, which already latches at `_REACH_RADIUS_M` and correctly closes the gap to `geom_floor`. Verified PASS + reach_err ~10x tighter on `left_right_far`/`front_far_discover` (DISCOVER-hand-off cases) and no regression on `front_back_close`/`left_right_close` (near-spawn cases) or g1 (its `geom_floor == 0.35 == _REACH_RADIUS_M`, so the shortcut condition is numerically unchanged for it).
- **`--camera` = visibility gate.** Without it the belief is every cube (privileged GT); with it a cube must be SEEN before the robot reaches for it. Perception is POLICY-owned (seam inversion): the harness reports only what the cams frame at the CURRENT realized gaze (`detect_at_gaze`, pure sensor), and the policy accrues it into a persistent `_belief`. Fixed-camera robots (g1, v2_fixed) use a static frustum. Actuated gimbals (v2) run the policy's `_GazeController` (`reach_policy.py`): it sweeps the known support anchors at a physically realistic slew, a cube enters belief only once the frustum frames it at the CURRENT gaze (honest search, never an all-anchor snapshot), then TRACKS the found cube(s). ACQUIRE holds the base still through a complete support pass, or until every pending cube is seen; later tracking does not delay its decision. The scan remains bounded by `_ACQUIRE_LOCK_TIMEOUT_TICKS`. The policy drives the physical gimbals through the `camera_ref` holder (kinematic writes qpos directly). In `--view` the baked FOV frustums (geom group 4) show automatically under `--camera` (no key press). The belief gate itself (`cube_visible_poses` / `_cube_visible_from_camera`, `curobo/scene.py`) tests against the sensor half-angles SHRUNK by `_VISIBILITY_MARGIN_RAD` (8 deg, same for every robot) — a cube must clear that margin, not just the raw `H_HALF`/`V_HALF` frustum, to enter belief/commit. The true `H_HALF`/`V_HALF` (unshrunk) still drive the drawn frustum overlay, training/reward FOV, and `gaze_sees` (the actuated-gimbal reachability pre-check) — only the reach-mission belief seam is narrowed. See the Gotchas entry below.
- **No pre-reach settle gate.** `ACQUIRE` makes its decision as soon as the visibility sweep reports enough support coverage, and `EXTEND` plans from the current live base pose on its first tick. The robot is already balanced before reaching; waiting for world uprightness neither corrects reach-time sway nor improves the route. Camera sweep remains because it actively acquires visibility, not because it waits.
- **Latch-driven close.** `GRASP` keeps the object-relative tracker active through the 0.5 s jaw stroke. A completed contact-triggered equality latch advances immediately; a missing latch gets at most another 0.5 s, then follows existing retry/failure flow. The separate IMU gravity-change gate before close stays event-driven and capped at 1.0 s, because it observes reach-time sway rather than world uprightness.
- **Occupied-arm guard (`_held`).** An arm holding a cube (jaws closed since GRASP) is never reassigned to a new grasp: the arm pick and `_reachable_cubes` exclude occupied arms and held cubes. Cleared by `release_grasp()` / `reset`. This is the seam pick-and-place will extend (add `CARRY/PLACE/RELEASE` states + a place-pose source to the same enum + dispatch; not implemented yet).
- **Nav-target debug visualization.** `policy.nav_target_w` exposes the live DISCOVER/APPROACH drive target (world XYZ — the anchor or point C); `--view` draws it as one directional marker, position = target, orientation = base→target bearing: a single `mjGEOM_ARROW` in the kinematic viewer (`draw_nav_arrow`, straight `user_scn`), an `add_frame` triad (X axis = bearing, Y/Z dimmed) in the dynamic/mjlab viewer (`update_visualizers`) since it has no raw `user_scn`. Both in `curobo_reach_harness.py`. Yellow = DISCOVER (raw anchor), cyan = APPROACH (point C).

### Actuated head-gimbal aim (v2 only; `reach_policy.py` `_GazeController`, `curobo/scene.py`)

v2's head carries two 2-DOF yaw/pitch camera gimbals (`cam_yaw_left`/`cam_pitch_left`, `cam_yaw_right`/`cam_pitch_right`; ROM `+-4.7124 rad` / `+-1.5708 rad`, `policies._GAZE_YAW_ROM`/ `_GAZE_PITCH_ROM`). `_GazeController` (owned by the policy, not the harness — see the `--camera` bullet above) decides where they point every tick; a fixed-camera robot (g1, v2_fixed) has none of this, its frustum is static.

**Per-camera IK is closed-form and unique.** `control_vec.gaze_aim` inverts the gimbal's own forward kinematics `forward(yaw,pitch) = (sign*cos(yaw)*cos(pitch), -sign*sin(yaw)*cos(pitch), -sin(pitch))`, where `sign` is `+1` for the left cam (zero-yaw datum faces forward) and `-1` for the right cam (zero-yaw datum faces BACKWARD — its mount is rotated 180 deg). Given a base-frame look direction `d = target - cam_pos`: `pitch = asin(-d_z)`, `yaw = atan2(-sign*d_y, sign*d_x)`. One direction maps to exactly one `(yaw, pitch)` in-ROM; there is no second solution to pick between.

**Bijective cam<->cube assignment (`aim_gaze_at_cubes`).** With 2 gimbals and up to 2 tracked cubes, which cube each cam looks at is a genuine choice, not derivable from the IK alone (the right cam's backward datum means a cube roughly in FRONT of the robot needs yaw near +-pi from it, but near 0 from the left cam — assignment matters). The function evaluates every injective cam->cube map (`itertools.permutations`) and scores each by, in order:
1. `-covered` — cover as many cubes as possible (a cam with no in-ROM cube in the map is left at rest).
2. `cross` — count of pairings where the cam and cube are on OPPOSITE lateral (base-Y) sides; prefers same-side pairings, which also avoids a cross-body reach that self-occludes.
3. `total_yaw` — sum of `|yaw|` across the map; the well-separated, primary efficiency signal.
4. `-worst_margin` — tie-break only: `margin` is the fractional in-ROM headroom on whichever axis (yaw or pitch) is closer to ITS limit, normalized to `[0,1]` so both axes compare on one scale. Maximizes the worst-case headroom across the map, but only among assignments already tied on `total_yaw` — whichever axis is the binding constraint (often pitch) is close to assignment-invariant, so ranking margin ahead of `total_yaw` lets sub-ULP float noise decide between assignments that actually differ hugely in quality. `cam_lat` (which lateral side a cam nominally serves, for the `cross` test) is read off the yaw joint's `xanchor` — the joint's own pivot point, fixed regardless of that joint's current angle — not the lens `site_xpos`, which physically swings across body-center as commanded yaw grows (the pivot-to-lens offset is baked into the URDF, `cam_pitch_left` mounted `-0.105 m` off its yaw joint).

**Sweep vs. track.** Before every pending cube has entered belief, `_GazeController` SWEEPs the known static support anchors (table/shelf edges) in bearing order (`known_anchors`), one at a time if there are more anchors than cams, or all concurrently via the same bijective `aim_gaze_at_cubes` (targeting anchor points instead of cube centers) when cams >= anchors. Once every pending cube is seen (`full_pass`), it switches to TRACKing the live cube centers via the same assignment function. `full_pass` is the perception proof the mission FSM's ACQUIRE state waits on (see `--camera` bullet).

**Slew rate limiting.** The IK/assignment above are solved fresh every tick (no persistent state); `self.cmd` (the actually-commanded joint angle) approaches that fresh `desired` at a bounded rate (`_GAZE_SLEW_MAX = 3.5 rad/s`, a hardware-realistic gimbal speed), not instantly — `cmd += clip(delta, -step, step)`. `delta` is `desired - cmd` WRAPPED to `(-pi, pi]` (`_wrap_pi`) before clamping, not the raw difference: since the right cam's natural operating point sits right on atan2's `+-pi` branch cut, an unwrapped delta can misread "same direction, opposite float rounding" as a huge rotation.

### Shared physical pick scene

Every `scenario.pick` cube is a collidable MuJoCo free body (`pp_obj_<cube>` / `pp_free_<cube>`) resting on its configured table (near edge at 0.20 m, cube at 0.28 m) or shelf (`PLACE_SHELF_X_M = 0.40` m, front edge 0.26 m clears torso). It is a 60 mm, 30 g block with friction `(sliding, torsional, rolling) = (1.2, 0.01, 0.001)`; workbench collision supports explicitly use `(0.9, 0.01, 0.001)`. Gripper jaw pads use `(1.5, 0.005, 0.0001)`. Kinematic and dynamic harnesses compile this exact same scene: kinematic reset writes sampled cube poses into free-joint `qpos0` and does not integrate them; dynamic reset starts from those same poses, then frozen RL plus MuJoCo contact owns cube motion. Planner/perception read live `mjData.geom_xpos`, never model-local geom positions. Every nonterminal reach, parked or `--walk`, runs the `EXTEND -> GRASP -> RETRACT_PLAN -> RETRACT` states of the mission FSM. A final completed two-target grasp terminates after the same closed-jaw hold, without queuing a home plan that the evaluator would cancel. Position-controlled jaws move at `PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S = 0.04` m/s (40 mm/s) to the 19 mm object-sized grasp target. From the -10 mm pregrasp opening, this 29 mm stroke takes 0.745 s. A 60 mm cube first contacts at 4.95 mm, so the command retains 14.05 mm per-side preload. In dynamics, a named equality latch (`pp_weld_{L|R}_<cube>`) activates only after real finger/rack contact and grasp-center distance below 0.03 m, then eases cube to the designed grasp center over 0.15 s. Contact against wrist or forearm cannot latch. Completed latch advances the FSM immediately after the full stroke; a missing latch has a 1.0 s total close cap before existing retry flow. Completed latch, not a 5 cm center-error threshold, is dynamic grasp success. Dynamic `--mpc` freezes final Cartesian grasp target in current base frame and keeps tracker active through close plus async planning. Fresh measured-qpos `goal="home"` route follows the same tracker through a 0.30 s, <=0.05 m/s Cartesian bridge; reverse reach is fallback only when planning fails. A later place controller calls `release_grasp()` to open; reset opens for a new episode.

### Parallel Gripper Open & Grasp Configuration

The open and grasp rack positions for the parallel gripper are defined in a **Single Source of Truth (SSOT)**:

```python
# mj_envs/asset_zoo/parallel_gripper.py
PARALLEL_GRIPPER_OPEN_RACK_POS_M: float = -0.010    # Open rack target (m); -10 mm per rack = 20 mm wider total jaw gap
PARALLEL_GRIPPER_GRASP_RACK_POS_M: float = 0.019    # Fixed 60 mm task-cube grasp target
PARALLEL_GRIPPER_CLOSE_STROKE_M: float = PARALLEL_GRIPPER_GRASP_RACK_POS_M - PARALLEL_GRIPPER_OPEN_RACK_POS_M
PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S: float = 0.03
```

#### How to Change the Open Jaw Width

To adjust the resting open spacing across **both cuRobo and simulation**, edit `PARALLEL_GRIPPER_OPEN_RACK_POS_M` in `mj_envs/asset_zoo/parallel_gripper.py`:

- **`q = 0.000 m`**: Natural XML rest pose (~70 mm inner pad spacing).
- **`q = -0.010 m`**: Retracts each rack outward by 10 mm per side, providing a **20 mm wider total opening gap** (~90 mm inner pad spacing).
- **`q = -0.015 m`**: Retracts each rack outward by 15 mm, providing a **30 mm wider total opening gap** (~100 mm inner pad spacing).

#### Automated Propagation Across the Pipeline

You do **not** need to modify cuRobo configs or simulation harnesses separately. Changing `PARALLEL_GRIPPER_OPEN_RACK_POS_M` automatically updates:

1. **cuRobo Planner (`curobo/planner.py`)**: `PARALLEL_GRIPPER_RACK_HOME` maps all 4 rack joints (`L_left_rack_y`, `L_right_rack_y`, `R_left_rack_y`, `R_right_rack_y`) to `PARALLEL_GRIPPER_OPEN_RACK_POS_M` in `home_joint_pos` and `planning_home_joint_pos`. cuRobo's `lock_joints` bakes this exact jaw geometry into its fixed transforms and self-collision models.
2. **Dynamic Harness (`curobo_reach_harness.py`)**: `_gripper_open` uses `PARALLEL_GRIPPER_OPEN_RACK_POS_M`; `_gripper_grasp = _gripper_open + PARALLEL_GRIPPER_CLOSE_STROKE_M`, which always resolves to the fixed `PARALLEL_GRIPPER_GRASP_RACK_POS_M`. The FSM close dwell derives from the same stroke and `PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S`, capped at 1 s, so contact cannot start retraction before the jaws finish closing.
3. **Standalone Environment (`pickplace_reach_env.py`)**: `gripper_home_target` initializes directly from `PARALLEL_GRIPPER_OPEN_RACK_POS_M`.

### Arm Home Posture Configuration

The arm home posture seeds the cuRobo cspace retract seed and planned route pre-positions. It is defined per robot:

- **humanoid (`v2` / `v2_fixed`)**: `_REACH_READY_ARM_JOINT_POS` (aliased as `HUMANOID_ARM_JOINT_HOME`) in `mj_envs/tasks/visual_manipulation/curobo/ik_curobo_robot_cfg.py`.
- **Unitree G1**: `G1_ARM_HOME` in `mj_envs/tasks/visual_manipulation/curobo/g1_curobo_robot_cfg.py`.

#### How to Change the Arm Home Pose for `v2` / `v2_fixed`

Edit `_REACH_READY_ARM_JOINT_POS` in `mj_envs/tasks/visual_manipulation/curobo/ik_curobo_robot_cfg.py`:

```python
_REACH_READY_ARM_JOINT_POS = {
    "right_shoulder_1_joint": 0.7297, "left_shoulder_1_joint": -0.7297,
    "right_shoulder_2_joint": 0.1926, "left_shoulder_2_joint": -0.1926,
    "right_shoulder_3_joint": -0.3546, "left_shoulder_3_joint": 0.3546,
    "right_elbow_joint": -1.6927, "left_elbow_joint": 1.6927,
    "right_wrist_1_joint": 1.3838, "left_wrist_1_joint": -1.3838,
    "right_wrist_2_joint": 1.5182, "left_wrist_2_joint": -1.5182,
    "right_wrist_3_joint": 1.5475, "left_wrist_3_joint": -1.5475,
}
```

To re-solve or generate a new symmetric 2-arm home pose using Mink QP IK:
```bash
$PY mj_envs/asset_zoo/humanoid_v21/generate_arm_joint_pos.py   # humanoid v2
$PY mj_envs/asset_zoo/g1/generate_arm_joint_pos.py             # G1
```

Changing `_REACH_READY_ARM_JOINT_POS` automatically propagates to `curobo/planner.py` (`HUMANOID_ARM_JOINT_HOME`), `ik_curobo_robot_cfg.py` (`build_robot_cfg_dict_from_urdf`), and `pickplace_reach_env.py`. Afterwards, verify reaching with:

```bash
$PY mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2
```

### Input/output flow

```
scene + observed cube poses + settled robot base pose
        │
        ▼
cuRobo plan-0 worker
  input: table/object collision world, 14 arm DOFs, grasp goal set
  output: collision-free whole-arm route q_plan[waypoint, joint]
        │
        ▼
JacobianReachTracker, each physics control tick
  input: route, live cube poses in base_link, measured arm qpos + tool-site poses in base_link
  feedback: Cartesian position integrator I_pos (per active tool frame)
  output: q_cmd {bare arm joint name: MuJoCo qpos target}
        │
        ├──► arm_ref passive command ──► frozen RL observation
        │
        └──► direct_arm position action ──► physical arm actuators
                                              ▲
frozen RL policy ──► visits/base actions ──────┘
```

`arm_ref` and `direct_arm` receive identical `q_cmd`, but serve different jobs:

- `arm_ref` is input to frozen RL. RL sees intended arm pose in its normal observation.
- `direct_arm` is physical target. It is applied **after** RL's `joint_pos_arms` residual, so RL continues balancing visits/base but cannot move planned arm target.
- `direct_arm` target-name order is asserted equal to `arm_ref`; route joint names are resolved into each robot's MuJoCo qpos/DOF addresses. Same wiring serves `v2`, `v2_fixed`, and `g1`.
- Parallel-gripper racks use separate direct position control. They are neither planning DOFs nor frozen-policy action/observation dimensions.

### Per-tick IK

Plan-0 first extracts every waypoint's gripper pose relative to planned cube pose. At tick `k`, tracker recomposes that offset using live cube pose in `base_link`; small base lean/drift therefore moves desired tool pose with cube instead of chasing stale world coordinates. Dynamic execution seeds DLS from measured arm qpos; kinematic replay has no measurement and seeds from interpolated `q_plan[k]`. Thus live position and orientation error enter one instantaneous 6D solve, with no orientation integral. It then runs bounded damped-least-squares MuJoCo Jacobian IK on active arm joints and preserves planned orientation target.

Tool feedback reads only live root and tool-site tensors, not full Warp `qpos` readback. (Camera-gated cube perception has its own separate full-qpos path.) A small leaky **position-only** Cartesian integrator offsets desired position, capped at 10 mm, to remove steady servo/compliance error:

```
e_pos = p_desired - p_tool_measured
I_pos = clip(0.995 * I_pos + 0.005 * e_pos, -0.010 m, +0.010 m)
p_ik  = p_desired + I_pos
```

`I_pos` persists across control ticks and resets only when plan-0 is rebound to a new route. Orientation has no integral term; prior orientation integration disturbed gripper alignment.

Before emitting target, tracker applies position-reference gravity pre-bend:

```
q_cmd = q_ik + alpha * qfrc_bias_arm(q_ik, gravity_in_live_base) / kp_arm
```

This is not torque control. MuJoCo's existing arm position servo still produces torque; pre-bend offsets its target by expected gravity droop. Gravity is rotated into live tilted base frame. `alpha=0` disables it. Collision safety remains plan-0's responsibility: tracker only makes bounded drift correction and does no per-tick collision solve.

### Relevant files

| File | Owns | Do not confuse with |
|---|---|---|
| `test/curobo_reach_verify.py` | CLI, dynamic harness creation, pre-plan settle, verdict | `--mpc` means plan-once + Jacobian tracker, not per-tick cuRobo MPC. |
| `curobo_reach_harness.py` | Dynamic MuJoCo/RL environment; compact measured arm/tool reads; `arm_ref` + `direct_arm` writes | `arm_ref` is RL observation; `direct_arm` is physical actuator target. |
| `reach_policy.py` | Plan-0 worker request; async submit/poll (`plan0_pending`); object poses in `base_link`; tracker lifecycle and gravity-frame conversion | It does not execute actuator control itself. |
| `jacobian_reach_tracker.py` | Route cursor, object-relative desired poses, measured-q seed, DLS IK, Cartesian I, gravity pre-bend | It has no collision rollout or global replan per tick. |
| `frozen_policy.py` | Nominal dynamic-eval mutation: removes DR/noise before building RL environment | Frozen checkpoint normalizer remains active; it is deterministic preprocessing. |
| `curobo/planner.py`, `curobo/scene.py` | Plan-0 goal set and collision world; spawned plan-0 worker + nominal CUDA-graph pre-warm | These establish safety margin; runtime tracker must stay bounded around route. |

### Tracking levers

Change one lever, use fixed `--seed`, compare final reach error, upright score, and viewer gripper orientation. Dynamic verdict currently has position error + upright only; it does **not** score orientation.

| Lever | Location / current value | Scope and expected effect | Status |
|---|---|---|---|
| Measured-q DLS seed | `JacobianReachTracker.step`; dynamic only | Starts 6D IK from live arm qpos rather than route q. Corrects instantaneous physical position/orientation error. | **Active.** |
| DLS damping | `_DLS_LAMBDA = 0.05` | Higher = more stable/less aggressive; lower = tighter but less conditioned near extension. | Internal; do not tune with other levers. |
| Per-iteration joint step cap | `_DQ_CLAMP = 0.05 rad`, `_IK_ITERS = 12` | Bounds instantaneous Jacobian correction. | Internal safety/stability lever. |
| Cartesian XYZ I | `KI=0.005`, decay `0.995`, cap `10 mm` | Removes steady physical tool-position error. No orientation state. | **Active.** Position-only by design; orientation integral previously mis-aligned gripper and is not added back. |
| Orientation or joint I | None | Would accumulate normal route lag and release as overshoot. | **Do not add.** Prior orientation feedback misaligned gripper. |
| Gravity pre-bend | `--grav-comp=True`, `--grav-alpha=1.0` | Adds `qfrc_bias/kp` to arm position target. Uses live base tilt and solved target arm pose. | **Active and at its measured optimum.** Sweep 0.25/0.5/0.75/1.0/1.25/1.5 on v2/left_right_close/seed42 is a clean U with the minimum AT 1.0 (median physical_pos_err 17.6/13.5/12.4/**11.4**/13.4/17.5 mm); do not retune. |
| Velocity feedforward | `VEL_FF_SCALE=1.0` | Adds `(kd/kp)·q̇_nom` to the command, where `q̇_nom` is the PLANNED route finite difference (no sensor noise). Cancels the velocity-proportional following lag that the gravity-only pre-bend leaves. Summed with the gravity bend BEFORE the `arm_bend_max` clip — both are torque demands on the same actuator. The gripper rack corridor is ~0.015 m/side, so error above that is what knocks the cube. | **Active.** Halved EXTEND tracking error and dropped ticks over the rack corridor by ~89%. v2/left_right_close median 11.2 → 4.9 mm (38 → 4 ticks over 15 mm); v2/front_back_close 33.2 → 30.0 mm (467 → 398, residual source unidentified); g1/left_right_close 9.7 → 4.9 mm (134 → 34). Set `VEL_FF_SCALE=0` to A/B. |
| Output trajectory filter | `--filter=False`, `--omega=15 rad/s` | Critically damped joint-target profile; lower omega slows more. | Off by default; adds lag and did not help this reach. |
| Pre-plan settle | `--settle-s=2.0` | Holds planning-home through physics before plan-0, avoiding plan from transient stance. | **Active default.** Do NOT shorten: with the warm solve now instant, a shorter settle starts the tracker from a barely-settled base and the recorded reach_err regresses from 0.008 m PASS to 0.052 m FAIL. Warm solve + full settle go together. |
| Route timing / cursor rate | Route `interpolation_dt`; `DYNAMIC_REACH_SPEED_SCALE` slows playback | Slower trajectory reduces command acceleration but keeps arm loaded longer; halving speed only reduced median error 25% (p90 / max unchanged) — so the binding error source is NOT pure velocity lag. | Rejected: speed scaling does not help on its own. |
| Per-tick cuRobo replan/MPC | N/A | Would add collision-aware optimization each tick. | Rejected: fixed ~16 ms tick cost and target-cube collision wedge. |
| Two-stage pre-grasp | `REACH_PREGRASP_STANDOFF_M`; default **0** (off) | Stage A to a goalset retreated along the approach axis; stage B a `ToolPoseCriteria.linear_motion` descent to the real grasp. Stage B measures clean (straightness 0.70–0.98, 0.0° off axis), but retreating the goalset picks a different IK branch whose max joint travel is ~2× the single-shot's for an easier target, which moves the knock earlier in the route. Still knocks after the velocity-FF fix. | **Rejected.** A worse plan is not rescued by better tracking. Off by default; re-enable only if stage A is pinned to the single-shot's winning candidate. |

For multiple sequential reaches, tracker `rebind(route, ...)` resets route cursor and Cartesian I. Grasp hold and return deliberately preserve both feedback states; only a new reach plan rebind resets them.

### Grasp-approach angle (`beta`)

Every grasp candidate is `rotz(yaw) . A(beta)` (`cube_grasp_poses_obj` / `approach_quat_tilt`, `ik_curobo_robot_cfg.py` / `g1_curobo_robot_cfg.py`): `yaw` picks one of the cube's 4 face-symmetric sides, `beta` is a single continuous tilt knob, **measured from straight-down vertical (0°) toward horizontal (90°)**:

```
beta=0   -> straight top-down
beta=60  -> 30 deg below horizontal (near-level, angled down)
beta=90  -> level side-grasp (horizontal), identity tilt -- the historical default
```

`beta_degs` (the goalset) can carry several values at once; cuRobo mins the cost over ALL yaw x beta x flip candidates per cube, so a stance unreachable at one tilt may still land at another.

| Robot | `beta_degs` | Where | Why |
|---|---|---|---|
| v2 / v2_fixed (humanoid) | `HUMANOID_GRASP_BETAS = (60.0, 45.0)` | `planner.py` | Union of near-level + steeper tilt, both clear the shared table slab. Replaces the earlier `(90.0, 60.0)` union (2026-07-22, user choice) -- see rejected levers below for why `90` and the `GRASP_Z_ABOVE_M=0` red herring are no longer in play. |
| g1 | `G1_GRASP_BETAS = (60.0, 45.0)` | `g1_curobo_robot_cfg.py` | Same union as humanoid (2026-07-22). `beta=90` (dead-level) still excluded -- rakes the parallel-gripper rack against the shared table slab for g1's shorter arm/lower shoulder (~10 cm hang). |

`flip` doubles each candidate with a 180 deg twist about the tool approach axis (top/down-handed grip); both robots call with `flip=False` (2 betas x 4 yaws = 8 already fills `_MAX_GOALSET`, no headroom for a flip doubling). Total candidates = `len(beta_degs) * 4 * (2 if flip else 1)`.

**Rejected/tested levers (do not re-run without a new premise):**
- Single global beta instead of a goalset (`beta_degs=(90.0,)` or `(60.0,)` alone for humanoid) -- 2026-07-15 A/B: `front_back_close` 0.900(β90)/1.000(β60), `bimanual_mixed_close` 0.800(β90)/0.700(β60) -- trades cells, no single value beats a union.
- `GRASP_Z_ABOVE_M = 0` (exact cube-center grasp target) -- 2026-07-22: IK-infeasible boundary, reproducibly 0% reach across every stance/arm-assignment on `front_back_close` regardless of beta. Looked like a beta-union defect (union scored WORSE than either single beta alone) but was purely the z-anchor sitting exactly on a kinematic singularity; `z_above=0.01` clears it. Do not re-diagnose a beta-union failure without first checking `GRASP_Z_ABOVE_M` is off zero.
- A variable/adaptive beta via one softened-weight solve (command 90, let the optimizer relax toward reachable) -- 2026-07-10, does not work with cuRobo (see `plan/CUROBO_HANDOFF.md`). If an adaptive tilt is wanted, the correct shape is a DESCENDING-beta fallback ladder (try 90, step down until IK-feasible; one-shot, ~+50 ms/grasp), not a solver weight trick.

### Nominal dynamic evaluation

`--dynamic` reach construction uses nominal evaluation configuration for all `v2`, `v2_fixed`, and `g1`: no observation noise, no IMU corruption, no reset/startup/interval domain-randomization events, and no randomized action term. Only deterministic base/joint reset mechanics remain. Frozen checkpoint normalizer still runs; it is fixed preprocessing, not noise.

Do not cite an aggregate unless its manifest, pinned checkpoints, complete cell set, and clock/decomposition audits all pass.

The frozen RL locomotion policies are resolved per robot via `_DYNAMIC_TASK` in `curobo_reach_harness.py`:
- `v2` / `v2_fixed`: `v2_best` (alias for `HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam`)
- `g1`: `g1_best` (alias for `G1RmaVelEstArmFlashSacStudentOnlyg1u1`)

`find_latest_checkpoint(task)` in `run.py` dynamically loads the latest trained `model_*.pt` checkpoint file under `runs/<task_name>/`. For deploy aliases like `v2_best` and `g1_best` (which set `fallback_checkpoint_to_parent = True`), `find_latest_checkpoint` automatically walks up the experiment class inheritance tree to locate the latest trained checkpoint under the parent run directory.

## 5. Inspect the collision model

```bash
$PY mj_envs/tasks/visual_manipulation/test/curobo_g1_view_collision_spheres.py       # G1
$PY mj_envs/tasks/visual_manipulation/test/curobo_view_collision_spheres.py           # humanoid
# --table also compiles the scenario tables for obstacle context
```

Overlays the cuRobo collision spheres on the resolved mesh at the standing home pose, colored by region (torso/waist cyan, legs/feet blue, arms red, gripper racks magenta, head yellow). The viewer drives the mesh's locked joints to the cfg's `lock_joints` values, so spheres == what the planner actually collides against. Needs a DISPLAY.

---

## Gotchas

- **Locked gripper is baked, not live.** Locked-joint values fold into cuRobo `fixed_transforms` at build. Moving the actuator at runtime does NOT inform the planner. To plan against a different jaw state, rebuild the cfg with a new lock value (~70 ms/config) then `Kinematics.update_kinematics_config(new_cfg)` (0.2 ms in-place `copy_`, preserves the CUDA graph). Both robots currently plan at the `qpos0` rest jaw state — no runtime open/close.
- **Head camera is visual only** for planning. G1 uses `builtin` (original head mesh, no collision spheres added); humanoid uses `actuated` (its sphere fit includes the camera body). Match whatever the cfg module loads.
- **Tool-frame order `[left, right]`** MUST match `set_bimanual_goalset`'s `[L, R]` order or each hand chases the wrong cube.
- **Excluded leg subtrees** (below each knee) are pruned from BOTH the collision spheres AND any `lock_joint` whose child body was excluded — else the cuRobo loader KeyErrors.
- **Any hand-rolled `DynamicHarness` stepping path must call `_update_weld_latch(gripper_closed)` itself — it is NOT automatic.** `DynamicHarness.realize()` (the canonical per-tick step, used by headless grading) calls it every tick; a custom loop that steps the harness directly without going through `realize()` (e.g. `curobo_reach_verify.py`'s `_run_view`'s `viewer_policy`, since the mjlab viewer owns `env.step`) will silently skip it. Without the weld latch, a grasp that lands within kinematic tolerance still relies on raw gripper-pad friction alone once TCP-to-cube dist < 0.06 m and real geom contact is detected — it activates a MuJoCo equality constraint welding hand-to-cube; skip it and the cube can look empty/missed under `--view` even though the identical mission PASSes headless (found 2026-07-22: headless grading was clean, `--view` looked broken — root cause was this missing call in `viewer_policy`, now fixed).
- **A geometrically-feasible bimanual pairing can spuriously fail at the planner's default 10 solve attempts (found/fixed 2026-07-24).** `_first_feasible_assignment` (`curobo/scene.py`) tries both L/R pairings bimanual-first, then falls back to single-arm; a bimanual solve has 2x the DOF plus an inter-arm collision constraint vs. a lone arm, so it needs more attempts to converge. Repro: `curobo_reach_verify.py --scenario left_right_close --robot v2 --camera --walk --seed 42` served both cubes as two sequential single-arm visits instead of one bimanual reach, even though the pairing is feasible — verified deterministic: the same pairing failed at 10/12/18 attempts, succeeded at 15/20/25/30. Fix: `_BIMANUAL_PLAN0_MIN_ATTEMPTS = 30` floors the bimanual try's attempt count on plan-0 only (`max_attempts is None`); an explicit small replan cap (e.g. `1`) is untouched. Attempts is a *retry* knob (session-agnostic, cheap to scope); `num_trajopt_seeds` is the PARALLEL search over the same failure, baked into the ONE warmed session at construction (CUDA-graph-captured). Verified: repro lands one route `{'L': 'cube_0', 'R': 'cube_1'}` in a single visit; `--selftest` still all-PASS.
  - **Superseded 2026-08-02 — the attempts-over-seeds argument this entry used to make is wrong on the dynamic path.** It rejected seeds because they "tax *every* solve for the process lifetime" (measured then: 4→8 = +35% per single-arm replan, 4→12 = +84%). That per-solve tax is real but is smaller than the retry loop it removes, and 30 attempts turned out not to be enough anyway once physics (rather than the kinematic harness) chooses the stance: the base parks ~14° further off square, which pushes the pairing to its basin edge. At 4 seeds BOTH pairings failed at 30 attempts on v2 `bimanual_mixed_front_back_close` seed 47 and the visit silently degraded to two single-arm reaches — that mission FAILED on a knock. Aggregate `plan_wait_s`, v2 seed 47: `bimanual_mixed_front_back_close` 4 seeds/150 attempts 1.64 s → **8 seeds/30 attempts 0.44 s**; `front_back_far` (single-arm heavy, no bimanual visit at all) 4/30 1.30 s → **8/30 0.58 s**. The marginal pairing itself lands in 0.30 s at 8 seeds vs 2.17 s of serial retry at 4. Shipped `_NUM_TRAJOPT_SEEDS = 8` (`ik_curobo_robot_cfg.py`), attempts left at 30. **8 is a shallow optimum, not a "more is better" knob** — 16 seeds is worse than 8 (0.53 s on the same pairing).
- **`--view --mpc` intermittently segfaulted (exit -11) from a GL/CUDA-graph-capture race (found/fixed 2026-07-23).** The eager-spawned worker's one-time CUDA-graph capture (`scene.py`, see "Startup" above) overlapped `NativeMujocoViewer.setup()`'s OpenGL context creation by design (both were racing to overlap the ~5-7 s startup). Confirmed via `journalctl -k`: crash sites drifted across runs (`.glXXXXXX (deleted)` JIT shader lib, `libX11.so`, bare `python3.12` heap corruption) with no `NVRM`/Xid entries — classic signature of a driver-level race, not a fixed app bug. Fixed by `SpawnedReachMpcWorker.wait_warmed()`, called right before the viewer is constructed — see "Startup" section for detail. Single GPU, no MPS/MIG on this machine, so no hardware isolation option existed; the fix is temporal sequencing only.
- **g1 could commit a grasp outside its own camera frustum under `--camera` (found/fixed 2026-07-25).** `_faced`/`_square_up_wz` (`reach_policy.py`) had been stubbed to unconditional `True`/`0.0` for every robot, reasoning reachability alone (cuRobo proves reach from the actual stance, base-facing- independent) — but that silently dropped a SEPARATE, still-valid concern the 2026-07-21 fix had correctly scoped to g1 only: g1's single fixed forward camera has no gimbal, so commit-time facing is the ONLY thing keeping an already-latched cube inside its frustum (v2/v2_fixed's all-around cameras never needed it). Live repro: `--robot g1 --scenario front_back_close --camera --walk --seed 47 --dynamic --mpc --view` visibly committed a grasp with the target outside the drawn FOV cone (group 4). Fix: restored the pre-stub, `_needs_reface`-conditional implementations (`_REACH_FACE_TOL_RAD = 3 deg`, proportional steer `_REACH_STEER_K/_REACH_TURN_WZ_MIN/MAX`) — g1 square-up before commit when off by more than 3 deg; v2/v2_fixed unaffected (`not _needs_reface` short-circuits both methods, byte-identical to the stubbed behavior for them). Verified g1 + v2_fixed both re-PASS `front_back_close` seed 47 headless (`--camera --walk --dynamic --mpc`) after the revert.
- **A cube right at the FOV boundary could still leave view mid-grasp (found/fixed 2026-07-25).** Even with the square-up above, a cube that only just clears the TRUE sensor half-angle (`H_HALF`/`V_HALF`) at commit time is not a safe commit: standing/reach sway is large at close range (documented elsewhere in this FSM — a few cm of g1 sway subtends ~8 deg at a 0.29 m target), so a border-line cube can drift outside the real frustum during EXTEND/GRASP — physically, a real single-camera robot cannot complete a grasp on something it can no longer see. Live repro: `--robot g1 --scenario bimanual_mixed_close --camera --walk --seed 47 --dynamic --mpc --view`. Fix: `_cube_visible_from_camera` (`curobo/scene.py`) now gates belief on `H_HALF/V_HALF` shrunk by `_VISIBILITY_MARGIN_RAD = 8 deg` (`_COMMIT_H_HALF`/ `_COMMIT_V_HALF`) — same margin for every robot, since the sway that motivated it is a physical-stance property, not sensor-specific. Only this ONE belief seam (`cube_visible_poses`, feeding every ACQUIRE/DISCOVER/APPROACH/commit decision under `--camera`) is narrowed; the true `H_HALF`/`V_HALF` still drive the drawn frustum overlay, training/reward FOV, and `gaze_sees` (the actuated-gimbal reachability pre-check) unchanged. Verified g1 re-PASSes `bimanual_mixed_close` and `front_back_close` seed 47 headless (`--camera --walk --dynamic --mpc`) at the 8 deg margin.
