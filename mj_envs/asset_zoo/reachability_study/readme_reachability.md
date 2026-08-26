# Reachability Study

Per-voxel reachability and visible-reachable-workspace (VRW) computation for humanoid arms, used to (1) generate the paper's cross-platform workspace comparison figure and (2) select the number of head cameras (K=1/2/3) via a camera-count ablation. Ground truth is the code in this directory; this document is the why and the run recipe.

**Contents:** [Definitions](#definitions) · [Package layout](#package-layout) · [Setup](#setup) · [Quickstart](#quickstart-one-robot-one-gpu) · [Scaling to production](#scaling-to-a-production-payload) · [Camera-count study](#camera-count-study-k--1-2-3) · [Cross-robot comparison](#cross-robot-comparison) · [Adding a new robot](#adding-a-new-robot) · [Modeling choices](#modeling-choices) · [Goal-point curator](#goal-point-curator)

## Definitions

For each robot, a batched cuRobo IK solve is run over a regular 3D voxel lattice (isotropic 0.02 m cubic voxels) crossed with a fixed set of `N` (usually 64) target orientations per voxel:

```text
D_reach(v)   = count(success[v, :]) / N
D_visible(v) = count(success[v, :] AND visible_any[v, :]) / N
```

- A voxel is **reachable** if IK succeeds at some orientation. `W_R` is the volume of the reachable set (`reachable_voxels * spacing^3`), in m³.
- A reachable voxel is **visible** if some camera can be aimed at it with clear line of sight. **η** ("eta") is the fraction of reachable voxels that are visible with gimbals free to steer; **η_fix** is the same fraction with gimbals locked at rest. `visible_any` is OR over physical eyes: never call a voxel visible because its center is seen in one orientation/state only.
- **W_VR = η · W_R** is the visible-reachable volume in m³, the design metric: what the robot can both put its hand in and look at. `W_VR^fix = η_fix · W_R` is the locked-gimbal counterpart.
- **η₂** (pairwise coverage) samples a manipulation point (reachable, dexterity-weighted) and a second point anywhere in the workspace box, and asks whether the cameras can cover both at once: "manipulate here while a camera watches there." It needs per-eye visibility masks, not just the OR-collapsed single-target mask.
- **Occlusion policy:** a target is visible iff it passes the camera's FOV cone AND the eye→target ray is unoccluded by the robot's own visual mesh (the camera's own carrying body is excluded from its own raycast). **Near clip:** every visibility path enforces a shared 0.10 m Euclidean distance from camera to target (not perpendicular depth).
- **Arm pose during visibility scoring** is a canonical neutral arms-down pose, not each voxel's own reaching configuration (see [Modeling choices](#modeling-choices)).
- Solve the right arm only and mirror positions to the left when the robot is Y-symmetric. PAL TALOS is the one robot in this study scored without folding, since its head-eye sits off the sagittal plane. Mirroring is positions/visibility only, never joint or camera commands.

## Package layout

| File | Role |
|---|---|
| `generate_workspace_curobo.py` | Core solver. Batched cuRobo IK over the voxel grid × orientation set. Can solve just a fixed slice of the row list (`--target-start/--limit`) so several GPUs can split one run. |
| `merge_workspace_curobo.py` | Concatenate shard outputs into one payload; checks grid-metadata consistency across shards. |
| `validate_workspace_curobo.py` | QC: FK-recomputes each `q_solution` and checks the result against `target_pos` (fails past ~1 mm error). |
| `ik_seed_oracle.py` | Builds a slim sidecar from the fat payload; `query(pos, quat) -> q` nearest reachable joint config, for warm-starting arbitrary online IK. |
| `gpu_visibility.py` | Batched, GPU-resident visibility scorer (MuJoCo-Warp FK + BVH raycast; occluder scene moves with the head). Writes the visibility sidecars for the five external platforms (`visibility_kind: "dynamic"`). Our rigs and G1 are scored by `plot_workspace_curobo._camera_visibility` instead, so the comparison figure uses two kernels; they agree to within 0.07 points on every steered rig, see [Why ours/G1 stay on the closed-form scorer](#why-oursg1-stay-on-the-closed-form-scorer). |
| `workspace_data.py` | Data layer: the `--robot` registry (payload filename, visibility kind, dynamic-visibility sidecar per robot) plus payload → per-voxel-`D` aggregation and the small committed `aggregated_cache/`. torch/numpy only, so aggregating (`--cache`) needs no rendering toolchain and no GPU. |
| `study_pose.py` | `set_home_pose` — the single "stand the robot the way the paper solved it" helper, shared by every figure/video/viewer path. |
| `plot_workspace_curobo.py` | Renders reachability/VRW figures. `--robot <key>` for a single-robot section-cut figure; `--reach-visible-compare` for the cross-platform comparison grid; `--camera-count-cuts` / `--vrw-3d` for the paper's other reachability figures; `--view` for a live interactive viewer. |
| `camera_count_ablation.py` | The K=1/2/3 camera-count study: table, separation curve, and figure. |
| `run_reachability_shards.py` | Multi-GPU supervisor: launches one slice per GPU, waits for completion, retains per-slice logs. |
| `visible_reachable_curator.py`, `calibrate_tau.py` | Offline goal-point curator built on the reachability/visibility fields, see [Goal-point curator](#goal-point-curator). |
| `score_visibility.py` | **The entrypoint for writing a visibility sidecar.** `score_visibility.py <workspace> <out>` runs the GPU scorer; `--device cpu` runs the per-robot reference scorer instead (T1 / ToddlerBot / Apollo only). Replaces the four near-identical `main()` blocks these backends used to carry one each. |
| `<robot>_visibility.py` (`t1_`, `toddlerbot_`, `apptronik_apollo_`, `fourier_gr3_`, `talos_`) | Per-robot dynamic-camera visibility scorers. Superseded by `gpu_visibility.py` for production, but kept as the only *independent* implementation for the external platforms; reachable via `score_visibility.py --device cpu`. All five also supply the CPU `eye_in_fov_batch` the live viewer's FOV-only "instantaneous" mode calls per frame. |
| `generate_curobo_safe_arm_poses.py`, `generate_mjcf_safe_arm_poses.py` | Safe-pose seed generators, split by which self-collision gate the source needs — cuRobo sphere pairs (`--robot berkeley/toddlerbot/apptronik_apollo`) vs. the source MJCF's own contacts (`--robot booster_t1/fourier_gr3/pal_talos`). Both sample arm hinges within MJCF limits, reject self-collision, and write the same `safe_arm_poses_<robot>.pt` cache schema. See [Setup](#safe-pose-seed-cache) for which generator owns which robot. |
| `view_common.py`, `view_<robot>.py` (`view_g1`, `view_berkeley_humanoid_lite`, `view_fourier_gr3`, `view_t1`, `view_toddlerbot`, `view_apptronik_apollo`) | Shared viewer scaffold + per-robot canonical viewer: source visuals/collision, camera FOV, and cuRobo spheres on separate toggle groups; shows source rest pose and live cuRobo FK. |
| `probe_orientation_quiver.py` | Standalone orientation/viewer probe. |
| `fix2_tilt_sweep.py` | Sweeps the Fix₂ fixed-camera baseline's mount pitch for the best static tilt (paper Sec. V-A). |
| `measure_frozen_occluder_error.py` | Measures the error from raycasting against a frozen (zero-gimbal) camera-module occluder — backs the welded-camera modeling choice (see [Modeling choices](#modeling-choices)). |
| `build_position_gate.py` | Builds a fine-grid position gate from a coarse position-reachability payload — a compute-saving pre-solve heuristic, not a proof rejected cells are unreachable. |
| `test/` | Regression and pass/fail check scripts: `pairwise_gate.py` (closed-form vs. unified kernel agreement), `armpose_bias.py` (canonical-pose vs. reaching-pose visibility bias), `run_eta2_platforms.py` (cross-platform η₂ manifest builder), `regress_gpu_visibility.py` (sidecar bitwise regression), `verify_ours_g1_port.py` (measures the ours/G1 closed-form → `gpu_visibility` port; reports, never asserts). |

## Setup

Requirements: an NVIDIA GPU (both the cuRobo IK solve and the GPU visibility raycast need CUDA), a Python environment able to run the rest of this repo (`mujoco`, `torch` with CUDA, this project's usual dependencies), and [NVIDIA cuRobo](https://github.com/NVlabs/curobo) importable on `PYTHONPATH`. cuRobo is pure Python plus a warp JIT kernel, no build step: clone it and point `PYTHONPATH` at the clone.

```bash
PY=python                    # your interpreter, with this repo's and cuRobo's deps installed
CUROBO_DIR=/path/to/curobo   # git clone https://github.com/NVlabs/curobo.git
export PYTHONPATH=$CUROBO_DIR
cd /path/to/legged_env_v2    # this repo's root
```

Everything below assumes you're in the repo root with `$PY` and `PYTHONPATH` set as above, and invokes scripts by their path under `mj_envs/asset_zoo/reachability_study/`. Only `generate_workspace_curobo.py` needs cuRobo; the plotting scripts don't import it. `MUJOCO_GL=egl` (used below for offscreen rendering) needs a working headless EGL install; use `MUJOCO_GL=glfw` instead if you have a display and EGL isn't set up.

### Safe-pose seed cache

Every solve needs a per-robot cache of collision-free arm poses first, `mj_envs/asset_zoo/cache/safe_arm_poses_<robot>.pt`. `cache/` is gitignored (it also holds the multi-GB workspace payloads and visibility sidecars), so a fresh clone has none of these, and `generate_workspace_curobo.py` fails with `FileNotFoundError: missing pose cache` until you build one.

Three generators cover the nine robots, split by how each source certifies self-collision. Every robot the solver accepts is covered by exactly one; find yours in this table:

| Generator | `--robot` values | Self-collision gate |
|---|---|---|
| `mj_envs/asset_zoo/generate_safe_arm_poses.py` | `humanoid_v21`, `unitree_g1`, `openarm_v2` | mujoco-warp batch sampler (~2 min, 2M poses, one GPU) |
| `reachability_study/generate_curobo_safe_arm_poses.py` | `berkeley`, `toddlerbot`, `apptronik_apollo` | cuRobo sphere-pair disjointness |
| `reachability_study/generate_mjcf_safe_arm_poses.py` | `booster_t1`, `fourier_gr3`, `pal_talos` | source MJCF's own contacts (`ncon`) |

```bash
$PY mj_envs/asset_zoo/generate_safe_arm_poses.py --robot humanoid_v21
$PY mj_envs/asset_zoo/reachability_study/generate_curobo_safe_arm_poses.py --robot apptronik_apollo
$PY mj_envs/asset_zoo/reachability_study/generate_mjcf_safe_arm_poses.py --robot pal_talos
```

The two study generators write the same cache schema and differ only in that gate: pick by what the source MJCF actually ships, not by preference. A source whose own collision geoms are trustworthy needs no sphere model to sample poses; one whose aren't (or which has none) needs the cuRobo spheres it will be planned with anyway. A robot in neither table needs a generator written, see [Adding a new robot](#adding-a-new-robot).

Cache filenames use the robot's full name, which is not always the `--robot` spelling: `--robot berkeley` writes `safe_arm_poses_berkeley_humanoid_lite.pt`. The generator prints the path it wrote.

## Quickstart: one robot, one GPU

A coarse, fast solve that exercises the full pipeline end to end (~1 minute on one GPU). This is the right first thing to run to confirm your environment works, before attempting a full-resolution payload:

```bash
# 1. Solve a coarse grid (0.08 m spacing instead of the production 0.02 m -- ~1 min, one GPU).
#    --limit 0 means "every row"; the flag defaults to 16 (a debug slice), which silently produces
#    an empty payload rather than an error. Production runs below pass a real per-shard --limit.
CUDA_VISIBLE_DEVICES=0 $PY mj_envs/asset_zoo/reachability_study/generate_workspace_curobo.py \
  --robot humanoid_v21 --side R --target-source grid --orientation-source so3 \
  --n-orientations 64 --grid-spacing 0.08 --pad 0.10 --num-seeds 16 --batch-size 256 \
  --limit 0 --out /tmp/smoke_v2.pt

# 2. Validate: same-q FK error, finite-difference Jacobian, target residual against a ~1mm pass/fail threshold.
$PY mj_envs/asset_zoo/reachability_study/validate_workspace_curobo.py \
  --robot humanoid_v21 --input /tmp/smoke_v2.pt

# 3. Plot a reachability-index figure from it. NOTE the key changes: the solver takes MODEL names
#    (humanoid_v21), the plotter takes REGISTRY keys (v2, v2_fixed, ...). Each --help lists its own.
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py \
  --robot v2 --input /tmp/smoke_v2.pt
```

Swap `--robot humanoid_v21` for any key in `generate_workspace_curobo.py --help`'s `--robot` list (`g1`, `booster_t1`, `fourier_gr3`, `pal_talos`, `toddlerbot_2xm_gripper`, `apptronik_apollo`, ...), and the plotter's `--robot` for the matching key in `workspace_data._ROBOTS` (`v2`, `g1`, `booster_t1`, `fourier_gr3`, `pal_talos`, `toddlerbot`, `apptronik_apollo`, plus the `v2_*` rig variants). The two vocabularies overlap for the external platforms and diverge for ours — `humanoid_v21` is `v2` there.

## Scaling to a production payload

The paper's payloads use `--grid-spacing 0.02` (not `0.08`), which is roughly 60x more rows and too slow for one GPU. Production runs cut the row list into non-overlapping slices (one per GPU), solve them in parallel, and merge:

```bash
# Total rows = n_grid_voxels * n_orientations; the grid size for a given --pad is printed by
# generate_workspace_curobo.py on startup, or read back from a payload's n_grid_per_axis field.
# Example: split TOTAL=8,000,000 rows across 4 GPUs -> CHUNK=2,000,000 rows each.
TOTAL=8000000; N_GPU=4; CHUNK=$(( (TOTAL + N_GPU - 1) / N_GPU ))
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY mj_envs/asset_zoo/reachability_study/generate_workspace_curobo.py \
    --robot humanoid_v21 --side R --target-source grid --orientation-source so3 \
    --n-orientations 64 --grid-spacing 0.02 --pad 0.10 --num-seeds 32 --batch-size 512 \
    --target-start $((g * CHUNK)) --limit $CHUNK --out /tmp/chunk_${g}.pt &
done
wait

# Merge in ascending target-start order (merge_workspace_curobo.py concatenates by argument
# order -- it does not sort or verify contiguity).
$PY mj_envs/asset_zoo/reachability_study/merge_workspace_curobo.py \
  --out mj_envs/asset_zoo/cache/workspace_curobo_ik_humanoid_v21_R_so3_dex_0p02.pt \
  /tmp/chunk_{0,1,2,3}.pt

$PY mj_envs/asset_zoo/reachability_study/validate_workspace_curobo.py --robot humanoid_v21 \
  --input mj_envs/asset_zoo/cache/workspace_curobo_ik_humanoid_v21_R_so3_dex_0p02.pt

MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --robot v2
```

An external platform needs one more step: its visibility sidecar, which the figures read alongside
the workspace payload. Our rigs and G1 skip this (they are scored inline by the closed-form kernel).

```bash
# 1. Score the sidecar. The output path defaults to the one the figures look for -- don't pass it
#    by hand (two naming conventions are in use and a wrong name writes a file nothing reads).
#    --device cpu runs the independent per-robot reference scorer instead (T1/ToddlerBot/Apollo
#    only, much slower); the two backends are not expected to agree bit for bit.
$PY mj_envs/asset_zoo/reachability_study/score_visibility.py \
  mj_envs/asset_zoo/cache/workspace_curobo_ik_toddlerbot_R_so3_dex_0p02.pt

# 2. Roll it into the aggregate the figures actually read. REQUIRED: sidecars live in gitignored
#    cache/, but figures read the tracked aggregated_cache/<stem>_aggregated.pt. Skip this and
#    every figure keeps rendering the previous numbers, silently.
$PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --cache --robot toddlerbot

# 3. Confirm the sidecar reproduces from its raw workspace. Covers all five dynamic-visibility
#    robots, expects diff=0 on each, SKIPs any whose payload or sidecar is absent locally.
$PY mj_envs/asset_zoo/reachability_study/test/regress_gpu_visibility.py
```

Each GPU in the loop above is one process taking a deterministic slice (`[target_start, target_start+limit)`) of the same global `(grid x orientations)` row list, so any two runs solve identical rows in identical order regardless of how many GPUs are used. `run_reachability_shards.py` wraps this pattern (launch, wait, retain per-chunk logs) for larger GPU counts.

**Choosing `--pad`.** `--pad` sets how far the grid extends past the seed-pose cloud; it changes how many voxels the grid has, never where they sit. (The lattice itself is anchored to integer multiples of `--grid-spacing`, so voxel `i` is the same physical point across any two payloads at the same spacing, which is what makes per-robot comparisons subtractable.) Too tight a pad silently caps the reachable set flat instead of showing where it naturally ends. Find a safe value with one coarse, large-pad probe (`--grid-spacing 0.08 --pad 0.35` or similar) before committing to the fine solve, then confirm after the fine solve that **all six grid faces have zero reachable voxels**. If any face is nonzero, the grid clipped the workspace and `--pad` must go up.

**Post-merge checks**, all of which must pass before a payload is trusted: **contiguous slices** (slice target-starts tile `[0, N)` with no gap/overlap), **rows == full grid** (`N == nx*ny*nz*n_orientations`), **voxel_index round-trip** (`voxel_index == arange(N) // 64`), and the six-empty-faces check above.

**The `success` flags are not bit-reproducible.** cuRobo runs many GPU seeds in parallel and thresholds `pos_err`/`rot_err`; re-solving an identical window flips roughly 0.1% of rows on float non-associativity in the reduction order. Never require exact set equality between two solves, and treat any effect below that ~0.1% floor as noise, not a finding.

**Payload schema.** Each row is `(voxel_index, orient_index)` and carries `target_pos, target_quat, success, q_solution, pos_err, rot_err, sigma_trans/rot, manip_trans/rot` plus grid metadata (`grid_spacing, n_grid_per_axis, grid_origin`). `q_solution` is stored `f32` (the validator's ~1 mm FK check needs that precision).

### Other figures

```bash
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --reach-visible-compare   # cross-platform grid figure
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --robot v2 --vrw-3d       # 3D teaser panel, fixed vs actuated
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/test/run_eta2_platforms.py                          # cross-platform eta_2 manifest
```

## Camera-count study (K = 1, 2, 3)

**Question:** how many head cameras should the robot carry? **Answer: two.**

Three candidate heads (K=1, one camera on the centerline; K=2, the shipped head, cameras at y = ±0.065 m; K=3, cameras at y = ±0.130 m and 0) are solved as **independent embodiments** on the same voxel lattice. Camera hardware is part of the arm's self-collision model, so adding cameras can only remove arm configurations, and the three rigs are not nested: no rig's obstacle set contains another's.

| K | reach vox | W_R (m³) | η | W_VR (m³) | ΔW_VR | η_fix | W_VR^fix | η₂ | mass (kg) | gimbal DOF |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 175,427 | 1.4034 | 0.9452 | 1.3265 | −2.22% | 0.1914 | 0.2686 | 0.4540 | 0.579 | 2 |
| 2 | 175,237 | 1.4019 | 0.9678 | 1.3567 | — | 0.3791 | 0.5315 | 0.9540 | 1.158 | 4 |
| 3 | 173,815 | 1.3905 | 0.9790 | 1.3614 | +0.34% | 0.4196 | 0.5835 | 0.9747 | 1.737 | 6 |

**Why two.** Reach falls with K (more hardware, more self-collision); coverage rises with K; their product W_VR rises steeply then nearly flattens.

- K=1 forfeits 2.22% of W_VR, more than twenty times the ~0.1% re-solve noise floor. Decisive.
- K=3 gains only 0.34% over K=2, a few times that floor. Not decisive.
- W_R and η alone leave all three heads Pareto-optimal (K=1 wins reach and mass, K=3 wins coverage), which forces an unstated exchange rate between m³ and kg. The product W_VR needs no such exchange rate to reject K=1, but it doesn't resolve K=2 vs. K=3 either. Cost breaks that tie: a third module costs about $600, 0.58 kg, and two more gimbal DOF for 0.34% more W_VR.
- A third camera also doesn't physically fit the shipped head plate (three module bases need 199 mm of width on a 180 mm plate), a hardware constraint independent of the metric.

**Saturation requires actuation.** With gimbals **locked**, a third fixed camera is worth 9.8% of W_VR^fix (vs. 0.34% steerable), and fixed heads are nowhere near saturated. The K-count result only holds because the cameras steer.

**Coverage vs. target separation** shows *why* K=1 loses, not just by how much. Sample a manipulation point and a second target at controlled separation, plot η₂ against that distance:
- K=1 decays from 0.92 (0.2 m separation) to 0.17 (1.2 m): one camera holds two targets only while both fit in its single frustum.
- K=2/K=3 stay flat and nearly coincide across the whole range: a two-camera rig assigns one target per camera.
- With gimbals locked, all three curves decay instead (K=3: 0.33 to 0.14). No camera count rescues a rigid head; steering, not count, is what buys the flat K=2/K=3 curves above.

**Where K=3's reach loss sits.** Its 2% reach loss (right arm) concentrates in a compact overhead cap roughly 1.3-1.6 m up and across the midline, not spread evenly: K=3 costs overhead reach specifically, cheap for tabletop work and expensive for high shelves.

**What the table alone can't establish.** η and η₂ are non-decreasing in K by set inclusion over per-camera masks (a violation would be a kernel bug, not a finding), and W_R is non-increasing in K because each module only adds obstacles to a serial chain. Both directions are theorems; only the magnitudes are empirical. Not established: layout at fixed K (each K uses the one head-plate geometry the plate admits), K=0, or any K beyond 3. The pairwise metric also uses the auxiliary-region variant (`x_i` must be reachable, `x_j` only needs to be visible), since the payload contains no simultaneous bimanual IK.

### Reproduce

```bash
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --camera-count-cuts      # section-cut figure (fig. panel a)
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --reach-visible-compare  # cross-robot comparison
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/camera_count_ablation.py                          # table + separation curve (panel b)
```

Each K rig is solved independently as a full workspace payload (73×91×73 = 484,939 voxels × 64 orientations ≈ 31M rows per rig); at 16 GPUs each rig takes roughly 25 minutes. Regenerating a rig from scratch is only needed if a rig's collision geometry, orientation sampling, or the grid changes. Otherwise the cross-figure numbers reproduce from the cached aggregates in minutes with no GPU solve, only offscreen rendering.

## Cross-robot comparison

Head-camera comparison across the study's eight columns (our two V2 configurations plus six other public humanoid platforms), scored by the same visibility kernel (`gpu_visibility.py`) so no column is an artifact of a different scorer.

| Robot | Head DOF | Cameras | Coverage | W_VR (m³) | W_R (m³) |
|---|---|---|---|---|---|
| Ours, fixed | 0 (locked) | 2× stereo, back-to-back | 37.9% | 0.531 | 1.402 |
| Ours, actuated | 4 (2 gimbals) | 2× stereo, back-to-back | 96.8% | 1.357 | 1.402 |
| Unitree G1 | 0 (fixed) | 1 | 15.5% | 0.136 | 0.879 |
| ToddlerBot | 2 (neck) | 2× stereo fisheye | 79.7% | 0.074 | 0.093 |
| Booster T1 | 2 (neck) | 1 (RealSense D455) | 67.1% | 0.734 | 1.094 |
| Apptronik Apollo | 3 (neck) | 2× stereo (uncalibrated, provisional) | 75.8% | 0.840 | 1.109 |
| Fourier GR-3 | 2 (neck) | 1 (Luxonis OAK-class, 128°×80°) | 69.7% | 1.092 | 1.567 |
| PAL TALOS | 2 (neck) | 1 (Orbbec Astra Pro, 63.1°×49.4°) | 47.8% | 0.923 | 1.930 |

Only G1 and our fixed configuration are true fixed heads; the other five carry an actuated neck and are scored over every legal neck state. Apollo's cameras have no vendor-published intrinsics (resolution, exact FOV, or optical calibration), so its head stereo pair uses an estimated envelope and is reported as provisional/diagnostic, not a calibrated sensor claim. GR-3's mount pitch is ambiguous between two vendor sources (15° vs. 40°) and moves its coverage by 1.5 points (69.7% vs 71.2%); both are reported, with 15° as primary.

**Single-target coverage is blind to camera independence.** A wide-FOV or actuated single eye can score well on η alone, since one eye can point at any one named target, while still failing to hold two separated targets at once. Scoring the same eight columns on pairwise η₂ instead shows our actuated configuration essentially flat with separation (η₂ ≈ η), while every other column falls as the two sampled targets move apart. That includes our own fixed configuration on the identical body and reachable field, which places the effect in camera independence rather than morphology. One caution: a smaller robot draws systematically closer target pairs from its own bounding-box statistics, which inflates η₂ for small platforms unless separation is matched across robots. Quote the separation-conditioned curve, not the flat scalar, for any cross-platform η₂ claim.

## Adding a new robot

Do these in order; do not generate a safe-pose cache, workspace payload, or visibility result before every preceding step passes.

1. **Audit the source MJCF/URDF.** Identify the real floating root, standing `qpos0`, serial arm chain, actuated coordinates, equality/gear couplings, collision groups, and terminal body. Add a fixed end-effector site at the intended tool/grasp center. Verify it against the mesh geometry, not the vendor's naming or docstring: a site named "grasp center" placed on a moving gripper finger is a real, recurring mistake. Never plan to a moving finger link.
2. **Fix the root frame before export.** The root must be the physical `base_link`, not an upstream bookkeeping/world body. Re-express every direct child transform in that frame; keep standing placement only in the floating-root `qpos0`; verify FK anchors in `base_link` coordinates.
3. **Produce and validate the cuRobo URDF.** cuRobo plans from a URDF while everything else in this study reads the source MJCF, so the two must stay in step. Register the source in `asset/create/export_mjspec_to_urdf.py` with a compile snapshot (`nbody, njnt, ngeom, nmesh, nsite, nu`) and tool sites, then regenerate both the full and cuRobo URDFs after every source-MJCF edit. Require source preflight, scene-graph FK, joint-limit parity, and a cuRobo `RobotBuilder` load. Note the exporter currently registers only `humanoid_v21`, `g1`, `fourier_gr3`, and `pal_talos`; the other robots' `*_curobo.urdf` are **tracked artifacts with no committed producer**, so for those a missing file is recovered with `git checkout`, not regenerated. Register a new robot rather than adding to that set.
4. **Define planner coordinates truthfully.** The cuRobo c-space must contain the serial physical output coordinates. If a source motor drives a sibling gear branch, do not force it into a URDF mimic. Plan the physical driven coordinate and implement the source-motor coupling explicitly.
5. **Build the collision approximation from source collision geoms**, not from scratch. Inspect the exact MuJoCo collision first; approximate only after visual comparison; keep critical joint housings; merge spheres only under a bounded unshared-volume test. Never widen spacing just to paper over a box's own over-coverage; use inscribed fitting instead.
6. **Build one canonical viewer**, not mode-specific scripts: source visuals on group 2, source collision on group 3, camera FOV on group 4, cuRobo spheres on group 5, all independently toggleable. It must show both the exact source rest pose and live cuRobo FK. Build it on `view_common.py` and copy the nearest `view_<robot>.py`. (`pal_talos` is the one study robot with no viewer — it was scored without one. That is a gap, not a precedent: the viewer is what catches a mis-placed tool site or an inverted camera axis before a multi-hour solve bakes it in.)
7. **Define cameras from physical metadata only**: parent link, lens pose, optical convention, resolution, and calibrated intrinsics/FOV where the vendor publishes them (state clearly when an envelope is an approximation rather than a calibration). An IMU is not a camera. For stereo, score each eye's FOV and occlusion independently and OR the result; never substitute a midpoint camera. MuJoCo cameras look along local `-Z`; verify the axis convention and render both viewports before scoring visibility.
8. **Only then build reachability.** Add the robot to whichever safe-pose generator matches its self-collision gate ([Setup](#safe-pose-seed-cache)) — a new `RobotSpec` entry in `generate_curobo_safe_arm_poses.py` or `generate_mjcf_safe_arm_poses.py`, not a new script — and confirm it rejects real self-collision (except named, intentional static rest contacts). Then run a coarse large-pad probe, raise `--pad` until all six fine-grid boundary faces are empty, and run/shard/merge/validate the full workspace solve. `validate_workspace_curobo.py` enforces the face check and the post-merge grid checks; a payload that fails them is not a payload.
9. **Register the robot for figures.** A validated payload is still invisible to every figure until it is in `workspace_data._ROBOTS`. That module's header carries the current four-step recipe (CLI key + model key + payload filename + `visibility_kind`, an `_overlay_assets` branch, and a pose entry if the arms-down pose differs); follow it there rather than a copy here, since it sits next to the code it describes. Reachability-only columns are done at this point — `--robot <key>` and `--cache` now work.
10. **Add dynamic visibility, if the robot has an actuated neck.** Declare a `_ADAPTERS` entry in `gpu_visibility.py` (eyes as `(MJCF camera, site)` pairs, aim site/axis/sign, head joints, visual group) plus a `_DYNAMIC_VISIBILITY_SIDECAR` path in `workspace_data.py`, then score it and roll it into the aggregate as in [Scaling to a production payload](#scaling-to-a-production-payload). Two fields are silent-error-prone and worth re-deriving rather than copying: `image_axes` (a rolled camera mount swaps H and V FOV, which on a non-square sensor is a real error) and `aim_groups` (what separates "two eyes on one neck" from "two independent gimbals" — single-target η cannot see the difference, η₂ can). Finally add the robot to `test/regress_gpu_visibility.py`'s `_CASES` so later refactors are pinned against its sidecar.

## Modeling choices

Computing the paper's formal VRW definition literally (a per-orientation simultaneous witness that each voxel is both reached *and* seen from that voxel's own reaching configuration) would be far more expensive for a negligible accuracy gain. The study instead makes two tractability choices — score visibility at a single canonical pose, and treat reach and visibility as separable fields — plus a few smaller ones. The two with the largest potential effect were measured directly and shift the reported numbers by only a few points; the smaller ones were not separately measured and are listed as caveats.

### Measured directly, small effect

Visibility is scored at one neutral arms-down pose, not each voxel's own reaching configuration. Measured directly (`test/armpose_bias.py`, re-solving occlusion against each voxel's actual `q_solution`): coverage moves -0.7 to -3.7 points depending on the rig, and a few voxels (8 of 6,000 sampled) are visible *only* from a reaching posture. That's not a provable bound in either direction — a reaching arm can create or remove occlusion — but the measured effect is small enough to keep the canonical pose as the default. Always report the voxel-level OR over reaching states, never a per-row mean; they differ by an order of magnitude because the definition is existential.

The reach solve also uses a locked/welded camera collision model, so steered visibility moves the lens without re-checking arm-camera collision at the aimed pose. Camera modules were measured to almost never occlude their own aimed cone (0/1,800 sampled in-cone rays across K=1-3), which supports low risk, not proof, but again backs the choice rather than leaving it open.

### Not separately measured

The K=1 pairwise aim is a heuristic, not exact: an angular bisector for a rectangular FOV misses about 5.3% of coverable target pairs at large separation (65-97°); K≥2 has no equivalent gap. The exact test — search all pan/tilt/roll aims for one that frames both targets in the rectangular cone — was skipped on purpose: it is far costlier than the closed-form bisector, and its only effect would be to *raise* K=1's coverage (the bisector undercounts), which cannot flip a decision that already rejects K=1 decisively on W_VR and on the η₂ decay curve. Each side's reachability ignores the opposite arm's occupancy during IK, so the mirrored bimanual union is mildly optimistic near the sagittal midline, with no measured bound. And external camera specs are vendor-published envelopes, not independent calibration, except where noted: Apollo's head stereo pair has no published intrinsics at all (diagnostic only), and GR-3's mount pitch is ambiguous between two vendor documents (both reported).

## Why ours/G1 stay on the closed-form scorer

The five external platforms are scored by `gpu_visibility`; our rigs and G1 by
`plot_workspace_curobo._camera_visibility`. The two kernels agree to within **0.07 points of η** on
every steered rig and are bit-identical on every fixed one, so the split costs nothing measurable;
unifying onto `gpu_visibility` remains open but is not worth re-cutting figures for.

One defect is still open, in the closed-form scorer: **frozen occluder.** Its steered path aims the
lens per target but raycasts a scene with every camera module frozen at zero gimbal, so the housing
is not where it really is once aimed (the fixed baseline's half is already repaired via `wd_fixed`).
Declared inline in that function. Measured magnitude ~0.05 points of η on K=1 — this is what keeps
the two kernels from agreeing exactly. `test/verify_ours_g1_port.py` is the acceptance test for
repairing it or for completing the unification; every fixed rig must stay bit-identical (currently
`agree 1.0000`, 0 lost / 0 gained on all four).

**If you add a robot with a wide-pitch neck, read this.** `gpu_visibility._aim_group` carries two
guards against singularities in its DLS aim loop. Both were live bugs: the loop reported
convergence while sitting on a fixed point, so a stalled aim looked like a solved one.

1. *Antipodal.* With the target directly behind, the residual `desired - forward` is purely
   parallel to `forward`, and every Jacobian column is `ω × forward`, hence orthogonal to it — DLS
   returns `delta ≡ 0` and the aim never moves. Guarded by `_capped_residual`, which past a right
   angle solves for an intermediate direction instead of the raw chord.
2. *Gimbal lock.* Least-norm steps drive `cam_pitch` to ±90°, where the yaw axis lies on the
   optical axis and yaw stops steering. Guarded by `_lattice_seed`, a one-shot restart from the
   best-aligned node of an angle-uniform joint lattice, applied only to rows the first pass leaves
   stuck.

Neither can fire on a 2-DOF external neck in this study — the widest pitch range is ToddlerBot's
−35..80°, against the ±90° needed to lock — and a fixed rig never aims at all. Measured, not just
argued: re-scoring T1 and TALOS with the guards in place reproduces their pre-guard sidecars
bit-for-bit. Only Apollo's 3-DOF neck moved at all, by 43 rows in 6.6M. A centre-mounted single
camera is the worst case, needing ±270° of yaw to cover behind.

A large share of off-target aims on an external platform (29–80% of a uniform sphere) is normal:
that is genuine joint-limit saturation, not solver failure. A dense brute-force sweep of the same
limits does no better on >90% of those rows. Don't read it as a bug to chase.

## Goal-point curator

Offline lookup that answers "is this a good goal point?" for a candidate 3D point, without a live IK call. It reuses two things already computed by the study: the reachability field `D_reach` (fraction of the 64 orientations that reach the point) and a visibility scorer (can any camera see it). A point is **useful** iff `visible AND D_reach >= tau`.

It also scores **base-shift robustness**: how well the point survives small base-pose drift (e.g. imprecise localization). `D_reach` is spatially smooth, so this is just the minimum `D_reach` over a small ball (default ±5 cm) around the point — a point with a high-scoring neighborhood stays reachable+visible even if the base moves slightly.

This is a cheap offline ranking hint for planning, not a replacement for a runtime IK feasibility check — always confirm with a real IK solve before committing to a goal.

```bash
# Build the field sidecar once per robot (loads the full payload + a GPU visibility raycast pass).
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/visible_reachable_curator.py --robot v2   # also: v2_fixed, g1

# Fit tau and validate the base-shift hypothesis against cuRobo ground truth.
MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/calibrate_tau.py --robot v2 --n 160 --n-shift 12
```

```python
from mj_envs.asset_zoo.reachability_study.visible_reachable_curator import VisibleReachableField

field = VisibleReachableField.from_robot("v2")
visible, d_reach = field.useful(pos_base, "R")               # base_link-frame point, arm in {"L","R"}
robust = field.robust_score(pos_base, "R", delta_m=0.05)     # min D over a +-5 cm ball; 0 if any blind
```

**Calibration.** `tau = 0.25` is the threshold that best balances false positives against missed points versus cuRobo ground truth, fit on one scenario only — re-fit per robot before trusting it elsewhere. A floor is needed because `D_reach > 0` alone overcounts: it means *some* orientation reached the point, not that the mission's actual grasp orientation did. The base-shift hypothesis checks out too: robustness score correlates with real feasibility under ±5 cm base jitter at `r = +0.56`.

**Out of scope:** not a runtime IK-check replacement. Dynamic-camera robots (ToddlerBot, T1, Apollo, GR-3, TALOS) use their own visibility sidecars, not this field.
