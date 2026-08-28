# Reproducing the paper

Full recipes for every number and figure in the paper. The short version, and the three
things you can watch without any of this, are in [README.md](README.md).

## Reproduce

### Locomotion policy (training)

```bash
export WANDB_MODE=offline   # or `wandb login` first; training logs through W&B
python mj_envs/run.py train --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam
python mj_envs/run.py play  --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam
```

Training logs to Weights & Biases and will stop at
`UsageError: No API key configured` if it can find neither a login nor `WANDB_MODE`. Offline mode
needs no account and still writes every metric under the run directory, so `wandb sync <dir>` can
upload it later. Add `--max-iterations N` for a short smoke run; 3 iterations is enough to confirm
the environment builds and a checkpoint gets written.

The other task classes behind the reported numbers are
`HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam` and
`G1RmaVelEstArmFlashSacStudentOnlyg1bsk2`.

You do not need to train anything to reproduce the tables. The checkpoints they were computed
from are in `mj_envs/tasks/visual_manipulation/test/checkpoints/`, and that directory's
`README.md` lists the md5 and the training command for each one. Only the weights named in
`dyn_sweep.PINS` are shipped. The same `README.md` describes the candidates that were evaluated
and rejected, whose weights are not included.

### Two-target reach-and-grasp benchmark (Table IV)

```bash
python mj_envs/tasks/visual_manipulation/test/dyn_sweep.py \
    --robots g1,v2_fixed,v2,v2_single_fixed,v2_single --out <run_name>
python mj_envs/tasks/visual_manipulation/test/dyn_aggregate.py <run_dir>
```

6 scenarios x 3 repeats x 10 seeded layouts = 900 trials. This runs multi-GPU, and `--hosts`
sets the device topology. The GPU contact solver is not bit-reproducible, which is why each
configuration is repeated 3 times.

`dyn_aggregate.py` prints per-cell and per-variant rows. The per-variant means are Table IV's
bottom row, and are what a rerun should land near:

| | Time T̄ (s) | Search (s) | Approach (s) | Manipulation (s) | Energy (J) |
| --- | ---: | ---: | ---: | ---: | ---: |
| G1 | 27.5 | 3.8 | 3.8 | 19.9 | 494 |
| Fix_2 | 17.0 | 0.2 | 3.2 | 13.6 | 425 |
| **Act_2** | **14.1** | **0.1** | **2.1** | **11.9** | **346** |
| Fix_1 | 20.5 | 3.3 | 3.3 | 13.9 | 595 |
| Act_1 | 15.5 | 0.6 | 2.7 | 12.2 | 385 |

Success-conditional means over 180 trials per variant. Success rate is 0.967 to 0.994 across
all five and does not separate them. Do not expect a bit-exact match: the contact solver is
nondeterministic and verdicts are not host-portable, so a rerun on different hardware moves
individual cells. The published set was measured on one L40S pair.

<table>
<tr>
<td width="50%"><img src="media/two_target_left_right_close.webp" width="100%" alt="Two targets left and right, benches close"></td>
<td width="50%"><img src="media/two_target_front_back_far.webp" width="100%" alt="Two targets front and back, benches far"></td>
</tr>
<tr>
<td><b>left_right_close.</b> Both targets within reach. Only the actuated pair keeps both in view
and reaches both.</td>
<td><b>front_back_far.</b> Benches at 0.8 m, so the robot locates the targets, then walks.</td>
</tr>
</table>

Each panel is one `--robots` column: `v2_single_fixed`, `v2_single` on the top row,
`v2_fixed`, `v2` on the bottom. The remaining four scenarios are on the
[entry-point README](https://github.com/generalroboticslab/duke_humanoid_v2#two-target-reach-and-grasp-benchmark).

The `--robots` order above is Table IV's column order. `dyn_aggregate.LABEL` holds the mapping:

| `--robots` value | Table IV column |
| --- | --- |
| `g1` | G1 |
| `v2_fixed` | Fix_2 |
| `v2` | Act_2 |
| `v2_single_fixed` | Fix_1 |
| `v2_single` | Act_1 |

`v2` and `v2_fixed` are the same robot and the same weights; they differ only in whether the head
cameras are welded. Same for the `v2_single` pair. Three checkpoints therefore cover five columns,
and `dyn_sweep.pin_report` prints the md5 each column will load.

#### Running one mission

`dyn_sweep.py` runs the whole table. To watch or debug a single cell, drive the same mission
directly:

```bash
python mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py \
    --robot v2 --scenario bimanual_mixed_close --dynamic --mpc --walk --camera --view
```

That is the full benchmark mission for one robot and one scenario. Each animation in the
entry-point README is four such runs, one per camera setup, tiled by
`media/blender_6scenario/build_v2_quadrants.sh`. Each flag changes what is being tested:

| flag | what it does | leaving it off |
| --- | --- | --- |
| `--dynamic` | realizes the plan through the frozen RL policy and MuJoCo physics | idealized kinematic `mj_forward`, so a pass proves only that the plan was valid |
| `--walk` | multi-cube mission, SEARCH to GO to REACH to PARK per visit; base drives until the cube is arm-reachable | one parked reach from the start stance |
| `--camera` | reaches only cubes the head cameras actually see, so the arm count follows the visible-reachable set | privileged: reaches any cube whose ground-truth pose is reachable |
| `--mpc` | per-tick reactive tracker, the only path with gravity compensation | a PD servo with no feedforward; the arm droops by `tau_gravity(q)/kp` on every grasp |
| `--view` | live MuJoCo viewer; `--viewer blender` renders the same rollout through EEVEE | headless, verdict printed only |

`--mpc` defaults on and every published evaluation passed it explicitly. `--camera` is the flag
that makes the run test the paper's claim rather than the arm alone: without it, a blind but
reachable cube still gets picked up, which is exactly the deficit VRW measures.

Robots: `g1`, `v2`, `v2_fixed`, `v2_single`, `v2_single_fixed`, and `both` (runs g1, v2_fixed and
v2 against one shared scene). Scenarios: `left_right_close`, `left_right_far`, `front_back_close`,
`front_back_far`, `front_far_discover`, `bimanual_mixed_close`, `bimanual_mixed_front_back_close`,
`shelf`, `shelf_pick_both`.

Two cautions. A single verdict is not a benchmark: one run grades one seeded layout under one
target jitter, and the GPU contact solver is not bit-reproducible, so the same command has been
seen to both pass and fail on `g1 left_right_close`. Table IV comes from `dyn_sweep.py` for that
reason. And g1 cannot reach `bimanual_mixed_close`'s far-lateral cube from a stationary stance;
with `--walk` the base repositions per cube and it passes.

### Visible-reachable workspace (Fig. 2 and Fig. 5)

```bash
# 1. generate per-robot workspace + visibility data (GPU, sharded, needs cuRobo)
python mj_envs/asset_zoo/reachability_study/generate_workspace_curobo.py --robot humanoid_v21
# 2. main cross-platform comparison figure
MUJOCO_GL=egl python mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py \
    --reach-visible-compare
# 3. camera count x articulation ablation
python mj_envs/asset_zoo/reachability_study/camera_count_ablation.py
# 4. pairwise eta_2 across platforms
python mj_envs/asset_zoo/reachability_study/test/run_eta2_platforms.py
```

![Visible-reachable workspace, fixed versus actuated cameras](media/vrw_fixed_vs_actuated.webp)

The measure the figures report, as a volume: the same arms and the same body, differing only in
whether the camera joints are free. The blue volume is what VRW exists to expose, space the arm
can reach and the cameras cannot see. Rendered by `plot_workspace_curobo.py --vrw-video --robot
v2` and `--robot v2_fixed`, which reads the same `aggregated_cache/` the figures do.

Step 2 is Fig. 2 and step 3 is Fig. 5. Both run as shipped from `aggregated_cache/` and finish in
seconds. Step 2 reads the per-column eta_2 for its titles out of `result/eta2_manifest.json`, which
is also shipped. Step 4's figure is not in the paper; it is the separation-resolved view behind the
scalar eta_2 that Fig. 2 prints in its titles.

Step 1 recomputes from a platform's safe-arm-pose cache. The two shipped caches are `humanoid_v21`
(this robot, shown above) and `unitree_g1`, so those two run as is; every other platform needs
its cache regenerated first (see Precomputed data below). Note that `generate_workspace_curobo.py`
takes the model name, `humanoid_v21`, not the `v2` shorthand the benchmark scripts use for the same
robot; `--help` lists the accepted values. A full step 4 needs all of them, but
`run_eta2_platforms.py --plot-only` replots every column from the shipped manifest without
touching a GPU. Since the Fig. 2 titles read their eta_2 from that same manifest, the two always
agree.

### Regression check

```bash
python mj_envs/asset_zoo/reachability_study/test/regress_gpu_visibility.py
```

This rescores each raw workspace payload through `gpu_visibility.score_targets` and diffs the
result against that robot's committed sidecar. Every figure reads `aggregated_cache/`, so a change
to the visibility kernel cannot move them; this check is what catches one.

The raw payloads are several GB each and are not distributed, so on a fresh clone every case skips
and the script exits non-zero. Run step 1 for at least one robot first.

### Task keyframe figure (Fig. 6)

```bash
python mj_envs/tasks/visual_manipulation/test/make_keyframe_figure.py \
    --keyframe-dir mj_envs/tasks/visual_manipulation/media/keyframes_6scenario
```

This runs as shipped. `keyframes_6scenario/` holds the six run manifests and the frames the figure
reads, one per stage per scenario. Add `--capture all` to re-shoot the frames rather than replot
them; that reruns the six missions through `reach_policy.py` and needs a GPU.

![Two-target reach-and-grasp benchmark](mj_envs/tasks/visual_manipulation/media/keyframes_6scenario/keyframe_figure.png)

The six benchmark scenarios, one row each.

## Verified

Every command below was run from this directory on one RTX 4090:

- `run.py train --algo flash_sac` trained 3 iterations and wrote a checkpoint.
- `curobo_reach_verify.py --dynamic --robot v2 --scenario left_right_close` returned
  `VERDICT: PASS` with `runs/` absent, which is the pinned-checkpoint path a fresh clone takes.
- `plot_workspace_curobo.py --reach-visible-compare` regenerated the figure with the fractions in
  the paper: ours actuated 96.8%, G1 15.5%, T1 67.1%, Apollo 75.8%, GR-3 69.7%, TALOS 47.8%.
- `camera_count_ablation.py` regenerated the K=1/2/3 whole-body coverage and the fixed versus
  actuated eta_2 curves from `aggregated_cache/` alone.
- `run_eta2_platforms.py --plot-only` replotted all eight columns from the shipped manifest.
- `make_keyframe_figure.py` rendered the 6x5 keyframe grid from the shipped manifests.
- `generate_mjcf_safe_arm_poses.py --robot pal_talos` sampled a fresh cache from the shipped MJCF,
  which is the regeneration path for the omitted caches above.
- `regress_gpu_visibility.py` scored 4 of 5 cases with `diff=0` in the source tree, GR-3 skipping
  for want of its payload. On this export it exits non-zero, since no raw payload is shipped.

The three paper figures this package regenerates were rasterised at 100 dpi and compared pixel by
pixel against the PDFs the submission was built from. Fig. 5 (`fig_camera_count.pdf`) and Fig. 6
(`fig_sim_keyframes.pdf`) came back identical; Fig. 2 (`reach_visible_compare.pdf`) differed on 4
pixels of 316,000 by one intensity level, which is font antialiasing. The paper's other figures are
hardware photographs, the teaser, and a diagram whose generator lives with the manuscript, so
nothing here regenerates them.

The `curobo_reach_verify.py` line in that list is the parked form, without `--walk`: one reach
under one unseeded jitter, which is a smoke test and the path a fresh clone takes. See Running one
mission above for the full-mission flags and for why a single verdict, parked or walking, is not a
regression signal.



