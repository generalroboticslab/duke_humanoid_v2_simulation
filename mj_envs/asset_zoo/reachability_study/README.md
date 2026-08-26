# reachability study

Where the visible-reachable workspace is computed. This directory produces the paper's
cross-platform workspace comparison and the camera-count ablation that selected K = 2
actuated camera modules.

**The method write-up is [`readme_reachability.md`](readme_reachability.md)** — definitions,
the occlusion model, the run recipes, and the modeling choices with their rationale. This
page is the map; that one is the manual.

## Run something

Both read the shipped `aggregated_cache/` and finish in seconds. Neither needs a GPU sweep.

```bash
# cross-platform comparison
MUJOCO_GL=egl python plot_workspace_curobo.py --reach-visible-compare

# camera count x articulation
python camera_count_ablation.py
```

## What each script is for

| Script | Produces |
|---|---|
| `generate_workspace_curobo.py` | The per-robot workspace + visibility payload. GPU, sharded. Everything below reads its output. |
| `plot_workspace_curobo.py` | The cross-platform comparison figure. `--vrw-video` instead renders one robot's volume as a rotating sweep. |
| `camera_count_ablation.py` | The K = 1 / 2 / 3 by fixed/actuated comparison. |
| `gpu_visibility.py` | The visibility kernel: FOV cone, self-occlusion raycast, near clip. |
| `test/run_eta2_platforms.py` | Pairwise coverage η₂ per platform. `--plot-only` replots from the shipped manifest. |
| `test/regress_gpu_visibility.py` | Rescores raw payloads and diffs against committed sidecars. The check that catches a change to the visibility kernel. |

## Before you regenerate a platform

`generate_workspace_curobo.py` needs that robot's safe-arm-pose cache to exist first, and
which sampler produces it depends on how the platform certifies a pose collision-free:

| Robots | Sampler |
|---|---|
| `humanoid_v21`, `unitree_g1` | `mj_envs/asset_zoo/generate_safe_arm_poses.py` |
| `booster_t1`, `fourier_gr3`, `pal_talos` | `generate_mjcf_safe_arm_poses.py` |
| `toddlerbot`, `apptronik_apollo` | `generate_curobo_safe_arm_poses.py` |

Two caches ship (`humanoid_v21`, `unitree_g1`), so those two robots run as-is. The others
need their cache regenerated; the figures do not, because they read `aggregated_cache/`.

`readme_reachability.md` has the shard-solve procedure. Read it before launching a solve — it
is a multi-GPU job that repeatedly costs hours of wall time when preflight is assumed.
