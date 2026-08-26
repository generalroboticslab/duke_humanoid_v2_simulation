# Pinned locomotion checkpoints for the dynamic reach/grasp benchmark

The five benchmark variants (`g1`, `v2`, `v2_fixed`, `v2_single`, `v2_single_fixed`) run a **frozen** locomotion policy; only the reach controller above it changes between sweeps. These are those frozen weights, version-controlled so a sweep number can be traced to exact bytes.

`dyn_sweep.py`'s `PINS` table points here. **Do not add a checkpoint here without adding its row below.**

## Files

| file | variants | md5 | source run dir (may no longer exist) |
|---|---|---|---|
| `g1__G1RmaVelEstArmFlashSacStudentOnlyg1bsk2__2026-07-25_14-54-46__model_0015000.pt` | `g1` (`dyn10`, `dyn11`, and later) | `b68621edf57dde57b2bac28aaa03bbd1` | `runs/G1RmaVelEstArmFlashSacStudentOnlyg1bsk2/2026-07-25_14-54-46_flash_sac/` |
| `g1_cos_s0__G1RmaVelEstArmFlashSacStudentOnlyg1bsk2Cosine__2026-08-08_02-13-30__model_0015000.pt` | (REJECTED — see below) | `8677c6ded438c6c65761113231dd0a5a` | `runs/G1RmaVelEstArmFlashSacStudentOnlyg1bsk2Cosine/2026-08-08_02-13-30_flash_sac_s0/` |
| `g1_cos_s1__G1RmaVelEstArmFlashSacStudentOnlyg1bsk2Cosine__2026-08-08_02-13-31__model_0015000.pt` | (REJECTED — see below) | `3aaa2a256cb36f53f9cae5b68eee2cf1` | `runs/G1RmaVelEstArmFlashSacStudentOnlyg1bsk2Cosine/2026-08-08_02-13-31_flash_sac_s1/` |
| `dual__HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam__2026-07-28_18-10-03__model_0015000.pt` | `v2`, `v2_fixed` | `efae3c42a40d763957aa2988972cdf8c` | `runs/HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam/2026-07-28_18-10-03_flash_sac/` |
| `single__HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam__grl2_s0__model_0015000.pt` | `v2_single`, `v2_single_fixed` | `15abfeef7c33bd8ef283b0392c2b948b` | `runs/HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam/grl2_s0/` |

Filename is `<role>__<TaskClass>__<run-id>__<iteration>.pt`. The `TaskClass` segment is the actual class in `experiments.py`, so the training config is readable from the filename with no lookup.

## Reproduction

Each was trained by its task class at 15000 iterations:

```bash
python mj_envs/run.py train --task G1RmaVelEstArmFlashSacStudentOnlyg1bsk2                --num_envs 4096 --seed 0 --max_iterations 15000
python mj_envs/run.py train --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam  --num_envs 4096 --seed 0 --max_iterations 15000
python mj_envs/run.py train --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam     --num_envs 4096 --seed 0 --max_iterations 15000
```

Retraining will **not** reproduce these bytes (nondeterministic), which is the whole reason the artifacts are committed rather than merely described. `MixedArmsCam` is the promoted `v2_best` policy (`MEMORY.md`, "Current keepers"); `SingleCam` is its single-camera sibling for the camera-count ablation.

## Which sweep used which

| sweep | date | pin state |
|---|---|---|
| `dyn8` | 2026-08-02 | **SPLIT, defective.** Dual-cam unpinned: ser15 loaded `2026-07-28_18-10-03`, ser16 loaded `2026-07-29_07-31-01` (md5 `f37e8981…`, **not committed here** — it is not the pin). Its Fix2/Act2 columns pool two policies. |
| `dyn9` | 2026-08-03 | Dual-cam + single-cam pinned by path under `runs/`; g1 still auto-resolved but landed on `14-54-46` on all hosts, verified by md5. Single-policy per column. |
| `dyn10` | 2026-08-06 | All five pinned to the files in this directory. g1 = bsk2. |
| `dyn11` | 2026-08-08 | **REJECTED.** Retargeted g1 to `g1_cos_s0` (cosine-anneal sibling of bsk2) on the strength of a 3-cmd x 20-ep locomotion eval (fell_over 0/0, disp +14%, VTF -36%, term -36%). The 18-cell cuRobo sweep (3-GPU pool, 9 concurrent trials, all 6 scenarios x 3 reps) returned **P g1 = 0.092 across 25 cell runs** (16 of 18 cells succeeded; 2 cells failed all retries — `bimanual_mixed_close` r1/r2 — and treating them as P=0 yields 0.085). **vs dyn10 baseline 0.989, delta -0.897. Catastrophic regression across every scenario.** Reverted `PINS["g1"]` to the bsk2 ckpt. Cosine ckpts retained in checkpoints/ for inspection but no longer pinned. **Lesson: the 3-cmd locomotion eval graded a different axis than cuRobo cares about — cosine-annealed policies appear well-trained at walking but break the precise hand-object grasp-latch timing that bimanual_mixed_* and front_back_far cells depend on. Do not promote a locomotion-only-class on the locomotion eval alone; re-sweep at paper-cell granularity before promoting anything that affects the deployed ckpt.** |

## Why in-repo and not `runs/`

`runs/` is gitignored *and* rsync-excluded, so a pin pointing there had to be hand-copied to each sweep host — and a hand copy is precisely how two hosts come to hold different bytes at one path, which is the `dyn8` defect above. Files here travel with the code by the same sync, so weights and the tree that loads them cannot drift apart. A `runs/` path is also destroyed by the next training run under that task; a name carrying its own task class is not.

~20 MB each, plain git (no LFS — the repo already tracks comparable `.pt` artifacts raw under `asset_zoo/reachability_study/aggregated_cache/`).
