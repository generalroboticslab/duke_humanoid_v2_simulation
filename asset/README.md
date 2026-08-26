# asset

Robot models. One directory per platform, each holding its own MJCF, meshes and the cuRobo
URDF exports the planner loads.

## This robot

| Directory | What |
|---|---|
| [`duke_v2/`](duke_v2/) | **Duke Humanoid V2.** The body, the head-camera gimbals and the end effectors, each a separate component. Start here. |
| [`create/`](create/) | The shared MJCF/URDF build tooling every component's creation script imports. |

## Comparison platforms

The five external humanoids in the cross-platform workspace comparison, in the figure's
column order. Each is redistributed under its own upstream license, kept in its directory
beside the meshes.

| Directory | Platform | Visible-reachable | Scalar η₂ | Upstream | License |
|---|---|---:|---:|---|---|
| `unitree_g1/` | Unitree G1 | 16% | 0.03 | mjlab, carrying the MuJoCo Menagerie model | see mjlab; only the cuRobo URDF exports are vendored here |
| `booster_t1/` | Booster T1 | 67% | 0.20 | MuJoCo Menagerie | Apache-2.0 |
| `apptronik_apollo/` | Apptronik Apollo | 76% | 0.25 | MuJoCo Menagerie | Apache-2.0 |
| `fourier_gr3/` | Fourier GR-3 | 70% | 0.24 | Fourier GRx | **GPL-3.0** |
| `pal_talos/` | PAL Talos | 48% | 0.09 | MuJoCo Menagerie | Apache-2.0 |
| `toddlerbot_2xm_gripper/` | ToddlerBot | — | — | upstream ToddlerBot release | MIT |

For reference, this robot scores **97%** coverage and **0.96** η₂ with its cameras actuated,
and 38% / 0.14 with the same cameras welded.

**The Fourier GR-3 model is GPL-3.0**, unlike the rest of this release. It is a comparison
column in the workspace figures and nothing else depends on it, so it can be deleted if that
license is a problem for your use.

ToddlerBot reports no paper figure. It is included as a worked example: it is the smallest
robot here and so the cheapest to regenerate a payload for, and it is the only platform that
exercises the coupled-neck branch of `gpu_visibility`.

## Adding a platform

`mj_envs/asset_zoo/reachability_study/` documents the sequence: sample a safe-arm-pose cache,
solve the workspace, then aggregate. See its README for which of the three samplers a given
robot needs and why they differ.
