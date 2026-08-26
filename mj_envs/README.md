# mj_envs

All the code. Every command in the top-level README starts with a path into this directory,
so this page is the map from "what do I want to do" to "which file does it".

## Entry points

There are only three, and everything else is a library one of them imports.

| Command | What it does |
|---|---|
| `run.py play --task <Class>` | Watch a trained policy in a MuJoCo viewer. Resolves a shipped checkpoint on its own. |
| `run.py train --task <Class>` | Train a locomotion policy. Days on one GPU. |
| `tasks/visual_manipulation/test/curobo_reach_verify.py` | Run one two-target mission end to end. |

`run.py` is a tyro CLI: `--task` names an experiment class, and the class decides the task, the
algorithm, the rewards and the observation set. Grep for the class name to find its definition.

Never pass an alias class to `train`. Aliases point at whatever is current, so training under one
writes `runs/<alias>/`, which then goes stale the moment the alias is retargeted and silently
loads a wrong-architecture checkpoint. Train under the concrete class.

## Where things live

| Directory | What |
|---|---|
| [`asset_zoo/reachability_study/`](asset_zoo/reachability_study/) | The visible-reachable workspace computation and both workspace figures. Has its own README. |
| [`tasks/visual_manipulation/`](tasks/visual_manipulation/) | The two-target benchmark: scenarios, cuRobo planning, trackers, mission state machine. Has its own README. |
| [`tasks/humanoid_velocity/`](tasks/humanoid_velocity/) | The locomotion policy: what it observes, how it is trained, and the variants `--task` names. Has its own README. |
| `tasks/g1_velocity/` | The same for the Unitree G1 baseline. |
| `flash_sac/`, `ppo/` | The two RL training stacks. `flash_sac` is what the shipped policies were trained with. |
| `asset_zoo/` | Robot constants and scene objects: the code that grafts cameras and grippers onto a bare body. |
| `asset_zoo/cache/` | Precomputed safe-arm-pose and collision-graph payloads. |
| `mjlab_util/`, `utils/` | Shared helpers: math, IK, config plumbing. |
| `photoreal/` | Blender rendering for the figure and video assets. |

## Which policy is which

Three checkpoints ship, in `tasks/visual_manipulation/test/checkpoints/`, and that directory's
README lists the md5 and the training command for each:

| Task class | Configuration |
|---|---|
| `HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam` | two actuated camera modules, the adopted design |
| `HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam` | one camera module |
| `G1RmaVelEstArmFlashSacStudentOnlyg1bsk2` | the Unitree G1 baseline |

The welded-camera configurations in the paper (`v2_fixed`, `v2_single_fixed`) are not separate
policies. They are the same weights with the camera joints frozen, which is why three checkpoints
cover five benchmark columns.

## Adding things

Both study directories document their own extension path: `asset_zoo/reachability_study/` for
adding a platform to the workspace comparison, `tasks/visual_manipulation/` for adding a scenario
to the benchmark. Start from their READMEs rather than from this one.
