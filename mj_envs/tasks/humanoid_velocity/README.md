# humanoid_velocity — the locomotion policy

One network, running at 50 Hz, that outputs a position target for every actuated joint on the
robot. It walks and turns on command, holds whatever pose the arms have been told to hold, and
aims both camera gimbals. Those are not three controllers stitched together. They are 31 numbers
out of one policy, and the camera joints sit in the same action vector as the ankles.

That last detail is the whole reason this task exists in a repository about where a robot can
see. A gimbal driven by a separate look-at controller would fight the walking policy every time
the torso pitched. Putting gaze in the action vector means the policy learns to hold a view
*through* a footstep, which is what makes a camera actually usable while the robot is moving.

## What the policy sees

120 numbers per control step, and it gets the last 20 steps of them:

| Term | Width | Where it comes from on the robot |
| --- | ---: | --- |
| `base_ang_vel` | 3 | IMU gyro, body frame |
| `projected_gravity` | 3 | IMU attitude, as a gravity vector in body frame |
| `joint_pos` | 31 | motor encoders |
| `joint_vel` | 31 | motor encoders, differenced |
| `actions` | 31 | what the policy asked for last tick |
| `command` | 3 | vx, vy, wz from the operator or the navigation layer |
| `target_arm_joint_pos` | 14 | the arm pose the reach stack wants |
| `target_camera_joint_pos` | 4 | the gaze targets, yaw and pitch per module |
| | **120** | |

Everything here exists on hardware. There is no base linear velocity, no foot contact flag, no
friction estimate, no privileged state of any kind. That constraint is the point: whatever the
policy is trained to consume is exactly what the robot can hand it at 50 Hz, so the deployed
network is byte-for-byte the network that trained.

The command triple and the two target blocks arrive on the same wire on hardware. Gaze keeps its
own silence timer, because a gaze-only keepalive must not be read as the arms still being
commanded — see the deploy repo for what happens when a stream goes quiet.

## The shape of the network

The 20-frame window is not flattened into one long vector. It goes through a small temporal
encoder first, and the current frame is then re-injected alongside the summary. Straight from
the shipped weights:

```
20 frames x 120 obs
   |
   +-- Linear(120 -> 32)              per frame, shared
   +-- Conv1d(32->32, k=3)            over the time axis
   +-- Conv1d(32->32, k=3)
   +-- Conv1d(32->128, k=3)
   |
   latent (128)
        \
         concat with the current frame (120)  ->  248
                                                   |
                                     MLP 248 -> 512 -> 256 -> 128
                                                   |
                                        mu (31)   log_std (31)
```

This is the RMA idea: the history is there to let the network infer what it cannot measure —
ground friction, payload, how much torque a motor is really producing today — and squeeze it
into a latent. The current frame is passed through unchanged so the policy never has to
reconstruct "where am I right now" out of a convolution.

`248 = 120 + 128` is a useful thing to check if you are ever unsure whether a checkpoint matches
the config it claims to.

Actions are joint position targets, applied as `default pose + scale * action`, clipped to +/-1.
The default pose is the robot's home keyframe, so a zero action is a standing robot rather than
a collapse. Physics runs at 200 Hz and the policy at 50, so every decision is held for four
substeps.

## Training: asymmetric, and deliberately not distilled

Training uses **FlashSAC**, a distributional soft actor-critic, on 4096 parallel environments
for 15k iterations — a bit under an hour on one modern GPU for the shipped runs.

The critic is privileged. It sees the same terms as the actor but with the ground-truth versions
substituted in (true angular velocity and true gravity instead of the simulated IMU, plus terms
the robot has no sensor for). The actor sees only the noisy deployable set. A value function
that knows the true state gives much cleaner gradients, and since the critic is discarded at
export, nothing it knows has to exist on hardware.

The obvious way to build a deployable policy from a privileged one is teacher-student
distillation, and this codebase has that path — the classes ending in `L2T`. It was measured
against the simpler alternative and lost. Behaviour cloning caps the student at the conditional
mean of the teacher given the student's observations, and that cap bites exactly where you care:
near-zero commands, and under a heavy payload. Training the deployable actor directly against
the privileged critic has no such ceiling. On mass-bias sweeps up to 1.3x body mass the direct
policy did not fall at all, where the distilled one fell 32% of the time. The shipped policies
take the direct path; `use_distilled_student=False` is the single flag that selects it.

A separate two-layer head predicts base linear velocity from the encoder latent, trained by MSE.
The navigation layer uses it at deployment, since the robot cannot measure body velocity. Its
gradient is **stopped** before the encoder. Letting it through degrades lateral stability on the
real robot: training commands keep `vy` near zero almost always, so the y component of that MSE
is a weak and degenerate signal, and it drags the encoder's sideways representation with it.

## Rewards

19 terms. They fall into four groups, and only the first is about the task:

- **Track the command.** `track_linear_velocity`, `track_angular_velocity`. Both are Gaussian
  kernels whose width narrows as the command gets smaller, so standing still is scored as
  precisely as sprinting.
- **Stay a humanoid.** `upright`, `pose`, `body_ang_vel`, `dof_pos_limits`, `self_collisions`.
- **Walk like something with feet.** `air_time`, `foot_phase_contact_match`,
  `foot_stance_slip_penalty`, `foot_contact_balance`, `foot_clearance`, `foot_swing_height`,
  `foot_slip`, `foot_orientation`, `foot_impact_velocity`. Most of these exist because a policy
  scored only on tracking will find some shuffling, scraping gait that tracks beautifully and
  destroys the hardware.
- **Cost.** `action_rate_l2`, `joint_torque`, `joint_power`. These are also what the paper's
  energy column measures.

Several weights are on a curriculum, ramping in over the first few thousand iterations. Penalties
imposed from step zero suppress the exploration that finds walking in the first place.

Episodes end on `time_out`, `fell_over`, or `illegal_contact`.

## Domain randomization

13 event terms, resampled per episode: `foot_friction`, `pd_gains`, `body_mass`,
`com_displacement`, `ee_payload`, `motor_strength`, `joint_frictionloss`, `joint_damping`,
`push_robot`, `randomize_gait_period`, `reset_base`, `reset_robot_joints`, `physics_recompute`.

The IMU is modelled separately and more carefully than a noise term, with delay, mounting
rotation, bias and drift, because attitude error is what actually ends a real run.

Randomization applies to the **actor's** observations only. The critic reads ground truth. This
is easy to get wrong by changing a default in the wrong place, and the failure is silent: the
policy still trains, it just quietly stops being deployable.

## Experiment classes

`experiments.py` is the list of trainable variants, one class each, selected by `--task`. A class
sets structural flags as class attributes (applied before the environment is built) and tunes
the MDP in `configure()` (applied after). Subclassing an existing variant and overriding one
value is the normal way to add one:

```python
class HumanoidRmaVelEstArmFlashSacMyVariant(HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam):
    """One sentence saying what changed and why."""
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].params["std_min"] = 0.05
```

Fair warning about that file: it is a research log, not a tidy catalogue. Class names are
version tags, docstrings carry the verdict that variant earned — including the ones that were
rejected, and the reason — and the inheritance chains are deep because each experiment changed
exactly one thing from its parent. Some docstrings reference internal notes that are not part of
this release. Kept as-is on purpose. The rejected variants are the more useful half of the file,
and a cleaned-up version would be a list of things that worked with no record of what didn't.

The three policies shipped in this release:

| `--task` | What it is |
| --- | --- |
| `HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam` | two actuated camera modules, the adopted design |
| `HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam` | one module |
| `G1RmaVelEstArmFlashSacStudentOnlyg1bsk2` | the Unitree G1 baseline |

The welded-camera configurations in the paper are these same weights with the camera joints
frozen, not separate training runs.

## Watching one, and training one

```bash
python mj_envs/run.py play  --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam
python mj_envs/run.py train --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam
```

`play` needs no checkpoint argument. It looks in `runs/<task>/`, then at the committed deploy
export, then at the pinned weights that ship with this repository, and prints which one it took.

Never train under an alias class. Aliases point at whatever the current best is; a run started
under one writes `runs/<alias>/`, and the moment the alias is retargeted that directory is a trap
that loads a wrong-architecture checkpoint. Train under the concrete class name.

## Getting a policy onto the robot

Training writes a checkpoint. The robot does not read checkpoints — it reads a *deploy run
directory*, which `utils/export_util.py` produces:

```
deploy/runs/<task>/
  policy_deployed.pt    TorchScript: observation normalizer + actor, fused
  env_config.yaml       obs term order and widths, action scale, default pose, joint order
  robot.xml + meshes    the calibrated model
```

`env_config.yaml` is the contract between the two repositories. It records the exact observation
layout the exported policy expects — which is why the table at the top of this page can be read
straight out of a shipped run directory rather than trusted from prose. The deploy stack
assembles its 120 numbers in that order and in no other. Nothing checks this at runtime beyond
the total width, so a reordered term is a policy that runs and behaves subtly wrong.

The `deploy` repository ships two such run directories under `control/legged_env_bundle/`, which
is why the robot side needs no checkout of this repository at all.

## Files

| File | What |
| --- | --- |
| `humanoid_velocity_env_cfg.py` | the environment: scene, physics, observation groups, rewards, events, terminations, curricula |
| `experiments.py` | every trainable variant, and the record of which ones worked |
| `observation.py` | observation terms, including the IMU model |
| `reward.py` | all 19 reward terms |
| `event.py` | resets, pushes, domain randomization |
| `command.py` | how velocity commands are sampled, including the curricula that widen them |
| `action.py` | the joint position action, with an optional low-pass filter |

The G1 baseline is the same structure under `tasks/g1_velocity/`. Training internals — the
replay buffer, the distributional critic, the Muon optimizer — are in `mj_envs/flash_sac/`.
