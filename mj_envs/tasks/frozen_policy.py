"""Load a FROZEN, deployable FlashSAC policy (e.g. v83) for in-env composition.

Used by the active-vision camera stack (plan/ACTIVE_VISION_PICK_PLACE_PLAN.md):
the camera learner runs ON TOP of a frozen loco+arm policy. The frozen policy
must act on the CAMERA env's live observation each step, so we cannot just step a
separate env — we need the frozen net itself, applied to our obs.

Approach (verified against run.py:_load_trained_policy + runner.get_inference_policy):
  - Use a supplied compatible live env when one already exists; otherwise build a tiny SHAPE-ORACLE env
    (the frozen policy's standard config, 2 envs) so FlashSACRunner.setup() can read obs/action shapes +
    strided indices.
  - Rebuild the FlashSACConfig from the checkpoint's saved args, load the
    checkpoint, and take `runner.get_inference_policy()`. The returned `policy_fn`
    closes over the loaded student net + normalizer + student_strided_idx and is
    BATCH-AGNOSTIC — it applies to any env's obs window of the matching feature
    layout, so the oracle env is never stepped after setup (negligible memory).

The frozen policy is L2T: its deployable is the STUDENT, keyed on the student obs
group, stateless given the (B, L, D) student history window (the window lives in
the consuming env's obs manager and is reset per-env by `_reset_idx` natively).
So the caller feeds `policy_fn({"student": <its own v83 student-obs window>})`.
"""

from __future__ import annotations

import os
import sys

import torch

# mj_envs on path (run.py is launched from there; mirror that for standalone use).
_MJ_ENVS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MJ_ENVS not in sys.path:
    sys.path.insert(0, _MJ_ENVS)


# Sentinel so ``reach_camera_actuator_names=None`` (skip the gaze holder) is distinguishable from
# "argument not supplied" (use the default _CAMERA_PATTERN).
_UNSET = object()

# The parallel-gripper driven joints are ``<L|R>_left_rack_y`` (``right_rack_y`` mimics via equality,
# no actuator). This negative-lookahead excludes them from every POLICY-facing term so the frozen
# policy -- trained on the WELDED robot that has no rack DOFs -- sees the exact obs/action layout it
# was trained on. The racks are driven ONLY by the separate zero-dim ``runtime_gripper`` term.
_NO_RACK = r"(?!.*rack_y)"                 # prepend to a fullmatch pattern to drop the gripper racks
_GRIPPER_ACTUATOR_PATTERN = r".*left_rack_y"   # matches the two driven rack actuators (L_/R_)


def build_configured_env_cfg(task: str, num_envs: int, device: str = "cuda:0", reach: bool = False, *,
                             reach_nominal: bool = False,
                             reach_actuated_gripper: bool = False,
                             reach_gripper_robot_cfg=None,
                             reach_arm_actuator_names=None,
                             reach_camera_actuator_names=_UNSET):
    """Build a fully-configured env_cfg exactly as the FlashSAC train path does.

    `get_env_cfg` only runs `build_env_cfg` and SKIPS `exp.configure(...)`, which is
    where obs-group history is set (e.g. `actor.history_length=20`). The frozen
    policy's runner.setup asserts a sequence actor obs, so we must mirror
    run.run_train (apply_experiment → base_env → create_agent_cfg → exp.configure).

    reach: when True, apply the pick-place reach mutations (formerly the deleted `...CamReach`
    subclass, commit 8fd4408). Makes arm_ref + camera_ref externally-settable passive holders (a
    scripted IK controller writes them; the frozen policy tracks them), zeros the twist (static
    reach, standing base), and pins the base spawn to origin/+x. Default False keeps the
    camera-learner path unchanged.

    Reach-only keyword seams (ignored unless ``reach``):
      reach_nominal: strip domain randomization -- drop interval/perturbation events + observation
        noise so the dynamics reach reads a clean tracking signal, not a DR'd one.
      reach_actuated_gripper: swap in a parallel-gripper-equipped robot and register the zero-dim
        ``runtime_gripper`` external position-action term on the rack joints (direct position
        control, NOT an RL action dim). Requires ``reach_gripper_robot_cfg`` (the caller knows the
        robot). The gripper racks are excluded from every policy obs/action term so the frozen
        (welded-trained) layout is preserved.
      reach_gripper_robot_cfg: the gripper-equipped ``EntityCfg`` to graft as ``scene.entities["robot"]``.
      reach_arm_actuator_names: override the arm holder / arm-residual actuator pattern (g1 differs
        from humanoid). None -> humanoid ``_ARM_ONLY_PATTERN``. Racks are always additionally excluded.
      reach_camera_actuator_names: None -> SKIP the camera_ref gaze holder (fixed-camera robots like
        g1 have no cam_* actuators). _UNSET (default) -> humanoid ``_CAMERA_PATTERN``. A tuple -> use it.

    Returns (env_cfg, exp).
    """
    from run import TrainConfig, apply_experiment, base_env, create_agent_cfg

    cfg = TrainConfig(task=task, num_envs=num_envs, device=device, algo="flash_sac")
    exp = apply_experiment(task, cfg, silent=True)
    if exp is None:
        raise ValueError(f"experiment {task!r} not found in registry")
    env_cfg = base_env(cfg, exp=exp, enable_reward_curriculum=True)
    agent_cfg = create_agent_cfg(cfg.task, cfg.max_iterations, cfg.run_name)
    exp.configure(env_cfg, agent_cfg)   # sets actor/student history → sequence obs
    if reach:
        _apply_reach_mutations(
            env_cfg, nominal=reach_nominal, actuated_gripper=reach_actuated_gripper,
            gripper_robot_cfg=reach_gripper_robot_cfg, arm_actuator_names=reach_arm_actuator_names,
            camera_actuator_names=reach_camera_actuator_names)
    env_cfg.scene.num_envs = num_envs
    return env_cfg, exp


def _apply_reach_mutations(env_cfg, *, nominal=False, actuated_gripper=False, gripper_robot_cfg=None,
                           arm_actuator_names=None, camera_actuator_names=_UNSET) -> None:
    """Re-home the deleted `...CamReach` subclass mutations (8fd4408) onto a configured env_cfg,
    plus the actuated-gripper + fixed-camera + nominal seams the reconstructed callers pass.

    Order matters: the robot swap (step 0) must run BEFORE any term is re-scoped, since the
    re-scoped terms resolve their actuator/joint patterns against the swapped-in robot.
    """
    from tasks.joint_ref_command import PassiveArmRefCommandTermCfg, PassiveGazeCommandTermCfg
    from tasks.humanoid_velocity.experiments import _ARM_ONLY_PATTERN, _CAMERA_PATTERN
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    # 0. Actuated gripper: graft the gripper-equipped robot so the rack DOFs exist for the
    # runtime_gripper term, then EXCLUDE those racks from every policy-facing term (proprio obs +
    # arm residual action) so the frozen (welded-trained) obs/action layout is byte-identical.
    if actuated_gripper:
        assert gripper_robot_cfg is not None, (
            "reach_actuated_gripper=True requires reach_gripper_robot_cfg (the caller supplies the "
            "gripper-equipped robot EntityCfg -- it knows whether this is v2 or g1)")
        env_cfg.scene.entities["robot"] = gripper_robot_cfg
        # Exclude racks from the proprio joint obs in EVERY group -- a StudentOnly policy (g1) reads the
        # ``student`` group, not ``actor``, so scoping only actor would still leak the racks into what the
        # policy actually sees. Scope wherever joint_pos/joint_vel appear (actor, student, critic, ...).
        non_rack = SceneEntityCfg("robot", joint_names=[_NO_RACK + r".*"])
        for group in env_cfg.observations.values():
            for term in ("joint_pos", "joint_vel"):
                if term in group.terms:
                    group.terms[term].params["asset_cfg"] = non_rack
        # Keep the frozen arm residual on the SAME joints it was trained on (racks excluded).
        arm_action = env_cfg.actions.get("joint_pos_arms")
        if arm_action is not None:
            arm_action.actuator_names = tuple(_NO_RACK + p for p in arm_action.actuator_names)
        # The zero-dim direct position-control term (set_targets(open|closed); NOT an RL action).
        from tasks.legs_only_task import ExternalJointPositionActionCfg
        env_cfg.actions["runtime_gripper"] = ExternalJointPositionActionCfg(
            entity_name="robot", actuator_names=(_GRIPPER_ACTUATOR_PATTERN,), use_default_offset=True)

    # 1. arm_ref -> passive holder (scripted controller writes; frozen policy tracks). Its target set
    # MUST equal the frozen arm-residual action ``joint_pos_arms`` (JointPosRefResidualAction asserts
    # equal order). So mirror THAT action's own pattern -- robot-agnostic (humanoid + g1 use different
    # arm regexes) and already rack-excluded above. Explicit override wins; else fall back to the
    # humanoid arm pattern (non-gripper reach paths keep their prior behavior).
    arm_action = env_cfg.actions.get("joint_pos_arms")
    if arm_actuator_names is not None:
        arm_names = tuple(_NO_RACK + p for p in arm_actuator_names)
    elif actuated_gripper and arm_action is not None:
        arm_names = tuple(arm_action.actuator_names)          # already rack-excluded, matches the action
    else:
        arm_names = (_NO_RACK + _ARM_ONLY_PATTERN,)
    env_cfg.commands["arm_ref"] = PassiveArmRefCommandTermCfg(
        entity_name="robot", actuator_names=arm_names)

    # 2. camera_ref -> passive gaze holder, UNLESS the robot has no gimbal (fixed-camera g1 passes None).
    if camera_actuator_names is not None:
        cam_names = (_CAMERA_PATTERN,) if camera_actuator_names is _UNSET else tuple(camera_actuator_names)
        env_cfg.commands["camera_ref"] = PassiveGazeCommandTermCfg(
            entity_name="robot", actuator_names=cam_names)

    # 3. Static reach: zero the twist command (standing base).
    env_cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands["twist"].rel_standing_envs = 1.0

    # 4. Pin base spawn to origin facing +x (world-fixed cube stays reachable; deterministic envs).
    env_cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}

    # 5. Nominal (dynamics-eval) reach: no stochasticity. Keep only deterministic reset mechanics;
    # every other event is DR, including the humanoid gait-clock reset. The parked reach policy has
    # zero twist and does not need a gait phase; without its reset event the existing observation term
    # returns its contract-preserving zero vector. Clear noise in every observation group because G1's
    # StudentOnly policy reads ``student``, not ``actor``. IMUModel has corruption inside its own cfg,
    # independent of ObservationTermCfg.noise, so disable that too.
    if nominal:
        env_cfg.events = {
            n: t for n, t in env_cfg.events.items()
            if n in ("reset_base", "reset_robot_joints")
        }
        env_cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
        env_cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
        for group in env_cfg.observations.values():
            for term in group.terms.values():
                term.noise = None
                imu_cfg = term.params.get("imu_cfg")
                if imu_cfg is not None:
                    imu_cfg.apply_corruption = False


def load_frozen_policy(
    task: str,
    checkpoint: str,
    device: str = "cuda:0",
    num_oracle_envs: int = 2,
    close_oracle: bool = True,
    shape_env=None,
):
    """Return (policy_fn, student_obs_group, action_dim).

    Args:
        task: registered experiment name of the frozen policy (e.g.
            "HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam").
        checkpoint: path to the FlashSAC `.pt` checkpoint.
        device: torch device.
        num_oracle_envs: tiny shape-oracle env size (>=1; 2 is safe) when ``shape_env`` is absent.
        close_oracle: free the oracle sim after building the policy (policy_fn does
            not reference the env). Set False to keep it for debugging.
        shape_env: existing compatible environment from which to derive network shapes. This avoids a
            redundant CUDA/MuJoCo simulation when an inference harness has already built its live env.

    Returns:
        policy_fn: callable obs_dict -> (B, action_dim) frozen action.
        student_obs_group: the obs-group key policy_fn reads (caller remaps to it).
        action_dim: full action dimension the frozen policy outputs.
    """
    from flash_sac.env_wrapper import ManagerBasedRlEnvWithFinalObs
    from flash_sac.runner import FlashSACRunner
    from flash_sac.config import FlashSACConfig

    oracle = shape_env
    owns_oracle = oracle is None
    if owns_oracle:
        env_cfg, _ = build_configured_env_cfg(task, num_oracle_envs, device)
        oracle = ManagerBasedRlEnvWithFinalObs(cfg=env_cfg, device=device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sac_cfg = FlashSACConfig.from_saved_args(ckpt.get("args", {}))
    sac_cfg.logger = "none"  # no wandb in inference
    runner = FlashSACRunner(oracle.unwrapped, sac_cfg, ".", device)
    runner.setup()
    if owns_oracle:
        runner.load(checkpoint)
    else:
        # Reach adds diagnostic-only critic features (e.g. driven gripper state), so its critic shape need
        # not equal the training critic. Deployment calls only actor/student + its observation normalizer;
        # loading qnets would reject a harmless critic mismatch and force construction of a second oracle.
        if ckpt.get("format") != runner.CHECKPOINT_FORMAT:
            raise ValueError(f"{checkpoint} is not a {runner.CHECKPOINT_FORMAT} checkpoint")
        runner.actor.load_state_dict(ckpt["networks"]["actor"])
        if ckpt["normalizers"].get("obs"):
            runner.obs_normalizer.load_state_dict(ckpt["normalizers"]["obs"])
        student = ckpt.get("student")
        if runner.student is not None and student:
            runner.student.load_state_dict(student["actor"])
            if student.get("obs_normalizer"):
                runner.student_obs_normalizer.load_state_dict(student["obs_normalizer"])
        runner.global_step = ckpt.get("global_step", 0)
        runner.env.common_step_counter = runner.global_step * runner.cfg.num_collect_steps
        print(f"[FlashSAC] Loaded inference actor from {checkpoint} at step {runner.global_step}")

    policy_fn = runner.get_inference_policy(device=device)
    student_group = sac_cfg.student_obs_group if runner.student is not None else sac_cfg.actor_obs_group
    action_dim = runner.n_act

    if owns_oracle and close_oracle:
        oracle.close()
    return policy_fn, student_group, action_dim


# --------------------------------------------------------------------------- #
# Smoke test: load v83, run its student net on real v83 obs, assert 31D finite.
#   python mj_envs/tasks/frozen_policy.py [checkpoint.pt]
# --------------------------------------------------------------------------- #
def _smoke(checkpoint: str | None = None) -> None:
    from flash_sac.env_wrapper import ManagerBasedRlEnvWithFinalObs
    from flash_sac.runner import FlashSACRunner
    from flash_sac.config import FlashSACConfig

    task = "HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam"
    if checkpoint is None:
        run_dir = os.path.join(
            _MJ_ENVS, "..", "runs", task, "2026-06-22_19-26-07_flash_sac"
        )
        checkpoint = os.path.join(run_dir, "model_0015000.pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] task={task}\n[smoke] ckpt={checkpoint}\n[smoke] device={device}")

    # Build oracle ourselves (keep it) so we can reset() and feed REAL v83 obs.
    env_cfg, _ = build_configured_env_cfg(task, 2, device)
    oracle = ManagerBasedRlEnvWithFinalObs(cfg=env_cfg, device=device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    sac_cfg = FlashSACConfig.from_saved_args(ckpt.get("args", {}))
    sac_cfg.logger = "none"
    runner = FlashSACRunner(oracle.unwrapped, sac_cfg, ".", device)
    runner.setup()
    runner.load(checkpoint)
    policy_fn = runner.get_inference_policy(device=device)
    student_group = sac_cfg.student_obs_group if runner.student is not None else sac_cfg.actor_obs_group
    print(f"[smoke] is_l2t={runner.student is not None}  student_group={student_group!r}  n_act={runner.n_act}")
    print(f"[smoke] obs groups: {list(oracle.observation_manager.compute().keys())}")

    obs, _ = oracle.reset()
    window = obs[student_group]
    print(f"[smoke] student window shape={tuple(window.shape)}")
    # Caller-side remap: policy_fn extracts its student group key from the dict.
    act = policy_fn({student_group: window})
    print(f"[smoke] frozen action shape={tuple(act.shape)}  finite={torch.isfinite(act).all().item()}")
    assert act.shape[0] == 2, act.shape
    assert act.shape[1] == runner.n_act, act.shape
    assert torch.isfinite(act).all()

    # Camera = last 4 of 31; assert the term order (legs -> arms -> camera).
    term_dims = oracle.action_manager.action_term_dim
    term_names = oracle.action_manager.active_terms
    print(f"[smoke] action terms: {list(zip(term_names, term_dims))}  total={oracle.action_manager.total_action_dim}")
    assert term_names[-1] == "joint_pos_camera", term_names
    assert term_dims[-1] == 4, term_dims

    oracle.close()
    print("frozen_policy smoke: PASS")


if __name__ == "__main__":
    _smoke(sys.argv[1] if len(sys.argv) > 1 else None)
