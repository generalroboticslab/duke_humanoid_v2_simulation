"""Experiment configurations for the G1 velocity tasks (auto-discovered).

Created in Phase 1 of the experiment-only env dispatch refactor: the g1_custom / g1_legs_only /
mjlab G1 tasks formerly had no experiment class and were reachable only as bare --task strings.
Each is now a base experiment carrying its env factory via build_env_cfg.

CLI name = class name (no hyphens): the mjlab tasks are launched as --task MjlabG1Flat /
MjlabG1Rough. The underlying task *string* keeps the original Mjlab-Velocity-*-Unitree-G1 value
so the mjlab loader and any cfg.task checks are unaffected.
"""

from __future__ import annotations

from typing import Final

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from utils.experiments import BaseExperiment

# Curriculum callbacks fire every _CURRICULUM_DECIMATION control steps (matches
# humanoid_velocity/experiments.py's constant -- see that file for rationale).
_CURRICULUM_DECIMATION: Final[int] = 1200
# Same G1-correct leg/arm split used by g1_velocity_env_cfg.py / g1_legs_only_env_cfg.py.
_LEG_PATTERN: Final[str] = r"(waist_.*_joint|.*_hip_.*|.*_knee_.*|.*_ankle_.*)"
_ARM_PATTERN: Final[str] = f"^(?!{_LEG_PATTERN}).*"
_CAMERA_PATTERN: Final[str] = r"cam_(yaw|pitch)_(left|right)"
# Excludes both leg joints AND camera joints -- required when head_camera="actuated" so the
# arm graph-nav command (which loads arm_collision_graph_unitree_g1.pt, lacking camera joints)
# does not receive camera actuator names and raise a ValueError.
_ARM_ONLY_PATTERN: Final[str] = rf"^(?!{_LEG_PATTERN})(?!cam_).*"
_SAFE_LIN_VEL_RANGE: Final[tuple[float, float]] = (-0.8, 0.8)
_SAFE_ANG_VEL_Z_RANGE: Final[tuple[float, float]] = (-0.5, 0.5)
_WARMUP_LIN_VEL_RANGE: Final[tuple[float, float]] = (-0.3, 0.3)
_WARMUP_ANG_VEL_Z_RANGE: Final[tuple[float, float]] = (-0.2, 0.2)
_MID_LIN_VEL_RANGE: Final[tuple[float, float]] = (-0.5, 0.5)
_MID_ANG_VEL_Z_RANGE: Final[tuple[float, float]] = (-0.3, 0.3)
_G1_LEG_STD_WALKING: Final[dict[str, float]] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
}
_G1_LEG_STD_RUNNING: Final[dict[str, float]] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
}


class G1Custom(BaseExperiment):
    """G1 flat-terrain velocity with reduced hip-pitch variance (smaller steps)."""

    task = "g1_custom"

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from tasks.g1_velocity.g1_velocity_env_cfg import G1_VELOCITY_SMALL_STEPS_ENV_CFG
        return G1_VELOCITY_SMALL_STEPS_ENV_CFG(play=play)


class G1LegsOnly(BaseExperiment):
    """G1 legs-only: legs+waist control/observation, arms randomized at safe poses."""

    task = "g1_legs_only"
    randomize_arms = True

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from tasks.g1_velocity.g1_legs_only_env_cfg import G1_LEGS_ONLY_ENV_CFG
        return G1_LEGS_ONLY_ENV_CFG(randomize_arms=self.randomize_arms)


class MjlabG1Flat(BaseExperiment):
    """mjlab Unitree G1 flat-terrain velocity tracking."""

    task = "Mjlab-Velocity-Flat-Unitree-G1"

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
        return unitree_g1_flat_env_cfg(play=play)


class MjlabG1Rough(BaseExperiment):
    """mjlab Unitree G1 rough-terrain velocity tracking."""

    task = "Mjlab-Velocity-Rough-Unitree-G1"

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
        return unitree_g1_rough_env_cfg(play=play)


class G1RmaVelEstArmFlashSacL2T(BaseExperiment):
    """G1 baseline replicating humanoid_v21's HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam
    lineage (minus the camera and the freeze-teacher tune -- see the ActuatedCam subclass and
    `plan/indexed-watching-kite.md` for both): RMA-CNN velocity estimator, collision-free arm
    graph-nav reference + residual action, and Learn-to-Teach (L2T) FlashSAC distillation.

    "Direct final composition": ports the humanoid lineage's cumulative FINAL values as one
    class instead of replaying its ~30-class incremental history (v9/v15/v18/v24 HP tunes, v29
    graph-nav migration, v30 relative-std tracking, v59L2T full L2T stack -- all inlined below).
    None of these HPs are re-derived for G1; they are literal starting values, ported per
    `feedback_one_change_per_experiment` (a G1-specific sweep is separate future work).

    Requires `arm_collision_graph_unitree_g1.pt` (built via
    `python -m mj_envs.asset_zoo.scripts.build_arm_collision_graph --robot unitree_g1`) before
    `arm_ref` can resolve a path.
    """

    task = "g1_manipulation"
    algo = "flash_sac"
    num_steps_per_env: int = 4
    num_envs: int = 4096

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from tasks.g1_velocity.g1_velocity_env_cfg import g1_manipulation_env_cfg
        return g1_manipulation_env_cfg(play=play, head_camera="builtin")

    def _configure_base(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        """Everything through the "v30-equivalent" state: RMA-CNN + estimator head, energy/HP
        tunes, relative-std velocity tracking, collision-free arm graph-nav. No camera, no L2T
        teacher-privileged additions -- those are layered in `configure()` below, onto the REAL
        env only. Called on a throwaway env too, so its resulting `observations["actor"]` can be
        snapshotted as the deployable L2T student BEFORE the privileged additions exist (mirrors
        humanoid_v21's v59L2T using a fresh v30-configured throwaway env for the same purpose).
        """
        import re

        from mjlab.envs import mdp
        from mjlab.managers.curriculum_manager import CurriculumTermCfg
        from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.managers.termination_manager import TerminationTermCfg

        from asset_zoo.g1.g1_constants import get_g1_action_scale
        from tasks.humanoid_velocity import event as task_event
        from tasks.humanoid_velocity.command import PiecewiseLinearVelocityRangeCurriculum
        from tasks.humanoid_velocity.observation import target_arm_joint_pos
        from tasks.humanoid_velocity.reward import (
            actuator_force_reward,
            arm_action_l2,
            arm_joint_tracking,
            feet_clearance_tanh,
            joint_power_reward,
            reward_weight_linear,
            track_linear_velocity_relative,
        )
        from tasks.joint_ref_command import (
            CommandGatedArmGraphRefCommandTermCfg,
            JointPosRefResidualActionCfg,
        )

        N = self.num_steps_per_env

        # Final G1 keeper command curriculum (V10): slow basin first, then expand to the
        # hardware-safe cap. These stage values reproduce the V10 run that reached full cap
        # by 15k iters; the earlier `*24` thresholds never fired, while later strict humanoid
        # timing reached the cap too early and reproduced the fall cliff.
        env.commands["twist"].ranges.lin_vel_x = _WARMUP_LIN_VEL_RANGE
        env.commands["twist"].ranges.lin_vel_y = _WARMUP_LIN_VEL_RANGE
        env.commands["twist"].ranges.ang_vel_z = _WARMUP_ANG_VEL_Z_RANGE
        env.curriculum["command_vel"].func = PiecewiseLinearVelocityRangeCurriculum
        env.curriculum["command_vel"].params["velocity_stages"] = [
            {
                "step": 0,
                "lin_vel_x": _WARMUP_LIN_VEL_RANGE,
                "lin_vel_y": _WARMUP_LIN_VEL_RANGE,
                "ang_vel_z": _WARMUP_ANG_VEL_Z_RANGE,
            },
            {
                "step": 8000,
                "lin_vel_x": _MID_LIN_VEL_RANGE,
                "lin_vel_y": _MID_LIN_VEL_RANGE,
                "ang_vel_z": _MID_ANG_VEL_Z_RANGE,
            },
            {
                "step": 12000,
                "lin_vel_x": _SAFE_LIN_VEL_RANGE,
                "lin_vel_y": _SAFE_LIN_VEL_RANGE,
                "ang_vel_z": _SAFE_ANG_VEL_Z_RANGE,
            },
        ]
        # Match humanoid_v21-style root reset randomization while keeping G1's balanced
        # keyframe as the nominal pose. z has no random lift/drop; height comes from init_state.
        env.events["reset_base"].params["pose_range"]["x"] = (-0.5, 0.5)
        env.events["reset_base"].params["pose_range"]["y"] = (-0.5, 0.5)
        env.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)
        env.events["reset_base"].params["pose_range"]["yaw"] = (-3.14, 3.14)
        # Keep startup/domain randomization active, close to humanoid_v21 settings. Encoder bias
        # is G1-native (humanoid_v21 has no matching term), so preserve the stock G1 range.
        env.events["foot_friction"].params["ranges"] = (0.3, 1.5)
        env.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
        env.events["base_com"].params["ranges"] = {
            0: (-0.01, 0.01),
            1: (-0.01, 0.01),
            2: (-0.01, 0.01),
        }
        env.events["ee_com_offset"].params["ranges"] = (-0.005, 0.005)

        # Pose reward must supervise G1 legs+waist only. Arm/camera joints are driven by
        # moving graph-nav/gaze references below; leaving them in default-pose reward zeros
        # the posture term and removes the nominal bent-knee stabilizing signal. This mirrors
        # humanoid_v21's arm-graph lineage while using G1-native joint names/stds.
        env.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(_LEG_PATTERN,)
        )
        env.rewards["pose"].weight = 2.0
        env.rewards["pose"].params["std_walking"] = _G1_LEG_STD_WALKING
        env.rewards["pose"].params["std_running"] = _G1_LEG_STD_RUNNING
        env.rewards["upright"].weight = 2.0
        env.rewards["body_ang_vel"].weight = -0.15
        env.rewards["angular_momentum"].weight = -0.05
        env.rewards["action_rate_l2"].weight = -0.2

        # RMA-CNN + concurrent velocity-estimator head (HumanoidVelocityRMACNNShortEstimator's 4
        # itemized pieces only -- not that method's humanoid-specific air-time/foot-friction DR
        # tuning, superseded below by the FlashSAC/L2T final values anyway).
        env.observations["estimator_target"] = ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"},
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
        env.observations["actor"].history_length = 20
        env.observations["actor"].flatten_history_dim = False
        agent.actor.class_name = "ActorCriticHistory"
        agent.actor.encoder_type = "rma_cnn"

        # Energy penalties -- V10 keeper relaxed torque relative to humanoid L2T teacher energy.
        # V11/V12 restored humanoid-strength teacher energy and collapsed again, so keep the
        # G1-specific torque budget that actually survived full command expansion.
        env.rewards["joint_torque"] = RewardTermCfg(
            func=actuator_force_reward, weight=-1e-3, params={"threshold_ratio": 0.5},
        )
        env.curriculum["joint_torque"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "joint_torque",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step": 500 * N, "weight": -1e-5},
                    {"step": 3000 * N, "weight": -2.5e-4},
                ],
            },
        )
        env.rewards["joint_power"] = RewardTermCfg(func=joint_power_reward, weight=-2e-4)
        env.curriculum["joint_power"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "joint_power",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step": 1000 * N, "weight": -1e-5},
                    {"step": 3000 * N, "weight": -2e-4},
                ],
            },
        )
        # air_time reward itself is already stock G1 (weight=0.0 default); align curriculum
        # timing and command_threshold to humanoid_v21's validated values (stock G1's
        # command_threshold=0.5 gates air_time out below fast commands; humanoid_v21 lowers
        # it to 0.05 so slow-walking gait still gets swing-time shaping).
        env.rewards["air_time"].params["command_threshold"] = 0.05
        env.curriculum["air_time"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "air_time",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step": 1000 * N, "weight": 0.5},
                    {"step": 5000 * N, "weight": 0.2},
                ],
            },
        )

        # foot_clearance: G1 stock uses mjlab's linear-velocity-gated `feet_clearance`
        # (cost -> ~0 when foot drags near-zero horizontal velocity, weight=-2.0). Replace
        # with humanoid_v21's validated `feet_clearance_tanh` (tanh saturates fast, so slow
        # dragging still costs -- docstring: "remains active even at lower speeds ... prevents
        # cheap foot dragging"). Reuse the already-resolved asset_cfg (G1 foot site_names set
        # by unitree_g1_flat_env_cfg()).
        # target_height 0.15, not stock 0.1: measured peak_height_mean sits ~0.024m across
        # every prior G1 variant -- 24% of even the stock 0.1 target, so the penalty ramp alone
        # isn't lifting feet. Raising the target directly rewards higher clearance instead of
        # just steepening a penalty the policy is already ignoring.
        old_clearance = env.rewards["foot_clearance"]
        env.rewards["foot_clearance"] = RewardTermCfg(
            func=feet_clearance_tanh,
            weight=-1.0,
            params={
                "target_height": 0.15,
                "tanh_scale": 4.0,
                "command_name": old_clearance.params["command_name"],
                "command_threshold": old_clearance.params["command_threshold"],
                "asset_cfg": old_clearance.params["asset_cfg"],
            },
        )

        # foot_swing_height target_height 0.15 too (was stock 0.1, untouched until now) --
        # same rationale as foot_clearance above, kept consistent across both clearance terms.
        env.rewards["foot_swing_height"].params["target_height"] = 0.15

        # foot_swing_height: G1 had no curriculum (static -0.25 for all 15k iters, measured
        # peak_height_mean ~0.024m against 0.1m target). Port humanoid_v21's ramp verbatim --
        # their own tuning note: "-0.5 at ep1000, peak_height ~0.027m -- penalty too weak,"
        # hence the extension to -1.0 by 5000N.
        env.curriculum["foot_swing_height"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "foot_swing_height",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step": 0, "weight": -0.25},
                    {"step": 1000 * N, "weight": -0.50},
                    {"step": 5000 * N, "weight": -1.00},
                ],
            },
        )

        # HP tunes carried as fixed starting values (v9/v15/v18 -- no re-derivation for G1).
        env.rewards["track_angular_velocity"].weight = 2.5
        env.terminations["velocity_tracking_failure"] = TerminationTermCfg(
            func=task_event.VelocityTrackingFailure,
            params={
                "cmd_xy_threshold": 0.1, "tracking_ratio": 0.3,
                "consecutive_steps": 70, "command_name": "twist",
            },
        )

        # v30: relative-std linear-velocity tracking (low-vel dead-zone fix).
        track_lin_vel = env.rewards["track_linear_velocity"]
        track_lin_vel.func = track_linear_velocity_relative
        track_lin_vel.params.pop("std", None)
        track_lin_vel.params["std_rel"] = 0.5
        track_lin_vel.params["std_min"] = 0.3

        # v29: collision-free arm graph-nav reference (command) + thin residual action.
        action_scale = get_g1_action_scale(end_effector="welded", hand="parallel_gripper")
        arm_scale = {k: v for k, v in action_scale.items() if re.search(_ARM_PATTERN, k)}
        env.commands["arm_ref"] = CommandGatedArmGraphRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_ARM_PATTERN,),
            robot_name="unitree_g1",
            steps_per_edge=50,
            locomotion_vel_threshold=0.3,
            loco_min_steps=80,
            command_name="twist",
        )
        env.actions["joint_pos_arms"] = JointPosRefResidualActionCfg(
            entity_name="robot",
            actuator_names=(_ARM_PATTERN,),
            scale=arm_scale,
            use_default_offset=False,
            command_name="arm_ref",
        )
        env.rewards["arm_joint_tracking"] = RewardTermCfg(
            func=arm_joint_tracking,
            weight=2.0,
            params={"std": 0.3, "command_name": "arm_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        env.rewards["arm_action_l2"] = RewardTermCfg(
            func=arm_action_l2, weight=-0.5, params={"arm_action_name": "joint_pos_arms"},
        )
        env.curriculum["arm_action_l2"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "arm_action_l2",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step": 500 * N, "weight": -0.01},
                    {"step": 5000 * N, "weight": -0.5},
                ],
            },
        )
        env.observations["actor"].terms["target_arm_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "arm_ref"},
        )

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        import copy
        import dataclasses

        from mjlab.envs import mdp
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.tasks.velocity import mdp as vel_mdp

        from tasks.g1_velocity.g1_velocity_env_cfg import g1_manipulation_env_cfg
        from tasks.humanoid_velocity.command import IntegralErrorVelocityCommandCfg, integral_error_obs
        from tasks.humanoid_velocity.observation import joint_actuator_force
        from tasks.humanoid_velocity.reward import track_integral_error

        # (1) v59L2T: capture the plain (pre-privileged) actor obs on a throwaway env -- the
        # deployable student.
        student_env = g1_manipulation_env_cfg(head_camera="builtin")
        self._configure_base(student_env, copy.deepcopy(agent))
        student_group = copy.deepcopy(student_env.observations["actor"])

        # (2) v30-equivalent chain on the real env.
        self._configure_base(env, agent)

        # (3) v30Teacher: 2x energy penalties (already final, set in step 2) + privileged clean
        # actor obs (no corruption).
        env.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"})
        env.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
            func=vel_mdp.projected_gravity)
        env.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})
        env.observations["actor"].terms["foot_height"] = ObservationTermCfg(
            func=vel_mdp.foot_height, params={"sensor_name": "foot_height_scan"})
        env.observations["actor"].terms["foot_air_time"] = ObservationTermCfg(
            func=vel_mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"})
        env.observations["actor"].terms["foot_contact"] = ObservationTermCfg(
            func=vel_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"})
        env.observations["actor"].terms["foot_contact_forces"] = ObservationTermCfg(
            func=vel_mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"})
        env.observations["actor"].terms["joint_torque"] = ObservationTermCfg(func=joint_actuator_force)
        env.observations["actor"].enable_corruption = False

        # (4) v51Teacher: clean integral-error command (dr_bias=dr_noise=0) + integral obs
        # (critic+actor) + track_integral_error reward.
        old = env.commands["twist"]
        env.commands["twist"] = IntegralErrorVelocityCommandCfg(
            **{f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init},
            dr_bias=(0.0, 0.0, 0.0),
            dr_noise=(0.0, 0.0, 0.0),
        )
        env.observations["critic"].terms["command_integral"] = ObservationTermCfg(
            func=integral_error_obs, params={"command_name": "twist"},
        )
        env.observations["actor"].terms["command_integral"] = ObservationTermCfg(
            func=integral_error_obs, params={"command_name": "twist"},
        )
        env.rewards["track_integral_error"] = RewardTermCfg(
            func=track_integral_error,
            weight=1.0,
            params={"command_name": "twist", "std": 0.3},
        )

        # (5) v51L2T: graft the deploy-realistic-noise student group.
        student_group.enable_corruption = True
        env.observations["student"] = student_group

    def flash_sac_configure(self, sac_cfg) -> None:
        sac_cfg.num_collect_steps = 4
        sac_cfg.num_updates = 3

        sac_cfg.use_sequence_encoder = True
        sac_cfg.use_velocity_estimator = False   # v30Teacher: actor sees GT lin_vel, no head

        sac_cfg.log_std_min = -3.0
        sac_cfg.normalize_reward = True
        sac_cfg.normalized_G_max = 5.0
        sac_cfg.v_min = -5.0
        sac_cfg.v_max = 5.0
        sac_cfg.num_atoms = 101
        sac_cfg.target_sigma = 0.15

        sac_cfg.sample_chunk_size = 2
        sac_cfg.num_steps = 3
        sac_cfg.alpha_init = 0.01
        sac_cfg.alpha_learning_rate = 2e-4
        sac_cfg.tau = 0.01
        sac_cfg.use_zeta_noise = True
        sac_cfg.zeta_mu = 2.0
        sac_cfg.zeta_max_n = 16
        sac_cfg.buffer_size = 256
        sac_cfg.batch_size = 4096
        sac_cfg.use_per_q_target = True

        sac_cfg.use_distilled_student = True     # v51L2T
        sac_cfg.student_obs_group = "student"    # v51L2T
        sac_cfg.student_action_prob_max = 0.5    # v54L2T: DAgger mixing ceiling
        sac_cfg.frame_ring_history = True        # v54L2T
        sac_cfg.episode_level_mixing = True      # v59L2T: per-episode driver draw


class G1RmaVelEstArmFlashSacStudentOnly(G1RmaVelEstArmFlashSacL2T):
    """NO-TEACHER BASELINE for G1RmaVelEstArmFlashSacL2T: train the deployable proprio policy DIRECTLY
    with SAC, no privileged teacher, no distillation, no sample-mixing/freeze curriculum.

    Mirrors `HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam` on G1. Env is BYTE-IDENTICAL to
    G1RmaVelEstArmFlashSacL2T (same rewards incl. track_integral_error + energy penalties, same 31D
    action space, same arm graph-ref + integral + relative-std tracking, same privileged critic group).
    The ONLY change is the training path: RL actor = deployable `student` group (corrupted proprio
    + arm command), no teacher distillation. Privileged critic (GT lin/ang vel, foot contacts,
    integral, projected gravity) stays unchanged → asymmetric actor-critic.

    Mechanism. `use_distilled_student=False` makes the runner skip ALL student machinery
    (self.student=None; imitation/mix/freeze gated off). `actor_obs_group="student"` trains the
    standard SAC actor on the deploy group; deploy export is that actor. The `actor` (teacher) obs
    group is still built by the env (inherited configure step (5)) but unread — harmless compute.
    freeze_teacher_after_iters / student_start_iters are no-ops here (gated on use_distilled_student)
    but reset to 0 for a clean config dump.

    Energy penalties stay at G1L2T's V10 values (joint_torque=-1e-3, joint_power=-2e-4 — NOT the
    human v83 1x-energy reset, since G1 V11/V12 tried human-strength teacher energy and collapsed).
    Keeping G1-specific torque budget.

    Inheritance: ...→G1RmaVelEstArmFlashSacL2T→G1RmaVelEstArmFlashSacStudentOnly.
    Env inherited verbatim (student obs group already built by parent configure step (5));
    only flash_sac_configure diverges.

    *** INITIAL BASELINE 2026-07-19 ***. Single-seed 15k iter launch pending; metrics TBD.
    Future work: reward-tune off this baseline (mirroring humanoid v158-v159 wave) for tracking +
    gait quality on the G1 +2kg manipulator body.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)       # G1 L2T stack (distilled student, frame-ring, DAgger)
        sac_cfg.use_distilled_student = False      # remove teacher/student distillation
        sac_cfg.actor_obs_group = "student"        # RL actor = deployable proprio group (asymmetric; critic stays privileged)
        sac_cfg.student_start_iters = 0            # no-op w/o distillation; reset for clean config


class G1RmaVelEstArmFlashSacL2TActuatedCam(G1RmaVelEstArmFlashSacL2T):
    """G1RmaVelEstArmFlashSacL2T + actuated head-camera gaze, mirroring
    HumanoidRmaVelEstArmFlashSacv59L2TActuatedCam. Both teacher (actor) and deployable student
    output the 4 camera dims and observe the gaze command.

    Deviation from v83L2TActuatedCam (humanoid_v21): does NOT set
    `sac_cfg.freeze_teacher_after_iters` -- that value came from a humanoid_v21-specific
    freeze-iter bracket sweep, a G1-untested magic number outside this baseline's scope (see
    `plan/indexed-watching-kite.md` "Out of scope").
    """

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from tasks.g1_velocity.g1_velocity_env_cfg import g1_manipulation_env_cfg
        return g1_manipulation_env_cfg(play=play, head_camera="actuated")

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        import re

        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        from asset_zoo.g1.g1_constants import get_g1_action_scale
        from tasks.humanoid_velocity.observation import target_arm_joint_pos
        from tasks.humanoid_velocity.reward import arm_joint_tracking
        from tasks.joint_ref_command import (
            CommandGatedArmGraphRefCommandTermCfg,
            GazeRefCommandTermCfg,
            JointPosRefResidualActionCfg,
        )

        # 1. Arm reference rebuilt with _ARM_ONLY_PATTERN (exclude camera joints -- the collision
        #    graph has none). Robot entity already has the camera actuators (build_env_cfg above
        #    swapped head_camera="actuated"); same CommandGatedArmGraphRefCommandTerm params
        #    otherwise.
        action_scale = get_g1_action_scale(end_effector="welded", hand="parallel_gripper")
        arm_scale = {k: v for k, v in action_scale.items() if re.search(_ARM_ONLY_PATTERN, k)}
        env.commands["arm_ref"] = CommandGatedArmGraphRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_ARM_ONLY_PATTERN,),
            robot_name="unitree_g1",
            steps_per_edge=50,
            locomotion_vel_threshold=0.3,
            loco_min_steps=80,
            command_name="twist",
        )
        env.actions["joint_pos_arms"] = JointPosRefResidualActionCfg(
            entity_name="robot",
            actuator_names=(_ARM_ONLY_PATTERN,),
            scale=arm_scale,
            use_default_offset=False,
            command_name="arm_ref",
        )

        # 2. Camera joint reference = random-walk gaze; residual action on top.
        env.commands["camera_ref"] = GazeRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            min_steps=30,
            max_steps=120,
            yaw_range=4.7124,
            pitch_range=1.5708,
            yaw_walk=1.0,
            pitch_walk=0.4,
        )
        env.actions["joint_pos_camera"] = JointPosRefResidualActionCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            scale={"cam_yaw.*": 0.3, "cam_pitch.*": 0.2},
            use_default_offset=False,
            command_name="camera_ref",
        )

        # 3. Camera tracking reward (reuses arm_joint_tracking against the camera_ref command).
        env.rewards["camera_joint_tracking"] = RewardTermCfg(
            func=arm_joint_tracking,
            weight=1.0,
            params={"std": 0.3, "command_name": "camera_ref", "asset_cfg": SceneEntityCfg("robot")},
        )

        # 4. Gaze command obs for BOTH teacher (actor) and student (deploy policy).
        env.observations["actor"].terms["target_camera_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "camera_ref"})
        env.observations["student"].terms["target_camera_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "camera_ref"})

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.student_start_iters = 3000


class G1RmaVelEstArmFlashSacStudentOnlyActuatedCam(G1RmaVelEstArmFlashSacL2TActuatedCam):
    """NO-TEACHER BASELINE for G1RmaVelEstArmFlashSacL2TActuatedCam. Same as
    G1RmaVelEstArmFlashSacStudentOnly but with the actuated head-camera gaze + arm graph-ref
    (matches `HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam` lineage on humanoid_v21).

    Env is BYTE-IDENTICAL to G1RmaVelEstArmFlashSacL2TActuatedCam (same rewards, same arm + cam
    ref commands, same 35D action space = 31D arm/leg + 4D camera, same privileged critic group).
    Training path: RL actor = deployable `student` group, no teacher distillation. Privileged
    critic (GT lin/ang vel, foot contacts, integral, projected gravity) stays → asymmetric AC.

    Inheritance: ...→G1RmaVelEstArmFlashSacL2T→G1RmaVelEstArmFlashSacL2TActuatedCam
                                  →G1RmaVelEstArmFlashSacStudentOnlyActuatedCam.
    Env inherited verbatim (student obs group already built by parent configure); only
    flash_sac_configure diverges.

    *** INITIAL BASELINE 2026-07-19 ***. Single-seed 15k iter launch pending; metrics TBD.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.use_distilled_student = False      # remove teacher/student distillation
        sac_cfg.actor_obs_group = "student"        # RL actor = deployable proprio group (asymmetric; critic stays privileged)
        sac_cfg.student_start_iters = 0            # no-op w/o distillation; reset for clean config


# ─────────────────────────────────────────────────────────────────────────────
# Reward-tuning wave off G1RmaVelEstArmFlashSacStudentOnly (2026-07-19).
# 6 single-var levers mirroring humanoid v158-v159 structure. Per "no new reward terms" +
# "one variable per experiment" constraints. Each subclass overrides configure() after super()
# to set ONE additional value vs the StudentOnly baseline.
# ─────────────────────────────────────────────────────────────────────────────


class G1RmaVelEstArmFlashSacStudentOnlyg1v1(G1RmaVelEstArmFlashSacStudentOnly):
    """StudentOnly + foot_clearance target_height raised (0.15 -> 0.20). Single lever.

    Motivation (gait-naturalness wave off G1 StudentOnly baseline, 2026-07-19): the StudentOnly
    baseline inherits V10's foot_clearance target_height=0.15 raise (measured peak_height_mean
    ~0.024m against stock 0.1, so V10 doubled to 0.15). This lever pushes target_height further
    (0.15 -> 0.20) to test whether the policy can sustain higher foot swing on G1 + manipulator.
    Target_height axis is untested beyond V10. Mirrors humanoid v158a (foot_clearance target
    0.1 -> 0.15 -- DEAD-LEVER per humanoid wave evidence). Expected: may regress on G1 too.

    Single lever vs StudentOnly: ONLY foot_clearance target_height. Weight, tanh_scale, command_threshold
    unchanged. No other term touched.

    Inheritance: ...->G1RmaVelEstArmFlashSacL2T->G1RmaVelEstArmFlashSacStudentOnly->g1v1.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_clearance"].params["target_height"] = 0.20


class G1RmaVelEstArmFlashSacStudentOnlyg1v2(G1RmaVelEstArmFlashSacStudentOnly):
    """StudentOnly + foot_clearance weight doubled (-1.0 -> -2.0). Single lever. **G1 DEPLOY KEEPER.**

    Motivation (gait-naturalness wave off G1 StudentOnly baseline, 2026-07-19): the StudentOnly
    baseline inherits V10's foot_clearance weight=-1.0 (tanh form, target 0.15, scale 4.0). Doubling
    the weight to -2.0 makes the policy commit to 0.15m swings or pay the deviation cost every step.
    Mirrors HUMANOID WINNER `v159b` (foot_clearance weight -1.0 -> -2.0 = +2kg-robot deploy keeper).
    On G1 + manipulator this lever is untested -- the G1 + parallel_gripper body has different
    dynamics vs humanoid_v21, so the same axis may behave differently. If this wins on G1, it
    becomes the G1 deploy keeper.

    RESULT (2026-07-19, 2-seed confirm): fell 0.128 (2-seed mean, vs L2T baseline 0.450-0.700),
    exy 0.522, eyaw 0.710. Mirrors humanoid v159b EXACTLY — same single lever wins on two distinct
    robot morphologies (cross-morphology robust principle). The `g1_best` alias below points to
    this class as the G1 deploy keeper; update `g1_best`'s parent to retarget the alias without
    touching downstream callers.

    Single lever vs StudentOnly: ONLY foot_clearance weight. Target_height, tanh_scale,
    command_threshold, foot_swing_height, foot_orientation, action_rate, integral all unchanged.

    Inheritance: ...->G1RmaVelEstArmFlashSacL2T->G1RmaVelEstArmFlashSacStudentOnly->g1v2.
    the G1 deploy keeper (2026-07-19, 2-seed confirmed).
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_clearance"].weight = -2.0


class G1RmaVelEstArmFlashSacStudentOnlyg1v3(G1RmaVelEstArmFlashSacStudentOnly):
    """StudentOnly + foot_swing_height target_height raised (0.15 -> 0.20). Single lever.

    Motivation (gait-naturalness wave off G1 StudentOnly baseline, 2026-07-19): the StudentOnly
    baseline inherits V10's foot_swing_height target_height=0.15 (same V10 raise as foot_clearance).
    This lever pushes target_height further (0.15 -> 0.20) to test peak swing commitment on G1 +
    manipulator. Mirrors humanoid v161a (foot_swing_height target 0.1 -> 0.15 -- DEAD-LEVER per
    humanoid wave evidence).

    Single lever vs StudentOnly: ONLY foot_swing_height target_height. Curriculum (-0.25/-0.5/-1.0),
    weight, command_threshold unchanged. foot_clearance untouched.

    Inheritance: ...->G1RmaVelEstArmFlashSacL2T->G1RmaVelEstArmFlashSacStudentOnly->g1v3.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_swing_height"].params["target_height"] = 0.20


class G1RmaVelEstArmFlashSacStudentOnlyg1v5(G1RmaVelEstArmFlashSacStudentOnly):
    """StudentOnly + air_time curriculum endpoint doubled (0.2 -> 0.4). Single lever.

    Motivation (natural-gait wave off G1 StudentOnly baseline, 2026-07-19): the StudentOnly baseline
    inherits V10's air_time curriculum 0.5 -> 0.2 (steps 1000N/5000N). Doubling the convergence
    endpoint to 0.4 increases air-time pressure at convergence = longer committed strides.
    Mirrors humanoid v151 (air_time weight doubled 0.2 -> 0.4 -- REJECT as bouncy gait, gait is
    BOUNCY with air +4% and yaw worse). Likely DEAD-LEVER on G1 too -- confirmed by humanoid
    evidence; testing for G1-specific effect.

    Single lever vs StudentOnly: ONLY air_time curriculum endpoint. Weight, command_threshold,
    all stages with endpoint doubled.

    Inheritance: ...->G1RmaVelEstArmFlashSacL2T->G1RmaVelEstArmFlashSacStudentOnly->g1v5.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        N = self.num_steps_per_env
        env.curriculum["air_time"].params["weight_stages"] = [
            {"step": 1000 * N, "weight": 0.5},
            {"step": 5000 * N, "weight": 0.4},
        ]


class G1RmaVelEstArmFlashSacStudentOnlyg1v6(G1RmaVelEstArmFlashSacStudentOnly):
    """StudentOnly + body_ang_vel penalty relaxed (-0.15 -> -0.10). Single lever.

    Motivation (yaw-tracking wave off G1 StudentOnly baseline, 2026-07-19): body_ang_vel penalizes
    trunk angular velocity, which can oppose commanded yaw rate. On G1 + manipulator, the parallel
    gripper adds mass to the trunk, increasing angular inertia. Relaxing -0.15 -> -0.10 frees trunk
    yaw velocity for tracking. Mirrors humanoid v155 (body_ang_vel halved -0.05 -> -0.025 --
    REJECT, trunk wobble doubled falls). G1's penalty is 3x human (v145 -0.05) because of gripper
    mass, so the relaxation is proportionally smaller (33% cut vs 50% on humanoid). Likely
    DEAD-LEVER on G1 too, but G1's different dynamics may allow it.

    Single lever vs StudentOnly: ONLY body_ang_vel weight. No curriculum on this term. All
    other terms unchanged.

    Inheritance: ...->G1RmaVelEstArmFlashSacL2T->G1RmaVelEstArmFlashSacStudentOnly->g1v6.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["body_ang_vel"].weight = -0.10


class G1RmaVelEstArmFlashSacStudentOnlyg1b1(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + foot_clearance tanh_scale sharpened (4.0 -> 6.0). Single lever.

    Motivation (stability/tracking wave off g1v2 deploy keeper, 2026-07-20): g1v2 doubled
    foot_clearance weight (deviation pressure at target 0.15m, G1's V10 target). tanh_scale
    controls the SHARPNESS of the deviation ramp around the target -- 4.0 is moderate, 6.0
    forces commitment to be NEAR 0.15m. With weight=-2.0 already committed, sharpening
    asks "closer to target", orthogonal to "stronger weight". Pure reward-only.

    Single lever vs g1v2: ONLY foot_clearance.params.tanh_scale. Weight, target, foot_swing_height,
    air_time, body_ang_vel, action_rate all unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b1.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_clearance"].params["tanh_scale"] = 6.0


class G1RmaVelEstArmFlashSacStudentOnlyg1b2(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + body_ang_vel penalty tightened (-0.15 -> -0.25, ~1.67x). Single lever.

    Motivation (stability wave off g1v2 deploy keeper, 2026-07-20): g1v6 relaxed body_ang_vel
    -0.15 -> -0.10 (frees yaw). This lever does the OPPOSITE: tighten trunk wobble (roll+pitch)
    for stability. With foot_clearance=-2.0 already committing to swings, tightening ang_vel
    asks "stable trunk + committed swing". Mirrors humanoid v2b (-0.01 -> -0.05) on G1's
    3x-higher starting penalty (G1 needs more ang_vel pressure because of gripper inertia).

    Single lever vs g1v2: ONLY body_ang_vel weight. No curriculum on this term.

    Inheritance: ...->StudentOnly->g1v2->g1b2.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["body_ang_vel"].weight = -0.25


class G1RmaVelEstArmFlashSacStudentOnlyg1b3(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + foot_clearance target raised (0.15 -> 0.20, ~1.33x). Single lever.

    Motivation (gait wave off g1v2 deploy keeper, 2026-07-20): g1v1 (target 0.20 against
    StudentOnly baseline) was middle-rank at iter 15000 (fell 0.150, exy 0.522). NEW PREMISE:
    test against g1v2 (weight=-2.0 base), where the policy is already doubly committed to
    whatever target. Maybe the higher target + doubled weight compound to commit to taller
    swings. G1's measured peak_height_mean sits ~0.024m -- both 0.15 and 0.20 targets are
    aspirational; the lever asks "what if we aim higher".

    Single lever vs g1v2: ONLY foot_clearance.params.target_height. Weight (-2.0), tanh_scale
    unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b3.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_clearance"].params["target_height"] = 0.20


class G1RmaVelEstArmFlashSacStudentOnlyg1ArmV2(G1RmaVelEstArmFlashSacStudentOnly):
    """Camera-free G1 StudentOnly bundle porting selected `v2_best` configuration.

    Decision history (2026-07-21): `v2_best` is the explicit alias for
    `HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam`, not v2fa. A canonical W&B two-seed audit
    found v159b vs v2fa: reward 123.63 vs 122.79, fell_over 0.0075 vs 0.0275, XY error 0.5567
    vs 0.5629, and yaw error 0.4732 vs 0.4850. v2fa removed only already-rare VTF events and
    visually made repeated active gait corrections; v159b looked calmer and more stable. Thus
    this class ports v159b-derived settings, not v2fa's VTF `cmd_xy_threshold=0.05` change.

    G1 reference: `G1RmaVelEstArmFlashSacStudentOnlyg1b3` is the no-camera G1 keeper, with
    two-seed fell=0.105, XY error=0.524, and yaw error=0.712. It is StudentOnly -> g1v2 -> g1b3.
    This class deliberately inherits raw `G1RmaVelEstArmFlashSacStudentOnly` instead, so every
    keeper delta is visible here and a fresh reader does not need to reconstruct its MRO.

    Relative to StudentOnly, this deliberately confounded user-requested bundle applies:
      1. g1v2: foot_clearance weight -1.0 -> -2.0.
      2. g1b3: foot_clearance target_height 0.15 -> 0.20.
      3. v2_best: action_rate_l2 -0.20 -> -0.125 (less action smoothing pressure).
      4. v2_best: body_ang_vel -0.15 -> -0.05 (less trunk angular-velocity penalty).
      5. v2 schedule: command curriculum stages at step 2000 and 5000*num_steps_per_env.
      6. v2 mixed arms: per-episode frozen-home vs graph-nav-swing Bernoulli population split.

    What is intentionally retained from G1: G1's higher 0.20m clearance target, its G1 robot,
    action scales, pose/energy/DR configuration, StudentOnly actor path, camera-free action and
    observation layout, VTF settings (`cmd_xy_threshold=0.1`, `tracking_ratio=0.3`,
    `consecutive_steps=70`), and `rel_standing_envs`. No camera configuration or v2fa termination
    setting is imported.

    Command safety: StudentOnly normally grows XY command range +/-0.3 -> +/-0.5 -> +/-0.8 at
    steps 0/8000/12000. This port uses v2 stage timing, but every stage is fixed at the hardware-safe
    G1 cap +/-0.8m/s XY and +/-0.5rad/s yaw. v2's default terminal +/-1.0m/s range is forbidden
    by the project safety constraint and is intentionally not copied. Result: this bundle starts
    with the final safe G1 range rather than a slow-range gait warmup.

    Arm curriculum semantics: resets sample frozen-home with probability home_ratio; frozen arms
    stay at default joint positions and skip locomotion graph-nav gating, while swing arms use the
    usual G1 graph-nav reference. home_ratio linearly anneals 1.0 -> 0.5 over 72000 environment
    steps (18000 FlashSAC iterations at four steps/iteration). A standard 15k run ends at about
    0.583, so the trained policy never reaches or holds the intended 50/50 floor. Checkpoint load
    restores `common_step_counter`, preserving this phase during play/evaluation.

    Interpretation boundary: G1's isolated zero-command attempts already rejected tighter
    body-angular/action-rate penalties; this bundle moves in the opposite, looser direction and
    combines it with a command-distribution and arm-regime change. It is not an ablation and does
    not test a zero-command hypothesis. Compare only aggregate behavior against StudentOnly, g1v2,
    g1b3, and v2_best. No training result or promotion claim exists at class creation.

    RESULT — REJECTED 2026-07-21 (single-seed 15k iter, grl3, runs/G1RmaVelEstArmFlashSacStudentOnlyg1ArmV2/2026-07-21_12-58-12_flash_sac/):
      term_fell_over       0.20  vs g1v2 0.13  +54%   ← disqualifying
      twist_error_vel_xy   0.4845 vs g1v2 0.5223  -7%  (within family budget, marginal)
      twist_error_vel_yaw  0.6542 vs g1v2 0.7111  -8%  (within family budget, marginal)
      ep_reward            140.40 vs g1v2 129.39  +8.4%  ← inflated by looser penalties
      raw_action_rate_l2   1.705 vs g1v2 1.561   +9%   (both above §5.2 >1.5 hardware-safety red flag)
      deterministic eval   fell 0.22 (vs g1v2 0.19, g1b3 0.23), mean disp 4.15m (vs g1v2 4.70)

    Reward gain decomposes as `looser penalties`, not new capability:
      action_rate_l2 -0.20→-0.125:  +0.10 per ep (relaxed)
      arm_action_l2 lighter penalty: +0.24 per ep (v2 mixed-arms frozen-home regime still mid-transition at 15k)
      arm_joint_tracking:             +0.17 per ep
      foot_clearance target 0.20 amplified by -2.0 weight:  -0.09 per ep
      net per-episode: +0.42. ep_reward gain +11 = aggregate over horizon.

    Dead zone eval §7.1 (cmd 0.05/0.10/0.20/0.30) PASSED all 4 thresholds — no Standing Local
    Optimum, no Slow Shuffle hack. No §6 reward-hacking signatures.

    Verdict: REJECT. Per §5.4 full-reward gate, a policy that gains +8.4% reward and +49% fell_over
    is NOT a better policy — the reward is inflated by relaxing action_rate_l2 (-0.20→-0.125) and
    body_ang_vel (-0.15→-0.05), which transfers v2_best's Humanoid tradeoff to G1 where it doesn't
    hold. v2_best (v159b) was a Humanoid selector (v159b vs v2fa audit), NOT a G1 selector.
    G1's tighter penalties reflect a different morphology. Do NOT retest v2_best transfer as a
    bundle on G1. If v2_best axes are worth testing on G1, isolate them per §8.1.G single-change
    rule (action_rate_l2 only / body_ang_vel only / v2 schedule only / v2 mixed arms only).

    Single-change rule §8.1.G violation: 6 deltas confounded (foot_clearance weight + target_height
    + action_rate_l2 + body_ang_vel + v2 schedule + v2 mixed arms). No causal attribution possible.
    """

    def configure(self, env, agent):
        super().configure(env, agent)

        env.rewards["foot_clearance"].weight = -2.0
        env.rewards["foot_clearance"].params["target_height"] = 0.20
        env.rewards["action_rate_l2"].weight = -0.125
        env.rewards["body_ang_vel"].weight = -0.05

        env.commands["twist"].ranges.lin_vel_x = _SAFE_LIN_VEL_RANGE
        env.commands["twist"].ranges.lin_vel_y = _SAFE_LIN_VEL_RANGE
        env.commands["twist"].ranges.ang_vel_z = _SAFE_ANG_VEL_Z_RANGE
        env.curriculum["command_vel"].params["velocity_stages"] = [
            {
                "step": 2000,
                "lin_vel_x": _SAFE_LIN_VEL_RANGE,
                "lin_vel_y": _SAFE_LIN_VEL_RANGE,
                "ang_vel_z": _SAFE_ANG_VEL_Z_RANGE,
            },
            {
                "step": 5000 * self.num_steps_per_env,
                "lin_vel_x": _SAFE_LIN_VEL_RANGE,
                "lin_vel_y": _SAFE_LIN_VEL_RANGE,
                "ang_vel_z": _SAFE_ANG_VEL_Z_RANGE,
            },
        ]

        arm_ref = env.commands["arm_ref"]
        arm_ref.home_ratio_start = 1.0
        arm_ref.home_ratio_end = 0.5
        arm_ref.home_ratio_curriculum_steps = 18000 * self.num_steps_per_env


class G1RmaVelEstArmFlashSacStudentOnlyg1u1(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + upright reward std tightened (sqrt(0.2) -> sqrt(0.1)). Single lever.

    New premise (2026-07-21): g1v2 ep_upright=1.911 (raw 0.955) is at the exp-term ceiling,
    yet user reports visual torso instability. The steady-state 5.5 deg average is fine; the
    instability is from transient spikes (~17 deg). Tighter std pushes the kernel falloff
    steeper so spikes cost ~23pp more (score 0.638 -> 0.407) while the 5.5 deg average only
    loses ~5pp (0.955 -> 0.905).

    NOT a retest of:
      - v157 (humanoid, std LOOSEN sqrt(0.2)->sqrt(0.3)) — opposite direction
      - v124-v126 / v2hc (humanoid, weight 1.0->2.0/3.0) — weight axis, not std
      - g1c8 (g1b3, upright -> upright_gated) — gating axis, not std
    TIGHTEN direction is an untested gap on both robots.

    Single lever vs g1v2: ONLY upright.params.std. body_ang_vel (-0.15), action_rate_l2 (-0.2),
    upright weight (2.0), foot_clearance weight (-2.0) all unchanged. Inherits torso_link body
    from mjlab stock G1 env_cfg (env_cfgs.py:147-148), so the reward stays on torso.

    Promotion criteria: ep_upright may stay near 0.9 raw (5pp drop OK), but transients should
    improve — measured via deterministic eval at vx=0,vy=0,wz=0 (displacement_m g1v2 0.075).
    Reject if fell_over 0.20 g1v2 baseline regresses, or exy 0.5223 regresses >5%.

    RESULT (2026-07-22, single-seed, runs/G1RmaVelEstArmFlashSacStudentOnlyg1u1/2026-07-21_22-28-03_flash_sac/, @iter 15000):
      Training: reward 127.82 vs g1v2 129.39 (-1.21%, within single-seed noise)
                fell_over 0.125 vs 0.140 (-10.7% AT 15k)
                fell_over 10k+ mean 0.146 vs 0.173 (-15.6% over late-train window)
                ep_upright 1.897 vs 1.918 (-1.10%, expected: tighter kernel = lower score at same posture)
      Deterministic eval (n=20/cmd, grl3 cuda:0):
                cmd 0,0,0:   fell 0.000=0.000  disp 0.070m=0.068m  (TIE — both pass user goal of zero-vel stability)
                cmd 0.5,0,0: fell 0.250 vs 0.300 (-16.7%)  disp 7.30m vs 7.77m (g1u1 slower/shy gait under DR)
                             lin_err 0.075 vs 0.082 (-8.5%, g1u1 closer to cmd)
                cmd 0,0,0.5: fell 0.000=0.000  yaw disp 0.184m vs 0.158m (g1u1 +16% lateral drift while rotating)
      ep_track_linear -2.04% (lost 0.027 to tighter upright); ep_track_angular +0.19% (noise)
      error_vel_xy +2.73% (loss: 0.531 vs 0.517); error_vel_yaw +0.64% (noise)
    VERDICT: REJECTED 2026-07-22 via §7.1 dead-zone sweep.
      §5.4 narrow probe (train fell -16.7% @ 0.5 m/s) passed but full sweep caught:
        cmd 0.05 disp 0.025m (thresh 0.10m FAIL, g1v2 PASS at 0.162m)
        cmd 0.10 disp 0.200m (thresh 0.50m FAIL, g1v2 PASS at 0.542m)
      Tighter upright rewards standing still more at low cmd → policy stiffens instead of
      walking slowly. Per §5.4 "narrow probe win" caveat: NOT a better policy. Alias reverted
      to g1v2 same day. See g1_best docstring for the revert audit. g1u1 class retained as
      dead-lever entry; s1 PID 939172 finished but moot. DO NOT retest TIGHTEN-std direction
      in isolation — pair with opposite-axis fix (relax VTF / add vel-gated reward) for low-cmd.
    """
    def configure(self, env, agent):
        import math
        super().configure(env, agent)
        env.rewards["upright"].params["std"] = math.sqrt(0.1)


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + arm_ref.stand_min_steps=80 (training-time slow-arm lever). Single lever.

    Motivation (Wave-19 anti-sway port to G1, 2026-07-25): humanoid v2ybsk (Wave-19 prom)
    won base-sway obliteration by setting arm_ref.stand_min_steps=80 at training time.
    The knob is a TRAINING-TIME lever — slow-arm graph-nav swing during RL rollouts
    teaches the policy cleaner base anticipation. Deploy reverts to default sms=0 (full
    speed). Same-condition eval (sway_probe --stand-min-steps 0 / eval_policy --stand-min-steps 0)
    confirmed v2ybsk @ sms=0 deploy-equivalent is strictly better than the v2ybdkc keeper
    on fell (-4%), vx0.5 ang_err (-11%), vz0.5 (-4%), stand tied. NOT cheating.

    Adjusted for G1: same fs=80 dose. G1 has the same CommandGatedArmGraphRefCommandTerm
    in joint_ref_command.py — kernel is morphology-agnostic. Mirrors humanoid @ sms=80.

    Single lever vs g1v2: ONLY arm_ref.stand_min_steps=80 (training-time). No reward change,
    no curriculum change, no observation change. Inherits ankle_torque etc from g1v2.

    Promotion criteria (same-condition vs g1v2 baseline):
      sway_probe drift_peak(ACTIVE-FROZEN) better ≥10%.
      sway_probe fell_over, lin_rms, yaw_rms not regressed >2%.
      eval_policy 5-cmd bench (--stand-min-steps 0): fell/disp/lin_err/ang_err/action_rate
        not regressed, action_rate not regressed >5%.
      §7.1 dead-zone cmd 0.05/0.10 disp ≥ thresholds.

    Reject if fell regresses >5% OR dead-zone fails (§7.1). Pre-screen with single seed,
    2-seed confirm before retargeting g1_best.

    Inheritance: ...->StudentOnly->g1v2->g1bsk.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.commands["arm_ref"].stand_min_steps = 80


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk2(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + arm_ref.stand_min_steps=40 (lower dose of g1bsk lever). Single lever.

    Companion to g1bsk (sms=80). Tests lower dose of the Wave-19 slow-arm training-time
    lever. Hypothesis: 40 = enough anticipation signal, less training drag than 80.

    Single lever vs g1v2: ONLY arm_ref.stand_min_steps=40.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.commands["arm_ref"].stand_min_steps = 40


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk2Cosine(
    G1RmaVelEstArmFlashSacStudentOnlyg1bsk2
):
    """g1bsk2 + cosine anneal of actor/critic/alpha over 15000 iters to 5%. Single lever.

    Mirror of the humanoid Cosine promote from the 2026-08-07 Muon campaign: the G1 lineage had
    NEVER trained with any lr schedule (lr_decay_iters defaults 0 and no ancestor overrides it,
    so `actor_scheduler is None`). On the humanoid spine, adding cosine anneal to the AdamW-wide
    path added **+8.49 reward at 2-seed mean (camera-config independent: dual +8.49, single +8.05)**
    with zero train-time cost. That same lever is untested on G1; the +8 it returned on humanoid
    was the campaign's actual finding (Muon's own contribution was inside noise).

    Single change vs g1bsk2: ONLY lr_decay_iters=15000 and lr_min_factor=0.05, set in
    flash_sac_configure so this inherits cleanly from any sibling. No reward / curriculum /
    init_state touched. NOT the Muon-group-only path; this is the AdamW-wide path that the
    humanoid wave-2 control used, which deliberately anneals actor+critic+alpha together.

    PROMOTION BAR: 2-seed vs the existing g1bsk2 checkpoint @ iter 15000. Match the humanoid
    pattern — promote on training return + gait block + fell_over, not on a single metric.
    Reject if fell_over regresses materially (g1bsk2 fell 0.080-0.100 2-seed range).
    """
    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.lr_decay_iters = 15000
        sac_cfg.lr_min_factor = 0.05


class g1_best(G1RmaVelEstArmFlashSacStudentOnlyg1bsk2):
    """Wave-19 G1 PROMOTE (2026-07-25): g1v2 + arm_ref stand_min_steps 0 -> 40 (training-time slow-arm).

    RETARGETED 2026-08-08 to g1bsk2Cosine: added cosine anneal (AdamW-wide, to 5% over 15000 iters)
    on top of g1bsk2. Mirrors the humanoid Cosine promote (2026-08-07 wave 2): this G1 lineage had
    never trained with any lr schedule, and humanoid +8.49 / single-cam +8.05 evidence says the
    schedule is the dominant gain. Single-variable change vs prior g1_best (g1bsk2); see
    G1RmaVelEstArmFlashSacStudentOnlyg1bsk2Cosine docstring for the bar.

    Retargeted from g1v2 to g1bsk2 after same-condition validation. Mirror of humanoid v2ybsk
    Wave-19 PROMOTE pattern: training-time slow-arm graph-nav lever (sms=40) teaches cleaner
    base anticipation. Deploy reverts to default sms=0 (full-speed arm). NOT cheating.

    2-seed confirm @ iter 15000, n=20, sms=0 deploy-equivalent vs g1v2 baseline (iter 15000):
      seed 1 (2026-07-25_14-50-43): sway Δ=-0.0004 (-101%), fell 0.080 (-43%), disp 3.050m (+17%),
                                     lin_err 0.078 (-10%), §7.1 FAIL × 2 (cmd 0.05/0.10 partial).
      seed 2 (2026-07-25_14-54-46): sway Δ=0.0161 (-56%), fell 0.100 (-29%), disp 2.635m (+1%),
                                     lin_err 0.085 (-1%), §7.1 PASS × 4 (clean).

    Seed 2 dominates on every metric. Seed 1 partial dead-zone FAIL = seed variance.

    Mechanism: same as humanoid v2ybsk -- slow arm during training → policy learns cleaner
    anticipation of arm-reaction impulses → base sway collapses at deploy (where arm swings
    full speed). TRAINING-ONLY (NOT a deploy lever): at deploy knob reverts to 0.

    Inheritance: ...->StudentOnly->g1v2->g1bsk2.

    REVERTED 2026-08-08 from g1bsk2Cosine to g1bsk2: dyn11 sweep result P=0.092 vs dyn10
    baseline 0.989 (delta -0.897, catastrophic regression on cuRobo reach-grasp cells). The
    cosine-annealed training does not break the 3-cmd locomotion eval (which passed fell_over
    0/0, disp +14%, VTF -36%) but disrupts cuRobo's hand-object grasp-latch timing that
    bimanual_mixed_* and front_back_far cells depend on. See MEMORY "g1 cosine retarget REJECTED
    at sweep scale 2026-08-08". The g1bsk2Cosine class is retained for reference but is no
    longer in the g1_best alias chain.

    Deploy checkpoints (g1bsk2, 2-seed confirmed 2026-07-25):
      runs/G1RmaVelEstArmFlashSacStudentOnlyg1bsk2/2026-07-25_14-50-43_flash_sac/model_0015000.pt (s1)
      runs/G1RmaVelEstArmFlashSacStudentOnlyg1bsk2/2026-07-25_14-54-46_flash_sac/model_0015000.pt (s2)

    Detail: MEMORY Wave-19 G1 port entry (2026-07-25).
    """
    fallback_checkpoint_to_parent = True


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk2_armh(G1RmaVelEstArmFlashSacStudentOnlyg1bsk2):
    """g1bsk2 (Wave-19 G1 PROMOTE) + custom G1_ARM_HOME arm init pose. Single lever.

    Motivation (user-provided home pose, 2026-07-25): retrain current G1 keeper g1bsk2
    with a specific 14-joint arm home pose (user-provided). Hypothesis: a more deliberate
    arm home (bent elbows, raised shoulders, neutral wrists) might improve manipulation
    tolerance under arm-graph-nav swing by reducing arm-reaction impulse magnitude at the
    start of each trajectory segment. Leg home pose preserved from mjlab HOME_KEYFRAME
    (hip_pitch -0.1, knee 0.3, ankle_pitch -0.2) so walking gait is unchanged.

    User-specified arm home (radians):
      left_shoulder_pitch=1.1447, left_shoulder_roll=0.1185, left_shoulder_yaw=0.1665,
      left_elbow=-0.9863, left_wrist_roll=-0.3769, left_wrist_pitch=-0.0892, left_wrist_yaw=-0.2650
      right_shoulder_pitch=1.1404, right_shoulder_roll=-0.1185, right_shoulder_yaw=-0.1818,
      right_elbow=-0.9829, right_wrist_roll=0.3894, right_wrist_pitch=-0.0852, right_wrist_yaw=0.2729

    Single lever vs g1bsk2: ONLY env.scene.entities["robot"].init_state.joint_pos arm joints
    overridden. stand_min_steps=40 (training-time slow-arm) inherited. No reward / curriculum.

    Promotion criteria: sway drift_peak Δ ≤ 0.016 (g1bsk2 s2 baseline) AND §7.1 dead-zone
    ALL 4 PASS AND 5-cmd fell/lin_err/disp not regressed.
    """
    ARM_HOME_OVERRIDE = {
        "left_shoulder_pitch_joint": 1.1447,
        "left_shoulder_roll_joint": 0.1185,
        "left_shoulder_yaw_joint": 0.1665,
        "left_elbow_joint": -0.9863,
        "left_wrist_roll_joint": -0.3769,
        "left_wrist_pitch_joint": -0.0892,
        "left_wrist_yaw_joint": -0.2650,
        "right_shoulder_pitch_joint": 1.1404,
        "right_shoulder_roll_joint": -0.1185,
        "right_shoulder_yaw_joint": -0.1818,
        "right_elbow_joint": -0.9829,
        "right_wrist_roll_joint": 0.3894,
        "right_wrist_pitch_joint": -0.0852,
        "right_wrist_yaw_joint": 0.2729,
    }
    LEG_HOME_PRESERVED = {
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
    }

    def configure(self, env, agent):
        super().configure(env, agent)
        merged = {}
        merged.update(self.ARM_HOME_OVERRIDE)
        merged.update(self.LEG_HOME_PRESERVED)
        env.scene.entities["robot"].init_state.joint_pos = merged


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk2_armh2(G1RmaVelEstArmFlashSacStudentOnlyg1bsk2):
    """g1bsk2 + user-specified 14-joint arm home pose (bent-elbow, raised-shoulder,
    neutral-wrist). User-specified values are ALL within g1.xml joint limits
    (verified 2026-07-25 against <mujoco_menagerie>/unitree_g1/g1.xml):
      elbow_joint range (-1.0472, 2.0944): -0.9863 OK (0.06 above lower bound)
      shoulder_pitch range (-3.0892, 2.6704): 1.1447 OK
      shoulder_roll range L(-1.5882, 2.2515) R(-2.2515, 1.5882): ±0.1185 OK
      shoulder_yaw range (-2.618, 2.618): ±0.1665 OK
      wrist_roll range (-1.9722, 1.9722): ±0.38 OK
      wrist_pitch/yaw range (-1.6144, 1.6144): ±0.27 OK

    An earlier armh test (2026-07-25 iter 9000) claimed "elbow below joint limit"
    based on a wrong source — that source listed elbow lower bound -0.262, but the
    actual g1.xml menagerie has -1.0472 (the actuator / doc I cross-checked showed
    the actuator-frcrange, not the joint range). Iter 9000 catastrophic
    (drift_peak=1.78m, sway Δ=0.78 vs bar 0.016) was NOT due to elbow clamp — it
    was a genuine bent-elbow home vs arm-graph-nav compatibility issue. This
    class retries with the ORIGINAL user values to re-test that hypothesis cleanly.

    Single lever vs g1bsk2: ONLY env.scene.entities["robot"].init_state.joint_pos
    arm joints overridden (14 exact-match). stand_min_steps=40 inherited. No graph
    modification. Hypothesis: bent-elbow home reduces arm-reaction impulse at
    trajectory segment boundaries (smaller delta to typical manipulation targets
    than extended HOME_KEYFRAME shoulder_pitch=0.2 elbow=1.28).

    Promotion criteria: sway drift_peak Δ ≤ 0.016 (g1bsk2 s2 baseline) AND §7.1
    dead-zone ALL 4 PASS AND 5-cmd fell/lin_err/disp not regressed.
    """
    ARM_HOME_OVERRIDE = {
        "left_shoulder_pitch_joint": 1.1447,
        "left_shoulder_roll_joint": 0.1185,
        "left_shoulder_yaw_joint": 0.1665,
        "left_elbow_joint": -0.9863,
        "left_wrist_roll_joint": -0.3769,
        "left_wrist_pitch_joint": -0.0892,
        "left_wrist_yaw_joint": -0.2650,
        "right_shoulder_pitch_joint": 1.1404,
        "right_shoulder_roll_joint": -0.1185,
        "right_shoulder_yaw_joint": -0.1818,
        "right_elbow_joint": -0.9829,
        "right_wrist_roll_joint": 0.3894,
        "right_wrist_pitch_joint": -0.0852,
        "right_wrist_yaw_joint": 0.2729,
    }
    LEG_HOME_PRESERVED = {
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
    }

    def configure(self, env, agent):
        super().configure(env, agent)
        merged = {}
        merged.update(self.ARM_HOME_OVERRIDE)
        merged.update(self.LEG_HOME_PRESERVED)
        env.scene.entities["robot"].init_state.joint_pos = merged


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk3(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + arm_ref.stand_min_steps=120 (higher dose of g1bsk lever). Single lever.

    Companion to g1bsk (sms=80). Tests ceiling dose of the Wave-19 slow-arm training-time
    lever. Hypothesis: 120 = maximum training drag, may over-fit to slow arm.

    Single lever vs g1v2: ONLY arm_ref.stand_min_steps=120.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.commands["arm_ref"].stand_min_steps = 120


class G1RmaVelEstArmFlashSacStudentOnlyg1bf1(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + upright weight relaxed (2.0 -> 1.5). Single lever.

    Motivation (parental lever probe, 2026-07-25): g1v2 inherits upright weight 2.0 from
    StockG1. Trunk stability is a parent axis. Relaxing to 1.5 trades absolute upright
    for tracking freedom — opposite g1u1 direction (g1u1 tightened std → dead-zone failure).

    Single lever vs g1v2: ONLY upright.weight. No curriculum change.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["upright"].weight = 1.5


class G1RmaVelEstArmFlashSacStudentOnlyg1bf2(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + arm_action_l2 relaxed (-0.1 -> -0.05). Single lever.

    Motivation (parental lever probe, 2026-07-25): G1 inherits arm_action_l2 -0.1 from
    V10. v2_best (v2ybsk) on humanoid uses a lighter arm_action penalty — that's one
    of the v2_best axis contributors. Isolating arm_action alone on G1 (not bundled).

    Single lever vs g1v2: ONLY arm_action_l2.weight. No other change.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["arm_action_l2"].weight = -0.05


class G1RmaVelEstArmFlashSacStudentOnlyg1bf3(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + action_rate_l2 relaxed (-0.20 -> -0.125). Single lever.

    Motivation (parental lever probe, 2026-07-25): v2_best axis on humanoid. g1ArmV2
    bundled this with 5 other deltas and fell +54%. Isolating action_rate alone to
    test whether the smooth-action cost is hurting G1.

    Single lever vs g1v2: ONLY action_rate_l2.weight. No other change.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["action_rate_l2"].weight = -0.125


class G1RmaVelEstArmFlashSacStudentOnlyg1bf4(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + body_ang_vel relaxed (-0.15 -> -0.10). Single lever.

    Motivation (parental lever probe, 2026-07-25): g1v6 (from StudentOnly baseline) tested
    THIS direction and was rejected. Retesting from g1v2 base — different parent, different
    reward combo (g1v2 has foot_clearance=-2.0 commit). g1v6 fail context may not apply.

    Single lever vs g1v2: ONLY body_ang_vel.weight. No other change.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["body_ang_vel"].weight = -0.10


class G1RmaVelEstArmFlashSacStudentOnlyg1bf5(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + action_rate_l2 TIGHTENED (-0.20 -> -0.30, 50% stronger). Single lever.

    Motivation (Wave-19 G1 port TIGHTEN direction, 2026-07-25): g1bf3 relax direction
    (-0.20→-0.125) passed §7.1 dead-zone at iter 7000 but failed at iter 10000 (FAIL × 2:
    cmd 0.10 0.455m, cmd 0.20 1.421m). Mechanism: relaxed action_rate → policy is lazy at
    low cmd → dead-zone regression. TIGHTENING (more penalty) is the unexplored opposite —
    forces aggressive action changes → may improve low-cmd tracking. Single lever from g1v2.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 (≥10% better than g1v2 0.0304) AND
    mean fell/lin_err/disp not regressed AND §7.1 dead-zone ALL 4 PASS.

    Reject if sway not improved OR §7.1 FAIL (any dead-zone regression).
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["action_rate_l2"].weight = -0.30


class G1RmaVelEstArmFlashSacStudentOnlyg1bf6(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + arm_action_l2 TIGHTENED (-0.1 -> -0.15, 50% stronger). Single lever.

    Companion to g1bf5 (action_rate tighten). Tests TIGHTEN direction on arm_action axis.
    Forces arm to settle to commanded pose faster — may reduce arm-reaction transient on base.

    Promotion criteria: same as g1bf5.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["arm_action_l2"].weight = -0.15


class G1RmaVelEstArmFlashSacStudentOnlyg1bf7(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + arm_ref home_ratio_end lowered (0.5 -> 0.3) - more arm-swing training.

    Motivation (curriculum lever, 2026-07-25): all 5 reward levers in Wave-19 G1 port
    REJECTED via §7.1 dead-zone. Reward-axis is closed. Try CURRICULUM axis: home_ratio_end
    controls fraction of envs that train with frozen-arm vs swing-arm resets. Lower end
    = more swing-arm training = policy learns to handle arm-active better. Hypothesis: more
    exposure to arm-active regime might teach better compensation, reducing base sway at deploy.

    Single lever vs g1v2: ONLY arm_ref.home_ratio_end 0.5 -> 0.3. home_ratio_start, curriculum_steps
    unchanged. No reward changes.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 (≥10% better than g1v2 0.0304) AND §7.1
    dead-zone ALL 4 PASS.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.commands["arm_ref"].home_ratio_end = 0.3


class G1RmaVelEstArmFlashSacStudentOnlyg1bf8(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + track_linear_velocity std_rel LOOSENED (0.5 -> 0.7). Single lever.

    Motivation (v2ybdkc mirroring lever, 2026-07-25): humanoid v2ybdkc (current v2_best
    parent of v2ybsk Wave-19 PROMOTE) sets track_angular_velocity std_rel=0.7 -- the
    relative-form mid-cmd basin width. G1 has track_linear_velocity with std_rel=0.5.
    g1b5 (tighten 0.5->0.3) was REJECT on g1v2 base (+41% fell). The OPS direction
    (loosen 0.5->0.7) is UNTESTED. Hypoth: looser basin = wider tracking tolerance at
    mid-cmd = less aggressive compensation = less base sway.

    Single lever vs g1v2: ONLY track_linear_velocity std_rel 0.5 -> 0.7. std_min=0.3, weight,
    command_name unchanged.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 (≥10% better than g1v2 0.0304) AND §7.1
    dead-zone ALL 4 PASS.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["track_linear_velocity"].params["std_rel"] = 0.7


class G1RmaVelEstArmFlashSacStudentOnlyg1bf9(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + upright WEIGHT TIGHTENED (2.0 -> 2.5). Single lever.

    Motivation (post dead-zone research, 2026-07-25): g1u1 REJECTED via §7.1 by
    TIGHTENING std axis (sqrt(0.2) -> sqrt(0.1)). The WEIGHT axis (2.0 -> 2.5) is
    different mechanism -- std changes kernel falloff shape, weight changes magnitude.
    Tighten upright weight = stronger posture pull = better base stability at zero cmd.
    Could improve dead-zone if the policy trades posture for tracking.

    Single lever vs g1v2: ONLY upright.weight 2.0 -> 2.5. std, params, all other terms
    unchanged.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 AND §7.1 dead-zone ALL 4 PASS.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["upright"].weight = 2.5


class G1RmaVelEstArmFlashSacStudentOnlyg1bf10(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + foot_clearance weight PUSHED (-2.0 -> -3.0). Single lever.

    Motivation (winning-axis extension, 2026-07-25): foot_clearance weight -2.0 was the
    g1v2 winner (g1v1 = -1.0 dead, g1v2 = -2.0 alive). The axis is OPEN above -2.0 on G1.
    Humanoid v159a (weight -3.0) tested further push and was NEUTRAL on humanoid. On G1,
    with gripper mass + different morphology, the optimum may sit at higher magnitude.
    Pushing -3.0 tests the boundary.

    Single lever vs g1v2: ONLY foot_clearance.weight -2.0 -> -3.0. target_height (0.15),
    tanh_scale (4.0) unchanged.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 AND §7.1 dead-zone ALL 4 PASS.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_clearance"].weight = -3.0


class G1RmaVelEstArmFlashSacStudentOnlyg1bf11(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + action_rate TIGHTEN (-0.30) + track_linear std_rel LOOSEN (0.7). 2-lever combine.

    Motivation (MEMORY-prescribed dead-zone fix, 2026-07-25): g1u1 REJECTED via §7.1 because
    tighter upright rewards standing still MORE at low cmd. MEMORY prescribes: 'pair with
    opposite-axis (VTF relax / add vel-gated reward) to fix dead-zone first.' This class
    implements that: action_rate TIGHTEN forces aggressive action changes (opposite of
    g1bf3 relax which made policy lazy at low cmd); track_linear std_rel LOOSEN widens
    tracking tolerance (the VTF-relax analog). Both reward axes compensate each other.

    Multi-lever combine violates §8.1.G single-change rule, but is the prescribed fix from
    MEMORY. If both individual levers (g1bf5 tighten action_rate, g1bf8 loosen std_rel)
    were kept, their combination might be the actual cure.

    2 levers vs g1v2: action_rate_l2.weight -0.20 -> -0.30 AND track_linear_velocity
    std_rel 0.5 -> 0.7. All other reward params unchanged.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 AND §7.1 dead-zone ALL 4 PASS.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["action_rate_l2"].weight = -0.30
        env.rewards["track_linear_velocity"].params["std_rel"] = 0.7


class G1RmaVelEstArmFlashSacStudentOnlyg1bf12(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + track_linear_velocity std_min TIGHTEN (0.3 -> 0.2). Single lever.

    Motivation (dead-zone basin tightening, 2026-07-25): g1b6 tested std_min 0.3->0.2
    on g1b3 base (foot_clearance target 0.20). UNTESTED on g1v2 base (foot_clearance -2.0).
    std_min controls BASIN WIDTH at zero cmd (std_eff = max(std_rel*||cmd||, std_min)).
    Tightening std_min = sharper zero-cmd pull = better dead-zone tracking. Counter-
    balances the dead-zone failure of relax-direction levers.

    Single lever vs g1v2: ONLY track_linear_velocity std_min 0.3 -> 0.2. std_rel=0.5, weight
    unchanged.

    Promotion criteria: sway drift_peak Δ ≤ 0.0274 AND §7.1 dead-zone ALL 4 PASS.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["track_linear_velocity"].params["std_min"] = 0.2


class G1RmaVelEstArmFlashSacStudentOnlyg1b4(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + joint_torque penalty doubled (-1e-3 -> -2e-3, 2x). Single lever.

    Motivation (smoothness wave off g1v2 deploy keeper, 2026-07-20): G1 has joint_torque
    curriculum 500N:-1e-5, 3000N:-1e-3 (early relax, late commit). Doubling ALL stages
    proportionally keeps the curriculum SHAPE but doubles steady-state smoothness pressure.
    Single-axis "joint_torque 2x".

    Single lever vs g1v2: ONLY joint_torque weight + curriculum stages (proportionally doubled).
    joint_power untouched.

    Inheritance: ...->StudentOnly->g1v2->g1b4.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        N = self.num_steps_per_env
        env.rewards["joint_torque"].weight = -2e-3
        env.curriculum["joint_torque"].params["weight_stages"] = [
            {"step": 500 * N, "weight": -2e-5},
            {"step": 3000 * N, "weight": -2e-3},
        ]


class G1RmaVelEstArmFlashSacStudentOnlyg1b5(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + track_linear_velocity std_rel tightened (0.5 -> 0.3). Single lever.

    Motivation (tracking wave off g1v2 deploy keeper, 2026-07-20): G1 uses track_linear_velocity
    with std_rel=0.5, std_min=0.3 (relative-std low-vel deadzone fix from v30). Halving std_rel
    to 0.3 tightens tracking tolerance -- smaller std = sharper reward peak = stronger tracking
    pressure at higher speeds (the rel-scale kicks in). Keeps std_min unchanged (still 0.3
    deadzone floor).

    Single lever vs g1v2: ONLY track_linear_velocity std_rel. Weight, std_min, command_name
    unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b5.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["track_linear_velocity"].params["std_rel"] = 0.3


class G1RmaVelEstArmFlashSacStudentOnlyg1b6(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """g1v2 + air_time doubled (initial weight 0.25 -> 0.5 + curriculum stages proportionally). Single lever.

    Motivation (gait wave off g1v2 deploy keeper, 2026-07-20): G1 air_time has initial 0.25
    + curriculum 1000N:0.5, 5000N:0.4. Doubling ALL scales proportionally: initial 0.5,
    1000N:1.0, 5000N:0.8. Single-axis "scale air_time by 2x" -- longer committed strides.
    g1v5 (curriculum endpoint 0.2 -> 0.4) was NEUTRAL on G1 at iter 15000; this lever
    changes the AXIS (weight + all stages), not just the endpoint. Different mechanism.

    Single lever vs g1v2: ONLY air_time (initial weight + curriculum). Threshold_min,
    threshold_max, command_threshold unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b6.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        N = self.num_steps_per_env
        env.rewards["air_time"].weight = 0.5
        env.curriculum["air_time"].params["weight_stages"] = [
            {"step": 1000 * N, "weight": 1.0},
            {"step": 5000 * N, "weight": 0.8},
        ]


class G1RmaVelEstArmFlashSacStudentOnlyg1c1(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + body_ang_vel penalty tightened (-0.15 -> -0.25, ~1.67x). Single lever.

    Motivation (zero-cmd stability wave off g1b3 deploy keeper, 2026-07-21): user reports the
    g1b3 policy is too jittery at zero-velocity command during manipulation, base shifts a lot.
    g1b3 inherits body_ang_vel=-0.15 from V10 (3x humanoid's -0.05 because of gripper mass).
    NEW PREMISE: at zero cmd, tighten the trunk wobble penalty 1.67x for stronger anti-wobble
    pressure during manipulation holds.

    Orthogonality to g1b2: g1b2 tested body_ang_vel -0.15 -> -0.25 on g1v2 base (NOT g1b3),
    fell +17% and was REJECT. g1b3 is a different base (foot_clearance target 0.20 + weight
    -2.0); the additional swing commitment may provide margin for tighter wobble penalty. Worth
    re-screening on the new base.

    Single lever vs g1b3: ONLY body_ang_vel weight. foot_clearance (weight -2.0 + target 0.20),
    upright (2.0), angular_momentum (-0.05), action_rate_l2 (-0.2), all other terms untouched.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c1.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 body_ang_vel weight: -0.15 -> -0.25 (1.67x trunk wobble penalty)
        env.rewards["body_ang_vel"].weight = -0.25


class G1RmaVelEstArmFlashSacStudentOnlyg1c2(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + action_rate_l2 penalty tightened (-0.2 -> -0.4, 2x). Single lever.

    Motivation (zero-cmd stability wave off g1b3 deploy keeper, 2026-07-21): user reports base
    jitter at zero cmd during manipulation. action_rate_l2 penalizes action DELTA; tighter
    action rate = smoother policy output = less base wiggle between control steps. G1 stock
    uses action_rate_l2=-0.1; V10 bumped to -0.2; this lever pushes to -0.4 for explicit
    zero-cmd smoothness.

    Orthogonality: action_rate_l2 is command-independent; tightening helps zero-cmd directly
    without affecting tracking pressure.

    Untested direction: action_rate_l2 tightening on g1b3 base. v153 (humanoid parent of v159b)
    RELAXED action_rate_l2 to -0.125 for tracking freedom; this lever REVERSES that direction
    on g1b3.

    Single lever vs g1b3: ONLY action_rate_l2 weight. body_ang_vel (-0.15), upright (2.0),
    angular_momentum (-0.05), all other terms untouched.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c2.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 action_rate_l2 weight: -0.2 -> -0.4 (2x action smoothness pressure)
        env.rewards["action_rate_l2"].weight = -0.4


class G1RmaVelEstArmFlashSacStudentOnlyg1c3(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + angular_momentum penalty tightened (-0.05 -> -0.10, 2x). Single lever.

    Motivation (zero-cmd stability wave off g1b3 deploy keeper, 2026-07-21): angular_momentum
    penalizes the angular momentum of the root body; a higher penalty reduces rotational
    inertia buildup from swing legs during zero-cmd stance. G1 stock sets -0.02; V10 bumped
    to -0.05; this lever pushes to -0.10 for stronger anti-rotation pressure.

    Orthogonality: angular_momentum is a separate reward from body_ang_vel. body_ang_vel
    penalizes the velocity (first derivative); angular_momentum penalizes the L quantity
    (mass * velocity). Both target base rotation but through different mechanisms.

    Untested direction: angular_momentum axis at higher magnitudes not screened since V10.
    Worth probing.

    Single lever vs g1b3: ONLY angular_momentum weight. body_ang_vel (-0.15), upright (2.0),
    action_rate_l2 (-0.2), all other terms untouched.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c3.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 angular_momentum weight: -0.05 -> -0.10 (2x angular momentum penalty)
        env.rewards["angular_momentum"].weight = -0.10


class G1RmaVelEstArmFlashSacStudentOnlyg1c4(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + track_linear_velocity std_min tightened (0.3 -> 0.2, 33% tighter). Single lever.

    Motivation (zero-cmd stability wave off g1b3 deploy keeper, 2026-07-21): user reports base
    shift during manipulation. track_linear_velocity uses `track_linear_velocity_relative` with
    std_eff = max(std_rel * ||cmd_xy||, std_min). At zero cmd, std_eff = std_min, so std_min
    controls the BASIN WIDTH at zero cmd. Current std_min=0.3; tightening to 0.2 sharpens the
    zero-cmd pull.

    Orthogonality: std_min vs std_rel are independent levers. g1b5 tightened std_rel 0.5->0.3
    on g1v2 base and was REJECT (+41% fell for -4.5% exy). The std_min axis is UNTESTED on G1.

    Risk: docstring warns std_min < 0.15 causes gait transient penalty. 0.2 is between warning
    threshold and current value, probing the safe region.

    Single lever vs g1b3: ONLY track_linear_velocity.params.std_min. Weight (2.0), std_rel
    (0.5), command_name, all other terms untouched.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c4.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 track_linear_velocity std_min: 0.3 -> 0.2 (sharper zero-cmd pull)
        env.rewards["track_linear_velocity"].params["std_min"] = 0.2


class G1RmaVelEstArmFlashSacStudentOnlyg1c5(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + rel_standing_envs raised (0.1 -> 0.6, 6x). Single lever.

    Motivation (zero-cmd stability wave off g1b3, 2026-07-21): all 4 single-var reward-tighten
    levers (g1c1-c4) regressed on fell by +38-114% at 15k iter. NEW PREMISE: change the
    TRAINING DATA distribution. G1 default rel_standing_envs = 0.1 (10% standing). Raising
    to 0.6 (60% standing) gives the policy 6x more zero-cmd transitions in the replay buffer.

    Orthogonality: rel_standing_envs is a TRAINING DATA change, not a reward change. The
    humanoid v159b base uses 0.3 (3x more than g1's 0.1); g1 is currently under-sampled at
    zero cmd. Raising to 0.6 matches the wave's goal of more zero-cmd exposure.

    Risk: too high rel_standing_envs (>0.7) historically hurts tracking closure. 0.6 is
    within the tested range (v123 humanoid 0.3 was the validated level; 0.6 is double).

    Single lever vs g1b3: ONLY rel_standing_envs. foot_clearance (weight -2.0 + target 0.20),
    body_ang_vel (-0.15), action_rate_l2 (-0.2), upright (2.0), all rewards untouched.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c5.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 rel_standing_envs: 0.1 -> 0.6 (6x more zero-cmd exposure in training)
        env.commands["twist"].rel_standing_envs = 0.6


class G1RmaVelEstArmFlashSacStudentOnlyg1c6(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + rel_standing_envs MILDLY raised (0.1 -> 0.3, 3x). Single lever.

    Motivation (zero-cmd stability wave, 2026-07-21): g1c5 (0.1->0.6) regressed on vtf
    (0.755 over 0.20 threshold) — too aggressive. NEW PREMISE: try a milder 3x bump (0.1->0.3)
    that gives more zero-cmd training without breaking vtf closure.

    Orthogonality: same as g1c5 but smaller magnitude. Also matches the validated humanoid
    v159b rel_standing_envs=0.3 level (g1 starts at 0.1, 3x lower than humanoid).

    Risk: small. 0.3 is the validated humanoid v159b level. If humanoid works at 0.3, g1
    at 0.3 should also work.

    Single lever vs g1b3: ONLY rel_standing_envs. All rewards and other params unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c6.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 rel_standing_envs: 0.1 -> 0.3 (3x more zero-cmd exposure, milder than g1c5 0.6)
        env.commands["twist"].rel_standing_envs = 0.3


class G1RmaVelEstArmFlashSacStudentOnlyg1c7(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + body_ang_vel switched to command-gated mode (standing_gain=3x). Single lever.

    Motivation (zero-cmd wave 4, 2026-07-21): g1c1 tightened body_ang_vel UNIFORMLY (-0.15 -> -0.25)
    and regressed fell +114%. NEW PREMISE: the EXISTING body_ang_vel reward has a built-in
    command-gated mode that boosts penalty 3x at low cmd (standing) and relaxes to 1x at high
    cmd (walking). Activating the gated mode adds 3x trunk-wobble pressure at zero cmd without
    affecting walking performance.

    Orthogonality to g1c1: g1c1 applied 1.67x penalty at ALL commands. Gated applies 3x at
    zero cmd only. Different mechanism.

    Single lever vs g1b3: ONLY body_ang_vel.params adds command_name + cmd_std + standing_gain
    + walking_gain. Weight (-0.15), other rewards, all other params unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c7.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 body_ang_vel -> local class (gated mode, 3x penalty at cmd~0, 1x at full cmd)
        from tasks.humanoid_velocity.reward import body_angular_velocity_penalty
        env.rewards["body_ang_vel"].func = body_angular_velocity_penalty
        env.rewards["body_ang_vel"].params["command_name"] = "twist"
        env.rewards["body_ang_vel"].params["cmd_std"] = 0.1
        env.rewards["body_ang_vel"].params["standing_gain"] = 3.0
        env.rewards["body_ang_vel"].params["walking_gain"] = 1.0


class G1RmaVelEstArmFlashSacStudentOnlyg1c8(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + upright switched to upright_gated (3x flatness at cmd~0). Single lever.

    Motivation (zero-cmd wave 4, 2026-07-21): upright wasn't tested as a single-var lever on
    g1b3. Activate upright_gated to add 3x base-flatness pressure at zero cmd only.

    Orthogonality: g1c7 boosts body_ang_vel (trunk ang velocity); g1c8 boosts upright
    (base tilt). Different mechanisms, both targeted at zero-cmd stability.

    Single lever vs g1b3: ONLY upright -> upright_gated. body_ang_vel (-0.15), foot_clearance
    (-2.0 + target 0.20), all other rewards unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c8.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 upright -> upright_gated (3x base-flatness at cmd~0, 1x at full command)
        from tasks.humanoid_velocity.reward import upright_gated
        env.rewards["upright"].func = upright_gated
        env.rewards["upright"].params["command_name"] = "twist"
        env.rewards["upright"].params["cmd_std"] = 0.1
        env.rewards["upright"].params["standing_gain"] = 3.0
        env.rewards["upright"].params["walking_gain"] = 1.0


class G1RmaVelEstArmFlashSacStudentOnlyg1c9(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + cmd-range cap tightened (±0.8 -> ±0.5 final stage). Single lever.

    Motivation (zero-cmd wave 6, 2026-07-21): prior cmd-distribution levers all REJECTED:
    - rel_standing 0.3 (g1c6) +67% fell
    - rel_standing 0.6 (g1c5) vtf 0.755 KILL

    NEW PREMISE via cmd-range (orthogonal to rel_standing): instead of over-sampling
    zero-cmd across envs at rate `rel_standing_envs` (which breaks tracking), shrink the
    velocity range cap from ±0.8 to ±0.5. Same uniform sampler, but smaller domain means
    cmd density near zero-cmd is HIGHER (1.0 per unit vs 0.625 per unit, +60% zero-cmd
    neighborhood exposure). Curriculum shape preserved: warmup ±0.3 -> mid ±0.5 -> safe
    (was ±0.8) NOW ±0.5. Last 3k iter (12k -> 15k) hold at mid instead of expanding to safe.
    ang_vel_z cap unchanged at ±0.5.

    Orthogonality to prior:
    - rel_standing_envs lever (g1c5/c6): adds zeros across envs at resample time. g1c9
      keeps env-level cmd resampling uniform; changes DOMAIN only.
    - std_min lever (g1c4): changes reward tracking tolerance. g1c9 doesn't touch rewards.
    - All g1c1-c8 reward/curriculum levers: reward structure unchanged in g1c9.

    Single lever vs g1b3: ONLY curriculum.params.velocity_stages[2].lin_vel_x/y. All
    rewards, events, DR, observation terms unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1c9.
    """

    _G1C9_TIGHTENED_LIN_VEL_RANGE: Final[tuple[float, float]] = (-0.5, 0.5)

    def configure(self, env, agent):
        super().configure(env, agent)
        # g1b3 curriculum final stage (step=12000): clamp lin_vel_x/y to mid stage values
        # instead of expanding to ±0.8. Keeps ang_vel_z at the full ±0.5 (unchanged).
        vel_stages = env.curriculum["command_vel"].params["velocity_stages"]
        for stage in vel_stages:
            if stage["step"] == 12000:
                stage["lin_vel_x"] = self._G1C9_TIGHTENED_LIN_VEL_RANGE
                stage["lin_vel_y"] = self._G1C9_TIGHTENED_LIN_VEL_RANGE


class G1RmaVelEstArmFlashSacStudentOnlyg1b3_30k(G1RmaVelEstArmFlashSacStudentOnlyg1b3):
    """g1b3 + max_iterations=30000 (2x longer training). Single lever.

    Motivation (zero-cmd wave 5, 2026-07-21): all 16 single-var reward/data-distribution levers
    REJECTED. The keepers' basin is at structural equilibrium for the current reward structure.
    LAST HYPOTHESIS: maybe the basin needs more training time to develop zero-cmd stability.
    Training to 30k iter (vs default 15k) may develop the deeper zero-cmd basin.

    Orthogonality: same as g1b3 except total iters. If 30k works, the issue was training time.
    If 30k fails, the basin is fundamentally saturated.

    Single lever vs g1b3: ONLY max_iterations=30000 (vs 15000). All rewards, obs, DR unchanged.

    Inheritance: ...->StudentOnly->g1v2->g1b3->g1b3_30k.
    """

    max_iterations: int = 30000
