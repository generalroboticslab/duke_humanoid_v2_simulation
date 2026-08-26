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


class G1RmaVelEstArmFlashSacL2T(BaseExperiment):
    """The Unitree G1 baseline, built to match the humanoid lineage rather than to be tuned.

    Same RMA encoder with velocity estimator, same collision-free arm reference and residual
    action, same teacher-student stack. The final values from the humanoid chain are ported
    across in one step instead of being rediscovered, so the comparison in the paper is between
    two robots, not between a tuned policy and an untuned one.
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
    """The G1 trained without a teacher, directly against a privileged critic.

    The same change that produced the shipped humanoid policies, applied to the baseline so both
    robots are trained the same way.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)       # G1 L2T stack (distilled student, frame-ring, DAgger)
        sac_cfg.use_distilled_student = False      # remove teacher/student distillation
        sac_cfg.actor_obs_group = "student"        # RL actor = deployable proprio group (asymmetric; critic stays privileged)
        sac_cfg.student_start_iters = 0            # no-op w/o distillation; reset for clean config


class G1RmaVelEstArmFlashSacStudentOnlyg1v2(G1RmaVelEstArmFlashSacStudentOnly):
    """Doubles the G1's foot clearance weight, matching the humanoid change that fixed foot
    scuffing.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.rewards["foot_clearance"].weight = -2.0


class G1RmaVelEstArmFlashSacStudentOnlyg1bsk2(G1RmaVelEstArmFlashSacStudentOnlyg1v2):
    """Slows the G1's arm reference while standing, at half the humanoid's dose.

    **The G1 baseline shipped with this release**, and the `g1` column of the benchmark.
    """
    def configure(self, env, agent):
        super().configure(env, agent)
        env.commands["arm_ref"].stand_min_steps = 40
