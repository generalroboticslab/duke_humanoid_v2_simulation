"""Experiment classes for humanoid_legs_only task.

Add new variants by subclassing an existing class.
- Class attributes override structural flags (task, arms, obs space).
- configure() overrides MDP tuning (rewards, terrain, commands, LR etc.).

Example::

    class HumanoidLegsOnlyMyVariant(HumanoidLegsOnly):
        \"\"\"My variant description. Available as: --task HumanoidLegsOnlyMyVariant\"\"\"
        randomize_arms = False           # structural: applied before env build
        def configure(self, env, agent): # MDP tuning: applied after env build
            env.commands["twist"].ranges.lin_vel_x = (0.0, 1.5)
"""
from typing import Final

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from utils.experiments import BaseExperiment

import re

# Curriculum callbacks fire every _CURRICULUM_DECIMATION control steps.
# 1200 steps = 50 PPO iterations — matches CURRICULUM_DECIMATION in env_cfg.
# All curriculum signals change on 100s-to-1000s-of-iter timescales; finer
# granularity just adds recomputation with no training benefit.
_CURRICULUM_DECIMATION: Final[int] = 1200
_LEG_PATTERN: Final[str] = r"(waist_joint|.*_hip_.*|.*_knee_.*|.*_ankle_.*)"
_ARM_PATTERN: Final[str] = f"^(?!{_LEG_PATTERN}).*"
_CAMERA_PATTERN: Final[str] = r"cam_(yaw|pitch)_(left|right)"
_CAMERA_PATTERN_SINGLE: Final[str] = r"cam_(yaw|pitch)"  # single centered module, no left/right suffix
# Excludes both leg joints AND camera joints — required when head_camera="actuated"
# so the arm graph-nav command (which loads arm_collision_graph lacking camera joints)
# does not receive camera actuator names and raise a ValueError.
_ARM_ONLY_PATTERN: Final[str] = rf"^(?!{_LEG_PATTERN})(?!cam_).*"


        # # 4. Remove EE payload / COM DR events and their curriculum entries
        # env.events.pop("ee_payload_mass", None)
        # env.events.pop("ee_com_offset", None)
        # env.curriculum["domain_randomization"].params["events"].pop("ee_payload_mass", None)
        # env.curriculum["domain_randomization"].params["events"].pop("ee_com_offset", None)

# best so far


class HumanoidVelocityRMACNNShortEstimator(BaseExperiment):
    """The base every shipped policy descends from: RMA over a 20-frame history, plus a head that
    estimates base linear velocity.

    The robot cannot measure its own forward velocity, so a small MLP predicts it from the
    encoder latent and the navigation layer reads that at deployment. The estimator's gradient
    is stopped before the encoder. Letting it through costs lateral stability: training commands
    keep the sideways velocity near zero almost always, so that component of the loss is a weak,
    degenerate signal that drags the encoder's representation with it.
    """
    task = "humanoid_velocity"
    num_steps_per_env: int = 24
    curriculum_decimation: int = 1200
    num_envs: int = 4096

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from tasks.humanoid_velocity.humanoid_velocity_env_cfg import humanoid_v21_velocity_env_cfg
        return humanoid_v21_velocity_env_cfg(
            enable_corruption=enable_corruption,
            enable_reward_curriculum=enable_reward_curriculum,
            plane_flag=getattr(self, "plane_flag", True),
            num_steps_per_env=self.num_steps_per_env,
            curriculum_decimation=self.curriculum_decimation,
        )

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
        from mjlab.envs import mdp
        from tasks.humanoid_velocity.humanoid_velocity_env_cfg import _RECOMPUTE_INTERVAL_S
        from asset_zoo.humanoid_v21.humanoid_v21_constants import FEET_GEOMS_PATTERN
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.managers.event_manager import EventTermCfg
        import tasks.humanoid_velocity.event as local_event
        from mjlab.tasks.velocity import mdp as vel_mdp
        from tasks.humanoid_velocity.event import DeferredModelFieldsWrapper
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.curriculum_manager import CurriculumTermCfg
        from tasks.humanoid_velocity.reward import reward_weight_linear
        import math

        # Estimator head.
        env.observations["estimator_target"] = ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(
                    func=mdp.builtin_sensor,
                    params={"sensor_name": "robot/imu_lin_vel"},
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        )
        agent.actor.class_name = "ActorCriticRMAEstimator"
        # Small entropy floor: 0.0003×|H_init=39|≈0.012 < surrogate_init≈0.017
        # (won't dominate at init); 0.0002×|H_collapse=36|≈0.007 ≈ surrogate_collapse
        # (provides pushback at full collapse). Safe window: 0.0002-0.0004.
        agent.algorithm.entropy_coef = 0.0002
    
        # Standing gradient.
        env.commands["twist"].rel_standing_envs = 0.2


        # Shared policy encoder setup.
        env.observations["actor"].history_length = 20
        env.observations["actor"].flatten_history_dim = False
        agent.actor.class_name = "ActorCriticHistory"
        agent.actor.encoder_type = "rma_cnn"

        # Energy penalties.
        env.rewards["joint_torque"].weight = -1e-3
        env.curriculum["joint_torque"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -1e-3},
        ]
        env.rewards["joint_power"].weight = -2e-4
        env.curriculum["joint_power"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -2e-4},
        ]

        # Air-time shaping.
        # command_threshold gates the command magnitude; it decides when this reward is active.
        env.rewards["air_time"].params["command_threshold"] = 0.05
        # threshold_min gates the measured foot air time; it filters out tiny flickers / hops.
        env.rewards["air_time"].params["threshold_min"] = 0.05
        env.rewards["air_time"].weight = 0.2
        env.curriculum["air_time"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": 0.5},
            {"step": 2000 * self.num_steps_per_env, "weight": 0.2},
        ]
        
        # Yaw tracking and slip.
        env.rewards["track_angular_velocity"] = RewardTermCfg(
            func=env.rewards["track_angular_velocity"].func,
            weight=1.5,
            params={**env.rewards["track_angular_velocity"].params, "std": math.sqrt(0.25)},
        )
        env.rewards["foot_slip"].weight = -0.2

        # Foot swing height.
        # foot_swing_height is evaluated once per landing from completed swing peak —
        # least hackable clearance signal (no benefit from micro-lifts or slow
        # horizontal velocity). Previous run stopped at -0.5 at ep 1000; peak_height
        # remained ~0.027m against 0.1m target — penalty too weak to close the gap.
        # Modify weight_stages in-place (stub entry exists in base env cfg). Creating a
        # NEW key here would not be picked up by the curriculum manager if it is built
        # before configure() runs — the in-place mutation IS visible to the manager.
        env.curriculum["foot_swing_height"].params["weight_stages"] = [
            {"step": 0,                              "weight": -0.25},
            {"step": 1000 * self.num_steps_per_env, "weight": -0.50},
            {"step": 5000 * self.num_steps_per_env, "weight": -1.00},
        ]
        env.curriculum["foot_swing_height"].params["decimation"] = self.curriculum_decimation

        # Sim-to-real DR: carpet contact.
        env.events["foot_friction"].func = DeferredModelFieldsWrapper(vel_mdp.dr.geom_friction)
        env.events["foot_friction"].mode = "interval"
        env.events["foot_friction"].is_global_time = True
        env.events["foot_friction"].interval_range_s = (_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S)
        env.events["foot_friction"].params["ranges"] = (0.3, 2.5)

        env.events["foot_solimp"] = EventTermCfg(
            func=DeferredModelFieldsWrapper(local_event.randomize_foot_solimp),
            mode="interval",
            is_global_time=True,
            interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(FEET_GEOMS_PATTERN,)),
                "d0_range": (0.85, 0.99),
                "width_range": (0.002, 0.01),
            },
        )
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["foot_friction"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (0.6, 1.0)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (0.3, 2.5)},
        ]
        dr_params["events"]["foot_solimp"] = [
            {"step": 200 * self.num_steps_per_env, "d0_range": (0.9, 0.95), "width_range": (0.005, 0.005)},
            {"step": 5000 * self.num_steps_per_env, "d0_range": (0.85, 0.95), "width_range": (0.002, 0.012)},
        ]

        env.events["foot_solref"] = EventTermCfg(
            func=DeferredModelFieldsWrapper(local_event.randomize_foot_solref),
            mode="interval",
            is_global_time=True,
            interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(FEET_GEOMS_PATTERN,)),
                "timeconst_range": (0.015, 0.025),   # near-default at start
                "dampratio_range": (0.9, 1.1),
            },
        )
        dr_params["events"]["foot_solref"] = [
            {"step": 200 * self.num_steps_per_env,  "timeconst_range": (0.015, 0.025), "dampratio_range": (0.9, 1.1)},
            {"step": 5000 * self.num_steps_per_env, "timeconst_range": (0.01, 0.04),   "dampratio_range": (0.8, 1.2)},
        ]


class HumanoidRmaVelEstFlashSac(HumanoidVelocityRMACNNShortEstimator):
    """The same environment trained with FlashSAC (distributional soft actor-critic) instead of PPO.

    Off-policy training reuses each transition many times, so the sample cost of a run drops
    sharply. The one thing that does not carry over unchanged is curriculum pacing: reward
    milestones are counted in environment steps, and the two stacks consume steps per iteration
    at different rates, so the milestones are scaled by `num_steps_per_env`. Without that they
    fire roughly six times too early and penalise the policy before it can walk.
    """

    algo = "flash_sac"
    num_steps_per_env: int = 4    
    
    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.curriculum["air_time"].params["weight_stages"] = [
            {"step": 1000 * self.num_steps_per_env, "weight": 0.5},
            {"step": 3000 * self.num_steps_per_env, "weight": 0.2},
        ]
        env.curriculum["joint_power"].params["weight_stages"] = [
            {"step": 1000 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -2e-4},
        ]
        env.curriculum["joint_torque"].params["weight_stages"] = [
            {"step": 1000 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -1e-3},
        ]    
    
    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.num_collect_steps = 4 # self.num_steps_per_env
        sac_cfg.num_updates = 3
        
        sac_cfg.use_sequence_encoder = True
        sac_cfg.use_velocity_estimator = True
        
        sac_cfg.log_std_min = -3.0
        sac_cfg.normalize_reward = True
        sac_cfg.normalized_G_max = 5.0
        sac_cfg.v_min = -5.0
        sac_cfg.v_max = 5.0
        sac_cfg.num_atoms = 101
        sac_cfg.target_sigma = 0.15
        
        sac_cfg.sample_chunk_size = 2
        sac_cfg.num_steps = 3 # n-step returns
        sac_cfg.alpha_init = 0.01
        sac_cfg.alpha_learning_rate = 2e-4
        sac_cfg.tau = 0.01
        sac_cfg.use_zeta_noise = True
        sac_cfg.zeta_mu = 2.0
        sac_cfg.zeta_max_n = 16
        sac_cfg.buffer_size = 256
        sac_cfg.batch_size = 4096
        sac_cfg.use_per_q_target = True


class HumanoidRmaVelEstArmFlashSac(HumanoidRmaVelEstFlashSac):
    """Adds the arms. The policy now tracks a commanded arm pose while it walks.

    Arm targets are perturbed during training so the policy cannot assume they are smooth, and
    the penalty on arm action magnitude ramps in only after the legs are competent. Penalising
    arm motion from step zero produces a policy that keeps its arms rigidly still, which tracks
    a reference badly and looks wrong.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        import re
        from mjlab.managers.curriculum_manager import CurriculumTermCfg
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from asset_zoo.humanoid_v21.humanoid_v21_constants import HUMANOID_V21_ACTION_SCALE
        from tasks.legs_only_task import apply_legs_only_modifiers, RandomizedOffsetJointPositionActionCfg
        from tasks.humanoid_velocity.observation import target_arm_joint_pos
        from tasks.humanoid_velocity.reward import arm_joint_tracking, arm_action_l2, reward_weight_linear

        N = self.num_steps_per_env  # 4

        apply_legs_only_modifiers(
            env,
            _LEG_PATTERN,
            _ARM_PATTERN,
            robot_name="humanoid_v21",
            observe_com=False,
            observe_full_joints=True,
            arm_action_cfg=RandomizedOffsetJointPositionActionCfg(
                entity_name="robot",
                actuator_names=(_ARM_PATTERN,),
                scale={k: v for k, v in HUMANOID_V21_ACTION_SCALE.items() if re.search(_ARM_PATTERN, k)},
                use_default_offset=False,
                robot_name="humanoid_v21",
                min_steps=20,
                max_steps=100,
            ),
        )
        env.actions["joint_pos"].scale = {
            k: v for k, v in HUMANOID_V21_ACTION_SCALE.items() if re.search(_LEG_PATTERN, k)
        }

        env.rewards["arm_joint_tracking"] = RewardTermCfg(
            func=arm_joint_tracking,
            weight=2.0,
            params={"std": 0.3, "arm_action_name": "joint_pos_arms", "asset_cfg": SceneEntityCfg("robot")},
        )
        env.rewards["arm_action_l2"] = RewardTermCfg(
            func=arm_action_l2,
            weight=-0.5,
            params={"arm_action_name": "joint_pos_arms"},
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
            func=target_arm_joint_pos,
            params={"arm_action_name": "joint_pos_arms"},
        )


class HumanoidRmaVelEstArmFlashSacv5(HumanoidRmaVelEstArmFlashSac):
    """Doubles the penalty on arm action magnitude.

    The parent leaves enough residual arm motion that the deployed robot buzzes: small
    high-frequency corrections that are free in simulation and heat real motors.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.curriculum["arm_action_l2"].params["weight_stages"][-1]["weight"] = -1.0


class HumanoidRmaVelEstArmFlashSacv9(HumanoidRmaVelEstArmFlashSacv5):
    """Raises the yaw tracking weight, because turning was the weakest tracking axis.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].weight = 2.5


class HumanoidRmaVelEstArmFlashSacv15(HumanoidRmaVelEstArmFlashSacv9):
    """Five times the penalty on body angular velocity, to damp the trunk wobble that appears once
    yaw tracking is weighted heavily enough to be taken seriously.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["body_ang_vel"].weight = -0.05


class HumanoidRmaVelEstArmFlashSacv18(HumanoidRmaVelEstArmFlashSacv15):
    """Terminates an episode when the policy stops tracking the command at all.

    The threshold is proportional to the command, which is what makes it useful at low speed: a
    policy that simply stands still scores well on a Gaussian tracking reward when the command
    is small, and nothing else in the reward set notices.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from mjlab.managers.termination_manager import TerminationTermCfg
        from tasks.humanoid_velocity import event as task_event
        env.terminations["velocity_tracking_failure"] = TerminationTermCfg(
            func=task_event.VelocityTrackingFailure,
            params={"cmd_xy_threshold": 0.1, "tracking_ratio": 0.3,
                    "consecutive_steps": 70, "command_name": "twist"},
        )


class HumanoidRmaVelEstArmFlashSacv24(HumanoidRmaVelEstArmFlashSacv18):
    """Halves the torque penalty, after the arm model gained a real end effector.

    The gripper took arm mass from approximately zero to 288 g per side. The torque budget
    inherited from the massless model made holding an arm out cost more than it was worth.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.curriculum["joint_torque"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -5e-4},
        ]


class HumanoidRmaVelEstArmFlashSacv29(HumanoidRmaVelEstArmFlashSacv24):
    """Arm targets come from a collision-free reference instead of arbitrary joint angles.

    A pose sampled freely in joint space is frequently one the arm cannot hold without hitting
    the torso, and training against unreachable references teaches the policy to ignore them.
    The reference is a command term that walks a precomputed graph of known-safe poses; the
    action is a residual on top of it.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        import re
        from asset_zoo.humanoid_v21.humanoid_v21_constants import HUMANOID_V21_ACTION_SCALE
        from tasks.joint_ref_command import (
            CommandGatedArmGraphRefCommandTermCfg,
            JointPosRefResidualActionCfg,
        )

        # Reference-as-command migration (plan/smooth-conjuring-rainbow): the arm graph-nav
        # REFERENCE generator moves from the action term to a CommandTerm; the action becomes a thin
        # residual joint_target = joint_ref + scale * action. Same generator values as the old
        # command-gated collision-free arm action, but the reference self-ticks at command-compute
        # (post-physics) -> a 1-step ref latency (= the real onboard pipeline latency). Register
        # arm_ref AFTER twist so the locomotion-gated arm command sees the current twist.
        env.commands["arm_ref"] = CommandGatedArmGraphRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_ARM_PATTERN,),
            robot_name="humanoid_v21",
            steps_per_edge=50,
            locomotion_vel_threshold=0.3,
            loco_min_steps=80,
            command_name="twist",
        )
        env.actions["joint_pos_arms"] = JointPosRefResidualActionCfg(
            entity_name="robot",
            actuator_names=(_ARM_PATTERN,),
            scale={k: v for k, v in HUMANOID_V21_ACTION_SCALE.items() if re.search(_ARM_PATTERN, k)},
            use_default_offset=False,
            command_name="arm_ref",
        )

        # Re-point inherited arm tracking obs/reward from the (now-residual) action to arm_ref.
        arm_track_params = env.rewards["arm_joint_tracking"].params
        arm_track_params.pop("arm_action_name", None)
        arm_track_params["command_name"] = "arm_ref"
        obs_params = env.observations["actor"].terms["target_arm_joint_pos"].params
        obs_params.pop("arm_action_name", None)
        obs_params["command_name"] = "arm_ref"


class HumanoidRmaVelEstArmFlashSacv30(HumanoidRmaVelEstArmFlashSacv29):
    """Makes the linear velocity tracking reward scale its width with the command.

    A fixed-width Gaussian is the dead-zone bug in its purest form. At a command of 0.2 m/s a
    standing robot still collects 85% of the tracking reward, so there is almost no gradient
    toward actually moving. Tying the width to the command keeps slow walking as sharply scored
    as fast walking.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_linear_velocity_relative

        track_lin_vel = env.rewards["track_linear_velocity"]
        track_lin_vel.func = track_linear_velocity_relative
        track_lin_vel.params.pop("std", None)
        track_lin_vel.params["std_rel"] = 0.5
        track_lin_vel.params["std_min"] = 0.3


class HumanoidRmaVelEstArmFlashSacv59L2T(HumanoidRmaVelEstArmFlashSacv30):
    """The teacher-student branch: a privileged teacher trains alongside, and the deployable policy
    learns from it.

    Distillation happens per step with the teacher still learning, rather than freezing a
    finished teacher and cloning it afterwards. Kept in this release because the classes that
    grafted on the camera were built here, and because the comparison against training the
    deployable policy directly is the reason the shipped policies do not use this path.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        # Flattened L2T stack on the v30 base (was v30Teacher→v51Teacher→v51L2T→v54L2T). Ordering
        # mirrors the former MRO exactly: (1) capture the plain-v30 student obs from a throwaway
        # v30 on a FRESH UNMUTATED env (before any teacher swap leaks in), (2) v30 chain, (3)
        # v30Teacher env deltas, (4) v51Teacher integral deltas, (5) graft the student group.
        import copy
        import dataclasses
        from mjlab.envs import mdp
        from mjlab.tasks.velocity import mdp as vel_mdp
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from tasks.humanoid_velocity.humanoid_velocity_env_cfg import (
            humanoid_v21_velocity_env_cfg, foot_site_cfg, foot_height_fast,
        )
        from tasks.humanoid_velocity.observation import joint_actuator_force
        from tasks.humanoid_velocity.command import IntegralErrorVelocityCommandCfg, integral_error_obs
        from tasks.humanoid_velocity.reward import track_integral_error

        # (1) v51L2T: capture plain-v30 actor obs (= deployable student group) on a throwaway env.
        v30_env = humanoid_v21_velocity_env_cfg()
        HumanoidRmaVelEstArmFlashSacv30().configure(v30_env, copy.deepcopy(agent))
        student_group = copy.deepcopy(v30_env.observations["actor"])

        # (2) v30 chain (track_linear_velocity_relative + arm setup + base FlashSAC).
        super().configure(env, agent)

        # (3) v30Teacher: 2× energy penalties + privileged GT actor obs (clean, no corruption).
        env.rewards["joint_torque"].weight = -2e-3
        env.curriculum["joint_torque"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -2e-3},
        ]
        env.rewards["joint_power"].weight = -4e-4
        env.curriculum["joint_power"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -4e-4},
        ]
        env.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"})
        env.observations["actor"].terms["projected_gravity"] = ObservationTermCfg(
            func=vel_mdp.projected_gravity)
        env.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
            func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})
        env.observations["actor"].terms["foot_height"] = ObservationTermCfg(
            func=foot_height_fast, params={"asset_cfg": foot_site_cfg()})
        env.observations["actor"].terms["foot_air_time"] = ObservationTermCfg(
            func=vel_mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"})
        env.observations["actor"].terms["foot_contact"] = ObservationTermCfg(
            func=vel_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"})
        env.observations["actor"].terms["foot_contact_forces"] = ObservationTermCfg(
            func=vel_mdp.foot_contact_forces, params={"sensor_name": "feet_ground_contact"})
        env.observations["actor"].terms["joint_torque"] = ObservationTermCfg(
            func=joint_actuator_force)
        env.observations["actor"].enable_corruption = False

        # (4) v51Teacher: clean integral-error command (dr_bias=dr_noise=0) + integral obs (critic+
        # actor) + track_integral_error reward.
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
        # Flattened: v30 base + v30Teacher (no estimator) + v51L2T (distilled student) + v54L2T
        # (α_mix=0.5, frame-ring) + v59L2T (per-episode mixing).
        super().flash_sac_configure(sac_cfg)
        sac_cfg.use_velocity_estimator = False     # v30Teacher: actor sees GT lin_vel, no head
        sac_cfg.use_distilled_student = True        # v51L2T
        sac_cfg.student_obs_group = "student"        # v51L2T
        sac_cfg.student_action_prob_max = 0.5        # v54L2T: DAgger mixing ceiling
        sac_cfg.frame_ring_history = True            # v54L2T: storage-only (now also frame-rings student)
        sac_cfg.episode_level_mixing = True          # v59L2T: per-episode driver draw


class HumanoidRmaVelEstArmFlashSacv59L2TActuatedCam(HumanoidRmaVelEstArmFlashSacv59L2T):
    """Grafts the two actuated head-camera gimbals on, and puts gaze in the action vector.

    Four extra actuated joints, four extra observed gaze targets. The policy is trained against
    randomly stepping gaze commands, so the real robot drives its own gimbals rather than
    handing them to a separate look-at controller that would fight the walking policy.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        import re
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from asset_zoo.humanoid_v21 import get_humanoid_v21_robot_cfg
        from asset_zoo.humanoid_v21.humanoid_v21_constants import HUMANOID_V21_ACTION_SCALE
        from tasks.joint_ref_command import (
            CommandGatedArmGraphRefCommandTermCfg,
            GazeRefCommandTermCfg,
            JointPosRefResidualActionCfg,
        )
        from tasks.humanoid_velocity.reward import arm_joint_tracking
        from tasks.humanoid_velocity.observation import target_arm_joint_pos

        # Reference-as-command migration (plan/smooth-conjuring-rainbow): the arm graph-nav and
        # camera gaze REFERENCE generators move from the action term to CommandTerms; the action
        # becomes a thin residual: joint_target = joint_ref + scale * action. Same generator
        # values as v59's old collision-free arm action / gaze action, but the
        # reference self-ticks at command-compute (post-physics) -> a 1-step ref latency vs the old
        # in-action tick (= the real onboard pipeline latency). Register arm_ref/camera_ref AFTER
        # twist so the locomotion-gated arm command sees the current twist.

        # 1. Switch to actuated camera robot (+4 cam DOFs + 4 cam actuators).
        env.scene.entities["robot"] = get_humanoid_v21_robot_cfg(head_camera="actuated")

        arm_scale = {k: v for k, v in HUMANOID_V21_ACTION_SCALE.items() if re.search(_ARM_ONLY_PATTERN, k)}

        # 2. Arm joint reference = collision-free graph nav (camera joints excluded: the graph has
        #    no cam joints). Same params v59 used on the old action.
        env.commands["arm_ref"] = CommandGatedArmGraphRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_ARM_ONLY_PATTERN,),
            robot_name="humanoid_v21",
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

        # 3. Camera joint reference = random-WALK gaze (velocity bounded by RS05 kd=1). Seeded
        #    absolute-uniform per episode, then bounded random-walk steps so the full ±2π yaw ROM
        #    is explored while each step stays reachable in its hold window (1.0 rad in 0.23 s «
        #    0.6 s min hold under 4.4 rad/s).
        env.commands["camera_ref"] = GazeRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            min_steps=30,
            max_steps=120,
            yaw_range=4.7124,    # ROM clamp ±3π/2 (full joint ROM ±270°)
            pitch_range=1.5708,  # ROM clamp ±π/2 (full pitch ROM)
            yaw_walk=1.0,        # per-step walk delta yaw (reachable in 0.23 s)
            pitch_walk=0.4,      # per-step walk delta pitch
        )
        env.actions["joint_pos_camera"] = JointPosRefResidualActionCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            scale={"cam_yaw.*": 0.3, "cam_pitch.*": 0.2},
            use_default_offset=False,
            command_name="camera_ref",
        )

        # 4. Camera tracking reward (reads the camera_ref command).
        env.rewards["camera_joint_tracking"] = RewardTermCfg(
            func=arm_joint_tracking,
            weight=1.0,
            params={"std": 0.3, "command_name": "camera_ref", "asset_cfg": SceneEntityCfg("robot")},
        )

        # 5. Gaze command obs for BOTH teacher (actor) and student (deploy policy).
        env.observations["actor"].terms["target_camera_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "camera_ref"})
        env.observations["student"].terms["target_camera_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "camera_ref"})

        # Arm tracking obs/reward already point at the "arm_ref" command (set in v29); the actor
        # term is inherited and the student term carries it via v59L2T's deepcopy of the v30 actor.

    def flash_sac_configure(self, sac_cfg) -> None:
        # Inlined teacher-warmup gate (= v61L2T's Exp-0 lever): skip ALL student-relevant work
        # (imitation update + collect-time student fwd/mix; α-mix ramp offset by this) until the
        # teacher is task-competent. 3000 = v59 teacher track_linear_velocity plateau (measured;
        # see HumanoidRmaVelEstArmFlashSacv61L2T docstring +
        # Exp 0). Distilling the camera+locomotion student before the teacher's action label
        # stabilizes chases a moving target + pollutes the student obs-normalizer.
        super().flash_sac_configure(sac_cfg)
        sac_cfg.student_start_iters = 3000


class HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam(HumanoidRmaVelEstArmFlashSacv59L2TActuatedCam):
    """Freezes the teacher partway through training and lets the student finish on its own.

    Used as the default task by the frozen-policy helpers and the camera field-of-view
    visualiser. The shipped policies take the no-teacher path instead; see the FlashSAC notes in
    `tasks/humanoid_velocity/README.md` for why.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.freeze_teacher_after_iters = 11000     # v83: unified-base optimum (beats 12k both axes, 2-seed)


class HumanoidRmaVelEstArmFlashSacv123(HumanoidRmaVelEstArmFlashSacv30):
    """Drops the teacher entirely: the deployable policy is trained directly by SAC against a
    privileged critic.

    This is the branch the shipped policies come from. Behaviour cloning caps a student at the
    conditional mean of its teacher given what the student can see, and that cap bites hardest
    at near-zero commands and under heavy payload, which is exactly where a humanoid falls over.
    Training the deployable actor directly has no such ceiling. The critic stays privileged, so
    what changes is the actor's supervision, not the value function's information.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)  # v30 base: deploy actor = proprio, privileged critic, arm setup

        import dataclasses
        import re
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from asset_zoo.humanoid_v21 import get_humanoid_v21_robot_cfg
        from asset_zoo.humanoid_v21.humanoid_v21_constants import HUMANOID_V21_ACTION_SCALE
        from tasks.joint_ref_command import (
            CommandGatedArmGraphRefCommandTermCfg,
            GazeRefCommandTermCfg,
            JointPosRefResidualActionCfg,
        )
        from tasks.humanoid_velocity.reward import arm_joint_tracking, track_integral_error
        from tasks.humanoid_velocity.observation import target_arm_joint_pos
        from tasks.humanoid_velocity.command import IntegralErrorVelocityCommandCfg, integral_error_obs

        # (A) Clean integral-error twist command + command_integral obs on CRITIC only + reward.
        #     Deploy actor never sees the integral (matches StudentOnly's student group). Register the
        #     integral wrap before the cam commands so arm_ref sees the current twist (v59L2TActuatedCam
        #     ordering).
        old = env.commands["twist"]
        env.commands["twist"] = IntegralErrorVelocityCommandCfg(
            **{f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init},
            dr_bias=(0.0, 0.0, 0.0),
            dr_noise=(0.0, 0.0, 0.0),
        )
        env.observations["critic"].terms["command_integral"] = ObservationTermCfg(
            func=integral_error_obs, params={"command_name": "twist"},
        )
        env.rewards["track_integral_error"] = RewardTermCfg(
            func=track_integral_error,
            weight=1.0,
            params={"command_name": "twist", "std": 0.3},
        )

        # (B) Actuated twin-D435 head camera + random-WALK gaze (velocity bounded by RS05 kd=1).
        #     Ported verbatim from v59L2TActuatedCam.configure items 1-4; item-5 gaze obs to the deploy
        #     actor group only.
        env.scene.entities["robot"] = get_humanoid_v21_robot_cfg(head_camera="actuated")
        arm_scale = {k: v for k, v in HUMANOID_V21_ACTION_SCALE.items() if re.search(_ARM_ONLY_PATTERN, k)}
        env.commands["arm_ref"] = CommandGatedArmGraphRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_ARM_ONLY_PATTERN,),
            robot_name="humanoid_v21",
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
        env.commands["camera_ref"] = GazeRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            min_steps=30,
            max_steps=120,
            yaw_range=4.7124,    # ROM clamp +-3pi/2 (full joint ROM +-270deg)
            pitch_range=1.5708,  # ROM clamp +-pi/2 (full pitch ROM)
            yaw_walk=1.0,        # per-step walk delta yaw
            pitch_walk=0.4,      # per-step walk delta pitch
        )
        env.actions["joint_pos_camera"] = JointPosRefResidualActionCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            scale={"cam_yaw.*": 0.3, "cam_pitch.*": 0.2},
            use_default_offset=False,
            command_name="camera_ref",
        )
        env.rewards["camera_joint_tracking"] = RewardTermCfg(
            func=arm_joint_tracking,
            weight=1.0,
            params={"std": 0.3, "command_name": "camera_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        env.observations["actor"].terms["target_camera_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "camera_ref"})

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)   # HumanoidRmaVelEstFlashSac base: full optimization knob set
        sac_cfg.use_velocity_estimator = False  # deploy actor reads noisy proprio directly (no GT-lin_vel head), matches StudentOnly


class HumanoidRmaVelEstArmFlashSacv145MixedArmsCam(HumanoidRmaVelEstArmFlashSacv123):
    """Mixes two arm behaviours across episodes: arms held at home, and arms swinging to a
    reference.

    Training on swinging arms alone produces a policy that cannot walk backward when the arms
    happen to be still, and vice versa. Sampling the mode per episode covers both.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)  # v123: camera + gaze + integral twist + arm graph-nav
        arm_ref = env.commands["arm_ref"]
        arm_ref.home_ratio_start = 1.0
        arm_ref.home_ratio_end = 0.5
        # 18000 iters x num_steps_per_env(=4 under flash_sac) = 72000 common_step_counter env-steps.
        # NOTE: completion iter 18000 EXCEEDS the 15000 train budget, so home_ratio never reaches the
        # 0.5 floor — the run ends at frac=15000/18000=0.833 -> home_ratio~0.583 (still annealing). Kept
        # as-trained; a corrected iter-3000 completion would be 3000 * self.num_steps_per_env = 12000.
        arm_ref.home_ratio_curriculum_steps = 18000 * self.num_steps_per_env
class HumanoidRmaVelEstArmFlashSacv153MixedArmsCam(HumanoidRmaVelEstArmFlashSacv145MixedArmsCam):
    """Quarters the penalty on how fast leg actions may change.

    The inherited value buys smoothness at the cost of stepping freedom, and the policy pays for
    it with a shuffling gait. Relaxing it lets the legs commit to a step. The hardware limit on
    how far this can go is the raw action rate, which has to stay under about 1.5.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        # v145 action_rate_l2 quartered: reward weight -0.5 -> -0.125, curriculum stages -0.05/-0.5 -> -0.0125/-0.125
        env.rewards["action_rate_l2"].weight = -0.125
        env.curriculum["action_rate_l2"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -0.0125},
            {"step": 3000 * self.num_steps_per_env, "weight": -0.125},
        ]
class HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam(HumanoidRmaVelEstArmFlashSacv153MixedArmsCam):
    """Doubles the foot clearance weight, so the policy commits to lifting its feet.

    At the inherited weight the clearance term is well under 1% of total reward and the policy
    correctly ignores it, scuffing the ground on every swing. This is the point where the gait
    starts to look like walking.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        # v153 foot_clearance weight: -1.0 -> -2.0 (2x deviation pressure at target 0.10m)
        env.rewards["foot_clearance"].weight = -2.0
class HumanoidRmaVelEstArmFlashSacv2ybMixedArmsCam(HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam):
    """Applies the command-scaled reward width to yaw as well.

    Linear velocity got this treatment much earlier; angular velocity kept a fixed width and
    kept the matching dead zone, saturating near a reward of 1.0 for any small turn command
    regardless of whether the robot actually turned.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_angular_velocity_relative
        env.rewards["track_angular_velocity"].func = track_angular_velocity_relative
        env.rewards["track_angular_velocity"].params.pop("std", None)
        env.rewards["track_angular_velocity"].params["std_rel"] = 0.5
        env.rewards["track_angular_velocity"].params["std_min"] = 0.1


class HumanoidRmaVelEstArmFlashSacv2ybiMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybMixedArmsCam):
    """Widens the yaw reward basin, which had been made too narrow at high commands.

    Scaling the width by the command fixes the low end and breaks the high end: at a 0.5 rad/s
    command the effective width ends up four times narrower than the fixed value it replaced,
    and the policy hesitates rather than accept a large penalty for a moderate error.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_angular_velocity_relative
        env.rewards["track_angular_velocity"].func = track_angular_velocity_relative
        env.rewards["track_angular_velocity"].params.pop("std", None)
        env.rewards["track_angular_velocity"].params["std_rel"] = 1.0
        env.rewards["track_angular_velocity"].params["std_min"] = 0.1

class HumanoidRmaVelEstArmFlashSacv2ybhciMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybiMixedArmsCam):
    """Doubles the upright weight to stop the trunk tilting during turns.

    A yaw reward that scales with the command rewards free rotation, and the cheapest way to
    rotate is to let the trunk lean into it, which destabilises the gait. Penalising tilt
    directly is more targeted than pulling the yaw weight back down.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        # v2ybi upright weight: 1.0 -> 2.0 (anti-tilt directly on destabilizer)
        env.rewards["upright"].weight = 2.0
class HumanoidRmaVelEstArmFlashSacv2ybdkcMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybhciMixedArmsCam):
    """Splits the difference on yaw reward width, at 0.7.

    The two ends of this axis were both worse: too narrow and the policy hesitates at speed,
    too wide and low-speed turning goes slack again.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_angular_velocity_relative
        env.rewards["track_angular_velocity"].func = track_angular_velocity_relative
        env.rewards["track_angular_velocity"].params.pop("std", None)
        env.rewards["track_angular_velocity"].params["std_rel"] = 0.7
        env.rewards["track_angular_velocity"].params["std_min"] = 0.1
class HumanoidRmaVelEstArmFlashSacv2ybskMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybdkcMixedArmsCam):
    """Slows the arm reference while the robot is standing.

    The arm command term already moved slowly during locomotion. Extending that to standing
    gives the policy time to prepare for an arm swing instead of being yanked, and it costs
    nothing at deployment because it changes the training reference rather than the policy.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["arm_ref"].stand_min_steps = 80
class HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybskMixedArmsCam):
    """Tightens the floor on yaw reward width, sharpening the gradient at small turn commands.

    **The adopted configuration**, and the two-camera policy shipped with this release. Named
    `v2` in the benchmark, and `v2_fixed` when the same weights run with the camera joints
    welded.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].params["std_min"] = 0.05


def _apply_single_camera(env: ManagerBasedRlEnvCfg) -> None:
    """Swap the dual head-camera module (4 gaze DOFs, 2 gimbals) for the single centered one
    (cam_yaw/cam_pitch, 2 gaze DOFs, 1 gimbal). Call AFTER ``super().configure()``.

    Overwrites exactly three cfg entries; every downstream consumer resolves its dims generically
    off whatever is bound to these keys at env BUILD time, so nothing else needs re-registration
    (see ``HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam`` docstring for the full argument).
    Shared by the v2_best_single lineage and the grid/gait-init single-cam sibling — the two must
    stay byte-identical for the camera-count ablation to isolate camera count, hence one helper.
    """
    from asset_zoo.humanoid_v21 import get_humanoid_v21_robot_cfg
    from tasks.joint_ref_command import GazeRefCommandTermCfg, JointPosRefResidualActionCfg

    # 1. Single-camera robot entity (2 gaze DOFs instead of 4). Arm/leg/gripper kinematics unchanged
    #    (end_effector/hand keep their get_humanoid_v21_robot_cfg defaults, matching every ancestor's
    #    own head_camera="actuated" call).
    env.scene.entities["robot"] = get_humanoid_v21_robot_cfg(head_camera="actuated_single")

    # 2. Camera reference: same random-walk gaze generator, single-camera actuator pattern.
    env.commands["camera_ref"] = GazeRefCommandTermCfg(
        entity_name="robot",
        actuator_names=(_CAMERA_PATTERN_SINGLE,),
        min_steps=30,
        max_steps=120,
        yaw_range=4.7124,    # ROM clamp +-3pi/2 (full joint ROM +-270deg)
        pitch_range=1.5708,  # ROM clamp +-pi/2 (full pitch ROM)
        yaw_walk=1.0,        # per-step walk delta yaw
        pitch_walk=0.4,      # per-step walk delta pitch
    )
    env.actions["joint_pos_camera"] = JointPosRefResidualActionCfg(
        entity_name="robot",
        actuator_names=(_CAMERA_PATTERN_SINGLE,),
        scale={"cam_yaw.*": 0.3, "cam_pitch.*": 0.2},
        use_default_offset=False,
        command_name="camera_ref",
    )


class HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam(HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam):
    """The adopted policy with one centred camera module instead of two.

    One gimbal, two gaze degrees of freedom rather than four. Everything else -- rewards, arm
    schedule, domain randomization -- is inherited unchanged, which is what makes this a clean
    measurement of what the second camera buys. Named `v2_single` in the benchmark, and
    `v2_single_fixed` welded.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        _apply_single_camera(env)
