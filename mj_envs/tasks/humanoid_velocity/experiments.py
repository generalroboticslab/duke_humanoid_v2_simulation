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
    """RMA-CNN (H=20) with concurrent linear velocity estimator head.
    most stable when deployed in the real, use as !!baseline!!

    Extends HumanoidVelocityRMACNNShort with:
    - vel_head MLP (latent→64→3) trained via MSE against ground-truth imu_lin_vel.
      Probe on encoder latent; used by HL navigation policy at deployment.
    - V2 foot friction + solimp DR to reduce sim-to-real gap on carpet.
    - foot_swing_height curriculum ramped to -1.0 to penalise foot dragging on carpet.

    Stop-gradient on vel_head (lateral stability):
        The MSE is computed on a detached latent (ActorCriticRMAEstimator._last_latent_sg),
        so the encoder receives gradients from PPO only. Without this, MSE backpropagates
        into the encoder and degrades its lateral representation: heading_command=True means
        lin_vel_y ≈ 0 throughout training (robot never strafes), so the MSE gradient for y
        is weak and degenerate, producing real-world y-direction instability. The vel_head
        still predicts velocity accurately enough for the HL navigation policy because the
        encoder already implicitly encodes velocity-correlated proprioceptive features.

    Deployment: ll_policy.get_estimated_velocity(obs) → (B, 3) base_lin_vel.
    Usage: python run.py train --task HumanoidVelocityRMACNNShortEstimator
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


class HumanoidLocoManipV3(HumanoidVelocityRMACNNShortEstimator):
    """Full 5-phase EE curriculum with weight ramps and growing target sphere (Milestone 5).

    EE observations (target, current, error, gate) are appended directly to the actor
    and critic obs groups. The actor's history encoder (RMA-CNN) sees them in every frame,
    providing temporal context; the current frame is concatenated to the MLP input at zero
    latency (architecture invariant of ActorCriticHistory). Per-dimension normalization
    means EE scales do not distort proprioceptive statistics.

    What changes from HumanoidVelocityRMACNNShortEstimator:
    - Payload/COM DR events popped (blueprint defers to Phase 4b)
    - rel_standing_envs = 0.3 (70% walking, 30% standing)
    - actor/critic obs: +19D — ee_targets_h(K*3) + ee_currents_h(K*3) + ee_errors_h(K*3)
                        + ee_gate(1), K=2 (R+L). All batched: one obs term per quantity.
    - resample_ee_targets event (interval 2-4s): resamples all K EEs at once
    - ee_tracking_coarse: mean over K, weight=0.0 start, velocity-gated (σ²=0.16), ramps 0→0.5→2.0
    - arm_action_rate_l2: starts weight=-2.0, ramps -2.0→-0.5→-0.1 (covers both arms)

    Phase schedule (×24 steps/env):
    Warmup  (0-200×24):      w_ee  0.0,   arm_rate 0.0→-2.0 ramp  (loco stabilization)
    Phase 1 (200-700×24):    w_ee  0→0.5, arm_rate -2.0 fixed,    sphere ±5cm
    Phase 2 (700-2200×24):   w_ee  0.5→2.0, arm_rate -2.0→-0.5,  sphere 5→15cm
    Phase 3 (2200-3700×24):  w_ee  2.0 held, arm_rate -0.5 held,  sphere ±15cm
    Phase 4a (3700-4700×24): w_ee  2.0,      arm_rate -0.5→-0.1,  sphere ±15cm

    Usage: python run.py train --task HumanoidLocoManipV3
    """

    _EE_BODY_R: str = "wrist_3_R"
    _EE_BODY_L: str = "wrist_3_L"

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.curriculum_manager import CurriculumTermCfg
        from tasks.humanoid_velocity.observation import (
            ee_targets_h, ee_currents_h, ee_errors_h, ee_gate,
        )
        from tasks.humanoid_velocity.reward import ee_tracking_coarse_batched, arm_action_rate_l2, reward_weight_linear
        from tasks.humanoid_velocity.event import EETargetsResample, ee_target_radius_linear

        N = self.num_steps_per_env  # 24
        EE_BODIES = [self._EE_BODY_R, self._EE_BODY_L]

        # 0. Defer EE payload/COM DR to Phase 4b (blueprint §5.1).
        env.events.pop("ee_payload", None)
        env.curriculum["domain_randomization"].params["events"].pop("ee_payload", None)

        # 1. Locomotion mix: 70% walking, 30% standing.
        env.commands["twist"].rel_standing_envs = 0.3

        # 2. Batched EE obs: 3 terms cover all K EEs in one call each (K*3 dims per term).
        ee_obs = {
            "ee_targets_h":  ObservationTermCfg(func=ee_targets_h,  params={"ee_body_names": EE_BODIES}),
            "ee_currents_h": ObservationTermCfg(func=ee_currents_h, params={"ee_body_names": EE_BODIES}),
            "ee_errors_h":   ObservationTermCfg(func=ee_errors_h,   params={"ee_body_names": EE_BODIES}),
            "ee_gate":       ObservationTermCfg(func=ee_gate,        params={"command_name": "twist"}),
        }
        env.observations["actor"].terms.update(ee_obs)
        env.observations["critic"].terms.update(ee_obs)

        # 3. Single batched resample event for all EEs + radius curriculum.
        env.events["resample_ee_targets"] = EventTermCfg(
            func=EETargetsResample,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"ee_body_names": EE_BODIES, "radius": 0.05},
        )
        env.curriculum["ee_target_radius"] = CurriculumTermCfg(
            func=ee_target_radius_linear,
            params={
                "event_name": "resample_ee_targets",
                "decimation": _CURRICULUM_DECIMATION,
                "radius_stages": [
                    {"step":       0 * N, "radius": 0.05},
                    {"step":     500 * N, "radius": 0.05},
                    {"step":    2000 * N, "radius": 0.15},
                ],
            },
        )

        # 4. Single batched EE reward (mean over K) + weight curriculum.
        # Gate σ²=0.16 keeps 37% signal at 0.4 m/s; weight=0.0 start lets locomotion
        # dominate Phase 1 before arm tracking is introduced.
        env.rewards["ee_tracking_coarse"] = RewardTermCfg(
            func=ee_tracking_coarse_batched,
            weight=0.0,
            params={
                "ee_body_names": EE_BODIES,
                "std": 0.05,
                "gate_command_name": "twist",
                "gate_sigma_sq": 0.16,
            },
        )
        env.curriculum["ee_tracking_coarse_weight"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "ee_tracking_coarse",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step":       0 * N, "weight": 0.0},
                    {"step":     500 * N, "weight": 0.5},
                    {"step":    2000 * N, "weight": 2.0},
                ],
            },
        )

        # 5. Arm smoothness: starts at 0.0 so random init policy doesn't destabilize locomotion.
        # Ramps to -2.0 once the locomotion policy has stabilized (~200 iters), then relaxes.
        env.rewards["arm_action_rate_l2"] = RewardTermCfg(
            func=arm_action_rate_l2,
            weight=0.0,
            params={"arm_action_name": "joint_pos", "arm_joint_pattern": _ARM_PATTERN},
        )
        env.curriculum["arm_action_rate_l2_weight"] = CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "arm_action_rate_l2",
                "decimation": _CURRICULUM_DECIMATION,
                "weight_stages": [
                    {"step":       0 * N, "weight":  0.0},
                    {"step":     200 * N, "weight": -2.0},
                    {"step":     700 * N, "weight": -2.0},
                    {"step":    2200 * N, "weight": -0.5},
                    {"step":    3700 * N, "weight": -0.5},
                    {"step":    4700 * N, "weight": -0.1},
                ],
            },
        )


class HumanoidLocoArmFollow(HumanoidVelocityRMACNNShortEstimator):
    """Joint-space arm following with collision-free pre-sampled pose curriculum.

    Policy controls all 27 joints via the unified joint_pos action. The 14 arm joints
    (waist excluded) track a smoothly interpolated sequence of pre-sampled collision-free
    poses drawn from safe_arm_poses_humanoid_v21.pt sorted by L2 from q_default.

    What changes from HumanoidVelocityRMACNNShortEstimator:
    - pose reward restricted to 13 leg joints + waist (_LEG_PATTERN); no longer conflicts
      with the moving arm target.
    - arm_pose reward (weight=1.0, fixed) covers 14 arm joints, velocity-gated:
        r = exp(-mean((q_arm - q_target)² / 0.01)) × exp(-‖v_cmd_xy‖² / 0.16)
      Replaces the arm component of baseline pose exactly — total reward budget unchanged.
    - arm_joint_target obs (+14D) added to actor and critic. Breaking change: RMA-CNN
      first conv layer incompatible with existing checkpoints. Train from scratch.
    - rel_standing_envs = 0.3 so 30% of envs provide unattenuated arm tracking signal.

    Curriculum (baked into arm_joint_target, driven by env.common_step_counter):
      Phase 1 (0-500N):    pool=5%,       hold=200-500 steps  (near-default, slow moves)
      Phase 2 (500-2000N): pool=5%→100%,  hold=200-50 steps   (expanding diversity)
      Phase 3 (2000N+):    pool=100%,     hold=50-200 steps   (full library, fast moves)
    Reset pool fixed at 5% throughout (keeps episode-start error ≈0 at all phases).

    Usage: python run.py train --task HumanoidLocoArmFollow
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from tasks.humanoid_velocity.observation import arm_joint_target
        from tasks.humanoid_velocity.reward import arm_pose

        N = self.num_steps_per_env  # 24

        # 1. Restrict baseline pose reward to leg joints + waist.
        #    _LEG_PATTERN already includes waist_joint, so no extra handling needed.
        #    arm joints (K=14) get a separate arm_pose reward with moving target,
        #    eliminating the conflict between "return to default" and "track target".
        #    Also update std_walking/std_running to only contain leg-joint patterns;
        #    variable_posture raises if any pattern in these dicts matches no joints.
        _LEG_STD_WALKING = {
            r".*hip_1.*": 0.3, r".*hip_2.*": 0.15, r".*hip_3.*": 0.15,
            r".*knee.*": 0.6, r".*ankle_1.*": 0.3, r".*ankle_2.*": 0.1,
            r"waist.*": 0.2,
        }
        _LEG_STD_RUNNING = {
            r".*hip_1.*": 0.4, r".*hip_2.*": 0.25, r".*hip_3.*": 0.25,
            r".*knee.*": 0.8, r".*ankle_1.*": 0.4, r".*ankle_2.*": 0.2,
            r"waist.*": 0.3,
        }
        env.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(_LEG_PATTERN,)
        )
        env.rewards["pose"].params["std_walking"] = _LEG_STD_WALKING
        env.rewards["pose"].params["std_running"] = _LEG_STD_RUNNING

        # 2. arm_pose: variable_posture formula over 14 arm joints, velocity-gated.
        #    weight=1.0 matches the original arm contribution of the baseline pose term
        #    (which covered all 27 joints at weight=1.0); total reward budget unchanged.
        env.rewards["arm_pose"] = RewardTermCfg(
            func=arm_pose,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(_ARM_PATTERN,)),
                "command_name": "twist",
                "std": 0.1,
                "gate_sigma_sq": 0.16,
            },
        )

        # 3. arm_joint_target obs: K=14 absolute arm targets, added to actor + critic.
        #    The same ObservationTermCfg instance is registered in both groups; the
        #    step-guard in arm_joint_target.__call__ ensures the Warp kernel fires once.
        arm_obs_cfg = ObservationTermCfg(
            func=arm_joint_target,
            params={
                "arm_joint_pattern": _ARM_PATTERN,
                "robot_name": "humanoid_v21",
                "reset_pool_fraction": 0.05,
                "pool_stages": [
                    {"step": 0,          "value": 0.05},
                    {"step": 500  * N,   "value": 0.05},
                    {"step": 2000 * N,   "value": 1.0},
                ],
                "min_steps_stages": [
                    {"step": 0,          "value": 200},
                    {"step": 2000 * N,   "value": 50},
                ],
                "max_steps_stages": [
                    {"step": 0,          "value": 500},
                    {"step": 2000 * N,   "value": 200},
                ],
            },
        )
        env.observations["actor"].terms["arm_joint_target"] = arm_obs_cfg
        env.observations["critic"].terms["arm_joint_target"] = arm_obs_cfg

        # 4. More standing envs: gate≈1 only at v_cmd≈0, so arm tracking signal comes
        #    primarily from standing envs. 30% ensures sufficient gradient flow.
        env.commands["twist"].rel_standing_envs = 0.3


class HumanoidRmaVelEstFlashSac(HumanoidVelocityRMACNNShortEstimator):
    """**CANONICAL FlashSAC baseline** (supersedes HumanoidVelocityEstimatorFlashSACRMACNNv4a, 2026-05-01).

    Switches training stack to FlashSAC and applies a curriculum pacing fix: air_time,
    joint_power, and joint_torque milestone steps are scaled by ``num_steps_per_env``
    so penalties ramp at PPO-equivalent wall-step counts. v4a lacked this fix —
    milestones triggered 6× too early in iteration count, over-penalizing the early
    policy before gait was established.

    Promotion criteria met vs v4a (run 2026-05-01, 15k iters):
      - term_fell_over: 0.010 vs v4a 0.068 (6.75× better stability)
      - raw_joint_torque: 5.30 vs v4a 5.75 (−7.8%)
      - raw_action_rate_l2: 1.055 vs v4a 1.071 (hardware-safe; below 1.5 threshold)
      - metric_air_time_mean: 0.1098 vs v4a 0.1046 (+4.9%)
      - metric_slip_velocity_mean: 0.1241 vs v4a 0.1320 (−6.0%; less slip)
      - tracking/entropy: flat (no regression)
    Canonical run: ``runs/HumanoidRmaVelEstFlashSac/2026-05-01_14-21-58_flash_sac/``

    Baseline update 2026-05-01: Tune04 was folded into this class.
    Effective SAC settings now use num_updates=3 and alpha_learning_rate=2e-4.

    Tune variants (Tune01, Tune02) tested 2026-05-01 with num_envs=8192 — both
    rejected on wall-time efficiency grounds. Base completes 15k in 45 min (5.48 it/s);
    Tune01 needs 66 min (3.79 it/s, +47% wall time). At equal wall time (Tune01@10k),
    stability is worse (fell_over 0.015 vs 0.010) — any tracking gains do not justify
    the cost. Tune02 flat/regressed at all normalizations.
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
        # sac_cfg.use_tanh = True # false -> unstable.


class HumanoidRmaVelEstFlashSacHeight(HumanoidRmaVelEstFlashSac):
    """FlashSAC baseline with explicit relative base-height command control.

    Reuses the existing height command/reward wiring so height behaves like cmd vel:
    the command is observed by the policy, and reward tracks the commanded base z.
    Adds a lower-body joint target prior so the policy sees the commanded leg pose
    that corresponds to the requested height.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.tasks.velocity import mdp as vel_mdp
        from tasks.humanoid_velocity.command import RelativeHeightCommandCfg
        from tasks.humanoid_velocity.observation import joint_target
        from tasks.humanoid_velocity.reward import rel_height_control, joint_target_tracking

        env.commands["rel_height"] = RelativeHeightCommandCfg(
            entity_name="robot",
            resampling_time_range=(2.0, 8.0),
            nominal_height=0.59,
            offset_range=(-0.15, 0),
        )

        env.observations["actor"].terms["rel_height_command"] = ObservationTermCfg(
            func=vel_mdp.generated_commands,
            params={"command_name": "rel_height"},
            delay_min_lag=0,
            delay_max_lag=0,
        )
        if "critic" in env.observations:
            env.observations["critic"].terms["rel_height_command"] = ObservationTermCfg(
                func=vel_mdp.generated_commands,
                params={"command_name": "rel_height"},
                delay_min_lag=0,
                delay_max_lag=0,
            )

        env.observations["actor"].terms["joint_target"] = ObservationTermCfg(
            func=joint_target,
            params={
                "command_name": "rel_height",
                "nominal_z_travel_mm": -40.0,
            },
        )
        if "critic" in env.observations:
            env.observations["critic"].terms["joint_target"] = ObservationTermCfg(
                func=joint_target,
                params={
                    "command_name": "rel_height",
                    "nominal_z_travel_mm": -40.0,
                },
            )

        env.rewards["rel_height"] = RewardTermCfg(
            func=rel_height_control,
            weight=1.0,
            params={
                "command_name": "rel_height",
                "nominal_height": 0.59,
                "std": 0.05,
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            },
        )
        env.rewards["pose"] = RewardTermCfg(
            func=joint_target_tracking,
            weight=0.2,
            params={
                "command_name": "rel_height",
                "nominal_z_travel_mm": -40.0,
                "std": 0.2,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


class HumanoidRmaVelEstArmFlashSac(HumanoidRmaVelEstFlashSac):
    """FlashSAC baseline + v7/v21 arm perturbation/tracking (no phase rewards).

    Ports the arm EEDR mechanism from HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv7
    and the arm_action_l2 legs-first ramp from v21, without gait-phase wiring.

    Changes from HumanoidRmaVelEstFlashSac:
      - apply_legs_only_modifiers: splits action space; legs policy-controlled,
        arms driven by RandomizedOffsetJointPositionAction (random target, 20-100 step hold).
      - arm_joint_tracking reward (weight=2.0, std=0.3): incentivizes leg adaptation
        to arm disturbances.
      - arm_action_l2 reward: legs-first ramp -0.01@500N -> -0.5@5000N (v21 style).
        Prevents arm counterweight exploit while letting gait establish before full penalty.
      - target_arm_joint_pos obs: policy sees arm target to compute correction delta.
      - ee_payload DR (mass + offset): inherited from the base curriculum, which is the
        single definition of the payload range. This class used to override it with a
        wider, later ramp (0.6 kg @ 3000N); removed so there is one schedule to reason
        about and one cap to raise.

    Phase rewards (foot_phase_contact_match, foot_stance_slip_penalty, gait_phase obs)
    are absent. air_time weight stays at FlashSAC base (0.2).
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
    """Stronger arm delta suppression to close sim-to-real gap.

    Single change from HumanoidRmaVelEstArmFlashSac (v1):
      arm_action_l2 final curriculum weight -0.5 → -1.0 (2×).

    Motivation: v1 raw_arm_action_l2=0.62@14999 despite -0.5 penalty. At deployment,
    arm joints are overridden by IK controller — policy arm delta is discarded. Large
    arm delta wastes policy capacity and creates a training/deploy mismatch.

    Invariants preserved:
      - arm_joint_tracking weight=2.0 (stability invariant; do not reduce)
      - curriculum ramp start -0.01@500N unchanged (legs establish before penalty)
      - curriculum endpoint 5000*N unchanged (reaches final weight by iter ~5000)
      - no arm scale change

    REJECTED attempt (2026-05-10): weight=-2.0 (4× step). fell_over=0.1050@5k (>> 0.05),
    0.0150@15k (3× worse than v1=0.0050). raw_arm_action_l2 reduced 0.62→0.28 (55%) but
    ramp 4× steeper interferes with simultaneous locomotion learning at mid-training.
    Run: runs/HumanoidRmaVelEstArmFlashSacv5/2026-05-10_01-34-38_flash_sac/

    CORRECTED RUN (2026-05-10): weight=-1.0 (2×). fell_over=0.0500@5k (marginal),
    0.0000@15k ✓. raw_arm_action_l2=0.408@15k — floor established. arm_joint_tracking
    w=2.0 requires arm movement to track targets; equilibrium raw_l2≈0.41 regardless of
    penalty weight. Target <0.15 unachievable without reducing arm_joint_tracking (invariant).
    PROMOTED on stability. Deploy-side IK clipping preferred for further arm delta reduction.
    Run: runs/HumanoidRmaVelEstArmFlashSacv5/2026-05-10_02-45-45_flash_sac/

    Promotion criteria:
      fell_over ≤ 0.0050 @14999
    Rejection:
      fell_over > 0.05 @5k
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.curriculum["arm_action_l2"].params["weight_stages"][-1]["weight"] = -1.0


class HumanoidRmaVelEstArmFlashSacv9(HumanoidRmaVelEstArmFlashSacv5):
    """Yaw tracking — larger weight to push past promotion criterion.

    Single change from HumanoidRmaVelEstArmFlashSacv5:
      track_angular_velocity weight 1.5 → 2.5 (67% increase).

    Data from v7 (weight=2.0, 33% increase): error_vel_yaw 0.627→0.5611 (10.4% better),
    missed <0.55 criterion by 0.011. Mechanism confirmed: actual tracking improvement, not
    just reward scaling. error_vel_xy regressed 9% (0.598→0.6517) — yaw weight redistributes
    effort. v7 BORDERLINE: stability identical to v5, visual OK, but XY-yaw tradeoff is zero-sum.
    v9 extends to weight=2.5: expect larger yaw gain. Key question: does 2.5 break the XY-yaw
    balance further (reject), maintain it (similar to v7), or reach clean improvement?
    Reject if error_vel_xy > 0.70 (regression budget exhausted).

    Promotion criteria:
      fell_over ≤ 0.0050 AND error_vel_yaw < 0.55 @14999
    Rejection:
      fell_over > 0.05 @5k OR error_vel_xy > 0.70 @14999

    Run: runs/HumanoidRmaVelEstArmFlashSacv9/2026-05-10_10-23-12_flash_sac/
    | iter | fell_over | error_vel_yaw | error_vel_xy | track_ang_vel |
    |------|-----------|--------------|-------------|--------------|
    | 5k   | 0.0500 →  | 0.5838       | 0.7150*     | 1.2234       |
    | 10k  | 0.0050 ✓  | 0.5311       | 0.6898      | 1.2625       |
    | 15k  | 0.0050 ✓  | 0.5400 ✓     | 0.6989 ✓    | 1.3268       |
    *above 0.70 budget at 5k; recovered by 10k

    PROMOTED.
    5k gate: fell_over=0.05 NOT > 0.05 → gate passes, training continued.
    Yaw: 0.627→0.54 = 13.9% improvement. XY: 0.598→0.699 = 16.9% regression.
    XY 0.001 below budget ceiling — zero-sum tradeoff exhausted at w=2.5.
    Chain: v5(w=1.5)→v7(w=2.0,BORDERLINE)→v9(w=2.5,PROMOTED). w=3.0 would push XY past 0.70.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].weight = 2.5


class HumanoidRmaVelEstArmFlashSacv15(HumanoidRmaVelEstArmFlashSacv9):
    """Lateral stability stacked on v9 yaw improvement.

    Single change from HumanoidRmaVelEstArmFlashSacv9:
      body_ang_vel weight -0.01 → -0.05 (5× increase).

    v9 is PROMOTED: yaw=2.5, fell_over=0.0050@15k, error_vel_yaw=0.54 (13.9% better).
    v14 tests body_ang_vel increase from v5 baseline. v15 stacks the same change on v9.

    body_ang_vel penalizes roll+pitch ONLY — orthogonal to v9's yaw tracking improvement.
    No interaction between the two changes: yaw rotation (ω_z) is not penalized by bav.
    If v14 is PROMOTED (bav reduces fell_over), v15 should compound cleanly with v9's win.

    Promotion criteria:
      fell_over ≤ 0.0050 AND error_vel_yaw < 0.55 @14999
    Rejection:
      fell_over > 0.05 @5k OR raw_joint_torque > 9.0

    RUN: runs/HumanoidRmaVelEstArmFlashSacv15/2026-05-10_20-05-02_flash_sac/
    | iter  | fell_over | error_vel_xy | error_vel_yaw | raw_joint_torque |
    |-------|-----------|--------------|---------------|------------------|
    | 5k    | 0.0200    | 0.7014       | 0.5733        | 5.4              |
    | 10k   | 0.0017    | 0.6958       | 0.5405        | 5.5              |
    | 14999 | 0.0050 ✓  | 0.6980       | 0.5211 ✓      | 5.4              |

    Both criteria met: fell_over=0.0050 ✓, error_vel_yaw=0.5211 < 0.55 ✓.
    XY=0.6980 at budget ceiling (limit=0.70, margin=0.002). yaw=0.5211 vs v9=0.5293 (1.5% better).
    raw_bav: 0.0060/0.05=0.12 vs v9=0.0012/0.01=0.12 — unchanged. v9's yaw mechanism already
    achieved 0.12; bav penalty does not further reduce lateral sway in this regime. Stability
    improvement (0.0075→0.0050) is real but modest; mechanism may be reward shaping under DR
    rather than direct sway reduction.
    VERDICT: PROMOTED. Best combined yaw+stability result. XY at budget ceiling — no room for
    further yaw increase. Informs v18 (VTF compound on this base).
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["body_ang_vel"].weight = -0.05


class HumanoidRmaVelEstArmFlashSacv18(HumanoidRmaVelEstArmFlashSacv15):
    """Full compound: bav + yaw + dead zone fix.

    Single change from HumanoidRmaVelEstArmFlashSacv15:
      VelocityTrackingFailure termination, tracking_ratio=0.3 (proportional to cmd).

    v15 is PROMOTED: fell_over=0.0050, error_vel_yaw=0.5211 (best yaw ever), XY=0.6980
    (at budget ceiling). v18 adds VTF dead zone pressure — the final missing piece.

    v15 inherits: yaw weight=2.5 (from v9), bav weight=-0.05 (from v15). VTF adds episode
    termination for persistent velocity non-tracking. All three improvements are orthogonal:
      - bav: penalizes roll+pitch angular velocity (lateral stability)
      - yaw: increases yaw tracking reward (better turning)
      - VTF: terminates episodes where robot ignores low velocity commands (dead zone)
    No reward-budget interaction among the three.

    Key risk: v15 XY=0.6980 is at budget ceiling (0.70). VTF termination adds additional
    pressure on velocity tracking — policy must track velocity to avoid termination, which
    may tighten XY tracking (beneficial) or stress it past 0.70 (harmful). v16 showed VTF
    from v9 did NOT increase XY error (0.6840 vs 0.6882 baseline). Expect v18 similar.
    Reject if error_vel_xy > 0.70.

    Promotion criteria:
      fell_over ≤ 0.0050 AND vtf_rate > 0.0 AND error_vel_yaw < 0.55 @14999 + §7.1 dead zone eval
    Rejection:
      fell_over > 0.05 @5k OR error_vel_xy > 0.70 @14999

    RUN (2026-05-11): runs/HumanoidRmaVelEstArmFlashSacv18/2026-05-10_23-48-13_flash_sac/
      | iter  | fell_over | error_vel_yaw | error_vel_xy | vtf_rate     |
      |-------|-----------|--------------|-------------|--------------|
      | 5k    | 0.0017 ✓  | 0.5703       | 0.6863      | 0.025–0.065 ✓|
      | 10k   | 0.0033    | 0.5501       | 0.6756      | active        |
      | 14999 | 0.0000 ✓  | 0.5354 ✓     | 0.6881 ✓    | active        |
    All criteria met. PROMOTED. Current best policy.
    Key finding: §13.5 rule was too broad. VTF fires on bav+yaw compound base (vtf_rate=0.02–0.065)
    — contradicts hypothesis that bav prevents VTF. vtf_rate=0.0000 is specific to bav-ALONE base
    (v17). Rule corrected: VTF incompatible with bav-alone (v14), compatible with bav+yaw (v15/v18).
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
    """Torque budget relaxed for longer-arm model.

    Single change from HumanoidRmaVelEstArmFlashSacv18:
      joint_torque curriculum final weight -1e-3 → -5e-4 (halved).

    Robot arm model updated (2026-05-19): end_effector mass 0.001→0.288 kg (+28700%),
    wrist_1 mass 0.387→0.682 kg (+76%), elbow reach +70%. Effective gravity torque at
    arm joints ~3× higher → joint_torque penalty at -1e-3 fires continuously at reference
    pose, suppressing walking. HumanoidVelocityRMACNNShortEstimator.configure() doubled
    base (-5e-4→-1e-3) for old arm model; this reverts to base level for new dynamics.

    Model change analysis:
      - Old distal arm mass at elbow: ~1.00 kg
      - New distal arm mass at elbow: ~1.70 kg at 1.7× reach → ~3× gravity torque
      - v23 (-2e-3) barely survived @5k on old model (fell_over=0.0767); new model
        would be catastrophic at that weight. v18 (-1e-3) effectively ≈ -3e-3 equivalent.
      - -5e-4 (base env value before ShortEstimator doubling) restores appropriate budget.

    Promotion: fell_over ≤ 0.0050 AND error_vel_xy < 0.70 @14999
    Rejection: fell_over > 0.05 @5k
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.curriculum["joint_torque"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -1e-5},
            {"step": 3000 * self.num_steps_per_env, "weight": -5e-4},
        ]


class HumanoidRmaVelEstArmFlashSacv29(HumanoidRmaVelEstArmFlashSacv24):
    """Collision-free arm reference + locomotion gating on v27 baseline.

    MIGRATED (plan/smooth-conjuring-rainbow): the arm reference moved from the action term to a
    CommandTerm (CommandGatedArmGraphRefCommandTerm, command "arm_ref"); the action is now a thin
    residual (JointPosRefResidualAction). Dated failure-analysis notes below name the original
    action classes (CollisionFree*Action) as they existed when written — historical record.

    Single conceptual change from HumanoidRmaVelEstArmFlashSacv27:
      command-gated randomized arm reference → command-gated collision-free graph reference.

    Root cause addressed: linear interpolation between collision-free poses does NOT
    stay collision-free — paths clip torso/opposing arm. CollisionFreeArmAction
    navigates a precomputed graph (10k nodes, k=20 edges, all edges verified
    collision-free by arm-only MuJoCo contact check).

    v28 failure analysis (v28 branched from v24):
      1. Reset bug (primary): at episode reset, sim resets arm to q_default but
         CollisionFreeArmAction set current_target to a random graph node →
         arm_joint_tracking error from episode start → tracking 0.66 vs v24's 1.44.
         Fix: _reset_envs now always snaps to _q_default_node (precomputed at init).
      2. steps_per_edge=50 (secondary): 1s per hop, matches original min_steps dwell time.

    v29_v1 failure analysis (ungated CollisionFreeArmAction, trained 2026-05-24):
      Dropped locomotion gating when switching from CommandGatedRandomizedArmAction →
      CollisionFreeArmAction. Effectively a two-change vs v27: added collision-free AND
      removed gating. Results @14999: fell_over=0.0100 (gate fail >0.005), XY=0.715,
      yaw=0.573, torque=8.20, self_coll=−0.072 — ALL worse than v27 baseline.
      Fix: CommandGatedCollisionFreeArmAction restores gating on top of collision-free graph.

    v29 retrained run (2026-05-24_20-21-38, PROMOTED 2026-05-25):
      Fixed generalization bug where locomotion forced target to default pose. Arm now
      moves slowly (1.6s per hop) during locomotion and normal speed (1.0s) stationary.
      Results @14999:
        - fell_over=0.0000 ✅ (Passes ≤0.005 gate)
        - error_vel_xy=0.6734 ✅ (Passes ≤0.674 no-regression vs v27=0.6739)
        - error_vel_yaw=0.4873 ✅ (Best ever, improved vs v27=0.5003)
        - ep_self_collisions=−0.0307 ✅ (35% reduction vs v27=−0.0474)
        - raw_joint_torque=6.4000 (Stable range)
        - ep_arm_joint_tracking @10k = 1.6911 ✅ (2.65% degradation vs v27, within 10% limit)
      New best policy. Replaces v27 as active best.
      Run: runs/HumanoidRmaVelEstArmFlashSacv29/2026-05-24_20-21-38_flash_sac/
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
    """Relative-std tracking reward — low-vel dead-zone fix Option A.

    Single change from v29: swap track_linear_velocity (fixed std=0.5) for
    track_linear_velocity_relative (std_rel=0.5, std_min=0.3). Weight unchanged at 2.0.

    Problem: fixed std=0.5 Gaussian gives exp(−0.16)=0.852 standing reward at cmd=0.2 m/s,
    so standing loses only 0.148/step → locally optimal. Relative std scales the tolerance
    proportionally to cmd magnitude, giving equal relative incentive at all speeds.

    Standing penalty comparison (standing still vs tracking perfectly):
      cmd=0.2 m/s: fixed→0.148 loss,  relative→0.644 loss (4.4×)
      cmd=0.5 m/s: fixed→0.632 loss,  relative→0.865 loss
      cmd=1.0 m/s: both→0.982 loss (std_eff=0.5, identical to base)

    std_min=0.3 m/s: protects against foot-liftoff vz transients at cmd≈0.
    Peak ~0.2 m/s transient at std_min=0.3 → exp(−0.04/0.09)≈0.64 (tolerable).
    At std_min=0.15 the same transient → 0.17 (suppresses gait).

    Note: ep_track_linear_velocity NOT comparable to v29 — reward formula changed.
    Use twist_error_vel_xy as orthogonal ground-truth metric.

    Results @14999 (2026-05-25): fell_over=0.0050, error_vel_xy=0.6053 (−10.1% vs v29),
    error_vel_yaw=0.4956, raw_joint_torque=6.80, raw_action_rate_l2=1.0454. All SOP gates pass.
    PROMOTED ★ (2026-05-26): cleanest gait among dead-zone-fixed variants — least foot dragging at
    low-speed commands (visual eval vs v31/v32 2026-05-26). v31/v32 not promoted: better XY metric
    but gait regression (foot dragging at low speed). Note: metric_slip_velocity_mean does NOT detect
    low-speed dragging — high-speed episodes dilute the average (v31 shows 0.1200, v32 shows 0.1162,
    both below v29=0.1290, misleadingly clean despite visible dragging).
    Run: runs/HumanoidRmaVelEstArmFlashSacv30/2026-05-25_18-09-58_flash_sac/
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
    """STATUS — PROMOTED (2026-06-16): fixed the v54 fall blocker (0.00190→0.00002, 95×) at its
    mechanism — per-STEP sample-mixing. NOT freeze-then-DAgger (the teacher stays live).

    RE-PARENTED 2026-06-16 to inherit v30 (the gold direct-trained deploy baseline) DIRECTLY,
    inlining the full L2T stack (v30Teacher privileged-obs + 2× energy, v51Teacher clean integral,
    v51L2T student graft + distillation, v54L2T α_mix=0.5 + frame-ring) into configure /
    flash_sac_configure below — byte-identical config to the former v54L2T-chained v59 (verified by
    config dump-diff). The intermediate classes (v30Teacher/v51Teacher/v51L2T/v54L2T) are retained
    for their other descendants (v54); v59 no longer inherits their future edits.

    Single change from v54L2T: episode_level_mixing False → True (the executed driver, student vs
    teacher, is drawn ONCE per episode and held to the next reset, instead of a fresh per-step coin).

    Why. v54's α_mix=0.5 is a per-step Bernoulli (flash_sac_runner mix_mask, resampled every step).
    Mean consecutive student run = 1/(1−α_mix) = 2 steps; an episode is 1000 steps (50 Hz × 20 s). So
    the teacher yanks the rollout back every ~2 steps → the replay's "student" states are ~2-step
    excursions tethered to the teacher distribution, never the sustained drift a 1000-step pure-student
    deploy reaches. Two train≠deploy consequences: (a) the buffer never holds deploy-horizon student
    states → no recovery supervision where play actually goes (the 7.3× covariate-shift action-error
    blow-up — probe_bc_agreement: teacher-driven MSE 0.0069 → student-driven 0.050); (b) the v30 student
    obs carries last_action, which under per-step mixing is the TEACHER's action ~α_mix of steps but is
    always the student's own at deploy → an obs-channel covariate shift / crutch. Per-episode driving
    rolls the student forward coherently for the full horizon and the teacher relabels those states →
    true DAgger coverage on the deploy distribution. α_mix now ⇒ fraction of envs running pure-student
    for the whole episode.

    Why not freeze-then-DAgger (rejected path). That freezes the teacher to push α_mix→1; here the
    teacher keeps doing RL on the teacher-driven envs, and (bonus) its critic/actor now also see full
    student-distribution states → its relabels get reliable exactly where the student needs them. One
    change, one co-trained run — no two-stage warm-start.

    Risk. Sustained pure-student episodes are MORE off-policy for the teacher than fragmented per-step
    mixing (a student-driven env may spend a whole episode falling). Mitigated by the inherited warmup
    ramp (α_mix 0→0.5 over student_action_prob_warmup_iters): early episodes mostly teacher-driven (clean
    teacher RL), student-episode fraction rises as the student improves. If the teacher regresses
    (probe_l2t_freerun / probe_teacher_deadzone / return), the single-variable follow-ups are a lower
    student_action_prob_max or a longer warmup — separate runs.

    Result (gate PASSED — v59 vs v54 06-15, both model_0015000, probes at egl; v30 = gold direct-
    trained deploy baseline 2026-06-15_09-38-24):
      - probe_l2t_freerun student fall 0.00002 vs v54 0.00190 → 95× drop; teacher fall 0.00000 (no
        regression). v59 student ≈ gold v30 (0.00006) on deploy stability.
      - vx dead-zone: v59 ≈ v30 (ratio ~0.94 @ cmd≥0.3; stands @ cmd≤0.1). v54's low-cmd ratio>1
        (1.7–2.1) was a TETHERING ARTIFACT — the stateless student parroting the teacher's integral
        overshoot into the per-step-tethered buffer, NOT a deployable capability. v30 (no integral)
        has the SAME v_min dead-zone, so v59 did NOT regress vs the real deploy target; the earlier
        "vx regression vs v54" measured against the artifact.
      - wz dead-zone: v59 BEATS v30 at mid yaw (ratio 0.40/0.63/0.82 @ cmd 0.2/0.3/0.5 vs v30
        0.03/0.11/0.62), stable (term≈0). v54 worst (under-rotates 0.51 @ 0.8, term up to 0.0047).
      - gait within noise of v30 (softest foot impact 0.0482). Training-time fell_over/illegal_contact
        higher than v54 = expected episode-mixing exploration cost (sustained off-policy student
        episodes), does NOT transfer to deploy (freerun bad-term 0.00002).
    Verdict: v59 student matches/beats direct-trained gold v30 on every deploy axis (stability, vx,
    yaw) while keeping privileged-critic co-training. Export parity bit-exact (probe_export_parity,
    max|Δ|=0) → deploy-ready. Low-cmd dead-zone is intrinsic to the stateless velocity policy (v30 has
    it) — closing it needs a student integral STATE (stateful recurrent student; explicit ∫(cmd−v_est)dt
    rejected = biased-v_est axis L2T deleted). SHELVED 2026-06-16 as marginal (v59 already matches v30) —
    see MEMORY "L2T fall gap CLOSED" entry for the full design verdict.

    Inheritance: …→v29→v30→v59L2T (re-parented; formerly …→v51L2T→v54L2T→v59L2T).
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


class HumanoidRmaVelEstArmFlashSacv82L2T(HumanoidRmaVelEstArmFlashSacv59L2T):
    """STATUS — 2-SEED-CONFIRMED WIN, PROMOTE CANDIDATE (2026-06-21): freeze-iter sweep PEAK.
    Single lever vs v59L2T (`freeze_teacher_after_iters` 0→12000).

    Re-parented to v59L2T directly (2026-06-22): the freeze-iter bracket intermediates v79L2T
    (freeze@10000) / v80L2T (freeze@10k + estimator-aux, REJECTED) were removed as superseded;
    v82 is the bracket PEAK and overrode the freeze value anyway, so inheriting v59L2T + setting
    freeze=12000 is net-identical config. The bracket record below is preserved here.

    Resolves the freeze-iter bracket (MEMORY): freeze@8000 (v81, 7k stage-3) REJECTED — earlier
    freeze = less-converged teacher → frozen BC labels worse → yaw regressed. freeze@10000 (v79)
    was prior best. freeze@12000 (THIS, 3k stage-3) is the PEAK: 2-seed beats v79 on the forward
    dead-zone (vx@0.20 1.01/0.97 vs v79 0.83/0.95, both seeds beat both v79 seeds) while HOLDING
    yaw ~1.0 (≈ v79) and falls~0. freeze@14000 (1k stage-3) REGRESSED vx (coverage too thin) →
    bracket 8k<10k<12k>14k, optimum=12000.

    Mechanism: forward dead-zone closure needs BOTH (a) converged frozen teacher labels and (b)
    enough stage-3 on-distribution (α_mix=1.0) coverage. 12k's 3k-iter stage-3 is the sweet spot —
    later (8k) = bad labels, earlier-ending (14k) = too little forward coverage. Yaw is gyro-observed
    / coverage-cheap so it stays flat across the bracket; forward is the binding constraint.

    Core-HP stacking (num_steps=1, tau=0.02) on top of this REJECTED — does not compound (MEMORY,
    v80 same-gap lesson). This single freeze-iter lever is the win; no further stacking.

    Originally realized via `--flash_sac_hp_override freeze_teacher_after_iters=12000` override on v79L2T
    (run dirs …v79L2T/2026-06-21_02-48-58_…freeze12k_s0 + …03-55-39_…freeze12k_s1); codified here
    as a named class for reproducibility + deploy export.

    Inheritance: …→v30→v59L2T→v82L2T (v79L2T freeze@10k record folded into the bracket note above).
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.freeze_teacher_after_iters = 12000     # v82: stage-3 starts at 12000 (3k stage-3) — bracket PEAK


class HumanoidRmaVelEstArmFlashSacv54L2T(HumanoidRmaVelEstArmFlashSacv59L2T):
    """STATUS — PROMOTED / KEEPER (2026-06-14, refactored 2026-06-16): the deploy-best L2T base.
    v59L2T inlines the full L2T stack on v30 with per-episode mixing; v54L2T re-uses that stack
    with per-STEP Bernoulli mixing instead. Behaviourally a 1-flag override of v59L2T — see
    flash_sac_configure below.

    Historical context (preserved from the original v54L2T before re-parenting): DAgger α_mix
    0.2→0.5 closed the vx dead zone (ratio ~0.92 down to cmd 0.05) AND gave the lowest L2T
    student fall rate (0.00145) from the PLAIN 0.4 s proprio window. Decisive finding: the dead
    zone was COVARIATE SHIFT (state coverage), not temporal representation. Every behavioural
    knob stacked on top (v55 multi-scale, v56 q_coef, v57 critic-history, v58 KL) REGRESSED →
    frame-ring is the only non-regressing addition. Falls (0.00145 vs teacher 0.00000) were the
    binding deploy blocker (now closed by v59L2T's per-episode mixing → 0.00002).

    frame_ring_history=True is set by v59L2T (storage-only replay optimisation, ~18× less
    actor-replay VRAM, byte-identical behaviour). v59L2T inlines v30Teacher + v51Teacher +
    v51L2T student-graft into its configure() — v54L2T inherits it all by inheritance.

    Inheritance: …→v30→v59L2T→v54L2T.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        # v59L2T's flash_sac_configure sets the full L2T stack + α_mix=0.5 + frame-ring. v54L2T
        # is the original per-STEP mixing variant (the DAgger coverage finder); v59L2T's
        # per-EPISODE mixing is the keeper (95× fall drop vs v54). Override the single flag.
        super().flash_sac_configure(sac_cfg)
        sac_cfg.episode_level_mixing = False  # v54: per-STEP Bernoulli (v59's per-episode is the keeper)


class HumanoidRmaVelEstArmFlashSacv72L2T(HumanoidRmaVelEstArmFlashSacv59L2T):
    """STATUS — CONFIRMED marginal YAW-only WIN (2026-06-18, 2 seeds + the α0.9 condition). Single
    change vs v59L2T — student_action_prob_max 0.5 → 0.8 (fraction of envs running PURE-STUDENT per
    episode). BC-only distillation (no PG), episode-level mixing, live teacher, warm-start gate 3000,
    frame-ring all inherited byte-for-byte from v59L2T.

    RESULT @15k (probes egl, within-run fair comparison): student wz low-cmd ratio @0.10/0.20/0.30 =
    seed1 0.56/0.67/0.80, seed2 0.40/0.48/0.64 vs v59 0.25/0.38/0.58 → BOTH seeds beat v59 (~1.5-2×
    @≤0.30, monotone across all 6 yaw cmds; direction robust, magnitude noisy = low-cmd-ratio variance).
    vx (forward) dead-zone INTACT (observability wall, no regression); falls ~0; teacher HELD (T-v72s2
    wz ≈ T-v59, 20%-teacher-driving keeps RL healthy). BC covariate-gap gate FAILED (student-driven
    RMSE/|a_t| noise across seeds) — yaw tracking improves without aggregate-RMSE change (coverage fixes
    the small low-cmd-yaw discharge-state subset). Mechanism: more pure-student episodes → buffer holds
    low-cmd yaw discharge states → teacher relabels → student reproduces intermittent-yaw-stepping (yaw
    rate gyro-observed → closeable; forward needs a lin-vel estimate → walled). SAME yaw ceiling as
    v60L2T (window-40) via an independent mechanism; α0.8 is the CHEAPER yaw-closer (one knob, no
    window/replay cost). α0.9 (v73L2T) REJECTED (plateau + teacher starvation) → coverage ceiling = 0.8.
    Stacking with window-40 (v74L2T) adds only a small mid-yaw bump (ceiling mostly shared). PROMOTABLE
    as the deploy student over v59 (cheap yaw upgrade). See MEMORY for the full record.

    PREMISE (new MEASURED evidence the v59 author lacked). v59 PROMOTED at α_mix=0.5 with a LIVE teacher
    and explicitly rejected freeze-then-DAgger (keep teacher co-training so its critic/actor also see
    student-dist states → reliable relabels, one run). This run does NOT revisit the freeze path — it
    raises the SAME documented coverage knob v59 names as its tuning lever. The new datum:
    probe_bc_agreement on v59 model_0015000 shows the deploy (student-driven) distribution still carries
    a 1.7× BC-error blow-up vs the supervised (teacher-driven) distribution (RMSE/|a_t| 0.355→0.610;
    MSE 0.00528→0.01608), with the worst dims on the leg/hip channels (0,1,4,5,10,11). So at 50%
    coverage the live teacher still under-supervises the full-horizon student drift states — exactly the
    covariate shift DAgger exists to close. Raising the pure-student episode fraction to 0.8 puts more
    deploy-horizon student trajectories into the buffer for the teacher to relabel → shrinks the
    student-dist BC gap → fewer deploy falls / better deploy tracking, with NO architecture change and
    NO PG (isolates the coverage effect from the shelved PG lever).

    Why 0.8 (not 1.0). α_mix→1 starves the live teacher of on-policy data → forces freeze_teacher (the
    v59-rejected two-stage path, 2 coupled flags). 0.8 keeps 20% teacher-driven envs/episode so the
    teacher's RL + relabels stay live (v59's design preserved) while ~doubling deploy-state coverage.
    If 0.8 helps without teacher regression, 1.0+freeze is the natural follow-up (separate run, then
    with "0.8 worked" as its premise).

    Risk. More sustained pure-student episodes = more off-policy for the teacher (a student env may
    spend a whole episode falling). Mitigated by the inherited α-mix warmup ramp (0→0.8 over
    student_action_prob_warmup_iters) and 20% retained teacher-driving. If the teacher regresses, the
    single-variable fallbacks are a lower α_mix or a longer warmup (separate runs).

    Gate (vs v59L2T at the same horizon; probes at egl): (1) probe_bc_agreement student-driven RMSE/|a_t|
    DROPS below v59's 0.610 toward the teacher-driven 0.355 (the mechanism check — coverage closed the
    gap); (2) probe_l2t_freerun / probe_velocity_deadzone deploy fall ≤ v59 AND vx/wz tracking ≥ v59;
    (3) probe_teacher_deadzone teacher NOT regressed (the off-policy risk check). Promote iff the
    deploy-distribution BC gap shrinks without a teacher or tracking regression.

    Inheritance: …→v29→v30→v59L2T→v72L2T. Warm-start v59 model_0003000.pt. Plan: plan/L2T_STUDENT_RL_REVISIT_PLAN.md.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)   # v59L2T: use_distilled_student, α_mix=0.5, episode_mix, BC-only
        sac_cfg.student_action_prob_max = 0.8   # 0.5→0.8: more pure-student episodes for the teacher to relabel


class HumanoidRmaVelEstArmFlashSacv59L2TActuatedCam(HumanoidRmaVelEstArmFlashSacv59L2T):
    """v59L2T + actuated twin-D435 head camera with random STEP gaze commands.

    The deploy-best L2T policy (v59) made to also actuate its dual yaw+pitch head gimbal while
    locomoting. Both the privileged teacher (actor group) and the deployable proprio student output
    the 4 camera dims and observe the gaze command, so the REAL robot drives its own gimbal at
    deploy. Mirrors HumanoidRmaVelEstArmFlashSacv30ActuatedCam on the v59 base.

    Single conceptual change vs v59L2T = "add the camera":
      1. Robot entity uses head_camera="actuated" (+4 DOFs: cam_[yaw|pitch]_[left|right]).
      2. Arm reference rebuilt with _ARM_ONLY_PATTERN to exclude camera joints — required because
         the arm graph reference loads a collision graph that has no camera joints (including them
         raises ValueError). Same CommandGatedArmGraphRefCommandTerm v59 uses (via v29), so the arm
         behaviour is unchanged; only the camera joints are excluded.
      3. Camera reference "camera_ref" (GazeRefCommandTerm): random-WALK gaze. Policy residual
         "joint_pos_camera" (JointPosRefResidualAction) adds a 4D delta on top. Each episode seeds
         absolute-uniform in the ROM box (clamp ±4.7124 rad yaw = ±270°, ±1.5708 rad pitch = ±90°),
         then every 30–120 steps takes a bounded delta step (±1.0 rad yaw, ±0.4 rad pitch, L/R
         independent) clamped back into the box, so the full yaw ROM is explored without the dead
         tracking gradient a 4π absolute jump would cause. joint_target = joint_ref + policy * scale.
         Velocity is bounded by the camera actuator, not a command limiter — see the RS05 fix below.
         Joint axis convention: yaw axis (0 0 -1), pitch_left (0 -1 0) / pitch_right (0 1 0); yaw
         ROM ±270° (±4.7124), not full ±2π — set in head_camera_dual.xml.
      4. camera_joint_tracking reward (weight=1.0, std=0.3): drives both policies' camera joints to
         the gaze command. Reuses arm_joint_tracking on the joint_pos_camera term.
      5. target_camera_joint_pos obs (the gaze command) added to BOTH actor (teacher) and student
         (deploy policy) so each can condition its 4 camera dims on the command. Critic omitted (the
         privileged value net does not need the command). The student's joint_pos/joint_vel/last_action
         auto-grow by 4 (terms resolve against the camera robot at build; observe_full_joints=True).

    Camera motor / "too fast" fix (humanoid_v21_constants.CAMERA_MOTORS, applied globally to the
    actuated robot): the gimbal is the same RS05 motor as wrist_3. Its damping was kd=0.1 (a bug:
    the gimbal slewed tens of rad/s and slammed to a stepped target in ms). Now kd=1 (kept kp=5,
    intentionally soft so the light gimbal stays compliant and does not fight base motion): heavily
    overdamped (no oscillation), peak velocity capped at ~effort/kd = 4.4/1 ≈ 4.4 rad/s. This single
    param fix is the speed bound; with a step command train ≡ deploy and the same actuator bounds
    velocity in sim and on hardware.

    Policy action space: 13D legs + 14D arms + 4D camera = 31D. rma_cnn student / SAC nets auto-size
    from the obs+action dims; the deploy export grows to 31D and the deploy ring auto-sizes.

    Camera-joint reward scope: DRIVING = camera_joint_tracking only. The inherited whole-body
    penalties action_rate_l2 (−0.5), joint_torque (−2e-3), joint_power (−4e-4) span the cam joints
    (no joint filter) — tiny on a light gimbal and a soft "don't whip the camera" term; left as-is
    (matches v30ActuatedCam). pose/dof_pos_limits are leg-only; arm_joint_tracking/arm_action_l2 are
    arm-only.

    Schedule (inlined, = v61L2T Exp 0): student_start_iters=3000 teacher-warmup gate — distill the
    camera+locomotion student only after the teacher's action label stabilizes (see flash_sac_configure).

    Promotion gate (vs v59, model_0015000, probes at egl): locomotion NOT regressed
    (probe_l2t_freerun student fall ≈ v59 0.00002; probe_velocity_deadzone vx/wz ≈ v59); camera
    learned (ep_camera_joint_tracking > 0.5); probe_export_parity bit-exact on the 31D student export.

    Inheritance: …→v30→v59L2T→v59L2TActuatedCam.
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
        # see HumanoidRmaVelEstArmFlashSacv61L2T docstring + plan/L2T_STUDENT_RL_REVISIT_PLAN.md
        # Exp 0). Distilling the camera+locomotion student before the teacher's action label
        # stabilizes chases a moving target + pollutes the student obs-normalizer.
        super().flash_sac_configure(sac_cfg)
        sac_cfg.student_start_iters = 3000


class HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam(HumanoidRmaVelEstArmFlashSacv59L2TActuatedCam):
    """UNIFIED-BEST (2026-06-22): the deploy-best combined arm+head-cam+locomotion policy,
    freeze@11000. Supersedes the freeze@12000 point (the legs-only bracket PEAK) on the unified base.

    Single lever vs v59L2TActuatedCam (`freeze_teacher_after_iters` 0→11000). The legs-only track
    peaked at 12000. A full freeze-iter bracket {10,11,12,13,14}k ON the unified cam
    base (probe/velocity_deadzone 128env/120win/70warmup vs v59cam) showed the legs-only optimum
    does NOT transfer: on the 31D policy (13 legs + 14 arm + 4 cam) EARLIER freeze (more stage-3
    on-distribution coverage) wins — peak at 11k, not 12k.

    Multi-seed gate means (f10k 2-seed, f11k 2-seed, f12k 3-seed; vs v59cam base):
      vx@0.20: base 0.52 | f10k 0.92 | f11k 0.90 | f12k 0.83 | f13k 0.82 | f14k 0.88
      wz@0.20: base 0.59 | f10k 0.73 | f11k 0.98 | f12k 0.77 | f13k 0.79 | f14k 0.72
      wz@0.10: base 0.48 | f10k 0.55 | f11k 0.79 | f12k 0.59
    f11k cleanly beats f12k on BOTH axes (vx 0.90>0.83, wz 0.98>0.77), both seeds tight (vx
    0.89/0.90, wz 0.92/1.03 = low variance), term_rate ≤0.0003 (stable), cam_track 0.83 +
    arm_track 1.69 retained. f10k matched vx but yaw was weak/noisy (wz 0.83/0.64). So 11k is the
    best-balanced unified optimum: near-top forward AND dominant yaw dead-zone closure.

    Mechanism: the higher-dim unified policy (arm+cam tracking + more action dims) needs MORE
    stage-3 (α_mix=1.0 all-student) on-distribution coverage to close the loco dead-zone than the
    legs-only student; frozen-teacher label-staleness costs less here than the extra coverage buys.
    So the bracket monotone direction FLIPS vs legs-only (legs: 8<10<12>14; unified: 14<13<12<11≈10,
    11 best-balanced). Single-variable freeze-iter; no stacking (v80 same-gap lesson).

    Realized via `--flash_sac_hp_override freeze_teacher_after_iters=11000` on v59cam (runs
    `…_flash_sac_ser16_f11k_s0` + `…_f11k_s1`); codified here as a named class for reproducibility
    + deploy export. Those two checkpoints ARE the deployable best-policy weights (identical config).

    Inheritance: …→v30→v59L2T→v59L2TActuatedCam→v83L2TActuatedCam.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.freeze_teacher_after_iters = 11000     # v83: unified-base optimum (beats 12k both axes, 2-seed)


class HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam(HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam):
    """NO-TEACHER BASELINE for v83L2TActuatedCam: train the deployable proprio policy DIRECTLY with
    SAC, no privileged teacher, no distillation, no sample-mixing/freeze curriculum.

    Purpose. Isolates the value of the whole L2T teacher→student distillation stack. The env is
    BYTE-IDENTICAL to v83 (same rewards incl. track_integral_error + 2× energy penalties, same
    31D action space, same actuated-cam + arm graph-ref setup, same privileged critic group). The
    ONLY change is the training path: the RL actor is repointed from the privileged teacher obs
    (`actor` group: GT lin_vel, foot contacts, integral) onto the deployable `student` group
    (corrupted proprio + gaze/arm command). So the RL actor IS the deploy policy — trained from
    scratch on deploy-realistic noisy obs against a PRIVILEGED critic (asymmetric actor-critic).

    Asymmetric, not blind. `critic_obs_group` stays the v83 privileged critic (unchanged): the value
    net keeps GT state, the actor is blind-deploy. This removes ONLY the teacher-actor + distillation
    — the privileged critic that L2T also uses is retained, so the comparison isolates
    "distillation vs direct RL" rather than confounding it with a critic-privilege change.

    Mechanism. `use_distilled_student=False` makes the runner skip ALL student machinery
    (self.student=None; imitation/mix/freeze gated off, see runner.py L330). `actor_obs_group=
    "student"` then trains the standard SAC actor on the deploy group; deploy export is that actor
    (the non-L2T default path). The `actor` (teacher) obs group is still built by the env but unread
    (harmless compute). freeze_teacher_after_iters / student_start_iters are no-ops here (gated on
    use_distilled_student) but reset to 0 for a clean config dump.

    Inheritance: …→v30→v59L2T→v59L2TActuatedCam→v83L2TActuatedCam→v83StudentOnlyActuatedCam
    (env inherited verbatim; only flash_sac_configure diverges).

    *** PROMOTED 2026-07-08 as v83 ROBUSTNESS SUCCESSOR. ***
    2-seed verified (2026-07-08). Direct-RL policy BEATS v83L2T on every mass-bias level tested:

      metric             v83 f11k     v83StudentOnly 2-seed mean     delta
      mass frac_below @×1.0   0.008      0.000                             −∞
      mass frac_below @×1.1   0.012      0.000                             −∞
      mass frac_below @×1.2   0.168      0.000                             −∞  (v108 = 0.004)
      mass frac_below @×1.3   0.316      0.000                             −∞  (v108 = 0.198, 200× worse)
      vx @0.15 ratio          0.37       0.375                             ~flat
      vx @0.20 ratio          0.73       0.77                              +5%
      wz @0.20 ratio          0.71       0.82                              +16%
      wz @0.30 ratio          0.82       0.88                              +7%
      term_rate (any axis)    ~0.0001    0.0000                            perfect

    Mechanism that wins. v83 L2T student = pure behavior cloning (BC ceiling = conditional mean of
    teacher | student_obs); collapses at low cmd / extreme mass. StudentOnly removes the teacher
    and trains the deploy actor directly via SAC against the privileged critic (asymmetric AC) — no
    BC ceiling. Single change `use_distilled_student=False`. Wall-time = 57 min to 15k iter (~9%
    faster than v83L2T, per-iter time equal). See memory/l2t_studentonly_speedup_measure.md.

    Decision path. Initially promoted v108 (body_mass ×2 DR widening) on -42×@×1.2 result, then ran
    cross-comparison vs StudentOnly: StudentOnly is 200× stronger at ×1.3 (0.000 vs 0.198 frac_below).
    v108 retained as secondary (tracking-primary backup, vz +62%@0.15 cleanest single-lever gain).

    Deploy ckpts (2-seed, both 21 MB, 15k iter):
      s0: runs/HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam/2026-07-08_13-53-08_flash_sac/model_0015000.pt
      s1: runs/HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam/2026-07-08_18-13-16_flash_sac/model_0015000.pt

    Detail: memory/l2t_studentonly_vs_v108.md (probes), memory/l2t_studentonly_baseline.md (origin).

    *** POST CAM-YAW-FLIP RETRAIN + MASS-BIAS RECONFIRM 2026-07-10. ***
    Head-cam yaw axis flipped 2026-07-09 (memory/cam_yaw_flip_retrain.md); all actuated-cam keepers
    retrained on ser16. Re-ran this class as a fresh s0 on the post-flip env
    (`HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam_1x_energy`, run_name descriptive — this
    class always uses 1× energy weights via configure() below; run_name tag = "the energy that was
    kept at 1× after the v83 2× was reverted", not a code delta). 15k iter, ~57 min wall, single seed.

    Mass-bias probe (mj_envs/probe/mass_bias_sweep.py, 256 envs × 2000 steps, single seed each,
    post-flip env, body_mass DR clamped to fixed scale s):

      metric                  scale   L2T f11k post-flip   StudentOnly 1xE s0   winner
      falls/step              ×1.0    0.00001               0.00009              L2T
      falls/step              ×1.1    0.00004               0.00011              L2T
      falls/step              ×1.2    0.00037               0.00016              1xE (2.3×)
      falls/step              ×1.3    0.00147               0.00029              1xE (5.1×)
      fell_over count         ×1.2    58                    2                    1xE (29×)
      fell_over count         ×1.3    248                   14                   1xE (18×)
      eplen ↑                 ×1.3    457                   842                  1xE (1.8×)
      track_lin               ×1.0    1.51                  1.14                 L2T (+0.40)
      track_ang               ×1.0    1.97                  2.05                 1xE (+0.09)

    In-distribution (×1.0–1.1) L2T still wins on falls + raw tracking. Under mass shift (×1.2–1.3)
    StudentOnly dominates every stability metric (5–18× fewer falls, ~2× longer eplen) — the
    robustness story from the 2026-07-08 promotion SURVIVES the cam-yaw flip.

    Why L2T collapses out-of-distribution. L2T's student is a BC clone of a teacher that was
    privileged (GT lin_vel, foot contacts, integral) — the BC target never saw the deploy-distribution
    mass shift, so the proprio→action mapping breaks under shift. StudentOnly's actor explored the
    full training DR (mass 0.85–1.15, plus com/ee/payload/friction) on deploy-realistic proprio
    directly, so the proprio→action mapping generalizes. The L2T in-distribution tracking win is
    largely TEACHER-INHERITED competence, not student-learned capability — at deploy on hardware
    near the trained mass StudentOnly is a strict win; L2T's tracking advantage is real but only on
    the training distribution.

    New deploy ckpt (post-flip, 21 MB, 15k iter):
      s0: runs/HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam/2026-07-10_12-32-36_flash_sac_HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam_1x_energy/model_0015000.pt

    In-distribution v83L2T f11k still wins raw tracking reward — keep both as deploy options; pick
    StudentOnly for unknown-payload / field / mass-varying hardware, pick L2T for calibrated lab.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)       # v83 L2T stack (distilled student, freeze@11000)
        sac_cfg.use_distilled_student = False      # remove teacher/student distillation
        sac_cfg.actor_obs_group = "student"        # RL actor = deployable proprio group (asymmetric; critic stays privileged)
        sac_cfg.freeze_teacher_after_iters = 0     # no-op w/o distillation; reset for clean config
        sac_cfg.student_start_iters = 0            # no-op w/o distillation; reset for clean config

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        # Energy penalties (joint_torque, joint_power) change back to original (2x->1x)
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


class HumanoidRmaVelEstArmFlashSacv123(HumanoidRmaVelEstArmFlashSacv30):
    """STANDALONE NO-L2T TWIN of v83StudentOnlyActuatedCam — same asymmetric direct-RL training,
    zero L2T lineage.

    Why. v83StudentOnlyActuatedCam is the promoted deploy keeper: the deployable proprio policy is
    trained DIRECTLY with SAC against a privileged critic (asymmetric actor-critic), no teacher, no
    distillation. But it INHERITS ...v83L2TActuatedCam, so its env still builds a privileged TEACHER
    `actor` obs group that is fed to nothing (dead compute; enable_corruption=False so it draws no
    RNG — the RL actor reads the `student` group, the critic reads `critic`, the teacher `actor`
    group is never indexed by the runner), and its MRO drags the whole L2T stack while its
    flash_sac_configure sets L2T flags only to reset them. This class removes ALL of that: it inherits
    the gold non-L2T deploy baseline v30 directly and re-inlines ONLY the non-distillation env deltas
    StudentOnly actually uses. There is a SINGLE deploy obs group named `actor` (= the RL actor = the
    deploy policy), a privileged `critic`, and NO teacher group / NO student graft / NO distillation.

    Byte-identical to StudentOnly (control, verified by config dump-diff — no new training run). The
    `actor` group here is content-identical to StudentOnly's `student` group (deploy proprio + gaze
    obs, no command_integral); the `critic` group and all rewards/commands/curriculum/actions/DR match.
    The only env difference is the absent teacher group; the only agent difference is the actor group
    rename (student->actor, equivalent content) plus StudentOnly's set-then-gated-off L2T flags. This
    confirms StudentOnly's promoted numbers were uncontaminated by the dead teacher group.

    Env deltas re-inlined on the v30 base (dropping every teacher/student/distillation piece):
      - Clean integral-error twist command (dr_bias=dr_noise=0) + command_integral obs on the CRITIC
        ONLY (the deploy actor never sees the integral, matching StudentOnly's student group) +
        track_integral_error reward. Ported from v59L2T.configure step (4), minus the actor obs.
      - Actuated twin-D435 head camera + random-walk gaze (robot swap, arm_ref/camera_ref commands,
        residual cam/arm actions, camera_joint_tracking reward). Ported from v59L2TActuatedCam.configure
        items 1-4; the item-5 gaze obs is added to the deploy `actor` group only.
      - Energy stays 1x (v30 base default; StudentOnly also runs 1x).

    All flash_sac optimization knobs (distributional critic, zeta noise, n-step, buffer/batch, tau,
    alpha, sequence encoder, per-q target, ...) come from the shared HumanoidRmaVelEstFlashSac base
    that BOTH v30 and StudentOnly inherit — nothing to re-list. The only optimization knob the L2T
    layer changed vs that base is use_velocity_estimator (True->False), overridden below.

    Inheritance: ...->v29->v30->v123 (NO L2T).
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


class HumanoidRmaVelEstArmFlashSacv83StudentOnlyCriticSeqCam(
        HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam):
    """CRITIC-HISTORY (encoder) probe on the promoted v83StudentOnly baseline. Single lever: give
    the PRIVILEGED critic a 5-frame RMA-CNN history encoder (SequenceCritic), replacing the 1-frame
    Markov critic. Actor / env / rewards / deploy path BYTE-IDENTICAL to v83StudentOnly.

    Why. v83StudentOnly is asymmetric AC — a history-encoder actor (deploy-noisy proprio) vs a
    1-frame privileged critic. Under per-episode DR (body_mass bias, foot_friction, payload) the
    true value depends on hidden physical latents the 1-frame critic never observes: a POMDP from
    the critic's input. A temporal critic can implicitly system-ID those latents from the state
    response → lower-variance value targets.

    New premise vs the v57 'critic-history' DEAD LEVER. v57 regressed because the critic-history
    path only FLAT-CONCATENATED the (L_c, D_c) window into one MLP — no temporal inductive bias,
    just a wider noisier input. This class supplies the missing ENCODER (RMA-CNN over the window,
    the actor's mechanism). v57's exact impl is unrecoverable (no surviving class) and predates the
    audited-correct frame-ring window reconstruction, so its rejection does not bind this design.
    See plan/glimmering-imagining-minsky.md.

    Mechanism. configure() sets observations["critic"].history_length = 5 → the critic obs group
    becomes (5, D_c); the frame-ring buffer reconstructs the window (episode-boundary masking shared
    with the actor, audited). flash_sac_configure sets use_critic_sequence_encoder=True → the runner
    builds SequenceCritic (encoder reused from the actor: rma_cnn / embed 32 / latent 128) + switches
    the critic normalizer to per-frame. Critic-only change; the deploy export (the actor) is untouched.

    Inheritance: …→v83L2TActuatedCam→v83StudentOnlyActuatedCam→v83StudentOnlyCriticSeqCam.

    First run (2026-07-09, ser10) REGRESSED but NEVER BUILT THE ENCODER — it silently ran a plain
    Critic on the flat (755,) window (= the v57 flat-concat mechanism), because configure() set
    history_length but not flatten_history_dim=False, so the runner's len(critic_obs_shape)==2 gate
    never fired. Fixed here (flatten_history_dim=False below) + a second latent-only Q-input bug in
    SequenceCritic (now concats the newest frame, mirrors SequenceActorBackbone). Post-fix the log
    shows critic_obs_shape=(5,151) + SequenceCritic(RMACNNEncoder) + Q in_features 310.

    RETEST (grl3, single-seed 15k, fixed encoder) = NEUTRAL, not promoted. Healthy policy (no
    collapse, walks normally). vs baseline: mass ≈ (×1.3 0.008 vs 0.023), wz slightly better
    (@0.20 0.86 vs 0.76), vx slightly worse (@0.20 0.74 vs 0.85) — all within single-seed noise. The
    1-frame privileged critic already sees GT lin/ang vel + foot contacts, so history buys little
    value info. Mechanism validated but lever neutral; kept gated-off. Detail: memory/l2t_dead_levers.md.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        env.observations["critic"].history_length = 5          # (L_c, D_c) privileged critic window
        env.observations["critic"].flatten_history_dim = False  # keep 2D (L_c, D_c) so the runner's
        #   len(critic_obs_shape)==2 gate fires and builds SequenceCritic. Default flatten=True
        #   collapses to flat (L_c·D_c) → plain Critic on the flat window (the v57 flat-concat
        #   mechanism), silently skipping the encoder. Mirrors the actor (L120-121).

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.use_critic_sequence_encoder = True       # RMA-CNN temporal critic over the 5-frame window


class HumanoidRmaVelEstArmFlashSacv83StudentOnlyCriticFrictionCam(
        HumanoidRmaVelEstArmFlashSacv83StudentOnlyActuatedCam):
    """OBSERVE-THE-LATENT critic probe on the promoted v83StudentOnly baseline. Single lever: add the
    GROUND-TRUTH per-foot friction coefficient (2-D) to the privileged critic obs group. Actor / env /
    rewards / deploy path BYTE-IDENTICAL to v83StudentOnly (critic-only obs add; +2 critic input dims).

    Why (new premise vs the CriticSeq DEAD/neutral lever). foot_friction is randomized at STARTUP
    (Uniform(0.3,1.5), per-geom independent, fixed the whole run) and is the highest-leverage hidden
    dynamics latent — it gates every push-off. The 1-frame privileged critic never sees it and cannot
    infer it (no direct signal until a slip); even the 5-frame SequenceCritic (which tried to INFER the
    latent from the state response) added 0 value-accuracy (value_calibration_probe: r 0.402 vs 0.416).
    This class instead lets the critic OBSERVE the exact latent — the textbook asymmetric-critic trick.
    Fundamentally different premise from history (observe vs infer), so NOT the CriticSeq dead lever.

    Diagnostic value. Exact GT friction is the UPPER BOUND of "observe the latent": if adding it does
    NOT lift value-calibration R², the ~84% unexplained realized-return variance is aleatoric (future
    command / push draws) and NO critic obs can help — a definitive close-out. If it DOES lift, latent-
    observability is real and warrants a full 2-seed screen. Expect possibly-neutral DEPLOY impact: the
    value probe shows the current critic is already near-sufficient and the actor (deploy-blind) is the
    bottleneck; a better critic need not move the actor. Run value_calibration_probe first, screen second.

    Design. Per-foot MEAN over each foot's 5 collision capsules (not the raw 10-vector): the capsules
    tile one contact patch so the mean is the effective friction, and 2-D keeps the paper story clean.
    GT read (no noise) from geom_friction; deploy actor is on the `student` group and never sees it.

    Inheritance: …→v83L2TActuatedCam→v83StudentOnlyActuatedCam→v83StudentOnlyCriticFrictionCam.
    """

    def configure(self, env, agent):
        super().configure(env, agent)
        from tasks.humanoid_velocity.observation import foot_friction_gt
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        env.observations["critic"].terms["foot_friction_gt"] = ObservationTermCfg(
            func=foot_friction_gt,
            params={
                "left_cfg":  SceneEntityCfg("robot", geom_names=(r"^foot_L_collision\d*$",)),
                "right_cfg": SceneEntityCfg("robot", geom_names=(r"^foot_R_collision\d*$",)),
            },
        )   # privileged critic obs only; actor/deploy path byte-identical


class HumanoidRmaVelEstArmFlashSacv145MixedArmsCam(HumanoidRmaVelEstArmFlashSacv123):
    """v144's mixed home/swing arm curriculum, now WITH the v123 camera + gaze stack.

    v144 (= v30 + home_ratio curriculum) fixed the arm-swing backward-dead bug (backward tracks 0.88
    under arms-swinging vs v143's -0.22 command-blind) but has NO camera. v123 (= v30 + integral-error
    twist command + actuated twin-D435 head camera + random-walk gaze + camera_joint_tracking) is the
    deployable camera baseline. This child stacks the EXACT v144 lever onto v123: per-episode
    Bernoulli(home_ratio) frozen/swing arm-population split with a curriculum home_ratio 1.0 -> 0.5 over
    72000 env-steps, then hold 0.5. See v144 for the full population-split-not-time mechanism.

    home_ratio freezes only the ARM target joints (arm_ref); the camera (camera_ref / GazeRefCommandTerm)
    is a SEPARATE command and keeps gazing in frozen envs — a frozen-arm env still slews its head. So the
    single lever vs v123 is the arm home_ratio curriculum; camera behaviour is unchanged. Compare v145 vs
    v123 (the camera baseline), NOT vs v144, to isolate the curriculum's effect with the camera present.
    Single-seed screening only.

    Inheritance: ...->v29->v30->v123->v145 (camera + integral + arm home_ratio curriculum).
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
    """v145 with leg action_rate_l2 penalty quartered (-0.5 -> -0.125) -- aggressive v149 extension.

    Motivation (stepping/yaw-freedom campaign, +2 kg robot): v149 halves action_rate_l2 (-0.5 ->
    -0.25) following the 2-seed-confirmed L2T v115 win. This class takes the SAME axis one point
    further (-0.125, quarter) to bracket how far the relax helps before smoothness/hardware-safety
    (raw_action_rate_l2 must stay < 1.5) degrades. Screened in parallel with v149 so the axis is
    mapped in one wave, not serially. Single lever vs v145: scale the action_rate_l2 weight and both
    curriculum stages by 0.25.

    Risk: if raw_action_rate_l2 exceeds ~1.5 at convergence the policy is not hardware-safe -- read
    that metric before promotion. v149 (-0.25) is the conservative point of the same bracket.

    15k RESULTS (2-seed mean vs v145 base):
                            exy    eyaw   fell   air   asym
      v145 base             0.590  0.496  0.025  0.096 0.089
      v153 2-seed mean      0.560  0.481  0.013  0.081 0.075
      delta                 -0.030 -0.015 -0.013 -0.015 -0.014
    Clean-deadzone sweep (wz ratio, seed 0):
                            cmd 0.10  cmd 0.20  cmd 0.30
      v145                  0.37     0.59      0.73
      v153                  0.75     0.96      1.00
    Verdict: BEST TRACKING but air_time artifact. Tracking wins come from high-frequency
    low-amplitude steps (frequent small corrections > committed strides) -- air -15%, but
    still stable (lowest falls, lowest asym). v149 (-0.25) is the lighter-air alternative.
    NOT promoted without gait_naturalness probe (step height + stride length unverified).

    Inheritance: ...->v123->v145->v153. Single-seed screening only.
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
    """v153 with foot_clearance reward weight raised (-1.0 -> -2.0, 2x). **DEPLOY KEEPER 2026-07-18.**

    Motivation (gait-naturalness wave off v153 winner, +2 kg robot): foot_clearance's tanh-form
    returns |foot_z - target_height| * vel_factor (continuous deviation penalty, target 0.1).
    At v153's actual weight -1.0 the penalty share is small (~0.6% of |total|); doubling to -2.0
    makes it ~1.2% and forces the policy to commit to 0.10 m swings or pay the deviation cost
    every step. Orthogonal to v158a (which raises the target itself, not the weight).

    Orthogonality to v158a: v158a asks for taller swings (target up); v159b asks for STRONGER
    penalty when swings deviate from current target. If the air -15% artifact is from "target
    too low," v158a is the fix; if it's from "penalty too weak," v159b is. Different
    mechanisms, both address v153 air loss.

    PROMOTED 2026-07-18 as the new +2kg-robot deploy keeper (2-seed + deadzone probe confirmed).
    Full-reward comparison vs v153 base: exy -1%, eyaw -2%, air +5%, fell -42%, slip -7%,
    total reward -1% (within seed noise). Single weight lever, no new reward terms, no
    curriculum change. arm_joint_tracking weight 2.0 stays (invariant). v153 demoted to
    predecessor; this class is the new reference for any future +2kg-robot experiment.

    Risk: 2x weight may push policy into shallow shuffles that minimize continuous penalty
    rather than commit to swings -- the opposite of intent. Tested direction on legs (no arms)
    was neutral-to-bad on v152-like terms. action_rate + foot_orientation + air_time untouched.
    Single lever vs v153: ONLY foot_clearance weight. Target unchanged.

    Inheritance: ...->v123->v145->v153->v159b. Single-seed screening only. Result pending.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        # v153 foot_clearance weight: -1.0 -> -2.0 (2x deviation pressure at target 0.10m)
        env.rewards["foot_clearance"].weight = -2.0
class HumanoidRmaVelEstArmFlashSacv2ybMixedArmsCam(HumanoidRmaVelEstArmFlashSacv159bMixedArmsCam):
    """v159b + track_angular_velocity swapped to track_angular_velocity_relative.

    Motivation (yaw dead-zone fix on v2_best, +2kg robot): v159b still uses the
    fixed-std track_angular_velocity (std=sqrt(0.25)). At low yaw cmd (<=0.2 rad/s)
    this saturates reward ~ 1.0 regardless of actual tracking -- the yaw dead zone.
    v30 already fixed the SAME bug for XY by swapping to track_linear_velocity_relative
    (std_rel=0.5, std_min=0.3). This child applies the analogous fix to yaw:
    std_rel=0.5, std_min=0.1. At cmd=0.1, std_eff=max(0.05, 0.1)=0.1 -> score=0.368
    (real gradient). At cmd=0.5, std_eff=max(0.25, 0.1)=0.25 -> score ~ 0.85 (saturated).

    Orthogonality to v154: v154 tightened track_angular_velocity std (sqrt(0.5)->sqrt(0.3))
    on v145 base and was REJECTED (fell +100%). The relative form does NOT tighten the
    kernel at high cmd, it only sharpens it at low cmd. New premise: the dead zone is
    the bug, not kernel width per se.

    NEW PREMISE: the yaw dead zone is structural (mirroring what v30 fixed for XY).
    Fixing it removes the principal disincentive to tracking low yaw commands.

    Risk: low-cmd gradient pressure may initially destabilize the +2kg robot (v154 was
    killed for aggressive kernel shaping). The relative form is GENTLE at high cmd
    (std_eff>=std_min only) so walking yaw tracking is not over-penalized. Watch the
    same dead-lever red flags (fell@1k, early illegal_contact).

    Single lever vs v159b: ONLY track_angular_velocity.func + params (std_rel/std_min).
    Weight (2.5) carried over. body_ang_vel, action_rate_l2, foot_clearance, all others
    untouched.

    15k RESULTS (single-seed vs v159b baseline 13-56-17, the canonical v2_best reference):
                            v159b     v2yb     delta
      fell_over            0.0150    0.0375   +150%   RED FLAG (v159b already RED FLAG, v2yb ×2.5 worse)
      error_vel_xy         0.5486    0.5070   -7.6%   BETTER
      error_vel_yaw        0.4760    0.4230   -11.1%  BETTER ← dead-zone fix WORKS
      ep_reward            124.37    118.33   -4.9%
      metric_peak_height   0.0219    0.0222   +1%
      air_time_mean        0.0852    0.0922   +8%
      Red flags @5k/10k/15k: fell_over 0.06/0.053/0.0375 (cliff + persistent)
    Verdict: REJECT per SOP §5.4 (fell_over RED FLAG is disqualifying regardless of
    tracking). The dead-zone fix is STRUCTURALLY CORRECT — yaw error -10.4%, xy error
    -8.9% (both best-ever on +2kg robot). BUT the relative-form gradient pressure at
    low cmd destabilizes gait (15× more falls). Same trade-off pattern as v154 (kernel
    tighten = destabilize). Mechanism: policy commits to small yaw commands even when
    gait is borderline.
    Next attempts (single-var only, off v159b):
      - v2yb-softer: std_rel=0.5 -> 0.3 (gentler low-cmd gradient)
      - v2yb-wider: std_min=0.1 -> 0.3 (wider basin floor)
      - v2yb+v2ha combine: only if both single-var screens pass
    Do NOT re-test std_rel=0.5, std_min=0.1 (this entry).
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_angular_velocity_relative
        env.rewards["track_angular_velocity"].func = track_angular_velocity_relative
        env.rewards["track_angular_velocity"].params.pop("std", None)
        env.rewards["track_angular_velocity"].params["std_rel"] = 0.5
        env.rewards["track_angular_velocity"].params["std_min"] = 0.1


class HumanoidRmaVelEstArmFlashSacv2ybiMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybMixedArmsCam):
    """v2yb + track_angular_velocity_relative std_rel 0.5 -> 1.0 (high-cmd basin fix). **STATUS — REJECTED 2026-07-23, single-seed.**

    Motivation (yaw-tracking + hesitation on v2yb, +2kg robot): v2yb's relative form gave a
    hidden structural bug at HIGH cmd. With std_rel=0.5 and std_min=0.1, at cmd=0.5 std_eff
    = max(0.25, 0.1) = 0.25 (4x narrower than v159b's fixed std=0.5). At err=0.3 rad/s while
    cmd=0.5: v2yb reward = exp(-0.09/0.0625) = 0.24 vs v159b = exp(-0.09/0.25) = 0.70. v2yb
    punishes imperfect rotation 3x harder than v159b -> policy learns to NOT rotate (live
    viewer feedback: 'hesitant to rotate on yaw command'). The aggregate -11.1% yaw error
    improvement is the dead-zone fix at LOW cmd (std_min=0.1 dominates), but the high-cmd
    loss is hidden in the average.

    FIX: std_rel=0.5 -> 1.0. At cmd=0.5: std_eff = max(0.5, 0.1) = 0.5 (matches v159b).
    At cmd=0.1: std_eff = max(0.1, 0.1) = 0.1 (dead-zone fix preserved). The two regions
    are now independent: low cmd = std_min controls (dead-zone fix holds), high cmd =
    std_rel*cmd controls (matches v159b -> no hesitation).

    Math (at err=0.3, cmd=0.5):
      v2yb std_rel=0.5: reward = exp(-0.09 / 0.25^2) = exp(-1.44) = 0.24  <- hesitation
      v2ybi std_rel=1.0: reward = exp(-0.09 / 0.50^2) = exp(-0.36) = 0.70  <- tracks
    Math (at err=0.1, cmd=0.1): both 0.37 (dead-zone fix preserved by std_min=0.1 floor).

    NEW PREMISE: the original std_rel=0.5 was a math error (basin at high cmd 4x narrower
    than v159b). The fix is to make the high-cmd basin equivalent to v159b's fixed std.

    Single lever vs v2yb: ONLY std_rel in track_angular_velocity.params. std_min=0.1
    unchanged. All other terms (body_ang_vel, action_rate_l2, foot_clearance, upright)
    untouched.

    15k RESULTS (single-seed vs v159b baseline 13-56-17, the canonical v2_best reference):
                            v159b     v2ybi    delta
      fell_over            0.0150    0.0750   +400%   RED FLAG (cliff 0.035@14900 -> 0.05@14950 -> 0.075@15000, accelerating)
      error_vel_xy         0.5486    0.5016   -8.6%   BETTER  ← dead-zone fix works
      error_vel_yaw        0.4760    0.4430   -7.0%   BETTER  ← dead-zone fix works
      ep_reward            124.37    123.92   -0.4%   noise
      ep_length            --        914      --      (long episodes, not short-terminating)
      metric_peak_height   0.0219    0.0222   flat
      air_time_mean        0.0852    0.0906   +6%
    Verdict: REJECT per SOP §5.4 (fell_over RED FLAG is disqualifying regardless of tracking).
    The HIGH-cmd hesitation is GONE (matched v159b's std) BUT the LOW-cmd dead-zone pressure
    from std_min=0.1 still destabilizes gait (fell 5x worse, still rising at end of training).
    Confirms: v2yb's destabilization is structurally tied to std_min=0.1 (the dead-zone fix
    itself causes gait disturbance), NOT to std_rel. std_rel=1.0 fixed only half the math.

    Next attempts (single-var only, off v2yb/v2ybi):
      - std_min=0.1 -> 0.2 / 0.3 (gentler dead-zone gradient that preserves some pressure)
      - std_rel < 1.0 (e.g. 0.7) WITH std_min=0.2 (looser both ends)
    Do NOT re-test std_rel=1.0, std_min=0.1 (this entry).

    Inheritance: ...->v159b->v2yb->v2ybi. Run: 2026-07-23_13-05-32_flash_sac (53 min, 15000 iter, ✓ HEALTHY).
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_angular_velocity_relative
        env.rewards["track_angular_velocity"].func = track_angular_velocity_relative
        env.rewards["track_angular_velocity"].params.pop("std", None)
        env.rewards["track_angular_velocity"].params["std_rel"] = 1.0
        env.rewards["track_angular_velocity"].params["std_min"] = 0.1

class HumanoidRmaVelEstArmFlashSacv2ybhciMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybiMixedArmsCam):
    """v2ybi + upright weight 1.0 -> 2.0. **STATUS — EVAL-PROMOTED to v2_best (2026-07-23, single-seed).**

    Motivation (yaw-tracking on +2kg robot, v2ybi base): v2ybi's relative-form dead-zone fix
    gave yaw wins (-7 to -11%) but training-log fell_RED_FLAG (0.075, accelerating). The
    mechanism is trunk-tilt wobble during yaw rotation: relative form rewards free rotation,
    trunk roll/pitch destabilizes gait. Adding upright=2.0 attacks trunk-tilt directly while
    keeping both v2ybi fixes (std_rel=1.0 high-cmd math, std_min=0.1 low-cmd gradient).

    Training-log 15k vs v159b 13-56-17: error_vel_yaw **-10%** (0.476→0.430),
    error_vel_xy **-22%** (0.549→0.430), term_fell_over +167% (0.015→0.040, RED FLAG).
    The training-log fell was a FALSE RED FLAG (see SOP rule below).

    **EVAL-PROMOTED (2026-07-23, single-seed).** Deterministic eval (50 ep × 5 fixed cmds, no
    noise/DR; script: `DISPLAY="" mj_envs/eval_policy.py --task X --checkpoint
    model_0015000.pt --episodes 50 --max-steps 500`) vs v159b baseline:
                                v159b     v2ybhc_i    delta
      mean fell_over            0.264     0.256       -3%    ✓
      mean track_lin_err        0.097     0.094       -3%    ✓
      yaw_err  dead-zone (0,0,0.1)  0.150  0.126       **-16%** ✓  best dead-zone
      yaw_err  mid        (0,0,0.3)  0.250  0.236       -5%   ✓
      yaw_err  high       (0,0,0.5)  0.322  0.285       -11%  ✓
      yaw_err  +vyaw      (0.5,0,0)  0.179  0.160       -11%  ✓
      yaw_err  +x+yaw     (1.0,0,0)  0.215  0.166       **-23%** ✓
    WINS ALL 5 yaw cmds. v2ybhc_i beats v2ybk on every yaw axis AND on the dead-zone.
    Live view confirmed: http://grl1:8080 (PID 53091, cmd_ang_z=+0.5 rad/s).

    **CRITICAL SOP RULE** (training-log fell != eval fell): training-log `term_fell_over`
    averaged per-step during noisy stochastic training captures INTER-STEP termination rate
    under exploration noise (overstates risk 5-10x). v159b baseline: training=0.015, eval=0.264
    (17x inflation). Eval (deterministic no-noise) is the verdict-driving metric. CONFIRM any
    REJECT from training-log via `mj_envs/eval_policy.py` BEFORE killing.

    Single lever vs v2ybi: ONLY upright weight. std_rel=1.0, std_min=0.1 unchanged. All other
    terms untouched.

    Promotion rationale (per Wave-15 / MEMORY): kept v2ybi's relative-form dead-zone fix
    (both fixes), ADDED upright tighten (anti-tilt) on top. Eval shows relative form +
    upright = better than v159b on the dead-zone AND on every yaw cmd. Falls and XY
    errors also slightly better. **Deploy keeper until v2ybkhc combine OR v2ybk+softer_std_min
    proves superior.** 2-seed confirm + cross-robot portability PENDING.

    Inheritance: ...->v159b->v2yb->v2ybi->v2ybhc_i. Run: `runs/HumanoidRmaVelEstArmFlashSacv2ybhciMixedArmsCam/2026-07-23_15-14-07_flash_sac/`.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        # v2ybi upright weight: 1.0 -> 2.0 (anti-tilt directly on destabilizer)
        env.rewards["upright"].weight = 2.0
class HumanoidRmaVelEstArmFlashSacv2ybdkcMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybhciMixedArmsCam):
    """v2ybhc_i + track_angular_velocity_relative std_rel 1.0 -> 0.7 (mid cmd basin tighten). Wave-17 #2.

    Motivation: Wave-15 std_rel sweep:
      std_rel=0.5 (v2yb) → REJECTED on v2yb base (high-cmd basin 4x too narrow → hesitation)
      std_rel=1.0 (v2ybi/v2ybhc_i) → WIN (high-cmd math matches v159b fixed std=0.5)
      std_rel=??? → 0.7 untested.

    Mid-std_rel=0.7 tests whether strict-matching of v159b's fixed-std is necessary, or
    whether a slight tightening of the mid-cmd basin (still looser than v2yb's 0.5) helps
    yaw push on the relative-form base. At cmd=0.5 std_eff at std_rel=0.7 = max(0.35, 0.1)
    = 0.35 — narrower than v2ybhc_i's 0.5 but still wider than v2yb's 0.25; the policy
    still sees yaw-tracking pressure at mid cmd.

    NEW PREMISE: the relative-form is a 2-knob system (std_rel for high-cmd basin,
    std_min for low-cmd floor). v2ybhc_i has std_rel=1.0 = matched to v159b; 0.7 mid-points
    between dead (0.5) and matched (1.0). Mid value may add tracking pressure at mid-cmd
    without breaking high-cmd math.

    Single lever vs v2ybhc_i: ONLY track_angular_velocity_relative.params.std_rel. std_min=0.1,
    upright=2.0 untouched.

    15k gate vs v2ybhc_i (eval-verified):
      PASS = error_vel_yaw < 0.42 (preserve winner) AND fell_over < 0.24
      AND any yaw cmd wins by ≥3%.

    Inheritance: ...->v159b->v2yb->v2ybi->v2ybhc_i->v2ybdkc.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.reward import track_angular_velocity_relative
        env.rewards["track_angular_velocity"].func = track_angular_velocity_relative
        env.rewards["track_angular_velocity"].params.pop("std", None)
        env.rewards["track_angular_velocity"].params["std_rel"] = 0.7
        env.rewards["track_angular_velocity"].params["std_min"] = 0.1
class HumanoidRmaVelEstArmFlashSacv2ybskMixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybdkcMixedArmsCam):
    """v2ybdkc + arm_ref stand_min_steps 0 -> 80 (TRAINING-TIME slow-arm lever). Wave-19.

    Wave-19 reward-lever route (v2ybstk weight=-2.0, v2ybstk2 weight=-0.5) DEAD -- gating a
    base-velocity penalty on arm-active fights the natural balance response, regressing
    ang_err on tracking-heavy branches. Switched to a NON-REWARD lever: the arm_ref
    CommandTerm already has loco_min_steps=80 (slow swing when |twist_xy|>0.3). The new
    stand_min_steps knob (added to CommandGatedArmGraphRefCommandTermCfg) parallel
    slow swing at |twist_xy|<=0.05 (zero base cmd). New premise: smaller arm-reaction
    impulse during training rollouts -> policy learns cleaner anticipation -> lower
    base sway at deploy (where the knob reverts to default 0 and arm swings full speed).

    Mechanism probe evidence: same-policy arm-speed sweep showed sway is NON-MONOTONIC in
    speed (spe50 peak 0.096 < spe100 0.117 > spe200 0.073); slower swing reduces impulse.
    stand_min_steps=80 is ~1.6x slower than keeper steps_per_edge=50 (analogous to loco).

    TRAINING-ONLY (NOT a deploy lever, per same-policy probe 2026-07-25): at deploy the
    knob reverts to default 0 and the arm swings at full speed. v2ybsk ckpt under runtime
    sms=0 probe shows drift_peak 0.0663 (still 22% better than v2ybdkc trained/evaluated
    at sms=0 = 0.0853) -- so deploy behavior is full-speed arm with sub-keeper sway. Do
    NOT set stand_min_steps > 0 on a deployed policy unless the user explicitly wants a
    slower arm at zero base cmd; the gain is from training-time, not from runtime speed.

    Single lever vs v2ybdkc: ONLY arm_ref.stand_min_steps=80 during training. loco_min_steps=80,
    upright=2.0, std_rel=0.7, std_min=0.1 all untouched. No reward terms added.

    15k gate (apples-to-apples vs keeper v2ybdkc, single-seed screen):
      PRIMARY: sway_probe active drift_peak / lin_rms DOWN vs keeper baseline.
      GUARD: deterministic eval fell <= 0.256, lin_err <= 0.087, all 5 ang_err comparable.
      RESULT (2-seed confirm 2026-07-24): sway active peak -28% mean, fell -16% mean,
      lin_err ~tied, all other guards tied-or-better; vx0.5 / vx0.5+wz0.5 ang_err +12-14%
      (mild regress accepted). EVAL-PROMOTED. v2_best alias retargeted 2026-07-24,
      RE-PROMOTED 2026-07-25 after same-policy probe ruled out cheating.

    Inheritance: ...->v2ybdkc->v2ybsk.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["arm_ref"].stand_min_steps = 80
class HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam(HumanoidRmaVelEstArmFlashSacv2ybskMixedArmsCam):
    """Wave-29 #4: v2ybsk + track_angular_velocity_relative std_min 0.1 -> 0.05 (TIGHTER).

    Motivation: v2ybsk_s1 (std_min 0.1->0.2, WIDER) REJECTED sway +35% — loosening
    dead-zone gradient softens low-cmd tracking. OPPOSITE direction (TIGHTER) UNTESTED
    on v2ybsk. std_eff at cmd=0.1: max(0.07, 0.05) = 0.05 vs keeper 0.10. Gradient
    ~2.4x sharper at low cmd. v2ybsk's sms=80 + upright=2.0 may absorb the
    destabilization that killed std_rel=0.5/std_min=0.1 on v159b.

    NEW PREMISE: std_min direction is unblocked on v2ybsk (s1 tested WIDER only).
    v2ybsk's sms=80 slow-arm + upright=2.0 anti-tilt + std_rel=0.7 wider basin form
    a more stable base than the v159b base where std_rel=0.5/std_min=0.1 cliffed.

    Single lever vs v2ybsk: ONLY track_angular_velocity.params.std_min (0.1 -> 0.05).
    std_rel=0.7, weight=1.5, upright=2.0, sms=80 all untouched.

    15k gate (same multi-axis gate as v2ybsk_yaw_w1):
      PASS = ang_err_wz0.5 < 0.3012 AND sway drift_peak ≤ 0.0663
      AND fell ≤ 0.218 AND lin_err ≤ 0.099.

    Risk: HIGH. Same cliff direction as v2yb std_min=0.1 dead-lever (fell +150% on
    v159b base). sms=80 + upright=2.0 may or may not absorb it.

    VERDICT (Wave-29 single-seed 2026-07-28): 5-cmd eval REJECT (+4% wz0.5 ang_err,
    +10% fell, +5% lin_err). BUT robust wz-sweep revealed wz0.3 ang_err 0.220 vs
    baseline 0.276 (-20% BETTER on low-wz tracking) — tighter std_min does improve
    yaw tracking at sub-0.5 cmds, but at wz0.5/wz0.7 it's tighter than the policy
    needs. Foundation for v2ybsk_yaw_s5 (std_min 0.07 intermediate).

    Inheritance: ...->v2ybsk->v2ybsk_yaw_s4.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].params["std_min"] = 0.05
class HumanoidRmaVelEstArmFlashSacv2GridAdaptiveCommands(HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam):
    """Opt-in v2_best child with local joint `(vx, vy, wz)` command curriculum.
    keep
    Keeps v2_best rewards, arm schedule, and domain randomization unchanged. The
    only training change is replacing independent command sampling with the grid
    sampler. Direct body-frame commands are required because heading, world, and forward
    modes modify a sampled command after its grid cell has been recorded. Exact standing
    commands retain the inherited 20% zero-command branch and do not update grid weights. Linear
    command samples stay inside a 1.0 m/s planar-speed disk, set explicitly below. Deadline graph
    shells reach the final safe grid at ``final_stage_fraction`` (0.8) of the runner horizon, then
    dwell there for the remaining 20%.

    Envelope note: the disk was raised 0.8 -> 1.0 m/s (ranges +/-0.8 -> +/-1.0) so this task is no
    longer strictly narrower than its v2_best parent, whose ``command_vel`` curriculum holds a
    +/-1.0 box. Comparisons against v2_best at planar speeds near 0.8 were previously rim-vs-interior
    and invalid; see ``memory/grid_curriculum_envelope_confound.md``. Yaw remains wider than
    v2_best's +/-0.7, so ``ang_err`` stays confounded.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from tasks.humanoid_velocity.command import GridAdaptiveVelocityCommandCfg

        env.observations["actor"].terms.pop("command_integral", None)
        env.observations["critic"].terms.pop("command_integral", None)
        env.rewards.pop("track_integral_error", None)
        # Grid owns command coverage. The inherited range curriculum mutates
        # twist.cfg after GridAdaptiveVelocityCommand fixes its cell boundaries.
        env.curriculum.pop("command_vel", None)
        twist = env.commands["twist"]
        env.commands["twist"] = GridAdaptiveVelocityCommandCfg(
            entity_name=twist.entity_name,
            resampling_time_range=twist.resampling_time_range,
            rel_standing_envs=twist.rel_standing_envs,
            debug_vis=twist.debug_vis,
            # Set explicitly, never inherited from the class default: this is the planar
            # speed the policy is actually trained to, so it belongs at the callsite next
            # to the ranges it clips.
            max_planar_speed=1.0,
            ranges=GridAdaptiveVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
            ),
        )


class HumanoidRmaVelEstArmFlashSacv2GridGaitInit(HumanoidRmaVelEstArmFlashSacv2GridAdaptiveCommands):
    """Grid child, single change: reset a fraction of envs into an ON-manifold WALKING state
    (deploy-safe reference-state-initialization) instead of always from standing.

    Motivation (memory/grid_low_command_deadzone.md, plan recursive-roaming-lemur, Exp2). The dead
    zone is a coverage hole: every episode resets FROM standing, so the critic never samples a low-cmd
    WALKING state and never grounds a high value there. This adds a reset event that puts fraction f of
    resetting envs into a broad gait state (parametric stride, speed s~U(0,s_max)); the independent grid
    command sampler then hands some of them a LOW command, giving exactly the low-cmd-walking exposure
    the dead zone lacks. The reward, command sampler, curriculum, and all other events are inherited
    from the grid parent unchanged (one variable = the init distribution).

    On-manifold by construction: the keyframe's per-joint pose/velocity are the MEASURED cmd-0.4
    marginals of the grid policy (see GaitKeyframeResetFast). Verified on-manifold by
    probe/gaitinit_feasibility.py (SYNTH re-entry term-rate/velocity matched the real captured-state
    upper bound, far from the standing baseline) — a keyframe off the gait surface would stumble and
    train recovery-TO-standing (wrong sign), so this gate was mandatory before training.

    Deploy-safe: the keyframe is reset-mode only, never read by an obs/reward — NOT the rejected
    control-side phase clock. Standing majority (1-f) keeps the deploy start condition and zero-cmd calm.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from mjlab.managers.event_manager import EventTermCfg
        from tasks.humanoid_velocity.event import GaitKeyframeResetFast
        # Appended after reset_base/reset_robot_joints so it overrides the standing pose for its
        # masked fraction; unmasked envs keep the standing reset already written by those events.
        env.events["reset_gait_keyframe"] = EventTermCfg(
            func=GaitKeyframeResetFast,
            mode="reset",
            params={"fraction": 0.3, "s_max": 0.6},
        )


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitSingleCam(HumanoidRmaVelEstArmFlashSacv2GridGaitInit):
    """Single-camera sibling of the gait-init grid policy — the ``v2_best_single`` counterpart.

    ONE conceptual change vs ``HumanoidRmaVelEstArmFlashSacv2GridGaitInit``: camera count 2 -> 1,
    applied by the shared ``_apply_single_camera`` helper so it is byte-identical to the swap the
    previous ``v2_best_single`` lineage used. Everything else (grid command sampler, gait-keyframe
    reset event, full reward lineage) is inherited unchanged.

    Composition is safe because the two levers touch disjoint cfg keys: the camera swap overwrites
    ``scene.entities["robot"]`` / ``commands["camera_ref"]`` / ``actions["joint_pos_camera"]``, while
    the grid+gait-init ancestors touch ``commands["twist"]``, ``curriculum``, ``observations``,
    ``rewards`` and ``events``. The camera swap runs last (after ``super().configure()``) exactly as
    it does in the ybsk single-cam class.

    Why it must be trained, not derived: swapping only the SCENE camera count on a dual-trained
    policy leaves 2 of its 4 camera-tracking obs/action dims permanently unused — not a fair
    single-camera baseline. Same reasoning as the original v2_best_single.

    NOTE: 29-dim action space vs the dual-cam 31. Per MEMORY, checkpoint auto-resolution walks the
    inheritance chain onto a dual-cam ancestor and dies on ``size mismatch for action_scale``, so
    always pass ``--checkpoint`` explicitly for this class.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        _apply_single_camera(env)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZero(HumanoidRmaVelEstArmFlashSacv2GridGaitInit):
    """GridGaitInit + lower the curriculum's initial-planar-speed band (0.25, 0.5) -> (0.10, 0.25).

    Historical: (0.25, 0.5) was the cfg default when this class ran. The default has since widened
    to the full low band ``(0.05, 0.6)``, so the explicit assignment below is what preserves this
    class's measured configuration -- do not drop it as redundant.

    Single lever vs GridGaitInit: ONE cfg field change on the grid twist cfg --
    ``initial_planar_speed_range`` lowered from (0.25, 0.5) to (0.10, 0.25). Same reward, same
    gait-init reset, same DR, same obs. Forces the curriculum to start its outward BFS from a
    LOWER band so the policy is exposed to walking at low cmd EARLIER in training, before the
    standing basin forms.

    Why on top of GaitInit (not GridAdaptiveCommands directly): the gait-init reset already makes
    the policy survive an early-life low-cmd walking exposure; lowering the start band pairs with
    it. Without gait init, lowering the start band would just put standing envs into low-cmd
    cells and reopen the dead-zone regression path. Single variable vs parent = the curriculum
    start band.

    MEMORY cites this lever as the #1-ranked fix for the dead zone but flags that "it does not
    remove the optimum, only avoids it" -- it is an INIT-side lever, not a reward-side fix. The
    gait-init reset already provides on-manifold low-cmd coverage; this lever changes WHEN the
    policy learns the lower band, providing the curriculum-ordering half of v2_best's
    PiecewiseLinearVelocityRangeCurriculum (which starts near zero and expands outward).

    Invariants:
      - lower bound > 0: required by ``command.py`` assert (line 567); 0.10 is above the
        ``cmd_xy_threshold=0.1`` floor, so standing envs aren't triggered prematurely.
      - upper bound <= box_diagonal_speed (line 573): the disk ranges are +-1.0 here so
        diagonal ~= 1.41; (0.10, 0.25) clears it.
      - The new lower band (0.10, 0.25) intersects the cmd-0.20 dead-zone band so the policy
        trains at low cmd from iter 0, NOT just at the BFS front later.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        twist = env.commands["twist"]
        twist.initial_planar_speed_range = (0.10, 0.25)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCam(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZero
):
    """Single-camera sibling of ``StartNearZero`` -- the ``v2_best_single`` counterpart.

    ONE conceptual change vs ``...GridGaitInitStartNearZero``: camera count 2 -> 1 via the shared
    ``_apply_single_camera`` helper, so the swap is byte-identical to the one every other
    ``v2_best_single`` lineage used. Everything else (grid sampler, low seed ring, gait-keyframe
    reset, full reward lineage) is inherited unchanged.

    Must be trained, not derived: dropping the scene camera on a dual-trained policy leaves 2 of
    its 4 camera-tracking obs/action dims permanently unused, which is not a fair single-camera
    baseline. 29-dim action space vs the dual-cam 31, so per MEMORY always pass ``--checkpoint``
    explicitly -- checkpoint auto-resolution walks onto a dual-cam ancestor and dies on
    ``size mismatch for action_scale``.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        _apply_single_camera(env)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlace(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCam
):
    """Single-camera sibling of StartNearZero+lane. The v2_best_single retarget (2026-08-03).

    Single change vs ``...GridGaitInitStartNearZeroSingleCam``: ``turn_lane_mass_range=(0.02,0.25)``
    applied AFTER ``_apply_single_camera`` (camera swap must precede lane opt-in; the lane reads
    cfg fields, the camera swap rewrites entity cfg, so order matters).

    Must be trained, not derived, for the same reason as ``...GridGaitInitSingleCam``: a
    single-cam body has 29-dim action space vs the dual-cam 31, so a checkpoint trained under the
    dual-cam chain loads with a size mismatch. Use ``--checkpoint`` explicitly (see
    ``...GridGaitInitStartNearZeroSingleCam``).
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["twist"].turn_lane_mass_range = (0.02, 0.25)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlace(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZero
):
    """StartNearZero + the turn-in-place curriculum lane. The v2_best retarget (2026-08-03).

    Single change vs ``...GridGaitInitStartNearZero``: ``env.commands["twist"].turn_lane_mass_range``
    set to ``(0.02, 0.25)`` AFTER StartNearZero's curriculum band has been lowered. The lane opt-in
    must come last because the grid cell boundaries are fixed by StartNearZero's
    ``initial_planar_speed_range=(0.10, 0.25)``; mutating the cfg out of order would either leave
    the lane with the parent's old bands or be silently overwritten on resample.

    Why STACK, not replace: StartNearZero and the lane address disjoint failures measured 2026-08-03:
    StartNearZero closes the *low-speed walking* dead zone by starting the curriculum's BFS inside
    the dead band; the lane closes the *zero-linear turn* dead zone by giving turn-in-place its own
    curriculum lane with mass derived from its angular tracking deficit. The lane is below the
    cmds the dead band starts at (lane: vx==vy==0; StartNearZero starts the BFS in a band that
    already has nonzero speed). They cannot substitute; they cover disjoint failures.

    Refs ``...GridGaitInitTurnInPlace`` for the lane design and ``memory/yaw_deadzone_command_coverage.md``
    for the v1 vs v2 measured comparison.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["twist"].turn_lane_mass_range = (0.02, 0.25)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBank(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlace
):
    """v2_best target + single change: the gait-init pose comes from REAL harvested states
    (``GaitBankResetFast``) instead of the hardcoded cmd-0.4 marginals (``GaitKeyframeResetFast``
    / the ``_KF_*`` arrays). ``fraction`` and the reset-speed distribution are unchanged.

    Motivation (plan/GAIT_STATE_BANK_PLAN.md). The ``_KF_*`` constants are absolute joint angles in
    ONE MJCF's frame, captured once from a past policy at cmd 0.4. Edit a link length, a mass, or
    ``default_joint_pos`` and they silently go off-manifold: the joint NAMES still resolve and the
    soft-limit clamp at ``event.py:276`` absorbs the drift, so the failure surfaces as "training got
    slightly worse", never as an error. Off-manifold init trains recovery-TO-standing, the wrong
    sign. ``GaitBankResetFast`` holds no MJCF-derived constant -- it records this run's own
    successful states and replays them, so it re-derives itself whenever the robot changes.

    Two consequences that are NOT free and are the reason this needs its own measurement:

    1. A REAL restore writes strictly more physical state than the incumbent partial keyframe (root
       z and orientation, every joint, full root twist -- not just 12 leg joints plus forward speed).
       Better-posed in principle, but it is a different intervention, not a drop-in.
    2. The effective injection rate is NOT 0.3. Injection additionally requires the randomly drawn
       bank slot to be populated, so the rate ramps from ~0 (cold bank, behaves exactly like the
       parent minus the keyframe) toward 0.3 as the policy learns to walk. Self-bootstrapping by
       design -- at iteration 0 the policy cannot walk, so there is nothing worth injecting -- but it
       means an underperforming result must rule out "applied less often" before concluding "worse
       states". The feasibility probe logs the realized rate for exactly this reason.

    Premise is NOT established. ``...GridGaitInit`` REGRESSED the dead zone (``+-0.20 disp`` 1.607 ->
    1.146 vs ybsk) while ``...GridDecoupledTrack`` reached 1.910, so low-cmd-walking COVERAGE was not
    the binding constraint for the synthetic keyframe. This class tests a narrower claim: that
    CORRELATED real state holds the walking branch across a state-command transition where
    uncorrelated synthetic state does not. Gate it on the BANK lane of
    ``probe/gaitinit_feasibility.py`` before spending a training run; abandonment criteria are in
    plan section 10.5.

    ``configure()`` REASSIGNS the existing ``reset_gait_keyframe`` key rather than adding a new one,
    which preserves the event's last-in-dict position -- it must run after ``reset_base`` and
    ``reset_robot_joints`` to override the standing pose they wrote.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from mjlab.managers.event_manager import EventTermCfg
        from tasks.humanoid_velocity.event import GaitBankResetFast
        env.events["reset_gait_keyframe"] = EventTermCfg(
            func=GaitBankResetFast,
            mode="reset",
            params={"fraction": 0.3, "s_max": 0.6},
        )


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankPool(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBank
):
    """...Bank + single change: ``sample_filled=True``. The injected row is drawn from the rows the
    bank HOLDS, instead of uniformly over its slot address space.

    Rung 1 of a three-rung ladder (plan/GAIT_STATE_BANK_PLAN.md 13), each rung one change from the
    last: this class fixes the SAMPLER, ...BankFlat then drops the speed strata, ...BankPri then
    weighted the draw by measured performance. Run in parallel, single seed each, all against the same
    10.8 ``...Bank`` and ``...StartNearZeroTurnInPlace`` numbers. Rung 3 (``...BankPri`` and its
    ``PriNorm`` / ``PriNormOpen`` children) is a DEAD LEVER and its classes were deleted 2026-08-05;
    see MEMORY's dead-lever index. Rungs 1 and 2 stand.

    The defect is arithmetic, not a hypothesis. The incumbent draws ``slot ~ U{0, depth-1}`` and
    declines if that slot is unwritten, so P(a selected env is served) = that bin's FILL FRACTION.
    ``probe/bank_occupancy.py`` measured the slow bin at 81 of depth 1024 after 3000 steps at
    checkpoint 2500 (10.11), i.e. **92% of slow-band injections silently no-op** back to the standing
    reset -- which is the reset distribution the dead zone is made of. The 81 real slow-walking states
    were already in the bank; the sampler could not reach them.

    Supersedes ``...BankWarm`` (10.12, class deleted 2026-08-05) as the fix for the same hole.
    BankWarm filled it with the hardcoded ``_KF_*`` sinusoid, and a hand-written gait model is the
    thing this whole line of work exists to delete -- rejected on that ground regardless of its
    numbers, which were in any case partial (recovered ``vx 0.20``, left ``vx 0.15`` at 0.36 vs the
    control's 0.61). This class closes the hole with real harvested states and no constants. The
    ``warm_start`` parameter it drove still exists on ``GaitBankResetFast``, defaulted off and now
    unused by any class.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.events["reset_gait_keyframe"].params["sample_filled"] = True


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlat(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankPool
):
    """...BankPool + single change: the six speed strata collapse to ONE flat ring.

    ``n_bins`` and the bin edges are hand-picked structure -- 6 uniform bins over [0.1, 0.6] chosen
    because they looked reasonable, never measured. Stratifying buys diversity only if the strata are
    the right ones; if they are not, they buy a sparser ring per stratum and a slower fill, which
    10.11 measured as the binding problem. ``depth`` rises 1024 -> 6144 so total capacity is held
    constant and this is a change of STRUCTURE, not of capacity.

    Isolates "the strata help" from "the sampler was broken", which ...BankPool alone cannot: if
    BankPool wins and this matches it, the strata were doing nothing and the next class can drop them
    for free; if this loses to BankPool, the strata are load-bearing and rung 3 must keep them.
    Outcome: this class won and became the lineage base; the strata were not load-bearing.

    ``v_floor`` stays at its inherited 0.1. It is also a hand pick, but removing it admits standing
    states, which the class docstring flags as a sign-inverting failure -- one change per class.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        p = env.events["reset_gait_keyframe"].params
        p["n_bins"], p["depth"] = 1, 6144


class v2_best(HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlace):
    """v2 policy pointer: GridGaitInitStartNearZero + turn-in-place lane (RETARGETED 2026-08-03, user promote).

    Inheritance: ...->v2ybsk->v2ybsk_yaw_s4->v2GridAdaptiveCommands->v2GridGaitInit->
    v2GridGaitInitStartNearZero->v2GridGaitInitStartNearZeroTurnInPlace->v2_best.

    Two changes stacked vs the previous, broken-as-source alias: (1) the 2026-08-02 retarget to
    StartNearZero is now actually wired up -- the previous class base was still ybsk_yaw_s4 with no
    configure() override, so the StartNearZero promotion was documented but never executed; (2) the
    2026-08-03 turn-in-place lane, a curriculum lane whose mass is derived from its angular
    tracking deficit, replaces the v1 ``rel_turning_envs=0.15`` bolt-on. Both are deploy-safe.

    Measured at 15k iter, single seed, on the lane alone (`runs/.../2026-08-03_00-08-20_flash_sac/`,
    cf. ``memory/yaw_deadzone_command_coverage.md``):
        axis                                parent ActionRate0p1  v2 lane
        frozen at (0,0,wz=0.2)             0.41                   0.20
        yaw p50 at (0,0,wz=0.2)            0.274                  0.559
        vx_ach at vx=1.0                    0.834                  0.809
        joint_power WZ=0                   -0.88                  -0.80
        self_collisions WZ=0                -50.11                 -58.18

    EVIDENCE CAVEAT -- single seed by the [[feedback_single_seed_decisive]] override: the
    failure-mode effect at (0,0,wz=0.2) is large enough (~5 sigma vs rerun noise on this regime)
    to be decisive without a confirm seed. self_collisions ticks +16% magnitude vs parent; within
    v1's cost range, not worse, but worth a second seed if you want a confidence interval on that
    term before exposing it to hardware.

    Reproduction (current v2_best = GridGaitInitStartNearZero + lane):
      python mj_envs/run.py train --task v2_best --num_envs 4096 --seed 0 --max_iterations 15000

    v2_best ckpt (the lane run that was promoted; loadable under the alias via fallback):
      runs/HumanoidRmaVelEstArmFlashSacv2GridGaitInitTurnInPlace/2026-08-03_00-08-20_flash_sac/model_0015000.pt

    Revert = repoint this base class back to ``HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam``.

    History of v2_best retargets (most-recent-first):
      2026-08-03: v2_best = GridGaitInitStartNearZero + lane (user promote; single seed decisive)
      2026-08-02: v2_best = GridGaitInitStartNearZero (DOCUMENTED BUT NEVER APPLIED -- the source
                  still inherited ybsk_yaw_s4 with no configure() override, so the StartNearZero
                  promotion was a doc-only change that did not affect training under the alias)
      2026-07-31: v2_best = v2ybsk_yaw_s4 (user visual preference)
      2026-07-29 to 2026-07-31: v2_best = v2ybsk_yaw_s4_f3 (DEMOTED, see v2_best_old_f3)
      2026-07-25: v2_best = v2ybsk (Wave-19 2-seed confirm, RE-PROMOTED)
      2026-07-23: v2_best = v2ybhci (Wave-15 eval-promoted)
      ... (earlier v2_best history in MEMORY.md)

    Previous v2_best (v2ybsk_yaw_s4) reproduction + ckpt, kept for revert:
      python mj_envs/run.py train --task HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam --num_envs 4096 --seed 0 --max_iterations 15000
      runs/HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam/2026-07-29_07-31-01_flash_sac/model_0015000.pt
      md5: f37e89815aa5814526571f68207a97e6
    """
    fallback_checkpoint = "HumanoidRmaVelEstArmFlashSacv2GridGaitInitTurnInPlace"

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)


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
    """Single-camera sibling of v2_best (= v2ybsk_yaw_s4, the new v2_best after f3 demotion).
    ONE conceptual change vs v2_best: swap the dual head-camera module
    (cam_[yaw|pitch]_[left|right], 4 gaze DOFs, 2 gimbals) for the single centered
    module (cam_yaw/cam_pitch, 2 gaze DOFs, 1 gimbal) — everything else (the full
    v2_best = v2ybsk_yaw_s4 reward lineage: foot_slip 0.2, sms=80, upright=2.0,
    yaw_s4 dead-zone fix, etc.) is inherited UNCHANGED via ``super().configure()``.

    Why safe to swap post-hoc rather than re-derive the lineage from scratch: ``super().configure()``
    runs the FULL ancestor chain first, which (inside ``HumanoidRmaVelEstArmFlashSacv59L2TActuatedCam``,
    the chain's camera-setup ancestor) already built a dual-camera ``env.scene.entities["robot"]`` +
    ``camera_ref``/``joint_pos_camera`` commands/actions. This subclass then OVERWRITES those same three
    dict entries with single-camera equivalents (same keys, so dict assignment fully replaces rather than
    merges). Every downstream consumer resolves its dims generically off whatever is bound to those keys
    at env BUILD time (which happens after ``configure()`` finishes), not off a value cached during
    ``configure()``:
      - ``camera_joint_tracking`` reward (``arm_joint_tracking``, reward.py:1150) reads
        ``self._term.target_names`` off the "camera_ref" command at ITS OWN init.
      - ``target_camera_joint_pos`` obs (``target_arm_joint_pos``, observation.py:605) reads
        ``self._term.command`` off the same command, same way.
      - actor/student ``joint_pos``/``joint_vel``/``last_action`` auto-grow from the entity's actual
        actuator count (2 fewer than dual).
    So no other obs/reward re-registration is needed. The residual action's ``scale={"cam_yaw.*":0.3,
    "cam_pitch.*":0.2}`` (unchanged from the parent) already matches both "cam_yaw"/"cam_pitch" (single)
    and "cam_yaw_left"/"cam_yaw_right" (dual) since ``.*`` matches zero characters, so it needs no edit
    either. ``_ARM_ONLY_PATTERN`` (``^(?!leg)(?!cam_).*``) already excludes ANY ``cam_``-prefixed joint by
    name, single or dual, so the arm-only actuator/observation carve-outs the parent's chain built are
    unaffected too.

    History: replaces ``HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4_f3SingleCam_old`` (f3-based single-cam,
    now ``v2_best_single_old``). Same camera-swap mechanism, but on the v2ybsk_yaw_s4 base (foot_slip
    -0.2, NOT -0.4 — avoids f3's high-step cheat that motivated the 2026-07-31 v2_best demotion).

    Motivation: the visual_manipulation reach-grasp benchmark needs a genuinely single-camera-trained
    policy (v2_best_single) to pair with the existing dual-camera v2_best for a 4-way camera-count
    ablation (v2 / v2_fixed / v2_single / v2_single_fixed) -- swapping only the SCENE camera count on an
    otherwise dual-trained policy would leave 2 of its 4 trained camera-tracking obs/action dims
    permanently unused, not a fair single-camera baseline.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        _apply_single_camera(env)


class v2_best_single(HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlace):
    """v2_single policy pointer: single-camera sibling of v2_best, same reward lineage (2026-08-03).

    Inheritance: ...->v2GridGaitInit->v2GridGaitInitStartNearZero->...SingleCamTurnInPlace->
    v2_best_single. Tracks the v2_best retarget to GridGaitInitStartNearZero+lane. REQUIRES ITS OWN
    TRAINING RUN -- 29-dim action space vs the dual-cam 31, no checkpoint exists until trained.

    Per MEMORY: pass ``--checkpoint`` explicitly. Camera-count ablation must keep comparing
    camera count and nothing else, so this class is the single-cam twin of v2_best on the SAME
    lineage chain, not a divergent branch. Previous-lineage ckpt, kept for revert:
    ``runs/HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam/grl2_s0/model_0015000.pt``.

    History (most-recent-first):
      2026-08-03: retargeted to GridGaitInitStartNearZeroSingleCamTurnInPlace (single-cam sibling
                  of v2_best's lane chain). REQUIRES ITS OWN TRAINING RUN.
      2026-08-02: retargeted to GridGaitInitSingleCam, tracking v2_best -> GridGaitInit
                  (DOCUMENTED BUT NEVER APPLIED -- source still inherited ybsk_yaw_s4SingleCam
                  with no configure(), same trap as v2_best). REQUIRES ITS OWN TRAINING RUN.
      2026-08-02: retargeted to the StartNearZeroSingleCam lineage (VOIDED -- StartNearZero
                  promotion voided at the alias; the snz+single-cam variant inherits the same
                  shuffle artifact).
      2026-07-31: retargeted from f3-based single-cam to v2ybsk_yaw_s4-based single-cam (along
                  with the v2_best retarget from f3 on user visual verdict).
    """
    fallback_checkpoint_to_parent = True

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlat(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlace
):
    """Single-camera sibling of ``...StartNearZeroTurnInPlaceBankFlat``, the §10.16 winner.

    Supersedes ``...SingleCamTurnInPlaceBankPri`` (deleted 2026-08-05) for the single-camera lane: it
    when the prioritized rung was the expected winner, and the ladder then falsified it (§10.16 --
    the tracking deficit selects FASTER states, not slower ones). Retrained here on the rung that
    actually won so the single-camera lane tracks the dual-camera keeper rather than a dead rung.

    Must be TRAINED, not derived: camera count changes the observation width.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        from mjlab.managers.event_manager import EventTermCfg
        from tasks.humanoid_velocity.event import GaitBankResetFast
        env.events["reset_gait_keyframe"] = EventTermCfg(
            func=GaitBankResetFast,
            mode="reset",
            params={"fraction": 0.3, "s_max": 0.6, "sample_filled": True,
                    "n_bins": 1, "depth": 6144},
        )


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlatDecoupled(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlat
):
    """Single-camera sibling of ``...TurnInPlaceBankFlatDecoupled``, the dual-cam keeper.

    Fills the gap the single-camera lane has carried since 2026-08-05: the lane stopped at
    ``...SingleCamTurnInPlaceBankFlat`` while the dual-cam lane went on to promote the decoupled
    tracking-std lever, so ``v2_best_single`` has been tracking a rung behind ``v2_best``.

    Single lever vs ``...SingleCamTurnInPlaceBankFlat``: ``track_linear_velocity`` gets its own
    vertical std and a lower commanded-XY floor. The chain already runs
    ``track_linear_velocity_relative`` (from v30) with ``std_rel=0.5, std_min=0.3``, so only the two
    params move -- ``func`` and every other reward are untouched.

    Why it should transfer: the mechanism is "grade only what was commanded". The UNCOMMANDED
    vertical gait bob (~0.2 m/s on this robot) otherwise shares one std with the commanded XY error,
    which forces ``std_min`` up to 0.3 to avoid punishing the bob -- and that loose floor is what
    hides the low-command dead zone. Decoupling the bob onto ``std_z`` lets the floor drop to 0.1.
    Camera count does not touch the reward, so the mechanism is camera-independent.

    CAVEAT inherited from the dual-cam parent: ``std_z=0.3`` / ``std_min=0.1`` are FITTED to this
    robot's measured bob, not derived. Wave 5 tried to derive them and lost (``std_z=inf`` alone is
    identical to ...BankFlat, so the win is the FLOOR not the decoupling; the derived floor 0.125
    breaks zero-command calm). Do not re-derive. Detail:
    ``memory/deadzone_wave_campaign_2026_08_05.md``.

    Must be TRAINED, not derived: camera count changes the observation width. 29-dim action space vs
    the dual-cam 31, so always pass ``--checkpoint`` explicitly -- auto-resolution walks onto a
    dual-cam ancestor and dies on ``size mismatch for action_scale``.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        track_lin = env.rewards["track_linear_velocity"]
        track_lin.params["std_min"] = 0.1  # 0.3 -> 0.1: floor freed by the std_z split below
        track_lin.params["std_z"] = 0.3    # uncommanded vertical bob gets its own basin


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlatDecoupledWideBand(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlatDecoupled
):
    """Single-cam Decoupled + the wide BFS seed band. One change vs its parent.

    Single lever: ``initial_planar_speed_range`` (0.10, 0.25) -> (0.05, 0.6). The narrow value comes
    from ``StartNearZero``, which the single-camera chain inherits explicitly; the assignment here
    overrides it. Set explicitly rather than deleted-to-inherit-the-default, because the chain writes
    the narrow value above us -- dropping the line would leave (0.10, 0.25), not the cfg default.

    Premise: the BFS seeds every cell whose center speed lies in the band, so a band matching the
    gait-keyframe injection range (``event.py`` ``s_max=0.6``) starts the curriculum where the robot
    is actually placed, instead of leaving the slowest commands to a later BFS stage.

    MEASURED ON THIS CLASS (2026-08-07, seed 0, 9 cmds x 64 eps, ``--symmetric``). Each policy was
    evaluated THREE times to establish its own eval-noise band before any delta was read; a band is
    quoted, not a point, because per-command displacement at |vx|=1.0 scatters 7-9% between identical
    re-evals of one checkpoint. Ranges below are min-max over the 3 runs:
        term (raw)              Decoupled x3       WideBand x3    bands overlap?
        disp_m               1.5608-1.5768     1.6120-1.6816    no -- WB +4.2%
        action_rate_l2       0.6420-0.6455     0.7116-0.7243    no -- WB +11.5% WORSE
        track_linear_vel     0.7748-0.7774     0.7785-0.7817    no -- WB +0.5%
        foot_impact_vel      0.0151-0.0154     0.0159-0.0164    no -- WB +6.6% WORSE
        termination_rate     0.1771-0.1927     0.1545-0.1771    touch -- WB -9.1%
        foot_slip            0.0706-0.0776     0.0772-0.0799    YES -- not separable
        joint_torque             7.74-7.94         7.68-8.06    yes -- tie
        joint_power            100.4-101.4       100.4-102.6    yes -- tie

    Verdict: a real capability/smoothness TRADEOFF, not a regression. WideBand covers more ground,
    tracks linear velocity better and terminates less; it pays with decisively jerkier actions and
    harder foot impacts. Consistent mechanism: the wide seed band exposes more of the speed space
    early, buying coverage at the cost of the narrow band's low-speed polish.

    ``foot_slip`` is NOT separable ON THIS LANE -- the bands overlap. It IS a real regression on the
    dual-cam lane (see below), so slip is camera-config dependent, not a property of the seed band.
    Either way do not reach for the ``foot_slip`` weight: dead in BOTH directions (v2aa -0.2->-0.4
    REJECTED, vtf climbed to 0.615; f3 weight -0.4 DEMOTED on the user's visual verdict for a
    high-step cheat; v156 halve REJECTED for sim-to-real).

    DUAL-CAM LANE, re-measured 2026-08-07 with block identity verified off the ``Loaded checkpoint``
    lines, 2 evals per side (keeper `...BankFlatDecoupled/2026-08-05_00-34-46` vs wideband
    `/2026-08-07_00-35-46_flash_sac_wideband0`). NOTE THE BASELINES DIFFER: the dual-cam keeper
    rebuilds its command cfg and so trained on the cfg DEFAULT (0.25, 0.5), not StartNearZero's
    (0.10, 0.25) that this lane's parent uses. The dual-cam contrast is therefore
    (0.25, 0.5) -> (0.05, 0.6); the single-cam contrast above is (0.10, 0.25) -> (0.05, 0.6):
        term (raw)                keeper x2      wideband x2    bands overlap?
        action_rate_l2       0.6146-0.6149    0.7632-0.7684    no -- WB +24.5% WORSE
        foot_slip            0.0570-0.0623    0.0739-0.0746    no -- WB +25% WORSE
        joint_power              93.3-94.2      106.6-107.5    no -- WB +14% WORSE
        joint_torque             8.06-8.23        9.10-9.44    no -- WB +14% WORSE
        pose                 0.7750-0.7766    0.7509-0.7519    no -- WB WORSE
        foot_impact_vel      0.0139-0.0141    0.0146-0.0149    no -- WB +5.4% WORSE
        track_linear_vel     0.7634-0.7767    0.7563-0.7620    touch -- WB marginally worse
        disp_m               1.5530-1.6005    1.5857-1.5998    yes -- tie
        termination_rate     0.1701-0.1892    0.1597-0.1736    yes -- tie

    On dual-cam the band is NET NEGATIVE: six non-overlapping regressions, zero non-overlapping wins.
    The earlier dual-cam slip finding (+37% at one matched cell) therefore STANDS -- band-means put it
    at +25%, same direction. What does NOT stand is the top-speed displacement claim: mean disp is a
    tie, and per-command displacement at |vx|=1.0 scatters 7-9% between identical re-evals, so the
    original per-cell reading was inside noise.

    CROSS-LANE: only ``action_rate_l2`` and ``foot_impact_velocity`` regress on BOTH lanes -- those are
    the seed band's real cost, and the one result that survives the differing baselines. Slip and
    energy are dual-cam only; the displacement gain is single-cam only. No claimed BENEFIT replicates.
    Do not read the lane split as a camera-count effect: the baselines differ (see above), so lane and
    baseline are confounded. Separating them needs a dual-cam run at (0.10, 0.25); not worth it unless
    the band becomes load-bearing again, since widening loses on both lanes as measured.

    HOW THE FIRST READING WENT WRONG -- ``eval_policy.py`` built its block header from ``args.task``,
    so comparing two variants under one ``--task`` printed an IDENTICAL label for every block. The
    A/B was read off those headers and came out backwards. Fixed 2026-08-07: the label is now derived
    from the checkpoint path. When reading any multi-checkpoint eval, confirm block identity from the
    ``Loaded checkpoint from ...`` line, never from the summary header.

    Read this run against its parent, not against the dual-cam numbers: camera count changes the
    observation width, so only the sign and rough magnitude of the effect are expected to carry.

    Must be TRAINED, not derived. Pass ``--checkpoint`` explicitly (29-dim action space).
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["twist"].initial_planar_speed_range = (0.05, 0.6)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlatDecoupledCosine(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroSingleCamTurnInPlaceBankFlatDecoupled
):
    """Single-camera sibling of the dual-cam `...BankFlatDecoupledCosine` promote (2026-08-07).

    Same single change vs the single-cam keeper (`...SingleCamTurnInPlaceBankFlatDecoupled`):
    cosine anneal of actor+critic+alpha over 15000 iters to 5% via the `lr_decay_iters` path
    (NOT the muon-group-only path). The single-cam keeper had never trained with a schedule
    either, so the +8.49 effect from the dual-cam wave-2 cosine promote is expected to carry.

    Inherits `_apply_single_camera` from the chain at the same point its parent does, so the
    camera count (29-dim action) is preserved.

    Plus the same two-reward pop as the dual-cam Cosine (zero-weight phase terms that contribute
    nothing but print permanently-zero log rows). Single-variable change vs the parent.

    Must be TRAINED, not derived.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.lr_decay_iters = 15000
        sac_cfg.lr_min_factor = 0.05

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards.pop("foot_phase_contact_match", None)
        env.rewards.pop("foot_stance_slip_penalty", None)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled(
    HumanoidRmaVelEstArmFlashSac
):
    """...BankFlat + the ...GridDecoupledTrack tracking-std lever. One change. **CURRENT KEEPER for
    the low-command dead zone (promoted 2026-08-05, single seed under the decisive override).**

    SELF-CONTAINED TEMPLATE: parent is the task base ``HumanoidRmaVelEstArmFlashSac``, and EVERY
    intervening ``configure()`` body (v5 .. BankFlat, plus this class's own lever) is folded into the
    single ``configure()`` below. It is organized BY MDP COMPONENT (scene / commands / actions /
    observations / rewards / terminations / curriculum / events) rather than by historical rung, so an
    RL engineer reads the final MDP top-to-bottom; each setting carries a ``[vNN]`` tag naming the rung
    it came from for provenance. Behaviour is byte-identical to the old inheritance chain -- verified
    by diffing a full resolved env+agent cfg dump (empty diff, dict/term order included).

    Three archaeological collapses were applied (each gated by that empty diff, all presentation-only):
    (1) v123's integral-twist machinery (``IntegralErrorVelocityCommandCfg`` twist + ``command_integral``
    critic obs + ``track_integral_error`` reward) is OMITTED -- the Grid command replaces ``twist`` and
    pops those keys, so it never reached the final cfg; the pops remain as no-ops. (2) the successive
    ``track_angular_velocity`` std_rel writes (v2yb 0.5 -> v2ybi 1.0 -> v2ybdkc 0.7) collapse to one at
    0.7. (3) ``arm_ref`` / ``joint_pos_arms``, rebuilt twice in the chain (v29 all-arm, then v123 arm-only),
    are constructed once at their final arm-only form.

    Why root at the base: v5 .. StartNearZeroTurnInPlace are SHARED with the single-camera lane and the
    ``v2_best`` alias, so tuning any of them for single-cam silently retuned this keeper with no diff
    touching this class. A base-rooted template is immune to edits anywhere in that chain. Cost: the
    per-rung audit trail now lives in the ``[vNN]`` tags here; the rung classes themselves (``v5`` ..
    ``...BankFlat``) are KEPT as the campaign's measured baselines and are no longer ancestors.

    One collapse was applied that the chain could not express: ``...GridGaitInit`` installed a
    ``GaitKeyframeResetFast`` event under key ``reset_gait_keyframe`` which ``...Bank`` then replaced
    wholesale. No class between the two touches ``env.events``, so the key lands in the same dict
    position either way and the synthetic-keyframe step is dropped. ``...BankPool`` /
    ``...BankFlat``'s param mutations fold into that one construction, matching how
    ``...SingleCamTurnInPlaceBankFlat`` already writes it.

    ``std_z=0.3`` / ``std_min=0.1`` are fitted to THIS robot's measured ~0.2 m/s vertical gait bob.
    The MECHANISM (grade only what was commanded) is portable; the two CONSTANTS are not. Wave 5 tried
    to derive both from the task spec and LOST -- ``std_z=inf`` alone is identical to ...BankFlat (so
    the win is the FLOOR, not the decoupling), and the derived floor 0.125 breaks zero-command calm.
    Cite this class with that caveat attached. Detail: memory/deadzone_wave_campaign_2026_08_05.md.

    NOTE: ``...GridDecoupledTrack`` and ``...GridDecoupledTrackStand``, named throughout below, were
    DELETED in the 2026-08-05 experiment prune (restore from git ``f435632``). Their numbers stand as
    recorded here; the std_stand / std_z / std_min rationale itself lives in
    ``track_linear_velocity_relative`` (reward.py), which is the durable reference.

    ...GridDecoupledTrack is the only run on this whole lineage that beat the ybsk reference on the
    dead zone, and it has never been tested on the current base. Everything trained since 2026-08-02
    -- gait init, StartNearZero, the turn lane, the bank ladder -- changed WHERE episodes start or
    WHICH command is practised, and the dead-lever index already records the conclusion those runs
    produced: "the lever is reward or command coverage, not where episodes start". The reward half of
    that sentence was measured once, on a base three lineages old, and then left alone.

    So this asks the one question nothing on disk answers: does the reward-shape win survive on top
    of the current best? A lever measured against a superseded parent is not evidence about the
    current one, and the two could plausibly be redundant -- the bank and the lane both move low-
    command behaviour -- or additive.

    Decoupling and the floor drop ship together because they are provably inseparable, not as a
    bundle of convenience: with the floor left at 0.3 the decoupled form exp(-(xy² + vz²)/0.09) is
    bit-identical to the shared-std formula, so the decoupling alone is a no-op and the floor drop
    alone suppresses the gait bob. See track_linear_velocity_relative.

    Bar: clear the ...BankFlat two-seed spread (vx 0.15 0.28-0.33, vx 0.20 0.58-0.79), and do not
    regress turn-in-place (yaw<20% at (0,0,wz=0.2)) or termination. Carries the parent's known cost
    as a watch item: ...GridDecoupledTrack paid +55% cmd-0 joint_power for this win, and the sibling
    ...GridDecoupledTrackStand tests whether that cost is separable.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)  # base: legs-only locomotion + arm graph-nav scaffold + DR curriculum
        import re
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.observation_manager import ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.managers.termination_manager import TerminationTermCfg
        from asset_zoo.humanoid_v21 import get_humanoid_v21_robot_cfg
        from asset_zoo.humanoid_v21.humanoid_v21_constants import HUMANOID_V21_ACTION_SCALE
        from tasks.joint_ref_command import (
            CommandGatedArmGraphRefCommandTermCfg,
            GazeRefCommandTermCfg,
            JointPosRefResidualActionCfg,
        )
        from tasks.humanoid_velocity import event as task_event
        from tasks.humanoid_velocity.reward import (
            arm_joint_tracking,
            track_angular_velocity_relative,
            track_linear_velocity_relative,
        )
        from tasks.humanoid_velocity.observation import target_arm_joint_pos
        from tasks.humanoid_velocity.command import GridAdaptiveVelocityCommandCfg
        from tasks.humanoid_velocity.event import GaitBankResetFast

        N = self.num_steps_per_env  # curriculum steps are env-steps = iters * N

        # ============================== SCENE ==============================
        # Actuated twin-D435 head camera (vs the base's fixed head).                        [v123]
        env.scene.entities["robot"] = get_humanoid_v21_robot_cfg(head_camera="actuated")

        # ============================ COMMANDS ============================
        # twist: local (vx, vy, wz) GRID command curriculum. Rebuilt (not mutated) so the grid  [v2Grid
        #   fixes its own cell boundaries; the base range curriculum is dropped below. Seeds     +StartNearZero
        #   the BFS across the whole low-cmd band (cfg default) and carves a turn-in-place lane.  +TurnInPlace]
        twist = env.commands["twist"]
        env.commands["twist"] = GridAdaptiveVelocityCommandCfg(
            entity_name=twist.entity_name,
            resampling_time_range=twist.resampling_time_range,
            rel_standing_envs=twist.rel_standing_envs,
            debug_vis=twist.debug_vis,
            max_planar_speed=1.0,  # the planar speed actually trained to; clips the ranges below
            ranges=GridAdaptiveVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0),
            ),
            turn_lane_mass_range=(0.02, 0.25),  # dedicated turn-in-place lane
        )
        # arm_ref: collision-free arm graph-nav REFERENCE; the residual arm action tracks it.    [v29->v123
        #   Built once at its final arm-only form (chain built it twice: all-arm then arm-only).   +v145+v2ybsk]
        env.commands["arm_ref"] = CommandGatedArmGraphRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_ARM_ONLY_PATTERN,),
            robot_name="humanoid_v21",
            steps_per_edge=50,
            locomotion_vel_threshold=0.3,
            loco_min_steps=80,
            command_name="twist",
        )
        env.commands["arm_ref"].stand_min_steps = 80   # slow swing at zero base cmd (train-time) [v2ybsk]
        env.commands["arm_ref"].home_ratio_start = 1.0  # anneal ref toward home; never reaches     [v145]
        env.commands["arm_ref"].home_ratio_end = 0.5    #   the 0.5 floor within the 15k budget
        env.commands["arm_ref"].home_ratio_curriculum_steps = 18000 * N
        # camera_ref: random-walk gaze target (velocity-bounded by the RS05 kd).               [v123]
        env.commands["camera_ref"] = GazeRefCommandTermCfg(
            entity_name="robot",
            actuator_names=(_CAMERA_PATTERN,),
            min_steps=30, max_steps=120,
            yaw_range=4.7124,    # +-3pi/2 joint ROM
            pitch_range=1.5708,  # +-pi/2 joint ROM
            yaw_walk=1.0, pitch_walk=0.4,  # per-step walk deltas
        )

        # ============================= ACTIONS ============================
        # Residual joint targets: joint_target = joint_ref + scale * action.                    [v123]
        arm_scale = {k: v for k, v in HUMANOID_V21_ACTION_SCALE.items() if re.search(_ARM_ONLY_PATTERN, k)}
        env.actions["joint_pos_arms"] = JointPosRefResidualActionCfg(
            entity_name="robot", actuator_names=(_ARM_ONLY_PATTERN,),
            scale=arm_scale, use_default_offset=False, command_name="arm_ref",
        )
        env.actions["joint_pos_camera"] = JointPosRefResidualActionCfg(
            entity_name="robot", actuator_names=(_CAMERA_PATTERN,),
            scale={"cam_yaw.*": 0.3, "cam_pitch.*": 0.2},
            use_default_offset=False, command_name="camera_ref",
        )

        # =========================== OBSERVATIONS =========================
        # Re-point arm-tracking obs from the base's action name onto the arm_ref command,       [v29]
        arm_obs = env.observations["actor"].terms["target_arm_joint_pos"].params
        arm_obs.pop("arm_action_name", None)
        arm_obs["command_name"] = "arm_ref"
        # and add the camera-target obs to the deploy actor group.                              [v123]
        env.observations["actor"].terms["target_camera_joint_pos"] = ObservationTermCfg(
            func=target_arm_joint_pos, params={"command_name": "camera_ref"})
        # Integral-error obs machinery was never added on this lineage; pops are no-ops (kept    [v2Grid]
        #   because the Grid command owns coverage and would otherwise pop them).
        env.observations["actor"].terms.pop("command_integral", None)
        env.observations["critic"].terms.pop("command_integral", None)

        # ============================= REWARDS ============================
        # Linear-velocity tracking: relative std (basin scales with cmd) with the vertical bob   [v30
        #   DECOUPLED onto its own std_z so the commanded-XY floor (std_min) can drop. std_z /     +Decoupled]
        #   std_min are fitted to this robot's ~0.2 m/s gait bob -- see track_linear_velocity_relative.
        track_lin = env.rewards["track_linear_velocity"]
        track_lin.func = track_linear_velocity_relative
        track_lin.params.pop("std", None)
        track_lin.params["std_rel"] = 0.5
        track_lin.params["std_min"] = 0.1
        track_lin.params["std_z"] = 0.3
        # Angular-velocity tracking: relative std, tight low-cmd floor, up-weighted.             [v9 + v2yb
        track_ang = env.rewards["track_angular_velocity"]                                       #  ..v2ybdkc
        track_ang.func = track_angular_velocity_relative                                        #  + v2ybsk_yaw_s4]
        track_ang.params.pop("std", None)
        track_ang.params["std_rel"] = 0.7  # collapsed from the chain's 0.5 -> 1.0 -> 0.7
        track_ang.params["std_min"] = 0.05
        track_ang.weight = 2.5
        # Arm / camera joint-reference tracking (re-point arm reward to arm_ref; add camera).    [v29 / v123]
        arm_reward = env.rewards["arm_joint_tracking"].params
        arm_reward.pop("arm_action_name", None)
        arm_reward["command_name"] = "arm_ref"
        env.rewards["camera_joint_tracking"] = RewardTermCfg(
            func=arm_joint_tracking, weight=1.0,
            params={"std": 0.3, "command_name": "camera_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        # Regularizers / stability shaping.
        env.rewards["body_ang_vel"].weight = -0.05     # [v15]
        env.rewards["foot_clearance"].weight = -2.0    # [v159b]
        env.rewards["upright"].weight = 2.0            # anti-tilt [v2ybhci]
        env.rewards["action_rate_l2"].weight = -0.125  # quartered [v153]
        env.rewards.pop("track_integral_error", None)  # no-op (integral machinery dead) [v2Grid]

        # ========================== TERMINATIONS ==========================
        # Watchdog: end an episode that stalls low-cmd tracking for 70 consecutive steps.        [v18]
        env.terminations["velocity_tracking_failure"] = TerminationTermCfg(
            func=task_event.VelocityTrackingFailure,
            params={"cmd_xy_threshold": 0.1, "tracking_ratio": 0.3,
                    "consecutive_steps": 70, "command_name": "twist"},
        )

        # =========================== CURRICULUM ===========================
        env.curriculum["arm_action_l2"].params["weight_stages"][-1]["weight"] = -1.0            # [v5]
        env.curriculum["action_rate_l2"].params["weight_stages"] = [                            # [v153]
            {"step": 500 * N, "weight": -0.0125},
            {"step": 3000 * N, "weight": -0.125},
        ]
        env.curriculum["joint_torque"].params["weight_stages"] = [                              # [v24]
            {"step": 500 * N, "weight": -1e-5},
            {"step": 3000 * N, "weight": -5e-4},
        ]
        env.curriculum.pop("command_vel", None)  # Grid owns command coverage [v2Grid]

        # ========================= EVENTS (reset) =========================
        # Reset a fraction of envs into a REAL harvested walking state (gait bank), not a         [Bank
        #   hardcoded keyframe: sample_filled draws from rows the bank HOLDS (else 92% of           +BankPool
        #   slow-band draws no-op), n_bins=1 is a single flat ring at held-constant capacity.       +BankFlat]
        #   Appended last so it overrides the standing pose from reset_base / reset_robot_joints
        #   for its masked fraction; unmasked envs keep the standing reset.
        env.events["reset_gait_keyframe"] = EventTermCfg(
            func=GaitBankResetFast, mode="reset",
            params={"fraction": 0.3, "s_max": 0.6, "sample_filled": True,
                    "n_bins": 1, "depth": 6144},
        )


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledHomeRatioFix(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + home_ratio curriculum LENGTH fix. One change.

    NEW PREMISE (2026-08-08, rotation-at-zero-command bug): deterministic eval at cmd=(0,0,0),
    stand_min_steps=0 (deploy-equivalent), shows ang_err (mean |actual wz|) is 4.3x worse with the
    arm ACTIVE (home_ratio=0, matches deploy -- the real robot's arm always runs its task) than with
    the arm FROZEN (home_ratio=1): 0.0945 vs 0.0222 on this exact keeper. Arm-swing reaction torque
    (Newton's third law through the trunk) is the dominant driver of the reported real-robot rotation
    bug, not observation-normalizer bias (tested and REJECTED: n=100 zero-bias patch on
    obs_normalizer._mean showed no significant change, 0.0524 -> 0.0583, within noise -- see
    scripts/patch_deploy_angvel_bias.py and this session's eval logs).

    The parent's own docstring flags an UNEXPLOITED correction: home_ratio_curriculum_steps=18000*N
    exceeds the 15000-iter training budget, so home_ratio never reaches its documented 0.5 floor --
    training ends at frac=15000/18000=0.833 -> home_ratio~0.583 (58% frozen-arm exposure), while
    DEPLOY runs 100% arm-active (home_ratio=0 equivalent). This is a train/deploy exposure mismatch:
    the policy under-trains on the exact regime deploy always runs in.

    Single lever: home_ratio_curriculum_steps 18000*N -> 12000*N. This lets the anneal actually
    COMPLETE and HOLD its documented 0.5 floor for the final 3000 iters (vs never reaching it).
    home_ratio_start/end (1.0/0.5) UNCHANGED -- this only fixes the curriculum LENGTH bug, not the
    floor value itself (see sibling HomeRatioZero for the floor-value lever).

    Distinct from the 17-lever zerocmd_wave_2026_07_21 REJECT campaign
    (memory/zerocmd_wave_2026_07_21.md): that campaign tested reward-penalty tightening
    (body_ang_vel, action_rate_l2, upright, std_min) and rel_standing_envs / cmd-distribution changes
    on the g1b3/v159b lineage for general zero-cmd JITTER during manipulation. This lever touches
    neither reward weights nor command sampling -- it fixes an arm-EXPOSURE curriculum bug on the
    current v2_best lineage, targeting sustained YAW ROTATION specifically. Not a retest of any dead
    lever in that campaign.

    Verify: eval_policy.py --stand-min-steps 0 --home-ratio 0.0 --cmds "0,0,0" at 15k; must beat the
    keeper's ang_err=0.0945 (this exact arm-active condition) without regressing fell_over/tracking.

    VERDICT (2026-08-09, single-seed, grl2_vicon, 15k iter): REJECTED. n=100 arm-active zero-cmd eval:
    ang_err=0.0943 vs keeper 0.0929 -- statistically flat, within run-to-run noise (~10-15% observed
    band). The curriculum-length fix alone does not measurably change zero-cmd yaw behavior. See sibling
    HomeRatioZero for the more aggressive floor-value lever (also REJECTED, see that docstring for the
    full per-episode heading-drift analysis that explains WHY exposure fraction is not the mechanism).
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        N = self.num_steps_per_env
        env.commands["arm_ref"].home_ratio_curriculum_steps = 12000 * N


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledHomeRatioZero(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledHomeRatioFix
):
    """HomeRatioFix + anneal all the way to fully deploy-matched arm-active (home_ratio_end 0.5 -> 0.0).
    One change vs HomeRatioFix (both curriculum-length AND floor value now target the deploy condition).

    Brackets the arm-exposure lever: HomeRatioFix restores the ORIGINAL intended 50/50 floor within
    budget; this class goes further and trains toward the ACTUAL deploy distribution (100% arm-active,
    0% frozen). Risk: losing frozen-arm-state training diversity entirely may hurt some other axis
    (frozen state matters if the real task pauses the arm) -- bracket against HomeRatioFix to see if
    the extra floor drop buys more zero-cmd stability or just trades one regime for another.

    VERDICT (2026-08-09, single-seed, grl3, 15k iter): REJECTED, marginal at best. n=100 arm-active
    zero-cmd eval: ang_err=0.0869 vs keeper 0.0929 (-6.5%, inside the ~10-15% noise band established
    by repeat evals of the SAME checkpoint in this session). Signed per-episode heading-drift probe
    (mj_envs eval over 1000 steps / 20s at true cmd=0, n=50) is decisive: this class's drift
    distribution is NEARLY IDENTICAL to the keeper's --
        keeper:        mean -3.15 deg, |drift|>10deg 10/50, |drift|>20deg 2/50, max |drift| 24.7 deg
        HomeRatioZero:  mean -2.74 deg, |drift|>10deg 10/50, |drift|>20deg 1/50, max |drift| 23.4 deg
    Both distributions are skewed negative (~70% of episodes drift one sign) with a long tail --
    that skew + tail is the real hardware symptom ("rotates in place"), not the population MEAN
    |wz| that ang_err reports. The arm-exposure-mismatch hypothesis (this class + HomeRatioFix) is
    REFUTED as the primary mechanism: training with MORE arm-active exposure barely moves the tail.
    Root cause is more likely tied to the SPECIFIC arm graph-nav path drawn at each episode's reset
    (asymmetric reaction torque from a particular swing trajectory that the leg policy cannot learn
    to counter in general, not a coverage/exposure gap). See MEMORY.md rotation-at-zero entry for
    the full investigation, including the REJECTED normalizer zero-bias hypothesis tested first.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["arm_ref"].home_ratio_end = 0.0


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledNoTurnLane(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + turn-in-place lane DISABLED. One change.

    NEW PREMISE (2026-08-09, user hypothesis): the keeper's command sampler carves a dedicated
    turn-in-place lane (`turn_lane_mass_range=(0.02, 0.25)`, `GridAdaptiveVelocityCommand`) that
    commands `(vx=0, vy=0, wz=nonzero)` for 2-25% of moving envs, deficit-weighted toward whichever
    active yaw bin is currently WORST -- historically the slowest-turn bin (see
    `memory/yaw_deadzone_command_coverage.md`, std_min campaigns). The yaw grid axis is 9 bins over
    (-1,1) (cell width ~0.222, zero-centered per `grid_bins=(8,8,9)`), and `_active_turn_bins()`
    excludes only the CENTER bin (roughly (-0.111, +0.111) rad/s, reserved for standing via
    `rel_standing_envs`) -- so the lane's nearest active bin starts right at ~0.111 rad/s. Deficit
    weighting concentrates training mass on that slowest, hardest bin for much of the run.

    Mechanism: the policy is heavily and repeatedly trained (via deficit-weighted over-sampling) on
    "linear=0, yaw~0.11-0.33 rad/s" sitting immediately adjacent to "linear=0, yaw=0" (true standing)
    in command space, with the COMMAND VALUE as the only signal separating the two regimes from
    otherwise-identical proprioceptive state. If the network's learned response generalizes/smooths
    across nearby command values rather than sharply gating on it, the standing-adjacent zero-command
    state can inherit some of the "produce yaw" prior from the over-trained neighboring bin -- a
    command-generalization bleed distinct from every previously tested mechanism: NOT observation-
    normalizer bias (REJECTED, see MEMORY.md), NOT arm-exposure coverage (REJECTED, see
    HomeRatioFix/HomeRatioZero above), and NOT a reward-tighten axis (the 17-lever
    `zerocmd_wave_2026_07_21` campaign never touched command-sampling structure).

    Single lever: `turn_lane_mass_range` set to `None`, disabling the lane entirely -- moving envs
    fall back to pure grid sampling (`_sample_grid_commands`), so the near-zero-yaw regime is only
    ever reached via natural grid density, same as it was before the lane existed.

    KNOWN, ACKNOWLEDGED trade if this works: the lane was built to fix a real, measured yaw-tracking
    dead zone (`memory/yaw_deadzone_command_coverage.md`, "TurnInPlace v1": frozen 0.51->0.20, yaw p50
    0.196->0.528 at `(0,0,wz=0.2)`). Disabling it plausibly reintroduces that dead zone. This is not
    disqualifying on its own -- report the trade honestly; whether a smaller yaw-tracking regression
    is an acceptable price for fixing the hardware rotation bug is a promotion-time judgment call,
    not a reason to skip the test.

    Verify: signed heading-drift probe (`/tmp/rotation_fix_verify/signed_wz_probe.py` this session,
    or its equivalent) at true cmd=(0,0,0), n=50, 1000 steps -- compare skew/tail against the keeper's
    mean -3.15 deg, 10/50 episodes >10 deg, max 24.7 deg. Also run the standard `(0,0,wz=0.2)`
    turn-in-place probe to quantify the dead-zone trade-off.

    VERDICT (2026-08-09, single-seed, grl3, 15k iter): REJECTED, and the effect runs OPPOSITE to the
    hypothesis. Signed heading-drift probe (n=50, true cmd=0, 1000 steps): mean drift **-7.81 deg**
    (vs keeper -3.15 deg, 2.5x worse), |drift|>10deg **18/50** (vs keeper 10/50), |drift|>20deg
    **4/50** (vs keeper 2/50), skew more pronounced (46/50 episodes negative vs keeper's 36/50).
    Sanity-checked against a same-condition keeper eval on the default command sweep -- general
    locomotion competence is comparable (forward-cmd `term_rate`/`disp_m` closely match the keeper's,
    e.g. vx=1.0: 0.65/9.65 here vs 0.55/11.57 keeper), so this is not a degenerate/undertrained run,
    the zero-cmd result is real. **Disabling the lane does not remove a generalization-bleed source
    -- it REMOVES dedicated fine-grained training on precise low-yaw control near the zero boundary,
    and the policy gets WORSE at holding yaw near zero without it, not better.** The command-
    generalization-bleed mechanism proposed in this class's docstring is REFUTED. Root cause remains
    OPEN; see MEMORY.md rotation-at-zero entry for the full chain of rejected hypotheses (normalizer
    bias, arm-exposure coverage, turn-lane bleed) and what has not yet been tried.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.commands["twist"].turn_lane_mass_range = None


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledStand(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + the zero-command atom keeps the pre-decoupling std. One change.

    Top rung of the ladder: ...BankFlat -> ...BankFlatDecoupled (reward shape) -> this (calm fix).
    Two changes from ...BankFlat, one from each sibling, so the pair of rungs stays attributable
    rather than arriving as a stack -- the same laddering that made the BankPool/BankFlat result
    legible, where running the two changes separately was the only reason it was possible to say
    that neither was the effect on its own.

    Rationale for the std_stand split is identical to ...GridDecoupledTrackStand (class deleted
    2026-08-05, git f435632) and lives in track_linear_velocity_relative; this class only carries it
    onto the current base.

    Bar: the ...BankFlat two-seed spread on the dead zone, with cmd-0 joint_power no worse than
    ...BankFlat's. If ...BankFlatDecoupled clears the dead-zone bar and this one clears it while also
    holding calm, this is the promote candidate and a confirm seed is the next spend.
    """

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)
        env.rewards["track_linear_velocity"].params["std_stand"] = 0.3
        # Dead phase-clock seam, dropped. Both axes were closed by measurement: v162a
        # (foot_phase_contact_match 0 -> 1) collapsed tracking (exy 1.139, fell 0.195) and v162b
        # (foot_stance_slip_penalty 0 -> 1) gave nothing (fell 0.040, vtf 0.110). No class on this
        # ladder has raised either weight since. Numerically a no-op -- RewardManager.compute skips
        # weight == 0.0 terms -- so this only removes two permanently-zero log rows, and is NOT the
        # experiment's one change (std_stand above is).
        env.rewards.pop("foot_phase_contact_match")
        env.rewards.pop("foot_stance_slip_penalty")



# --- Muon optimizer screen (2026-08-07) -------------------------------------------------------
# Five lr rungs, each rooted DIRECTLY on ...BankFlatDecoupled rather than chained off one another.
# The keeper was flattened onto the task base for exactly this reason: a chain lets an edit to an
# intermediate rung silently retune every class below it. A one-line lr override is not worth
# reintroducing that coupling, so each rung repeats `use_muon = True` and pays two duplicated
# lines to stay independently editable.


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledMuonLr1e3(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """Keeper + Muon on the actor/critic 2D params, lr 1e-3. WAVE 1 -- kept as the constant-lr
    reference that wave 2 is graded against. The other four rungs (1e-4, 3e-4, 3e-3, 1e-2) were
    deleted 2026-08-07 once the ladder closed; recover from git if the lr axis ever reopens.

    Premise, and its honest weight. Kimi-K3, GLM-5 and DeepSeek-V4 independently replaced AdamW
    with Muon for matrix parameters -- three labs converging on one optimizer is the strongest
    signal any of those reports offers that transfers outside an LLM. It is also the ONLY idea from
    that read-through that survived screening: async rollout/training was killed by measurement
    (collect 0.167 s vs learn 0.055 s on the SAME GPU, so overlap caps at ~25%, and SAC's replay
    buffer already decouples what their async solves), staged distillation is already built here as
    L2T (flash_sac/config.py:141 onward, v58..v88), and outcome-weighted batch sampling is the
    closed BankPriNorm axis in the dead-lever index.

    Counter-evidence, stated up front. The motivation first offered for Muon here was that it would
    damp late-training value spikes. Checked against this keeper's own log: there are none -- critic
    loss runs 8.34 -> 4.38 and sits flat. So this is NOT fixing an observed pathology; it tests a
    transfer whose supporting evidence comes from models ~4 orders of magnitude larger, where
    conditioning dominates in a way it may simply not here (actor 0.33M params, qnet 1.06M).
    Coverage is at least total: 93.5% of actor and 99.2% of qnet parameter mass is ndim==2 and
    therefore on the Muon path (measured on a 30-iter smoke checkpoint).

    Why an lr ladder rather than one run. Muon's update is spectrally normalized -- Newton-Schulz
    drives every singular value into ~[0.67, 1.15] -- so its step is not commensurate with AdamW's
    per-element-RMS step and the usable lr is unknown a priori. A single run at AdamW's lr could not
    separate "Muon does not help here" from "Muon was mis-scaled", leaving the axis open. Five rungs
    1e-4 .. 1e-2 bracketed the range; 1e-2 is roughly where Muon is run at LLM scale.

    OUTCOME (2 seeds, 15000 iters, vs the same-wave ...BankFlatDecoupled baseline). This rung was
    the ladder's best and still LOST: 127.48 vs 132.11. Every other rung was worse and was stopped
    early. Critically the loss was not flat but a REVERSAL (+7.30 @1800 -> -4.63 @15000), and the
    cause was mechanical, not optimizer quality: `lr_decay_iters` defaults 0 and this lineage never
    overrode it, so lr is CONSTANT for all 15000 iters. AdamW tolerates that (per-parameter
    normalization means the step shrinks as gradients do); Muon cannot, because Newton-Schulz fixes
    the update's singular values by construction, so iter 15000 takes the same size step as iter 100.
    Wave 1 therefore did not test Muon -- it tested Muon denied the schedule every published Muon
    result uses. See ...MuonLr1e3Cosine below, which supplies the schedule and wins.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.use_muon = True
        sac_cfg.muon_learning_rate = 1e-3


# --- Wave 2: annealed Muon (2026-08-07) ------------------------------------------------------
# Wave 1 rejected Muon on return: best rung (lr 1e-3) finished 127.48 vs baseline 132.11. But the
# matched-iteration trace was not a flat loss -- it was a REVERSAL: +7.30 @1800, +1.42 @4650,
# -10.4 @8700, -4.63 @15000. Wave 1 never diagnosed the reversal, and the diagnosis is mechanical:
#
#   * `lr_warmup_iters` and `lr_decay_iters` both default to 0 and the keeper does not override
#     them, so `actor_scheduler is None` -- lr is CONSTANT for all 15000 iterations.
#   * For AdamW that is harmless: the update is per-parameter normalized, so as gradients shrink
#     near convergence the effective step anneals on its own.
#   * A Newton-Schulz update is orthogonalized to FIXED singular values by construction, so its
#     norm is independent of the gradient. Constant lr therefore means a literally constant step
#     size at iteration 15000 -- helpful far from the optimum, harmful next to it.
#
# So wave 1 confounded "Muon" with "Muon denied the annealing every published Muon result uses".
# These two classes separate them. New premise, so not a re-test of the rejected lever.
#
# The control is load-bearing: cosine decay might simply help ANY optimizer here, since the keeper
# has never run with a schedule at all. Without ...Cosine, a win by ...MuonLr1e3Cosine over the
# wave-1 baseline is unattributable. Wave-1's two baseline seeds (constant lr, 15000) are the
# third arm and do not need retraining.


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledMuonLr1e3Cosine(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """Keeper + Muon lr 1e-3 + cosine anneal of the MUON GROUP ONLY, to 5% over the full run.

    Single variable against ...MuonLr1e3 (wave 1): the AdamW fallback group inside the same
    optimizer, and the alpha optimizer, both keep their constant baseline lr -- see the per-group
    LambdaLR construction in `flash_sac/runner.py` and the assertions in `flash_sac/muon.py`.

    Grading bar: wave 1's lead peaked at +7.30 @1800 before decaying. If annealing is the fix,
    this should hold a lead at 15000 rather than surrender it; anything that again crosses below
    the baseline by ~8700 closes the axis for good, because the only remaining explanation would
    be that Muon's advantage is confined to early training regardless of step size.

    min_factor 0.05 rather than the config default's 0.1: the failure is specifically too-large
    steps late, so the schedule should end nearer zero than a generic cosine would.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.use_muon = True
        sac_cfg.muon_learning_rate = 1e-3
        sac_cfg.muon_lr_decay_iters = 15000
        sac_cfg.muon_lr_min_factor = 0.05


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """CONTROL: keeper + AdamW + the same cosine anneal, no Muon.

    Isolates the schedule from the optimizer. The keeper has never trained with any lr schedule,
    so cosine decay is itself untested here; if this beats the constant-lr baseline by as much as
    ...MuonLr1e3Cosine does, the win belongs to the schedule and Muon stays rejected.

    Uses the pre-existing `lr_decay_iters` path, which anneals actor/critic/alpha together --
    deliberately NOT the muon-group-only path, since the point is to decay everything AdamW drives.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.lr_decay_iters = 15000
        sac_cfg.lr_min_factor = 0.05

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        # Drop two zero-weight phase terms. RewardManager.compute skips weight==0.0 terms, so
        # these contribute nothing; popping cleans two permanently-zero log rows. The siblings
        # ...BankFlatDecoupledStand (line 3129) and ...GridAdaptiveCommands (line 3133-3139) do
        # the same pop -- bringing this class in line with that convention. Single change vs the
        # promoted run: removes log rows, no effect on training math.
        env.rewards.pop("foot_phase_contact_match", None)
        env.rewards.pop("foot_stance_slip_penalty", None)


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledMuonLr1e3CosineAll(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """CONFOUND-CLOSER for ...MuonLr1e3Cosine vs ...Cosine (2026-08-07). Anneal EVERYTHING.

    Why this exists. The wave-2 pair was read as "Muon adds +2.32 on top of cosine", but that
    comparison is not single-variable -- the two arms anneal different parameter SETS:

        arm         2D params           non-2D params      alpha lr
        Cosine      AdamW, cosine       AdamW, cosine      cosine     (lr_decay_iters=15000)
        MuonLr1e3Cosine  Muon, cosine   AdamW, CONSTANT    CONSTANT   (muon_lr_decay_iters only)

    So the +2.32 confounds the optimizer swap with "did the non-2D params and alpha also decay".
    Bounding evidence said the confound is probably small -- non-2D is 6.5% of actor / 0.8% of qnet
    parameter mass, and final entropy is identical across arms (-14.77..-14.83) even though alpha
    lands higher without its anneal (4.2e-4 vs 3.7e-4) -- but "probably" is not measured.

    This class sets BOTH schedules, so its only difference from ...Cosine is the optimizer on the
    2D params. Read it as: (this - Cosine) is Muon's true isolated contribution.

    OUTCOME (2 seeds, 15000 iters, ser15 GPUs 1/2): 143.00 / 139.83, mean 141.42, seed spread 3.17.

        Muon isolated  = CosineAll - Cosine   = +0.82   <-- vs a 3.17 spread in its own arm
        anneal cost    = CosineAll - MuonCos  = -1.50   (decaying alpha/non-2D HURTS the Muon arm)

    The attribution COLLAPSED. +0.82 is ~4x smaller than the seed noise, and on the per-1000-step
    sign-corrected gait block 0 of 7 terms exceed seed spread (direction 4/7 = coin flip, sum
    +0.0633 vs spread 0.0950). Resolving +0.82 against per-seed sd ~2.2 needs ~100 seeds per arm.
    So the wave-2 "+2.32 for Muon" was mostly the annealing SCOPE, not the optimizer.

    PROMOTE IS ...Cosine: +8.49 over baseline, tightest seeds of any arm (0.58), best fell_over
    (0.0025), zero train-time cost. Muon is not refuted -- it is unmeasurable at this scale (0.33M
    actor / 1.06M qnet), which is the extrapolation risk ...MuonLr1e3's docstring flagged up front.

    Keep this class. It is the only arm that makes the Muon axis falsifiable, and re-deriving why
    the obvious muon-vs-cosine comparison is invalid cost three separate written verdicts.
    """

    def flash_sac_configure(self, sac_cfg) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.use_muon = True
        sac_cfg.muon_learning_rate = 1e-3
        sac_cfg.muon_lr_decay_iters = 15000
        sac_cfg.muon_lr_min_factor = 0.05
        sac_cfg.lr_decay_iters = 15000
        sac_cfg.lr_min_factor = 0.05


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineAR(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosine
):
    """Single change vs ...Cosine: action_rate_l2 final weight -0.125 -> -0.25 (2x).

    Targets the "trips easily / knee bends too much on hardware" symptoms. action_rate_l2
    penalises (a_t - a_{t-1})^2 per joint, so doubling it pushes the policy toward smoother
    action sequences -- the natural mechanism for both smaller knee-bend excursions under
    a noisy command and lower peak angular-momentum injection on a small disturbance (the
    same direction the 5.8 Hz Muon hardware chatter took, but the deployment noise floor
    is what we are fixing, not the optimizer). MOTIVATION: MEMORY 96 explicitly records
    action_rate_l2 tighten as WITHDRAWN only because the original lever used the
    curriculum-overwrite bug (curriculum'd term name set term_cfg.weight, which the
    bug ate); on this spine the term IS curriculum'd so the fix is mutating
    env.curriculum["action_rate_l2"].params["weight_stages"] in place -- the exact
    path the always-on critical 12 documents. Single variable change vs ...Cosine,
    verified by diff of weight_stages against the parent at line 3116-3119 (stage 0
    unchanged; stage 1 -0.125 -> -0.25).

    Bar: deterministic eval `fell_over` must not regress on the 9-cmd sweep; expect a
    term_rate hit and joint_torque / joint_power drops (smoother = less work). Pair
    with hardware verification on the deployed JIT -- sim is blind to Muon chatter
    (MEMORY 156), so this lever MUST be hardware-validated before promoting.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.curriculum["action_rate_l2"].params["weight_stages"] = [
            {"step": 500 * self.num_steps_per_env, "weight": -0.0125},  # stage 0 unchanged
            {"step": 3000 * self.num_steps_per_env, "weight": -0.25},  # stage 1 tightened 2x
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledPdgainTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + PD-gain DR range narrowed from (0.9, 1.05) to (0.95, 1.02). One change.

    NEW PREMISE (2026-08-13, sim2real robustness): MEMORY line 161 records that within the TRAINED
    PD-gain range (+-5% on kp/kd) the per-episode DR draw does not predict which episodes drift,
    but explicitly notes the real robot's fixed calibration could sit OUTSIDE that trained range
    altogether -- a true sim2real gap the per-episode sim probe cannot test. This lever narrows
    the trained range to a HARDWARE-NEAR band (+-2% on kp/kd), so the policy trains closer to
    the actual hardware operating point and has less variance to absorb at deploy.

    Distinct from `...MuonLr*` family (optimizer-only) and from `...Cosine` (lr schedule) -- this
    touches only the DR machinery, single-var. Bracket against the keeper with the signed-heading
    drift probe (n=50, true cmd=0, 1000 steps); a clear tail reduction would be the success bar.
    Tracking-failure termination rate and forward-cmd locomotion should stay within seed noise.

    Verify: signed-drift probe + 9-cmd deterministic eval at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        # IMPORTANT: pd_gains ranges are overwritten by the `domain_randomization` curriculum
        # at runtime (base cfg lines 1140-1143). Mutating the event params alone is silently
        # overridden at the curriculum step. Override the curriculum stages instead.
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["pd_gains"] = [
            {"step": 200 * self.num_steps_per_env, "kp_range": (0.97, 1.03), "kd_range": (0.97, 1.03)},
            {"step": 5000 * self.num_steps_per_env, "kp_range": (0.95, 1.02), "kd_range": (0.95, 1.02)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledMassTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + body_mass DR range narrowed from (0.85, 1.15) to (0.92, 1.08). One change.

    NEW PREMISE (2026-08-13, sim2real robustness): same hardware-near rationale as the sibling
    PdgainTight -- MEMORY line 161 left the question "does the real robot's mass calibration sit
    outside the trained +-15% band?" open. The pose signature on hardware matches a model with
    a +~3 cm WHOLE-BODY CoM offset (battery/cabling/head), which is plausible if the mass model
    itself is also off. Narrowing the trained mass range to +-8% (still wide enough to cover
    battery state-of-charge variation, but closer to nominal) should reduce over-fit to mass
    extremes the hardware never reaches.

    Does NOT touch inertia (a separate lever); does NOT touch com_displacement (separate).

    IMPORTANT: body_mass ranges are overwritten by the `domain_randomization` curriculum at
    runtime. Override the curriculum stages, not just the event params.

    Verify: signed-drift probe + forward-cmd tracking at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["body_mass"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (0.96, 1.04)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (0.92, 1.08)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledFrictionTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + joint_frictionloss DR range narrowed from (-0.1, +0.1) to (-0.05, +0.05). One change.

    NEW PREMISE (2026-08-13, sim2real robustness): same hardware-near rationale as the sibling
    MassTight / PdgainTight -- if the real robot's joint friction calibration is consistent and
    narrow (a well-lubricated gear train measured once at the factory), training the policy to
    absorb +-0.1 Nm friction variation per DOF trains for a robustness the hardware doesn't need
    and pulls mass away from the nominal regime. The base cfg curriculum sets the FINAL stage to
    +-0.1 (loose), so narrowing to +-0.05 targets a hardware-near band while keeping the warmup
    stage at the base's +-0.01.

    Acknowledge risk: friction variation IS a real sim2real axis at deploy (temperature, wear),
    so under-training it can hurt. Half-range is a conservative first move; widen if it loses.

    IMPORTANT: joint_frictionloss ranges are overwritten by the curriculum at runtime. Override
    the curriculum stages.

    Verify: signed-drift probe + locomotion competence at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["joint_frictionloss"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (-0.01, 0.01)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (-0.05, 0.05)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCmdFailShort(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + velocity_tracking_failure.consecutive_steps 70 -> 35. One change.

    NEW PREMISE (2026-08-13, drift-tail mitigation): the keeper's VTF patience is 70 steps; an
    episode that stalls near zero command can accumulate >1000 steps (full episode length) without
    triggering VTF, giving the policy up to ~20 s of pure-drift time per episode. Reducing patience
    to 35 cuts that budget in half -- a tail episode that would have drifted 18 deg gets terminated
    at ~9 deg. Doesn't fix the root cause, but bounds the worst-case by half.

    Cost: VTF becomes a noisier estimator of "actually lost tracking" vs "just slow". Forward-cmd
    episodes may see spurious terminations during slow-curriculum phases. Mitigation: VTF's
    reorient exemption and tracking_ratio gate already filter some of this; if forward-cmd
    `term_rate` regresses >5% the lever is too aggressive.

    Distinct from any rejected drift lever: the rejected ones (HomeRatioFix/Zero, NoTurnLane)
    changed the training distribution; this only changes the termination gate.

    Verify: signed-drift tail count (|drift|>10deg / n=50) at 15k; bar = strictly fewer than
    keeper's 10/50 without forward-cmd regression.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.terminations["velocity_tracking_failure"].params["consecutive_steps"] = 35


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledComOffsetUp(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + com_displacement DR range widened from (-0.01, 0.01) to (-0.03, 0.03). One change.

    NEW PREMISE (2026-08-13, sim2real robustness): MEMORY forward-lean entry concludes the real
    robot has a ~3 cm whole-body CoM offset that the trained policy does not anticipate. The
    base-link `com_displacement` range is +-1 cm; widening to +-3 cm lets the policy see the
    regime hardware actually sits in and learn a more robust stance around it.

    Targets the FORWARD-LEAN sim/hardware mismatch (not the drift tail). Bracket against the
    keeper's hardware lean signature -- sim-side knee/hip match should move toward hardware's
    straighter-knee stance if this lever reaches the right regime.

    Risk: wider COM offset can destabilize walking during training (more extreme spawn poses).
    Mitigation: 3 cm is the estimated upper bound of the hardware offset; if it destabilizes
    scale back to +-2 cm.

    Verify: lean signature (knee asym, hip bend, ankle) on a deployed ckpt + signed-drift probe.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["com_displacement"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (-0.001, 0.001)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (-0.03, 0.03)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledFootFrictionTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + foot_friction range narrowed from (0.3, 1.5) to (0.5, 1.2). One change.

    NEW PREMISE (2026-08-13, sim2real robustness): the keeper's foot friction range (0.3-1.5) is
    very wide and covers regimes the real robot's feet-on-floor never reach (rubber-on-concrete
    sits reliably around 0.7-1.1 depending on surface). The TODO CHECK comment in the cfg itself
    flags that this range was never measured against a hardware reference. Narrowing to (0.5, 1.2)
    still covers the hardware-near band and drops two extreme regimes the policy wastes mass on.

    `mode="startup"` -- set once per training run, so this does NOT add per-episode variance, only
    removes some.

    Verify: forward-cmd locomotion + slip-related reward terms at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["foot_friction"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (0.7, 1.0)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (0.5, 1.2)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledStdRelTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + track_linear_velocity std_rel 0.5 -> 0.3. One change.

    NEW PREMISE (2026-08-13, sim2real robustness): the linear-tracking reward has a relative
    std that defines how much velocity error is "good enough" -- std_rel=0.5 means an error of
    0.5 m/s gives exp(-1)~0.37 of reward (the steep part of the basin). Real hardware has a
    tighter basin (measured ~0.18 m/s mean error on flat per ArgusMini analysis, suggests a
    similar humanoid story). Tightening to std_rel=0.3 trains the policy to aim at a sharper
    basin that matches the actual hardware capability, not the looser one sim currently rewards.

    Targets the +33% flat-ground tracking-error floor (per ArgusMini finding, likely similar on
    humanoid). Does NOT touch std_min (the gait-bob floor stays at 0.1).

    Risk: tighter basin can over-fit to the current gait cycle and lose robustness to a wider
    command range. Mitigation: std_min still floors the basin for slow cmds; the tightening
    only affects mid-to-fast regime.

    Verify: forward-cmd tracking error + dead-zone regression at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["track_linear_velocity"].params["std_rel"] = 0.3


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledBodyAngVelUp(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + body_ang_vel reward weight -0.05 -> -0.15 (3x). One change.

    NEW PREMISE (2026-08-13, drift-tail mitigation): the keeper penalizes trunk angular velocity
    at -0.05 per rad/s of roll/pitch combined (the reward.py docstring notes it's base-frame,
    yaw-invariant). The drift symptom is yaw, but body_ang_vel is yaw-INVARIANT -- raising its
    weight here targets PITCH/ROLL oscillation, not yaw. Mechanism is INDEPENDENT of the drift
    axis but plausibly helps if a small secondary oscillation amplifies the yaw drift through
    coupling. Cheaper lever than the drift-tail probes.

    The 17-lever zerocmd_wave_2026_07_21 campaign (per MEMORY line 151) tested body_ang_vel
    tighten on an older lineage (g1b3/v159b) -- this is the same lever on the CURRENT lineage,
    where the dead-lever check applies to that lineage only. The current keeper's reward
    weights differ, so this is NOT a retest of a documented dead lever.

    Risk: tighter body_ang_vel may make the policy less compliant under push disturbances
    (the push_robot perturbation already exists). Mitigation: a 3x bump from -0.05 to -0.15
    is well below the saturation regime where compliance collapses.

    Verify: signed-drift probe + push-recovery metrics at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["body_ang_vel"].weight = -0.15


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledMassLoose(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + body_mass DR range WIDENED from (0.85, 1.15) to (0.7, 1.3). One change.

    NEW PREMISE (2026-08-13, sim2real robustness bracket): the sibling MassTight tests the
    "hardware-near" hypothesis (narrower DR); this tests the OPPOSITE end of the same axis --
    wider DR. If MassTight regresses forward locomotion and MassLoose also regresses, mass DR
    sensitivity is the dominant axis and the keeper's +-15% is already optimal. If MassLoose
    HELPS, the policy is under-trained for hardware mass variation and the keeper's band is too
    narrow.

    Pure bracket: same axis, opposite direction. Per single-variable rule this is a single-var
    lever; the BRACKET as a whole tests the axis, not a multi-lever bundle.

    Verify: forward-cmd tracking error + termination rate at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["body_mass"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (0.9, 1.1)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (0.7, 1.3)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledFrictionLoose(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + joint_frictionloss DR range WIDENED from (-0.01, +0.01) to (-0.02, +0.02). One change.

    NEW PREMISE (2026-08-13, sim2real robustness bracket): sibling of FrictionTight. If
    FrictionTight helps (narrower DR is more robust to the hardware-near friction calibration),
    FrictionLoose tests the opposite hypothesis (the policy needs MORE friction tolerance, e.g.
    to handle a cold-gear or dusty-joint regime). If both regress, the keeper's +-0.01 is the
    local optimum.

    Pure bracket, single-var lever. Single-variable rule still holds (this is one lever, not
    a bundle with FrictionTight).

    Verify: tracking + termination at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["joint_frictionloss"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (-0.01, 0.01)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (-0.02, 0.02)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledVTFMore(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + velocity_tracking_failure.consecutive_steps 70 -> 100. One change.

    NEW PREMISE (2026-08-13, drift-tail bracket): sibling of CmdFailShort (which tightens VTF to
    35 to bound drift-tail episodes). This widens VTF to 100, giving drift episodes MORE budget
    to either self-correct or accumulate drift. If CmdFailShort and VTFMore both regress
    forward-cmd tracking, the keeper's 70 is local optimum. If CmdFailShort helps and VTFMore
    also helps, the drift mechanism is upstream of VTF and neither lever reaches it.

    Pure bracket, single-var.

    Verify: signed-drift tail + forward-cmd tracking at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.terminations["velocity_tracking_failure"].params["consecutive_steps"] = 100


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledComOffsetTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + com_displacement DR range NARROWED from (-0.01, 0.01) to (-0.005, 0.005). One change.

    NEW PREMISE (2026-08-13, sim2real robustness bracket): sibling of ComOffsetUp (which widens
    to +-0.03 to expose the policy to the hardware's ~3 cm CoM offset). This NARROWS to +-0.005
    to test whether a more precise CoM assumption improves forward tracking without sacrificing
    the lean tolerance. If both regress, the keeper's +-0.01 is local optimum and the right
    fix is to widen ONLY for the lean-compensation path, not the full DR.

    Pure bracket, single-var.

    Verify: forward tracking + lean signature at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["com_displacement"] = [
            {"step": 200 * self.num_steps_per_env, "ranges": (-0.001, 0.001)},
            {"step": 5000 * self.num_steps_per_env, "ranges": (-0.005, 0.005)},
        ]


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledUpRightUp(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + upright reward weight 2.0 -> 3.0. One change.

    NEW PREMISE (2026-08-13, forward-lean mitigation): the keeper rewards trunk upright at
    weight=2.0. The real robot's trunk sits ~5 deg nose-down (MEMORY line 167) -- a posture
    signature the keeper reward does not currently target strongly enough to bias toward.
    Raising the upright weight to 3.0 increases the penalty for any trunk tilt, which may push
    the policy toward a more vertical stance on hardware.

    Risk: the upright reward fires in both directions from vertical. A heavier weight makes
    ANY tilt more costly, which could reduce trunk compliance under pushes and during forward
    motion. Mitigation: 50% bump is below the regime where compliance typically collapses;
    observe push-recovery at eval.

    Verify: hardware lean signature (knee, hip, ankle) + forward tracking at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["upright"].weight = 3.0


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledTrackAngTight(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + track_angular_velocity std_rel 0.7 -> 0.4. One change.

    NEW PREMISE (2026-08-13, sim2real robustness): the angular-tracking kernel uses std_rel=0.7
    (so a yaw error of 0.7 rad/s gives exp(-1)~0.37 of reward). Tightening to 0.4 sharpens the
    basin toward zero yaw error, potentially reducing drift tail episodes that sit just outside
    the current loose basin.

    Bracket against `...StdRelTight` (which does the linear axis, also std_rel tightened) -- if
    both regress, the std_rel axis is too sensitive to tighten on either component. If
    TrackAngTight helps and StdRelTight doesn't, the yaw axis is the discriminating one.

    Pure bracket, single-var.

    Verify: signed-drift tail + turn-in-place tracking at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].params["std_rel"] = 0.4


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledHomeRatioFloor(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + home_ratio_end 0.5 -> 0.3 (mid-bracket). One change.

    NEW PREMISE (2026-08-13, drift-tail bracket): sits between the two rejected arm-exposure
    levers -- HomeRatioFix (length only, 0.5 floor reached) and HomeRatioZero (0.0 floor,
    deploy-matched). HomeRatioFloor anneals to 0.3 instead of 0.5 or 0.0 -- a middle ground
    that exposes the policy to more deploy-like arm-active training (60% instead of 50% or
    100%) without going all the way to deploy-matched.

    Likely DEAD per the 2026-08-09 verdict (drift tail distribution nearly identical at 0.5
    and 0.0 floors), but worth confirming the axis is monotonically dead rather than a
    V-shape -- if 0.3 is also flat, the arm-exposure lever is rejected across the entire
    range, not just at its endpoints.

    Verify: signed-drift tail at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.commands["arm_ref"].home_ratio_end = 0.3


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledFootClearUp(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + foot_clearance weight -2.0 -> -3.0 (1.5x). One change.

    NEW PREMISE (2026-08-13, sim2real robustness): the keeper penalizes foot clearance at -2.0
    to keep feet from dragging on the floor. Hardware-side foot clearance on a flat surface is
    typically well-bounded; raising the penalty trains for higher clearance which helps in
    deployment on uneven surfaces (cables, thresholds) without losing flat-surface competence.

    Distinct from `zerocmd_wave_2026_07_21` levers (which targeted zero-cmd jitter, not foot
    clearance) -- this lever's target axis is terrain robustness, not zero-cmd stability.

    Verify: forward-cmd tracking + foot_clearance term trajectory at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["foot_clearance"].weight = -3.0


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledTrackAngLoose(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + track_angular_velocity std_rel 0.7 -> 1.0. One change.

    NEW PREMISE (2026-08-13, sim2real robustness bracket): opposite of TrackAngTight. If
    TrackAngTight (std_rel 0.7 -> 0.4) helps, this (0.7 -> 1.0) tests the OTHER direction --
    a looser basin that may give the policy more room to learn smoother yaw trajectories
    without over-fitting to tight tolerance.

    Bracket against TrackAngTight.

    Verify: signed-drift tail + turn-in-place tracking at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["track_angular_velocity"].params["std_rel"] = 1.0


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledUprightDown(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + upright weight 2.0 -> 1.0. One change.

    NEW PREMISE (2026-08-13, sim2real robustness bracket): opposite of UpRightUp. If UpRightUp
    helps push the policy toward a more vertical stance, this tests the opposite -- a more
    relaxed upright reward may allow a more natural stance that better matches hardware's
    forward lean without the policy fighting the reward.

    Bracket against UpRightUp.

    Verify: hardware lean signature + forward tracking at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.rewards["upright"].weight = 1.0


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledPdGainLoose(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupled
):
    """...BankFlatDecoupled + pd_gains DR range widened from (0.9, 1.05) to (0.85, 1.15). One change.

    NEW PREMISE (2026-08-13, sim2real robustness bracket): opposite of PdgainTight. If
    PdgainTight (0.9, 1.05 -> 0.95, 1.02) helps by training closer to hardware-nominal,
    PdGainLoose (0.9, 1.05 -> 0.85, 1.15) tests the OTHER hypothesis -- the policy needs
    MORE PD-gain tolerance to handle calibration drift / temperature effects across deploys.

    Bracket against PdgainTight.

    Verify: forward tracking + termination rate at 15k.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        dr_params = env.curriculum["domain_randomization"].params
        dr_params["events"]["pd_gains"] = [
            {"step": 200 * self.num_steps_per_env, "kp_range": (0.95, 1.05), "kd_range": (0.95, 1.05)},
            {"step": 5000 * self.num_steps_per_env, "kp_range": (0.85, 1.15), "kd_range": (0.85, 1.15)},
        ]




class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStand(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineAR
):
    """...CosineAR + `pose` std_standing tightened on the waist only (0.1 -> 0.05). One change.

    NEW PREMISE (2026-08-14, hardware IK-reach report): the user reports that walking on the real
    robot is smooth (the CosineAR promote holds) but the torso rotates consistently while the arm
    holds an IK reach pose, which corrupts a world-fixed reach target.

    Mechanism, measured not assumed (`mj_envs/probe/waist_yaw_reach_probe.py`, 128 lanes, standing,
    arm reference FROZEN at a far graph node vs frozen at home):

        ckpt              |waist| home -> reach   waist p95   base yaw @reach
        Decoupled  s0/s2   2.45->2.38 / 1.02->3.60    5.7/10.0     9.5 / 16.1
        Cosine     s0/s1   0.68->3.66 / 1.71->4.86    9.5/ 9.8     9.6 /  6.2
        CosineAR   s0/s1   0.67->4.79 / 1.13->4.29    9.9/11.2     7.0 /  6.2   (deployed)

    Lineage-wide, ~2.6-4.1 deg of extra waist yaw and roughly doubled base yaw drift, NOT a
    Cosine/AR regression. `waist_joint` is a yaw hinge sitting between base_link (which carries the
    torso, BOTH arms and the cameras) and the legs -- so with the feet planted, waist yaw rotates
    the entire upper body in the world. That is the reported symptom.

    The policy COMMANDS it: at reach, commanded 4.74 deg vs actual 4.79 deg, servo error 0.16 deg.
    Not droop, not a soft joint -- stiffer waist gains cannot fix this.

    Why the price is wrong rather than the behaviour. `pose` (variable_posture, weight 1.0) is
    already restricted to `_LEG_PATTERN` (13 joints = waist + both legs), so the arms do not dilute
    it, and at standing (`walking_threshold` 0.05) it uses `std_standing = {".*": 0.1}`. The reward
    is `exp(-mean(err^2 / std^2))`, so the measured 4.79 deg = 0.0836 rad costs
    `(0.0836/0.1)^2 / 13 = 0.054` of exponent -> 5.2% of one weight-1.0 term, with a restoring
    gradient of only `2*err/(std^2 * 13) = 1.29 /rad`. The policy buys its balance twist for ~5% of
    one term. Halving the waist std to 0.05 quadruples both: 19.4% cost, 5.15 /rad gradient.

    Why std_standing and not std_walking: `walking_threshold=0.05` gates the three regimes on
    commanded speed, and turn-in-place is commanded at |wz| ~ 0.5, so it resolves to std_walking.
    Tightening std_standing therefore CANNOT touch turning. That distinction is load-bearing --
    a naive deploy-side waist clamp was tried and destroys steering: at cmd (0,0,0.5) the robot
    accumulates 113 deg over the 4 s settle window normally but only 12 deg with the waist pinned,
    because the waist IS the policy's yaw actuator. The same clamp was tried on hardware and
    destabilised the robot, both ungated and gated on zero command. Hence a reward lever, which
    reprices the waist at standing while leaving its authority intact everywhere else.

    Existence proof that the target is reachable: the same probe with the waist action pinned to 0
    at standing gives |waist| 0.14 deg, base yaw 7.00 -> 4.08 deg and fell 0.000. The legs and
    ankles CAN absorb an extended-arm load without the twist; this lever asks the policy to learn
    that natively instead of having it imposed.

    Not a re-test. MEMORY's waist plateau (`memory/waist_integration_plateau.md`) forbids LOOSENING
    waist std (v165 collapsed at 3.0) and forbids waist REFERENCE TRACKING; this tightens the
    home anchor, the same direction the anchor already pulls. MEMORY 90's v159d moved variable_posture
    WEIGHT (1.0->0.5, rejected). The 17-lever `zerocmd_wave_2026_07_21` campaign tested body_ang_vel /
    action_rate_l2 / upright / track std_min and `rel_standing_envs`, never variable_posture std.

    Bearing on MEMORY 152/154 (rotation-at-zero-command, root cause OPEN after 20 levers): none of
    those 20 touched the waist joint -- they measured `ang_err` and signed heading drift on the ROOT
    and left the one joint that yaws the torso against planted feet unexamined. This probe shows
    base yaw drift roughly doubles (3.9 -> 7.0 deg) when the arm is extended, and that pinning the
    waist recovers most of it (7.0 -> 4.1). That matches MEMORY 154's own conclusion that the
    signature is arm-path-specific rather than a global bias. This lever is therefore also a live
    candidate for that open bug, not only for the IK-reach symptom.

    Verify: `mj_envs/probe/waist_yaw_reach_probe.py`, 2 seeds. Target reach-condition |waist| back
    to the home-condition level (~1 deg, from 4.3-4.8) and base yaw at reach <= the home condition
    (~4 deg, from 6.2-7.0). Guard against regression with the standard deterministic eval
    (`eval_policy.py --home-ratio 0.0`): fell_over, term_rate, dead zone at 0.30 and turn-in-place
    yaw at (0,0,0.2) must all hold vs CosineAR. Tightening a posture anchor is exactly the move that
    can suppress the gait, so `disp` and `track_xy_err` are the ones to watch.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        # resolve_matching_names_values REJECTS overlapping patterns ("Multiple matches for
        # 'waist_joint'"), so the inherited ".*" catch-all is re-expressed as "everything but the
        # waist" via negative lookahead (same idiom as _ARM_PATTERN above). Every other joint keeps
        # its inherited 0.1; the only semantic change is the waist.
        env.rewards["pose"].params["std_standing"] = {r"^(?!waist).*": 0.1, r"waist.*": 0.05}


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNode(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStand
):
    """WaistStand + frozen envs hold a random graph node, not only home. One change: 0.0 -> 0.5.

    WHY, and why this is not another pass at the same lever. WaistStand priced the waist deviation
    harder and won 34% (|waist| at reach 4.34-4.52 -> 2.46-3.34 deg over 3 seeds, band-disjoint,
    every other reward term overlapping). It then plateaued, because a price fights the OUTPUT of a
    behaviour whose CAUSE is still present.

    The cause, measured (`mj_envs/probe/waist_cause_probe.py`, which crosses what the body does
    against what the policy is told, on the deployed CosineAR ckpt):

        cell                  |waist|
        phys=home  obs=home     0.61
        phys=reach obs=reach    4.60   normal reach
        phys=reach obs=home     2.87   body reaches, policy told "arm is home"
        phys=home  obs=reach    5.19   body STILL, policy told "arm reached"

    The last row is the finding: 5.19 deg of waist yaw with the arm physically at home (arm_err
    2.38 deg vs 1.81 at rest) -- MORE than a real reach produces. Roughly two thirds of the effect
    is a response to the 14 numbers in `target_arm_joint_pos`, not to displaced arm mass. WaistStand
    shrinks both components ~34% proportionally, i.e. it squeezes the output without touching this.

    Why the policy learned it: with `frozen_at_graph_node = 0` the arm reference is static IF AND
    ONLY IF it equals home, since every non-frozen env traverses the graph continuously
    (`joint_ref_command.py` `_update_command` reasserts home for frozen envs, and nothing else ever
    holds). So "target != home" perfectly predicts "the arm is slewing", and yawing the waist to
    anticipate that disturbance is CORRECT in distribution. Deploy then holds a far IK pose STATIC
    -- non-home and not moving, a combination absent from training -- and the anticipation fires
    forever because the target never stops being non-home. This also explains why the symptom is
    specific to IK reaching while walking stayed smooth.

    The change: half of frozen envs hold a random graph node instead of home, making "far target,
    holding still" in-distribution. Chose the frozen path rather than adding dwell to the swing path
    because frozen envs are already the mechanism for "arm reference is not moving" -- reusing it
    keeps the diff to one sampled subset and leaves graph traversal untouched.

    Rejected alternatives. Halving waist std again (0.05 -> 0.025): more of the lever that just
    plateaued, and it presses on a response that is correct in distribution. Noise/DR on
    `target_arm_joint_pos`: the arm residual head consumes that same vector, and MEMORY forbids
    weakening `arm_joint_tracking`. Dropping the arm target from the leg head's observation: the
    clean fix in principle, but `Decoupled` names the track_linear_velocity std_z change, not an
    observation split -- both heads read one 120-dim input, so this is architecture work.

    ASSUMED, NOT VERIFIED: that 0.5 is the right mix. Too high and the policy loses slew
    anticipation it genuinely needs; `joint_ref_command.py`'s `stand_min_steps` note records a
    fast-arm ckpt going OOD and FALLING when probed at a slow-arm setting, which is this same axis.
    So the regression guard matters more here than for WaistStand: a policy that never learned to
    anticipate arm swing may walk worse when the arm DOES swing.

    Verify: `waist_cause_probe.py` -- the `phys=home obs=reach` cell is the target, it should fall
    toward the `phys=home obs=home` floor; `waist_yaw_reach_probe.py` for the headline |waist|; then
    `eval_policy.py --home-ratio 0.0` on all 20 reward terms, fell_over, term_rate, disp,
    track_xy_err, read as BANDS against the 2-seed CosineAR baseline (a point comparison already
    produced one false regression call in this campaign).
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.commands["arm_ref"].frozen_at_graph_node = 0.5


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNodeFastGrid(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNode
):
    """FrozenAtNode + earlier grid-curriculum deadline. One change: 0.8 -> 0.45.

    FrozenAtNode won the waist axis outright (|waist| at reach 1.55/1.79 deg vs WaistStand's
    2.46-3.34, p95 ~4.0 vs ~7.3, signed bias still ~0, no falls) and then lost reverse locomotion
    completely: at cmd vx=-0.5 it covers 0.10 m with vx_err 0.53, i.e. actual vx ~ 0, against
    baseline's 5.23 m at vx_err 0.099 -- a 52x gap, both seeds.

    Why, measured. `GridAdaptiveVelocityCommand` unlocks cells by MASTERY (a cell beating
    `master_error_frac` activates its 6 face-neighbors) with `_advance_deadline_stages` as a
    time-based fallback, and `_update_grid_weights` credits a cell ONLY from periods that survived
    without early termination. FrozenAtNode's harder mix (half its frozen envs stand with an
    extended arm) terminates more often, so mastery never cascaded and the run rode the deadline
    for all 15k iterations -- `Metrics/twist/grid_active_cells` 64/148/252/356/468 at
    2.5k/5k/7.5k/10k/12.5k in near-equal increments, versus WaistStand's mastery burst
    64 -> 420 by 5k. Full coverage therefore landed at ~12.5k of 15k, leaving backward commands
    roughly 2.5k iterations of exposure instead of ~7.5k.

    The change: `final_stage_fraction` 0.8 -> 0.45 puts the final shell at ~6.75k, matching the
    coverage timing WaistStand reached on its own. Chose this over simply training to 25k because
    it holds compute fixed and varies ONLY curriculum timing; a longer run changes budget and
    timing together and could not separate them.

    ASSUMED, NOT VERIFIED: that exposure time is the whole story. Stage shells expand by symmetric
    BFS, so vx=+1.0 and vx=-1.0 unlock in the same stage, yet forward WAS learned and backward was
    not -- late activation alone does not explain that asymmetry, so forward is evidently cheaper to
    acquire per unit of exposure. If backward stays dead here with coverage complete from 6.75k,
    exposure is NOT the cause and the mix itself blocks reverse.

    Rejected alternatives. Lowering the mix to 0.25: the eval is identical at `--home-ratio` 1.0 and
    0.0, so the damage is unconditional on arm state and dosage has no mechanism to act through.
    Training to 25k: confounds budget with timing, see above. Relaxing `master_error_frac`: changes
    what "mastered" means for every downstream comparison, breaking the band comparisons this
    campaign is graded on.

    Verify: FIRST `Metrics/twist/grid_active_cells` against WaistStand (must reach 468 by ~7k, else
    the lever did not take); then `eval_policy.py --home-ratio 0.0` reading the vx=-0.5 row's
    `disp_m` and `vx_err` (target: disp -> ~5 m, vx_err -> ~0.1, i.e. baseline-like); then
    `waist_yaw_reach_probe.py` to confirm the waist win survived; then the full 20-term band check.

    RESULT (2026-08-16, 2 seeds, NOT PROMOTED). Curriculum gate passed: 468 cells at iter 6950 vs
    FrozenAtNode's 12650. Tracking recovered -- vx=-0.5 disp_m 0.09-0.37 -> 4.33-4.94 (base
    4.97-5.12), vx_err 0.52-0.58 -> 0.11-0.13; lateral fully back in the baseline band. The
    "asymmetry" above was an artifact: the default eval command set has no vy at all, and a
    17-command sweep shows the damage was radial, hitting vy exactly as hard as -vx. Outer shell
    still short of exposure (vx=-1.0 vx_err 0.64-0.65 vs base 0.44-0.45). Waist win fully retained
    (|waist| 1.59/1.74 deg, p95 4.22). Rejected on GAIT COST: joint_torque 9.05-9.30 vs base
    6.77-6.85 (+35%), foot_impact_velocity 0.017 vs 0.011-0.012 (+45%), joint_power +11%, air_time
    0.378-0.388 vs 0.425-0.438; no falls anywhere and upright/pose/limits in band. Torque and impact
    are the hardware-damaging terms, so ~1.4 deg of waist is not worth them -- keep WaistStand.
    Next lever is the credit rule, not the deadline: `final_stage_fraction` opens cells on a clock
    without restoring mastery-driven expansion, so the outer shells stay under-trained while the
    policy is handed full-range commands. Weight `_update_grid_weights` credit by surviving fraction
    instead of discarding terminated periods, or cut FrozenAtNode's termination pressure at source.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.commands["twist"].final_stage_fraction = 0.45


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNodeStableCredit(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNode
):
    """FrozenAtNode with the grid's mastery-credit veto narrowed to instability. One change.

    FastGrid (above) proved the FrozenAtNode collapse was curriculum timing, then failed on gait:
    forcing the shells open on a clock hands the policy commands it has not mastered, costing +35%
    joint_torque and +45% foot_impact_velocity. So fix WHY mastery stalled instead of overriding it.

    Measured cause. `_update_grid_weights` can only credit a command period that was not cut short,
    because `reset` zeroes the accumulators first. That veto exists to stop a fall from counting as
    mastery -- but on this task the term it actually fires on is `velocity_tracking_failure`. Over
    iters 1k-6k, where the mastery cascade has to happen: FrozenAtNode 0.593 tracking-failure
    terminations per episode vs WaistStand 0.051 (12x), while `illegal_contact` is 0.836 vs 0.862
    and `fell_over` 0.146 vs 0.060. The excess terminations that froze its curriculum were entirely
    the tracking cutoff, not instability.

    So the gate double-counts tracking: a cell must clear a hard error cliff (the termination) AND
    then `master_error_frac`. The cliff adds nothing the mean-error test lacks -- a period bad enough
    to trip it carries a large `mean_err` and fails `master_error_frac` anyway -- but it vetoes every
    period that merely dipped. `stability_only_survivorship=True` narrows the veto to `fell_over` /
    `illegal_contact`, so a tracking-cut period is scored on the steps it did run.

    Why this and not FastGrid's deadline: this restores mastery-DRIVEN expansion, so a shell opens
    when the policy can hold it rather than when the clock says so. That is the mechanism that
    produced WaistStand's clean gait at full coverage, and it is the one FastGrid could not restore.

    Rejected alternatives. Weighting credit by surviving step fraction: redefines `master_error_frac`
    for every already-graded run and breaks this campaign's band comparisons. Dropping the veto
    entirely: a fall mid-period would credit the cell from the good steps before it, which is the
    failure the original rule correctly prevents. Lowering the graph-node mix to 0.25: the eval is
    identical at `--home-ratio` 1.0 and 0.0, so the damage is unconditional on arm state and dosage
    has no mechanism to act through. Stacking FastGrid's 0.45 on top: two changes, and it would mask
    exactly the signal this run exists to read.

    Verify, in order. FIRST `Metrics/twist/grid_active_cells` -- must burst like WaistStand
    (64 -> ~420 by 5k) rather than climb in deadline-sized steps; a clock-shaped curve means mastery
    still is not firing and the veto was not the binding constraint. Then the 17-command full-axis
    sweep (`eval_policy.py --home-ratio 0.0`, negative commands need a leading space), reading
    `disp_m`/`vx_err`/`vy_err` against the base bands in /tmp/axeval/ref_bands.txt -- vy coverage is
    mandatory, the default command set has none and that blindness hid half of FrozenAtNode's damage.
    Then gait: `joint_torque` and `foot_impact_velocity` must land in the baseline band, which is
    where FastGrid failed. Then `waist_yaw_reach_probe.py` for |waist| ~1.6 deg.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        env.commands["twist"].stability_only_survivorship = True


class HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNodeArmWarmup(
    HumanoidRmaVelEstArmFlashSacv2GridGaitInitStartNearZeroTurnInPlaceBankFlatDecoupledCosineARWaistStandFrozenAtNode
):
    """FrozenAtNode with the held-node mix delayed until locomotion exists. One change.

    FrozenAtNode collapsed because the perturbation arrived before the skill it perturbs. Holding a
    far arm pose makes walking cost more while standing costs the same, deepening the low-command
    standing optimum already on record for this grid (memory/grid_low_command_deadzone.md). Measured
    at iter 2500 (`eval_policy.py --home-ratio 0.0`, commands 0.3/0.5): FrozenAtNode disp_m
    0.033-0.092 (it stands still) vs WaistStand 0.572-0.629, with `velocity_tracking_failure`
    0.80-1.00 per episode vs 0.000 and `illegal_contact` equal -- the tracking cutoff firing on a
    robot that declines to walk, not instability. It never masters an initial cell, so the grid
    never cascades. The grid stall was the symptom.

    The change: `frozen_at_graph_node_warmup_steps = 4000 * self.num_steps_per_env`, i.e. mix held
    at 0 for the first 4000 of 15000 iterations -- WaistStand's own cascade completes at ~4000 --
    then on for the remaining 11k. A step, not a ramp; a ramp reintroduces the harder regime during
    exactly the window it must be absent from.

    RESULT (2026-08-16, 2 seeds, PROMOTED over ...WaistStand). Every gate passed.

    Curriculum: vtf 0.00 through iters 2500-4000, `grid_active_cells` 64 -> 398/384 by 4000 (matches
    WaistStand's 402). Switching the mix on at 4000 spikes vtf to 1.08/1.43 at 4500, decaying to
    0.00 by 6000 while cells continue to 456/454 -- a policy that can already walk absorbs the
    perturbation. That spike-and-recover is the mechanism's signature.

    Tracking (17 commands): every row in or better than the baseline band, including the outer shell
    -- vx=-1.0 vx_err 0.429-0.448 vs base 0.443-0.453, disp_m 7.02-7.11 vs 5.46-6.93; vx=+1.0 0.292-
    0.302 vs 0.306-0.307. `track_linear_velocity` 0.758-0.759, best of the line.

    Gait (mean over 17 commands): joint_torque 7.22-7.39 INSIDE the WaistStand band 6.97-8.01,
    air_time 0.439-0.456 in band, upright 0.989, fell_over 0.000 everywhere. Honest cost: three
    terms marginally high -- foot_impact_velocity 0.013-0.015 vs 0.011-0.012, action_rate_l2
    0.526-0.540 vs 0.483-0.494, self_collisions 0.123-0.133 vs 0.094-0.125.

    Waist objective retained despite 4000 fewer iterations of the mix: |waist| at reach 1.54-2.06
    deg vs WaistStand 2.46-3.34 and baseline 4.34-4.52, no overlap with either. ~0.2 deg of that
    range is probe noise (three repeats of the same checkpoint: 1.85/1.89/1.97/2.06), so read any
    single number as +-0.1. waist_cause_probe `phys=home obs=reach` 2.32/1.85 vs baseline 5.19 --
    the observation-driven brace is ~60% removed.
    """

    def configure(self, env, agent) -> None:
        super().configure(env, agent)
        # Mix off for the first 4000 iterations -- the window in which WaistStand's own mastery
        # cascade completes. `<iters> * num_steps_per_env` (the weight_stages form) keeps it a
        # constant iteration count, so a longer run does not push the switch later.
        env.commands["arm_ref"].frozen_at_graph_node_warmup_steps = 4000 * self.num_steps_per_env
