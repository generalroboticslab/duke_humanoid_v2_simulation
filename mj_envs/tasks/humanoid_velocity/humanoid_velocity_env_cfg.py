"""Humanoid V2.1 velocity tracking task environment configuration.

This module defines a complete MDP (Markov Decision Process) for training a humanoid
robot to track velocity commands while maintaining balance and natural gait patterns.

Task Overview:
    The robot learns to:
    - Track desired linear velocity (forward/backward, sideways)
    - Track desired angular velocity (turning)
    - Maintain upright posture and balance
    - Exhibit natural walking/running gaits
    - Handle terrain variations (if using rough terrain)

Training Details:
    - Episode length: 20 seconds
    - Control frequency: 50 Hz (decimation=4, sim dt=0.005s)
    - Observation: 48-dim for policy (IMU, joints, commands)
    - Action: 23-dim joint position targets
    - Rewards: Multi-objective (velocity tracking, posture, energy efficiency)
"""

from __future__ import annotations


from asset_zoo.humanoid_v21.humanoid_v21_constants import (
    HUMANOID_V21_ACTION_SCALE,
    FEET_GEOMS_PATTERN,
    FEET_ILLEGAL_CONTACT_PATTERN,
)
from asset_zoo.humanoid_v21 import get_humanoid_v21_robot_cfg
import math
from pathlib import Path
import sys
from dataclasses import replace
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch, TerrainHeightSensorCfg, ObjRef, GridPatternCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
import mjlab.terrains as terrain_gen
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
# Import velocity-specific mdp functions
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise, GaussianNoiseCfg as Gnoise, NoiseModelWithAdditiveBiasCfg
from mjlab.viewer import ViewerConfig
from mjlab.envs import mdp  # Import base mdp functions

# Add parent directory to path to import local modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from command import PiecewiseLinearVelocityRangeCurriculum
from observation import joint_actuator_force, IMUModelCfg, imu_ang_vel, imu_projected_gravity, joint_pos_abs, joint_vel_abs, GaitPhase
from reward import (
    reward_weight_linear,
    foot_distance,
    actuator_force_reward,
    joint_power_reward,
    foot_impact_velocity,
    foot_flat_orientation,
    feet_clearance_tanh,
    foot_contact_balance,
    foot_phase_contact_match,
    foot_stance_slip_penalty,
)
from event import (
    RootResetFast,
    JointResetFast,
    PushRobotFast,
    foot_height_fast,
    DeferredModelFieldsWrapper,
    PeriodicPhysicsRecompute,
    ee_payload,
    randomize_effort_limits_with_delay,
    EventParamCurriculum,
    randomize_gait_period,
)


# Terrain for 1m humanoid blind walking - moderate difficulty for proprioception-only locomotion
HUMANOID_BLIND_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=25.0, num_rows=10, num_cols=20,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.40),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.12, step_height_range=(0.0, 0.03), step_width=0.25, platform_width=2.0),
        "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
            proportion=0.12, step_height_range=(0.0, 0.03), step_width=0.25, platform_width=2.0),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.09, slope_range=(0.0, 0.14), platform_width=1.5, horizontal_scale=0.2),  # ~8° max
        "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.09, slope_range=(0.0, 0.14), platform_width=1.5, inverted=True, horizontal_scale=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.12, noise_range=(0.01, 0.03), noise_step=0.01, horizontal_scale=0.2),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.06, amplitude_range=(0.0, 0.04), num_waves=3, horizontal_scale=0.2),
    },
)

# Scene Configuration
SCENE_CFG = SceneCfg(
    # Terrain configuration - can switch between flat and rough
    terrain=TerrainEntityCfg(
        terrain_type="generator",  # Use procedural terrain generation
        terrain_generator=HUMANOID_BLIND_TERRAIN_CFG,  # Custom config for 1m blind humanoid
        max_init_terrain_level=3,  # Start on easier terrains for blind walking
    ),
    num_envs=1,  # Number of parallel environments (increase for faster training)
    env_spacing=2.0,  # Spacing between parallel environments (meters)
)

# Viewer configuration for visualization
VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,  # Camera follows robot
    entity_name="robot",
    body_name="base_link",  # Track the robot's base/torso
    distance=3.0,  # Camera distance (meters)
    elevation=-5.0,  # Camera elevation angle (degrees, negative = below)
    azimuth=90.0,  # Camera azimuth angle (degrees)
)

_CONTROL_DECIMATION = 4  # 4 × 5ms physics steps = 20ms control period (50 Hz)
_SIM_DT = 0.005  # 5ms physics timestep
# Shared interval for all deferred DR events + physics_recompute (set_const fires once per period).
# 480 steps × 0.02s/step = 9.6s = 20 rollouts of num_steps_per_env=24.
_RECOMPUTE_INTERVAL_S = 480 * _CONTROL_DECIMATION * _SIM_DT

# Simulation configuration
SIM_CFG = SimulationCfg(
    nconmax=100,  # Increased to handle complex falls on rough terrain
    njmax=900,   # Was 600 (legs-only). parallel_gripper hand adds capsule colliders → init nefc=717 (qpos0). Bumped to 900 with margin.
    contact_sensor_maxmatch=64,   # Reduced from 500 to improve memory footprint and speed
    mujoco=MujocoCfg(
        timestep=_SIM_DT,  # 5ms physics timestep = 200 Hz
        iterations=10,  # Match base G1
        ls_iterations=20,  # Match base G1
        ccd_iterations=150, # Reduced from 500 to improve throughput; bumped from 100 (was hitting cap)
    ),
)



# Contact Sensors
feet_ground_contact_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    # Primary: foot bodies — body mode gives 1 entry per foot regardless of capsule count
    primary=ContactMatch(
        mode="body",
        pattern=r"^foot_[LR]$",
        entity="robot",
    ),
    # Secondary: ground/terrain
    secondary=ContactMatch(
        mode="body",
        pattern="terrain",  # Matches terrain body
    ),
    fields=("found", "force"),  # Track contact existence and forces
    reduce="netforce",  # Reduce multiple contacts to net force
    num_slots=1,
    track_air_time=True,  # Track how long feet are in the air (for gait rewards)
)

# Self-collision sensor
# Penalizes when robot parts collide with each other (unnatural configurations)
self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    # Detect collisions within the robot itself (base_link as reference)
    primary=ContactMatch(
        mode="subtree",
        pattern="base_link",
        entity="robot",
    ),
    secondary=ContactMatch(
        mode="subtree",
        pattern="base_link",
        entity="robot",
    ),
    fields=("found",),  # Only need to know if collision occurred
    reduce="none",
    num_slots=1,
)

# Illegal contact sensor - detects if upper body/arms touch ground
# This catches the failure mode where robot props itself up with arms
illegal_contact_cfg = ContactSensorCfg(
    name="illegal_contact",
    # Detect NON-FOOT parts touching the ground
    # Only foot_L_collision and foot_R_collision should touch ground, everything else is illegal
    primary=ContactMatch(
        mode="geom",
        # Match all collision geoms EXCEPT foot geoms (ankle is illegal too!)
        pattern=FEET_ILLEGAL_CONTACT_PATTERN,
        entity="robot",
    ),
    secondary=ContactMatch(
        mode="body",
        pattern="terrain",  # Ground/terrain
    ),
    fields=("found",),
    reduce="none",  # Use 'none' to get all contacts (check in termination function)
    num_slots=1,
)

# Single downward ray per foot for terrain clearance height measurement
foot_height_scan_cfg = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(
        ObjRef(type="site", name="foot_L_contact", entity="robot"),
        ObjRef(type="site", name="foot_R_contact", entity="robot"),
    ),
    pattern=GridPatternCfg(size=(0.0, 0.0), resolution=0.1),  # 1 ray at foot center
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only
)

# Factory function for foot site config (returns fresh instance to avoid mutation issues)
def foot_site_cfg():
    return SceneEntityCfg("robot", site_names=("foot_L_contact", "foot_R_contact"))

# Factory function for foot body config (for foot orientation reward)
def foot_body_cfg():
    return SceneEntityCfg("robot", body_names=("foot_L", "foot_R"))

# MDP: Actions
actions = { 
    "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),  # All actuators (actuator names now match joint names)
        scale=HUMANOID_V21_ACTION_SCALE,  # Scale [-1,1] to radians
        use_default_offset=True,  # Add to default pose (not absolute positions)
    )
}


# MDP: Commands
commands = {
    "twist": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(2.0, 8.0),  # Resample commands every 2-8 seconds
        rel_standing_envs=0.1,  # 10% of envs get zero velocity (standing)
        rel_heading_envs=0.3,  # 30% of envs use heading control instead of angular vel
        heading_command=True,  # Enable heading (absolute direction) commands
        heading_control_stiffness=0.5,  # PD gain for heading -> angular velocity
        debug_vis=True,  # Visualize velocity commands as arrows
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # Forward/backward velocity range (m/s)
            lin_vel_y=(-1.0, 1.0),  # Sideways velocity range (m/s)
            ang_vel_z=(-0.5, 0.5),  # Turning velocity range (rad/s)
            heading=(-math.pi, math.pi),  # Target heading range (rad)
        ),
    )
}
commands["twist"].viz.z_offset = 0.5  # Post-construction: viz is a nested sub-config not exposed in constructor


# MDP: Observations

def _get_imu_cfg(apply_corruption: bool) -> IMUModelCfg:
    """Create IMU config matching real-world IMU specs (with 5x aggressive margin).

    Real IMU specs (1x baseline):
      - Mounting misalignment: 0.3% (~0.0047 rad = 0.27°)
      - Accel bias instability: 0.05 mg = 5e-5 g (fractional, gravity is normalized)
      - Gyro bias stability: 5.5 °/h = 2.67e-5 rad/s
      - Drift: ~6e-7 rad/s per step @ 50Hz (estimated from heading drift spec)
      - Simulation rate: 50 Hz (20ms steps)
      - No magnetometer (yaw drift is unbounded at integration level)

    Training values include 5x margin for maximum robustness and domain variation.
    """
    return IMUModelCfg(
        max_mount_angle=(0.003, 0.015),      # 5x margin: 0.17-0.86 deg (1x = 0.27°)
        accel_bias_range=(5e-5, 2.5e-4),     # 5x margin (1x = 5e-5 g)
        gyro_bias_range=(2.67e-5, 1.5e-4),   # 5-6x margin (1x = 2.67e-5 rad/s)
        drift_std_range=(5e-6, 5e-5),        # 5x margin on drift
        drift_beta=0.99,                      # ~2s correlation time @ 50Hz (bounded drift)
        delay_min_lag=1,
        delay_max_lag=3,
        delay_update_period=6,
        apply_corruption=apply_corruption,
    )

def _get_policy_obs_terms(imu_cfg: IMUModelCfg) -> dict:
    """Create policy observation terms with given IMU config."""
    return {
        # Base angular velocity in body frame (from simulated IMU)
        # Shape: (3,) - [omega_x, omega_y, omega_z]
        # Delay/rotation/bias/drift handled inside IMUModel; per-term noise added by framework
        "base_ang_vel": ObservationTermCfg(
            func=imu_ang_vel,
            params={"imu_cfg": imu_cfg},
            noise=Gnoise(std=0.1),
        ),

        # Gravity vector projected into body frame (from simulated IMU)
        # Shape: (3,) - encodes orientation without gimbal lock
        # IMUModel replaces RotationBiasNoiseModel with proper Rodrigues rotation
        "projected_gravity": ObservationTermCfg(
            func=imu_projected_gravity,
            params={"imu_cfg": imu_cfg},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),

        # Absolute joint positions (rad). Normalizer handles centering.
        # Shape: (23,) - one per actuated joint
        "joint_pos": ObservationTermCfg(
            func=joint_pos_abs,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.01, n_max=0.01),  # Per-timestep noise
                bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05),  # Per-episode bias
                sample_bias_per_component=True,  # Different bias per joint
            ),
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=6,
        ),

        # Absolute joint velocities (rad/s).
        # Shape: (23,)
        "joint_vel": ObservationTermCfg(
            func=joint_vel_abs,
            noise=Unoise(n_min=-1.0, n_max=1.0),
            delay_min_lag=0,
            delay_max_lag=2,
            delay_update_period=6,
        ),

        # Previous actions (for temporal consistency)
        # Shape: (23,)
        "actions": ObservationTermCfg(
            func=vel_mdp.last_action,
            delay_min_lag=0,
            delay_max_lag=0,  # no delay for previous action
        ),

        # Commanded velocity to track
        # Shape: (3,) - [cmd_vel_x, cmd_vel_y, cmd_ang_z]
        "command": ObservationTermCfg(
            func=vel_mdp.generated_commands,
            params={"command_name": "twist"},
            delay_min_lag=0,
            delay_max_lag=0,  # no delay for previous command
        ),
    }

def _get_critic_obs_terms(policy_obs_terms: dict) -> dict:
    """Create critic observation terms (includes privileged info)."""
    return {
        **policy_obs_terms,  # Include all policy observations

        # Override IMU terms with ground truth for critic (privileged info)
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "projected_gravity": ObservationTermCfg(
            func=vel_mdp.projected_gravity,
        ),

        # Base linear velocity in body frame (from IMU sensor)
        # Shape: (3,) - [vel_x, vel_y, vel_z]
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            # noise=Unoise(n_min=-0.5, n_max=0.5),  # Add noise for robustness
        ),

        # Foot height above ground (from simulation state)
        # Shape: (2,) - [left_foot_height, right_foot_height]
        "foot_height": ObservationTermCfg(
            func=foot_height_fast,
            params={"asset_cfg": foot_site_cfg()},
        ),

        # How long each foot has been in the air
        # Shape: (2,) - helps learn gait timing
        "foot_air_time": ObservationTermCfg(
            func=vel_mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),

        # Binary contact state for each foot
        # Shape: (2,) - [left_contact, right_contact]
        "foot_contact": ObservationTermCfg(
            func=vel_mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),

        # Contact forces on feet
        # Shape: (6,) - [left_fx, left_fy, left_fz, right_fx, right_fy, right_fz]
        "foot_contact_forces": ObservationTermCfg(
            func=vel_mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),

        # added new
        "joint_torque": ObservationTermCfg(
            func=joint_actuator_force,
        ),
    }

# reference_obs_terms = {
#     # Joint torques (actuator forces)
#     # Shape: (num_actuators,)
#     "actuator_force": ObservationTermCfg(
#         func=joint_actuator_force,
#     ),
# }

# Observations dict created dynamically in humanoid_v21_velocity_env_cfg()


# MDP: Rewards
num_steps_per_env = 24
# Curriculum callbacks fire every CURRICULUM_DECIMATION control steps.
# env.common_step_counter increments once per control step (after the physics
# decimation loop), so this is a count of control steps, not physics substeps.
# 2400 control steps = 100 PPO iterations (24 steps/iter) = 48 s of sim time.
# All curriculum signals (reward weights, DR ranges, command ranges) change on
# timescales of hundreds-to-thousands of iterations, so 100-iter granularity is
# more than sufficient and avoids unnecessary recomputation.
CURRICULUM_DECIMATION = 1200

rewards = {}

# --- Primary Objectives ---

rewards["track_linear_velocity"] = RewardTermCfg(
    func=vel_mdp.track_linear_velocity,
    weight=2.0,
    params={
        "command_name": "twist",
        "std": math.sqrt(0.25),
    },
)

rewards["track_angular_velocity"] = RewardTermCfg(
    func=vel_mdp.track_angular_velocity,
    weight=1.5,
    params={
        "command_name": "twist",
        "std": math.sqrt(0.5),
    },
)

# --- Posture & Balance ---

rewards["upright"] = RewardTermCfg(
    func=vel_mdp.upright,
    weight=1.0,
    params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
)

rewards["pose"] = RewardTermCfg(
    func=vel_mdp.variable_posture,
    weight=1.0,
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "command_name": "twist",
        "std_standing": {".*": 0.1},
        "std_walking": {
            r".*hip_1.*": 0.3,
            r".*hip_2.*": 0.15,
            r".*hip_3.*": 0.15,
            r".*knee.*": 0.6,
            r".*ankle_1.*": 0.3,
            r".*ankle_2.*": 0.1,
            r"waist.*": 0.2,
            r".*shoulder.*": 0.1,
            r".*elbow.*": 0.1,
            r".*wrist.*": 0.1,
        },
        "std_running": {
            r".*hip_1.*": 0.4,
            r".*hip_2.*": 0.25,
            r".*hip_3.*": 0.25,
            r".*knee.*": 0.8,
            r".*ankle_1.*": 0.4,
            r".*ankle_2.*": 0.2,
            r"waist.*": 0.3,
            r".*shoulder.*": 0.1,
            r".*elbow.*": 0.1,
            r".*wrist.*": 0.1,
        },
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
    },
)

rewards["body_ang_vel"] = RewardTermCfg(
    func=vel_mdp.body_angular_velocity_penalty,
    weight=-0.01,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
)

rewards["dof_pos_limits"] = RewardTermCfg(
    func=vel_mdp.joint_pos_limits,
    weight=-1.0,
)

rewards["self_collisions"] = RewardTermCfg(
    func=vel_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision"},
)
# --- Smoothness (with curriculum) ---

rewards["action_rate_l2"] = RewardTermCfg(
    func=vel_mdp.action_rate_l2,
    weight=-0.5,
)

# # joint vel l2

# rewards["joint_vel_l2"] = RewardTermCfg(
#     func=vel_mdp.joint_vel_l2,
#     weight=-2e-3,
# )
# reward_curriculum["joint_vel_l2"] = CurriculumTermCfg(
#     func=reward_weight_linear,
#     params={
#         "reward_name": "joint_vel_l2",
#         "decimation": CURRICULUM_DECIMATION,
#         "weight_stages": [
#             {"step": 500 * num_steps_per_env, "weight": -1e-4},
#             {"step": 3000 * num_steps_per_env, "weight": -2e-3},
#         ],
#     },
# )

# # joint acc l2
# rewards["joint_acc_l2"] = RewardTermCfg(
#     func=vel_mdp.joint_acc_l2,
#     weight=-5e-5,
# )
# reward_curriculum["joint_acc_l2"] = CurriculumTermCfg(
#     func=reward_weight_linear,
#     params={
#         "reward_name": "joint_acc_l2",
#         "decimation": CURRICULUM_DECIMATION,
#         "weight_stages": [
#             {"step": 500 * num_steps_per_env, "weight": -1e-7},
#             {"step": 3000 * num_steps_per_env, "weight": -5e-5},
#         ],
#     },
# )

# --- Gait: Air Time (with curriculum) ---

rewards["air_time"] = RewardTermCfg(
    func=vel_mdp.feet_air_time,
    weight=0.1,
    params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "twist",
        "command_threshold": 0.25,  # orig: 0.5 — lowered so stepping is rewarded at slow commands too
    },
)

# --- Gait: Phase Clock Rewards (activated by experiments, weight=0.0 here) ---
# Experiments enable these via configure() by setting weights and curriculum.
# Thresholds (lin_thresh/yaw_thresh) match GaitPhase obs params above.

rewards["foot_phase_contact_match"] = RewardTermCfg(
    func=foot_phase_contact_match,
    weight=0.0,
    params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "lin_thresh": 0.05,
        "yaw_thresh": 0.05,
        "stance_ratio": 0.55,
        "ang_vel_std": 0.5,
        "tilt_thresh": 0.25,
    },
)
rewards["foot_stance_slip_penalty"] = RewardTermCfg(
    func=foot_stance_slip_penalty,
    weight=0.0,
    params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": foot_site_cfg(),
        "command_name": "twist",
        "lin_thresh": 0.05,
        "yaw_thresh": 0.05,
        "stance_ratio": 0.55,
        "ang_vel_std": 0.5,
        "tilt_thresh": 0.25,
    },
)

# --- Gait: Foot Contact Balance (with curriculum) ---

rewards["foot_contact_balance"] = RewardTermCfg(
    func=foot_contact_balance,
    weight=-1.0,
    params={"sensor_name": "feet_ground_contact"},
)

# --- Gait: Foot Movement ---

rewards["foot_clearance"] = RewardTermCfg(
    func=feet_clearance_tanh,
    weight=-1.0,
    params={
        "target_height": 0.1,
        "tanh_scale": 4.0,
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": foot_site_cfg(),
    },
)

rewards["foot_swing_height"] = RewardTermCfg(
    func=vel_mdp.feet_swing_height,
    weight=-0.25,
    params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": "twist",
        "command_threshold": 0.05,
    },
)

rewards["foot_slip"] = RewardTermCfg(
    func=vel_mdp.feet_slip,
    weight=-0.1,
    params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": foot_site_cfg(),
    },
)

# --- Gait: Foot Orientation ---
# Encourage feet to stay flat (horizontal) relative to gravity
# Higher reward when foot xy-plane aligns with world horizontal

rewards["foot_orientation"] = RewardTermCfg(
    func=foot_flat_orientation,
    weight=0.2,
    params={
        "std": 0.3,  # 0.3 rad (~17 deg) deviation = 37% of max reward
        "asset_cfg": foot_body_cfg(),
    },
)

# # --- Gait: Feet Distance (Stance Width Regularization) ---
# # Only active when standing (command < threshold) to avoid interfering with walking
# rewards["foot_distance"] = RewardTermCfg(
#     func=foot_distance,
#     weight=0.2,  # Moderate: ~6% of positive reward budget
#     params={
#         "std": 0.05,  # 5cm deviation = 63% penalty (moderate tolerance)
#         "command_name": "twist",
#         "command_threshold": 0.05,  # Only active when nearly stationary
#         "asset_cfg": foot_site_cfg(),
#     },
# )

# --- Gait: Soft Landing (DISABLED — redundant with foot_impact_velocity) ---
# Soft landing penalizes total contact force magnitude (includes horizontal forces).
# This creates conflicting gradients: the policy sees both landing softness AND foot placement
# costs combined, leading to hesitant footfall. foot_impact_velocity (below) is superior:
# - Uses downward velocity only (v_z) — purely about landing gentleness
# - Has a free zone (0.3 m/s) matching human walking — allows gentle landings cost-free
# - Uses 50× stronger weight (-0.5 vs -1e-4) without causing hesitation
# - Fires every frame in contact, not just first contact
# Result: foot_impact_velocity's -0.006 W.Mean far outweighs soft_landing's -0.004,
# and the policy already learned gentle landings from the better-designed reward.
# Keeping soft_landing adds only noise to the reward table display and no gradient signal.
#
# rewards["soft_landing"] = RewardTermCfg(
#     func=vel_mdp.soft_landing,
#     weight=-1e-4,
#     params={
#         "sensor_name": "feet_ground_contact",
#         "command_name": "twist",
#         "command_threshold": 0.05,
#     },
# )
# reward_curriculum["soft_landing"] = CurriculumTermCfg(
#     func=reward_weight_linear,
#     params={
#         "reward_name": "soft_landing",
#         "decimation": CURRICULUM_DECIMATION,
#         "weight_stages": [
#             {"step": 500 * num_steps_per_env, "weight": -1e-5},
#             {"step": 3000 * num_steps_per_env, "weight": -1e-4},
#         ],
#     },
# )

# --- Gait: Foot Impact Velocity ---
# Penalizes downward foot speed above threshold at ground contact.
# Formula: clamp(v_z_down² - threshold², 0) * in_contact — free zone below 0.3 m/s.
# Stronger weight (-0.5) is safe because gentle landings cost nothing (no hesitance risk).

rewards["foot_impact_velocity"] = RewardTermCfg(
    func=foot_impact_velocity,
    weight=-0.5,
    params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": foot_site_cfg(),
        "threshold_vel": 0.3,   # m/s downward; human walking ~0.1–0.3 m/s at contact
    },
)

# --- Energy Efficiency (with curriculum) ---

rewards["joint_torque"] = RewardTermCfg(
    func=actuator_force_reward,
    weight=-5e-4,
    params={"threshold_ratio": 0.5},
)
rewards["joint_power"] = RewardTermCfg(
    func=joint_power_reward,
    weight=-1e-4,
)


# MDP: Events
events = {
    # Reset base pose at episode start
    "reset_base": EventTermCfg(
        func=RootResetFast,
        mode="reset",  # Triggered on environment reset
        params={
            # Randomize initial position and orientation
            "pose_range": {
                "x": (-0.5, 0.5),  # meters
                "y": (-0.5, 0.5),  # meters
                "z": (0.0, 0.0),   # No Z randomization - robot height is set in init_state
                "yaw": (-3.14, 3.14),  # radians (full circle)
            },
            "velocity_range": {},  # Start with zero velocity
        },
    ),

    # Gait phase clock: allocate buffers and randomize period per episode.
    # Must run on every reset (mode="reset") alongside other reset events.
    "randomize_gait_period": EventTermCfg(
        func=randomize_gait_period,
        mode="reset",
        params={"period_range": (0.55, 0.75)},
    ),

    # Reset joint positions to stable standing pose
    # For a humanoid to stand without falling, knees must be bent
    "reset_robot_joints": EventTermCfg(
        func=JointResetFast,
        mode="reset",
        params={
            # Small randomization around stable pose for robustness
            "position_range": (-0.1, 0.1),  # ±0.1 rad around default
            "velocity_range": (0.0, 0.0),  # Zero velocity
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    ),

    # Apply random pushes during episode (robustness training)
    "push_robot": EventTermCfg(
        func=PushRobotFast,
        mode="interval",  # Periodic event
        interval_range_s=(5.0, 15.0),  # Random interval between 5-15 seconds
        params={
            "x_range": (-0.5, 0.5),  # m/s
            "y_range": (-0.5, 0.5),  # m/s
            "z_range": (-0.2, 0.2),  # m/s
            "roll_range": (-0.5, 0.5),    # rad/s
            "pitch_range": (-0.5, 0.5),   # rad/s
            "yaw_range": (-0.5, 0.5),     # rad/s
        },
    ),

    # Domain randomization: randomize foot friction at episode start
    "foot_friction": EventTermCfg(
        mode="startup",  # Set once: friction is fixed per training run (re-randomizing each episode is too hard)
        func=vel_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                # Target foot collision geometries
                geom_names=(FEET_GEOMS_PATTERN,),
            ),
            "operation": "abs",
            "ranges": (0.3, 1.5),  # Friction coefficient range # TODO CHECK
        },
    ),

    # Domain randomization: randomize PD gains at episode reset
    "pd_gains": EventTermCfg(
        func=vel_mdp.dr.pd_gains,
        mode="reset",  # Randomize on each episode reset
        params={
            "kp_range": (0.9, 1.05),  # Scale Kp by 0.9x to 1.05x
            "kd_range": (0.9, 1.05),  # Scale Kd by 0.9x to 1.05x
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",  # Multiply default gains
        },
    ),

    # Deferred mass DR: all k deferred events write at the same global interval, then
    # physics_recompute fires set_const exactly once. Fixed interval (not randomized) keeps
    # physics constants stable within each PPO rollout (192 steps = 8 × num_steps_per_env).
    "body_mass": EventTermCfg(
        func=DeferredModelFieldsWrapper(vel_mdp.dr.body_mass),
        mode="interval",
        is_global_time=True,
        interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
            "operation": "scale",
            "ranges": (0.85, 1.15),
        },
    ),
    "com_displacement": EventTermCfg(
        func=DeferredModelFieldsWrapper(vel_mdp.dr.body_com_offset),
        mode="interval",
        is_global_time=True,
        interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            "operation": "add",
            "ranges": (-0.01, 0.01),
        },
    ),
    # EE payload: rigidly attaches a point mass (gripper/tool up to ~0.5 kg) at the
    # end_effector_*_site, writing mass + COM + inertia together.
    # Anchored to the sites, not to wrist_3_L/R: the payload must sit DISTAL to
    # left/right_wrist_3_joint (the roll DOF, effort 12), otherwise that actuator never
    # feels the tool it is carrying. Body and mount point are read from the compiled
    # model, so moving the site in the asset moves the payload.
    "ee_payload": EventTermCfg(
        func=DeferredModelFieldsWrapper(ee_payload),
        mode="interval",
        is_global_time=True,
        interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=("end_effector_L_site", "end_effector_R_site")
            ),
            "mass_range": (0.0, 0.0),  # curriculum increases these
            "offset_range": (-0.005, 0.005),
        },
    ),

    # Fires set_const once per interval, after all deferred DR events have written.
    "physics_recompute": EventTermCfg(
        func=PeriodicPhysicsRecompute,
        mode="interval",
        is_global_time=True,
        interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
    ),

    # # Domain randomization: randomize motor strength (90-110% of nominal)
    # "motor_strength": EventTermCfg(
    #     func=randomize_effort_limits_with_delay,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "effort_limit_range": (0.9, 1.10),  # 90-110% torque variation
    #         "operation": "scale",
    #     },
    # ),

    # Domain randomization: joint friction loss (Coulomb friction at DOFs)
    "joint_frictionloss": EventTermCfg(
        func=vel_mdp.dr.dof_frictionloss,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            # Small initial range, curriculum increases
            "ranges": (-0.01, 0.01),
        },
    ),

    # Domain randomization: joint damping (viscous damping at DOFs)
    "joint_damping": EventTermCfg(
        func=vel_mdp.dr.dof_damping,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (0.0, 0.01),  # Positive only to avoid negative damping
        },
    ),
}


# MDP: Terminations
terminations = {
    # Normal episode timeout (max length reached)
    "time_out": TerminationTermCfg(
        func=vel_mdp.time_out,
        time_out=True,  # Flag as successful completion
    ),

    # Robot fell over (orientation too far from upright)
    "fell_over": TerminationTermCfg(
        func=vel_mdp.bad_orientation,
        params={
            "limit_angle": math.radians(70.0),  # FIXED: 70° like G1 (was 45°)
        },
    ),

    # Robot upper body/arms touching ground (failure mode)
    # Catches cases where robot props itself up with arms instead of walking
    "illegal_contact": TerminationTermCfg(
        func=vel_mdp.illegal_contact,
        params={
            "sensor_name": "illegal_contact",
        },
    ),
}

# Environment Configuration Factory
def humanoid_v21_velocity_env_cfg(
    enable_corruption: bool = True,
    enable_reward_curriculum: bool = True,
    plane_flag: bool = True,
    num_steps_per_env: int = 24,
    curriculum_decimation: int = 1200,
) -> ManagerBasedRlEnvCfg:
    """Create the complete environment configuration for humanoid velocity tracking.

    This function assembles all MDP components into a complete environment
    configuration ready for training.

    Args:
        enable_corruption: Whether to apply observation corruption (IMU errors, noise).
            True = training with noise/errors, False = clean evaluation
        enable_reward_curriculum: Whether to use dynamic reward weight schedules.
            True = training mode (weights change over time), False = eval mode (fixed weights)
        num_steps_per_env: Env steps per training iteration (PPO rollout length or FlashSAC
            num_collect_steps). Curriculum stage thresholds are multiplied by this value so
            stage activation is measured in outer iterations, not raw env steps.
        curriculum_decimation: Env steps between curriculum evaluation callbacks.

    Returns:
        ManagerBasedRlEnvCfg: Complete environment configuration with:
            - Scene with robot and terrain
            - Observations (policy and critic)
            - Actions (joint position control)
            - Commands (velocity tracking)
            - Rewards (multi-objective)
            - Events (resets and randomization)
            - Terminations (fall detection)
            - Curriculum (progressive difficulty)

    Usage:
        >>> env_cfg = humanoid_v21_velocity_env_cfg()
        >>> gym.register("Mjlab-HumanoidV21-Velocity", kwargs={"env_cfg_entry_point": env_cfg})
        >>> env = gym.make("Mjlab-HumanoidV21-Velocity")
    """
    # Configure scene with robot entity
    scene = SceneCfg(
        terrain=SCENE_CFG.terrain,
        num_envs=SCENE_CFG.num_envs,
        env_spacing=SCENE_CFG.env_spacing,
        entities={"robot": get_humanoid_v21_robot_cfg()},  # Add robot to scene
        sensors=(feet_ground_contact_cfg, self_collision_cfg,
                 illegal_contact_cfg, foot_height_scan_cfg),  # Add sensors
    )

    # Enable terrain curriculum if using terrain generator
    if scene.terrain is not None and scene.terrain.terrain_generator is not None:
        scene.terrain.terrain_generator.curriculum = True

    # Switch to plane if plane_flag is enabled
    if plane_flag:
        scene.terrain.terrain_type = "plane"
        scene.terrain.terrain_generator = None
        
        # Apply fast settings for flat terrain (matching G1)
        sim_cfg = replace(SIM_CFG)
        sim_cfg.njmax = 900  # >= 717 for humanoid + parallel_gripper (init nefc=717 at qpos0)
        sim_cfg.mujoco.ccd_iterations = 150
        sim_cfg.contact_sensor_maxmatch = 64
        sim_cfg.nconmax = None
    else:
        sim_cfg = SIM_CFG

    # Create observation terms with corruption enabled/disabled
    imu_cfg = _get_imu_cfg(apply_corruption=enable_corruption)
    policy_obs_terms = _get_policy_obs_terms(imu_cfg)
    critic_obs_terms = _get_critic_obs_terms(policy_obs_terms)

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_obs_terms,
            concatenate_terms=True,
            enable_corruption=enable_corruption,  # Framework-level noise
        ),
        "critic": ObservationGroupCfg(
            terms=critic_obs_terms,
            concatenate_terms=True,
            enable_corruption=False,  # No noise for critic (privileged info)
        ),
    }

    # Build curriculum inline — fresh CurriculumTermCfg objects each call; no module-level
    # singletons, no deepcopy needed. num_steps_per_env and curriculum_decimation flow from
    # the experiment class attribute via get_env_cfg().
    reward_curriculum = {}
    reward_curriculum["action_rate_l2"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "action_rate_l2",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 500 * num_steps_per_env, "weight": -0.05},
                {"step": 3000 * num_steps_per_env, "weight": -0.5},
            ],
        },
    )
    reward_curriculum["air_time"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "air_time",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 500 * num_steps_per_env, "weight": 0.5},
                {"step": 2000 * num_steps_per_env, "weight": 0.1},
            ],
        },
    )
    # Stub entry so the curriculum manager registers this term before configure() runs.
    # HumanoidVelocityRMACNNShortEstimator.configure() overrides weight_stages in-place
    # (same pattern as air_time above). Without this stub, configure() adds a NEW key
    # after the manager is built and the curriculum never runs.
    reward_curriculum["foot_swing_height"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "foot_swing_height",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 0,                         "weight": -0.25},
                {"step": 1000 * num_steps_per_env,  "weight": -0.50},
                {"step": 5000 * num_steps_per_env,  "weight": -1.00},
            ],
        },
    )
    reward_curriculum["foot_contact_balance"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "foot_contact_balance",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 500 * num_steps_per_env,  "weight": -0.1},
                {"step": 2000 * num_steps_per_env, "weight": -1.0},
            ],
        },
    )
    reward_curriculum["foot_impact_velocity"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "foot_impact_velocity",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 500 * num_steps_per_env, "weight": -0.001},
                {"step": 3000 * num_steps_per_env, "weight": -0.5},
            ],
        },
    )
    reward_curriculum["joint_torque"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "joint_torque",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 500 * num_steps_per_env, "weight": -1e-5},
                {"step": 3000 * num_steps_per_env, "weight": -5e-4},
            ],
        },
    )
    reward_curriculum["joint_power"] = CurriculumTermCfg(
        func=reward_weight_linear,
        params={
            "reward_name": "joint_power",
            "decimation": curriculum_decimation,
            "weight_stages": [
                {"step": 500 * num_steps_per_env, "weight": -1e-5},
                {"step": 3000 * num_steps_per_env, "weight": -1e-4},
            ],
        },
    )

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=vel_mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        # WARNING: overrides twist.ranges at runtime. Experiments that restrict velocity
        # MUST also override velocity_stages to prevent ranges expanding back out.
        "command_vel": CurriculumTermCfg(
            func=PiecewiseLinearVelocityRangeCurriculum,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 2000,
                        "lin_vel_x": (-0.8, 0.8),
                        "lin_vel_y": (-0.8, 0.8),
                        "ang_vel_z": (-0.5, 0.5),
                    },
                    {
                        "step": 5000 * num_steps_per_env,
                        "lin_vel_x": (-1.0, 1.0),
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-0.7, 0.7),
                    },
                ],
            },
        ),
        **reward_curriculum,
        "domain_randomization": CurriculumTermCfg(
            func=EventParamCurriculum,
            params={
                "decimation": curriculum_decimation,
                "events": {
                    "pd_gains": [
                        {"step": 200 * num_steps_per_env, "kp_range": (0.97, 1.03), "kd_range": (0.97, 1.03)},
                        {"step": 5000 * num_steps_per_env, "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)},
                    ],
                    "push_robot": [
                        {
                            "step": 200 * num_steps_per_env,
                            "x_range": (-0.1, 0.1),
                            "y_range": (-0.1, 0.1),
                            "z_range": (-0.02, 0.02),
                            "roll_range": (-0.02, 0.02),
                            "pitch_range": (-0.02, 0.02),
                            "yaw_range": (-0.02, 0.02),
                        },
                        {
                            "step": 5000 * num_steps_per_env,
                            "x_range": (-0.8, 0.8),
                            "y_range": (-0.8, 0.8),
                            "z_range": (-0.2, 0.2),
                            "roll_range": (-0.8, 0.8),
                            "pitch_range": (-0.8, 0.8),
                            "yaw_range": (-0.8, 0.8),
                        },
                    ],
                    "body_mass": [
                        {"step": 200 * num_steps_per_env, "ranges": (0.95, 1.05)},
                        {"step": 5000 * num_steps_per_env, "ranges": (0.85, 1.15)},
                    ],
                    "com_displacement": [
                        {"step": 200 * num_steps_per_env, "ranges": (-0.001, 0.001)},
                        {"step": 5000 * num_steps_per_env, "ranges": (-0.01, 0.01)},
                    ],
                    "foot_friction": [
                        {"step": 200 * num_steps_per_env, "ranges": (0.6, 1.0)},
                        {"step": 5000 * num_steps_per_env, "ranges": (0.3, 1.2)},
                    ],
                    "joint_frictionloss": [
                        {"step": 200 * num_steps_per_env, "ranges": (-0.01, 0.01)},
                        {"step": 5000 * num_steps_per_env, "ranges": (-0.1, 0.1)},
                    ],
                    "joint_damping": [
                        {"step": 200 * num_steps_per_env, "ranges": (0.0, 0.01)},
                        {"step": 5000 * num_steps_per_env, "ranges": (0.0, 0.1)},
                    ],
                    "ee_payload": [
                        {"step": 200 * num_steps_per_env,  "mass_range": (0.0, 0.0), "offset_range": (-0.005, 0.005)},
                        {"step": 2000 * num_steps_per_env, "mass_range": (0.0, 0.25), "offset_range": (-0.02, 0.02)},
                        {"step": 5000 * num_steps_per_env, "mass_range": (0.0, 0.5), "offset_range": (-0.05, 0.05)},
                    ],
                },
            },
        ),
    }

    # Filter curriculum based on settings
    curriculum_cfg = {
        k: v for k, v in curriculum.items()
        if not (plane_flag and k == "terrain_levels")  # Remove terrain curriculum if using plane
        and not (not enable_reward_curriculum and v.func is reward_weight_linear)  # Remove reward weight curriculum if disabled
    }

    # Create and return complete configuration
    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        rewards=rewards,
        terminations=terminations,
        events=events,
        curriculum=curriculum_cfg,
        sim=sim_cfg,
        viewer=VIEWER_CONFIG,
        decimation=_CONTROL_DECIMATION,
        episode_length_s=20.0,  # 20 second episodes
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Humanoid V2.1 Velocity Tracking Environment Configuration Test")
    print("=" * 70)

    cfg = humanoid_v21_velocity_env_cfg()

    print(f"\n✓ Environment configuration created successfully")
    print(f"\nConfiguration Summary:")
    print(f"  - Scene: {cfg.scene.num_envs} parallel environments")
    print(
        f"  - Terrain: {cfg.scene.terrain.terrain_type if cfg.scene.terrain else 'None'}")
    print(
        f"  - Decimation: {cfg.decimation} (control freq: {1000/cfg.decimation/cfg.sim.mujoco.timestep:.0f} Hz)")
    print(f"  - Episode length: {cfg.episode_length_s}s")
    print(f"  - Observations: {len(cfg.observations)} groups")
    print(f"  - Actions: {len(cfg.actions)} groups")
    print(f"  - Commands: {len(cfg.commands)} types")
    print(f"  - Rewards: {len(cfg.rewards)} terms")
    print(f"  - Events: {len(cfg.events)} events")
    print(f"  - Terminations: {len(cfg.terminations)} conditions")
    print(f"  - Curriculum: {len(cfg.curriculum)} stages")

    print(f"\n" + "=" * 70)
    print("Configuration is valid and ready for training!")
    print("=" * 70)
