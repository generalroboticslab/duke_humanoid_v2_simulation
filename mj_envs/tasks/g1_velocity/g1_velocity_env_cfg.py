"""Custom Unitree G1 velocity configuration with REDUCED STEP SIZE.

This config modifies the hip_pitch variance to reduce step length.
Test ONE parameter at a time for iterative tuning.
"""

import re
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp as vel_mdp  # Import velocity-specific mdp functions

from asset_zoo.g1.g1_constants import get_g1_action_scale, get_g1_robot_cfg
from tasks.humanoid_velocity.event import DeferredModelFieldsWrapper

# Control period 4 * 5ms = 20ms (50 Hz), matching unitree_g1_flat_env_cfg's sim cfg
# (timestep=0.005, decimation=4). DR recompute interval: 480 control steps = 9.6s,
# same cadence used by humanoid_velocity_env_cfg.py's DeferredModelFieldsWrapper events.
_RECOMPUTE_INTERVAL_S = 480 * 4 * 0.005

# Leg joints (incl. waist, for stability) vs. arm joints — matches g1_legs_only_env_cfg.py.
_LEG_PATTERN = r"(waist_.*_joint|.*_hip_.*|.*_knee_.*|.*_ankle_.*)"


def g1_manipulation_env_cfg(
    play: bool = False,
    head_camera: Literal["builtin", "actuated"] = "builtin",
) -> ManagerBasedRlEnvCfg:
    """G1 flat terrain + welded parallel-gripper hand, for arm-manipulation baselines.

    Builds on `unitree_g1_flat_env_cfg()` (already provides G1-native self-collision,
    foot sites, pose-reward std dicts) and layers only what stock G1 velocity lacks:
    a hand/gripper + optional actuated head camera, EE payload/COM domain
    randomization, and a legs-only `joint_pos` action (arm action is built at the
    experiment-class level, once the arm-ref command that `joint_pos_arms` tracks
    against exists).
    """
    cfg = unitree_g1_flat_env_cfg(play=play)

    # Graft welded parallel-gripper hand + selected head-camera mode (stock G1 has neither).
    cfg.scene.entities["robot"] = get_g1_robot_cfg(
        end_effector="welded", hand="parallel_gripper", head_camera=head_camera,
    )

    if head_camera == "actuated":
        # Stock `pose` reward's std_walking/std_running dicts have no entry matching the new
        # cam_yaw/cam_pitch joints (only std_standing has a ".*" catch-all) -- resolve_matching_names_values
        # silently drops unmatched joints, so std_walking/std_running end up shorter than
        # std_standing and the reward's elementwise combination shape-mismatches at runtime.
        # Loose std (matches the wrist entries) since camera joints are already tracked by
        # camera_joint_tracking; pose reward is just loose regularization here.
        cfg.rewards["pose"].params["std_walking"][r"cam_(yaw|pitch)_.*"] = 0.3
        cfg.rewards["pose"].params["std_running"][r"cam_(yaw|pitch)_.*"] = 0.3

    # EE payload/COM domain randomization -- genuinely new, stock G1 has no hand to
    # put noise on. Ported verbatim from humanoid_velocity_env_cfg.py:846-871 (static
    # defaults only; curriculum ramp is out of scope for this baseline).
    ee_body_names = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    cfg.events["ee_payload_mass"] = EventTermCfg(
        func=DeferredModelFieldsWrapper(vel_mdp.dr.body_mass),
        mode="interval",
        is_global_time=True,
        interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ee_body_names),
            "operation": "add",
            "ranges": (0.0, 0.0),
        },
    )
    cfg.events["ee_com_offset"] = EventTermCfg(
        func=DeferredModelFieldsWrapper(vel_mdp.dr.body_com_offset),
        mode="interval",
        is_global_time=True,
        interval_range_s=(_RECOMPUTE_INTERVAL_S, _RECOMPUTE_INTERVAL_S),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ee_body_names),
            "operation": "add",
            "ranges": (-0.005, 0.005),
        },
    )

    # Restrict the stock full-body joint_pos action to legs+waist; arms get their own
    # action (joint_pos_arms, tracking the graph-nav arm_ref command) at the
    # experiment-class level in experiments.py.
    action_scale = get_g1_action_scale(end_effector="welded", hand="parallel_gripper")
    cfg.actions["joint_pos"].actuator_names = (_LEG_PATTERN,)
    cfg.actions["joint_pos"].scale = {
        k: v for k, v in action_scale.items() if re.search(_LEG_PATTERN, k)
    }

    return cfg


def G1_VELOCITY_SMALL_STEPS_ENV_CFG(play: bool = False) -> ManagerBasedRlEnvCfg:
    """G1 flat terrain with REDUCED hip pitch variance for smaller steps.

    CHANGE #1: Reduce hip_pitch std_walking from 0.3 to 0.2
    - This is the MOST IMPACTFUL parameter for step size
    - Hip pitch controls forward/backward leg swing
    - Smaller variance = shorter stride length
    """
    cfg = unitree_g1_flat_env_cfg(play=play)

    # # MODIFICATION: Reduce hip pitch variance for smaller steps 
    # cfg.rewards["pose"].params["std_walking"][r".*hip_pitch.*"] = 0.2  # Original: 0.3. modification make it worse
    
    # fix waist roll and pitch to zero variance
    # Note: std_walking already has specific patterns, so we can update values directly

    # cfg.rewards["pose"].params["std_walking"] = {
    #     # Lower body.
    #     r".*hip_pitch.*": 0.3,
    #     r".*hip_roll.*": 0.15,
    #     r".*hip_yaw.*": 0.15,
    #     r".*knee.*": 0.35,
    #     r".*ankle_pitch.*": 0.25,
    #     r".*ankle_roll.*": 0.1,
    #     # Waist.
    #     r".*waist_yaw.*": 0.2,
    #     r".*waist_roll.*": 0,
    #     r".*waist_pitch.*": 0,
    #     # Arms.
    #     r".*shoulder_pitch.*": 0.01,
    #     r".*shoulder_roll.*": 0.01,
    #     r".*shoulder_yaw.*": 0.01,
    #     r".*elbow.*": 0.01,
    #     r".*wrist.*": 0.01,
    # }
    
    # # std_standing uses ".*" wildcard - must replace with non-overlapping patterns
    # cfg.rewards["pose"].params["std_standing"] = {
    #     # Lower body
    #     r".*hip_pitch.*": 0.05,
    #     r".*hip_roll.*": 0.05,
    #     r".*hip_yaw.*": 0.05,
    #     r".*knee.*": 0.05,
    #     r".*ankle_pitch.*": 0.05,
    #     r".*ankle_roll.*": 0.05,
    #     # Waist - zero variance for roll/pitch
    #     r".*waist_yaw.*": 0.05,
    #     r".*waist_roll.*": 0,
    #     r".*waist_pitch.*": 0,
    #     # Arms
    #     r".*shoulder_pitch.*": 0.01,
    #     r".*shoulder_roll.*": 0.01,
    #     r".*shoulder_yaw.*": 0.01,
    #     r".*elbow.*": 0.01,
    #     r".*wrist.*": 0.01,
    # }

    # cfg.rewards["pose"].params["std_running"] = {
    #     # Lower body.
    #     r".*hip_pitch.*": 0.5,
    #     r".*hip_roll.*": 0.2,
    #     r".*hip_yaw.*": 0.2,
    #     r".*knee.*": 0.6,
    #     r".*ankle_pitch.*": 0.35,
    #     r".*ankle_roll.*": 0.15,
    #     # Waist.
    #     r".*waist_yaw.*": 0.3,
    #     r".*waist_roll.*": 0.0,
    #     r".*waist_pitch.*": 0.0,
    #     # Arms.
    #     r".*shoulder_pitch.*": 0.02,
    #     r".*shoulder_roll.*": 0.02,
    #     r".*shoulder_yaw.*": 0.02,
    #     r".*elbow.*": 0.02,
    #     r".*wrist.*": 0.02,
    # }

    cfg.rewards["air_time"] = RewardTermCfg(
        func=vel_mdp.feet_air_time,
        weight=0.25,  # Disabled by default (tune if needed)
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.05,  # Minimum air time to count (seconds)
            "threshold_max": 0.5,   # Maximum air time before penalty
            "command_name": "twist",
            "command_threshold": 0.5,  # Only apply when moving fast enough
        },
    )
    cfg.rewards["angular_momentum"].weight = 0 # Less penalty on angular momentum, original: -0.02

    return cfg


# For easier import
def g1_small_steps_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Alias for consistency with mjlab naming."""
    return G1_VELOCITY_SMALL_STEPS_ENV_CFG(play=play)
