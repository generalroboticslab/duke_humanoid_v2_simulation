"""G1 legs-only velocity config.

Controls hip/knee/ankle + all 3 waist joints (yaw/pitch/roll); arms held at default pose.
"""
import re

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from tasks.legs_only_task import apply_legs_only_modifiers

# Leg joints: hip (3 DOF), knee, ankle (2 DOF), waist (3 DOF) — waist is part of leg group for stability
LEG_PATTERN = r"(waist_.*_joint|.*_hip_.*|.*_knee_.*|.*_ankle_.*)"
ARM_PATTERN = f"^(?!{LEG_PATTERN}).*"


def G1_LEGS_ONLY_ENV_CFG(
    randomize_arms: bool = True,
    observe_com: bool = True,
    observe_full_joints: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Creates a legs-only configuration for the G1 robot.
    
    Restricts control/observations to legs+waist and randomizes arms at safe poses.
    Waist is treated as part of the leg group for improved stability.
    """
    cfg = unitree_g1_flat_env_cfg()

    # 1. Apply generic legs-only modifications (Warp kernels, fused COM, restricted joints)
    # Waist is included in LEG_PATTERN so it's controlled, observed, and rewarded with legs
    apply_legs_only_modifiers(
        cfg, 
        LEG_PATTERN, 
        ARM_PATTERN, 
        robot_name="g1", 
        randomize_arms=randomize_arms,
        observe_com=observe_com,
        observe_full_joints=observe_full_joints,
    )

    # 2. G1-specific scale refinements
    cfg.actions["joint_pos"].scale = {
        k: v for k, v in G1_ACTION_SCALE.items()
        if re.search(LEG_PATTERN, k)
    }
    cfg.actions["joint_pos_arms"].scale = {
        k: v for k, v in G1_ACTION_SCALE.items()
        if re.search(ARM_PATTERN, k)
    }

    return cfg
