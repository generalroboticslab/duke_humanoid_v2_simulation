"""Cartesian parallel-jaw hand as a swappable :class:`~mj_envs.asset_zoo.hand.Hand`.

A 4-finger, rack-and-pinion parallel-jaw hand actuated through CARTESIAN slide
DOFs (x/y/z prismatic joints), sourced from the ``duke_v2`` asset. Exposes
``CARTESIAN_HAND`` — pass it (or any other ``Hand``) to an arm factory:

    from mj_envs.asset_zoo.ur5e import get_ur5e_robot_cfg
    from mj_envs.asset_zoo.cartesian_hand import CARTESIAN_HAND
    cfg = get_ur5e_robot_cfg(hand=CARTESIAN_HAND)   # also the default

Single source of truth for everything hand-intrinsic: mounting geometry, the
(uniform stiff position servo) actuator model, and the spec surgery that prepares
it for ``attach_body``. Mount geometry source:
``asset/duke_v2/cartesian_hand/PositionDeter/RELATIVE_POSITION_flange__gripper.md``.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg

from mj_envs.asset_zoo.hand import Hand, make_range_action_scale


# ═══════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════

HAND_DIR = Path(__file__).parents[2] / "asset" / "duke_v2" / "cold" / "cartesian_hand"
HAND_XML = HAND_DIR / "cartesian_hand.xml"
HAND_MESH_DIR = HAND_DIR / "meshes"

HAND_ROOT_BODY = "base"
# Collision hulls are named ``*_col_NN`` in cartesian_hand.xml (may carry an attach prefix).
HAND_COLLISION_GEOM_REGEX = r".*_col_\d+"


# ═══════════════════════════════════════════════════════════
#  MOUNT GEOMETRY  (edit here to re-seat the hand)
# ═══════════════════════════════════════════════════════════

# Jaws reach along the hand base-frame +X axis.
HAND_REACH_AXIS_IN_BASE = np.array([1.0, 0.0, 0.0])

# Mating-face center in the hand base-frame (the −X spigot end; jaws are +X).
HAND_MATING_FACE_POS_IN_BASE = np.array([-0.0534, 0.0, 0.0325])


# ═══════════════════════════════════════════════════════════
#  ACTUATORS  (one shared servo model — uniform stiff position servo)
# ═══════════════════════════════════════════════════════════

# Every DOF is driven by the SAME FEETECH HLS3915M serial-bus position servo through
# the SAME rack-and-pinion, so the actuator model is UNIFORM across all 7 — the per-
# joint kp/kv tiers (3000/800/150, 15/8/1) inherited from the source MJCF had no
# physical basis (one servo doesn't change gain by which joint it drives).
#
# Force limit: rack-and-pinion converts servo stall torque to linear force F = tau/r.
#   tau = 1.4 N.m (HLS3915M stall),  r = 0.018 m (pinion pitch radius)
#   F   = 1.4 / 0.018 = 78 N  (one pinion -> one rack; the mimic-coupled pair's other
#         side mirrors kinematically and adds no force in the model).
HAND_RACK_FORCE_LIMIT = 78.0  # N

# Stiff (but not rigid) position servo: kp chosen so the 78 N cap is reached at a
# ~16 mm deflection (F/kp = 78/5000), i.e. firm hold with some give over the ~55 mm
# jaw travel. kv ~ critical for the moved masses (2*sqrt(kp*m), m ~ 0.003-0.04 kg).
HAND_STIFFNESS = 5000.0
HAND_DAMPING = 20.0

# Driven joint -> (lo, hi) slide range (m) from cartesian_hand.xml. Only these 7 are
# actuated; right_up_y / right_down_y follow via mimic equality. Ranges are the
# single source for both the actuators and the (range-based) action scale.
HAND_DRIVEN_JOINT_RANGES: dict[str, tuple[float, float]] = {
    "bridge_z": (-0.027, 0.030),
    "left_up_y": (-0.020, 0.035),
    "left_down_y": (-0.040, 0.017),
    "left_up_finger_x": (0.0, 0.055),
    "right_up_finger_x": (0.0, 0.055),
    "left_down_finger_x": (0.0, 0.055),
    "right_down_finger_x": (0.0, 0.055),
}


def hand_actuators(name_prefix: str = "") -> tuple[BuiltinPositionActuatorCfg, ...]:
    """BuiltinPositionActuatorCfg for the hand's 7 driven joints (uniform servo).

    Args:
        name_prefix: Prefix ``attach_body`` applies to joint names (must match).
    """
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name_prefix + joint,),
            stiffness=HAND_STIFFNESS,
            damping=HAND_DAMPING,
            effort_limit=HAND_RACK_FORCE_LIMIT,
        )
        for joint in HAND_DRIVEN_JOINT_RANGES
    )


def hand_action_scale(name_prefix: str = "") -> dict[str, float]:
    """Range-based per-joint action scale (stiff position servo: command position
    across the jaw travel, not the YAM effort/stiffness deflection)."""
    ranges = {name_prefix + joint: rng for joint, rng in HAND_DRIVEN_JOINT_RANGES.items()}
    return make_range_action_scale(ranges)


# ═══════════════════════════════════════════════════════════
#  SPEC SURGERY
# ═══════════════════════════════════════════════════════════

def load_hand_module() -> mujoco.MjSpec:
    """Load the hand MjSpec ready for grafting onto an arm.

    Retargets meshes to absolute paths (survive the attach), drops standalone-scene
    furniture, and strips the native <position> actuators (re-added as Builtin via
    ``hand_actuators``). The 9 slide joints (with their XML limits) + 2 mimic
    equalities are KEPT unchanged and ride ``attach_body``.
    """
    hand = mujoco.MjSpec.from_file(str(HAND_XML))
    for mesh in hand.meshes:                  # absolute paths survive attach
        mesh.file = str((HAND_MESH_DIR / mesh.file).resolve())
    for geom in list(hand.worldbody.geoms):   # floor plane (scene furniture)
        hand.delete(geom)
    for light in list(hand.lights):
        hand.delete(light)
    for material in list(hand.materials):
        if material.name == "grid":              # floor finish
            hand.delete(material)
    for texture in list(hand.textures):
        hand.delete(texture)
    for actuator in list(hand.actuators):     # rebuilt as Builtin
        hand.delete(actuator)
    return hand


# ═══════════════════════════════════════════════════════════
#  HAND INSTANCE
# ═══════════════════════════════════════════════════════════

CARTESIAN_HAND = Hand(
    name="cartesian_hand",
    root_body=HAND_ROOT_BODY,
    collision_geom_regex=HAND_COLLISION_GEOM_REGEX,
    load_module=load_hand_module,
    actuators=hand_actuators,
    action_scale=hand_action_scale,
    reach_axis=HAND_REACH_AXIS_IN_BASE,
    mating_face_pos=HAND_MATING_FACE_POS_IN_BASE,
)
