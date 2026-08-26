"""Swappable end-effector (hand) interface for arm asset_zoo modules.

An arm module (``franka_panda``, ``ur5e``) grafts a decoupled hand module onto its
wrist flange. The arm needs only a small, hand-agnostic contract — captured by the
``Hand`` dataclass — so any conforming hand can be passed to an arm factory:

    get_ur5e_robot_cfg(hand=CARTESIAN_HAND)        # default
    get_ur5e_robot_cfg(hand=my_other_hand)        # swap

Reference implementation: ``mj_envs.asset_zoo.cartesian_hand.CARTESIAN_HAND``.

This module also holds the generic quaternion/frame helpers used to seat a hand on
a flange whose +Z is the tool-out axis (the standard manipulator convention).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg


# ═══════════════════════════════════════════════════════════
#  QUATERNION / FRAME HELPERS  (wxyz, via mujoco builtins)
# ═══════════════════════════════════════════════════════════

def _unit(quat: np.ndarray) -> np.ndarray:
    """Normalize a quaternion. MJCF quats may be un-normalized (e.g. ``-1 1 0 0``),
    but the mju_* helpers below assume unit quaternions."""
    quat = np.asarray(quat, dtype=float)
    return quat / np.linalg.norm(quat)


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix from a (possibly un-normalized) wxyz quaternion."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, _unit(quat))
    return mat.reshape(3, 3)


def quat_between_axes(from_axis: np.ndarray, to_axis: np.ndarray) -> np.ndarray:
    """Unit quaternion rotating ``from_axis`` onto ``to_axis`` (shortest arc)."""
    a = np.asarray(from_axis, float) / np.linalg.norm(from_axis)
    b = np.asarray(to_axis, float) / np.linalg.norm(to_axis)
    rotation_axis = np.cross(a, b)
    angle = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if np.linalg.norm(rotation_axis) < 1e-9:  # parallel or anti-parallel
        if a @ b > 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        rotation_axis = np.cross(a, seed)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, rotation_axis, angle)
    return quat


def compose_frames(
    outer_pos: np.ndarray, outer_quat: np.ndarray,
    inner_pos: np.ndarray, inner_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two frames: place ``inner`` expressed within ``outer``.

    result_pos  = outer_pos + R(outer_quat) @ inner_pos
    result_quat = outer_quat * inner_quat
    """
    result_pos = np.asarray(outer_pos, float) + quat_to_mat(outer_quat) @ np.asarray(inner_pos, float)
    result_quat = np.zeros(4)
    mujoco.mju_mulQuat(result_quat, _unit(outer_quat), _unit(inner_quat))
    return result_pos, result_quat


# ═══════════════════════════════════════════════════════════
#  HAND INTERFACE
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Hand:
    """A decoupled end-effector module graftable onto an arm wrist flange.

    Any hand exposing these attributes can be passed to an arm factory via
    ``hand=``. The arm strips its own native actuators, attaches ``load_module()``'s
    root body onto its flange at ``mount_pose(flange_axis)``, adds ``actuators()``,
    and includes ``collision_geom_regex`` in its CollisionCfg.
    """

    name: str
    """Human-readable id (for messages/logging)."""

    root_body: str
    """Hand root body name. Attached under a name prefix when it would clash with
    an arm body (see arm modules' ``*_NAME_PREFIX``)."""

    collision_geom_regex: str
    """Regex matching the hand's collision geoms after the attach prefix is applied."""

    load_module: Callable[[], mujoco.MjSpec]
    """Return the hand MjSpec ready for ``attach_body`` (meshes absolute, scene
    furniture + native actuators stripped, joints/equalities kept)."""

    actuators: Callable[[str], tuple[BuiltinPositionActuatorCfg, ...]]
    """``(name_prefix) -> actuator cfgs`` for the hand's driven joints. The prefix
    must match the one passed to ``attach_body``."""

    action_scale: Callable[[str], dict[str, float]]
    """``(name_prefix) -> {joint_or_tendon: scale}`` mapping a unit policy action to a
    target offset. Hand-specific: a stiff position servo uses a range-based scale
    (``make_range_action_scale``); a compliant/torque-limited one uses the YAM
    effort/stiffness heuristic (``make_action_scale``)."""

    # Mount geometry (config data, hand root-body frame). The arm supplies its own
    # flange tool-out axis to ``mount_pose``; the same hand mounts on different
    # flanges (panda/ur5e +Z, humanoid wrist-roll +X) without per-hand code.
    reach_axis: np.ndarray
    """Hand body axis pointing out of the jaws (aligned to the flange tool-out axis)."""

    mating_face_pos: np.ndarray
    """Hand mating-face center in the root-body frame (lands on the flange origin)."""

    def mount_pose(self, flange_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat) seating the hand on a flange whose tool-out axis is ``flange_axis``.

        Rotates the hand's reach axis onto ``flange_axis``, then translates so the
        mating face lands on the flange origin. Resolved once at graft time.
        """
        quat = quat_between_axes(self.reach_axis, flange_axis)
        pos = -quat_to_mat(quat) @ np.asarray(self.mating_face_pos, float)
        return pos, quat


# ═══════════════════════════════════════════════════════════
#  ACTION SCALE
# ═══════════════════════════════════════════════════════════

DEFAULT_ACTION_SCALE_EFFORT_FRACTION = 0.25  # YAM convention


def make_action_scale(
    actuators: tuple[BuiltinPositionActuatorCfg, ...],
    effort_fraction: float = DEFAULT_ACTION_SCALE_EFFORT_FRACTION,
) -> dict[str, float]:
    """Per-joint action scale ``effort_fraction * effort / stiffness`` (YAM convention).

    Valid for compliant / torque-limited position actuators, where ``effort/stiffness``
    is the deflection that saturates the motor. For a STIFF position servo (high
    stiffness, small saturating deflection) this collapses to a near-zero offset and
    no longer maps the action across the joint travel — use ``make_range_action_scale``.
    """
    scale: dict[str, float] = {}
    for actuator in actuators:
        assert isinstance(actuator, BuiltinPositionActuatorCfg)
        assert actuator.effort_limit is not None
        joint_scale = effort_fraction * actuator.effort_limit / actuator.stiffness
        for joint in actuator.target_names_expr:
            scale[joint] = joint_scale
    return scale


DEFAULT_RANGE_ACTION_FRACTION = 0.5  # |action|=1 -> half the travel each way (full span about midpoint)


def make_range_action_scale(
    joint_ranges: dict[str, tuple[float, float]],
    fraction: float = DEFAULT_RANGE_ACTION_FRACTION,
) -> dict[str, float]:
    """Per-joint action scale ``fraction * (hi - lo)`` from the joint travel.

    For a STIFF position servo, the policy commands a position within the joint range,
    so the scale should track the travel, not the (tiny) motor-saturating deflection.
    ``fraction=0.5`` makes ``|action|=1`` span half the range each way — i.e. the full
    open/close travel about the range midpoint.
    """
    return {joint: fraction * (hi - lo) for joint, (lo, hi) in joint_ranges.items()}
