"""Humanoid V2.1 constants."""

import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable, Literal, Optional

import math
import re

import numpy as np
import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab.utils.spec_config import CollisionCfg

# Add mj_envs root + repo root to path to allow script to run standalone
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[3]))  # repo root: cartesian_hand's mj_envs.* imports
from generate_leg_joint_pos import vertical_translate_lower_body_joints
from utils.spec_utils import attach_cube_to_link
from mj_envs.asset_zoo.cartesian_hand import CARTESIAN_HAND
from mj_envs.asset_zoo import parallel_gripper as _parallel_gripper
# Head-camera FOV overlay (shared with g1_constants). FOV_GEOM_GROUP is re-exported:
# callers (record_humanoid, curobo verify scripts) import it from here.
from mj_envs.asset_zoo.fov_frustum import (
    FOV_GEOM_GROUP,
    add_fov_frustum_hull,
    add_fov_frustums,
    add_fov_spotlights,
)

M_PI = 3.14159265358979323846

##
# MJCF and assets.
##


HUMANOID_V21_XML = Path(__file__).parent.parent.parent.parent / "asset" / "duke_v2" / "humanoid_v21" / "humanoid_v21.xml"
# Swap to humanoid_v21_high_res.xml for high-quality rendering (same body/joint/site order, bit-identical
# kinematics/dynamics -- see asset/robot_studio/README.md); not used for training/sim, so left off by default.
# HUMANOID_V21_XML = Path(__file__).parent.parent.parent.parent / "asset" / "duke_v2" / "humanoid_v21" / "humanoid_v21_high_res.xml"

# Outboard seat correction (m): push the decoupled hand this far along the wrist tool-out
# axis. The v3 end_effector_attachment_new flange seats the hand 32 mm too close; the joint
# origin (roll axis) is correct, only the attach location is off. Applied here (v21 attach),
# NOT in parallel_gripper's intrinsic mating_face_pos (mesh-measured gripper geometry, shared
# with g1 whose wrist is unchanged). Shifts BOTH gripper base frame AND its grasp/IK site by
# the same vector so the grasp center stays on the jaws.
EE_MOUNT_EXTRA_OUT_M = 0.047 # for the new wrist



assert HUMANOID_V21_XML.exists()

BASE_BODY_NAME = "base_link"
FEET_SITES = ("left_foot", "right_foot")
_FEET_GEOMS_CORE = r"foot_[LR]_collision\d*"           # no anchors — compose below
FEET_GEOMS_PATTERN = rf"^{_FEET_GEOMS_CORE}$"         # matches foot_L/R_collision[0-4] or foot_L/R_collision
FEET_ILLEGAL_CONTACT_PATTERN = rf"^(?!.*{_FEET_GEOMS_CORE}).*_collision$"  # non-foot collision geoms


##
# Actuator config.
##

# PD control gains: 10Hz natural frequency, 2.0 damping ratio.
NATURAL_FREQ_LOW = 8.0 * 2.0 * M_PI
NATURAL_FREQ_HIGH = 10.0 * 2.0 * M_PI

DAMPING_RATIO = 2.0


@dataclass
class MotorSpec:
    """Motor specs: effort_limit (Nm), reflected inertia (kg⋅m²), and target joints."""
    effort_limit: float
    armature: float
    motor_type: str = ""                    # "RS00" | "RS02" | "RS03" | "RS04" | "RS05" | "RS06"
    joints: tuple[str, ...] | None = None  # regex patterns matching joint names
    
    stiffness_range: tuple[float, float] = (1, 100) # real motor stiffness have fixed range
    damping_range: tuple[float, float] = (0.5, 5) # real motor damping have fixed range

    joint_damping: float | None = 0.05  # passive damping at joint level (independent of actuator)
    joint_friction: float | None = 0.1  # dry friction at joint level (independent of actuator)
    natural_freq: float = NATURAL_FREQ_LOW # natural frequency of the joint
    damping_ratio: float = DAMPING_RATIO # damping ratio of the joint

    kp: float | None = None  # motor position pd control stiffness gain in Nm/rad
    kd: float | None = None  # motor velocity pd control damping gain in Nm/(rad/s)

    @property
    def est_kp(self) -> float:
        raw = self.armature * self.natural_freq**2
        return round(raw, 1)

    @property
    def est_kd(self) -> float:
        # Critical damping: ζ = 1.0
        raw = 2.0 * 1.0 * self.armature * self.natural_freq
        return round(raw, 1)

    @property
    def action_scale(self) -> float:
        """Action scale: 0.5 * effort / stiffness, clamped to [0.2, 2.0]."""
        return min(max(0.5 * self.effort_limit / self.kp, 0.2), 2.0)

    def to_actuator_cfg(
        self,
        delay_min_lag: int = 0,
        delay_max_lag: int = 0,
        delay_update_period: int = 0,
    ) -> BuiltinPositionActuatorCfg:

        return BuiltinPositionActuatorCfg(
            target_names_expr=self.joints,
            stiffness=self.kp,
            damping=self.kd,
            effort_limit=self.effort_limit,
            armature=self.armature,
            frictionloss=self.joint_friction,
            delay_min_lag=delay_min_lag,
            delay_max_lag=delay_max_lag,
            delay_update_period=delay_update_period,
        )
    
MOTOR_TORQUE_LIMIT_SCALE = 0.8 # 80% of specified torque limit for safety margin.

spec_r00 = MotorSpec(motor_type="RS00", effort_limit=14.0* MOTOR_TORQUE_LIMIT_SCALE, armature=0.0007, natural_freq=NATURAL_FREQ_HIGH)
spec_r02 = MotorSpec(motor_type="RS02", effort_limit=17.0* MOTOR_TORQUE_LIMIT_SCALE, armature=0.0042, natural_freq=NATURAL_FREQ_HIGH)
spec_r03 = MotorSpec(motor_type="RS03", effort_limit=60.0* MOTOR_TORQUE_LIMIT_SCALE, armature=0.02,   natural_freq=NATURAL_FREQ_LOW, damping_range=(1, 10))
spec_r04 = MotorSpec(motor_type="RS04", effort_limit=90.0* MOTOR_TORQUE_LIMIT_SCALE, armature=0.04,   natural_freq=NATURAL_FREQ_LOW, damping_range=(1, 10))  # max effort_limit:120, changed to 90 to limit current
spec_r05 = MotorSpec(motor_type="RS05", effort_limit=5.5 * MOTOR_TORQUE_LIMIT_SCALE, armature=0.001,  natural_freq=NATURAL_FREQ_HIGH)
spec_r06 = MotorSpec(motor_type="RS06", effort_limit=36.0* MOTOR_TORQUE_LIMIT_SCALE, armature=0.012,  natural_freq=NATURAL_FREQ_LOW)




# Motor definitions: 27 actuators across 6 motor types.
MOTORS = [
    replace(spec_r03, kp=40, kd=4, joints=("waist_joint",), joint_friction=0.2, joint_damping=0.0),
    replace(spec_r03, kp=40, kd=4, joints=(".*_hip_1_joint",), joint_friction=0.2, joint_damping=0.0),
    replace(spec_r03, kp=40, kd=4, joints=(".*_hip_2_joint",), joint_friction=0.5, joint_damping=0.0),
    replace(spec_r03, kp=40, kd=4, joints=(".*_hip_3_joint",), joint_friction=0.2, joint_damping=0.0),
    replace(spec_r04, kp=60, kd=4, joints=(".*_knee_joint",), joint_friction=0.5, joint_damping=0.0),
    replace(spec_r03, kp=40, kd=4, joints=(".*_ankle_1_joint",), joint_friction=0.2, joint_damping=0.0),
    replace(spec_r06, kp=40, kd=4, joints=(".*_ankle_2_joint",), joint_friction=0.5, joint_damping=0.0),
    replace(spec_r03, kp=40, kd=4, joints=(".*_shoulder_1_joint",), joint_friction=0.5, joint_damping=0.0),
    replace(spec_r06, kp=40, kd=4, joints=(".*_shoulder_2_joint",), joint_friction=0.2, joint_damping=0.0),
    replace(spec_r02, kp=40, kd=4, joints=(".*_shoulder_3_joint",), joint_friction=0.1, joint_damping=0.0),
    replace(spec_r02, kp=30, kd=2, joints=(".*_elbow_joint",), joint_friction=0.1, joint_damping=0.0),
    replace(spec_r02, kp=30, kd=2, joints=(".*_wrist_1_joint",), joint_friction=0.1, joint_damping=0.0),
    replace(spec_r00, kp=15, kd=1, joints=(".*_wrist_2_joint",), joint_friction=0.05, joint_damping=0.0),
    replace(spec_r05, kp=15, kd=1, joints=(".*_wrist_3_joint",), joint_friction=0.05, joint_damping=0.0),
]

_ARM_JOINT_SUFFIX_RE = re.compile(r"^\.\*_(shoulder_\d|elbow|wrist_\d)_joint$")

# Per-arm joint suffixes in kinematic order (shoulder->elbow->wrist), deduced from
# MOTORS' per-joint regex patterns -- MOTORS is the single source of truth for which
# arm joints exist and their order, so callers needing that list (e.g. IK warm-start
# scripts) derive it here instead of maintaining a separate hardcoded copy.
ARM_JOINT_SUFFIXES: tuple[str, ...] = tuple(
    match.group(1)
    for m in MOTORS
    if m.joints and (match := _ARM_JOINT_SUFFIX_RE.match(m.joints[0]))
)

# Ready-to-use per-side joint names (same order as ARM_JOINT_SUFFIXES), so callers
# building an IK joint list don't each re-implement the "left_"/"right_" + "_joint" formatting.
LEFT_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"left_{s}_joint" for s in ARM_JOINT_SUFFIXES)
RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"right_{s}_joint" for s in ARM_JOINT_SUFFIXES)

# Camera gimbal joints (yaw + pitch). Same RS05 motor as wrist_3 (effort 4.4 Nm,
# armature 0.001). kp is intentionally soft (5, not wrist_3's 15): a light gimbal
# needs no wrist stiffness, and soft kp keeps it compliant so it does not fight base
# motion. Joint inertia is armature-dominated (I~1.15e-3); critical kd ~0.15. kd=0.3
# gives zeta~2.0 (overdamped, no overshoot) with ~335ms 2% settle — responsive yet
# damped enough to ride out base motion. Earlier kd=1/0.5 (zeta~6.6/3.3) were too
# sluggish. Requires implicitfast integrator: under Euler the explicit velocity
# damping diverges at this small inertia (kd*dt/I>2).
# Joint names after attach: cam_yaw_<tag>, cam_pitch_<tag> for each module of the rig,
# where <tag> is the module's Mount tag in head_camera_creation.RIGS (an un-tagged
# one-module rig drops the suffix entirely).
def _camera_motors(joint_suffix: str) -> list[MotorSpec]:
    """RS05 gimbal motor pair for one head-camera rig (gains rationale above).

    joint_suffix -- regex appended to "cam_yaw"/"cam_pitch" to match that rig's
    module tags, e.g. "_(left|right)" for the dual rig or "" for the un-tagged
    single. All rigs share identical hardware, so only the name pattern differs.
    """
    return [
        replace(spec_r05, kp=5, kd=0.3, joints=(f"cam_yaw{joint_suffix}",),   joint_friction=0.05, joint_damping=0.0),
        replace(spec_r05, kp=5, kd=0.3, joints=(f"cam_pitch{joint_suffix}",), joint_friction=0.05, joint_damping=0.0),
    ]

CAMERA_MOTORS        = _camera_motors("_(left|right)")         # head_camera="actuated"
CAMERA_MOTORS_SINGLE = _camera_motors("")                      # "actuated_single" (one centered column)
CAMERA_MOTORS_TRIPLE = _camera_motors("_(left|center|right)")  # "actuated_triple"

# Welded head-camera downtilt. Horizontal weld at head height cannot see a table cube at the reach
# pose (~70deg below the lens). Exact G1 official D435-mount inclination keeps the fixed-camera
# phase-3 condition and reach-visible paper baseline physically consistent.
WELDED_CAM_DOWNTILT_RAD = 0.8307767239493009
WELDED_CAM_DOWNTILT_DEG = math.degrees(WELDED_CAM_DOWNTILT_RAD)

def resolve_motor_per_joint(
    mj_model, motors: list[MotorSpec] = MOTORS
) -> list[str]:
    """Return motor_type for each actuator in ctrl-index order (= action-space order).

    Matches each actuator's driven joint name against motors regex patterns.
    Pass motors=MOTORS+CAMERA_MOTORS when the model includes the dual actuated
    head camera, or motors=MOTORS+CAMERA_MOTORS_SINGLE for the single-camera variant.
    Raises ValueError on any unmatched joint — catches config drift early.
    Uses mujoco and re, both already imported; no torch dependency.
    """
    result = []
    for i in range(mj_model.nu):
        jid   = mj_model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        # Strip namespace prefix (e.g. "robot/waist_joint" → "waist_joint").
        jname = jname.split("/")[-1]
        for spec in motors:
            if any(re.fullmatch(pat, jname) for pat in spec.joints):
                result.append(spec.motor_type)
                break
        else:
            raise ValueError(f"No motor spec matched joint: {jname!r}")
    return result


HUMANOID_V21_ACTION_SCALE: dict[str, float] = {
    j: m.action_scale for m in MOTORS for j in m.joints
}


def _apply_joint_properties(spec: mujoco.MjSpec, motors: list[MotorSpec]) -> None:
    """Apply joint_damping and joint_friction from motor specs to matching joints."""
    for joint in spec.joints:
        for motor in motors:
            if any(re.fullmatch(pat, joint.name) for pat in motor.joints):
                if motor.joint_damping is not None:
                    joint.damping = np.array([motor.joint_damping, 0.0, 0.0])
                    # print(f"Applied joint damping {motor.joint_damping} to joint {joint.name}")
                if motor.joint_friction is not None:
                    joint.frictionloss = motor.joint_friction
                    # print(f"Applied joint friction {motor.joint_friction} to joint {joint.name}")
                break  # first match wins


def get_spec(
    head_camera: Literal["none", "welded", "welded_single", "actuated", "actuated_single",
                          "actuated_triple"] = "welded",
    end_effector: Literal["builtin", "welded", "actuated"] = "welded",
    hand: str = "parallel_gripper",   # == DEFAULT_HAND (defined below); a key of HAND_REGISTRY
) -> mujoco.MjSpec:
    """Return the humanoid MjSpec, optionally with a D435 head camera rig.

    head_camera: "none" or a key of HEAD_CAMERA_RIGS, which pairs each rig's
    generated MJCF with its gimbal motors. Every actuated rig contributes 2 DOFs
    and 2 actuators per module.
      "none"            -- bare robot (default); joint/actuator dims unchanged.
      "welded"          -- dual camera geometry attached, gimbal joints/actuators
                            removed; adds mass/inertia/collisions, no extra DOFs.
      "welded_single"   -- single centered camera, gimbal joint/actuator removed;
                            same fixed-downtilt bake as "welded", no extra DOFs.
      "actuated"        -- dual camera with live yaw+pitch joints driven by
                            CAMERA_MOTORS; adds 4 DOFs and 4 actuators
                            (cam_yaw_left, cam_pitch_left, cam_yaw_right, cam_pitch_right).
      "actuated_single" -- single camera centered on the top plate (y=0), live
                            yaw+pitch joints driven by CAMERA_MOTORS_SINGLE; adds
                            2 DOFs and 2 actuators (cam_yaw, cam_pitch).
      "actuated_triple" -- the dual pair pushed out to y=-/+0.130 plus a centered
                            forward module, driven by CAMERA_MOTORS_TRIPLE; adds
                            6 DOFs and 6 actuators (cam_[yaw|pitch]_[left|center|right]).
                            Spacing is set by the +-270 deg yaw sweep, not the parked
                            footprint -- see head_camera_creation.RIGS.

    end_effector: how the hand is provided (decoupled CARTESIAN_HAND vs none).
      "builtin"  -- bare wrist flange, no hand (default; byte-identical to today).
      "welded"   -- decoupled CARTESIAN_HAND module grafted onto each intact wrist
                    flange rigidly (finger DOFs dropped). Flange dynamics + visual are
                    kept; the hand is additive (flange ~0.041 kg + hand mass).
      "actuated" -- like "welded" but keeps the module's finger DOFs + mimics,
                    driven by CARTESIAN_HAND's uniform stiff position servo
                    (actuators added to the articulation per wrist, L_/R_).

    The full camera structure is attached: the two gimbal chains (yaw ->
    pitch) plus the two independent per-side base columns (cam_base_left /
    cam_base_right, each mount + yaw motor = 0.24306 kg; 0.48612 kg total)
    under a massless cam_base frame. base_link does not model this structure,
    so its real mass and geometry are grafted on. Only the static CNC top
    plate stays out (it belongs to base_link). Moving links carry
    humanoid-convention collision capsules named cam_<body>_collision[N].
    """
    spec = mujoco.MjSpec.from_file(str(HUMANOID_V21_XML))
    _apply_joint_properties(spec, MOTORS)
    if end_effector != "builtin":
        choice = HAND_REGISTRY[hand]
        _decouple_end_effector(
            spec, mode=end_effector,
            hands=choice.hands, place_ee_site=choice.place_ee_site,
        )
    if head_camera == "none":
        return spec
    cam_xml, cam_motors = HEAD_CAMERA_RIGS[head_camera]
    cam = mujoco.MjSpec.from_file(str(cam_xml))
    # Camera meshes live outside this repo's meshdir; absolute per-mesh
    # file paths sidestep the single global meshdir after attach.
    for m in cam.meshes:
        m.file = str(CAMERA_DIR / "meshes" / m.file)
    # Always remove XML actuators — defined via CAMERA_MOTORS instead.
    for a in list(cam.actuators):
        cam.delete(a)
    if head_camera in ("welded", "welded_single"):
        # Fixed pair (or single centered module for "welded_single"): yaw 0 (left faces +x, right faces
        # -x -> front+back coverage from one stop; single faces +x only), pitch locked
        # WELDED_CAM_DOWNTILT_DEG down. Bake the tilt into each pitch link's rest quat (about the
        # joint's local-z), then drop the cam joints -> 0-DOF. Actuated variant keeps the live joints.
        # The bake loop is rig-agnostic (matches every body named "pitch_*"), so it works unchanged for
        # the single rig's lone "pitch_link".
        half = WELDED_CAM_DOWNTILT_RAD / 2
        dq = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        # Every module's pitch link, whatever the rig's module count/tags.
        for b in [b for b in cam.bodies if b.name.startswith("pitch_")]:
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, np.asarray(b.quat, dtype=float), dq)
            b.quat = out
        for j in list(cam.joints):
            cam.delete(j)
    # The "base" subtree carries the lower structure as TWO independent
    # per-side columns (cam_base_left / cam_base_right at y=-/+0.065); "base"
    # itself is a massless attach frame. base_link does not model this, so the
    # whole subtree (geometry + mass) is grafted on. Only the static CNC top
    # plate is omitted.
    # head_camera_dual.xml is modeled in base_link's own frame (top-plate top
    # face z=0.412, central hole at x=y=0), so the gimbal chains attach at
    # identity; L/R mirroring is already baked into the model.
    frame = spec.body("base_link").add_frame(pos=(0, 0, 0), quat=(1, 0, 0, 0))
    frame.attach_body(cam.body("base"), "cam_", "")
    add_fov_frustums(spec)
    add_fov_frustum_hull(spec)
    add_fov_spotlights(spec)
    _apply_joint_properties(spec, cam_motors)   # no-op for "welded" (joints deleted)
    return spec


##
# End-effector (hand) module — decoupled CARTESIAN_HAND, grafted onto both wrists.
##

# The decoupled hand is the shared CARTESIAN_HAND (4-finger parallel-jaw rack-and-
# pinion: 9 slide joints, 2 mimic equalities). Same asset, actuators (uniform stiff
# position servo), and mount geometry as the panda/ur5e arms — single source of truth,
# so the BuiltinPositionActuatorCfg is defined ONCE in cartesian_hand.py. One design
# mounts on BOTH wrists, attached once per arm with a per-arm name prefix (L_ / R_) to
# keep the internal left/right jaw names unique. FPS: base/bridge collision were coarse-
# CoACD'd (101->6, 35->6 hulls); rack/finger grasp hulls kept; visual meshes full-res.

# Wrist-roll output flange (end_effector_L/R) tool-out axis. The hand's reach axis
# aligns to it: CARTESIAN_HAND.mount_pose(WRIST_TOOL_OUT_AXIS) lands the mating face on
# the flange origin (== legacy (0.0534, 0, -0.0325) / identity quat; reach axis +X, so
# no rotation). See asset/duke_v2/cartesian_hand/PositionDeter/RELATIVE_POSITION_flange__gripper.md.
WRIST_TOOL_OUT_AXIS = (1.0, 0.0, 0.0)



# wrist-roll flange -> per-arm attach prefix (keeps internal jaw names unique).
EE_FLANGE_PREFIX = {
    "end_effector_L": "L_",
    "end_effector_R": "R_",
}


# ── Selectable decoupled hand ─────────────────────────────────────────────────────
# Which Hand rides each wrist when end_effector != "builtin". `cartesian_hand` is ONE
# design on both wrists; `parallel_gripper` uses its LEFT/RIGHT variants (DISJOINT
# AprilTag id sets) so a detector can tell the two grippers apart. A choice may also carry
# a post-attach EE-site placer (run after each wrist's attach, e.g. move the IK site onto
# the gripper's grasp center) and a compiled-model ctrlrange pin (so the viewer's grip
# slider spans the real jaw travel). Pick one via the `hand=` arg / `--hand` flag.
@dataclass(frozen=True)
class _HandChoice:
    hands: dict                                   # attach prefix ("L_"/"R_") -> Hand
    place_ee_site: Optional[Callable] = None      # (spec, site_name, tool_axis) -> None
    pin_ctrlrange: Optional[Callable] = None      # (compiled_model) -> int


HAND_REGISTRY: dict[str, _HandChoice] = {
    "cartesian_hand": _HandChoice(
        hands={prefix: CARTESIAN_HAND for prefix in EE_FLANGE_PREFIX.values()},
    ),
    "parallel_gripper": _HandChoice(
        # Flange-LESS variants: the wrist end_effector link IS the mount flange (kept intact,
        # ~41 g), so the hand must not bring its own cnc_flange or the flange is double-counted.
        hands={
            "L_": _parallel_gripper.PARALLEL_GRIPPER_LEFT_NO_FLANGE,
            "R_": _parallel_gripper.PARALLEL_GRIPPER_RIGHT_NO_FLANGE,
        },
        place_ee_site=_parallel_gripper.place_ee_site_at_grasp_center,
        pin_ctrlrange=_parallel_gripper.pin_gripper_ctrlrange,
    ),
}
DEFAULT_HAND = "parallel_gripper"   # default hand when end_effector != "builtin"


def _decouple_end_effector(
    spec: mujoco.MjSpec,
    mode: Literal["welded", "actuated"],
    hands: Optional[dict] = None,
    place_ee_site: Optional[Callable] = None,
) -> None:
    """Graft a decoupled hand onto each (intact) wrist flange.

    The end_effector_L/R body is the robot's mounting flange (~0.041 kg; the wrist-roll
    motor mass lives in wrist_3). It is NOT a baked gripper, so its dynamics + visual are
    KEPT — the hand is purely additive, attached as a child of the flange at the hand's
    mount pose, name-prefixed per arm. "welded" drops the module's finger joints + mimic
    equalities for a rigid mount; "actuated" keeps them — driven by the hand's actuators,
    which get_humanoid_v21_robot_cfg adds to the articulation.

    hands: {attach-prefix -> Hand} per wrist (default: CARTESIAN_HAND on both, the legacy
      behavior). Lets a DIFFERENT hand ride each wrist — e.g. parallel_gripper's
      LEFT/RIGHT variants (disjoint AprilTag ids). The mount pose is taken per-hand.
    place_ee_site: optional callable(spec, site_name, tool_axis), run after each wrist's
      attach to reposition that wrist's IK site (e.g. onto the gripper's grasp center).
    """
    if hands is None:
        hands = {prefix: CARTESIAN_HAND for prefix in EE_FLANGE_PREFIX.values()}
    for flange, prefix in EE_FLANGE_PREFIX.items():
        hand = hands[prefix]
        mount_pos, mount_quat = hand.mount_pose(WRIST_TOOL_OUT_AXIS)
        # Outboard seat correction along the tool-out axis; see EE_MOUNT_EXTRA_OUT_M.
        mount_pos = mount_pos + EE_MOUNT_EXTRA_OUT_M * np.asarray(WRIST_TOOL_OUT_AXIS)
        mount_pos, mount_quat = mount_pos.tolist(), mount_quat.tolist()
        module = hand.load_module()
        if mode == "welded":                   # rigid mount: drop the driven DOFs
            for eq in list(module.equalities):  # mimics before the joints they bind
                module.delete(eq)
            for j in list(module.joints):
                module.delete(j)
        frame = spec.body(flange).add_frame(pos=mount_pos, quat=mount_quat)
        frame.attach_body(module.body(hand.root_body), prefix, "")
        if place_ee_site is not None:
            place_ee_site(spec, f"{flange}_site", WRIST_TOOL_OUT_AXIS)
            # Placer recomputes from the gripper's intrinsic mating_face_pos, so it misses the
            # seat correction above; apply the same shift so grasp center tracks the jaws.
            site = spec.site(f"{flange}_site")
            site.pos = (np.asarray(site.pos) + EE_MOUNT_EXTRA_OUT_M * np.asarray(WRIST_TOOL_OUT_AXIS)).tolist()


# End-effector cube: 0.04 m side (half-extent 0.02 m), 0.4 kg, offset −4 cm along x of wrist_3_R.
EE_CUBE_BODY = "wrist_3_R"
EE_CUBE_HALF_EXTENTS = (0.02, 0.02, 0.02)   # 0.04 m full side length
EE_CUBE_POS = (0.04, 0.0, 0.0)
EE_CUBE_MASS = 0.4  # kg


def get_spec_with_ee_cube() -> mujoco.MjSpec:
    """Return the humanoid spec with a 0.04 m / 0.4 kg cube attached to wrist_3_R.

    The cube is offset −4 cm along the wrist x-axis (pos=(-0.04, 0, 0)) and
    participates in contacts (collisions_enabled=True). Use this spec_fn when
    training with an end-effector payload attached to the right wrist.
    """
    spec = get_spec()
    attach_cube_to_link(
        spec,
        body_name=EE_CUBE_BODY,
        half_extents=EE_CUBE_HALF_EXTENTS,
        pos=EE_CUBE_POS,
        mass=EE_CUBE_MASS,
        name="ee_cube",
        collisions_enabled=True,
    )
    return spec


##
# Depth camera (yaw-pitch D435 gimbal) attachment.
##

# Camera module MJCF lives at asset/duke_v2/head_cam (HeadCameraV2, dual
# yaw-pitch D435 gimbal; supersedes the removed v1 twin modules). CAD source
# STEP files are under head_cam/source/.
# Install spec: head_cam/PositionDeter/RELATIVE_POSITION_top_plate__head_cameras.md
CAMERA_DIR = Path(__file__).parents[3] / "asset" / "duke_v2" / "head_cam"
CAMERA_XML = CAMERA_DIR / "head_camera_dual.xml"
CAMERA_XML_SINGLE = CAMERA_DIR / "head_camera_single.xml"
CAMERA_XML_TRIPLE = CAMERA_DIR / "head_camera_triple.xml"

# head_camera mode -> (rig MJCF, gimbal motors). One row per rig, so adding an
# N-module rig to head_camera_creation.RIGS costs one row here and nothing in
# get_spec. "welded" reuses the dual rig with its joints deleted, hence no motors.
# Defined after the CAMERA_XML_* paths; get_spec reads it at call time.
HEAD_CAMERA_RIGS: dict[str, tuple[Path, list[MotorSpec]]] = {
    "welded":          (CAMERA_XML,        []),
    "welded_single":   (CAMERA_XML_SINGLE, []),
    "actuated":        (CAMERA_XML,        CAMERA_MOTORS),
    "actuated_single": (CAMERA_XML_SINGLE, CAMERA_MOTORS_SINGLE),
    "actuated_triple": (CAMERA_XML_TRIPLE, CAMERA_MOTORS_TRIPLE),
}

# head_camera_dual.xml is GENERATED by head_cam/head_camera_creation.py
# (frame-on-joint, via asset/create/robot_builder.py): each link's body frame sits
# on its joint and per-link inertia is diaginertia+quat in the joint-local frame.
# The massless root body "base" carries no pos/quat, so the module still attaches at
# identity onto base_link (top-plate top face z=0.412, central hole x=y=0) with no
# extra mount transform. Yaw axes sit at (0, -/+0.065) -- 130 mm apart -- with the
# right column the left rotated 180 deg about the plate-center z axis (a proper
# rotation; physics via cam_fusion_info.mirror_180z). Both lenses face robot forward
# at qpos=0. Verified in MuJoCo 2026-06-17: yaw anchors land at (0, -/+0.065, 0.52)
# rel base_link, spacing 130.00 mm; geometry world-identical to the prior hand-authored
# XML. (Physics source of truth: cam_fusion_info.py, global frame; do NOT run its
# deprecated --write.)


##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.59),
    joint_pos={
        # Bent knees for stable standing.
        **vertical_translate_lower_body_joints(z_travel=-20, is_forward_bend=True),

        "left_shoulder_1_joint": -0.2,
        "right_shoulder_1_joint": 0.2,
        "left_shoulder_2_joint": -0.2,
        "right_shoulder_2_joint": 0.2,
        "left_elbow_joint": 1.,
        "right_elbow_joint": -1.,
        "left_wrist_1_joint": 0,
        "right_wrist_1_joint": 0,
        "left_wrist_2_joint": 1.0,
        "right_wrist_2_joint": 1.0,

        # parallel_gripper rack_y is left UNSET here on purpose: mjlab (dynamic env),
        # _apply_home_pose (kinematic verify), and cuRobo _lock_joints_from_resolved all resolve
        # an unlisted joint to its compiled qpos0 = 0.0 = jaws half-open (jaw_sep 0.088 m on the
        # [-0.05, 0.0347] rack range). This is the contract pickplace_reach_env documents
        # ("rack_y unset in HOME_KEYFRAME = natural open, q=0.0"). A prior explicit 0.05 override
        # was stale from the OLD [0, 0.0847] range -- on the current range it clamps to 0.0347 =
        # jaw_sep 0.018 m (nearly closed) in BOTH sim and solver. Leave unset to keep one contract.
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype={".*_collision": 1},
    conaffinity={".*_collision": 1},
    condim={FEET_GEOMS_PATTERN: 3, ".*_collision": 1},
    priority={FEET_GEOMS_PATTERN: 1, ".*": 0},
    friction={FEET_GEOMS_PATTERN: (1.0, 0.005, 0.0001)},  # gym rubber (slide, spin, roll)
    solref={FEET_GEOMS_PATTERN: (0.01, 1.0)},  # 10ms correction, critically damped (no bounce)
    solimp={FEET_GEOMS_PATTERN: (0.9, 0.95, 0.005, 0.5, 2)},  # firm with slight give
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=0,
    conaffinity=1,
    condim={FEET_GEOMS_PATTERN: 3, ".*_collision": 1},
    priority={FEET_GEOMS_PATTERN: 1},
    friction={FEET_GEOMS_PATTERN: (1.0, 0.005, 0.0001)},
    solref={FEET_GEOMS_PATTERN: (0.01, 1.0)},
    solimp={FEET_GEOMS_PATTERN: (0.9, 0.95, 0.005, 0.5, 2)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(FEET_GEOMS_PATTERN,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(1.0, 0.005, 0.0001),
    solref=(0.01, 1.0),
    solimp=(0.9, 0.95, 0.005, 0.5, 2),
)

# articulation config
HUMANOID_V21_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(
        replace(cfg, delay_min_lag=1, delay_max_lag=2, delay_update_period=4)
        for cfg in [m.to_actuator_cfg() for m in MOTORS]
    ),
    soft_joint_pos_limit_factor=0.9,  # 10% safety margin from joint limits.
)

# Articulation config for actuated head camera (base robot + 4 camera joints).
HUMANOID_V21_WITH_CAMERA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(
        replace(cfg, delay_min_lag=1, delay_max_lag=2, delay_update_period=4)
        for cfg in [m.to_actuator_cfg() for m in MOTORS + CAMERA_MOTORS]
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_humanoid_v21_robot_cfg(
    head_camera: Literal["none", "welded", "welded_single", "actuated", "actuated_single",
                          "actuated_triple"] = "welded",
    end_effector: Literal["builtin", "welded", "actuated"] = "welded",
    hand: str = "parallel_gripper",   # == DEFAULT_HAND; a key of HAND_REGISTRY
) -> EntityCfg:
    """Get a fresh Humanoid V2.1 robot configuration instance.

    head_camera="none"            -- bare robot.
    head_camera="welded"          -- twin D435 camera geometry, no extra DOFs (default).
    head_camera="welded_single"   -- single centered D435, no extra DOFs.
    head_camera="actuated"        -- live yaw+pitch gimbal joints (4 extra DOFs/actuators).
    head_camera="actuated_single" -- single centered camera, live yaw+pitch joints
                                      (2 extra DOFs/actuators).
    head_camera="actuated_triple" -- dual pair widened to y=-/+0.130 plus a centered
                                      module (6 extra DOFs/actuators). No welded variant.

    end_effector="builtin" -- bare wrist flange, no hand (default).
    end_effector="welded"  -- decoupled hand mounted rigidly on the intact flange (no finger DOFs).
    end_effector="actuated"-- decoupled hand with finger DOFs, driven by its uniform stiff
                              position servo (hand actuators added per wrist, L_/R_).

    hand -- which decoupled hand when end_effector != "builtin": "cartesian_hand" (default;
            byte-identical to before) or "parallel_gripper" (LEFT/RIGHT variants with
            disjoint AprilTag ids, EE site at the grasp center). See HAND_REGISTRY.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    # Body (+ optional camera) actuators, same latency model as the module-level
    # constants. "actuated" appends the shared hand's finger actuators per wrist —
    # the single BuiltinPositionActuatorCfg definition lives in cartesian_hand.py.
    motors = MOTORS + (HEAD_CAMERA_RIGS[head_camera][1] if head_camera != "none" else [])
    actuators = [
        replace(m.to_actuator_cfg(), delay_min_lag=1, delay_max_lag=2, delay_update_period=4)
        for m in motors
    ]
    if end_effector == "actuated":
        ee_hands = HAND_REGISTRY[hand].hands
        ee_actuators = ee_hands["L_"].actuators("L_") + ee_hands["R_"].actuators("R_")
        actuators += [
            replace(a, delay_min_lag=1, delay_max_lag=2, delay_update_period=4)
            for a in ee_actuators
        ]
    articulation = EntityArticulationInfoCfg(
        actuators=tuple(actuators),
        soft_joint_pos_limit_factor=0.9,
    )
    # Collision. FULL_COLLISION only enables the robot's own `.*_collision` geoms; a grafted
    # hand's geoms (e.g. `.*_rack_collision*`) match nothing, so mjlab leaves them contype=0 and the
    # hand collides with NOTHING — not the body (arm-motion self-collision), not even grasped
    # objects. Add a hand CollisionCfg that ENABLES them: contype=1/conaffinity=0 so the hand
    # collides with the body & objects (both conaffinity=1) but the jaws don't self-collide;
    # disable_other_geoms=False so it doesn't undo FULL_COLLISION's body geoms.
    collisions: tuple[CollisionCfg, ...] = (FULL_COLLISION,)
    if end_effector != "builtin":
        hand_regex = HAND_REGISTRY[hand].hands["L_"].collision_geom_regex
        collisions = (FULL_COLLISION, CollisionCfg(
            geom_names_expr=(hand_regex,),
            contype=1, conaffinity=0, condim=3, priority=0,
            friction=(0.5, 0.005, 0.0001), solref=(0.005, 1.0),
            disable_other_geoms=False,
        ))
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=collisions,
        spec_fn=lambda: get_spec(head_camera=head_camera, end_effector=end_effector, hand=hand),
        articulation=articulation,
    )


def get_action_scale(name: str, default: float = 1.0) -> float:
    """Get action scale for a joint name, matching against regex patterns."""
    for pattern, scale in HUMANOID_V21_ACTION_SCALE.items():
        if re.fullmatch(pattern, name):
            return scale
    return default


if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass as _dataclass, field as _field
    import numpy as np
    from mjlab.entity.entity import Entity
    from robot_viewer import GROUND_GEOM_GROUP, launch_robot_viewer

    @_dataclass
    class _Args:
        add_right_ee: bool = False
        """Attach EE cube to wrist_3_R."""
        head_camera: Literal["none", "welded", "actuated", "actuated_single", "actuated_triple"] = "actuated"
        """Head camera mode: "none" | "welded" | "actuated" | "actuated_single" | "actuated_triple"."""
        end_effector: Literal["builtin", "welded", "actuated"] = "actuated"
        """Hand mode: "builtin" (baked) | "welded" (decoupled module, rigid) | "actuated" (finger DOFs)."""
        hand: Literal["cartesian_hand", "parallel_gripper"] = "parallel_gripper"
        """Decoupled hand for --end_effector welded/actuated: "cartesian_hand" | "parallel_gripper"."""
        axis_bodies: list[str] = _field(default_factory=list)
        """Body names whose local coordinate axes to draw (RGB=xyz)."""
        axis_sites: list[str] = _field(default_factory=list)
        """Site names whose local coordinate axes to draw."""
        axis_geoms: list[str] = _field(default_factory=list)
        """Geom names whose local coordinate axes to draw."""

    args = tyro.cli(_Args)

    cfg = get_humanoid_v21_robot_cfg(
        head_camera=args.head_camera, end_effector=args.end_effector, hand=args.hand
    )
    if args.add_right_ee:
        cfg = EntityCfg(
            init_state=cfg.init_state,
            collisions=cfg.collisions,
            spec_fn=get_spec_with_ee_cube,
            articulation=cfg.articulation,
        )

    robot = Entity(cfg)

    # Ground plane on the last valid geom group (5), clear of the FOV overlay on 4.
    # Toggled by the native digit-5 key; robot_viewer turns it on once at startup.
    robot.spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.9, 0.9, 0.9, 1],
        group=GROUND_GEOM_GROUP,
    )

    model = robot.spec.compile()
    model.opt.timestep = 0.005 # 200 hz
    # implicitfast: the camera gimbal's actuator velocity-damping (kd=1) lives in
    # actuator biasprm and is integrated EXPLICITLY under Euler. The cam joint
    # inertia is armature-dominated (~1.15e-3), so the explicit-damping stability
    # limit kd*dt/I < 2 is violated at this 0.005 dt (4.35), causing the gimbal to
    # diverge/shake. implicitfast integrates the velocity term implicitly -> stable
    # at any dt. No effect on the heavier leg joints (well within the Euler limit).
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    # Precompute actuator → qpos index for fast ctrl initialisation on reset.
    actuator_qpos_idx = np.array(
        [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    )

    # Load initial keyframe and zero gravity for clean visual inspection.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = data.qpos[actuator_qpos_idx]
    model.opt.gravity[:] = [0, 0, 0]

    # Viewer-only: mjlab leaves actuators ctrl-unlimited, so MuJoCo's Control
    # sliders default to [-1, 1] and can't drive the camera gimbals over their
    # real range (yaw +/-1.5pi, pitch +/-1.55). Clamp the cam actuators' ctrl to
    # their joint range so the sliders span full travel. No-op without a head
    # camera; affects only this inspection viewer, not training.
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("cam_"):
            jid = model.actuator_trnid[i, 0]
            model.actuator_ctrllimited[i] = 1
            model.actuator_ctrlrange[i] = model.jnt_range[jid]

    # Same idea for a decoupled gripper: pin its grip actuator ctrlrange to the source
    # MJCF command range so the viewer slider spans the real jaw travel (open<->close).
    # No-op for cartesian_hand / builtin (the choice carries no pin).
    _hand_choice = HAND_REGISTRY.get(args.hand)
    if (args.end_effector != "builtin" and _hand_choice is not None
            and _hand_choice.pin_ctrlrange is not None):
        _hand_choice.pin_ctrlrange(model)

    def reset_to_home() -> None:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.ctrl[:] = data.qpos[actuator_qpos_idx]

    foot_geom_names = [
        f"foot_{side}_collision{i}" for side in ("L", "R") for i in range(5)
    ]

    # Coordinate axes to draw. If none requested but a head camera is present,
    # default to sites matching "*_cam_front_center" (one per camera module).
    axis_sites_default: list[str] = []
    if not (args.axis_bodies or args.axis_sites or args.axis_geoms):
        if args.head_camera != "none":
            axis_sites_default = [
                name
                for i in range(model.nsite)
                if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))
                and name.endswith("_cam_front_center")
            ]
        elif (args.end_effector != "builtin" and _hand_choice is not None
                and _hand_choice.place_ee_site is not None):
            # a relocated hand (e.g. parallel_gripper): draw the grasp-center sites.
            axis_sites_default = [
                f"{flange}_site" for flange in EE_FLANGE_PREFIX
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{flange}_site") != -1
            ]
    axis_frames = {
        "body": args.axis_bodies,
        "body_com": [],
        "site": args.axis_sites or axis_sites_default,
        "geom": args.axis_geoms,
    }

    launch_robot_viewer(
        model,
        data,
        keyframes={"R": ("HOME_KEYFRAME", reset_to_home)},
        base_body=BASE_BODY_NAME,
        foot_geom_names=foot_geom_names,
        get_action_scale=get_action_scale,
        axis_frames=axis_frames,
    )
