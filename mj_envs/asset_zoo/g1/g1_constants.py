"""Unitree G1 + decoupled CARTESIAN_HAND — mjlab EntityCfg wrapper.

G1's stock end-effector is a non-articulated ``rubber_hand`` (a visual mesh geom +
one collision capsule baked onto each ``*_wrist_yaw_link``). This module ALWAYS strips
that baked hand, then optionally grafts on the swappable parallel-jaw ``CARTESIAN_HAND``
already used by humanoid_v21 / ur5e / panda, so G1 can train/manipulate with a real
driven gripper.

Mechanism mirrors humanoid_v21 (strip baked hand → graft module) and ur5e (mount on a
flange **site** via ``Hand.mount_pose`` ∘ ``compose_frames``):

    get_g1_robot_cfg()                 # "builtin": stock rubber hand, untouched (default)
    get_g1_robot_cfg("none")           # bare wrist, no hand
    get_g1_robot_cfg("welded")         # CARTESIAN_HAND, rigid mount, finger DOFs dropped
    get_g1_robot_cfg("actuated")       # CARTESIAN_HAND with driven finger DOFs

Four modes:
  "builtin"  -- leave G1's stock rubber hand as-is: visual mesh + collision capsule stay
                baked onto each wrist_yaw_link. No graft, no extra DOFs (default).
  "none"     -- strip the baked rubber hands, graft nothing: bare wrist-yaw flanges.
                No hand mass/geometry, no extra DOFs.
  "welded"   -- module grafted rigidly (finger joints + mimic equalities dropped);
                adds the hand's mass/geometry, no extra DOFs.
  "actuated" -- keeps the module's finger DOFs + mimics, driven by CARTESIAN_HAND's
                uniform stiff position servo (14 actuators added, L_/R_).

Mount: a ``*_hand_mount`` site is added to each wrist_yaw_link in ``get_spec`` (the
stock g1.xml asset is NOT modified); the mount frame is that site pose composed with
the hand's +X tool mount. Re-seat the hand by editing ``G1_HAND_MOUNT_POS``.

The hand root body is named "base" (= many arm roots), so it is attached under a per-arm
``L_``/``R_`` name prefix to keep the left/right jaw names unique.

Note: each ``*_wrist_yaw_link`` inertial is the COMBINED wrist-motor + rubber-hand mass.
Stripping only removes the rubber-hand geoms (not its inertial share), so the grafted
module's mass is a small overcount on top of the link. The link is mostly the wrist
actuator, so the residual rubber-hand mass is left in rather than guessed-apart.

Standalone viewer:
    python -m mj_envs.asset_zoo.g1.g1_constants
"""

# Repo root not on sys.path when this module is imported normally (e.g. via
# `from asset_zoo.g1.g1_constants import ...`, only `mj_envs/` itself is on path) --
# the absolute `mj_envs.asset_zoo.*` imports below need it. Insert unconditionally
# (idempotent), matching ballbot_constants.py's pattern -- the prior `__package__`-gated
# guard only fired for direct script runs, never for normal imports, so any G1 task
# run in a process that hadn't already imported humanoid_v21_constants.py (which does
# this unconditionally) crashed with ModuleNotFoundError: No module named 'mj_envs'.
import pathlib
import sys

_G1_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _G1_REPO_ROOT not in sys.path:
    sys.path.insert(0, _G1_REPO_ROOT)

import math
import re
from dataclasses import replace
from functools import partial
from typing import Literal

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    FULL_COLLISION,
    G1_ACTION_SCALE,
    G1_ARTICULATION,
    HOME_KEYFRAME as _BASE_HOME_KEYFRAME,
    KNEES_BENT_KEYFRAME,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as _base_get_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mj_envs.asset_zoo.cartesian_hand import CARTESIAN_HAND
from mj_envs.asset_zoo.parallel_gripper import (
    GRASP_CENTER_IN_BASE,
    PARALLEL_GRIPPER_LEFT_NO_FLANGE,
    PARALLEL_GRIPPER_RIGHT_NO_FLANGE,
)
from mj_envs.asset_zoo.hand import compose_frames
# Head-camera FOV overlay (shared with humanoid_v21_constants).
from mj_envs.asset_zoo.fov_frustum import (
    FOV_GEOM_GROUP,
    add_fov_frustum_hull,
    add_fov_frustums,
    add_fov_spotlights,
)

HOME_KEYFRAME = replace(
    _BASE_HOME_KEYFRAME,
    joint_pos={
        **_BASE_HOME_KEYFRAME.joint_pos,
        # # Tucked reach-ready arm pose (solved via Mink QP IK at tx=0.25, ty=0.20, tz=-0.05, beta=60.0):
        # "left_shoulder_pitch_joint": 1.0174,
        # "left_shoulder_roll_joint": 0.1203,
        # "left_shoulder_yaw_joint": 0.1266,
        # "left_elbow_joint": -0.9972,
        # "left_wrist_roll_joint": -1.7342,
        # "left_wrist_pitch_joint": 0.1371,
        # "left_wrist_yaw_joint": -0.4261,
        # "right_shoulder_pitch_joint": 1.0108,
        # "right_shoulder_roll_joint": -0.1215,
        # "right_shoulder_yaw_joint": -0.1239,
        # "right_elbow_joint": -0.9962,
        # "right_wrist_roll_joint": 1.7308,
        # "right_wrist_pitch_joint": 0.1353,
        # "right_wrist_yaw_joint": 0.4296,
    },
)

# Re-export the keyframes so callers (e.g. the viewer) import them from one place.
__all__ = [
    "HOME_KEYFRAME",
    "KNEES_BENT_KEYFRAME",
    "G1_BALANCED_KEYFRAME",
    "get_g1_robot_cfg",
    "get_g1_action_scale",
    "get_action_scale",
    "get_spec",
    "full_collision",
]

# Base (pelvis) body, for viewer base-frame tracking.
G1_BASE_BODY = "pelvis"

# G1's base posture stays mjlab's KNEES_BENT_KEYFRAME. Grafted parallel gripper rack_y is left
# UNSET (not pinned): mjlab / cuRobo / kinematic all resolve an unlisted joint to compiled qpos0
# = 0.0 = jaws half-open (jaw_sep 0.088 m on the [-0.05, 0.0347] rack range). A prior explicit
# 0.05 was stale from the OLD [0, 0.0847] range -- on the current range it clamps to 0.0347 =
# nearly closed. Matches humanoid_v21's rack default (also unset); no-op until an actuated hand
# is attached.
G1_BALANCED_KEYFRAME = replace(
    KNEES_BENT_KEYFRAME,
    joint_pos=dict(KNEES_BENT_KEYFRAME.joint_pos),
)

# Unitree's official G1 29-DoF Rev. 1.0 URDF fixes ``d435_link`` to ``torso_link``
# at this pose.  Our D436 replacement shares this chassis mount; only its intrinsics
# are modeled elsewhere.  The official link's +X forward ray maps to this module's
# RGB site +Z optical ray (the convention used by the FOV and visibility code).
_G1_D435_MOUNT_POS = (0.0576235, 0.01753, 0.42987)
_G1_D435_MOUNT_PITCH_RAD = 0.8307767239493009


def _builtin_head_camera_quats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return (RGB-site, MuJoCo-camera) quaternions for official D435 mount pitch.

    MuJoCo RGB sites use +Z as the optical ray, while MuJoCo cameras look along -Z.
    Both rotations encode the same official torso-frame forward/down ray, so rendering,
    FOV frusta, and raycast visibility stay co-registered.
    """
    half = 0.5 * _G1_D435_MOUNT_PITCH_RAD + 0.25 * math.pi
    c = math.cos(half) / math.sqrt(2.0)
    s = math.sin(half) / math.sqrt(2.0)
    return (-c, s, -s, c), (-s, -c, c, s)


# ═══════════════════════════════════════════════════════════
#  FLANGE / MOUNT
# ═══════════════════════════════════════════════════════════

# Terminal wrist body -> (mount-site name, hand name prefix). The mount site is added
# programmatically in get_spec (the stock g1.xml asset is NOT modified); its pose is
# the hand mount frame. Per-arm prefix keeps the hand's internal left/right jaw names
# unique (the same one design on both wrists).
G1_WRIST_FLANGES: dict[str, tuple[str, str]] = {
    "left_wrist_yaw_link": ("left_hand_mount", "L_"),
    "right_wrist_yaw_link": ("right_hand_mount", "R_"),
}

# Mount-site position in the wrist_yaw_link frame (identity orientation). Seat point of
# the cartesian hand along the wrist +X reach axis. Stock rubber hand sat at x=0.0415;
# pulled 0.5 cm closer to the wrist (0.0415 - 0.005). Confirmed against the gripper's
# relative placement in the viewer (--axis-sites left_hand_mount right_hand_mount); re-seat
# by editing this if the wrist-flange-to-hand mating offset changes.
G1_HAND_MOUNT_POS = (0.0365, 0.0, 0.0)

# Wrist reach axis (rubber hand and palm both extend along the link +X). The hand's
# reach axis aligns to it: CARTESIAN_HAND.mount_pose lands the mating face on the site.
G1_FLANGE_TOOL_OUT_AXIS = (1.0, 0.0, 0.0)

# Baked-hand geoms to strip from each wrist_yaw_link: the rubber-hand visual mesh
# (unnamed; matched by mesh name) and the hand collision capsule (matched by name).
_RUBBER_HAND_MESH_SUFFIX = "rubber_hand"
_HAND_COLLISION_SUFFIX = "hand_collision"


# ═══════════════════════════════════════════════════════════
#  SPEC SURGERY
# ═══════════════════════════════════════════════════════════

def _strip_rubber_hand(spec: mujoco.MjSpec, flange_body: str) -> None:
    """Delete G1's baked rubber-hand geoms from a wrist_yaw_link.

    Removes the rubber-hand visual mesh geom + the hand-collision capsule, but KEEPS
    the wrist_yaw_link mesh, its incoming wrist-yaw joint, the inertial, and the palm
    site. The hand's mass/geometry come back via the grafted module.
    """
    body = spec.body(flange_body)
    for geom in list(body.geoms):
        if geom.meshname.endswith(_RUBBER_HAND_MESH_SUFFIX) or geom.name.endswith(
            _HAND_COLLISION_SUFFIX
        ):
            spec.delete(geom)


def _graft_hand(
    spec: mujoco.MjSpec,
    mode: Literal["welded", "actuated"],
    hand_type: Literal["parallel_gripper", "cartesian_hand"] = "parallel_gripper",
) -> None:
    """Strip the baked rubber hands and graft the selected hand onto each wrist flange.

    Per wrist: strip the baked hand, add the mount site (no asset edit), load a fresh
    hand module ("welded" drops its finger joints + mimic equalities for a rigid mount;
    "actuated" keeps them), then attach it at the mount-site frame, name-prefixed per arm.
    """
    hands = {
        "parallel_gripper": {
            "L_": PARALLEL_GRIPPER_LEFT_NO_FLANGE,
            "R_": PARALLEL_GRIPPER_RIGHT_NO_FLANGE,
        },
        "cartesian_hand": {
            "L_": CARTESIAN_HAND,
            "R_": CARTESIAN_HAND,
        }
    }[hand_type]

    for flange, (site_name, prefix) in G1_WRIST_FLANGES.items():
        _strip_rubber_hand(spec, flange)
        body = spec.body(flange)
        site = body.add_site()                  # mount site (also drawn in the viewer)
        site.name = site_name
        site.pos = list(G1_HAND_MOUNT_POS)
        
        hand_obj = hands[prefix]
        hand_pos, hand_quat = hand_obj.mount_pose(G1_FLANGE_TOOL_OUT_AXIS)
        
        module = hand_obj.load_module()
        if mode == "welded":                    # rigid mount: drop the driven DOFs
            for eq in list(module.equalities):  # mimics before the joints they bind
                module.delete(eq)
            for j in list(module.joints):
                module.delete(j)
        pos, quat = compose_frames(
            np.asarray(site.pos), np.asarray(site.quat), hand_pos, hand_quat
        )
        # Grasp-center site: the gripper's grasp point (center of the jaw contact-face
        # cuboid) for IK / reach targeting -- the G1 analog of the humanoid's relocated
        # end_effector_*_site. GRASP_CENTER_IN_BASE is expressed in the gripper base frame,
        # and (pos, quat) IS that base frame in the wrist body, so map the grasp center
        # through it. Named *_hand_grasp to match the *_hand_mount sibling. parallel_gripper
        # only (cartesian_hand has different grasp geometry). Sites are always massless and
        # non-colliding -> zero effect on the trained checkpoint's physics.
        if hand_type == "parallel_gripper":
            offset = np.zeros(3)
            mujoco.mju_rotVecQuat(offset, np.asarray(GRASP_CENTER_IN_BASE, dtype=float),
                                  np.asarray(quat, dtype=float))
            gsite = body.add_site()
            gsite.name = site_name.replace("mount", "grasp")
            gsite.pos = (np.asarray(pos) + offset).tolist()
            gsite.quat = list(quat)
        frame = body.add_frame(pos=pos.tolist(), quat=quat.tolist())
        frame.attach_body(module.body(hand_obj.root_body), prefix, "")


def get_spec(
    end_effector: Literal["builtin", "none", "welded", "actuated"] = "builtin",
    hand: Literal["parallel_gripper", "cartesian_hand"] = "parallel_gripper",
    head_camera: Literal["builtin", "none", "welded", "actuated"] = "builtin",
) -> mujoco.MjSpec:
    """Return the G1 MjSpec.

    "builtin" leaves stock rubber hands untouched; "none" strips them leaving bare
    wrists; "welded"/"actuated" graft the selected hand. Adds ``end_effector_L`` /
    ``end_effector_R`` tool sites on each wrist in every mode for IK targeting.
    """
    spec = _base_get_spec()
    if end_effector == "builtin":
        pass  # stock rubber hand stays on each wrist_yaw_link as baked
    elif end_effector == "none":
        for flange in G1_WRIST_FLANGES:
            _strip_rubber_hand(spec, flange)
    else:
        _graft_hand(spec, mode=end_effector, hand_type=hand)

    # IK tool site per arm: the pose IK targets, kept consistent across hand modes.
    # builtin/none/cartesian_hand -> flange seat (G1_HAND_MOUNT_POS); parallel_gripper
    # -> the grafted grasp center (same pose as the *_hand_grasp site added in graft).
    if end_effector in ("builtin", "none") or hand == "cartesian_hand":
        ee_pos = list(G1_HAND_MOUNT_POS)
        ee_quat = [1.0, 0.0, 0.0, 0.0]
    else:  # parallel_gripper welded/actuated
        offset = np.zeros(3)
        mujoco.mju_rotVecQuat(offset, np.asarray(GRASP_CENTER_IN_BASE, dtype=float),
                              np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float))
        ee_pos = (np.asarray(G1_HAND_MOUNT_POS) + offset).tolist()
        ee_quat = [1.0, 0.0, 0.0, 0.0]
    for flange, _mount in G1_WRIST_FLANGES.items():
        body = spec.body(flange)
        # Side suffix matches IK convention used elsewhere (humanoid_v21-style).
        ee_name = "end_effector_L" if "left" in flange else "end_effector_R"
        site = body.add_site()
        site.name = ee_name
        site.pos = ee_pos
        site.quat = ee_quat

    if head_camera == "builtin":
        torso = spec.body("torso_link")
        rgb_quat, camera_quat = _builtin_head_camera_quats()
        site = torso.add_site()
        site.name = "head_camera_rgb"
        site.pos = list(_G1_D435_MOUNT_POS)
        site.quat = list(rgb_quat)

        cam = torso.add_camera()
        cam.name = "head_camera"
        cam.pos = list(_G1_D435_MOUNT_POS)
        cam.quat = list(camera_quat)
        cam.fovy = 65.0

        add_fov_frustums(spec)
        add_fov_frustum_hull(spec)
        add_fov_spotlights(spec)

    elif head_camera == "none":
        torso = spec.body("torso_link")
        # Delete head visual mesh geom
        for geom in list(torso.geoms):
            if geom.meshname and geom.meshname.endswith("head_link"):
                spec.delete(geom)

    elif head_camera in ("welded", "actuated"):
        from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import (
            CAMERA_DIR,
            CAMERA_MOTORS,
            CAMERA_XML,
            _apply_joint_properties,
        )
        torso = spec.body("torso_link")
        # Delete head visual mesh geom to avoid obstructing the camera view
        for geom in list(torso.geoms):
            if geom.meshname and geom.meshname.endswith("head_link"):
                spec.delete(geom)

        cam = mujoco.MjSpec.from_file(str(CAMERA_XML))
        for m in cam.meshes:
            m.file = str(CAMERA_DIR / "meshes" / m.file)
        for a in list(cam.actuators):
            cam.delete(a)
        if head_camera == "welded":
            for j in list(cam.joints):
                cam.delete(j)

        # Attach the camera gimbals to torso_link.
        # Top face height adjustment: pos=(0, 0, -0.09) relative to torso_link
        # places the camera columns at z=0.43 (head center).
        frame = torso.add_frame(pos=(0, 0, -0.09), quat=(1, 0, 0, 0))
        frame.attach_body(cam.body("base"), "cam_", "")

        add_fov_frustums(spec)
        add_fov_frustum_hull(spec)
        add_fov_spotlights(spec)

        if head_camera == "actuated":
            _apply_joint_properties(spec, CAMERA_MOTORS)

    return spec


# ═══════════════════════════════════════════════════════════
#  COLLISION CONFIG
# ═══════════════════════════════════════════════════════════

def full_collision(
    hand_type: Literal["parallel_gripper", "cartesian_hand"] = "parallel_gripper",
) -> CollisionCfg:
    """G1 FULL_COLLISION extended with the selected hand's collision hulls.

    G1's ``.*_collision`` regex does not match the hand hulls (``*_col_NN``), so the
    hand's ``collision_geom_regex`` is appended and given condim=3 (frictional grasp
    contacts), leaving the base feet/self-collision config untouched.
    """
    hand_obj = {
        "parallel_gripper": PARALLEL_GRIPPER_LEFT_NO_FLANGE,
        "cartesian_hand": CARTESIAN_HAND,
    }[hand_type]
    
    conaffinity = {hand_obj.collision_geom_regex: 0}
    if isinstance(FULL_COLLISION.conaffinity, dict):
        conaffinity.update(FULL_COLLISION.conaffinity)
    else:
        for expr in FULL_COLLISION.geom_names_expr:
            conaffinity[expr] = FULL_COLLISION.conaffinity

    return replace(
        FULL_COLLISION,
        geom_names_expr=(hand_obj.collision_geom_regex, *FULL_COLLISION.geom_names_expr),
        conaffinity=conaffinity,
        condim={hand_obj.collision_geom_regex: 3, **FULL_COLLISION.condim},
    )


# ═══════════════════════════════════════════════════════════
#  PUBLIC FACTORY
# ═══════════════════════════════════════════════════════════

def get_g1_robot_cfg(
    end_effector: Literal["builtin", "none", "welded", "actuated"] = "builtin",
    hand: Literal["parallel_gripper", "cartesian_hand"] = "parallel_gripper",
    head_camera: Literal["builtin", "none", "welded", "actuated"] = "builtin",
) -> EntityCfg:
    """Return EntityCfg for G1.

    end_effector="builtin"  -- stock rubber hand stays baked on each wrist (default).
    end_effector="none"     -- bare wrist flanges, no hand; base articulation + base collision.
    end_effector="welded"   -- selected hand rigid mount, no extra DOFs (base articulation).
    end_effector="actuated" -- selected hand with driven finger DOFs; appends the hand's
                               actuators (L_/R_) to the G1 articulation.
    head_camera="builtin"   -- keep original G1 head mesh, no extra camera/FOV (default).
    head_camera="none"      -- delete head visual mesh, no camera/FOV.
    head_camera="welded"    -- add welded camera gimbals and FOV frustums.
    head_camera="actuated"  -- add actuated camera gimbals and FOV frustums.

    Returns a new EntityCfg instance each time to avoid mutation issues when the
    config is shared across multiple places.
    """
    hands = {
        "parallel_gripper": {
            "L_": PARALLEL_GRIPPER_LEFT_NO_FLANGE,
            "R_": PARALLEL_GRIPPER_RIGHT_NO_FLANGE,
        },
        "cartesian_hand": {
            "L_": CARTESIAN_HAND,
            "R_": CARTESIAN_HAND,
        }
    }[hand]

    actuators = list(G1_ARTICULATION.actuators)
    if end_effector == "actuated":
        actuators += list(hands["L_"].actuators("L_")) + list(hands["R_"].actuators("R_"))
    if head_camera == "actuated":
        from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import CAMERA_MOTORS
        actuators += [
            replace(m.to_actuator_cfg(), delay_min_lag=1, delay_max_lag=2, delay_update_period=4)
            for m in CAMERA_MOTORS
        ]

    articulation = EntityArticulationInfoCfg(
        actuators=tuple(actuators),
        soft_joint_pos_limit_factor=G1_ARTICULATION.soft_joint_pos_limit_factor,
    )
    # Hand collision hulls only exist when a hand is grafted.
    collisions = (FULL_COLLISION,) if end_effector in ("builtin", "none") else (full_collision(hand),)
    return EntityCfg(
        init_state=G1_BALANCED_KEYFRAME,
        collisions=collisions,
        spec_fn=partial(get_spec, end_effector, hand, head_camera),
        articulation=articulation,
    )


def get_g1_action_scale(
    end_effector: Literal["builtin", "none", "welded", "actuated"] = "none",
    hand: Literal["parallel_gripper", "cartesian_hand"] = "parallel_gripper",
) -> dict[str, float]:
    """Action scale for G1 (+ hand when actuated): base G1 scale plus the hand's own
    range-based scale per arm prefix."""
    hands = {
        "parallel_gripper": {
            "L_": PARALLEL_GRIPPER_LEFT_NO_FLANGE,
            "R_": PARALLEL_GRIPPER_RIGHT_NO_FLANGE,
        },
        "cartesian_hand": {
            "L_": CARTESIAN_HAND,
            "R_": CARTESIAN_HAND,
        }
    }[hand]

    scale = dict(G1_ACTION_SCALE)
    if end_effector == "actuated":
        scale.update(hands["L_"].action_scale("L_"))
        scale.update(hands["R_"].action_scale("R_"))
    return scale


def get_action_scale(name: str, default: float = 1.0) -> float:
    """Action scale for a joint name by regex match (base G1 + actuated hand DOFs)."""
    # Check parallel_gripper first
    for pattern, scale in get_g1_action_scale("actuated", "parallel_gripper").items():
        if re.fullmatch(pattern, name):
            return scale
    # Check cartesian_hand
    for pattern, scale in get_g1_action_scale("actuated", "cartesian_hand").items():
        if re.fullmatch(pattern, name):
            return scale
    return default


# ═══════════════════════════════════════════════════════════
#  STANDALONE VIEWER
# ═══════════════════════════════════════════════════════════
#
# Run (from repo root, project python):
#   PY=/home/grl/repo/micromamba/envs/py312/bin/python
#   $PY -m mj_envs.asset_zoo.g1.g1_constants                       # stock rubber hand (default)
#   $PY -m mj_envs.asset_zoo.g1.g1_constants --hand none           # bare wrist
#   $PY -m mj_envs.asset_zoo.g1.g1_constants --hand parallel_gripper
#   $PY -m mj_envs.asset_zoo.g1.g1_constants --hand cartesian_hand \
#       --axis-sites left_hand_mount right_hand_mount              # + draw mount axes
# Direct path form also works ($PY mj_envs/asset_zoo/g1/g1_constants.py ...).
# Needs an X display (interactive viewer; MUJOCO_GL=egl won't help over headless SSH).
#
# Keyboard controls (robot_viewer): R reset KNEES_BENT, H reset HOME, G gravity,
# C/V/T toggle collision/visual/ground, Space pause.

if __name__ == "__main__":
    import pathlib
    import sys
    from dataclasses import dataclass as _dataclass
    from dataclasses import field as _field

    import tyro
    from mjlab.entity.entity import Entity

    # robot_viewer is the shared standalone-viewer helper living in humanoid_v21.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "humanoid_v21"))
    from robot_viewer import GROUND_GEOM_GROUP, launch_robot_viewer

    @_dataclass
    class _Args:
        hand: Literal["builtin", "none", "parallel_gripper", "cartesian_hand"] = "builtin"
        """Hand preset: "builtin" (stock rubber hand, default) | "none" (bare wrist) | "parallel_gripper" (welded gripper) | "cartesian_hand" (welded cartesian hand)."""
        head_camera: Literal["builtin", "none", "welded", "actuated"] = "builtin"
        """Head camera mode: "builtin" | "none" | "welded" | "actuated"."""
        axis_bodies: list[str] = _field(default_factory=list)
        """Body names whose local coordinate axes to draw (RGB=xyz)."""
        axis_sites: list[str] = _field(default_factory=list)
        """Site names whose local axes to draw (e.g. left_hand_mount to check the graft)."""
        axis_geoms: list[str] = _field(default_factory=list)
        """Geom names whose local axes to draw."""

    args = tyro.cli(_Args)

    # Map the single --hand flag onto (end_effector, hand_type): builtin/none are
    # direct mode values; parallel_gripper/cartesian_hand imply welded + that gripper.
    if args.hand in ("builtin", "none"):
        end_effector, hand_type = args.hand, "parallel_gripper"
    else:
        end_effector, hand_type = "welded", args.hand

    robot = Entity(get_g1_robot_cfg(end_effector=end_effector, hand=hand_type, head_camera=args.head_camera))

    # Ground plane on the last valid geom group (5), clear of the FOV overlay on 4.
    # Toggled by the native digit-5 key; robot_viewer turns it on once at startup.
    robot.spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.9, 0.9, 0.9, 1],
        group=GROUND_GEOM_GROUP,
    )

    model = robot.spec.compile()
    # Stiff gripper position servos on small finger inertias diverge under explicit
    # Euler velocity damping; implicitfast integrates the damping implicitly -> stable.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    # Precompute actuator -> qpos index for fast ctrl init on reset.
    actuator_qpos_idx = np.array(
        [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    )

    # KNEES_BENT keyframe (keyframe 0) + zero gravity for clean inspection.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = data.qpos[actuator_qpos_idx]
    model.opt.gravity[:] = [0, 0, 0]

    def _apply_keyframe(kf) -> None:
        """Reset to an EntityCfg keyframe (joint_pos pattern->value, pos[2]) and hold ctrl."""
        mujoco.mj_resetData(model, data)
        for pattern, value in kf.joint_pos.items():
            for jid in range(model.njnt):
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                if re.fullmatch(pattern, jname):
                    data.qpos[model.jnt_qposadr[jid]] = value
        data.qpos[2] = kf.pos[2]
        
        data.ctrl[:] = data.qpos[actuator_qpos_idx]
        mujoco.mj_forward(model, data)

    # G1 has 7 foot collision geoms per side (numbered 1-7).
    foot_geom_names = [
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    ]
    axis_frames = {
        "body": args.axis_bodies,
        "body_com": [],
        "site": args.axis_sites,
        "geom": args.axis_geoms,
    }

    launch_robot_viewer(
        model,
        data,
        keyframes={
            "R": ("KNEES_BENT_KEYFRAME", lambda: _apply_keyframe(KNEES_BENT_KEYFRAME)),
            "H": ("HOME_KEYFRAME", lambda: _apply_keyframe(HOME_KEYFRAME)),
        },
        base_body=G1_BASE_BODY,
        foot_geom_names=foot_geom_names,
        get_action_scale=get_action_scale,
        axis_frames=axis_frames,
    )
