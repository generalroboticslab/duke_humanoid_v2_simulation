"""parallel_gripper — the ParallelGripper mini-gripper as a swappable :class:`Hand`.

A single-servo, rack-and-pinion PARALLEL-jaw gripper (two jaws on mirrored Y slides, 1:1
coupled so one FEETECH servo drives both), sourced from
``asset/duke_v2/parallel_gripper``. Exposes ``PARALLEL_GRIPPER`` — pass it (or any
other ``Hand``) to an arm factory::

    from mj_envs.asset_zoo.ur5e import get_ur5e_robot_cfg
    from mj_envs.asset_zoo.parallel_gripper import PARALLEL_GRIPPER
    cfg = get_ur5e_robot_cfg(hand=PARALLEL_GRIPPER)

WHAT DIFFERS FROM cartesian_hand_v2
-----------------------------------
Simpler hardware: a pure gripper of 3 bodies (base + 2 racks) plus ONE decoupled mount
flange (``cnc_flange`` — the silver disc from ``CNC.step``, composed onto ``base`` at
load time, NOT baked into the XML), ONE position actuator ``m_grip`` on ``left_rack_y``
(``right_rack_y`` mimics via ``<equality>``), and **capsule collision** on the two jaws
(``*_rack_collisionN``, box->capsule approximation: pad + finger-back capsules along each finger
+ one slider capsule across each rack base; group 3, contype=1/conaffinity=0 so the jaws collide
with objects but not each other). It also carries 8 AprilTag (tag36h11) fiducials per hand as
zero-thickness textured decals — 4 riding the jaw plates + 4 on the base tag-holder pads (the
holders/pads are CAD parts baked into ``base.obj`` since the ParallelGripper0710 re-export; the
configurable tag pipeline lives in the asset dir: ``tag_grids.py`` + ``make_tags.py``). No SDF
plugin is needed.

Like cartesian_hand_v2, the flange is DECOUPLED (gripper XML stays pure; the flange mesh
lives in ``flanges/`` and its physics is Fusion-text, composed by ``_attach_flanges``), so
it can be edited/swapped independently.

ROBOT WIRING: registered as a selectable hand on the humanoid_v21 — its LEFT/RIGHT variants
ride the two wrists via ``humanoid_v21_constants.HAND_REGISTRY["parallel_gripper"]``.
Launch it with::

    python mj_envs/asset_zoo/humanoid_v21/humanoid_v21_constants.py \\
        --end_effector actuated --hand parallel_gripper

The mount geometry below is mesh-derived; confirm it against the CAD flange datum before
relying on it on a robot (see the PositionDeter doc).
"""
from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg

from mj_envs.asset_zoo.hand import Hand, make_range_action_scale, quat_between_axes, quat_to_mat


# ═══════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════

HAND_DIR = Path(__file__).parents[2] / "asset" / "duke_v2" / "parallel_gripper"
HAND_XML = HAND_DIR / "parallel_gripper.xml"
HAND_MESH_DIR = HAND_DIR / "meshes"
TAG_DIR = HAND_MESH_DIR / "tags"             # decoupled AprilTag decals (composed per-hand at load)
FLANGE_DIR = HAND_DIR / "flanges"            # decoupled mount flange mesh (composed at load time)

# Flange asset config (mesh / parent body / geom group / colour). The PHYSICS (mass, COM,
# inertia) is NOT here and NOT in JSON — it lives as Fusion-360 text in
# parallel_gripper_fusion_info.FLANGE_INFO (same string convention as head_cam) and is
# parsed live in _attach_flanges below. `cnc_flange` IS this gripper's mounting flange (the
# silver disc from CNC.step), decoupled out of the gripper XML so it is managed/swapped
# independently — same pattern as cartesian_hand_v2.
FLANGE_CONFIG = {
    # The base tag-holder plates + their pocket-seated tag pads and the usb-c protector are
    # NOT flanges anymore: since the ParallelGripper0710 re-export they are CAD parts of the
    # base link (merged into meshes/base.obj, mass/COM/inertia inside the base Fusion text),
    # the same way the jaw tag plates live inside the rack meshes.
    "cnc_flange": {"mesh": "cnc_flange.obj", "parent": "base", "group": 2, "rgba": [0.75, 0.76, 0.80, 1.0]},
}

# Load the module's Fusion-text physics (FLANGE_INFO + parse_fusion) from the asset dir.
_fi_spec = importlib.util.spec_from_file_location(
    "parallel_gripper_fusion_info", HAND_DIR / "parallel_gripper_fusion_info.py")
_FUSION_INFO = importlib.util.module_from_spec(_fi_spec)
_fi_spec.loader.exec_module(_FUSION_INFO)

# Per-hand AprilTag layout (slots + L/R tag-id sets). LEFT and RIGHT grippers SHARE this XML
# but get DIFFERENT tag ids, composed at load by _attach_tags (the XML is tag-free).
_tl_spec = importlib.util.spec_from_file_location(
    "parallel_gripper_tag_layout", HAND_DIR / "tag_layout.py")
_TAG_LAYOUT = importlib.util.module_from_spec(_tl_spec)
_tl_spec.loader.exec_module(_TAG_LAYOUT)

HAND_ROOT_BODY = "base"
# Matches the jaw capsule colliders ``left_rack_collisionN`` / ``right_rack_collisionN`` (also
# under an attach prefix), emitted by the builder's box->capsule approximation. Excludes the
# physics-inert IK proxy ``*_rack_ikproxy``. The arm includes this in its CollisionCfg.
HAND_COLLISION_GEOM_REGEX = r".*_rack_collision\d+"


# ═══════════════════════════════════════════════════════════
#  MOUNT GEOMETRY  (edit here to re-seat the hand)
#  Source: PositionDeter/RELATIVE_POSITION_flange__parallel_gripper.md
#
#  The gripper mounts via its decoupled ``cnc_flange`` (the silver disc, -X end); the jaws
#  reach +X. mating_face_pos is the flange -X face center, mesh-measured — refine against the
#  CAD flange datum at robot-wiring time.
# ═══════════════════════════════════════════════════════════

# Jaws open/close along ±Y; the gripper reaches toward objects along base-frame +X
# (fingertips at x ~ +0.12 m), and mounts on the arm at the -X cnc_flange face.
HAND_REACH_AXIS_IN_BASE = np.array([1.0, 0.0, 0.0])

# -X mount-face center (cnc_flange), hand base frame, mesh-measured from cnc_flange.obj bbox.
HAND_MATING_FACE_POS_IN_BASE = np.array([-0.032, 0.006, -0.010])

# Grasp/end-effector reference point = geometric center of the cuboid formed by the two jaws'
# CONTACT FACES (the fingers' inner grasping surfaces), in the gripper base frame: X is the
# mid-finger grasping region (~0.086), Y the midpoint between the two faces, Z the face-height
# center. Stable under jaw open/close (the racks slide symmetrically, so the Y midpoint is
# fixed). Use this to seat an arm's end-effector / IK site at the gripper's grasp center:
# site_pos(in flange frame) = mount_pos + R(mount_quat) @ GRASP_CENTER_IN_BASE  (see
# place_ee_site_at_grasp_center()).
# After changing this TCP, regenerate both static cuRobo tool-frame exports:
# /home/grl/repo/micromamba/envs/py312/bin/python asset/create/export_mjspec_to_urdf.py --robot humanoid_v21 --head-camera actuated --end-effector actuated --hand parallel_gripper --format urdf
# /home/grl/repo/micromamba/envs/py312/bin/python asset/create/export_mjspec_to_urdf.py --robot g1 --head-camera builtin --end-effector actuated --hand parallel_gripper --format urdf
# GRASP_CENTER_IN_BASE = np.array([0.08564, 0.00696, -0.01000]) # orginal
GRASP_CENTER_IN_BASE = np.array([0.09, 0.00696, -0.01000]) # move slightly out


def place_ee_site_at_grasp_center(spec, site_name, flange_axis=np.array([1.0, 0.0, 0.0])):
    """Move an arm's end-effector site to the gripper's grasp center (the 4-plate cuboid
    center). Call AFTER the gripper is attached onto the flange that carries `site_name`.
    The site is expressed in its own (flange) body frame, so we map the grasp center through
    the same mount pose (mount_pose) used to seat the gripper base on the flange."""
    quat = quat_between_axes(HAND_REACH_AXIS_IN_BASE, flange_axis)   # gripper base -> flange rot
    mount_pos = -quat_to_mat(quat) @ HAND_MATING_FACE_POS_IN_BASE    # base origin in flange frame
    grasp_in_flange = mount_pos + quat_to_mat(quat) @ GRASP_CENTER_IN_BASE
    spec.site(site_name).pos = grasp_in_flange.tolist()


# ═══════════════════════════════════════════════════════════
#  ACTUATORS  (one position servo, exactly as the source MJCF)
#
#  One FEETECH servo drives both jaws through the rack-and-pinion; only ``left_rack_y`` is
#  actuated (``right_rack_y`` follows 1:1 via the mimic equality). The source MJCF uses a
#  single <position> actuator (kp=800, kv=8, forcerange ±150, ctrlrange [0, 0.0847]); we
#  reproduce it 1:1 — kp->stiffness, kv->damping, forcerange->effort_limit.
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
#  GRIPPER RACK TARGET POSITIONS (Single Source of Truth)
# ═══════════════════════════════════════════════════════════
PARALLEL_GRIPPER_OPEN_RACK_POS_M: float = -0.010    # Open rack target (m); -10 mm per rack = 20 mm wider total jaw gap
PARALLEL_GRIPPER_GRASP_RACK_POS_M: float = 0.019    # Fixed 60 mm task-cube grasp target; do not move when open width changes
PARALLEL_GRIPPER_CLOSE_STROKE_M: float = (
    PARALLEL_GRIPPER_GRASP_RACK_POS_M - PARALLEL_GRIPPER_OPEN_RACK_POS_M
)
_PARALLEL_GRIPPER_CONTACT_CUBE_SIZE_M: float = 0.05
_PARALLEL_GRIPPER_CONTACT_RACK_POS_M: float = 0.00995
PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S: float = 0.04


def parallel_gripper_contact_rack_pos(cube_size_m: float) -> float:
    """Rack coordinate at first symmetric jaw contact for a cube edge in metres.

    The two jaws approach by the rack coordinate on opposing sides, so increasing cube edge by
    ``d`` moves first contact outward by ``d / 2`` per rack. Dynamic physics discovers contact from
    the actual geom; kinematic replay needs this analytic counterpart.
    """
    return _PARALLEL_GRIPPER_CONTACT_RACK_POS_M - 0.5 * (
        float(cube_size_m) - _PARALLEL_GRIPPER_CONTACT_CUBE_SIZE_M
    )


HAND_FORCE_LIMIT = 150.0   # N  (source forcerange ±150)
HAND_STIFFNESS = 800.0     # source kp
HAND_DAMPING = 8.0         # source kv

# Driven joint -> (lo, hi) slide range (m) from parallel_gripper.xml. Only this one is
# actuated; right_rack_y follows via the mimic equality.
HAND_DRIVEN_JOINT_RANGES: dict[str, tuple[float, float]] = {
    "left_rack_y": (0.0, 0.0847),
}


def hand_actuators(name_prefix: str = "") -> tuple[BuiltinPositionActuatorCfg, ...]:
    """BuiltinPositionActuatorCfg for the hand's single driven joint.

    Args:
        name_prefix: Prefix ``attach_body`` applies to joint names (must match).
    """
    return tuple(
        BuiltinPositionActuatorCfg(
            target_names_expr=(name_prefix + joint,),
            stiffness=HAND_STIFFNESS,
            damping=HAND_DAMPING,
            effort_limit=HAND_FORCE_LIMIT,
        )
        for joint in HAND_DRIVEN_JOINT_RANGES
    )


def hand_action_scale(name_prefix: str = "") -> dict[str, float]:
    """Range-based action scale (stiff position servo: command a position across the
    jaw travel)."""
    ranges = {name_prefix + joint: rng for joint, rng in HAND_DRIVEN_JOINT_RANGES.items()}
    return make_range_action_scale(ranges)


def pin_gripper_ctrlrange(model: mujoco.MjModel) -> int:
    """Pin the gripper actuator to exactly the source MJCF command range:
    ctrlrange = joint range ([0, 0.0847]) AND ctrllimited=True, on a COMPILED model.

    mjlab leaves position actuators ctrllimited=False with a wider *informational*
    ctrlrange (joint_range ± effort/stiffness) so the servo can push past the joint limit.
    The source MJCF instead uses ctrllimited="true" ctrlrange="0 0.0847" — and the MuJoCo
    viewer's Control sliders only honor ctrlrange when ctrllimited=True (else they span
    [-1,1]). Matching the source requires BOTH: set the range and enable the limit. Matches
    the gripper joint by name suffix, so it works under any attach prefix. Returns the count
    pinned. Call after spec.compile().
    """
    suffixes = tuple(HAND_DRIVEN_JOINT_RANGES)
    n = 0
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if any(jname.endswith(s) for s in suffixes):
            model.actuator_ctrlrange[i] = model.jnt_range[jid]
            model.actuator_ctrllimited[i] = 1     # source uses ctrllimited="true"
            n += 1
    return n


# ═══════════════════════════════════════════════════════════
#  SPEC SURGERY
# ═══════════════════════════════════════════════════════════

def load_hand_module() -> mujoco.MjSpec:
    """Load the hand MjSpec ready for grafting onto an arm.

    Makes mesh AND tag-texture paths absolute (so they survive the attach and an
    arbitrary CWD); drops the standalone-scene furniture (floor geom, lights, grid
    material/texture); strips the native <position> actuator (re-added as Builtin via
    ``hand_actuators``). The 2 slide joints (with their XML limits) + the mimic equality,
    and the 4 visual-only AprilTag decals, are KEPT and ride ``attach_body``.
    """
    hand = mujoco.MjSpec.from_file(str(HAND_XML))

    for mesh in hand.meshes:                   # absolute paths survive attach
        if mesh.file:
            mesh.file = str((HAND_MESH_DIR / mesh.file).resolve())
    for texture in list(hand.textures):
        if texture.name == "grid_tex":         # scene-floor checker (no file)
            hand.delete(texture)
        elif texture.file:                     # AprilTag PNGs — keep, make absolute
            texture.file = str((HAND_MESH_DIR / texture.file).resolve())
    for material in list(hand.materials):
        if material.name == "grid":            # floor finish
            hand.delete(material)
    for geom in list(hand.worldbody.geoms):    # floor plane (scene furniture)
        hand.delete(geom)
    for light in list(hand.lights):
        hand.delete(light)
    for actuator in list(hand.actuators):      # rebuilt as Builtin
        hand.delete(actuator)
    return hand


# ═══════════════════════════════════════════════════════════
#  DECOUPLED FLANGE  (composed onto the gripper at load time, NOT baked in the XML)
# ═══════════════════════════════════════════════════════════

def _attach_flanges(spec: mujoco.MjSpec, flange_names: tuple[str, ...]) -> None:
    """Compose decoupled mount flanges onto the gripper's ``base``.

    Each flange is a VISUAL-ONLY welded child body of its configured parent: mesh + material
    from FLANGE_CONFIG, and an <inertial> parsed from the Fusion-360 text in
    parallel_gripper_fusion_info.FLANGE_INFO (no JSON), with NO collider (the arm owns
    collision). The gripper model stays flange-free, so the flange is edited independently.
    """
    for name in flange_names:
        cfg = FLANGE_CONFIG[name]
        _add_flange_mesh_mat(spec, name, cfg)
        mass, com, inertia = _FUSION_INFO.parse_fusion(_FUSION_INFO.FLANGE_INFO[name])
        body = spec.body(cfg["parent"]).add_body()
        body.name = name
        body.explicitinertial = True
        body.mass = float(mass)
        body.ipos = [float(c) for c in com]
        body.inertia = [float(inertia[0, 0]), float(inertia[1, 1]), float(inertia[2, 2])]  # flange: diagonal
        body.iquat = [1.0, 0.0, 0.0, 0.0]
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = f"{name}_mesh"
        geom.material = f"{name}_mat"
        geom.name = f"{name}_visual"
        geom.group = int(cfg["group"])
        geom.contype = 0
        geom.conaffinity = 0


def _add_flange_mesh_mat(spec: mujoco.MjSpec, name: str, cfg: dict) -> None:
    """Register a flange entry's OBJ mesh (mm -> m) + its material."""
    mesh = spec.add_mesh()
    mesh.name = f"{name}_mesh"
    mesh.file = str((FLANGE_DIR / cfg["mesh"]).resolve())
    mesh.scale = [1e-3, 1e-3, 1e-3]
    mat = spec.add_material()
    mat.name = f"{name}_mat"
    mat.rgba = cfg["rgba"]


# ═══════════════════════════════════════════════════════════
#  DECOUPLED APRILTAGS  (composed per-hand at load; the gripper XML is tag-free)
# ═══════════════════════════════════════════════════════════

def _attach_tags(spec: mujoco.MjSpec, hand: str | None) -> None:
    """Compose one hand's 4 AprilTag decals onto the gripper jaws.

    ``hand`` is "L", "R", or None (no tags). LEFT and RIGHT use DISJOINT tag-id sets
    (tag_layout.HAND_TAGS) so an AprilTag detector can tell the two grippers apart, while
    SHARING this one gripper XML. Each tag is a zero-thickness textured quad (the cv2.aruco
    36h11 PNG + flat OBJ, rot=180 so detected +X = +gripper X) added on its jaw body —
    visual-only (group 2, no collider).
    """
    if hand is None:
        return
    for tid in _TAG_LAYOUT.hand_tag_ids(hand):
        body_name = _TAG_LAYOUT.tag_body(tid)
        tex = spec.add_texture()
        tex.name = f"tag_{tid}_tex"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str((TAG_DIR / f"tag_{tid}.png").resolve())
        mat = spec.add_material()
        mat.name = f"tag_{tid}_mat"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"tag_{tid}_tex"
        mat.texuniform = False
        mat.specular = 0.0
        mat.shininess = 0.0
        mat.reflectance = 0.0
        mesh = spec.add_mesh()
        mesh.name = f"tag_{tid}_quad"
        mesh.file = str((TAG_DIR / f"tag_{tid}.obj").resolve())
        mesh.scale = [1e-3, 1e-3, 1e-3]
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL   # flat quad has no volume
        geom = spec.body(body_name).add_geom()
        geom.name = f"tag_{tid}"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = f"tag_{tid}_quad"
        geom.material = f"tag_{tid}_mat"
        geom.group = 2
        geom.contype = 0
        geom.conaffinity = 0


# ═══════════════════════════════════════════════════════════
#  HAND INSTANCES  (pure gripper + decoupled flange + per-hand tags composed on)
# ═══════════════════════════════════════════════════════════

def make_parallel_gripper(flanges: tuple[str, ...] = ("cnc_flange",),
                               tags: str | None = None,
                               name: str = "parallel_gripper") -> Hand:
    """A ``Hand`` = pure gripper (base + 2 racks) + decoupled flange(s) + one hand's tags.

    ``flanges`` defaults to the gripper's own ``cnc_flange`` (pass ``()`` for none).
    ``tags`` is "L" / "R" (compose that hand's 4 AprilTags) or None (no tags).
    """
    def _load() -> mujoco.MjSpec:
        spec = load_hand_module()        # pure gripper (scene/actuator surgery)
        _attach_flanges(spec, flanges)   # compose the decoupled flange (visual-only)
        _attach_tags(spec, tags)         # compose this hand's AprilTags (visual-only)
        return spec
    return Hand(
        name=name,
        root_body=HAND_ROOT_BODY,
        collision_geom_regex=HAND_COLLISION_GEOM_REGEX,
        load_module=_load,
        actuators=hand_actuators,
        action_scale=hand_action_scale,
        reach_axis=HAND_REACH_AXIS_IN_BASE,
        mating_face_pos=HAND_MATING_FACE_POS_IN_BASE,
    )


# Per-hand variants: same gripper + cnc_flange, DISJOINT AprilTag id sets (L: jaws 80-83 +
# base pads 84-87, R: jaws 90-93 + base pads 94-97) so the detector distinguishes the two
# grippers. The base tag-holders/pads ride inside base.obj (CAD parts, not flanges).
# Mount LEFT on the left wrist and RIGHT on the right. Plus a tag-free bare variant.
PARALLEL_GRIPPER_LEFT = make_parallel_gripper(("cnc_flange",), tags="L", name="parallel_gripper+L_tags")
PARALLEL_GRIPPER_RIGHT = make_parallel_gripper(("cnc_flange",), tags="R", name="parallel_gripper+R_tags")
PARALLEL_GRIPPER = PARALLEL_GRIPPER_LEFT          # default (back-compat alias)
PARALLEL_GRIPPER_BARE = make_parallel_gripper((), None, "parallel_gripper_bare")

# Flange-LESS per-hand variants (gripper + tags, NO cnc_flange) — for a host whose
# end-effector link ALREADY provides the mount flange (e.g. humanoid_v21's
# end_effector_attachment, ~41 g): the hand must NOT bring its own cnc_flange or the flange is
# double-counted. Same mount pose as the flanged variants, so the gripper base sits in the
# same place.
PARALLEL_GRIPPER_LEFT_NO_FLANGE = make_parallel_gripper((), tags="L", name="parallel_gripper+L_tags_no_flange")
PARALLEL_GRIPPER_RIGHT_NO_FLANGE = make_parallel_gripper((), tags="R", name="parallel_gripper+R_tags_no_flange")


if __name__ == "__main__":
    import re
    from dataclasses import dataclass as _dataclass
    from typing import Literal

    import tyro

    from mj_envs.asset_zoo.humanoid_v21.robot_viewer import launch_robot_viewer

    @_dataclass
    class _Args:
        which: Literal["left", "right", "bare"] = "bare"
        """Gripper variant: "bare" (no flange, no tags, default) or "left"/"right" (flange + disjoint AprilTags)."""

    args = tyro.cli(_Args)
    hand = {
        "left": PARALLEL_GRIPPER_LEFT,
        "right": PARALLEL_GRIPPER_RIGHT,
        "bare": PARALLEL_GRIPPER_BARE,
    }[args.which]

    spec = hand.load_module()   # load_module strips the actuator for grafting; preview re-adds it
    # preview only: re-add the position servo load_hand_module strips for grafting, so the viewer
    # Control slider drives the jaws (kp/kv/forcerange from the shared HAND_* constants).
    a = spec.add_actuator()
    a.name, a.target = "m_grip", "left_rack_y"
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm[0], a.biasprm[1], a.biasprm[2] = HAND_STIFFNESS, -HAND_STIFFNESS, -HAND_DAMPING
    a.forcerange = [-HAND_FORCE_LIMIT, HAND_FORCE_LIMIT]
    model = spec.compile()
    pin_gripper_ctrlrange(model)   # ctrllimited=True + ctrlrange=joint range -> real viewer slider
    data = mujoco.MjData(model)
    qpos0 = data.qpos.copy()

    def reset_to_home() -> None:
        data.qpos[:] = qpos0
        data.qvel[:] = 0.0

    collision_geoms = [
        name for i in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
        and re.fullmatch(HAND_COLLISION_GEOM_REGEX, name)
    ]

    launch_robot_viewer(
        model, data,
        keyframes={"R": ("initial pose", reset_to_home)},
        base_body=HAND_ROOT_BODY,
        foot_geom_names=collision_geoms,
    )
