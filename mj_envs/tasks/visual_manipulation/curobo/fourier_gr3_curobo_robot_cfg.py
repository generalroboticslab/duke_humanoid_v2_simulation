"""cuRobo arm-only configuration for the Fourier GR-3 v2.1.1 source MJCF.

GR-3 has 31 revolute joints (3 waist, 2 head, 7 per arm, 6 per leg) on a floating
``base_link``. Workspace planning uses the seven physical arm outputs per side
(shoulder pitch/roll/yaw, elbow pitch, wrist yaw/pitch/roll). Every joint is an independent
hinge -- the source declares no equality, tendon or gear coupling (neq=0, ntendon=0) -- so
the planner c-space maps 1:1 onto source qpos and there is no motor writeback to apply.

Standing planning origin is (0, 0, 0.9270331), the ``standing`` keyframe height baked by
``asset/fourier_gr3/gr3_import.py`` (exact analytic ground contact of the foot collision
cylinders). Tool frames are the two fixed ``end_effector_{L,R}_site`` sites at the dummy
hands' distal tip.

Collision spheres come from source group 3, which for this robot is the REFITTED upper-body
model built by the import script: the vendor's own primitives enclose only 20.4% of the
robot's visual surface (16.4% across the arm links) against 50.3% / 47.6% for Booster T1,
and leave the forearm and both wrist links with no collision geometry at all. The refit is
inscribed in each link's visual mesh, never inflated, and raises coverage to 58.4% / 69.8%.

Legs keep their vendor (unrefitted) primitives and are included whole-body: after fixing
``mj_collision_spheres._box_spheres`` (it was emitting duplicate coincident spheres on any
axis narrower than the sphere radius), the whole-body sphere count dropped from 803 to well
under the 500 budget even with legs in, so there is no broadphase-cost reason left to prune
them. All non-arm joints they hang off (hip/knee/ankle, same as waist/head) are locked at
source ``qpos0`` by :func:`_lock_nonarm_joints`.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Mapping

import mujoco

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

from tasks.visual_manipulation.curobo.urdf_localize import localize_urdf
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mj_envs.tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import (  # noqa: E402
    _rest_overlapping_link_pairs,
)
from mj_envs.utils.mj_collision_spheres import (  # noqa: E402
    SPHERE_BUDGET,
    collision_geoms_for_body,
    geom_to_spheres,
)

FOURIER_GR3_MJCF = _REPO_ROOT / "asset" / "fourier_gr3" / "gr3.xml"
FOURIER_GR3_CUROBO_URDF = str(_REPO_ROOT / "asset" / "fourier_gr3" / "fourier_gr3_curobo.urdf")
BASE_LINK = "base_link"
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]
SPHERE_SPACING = 1.5    # see note below
MIN_SPHERE_GAP = 0.03   # floor on the sphere pitch, so hair-thin primitives stay bounded
# Sphere pitch = max(SPHERE_SPACING * radius, MIN_SPHERE_GAP). 1.5 keeps consecutive spheres
# OVERLAPPING (pitch < 2r, so no interstitial gap) while holding the whole-body model (arms +
# torso + legs) to 125 spheres, well inside the study's 500 budget. The default 1.0 gives 167:
# still comfortably under budget, but 1.5 costs nothing in coverage on this refitted model and
# keeps broadphase cheaper.
HOME_ROOT_POS = (0.0, 0.0, 0.9270331)  # `standing` keyframe height from gr3_import.py

LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINT_NAMES)
_ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_SET = frozenset(_ARM_JOINT_NAMES)


def _load_model() -> mujoco.MjModel:
    """Compile the source MJCF carrying tool sites, head cameras and refitted collision."""
    return mujoco.MjModel.from_xml_path(str(FOURIER_GR3_MJCF))


def apply_source_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """GR-3 has no sibling drive/equality coupling; planner c-space maps 1:1 to source qpos."""


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None, f"unnamed MuJoCo body {body_id}"
    return name


def _build_arm_model_spheres(model: mujoco.MjModel) -> dict[str, list[dict]]:
    """Sphere-ise the collision primitives of every body in the source model (whole body,
    including legs -- see module docstring for why legs are no longer pruned).

    Same per-primitive conversion as :func:`mj_collision_spheres.build_collision_spheres`
    (identical ``geom_to_spheres`` calls, so sphere radii are the true primitive radii and
    nothing is inflated); reimplemented as a plain loop here only because that shared helper
    takes a single ``spacing``/``min_gap`` pair for the whole body, whereas GR-3 wants a
    coarser pitch specifically on its flat-paddle dummy hands (see ``SPHERE_SPACING`` note).
    """
    out: dict[str, list[dict]] = {}
    for body_id in range(1, model.nbody):
        geoms = collision_geoms_for_body(model, body_id)
        if not geoms:
            continue
        spheres = [
            {"center": [float(c[0]), float(c[1]), float(c[2])], "radius": float(r)}
            for g in geoms
            for c, r in geom_to_spheres(model, g, spacing=SPHERE_SPACING, min_gap=MIN_SPHERE_GAP)
        ]
        out[_body_name(model, body_id)] = spheres
    total = sum(len(v) for v in out.values())
    assert total <= SPHERE_BUDGET, (
        f"GR-3 whole-body collision model is {total} spheres, over the {SPHERE_BUDGET} budget"
    )
    return out


def _lock_nonarm_joints(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, float]:
    """Pin every non-arm joint on a collision-bearing or tool-bearing chain at its qpos0.

    cuRobo drops visual-only branches from its reduced tree, so passing a lock for a joint on
    an absent branch fails its loader; only joints ancestral to a retained body are emitted.
    For GR-3 that is the three waist joints, two head joints, and the six leg joints per side
    (hip pitch/roll/yaw, knee pitch, ankle pitch/roll) -- legs are collision-bearing now but
    still not planner-active, so they lock at source rest same as waist/head.
    """
    needed_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        for link_name in collision_spheres
    }
    for site_name in TOOL_FRAMES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        assert site_id >= 0, f"tool site {site_name!r} absent from source MJCF"
        needed_bodies.add(int(model.site_bodyid[site_id]))
    needed_joints: set[int] = set()
    for body_id in needed_bodies:
        while body_id:
            if model.body_jntnum[body_id]:
                needed_joints.add(int(model.body_jntadr[body_id]))
            body_id = int(model.body_parentid[body_id])
    lock: dict[str, float] = {}
    for joint_id in needed_joints:
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        assert name is not None, f"unnamed MuJoCo joint {joint_id}"
        if name not in _ARM_JOINT_SET:
            lock[name] = float(model.qpos0[model.jnt_qposadr[joint_id]])
    return lock


def _adjacency_and_rest_ignore(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, list[str]]:
    """Ignore adjacent links plus any pair whose spheres already overlap at the rest pose.

    The rest-overlap set is what the refitted upper body makes unavoidable -- the shoulder
    ball sits inside the torso shell, the waist boxes nest, the wrist capsules nest -- and it
    is the same set the source MJCF declares as ``<contact><exclude>``; both are derived, not
    hand-listed, so they cannot drift apart.
    """

    def add_both(first: str, second: str) -> None:
        if first == second:
            return
        for a, b in ((first, second), (second, first)):
            if b not in ignore.setdefault(a, []):
                ignore[a].append(b)

    ignore: dict[str, list[str]] = {}
    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id:
            add_both(_body_name(model, body_id), _body_name(model, parent_id))
    for first, second in map(tuple, _rest_overlapping_link_pairs(model, collision_spheres)):
        add_both(first, second)
    return ignore


def _arm_cspace(model: mujoco.MjModel, arm_joint_home: Mapping[str, float] | None) -> dict:
    """Fourteen-DOF cspace from source limits, with an optional source-coordinate home."""
    home = {} if arm_joint_home is None else dict(arm_joint_home)
    unknown = set(home) - _ARM_JOINT_SET
    assert not unknown, f"arm_joint_home has non-arm joints: {sorted(unknown)}"
    default_position = []
    for name in _ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"arm joint {name!r} missing from source MJCF"
        qpos0 = float(model.qpos0[model.jnt_qposadr[joint_id]])
        default_position.append(float(home.get(name, qpos0)) - qpos0)
    return {
        "joint_names": list(_ARM_JOINT_NAMES),
        "default_joint_position": default_position,
        "cspace_distance_weight": [1.0] * len(_ARM_JOINT_NAMES),
        "null_space_weight": [1.0] * len(_ARM_JOINT_NAMES),
    }


def build_robot_cfg_dict(
    load_dynamics: bool = False,
    arm_joint_home: Mapping[str, float] | None = None,
) -> dict:
    """Build the source-faithful, collision-aware GR-3 config for arm-only IK."""
    model = _load_model()
    if not pathlib.Path(FOURIER_GR3_CUROBO_URDF).exists():
        raise FileNotFoundError(
            f"missing generated cuRobo URDF: {FOURIER_GR3_CUROBO_URDF}; run "
            "asset/create/export_mjspec_to_urdf.py --robot fourier_gr3 --format urdf "
            "--output asset/fourier_gr3"
        )
    spheres = _build_arm_model_spheres(model)
    lock_joints = _lock_nonarm_joints(model, spheres)
    for frame in TOOL_FRAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame) >= 0
    print(
        f"[curobo] Fourier GR-3 collision spheres: {len(spheres)} links / "
        f"{sum(len(values) for values in spheres.values())} spheres | "
        f"lock_joints: {len(lock_joints)} | active arm DOFs: {len(_ARM_JOINT_NAMES)}"
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": localize_urdf(FOURIER_GR3_CUROBO_URDF),
                "asset_root_path": "/",
                "collision_link_names": list(spheres.keys()),
                "collision_spheres": spheres,
                "self_collision_ignore": _adjacency_and_rest_ignore(model, spheres),
                "self_collision_buffer": {},
                "lock_joints": lock_joints,
                "cspace": _arm_cspace(model, arm_joint_home),
            },
            "load_dynamics": load_dynamics,
        }
    }


__all__ = [
    "BASE_LINK",
    "FOURIER_GR3_CUROBO_URDF",
    "FOURIER_GR3_MJCF",
    "HOME_ROOT_POS",
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "TOOL_FRAMES",
    "apply_source_coupling",
    "build_robot_cfg_dict",
]
