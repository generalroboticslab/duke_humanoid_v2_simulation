"""cuRobo arm-only configuration for the PAL Robotics TALOS study MJCF.

TALOS has 44 revolute joints (2 torso, 2 head, 7 per arm, 7 per gripper, 6 per leg) on a
floating ``base_link``. Workspace planning uses the seven physical arm outputs per side
(shoulder pitch/roll/yaw, elbow pitch, wrist yaw/pitch/roll). Every joint is an independent
hinge -- the source declares no equality, tendon or gear coupling (neq=0, ntendon=0) -- so the
planner c-space maps 1:1 onto source qpos and there is no motor writeback to apply. The seven
gripper joints per side are real degrees of freedom but are not planning DOFs here; they are
locked at their home value so the hand still contributes its collision volume.

Standing planning origin is (0, 0, 1.08205), the ``standing`` keyframe height baked by
``asset/pal_talos/talos_study_import.py`` (exact analytic ground contact of the foot collision
primitives). Tool frames are the two fixed ``end_effector_{L,R}_site`` sites at the grasp
centre of each three-finger hand, mounted on the last ARM link rather than on a finger, so the
tool frame does not move when the gripper opens.

HOME POSE. Unlike the other robots in this study, TALOS's ``qpos0`` is NOT its home pose and
is not even a legal configuration: the vendor gives ``arm_*_2`` and ``arm_*_4`` limit intervals
that exclude zero. The canonical home is the ``standing`` keyframe (shoulders abducted 15 deg,
elbows parked at their own straightest limit), which is what the exporter bakes into the URDF
``<default>`` tags and what :data:`ARM_HOME_QPOS` reproduces in source coordinates.

Collision spheres come from source group 3, which for this robot is the REFITTED model built by
the import script: the vendor ships triangle-mesh colliders, which cuRobo cannot consume, so
each is refitted to a capsule inscribed along the collider's own PRINCIPAL axis (TALOS's arm
links are angled, and an axis-aligned box fit collapses the upper arm to 12% self-coverage).
The refit never inflates a collider and reaches 76.7% coverage overall / 79.1% across the arm,
the highest of any robot currently in the study (GR-3 58.4%/71.4%, Booster T1 50.3%). Legs keep
their vendor primitives and are pruned below the hips here regardless, since they sit far
outside the arm-reach envelope and would only add broadphase cost.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Mapping

import mujoco

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
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

TALOS_MJCF = _REPO_ROOT / "asset" / "pal_talos" / "talos_study.xml"
TALOS_CUROBO_URDF = str(_REPO_ROOT / "asset" / "pal_talos" / "pal_talos_curobo.urdf")
BASE_LINK = "base_link"
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]
SPHERE_SPACING = 1.0    # sphere pitch = max(SPHERE_SPACING * radius, MIN_SPHERE_GAP)
MIN_SPHERE_GAP = 0.03   # floor on the pitch, so hair-thin primitives stay bounded
HOME_ROOT_POS = (0.0, 0.0, 1.08205)  # `standing` keyframe height from talos_study_import.py
HOME_KEYFRAME = "standing"

LEFT_ARM_JOINT_NAMES = tuple(f"arm_left_{i}_joint" for i in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"arm_right_{i}_joint" for i in range(1, 8))
_ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_SET = frozenset(_ARM_JOINT_NAMES)

# First body of each leg; everything at or below these is pruned from the arm-only model.
_HIP_ROOT_BODIES = ("leg_left_1_link", "leg_right_1_link")


def _load_model() -> mujoco.MjModel:
    """Compile the source MJCF carrying tool sites, head cameras and refitted collision."""
    return mujoco.MjModel.from_xml_path(str(TALOS_MJCF))


def _home_qpos(model: mujoco.MjModel):
    """The ``standing`` keyframe, the canonical home pose (see the HOME POSE note above)."""
    for k in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == HOME_KEYFRAME:
            return model.key_qpos[k]
    raise AssertionError(f"source MJCF has no {HOME_KEYFRAME!r} keyframe; re-run the import")


def _joint_home(model: mujoco.MjModel, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    assert joint_id >= 0, f"joint {joint_name!r} missing from source MJCF"
    return float(_home_qpos(model)[model.jnt_qposadr[joint_id]])


def apply_source_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """TALOS has no sibling drive/equality coupling; planner c-space maps 1:1 to source qpos."""


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None, f"unnamed MuJoCo body {body_id}"
    return name


def _at_or_below_hip(model: mujoco.MjModel, body_id: int) -> bool:
    """True for each ``leg_*_1_link`` and everything descending from it.

    The arm-only model drops the legs: reachability never approaches them, and every retained
    link costs cuRobo broadphase work on every sphere-pair test of a production solve.
    """
    hip_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _HIP_ROOT_BODIES
    }
    assert -1 not in hip_ids, "TALOS leg_*_1_link body missing from source MJCF"
    while body_id:
        if body_id in hip_ids:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _build_arm_model_spheres(model: mujoco.MjModel) -> dict[str, list[dict]]:
    """Sphere-ise the collision primitives of every link the arm-only model keeps.

    Same per-primitive conversion as :func:`mj_collision_spheres.build_collision_spheres`
    (identical ``geom_to_spheres`` calls, so sphere radii are the true primitive radii and
    nothing is inflated), but the legs are skipped BEFORE sphere-ising rather than after, so
    the shared helper's global 500-sphere budget is not spent on links this model never keeps.
    """
    out: dict[str, list[dict]] = {}
    for body_id in range(1, model.nbody):
        if _at_or_below_hip(model, body_id):
            continue
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
        f"TALOS arm-only collision model is {total} spheres, over the {SPHERE_BUDGET} budget"
    )
    return out


def _lock_nonarm_joints(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, float]:
    """Pin every non-arm joint on a collision-bearing or tool-bearing chain at its home value.

    cuRobo drops visual-only branches from its reduced tree, so passing a lock for a joint on
    an absent branch fails its loader; only joints ancestral to a retained body are emitted.
    For TALOS that is the two torso joints, the two head joints and the seven gripper joints
    per side (the legs having been pruned above).
    """
    home = _home_qpos(model)
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
            lock[name] = float(home[model.jnt_qposadr[joint_id]])
    return lock


def _adjacency_and_rest_ignore(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, list[str]]:
    """Ignore adjacent links plus any pair whose spheres already overlap at the home pose.

    The rest-overlap set is what the inscribed refit makes unavoidable -- the shoulder ball
    sits inside the torso shell, the gripper base nests into the wrist -- and it is the same
    set the source MJCF declares as ``<contact><exclude>``; both are derived, not hand-listed,
    so they cannot drift apart.
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
    """Fourteen-DOF cspace from source limits, with an optional source-coordinate home.

    ``default_joint_position`` is an OFFSET from the value the URDF already carries in its
    ``<default>`` tags, which for TALOS is the ``standing`` keyframe, so an unspecified joint
    contributes exactly zero and the cuRobo seed lands on the standing pose.
    """
    home = {} if arm_joint_home is None else dict(arm_joint_home)
    unknown = set(home) - _ARM_JOINT_SET
    assert not unknown, f"arm_joint_home has non-arm joints: {sorted(unknown)}"
    default_position = []
    for name in _ARM_JOINT_NAMES:
        urdf_default = _joint_home(model, name)
        default_position.append(float(home.get(name, urdf_default)) - urdf_default)
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
    """Build the source-faithful, collision-aware TALOS config for arm-only IK."""
    model = _load_model()
    if not pathlib.Path(TALOS_CUROBO_URDF).exists():
        raise FileNotFoundError(
            f"missing generated cuRobo URDF: {TALOS_CUROBO_URDF}; run "
            "asset/create/export_mjspec_to_urdf.py --robot pal_talos --format urdf "
            "--output asset/pal_talos"
        )
    spheres = _build_arm_model_spheres(model)
    lock_joints = _lock_nonarm_joints(model, spheres)
    for frame in TOOL_FRAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame) >= 0
    print(
        f"[curobo] PAL TALOS collision spheres: {len(spheres)} links / "
        f"{sum(len(values) for values in spheres.values())} spheres | "
        f"lock_joints: {len(lock_joints)} | active arm DOFs: {len(_ARM_JOINT_NAMES)}"
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": TALOS_CUROBO_URDF,
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
    "HOME_ROOT_POS",
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "TALOS_CUROBO_URDF",
    "TALOS_MJCF",
    "TOOL_FRAMES",
    "apply_source_coupling",
    "build_robot_cfg_dict",
]
