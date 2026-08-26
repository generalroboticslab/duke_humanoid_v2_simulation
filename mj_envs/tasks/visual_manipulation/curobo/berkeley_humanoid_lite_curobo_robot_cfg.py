"""cuRobo robot configuration for Berkeley Humanoid Lite bare-hand reachability.

Kinematics load from the generated fixed-base URDF, while collision spheres come directly from
the source MJCF group-3 primitives.  This keeps the planner model tied to the same source geometry
used for MuJoCo FK.  The tool frames are the explicit endpoint sites, 0.10 m along each terminal
hand-link local -Z axis; they describe bare-hand reach, not a gripper grasp center.
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

from mj_envs.utils.mj_collision_spheres import build_collision_spheres  # noqa: E402
from mj_envs.tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import (  # noqa: E402
    _rest_overlapping_link_pairs,
)

BERKELEY_MJCF = _REPO_ROOT / "asset" / "berkeley_humanoid_lite" / "berkeley_humanoid_lite.xml"
BERKELEY_CUROBO_URDF = str(
    _REPO_ROOT / "asset" / "berkeley_humanoid_lite" / "berkeley_humanoid_lite_curobo.urdf"
)
BASE_LINK = "base_link"
BOX_SPHERE_RADIUS_SCALE = 0.75
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]

LEFT_ARM_JOINT_NAMES = (
    "arm_left_shoulder_pitch_joint",
    "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
)
RIGHT_ARM_JOINT_NAMES = (
    "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
)
_ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_SET = frozenset(_ARM_JOINT_NAMES)


def _load_model() -> mujoco.MjModel:
    """Compile source MJCF containing the authoritative endpoint sites and collision geoms."""
    return mujoco.MjModel.from_xml_path(str(BERKELEY_MJCF))


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None, f"unnamed MuJoCo body {body_id}"
    return name


def _lock_nonarm_joints(model: mujoco.MjModel) -> dict[str, float]:
    """Freeze every articulated non-arm joint at source ``qpos0`` for arm-only IK."""
    lock: dict[str, float] = {}
    for joint_id in range(model.njnt):
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
    """Ignore adjacent links and source-rest sphere overlaps, but no other self-collisions."""
    ignore: dict[str, list[str]] = {}

    def add_both(a: str, b: str) -> None:
        if a == b:
            return
        for first, second in ((a, b), (b, a)):
            peers = ignore.setdefault(first, [])
            if second not in peers:
                peers.append(second)

    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id != 0:
            add_both(_body_name(model, body_id), _body_name(model, parent_id))
    for first, second in map(tuple, _rest_overlapping_link_pairs(model, collision_spheres)):
        add_both(first, second)
    return ignore


def _arm_cspace(model: mujoco.MjModel, arm_joint_home: Mapping[str, float] | None) -> dict:
    """Return ten-arm-DOF cspace anchored at source rest unless an explicit home is supplied.

    The asset defines no validated reach-ready posture.  Defaulting to its compiled rest state
    keeps this configuration source-faithful; workspace solves seed from collision-free cache poses.
    """
    home = {} if arm_joint_home is None else dict(arm_joint_home)
    unknown = set(home) - _ARM_JOINT_SET
    assert not unknown, f"arm_joint_home has non-arm joints: {sorted(unknown)}"
    default_position = []
    for name in _ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"active arm joint {name!r} missing from source MJCF"
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
    """Build source-faithful Berkeley Lite cuRobo config for arm-only reachability IK."""
    model = _load_model()
    if not pathlib.Path(BERKELEY_CUROBO_URDF).exists():
        # Tracked artifact, not a generated one: `export_mjspec_to_urdf.py` accepts only
        # humanoid_v21/g1/fourier_gr3/pal_talos, so restoring from git is the only recovery.
        # Registering an exporter entry would need a compile snapshot, and any drift from the
        # tracked file moves the published Berkeley numbers -- don't fabricate a replacement.
        raise FileNotFoundError(
            f"missing cuRobo URDF: {BERKELEY_CUROBO_URDF}; restore it with "
            "`git checkout -- asset/berkeley_humanoid_lite/berkeley_humanoid_lite_curobo.urdf`"
        )
    # Source boxes are intended tight envelopes. Keep the sphere union inside them; the generic
    # converter's face-centered spheres extend one radius beyond each face.
    spheres = build_collision_spheres(
        model,
        box_radius_scale=BOX_SPHERE_RADIUS_SCALE,
        box_inscribed=True,
    )
    lock_joints = _lock_nonarm_joints(model)
    arm_in_lock = _ARM_JOINT_SET.intersection(lock_joints)
    assert not arm_in_lock, f"arm joints leaked into lock_joints: {sorted(arm_in_lock)}"
    for frame in TOOL_FRAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame) >= 0, (
            f"tool site {frame!r} missing from source MJCF"
        )

    total_spheres = sum(len(link_spheres) for link_spheres in spheres.values())
    print(
        f"[curobo] Berkeley Lite collision-geom spheres: {len(spheres)} links / {total_spheres} spheres | "
        f"lock_joints: {len(lock_joints)} | active arm DOFs: {len(_ARM_JOINT_NAMES)}"
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": BERKELEY_CUROBO_URDF,
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
    "BOX_SPHERE_RADIUS_SCALE",
    "BERKELEY_CUROBO_URDF",
    "BERKELEY_MJCF",
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "TOOL_FRAMES",
    "build_robot_cfg_dict",
]
