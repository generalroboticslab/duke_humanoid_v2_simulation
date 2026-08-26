"""cuRobo arm-only configuration for Apptronik Apollo source MJCF.

Apollo workspace IK actuates seven physical serial arm joints per side and locks torso,
neck, and leg joints at source rest. Tool frames are user-approved palm-center sites.
Collision spheres derive directly from source group-3 primitives; the approved pelvis
under-fit is repeated here so planner and canonical viewer use identical geometry.
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
from mj_envs.utils.mj_collision_spheres import build_collision_spheres  # noqa: E402
try:  # absent in this clone; only the cuRobo-solve path uses it, the visibility scorer does not
    from mj_envs.utils.mj_collision_spheres import CapsuleSphereFit
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class CapsuleSphereFit:
        radius_scale: float = 1.0
        sphere_count: int = 0

APOLLO_MJCF = _REPO_ROOT / "asset" / "apptronik_apollo" / "apptronik_apollo.xml"
APOLLO_CUROBO_URDF = str(_REPO_ROOT / "asset" / "apptronik_apollo" / "apptronik_apollo_curobo.urdf")
BASE_LINK = "base_link"
HOME_ROOT_POS = (0.0, 0.0, 1.0813)
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]

LEFT_ARM_JOINT_NAMES = (
    "l_shoulder_aa", "l_shoulder_ie", "l_shoulder_fe", "l_elbow_fe",
    "l_wrist_roll", "l_wrist_yaw", "l_wrist_pitch",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("l_", "r_", 1) for name in LEFT_ARM_JOINT_NAMES)
_ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_SET = frozenset(_ARM_JOINT_NAMES)
# User-approved source-vs-sphere fit: preserve 0.38 m pelvis axial span with five 80 mm spheres.
_CAPSULE_SPHERE_FITS = {
    "base_link": CapsuleSphereFit(radius_scale=0.8, sphere_count=5),
}


def _load_model() -> mujoco.MjModel:
    """Compile source MJCF carrying approved palm sites and group-3 collision."""
    return mujoco.MjModel.from_xml_path(str(APOLLO_MJCF))


def apply_source_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Apollo arm outputs are direct serial hinges; source defines no gearbox writeback."""


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None, f"unnamed MuJoCo body {body_id}"
    return name


def _lock_nonarm_joints(model: mujoco.MjModel) -> dict[str, float]:
    """Lock every non-arm hinge at source rest for fixed-base arm workspace IK."""
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
    """Ignore direct kinematic neighbors and only source-rest sphere overlaps."""
    ignore: dict[str, list[str]] = {}

    def add_both(first: str, second: str) -> None:
        if first == second:
            return
        for a, b in ((first, second), (second, first)):
            if b not in ignore.setdefault(a, []):
                ignore[a].append(b)

    for body_id in range(1, model.nbody):
        parent_id = int(model.body_parentid[body_id])
        if parent_id:
            add_both(_body_name(model, body_id), _body_name(model, parent_id))
    for first, second in map(tuple, _rest_overlapping_link_pairs(model, collision_spheres)):
        add_both(first, second)
    return ignore


def _arm_cspace(
    model: mujoco.MjModel,
    arm_joint_home: Mapping[str, float] | None,
    active_joint_names: tuple[str, ...],
) -> dict:
    """Return selected physical arm coordinates, anchored at source rest by default."""
    home = {} if arm_joint_home is None else dict(arm_joint_home)
    unknown = set(home) - _ARM_JOINT_SET
    assert not unknown, f"arm_joint_home has non-arm joints: {sorted(unknown)}"
    default_position = []
    for name in active_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"active arm joint {name!r} missing from source MJCF"
        qpos0 = float(model.qpos0[model.jnt_qposadr[joint_id]])
        default_position.append(float(home.get(name, qpos0)) - qpos0)
    return {
        "joint_names": list(active_joint_names),
        "default_joint_position": default_position,
        "cspace_distance_weight": [1.0] * len(active_joint_names),
        "null_space_weight": [1.0] * len(active_joint_names),
    }


def build_robot_cfg_dict(
    load_dynamics: bool = False,
    arm_joint_home: Mapping[str, float] | None = None,
    active_side: str | None = None,
) -> dict:
    """Build source-faithful Apollo config, optionally freezing idle arm at source rest.

    Workspace solves target one hand at a time. Passing ``active_side`` exposes only that
    serial arm to IK and locks opposite arm at ``qpos0``. Pinning idle palm pose alone is
    insufficient: a redundant 7-DOF arm can move while retaining same palm transform.
    ``None`` preserves full dual-arm config for utilities that require both arms active.
    """
    model = _load_model()
    if active_side is None:
        active_joint_names = _ARM_JOINT_NAMES
        idle_joint_names: tuple[str, ...] = ()
    else:
        active_side = active_side.upper()
        if active_side not in ("L", "R"):
            raise ValueError(f"active_side must be 'L', 'R', or None, got {active_side!r}")
        active_joint_names = LEFT_ARM_JOINT_NAMES if active_side == "L" else RIGHT_ARM_JOINT_NAMES
        idle_joint_names = RIGHT_ARM_JOINT_NAMES if active_side == "L" else LEFT_ARM_JOINT_NAMES
    if not pathlib.Path(APOLLO_CUROBO_URDF).exists():
        # Tracked artifact, not a generated one: `export_mjspec_to_urdf.py` accepts only
        # humanoid_v21/g1/fourier_gr3/pal_talos, so restoring from git is the only recovery.
        # Registering an exporter entry would need a compile snapshot, and any drift from the
        # tracked file moves the published Apollo numbers -- don't fabricate a replacement.
        raise FileNotFoundError(
            f"missing cuRobo URDF: {APOLLO_CUROBO_URDF}; restore it with "
            "`git checkout -- asset/apptronik_apollo/apptronik_apollo_curobo.urdf`"
        )
    spheres = build_collision_spheres(
        model,
        box_inscribed=True,
        capsule_sphere_overrides=_CAPSULE_SPHERE_FITS,
    )
    total_spheres = sum(len(values) for values in spheres.values())
    assert total_spheres == 295, f"Apollo collision sphere drift: {total_spheres}, expected 295"
    lock_joints = _lock_nonarm_joints(model)
    for name in idle_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"idle arm joint {name!r} missing from source MJCF"
        lock_joints[name] = float(model.qpos0[model.jnt_qposadr[joint_id]])
    for frame in TOOL_FRAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame) >= 0, (
            f"tool site {frame!r} missing from source MJCF"
        )
    print(
        f"[curobo] Apollo collision spheres: {len(spheres)} links / {total_spheres} spheres | "
        f"lock_joints: {len(lock_joints)} | active arm DOFs: {len(active_joint_names)}"
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": APOLLO_CUROBO_URDF,
                "asset_root_path": "/",
                "collision_link_names": list(spheres.keys()),
                "collision_spheres": spheres,
                "self_collision_ignore": _adjacency_and_rest_ignore(model, spheres),
                "self_collision_buffer": {},
                "lock_joints": lock_joints,
                "cspace": _arm_cspace(model, arm_joint_home, active_joint_names),
            },
            "load_dynamics": load_dynamics,
        }
    }


__all__ = [
    "APOLLO_CUROBO_URDF",
    "APOLLO_MJCF",
    "BASE_LINK",
    "HOME_ROOT_POS",
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "TOOL_FRAMES",
    "apply_source_coupling",
    "build_robot_cfg_dict",
]
