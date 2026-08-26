"""cuRobo arm-only configuration for Toddlerbot's source MJCF.

Gearbox drive joints are sibling branches; output joints form serial arm chain. This config plans
in seven physical output coordinates per arm and locks independent non-arm joints at source
``qpos0``. MuJoCo writeback must apply source drive/output equality signs.
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
try:  # absent in this clone; only the cuRobo-solve path calls it, the visibility scorer does not
    from mj_envs.utils.mj_collision_spheres import merge_strongly_overlapping_spheres
except ImportError:
    def merge_strongly_overlapping_spheres(*_a, **_k):
        raise NotImplementedError(
            "merge_strongly_overlapping_spheres is missing from this clone's mj_collision_spheres.py"
        )

TODDLERBOT_MJCF = _REPO_ROOT / "asset" / "toddlerbot_2xm_gripper" / "toddlerbot_2xm_gripper_pos.xml"
TODDLERBOT_CUROBO_URDF = str(_REPO_ROOT / "asset" / "toddlerbot_2xm_gripper" / "toddlerbot_curobo.urdf")
BASE_LINK = "base_link"
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]
BOX_SPHERE_RADIUS_SCALE = 1.0
# Toddlerbot's 4.5 mm finger/foot box features otherwise exceed global 500-sphere budget.
# 3 cm axial spacing preserves their envelope while keeping full-body source collision enabled.
MIN_SPHERE_GAP = 0.03
# Merge only pairs whose unshared caps are <=10 cm³; replacement grows radius <=35%.
MAX_SPHERE_MERGE_RADIUS_GROWTH = 1.35
MAX_SPHERE_NONOVERLAP_VOLUME_M3 = 1e-5

LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw_driven",
    "left_elbow_roll",
    "left_elbow_yaw_driven",
    "left_wrist_pitch_driven",
    "left_wrist_roll",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("left_", "right_") for name in LEFT_ARM_JOINT_NAMES)
_ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_SET = frozenset(_ARM_JOINT_NAMES)
_DRIVE_FROM_DRIVEN = {
    "left_shoulder_yaw_driven": "left_shoulder_yaw_drive",
    "left_elbow_yaw_driven": "left_elbow_yaw_drive",
    "left_wrist_pitch_driven": "left_wrist_pitch_drive",
    "right_shoulder_yaw_driven": "right_shoulder_yaw_drive",
    "right_elbow_yaw_driven": "right_elbow_yaw_drive",
    "right_wrist_pitch_driven": "right_wrist_pitch_drive",
}


def _load_model() -> mujoco.MjModel:
    """Compile source MJCF holding endpoint sites, source collisions, and equality coupling."""
    return mujoco.MjModel.from_xml_path(str(TODDLERBOT_MJCF))


def apply_source_drive_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Write sibling gearbox coordinates implied by ToddlerBot's physical arm outputs.

    Planner c-space contains serial ``*_driven`` joints.  The source MJCF instead couples each
    one to a sibling ``*_drive`` gearbox through ``driven = -drive``.  MuJoCo FK and contact
    checks must populate that branch explicitly because equality constraints do not rewrite
    ``qpos`` during ``mj_forward``.
    """
    for driven_name, drive_name in _DRIVE_FROM_DRIVEN.items():
        driven_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, driven_name)
        drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, drive_name)
        assert driven_id >= 0 and drive_id >= 0, f"missing source gearbox pair: {driven_name}"
        driven_qpos = int(model.jnt_qposadr[driven_id])
        drive_qpos = int(model.jnt_qposadr[drive_id])
        data.qpos[drive_qpos] = -data.qpos[driven_qpos]


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None, f"unnamed MuJoCo body {body_id}"
    return name


def _prune_below_knee(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    """Drop collision links strictly below either knee for arm-only reachability."""
    knee_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_knee_link", "right_knee_link")
    }
    assert -1 not in knee_ids, "Toddlerbot knee body missing from source MJCF"

    def is_below_knee(body_id: int) -> bool:
        parent_id = int(model.body_parentid[body_id])
        while parent_id:
            if parent_id in knee_ids:
                return True
            parent_id = int(model.body_parentid[parent_id])
        return False

    return {
        link_name: spheres
        for link_name, spheres in collision_spheres.items()
        if not is_below_knee(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name))
    }


def _lock_nonarm_joints(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, float]:
    """Lock independent non-arm joints on collision/tool chains only.

    cuRobo omits visual-only branches from its reduced tree. Passing locks for those absent
    branches fails its loader, so retain only joints ancestral to a collision-bearing body or
    tool site.
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
                joint_id = int(model.body_jntadr[body_id])
                needed_joints.add(joint_id)
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
    """Ignore only adjacent links and source-rest sphere overlaps."""
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


def _arm_cspace(model: mujoco.MjModel, arm_joint_home: Mapping[str, float] | None) -> dict:
    """Build fourteen-actuator cspace from source limits and optional source-coordinate home."""
    home = {} if arm_joint_home is None else dict(arm_joint_home)
    unknown = set(home) - _ARM_JOINT_SET
    assert not unknown, f"arm_joint_home has non-actuated joints: {sorted(unknown)}"
    default_position = []
    for name in _ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"actuated arm joint {name!r} missing from source MJCF"
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
    """Build source-faithful collision-aware Toddlerbot config for arm-only IK."""
    model = _load_model()
    if not pathlib.Path(TODDLERBOT_CUROBO_URDF).exists():
        # Tracked artifact, not a generated one: `export_mjspec_to_urdf.py` accepts only
        # humanoid_v21/g1/fourier_gr3/pal_talos, so restoring from git is the only recovery.
        # Registering an exporter entry would need a compile snapshot, and any drift from the
        # tracked file moves the published ToddlerBot numbers -- don't fabricate a replacement.
        raise FileNotFoundError(
            f"missing cuRobo URDF: {TODDLERBOT_CUROBO_URDF}; restore it with "
            "`git checkout -- asset/toddlerbot_2xm_gripper/toddlerbot_curobo.urdf`"
        )
    spheres = build_collision_spheres(
        model,
        min_gap=MIN_SPHERE_GAP,
        box_radius_scale=BOX_SPHERE_RADIUS_SCALE,
        box_inscribed=True,
        skip_mesh=True,  # only meshes are the locked ankle-roll feet, pruned below-knee below
    )
    spheres = merge_strongly_overlapping_spheres(
        spheres,
        max_radius_growth=MAX_SPHERE_MERGE_RADIUS_GROWTH,
        max_nonoverlap_volume=MAX_SPHERE_NONOVERLAP_VOLUME_M3,
    )
    spheres = _prune_below_knee(model, spheres)
    lock_joints = _lock_nonarm_joints(model, spheres)
    for frame in TOOL_FRAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame) >= 0
    print(
        f"[curobo] Toddlerbot collision spheres: {len(spheres)} links / "
        f"{sum(len(values) for values in spheres.values())} spheres | "
        f"lock_joints: {len(lock_joints)} | active arm DOFs: {len(_ARM_JOINT_NAMES)}"
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": TODDLERBOT_CUROBO_URDF,
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
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "TODDLERBOT_CUROBO_URDF",
    "TODDLERBOT_MJCF",
    "TOOL_FRAMES",
    "apply_source_drive_coupling",
    "build_robot_cfg_dict",
]
