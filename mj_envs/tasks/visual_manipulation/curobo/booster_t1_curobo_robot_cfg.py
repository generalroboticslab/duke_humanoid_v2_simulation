"""cuRobo arm-only configuration for Booster T1's source MJCF.

Booster T1 has 23 actuated revolute joints. Workspace planning uses four physical arm outputs
per side (Shoulder_Pitch, Shoulder_Roll, Elbow_Pitch, Elbow_Yaw). The hand link has no
sub-joints/fingers, so there is no sibling drive branch and no source-motor equality
writeback — the planner c-space is the four source revolute coordinates directly.

Floating root is `base_link` (renamed from upstream `Trunk` to match the root-frame convention used here;
all child bodies retain their source-relative poses). Standing planning origin = (0, 0, 0.665)
matching the home keyframe; cuRobo seeds the IK solver from this pose, not the body's
default Z=0.7. Two fixed tool sites at the hand terminal: ``end_effector_{L,R}_site``. Head
``H2`` carries a co-located ``head_cam`` + ``head_cam_site`` for the dynamic-actuated visibility
scoring path (T1 RealSense version = Intel RealSense D455; a declared envelope — not
calibrated).

Collision spheres built from source group 3: trunk box, H2 sphere, elbow/hand cylinders,
hip-yaw + shank cylinders, foot boxes. Volume merge policy uses absolute unshared volume
with bounded enclosing-radius growth so the source envelopes are preserved exactly. Below-hip
collision is pruned (knee/ankle are well outside the arm reach envelope and add bulk cuRobo
must broadphase against on every solve).
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
from mj_envs.utils.mj_collision_spheres import build_collision_spheres  # noqa: E402
try:  # absent in this clone; only the cuRobo-solve path calls it, the visibility scorer does not
    from mj_envs.utils.mj_collision_spheres import merge_strongly_overlapping_spheres
except ImportError:
    def merge_strongly_overlapping_spheres(*_a, **_k):
        raise NotImplementedError(
            "merge_strongly_overlapping_spheres is missing from this clone's mj_collision_spheres.py"
        )

BOOSTER_T1_MJCF = _REPO_ROOT / "asset" / "booster_t1" / "t1.xml"
BOOSTER_T1_CUROBO_URDF = str(_REPO_ROOT / "asset" / "booster_t1" / "booster_t1_curobo.urdf")
BASE_LINK = "base_link"
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]
BOX_SPHERE_RADIUS_SCALE = 0.75  # Berkeley-style; use full inscribed boxes + tighter merge.
MIN_SPHERE_GAP = 0.03
MAX_SPHERE_MERGE_RADIUS_GROWTH = 1.35
MAX_SPHERE_NONOVERLAP_VOLUME_M3 = 1e-5
HOME_ROOT_POS = (0.0, 0.0, 0.665)  # Trunk/now-base_link standing Z from home keyframe.

LEFT_ARM_JOINT_NAMES = (
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Left_Wrist_Pitch",
    "Left_Wrist_Yaw",
    "Left_Hand_Roll",
)
RIGHT_ARM_JOINT_NAMES = tuple(name.replace("Left_", "Right_") for name in LEFT_ARM_JOINT_NAMES)
_ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_SET = frozenset(_ARM_JOINT_NAMES)


def _load_model() -> mujoco.MjModel:
    """Compile source MJCF holding endpoint sites, head camera, and source collisions."""
    return mujoco.MjModel.from_xml_path(str(BOOSTER_T1_MJCF))


def apply_source_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """T1 has no sibling drive/equality coupling; planner c-space maps 1:1 to source qpos."""


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None, f"unnamed MuJoCo body {body_id}"
    return name


def _prune_below_hip(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    """Drop collision links strictly below either hip_yaw for arm-only reachability.

    T1's source defines hip_yaw collision only (no knee/ankle/foot collision sphere kept),
    but prune explicitly here so future source edits adding knee/ankle don't silently inflate
    the cuRobo broadphase.
    """
    hip_yaw_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("Hip_Yaw_Left", "Hip_Yaw_Right")
    }
    assert -1 not in hip_yaw_ids, "T1 hip_yaw body missing from source MJCF"

    def is_below_hip(body_id: int) -> bool:
        parent_id = int(model.body_parentid[body_id])
        while parent_id:
            if parent_id in hip_yaw_ids:
                return True
            parent_id = int(model.body_parentid[parent_id])
        return False

    return {
        link_name: spheres
        for link_name, spheres in collision_spheres.items()
        if not is_below_hip(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name))
    }


def _lock_nonarm_joints(
    model: mujoco.MjModel, collision_spheres: dict[str, list[dict]]
) -> dict[str, float]:
    """Lock independent non-arm joints on collision/tool chains only.

    cuRobo omits visual-only branches from its reduced tree. Passing locks for those absent
    branches fails its loader, so retain only joints ancestral to a collision-bearing body or
    tool site. T1 non-arm actuated joints are: AAHead_yaw, Head_pitch, Waist, six per leg.
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
    """Ignore only adjacent links and any source-rest sphere overlaps (defensive)."""

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
    """Build eight-actuator cspace from source limits and optional source-coordinate home."""
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
    """Build source-faithful collision-aware Booster T1 config for arm-only IK."""
    model = _load_model()
    if not pathlib.Path(BOOSTER_T1_CUROBO_URDF).exists():
        # Tracked artifact, not a generated one: `export_mjspec_to_urdf.py` accepts only
        # humanoid_v21/g1/fourier_gr3/pal_talos, so restoring from git is the only recovery.
        # Registering an exporter entry would need a compile snapshot, and any drift from the
        # tracked file moves the published T1 numbers -- don't fabricate a replacement.
        raise FileNotFoundError(
            f"missing cuRobo URDF: {BOOSTER_T1_CUROBO_URDF}; restore it with "
            "`git checkout -- asset/booster_t1/booster_t1_curobo.urdf`"
        )
    spheres = build_collision_spheres(
        model,
        min_gap=MIN_SPHERE_GAP,
        box_radius_scale=BOX_SPHERE_RADIUS_SCALE,
        box_inscribed=True,
    )
    spheres = merge_strongly_overlapping_spheres(
        spheres,
        max_radius_growth=MAX_SPHERE_MERGE_RADIUS_GROWTH,
        max_nonoverlap_volume=MAX_SPHERE_NONOVERLAP_VOLUME_M3,
    )
    spheres = _prune_below_hip(model, spheres)
    lock_joints = _lock_nonarm_joints(model, spheres)
    for frame in TOOL_FRAMES:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame) >= 0
    print(
        f"[curobo] Booster T1 collision spheres: {len(spheres)} links / "
        f"{sum(len(values) for values in spheres.values())} spheres | "
        f"lock_joints: {len(lock_joints)} | active arm DOFs: {len(_ARM_JOINT_NAMES)}"
    )
    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": localize_urdf(BOOSTER_T1_CUROBO_URDF),
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
    "BOOSTER_T1_CUROBO_URDF",
    "BOOSTER_T1_MJCF",
    "BOX_SPHERE_RADIUS_SCALE",
    "HOME_ROOT_POS",
    "LEFT_ARM_JOINT_NAMES",
    "RIGHT_ARM_JOINT_NAMES",
    "TOOL_FRAMES",
    "apply_source_coupling",
    "build_robot_cfg_dict",
]
