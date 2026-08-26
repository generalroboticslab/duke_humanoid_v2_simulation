"""Build a cuRobo ``RobotCfg`` dict for the full-body humanoid_v21 URDF (Phase 2 of
plan/MJCF_TO_URDF_MINIMUM_LOSS_PLAN.md).

Collision spheres come directly from the resolved model's MuJoCo **collision** geoms
(group 3 + group 5 primitives) via ``mj_collision_spheres.build_collision_spheres`` -- NOT from
a MORPHIT fit of the visual meshes. This replaces the previous RobotBuilder + yourdfpy + ~60s
MORPHIT + JSON-cache path: the primitives are analytic, pose-independent, and fast, so no cache
is needed. The full robot (torso + waist + legs + arms + grippers + head cam) is covered;
planner DOFs stay arm-only via ``lock_joints`` -- every non-arm joint is pinned to the resolved
``mj_model.qpos0`` value.

Self-collision ignore is **self-contained** (no cuRobo collision-matrix sampling): the kinematic
tree's parent-child adjacency (links always touch at their shared joint) UNION the Mink
self-collision set (``_ADJACENT_BODY_PAIRS | EXCLUDED_COLLISION_PAIRS``). A leftover false
positive is fixed by adding the pair to ``humanoid_v21_collision_exclusions.py`` -- transparent
and debuggable, no opaque sampler.

The URDF (``HUMANOID_V21_URDF``) is still the kinematics source ``RobotCfg.create`` loads (links,
joints, tool frames); only the collision geometry now bypasses it.

Verification (forward-kinematics cross-check, reach replay, sphere viewers) lives in
``mj_envs/tasks/visual_manipulation/test/curobo_humanoid_v21_phase{2,3}_verify.py``,
``curobo_view_collision_spheres.py``, and ``view_collision_geom_to_sphere.py``. IK/trajopt patch lives in
``build_motion_planner_kwargs``.

Run (debug):
    /home/grl/repo/micromamba/envs/py312/bin/python -c "from tasks.visual_manipulation.curobo.ik_curobo_robot_cfg \\
        import build_robot_cfg_dict_from_urdf; cfg = build_robot_cfg_dict_from_urdf(); \\
        cs = cfg['robot_cfg']['kinematics']['collision_spheres']; \\
        print(len(cs), 'links', sum(len(v) for v in cs.values()), 'spheres')"
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import sys
import tempfile
from collections.abc import Mapping

import curobo
import mujoco
import numpy as np
import torch
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

from tasks.visual_manipulation.curobo.urdf_localize import localize_urdf
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from asset_zoo.humanoid_v21.humanoid_v21_constants import (  # noqa: E402
    HOME_KEYFRAME,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    get_humanoid_v21_robot_cfg,
)
from mjlab.entity.entity import Entity  # noqa: E402

from mj_envs.utils.humanoid_v21_collision_exclusions import EXCLUDED_COLLISION_PAIRS  # noqa: E402
from mj_envs.utils.ik_mink import _ADJACENT_BODY_PAIRS  # noqa: E402
from mj_envs.utils.mj_collision_spheres import build_collision_spheres  # noqa: E402

HUMANOID_V21_URDF = str(_REPO_ROOT / "asset" / "duke_v2" / "humanoid_v21" / "humanoid_v21_curobo.urdf")


def _urdf_for_rig(head_camera: str) -> str:
    """cuRobo URDF for a head-camera rig. The URDF supplies the KINEMATIC CHAIN (links/joints);
    collision spheres come separately off the compiled MuJoCo model. Both must describe the same
    rig -- naming a link in the sphere set that the URDF's parent map lacks fails at
    ``RobotCfg.create`` with ``Link <name> not found in parent map``.

    "none" reuses the shipped dual URDF on purpose: the camera links exist in the chain but carry no
    collision spheres and no active DOF (they are locked), so the solved geometry is camera-free.
    The K=1/K=3 rigs rename/add camera links, so they need their own export
    (``export_mjspec_to_urdf.py --head-camera actuated_{single,triple} --output asset/duke_v2/humanoid_v21/rig_*``)."""
    variant = {"actuated_single": "rig_single", "actuated_triple": "rig_triple"}.get(head_camera)
    if variant is None:
        return HUMANOID_V21_URDF
    return str(_REPO_ROOT / "asset" / "duke_v2" / "humanoid_v21" / variant / "humanoid_v21_curobo.urdf")
BASE_LINK = "base_link"   # kept (viewer imports it)
# Tool frames = the grasp SITES (URDF fixed links `end_effector_{L,R}_site`, ~11.8cm out on the
# flange x-axis = the jaw grasp point), NOT the flange body origin `end_effector_{L,R}`. cuRobo
# plans the actual grasp point to the target; using the body origin would leave the gripper 11.8cm
# short. Goals against this cfg MUST declare these exact frame names.
TOOL_FRAMES = ["end_effector_L_site", "end_effector_R_site"]

# Per-side arm chain: 7 actuated joints. The exporter's resolved URDF names them with the
# ``left_/right_`` prefix and a ``_joint`` suffix; that's the form cuRobo's joint loader
# exposes and it equals the MuJoCo joint name. Any joint NOT in this set is a non-arm joint
# (legs + waist + head camera gimbal + gripper racks) and must end up in lock_joints.
_ARM_JOINT_CSPACE_NAMES = (*LEFT_ARM_JOINT_NAMES, *RIGHT_ARM_JOINT_NAMES)
_ARM_JOINT_NAMES = frozenset(_ARM_JOINT_CSPACE_NAMES)

# Planning-only arm posture in MuJoCo qpos coordinates. It seeds the Phase-3 cuRobo
# cspace and the Phase-4 planned-route pre-position; it is deliberately separate from
# HOME_KEYFRAME, which remains the frozen-policy reset posture.
_REACH_READY_ARM_JOINT_POS = {
    "right_shoulder_1_joint": 1.3740, "left_shoulder_1_joint": -1.3740,
    "right_shoulder_2_joint": 0.1870, "left_shoulder_2_joint": -0.1870,
    "right_shoulder_3_joint": -0.1310, "left_shoulder_3_joint": 0.1310,
    "right_elbow_joint": -1.9290, "left_elbow_joint": 1.9290,
    # wrist_1 rotated -90deg from the old 1.2403 to swing the grippers forward instead of
    # across the chest. With the mirror signs below, both arms land 28deg off base +x; that is
    # the best wrist_1 alone can do (sweep optimum is -0.1916 at 26.1deg), the rest of the
    # residual is set by the shoulder/elbow values.
    "right_wrist_1_joint": -0.3305, "left_wrist_1_joint": 0.3305,
    # left = +right for this pair only: left_wrist_2_joint's MJCF frame carries rpy=[0,0,180deg]
    # where the right side carries [0,0,0], which flips that axis's world direction. Negating it
    # like the others left the arms 0.126 m out of mirror, and made the two arms respond to
    # wrist_1 in opposite directions -- no wrist_1 value pointed BOTH forward. Every other pair
    # is equal-and-opposite. Signs are derived from the compiled model by
    # asset_zoo/humanoid_v21/generate_arm_joint_pos.py:arm_mirror_signs -- re-run it after any
    # per-side joint-frame change rather than assuming -1.
    "right_wrist_2_joint": 1.4546, "left_wrist_2_joint": 1.4546,
    "right_wrist_3_joint": 0.0090, "left_wrist_3_joint": -0.0090,
}
HUMANOID_ARM_JOINT_HOME = _REACH_READY_ARM_JOINT_POS
assert set(HUMANOID_ARM_JOINT_HOME) == _ARM_JOINT_NAMES


def _build_mj_model_resolved(head_camera: str = "actuated") -> mujoco.MjModel:
    """The exporter's resolved humanoid_v21 variant (head-camera + parallel_gripper actuated).
    ``qpos0`` encodes the home snapshot for the locked camera + gripper-rack joints; collision
    geom poses (body-frame) are read straight off the compiled model.

    ``head_camera`` selects the camera rig and is NOT cosmetic: the modules carry collision spheres
    (26 of 104 on the dual rig) that the IK solver must avoid, so a different rig yields a different
    reachable set. Default keeps the shipped dual rig, so existing callers are unchanged."""
    return Entity(
        get_humanoid_v21_robot_cfg(head_camera=head_camera, end_effector="actuated",
                                   hand="parallel_gripper")
    ).compile()


def _body_name(mj_model, b: int) -> str:
    return mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)


def _lock_joints_from_resolved(mj_model, joint_pos: Mapping[str, float]) -> dict[str, float]:
    """Lock non-arm joints at configured home, not raw MuJoCo ``qpos0``.

    ``EntityCfg.InitialStateCfg`` is applied by mjlab at reset and does not alter compiled
    ``qpos0``. Resolve its exact-or-regex joint map here so Phase-3 cuRobo collision geometry,
    including parallel-gripper racks, matches physical home state. Unlisted joints retain qpos0.
    """
    home = {
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid):
        float(mj_model.qpos0[mj_model.jnt_qposadr[jid]])
        for jid in range(mj_model.njnt)
        if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid) is not None
    }
    for pattern, value in joint_pos.items():
        for jname in home:
            if re.fullmatch(pattern, jname):
                home[jname] = float(value)

    lock: dict[str, float] = {}
    for jid in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname is None or jname in _ARM_JOINT_NAMES:
            continue
        if mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        lock[jname] = home[jname]
    return lock


# Arm joint accel/jerk caps fed to cuRobo trajopt time-parametrization. Lowered from cuRobo's
# industrial default (10.0 / 500.0) to gentle the peak reaction wrench the moving arm exerts on the
# FLOATING base: reaction force ~ arm_mass * joint_accel and jerk spikes are what kick the base
# hardest, so a world-stationary target does not drift in the base frame mid-reach (the quasi-static
# plan assumption holds). Value chosen at the measured KNEE of the time-vs-momentum frontier
# (probe sweep 10/6/4/3/2, humanoid bimanual left_right_close): 4.0/200.0 gives peak linear momentum
# -46% and peak jerk -69% vs baseline at only 1.67x trajectory time (2.5 vs 1.5 s). Lowered a further
# step to 3.0/150.0 to gentle the VISIBLE planner + MPC drift-tracking trajectory (user: too aggressive):
# 3.0 costs marginal extra time for a smoother path, staying ABOVE the 2.0 knee where the optimizer
# BACKFIRES (wiggles -- CoM path 37->54 mm, momentum rises). NOTE the reach's CoM *displacement*
# (~36 mm) is invariant to these caps (task-inherent, already a near-straight CoM path); caps cut
# momentum-RATE, not CoM change -- reducing that needs a path/joint lever, not slowing. cuRobo has no
# centroidal-momentum cost; joint accel/jerk is the proxy. Proximal-vs-distal cspace weighting NOT
# used: uniform null_space_weight was already found inert for this arm (MEMORY 2026-07-10 travel study).
# RAISED 2026-08-01 from 3.0/150.0 to 6.0/300.0 after the two-stage pregrasp shipped. Screened 5 seeds
# (42-46, v2 left_right_close, dynamic+walk+camera) at 1x vs 2x: EXTEND -19%, RETRACT -30%, t_sim -24%,
# energy -19%, 5/5 SUCCESS, penetration <=0.001 m, zero knock witnesses either arm. The base-disturbance
# probe (the quantity these caps actually protect) found no penalty: peak base speed +4%, drift rate +15%
# but confounded by the shorter span, and energy FELL, which argues against a hidden torque cost. The
# earlier 3.0 choice was made when EXTEND drove straight at the cube in one shot, so approach speed and
# knock risk were the same lever; the pregrasp standoff split those, and the descent segment is short
# enough that the raised cap buys time without raising contact speed.
# These MUST be set HERE, at build time, because two consumers read them and only one reads them live --
# the retimer (``solver_trajopt.compute_trajectory_dt``) hits ``transition_model.max_acceleration`` every
# solve, while the bound cost that decides feasibility keeps a ``bounds.clone()`` frozen at construction
# (``cost_cspace_cfg.set_bounds``). Mutating them around a ``plan_pose`` call retimes the trajectory past
# the frozen cost bound and every seed fails feasibility -- a dead lever, do not retry it.
ARM_MAX_ACCELERATION = 6.0
ARM_MAX_JERK = 300.0
# Velocity is capped SEPARATELY from accel/jerk because they bound different things: accel/jerk bound the
# reaction wrench on the floating base, velocity bounds the peak SPEED the arm is ever seen moving at. The
# URDF declares 10 rad/s on every joint, which the arm comes nowhere near: measured peak on the retract
# solve is 0.823 rad/s, because at these accel caps the trajectory accelerates and decelerates without ever
# cruising. The default is therefore a HEADROOM CEILING, not an active constraint -- it bounds what the
# raised accel caps could produce on a longer trajectory without touching any motion measured today.
# A SCALE, not an absolute, because that is the only form cuRobo's ``CSpaceConfig`` accepts for velocity
# (unlike accel/jerk, which it takes as absolutes).
# GOTCHA -- THE SCALE IS APPLIED TWICE, so the effective limit is ``10 * scale**2``, NOT ``10 * scale``:
# 0.5 measures as 2.5 rad/s, not 5.0 (verified via ``LIMIT_PROBE=1``). ``scale_joint_limits`` runs in
# ``KinematicsParams.__post_init__`` and MULTIPLIES the limits; ``KinematicsReducer`` then rebuilds that
# dataclass (``kinematics_reducer.py:532``) handing it the ALREADY-SCALED limits together with the same
# ``velocity_scale``, so post-init scales them a second time. Any config with ``lock_joints`` goes through
# the reducer, and this one locks 13 non-arm joints, so the squaring is unconditional here. Accel/jerk are
# immune only because they are passed as absolutes (overwrite, idempotent) with their *_scale left at 1.0.
# Do NOT "fix" this by pre-dividing -- write the intent here and read the effective value off LIMIT_PROBE.
ARM_VELOCITY_SCALE = float(os.environ.get("ARM_VEL_SCALE", "0.5"))     # => 2.5 rad/s effective


def _arm_cspace_from_home(
    mj_model,
    arm_joint_home: Mapping[str, float] | None = None,
) -> dict[str, list[float] | list[str] | float]:
    """cuRobo active-arm cspace default from a configurable MuJoCo home qpos map.

    ``None`` uses ``HUMANOID_ARM_JOINT_HOME``: the Phase-3/Phase-4 planning posture.
    Input values are MuJoCo qpos coordinates, so each active arm
    joint is converted to cuRobo coordinates by subtracting compiled ``mj_model.qpos0``. This is
    required for joints with XML ``ref`` folded into qpos0 (currently wrist_3); using raw home
    values there would bias cuRobo about 90 deg off the actual HOME pose.

    Home semantics: an active arm joint the map names is placed at that qpos; a redundant DOF the
    map OMITS homes to its compiled ``qpos0`` (the joint's sim rest -- e.g. ``wrist_3`` rests at its
    XML ``ref``, ``shoulder_3`` at 0). This is cuRobo-default 0 for the omitted joint, NOT raw
    mjlab-zero. The hanging-arm HOME_KEYFRAME intentionally lists only the primary DOFs and leaves
    the redundant shoulder_3/wrist_3 at rest, so a hard-fail on their absence (the earlier guard)
    wrongly rejected a valid home. cuRobo-coord conversion (subtract qpos0) is still applied per
    named joint -- required for ref-folded joints like wrist_3. Filled joints are logged, not
    silent, so a wholly-forgotten home is still visible rather than a silent all-rest fallback.
    """
    home = HUMANOID_ARM_JOINT_HOME if arm_joint_home is None else arm_joint_home
    default_joint_position, filled = [], []
    for jname in _ARM_JOINT_CSPACE_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"active arm joint {jname!r} not found in resolved mj_model"
        qpos0 = float(mj_model.qpos0[mj_model.jnt_qposadr[jid]])
        if jname in home:
            default_joint_position.append(float(home[jname]) - qpos0)
        else:
            default_joint_position.append(0.0)   # omitted DOF rests at compiled qpos0
            filled.append(jname)
    if filled:
        print(f"[curobo] arm cspace: {len(filled)} unspecified arm joints homed to compiled rest "
              f"(qpos0): {filled}")
    return {
        "joint_names": list(_ARM_JOINT_CSPACE_NAMES),
        "default_joint_position": default_joint_position,
        "cspace_distance_weight": [1.0] * len(_ARM_JOINT_CSPACE_NAMES),
        "null_space_weight": [1.0] * len(_ARM_JOINT_CSPACE_NAMES),
        "max_acceleration": ARM_MAX_ACCELERATION,
        "max_jerk": ARM_MAX_JERK,
        "velocity_scale": ARM_VELOCITY_SCALE,
    }


def _expand_gripper_tuple(body) -> list[str]:
    """Mink authored some collision "bodies" as gripper tuples (``_GRIPPER_L/R`` = the two
    rack links per side). The exporter's URDF keeps those racks as real links, so a tuple
    expands to its member link names; a plain name passes through."""
    return list(body) if isinstance(body, tuple) else [body]


def _rest_overlapping_link_pairs(mj_model, collision_spheres: dict[str, list[dict]]) -> set[frozenset]:
    """Link pairs whose collision spheres already OVERLAP at the neutral (``qpos0``) config.

    The rest pose is collision-free by construction, so any sphere overlap there is a structural
    artifact -- adjacent/near-adjacent links (or coarsened bundles whose enclosing capsule bulges
    into a neighbor) that are permanently in contact and can never be a real dynamic collision.
    cuRobo's own sphere generators build the ignore set exactly this way. Critically, the leg
    links are LOCKED at qpos0, so their mutual overlap is present in *every* IK/plan config;
    leaving it un-ignored makes cuRobo mark every seed self-colliding (0 IK success, the phase-3
    reach-gate failure). Computed on the same spheres cuRobo consumes, so it self-maintains as the
    coarsening/sphere policy changes.
    """
    data = mujoco.MjData(mj_model)
    data.qpos[:] = mj_model.qpos0
    mujoco.mj_forward(mj_model, data)
    world: dict[str, list[tuple[np.ndarray, float]]] = {}
    for bname, sl in collision_spheres.items():
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        pos, rot = data.xpos[bid], data.xmat[bid].reshape(3, 3)
        world[bname] = [(pos + rot @ np.asarray(s["center"]), float(s["radius"])) for s in sl]
    bodies = list(world)
    pairs: set[frozenset] = set()
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a, b = bodies[i], bodies[j]
            if any(np.linalg.norm(ca - cb) < ra + rb
                   for ca, ra in world[a] for cb, rb in world[b]):
                pairs.add(frozenset((a, b)))
    return pairs


def build_self_collision_ignore(mj_model, collision_spheres: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Self-contained cuRobo self-collision ignore over the collision-link set.

    Three sources, unioned (both directions -- cuRobo reads the dict as a matrix):
      1. Kinematic tree parent-child adjacency: adjacent links share a joint and always touch,
         so their spheres must never count as a collision. Covers the WHOLE body (legs, head
         camera, grippers) -- the Mink set below only spans the arm/torso region.
      2. Mink self-collision pairs (``_ADJACENT_BODY_PAIRS | EXCLUDED_COLLISION_PAIRS``): the
         non-adjacent arm/gripper pairs a 30k-sample sweep proved never usefully collide.
      3. Rest-pose sphere overlaps (``_rest_overlapping_link_pairs``): 2-hop structural neighbors
         and coarsened locked-leg bundles whose spheres bulge into each other at qpos0. Source 1
         only catches DIRECT parent-child; the coarsened foot/shank/hip and camera-base spheres
         overlap across a 2-hop gap and would otherwise make every locked-leg config self-collide.

    Unknown link names (e.g. a geomless frame) are harmless -- cuRobo silently ignores them.
    """
    ignore: dict[str, list[str]] = {}

    def add(a: str, b: str) -> None:
        if a == b:
            return
        ignore.setdefault(a, [])
        if b not in ignore[a]:
            ignore[a].append(b)

    def add_both(a: str, b: str) -> None:
        add(a, b)
        add(b, a)

    for b in range(1, mj_model.nbody):
        p = mj_model.body_parentid[b]
        if p == 0:
            continue
        add_both(_body_name(mj_model, b), _body_name(mj_model, p))

    for frozen_pair in _ADJACENT_BODY_PAIRS | EXCLUDED_COLLISION_PAIRS:
        a_raw, b_raw = tuple(frozen_pair)
        for a in _expand_gripper_tuple(a_raw):
            for b in _expand_gripper_tuple(b_raw):
                add_both(a, b)

    for frozen_pair in _rest_overlapping_link_pairs(mj_model, collision_spheres):
        a, b = tuple(frozen_pair)
        add_both(a, b)
    return ignore


# Locked leg links modeled as multi-capsule cages (shank/foot = 6 parallel capsules each,
# ~300 of 392 spheres). They sit far from the arm workspace and are pinned at qpos0, so their
# only planner role is a coarse self-collision bound. Collapsing each bundle to one enclosing
# capsule cuts ~75% of per-plan collision cost and ~halves planner warmup (profiled) with no
# reach-relevant fidelity loss.
_COARSEN_LEG_LINKS = r"shank|foot"

# Collision-exclusion boundary links: everything STRICTLY BELOW each (its kinematic subtree, the
# boundary link itself kept) is removed from the collision model. General cut-point form -- name the
# highest link whose descendants the arm can never reach, the mj tree supplies the rest -- rather
# than enumerating individual leaf-link names. Here: the lower legs. The arm reaching a front/rear
# table can never bring the gripper near the shin/foot, and every leg joint is locked at qpos0
# (static), so those spheres only add a constant self-collision bound the ignore-set already zeroes.
# Excluding the two hip_3 subtrees drops knee+shank+ankle+foot (both sides) ~= 22% of the 108 spheres
# off every per-plan collision query with no reach-relevant fidelity loss; hip_2/hip_3 stay (closest
# leg links to the torso/arm sweep).
_COLLISION_EXCLUDE_SUBTREES = ("hip_3_L", "hip_3_R")


def _strict_descendant_bodies(mj_model, root_names: tuple[str, ...]) -> set[str]:
    """Body names strictly below (descendants of, excluding) each root in the mj kinematic tree.
    A body qualifies if any proper ancestor is a root -- the roots themselves are NOT included."""
    roots = {mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in root_names}
    out: set[str] = set()
    for b in range(1, mj_model.nbody):
        p = mj_model.body_parentid[b]
        while p > 0:
            if p in roots:
                out.add(_body_name(mj_model, b))
                break
            p = mj_model.body_parentid[p]
    return out


def _humanoid_v21_cfg(
    load_dynamics: bool,
    arm_joint_home: Mapping[str, float] | None = None,
    head_camera: str = "actuated",
) -> dict:
    mj_model = _build_mj_model_resolved(head_camera)
    spheres = build_collision_spheres(mj_model, coarsen_links=_COARSEN_LEG_LINKS)
    excluded = _strict_descendant_bodies(mj_model, _COLLISION_EXCLUDE_SUBTREES)
    spheres = {ln: sl for ln, sl in spheres.items() if ln not in excluded}
    home_joint_pos = dict(HOME_KEYFRAME.joint_pos)
    if arm_joint_home is not None:
        home_joint_pos.update(arm_joint_home)
    lock_joints = _lock_joints_from_resolved(mj_model, home_joint_pos)
    # An excluded link's subtree is pruned from cuRobo's kinematic chain (only links on a path to a
    # retained collision/tool link survive), so any joint whose CHILD body was excluded no longer
    # exists in the chain -- leaving it in lock_joints makes the loader KeyError. Prune by the same
    # excluded-body set (joint's child = mj jnt_bodyid). Joints whose child is a kept boundary link
    # (hip_1->hip_2, hip_2->hip_3) stay; hip_3->knee and below go with their links.
    lock_joints = {
        j: v for j, v in lock_joints.items()
        if _body_name(mj_model, mj_model.jnt_bodyid[
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)]) not in excluded
    }
    arm_in_lock = [j for j in lock_joints if j in _ARM_JOINT_NAMES]
    assert not arm_in_lock, f"arm joints leaked into lock_joints: {arm_in_lock}"

    total = sum(len(v) for v in spheres.values())
    n_per_link = sorted(((ln, len(s)) for ln, s in spheres.items()), key=lambda kv: -kv[1])
    print(f"[curobo] collision-geom spheres: {len(spheres)} links / {total} spheres")
    for ln, n in n_per_link[:10]:
        print(f"  {ln}: {n}")
    print(f"[curobo] lock_joints: {len(lock_joints)} non-arm joints pinned to configured home")
    active = sum(1 for j in _ARM_JOINT_NAMES
                 if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) >= 0)
    print(f"[curobo] active DOFs (arm joints, NOT locked): {active}")

    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": TOOL_FRAMES,
                "urdf_path": localize_urdf(_urdf_for_rig(head_camera)),
                "asset_root_path": "/",
                "collision_link_names": list(spheres.keys()),
                "collision_spheres": spheres,
                "self_collision_ignore": build_self_collision_ignore(mj_model, spheres),
                "self_collision_buffer": {},
                "lock_joints": lock_joints,
                "cspace": _arm_cspace_from_home(mj_model, arm_joint_home),
            },
            "load_dynamics": load_dynamics,
        }
    }


def build_robot_cfg_dict_from_urdf(
    robot: str = "humanoid_v21",
    load_dynamics: bool = False,
    arm_joint_home: Mapping[str, float] | None = None,
    head_camera: str = "actuated",
) -> dict:
    """Phase 2 entry point. Single-signature contract:
    ``{robot_cfg: {kinematics: {...}, load_dynamics: bool}}``. Collision spheres are the
    resolved model's MuJoCo collision primitives (see module docstring). ``arm_joint_home`` is an
    optional named MuJoCo-qpos map for the active arms; default is
    ``HUMANOID_ARM_JOINT_HOME``. ``HOME_KEYFRAME`` remains frozen-policy reset state.
    ``head_camera`` selects the camera rig whose collision spheres the solver must avoid; it changes
    the reachable set, so per-rig results must not share a payload."""
    if robot == "humanoid_v21":
        return _humanoid_v21_cfg(load_dynamics, arm_joint_home, head_camera)
    raise NotImplementedError(
        f"G1 cfg deferred -- plan/MJCF_TO_URDF_MINIMUM_LOSS_PLAN.md G1 scope; "
        f"blocker is memory/g1_missing_grasp_site.md"
    )


# ---------------------------------------------------------------------------
# IK/trajopt tool-pose patch: enforce the FULL goal orientation per goalset candidate. The reach
# goal is a *goalset* of discrete full-6-DOF grasp candidates (see set_reaching_goalset below);
# cuRobo mins over the set per link, so reachability (e.g. the rear target needing a ~180deg yaw)
# is handled by including that candidate, NOT by masking an axis. The earlier free-yaw mask (yaw
# weight 0) is dropped: with a real grasp planner the candidate carries the exact desired yaw, and
# a global per-axis yaw mask would discard it. "Tolerate some grasp error" is now a single uniform
# angular slack (`_ORIENTATION_TOLERANCE`), not an axis-specific mask.
# See plan/MJCF_TO_URDF_MINIMUM_LOSS_PLAN.md phase-change rationale.
# ---------------------------------------------------------------------------
# Anchor on the INSTALLED curobo via `_src.__path__` (a namespace subpackage that always resolves to
# the real install), not `curobo.__file__`: when an entrypoint's script dir puts this repo's local
# `tasks/visual_manipulation/curobo/` package on sys.path[0], top-level `import curobo` binds to the
# local shadow whose `__file__` has no `content/` tree. `_src.__path__[0]`.parent is the real root.
import curobo._src as _curobo_src  # noqa: E402
_CUROBO_TASK_CFG_ROOT = pathlib.Path(list(_curobo_src.__path__)[0]).parent / "content" / "configs" / "task"
# IK seed count. Per-plan solve time is ~FLAT in seed count (GPU-parallel IK), so seeds cost only
# WARMUP CUDA-graph capture -- a one-shot build cost, NOT per-plan latency. Precision is already
# saturated at 128 (0 fails, gripper-site error unchanged 128..512). What MORE seeds buy is lower
# ARM TRAVEL: the IK topk ranks feasible solutions by pose convergence, not joint distance, so with
# few seeds the returned terminal config can sit in a far grasp-goalset basin (large swing) even when
# a near basin exists. Denser seeding surfaces the near basin. Profiled 2026-07-10 (single-arm
# front/rear, front_back_close): 128->512 cut front max-joint swing 2.86->1.88 rad (-34%), travel L2
# 5.80->3.57 (-38%); rear 2.90->2.63 rad, L2 5.06->4.51. Per-plan time held ~47-51 ms (<=50 ms
# budget); >512 pushes past 50 ms with only marginal further travel gain. num_trajopt_seeds is
# IRRELEVANT to travel (identical L2 at 4/8/12/16) and only adds time -- see below for why it is
# nonetheless raised.
_NUM_IK_SEEDS = 512

# Trajopt seeds are a PARALLEL search; ``_BIMANUAL_PLAN0_MIN_ATTEMPTS`` (curobo/scene.py) is the SERIAL
# one over the same failure. They fix the same defect -- a geometrically feasible bimanual pairing that
# a narrow search reports infeasible, silently demoting the visit to two sequential single-arm reaches --
# and the parallel one is strictly cheaper in wall time, which is the whole reason to prefer it.
#
# The 2026-07-24 note in ``readme_visual_manipulation.md`` chose attempts over seeds on the grounds that
# seeds are baked into the ONE warmed CUDA-graph session and so "tax every solve" (measured then: 4->8
# cost +35% per single-arm replan). Re-measured 2026-08-02 on the DYNAMIC path and the tax does not
# appear at the mission level -- aggregate ``plan_wait_s``, v2 seed 47:
#   bimanual_mixed_front_back_close   4 seeds/150 attempts 1.64 s   ->  8 seeds/30 attempts 0.44 s
#   front_back_far (single-arm heavy) 4 seeds/ 30 attempts 1.30 s   ->  8 seeds/30 attempts 0.58 s
# The per-solve tax is real but smaller than the retry loop it removes, and it shows up even on the cell
# with no bimanual visit at all. The marginal pairing itself lands in 0.30 s at 8 seeds vs 2.17 s of serial
# retry at 4. 16 seeds is WORSE than 8 (0.53 s on the same pairing) -- past 8 the per-solve cost wins, so
# this is a shallow optimum and not a "more is better" knob.
_NUM_TRAJOPT_SEEDS = 8
_POSITION_TOLERANCE = 0.01
# Success gate: cuRobo's convergence rotation_error is the *geometric* angle in RADIANS (the
# `best_angle`, see curobo _src/cost/wp_tool_pose.py:662,679), the min over the goalset of the full
# relative-rotation angle between the reached tool pose and each candidate. This is ONLY the
# acceptance gate -- the cost still pulls fully toward the nearest candidate, so a loose tolerance
# does not slacken alignment when the candidate is reachable; it only avoids rejecting a
# reachability-limited reach that lands a bit off its candidate. Set permissive: ~0.5 rad ~= 28.6
# deg, still inside the 45 deg half-gap between adjacent cube-face candidates (can't cross into the
# wrong candidate's basin).
_ORIENTATION_TOLERANCE = 0.5
# Terminal (at-target) tool-pose per-axis weight factor, order [x, y, z, roll, pitch, yaw]. It
# multiplies the scalar tool_pose weight per axis at the FINAL waypoint. All axes = 1.0: the full
# grasp orientation each goalset candidate specifies is enforced (no masking). Kept as a named
# vector so a future object could deliberately soften a single axis, but the default is strict.
# The non-terminal (approach) factor is left at cuRobo's all-zero default => orientation is free
# until the gripper is at the target.
#
# The factor is injected into BOTH cost paths so they agree: the optimizer cost
# (lbfgs *.yml -> rollout.cost_cfg) shapes the gradient, and the SEPARATE metrics rollout
# (metrics_base.yml -> rollout.convergence_cfg) computes the success-gate rotation_error.
_TERMINAL_POSE_AXES_WEIGHT_FACTOR = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Keep planned collision spheres away from scene obstacles before hard contact.  The physical
# dynamic rollout can lag the joint reference by several millimetres, so the former 10 mm
# activation band let table-skimming plans turn into table contact.  This stays a world-cost
# margin (rather than inflating robot spheres) so the grasped cube and gripper fit geometry are
# unchanged.
WORLD_COLLISION_ACTIVATION_DISTANCE = 0.07


def _load_and_patch_tool_pose(relpath: str, section: str) -> dict:
    """Load a cuRobo yml and inject the terminal per-axis pose weight factor into its tool_pose
    cost under ``rollout.<section>`` (``cost_cfg`` for the lbfgs optimizers, ``convergence_cfg``
    for the metrics rollout). The factor is the strict all-ones default; rotation weight is left at
    the yml's stock value (no override -- the prefer-horizontal lever was rejected, see handoff)."""
    with open(_CUROBO_TASK_CFG_ROOT / relpath) as f:
        cfg = yaml.safe_load(f)
    cfg["rollout"][section]["tool_pose_cfg"]["_terminal_pose_axes_weight_factor"] = list(
        _TERMINAL_POSE_AXES_WEIGHT_FACTOR
    )
    return cfg


def _write_patched_metrics_rollout() -> str:
    """Write the terminal-pose-factor-patched metrics rollout to a temp file, return its abs path.

    The metrics rollout must be a FILE PATH, not a dict: ``MotionPlannerCfg.create`` hands the
    same object to both the IK and trajopt solvers, and ``create_with_component_types`` mutates it
    in place (replacing ``convergence_cfg`` with a built cfg). A shared dict is corrupted after the
    first solver; a path is re-read fresh per solver (``resolve_config``), so each gets its own
    dict. The file is regenerated from the committed cuRobo default on every build (deterministic
    name, overwritten), so it stays reproducible and leaves at most one small file in the tempdir.
    """
    cfg = _load_and_patch_tool_pose("metrics_base.yml", "convergence_cfg")
    path = pathlib.Path(tempfile.gettempdir()) / "curobo_metrics_base_level_yaw.yml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


def build_motion_planner_kwargs() -> dict:
    """Extra kwargs for ``MotionPlannerCfg.create`` for the reach: IK/trajopt optimizer costs AND
    the success-metrics rollout all patched with the terminal per-axis pose factor (full grasp
    orientation enforced per goalset candidate), wider IK seeds, and a uniform orientation
    tolerance (radians) that gates the full grasp-orientation residual. World collision cost
    activates 70 mm before hard contact, leaving tracking slack for dynamic execution without
    inflating gripper/cube collision geometry."""
    return {
        "ik_optimizer_configs": [
            _load_and_patch_tool_pose("ik/lbfgs_ik.yml", "cost_cfg")
        ],
        "trajopt_optimizer_configs": [
            _load_and_patch_tool_pose("trajopt/lbfgs_bspline_trajopt.yml", "cost_cfg")
        ],
        "metrics_rollout": _write_patched_metrics_rollout(),
        "num_ik_seeds": _NUM_IK_SEEDS,
        "num_trajopt_seeds": _NUM_TRAJOPT_SEEDS,
        "position_tolerance": _POSITION_TOLERANCE,
        "orientation_tolerance": _ORIENTATION_TOLERANCE,
        "optimizer_collision_activation_distance": WORLD_COLLISION_ACTIVATION_DISTANCE,
        # Legacy standalone default. Production ``RobotDescriptor`` sessions override this with their
        # emitted grasp-candidate count before graph warmup.
        "max_goalset": _MAX_GOALSET,
        # Unset collision_cache auto-sizes the cuboid cache from the FIRST scene passed to
        # MotionPlannerCfg.create; the session (and its cache) then survives every later
        # update_world call for the process lifetime. A later trial whose scene has MORE
        # cuboids than that first one overflows and raises "cache is full" (hit at 11 for
        # v2_fixed/left_right_close kinematic sweep). Fix the cache size well above any
        # scenario's obstacle count instead of relying on trial-1 obstacle count.
        "collision_cache": {"obb": 32},
    }


# ---------------------------------------------------------------------------
# Candidate-grasp-pose goalset. The reach goal is a SET of full-6-DOF grasp candidates: cuRobo's
# tool-pose cost mins over the goalset per link (wp_tool_pose.py goalset loop), i.e. "reach
# whichever candidate is closest/reachable". Real pipeline: camera -> object pose (base_link) -> an
# (out-of-scope) grasp planner emits candidate grasp poses in the OBJECT frame -> transform to
# base_link -> goalset. Three layers, only the last knows about cubes:
#   set_reaching_goalset  -- object-agnostic tensor assembly (idle link duplicated, reaching link
#                            = the candidates).
#   grasp_poses_to_base   -- object-agnostic rigid transform (object frame -> base_link).
#   cube_grasp_poses_obj  -- the cube stand-in grasp set in the object frame (swappable placeholder
#                            for the real grasp planner).
# ---------------------------------------------------------------------------
_TOPDOWN_FLIP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)  # 180deg about tool +x (approach axis)
# Cube stand-in: 4 level face-grasp yaws x {no-flip, top/down flip} = 8 candidates.
_MAX_GOALSET = 8


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two wxyz quaternions, broadcast over leading dims (last dim = 4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _quat_rotate_vec_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vec(s) ``v`` [..., 3] by wxyz quaternion(s) ``q`` [..., 4]: ``v' = q v q*``."""
    w = q[..., 0:1]
    u = q[..., 1:]                       # [..., 3] (xyz)
    uv = torch.linalg.cross(u, v, dim=-1)
    return v + 2.0 * (w * uv + torch.linalg.cross(u, uv, dim=-1))


def _rotz_wxyz(yaw_rad: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion for a rotation ``yaw_rad`` about +Z. ``yaw_rad`` [...] -> [..., 4]."""
    half = 0.5 * yaw_rad
    zeros = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def _approach_quat_tilt_wxyz(beta_deg: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Approach quat ``A(beta) = roty(90-beta)`` (wxyz): tool +x pitched ``beta`` deg up from straight-
    down toward horizontal, swung in the sagittal (x-z) plane. beta=90 -> level side grasp (identity,
    tool +x horizontal); beta=60 -> 30 deg below horizontal (near-level, angled down). Same convention
    and formula as ``g1_curobo_robot_cfg.approach_quat_tilt``; duplicated here (not imported) because
    the g1 cfg imports FROM this module -- importing back would be circular."""
    half = math.radians(90.0 - beta_deg) / 2.0
    return torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0], device=device, dtype=dtype)


def cube_grasp_poses_obj(flip: bool = True, device=None, dtype=torch.float32,
                         beta_degs: tuple[float, ...] = (90.0,), z_above: float = 0.0,
                         standoff: float = 0.0
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Cube stand-in grasp set, expressed in the CUBE's own frame (placeholder for the out-of-scope
    grasp planner). A parallel gripper side-grasps the cube; the cube's 4-fold vertical symmetry
    affords 4 equivalent face-grasp yaws (0/90/180/270 about the cube +Z), and the gripper is itself
    symmetric under a 180deg flip about its approach axis (tool +x), doubling each to a top/down twin.

    ``beta_degs`` is the set of APPROACH TILTS to union into the goalset (see ``_approach_quat_tilt_wxyz``):
    the default ``(90,)`` = pure level side grasp = the historical behavior BIT-FOR-BIT (A(90)=identity,
    so ``rotz(yaw) . A(90) == rotz(yaw)``). Passing several betas (e.g. ``(90, 60)``) stacks a level AND
    a near-level tilted candidate PER YAW, so cuRobo's goalset-min picks whichever orientation is
    kinematically feasible for a given cube -- a jitter-tail cube unreachable dead-level may still be
    grasped at a downward tilt (the same lever g1 ships as a fixed tilt; here offered as a choice, not
    forced). Total candidates ``= len(beta_degs) * 4 * (2 if flip else 1)`` -- keep within ``_MAX_GOALSET``.

    Position = ``z_above`` m along the object +Z (cube top-ward), SAME for every candidate -- the
    unified robot-independent grasp anchor (``z_above`` cube-frame metres above center). ``z_above=0``
    (module default) = grasp AT cube center = bit-for-bit historical humanoid/RL-planner behaviour; the
    phase-3/4 descriptors pass ``z_above=GRASP_Z_ABOVE_M``. The tilt rotates the approach ORIENTATION
    only, independent of this position.

    ``standoff`` backs each candidate off along ITS OWN approach axis (tool +x, the direction pointing
    INTO the cube), i.e. ``-standoff * R(q_cand) @ x_hat``. Per-candidate, NOT a shared world +Z lift:
    the shipped grasps are tilted side-grasps (humanoid 15/30/45 deg below horizontal, g1 30/45), so a
    vertical lift would slide diagonally across the jaws instead of retreating along the approach. Used
    by ``goal="pregrasp"`` (stage A of the two-stage reach) to place the pre-grasp waypoint; ``0.0``
    (module default) = grasp pose itself = every historical caller unchanged.

    Returns ``(grasp_pos_obj[G,3], grasp_quat_obj[G,4])`` (wxyz), G = ``len(beta_degs)*4*(2 if flip else 1)``.
    """
    yaws = torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2], device=device, dtype=dtype)
    per_beta = []
    for beta in beta_degs:
        approach = _approach_quat_tilt_wxyz(beta, device, dtype)
        per_beta.append(_quat_mul_wxyz(_rotz_wxyz(yaws), approach.expand(4, 4)))   # rotz(yaw) then tilt
    quats = torch.cat(per_beta, dim=0)                                    # [4*B, 4]
    if flip:
        flip_q = torch.tensor(_TOPDOWN_FLIP_QUAT_WXYZ, device=device, dtype=dtype)
        flipped = _quat_mul_wxyz(quats, flip_q.expand_as(quats))          # right-mult: tool-frame flip
        quats = torch.cat([quats, flipped], dim=0)                        # [8*B, 4]
    pos = torch.zeros(quats.shape[0], 3, device=device, dtype=dtype)
    pos[:, 2] = z_above                                                   # +Z above cube center (obj frame)
    if standoff:
        x_hat = torch.zeros_like(pos)
        x_hat[:, 0] = 1.0
        pos = pos - standoff * _quat_rotate_vec_wxyz(quats, x_hat)        # retreat along each candidate's approach
    return pos, quats


def grasp_poses_to_base(obj_pos_base: torch.Tensor, obj_quat_base: torch.Tensor,
                        grasp_pos_obj: torch.Tensor, grasp_quat_obj: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid-transform object-frame grasp poses into base_link (object-agnostic).

    Args:
        obj_pos_base: ``[3]`` object origin in base_link.
        obj_quat_base: ``[4]`` object orientation in base_link (wxyz).
        grasp_pos_obj: ``[G, 3]``, grasp_quat_obj: ``[G, 4]`` (wxyz) -- grasp poses in object frame.
    Returns:
        ``(cand_pos_base[G,3], cand_quat_base[G,4])`` -- ``p = obj_pos + R(obj_quat) @ p_obj``,
        ``q = obj_quat (x) q_obj``.
    """
    obj_quat = obj_quat_base.view(1, 4).expand(grasp_quat_obj.shape[0], 4)
    cand_pos = obj_pos_base + _quat_rotate_vec_wxyz(obj_quat, grasp_pos_obj)
    cand_quat = _quat_mul_wxyz(obj_quat, grasp_quat_obj)
    return cand_pos, cand_quat


def set_reaching_goalset(idle_position: torch.Tensor, idle_quaternion: torch.Tensor,
                         cand_positions: torch.Tensor, cand_quaternions: torch.Tensor,
                         reaching_link_idx: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble a 2-link ``[B=1,H=1,L=2,G,3/4]`` GoalToolPose goalset (object-agnostic).

    Every declared tool frame must be covered (cuRobo rejects a subset -- see ``reorder_links``), so
    the idle link is duplicated across the goalset and the reaching link carries the G candidates.
    The solver mins over the goalset per link, reaching whichever candidate is closest/reachable.

    Args:
    idle_position: ``[3]``, idle_quaternion: ``[4]`` (wxyz) -- the forward-kinematics-pinned idle-tool pose.
        cand_positions: ``[G, 3]``, cand_quaternions: ``[G, 4]`` (wxyz) -- reaching-link candidates
            in base_link.
        reaching_link_idx: Index of reaching link in the configured tool-frame order.
    Returns:
        ``(position[1,1,2,G,3], quaternion[1,1,2,G,4])``.
    """
    assert reaching_link_idx in (0, 1), f"expected two tool frames, got reaching_link_idx={reaching_link_idx}"
    g = cand_positions.shape[0]
    idle_pos = idle_position.view(1, 3).expand(g, 3)
    idle_quat = idle_quaternion.view(1, 4).expand(g, 4)
    positions = [idle_pos, idle_pos]
    quaternions = [idle_quat, idle_quat]
    positions[reaching_link_idx] = cand_positions
    quaternions[reaching_link_idx] = cand_quaternions
    position = torch.stack(positions, dim=0).view(1, 1, 2, g, 3)
    quaternion = torch.stack(quaternions, dim=0).view(1, 1, 2, g, 4)
    return position, quaternion


def set_bimanual_goalset(left_positions: torch.Tensor, left_quaternions: torch.Tensor,
                         right_positions: torch.Tensor, right_quaternions: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble a 2-link ``[B=1,H=1,L=2,G,3/4]`` GoalToolPose goalset where BOTH links carry real
    grasp candidates (object-agnostic) -- simultaneous bimanual reach, each hand to its own object.

    Unlike ``set_reaching_goalset`` (one forward-kinematics-pinned idle link + one reaching link), both links get a
    G-candidate set here; the solver mins over the goalset per link INDEPENDENTLY, so each hand
    reaches whichever of ITS OWN object's candidates is closest/reachable. Fixed L/R assignment:
    link 0 = left object's candidates, link 1 = right object's candidates.

    Link order ``[L, R]`` -- the caller's ``tool_frames`` MUST be ``[..._L_site, ..._R_site]`` in
    that order, else each hand chases the wrong object with no error.

    The goalset shares a single G axis across links, so both sides must supply the same candidate
    count (asserted). Cube grasps give G=8 each; a future object with a different count would need
    padding.

    Args:
        left_positions: ``[G, 3]``, left_quaternions: ``[G, 4]`` (wxyz) -- left-hand candidates in
            base_link. right_positions/right_quaternions: same for the right hand.
    Returns:
        ``(position[1,1,2,G,3], quaternion[1,1,2,G,4])``.
    """
    assert left_positions.shape[0] == right_positions.shape[0], (
        f"bimanual goalset needs equal candidate counts per side (shared G axis); got "
        f"left={left_positions.shape[0]} right={right_positions.shape[0]}"
    )
    g = left_positions.shape[0]
    position = torch.stack([left_positions, right_positions], dim=0).view(1, 1, 2, g, 3)
    quaternion = torch.stack([left_quaternions, right_quaternions], dim=0).view(1, 1, 2, g, 4)
    return position, quaternion


__all__ = [
    "HUMANOID_V21_URDF", "BASE_LINK", "TOOL_FRAMES", "HUMANOID_ARM_JOINT_HOME",
    "build_self_collision_ignore", "build_motion_planner_kwargs",
    "build_robot_cfg_dict_from_urdf",
    "cube_grasp_poses_obj", "grasp_poses_to_base", "set_reaching_goalset",
    "set_bimanual_goalset",
]
