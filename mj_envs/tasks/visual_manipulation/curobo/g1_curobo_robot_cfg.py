"""Build a cuRobo ``RobotCfg`` dict for the Unitree G1 + parallel-gripper URDF.

Third robot on the shared cuRobo planning machinery (after humanoid_v21 and dual-UR5e). G1 is a
FLOATING-BASE humanoid, so this cfg is a hybrid of the other two:

  - Like ``ik_curobo_robot_cfg`` (humanoid_v21): only the ARMS plan; every non-arm joint (legs,
    waist, gripper racks) is pinned to ``qpos0`` via ``lock_joints`` while its links still take part
    in collision. The two feet are the bulk of the collision spheres (61 each) and sit far from the
    arm workspace, so their locked subtrees are excluded from the collision model (mirrors the
    humanoid ``hip_3`` exclusion, cut here at the KNEE so the feet+ankles drop).
  - Like ``ur5e_dual_robot_cfg``: self-collision ignore = kinematic-tree adjacency UNION rest-pose
    sphere overlap only -- NO humanoid Mink pair sets (those are authored for humanoid_v21 geometry
    and meaningless here). The rest-overlap source is critical: G1's legs are LOCKED at ``qpos0``, so
    any structural sphere overlap there is present in EVERY IK config and would otherwise make cuRobo
    flag every seed self-colliding (the humanoid 0-IK-success failure mode).

Frame / naming differences from humanoid_v21 (base = ``base_link``, tool = ``end_effector_{L,R}_site``):
  - base link is ``pelvis``; tool frames are the parallel gripper's grasp-center sites
    ``left_hand_grasp`` / ``right_hand_grasp`` (added by ``g1_constants._graft_hand``).
  - arms are 7-DOF/side with ``_pitch/_roll/_yaw`` names; wrist ``qpos0`` is 0 (no ``ref`` fold, so
    curobo_q == mjlab_qpos on playback -- simpler than the humanoid wrist_3 +-1.5708).
  - gripper racks are locked at ``qpos0`` (== the humanoid path; the finger collision spheres pass
    outside the 60 mm task cube at the grasp goalset even at the 20 mm jaw gap, verified on humanoid).

The IK/trajopt patch + grasp-goalset assembly are robot-agnostic, so ``build_motion_planner_kwargs``,
``set_bimanual_goalset``, ``cube_grasp_poses_obj``, ``grasp_poses_to_base`` and the rest-overlap
detector are reused straight from ``ik_curobo_robot_cfg`` (as ur5e does).

Kinematics source = the exported ``g1_curobo.urdf`` (``export_mjspec_to_urdf.py --robot g1
--head-camera none``); collision geometry = the compiled G1 model's MuJoCo group-3/5 primitives via
``build_collision_spheres`` (same bypass-the-URDF-<collision> policy as the other two).

Run (debug):
    /home/grl/repo/micromamba/envs/py312/bin/python -c "from tasks.visual_manipulation.curobo.g1_curobo_robot_cfg \\
        import build_robot_cfg_dict; cfg = build_robot_cfg_dict(); \\
        cs = cfg['robot_cfg']['kinematics']['collision_spheres']; \\
        print(len(cs), 'links', sum(len(v) for v in cs.values()), 'spheres')"
"""

from __future__ import annotations

import math
import pathlib
import re
import sys
from collections.abc import Mapping

import mujoco
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mjlab.entity.entity import Entity  # noqa: E402

from mj_envs.asset_zoo.g1.g1_constants import get_g1_robot_cfg  # noqa: E402
from mj_envs.utils.mj_collision_spheres import build_collision_spheres  # noqa: E402

# Robot-agnostic reuse from the humanoid cfg (none of it is humanoid-specific): the rest-overlap
# detector, the IK/trajopt-patch, the base->object transform + bimanual goalset assembler, and the
# quaternion helpers. Only the grasp-candidate ORIENTATIONS differ (tilted near-level for G1, see below), so
# `cube_grasp_poses_obj` is redefined locally rather than imported.
from mj_envs.tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import (  # noqa: E402
    _quat_mul_wxyz,
    _quat_rotate_vec_wxyz,
    _rest_overlapping_link_pairs,
    _rotz_wxyz,
    _TOPDOWN_FLIP_QUAT_WXYZ,
    build_motion_planner_kwargs,
    grasp_poses_to_base,
    set_bimanual_goalset,
)

G1_CUROBO_URDF = str(_REPO_ROOT / "asset" / "unitree_g1" / "g1_curobo.urdf")
BASE_LINK = "pelvis"
# Tool frames = the parallel gripper grasp-center sites (URDF fixed links). Link order [left, right]
# MUST match set_bimanual_goalset's [L, R] order, else each hand chases the wrong cube.
GRASP_TOOL_FRAMES = ["left_hand_grasp", "right_hand_grasp"]

# G1 grasp-candidate approach = FIXED tilt beta=60deg (30deg off dead-level), most-horizontal feasible
# on the SHARED table. A DEAD-level side-grasp (beta=90, like humanoid/ur5e) is infeasible for G1 (the
# parallel gripper hangs below the grasp site and collides). The hang is ORIENTATION-dependent, so a
# tilted approach threads it; tilt sweep found feasibility to beta~64-70deg, ship 60 for jitter margin.
# beta=0 reproduces the old top-down grasp bit-for-bit. Scene stays IDENTICAL to humanoid (baseline
# parity, MEMORY line 20); the only lever is this grasp tilt.
#   Rejected (2026-07-10): a single-solve "prefer horizontal, allow tilt" (command beta=90, soften the
#   tilt axis so the optimizer lands as-horizontal-as-reachable). Does NOT work with cuRobo -- see the
#   plan/CUROBO_HANDOFF.md "prefer-horizontal" section. If a variable tilt is ever wanted, the correct
#   shape is a DESCENDING-beta fallback (command 90, step down until IK feasible; ~+50 ms/grasp,
#   one-shot), NOT the weight lever.
G1_GRASP_TILT_DEG = 60.0
# Approach-tilt goalset for the grasp generator (symmetry with humanoid ``HUMANOID_GRASP_BETAS``).
# Union (60,45): dead-level (beta=90) rakes the shared slab (parallel-gripper hang ~10 cm) so it stays
# excluded; (60,45) both clear the slab and widen the goalset beyond the old single-tilt (60,) --
# matches humanoid's (60,45) union, user choice 2026-07-22.
G1_GRASP_BETAS = (G1_GRASP_TILT_DEG, 45.0)



def approach_quat_tilt(beta_deg: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """Approach quat ``A(beta) = roty(90-beta)`` (wxyz): tool +x -> a dir ``beta`` deg up from straight-
    down toward horizontal. beta=0 -> top-down (tool +x -> world -z); beta=90 -> level side-grasp.

    ANGLE CONVENTION (read before tuning beta): beta is measured FROM the DOWNWARD VERTICAL (-z),
    NOT from horizontal. It is the angle of the approach axis above straight-down, swung in the x-z
    SAGITTAL/VERTICAL plane (rotation is about Y, so the approach has no lateral y-component; the
    later ``rotz(yaw)`` spins it onto one of the cube's 4 faces). Equivalently ``(90-beta)`` = pitch
    DOWN below horizontal. So: beta=0 -> straight down (0 above vertical); beta=60 -> 30 deg below
    horizontal (near-level, angled down); beta=90 -> horizontal (90 deg from the down-vertical)."""
    half = math.radians(90.0 - beta_deg) / 2.0
    return torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0], device=device, dtype=dtype)


def cube_grasp_poses_obj(flip: bool = False, device=None, dtype=torch.float32,
                         beta_degs: tuple[float, ...] = (G1_GRASP_TILT_DEG,),
                         z_above: float = 0.0, standoff: float = 0.0
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Tilted cube grasp set in the CUBE frame (G1's max-horizontal analog of the humanoid side grasp).

    Orientation = ``rotz(yaw) . A(beta)``: the tilted approach (``approach_quat_tilt(beta)``) yawed
    to one of the cube's 4 vertical-symmetry faces (0/90/180/270 about the cube +Z), optionally doubled
    by the gripper's 180deg approach-axis flip. Position = ``z_above`` m along the object +Z (same for
    every candidate) -- the unified robot-independent grasp anchor (shared with the humanoid
    ``cube_grasp_poses_obj``). ``z_above=0`` (module default) = grasp AT cube center; the phase-3/4 g1
    descriptor passes ``z_above=GRASP_Z_ABOVE_M``. (``G1_PREGRASP_STANDOFF_M`` is unused here -- the
    two-stage reach's pre-grasp retreat comes from ``standoff`` below, sized by the shared
    ``planner._PREGRASP_STANDOFF_M``.)

    ``standoff`` backs each candidate off along ITS OWN approach axis (``-standoff * R(q_cand) @ x_hat``,
    tool +x = the direction into the cube). Per-candidate, not a world +Z lift: g1 grasps 30/45 deg below
    horizontal, so a vertical lift would slide across the jaws. Set by ``goal="pregrasp"`` (stage A of the
    two-stage reach); ``0.0`` (module default) = the grasp pose itself, every historical caller unchanged.

    ``beta_degs`` is the set of APPROACH TILTS unioned into the goalset (same interface as the shared
    humanoid ``cube_grasp_poses_obj``): the default ``(G1_GRASP_TILT_DEG,)`` = a single tilt = the
    historical behavior BIT-FOR-BIT. Passing several betas stacks one candidate PER YAW PER BETA so
    cuRobo's goalset-min picks whichever orientation is kinematically feasible. G1 ships a single
    feasible tilt (dead-level beta=90 rakes the shared slab -- gripper hang); the humanoid unions {90,60}.

    Returns ``(grasp_pos_obj[G,3], grasp_quat_obj[G,4])`` (wxyz),
    G = ``len(beta_degs) * 4 * (2 if flip else 1)`` -- keep within ``_MAX_GOALSET``.
    """
    yaws = torch.tensor([0.0, torch.pi / 2, torch.pi, 3 * torch.pi / 2], device=device, dtype=dtype)
    per_beta = []
    for beta in beta_degs:
        approach = approach_quat_tilt(beta, device, dtype)
        per_beta.append(_quat_mul_wxyz(_rotz_wxyz(yaws), approach.expand(4, 4)))   # rotz(yaw) then tilt
    quats = torch.cat(per_beta, dim=0)                                    # [4*B, 4]
    if flip:
        flip_q = torch.tensor(_TOPDOWN_FLIP_QUAT_WXYZ, device=device, dtype=dtype)
        flipped = _quat_mul_wxyz(quats, flip_q.expand_as(quats))           # right-mult: tool-frame flip
        quats = torch.cat([quats, flipped], dim=0)                         # [8*B, 4]
    pos = torch.zeros(quats.shape[0], 3, device=device, dtype=dtype)
    pos[:, 2] = z_above                                                   # +Z above cube center (obj frame)
    if standoff:
        x_hat = torch.zeros_like(pos)
        x_hat[:, 0] = 1.0
        pos = pos - standoff * _quat_rotate_vec_wxyz(quats, x_hat)        # retreat along each candidate's approach
    return pos, quats

# 7 actuated joints per side; any joint NOT here is a non-arm joint (legs + waist + gripper racks)
# and must end up in lock_joints.
_ARM_SUBJOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")
# Per-side arm joint names (G1 analog of humanoid_v21_constants.LEFT/RIGHT_ARM_JOINT_NAMES). Exported
# so the phase-3 verify's mink localik + drift can build per-side joint groups without reaching into
# the private cspace tuple.
LEFT_ARM_JOINT_NAMES = tuple(f"left_{j}_joint" for j in _ARM_SUBJOINTS)
RIGHT_ARM_JOINT_NAMES = tuple(f"right_{j}_joint" for j in _ARM_SUBJOINTS)
_ARM_JOINT_CSPACE_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
_ARM_JOINT_NAMES = frozenset(_ARM_JOINT_CSPACE_NAMES)

# Reach-ready arm home (MuJoCo qpos) = the cuRobo cspace retract seed. TILTED front-reach pose matching
# the shipped grasp tilt (``G1_GRASP_TILT_DEG=60deg``, near-level): a clean SYMMETRIC solve from
# Elbows-bent tucked ready pose: hands held IN and UP off the table plane so the arms do not sweep the
# cube during the walk-in approach (the old raised-but-forward pose skimmed the table and knocked the
# cube). Presents the jaws already near-horizontal, matching where cuRobo drives the arms. Every active
# arm joint must appear (cuRobo cspace is per named joint; a silent 0 default would fold the arm off the
# reach-ready seed).
G1_ARM_HOME: dict[str, float] = {
    "left_shoulder_pitch_joint": 1.1447,
    "left_shoulder_roll_joint": 0.1185,
    "left_shoulder_yaw_joint": 0.1665,
    "left_elbow_joint": -0.9863,
    "left_wrist_roll_joint": -0.3769,
    "left_wrist_pitch_joint": -0.0892,
    "left_wrist_yaw_joint": -0.2650,
    "right_shoulder_pitch_joint": 1.1404,
    "right_shoulder_roll_joint": -0.1185,
    "right_shoulder_yaw_joint": -0.1818,
    "right_elbow_joint": -0.9829,
    "right_wrist_roll_joint": 0.3894,
    "right_wrist_pitch_joint": -0.0852,
    "right_wrist_yaw_joint": 0.2729,
}

# Locked-leg collision-exclusion boundary: everything STRICTLY BELOW the 2nd hip link (hip_roll)
# is dropped from the collision model -- mirrors the humanoid cfg, which cuts strictly below hip_3.
# The G1 hip chain is pelvis -> hip_pitch (1st) -> hip_roll (2nd) -> hip_yaw (3rd) -> knee -> ankles;
# cutting below hip_roll drops hip_yaw + knee + both ankle feet (all pinned at qpos0, far from the
# arm workspace, only a constant self-collision bound the ignore set already zeroes). Pelvis +
# hip_pitch + hip_roll stay (closest leg links to the torso/arm sweep).
_COLLISION_EXCLUDE_SUBTREES = ("left_hip_roll_link", "right_hip_roll_link")


def _body_name(mj_model, b: int) -> str:
    return mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b)


def _build_mj_model_resolved() -> mujoco.MjModel:
    """The exporter's resolved G1 variant (parallel_gripper actuated, original ``builtin`` head --
    matches ``export_mjspec_to_urdf.py --robot g1 --head-camera builtin``). ``qpos0`` encodes the home
    snapshot for the locked leg/waist/rack joints; collision geom poses are read straight off the model."""
    return Entity(
        get_g1_robot_cfg(end_effector="actuated", hand="parallel_gripper", head_camera="builtin")
    ).compile()


def _strict_descendant_bodies(mj_model, root_names: tuple[str, ...]) -> set[str]:
    """Body names strictly below (descendants of, excluding) each root in the mj kinematic tree."""
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


def _lock_joints_from_resolved(
    mj_model, excluded: set[str], joint_pos: Mapping[str, float],
) -> dict[str, float]:
    """Lock non-arm joints at configured home, falling back to compiled ``qpos0``.

    The shared planner supplies the open parallel-gripper rack targets through ``joint_pos``. Without
    this override, G1 planned with qpos0 jaws while physics opened them wider before every grasp. The
    free root has no lockable scalar; joints below excluded collision subtrees are absent from cuRobo.
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
        child_body = _body_name(mj_model, mj_model.jnt_bodyid[jid])
        if child_body in excluded:
            continue
        lock[jname] = home[jname]
    return lock


def _adjacency_and_rest_ignore(mj_model, collision_spheres: dict[str, list[dict]]) -> dict[str, list[str]]:
    """Self-collision ignore = kinematic-tree parent-child adjacency UNION rest-pose sphere overlap.

    Both directions (cuRobo reads the dict as a matrix). Adjacency (whole body): links sharing a
    joint always touch. Rest-overlap (``_rest_overlapping_link_pairs``): 2-hop structural neighbors
    (and the coarsened locked-leg links) whose spheres overlap at ``qpos0`` -- collision-free by
    construction, so any overlap there is a permanent artifact that must be ignored or every locked-
    leg IK config self-collides. No Mink sets (humanoid-specific). Same construction as ur5e."""
    ignore: dict[str, list[str]] = {}

    def add_both(a: str, b: str) -> None:
        if a == b:
            return
        for x, y in ((a, b), (b, a)):
            ignore.setdefault(x, [])
            if y not in ignore[x]:
                ignore[x].append(y)

    for b in range(1, mj_model.nbody):
        p = mj_model.body_parentid[b]
        if p == 0:
            continue
        add_both(_body_name(mj_model, b), _body_name(mj_model, p))

    for frozen_pair in _rest_overlapping_link_pairs(mj_model, collision_spheres):
        a, b = tuple(frozen_pair)
        add_both(a, b)
    return ignore


# Arm joint accel/jerk caps for cuRobo trajopt. Lowered from the industrial default (10.0 / 500.0)
# to gentle the reaction wrench the arm exerts on G1's FLOATING base mid-reach. Value = the frontier
# knee measured on the humanoid (see the fuller rationale in ik_curobo_robot_cfg.ARM_MAX_ACCELERATION:
# -46% peak momentum at 1.67x time; 2.0 backfires). Lowered a further step to 3.0/150.0 to gentle the
# visible planner + MPC trajectory (stays above the 2.0 wiggle knee). Floating-base robot -> real benefit.
ARM_MAX_ACCELERATION = 3.0
ARM_MAX_JERK = 150.0


def _arm_cspace_from_home(mj_model, arm_joint_home: Mapping[str, float] | None) -> dict:
    """cuRobo active-arm cspace default from a MuJoCo-qpos home map (``None`` -> ``G1_ARM_HOME``).

    Values are converted to cuRobo coordinates by subtracting compiled ``mj_model.qpos0`` (0 for G1
    arm joints -- no ``ref`` -- but kept for parity with the humanoid path). Every active arm joint
    must be present explicitly."""
    home = dict(G1_ARM_HOME) if arm_joint_home is None else dict(arm_joint_home)
    missing = [j for j in _ARM_JOINT_CSPACE_NAMES if j not in home]
    assert not missing, f"arm_joint_home missing active arm joints: {missing}"

    default_joint_position = []
    for jname in _ARM_JOINT_CSPACE_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"active arm joint {jname!r} not found in resolved mj_model"
        qpos0 = float(mj_model.qpos0[mj_model.jnt_qposadr[jid]])
        default_joint_position.append(float(home[jname]) - qpos0)
    return {
        "joint_names": list(_ARM_JOINT_CSPACE_NAMES),
        "default_joint_position": default_joint_position,
        "cspace_distance_weight": [1.0] * len(_ARM_JOINT_CSPACE_NAMES),
        "null_space_weight": [1.0] * len(_ARM_JOINT_CSPACE_NAMES),
        "max_acceleration": ARM_MAX_ACCELERATION,
        "max_jerk": ARM_MAX_JERK,
    }


def build_robot_cfg_dict(
    load_dynamics: bool = False,
    arm_joint_home: Mapping[str, float] | None = None,
) -> dict:
    """cuRobo RobotCfg dict for the G1 + parallel gripper.

    ``{robot_cfg: {kinematics: {...}, load_dynamics: bool}}``. Collision spheres are the resolved
    model's MuJoCo collision primitives with the below-knee (feet) subtrees excluded; kinematics load
    from the exported URDF. 14 arm joints are the active cspace DOFs; non-arm joints use configured
    home when supplied, otherwise ``qpos0``. ``arm_joint_home`` is a MuJoCo-qpos map keyed by joint name."""
    mj_model = _build_mj_model_resolved()
    excluded = _strict_descendant_bodies(mj_model, _COLLISION_EXCLUDE_SUBTREES)
    spheres = build_collision_spheres(mj_model)
    spheres = {ln: sl for ln, sl in spheres.items() if ln not in excluded}
    joint_home = dict(G1_ARM_HOME) if arm_joint_home is None else dict(arm_joint_home)
    lock_joints = _lock_joints_from_resolved(mj_model, excluded, joint_home)
    arm_in_lock = [j for j in lock_joints if j in _ARM_JOINT_NAMES]
    assert not arm_in_lock, f"arm joints leaked into lock_joints: {arm_in_lock}"

    total = sum(len(v) for v in spheres.values())
    print(f"[curobo] G1 collision-geom spheres: {len(spheres)} links / {total} spheres "
          f"(excluded {len(excluded)} leg bodies below hip_roll)")
    print(f"[curobo] lock_joints: {len(lock_joints)} non-arm joints pinned to configured home | "
          f"active arm DOFs: {len(_ARM_JOINT_CSPACE_NAMES)}")

    return {
        "robot_cfg": {
            "kinematics": {
                "base_link": BASE_LINK,
                "tool_frames": GRASP_TOOL_FRAMES,
                "urdf_path": G1_CUROBO_URDF,
                "asset_root_path": "/",
                "collision_link_names": list(spheres.keys()),
                "collision_spheres": spheres,
                "self_collision_ignore": _adjacency_and_rest_ignore(mj_model, spheres),
                "self_collision_buffer": {},
                "lock_joints": lock_joints,
                "cspace": _arm_cspace_from_home(mj_model, joint_home),
            },
            "load_dynamics": load_dynamics,
        }
    }


__all__ = [
    "G1_CUROBO_URDF", "BASE_LINK", "GRASP_TOOL_FRAMES", "G1_ARM_HOME",
    "LEFT_ARM_JOINT_NAMES", "RIGHT_ARM_JOINT_NAMES",
    "build_robot_cfg_dict", "build_motion_planner_kwargs", "set_bimanual_goalset",
    "cube_grasp_poses_obj", "grasp_poses_to_base",
]
