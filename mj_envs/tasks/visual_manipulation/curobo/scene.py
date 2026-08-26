"""cuRobo-facing scene derivation: compiled model + scenario -> obstacle dict + base-frame targets.

Thin DERIVATION on top of ``pickplace_scenarios`` (authoring) and ``ik_curobo.build_curobo_scene``
(fixed ``pp_*`` obstacle extraction). This is the ENV dimension of the reach spine -- the only module
that varies sim vs real: a future perception backend swaps ``cube_world_poses`` alone, leaving all
target/obstacle math untouched. Robot-independent apart from reading ``base_link`` / ``target_mode``
off a passed ``RobotDescriptor``.

Layering (one-way): ``planner.py`` <- ``scene.py`` <- ``pickplace_scenarios.py``.

Frame convention (load-bearing): cube poses are WORLD-frame at the ``cube_world_poses`` seam; the ONE
explicit world->base transform is ``to_base_frame`` (from ``build_curobo_scene``), applied here to
produce base-frame grasp/transit TARGETS. No downstream code re-does world<->base.
"""

from __future__ import annotations

import atexit
import ctypes
import dataclasses
import itertools
import math
import multiprocessing as mp
import os
import pathlib
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from asset_zoo.scene_object.human_figure import human_figure  # noqa: E402
from tasks.visual_manipulation.pickplace_scenarios import (  # noqa: E402
    make_scenarios,
    table_z_for,
)
from curobo._src.state.state_joint import JointState  # noqa: E402
from curobo._src.types.tool_pose import GoalToolPose  # noqa: E402
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria  # noqa: E402
from tasks.visual_manipulation.curobo.ik_curobo import (  # noqa: E402
    build_curobo_scene, final_active_q)
from tasks.visual_manipulation.curobo.planner import (  # noqa: E402
    _PREGRASP_STANDOFF_M,
    CFG_BY_ROBOT,
    CuroboPlannerSession,
    RobotDescriptor,
    assign_cubes_to_sides,
    grasp_candidates_base,
    ik_feasible_assigned,
    make_planner_session,
    plan_assigned,
    reach_error,
    site_base,
    world_axes_in_base_quat,
)
from tasks.camera_perception import fov_detect  # noqa: E402
from tasks.camera_terms import H_HALF, V_HALF, NEAR, FAR  # noqa: E402
from tasks.visual_manipulation.policies import _GAZE_YAW_ROM, _GAZE_PITCH_ROM  # noqa: E402
from tasks.visual_manipulation.control_vec import gaze_aim  # noqa: E402


def pose_only_forward(mj_model, mj_data) -> None:
    """Refresh body/site/geom world transforms from ``qpos`` -- and nothing else.

    The gaze solve is pure kinematics: it writes gimbal ``qpos`` and reads back ``xpos``/``xmat``/
    ``site_xpos``/``site_xmat`` (plus ``geom_xpos``/``geom_xmat`` for the ``mj_ray`` sight-line test).
    ``mj_forward`` also runs collision detection, the constraint solver, passive/actuator forces and
    inverse dynamics, none of which any gaze consumer reads -- measured 0.565 ms vs 0.002 ms for
    ``mj_kinematics`` on the v2 bimanual scene (191 geoms), and the gaze scratch is forwarded several
    times per control tick. ``mj_camlight`` is included because it is free and fills ``cam_xpos`` for
    any future consumer that expects ``mj_forward``'s full pose set.

    ONLY safe on a scratch ``MjData`` whose consumers are pose queries. It is NOT a drop-in for
    ``mj_forward`` on data that is later read for contacts (``ncon``/``contact`` -- needs
    ``mj_collision``), Jacobians (``mj_jacSite`` -- needs ``cdof`` from ``mj_comPos``), or bias forces
    (``qfrc_bias``). The harness ``_scratch`` and the Jacobian tracker's ``data`` are all three, and
    deliberately keep ``mj_forward``.
    """
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_camlight(mj_model, mj_data)


def _solve_gaze(d_base, sign) -> tuple[float, float]:
    """Closed-form (yaw, pitch) for a BASE-frame look direction ``d_base`` (np 3-vec), via the shared
    deploy-safe ``control_vec.gaze_aim`` (single source of truth for the aim math + yaw sign). ``gaze_aim``
    normalizes internally, so pass the raw direction (target=d, cam=0). Returns UNCLAMPED python floats
    (ROM=inf); callers apply their own ROM gate."""
    yaw, pitch = gaze_aim(torch.as_tensor(d_base, dtype=torch.float32), torch.zeros(3),
                          float(sign), float("inf"), float("inf"))
    return float(yaw), float(pitch)

# ------------------------------------------------------------------------------------------------
# Scene constants
# ------------------------------------------------------------------------------------------------
PICK_JITTER_M = 0.08                                # study knob; within humanoid's proven-feasible range
TRANSIT_Z_BASE = 0.10                               # single-arm transit-hover reach Z (base frame)
_LIMIT_PROBE = bool(os.environ.get("LIMIT_PROBE"))  # diagnostic: retract-plan outcome + binding speed limit
# DEAD LEVER -- do not retry. The RETRACT route is ACCELERATION-bound (``calculate_dt_no_clamp``:
# dt = max(vel, sqrt(acc), cbrt(jerk)) scores; measured 0.84 acc vs 0.06 vel, 0.09 jerk), so raising the
# accel cap for the home solve alone looks like free speed. It is not reachable from here: mutating the
# acceleration row of the LIVE ``get_state_bounds()`` around ``plan_pose`` makes every home solve fail
# identically at 1.25x, 1.5x and 2.0x -- flat, not graded, so it is the warmed/graph-captured trajopt
# rejecting a bounds change rather than physical infeasibility. Scaling the playback CLOCK is not a
# substitute either: a route timed for acceleration A run s times faster executes at s^2 A, past the cap
# the trajopt honored.
_SIGHT_EPS_M = 1e-3                                  # occluder must lie strictly BEFORE the target center by
#   this margin to count (skips the target's own front face / grazing self-contact).
_SELF_HIT_TOL_M = 1e-3                               # a pp_ geom within this of the target centroid IS the
#   target cube -> never its own occluder (robot geoms are not in the pp_ set, so no housing self-block).
# Commit-visibility SAFETY MARGIN (angular): a cube that only just clears the true sensor half-angle
# (H_HALF/V_HALF, the real RGB frustum -- unchanged, still used for training/reward/overlay) is a bad
# COMMIT decision -- the belief this margin gates feeds every ACQUIRE/DISCOVER/APPROACH decision and the
# EXTEND commit, and standing/reach sway is large at close range (documented elsewhere in this reach FSM:
# a few cm of g1 sway subtends ~8 deg at a 0.29 m target). A border-line cube can drift outside the TRUE
# frustum mid-reach -- physically, a real single-camera robot cannot complete a grasp on something it can
# no longer see. Shrinking the half-angles used for the belief gate (NOT the true FOV model elsewhere)
# keeps every committed cube comfortably inside the real cone with headroom for that sway. Same margin for
# every robot (the sway magnitude that motivated it is a physical-stance property, not sensor-specific).
_VISIBILITY_MARGIN_RAD = np.radians(8.0)
_COMMIT_H_HALF = H_HALF - _VISIBILITY_MARGIN_RAD
_COMMIT_V_HALF = V_HALF - _VISIBILITY_MARGIN_RAD


# ------------------------------------------------------------------------------------------------
# Line-of-sight (occlusion) test: a cube in the FOV cone is SEEN only if nothing blocks the sight line
# ------------------------------------------------------------------------------------------------
_OCCLUDER_CACHE: dict[int, np.ndarray] = {}


def _scene_occluder_geoms(mj_model) -> np.ndarray:
    """Geom ids of SCENE occluders -- the ``pp_*`` namespace (shelf/bench boxes, posts, hand shapes, pick
    cubes). Robot geoms are EXCLUDED by construction, so the camera housing sitting at a sight ray's origin
    can never self-block. Cached per compiled model (the id key is stable for a run)."""
    key = id(mj_model)
    ids = _OCCLUDER_CACHE.get(key)
    if ids is None:
        ids = np.array(
            [g for g in range(mj_model.ngeom)
             if (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("pp_")],
            dtype=np.int64)
        _OCCLUDER_CACHE[key] = ids
    return ids


def _sight_line_clear(mj_model, mj_data, cam_pos, target, target_center=None, eye_bodies=()) -> bool:
    """True if the segment camera -> target cube center is unobstructed by any SCENE occluder.

    CPU path (default): per-occluder ``mju_rayGeom`` over ``_scene_occluder_geoms``. The GPU path
    BATCHES this into a single ``mjw.rays`` launch via ``_WarpBatch.ray_clear`` (gated by
    ``LOS_USE_WARP=1`` env var). ``eye_bodies`` is the (cam_body, cube_body) chain the GPU path
    passes to ``ray_clear`` so self-hits advance past the camera mount AND the cube's own near face
    (mirrors the CPU centroid exclusion). The CPU path ignores it.

    Closes the perception fake this module previously carried ("occlusion deferred"): a cube behind a
    shelf board passed the FOV-cone gate and the robot 'saw' it through solid geometry. Per-geom analytic
    ray (``mju_rayGeom``) over ``_scene_occluder_geoms`` on the CPU seam; the GPU env batches the identical
    test with ``mujoco_warp.rays`` (multi-world, BVH). The target cube is excluded by centroid match (it
    would otherwise occlude its own center via its front face); robot self-geoms are never in the set."""
    cam = np.asarray(cam_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_center = target if target_center is None else np.asarray(target_center, dtype=np.float64)
    vec = target - cam
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return True
    if _LOS_USE_WARP and _warp_batch is not None:
        return _sight_line_clear_warp(_warp_batch, cam, target, eye_bodies)
    u = vec / dist
    for gid in _scene_occluder_geoms(mj_model):
        if float(np.linalg.norm(mj_data.geom_xpos[gid] - target_center)) < _SELF_HIT_TOL_M:
            continue                                          # the target cube itself
        t = mujoco.mju_rayGeom(mj_data.geom_xpos[gid], mj_data.geom_xmat[gid],
                               mj_model.geom_size[gid], cam, u, int(mj_model.geom_type[gid]))
        if 0.0 < t < dist - _SIGHT_EPS_M:
            return False
    return True


# Module-level switches for the GPU LOS path. The harness sets _warp_batch (and optionally
# `_LOS_USE_WARP=True`) at construction; the LOS function reads them per call. Module-global
# avoids touching every caller's signature; the function is internal so the blast radius is bounded.
_LOS_USE_WARP = os.environ.get("LOS_USE_WARP") == "1"
_warp_batch = None        # type: _WarpBatch | None


def set_warp_los(enabled: bool, batch=None) -> None:
    """Harness-side hook: enable the GPU LOS path and hand the warp batch to the scene module.

    Called once per _ReachEvalEnv construction. Idempotent; both the enabled flag and the batch
    reference are module-global so the LOS function needs no kwarg."""
    global _LOS_USE_WARP, _warp_batch
    _LOS_USE_WARP = bool(enabled)
    _warp_batch = batch


def _sight_line_clear_warp(batch, cam, target, eye_bodies) -> bool:
    """Batched GPU LOS test. Single ``mjw.rays`` launch via ``_WarpBatch.ray_clear_with_hit``,
    which tests groups 2, 3, 5 (visual + collision + planner spheres) and returns the
    per-ray first non-self-hit geom id. The post-filter against the CPU's
    ``_scene_occluder_geoms`` set keeps the GPU path's source-of-truth aligned with the CPU
    prefix filter -- the GPU's geometric test is broader (more rays hit candidates), but the
    hit geom decides whether it's a real occluder in the CPU sense.

    Returns a 1-element bool tensor; ``bool(...)`` is one CUDA sync (~5 us)."""
    cam_t = torch.as_tensor(cam, dtype=torch.float32, device="cuda:0").view(1, 3)
    target_t = torch.as_tensor(target, dtype=torch.float32, device="cuda:0").view(1, 3)
    eye_bodies = tuple(b for b in eye_bodies if b >= 0)
    clear, hit_geom = batch.ray_clear_with_hit(cam_t, target_t, eye_bodies=eye_bodies, group=2)
    if bool(clear[0]):
        return True
    # GPU found a blocker -- check whether it's in the CPU's strict `pp_*` set.
    return int(hit_geom[0].item()) not in _LOS_CPU_OCCLUDER_SET


# Cached CPU occluder set for the GPU-warp post-filter. Built lazily by the harness.
_LOS_CPU_OCCLUDER_SET: set[int] = set()


def _set_los_cpu_occluder_set(mj_model) -> None:
    """The GPU path's group mask (2, 3, 5) is broader than the CPU's ``_scene_occluder_geoms``
    set (the ``pp_*`` prefix). The GPU path uses the geometric broader set + a per-geom
    post-filter that consults the CPU's strict set so the source of truth stays the CPU.
    This set is internal to the scene module; the harness calls this once per env reset."""
    global _LOS_CPU_OCCLUDER_SET
    _LOS_CPU_OCCLUDER_SET = set(int(g) for g in _scene_occluder_geoms(mj_model).tolist())


def refresh_warp_los_geoms(mj_model, mj_data) -> None:
    """Re-upload mutated ``geom_pos``/``geom_quat`` into the registered GPU LOS batch.

    The batch owns its OWN ``put_model``/``make_data`` snapshot of ``mj_model``, taken once when the
    harness lazily built it. ``move_pick_geoms`` (per seeded reset) and ``_set_live_handover_angle``
    (per handover tick) REWRITE the offered figure's geoms in ``mj_model``, so without this the GPU
    tests every sight line against the figure's pose at batch-construction time -- a cube behind the
    NEW hand reads as visible. Mirrors the harness's own ``_push_static_geoms_to_warp``: the figure
    is world-welded, and warp bakes static geom transforms at ``make_data`` and never recomputes
    them in ``kinematics``, so both the model parameter and the cached ``geom_xpos``/``geom_xmat``
    have to be written. No-op when no batch is registered (CPU LOS path).

    ``mj_data`` must already be ``mj_forward``-ed on ``mj_model``; only its static-geom rows are read.
    """
    if _warp_batch is None:
        return
    import warp as wp

    for arr, src in ((_warp_batch.wm.geom_pos, mj_model.geom_pos),
                     (_warp_batch.wm.geom_quat, mj_model.geom_quat)):
        t = wp.to_torch(arr)
        t[...] = torch.as_tensor(src, device=t.device, dtype=t.dtype)
    static_geom_ids = np.flatnonzero(mj_model.geom_bodyid == 0)
    t = wp.to_torch(_warp_batch.wd.geom_xpos)
    t[:, static_geom_ids] = torch.as_tensor(
        mj_data.geom_xpos[static_geom_ids], device=t.device, dtype=t.dtype)
    t = wp.to_torch(_warp_batch.wd.geom_xmat)
    t[:, static_geom_ids] = torch.as_tensor(
        mj_data.geom_xmat[static_geom_ids].reshape(-1, 3, 3), device=t.device, dtype=t.dtype)


def _cube_visible_from_camera(mj_model, mj_data, cam_pos, cam_mat, center, dims) -> bool:
    """Detect a cube when its center or exposed top face is framed (with commit-safety margin) and
    unoccluded.

    A center-only ray falsely hides a cube resting on a collidable hand or table edge: the support can
    block the cube centroid while a real RGB detector still sees its top face. Sample the center and five
    points on the top face; every ray still obeys live FOV (shrunk by ``_VISIBILITY_MARGIN_RAD``, see its
    comment -- a cube only just inside the TRUE frustum is not a safe commit) and scene occlusion, and the
    returned detection remains the cube center. The target cube itself is excluded from its sample rays by
    its true center.
    """
    center = np.asarray(center, dtype=float)
    dims = np.asarray(dims, dtype=float)
    top_z = center[2] + 0.45 * dims[2]
    samples = (center,
               np.array([center[0], center[1], top_z]),
               *tuple(np.array([center[0] + sx * dims[0], center[1] + sy * dims[1], top_z])
                      for sx, sy in ((-0.35, -0.35), (-0.35, 0.35), (0.35, -0.35), (0.35, 0.35))))
    for sample in samples:
        target = torch.as_tensor(sample, dtype=torch.float32)
        if bool(fov_detect(cam_pos, cam_mat, target, _COMMIT_H_HALF, _COMMIT_V_HALF, NEAR, FAR)["in_fov"]) \
                and _sight_line_clear(mj_model, mj_data, cam_pos, sample, center):
            return True
    return False


# ------------------------------------------------------------------------------------------------
# Object-scene config
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectSceneCfg:
    """Robot-INDEPENDENT scene: a 2-pick stationary scenario + jitter magnitude. Cube world poses
    derive from ``(scenario, seed)`` alone, so both robots reach the SAME scene for a given seed."""
    scenario: object
    pick_jitter_m: float = PICK_JITTER_M

    def jittered(self, seed: int):
        return jitter_pick_markers(self.scenario, seed, self.pick_jitter_m)


# ------------------------------------------------------------------------------------------------
# Object-side helpers (robot-independent)
# ------------------------------------------------------------------------------------------------
def jitter_pick_markers(scenario, seed: int, jitter_m: float = PICK_JITTER_M):
    """Deterministically jitter pick positions in a scenario copy.

    Markers with ``jitter_half_range_xyz`` use independent symmetric-uniform offsets within those
    finite bounds. Table markers set ``hz=0``; the offered hand uses a nonzero ``hz`` (handoff-height
    uncertainty). Other markers retain legacy inward-reflected XY jitter with ``dz=0``.

    Offered objects are NOT sampled: a pick with ``pos_follows_hand`` inherits the sampled offset of
    the hand holding it, so it stays on that palm whatever the hand does. Hands are sampled after the
    free picks, which keeps the draw order (and therefore every seed's scene) identical to the earlier
    revision where the cube led and the hand followed.
    """
    rng = np.random.default_rng(seed)
    jittered = []
    for pick_marker in scenario.pick:
        if pick_marker.pos_follows_hand is not None:
            jittered.append(pick_marker)        # placeholder: filled below, draws no randomness
            continue
        if pick_marker.jitter_half_range_xyz is None:
            dx, dy = rng.uniform(-jitter_m, jitter_m, size=2)
            if pick_marker.pos[0] * dx > 0.0:
                dx = -dx
            if pick_marker.pos[1] * dy > 0.0:
                dy = -dy
            dz = 0.0
        else:
            hx, hy, hz = pick_marker.jitter_half_range_xyz
            if pick_marker.jitter_x_offsets is None:
                dx = rng.uniform(-hx, hx)
            else:
                dx = rng.choice(pick_marker.jitter_x_offsets,
                                p=(pick_marker.jitter_x_first_prob,
                                   1.0 - pick_marker.jitter_x_first_prob))
            dy = rng.uniform(-hy, hy)
            dz = rng.uniform(-hz, hz) if hz else 0.0
        jittered.append(dataclasses.replace(
            pick_marker, pos=(pick_marker.pos[0] + dx, pick_marker.pos[1] + dy, pick_marker.pos[2] + dz),
        ))
    hands = []
    hand_offsets = {}
    for hand_marker in scenario.hand:
        if hand_marker.jitter_half_range_xyz is None:
            hands.append(hand_marker)
            continue
        hx, hy, hz = hand_marker.jitter_half_range_xyz
        dx, dy = rng.uniform(-hx, hx), rng.uniform(-hy, hy)
        dz = rng.uniform(-hz, hz) if hz else 0.0
        hand_offsets[hand_marker.name] = (dx, dy, dz)
        hands.append(dataclasses.replace(
            hand_marker, pos=(hand_marker.pos[0] + dx, hand_marker.pos[1] + dy, hand_marker.pos[2] + dz),
        ))
    jittered = [m if m.pos_follows_hand is None else dataclasses.replace(
        m, pos=tuple(c + d for c, d in zip(m.pos, hand_offsets[m.pos_follows_hand])))
        for m in jittered]
    return dataclasses.replace(scenario, pick=jittered, hand=hands)


def move_pick_geoms(mj_model, scenario):
    """Reset free pick-cube poses and rebuild every offered figure at its sampled palm pose.

    Pick poses write the free-joint entries in ``mj_model.qpos0``; callers then reset their data from
    qpos0. This preserves one shared scene: kinematic replay keeps cubes fixed at their sampled poses
    while dynamic physics starts from those exact poses and subsequently owns their motion through
    contact.
    """
    for m in scenario.pick:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"pp_obj_{m.name}")
        assert body_id >= 0, f"pick body pp_obj_{m.name} not found"
        joint_id = int(mj_model.body_jntadr[body_id])
        assert joint_id >= 0 and mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE, (
            f"pick body pp_obj_{m.name} must have one free joint")
        qadr = int(mj_model.jnt_qposadr[joint_id])
        mj_model.qpos0[qadr:qadr + 3] = m.pos
        mj_model.qpos0[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
    for hand_marker in scenario.hand:
        _rebuild_figure_geoms(mj_model, hand_marker)


def _rebuild_figure_geoms(mj_model, hand_marker) -> None:
    """Rewrite one welded ``human_figure``'s geoms in place for the marker's sampled palm pose.

    REBUILT, not translated. The figure is FLOOR-ANCHORED (``human_figure``: the body stands at a fixed
    absolute ``shoulder_z`` and the arm bend absorbs the palm height), so rigidly shifting every geom by
    the sampled delta -- what this did before -- carried the sampled dz into the legs and sank the feet
    up to 17 mm through the floor. Re-running the builder puts the palm exactly where the marker says
    while the feet stay planted, and it is the same call ``make_scene_spec_fn`` used to author the geoms,
    so names, order and count match by construction.

    Capsules are authored from world endpoints, which MuJoCo's compiler had already folded into
    pos/quat/half-length; recompute all three rather than only the position, since a changed palm height
    re-solves the elbow and therefore re-aims the arm capsules. The rebuilt capsule quats differ from the
    compiler's by a roll about the capsule axis -- verified irrelevant (a capsule is rotationally
    symmetric, and both the renderer and ``ik_curobo._capsule_world_aabb`` read only the axis), while every
    non-capsule geom reproduces bit-exactly.
    """
    axis = np.zeros(3)
    for piece in human_figure(hand_marker.name, *hand_marker.pos, yaw_deg=hand_marker.yaw_deg,
                              collidable_palm=hand_marker.collidable):
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"pp_{piece.name}")
        assert geom_id >= 0, f"figure geom pp_{piece.name} not found"
        if piece.fromto is None:
            mj_model.geom_pos[geom_id] = piece.center
            mj_model.geom_quat[geom_id] = piece.quat
            continue
        p0, p1 = np.asarray(piece.fromto[:3]), np.asarray(piece.fromto[3:])
        axis[:] = p1 - p0
        half_len = np.linalg.norm(axis) / 2.0
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, axis / (2.0 * half_len))
        mj_model.geom_pos[geom_id] = (p0 + p1) / 2.0
        mj_model.geom_quat[geom_id] = quat
        mj_model.geom_size[geom_id, 1] = half_len


def cube_world_poses(mj_model, scenario, mj_data=None) -> dict:
    """SEAM (cube-pose source): world-frame ``{name: (center_world[3], dims[3])}`` for each pick cube.

    MuJoCo backend reads live ``pp_mark_*`` geom positions. When no data is supplied, it forwards a
    qpos0 data instance for static planning callers. Pick cubes are free bodies, so using model geom
    positions would return their local origin and silently plan at (0, 0, 0). A future perception backend
    returns the SAME dict shape from sensors. This is the ONE place world cube geometry enters the planner
    path; every target/obstacle computation below is source-agnostic.
    """
    if mj_data is None:
        mj_data = mujoco.MjData(mj_model)
        mj_data.qpos[:] = mj_model.qpos0
        mujoco.mj_forward(mj_model, mj_data)
    poses = {}
    for pick_marker in scenario.pick:
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"pp_mark_{pick_marker.name}")
        assert geom_id >= 0, f"object marker geom pp_mark_{pick_marker.name} not found"
        assert mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX, (
            f"pick marker pp_mark_{pick_marker.name} must be a box")
        center_world = mj_data.geom_xpos[geom_id]
        box_dims = 2.0 * mj_model.geom_size[geom_id]
        poses[pick_marker.name] = (np.asarray(center_world, dtype=float), np.asarray(box_dims, dtype=float))
    return poses


# ------------------------------------------------------------------------------------------------
# Namespace-tolerant robot-part lookups: the KINEMATIC harness compiles a BARE-name model (``base_link``),
# the DYNAMIC (mjlab/warp) env namespaces every robot part as ``robot/base_link``. The camera-gate helpers
# resolve descriptor names by trying the bare name first, then the ``robot/`` prefix, so ONE code path
# serves both harnesses (scene props like ``pp_mark_*`` are never namespaced -- only robot parts are).
# ------------------------------------------------------------------------------------------------
def _ns_id(mj_model, objtype, name) -> int:
    i = mujoco.mj_name2id(mj_model, objtype, name)
    if i == -1:
        i = mujoco.mj_name2id(mj_model, objtype, f"robot/{name}")
    return i


def _ns_joint_qadr(mj_model, name) -> int:
    return int(mj_model.jnt_qposadr[_ns_id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)])


# ------------------------------------------------------------------------------------------------
# Visibility gate: FIXED head camera(s) -> cubes inside the live frustum (static, no gaze slew)
# ------------------------------------------------------------------------------------------------
def cube_visible_poses(mj_model, mj_data, scenario, descriptor: RobotDescriptor) -> dict:
    """SEAM (perception): world poses of the cubes the robot can SEE from the current base + head pose.

    The robots carry FIXED head cameras (no gimbal slew), so "seen" is a static frustum test: a cube is
    visible if it falls inside ANY ``descriptor.head_cam_sites`` camera's live FOV+range at the pose
    ``mj_data`` currently holds AND the sight line to it is unobstructed (``_sight_line_clear``: a cube
    behind a shelf board / bench / other cube is NOT seen). Reuses the camera policy's ``fov_detect``
    primitive (the same detector the gaze eval uses). Returns the detected SUBSET of
    ``cube_world_poses`` -- the belief the reach is allowed to use, never raw GT. A real detector swaps
    this ONE function, leaving reach math untouched, exactly like ``cube_world_poses``.

    Caller must have ``mj_forward``-ed ``mj_data`` at the pose to test (fixed cameras follow the head).
    """
    cubes = cube_world_poses(mj_model, scenario, mj_data)
    site_ids = [_ns_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name) for name in descriptor.head_cam_sites]
    visible = {}
    for cid, (center, dims) in cubes.items():
        for sid in site_ids:
            cam_pos = torch.as_tensor(mj_data.site_xpos[sid], dtype=torch.float32)
            cam_mat = torch.as_tensor(mj_data.site_xmat[sid].reshape(3, 3), dtype=torch.float32)
            if _cube_visible_from_camera(mj_model, mj_data, cam_pos, cam_mat, center, dims):
                visible[cid] = (center, dims)
                break
    return visible


# ------------------------------------------------------------------------------------------------
# Object-INDEPENDENT anchor perception: aim gimbals at KNOWN supports, detect whatever is framed
# ------------------------------------------------------------------------------------------------
def known_anchors(scenario, viewpoint_xy) -> list[tuple[str, np.ndarray, np.ndarray, float | None, np.ndarray | None]]:
    """Object-INDEPENDENT support anchors ``[(name, world_xyz, tangent_xy, half_width, face_normal_xy), ...]`` the
    robot knows a priori.

    The non-privileged prior is "objects rest on KNOWN supports", NOT "object X is at pose P" -- so this
    reads ONLY prop/hand geometry, never a pick pose. Support surfaces:
      * workbench tops: the visual slab ``{table}_top``. Excludes the coincident collision piece
        ``{table}_col_top`` so each bench yields ONE anchor.
      * shelf tiers: pieces named ``{shelf}_tierK`` (shelves have no ``_top`` piece).
      * human-hand markers: each ``scenario.hand`` point.

    A support's anchor is its ROBOT-FACING near-edge center, not the centroid: the centroid is offset half
    the slab depth (``prop.half[0]``) toward ``viewpoint_xy`` (the robot's known base XY), where near-edge
    objects sit. Aiming / walking to the edge frames those objects without overshooting the far centroid
    (a forward camera that stops a fixed radius from the centroid drives PAST an edge object). Uses only
    the robot's own pose + known table geometry (no object pose). Assumes anchors do not overlap in XY.

    Each entry also carries the anchor's own world-XY unit ``tangent`` (its local width axis: desk/shelf
    edge direction for a Prop, hand-width direction perpendicular to the finger-point axis for a Marker --
    fixed by the anchor's own yaw, NOT the viewpoint) and its ``half_width`` along that axis if bounded
    (``Prop.half[1]``; ``None`` for a Marker -- a point/capsule cluster, no bounding box). A caller can
    clamp a lateral target to ``half_width`` so it never walks past the anchor's own physical edge. Bounded
    supports additionally carry the signed face normal pointing from support toward the current viewpoint;
    it is the collision-clearance normal, not a target-drive direction. Hand markers have no support face.
    """
    view = np.asarray(viewpoint_xy, dtype=float)[:2]
    anchors = []
    for prop in scenario.props:
        if not ((prop.name.endswith("_top") and not prop.name.endswith("_col_top"))
                or "_tier" in prop.name):
            continue
        w, x, y, z = prop.quat                        # pure-yaw quat (_yawq, scene_object/base.py)
        yaw = 2.0 * math.atan2(z, w)
        tangent = np.array([-math.sin(yaw), math.cos(yaw)])
        edge = np.asarray(prop.center, dtype=float)
        # The near edge lies on the slab's OWN depth axis (perpendicular to ``tangent``), offset by half the
        # slab depth; the viewpoint only picks WHICH of the two opposite faces is the near one (sign). An
        # earlier version instead offset RADIALLY along the centroid->viewpoint ray and returned that ray as
        # ``face_normal``. For a head-on support the two agree, but for a support viewed from off-axis (flank
        # table in `left_right_*`, resolved while the base stands at the OTHER table) the radial version
        # returns a diagonal phantom face: measured on g1/left_right_close seed 47 it placed table_R's edge at
        # (0.117, -0.279) with normal (0.46, 0.89) instead of (0.0, -0.25) / (0.0, 1.0) -- a 0.117 m lateral
        # anchor error, and a diagonal normal that inflates the projected torso footprint in
        # ``_support_clearance_at`` (0.227 vs 0.180 for g1's square footprint), so APPROACH braked ~0.05 m
        # farther out than the table physically requires and then declared the cube unreachable.
        normal = np.array([tangent[1], -tangent[0]])   # slab depth axis (unit: tangent is unit)
        offset_to_view = float(np.dot(view - edge[:2], normal))
        face_normal = normal if offset_to_view >= 0.0 else -normal
        edge[:2] += prop.half[0] * face_normal
        anchors.append((prop.name, edge, tangent, float(prop.half[1]), face_normal))
    for hand_marker in scenario.hand:
        if not hand_marker.collidable:
            continue
        yaw = math.radians(hand_marker.yaw_deg)
        tangent = np.array([-math.sin(yaw), math.cos(yaw)])
        anchors.append((hand_marker.name, np.asarray(hand_marker.pos, dtype=float), tangent, None, None))
    return anchors


def detect_at_anchors(mj_model, mj_data, scenario, descriptor: RobotDescriptor, anchors) -> dict:
    """SEAM (gimbal perception): aim each actuated head camera at KNOWN anchors, return the world poses
    of whatever pick cubes fall inside the slewed frustum -- object-independent belief.

    Non-privileged: aim targets are ``anchors`` (support surfaces), never a cube pose. Each physical
    gimbal points ONE way, so anchors are the outer loop and each anchor claims the first UNUSED camera
    that can frame a cube there (closed-form aim, BASE frame: ``pitch=asin(-d_z)``,
    ``yaw=atan2(-sign*d_y,sign*d_x)``). NOTE the ``-sign`` on d_y: the humanoid-ATTACHED gimbal flips the
    yaw sense vs the standalone ``duke_v2/head_camera_dual.xml`` that ``policies._compute_camera`` was
    verified against (there yaw is ``atan2(sign*d_y,sign*d_x)``); on this asset that formula mis-aims ~57
    deg off-axis. Verified: aim_error->0 for both cams over the full ROM. A claimed camera LATCHES those
    joints (viewer shows the gimbal on its anchor for replay capture); an anchor whose aim exceeds the
    gimbal ROM, or that has no free camera, is skipped -> its cubes stay UNSEEN. Unused cameras reset to
    neutral. Confirmation reads GT ``cube_world_poses`` behind the detector -- the same legitimate pattern
    as ``cube_visible_poses``. Returns the detected SUBSET ``{id: (center, dims)}``; leaves the latched
    cam qpos in ``mj_data`` for the caller to capture.
    """
    cubes = cube_world_poses(mj_model, scenario, mj_data)
    base_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_BODY, descriptor.base_link)
    detected, used = {}, set()
    for _name, anchor, *_ in anchors:
        for site, yaw_joint, pitch_joint, sign in descriptor.gaze_cams:
            if site in used:
                continue
            site_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
            R_base = mj_data.xmat[base_id].reshape(3, 3)
            d = R_base.T @ (anchor - mj_data.site_xpos[site_id])
            yaw, pitch = _solve_gaze(d, sign)
            if abs(yaw) > _GAZE_YAW_ROM or abs(pitch) > _GAZE_PITCH_ROM:
                continue                                    # anchor outside this gimbal's ROM
            mj_data.qpos[_ns_joint_qadr(mj_model, yaw_joint)] = yaw
            mj_data.qpos[_ns_joint_qadr(mj_model, pitch_joint)] = pitch
            mujoco.mj_forward(mj_model, mj_data)
            cam_pos = torch.as_tensor(mj_data.site_xpos[site_id], dtype=torch.float32)
            cam_mat = torch.as_tensor(mj_data.site_xmat[site_id].reshape(3, 3), dtype=torch.float32)
            hits = {}
            for cid, (center, dims) in cubes.items():
                if _cube_visible_from_camera(mj_model, mj_data, cam_pos, cam_mat, center, dims):
                    hits[cid] = (center, dims)
            if hits:
                detected.update(hits)
                used.add(site)
                break                                       # anchor covered; next anchor
    for site, yaw_joint, pitch_joint, _sign in descriptor.gaze_cams:
        if site not in used:
            mj_data.qpos[_ns_joint_qadr(mj_model, yaw_joint)] = 0.0
            mj_data.qpos[_ns_joint_qadr(mj_model, pitch_joint)] = 0.0
    mujoco.mj_forward(mj_model, mj_data)
    return detected


def gaze_sees(mj_model, mj_data, descriptor: RobotDescriptor, center) -> bool:
    """Can ANY actuated gimbal, aimed at CENTER, frame it? Per-OBJECT visibility test for the reach gate.

    ``detect_at_anchors`` is object-independent and GREEDY -- it aims per anchor and latches a camera on the
    first cube it happens to frame, so it cannot answer "does a camera see THIS specific cube". This aims
    each gimbal directly at ``center`` (one-shot closed-form, shared ``gaze_aim``) and runs ONE ``fov_detect``
    on the target. Aiming AT the cube (not the near-edge anchor) keeps the parallax residual small vs the
    wide 90x65 deg frustum, so a single solve suffices (the rejected iterative re-solve is gone; closed-loop
    parallax correction belongs to the per-step control loop, not this offline gate). Mutates ``mj_data``
    gimbal joints (caller passes scratch data)."""
    base_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_BODY, descriptor.base_link)
    R_base = mj_data.xmat[base_id].reshape(3, 3)
    center = np.asarray(center, dtype=float)
    target = torch.as_tensor(center, dtype=torch.float32)
    for site, yaw_joint, pitch_joint, sign in descriptor.gaze_cams:
        site_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
        d = R_base.T @ (center - mj_data.site_xpos[site_id])
        yaw, pitch = _solve_gaze(d, sign)
        if abs(yaw) > _GAZE_YAW_ROM or abs(pitch) > _GAZE_PITCH_ROM:
            continue
        mj_data.qpos[_ns_joint_qadr(mj_model, yaw_joint)] = yaw
        mj_data.qpos[_ns_joint_qadr(mj_model, pitch_joint)] = pitch
        mujoco.mj_forward(mj_model, mj_data)
        cam_pos = torch.as_tensor(mj_data.site_xpos[site_id], dtype=torch.float32)
        cam_mat = torch.as_tensor(mj_data.site_xmat[site_id].reshape(3, 3), dtype=torch.float32)
        if bool(fov_detect(cam_pos, cam_mat, target, H_HALF, V_HALF, NEAR, FAR)["in_fov"]) \
                and _sight_line_clear(mj_model, mj_data, mj_data.site_xpos[site_id], center):
            return True
    return False


# Gaze-assignment cost weights, all in radians so they trade off against the raw ``total|yaw|`` term.
# Sized by what each preference must be able to OUTRANK (see ``aim_gaze_at_cubes``):
_GAZE_CROSS_W = 4.0     # per cross-body pairing. > pi, so crossing never wins on the yaw it saves.
_GAZE_SWITCH_W = 2.0    # per rad of yaw re-aim. A 180 deg swap costs ~6.3 > _GAZE_CROSS_W, so NO single
                        # soft term can force a re-aim; only a coverage change (lexicographic) can.
_GAZE_MARGIN_W = 0.05   # weak pull off ROM-pinned poses; deliberately too small to reorder anything else.


def _wrap_pi(a: float) -> float:
    """Signed angle wrapped to (-pi, pi] -- shortest rotation between two yaw commands."""
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def aim_gaze_at_cubes(mj_model, mj_data, descriptor: RobotDescriptor, cube_centers) -> dict:
    """Assign each gimbal a DISTINCT detected cube (1:1) and return the joint targets. FOR THE
    HELD/DISPLAYED gaze only -- detection stays object-independent (anchor aim). Aiming at the near-EDGE
    anchor over-depresses to a near-vertical, pitch-limit-pinned pose on CLOSE side tables (the anchor sits
    almost under the head); the cube sits further onto the table, so aiming at it gives a natural elevation.
    Closed-form aim (same datum as ``detect_at_anchors``), ROM-clamped.

    BIJECTIVE assignment (not independent per-cam nearest-yaw): the right gimbal's zero-yaw datum faces
    BACKWARD (``sign=-1``), so every cube needs a ~180 deg yaw and both cams' independent min-``|yaw|``
    pick collapses onto the SAME cube -- leaving the other cube untracked in the viewer. Instead score every
    injective cam->cube map over the (small) gimbal/cube sets and take the best-scoring one. A cam with no
    feasible cube is skipped (left at neutral). Mutates ``mj_data`` gimbal joints (caller passes scratch
    data).

    SCORE = ``(-covered, cost)``: coverage is the only LEXICOGRAPHIC key; everything else is one additive
    continuous ``cost``. This shape is deliberate and replaces an all-lexicographic
    ``(-covered, cross, total, -worst_margin)`` key. Under a lexicographic key, whichever term ranks first
    decides ALONE, so a term that is float-noisy near its own decision boundary flips the entire winning
    permutation every tick -- and the two winners aim the SAME gimbal ~180 deg apart, so the rate limiter
    below turns that into a sustained physical bounce. That failure was hit three separate times, each on a
    DIFFERENT noisy term (lens-derived side classification, then ``worst_margin``, then the ``cross`` sign
    test on a front/back layout where every ``cube_lat`` is ~0). Patching the noisy term of the day is a
    treadmill; making every soft preference an additive contribution is the fix that retires the class,
    because no single term can now flip the winner on its own -- a challenger must beat the incumbent's
    TOTAL, including the switching cost below.

    ``cost`` terms, all in radians so the weights are physically comparable:
      * ``total`` -- sum of in-ROM ``|yaw|``. Don't crane the neck.
      * ``_GAZE_CROSS_W * cross`` -- penalty per cam reaching to the far-side cube (self-occludes, pins
        pitch). Weight exceeds pi so a cross-body pairing can never win on saved yaw alone, which is what
        the old hard lexicographic ordering of this term bought.
      * ``_GAZE_SWITCH_W * slew`` -- HYSTERESIS, and the term that actually kills the bounce. Cost of the
        yaw rotation each cam must still travel from its CURRENT command to that permutation's aim. The
        incumbent assignment is already (partly) slewed to, so its ``slew`` decays toward 0 while a
        challenger's stays large; a challenger must be better by the full cost of the re-aim to win. This
        needs NO stored previous assignment: current gimbal command is read straight off ``mj_data``, which
        also means it stays correct across a switch of what is being tracked (anchors vs cubes) and never
        depends on list-index identity holding between ticks.
      * ``-_GAZE_MARGIN_W * worst_margin`` -- fractional ROM headroom of the tightest axis; kept as a weak
        preference away from pinned poses. Small weight ON PURPOSE: this term is near-invariant across
        competing permutations, so its inter-permutation differences are mostly float noise.
    """
    base_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_BODY, descriptor.base_link)
    R_base = mj_data.xmat[base_id].reshape(3, 3)
    base_pos = mj_data.xpos[base_id]
    centers = [np.asarray(c, dtype=float) for c in cube_centers]
    cams = list(descriptor.gaze_cams)
    # Lateral (base-y) offset of each cam site and each cube, measured in the base frame. Their signed
    # product classifies a pairing as same-side (>0) or cross-body (<0); used to PREFER same-side maps.
    # Geometric, not name-based, so it generalizes to any cam layout: the crossed map (a cam reaching to
    # the far-side cube) has smaller total |yaw| for a backward-datum cam, so a pure min-yaw score picks
    # it -- forcing a cross-body reach that occludes and pins pitch near its limit.
    cube_lat = [float((R_base.T @ (c - base_pos))[1]) for c in centers]
    cam_lat = {}
    # In-ROM aim solution per (cam, cube); missing key => that pairing is infeasible. ``margin`` is the
    # fractional headroom to whichever axis (yaw or pitch) sits CLOSER to its own ROM, normalized so the
    # two axes (different ROM magnitudes) compare on the same [0, 1] scale -- 0 means pinned AT the limit.
    sol = {}
    cur_yaw = {}
    for ci, (site, yaw_joint, _pj, sign) in enumerate(cams):
        site_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
        # Gimbal's CURRENT yaw command, read before this function overwrites it below -- the datum the
        # switching cost is measured from. Caller has already written its live command into this scratch.
        cur_yaw[ci] = float(mj_data.qpos[_ns_joint_qadr(mj_model, yaw_joint)])
        # STRUCTURAL side, not the current optical position: ``xanchor`` is the yaw joint's own pivot,
        # invariant to THAT joint's rotation (only ancestor/base kinematics move it) -- unlike
        # ``site_xpos`` (the lens), which swings across body-center as commanded yaw nears its far side.
        # Deriving cam_lat from the lens made the "same-side" classification depend on the gimbal's OWN
        # current (lagging, slew-limited) command; near the antipodal region that fed back into the very
        # assignment choosing that command, producing a persistent every-tick assignment flip (verified:
        # seed 182 front_back_close alternated the full bijective cam<->cube map every tick).
        yaw_jid = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, yaw_joint)
        cam_lat[ci] = float((R_base.T @ (mj_data.xanchor[yaw_jid] - base_pos))[1])
        for uj, center in enumerate(centers):
            yaw, pitch = _solve_gaze(R_base.T @ (center - mj_data.site_xpos[site_id]), sign)
            if abs(yaw) <= _GAZE_YAW_ROM and abs(pitch) <= _GAZE_PITCH_ROM:
                margin = min(1.0 - abs(yaw) / _GAZE_YAW_ROM, 1.0 - abs(pitch) / _GAZE_PITCH_ROM)
                sol[(ci, uj)] = (yaw, pitch, margin)
    # Best injective cam->cube map: pad cube slots with None when cams outnumber cubes so every
    # permutation assigns one distinct slot per cam. See the docstring for why coverage is the only
    # lexicographic key and every soft preference is summed into one continuous ``cost`` instead.
    cube_slots = list(range(len(centers))) + [None] * max(0, len(cams) - len(centers))
    best_assign, best_key = None, None
    for perm in itertools.permutations(cube_slots, len(cams)):
        total, covered, cross, slew = 0.0, 0, 0, 0.0
        worst_margin = float("inf")
        feasible = True
        for ci, uj in enumerate(perm):
            if uj is None:
                continue
            if (ci, uj) not in sol:
                feasible = False
                break
            yaw, _pitch, margin = sol[(ci, uj)]
            total += abs(yaw)
            covered += 1
            worst_margin = min(worst_margin, margin)
            if cam_lat[ci] * cube_lat[uj] < 0:
                cross += 1
            # Shortest signed rotation: a backward-datum cam aims near +-pi, so the raw difference would
            # read the same physical direction as a ~2*pi slew and mis-rank the incumbent as expensive.
            slew += abs(_wrap_pi(yaw - cur_yaw[ci]))
        if not feasible:
            continue
        cost = (total + _GAZE_CROSS_W * cross + _GAZE_SWITCH_W * slew
                - _GAZE_MARGIN_W * (0.0 if worst_margin == float("inf") else worst_margin))
        if best_key is None or (-covered, cost) < best_key:
            best_key, best_assign = (-covered, cost), perm
    cam_qpos = {}
    if best_assign is not None:
        for ci, uj in enumerate(best_assign):
            if uj is None:
                continue
            _site, yaw_joint, pitch_joint, _sign = cams[ci]
            yaw, pitch, _margin = sol[(ci, uj)]
            mj_data.qpos[_ns_joint_qadr(mj_model, yaw_joint)] = yaw
            mj_data.qpos[_ns_joint_qadr(mj_model, pitch_joint)] = pitch
            cam_qpos[yaw_joint], cam_qpos[pitch_joint] = yaw, pitch
    pose_only_forward(mj_model, mj_data)     # gaze scratch: pose queries only, see pose_only_forward
    return cam_qpos


# ------------------------------------------------------------------------------------------------
# Rate-limited head-gimbal SCAN (actuated cameras): honest search + physically realistic slew
# ------------------------------------------------------------------------------------------------
_GAZE_SLEW_MAX = 3.5        # max gimbal joint speed (rad/s): hardware-realistic fast pan/tilt (~200 deg/s)
_GAZE_SETTLE_EPS = 0.035    # |cmd - desired| (rad) below which a gimbal counts as ON its aim target (~2.0 deg)


def solve_gaze_to(mj_model, mj_data, descriptor, target_world) -> dict:
    """Closed-form in-ROM gaze joint angles ``{joint: angle}`` to aim each actuated cam at ``target_world``
    from the CURRENT base pose. Read-only (does NOT mutate ``mj_data``). Cams whose aim exceeds ROM are
    omitted. Shares the ``_solve_gaze`` datum used by ``detect_at_anchors``/``gaze_sees``."""
    base_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_BODY, descriptor.base_link)
    R_base = mj_data.xmat[base_id].reshape(3, 3)
    target = np.asarray(target_world, dtype=float)
    out = {}
    for site, yaw_joint, pitch_joint, sign in descriptor.gaze_cams:
        site_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
        yaw, pitch = _solve_gaze(R_base.T @ (target - mj_data.site_xpos[site_id]), sign)
        if abs(yaw) <= _GAZE_YAW_ROM and abs(pitch) <= _GAZE_PITCH_ROM:
            out[yaw_joint], out[pitch_joint] = float(yaw), float(pitch)
    return out


def detect_at_gaze(mj_model, mj_data, scenario, descriptor) -> dict:
    """SEAM (honest per-tick perception): the pick cubes framed by the gaze cams at their CURRENT joint
    pose. Unlike ``detect_at_anchors`` (which aims-then-detects EVERY anchor at once -- omniscient within
    ROM), this reads only what the gimbals point at RIGHT NOW, so a cube enters belief solely once the live
    sweep frames it. Caller has already written the gaze qpos + ``mj_forward``-ed ``mj_data``."""
    cubes = cube_world_poses(mj_model, scenario, mj_data)
    detected = {}
    for site, _yaw_joint, _pitch_joint, _sign in descriptor.gaze_cams:
        site_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
        cam_pos = torch.as_tensor(mj_data.site_xpos[site_id], dtype=torch.float32)
        cam_mat = torch.as_tensor(mj_data.site_xmat[site_id].reshape(3, 3), dtype=torch.float32)
        for cid, (center, dims) in cubes.items():
            if _cube_visible_from_camera(mj_model, mj_data, cam_pos, cam_mat, center, dims):
                detected[cid] = (center, dims)
    return detected




def object_cuboids(scenario, to_base_frame, base_quat, cube_poses, height_pad: float = 0.02) -> dict:
    """Each OBSERVED cube as a cuRobo Cuboid so the planner AVOIDS every object -- including the one it
    grasps (a parallel gripper straddles the 50 mm cube: grasp site at the cube, finger-rack spheres
    pass outside, verified feasible). World-axis-aligned box expressed in base_link frame.

    When ``height_pad > 0``, inflates the Z height in CuRobo's obstacle world perspective, extending
    the top upward while keeping the bottom face flush on the table surface. Physical MuJoCo geoms and
    grasp targets stay untouched.

    Iterates ``cube_poses`` keys (NOT ``scenario.pick``), so a camera-visible SUBSET yields obstacle
    boxes only for the seen cubes -- an unseen cube is neither a target nor an obstacle. ``scenario`` is
    retained for the call signature; full-GT callers pass ``cube_poses`` keyed by every pick cube."""
    # Pick marker boxes are world-axis-aligned. In base coordinates they carry the inverse
    # base orientation, matching `build_curobo_scene` for fixed workbench obstacles.
    world_axes_in_base_quat = [base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]]
    cuboids = {}
    for name, (center_world, box_dims) in cube_poses.items():
        dx, dy, dz = box_dims
        dz_planner = float(dz + height_pad)
        center_planner_world = np.array(center_world, dtype=float)
        if height_pad != 0.0:
            center_planner_world[2] += height_pad / 2.0
        cuboids[f"pp_obj_{name}"] = {
            "dims": [float(dx), float(dy), dz_planner],
            "pose": to_base_frame(center_planner_world).tolist() + world_axes_in_base_quat,
        }
    return cuboids


def _cube_target_base(name: str, robot_cfg: RobotDescriptor, to_base_frame, cube_poses) -> np.ndarray:
    """Base-frame grasp target for one cube, per the robot's ``target_mode``."""
    center_world, _ = cube_poses[name]
    center_base = to_base_frame(center_world)
    if robot_cfg.target_mode == "center":
        return np.asarray(center_base, dtype=float)
    assert robot_cfg.target_mode == "center_clamp", robot_cfg.target_mode
    return np.array([center_base[0], center_base[1], max(float(center_base[2]), robot_cfg.grasp_z_floor)])


# ------------------------------------------------------------------------------------------------
# Bridge: scene + reach targets (object cubes -> robot base-frame grasp/transit targets)
# ------------------------------------------------------------------------------------------------
def build_bimanual_scene_and_targets(scenario, robot_cfg: RobotDescriptor, mj_model, height_pad: float = 0.02):
    """(curobo scene dict, [(name, target_base), (name, target_base)]) for a close 2-object scene.
    Assignment-agnostic (caller decides object->hand later). Target from the cube geom via the robot's
    grasp rule: ``center`` = cube center (G1's tilted grasp applies its per-candidate collision-safe
    pre-grasp standoff); ``center_clamp`` = cube center Z clamped up to the hand z-floor (humanoid)."""
    assert not isinstance(scenario, str), (
        "pass the height-correct Scenario object from scene.robot_scene(robot, name), not a name: "
        "resolving a name here would silently use the default-height SCENARIOS and diverge from the "
        "per-robot table_z.")
    assert len(scenario.pick) == 2, f"bimanual needs exactly 2 pick objects; {scenario.name!r} has {len(scenario.pick)}"
    cuboids, to_base_frame = build_curobo_scene(
        mj_model, robot_cfg.base_link, base_pos=robot_cfg.home_base_pos, base_quat=robot_cfg.home_base_quat
    )
    cube_poses = cube_world_poses(mj_model, scenario)
    objects = []
    for pick_marker in scenario.pick:
        objects.append((pick_marker.name, _cube_target_base(pick_marker.name, robot_cfg, to_base_frame, cube_poses)))
    cuboids.update(object_cuboids(scenario, to_base_frame, robot_cfg.home_base_quat, cube_poses, height_pad=height_pad))
    return {"cuboid": cuboids}, objects


def build_scene_and_targets(scenario, robot_cfg: RobotDescriptor, mj_model, cube_poses=None, height_pad: float = 0.02):
    """(curobo scene dict, {"front": pos_base, "rear": pos_base}) for the single-arm regression.
    front = max world-x pick, rear = min. Targets are the cube GRASP point (``_cube_target_base``:
    center, or center clamped up to ``grasp_z_floor``) -- the SAME target phase-3's grasp gate proves
    and phase-4 ``--planned`` reaches. NOT a separate transit-hover Z (removed): reach the cube exactly
    as phase-3 does. The grasped cube stays a collision obstacle (``object_cuboids``): the parallel
    gripper straddles the 50 mm cube, grasp site at the cube with finger spheres outside.

    ``cube_poses`` (perception seam): world ``{name: (center[3], dims[3])}`` for the pick cubes. ``None``
    reads GROUND TRUTH from ``mj_model`` (``cube_world_poses``); an OBSERVED dict (detector output, same
    shape) can be injected instead so the target tracks perception, not sim state. ``mj_model`` still
    supplies the static workbench OBSTACLES either way (the planner's world model)."""
    assert not isinstance(scenario, str), (
        "pass the height-correct Scenario object from scene.robot_scene(robot, name), not a name: "
        "resolving a name here would silently use the default-height SCENARIOS and diverge from the "
        "per-robot table_z.")
    cuboids, to_base_frame = build_curobo_scene(
        mj_model, robot_cfg.base_link, base_pos=robot_cfg.home_base_pos, base_quat=robot_cfg.home_base_quat
    )
    cube_poses = cube_world_poses(mj_model, scenario) if cube_poses is None else cube_poses
    pick_by_x = {"front": max, "rear": min}
    targets = {}
    for target_name, reducer in pick_by_x.items():
        marker = reducer(scenario.pick, key=lambda m: m.pos[0])
        targets[target_name] = _cube_target_base(marker.name, robot_cfg, to_base_frame, cube_poses)
    print(f"grasp targets (base): {targets}")
    cuboids.update(object_cuboids(scenario, to_base_frame, robot_cfg.home_base_quat, cube_poses, height_pad=height_pad))
    return {"cuboid": cuboids}, targets


def build_reach_scene(scenario, robot_cfg: RobotDescriptor, mj_model, cube_poses=None, height_pad: float = 0.02):
    """(curobo scene dict, ``{name: target_base}``) for the OBSERVED pick cubes -- assignment-agnostic
    AND count-agnostic (1 or 2 cubes). The reachability-driven producer's scene builder: the caller
    later assigns targets to arms.

    ``cube_poses`` (perception seam): world ``{name: (center[3], dims[3])}``. ``None`` = GROUND TRUTH
    from ``mj_model`` (``cube_world_poses``); a detector's VISIBLE subset (same shape) reaches only the
    seen cubes -- both their grasp targets AND their collision cuboids derive from ``cube_poses`` keys,
    so an unseen cube is neither reached nor an obstacle. ``mj_model`` supplies the static workbench
    OBSTACLES either way (the planner's world model)."""
    assert not isinstance(scenario, str), (
        "pass the height-correct Scenario object from scene.robot_scene(robot, name), not a name")
    cuboids, to_base_frame = build_curobo_scene(
        mj_model, robot_cfg.base_link, base_pos=robot_cfg.home_base_pos, base_quat=robot_cfg.home_base_quat
    )
    cube_poses = cube_world_poses(mj_model, scenario) if cube_poses is None else cube_poses
    targets = {name: _cube_target_base(name, robot_cfg, to_base_frame, cube_poses) for name in cube_poses}
    cuboids.update(object_cuboids(scenario, to_base_frame, robot_cfg.home_base_quat, cube_poses, height_pad=height_pad))
    return {"cuboid": cuboids}, targets


@dataclass(frozen=True)
class ReachRoute:
    """One base-live reach solve over 1 OR 2 active arms: the arm-trajectory command buffer's payload.

    ``route_q_curobo`` is cuRobo ACTIVE-ARM cspace (``joint_names`` order = every arm DOF), NOT mjlab
    qpos -- the consumer adds ``qpos0[qposadr]`` per joint to cross the cspace->mjlab seam (the SAME
    offset phase-3 replay and ``CuroboArmPlanner.to_mjlab_qpos`` use). An IDLE arm simply holds its home
    columns. ``reaches`` maps each ACTIVE tool frame to ``(target_base[3], grasp_cand_base[G,3],
    reach_error)`` in the LIVE base frame the route was solved in; the verdict compares each frame's
    reached site to ITS ``grasp_cand_base``. ``controlled_joints`` = the union of active arms' joints the
    executor drives. The single-active convenience properties (``reach_frame``/``target_base``/
    ``grasp_cand_base``) subsume the former single-arm fields for one-arm callers."""
    joint_names: tuple[str, ...]
    route_q_curobo: np.ndarray          # (H, n_arm_dof) cuRobo cspace (all arm DOFs)
    interpolation_dt: float
    reaches: dict                       # {frame: (target_base[3], grasp_cand_base[G,3], reach_error)}
    controlled_joints: tuple[str, ...]  # active arms' joints (the executor's driven columns)
    base_pos: np.ndarray                # (3,) base position the route was SOLVED in (world verdict anchor)
    base_quat: np.ndarray               # (4,) base quat (wxyz) the route was solved in (yaw-only planar)
    assignment: dict                    # {side: cube_name} the ladder chose; a REPLAN reuses it verbatim
                                        # (no re-ladder, no single-arm fallback -> count stays invariant)

    @property
    def reach_error(self) -> float:
        """Max planned final tool-site error over ACTIVE frames (the kinematic phase-3 gate value)."""
        return max(err for _t, _c, err in self.reaches.values())

    @property
    def reach_frame(self) -> str:
        """The single active frame (single-arm compat); asserts exactly one active frame."""
        assert len(self.reaches) == 1, f"reach_frame needs a single-active route; {len(self.reaches)} active"
        return next(iter(self.reaches))

    @property
    def target_base(self) -> np.ndarray:
        return self.reaches[self.reach_frame][0]

    @property
    def grasp_cand_base(self) -> np.ndarray:
        return self.reaches[self.reach_frame][1]


# A bimanual pairing solves 2x the DOF with an added inter-arm collision constraint (vs. a lone arm), so
# the planner's general plan-0 default (10 attempts) can spuriously report an actually-feasible pairing as
# infeasible and silently fall back to a sequential single-arm reach (live-test finding: v2 left_right_close
# seed 42 -- the SAME pairing that fails at 10-18 attempts lands at 20-30). Only raises plan-0 (``max_attempts
# is None``); an explicit replan cap (e.g. ``1``, cheap warm-seeded correction) is left untouched.
#
# 2026-08-02: 30 was STILL too low on the DYNAMIC path, and the fix is NOT a bigger number here -- it is
# ``num_trajopt_seeds`` (see ``ik_curobo_robot_cfg.build_motion_planner_kwargs``). Recorded because the
# measurement invalidates this constant's own escalation path. The kinematic harness commits from a
# squared-up stance while physics parks the base ~14 deg further off (v2 ``bimanual_mixed_front_back_close``
# seed 47: the same cube_0 sits at base-frame y 0.026 kinematic vs 0.093 dynamic), which pushes the pairing
# to the edge of its basin without making it infeasible. At 4 trajopt seeds BOTH pairings failed at this cap
# and the visit silently degraded to two sequential single-arm reaches; a cap of 150 recovered the pairing but
# burned 2.17 s of serial retry, whereas 8 trajopt seeds land the SAME pairing in 0.30 s at this cap of 30.
# Attempts retry a serial search, seeds widen a parallel one -- prefer the seeds. Leave this at 30.
_BIMANUAL_PLAN0_MIN_ATTEMPTS = 30

# The single-arm plan-0 fallback (one cube, after the bimanual pairing above fails or only one cube is
# in view) previously stayed at the raw planner default (10 attempts) -- unboosted, unlike the bimanual
# case above. Same mechanism applies: a near-reachability-margin solve is a randomized-seed search, so a
# fixed 10-attempt cap can report a spuriously infeasible verdict that more restarts would find (confirmed
# 2026-07-27: `g1 left_right_close seed43`, solo/idle-GPU reruns of the IDENTICAL single-arm fallback solve
# flip FAIL/FAIL/PASS/FAIL/FAIL with zero code/seed/load difference -- see MEMORY.md "bucket 1"). Floors it
# at the same vetted magnitude as the bimanual case (more search, not a threshold/tolerance widening).
_SINGLE_PLAN0_MIN_ATTEMPTS = 30


# Marks the synthetic CUDA-graph-warmup targets so diagnostics can tell them from a real visit's cubes.
_WARM_TARGET_PREFIX = "warm_"

_VISIBLE_REACHABLE_FIELDS: dict[str, object | None] = {}


def visible_reachable_field(robot: str):
    """Return cached offline margin field, or ``None`` when no sidecar exists.

    Runtime cuRobo remains feasibility authority.  This light query object is shared by goal-assignment
    ordering and failure-triggered stance recovery so both use the same robot-specific sidecar and the
    first load happens once per process.
    """
    try:
        field = _VISIBLE_REACHABLE_FIELDS.get(robot)
        if robot not in _VISIBLE_REACHABLE_FIELDS:
            from mj_envs.asset_zoo.reachability_study.visible_reachable_curator import VisibleReachableField
            field = VisibleReachableField.from_robot(robot)
            _VISIBLE_REACHABLE_FIELDS[robot] = field
        return field
    except FileNotFoundError:
        _VISIBLE_REACHABLE_FIELDS[robot] = None
        return None


def _robust_assignment_order(robot_cfg, assignments):
    """Rank legacy arm-target assignments by their weakest per-target drift margin.

    ``assign_cubes_to_sides`` remains the candidate generator and deterministic fallback.  The offline
    visible-reachable field only changes TRY ORDER: for every already-valid candidate, score the actual
    base-frame goalset positions emitted for its assigned arm and prefer the assignment whose least robust
    target has the larger local ``D_reach`` neighborhood.  It neither filters targets nor replaces
    cuRobo's live IK+collision solve, so a stale/missing sidecar cannot turn a reachable target into a
    false negative.
    """
    if os.environ.get("VISIBLE_REACHABLE_ORDER", "1") == "0":
        return assignments                         # explicit legacy baseline for A/B evaluation
    field = visible_reachable_field(robot_cfg.name)
    if field is None:
        return assignments

    scored = []
    obj_quat = torch.as_tensor(world_axes_in_base_quat(robot_cfg.home_base_quat), dtype=torch.float32)
    grasp_pos_obj, grasp_quat_obj = robot_cfg.cube_grasp_poses_obj(device="cpu")
    for legacy_index, assigned in enumerate(assignments):
        robustness = []
        for side, target in assigned.items():
            candidate_pos, _ = robot_cfg.grasp_poses_to_base(
                torch.as_tensor(target, dtype=torch.float32), obj_quat, grasp_pos_obj, grasp_quat_obj)
            # cuRobo receives this full goalset, so an assignment gets credit for its most drift-tolerant
            # emitted target. Current face/tilt alternatives share one grasp position, so unique avoids
            # re-querying the same 5 cm neighborhood per orientation. The live solver still decides
            # collision/IK feasibility and the final pose.
            robustness.append(max(
                field.robust_score(pos, side)
                for pos in np.unique(candidate_pos.numpy(), axis=0)
            ))
        # Maximize worst target first: a bimanual candidate is only as dynamic-tolerant as its weaker arm.
        # Keep legacy order for an exact score tie, including all-zero out-of-grid/blind candidates.
        scored.append((min(robustness), sum(robustness), -legacy_index, assigned))
    scored.sort(reverse=True, key=lambda item: item[:3])
    return [assigned for *_score, assigned in scored]


def _first_feasible_assignment(robot_cfg, kin, planner, home_state, targets, start_state=None, max_attempts=None,
                               standoff: float = 0.0):
    """Reachability ladder over the OBSERVED cube targets. Try the geometric-prior assignment(s)
    bimanual-first (both L/R pairings for 2 cubes, cheaper first); on both-pairing failure fall back to
    the BEST feasible SINGLE cube. Returns ``(assigned {side: target}, plan_pose result)`` or ``None``
    when nothing is reachable. ``plan_assigned`` is non-asserting, so feasibility = a successful solve.

    ``start_state`` = the trajopt/IK warm seed threaded to ``plan_assigned`` (the CURRENT arm state under
    a base-live replan; ``None`` = plan from home). Assignment itself stays keyed on ``home_state``.
    ``max_attempts`` = per-solve attempt cap (``None`` = planner default; a replan passes ``1``). The
    bimanual (2-cube) try specifically floors this at ``_BIMANUAL_PLAN0_MIN_ATTEMPTS`` on plan-0 -- see
    that constant."""
    bimanual_attempts = max_attempts
    if max_attempts is None and len(targets) == 2:
        bimanual_attempts = max(planner._plan_max_attempts, _BIMANUAL_PLAN0_MIN_ATTEMPTS)
    elif max_attempts is None and len(targets) == 1:
        # A single visible cube resolves to a single-arm assignment here too (``assign_cubes_to_sides``
        # with one target) -- same near-margin randomized-seed sensitivity as the len==2 fallback below.
        bimanual_attempts = max(planner._plan_max_attempts, _SINGLE_PLAN0_MIN_ATTEMPTS)
    assignments = _robust_assignment_order(
        robot_cfg, assign_cubes_to_sides(kin, home_state, robot_cfg, targets))
    for i, assigned in enumerate(assignments):
        # Roadmap retry on the FIRST plan-0 assignment only (``max_attempts is None`` == plan-0, not a
        # replan). Placed here rather than after the back-off ladder because the ladder IS the cost being
        # replaced: cm4 measured a failing trip at 28-30 solves / 12.9-14.3 s, all of it pressing on a wall
        # trajopt cannot get around. One query here either solves it or reports the cube genuinely walled in,
        # and both answers let the caller skip the five remaining radial rungs.
        result = plan_assigned(robot_cfg, kin, planner, assigned, home_state,
                               start_state=start_state, max_attempts=bimanual_attempts, standoff=standoff,
                               graph_fallback=(max_attempts is None and i == 0))
        if result is not None and result.success.any():
            return assigned, result
    if len(targets) < 2:
        return None
    # Silent bimanual -> sequential-single degradation was a real debugging blind spot (2026-08-02): the
    # visit still SUCCEEDS, so nothing in the verdict distinguishes it from a scene that never had a pairing.
    # One line per demotion, on a path that by construction runs at most once per visit. Skipped for the
    # graph-warmup solve, whose nominal targets are expected to fail (see ``_warm_scene_and_targets``).
    if not all(name.startswith(_WARM_TARGET_PREFIX) for name in targets):
        print(f"[reach] bimanual infeasible for {sorted(targets)} at {bimanual_attempts} attempts"
              f" -- falling back to sequential single-arm", flush=True)
    single_attempts = max_attempts
    if max_attempts is None:
        single_attempts = max(planner._plan_max_attempts, _SINGLE_PLAN0_MIN_ATTEMPTS)
    for name, target in targets.items():
        assignments = _robust_assignment_order(
            robot_cfg, assign_cubes_to_sides(kin, home_state, robot_cfg, {name: target}))
        for assigned in assignments:
            result = plan_assigned(robot_cfg, kin, planner, assigned, home_state,
                                   start_state=start_state, max_attempts=single_attempts, standoff=standoff)
            if result is not None and result.success.any():
                return assigned, result
    return None


def plan_reach_route(session: CuroboPlannerSession, scenario, mj_model, base_pose,
                     cube_poses=None, seed_q=None, max_attempts=None,
                     fixed_assignment=None, goal: str = "reach",
                     height_pad: float = 0.02) -> ReachRoute | None:
    """Base-LIVE, reachability-driven reach producer shared by phase-3 (kinematic) and phase-4 (dynamic).

    ONE ego-centric re-solve driven by what the robot OBSERVES: the cube is world-fixed but the base
    floats, so a plan is valid only in the base frame it was solved in. Each call re-expresses scene +
    per-cube grasp targets in the live ``base_pose`` (``dataclasses.replace(cfg, home_base_pos/quat)``
    -> ``build_reach_scene``), hot-swaps the collision world, then lets the reachability ladder DERIVE the
    arm count: both observed cubes reachable -> bimanual (2 active); one reachable -> single; none -> hold.
    Only the SCENE moves per call; the robot kinematics/cspace are fixed, so the warmed session's planner
    is reused (no per-call CUDA-graph rebuild).

    Lives in ``scene.py`` (not ``planner.py``) because it needs BOTH ``build_reach_scene`` (here) and
    ``plan_assigned`` (planner.py); the layering is one-way ``planner.py <- scene.py``.

    Args:
        session: warmed ``CuroboPlannerSession`` (robot descriptor, kinematics, pose planner).
        scenario: height-correct ``Scenario`` (from ``robot_scene``); its ``pick`` cubes source targets.
        mj_model: compiled model carrying the ``pp_mark_*`` cube geoms (cube-pose seam) + static obstacles.
        base_pose: ``(pos[3], quat_wxyz[4])`` the base is estimated at THIS cycle.
        cube_poses: OBSERVED world cube dict (perception seam) -- ALL cubes (GT) or the camera-VISIBLE
            subset; ``None`` = GROUND TRUTH from ``mj_model`` (reach every cube).
        seed_q: OPTIONAL cuRobo-cspace warm seed ``{joint_name: q}`` (a subset of ``planner.joint_names``)
            = the CURRENT arm config for a base-live replan, so the re-solve starts the trajectory where
            the arm IS and stays on the same IK branch (no elbow flip). ``None`` = plan from planning home.
        max_attempts: OPTIONAL per-solve attempt cap forwarded to the solve (``None`` = planner default,
            robust, for the cold plan-0; a warm-seeded replan passes a small cap since the correction is
            easy and the retry loop dominates the wall time).
        fixed_assignment: OPTIONAL ``{side: cube_name}`` from a prior route. Given (a REPLAN) it SKIPS the
            reachability ladder AND the single-arm fallback: it reuses the exact side<->cube assignment and
            does ONE ``plan_assigned``. So (a) the active-arm count cannot change mid-reach (the ladder's
            fallback is what silently drops an arm), and (b) the per-tick cost is one solve, not a ladder.
            A cube that is no longer observed, or a failed solve, returns ``None`` -> the caller HOLDS the
            last route (no degrade). ``None`` = run the full ladder (plan-0).
        goal: ``"reach"`` (default) = plan to the cube grasp goalset. ``"home"`` = the RETRACT: the SAME
            planner invoked a SECOND time, planning to the HOME tool pose instead of the grasp (requires
            ``fixed_assignment``, reusing the reach's). ``cube_poses`` still supplies the collision scene so
            the return path is genuinely collision-aware; only the goal pose differs. See ``solve_reach_route``.

    Returns a ``ReachRoute`` over the active frames, or ``None`` when nothing is reachable (caller holds).
    """
    base_pos, base_quat = base_pose
    live_cfg = dataclasses.replace(
        session.robot_cfg, home_base_pos=tuple(base_pos), home_base_quat=tuple(base_quat))
    scene_dict, targets = build_reach_scene(scenario, live_cfg, mj_model, cube_poses=cube_poses, height_pad=height_pad)
    if not targets:
        return None
    return solve_reach_route(session, scene_dict, targets, base_pos, base_quat,
                             seed_q=seed_q, max_attempts=max_attempts, fixed_assignment=fixed_assignment,
                             goal=goal)


def ik_feasible_route(session: CuroboPlannerSession, scenario, mj_model, base_pose,
                      cube_poses=None, fixed_assignment=None, height_pad: float = 0.02) -> bool:
    """Cheap COLLISION-BLIND pre-filter for ``plan_reach_route``: IK-only (no trajopt), True iff SOME
    assignment the reachability ladder would try is IK-reachable. Sibling of ``plan_reach_route`` (same
    ``build_reach_scene``/``update_world`` preamble); tries the same ``assign_cubes_to_sides`` order (or
    the single ``fixed_assignment`` under a REPLAN, matching ``solve_reach_route``'s fast path), calling
    ``ik_feasible_assigned`` per candidate and returning True on the first pass. A False here means the
    FULL ladder (every assignment, IK+trajopt) would also fail -- IK is collision-blind, so it can only
    admit a superset of what trajopt admits -- so ``_handle_extend`` can skip straight to its back-off
    retry instead of paying for the full plan. Never used to CHOOSE an assignment (that stays
    ``plan_assigned``'s job); this only answers reachable-at-all."""
    base_pos, base_quat = base_pose
    live_cfg = dataclasses.replace(
        session.robot_cfg, home_base_pos=tuple(base_pos), home_base_quat=tuple(base_quat))
    scene_dict, targets = build_reach_scene(scenario, live_cfg, mj_model, cube_poses=cube_poses, height_pad=height_pad)
    return ik_feasible_scene(session, scene_dict, targets, base_pos, base_quat, fixed_assignment)


def ik_feasible_scene(session: CuroboPlannerSession, scene_dict, targets, base_pos, base_quat,
                      fixed_assignment=None) -> bool:
    """Legacy collision-blind IK predicate for an already-built live scene.

    Worker callers already receive base-frame ``scene_dict``/``targets`` from the parent. Keeping this
    scene-level form avoids constructing a second cuRobo session merely to compare a field-guided recovery
    stance with the same fast IK predicate used by the synchronous path.
    """
    if not targets:
        return False
    live_cfg = dataclasses.replace(
        session.robot_cfg, home_base_pos=tuple(base_pos), home_base_quat=tuple(base_quat))
    planner = session.update_world(scene_dict)
    kin = session.kin
    home_state = planner.default_joint_state.clone().unsqueeze(0)
    if fixed_assignment is not None:
        if not set(fixed_assignment.values()) <= set(targets):
            return False                    # an assigned cube is no longer observed -> hold
        assigned = {side: targets[name] for side, name in fixed_assignment.items()}
        return ik_feasible_assigned(live_cfg, kin, planner, assigned, home_state)
    for assigned in assign_cubes_to_sides(kin, home_state, live_cfg, targets):
        if ik_feasible_assigned(live_cfg, kin, planner, assigned, home_state):
            return True
    if len(targets) < 2:
        return False
    for name, target in targets.items():
        for assigned in assign_cubes_to_sides(kin, home_state, live_cfg, {name: target}):
            if ik_feasible_assigned(live_cfg, kin, planner, assigned, home_state):
                return True
    return False



def _plan_linear_descent(live_cfg, kin, planner, assigned, home_state, start_state, max_attempts):
    """``plan_assigned`` to the real grasp goalset under a Cartesian straight-line constraint on each tool
    frame's +x (approach) axis -- stage B of the two-stage reach (``solve_reach_route(goal="descend")``).

    Two deviations from an ordinary re-solve, both copied from cuRobo's own ``plan_grasp`` approach->grasp
    visit (``motion_planner.plan_grasp``), both necessary:

    * ``ToolPoseCriteria.linear_motion`` on the tool +x axis. Joint-space trajopt over a 5 cm segment does
      NOT tend straight: measured unconstrained it loops 0.35-0.50 m around the cube (straightness 0.10-0.14),
      which would knock the cube harder than the single-shot reach the split is meant to improve.
    * ``disable_link_collision`` on the grasp-contact links (the gripper racks). The grasp goal puts those
      links inside the cube obstacle by construction, so with their spheres live the segment is
      start-or-end-infeasible. Only the contact links are disabled; every other link stays collision-checked,
      which is what keeps this a genuine planned descent rather than the open-loop tracker rebind it replaces.

    Both are restored in a ``finally`` -- the planner is a long-lived warmed session shared with every other
    solve, so leaking either would silently corrupt all subsequent reaches.
    """
    frames = list(live_cfg.tool_frames)
    contact_links = planner.kinematics.config.kinematics_config.grasp_contact_link_names or []
    linear = ToolPoseCriteria.linear_motion(axis="x", non_terminal_scale=1.0,
                                            project_distance_to_goal=True)
    planner.update_tool_pose_criteria({frame: linear for frame in frames})
    planner.disable_link_collision(contact_links)
    try:
        return plan_assigned(live_cfg, kin, planner, assigned, home_state,
                             start_state=start_state, max_attempts=max_attempts, standoff=0.0)
    finally:
        planner.enable_link_collision(contact_links)
        planner.update_tool_pose_criteria({frame: ToolPoseCriteria() for frame in frames})


# Rear-prime acceptance tolerances. POSE grades the primed arm at its tool, not in cspace: the chain is
# redundant, so a 6-DoF goal admits several joint solutions and any that puts the tool at the flipped pose
# has the arm reaching rearward, which is the property EXTEND needs. (A cspace check tried first rejected
# good routes at 3.8 rad for landing on a different, equally rear branch.) CARRY bounds the OTHER arm over
# every waypoint: only the primed arm's columns are commanded, so planned motion of the carrying arm is
# motion the collision check assumed and the robot will not perform. ``plan_pose`` pins that arm by pose
# alone and will not surrender the leftover null space, so this bounds the error rather than forbidding it.
_PRIME_POSE_TOL_M = 0.02
_PRIME_CARRY_TOL_RAD = 0.25


def _tool_pose_goalset(fk, tool_frames, g: int) -> GoalToolPose:
    """Goal for "plan to this joint CONFIGURATION": every tool frame's FK pose, repeated across the
    goalset width the planner was CUDA-graph-captured for (``max_goalset`` = ``grasp_goalset_size``).

    The repeat is not cosmetic. A width-1 goal into a width-g captured graph leaves slots 1..g-1 holding
    whatever the previous solve wrote, so IK mins over stale grasp poses and returns zero successes --
    ``plan_pose`` then yields None. Same trick ``_assigned_goal_pose`` uses for an idle side.
    """
    pos, quat = [], []
    for frame in tool_frames:
        hp = fk.tool_poses.get_link_pose(frame)
        pos.append(hp.position[0].view(1, 3).expand(g, 3))
        quat.append(hp.quaternion[0].view(1, 4).expand(g, 4))
    n = len(tool_frames)
    return GoalToolPose(tool_frames=list(tool_frames),
                        position=torch.stack(pos, dim=0).view(1, 1, n, g, 3),
                        quaternion=torch.stack(quat, dim=0).view(1, 1, n, g, 4))


def solve_reach_route(session: CuroboPlannerSession, scene_dict, targets, base_pos, base_quat,
                      seed_q=None, max_attempts=None, fixed_assignment=None,
                      goal: str = "reach") -> ReachRoute | None:
    """GPU solve core of ``plan_reach_route``, split out so the async worker (``SpawnedReachWorker``) can run
    it in the ONE cuRobo process: ``build_reach_scene`` (CPU: needs ``mj_model`` geometry) runs in the caller
    and hands this the already-built ``scene_dict`` + base-frame ``targets``, so the worker needs no MuJoCo
    model. Args match ``plan_reach_route`` from ``update_world`` onward; ``targets`` = ``{cube_name:
    base-frame target}``; ``base_pos``/``base_quat`` reconstruct the live descriptor for grasp geometry.

    ``goal`` = the RETRACT symmetry seam. ``"reach"`` (default) = plan to the cube grasp goalset.
    ``"pregrasp"`` = the SAME solve as ``"reach"`` with the goalset retreated ``_PREGRASP_STANDOFF_M`` along
    each candidate's own approach axis -- stage A of the two-stage reach.
    ``"descend"`` = stage B: the real grasp goalset, seeded at the realized pre-grasp config, under a
    ``ToolPoseCriteria.linear_motion`` constraint on the tool +x (approach) axis and with the grasp-contact
    links' spheres disabled -- the same pairing cuRobo's own ``plan_grasp`` uses for its approach->grasp visit,
    because the goal sits INSIDE the cube obstacle. Both stages are genuine collision-checked cuRobo routes
    (the point of the split: the removed ``_MPC_PREGRASP_LIFT_M`` descended open-loop through the tracker).
    The constraint is NOT cosmetic -- measured on ``left_right_close`` / ``bimanual_mixed_front_back_close``,
    an UNCONSTRAINED stage B travels 0.35-0.50 m of tool path to cover 0.05 m of net displacement
    (straightness 0.10-0.14), looping around the cube and knocking it harder than the single-shot reach it
    replaces; constrained it travels 0.051-0.071 m (straightness 0.70-0.98) at 0.0 deg off stage A's own
    tool +x. That 0.0 deg also settles the candidate-consistency question: stage B's goalset-min lands on
    the candidate stage A backed off from, so the descent is a translation, not a re-orientation swing.
    ``"home"`` = the SAME planner invoked a SECOND time: plan_pose to the HOME tool pose of every frame (FK of
    ``planning_home`` == ``default_joint_state``) instead of the grasp -- a GENUINE collision-aware path back
    (the ``scene_dict`` still carries the cubes as hard obstacles), NOT a reversed replay. It is always a
    REPLAN (retract reuses the reach's ``fixed_assignment``), seeded at the CURRENT grasp config (``seed_q``)
    so the return starts where the arm IS and stays on its IK branch; only the goal pose differs from reach,
    so the interpolation + ``ReachRoute`` wrap tail are SHARED. ``reaches[frame]`` records the HOME tool pose
    (symmetric with the grasp triad: the overlay's goal-triad then marks HOME).
    ``"prime:L"``/``"prime:R"`` = the REAR PRIME: plan ONE named side's arm from wherever it is into the
    rear-facing configuration while the other side holds its planning-home pose, so a later reach behind
    the robot plans a short local motion instead of failing from the forward basin. Shares the ``"home"``
    branch's whole shape (FK a joint configuration, take each frame's tool pose, expand to the goalset
    width, one ``plan_pose``) because ``plan_cspace`` cannot reuse the pose-warmed CUDA graph -- the two
    capture different shapes (see ``ik_curobo.py``). Target configuration = planning home with the primed
    side's joints NEGATED: cuRobo cspace == MuJoCo qpos for the arms and the planner default IS the
    forward-facing reach-ready pose, so a plain sign flip is the one-variable move to the rear basin
    (FK-verified: that side's tool goes base-frame x +0.106 -> -0.073 with z preserved, and the other side
    is bit-identical, the chains being independent). Unlike ``"home"`` this does NOT disable the
    grasp-contact links: a priming visit's belief carries only cubes it has yet to grasp, so the CARRIED cube
    is absent from ``scene_dict`` and the start state is not start-infeasible."""
    prime_side = goal[len("prime:"):] if goal.startswith("prime:") else None
    assert goal in ("reach", "pregrasp", "descend", "home") or prime_side in ("L", "R"), \
        f"goal must be 'reach', 'pregrasp', 'descend', 'home' or 'prime:L'/'prime:R'; got {goal!r}"
    assert goal != "descend" or fixed_assignment is not None, \
        "goal='descend' is stage B of a committed reach; it requires the stage-A fixed_assignment"
    standoff = _PREGRASP_STANDOFF_M if goal == "pregrasp" else 0.0
    live_cfg = dataclasses.replace(
        session.robot_cfg, home_base_pos=tuple(base_pos), home_base_quat=tuple(base_quat))
    planner = session.update_world(scene_dict)
    kin = session.kin
    home_state = planner.default_joint_state.clone().unsqueeze(0)
    start_state = home_state
    if seed_q is not None:
        # Build the warm-seed start state the SAME way the replan worker does (planner.py): a fresh
        # JointState.from_position tagged with planner.joint_names, so cuRobo reorders by NAME internally.
        # (Writing into default_joint_state.position by index fails: that buffer is in the solver-internal
        # order, not planner.joint_names, so an index-write scrambles the seed -> the arm flails.)
        q = torch.tensor([[seed_q[n] for n in planner.joint_names]],
                         device=home_state.position.device, dtype=home_state.position.dtype)
        start_state = JointState.from_position(q, joint_names=list(planner.joint_names))
    home_fk = prime_fk = prime_q = None
    if prime_side is not None:
        # Flip vector built as a NAME-keyed dict and routed through ``JointState`` by name for the same
        # reason the ``seed_q`` block above does: ``default_joint_state.position`` sits in the
        # solver-internal order, so an index-write into that buffer scrambles the configuration.
        # ``planning_home_joint_pos`` IS the dict the planner default was built from (``planner.py``), so
        # negating the primed side's arm joints in it -- and nothing else, gripper racks included -- is
        # exactly "planning home, that arm reversed". ONE FK covers both goal frames: the untouched side of
        # ``prime_q`` already IS home, which is the pin that keeps a carrying arm still while the free one
        # swings through.
        home_q = live_cfg.planning_home_joint_pos
        missing = [n for n in planner.joint_names if n not in home_q]
        assert not missing, f"planning_home_joint_pos is missing planner joints {missing}"
        flip_names = set(live_cfg.arm_joints_by_side[prime_side])
        prime_q = {n: (-home_q[n] if n in flip_names else home_q[n]) for n in planner.joint_names}
        prime_fk = kin.compute_kinematics(JointState.from_position(
            torch.tensor([[prime_q[n] for n in planner.joint_names]],
                         device=home_state.position.device, dtype=home_state.position.dtype),
            joint_names=list(planner.joint_names)))
        goal_pose = _tool_pose_goalset(prime_fk, live_cfg.tool_frames, live_cfg.grasp_goalset_size)
        result = planner.plan_pose(
            goal_pose, start_state,
            max_attempts=planner._plan_max_attempts if max_attempts is None else max_attempts,
            enable_graph_attempt=planner._plan_enable_graph_attempt)
        if result is None or not result.success.any():
            # One line per prime is the whole health signal: distinguishes a rejected QUERY (start or goal
            # invalid -- start-in-collision is the usual cause) from a query cuRobo accepted but could not
            # solve. Cheap: at most one prime per target visit.
            print(f"[curobo] {goal} infeasible ({'no result' if result is None else 'no success'})",
                  flush=True)
            return None                     # infeasible prime -> caller keeps the un-primed forward arm
        assigned = {prime_side: None}       # drives the controlled-joint union below: the primed arm only
        assignment = {}                     # no cube: a prime is not a reach visit
    elif goal == "home":
        # RETRACT: every tool frame targets its HOME pose (FK of planning home). The active frames journey
        # home; idle frames were already home. One plan_pose, seeded at the grasp config. Always a REPLAN.
        assert fixed_assignment is not None, "goal='home' requires fixed_assignment (retract reuses the reach's)"
        home_fk = kin.compute_kinematics(home_state)
        goal_pose = _tool_pose_goalset(home_fk, live_cfg.tool_frames, live_cfg.grasp_goalset_size)
        # The retract belief still carries the GRASPED cube (``_reach_belief`` never retires a held cube), so
        # ``scene_dict`` holds it as a static obstacle at the pose it occupies INSIDE the closed jaws. The
        # start state is therefore start-infeasible with the contact links live: IK solves, trajopt fails for
        # every seed, and RETRACT silently falls back to the reversed corridor. Disable exactly the
        # grasp-contact links for this solve -- the same pairing ``_plan_linear_descent`` uses for the mirror
        # situation on the way in. Every other link stays checked, so the return is still a genuine
        # collision-aware path. Restored in ``finally``: the planner is a long-lived warmed session.
        contact_links = planner.kinematics.config.kinematics_config.grasp_contact_link_names or []
        planner.disable_link_collision(contact_links)
        try:
            result = planner.plan_pose(
                goal_pose, start_state,
                max_attempts=planner._plan_max_attempts if max_attempts is None else max_attempts,
                enable_graph_attempt=planner._plan_enable_graph_attempt)
        finally:
            planner.enable_link_collision(contact_links)
        if result is None or not result.success.any():
            if _LIMIT_PROBE:
                print(f"[limit] home solve FAILED (result={'None' if result is None else 'no success'})",
                      flush=True)
            return None                     # infeasible return -> caller falls back to the reversed replay
        if _LIMIT_PROBE:
            try:
                _tm = planner.trajopt_solver.transition_model
                _dt = float(planner.trajopt_solver.config.interpolation_dt)
                _ip = result.get_interpolated_plan()
                _ni = {n: k for k, n in enumerate(_ip.joint_names)}
                _ai = [_ni[n] for n in planner.joint_names]
                _q = _ip.position[0, 0][:, _ai].detach().cpu().numpy().astype(float)
                _d = _q
                for _n, _lim in (("vel", _tm.max_velocity), ("acc", _tm.max_acceleration),
                                 ("jerk", _tm.max_jerk)):
                    _d = np.diff(_d, axis=0) / _dt
                    _pk = np.abs(_d).max(axis=0)
                    _l = float(_lim.detach().cpu().numpy().reshape(-1).min())
                    print(f"[limit] home {_n}: peak={_pk.max():.3f} lim={_l:.3f} "
                          f"ratio={_pk.max() / _l:.3f} n={_q.shape[0]} dt={_dt:.4f}", flush=True)
            except Exception as _e:                        # probe must never break planning
                print(f"[limit] probe failed: {type(_e).__name__}: {_e}", flush=True)
        assigned = dict(fixed_assignment)   # {side: cube_name}; the sides drive the reaches loop below
        assignment = dict(fixed_assignment)
    elif fixed_assignment is not None:
        # REPLAN: reuse the prior side<->cube assignment; one solve, no ladder, no single-arm fallback.
        if not set(fixed_assignment.values()) <= set(targets):
            return None                     # an assigned cube is no longer observed -> hold
        assigned = {side: targets[name] for side, name in fixed_assignment.items()}
        if goal == "descend":
            result = _plan_linear_descent(live_cfg, kin, planner, assigned, home_state,
                                          start_state, max_attempts)
        else:
            # The back-off ladder re-enters HERE, not through ``_first_feasible_assignment``: once a visit has
            # committed a side<->cube pairing it is carried as ``fixed_assignment``, so every one of the 6
            # retry rungs lands on this single solve. This is the branch that burns 28-30 solves on a
            # failing trip, and therefore the one the roadmap retry has to cover.
            result = plan_assigned(live_cfg, kin, planner, assigned, home_state,
                                   start_state=start_state, max_attempts=max_attempts, standoff=standoff,
                                   graph_fallback=True)
        if result is None or not result.success.any():
            return None                     # failed re-solve -> hold the last route (count preserved)
        assignment = dict(fixed_assignment)
    else:
        chosen = _first_feasible_assignment(live_cfg, kin, planner, home_state, targets,
                                            start_state=start_state, max_attempts=max_attempts,
                                            standoff=standoff)
        if chosen is None:
            return None
        assigned, result = chosen
        name_by_target = {id(t): name for name, t in targets.items()}   # ta/tb pass by ref through assign
        assignment = {side: name_by_target[id(t)] for side, t in assigned.items()}
    interpolated = result.get_interpolated_plan()
    name_to_i = {n: k for k, n in enumerate(interpolated.joint_names)}
    active_idx = [name_to_i[n] for n in planner.joint_names]
    route = np.ascontiguousarray(interpolated.position[0, 0][:, active_idx].detach().cpu().numpy(), dtype=np.float32)
    final_q = final_active_q(interpolated, planner.joint_names)
    if prime_side is not None:
        # See ``_PRIME_POSE_TOL_M`` for why the primed arm is graded at its TOOL and the carrying arm in
        # cspace over every waypoint rather than only at the end.
        primed_frame = {"L": live_cfg.tool_frames[0], "R": live_cfg.tool_frames[1]}[prime_side]
        want_pos = np.asarray(prime_fk.tool_poses.get_link_pose(primed_frame).position[0]
                              .detach().cpu().numpy(), dtype=float)
        err_pose = float(np.linalg.norm(
            want_pos - site_base(kin, planner.joint_names, final_q, primed_frame)))
        col = {n: i for i, n in enumerate(planner.joint_names)}
        other_side = "R" if prime_side == "L" else "L"
        hold_cols = [col[n] for n in live_cfg.arm_joints_by_side[other_side]]
        want_hold = np.array([prime_q[n] for n in planner.joint_names], dtype=float)[hold_cols]
        err_carry = float(np.abs(route[:, hold_cols] - want_hold[None, :]).max())
        ok = err_pose <= _PRIME_POSE_TOL_M and err_carry <= _PRIME_CARRY_TOL_RAD
        print(f"[curobo] {goal} {'ok' if ok else 'REJECTED'} err_pose={err_pose:.4f} "
              f"err_carry={err_carry:.3f} n={route.shape[0]}", flush=True)
        if not ok:
            return None                     # tool short of the flip, or carrying arm disturbed -> no prime
    side_to_frame = {"L": live_cfg.tool_frames[0], "R": live_cfg.tool_frames[1]}
    obj_quat = world_axes_in_base_quat(base_quat)   # world-level grasp at the live (tilted) base
    reaches, controlled = {}, []
    for side in assigned:
        frame = side_to_frame[side]
        controlled += list(live_cfg.arm_joints_by_side[side])
        if prime_side is not None:
            # Goal-triad marks the FLIPPED tool pose, mirroring what the home branch does for HOME.
            prime_pos = np.asarray(
                prime_fk.tool_poses.get_link_pose(frame).position[0].detach().cpu().numpy(), dtype=float)
            reached = site_base(kin, planner.joint_names, final_q, frame)
            reaches[frame] = (prime_pos, prime_pos[None, :], float(np.linalg.norm(prime_pos - reached)))
            continue
        if goal == "home":
            # Goal-triad marks HOME (symmetric with the grasp triad); err vs the home tool pose.
            home_pos = home_fk.tool_poses.get_link_pose(frame).position[0].detach().cpu().numpy()
            home_pos = np.asarray(home_pos, dtype=float)
            reached = site_base(kin, planner.joint_names, final_q, frame)
            err = float(np.linalg.norm(home_pos - reached))
            reaches[frame] = (home_pos, home_pos[None, :], err)
            continue
        target = assigned[side]
        # Grade stage A against the PRE-GRASP goalset it actually solved, not the grasp: measuring the
        # retreat pose against the grasp candidates would report a spurious ~standoff error and put the
        # overlay's goal triad on the wrong point.
        err = reach_error(live_cfg, kin, planner, final_q, target, frame, obj_quat=obj_quat, standoff=standoff)
        cand = grasp_candidates_base(live_cfg, planner, target, obj_quat=obj_quat,
                                     standoff=standoff)[0].detach().cpu().numpy()
        reaches[frame] = (np.asarray(target, dtype=float), np.asarray(cand, dtype=float), float(err))
    return ReachRoute(
        tuple(planner.joint_names), route, float(planner.trajopt_solver.config.interpolation_dt),
        reaches, tuple(controlled),
        np.asarray(base_pos, dtype=float), np.asarray(base_quat, dtype=float), assignment)


def reach_scene_and_targets(robot_cfg: RobotDescriptor, scenario, mj_model, base_pose, cube_poses=None, height_pad: float = 0.02):
    """Base-frame collision scene + per-cube grasp targets at a live base pose -- the CPU-only inputs the
    async cuRobo worker (``SpawnedReachWorker``) needs, WITHOUT any GPU/cuRobo call. Takes the descriptor
    (CPU), NOT a warmed session, so the ASYNC policy holds no cuRobo: it computes scene+targets on the CPU
    each tick and ships them to the ONE worker process. Same two lines as ``plan_reach_route`` head
    (``dataclasses.replace`` to the live base, then ``build_reach_scene``), so ``targets[name]`` is the
    identical base-frame target ``solve_reach_route`` expects in-worker."""
    base_pos, base_quat = base_pose
    live_cfg = dataclasses.replace(
        robot_cfg, home_base_pos=tuple(base_pos), home_base_quat=tuple(base_quat))
    return build_reach_scene(scenario, live_cfg, mj_model, cube_poses=cube_poses, height_pad=height_pad)




def scene_and_targets(scenario, robot_cfg: RobotDescriptor, mj_model, single: bool):
    if single:
        scene_dict, target_spec = build_scene_and_targets(scenario, robot_cfg, mj_model)
    else:
        scene_dict, target_spec = build_bimanual_scene_and_targets(scenario, robot_cfg, mj_model)
    return scene_dict, target_spec


# ------------------------------------------------------------------------------------------------
# Per-robot placement bridge (descriptor.shoulder_z -> scenario table height + z-floor override)
# ------------------------------------------------------------------------------------------------
def robot_scene(robot: str, scenario_name: str):
    """Build the (robot descriptor, object scene, scenario, table_z) tuple with a PER-ROBOT table height.

    ``table_z = table_z_for(descriptor.shoulder_z)`` puts every bench top a fixed distance below the
    robot's shoulder (uniform across scenes; cube rests flush at oz = table_z + CUBE_HALF). A
    taller-shouldered robot (g1) gets a proportionally higher bench and reaches at the same
    shoulder-relative height. Returns table_z for logging. For g1, the gripper z-floor
    (``planner_kwargs['hand_z_floor']``, base frame) tracks the new table top, replacing the static
    default so the floor stays '= workbench top' after the bench moves.
    """
    cfg = CFG_BY_ROBOT[robot]
    table_z = table_z_for(cfg.shoulder_z)
    scenario = make_scenarios(table_z=table_z, shoulder_z=cfg.shoulder_z)[scenario_name]
    if cfg.planner_kwargs.get("hand_z_floor") is not None:
        pk = dict(cfg.planner_kwargs)
        pk["hand_z_floor"] = table_z - cfg.home_base_pos[2]  # world table top -> base frame
        cfg = dataclasses.replace(cfg, planner_kwargs=pk)
    return cfg, ObjectSceneCfg(scenario=scenario), scenario, table_z


# ------------------------------------------------------------------------------------------------
# Async reach planner: the ONE cuRobo runtime, isolated in a spawned process
# ------------------------------------------------------------------------------------------------
# The control/env process holds NO cuRobo. It computes scene+targets on the CPU (build_reach_scene needs
# only the descriptor + mj_model geometry) and ships them here; this process owns the sole cuRobo session
# and returns a solved ReachRoute. Plan-0 (fixed_assignment=None) runs the full reachability ladder to
# DERIVE the arm count/assignment; every later request reuses that assignment (fixed) + a warm seed. So the
# whole planner -- including plan-0 -- is one runtime, warmed once, with no duplicate in the parent.
@dataclass(frozen=True)
class ReachRequest:
    """CPU-only reach request crossing parent -> the cuRobo process. ``scene_dict``/``targets`` are the
    parent's ``build_reach_scene`` output at the live base; ``fixed_assignment=None`` = plan-0 (run the
    ladder); ``seed_q=None`` = plan from home. ``request_generation`` lets ``poll`` drop stale routes."""
    request_generation: int
    scene_dict: dict
    targets: dict                     # {cube_name: base-frame target (np.ndarray)}
    base_pos: np.ndarray
    base_quat: np.ndarray
    seed_q: dict | None               # {joint: cuRobo-cspace q} warm seed, or None (home)
    max_attempts: int | None
    fixed_assignment: dict | None     # {side: cube_name} reuse, or None (plan-0 ladder)
    goal: str = "reach"              # "reach" or "home"; home is the post-grasp return plan


@dataclass(frozen=True)
class ReachResult:
    """The solved ``ReachRoute`` (or ``None`` = infeasible/hold) returned by the cuRobo process."""
    request_generation: int
    route: ReachRoute | None
    latency_s: float
    error: str | None = None


def _reach_worker(robot_name: str, scenario_name: str, requests, results, stop_event) -> None:
    """Own the ONE cuRobo session in a spawned process and solve reach requests serially. Rebuilds the
    descriptor via ``robot_scene`` (same per-robot table/z-floor adjustment the parent used) so the solve
    matches; ``spawn`` is mandatory (cuRobo CUDA graphs cannot survive a fork). Coalesces: before each solve
    it drains the queue to the newest request, so a slow solve never leaves a backlog of stale routes."""
    try:
        cfg, _, _, _ = robot_scene(robot_name, scenario_name)
        session = make_planner_session(cfg)
        while not stop_event.is_set():
            try:
                request = requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if request is None:
                break
            while True:                                   # drain to newest (coalesce stale replans)
                try:
                    candidate = requests.get_nowait()
                except queue.Empty:
                    break
                if candidate is None:
                    stop_event.set()
                    break
                request = candidate
            if stop_event.is_set():
                break
            started = time.monotonic()
            try:
                route = solve_reach_route(
                    session, request.scene_dict, request.targets, request.base_pos, request.base_quat,
                    seed_q=request.seed_q, max_attempts=request.max_attempts,
                    fixed_assignment=request.fixed_assignment, goal=request.goal)
                results.put(ReachResult(request.request_generation, route, time.monotonic() - started))
            except Exception as error:
                results.put(ReachResult(request.request_generation, None, time.monotonic() - started,
                                        f"{type(error).__name__}: {error}"))
    except Exception:
        results.put(("worker_init_error", traceback.format_exc()))


class SpawnedReachWorker:
    """Persistent nonblocking cuRobo reach process with latest-request-wins coalescing -- the sole cuRobo
    runtime for the async policy. Parent sends CPU numpy/dict payloads and calls :meth:`poll`/:meth:`wait`.
    One solve may run while one newer request waits in ``_pending``; further requests overwrite it."""
    def __init__(self, robot_name: str, scenario_name: str):
        self._ctx = mp.get_context("spawn")
        self._requests = self._ctx.Queue(maxsize=1)
        self._results = self._ctx.Queue()
        self._stop_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_reach_worker, args=(robot_name, scenario_name, self._requests, self._results,
                                        self._stop_event),
            daemon=True, name="curobo-reach-worker")
        self._process.start()
        self._inflight: ReachRequest | None = None
        self._pending: ReachRequest | None = None
        self.coalesced_requests = 0
        self.closed = False
        atexit.register(self.close)          # backstop: reap the child even on an abnormal (unclosed) exit

    def submit(self, request: ReachRequest) -> bool:
        """Submit now when idle, else replace the single pending snapshot without blocking."""
        if self.closed:
            return False
        if self._inflight is not None:
            self._pending = request
            self.coalesced_requests += 1
            return False
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._pending = request
            self.coalesced_requests += 1
            return False
        self._inflight = request
        return True

    def poll(self) -> list[ReachResult]:
        """Drain completed routes and dispatch the newest pending request without waiting."""
        completed = []
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                raise RuntimeError(item[1])
            completed.append(item)
            if self._inflight is not None and item.request_generation == self._inflight.request_generation:
                self._inflight = None
        if self._inflight is None and self._pending is not None and not self.closed:
            pending, self._pending = self._pending, None
            self.submit(pending)
        return completed

    def wait(self, request_generation: int, timeout_s: float) -> ReachResult:
        """BLOCK until the result for ``request_generation`` arrives -- used only for plan-0 (the one-time
        warmup: the first solve triggers the in-child cuRobo/CUDA-graph build). Raises on timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            for result in self.poll():
                if result.request_generation == request_generation:
                    return result
            time.sleep(0.02)
        raise TimeoutError(f"cuRobo worker cold after {timeout_s}s (plan-0 never returned)")

    def close(self) -> None:
        """Stop the worker without waiting for a result; safe to call repeatedly."""
        if self.closed:
            return
        self.closed = True
        self._pending = None
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        # ``spawn`` registers queue semaphores with the parent resource tracker. Explicitly close them
        # after the child is reaped, otherwise repeated short evaluator runs warn about leaked semaphores.
        for ipc_queue in (self._requests, self._results):
            ipc_queue.close()
            ipc_queue.join_thread()


# ------------------------------------------------------------------------------------------------
# Object-anchored pose algebra: the seam shared by plan-0 (extract the constant grasp-in-object offset)
# and the runtime Jacobian drift-servo (recompose the grasp goal from the live observed object). No
# cuRobo, no base-in-world -- pure SE(3) on ``(pos, quat_wxyz)`` pairs in the base frame.
# ------------------------------------------------------------------------------------------------


def pose_compose(a, b):
    """SE(3) compose ``a ∘ b`` (both ``(pos[3], quat_wxyz[4])``): express pose ``b``, given in frame A, in
    A's parent frame. Here ``a = objects_in_base[name]`` (object in base_link) and ``b = grasp_in_object``,
    so the result is the grasp pose in base_link -- the goal fed to the MPC. ``q = q_a ⊗ q_b`` (A first),
    ``p = p_a + R(q_a)·p_b``."""
    p_a, q_a = a
    p_b, q_b = b
    p = np.zeros(3)
    mujoco.mju_rotVecQuat(p, np.asarray(p_b, dtype=float), np.asarray(q_a, dtype=float))
    p += np.asarray(p_a, dtype=float)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.asarray(q_a, dtype=float), np.asarray(q_b, dtype=float))
    return p, q


def pose_relative(ref, x):
    """Inverse of ``pose_compose``: express pose ``x`` relative to frame ``ref`` (both in the same parent
    frame). Used ONCE at rebind to extract the constant ``grasp_in_object`` from the plan's achieved grasp
    (``x``, in base_link) and the plan-time object pose (``ref``, in base_link):
    ``grasp_in_object = pose_relative(objects_in_base[target], grasp_in_base)``.
    ``q = conj(q_ref) ⊗ q_x``, ``p = R(conj(q_ref))·(p_x - p_ref)``."""
    p_ref, q_ref = ref
    p_x, q_x = x
    qinv = np.zeros(4)
    mujoco.mju_negQuat(qinv, np.asarray(q_ref, dtype=float))
    p = np.zeros(3)
    mujoco.mju_rotVecQuat(p, np.asarray(p_x, dtype=float) - np.asarray(p_ref, dtype=float), qinv)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qinv, np.asarray(q_x, dtype=float))
    return p, q


_WARM_CUBOID_COUNT = 16   # collision-cache upper bound the warm graph is captured with. Real reach scenes
                          # load FEWER cuboids (workbench slabs + visible cubes), so the pose graph is REUSED
                          # (cheap update_world), never recaptured. Over-sized on purpose: sizing the cache
                          # too small would force a recapture when a richer real scene arrives.


def _exit_when_parent_dies() -> None:
    """Bind worker lifetime to its harness parent before allocating CUDA state.

    A parent SIGSEGV bypasses ``SpawnedReachMpcWorker.close``. Without Linux's
    ``PR_SET_PDEATHSIG``, the spawned CUDA process becomes an init-owned orphan
    and retains its GPU context, making later captures hang or crash. The worker
    has no valid request channel after its parent dies, so terminate it instead.
    The second parent-PID check closes the fork-to-prctl race.
    """
    parent_pid = os.getppid()
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        err = ctypes.get_errno()
        raise OSError(err, "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _nominal_warm_scene_and_targets(cfg):
    """Perception-free stand-in world that CAPTURES the cuRobo pose CUDA-graph at worker boot -- BEFORE any
    real table/shelf/cube is known. This is the deployment reality: the planner process boots and warms on a
    GENERIC world, then perception streams the true obstacles via ``update_world`` (which reuses the graph).
    So the warmup must NOT depend on ``mj_model`` / the real scene.

    One base-frame 'table' slab under two reachable targets -- a feasible bimanual reach that exercises the
    full IK+trajopt+interpolate path a real plan-0 hits -- plus far-away filler boxes that ONLY size the
    collision cache (``_WARM_CUBOID_COUNT``) so a richer real scene fits without recapture. Geometry is
    nominal: only the graph SHAPE (robot cspace + goalset + cache size) must match the real solves, never the
    obstacle VALUES. A failed nominal solve still captures the graph (the first ``plan_pose`` builds it),
    so feasibility here is a bonus, not a requirement."""
    q_ident = [1.0, 0.0, 0.0, 0.0]
    cuboids = {"warm_table": {"dims": [1.0, 1.0, 0.1], "pose": [0.45, 0.0, -0.15] + q_ident}}
    for i in range(_WARM_CUBOID_COUNT - 1):
        cuboids[f"warm_filler_{i}"] = {"dims": [0.05, 0.05, 0.05], "pose": [10.0 + i, 10.0, 10.0] + q_ident}
    targets = {f"{_WARM_TARGET_PREFIX}a": np.array([0.45, 0.18, 0.02]),
               f"{_WARM_TARGET_PREFIX}b": np.array([0.45, -0.18, 0.02])}
    return {"cuboid": cuboids}, targets


_CAPTURE_DIR = os.environ.get("REACH_CAPTURE_FAILED_PLAN0")


def _capture_failed_plan0(robot_name: str, scenario_name: str, msg) -> None:
    """TEMPORARY probe: pickle an infeasible plan-0 request so it can be re-solved offline.

    The worker request tuple IS the complete problem statement (obstacles, base-frame targets, live base
    pose, seed config, goal kind) -- ``solve_reach_route`` reads nothing else -- so replaying it against a
    fresh ``make_planner_session(cfg)`` reproduces the failure exactly, with no MuJoCo, no walk, no gaze.
    That turns a ~60 s mission into a ~0.5 s solve and makes N-repeat A/B on ONE query possible, which a
    3-failure sweep population cannot support. ``generation`` is dropped: it is queue bookkeeping, not
    problem data.
    """
    import pickle
    out = pathlib.Path(_CAPTURE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"plan0_{robot_name}__{scenario_name}__{msg[8]}__{time.monotonic_ns()}.pkl"
    with open(path, "wb") as handle:
        pickle.dump({"robot_name": robot_name, "scenario_name": scenario_name,
                     "scene_dict": msg[2], "targets": msg[3], "base_pos": msg[4],
                     "base_quat": msg[5], "fixed_assignment": msg[6], "seed_q": msg[7],
                     "goal": msg[8]}, handle)
    print(f"[capture] infeasible plan-0 -> {path}", flush=True)


def _reach_mpc_worker(robot_name: str, scenario_name: str, requests, results, stop_event,
                      warmed_event) -> None:
    """Own the ONE cuRobo session in a spawned process for plan-0 ONLY (``solve_reach_route`` -- a
    collision-free reach trajectory + assignment). The per-tick reactive tracking is a MuJoCo Jacobian
    drift-servo in the CONTROL process (``JacobianReachTracker``), so no cuRobo runs per tick and this
    worker keeps ZERO GPU state between plan-0 solves. BLOCKING request/response. Messages:
    ``("plan0", generation, scene_dict, targets, base_pos, base_quat, fixed_assignment, seed_q, goal)`` ->
    ``("plan0", route_or_None, generation)``; ``("fast_ik", scene_dict, targets, base_pos, base_quat, fixed_assignment)``
    -> ``("fast_ik", bool)``. ``None``/``("stop",)`` stops. (``goal='home'`` + ``seed_q`` = the retract:
    the SAME request sent a second time, planning back to home from the current grasp config.)
    ``warmed_event`` is set once the one-time CUDA-graph capture below has been attempted (success or
    failure -- capture happens either way, see ``_nominal_warm_scene_and_targets``), or on init failure.
    Callers about to create a CONCURRENT GPU context (e.g. an OpenGL viewer) on this GPU must wait on it
    first: capture is a driver-global critical section with no cross-process safety contract, and
    unsynchronized concurrent GL driver activity during capture is undefined/crash-prone (observed as a
    SIGSEGV, exit -11)."""
    _exit_when_parent_dies()
    if os.environ.get("CUROBO_DETERMINISTIC") == "1":
        # POC gate (MEMORY.md "bucket 1"/Layer 2): forces bit-reproducible cuBLAS/cuSolver reduction order
        # and disables nondeterministic CUDA kernels, to test whether the documented near-margin
        # FAIL/PASS flip-flop (identical seed/host/idle-GPU/code) is CUDA kernel-scheduling nondeterminism.
        # Must be set before this fresh spawned process makes its first CUDA call (session creation below).
        # Opt-in only -- NOT the default until the wall-clock cost is measured against the repro.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if os.environ.get("CUROBO_WORKER_QUIET") == "1":
        sink_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(sink_fd, 1)
        os.dup2(sink_fd, 2)
        os.close(sink_fd)
    # Parent owns terminal interrupts.  Letting a Ctrl-C hit CUDA graph capture in this child leaves an
    # indeterminate GPU process while the parent may immediately create a new GL viewer.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        cfg, _, _, _ = robot_scene(robot_name, scenario_name)
        print("[curobo] MPC worker initializing planner session", flush=True)
        session = make_planner_session(cfg)
        print("[curobo] MPC worker capturing pose graph", flush=True)
        # Warm the pose CUDA-graph NOW on a perception-free nominal world (deployment parity: the planner
        # boots before it knows the real table/shelf/cube). The ~3.3s one-time capture overlaps the parent
        # env build + settle; every real plan-0 then reuses the graph (~0.1s update_world), off the reach
        # critical path. Best-effort: a failed nominal solve still captures the graph, so swallow errors
        # (a hard raise here would kill the worker and just defer the capture to the first real plan-0).
        try:
            warm_scene, warm_targets = _nominal_warm_scene_and_targets(cfg)
            t0 = time.monotonic()
            solve_reach_route(session, warm_scene, warm_targets, cfg.home_base_pos, cfg.home_base_quat,
                              goal="reach")
            print(f"[curobo] pose graph warmed on nominal world in {time.monotonic() - t0:.2f}s", flush=True)
        except Exception as warm_error:
            print(f"[curobo] nominal warm failed (non-fatal, graph likely still captured): "
                  f"{type(warm_error).__name__}: {warm_error}", flush=True)
        finally:
            warmed_event.set()
        while not stop_event.is_set():
            try:
                msg = requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if msg is None or msg[0] == "stop":
                break
            try:
                if msg[0] == "plan0":
                    _, generation, scene_dict, targets, base_pos, base_quat, fixed_assignment, seed_q, goal = msg
                    t_solve = time.monotonic()
                    route = solve_reach_route(session, scene_dict, targets, base_pos, base_quat,
                                              fixed_assignment=fixed_assignment, seed_q=seed_q, goal=goal)
                    print(f"[curobo] {goal} solve {time.monotonic() - t_solve:.2f}s "
                          f"(warm graph reused)", flush=True)
                    if _CAPTURE_DIR and route is None:
                        _capture_failed_plan0(robot_name, scenario_name, msg)
                    results.put(("plan0", route, generation))
                elif msg[0] == "fast_ik":
                    _, scene_dict, targets, base_pos, base_quat, fixed_assignment = msg
                    t_ik = time.monotonic()
                    feasible = ik_feasible_scene(
                        session, scene_dict, targets, base_pos, base_quat, fixed_assignment)
                    print(f"[curobo] fast-IK {'pass' if feasible else 'reject'} "
                          f"{time.monotonic() - t_ik:.2f}s", flush=True)
                    results.put(("fast_ik", feasible))
            except Exception as error:
                results.put(("error", f"{type(error).__name__}: {error}", traceback.format_exc()))
    except Exception:
        results.put(("worker_init_error", traceback.format_exc()))
        warmed_event.set()          # unblock a wait_warmed() waiter rather than dead-waiting the timeout


class SpawnedReachMpcWorker:
    """Persistent cuRobo process for plan-0 and asynchronous post-grasp return plans.

    ``plan0`` blocks only for initial route warmup. ``submit_plan0``/``poll_plan0`` let the shared walk FSM
    hold the final grasp command while the worker plans grasp->home. Per-tick tracking stays local Jacobian
    DLS, so no cuRobo solve enters the control loop.
    """

    def __init__(self, robot_name: str, scenario_name: str):
        self._ctx = mp.get_context("spawn")
        self._requests = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._stop_event = self._ctx.Event()
        self._warmed_event = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_reach_mpc_worker,
            args=(robot_name, scenario_name, self._requests, self._results, self._stop_event,
                 self._warmed_event),
            daemon=True, name="curobo-reach-mpc-worker")
        self._process.start()
        self.closed = False
        atexit.register(self.close)          # backstop: reap the child even on an abnormal (unclosed) exit
        # Every plan0-family request (reach plan-0, final-replan, retract-home) shares this ONE serial
        # worker + single-slot results queue. A request submitted but never polled before the caller moves
        # on (a --record-trials sweep resetting into the next trial; a superseded final-replan) leaves its
        # eventual reply queued FIFO ahead of the NEXT submission's reply. Tagging submit/poll with a
        # monotonic generation lets ``poll_plan0`` silently drop such orphaned stale replies instead of
        # mismatching them against the CURRENT request's scene (observed as ``JacobianReachTracker.rebind``
        # KeyError: a stale multi-cube route installed against a since-narrowed single-cube scene).
        self._plan0_gen = 0

    def wait_warmed(self, timeout_s: float = 90.0) -> None:
        """Block until the worker's one-time CUDA-graph capture has been attempted (success or failure).
        Callers about to create a concurrent GPU context (e.g. an OpenGL viewer) on this GPU MUST call this
        first -- see ``_reach_mpc_worker`` docstring for why concurrent capture + GL activity crashes."""
        deadline = time.monotonic() + timeout_s
        next_status = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._warmed_event.wait(timeout=0.1):
                return
            if not self._process.is_alive():
                raise RuntimeError(f"cuRobo MPC worker died (exit {self._process.exitcode}) before warming")
            if time.monotonic() >= next_status:
                print("[curobo] waiting for MPC CUDA-graph warmup before concurrent GPU initialization", flush=True)
                next_status += 5.0
        raise TimeoutError(f"cuRobo MPC worker: warm-up not confirmed after {timeout_s}s")

    def _recv(self, kind: str, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                item = self._results.get(timeout=0.05)
            except queue.Empty:
                # Fast-fail on a dead worker. A CUDA fault (e.g. the wider-gripper trajopt crash)
                # aborts the child before its except-handler can enqueue an "error", so without a
                # liveness check the parent dead-waits the full timeout_s (~180s) and looks hung.
                # Drain once more in case the final reply is still in flight over the pipe, then raise.
                if not self._process.is_alive():
                    try:
                        item = self._results.get(timeout=0.2)
                    except queue.Empty:
                        raise RuntimeError(
                            f"cuRobo MPC worker died (exit {self._process.exitcode}) "
                            f"before replying {kind!r}")
                else:
                    continue
            if item[0] in ("error", "worker_init_error"):
                raise RuntimeError(item[1] if len(item) < 3 else f"{item[1]}\n{item[2]}")
            assert item[0] == kind, f"expected {kind!r} reply, got {item[0]!r}"
            return item
        raise TimeoutError(f"cuRobo MPC worker: no {kind!r} reply after {timeout_s}s")

    def plan0(self, scene_dict, targets, base_pos, base_quat, timeout_s: float = 180.0,
              fixed_assignment=None, seed_q=None, goal: str = "reach"):
        """BLOCK on plan-0: solve the collision-free reach in-child. Returns the ``ReachRoute`` (or ``None``
        if nothing reachable). This is the one-time cuRobo/CUDA warmup. ``fixed_assignment`` = OPTIONAL
        ``{side: cube_name}`` pin (walk-search passes the gate-winning arm):
        skips cuRobo's geometric-prior ladder + single-arm fallback, forcing the reach onto that side.
        ``seed_q``/``goal`` = the RETRACT seam (same request, sent a second time): ``goal='home'`` plans back
        to the home tool pose, ``seed_q`` = the current grasp config so the return starts where the arm IS
        (else a home-seeded solve is degenerate home->home). ``goal='reach'``/``seed_q=None`` = the reach."""
        self._plan0_gen += 1
        self._requests.put(("plan0", self._plan0_gen, scene_dict, targets,
                            np.asarray(base_pos, dtype=np.float32), np.asarray(base_quat, dtype=np.float32),
                            fixed_assignment, seed_q, goal))
        _, route, _generation = self._recv("plan0", timeout_s)
        return route

    def submit_plan0(self, scene_dict, targets, base_pos, base_quat, *, fixed_assignment, seed_q,
                     goal: str) -> int:
        """Queue one warmed solve without blocking the control loop. Returns the request's generation --
        the caller must thread it through to the matching ``poll_plan0`` so a reply orphaned by a
        superseded/abandoned request (e.g. a prior sweep trial) is never mismatched against a newer one."""
        assert not self.closed, "cannot submit to a closed cuRobo worker"
        self._plan0_gen += 1
        self._requests.put(("plan0", self._plan0_gen, scene_dict, targets,
                            np.asarray(base_pos, dtype=np.float32), np.asarray(base_quat, dtype=np.float32),
                            fixed_assignment, seed_q, goal))
        return self._plan0_gen

    def submit_fast_ik(self, scene_dict, targets, base_pos, base_quat, *, fixed_assignment) -> None:
        """Queue legacy collision-blind IK for one predicted recovery stance."""
        assert not self.closed, "cannot submit to a closed cuRobo worker"
        self._requests.put(("fast_ik", scene_dict, targets,
                            np.asarray(base_pos, dtype=np.float32), np.asarray(base_quat, dtype=np.float32),
                            fixed_assignment))

    def poll_plan0(self, expected_generation: int) -> tuple[bool, object | None]:
        """Return ``(ready, route)`` for the request tagged ``expected_generation``; a ready ``None`` route
        means return planning was infeasible. Drains and discards any reply whose generation predates
        ``expected_generation`` -- an orphaned reply from a request the caller already abandoned (a
        superseded final-replan, or a prior ``--record-trials`` trial's unpolled submission) -- so it can
        never be installed against the wrong (newer) scene."""
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError(f"cuRobo MPC worker died (exit {self._process.exitcode}) while planning")
                return False, None
            if item[0] in ("error", "worker_init_error"):
                raise RuntimeError(item[1] if len(item) < 3 else f"{item[1]}\n{item[2]}")
            assert item[0] == "plan0", f"expected 'plan0' reply, got {item[0]!r}"
            _, route, generation = item
            if generation == expected_generation:
                return True, route

    def poll_fast_ik(self) -> tuple[bool, bool | None]:
        """Poll one predicted-stance IK predicate; never overlaps a plan-0 request."""
        try:
            item = self._results.get_nowait()
        except queue.Empty:
            if not self._process.is_alive():
                raise RuntimeError(f"cuRobo MPC worker died (exit {self._process.exitcode}) during fast IK")
            return False, None
        if item[0] in ("error", "worker_init_error"):
            raise RuntimeError(item[1] if len(item) < 3 else f"{item[1]}\n{item[2]}")
        assert item[0] == "fast_ik", f"expected 'fast_ik' reply, got {item[0]!r}"
        return True, bool(item[1])

    def close(self, *, force: bool = False) -> None:
        """Reap worker; force-kill during parent interruption so next viewer starts GPU-clean.

        Normal shutdown lets a completed solve exit through its queue.  Ctrl-C cannot safely wait for an
        in-flight CUDA capture, so callers use ``force=True`` and synchronously reap it before re-raising.
        """
        if self.closed:
            return
        self.closed = True
        self._stop_event.set()
        if not force:
            try:
                self._requests.put_nowait(("stop",))
            except queue.Full:
                pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
