"""Planning-based arm IK via cuRobo (plan `glistening-tinkering-gosling.md`).

Alternative to ``ik_mink.BatchedMinkIK``: instead of a 1-step reactive QP per tick, this
module plans a collision-free, non-awkward full joint trajectory ONCE per target via
cuRobo's ``MotionPlanner``, then replays it open-loop for `step_dt` ticks (one waypoint
per tick), then re-plans when the target moves.

Public surface mirrors ``BatchedMinkIK``'s ``solve()`` shape but does NOT expose a per-tick
QP solve -- it exposes:

* ``plan(target_in_base, side, current_state) -> JointState`` -- replanned on call; caches
  the interpolated trajectory for subsequent ``step()`` calls until the next plan/swap.
* ``step() -> np.ndarray | None`` -- returns the next curobo-waypoint joint state
  (waist+arm template + reaching arm columns, mjlab arm_ref convention), advancing the
  internal waypoint index. Returns None once the buffered trajectory is fully consumed.

Phase 3 verify (front reach, 0.0000 m gripper-site error, 61 waypoints, 0.06 s) precedes this code.
The lower bound on `step()` consumer-step frequency is `interpolation_dt` (typically 0.025 s
in curobo's motion planner).

Scope decisions baked in here (don't re-litigate without a new premise):
- **Front and rear both reachable**: the reach goal is a GOALSET of full-6-DOF cube grasp
  candidates (``ik_curobo_robot_cfg.cube_grasp_poses_obj`` transformed to base_link by the
  object pose, fed via ``set_reaching_goalset``); cuRobo mins over the set, so rear is reached
  by whichever candidate (e.g. a rotated-yaw face) is reachable rather than by de-weighting
  orientation. Full orientation is enforced per candidate. IK seed count is 128: same per-plan
  precision as 256, less warmup cost. Validated: front + rear gripper-site error 0.0000 m
  (tolerance 0.01 m).
- **Plan-once open-loop**: no periodic replanning / base-sway correction. Plan is
  refreshed only when ``plan(...)`` is called again.
- **Idle arm stay-put at current pose**: the plan pins the non-reaching
  ``end_effector_<idle>_site`` to its live forward-kinematics pose so cuRobo's IK doesn't try to drive it
  too (the goal tool pose must cover all declared tool frames; see Phase-3 fixing
  ``reorder_links`` error).

Frame / mapping conventions:
- All targets / outputs are in the **base_link frame** (matches ``ik_mink``'s convention
  and ``pickplace_reach_env``'s reach_target_in_base computation).
- curobo_q and mjlab's joint_qpos are equal *except* the wrist_3 MuJoCo ``ref`` offset
  (``left_wrist_3_joint ref=-1.5708``, ``right_wrist_3_joint ref=+1.5708``). Apply via
  ``mujoco_qpos = curobo_q + ref_offsets[jname]`` on trajectory playback.
"""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Callable, Mapping

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.robot import RobotCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.batch_motion_planner import BatchMotionPlanner

from tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import BASE_LINK, build_motion_planner_kwargs, build_robot_cfg_dict_from_urdf, cube_grasp_poses_obj, grasp_poses_to_base, set_bimanual_goalset, set_reaching_goalset


def _geom_world_rot(mj_model, geom_id) -> np.ndarray:
    quat = mj_model.geom_quat[geom_id]
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _box_oriented(mj_model, geom_id) -> tuple[np.ndarray, np.ndarray]:
    """A box IS a cuRobo cuboid -- its own half-extents and its own frame, no bounding needed."""
    return 2.0 * mj_model.geom_size[geom_id], _geom_world_rot(mj_model, geom_id)


def _capsule_oriented(mj_model, geom_id) -> tuple[np.ndarray, np.ndarray]:
    """Tightest box around a capsule: square cross-section 2r, length 2*(half_len + r) for the caps.

    Roll about the capsule's own axis is DROPPED (the returned frame is the minimal rotation from
    +z to that axis). A capsule is rotationally symmetric so its roll is arbitrary, but the box is
    not -- keeping the compiler's roll would spin a square cross-section by whatever angle the
    figure builder happened to emit, changing the obstacle without changing the capsule.
    """
    radius, half_length = mj_model.geom_size[geom_id][:2]
    dims = 2.0 * np.array([radius, radius, half_length + radius])
    axis = _geom_world_rot(mj_model, geom_id)[:, 2]
    cross = np.cross([0.0, 0.0, 1.0], axis)
    sin_a = np.linalg.norm(cross)
    if sin_a < 1e-9:                                   # parallel or antiparallel to +z
        rot = np.diag([1.0, -1.0, -1.0]) if axis[2] < 0 else np.eye(3)
        return dims, rot
    rotvec = cross / sin_a * np.arctan2(sin_a, float(axis[2]))
    return dims, Rotation.from_rotvec(rotvec).as_matrix()


def _resolve_base_body_id(mj_model, base_link_name: str) -> int:
    for bid in range(mj_model.nbody):
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if n is not None and (n == base_link_name or n.split("/")[-1] == base_link_name):
            return bid
    raise AssertionError(f"base link {base_link_name!r} not found in mj_model")


_PROBE_DROP = tuple(s for s in os.environ.get("REACH_PROBE_DROP_GEOMS", "").split(",") if s)
_PROBE_DUMP = bool(os.environ.get("REACH_PROBE_DUMP_OBS"))
_probe_dumped_rear = False

# Cache for ``build_curobo_scene``. The static obstacle world (workbench/shelf/hand figure, every
# ``pp_*`` non-marker geom) is determined by ``mj_model.qpos0`` + ``mj_model.geom_pos`` + ``base_pos``
# + ``base_quat``. Every call from the per-solve path goes through here with the same inputs after the
# first reset of a given scenario, so the second-and-later calls hit. ``move_pick_geoms`` mutates
# ``geom_pos`` on every seeded reset (curobo_reach_harness.py:1842), so the bytes fingerprint evicts
# cached entries that would be stale against the new world. ``id(mj_model)`` splits per-robot and
# per-process so two harnesses in the same process don't conflate. Probes bypass: a probe must see
# every call to attribute misbehavior. ponytail: 64-entry ceiling; raise if it ever fills.
_SCENE_CACHE: dict[tuple, tuple[dict, Callable]] = {}
_SCENE_CACHE_MAX = 64


def _scene_cache_key(mj_model, base_pos, base_quat):
    return (id(mj_model),
            tuple(base_pos) if base_pos is not None else None,
            tuple(base_quat) if base_quat is not None else None,
            mj_model.qpos0.tobytes(),
            mj_model.geom_pos.tobytes())


def _probe_dump_robot_rear(mj_model, d, to_base_frame):
    """TEMPORARY probe: rearmost extent of the robot's cuRobo collision spheres in base frame."""
    global _probe_dumped_rear
    if _probe_dumped_rear:
        return
    worst = []
    for gi in range(mj_model.ngeom):
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if not n.startswith("planner_sphere"):
            continue
        worst.append((to_base_frame(d.geom_xpos[gi])[0] - mj_model.geom_size[gi][0], n))
    if worst:
        worst.sort()
        print(f"[probe] robot sphere rearmost base_x: "
              + ", ".join(f"{x:.3f} {n}" for x, n in worst[:5]), file=sys.stderr)
        _probe_dumped_rear = True


def build_curobo_scene(mj_model, base_link_name: str = BASE_LINK,
                       base_pos=None, base_quat=None) -> tuple[dict, Callable]:
    """Build obstacle Cuboids in base_link frame from compiled MuJoCo collision geoms.

    Runtime planner and standalone tests share this path so obstacle geometry stays identical.
    Accepts box/capsule props welded by ``add_props_to_spec`` (the ``pp_*`` geoms with
    ``contype != 0`` -- the workbench tops/legs, the shelf tiers/posts, and the offered-hand
    proxies) at ANY orientation: cuRobo cuboids are oriented boxes (``CuboidData.inv_pose`` carries
    a quaternion and the SDF kernel transforms each query point into obstacle-local frame
    unconditionally), so the true frame is passed through rather than bounded away. An earlier
    revision emitted world-axis-aligned bounding boxes and asserted every geom was already
    axis-aligned; that forced the human proxies to be padded to their larger half-extent to stay
    legal at arbitrary yaw, which cost more clearance than the bounding ever saved. Pick markers are
    excluded because they are added explicitly as grasp-object obstacles.
    Optional base pose override handles verify scripts whose compiled qpos0 root sits at origin
    while env reset applies HOME later.
    """
    base_body_id = _resolve_base_body_id(mj_model, base_link_name)
    if not (_PROBE_DUMP or _PROBE_DROP):
        key = _scene_cache_key(mj_model, base_pos, base_quat)
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            return hit
    d = mujoco.MjData(mj_model)
    d.qpos[:] = mj_model.qpos0
    if base_pos is not None or base_quat is not None:
        base_joint = mj_model.joint(mj_model.body(base_body_id).jntadr[0])
        adr = base_joint.qposadr[0]
        if base_pos is not None:
            d.qpos[adr:adr + 3] = base_pos
        if base_quat is not None:
            d.qpos[adr + 3:adr + 7] = base_quat
    mujoco.mj_forward(mj_model, d)
    base_pos_w = d.xpos[base_body_id].copy()
    base_rot = d.xmat[base_body_id].reshape(3, 3).copy()

    def to_base_frame(world_pos):
        return base_rot.T @ (np.asarray(world_pos) - base_pos_w)

    cuboids = {}
    for gi in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        # Pick `pp_mark_*` markers are excluded here -- they're added explicitly as obstacles
        # via ``_object_cuboids`` (keyed `pp_obj_*`). Everything else with ``contype != 0`` is
        # imported, including the shelf tier boxes + corner posts -- collision avoidance on the
        # real rack is the headline behavior.
        if not (name.startswith("pp_") and mj_model.geom_contype[gi] != 0
                and not name.startswith("pp_mark_")):
            continue
        if _PROBE_DROP and any(s in name for s in _PROBE_DROP):
            print(f"[probe] dropped obstacle {name}", file=sys.stderr)
            continue   # TEMPORARY probe: attribute a plan-0 infeasibility to a specific obstacle
        if _PROBE_DUMP and ("hand_R" in name or "table_F_col_top" in name):
            print(f"[probe] obs {name} base_xyz="
                  f"{np.round(to_base_frame(d.geom_xpos[gi]), 3).tolist()} "
                  f"size={np.round(mj_model.geom_size[gi], 3).tolist()}", file=sys.stderr)
            _probe_dump_robot_rear(mj_model, d, to_base_frame)
        geom_type = mj_model.geom_type[gi]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            dims, rot_w = _box_oriented(mj_model, gi)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            dims, rot_w = _capsule_oriented(mj_model, gi)
        else:
            raise AssertionError(f"{name}: unsupported geom type {geom_type}")
        # Express the geom's own world frame in the moving base frame, so a fixed table keeps its
        # world orientation as the root yaws. Identity here would spin every obstacle with the base.
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (base_rot.T @ rot_w).reshape(9))
        cuboids[name] = {
            "dims": dims.tolist(),
            "pose": to_base_frame(mj_model.geom_pos[gi]).tolist() + quat.tolist(),
        }
    result = (cuboids, to_base_frame)
    if not (_PROBE_DUMP or _PROBE_DROP):
        if len(_SCENE_CACHE) < _SCENE_CACHE_MAX:
            _SCENE_CACHE[key] = result
    return result


class HandFloorSelfCollisionCost:
    """Wrap self-collision cost and add scoped one-sided z-floor on gripper spheres."""

    def __init__(self, wrapped, hand_sphere_idx: torch.Tensor, z_floor: float, weight: float):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_hand_idx", hand_sphere_idx)
        object.__setattr__(self, "_z_floor", float(z_floor))
        object.__setattr__(self, "_floor_weight", float(weight))

    def forward(self, robot_spheres: torch.Tensor) -> torch.Tensor:
        base = self._wrapped.forward(robot_spheres)
        z = robot_spheres[..., self._hand_idx, 2]
        below = torch.clamp(self._z_floor - z, min=0.0)
        floor = self._floor_weight * (below ** 2).sum(dim=-1, keepdim=True)
        return base + floor

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped"), name)


def _hand_sphere_indices(kinematics_config, reaching_link_names: list[str]) -> torch.Tensor:
    sphere_link_ids = kinematics_config.link_sphere_idx_map
    link_name_to_id = kinematics_config.link_name_to_idx_map
    reaching_link_ids = [link_name_to_id[name] for name in reaching_link_names]
    is_reaching_link_sphere = torch.zeros_like(sphere_link_ids, dtype=torch.bool)
    for link_id in reaching_link_ids:
        is_reaching_link_sphere |= sphere_link_ids == link_id
    return torch.nonzero(is_reaching_link_sphere).flatten().to(sphere_link_ids.device)


_OPTIMIZED_MANAGER_ATTRS = ("cost_manager", "constraint_manager", "hybrid_cost_constraint_manager")


def attach_hand_floor_cost(solver, reaching_link_names: list[str], z_floor: float, weight: float) -> int:
    """Attach scoped z-floor by riding cuRobo's existing self_collision slot before graph capture."""
    hand_sphere_count, wrapped_manager_count = 0, 0
    for rollout in solver.get_all_rollout_instances():
        kinematics_config = rollout.transition_model.robot_model.kinematics_config
        hand_sphere_idx = _hand_sphere_indices(kinematics_config, reaching_link_names)
        for attr in _OPTIMIZED_MANAGER_ATTRS:
            manager = getattr(rollout, attr, None)
            if manager is None or not manager.has_cost("self_collision"):
                continue
            wrapped = manager.get_cost("self_collision")
            if isinstance(wrapped, HandFloorSelfCollisionCost):
                continue
            hand_sphere_count = int(hand_sphere_idx.numel())
            wrapped_manager_count += 1
            manager.costs["self_collision"] = HandFloorSelfCollisionCost(wrapped, hand_sphere_idx, z_floor, weight)
    if wrapped_manager_count == 0:
        raise RuntimeError("attach_hand_floor_cost: no optimized cost manager with a self_collision "
                           "cost found -- the ride-along seam is broken; inspect rollout managers")
    return hand_sphere_count


GRIPPER_RACK_LINKS = ("L_left_rack", "L_right_rack", "R_left_rack", "R_right_rack")
HAND_FLOOR_WEIGHT = 1_000_000.0


def attach_hand_floor_to_planner(planner, z_floor: float, weight: float,
                                 reaching_link_names: list[str] = list(GRIPPER_RACK_LINKS)) -> int:
    """Attach scoped gripper z-floor to plan_pose solvers before `planner.warmup()`."""
    n = 0
    for solver in (planner.trajopt_solver, planner.ik_solver):
        n = attach_hand_floor_cost(solver, reaching_link_names, z_floor, weight)
    return n


def build_curobo_motion_planner(scene_dict: dict, robot_cfg_dict: dict | None = None,
                                 max_attempts: int = 10, enable_graph_attempt: int = 0,
                                 hand_z_floor: float | None = 0.06,
                                 hand_floor_weight: float = HAND_FLOOR_WEIGHT,
                                 arm_joint_home: Mapping[str, float] | None = None,
                                 use_cuda_graph: bool = True,
                                 max_goalset: int | None = None) -> MotionPlanner:
    """Build warmed MotionPlanner with project-default reach config and table hand-floor guard.

    ``arm_joint_home`` is a named MuJoCo-qpos map for active arm cspace defaults. Default ``None``
    means ``humanoid_v21_constants.HOME_KEYFRAME.joint_pos``. Ignored when caller supplies a fully
    built ``robot_cfg_dict``; that dict already fixes the cuRobo cspace.

    ``use_cuda_graph=False`` is for cspace return planning; pose-goal and cspace-goal graph shapes
    differ, so a pose-warmed graph cannot be reused for the latter.
    """
    robot_cfg = (
        robot_cfg_dict
        if robot_cfg_dict is not None
        else build_robot_cfg_dict_from_urdf(arm_joint_home=arm_joint_home)
    )
    planner_kwargs = build_motion_planner_kwargs()
    if max_goalset is not None:
        planner_kwargs["max_goalset"] = int(max_goalset)
    planner = MotionPlanner(
        MotionPlannerCfg.create(robot=robot_cfg, scene_model=scene_dict,
                                 **planner_kwargs, use_cuda_graph=use_cuda_graph)
    )
    print("[curobo] world collision activation: "
          f"{planner_kwargs['optimizer_collision_activation_distance']:.3f} m")
    if hand_z_floor is not None:
        n = attach_hand_floor_to_planner(planner, z_floor=hand_z_floor, weight=hand_floor_weight)
        print(f"[curobo] hand z-floor: {n} gripper spheres >= {hand_z_floor:.3f} m (base), weight {hand_floor_weight:g}")
    planner.warmup(enable_graph=use_cuda_graph, num_warmup_iterations=5)
    planner._plan_max_attempts = max_attempts
    planner._plan_enable_graph_attempt = enable_graph_attempt
    return planner


def build_curobo_batch_planner(scene_dict: dict, robot_cfg_dict: dict, max_batch_size: int,
                               max_attempts: int = 10,
                               hand_z_floor: float | None = 0.06,
                               hand_floor_weight: float = HAND_FLOOR_WEIGHT,
                               max_goalset: int | None = None) -> BatchMotionPlanner:
    """Warmed ``BatchMotionPlanner`` (``multi_env=True``): one INDEPENDENT collision env per batch row,
    solving up to ``max_batch_size`` pose problems in a single IK+TrajOpt pass. Mirrors
    ``build_curobo_motion_planner`` reach config (same optimizer costs, hand z-floor, goalset cap) so a
    batched stance's arm plan is identical to what a B=1 ``commit`` at that stance would produce.

    ``multi_env=True`` disables the PRM graph planner (trajopt-seed only; graph shapes are per-env),
    so ``enable_graph_attempt`` is not accepted here -- raise ``max_attempts`` to recover the seeds a
    graph would have provided. ``multi_env`` is an INDEPENDENT allocation, so it coexists in-process
    with the B=1 ``CuroboPlannerSession`` (the historical 6-way CUDA crash was cross-PROCESS).

    ``max_goalset`` is descriptor-derived by the production caller; fewer candidates pad internally.
    ``scene_dict`` seeds env 0 at build; per-row worlds are loaded later via
    ``BatchMotionPlanner.scene_collision_checker.load_collision_model(SceneCfg, env_idx=i)``."""
    # multi_env sizes the collision world to len(scene_model): pass a LIST of N seed worlds so
    # scene_collision_checker allocates one env per batch row (MotionPlannerCfg.create does NOT forward
    # num_envs to SceneCollisionCfg; a single dict would build a 1-env world and load_collision_model(
    # env_idx>0) would index out of bounds). Rows are overwritten per-reach via load_worlds.
    # use_cuda_graph=True is valid ONLY because every ``plan_pose`` call fills a CONSTANT batch of
    # exactly ``max_batch_size`` rows (the caller pads the seen candidates up to it): a fixed shape lets
    # the graph be captured once and replayed, which is what makes the batch a net WIN over the serial
    # short-circuit (without it, per-call graph capture dominates and the batch is slower). A variable
    # batch would trip "CUDA graph reset is not available" -- hence the mandatory padding.
    planner_kwargs = build_motion_planner_kwargs()
    if max_goalset is not None:
        planner_kwargs["max_goalset"] = int(max_goalset)
    planner = BatchMotionPlanner(
        MotionPlannerCfg.create(robot=robot_cfg_dict, scene_model=[scene_dict] * max_batch_size,
                                **planner_kwargs,
                                use_cuda_graph=True, max_batch_size=max_batch_size, multi_env=True)
    )
    if hand_z_floor is not None:
        n = attach_hand_floor_to_planner(planner, z_floor=hand_z_floor, weight=hand_floor_weight)
        print(f"[curobo-batch] hand z-floor: {n} gripper spheres >= {hand_z_floor:.3f} m (base), "
              f"weight {hand_floor_weight:g}")
    planner.warmup(enable_graph=False)
    planner._plan_max_attempts = max_attempts
    return planner


def plan_single_reach(planner: MotionPlanner, kin: Kinematics, target_pos_base: np.ndarray,
                      current_state: JointState, side: str,
                      obj_quat_base=(1.0, 0.0, 0.0, 0.0), flip: bool = True):
    """Plan one reaching arm while pinning the other hand to its current forward-kinematics pose.

    ``flip`` -> ``cube_grasp_poses_obj(flip)``: True = 8-candidate goalset (4 yaws x top/down twin),
    False = 4 (yaws only, no top/down twin). A smaller goalset means fewer discrete candidates the
    solver can jump between across successive re-plans (the source of the --replan mid-run branch
    flip); default True keeps the RL planner's full set unchanged."""
    side = side.upper()
    idle_side = "L" if side == "R" else "R"
    device = planner.device_cfg.device
    idle_state = kin.compute_kinematics(current_state)
    idle_pose = idle_state.tool_poses.get_link_pose(f"end_effector_{idle_side}_site")

    obj_pos = torch.as_tensor(target_pos_base, device=device, dtype=torch.float32)
    obj_quat = torch.as_tensor(obj_quat_base, device=device, dtype=torch.float32)
    grasp_pos_obj, grasp_quat_obj = cube_grasp_poses_obj(flip=flip, device=device)
    cand_pos, cand_quat = grasp_poses_to_base(obj_pos, obj_quat, grasp_pos_obj, grasp_quat_obj)
    position, quaternion = set_reaching_goalset(
        idle_pose.position[0], idle_pose.quaternion[0], cand_pos, cand_quat
    )
    goal_pose = GoalToolPose(
        tool_frames=[f"end_effector_{idle_side}_site", f"end_effector_{side}_site"],
        position=position,
        quaternion=quaternion,
    )
    result = planner.plan_pose(
        goal_pose, current_state,
        max_attempts=planner._plan_max_attempts,
        enable_graph_attempt=planner._plan_enable_graph_attempt,
    )
    assert result is not None and result.success.any(), f"plan_pose failed for target {target_pos_base}"
    return result


def plan_bimanual_reach(planner: MotionPlanner, left_target_base: np.ndarray,
                        right_target_base: np.ndarray, current_state: JointState,
                        left_obj_quat=(1.0, 0.0, 0.0, 0.0),
                        right_obj_quat=(1.0, 0.0, 0.0, 0.0), flip: bool = True):
    """Plan both hands in one `plan_pose`; caller checks success for assignment fallback.

    ``flip`` -> ``cube_grasp_poses_obj(flip)`` (8 vs 4 candidates); see ``plan_single_reach``."""
    device = planner.device_cfg.device
    grasp_pos_obj, grasp_quat_obj = cube_grasp_poses_obj(flip=flip, device=device)

    def grasp_candidates_for_target(target_base, obj_quat):
        obj_pos = torch.as_tensor(target_base, device=device, dtype=torch.float32)
        obj_quat_tensor = torch.as_tensor(obj_quat, device=device, dtype=torch.float32)
        return grasp_poses_to_base(obj_pos, obj_quat_tensor, grasp_pos_obj, grasp_quat_obj)

    left_pos, left_quat = grasp_candidates_for_target(left_target_base, left_obj_quat)
    right_pos, right_quat = grasp_candidates_for_target(right_target_base, right_obj_quat)
    position, quaternion = set_bimanual_goalset(left_pos, left_quat, right_pos, right_quat)
    goal_pose = GoalToolPose(
        tool_frames=["end_effector_L_site", "end_effector_R_site"],
        position=position,
        quaternion=quaternion,
    )
    return planner.plan_pose(
        goal_pose, current_state,
        max_attempts=planner._plan_max_attempts,
        enable_graph_attempt=planner._plan_enable_graph_attempt,
    )


def final_active_q(interpolated, planner_joint_names: list[str]) -> torch.Tensor:
    """Slice final waypoint from full-body interpolated plan down to active planner joints by name."""
    name_to_i = {n: k for k, n in enumerate(interpolated.joint_names)}
    active_idx = [name_to_i[n] for n in planner_joint_names]
    return interpolated.position[0, 0, -1, active_idx]


def final_reach_error(kin: Kinematics, joint_names: list[str], final_q: torch.Tensor,
                      target_pos_base: np.ndarray, side: str) -> float:
    joint_state = JointState.from_position(final_q.unsqueeze(0), joint_names=joint_names)
    state = kin.compute_kinematics(joint_state)
    gripper_site_pos = state.tool_poses.get_link_pose(f"end_effector_{side.upper()}_site").position[0].cpu().numpy()
    return float(np.linalg.norm(gripper_site_pos - target_pos_base))


def cube_grasp_pose_base(mpc: MPCSolver, cube_pos_base, cube_quat_base, grasp_index: int = 0):
    """One FIXED grasp candidate ``(pos[3], quat[4])`` in base_link for a cube pose in base_link.

    Transforms candidate ``grasp_index`` of ``cube_grasp_poses_obj`` by the LIVE cube pose
    (``cube_pos_base`` position, ``cube_quat_base`` orientation -- both drift with the base). "Fixed
    candidate" is a fixed CHOICE of grasp, not a static base-frame pose: the returned pose updates every
    call as the cube drifts. v1 uses one candidate (deterministic goal, no goalset-min flip); v2 = pass
    all G candidates with ``max_goalset=G``.
    """
    device = mpc.device_cfg.device
    grasp_pos_obj, grasp_quat_obj = cube_grasp_poses_obj(device=device)
    obj_pos = torch.as_tensor(cube_pos_base, device=device, dtype=torch.float32)
    obj_quat = torch.as_tensor(cube_quat_base, device=device, dtype=torch.float32)
    cand_pos, cand_quat = grasp_poses_to_base(obj_pos, obj_quat, grasp_pos_obj, grasp_quat_obj)
    return cand_pos[grasp_index], cand_quat[grasp_index]


def mpc_tool_goal(sides, positions, quaternions) -> GoalToolPose:
    """Stack per-hand final base-frame tool poses into a ``[1,1,L,1,3/4]`` ``GoalToolPose`` (num_goalset=1).

    ``sides`` order MUST match ``mpc.tool_frames`` (both hands, ``["L","R"]``). ``positions[i]`` /
    ``quaternions[i]`` are the ALREADY-final goal pose for hand ``i`` -- a tracked hand passes a
    ``cube_grasp_pose_base`` output, a pinned (idle) hand passes its constant FK pose.
    """
    position = torch.stack(list(positions), dim=0).view(1, 1, len(sides), 1, 3)
    quaternion = torch.stack(list(quaternions), dim=0).view(1, 1, len(sides), 1, 4)
    return GoalToolPose(
        tool_frames=[f"end_effector_{s.upper()}_site" for s in sides],
        position=position, quaternion=quaternion,
    )


class CuroboArmPlanner:
    """Plan-then-follow IK for a 7-DOF arm subtree, single-shot per target.

    Builds the curobo ``MotionPlanner`` in ``__init__`` (CUDA-graph warmup is a one-shot cost,
    after which plans are ~50-100 ms each). Holds the
    interpolated trajectory in a CPU float32 buffer until consumed.

    Side-agnostic: ``side="L"|"R"`` selects which ``end_effector_<side>_site`` to drive; the
    other side is pinned to its current forward-kinematics pose via ``tool_frames=[.., ..]`` covering
    both cuRobo ``tool_frames`` declared on the URDF (Phase-3 ``reorder_links`` gotcha).
    """

    def __init__(
        self,
        mj_model,
        device: str | torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        max_attempts: int = 10,
        enable_graph_attempt: int = 0,
        hand_z_floor: float | None = 0.06,
        hand_floor_weight: float = HAND_FLOOR_WEIGHT,
        arm_joint_home: Mapping[str, float] | None = None,
        robot_cfg_dict: dict | None = None,
        robot_descriptor=None,
        base_link: str = BASE_LINK,
        home_base_pos=None,
    ) -> None:
        """mj_model: live compiled MuJoCo model (required to merge scenario table
        geometry + to read ref offsets for curobo_q -> mjlab_qpos).

        ``hand_z_floor``: minimum gripper-sphere height in base_link frame (meters). Scoped z-floor
        keeps the graspers off the table top (base-frame table top ~0.02 m; without it plan_pose
        skims the hand to ~0.05 m center, <3.5 cm clearance). ``None`` disables. ``hand_floor_weight``
        tunes softness -- a gentle push (default) that lifts the trajectory without flipping
        plan_pose to infeasible. ``arm_joint_home`` overrides the active-arm cuRobo cspace default;
        leave ``None`` to use ``HOME_KEYFRAME.joint_pos``. See ``attach_hand_floor_to_planner`` in
        this module.

        Multi-robot (workspace generator): a fully built ``robot_cfg_dict`` (e.g. the G1 cfg) plus its
        ``base_link`` and ``home_base_pos`` (base translation the scene is baked at) override the
        humanoid_v21 defaults, so the same plan-then-follow machinery drives another floating-base
        arm. When ``robot_cfg_dict`` is given, ``arm_joint_home`` is ignored (the dict already fixes
        the cspace)."""
        self._mj_model = mj_model
        self._device = torch.device(device)
        self._robot_descriptor = robot_descriptor
        robot_cfg_dict = (
            robot_cfg_dict
            if robot_cfg_dict is not None
            else build_robot_cfg_dict_from_urdf(arm_joint_home=arm_joint_home)
        )

        # Table collision scene, baked ONCE from the live mj_model at construction (same builder the
        # standalone phase-3 verify uses -- ``build_curobo_scene``). ``base_link`` is
        # suffix-resolved so the entity-namespaced ``robot/base_link`` in the merged scene works.
        # CRITICAL: the env's compiled ``qpos0`` puts the base free-joint at the ORIGIN (z=0), NOT the
        # standing HOME pose -- mjlab applies the HOME_KEYFRAME init state at reset, not into qpos0. So
        # the scene MUST be baked with the HOME base pose (z=0.59) explicitly, exactly as verify does
        # (``HOME_BASE_POS``); baking at qpos0 placed the table ~0.59 m too high in base_link frame,
        # and the planner then routed the reaching hand straight through the real table (it avoided a
        # phantom table up near the shoulders). Baked at construction, NOT re-baked per plan: cuRobo's
        # MotionPlanner scene is fixed at cfg time (a rebuild needs a fresh ~5s CUDA-graph warmup), so
        # plan-once open-loop accepts the ~cm base sway while standing.
        from asset_zoo.humanoid_v21.humanoid_v21_constants import HOME_KEYFRAME
        scene_base_pos = HOME_KEYFRAME.pos if home_base_pos is None else home_base_pos
        self._scene_dict: dict = {"cuboid": build_curobo_scene(
            mj_model, base_link, base_pos=scene_base_pos, base_quat=(1.0, 0.0, 0.0, 0.0))[0]}
        self._planner = build_curobo_motion_planner(
            self._scene_dict,
            robot_cfg_dict=robot_cfg_dict,
            max_attempts=max_attempts,
            enable_graph_attempt=enable_graph_attempt,
            hand_z_floor=hand_z_floor,
            hand_floor_weight=hand_floor_weight,
        )

        # cuRobo's read-only kinematics instance for forward-kinematics-pinning the idle gripper site during plan().
        self._kin = Kinematics(RobotCfg.create(robot_cfg_dict).kinematics)
        self._joint_names = list(self._planner.joint_names)

        # ref offset = mj_model.qpos0[joint.qposadr[0]] (MuJoCo has no live ``jnt_ref``
        # field -- the XML ref is folded into qpos0 at compile time). Used to convert
        # curobo_q -> mjlab_qpos for trajectory playback. Env-level mj_models may carry
        # an entity-namespace prefix ("robot/left_shoulder_1_joint") that doesn't match
        # cuRobo's bare names; resolve via id-based qposadr instead of name lookup.
        self._ref_offsets = np.zeros(len(self._joint_names), dtype=np.float32)
        self._qpos_adr = np.zeros(len(self._joint_names), dtype=np.int64)  # live-qpos column per curobo joint
        for i, j in enumerate(self._joint_names):
            qposadr = None
            for jid in range(mj_model.njnt):
                n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                # match by suffix so both "robot/..." and bare names work
                if n is not None and (n == j or n.endswith(f"/{j}") or n.split("/")[-1] == j):
                    qposadr = int(mj_model.jnt_qposadr[jid])
                    break
            assert qposadr is not None, f"joint {j!r} not found in mj_model"
            self._qpos_adr[i] = qposadr
            self._ref_offsets[i] = mj_model.qpos0[qposadr] + 0.0  # explicit for clarity

        # Trajectory buffer (CPU float32), populated by plan(); consumed by step().
        self._traj_q_curobo: np.ndarray | None = None  # shape (H, n_arm_dof)
        self._traj_idx: int = 0

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def home_state(self) -> JointState:
        """cuRobo's canonical reach-ready home config as a ``[1, n_arm_dof]`` JointState.

        This is the SAME seed phase-3 verify plans from (``default_joint_state`` = the cfg's
        ``arm_joint_home``). A plan-once open-loop driver must seed here, NOT from the live arm:
        ``plan_pose`` is seed-sensitive and a frozen controller holds the arm several tenths of a
        radian off this pose (steady-state tracking error), which alone flips a reachable target to
        infeasible."""
        pos = self._planner.default_joint_state.clone().position.unsqueeze(0)
        return JointState.from_position(pos, joint_names=list(self._joint_names))

    def plan(self, target_in_base: np.ndarray, side: str, current_state: JointState,
             obj_quat_base=(1.0, 0.0, 0.0, 0.0)) -> JointState:
        """Plan a collision-free trajectory from ``current_state`` to the object at
        ``target_in_base`` using the named arm (``side`` in ``{'L', 'R'}``). Refreshes the buffered
        trajectory -- call before each new reach.

        ``target_in_base`` shape (3,) float32 -- object position in base_link (no world conversion
        needed; the env does that). ``obj_quat_base`` (wxyz) is the object orientation in base_link
        (default identity); the reaching side is driven to a GOALSET of cube grasp candidates built
        in the object frame and transformed to base_link by (target_in_base, obj_quat_base). A
        perception/grasp-planner object pose can be threaded here later without signature churn.

        Returns the curobo ``TrajOptSolverResult`` so callers can inspect success/position
        error/total_time. Raises ``AssertionError`` on plan failure (caller catches and
        falls back to the home pose / logs)."""
        if self._robot_descriptor is None:
            result = plan_single_reach(
                self._planner, self._kin, target_in_base, current_state, side=side,
                obj_quat_base=obj_quat_base,
            )
        else:
            # Import lazily: planner.py imports this facade to construct its warmed sessions.
            from tasks.visual_manipulation.curobo.planner import plan_single
            descriptor = self._robot_descriptor
            if side.upper() != descriptor.reaching_side:
                import dataclasses
                descriptor = dataclasses.replace(descriptor, reaching_side=side.upper())
            result = plan_single(descriptor, self._kin, self._planner, target_in_base, current_state)

        # The interpolated plan carries the FULL body (35 joints, active arm + locked), NOT just the
        # 14 active DOFs. Name-slice down to the active planner joints (``self._joint_names`` order,
        # which ``_ref_offsets`` is aligned to) so ``step``/``to_mjlab_qpos`` see (H, 14). Same
        # name-slice the phase-3 verify uses (``final_active_q``); trusting positional order broke on
        # the 35-vs-14 mismatch.
        interpolated = result.get_interpolated_plan()  # layout [B=1, S=1, H, dof]
        name_to_i = {n: k for k, n in enumerate(interpolated.joint_names)}
        active_idx = [name_to_i[n] for n in self._joint_names]
        self._traj_q_curobo = interpolated.position[0, 0][:, active_idx].cpu().numpy()  # (H, 14)
        self._traj_idx = 0
        return result

    def current_state_from_qpos(self, qpos_row: np.ndarray) -> JointState:
        """Build a curobo ``JointState`` (position only) from one env's LIVE mjlab qpos row.

        ``qpos_row`` shape ``(nq,)`` -- the full model qpos (mjlab column order). Slices the active
        arm columns (``self._qpos_adr``) and undoes the wrist_3 ``ref`` fold: ``curobo_q =
        mjlab_qpos - ref_offsets`` (inverse of ``to_mjlab_qpos``). Used to seed closed-loop replans
        from the arm's REAL pose instead of the home ``default_joint_state`` -- without this a replan
        just recomputes the same home->goal path and can't chase the tracking residual."""
        q_mjlab = np.asarray(qpos_row, dtype=np.float32)[self._qpos_adr]  # (n_arm_dof,)
        q_curobo = q_mjlab - self._ref_offsets
        pos = torch.as_tensor(q_curobo, device=self._device, dtype=torch.float32).unsqueeze(0)
        return JointState.from_position(pos, joint_names=list(self._joint_names))

    def waypoint(self, index: int) -> np.ndarray | None:
        """Return buffered waypoint at ``index`` (clamped to the last) as curobo_q, WITHOUT advancing
        the internal cursor. For closed-loop/MPC use: replan each tick, then command a fixed lookahead
        step along the fresh collision-free path (index~1), not the far goal (which would beeline
        through the table in joint space). Returns None if no trajectory is buffered."""
        if self._traj_q_curobo is None or self._traj_q_curobo.shape[0] == 0:
            return None
        i = min(int(index), self._traj_q_curobo.shape[0] - 1)
        return self._traj_q_curobo[i]

    def step(self) -> np.ndarray | None:
        """Pop the next waypoint from the buffered trajectory as curobo_q (shape
        ``(n_arm_dof,)`` float32). Returns None once the trajectory is exhausted."""
        if self._traj_q_curobo is None:
            return None
        if self._traj_idx >= self._traj_q_curobo.shape[0]:
            return None
        q_curobo = self._traj_q_curobo[self._traj_idx]
        self._traj_idx += 1
        return q_curobo

    def to_mjlab_qpos(self, q_curobo: np.ndarray) -> np.ndarray:
        """Convert curobo_q -> mjlab_qpos for one arm row: ``mjlab_qpos = curobo_q + ref_offsets``.
        Caller is responsible for writing into the entity's qpos using the arm_ref column
        order (current task convention: arm ref = waist + L-arm + R-arm)."""
        return q_curobo + self._ref_offsets

    @property
    def trajectory_mjlab_qpos(self) -> np.ndarray:
        """Immutable copy of current route in MuJoCo qpos coordinates, shape ``[H, dof]``.

        Dynamic execution samples this route into ``arm_ref``. Returning a copy prevents a caller from
        mutating the planner buffer or treating it as simulator state.
        """
        assert self._traj_q_curobo is not None, "plan() must succeed before reading its trajectory"
        return (self._traj_q_curobo + self._ref_offsets[None, :]).copy()

    @property
    def ref_offsets(self) -> np.ndarray:
        return self._ref_offsets.copy()

    @property
    def interpolation_dt(self) -> float:
        """Seconds between buffered waypoints. Callers should pace step() at this cadence
        (or slower) to faithfully replay the planned motion."""
        return float(self._planner.trajopt_solver.config.interpolation_dt)

    @property
    def n_remaining(self) -> int:
        if self._traj_q_curobo is None:
            return 0
        return int(self._traj_q_curobo.shape[0] - self._traj_idx)
