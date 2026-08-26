"""Robot-generic cuRobo reach/return planning spine (config + planner + reactive tracker).

Extracted from ``test/curobo_reach_verify.py`` so the SAME planner serves the kinematic
phase-3 verify, later RL dynamics joint control, and a future single-robot real deploy. This
module is the invariant spine: it never imports the MuJoCo viewer.

Layers:
  * **Config (top section)** -- ``RobotDescriptor`` + ``G1_CFG``/``HUMANOID_CFG`` + ``CFG_BY_ROBOT``.
    Declarative robot identity only (URDF cfg, home, tool frames, shoulder_z, grasp callables). Points
    INTO the existing per-robot cfg modules (``ik_curobo_robot_cfg``/``g1_curobo_robot_cfg``); those
    large collision/YAML builders stay split by robot.
  * **Planner** -- ``CuroboPlannerSession`` (warm once, ``update_world`` per episode) + frame-generic
    reach/return primitives. Single-env (`B=1`) impl; goalset tensors carry a leading batch dim so a
    future ``max_batch_size=N`` solver is an impl swap, not a re-architecture.
  * **Reactive tracker** -- ``ReactiveLocalIK`` (batched ``BatchedMinkIK`` servo; see below). Follows a
    session-precomputed Cartesian route under live base drift; the deploy/RL per-tick path.

Frame convention (load-bearing): cube poses are WORLD-frame; ONE explicit world->base transform
produces base-frame targets (in ``scene.py``); reach primitives here consume base-frame targets. The
tool-frame order ``tool_frames[i]`` is the single source of truth for the goalset link order.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from curobo._src.geom.types import SceneCfg  # noqa: E402
from curobo._src.robot.kinematics.kinematics import Kinematics  # noqa: E402
from curobo._src.state.state_joint import JointState  # noqa: E402
from curobo._src.types.robot import RobotCfg  # noqa: E402
from curobo._src.types.tool_pose import GoalToolPose  # noqa: E402

from tasks.visual_manipulation.curobo.networkx_compat import (  # noqa: E402
    install_networkx_edge_buffer_compat,
)

# Install before ``CuroboPlannerSession`` can allocate the PRM planner. The shim
# is process-local, idempotent, and keeps third-party cuRobo source unmodified.
install_networkx_edge_buffer_compat()

from asset_zoo.g1.g1_constants import KNEES_BENT_KEYFRAME, get_g1_robot_cfg  # noqa: E402
from asset_zoo.humanoid_v21.humanoid_v21_constants import (  # noqa: E402
    HOME_KEYFRAME,
    LEFT_ARM_JOINT_NAMES as HUM_LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES as HUM_RIGHT_ARM_JOINT_NAMES,
    get_humanoid_v21_robot_cfg,
)
from mjlab.entity.entity import Entity  # noqa: E402
from mj_envs.utils.ik_mink import BatchedMinkIK  # noqa: E402

from tasks.visual_manipulation.curobo import g1_curobo_robot_cfg as g1cfg  # noqa: E402
from tasks.visual_manipulation.curobo import ik_curobo_robot_cfg as humcfg  # noqa: E402
from tasks.visual_manipulation.curobo.ik_curobo import (  # noqa: E402
    build_curobo_batch_planner,
    build_curobo_motion_planner,
    final_active_q,
)

# ------------------------------------------------------------------------------------------------
# Spine constants
# ------------------------------------------------------------------------------------------------
from asset_zoo.parallel_gripper import PARALLEL_GRIPPER_OPEN_RACK_POS_M  # noqa: E402

PARALLEL_GRIPPER_RACK_HOME = {
    f"{prefix}{rack}_rack_y": PARALLEL_GRIPPER_OPEN_RACK_POS_M
    for prefix in ("L_", "R_")
    for rack in ("left", "right")
}

REACHING_SIDE = "R"
IDLE_SIDE = "L"
TARGET_TOLERANCE = 0.01                            # meters, "<~1 cm" success criterion
# Anchor pair below is the PUBLISHED one: dyn10 (commit c163845) and every mp4 in
# media/two_target_videos_2k/ were measured at (0.01, 0.04). The 2026-08-10 "river" values
# (GRASP_Z_ABOVE_M=0.005, GRASP_Z_FLOOR_BASE=-0.02, matching the deployed lev2_plan_server's deep
# pinch) were tried locally AFTER the paper measurement and reverted here -- they change
# grasp feasibility, so a sweep run under them is not comparable to Table IV. Re-apply the river
# pair only for hardware-facing work, never for a paper re-baseline.
GRASP_Z_FLOOR_BASE = 0.04                          # humanoid center-clamp grasp floor (base frame); below every table cube center (0.0445 base) so the clamp no longer overrides the unified center+GRASP_Z_ABOVE_M anchor -- only nets a cube genuinely below the bench. Was 0.06 (fired on all table cubes, defeating the +Z anchor).
GRASP_Z_ABOVE_M = 0.01 #0.025                            # unified grasp anchor: m above cube center (obj +Z), top face of 50 mm cube. Was 0 -- exact cube-center target IK-infeasible for humanoid (front_back_close: 0% reach, every stance/assignment); +1cm off that boundary fixes it (2026-07-22).
DRIFT_REACH_Z = 0.08                               # drift-demo reach Z (pre-grasp standoff over cube top)
# TWO-STAGE REACH, stage-A retreat (metres, along each grasp candidate's OWN approach axis -- see
# ``cube_grasp_poses_obj(standoff=)``). Stage A plans home -> here; stage B plans here -> the true grasp,
# seeded at the realized config. Both are real cuRobo routes: an earlier `_MPC_PREGRASP_LIFT_M` drove
# stage 2 open-loop through the Jacobian tracker with NO collision check and was removed for it (a6f9840).
# WHY: one route covers the whole 4.4 s EXTEND while the base drifts ~2 cm, against a ~0.015 m/side rack
# corridor, so the fingers clip the cube on approach -- 18% of dyn3 trials nudged it >1 cm, 133 of 176 such
# events during EXTEND. Re-planning the last visit from a FRESH base pose + cube belief collapses the drift
# budget on the only span that touches the cube. 0.05 m = a standard pre-grasp standoff (and the value the
# removed constant used); env-overridable so the A/B does not need a per-host edit (the sweep hosts
# share a single working tree).
#
# DEFAULT 0 == OFF, and the reason is measured, not cautionary. A/B on v2/left_right_close/seed 42: OFF took
# 0 knocks, ON took one at t=3.46 s -- during STAGE A, whose goal is already 5 cm short of the cube. The
# standoff cannot help because the binding error is not along the approach, it is the tracker's own
# ``physical_pos_err``: 11 mm median / 26 mm max over EXTEND, past the ~0.015 m/side rack corridor on 40 of
# 236 ticks. Retreating the goalset also moves the winning IK branch -- stage A's max joint travel is
# 3.685 rad against the single-shot's 1.828 rad for an EASIER target -- so the arm swings harder and tracks
# worse, which is what put the rack into the cube. Re-enable only once EXTEND tracking is inside the
# corridor; the split itself is sound (stage B measures straightness 0.70-0.98 at 0.0 deg off the approach
# axis) and is kept intact for that retest.
_PREGRASP_STANDOFF_M = float(os.environ.get("REACH_PREGRASP_STANDOFF_M", "0") or 0.0)

# ROADMAP FALLBACK. cuRobo's ``MotionPlanner.plan_pose`` dispatches on goalset width
# (``motion_planner.py:224``): ``num_goalset > 1`` routes to ``_plan_pose_goalset``, which runs IK+trajopt
# ONLY and silently DROPS the ``enable_graph_attempt`` argument. Every reach here targets a goalset (g =
# ``grasp_goalset_size``, 12 humanoid / 8 g1), so the PRM graph planner -- built at
# ``motion_planner.py:69`` and warmed by ``planner.warmup(enable_graph=True)`` -- has never been queried on
# any reach, and this file's own ``enable_graph_attempt=0`` (``ik_curobo.py:308``) is dead.
#
# That matters because trajopt is a LOCAL optimizer: it deforms a straight-line joint-space guess out of
# obstacles, so it solves detours that are deformations of the straight line and cannot solve detours that
# are topologically different. A roadmap search can, which is exactly the failure left on
# ``bimanual_mixed_front_back_close`` -- the arm must fold rearward through the inflated human torso proxy,
# and 6 back-off rungs x 30 attempts all press on the same wall (measured cm4: 28-30 solves, 12.9-14.3 s,
# no route).
#
# DEFAULT OFF pending that measurement; ON must not change the 900-trial Table IV. The budget is the real
# constraint, not the control loop: plan-0 runs in a spawned process (``scene.py``) and never blocks a 50 Hz
# tick, but the policy HOLDS while it flies, so planner wall-clock converts 1:1 into mission ticks.
# ``_GRAPH_FALLBACK_BUDGET_S`` bounds the added time deterministically; past it the caller gets today's
# behaviour.
#
# ``_GRAPH_FALLBACK_RESET`` picks between two wrong-in-different-ways options, to be settled by measurement.
# The roadmap PERSISTS across world hot-swaps (``graph_planner_prm.py:330`` resets only above 75% of
# ``max_nodes`` = 15000), and our world changes every solve as the base moves, so reused nodes/edges were
# validated against earlier scenes. Reuse is fast but stale; reset is honest but pays a rebuild per query.
# Staleness cannot produce a COLLIDING route either way -- the seed is only a starting shape and trajopt
# owns the final trajectory under live collision costs -- so the worst case of reuse is a wasted attempt.
_GRAPH_FALLBACK = bool(os.environ.get("REACH_GRAPH_FALLBACK"))
_GRAPH_FALLBACK_BUDGET_S = float(os.environ.get("REACH_GRAPH_FALLBACK_BUDGET_S", "1.0"))
_GRAPH_FALLBACK_RESET = bool(os.environ.get("REACH_GRAPH_FALLBACK_RESET"))
# Widen the joint-limit clamp from the graph query alone to IK and trajopt too -- see ``_clamped_start``.
# Separate lever because it changes what the SOLVED trajectory starts from, not just what the roadmap is
# asked; 0.5 deg is physically nothing, but "seed the arm where it is not" has bitten this file before.
_GRAPH_FALLBACK_CLAMP_ALL = bool(os.environ.get("REACH_GRAPH_FALLBACK_CLAMP_ALL"))
SHARED_BASE_XY = (0.0, 0.0)                         # all robots start at the world origin
G1_TABLE_TOP_Z_BASE = -0.15                        # g1 gripper z-floor = workbench top in pelvis frame

# Home standing pose, FK-derived from each compiled model with the TRUE home joints (regex-expanded:
# g1's bent-knee legs come from KNEES_BENT_KEYFRAME patterns) and the lowest foot-collision geom planted
# on the floor (z=0). The bare keyframe pos[2] left the feet off-floor -- g1's dropped-leg bug sank them
# 32 mm, the humanoid floated 16 mm -- so pelvis z is corrected to plant them. Because pelvis, shoulder,
# table (table_z_for), and cube (table_top + CUBE_HALF) are all shoulder-relative, this shift moves them
# TOGETHER: pelvis->cube and camera->cube geometry is unchanged (grasps/FOV identical), only the feet
# meet the floor. Repro: forward the home pose, then pelvis_z -= min collidable-geom world z; read the
# shoulder-joint world z at that planted pelvis. A per-robot property consumed by scene placement.
_G1_PELVIS_Z = 0.757081
_HUM_PELVIS_Z = 0.573707
_G1_SHOULDER_Z = 1.048861
_HUM_SHOULDER_Z = 0.921207

# Humanoid grasp-approach tilt (deg above straight-down; 90=level side grasp, 60=30deg below horizontal,
# 45=45deg below horizontal). Union (60,45): union (90,60) was measured 0% on front_back_close on
# 2026-07-22, later traced to GRASP_Z_ABOVE_M=0 (exact cube-center target, IK-infeasible boundary; see
# above), not a goalset-union defect. Three betas x four yaws = 12; planner capacity is derived from
# this descriptor's emitted candidate count (flip remains disabled).
HUMANOID_GRASP_BETAS = (75, 60.0, 45.0)  # 2026-07-22: union(90,60) failed front_back_close; union(60,45) recovered it. 30deg added for far-lateral / high-handoff cubes (tilted through the cube). 75deg added for near-level side grasp (recovered a few far-lateral cubes that 60deg missed). 4 betas x 4 yaws = 16 candidates, still < _MAX_GOALSET=32.


# ------------------------------------------------------------------------------------------------
# Config: robot descriptors (declarative, one place)
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotDescriptor:
    """Robot IDENTITY only: base pose, tool frames, arm plumbing, gripper-approach rule, planner
    kwargs, shoulder height, and the robot cfg module's cube-grasp callables. Enough that
    bimanual/single/drift are robot-generic (no per-robot branch in the shared code)."""
    name: str
    base_link: str
    home_base_pos: tuple
    home_base_quat: tuple
    tool_frames: tuple                 # (L, R) -- ORDER is load-bearing (see __post_init__)
    ee_frame_type: str                 # "site" (mink)
    arm_joints_by_side: dict           # {"L":[...], "R":[...]}
    home_joint_pos: dict               # reset/replay qpos map (Mink warm-start seed)
    planning_home_joint_pos: dict      # exact Phase-3/Phase-4 planned-route arm qpos map
    reaching_side: str                 # single-arm / drift reaching hand
    target_mode: str                   # "center" | "center_clamp"
    grasp_z_floor: float               # center_clamp only
    shoulder_z: float                  # shoulder-joint world z at home (scene placement input)
    planner_kwargs: dict
    build_robot_cfg_dict: Callable[[dict], dict]
    build_entity_spec: Callable[[], object]
    cube_grasp_poses_obj: Callable
    grasp_poses_to_base: Callable
    head_cam_sites: tuple               # (left, right) head-camera site names for the visibility gate
    walk_visibility_dirs: tuple         # fixed-camera target bearings trackable while moving; () = gimbal tracks all
    walk_drive_dirs: tuple              # body-frame target directions the locomotion checkpoint may translate toward
    body_clearance_points_b: tuple      # conservative root-frame XY footprint corners for support-face clearance
    walk_min_standoff_m: float          # nearest normal approach stance for this body's arm/workspace geometry
    walk_support_clearance_m: float | None = None  # root-to-face override; None uses projected body footprint
    reach_max_m: float | None = None    # in-place commit ceiling override; None uses the shared _REACH_MAX_M
    walk_precommit_contour_align: bool = False  # align Point-C tangent at safe support clearance before first reach
    walk_stance_dirs: tuple = ()        # visible body-frame target bearings worth retrying after a no-route stance
    walk_side_stance_adjust: bool = False  # recovery can command a bounded raw sideways stance shift
    walk_cruise_vy_max: float | None = None  # per-robot override of the mover's lateral cross-track cap (m/s);
    #   None = HeuristicMovingPolicy.CRUISE_VY_MAX = 0.6, i.e. strafe IS enabled by default (only g1 overrides, to 0.3)
    walk_turn_cruise: bool = False      # skip turn-in-place: enter WALK on tick 1 and arc into the facing at
    #   cruise, holding the yaw floor until faced (see MoverCfg.turn_cruise for the measured premise). Safe
    #   only for a robot with drive directions on BOTH sides of its heading -- with a single forward
    #   direction, a rear target means walking away for the whole 180 deg.
    walk_arm_joint_pos: dict[str, float] | None = None  # one arm pose held while base walks
    dynamic_tracker_filter: bool = False  # damp raw Cartesian arm references before physics position servo
    dynamic_tracker_omega: float = 15.0  # filter natural frequency; only read when dynamic_tracker_filter is true
    dynamic_tracker_max_correction_rad: float | None = None  # measured-joint cap against singular raw-IK branch
    dynamic_tracker_transit_max_correction_rad: float | None = None  # nominal-route tube before final grasp
    dynamic_tracker_resid_ki: float = 0.0  # joint-space integral on the measured physical tool error; 0 = off
    dynamic_extend_settle_s: float = 0.4  # extra final-target hold for physical tracker convergence
    dynamic_gravity_comp_alpha: float = 1.0  # PD-droop feedforward scale for this embodiment's arm/base coupling
    gaze_cams: tuple = ()               # actuated gimbals: (site, yaw_joint, pitch_joint, aim_sign) each; () = fixed cams
    extend_max_retries: int | None = None  # per-robot override of reach_policy._EXTEND_MAX_RETRIES; None = shared default
    needs_reface_override: bool | None = None  # force ReachPolicy._needs_reface; None = computed (len(walk_visibility_dirs)==1)

    def __post_init__(self):
        # Tool-order invariant (single source of truth): index i <-> tool_frames[i] <-> sorted
        # cube-key i <-> goalset link i. A list/wrong length silently routes each hand to the wrong
        # object; assert once here instead of re-checking in four files.
        assert isinstance(self.tool_frames, tuple) and len(self.tool_frames) == 2, (
            f"tool_frames must be a 2-tuple (L, R); got {self.tool_frames!r}"
        )
        assert self.walk_drive_dirs, "walk_drive_dirs must contain at least one nonzero direction"
        assert all(np.hypot(dx, dy) > 0.0 for dx, dy in self.walk_drive_dirs), (
            f"walk_drive_dirs contains a zero direction: {self.walk_drive_dirs!r}"
        )
        assert self.body_clearance_points_b and all(len(p) == 2 for p in self.body_clearance_points_b), (
            f"body_clearance_points_b must be nonempty XY points: {self.body_clearance_points_b!r}"
        )

    @property
    def grasp_goalset_size(self) -> int:
        """Exact candidate capacity required by this robot's configured grasp generator.

        Goalset shape is part of cuRobo's warmed CUDA graph. Derive it from the descriptor instead of
        duplicating ``4 * len(beta_degs) * flip`` beside every capability configuration; a robot can add
        or remove approach orientations without silently exceeding a global cap.
        """
        positions, _ = self.cube_grasp_poses_obj(device="cpu")
        return int(positions.shape[0])

    @property
    def reach_idx(self) -> int:
        return {"L": 0, "R": 1}[self.reaching_side]

    @property
    def reach_frame(self) -> str:
        return self.tool_frames[self.reach_idx]

    @property
    def idle_frame(self) -> str:
        return self.tool_frames[1 - self.reach_idx]


G1_CFG = RobotDescriptor(
    name="g1",
    base_link=g1cfg.BASE_LINK,
    home_base_pos=(*SHARED_BASE_XY, _G1_PELVIS_Z),
    home_base_quat=(1.0, 0.0, 0.0, 0.0),
    tool_frames=tuple(g1cfg.GRASP_TOOL_FRAMES),
    ee_frame_type="site",
    arm_joints_by_side={"L": list(g1cfg.LEFT_ARM_JOINT_NAMES), "R": list(g1cfg.RIGHT_ARM_JOINT_NAMES)},
    home_joint_pos={**KNEES_BENT_KEYFRAME.joint_pos, **g1cfg.G1_ARM_HOME, **PARALLEL_GRIPPER_RACK_HOME},
    planning_home_joint_pos={**g1cfg.G1_ARM_HOME, **PARALLEL_GRIPPER_RACK_HOME},
    reaching_side="R",
    # G1's grasp site is defined at the cube center. The tilted grasp generator supplies its
    # collision-safe pre-grasp standoff; targeting the top face instead visibly misses the cube.
    target_mode="center",
    grasp_z_floor=0.0,
    shoulder_z=_G1_SHOULDER_Z,
    planner_kwargs={
        "hand_z_floor": G1_TABLE_TOP_Z_BASE,
        "max_attempts": 2,
    },
    build_robot_cfg_dict=lambda arm_joint_home=g1cfg.G1_ARM_HOME: g1cfg.build_robot_cfg_dict(
        arm_joint_home={**arm_joint_home, **PARALLEL_GRIPPER_RACK_HOME}),
    build_entity_spec=lambda: Entity(
        get_g1_robot_cfg(head_camera="builtin", end_effector="actuated", hand="parallel_gripper")
    ).spec,
    cube_grasp_poses_obj=functools.partial(
        g1cfg.cube_grasp_poses_obj, beta_degs=g1cfg.G1_GRASP_BETAS, flip=False, z_above=GRASP_Z_ABOVE_M),
    grasp_poses_to_base=g1cfg.grasp_poses_to_base,
    head_cam_sites=("head_camera_rgb",),               # single FIXED head camera (builtin)
    walk_visibility_dirs=((1.0, 0.0),),                 # single fixed forward camera
    walk_drive_dirs=((1.0, 0.0),),                      # target-directed forward walk; recovery may reverse
    body_clearance_points_b=((0.18, 0.18), (0.18, -0.18), (-0.18, 0.18), (-0.18, -0.18)),
    # Same 5 cm root-to-face clearance HUMANOID_CFG uses, and for the same reason. Without an override the
    # support brake takes the projected 0.18 m square footprint, which is 0.204--0.218 m along g1's own
    # diagonal stance headings (``walk_stance_dirs``, +-60 deg) and 0.2546 m worst case, so
    # ``stop_clearance = extent + 0.05 + braking`` held the SHORTEST-arm robot 0.29--0.33 m off the table
    # FACE -- measured 0.32--0.64 m from the cube itself at commit, i.e. at or past its ~0.42 m envelope,
    # which is exactly the observed `physical grasp latch did not engage` / `unreachable after 2 back-off
    # retries` pair on the far and handoff cells. The brake also dominated ``minimum_stop`` on every
    # commit, so the back-off retries only pushed the stance further out. Geometrically 5 cm is right for
    # g1 too: the bench collision is a top plate plus inset leg capsules, and the frontmost g1 collision
    # geometry that dips below the plate is the toe at 0.142 m ahead of the pelvis (v2-fixed's knee is
    # 0.138 m, i.e. the same body-vs-plate case), so a toe slipping under the plate cannot strike it.
    # Unlike V2 this robot only drives forward (``walk_drive_dirs``), so it takes v2-fixed's 5 cm rather
    # than V2's lateral-drive 10 cm. MuJoCo still rejects any real body/table contact.
    walk_support_clearance_m=0.05,
    walk_min_standoff_m=0.24,
    # In-place commit ceiling (``reach_max_m``): DELETED 2026-08-18 (was 0.32). That value existed to stop ACQUIRE committing in place at
    # near-full arm extension (~0.40-0.42 m), where cuRobo solved but the physical latch missed -- measured
    # then as P 0.844 (default 0.50) vs 0.978 (0.32) on bimanual_mixed_close. Re-tested against the current
    # tree (900-trial full sweep): 0.32 was now COSTING P and speed for no return -- full-default 180-trial
    # G1-only re-sweep gave P 177/180=0.983 (vs 172/180=0.956 at 0.32), Tbar 28.03s (vs 29.92s), ZERO
    # `physical grasp latch did not engage` failures (the 3 failures were all `unreachable after 2 back-off
    # retries`, a different mechanism). The latch-reliability problem 0.32 was tuned against is gone --
    # most likely fixed by the 2026-08-10 GRASP_Z_ABOVE_M correction, which postdates this tuning and
    # directly targets grasp-latch reliability. Reverting to the shared default un-does a workaround for a
    # since-fixed bug rather than reintroducing the bug itself; re-verify against a fresh sweep before
    # touching this again if grasp-latch failures reappear on G1.
    walk_stance_dirs=((0.5, -0.8660254), (0.5, 0.8660254)),  # mirrored arm-workspace headings, inside forward camera cone
    walk_side_stance_adjust=True,
    # Per-robot strafe cap. Headless rollouts / paper benchmarks: 0.3 m/s gives a bounded lateral step
    # toward an off-axis cube without committing the gait to full freedom. Cap is half humanoid's 0.6
    # because g1's fixed forward camera cannot compensate for sustained lateral body sway as cleanly as
    # the gimbals do. Interacts with the anti-sway limits: any widening needs a fresh low-command sweep.
    walk_cruise_vy_max=0.3,
    dynamic_tracker_filter=False,
    dynamic_extend_settle_s=0.5,
)

HUMANOID_CFG = RobotDescriptor(
    name="v2_fixed",
    base_link=humcfg.BASE_LINK,
    home_base_pos=(*SHARED_BASE_XY, _HUM_PELVIS_Z),
    home_base_quat=(1.0, 0.0, 0.0, 0.0),
    tool_frames=("end_effector_L_site", "end_effector_R_site"),
    ee_frame_type="site",
    arm_joints_by_side={"L": list(HUM_LEFT_ARM_JOINT_NAMES), "R": list(HUM_RIGHT_ARM_JOINT_NAMES)},
    home_joint_pos={**HOME_KEYFRAME.joint_pos, **PARALLEL_GRIPPER_RACK_HOME},
    planning_home_joint_pos={**humcfg.HUMANOID_ARM_JOINT_HOME, **PARALLEL_GRIPPER_RACK_HOME},
    reaching_side="R",
    target_mode="center_clamp",
    grasp_z_floor=GRASP_Z_FLOOR_BASE,
    shoulder_z=_HUM_SHOULDER_Z,
    planner_kwargs={},
    build_robot_cfg_dict=lambda arm_joint_home=humcfg.HUMANOID_ARM_JOINT_HOME: humcfg.build_robot_cfg_dict_from_urdf(
        arm_joint_home={**arm_joint_home, **PARALLEL_GRIPPER_RACK_HOME}),
    build_entity_spec=lambda: Entity(
        # Fixed twin D435 head (welded at the exact G1 D435-mount pitch): the visibility gate reads
        # these sites' live frusta; no gaze slew. cuRobo collision_spheres includes the fixed head-camera links -- keep matching
        # MuJoCo bodies so planner-view spheres parent to their actual links. (Actuated is a variant.)
        get_humanoid_v21_robot_cfg(head_camera="welded", end_effector="actuated", hand="parallel_gripper")
    ).spec,
    # Grasp-orientation GOALSET: offer a level AND a near-level tilted approach per cube face and let
    # cuRobo pick whichever is kinematically feasible. Dead-level (beta=90) alone missed jitter-tail
    # cubes (far-lateral / high-handoff) that a downward tilt threads -- the humanoid then scored below
    # the g1 baseline. Union {90,60} recovers those without the single global tilt's opposite-corner
    # regression. flip=True doubles each tilt/yaw to its top/down twin (parallel gripper is symmetric
    # under a 180deg roll about the tool approach axis, so both twins are equally valid grasps) --
    # widens the candidate set for back-reach/high-tilt cases where one twin is near-singular; total
    # stays under _MAX_GOALSET (12*2=24 of 32).
    cube_grasp_poses_obj=functools.partial(
        humcfg.cube_grasp_poses_obj, beta_degs=HUMANOID_GRASP_BETAS, flip=False, z_above=GRASP_Z_ABOVE_M),
    grasp_poses_to_base=humcfg.grasp_poses_to_base,
    head_cam_sites=("cam_left_rgb", "cam_right_rgb"),   # fixed twin D435 (welded)
    walk_visibility_dirs=((1.0, 0.0), (-1.0, 0.0)),     # fixed front+rear camera pair
    walk_drive_dirs=((1.0, 0.0), (-1.0, 0.0)),          # target-directed forward/backward walk
    body_clearance_points_b=((0.13, 0.20), (0.13, -0.20), (-0.13, 0.20), (-0.13, -0.20)),
    # The old projected side-width proxy held this robot 0.20 m from the table face before its
    # 0.15 m Point-C depth offset, leaving the hand ~0.39 m from Point C. Keep a 5 cm root-to-face
    # clearance instead; MuJoCo still rejects any actual body/table collision.
    walk_support_clearance_m=0.05,
    walk_min_standoff_m=0.24,
    walk_precommit_contour_align=True,
    walk_side_stance_adjust=True,
    dynamic_tracker_filter=False,
    dynamic_tracker_omega=30.0,
    # Same nominal-route margin cap as V2: unconstrained repeated DLS corrections on the floating base can
    # leave a collision-checked bimanual route by ~1.9 rad and drive into a near-singular branch.  This is
    # a tracker safety bound, not a reachability or latch-score change.
    dynamic_tracker_max_correction_rad=0.25,
    dynamic_tracker_transit_max_correction_rad=0.06,
    # Joint-space integral on the measured tool error, through the cap-exempt pre-bend channel. Screened
    # 0.0 vs 0.02 at 180 trials/arm on `v2` (`~/tmp/ki00` vs `~/tmp/ki002`): nudge rate 0.233 -> 0.167,
    # P 0.961 -> 0.972, Tbar 15.23 -> 14.54 s -- better or flat on every column. Set HERE and not on the
    # class default because the SAME value regresses g1 hard (nudge 0.011 -> 0.117, p=4e-5, and 2 new
    # `cube knocked off support` failures where it had none): the clip budget is
    # `arm_bend_max - |grav_bend|` = forcerange/kp, so an identical gain buys a different physical
    # overshoot on a different arm. Inherited by v2 / v2_single / v2_single_fixed, which share this arm.
    dynamic_tracker_resid_ki=0.02,
    # Gripper closure starts after the tracker reaches its final waypoint. Use one shared 0.5 s
    # dwell across embodiments: enough contact-settle margin without a visible idle pause.
    dynamic_extend_settle_s=0.5,
)

# Actuated head variant: identical reach to welded (cuRobo cfg is URDF/`build_robot_cfg_dict`-derived and
# UNCHANGED; the resolved model already locks the 4 cam joints at qpos0), only the SCENE spec swaps to the
# actuated head so its `cam_yaw_*`/`cam_pitch_*` DOFs exist for the gimbal gaze gate + FOV rendering.
HUMANOID_ACTUATED_CFG = dataclasses.replace(
    HUMANOID_CFG,
    name="v2",
    build_entity_spec=lambda: Entity(
        get_humanoid_v21_robot_cfg(head_camera="actuated", end_effector="actuated", hand="parallel_gripper")
    ).spec,
    gaze_cams=(
        ("cam_left_rgb", "cam_yaw_left", "cam_pitch_left", 1.0),
        ("cam_right_rgb", "cam_yaw_right", "cam_pitch_right", -1.0),
    ),
    walk_visibility_dirs=(),                            # actuated gimbals track target at any body bearing
    # walk_drive_dirs=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
    walk_drive_dirs=((1.0, 0.0), (-1.0, 0.0)),
    # With only +-X to latch, a lateral target costs a ~90 deg turn -- and this checkpoint's turn-in-place
    # authority decays to ~20% of command past ~60 deg of accumulated yaw. Arc into the facing at cruise
    # instead of pivoting; +-X means a rear target drives backward, so cruise never heads away from it.
    walk_turn_cruise=True,

    # Sideways drive presents V2's longer lateral body envelope to the support/cube. V2-fixed only
    # drives forward/backward and safely uses the 5 cm root clearance above; 10 cm keeps V2 clear
    # without the 13 cm stance that made far lateral cuRobo routes infeasible.
    walk_support_clearance_m=0.10,
    # Keep V2 raw: actuated-camera body motion is distinct from V2-fixed's front/back sway case.
    dynamic_tracker_filter=False,
    dynamic_tracker_omega=15.0,
    dynamic_tracker_max_correction_rad=0.25,
    dynamic_extend_settle_s=0.5,
    # FULL pre-bend (was 0.5, 2026-08-01). Half compensation leaves the hand 0.5*tau_g/kp below the planned
    # path, which is invisible until something lets the gripper approach an object from below its support:
    # it then closes the jaws on the support instead of the object ("physical grasp latch did not engage").
    # Reproduced on bimanual_mixed_front_back_close seeds 42/43 -- FAIL at 0.5, PASS at 1.0, geometry
    # unchanged. The retired comment justified 0.5 as avoiding floating-base excitation "in close
    # simultaneous reaches"; that case is bimanual_mixed_close (both cubes in ONE bimanual visit) and it does
    # NOT reproduce -- 3/3 seeds PASS at either alpha, with energy equal or better at 1.0 (seed 42: 24.5 J
    # vs 39.0 J) and penetration equal or lower. The feedforward itself is exact under base tilt (identity
    # scratch root + world gravity rotated into base matches a genuinely tilted root to 4e-15 Nm), so alpha
    # was scaling a correct torque, not hiding a model error. Still bounded by the tracker's per-joint
    # actuator-force ceiling (``JacobianReachTracker.arm_bend_max``).
    dynamic_gravity_comp_alpha=1.0,
)

# Single-camera variants: same arm/body/gripper kinematics as the dual-camera humanoid (the
# cuRobo-planning URDF/collision model is UNCHANGED -- reuses humanoid_v21_curobo.urdf; a single head
# has FEWER camera collision spheres than dual, so planning against the dual model is conservative,
# never plans an arm into a collision the real single-cam robot lacks). Only the SCENE spec + camera
# descriptor fields differ, mirroring how HUMANOID_ACTUATED_CFG vs HUMANOID_CFG only swap those.
# extend_max_retries=6 (default 2): single-camera robots hit "unreachable after N back-off retries" far
# more often than dual on bimanual_mixed_front_back_close (narrower simultaneous multi-target coverage
# -> harder stance search). Validated safe + effective 2026-07-31: raising ONLY the deepest radial
# back-off ceiling is a strictly-widening search (cannot change the outcome of any case that already
# resolved within the old ceiling of 2, only rescue some that exhausted it) -- confirmed empirically
# verdict-identical on v2/left_right_close and g1/bimanual_mixed_close at 2 vs 6. Measured effect,
# bimanual_mixed_front_back_close (30 trials/robot): v2_single_fixed 0.833->0.967 (real fix,
# now matches the dual-camera quality bar); v2_single 0.600->0.533 (no net improvement -- its residual
# failures need >6 retries non-monotonically per-seed, or are a deeper limitation of the actuated
# single-gimbal's coverage in this scenario; kept at 6 anyway since it is provably non-regressive and
# helps other cells/seeds). Scoped to these two NEW robots only, not the shared reach_policy.py default
# -- v2/v2_fixed/g1's published Table IV numbers used retries=2 and are left untouched.
HUMANOID_SINGLE_CFG = dataclasses.replace(
    HUMANOID_CFG,
    name="v2_single_fixed",
    build_entity_spec=lambda: Entity(
        get_humanoid_v21_robot_cfg(head_camera="welded_single", end_effector="actuated", hand="parallel_gripper")
    ).spec,
    head_cam_sites=("cam_rgb",),                        # single fixed forward D435
    walk_visibility_dirs=((1.0, 0.0),),                  # one fixed forward camera, no rear coverage
    walk_drive_dirs=((1.0, 0.0),),                       # forward-only target-directed walk (no rear cam to back into)
    extend_max_retries=6,
)

HUMANOID_ACTUATED_SINGLE_CFG = dataclasses.replace(
    HUMANOID_ACTUATED_CFG,
    name="v2_single",
    build_entity_spec=lambda: Entity(
        get_humanoid_v21_robot_cfg(head_camera="actuated_single", end_effector="actuated", hand="parallel_gripper")
    ).spec,
    gaze_cams=(
        ("cam_rgb", "cam_yaw", "cam_pitch", 1.0),
    ),
    head_cam_sites=("cam_rgb",),
    # 2026-08-10: lowering this to 2 was TRIED and REJECTED. Premise was sound-looking: each retry grows
    # ``_extra_standoff`` and the back-off branch (reach_policy.py ~2595) physically REVERSES the base by
    # it, so 6 escalating retries walk v2_single backward into the handoff human the bimanual_mixed_*
    # scenarios stand behind it. Locally it looked decisive -- APPROACH 73.5 s -> 7.0 s mean on seeds
    # 46/47/48, missions terminating instead of timing out. At full 180-trial sweep scale it did NOTHING:
    # P 158/180 -> 156/180 (inside the noise floor), and critically the retries=6 CONTROL already showed
    # mean APPROACH 7.3 s with ZERO livelocks (>40 s) and ZERO timeouts across 180 trials. The livelock is
    # therefore NOT a property of this parameter -- it appears only under a local
    # ``GRASP_Z_ABOVE_M=-0.01`` anchor, which makes grasp routes infeasible and drives the retry
    # escalation that produces the back-off oscillation. Kept at 6.
    #
    extend_max_retries=6,
    # 2026-08-02: drop +-Y, restore +X/-X to match v2/v2_fixed/v2_single_fixed. The 2026-07-31
    # validation that picked +X/+-Y pre-supposed a single gimbal couldn't cover a backward walk -- but
    # it now ships the same ``needs_reface_override=True`` (pre-commit square-up) as the rest, so the
    # facing problem it was solving is already handled. Driving a lateral target sideways wastes the
    # +X drive dir for cells where the cube is in front of the robot; +X/-X covers those without the
    # +Y/-Y excursion.
    # 2026-08-10: removing ``needs_reface_override`` was TRIED and had NO measurable effect on the
    # APPROACH livelock (seeds 46/47/48, solo headless: APPROACH 78.96/68.38/73.24 s before vs
    # 65.86/80.94/70.82 s after, all six timing out with ``mission_verdict: None``). Kept as-is.
    walk_drive_dirs=((1.0, 0.0), (-1.0, 0.0)),
    needs_reface_override=True,
    # DEAD LEVER, 2026-08-10: v2_single's EXTEND-phase tool tracking is measurably noisier than every
    # other arm-sharing variant despite IDENTICAL tracker gains (all inherited unchanged from
    # HUMANOID_CFG's resid_ki=0.02) -- knock-witness events (dyn_sweep rescore, 180 trials/variant)
    # were v2_fixed 26, v2 33, v2_single_fixed 32, v2_single 124 (4x outlier, 107/124 in EXTEND), and
    # the codebase's own predictive signal for this (``[handoff]`` tool_err at EXTEND->GRASP, see
    # ``curobo_reach_verify.py``) ran 0.04-0.10 m for v2_single vs v2's tight 0.01-0.04 m band on the
    # same scenario. This diagnosis is solid and reproduced across 3 independent 180-trial sweeps this
    # session -- v2_single sits ~93% P vs siblings' ~98%, consistently, sibling-relative within each run.
    # Tried raising resid_ki 0.02->0.04 (same joint-space integral already screened for v2 above) to
    # correct it. Single-seed screen looked decisive: both known bimanual_mixed_front_back_close
    # failures (seeds 46, 49) flipped FAIL->PASS, handoff tool_err halved, knock cascade gone. A full
    # 180-trial resweep at 0.04 falsified it: bimanual_mixed_front_back_close genuinely improved
    # (0.80->0.97) but 4 of the other 5 scenarios got WORSE, including a failure class that did not
    # exist at 0.02 (`unreachable after 6 back-off retries`, 2 new cases) -- the joint-space correction
    # nudges the ARM'S configuration enough to flip cuRobo feasibility for a *later* commit in the
    # route, not just fix the targeted tool-position error. Net: 168/180 vs 169/180, flat-to-worse.
    # Do not re-test a blanket resid_ki bump for v2_single on a single-seed screen alone -- this
    # system's chain-embedded ``--record-trials 10`` RNG state means a scenario's single-seed result is
    # not a reliable predictor of its full-sweep outcome (P swings up to 0.15/180 trials from unseeded
    # DR alone, already documented elsewhere in this file's sweep-protocol notes). A real fix would need
    # to be scoped to the specific commit position that needs it, not applied globally.
    # Tried that too: resid_ki boosted ONLY on a route's LAST commit (reasoned safe because that commit
    # has no later commit's feasibility check left to corrupt). Full 180-trial resweep: 157/180=0.872,
    # the worst result of this whole session, including the TARGETED scenario itself
    # (bimanual_mixed_front_back_close 0.63, worse than doing nothing at 0.80). Not noise: a 12-trial
    # drop is far outside this protocol's noise floor. Whatever the real mechanism is, it is not "corrupts
    # a later commit's feasibility" -- that theory predicted this version would be safe and it was the
    # worst of all three levers tried. Do not retry any resid_ki variant, scoped or blanket, for
    # v2_single without a materially different premise.
)

CFG_BY_ROBOT = {
    "g1": G1_CFG,
    "v2_fixed": HUMANOID_CFG,
    "v2": HUMANOID_ACTUATED_CFG,
    "v2_single_fixed": HUMANOID_SINGLE_CFG,
    "v2_single": HUMANOID_ACTUATED_SINGLE_CFG,
}


# ------------------------------------------------------------------------------------------------
# Planner session
# ------------------------------------------------------------------------------------------------
@dataclass
class CuroboPlannerSession:
    """One robot-local cuRobo allocation: warm once, then only replace world obstacles per episode.

    ``MotionPlanner.update_world`` is mutable, so sessions must never be shared across robot workers
    while ``B=1``. Reach and return are both pose-goal plans, so they share one CUDA graph.
    """
    robot_cfg: RobotDescriptor
    robot_cfg_dict: dict
    kin: Kinematics
    pose_planner: object | None = None

    def update_world(self, scene_dict: dict):
        if self.pose_planner is None:
            planner_kwargs = dict(self.robot_cfg.planner_kwargs)
            planner_kwargs["max_goalset"] = self.robot_cfg.grasp_goalset_size
            self.pose_planner = build_curobo_motion_planner(
                scene_dict, self.robot_cfg_dict, **planner_kwargs)
        else:
            self.pose_planner.update_world(SceneCfg.create(scene_dict))
        return self.pose_planner


# Base-yaw offsets (deg) the stance-search tries in-place around the target bearing, least-turn (0) first.
# WHY these values (measured, not guessed): a cube is arm-reachable only at ~90deg-PERIODIC body headings
# (the arm mounts to the side, so reach happens when the cube falls in a lateral reach-window, NOT when the
# body points AT the cube). Those windows are NARROW (~+-10deg) and sit anywhere relative to the cube
# bearing, so the search must sample finely enough to land inside one. The nearest window is always <=45deg
# off the bearing (windows 90deg apart), and 45deg == the head-cam half-FOV, so a window within reach is
# ALWAYS still seen -- the search is bounded by FOV, not blind. Coverage measured on left_right_close v2_fixed
# x10 seeds (deep cube): +-50/step10 (11 offs)=10/10, +-40/step10 (9)=10/10, +-45/step15 (7)=9/10,
# +-45/step22.5 (5)=8/10. The OLD +-30 missed seed 51 cube_0, whose bearing (51deg) sat 39deg from the
# nearest window (90deg) -- just past +-30. (Earlier "welded-camera limit" hypothesis was WRONG: SEE&REACH
# stances exist at home; the sweep was simply too narrow.) Cube placement was also moved nearer the table
# edge (pickplace_scenarios LATERAL_CUBE_EDGE_INSET_M) for a shallower reach: that widens the windows enough
# that a COARSE 5-offset {0,+-20,+-40} hits 10/10 on left_right_CLOSE -- BUT it regresses left_right_FAR
# v2_fixed to 0/10 ("cube not reachable from any table-safe stance"): the far walk-in stops at a different
# base bearing, so its windows fall between the coarse samples. So +-40 step10 (9 offsets) is the robust
# floor that holds across BOTH close and far walk-in; keep it regardless of placement.
REACH_YAW_OFFSETS_DEG = (0.0,10.0,-10.0,20.0,-20.0,30.0,-30.0,40.0,-40.0)


def make_planner_session(robot_cfg: RobotDescriptor) -> CuroboPlannerSession:
    """Build robot kinematics once. One pose planner warms lazily at first gated reach and survives reset."""
    robot_cfg_dict = robot_cfg.build_robot_cfg_dict(robot_cfg.planning_home_joint_pos)
    return CuroboPlannerSession(
        robot_cfg, robot_cfg_dict, Kinematics(RobotCfg.create(robot_cfg_dict).kinematics))


@dataclass
class CuroboBatchSession:
    """Robot-local ``BatchMotionPlanner`` (``multi_env``) for the stance-search: solve every seen
    candidate stance's single-arm grasp in ONE IK+TrajOpt pass instead of the serial B=1 short-circuit.

    Fixed ``max_batch_size``; warms lazily on the first batched reach and survives reset. Coexists
    in-process with the per-robot ``CuroboPlannerSession`` (independent allocation)."""
    robot_cfg: RobotDescriptor
    robot_cfg_dict: dict
    kin: Kinematics
    max_batch_size: int
    batch_planner: object | None = None

    def _ensure(self, seed_scene: dict):
        if self.batch_planner is None:
            planner_kwargs = _batch_planner_kwargs(self.robot_cfg.planner_kwargs)
            planner_kwargs["max_goalset"] = self.robot_cfg.grasp_goalset_size
            self.batch_planner = build_curobo_batch_planner(
                seed_scene, self.robot_cfg_dict, self.max_batch_size,
                **planner_kwargs)
        return self.batch_planner

    def load_worlds(self, scene_dicts: list[dict]):
        """Load one collision world per batch row (env_idx). Rows beyond ``len(scene_dicts)`` keep the
        previous env's world; they are masked out of the winner pick, so their contents are irrelevant."""
        planner = self._ensure(scene_dicts[0])
        for i, scene_dict in enumerate(scene_dicts):
            planner.scene_collision_checker.load_collision_model(SceneCfg.create(scene_dict), env_idx=i)
        return planner


def _batch_planner_kwargs(planner_kwargs: dict) -> dict:
    """Filter a robot's B=1 ``planner_kwargs`` to the subset ``build_curobo_batch_planner`` accepts
    (drops ``enable_graph_attempt``/``arm_joint_home``: graph seeding is off under ``multi_env`` and the
    cspace is already fixed in ``robot_cfg_dict``)."""
    keep = {"max_attempts", "hand_z_floor", "hand_floor_weight"}
    return {k: v for k, v in planner_kwargs.items() if k in keep}


def batched_stance_feasible(batch_session: CuroboBatchSession, robot_cfg: RobotDescriptor,
                            candidates, marker: str, scene_builder) -> list[bool]:
    """One ``multi_env`` pass over candidate single-arm grasp stances -> boolean feasibility per
    candidate. A FAST PRE-FILTER only: the caller still runs the B=1 ``commit`` on the batch-feasible
    candidates (nearest-first) so the committed stance is always B=1-verified (segment fidelity +
    decision parity unchanged), with a full-serial fallback if none commit. The cuRobo success gate here
    is permissive (any trajopt seed succeeds), so it rarely prunes a stance B=1 would have taken.

    Args:
        candidates: list of ``(stance_cfg, side)`` -- ``stance_cfg`` = a ``dataclasses.replace`` of
            ``robot_cfg`` at the candidate base pose; ``side`` in {"L","R"} = the reaching hand.
        marker: pick-object name; its base-frame grasp target is read from ``scene_builder(stance_cfg)``.
        scene_builder: ``stance_cfg -> (scene_dict, {name: target_base})`` (the harness's
            ``build_bimanual_scene_and_targets`` bound to the scenario + mj_model).
    Returns: ``[bool]`` aligned to ``candidates`` (True = a trajopt seed reached the grasp goalset)."""
    n = len(candidates)
    assert 0 < n <= batch_session.max_batch_size, f"batch {n} exceeds max {batch_session.max_batch_size}"
    scene_dicts, pos_rows, quat_rows = [], [], []
    for stance_cfg, side in candidates:
        side_cfg = dataclasses.replace(stance_cfg, reaching_side=side)
        scene_dict, targets = scene_builder(stance_cfg)
        target_base = dict(targets)[marker]
        scene_dicts.append(scene_dict)
        planner = batch_session._ensure(scene_dict)          # builds on first candidate
        device = planner.default_joint_state.position.device
        home_state = planner.default_joint_state.clone().unsqueeze(0)
        grasp_pos_obj, grasp_quat_obj = side_cfg.cube_grasp_poses_obj(device=device)
        obj_pos = torch.as_tensor(target_base, device=device, dtype=torch.float32)
        obj_quat = torch.as_tensor((1.0, 0.0, 0.0, 0.0), device=device, dtype=torch.float32)
        cand_pos, cand_quat = side_cfg.grasp_poses_to_base(obj_pos, obj_quat, grasp_pos_obj, grasp_quat_obj)
        idle_pose = batch_session.kin.compute_kinematics(home_state).tool_poses.get_link_pose(side_cfg.idle_frame)
        pos_row, quat_row = humcfg.set_reaching_goalset(
            idle_pose.position[0], idle_pose.quaternion[0], cand_pos, cand_quat, side_cfg.reach_idx)
        pos_rows.append(pos_row)
        quat_rows.append(quat_row)
    # multi_env requires the batch to fill EXACTLY num_envs (== max_batch_size) rows: pad the trailing
    # rows by repeating the last candidate (its goal + scene). Padded rows are dropped from the mask.
    b = batch_session.max_batch_size
    scene_dicts = scene_dicts + [scene_dicts[-1]] * (b - n)
    pos_rows = pos_rows + [pos_rows[-1]] * (b - n)
    quat_rows = quat_rows + [quat_rows[-1]] * (b - n)
    planner = batch_session.load_worlds(scene_dicts)
    position = torch.cat(pos_rows, dim=0)                     # [B,1,2,G,3]
    quaternion = torch.cat(quat_rows, dim=0)                  # [B,1,2,G,4]
    start = planner.default_joint_state.position.unsqueeze(0).repeat(b, 1)
    current_state = JointState.from_position(start, joint_names=planner.joint_names)
    goal_pose = GoalToolPose(tool_frames=list(robot_cfg.tool_frames), position=position, quaternion=quaternion)
    result = planner.plan_pose(goal_pose, current_state, max_attempts=planner._plan_max_attempts)
    if result is None:
        return [False] * n
    success = result.success.any(dim=-1)[:n]                 # [N] (padded rows dropped)
    return [bool(s) for s in success.cpu().tolist()]


# ------------------------------------------------------------------------------------------------
# Frame-generic grasp / plan primitives (used for BOTH robots; production ik_curobo untouched)
# ------------------------------------------------------------------------------------------------
def world_axes_in_base_quat(base_quat):
    """World-frame axes expressed in the base_link frame = ``conj(base_quat)`` (wxyz), where ``base_quat``
    is the base orientation in world. This is the cube's orientation in base for a world-axis-aligned cube
    (the AABB grasp-object treatment), so it is the correct ``obj_quat`` for ``grasp_candidates_base``: it
    keeps the grasp goalset WORLD-LEVEL (gravity-aligned) regardless of the floating base's roll/pitch,
    instead of tilting the grasp with the base (the old ``obj_quat=identity`` upright assumption). Upright
    base -> ``(1,0,0,0)`` -> identity, so every non-tilted caller is bit-for-bit unchanged. Same quantity
    as the inline form in ``scene.object_cuboids`` / ``ik_curobo.build_curobo_scene``."""
    w, x, y, z = (float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]))
    return (w, -x, -y, -z)


def grasp_candidates_base(robot_cfg, planner, cube_target_base, obj_quat=(1.0, 0.0, 0.0, 0.0),
                          standoff: float = 0.0):
    """Cube grasp goalset candidates ``(cand_pos[G,3], cand_quat[G,4])`` in base frame for one cube.

    ``standoff`` > 0 retreats every candidate along its OWN approach axis (the two-stage reach's stage-A
    pre-grasp waypoint); 0.0 = the grasp poses themselves. Orientation is identical either way, so a
    stage-A route and the stage-B route that follows it target the same goalset geometry, offset."""
    device = planner.device_cfg.device
    grasp_pos_obj, grasp_quat_obj = robot_cfg.cube_grasp_poses_obj(device=device, standoff=standoff)
    obj_pos = torch.as_tensor(cube_target_base, device=device, dtype=torch.float32)
    obj_quat_t = torch.as_tensor(obj_quat, device=device, dtype=torch.float32)
    return robot_cfg.grasp_poses_to_base(obj_pos, obj_quat_t, grasp_pos_obj, grasp_quat_obj)


def plan_assigned(robot_cfg, kin, planner, assigned, home_state, start_state=None, max_attempts=None,
                  standoff: float = 0.0, graph_fallback: bool = False):
    """Plan <=2 arms in ONE ``plan_pose``: each ACTIVE side's tool frame reaches its cube's grasp
    goalset; each inactive frame is pinned to its home FK pose. ONE solver for single-arm (1 active,
    idle pinned) and bimanual (both active) -- the count is DERIVED from ``assigned``.

    ``assigned`` = ``{side: target_base}`` for the active sides (side in ``{"L","R"}``; 1 or 2 keys).
    ``kin`` is used only to FK an inactive frame's home pin, so a fully-active (bimanual) call may pass
    ``kin=None``. Non-asserting: returns the raw ``plan_pose`` result (``None`` or a result carrying
    ``.success``); the caller decides feasibility (the reachability ladder). Generalizes the former
    ``plan_single`` (idle-pin one link) and ``plan_bimanual`` (both links real).

    cuRobo requires EVERY declared tool frame covered (see ``set_reaching_goalset``), and the goalset
    shares one G axis across links, so an inactive frame's single home pose is expanded to G to match
    the active cube grasp count -- the solver mins per link independently, reaching each active cube's
    nearest candidate while holding each idle link at home.

    ``start_state`` = the trajopt/IK START (warm seed). ``None`` = plan from ``home_state`` (planning
    home). Under a base-live REPLAN it is the CURRENT arm joint state, so the re-solve (a) starts the
    trajectory at where the arm IS -- no home snap -- and (b) seeds IK/graph near the current config, so
    cuRobo stays on the SAME redundancy branch instead of flipping (elbow up<->down) between solves. The
    idle-pin FK + side-assignment stay keyed on ``home_state`` so a seed cannot perturb them.

    ``max_attempts`` = ``None`` uses the planner default (robust, for the cold home->goal plan-0). A base-live
    REPLAN passes ``1``: seeded from the current config the correction is small + easy, so one attempt
    solves and the per-solve cost drops ~30% (the retry loop dominates the wall time).

    ``standoff`` > 0 aims the goalset at the pre-grasp retreat instead of the grasp (two-stage reach
    stage A); see ``grasp_candidates_base``.

    ``graph_fallback`` opts this call into the roadmap retry on failure (``plan_pose_graph_seeded``); it
    still requires ``REACH_GRAPH_FALLBACK``. Off by default so only the caller that owns the time budget --
    the FIRST plan-0 of a cube's trip -- can spend it: the succeeding ~95% of solves never reach this
    branch, which is why enabling the lever cannot perturb the published table."""
    goal_pose = _assigned_goal_pose(robot_cfg, kin, planner, assigned, home_state, standoff=standoff)
    start = _clamped_start(planner, home_state if start_state is None else start_state)
    result = planner.plan_pose(goal_pose, start,
                               max_attempts=planner._plan_max_attempts if max_attempts is None else max_attempts,
                               enable_graph_attempt=planner._plan_enable_graph_attempt)
    if graph_fallback and _GRAPH_FALLBACK and (result is None or not result.success.any()):
        return plan_pose_graph_seeded(planner, goal_pose, start)
    return result


# Clamping to EXACTLY the bound is not enough: trajopt grades feasibility as ``sum_constraint <= 0.0``
# with no tolerance (``metrics.py:285``), and a start pinned on the bound leaves ~6e-10 of residual bound
# cost that fails that test on every seed. Measured: the captured plan-0's ``cspace`` constraint falls
# 6.1 -> 5.68e-10 on an exact clamp and the solve is still 0/3. Margin is 0.006 deg -- below the servo's
# own tracking error, so it moves the trajectory's first waypoint by nothing physical.
_BOUND_MARGIN_RAD = 1e-4


def _clamped_start(planner, start_state):
    """``start_state`` with every joint pulled strictly inside cuRobo's own limits.

    Measured on all 27 captured infeasible plan-0s of ``v2_single`` seed 45: every single one has
    ``right_wrist_3_joint`` 0.00869 rad (0.498 deg) past its limit. The arm is servo-tracked, so the
    realized angle parks fractionally beyond a hard stop that cuRobo's URDF and MuJoCo's XML round
    differently -- physically meaningless, and yet it is fatal twice over:

    * ``graph_planner.find_path`` hard-checks start and goal before sampling
      (``graph_planner_prm.py:337``) and aborts with "Start or End state in collision".
    * trajopt grades ``cspace`` (the joint-bound term) as a hard CONSTRAINT, not the soft cost an earlier
      revision of this docstring assumed. ``success`` requires ``feasible``, ``feasible`` is
      ``sum_constraint <= 0.0`` (``metrics.py:285``), and an out-of-bounds start makes that positive at
      t=0 on EVERY seed of EVERY attempt. Retrying cannot help: the captured plan-0 was 0/3 at 30
      attempts and is 3/3 with this clamp, at 0.091 s median instead of 0.424 s.

    Applied at the single ``plan_assigned`` seam rather than at the ``seed_q`` source: every planning
    caller routes through it, and fixing it there would also rewrite the state the CALLER believes the arm
    is in. The margin keeps the correction below the servo's own tracking error, so the trajectory's first
    waypoint still starts where the arm physically is -- the lie the rear-prime work had to undo was
    radians, not 1e-4.
    """
    lo = planner.graph_planner.action_bound_lows.view(1, -1).to(start_state.position.device)
    hi = planner.graph_planner.action_bound_highs.view(1, -1).to(start_state.position.device)
    clamped = start_state.clone()
    clamped.position = torch.clamp(clamped.position, lo + _BOUND_MARGIN_RAD, hi - _BOUND_MARGIN_RAD)
    return clamped


def _traj_diag(result) -> str:
    """One-line trajopt failure attribution: does the solve MISS the goal, or reach it and get rejected?

    Probe only, printed beside every graph-seeded verdict. ``success`` alone cannot distinguish
    "the seed leads somewhere else" (pose error in cm) from "the pose is hit but the trajectory is
    infeasible" (pose error in mm, ``feasible`` False) -- and those want opposite fixes.
    """
    if result is None:
        return "diag=none"
    def _m(t):
        return "-" if t is None else f"{float(t.min()):.4f}"
    # success = converged AND feasible (solver_trajopt_result.py:199). pos/rot error report the
    # CONVERGED half; this names the constraint terms whose horizon-sum is positive, which is the
    # only thing that can then be false.
    cc = getattr(result.metrics, "costs_and_constraints", None) if result.metrics else None
    if cc is None:
        hot = "metrics_dropped"
    else:
        names = list(cc.constraints.names) + list(cc.hybrid_costs_constraints.names)
        values = list(cc.constraints.values) + list(cc.hybrid_costs_constraints.values)
        hot = ",".join(f"{n}={float(v.sum()):.3g}" for n, v in zip(names, values)
                       if v is not None and float(v.sum()) > 0.0) or f"none/{len(names)}terms"
    return (f"succ={int(result.success.sum())}/{result.success.numel()} "
            f"pos_err={_m(result.position_error)} rot_err={_m(result.rotation_error)} "
            f"violated=[{hot}]")


def plan_pose_graph_seeded(planner, goal_pose, start_state, max_attempts: int = 2):
    """Goalset ``plan_pose`` WITH roadmap seeding -- the branch cuRobo omits (see ``_GRAPH_FALLBACK``).

    Reimplemented here rather than patched into cuRobo because the cuRobo source checkout is a pristine
    upstream checkout: an edit there is invisible to this repo's git and dies on reinstall. Mirrors
    ``MotionPlanner._plan_pose_single`` (``motion_planner.py:233``) but keeps the GOALSET as the goal, which
    costs nothing -- ``_get_graph_seed_trajectories`` is pure joint-space (start config -> goal configs) and
    never sees the goal poses, so goalset width was never what excluded it.

    Three defects in ``_plan_pose_goalset`` are fixed in passing, all upstream's:
      * it ``return``s on an IK miss instead of retrying, so ``max_attempts`` is dead after attempt 0;
      * it never repairs failed IK slots, so junk configs go to trajopt as seeds;
      * ``_plan_pose_single``'s repair (``:264``) writes into ``seed_config[mask][:, :]`` -- boolean-mask
        indexing returns a COPY, so that assignment is a no-op there too. ``seed_config[mask] = x`` is the
        form that actually writes.

    Returns the trajopt result, or ``None`` when no route was found. ``None`` with ``graph_none`` logged is
    the load-bearing DIAGNOSTIC: IK already proved the goal config reachable and collision-free (the IK
    solver carries the shared world checker, ``motion_planner.py:63``), so a roadmap that also finds nothing
    means the arm is genuinely walled in at this stance -- step the base sideways, do not search harder or
    back off radially. A route found here means the opposite: it was always reachable and only the local
    optimizer could not see the detour.
    """
    num_seeds = planner.trajopt_solver.config.num_seeds
    if _GRAPH_FALLBACK_CLAMP_ALL:
        start_state = _clamped_start(planner, start_state)
    deadline = time.monotonic() + _GRAPH_FALLBACK_BUDGET_S
    for attempt in range(max_attempts):
        if time.monotonic() > deadline:
            print(f"[graph] budget {_GRAPH_FALLBACK_BUDGET_S:.2f}s exhausted at attempt {attempt}", flush=True)
            return None
        ik_result = planner.ik_solver.solve_pose(goal_pose, return_seeds=num_seeds,
                                                 current_state=start_state)
        if torch.count_nonzero(ik_result.success) == 0:
            print("[graph] ik_none", flush=True)
            continue
        seed_config = ik_result.solution.clone()
        if torch.count_nonzero(ik_result.success) < num_seeds:
            seed_config[~ik_result.success] = seed_config[ik_result.success][0]
        if _GRAPH_FALLBACK_RESET:
            planner.graph_planner.reset_buffer()
        t_graph = time.monotonic()
        seed_traj = planner._get_graph_seed_trajectories(_clamped_start(planner, start_state), seed_config)
        graph_s = time.monotonic() - t_graph
        if seed_traj is None:
            # Split WHICH end the roadmap rejected. ``find_path`` hard-checks start and goal before it
            # samples anything (``graph_planner_prm.py:337``) and bails with "Start or End state in
            # collision", so a 0-node/1 ms ``graph_none`` is NOT "no route exists" -- no search ran. The
            # distinction is the whole diagnostic: IK grades collision as a SOFT cost at 0.07 m activation,
            # so an IK "success" can still be a configuration the binary checker rejects, and this is the
            # only place the two disagree in the open.
            dof = planner.trajopt_solver.action_dim
            feasible = planner.graph_planner.check_samples_feasibility(
                torch.cat([_clamped_start(planner, start_state).position.view(-1, dof)[:1],
                           seed_config.view(-1, dof)], dim=0))
            n_goals = feasible.shape[0] - 1
            print(f"[graph] graph_none {graph_s:.3f}s nodes={planner.graph_planner.n_nodes} "
                  f"start_ok={bool(feasible[0])} "
                  f"goals_ok={int(feasible[1:].sum())}/{n_goals}", flush=True)
            return None
        t_traj = time.monotonic()
        result = planner.trajopt_solver.solve_pose(
            goal_pose, start_state, seed_config=seed_config, seed_traj=seed_traj,
            use_implicit_goal=True, finetune_attempts=3, finetune_dt_scale=0.75)
        ok = result is not None and bool(result.success.any())
        print(f"[graph] {'solved' if ok else 'trajopt_fail'} graph={graph_s:.3f}s "
              f"paths={seed_traj.shape[1]}/{num_seeds} trajopt={time.monotonic() - t_traj:.3f}s "
              f"nodes={planner.graph_planner.n_nodes} {_traj_diag(result)}", flush=True)
        if ok:
            return result
    return None


def _assigned_goal_pose(robot_cfg, kin, planner, assigned, home_state, standoff: float = 0.0) -> GoalToolPose:
    """Build the ``GoalToolPose`` (active sides' grasp goalset + inactive sides pinned to their home FK
    pose) that ``plan_assigned`` feeds to ``plan_pose`` -- extracted so ``ik_feasible_assigned`` can reuse
    the identical goal for a cheap IK-only pre-filter. Same ``assigned``/``kin``/``home_state`` contract
    as ``plan_assigned``."""
    assert assigned and set(assigned) <= {"L", "R"}, f"assigned sides must be a nonempty subset of L/R; got {assigned}"
    obj_quat = world_axes_in_base_quat(robot_cfg.home_base_quat)   # world-level grasp at the live (tilted) base
    cands = {side: grasp_candidates_base(robot_cfg, planner, target, obj_quat=obj_quat, standoff=standoff)
             for side, target in assigned.items()}
    g = next(iter(cands.values()))[0].shape[0]
    pos_links, quat_links = [], []
    for i, side in enumerate(("L", "R")):
        frame = robot_cfg.tool_frames[i]
        if side in cands:
            cp, cq = cands[side]
        else:
            idle = kin.compute_kinematics(home_state).tool_poses.get_link_pose(frame)
            cp = idle.position[0].view(1, 3).expand(g, 3)
            cq = idle.quaternion[0].view(1, 4).expand(g, 4)
        pos_links.append(cp)
        quat_links.append(cq)
    position = torch.stack(pos_links, dim=0).view(1, 1, 2, g, 3)
    quaternion = torch.stack(quat_links, dim=0).view(1, 1, 2, g, 4)
    return GoalToolPose(tool_frames=list(robot_cfg.tool_frames), position=position, quaternion=quaternion)


def ik_feasible_assigned(robot_cfg, kin, planner, assigned, home_state, start_state=None,
                         standoff: float = 0.0) -> bool:
    """Cheap COLLISION-BLIND pre-filter for ``plan_assigned``: IK-only (no trajopt -- ``ik_solver.solve_pose``
    never calls trajopt regardless), True iff any seed reaches the same goalset ``plan_assigned`` would
    target. A FAIL here is a hard proof of kinematic infeasibility (IK ignores obstacles, so it can only be
    a superset of what trajopt admits) -- never a false positive short-circuit, only ever skips a doomed
    full IK+trajopt plan. Same collision-blind tradeoff class as the rejected reachability-MLP (MEMORY.md
    2026-07-14), accepted here because the negative is provably hard, not learned.

    MUST run with the optimizer on (``run_optimizer=True``, the default): in this cuRobo build,
    ``run_optimizer=False`` skips the IK solve entirely and only checks whether the raw SEED
    (``home_state``) already satisfies the goal pose -- never true for an arm-at-home vs. a grasp target,
    so it silently reported every stance infeasible (verified: every kinematic walk-reach cube failed
    after 1 back-off retry, 2026-07-22).

    ``standoff`` must match the ``plan_assigned`` call it gates, else the pre-filter screens a different
    goalset than the plan targets and its "provably hard negative" guarantee no longer holds."""
    goal_pose = _assigned_goal_pose(robot_cfg, kin, planner, assigned, home_state, standoff=standoff)
    result = planner.ik_solver.solve_pose(
        goal_pose, current_state=home_state if start_state is None else start_state)
    return bool(result is not None and result.success.any())


def assign_cubes_to_sides(kin, home_state, robot_cfg, targets):
    """Ordered candidate side-assignments for 1 or 2 base-frame cube targets -- cheapest geometric prior
    first (the ``run_bimanual`` prior, extracted). ``targets`` = ``{name: target_base}``. Returns a list
    of ``{side: target_base}`` dicts the caller tries in order under a feasibility ladder.

    1 cube -> reach with the NEARER arm (min initial-tool-site distance). 2 cubes -> both L/R pairings,
    sorted by summed initial-site-to-cube distance (each hand prefers the cube nearer its home site)."""
    items = list(targets.values())
    left_frame, right_frame = robot_cfg.tool_frames
    left_initial = initial_site_base(kin, home_state, left_frame)
    right_initial = initial_site_base(kin, home_state, right_frame)
    if len(items) == 1:
        t = items[0]
        near = "L" if np.linalg.norm(left_initial - t) <= np.linalg.norm(right_initial - t) else "R"
        far = "R" if near == "L" else "L"
        # BOTH sides, nearer-first: the tool-site distance is only a TRY ORDER, not a hard match -- the
        # feasibility ladder falls back to the far arm when the near one cannot plan (no geometric arm<->cube
        # assumption; the planner picks the arm that actually solves).
        return [{near: t}, {far: t}]
    assert len(items) == 2, f"assign_cubes_to_sides supports 1 or 2 cubes; got {len(items)}"
    ta, tb = items

    def cost(left_target, right_target):
        return float(np.linalg.norm(left_initial - left_target) + np.linalg.norm(right_initial - right_target))

    pairings = [({"L": ta, "R": tb}, cost(ta, tb)), ({"L": tb, "R": ta}, cost(tb, ta))]
    pairings.sort(key=lambda p: p[1])
    return [p[0] for p in pairings]


def plan_bimanual(robot_cfg, planner, left_target, right_target, home_state):
    """Thin ``plan_assigned`` wrapper: both hands active (L<-left cube, R<-right cube), no idle pin.
    Non-asserting. Tool frames in ``robot_cfg.tool_frames`` [L, R] order."""
    return plan_assigned(robot_cfg, None, planner, {"L": left_target, "R": right_target}, home_state)


def plan_single(robot_cfg, kin, planner, target, home_state):
    """Thin ``plan_assigned`` wrapper: reach with ``robot_cfg.reaching_side``, idle frame pinned home.
    Asserts success (caller catches for the drift feasibility flag)."""
    result = plan_assigned(robot_cfg, kin, planner, {robot_cfg.reaching_side: target}, home_state)
    assert result is not None and result.success.any(), f"plan_pose failed for target {target}"
    return result


def site_base(kin, joint_names, q: torch.Tensor, frame: str) -> np.ndarray:
    """Forward-kinematics tool-frame position (base frame) at active-arm config ``q``."""
    state = kin.compute_kinematics(JointState.from_position(q.unsqueeze(0), joint_names=list(joint_names)))
    return state.tool_poses.get_link_pose(frame).position[0].cpu().numpy()


def initial_site_base(kin, home_state, frame: str) -> np.ndarray:
    """Tool-frame position at the home pose -- the geometric prior for object->hand assignment."""
    return kin.compute_kinematics(home_state).tool_poses.get_link_pose(frame).position[0].cpu().numpy()


def reach_error(robot_cfg, kin, planner, final_q, target, frame, obj_quat=(1.0, 0.0, 0.0, 0.0),
                standoff: float = 0.0) -> float:
    """Distance from the reached tool-frame position to the NEAREST grasp candidate (subsumes the old
    single-point metric; a 1-candidate goalset reduces to it). ``obj_quat`` must match the value used to
    BUILD the goalset (``world_axes_in_base_quat(base_quat)`` at a tilted base): it rotates the ``z_above``
    grasp offset, so the candidate POSITION shifts with it -- passing identity here at a tilted base would
    measure error against a different candidate set than was solved. ``standoff`` carries the same
    obligation for the two-stage reach: it must match the value the route was SOLVED with, else a stage-A
    pre-grasp route is graded against the grasp candidates and reports a spurious ~standoff error."""
    reached = site_base(kin, planner.joint_names, final_q, frame)
    cand_pos = grasp_candidates_base(robot_cfg, planner, target, obj_quat=obj_quat,
                                     standoff=standoff)[0].cpu().numpy()
    return float(np.linalg.norm(cand_pos - reached[None, :], axis=1).min())


# ------------------------------------------------------------------------------------------------
# Reach modes (bimanual / single-arm)
# ------------------------------------------------------------------------------------------------
def run_bimanual(planner, kin, objects, robot_cfg, home_state):
    """Simultaneous bimanual reach on a close 2-object scene. Assign objects to hands by a GEOMETRIC
    prior (each hand to the object nearer its initial tool site), confirmed by a FEASIBILITY search:
    plan the cheaper pairing first, fall back to the other only if it fails. Both-infeasible -> hard
    error. Returns ``({"bimanual": interpolated}, {"left": left_target, "right": right_target})`` --
    the interpolated plan is the full-body JointState (viewer/reach-error name-slice to active joints)."""
    (a_name, a_target), (b_name, b_target) = objects
    left_frame, right_frame = robot_cfg.tool_frames
    left_initial = initial_site_base(kin, home_state, left_frame)
    right_initial = initial_site_base(kin, home_state, right_frame)

    def cost(left_target, right_target):
        return float(np.linalg.norm(left_initial - left_target) + np.linalg.norm(right_initial - right_target))

    pairings = [
        (a_name, a_target, b_name, b_target, cost(a_target, b_target)),
        (b_name, b_target, a_name, a_target, cost(b_target, a_target)),
    ]
    pairings.sort(key=lambda p: p[4])

    chosen = None
    for left_name, left_target, right_name, right_target, pairing_cost in pairings:
        result = plan_bimanual(robot_cfg, planner, left_target, right_target, home_state)
        ok = result is not None and result.success.any()
        print(f"{'OK' if ok else 'FAIL'}: pairing L<-{left_name} R<-{right_name} "
              f"(sum initial site-to-object distance {pairing_cost:.3f} m)")
        if ok:
            chosen = (left_name, left_target, right_name, right_target, result)
            break
    assert chosen is not None, (
        f"bimanual plan_pose infeasible for BOTH pairings of {a_name!r}/{b_name!r} "
        f"(scene may be kinematically unreachable bimanually)"
    )
    left_name, left_target, right_name, right_target, result = chosen

    interpolated = result.get_interpolated_plan()
    n_waypoints = interpolated.position.shape[-2]
    final_q = final_active_q(interpolated, planner.joint_names)
    for frame, name, target in ((left_frame, left_name, left_target), (right_frame, right_name, right_target)):
        err = reach_error(robot_cfg, kin, planner, final_q, target, frame)
        assert err < TARGET_TOLERANCE, (
            f"{frame} -> {name}: final tool-site error {err:.4f} m exceeds tolerance {TARGET_TOLERANCE} m"
        )
        print(f"OK: {frame} -> {name} grasp planned, final tool-site error {err:.4f} m")
    print(f"OK: bimanual plan, {n_waypoints} waypoints, "
          f"interpolation_dt={planner.trajopt_solver.config.interpolation_dt}")
    return {"bimanual": interpolated}, {"left": left_target, "right": right_target}


def run_single_arm(planner, kin, targets, robot_cfg, home_state):
    """Front and rear are INDEPENDENT reach tests: each plans from HOME (not chained -- chaining gave
    rear a poor seed and spuriously failed). Returns ``{name: interpolated}`` for the feasible ones."""
    results = {}
    for target_name in ("front", "rear"):
        try:
            result = plan_single(robot_cfg, kin, planner, targets[target_name], home_state)
        except AssertionError as e:
            print(f"SKIP: {target_name} reach failed ({e})")
            continue
        interpolated = result.get_interpolated_plan()
        n_waypoints = interpolated.position.shape[-2]
        final_q = final_active_q(interpolated, planner.joint_names)
        results[target_name] = interpolated
        err = reach_error(robot_cfg, kin, planner, final_q, targets[target_name], robot_cfg.reach_frame)
        assert err < TARGET_TOLERANCE, (
            f"{target_name}: final tool-site error {err:.4f} m exceeds tolerance {TARGET_TOLERANCE} m"
        )
        print(f"OK: {target_name} reach planned, {n_waypoints} waypoints, final tool-site error {err:.4f} m, "
              f"interpolation_dt={planner.trajopt_solver.config.interpolation_dt}")
    return results


def plan_reach(planner, kin, single, target_spec, robot_cfg, home_state):
    """Plan + tolerance-verify. Returns ``(results, replay_targets)``."""
    if single:
        return run_single_arm(planner, kin, target_spec, robot_cfg, home_state), target_spec
    return run_bimanual(planner, kin, target_spec, robot_cfg, home_state)


def plan_return_home(session: CuroboPlannerSession, forward_plan):
    """Plan collision-aware return to both home hand poses using the same pose-goal planner graph."""
    planner = session.pose_planner
    current = JointState.from_position(
        final_active_q(forward_plan, planner.joint_names).unsqueeze(0), joint_names=planner.joint_names)
    home = planner.default_joint_state.clone().unsqueeze(0)
    home_fk = session.kin.compute_kinematics(home)
    position = torch.stack(
        [home_fk.tool_poses.get_link_pose(frame).position[0] for frame in session.robot_cfg.tool_frames]
    ).view(1, 1, len(session.robot_cfg.tool_frames), 1, 3)
    quaternion = torch.stack(
        [home_fk.tool_poses.get_link_pose(frame).quaternion[0] for frame in session.robot_cfg.tool_frames]
    ).view(1, 1, len(session.robot_cfg.tool_frames), 1, 4)
    goal_pose = GoalToolPose(tool_frames=list(session.robot_cfg.tool_frames), position=position, quaternion=quaternion)
    # Home FK is an exact target solution. Seed IK there so G1 does not choose a different,
    # table-blocked branch for the same hand pose, then optimize a pose-goal trajectory from reach.
    ik_result = planner.ik_solver.solve_pose(
        goal_pose, return_seeds=planner.trajopt_solver.config.num_seeds, current_state=home)
    assert ik_result is not None and ik_result.success.any(), "IK failed to recover home hand pose"
    result = planner.trajopt_solver.solve_pose(
        goal_pose, current, seed_config=ik_result.solution, use_implicit_goal=True)
    assert result is not None and result.success.any(), "pose trajectory failed to return hands home"
    interpolated = result.get_interpolated_plan()
    final_q = final_active_q(interpolated, planner.joint_names)
    for frame in session.robot_cfg.tool_frames:
        home_pos = home_fk.tool_poses.get_link_pose(frame).position[0].cpu().numpy()
        error = float(np.linalg.norm(site_base(session.kin, planner.joint_names, final_q, frame) - home_pos))
        assert error < TARGET_TOLERANCE, f"return plan misses {frame} home pose by {error:.4f} m"
    print(f"OK: planned return home, {interpolated.position.shape[-2]} waypoints")
    return interpolated


def plan_gripper_path(planner, kin, result, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """FK the plan's interpolated JOINT waypoints to a per-waypoint tool-site Cartesian path in base
    frame. Returns ``(pos[H,3], quat[H,4])`` (wxyz) -- the route the local IK will track."""
    interpolated = result.get_interpolated_plan()
    name_to_i = {n: k for k, n in enumerate(interpolated.joint_names)}
    active_idx = [name_to_i[n] for n in planner.joint_names]
    qwp = interpolated.position[0, 0][:, active_idx]
    state = kin.compute_kinematics(JointState.from_position(qwp, joint_names=list(planner.joint_names)))
    link = state.tool_poses.get_link_pose(frame)
    return link.position.cpu().numpy().astype(np.float32), link.quaternion.cpu().numpy().astype(np.float32)


# ------------------------------------------------------------------------------------------------
# Drift math (base rigid-motion helpers; frame-generic)
# ------------------------------------------------------------------------------------------------
def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """World-frame free-base quaternion for a yaw-only kinematic phase-3 update."""
    return (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))


def rot_rpy_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Base->world rotation from intrinsic roll(x)/pitch(y)/yaw(z) in degrees (R = Rz Ry Rx)."""
    r, p, y = np.radians([roll, pitch, yaw])
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def drift_base_quat(R: np.ndarray) -> np.ndarray:
    """world-frame base quaternion (wxyz) for base->world rotation R (identity yaw at HOME)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9).astype(np.float64))
    return q


# ------------------------------------------------------------------------------------------------
# Reactive tracker: follow a session-precomputed Cartesian route under live base drift
# ------------------------------------------------------------------------------------------------
class ReactiveLocalIK:
    """Batched local-IK servo (the deploy/RL per-tick path). The SESSION plans a collision-free reach
    ONCE; its joint waypoints are FK'd to a base-frame tool-site Cartesian route (``plan_gripper_path``).
    Each tick this re-expresses the current route waypoint in the LIVE drifted base and tracks it with a
    warm-started 1-step ``BatchedMinkIK`` QP -- so the arm traces the PLANNED path while local IK absorbs
    base sway, no global re-solve.

    Separation of concerns: planning (session) is external; this class is pure state-in/action-out.
    ``num_envs`` is fixed at construction (``BatchedMinkIK`` allocates the batch); one instance per N.
    A tracked frame follows ``paths[frame]``; a frame absent from ``paths`` holds ``hold[frame]`` (idle).
    """

    def __init__(self, mj_model, robot_cfg: RobotDescriptor, paths: dict, hold: dict,
                 num_envs: int = 1, device: str = "cpu"):
        left = list(robot_cfg.arm_joints_by_side["L"])
        right = list(robot_cfg.arm_joints_by_side["R"])
        left_frame, right_frame = robot_cfg.tool_frames
        self._frames = [left_frame, right_frame]
        self._qadr = np.array([mj_model.joint(j).qposadr[0] for j in left + right])
        self._paths = paths
        self._hold = hold
        self._num_envs = num_envs
        self.horizon = next(iter(paths.values()))[0].shape[0]
        self._k = 0
        self.solver = BatchedMinkIK(
            mj_model, None, num_envs, device, left_frame, right_frame, left, right,
            ee_left_type=robot_cfg.ee_frame_type, ee_right_type=robot_cfg.ee_frame_type,
            mink_source="local", mink_position_cost=5.0, mink_posture_weight=0.02,
            max_target_pos_step=float("inf"), max_target_ori_step_rad=float("inf"),
            use_yaw_frame=False, debug=False,
        )

    @property
    def qadr(self) -> np.ndarray:
        """qpos indices this tracker writes (arm joints, L then R)."""
        return self._qadr

    def reset(self, qpos0: np.ndarray) -> None:
        """Seed the solver's internal state from a full qpos row and rewind the route cursor."""
        self.solver.reset(torch.arange(self._num_envs),
                           torch.from_numpy(np.ascontiguousarray(qpos0)).float().unsqueeze(0))
        self._k = 0

    def step(self, qpos_full: np.ndarray, R: np.ndarray, t: np.ndarray, num_iters: int = 5) -> np.ndarray:
        """One servo tick. ``qpos_full`` = current robot config (base already drifted by ``(R,t)``);
        ``(R,t)`` = base rigid drift relative to the plan-time nominal base. Returns arm qpos for
        ``self.qadr``. Route cursor advances each call, holding the final waypoint once drained."""
        cursor = min(self._k, self.horizon - 1)
        q_Rt = drift_base_quat(R.T)
        pos = torch.zeros(1, 2, 3)
        quat = torch.zeros(1, 2, 4)
        for i, frame in enumerate(self._frames):
            if frame in self._paths:
                p_i, q_i = self._paths[frame][0][cursor], self._paths[frame][1][cursor]
                live_pos = (R.T @ (p_i - t)).astype(np.float32)
                live_quat = np.zeros(4)
                mujoco.mju_mulQuat(live_quat, q_Rt, q_i.astype(np.float64))
                pos[0, i] = torch.from_numpy(np.ascontiguousarray(live_pos))
                quat[0, i] = torch.from_numpy(live_quat.astype(np.float32))
            else:
                pos[0, i] = torch.from_numpy(np.ascontiguousarray(self._hold[frame]))
                quat[0, i, 0] = 1.0
        q = torch.from_numpy(np.ascontiguousarray(qpos_full)).float().unsqueeze(0)
        out = self.solver.solve(q, pos, quat, num_iters=num_iters)[0].numpy()
        self._k += 1
        return out
