"""Transferable high-level pick-reach policy: ONE object, kinematic + dynamic + real harnesses.

``ReachPolicy`` is the deliverable the phase-3 (kinematic) and phase-4 (dynamic) verifies both drive,
and the SAME object a real robot would run onboard. It takes NO simulator handle in ``step``: its only
per-cycle inputs are values a real robot also has -- an estimated base pose, proprioception, and an
OBSERVED cube pose -- and its outputs are exactly what a low-level controller consumes:

    step(base_pose, proprio_qpos, observed_cube_pose)
        -> (base_twist[vx,vy,wz], arm_reference{joint: mjlab_qpos}, camera_command{joint: qpos})

Internals it OWNS (allowed coupling, not a sim leak): a cuRobo planner + its world model. The world
model = a warmed ``CuroboPlannerSession`` + the static workbench geometry carried on ``mj_model`` + the
scenario descriptor; a real planner needs the identical world model built from perception. The cube
TARGET, by contrast, enters ONLY through ``observed_cube_pose`` (GT-as-observed now, a detector later),
so the policy never reads live sim state for control.

Robot-facing seam = an arm-trajectory command buffer (``DynamicArmReferenceExecutor``): the harness/robot
samples the current arm reference each control tick; WHETHER the planner replans sync (this module) or
async (a later internal swap) changes only WHEN the buffer reloads, never the interface.

Coordinate seam: the cuRobo route is ACTIVE-ARM cspace; the executor and every harness consume mjlab
qpos, so each route joint gets ``+ qpos0[qposadr]`` once here (the SAME fold ``CuroboArmPlanner`` and
phase-3 replay use). ``base_twist`` is zero until the base-live driving step lands.

Replan runs SYNC (``plan_reach_route`` blocks ``step`` on an in-process cuRobo) or ASYNC (``async_replan=True``):
async puts the ENTIRE cuRobo in ONE spawned process (``SpawnedReachWorker``) and keeps this policy cuRobo-free
-- it computes scene+targets on the CPU and ships them, so plan-0 and every replan use the SAME single runtime
(warmed once). Replans never stall the arm buffer -- only ``submit``/``poll`` cross the boundary. Either way the
executor interface is identical; async only changes WHEN the buffer reloads.
"""

from __future__ import annotations

import os
import pathlib
import sys

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tasks.visual_manipulation.dynamics_executor import DynamicArmReferenceExecutor  # noqa: E402
from tasks.visual_manipulation.jacobian_reach_tracker import JacobianReachTracker  # noqa: E402
from tasks.visual_manipulation.mink_reach_tracker import MinkReachTracker  # noqa: E402
from asset_zoo.parallel_gripper import (  # noqa: E402
    PARALLEL_GRIPPER_CLOSE_STROKE_M,
    PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S,
)
from tasks.visual_manipulation.curobo.planner import _PREGRASP_STANDOFF_M, yaw_quat  # noqa: E402
from tasks.visual_manipulation.curobo.scene import (  # noqa: E402
    aim_gaze_at_cubes,
    ik_feasible_route,
    known_anchors,
    plan_reach_route,
    pose_only_forward as _pose_only_forward,
    reach_scene_and_targets,
    robot_scene,
    solve_gaze_to,
    visible_reachable_field,
    SpawnedReachWorker,
    SpawnedReachMpcWorker,
    ReachRequest,
    _GAZE_SETTLE_EPS,
    _GAZE_SLEW_MAX,
    _COMMIT_H_HALF,
    _ns_id,
    _ns_joint_qadr,
)


# A base-live REPLAN reuses the fixed plan-0 assignment + warm seed, so the correction is small and easy;
# a low attempt cap keeps the per-solve wall down. A failed solve is SAFE (holds the last route, no arm-count
# degrade), so this only trades a rare 1-tick hold for speed -- it is not a correctness knob.
_REPLAN_MAX_ATTEMPTS = 2

# TWO-STAGE EXTEND. When ``_PREGRASP_STANDOFF_M > 0`` (planner.py, env ``REACH_PREGRASP_STANDOFF_M``; 0
# disables and restores the single-shot reach byte-for-byte), a committed reach lands in two PLANNED cuRobo
# routes instead of one: stage A to the goalset retreated by that standoff along each candidate's OWN
# approach axis, then stage B straight down that axis to the real grasp under a Cartesian linear-motion
# constraint. WHY: the single-shot route runs ~4.4 s while the base drifts ~2 cm against a rack corridor of
# ~0.015 m/side, so the gripper strikes the cube on the way in -- 133 of 176 measured nudge events land in
# EXTEND. Stage B re-plans from a FRESH base pose and cube belief over 5 cm, collapsing that drift budget on
# exactly the span that does the damage. Distinct from the REMOVED ``_MPC_PREGRASP_LIFT_M``, whose stage 2
# was an open-loop tracker descent with no collision check: both stages here are real collision-checked
# solves. Rides the existing mid-EXTEND replan seam (submit -> ``_final_replan_pending`` -> ``_install_reach``),
# so no new FSM phase and no change to phase accounting or the EXTEND->GRASP adjacency probe.
_TWO_STAGE_EXTEND = _PREGRASP_STANDOFF_M > 0.0

# Walk-search per-visit reach timing. REACH runs on ROUTE PROGRESS, not a tick budget: the arm EXTENDS until
# the cuRobo route's OWN clock finishes (``executor.cursor_s >= duration_s``) plus a brief settle, then
# RETRACTS along the REVERSED extend route (collision-free by construction; same duration as extend).
# Progress is dt-agnostic, so the visit lands IDENTICALLY in kinematic and dynamics; a fixed tick count does
# NOT -- it truncates a long dynamics reach or idles a short one (dynamics: the arm tracks its reference
# through PD lag, so N ticks != trajectory done). The route clock is the policy's own plan, not a
# realized-error probe, so this stays sim-handle-free.
_REACH_SETTLE_S = 0.4    # dwell after the route clock finishes, before retract (default physics settle onto the goal)
# GRASP closes only after base orientation STOPS CHANGING, not after it becomes world-level.  Bimanual carry
# can hold a deliberate lean indefinitely; a world-upright gate then idles despite an already-valid live
# object-relative target.  Compare normalized IMU gravity in the base frame over one fixed window: a static
# lean has zero delta, while post-walk sway does not.  Timeout prevents a reached waypoint from idling.
_GRASP_GRAVITY_WINDOW_S = 0.20        # fixed change-observation window
_GRASP_GRAVITY_DELTA_MAX = 0.03       # unit-gravity delta ~= 1.7 deg orientation change across the window
_GRASP_STABLE_TIMEOUT_S = 1.0         # reached waypoint must close within one second
# EXTEND-entry settle gate (separate budget from the grasp-latch gate above): the commit lands EXTEND on the
# ACQUIRE support-sweep backstop (actuated head only). A full pass is normally sub-second; this only prevents
# a pathological gimbal command from blocking the perception decision forever.
_ACQUIRE_LOCK_TIMEOUT_TICKS = 300    # ~6 s @ 50 Hz
# Env A/B switch for the framing standoff below (house pattern, cf. ``_PREGRASP_STANDOFF_M``); ``0`` restores
# the plain ``_REACH_RADIUS_M`` brake so the lever can be screened against a matched arm.
_DISCOVER_FRAMING_STANDOFF = os.environ.get("REACH_DISCOVER_FRAMING", "1") != "0"


def _framing_standoff(half_width: float | None) -> float:
    """DISCOVER brake distance that can actually FRAME a support of half-extent ``half_width``.

    ``known_anchors`` defines a bench anchor as its near-EDGE CENTER, so braking at the arm's reach radius
    (``_REACH_RADIUS_M``) says nothing about the cubes: one sitting ~0.25 m along the table's long axis is
    then both closer than ``NEAR`` and ~60 deg off the camera axis, outside the usable cone -- measured on
    the g1 seeds that mark cube_0 unseeable and burn the anchor. Standing back by
    ``half_width / tan(usable half-FOV)`` puts the support's far end exactly on the cone edge instead.

    Unbounded hand markers (``half_width is None``) keep ``_REACH_RADIUS_M``: a point anchor has no extent
    to frame. DISCOVER is only a SEARCH stop -- APPROACH drives to point C at reach distance afterwards --
    so a larger search standoff costs a little walking and nothing else."""
    if half_width is None or not _DISCOVER_FRAMING_STANDOFF:
        return _REACH_RADIUS_M
    return max(_REACH_RADIUS_M, float(half_width) / np.tan(_COMMIT_H_HALF))
# RETRACT_PLAN backstop: the async grasp->home plan runs on a spawned cuRobo worker (SpawnedReachWorker /
# SpawnedReachMpcWorker). If that child never returns a route (a solve that hangs, or a child that died
# silently -- observed g1-specific: its home plan never lands), RETRACT_PLAN would poll forever, stranding
# the mission holding the grasp. Bound the wait; on timeout fall back to ``_start_retract(None)`` -- the
# REVERSED extend corridor, collision-free by construction (it retraces the path the arm just extended
# along), same duration as extend. A healthy worker (v2/v2_fixed) returns in a handful of ticks, well inside
# this cap, so the fallback only ever fires for a wedged/dead solve -- it never masks a good async plan.
_RETRACT_PLAN_TIMEOUT_TICKS = 400    # ~8 s @ 50 Hz; worst cold worker solve ~2 s, so 4x margin
_GRIPPER_CLOSE_MAX_S = 1.0
# Do not retract on first rack contact. The wider pregrasp opening must finish its whole commanded stroke
# before a completed latch may advance, otherwise the hand visibly carries the cube with a jaw gap. This
# derives from the same open/close geometry and rate limit as the harness, yet stays within the 1 s cap.
_GRIPPER_CLOSE_MIN_S = min(
    _GRIPPER_CLOSE_MAX_S,
    PARALLEL_GRIPPER_CLOSE_STROKE_M / PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S + 0.02,
)
_GRASP_CAPTURE_MAX_RETRIES = 1
_DYNAMIC_FINAL_TRACK_ERROR_M = 0.04  # refresh final route when measured MPC tool error exceeds this
_DYNAMIC_FINAL_REPLAN_MAX = 1  # collision-aware route refresh after moving base leaves Jacobian envelope
# MID-ROUTE divergence abort (see ``_handle_extend``). Threshold sits above the ~0.03 m band healthy reaches
# hold and below the ~0.07 m at which a rack finger starts striking the cube (clearance is ~0.015 m/side).
# The debounce is ~0.5 s at 50 Hz: long enough that a transient spike cannot trigger a replan, short enough to
# still leave seconds of travel before contact. Progress bound keeps it off the final approach, which the
# settle-time check at ``_DYNAMIC_FINAL_TRACK_ERROR_M`` already owns.
# Replan bound raised 1 -> 3 (2026-08-01). One rescue assumes a single accumulated-lag event per commit; a
# reach that diverges twice (long approach, or a second lag build-up after the first re-solve) exhausted the
# budget and rode the stale route into the cube. Chatter stays bounded WITHOUT the bound doing the work: each
# fire zeroes the debounce, so consecutive replans are >= _EXTEND_DIVERGE_TICKS (~0.5 s) apart, and
# _EXTEND_DIVERGE_MAX_PROGRESS closes the guard for the last quarter of the route entirely. Worst case is 3
# solves and ~1.5 s of debounce inside the travelling phase. Capped trackers only, unchanged.
_EXTEND_DIVERGE_ERROR_M = 0.05
_EXTEND_DIVERGE_TICKS = 25
_EXTEND_DIVERGE_MAX_PROGRESS = 0.75
# Bound is 1, not 3: the 1-vs-3 A/B on ``bimanual_mixed_front_back_close`` v2 (10 seeds, --walk --dynamic)
# scored 8/10 for BOTH arms and failed the SAME two seeds (46, 47), so extra replans buy no P. They are not
# free -- ``route_progress()`` resets to 0 on every replan, so each rescue re-traverses the whole route and
# is charged to `Tbar_s`/`energy_J`. Repeated fires signal a chronic tracking deficit the replan rediscovers
# rather than fixes. Env-overridable (same idiom as ``_EXTEND_MAX_RETRIES``) because the sweep hosts
# share a single working tree: an A/B on this bound cannot be run by editing the constant on each
# host, they all read the same file.
_EXTEND_DIVERGE_REPLAN_MAX = int(os.environ.get("REACH_EXTEND_DIVERGE_REPLAN_MAX", "1") or 1)
# TARGET-MOVED replan (see ``_handle_extend``). The cuRobo route is solved ONCE per commit, so a cube the arm
# nudges leaves that route driving at where the cube WAS while the finger keeps pushing -- the measured
# `cube knocked off support` mechanism, since the handoff cube's palm support is a flat 0.083 x 0.091 m
# half-extent plate over open floor and the only feedback path (``note_fallen``) fires at 0.10 m of DROP,
# long after the cube is unrecoverable. Requires ``_pin_pose``'s real snapshots: the first version of this
# lever compared against live VIEWS of MuJoCo state, so it never fired once in 120 trials and every number
# attributed to it was void (``media/cotarget_replan_20260730/``). 0.02 m sits above table-contact noise and
# well below the ~0.05 m a cube must travel to leave the plate. Bounded to one replan per commit: a replan is
# a 1-tick hold, but an unbounded loop could chatter against a cube the arm is continuously grazing.
_TARGET_MOVED_REPLAN_M = 0.02
_TARGET_MOVED_REPLAN_MAX = 1
# A warm-seeded return solve can still start a few milliradians from the exact held grasp reference. Bridge
# that seam before executing the collision-aware home route, preventing the observed one-tick height drop.
_RETRACT_JOIN_S = 0.30
_RETRACT_JOIN_MAX_SPEED_RAD_S = 1.0
# RETRACT no longer ends at ``planning_home_joint_pos``. The return targets the home TOOL pose and the arm
# is REDUNDANT, so the configuration it settles in can sit a genuinely large distance from the static home
# config at zero Cartesian error (a different elbow/wrist branch for the same end-effector pose -- observed
# hold-phase ``corr_max`` up to 2.24 rad against the same nominal). Closing that gap used to mean a second,
# gap-scaled joint-space lerp to home after the return had already finished: a whole extra arm motion,
# costing time and energy to reach a pose kinematically no better than the one already held. The arm is now
# PARKED where cuRobo left it (``_park_arm_at``), which removes the motion instead of smoothing it.
# Parked-path replan throttle: a parked FSM with a far/unreachable target will get cuRobo result=None every
# tick (e.g. v2 spawn in left_right_far: 0.5 m from each cube, outside the reach envelope). Without a bound
# each tick resubmits -> ~50 GPU solves/sec with zero progress. THROTTLE: only resubmit every N ticks when
# the previous attempt returned None; TIMEOUT: hard cap on total wait so the driver gets a deterministic
# MISS verdict (cube missing from visit_min -> FAIL) instead of the 2800-step walk-cap hang. Tick-based =
# wall-clock at the fixed 50 Hz control_dt; safe under the file's existing tick convention. WALK PATH
# unaffected (walk drives the base to bring the cube into the envelope; gate below skips for self._walk).
_PARKED_REPLAN_HOLD_TICKS = 25          # ~0.5 s @ 50 Hz between resubmits on a failed solve
_PARKED_REPLAN_TIMEOUT_TICKS = 200       # ~4 s wall -> terminate cleanly (worst cold solve ~2 s; 2x margin)
# Extra closing distance (m) driven AFTER the reachability gate latches, before the base freezes. The gate
# fires at the reach-EDGE (its geometric floor + margin); reaching from there under physics leaves the arm
# near full extension, so the arm drooped ~0.10 m short of the 0.05 gate on a far visit (kinematic = 0 m, no droop).
# Closing this margin brings the stance nearer the parked reach distance (parked lands ~0.02 m). The gate
# is a latch, so the arm winner is already frozen -- only the STOP is delayed. Kept well inside the floor.
_WALK_APPROACH_M = 0.15

# COMMIT CEILING (REACH_FSM_REDESIGN.md decision 1): the reach/walk decision is by VISIBILITY, not a tight
# distance gate. A visible cube within this GENEROUS outer bound is committed IN PLACE and cuRobo proves/denies
# reach in EXTEND (feasibility = planner, not a radius); a cube BEYOND it is obviously out of the envelope and
# routes straight to a drive (APPROACH), skipping futile solves. Set at/above true max reach (~0.42 m: 0.382
# reaches, 0.48 is the retry-step "overshoots envelope" number) so it NEVER rejects a reachable cube -- the bug
# the old tight 0.35 gate caused (it wrongly rejected the spawn-reachable 0.382 m left_right_close cube). The
# far-scene cubes sit AT 0.5 m, so they commit in place, fail cuRobo, and fall to the EXTEND-no-route drive.
_REACH_MAX_M = 0.50         # commit iff base-to-cube planar dist <= this (visibility gate; planner proves reach)
#   SHARED default only -- ``RobotDescriptor.reach_max_m`` overrides it per robot, because "commit and let the
#   planner deny" is only cheap for a robot whose marginal solves FAIL. g1 overrides to 0.32; see that field.
# LIVENESS of the accrued ``_belief``, gating the PRE-COMMIT decision only (see ``_reachable_cubes``). Without
# it the belief is monotone -- a cube framed once by ACQUIRE's opening sweep stays "seen" forever, so the FSM
# commits a reach toward a cube the camera is not on (blind reach). Two SEPARATE gates are needed; neither
# alone is sufficient:
#   * FRESHNESS (this window) rejects seconds-old memory. Sized well above the per-tick refresh of a cube the
#     gaze is actively tracking, so it never blocks a legitimate commit; it only kills stale memory.
#   * SIMULTANEITY (same-tick, no constant) rejects a MULTI-cube commit whose members the rig cannot frame at
#     once. A time window CANNOT do this job: the gaze re-aims between believed cubes, so any window long
#     enough to permit a legitimate single-cube commit is also long enough for an alternating rig to mark
#     both "fresh". Simultaneity is self-sizing -- it asks the actual camera rig instead of encoding a guess
#     about sweep rate or camera count.
_BELIEF_FRESH_TICKS = 50    # 1 s @ 50 Hz: last-framed age above which a cube is too stale to COMMIT a reach to
# Budget for the APPROACH->ACQUIRE re-aim bounce, in TICKS, not attempts. A count was the wrong unit: the
# bounce cycles in ~26 ticks whether or not the gaze is still moving, so 3 attempts expired mid-slew and
# marked a cube permanently unreachable while its camera was still swinging toward it. The gimbal has the
# range (+-270 deg yaw) -- it was never a coverage limit, only a timing race. Sized off the SLEW: a worst-case
# full traverse is 9.42 rad at ``_GAZE_SLEW_MAX`` = 3.5 rad/s = 2.7 s = 135 ticks, and settle + re-detect
# follow. 300 matches ``_ACQUIRE_LOCK_TIMEOUT_TICKS``, the house bound for "gaze given its time and still
# did not converge". Does NOT relax ``_BELIEF_FRESH_TICKS``: the cube must still be freshly SEEN to commit.
_COMMIT_STALE_MAX_TICKS = 300   # ~6 s @ 50 Hz of re-aim time before FAILing the cube loud
# Proximity STOP RADIUS for the mover (NO LONGER the reach/walk discriminator -- decision 1 keeps it only as a
# locomotion brake distance). The learned reachability MLP this replaced was obstacle-blind and green-lit stances
# cuRobo found infeasible; cuRobo is the ground-truth feasibility check in EXTEND. Biased CLOSE so the base stops
# well inside the envelope for a reliable solve; the floor clamp guards the short-arm low end.
_REACH_RADIUS_M = 0.35      # mover stop radius: brake the base once base-to-cube planar dist <= this
# EXTEND commands exactly ZERO base twist, deliberately. Base drift since the (base-relative) route was
# solved does correlate with topple outcome in dyn8 (median 0.0446 m at fatal nudges vs 0.0267 m at benign,
# p = 0.0135), so a bounded station-keeping twist was built and MEASURED 2026-08-03 on `v2 left_right_close`
# seed 42 x10 against a matched REACH_EXTEND_HOLD_KP=0 arm. It made every metric WORSE: P 9/10 -> 7/10, drift
# median 0.0203 -> 0.0254 m, nudge events 7 -> 10. Cause: the only base actuator here is the frozen
# locomotion policy's velocity command, whose displacement quantum is a FOOTSTEP -- an order above the ~2 cm
# it was asked to cancel. Commanding a correction buys a step that displaces the base further than the drift
# did. Sub-cm station-keeping is unreachable through this seam; do not re-attempt it here.
_GEOMETRIC_FLOOR_M = 0.20   # never brake closer than this (short-arm g1 needs standoff to plan a route); with the
#   0.10 m close-in past the latch, this floor sets the final stance (~0.28 m from the cube) once radius < 0.38
# APPROACH-commit facing tolerance. A target the base is ALREADY within standoff of (front_back_close: base spawns
# equidistant 0.255 m from front AND rear cube, so a visit only TURNs in place, never WALKs) never trips the mover's
# TURN->WALK handoff (``FACE_TOL_RAD``, 2 deg when this was written, 8 deg since 2026-08-03): g1's standing sway
# swings the base ~8 deg on a 0.29 m target (few-cm sway subtends a large angle up close), so the gate may never
# see 3 consecutive in-tol steps and the mover TURNs forever -- the same interaction the 2->8 deg widening
# addresses at the mover, still worth a loose escape here because sway ~8 deg is right AT the widened gate. The
# base is nonetheless facing the cube to ~8 deg -- well within the arm's reach cone -- so commit once heading is
# under this LOOSE tolerance regardless of mover WALK state. Far visits still reach WALK (few-cm sway ~2 deg at 1 m)
# and brake through the WALK branch, so this only rescues the close turn-in-place case.
_APPROACH_FACE_TOL_RAD = np.radians(15.0)
# Point-C depth cap for a bounded (table/shelf) anchor: how far C may advance past the near edge toward a
# cube set back on the surface. Must stay well under the standoff the base holds from C (~0.28-0.35 m,
# geom_floor/robot minimum standoff) so the final base stance -- point_c_advanced minus that standoff along the
# approach normal -- still lands in FRONT of the edge, never on the table (live-test finding: left_right_far's
# cube sits ~0.119 m in from the edge; depth pinned to 0 left the base that far short on IK, unreachable).
_POINT_C_MAX_DEPTH_M = 0.15
_BACKOFF_VX = 0.40          # m/s while backing off, along the mover's latched direction reversed (raw twist,
#   emitted directly -- these back-off branches bypass ``_navigate`` so the mover's ramp/floors do NOT apply).
#   Was 0.20 = EXACTLY the checkpoint's measured linear dead zone ("0.2 m/s is inside the LINEAR dead zone,
#   75% velocity_tracking_failure", MoverCfg.turn_cruise premise, 2026-08-02) -- the base barely reversed, so
#   a back-off retry burned its whole tick budget without changing stance. 0.40 = HeuristicMovingPolicy.CRUISE_VX,
#   the user-verified stable magnitude and the only regime where the checkpoint tracks. Overshoot is bounded:
#   the branch re-tests ``dist < standoff`` every tick, so at 50 Hz it exits within 8 mm of the standoff.
# Bounded supports need root-to-face clearance, not only root-to-point-C distance. The configured body
# footprint handles direction-dependent torso reach; this fixed margin absorbs model mismatch after the
# mover's analytic command-ramp braking distance is included.
_SUPPORT_CLEARANCE_MARGIN_M = 0.05
# Square-up before committing a CLOSE (back-off) reach. The DISCOVER 180 deg turn leaves the base up to
# ~13 deg off the rear cube, and the straight back-off never corrects heading, so the arm reaches OBLIQUE
# (~0.04 m realized miss under carry, right at the 0.05 m grade). This motivates a pre-commit square-up
# turn for g1 ONLY (3 deg face tolerance, ``_REACH_FACE_TOL_RAD``): its single fixed forward camera has
# no gimbal, so commit-time facing is the ONLY thing keeping a latched cube inside the frustum -- v2/
# v2_fixed skip it (``_faced``/``_square_up_wz`` gate on ``_needs_reface``), since their all-around cameras
# make reachability (base-facing-independent; cuRobo proves reach from the actual stance in EXTEND) the
# only concern. (A later change stubbed ``_faced``/``_square_up_wz`` to unconditional True/0.0 for every
# robot, reasoning reachability alone -- that also silently dropped g1's visibility guarantee, which is
# the ONLY thing this turn buys: a live g1 --camera run committed a grasp outside its own FOV frustum.
# Restored 2026-07-25.)
# Soft guard on APPROACH square-up: if the base never crosses the face tolerance within this many ticks
# (e.g. simulator wedge, RL base stuck), fall through and commit anyway. Never hang waiting to face.
_REACH_SQUARE_TIMEOUT_TICKS = 200   # ~4 s @ 50 Hz
_REACH_FACE_TOL_RAD = np.radians(3.0)      # g1-only commit-time facing tolerance (frustum visibility)
_REACH_STEER_K = 1.2                       # proportional steer gain, bearing (rad) -> wz (rad/s)
_REACH_TURN_WZ_MIN = 0.6                   # clears the RL base's yaw deadzone (sub-floor command = no turn).
#   Tracks HeuristicMovingPolicy.TURN_WZ_MIN, which was raised 0.4 -> 0.6 for ALL robots (2026-08-02); this
#   SECOND floor was left behind at 0.2 -- the command that freezes 51% of envs, and ``_square_up_wz`` is emitted as a PURE yaw twist `(0, 0, wz)`, the
#   zero-companion-linear cell where that freeze is worst. Applies to the ``_needs_reface`` robots only
#   (g1, v2_single, v2_single_fixed); v2/v2_fixed return 0.0 here regardless.
#   ACCEPTED RISK (user decision 2026-08-05, flat floor chosen over a graded one): this floor is 0.6 against a
#   3 deg ``_REACH_FACE_TOL_RAD`` window, and MEMORY 2026-08-03 records that a 0.6 floor against a window
#   narrower than the gait's own +-6 deg bounce is the limit-cycle geometry that made the MOVER's 2 deg gate
#   unwinnable (fixed there by widening FACE_TOL_RAD 2 -> 8 deg, which is NOT available here: the 3 deg is
#   g1's frustum-visibility guarantee, not a settle tolerance). On the v2 checkpoint the in-place achieved
#   rate is only ~0.3 rad/s of a 0.6 command (ang_err 0.299, MoverCfg), i.e. ~0.34 deg/tick -- too slow to
#   kick through 3 deg -- but g1 runs a DIFFERENT policy (G1RmaVelEstArmFlashSacL2T) with no equivalent
#   measurement, so that argument does not cover it. Bounded either way by _REACH_SQUARE_TIMEOUT_TICKS, which
#   commits after ~4 s; the failure mode if it does cycle is a slow off-facing commit, not a hang. Watch g1's
#   square-up dwell on the next dynamic run.
#   Residual (not fixed here): the measured rescue for pure-yaw freeze is a companion LINEAR command, which
#   this branch cannot add -- it squares up at standoff and must not translate.
_REACH_TURN_WZ_MAX = 1.0                   # turn-rate cap; tracks HeuristicMovingPolicy.TURN_WZ_MAX. MUST stay
#   above the MIN above: np.clip(x, min, max) with min > max returns max, so a 0.5 cap would silently pin every
#   square-up to 0.5 and make the new floor dead code.
# Walk-path EXTEND feasibility bound + back-off retry. A committed reach whose cuRobo plan-0 never lands
# (stance TOO CLOSE / the reaching arm blocked by the already-carried cube / geometrically unreachable)
# leaves ``_route`` None, so ``_extend_settled`` never fires. The PARKED path guards this
# (``_PARKED_REPLAN_TIMEOUT_TICKS`` -> MISS); the WALK path did NOT, so an infeasible in-place reach HUNG to
# the step cap (verified: bimanual_mixed_close 2nd cube idled ~1900 ticks with route=None). Bound it: on
# an infeasible stance, BACK OFF one standoff step and re-commit (the arm needs room; "walking too close" is
# the common cause), up to ``_EXTEND_MAX_RETRIES`` times, THEN abandon the cube (report MISS) and move on --
# never freeze. Implements "check the grasp plan works from HERE first; if not, reposition, don't stall".
# Trip on the count of actual cuRobo INFEASIBLE plan-0 verdicts, NOT wall-ticks: under GPU contention a
# feasible solve's latency balloons, and a tick counter then false-tripped a GOOD stance (verified:
# left_right_close regressed to a 0.22 m droop-miss when a slow-but-feasible solve was mistaken for
# infeasible and backed off). Infeasible-verdict counting is load-independent. A hard tick backstop still
# bounds a totally wedged solve (worker hang) so the visit can never freeze.
_EXTEND_MAX_INFEASIBLE = 3      # consecutive cuRobo "no route" verdicts from this stance -> infeasible
_EXTEND_HARD_TICKS = 600        # ~12 s @ 50 Hz absolute no-route backstop (worker wedge) -> break regardless
# Back off ONE step only, AFTER bounded-support tangent candidates fail. Nominal standoff 0.24 m; +0.10 -> 0.34 m gives the arm room to plan (fixes the
# too-close infeasible reach) while staying INSIDE the reach envelope. Must stay < _REACH_RADIUS_M (0.35 m,
# the latch ceiling): 0.24+0.12=0.36 was tried and OVERSHOT that ceiling, so a stance already latched at the
# ceiling (0.35 m -- not actually too close, e.g. a far cube EXTEND failed to plan for a different reason)
# always read as "too close" (0.35 < 0.36) and got forced to back off further, away from a cube that instead
# needed the base to drive CLOSER (live-test finding: left_right_far backed the base off until unreachable).
# A second step (0.48 m) overshoots the envelope -> the arm reaches at full stretch and droops ~0.22 m short
# (a miss). If one back-off does not make it feasible, the stance is genuinely unreachable (e.g. the target
# arm is occupied by the carried cube) -> drop.
# A no-route verdict from a table/shelf stance is not proof that radial back-off is the right correction. The
# robot may remain safely outside the support while shifting along its clearance contour, which changes arm
# workspace but leaves support-face clearance unchanged. Try targetward contour shift first, then its opposite from that first
# stance; only then use the historical radial back-off fallback.
_SIDE_STANCE_RETRY_STEPS_M = (0.12, -0.24)
_CURATED_STANCE_OFFSETS_M = (-0.12, 0.0, 0.12)  # same bounded support contour as first legacy retry
_SIDE_STANCE_SPEED_M_S = 0.40   # raw twist, same bypass and same dead-zone reason as _BACKOFF_VX above
_SIDE_STANCE_DONE_M = 0.015
_SIDE_STANCE_TIMEOUT_S = 3.0
_SIDE_STANCE_CLEARANCE_BUFFER_M = 0.01
# A stance candidate this close to one already occupied for the SAME cube IS that stance, so re-entering
# it cannot produce a different reach outcome -- only another scene disturbance. ``_side_stance_retries``
# resets per VISIT (``_begin_visit``), so without cross-visit memory a second visit re-derives the identical
# candidate set and walks back to the winner it already failed at. Observed as the paired
# ``visible-reachable side stance 0.156 -> 0.266`` / ``0.000 -> 0.266`` logs on a
# ``physical grasp latch did not engage`` failure: two full visits, same accepted score, jaws closing on air
# both times. Kept BELOW the 0.12 m curated offset spacing so distinct offsets stay distinct and only
# near-duplicates collapse. Note the already-rejected lever here was retry COUNT
# (``_GRASP_CAPTURE_MAX_RETRIES`` 1->2, measured WORSE precisely because a third attempt re-entered the same
# saturated stance); this suppresses the repeat instead of buying more of them.
_STANCE_REVISIT_TOL_M = 0.08
_EXTEND_MAX_RETRIES = int(os.environ.get("REACH_EXTEND_MAX_RETRIES", "2") or 2)
# radial back-off attempts after bounded-support tangent retries. Env-overridable (default = the published
# 2, so every existing table row reproduces) because the live-handover cells need a LARGER budget by
# design: there the offered object legitimately ends somewhere the current stance cannot reach, and the
# intended response is to step and reach again rather than to report the cube unreachable. Raising the
# constant outright would have re-defined the controller for all 18 published cells at once; an override
# set identically for every robot in a run keeps the ablation fair and the change disclosed at run time.
# Bumped 1->2 (2026-07-26): fixed retry COUNT exhausted against a stale cube-pose estimate under slow
# (contended-GPU) async solves flips would-be PASS into "unreachable" FAIL -- same seed/robot/scenario
# reproduced both FAIL and PASS headless back-to-back with zero code change, only ambient GPU load
# differed. One extra retry adds slack against solve-latency jitter without touching the async solve
# architecture.
_RETRY_STANDOFF_STEP = 0.10     # extra standoff (m) added on radial retry so re-commit backs base off

# BENCHMARK PROTOCOL, not a controller tunable: hold the body and arm for the first N control ticks while
# perception and gaze run normally. cuRobo plans one trajectory against one frozen world snapshot, so a
# scene disturbance may only land BEFORE the first commit -- and the published stationary cells commit at
# tick 0-11 (measured, every robot/cell/seed: they start the robot already in reach), leaving no such
# window. The hold opens one: the robot watches, the offered object moves, the robot then commits against
# whatever belief it managed to hold. 0 (the default) reproduces every published run tick-for-tick, and one
# value shared by all three robots keeps the disturbance robot-independent.
_PREACT_HOLD_TICKS = int(os.environ.get("REACH_PREACT_HOLD_TICKS", "0") or 0)

# REAR PRIME. On a carry visit whose next cube sits BEHIND the stance being walked to, the free arm arrives
# forward-facing, the cube is behind its shoulder, and cuRobo returns INFEASIBLE until the visit is abandoned
# (``cube_1 unreachable after 2 back-off retries``). Fix: plan the arm's flip into the rear basin and play
# it DURING the walk, in open space. Two earlier attempts are recorded as dead ends -- seeding plan-0 with a
# flipped config the arm was NOT in (the route then starts radians from the real arm and the whole-body
# disturbance dislodges the carried cube), and flipping at the desk, where the table blocks the swing.
# Threshold = degrees off the PREDICTED stance's +x, past which the cube counts as behind. 0 disables the
# feature and leaves every un-primed code path bit-identical, which is what makes the paired A/B possible;
# the metric is per-visit ``_plan0_infeasible``, not P, which has no power at the available sample sizes.
_REAR_PRIME_BEARING_DEG = float(os.environ.get("REACH_REAR_PRIME_BEARING_DEG", "120") or 0.0)
_REAR_PRIME_MIN_TRAVEL_M = 0.6  # don't start a flip the walk is too short to hide: the arm must physically
# REACH the primed configuration before ``_step_mpc`` may seed plan-0 with it, or the seed is a lie again.

_ARM_DEBUG = bool(os.environ.get("REACH_ARM_DEBUG"))  # per-tick arm_ref discontinuity trace
_COMMIT_DEBUG = bool(os.environ.get("REACH_COMMIT_DEBUG"))  # per-visit commit-gate margin trace

# Slew limit (rad/s) on the emitted arm reference, applied at the ``_dispatch`` seam. Every ROUTE the arm
# follows is already speed-bounded, but the SOURCE of the reference is not continuous across a handoff, and
# each switch is a step change in a position-servo command -- a whip with no plan behind it. Measured on
# v2/front_back_far, all three seen in one mission:
#   * EXTEND -> GRASP: the tracker leaves ``_tag="extend"`` and the transit tube releases in one tick
#     (0.06 -> 0.25 rad of allowed correction), landing ~0.12-0.16 rad of previously-clamped IK error.
#   * hold -> RETURN: ``begin_return`` re-anchors ``q_nom`` from the planned ``route_q[-1]`` onto MEASURED
#     qpos, which differ by the gravity droop (~0.14 rad).
#   * parked hold -> next visit's tracker: ``_arm_hold`` holds the previous visit's landed pose while plan-0
#     flies, then the new route's first tick emits from a config up to 2.5 rad away.
# The gate lives HERE and not inside the tracker because only this seam sees every source (tracker route,
# ``DynamicArmReferenceExecutor`` playback, ``_arm_hold`` park, retract bridge). A tracker-internal anchor
# was tried first and is stale by construction: it carries the tracker's OWN last command across a handoff
# it did not emit, which clamps the new command toward a pose the arm is no longer at.
# Measured per-tick deltas during planned motion top out at 0.055 rad (p99 0.050) against a 20 ms
# control_dt, so 4.0 rad/s (0.08 rad/tick) sits above everything the routes ask for and binds only on the
# switches, spreading each over a few ticks. Rejected: the tracker's ``output_filter`` (a global 2nd-order
# lag on every tick, not just the discontinuous ones, and deliberately off for all robots).
_MAX_ARM_SLEW_RAD_S = 4.0

_PLAN0_STALE_DRIFT_M = 0.03     # cube-center drift (m), submit-time vs landed-time, past which a plan-0
# route is discarded and resubmitted instead of installed. A slow (contended-GPU) async solve lets real
# wall-clock time pass; the cube can physically move (base disturbance, not domain-rand jitter -- jitter
# only fires at reset()) while the solve is in flight, so the landed route targets a pose the cube is no
# longer at. Committing anyway wastes an _EXTEND_MAX_RETRIES attempt discovering it downstream; catching
# it here and resubmitting fresh is not a reachability failure, so it does NOT charge that budget.

# Field-guided TARGET stance selection is a distinct treatment from the existing post-no-route contour
# recovery. It is opt-in for isolated A/B: legacy Point-C remains the default. Thresholds are calibrated
# against collision-blind cuRobo IK (`calibrate_tau.py`, N=48, seed 17): V2=0.25 prior study,
# V2-fixed=0.20, G1=0.10. ``VISIBLE_REACHABLE_TARGET_TAU`` permits a deliberate calibration override.
# The field proposes a fixed-yaw base
# target only when the current target's robust margin is below tau. Fast IK then full cuRobo remain authority.
_FIELD_TARGET_MAX_CANDIDATES = 4
_FIELD_TARGET_MIN_TRAVEL_M = 0.06
_FIELD_TARGET_DIRECTION_COS = float(np.cos(np.radians(2.0)))  # no pre-walk body turn
_FIELD_TARGET_TAU_BY_ROBOT = {"v2": 0.25, "v2_fixed": 0.20, "g1": 0.10}
_FIELD_TARGET_DUAL_VIEW_CROSS_TRACK_RATIO = 0.60
_FIELD_TARGET_GAZE_CROSS_TRACK_RATIO = 1.50
_FIELD_TARGET_ARRIVAL_M = 0.05

# ONE flat mission FSM for BOTH parked and walk reaches. Every phase is an explicit state (no scattered
# sub-phase booleans); state is written ONLY by ``_transition`` and each handler RETURNS its next state
# (``None`` = stay), so a guard and its transition can never drift into separate blocks (the old bug class).
#
#   ACQUIRE       PERCEPTION-FIRST search+decide, runs for EVERY walk robot (REACH_FSM_REDESIGN.md): the
#                 actuated head sweeps the gimbals (base still) until one anchor pass completes; a fixed cam
#                 has no sweep and decides immediately. Then commits the whole VISIBLE-reachable SET in place
#                 (bimanual when >=2) -> EXTEND, or hands to ``_select_next`` (walk to see / reach)
#   DISCOVER      body-search a hidden cube into the camera belief (walk only)
#   APPROACH      locomote until the cube is arm-reachable, pin the reaching arm(s)  (walk only)
#   EXTEND        run the reach dispatch (cuRobo route / tracker) until the arm lands on the grasp
#   GRASP         hold the final arm command while the position-servo jaws close, then submit the home plan
#   RETRACT_PLAN  poll the async grasp->home plan, holding the grasp
#   RETRACT       play the grasp->home return, jaws closed; on settle finish the visit
#   TERMINATE     mission done (walk: all cubes serviced; parked: returned home == ``parked_done``)
#
# Parked reach starts at EXTEND (base already standing); walk ALWAYS enters at ACQUIRE (perception-first for
# every embodiment -- the fixed-cam "skip SCAN, straight to _select_next" path was the walk-before-see defect).
# Pick-and-place slots in as a localized addition to THIS enum + dispatch: the GRASP->next edge would branch on
# a mission kind to CARRY/PLACE/RELEASE instead of RETRACT_PLAN (not implemented -- reach-home is the only
# branch today). The occupied-arm guard (``_held``) is the state pick-place relies on.
_ACQUIRE, _DISCOVER, _APPROACH, _EXTEND, _GRASP, _RETRACT_PLAN, _RETRACT, _TERMINATE = (
    "ACQUIRE", "DISCOVER", "APPROACH", "EXTEND", "GRASP", "RETRACT_PLAN", "RETRACT", "TERMINATE")
_REACH_STATES = (_EXTEND, _GRASP, _RETRACT_PLAN, _RETRACT)


def _wrap_pi(angle: float) -> float:
    """Wrap ``angle`` (rad) to ``(-pi, pi]`` -- the shortest signed distance around the circle."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _yaw_only(base_pose):
    """Project a base pose to its PLANAR (yaw-only) form: keep XY/Z, discard transient roll/pitch.

    The arm planner reasons in the UPRIGHT base frame; the low-level policy owns balance, so a
    floating-base humanoid's few-degree physics lean is not a commanded orientation. Planning against
    the full tilted quat rotates the level grasp goalset out of reach (verified: ~1 deg pitch flips a
    feasible reach to infeasible), whereas the yaw-only frame solves cleanly. Arm references are
    joint-space, so executing the route is orientation-independent -- the tilt is absorbed by tracking."""
    pos, quat = base_pose
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (np.asarray(pos, dtype=float), np.asarray(yaw_quat(float(yaw)), dtype=float))


class _GazeController:
    """POLICY-owned head-gimbal aim for an actuated-camera robot (the ONLY hardware-conditioned surface,
    per REACH_FSM_REDESIGN.md). Decides WHERE the gimbals point; it does NOT perceive -- detection stays a
    harness/robot sensor (``detect_at_gaze`` at the realized gaze), fed back to the policy as the belief.
    This is the seam inversion the redesign asks for: on a real robot the controller commands the gimbal
    and the camera+detector report what is framed; the policy never reads sim state to aim.

    Aim = pure kinematics of the robot's OWN camera mounts from its OWN estimated base pose: a scratch
    ``MjData`` (built from the policy's ``mj_model``, allowed self-knowledge -- the same model the cuRobo
    world uses) is posed at the live base + current gimbal command, refreshed with ``pose_only_forward``
    (kinematics only -- this scratch is read for poses, never contacts/Jacobians/bias forces; see that
    function), and the closed-form in-ROM gaze solve (``solve_gaze_to`` / ``aim_gaze_at_cubes``) run
    against it. Non-cam joints stay at
    ``qpos0`` -- the head mounts rigidly to the base, so arm/leg pose does not move the camera sites.

    Behavior (relocated from the harness ``GazeScanner``, minus detection + the fragile ``reached`` settle):
    SWEEP the known support anchors one at a time at a realistic slew (``_GAZE_SLEW_MAX``) while any pending
    cube is still unseen; once every cube the policy cares about is in belief, TRACK them (hold gaze on the
    seen centers for the reach). ``full_pass`` flips True after one complete anchor pass -- the RELAXED
    FIXATE gate (the mission may act once the pass is done; it does NOT wait for the gimbals to lock onto a
    cube, which on a static base never converged -- the kinematic settle-hang this replaces)."""

    def __init__(self, mj_model, descriptor, control_dt: float):
        self._mj_model = mj_model
        self._cfg = descriptor
        self._dt = float(control_dt)
        self._joints = [j for _s, yj, pj, _sg in descriptor.gaze_cams for j in (yj, pj)]
        # Rest pose = v2_fixed's WELDED camera stop (yaw 0, pitch downtilt), so an actuated v2 head
        # INITIALIZES looking at the same place the fixed-camera build does (level yaw, ~48 deg down at
        # a table) instead of staring horizontally; the sweep then slews off this baseline.
        from asset_zoo.humanoid_v21.humanoid_v21_constants import WELDED_CAM_DOWNTILT_RAD
        self._rest = {}
        for _s, yj, pj, _sg in descriptor.gaze_cams:
            self._rest[yj], self._rest[pj] = 0.0, WELDED_CAM_DOWNTILT_RAD
        # Gimbal stops, needed because ``cmd`` is integrated open-loop below and must not drift outside what
        # the joint can physically reach (see the clamp in ``step``).
        self._rom = {}
        for j in self._joints:
            jid = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self._rom[j] = tuple(float(v) for v in mj_model.jnt_range[jid]) if mj_model.jnt_limited[jid] else None
        root_id = _ns_id(mj_model, mujoco.mjtObj.mjOBJ_BODY, descriptor.base_link)
        self._root = int(mj_model.joint(mj_model.body(root_id).jntadr[0]).qposadr[0])
        self._scratch = mujoco.MjData(mj_model)
        self.reset()

    def reset(self) -> None:
        self.cmd = dict(self._rest)
        self.full_pass = False
        self.settled = False            # sufficient support coverage has completed
        self._anchors = None            # lazily built from the live base_xy on the first step
        self._idx = 0

    def _aim_committed_camera(self, data, cam_i: int, target_world) -> tuple[str, float, str, float] | None:
        """Return one camera's centerline aim after bounded live-FK refinement.

        ``solve_gaze_to`` is closed form for a lens at its current pose.  A real
        gimbal lens sits off its yaw/pitch pivots, so writing that solution moves
        the lens and leaves a small parallax error in the rendered +Z FOV axis.
        Two fixed-point FK updates remove that error without changing the camera
        gate, target, or joint-ROM policy.  ``data`` is this controller's scratch
        state only; physical motion remains rate-limited below.
        """
        _site, yaw_joint, pitch_joint, _sign = self._cfg.gaze_cams[cam_i]
        for _ in range(2):
            aim = solve_gaze_to(self._mj_model, data, self._cfg, target_world)
            if yaw_joint not in aim:
                return None
            yaw, pitch = aim[yaw_joint], aim[pitch_joint]
            data.qpos[_ns_joint_qadr(self._mj_model, yaw_joint)] = yaw
            data.qpos[_ns_joint_qadr(self._mj_model, pitch_joint)] = pitch
            _pose_only_forward(self._mj_model, data)
        return yaw_joint, float(yaw), pitch_joint, float(pitch)

    def _rom_target(self, joint: str, tgt: float, cmd: float) -> float:
        """Resolve a solved gimbal angle to the equivalent angle this joint can actually REACH.

        A gaze solution names a physical DIRECTION, so every ``tgt + 2*pi*k`` aims the lens the same way.
        Taking the nearest one (what a bare ``_wrap_pi`` does) is right whenever it is inside the stop and
        WRONG at the stops: a direction just past the near limit is still reachable by turning the LONG way
        round, and the yaw gimbal's +-270 deg range (span 9.42 rad, 1.5 turns) exists precisely to give that
        overlap. Without this the slew pins at the stop carrying a large standing error, the lens stares at
        the limit, and the cube behind it is never framed again -- the visit then declines `stale belief` and
        no amount of extra re-aim time helps, because the command has stopped moving (measured: 300-400
        ticks, 0 detections, base parked at a constant distance).

        Picks the in-ROM equivalent CLOSEST to the current command, so the normal case still takes the short
        path and is bit-identical to the old wrap. Falls back to the nearest equivalent when no ``k`` lands
        inside the stop (pitch: span pi, no overlap, so there is never a second option) -- the caller's clamp
        then handles it exactly as before.
        """
        nearest = cmd + _wrap_pi(tgt - cmd)
        rom = self._rom.get(joint)
        if rom is None:
            return nearest
        lo, hi = rom
        reachable = [c for k in (-1, 0, 1) if lo <= (c := nearest + 2.0 * np.pi * k) <= hi]
        return min(reachable, key=lambda c: abs(c - cmd)) if reachable else nearest

    def step(self, base_pose, scenario, seen_centers, locked_targets_by_cam=None) -> dict:
        """Advance one aim tick and return the gimbal command ``{joint: angle}``. ``seen_centers`` = world
        centres of the cubes already in belief (drives the TRACK vs SWEEP choice, mirroring the old scanner)."""
        d = self._scratch
        d.qpos[:] = self._mj_model.qpos0
        d.qpos[self._root:self._root + 3] = np.asarray(base_pose[0], dtype=float)
        d.qpos[self._root + 3:self._root + 7] = np.asarray(base_pose[1], dtype=float)
        for j, a in self.cmd.items():
            d.qpos[_ns_joint_qadr(self._mj_model, j)] = a
        _pose_only_forward(self._mj_model, d)
        base_xy = np.asarray(base_pose[0], dtype=float)[:2]
        if self._anchors is None:
            anchors = known_anchors(scenario, base_xy)          # bearing order -> a smooth continuous pan
            self._anchors = sorted(
                anchors, key=lambda na: float(np.arctan2(na[1][1] - base_xy[1], na[1][0] - base_xy[0])))
        num_targets = len(scenario.pick) if hasattr(scenario, "pick") else 0
        if num_targets > 0 and len(seen_centers) >= num_targets:
            self.full_pass = True
        tracking = self.full_pass and bool(seen_centers)
        parallel_sweep = self._anchors is not None and len(self._cfg.gaze_cams) >= len(self._anchors)
        if locked_targets_by_cam:
            desired = {}
            for cam_i, target in locked_targets_by_cam.items():
                refined = self._aim_committed_camera(d, cam_i, target)
                if refined is not None:
                    yaw_joint, yaw, pitch_joint, pitch = refined
                    desired[yaw_joint], desired[pitch_joint] = yaw, pitch
            tracking = True
        elif tracking:
            aim_gaze_at_cubes(self._mj_model, d, self._cfg, list(seen_centers))
            desired = {j: float(d.qpos[_ns_joint_qadr(self._mj_model, j)]) for j in self._joints}
        elif parallel_sweep:
            # Known support anchors are static scene geometry, never target poses. A gimbal pair can inspect
            # two supports concurrently; serial sweep wastes one camera and makes V2 slower than fixed heads.
            aim_gaze_at_cubes(self._mj_model, d, self._cfg, [anchor[1] for anchor in self._anchors])
            desired = {j: float(d.qpos[_ns_joint_qadr(self._mj_model, j)]) for j in self._joints}
        elif self._anchors:
            # Two fixed-point FK passes, same correction as ``_aim_committed_camera`` (see its docstring):
            # the lens sits off its yaw/pitch pivots, so a single ``solve_gaze_to`` against the SCRATCH's
            # current (lagged) lens position is systematically wrong by the parallax offset, and slewing
            # toward that wrong target every tick moves the lens again -- an undamped fixed-point iteration
            # through real kinematics. Verified unstable: v2_single/left_right_far seed 42 chattered cam_yaw
            # and cam_pitch every tick from t~1s, amplitude growing until pinned at the rate limit (measured
            # delta == _GAZE_SLEW_MAX * dt exactly, i.e. overshoot both ways every tick, forever). Two passes
            # converges the same way the committed-camera path already does; a single-cam anchor target has
            # no combinatorial assignment to redo, so this is cheap.
            desired = {}
            for _ in range(2):
                desired = solve_gaze_to(self._mj_model, d, self._cfg, self._anchors[self._idx][1])
                for j, a in desired.items():
                    d.qpos[_ns_joint_qadr(self._mj_model, j)] = a
                _pose_only_forward(self._mj_model, d)
        else:
            desired = {}
        step_max = _GAZE_SLEW_MAX * self._dt
        reached = True
        for j in self._joints:
            tgt = desired.get(j, self.cmd[j])
            # The right gimbal's zero-yaw datum faces BACKWARD, so aiming it at a FRONT cube legitimately
            # needs yaw near +-pi -- exactly atan2's branch cut. A frame's rounding can flip the solved
            # value between +pi-eps and -pi+eps (the SAME physical direction, a ~2*pi apart pair of
            # numbers). The raw difference misreads that as a huge required rotation; wrapping to the
            # shortest signed distance keeps the slew on the correct (short) side and prevents the
            # sustained hunt-the-branch limit cycle this caused (verified: seed 182 front_back_close,
            # cam_yaw_right chattered +-pi every tick, forever, without this wrap).
            # ``_rom_target`` subsumes the wrap AND adds the long-way-round option at the stops, so the
            # target is already an absolute angle in the joint's reachable set -- no second wrap here.
            tgt = self._rom_target(j, tgt, self.cmd[j])
            self.cmd[j] += float(np.clip(tgt - self.cmd[j], -step_max, step_max))
            # CLAMP TO THE GIMBAL STOP. ``delta`` is wrapped to the shortest signed path (above), which keeps
            # each STEP short but places no bound on the ACCUMULATED command: repeated wrapped steps let
            # ``cmd`` random-walk a whole turn outside the joint's ROM. Once it does, the joint is physically
            # pinned at its stop while ``_wrap_pi(tgt - cmd)`` still reads ~0 -- the controller believes it is
            # on target, stops slewing, and reports ``reached``, so the lens stares in a fixed wrong direction
            # forever. Measured on v2_single/left_right_far seed 47: cam_yaw cmd 8.134 vs desired 1.851,
            # exactly 2*pi apart, against a +-4.7124 rad stop; the second cube was never framed again and the
            # visit failed `not visible at commit (stale belief)`. Clamping keeps ``cmd`` in the reachable set so
            # the wrapped error measures REAL pointing error.
            if (rom := self._rom.get(j)) is not None:
                self.cmd[j] = float(np.clip(self.cmd[j], rom[0], rom[1]))
            # ABSOLUTE error, not wrapped: a command pinned at the stop is 2*pi from a target it cannot
            # reach, and the old wrapped test read that as ~0 -- the controller called itself settled while
            # the lens pointed at the limit. Absolute error also keeps ``reached`` False for the full
            # long-way slew, which is exactly when the sweep must not advance.
            if abs(tgt - self.cmd[j]) > _GAZE_SETTLE_EPS:
                reached = False
        # Advance the sweep only while patrolling (not tracking) and once the gimbals reach the anchor. A FULL
        # pass must complete before the mission acts so a bimanual scene gathers BOTH side cubes.
        if not tracking and reached and self._anchors:
            if parallel_sweep:
                self.full_pass = True
            else:
                self._idx = (self._idx + 1) % len(self._anchors)
                if self._idx == 0:
                    self.full_pass = True
        # A full support pass is the perception proof: every pending cube has either entered the live-camera
        # belief or its known support was inspected empty.  Continuing to chase an already-observed cube does
        # not add information, and assignment changes can keep ``reached`` false until ACQUIRE's 6 s backstop.
        # Keep the camera tracking command for the following reach, but let the mission decide immediately.
        self.settled = self.full_pass
        return dict(self.cmd)


class ReachPolicy:
    """Plan-and-track a reachability-driven reach for 1 OR 2 cubes. See the module docstring for the
    transfer contract. The arm COUNT is not configured -- it is DERIVED each replan from what is reachable
    (and, under a camera-gated harness, visible): both observed cubes reachable -> both arms; one -> one;
    none -> hold. The planner's feasibility ladder (bimanual-first, single fallback) lives in
    ``plan_reach_route``; this class only installs the resulting route into the arm-trajectory buffer.

    Args:
        session: warmed ``CuroboPlannerSession`` (robot descriptor + kinematics + pose planner) for the SYNC
            path. Pass ``None`` for ``async_replan`` -- the async policy holds no cuRobo (sole runtime = the
            spawned worker) and derives its CPU descriptor from ``robot_scene(robot_name, scenario_name)``.
        scenario: height-correct ``Scenario`` (its ``pick`` cubes source the candidate targets).
        mj_model: compiled model = the planner's world model (static obstacles + qpos0 ref offsets).
        control_dt: seconds between control ticks (executor retiming cadence).
        replan_interval_s: ``None`` = plan ONCE (first step it succeeds; open-loop thereafter); a float
            re-solves from the live base pose every ``round(replan_interval_s/control_dt)`` ticks
            (base-live tracking).
        async_replan: ``False`` = the SYNC re-solve (``plan_reach_route`` blocks ``step`` on the ``session``'s
            in-process cuRobo each replan). ``True`` = the ASYNC path: cuRobo runs in ONE spawned process
            (``SpawnedReachWorker``) and this policy holds NO GPU cuRobo -- it computes scene+targets on the
            CPU and ships them. Plan-0 BLOCKS on the worker's first solve (establishes the arm count +
            assignment AND is the single warmup); every later replan only SUBMITS (non-blocking, coalesced) +
            POLLS. Requires ``robot_name`` + ``scenario_name`` (the worker rebuilds the descriptor in-child
            via ``robot_scene`` -- the descriptor's ``build_entity_spec`` lambda cannot cross a spawn pickle).
        robot_name / scenario_name: ``CFG_BY_ROBOT`` key + scenario name; only needed for ``async_replan``
            (worker spawn). The sync path ignores both.

    The executor + controlled-joint set are built LAZILY on the first successful route, because the arm
    count is unknown until then; a later re-solve must keep the SAME active set (asserted). ONE executor over
    all controlled joints in BOTH modes: the async worker returns a full coherent bimanual route (both arms
    solved together), so there is no per-side split -- the async and sync install path is identical."""

    def __init__(self, session, scenario, mj_model, control_dt: float,
                 replan_interval_s: float | None = None, async_replan: bool = False,
                 mpc_track: bool = False,
                 robot_name: str | None = None, scenario_name: str | None = None,
                 worker=None, gravity_comp: bool = True, gravity_comp_alpha: float = 1.0,
                 output_filter: bool = False, traj_omega: float = 15.0,
                 max_tracker_correction_rad: float | None = None,
                 walk: bool = False, walk_device: str = "cpu",
                 height_pad: float = 0.02) -> None:
        self._session = session
        self._scenario = scenario
        self._mj_model = mj_model
        self._height_pad = float(height_pad)
        # Async / MPC hold NO cuRobo: pass session=None and derive the CPU descriptor from robot_scene (the
        # same per-robot table/z-floor adjustment the worker rebuilds in-child), so this policy CANNOT touch a
        # GPU runtime -- the sole cuRobo lives in the spawned worker. Sync replan needs the in-process session.
        if session is not None:
            self._cfg = session.robot_cfg
        else:
            assert (async_replan or mpc_track) and robot_name is not None and scenario_name is not None, (
                "session=None is only valid for async_replan / mpc_track with robot_name + scenario_name")
            self._cfg = robot_scene(robot_name, scenario_name)[0]
        self._control_dt = float(control_dt)

        self._executor: DynamicArmReferenceExecutor | None = None
        self._controlled: list[str] = []
        self._template = None
        self._replan_interval_steps = (
            None if replan_interval_s is None else max(1, round(replan_interval_s / control_dt)))
        self._k = 0
        self._route = None
        self._viz_route = None       # route the reach overlay draws; spans EXTEND+RETRACT (see ``viz_route``)
        self._adr_cache: dict[str, int] = {}
        self.replan_failures = 0
        self._async = async_replan
        self._mpc_track = mpc_track
        self._robot_name = robot_name
        self._scenario_name = scenario_name
        # async/MPC: the ONE spawned cuRobo runtime. A caller (e.g. the viewer running many back-to-back
        # reaches) may INJECT a persistent worker so the ~7 s cold cuRobo/CUDA warmup is paid once, not per
        # reach; an injected worker is NOT closed here (the injector owns its lifetime).
        self._worker = worker
        self._own_worker = worker is None
        # ``jacobian`` is existing baseline. ``mink`` is opt-in local-QP A/B: cuRobo still owns
        # plan-0/world collision and physical latch still owns success. Do not promote a tracker from
        # headless latch samples alone; native viewer behavior must agree.
        self._tracker_backend = os.environ.get("DYNAMIC_REACH_TRACKER", "jacobian").lower()
        if self._tracker_backend not in {"jacobian", "mink"}:
            raise ValueError("DYNAMIC_REACH_TRACKER must be 'jacobian' or 'mink'")
        self._tracker: JacobianReachTracker | MinkReachTracker | None = None
        self._gravity_comp = bool(gravity_comp)
        self._gravity_comp_alpha = float(gravity_comp_alpha)
        self._output_filter = bool(output_filter)
        self._traj_omega = float(traj_omega)
        self._max_tracker_correction_rad = max_tracker_correction_rad
        self._req_gen = 0
        assert not (async_replan and mpc_track), "async_replan and mpc_track are mutually exclusive paths"
        if async_replan:
            assert robot_name is not None and scenario_name is not None, (
                "async_replan needs robot_name + scenario_name (worker rebuilds the descriptor in-child)")
            assert replan_interval_s is not None, "async_replan needs a replan cadence"
        if mpc_track:
            assert robot_name is not None and scenario_name is not None, (
                "mpc_track needs robot_name + scenario_name (worker rebuilds the descriptor in-child)")

        # ---- walk-search (base-live driving) ------------------------------------------------------
        # When ``walk``, ``step`` runs a CENTRALIZED body FSM (one visit = one pick cube): DISCOVER ->
        # APPROACH -> REACH -> (``_select_next``) -> ... -> TERMINATE. APPROACH DRIVES the base toward the
        # current cube until it is arm-reachable (a PROXIMITY gate -- base-to-cube planar distance enters the
        # reach envelope -- + the ``HeuristicMovingPolicy`` brain LIFTED from the pickplace mission; the learned
        # reachability MLP was removed as obstacle-blind, cuRobo is the real feasibility check in REACH), THEN
        # REACH runs the SAME reach dispatch used by parked reach and retracts home. State is written ONLY by
        # ``_transition``; handlers RETURN their next state.
        # ``walk=False`` leaves ``step`` byte-identical to the parked path.
        #
        # Enrollment is OPPORTUNISTIC: ``_pending`` is the SET of pick-cube names (you cannot pre-order an
        # UNSEEN cube by pose). Each visit takes WHICHEVER pending cube is currently in the camera-gated
        # belief (privileged: all at once -> nearest-first; camera: whatever SEARCH uncovers). When no pending
        # cube is visible, SEARCH brings one into view: the FIXED-camera baseline (g1, v2_fixed) must YAW THE
        # BASE toward a known support anchor (a real reorientation cost, counted in T̄); the ACTUATED-gimbal
        # robot (v2) needs no base turn -- the harness gate already aims its gimbals over their full ROM at
        # ``observe`` time, so a cube within ROM is seen at INIT and one beyond ROM cannot be revealed by any
        # policy slew (that asymmetry IS the paper thesis). The per-visit REACH extends on ROUTE PROGRESS (the
        # executor's own trajectory clock) then retracts along a synthetic home trajectory -- both dt-agnostic
        # and sim-handle-free (the route clock is the policy's own plan, not a realized-error probe).
        self._walk = bool(walk)
        # Base-motion capabilities (REACH_FSM_REDESIGN.md): gate the look/reach REMEDIES and the FAIL-loud
        # verdicts. Walk can turn+drive; the parked path cannot (it enters at EXTEND and never consults these,
        # so parked stays byte-identical). Latent for Stage 3 (walk always True); wires the Stage-5 parked fold.
        self._can_turn = self._can_drive = self._walk
        # Does the base have to REFACE the cube before a reach? Only a SINGLE forward camera
        # (``walk_visibility_dirs`` len 1, e.g. g1) does -- to keep the cube inside its fixed frustum for the
        # visibility gate, NOT for reachability. Arm reach is base-facing-independent (cuRobo proves reach
        # from the actual stance in EXTEND); left_right_close proves in-place bimanual LATERAL reach with
        # zero base turn. Twin/gimbal-camera humanoids (v2/v2_fixed, len 2) see all around, so squaring the
        # base to ONE cube is pure waste AND skews a bimanual stance away from the other cube.
        self._needs_reface = (self._cfg.needs_reface_override if self._cfg.needs_reface_override is not None
                              else len(self._cfg.walk_visibility_dirs) == 1)
        self._mover = None
        self._phase = None                             # unified mission state; None until the first tick seeds it
        self._pending: set = set()                     # pick-cube names not yet visited (opportunistic set)
        self._failed: dict[str, str] = {}              # cube -> give-up reason (unseen / unreachable); FAIL verdict
        self._fallen: set = set()                      # cubes physics shoved off support (see ``note_fallen``)
        self._walk_current: str | None = None
        self._anchors_visited: set = set()             # support-anchor names already SEARCH-turned to (by name)
        self._anchors_served: set = set()              # support-anchor names that already yielded a completed target
        self._search_anchor: str | None = None         # anchor the current SEARCH-turn is aiming at
        self._search_anchor_pos = None                 # its world center (frozen while turning)
        self._search_anchor_standoff = _REACH_RADIUS_M  # brake distance that can FRAME it (_framing_standoff)
        self._phase_t = 0
        self._latch_dist: float | None = None          # base->cube dist captured at proximity latch; drives the extra close-in
        self._latch_cube_pose = None                    # last-seen belief for the current target cube; the reach COMMITS to it
        self._approach_anchor_pos = None                # fixed support (table/shelf/hand) nearest the current target cube;
        #   APPROACH TRIAL: drive/brake/back-off/square-up target this instead of the cube's own (jittered) pose
        self._approach_anchor_name = None               # anchor's name (paired with _approach_anchor_pos)
        self._approach_anchor_tangent = None            # anchor's world-XY unit tangent (point-C projection axis)
        self._approach_anchor_half_w = None              # anchor's half-width along tangent, or None if unbounded (hand)
        self._approach_support_normal_w = None           # bounded-support face normal, support -> robot at target selection
        self._from_discover = False                     # ACQUIRE was entered via DISCOVER's hand-off (see _handle_acquire)
        self._nav_target_w = None                       # current DISCOVER/APPROACH drive target, world XYZ (debug-viz only)
        self._reach_poses: dict = {}                     # {cube: committed pose} for EVERY cube assigned this visit (bimanual)
        self._reach_anchor_names: dict = {}              # committed cube -> nearest support-anchor name
        self._forced_assignment: dict | None = None   # {side: cube} pinned to the gate winner(s) during a walk visit
        # Bug-4 occupied-arm ledger {arm_side: cube}: an arm holding a cube (jaws closed since GRASP) is
        # NEVER reassigned to a new grasp. Set at GRASP from the extend route's assignment, cleared only by
        # ``release_grasp`` (a place controller put the object down) or ``reset``. The assignment path
        # (``_reachable_cubes`` + ``_handle_approach``'s arm pick) excludes held cubes and occupied arms.
        self._held: dict[str, str] = {}
        self._retract_uses_tracker = False             # MPC return follows Cartesian tracker, fallback uses executor
        self._plan0_pending = False                    # initial reach plan-0 submitted, awaiting the async worker (mpc path):
        #   hold planning-home while cuRobo solves off-loop during EXTEND (real-robot planner contract)
        self._plan0_scene = None                        # exact object frames paired with pending plan-0 request
        self._plan0_cube_pose = None                     # observed_cube_pose snapshot at submit time (staleness check)
        self._plan0_infeasible = 0                      # consecutive cuRobo infeasible plan-0 verdicts (walk EXTEND guard)
        self._plan0_gen: int | None = None              # generation of the outstanding submit_plan0 request (mpc path);
        #   passed to poll_plan0 so a reply orphaned by reset()/a superseded submit is dropped, not misinstalled
        self._retract_worker = None                    # dedicated async worker for non-MPC return plans
        self._retract_request_gen: int | None = None
        self._retract_plan_ticks = 0            # RETRACT_PLAN wall-tick counter (worker-return backstop)
        self._acquire_ticks = 0                 # ACQUIRE wall-tick counter (gaze-lock backstop)
        self._acquire_gaze_wait_ticks = 0       # diagnostic: ticks blocked only by gimbal fixation
        self._acquire_level_wait_ticks = 0      # retained zero diagnostic for metrics compatibility; no level wait
        self._gripper_closed: set[str] = set()         # arm sides whose jaws hold a grasp; per-arm so a held cube
        #   in one hand never forces the other jaw shut (that would block grasping the next object)
        self._physical_grasps: dict[str, str] | None = None  # completed dynamic welds from prior physics tick
        self._grasp_capture_retries: dict[str, int] = {}
        self._grasp_gravity_ref = None                 # normalized IMU gravity at current close-stability window start
        self._grasp_gravity_elapsed = 0.0              # elapsed time in the current gravity-change window
        self._grasp_gravity_max_delta = 0.0            # largest gravity movement inside that window
        self._grasp_stable_wait = 0.0                  # total close-stability wait (timeout guard)
        self._final_replan_pending = False              # async collision-aware refresh of an invalidated reach route
        self._final_replan_scene = None                 # scene paired with the pending worker request
        self._final_replans = 0                          # bounded refresh count for one committed target set
        self._pregrasp_done = False                      # two-stage EXTEND: stage A (pre-grasp standoff) landed
        # ARM HOLD: the configuration the arm is PARKED at between reaches, instead of the harness's default
        # planning-home fill. Two producers -- the rear prime below, and the end of RETRACT, which keeps the
        # pose cuRobo planned rather than paying a second move back to planning home. ``_arm_hold`` is the
        # mjlab command ``_dispatch`` merges UNDER every tick's ``arm_ref``; ``_arm_hold_q`` is the SAME
        # configuration in cuRobo cspace, which ``_step_mpc`` seeds plan-0 with -- seeding at a configuration
        # the arm is not in is the bug that dropped the carried cube. Both survive ``_commit_reach``,
        # ``_begin_visit`` and the infeasible-retry bounce; both die at ``_install_reach``, where the tracker
        # takes the arm over. Producers MERGE into them per joint, so priming one arm cannot silently release
        # the other back to planning home.
        self._arm_hold: dict | None = None               # {joint: mjlab qpos} merged under every tick
        self._arm_hold_q: dict | None = None             # same configuration in cspace -> plan-0 ``seed_q``
        self._arm_ref_prev: dict = {}                    # last EMITTED arm command; anchors the slew gate
        # REAR PRIME (see ``_REAR_PRIME_BEARING_DEG``): one attempt per target visit.
        self._prime_gen: int | None = None               # outstanding prime solve generation (None = idle)
        self._prime_executor: DynamicArmReferenceExecutor | None = None   # NEVER ``self._executor``:
        #   ``_install``'s stable-controlled-set assertion must not see the prime's one-arm set
        self._prime_template = None
        self._prime_controlled: list[str] = []
        self._prime_tried = False
        self._extend_diverge_ticks = 0                    # consecutive over-threshold ticks, mid-route abort
        self._target_moved_replans = 0                   # bounded re-plans after a committed cube was pushed
        self._extend_diverge_replans = 0                 # bounded mid-route re-plans for one committed target set
        # Mid-route abort applies only to a tracker whose correction is clipped to the nominal-route tube.
        self._extend_diverge_guard = self._cfg.dynamic_tracker_transit_max_correction_rad is not None
        self._extend_ik_ok: bool | None = None          # EXTEND fast-IK gate verdict for the current commit
        #   (None = not yet checked this commit; cached so the collision-blind pre-filter runs ONCE per EXTEND)
        self._square_ticks = 0                       # ticks spent in APPROACH square-up (timeout guard)
        self._extend_no_route_ticks = 0                # walk EXTEND ticks with no route (infeasibility bound)
        self._side_stance_retries = 0                  # bounded-support tangent stances tried for this target
        self._side_stance_target_w = None              # world XY stance target, reached by a bounded raw side step
        self._side_stance_elapsed = 0.0
        self._curated_stance_probe_w = None            # candidate waiting on sole dynamic worker's legacy IK
        self._curated_stance_ik_rejected = False       # do not resubmit same field candidate after a false probe
        # cube -> world XY stances already occupied for it, surviving _begin_visit (see _STANCE_REVISIT_TOL_M).
        self._spent_stances: dict[str, list[np.ndarray]] = {}
        # Pre-plan field stance treatment. Candidate points are field body-frame grasp points inverse-mapped
        # to fixed-yaw world base positions. They are screened before any locomotion, so a rejected point never
        # creates speculative walking. ``None`` target means use legacy Point-C unchanged.
        self._field_stance_candidates = []             # [(world_base_pos, target_yaw_quat, arm, robust_score)]
        self._field_stance_candidate_i = 0
        self._field_stance_probe = None                # dynamic worker request currently in flight
        self._field_stance_target_w = None             # fast-IK-approved base target being walked to
        self._field_stance_target_quat = None           # expected yaw at field target (V2 keeps current yaw)
        self._field_stance_assignment = None           # arm pinned by the field candidate through plan-0
        self._field_stance_current_score = None
        self._stance_turn_retries = 0                  # visible arm-workspace headings tried after a no-route stance
        self._stance_turn_target_w = None              # fixed world point used by the normal TURN->STOP mover path
        self._stance_turn_probe_w = None               # turn-heading candidate awaiting the dynamic worker's fast-IK
        self._extend_retries = 0                        # back-off-and-retry attempts spent on the current cube
        self._extra_standoff = 0.0                      # extra APPROACH standoff (m) accumulated by EXTEND retries
        self._parked_replan_wait = 0                   # ticks spent parked with no route (throttle + timeout)
        self._parked_replan_warned = False             # one-shot "parked infeasible" warning latch
        self._grasp_close_elapsed = 0.0                # GRASP dwell timer while the position servo closes the jaws
        self._held_grasp_command: dict | None = None   # exact final arm ref held across close/async-plan seam
        self._grasp_route = None                        # completed extend route, retained for parked verdict after home
        self._parked_poses: dict = {}                  # last seen cube poses, retained through close/retract
        self._parked_done = False                       # parked FSM terminal: do not replan after returning home
        self._warned_no_route = False                  # warn-once latch: committed cube whose first plan found no route
        # POLICY-OWNED perception (seam inversion, REACH_FSM_REDESIGN.md): the gaze controller commands the
        # gimbals (returned as ``camera_command``); the harness renders that aim and reports what it frames,
        # which the policy accrues into ``_belief``. Actuated robots only (fixed cams have static extrinsics
        # the harness pins); ``_belief`` is the last-seen memory every phase reasons over.
        self._belief: dict = {}
        # LIVENESS bookkeeping for the pre-commit gates (see ``_reachable_cubes``). ``_belief_tick`` stamps
        # when each cube was last framed LIVE; ``_live`` is this tick's raw detections, retained because the
        # simultaneity gate needs same-tick co-visibility, which the accrued ``_belief`` has already erased.
        # INVARIANT: written together with ``_belief`` in ``_drive_gaze`` and cleared together in ``reset``,
        # so ``_belief_tick`` always keys every ``_belief`` entry.
        self._belief_tick: dict = {}
        self._live: dict = {}
        self._commit_stale_tick = None      # tick of the FIRST of the current run of stale-target declines
        self._ticks = 0                                 # control ticks since construction (pre-act hold clock)
        self._gaze = _GazeController(mj_model, self._cfg, control_dt) if self._cfg.gaze_cams else None
        # ONE dispatch table for the flat FSM: state -> handler; each handler returns the next state (None =
        # stay). Parked and walk share it (parked simply never visits DISCOVER/APPROACH).
        self._handlers = {
            _DISCOVER: self._handle_discover,
            _APPROACH: self._handle_approach,
            _EXTEND: self._handle_extend,
            _GRASP: self._handle_grasp,
            _RETRACT_PLAN: self._handle_retract_plan,
            _RETRACT: self._handle_retract,
        }
        self._handlers[_ACQUIRE] = self._handle_acquire       # perception-first decide hub (parked + walk)
        if self._walk:
            from tasks.visual_manipulation.moving_policy import HeuristicMovingPolicy
            # Drive directions are independent of camera coverage. G1's sole +X direction makes it turn
            # toward a target before walking, keeping its fixed forward camera on target. V2-fixed keeps
            # +X/-X; actuated V2 adds +/-Y because its gimbals track target at every body bearing.
            # Per-robot vy cap (RobotDescriptor.walk_cruise_vy_max) lets g1 strafe within a bounded band
            # toward off-axis cubes; defaults to 0.0 (legacy) when not set on a robot's descriptor.
            mover_kwargs: dict = {"step_dt": self._control_dt, "drive_dirs": self._cfg.walk_drive_dirs}
            if getattr(self._cfg, "walk_cruise_vy_max", None) is not None:
                mover_kwargs["cruise_vy_max"] = float(self._cfg.walk_cruise_vy_max)
            # Env override outranks the descriptor: sets ``WALK_TURN_CRUISE=1`` to force on, ``=0`` to
            # force off regardless of the per-robot default. Use for A/B on a single run.
            cruise_env = os.environ.get("WALK_TURN_CRUISE")
            if cruise_env is not None:
                mover_kwargs["turn_cruise"] = cruise_env not in ("0", "false", "False", "")
            elif getattr(self._cfg, "walk_turn_cruise", False):
                mover_kwargs["turn_cruise"] = True
            # EXPERIMENTAL (2026-07-26, env-gated, off by default): WALK-only proximity taper, testing
            # the hypothesis that g1's overshoot-then-backoff is fixable by slowing an ALREADY-established
            # gait near the target, distinct from the documented cold-start dead zone. Not yet validated
            # or made default -- see MEMORY.md.
            taper_env = os.environ.get("WALK_TAPER_DIST_M")
            if taper_env:
                mover_kwargs["taper_dist_m"] = float(taper_env)
            self._mover = HeuristicMovingPolicy(**mover_kwargs)

    @property
    def controlled_joint_names(self) -> list[str]:
        """Active arm joints (mjlab names) the policy commands; empty until the first route lands."""
        return list(self._controlled)

    @property
    def last_route(self):
        """Most recent ``ReachRoute`` (per-frame grasp candidates + planned reach_error for the verdict)."""
        return self._route if self._route is not None else (
            self._grasp_route if not self._walk else None)

    @property
    def grasp_route(self):
        """Extend route that reached the object, retained after parked home retract for grading."""
        return self._grasp_route

    @property
    def parked_done(self) -> bool:
        """Parked reach has closed, returned home, and now holds its grasp width."""
        return not self._walk and self._parked_done

    @property
    def plan0_pending(self) -> bool:
        """True while the initial reach plan-0 is in flight on the async worker (mpc path): ``step`` returns a
        hold and the arm has not started reaching. Lets a fixed-rate driver (``run_reach``) keep stepping
        physics without spending the reach budget on the off-loop solve -- the real-robot planner contract."""
        return self._plan0_pending

    @property
    def planner_pending(self) -> bool:
        """True while the mission is STALLED on an off-loop cuRobo solve: the policy holds a fixed pose and
        the rollout makes no task progress until the reply lands. Grading subtracts these ticks from reported
        sim time, so a mission's measured duration matches what a SYNCHRONOUS planner would have produced
        (solve returns inside the tick, costing zero sim time). Without it, the async worker's WALL latency
        leaks into a sim-time metric: the number of ticks spent waiting is ``solve wall time / wall time per
        tick``, so the same mission measures differently on a loaded GPU (observed: one cell's Tbar moved
        61.96 s -> 56.05 s between a 1-worker and a 3-worker sweep, same code and seeds).

        Two stalls qualify.  ``_plan0_pending`` holds planning-home awaiting the initial reach route;
        ``_RETRACT_PLAN`` holds the closed grasp awaiting the grasp->home route.  ``_final_replan_pending``
        does NOT: the arm keeps tracking its installed route while that refresh solves, so those ticks are
        productive motion and stay in the measurement."""
        return self._plan0_pending or self._phase == _RETRACT_PLAN

    @property
    def viz_route(self):
        """Route the reach OVERLAY should draw -- the planned reach whose waypoint frames are on screen.
        Distinct from ``last_route`` (the grade datum, nulled the instant retract starts): ``viz_route``
        SPANS both reach sub-phases -- EXTEND and RETRACT. Held as the installed plan across the
        extend->retract switch (``_install`` re-points it to the RETRACT plan when the retract installs), so
        the viewer draws exactly ONE waypoint set at a time (reach plan while reaching, retract plan while
        retracting), never two overlaid. Cleared when the visit's retract completes (no active plan during
        DISCOVER/APPROACH) and at ``reset``. ``None`` = no plan drawn."""
        return self._viz_route

    @property
    def walk_done(self) -> bool:
        """Walk-search finished all visits (all observed cubes serviced). Always False on the parked path;
        lets a driver end the rollout once the mission completes instead of running a fixed budget."""
        return self._walk and self._phase == _TERMINATE

    @property
    def mission_verdict(self):
        """Explicit end-of-mission verdict (REACH_FSM_REDESIGN.md invariant: no silent hold). ``None`` until
        the walk mission reaches TERMINATE, then ``("SUCCESS", None)`` if every pick cube was grasped, else
        ``("FAIL", {cube: reason})`` naming each cube given up (``unseen`` = no viewpoint framed it;
        ``unreachable`` = the planner found no route from any tried stance). ``None`` off the walk path."""
        if not self._walk or self._phase != _TERMINATE:
            return None
        return ("FAIL", dict(self._failed)) if self._failed else ("SUCCESS", None)

    @property
    def is_reaching(self) -> bool:
        """In a reach state (EXTEND/GRASP/RETRACT_PLAN/RETRACT) -- the arm is executing a grasp+home visit.
        A driver keys the per-visit realized-error grade on this (replaces the old ``walk_phase == 'REACH'``
        check, since REACH is now the four explicit reach states)."""
        return self._phase in _REACH_STATES

    @property
    def walk_current(self) -> str | None:
        """The cube the current walk visit targets (``None`` off the walk path / before INIT). Lets a headless
        driver key the per-visit realized-error grade."""
        return self._walk_current

    @property
    def walk_phase(self) -> str | None:
        """Current mission FSM state (``DISCOVER``/``APPROACH``/``EXTEND``/``GRASP``/``RETRACT_PLAN``/
        ``RETRACT``/``TERMINATE``); ``None`` off the walk path or before the first tick selects a target."""
        return self._phase if self._walk else None

    @property
    def nav_target_w(self):
        """Current DISCOVER/APPROACH drive target, world XYZ (``_search_anchor_pos`` or ``_point_c``) --
        stale (holds the last visit's target) once EXTEND/GRASP/RETRACT starts, since only DISCOVER/
        APPROACH write it. Debug-viz only (nav arrow); no FSM logic reads this."""
        return self._nav_target_w

    def note_fallen(self, fallen) -> None:
        """Condemn pick cubes the driver's harness reports as knocked off their support.

        A cube on the floor is unrecoverable -- the weld trigger needs finger contact within
        ``_WELD_TRIGGER_DIST_M`` of a cube standing where the reach was planned. Without this the mission
        spends both physical-latch attempts on it and then reports ``physical grasp latch did not engage``,
        or ``unreachable after 2 back-off retries`` once the re-plan cannot route to the floor. Recording it
        here makes the verdict name the ACTUAL cause and drops the cube from ``_pending`` so no further visit
        re-approaches it.

        Already-``_held`` cubes are ignored: a completed physical grasp is never retroactively revoked (the
        harness excludes welded cubes anyway; this is the second line of defence for a released-then-placed
        cube). Only a physics harness supplies this -- the kinematic path never calls it, so its grading is
        byte-identical.
        """
        for cube in fallen:
            if cube in self._held or cube in self._fallen:
                continue
            self._fallen.add(cube)
            self._failed.setdefault(cube, "cube knocked off support")
            self._pending.discard(cube)

    def release_grasp(self) -> None:
        """Open jaws after a place controller has put down the held object.

        Reach/retract never releases by itself: that would drop a successful grasp at home. The caller that
        owns a later place waypoint must call this only after its placement condition is satisfied. ``reset``
        also opens jaws for the next episode. Clears the occupied-arm ledger (the freed arm can grasp again).
        """
        self._gripper_closed = set()
        self._held = {}
        self._physical_grasps = None
        self._grasp_capture_retries = {}

    @property
    def gripper_closed(self) -> set[str]:
        """Per-arm jaw command owned by the shared FSM: the set of arm sides ("L"/"R") whose jaws hold a
        grasp position (through retract, open on place/reset). Empty = both open. Per-arm so an arm carrying
        an earlier cube keeps its grip while the free arm opens to grasp the next object."""
        return self._gripper_closed

    @property
    def completed_grasps(self) -> dict[str, str]:
        """Physical captures that completed the required closed-jaw hold.

        A side enters this ledger only at the end of ``GRASP`` after its contact latch was observed. Evaluation
        uses it as the reach-and-grasp completion event; subsequent home retraction is safety cleanup, not a
        prerequisite for a completed task."""
        return dict(self._held)

    def _qpos_adr(self, joint: str) -> int:
        """``qposadr`` for a bare cuRobo joint name, suffix-resolved against the model (the env's model
        namespaces joints as ``robot/<name>``; the standalone kinematic model uses bare names). Cached per
        name. Same id-based resolution ``CuroboArmPlanner`` uses."""
        if joint not in self._adr_cache:
            adr = None
            for jid in range(self._mj_model.njnt):
                n = self._mj_model.joint(jid).name
                if n == joint or n.split("/")[-1] == joint:
                    adr = int(self._mj_model.jnt_qposadr[jid])
                    break
            assert adr is not None, f"route joint {joint!r} not found in model"
            self._adr_cache[joint] = adr
        return self._adr_cache[joint]

    def _ref_offset(self, joint: str) -> float:
        """``qpos0[qposadr]`` for a route joint (the ref-fold offset)."""
        return float(self._mj_model.qpos0[self._qpos_adr(joint)])

    def _extend_goal(self) -> str:
        """cuRobo ``goal`` for any solve submitted on behalf of the current EXTEND -- plan-0 and every
        mid-route replan share it, so a replan can never re-target the stage the visit has already left.

        Single-stage (``_TWO_STAGE_EXTEND`` off) is always ``"reach"``. Two-stage returns ``"pregrasp"``
        until stage A lands and ``"descend"`` after, which also fixes the replan-during-stage-B case: a
        TARGET-MOVED or divergence re-solve inside the descent stays linear-motion constrained instead of
        silently reverting to the unconstrained trajopt that loops around the cube."""
        if not _TWO_STAGE_EXTEND:
            return "reach"
        return "descend" if self._pregrasp_done else "pregrasp"

    def _seed_from_proprio(self, proprio_qpos) -> dict[str, float]:
        """cuRobo-cspace warm seed ``{joint: q}`` = the CURRENT arm config, inverting the ref fold
        (``q_curobo = qpos - qpos0``, the reverse of ``_route_to_mjlab``) over the active route joints.
        Feeds a base-live re-solve so the new trajectory STARTS where the arm is and the IK stays on the
        current redundancy branch (else an unseeded re-solve flips elbow up<->down = a violent jump)."""
        names = self._route.joint_names
        if isinstance(proprio_qpos, dict):
            return {j: float(proprio_qpos[j] - self._ref_offset(j)) for j in names}
        return {j: float(proprio_qpos[self._qpos_adr(j)] - self._ref_offset(j))
                for j in names}

    def _route_to_mjlab(self, route) -> tuple[list[str], np.ndarray]:
        """cuRobo cspace route -> mjlab qpos route: add ``qpos0[qposadr]`` per route joint (ref fold)."""
        offsets = np.array([self._ref_offset(j) for j in route.joint_names], dtype=np.float32)
        return list(route.joint_names), route.route_q_curobo + offsets[None, :]

    def _replan_due(self) -> bool:
        if self._route is None:
            return True
        return self._replan_interval_steps is not None and self._k % self._replan_interval_steps == 0

    def _install(self, route) -> None:
        """Load a fresh route into the executor, building it lazily on the first route (the arm count is
        unknown until then). A later re-solve MUST keep the same active-arm set (invariant: the
        reachability-driven count is stable once the reach is underway).

        Resumes a re-solve at the accumulated cursor (``start_s=cursor_s``), NOT 0. The re-solve is
        warm-seeded from the CURRENT arm config so ``route[0]`` is where the arm is AND the IK stays on the
        current redundancy branch (no elbow flip). The accumulated cursor then rides that fresh
        current->goal route forward through its ease-in->cruise->goal profile: cursor grows toward the
        route duration and the arm converges. Loading at ``start_s=0`` instead restarts every re-solve from
        REST (cuRobo trajectories ease in from zero velocity), so the arm never leaves the ease-in region
        and crawls -- verified 0.51 m stall. The seed kills the FLIP; the cursor carries the PROGRESS."""
        names_mjlab, route_mjlab = self._route_to_mjlab(route)
        controlled = list(route.controlled_joints)
        if self._executor is None:
            self._controlled = controlled
            self._template = torch.zeros(1, len(controlled))
            self._executor = DynamicArmReferenceExecutor(controlled, self._control_dt)
        else:
            assert controlled == self._controlled, (
                f"active-arm set changed mid-reach {self._controlled} -> {controlled}; the "
                "reachability-driven count must be stable after the first plan")
        start_s = 0.0 if self._route is None else self._executor.cursor_s
        self._executor.load(names_mjlab, route_mjlab, route.interpolation_dt, start_s=start_s)
        self._route = route
        self._viz_route = route     # overlay follows the installed plan (held through the coming retract)
        self._clear_arm_hold()         # tracker owns the arm now; stop re-emitting the rear-prime hold

    # -- async path: ONE cuRobo runtime in a spawned process; main holds NO cuRobo ----------------------
    def _submit_reach(self, yaw_base, cube_poses, seed_q, fixed_assignment) -> int | None:
        """Build scene+targets on the CPU (no GPU) and submit ONE reach request to the worker process.
        Returns the request generation (to await plan-0), or ``None`` if nothing is observed (skip)."""
        base_pos, base_quat = yaw_base
        scene_dict, targets = reach_scene_and_targets(
            self._cfg, self._scenario, self._mj_model, yaw_base, cube_poses=cube_poses)
        if not targets:
            return None
        gen = self._req_gen
        self._req_gen += 1
        self._worker.submit(ReachRequest(
            gen, scene_dict, targets, np.asarray(base_pos, dtype=np.float32),
            np.asarray(base_quat, dtype=np.float32), seed_q,
            None if fixed_assignment is None else _REPLAN_MAX_ATTEMPTS, fixed_assignment))
        return gen

    def close(self) -> None:
        """Stop the spawned cuRobo worker (idempotent). Call after the rollout in async mode. A worker
        INJECTED by the caller is left running (the caller owns its lifetime -- persists across reaches)."""
        if self._worker is not None and self._own_worker:
            self._worker.close()
            self._worker = None
        if self._retract_worker is not None:
            self._retract_worker.close()
            self._retract_worker = None

    def reset(self) -> None:
        """Clear per-reach state so the NEXT ``step`` re-plans from scratch, for a viewer/harness that
        RESTARTS the episode WITHOUT rebuilding the policy (e.g. phase-4's single-policy viewer on a
        reset keypress / episode rollover). Drops the route + executor + tick counter, so the next tick
        re-runs plan-0 (mpc_track: the next ``plan0`` re-``rebind``s the Jacobian tracker, re-seeding its
        cursor at home; sync/async: ``_install`` sees ``_route is None`` -> ``start_s=0`` and replays the
        approach from home). The spawned worker + tracker object are KEPT (persistent runtime; only
        ``close`` tears the worker down, and ``rebind`` re-points the tracker on the next route). Robust to
        the reachable-arm count changing across episodes: the executor/tracker rebuild on the next route,
        so ``_controlled`` may differ."""
        self._route = None
        self._viz_route = None
        self._executor = None
        self._controlled = []
        self._template = None
        self._k = 0
        self._forced_assignment = None
        self._warned_no_route = False
        self._latch_dist = None
        self._approach_anchor_pos = None
        self._approach_anchor_name = None
        self._approach_anchor_tangent = None
        self._approach_anchor_half_w = None
        self._approach_support_normal_w = None
        self._from_discover = False
        self._nav_target_w = None
        self._grasp_route = None
        self._parked_poses = {}
        self._parked_done = False
        self._phase, self._phase_t = None, 0           # next tick re-seeds the FSM (parked -> EXTEND, walk -> select)
        self._gripper_closed = set()
        self._held = {}
        self._retract_uses_tracker = False
        self._plan0_pending = False                    # next reach re-submits plan-0 async from scratch
        self._plan0_scene = None
        self._plan0_cube_pose = None
        self._plan0_infeasible = 0
        self._plan0_gen = None
        self._final_replan_pending = False
        self._final_replan_scene = None
        self._final_replans = 0
        self._pregrasp_done = False
        self._clear_arm_hold()
        self._arm_ref_prev = {}        # new episode: nothing to rate-limit the first command against
        self._extend_diverge_ticks = 0
        self._extend_diverge_replans = 0
        self._target_moved_replans = 0
        self._grasp_gravity_ref = None
        self._grasp_gravity_elapsed = self._grasp_stable_wait = 0.0
        self._grasp_gravity_max_delta = 0.0
        self._extend_ik_ok = None
        self._square_ticks = 0
        self._extend_no_route_ticks = 0
        self._side_stance_retries = 0
        self._side_stance_target_w = None
        self._side_stance_elapsed = 0.0
        self._curated_stance_probe_w = None
        self._curated_stance_ik_rejected = False
        self._spent_stances = {}       # per-episode, NOT per-visit -- that is the whole point
        self._field_stance_candidates = []
        self._field_stance_candidate_i = 0
        self._field_stance_probe = None
        self._field_stance_target_w = None
        self._field_stance_target_quat = None
        self._field_stance_assignment = None
        self._field_stance_current_score = None
        self._stance_turn_retries = 0
        self._stance_turn_target_w = None
        self._stance_turn_probe_w = None
        self._extend_retries = 0
        self._extra_standoff = 0.0
        self._parked_replan_wait = 0
        self._parked_replan_warned = False
        self._grasp_close_elapsed = 0.0
        self._held_grasp_command = None
        # Per-cube physical-latch retry budget (_GRASP_CAPTURE_MAX_RETRIES). Was NOT cleared here (bug,
        # 2026-07-25): in a warm --record-trials N chain this dict is never rebuilt, so a cube name that
        # missed-then-recovered in an EARLIER trial arrives at a LATER trial with its retry budget already
        # spent -- that trial's first (and only) miss on the same cube name then immediately exceeds
        # _GRASP_CAPTURE_MAX_RETRIES and fails outright instead of getting its own fresh retry. Same bug
        # class as the plan0-generation leak (persistent object, not reset per trial) but in a different
        # field; see MEMORY.md "Fixed regressions".
        self._grasp_capture_retries = {}
        self._retract_request_gen = None
        self._retract_plan_ticks = 0
        self._acquire_ticks = 0
        self._acquire_gaze_wait_ticks = 0
        self._acquire_level_wait_ticks = 0
        self._belief = {}
        self._belief_tick = {}
        self._live = {}
        self._commit_stale_tick = None
        if self._gaze is not None:
            self._gaze.reset()
        self._pending, self._failed = set(), {}    # both paths re-seed pending in step; parked now clears
        #                                             _failed too (it can FAIL-loud on a caps-off dead stance)
        self._fallen = set()
        if self._walk:
            self._walk_current = None
            self._anchors_visited, self._anchors_served = set(), set()
            self._search_anchor, self._search_anchor_pos = None, None
            self._reach_poses = {}
            self._reach_anchor_names = {}
            self._mover.retarget()

    def _step_async(self, base_pose, proprio_qpos, observed_cube_pose):
        """Async tick: the ONE cuRobo runtime lives in ``self._worker`` (a spawned process). Plan-0 BLOCKS
        (``wait``) -- it both establishes the arm count/assignment AND is the single warmup (the worker's
        first solve builds its cuRobo/CUDA graphs in-child). Every later replan only SUBMITS (non-blocking,
        coalesced) + POLLS, hot-swapping the single executor cursor-preserving. Same install path + failure
        contract (hold on ``None``) as the sync route; only WHEN the buffer reloads differs."""
        yaw_base = _yaw_only(base_pose)
        if self._route is None:
            # Parked infeasibility guard: throttle cuRobo resubmits and bound the wait so a parked
            # unreachable target doesn't storm the GPU for the full 2800-tick budget. Walk drives the base
            # to bring cubes into the envelope, so the gate is parked-only.
            if not self._walk:
                self._parked_replan_wait += 1
                if self._parked_replan_wait >= _PARKED_REPLAN_TIMEOUT_TICKS:
                    if not self._parked_replan_warned:
                        import warnings as _w
                        _w.warn(f"[reach] parked target infeasible from base after "
                                f"{_PARKED_REPLAN_TIMEOUT_TICKS} ticks; reporting MISS")
                        self._parked_replan_warned = True
                    self._parked_done = True                 # terminate parked FSM cleanly; verdict sees MISS
                    self._k += 1
                    return (0.0, 0.0, 0.0), {}, {}
                if self._parked_replan_wait % _PARKED_REPLAN_HOLD_TICKS != 0:
                    self._k += 1
                    return (0.0, 0.0, 0.0), {}, {}           # hold; don't resubmit yet
            if self._worker is None:
                self._worker = SpawnedReachWorker(self._robot_name, self._scenario_name)
            gen = self._submit_reach(yaw_base, observed_cube_pose, seed_q=None, fixed_assignment=None)
            if gen is not None:
                result = self._worker.wait(gen, timeout_s=180.0)   # plan-0 = the one-time warmup
                if result.route is not None:
                    self._install(result.route)
                    if not self._walk:
                        self._parked_replan_wait = 0          # success resets the wait
                        self._parked_replan_warned = False
        else:
            if self._replan_interval_steps is not None and self._k % self._replan_interval_steps == 0:
                self._submit_reach(yaw_base, observed_cube_pose,
                                   seed_q=self._seed_from_proprio(proprio_qpos),
                                   fixed_assignment=dict(self._route.assignment))
            for result in self._worker.poll():
                if result.route is None:
                    self.replan_failures += 1
                else:
                    self._install(result.route)
        self._k += 1
        if self._route is None:
            return (0.0, 0.0, 0.0), {}, {}
        arm_cmd = self._executor.command(self._template, self._controlled)[0].numpy()
        return (0.0, 0.0, 0.0), dict(zip(self._controlled, arm_cmd.tolist())), {}

    def _step_mpc(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """Reactive drift-servo tick (the ``mpc_track`` path; MPC removed -- see below). plan-0 runs ASYNC on
        the spawned cuRobo worker (``SpawnedReachMpcWorker``): SUBMIT once, POLL each tick, and HOLD
        planning-home until the collision-free reach trajectory (assignment + whole path) lands -- the control
        loop never blocks on cuRobo (real-robot planner contract; twin of ``_submit_retract``/``_poll_retract``,
        the SAME async seam the return plan already uses). That is the ONLY cuRobo use. The per-tick tracking is
        then a local MuJoCo
        local tracker -- baseline DLS or opt-in Mink QP -- toward the drift-corrected grasp, so this policy
        stays cuRobo-free (``session=None``) and control-loop latency remains bounded.

        WHY not MPC: the per-tick cuRobo MPC was a fixed ~16 ms/tick (collision rollout, un-tunable) that
        blew the 20 ms budget, and it was asked to reach a target ON the cube while treating the cube as a
        HARD obstacle to the same arm -- a contradiction that wedged the hand. The runtime job is only
        drift correction of an already-collision-free plan, which is one Jacobian step. The tracker seeds at
        the planned config for the current cursor and servos to ``pose_compose(objects_in_base[obj],
        planned_waypoint)``, marching home->grasp; no per-tick collision term,
        so no wedge (grasp contact is intended). Safety = the nominal path is collision-free + drift is small.

        Every tick ships only the OBSERVED object poses in base_link (``objects_in_base``, the sim sensor
        model -- NO base-in-world). ``last_route`` = the plan-0 route (its grasp candidates are the verdict
        target).

        Base frame: the FULL base pose (roll+pitch+yaw), NOT ``_yaw_only``. The arm executes on the actual
        tilted base, so planning/servoing in the upright frame lands the gripper at ``full_base (X) goal`` --
        off the cube by the tilt. Planning at the true base needs the grasp goalset held world-level (done in
        ``solve_reach_route`` via ``world_axes_in_base_quat``) so the tilt no longer flips it infeasible; the
        servo then re-measures the full base every tick and recomposes the goal, absorbing the (oscillating)
        lean live. Async/sync paths stay ``_yaw_only`` -- they plan at the upright home where full == yaw."""
        base_pos, base_quat = base_pose
        if self._final_replan_pending:
            # Base motion exceeded local Jacobian correction. Hold the old endpoint while cuRobo builds a
            # collision-checked route from measured joints; never force raw feedback through a limit branch.
            ready, route = self._worker.poll_plan0(self._plan0_gen)
            if ready:
                self._final_replan_pending = False
                scene_dict = self._final_replan_scene
                self._final_replan_scene = None
                if route is not None:
                    self._install_reach(route, scene_dict)
        if self._route is None:
            if self._worker is None:
                self._worker = SpawnedReachMpcWorker(self._robot_name, self._scenario_name)
            scene_dict, targets = reach_scene_and_targets(
                self._cfg, self._scenario, self._mj_model, base_pose, cube_poses=observed_cube_pose)
            if not self._plan0_pending:
                # SUBMIT plan-0 async and hold this tick -- never block the control loop on cuRobo. No target
                # (nothing observed) is a genuine hold, not a plan wait, so _plan0_pending stays False.
                # Parked infeasibility throttle + timeout (same contract as _step_async): walk drives the
                # base so the gate is parked-only.
                if targets:
                    if not self._walk:
                        self._parked_replan_wait += 1
                        if self._parked_replan_wait >= _PARKED_REPLAN_TIMEOUT_TICKS:
                            if not self._parked_replan_warned:
                                import warnings as _w
                                _w.warn(f"[reach] parked target infeasible from base after "
                                        f"{_PARKED_REPLAN_TIMEOUT_TICKS} ticks; reporting MISS")
                                self._parked_replan_warned = True
                            self._parked_done = True
                            self._k += 1
                            return (0.0, 0.0, 0.0), {}, {}
                        if self._parked_replan_wait % _PARKED_REPLAN_HOLD_TICKS != 0:
                            self._k += 1
                            return (0.0, 0.0, 0.0), {}, {}
                    self._plan0_gen = self._worker.submit_plan0(
                        scene_dict, targets, base_pos, base_quat,
                        fixed_assignment=self._forced_assignment, seed_q=self._arm_hold_q,
                        goal=self._extend_goal())
                    self._plan0_pending = True
                    self._plan0_scene = scene_dict
                    self._plan0_cube_pose = observed_cube_pose
            else:
                ready, route = self._worker.poll_plan0(self._plan0_gen)  # non-blocking (twin of _poll_retract)
                if ready:
                    # A ready-but-None route (infeasible) clears the flag; next tick re-submits -- the old
                    # blocking loop's resolve-until-feasible semantics, now off the control loop.
                    self._plan0_pending = False
                    plan_scene = self._plan0_scene
                    self._plan0_scene = None
                    submit_cube_pose = self._plan0_cube_pose
                    self._plan0_cube_pose = None
                    stale = route is not None and submit_cube_pose is not None and any(
                        cube in submit_cube_pose and cube in observed_cube_pose
                        and float(np.linalg.norm(np.asarray(observed_cube_pose[cube][0], dtype=float)
                                                  - np.asarray(submit_cube_pose[cube][0], dtype=float)))
                        > _PLAN0_STALE_DRIFT_M
                        for cube in route.assignment.values())
                    if stale:
                        # The cube this route targets moved past tolerance while the solve was in flight
                        # (slow/contended async solve let real wall-clock time pass) -- the landed route
                        # targets a pose the cube is no longer at. Not a reachability failure: resubmit
                        # fresh against the current pose instead of committing a route that would just
                        # discover infeasibility downstream and burn an _EXTEND_MAX_RETRIES attempt.
                        if targets:
                            self._plan0_gen = self._worker.submit_plan0(
                                scene_dict, targets, base_pos, base_quat,
                                fixed_assignment=self._forced_assignment, seed_q=self._arm_hold_q,
                                goal=self._extend_goal())
                            self._plan0_pending = True
                            self._plan0_scene = scene_dict
                            self._plan0_cube_pose = observed_cube_pose
                    elif route is not None:
                        assert plan_scene is not None, "plan-0 route returned without its submission scene"
                        self._install_reach(route, plan_scene)
                        self._plan0_infeasible = 0            # feasible solve: clear the infeasibility tally
                        if not self._walk:
                            self._parked_replan_wait = 0      # success resets the wait
                            self._parked_replan_warned = False
                    else:
                        # Count actual cuRobo INFEASIBLE verdicts (not wall-ticks): the walk EXTEND guard trips
                        # on this, so it is immune to plan-0 latency ballooning under GPU contention (a slow
                        # solve is not an infeasible one -- the old tick counter false-tripped good stances).
                        self._plan0_infeasible += 1
        self._k += 1
        if self._route is None:
            return (0.0, 0.0, 0.0), {}, {}
        objects_in_base = self._observe_objects_in_base(base_pose, observed_cube_pose)
        g_base = self._gravity_in_base(base_quat) if self._gravity_comp else None
        arm_reference = self._tracker.step(
            objects_in_base, gravity_in_base=g_base, actual_tool_poses_base=actual_tool_poses_base,
            measured_arm_qpos=proprio_qpos if isinstance(proprio_qpos, dict) else None)
        return (0.0, 0.0, 0.0), arm_reference, {}

    def _install_reach(self, route, scene_dict) -> None:
        """Bind the Jacobian tracker to a landed plan-0 ``route`` + its planned object poses and mark the reach
        live (``_route`` set -> ``step`` tracks instead of holding). ``scene_dict`` = the plan tick's
        ``reach_scene_and_targets`` output; its ``cuboid`` poses are the base-frame object anchors the tracker
        servos against. Extracted from ``_step_mpc`` so the async submit/poll body reads as the retract twin."""
        objects_plan = {name: (np.asarray(box["pose"][:3], dtype=float),
                               np.asarray(box["pose"][3:], dtype=float))
                        for name, box in scene_dict["cuboid"].items()}
        if self._tracker is None:
            if self._tracker_backend == "mink":
                self._tracker = MinkReachTracker(
                    self._mj_model, self._cfg, self._control_dt,
                    gravity_comp=self._gravity_comp, gravity_comp_alpha=self._gravity_comp_alpha,
                    max_measured_correction_rad=self._max_tracker_correction_rad,
                    transit_max_correction_rad=self._cfg.dynamic_tracker_transit_max_correction_rad,
                )
            else:
                self._tracker = JacobianReachTracker(
                    self._mj_model, self._cfg.tool_frames, self._control_dt,
                    gravity_comp=self._gravity_comp, gravity_comp_alpha=self._gravity_comp_alpha,
                    output_filter=self._output_filter, traj_omega=self._traj_omega,
                    max_measured_correction_rad=self._max_tracker_correction_rad,
                    transit_max_correction_rad=self._cfg.dynamic_tracker_transit_max_correction_rad,
                    resid_ki=self._cfg.dynamic_tracker_resid_ki)
        self._tracker.rebind(route, objects_plan)
        self._end_prime()           # tracker owns the arm now; stop re-emitting the rear-prime flip
        self._route = route
        self._viz_route = route
        self._controlled = list(route.controlled_joints)

    def _gravity_in_base(self, base_quat):
        """World gravity rotated into base_link = R(conj(base_quat)) @ g_world (the IMU seam: on hardware the
        IMU reports gravity in base directly). Feeds the tracker's gravity-comp feedforward so the arm's
        gravity droop is computed on the ACTUAL tilted base. Upright base -> world gravity unchanged."""
        g_world = np.asarray(self._mj_model.opt.gravity, dtype=float)
        conj = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]], dtype=float)
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, g_world, conj)
        return out

    def _grasp_base_stable(self, base_quat) -> bool:
        """True once IMU gravity changes little across one fixed window.

        Dynamic grasp targets are object-relative to the live full base, so a static bimanual carry lean is
        valid.  Closing during changing gravity is not: the base is still swaying underneath the tracker.
        A windowed gravity delta distinguishes those cases without assuming world-level posture.
        """
        gravity = self._gravity_in_base(base_quat)
        gravity /= np.linalg.norm(gravity)
        if self._grasp_gravity_ref is None:
            self._grasp_gravity_ref = gravity
            self._grasp_gravity_elapsed = 0.0
            self._grasp_gravity_max_delta = 0.0
        self._grasp_gravity_max_delta = max(
            self._grasp_gravity_max_delta, float(np.linalg.norm(gravity - self._grasp_gravity_ref)))
        self._grasp_gravity_elapsed += self._control_dt
        if self._grasp_gravity_elapsed < _GRASP_GRAVITY_WINDOW_S:
            return False
        stable = self._grasp_gravity_max_delta <= _GRASP_GRAVITY_DELTA_MAX
        if not stable:
            self._grasp_gravity_ref = gravity
            self._grasp_gravity_elapsed = 0.0
            self._grasp_gravity_max_delta = 0.0
        return stable

    def _observe_objects_in_base(self, full_base, observed_cube_pose):
        """SIM SENSOR MODEL -- the ONE place base pose is read on the MPC path. Returns
        ``objects_in_base = {name: (pos, quat_wxyz)}``, the base_link pose of every scene entity (grasp cubes
        + static obstacles), which the object-anchored tracker consumes each tick. Built the SAME way as
        plan-0 (``reach_scene_and_targets`` -> the base-frame ``scene_dict['cuboid']`` poses), so the names
        match ``frame_to_object``/the collision world exactly. ``full_base`` = the FULL base pose (matching the
        plan frame), so the servo target sits in the same frame the arm actually executes in (no tilt miss).

        On HARDWARE this is replaced by perception: the camera reports each object pose in base_link directly,
        so the base's world pose is never formed. In SIM we synthesize it from the estimated ``full_base`` +
        the observed cubes -- the base pose stays confined to this sensor stand-in and out of the tracker."""
        scene_dict, _ = reach_scene_and_targets(
            self._cfg, self._scenario, self._mj_model, full_base, cube_poses=observed_cube_pose)
        return {name: (np.asarray(box["pose"][:3], dtype=float), np.asarray(box["pose"][3:], dtype=float))
                for name, box in scene_dict["cuboid"].items()}

    def _drive_gaze(self, base_pose, observed_cube_pose) -> dict:
        """Accrue the harness detections into the policy's persistent ``_belief`` and return the gimbal
        ``camera_command`` for this tick (empty for fixed/no-gaze robots). Perception is now policy-owned
        (seam inversion): the harness reports only what the cams frame at the CURRENT realized gaze (LIVE,
        non-accrued), so the policy is the one that REMEMBERS -- every phase then reasons over ``_belief``,
        not the blink-in/blink-out live frame. Actuated robots also get a fresh gimbal aim (sweep-to-find /
        track-to-hold) via the ``_GazeController``; that command is what the harness renders next tick.

        Remembering is NOT the same as currently seeing: the raw live frame and a per-cube last-framed stamp
        are retained alongside the accrual so the pre-commit gates (``_reachable_cubes``) can tell the two
        apart. Everything POST-commit still reasons over the accrued belief."""
        self._belief.update(observed_cube_pose)
        self._live = dict(observed_cube_pose)
        for cube in observed_cube_pose:
            self._belief_tick[cube] = self._ticks
        if self._gaze is None:
            return {}
        active_phases = {_EXTEND, _GRASP, _RETRACT_PLAN, _RETRACT}
        active_route = self._route if self._route is not None else self._grasp_route
        # NEVER AIM AT A CUBE ALREADY IN THE JAWS. ``_pending`` only retires a cube at the END of RETRACT
        # (``_handle_retract``), so for the whole GRASP -> RETRACT carry the held cube is STILL both a route
        # target and a pending candidate. On a rig with one gimbal and two candidates the assignment then
        # ping-pongs between the cube in the hand -- whose pose is already known by kinematics, so framing it
        # buys nothing -- and the cube still owed, which is the only one whose belief must stay fresh for the
        # next commit. Observed in the viewer: the lens oscillates for the whole carry and the next visit is
        # declined on arrival. Suppressed only while work remains: with nothing else owed, keeping the
        # existing lock beats kicking off an anchor sweep during the return.
        unheld_pending = self._pending - set(self._held.values())
        held_blind = set(self._held.values()) if unheld_pending else set()
        route_assignment = ({side: cube for side, cube in active_route.assignment.items()
                             if cube not in held_blind} if active_route is not None else {})
        locked_targets_by_cam = None
        if self._phase in active_phases and route_assignment:
            # Route side is camera identity: aim that gimbal at its arm's committed cube.  Re-solving an
            # injective camera assignment each tick can swap a one-cube grasp between gimbals on base sway.
            locked_targets_by_cam = {}
            # "R" -> 1 mirrors the 2-gimbal rig's physical mount (cam 0/L, cam 1/R). A 1-gaze-cam rig
            # (v2_single) has no camera 1, so a SOLO right-arm visit hit the drop below and never got a
            # live lock (measured: left_right_far/front_back_far, sequential single-arm ROUTES). min()
            # shares the one camera between both sides instead -- safe because any commit with more
            # targets than cameras is intercepted by the centroid branch above, before this is reached.
            side_to_cam = {"L": 0, "R": min(1, len(self._cfg.gaze_cams) - 1)}
            target_names = tuple(dict.fromkeys(route_assignment.values()))
            seen_centers = [self._belief[name][0] for name in target_names if name in self._belief]
            if 0 < len(self._cfg.gaze_cams) < len(seen_centers):
                # FEWER GIMBALS THAN COMMITTED TARGETS. The per-side 1:1 lock below would park the only
                # gimbal on ONE arm's cube (``side_to_cam`` silently drops any side with no matching gimbal),
                # discarding the very co-visibility the pre-commit simultaneity gate checked: the commit was
                # licensed by "this rig can frame these together", so the reach must keep framing them
                # together or the licence was meaningless. Hold the centroid instead -- an FOV wide enough to
                # frame the set at commit still frames it when centred on that set. Without this, a legitimate
                # bimanual commit (front_back_close: one lens spans both near tables) still executed with the
                # gimbal parked on one cube, measuring VRW duty 0.
                centroid = np.mean(np.asarray(seen_centers, dtype=float), axis=0)
                locked_targets_by_cam = {ci: centroid for ci in range(len(self._cfg.gaze_cams))}
            else:
                for side, target_name in route_assignment.items():
                    cam_i = side_to_cam[side]
                    if cam_i < len(self._cfg.gaze_cams) and target_name in self._belief:
                        locked_targets_by_cam[cam_i] = self._belief[target_name][0]
        elif (self._phase == _APPROACH and self._walk_current in self._belief
              and len(self._cfg.gaze_cams) < len(self._belief)):
            # A rig with fewer gimbals than believed cubes cannot watch them all, and APPROACH has ALREADY
            # chosen which cube this visit is for. Pin the gaze to that one rather than letting the bijective
            # assignment keep whichever cube currently scores best on total yaw: the freshness gate at commit
            # reads the CHOSEN target, so a gimbal parked on the other cube would leave the visit permanently
            # un-committable and livelock APPROACH -> declined commit -> ACQUIRE. Rigs that CAN cover every
            # cube (v2's two gimbals on a two-cube scene) keep the unrestricted track -- no behavior change.
            seen_centers = [self._belief[self._walk_current][0]]
        else:
            # PENDING cubes only, not the whole belief. Belief is deliberately never forgotten, so a cube
            # that has already been picked and returned stays in it forever -- and an under-provisioned rig
            # (one gimbal, two cubes) then has both as candidates with equal coverage, so the assignment is
            # settled by cost alone and parks on whichever scores lowest. That is routinely the FINISHED
            # cube (the retired visit ended with the base squared up to it, so its |yaw| is near zero while
            # the remaining cube's is near pi), and the gaze switching cost then PINS it there: the robot
            # stares at work it has already done while the cube it still owes goes unwatched, starving the
            # freshness gate that must see it to commit the next visit. Filtering to ``_pending`` makes the
            # candidate set mean "work still owed", which is what the tracker was always for -- minus the
            # cube currently IN the jaws, which is pending-but-done (see ``unheld_pending`` above).
            seen_centers = [self._belief[name][0]
                            for name in (unheld_pending or self._pending) if name in self._belief]
        return self._gaze.step(base_pose, self._scenario, seen_centers, locked_targets_by_cam)

    def step(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None,
             physical_grasps: dict[str, str] | None = None):
        """One control tick. Returns ``(base_twist, arm_reference, camera_command)``.

        ``base_pose`` = ``(pos[3], quat_wxyz[4])`` estimate; ``proprio_qpos`` reserved (seed/telemetry);
        ``observed_cube_pose`` = world ``{name: (center, dims)}`` = the cubes the cams frame at the CURRENT
        realized gaze THIS tick (perception seam; LIVE, the policy accrues it into ``_belief``). The FSM then
        reasons over the accrued belief. ``arm_reference`` = ``{mjlab_joint: qpos}`` over the ACTIVE arm(s);
        ``camera_command`` = ``{gimbal_joint: qpos}`` the policy now OWNS (drives the harness gimbals);
        ``base_twist`` = ``(vx, vy, wz)`` -- zero on the parked path, driven by the walk-search when ``walk``.
        ``actual_tool_poses_base`` is optional live `{frame: (pos, quat)}`` feedback for the dynamic Jacobian
        tracker's bounded position integrator; kinematic callers pass None. ``physical_grasps`` is the
        completed dynamic weld mapping from the prior physics tick. A closed planned jaw does not retire a
        target; only this physical-contact feedback does.

        A replan that finds nothing reachable NEVER pauses the arm (invariant 4): hold the loaded route
        if one exists, else there is simply nothing to do yet (arm stays at its home/reset pose).
        """
        self._physical_grasps = None if physical_grasps is None else dict(physical_grasps)
        cam_cmd = self._drive_gaze(base_pose, observed_cube_pose)
        self._ticks += 1
        if self._ticks <= _PREACT_HOLD_TICKS:
            # PRE-ACT HOLD: gaze already ran and ``_drive_gaze`` already accrued this tick's detections into
            # ``_belief``, so the robot is watching and remembering -- it simply does not drive or commit yet.
            # Returning before the walk brain matters: the base must stay put, or the disturbance would be
            # measured against a different stance per robot.
            return (0.0, 0.0, 0.0), {}, cam_cmd
        if self._walk:
            twist, arm_ref, _ = self._walk_step(base_pose, proprio_qpos, self._belief, actual_tool_poses_base)
            return twist, arm_ref, cam_cmd
        if self._parked_done:
            return (0.0, 0.0, 0.0), self._hold_only_arm_ref(), cam_cmd
        if self._phase is None:
            # Parked and walk now share ONE FSM entry AND one decision hub: seed the pending pick set and
            # enter ACQUIRE (perception-first for every embodiment) through the sole writer (``_transition``,
            # which also zeroes ``_phase_t``). Parked runs ACQUIRE with caps OFF -- it commits the set
            # reachable from its standing stance (parked EXTEND self-plans it) or FAILs loud, and never
            # DISCOVER/APPROACH (it cannot drive). Same dispatch table, same single-writer invariant.
            self._pending = {m.name for m in self._scenario.pick}
            self._transition(_ACQUIRE)
        twist, arm_ref, _ = self._dispatch(base_pose, proprio_qpos, self._belief, actual_tool_poses_base)
        return twist, arm_ref, cam_cmd

    # ---- walk-search: DRIVE the base to a reachable stance, then reach (lifted from the mission) ----------
    def _base_frame_xy(self, base_pose, cube_world):
        """``(x_base, y_base, dist_xy)`` of a world cube center from the live base pose -- the mover's input
        (it steers on the base-frame target). Yaw+full quat inverse via ``mju_rotVecQuat(conj(quat))``."""
        base_pos, base_quat = base_pose
        d = np.asarray(cube_world, dtype=float)[:3] - np.asarray(base_pos, dtype=float)
        conj = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]], dtype=float)
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, d, conj)
        return float(out[0]), float(out[1]), float(np.linalg.norm(out[:2]))

    def _anchor_bearing(self, base_pose, anchor) -> float:
        """Signed base-frame bearing (rad) from the forward +X axis to ``anchor``: the yaw a fixed forward
        camera must turn to point at it. Reuses ``_base_frame_xy`` (base-frame anchor coords)."""
        x_b, y_b, _ = self._base_frame_xy(base_pose, anchor)
        return float(np.arctan2(y_b, x_b))

    def _unvisited_anchors(self, base_pose):
        """Known support anchors (object-independent) not yet SEARCH-turned to,
        ``[(name, world_xyz, half_width)]`` (``half_width`` ``None`` for an unbounded hand marker).
        Anchor NAMES are viewpoint-stable (``known_anchors`` only shifts the edge position with the
        viewpoint), so the visited set stays valid as the base moves."""
        base_pos = base_pose[0]
        return [(n, a, hw) for n, a, _tan, hw, _fn in known_anchors(self._scenario, base_pos)
                if n not in self._anchors_visited]

    def _nearest_anchor(self, base_pos, cube_world_pos) -> tuple[str, np.ndarray, np.ndarray, float | None, np.ndarray | None]:
        """APPROACH TRIAL: the ``known_anchors`` entry nearest a cube's pose -- the fixed support (table
        edge/shelf tier/hand marker) it rests on, plus its tangent + half-width (point-C projection,
        see ``_point_c``). Full 3D distance, not XY: shelf tiers share XY and differ only in Z, so an
        XY-only nearest ties between them.

        Bounded supports (table/shelf, ``half_w is not None``) are tried FIRST, by distance, and accepted
        as soon as one's footprint actually contains the cube's lateral offset (within ``half_w`` + margin)
        -- raw nearest-distance alone can pick an unbounded hand marker over the table a cube physically
        rests on (e.g. a cube near the table's far edge, close to a hand marker on the other side of the
        table). A hand marker has no bounded footprint and no support-clearance safety (``_point_c`` leaves
        both lateral AND depth unclamped for it, ``_support_clearance`` returns ``None``), so APPROACH then
        drives an obstacle-blind straight line toward it (``_handle_discover``'s mover has "no obstacle
        avoidance") and clips whatever geometry -- e.g. a table corner -- sits between the base and that
        point. Falls back to plain nearest-distance (may still pick a hand marker) only when no bounded
        support actually contains the cube -- e.g. a cube genuinely resting at a hand-offer point."""
        anchors = known_anchors(self._scenario, base_pos)
        cube_pos = np.asarray(cube_world_pos, dtype=float)
        def _dist(na):
            return np.linalg.norm(np.asarray(na[1], dtype=float) - cube_pos)
        bounded = sorted((na for na in anchors if na[3] is not None), key=_dist)
        for name, pos, tangent, half_w, face_normal in bounded:
            offset = cube_pos[:2] - np.asarray(pos, dtype=float)[:2]
            lateral = abs(float(np.dot(offset, tangent)))
            # Lateral-only containment (within the anchor's own width) is NECESSARY but not SUFFICIENT: a
            # cube can share a table's width-span while resting somewhere else entirely (e.g. handed off
            # from a human hand well off the table's actual surface, see `bimanual_mixed_front_back_close`'s
            # ``_fb_cube_1``). Also check DEPTH -- how far the cube sits from the anchor's near-edge point
            # along ``face_normal`` (the ALREADY-COMPUTED signed normal `known_anchors` returns, pointing
            # from the support toward the resolving viewpoint -- NOT a generic 90-deg rotation of `tangent`,
            # which would silently mean something different for a front-facing table vs a side-mounted bench
            # where the tangent/normal axes are swapped; an earlier attempt using a tangent-derived normal
            # regressed `left_right_close`/`left_right_far` for exactly this reason, reverted 2026-07-26).
            # A cube genuinely resting on this support should sit within a modest depth of its near edge;
            # ``_POINT_C_MAX_DEPTH_M`` is the SAME bound ``_point_c`` itself already treats as the sane
            # advance-past-the-edge range, reused here (no new magic number) so a cube whose depth exceeds
            # what `_point_c` could plausibly represent falls through to the next candidate instead of
            # getting force-fit onto a table it doesn't actually rest on.
            depth = abs(float(np.dot(offset, face_normal))) if face_normal is not None else 0.0
            if lateral <= half_w + _SUPPORT_CLEARANCE_MARGIN_M and depth <= _POINT_C_MAX_DEPTH_M + _SUPPORT_CLEARANCE_MARGIN_M:
                return name, np.asarray(pos, dtype=float), tangent, half_w, face_normal
        name, pos, tangent, half_w, face_normal = min(anchors, key=_dist)
        return name, np.asarray(pos, dtype=float), tangent, half_w, face_normal

    def _support_clearance(self, base_pose):
        """Return ``(root_clearance, min_clearance, stop_clearance, face_normal_base)`` for a bounded support.

        The support normal is a geometric face property cached when this target's anchor is selected. It is
        deliberately independent of the selected drive direction: the latter controls motion and braking,
        while the former projects the root-frame body footprint to prevent torso-side table penetration.
        Hand markers return ``None`` because they have no bounded support face.
        """
        return self._support_clearance_at(base_pose[0], base_pose[1])

    def _support_clearance_at(self, base_pos, base_quat):
        """Support clearance at an arbitrary fixed-yaw candidate base pose.

        Field-target selection uses this before calling fast IK. Same projected-footprint calculation as
        live APPROACH means a high offline score cannot nominate a base target inside a table/shelf.
        """
        if self._approach_support_normal_w is None or self._approach_anchor_pos is None:
            return None
        normal_w = np.asarray(self._approach_support_normal_w, dtype=float)
        conj = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]], dtype=float)
        normal_b3 = np.zeros(3)
        mujoco.mju_rotVecQuat(normal_b3, np.array([normal_w[0], normal_w[1], 0.0]), conj)
        normal_b = normal_b3[:2] / max(np.linalg.norm(normal_b3[:2]), 1e-6)
        points_b = np.asarray(self._cfg.body_clearance_points_b, dtype=float)
        body_extent = float(np.max(points_b @ (-normal_b)))
        root_clearance = float(normal_w @ (np.asarray(base_pos, dtype=float)[:2]
                                            - np.asarray(self._approach_anchor_pos, dtype=float)[:2]))
        min_clearance = (self._cfg.walk_support_clearance_m
                         if self._cfg.walk_support_clearance_m is not None
                         else body_extent + _SUPPORT_CLEARANCE_MARGIN_M)
        stop_clearance = min_clearance + self._mover.braking_distance_m
        return root_clearance, min_clearance, stop_clearance, normal_b

    def _point_c(self, anchor_w, cube_w):
        """Point C: ``anchor_w`` shifted to the cube's own XY position -- both along the anchor's CACHED
        tangent (``_approach_anchor_tangent``, set once per target by ``_nearest_anchor``, LATERAL) and its
        perpendicular (DEPTH) -- so the base's approach/standoff distance is measured from the cube itself,
        not from the anchor's near-edge line. Human-analogy fix for approaching an off-center/set-back object
        on a desk/shelf/hand instead of a shared fixed anchor point. Earlier version only corrected lateral
        (bearing), leaving depth pinned to the anchor's edge -- correct when the object sits AT the edge, but
        leaves the base standoff too far from an object set back on the surface (live-test finding). LATERAL
        clamped to ``_approach_anchor_half_w`` when the anchor is bounded (table/shelf edge -- never walk C
        past the anchor's own physical extent on a noisy cube belief); DEPTH advances toward a cube set back
        on a bounded support too, clamped to ``_POINT_C_MAX_DEPTH_M`` (stays well under the base's own
        standoff from C, so the final stance never lands on the support surface -- see that constant).
        Unbounded (lateral) for a hand marker (no box). Falls back to ``anchor_w`` unchanged if no tangent is
        cached (defensive; should not trigger, ``_nearest_anchor`` always returns one)."""
        if self._approach_anchor_tangent is None:
            return anchor_w
        anchor_xy = np.asarray(anchor_w, dtype=float)[:2]
        cube_xy = np.asarray(cube_w, dtype=float)[:2]
        offset = cube_xy - anchor_xy
        tangent = self._approach_anchor_tangent
        normal = np.array([tangent[1], -tangent[0]])
        lateral = float(np.dot(offset, tangent))
        if self._approach_anchor_half_w is not None:
            lateral = float(np.clip(lateral, -self._approach_anchor_half_w, self._approach_anchor_half_w))
        depth = float(np.dot(offset, normal))
        if self._approach_anchor_half_w is not None:            # bounded (table/shelf): cap the advance magnitude.
            # ``normal``'s sign (a fixed +90 deg rotation of ``tangent``) is NOT always "into the table" --
            # table_L/table_R anchors have opposite yaw, so the same rotation points into the table for one
            # and out of it for the other. Clamp by MAGNITUDE (sign-agnostic) rather than to [0, MAX]: a cube
            # resting on the table always has a bounded depth offset in the true into-table direction, so
            # capping |depth| bounds the advance correctly regardless of which sign this anchor's normal is.
            depth = float(np.clip(depth, -_POINT_C_MAX_DEPTH_M, _POINT_C_MAX_DEPTH_M))
        point_c = np.asarray(anchor_w, dtype=float).copy()
        point_c[:2] = anchor_xy + lateral * tangent + depth * normal
        return point_c

    def _field_target_tau(self) -> float | None:
        """Return field threshold for baseline, retry, or direct target-stance A/B modes.

        ``0`` retains Point-C, ``1`` proposes field stance only after a failed physical latch,
        and ``direct`` proposes it for the first visit.  Direct mode isolates base-target
        selection; reach plan, tracker, jaw timing, and physical latch stay unchanged.
        """
        if os.environ.get("VISIBLE_REACHABLE_TARGET_STANCE", "0") == "0":
            return None
        raw = os.environ.get("VISIBLE_REACHABLE_TARGET_TAU")
        if raw is not None:
            return float(raw)
        return _FIELD_TARGET_TAU_BY_ROBOT.get(self._cfg.name)

    def _prepare_field_stance_search(self, base_pose, belief) -> None:
        """Build fixed-yaw, capability-aligned field stance candidates for one low-margin target.

        Inverse map: selected field grasp point ``p_b`` gives world base target
        ``b_xy = cube_xy - R_yaw p_b_xy``. We retain only targets that (1) exceed an absolute robust
        threshold, (2) keep declared support clearance and support width, and (3) lie on a declared drive
        axis within mover face tolerance. Thus V2 can retain yaw while moving laterally; V2-fixed/G1 can
        participate only where their forward/backward capability permits. No field point is a success gate.
        """
        tau = self._field_target_tau()
        if tau is None or self._latch_cube_pose is None:
            return
        # ``1`` keeps first dynamic visit legacy and uses a field stance only after a closed-jaw miss.
        # ``direct`` makes field stance first choice, isolating base-target selection in an A/B run.
        # Existing capture contract permits one retry, so neither mode can create unbounded search.
        target_stance_mode = os.environ.get("VISIBLE_REACHABLE_TARGET_STANCE", "0")
        if (self._mpc_track and target_stance_mode != "direct"
                and self._grasp_capture_retries.get(self._walk_current, 0) == 0):
            return
        # Do not split a valid in-place bimanual opportunity into a field-guided single target visit.
        if len(self._reachable_cubes(base_pose, belief)) > 1:
            return
        field = visible_reachable_field(self._cfg.name)
        if field is None:
            return
        cube_w = np.asarray(self._latch_cube_pose[0], dtype=float)
        base_link_pos, base_link_quat = base_pose
        yaw_pos, base_yaw_quat = _yaw_only(base_pose)
        # Field lookup uses the live base_link frame, matching field build and
        # real camera/arm geometry. The arm planner remains yaw-only below:
        # its learned tracker cannot safely consume transient locomotion tilt.
        current = max(field.score_world(cube_w, base_link_pos, base_link_quat, side) for side in ("L", "R"))
        self._field_stance_current_score = current
        if current >= tau:
            return                                      # already robust at fixed yaw: plan in place
        (self._approach_anchor_name, self._approach_anchor_pos, self._approach_anchor_tangent,
         self._approach_anchor_half_w, self._approach_support_normal_w) = self._nearest_anchor(yaw_pos, cube_w)
        grasp_b = field._grasp_point_base(cube_w, base_link_pos, base_link_quat)
        current_yaw = float(np.arctan2(
            2.0 * (base_yaw_quat[0] * base_yaw_quat[3] + base_yaw_quat[1] * base_yaw_quat[2]),
            1.0 - 2.0 * (base_yaw_quat[2] ** 2 + base_yaw_quat[3] ** 2)))
        # A gimballed head or opposed fixed cameras keep coverage at the current body yaw. Retaining it
        # avoids an unnecessary learned-policy turn; the mover can make a bounded cross-track correction.
        # A single forward camera still samples a camera-compatible final yaw.
        hold_yaw = self._rear_capable
        yaw_samples = (current_yaw,) if hold_yaw else tuple(
            current_yaw + np.linspace(-np.pi, np.pi, 181, endpoint=True))
        candidates = []
        # An arm already holding a cube cannot reach the next one: planning it commands the carrying arm
        # off its grasp config, which drops the carried cube and snaps the arm violently (plan-0 seeds at
        # the reach-ready pose, not at the grasp pose the arm is actually in). The field is occupancy-blind
        # -- ``robust_grasp_points`` emits an L and an R candidate for every point -- and the assignment it
        # produces OVERRIDES the ``free_arms`` pin in ``_commit_reach``, so the filter has to happen here.
        held_arms = set(self._held)
        for point_b, arm, score in field.robust_grasp_points(
                grasp_b[2], tau, max_points=_FIELD_TARGET_MAX_CANDIDATES * 4):
            if arm in held_arms:
                continue
            for target_yaw in yaw_samples:
                target_quat = np.asarray(yaw_quat(float(target_yaw)), dtype=float)
                rot = np.empty(9)
                mujoco.mju_quat2Mat(rot, target_quat)
                target = np.asarray(yaw_pos, dtype=float).copy()
                target[:2] = cube_w[:2] - rot.reshape(3, 3)[:2, :2] @ point_b[:2]
                delta_w = target[:2] - yaw_pos[:2]
                distance = float(np.linalg.norm(delta_w))
                if distance < _FIELD_TARGET_MIN_TRAVEL_M:
                    continue
                # At tick 0 the mover chooses its nearest declared body drive vector. A yaw-holding robot
                # may correct cross-track error while translating, but must remain primarily along that
                # vector. A single-forward-camera robot instead turns to the candidate's predicted yaw.
                world_bearing = float(np.arctan2(delta_w[1], delta_w[0]))
                initial_bearing = _wrap_pi(world_bearing - current_yaw)
                drive = max(self._cfg.walk_drive_dirs, key=lambda d: float(np.dot(
                    np.array([np.cos(initial_bearing), np.sin(initial_bearing)]), np.asarray(d, dtype=float))))
                drive_heading = float(np.arctan2(drive[1], drive[0]))
                if hold_yaw:
                    target_dir = np.array([np.cos(initial_bearing), np.sin(initial_bearing)])
                    along = float(np.dot(target_dir, np.asarray(drive, dtype=float)))
                    cross = abs(float(target_dir[0] * drive[1] - target_dir[1] * drive[0]))
                    cross_ratio = (_FIELD_TARGET_GAZE_CROSS_TRACK_RATIO if self._cfg.gaze_cams
                                   else _FIELD_TARGET_DUAL_VIEW_CROSS_TRACK_RATIO)
                    if along <= 0.0 or cross > cross_ratio * along:
                        continue
                else:
                    expected_yaw = _wrap_pi(world_bearing - drive_heading)
                    if abs(_wrap_pi(expected_yaw - target_yaw)) > np.radians(2.0):
                        continue
                clearance = self._support_clearance_at(target, target_quat)
                if clearance is not None and clearance[0] < clearance[1]:
                    continue
                if self._approach_anchor_half_w is not None:
                    lateral = float(np.dot(target[:2] - self._approach_anchor_pos[:2],
                                           self._approach_anchor_tangent))
                    if abs(lateral) > self._approach_anchor_half_w:
                        continue
                candidates.append((target, target_quat, arm, score, distance,
                                   abs(_wrap_pi(target_yaw - current_yaw))))
        candidates.sort(key=lambda item: (-item[3], item[4], item[5]))
        self._field_stance_candidates = [item[:4] for item in candidates[:_FIELD_TARGET_MAX_CANDIDATES]]
        self._field_stance_candidate_i = 0
        if self._field_stance_candidates:
            best = candidates[0]
            print(f"[reach] field target current={current:.3f} tau={tau:.3f} "
                  f"candidates={len(self._field_stance_candidates)} "
                  f"base_xy=({best[0][0]:.3f},{best[0][1]:.3f}) "
                  f"travel={best[4]:.3f}m yaw_turn={np.degrees(best[5]):.1f}deg")

    @staticmethod
    def _pin_pose(pose):
        """SNAPSHOT one observed cube pose so a "committed" pose stops tracking live physics.

        ``cube_world_poses`` builds each pose as ``np.asarray(mj_data.geom_xpos[gid], dtype=float)``, and
        ``np.asarray`` on a matching-dtype array returns the input, SHARING MEMORY with MuJoCo's buffer. So a
        pose merely stored (``_latch_cube_pose``, ``_reach_poses``) is a live VIEW that silently follows the
        cube: the FSM's "pose the reach committed to at APPROACH" was equal to the live pose by construction.
        Measured before this fix: 1937/1937 EXTEND ticks reported ``||live - pinned|| == 0.0`` exactly,
        including 225 ticks after a ground-truth 0.01 m knock witness, so every displacement test was dead
        code (``media/cotarget_replan_20260730/``). Copies only the CENTER; dims come from ``geom_size``
        (model-static, no aliasing risk)."""
        center, dims = pose
        return (np.array(center, dtype=float), dims)

    def _begin_visit(self, base_pose, belief, cube: str) -> None:
        """Initialize one target visit before either legacy Point-C or field target selection.

        ACQUIRE and SELECT_NEXT share this setup. Keeping it one function prevents ACQUIRE's direct
        in-place branch from silently bypassing an enabled field treatment, while bimanual in-place
        candidates still stay legacy inside ``_prepare_field_stance_search``.
        """
        self._walk_current = cube
        self._latch_dist = None
        self._extend_retries = self._extend_no_route_ticks = 0
        self._side_stance_retries = 0
        self._side_stance_target_w = None
        self._curated_stance_probe_w = None
        self._curated_stance_ik_rejected = False
        self._field_stance_candidates = []
        self._field_stance_candidate_i = 0
        self._field_stance_probe = None
        self._field_stance_target_w = None
        self._field_stance_target_quat = None
        self._field_stance_assignment = None
        self._field_stance_current_score = None
        self._stance_turn_retries = 0
        self._stance_turn_target_w = None
        self._stance_turn_probe_w = None
        self._extra_standoff = 0.0
        self._approach_anchor_pos = None
        self._approach_anchor_name = None
        self._approach_anchor_tangent = None
        self._approach_anchor_half_w = None
        self._approach_support_normal_w = None
        self._latch_cube_pose = self._pin_pose(belief[cube])
        # New target: re-arm the rear prime, but KEEP any parked configuration -- it is where the arm
        # physically is, and dropping it would servo the arm to planning home for no reason.
        self._prime_executor = self._prime_template = None
        self._prime_controlled = []
        self._prime_tried = False
        self._prepare_field_stance_search(base_pose, belief)
        self._mover.retarget()

    def _probe_field_stance(self, base_pose):
        """Screen one field target through existing fast IK before locomotion.

        Returns ``True`` after an approved candidate is installed, ``False`` once all candidates fail,
        and ``None`` while dynamic MPC's sole planner worker is solving. The planner assignment is pinned
        to candidate arm, making field score and fast IK test same target-to-arm hypothesis.
        """
        if self._field_stance_probe is not None:
            ready, feasible = self._worker.poll_fast_ik()
            if not ready:
                return None
            target, target_quat, arm, score = self._field_stance_probe
            self._field_stance_probe = None
            if feasible:
                self._field_stance_target_w = target
                self._field_stance_target_quat = target_quat
                self._field_stance_assignment = {arm: self._walk_current}
                print(f"[reach] field target {self._field_stance_current_score:.3f} -> {score:.3f} fast-IK PASS")
                return True
            self._field_stance_candidate_i += 1
        while self._field_stance_candidate_i < len(self._field_stance_candidates):
            target, target_quat, arm, score = self._field_stance_candidates[self._field_stance_candidate_i]
            assignment = {arm: self._walk_current}
            belief = {self._walk_current: self._latch_cube_pose}
            if self._session is not None:
                feasible = ik_feasible_route(
                    self._session, self._scenario, self._mj_model, (target, target_quat),
                    cube_poses=belief, fixed_assignment=assignment, height_pad=self._height_pad)
                if feasible:
                    self._field_stance_target_w = target
                    self._field_stance_target_quat = target_quat
                    self._field_stance_assignment = assignment
                    print(f"[reach] field target {self._field_stance_current_score:.3f} -> {score:.3f} fast-IK PASS")
                    return True
                self._field_stance_candidate_i += 1
                continue
            scene_dict, targets = reach_scene_and_targets(
                self._cfg, self._scenario, self._mj_model, (target, target_quat), cube_poses=belief,
                height_pad=self._height_pad)
            self._worker.submit_fast_ik(scene_dict, targets, target, target_quat, fixed_assignment=assignment)
            self._field_stance_probe = (target, target_quat, arm, score)
            return None
        return False

    def _select_next(self, base_pose, observed_cube_pose) -> str:
        """Body-FSM hub: choose the next visit target and RETURN the state to enter -- the caller wraps the
        result in ``_transition``, so this NEVER writes the state itself (``_transition`` stays the sole
        writer). WHICHEVER pending cube is in belief -> APPROACH the nearest (seed its remembered pose);
        else a pending cube stays hidden and an unvisited anchor remains -> DISCOVER; else TERMINATE.
        Opportunistic: an unseen cube is never ordered by pose."""
        visible = [c for c in self._pending if c in observed_cube_pose]
        if visible:
            cube = min(visible, key=lambda c: self._base_frame_xy(base_pose, observed_cube_pose[c][0])[2])
            self._begin_visit(base_pose, observed_cube_pose, cube)
            return _APPROACH
        if self._pending and self._unvisited_anchors(base_pose):
            self._walk_current = None
            self._search_anchor = None
            return _DISCOVER
        # No pending cube is visible and no viewpoint is left to try: every still-pending cube is unseeable
        # from any reachable stance -> record the reason so the mission verdict is FAIL(unseen), not a silent
        # hold (REACH_FSM_REDESIGN.md: every mission ends in an explicit verdict).
        for cube in self._pending:
            self._failed.setdefault(cube, "unseen (no viewpoint framed it)")
        return _TERMINATE

    def _transition(self, next_state: str) -> None:
        """The SOLE writer of the mission FSM state: set the state + reset the phase timer. Centralizing every
        edge here is what keeps a guard and its transition in the same place -- scattered inline
        ``self._phase = ...`` mutations are the bug class this removes."""
        if next_state == _ACQUIRE:
            self._acquire_ticks = 0                          # fresh gaze-lock backstop per ACQUIRE entry
        self._phase, self._phase_t = next_state, 0

    @property
    def _rear_capable(self) -> bool:
        """True when the robot can PERCEIVE behind itself: a gimballed head, or opposed fixed cameras.
        Gates the rear prime (flipping an arm toward a region the robot cannot see is not a capability) and
        the field-stance search's yaw hold, which is the same question asked of the same geometry."""
        return bool(self._cfg.gaze_cams) or len(self._cfg.walk_visibility_dirs) > 1

    def _end_prime(self) -> None:
        """Retire the rear prime the moment something else owns the primed arm.

        ``DynamicArmReferenceExecutor.command`` CLAMPS past the end of its route and holds forever -- which is
        what makes the flip survive the plan-0 flight, and also what makes a LIVE prime keep re-writing the
        flip into ``_arm_hold`` on every ``_dispatch`` for the rest of the mission. After RETRACT parks the arm
        at the pose cuRobo landed in, every later tick then re-emitted the FLIP instead: measured
        3.44 rad on ``right_shoulder_1_joint`` at the mission's last tick, exactly the retract-handoff whip.

        The parked VALUES stay: the other arm may be holding a cube from an earlier visit, and the primed arm's
        own entries are overwritten by ``_park_arm_at`` at this visit's retract settle before anything reads them.
        """
        self._prime_gen = None
        self._prime_executor = self._prime_template = None
        self._prime_controlled = []

    def _clear_arm_hold(self) -> None:
        """Release the parked configuration (and any rear prime feeding it): the arm reverts to the
        harness's planning-home fill, and plan-0 goes back to its default home seed."""
        self._end_prime()
        self._arm_hold_q = self._arm_hold = None
        self._prime_tried = False

    def _park_arm_at(self, arm_ref: dict) -> None:
        """Park the arm at the configuration it is physically holding, per joint, instead of letting the
        harness servo it back to planning home. ``arm_ref`` is an mjlab-qpos command; the cspace twin that
        plan-0 has to be seeded with inverts the ref fold, exactly as ``_seed_from_proprio`` does."""
        self._arm_hold = {**(self._arm_hold or {}), **arm_ref}
        self._arm_hold_q = {**(self._arm_hold_q or dict(self._cfg.planning_home_joint_pos)),
                            **{j: float(v) - self._ref_offset(j) for j, v in arm_ref.items()}}

    def _maybe_prime_rear(self, base_pose, belief, cube_w, dist) -> None:
        """Submit the free arm's rear flip when this visit's cube sits BEHIND the stance being walked to.

        Graded against the PREDICTED stance (point C), not the live base pose: the trigger has to fire while
        there is still walk left to hide the flip in, and at that moment the base is typically still at the
        previous support. A rear-capable robot holds its yaw through the approach (the ``hold_yaw``
        predicate), so the predicted stance is simply the current orientation at point C. Building the
        collision scene there too is the conservative choice -- a flip that clears the table from the
        ARRIVAL pose clears it a metre out.

        Requires exactly ONE free arm: with two free, ``_commit_reach`` may pin either, and priming an arm
        the visit then declines to use is a coin flip. ``seed_q=None`` is truthful here -- APPROACH emits an
        empty ``arm_ref`` every tick, which the harness fills with ``_arm_home_pose``, so both arms really
        are at planning home (the carrying one included: ``_handle_retract`` lerped it there at the end of
        the previous visit, holding its cube AT the home posture).
        """
        free = [a for a in ("L", "R") if a not in self._held]
        if (self._prime_tried or not _REAR_PRIME_BEARING_DEG or not self._walk or not self._mpc_track
                or not self._rear_capable or len(free) != 1 or self._worker is None
                or self._prime_gen is not None or dist < _REAR_PRIME_MIN_TRAVEL_M):
            return
        # Point C is a DRIVE target on the support's own edge line, not a stance: the mover brakes short of
        # it by ``stop_dist``. Priming at point C itself puts the body inside the table, cuRobo rejects the
        # query outright (``plan_pose`` -> None, ~0.04 s) and every prime fails. Back off along the actual
        # approach line by the same floor the drive-in will honour. The ``_latch_dist`` term of the real
        # ``stop_dist`` is not available yet -- the proximity latch has not fired this far out -- and
        # dropping it only makes the predicted stance more conservative (further from the support).
        approach = (np.asarray(self._nav_target_w, dtype=float)[:2]
                    - np.asarray(base_pose[0], dtype=float)[:2])
        travel = float(np.linalg.norm(approach))
        stop_dist = max(_GEOMETRIC_FLOOR_M,
                        self._cfg.walk_min_standoff_m + self._mover.braking_distance_m)
        stance_pos = np.asarray(base_pose[0], dtype=float).copy()
        stance_pos[:2] += approach * (1.0 - min(stop_dist, travel) / travel)
        stance = (stance_pos, base_pose[1])
        x_b, y_b, _ = self._base_frame_xy(stance, cube_w)
        bearing_deg = abs(float(np.degrees(np.arctan2(y_b, x_b))))
        if bearing_deg < _REAR_PRIME_BEARING_DEG:
            return
        # A CARRIED cube is not a static obstacle -- it rides in the jaws. Left in the scene it sits exactly
        # where the gripper is, so the start state is start-infeasible and every prime fails in ~0.04 s.
        # (``goal="home"`` meets the same situation and answers it by disabling the grasp-contact links;
        # here the cube can simply be dropped from the belief, which is strictly more honest.)
        carried = set(self._held.values())
        scene_dict, targets = reach_scene_and_targets(
            self._cfg, self._scenario, self._mj_model, stance,
            cube_poses={c: p for c, p in belief.items() if c not in carried})
        self._prime_tried = True
        self._prime_gen = self._worker.submit_plan0(
            scene_dict, targets, stance[0], stance[1],
            fixed_assignment=None, seed_q=None, goal=f"prime:{free[0]}")
        print(f"[reach] rear prime {free[0]}: bearing={bearing_deg:.0f}deg travel={dist:.2f}m")

    def _tick_prime(self) -> None:
        """Land a submitted flip, then advance it one control interval and refresh the hold command.

        A landed ``None`` (cuRobo infeasible, or the in-worker basin/carry check rejected the route) leaves
        every field cleared, so the visit runs exactly as it did before this feature existed.
        """
        if self._prime_gen is not None:
            ready, route = self._worker.poll_plan0(self._prime_gen)
            if not ready:
                return
            self._prime_gen = None
            if route is None:
                return
            names_mjlab, route_mjlab = self._route_to_mjlab(route)
            self._prime_controlled = list(route.controlled_joints)
            self._prime_template = torch.zeros(1, len(self._prime_controlled))
            self._prime_executor = DynamicArmReferenceExecutor(self._prime_controlled, self._control_dt)
            self._prime_executor.load(names_mjlab, route_mjlab, route.interpolation_dt)
        if self._prime_executor is not None:
            # Park at the FLIP's own final waypoint, not the live interpolation: the parked configuration is
            # what ``_step_mpc`` seeds plan-0 with, and ``_commit_reach`` holds the visit in APPROACH until the
            # executor has actually played the arm there, so the two agree by the time the seed is used.
            arm_cmd = self._prime_executor.command(self._prime_template, self._prime_controlled)[0].numpy()
            self._park_arm_at(dict(zip(self._prime_controlled, arm_cmd.tolist())))

    def _hold_only_arm_ref(self) -> dict:
        """Arm command for the mission exits that run NO handler (TERMINATE, ``_parked_done``).

        Those paths return before ``_dispatch``, so they miss both the ``_arm_hold`` merge and the slew gate.
        An empty ``arm_ref`` is not a hold -- the harness fills every absent joint with ``_arm_home_pose``
        and servos there -- so returning ``{}`` teleported the arm from the pose RETRACT parked it at
        (measured 2.2 rad on ``left_shoulder_2``, v2/front_back_close/seed 47) to planning home in one tick.
        Empty when nothing was ever parked, which is the pre-park behaviour unchanged.
        """
        return self._slew_arm_ref(dict(self._arm_hold or {}))

    def _slew_arm_ref(self, arm_ref: dict) -> dict:
        """Rate-limit the emitted arm reference against the previous tick's, per joint (``_MAX_ARM_SLEW_RAD_S``).

        A joint absent from the previous tick has no anchor and passes through unclamped -- that is the arm
        entering the command set for the first time, not a discontinuity. The anchor is the COMMAND, not the
        measured qpos: clamping toward measurement would fight the deliberate gravity pre-bend, which exists
        precisely to make the command differ from where the arm currently is.
        """
        prev = self._arm_ref_prev
        if prev:
            step = _MAX_ARM_SLEW_RAD_S * self._control_dt
            arm_ref = {j: (v if j not in prev else min(max(v, prev[j] - step), prev[j] + step))
                       for j, v in arm_ref.items()}
        self._arm_ref_prev = arm_ref
        return arm_ref

    def _dispatch(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base):
        """Run the current state's handler and apply its returned transition through ``_transition`` (the
        single writer). Handlers return ``(base_twist, arm_ref, cam, next_state|None)``; ``None`` = stay.
        Shared by the parked ``step`` and the walk ``_walk_step`` -- ONE FSM driver for both paths.

        Also the rear prime's ONLY seam, for two reasons. Polling here rather than beside the trigger in
        ``_handle_approach`` means no branch can return past an in-flight reply: the results queue is shared
        with fast IK, whose ``poll_fast_ik`` asserts on the tag and would kill the rollout on an orphan. And
        an ABSENT joint is not a hold -- the harness fills it with ``_arm_home_pose`` and actively servos the
        arm back -- so the parked configuration must be re-emitted on every tick that omits it, of which
        ``_handle_approach`` alone has nine. The merge is per joint and ``arm_ref`` wins, so any handler that
        DOES emit (tracker, retract bridge, held grasp) still owns its own joints.

        Also the single rate gate on the emitted command (``_slew_arm_ref``): this is the one place that sees
        every reference source, so it is the only place a source HANDOFF can be bounded.
        """
        self._tick_prime()
        twist, arm_ref, cam, nxt = self._handlers[self._phase](
            base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base)
        if self._arm_hold is not None:
            # PER JOINT, not all-or-nothing: a handler that emits for ONE arm (the tracker, whose route is
            # single-arm on a solo visit) used to drop the other arm out of ``arm_ref`` entirely, and the
            # harness fills an absent joint with ``_arm_home_pose`` -- so the arm parked holding the previous
            # visit's cube snapped to planning home the moment this visit's route installed.
            arm_ref = {**self._arm_hold, **arm_ref}
        arm_ref = self._slew_arm_ref(arm_ref)
        if _ARM_DEBUG:
            meas = proprio_qpos if isinstance(proprio_qpos, dict) else {}
            row = {"phase": self._phase, "next": nxt, "t": self._phase_t,
                   "ref": {j: round(float(v), 4) for j, v in sorted(arm_ref.items())},
                   "meas": {j: round(float(v), 4) for j, v in sorted(meas.items())}}
            print(f"[armcsv] {row}", flush=True)
        if nxt is not None:
            self._transition(nxt)
        else:
            # Handlers inspect the elapsed tick count before returning. Advance only after a
            # non-transition tick so a fresh state always observes ``_phase_t == 0`` once.
            self._phase_t += 1
        return twist, arm_ref, cam

    def _navigate(self, x_base: float, y_base: float):
        """Locomotion primitive: ONE mover tick toward a base-frame target; returns ``(twist, settled)``.
        ALWAYS ticks the mover (invariant: never return a twist without advancing the accel ramp -- a guard
        that returned BEFORE ``compute`` froze the ramp and deadlocked the approach). Braking is latched by
        the caller via ``self._mover.stop()`` BEFORE this call, so the STOP ramps down here."""
        vx, vy, wz = self._mover.compute(x_base, y_base)
        return (vx, vy, wz), self._mover.settled

    def _walk_step(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """Walk tick: on the first tick (state ``None``) seed the pending set and ALWAYS enter ACQUIRE
        (perception-first for every embodiment -- the old fixed-cam "straight to ``_select_next``" skip was
        the walk-before-see defect). Then run the shared FSM ``_dispatch``. TERMINATE holds zero base twist.
        Sweep-done is POLICY-owned (``_gaze.full_pass``); ``observed_cube_pose`` is the accrued ``_belief``."""
        if self._phase is None:
            self._pending = {m.name for m in self._scenario.pick}
            self._anchors_visited, self._search_anchor = set(), None
            self._transition(_ACQUIRE)
        if self._phase == _TERMINATE:
            return (0.0, 0.0, 0.0), self._hold_only_arm_ref(), {}
        phase = self._phase
        twist, arm_ref, cam = self._dispatch(
            base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base)
        return twist, arm_ref, cam

    def _handle_acquire(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """ACQUIRE (perception-first, ALL robots parked + walk): search+decide before any manipulation.

        Caps-off (parked, ``not _can_drive``) short-circuits AFTER the gaze gate: reach the set
        reachable from the standing stance (parked EXTEND self-plans it) or FAIL-loud + TERMINATE -- it
        cannot DISCOVER/APPROACH. Walk (caps on) commits in place or routes via ``_select_next``.

        Actuated head: sweep the gimbals ALONE (base still) through the known support anchors. A complete pass
        means every pending cube is visible or its expected support was checked empty; camera tracking continues
        afterward but does not delay the decision. Bounded by ``_ACQUIRE_LOCK_TIMEOUT_TICKS`` so a non-converging
        aim cannot hang. Fixed cam: no sweep, decide immediately. Once visibility is available, decide from the
        current live base pose without an artificial base-settle delay:
          * a pending cube reachable from HERE, and this stance was NOT just handed off by DISCOVER
            (``_from_discover``) -> commit the whole reachable SET in place (bimanual when >=2) WITHOUT
            walking (the spawn stance was already verified feasible; cuRobo re-checks in EXTEND);
          * a cube seen but out of reach, OR reachable from a DISCOVER hand-off -> ``_select_next``
            (-> APPROACH: walk to point C, not the anchor DISCOVER stopped at);
          * nothing seen -> ``_select_next`` (-> DISCOVER: drive to the next viewpoint; else FAIL/TERMINATE).
        The ``_from_discover`` exclusion matters only for a single-facing robot (g1): DISCOVER's raw-anchor
        walk can land it inside the generous ``_reachable_cubes`` bound before the base ever squares up to
        the cube itself, so an unconditional in-place commit here would silently skip APPROACH's point-C
        target and bearing correction -- the base would commit from wherever DISCOVER happened to stop,
        not from the geometrically-corrected stance (live-test finding: g1's final approach stance/bearing
        was unaffected by the point-C fix because this branch bypassed it entirely)."""
        if self._gaze is not None:
            self._acquire_ticks += 1
            if not self._gaze.settled and self._acquire_ticks < _ACQUIRE_LOCK_TIMEOUT_TICKS:
                self._acquire_gaze_wait_ticks += 1
                return (0.0, 0.0, 0.0), {}, {}, None             # sweep, THEN slew gaze onto the cube; base still
        via_discover, self._from_discover = self._from_discover, False   # consume once per ACQUIRE entry
        # DECIDE. Plan-in-place-first: commit the whole set of cubes already reachable from this stance.
        reachable = self._reachable_cubes(base_pose, observed_cube_pose)
        if not self._can_drive:
            # PARKED (caps off): cannot search or drive to a better stance -- reach whatever is reachable
            # from HERE and let the parked EXTEND self-plan the visible scene (its own ``_parked_poses``
            # belief + replan timeout; do NOT ``_commit_reach``, that path stays byte-identical), else
            # FAIL-loud with an explicit verdict (was a silent ``_parked_done`` hang). Parked terminates
            # after one visit (RETRACT -> ``_parked_done``), so this decision runs ONCE -- no held-cube re-fail.
            if reachable:
                return (0.0, 0.0, 0.0), {}, {}, _EXTEND
            for cube in self._pending:
                self._failed.setdefault(cube, "unreachable from parked stance (cannot drive)")
            self._parked_done = True                         # terminal: step() short-circuits before dispatch
            return (0.0, 0.0, 0.0), {}, {}, _TERMINATE
        if reachable and not via_discover:
            cube = min(reachable, key=lambda c: self._base_frame_xy(base_pose, reachable[c][0])[2])
            self._begin_visit(base_pose, observed_cube_pose, cube)
            if self._field_stance_candidates:
                # Only treatment candidates divert this direct in-place commit. No candidate means exact
                # legacy behavior, including robust current stances and bimanual in-place plan-0.
                return (0.0, 0.0, 0.0), {}, {}, _APPROACH
            x_b, y_b, dist = self._base_frame_xy(base_pose, self._latch_cube_pose[0])
            return self._commit_reach(base_pose, observed_cube_pose, x_b, y_b, dist)
        return (0.0, 0.0, 0.0), {}, {}, self._select_next(base_pose, observed_cube_pose)

    def _handle_discover(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """DISCOVER: a pending cube is hidden -- out of camera RANGE (too far, beyond the FAR clip) or, for a
        fixed forward camera, out of ANGLE (behind the base). DRIVE the base toward the nearest unvisited
        support anchor via the shared ``_navigate`` mover: the mover TURNs to face the anchor first (reveals
        an ANGLE-hidden cube -- a fixed camera now points at it) then WALKs to close the gap (reveals a
        RANGE-hidden cube as it enters the FAR clip). The instant ANY pending cube frames, hand back to
        ACQUIRE (re-perceive+commit). ONE primitive for fixed and gimbal cameras: a gimbal already slews
        every ANGLE at ``observe()``, so its sole hidden cause is RANGE, which the walk closes -- there is no
        longer a ``gaze_cams -> TERMINATE`` give-up (it conflated angle-ROM exhaustion with range, quitting on
        a cube a few steps of walking would reveal). Brake a reach-radius short of the anchor edge so the base
        does not drive onto the support; still unseen once settled there -> the cube is unseeable from this
        viewpoint, so mark the anchor visited and try the next. Anchors exhausted -> the remaining cubes are
        genuinely unseeable -> TERMINATE. Nearest anchor is chosen by DISTANCE (we drive to it now, not just
        yaw in place). Returns ``(twist, {}, {}, next_state|None)``.

        Assumes an open drive path to the anchor (no obstacle avoidance in the mover): the settle-brake
        happens at the standoff, and a base spawned already inside the standoff of an ANGLE-hidden cube would
        brake before turning to reveal it -- unexercised (every bench scenario spawns >standoff from its
        anchors), deferred to an obstacle-aware mover."""
        if any(c in observed_cube_pose for c in self._pending):
            self._from_discover = True    # force ACQUIRE's DECIDE to route via _select_next/APPROACH (point C),
            #   not commit in place from the anchor-facing stance DISCOVER's walk left the base at -- see
            #   _handle_acquire's ``via_discover`` guard.
            return (0.0, 0.0, 0.0), {}, {}, _ACQUIRE     # a pending cube framed -> re-perceive+commit via ACQUIRE
        remaining = self._unvisited_anchors(base_pose)
        if not remaining:
            for cube in self._pending:
                self._failed.setdefault(cube, "unseen (all viewpoints exhausted)")
            return (0.0, 0.0, 0.0), {}, {}, _TERMINATE           # anchors exhausted, rest unseeable -> FAIL(unseen)
        if self._search_anchor is None:
            # A completed target already proved its support was visible. Prefer a fresh support before
            # re-searching that same one, otherwise G1 can turn back toward the front table after servicing
            # it while the only pending cube is behind. Served supports remain candidates as a fallback:
            # another unseen target may share one.
            name, anchor, half_w = min(
                remaining,
                key=lambda na: (na[0] in self._anchors_served,
                                self._base_frame_xy(base_pose, na[1])[2]),
            )
            self._search_anchor, self._search_anchor_pos = name, anchor
            # Framing standoff applies to FIXED forward cameras only (``_needs_reface``): a gimbal re-aims in
            # place, so making it walk farther back buys nothing and costs time.
            self._search_anchor_standoff = (_framing_standoff(half_w) if self._needs_reface
                                            else _REACH_RADIUS_M)
            self._mover.retarget()                               # fresh facing + ramp for this anchor visit
        self._nav_target_w = np.asarray(self._search_anchor_pos, dtype=float)
        x_b, y_b, dist = self._base_frame_xy(base_pose, self._search_anchor_pos)
        # Brake only once the mover has TURNED to FACE the anchor (state WALK == facing achieved), not merely
        # once within translation radius. A fixed forward camera must be POINTED at the support to frame its
        # cubes; a near-but-BEHIND anchor (front_back_close: the rear table sits ~0.17 m behind the between-
        # tables stance) would otherwise trip the proximity brake on tick 1, freezing the base facing away so
        # the camera never looks at it and the cube is wrongly marked unseeable.
        if (self._needs_reface and self._mover.state == "WALK"
                and x_b > 0.0 and dist < self._search_anchor_standoff):
            # A rear support can start inside the normal discovery brake radius. The body has already turned
            # toward it, but at this near stance G1's fixed camera cannot frame the tabletop cube. Reverse
            # only until the camera has a valid view distance, keeping the support ahead and increasing
            # clearance from it. The threshold is the FRAMING standoff, not a fixed clip-plane guard: braking
            # alone cannot fix a base that is ALREADY too close, which is why raising the brake radius on its
            # own measured no change on the g1 cell this addresses.
            twist, _ = self._navigate(x_b, y_b)
            return (-_BACKOFF_VX, 0.0, twist[2]), {}, {}, None
        if dist <= self._search_anchor_standoff and self._mover.state == "WALK":
            self._mover.stop()                                   # standoff reached AND facing it: brake
        twist, settled = self._navigate(x_b, y_b)
        if settled:
            self._anchors_visited.add(self._search_anchor)       # arrived, still no cube framed -> unseeable here
            self._search_anchor = None                           # next tick: re-check belief / next anchor
            return (0.0, 0.0, 0.0), {}, {}, None
        return twist, {}, {}, None

    def _handle_approach(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """APPROACH: drive the base toward POINT C (``_point_c``) -- the FIXED ANCHOR (table edge/shelf
        tier/hand marker, ``_nearest_anchor``) that ``_walk_current`` rests on, shifted along the anchor's
        own tangent to the cube's live lateral position -- not the anchor point itself, and not the cube's
        raw pose either. Human-analogy: walking up to an object on a desk, you square up to a point on the
        desk edge ahead of the object, not the desk's own fixed corner. Drives until the base-to-C planar
        distance enters the arm's reach envelope (``_REACH_RADIUS_M``) and the mover settles, then pins the
        reaching arm and hands off to EXTEND. TRIAL rationale: the scene's object distributions are built
        relative to these anchors (MEMORY.md 2026-07-15: shelf tiers are shoulder-relative like the table,
        far bimanual walks in to the anchor-relative centroid), so the anchor's standoff line is the stance
        cuRobo was actually tuned to reach from -- point C keeps that standoff distance while removing the
        residual bearing to an off-anchor cube. Anchor identity resolved ONCE per target (cached in
        ``_approach_anchor_pos``/``_approach_anchor_tangent``/``_approach_anchor_half_w``, cleared by
        ``_select_next``); point C itself is recomputed fresh each tick from the cheap cached tangent (one
        dot product + one clip) so it tracks the live cube belief without rescanning ``known_anchors``.

        Gate is pure PROXIMITY -- the learned reachability MLP was removed (obstacle-blind, it green-lit
        stances cuRobo could not plan). cuRobo does the real feasibility check in EXTEND, base-facing-
        independent, so nothing downstream needs the base to have specifically faced the cube vs. the anchor.
        Arm choice is GEOMETRIC: the cube's base-frame y-sign (+y = left half -> left arm), unless that arm is
        already OCCUPIED (bug 4: holding a cube from an earlier visit) -- then use the free arm. Commits to a
        remembered cube pose so a cube blinking out of a fixed frustum mid-approach does not deadlock the visit.
        Returns ``(twist, {}, {}, next_state|None)``."""
        cube = self._walk_current
        if self._field_stance_target_w is None and (
                self._field_stance_probe is not None
                or self._field_stance_candidate_i < len(self._field_stance_candidates)):
            approved = self._probe_field_stance(base_pose)
            if approved is None:
                return (0.0, 0.0, 0.0), {}, {}, None
            if approved:
                self._mover.retarget()
            # Rejected field candidates fall through to legacy Point-C on this tick.
        if self._field_stance_target_w is not None:
            # Candidate was fast-IK approved before locomotion. It is capability-aligned,
            # so this is translation without the body reorientation that would erase V2's camera advantage.
            # Robust field scores are the minimum over a +/-5 cm base neighborhood. Demand that same
            # reachable tolerance, not the 1.5 cm contour-adjustment precision the learned walker cannot
            # reliably settle to; full cuRobo revalidates from the actual pose before arm motion.
            x_field, y_field, dist_field = self._base_frame_xy(base_pose, self._field_stance_target_w)
            if dist_field <= _FIELD_TARGET_ARRIVAL_M:
                self._mover.stop()
            twist, settled = self._navigate(x_field, y_field)
            if settled:
                cube_w = self._latch_cube_pose[0] if self._latch_cube_pose is not None else observed_cube_pose[cube][0]
                x_cube, y_cube, dist_cube = self._base_frame_xy(base_pose, cube_w)
                return self._commit_reach(base_pose, observed_cube_pose, x_cube, y_cube, dist_cube)
            return twist, {}, {}, None
        if self._stance_turn_target_w is not None:
            x_turn, y_turn, _ = self._base_frame_xy(base_pose, self._stance_turn_target_w)
            if self._mover.state == "WALK":
                self._mover.stop()      # target heading reached; keep normal rate-limited stop, never translate
            twist, settled = self._navigate(x_turn, y_turn)
            if settled:
                self._stance_turn_target_w = None
                cube_w = self._latch_cube_pose[0] if self._latch_cube_pose is not None else observed_cube_pose[cube][0]
                x_cube, y_cube, dist_cube = self._base_frame_xy(base_pose, cube_w)
                return self._commit_reach(base_pose, observed_cube_pose, x_cube, y_cube, dist_cube)
            return twist, {}, {}, None
        if self._side_stance_target_w is not None:
            # No-route recovery: move parallel to the bounded support face, never toward it. This is an
            # explicit, short side adjustment, not another target-directed walk that could undo the offset.
            x_side, y_side, dist_side = self._base_frame_xy(base_pose, self._side_stance_target_w)
            self._side_stance_elapsed += self._control_dt
            if dist_side <= _SIDE_STANCE_DONE_M or self._side_stance_elapsed >= _SIDE_STANCE_TIMEOUT_S:
                self._side_stance_target_w = None
                cube_w = self._latch_cube_pose[0] if self._latch_cube_pose is not None else observed_cube_pose[cube][0]
                x_cube, y_cube, dist_cube = self._base_frame_xy(base_pose, cube_w)
                return self._commit_reach(base_pose, observed_cube_pose, x_cube, y_cube, dist_cube)
            speed = min(_SIDE_STANCE_SPEED_M_S, dist_side / self._control_dt)
            return (speed * x_side / dist_side, speed * y_side / dist_side, 0.0), {}, {}, None
        # geom_floor: the ONE final "close enough to stop" distance. Hoisted here so the SPAWN-READY
        # shortcut and the normal drive-in share the exact same threshold -- see next comment.
        # The reface branch is kept (single-camera G1 may yet need its own floor) but now carries the same
        # 0.20 as everyone else: at 0.35 it OVERRODE g1's own descriptor minimum (walk_min_standoff_m=0.24,
        # planner.py) by 0.11 m and, with the reface branch also zeroing walk_appr, parked the SHORTEST-arm
        # robot FURTHEST out -- ~0.35 m against a ~0.42 m envelope, i.e. reaching at ~83% extension, where
        # the arm has least authority and its COM excursion swings the floating base (user-observed).
        # Either way ``minimum_stop`` (0.24 + ~0.027 braking) binds, so the realized stance is ~0.27 m --
        # identical to what v2/v2_fixed already stop at. Visibility is NOT the casualty: the proximity latch
        # fires earlier at _REACH_RADIUS_M, commit-time facing is angular (_REACH_FACE_TOL_RAD, unaffected by
        # closing in), and the "cube dropped below the near frustum after the latch" case is already handled
        # below by driving to the remembered static pose.
        geom_floor = 0.20 if self._needs_reface else _GEOMETRIC_FLOOR_M
        # SPAWN-READY SHORTCUT: if the base is ALREADY inside the FINAL standoff (geom_floor) AND facing the
        # anchor, skip the drive entirely and commit. Gated on ``geom_floor``, not the wider reach-envelope
        # radius (``_REACH_RADIUS_M``): DISCOVER's own walk-in brake (``_handle_discover``) also stops at
        # ``_REACH_RADIUS_M``, so gating this shortcut on that same radius made it fire on ~every DISCOVER
        # hand-off too, before the base ever drove the remaining gap down to ``geom_floor`` -- silently
        # dead-ending the NORMAL-approach walk-in below for any scenario needing a DISCOVER phase first
        # (found live: v2 stopped much farther than ``_GEOMETRIC_FLOOR_M``/``_WALK_APPROACH_M`` implied).
        # Gating on ``geom_floor`` instead means the shortcut now only fires for a TRUE close spawn (v2's
        # front_back_close spawns ~0.25 m, just outside ``geom_floor`` -- it now falls through and drives
        # the small remaining gap in, rather than freezing at spawn pose; not a back-off cycle, since
        # ``latch_dist`` there is still above the humanoid's configured minimum standoff -- see the TOO-CLOSE
        # branch below).
        cube_w = self._latch_cube_pose[0] if self._latch_cube_pose is not None else (
            observed_cube_pose[cube][0] if cube in observed_cube_pose else None)
        if cube_w is not None and self._approach_anchor_pos is None:
            (self._approach_anchor_name, self._approach_anchor_pos, self._approach_anchor_tangent,
             self._approach_anchor_half_w, self._approach_support_normal_w) = self._nearest_anchor(base_pose[0], cube_w)
        anchor_w = self._approach_anchor_pos
        support_clearance = self._support_clearance(base_pose)
        if (anchor_w is not None and self._latch_cube_pose is not None
                and self._phase_t == 0 and self._extra_standoff == 0.0):   # skip on a back-off RETRY: the
            #   in-place stance already failed to plan, so don't re-commit it -- fall through to drive back out
            self._nav_target_w = self._point_c(anchor_w, cube_w)
            x_b0, y_b0, dist0 = self._base_frame_xy(base_pose, self._nav_target_w)
            support_safe = (support_clearance is None
                            or support_clearance[0] >= support_clearance[1])
            if dist0 <= geom_floor and support_safe and self._faced(float(np.arctan2(y_b0, x_b0))):
                self._latch_dist = dist0                          # latch immediately so commit path is uniform
                self._mover.stop()                               # no drive needed; square-up owns the base
                self._square_ticks = 0
                # Every OTHER commit site in this function waits for ``_mover.settled`` (ramped command,
                # including yaw, actually reaches zero) before handing off to EXTEND -- committing here
                # unconditionally on the SAME tick as ``stop()`` skipped that ramp-down entirely, so a spawn
                # carrying residual base momentum could start extending the arm while still physically
                # drifting, visibly nudging/pushing the cube before the grasp geometry was valid (user-observed,
                # v2_fixed/left_right_close). ``_navigate`` here just polls the already-latched STOP ramp (no
                # new target driven to); once settled this fires next tick, matching the other commit sites.
                twist, settled = self._navigate(x_b0, y_b0)
                if settled:
                    return self._commit_reach(base_pose, observed_cube_pose, x_b0, y_b0, dist0)
                return twist, {}, {}, None
        if cube in observed_cube_pose:
            self._latch_cube_pose = self._pin_pose(observed_cube_pose[cube])   # re-pin to the live belief in frame
        elif self._latch_cube_pose is not None:
            # Cube seen earlier this visit but not in frame right now: keep approaching the remembered STATIC
            # pose. Both cases fixed-camera-only: a FAR cube swept out of the frustum (driving re-frames it),
            # or a CLOSE cube dropped below the near frustum after the latch (reach already committed).
            # Holding would deadlock. (The humanoid keeps the cube in frame, so v2/v2_fixed never hit this.)
            pass
        else:
            return (0.0, 0.0, 0.0), {}, {}, None                 # never seen this visit -> hold (DISCOVER's job)
        if self._approach_anchor_pos is None:                    # cube just latched this tick -> resolve now
            (self._approach_anchor_name, self._approach_anchor_pos, self._approach_anchor_tangent,
             self._approach_anchor_half_w, self._approach_support_normal_w) = self._nearest_anchor(base_pose[0], self._latch_cube_pose[0])
        anchor_w = self._approach_anchor_pos
        self._nav_target_w = self._point_c(anchor_w, self._latch_cube_pose[0])
        x_b, y_b, dist = self._base_frame_xy(base_pose, self._nav_target_w)
        self._maybe_prime_rear(base_pose, observed_cube_pose, self._latch_cube_pose[0], dist)
        support_clearance = self._support_clearance(base_pose)
        if self._latch_dist is None and dist <= _REACH_RADIUS_M:
            self._latch_dist = dist                              # proximity latch: base entered the reach envelope
        # TOO CLOSE -> back off: each descriptor supplies its own physically plannable normal standoff. Braking
        # above that threshold avoids the old immediate reverse while command ramp-down lands the base at the
        # configured stance. A failed route widens any robot's retry stance. Move opposite this target's selected
        # drive direction, not always body-reverse: a rear or lateral approach must retreat away from its support.
        # The bearing fallback covers an initially-too-close spawn before the mover has selected a direction.
        standoff = self._cfg.walk_min_standoff_m + self._extra_standoff
        if (self._latch_dist is not None and support_clearance is not None
                and support_clearance[0] < support_clearance[1]):
            # Root is too close to the physical support face for its projected torso footprint. Retreat along
            # the face normal (support -> robot), not the target-drive vector: geometry remains correct when
            # a cross-track correction or a non-cardinal target direction is active.
            self._mover.stop()
            normal_b = support_clearance[3]
            return (normal_b[0] * _BACKOFF_VX, normal_b[1] * _BACKOFF_VX, 0.0), {}, {}, None
        if self._latch_dist is not None and self._latch_dist < standoff:
            bearing = float(np.arctan2(y_b, x_b))
            wz = self._square_up_wz(bearing)                     # steer to face the cube (square up before commit)
            if dist < standoff:
                self._mover.stop()                               # kill any forward drive; back-off owns the base
                self._square_ticks += 1                          # back-off + square-up shares the timeout budget
                if self._square_ticks >= _REACH_SQUARE_TIMEOUT_TICKS:
                    self._square_ticks = 0                      # commit anyway (never hang)
                    return self._commit_reach(base_pose, observed_cube_pose, x_b, y_b, dist)
                direction = np.asarray(self._mover.direction, dtype=float)
                if np.linalg.norm(direction) < 1e-6:
                    direction = np.array([x_b, y_b], dtype=float)
                    direction /= max(np.linalg.norm(direction), 1e-6)
                return (-direction[0] * _BACKOFF_VX, -direction[1] * _BACKOFF_VX, wz), {}, {}, None
            if not self._faced(bearing):
                self._mover.stop()                               # at standoff but oblique: turn in place to face
                self._square_ticks += 1
                if self._square_ticks >= _REACH_SQUARE_TIMEOUT_TICKS:
                    self._square_ticks = 0
                    return self._commit_reach(base_pose, observed_cube_pose, x_b, y_b, dist)
                return (0.0, 0.0, wz), {}, {}, None
            self._square_ticks = 0                               # squared up -- reset for next visit
            return self._commit_reach(base_pose, observed_cube_pose, x_b, y_b, dist)
        # NORMAL forward approach. Keep walking _WALK_APPROACH_M past the latch edge before braking (physics reach
        # droops ~0.10 m at the edge). Include descriptor standoff plus command-ramp braking distance so the
        # realized stop remains outside the same no-reverse guard. Brake only once FACING the cube: WALK reached (far visits,
        # drove in on-heading) OR heading under a loose tolerance (close turn-in-place visits never reach WALK --
        # g1 sway is at or past the mover's FACE_TOL_RAD handoff; see _APPROACH_FACE_TOL_RAD).
        # Applies to EVERY robot now; the old ``0.00 if self._needs_reface`` opt-out was the bug. Zeroing it
        # left the FIRST
        # max() term as the raw ``_latch_dist`` (~_REACH_RADIUS_M = 0.35, wherever the proximity latch fired),
        # which then DOMINATED both other terms -- so g1 braked at the reach EDGE and no floor/standoff tuning
        # could move it. That is the exact droop this constant was introduced to fix, and g1 (shortest arm,
        # ~0.42 m envelope) was the only robot opted out of it: reaching at ~83% extension, least arm
        # authority, COM excursion swinging the floating base (user-observed).
        walk_appr = _WALK_APPROACH_M
        minimum_stop = standoff + self._mover.braking_distance_m
        stop_dist = max(self._latch_dist - walk_appr, geom_floor, minimum_stop) \
            if self._latch_dist is not None else None
        facing = (not self._needs_reface) or self._mover.state == "WALK" \
            or abs(np.arctan2(y_b, x_b)) < _APPROACH_FACE_TOL_RAD
        support_stop = (support_clearance is not None
                        and support_clearance[0] <= support_clearance[2])
        if self._latch_dist is None and support_stop:
            # Physical support clearance is a valid safe-standoff latch even outside the old point-C
            # radius. EXTEND's cuRobo solve is the actual reachability verdict from this safe stance.
            self._latch_dist = dist
        if (support_stop and self._cfg.walk_precommit_contour_align
                and self._approach_anchor_tangent is not None
                and self._approach_support_normal_w is not None):
            # The support brake protects the torso footprint along the face normal, but it can fire before a
            # diagonal approach reaches Point C's tangent coordinate. Stop advancing toward the table, then
            # translate along its safe clearance contour to remove that lateral miss before committing. This
            # preserves the same support clearance while shortening the actual base-to-cube distance.
            tangent_w = np.asarray(self._approach_anchor_tangent, dtype=float)
            normal_w = np.asarray(self._approach_support_normal_w, dtype=float)
            contour_w = tangent_w - np.dot(tangent_w, normal_w) * normal_w
            contour_norm = float(np.linalg.norm(contour_w))
            if contour_norm >= 1e-6:
                contour_w /= contour_norm
                lateral_error = float(np.dot(self._nav_target_w[:2] - base_pose[0][:2], contour_w))
                if abs(lateral_error) > _SIDE_STANCE_DONE_M:
                    self._mover.stop()
                    target_w = np.asarray(base_pose[0], dtype=float).copy()
                    target_w[:2] += lateral_error * contour_w
                    target_w[:2] += max(0.0, support_clearance[1] - support_clearance[0]) * normal_w
                    target_w[:2] += _SIDE_STANCE_CLEARANCE_BUFFER_M * normal_w
                    self._side_stance_target_w = target_w
                    self._side_stance_elapsed = 0.0
                    return (0.0, 0.0, 0.0), {}, {}, None
        if ((stop_dist is not None and dist <= stop_dist) or support_stop) and facing:
            self._mover.stop()
        twist, settled = self._navigate(x_b, y_b)
        if self._latch_dist is not None and settled:
            # Square up before committing (g1 only, ``_faced``/``_needs_reface``): the loose
            # WALK/_APPROACH_FACE_TOL brake can settle up to ~14 deg off the cube (oblique reach under carry),
            # which is fine for reachability but can leave the cube outside g1's single fixed frustum.
            bearing = float(np.arctan2(y_b, x_b))
            if not self._faced(bearing):
                self._mover.stop()
                self._square_ticks += 1
                if self._square_ticks >= _REACH_SQUARE_TIMEOUT_TICKS:
                    self._square_ticks = 0
                    return self._commit_reach(base_pose, observed_cube_pose, x_b, y_b, dist)
                return (0.0, 0.0, self._square_up_wz(bearing)), {}, {}, None
            self._square_ticks = 0                               # squared up -- reset for next visit
            return self._commit_reach(base_pose, observed_cube_pose, x_b, y_b, dist)
        return twist, {}, {}, None

    def _faced(self, bearing: float) -> bool:
        """True when the base is aimed at the cube closely enough to commit, OR the robot needs no base
        reface at all. Only a single-forward-camera robot (``_needs_reface``, e.g. g1) squares its base to
        the cube -- for frustum visibility, NOT reachability (cuRobo proves reach from the actual stance in
        EXTEND). Twin/gimbal humanoids (v2/v2_fixed) reach a lateral/rear cube in place, so they are always
        'faced'; skipping the square-up kills the wasted turn AND the bimanual-stance skew."""
        return not self._needs_reface or abs(bearing) < _REACH_FACE_TOL_RAD

    def _square_up_wz(self, bearing: float) -> float:
        """Yaw command to null a base->cube ``bearing`` (rad) before committing a close reach. Zero when the
        robot needs no reface (``not _needs_reface``) or inside the facing tol; else
        ``sign(bearing) * |k*bearing|`` clamped to ``[_REACH_TURN_WZ_MIN, _REACH_TURN_WZ_MAX]``. MIN clears the
        RL base's yaw deadzone (a sub-floor command does not rotate the base); MAX caps the turn rate. Same
        sign convention as the mover: +bearing (cube to the left) -> +wz (turn left)."""
        if not self._needs_reface or abs(bearing) < _REACH_FACE_TOL_RAD:
            return 0.0
        return float(np.sign(bearing) * np.clip(
            abs(_REACH_STEER_K * bearing), _REACH_TURN_WZ_MIN, _REACH_TURN_WZ_MAX))

    def _commit_reach(self, base_pose, observed_cube_pose, x_b, y_b, dist):
        """Commit the reach visit: clear the per-visit executor, pick the reachable-cube set + arm availability, and
        hand to EXTEND. Shared by the normal-approach settle and the back-off arrival so both commit identically.

        Start a CLEAN per-visit reach: drop the route AND executor/controlled set. The active-arm set is pinned for
        the whole visit by ``_forced_assignment``, so ``_install``'s "arm set stable mid-reach" invariant holds --
        but a NEW visit may pin a different set, so the executor is rebuilt. Arm choice is the PLANNER's job
        (feasibility), NOT a geometric guess: ``_forced_assignment=None`` when both arms free lets the ladder
        assign across arms (bimanual for >=2, or whichever arm solves for a lone cube); the ONLY policy constraint
        is AVAILABILITY -- a held arm (bug 4) leaves one free arm, which is pinned.

        This can now DECLINE. APPROACH spans many ticks of walking, so a set that was live at the DECIDE tick
        can be stale by the time the base settles here -- a real branch, not a defensive one. The gated set
        being empty means the robot is about to reach for something it cannot currently see, so bounce back to
        ACQUIRE (whose gaze-settle wait re-aims at the target) instead. Bounded by ``_COMMIT_STALE_MAX_TICKS``
        so a target the rig genuinely cannot re-frame FAILs loud rather than livelocking APPROACH->ACQUIRE.
        The former unconditional ``or {_walk_current: _latch_cube_pose}`` fallback is exactly what let a
        remembered pose authorize a blind reach; it is deliberately gone."""
        if self._prime_gen is not None or (
                self._prime_executor is not None
                and self._prime_executor.cursor_s < self._prime_executor.duration_s):
            # Hold in APPROACH until the rear flip is both planned AND physically played out: ``_step_mpc``
            # seeds plan-0 with ``_prime_q``, and seeding at a configuration the arm has not reached yet is
            # the exact lie that made the first version of this drop the carried cube. Bounded -- the
            # executor's cursor advances every tick from ``_dispatch``.
            return (0.0, 0.0, 0.0), {}, {}, None
        # ``_walk_current`` specifically must survive the gate, not merely SOME cube: the single-free-arm
        # ``_forced_assignment`` below pins an arm to it, so committing a set without it would pin an arm to a
        # cube this visit never planned for. Re-ACQUIRE instead, which re-runs the choice from scratch.
        if os.environ.get("REACH_DEBUG_STANCE"):
            # Commit-time stance, on EVERY commit (the pre-existing REACH_DEBUG_STANCE print fires only on the
            # terminal give-up, which is far too late to see the stance inflate across retries).
            sc = self._support_clearance(base_pose)
            clr = f"root_clr={sc[0]:.3f} min_clr={sc[1]:.3f}" if sc is not None else "clr=None"
            print(f"[DEBUG_COMMIT] cube={self._walk_current!r} dist={dist:.3f}m "
                  f"standoff={self._cfg.walk_min_standoff_m + self._extra_standoff:.3f} "
                  f"latch={self._latch_dist} extra={self._extra_standoff:.3f} "
                  f"retries={self._extend_retries} tick={self._ticks} phase={self._phase} "
                  f"anchor={self._approach_anchor_name!r} {clr}",
                  file=sys.stderr)
        committed = self._reachable_cubes(base_pose, observed_cube_pose)
        if _COMMIT_DEBUG:
            # Every commit decision, PASSED ones included: the decline rate is ~1.7% of visits, far too rare to
            # attribute by re-running (and async cuRobo latency means a fixed seed does not reproduce the
            # trajectory anyway). Logging the margin on every visit turns that rare binary event into a
            # continuous distribution -- if the walk target's belief age routinely crowds
            # ``_BELIEF_FRESH_TICKS``, freshness is the binding gate; if ages are small except at declines,
            # it is not.
            print(f"[commitcsv] {{'target': {self._walk_current!r}, 'ok': "
                  f"{self._walk_current in committed}, 'cubes': " + repr({
                      c: (self._ticks - self._belief_tick.get(c, -_BELIEF_FRESH_TICKS - 1),
                          round(float(b[2]), 3),
                          # BEARING + vertical drop, added because distance alone did not separate the
                          # never-framed stuck visits from the ones that commit at the same 0.24-0.27 m.
                          round(float(np.degrees(np.arctan2(b[1], b[0]))), 1),
                          round(float(p[0][2] - base_pose[0][2]), 3), c in self._live)
                      for c in self._pending if (p := observed_cube_pose.get(c) or self._belief.get(c))
                      and (b := self._base_frame_xy(base_pose, p[0])) is not None
                  }) + "}", file=sys.stderr)
        if self._walk_current not in committed:
            if self._commit_stale_tick is None:
                self._commit_stale_tick = self._ticks
            waited = self._ticks - self._commit_stale_tick
            if waited < _COMMIT_STALE_MAX_TICKS:
                # DROP the declined cube from ``_belief`` before bouncing. ``_reachable_cubes`` (above) gates
                # on FRESHNESS while ``_select_next`` gates only on MEMBERSHIP, so a stale entry makes ACQUIRE
                # hand the cube straight back to APPROACH -- a livelock that burns the whole
                # ``_COMMIT_STALE_MAX_TICKS`` budget with the base stationary and DISCOVER unreachable
                # (``visible`` is never empty). Pruning here -- not in ``_select_next``, whose membership test
                # is also the routine "seen but far -> walk to it" path -- routes only the broken case to
                # DISCOVER, which is the one primitive that turns the base and re-frames the cube.
                self._belief.pop(self._walk_current, None)
                return (0.0, 0.0, 0.0), {}, {}, _ACQUIRE
            cls, detail = self._commit_reject_reason(base_pose, observed_cube_pose, self._walk_current)
            print(f"[reach] WARNING: cube {self._walk_current!r} declined at commit -- {cls}"
                  f"{f' ({detail})' if detail else ''} -- after {waited} ticks of re-aim; skipping.",
                  file=sys.stderr)
            self._failed[self._walk_current] = f"not committable at commit ({cls})"
            self._pending.discard(self._walk_current)
            self._commit_stale_tick = None
            self._forced_assignment = None
            self._reach_poses = {}
            return (0.0, 0.0, 0.0), {}, {}, _ACQUIRE
        self._commit_stale_tick = None
        self._route = self._executor = self._template = None
        self._controlled = []
        # SNAPSHOT the committed set (see ``_pin_pose``): these are the poses this visit planned against, so
        # they must not follow the cubes afterwards, or any live-vs-committed comparison is identically zero.
        self._reach_poses = {cube: self._pin_pose(pose) for cube, pose in committed.items()}
        anchors = known_anchors(self._scenario, base_pose[0])
        self._reach_anchor_names = {
            cube: min(anchors, key=lambda na: np.linalg.norm(np.asarray(na[1]) - pose[0]))[0]
            for cube, pose in self._reach_poses.items()
        }
        free_arms = [a for a in ("L", "R") if a not in self._held]
        self._forced_assignment = (dict(self._field_stance_assignment)
                                   if self._field_stance_assignment is not None
                                   else None if len(free_arms) >= 2
                                   else {free_arms[0]: self._walk_current})
        self._extend_ik_ok = None                                # re-arm the fast-IK gate for this stance
        self._plan0_scene = None
        self._plan0_cube_pose = None
        self._final_replan_pending = False
        self._final_replan_scene = None
        self._final_replans = 0
        self._pregrasp_done = False
        self._extend_diverge_ticks = 0
        self._extend_diverge_replans = 0
        self._target_moved_replans = 0
        return (0.0, 0.0, 0.0), {}, {}, _EXTEND

    def _reachable_cubes(self, base_pose, belief):
        """``{cube: pose}`` for every pending, NOT-already-held cube reachable from the current stance. The
        reach visit commits to this whole set -- two reachable cubes -> bimanual reach (both arms at once); a
        cube out of reach waits for a later visit. Held cubes (bug 4) are excluded so a carried object is never
        re-grasped.

        Commit criterion for walk is VISIBILITY within a GENEROUS outer bound (``RobotDescriptor.reach_max_m``,
        defaulting to ``_REACH_MAX_M``), NOT the tight
        proximity gate. A parked policy cannot reposition, so it keeps the former all-visible stationary
        behavior and lets cuRobo report infeasibility. A walking policy applies the same outer bound to
        stationary and non-stationary scenes: after servicing one target, physics can leave its base near that
        target and far from another visible target. That target must route through ``_select_next`` ->
        APPROACH, not commit an in-place reach from the wrong table. The old tight ``dist <= _REACH_RADIUS_M`` gate
        wrongly rejected the spawn-reachable 0.382 m left_right_close cube (0.382 > 0.35), forcing an unwanted
        APPROACH/turn AND collapsing the bimanual set -- the reported bug this bound fixes.

        LIVENESS (see ``_BELIEF_FRESH_TICKS``): "in belief" alone means "framed at SOME point", which is not a
        licence to reach. Two further gates apply, and every caller of this method is a PRE-COMMIT decision, so
        gating here covers all of them at once:
          * FRESHNESS -- drop any cube not framed within ``_BELIEF_FRESH_TICKS``.
          * SIMULTANEITY -- a set of 2+ survivors additionally must be co-visible in THIS tick's live frame.
            This is what forbids a bimanual commit on a rig that can only ever frame one of the pair: during
            EXTEND the gaze locks per route side (``side_to_cam`` in ``_drive_gaze``, dropping any side with no
            corresponding gimbal), so a single-gimbal robot's second arm would be blind for the WHOLE reach by
            construction. Falling back to the largest live co-visible subset (rather than rejecting outright)
            keeps the visit alive as a single-arm reach instead of stalling the mission.
        Post-commit code is deliberately untouched: ``_reach_belief`` keeps its remembered-pose fallback for
        the reaching arm occluding its own target as the jaws close -- unavoidable geometry, not stale memory."""
        reach_max = self._cfg.reach_max_m if self._cfg.reach_max_m is not None else _REACH_MAX_M
        reachable = {cube: pose for cube in self._pending
                     if cube not in self._held.values()
                     and (pose := belief.get(cube)) is not None
                     and self._ticks - self._belief_tick.get(cube, -_BELIEF_FRESH_TICKS - 1) <= _BELIEF_FRESH_TICKS
                     and (not self._can_drive
                          or self._base_frame_xy(base_pose, pose[0])[2] <= reach_max)}
        if len(reachable) > 1:
            live = {cube: pose for cube, pose in reachable.items() if cube in self._live}
            if live:
                return live
            # Nothing co-visible this tick (every survivor is inside the freshness window but the rig is
            # between aims): keep the single nearest so the visit proceeds one-armed rather than committing a
            # pair the camera has never held together.
            nearest = min(reachable, key=lambda c: self._base_frame_xy(base_pose, reachable[c][0])[2])
            return {nearest: reachable[nearest]}
        return reachable

    def _commit_reject_reason(self, base_pose, belief, cube) -> tuple[str, str]:
        """Which of ``_reachable_cubes``' gates actually excluded ``cube``. Diagnostic only -- no behavior.

        Returns ``(class, detail)``. The CLASS goes in the mission verdict and must stay a closed, coarse
        vocabulary: ``dyn_aggregate.py`` histograms failures by the raw verdict string, so any measurement
        (an age in ticks, a distance in metres) embedded there splits one class into a bucket per value and
        destroys the count. The DETAIL carries those numbers to stderr instead.

        Exists because the four causes are NOT interchangeable and the single string they used to share
        ("stale belief") was wrong for three of them, so the class could not be acted on: ``no belief`` is a
        search failure, ``stale belief`` a gaze-scheduling failure, ``out of range`` an approach failure, and
        ``deselected`` is not a failure at all -- the cube passed every gate and lost the simultaneity
        tie-break to a nearer one. Re-evaluates rather than threading a reason out of the comprehension: this
        runs once per abandoned visit, so the duplicate predicate costs nothing and leaves the hot path
        untouched. Mirror of the gate above -- edit both together."""
        if cube in self._held.values():
            return "already held", ""
        pose = belief.get(cube)
        if pose is None:
            return "no belief", "never framed"
        age = self._ticks - self._belief_tick.get(cube, -_BELIEF_FRESH_TICKS - 1)
        if age > _BELIEF_FRESH_TICKS:
            return "stale belief", f"last framed {age} ticks ago > {_BELIEF_FRESH_TICKS}"
        dist = self._base_frame_xy(base_pose, pose[0])[2]
        reach_max = self._cfg.reach_max_m if self._cfg.reach_max_m is not None else _REACH_MAX_M
        if self._can_drive and dist > reach_max:
            return "out of range", f"{dist:.3f} m > {reach_max} m"
        return "deselected", "passed every gate; lost the simultaneity tie-break"

    def _extend_settled(self) -> bool:
        """Has the final-grasp reach route fully played out plus settle? Reads whichever
        execution primitive is active, so the walk FSM stays primitive-agnostic: the MPC/tracker path
        (``_step_mpc``, no executor -- it servos per tick) answers via ``JacobianReachTracker.extend_settled``;
        the executor path (kinematic + dynamic non-MPC) via ``cursor_s >= duration_s + settle``. Same wall-time
        semantics on both, so the extend->retract switch fires identically.

        Total on purpose: no executor means no route to have played out, so False. The FSM callers all
        guard on ``_route is not None`` first, but the headless grader polls this every reaching tick and
        an EXTEND back-off retry leaves ``_route``/``_executor`` None while ``last_route`` still lingers."""
        if self._mpc_track:
            return self._tracker is not None and self._tracker.extend_settled(self._cfg.dynamic_extend_settle_s)
        return (self._executor is not None
                and self._executor.cursor_s >= self._executor.duration_s + _REACH_SETTLE_S)

    def _hold_grasp_command(self, proprio_qpos=None, actual_tool_poses_base=None, base_quat=None) -> dict:
        """Track frozen final grasp pose while jaws close or return planning runs.

        The dynamic tracker keeps solving from measured joints instead of freezing a base-frame joint command:
        a small floating-base shift therefore changes the arm reference smoothly, not the held Cartesian pose.
        Non-MPC backends retain their final executor command.
        """
        if self._mpc_track and proprio_qpos is not None:
            gravity = self._gravity_in_base(base_quat) if self._gravity_comp else None
            return self._tracker.hold_step(
                gravity_in_base=gravity, actual_tool_poses_base=actual_tool_poses_base,
                measured_arm_qpos=proprio_qpos if isinstance(proprio_qpos, dict) else None)
        if self._held_grasp_command is not None:
            return dict(self._held_grasp_command)
        if self._mpc_track:
            return self._tracker.hold_command()
        arm_cmd = self._executor.command(self._template, self._controlled)[0].numpy()
        return dict(zip(self._controlled, arm_cmd.tolist()))

    def _load_retract_trajectory(self, trajectory_joint_names, trajectory_q_mjlab, trajectory_dt,
                                 controlled) -> None:
        """Install return path with a finite held-grasp -> planner-start continuity bridge.

        The asynchronous home route is collision-aware, but its first interpolated q can differ from the
        physical command held at grasp. Prepending a 0.30 s bridge preserves that exact waypoint at handoff
        and turns the mismatch into ordinary trajectory motion, eliminating the sudden gripper-height drop.
        """
        controlled = list(controlled)
        if self._executor is None:
            self._controlled = controlled
            self._template = torch.zeros(1, len(controlled))
            self._executor = DynamicArmReferenceExecutor(controlled, self._control_dt)
        else:
            assert controlled == self._controlled, (
                f"active-arm set changed at retract {self._controlled} -> {controlled}")
        index = {name: i for i, name in enumerate(trajectory_joint_names)}
        route_q = np.asarray(trajectory_q_mjlab, dtype=np.float32)[:, [index[name] for name in controlled]]
        if self._held_grasp_command is not None:
            held_q = np.asarray([self._held_grasp_command[name] for name in controlled], dtype=np.float32)
            n_join = max(1, int(np.ceil(_RETRACT_JOIN_S / trajectory_dt)),
                         int(np.ceil(np.max(np.abs(route_q[0] - held_q)) /
                                     (_RETRACT_JOIN_MAX_SPEED_RAD_S * trajectory_dt))))
            bridge = np.linspace(held_q, route_q[0], n_join + 1, dtype=np.float32)[:-1]
            route_q = np.vstack((bridge, route_q))
            assert np.array_equal(route_q[0], held_q), "retract must start at held grasp command"
        self._executor.load(controlled, route_q, trajectory_dt, start_s=0.0)

    def _start_retract(self, retract_route, proprio_qpos=None) -> None:
        """Install fresh async home route, or replay extend corridor only when planning genuinely failed."""
        self._retract_uses_tracker = False
        if retract_route is not None:
            self._route = retract_route
            if self._mpc_track:
                measured = proprio_qpos if isinstance(proprio_qpos, dict) else None
                self._tracker.begin_return(retract_route, measured)
                self._retract_uses_tracker = True
            else:
                names_mjlab, route_mjlab = self._route_to_mjlab(retract_route)
                self._load_retract_trajectory(names_mjlab, route_mjlab, retract_route.interpolation_dt,
                                              retract_route.controlled_joints)
            self._viz_route = retract_route
        else:
            names_mjlab, route_mjlab = self._route_to_mjlab(self._route)
            retract = route_mjlab[::-1].copy()
            self._load_retract_trajectory(names_mjlab, retract, self._route.interpolation_dt,
                                          self._route.controlled_joints)
            self._route = None

    def _submit_retract(self, base_pose, proprio_qpos, belief) -> str:
        """Submit grasp->home planning from measured physical arm qpos, without blocking the FSM. Returns the
        next state: ``RETRACT_PLAN`` (async plan queued -- poll it in RETRACT_PLAN) or ``RETRACT`` (legacy
        in-process parked reverses its already-collision-free extend corridor now, no plan to wait on).

        Dynamic planning must start at MuJoCo's measured state, never commanded qpos: the position servo and
        contacts own physical motion. The tracker bridges its frozen Cartesian grasp target into this route;
        it never writes qpos or teleports robot/object state.
        """
        self._retract_plan_ticks = 0            # start the RETRACT_PLAN wall (backstop the worker never returns)
        seed_q = self._seed_from_proprio(proprio_qpos)
        fixed_assignment = dict(self._route.assignment)
        if self._mpc_track:
            base_pos, base_quat = base_pose
            scene_dict, targets = reach_scene_and_targets(
                self._cfg, self._scenario, self._mj_model, base_pose, cube_poses=belief)
            assert targets, "committed reach lost every target before retract"
            self._plan0_gen = self._worker.submit_plan0(
                scene_dict, targets, base_pos, base_quat,
                fixed_assignment=fixed_assignment, seed_q=seed_q, goal="home")
        elif self._walk and self._session is not None:
            # SYNC WALK path (kinematic walk + sync dynamic): a warm in-process cuRobo session already drives
            # EXTEND, so plan the collision-aware grasp->home route HERE, synchronously, instead of spawning
            # a SECOND cuRobo runtime (``SpawnedReachWorker``) that contends the GPU with the in-process one
            # and was observed to hang -- never returning a route (g1 always; v2 far intermittently), which
            # stranded RETRACT_PLAN until the backstop reversed the corridor. In-process removes the worker,
            # the contention, and the stall. ``None`` (infeasible home) falls back to the reversed corridor.
            route = plan_reach_route(
                self._session, self._scenario, self._mj_model, _yaw_only(base_pose),
                cube_poses=belief, seed_q=seed_q, max_attempts=_REPLAN_MAX_ATTEMPTS,
                fixed_assignment=fixed_assignment, goal="home",
                height_pad=self._height_pad)
            self._start_retract(route)
            return _RETRACT
        else:
            if self._robot_name is None:
                # Legacy in-process parked callers provide only a warmed session, not worker reconstruction
                # keys. Their extend corridor is already collision-free, so reverse it without inventing a
                # second planner runtime. Dynamic/async callers provide names and use the async home plan.
                self._start_retract(None)
                return _RETRACT
            if self._retract_worker is None:
                self._retract_worker = SpawnedReachWorker(self._robot_name, self._scenario_name)
            yaw_base = _yaw_only(base_pose)
            scene_dict, targets = reach_scene_and_targets(
                self._cfg, self._scenario, self._mj_model, yaw_base, cube_poses=belief)
            assert targets, "committed reach lost every target before retract"
            gen = self._req_gen
            self._req_gen += 1
            request = ReachRequest(gen, scene_dict, targets,
                                   np.asarray(yaw_base[0], dtype=np.float32),
                                   np.asarray(yaw_base[1], dtype=np.float32),
                                   seed_q, None, fixed_assignment, goal="home")
            self._retract_worker.submit(request)
            self._retract_request_gen = gen
        return _RETRACT_PLAN

    def _poll_retract(self):
        """Return ``(ready, route)`` from the backend-specific transport; FSM behavior stays identical."""
        if self._mpc_track:
            return self._worker.poll_plan0(self._plan0_gen)
        for result in self._retract_worker.poll():
            if result.request_generation == self._retract_request_gen:
                return True, result.route
        return False, None

    def _reach_belief(self, observed_cube_pose):
        """Cube belief the reach commits to. Walk pins its visit targets at APPROACH (``_reach_poses``), falling
        back to a remembered pose if a fixed camera loses the cube mid-reach; parked accumulates every
        last-seen target so a camera loss during jaw close cannot erase the return-plan request."""
        if self._walk:
            return {c: (observed_cube_pose.get(c) or pose) for c, pose in self._reach_poses.items()}
        self._parked_poses.update(observed_cube_pose)
        return self._parked_poses

    def _stance_margin(self, field, base_pos, base_quat) -> float:
        """Worst active-target visible-reachable margin at one hypothetical base stance.

        This is a ranking proxy only.  For a free bimanual visit it evaluates both legal L/R pairings and
        keeps their better weakest target; for a carried-object visit it respects the forced free-arm
        assignment.  The later cuRobo fast-IK gate and collision-aware route still decide feasibility.
        """
        poses = self._reach_poses or {self._walk_current: self._latch_cube_pose}
        scores = {
            cube: {side: field.score_world(pose[0], base_pos, base_quat, side)
                   for side in ("L", "R")}
            for cube, pose in poses.items()
        }
        if self._forced_assignment:
            return min(scores[cube][side] for side, cube in self._forced_assignment.items()
                       if cube in scores)
        cubes = tuple(scores)
        if len(cubes) == 1:
            return max(scores[cubes[0]].values())
        if len(cubes) == 2:
            first, second = cubes
            return max(min(scores[first]["L"], scores[second]["R"]),
                       min(scores[first]["R"], scores[second]["L"]))
        return 0.0

    def _note_stance_spent(self, target_w) -> None:
        """Record a stance this episode has now occupied for ``_walk_current`` (see ``_STANCE_REVISIT_TOL_M``)."""
        if self._walk_current is None or target_w is None:
            return
        self._spent_stances.setdefault(self._walk_current, []).append(
            np.asarray(target_w, dtype=float)[:2].copy())

    def _stance_spent(self, target_w) -> bool:
        """True if ``target_w`` is within ``_STANCE_REVISIT_TOL_M`` of a stance already occupied for this cube."""
        return any(float(np.linalg.norm(np.asarray(target_w, dtype=float)[:2] - spent))
                   <= _STANCE_REVISIT_TOL_M
                   for spent in self._spent_stances.get(self._walk_current, ()))

    def _curated_side_stance(self, base_pose, contour_w, normal_w, clearance):
        """Choose one higher-margin bounded contour recovery after a proven no-route stance.

        Legacy Point-C recovery is baseline: this method runs only for its first side retry, keeps the
        exact support-normal clearance repair, samples ``Point-C +/- 12 cm`` only within the support's
        declared width, and returns ``None`` unless one target strictly improves robust visible-reachable
        margin.  This prevents an offline field from creating speculative locomotion; a missing/zero field
        follows legacy recovery.  cuRobo fast-IK still runs after arrival as comparison and runtime authority.

        Candidates already occupied for this cube in an EARLIER visit are dropped (``_stance_spent``): the
        margin score is a pure function of stance and cube pose, so an unfiltered re-derivation returns the
        same winner and walks the robot back to a stance it has already proven cannot grasp. Falling through
        to legacy radial back-off is strictly better than re-entering it.
        """
        if (self._side_stance_retries != 0
                or self._curated_stance_ik_rejected
                or os.environ.get("VISIBLE_REACHABLE_STANCE", "1") == "0"):
            return None
        if self._session is None and not self._mpc_track:
            return None                              # non-MPC async worker has no fast-IK message contract
        field = visible_reachable_field(self._cfg.name)
        if field is None:
            return None
        base_pos, base_quat = base_pose
        baseline = self._stance_margin(field, base_pos, base_quat)
        cube_w = self._latch_cube_pose[0]
        point_c = self._point_c(self._approach_anchor_pos, cube_w)
        center = np.asarray(base_pos, dtype=float).copy()
        center[:2] += np.dot(point_c[:2] - center[:2], contour_w) * contour_w
        center[:2] += max(0.0, clearance[1] - clearance[0]) * normal_w
        center[:2] += _SIDE_STANCE_CLEARANCE_BUFFER_M * normal_w
        candidates = []
        for offset in _CURATED_STANCE_OFFSETS_M:
            target = center.copy()
            target[:2] += offset * contour_w
            if self._approach_anchor_half_w is not None:
                lateral = float(np.dot(target[:2] - self._approach_anchor_pos[:2],
                                       self._approach_anchor_tangent))
                if abs(lateral) > self._approach_anchor_half_w:
                    continue                         # never side-step past the physical support edge
            distance = float(np.linalg.norm(target[:2] - base_pos[:2]))
            if distance <= _SIDE_STANCE_DONE_M:
                continue
            if self._stance_spent(target):
                continue                             # already occupied and failed for this cube
            candidates.append((self._stance_margin(field, target, base_quat), -distance, target))
        if not candidates:
            return None
        score, _neg_distance, target = max(candidates, key=lambda item: item[:2])
        if score <= baseline:
            return None
        print(f"[reach] visible-reachable side stance {baseline:.3f} -> {score:.3f}")
        return target

    def _probe_curated_stance_ik(self, base_pose, target_w, target_quat=None):
        """Run legacy fast IK at a predicted field-selected stance before commanding locomotion.

        Synchronous walking calls the existing ``ik_feasible_route`` directly. Dynamic MPC keeps its one
        cuRobo runtime: queue the scene-level predicate in that worker and return ``None`` until a later
        EXTEND tick polls it. A false result rejects only this offline-field candidate; normal legacy
        contour/back-off recovery remains available.

        ``target_quat`` defaults to the live base orientation (the lateral side-stance caller: same
        facing, new position). The turn-heading caller passes the CANDIDATE heading instead (same
        position, new facing) -- ``target_w`` there is just the current base position.
        """
        target_pose = (target_w, target_quat if target_quat is not None else base_pose[1])
        belief = self._reach_poses or {self._walk_current: self._latch_cube_pose}
        if self._session is not None:
            return ik_feasible_route(
                self._session, self._scenario, self._mj_model, _yaw_only(target_pose),
                cube_poses=belief, fixed_assignment=self._forced_assignment, height_pad=self._height_pad)
        assert self._worker is not None, "dynamic recovery requires its warmed cuRobo worker"
        scene_dict, targets = reach_scene_and_targets(
            self._cfg, self._scenario, self._mj_model, target_pose, cube_poses=belief,
            height_pad=self._height_pad)
        self._worker.submit_fast_ik(
            scene_dict, targets, target_pose[0], target_pose[1], fixed_assignment=self._forced_assignment)
        self._curated_stance_probe_w = target_w
        return None

    def _extend_infeasible_retry(self, base_pose):
        """Recover from a stance PROVEN infeasible in EXTEND -- either by the fast-IK
        gate (before any route attempt) or by no-route tick/verdict exhaustion (after plan-0 attempts all
        failed). First try configured camera-visible arm-workspace headings through the normal rate-limited
        locomotion turn. For a bounded support, then align with Point C along its clearance contour and sample
        its opposite local offset; both preserve clearance while changing arm workspace. Only after those candidates
        fail does the existing radial back-off run. Retries exhausted
        (or cannot drive): drop the cube (verdict sees FAIL(unreachable)) and re-perceive the rest via
        ACQUIRE -- the one shared entry -- instead of hanging to the step cap."""
        self._extend_no_route_ticks = self._plan0_infeasible = 0
        if self._approach_anchor_pos is None and self._latch_cube_pose is not None:
            # ACQUIRE can commit an outer-bound-visible target directly to EXTEND. Resolve its support here so
            # a no-route recovery still has the same per-target clearance geometry as an APPROACH commit.
            (self._approach_anchor_name, self._approach_anchor_pos, self._approach_anchor_tangent,
             self._approach_anchor_half_w, self._approach_support_normal_w) = self._nearest_anchor(
                 base_pose[0], self._latch_cube_pose[0])
        stance_dirs = self._cfg.walk_stance_dirs
        # A held cube pins this target to the other free arm. Prefer that target's mirrored workspace
        # heading before trying the opposite one; both remain fallbacks if the planner rejects it.
        if self._forced_assignment:
            target_side = next((side for side, cube in self._forced_assignment.items()
                                if cube == self._walk_current), None)
            if target_side is not None:
                desired_y_sign = 1.0 if target_side == "L" else -1.0
                stance_dirs = tuple(sorted(
                    stance_dirs, key=lambda direction: 0 if direction[1] * desired_y_sign >= 0.0 else 1))
        side_first = False
        if (self._latch_cube_pose is not None and self._cfg.walk_side_stance_adjust
                and self._approach_anchor_tangent is not None
                and self._approach_support_normal_w is not None):
            tangent_w = np.asarray(self._approach_anchor_tangent, dtype=float)
            normal_w = np.asarray(self._approach_support_normal_w, dtype=float)
            contour_w = tangent_w - np.dot(tangent_w, normal_w) * normal_w
            contour_norm = float(np.linalg.norm(contour_w))
            if contour_norm >= 1e-6:
                point_c = self._point_c(self._approach_anchor_pos, self._latch_cube_pose[0])
                side_first = True
        if (self._latch_cube_pose is not None
                and (not side_first or self._side_stance_retries > 0)
                and self._stance_turn_retries < len(stance_dirs)):
            bearing_w = self._latch_cube_pose[0][:2] - np.asarray(base_pose[0], dtype=float)[:2]
            world_bearing = float(np.arctan2(bearing_w[1], bearing_w[0]))
            target_dir = np.asarray(stance_dirs[self._stance_turn_retries], dtype=float)
            target_dir /= np.linalg.norm(target_dir)
            stance_yaw = world_bearing - float(np.arctan2(target_dir[1], target_dir[0]))
            turn_target_w = np.asarray(base_pose[0], dtype=float).copy()
            turn_target_w[:2] += (np.cos(stance_yaw), np.sin(stance_yaw))
            self._stance_turn_retries += 1
            # Fast-IK gate BEFORE committing to the physical turn (parity with the lateral side-stance
            # probe below): this heading only changes FACING, not position, so probe at the current base
            # position with the candidate yaw. Skips a doomed heading instead of paying a full turn to
            # discover it's infeasible on the next EXTEND attempt.
            fast_ik = self._probe_curated_stance_ik(
                base_pose, np.asarray(base_pose[0], dtype=float), target_quat=yaw_quat(stance_yaw))
            if fast_ik is None:
                # _probe_curated_stance_ik's async branch unconditionally sets _curated_stance_probe_w
                # (the LATERAL poll's flag) -- this call site tracks pending-ness via _stance_turn_probe_w
                # instead, so clear the other flag immediately. Left stale, it later makes _handle_extend's
                # lateral poll branch spin forever on a request that was never actually submitted under it.
                self._curated_stance_probe_w = None
                self._stance_turn_probe_w = turn_target_w
                return (0.0, 0.0, 0.0), {}, {}, None
            if not fast_ik:
                return self._extend_infeasible_retry(base_pose)   # rejected -- try next heading, same tick
            self._latch_dist = None
            self._stance_turn_target_w = turn_target_w
            self._mover.retarget()
            return (0.0, 0.0, 0.0), {}, {}, _APPROACH
        if (self._can_drive and self._cfg.walk_side_stance_adjust
                and self._approach_anchor_tangent is not None
                and self._approach_support_normal_w is not None
                and self._side_stance_retries < len(_SIDE_STANCE_RETRY_STEPS_M)):
            tangent_w = np.asarray(self._approach_anchor_tangent, dtype=float)
            normal_w = np.asarray(self._approach_support_normal_w, dtype=float)
            clearance = self._support_clearance(base_pose)
            # Anchor normals can point from a support center to the current viewpoint, not exactly normal to
            # its named width axis. Project that axis onto the constant-clearance contour before side stepping.
            contour_w = tangent_w - np.dot(tangent_w, normal_w) * normal_w
            contour_norm = float(np.linalg.norm(contour_w))
            # Never assume a hand marker or degenerate support geometry is safe. If braking coasted the root a
            # few cm inside the footprint limit, restore only that missing normal clearance while shifting sideways.
            if contour_norm < 1e-6 or clearance is None:
                self._side_stance_retries = len(_SIDE_STANCE_RETRY_STEPS_M)
                return self._extend_infeasible_retry(base_pose)
            contour_w /= contour_norm
            cube_w = self._latch_cube_pose[0]
            target_w = self._curated_side_stance(base_pose, contour_w, normal_w, clearance)
            curated = target_w is not None
            if curated:
                fast_ik = self._probe_curated_stance_ik(base_pose, target_w)
                if fast_ik is None:
                    return (0.0, 0.0, 0.0), {}, {}, None
                if not fast_ik:
                    self._curated_stance_ik_rejected = True
                    curated = False
            if not curated and self._side_stance_retries == 0:
                # Normal APPROACH can stop at the support-clearance line before it reaches Point C's
                # tangent coordinate. Align with that target along the safe contour before retrying IK.
                target_w = np.asarray(base_pose[0], dtype=float).copy()
                point_c = self._point_c(self._approach_anchor_pos, cube_w)
                target_w[:2] += np.dot(point_c[:2] - target_w[:2], contour_w) * contour_w
            elif not curated:
                target_w = np.asarray(base_pose[0], dtype=float).copy()
                toward_cube = 1.0 if np.dot(np.asarray(cube_w, dtype=float)[:2]
                                            - target_w[:2], contour_w) >= 0.0 else -1.0
                target_w[:2] += toward_cube * _SIDE_STANCE_RETRY_STEPS_M[self._side_stance_retries] * contour_w
            if not curated:
                target_w[:2] += max(0.0, clearance[1] - clearance[0]) * normal_w
                target_w[:2] += _SIDE_STANCE_CLEARANCE_BUFFER_M * normal_w
            self._side_stance_retries += 1
            self._side_stance_target_w = target_w
            self._note_stance_spent(target_w)
            self._side_stance_elapsed = 0.0
            self._latch_dist = None
            return (0.0, 0.0, 0.0), {}, {}, _APPROACH
        # Per-robot override (RobotDescriptor.extend_max_retries): None -> shared _EXTEND_MAX_RETRIES default.
        max_retries = self._cfg.extend_max_retries if self._cfg.extend_max_retries is not None else _EXTEND_MAX_RETRIES
        if self._can_drive and self._extend_retries < max_retries:
            self._extend_retries += 1
            self._extra_standoff += _RETRY_STANDOFF_STEP
            self._latch_dist = None
            self._square_ticks = 0
            self._mover.retarget()          # STOP is a one-way latch (HeuristicMovingPolicy.stop()); without
            #   this the base never drives again on the retry -- frozen wherever the failed commit stopped it
            return (0.0, 0.0, 0.0), {}, {}, _APPROACH
        if os.environ.get("REACH_DEBUG_STANCE"):
            cube_w = self._latch_cube_pose[0] if self._latch_cube_pose is not None else None
            dist = float(np.linalg.norm(np.asarray(base_pose[0], dtype=float)[:2] - np.asarray(cube_w, dtype=float)[:2])) if cube_w is not None else None
            print(f"[DEBUG_STANCE] cube={self._walk_current!r} base_xy={tuple(round(v,3) for v in base_pose[0][:2])} "
                  f"cube_xy={tuple(round(v,3) for v in cube_w[:2]) if cube_w is not None else None} "
                  f"dist={dist:.3f}m extra_standoff={self._extra_standoff:.3f} "
                  f"anchor={self._approach_anchor_name!r} side_stance_retries={self._side_stance_retries}",
                  file=sys.stderr)
        print(f"[reach] WARNING: cube {self._walk_current!r} unreachable after "
              f"{max_retries} back-off retry; skipping.", file=sys.stderr)
        self._failed[self._walk_current] = f"unreachable after {max_retries} back-off retries"
        self._pending.discard(self._walk_current)
        self._forced_assignment = None
        self._reach_poses = {}
        return (0.0, 0.0, 0.0), {}, {}, _ACQUIRE

    def _handle_extend(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """EXTEND: plans immediately from the live base pose. A cheap COLLISION-BLIND fast-IK gate
        (``ik_feasible_route``, IK-only, no trajopt) proves the stance is kinematically reachable at all -- a FAIL is a hard proof of
        infeasibility (IK is a superset of trajopt's admissible set), so it short-circuits straight to
        ``_extend_infeasible_retry()`` instead of paying for the full IK+trajopt plan below. Runs ONCE per
        commit (cached in ``_extend_ik_ok``, re-armed by ``_commit_reach``). Then falls through to the real
        reach dispatch (cuRobo route / async worker / Jacobian tracker) until the arm lands on the grasp -- the
        route's OWN trajectory clock finishes plus a settle (``_extend_settled``), NOT a tick budget (so
        kinematic and dynamics land identically). Then latch the grasp: close the jaws, snapshot the exact
        final arm command, and advance to GRASP. Returns ``(twist, arm_ref, cam, next)``."""
        belief = self._reach_belief(observed_cube_pose)
        if self._stance_turn_probe_w is not None:
            ready, feasible = self._worker.poll_fast_ik()
            if not ready:
                return (0.0, 0.0, 0.0), {}, {}, None
            turn_target_w, self._stance_turn_probe_w = self._stance_turn_probe_w, None
            if feasible:
                self._latch_dist = None
                self._stance_turn_target_w = turn_target_w
                self._mover.retarget()
                return (0.0, 0.0, 0.0), {}, {}, _APPROACH
            return self._extend_infeasible_retry(base_pose)
        if self._curated_stance_probe_w is not None:
            ready, feasible = self._worker.poll_fast_ik()
            if not ready:
                return (0.0, 0.0, 0.0), {}, {}, None
            target_w, self._curated_stance_probe_w = self._curated_stance_probe_w, None
            if feasible:
                self._side_stance_retries += 1
                self._side_stance_target_w = target_w
                self._note_stance_spent(target_w)
                self._side_stance_elapsed = 0.0
                self._latch_dist = None
                return (0.0, 0.0, 0.0), {}, {}, _APPROACH
            self._curated_stance_ik_rejected = True
            return self._extend_infeasible_retry(base_pose)
        if self._extend_ik_ok is None:
            if self._session is None:
                # Async/MPC-track hold NO in-process cuRobo session (worker/tracker owns the sole GPU runtime,
                # see __init__ docstring) -- the gate only applies to the sync in-process walk path, same
                # ``self._session is not None`` guard used by ``_submit_retract``/``_reach_step``.
                self._extend_ik_ok = True
            else:
                self._extend_ik_ok = ik_feasible_route(
                    self._session, self._scenario, self._mj_model, _yaw_only(base_pose),
                    cube_poses=belief, fixed_assignment=self._forced_assignment, height_pad=self._height_pad)
                if not self._extend_ik_ok:
                    return self._extend_infeasible_retry(base_pose)
        _, arm_ref, cam = self._reach_step(base_pose, proprio_qpos, belief, actual_tool_poses_base)
        if (self._walk and not self._mpc_track and not self._async
                and self._route is None and self._warned_no_route):
            # A synchronous plan_reach_route() returning None is a complete cuRobo verdict, not a pending
            # worker result. Retrying it for _EXTEND_HARD_TICKS only re-solves the same infeasible stance;
            # immediately use the existing side-stance/back-off recovery instead.
            return self._extend_infeasible_retry(base_pose)
        # Walk-path feasibility bound: a committed reach whose plan-0 never lands (route stays None) would hold
        # here forever (the parked path has a timeout, the walk path did not). Declare the stance infeasible on
        # ``_EXTEND_MAX_INFEASIBLE`` actual cuRobo "no route" verdicts (load-independent), or a hard no-route
        # tick backstop (worker wedge). Then re-enter APPROACH to DRIVE to a reachable stance (closer for a
        # still-far cube, back off for a too-close one), up to ``_EXTEND_MAX_RETRIES``; then drop the cube and
        # move on (FAIL(unreachable), re-perceive the rest via ACQUIRE). Never freeze.
        if self._walk and self._route is None:
            self._extend_no_route_ticks += 1
            if (self._plan0_infeasible >= _EXTEND_MAX_INFEASIBLE
                    or self._extend_no_route_ticks >= _EXTEND_HARD_TICKS):
                return self._extend_infeasible_retry(base_pose)
        elif self._route is not None:
            self._extend_no_route_ticks = 0
        # TARGET-MOVED replan: a cube this visit committed to has been pushed off the pose the route was solved
        # against. Distinct from the divergence guard below, which watches the TOOL against its own route and
        # is therefore blind to a target that moves -- here the tool may track perfectly and still drive at a
        # stale goal while its own finger keeps pushing the cube. Checked over EVERY cube in
        # ``_route.assignment``, not only ``_walk_current``: a bimanual visit commits two cubes to two arms while
        # ``_walk_current`` names one, so keying on it left the CO-TARGET unwatched, which is the measured
        # residual knock (v2_fixed/bimanual_mixed_front_back_close seed 46 pushed cube_1 off the palm with
        # ``robot/R_left_rack_collision*``; see ``media/target_moved_remeasure_20260730/FAILURE_CASES.md``).
        # Needs a LIVE observation to disagree with the snapshot, so a cube the camera has lost cannot trigger.
        # CAPPED TRACKERS ONLY, reusing ``_extend_diverge_guard``: with an ungated version g1 lost a trial to an
        # upright violation at 5x energy, because g1 configures no correction tube and absorbs a mid-route
        # replan by re-solving its whole long tilted approach, destabilizing the gait.
        if (self._extend_diverge_guard and self._route is not None
                and not self._extend_settled() and not self._final_replan_pending
                and self._target_moved_replans < _TARGET_MOVED_REPLAN_MAX):
            moved_target = None
            for cube in self._route.assignment.values():
                pinned = self._reach_poses.get(cube)
                if pinned is None or observed_cube_pose.get(cube) is None or cube not in belief:
                    continue
                moved = float(np.linalg.norm(np.asarray(belief[cube][0][:2], dtype=float)
                                             - np.asarray(pinned[0][:2], dtype=float)))
                if moved > _TARGET_MOVED_REPLAN_M:
                    moved_target = cube
                    break
            if moved_target is not None:
                scene_dict, targets = reach_scene_and_targets(
                    self._cfg, self._scenario, self._mj_model, base_pose, cube_poses=belief)
                if targets:
                    self._plan0_gen = self._worker.submit_plan0(
                        scene_dict, targets, base_pose[0], base_pose[1],
                        fixed_assignment=dict(self._route.assignment),
                        seed_q=self._seed_from_proprio(proprio_qpos), goal=self._extend_goal())
                    self._final_replan_pending = True
                    self._final_replan_scene = scene_dict
                    self._target_moved_replans += 1
                    # Re-pin to the pose just planned against, so the bound counts DISTINCT displacements
                    # instead of re-firing every tick on the same one.
                    self._reach_poses[moved_target] = self._pin_pose(belief[moved_target])
                    if moved_target == self._walk_current:
                        self._latch_cube_pose = self._reach_poses[moved_target]
                    print(f"[reach] target moved {moved:.3f}m: replan for {moved_target}")
                    return (0.0, 0.0, 0.0), arm_ref, cam, None
        # MID-ROUTE divergence abort. The convergence check below only runs once the route's wall clock has
        # finished, which on a diverging reach is far too late: instrumented on v2/front_back_close/seed50, the
        # left tool error crosses this threshold at t~4.0 s and the gripper finger does not touch the cube until
        # t~7.3 s, so the cube is already off its support before the settle-time check is ever reached. The rack
        # -vs-cube clearance is only ~0.015 m, so a residual this size means the finger strikes the cube instead
        # of straddling it. Re-planning from the LIVE measured configuration resets the accumulated tracking lag,
        # which is exactly what the observed self-rescues did (seeds 42/45/51 all completed via a re-reach).
        # Debounced over consecutive ticks so a single transient spike cannot trigger a replan, and confined to
        # the travelling part of the path so it never fires while the hand is already at the cube.
        # CAPPED TRACKERS ONLY (``_extend_diverge_guard``): the mechanism this rescues is a tracker whose
        # correction is clipped to the nominal-route tube, so a residual it cannot work off keeps growing. G1
        # configures no tube (``dynamic_tracker_transit_max_correction_rad is None``) and its long tilted
        # approach legitimately holds >0.05 m mid-route, so the guard fired spuriously there and cost the
        # second latch: measured on the 10-seed far cells, front_back_far 0.70/0.80 -> 0.90 and
        # left_right_far 0.70/0.80 -> 0.90 with it disabled, while the capped embodiments kept 1.00.
        if (self._extend_diverge_guard
                and self._mpc_track and self._route is not None and actual_tool_poses_base is not None
                and not self._extend_settled() and not self._final_replan_pending
                and self._tracker.route_progress() < _EXTEND_DIVERGE_MAX_PROGRESS
                and self._extend_diverge_replans < _EXTEND_DIVERGE_REPLAN_MAX):
            diverge_err = self._tracker.final_position_error(actual_tool_poses_base)
            if diverge_err > _EXTEND_DIVERGE_ERROR_M:
                self._extend_diverge_ticks += 1
            else:
                self._extend_diverge_ticks = 0
            if self._extend_diverge_ticks >= _EXTEND_DIVERGE_TICKS:
                scene_dict, targets = reach_scene_and_targets(
                    self._cfg, self._scenario, self._mj_model, base_pose, cube_poses=belief)
                if targets:
                    # Progress BEFORE the submit: ``_install_reach`` -> ``rebind`` zeroes ``_elapsed``, so each
                    # rescue re-traverses the WHOLE path from 0 rather than resuming. Logged like the
                    # TARGET-MOVED line because an unlogged rescue is invisible in T-bar/energy attribution.
                    progress = self._tracker.route_progress()
                    self._plan0_gen = self._worker.submit_plan0(
                        scene_dict, targets, base_pose[0], base_pose[1],
                        fixed_assignment=dict(self._route.assignment),
                        seed_q=self._seed_from_proprio(proprio_qpos), goal=self._extend_goal())
                    self._final_replan_pending = True
                    self._final_replan_scene = scene_dict
                    self._extend_diverge_replans += 1
                    self._extend_diverge_ticks = 0
                    print(f"[reach] extend diverged {diverge_err:.3f}m at progress {progress:.2f}: "
                          f"replan {self._extend_diverge_replans}/{_EXTEND_DIVERGE_REPLAN_MAX}")
                    return (0.0, 0.0, 0.0), arm_ref, cam, None
        if self._route is not None and self._extend_settled():
            if (self._mpc_track and actual_tool_poses_base is not None
                    and self._tracker.final_position_error(actual_tool_poses_base) > _DYNAMIC_FINAL_TRACK_ERROR_M
                    and self._final_replans < _DYNAMIC_FINAL_REPLAN_MAX
                    and not self._final_replan_pending):
                scene_dict, targets = reach_scene_and_targets(
                    self._cfg, self._scenario, self._mj_model, base_pose, cube_poses=belief)
                if targets:
                    self._plan0_gen = self._worker.submit_plan0(
                        scene_dict, targets, base_pose[0], base_pose[1],
                        fixed_assignment=dict(self._route.assignment),
                        seed_q=self._seed_from_proprio(proprio_qpos), goal=self._extend_goal())
                    self._final_replan_pending = True
                    self._final_replan_scene = scene_dict
                    self._final_replans += 1
                    return (0.0, 0.0, 0.0), arm_ref, cam, None
            if _TWO_STAGE_EXTEND and not self._pregrasp_done:
                # Stage A landed at the pre-grasp standoff. Submit stage B -- the real grasp, re-planned
                # against a FRESH base pose and cube belief, so the 5 cm that actually approaches the cube
                # carries none of stage A's accumulated base drift. Latching here would close the jaws a
                # full standoff short of the cube, so this branch returns UNCONDITIONALLY while the flag is
                # clear: an empty ``targets`` (every committed cube momentarily unobserved) holds and
                # retries next tick rather than falling through to GRASP.
                scene_dict, targets = reach_scene_and_targets(
                    self._cfg, self._scenario, self._mj_model, base_pose, cube_poses=belief)
                if targets and not self._final_replan_pending:
                    self._pregrasp_done = True          # BEFORE the submit: ``_extend_goal`` reads it
                    self._plan0_gen = self._worker.submit_plan0(
                        scene_dict, targets, base_pose[0], base_pose[1],
                        fixed_assignment=dict(self._route.assignment),
                        seed_q=self._seed_from_proprio(proprio_qpos), goal=self._extend_goal())
                    self._final_replan_pending = True
                    self._final_replan_scene = scene_dict
                    print("[reach] pregrasp reached: planning descent")
                return (0.0, 0.0, 0.0), arm_ref, cam, None
            # A static carry lean is valid; wait only for gravity to stop changing across the fixed IMU
            # window.  Bound the wait so the reached waypoint cannot visibly idle before jaw closure.
            self._grasp_stable_wait += self._control_dt
            if (not self._grasp_base_stable(base_pose[1])
                    and self._grasp_stable_wait < _GRASP_STABLE_TIMEOUT_S):
                # NOT station-kept: this gate waits for base ORIENTATION to stop changing before the jaws
                # close, and a commanded twist is the one thing guaranteed to keep it changing. The route is
                # already finished here, so there is no base-relative path left for drift to drag.
                return (0.0, 0.0, 0.0), arm_ref, cam, None
            self._grasp_gravity_ref = None
            self._grasp_gravity_elapsed = self._grasp_stable_wait = 0.0
            self._grasp_gravity_max_delta = 0.0
            # Close ONLY this target's grasping arm(s); arms already holding a cube (``_held``) stay closed.
            self._gripper_closed = set(self._held) | set(self._route.assignment)
            self._grasp_close_elapsed = 0.0
            self._grasp_route = self._route
            self._held_grasp_command = self._hold_grasp_command()
            return (0.0, 0.0, 0.0), dict(self._held_grasp_command), cam, _GRASP
        return (0.0, 0.0, 0.0), arm_ref, cam, None

    def _handle_grasp(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """GRASP: keep the final object-relative arm target tracked while the position-servo jaws close.
        After its shared-geometry-derived minimum stroke, a completed physical latch advances immediately; a 1.0 s cap turns a
        missing latch into a retry. The tracker freezes only after closure before return planning. When the close stroke is
        done, record only physical-latch-confirmed arm(s) as OCCUPIED (bug 4: ``_held``) and submit the async
        grasp->home plan, advancing to whichever state ``_submit_retract`` returns (RETRACT_PLAN, or RETRACT
        for the legacy in-process reverse). Returns ``(twist, hold, {}, next)``."""
        if self._mpc_track:
            gravity = self._gravity_in_base(base_pose[1]) if self._gravity_comp else None
            hold = self._tracker.final_step(
                self._observe_objects_in_base(base_pose, self._reach_belief(observed_cube_pose)),
                gravity_in_base=gravity, actual_tool_poses_base=actual_tool_poses_base,
                measured_arm_qpos=proprio_qpos if isinstance(proprio_qpos, dict) else None)
        else:
            hold = self._hold_grasp_command(proprio_qpos, actual_tool_poses_base, base_pose[1])
        self._grasp_close_elapsed += self._control_dt
        planned = dict(self._grasp_route.assignment)
        captured_now = planned if self._physical_grasps is None else {
            side: cube for side, cube in planned.items()
            if self._physical_grasps.get(side) == cube}
        all_latched = set(captured_now.values()) == set(planned.values())
        # ``physical_grasps`` was sampled before this tick's physics step. Keep the jaw command through the
        # cap tick itself, so a rack contact created by its final 0.6 mm close increment is observable on
        # the following tick instead of immediately reopening the hand. This is one 20 ms post-step sample,
        # not an extra settle pause.
        if (self._grasp_close_elapsed < _GRIPPER_CLOSE_MIN_S
                or (not all_latched and self._grasp_close_elapsed <= _GRIPPER_CLOSE_MAX_S)):
            return (0.0, 0.0, 0.0), hold, {}, None
        if self._mpc_track:
            self._tracker.freeze_hold()
            self._held_grasp_command = dict(hold)
        captured = captured_now
        missing = set(planned.values()) - set(captured.values())
        self._held.update(captured)                                 # retire only a completed physical capture
        self._gripper_closed = set(self._held)
        for cube in missing:
            if cube in self._fallen:
                continue        # already condemned by note_fallen; no retry can latch a cube on the floor
            retries = self._grasp_capture_retries.get(cube, 0) + 1
            self._grasp_capture_retries[cube] = retries
            if retries > _GRASP_CAPTURE_MAX_RETRIES:
                self._failed[cube] = "physical grasp latch did not engage"
                self._pending.discard(cube)
        # Every successful grasp returns along the same collision-aware home route before terminal state.
        # Latch grading records capture independently, so the return cannot erase a completed physical grasp.
        belief = self._reach_belief(observed_cube_pose)
        return (0.0, 0.0, 0.0), hold, {}, self._submit_retract(base_pose, proprio_qpos, belief)

    def _handle_retract_plan(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """RETRACT_PLAN: hold the grasp command while polling the async grasp->home plan; once it lands,
        install it and advance to RETRACT. Returns ``(twist, hold, {}, next)``."""
        ready, retract_route = self._poll_retract()
        hold = self._hold_grasp_command(proprio_qpos, actual_tool_poses_base, base_pose[1])
        self._retract_plan_ticks += 1
        if not ready:
            if self._retract_plan_ticks >= _RETRACT_PLAN_TIMEOUT_TICKS:
                # Worker wedged/dead (observed g1): stop waiting and return along the reversed extend corridor
                # (collision-free by construction). Never strand the mission holding the grasp.
                print(f"[reach] WARNING: grasp->home plan did not return in "
                      f"{_RETRACT_PLAN_TIMEOUT_TICKS} ticks; reversing the extend corridor.", file=sys.stderr)
                self._start_retract(None)
                return (0.0, 0.0, 0.0), hold, {}, _RETRACT
            return (0.0, 0.0, 0.0), hold, {}, None
        if (retract_route is not None
                and self._tracker is not None
                and set(retract_route.controlled_joints) != set(self._tracker.controlled)):
            print("[reach] WARNING: return active-arm set changed; reversing extend corridor.", file=sys.stderr)
            retract_route = None
        self._start_retract(retract_route, proprio_qpos)
        return (0.0, 0.0, 0.0), hold, {}, _RETRACT

    def _handle_retract(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        """RETRACT: play the grasp->home return (MPC keeps Cartesian tracking active; other backends execute
        their return reference buffer), jaws stay closed -- a later place controller calls ``release_grasp``.
        On settle: parked TERMINATEs (``parked_done``); walk marks the visit's cubes done and re-enters ACQUIRE
        to re-perceive+commit the remaining pending cubes (the one shared entry). Returns
        ``(twist, arm_ref, {}, next)``."""
        if self._retract_uses_tracker:
            gravity = self._gravity_in_base(base_pose[1]) if self._gravity_comp else None
            arm_ref = self._tracker.return_step(
                gravity_in_base=gravity, actual_tool_poses_base=actual_tool_poses_base,
                measured_arm_qpos=proprio_qpos if isinstance(proprio_qpos, dict) else None)
            retract_settled = self._tracker.return_settled(_REACH_SETTLE_S)
        else:
            arm_cmd = self._executor.command(self._template, self._controlled)[0].numpy()
            arm_ref = dict(zip(self._controlled, arm_cmd.tolist()))
            retract_settled = self._executor.cursor_s >= self._executor.duration_s + _REACH_SETTLE_S
        if not retract_settled:
            return (0.0, 0.0, 0.0), arm_ref, {}, None
        # PARK where cuRobo left the arm. The return targets the home TOOL pose, but the chain is redundant,
        # so the configuration it lands in can be far from ``planning_home_joint_pos`` in joint space -- and
        # an empty ``arm_ref`` makes the harness servo there. That used to be smoothed by a lerp to
        # planning-home, which is a whole SECOND arm motion, costing time and energy to reach a pose that is
        # kinematically no better than the one already held. Parking instead removes the motion rather than
        # smoothing it, and ``_arm_hold_q`` keeps the next plan-0 seeded where the arm actually is.
        self._park_arm_at(arm_ref)
        self._retract_uses_tracker = False
        self._route = None
        self._held_grasp_command = None
        if not self._walk:
            self._parked_done = True
            self._viz_route = None
            return (0.0, 0.0, 0.0), {}, {}, _TERMINATE
        # Retire ONLY the cubes cuRobo actually ROUTED this visit (``_grasp_route.assignment``), not every cube
        # the proximity gate committed (``_reach_poses``). The gate is planar-distance only, so it commits a
        # cube that is near but UNREACHABLE from this stance (front_back_close: both cubes fall inside the
        # radius, yet a forward-arm robot can route only the one it faces). Discarding all of ``_reach_poses``
        # then silently dropped the un-routed cube and terminated the mission one cube short; discarding the
        # routed set leaves the rest PENDING for their own visit (the base turns to face them next).
        routed = ({cube for side, cube in self._grasp_route.assignment.items()
                   if self._held.get(side) == cube}
                  if self._grasp_route is not None else set(self._reach_poses))
        for c in routed:
            self._pending.discard(c)
            if c in self._reach_anchor_names:
                self._anchors_served.add(self._reach_anchor_names[c])
        # A completed visit MOVED the base and REMOVED a cube, so every prior "nothing framable from here"
        # conclusion is stale -- give DISCOVER its anchors back. Bounded: each clear costs a grasp and cubes
        # are finite, so this cannot livelock the search.
        if routed:
            self._anchors_visited.clear()
        self._forced_assignment = None                              # next visit re-pins its own winner(s)
        self._reach_poses = {}
        self._reach_anchor_names = {}
        self._viz_route = None                                      # plan done: drop the reach overlay
        return (0.0, 0.0, 0.0), {}, {}, _ACQUIRE     # visit done -> re-perceive+commit remaining via ACQUIRE

    def _reach_step(self, base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base=None):
        if self._mpc_track:
            return self._step_mpc(base_pose, proprio_qpos, observed_cube_pose, actual_tool_poses_base)
        if self._async:
            return self._step_async(base_pose, proprio_qpos, observed_cube_pose)
        if self._replan_due():
            replanning = self._route is not None
            seed_q = self._seed_from_proprio(proprio_qpos) if replanning else None
            route = plan_reach_route(
                self._session, self._scenario, self._mj_model, _yaw_only(base_pose),
                cube_poses=observed_cube_pose, seed_q=seed_q,
                max_attempts=_REPLAN_MAX_ATTEMPTS if replanning else None,
                # First plan: pin the gate-winning arm (``_forced_assignment``, set by a walk visit's APPROACH).
                # cuRobo's own ladder assigns a lone FAR cube to the geometrically-nearer arm, which on a v2
                # (two arms) diverges from the reachability winner and returns NO IK -> route never installs ->
                # ``_replan_due`` stays True -> a cuRobo solve EVERY tick (unreasonably slow) and the arm never
                # moves. Mirrors the MPC path (``_step_mpc``). None off the walk path -> unchanged parked reach.
                fixed_assignment=self._route.assignment if replanning else self._forced_assignment,
                height_pad=self._height_pad)
            if route is None:
                if self._route is not None:
                    self.replan_failures += 1
                elif not self._warned_no_route:
                    # FIRST plan for a committed cube returned no route: the gate latched (says reachable) but
                    # cuRobo found no feasible arm motion from this stance. Silently retrying re-solves cuRobo
                    # EVERY tick (the GPU storm = the "unreasonably slow" play) and the arm never moves. Surface
                    # it once (per the plan's "committed cube with no route = FAIL, not silent skip").
                    print(f"[reach] WARNING: no cuRobo route for committed cube {self._walk_current!r} "
                          f"(assignment={self._forced_assignment}) -- gate latched but planner infeasible "
                          f"from this stance; arm will not reach.", file=sys.stderr)
                    self._warned_no_route = True
            else:
                self._install(route)
                self._warned_no_route = False
        self._k += 1
        if self._route is None:
            return (0.0, 0.0, 0.0), {}, {}
        arm_cmd = self._executor.command(self._template, self._controlled)[0].numpy()
        arm_reference = dict(zip(self._controlled, arm_cmd.tolist()))
        return (0.0, 0.0, 0.0), arm_reference, {}
