"""Unified cuRobo reach/grasp verify for BOTH the humanoid_v21 and the Unitree G1 (+parallel gripper),
driving ONE transferable ``ReachPolicy`` through EITHER harness:
  * default (kinematic): ``KinematicHarness`` = ``mj_forward`` on written qpos -- an idealized robot
    proving geometric reachability (the former phase-3 verify).
  * ``--dynamic``: ``DynamicHarness`` = a frozen RL low-level policy + MuJoCo/warp physics realizing the
    SAME plan (the former ``curobo_phase4_verify.py``). A PASS here means the low-level policy tracked a
    kinematically-valid plan THROUGH real dynamics.
Same policy, same scene, same 5 cm tolerance. Kinematic mode grades realized tool error to its planned
route candidate. Dynamic mode grades physical capture: each assigned cube must complete a real contact latch
and end within tolerance of its live grasp center. Only the ``realize`` seam (physics vs ``mj_forward``)
differs. Merged the former per-robot phase-3 scripts and the phase-4 dynamic wrapper into one robot-generic
entrypoint.

``--drift`` (kinematic-only) injects a zero-mean base-drift sinusoid (a phase-4-analog disturbance) into
the reach loop and reports the plan-once open-loop MISS against a live-base replan sweep -- redundant under
real physics, so it is rejected with ``--dynamic``. ``--walk`` drives the SAME online ``ReachPolicy(walk=True)``
through EITHER harness (kinematic ``mj_forward`` or ``--dynamic`` physics) -- one walk brain, per-visit
SEARCH->GO->REACH->PARK. Parked ``--dynamic --record`` delegates its offscreen MP4 to ``pickplace_reach_env``;
``--dynamic --walk --record DIR`` renders the graded walk rollout to an offscreen MP4 (chase cam) directly in
``_grade_walk_headless`` -- the SAME physics run that produces the verdict, no replay.

Design (two plain config dataclasses, cleanly separated):
  * ``ObjectSceneCfg`` -- ROBOT-INDEPENDENT single source of truth for the scene: scenario + jitter.
    Cube world poses come from ``(scenario, seed)`` ALONE, then are PAINTED into whichever robot's
    compiled model. Identical seed => identical cubes for either robot (the ``--robot both`` compare
    guarantee).
  * ``RobotDescriptor`` (from ``curobo.planner``) -- base pose, tool frames, per-side arm joints, planner
    kwargs, and the robot's cube-grasp callables.

Run (bimanual, default scenario ``bimanual_mixed_close``):
  /home/grl/repo/micromamba/envs/py312/bin/python \
      mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py --robot v2 [--view]
  ... --robot g1 left_right_close            # g1 stands back; left_right_close is g1-reachable
  ... --robot both left_right_close          # same scene, g1 + v2_fixed + v2 summary (kinematic only)
  PYTHONPATH=mj_envs ... --dynamic --robot v2 --scenario front_back_close --mpc   # physics + frozen RL
Object XY is jittered deterministically (inward-reflected, stays reachable) on every run from a fresh
OS-entropy seed (printed); pass ``--seed N`` to reproduce a layout.
"""

import dataclasses
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import tyro

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
# Per-tick tool-error trace through EXTEND. Off by default: it answers WHEN a diverging reach diverges (so a
# ``[handoff]`` error of 0.6 m can be told from a late blow-up), which is worth many lines only while diagnosing.
_KNOCK_TRACE = bool(os.environ.get("KNOCK_TRACE"))
# Base disturbance over the whole manipulation span. Off by default. ``ARM_MAX_ACCELERATION``/``ARM_MAX_JERK``
# exist to bound the reaction wrench the moving arm exerts on the FLOATING base, and neither PENETRATION nor
# energy_J measures that -- so any screen of those caps that reports only the shipped metrics is blind to the
# quantity they were tuned against. This reports it directly: peak base speed and worst base XY excursion
# from the pose the cuRobo route was solved at.
_BASE_PROBE = bool(os.environ.get("BASE_PROBE"))
_MANIP_PHASES = ("EXTEND", "GRASP", "RETRACT_PLAN", "RETRACT")
sys.path.insert(0, str(_REPO_ROOT / "mj_envs"))
os.chdir(_REPO_ROOT)          # asset/media paths in the reach spine are repo-root relative

from asset_zoo.fov_frustum import FOV_GEOM_GROUP  # noqa: E402 (single source for FOV geom group)
from tasks.visual_manipulation.pickplace_scenarios import (  # noqa: E402
    SCENARIO_NAMES,
    SCENARIOS,
    make_scene_spec_fn,
)

# Reach spine (config + planner); this script drives ONE transferable ReachPolicy through either harness.
from tasks.visual_manipulation.curobo.planner import (  # noqa: E402
    CFG_BY_ROBOT,
    RobotDescriptor,
)
from tasks.visual_manipulation.curobo.scene import (  # noqa: E402
    ObjectSceneCfg,
    move_pick_geoms as _move_pick_geoms,
    robot_scene as _robot_scene,
)

# ------------------------------------------------------------------------------------------------
# Robot-independent constants
# ------------------------------------------------------------------------------------------------
BIMANUAL_SCENARIO_NAME = "bimanual_mixed_close"    # humanoid bimanual default
_REACH_TOL_M = 0.05            # SHARED reach gate (kinematic + dynamic): idealized robot tracks to ~0 m;
                               # the frozen-policy dynamic steady-state (~0.020 m) also clears it. Was 0.02
                               # kinematic-only; loosened so ONE gate grades both harnesses (see _UPRIGHT_MIN).
_UPRIGHT_MIN = 0.9             # SHARED gate: base up-axis vs world up. Kinematic upright()==1.0 (never falls),
                               # so this only bites the dynamic harness (a toppled physics reach FAILs).
_REACH_STEPS = 300             # reach TIMEOUT cap: the dynamic reach converges (sustained settle) or times out
                               # here; kinematic full-runs to it. Also the drift settle window base (>= period).
_MPC_WARM_RETRIES = 2          # spawned MPC worker warmup attempts before giving up
_REACH_SETTLE_TOL_M = 0.02     # DYNAMIC converge-early SETTLE tol (tight, well BELOW the 0.05 gate): the reach
                               # stops once the trailing window is all < this. Distinct from the gate so a
                               # converged report keeps margin (converging ON the gate tol pins the report to
                               # the boundary). Below the frozen-policy steady-state (~0.020), so the dynamic
                               # reach usually just times out at _REACH_STEPS and reports its true steady peak
                               # -- correct + bounded; the early stop is only a best-effort speedup.
_REACH_DWELL = 60              # DYNAMIC sustained-settle window (ticks): converge only when the last _REACH_DWELL
                               # routed verdicts are ALL < _REACH_SETTLE_TOL_M. Must exceed the frozen-policy
                               # limit-cycle period (a lucky low-phase instant must not trip it). Kinematic does
                               # NOT converge-early (it tracks a ~150-tick trajectory; a window-max stop would
                               # pin the report near the gate) -- it full-runs and reads the settled ~0 m.
_WALK_STEP_CAP = 2200          # headless walk-mission safety cap (matches the mission's default length): a
                               # walk+reach+park-per-cube run ends early on walk_done well inside this.
_DYNAMIC_SETTLE_S = 2.0        # pre-plan home hold, long enough to enter the standing limit cycle


def _vrw_tick_ok(walk_phase: str, extend_settled: bool, observed, targets: tuple[str, ...]) -> bool:
    """Executed pairwise-VRW conjunction for one manipulation tick.

    Returns True when the tool is at/holding its target AND every cube in ``targets`` is camera-visible this
    tick. ``targets`` is the mission's full pick-cube set, not the current route's assignment -- dropping a
    target silently weakens "watch both" into "watch whichever one I'm doing" on any single-arm visit, which is
    a planning-behavior artifact rather than a sensing signal. Reach term is trivially true past EXTEND; on
    EXTEND it is exactly the FSM's own settle gate (``policy._extend_settled()``). Visibility is membership
    in ``harness.observe()``'s per-tick detection dict -- that dict already gates on
    ``_cube_visible_from_camera``/``_scene_occluder_geoms`` internally, so membership is ``vis(x)`` for free.
    Pure and side-effect-free so it is unit-testable without a live harness.
    """
    return (walk_phase != "EXTEND" or extend_settled) and all(t in observed for t in targets)


def _capture_result(harness, route) -> tuple[float, float]:
    """Return ``(success_score, diagnostic_error)`` for one completed grasp route.

    Dynamic success is completed physical contact capture. Its live center error stays diagnostic, while a
    completed latch scores zero. Kinematic mode has no latch, so route tracking remains both score and
    diagnostic. ``nan`` score means no completed dynamic latch.
    """
    capture = getattr(harness, "grasp_capture_error", None)
    if capture is not None:
        diagnostic = capture(route)
        return (0.0, diagnostic) if np.isfinite(diagnostic) else (float("nan"), diagnostic)
    diagnostic = harness.verdict(route)
    return diagnostic, diagnostic


INITIAL_RECORD_SCENARIOS = (
    "left_right_close", "left_right_far", "front_back_close", "front_back_far",
    "bimanual_mixed_close", "bimanual_mixed_front_back_close", "shelf",
)
_POST_GRADE_HOLD_TICKS = 100           # --view: control ticks the graded mission holds on screen before the
                                       # viewer auto-resets to the next seed. Long enough to read METRICS2
                                       # (printed one callback after grading) and see the final grasp pose;
                                       # short enough that an unattended --view run keeps sweeping seeds
                                       # instead of parking forever on one. ~2 s of sim at a 20 ms control_dt
                                       # (~1 s of wall time at the 2x default playback).


def _view_start_trial() -> int:
    """``--view`` only: index of the seed-sweep trial the viewer opens on (``REACH_VIEW_START_TRIAL``).

    Reproducing a sweep log's failing trial needs the sweep's OWN ``--seed``, because that value also seeds
    harness construction -- passing the failing trial's seed directly builds an env the sweep never built,
    and the failure does not recur. Pass the sweep seed plus this offset instead; trial N runs at
    ``seed + N``. Missions before N are skipped, so warm-process history differs from the sweep's."""
    return int(os.environ.get("REACH_VIEW_START_TRIAL", "0"))


def _resolved_seed(explicit_seed: int | None) -> int:
    return int(np.random.SeedSequence().entropy % (2**32)) if explicit_seed is None else explicit_seed


def _apply_seed(object_cfg: ObjectSceneCfg, explicit_seed: int | None):
    """Draw/resolve a seed, jitter the scene's pick XY, print the seed for replay. Robot-free."""
    seed = _resolved_seed(explicit_seed)
    scenario = object_cfg.jittered(seed)
    print(f"seed {seed}: jittered pick XY -> "
          f"{[(m.name, tuple(round(c, 3) for c in m.pos)) for m in scenario.pick]}")
    return scenario


# ------------------------------------------------------------------------------------------------
# Kinematic walk-in pre-phase (no MuJoCo dynamics)
# ------------------------------------------------------------------------------------------------
def _apply_home_pose(mj_model, d, robot_cfg: RobotDescriptor) -> None:
    """Write ``robot_cfg.home_joint_pos`` into ``d.qpos``, expanding regex joint-name PATTERNS
    (mjlab-keyframe style, e.g. ``.*_knee_joint``) via ``re.fullmatch`` against the compiled model --
    mirrors ``asset_zoo/g1/g1_constants._apply_keyframe``. Literal keys fullmatch themselves.

    Why regex: g1's home is ``{**KNEES_BENT_KEYFRAME.joint_pos, **G1_ARM_HOME}`` and the keyframe's leg
    keys are patterns (``.*_hip_pitch_joint``/``.*_knee_joint``/``.*_ankle_pitch_joint``). A literal
    ``mj_name2id`` lookup finds no joint named ``.*_knee_joint`` and silently DROPS every leg bend, so
    g1 stands with straight (zero) legs and its feet sink ~0.14 m below the floor. Expanding the
    patterns applies the true bent-knee home. v2/v2_fixed use literal keys (unaffected)."""
    joint_names = [mj_model.joint(j).name for j in range(mj_model.njnt)]
    for pattern, value in robot_cfg.home_joint_pos.items():
        for jname in joint_names:
            if re.fullmatch(pattern, jname):
                d.qpos[mj_model.joint(jname).qposadr[0]] = value
    for jname, value in robot_cfg.planning_home_joint_pos.items():
        d.qpos[mj_model.joint(jname).qposadr[0]] = value


# ------------------------------------------------------------------------------------------------
# Offscreen recording (headless mp4 of the walk-in + camera reach; no interactive viewer)
# ------------------------------------------------------------------------------------------------
RECORD_W, RECORD_H = 2560, 1440         # 2K (1440p): a 60 mm cube and the jaw gap must stay legible when a
                                        # keyframe is cropped into a figure panel, which 720p did not survive
RECORD_CAMERA_DISTANCE = 2.5            # close-task prop detail
RECORD_FAR_CAMERA_DISTANCE = 4.2         # encompasses both distant support tables in far configurations
RECORD_CAMERA_AZIMUTH_DEG = 135.0
RECORD_CAMERA_ELEVATION_DEG = -55.0     # top-down task-layout view

# Per-scenario initial camera pose for --view's live NativeMujocoViewer (distinct from the offscreen
# RECORD_* pose above). Captured via the 'o' key (_maybe_capture prints cam.lookat/distance/azimuth/
# elevation) and pasted back here so that scenario's --view opens pre-framed instead of at the mjlab
# default free-camera view. Scenarios absent from this table fall back to that default.
_VIEW_CAMERA_POSE: dict[str, tuple[tuple[float, float, float], float, float, float]] = {
    "left_right_far": ((0.023236377251640558, -0.005259805857486136, 0.545225059989139), 2.5084, 141.28, -35.61),
}


def _record_camera_pose(scenario_name: str) -> tuple[float, float]:
    """Use straight cardinal setup views; shelf needs a shallow side view to expose its tiers."""
    if scenario_name not in ("shelf", "shelf_pick_both"):
        return 0.0, -50.0
    return -45.0, -35.0


# Walk-rollout chase cam, deliberately DIFFERENT from the setup-PNG pose above. The PNG frames a static
# whole-scene layout; the video must show a 60 mm cube being grasped. At the old settings (4.2 m,
# elevation -50, aimed at the base) the cube was a few pixels and the near-top-down line of sight put the
# robot's own torso/arm between the lens and the table, so the grasp was not watchable. Pull in, drop to a
# 3/4 view that sees ACROSS the table surface, and aim between base and target (see `_walk_cam_lookat`).
RECORD_WALK_CAMERA_DISTANCE = 1.9       # chase cam re-frames every tick, so the far-scene 4.2 m is not needed
                                        # Override with WALK_CAM_DISTANCE_M; the keyframe figure pulls in to
                                        # ~1.35 m, where the 60 mm cube and the jaws read at panel size.
RECORD_WALK_CAMERA_ELEVATION_DEG = -22.0
RECORD_WALK_CAMERA_YAW_OFFSET_DEG = 145.0   # relative to base heading: front 3/4 (180 = dead ahead of robot)
RECORD_WALK_LOOKAT_Z = 0.68             # just above the table plate, where cube and gripper meet
                                        # Override with WALK_CAM_LOOKAT_Z. Pulling the camera in for the
                                        # keyframe figure crops the head off at this aim height, because the
                                        # frame is centred on the table plate; aiming higher recentres the
                                        # robot and spends the lost pixels on empty foreground floor.
RECORD_WALK_TARGET_WEIGHT = 0.65        # lookat bias toward the cube; 0.5 (midpoint) still cropped it small
RECORD_WALK_LERP = 0.04                 # per-tick smoothing; hard cuts on target/heading switch look broken

# Distance is PHASE-KEYED, because the two things a panel must show want opposite framings and one fixed
# distance can only buy one of them. While the robot is still acquiring/walking, the panel's job is the SCENE:
# both targets, both supports, the human. At a single 1.35 m that panel showed only ONE of the two benches in
# the ``_far`` layouts and clipped the handoff human to an arm -- losing exactly the second target that makes a
# two-target mission legible. Once the arm is out, the panel's job is the GRASP: a 60 mm cube and the jaw gap,
# which 2.4 m renders as a few pixels. Easing between the two costs nothing (the lookat filter already runs
# every tick) and gives each column the framing it actually needs.
RECORD_WALK_CAMERA_DISTANCE_WIDE = 2.40     # establishing phases: whole task layout
# Phases where the robot is still acquiring/walking, i.e. where a frame's job is to ESTABLISH the scene rather
# than to show a grasp.
RECORD_WALK_ESTABLISHING_PHASES = frozenset({"ACQUIRE", "DISCOVER", "APPROACH"})
# Faster than RECORD_WALK_LERP so the pull-in has CONVERGED by the first EXTEND sample. The figure picks that
# sample 30% into EXTEND (>= ~1 s = 50 ticks after the switch); at 0.12 the residual is 0.88^50 ~= 0.2%, while
# the 0.04 pan rate would still be 13% wide there and the Extend column would read mid-zoom.
RECORD_WALK_DISTANCE_LERP = 0.12

# Planned-route waypoint markers are drawn this much larger in keyframes than in the live viewers, because a
# figure panel is ~1.2 in wide and the viewer-tuned 1.5 mm shaft prints below one pixel there.
KEYFRAME_ROUTE_MARKER_SCALE = 2.0


def _walk_cam_lookat_z() -> float:
    """Chase-cam aim height, overridable via ``WALK_CAM_LOOKAT_Z`` (see ``RECORD_WALK_LOOKAT_Z``)."""
    return float(os.environ.get("WALK_CAM_LOOKAT_Z", RECORD_WALK_LOOKAT_Z))


def _walk_cam_near_distance() -> float:
    """Tight (grasp) chase-cam distance, overridable via ``WALK_CAM_DISTANCE_M``. The wide end of the phase
    ramp is fixed at ``RECORD_WALK_CAMERA_DISTANCE_WIDE``."""
    return float(os.environ.get("WALK_CAM_DISTANCE_M", RECORD_WALK_CAMERA_DISTANCE))


def _walk_cam_yaw_offset() -> float:
    """Chase-cam yaw offset relative to base heading, overridable via ``WALK_CAM_YAW_OFFSET_DEG``.

    The offset is measured against the ROBOT, and the 145 deg default puts the lens on the robot's
    right-front -- exactly where the handoff scenarios (``bimanual_mixed*``) stand their human, whose body
    then blocks the torso in every frame. 215 mirrors the same 3/4 view onto the left-front, clearing the
    human without changing distance, elevation, or cube bias. Read here rather than baked into the camera at
    open time because ``_walk_cam_track`` re-derives the target azimuth every tick and would otherwise ease
    an overridden start value straight back to the default.

    Environment variable rather than a CLI flag: this is per-scene camera calibration, not a mission
    parameter, and nothing but the figure sweep sets it."""
    return float(os.environ.get("WALK_CAM_YAW_OFFSET_DEG", RECORD_WALK_CAMERA_YAW_OFFSET_DEG))


def _walk_cam_track(cam, base_pose, target_xy, phase: str | None = None) -> None:
    """Ease ``cam.lookat`` toward the working cube, ``cam.azimuth`` to a front 3/4 view of the base, and
    ``cam.distance`` between the wide scene framing and the tight grasp framing (see
    ``RECORD_WALK_CAMERA_DISTANCE_WIDE``). ``phase`` None keeps the near distance, i.e. the old behaviour.

    Two framing failures this fixes, both found by looking at rendered frames. (1) A base-centred lookat put
    the cube at the frame edge -- it sits ~0.3-0.5 m in front of the stance -- so the crop spent its pixels on
    floor behind the robot; bias toward the target instead. (2) A FIXED world azimuth filmed whichever side
    the walk happened to present, and at spawn heading (+x) that was the robot's BACK, with its own torso
    between the lens and the table. Deriving azimuth from live base yaw holds one relative viewpoint for every
    scenario and heading, so the gripper and cube stay on the camera side of the robot.

    ``cam`` doubles as the filter state (the recorder outlives the trial loop), so the pan stays continuous
    across a seed boundary too. Azimuth eases along the SHORTEST arc, else a heading crossing +/-180 spins
    the camera the long way round.
    """
    base_xy, base_quat = base_pose
    if target_xy is None:
        goal_x, goal_y = base_xy[0], base_xy[1]
    else:
        w = RECORD_WALK_TARGET_WEIGHT
        goal_x = (1.0 - w) * base_xy[0] + w * target_xy[0]
        goal_y = (1.0 - w) * base_xy[1] + w * target_xy[1]
    cam.lookat[0] += RECORD_WALK_LERP * (goal_x - cam.lookat[0])
    cam.lookat[1] += RECORD_WALK_LERP * (goal_y - cam.lookat[1])
    cam.lookat[2] = _walk_cam_lookat_z()
    qw, qx, qy, qz = base_quat
    yaw = np.degrees(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    delta = (yaw + _walk_cam_yaw_offset() - cam.azimuth + 180.0) % 360.0 - 180.0
    cam.azimuth += RECORD_WALK_LERP * delta
    goal_d = (RECORD_WALK_CAMERA_DISTANCE_WIDE if phase in RECORD_WALK_ESTABLISHING_PHASES
              else _walk_cam_near_distance())
    cam.distance += RECORD_WALK_DISTANCE_LERP * (goal_d - cam.distance)


def _record_initial_setup(mj_model, robot_cfg, out_dir: str, scenario_name: str) -> str:
    """Render one clean initial task-layout PNG without invoking cuRobo planning or trajectory replay."""
    import imageio.v2 as imageio

    from tasks.visual_manipulation.curobo_reach_harness import apply_view_options

    renderer = mujoco.Renderer(mj_model, height=RECORD_H, width=RECORD_W)
    opt = mujoco.MjvOption()
    # Setup sheet exists to tell the camera-aim story, so the frustums are always on here.
    apply_view_options(opt, show_fov=True)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.65]
    cam.distance = RECORD_FAR_CAMERA_DISTANCE if scenario_name.endswith("_far") else RECORD_CAMERA_DISTANCE
    cam.azimuth, cam.elevation = _record_camera_pose(scenario_name)
    d = mujoco.MjData(mj_model)
    d.qpos[:] = mj_model.qpos0
    _apply_home_pose(mj_model, d, robot_cfg)
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=cam, scene_option=opt)
    out_path = os.path.join(out_dir, f"{robot_cfg.name}_{scenario_name}_initial.png")
    imageio.imwrite(out_path, renderer.render())
    renderer.close()
    print(f"[record-initial] wrote {out_path}", flush=True)
    return out_path


# ------------------------------------------------------------------------------------------------
# CLI + main
# ------------------------------------------------------------------------------------------------
@dataclass
class Args:
    robot: str = "v2"
    """Which robot: v2 | v2_fixed | g1 | both. `v2` is the humanoid_v21 with actuated gimbal cams
    (default); `v2_fixed` is the same robot with welded fixed cams. `both` runs the selected mode on each
    of g1, v2_fixed, and v2 against ONE shared seeded scene and prints a side-by-side summary."""
    scenario: str = BIMANUAL_SCENARIO_NAME
    """One scenario per run. Validated by ``_resolve_scenarios`` against ``SCENARIOS``.
    Close (stationary) 2-object scenario for the reach loop / drift study. `shelf` is the standard shelf scenario;
    reachability-driven (reaches whatever the robot can plan to on this scene: both cubes -> bimanual,
    one -> single). For --robot both pick a mutually reachable scene (e.g. front_back_close);
    g1 cannot reach bimanual_mixed_close's far-lateral object from a single STATIONARY stance (no --walk) --
    with --walk the base repositions per cube and g1 passes (verified 2026-07-22, P=1.000 10/10)."""
    walk: bool = False
    """Run the multi-cube WALK mission on the shared online ``ReachPolicy(walk=True)`` (per-visit
    SEARCH->GO->REACH->PARK): the base drives toward each cube until arm-reachable, then reaches it. Kinematic
    (default) integrates the base twist on ``mj_forward``; with --dynamic the frozen policy + physics realize
    it. Combine with --view for the live passive viewer (walk->search->reach->park per cube)."""
    camera: bool = False
    """Visibility gate for the reach loop AND --walk. Default (privileged): reach any cube whose
    GROUND-TRUTH pose is reachable. --camera: reach ONLY cubes the head cameras actually SEE (fixed-cam
    static frustum; actuated v2 gimbals aimed per cube), so the reachability-driven arm count follows the
    visible∩reachable set; unseen cubes are not reached."""
    drift: bool = False
    """Inject a scripted-sinusoid base drift into the unified reach loop (KinematicHarness): the plan-once
    ReachPolicy tracks a joint reference fixed in the PLAN-time base frame while the base drifts, so the
    open-loop tool slides off the world-fixed target. Headless: per-tick (reach_err, base_drift) log + the
    final open-loop MISS baseline (the step-2 corrector must beat it). With --view: animate live."""
    view: bool = False
    """Open the MuJoCo viewer and replay the planned trajectory. The cuRobo planner collision spheres are
    added as child geoms on the robot links (geom group 5); press the "5" key in the viewer to toggle them."""
    viewer: Literal["native", "newton", "blender"] = "native"
    """Renderer for BOTH --view and --record. "native" = MuJoCo (default): passive viewer live, offscreen
    ``mujoco.Renderer`` for the MP4; overlays, screenshot capture and the "5"/"o" keys all work. "newton" =
    Newton's ViewerGL, real-time PBR instead of MuJoCo's fixed-function Phong (--view only). "blender" =
    Blender/EEVEE global illumination -- live for --view (kinematic and --dynamic alike), and for --record
    the SAME graded rollout re-rendered offscreen into the SAME MP4 path. Both non-native renderers are
    scene only: no overlays, no capture, no key toggles (see ``photoreal.bridge`` /
    ``photoreal.newton_bridge`` module docstrings)."""
    speed: float = 2.0
    """--view only: viewer playback speed multiplier (mjlab's ``BaseViewer.SPEED_MULTIPLIERS``: 1/32 .. 8;
    the nearest listed value is used). The viewer paces physics to WALL CLOCK, so a mission never finishes
    faster than its simulated length at 1x no matter how much CPU headroom exists -- the 2.0 default halves
    the watch time, when compute can keep up, and falls back to whatever the machine sustains when it cannot.
    Defaulting above 1x is only safe because the metrics below are speed-invariant; pass ``--speed 1`` to
    watch at real time.
    Metric-neutral for everything METRICS2 reports as a mission quantity: durations and ``energy_J`` are
    measured in EXEC ticks (see ``_report_walk_metrics``), which freeze while the mission is stalled on an
    off-loop cuRobo solve, so the wall-clock latency this flag rescales is excluded rather than measured.
    ``plan_wait_s`` is the exception BY DESIGN -- it reports that excluded wall latency, so it does move with
    speed. Small residual jitter remains in the rollout itself: physics keeps stepping through a stall, and
    ``_final_replan_pending`` ticks stay in the measurement, so a refresh can land on a different tick.
    Verdicts (physical latch + upright) are robust."""
    window_size: str | None = None
    """--view only: request the native-viewer OS window size as "WIDTHxHEIGHT" (e.g. "1920x1920").
    Applied via xdotool after the window opens (X11 only, best-effort; the window manager clamps a
    request bigger than the monitor). No effect on the viser web-viewer fallback (headless/no-DISPLAY)."""
    mpc: bool = True
    """Drive the reach with the MPC reactive TRACKER (per-tick optimize against measured joints) instead of
    replaying the planned trajectory open-loop. Default ON because it is the ONLY path with gravity
    compensation: ``--no-mpc`` runs ``DynamicArmReferenceExecutor``, which interpolates the planned q into a
    PD position servo with no feedforward, so the arm settles ``tau_gravity(q)/kp`` low on every grasp. That
    droop silently ate a grasp whenever anything let the gripper approach from below the object's support,
    and every published/SOP evaluation already passed --mpc explicitly -- the old default only ever applied
    to someone who forgot the flag. ``--no-mpc`` is kept for the open-loop baseline (and for --view --drift,
    where it selects the full-trajectory replan corrector rather than the tracker)."""
    seed: int | None = None
    """Deterministically jitter pick-object XY for this seed. None -> a fresh seed each run (printed) on the
    --walk/--dynamic paths, and the un-jittered nominal layout on the parked kinematic reach (the selftest's
    determinism contract). With --record-trials N the sweep runs seeds SEED..SEED+N-1 (SEED defaults to 42)."""
    record: str | None = None
    """Directory for offscreen MP4s. --dynamic (parked reach): delegates to the pickplace mission's offscreen
    path. --dynamic --walk: renders the graded walk rollout (chase cam) to DIR/<robot>_<scenario>_walk.mp4.
    Kinematic --walk --record is still unwired (loud-fails in main); use --dynamic --walk --record for video.
    ``--viewer blender`` re-renders that same walk MP4 through Blender/EEVEE instead of MuJoCo (offscreen
    either way; the parked delegation is MuJoCo-only and loud-fails)."""
    record_initial: str | None = None
    """Render one clean initial task-layout PNG to this directory, without cuRobo planning or rollout replay."""
    record_keyframes: str | None = None
    """--walk only: directory for FSM KEYFRAME PNGs of the graded rollout. One frame per ``walk_phase``
    transition (ACQUIRE/DISCOVER/APPROACH/EXTEND/GRASP/RETRACT_PLAN/RETRACT/TERMINATE), on the SAME chase cam
    the MP4 uses, plus a ``*_keyframes.json`` manifest carrying each frame's phase, visit cube, and sim time.
    Combines with --record. Intended for the per-scenario keyframe figure: run the 6 scenarios, keep the seeds
    whose manifest says ``"pass": true``, and slice a uniform column set from the manifests (n varies per
    scenario -- a DISCOVER phase or a second visit adds transitions)."""
    keyframe_every: float = 0.0
    """--record-keyframes only: ALSO sample a frame every this many sim seconds WITHIN a phase (0 = phase
    transitions only). Transitions alone give a thin figure: a phase's entry frame is the PREVIOUS phase's
    achieved state, so EXTEND-entry duplicates ACQUIRE (arm still at home) and RETRACT-entry duplicates
    RETRACT_PLAN (jaws already shut), and no transition lands mid-reach or mid-walk -- exactly the motion a
    figure panel must show. 0.5 s samples the 3-4 s EXTEND and the walk-in densely enough for a figure script
    to pick a frame at any fraction into a phase."""
    record_trials: int = 1
    """Number of sequential seeded trials to evaluate in this single process, reporting the aggregate
    METRICS line. Supported on every headless grade (parked reach, --walk, --dynamic). Prefer this over a
    per-seed subprocess fan-out: the compiled model + warmed cuRobo session are built ONCE, so trials
    2..N cost a warm solve instead of the ~8.4 s cold start (import + compile + cuRobo graph capture)."""
    metrics_only: bool = False
    """Print standard METRICS summary line."""
    checkpoint: str | None = None
    """--dynamic only: pin the frozen low-level policy to this exact .pt path instead of
    ``find_latest_checkpoint`` (which picks the newest-mtime checkpoint under runs/<task>/*/, and can
    silently hijack eval onto an unrelated, freshly-started/undertrained training run sharing the same
    task name). None (default) = current latest-wins behavior."""

    selftest: bool = False
    """Run the reachability-driven reach REGRESSION suite: the fixed (robot x camera) arm-count matrix on
    the nominal front_back_close scene, asserting each cell's active-arm count + reach error, plus a fast
    empty-belief hold check. Ignores --robot/--scenario/--seed. Exits 0 only if every cell PASSES."""
    expect_arms: int | None = None
    """Assert the reach loop plans EXACTLY this many active arms (0 = hold). None (default) = print-only,
    no count assertion. Applies to the single ad-hoc reach run; --selftest sets it per matrix cell."""

    height_pad: float = 0.02
    """Additive Z padding (in meters) for object cuboids in CuRobo's planner perspective and geom group 5 (default: 0.02)."""



    dynamic: bool = False
    """Realize the SAME ReachPolicy through a FROZEN RL low-level policy + MuJoCo/warp physics
    (DynamicHarness) instead of the idealized kinematic mj_forward. A PASS means the policy tracked a
    kinematically-valid plan THROUGH real dynamics. Single-robot (physics env is one agent); rejects
    --drift (redundant under physics) and --both. Parked --record delegates to the proven pickplace_reach_env
    mission; --walk runs the physics walk brain here (--record renders its rollout offscreen). Camera gate
    (--camera), --view, --mpc all work here too."""
    steps: int = _REACH_STEPS
    """Reach TIMEOUT cap (max ticks). The reach converges early on sustained proximity, or stops here.
    Dynamic delegation (--walk/--record): the pickplace mission auto-resolves its own length unless this
    is set away from the default."""
    settle_s: float = _DYNAMIC_SETTLE_S
    """--dynamic parked reach only: pre-plan planning-home hold duration in seconds. Default 2 s lets the
    base + arm enter their standing limit cycle; mj_forward is instant, so this is a no-op kinematically."""
    device: str = "cuda:0"
    """--dynamic only: torch device for the frozen policy + warp physics."""
    replan_interval_s: float | None = None
    """--dynamic non-MPC only: None = plan ONCE (open-loop, correct for a PARKED reach). A float re-solves
    the full trajectory every interval (only useful for a moving base; restarts the arm each replan)."""
    grav_comp: bool = True
    """--dynamic --mpc only: enabled-by-default gravity-comp feedforward; pre-bends the arm reference by
    the PD gravity droop so the position servo lands ON target."""
    grav_alpha: float | None = None
    """--dynamic --mpc only: override robot gravity-comp gain [0,1]. 0 = un-compensated position servo."""
    filter: bool = False
    """--dynamic --mpc only: enable the tracker's critically-damped output filter (ik_mink profile) to
    damp the parked-reach limit cycle."""
    omega: float | None = None
    """--dynamic --mpc only: override robot tracker-filter frequency (rad/s); lower = more damping."""


def _resolve_scenarios(spec: str, *, allow_all: bool) -> tuple[str, ...]:
    """Resolve ``--scenario`` to canonical scenario name(s).

    One scenario per run: each compiles its own world, so a name is a whole evaluation. ``all`` expands to
    ``INITIAL_RECORD_SCENARIOS`` and is the only multi-valued form (``--record-initial`` only).

    Args:
        spec: raw ``--scenario`` value.
        allow_all: whether the calling mode implements the ``all`` expansion (only ``--record-initial``
            does today; every other mode must reject it loudly rather than silently run one scenario).
    """
    token = spec.strip()
    if token == "all":
        if not allow_all:
            raise SystemExit("--scenario all requires --record-initial")
        return tuple(dict.fromkeys(SCENARIOS[t].name for t in INITIAL_RECORD_SCENARIOS))
    if token not in SCENARIOS:
        raise SystemExit(f"unknown scenario {token!r}; choices: {sorted(SCENARIO_NAMES)}")
    return (SCENARIOS[token].name,)


def _parse_window_size(spec: str | None) -> tuple[int, int] | None:
    if spec is None:
        return None
    width, _, height = spec.partition("x")
    return int(width), int(height)


_DRIFT_REPLAN_STEPS_SWEEP = (1, 5, 10, 25)     # serial-replan cadences to sweep under --drift (X ticks/solve)
_DRIFT_REPLAN_S = 0.10                          # viewer corrector cadence (~5 ticks @ 50 Hz control)


def _run_drift_eval(harness, robot: str, scenario_name: str) -> bool:
    """``--drift`` testbed: ZERO-MEAN base-drift SINUSOID (a continuously MOVING base). Run the plan-once
    BASELINE (open-loop miss ~= sway amplitude), then a SERIAL live-base replan SWEEP over X-step cadences
    (``ReachPolicy.replan_interval_s``) and the ASYNC spawned-worker path, reporting each PEAK settled
    reach_err (max over the tail window -- the final instant is phase-dependent under a sinusoid) + wall-clock
    + solve count. PASS = the best corrected residual beats the reach tol."""
    from tasks.visual_manipulation.curobo_reach_harness import run_reach, _DRIFT_AMP_M, _DRIFT_PERIOD_STEPS
    from tasks.visual_manipulation.reach_policy import ReachPolicy

    def _run(replan_interval_s, *, async_replan=False, mpc_track=False):
        harness.reset()
        # Async/MPC pass session=None: the sole cuRobo is the spawned worker, so the policy provably touches
        # no in-process runtime. Baseline/serial pass the harness session (they solve sync in-process).
        pol = ReachPolicy(None if (async_replan or mpc_track) else harness.session, harness.scenario,
                          harness.mj_model, harness.control_dt, replan_interval_s=replan_interval_s,
                          async_replan=async_replan, mpc_track=mpc_track,
                          robot_name=robot, scenario_name=scenario_name)
        t0 = time.monotonic()
        try:
            # PEAK reach_err over the last full sway cycle (phase-independent worst tracking error).
            err = run_reach(harness, pol, _REACH_STEPS, settle_window=_DRIFT_PERIOD_STEPS)
        finally:
            pol.close()                                    # stop spawned workers (no-op in sync mode)
        return err, time.monotonic() - t0, pol.replan_failures

    base_err, base_wall, _ = _run(None)
    print(f"  drift [{robot} {scenario_name}] sway amplitude {_DRIFT_AMP_M:.3f} m; peak reach_err = worst "
          f"tracking error over the last {_DRIFT_PERIOD_STEPS}-tick cycle")
    print(f"  drift [{robot} {scenario_name}] BASELINE plan-once: peak reach_err {base_err:.4f} m "
          f"(= OPEN-LOOP MISS ~ amplitude), wall {base_wall:.1f}s")
    best = float("inf")
    for x in _DRIFT_REPLAN_STEPS_SWEEP:
        err, wall, fails = _run(x * harness.control_dt)
        solves = 1 + (_REACH_STEPS - 1) // x
        best = min(best, err)
        print(f"  drift [{robot} {scenario_name}] SERIAL replan every {x:3d} step "
              f"({x * harness.control_dt:.2f}s): peak reach_err {err:.4f} m, "
              f"~{solves} solves, {fails} fail, wall {wall:.1f}s")
    # ASYNC: spawned single-arm-per-side workers, per-tick non-blocking submit. The base keeps MOVING during
    # each solve, so this is a tracking-LATENCY test (solve lag -> residual lag); wall is wall-clock, not
    # solver time (solves overlap the rollout, warmed once at plan-0).
    a_err, a_wall, a_fails = _run(harness.control_dt, async_replan=True)
    best = min(best, a_err)
    print(f"  drift [{robot} {scenario_name}] ASYNC replan (per-tick submit): peak reach_err {a_err:.4f} m, "
          f"{a_fails} fail, wall {a_wall:.1f}s")
    # MPC reactive TRACKER: the ONE cuRobo process runs plan-0 (plan_pose) + a warm MPCSolver that HOLDS each
    # hand's world-fixed achieved grasp as the base drifts. Not a full-trajectory replan -> no solve-latency
    # staleness; the open question is whether reactive tracking clears the bandwidth wall the replan sweep hit.
    m_err, m_wall, _ = _run(harness.control_dt, mpc_track=True)
    best = min(best, m_err)
    print(f"  drift [{robot} {scenario_name}] MPC tracker (per-tick optimize): peak reach_err {m_err:.4f} m, "
          f"wall {m_wall:.1f}s")
    passed = best < _REACH_TOL_M
    print(f"  drift [{robot} {scenario_name}] -> best corrected {best:.4f} m vs tol {_REACH_TOL_M} m "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def _run_reach_loop(robot: str, scenario_name: str, args, *, harness=None, expect_arms=None) -> bool:
    """Reachability-driven KINEMATIC reach eval on the unified harness loop -- the kinematic twin of the
    ``--dynamic`` path (``_run_dynamic_reach``): SAME ``ReachPolicy``, SAME scene, SAME shared gate;
    kinematic ``mj_forward`` instead of frozen-RL physics. ONE policy reaches whatever cubes are reachable
    on the scene (both -> bimanual,
    one -> single, none -> hold); with ``--camera`` only cubes the head cameras SEE are eligible, so the
    arm count follows the visible∩reachable set. Gates on the realized tool-site error (MAX over active
    arms) to the route grasp candidate. This is the sole reach path (bimanual, single, and hold all fall
    out of the reachability-driven arm count); ``--drift`` injects the open-loop-miss baseline here.

    Idealized robot -> ~0 m; gate ``_REACH_TOL_M``. With ``--view`` opens the live passive viewer
    (re-runs the reach until closed) instead of the headless gate (always True in viewer mode --
    visual inspection, no gate).

    ``harness``: a prebuilt ``KinematicHarness`` to REUSE (its warmed cuRobo session) -- the selftest
    passes one per robot and toggles ``.camera`` across cells so the session is built once, not per cell.
    When reused the harness's ``camera`` is set to ``args.camera`` here. ``expect_arms``: if not None,
    assert the plan's active-arm count equals it (0 = a correct hold); PASS then requires BOTH the count
    match AND (hold, or reach_err < tol)."""
    from tasks.visual_manipulation.curobo_reach_harness import KinematicHarness, run_reach, run_reach_viewer
    from tasks.visual_manipulation.reach_policy import ReachPolicy

    if harness is None:
        harness = KinematicHarness(robot, scenario_name, camera=args.camera, drift=args.drift,
                                   height_pad=args.height_pad)
    else:
        harness.camera = args.camera

    def _policy(replan_interval_s=None):
        return ReachPolicy(harness.session, harness.scenario, harness.mj_model, harness.control_dt,
                           replan_interval_s=replan_interval_s,
                           height_pad=args.height_pad)



    if args.view:
        # Under --drift the viewer runs a CORRECTOR so the tool stays on the world-fixed target as the base
        # sways: --mpc = the MPC reactive tracker, else the full-trajectory live-base replan. No drift = the
        # plain plan-once reach.
        if args.mpc:
            # ONE persistent cuRobo worker shared across every reach: the ~7 s cold cuRobo/CUDA warmup (cold
            # spawn import + first-solve cuda-graph capture, 3.5 s alone) is paid ONCE, not per reach. Each
            # fresh policy re-runs plan-0 in this warm worker (re-solve ~0.08 s + MPC rebuild ~0.8 s), so
            # between-reach latency drops from ~8 s to <1 s. Worker owned here (closed after the viewer loop).
            from tasks.visual_manipulation.curobo.scene import SpawnedReachMpcWorker
            shared_worker = SpawnedReachMpcWorker(robot, scenario_name)
            factory = lambda: ReachPolicy(  # noqa: E731
                None, harness.scenario, harness.mj_model, harness.control_dt,
                replan_interval_s=harness.control_dt, mpc_track=True,
                robot_name=robot, scenario_name=scenario_name, worker=shared_worker,
                # KinematicHarness writes this reference directly into qpos. Its idealized arm has no
                # low-level position-servo gravity droop, so dynamic-only pre-bending would corrupt the
                # cuRobo trajectory instead of compensating anything.
                gravity_comp=False)
            try:
                run_reach_viewer(harness, factory, _REACH_STEPS, start_seed=args.seed,
                                  window_size=_parse_window_size(args.window_size), viewer=args.viewer)
            finally:
                shared_worker.close()
            return True
        factory = (lambda: _policy(_DRIFT_REPLAN_S)) if args.drift else _policy
        run_reach_viewer(harness, factory, _REACH_STEPS, start_seed=args.seed,
                         window_size=_parse_window_size(args.window_size), viewer=args.viewer)
        return True

    if args.drift:
        return _run_drift_eval(harness, robot, scenario_name)

    def _one_reach(seed):
        """Reset to ``seed``'s cube layout, run ONE full kinematic reach, grade it.

        ``seed=None`` keeps the harness's current (nominal) layout -- the selftest's determinism contract.
        Kinematic full-runs to the _REACH_STEPS cap and reads the settled tool error (~0 m for the idealized
        robot -- the clean geometric-reachability signal). No converge-early: kinematic tracks a ~150-tick
        trajectory, so a window-max early stop would pin the report near the gate. NaN if the policy held.
        """
        harness.reset(seed)
        policy = _policy()
        err = run_reach(harness, policy, _REACH_STEPS)
        route = policy.last_route
        arms = len(route.reaches) if route is not None else 0
        planned = route.reach_error if route is not None else float("nan")
        up = harness.upright()                              # 1.0 for the idealized kinematic robot (never falls)
        # A hold has no reach to grade -- but "planned nothing" only counts as a pass when the caller ASKED
        # for zero arms. Unconditionally passing arms==0 graded a total planner failure (no goal solved, err
        # NaN) as PASS, silently, on every un-gated run. NaN also fails the err compare below, so a policy
        # that held while arms>0 no longer slips through either.
        reach_ok = (err < _REACH_TOL_M and up > _UPRIGHT_MIN) if arms else expect_arms == 0
        count_ok = expect_arms is None or arms == expect_arms
        return reach_ok and count_ok, arms, err, planned

    gate = "camera-gated" if args.camera else "privileged"
    expect_str = "" if expect_arms is None else f" (expected {expect_arms}-arm)"
    if args.record_trials > 1:
        # Seed sweep IN-PROCESS: harness (compiled model + warmed cuRobo session) is built once, so trial 2..N
        # cost one warm solve (~0.3 s) each instead of a fresh process's ~8.4 s of import + compile + cuRobo
        # cold-start. Same amortization the --walk path already had; the parked path used to force the
        # per-seed subprocess fan-out in test/run_*_eval.py.
        base_seed = args.seed if args.seed is not None else 42
        successes = 0
        for trial in range(args.record_trials):
            seed = base_seed + trial
            passed, arms, err, planned = _one_reach(seed)
            successes += passed
            print(f"  reach [{robot}] {gate} seed {seed}: {arms}-arm{expect_str}, reach_err {err:.4f} m "
                  f"(planned {planned:.4f} m) -> {'PASS' if passed else 'FAIL'}")
        P = successes / float(args.record_trials)
        # Tbar is the parked reach's fixed sim length (it always runs the full cap); emitted so the sweep
        # runners' shared METRICS regex parses this path too.
        print(f"METRICS: robot={robot:<10} scenario={scenario_name:<35} P={P:.3f} "
              f"Tbar={_REACH_STEPS * harness.control_dt:.2f}s success={successes}/{args.record_trials}",
              flush=True)
        return P > 0.0

    passed, arms, err, planned = _one_reach(args.seed)
    print(f"  reach [{robot}] {gate}: {arms}-arm{expect_str}, reach_err {err:.4f} m "
          f"(planned {planned:.4f} m) -> {'PASS' if passed else 'FAIL'}")
    return passed


def _run_kinematic_walk(robot: str, scenario_name: str, args) -> bool:
    """KINEMATIC walk mission on the unified harness loop -- the kinematic twin of
    ``_run_dynamic_reach(walk=True)``: SAME ``ReachPolicy(walk=True)`` + SAME per-visit
    SEARCH->GO->REACH->PARK, realized on bare ``mj_forward`` (idealized robot). Headless grade via
    ``_grade_walk_headless``; ``--view`` opens the walk-aware passive viewer (walk->search->reach->park per
    cube until the window closes). Replaces the retired offline ``_run_walk_in_reach`` planner+replay island
    -- ONE walk brain (the online policy) drives both the kinematic and dynamic paths."""
    from tasks.visual_manipulation.curobo_reach_harness import make_harness, run_reach_viewer
    from tasks.visual_manipulation.reach_policy import ReachPolicy

    harness = make_harness(robot, scenario_name, dynamic=False, camera=args.camera, walk=True)

    def make_policy():
        return ReachPolicy(harness.session, harness.scenario, harness.mj_model, control_dt=harness.control_dt,
                           robot_name=robot, scenario_name=scenario_name, walk=True, walk_device=args.device,
                           height_pad=args.height_pad)


    mode = "camera-gated (must SEE cube to reach)" if args.camera else "privileged (ground-truth, no camera)"
    print(f"kinematic walk-reach: robot={robot} scenario={scenario_name!r} table_z={harness.table_z:.3f} "
          f"control_dt={harness.control_dt:.3f}s mode={mode}")
    if args.seed is not None:
        harness.reset(args.seed)                         # match --view's run_reach_viewer(start_seed=...)
    harness.settle()
    cap = args.steps if args.steps != _REACH_STEPS else _WALK_STEP_CAP
    if args.view:
        run_reach_viewer(harness, make_policy, cap, start_seed=args.seed,
                         window_size=_parse_window_size(args.window_size))
        return True
    pol = make_policy()
    try:
        return _run_walk_trials(harness, pol, cap, args, robot, scenario_name)
    finally:
        pol.close()                                        # stop spawned cuRobo worker (else it orphans)


def _run_record_initial(robot_cfg: RobotDescriptor, object_cfg: ObjectSceneCfg, args) -> bool:
    """Render a clean initial task-layout PNG (no cuRobo, no replay): compile the scene with the robot at
    home + snapshot. Standalone dispatch extracted from the retired offline walk planner -- ``--record-initial``
    has NO pickplace-mission equivalent (the mission renders a physics rollout, not a static setup sheet), so
    it keeps its own tiny compile->``_record_initial_setup`` path. Honours the ``FLOOR_GRID`` env var via the
    harness helper so the static sheet matches the live rendered scene."""
    from tasks.visual_manipulation.curobo_reach_harness import (
        _FLOOR_GRID_CELL_M, _FLOOR_GRID_MAT, _FLOOR_GRID_TEX, _floor_grid_enabled,
    )
    scenario = _apply_seed(object_cfg, args.seed)
    merged_spec = robot_cfg.build_entity_spec()
    ground = merged_spec.worldbody.add_geom()
    ground.name = "phase3_record_ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [30.0, 30.0, 0.1]
    ground.contype = 0
    ground.conaffinity = 0
    if _floor_grid_enabled():
        # Same checker pattern the live harness renders (paper/video aesthetic + manual travel scale).
        # Lives on the inline sheet plane, not the harness's "terrain" body -- different name, same
        # texture/material so the printed grid matches the rendered chase cam exactly.
        if not any(m.name == _FLOOR_GRID_MAT for m in merged_spec.materials):
            tex = merged_spec.add_texture(name=_FLOOR_GRID_TEX)
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
            tex.rgb1 = [0.78, 0.80, 0.84]
            tex.rgb2 = [0.30, 0.34, 0.40]
            tex.mark = mujoco.mjtMark.mjMARK_EDGE
            tex.markrgb = [0.20, 0.22, 0.26]
            tex.width = 512
            tex.height = 512
            mat = merged_spec.add_material(name=_FLOOR_GRID_MAT)
            mat.texuniform = True
            n = max(1, int(round(1.0 / _FLOOR_GRID_CELL_M)))
            mat.texrepeat = [float(n), float(n)]
            mat.reflectance = 0.0
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _FLOOR_GRID_TEX
        ground.material = _FLOOR_GRID_MAT
    else:
        ground.rgba = [0.18, 0.22, 0.25, 1.0]
    make_scene_spec_fn(scenario)(merged_spec)
    merged_spec.visual.global_.offwidth = max(RECORD_W, 640)
    merged_spec.visual.global_.offheight = max(RECORD_H, 480)
    mj_model_scene = merged_spec.compile()
    _move_pick_geoms(mj_model_scene, scenario)
    os.makedirs(args.record_initial, exist_ok=True)
    _record_initial_setup(mj_model_scene, robot_cfg, args.record_initial, scenario.name)
    return True


# Regression matrix: active-arm count per (robot, camera) on the nominal front_back_close scene. Privileged
# reaches both cubes with both arms; under --camera the count follows the visible∩reachable set -- g1's
# single forward camera sees only the front cube (1 arm), v2/v2_fixed's paired/actuated cams see both (2).
_SELFTEST_SCENARIO = "front_back_close"
_SELFTEST_MATRIX = {                       # robot -> (privileged_arms, camera_arms)
    "v2": (2, 2),
    "v2_fixed": (2, 2),
    "g1": (2, 1),
}


def _run_reach_selftest(args) -> None:
    """Deterministic reach regression suite. For each robot builds ONE camera-capable KinematicHarness
    (warms cuRobo once, allocates the gaze scratch), then grades the privileged and camera-gated cells by
    toggling ``harness.camera`` -- 3 cold starts, not 6. Each cell asserts the expected active-arm count
    (``_SELFTEST_MATRIX``) plus reach_err < tol. Also runs a fast EMPTY-belief hold check (no cuRobo plan):
    an empty observed-cube set must yield the invariant-4 hold (zero base twist, no arm command, rest/no
    camera command, no route). ``os._exit(0)`` only if every cell + the hold check passes."""
    from tasks.visual_manipulation.curobo_reach_harness import KinematicHarness
    from tasks.visual_manipulation.reach_policy import ReachPolicy

    ok = True
    for robot, (priv_arms, cam_arms) in _SELFTEST_MATRIX.items():
        print(f"\n########## selftest [{robot}] on {_SELFTEST_SCENARIO!r} ##########")
        harness = KinematicHarness(robot, _SELFTEST_SCENARIO, camera=True, height_pad=args.height_pad)   # camera=True allocates gaze scratch
        ok = _run_reach_loop(robot, _SELFTEST_SCENARIO, dataclasses.replace(args, camera=False),
                             harness=harness, expect_arms=priv_arms) and ok
        ok = _run_reach_loop(robot, _SELFTEST_SCENARIO, dataclasses.replace(args, camera=True),
                             harness=harness, expect_arms=cam_arms) and ok

        # Hold branch (invariant 4): an EMPTY observed-cube belief has no candidates, so the policy holds
        # -- zero twist, no arm command, no route -- WITHOUT any cuRobo solve. Cheap + deterministic.
        harness.camera = False
        harness.reset()
        policy = ReachPolicy(harness.session, harness.scenario, harness.mj_model, harness.control_dt)
        base_pose, proprio, _cube = harness.observe()
        base_twist, arm_ref, cam = policy.step(base_pose, proprio, {})
        cam_rest_ok = cam == {} or (getattr(policy, "_gaze", None) is not None and cam == policy._gaze.cmd)
        hold_ok = base_twist == (0.0, 0.0, 0.0) and arm_ref == {} and cam_rest_ok and policy.last_route is None
        print(f"  hold  [{robot}]: empty belief -> "
              f"twist={base_twist} arm={len(arm_ref)} cam={len(cam)} route={policy.last_route} "
              f"-> {'PASS' if hold_ok else 'FAIL'}")
        ok = hold_ok and ok

    sys.stdout.flush()
    os._exit(0 if ok else 1)


def _run_view(harness, policy, settle_steps: int, *, walk: bool, steps_cap: int, dwell: int,
             base_seed: int, window_size: tuple[int, int] | None = None, speed: float = 1.0,
             viewer_kind: str = "native") -> bool:
    """Live viewer for the DYNAMIC harness: drive the SAME ``ReachPolicy`` through the mjlab viewer's
    obs->policy->step loop. The viewer owns ``env.step``; our closure produces the frozen low-level action
    each tick (``ReachPolicy.step`` -> arm reference -> ``env.reach_action``). Base twist is asserted ~0
    (parked reach). Native viewer when a display exists, else the viser web viewer. Ported verbatim from the
    former ``curobo_phase4_verify._run_view``, PLUS the SAME grading the headless path prints
    (``_grade_walk_headless`` / the ``run_reach``-based parked gate in ``_run_dynamic_reach``): the viewer
    owns ``env.step``, so a tick's action is not REALIZED until after our callback returns -- grade the
    PREVIOUS tick's route against the state the viewer just stepped to (one frame deferred), which
    reproduces headless's observe->step->realize->verdict order. Prints the identical ``VERDICT:`` /
    ``mission_verdict:`` / ``final:`` lines once, on convergence (walk: ``walk_done``; parked: the same
    converge-early window or ``parked_done``) or on ``steps_cap`` timeout, then keeps the viewer running so
    the user can keep watching. Returns the graded ``ok`` once the viewer window closes, so ``--view``
    drives the same process exit code as headless."""
    import torch
    from collections import deque

    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    from tasks.visual_manipulation.curobo_reach_harness import (
        _CAPTURE_DIR, ViewCapture, apply_view_options, resize_native_viewer_window,
    )

    env = harness.env

    # "o" -> save the current view (native viewer only; the viser web viewer has no GLFW key hook).
    # Same ViewCapture the kinematic viewer uses, so both viewers save the identical thing.
    capture = ViewCapture(harness, "dyn")

    settle_remaining = 0
    tick = 0
    # ``tick`` counts raw viewer frames; ``exec_tick`` counts execution frames and stops while the mission is
    # stalled on an off-loop cuRobo solve. The timeout MUST use ``exec_tick`` too: ``run_reach`` excludes
    # plan-0 waits from its timeout, and charging them only in --view can grade an arm mid-approach before it
    # receives the same 300 control steps as the headless path. The two grades share
    # ``_report_walk_metrics`` and must not measure a mission two ways.
    exec_tick = 0
    planner_wait_ticks = 0
    ok = False
    graded = False
    stop_reason = ""
    post_grade_ticks = 0            # ticks since the verdict; drives the auto-advance to the next seed
    hold_arm_ref = None             # last commanded arm reference, re-emitted while the verdict holds on screen
    pending_route = None            # route decided by the PREVIOUS tick; graded once physics has advanced
    pending_reaching = False        # walk: policy.is_reaching captured alongside pending_route
    pending_grasp_route = None      # parked: policy.grasp_route captured alongside pending_route
    pending_capture_ready = False   # closed-grasp or retract tick, physical latch may now be scored
    pending_exec = False            # prior callback issued a headless-counted control action
    pending_energy = False          # a mission tick was issued; its power sample is due once realized
    visit_min: dict[str, float] = {}
    capture_error: dict[str, float] = {}
    min_upright = 1.0
    window: deque = deque(maxlen=dwell)
    grasp_error = None
    # NOT walk-gated: only ``_report_walk`` is walk-only, but the VRW duty tick below reads this on every
    # reaching tick, parked included (crash 2026-08-02: ``--view`` without ``--walk`` hit
    # "'NoneType' object is not iterable" the first tick the arm reached). Matches the headless grade,
    # which has always derived it unconditionally from the scenario spec.
    expected = {m.name for m in harness.scenario.pick}
    # METRICS2 bookkeeping, mirroring ``_grade_walk_headless``; reported through the shared
    # ``_report_walk_metrics`` so the viewer and headless grades cannot diverge. Walk missions only --
    # a parked reach has no visits to chain.
    accumulate_energy = getattr(harness, "accumulate_energy", None)
    first_observed_tick: dict[str, int] = {}
    visit_start_tick: dict[str, int] = {}
    visit_end_tick: dict[str, int] = {}
    plan_first_tick: dict[str, int] = {}
    phase_ticks: dict[str, int] = {}
    route_assignments: list[dict[str, str]] = []
    prev_current: str | None = None
    metrics_due = False
    vrw_manip_ticks = 0
    vrw_duty_ticks = 0

    def _grade_pending() -> None:
        nonlocal min_upright, grasp_error
        if walk:
            if pending_reaching and pending_route is not None and not hasattr(harness, "grasp_capture_error"):
                err = harness.verdict(pending_route)
                for c in pending_route.assignment.values():
                    visit_min[c] = min(visit_min.get(c, float("inf")), err)
            if pending_capture_ready and pending_grasp_route is not None:
                capture_errors = getattr(harness, "grasp_capture_errors", None)
                if capture_errors is not None:
                    for c, diagnostic in capture_errors(pending_grasp_route).items():
                        visit_min[c] = 0.0
                        capture_error[c] = min(capture_error.get(c, float("inf")), diagnostic)
                else:
                    score, diagnostic = _capture_result(harness, pending_grasp_route)
                    if np.isfinite(score):
                        for c in pending_grasp_route.assignment.values():
                            visit_min[c] = min(visit_min.get(c, float("inf")), score)
                            capture_error[c] = min(capture_error.get(c, float("inf")), diagnostic)
            min_upright = min(min_upright, harness.upright())
        else:
            if pending_grasp_route is not None and grasp_error is None:
                score, _ = _capture_result(harness, pending_grasp_route)
                if np.isfinite(score):
                    grasp_error = score
            if pending_route is not None:
                window.append(harness.verdict(pending_route))
            else:
                window.clear()          # route drop resets convergence (no pre/post-drop blend)

    def _report_walk() -> bool:
        missing = expected - set(visit_min)
        reached = {c: (e < _REACH_TOL_M) for c, e in visit_min.items()}
        result = (set(visit_min) == expected and all(reached[c] for c in expected)
                  and min_upright > _UPRIGHT_MIN)
        criterion = ("completed physical latch" if hasattr(harness, "grasp_capture_error")
                     else f"each visit reach_tol {_REACH_TOL_M} m")
        print(f"walk mission: {exec_tick} executing steps, stop={stop_reason}, "
              f"planner_wait={planner_wait_ticks} frames, "
              f"visits={ {c: round(e, 4) for c, e in visit_min.items()} } m, "
              f"expected={sorted(expected)}, missing={sorted(missing)}, min_upright {min_upright:.3f}, "
              f"done={policy.walk_done}")
        if capture_error:
            print(f"capture diagnostic: { {c: round(e, 4) for c, e in capture_error.items()} } m")
        print(f"mission_verdict: {policy.mission_verdict}")
        print(f"VERDICT: {'PASS' if result else 'FAIL'} ({criterion}, "
              f"upright_min {_UPRIGHT_MIN})")
        return result

    def _report_parked() -> bool:
        err = grasp_error if grasp_error is not None else (max(window) if window else float("nan"))
        up = harness.upright()
        result = err < _REACH_TOL_M and up > _UPRIGHT_MIN
        criterion = ("completed physical latch" if hasattr(harness, "grasp_capture_error")
                     else f"reach_tol {_REACH_TOL_M} m")
        plans = policy._executor.plan_id if policy._executor is not None else 0
        planned = policy.last_route.reach_error if policy.last_route is not None else float("nan")
        latch = "completed" if grasp_error is not None else "not completed"
        print(f"final: realized reach_err {err:.4f} m (planned {planned:.4f} m), upright {up:.3f}, "
              f"plans {plans}, replan_failures {policy.replan_failures}, latch {latch}, "
              f"stop={stop_reason}, executing_steps={exec_tick}, "
              f"planner_wait_frames={planner_wait_ticks}")
        print(f"VERDICT: {'PASS' if result else 'FAIL'} ({criterion}, upright_min {_UPRIGHT_MIN})")
        return result

    def _maybe_capture() -> None:
        # Flag-checked first: viser has no native handle to lock and never arms a capture.
        if not capture.requested:
            return
        with viewer.viewer.lock():
            capture.save(viewer.mjm, viewer.mjd, viewer.viewer.cam, viewer.viewer.opt)

    def viewer_policy(_obs):
        nonlocal settle_remaining, tick, ok, graded, stop_reason, pending_route, pending_reaching, pending_grasp_route
        nonlocal pending_capture_ready, pending_energy, prev_current, metrics_due
        nonlocal pending_exec, exec_tick, planner_wait_ticks, post_grade_ticks, vrw_manip_ticks, vrw_duty_ticks
        nonlocal hold_arm_ref
        _maybe_capture()
        if settle_remaining:
            # Reset uses the viewer-owned stepping loop. Hold the planner home reference until its physics
            # transient reaches the standing limit cycle; do NOT call policy.step, which would plan from it.
            settle_remaining -= 1
            harness.drive_base((0.0, 0.0, 0.0))
            return env.reach_action(None)
        # The viewer runs the policy under torch.no_grad(); cuRobo's IK solver is backward-based and needs
        # autograd, so re-enable grad around ReachPolicy.step (the frozen action stays detached).
        # Power belongs to the tick that COMMANDED it, sampled once physics has realized that command --
        # the same "one frame deferred" seam the grading uses. Runs even after ``graded``, so the final
        # mission tick contributes its rectangle and the viewer integrates exactly as many steps as
        # headless does.
        if pending_energy:
            pending_energy = False
            accumulate_energy()
        if metrics_due:
            metrics_due = False
            _report_walk_metrics(harness, policy, ticks=exec_tick,
                                 planner_wait_ticks=planner_wait_ticks,
                                 first_observed_tick=first_observed_tick,
                                 visit_start_tick=visit_start_tick, visit_end_tick=visit_end_tick,
                                 plan_first_tick=plan_first_tick, phase_ticks=phase_ticks,
                                 route_assignments=route_assignments,
                                 vrw_manip_ticks=vrw_manip_ticks, vrw_duty_ticks=vrw_duty_ticks)
        with torch.enable_grad():
            base_pose, proprio, cube = harness.observe()
            if not graded:
                # Headless ``run_reach`` grades only AFTER ``harness.realize``.  The viewer realizes a
                # callback's returned action after this function returns, so grade/count the previous
                # callback here.  In particular, never let the final issued action hit ``--steps`` before
                # physics has realized it.
                if pending_exec:
                    _grade_pending()
                    exec_tick += 1
                for c in cube:
                    first_observed_tick.setdefault(c, exec_tick)
                if walk:
                    done = policy.walk_done
                else:
                    converged = (len(window) == dwell and max(window) < _REACH_SETTLE_TOL_M
                                 and pending_grasp_route is None)
                    done = converged or policy.parked_done
                if done:
                    stop_reason = "walk_done" if walk else (
                        "parked_done" if policy.parked_done else "converged")
                elif exec_tick >= steps_cap:
                    stop_reason = "timeout"
                if done or exec_tick >= steps_cap:
                    ok = _report_walk() if walk else _report_parked()
                    # METRICS2 waits one callback: this tick's power sample is still pending (physics has not
                    # run it yet), and reporting now would drop the final tick from the energy integral.
                    metrics_due = walk
                    graded = True
                else:
                    policy.note_fallen(harness.fallen_cubes())
                    base_twist, arm_ref, cam_cmd = policy.step(
                        base_pose, proprio, cube, harness.tool_poses_base(), harness.physical_grasps())
        if graded:
            # The terminal state has already been graded against the prior realized action.  Returning an
            # unmodified low-level action for this one viewer-owned bookkeeping step cannot alter that
            # verdict, and avoids issuing an ungraded policy command after the cap.
            post_grade_ticks += 1
            if post_grade_ticks == _POST_GRADE_HOLD_TICKS:
                viewer.request_reset()
            # HOLD the last commanded reference, not ``None``: ``None`` means planning-home on both channels
            # (``reach_action``), so it snapped the arm out of the pose RETRACT parked it at -- the visible
            # post-mission flip, viewer-only because headless stops stepping at ``walk_done``.
            return env.reach_action(hold_arm_ref)
        # BaseViewer executes ``policy(obs)`` then ``env.step(returned_action)`` in this SAME control
        # tick. Reuse the headless pre-step seam, then return the low-level arm action; base, camera,
        # gripper, and arm therefore enter one physics step together. Capture grading still samples the
        # following observation, as headless grading does after ``realize``.
        harness.prepare_reach_action(base_twist, cam_cmd, policy.gripper_closed)
        if not graded:
            pending_route = policy.last_route
            pending_reaching = policy.is_reaching
            pending_grasp_route = policy.grasp_route
            pending_capture_ready = policy.walk_phase in {"GRASP", "RETRACT_PLAN", "RETRACT"}
            # Match ``run_reach`` precisely: only the initial async plan-0 wait is outside its control-step
            # budget.  A later replanning hold is a real mission tick in both headless and viewer modes.
            pending_exec = not policy.plan0_pending
            # Stalled ticks are excluded from the energy integral for the same reason they are excluded from
            # ``exec_tick``: their COUNT is (solve wall time / wall time per tick), so leaving their idle-hold
            # power in makes a joule total depend on GPU load and viewer playback speed. Latched here, at the
            # command seam, so it reads the same ``planner_pending`` the tick counter below does.
            pending_energy = accumulate_energy is not None and not getattr(policy, "planner_pending", False)
            # Gated on ``pending_exec`` for the same reason as the headless copy: ``exec_tick`` (and so
            # ``t_sim_s``) counts only these ticks, so charging the phase split on a wider set makes
            # Tloco + Treach exceed the Tbar it decomposes.
            if policy.walk_phase is not None and pending_exec:
                phase_ticks[policy.walk_phase] = phase_ticks.get(policy.walk_phase, 0) + 1
            current = policy.walk_current
            if current != prev_current:
                if current is not None and current not in visit_start_tick:
                    visit_start_tick[current] = exec_tick
                if prev_current is not None and prev_current not in visit_end_tick:
                    visit_end_tick[prev_current] = exec_tick
            prev_current = current
            vrw_route = policy.last_route if policy.last_route is not None else policy.grasp_route
            if policy.is_reaching and vrw_route is not None:
                vrw_manip_ticks += 1
                # ALL scenario targets, not just this visit's assignment: a single-arm visit must still require
                # the OTHER, currently unassigned cube to be visible -- dropping it turns "watch both" into
                # "watch whichever one I'm doing" for any robot/cell that splits.
                if _vrw_tick_ok(policy.walk_phase, policy._extend_settled(), cube, tuple(expected)):
                    vrw_duty_ticks += 1
            if policy.is_reaching and policy.last_route is not None:
                assignment = dict(policy.last_route.assignment)
                if not route_assignments or assignment != route_assignments[-1]:
                    route_assignments.append(assignment)
                if current is not None and current not in plan_first_tick:
                    plan_first_tick[current] = exec_tick
            tick += 1
            if not pending_exec:
                planner_wait_ticks += 1
        hold_arm_ref = arm_ref
        return env.reach_action(arm_ref)

    # The viewer resets via getattr(self.policy, "reset") -- but self.policy is THIS closure, not the
    # ReachPolicy. Reset policy state AND defer its first plan until the viewer has replayed the same home
    # hold. The viewer framework already called ``env.reset()`` before this runs (``base.py
    # reset_environment``); ``harness.reset(seed)`` here re-jitters the pick cubes to the requested seed on
    # first launch, then increments it for each explicit viewer reset, and calls ``env.reset()`` again to
    # apply it -- a second reset is harmless (main-thread action, not nested under env.step), unlike
    # ``harness.settle``, which steps physics in a loop and WOULD nest under the viewer's sim lock.
    # The launch mission already ran at ``base_seed + start`` (the caller's reset), so the FIRST auto-advance
    # must land on the trial after it. Starting at ``start`` instead replays the launch seed -- the
    # pre-existing behaviour, which showed trial 0 twice before reaching trial 1.
    reach_i = _view_start_trial() + 1

    def reset_viewer_policy() -> None:
        nonlocal settle_remaining, tick, graded, pending_route, pending_reaching, pending_grasp_route
        nonlocal pending_capture_ready, pending_energy, pending_exec, prev_current, metrics_due
        nonlocal min_upright, grasp_error, reach_i, exec_tick, planner_wait_ticks, post_grade_ticks, stop_reason
        nonlocal hold_arm_ref
        seed = base_seed + reach_i
        reach_i += 1
        harness.reset(seed)
        print(f"seed {seed}: jittered pick XY (--seed {seed} to reproduce)")
        policy.reset()
        settle_remaining = settle_steps
        tick = 0
        exec_tick = 0                   # per-trial like ``tick``: metrics must not accumulate across resets
        planner_wait_ticks = 0
        stop_reason = ""
        graded = False
        post_grade_ticks = 0
        hold_arm_ref = None             # next trial starts from the planning-home hold, not the old park
        pending_route = None
        pending_reaching = False
        pending_grasp_route = None
        pending_capture_ready = False
        pending_exec = False
        pending_energy = False
        metrics_due = False
        prev_current = None
        visit_min.clear()
        capture_error.clear()
        first_observed_tick.clear()
        visit_start_tick.clear()
        visit_end_tick.clear()
        plan_first_tick.clear()
        phase_ticks.clear()
        route_assignments.clear()
        min_upright = 1.0
        window.clear()
        grasp_error = None

    viewer_policy.reset = reset_viewer_policy
    # The Blender viewer only sees this closure, which owns no route; point it at the policy that does
    # (read by ``BlenderViewer._maybe_send_route`` under MJ_ROUTE_OVERLAY=1).
    viewer_policy.viz_source = policy
    # Same planner-target visualization seam as the kinematic viewer: draw the reachable grasp triads via
    # the mjlab DebugVisualizer (``update_visualizers`` hook reads ``policy.last_route``).
    env.set_debug_policy(policy, robot_cfg=harness.cfg)

    frame_rate = 1.0 / harness.control_dt          # real-time playback at the control cadence
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    print(f"[view] {'native' if has_display else 'viser'} viewer @ {frame_rate:.0f} FPS "
          f"(plan-0 solves off-loop; the robot balances in place, then reaches once the plan lands)")
    if viewer_kind == "blender":
        # Scene-only like the kinematic Blender path: no overlays, no 'o' capture, no FOV/window
        # hooks below (they all reach for a MuJoCo passive handle this viewer does not own).
        from photoreal.bridge import BlenderViewer
        from tasks.visual_manipulation.curobo_reach_harness import apply_view_options

        blender_opt = mujoco.MjvOption()
        apply_view_options(blender_opt, show_fov=getattr(harness, "camera", False))
        # Same visibility and MuJoCo light levels as every other view of this scene; the bridge's
        # photoreal grade drops the headlight, which this scene is mostly lit by.
        viewer = BlenderViewer(env, viewer_policy, frame_rate=frame_rate,
                               geomgroup=blender_opt.geomgroup, mujoco_look=True)
        print("  [view] Blender photoreal viewer (scene only -- no overlays/screenshot capture)")
    elif has_display:
        viewer = NativeMujocoViewer(env, viewer_policy, frame_rate=frame_rate, enable_perturbations=False,
                                     key_callback=capture.on_key)
        print(f"  [view] press 'o' to save the current view to {_CAPTURE_DIR}")
    else:
        viewer = ViserPlayViewer(env, viewer_policy, frame_rate=frame_rate)
    if speed != 1.0:
        # Same state the viewer's own speed keys drive, set up front. Snap to the supported ladder rather
        # than writing an arbitrary multiplier, so the on-screen speed label stays truthful.
        nearest = min(viewer.SPEED_MULTIPLIERS, key=lambda m: abs(m - speed))
        viewer._speed_index = viewer.SPEED_MULTIPLIERS.index(nearest)
        viewer._time_multiplier = nearest
        print(f"[view] playback speed {nearest}x (physics still steps at {harness.control_dt * 1e3:.0f} ms; "
              f"only the wall-clock pacing changes)")
        if walk:
            print("[view] METRICS2 durations and energy_J are EXEC-tick measured (stalled ticks excluded), "
                  "so they are comparable against 1x runs. plan_wait_s reports the excluded wall latency "
                  "and does move with speed.")
    view_pose = _VIEW_CAMERA_POSE.get(harness.scenario.name)
    if harness.camera or window_size is not None or view_pose is not None:
        # --camera: reveal the baked head-camera FOV frustums on launch, no key press. Both viewers
        # default that group OFF; the option handle only exists after the viewer's setup() builds it, so
        # wrap setup() (base.run calls it before the loop) and apply the shared visibility policy there.
        # Native drives the MuJoCo passive-handle MjvOption (same as pressing key "4"); viser drives the
        # mjviser scene's per-group visibility list read every frame, a different API that has to be set
        # by hand. Same setup() hook resizes the native OS window (window_size), since it runs once the
        # GLFW window is created but before the render loop starts. view_pose (native-only; matches the
        # 'o'-key capture path, which is native-only too) seeds cam.lookat/distance/azimuth/elevation so
        # a scenario in _VIEW_CAMERA_POSE opens pre-framed.
        _base_setup = viewer.setup

        def _setup_hook() -> None:
            _base_setup()
            if window_size is not None:
                resize_native_viewer_window(*window_size)
            native = getattr(viewer, "viewer", None)          # NativeMujocoViewer: MuJoCo passive handle
            if native is not None:
                apply_view_options(native.opt, show_fov=harness.camera)
            if view_pose is not None and native is not None:
                lookat, distance, azimuth, elevation = view_pose
                native.cam.lookat[:] = lookat
                native.cam.distance = distance
                native.cam.azimuth = azimuth
                native.cam.elevation = elevation
            scene = getattr(viewer, "_scene", None)            # ViserPlayViewer: mjviser scene
            if harness.camera and scene is not None:
                scene.geom_groups_visible[FOV_GEOM_GROUP] = True

        viewer.setup = _setup_hook
    viewer.run()
    return ok


def _run_dynamic_reach(robot: str, scenario_name: str, args: Args, walk: bool = False) -> bool:
    """--dynamic driver: build the worker + env, then grade one scenario's seed sweep.

    ``walk`` un-gags the base twist and hosts the walk-search brain (drive-to-reachable per cube, then the
    same reach); the headless grade then runs the multi-visit mission (``_grade_walk_headless``).
    ``--view`` opens the mjlab viewer AND grades the same way (``_run_view``), printing the identical
    verdict once, live, without stopping the viewer.
    """
    from tasks.visual_manipulation.curobo_reach_harness import make_harness, run_reach
    from tasks.visual_manipulation.curobo.scene import SpawnedReachMpcWorker
    from tasks.visual_manipulation.reach_policy import ReachPolicy

    # Create and fully capture the worker graph before constructing Warp's dynamic environment. Concurrent
    # CUDA graph capture and environment initialization can SIGSEGV even headless, so startup serialization
    # is required; only the one-time cold-start overlap is lost. INJECTED (worker=), so it is caller-owned
    # (own_worker stays False) and closed in the finally block below -- NOT by policy.close. Non-MPC keeps
    # the in-process sync session.
    base_seed = _resolved_seed(args.seed)
    mpc_worker = None
    if args.mpc:
        for attempt in range(_MPC_WARM_RETRIES + 1):
            mpc_worker = SpawnedReachMpcWorker(robot, scenario_name)
            try:
                mpc_worker.wait_warmed()
                break
            except KeyboardInterrupt:
                mpc_worker.close(force=True)
                raise
            except (TimeoutError, RuntimeError):
                mpc_worker.close(force=True)
                if attempt == _MPC_WARM_RETRIES:
                    raise
                print(f"[curobo] MPC worker warmup failed; retrying ({attempt + 1}/{_MPC_WARM_RETRIES})",
                      flush=True)
    harness = make_harness(robot, scenario_name, dynamic=True, camera=args.camera, walk=walk,
                           device=args.device, seed=base_seed, checkpoint=args.checkpoint)
    try:
        return _run_dynamic_scenario(harness, mpc_worker, robot, scenario_name, args, walk, base_seed)
    finally:
        if mpc_worker is not None:
            mpc_worker.close()         # injected worker: caller-owned, not closed by policy.close


def _run_dynamic_scenario(harness, mpc_worker, robot: str, scenario_name: str, args: Args, walk: bool,
                          base_seed: int) -> bool:
    """Grade the seed sweep on an already-built harness/worker."""
    from tasks.visual_manipulation.curobo_reach_harness import run_reach
    from tasks.visual_manipulation.reach_policy import ReachPolicy

    cfg = _robot_scene(robot, scenario_name)[0]
    # The launch mission does NOT pass through the viewer's reset hook: mjlab's ``BaseViewer.run`` calls
    # ``setup()`` only, and ``reset_environment`` (which invokes it) fires just on ViewerAction.RESET, i.e.
    # the post-verdict auto-advance. So the start-trial offset has to be applied to THIS reset as well, or
    # ``--view`` opens on the sweep's trial 0 whatever the offset says.
    start_trial = _view_start_trial() if args.view else 0
    harness.reset(base_seed + start_trial)
    # MPC path: session=None, the sole cuRobo lives in the spawned worker (the policy stays cuRobo-free
    # in-process). Non-MPC path keeps the sync in-process session. ``walk`` un-gags the base twist and hosts
    # the walk-search brain (drive-to-reachable, then the SAME reach dispatch); parked leaves it at zero.
    if args.mpc:
        policy = ReachPolicy(None, harness.scenario, harness.mj_model, control_dt=harness.control_dt,
                             mpc_track=True, robot_name=robot, scenario_name=scenario_name,
                             worker=mpc_worker,
                             gravity_comp=args.grav_comp,
                             gravity_comp_alpha=(cfg.dynamic_gravity_comp_alpha
                                                 if args.grav_alpha is None else args.grav_alpha),
                             output_filter=args.filter or cfg.dynamic_tracker_filter,
                             traj_omega=cfg.dynamic_tracker_omega if args.omega is None else args.omega,
                             max_tracker_correction_rad=cfg.dynamic_tracker_max_correction_rad,
                             walk=walk, walk_device=args.device)
    else:
        policy = ReachPolicy(harness.session, harness.scenario, harness.mj_model,
                             control_dt=harness.control_dt, replan_interval_s=args.replan_interval_s,
                             robot_name=robot, scenario_name=scenario_name,
                             walk=walk, walk_device=args.device)
    # Name the missing gravity compensation in the banner, not only in --help: an open-loop run looks
    # healthy right up to the grasp it drops.
    mode = ("mpc" if args.mpc else
            ("once" if args.replan_interval_s is None else f"{args.replan_interval_s}s")
            + " OPEN-LOOP (no gravity comp)")
    settle_steps = int(round(args.settle_s / harness.control_dt))
    print(f"dynamic {'walk-' if walk else ''}reach: robot={robot} scenario={scenario_name!r} "
          f"table_z={harness.table_z:.3f} control_dt={harness.control_dt:.3f}s mode={mode} "
          f"preplan_hold={settle_steps} control steps ({args.settle_s:.2f}s)", flush=True)

    print(f"seed {base_seed + start_trial}: jittered pick XY "
          f"(--seed {base_seed} to reproduce)")
    harness.settle(settle_steps)
    if args.view:
        cap = (args.steps if args.steps != _REACH_STEPS else _WALK_STEP_CAP) if walk else args.steps
        dwell = min(_REACH_DWELL, args.steps)
        try:
            ok = _run_view(harness, policy, settle_steps, walk=walk, steps_cap=cap, dwell=dwell,
                          base_seed=base_seed, window_size=_parse_window_size(args.window_size),
                          speed=args.speed, viewer_kind=args.viewer)
        finally:
            policy.close()
        return ok
    if walk:
        # A full walk mission (walk + reach + park, per cube) needs far more than the parked default; use the
        # mission's length (2200) unless the user overrode --steps. Ends early on walk_done regardless.
        cap = args.steps if args.steps != _REACH_STEPS else _WALK_STEP_CAP
        seed_tag = ""
        if args.seed is not None:
            last = args.seed + args.record_trials - 1                  # sweep: one MP4 spans seeds SEED..last
            seed_tag = f"_seed{args.seed}" + (f"-{last}" if args.record_trials > 1 else "")
        record_path = (os.path.join(args.record, f"{robot}_{scenario_name}{seed_tag}_walk.mp4")
                       if args.record else None)
        # No seed_tag on a sweep: _run_walk_trials appends the per-trial seed itself (one PNG set per seed).
        keyframe_prefix = (os.path.join(args.record_keyframes,
                                        f"{robot}_{scenario_name}" + (seed_tag if args.record_trials <= 1 else ""))
                           if args.record_keyframes else None)
        try:
            return _run_walk_trials(harness, policy, cap, args, robot, scenario_name,
                                    settle_steps=settle_steps, record_path=record_path,
                                    keyframe_prefix=keyframe_prefix, keyframe_every=args.keyframe_every)
        finally:
            policy.close()
    dwell = min(_REACH_DWELL, args.steps)      # dwell <= timeout (converge-early contract)

    def _one_reach() -> bool:
        """Grade ONE parked reach from the harness's current (already reset + settled) state."""
        err = run_reach(harness, policy, args.steps, verbose=True, stop_tol=_REACH_SETTLE_TOL_M, dwell=dwell)
        up = harness.upright()
        ok = err < _REACH_TOL_M and up > _UPRIGHT_MIN
        plans = policy._executor.plan_id if policy._executor is not None else 0
        planned = policy.last_route.reach_error if policy.last_route is not None else float("nan")
        print(f"final: realized reach_err {err:.4f} m (planned {planned:.4f} m), upright {up:.3f}, "
              f"plans {plans}, replan_failures {policy.replan_failures}")
        print(f"VERDICT: {'PASS' if ok else 'FAIL'} (reach_tol {_REACH_TOL_M} m, upright_min {_UPRIGHT_MIN})")
        return ok

    try:
        if args.record_trials <= 1:
            return _one_reach()
        # In-process seed sweep: the physics env + frozen policy + warm cuRobo are built once, so trial 2..N
        # skip the ~8.4 s cold start a per-seed subprocess pays. Trial 1 reuses the reset/settle done above.
        successes = int(_one_reach())
        for trial in range(1, args.record_trials):
            harness.reset(base_seed + trial)
            harness.settle(settle_steps)     # re-settle the post-reset physics transient
            policy.reset()
            successes += _one_reach()
        P = successes / float(args.record_trials)
        print(f"METRICS: robot={robot:<10} scenario={scenario_name:<35} P={P:.3f} "
              f"Tbar={args.steps * harness.control_dt:.2f}s success={successes}/{args.record_trials}",
              flush=True)
        return P > 0.0
    finally:
        policy.close()


def _write_keyframe(prefix: str, frame, idx: int, phase, prev_phase, cube, t_s: float, t_exec_s: float) -> dict:
    """Write one FSM-transition PNG and return its manifest row.

    ``t_s`` is wall-of-mission sim time (every tick); ``t_exec_s`` excludes ticks stalled on an off-loop
    cuRobo solve, matching what ``METRICS2``/``PHASES`` report -- a figure caption must be able to quote the
    same clock the tables do."""
    import imageio.v2 as imageio

    path = f"{prefix}_k{idx:02d}_{phase}_t{t_s:06.2f}.png"
    imageio.imwrite(path, frame)
    return {"idx": idx, "phase": phase, "prev_phase": prev_phase, "cube": cube,
            "t_s": round(t_s, 3), "t_exec_s": round(t_exec_s, 3), "file": os.path.basename(path)}


def _walk_cam_ctx(harness, scenario_name: str):
    """Open the offscreen chase-cam state: ``(opt, cam, scratch)``.

    Split from ``_walk_render_ctx`` because the Blender recorder wants the identical camera easing and
    the identical scratch ``MjData`` but renders in another process, so it must not allocate a
    ``mujoco.Renderer`` (a GPU context it would never draw with)."""
    from tasks.visual_manipulation.curobo_reach_harness import apply_view_options
    opt = mujoco.MjvOption()
    # Same visibility the live viewer would show for this run, so the MP4 matches the window (WYSIWYG).
    # This DID hard-disable the frustums: at this chase cam's 1.9 m their edges rake across the grasp
    # region and can hide the 60 mm cube the video exists to show. If that bites, pass show_fov=False.
    apply_view_options(opt, show_fov=getattr(harness, "camera", False))
    cam = mujoco.MjvCamera()
    cam.distance = RECORD_WALK_CAMERA_DISTANCE_WIDE   # seed the ramp; missions open in a wide phase
    if scenario_name in ("shelf", "shelf_pick_both"):
        cam.azimuth, cam.elevation = _record_camera_pose(scenario_name)   # shelf tiers need their side view
    else:
        cam.azimuth = _walk_cam_yaw_offset()                               # eased to live base yaw per tick
        cam.elevation = RECORD_WALK_CAMERA_ELEVATION_DEG
    cam.lookat[:] = [0.0, 0.0, _walk_cam_lookat_z()]       # seed the lookat filter; base starts near origin
    scratch = mujoco.MjData(harness.mj_model)
    return opt, cam, scratch


def _walk_render_ctx(harness, scenario_name: str):
    """Open the offscreen 720p chase-cam render context: ``(renderer, opt, cam, scratch)``.

    Shared by the MP4 recorder and the ``--record-keyframes`` PNG path so the video and the figure frames
    are the SAME view of the SAME rollout -- a keyframe with its own camera could not be cross-checked
    against the clip it came from."""
    # ``mujoco.Renderer`` refuses a size above the model's offscreen framebuffer, and the walk harness compiles
    # its scene without setting one. Raise it here (visual-only field) rather than in every scene builder.
    harness.mj_model.vis.global_.offwidth = max(harness.mj_model.vis.global_.offwidth, RECORD_W)
    harness.mj_model.vis.global_.offheight = max(harness.mj_model.vis.global_.offheight, RECORD_H)
    renderer = mujoco.Renderer(harness.mj_model, height=RECORD_H, width=RECORD_W)
    opt, cam, scratch = _walk_cam_ctx(harness, scenario_name)
    return renderer, opt, cam, scratch


def _open_walk_recorder(harness, scenario_name: str, record_path: str, viewer_kind: str = "native"):
    """Open the offscreen chase-cam recorder for walk rollouts:
    ``(writer, renderer, opt, cam, scratch, blender)``.

    Split out of ``_grade_walk_headless`` so a ``--record-trials N`` sweep can hold ONE writer across all its
    trials and emit a single continuous MP4 (seed rollouts back to back). Previously recording was
    single-trial only, because a per-trial writer would have overwritten the same file N times.

    ``viewer_kind="blender"`` swaps the renderer only: the same chase cam over the same graded rollout is
    dumped frame by frame and rendered offscreen in Blender at ``close``, writing the SAME MP4 path. It
    cannot render inline -- EEVEE needs ~0.16 s a frame against the rollout's ``control_dt`` -- so ``writer``
    and ``renderer`` stay None and ``blender`` owns the output (see ``photoreal.bridge.BlenderRecorder``)."""
    import imageio.v2 as imageio

    fps = max(1, round(1.0 / harness.control_dt))
    os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
    if viewer_kind == "blender":
        from photoreal.bridge import BlenderRecorder

        opt, cam, scratch = _walk_cam_ctx(harness, scenario_name)
        # Pose the scratch once before the export: it seeds the scene file's rest pose, which is what a
        # still render falls back to when no frame has streamed yet.
        scratch.qpos[:] = harness.live_qpos()
        mujoco.mj_forward(harness.mj_model, scratch)
        # ``opt`` and the record size go across too: the two recorders must frame the same geoms at the
        # same resolution or their MP4s cannot be compared. Without ``opt`` the bridge shows the FOV hull
        # unconditionally and its 5 m emissive shells fill most of the frame.
        blender = BlenderRecorder(harness.mj_model, scratch, cam, record_path, fps,
                                  opt=opt, size=(RECORD_W, RECORD_H))
        return None, None, opt, cam, scratch, blender
    renderer, opt, cam, scratch = _walk_render_ctx(harness, scenario_name)
    writer = imageio.get_writer(record_path, fps=fps)
    return writer, renderer, opt, cam, scratch, None


def _run_walk_trials(harness, policy, cap: int, args: Args, robot: str, scenario_name: str, *,
                     settle_steps: int = 0, record_path: str | None = None,
                     keyframe_prefix: str | None = None, keyframe_every: float = 0.0) -> bool:
    """Headless walk grade: ONE graded rollout, or (``--record-trials N``) a seeded sweep reporting the
    aggregate ``METRICS``/``METRICS2`` lines. Returns the pass verdict (sweep: P > 0).

    Shared by the kinematic (``_run_kinematic_walk``) and dynamic (``_run_dynamic_reach``) paths so one
    sweep cannot be scored two ways. Only ``settle_steps`` differs: physics must re-enter its standing
    limit cycle after each reset, while ``mj_forward`` snaps (kinematic passes 0). With ``record_path`` a
    sweep writes ONE continuous MP4 holding every trial's rollout in seed order (shared writer), so a
    per-condition video shows all N seeds without a concat step."""
    if args.record_trials <= 1:
        ok, _m = _grade_walk_headless(harness, policy, cap, scenario_name=scenario_name,
                                      record_path=record_path, keyframe_prefix=keyframe_prefix,
                                      keyframe_every=keyframe_every, viewer_kind=args.viewer)
        return ok
    base_seed = args.seed if args.seed is not None else 42
    recorder = (_open_walk_recorder(harness, scenario_name, record_path, args.viewer)
                if record_path else None)
    successful = []
    for trial in range(args.record_trials):
        harness.reset(base_seed + trial)
        if settle_steps:
            harness.settle(settle_steps)
        policy.reset()
        # Per-seed keyframe prefix: a sweep's trials must not overwrite each other's PNGs/manifest, and the
        # figure wants whichever seed PASSed.
        seed_prefix = None if keyframe_prefix is None else f"{keyframe_prefix}_seed{base_seed + trial}"
        ok, m = _grade_walk_headless(harness, policy, cap, scenario_name=scenario_name, recorder=recorder,
                                     keyframe_prefix=seed_prefix, keyframe_every=keyframe_every,
                                     viewer_kind=args.viewer)
        if ok:
            successful.append(m)
    if recorder is not None and recorder[5] is not None:
        recorder[5].close()                    # Blender: renders every trial's frames, then muxes + prints
    elif recorder is not None:
        recorder[0].close()
        recorder[1].close()
        print(f"[record] wrote {record_path} ({args.record_trials} seeds, "
              f"{round(1.0 / harness.control_dt)} FPS)", flush=True)
    P = len(successful) / float(args.record_trials)
    Tbar = (sum(m["t_sim_s"] for m in successful) / len(successful)) if successful else 0.0
    print(f"METRICS: robot={robot:<10} scenario={scenario_name:<35} P={P:.3f} Tbar={Tbar:.2f}s "
          f"success={len(successful)}/{args.record_trials}", flush=True)
    _print_metrics2_aggregate(successful)
    return P > 0.0


def _grade_walk_headless(harness, policy, step_cap: int, scenario_name: str = "",
                         record_path: str | None = None, recorder=None,
                         keyframe_prefix: str | None = None,
                         keyframe_every: float = 0.0, viewer_kind: str = "native") -> tuple[bool, dict]:
    """Headless walk-mission grade: drive ``policy`` (walk-search + reach) on ``harness`` until every visit is
    done (``walk_done``) or ``step_cap`` ticks, tracking the BEST realized reach error per visit (the arm's
    limit cycle dips then drifts, so the min over the REACH window is the fair 'did it land' read -- same
    spirit as the mission's ``reach_min``). PASS = every EXPECTED pick cube (``harness.scenario.pick``) got
    serviced AND reached < ``_REACH_TOL_M`` AND upright held. Grading against the ground-truth pick set (not
    just whatever landed in ``visit_min``) is deliberate: a cube never discovered, or one that entered REACH
    but got no cuRobo route (``last_route is None``), never records a ``visit_min`` entry -- the old
    ``all(reached.values())`` was then vacuously true over the serviced subset and PASSed a dropped cube.
    PASS also requires the policy's ``SUCCESS`` terminal verdict.  In particular, every physical grasp must
    finish its collision-aware return-home route before terminal completion; a latch records capture but is
    not mission completion.  The step budget doubles as a safety cap; a real mission ends on ``walk_done``
    well inside it.

    ``keyframe_prefix``: when set, one PNG per ``walk_phase`` transition of THIS graded rollout is written as
    ``<prefix>_k<idx>_<phase>_t<sim_s>.png``, plus a ``<prefix>_keyframes.json`` manifest (phase, prior phase,
    visit cube, sim time, mission verdict). Same chase cam and same rollout as the MP4 -- the frames ARE the
    graded run, not a replay. Combines with ``record_path``/``recorder``. ``keyframe_every`` > 0 additionally
    samples every that-many sim seconds inside a phase, so a figure script can pick a mid-reach or mid-walk
    frame (no transition lands there).

    ``record_path``: when set, an offscreen 720p MP4 of THIS graded rollout is written there (a chase cam
    follows the base through the walk). The video is the SAME physics rollout being graded -- no second run.
    The renderer reads the live warp qpos (``harness.live_qpos``) into a scratch ``MjData`` each tick, so the
    frame shows the real simulated pose, not a replan replay. ``recorder``: an already-open recorder owned by
    the caller (``_run_walk_trials``'s multi-seed sweep), appended to and left open so N trials land in one
    continuous MP4; mutually exclusive with ``record_path``.

    Returns ``(ok, metrics)``. ``metrics`` covers 4 diagnostics beyond pass/fail, all keyed purely off
    ``policy.walk_current`` transitions (no phase-string literals): ``energy_j`` (mission total mechanical
    power integral, 0.0 on the kinematic harness -- no physics); ``avg_t_observe_s`` (per-visit: ticks from
    the PREVIOUS visit's completion, i.e. ``walk_current`` moving off that cube, to this cube's first
    appearance in the observed belief -- clamped >=0; a cube already visible during a prior visit scores 0);
    ``avg_t_plan_s`` (per-visit: ticks from this visit's start, i.e. ``walk_current`` becoming this cube, to
    the first tick ``is_reaching and last_route is not None`` for it -- reuses the same gating condition as
    ``visit_min`` below); ``avg_gap_s`` (per-visit, from visit 2: ticks from the previous visit's completion
    to this visit's start -- 0 when the FSM commits to an already-visible cube the same tick the prior one
    finishes); and ``phase_s`` (whole-mission FSM dwell time, for policy-timing diagnosis). All timing is
    converted through ``control_dt``."""
    own_recorder = recorder is None and record_path is not None
    if own_recorder:
        recorder = _open_walk_recorder(harness, scenario_name, record_path, viewer_kind)
    writer = renderer = blender = None
    if recorder is not None:
        writer, renderer, opt, cam, scratch, blender = recorder
    elif keyframe_prefix is not None:
        renderer, opt, cam, scratch = _walk_render_ctx(harness, scenario_name)
    if keyframe_prefix is not None and renderer is None:
        # Keyframe PNGs are mjv-drawn (route triads, FOV frustums), so they keep the MuJoCo renderer
        # even when the MP4 is going through Blender. Its own cam/opt/scratch are dropped: the eased
        # ``cam`` above is shared, so the figure frames still match the clip.
        renderer = _walk_render_ctx(harness, scenario_name)[0]
    keyframe_opt = None
    if keyframe_prefix is not None:
        os.makedirs(os.path.dirname(keyframe_prefix) or ".", exist_ok=True)
        # Keyframes carry the FOV frustum and the planned-route overlay in EVERY panel. Both were stripped in
        # an earlier version, on the theory that a figure panel wants a clean scene -- wrong, because the
        # figure's claim is that a CAMERA-DRIVEN planner reached these targets, and a panel showing neither
        # the FOV it sensed from nor the path it planned cannot support it. Showing the frustum everywhere is
        # only affordable because ``FOV_LARGE_VIS`` is now False: the outer + stereo shells (which at 1.35 m
        # covered most of the frame) are no longer baked, leaving the tip, which marks the lens without
        # occluding the gripper. An intermediate version gated the frustum to the establishing phases to work
        # around those shells; that gate is gone, since it dimmed the evidence in exactly the manipulation
        # panels the claim rests on.
        from tasks.visual_manipulation.curobo_reach_harness import apply_view_options
        # Own option set, never ``opt``: that one belongs to the MP4 and must not inherit figure choices.
        keyframe_opt = mujoco.MjvOption()
        apply_view_options(keyframe_opt, show_fov=getattr(harness, "camera", False))
    keyframes: list[dict] = []
    last_keyframe_t = -float("inf")
    visit_min: dict[str, float] = {}
    capture_error: dict[str, float] = {}
    min_upright = 1.0
    steps_run = 0
    expected = {m.name for m in harness.scenario.pick}
    grasp_capture_tick: int | None = None
    # Dynamic harness only: the kinematic one has no actuator forces, so its mission energy stays 0.0.
    accumulate_energy = getattr(harness, "accumulate_energy", None)
    first_observed_tick: dict[str, int] = {}
    visit_start_tick: dict[str, int] = {}
    visit_end_tick: dict[str, int] = {}
    plan_first_tick: dict[str, int] = {}
    phase_ticks: dict[str, int] = {}
    route_assignments: list[dict[str, str]] = []
    prev_current: str | None = None
    # Reported mission time is measured in EXEC ticks: total ticks minus those the mission spent stalled on an
    # off-loop cuRobo solve (``policy.planner_pending``). The stall length in ticks is (solve WALL time / wall
    # time per tick), so leaving it in makes a sim-time metric depend on GPU load -- one cell's Tbar measured
    # 61.96 s at 1 sweep per GPU and 56.05 s at 3, same code and seeds. Subtracting it reports what a
    # SYNCHRONOUS planner would have produced (the solve returns inside its tick, costing zero sim time), which
    # is the quantity the paper claims. Physics still steps during the stall and the step cap still counts
    # every tick -- this changes measurement only, never the rollout. ``getattr`` default False keeps the
    # kinematic/sync paths (no async worker, never pending) byte-identical.
    exec_ticks = 0
    planner_wait_ticks = 0
    vrw_manip_ticks = 0
    vrw_duty_ticks = 0
    extend_base_xy = None               # see the knock-witness block below
    _base_probe = {"anchor": None, "prev": None, "drift": 0.0, "speed": 0.0}   # _BASE_PROBE only
    recorded_viz_route = object()
    for _k in range(step_cap):
        steps_run = _k + 1
        base_pose, proprio, observed = harness.observe()
        # Kinematic harness has no live tool feedback (idealized robot); guard like ``run_reach``.
        tool_poses = harness.tool_poses_base() if hasattr(harness, "tool_poses_base") else None
        physical_grasps = harness.physical_grasps() if hasattr(harness, "physical_grasps") else None
        if hasattr(harness, "fallen_cubes"):        # physics only; kinematic cubes cannot be knocked over
            policy.note_fallen(harness.fallen_cubes())
            # Base XY the cuRobo reach was solved from, latched on EXTEND entry. The plan is base-relative, so
            # a knock caused by the base DRIFTING under the live locomotion policy while the arm extends is
            # indistinguishable from a tracker miss unless the drift is reported next to the tool error.
            if policy.walk_phase == "EXTEND":
                if extend_base_xy is None:
                    extend_base_xy = np.asarray(base_pose[0][:2], dtype=float).copy()
            else:
                extend_base_xy = None
            # Attribute the 'cube knocked off support' class: names the geom touching the cube on the tick it
            # first moves, plus the FSM phase (which the harness cannot see). Must run after fallen_cubes so
            # it reuses that call's ``_scratch`` refresh. Prints at most once per cube per mission.
            harness.max_penetration()       # accumulate while _scratch is live; reported once after the loop
            for _w in harness.knock_witness():
                drift = ("n/a" if extend_base_xy is None else
                         f"{np.linalg.norm(np.asarray(base_pose[0][:2], dtype=float) - extend_base_xy):.4f}m")
                tracker = getattr(policy, "_tracker", None)
                errs = "n/a"
                if tracker is not None and tool_poses is not None and hasattr(tracker, "position_errors"):
                    errs = {f: round(e, 4) for f, e in tracker.position_errors(tool_poses).items()}
                print(f"[knock] t={_k * harness.control_dt:.2f}s phase={policy.walk_phase} {_w} "
                      f"base_drift={drift} tool_err={errs}", flush=True)
        prev_phase = policy.walk_phase
        base_twist, arm_ref, cam_cmd = policy.step(base_pose, proprio, observed, tool_poses, physical_grasps)
        # Tool error at the EXTEND -> GRASP handoff. ``extend_settled`` is a pure WALL-CLOCK test, so this is
        # the only place the actual convergence the jaws close on becomes visible. Printed unconditionally
        # because a knock and a failed latch are both hypothesized to start here.
        if _KNOCK_TRACE and prev_phase == "EXTEND" and _k % 25 == 0:
            _tr = getattr(policy, "_tracker", None)
            if _tr is not None and tool_poses is not None and getattr(_tr, "_last_desired_base", None):
                print(f"[trace] t={_k * harness.control_dt:.2f}s tool_err="
                      f"{ {f: round(e, 4) for f, e in _tr.position_errors(tool_poses).items()} }", flush=True)
        if prev_phase == "EXTEND" and policy.walk_phase == "GRASP":
            _tr = getattr(policy, "_tracker", None)
            if _tr is not None and tool_poses is not None and hasattr(_tr, "position_errors"):
                print(f"[handoff] t={_k * harness.control_dt:.2f}s tool_err="
                      f"{ {f: round(e, 4) for f, e in _tr.position_errors(tool_poses).items()} }", flush=True)
        stalled = getattr(policy, "planner_pending", False)
        if stalled:
            planner_wait_ticks += 1
        else:
            exec_ticks += 1
        tick = exec_ticks               # timestamp for every per-visit duration below; frozen while stalled
        for c in observed:
            if c not in first_observed_tick:
                first_observed_tick[c] = tick
        # NOT stalled ticks: ``t_sim_s`` is ``exec_ticks * dt`` and excludes them, so counting them here
        # made the phase split overshoot the total it decomposes -- measured +1.480 s on one g1 trial,
        # exactly its ``plan_wait_s``. The two must be charged on the same clock or Tloco + Treach is not
        # a decomposition of Tbar. Plan waits stay reported separately as ``plan_wait_s``.
        if policy.walk_phase is not None and not stalled:
            phase_ticks[policy.walk_phase] = phase_ticks.get(policy.walk_phase, 0) + 1
        if _BASE_PROBE:
            # Anchor on the FIRST manipulation tick of each visit (the base pose the route was solved at) and
            # keep the worst excursion/speed over the whole span, so a slow creep and a sharp kick are both
            # visible. Speed is finite-differenced from base XY -- the dynamic harness reports pose, and a
            # per-tick difference is the disturbance the arm actually caused, whatever produced it.
            _p = np.asarray(base_pose[0][:2], dtype=float)
            if policy.walk_phase in _MANIP_PHASES:
                if _base_probe["anchor"] is None:
                    _base_probe["anchor"] = _p.copy()
                _base_probe["drift"] = max(_base_probe["drift"],
                                           float(np.linalg.norm(_p - _base_probe["anchor"])))
                if _base_probe["prev"] is not None:
                    _base_probe["speed"] = max(
                        _base_probe["speed"],
                        float(np.linalg.norm(_p - _base_probe["prev"])) / harness.control_dt)
                _base_probe["prev"] = _p.copy()
            else:
                _base_probe["anchor"] = None
                _base_probe["prev"] = None
        harness.realize(base_twist, arm_ref, cam_cmd, gripper_closed=policy.gripper_closed)
        if accumulate_energy is not None and not stalled:
            accumulate_energy(phase=policy.walk_phase)   # device-side integral; read once, after the loop
        current = policy.walk_current
        if current is not None and current != prev_current and current not in visit_start_tick:
            visit_start_tick[current] = tick
        if prev_current is not None and current != prev_current and prev_current not in visit_end_tick:
            visit_end_tick[prev_current] = tick
        prev_current = current
        vrw_route = policy.last_route if policy.last_route is not None else policy.grasp_route
        if policy.is_reaching and vrw_route is not None:
            vrw_manip_ticks += 1
            # ALL scenario targets, not just this visit's assignment -- see the viewer-closure copy of this
            # comment for why (a single-arm visit must still require the OTHER cube visible).
            if _vrw_tick_ok(policy.walk_phase, policy._extend_settled(), observed, tuple(expected)):
                vrw_duty_ticks += 1
        if policy.is_reaching and policy.last_route is not None:
            assignment = dict(policy.last_route.assignment)
            if not route_assignments or assignment != route_assignments[-1]:
                route_assignments.append(assignment)
            if current is not None and current not in plan_first_tick:
                plan_first_tick[current] = tick
            # Kinematic mode proves route tracking throughout EXTEND. Dynamic mode may only score after
            # contact capture: a free cube can move after planning, so its stale route waypoint is diagnostic
            # only. Both modes attribute by route.assignment, never the driven walk_current.
            if not hasattr(harness, "grasp_capture_error"):
                err = harness.verdict(policy.last_route)
                for cube in policy.last_route.assignment.values():
                    visit_min[cube] = min(visit_min.get(cube, float("inf")), err)
        if (policy.walk_phase in {"GRASP", "RETRACT_PLAN", "RETRACT", "TERMINATE"}
                and policy.grasp_route is not None):
            capture_errors = getattr(harness, "grasp_capture_errors", None)
            if capture_errors is not None:
                for cube, diagnostic in capture_errors(policy.grasp_route).items():
                    visit_min[cube] = 0.0
                    capture_error[cube] = min(capture_error.get(cube, float("inf")), diagnostic)
            else:
                score, diagnostic = _capture_result(harness, policy.grasp_route)
                if np.isfinite(score):
                    for cube in policy.grasp_route.assignment.values():
                        visit_min[cube] = min(visit_min.get(cube, float("inf")), score)
                        capture_error[cube] = min(capture_error.get(cube, float("inf")), diagnostic)
        # A target is physically captured only after GRASP's full jaw-close hold observes its latch.  Keep
        # stepping through RETRACT: return-home is part of every successful mission, including final capture.
        if set(policy.completed_grasps.values()) == expected:
            grasp_capture_tick = _k + 1
        min_upright = min(min_upright, harness.upright())
        if renderer is not None or blender is not None:
            scratch.qpos[:] = harness.live_qpos()
            mujoco.mj_forward(harness.mj_model, scratch)
            from tasks.visual_manipulation.curobo_reach_harness import (
                draw_nav_arrow, draw_target_axes, draw_triads, route_path_triads, cube_world_poses)
            cube_poses = cube_world_poses(harness.mj_model, harness.scenario, scratch)
            target = cube_poses.get(policy.walk_current)
            # Camera filter is eased EVERY tick even when only keyframes are wanted: it lerps 4% per tick, so
            # tracking it only on transitions would leave each keyframe framed by the previous phase's pose.
            _walk_cam_track(cam, base_pose, None if target is None else target[0], policy.walk_phase)
            sim_t = steps_run * harness.control_dt
            keyframe_due = keyframe_prefix is not None and (
                policy.walk_phase != prev_phase
                or (keyframe_every > 0.0 and sim_t - last_keyframe_t >= keyframe_every))
            if blender is not None:
                route = policy.viz_route
                if route is not recorded_viz_route:
                    points = None if route is None else np.asarray(
                        [p[0] for p in route_path_triads(harness.mj_model, route)], dtype=np.float32
                    ).reshape(-1, len(route.reaches), 3).transpose(1, 0, 2)
                    blender.append(scratch, cam, points)
                    recorded_viz_route = route
                else:
                    blender.append(scratch, cam)
            elif writer is not None:
                renderer.update_scene(scratch, camera=cam, scene_option=opt)
                if hasattr(policy, "nav_target_w") and hasattr(policy, "walk_phase"):
                    draw_nav_arrow(renderer.scene, base_pose[0], policy.nav_target_w, policy.walk_phase)
                cube_frames = [(pos, np.eye(3)) for pos, _dims in cube_poses.values()]
                draw_target_axes(renderer.scene, cube_frames)
                writer.append_data(renderer.render())
            if keyframe_due:
                renderer.update_scene(scratch, camera=cam, scene_option=keyframe_opt)
                # Planned route path only (no grasp triads): the figure claims a planner drove the reach, and
                # a panel with no visible plan cannot show that. Recomputed per keyframe rather than cached by
                # route identity -- keyframes are ~2 Hz, so the ~40 FK calls are free here, unlike the
                # every-tick viewer path that needed the cache.
                route = policy.last_route if policy.last_route is not None else policy.grasp_route
                if route is not None:
                    draw_triads(renderer.scene, route_path_triads(harness.mj_model, route,
                                                                  scale=KEYFRAME_ROUTE_MARKER_SCALE))
                keyframes.append(_write_keyframe(keyframe_prefix, renderer.render(), len(keyframes),
                                                 policy.walk_phase, prev_phase, policy.walk_current,
                                                 sim_t, tick * harness.control_dt))
                last_keyframe_t = sim_t
        if policy.walk_done:
            break
    if own_recorder and blender is not None:
        blender.close()                                         # renders offscreen, then muxes + prints
    elif own_recorder:
        writer.close()
        print(f"[record] wrote {record_path} ({steps_run} frames @ {round(1.0 / harness.control_dt)} FPS)",
              flush=True)
    elif writer is not None or blender is not None:
        print(f"[record] appended {steps_run} frames", flush=True)
    if renderer is not None and (own_recorder or writer is None):
        renderer.close()                    # ours: the recorder we own, or the keyframe-only context
    missing = expected - set(visit_min)                       # never discovered, or entered REACH with no cuRobo route
    reached = {c: (e < _REACH_TOL_M) for c, e in visit_min.items()}
    verdict = policy.mission_verdict                             # policy SUCCESS means all grasps returned home
    ok = (set(visit_min) == expected and all(reached[c] for c in expected)
          and min_upright > _UPRIGHT_MIN and verdict == ("SUCCESS", None))
    print(f"walk mission: {steps_run} steps, visits={ {c: round(e, 4) for c, e in visit_min.items()} } m, "
          f"expected={sorted(expected)}, missing={sorted(missing)}, min_upright {min_upright:.3f}, "
          f"done={policy.walk_done}, grasp_complete={grasp_capture_tick is not None}")
    if capture_error:
        print(f"capture diagnostic: { {c: round(e, 4) for c, e in capture_error.items()} } m")
    print(f"mission_verdict: {verdict}")
    criterion = ("completed physical latch" if hasattr(harness, "grasp_capture_error")
                 else f"each visit reach_tol {_REACH_TOL_M} m")
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} ({criterion}, upright_min {_UPRIGHT_MIN})")
    if keyframe_prefix is not None:
        # Manifest carries the verdict so a figure script can drop a FAILed seed's frames without re-parsing
        # stdout -- the run is only usable as a "successful task" panel if this says pass.
        manifest = f"{keyframe_prefix}_keyframes.json"
        with open(manifest, "w") as f:
            json.dump({"scenario": scenario_name, "pass": ok, "verdict": list(verdict),
                       "t_sim_s": round(steps_run * harness.control_dt, 3),
                       "keyframes": keyframes}, f, indent=2)
        print(f"[keyframes] wrote {len(keyframes)} PNGs + {manifest} (pass={ok})", flush=True)

    if _BASE_PROBE:
        print(f"BASE: max_drift_m={_base_probe['drift']:.5f} max_speed_mps={_base_probe['speed']:.5f}")
    metrics = _report_walk_metrics(harness, policy, ticks=exec_ticks,
                                   planner_wait_ticks=planner_wait_ticks,
                                   first_observed_tick=first_observed_tick,
                                   visit_start_tick=visit_start_tick, visit_end_tick=visit_end_tick,
                                   plan_first_tick=plan_first_tick, phase_ticks=phase_ticks,
                                   route_assignments=route_assignments,
                                   vrw_manip_ticks=vrw_manip_ticks, vrw_duty_ticks=vrw_duty_ticks)
    return ok, metrics


def _report_walk_metrics(harness, policy, *, ticks: int, first_observed_tick: dict[str, int],
                         visit_start_tick: dict[str, int], visit_end_tick: dict[str, int],
                         plan_first_tick: dict[str, int], phase_ticks: dict[str, int],
                         route_assignments: list[dict[str, str]], planner_wait_ticks: int = 0,
                         vrw_manip_ticks: int = 0, vrw_duty_ticks: int = 0) -> dict:
    """Print the METRICS2/PHASES/GATE_WAIT/ROUTES block and return the metrics dict.

    Shared by the headless grade (``_grade_walk_headless``) and the viewer grade (``_run_view``) so one
    mission cannot be scored by two different definitions of a visit. Callers own the tick bookkeeping
    (they observe the mission at different points in the tick: headless realizes physics itself, the
    viewer realizes it after the policy callback returns); this owns the chained per-visit arithmetic.

    Every duration here is SIM time -- tick counts scaled by ``control_dt``. Callers pass EXEC ticks (total
    minus ticks stalled on an off-loop cuRobo solve), so no duration here carries the async worker's wall
    latency: the numbers describe what a synchronous planner, returning inside its own tick, would have
    produced. That makes them invariant to GPU load, concurrent sweeps, and viewer playback speed
    (``Args.speed``) -- previously ``avg_t_plan_s`` and ``t_sim_s`` all shifted with those. ``planner_wait_ticks``
    is the excluded amount, reported as ``plan_wait_s`` so the subtraction stays auditable rather than silent.

    ``energy_J`` obeys the same contract: callers skip ``accumulate_energy`` on stalled ticks, so the total is
    the joules a synchronous planner would have burned. The robot is not free while stalled -- it holds a
    standing pose and its actuators fight gravity -- so integrating those ticks would have made the joule
    total scale with the async worker's wall latency exactly as the durations once did.
    """
    # Chained per-visit timing (order = commit order, by visit_start_tick).
    order = sorted(visit_start_tick, key=visit_start_tick.get)
    observe_deltas, plan_deltas, gap_deltas = [], [], []
    for i, c in enumerate(order):
        prev_end = visit_end_tick[order[i - 1]] if i > 0 else 0
        if c in first_observed_tick:
            observe_deltas.append(max(0, first_observed_tick[c] - prev_end))
        if i > 0:
            gap_deltas.append(visit_start_tick[c] - prev_end)
        if c in plan_first_tick:
            plan_deltas.append(plan_first_tick[c] - visit_start_tick[c])
    dt = harness.control_dt
    avg_t_observe_s = (sum(observe_deltas) / len(observe_deltas) * dt) if observe_deltas else 0.0
    avg_t_plan_s = (sum(plan_deltas) / len(plan_deltas) * dt) if plan_deltas else 0.0
    avg_gap_s = (sum(gap_deltas) / len(gap_deltas) * dt) if gap_deltas else 0.0
    # Kinematic harness has no actuator forces, so its mission energy is 0.0 by construction.
    energy_j = harness.mission_energy_j() if hasattr(harness, "mission_energy_j") else 0.0
    # ``t_sim_s`` is printed, not just returned: stdout-parsing sweep drivers otherwise have to reconstruct
    # mission length from the ``walk mission: N steps`` line, whose N is TOTAL ticks (stall included) and so
    # carries the wall latency this function exists to exclude.
    print(f"METRICS2: energy_J={energy_j:.2f} t_sim_s={ticks * dt:.3f} avg_t_observe_s={avg_t_observe_s:.3f} "
          f"avg_t_plan_s={avg_t_plan_s:.3f} avg_gap_s={avg_gap_s:.3f} visits={len(order)} "
          f"plan_wait_s={planner_wait_ticks * dt:.3f}")
    print(f"PHASES: { {phase: round(ticks_in * dt, 3) for phase, ticks_in in phase_ticks.items()} }")
    if hasattr(harness, "mission_phase_energy_j"):
        phase_energy = harness.mission_phase_energy_j()
        print(f"PHASE_ENERGY: { {phase: round(j, 2) for phase, j in phase_energy.items()} }")
    if hasattr(harness, "max_penetration"):
        # Paired collision metric: a tracking change must not buy accuracy by pushing through the scene.
        print(f"PENETRATION: max_m={harness.max_penetration():.5f}")
    print("GATE_WAIT:", {
        "acquire_gaze_s": round(policy._acquire_gaze_wait_ticks * dt, 3),
        "acquire_level_s": round(policy._acquire_level_wait_ticks * dt, 3),
    })
    print(f"ROUTES: {route_assignments}")
    # Executed pairwise-VRW conjunction (plan Decision 2 / build order step 3): fraction of manipulation
    # ticks (EXTEND/GRASP/RETRACT_PLAN/RETRACT) where the tool is at/holding its target AND both co-targeted
    # cubes are camera-visible. NaN (not 0.0) when the mission never reached a manipulation tick at all --
    # 0.0 would silently read as "always blind" rather than "never measured."
    vrw_duty = (vrw_duty_ticks / vrw_manip_ticks) if vrw_manip_ticks else float("nan")
    print(f"VRW_DUTY: {vrw_duty:.3f} ({vrw_duty_ticks}/{vrw_manip_ticks} manipulation ticks)")
    return {
        "energy_j": energy_j,
        "avg_t_observe_s": avg_t_observe_s,
        "avg_t_plan_s": avg_t_plan_s,
        "avg_gap_s": avg_gap_s,
        "visits": len(order),
        "t_sim_s": ticks * dt,
        "plan_wait_s": planner_wait_ticks * dt,
        "phase_s": {phase: ticks_in * dt for phase, ticks_in in phase_ticks.items()},
        "vrw_duty": vrw_duty,
    }


def _print_metrics2_aggregate(metrics_list: list[dict]) -> None:
    """Average diagnostics over successful trials only, matching ``Tbar``'s paper convention.

    Failed/cap-censored trials count in P but never in timing or energy means. Callers append a metric only
    after its physical/kinematic verdict passes, so partial failed visits cannot dilute reported diagnostics.
    """
    usable = [m for m in metrics_list if m["visits"] > 0]
    if not usable:
        print("METRICS2: energy_J=0.00 avg_t_observe_s=0.000 avg_t_plan_s=0.000 avg_gap_s=0.000 trials=0",
              flush=True)
        return
    n = len(usable)
    energy = sum(m["energy_j"] for m in usable) / n
    observe = sum(m["avg_t_observe_s"] for m in usable) / n
    plan = sum(m["avg_t_plan_s"] for m in usable) / n
    gap = sum(m["avg_gap_s"] for m in usable) / n
    print(f"METRICS2: energy_J={energy:.2f} avg_t_observe_s={observe:.3f} avg_t_plan_s={plan:.3f} "
          f"avg_gap_s={gap:.3f} trials={n}", flush=True)
    duties = [m["vrw_duty"] for m in usable if m.get("vrw_duty") == m.get("vrw_duty")]  # drop NaN (never manipulated)
    duty_mean = sum(duties) / len(duties) if duties else float("nan")
    print(f"VRW_DUTY_MEAN: {duty_mean:.3f} (n={len(duties)})", flush=True)


def _run_dynamic(args: Args) -> None:
    """--dynamic dispatch: PARKED or WALK reach on the shared ``ReachPolicy`` + ``DynamicHarness`` (walk
    un-gags the base twist + hosts the walk-search brain, then the SAME reach). Parked ``--record`` delegates
    to the proven ``pickplace_reach_env`` mission's offscreen MP4; ``--walk --record`` renders the graded walk
    rollout offscreen in ``_grade_walk_headless``. Single physics robot; ``os._exit`` with the verdict."""
    if args.drift:
        raise SystemExit("--drift is kinematic-only (under physics the base motion IS the disturbance)")
    if args.robot == "both":
        raise SystemExit("--dynamic runs a single physics robot; pick v2 | v2_fixed | g1, not both")
    scenario_name = _resolve_scenarios(args.scenario, allow_all=False)[0]
    if args.record and not args.walk:
        # Stationary video still delegates to the mission's offscreen MP4 path (unified --record dir is
        # kinematic-only). Import lazily (keeps warp/frozen deps off the kinematic path).
        if args.viewer != "native":
            # The delegated mission owns its own renderer; silently handing back a MuJoCo video from a
            # run that asked for Blender would misreport what produced it.
            raise SystemExit(
                f"--viewer {args.viewer} --record is only wired on the walk path, which renders here; "
                "the parked reach delegates its MP4 to pickplace_reach_env (MuJoCo offscreen). "
                "Use --walk --record for a Blender video.")
        from tasks.visual_manipulation import pickplace_reach_env as pp
        steps = None if args.steps == _REACH_STEPS else args.steps   # let the mission auto-resolve its length
        pp_args = pp.Args(robot=args.robot, scenario=scenario_name, walk=False,
                          render=True, view=args.view, steps=steps)
        pp.main(pp_args)
        sys.stdout.flush()
        os._exit(0)
    ok = _run_dynamic_reach(args.robot, scenario_name, args, walk=args.walk)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def main(args: Args) -> None:
    if args.record_keyframes:
        # Drop the outer + stereo FOV shells for the FIGURE ONLY, before any spec is built (the hull is baked
        # at compile time, so this must precede harness construction). At the figure's 1.35 m chase-cam
        # distance those shells cover most of the frame: invisible at 2560x1440, but they wash the panel out
        # at 1.2 in on the page. The tip and the optical-axis pyramid still ship, so the panel keeps both the
        # lens and its aim direction. Patched here rather than by flipping the module default, which was
        # tried on 2026-08-06 and reverted -- it silently stripped the cone from every ``--view`` session and
        # from the orbit mp4s too.
        from asset_zoo import fov_frustum
        fov_frustum.FOV_LARGE_VIS = False
    if args.selftest:
        if args.dynamic:
            raise SystemExit("--selftest is the kinematic arm-count regression suite; run without --dynamic")
        _run_reach_selftest(args)   # asserts the (robot x camera) arm-count matrix + hold; os._exit
    if args.dynamic:
        _run_dynamic(args)          # physics parked reach OR delegated walk/record; os._exit
    robots = ["g1", "v2_fixed", "v2"] if args.robot == "both" else [args.robot]
    for r in robots:
        if r not in CFG_BY_ROBOT:
            raise SystemExit(f"unknown robot {r!r}; choices: {sorted(CFG_BY_ROBOT)}, both")
    if args.robot == "both" and args.view and args.drift:
        raise SystemExit("--robot both with --drift --view is not supported (single live viewer); run per robot")
    if args.walk and args.drift:
        raise SystemExit("--walk is a far-scene sequential pre-phase; it cannot combine with --drift")

    # One scenario per run, except ``--record-initial all`` which sweeps the snapshot sheet's list. Each
    # compiles its own scene; this path pays no CUDA env, no frozen-policy load and no cuRobo warm, so a
    # per-scenario compile is a fraction of a second.
    scenario_names = _resolve_scenarios(args.scenario, allow_all=bool(args.record_initial))

    ok = True
    for scenario_name in scenario_names:
        for r in robots:
            cfg, object_cfg, _scenario, table_z = _robot_scene(r, scenario_name)
            print(f"\n########## reach [{r}] on {scenario_name!r} (table {table_z:.3f} m) ##########")
            if (args.record or args.record_keyframes) and args.walk:
                # KINEMATIC walk MP4 is not wired (this loop is the mj_forward path). The offscreen walk video
                # lives on the physics path: --dynamic --walk --record DIR renders the graded rollout. Fail
                # loudly here instead of silently producing nothing.
                raise SystemExit(
                    "kinematic --walk --record/--record-keyframes is not wired. For a walk MP4 or keyframe PNGs "
                    "use --dynamic --walk --record DIR / --record-keyframes DIR (offscreen, physics rollout), "
                    "or --walk --view for a live kinematic viewer.")
            elif args.walk:
                res = _run_kinematic_walk(r, scenario_name, args)   # unified walk brain (kinematic twin of --dynamic --walk)
            elif args.record_initial:
                res = _run_record_initial(cfg, object_cfg, args)    # static setup-snapshot sheet (no mission equivalent)
            else:
                res = _run_reach_loop(r, scenario_name, args, expect_arms=args.expect_arms)  # unified reachability-driven loop (default; --view = live)
            ok = res and ok
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main(tyro.cli(Args))
