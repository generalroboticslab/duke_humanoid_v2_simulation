"""Sim harnesses that evaluate ONE transferable ``ReachPolicy`` -- kinematic (phase-3) and, later,
dynamic (phase-4). A harness owns the world; the policy owns the plan. They meet on a fixed loop:

    base_pose, proprio, cube = harness.observe()
    base_twist, arm_ref, cam  = policy.step(base_pose, proprio, cube)   # SAME policy every harness
    harness.realize(base_twist, arm_ref, cam)                          # kinematic | dynamic | hardware
    metric = harness.verdict()                                         # eval-only

``KinematicHarness`` realizes commands on a BARE ``mj_model``/``mj_data`` (``mj_forward`` on written
qpos) -- an idealized robot proving geometric reachability, and deliberately NOT the warp-backed RL env
(kinematic replay on GPU state is the phase-4 risk R1; sidestepped by using plain MuJoCo here). Base is
kinematically integrated from the policy's own ``base_twist`` (planar: yaw + planar translation), so the
same base-live re-solve the dynamic harness needs is exercised without physics.

Frame note: ``observe`` returns the base pose as ``(pos[3], quat_wxyz[4])`` read straight off the root
free joint; ``realize`` writes the reaching-arm reference (already mjlab qpos) onto its joints and the
camera command onto the gimbals. ``verdict`` = realized tool-site error to the route's grasp candidate
(the SAME datum phase-3's gate uses), so a kinematic PASS means the plan is geometrically reachable.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pathlib
import re
import sys

import glfw
import mujoco
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _path in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tasks.visual_manipulation.pickplace_scenarios import (  # noqa: E402
    CUBE_SIZE,
    make_scene_spec_fn,
)
from asset_zoo.parallel_gripper import (  # noqa: E402
    PARALLEL_GRIPPER_CLOSE_STROKE_M,
    PARALLEL_GRIPPER_OPEN_RACK_POS_M,
    PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S,
    parallel_gripper_contact_rack_pos,
)
from asset_zoo.fov_frustum import FOV_GEOM_GROUP  # noqa: E402 (single source for FOV geom group)
from tasks.visual_manipulation.curobo.planner import (  # noqa: E402
    drift_base_quat,
    make_planner_session,
    rot_rpy_deg,
    yaw_quat,
)

# Sentinel: ``harness.session`` is lazy in the dynamic harness, mirroring the kinematic one -- the cold
# cuRobo/CUDA warmup is paid ONLY by the path that uses it (the non-MPC sync path), never by the spawned
# MPC worker (which warms its OWN copy in-child). See ``DynamicHarness.session``.
_PLANNER_SESSION_UNINIT = object()
from tasks.visual_manipulation.curobo.scene import (  # noqa: E402
    _GAZE_SLEW_MAX,
    _ns_id,
    _ns_joint_qadr,
    cube_visible_poses,
    cube_world_poses,
    detect_at_gaze,
    jitter_pick_markers,
    move_pick_geoms,
    pose_relative,
    refresh_warp_los_geoms,
    robot_scene,
)

# The SINGLE ground plane both reach harnesses stand on. WE own it -- mjlab supplies only the RL
# robot/policy, not the floor (the dynamic env sets ``scene.terrain = None``; the kinematic model never
# had a ground). Name + params match mjlab's ``TerrainEntity._import_ground_plane`` EXACTLY (body/geom
# ``terrain``, type PLANE, size (0,0,0.01), MuJoCo defaults for friction/condim/contype/conaffinity): the
# name must stay ``terrain`` because the RL foot-ground contact sensors (``feet_ground_contact_foot_*``)
# reference the ground geom BY THAT NAME, and the byte-identical params keep the frozen policy's contact
# unchanged from today's ``plane_flag`` groundplane.
_GROUND_NAME = "terrain"
# Distinguishable floor pattern (ICRA paper / video aesthetic + manual travel-distance scale). Built-in
# checker texture: zero asset deps, lit correctly under the harness headlight, ``mark="edge"`` paints a
# brighter border on every cell so cells stay readable at chase-cam distance. ``texrepeat`` is in WORLD
# METRES per cell (``texuniform=True``); 0.5 m gives 2-3 cells under the typical 1.9 m chase-cam view --
# fine enough to read sub-metre travel by eye, coarse enough that the lines do not alias. The plane is
# PURELY VISUAL: contype/conaffinity/size/condim unchanged from the original flat plane, so the frozen
# policy's foot contact sensors see the same plane they were trained on.
_FLOOR_GRID_TEX = "reach_floor_grid_tex"
_FLOOR_GRID_MAT = "reach_floor_grid_mat"
_FLOOR_GRID_CELL_M = 0.5            # edge-to-edge; texrepeat=(N,N) below gives N cells per 1 m world
# Ungridded floor colour. A plane with no material takes MuJoCo's default 0.5 grey, which is
# BRIGHTER than the dark-anodized robot -- the robot then reads as a silhouette cut out of its
# own backdrop. 0.16 keeps the floor a plausible dark studio surface while staying above the
# 0.11 shell shade, so the feet and their contact shadow stay separable from the ground.
_FLOOR_RGBA = [0.155, 0.16, 0.17, 1.0]


def _floor_grid_enabled() -> bool:
    """Env-var toggle for the checker floor pattern. Default OFF (original flat grey plane); set
    ``FLOOR_GRID=1`` for the paper/video checker grid. Read at every call so the user can flip it
    per-run without recompiling."""
    return os.environ.get("FLOOR_GRID", "0") in ("1", "true", "True", "yes", "YES")


def add_floor_grid(spec) -> None:
    """Register the checker texture + material the floor plane binds to. Idempotent within one spec
    (skip if already present) so calling twice on the SAME spec is safe -- e.g. a harness re-invokes
    ``add_ground_plane`` after a recompile. No-op when the material already exists."""
    if any(m.name == _FLOOR_GRID_MAT for m in spec.materials):
        return
    tex = spec.add_texture(name=_FLOOR_GRID_TEX)
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    # Darkened 2026-08-12 to sit under the dark-anodized robot (see _FLOOR_RGBA). The pale
    # 0.78/0.30 pair was tuned against the old light-grey links and now outshines them. The
    # cell CONTRAST RATIO is what makes the grid readable as a scale bar, so both shades drop
    # together rather than only the bright one.
    tex.rgb1 = [0.26, 0.27, 0.29]               # lighter cell
    tex.rgb2 = [0.13, 0.145, 0.17]              # darker cell (cell contrast)
    tex.mark = mujoco.mjtMark.mjMARK_EDGE
    tex.markrgb = [0.07, 0.075, 0.09]           # near-black border -> sharp meter lines
    tex.width = 512
    tex.height = 512
    mat = spec.add_material(name=_FLOOR_GRID_MAT)
    mat.texuniform = True
    n = max(1, int(round(1.0 / _FLOOR_GRID_CELL_M)))
    mat.texrepeat = [float(n), float(n)]
    mat.reflectance = 0.0                       # no specular shimmer; flat for paper prints
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _FLOOR_GRID_TEX


def add_ground_plane(spec, *, with_grid: bool = True) -> None:
    """Weld ONE infinite ground plane at z=0 onto ``spec`` (mujoco.MjSpec). Single source of the floor
    for BOTH the kinematic and dynamic reach harnesses -- the only scene element that used to diverge.

    ``with_grid=True`` (default) attaches the checker-pattern material from ``add_floor_grid`` so the
    plane reads as a measured grid in PNG/MP4. ``with_grid=False`` falls back to the original flat
    uniform plane (the "uninteresting grey" option for runs that need an untextured control)."""
    if with_grid:
        add_floor_grid(spec)
    geom = dict(material=_FLOOR_GRID_MAT) if with_grid else dict(rgba=_FLOOR_RGBA)
    spec.worldbody.add_body(name=_GROUND_NAME).add_geom(
        name=_GROUND_NAME,
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0, 0, 0.01),
        **geom,
    )


# The cuRobo planner's whole-body collision-sphere approximation, added as VISUAL-ONLY child geoms on the
# robot link bodies in geom group 5 so MuJoCo's native per-group toggle reveals it on demand (viewer key
# "5"). Group 5 is the rack-ikproxy role (a handful of geoms) and is OFF in the default visibility mask
# (geomgroup [1,1,1,0,0,0]), so the spheres start hidden with no extra setup. contype/conaffinity=0 ->
# pure marker, zero physics; being child geoms of the link bodies, MuJoCo FK moves them with the arm
# automatically -- no per-frame draw code, working identically in the kinematic and dynamic (mjlab) viewers.
PLANNER_SPHERE_GROUP = 5
_PLANNER_SPHERE_RGBA = (0.1, 0.9, 0.9, 0.35)   # translucent cyan


def add_planner_collision_geoms(spec, collision_spheres, ns: str = "") -> None:
    """Add cuRobo's ``collision_spheres`` (``{link_name: [{center[3], radius}]}``, body frame -- the SAME
    dict the planner collides against) as child geoms on their link bodies in ``spec`` (mujoco.MjSpec), in
    group ``PLANNER_SPHERE_GROUP`` as visual markers. Single source for BOTH harnesses' viewers (kinematic
    bare names; dynamic passes ``ns='robot/'`` for mjlab's namespaced bodies). Skips links absent from
    ``spec`` (a coarsened/locked link the spec does not expose). Must run BEFORE ``spec.compile()``."""
    existing = {b.name for b in spec.bodies}
    for link, sl in collision_spheres.items():
        if ns + link not in existing:
            continue
        body = spec.body(ns + link)
        for i, s in enumerate(sl):
            g = body.add_geom()
            g.name = f"planner_sphere_{ns}{link}_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.pos = list(s["center"])
            g.size = [float(s["radius"]), 0.0, 0.0]
            g.group = PLANNER_SPHERE_GROUP
            g.contype = 0
            g.conaffinity = 0
            g.rgba = list(_PLANNER_SPHERE_RGBA)


def add_planner_object_geoms(spec, height_pad: float = 0.02) -> None:
    """Add group 5 (PLANNER_SPHERE_GROUP) visual geoms for pick objects matching height_pad."""
    if height_pad <= 0.0:
        return
    from tasks.visual_manipulation.pickplace_scenarios import CUBE_HALF
    for body in spec.bodies:
        if body.name.startswith("pp_obj_"):
            g = body.add_geom()
            g.name = f"planner_geom_{body.name}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [CUBE_HALF, CUBE_HALF, CUBE_HALF + height_pad / 2.0]
            g.pos = [0.0, 0.0, height_pad / 2.0]
            g.group = PLANNER_SPHERE_GROUP
            g.contype = 0
            g.conaffinity = 0
            g.rgba = list(_PLANNER_SPHERE_RGBA)



# Scripted --drift disturbance (KinematicHarness): a ZERO-MEAN base-drift SINUSOID -- a continuously MOVING
# base (a balancing humanoid's floating-base sway), NOT a plateau. Unlike a ramp (which settles, letting a
# tracker catch up once), the sinusoid never stops, so it stress-tests replan tracking-LATENCY: the target
# moves during each solve. Oscillation is along the measured lean axis (unit of [0.070, 0.075]) so it sways
# the same direction the real base creeps. ``k=0`` at plan time -> ``sin(0)=0`` -> the base is at home for the
# first (clean) plan. Amplitude 0.02 m = a REALISTIC balancing-humanoid floating-base sway (peak speed
# A*omega = 0.02*2pi/2s ~ 0.063 m/s); the earlier 0.05 m (0.157 m/s) was an unrealistically violent lurch.
# ``base_drift_norm`` then oscillates 0..amplitude; the reach residual is graded as the PEAK over a settled
# tail window (see ``run_reach(settle_window=...)``), phase-independent.
_DRIFT_TARGET = np.array([0.070, 0.075, 0.0])      # measured plan-once base creep (world XY) -- sway AXIS
_DRIFT_DIR = _DRIFT_TARGET / np.linalg.norm(_DRIFT_TARGET)   # unit sway direction (XY)
_DRIFT_AMP_M = 0.02                                # zero-mean sway amplitude (peak displacement), metres
_DRIFT_PITCH_DEG = 1.0                             # pitch-lean oscillation amplitude (exercises _yaw_only)
_DRIFT_PERIOD_STEPS = 150                          # sinusoid period (ticks); slower = gentler sway (0.042 m/s)

# Kinematic grasp attachment is deliberately more permissive than the dynamic weld latch: this harness
# represents an optimistic geometric upper bound, so jaws attach the nearest free target after reaching their
# modeled contact position. Dynamic execution additionally requires real finger contact.
_KINEMATIC_GRASP_ATTACH_DIST_M = 0.05


def _apply_home_pose(mj_model, d, robot_cfg) -> None:
    """Write the standing home pose into ``d.qpos``, regex-expanding keyframe joint PATTERNS (g1's
    ``.*_knee_joint`` bent-leg keys) then overriding the arm with ``planning_home_joint_pos``. Same
    logic as phase-3 verify's ``_apply_home_pose`` (kept local; that module is a script, not a lib)."""
    joint_names = [mj_model.joint(j).name for j in range(mj_model.njnt)]
    for pattern, value in robot_cfg.home_joint_pos.items():
        for jname in joint_names:
            if re.fullmatch(pattern, jname):
                d.qpos[mj_model.joint(jname).qposadr[0]] = value
    for jname, value in robot_cfg.planning_home_joint_pos.items():
        d.qpos[mj_model.joint(jname).qposadr[0]] = value


def _quat_rot(q, v) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=np.float64), np.asarray(q, dtype=np.float64))
    return out


def _world_reach_error(route, reached_by_frame) -> float:
    """Max realized tool-site error (m) over the route's ACTIVE frames, WORLD frame (same nearest-of-
    goalset rule as ``planner.reach_error``). ``reached_by_frame`` = ``{frame: reached_world_pos[3]}``.
    Each active frame's ``grasp_cand_base`` ``[G,3]`` is mapped to world with the route's stored planning
    pose (``base_pos``/``base_quat``) -- NOT the live tilted base -- so the datum matches what was
    planned; the verdict is the WORST active frame (a bimanual reach passes only if BOTH hands land)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(route.base_quat, dtype=np.float64))
    Rm = R.reshape(3, 3)
    worst = 0.0
    for frame, (_target, cand_base, _err) in route.reaches.items():
        world_cands = route.base_pos[None, :] + (Rm @ cand_base.T).T   # [G,3]
        reached = np.asarray(reached_by_frame[frame])
        worst = max(worst, float(np.linalg.norm(world_cands - reached[None, :], axis=1).min()))
    return worst


# ------------------------------------------------------------------------------------------------
class KinematicHarness:
    """Idealized (kinematic) sim harness on bare MuJoCo. See module docstring for the loop contract.

    Args:
        robot: ``"v2"`` | ``"v2_fixed"`` | ``"g1"``.
        scenario_name: pickplace scenario (e.g. ``"front_back_close"``).
        control_dt: seconds per tick (base integration + executor cadence).
        camera: ``False`` (default) = PRIVILEGED, ``observe`` returns EVERY cube (GT); ``True`` = the
            visibility gate, ``observe`` returns only the cubes the head cameras SEE (fixed-cam static
            frustum ``cube_visible_poses``; actuated v2 gimbals aimed per-cube via ``gaze_sees``). The
            reachability-driven count then follows from the visible∩reachable set.
        drift: ``False`` (default) = the base stands where the policy leaves it (parked reach, the
            byte-identical 0.0000 m gate). ``True`` = inject a scripted ZERO-MEAN base-drift SINUSOID each
            ``realize`` (XY sway of amplitude ``_DRIFT_AMP_M`` along the measured lean axis + in-phase pitch),
            a CONTINUOUSLY MOVING base (balance sway), so the base MOVES during the reach as an EXTERNAL
            disturbance -- and keeps moving during each replan solve (tracking-latency test). The
            plan-once policy tracks its plan-time joints, so the tool slides off the world-fixed target ->
            the phase-3 kinematic analog of the phase-4 base-drift miss (which a live-base replan rejects).
            Drift OWNS the base when on (the policy ``base_twist`` must be ~0; asserted in ``realize``).
    """

    def __init__(self, robot: str, scenario_name: str, control_dt: float = 0.02,
                 camera: bool = False, drift: bool = False, walk: bool = False,
                 height_pad: float = 0.02) -> None:
        assert not (walk and drift), "walk and drift both own the base motion; pick one"
        cfg, _object_cfg, nominal_scenario, table_z = robot_scene(robot, scenario_name)
        spec = cfg.build_entity_spec()
        make_scene_spec_fn(nominal_scenario)(spec)
        add_ground_plane(spec, with_grid=_floor_grid_enabled())   # OUR plane; kinematic had no ground before (SAME plane as dynamic)
        if height_pad > 0.0:
            add_planner_object_geoms(spec, height_pad=height_pad)
        # Session built BEFORE compile so its planner collision spheres can be added as child geoms on the
        # link bodies (group PLANNER_SPHERE_GROUP, viewer-togglable). No extra cost: this path builds the
        # session eagerly.
        session = make_planner_session(cfg)
        add_planner_collision_geoms(spec, session.robot_cfg_dict["robot_cfg"]["kinematics"]["collision_spheres"])
        mj_model = spec.compile()
        # Match mjlab's scene.xml headlight (mjlab/src/mjlab/scene/scene.xml) so kinematic-viewer lighting
        # is not dimmer than the DynamicHarness's mjlab-built model (default MuJoCo headlight is
        # ambient 0.1/diffuse 0.4/specular 0.5 -- visibly darker side-by-side).
        mj_model.vis.headlight.ambient[:] = [0.3, 0.3, 0.3]
        mj_model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
        mj_model.vis.headlight.specular[:] = [0.0, 0.0, 0.0]
        self.cfg = cfg
        self.height_pad = float(height_pad)

        # Keep the NOMINAL (un-jittered) scenario so every jitter samples from it and no reset compounds a
        # previous sample.
        self._nominal_scenario = nominal_scenario
        self.scenario_name = scenario_name
        self.scenario = nominal_scenario
        move_pick_geoms(mj_model, self.scenario)
        self.table_z = table_z
        self.control_dt = float(control_dt)
        self.camera = bool(camera)
        self._drift = bool(drift)
        self._home_base_pos = np.asarray(cfg.home_base_pos, dtype=float)
        self._k = 0                          # drift phase step counter (zeroed on reset)
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(mj_model)
        self.spec = spec         # retained for the Newton PBR viewer, which reparses from source
        self.session = session

        root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, cfg.base_link)
        self._root_adr = mj_model.joint(mj_model.body(root_id).jntadr[0]).qposadr[0]
        self._site_ids = {frame: mj_model.site(frame).id for frame in cfg.tool_frames}
        self._gripper_qadr: dict[str, list[int]] = {"L": [], "R": []}
        for joint_id in range(mj_model.njnt):
            joint_name = mj_model.joint(joint_id).name
            if not joint_name.endswith(("left_rack_y", "right_rack_y")):
                continue
            side = "L" if joint_name.startswith("L_") else "R" if joint_name.startswith("R_") else None
            if side is not None:
                self._gripper_qadr[side].append(int(mj_model.jnt_qposadr[joint_id]))
        assert all(self._gripper_qadr.values()), "parallel-gripper rack joints missing from kinematic model"
        self._cube_qadr = {}
        for m in self.scenario.pick:
            body_id = mj_model.body(f"pp_obj_{m.name}").id
            jid = int(mj_model.body_jntadr[body_id])
            assert mj_model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE, (
                f"pick body pp_obj_{m.name} must have a free joint")
            self._cube_qadr[m.name] = int(mj_model.jnt_qposadr[jid])
        # ``side -> (cube name, hand-frame grasp-center position, hand-frame cube orientation)``. The
        # kinematic harness writes this rigid attachment directly into free-joint qpos instead of stepping
        # contact dynamics, making pickup/carry an explicit upper-bound assumption.
        self._kinematic_grasps: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        self._yaw = 0.0
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Reset robot to the planning-home stance. ``seed`` (viewer only) re-jitters the pick cubes from
        the NOMINAL layout so each reach sees a fresh sample; ``None`` (selftest / headless default) keeps
        the current layout deterministic."""
        if seed is not None:
            self.scenario = jitter_pick_markers(self._nominal_scenario, seed)
            move_pick_geoms(self.mj_model, self.scenario)
        d = self.mj_data
        d.qpos[:] = self.mj_model.qpos0
        d.qpos[self._root_adr:self._root_adr + 3] = self.cfg.home_base_pos
        d.qpos[self._root_adr + 3:self._root_adr + 7] = self.cfg.home_base_quat
        _apply_home_pose(self.mj_model, d, self.cfg)
        self._yaw = 0.0
        self._k = 0
        self._kinematic_grasps.clear()
        mujoco.mj_forward(self.mj_model, d)

    def _observed_cubes(self) -> dict:
        """The cubes the cams FRAME at the CURRENT realized gaze this tick (LIVE, non-accrued): ALL cubes
        (GT) when ``camera=False``, else the visible subset. Perception is a pure sensor now -- the POLICY
        owns the gimbal aim + belief accrual (seam inversion, REACH_FSM_REDESIGN.md). Actuated gimbals were
        already driven to the policy's ``camera_command`` in ``realize``, so ``detect_at_gaze`` reads whatever
        the head points at RIGHT NOW; fixed cams use the static-frustum ``cube_visible_poses``. No sweep, no
        accrual, no ``scan_settled`` here."""
        if not self.camera:
            return cube_world_poses(self.mj_model, self.scenario, self.mj_data)
        if self.cfg.gaze_cams:
            return detect_at_gaze(self.mj_model, self.mj_data, self.scenario, self.cfg)
        return cube_visible_poses(self.mj_model, self.mj_data, self.scenario, self.cfg)

    def observe(self):
        """``(base_pose=(pos,quat), proprio_qpos, observed_cube_pose)``. Cube belief per the camera gate."""
        pos = self.mj_data.qpos[self._root_adr:self._root_adr + 3].copy()
        quat = self.mj_data.qpos[self._root_adr + 3:self._root_adr + 7].copy()
        proprio = self.mj_data.qpos.copy()
        return (pos, quat), proprio, self._observed_cubes()

    @staticmethod
    def _closed_arm_sides(gripper_closed: set[str] | None) -> set[str]:
        """Normalize policy gripper labels to the harness's ``L``/``R`` arm keys."""
        closed = set()
        for item in gripper_closed or ():
            label = str(item).lower()
            if "l" in label or "left" in label:
                closed.add("L")
            if "r" in label or "right" in label:
                closed.add("R")
        return closed

    def _update_kinematic_grasps(self, gripper_closed: set[str] | None) -> None:
        """Attach closed-jaw targets to grasp sites without simulating contact.

        The dynamic harness latches only after finger contact, then solves a MuJoCo weld. This idealized
        harness instead waits for its rack coordinates to reach first contact, attaches the nearest unattached
        cube within the grasp-center tolerance, then writes its free-joint pose from the hand transform every
        tick. Result: viewer and kinematic rollouts show close-then-pickup-carry semantics while remaining
        an optimistic upper bound on dynamic execution.
        """
        d = self.mj_data
        closed = self._closed_arm_sides(gripper_closed)
        for side in tuple(self._kinematic_grasps):
            if side not in closed:
                del self._kinematic_grasps[side]

        attached = {cube for cube, _rel_pos, _rel_quat in self._kinematic_grasps.values()}
        for side, frame in zip(("L", "R"), self.cfg.tool_frames):
            if side not in closed or side in self._kinematic_grasps:
                continue
            contact = parallel_gripper_contact_rack_pos(CUBE_SIZE)
            if any(abs(d.qpos[qadr] - (self.mj_model.qpos0[qadr] + contact)) > 1e-9
                   for qadr in self._gripper_qadr[side]):
                continue
            tool_pos = d.site(self._site_ids[frame]).xpos
            candidates = []
            for marker in self.scenario.pick:
                if marker.name in attached:
                    continue
                cube_pos = d.body(f"pp_obj_{marker.name}").xpos
                candidates.append((float(np.linalg.norm(tool_pos - cube_pos)), marker.name))
            if not candidates:
                continue
            distance, cube_name = min(candidates)
            if distance > _KINEMATIC_GRASP_ATTACH_DIST_M:
                continue

            hand = d.body(f"{side}_base")
            cube = d.body(f"pp_obj_{cube_name}")
            hand_rot = np.zeros(9)
            mujoco.mju_quat2Mat(hand_rot, hand.xquat)
            rel_pos = hand_rot.reshape(3, 3).T @ (tool_pos - hand.xpos)
            hand_inv = np.zeros(4)
            rel_quat = np.zeros(4)
            mujoco.mju_negQuat(hand_inv, hand.xquat)
            mujoco.mju_mulQuat(rel_quat, hand_inv, cube.xquat)
            self._kinematic_grasps[side] = (cube_name, rel_pos, rel_quat)
            attached.add(cube_name)

        for side, (cube_name, rel_pos, rel_quat) in self._kinematic_grasps.items():
            hand = d.body(f"{side}_base")
            hand_rot = np.zeros(9)
            mujoco.mju_quat2Mat(hand_rot, hand.xquat)
            qadr = self._cube_qadr[cube_name]
            d.qpos[qadr:qadr + 3] = hand.xpos + hand_rot.reshape(3, 3) @ rel_pos
            cube_quat = np.zeros(4)
            mujoco.mju_mulQuat(cube_quat, hand.xquat, rel_quat)
            d.qpos[qadr + 3:qadr + 7] = cube_quat

    def _update_kinematic_grippers(self, gripper_closed: set[str] | None) -> None:
        """Rate-limit rack qpos toward open or first-contact targets for viewer-visible jaw motion.

        Dynamic execution commands past first contact to generate position-servo preload. Kinematic
        rollouts have no contact force, so they stop at the shared task cube's geometric contact position rather
        than visibly intersecting it. Both rack coordinates are written because the kinematic harness uses
        ``mj_forward`` rather than the equality-constraint solver that enforces their mimic relation during
        dynamics.
        """
        closed = self._closed_arm_sides(gripper_closed)
        max_delta = PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S * self.control_dt
        for side, qaddrs in self._gripper_qadr.items():
            offset = (parallel_gripper_contact_rack_pos(CUBE_SIZE)
                      if side in closed else PARALLEL_GRIPPER_OPEN_RACK_POS_M)
            for qadr in qaddrs:
                goal = self.mj_model.qpos0[qadr] + offset
                current = self.mj_data.qpos[qadr]
                self.mj_data.qpos[qadr] = current + np.clip(goal - current, -max_delta, max_delta)

    def realize(self, base_twist, arm_reference, camera_command, gripper_closed: set[str] | None = None) -> None:
        """Kinematically integrate the base twist (planar yaw+translation), write the arm reference and
        camera command onto their joints, ``mj_forward``. ``arm_reference``/``camera_command`` are
        ``{mjlab_joint: qpos}``; both are absolute references (no gains) in the idealized robot."""
        d = self.mj_data
        vx, vy, wz = base_twist
        if vx or vy or wz:
            self._yaw += wz * self.control_dt
            quat = np.asarray(yaw_quat(self._yaw), float)
            step_world = _quat_rot(quat, np.array([vx, vy, 0.0]) * self.control_dt)
            d.qpos[self._root_adr:self._root_adr + 3] += step_world
            d.qpos[self._root_adr + 3:self._root_adr + 7] = quat
        for jname, value in arm_reference.items():
            d.qpos[self.mj_model.joint(jname).qposadr[0]] = value
        for jname, value in camera_command.items():
            d.qpos[self.mj_model.joint(jname).qposadr[0]] = value
        if self._drift:
            assert abs(vx) + abs(vy) + abs(wz) < 1e-9, (
                f"drift owns the base; policy base_twist {base_twist} must be ~0 under --drift")
            # ZERO-MEAN base-drift sinusoid = a continuously MOVING base (balance sway), NOT a plateau, so a
            # replan must chase a target that moves DURING its own solve (tracking-latency test). Sway along
            # the measured lean axis; pitch oscillates in phase. k=0 -> sin=0 -> base at home for the first
            # (clean) plan. Plan-once tracks its plan-time joints so the tool slides off the moving target.
            s = np.sin(2.0 * np.pi * self._k / _DRIFT_PERIOD_STEPS)
            t = _DRIFT_AMP_M * s * _DRIFT_DIR
            R = rot_rpy_deg(0.0, _DRIFT_PITCH_DEG * s, 0.0)
            d.qpos[self._root_adr:self._root_adr + 3] = self._home_base_pos + t
            d.qpos[self._root_adr + 3:self._root_adr + 7] = drift_base_quat(R)
            self._k += 1
        self._update_kinematic_grippers(gripper_closed)
        # Update robot FK before testing the grasp-center distance, then again after writing carried-cube
        # free-joint poses so perception and viewer rendering observe the attachment on this same tick.
        mujoco.mj_forward(self.mj_model, d)
        self._update_kinematic_grasps(gripper_closed)
        mujoco.mj_forward(self.mj_model, d)

    def verdict(self, route) -> float:
        reached = {frame: self.mj_data.site_xpos[self._site_ids[frame]].copy() for frame in route.reaches}
        return _world_reach_error(route, reached)

    def base_drift_norm(self) -> float:
        """``‖live_base_pos - nominal_home_base_pos‖`` (m) -- the injected base translation magnitude, for
        the drift-vs-reach-error decompose log. Ignores the pitch/quat component (translation dominates)."""
        pos = self.mj_data.qpos[self._root_adr:self._root_adr + 3]
        return float(np.linalg.norm(pos - self._home_base_pos))

    def upright(self) -> float:
        """The idealized robot never falls; the shared gate's upright check is a constant pass here (contract
        parity with ``DynamicHarness.upright``, so ``run_reach`` is harness-agnostic)."""
        return 1.0

    def settle(self, steps: int = 0) -> None:
        """No-op: ``mj_forward`` is instant, so there is no physics transient to settle before the first plan
        (contract parity with ``DynamicHarness.settle``)."""


def run_reach(harness, policy, timeout: int, verbose: bool = False, settle_window: int | None = None,
              stop_tol: float | None = None, dwell: int | None = None) -> float:
    """Drive ``policy`` on ``harness`` for AT MOST ``timeout`` ticks; return success score, or
    ``nan`` if the policy HELD the whole time (no reachable/visible cube -> ``last_route`` stays None). A
    hold is a valid graded outcome (0 active arms), so the caller decides via ``policy.last_route`` whether
    that was expected -- this function does not assert a route exists.

    ``timeout`` is a CAP, not a fixed budget (it doubled as a per-harness convergence budget before -- pick
    it too small and a valid dynamic policy false-fails on a mid-transient read). Two tail-window modes,
    MUTUALLY EXCLUSIVE:
    * ``stop_tol`` (parked reach): CONVERGE-EARLY. Keep a rolling window of the last ``dwell`` ROUTED
      verdicts and stop once they are ALL < ``stop_tol`` -- SUSTAINED proximity, not a first-touch crossing
      (the dynamic frozen policy is a limit cycle whose error dips under tol mid-swing then drifts back out,
      so a single crossing does not prove a HELD grasp). Return the PEAK over that window (worst steady
      state). A route drop mid-run CLEARS the window (never blend pre/post-drop ticks). ``dwell`` must
      exceed the limit-cycle period AND be <= ``timeout`` (else the window never fills -> never converges).
      Kinematic converges in a few ticks (``mj_forward`` plateaus instantly); dynamic runs until the limit
      cycle sits under tol or times out.
    * ``settle_window`` (drift eval): FIXED-duration, NO early stop. Return the PEAK error over the LAST
      ``settle_window`` routed ticks (phase-independent under the zero-mean sway). The base never settles,
      so an early stop would corrupt the correction-vs-correction comparison.
    * neither: return the FINAL-tick error -- correct for a static/plateaued base."""
    from collections import deque
    assert not (settle_window is not None and stop_tol is not None), (
        "settle_window (fixed-duration) and stop_tol (converge-early) are mutually exclusive")
    if stop_tol is not None:
        assert dwell is not None and dwell <= timeout, "converge-early needs dwell set and dwell <= timeout"
    tail = []
    grasp_error = None
    window = deque(maxlen=dwell) if stop_tol is not None else None
    k = 0
    while k < timeout:
        base_pose, proprio, cube = harness.observe()
        tool_poses_base = harness.tool_poses_base() if hasattr(harness, "tool_poses_base") else None
        physical_grasps = harness.physical_grasps() if hasattr(harness, "physical_grasps") else None
        base_twist, arm_ref, cam = policy.step(base_pose, proprio, cube, tool_poses_base, physical_grasps)
        harness.realize(base_twist, arm_ref, cam, gripper_closed=policy.gripper_closed)
        if getattr(policy, "plan0_pending", False):
            # The planner is still solving plan-0 off-loop (async mpc path). Physics advanced above (the robot
            # holds planning-home), but this tick is NOT part of the reach budget -- ``timeout`` caps REACHING,
            # not waiting on the planner. ``getattr`` default False keeps every other path (kinematic / sync /
            # async / selftest) byte-identical (route installs at tick 0, no pending phase).
            continue
        routed = policy.last_route is not None
        if policy.grasp_route is not None and grasp_error is None:
            capture = getattr(harness, "grasp_capture_error", None)
            err = capture(policy.grasp_route) if capture is not None else harness.verdict(policy.grasp_route)
            if np.isfinite(err):
                # Dynamic capture is binary: real contact started the latch and easing completed. Its live
                # center error is diagnostic, not a second, looser success condition. Kinematic mode has no
                # latch, so its geometric error remains the score.
                grasp_error = 0.0 if capture is not None else err
        if routed and settle_window is not None and k >= timeout - settle_window:
            tail.append(harness.verdict(policy.last_route))
        if window is not None:
            if routed:
                window.append(harness.verdict(policy.last_route))
            else:
                window.clear()          # route drop resets convergence (invariant: no pre/post-drop blend)
        if verbose and routed and (k % 25 == 0 or k == timeout - 1):
            drift = (f", base_drift {harness.base_drift_norm():.4f} m"
                     if getattr(harness, "_drift", False) else "")
            print(f"  step {k:4d}: reach_err {harness.verdict(policy.last_route):.4f} m{drift}")
        if window is not None and len(window) == dwell and max(window) < stop_tol and policy.grasp_route is None:
            break                        # SUSTAINED proximity over the full dwell window -> converged
        if policy.parked_done:
            break
        k += 1
    if policy.grasp_route is not None:
        return grasp_error if grasp_error is not None else float("nan")
    if policy.last_route is None:        # invariant: hold checked BEFORE max(window) (no max([]) on a hold)
        return float("nan")
    if window is not None:
        return max(window) if window else float("nan")
    if settle_window is not None:
        return max(tail) if tail else float("nan")
    return harness.verdict(policy.last_route)


_TRIAD_AXIS_M = 0.06                                # grasp-frame axis length drawn in the viewer
_TRIAD_WIDTH_M = 0.005                              # arrow shaft width
_TRIAD_RGBA = (                                     # x=red, y=green, z=blue
    np.array([1.0, 0.2, 0.2, 1.0], dtype=np.float32),
    np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32),
    np.array([0.2, 0.4, 1.0, 1.0], dtype=np.float32),
)

_WP_AXIS_M = 0.03                                  # key-waypoint tool-frame axis length (half the grasp triad)
_WP_WIDTH_M = 0.0015                               # thin shaft so dense key-waypoint triads stay legible
_WP_SAMPLE = 4                                     # draw every Nth planned waypoint (key waypoints, not all H)

# Gripper-target coordinate frame (X=red, Y=green, Z=blue, full alpha), same convention as the deploy
# viewer's IK-target axes. Distinct from the _TRIAD_/_WP_ route triads above: these mark the world-fixed
# cube GRASP GOALS. Shared by BOTH walk viewers (kinematic ``_replay_walk_in_viewer`` +
# dynamic ``pickplace_reach_env._run_walk_mission``) so the two walk paths draw the SAME waypoint
# trajectory and differ only in physics.
_TARGET_AXIS_RGBA = (np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                     np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
                     np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
_TARGET_AXIS_WIDTH, _TARGET_AXIS_LEN = 0.004, 0.12


def draw_target_axes(scn, frames) -> None:
    """Draw each ``(world_pos[3], world_rot[3x3])`` as an RGB coordinate triad (``mjGEOM_ARROW``, local
    +Z permuted onto each axis) into a passive-viewer ``user_scn``. Shared drawer for the cube grasp-goal
    trajectory: the kinematic walk viewer (``_draw_target_frames``) and the dynamic walk viewer
    (``_run_walk_mission.sync_view``) both call it so the two paths show the SAME world-fixed waypoint
    frames -- they differ only in physics. Caller owns ``scn.ngeom`` reset/append bookkeeping."""
    for pos, rot in frames:
        rot = np.asarray(rot, dtype=np.float64)
        if rot.shape == (4,):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, rot)
            rot = R.reshape(3, 3)
        else:
            rot = rot.reshape(3, 3)
        for axis_idx in range(3):
            if scn.ngeom >= scn.maxgeom:
                return
            col_order = [(axis_idx + 1) % 3, (axis_idx + 2) % 3, axis_idx]
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                np.array([_TARGET_AXIS_WIDTH, _TARGET_AXIS_WIDTH, _TARGET_AXIS_LEN]),
                                np.asarray(pos, dtype=np.float64), rot[:, col_order].flatten(),
                                _TARGET_AXIS_RGBA[axis_idx])
            scn.ngeom += 1


_NAV_ARROW_WIDTH_M = 0.012                          # nav-target arrow shaft radius (thicker than route triads --
#   this is the single most important debug marker: base->target, position AND drive/facing direction in one geom)
_NAV_ARROW_RGBA = {
    "DISCOVER": np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32),   # yellow: driving to the raw anchor
    "APPROACH": np.array([0.1, 0.9, 1.0, 1.0], dtype=np.float32),    # cyan: driving to point C
}


def draw_nav_arrow(scn, base_pos, target_w, phase: str | None) -> None:
    """One ``mjGEOM_ARROW`` from the base's current XY to ``policy.nav_target_w`` (raised to a fixed height
    so it clears the robot/table geometry): arrow tip = target POSITION, arrow orientation = drive/facing
    DIRECTION -- both in a single cheap geom, no extra state. Colored by ``policy.walk_phase`` (DISCOVER's
    raw anchor vs. APPROACH's point C) so the FSM branch is visible alongside the target itself. No-op if
    ``target_w`` is ``None`` (before the first DISCOVER/APPROACH tick) or ``phase`` isn't one of the two nav
    states (stale target held during EXTEND/GRASP/RETRACT). Caller owns ``scn.ngeom`` reset."""
    if target_w is None or phase not in _NAV_ARROW_RGBA or scn.ngeom >= scn.maxgeom:
        return
    z = 0.9
    base = np.array([base_pos[0], base_pos[1], z])
    tip = np.array([target_w[0], target_w[1], z])
    if np.linalg.norm(tip[:2] - base[:2]) < 1e-3:      # arrived: degenerate arrow, skip
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        _NAV_ARROW_RGBA[phase])
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, _NAV_ARROW_WIDTH_M, base, tip)
    scn.ngeom += 1


_NAV_AXIS_M = 0.12                                  # dynamic-viewer nav-target marker axis length (mjlab has no
#   raw user_scn access, so the nav target is drawn as an ``add_frame`` triad instead of ``draw_nav_arrow``'s
#   single ``mjGEOM_ARROW`` -- X axis (bright, phase-colored) points along the base->target bearing, Y/Z axes
#   dimmed so the triad still reads as ONE directional marker instead of a generic RGB frame)
_NAV_AXIS_WIDTH = 0.006
_NAV_DIM_RGB = (0.35, 0.35, 0.35)


def _nav_axis_colors(phase: str):
    """3-axis ``axis_colors`` for ``DebugVisualizer.add_frame``: X = ``_NAV_ARROW_RGBA[phase]`` (bright,
    encodes the bearing), Y/Z dimmed gray (present only because ``add_frame`` always draws a full triad)."""
    r, g, b, _a = _NAV_ARROW_RGBA[phase]
    return ((float(r), float(g), float(b)), _NAV_DIM_RGB, _NAV_DIM_RGB)


def _route_path_world(mj_model, route, every=_WP_SAMPLE):
    """World-frame tool-frame triads at the KEY planned waypoints of a plan-0 route: FK each sampled
    ``route.route_q_curobo`` config in a scratch ``MjData`` (root free joint pinned to identity == base
    frame), read each active tool site, transform base->world with the route's SOLVED base pose (the same
    datum ``_grasp_triads_world`` uses). Returns ``[(pos_world[3], rotmat_world9)]`` flat over active frames.

    Tracker-free so the kinematic (phase-3) passive viewer -- which has no live ``JacobianReachTracker`` --
    can draw the planned approach path; the dynamic (phase-4) viewer instead overlays the tracker's live
    drift-corrected key goals (``JacobianReachTracker.key_goals_base``)."""
    data = mujoco.MjData(mj_model)
    root = next(int(mj_model.jnt_qposadr[j]) for j in range(mj_model.njnt)
                if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)

    def qadr(name):
        for j in range(mj_model.njnt):
            n = mj_model.joint(j).name
            if n == name or n.split("/")[-1] == name:
                return int(mj_model.jnt_qposadr[j])
        raise KeyError(name)

    def sid(frame):
        for n in (frame, f"robot/{frame}"):
            i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
            if i != -1:
                return i
        raise KeyError(frame)

    adr = np.array([qadr(n) for n in route.joint_names])
    offset = mj_model.qpos0[adr]
    sids = {f: sid(f) for f in route.reaches}
    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, np.asarray(route.base_quat, dtype=np.float64))
    Rb = Rb.reshape(3, 3)
    rq = np.asarray(route.route_q_curobo, dtype=float)
    idx = list(range(0, len(rq), every))
    if idx[-1] != len(rq) - 1:
        idx.append(len(rq) - 1)
    triads = []
    for k in idx:
        data.qpos[:] = mj_model.qpos0
        data.qpos[root:root + 3] = 0.0
        data.qpos[root + 3:root + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[adr] = rq[k] + offset
        mujoco.mj_forward(mj_model, data)
        for f in route.reaches:
            pw = np.asarray(route.base_pos) + Rb @ data.site_xpos[sids[f]]
            Rw = Rb @ data.site_xmat[sids[f]].reshape(3, 3)
            triads.append((pw, Rw.reshape(9)))
    return triads


def _grasp_triads_world(cfg, mj_model, site_xpos_view, site_ids, base_pos, base_quat, route):
    """``[(origin_world[3], rotmat_world[9])]`` one per active frame: the grasp candidate the arm actually
    reached (the nearest goalset member, matching the ``.min()`` verdict rule), recomputed to recover the
    grasp ORIENTATION that ``ReachRoute`` drops (``reaches`` stores candidate POSITIONS only). Rebuilt via
    the same ``cube_grasp_poses_obj`` -> ``grasp_poses_to_base`` pipeline as the planner, then base->world
    with the route's SOLVED base pose (the datum the plan is valid in).

    Caller-supplied handles: ``cfg`` (ik robot cfg, owns ``cube_grasp_poses_obj`` / ``grasp_poses_to_base``);
    ``mj_model`` (model bindings); ``site_xpos_view`` = ``MjData.site_xpos``-shaped ndarray for the live site
    world positions (the kinematic harness passes its ``self.mj_data.site_xpos``; the dynamic harness's
    ``_ReachEvalEnv`` passes an equivalent map derived from ``scene["robot"].data.site_pose_w``);
    ``site_ids`` = ``{frame: site_idx}`` map; ``base_pos``/``base_quat`` = route's solved base pose."""
    import torch

    Rb = np.zeros(9)
    mujoco.mju_quat2Mat(Rb, np.asarray(base_quat, dtype=np.float64))
    Rb = Rb.reshape(3, 3)
    gp_obj_pos, gp_obj_quat = cfg.cube_grasp_poses_obj(device="cpu")   # [G,3], [G,4] (obj frame)
    # World-level grasp at the (possibly tilted) solved base -- the SAME obj_quat the planner used to build
    # the goalset (world_axes_in_base_quat), so the triads match the solved grasp instead of a base-tilted one.
    from tasks.visual_manipulation.curobo.planner import world_axes_in_base_quat
    obj_quat = torch.tensor(world_axes_in_base_quat(base_quat), dtype=torch.float32)
    triads = []
    for frame, (target_base, _cand, _err) in route.reaches.items():
        cand_pos_b, cand_quat_b = cfg.grasp_poses_to_base(
            torch.tensor(np.asarray(target_base), dtype=torch.float32), obj_quat, gp_obj_pos, gp_obj_quat)
        cand_pos_b, cand_quat_b = cand_pos_b.numpy(), cand_quat_b.numpy()
        world = np.asarray(base_pos)[None, :] + (Rb @ cand_pos_b.T).T    # [G,3] candidate world pos
        reached = np.asarray(site_xpos_view[site_ids[frame]], dtype=float)
        k = int(np.linalg.norm(world - reached[None, :], axis=1).argmin())   # goalset member the arm hit
        wq = np.zeros(4)
        mujoco.mju_mulQuat(wq, np.asarray(base_quat, float), np.asarray(cand_quat_b[k], float))
        wm = np.zeros(9)
        mujoco.mju_quat2Mat(wm, wq)
        triads.append((world[k], wm))
    return triads


def reach_overlay_triads(cfg, mj_model, site_xpos_view, site_ids, route):
    """SINGLE SOURCE OF TRUTH for the reach-viewer overlay: the reached grasp-pose triads PLUS the planned
    route-path waypoints, as ``[(origin_world[3], rot9, axis_len_m, shaft_radius_m)]``. BOTH viewers draw
    EXACTLY this list -- the kinematic passive viewer (``_draw_reach_overlay`` -> ``user_scn`` arrows) and
    the dynamic mjlab viewer (``update_visualizers`` -> ``DebugVisualizer.add_frame``) -- so the two show
    identical markers and differ ONLY in RL policy + physics. Grasp triads are the dominant (long/thick)
    markers; route-path waypoints are half-length/thin. ``site_xpos_view``/``site_ids`` are the live tool-site
    world positions the grasp producer needs (kinematic passes ``mj_data.site_xpos``; the dynamic env passes a
    ``site_pose_w``-derived view)."""
    triads = [(o, r, _TRIAD_AXIS_M, _TRIAD_WIDTH_M)
              for o, r in _grasp_triads_world(cfg, mj_model, site_xpos_view, site_ids,
                                              route.base_pos, route.base_quat, route)]
    triads += [(o, r, _WP_AXIS_M, _WP_WIDTH_M) for o, r in _route_path_world(mj_model, route)]
    return triads


def draw_triads(scn, triads) -> None:
    """Draw a ``[(origin_world[3], rot9, axis_len_m, shaft_radius_m)]`` list into an ``mjvScene`` as RGB arrow
    triads. Caller owns ``scn.ngeom`` reset; stops silently if the scene runs out of geom slots.

    Split out of ``_draw_reach_overlay`` so the offscreen recorder can draw a SUBSET of the overlay (the
    planned route path without the grasp triads) without duplicating the arrow math or pulling in the
    producer's ``harness``-shaped arguments."""
    for origin, rot9, axis_len, radius in triads:
        Rm = rot9.reshape(3, 3)
        for i, rgba in enumerate(_TRIAD_RGBA):
            if scn.ngeom >= scn.maxgeom:
                return
            tip = np.asarray(origin, float) + axis_len * Rm[:, i]
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3), np.zeros(9), rgba)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, radius,
                                 np.asarray(origin, float), tip)
            scn.ngeom += 1


def route_path_triads(mj_model, route, scale: float = 1.0):
    """The planned cuRobo route path alone, as a ``draw_triads`` list (thin/short waypoint markers).

    The grasp-pose half of ``reach_overlay_triads`` is deliberately excluded: it needs live tool-site world
    positions, and in a figure panel the long/thick grasp axes sit exactly on top of the gripper they are
    meant to annotate.

    ``scale`` multiplies both axis length and shaft width. The defaults are sized for a screen-filling
    viewer; a paper panel is ~1.2 in wide, where a 1.5 mm shaft falls below one printed pixel and the path
    disappears. Scaled here rather than by raising ``_WP_WIDTH_M`` because the viewers are the constants'
    primary consumer and thickening them there would clutter the dense live overlay."""
    return [(o, r, _WP_AXIS_M * scale, _WP_WIDTH_M * scale)
            for o, r in _route_path_world(mj_model, route)]


def _draw_reach_overlay(scn, harness, route) -> None:
    """Kinematic backend of the single-source overlay: draw the shared ``reach_overlay_triads`` list (grasp
    triads + planned route path) into a passive viewer's ``user_scn`` as RGB arrow triads. The dynamic viewer
    draws the SAME list via ``DebugVisualizer.add_frame``."""
    draw_triads(scn, reach_overlay_triads(
        harness.cfg, harness.mj_model, harness.mj_data.site_xpos, harness._site_ids, route))


_CAPTURE_DIR = pathlib.Path(__file__).parent / "test" / "captures"


def _draw_overlays(scn, harness, policy, base_pose) -> None:
    """Reach-overlay triad (planned grasp pose) + nav arrow (walk target), shared between the live
    ``v.user_scn`` (drawn every frame) and an offscreen capture scene (drawn once, on demand)."""
    if policy.last_route is not None:
        _draw_reach_overlay(scn, harness, policy.last_route)
    draw_nav_arrow(scn, base_pose[0], policy.nav_target_w, policy.walk_phase)


def apply_view_options(opt, *, show_fov: bool) -> None:
    """WYSIWYG seed for live / offscreen / recording -- single answer to "what is visible".

    Planner spheres start HIDDEN (press 5); FOV follows the flag.

    Note: walk chase-cam MP4 now renders the frustums too. At that 1.9 m the edges can rake
    across the grasp region and hide the 60 mm cube -- pass ``show_fov=False`` at the recorder
    call site if it bites.
    """
    opt.geomgroup[FOV_GEOM_GROUP] = int(show_fov)
    opt.geomgroup[PLANNER_SPHERE_GROUP] = 0


class ViewCapture:
    """``o``-key PNG, shared by both live viewers. Caller hands in its own ``cam``+``opt`` so the
    PNG cannot disagree with the window (WYSIWYG by construction).

    Threading: ``on_key`` runs on the RENDER thread (flag-only); ``save`` touches ``mj_data`` and
    MUST run on the main thread inside the caller's own lock -- two viewers, two lock handles,
    so we take neither.

    The renderer is built lazily on first capture because the dynamic viewer's ``viewer.mjm``
    exists only after ``setup()`` runs -- well after this object has to be built to hand
    ``on_key`` to the viewer.
    """

    def __init__(self, harness, tag: str):
        self._harness, self._tag, self._renderer = harness, tag, None
        self.requested = False

    def on_key(self, key: int) -> None:
        if key == glfw.KEY_O:
            self.requested = True

    def save(self, model, data, cam, opt, decorate=None) -> None:
        """Render + write the pending capture. No-op unless ``o`` armed one.

        ``decorate(scene)`` is the caller's chance to redraw ``user_scn`` overlays -- a fresh
        ``mjvScene`` is built by ``update_scene`` and does not inherit them. The dynamic viewer
        uses mjlab's ``DebugVisualizer`` and passes None.
        """
        if not self.requested:
            return
        import imageio.v2 as imageio          # lazy, matching the rest of this module

        self.requested = False
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model,
                                             height=model.vis.global_.offheight,
                                             width=model.vis.global_.offwidth)
        self._renderer.update_scene(data, camera=cam, scene_option=opt)
        if decorate is not None:
            decorate(self._renderer.scene)
        img = self._renderer.render()
        _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (_CAPTURE_DIR /
                    f"{self._harness.scenario.name}-{self._harness.cfg.name}-{self._tag}-{stamp}.png")
        imageio.imwrite(out_path, img)
        print(f"  [view] saved capture -> {out_path}")
        # cam.lookat is an ndarray; formatting a LIST renders each element via repr, which under
        # NumPy 2 is "np.float64(0.3)" not "0.3". The cast keeps this line pastable into
        # _VIEW_CAMERA_POSE, which is its only job.
        print(f"  [view] cam.lookat={[round(float(v), 4) for v in cam.lookat]}, "
              f"cam.distance={cam.distance:.4f}, "
              f"cam.azimuth={cam.azimuth:.2f}, cam.elevation={cam.elevation:.2f}")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()


def resize_native_viewer_window(width: int, height: int) -> None:
    """Best-effort native-viewer window resize via ``xdotool`` (X11 only). Neither
    ``mujoco.viewer.launch_passive`` nor mjlab's ``NativeMujocoViewer`` expose a window-size argument --
    the GLFW window is created inside the compiled ``_simulate`` binding with no Python-level size hook --
    so this shells out to resize the OS window by title match (``"MuJoCo : ..."``) after the fact. Call
    ONLY once the window is confirmed created (``launch_passive`` returns synchronously post-creation;
    mjlab's ``viewer.setup()`` likewise runs before its render loop starts). No-op without a DISPLAY or
    without ``xdotool`` installed. Actual size is capped by the monitor's resolution (the window manager
    clamps an oversized request)."""
    import shutil
    import subprocess

    if not os.environ.get("DISPLAY") or shutil.which("xdotool") is None:
        return
    found = subprocess.run(["xdotool", "search", "--name", "MuJoCo"],
                            capture_output=True, text=True).stdout.split()
    if found:
        subprocess.run(["xdotool", "windowsize", found[-1], str(width), str(height)])


def run_reach_viewer(harness, policy_factory, steps: int, start_seed: int | None = None,
                     window_size: tuple[int, int] | None = None, viewer: str = "native") -> None:
    """Live passive-viewer variant of ``run_reach`` for a KINEMATIC harness: run ONE reachability-driven
    reach (the arm count follows from what is reachable/visible) through the SAME observe->step->realize
    loop at wall-clock control cadence. A completed walk keeps stepping its terminal grasp reference until
    the user closes the viewer, matching dynamic viewer behavior. ``policy_factory()`` builds a fresh
    ``ReachPolicy`` so each reach starts from the planning-home seed. Kinematic only (bare
    ``mj_model``/``mj_data``); the dynamic harness is verified headless.

    A WALK policy (``policy._walk``) runs the full multi-visit mission per episode: the per-visit reach
    (SEARCH->GO->REACH->PARK) ends the episode on ``policy.walk_done`` rather than a single route's cursor,
    so the viewer plays walk->search->reach->park for every cube. A parked policy ends one reach when its
    interpolated route has fully played plus a short hold.

    Each reach RE-JITTERS the pick cubes: reach ``i`` uses seed ``base + i`` (``base`` = ``start_seed`` or,
    when None, a random draw), printed for replay. Set env ``VIEW_LOCK_SEED=1`` to hold ``base`` every reach
    (freeze the layout). This is why layouts now vary across reaches; the headless selftest stays nominal.

    The cuRobo planner collision spheres are added as child geoms on the robot links at build time (group
    ``PLANNER_SPHERE_GROUP``, ``add_planner_collision_geoms``); this viewer starts with that group HIDDEN --
    press the ``5`` key to toggle the whole-body planner sphere overlay (MuJoCo's native geom-group toggle,
    so it also works in the dynamic/mjlab viewer).

    Press ``o`` to save the current view (same camera + geom-group visibility as on screen) as a PNG under
    ``_CAPTURE_DIR``. ``key_callback`` runs on the render thread, so it only sets a flag; the actual
    offscreen render happens from the main thread inside the existing per-step ``v.lock()`` section, same
    as every other ``mj_data`` read here.

    Threading invariant: ``launch_passive`` renders on a separate thread. Every access to the shared
    ``mj_data`` (including observations, reset/FK, and overlay reads) stays inside ``v.lock()``. Planning
    runs from copied observations after releasing that lock, so a slow cuRobo solve never blocks rendering.
    Without this split the renderer can race ``mj_forward``/``qpos`` writes and intermittently segfault.

    ``viewer="newton"`` renders the SAME observe->step->realize loop through Newton's ``ViewerGL``
    (PBR lighting) instead of MuJoCo's rasterizer -- see ``_run_reach_viewer_newton``. Scene only: no
    ``_draw_overlays``/``ViewCapture``/key toggles/``window_size`` (single-threaded GL render, no
    ``v.lock()`` needed since nothing else touches ``harness.mj_data`` concurrently).
    ``viewer="blender"`` streams the same loop to a Blender GUI (EEVEE global illumination) -- see
    ``_run_reach_viewer_blender``, same scene-only scope.
    """
    if viewer == "newton":
        _run_reach_viewer_newton(harness, policy_factory, steps, start_seed=start_seed)
        return
    if viewer == "blender":
        _run_reach_viewer_blender(harness, policy_factory, steps, start_seed=start_seed)
        return
    import mujoco.viewer

    base_seed = start_seed if start_seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32))
    lock = bool(os.environ.get("VIEW_LOCK_SEED"))
    hold_s = 0.6                           # brief dwell on the grasp pose after the route finishes, then reset
    reach_i = 0

    capture = ViewCapture(harness, "kin")
    with mujoco.viewer.launch_passive(harness.mj_model, harness.mj_data,
                                      key_callback=capture.on_key) as v:
        if window_size is not None:
            resize_native_viewer_window(*window_size)
        # Frame the task: default free camera sits at the origin (== the robot base) and renders black.
        # Camera FOV frustums are welded to the camera bodies, so they follow the gimbals as the gaze
        # aims. Show them STATICALLY under --camera -- no per-frame alpha pulse (distracting strobe).
        with v.lock():
            v.cam.lookat[:] = [0.0, 0.0, 0.65]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.2, 135.0, -15.0
            apply_view_options(v.opt, show_fov=getattr(harness, "camera", False))
        print("  [view] keyboard shortcuts:")
        print("  [view]   5 -> toggle planner collision-sphere overlay")
        print(f"  [view]   o -> save current view (PNG) to {_CAPTURE_DIR}")
        while v.is_running():
            seed = base_seed if lock else base_seed + reach_i
            reach_i += 1
            with v.lock():
                harness.reset(seed)
                pick_layout = [(m.name, tuple(round(c, 3) for c in m.pos)) for m in harness.scenario.pick]
            print(f"  [view] {harness.cfg.name}: seed {seed} -> {pick_layout}")
            v.sync()                       # draw the standing robot BEFORE the seconds-long first cuRobo warm-plan
            policy = policy_factory()
            step_i = 0
            while v.is_running():
                if step_i >= steps and not (getattr(policy, "_walk", False) and policy.walk_done):
                    break
                # Snapshot all MuJoCo-owned values while the passive renderer is excluded. ``observe``
                # returns copies, so cuRobo can solve from them after releasing the lock.
                with v.lock():
                    base_pose, proprio, cube = harness.observe()
                base_twist, arm_ref, cam = policy.step(base_pose, proprio, cube)
                with v.lock():
                    harness.realize(base_twist, arm_ref, cam, gripper_closed=policy.gripper_closed)
                    v.user_scn.ngeom = 0   # redraw the reach overlay each frame (pose = where the arm aims)
                    _draw_overlays(v.user_scn, harness, policy, base_pose)
                    capture.save(harness.mj_model, harness.mj_data, v.cam, v.opt,
                                 decorate=lambda scn: _draw_overlays(scn, harness, policy, base_pose))
                v.sync()                   # no wall-clock throttle: render as fast as the loop computes
                step_i += 1
                if getattr(policy, "_walk", False):
                    continue
                # Stop once the interpolated route has fully played + a short hold, instead of dwelling
                # on the held final waypoint for the remaining fixed steps (reads as a frozen/sleeping arm).
                ex = policy._executor
                if ex is not None and ex.cursor_s >= ex.duration_s + hold_s:
                    break
            if not v.is_running():
                break
            if policy.last_route is not None:
                with v.lock():
                    reach_err = harness.verdict(policy.last_route)
                print(f"  [view] {harness.cfg.name}: {len(policy.last_route.reaches)}-arm reach, "
                      f"reach_err {reach_err:.4f} m")
            else:
                print(f"  [view] {harness.cfg.name}: no reachable cube")
            if hasattr(policy, "close"):
                policy.close()             # stop any spawned cuRobo worker (MPC/async) before the next reach
    capture.close()


def reach_viewer_config(distance: float = 3.2, azimuth: float = 135.0, elevation: float = -15.0):
    """``ViewerConfig`` describing the reach viewer's shot, for the photoreal bridge.

    The bridge is written against mjlab's ``ViewerConfig`` because that is what an mjlab env hands
    it; this harness has no env, so the same shot the native passive viewer opens on is spelled out
    as one here. ``AUTO`` (not the native path's fixed ``lookat``) so the pivot tracks the robot's
    floating base: identical while parked, and the only workable choice once ``--walk`` drives the
    base across the room.
    """
    from mjlab.viewer.viewer_config import ViewerConfig

    return ViewerConfig(origin_type=ViewerConfig.OriginType.AUTO, distance=distance,
                        azimuth=azimuth, elevation=elevation)


def _run_reach_viewer_blender(harness, policy_factory, steps: int, start_seed: int | None = None) -> None:
    """Blender (EEVEE) variant of ``run_reach_viewer``'s loop: SAME reset->observe->step->realize
    sequence, scene-only render. Blender runs as a separate process fed world geom poses per tick
    (``photoreal.bridge.BlenderStream``), so like the Newton path there are no ``_draw_overlays``,
    no ``ViewCapture`` and no key toggles -- and no ``v.lock()``, since the render happens in
    another process that never touches ``harness.mj_data``.

    ``realize`` ends in ``mj_forward``, so the geom and light world poses the stream reads are
    already current -- no extra kinematics pass here.

    Visibility and light levels come from MuJoCo (``apply_view_options``, ``mujoco_look``) rather
    than the bridge's photoreal grade: this scene declares one active light against a 0.6 headlight,
    so dropping the headlight renders it nearly black, and the bridge's default group set shows the
    FOV hull, whose 5 m emissive shells then fill the frame."""
    from photoreal.bridge import BlenderStream, track_body_id

    base_seed = start_seed if start_seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32))
    lock = bool(os.environ.get("VIEW_LOCK_SEED"))
    hold_s = 0.6
    reach_i = 0

    cfg = reach_viewer_config()
    opt = mujoco.MjvOption()
    apply_view_options(opt, show_fov=getattr(harness, "camera", False))
    harness.reset(base_seed)
    stream = BlenderStream(harness.mj_model, harness.mj_data, cfg,
                           track_body_id(harness.mj_model, cfg, {}),
                           geomgroup=opt.geomgroup, mujoco_look=True)
    print("  [view] Blender photoreal viewer (planned trajectory only -- no capture/key toggles)")
    while stream.is_running():
        seed = base_seed if lock else base_seed + reach_i
        reach_i += 1
        harness.reset(seed)
        pick_layout = [(m.name, tuple(round(c, 3) for c in m.pos)) for m in harness.scenario.pick]
        print(f"  [view] {harness.cfg.name}: seed {seed} -> {pick_layout}")
        stream.send(harness.mj_data)
        policy = policy_factory()
        seen_viz_route = object()
        step_i = 0
        while stream.is_running():
            if step_i >= steps and not (getattr(policy, "_walk", False) and policy.walk_done):
                break
            base_pose, proprio, cube = harness.observe()
            base_twist, arm_ref, cam = policy.step(base_pose, proprio, cube)
            route = policy.viz_route
            if route is not seen_viz_route:
                points = None if route is None else np.asarray(
                    [p[0] for p in route_path_triads(harness.mj_model, route)], dtype=np.float32
                ).reshape(-1, len(route.reaches), 3).transpose(1, 0, 2)
                stream.send_route(points)
                seen_viz_route = route
            harness.realize(base_twist, arm_ref, cam, gripper_closed=policy.gripper_closed)
            stream.send(harness.mj_data)
            step_i += 1
            if getattr(policy, "_walk", False):
                continue
            ex = policy._executor
            if ex is not None and ex.cursor_s >= ex.duration_s + hold_s:
                break
        if not stream.is_running():
            break
        if policy.last_route is not None:
            reach_err = harness.verdict(policy.last_route)
            print(f"  [view] {harness.cfg.name}: {len(policy.last_route.reaches)}-arm reach, "
                  f"reach_err {reach_err:.4f} m")
        else:
            print(f"  [view] {harness.cfg.name}: no reachable cube")
        if hasattr(policy, "close"):
            policy.close()
    stream.close()


def _run_reach_viewer_newton(harness, policy_factory, steps: int, start_seed: int | None = None) -> None:
    """Newton ``ViewerGL`` (PBR) variant of ``run_reach_viewer``'s loop: SAME
    reset->observe->step->realize sequence, scene-only render (no ``_draw_overlays``/
    ``ViewCapture``/key toggles -- see ``photoreal.newton_bridge`` module docstring for the scope
    decision). ``ViewerGL`` renders synchronously on the calling thread (unlike
    ``mujoco.viewer.launch_passive``'s separate render thread), so no ``v.lock()`` is needed."""
    from photoreal.newton_bridge import (
        auto_frame_camera, build_newton_model_and_body_map, sync_mjdata_to_newton_state,
    )
    import newton
    import newton.viewer

    base_seed = start_seed if start_seed is not None else int(np.random.SeedSequence().entropy % (2 ** 32))
    lock = bool(os.environ.get("VIEW_LOCK_SEED"))
    hold_s = 0.6
    reach_i = 0

    model, body_map = build_newton_model_and_body_map(harness.mj_model, harness.spec)
    state = model.state()
    v = newton.viewer.ViewerGL()
    v.set_model(model)
    sync_mjdata_to_newton_state(harness.mj_data, body_map, state)
    auto_frame_camera(v, state)

    def _render(t: float) -> None:
        sync_mjdata_to_newton_state(harness.mj_data, body_map, state)
        v.begin_frame(t)
        v.log_state(state)
        v.end_frame()

    print("  [view] Newton PBR viewer (scene only -- no overlays/screenshot capture)")
    t = 0.0
    while v.is_running():
        seed = base_seed if lock else base_seed + reach_i
        reach_i += 1
        harness.reset(seed)
        pick_layout = [(m.name, tuple(round(c, 3) for c in m.pos)) for m in harness.scenario.pick]
        print(f"  [view] {harness.cfg.name}: seed {seed} -> {pick_layout}")
        _render(t); t += harness.control_dt
        policy = policy_factory()
        step_i = 0
        while v.is_running():
            if step_i >= steps and not (getattr(policy, "_walk", False) and policy.walk_done):
                break
            base_pose, proprio, cube = harness.observe()
            base_twist, arm_ref, cam = policy.step(base_pose, proprio, cube)
            harness.realize(base_twist, arm_ref, cam, gripper_closed=policy.gripper_closed)
            _render(t); t += harness.control_dt
            step_i += 1
            if getattr(policy, "_walk", False):
                continue
            ex = policy._executor
            if ex is not None and ex.cursor_s >= ex.duration_s + hold_s:
                break
        if not v.is_running():
            break
        if policy.last_route is not None:
            reach_err = harness.verdict(policy.last_route)
            print(f"  [view] {harness.cfg.name}: {len(policy.last_route.reaches)}-arm reach, "
                  f"reach_err {reach_err:.4f} m")
        else:
            print(f"  [view] {harness.cfg.name}: no reachable cube")
        if hasattr(policy, "close"):
            policy.close()
    v.close()


# ------------------------------------------------------------------------------------------------
# Dynamic harness: realize commands through the frozen RL low-level policy + MuJoCo/warp physics
# ------------------------------------------------------------------------------------------------
_REACH_EVAL_ENV_CLS = None
# Rack target rate and object-sized grasp position for direct gripper position control. The jaw collision
# pads have a 69.9 mm open inner gap; the shared 60 mm cube contacts at rack q=4.95 mm. Command 19 mm,
# leaving 14.05 mm per-side position-servo preload (about 11.24 N per jaw at kp=800), never hard-close at 34.7 mm.
# Weld-latch trigger + ease (see DynamicHarness._update_weld_latch). Tight trigger (real contact,
# small gap) so the cube doesn't snap in from a visibly-still-approaching hand; the correction from
# the observed gap to the designed grasp-center offset is then spread over this many seconds instead
# of written in one instant tick, so any residual gap closes as a smooth slide, not a teleport.
# Widened from 30 mm: the TCP site is the jaw-gap midpoint, ~9 mm distal of the pad centroid, so a
# cube pinched at the pad TIPS legitimately plateaus at 33-42 mm of TCP gap (measured on the
# bimanual_mixed_close/seed68 latch failure) and 30 mm rejected it forever. This is a sanity cap, not
# the discriminator -- the jaw-pad contact requirement below is what proves the cube is between the
# jaws, and a rack pad cannot touch a cube more than ~half a jaw away from the TCP anyway.
# Two rejected alternatives, both of which LOOKED like they worked on seed68:
#   1. Jaw-pad contact NORMAL FORCE above a threshold. Not scale-invariant: a genuine grasp reports
#      1.9-3.3 N on seed68 but only 0.06-0.3 N on bimanual_mixed_close/seed99 (both hands there,
#      including the one that latched fine), so any threshold clearing seed99's noise is under
#      seed68's floor. A 1 N gate turned seed99 from PASS into FAIL.
#   2. Driven-rack servo lag (target minus realized). Predicted a ~14 mm stall on the 60 mm cube, but
#      measured max lag was 6.5 mm WITH or WITHOUT a cube -- the stiff kp=800 servo squeezes through
#      MuJoCo's soft contact and reaches its target either way, so lag cannot separate a grasp from
#      free motion at all.
_WELD_TRIGGER_DIST_M = 0.06
# 0.15 s was fast enough to read as a snap on video: the ramp moves the cube from where it was physically
# pinched onto the DESIGNED grasp center, and that gap is legitimately 33-42 mm at the pad tips (see
# _WELD_TRIGGER_DIST_M above), so 60 mm / 0.15 s = 0.4 m/s of cube motion with no visible cause -- "the object
# suddenly gets attracted to the hand". Same total correction over 0.6 s is ~0.1 m/s and reads as settling.
# Safe to lengthen only because the ease now interpolates in the HAND frame (see _update_weld_latch): a 0.6 s
# ramp outlives the GRASP hold and runs into RETRACT, which the old world-frame blend would have turned into a
# drag toward a stale point in the room.
_WELD_EASE_S = 0.6
# A single tick where the policy's reported ``gripper_closed`` set momentarily excludes an arm
# (jaw micro-reopen during grasp refine, a phase-transition command blip) must not drop an
# already-engaged weld outright -- that free-falls the cube from mid-air back toward the table
# before any re-latch condition can re-fire, which looks like a teleport. Require this many
# consecutive open ticks before releasing.
_WELD_RELEASE_DEBOUNCE_TICKS = 5
# A cube the hand shoved off its support is unrecoverable: it can never satisfy the weld trigger again
# (see ``_WELD_TRIGGER_DIST_M``), so without this the mission burns BOTH ``_GRIPPER_CLOSE_MAX_S`` windows
# and reports ``physical grasp latch did not engage``, or -- once the re-plan cannot route to the floor --
# ``unreachable after 2 back-off retries``. Two labels for one cause, and the knock is contact-timing
# sensitive, so the SAME seed reports different classes run to run. Threshold is well below a table height
# (~0.4 m) so a cube nudged along the slab or resting on a lower shelf is not falsely condemned, and well
# above the settling jitter of a free body at spawn.
_CUBE_FALL_DROP_M = 0.10
# --- live handover: ONE discrete displacement of the offered object, inside the pre-act hold -----------
# Off unless ``LIVE_HANDOVER_SHIFT_M`` is set, so every published cell stays byte-identical by default.
#
# THE OBJECT NEVER MOVES WHILE THE ROBOT IS REACHING. cuRobo plans one trajectory against one frozen world
# snapshot, and an earlier revision that moved the figure mid-EXTEND was measured to fail for every robot
# alike (P 1.000 -> 0.200 on v2 lateral, ``unreachable after 2 back-off retries``): the body swings on a
# larger radius than the palm and rakes the torso obstacle across the arm's approach corridor, which has
# only 45 mm of clearance. That is a planner-geometry artifact, not a sensing result.
#
# The move therefore lands entirely inside ``reach_policy._PREACT_HOLD_TICKS``, during which the robot
# gazes and accrues belief but neither drives nor commits. Ordering (the whole experiment is this ordering):
# the robot WATCHES, the person MOVES, the robot COMMITS. The ramp ENDS at the published palm pose and
# STARTS one chord back along the arc, so the pose the robot must actually reach is the one 540 published
# trials already proved reachable, and only the stale belief is new.
#
# What it discriminates: whether the robot was watching the object when it moved. Measured own-target
# visibility (seeds 7/13/21) is v2 100.0%, v2_fixed 82.4% (66.7% on the handoff cube, ONE SEED AT 0.0%),
# g1 89.4% on bimanual_mixed_close, and ~100% for all three on the front_back twin -- so the lateral cell
# separates the ablation and the front/back cell is the negative control.
_LIVE_HANDOVER_ENV = "LIVE_HANDOVER_SHIFT_M"
# Rotation about the world z axis through the SCENE ORIGIN, not the live robot base: the perturbation must
# be a pure function of tick index so it is bit-identical across robots (a base-relative pivot would make
# the disturbance depend on each robot's gait). The origin is where every robot starts, so rotating the
# figure about it keeps the human's facing relative to the robot -- the person steps around the robot and
# keeps offering the object toward it, rather than sliding sideways.
_LIVE_HANDOVER_TRIGGER_TICK = 20    # ticks 0-19: the robot watches the pre-move pose, so a belief exists
_LIVE_HANDOVER_RAMP_TICKS = 20      # ticks 20-39: spread the move; a one-tick teleport ejects the cube
# Read, not imported: ``reach_policy`` OWNS the hold (it is the one that stops committing) but nothing in the
# harness imports the policy, and adding that edge would invert the dependency. The harness only needs the
# value to assert its ramp finishes inside the hold -- if the two ever disagree, that assert is what fires.
_PREACT_HOLD_TICKS = int(os.environ.get("REACH_PREACT_HOLD_TICKS", "0") or 0)
# Specified as a CHORD DISPLACEMENT IN METRES, not an angle: the two mixed cells hold their handoff cube at
# different radii (0.398 m lateral vs 0.255 m front/back), so a shared angle would move them by different
# distances and the control pair would differ in disturbance magnitude as well as in visibility. The angle
# is derived per cell from the radius instead, which holds the perturbation fixed and leaves visibility the
# only difference. 0.08 m is above _TARGET_MOVED_REPLAN_M (0.02, so the one permitted replan arms) with room
# to spare over the 0.05 m grade tolerance (so an unnoticed move actually costs the trial -- a shared 0.20
# rad gave the front/back cell only 0.0509 m, 0.9 mm of margin).
_LIVE_HANDOVER_DEFAULT_SHIFT_M = 0.08
# Planar shift that counts as a cube having been DISTURBED (``knock_witness``), as opposed to having fallen.
# Well above free-body settling jitter at spawn, well below the ~0.05 m cube half-width, so it trips while
# the pushing geom is still in contact -- which is the whole point: the report must name the culprit link.
_KNOCK_WITNESS_M = 0.01
_RELAX_ARM_LIMIT_ENV = "DYNAMIC_RELAX_ARM_LIMIT_RAD"


def _reach_eval_env_cls():
    """Lazily define ``_ReachEvalEnv`` so importing this module for the KINEMATIC path pulls no warp /
    flash_sac / frozen-policy deps. Built once, cached."""
    global _REACH_EVAL_ENV_CLS
    if _REACH_EVAL_ENV_CLS is not None:
        return _REACH_EVAL_ENV_CLS

    import torch
    from flash_sac.env_wrapper import ManagerBasedRlEnvWithFinalObs
    from tasks.frozen_policy import load_frozen_policy
    from tasks.visual_manipulation.policies import _GAZE_YAW_ROM, _GAZE_PITCH_ROM

    class _ReachEvalEnv(ManagerBasedRlEnvWithFinalObs):
        """Minimal reach-eval env with separate RL reference and actuator-target seams.

        ``ReachPolicy`` supplies an absolute arm command. This env writes it twice: ``arm_ref`` keeps the
        frozen policy's observation unchanged, while ``direct_arm`` writes the same command directly to
        the arm position actuators after ``joint_pos_arms``. Therefore RL still sees the intended arm pose
        and controls legs/base, but its untrained arm residual cannot corrupt the planned arm target.
        ``direct_arm`` must follow its frozen residual term; otherwise the residual overwrites it.
        """

        def __init__(self, *, cfg, device, robot_cfg, policy_task, policy_checkpoint, **kw):
            super().__init__(cfg=cfg, device=device, **kw)
            # This live reach env already has policy-compatible actor/action layouts. Reusing it as the
            # FlashSAC shape source avoids constructing a second two-env CUDA/MuJoCo oracle beside it.
            self._policy_fn, self._policy_obs_group, _action_dim = load_frozen_policy(
                policy_task, policy_checkpoint, device=str(device), shape_env=self)
            self._arm_ref = self.command_manager.get_term("arm_ref")
            assert hasattr(self._arm_ref, "set_command"), "arm_ref must be a passive holder (reach=True)"
            self._direct_arm = self.action_manager.get_term("direct_arm")
            assert self._direct_arm.action_dim == 0, "direct_arm must not consume frozen-policy actions"
            assert tuple(self._direct_arm._target_names) == tuple(self._arm_ref.target_names), (
                "direct_arm and arm_ref joint order diverged")
            arm_idx = self.action_manager.active_terms.index("joint_pos_arms")
            assert self.action_manager.active_terms.index("direct_arm") > arm_idx, (
                "direct_arm must apply after joint_pos_arms")
            entity = self.scene["robot"]
            # GPU LOS batch (lazy-init: the model is compiled and warp data is populated at the
            # first observation step, so defer until then). Per-harness cache (no cross-harness
            # sharing -- ``_WarpBatch.ray_clear`` mutates ``self.wd`` in place).
            self._warp_batch = None        # type: _WarpBatch | None
            # Actuated gripper (reach_actuated_gripper=True): racks are a zero-dim external action term the
            # RL policy never touches -- driven ONLY by direct position control (``set_gripper``). Restore
            # their full physical range first (the robot's 0.9 locomotion soft-limit margin would clip the
            # closed rack target); the term auto-holds default_joint_pos (jaws half-open) until commanded.
            self._runtime_gripper = self.action_manager.get_term("runtime_gripper")
            assert self._runtime_gripper.action_dim == 0, "runtime gripper changed v123 action shape"
            from asset_zoo.humanoid_v21.humanoid_v21_constants import HAND_REGISTRY
            gripper_joints = tuple(
                prefix + joint.name
                for prefix, hand in HAND_REGISTRY["parallel_gripper"].hands.items()
                for joint in hand.load_module().joints
            )
            gripper_joint_ids, _ = entity.find_joints(gripper_joints, preserve_order=True)
            entity.data.soft_joint_pos_limits[:, gripper_joint_ids] = \
                entity.data.joint_pos_limits[:, gripper_joint_ids]
            # Direct position-control targets for the driven rack joints (the term's own target set): open =
            # default_joint_pos - 10 mm; grasp = q=19 mm, sized for the shared task cube. It deliberately does NOT
            # use the q=34.7 mm hard-close limit, which would over-squeeze the object and gripper.
            gr_ids = self._runtime_gripper.target_ids
            self._gripper_open = entity.data.default_joint_pos[:, gr_ids] + PARALLEL_GRIPPER_OPEN_RACK_POS_M
            self._gripper_grasp = self._gripper_open + PARALLEL_GRIPPER_CLOSE_STROKE_M
            assert bool((self._gripper_grasp <= entity.data.joint_pos_limits[:, gr_ids, 1]).all()), (
                "task-cube grasp target exceeds gripper travel")
            self._gripper_target = self._gripper_open.clone()
            # Per-arm rack columns keyed by the L_/R_ attach prefix (HAND_REGISTRY), so a grasp closes ONLY
            # the grasping hand -- a cube held in one gripper must not force the other jaw shut and block it.
            self._gripper_arm_cols: dict[str, list[int]] = {"L": [], "R": []}
            for col, name in enumerate(self._runtime_gripper.target_names):
                self._gripper_arm_cols["L" if name.rsplit("/", 1)[-1].startswith("L_") else "R"].append(col)
            self._arm_home_pose = entity.data.default_joint_pos[:, self._arm_ref.target_ids].clone()
            self._arm_name_to_col = {n.rsplit("/", 1)[-1]: i for i, n in enumerate(self._arm_ref.target_names)}
            for key, col in self._arm_name_to_col.items():
                if key in robot_cfg.planning_home_joint_pos:
                    self._arm_home_pose[:, col] = robot_cfg.planning_home_joint_pos[key]
            # Per-tool-frame site map (both L/R), so the verdict reads whatever frames a route drives --
            # single OR bimanual. Mirrors KinematicHarness._site_ids over cfg.tool_frames.
            self._site_ids = {frame: entity.find_sites(frame)[0][0] for frame in robot_cfg.tool_frames}
            # Head-camera gaze seam (actuated-gimbal robots only; fixed-camera g1 has no camera_ref term).
            # ``set_camera`` writes the harness-computed search/track gaze angles into the camera_ref holder;
            # the frozen loco policy then servos the physical head gimbals to it -- the SAME holder the camera
            # learner drives (camera_learner_env.py). Per-column ROM clamp aligned to the camera_ref joint order.
            self._camera_ref = (self.command_manager.get_term("camera_ref")
                                if "camera_ref" in self.command_manager.active_terms else None)
            if self._camera_ref is not None:
                self._camera_name_to_col = {
                    n.rsplit("/", 1)[-1]: i for i, n in enumerate(self._camera_ref.target_names)}
                rom = [_GAZE_YAW_ROM if "yaw" in n else _GAZE_PITCH_ROM for n in self._camera_ref.target_names]
                self._camera_rom = torch.tensor(rom, device=self.device)
                self._camera_target = None

        def reach_action(self, arm_reference):
            """Write one planned command to separate RL-reference and actuator-target channels.

            ``arm_ref`` is observation only: RL consumes it and still generates its full action. The same
            ``cmd`` becomes ``direct_arm``'s physical position target, applied after RL's
            ``joint_pos_arms`` residual. ``None`` holds planning-home on both channels.
            """
            cmd = self._arm_home_pose.clone()
            if arm_reference is not None:
                for jname, value in arm_reference.items():
                    cmd[:, self._arm_name_to_col[jname]] = value
            self._arm_ref.set_command(cmd)
            self._direct_arm.set_targets(cmd)
            return self._policy_fn(
                {self._policy_obs_group: self.obs_buf[self._policy_obs_group]}).detach()

        def step_reach(self, arm_reference):
            """Write the arm reference, run frozen inference, integrate one control step."""
            return self.step(self.reach_action(arm_reference))

        def step(self, action):
            """Advance physics, then apply the bounded fallback for grafted camera motors.

            Warp accepts their position targets but leaves their joint qpos near reset.  Apply the same
            bounded position servo after its physics step so the next rendered/perceived state has the
            commanded physical camera pose.  Arms and legs remain on normal physics actuators.
            """
            out = super().step(action)
            self._enforce_camera_pose()
            return out

        def set_gripper(self, closed_arms) -> None:
            """Rate-limit per-arm gripper position-control targets, then let the stiff servo hold force.

            ``closed_arms`` is the set of arm sides ("L"/"R") whose jaws should hold the 19 mm cube-grasp
            position; every other rack rate-limits back to open. Per-arm (not one global bool) so a hand
            holding an earlier cube keeps its grip while the free hand opens to grasp the next object -- the
            single-bool version force-closed BOTH jaws and blocked the second grasp. Each control step moves
            each rack target by at most ``PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S * step_dt``; contact then holds the cube with
            modest position-servo preload.
            """
            closed_arms = set(closed_arms)
            norm_closed = set()
            for item in closed_arms:
                item_str = str(item).lower()
                if "l" in item_str or "left" in item_str:
                    norm_closed.add("L")
                if "r" in item_str or "right" in item_str:
                    norm_closed.add("R")
            closed_arms = norm_closed

            goal = self._gripper_open.clone()
            for side, cols in self._gripper_arm_cols.items():
                if side in closed_arms:
                    goal[:, cols] = self._gripper_grasp[:, cols]
            max_delta = PARALLEL_GRIPPER_RACK_MAX_SPEED_M_S * self.step_dt
            self._gripper_target += (goal - self._gripper_target).clamp(-max_delta, max_delta)
            self._runtime_gripper.set_targets(self._gripper_target)

        def set_camera(self, gaze_qpos: dict) -> None:
            """Write per-gimbal yaw/pitch gaze targets into the camera_ref holder; the frozen loco policy
            servos the physical head gimbals to them (one-step obs latency, same as the camera learner).

            ``gaze_qpos`` maps bare gimbal joint name (``cam_yaw_left`` ...) -> angle (rad). Joints absent
            from the map hold default joint position (tilt-down default). No-op for fixed-camera robots
            (g1 has no camera_ref term). Angles are already in-ROM (the gaze solver clamps), but the
            per-column ROM clamp guards against a stray write."""
            if self._camera_ref is None:
                return
            cmd = self.scene["robot"].data.default_joint_pos[:, self._camera_ref.target_ids].clone()
            for jname, angle in gaze_qpos.items():
                cmd[:, self._camera_name_to_col[jname]] = angle
            cmd = cmd.clamp(-self._camera_rom, self._camera_rom)
            self._camera_ref.set_command(cmd)
            self._camera_target = cmd

        def _enforce_camera_pose(self) -> None:
            """Apply one rate-limited camera-servo tick after Warp physics.

            Warp's grafted gimbal actuators retain their target but do not evolve joint state.  This is a
            simulator integration fallback, not a perception shortcut: target comes only from ``set_camera``
            and every joint advances at the hardware 3.5 rad/s limit.
            """
            if self._camera_ref is None or self._camera_target is None:
                return
            current = self.scene["robot"].data.joint_pos[:, self._camera_ref.target_ids]
            delta = self._camera_target - current
            for col, name in enumerate(self._camera_ref.target_names):
                if "yaw" in name:
                    delta[:, col] = torch.atan2(torch.sin(delta[:, col]), torch.cos(delta[:, col]))
            max_step = _GAZE_SLEW_MAX * self.step_dt
            next_pos = current + delta.clamp(-max_step, max_step)
            self.scene["robot"].write_joint_state_to_sim(
                next_pos, torch.zeros_like(next_pos), joint_ids=self._camera_ref.target_ids)
            self.sim.forward()
            # Refresh entity mirrors after the post-step qpos write.  The next policy observation and
            # viewer therefore read the same physical gimbal pose as the MuJoCo FOV geometry.
            self.scene.update(dt=0.0)

        def gripper_qpos(self):
            """Live driven-rack joint positions ``[L_left_rack_y, R_left_rack_y]`` (m) -- the realized jaw
            state under the position servo. Read for the open/close demo verdict."""
            gr_ids = self._runtime_gripper.target_ids
            return self.scene["robot"].data.joint_pos[0, gr_ids].detach().cpu().numpy()

        def reach_site_pos_w(self, frames):
            """``{frame: world_pos[3]}`` for the requested tool frames (a route's active frames)."""
            pose_w = self.scene["robot"].data.site_pose_w
            return {f: pose_w[0, self._site_ids[f], :3].detach().cpu().numpy().astype(float) for f in frames}

        def tool_poses_base(self, frames):
            """Measured live tool poses in base_link, read from two site tensors, never Warp qpos.

            This is the deploy seam for the dynamic tracker's bounded Cartesian position integrator. Root and
            tool site poses are already produced by physics FK; converting their compact tensors to CPU avoids
            the former full ``wp_data.qpos`` readback. The tracker intentionally uses only position feedback,
            but quaternions are carried for the hardware pose-sensor contract.
            """
            entity = self.scene["robot"]
            root_w = entity.data.root_link_pose_w[0].detach().cpu().numpy().astype(float)
            site_w = entity.data.site_pose_w[0, [self._site_ids[f] for f in frames]].detach().cpu().numpy()
            root = (root_w[:3], root_w[3:])
            return {f: pose_relative(root, (site_w[i, :3], site_w[i, 3:])) for i, f in enumerate(frames)}

        def update_visualizers(self, visualizer) -> None:
            """Draw the planner's reachable grasp triads (RGB axes at each active frame's NEAREST planned
            grasp candidate, in WORLD coords) via the viewer debug-visualizer seam (mjlab hooks
            ``env.unwrapped.update_visualizers`` from ``NativeMujocoViewer.sync_env_to_viewer``). The
            current route + session route map live on ``self._debug_policy`` (set via ``set_debug_policy``)
            -- keeps the env from touching the high-level policy directly while letting the share-the-same-
            visualization-as-phase-3 contract hold (the kinematic viewer's
            ``_grasp_triads_world`` -> DebugVisualizer ``add_frame`` for the dynamic viewer).

            Falls back silently when no policy/route is attached (e.g. plain frozen-policy playback): a
            debug-visualizer hook with no debug to draw is fine -- the native viewer just renders the
            user_scn empty after its own markers.

            Also draws the WALK-mission nav target (``policy.nav_target_w``, DISCOVER's raw anchor or
            APPROACH's point C) as a single directional triad -- BEFORE the ``last_route is None`` guard
            below, since DISCOVER/APPROACH have no route yet (that's exactly when this marker matters).
            mjlab exposes no raw ``user_scn`` here, so this is ``add_frame`` (a full triad), not the
            kinematic viewer's single ``mjGEOM_ARROW`` (``draw_nav_arrow``) -- see ``_nav_axis_colors``."""
            policy = getattr(self, "_debug_policy", None)
            if policy is None:
                return
            target_w, phase = policy.nav_target_w, policy.walk_phase
            if target_w is not None and phase in _NAV_ARROW_RGBA:
                base_w = self.scene["robot"].data.root_link_pos_w[0].detach().cpu().numpy().astype(float)
                dx, dy = float(target_w[0] - base_w[0]), float(target_w[1] - base_w[1])
                if dx * dx + dy * dy > 1e-6:            # not yet arrived -- degenerate frame otherwise
                    bearing = float(np.arctan2(dy, dx))
                    c, s = np.cos(bearing), np.sin(bearing)
                    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                    visualizer.add_frame(np.array([target_w[0], target_w[1], 0.9]), rot,
                                         scale=_NAV_AXIS_M, axis_radius=_NAV_AXIS_WIDTH,
                                         axis_colors=_nav_axis_colors(phase))
            if policy.last_route is None:
                return
            from tasks.visual_manipulation.curobo_reach_harness import (
                _TRIAD_AXIS_M,
                _TRIAD_RGBA,
                _TRIAD_WIDTH_M,
                _WP_AXIS_M,
                _WP_WIDTH_M,
                _grasp_triads_world,
                _route_path_world,
            )
            robot_cfg = getattr(self, "_debug_robot_cfg", None)
            if robot_cfg is None:
                return
            # Build an (Nsites, 3) world-pos view shaped like ``mj_data.site_xpos`` so the shared
            # ``reach_overlay_triads`` producer -- the SAME one the kinematic passive viewer draws -- can be
            # called unchanged on the dynamic side (single source of truth for the overlay).
            pose_w = self.scene["robot"].data.site_pose_w[0].detach().cpu().numpy()
            n_sites = int(self.sim.mj_model.nsite)
            site_xpos_view = np.zeros((n_sites, 3), dtype=np.float64)
            for fid, sid in self._site_ids.items():
                site_xpos_view[sid] = pose_w[sid, :3]
            axis_colors = tuple(tuple(float(rgba[i]) for i in range(3)) for rgba in _TRIAD_RGBA)
            route = policy.last_route
            # Route-waypoint FK is immutable after plan-0. Recomputing ~40 MuJoCo FK calls every rendered
            # frame made the dynamic viewer miss its 50 Hz budget. Cache it by route identity; the two grasp
            # triads remain live because their chosen goalset orientation follows measured tool position.
            if getattr(self, "_debug_path_route", None) is not route:
                self._debug_path_route = route
                self._debug_path_triads = _route_path_world(self.sim.mj_model, route)
            triads = [(o, r, _TRIAD_AXIS_M, _TRIAD_WIDTH_M)
                      for o, r in _grasp_triads_world(
                          robot_cfg, self.sim.mj_model, site_xpos_view, self._site_ids,
                          route.base_pos, route.base_quat, route)]
            triads += [(o, r, _WP_AXIS_M, _WP_WIDTH_M) for o, r in self._debug_path_triads]
            # Draw EXACTLY the kinematic overlay (grasp triads + planned route path) via this viewer's
            # ``add_frame`` backend. This is the SOLE overlay set; the former SECOND set drawn here -- the
            # tracker's live ``key_goals_base`` -- is DROPPED so the dynamic viewer matches kinematic output.
            for origin, rot9, axis_len, radius in triads:
                visualizer.add_frame(np.asarray(origin, float), rot9.reshape(3, 3),
                                     scale=axis_len, axis_radius=radius, axis_colors=axis_colors)

        def set_debug_policy(self, policy, *, robot_cfg=None) -> None:
            """Attach the live ``ReachPolicy`` whose ``last_route`` the viewer should draw. Idempotent;
            pass ``None`` to clear. ``robot_cfg`` is the ik robot cfg (``harness.cfg``); the
            ``update_visualizers`` hook needs it for ``cube_grasp_poses_obj`` / ``grasp_poses_to_base``."""
            self._debug_policy = policy
            self._debug_robot_cfg = robot_cfg
            self._debug_path_route = None
            self._debug_path_triads = ()

    _REACH_EVAL_ENV_CLS = _ReachEvalEnv
    return _REACH_EVAL_ENV_CLS


def _reach_gripper_robot_cfg(robot: str):
    """Actuated-parallel-gripper robot EntityCfg to graft as ``scene.entities["robot"]`` in the dynamic
    reach env. Always non-None (``reach_actuated_gripper=True`` requires a gripper-equipped robot: the
    frozen policies train on the WELDED robot -- no rack DOFs -- so the reach env must swap in the
    actuated-gripper variant for the ``runtime_gripper`` position term to bind, with the racks excluded
    from the policy obs/action layout in ``frozen_policy._apply_reach_mutations``).

    g1 mirrors ``G1_CFG.build_entity_spec`` (curobo/planner.py) EXACTLY -- builtin head (the
    ``head_camera_rgb`` site the visibility gate reads) -- so it carries g1's own foot/body/sensor names
    (not the humanoid's, which g1's ankle-roll contact sensor cannot resolve). v2/v2_fixed/v2_single/
    v2_single_fixed all take the ACTUATED humanoid head-camera variant (dual or single module), matching
    the frozen humanoid policy's live camera gimbals -- v2_fixed/v2_single_fixed only differ in the
    SOFTWARE gaze (statically pinned, see ``_welded_cam_gaze``), never the physics model's DOF count."""
    if robot == "g1":
        from asset_zoo.g1.g1_constants import get_g1_robot_cfg
        return get_g1_robot_cfg(head_camera="builtin", end_effector="actuated", hand="parallel_gripper")
    from asset_zoo.humanoid_v21 import get_humanoid_v21_robot_cfg
    head_camera = "actuated_single" if robot in ("v2_single", "v2_single_fixed") else "actuated"
    return get_humanoid_v21_robot_cfg(head_camera=head_camera, end_effector="actuated",
                                      hand="parallel_gripper")


def _apply_planning_home_init_state(env_cfg, planning_home_joint_pos: dict[str, float]) -> None:
    """Set the dynamic robot's initial arm qpos from the same descriptor map cuRobo receives.

    ``planning_home_joint_pos`` is the sole arm-home source: kinematic reset, cuRobo cspace default,
    arm-reference hold, and this physics reset all consume it. Preserve every non-arm reset entry. mjlab
    resolves joint-position maps by FIRST matching regex, so remove any source pattern matching a planner
    arm before appending its exact value; otherwise a broad source key (for example G1's
    ``.*_elbow_joint``) silently wins over the planner home.
    """
    robot_cfg = env_cfg.scene.entities["robot"]
    joint_pos = {
        pattern: value for pattern, value in robot_cfg.init_state.joint_pos.items()
        if not any(re.match(pattern, joint) for joint in planning_home_joint_pos)
    }
    env_cfg.scene.entities["robot"] = dataclasses.replace(
        robot_cfg,
        init_state=dataclasses.replace(
            robot_cfg.init_state,
            joint_pos={**joint_pos, **planning_home_joint_pos},
        ),
    )


def _dynamic_env_cfg(scenario, device: str, task: str, base_body: str, stand_z: float, has_camera: bool,
                     walk: bool = False, reach_gripper_robot_cfg=None, planning_home_joint_pos=None,
                     collision_spheres=None, single_camera: bool = False):
    """Reach-configured env_cfg (frozen-policy-tracked arm_ref, zeroed twist, origin base) with the
    Phase-3 scene merged in and the base qpos0 lifted so the constraint-buffer sizing pose stands.

    The descriptor planning-home map is applied before environment construction, so physics begins at the
    same arm pose cuRobo plans from. The twist un-gag is applied ONLY when ``walk`` (the parked
    reach base stays standing, twist pinned to zero).

    ``walk``: un-gag the twist command channel so the harness ``realize`` can drive a scripted
    ``vel_command_b`` (the walk-search base motion). The Reach experiment pins twist ranges to (0,0) and
    ``rel_standing_envs=1.0`` -> standing envs get ``vel_command_b`` zeroed every step and heading envs
    get wz overwritten, so a scripted write would be erased; the velocity curriculum would un-pin the
    (0,0) ranges at the first reset and ``velocity_tracking_failure`` would hard-reset a blocked walk.
    Verified recipe (identical to ``_run_walk_mission``). Parked (``walk=False``) leaves all of this
    untouched -> byte-identical to the pre-walk phase-4 env.

    ``task`` / ``base_body`` / ``stand_z`` are per-robot (``_DYNAMIC_TASK`` + ``_dynamic_stand``): the
    frozen low-level policy, the base body to pin, and its standing height (humanoid HOME base_link 0.59;
    g1 KNEES_BENT pelvis 0.76). Pinning the WRONG body/height leaves the constraint-buffer sizing pose
    off the ground and the base sinks/topples in ``settle``."""
    from tasks.frozen_policy import build_configured_env_cfg

    assert planning_home_joint_pos is not None, "dynamic reach needs descriptor planning_home_joint_pos"
    assert reach_gripper_robot_cfg is not None, "dynamic reach needs the gripper-equipped robot cfg"
    # Build through the reach seam: swaps in the parallel-gripper robot, registers the zero-dim
    # runtime_gripper direct-position term, excludes the racks from the frozen policy's obs/action
    # layout (so the welded-trained net is unchanged), strips DR (reach_nominal), and adds the
    # arm_ref/camera_ref passive holders (camera_ref skipped for fixed-camera g1). Default pattern is
    # DUAL (_CAMERA_PATTERN); single-camera robots must override it or the passive gaze holder's
    # actuator regex matches zero joints on the single-cam entity and the command build fails loud.
    if not has_camera:
        cam_kw = {"reach_camera_actuator_names": None}
    elif single_camera:
        from tasks.humanoid_velocity.experiments import _CAMERA_PATTERN_SINGLE
        cam_kw = {"reach_camera_actuator_names": (_CAMERA_PATTERN_SINGLE,)}
    else:
        cam_kw = {}
    env_cfg, _exp = build_configured_env_cfg(
        task, num_envs=1, device=device, reach=True, reach_nominal=True,
        reach_actuated_gripper=True, reach_gripper_robot_cfg=reach_gripper_robot_cfg, **cam_kw)
    # Physics arm starts at the SAME planning-home pose cuRobo plans from (descriptor map).
    _apply_planning_home_init_state(env_cfg, planning_home_joint_pos)
    if has_camera:
        # mjlab has no init_state entry for the actuated head gimbals, so they default to 0 (level) and the
        # cameras stare at the horizon on the FIRST ticks -- the frozen loco policy then has to servo them
        # down. Seed the WELDED downtilt (yaw 0, pitch WELDED_CAM_DOWNTILT_RAD) so the head INITIALIZES aimed
        # at a table, matching the kinematic welded mount and the policy's gaze rest. v2/v2_single slew off
        # this baseline; v2_fixed/v2_single_fixed hold it (their camera_ref pin targets the same pose).
        from asset_zoo.humanoid_v21.humanoid_v21_constants import WELDED_CAM_DOWNTILT_RAD
        if single_camera:
            _apply_planning_home_init_state(env_cfg, {
                "cam_yaw": 0.0, "cam_pitch": WELDED_CAM_DOWNTILT_RAD})
        else:
            _apply_planning_home_init_state(env_cfg, {
                "cam_yaw_left": 0.0, "cam_pitch_left": WELDED_CAM_DOWNTILT_RAD,
                "cam_yaw_right": 0.0, "cam_pitch_right": WELDED_CAM_DOWNTILT_RAD})
    # Keep arm_ref as frozen-RL observation, but bypass its residual at physical arm actuators. This is
    # dynamic-harness-only: normal reach/training environments retain learned residual arm control. Dict
    # insertion order is action-application order, so direct_arm is appended after joint_pos_arms and wins
    # on their shared actuator slots.
    from tasks.legs_only_task import ExternalJointPositionActionCfg
    arm_action = env_cfg.actions["joint_pos_arms"]
    env_cfg.actions["direct_arm"] = ExternalJointPositionActionCfg(
        entity_name="robot", actuator_names=tuple(arm_action.actuator_names), use_default_offset=True)
    env_cfg.episode_length_s = 3600.0
    if walk:
        # Un-gag the twist channel (verified recipe; cfg-local). See the ``walk`` docstring above.
        tw_cfg = env_cfg.commands["twist"]
        tw_cfg.resampling_time_range = (1.0e9, 1.0e9)   # resample timer never fires
        tw_cfg.rel_standing_envs = 0.0                  # no standing zero-mask
        tw_cfg.rel_heading_envs = 0.0                   # no heading wz override
        env_cfg.curriculum.pop("command_vel", None)     # keep (0,0) ranges: reset draws = 0
        env_cfg.terminations.pop("velocity_tracking_failure", None)
    env_cfg.scene.spec_fn = make_scene_spec_fn(scenario, base_spec_fn=env_cfg.scene.spec_fn)

    # WE own the ground: strip mjlab's TerrainEntity so mjlab supplies only the RL robot/policy, and add
    # our OWN plane (same ``add_ground_plane`` the kinematic harness uses). ``Scene.env_origins`` falls
    # back to zeros when ``terrain is None`` (scene.py) and base ``scene.xml`` carries no ground, so our
    # plane becomes the sole floor. Reach path already ran flat (plane_flag), so this is contact-compatible.
    env_cfg.scene.terrain = None

    def _spec_fn_stand_qpos0(spec, _base=env_cfg.scene.spec_fn):
        _base(spec)
        add_ground_plane(spec, with_grid=_floor_grid_enabled())   # SAME plane as kinematic (single source of the floor)
        if collision_spheres is not None:           # SAME planner spheres as kinematic (group 5, mjlab-namespaced)
            add_planner_collision_geoms(spec, collision_spheres, ns="robot/")
        b = spec.body(f"robot/{base_body}")
        b.pos = [b.pos[0], b.pos[1], stand_z]

    env_cfg.scene.spec_fn = _spec_fn_stand_qpos0
    env_cfg.sim.nconmax = 1024
    env_cfg.sim.njmax = 2048
    return env_cfg


# Frozen low-level policy per robot for the dynamic (phase-4) harness. The policy IS the robot: it turns
# the arm reference into joint actions physics integrates. v2/v2_fixed share the DUAL-camera humanoid
# policy; v2_single/v2_single_fixed share the SINGLE-camera humanoid policy (a separately trained
# checkpoint -- the single-cam robot exposes 2 camera obs/action dims instead of 4, so the dual-camera
# net's shapes do not fit it). g1 has its own (StudentOnly, fixed-camera -- the dynamic loop drops the
# camera command anyway).
_DYNAMIC_TASK = {
    "v2":              "v2_best",
    "v2_fixed":        "v2_best",
    "v2_single":       "v2_best_single",
    "v2_single_fixed": "v2_best_single",
    "g1":              "g1_best",
}

def _pinned_checkpoint(robot: str) -> pathlib.Path | None:
    """Absolute path to the in-repo weights behind the reported dynamic numbers for ``robot``.

    Single source of truth is ``dyn_sweep.PINS``; this only resolves it against the repo root.
    Imported lazily because dyn_sweep is otherwise a stdlib-only dispatcher and must not pull
    mujoco/cuRobo in -- and because this path is reached only when ``runs/`` came back empty.
    """
    from tasks.visual_manipulation.test.dyn_sweep import PINS
    rel = PINS.get(robot)
    return _REPO_ROOT / rel if rel else None


def _dynamic_stand(robot: str):
    """Per-robot phase-4 policy attributes ``(base_body_name, base_z, has_camera)``. ``base_body``/``base_z``
    = the standing pose the constraint-buffer sizing must stand at (local per-robot asset imports keep the
    kinematic path free of them). ``has_camera`` = the frozen policy has actuated head-camera gimbals; g1 is
    FIXED-camera, so its reach env must NOT inject the ``camera_ref`` gaze command (it references
    non-existent ``cam_*`` actuators and the command manager rejects it). Humanoid stands at HOME (base_link,
    0.59) with a gaze; g1 stands knees-bent (pelvis, 0.76 -- its init_state) with no gaze."""
    if robot == "g1":
        from asset_zoo.g1.g1_constants import G1_BASE_BODY, KNEES_BENT_KEYFRAME
        return G1_BASE_BODY, KNEES_BENT_KEYFRAME.pos[2], False
    from asset_zoo.humanoid_v21.humanoid_v21_constants import BASE_BODY_NAME, HOME_KEYFRAME
    return BASE_BODY_NAME, HOME_KEYFRAME.pos[2], True


class DynamicHarness:
    """Dynamic (physics) sim harness: realizes the policy's commands through a FROZEN RL low-level
    policy + MuJoCo/warp physics on the SAME Phase-3 scene. See module docstring for the loop contract.

    The frozen policy IS the robot: it converts the arm reference into low-level joint actions the
    physics integrates. A verdict FAIL therefore means physics/RL broke a kinematically-valid plan
    (the KinematicHarness would have passed the identical route). ``base_twist`` is asserted ~0 for
    now (parked reach; the reach env keeps a standing base) -- base motion under physics is DRIFT the
    replan tracks, not commanded twist. Camera command is ignored until gaze lands.

    ``camera``: ``False`` (default) = PRIVILEGED, ``observe`` returns EVERY cube (GT); ``True`` = the
    visibility gate, mirroring ``KinematicHarness`` -- the belief is only the cubes the head cameras SEE.
    The live warp physics qpos is read into a scratch ``MjData`` (``wp_data.qpos`` -> ``mj_forward``), then
    the SAME ``cube_visible_poses`` / ``gaze_sees`` seam runs, so the dynamic camera gate is identical to
    the kinematic one -- only the qpos source (physics vs ``mj_forward``) differs.
    """

    def __init__(self, robot: str, scenario_name: str, device: str = "cuda:0",
                 camera: bool = False, walk: bool = False, seed: int | None = None,
                 checkpoint: str | None = None) -> None:
        import torch  # local: keep the kinematic path torch-free at import

        assert robot in _DYNAMIC_TASK, f"no frozen phase-4 policy for {robot!r}"
        cfg, _object_cfg, nominal_scenario, table_z = robot_scene(robot, scenario_name)
        self.cfg = cfg
        self.scenario_name = scenario_name
        self._nominal_scenario = nominal_scenario
        self.scenario = nominal_scenario
        self.object_cfg = _object_cfg
        self.table_z = table_z
        self.device = device
        self.camera = bool(camera)
        self._walk = bool(walk)
        self._torch = torch

        task = _DYNAMIC_TASK[robot]
        base_body, stand_z, has_camera = _dynamic_stand(robot)
        if checkpoint is None:
            from run import find_latest_checkpoint
            checkpoint = find_latest_checkpoint(task)
        if checkpoint is None:
            # runs/ is gitignored, so it is empty on a fresh clone and resolving through it alone
            # would leave the dynamic harness unrunnable for anyone who has not trained first. Fall
            # back to the in-repo pinned weights -- the same ones behind the reported numbers. A
            # locally trained checkpoint still wins: whoever just trained the task means to evaluate
            # what they trained, so this is consulted only when the lookup came back empty.
            pin = _pinned_checkpoint(robot)
            if pin is not None and pin.is_file():
                print(f"[harness] runs/{task} empty; falling back to pinned {pin.name}")
                checkpoint = str(pin)
        assert checkpoint is not None, (
            f"no checkpoint under runs/{task}/*/model_*.pt and no pinned fallback for {robot!r} "
            f"(expected {_pinned_checkpoint(robot)})")
        # cuRobo planner collision spheres for the viewer group-5 overlay (SAME dict + SAME
        # add_planner_collision_geoms as the kinematic harness). build_robot_cfg_dict is CPU-only (no CUDA),
        # so this does NOT force the lazy planner session's cuRobo warmup.
        collision_spheres = cfg.build_robot_cfg_dict(
            cfg.planning_home_joint_pos)["robot_cfg"]["kinematics"]["collision_spheres"]
        env_cfg = _dynamic_env_cfg(
            nominal_scenario, device, task, base_body, stand_z, has_camera, walk=self._walk,
            reach_gripper_robot_cfg=_reach_gripper_robot_cfg(robot),
            planning_home_joint_pos=cfg.planning_home_joint_pos,
            collision_spheres=collision_spheres,
            single_camera=robot in ("v2_single", "v2_single_fixed"),
        )
        # ``--seed`` must govern the physics/RL reset too, not only cube XY.  The reach path is nominal,
        # but MjLab still initializes Torch/Warp RNG state while it constructs reset/command managers.
        # Leaving cfg.seed=None made identical layouts execute from unrelated RNG states across processes.
        env_cfg.seed = seed
        self.env = _reach_eval_env_cls()(cfg=env_cfg, device=device, robot_cfg=cfg,
                                         policy_task=task, policy_checkpoint=checkpoint)
        self.env.reset(seed=seed)
        # The reach/walk verify grades physical capture + upright; it never reads the RL training reward.
        # Computing every reward term each control step was 26% of the viewer loop's CPU (py-spy, v2
        # bimanual_mixed_close) for a scalar that is discarded. Zero the buffer instead. Patched on THIS
        # INSTANCE only -- never on the shared RewardManager class, which training still needs (contrast
        # mjlab_util.patched_reward_manager, which deliberately patches the class). Termination and
        # curriculum are untouched, so timeouts, fall resets and weight schedules behave identically; the
        # viewer's reward-term panel would read zeros, but neither viewer displays it.
        _reward_buf = self.env.reward_manager._reward_buf
        self.env.reward_manager.compute = lambda dt: _reward_buf.zero_()
        self.mj_model = self.env.sim.mj_model
        # mjlab's terrain generator adds a shadow-casting mjLIGHT_DIRECTIONAL to the "terrain"
        # body AFTER the robot spec is built, so add_fov_frustum_hull's spec-level castshadow
        # clear never reaches it. Left on, revealing the translucent FOV hull (key "4") slabs the
        # floor -- the shadow pass ignores alpha and has no per-geom opt-out. Kinematic harness
        # needs no equivalent: its spec carries only robot lights, already cleared at the source.
        self.mj_model.light_castshadow[:] = 0
        self.mj_model.mat_reflectance[:] = 0    # floor mirror doubles the hull's tinted area
        # ``move_pick_geoms``/``_rebuild_figure_geoms`` mutate ``geom_pos``/``geom_quat`` for the offered
        # figure on every seeded reset. mjlab's native viewer decides whether to do a full model resync or
        # a cheap qpos-only one via ``has_visual_dr = bool(sim.expanded_fields & VIEWER_MODEL_FIELDS)``
        # (mjlab/viewer/native/viewer.py) -- without registering these two fields as "expanded" up front,
        # that flag never goes True, so the viewer keeps refreshing the cube (qpos-driven, always synced)
        # but never re-uploads the welded figure's position -- it displays the FIRST-ever pose forever,
        # regardless of how many correct resets happen underneath. Calling this once here (num_envs=1, so
        # the per-env expansion is trivial) is the mechanism mjlab itself uses for domain randomization;
        # doing it here beats a raw ``viewer.sync()`` call reached in from the verify script, which
        # depended on undocumented native ``state_only`` semantics that turned out not to gate this.
        self.env.sim.expand_model_fields(("geom_pos", "geom_quat"))
        entity = self.env.scene["robot"]
        relax_limit = float(os.environ.get(_RELAX_ARM_LIMIT_ENV, "0.0"))
        if relax_limit > 0.0:
            self._relax_physics_arm_limits(entity, relax_limit)
        # Dynamic ``joint_pos`` is already a compact Torch tensor. Keep its named representation instead of
        # mirroring the complete Warp qpos buffer to CPU every 20 ms; the tracker only measures arm joints.
        self._proprio_joint_names = tuple(n.rsplit("/", 1)[-1] for n in entity.joint_names)
        # Walk drives a scripted base twist; grab the (now un-gagged) velocity command term so ``realize``
        # can write ``vel_command_b`` each tick. Parked reach never touches it (twist stays pinned to 0).
        self._twist = self.env.command_manager.get_term("twist") if self._walk else None
        # Scratch MjData for the camera gate: the live belief is computed by reading the warp physics qpos
        # into it + mj_forward, then running the SAME visibility seam as the kinematic harness. Built only
        # when the gate is on (privileged GT path never touches it).
        # Pick cubes are free bodies, so both privileged and camera-gated belief must read their LIVE
        # physics poses from the warp qpos snapshot. The scratch also hosts visibility/gaze queries.
        self._scratch = mujoco.MjData(self.mj_model)
        # ``observe`` refreshes this from Warp before policy/action processing.  Weld handling runs before
        # that tick's physics advance, so it can reuse the exact same state instead of forcing a second
        # GPU-to-CPU synchronization and MuJoCo FK pass.
        self._scratch_is_live = False
        # ``knock_witness`` bookkeeping: per-cube reference XY latched on first look, and the cubes already
        # reported (one report each, so the FIRST disturbance is the one attributed).
        self._knock_origin: dict[str, np.ndarray] = {}
        self._knock_seen: set[str] = set()
        self._max_penetration = 0.0     # mission-max robot-vs-scene interpenetration; see max_penetration
        # Device-resident mission energy integral; see ``accumulate_energy``. Zeroed on every reset.
        # float64: the host-side integral this replaces summed python floats, and 600+ float32
        # accumulations would drift from it in the reported digits.
        self._energy_j = torch.zeros((), device=device, dtype=torch.float64)
        # Per-phase split of the same integral (same device-resident, single-sync-at-end pattern), keyed by
        # the caller's ``policy.walk_phase`` string. Lazily created per phase name on first use.
        self._phase_energy_j: dict[str, torch.Tensor] = {}
        # ``session`` is built LAZILY (mirror of ``KinematicHarness.__init__``): the dynamic harness path
        # is fundamentally the SAME harness, only the ``realize`` seam differs. The MPC path passes
        # ``session=None`` (the spawned worker owns the cuRobo runtime); the sync path demands a real
        # session. Build-on-first-access keeps the warmup paid ONCE, only when needed, instead of freezing
        # the parent for ~7 s before ``env.reset`` even runs.
        self._session = _PLANNER_SESSION_UNINIT
        self._cfg = cfg
        self.control_dt = float(self.env.step_dt)
        # In-progress weld snaps (see ``_update_weld_latch``): {eq_id: {"start_pos": world xyz at
        # trigger tick, "t": elapsed s}}. Eased out over ``_WELD_EASE_S`` instead of writing the
        # designed grasp-center offset in one instant tick (looked like a teleport).
        self._weld_ease: dict[int, dict] = {}
        # Equality latch is an assist after physical gripper contact, never a substitute for one. Accumulate
        # rack/finger contact only while a jaw remains commanded closed; a wrist/forearm touch must not
        # satisfy a grasp verdict. Contact solver may expose one jaw of a real pinch per tick, so do not
        # require simultaneous contacts from both opposing racks.
        self._weld_contacted_racks: dict[int, set[str]] = {}
        # Consecutive-tick counter of "commanded open while weld active", per eq_id. See
        # ``_WELD_RELEASE_DEBOUNCE_TICKS``.
        self._weld_open_ticks: dict[int, int] = {}
        # v2_fixed/v2_single_fixed welded-camera fix. The DYNAMIC model is built from the ACTUATED
        # humanoid asset (live cam_yaw[_left/right]/cam_pitch[_left/right] joints), but the "fixed"
        # robots drive NO gaze (cfg.gaze_cams == ()), so the frozen loco policy leaves the "fixed"
        # cameras ~level -- they stare at the horizon and never frame a table cube ~70 deg below the
        # lens, so the belief is empty and the walk mission gives up instantly (21-step TERMINATE). The
        # KINEMATIC fixed asset bakes the D435 downtilt into the mount and deletes the joints; reproduce
        # it on the physics model by pinning the live pitch joint(s) to WELDED_CAM_DOWNTILT_RAD (yaw 0 ->
        # dual left +x/front, right -x/back; single +x/front only), STATICALLY (never aimed per-cube),
        # which keeps the fixed-camera semantics -- NOT a gimbal. None for g1 (single welded cam, no
        # gimbal joints, tilted in-asset) and v2/v2_single (drive their own actuated gaze in the
        # gaze_cams branch).
        # One-shot offered-object displacement state; armed per reset by ``_arm_live_handover``.
        self._live: dict | None = None
        self._welded_cam_gaze = None
        _single_cam_joint = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cam_pitch") != -1
        _dual_cam_joint = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cam_pitch_left") != -1
        if not cfg.gaze_cams and _single_cam_joint:
            from asset_zoo.humanoid_v21.humanoid_v21_constants import WELDED_CAM_DOWNTILT_RAD
            self._welded_cam_gaze = {"cam_pitch": WELDED_CAM_DOWNTILT_RAD, "cam_yaw": 0.0}
        elif not cfg.gaze_cams and _dual_cam_joint:
            from asset_zoo.humanoid_v21.humanoid_v21_constants import WELDED_CAM_DOWNTILT_RAD
            self._welded_cam_gaze = {"cam_pitch_left": WELDED_CAM_DOWNTILT_RAD,
                                     "cam_pitch_right": WELDED_CAM_DOWNTILT_RAD,
                                     "cam_yaw_left": 0.0, "cam_yaw_right": 0.0}

    @property
    def session(self):
        """Lazy planner session. The MPC viewer path never touches it; the sync / kinematic path pays the
        one-time cuRobo warmup here (mirror of ``KinematicHarness.session``)."""
        if self._session is _PLANNER_SESSION_UNINIT:
            self._session = make_planner_session(self._cfg)
        return self._session

    def _relax_physics_arm_limits(self, entity, limit_rad: float) -> None:
        """Temporarily widen active-arm physics limits for an A/B tracking diagnosis.

        Enabled only by ``DYNAMIC_RELAX_ARM_LIMIT_RAD``. cuRobo keeps its real URDF limits, so planned
        routes do not gain artificial reach; this changes only the Warp/MuJoCo constraints that could clip
        an otherwise valid direct-arm target during physical execution. Never use for reported results.
        """
        import warp as wp

        arm_names = {name.rsplit("/", 1)[-1] for name in self.env._arm_ref.target_names}
        model_ids = []
        entity_ids = []
        for entity_id, full_name in enumerate(entity.joint_names):
            name = full_name.rsplit("/", 1)[-1]
            if name not in arm_names:
                continue
            model_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, full_name)
            if model_id == -1:
                model_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert model_id != -1, f"active arm joint {name!r} absent from MuJoCo model"
            model_ids.append(model_id)
            entity_ids.append(entity_id)
        assert len(model_ids) == len(arm_names), "some direct-arm targets were not resolved"
        assert model_ids == list(dict.fromkeys(model_ids)), "direct-arm targets resolved duplicate MuJoCo joints"
        self.mj_model.jnt_range[model_ids] = (-limit_rad, limit_rad)
        wp_range = wp.to_torch(self.env.sim.wp_model.jnt_range)
        wp_range[:, model_ids, 0] = -limit_rad
        wp_range[:, model_ids, 1] = limit_rad
        entity.data.joint_pos_limits[:, entity_ids, 0] = -limit_rad
        entity.data.joint_pos_limits[:, entity_ids, 1] = limit_rad
        entity.data.soft_joint_pos_limits[:, entity_ids, 0] = -limit_rad
        entity.data.soft_joint_pos_limits[:, entity_ids, 1] = limit_rad
        print(f"[diagnostic] relaxed {len(model_ids)} active arm physics limits to "
              f"±{limit_rad:.3f} rad; cuRobo limits unchanged")

    def reset(self, seed: int | None = None) -> None:
        """Reset physics, then install the current cell's free-cube layout into live Warp state.

        ``env.reset()`` restores Warp's initialization snapshot, which is made before a
        per-trial seed is known. Updating only ``mj_model.qpos0`` before that reset therefore
        left physics at the nominal layout while planning used the jittered scenario. Keep the
        model copy for cuRobo/static hand supports, then mirror both free-joint qpos and static
        geom local positions into Warp after reset. This keeps kinematic and dynamic trials on
        the same sampled scene.
        """
        import mujoco_warp as mjwarp
        import warp as wp

        self.env.reset(seed=seed)
        self._weld_ease.clear()
        self._weld_contacted_racks.clear()
        self._weld_open_ticks.clear()
        self._energy_j.zero_()          # energy is per-mission, and each reset starts a new one
        self._phase_energy_j.clear()
        self._knock_origin.clear()      # seed jitter moves spawn XY, so the reference must be re-latched
        self._knock_seen.clear()
        self._max_penetration = 0.0
        if seed is None:
            self._arm_live_handover()   # historical no-seed path: layout untouched
            return
        self.scenario = jitter_pick_markers(self._nominal_scenario, seed)
        move_pick_geoms(self.mj_model, self.scenario)
        qpos = wp.to_torch(self.env.sim.wp_data.qpos)
        # A reseed only needs the static push when the scenario carries a hand figure whose geoms the
        # jitter rebuilt; table-only props are invariant, so the copy would be pure cost.
        if self.scenario.hand:
            self._push_static_geoms_to_warp(qpos.device, qpos.dtype)
        for marker in self.scenario.pick:
            qadr = self._pick_qposadr(marker.name)
            qpos[0, qadr:qadr + 7] = self._torch.as_tensor(
                self.mj_model.qpos0[qadr:qadr + 7], device=qpos.device, dtype=qpos.dtype)
        mjwarp.forward(self.env.sim.wp_model, self.env.sim.wp_data)
        live = wp.to_torch(self.env.sim.wp_data.qpos)[0].detach().cpu().numpy()
        for marker in self.scenario.pick:
            qadr = self._pick_qposadr(marker.name)
            assert np.allclose(live[qadr:qadr + 3], marker.pos), (
                f"dynamic reset lost sampled pose for {marker.name}: "
                f"live={live[qadr:qadr + 3]}, sampled={marker.pos}")
        self._arm_live_handover()

    def _pick_qposadr(self, cube_name: str) -> int:
        """Free-joint qpos address of a pick cube's body. Names are mjlab-namespaced, hence ``_ns_id``."""
        body_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"pp_obj_{cube_name}")
        assert body_id != -1, f"pick body pp_obj_{cube_name} missing from compiled model"
        return int(self.mj_model.jnt_qposadr[int(self.mj_model.body_jntadr[body_id])])

    def _push_static_geoms_to_warp(self, device, dtype) -> None:
        """Mirror ``mj_model.geom_pos``/``geom_quat`` for world-welded geoms into the Warp model + data.

        Warp computes static-world geom transforms once in ``make_data`` and ``forward`` deliberately never
        refreshes them, so a moved static geom needs BOTH the model parameter and the cached ``geom_xpos``/
        ``geom_xmat``. Skipping the cached data leaves a seeded (or, with the live handover, a displaced)
        cube resting on a palm collider that is still at the previous pose, and the object falls through it.
        Table-only trials never call this: their props are invariant, so the copy would be pure cost.
        """
        import warp as wp

        wp.to_torch(self.env.sim.wp_model.geom_pos)[...] = self._torch.as_tensor(
            self.mj_model.geom_pos, device=device, dtype=dtype)
        wp.to_torch(self.env.sim.wp_model.geom_quat)[...] = self._torch.as_tensor(
            self.mj_model.geom_quat, device=device, dtype=dtype)
        self._scratch.qpos[:] = self.mj_model.qpos0
        mujoco.mj_forward(self.mj_model, self._scratch)
        static_geom_ids = np.flatnonzero(self.mj_model.geom_bodyid == 0)
        wp.to_torch(self.env.sim.wp_data.geom_xpos)[:, static_geom_ids] = self._torch.as_tensor(
            self._scratch.geom_xpos[static_geom_ids], device=device, dtype=dtype)
        wp.to_torch(self.env.sim.wp_data.geom_xmat)[:, static_geom_ids] = self._torch.as_tensor(
            self._scratch.geom_xmat[static_geom_ids].reshape(-1, 3, 3), device=device, dtype=dtype)
        # The GPU LOS batch holds a SEPARATE warp snapshot of the same mj_model, so it needs the same
        # push -- otherwise every sight line is tested against the figure's pose at batch-construction
        # time. No-op while LOS runs on the CPU (no batch registered).
        refresh_warp_los_geoms(self.mj_model, self._scratch)
        self._scratch_is_live = False   # the scratch now holds qpos0, not the live physics state

    def _arm_live_handover(self) -> None:
        """Latch the offered figure + its cube, then DISPLACE both to the pre-move pose the trial starts at.

        No-op unless ``LIVE_HANDOVER_SHIFT_M`` is set and the scenario has a hand, which keeps every
        published cell byte-identical. The whole figure is captured by geom-name prefix (``pp_<marker>_``):
        the move is a RIGID rotation of the person, so there is no need to separate the offering arm from
        the rest of the body -- the earlier plan's 14-geom offering-arm list was only needed for a
        deformation, which this is not.

        Direction is chosen per cell so the pre-move pose lies TOWARD the robot's front (rotating the palm's
        bearing toward world +x). Both endpoints then sit inside the reach envelope the published cells
        already validated, and the ramp ends exactly at the published pose -- so a robot that never saw the
        move reaches a pose 0.08 m stale, while a robot that saw it reaches the validated one.

        The palm collision proxy is recorded separately because its quaternion must stay identity at every
        angle: ``ik_curobo._box_world_aabb`` asserts box frames are axis-aligned. The torso capsule needs no
        such exemption -- its axis is vertical, which yaw leaves alone.
        """
        self._live = None
        shift = float(os.environ.get(_LIVE_HANDOVER_ENV, "0.0") or 0.0)
        if shift == 0.0 or not self.scenario.hand:
            return
        assert len(self.scenario.hand) == 1, (
            f"live handover expects exactly one offered hand, got {[m.name for m in self.scenario.hand]}")
        hand = self.scenario.hand[0]
        # Chord -> angle at this cell's own handoff radius, so the metre displacement is what stays fixed.
        radius = float(np.linalg.norm(np.asarray(hand.pos)[:2]))
        assert 2.0 * radius > shift, (
            f"{hand.name} sits {radius:.3f} m from the scene origin; a {shift:.3f} m chord is unreachable "
            f"by rotating about it")
        yaw = 2.0 * float(np.arcsin(shift / (2.0 * radius)))
        # Toward the robot's front: rotate the palm's bearing toward world +x, so neither endpoint of the
        # ramp is further out laterally than the published pose already is.
        yaw *= -1.0 if np.arctan2(hand.pos[1], hand.pos[0]) > 0.0 else 1.0
        assert _LIVE_HANDOVER_TRIGGER_TICK + _LIVE_HANDOVER_RAMP_TICKS <= _PREACT_HOLD_TICKS, (
            f"live handover would still be moving at tick "
            f"{_LIVE_HANDOVER_TRIGGER_TICK + _LIVE_HANDOVER_RAMP_TICKS} but the policy stops holding at "
            f"{_PREACT_HOLD_TICKS}: set REACH_PREACT_HOLD_TICKS so the move completes before the first "
            f"commit, or cuRobo replans against a moving obstacle")
        prefix = f"pp_{hand.name}_"
        ids = np.array([g for g in range(self.mj_model.ngeom)
                        if (mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                        .startswith(prefix)], dtype=int)
        assert ids.size, f"no geoms named {prefix}* -- the offered figure is not in this model"
        flat_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}collision")
        carried = [m for m in self.scenario.pick if m.pos_follows_hand == hand.name]
        assert len(carried) == 1, (
            f"expected exactly one cube resting in {hand.name}, found {[m.name for m in carried]}")
        body_id = self.mj_model.body(f"pp_obj_{carried[0].name}").id
        qadr = int(self.mj_model.jnt_qposadr[int(self.mj_model.body_jntadr[body_id])])
        self._live = {
            "yaw": yaw, "ids": ids, "flat_id": flat_id, "cube": carried[0].name, "qadr": qadr,
            "pos0": self.mj_model.geom_pos[ids].copy(), "quat0": self.mj_model.geom_quat[ids].copy(),
            "tick": 0,
        }
        self._set_live_handover_angle(yaw, carry_cube=True)   # trial STARTS one chord back along the arc

    def _set_live_handover_angle(self, theta: float, carry_cube: bool, moving: bool = False) -> np.ndarray:
        """Place the offered figure at absolute angle ``theta`` about world z through the SCENE ORIGIN.

        ``theta`` is measured from the PUBLISHED palm pose, so ``theta = 0`` restores it exactly. The figure's
        geoms are world-welded (no body, no joint), so physics cannot move them and the pose has to be written
        into the model and mirrored to Warp.

        The cube IS a free body, so ``carry_cube`` carries it along: its LIVE position is moved by the same
        increment (relative, not absolute, so whatever settling or contact drift physics already produced is
        carried along instead of undone) and given the matching linear velocity, because a teleported palm
        imparts no friction and the cube would otherwise be left behind or flicked off by the contact solver.
        Returns the cube's new world XY. Pass ``moving = False`` on the final tick -- the person stops, and
        leaving velocity in would launch the cube off the palm the instant the drive releases.

        THE CUBE'S ORIENTATION IS NEVER DRIVEN, even though the palm under it rotates. Imposing the palm's
        yaw on the cube would leave the ``on`` arm's cube at the published POSITION but yawed 11.5-18 deg
        while the ``off`` arm's is axis-aligned -- different grasp geometry, so the ablation would no longer
        compare observation histories alone. Measured on the first pass of this sweep: the yawed cube made
        the front/back cell EASIER (5/10 -> 9/10), inverting the sign of the effect being measured. Contact
        friction may still spin it slightly, which is physics rather than an imposed asymmetry.
        """
        import warp as wp

        live = self._live
        ids, pos0, quat0 = live["ids"], live["pos0"], live["quat0"]
        yaw_q = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
        c, s = np.cos(theta), np.sin(theta)
        self.mj_model.geom_pos[ids] = pos0 @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
        for row, gid in enumerate(ids):
            if gid == live["flat_id"]:
                continue        # square axis-aligned proxy: translate only, never rotate (see _arm_live_handover)
            spun = np.zeros(4)
            mujoco.mju_mulQuat(spun, yaw_q, quat0[row])
            self.mj_model.geom_quat[gid] = spun

        qpos_t = wp.to_torch(self.env.sim.wp_data.qpos)
        qvel_t = wp.to_torch(self.env.sim.wp_data.qvel)
        self._push_static_geoms_to_warp(qpos_t.device, qpos_t.dtype)
        if not carry_cube:
            return None
        qadr = live["qadr"]
        cube = qpos_t[0, qadr:qadr + 7].detach().cpu().numpy().astype(float)
        step = theta - live.get("theta", 0.0)
        cs, ss = np.cos(step), np.sin(step)
        nxt = np.array([cs * cube[0] - ss * cube[1], ss * cube[0] + cs * cube[1], cube[2]])
        qpos_t[0, qadr:qadr + 3] = self._torch.as_tensor(nxt, device=qpos_t.device, dtype=qpos_t.dtype)
        # Free-joint qvel is [linear(3), angular(3)]; angular stays zero (orientation is not driven).
        vel = (nxt[:2] - cube[:2]) / self.control_dt if moving else np.zeros(2)
        twist = np.array([vel[0], vel[1], 0.0, 0.0, 0.0, 0.0])
        vadr = int(self.mj_model.jnt_dofadr[int(self.mj_model.body_jntadr[
            self.mj_model.body(f"pp_obj_{live['cube']}").id])])
        qvel_t[0, vadr:vadr + 6] = self._torch.as_tensor(twist, device=qvel_t.device, dtype=qvel_t.dtype)
        live["theta"] = theta
        # Carry the knock-witness reference along with the drive. Without this the deliberate handover motion
        # trips ``knock_witness`` on its 5th ramp tick (measured: "cube_1 shift=0.014m touching=['NONE']"),
        # which both consumes the cube's one-shot report and reports a knock that never happened -- and knock
        # rate is a real graded quantity here, not decoration.
        if live["cube"] in self._knock_origin:
            self._knock_origin[live["cube"]] = nxt[:2].copy()
        return nxt[:2]

    def _live_handover_step(self) -> None:
        """Walk the offered object from its pre-move pose HOME to the published pose, one control tick.

        The angle is a pure function of the tick counter, so the disturbance is bit-identical across robots --
        the whole point of the ablation is that the perturbation is shared and only the response differs.
        Outside the ramp this costs one integer compare. The ramp is asserted (in ``_arm_live_handover``) to
        finish inside the policy's pre-act hold, so no plan is ever live while this runs.
        """
        if self._live is None:
            return
        live = self._live
        tick, live["tick"] = live["tick"], live["tick"] + 1
        if not _LIVE_HANDOVER_TRIGGER_TICK <= tick < _LIVE_HANDOVER_TRIGGER_TICK + _LIVE_HANDOVER_RAMP_TICKS:
            return
        done = tick - _LIVE_HANDOVER_TRIGGER_TICK + 1
        theta = live["yaw"] * (1.0 - done / _LIVE_HANDOVER_RAMP_TICKS)
        self._set_live_handover_angle(theta, carry_cube=True,
                                      moving=done != _LIVE_HANDOVER_RAMP_TICKS)

    def settle(self, steps: int = 100) -> None:
        """Hold the planning-home arm through physics so the base + arm reach steady state BEFORE the
        first plan (invariant 5): plan_pose is seed-sensitive and planning from a transient start pose
        can flip a reachable target infeasible. ``steps=100`` is 2 s at the 50 Hz control cadence."""
        for _ in range(steps):
            self.env.step_reach(None)

    def _observed_cubes(self) -> dict:
        """The cubes the cams FRAME at the CURRENT realized gaze this tick (LIVE, non-accrued). Perception is
        a pure sensor now -- the POLICY owns gimbal aim + belief accrual (seam inversion). The live warp
        physics qpos is read into the scratch ``MjData`` (``wp_data.qpos`` -> ``mj_forward``), so ``detect_at_gaze``
        reads whatever the PHYSICAL head points at RIGHT NOW (the frozen loco policy has been servoing the
        gimbals toward the policy's ``camera_command``, applied in ``realize``). Fixed cams use the static
        ``cube_visible_poses``; the welded-downtilt pin is STATIC camera extrinsics the harness still owns."""
        import warp as wp  # local: only the physics-backed camera gate needs the warp readback
        d = self._scratch
        d.qpos[:] = wp.to_torch(self.env.sim.wp_data.qpos)[0].detach().cpu().numpy()
        mujoco.mj_forward(self.mj_model, d)
        self._scratch_is_live = True
        # GPU LOS path: lazy-build the _WarpBatch on the inner env (one per harness, per Decision 5),
        # refresh its warp qpos to live, and register with the scene module. The env's per-harness
        # batch is the only place that owns the appropriate warp model; passing it to the scene module
        # via set_warp_los() avoids touching every LOS caller's signature.
        if self.cfg.gaze_cams:
            inner = self.env
            if inner._warp_batch is None:
                from asset_zoo.reachability_study.gpu_visibility import _WarpBatch
                qpos_init = self._torch.tensor(
                    d.qpos, dtype=self._torch.float32, device="cuda:0").view(1, -1)
                inner._warp_batch = _WarpBatch(self.mj_model, qpos_init)
                from tasks.visual_manipulation.curobo.scene import set_warp_los, _set_los_cpu_occluder_set
                set_warp_los(enabled=True, batch=inner._warp_batch)
                _set_los_cpu_occluder_set(self.mj_model)
            # Per-tick FK refresh from live warp qpos (Decision 5: ray_clear uses wd FK state).
            # The harness uses num_envs=1 and the warp batch was built with nworld=1, so
            # `wp_data.qpos` and `_warp_batch.wd.qpos` share shape [1, nqpos] -> a direct
            # GPU-side wp.copy skips the CPU->GPU roundtrip that _WarpBatch.forward does
            # via torch.as_tensor + wp.from_torch.
            wp.copy(inner._warp_batch.wd.qpos, self.env.sim.wp_data.qpos)
            inner._warp_batch.mjw.kinematics(
                inner._warp_batch.wm, inner._warp_batch.wd)
        if not self.camera:
            return cube_world_poses(self.mj_model, self.scenario, d)
        if self.cfg.gaze_cams:
            return detect_at_gaze(self.mj_model, d, self.scenario, self.cfg)
        if self._welded_cam_gaze is not None:
            # v2_fixed: the physics model's "fixed" cams sit on live joints the loco policy leaves ~level.
            # Pin them to the welded downtilt in the scratch (so the frustum test sees the tilted lens) AND
            # drive the real gimbals to the same STATIC pose (so the physical cameras + rendered frustums
            # point at the table). Static extrinsics == harness-owned (not a policy gaze command).
            for joint, angle in self._welded_cam_gaze.items():
                d.qpos[_ns_joint_qadr(self.mj_model, joint)] = angle
            mujoco.mj_forward(self.mj_model, d)
            self.env.set_camera(self._welded_cam_gaze)
        return cube_visible_poses(self.mj_model, d, self.scenario, self.cfg)

    def observe(self):
        """Return named joint measurements for dynamics, full qpos for the kinematic harness.

        ``joint_pos`` is already exposed by the physics entity. Naming its compact columns lets the policy
        select route arms without a second GPU-to-CPU qpos mirror, while avoiding the invalid assumption that
        compact columns equal MuJoCo ``jnt_qposadr`` indices.
        """
        entity = self.env.scene["robot"]
        pos = entity.data.root_link_pos_w[0].detach().cpu().numpy().astype(float)
        quat = entity.data.root_link_quat_w[0].detach().cpu().numpy().astype(float)
        q = entity.data.joint_pos[0].detach().cpu().numpy()
        proprio = dict(zip(self._proprio_joint_names, q.tolist(), strict=True))
        return (pos, quat), proprio, self._observed_cubes()

    def accumulate_energy(self, phase: str | None = None) -> None:
        """Integrate this tick's mechanical power into the running mission energy total (J), ON DEVICE.

        Power = sum of |actuator generalized force . joint velocity| over all DOF. Uses ``qfrc_actuator``
        (DOF-space, aligned with ``joint_vel``'s ``joint_v_adr`` indexing), NOT ``actuator_force``
        (actuation-space; misaligned index-wise when gearing/tendons make actuator count != DOF count).

        WHICH ticks get integrated is the CALLER's decision: both grades in ``curobo_reach_verify.py`` skip
        this on ticks stalled on an off-loop cuRobo solve, so the total is what a synchronous planner would
        have burned. Calling it unconditionally is what makes a joule total scale with GPU load.

        The integral stays a device-resident scalar. The former ``joint_power() -> float`` forced a
        GPU-to-CPU sync every control step (~640 per walk mission) to build a total that only the final
        METRICS2 line reads; ``mission_energy_j`` now pays exactly one sync, at the end. Same value --
        the same per-tick rectangles, summed in the same order, just not materialized on the host.

        ``phase``, when given (``policy.walk_phase`` at this same tick), also adds the identical rectangle
        into a per-phase accumulator (``mission_phase_energy_j``), mirroring how ``phase_ticks`` splits
        duration in ``curobo_reach_verify.py``. Same device-resident, single-sync-at-end pattern -- summing
        into a second scalar costs one extra device add, not a host round trip.
        """
        entity = self.env.scene["robot"]
        tau = entity.data.qfrc_actuator[0]
        qvel = entity.data.joint_vel[0]
        power = (tau * qvel).abs().sum().double() * self.control_dt
        self._energy_j += power
        if phase is not None:
            if phase not in self._phase_energy_j:
                self._phase_energy_j[phase] = self._torch.zeros((), device=power.device, dtype=self._torch.float64)
            self._phase_energy_j[phase] += power

    def mission_energy_j(self) -> float:
        """Mechanical energy (J) accumulated since the last ``reset``. One GPU-to-CPU sync."""
        return float(self._energy_j)

    def mission_phase_energy_j(self) -> dict[str, float]:
        """Per-phase mechanical energy (J) since the last ``reset``. One GPU-to-CPU sync per phase seen;
        empty if ``accumulate_energy`` was never called with a ``phase``. Values sum to ``mission_energy_j``
        exactly (same rectangles, split by tick instead of pre-summed)."""
        return {phase: float(v) for phase, v in self._phase_energy_j.items()}

    def tool_poses_base(self):
        """Live base-frame pose of every configured tool frame for Cartesian tracking feedback."""
        return self.env.tool_poses_base(self.cfg.tool_frames)

    def drive_base(self, base_twist) -> None:
        """Write the policy's walk twist into the un-gagged command term BEFORE the step so this step's obs
        carry it, and clear the standing/heading auto-masks (mjlab re-rolls them on resample). The frozen
        loco policy then tracks this twist -- same recipe as the proven mission. Walk-only (asserts ~0 on
        the parked path). Shared by ``realize`` (headless ``run_reach``) and the mjlab ``_run_view`` seam
        (which owns ``env.step`` itself, so it must drive the base here rather than through ``realize``)."""
        vx, vy, wz = base_twist
        if not self._walk:
            assert abs(vx) + abs(vy) + abs(wz) < 1e-6, (
                f"dynamic harness reach base is standing; nonzero base_twist {base_twist} not wired yet")
            return
        self._twist.vel_command_b[0, 0] = vx
        self._twist.vel_command_b[0, 1] = vy
        self._twist.vel_command_b[0, 2] = wz
        self._twist.is_standing_env[:] = False
        self._twist.is_heading_env[:] = False

    def _update_weld_latch(self, gripper_closed: set[str] | None) -> None:
        """Manages dynamic equality weld constraints between hand base and target pick cubes."""
        import torch
        import warp as wp

        closed_set = set()
        if gripper_closed is not None:
            for item in gripper_closed:
                item_str = str(item).lower()
                if "l" in item_str or "left" in item_str:
                    closed_set.add("L")
                if "r" in item_str or "right" in item_str:
                    closed_set.add("R")

        if not closed_set:
            # Open jaws cannot acquire. Preserve the old release semantics without a live-state readback.
            for arm_prefix in ("L", "R"):
                for marker in self.scenario.pick:
                    eq_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                   f"pp_weld_{arm_prefix}_{marker.name}")
                    if eq_id != -1 and bool(self.env.sim.data.eq_active[0, eq_id].item()):
                        self.env.sim.data.eq_active[0, eq_id] = False
                    self._weld_ease.pop(eq_id, None)
                    self._weld_contacted_racks.pop(eq_id, None)
            return

        d = self._scratch
        if not self._scratch_is_live:
            d.qpos[:] = wp.to_torch(self.env.sim.wp_data.qpos)[0].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_model, d)
            self._scratch_is_live = True

        for idx, arm_prefix in enumerate(("L", "R")):
            hand_body = f"robot/{arm_prefix}_base"
            tool_frame = self.cfg.tool_frames[idx]
            tcp_site = f"robot/{tool_frame}" if _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{tool_frame}") != -1 else tool_frame

            for m in self.scenario.pick:
                cube_name = m.name
                eq_name = f"pp_weld_{arm_prefix}_{cube_name}"
                eq_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                if eq_id == -1:
                    continue

                cube_body = f"pp_obj_{cube_name}"
                cube_geom = f"pp_mark_{cube_name}"

                if arm_prefix in closed_set:
                    self._weld_open_ticks.pop(eq_id, None)
                    active = bool(self.env.sim.data.eq_active[0, eq_id].item())
                    easing = eq_id in self._weld_ease
                    if active and not easing:
                        continue    # already welded at its final (rigid) offset, nothing to update

                    tcp_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
                    cube_body_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, cube_body)
                    if tcp_id == -1 or cube_body_id == -1:
                        continue
                    p_tcp = d.site(tcp_id).xpos
                    p_cube = d.body(cube_body_id).xpos

                    if not active:
                        # Tight trigger: a real rack/finger contact plus small TCP gap while the jaw stays
                        # closed. The old ``L_``/``R_`` test admitted wrist/forearm contact and could weld
                        # an empty-looking hand. A pinch often reports one jaw at a time, so requiring both
                        # rack contacts simultaneously rejects valid physical grasps. History resets on open.
                        dist = np.linalg.norm(p_tcp - p_cube)
                        contacted_racks = self._weld_contacted_racks.setdefault(eq_id, set())
                        rack_names = {
                            f"{arm_prefix}_left_rack": "left",
                            f"{arm_prefix}_right_rack": "right",
                        }
                        for i in range(d.ncon):
                            c = d.contact[i]
                            g1_name = self.mj_model.geom(c.geom1).name
                            g2_name = self.mj_model.geom(c.geom2).name
                            if cube_geom not in (g1_name, g2_name):
                                continue
                            other_name = g2_name if g1_name == cube_geom else g1_name
                            for rack_name, rack_side in rack_names.items():
                                if rack_name in other_name:
                                    contacted_racks.add(rack_side)
                        if not contacted_racks or dist >= _WELD_TRIGGER_DIST_M:
                            continue
                        # Start eased at the cube's OWN current offset IN THE HAND FRAME (zero perceived jump
                        # on this tick); ramp toward the designed grasp-center offset over ``_WELD_EASE_S``.
                        # Stored hand-relative, not as a world point: see the interpolation below.
                        b1_ease = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
                        if b1_ease == -1:
                            b1_ease = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{arm_prefix}_base")
                        if b1_ease == -1:
                            continue
                        R1_ease = np.zeros(9)
                        mujoco.mju_quat2Mat(R1_ease, d.body(b1_ease).xquat)
                        start_rel = R1_ease.reshape(3, 3).T @ (p_cube - d.body(b1_ease).xpos)
                        self._weld_ease[eq_id] = {"start_rel": start_rel, "t": 0.0}
                        self.env.sim.data.eq_active[0, eq_id] = True

                    b1_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
                    if b1_id == -1:
                        b1_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"{arm_prefix}_base")
                    if b1_id == -1:
                        continue

                    ease = self._weld_ease[eq_id]
                    ease["t"] += self.control_dt
                    alpha = min(1.0, ease["t"] / _WELD_EASE_S)
                    smooth = alpha * alpha * (3.0 - 2.0 * alpha)   # smoothstep, C1-continuous at both ends
                    # Target is the gripper's DESIGNED grasp-center point (``p_tcp``, rigid -- no joint
                    # between hand_body and the tcp site -- constant regardless of tracking error), NOT
                    # the cube's literal observed offset: else a dynamic-tracker residual (the drift
                    # servo's Cartesian integral is clamped to 1 cm, see JacobianReachTracker) gets frozen
                    # into the weld forever, leaving a permanent visible gap.
                    #
                    # Interpolate IN THE HAND FRAME. The ease used to blend a WORLD point captured at latch
                    # toward the live ``p_tcp``; because the hand keeps moving during the 0.15 s ramp (jaw
                    # close, then RETRACT), the early-alpha target was a world-fixed location the hand had
                    # already left, so the weld dragged the cube toward a stale spot in the room and then
                    # whipped it back onto the grasp center -- read on video as the cube suddenly getting
                    # attracted to the hand. Both endpoints hand-relative makes the ramp a pure offset
                    # correction that moves with the gripper.
                    p1 = d.body(b1_id).xpos
                    q1 = d.body(b1_id).xquat
                    q2 = d.body(cube_body_id).xquat

                    R1 = np.zeros(9)
                    mujoco.mju_quat2Mat(R1, q1)
                    R1 = R1.reshape(3, 3)
                    tcp_rel = R1.T @ (p_tcp - p1)
                    rel_pos = ease["start_rel"] + smooth * (tcp_rel - ease["start_rel"])

                    q1_inv = np.zeros(4)
                    mujoco.mju_negQuat(q1_inv, q1)
                    rel_quat = np.zeros(4)
                    mujoco.mju_mulQuat(rel_quat, q1_inv, q2)

                    self.mj_model.eq_data[eq_id, 0:3] = [0, 0, 0]
                    self.mj_model.eq_data[eq_id, 3:6] = rel_pos
                    self.mj_model.eq_data[eq_id, 6:10] = rel_quat
                    self.mj_model.eq_data[eq_id, 10] = 1.0

                    wp_eq_data = wp.to_torch(self.env.sim.wp_model.eq_data)
                    wp_eq_data[0, eq_id, 0:3] = 0.0
                    wp_eq_data[0, eq_id, 3:6] = torch.tensor(rel_pos, device=wp_eq_data.device, dtype=torch.float32)
                    wp_eq_data[0, eq_id, 6:10] = torch.tensor(rel_quat, device=wp_eq_data.device, dtype=torch.float32)
                    wp_eq_data[0, eq_id, 10] = 1.0

                    if alpha >= 1.0:
                        del self._weld_ease[eq_id]
                else:
                    if bool(self.env.sim.data.eq_active[0, eq_id].item()):
                        open_ticks = self._weld_open_ticks.get(eq_id, 0) + 1
                        if open_ticks < _WELD_RELEASE_DEBOUNCE_TICKS:
                            self._weld_open_ticks[eq_id] = open_ticks
                            continue    # transient open tick on an active weld -- not a real release yet
                        self._weld_open_ticks.pop(eq_id, None)
                        self.env.sim.data.eq_active[0, eq_id] = False
                    self._weld_ease.pop(eq_id, None)

        if os.environ.get("REACH_DEBUG_XYZ"):
            self._debug_xyz_tick = getattr(self, "_debug_xyz_tick", 0) + 1
            for idx, name in enumerate(("cube_0", "cube_1")):
                bid = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"pp_obj_{name}")
                eq_state = []
                for side in ("L", "R"):
                    eq_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, f"pp_weld_{side}_{name}")
                    if eq_id != -1:
                        active = bool(self.env.sim.data.eq_active[0, eq_id].item())
                        easing = eq_id in self._weld_ease
                        open_ticks = self._weld_open_ticks.get(eq_id, 0)
                        racks = self._weld_contacted_racks.get(eq_id, set())
                        dist_str = "n/a"
                        tool_frame = self.cfg.tool_frames[0 if side == "L" else 1]
                        tcp_site = f"robot/{tool_frame}" if _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{tool_frame}") != -1 else tool_frame
                        tcp_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
                        if tcp_id != -1 and bid != -1:
                            dist_str = f"{np.linalg.norm(d.site(tcp_id).xpos - d.body(bid).xpos):.4f}"
                        eq_state.append(f"{side}:active={active},easing={easing},open_ticks={open_ticks},"
                                         f"dist={dist_str},racks={sorted(racks)}")
                if bid != -1:
                    xpos = d.body(bid).xpos
                    # Realized vs commanded driven-rack position: a genuine grasp shows the rack STALLED
                    # short of its target (cube blocking the jaw), which is the only reliable "the cube is
                    # actually in the hand" readout -- contact force is not (see _WELD_TRIGGER_DIST_M).
                    print(f"[xyz] tick={self._debug_xyz_tick} {name} z={xpos[2]:.4f} "
                          f"below_table={xpos[2] < self.table_z} xyz={xpos} "
                          f"rack={self.env.gripper_qpos()} tgt={self.env._gripper_target[0].tolist()} "
                          f"gripper_closed={gripper_closed} eq=[{' '.join(eq_state)}]", flush=True)

    def prepare_reach_action(self, base_twist, camera_command,
                             gripper_closed: set[str] | None = None) -> None:
        """Apply high-level command side effects before one physics action.

        Headless execution calls this then ``step_reach``. The native viewer calls this then returns its
        action to the viewer-owned ``env.step``. Keeping this pre-step sequence shared prevents camera,
        gripper, or weld-latch behavior from diverging between ``--view`` and no-view execution.
        """
        self._live_handover_step()
        self.drive_base(base_twist)
        if camera_command:
            # Policy owns the gimbal aim now (actuated robots): drive the physical head to it via camera_ref;
            # the frozen loco policy servos there and next ``observe`` detects at the realized gaze. Empty for
            # fixed cams (their static welded extrinsics are pinned in ``_observed_cubes``).
            self.env.set_camera(camera_command)
        if gripper_closed is not None:
            self.env.set_gripper(gripper_closed)
        self._update_weld_latch(gripper_closed)

    def realize(self, base_twist, arm_reference, camera_command, gripper_closed: set[str] | None = None) -> None:
        """Prepare one shared reach action, then advance one headless physics tick."""
        self.prepare_reach_action(base_twist, camera_command, gripper_closed)
        self.env.step_reach(arm_reference)
        self._scratch_is_live = False

    def physical_grasps(self) -> dict[str, str]:
        """Return side-to-cube mappings whose contact-triggered weld is active.

        This is the dynamic policy's grasp acknowledgement. It intentionally reads only equality state, not
        distance: policy may retire a target only after physical finger contact caused the latch to engage.
        ``eq_active`` is set True at that trigger instant (see ``_update_weld_latch``'s tight rack-contact +
        TCP-gap trigger); the subsequent ``_weld_ease`` ramp only smooths the weld's offset math, it is not a
        re-verification of grasp integrity, so it must NOT gate this. Gating on it used to let
        ``_GRIPPER_CLOSE_MAX_S`` expire on an already-latched arm before its short ease finished, wrongly
        excluding it from ``_held``/``gripper_closed`` and dropping the just-formed weld.
        """
        captured = {}
        for side in ("L", "R"):
            for marker in self.scenario.pick:
                eq_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, f"pp_weld_{side}_{marker.name}")
                if eq_id != -1 and bool(self.env.sim.data.eq_active[0, eq_id].item()):
                    captured[side] = marker.name
                    break
        return captured

    def fallen_cubes(self) -> set[str]:
        """Return pick cubes whose center sits more than ``_CUBE_FALL_DROP_M`` below its spawn height.

        Spawn Z comes from the scenario marker (seed jitter is XY-only), so no reset bookkeeping is needed.
        Reads the shared ``_scratch`` snapshot under the same lazy-refresh guard ``_update_weld_latch`` uses,
        so on a tick where the weld pass already refreshed it this costs no extra GPU readback.

        A cube whose weld is active is EXCLUDED: transport legitimately carries it below spawn height, and
        condemning it would fail a grasp that physically succeeded.
        """
        import warp as wp

        d = self._scratch
        if not self._scratch_is_live:
            d.qpos[:] = wp.to_torch(self.env.sim.wp_data.qpos)[0].detach().cpu().numpy()
            mujoco.mj_forward(self.mj_model, d)
            self._scratch_is_live = True
        welded = set(self.physical_grasps().values())
        fallen = set()
        for marker in self.scenario.pick:
            if marker.name in welded:
                continue
            body_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"pp_obj_{marker.name}")
            if body_id == -1:
                continue
            if float(marker.pos[2]) - float(d.body(body_id).xpos[2]) > _CUBE_FALL_DROP_M:
                fallen.add(marker.name)
        return fallen

    def knock_witness(self) -> list[str]:
        """Name the geom that first disturbs each non-welded pick cube, once per cube.

        ``fallen_cubes`` fires only AFTER the cube has already dropped ``_CUBE_FALL_DROP_M``, by which point
        the contact that pushed it is long gone -- so it cannot attribute the 'cube knocked off support'
        verdict to a link. This watches planar displacement instead and reports on the FIRST tick a cube has
        moved more than ``_KNOCK_WITNESS_M`` from where this run found it, listing every geom then touching
        it. That is the attribution: an arm link, a finger, the torso, or the sibling cube.

        Diagnostic only -- returns strings, mutates no simulation state, and no FSM reads it. Assumes the
        caller already refreshed ``_scratch`` this tick (``fallen_cubes`` does), so contacts come free from
        that ``mj_forward``; welded cubes are skipped because transport moves them legitimately.
        """
        d = self._scratch
        welded = set(self.physical_grasps().values())
        out = []
        for marker in self.scenario.pick:
            if marker.name in welded or marker.name in self._knock_seen:
                continue
            body_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"pp_obj_{marker.name}")
            if body_id == -1:
                continue
            xy = np.asarray(d.body(body_id).xpos[:2], dtype=float)
            origin = self._knock_origin.setdefault(marker.name, xy.copy())
            shift = float(np.linalg.norm(xy - origin))
            if shift < _KNOCK_WITNESS_M:
                continue
            self._knock_seen.add(marker.name)
            touching = []
            for i in range(d.ncon):
                g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
                b1, b2 = int(self.mj_model.geom_bodyid[g1]), int(self.mj_model.geom_bodyid[g2])
                other = g2 if b1 == body_id else (g1 if b2 == body_id else None)
                if other is None:
                    continue
                name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, other)
                touching.append(name or f"geom{other}")
            out.append(f"{marker.name} shift={shift:.3f}m touching={touching or ['NONE']}")
        return out

    def max_penetration(self) -> float:
        """Deepest robot-vs-scene contact interpenetration seen so far this mission, metres (>= 0).

        The tracker has no per-tick collision term: safety rests on the cuRobo route being collision-free plus a
        bounded departure from it. Any change touching that departure therefore has to be judged on collision
        fidelity as well as on pass/fail, otherwise a tracking win that is really the arm pushing through a table
        reads as an improvement. Reports ``-min(contact.dist)`` over contacts pairing a robot geom with a
        non-robot one, so cube-vs-table and rack-vs-cube grasp contacts (which are legitimate and shallow) are
        included but robot self-contact is not. Reuses the caller's ``_scratch`` refresh like ``knock_witness``.
        """
        d = self._scratch
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
            n1 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
            n2 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
            if n1.startswith("robot/") == n2.startswith("robot/"):
                continue                                # both robot (self-contact) or neither (scene-only)
            self._max_penetration = max(self._max_penetration, -float(d.contact.dist[i]))
        return self._max_penetration

    def live_qpos(self) -> np.ndarray:
        """Current physics qpos read out of warp (same readback the camera gate uses in ``_observed_cubes``),
        so an offscreen renderer can draw the LIVE simulated pose. Minimal -- caller owns the ``MjData`` +
        renderer."""
        import warp as wp  # local: only the physics-backed record/gate paths need the warp readback
        return wp.to_torch(self.env.sim.wp_data.qpos)[0].detach().cpu().numpy()

    def verdict(self, route) -> float:
        """Realized (physics) tool-site error to the NEAREST route grasp candidate, WORLD frame. Same
        n-agnostic rule as the kinematic harness: reads the REAL simulated site pose for each ACTIVE
        frame and returns the WORST (a bimanual reach passes only if BOTH hands land)."""
        return _world_reach_error(route, self.env.reach_site_pos_w(route.reaches))

    def grasp_capture_errors(self, route) -> dict[str, float]:
        """Return live grasp-center error for each individually completed latch in ``route``.

        A bimanual route can capture one cube, safely retract it, then retry the other cube. Return the
        completed side independently so the evaluator does not erase that first physical capture merely
        because its paired side missed during the same close window. Dynamic cubes are free bodies, so this
        diagnostic compares live grasp site to live cube center after latch easing; it is not success logic.
        """
        import warp as wp

        d = self._scratch
        d.qpos[:] = wp.to_torch(self.env.sim.wp_data.qpos)[0].detach().cpu().numpy()
        mujoco.mj_forward(self.mj_model, d)
        errors = {}
        for side, cube_name in route.assignment.items():
            arm_prefix = str(side).upper()
            eq_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_EQUALITY,
                           f"pp_weld_{arm_prefix}_{cube_name}")
            if (eq_id == -1 or not bool(self.env.sim.data.eq_active[0, eq_id].item())
                    or eq_id in self._weld_ease):
                continue
            frame = self.cfg.tool_frames[("L", "R").index(arm_prefix)]
            tool_site = (f"robot/{frame}"
                         if _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{frame}") != -1
                         else frame)
            tool_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, tool_site)
            cube_id = _ns_id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"pp_obj_{cube_name}")
            assert tool_id != -1 and cube_id != -1, (frame, cube_name)
            errors[cube_name] = float(np.linalg.norm(d.site(tool_id).xpos - d.body(cube_id).xpos))
        return errors

    def grasp_capture_error(self, route) -> float:
        """Return worst live diagnostic only when every side in ``route`` has completed its latch.

        Parked reach remains one atomic route. Walk grading uses ``grasp_capture_errors`` above so a valid
        first-side capture survives a retry for the other side.
        """
        errors = self.grasp_capture_errors(route)
        if set(errors) != set(route.assignment.values()):
            return float("nan")
        return max(errors.values(), default=float("nan"))

    def upright(self) -> float:
        """Base up-axis vs world up (1.0 = perfectly upright); a phase-4 gate beyond reach_error."""
        entity = self.env.scene["robot"]
        quat = entity.data.root_link_quat_w[0].detach().cpu().numpy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=np.float64))
        return float(R.reshape(3, 3)[2, 2])


def make_harness(robot: str, scenario_name: str | tuple[str, ...], *, dynamic: bool = False,
                 camera: bool = False,
                 drift: bool = False, walk: bool = False, device: str = "cuda:0", height_pad: float = 0.02,
                 seed: int | None = None, checkpoint: str | None = None):
    """Build the harness the unified driver runs against, so ``run_reach`` stays harness-agnostic:
    ``DynamicHarness`` (physics + frozen RL) when ``dynamic``, else ``KinematicHarness`` (``mj_forward``).
    Both satisfy the observe/realize/verdict/upright/settle contract; only the ``realize`` seam (physics vs
    ``mj_forward``) differs. ``walk`` un-gags the base twist so the policy's walk-search can drive the base
    (dynamic: writes ``vel_command_b``; kinematic: integrated for free). ``drift`` is kinematic-only --
    under real physics the base motion IS the disturbance, so a scripted sinusoid would be redundant
    (rejected in ``main``, asserted here); ``walk`` and ``drift`` are mutually exclusive (both own the base)."""
    assert not (walk and drift), "--walk and --drift both own the base motion; pick one"
    if dynamic:
        assert not drift, "--drift is kinematic-only (under physics the base motion IS the disturbance)"
        return DynamicHarness(robot, scenario_name, device=device, camera=camera, walk=walk, seed=seed,
                              checkpoint=checkpoint)
    return KinematicHarness(robot, scenario_name, camera=camera, drift=drift, walk=walk, height_pad=height_pad)
