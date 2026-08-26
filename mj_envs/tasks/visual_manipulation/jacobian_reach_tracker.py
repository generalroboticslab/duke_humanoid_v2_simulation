"""Reactive drift-servo that tracks a plan-0 reach WITHOUT a per-tick optimizer.

Replaces the cuRobo MPC reactive tracker on the ``mpc_track`` policy path. plan-0 already solves a
collision-free arm trajectory (home -> grasp); at runtime the ONLY thing that changes is the
floating base drifting a few cm under arm reaction. Correcting a known-good path for small drift is one
damped-least-squares Jacobian step -- not a full trajectory re-optimization. So this class:

  * extracts the plan's WHOLE trajectory as an object-anchored task-space waypoint path (``path_in_object``,
    per active tool frame) via ONE-TIME MuJoCo forward kinematics of the plan configs, and
  * each control tick seeds at the planned config for the current cursor and takes ONE base-frame Jacobian
    step toward the drift-corrected waypoint goal (``pose_compose(objects_in_base[obj], waypoint)``).

Why this beats the MPC it replaces (measured this session):
  * SPEED -- the MPC ``optimize`` was a fixed ~16 ms/tick (collision-world rollout + CUDA-graph launch),
    independent of every solver knob, blowing the 20 ms budget. This servo is pure MuJoCo FK + a 6x6..12x12
    linear solve = sub-millisecond.
  * WEDGE -- the MPC was asked to reach a target ON the cube while treating the cube as a HARD obstacle to
    the same arm: a contradiction it resolved by parking the hand short of / pressed into the cube. This
    servo has NO per-tick collision term, so it simply reaches the grasp; contact with the target cube is
    the intended grasp, not a wedge. Collision safety comes from the nominal path already being
    collision-free plus small, bounded drift (see the scope note on the policy).

Deploy-faithful (no base-in-world in the loop): the per-tick input is ``objects_in_base = {name: (pos,
quat_wxyz)}`` (each object's base_link pose = the camera measurement on hardware). FK runs on a scratch
``mj_data`` with the floating-base root pinned to identity, so the scratch world == the base frame and no
base world pose is ever read.

Output = ``{mjlab_joint: qpos}`` over the active arms' controlled joints (the ``+qpos0`` ref fold applied),
exactly what ``DynamicArmReferenceExecutor`` / the harnesses consume.
"""

from __future__ import annotations

import enum
import os
import time

import mujoco
import numpy as np

from tasks.visual_manipulation.curobo.scene import pose_compose, pose_relative


class _FkStage(enum.Enum):
    """Pipeline stage the FK needs to advance the scratch to.

    Pick the cheapest that covers what the caller reads after the call:

    * ``KINEMATICS`` -> ``mj_kinematics`` only. Reads: ``site_xpos``, ``site_xmat``, ``xpos``,
      ``xmat``. Order ~0.005 ms.
    * ``JACOBIAN`` -> ``mj_kinematics`` + ``mj_comPos``. Adds ``cdof`` (and the rest of the
      composite-rigid-body state ``mj_jacSite``/``mj_jacBody`` consume). Required for the per-iter
      ``mj_jacSite`` in the IK loop; ``mj_kinematics`` alone leaves cdof stale and the
      Jacobian differs from the dynamics-pipeline value by ~0.14 per element.
    * ``PASSIVE`` -> ``JACOBIAN`` + ``mj_comVel`` + ``mj_passive`` + ``mj_rne(flg_acc=0)``, i.e. the
      body of ``mj_fwdVelocity``. Populates ``qfrc_bias``, which the gravity comp site reads. Skips
      collision/constraint/transmission/actuation. ~0.2-0.4 ms/tick saved over ``DYNAMICS`` there.
      ``mj_comVel`` and ``mj_rne`` are BOTH required: ``mj_passive`` writes ``qfrc_passive``
      (springs/dampers/fluid -- no gravity) and never touches ``qfrc_bias``, so omitting them left
      the caller reading the value the last full ``mj_forward`` wrote, at a DIFFERENT arm config and
      base tilt. Measured on a 3 kg two-link probe: 8.1 N*m of pure error in ``tau_g``, which the
      gravity bend then divides by ``kp`` and adds to the position reference every tick.
    * ``DYNAMICS`` -> ``mj_forward``. Adds collision, constraint, transmission, actuation, and
      the RNE forward passive force. Required for ``qfrc_bias`` when qvel != 0.

    Default ``DYNAMICS`` keeps every existing caller byte-identical; opt in at the call site.
    """
    KINEMATICS = "kinematics"
    JACOBIAN   = "jacobian"
    PASSIVE    = "passive"
    DYNAMICS   = "dynamics"

_JAC_DEBUG = bool(os.environ.get("JAC_DEBUG"))
# Diagnostic-only route-playback speed scale, see ``JacobianReachTracker.step``. 1.0 = unchanged.
_REACH_SPEED_SCALE = float(os.environ.get("DYNAMIC_REACH_SPEED_SCALE", "1.0"))
# Scale on the servo velocity feedforward (kd*qdot/kp). 0 disables it, restoring the gravity-only pre-bend.
_VEL_FF_SCALE = float(os.environ.get("VEL_FF_SCALE", "1.0"))
# A/B override of the per-embodiment ``dynamic_tracker_resid_ki``; unset = use the config value.
_RESID_KI_ENV = os.environ.get("RESID_KI")
# RETRACT route-playback speed scale, see ``JacobianReachTracker.return_step``. 1.0 = unchanged.
_RETURN_SPEED_SCALE = float(os.environ.get("DYNAMIC_RETURN_SPEED_SCALE", "1.0"))

# Damped-least-squares regularization (rad-scaled): trades tracking tightness vs conditioning near arm
# singularities. 0.05 keeps the correction crisp while staying stable; raise if the servo oscillates.
_DLS_LAMBDA = 0.05
# Per-INNER-ITER joint-step clamp (rad): stabilizes each damped-LS Newton step; the tick runs several such
# steps (``_IK_ITERS``) re-FK'ing between them, so the correction still converges near singularities. The
# clamp only bounds EACH step, not the tick total -- worst case (every iteration saturating the same
# direction) was 12*0.2=2.4 rad/tick, unbounded in practice near a true singularity (measured normal-tick
# raw_net_dq stays <=0.12 rad, so 0.05 costs normal ticks nothing while cutting the worst case to 0.6 rad).
_DQ_CLAMP = 0.05
# Inner damped-LS iterations per control tick (each = FK + one Newton step). A handful fully converges the
# small drift correction (seed = the planned config, so the goal is close); pure MuJoCo, sub-ms.
_IK_ITERS = 12
_IK_TOL_M = 0.002               # stop early only when every active tool is this close in position
_IK_TOL_RAD = 0.03              # and within 1.7 deg of planned tool orientation
# The live Cartesian correction has redundant arm DOFs (two for bimanual V2 routes, more for a single arm).
# Keep that null space near cuRobo's collision-checked waypoint instead of letting moving-base feedback drift
# into an arbitrary joint-limit branch. This acts only through ``I - J^+J``; Cartesian correction stays first.
_NULLSPACE_POSTURE_GAIN = 0.10
# Viewer overlay: draw every Nth planned waypoint as a live tool-frame triad (key waypoints, not all H).
_WP_SAMPLE = 4
# Dynamic-only outer Cartesian integral: correct steady physical tool error left by position-servo compliance
# and gravity-model mismatch. Small + leaky + tightly clamped because the nominal cuRobo route remains the
# collision-safety authority; this tracker has no per-tick collision solve.
_CARTESIAN_KI = 0.005
_CARTESIAN_I_DECAY = 0.995
_CARTESIAN_I_MAX_M = 0.01
# Return starts from a physically tracked grasp hold, not an assumed planner joint state. This bounds the
# Cartesian handoff into the asynchronously solved route without resetting physical or feedback state.
_RETURN_BRIDGE_MIN_S = 0.30
_RETURN_BRIDGE_MAX_SPEED_M_S = 0.05
_RETURN_BRIDGE_MAX_JOINT_SPEED_RAD_S = 1.0


def _lerp_pose(a, b, alpha: float):
    """Interpolate base-frame ``(position, quat_wxyz)`` poses without a frame discontinuity."""
    pa, qa = np.asarray(a[0], dtype=float), np.asarray(a[1], dtype=float)
    pb, qb = np.asarray(b[0], dtype=float), np.asarray(b[1], dtype=float)
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if dot > 0.9995:
        q = qa + alpha * (qb - qa)
        q /= np.linalg.norm(q)
    else:
        angle = np.arccos(dot)
        s = np.sin(angle)
        q = (np.sin((1.0 - alpha) * angle) * qa + np.sin(alpha * angle) * qb) / s
    return (pa + alpha * (pb - pa), q)


def _resolve_site(mj_model, name):
    """Site id for a bare or ``robot/<name>``-namespaced site (the env namespaces sites)."""
    for n in (name, f"robot/{name}"):
        i = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, n)
        if i != -1:
            return i
    raise KeyError(f"site {name!r} not in model")


class JacobianReachTracker:
    """MuJoCo damped-LS Jacobian drift-servo. Built once per policy; ``rebind`` re-points it at each new
    plan-0 route, ``step`` emits one control tick's arm reference."""

    def __init__(self, mj_model, tool_frames, control_dt: float,
                 gravity_comp: bool = True, gravity_comp_alpha: float = 1.0,
                 output_filter: bool = False, traj_omega: float = 15.0,
                 traj_max_dq: float = 10.0, traj_max_ddq: float = 40.0,
                 max_measured_correction_rad: float | None = None,
                 transit_max_correction_rad: float | None = None,
                 resid_ki: float = 0.0):
        self.mj_model = mj_model
        self.data = mujoco.MjData(mj_model)
        self.control_dt = float(control_dt)
        self.tool_frames = list(tool_frames)                       # (L_site, R_site), model order
        self._site_id = {f: _resolve_site(mj_model, f) for f in self.tool_frames}
        # Floating-base free joint -> pinned to identity every FK so scratch-world == base_link frame.
        self._root_qadr = next(int(mj_model.jnt_qposadr[j]) for j in range(mj_model.njnt)
                               if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
        # Dynamic proprioception names are bare while the MuJoCo model may namespace them.  The Jacobian
        # scratch state must include live waist/leg posture: resetting those joints to qpos0 changes tool
        # orientation even when measured arm joints are exact.
        self._qadr_by_joint = {}
        for j in range(mj_model.njnt):
            name = mj_model.joint(j).name
            if name is not None:
                self._qadr_by_joint[name] = int(mj_model.jnt_qposadr[j])
                self._qadr_by_joint.setdefault(name.split("/")[-1], int(mj_model.jnt_qposadr[j]))
        # Gravity-comp feedforward: pre-bend the emitted arm reference by the PD droop the frozen policy's
        # position servo suffers under gravity (steady state kp*(ctrl-q)=g(q) => arm sits g(q)/kp low).
        # Enabled by default for the physics/deploy tracker; callers without a PD gravity droop can explicitly
        # disable it.
        self._gravity_comp = bool(gravity_comp)
        # alpha=1.0 is MEASURED-optimal, not a default nobody checked: swept 0.25/0.5/0.75/1.0/1.25/1.5 on
        # v2/left_right_close/seed42, physical_pos_err is a clean U with its minimum AT 1.0 (median 17.6 /
        # 13.5 / 12.4 / 11.4 / 13.4 / 17.5 mm). Both directions are dead levers -- do not retune this to
        # chase tracking error.
        self._grav_alpha = float(gravity_comp_alpha)
        # Critically-damped output filter (ported from ik_mink._apply_trajectory_profile). The raw solved q
        # jumps each tick; feeding it straight to the position servo excites the floating base into a limit
        # cycle (extended arm -> gravity -> base rocks -> object-in-base shifts -> servo re-commands). The
        # filter is a 2nd-order chase (ddq = w^2*(q_cmd-q_filt) - 2w*dq_filt, ddq accel-clamped) that
        # low-passes the command AND actively damps its velocity (2w => zeta=1, no overshoot). max_ddq is the
        # base-disturbance budget (arm accel reacts on the base). OFF by default so phase-3 stays byte-
        # identical; only the dynamic (phase-4)/deploy path opts in. State (_traj_q/_traj_dq) is lazily seeded
        # from the first solved q in ``step`` (no start jerk) and reset each ``rebind``.
        self._filter = bool(output_filter)
        self._traj_omega_sq = float(traj_omega) ** 2
        self._traj_2omega = 2.0 * float(traj_omega)
        self._traj_max_dq = float(traj_max_dq)
        self._traj_max_ddq = float(traj_max_ddq)
        self._traj_q = None
        self._traj_dq = None
        # A dynamic embodiment can opt into a total per-tick correction cap. `_DQ_CLAMP` bounds one Newton
        # iteration, but twelve iterations can still select a distant IK branch in 20 ms. Bound against BOTH
        # measured joints and the collision-checked nominal waypoint: the first rejects branch jumps; the
        # second prevents accumulated moving-base correction from spending the route's joint-limit margin.
        self._max_measured_correction_rad = max_measured_correction_rad
        self._transit_max_correction_rad = transit_max_correction_rad
        # Joint-space integral on the MEASURED physical tool error, applied through the same cap-exempt
        # pre-bend channel as the gravity feedforward. The Cartesian integrator above it is clamped to
        # ``_CARTESIAN_I_MAX_M`` = 1 cm because it displaces the TARGET (growing it commands the gripper
        # past the cube, straight into the dominant knock failure class); measured steady error is 2.8x
        # that clamp single-arm and 7.8x bimanual, so the clamp cannot close it and must not be raised.
        # This term carries the residual instead: it is a PD reference offset, so saturating it merely
        # under-pushes toward the collision-checked plan rather than over-reaching past the target.
        # 0 = off, bit-identical to the gravity-only pre-bend.
        self._resid_ki = float(_RESID_KI_ENV) if _RESID_KI_ENV is not None else float(resid_ki)

    def _joint_addr(self, name):
        """(qposadr, dofadr) for a bare cuRobo joint name, suffix-resolved against the model."""
        for j in range(self.mj_model.njnt):
            n = self.mj_model.joint(j).name
            if n == name or n.split("/")[-1] == name:
                return int(self.mj_model.jnt_qposadr[j]), int(self.mj_model.jnt_dofadr[j])
        raise KeyError(f"route joint {name!r} not in model")

    def rebind(self, route, objects_plan):
        """Re-point at a NEW plan-0 ``route``. ``objects_plan = {name: (pos, quat_wxyz)}`` = plan-time
        base-frame cuboid poses (the object frame each grasp path is extracted relative to)."""
        self.active = list(route.reaches)                          # active tool frames
        self.controlled = list(route.controlled_joints)            # active arms' joints (output columns)
        self.frame_to_object = {self.tool_frames[0]: f"pp_obj_{route.assignment['L']}"} if "L" in route.assignment else {}
        if "R" in route.assignment:
            self.frame_to_object[self.tool_frames[1]] = f"pp_obj_{route.assignment['R']}"
        # Arm cspace: route columns are in route.joint_names order (all arm DOFs). Resolve addresses + fold.
        self.arm_joints = list(route.joint_names)
        addrs = [self._joint_addr(n) for n in self.arm_joints]
        self.arm_qadr = np.array([a[0] for a in addrs])
        self.arm_dofadr = np.array([a[1] for a in addrs])
        self.offset = self.mj_model.qpos0[self.arm_qadr].copy()    # +qpos0 ref fold (cspace -> mjlab)
        if self._gravity_comp:
            # kp per arm DOF, vectorized from the model: arm actuators are MuJoCo position servos
            # (torque = kp*(ctrl-q)-kd*qdot), so gainprm[:,0] IS kp. Scatter each actuator's kp onto its
            # DOF, then gather the arm DOFs. One-time; no per-joint python loop.
            m = self.mj_model
            kp_by_dof = np.zeros(m.nv)
            kp_by_dof[m.jnt_dofadr[m.actuator_trnid[:, 0]]] = m.actuator_gainprm[:, 0]
            self.arm_kp = kp_by_dof[self.arm_dofadr]
            # kd for the velocity feedforward, same scatter/gather as kp. A MuJoCo position servo is
            # tau = kp*(ctrl-q) - kd*qdot, so biasprm[:,2] IS -kd; joint damping acts on the same qdot and
            # adds to it. Measured kd/kp = 0.100 (shoulders) / 0.0667 (elbow, wrist).
            kd_by_dof = np.zeros(m.nv)
            kd_by_dof[m.jnt_dofadr[m.actuator_trnid[:, 0]]] = -m.actuator_biasprm[:, 2]
            self.arm_kd = kd_by_dof[self.arm_dofadr] + m.dof_damping[self.arm_dofadr]
            # Per-joint ceiling on the gravity pre-bend, in the same units the bend is applied.
            # The bend exists to make the servo hold tau_g = kp*(ctrl-q); the point where that
            # demand exceeds what the actuator can produce is forcerange/kp, and beyond it the
            # extra reference offset is pure windup the joint can never realize. That IS the
            # physical bound, so it replaces the flat scalar this used to clip against: 0.15 rad
            # sat 12x under the g1 shoulder's 1.75 rad capability and clipped 55-72% of the
            # compensation on 98.6% of arm configurations (g1 kp=14.25 needs 0.34 rad median),
            # leaving the measured 4-9 cm tool droop that ate the grasp latch. An unlimited
            # actuator reports forcerange 0; map it to inf, since a 0 ceiling would silently
            # disable gravity comp altogether rather than merely bound it.
            frc_by_dof = np.zeros(m.nv)
            frc_by_dof[m.jnt_dofadr[m.actuator_trnid[:, 0]]] = np.where(
                m.actuator_forcelimited, m.actuator_forcerange[:, 1], np.inf)
            self.arm_bend_max = frc_by_dof[self.arm_dofadr] / self.arm_kp
        self.route_q = np.asarray(route.route_q_curobo, dtype=float)   # (H, n_arm) cuRobo cspace
        self.H = int(self.route_q.shape[0])
        self.interpolation_dt = float(route.interpolation_dt)
        # One-time MuJoCo FK of the whole plan path -> per active frame a length-H tool pose in base frame,
        # then expressed relative to its plan-time object -> constant base-independent waypoint path.
        self.path_in_object = {f: [] for f in self.active}
        for k in range(self.H):
            self._fk(self.route_q[k], stage=_FkStage.KINEMATICS)
            for f in self.active:
                site = (self.data.site_xpos[self._site_id[f]].copy(), self._site_quat(f))
                self.path_in_object[f].append(pose_relative(objects_plan[self.frame_to_object[f]], site))
        self._elapsed = 0.0
        self._last_objects = None                                  # last observed objects_in_base (viewer overlay)
        self._pos_integral = {f: np.zeros(3) for f in self.active} # dynamic physical-tool feedback, base frame
        self._resid_bend = np.zeros(len(self.arm_joints))          # joint-space physical-error integral
        self._traj_q = None                                        # output-filter state: lazy-seed on first step
        self._traj_dq = None
        self._dbg_prev_raw_q = None                                 # JAC_DEBUG-only tick-over-tick raw q
        self._last_command = None                                  # held while asynchronous return planning runs
        self._last_desired_base = None                             # final tracked target, frozen at grasp
        self._hold_pose_base = None
        self._return_q = None
        self._return_path_base = None
        self._return_dt = 0.0
        self._return_elapsed = 0.0

    def _fk(self, q_arm_cspace, measured_qpos=None, stage: _FkStage = _FkStage.DYNAMICS):
        """Place scratch FK in live non-arm posture and candidate arm configuration.

        Root remains identity, so scratch world is base frame.  Dynamic arm tracking receives all measured
        joint positions, not only arms: live waist/leg articulation moves the arm mount and must be present
        before evaluating a base-frame tool orientation.  The arm candidate overrides its measurement.

        ``stage`` selects the cheapest pipeline that still serves the caller (see ``_FkStage``). Default
        ``DYNAMICS`` keeps the legacy behavior identical (full ``mj_forward``); the IK loop, rebind, and
        begin-return opt into ``JACOBIAN`` / ``KINEMATICS`` to skip collision / constraint / actuation
        stages they never read.
        """
        self.data.qpos[:] = self.mj_model.qpos0
        if measured_qpos is not None:
            for name, value in measured_qpos.items():
                qadr = self._qadr_by_joint.get(name)
                if qadr is not None:
                    self.data.qpos[qadr] = float(value)
        self.data.qpos[self._root_qadr:self._root_qadr + 3] = 0.0
        self.data.qpos[self._root_qadr + 3:self._root_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.arm_qadr] = np.asarray(q_arm_cspace, dtype=float) + self.offset
        if stage is _FkStage.DYNAMICS:
            mujoco.mj_forward(self.mj_model, self.data)
        elif stage is _FkStage.KINEMATICS:
            mujoco.mj_kinematics(self.mj_model, self.data)
        elif stage is _FkStage.JACOBIAN:
            mujoco.mj_kinematics(self.mj_model, self.data)
            mujoco.mj_comPos(self.mj_model, self.data)
        else:                                                          # _FkStage.PASSIVE
            mujoco.mj_kinematics(self.mj_model, self.data)
            mujoco.mj_comPos(self.mj_model, self.data)
            mujoco.mj_comVel(self.mj_model, self.data)
            mujoco.mj_passive(self.mj_model, self.data)
            mujoco.mj_rne(self.mj_model, self.data, 0, self.data.qfrc_bias)

    def _site_quat(self, frame):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.data.site_xmat[self._site_id[frame]])
        return q

    def _track_desired(self, desired, q_nom, gravity_in_base=None, actual_tool_poses_base=None,
                       measured_arm_qpos=None, _tag: str = "", qdot_nom=None):
        """Solve one bounded Cartesian tracking tick from measured arm state to base-frame tool targets."""
        desired = {f: (np.asarray(p[0], dtype=float).copy(), np.asarray(p[1], dtype=float).copy())
                   for f, p in desired.items()}
        physical_pos_err = physical_rot_err = 0.0
        # Measured tool error stacked into task-vector layout (6 rows per active frame, rotation rows left
        # zero) so it maps through the tick's own ``J_pinv`` without rebuilding the row index.
        e_phys_task = np.zeros(6 * len(self.active))
        if actual_tool_poses_base is not None:
            for k_f, f in enumerate(self.active):
                assert f in actual_tool_poses_base, f"missing measured tool pose for active frame {f!r}"
                e_pos = np.asarray(desired[f][0], dtype=float) - np.asarray(actual_tool_poses_base[f][0], dtype=float)
                physical_pos_err = max(physical_pos_err, float(np.linalg.norm(e_pos)))
                e_phys_task[6 * k_f:6 * k_f + 3] = e_pos
                neg, dq_quat, e_rot = np.zeros(4), np.zeros(4), np.zeros(3)
                mujoco.mju_negQuat(neg, np.asarray(actual_tool_poses_base[f][1], dtype=float))
                mujoco.mju_mulQuat(dq_quat, np.asarray(desired[f][1], dtype=float), neg)
                mujoco.mju_quat2Vel(e_rot, dq_quat, 1.0)
                physical_rot_err = max(physical_rot_err, float(np.linalg.norm(e_rot)))
                integral = self._pos_integral[f]
                integral *= _CARTESIAN_I_DECAY
                integral += _CARTESIAN_KI * e_pos
                np.clip(integral, -_CARTESIAN_I_MAX_M, _CARTESIAN_I_MAX_M, out=integral)
                desired[f] = (np.asarray(desired[f][0], dtype=float) + integral, desired[f][1])
        self._last_desired_base = {
            f: (np.asarray(p[0], dtype=float).copy(), np.asarray(p[1], dtype=float).copy())
            for f, p in desired.items()}
        _t0 = time.perf_counter() if _JAC_DEBUG else 0.0
        if measured_arm_qpos is None:
            q = q_nom.copy()
        else:
            try:
                q = np.asarray([measured_arm_qpos[j] for j in self.arm_joints], dtype=float) - self.offset
            except KeyError as exc:
                raise KeyError(f"missing measured qpos for route arm joint {exc.args[0]!r}") from exc
        q_measured = q.copy()
        jacp, jacr = np.zeros((3, self.mj_model.nv)), np.zeros((3, self.mj_model.nv))
        live_pos_err = live_rot_err = 0.0
        for _ in range(_IK_ITERS):
            self._fk(q, measured_arm_qpos, stage=_FkStage.JACOBIAN)
            rows, err, perr = [], [], []
            for f in self.active:
                p_fk = self.data.site_xpos[self._site_id[f]].copy()
                e_pos = np.asarray(desired[f][0], dtype=float) - p_fk
                e_rot, neg, dq_quat = np.zeros(3), np.zeros(4), np.zeros(4)   # base-frame rot err (world==base)
                mujoco.mju_negQuat(neg, self._site_quat(f))
                mujoco.mju_mulQuat(dq_quat, np.asarray(desired[f][1], dtype=float), neg)
                mujoco.mju_quat2Vel(e_rot, dq_quat, 1.0)
                mujoco.mj_jacSite(self.mj_model, self.data, jacp, jacr, self._site_id[f])
                rows.append(np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]]))
                err.append(np.concatenate([e_pos, e_rot]))
                perr.append(e_pos)
            J = np.vstack(rows)                                   # (6*nactive, n_arm)
            e = np.concatenate(err)
            live_pos_err = max(float(np.linalg.norm(error)) for error in perr)
            # Position often converges before wrist orientation.  Do not terminate the full 6-D solve
            # on position alone: dynamic measured-joint FK then holds an orientation error that kinematic
            # plan-0 never has.  ``mju_quat2Vel(..., 1)`` gives this residual directly in radians.
            live_rot_err = max(float(np.linalg.norm(error[3:])) for error in err)
            if live_pos_err < _IK_TOL_M and live_rot_err < _IK_TOL_RAD:
                break
            # Damped task correction plus a small null-space pull to the collision-checked cuRobo waypoint.
            # The earlier task-only solve left redundant DOFs unconstrained. Under base motion that wandered
            # into a joint-limit branch even while the tool target remained reachable. ``J_pinv`` is damped,
            # so the posture term stays bounded near singularities and cannot replace task tracking.
            J_pinv = J.T @ np.linalg.solve(J @ J.T + (_DLS_LAMBDA ** 2) * np.eye(J.shape[0]),
                                            np.eye(J.shape[0]))
            dq = J_pinv @ e
            dq += _NULLSPACE_POSTURE_GAIN * (np.eye(J.shape[1]) - J_pinv @ J) @ (q_nom - q)
            q = q + np.clip(dq, -_DQ_CLAMP, _DQ_CLAMP)
        _sigma_min = _raw_net_dq = 0.0
        if _JAC_DEBUG:
            # Diagnostic only, pre-gravity-comp/pre-filter: sigma_min = smallest singular value of the LAST
            # solved Jacobian (conditioning at the raw IK solution), raw_net_dq = tick-over-tick jump in the
            # raw solved q (before any smoothing) -- both meant to empirically characterize a "flip" tick
            # (candidate: sigma_min collapses, raw_net_dq spikes) vs normal ticks, before picking any threshold.
            sigma_min_sq = float(np.linalg.eigvalsh(J @ J.T)[0])
            _sigma_min = float(np.sqrt(max(sigma_min_sq, 0.0)))
            _dbg_prev_raw_q = getattr(self, "_dbg_prev_raw_q", None)
            _raw_net_dq = float(np.abs(q - _dbg_prev_raw_q).max()) if _dbg_prev_raw_q is not None else 0.0
            self._dbg_prev_raw_q = q.copy()
        _grav_bend = np.zeros_like(q)       # split out so JAC_DEBUG can size it against the correction cap
        _resid_bend = np.zeros_like(q)      # physical-error integral, same channel, reported separately
        if self._gravity_comp:
            # Pre-bend the reference by the PD gravity droop g(q)/kp so the position servo settles ON target
            # instead of g/kp low. tau_g = gravity generalized force at the solved config, read from
            # qfrc_bias after one FK with the scratch gravity set to the live base-frame direction (tilt) and
            # qvel=0 (so qfrc_bias is pure gravity, no Coriolis). alpha is a model/gain-mismatch screen;
            # direct_arm bypasses frozen RL's arm residual, so it is NOT compensating an RL contribution.
            # alpha=0 gives the un-compensated position servo.
            assert gravity_in_base is not None, "gravity_comp=True needs gravity_in_base in step()"
            g0 = self.mj_model.opt.gravity.copy()
            self.mj_model.opt.gravity[:] = np.asarray(gravity_in_base, dtype=float)
            self._fk(q, measured_arm_qpos, stage=_FkStage.PASSIVE)
            tau_g = self.data.qfrc_bias[self.arm_dofadr].copy()
            self.mj_model.opt.gravity[:] = g0
            # Velocity feedforward, the SECOND half of inverting the same position servo. Holding a MOVING
            # reference needs kp*(ctrl-q) = tau_g + kd*qdot, so the command must lead by tau_g/kp (above)
            # AND kd*qdot/kp -- only the first existed, leaving a purely velocity-proportional following
            # error. Measured on v2/left_right_close: physical_pos_err regresses as
            # 0.00589*qdot + 0.08174*grav_bend, i.e. ~4 mm of the p90 error is this missing term, and it
            # tracked route speed at r=0.669. ``qdot_nom`` is the PLANNED joint velocity (route finite
            # difference), not a measured derivative, so it carries no sensor noise into the command.
            # Summed with the gravity bend BEFORE the clip: both are torque demands on the same actuator,
            # so ``arm_bend_max`` = forcerange/kp bounds their SUM, not each in isolation -- clipping them
            # separately would let the pair request more torque than the joint can produce.
            _vel_ff = (_VEL_FF_SCALE * self.arm_kd / self.arm_kp) * qdot_nom if qdot_nom is not None else 0.0
            _grav_bend = np.clip(self._grav_alpha * tau_g / self.arm_kp + _vel_ff,
                                 -self.arm_bend_max, self.arm_bend_max)
            if self._resid_ki and actual_tool_poses_base is not None:
                # Residual physical-error integral, joint space. ``J`` is the last solved Jacobian of this
                # tick; re-damp it rather than reuse the loop's ``J_pinv``, which is undefined on a tick that
                # converged and broke before computing one. Budget is what ``arm_bend_max`` (= forcerange/kp,
                # the point past which reference offset is windup the actuator can never realize) has left
                # after the gravity bend: both are torque demands on the same actuator, so the ceiling bounds
                # their SUM.
                J_pinv_r = J.T @ np.linalg.solve(J @ J.T + (_DLS_LAMBDA ** 2) * np.eye(J.shape[0]),
                                                 np.eye(J.shape[0]))
                self._resid_bend += self._resid_ki * (J_pinv_r @ e_phys_task)
                budget = np.maximum(self.arm_bend_max - np.abs(_grav_bend), 0.0)
                np.clip(self._resid_bend, -budget, budget, out=self._resid_bend)
                _resid_bend = self._resid_bend.copy()
        _q_preclip = q.copy()
        if measured_arm_qpos is not None and self._max_measured_correction_rad is not None:
            np.clip(q, q_measured - self._max_measured_correction_rad,
                    q_measured + self._max_measured_correction_rad, out=q)
            nominal_cap = self._max_measured_correction_rad
            if _tag == "extend" and self._transit_max_correction_rad is not None:
                nominal_cap = min(nominal_cap, self._transit_max_correction_rad)
            np.clip(q, q_nom - nominal_cap,
                    q_nom + nominal_cap, out=q)
        # Gravity feedforward is added AFTER the caps, not before, and is deliberately exempt from them.
        # The caps bound how far the EXECUTED configuration may depart from the collision-checked cuRobo plan.
        # ``q`` here is a PD *reference*, and ``g(q)/kp`` is exactly the reference-vs-realized offset that makes
        # the realized configuration equal ``q_nom``. Clipping the pre-bend toward ``q_nom`` therefore does not
        # bound executed departure -- it CREATES one, of the droop it was computed to cancel. Measured on
        # v2/front_back_close/seed50: the pre-bend alone wants 0.10 rad against a 0.06 cap once the arm is
        # extended, so it was being deleted by itself before the task correction got any budget, and the left
        # tool parked ~0.09-0.14 m off target -- far outside the 0.015 m rack-vs-cube clearance, which is what
        # shoves the cube off its support. Adding it here makes the realized configuration land CLOSER to the
        # collision-checked plan, i.e. it strengthens the invariant the caps exist to protect rather than
        # relaxing it; the cap on the task correction is unchanged. Bounded by ``arm_bend_max`` so a
        # near-singular or mis-modelled configuration cannot inject an unbounded reference offset.
        q = q + _grav_bend + _resid_bend
        # A per-joint correction cap bounds joint deviation, but the task is Cartesian, so the SAME cap buys
        # a configuration-dependent amount of tool motion through J.  Measured in metres so it compares
        # directly against live_pos_err:
        #   cart_lost = ||J_pos @ (q - q_preclip)||. NOTE this is measured after the gravity/velocity
        #               pre-bend is added above, so it is cap-removal PLUS bend, and the bend dominates
        #               (0.13 rad median on v2/front_back_close vs a cap that never saturates there).
        #               Read n_sat, not cart_lost, to decide whether the cap is binding.
        #   cart_auth = best tool motion the cap can buy ALONG the current error direction u,
        #               cap * sum_j |u . J_pos[:,j]| (L1 over joints -- the box's support in u)
        # cart_auth < live_pos_err proves the cap cannot close the error regardless of which joints move.
        _cart_lost = _cart_auth = 0.0
        _n_sat = 0
        _cap = self._max_measured_correction_rad or 0.0
        if _JAC_DEBUG:
            if _tag == "extend" and self._transit_max_correction_rad is not None:
                _cap = min(_cap, self._transit_max_correction_rad) if _cap else self._transit_max_correction_rad
            pos_rows = np.concatenate([np.arange(6 * k, 6 * k + 3) for k in range(len(self.active))])
            J_pos = J[pos_rows]
            _cart_lost = float(np.linalg.norm(J_pos @ (q - _q_preclip)))
            e_pos_all = e[pos_rows]
            n_e = float(np.linalg.norm(e_pos_all))
            _cart_auth = _cap * float(np.abs((e_pos_all / n_e) @ J_pos).sum()) if n_e > 1e-9 else 0.0
            # Measure saturation on the PRE-BEND ``q``, which is what the caps actually clipped. Reading it
            # off the emitted ``q`` compared a cap-exempt pre-bend against the cap, so ``|q - q_nom|`` never
            # landed on ``_cap`` exactly and this counted 0 on every tick since the bend moved after the
            # caps -- the instrument this whole failure class is diagnosed with, silently dead.
            _q_capped = q - _grav_bend - _resid_bend
            _n_sat = int((np.abs(np.abs(_q_capped - q_nom) - _cap) < 1e-9).sum()) if _cap else 0
        if self._filter:
            # Critically-damped 2nd-order chase toward the solved q (cspace, pre-fold). Euler-integrate accel
            # then velocity; accel/velocity hard-clamped. Lazy-seed at the first solved q so tick 0 is a no-op.
            if self._traj_q is None:
                self._traj_q = q.copy()
                self._traj_dq = np.zeros_like(q)
            ddq = self._traj_omega_sq * (q - self._traj_q) - self._traj_2omega * self._traj_dq
            np.clip(ddq, -self._traj_max_ddq, self._traj_max_ddq, out=ddq)
            self._traj_dq += ddq * self.control_dt
            np.clip(self._traj_dq, -self._traj_max_dq, self._traj_max_dq, out=self._traj_dq)
            self._traj_q += self._traj_dq * self.control_dt
            q = self._traj_q.copy()
        if _JAC_DEBUG:
            # tilt_xy: base-frame gravity's horizontal components -- (0, 0) when level, nonzero grows with
            # base tilt. Diagnostic-only signal for whether a hold-phase drift/shake correlates with the base
            # reacting to the newly grasped mass (candidate root cause distinct from IK-reseed branch flip).
            g = np.asarray(gravity_in_base, dtype=float) if gravity_in_base is not None else np.zeros(3)
            print(f"[jac:{_tag}] live_pos_err={live_pos_err:.4f} live_rot_err={live_rot_err:.4f} "
                  f"physical_pos_err={physical_pos_err:.4f} physical_rot_err={physical_rot_err:.4f} "
                  f"corr_max={np.abs(q - q_nom).max():.4f} "
                  f"sigma_min={_sigma_min:.4f} raw_net_dq={_raw_net_dq:.4f} "
                  f"cap={_cap:.4f} n_sat={_n_sat} cart_lost={_cart_lost:.4f} cart_auth={_cart_auth:.4f} "
                  f"grav_bend={np.abs(_grav_bend).max():.4f} resid_bend={np.abs(_resid_bend).max():.4f} vel_ff={np.abs(_vel_ff).max() if np.ndim(_vel_ff) else 0.0:.4f} ik_corr={np.abs(_q_preclip - _grav_bend - q_nom).max():.4f} "
                  f"tilt_xy=({g[0]:.4f},{g[1]:.4f}) "
                  f"solve_ms={(time.perf_counter()-_t0)*1e3:.3f}", flush=True)
        mjlab = {n: float(q[i] + self.offset[i]) for i, n in enumerate(self.arm_joints)}
        self._last_command = {j: mjlab[j] for j in self.controlled}
        return dict(self._last_command)

    def step(self, objects_in_base, gravity_in_base=None, actual_tool_poses_base=None,
             measured_arm_qpos=None):
        """Track object-relative extend waypoints. ``freeze_hold`` switches later grasp control to base frame."""
        self._last_objects = objects_in_base                       # cache for the viewer key-waypoint overlay
        progress = self._elapsed / self.interpolation_dt
        lo = min(int(progress), self.H - 1)
        hi = min(lo + 1, self.H - 1)
        alpha = progress - lo
        q_nom = (1.0 - alpha) * self.route_q[lo] + alpha * self.route_q[hi]
        # Planned joint velocity at this instant, for the servo's velocity feedforward. Scaled by
        # _REACH_SPEED_SCALE because that is what actually advances ``_elapsed`` below: at half playback the
        # arm really does move half as fast, so the lead must halve with it.
        qdot_nom = (self.route_q[hi] - self.route_q[lo]) / self.interpolation_dt * _REACH_SPEED_SCALE
        goal_off = {f: self.path_in_object[f][lo] for f in self.active}
        desired = {f: pose_compose(objects_in_base[self.frame_to_object[f]], goal_off[f]) for f in self.active}
        command = self._track_desired(desired, q_nom, gravity_in_base, actual_tool_poses_base,
                                      measured_arm_qpos, _tag="extend", qdot_nom=qdot_nom)
        # Diagnostic-only: DYNAMIC_REACH_SPEED_SCALE slows route playback (e.g. 0.5 = half speed) to
        # sanity-check whether tracking lag against a still-shifting base is the failure mechanism.
        # Default 1.0 leaves EXTEND-phase speed unchanged.
        self._elapsed += self.control_dt * _REACH_SPEED_SCALE
        return command

    def freeze_hold(self) -> None:
        """Freeze final planned Cartesian grasp target in current base frame, never in joint/cube coordinates."""
        assert self._last_desired_base is not None, "cannot freeze hold before tracker emits a target"
        self._hold_pose_base = {
            f: (np.asarray(p[0], dtype=float).copy(), np.asarray(p[1], dtype=float).copy())
            for f, p in self._last_desired_base.items()}

    def final_step(self, objects_in_base, gravity_in_base=None, actual_tool_poses_base=None,
                   measured_arm_qpos=None) -> dict:
        """Track final object-relative grasp target while jaws close.

        The target remains attached to the live observed object until closure completes. Freezing it in the
        base frame at EXTEND completion made a floating-base shift during jaw closure look like a grasp miss.
        ``freeze_hold`` is deliberately deferred until closure finishes, when return planning needs a stable
        base-frame bridge.
        """
        desired = {f: pose_compose(objects_in_base[self.frame_to_object[f]], self.path_in_object[f][-1])
                   for f in self.active}
        return self._track_desired(desired, self.route_q[-1], gravity_in_base, actual_tool_poses_base,
                                   measured_arm_qpos, _tag="grasp")

    def hold_step(self, gravity_in_base=None, actual_tool_poses_base=None, measured_arm_qpos=None) -> dict:
        """Keep frozen base-frame grasp pose tracked while jaws close or asynchronous return planning runs."""
        assert self._hold_pose_base is not None, "freeze grasp hold before tracking it"
        return self._track_desired(self._hold_pose_base, self.route_q[-1], gravity_in_base,
                                   actual_tool_poses_base, measured_arm_qpos, _tag="hold")

    def begin_return(self, route, measured_arm_qpos=None) -> None:
        """Track a return route through matched Cartesian and joint-nominal bridges.

        The DLS null-space term consumes ``q_nom`` as well as the Cartesian target.  Bridging only the
        tool pose while immediately setting ``q_nom=route_q[0]`` pulls redundant joints toward home at
        grasp release.  Start both signals at measured grasp qpos and bound their respective speeds.
        """
        assert self._hold_pose_base is not None, "freeze grasp hold before starting return"
        assert set(route.controlled_joints) == set(self.controlled), (
            f"return changed active arm set: {route.controlled_joints} vs {self.controlled}")
        assert list(route.joint_names) == self.arm_joints, "return changed arm joint order"
        route_q = np.asarray(route.route_q_curobo, dtype=float)
        if measured_arm_qpos is None:
            hold_q = route_q[0].copy()
        else:
            try:
                hold_q = np.asarray([measured_arm_qpos[j] for j in self.arm_joints], dtype=float) - self.offset
            except KeyError as exc:
                raise KeyError(f"missing measured qpos for return arm joint {exc.args[0]!r}") from exc
        path = {f: [] for f in self.active}
        for q in route_q:
            self._fk(q, stage=_FkStage.KINEMATICS)
            for f in self.active:
                path[f].append((self.data.site_xpos[self._site_id[f]].copy(), self._site_quat(f)))
        max_delta = max(float(np.linalg.norm(path[f][0][0] - self._hold_pose_base[f][0])) for f in self.active)
        n_bridge = max(1, int(np.ceil(_RETURN_BRIDGE_MIN_S / route.interpolation_dt)),
                       int(np.ceil(max_delta / (_RETURN_BRIDGE_MAX_SPEED_M_S * route.interpolation_dt))),
                       int(np.ceil(np.max(np.abs(route_q[0] - hold_q)) /
                                       (_RETURN_BRIDGE_MAX_JOINT_SPEED_RAD_S * route.interpolation_dt))))
        bridge = {f: [_lerp_pose(self._hold_pose_base[f], path[f][0], k / n_bridge)
                      for k in range(n_bridge)] for f in self.active}
        self._return_path_base = {f: bridge[f] + path[f] for f in self.active}
        joint_bridge = np.linspace(hold_q, route_q[0], n_bridge + 1, dtype=float)[:-1]
        self._return_q = np.vstack((joint_bridge, route_q))
        assert np.allclose(self._return_q[0], hold_q), "return nominal must start at measured grasp qpos"
        self._return_dt = float(route.interpolation_dt)
        self._return_elapsed = 0.0

    def return_step(self, gravity_in_base=None, actual_tool_poses_base=None, measured_arm_qpos=None) -> dict:
        """Emit one dynamic Cartesian return command; route completion is exposed by ``return_settled``."""
        assert self._return_q is not None and self._return_path_base is not None, "start return first"
        progress = self._return_elapsed / self._return_dt
        lo = min(int(progress), self._return_q.shape[0] - 1)
        hi = min(lo + 1, self._return_q.shape[0] - 1)
        alpha = progress - lo
        q_nom = (1.0 - alpha) * self._return_q[lo] + alpha * self._return_q[hi]
        # Same velocity feedforward the extend path gets (see ``_track_desired``); the return previously
        # passed none, so it held a moving reference on the gravity term alone and ate the full
        # velocity-proportional following error. Scaled by _RETURN_SPEED_SCALE for the same reason
        # ``step`` scales by _REACH_SPEED_SCALE: that scale is what advances the clock below, so the lead
        # must track the speed the arm actually moves at.
        qdot_nom = (self._return_q[hi] - self._return_q[lo]) / self._return_dt * _RETURN_SPEED_SCALE
        desired = {f: _lerp_pose(self._return_path_base[f][lo], self._return_path_base[f][hi], alpha)
                   for f in self.active}
        command = self._track_desired(desired, q_nom, gravity_in_base, actual_tool_poses_base,
                                      measured_arm_qpos, _tag="return", qdot_nom=qdot_nom)
        self._return_elapsed += self.control_dt * _RETURN_SPEED_SCALE
        return command

    def return_settled(self, settle_s: float) -> bool:
        """Whether tracked return bridge plus route has completed its requested physical settle."""
        assert self._return_q is not None, "start return first"
        duration = (self._return_q.shape[0] - 1) * self._return_dt
        # ``_return_elapsed`` runs on the SCALED route clock, so a wall-clock settle must be converted into
        # it -- comparing against a bare settle_s would shrink the physical settle by the speed scale.
        return self._return_elapsed >= duration + settle_s * _RETURN_SPEED_SCALE

    def hold_command(self) -> dict:
        """Return the final emitted grasp command unchanged while the FSM waits for a return route."""
        assert self._last_command is not None, "cannot hold before tracker emits a command"
        return dict(self._last_command)

    def route_progress(self) -> float:
        """Fraction of the extend route's own trajectory clock consumed, clipped to [0, 1].

        Lets the FSM distinguish "still travelling" from "arrived", which ``extend_settled`` (a pure wall-clock
        end-of-route test) cannot. Used to confine the mid-route divergence abort to the part of the path where
        the hand is not yet at the cube.
        """
        route_s = (self.H - 1) * self.interpolation_dt
        return 1.0 if route_s <= 0.0 else float(min(1.0, max(0.0, self._elapsed / route_s)))

    def extend_settled(self, settle_s: float) -> bool:
        """Whether extend has completed its planned path and requested dwell.

        ``ReachPolicy`` owns the walk FSM and asks this instead of inspecting tracker internals. The
        The final planned waypoint is the grasp. Only after its requested dwell may the FSM plan the retract.
        """
        assert settle_s >= 0.0, f"settle_s must be nonnegative, got {settle_s}"
        route_s = (self.H - 1) * self.interpolation_dt
        return self._elapsed >= route_s + settle_s

    def final_position_error(self, actual_tool_poses_base: dict) -> float:
        """Maximum measured tool-position error to the final live object-relative target."""
        assert self._last_desired_base is not None, "tracker has not emitted a reach command"
        return max(float(np.linalg.norm(np.asarray(actual_tool_poses_base[frame][0]) - target[0]))
                   for frame, target in self._last_desired_base.items())

    def position_errors(self, actual_tool_poses_base: dict) -> dict[str, float]:
        """Per-frame measured tool-position error, BASE frame -- the breakdown ``final_position_error`` maxes
        over. A bimanual reach misses with ONE hand while the other latches fine, so the max alone cannot say
        which hand missed; this can. Diagnostic only, no control path reads it."""
        assert self._last_desired_base is not None, "tracker has not emitted a reach command"
        return {frame: float(np.linalg.norm(np.asarray(actual_tool_poses_base[frame][0]) - target[0]))
                for frame, target in self._last_desired_base.items()}

    def key_goals_base(self, every: int = _WP_SAMPLE):
        """Live drift-corrected tool-frame goals at the KEY planned waypoints in BASE frame -- for the viewer
        overlay. Each
        goal = ``pose_compose(last observed object, object-anchored planned waypoint)``, so the drawn frames
        are exactly the target path the servo chases and TRACK the observed cube as the base drifts (the same
        object-anchored recomposition ``step`` uses). Returns ``{frame: [(pos, quat_wxyz)]}``; empty until the
        first ``step`` (no observed objects cached yet). Diagnostic only -- not used by the control loop."""
        if self._last_objects is None:
            return {}
        idxs = list(range(0, self.H, every))
        if idxs[-1] != self.H - 1:
            idxs.append(self.H - 1)
        goals = {}
        for f in self.active:
            o = self._last_objects[self.frame_to_object[f]]
            goals[f] = [pose_compose(o, self.path_in_object[f][k]) for k in idxs]
        return goals
