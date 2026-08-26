"""Minimal constrained local-QP tracker for a cuRobo dynamic reach route.

cuRobo owns global route/table/world collision.  This tracker only corrects one
object-anchored cuRobo waypoint against measured floating-base sway.  It runs one
NumPy/MuJoCo Mink QP per 50 Hz tick: active-arm pose tasks, nominal-route posture,
hard configuration/step limits, and applicable self-collision constraints.  Physical
latch remains the sole grasp-success verdict.  It retains the dynamic tracker's
gravity-droop feedforward; output trajectory filtering remains disabled.

This is deliberately not ``BatchedMinkIK``. That deploy-facing wrapper owns Torch
batch state, both-arm tasks, target filters, and a trajectory profile; those are wrong
for a one-route, one-tick dynamic correction and added avoidable latency/dwell.
"""

from __future__ import annotations

from collections import deque
import os
import time

import mujoco
import numpy as np

from mj_envs.utils.ik_mink import (
    HUMANOID_V21_MINK_COLLISION_PAIRS,
    NoSolutionFound,
    _build_hybrid_mink_namespace,
    _build_mink_geom_pairs,
)
from tasks.visual_manipulation.jacobian_reach_tracker import JacobianReachTracker


_MINK_STEP_LIMIT_RAD = 0.12  # 6 rad/s × 20 ms; QP uses unit pseudo-time.
_MINK_POSTURE_WEIGHT = 0.02
_MINK_DAMPING = 1e-2
_MINK_DEBUG = bool(os.environ.get("MINK_DEBUG"))


def _model_name(mj_model, obj_type, name: str) -> str:
    """Resolve planner bare name against standalone or dynamic ``robot/`` model."""
    for candidate in (name, f"robot/{name}"):
        if mujoco.mj_name2id(mj_model, obj_type, candidate) != -1:
            return candidate
    raise KeyError(f"{name!r} missing from MuJoCo model")


class MinkReachTracker(JacobianReachTracker):
    """One-QP, active-arm-only drift correction with parent route/FSM lifecycle.

    Parent extracts object-relative waypoints and implements extend/final/return
    semantics.  This subclass replaces only its DLS solve.  Every call seeds Mink
    from measured full-body qpos, so live legs/waist and arms define the base-frame
    kinematic chain.  Inactive arm and all non-arm DOFs are equality-frozen, rather
    than competing through an idle pose task.

    Humanoids use generated arm/gripper self-collision pairs. G1 has no validated
    Mink collision-pair set, so its local QP leaves collision rows empty and remains
    bounded to cuRobo's collision-checked route. Do not claim G1 local collision
    coverage until such a set exists.
    """

    def __init__(self, mj_model, robot_cfg, control_dt: float,
                 gravity_comp: bool = True, gravity_comp_alpha: float = 1.0,
                 max_measured_correction_rad: float | None = None,
                 transit_max_correction_rad: float | None = None):
        super().__init__(
            mj_model, robot_cfg.tool_frames, control_dt,
            gravity_comp=gravity_comp, gravity_comp_alpha=gravity_comp_alpha,
            output_filter=False,
            max_measured_correction_rad=max_measured_correction_rad,
            transit_max_correction_rad=transit_max_correction_rad,
        )
        self._robot_cfg = robot_cfg
        self._mink = _build_hybrid_mink_namespace()
        self._config = self._mink.Configuration(mj_model)
        self._root_name = _model_name(mj_model, mujoco.mjtObj.mjOBJ_BODY, robot_cfg.base_link)
        self._frame_name = {
            frame: _model_name(mj_model, mujoco.mjtObj.mjOBJ_SITE, frame)
            for frame in robot_cfg.tool_frames
        }
        self._joint_name = {
            name: _model_name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for side in ("L", "R") for name in robot_cfg.arm_joints_by_side[side]
        }
        self._tick_ms: deque[float] = deque(maxlen=512)
        self._no_solution_count = 0

    def rebind(self, route, objects_plan) -> None:
        """Extract parent object-relative route, then rebuild active-arm QP objects once."""
        super().rebind(route, objects_plan)
        self._build_qp()

    def _humanoid_pairs(self):
        if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/shoulder_2_L") == -1:
            return []

        active_sides = {"L" if frame == self.tool_frames[0] else "R" for frame in self.active}

        def _active(side) -> bool:
            names = (side,) if isinstance(side, str) else tuple(side)
            return any(name.endswith(f"_{arm}") for name in names for arm in active_sides)

        def _ns(side):
            if isinstance(side, str):
                return f"robot/{side}"
            return tuple(f"robot/{name}" for name in side)

        # Keep pairs involving an active arm. Idle-arm/idle-arm rows cannot change because
        # those DOFs are equality-frozen; dropping them reduces QP rows without reducing safety.
        return [(_ns(left), _ns(right)) for left, right in HUMANOID_V21_MINK_COLLISION_PAIRS
                if _active(left) or _active(right)]

    def _build_qp(self) -> None:
        """Build only route-active task/limit rows; construction is outside 50 Hz loop."""
        self._pose_tasks = {}
        self._target_buf = {}
        for frame in self.active:
            task = self._mink.RelativeFrameTask(
                frame_name=self._frame_name[frame], frame_type=self._robot_cfg.ee_frame_type,
                root_name=self._root_name, root_type="body",
                position_cost=5.0, orientation_cost=[1.0, 1.0, 0.01], lm_damping=2.0,
            )
            self._pose_tasks[frame] = task
            self._target_buf[frame] = np.empty(7, dtype=np.float64)
        self._posture_task = self._mink.PostureTask(self.mj_model, cost=_MINK_POSTURE_WEIGHT)
        self._tasks = [*self._pose_tasks.values(), self._posture_task]
        active_names = [self._joint_name[name] for name in self.controlled]
        active_dofs = {int(self.mj_model.joint(name).dofadr[0]) for name in active_names}
        frozen = [dof for dof in range(self.mj_model.nv) if dof not in active_dofs]
        self._constraints = [self._mink.DofFreezingTask(self.mj_model, dof_indices=frozen)]
        self._limits = [
            self._mink.ConfigurationLimit(self.mj_model),
            # Mink solves unit pseudo-time; this is a per-control-step displacement cap.
            self._mink.VelocityLimit(
                self.mj_model, velocities={name: _MINK_STEP_LIMIT_RAD for name in active_names}
            ),
        ]
        pairs = _build_mink_geom_pairs(self.mj_model, self._humanoid_pairs(), self._mink)
        if pairs:
            self._limits.append(self._mink.CollisionAvoidanceLimit(
                model=self.mj_model, geom_pairs=pairs,
                minimum_distance_from_collisions=0.05,
                collision_detection_distance=0.05,
            ))

    def _measured_route_q(self, measured_qpos: dict | None, q_nom: np.ndarray) -> np.ndarray:
        if measured_qpos is None:
            return q_nom.copy()
        try:
            return np.asarray([measured_qpos[name] for name in self.arm_joints], dtype=float) - self.offset
        except KeyError as exc:
            raise KeyError(f"missing measured qpos for route arm joint {exc.args[0]!r}") from exc

    def _set_config_qpos(self, qpos: np.ndarray) -> None:
        self._config.data.qpos[:] = qpos
        self._config.data.qpos[self._root_qadr:self._root_qadr + 3] = 0.0
        self._config.data.qpos[self._root_qadr + 3:self._root_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self._config.update()

    def _set_posture_target(self, qpos_live: np.ndarray, q_nom: np.ndarray) -> None:
        qpos_ref = qpos_live.copy()
        qpos_ref[self.arm_qadr] = q_nom + self.offset
        self._set_config_qpos(qpos_ref)
        self._posture_task.set_target_from_configuration(self._config)
        self._set_config_qpos(qpos_live)

    def _track_desired(self, desired, q_nom, gravity_in_base=None, actual_tool_poses_base=None,
                       measured_arm_qpos=None, _tag: str = ""):
        """Solve one direct QP, hold last command on infeasibility, emit active-arm qpos."""
        started = time.perf_counter()
        q_nom = np.asarray(q_nom, dtype=float)
        q_measured = self._measured_route_q(measured_arm_qpos, q_nom)
        self._fk(q_measured, measured_arm_qpos)
        qpos_live = self.data.qpos.copy()
        self._set_posture_target(qpos_live, q_nom)
        for frame in self.active:
            pos, quat = desired[frame]
            target = self._target_buf[frame]
            target[:4] = quat
            target[4:] = pos
            self._pose_tasks[frame].set_target(self._mink.SE3(wxyz_xyz=target))
        try:
            velocity = self._mink.solve_ik(
                self._config, self._tasks, dt=1.0, solver="daqp", damping=_MINK_DAMPING,
                limits=self._limits, constraints=self._constraints, safety_break=False,
            )
            self._config.integrate_inplace(velocity, 1.0)
            self._no_solution_count = 0
        except (self._mink.NoSolutionFound, NoSolutionFound):
            self._no_solution_count += 1
            self._tick_ms.append((time.perf_counter() - started) * 1e3)
            return self.hold_command() if self._last_command is not None else {
                name: float(q_nom[index] + self.offset[index])
                for index, name in enumerate(self.arm_joints) if name in self.controlled
            }

        # Preserve the dynamic tracker's gravity feedforward. The QP finds the collision-constrained
        # kinematic target; the frozen direct-arm position servo still settles low by g(q)/kp under a
        # carried arm. This is not trajectory filtering: it is a bounded steady-state torque correction.
        q_command = np.asarray(self._config.data.qpos[self.arm_qadr], dtype=float) - self.offset
        if self._gravity_comp:
            assert gravity_in_base is not None, "gravity_comp=True needs gravity_in_base in step()"
            gravity_world = self.mj_model.opt.gravity.copy()
            self.mj_model.opt.gravity[:] = np.asarray(gravity_in_base, dtype=float)
            self._fk(q_command, measured_arm_qpos)
            tau_g = self.data.qfrc_bias[self.arm_dofadr].copy()
            self.mj_model.opt.gravity[:] = gravity_world
            q_command += self._grav_alpha * tau_g / self.arm_kp

        command = {}
        for route_index, name in enumerate(self.arm_joints):
            if name not in self.controlled:
                continue
            value = float(q_command[route_index] + self.offset[route_index])
            if self._max_measured_correction_rad is not None:
                value = float(np.clip(value,
                                      q_measured[route_index] + self.offset[route_index] - self._max_measured_correction_rad,
                                      q_measured[route_index] + self.offset[route_index] + self._max_measured_correction_rad))
                value = float(np.clip(value,
                                      q_nom[route_index] + self.offset[route_index] - self._max_measured_correction_rad,
                                      q_nom[route_index] + self.offset[route_index] + self._max_measured_correction_rad))
            command[name] = value
        self._last_desired_base = {
            frame: (np.asarray(pose[0], dtype=float).copy(), np.asarray(pose[1], dtype=float).copy())
            for frame, pose in desired.items()
        }
        self._last_command = command
        self._tick_ms.append((time.perf_counter() - started) * 1e3)
        if _MINK_DEBUG and len(self._tick_ms) % 100 == 0:
            values = np.asarray(self._tick_ms, dtype=float)
            physical_err = float("nan")
            if actual_tool_poses_base is not None:
                physical_err = max(
                    float(np.linalg.norm(np.asarray(desired[frame][0]) -
                                         np.asarray(actual_tool_poses_base[frame][0])))
                    for frame in self.active
                )
            print(f"[mink:{_tag}] median_ms={np.median(values):.3f} "
                  f"p95_ms={np.percentile(values, 95):.3f} physical_err={physical_err:.4f}")
        return dict(command)

    def timing_stats_ms(self) -> dict[str, float]:
        """Recent complete-tick timing; acceptance requires median <5 ms and p95 <20 ms."""
        if not self._tick_ms:
            return {"median": float("nan"), "p95": float("nan"), "count": 0.0}
        values = np.asarray(self._tick_ms, dtype=float)
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "count": float(values.size)}
