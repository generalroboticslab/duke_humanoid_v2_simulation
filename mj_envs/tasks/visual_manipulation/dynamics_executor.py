"""Time-retimed arm-reference execution for physics rollouts.

cuRobo owns route feasibility. A frozen RL policy owns low-level joint actions. This module bridges
them without ever writing MuJoCo state: it samples a named cuRobo trajectory at control time and writes
only the corresponding ``arm_ref`` columns. Physics then advances through the normal action path.
"""

from __future__ import annotations

import numpy as np
import torch


class DynamicArmReferenceExecutor:
    """Sample one immutable named arm route at a fixed RL-control cadence.

    Args:
        controlled_joint_names: arm-reference joints this route may overwrite. Other ``arm_ref``
            columns stay at the caller's home/reference template.
        control_dt: seconds between RL actions, not MuJoCo's inner physics timestep.

    ``load`` receives MuJoCo-qpos coordinates, including any XML reference offsets. This keeps
    q-coordinate conversion with the cuRobo adapter and makes execution a pure named interpolation.
    The class intentionally has no simulator handle: a state write would violate phase-4 dynamics.
    """

    def __init__(self, controlled_joint_names: list[str], control_dt: float) -> None:
        assert controlled_joint_names, "executor needs at least one controlled joint"
        assert control_dt > 0.0, f"control_dt must be positive, got {control_dt}"
        self._controlled_joint_names = tuple(controlled_joint_names)
        self._control_dt = float(control_dt)
        self._trajectory: np.ndarray | None = None
        self._trajectory_dt = 0.0
        self._elapsed_s = 0.0
        self._plan_id = 0
        self._last_reference: torch.Tensor | None = None

    @property
    def plan_id(self) -> int:
        return self._plan_id

    @property
    def cursor_s(self) -> float:
        return self._elapsed_s

    @property
    def duration_s(self) -> float:
        assert self._trajectory is not None, "load a route before reading its duration"
        return (self._trajectory.shape[0] - 1) * self._trajectory_dt

    @property
    def n_waypoints(self) -> int:
        assert self._trajectory is not None, "load a route before reading its length"
        return int(self._trajectory.shape[0])

    @property
    def last_reference(self) -> torch.Tensor | None:
        return None if self._last_reference is None else self._last_reference.clone()

    def load(self, trajectory_joint_names: list[str], trajectory_q_mjlab: np.ndarray,
             trajectory_dt: float, start_s: float = 0.0) -> None:
        """Install one cuRobo route, retaining only named controlled-joint columns.

        ``start_s`` skips already-passed leading waypoints when a live-state replan replaces an
        active route. It is clamped to the route duration, so route installation cannot expose an
        out-of-bounds cursor.
        """
        assert trajectory_dt > 0.0, f"trajectory_dt must be positive, got {trajectory_dt}"
        q = np.asarray(trajectory_q_mjlab, dtype=np.float32)
        assert q.ndim == 2 and q.shape[0] > 0, f"expected nonempty [H,D] route, got {q.shape}"
        index = {name: i for i, name in enumerate(trajectory_joint_names)}
        missing = [name for name in self._controlled_joint_names if name not in index]
        assert not missing, f"cuRobo route missing arm_ref joints: {missing}"
        self._trajectory = q[:, [index[name] for name in self._controlled_joint_names]].copy()
        self._trajectory_dt = float(trajectory_dt)
        self._elapsed_s = min(max(0.0, float(start_s)), self.duration_s)
        self._plan_id += 1
        # Seed last_reference from the first waypoint so the policy's
        # ``_executor.last_reference is not None`` gate is satisfied on the
        # very next compute() without waiting for command() to be called.
        self._last_reference = torch.from_numpy(self._trajectory[0].copy())

    def clear(self) -> None:
        """Discard route at episode reset; the next command requires a fresh live-state plan."""
        self._trajectory = None
        self._elapsed_s = 0.0
        self._last_reference = None

    def command(self, arm_ref_template: torch.Tensor, arm_ref_joint_names: list[str]) -> torch.Tensor:
        """Return next absolute ``arm_ref`` command and advance one RL control interval.

        Linear interpolation prevents a hidden dependence on whether planner waypoint spacing equals
        environment control dt. The final route point is held indefinitely for physics settling.
        """
        assert self._trajectory is not None, "load a route before requesting a command"
        ref_index = {name: i for i, name in enumerate(arm_ref_joint_names)}
        missing = [name for name in self._controlled_joint_names if name not in ref_index]
        assert not missing, f"arm_ref missing controlled joints: {missing}"
        progress = self._elapsed_s / self._trajectory_dt
        lo = min(int(progress), self._trajectory.shape[0] - 1)
        hi = min(lo + 1, self._trajectory.shape[0] - 1)
        alpha = progress - lo
        q_ref = (1.0 - alpha) * self._trajectory[lo] + alpha * self._trajectory[hi]
        command = arm_ref_template.clone()
        q_ref_t = torch.as_tensor(q_ref, device=command.device, dtype=command.dtype)
        for source_col, name in enumerate(self._controlled_joint_names):
            command[:, ref_index[name]] = q_ref_t[source_col]
        self._last_reference = command.detach().clone()
        self._elapsed_s += self._control_dt
        return command
