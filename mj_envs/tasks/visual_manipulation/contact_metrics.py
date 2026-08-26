"""Read-only MuJoCo contact diagnostics for dynamic manipulation evaluation."""

from __future__ import annotations

import numpy as np
import mujoco


class TableContactMonitor:
    """Measure table contact count and estimated normal impulse from copied simulator state.

    The monitor owns shadow ``MjData`` objects. It never touches the live MuJoCo-Warp state, so enabling
    diagnostics cannot change dynamics. CPU re-forward is intentionally evaluation-only; callers should
    keep it disabled for high-throughput vector rollouts.
    """

    def __init__(self, model: mujoco.MjModel, num_envs: int, control_dt: float) -> None:
        self._model = model
        self._data = [mujoco.MjData(model) for _ in range(num_envs)]
        self._control_dt = float(control_dt)
        self._table_geom_ids = {
            geom_id for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith("pp_table_")
        }
        self.last_pairs: list[tuple[tuple[str, str], ...]] = [()] * num_envs

    def sample(self, qpos: np.ndarray, qvel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(table_contact_count, normal_impulse)`` for rows of live state copies."""
        qpos = np.asarray(qpos)
        assert qpos.shape == (len(self._data), self._model.nq), qpos.shape
        if qvel is not None:
            qvel = np.asarray(qvel)
            assert qvel.shape == (len(self._data), self._model.nv), qvel.shape
        counts = np.zeros(len(self._data), dtype=np.int32)
        impulses = np.zeros(len(self._data), dtype=np.float32)
        for env_id, data in enumerate(self._data):
            data.qpos[:] = qpos[env_id]
            if qvel is not None:
                data.qvel[:] = qvel[env_id]
            mujoco.mj_forward(self._model, data)
            pairs = []
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                if contact.geom1 not in self._table_geom_ids and contact.geom2 not in self._table_geom_ids:
                    continue
                name1 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                name2 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self._model, data, contact_id, force)
                counts[env_id] += 1
                impulses[env_id] += max(0.0, force[0]) * self._control_dt
                pairs.append((name1, name2))
            self.last_pairs[env_id] = tuple(pairs)
        return counts, impulses
