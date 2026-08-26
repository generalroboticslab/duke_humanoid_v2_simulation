"""Offline visible-reachable goal-set curator.

Reuses the reachability-study capability field (per-voxel ``D_reach = n_success/64`` over the 64
SO(3) refs) plus the head-camera visibility scorer as an OFFLINE proxy for "is this a good goal
point?". It replaces a live cuRobo IK call (``ik_feasible_route``) for goal-set CURATION only --
NOT the runtime EXTEND gate, which stays cuRobo.

A point is USEFUL iff ``visible AND D_reach >= tau``. Base-shift robustness = ``min D_reach`` over a
base-frame neighborhood ball (0 if any voxel in the ball is blind); high => tolerant to base drift
(the continuity hypothesis: the visible-reachable field is smooth, so a high-scoring neighborhood
survives a small base shift that relocates the goal to a nearby base-frame voxel).

PER-ARM, not bimanual-max: a cube is scored against the arm that would grasp it. The payload is
R-only and the robot is Y-symmetric, so the LEFT arm's reachability at base-frame point ``(x,y,z)``
equals ``D_R`` at ``(x,-y,z)`` -- one field serves both arms by mirroring the query. This matches the
repo's existing per-arm rule (``ReachabilityModel.from_robot`` rejects ``arm='both'``).

Frame: the workspace field and head camera are rigid to ``base_link``, so the field is
base-pose-INVARIANT -- built once, queried at any stance. A cube's world pose is transformed into
the live ``base_link`` frame before lookup.

Build/query split mirrors ``ik_seed_oracle.py``. The heavy build path (payload load, GPU visibility
raycast) lazy-imports the plotter + cuRobo; the query path (``VisibleReachableField``) stays
light (numpy + one small torch transform), needing neither cuRobo nor matplotlib.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MJ_ENVS = _REPO_ROOT / "mj_envs"
for _p in (str(_REPO_ROOT), str(_MJ_ENVS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mj_envs.utils.reachability import world_to_base_link_frame  # noqa: E402

_CACHE_DIR = _REPO_ROOT / "mj_envs" / "asset_zoo" / "cache"

# D_reach gate, CALIBRATED against ik_feasible_route (calibrate_tau.py, v2/front_back_close, N=120):
# D_reach>0 (64-SO(3) any-orientation) is a SUPERSET of the tilted grasp goalset (80 vs 49 reachable),
# so a floor is needed. tau=0.25 is the agreement knee (~0.79 agreement, precision ~0.85, recall ~0.58);
# higher tau buys precision at steep recall cost. Re-fit per robot before trusting a hard verdict.
_DEFAULT_TAU = 0.25


def _sidecar_path(model_key: str, visibility_kind: str) -> pathlib.Path:
    return _CACHE_DIR / f"visible_reachable_{model_key}_{visibility_kind}.pt"


# ---------------------------------------------------------------------------
# Build (heavy: payload load + GPU visibility raycast)
# ---------------------------------------------------------------------------

def build_field(robot: str = "v2", payload_path: pathlib.Path | None = None,
                out: pathlib.Path | None = None) -> pathlib.Path:
    """Build the slim dense visible-reachable sidecar for ``robot`` (v2 / v2_fixed / g1).

    Aggregates ``D_reach`` per voxel (reusing the plotter's ``_aggregate_by_voxel``) and the head-cam
    ``visible`` mask (``_camera_visibility``: ``vis_steer`` for the actuated gimbal, ``vis_fixed`` for a
    fixed head cam), scatters both onto the dense ``(nx,ny,nz)`` lattice (0 / False on unsolved voxels),
    and saves grid metadata + ``D_reach`` (f16) + ``visible`` (bool). One file per robot, ~MB, gitignored.
    """
    # Lazy: the plotter pulls matplotlib + (via _camera_visibility) mujoco/warp; only the build needs it.
    from mj_envs.asset_zoo.reachability_study.plot_workspace_curobo import (
        _ROBOTS, _aggregate_by_voxel, _camera_visibility,
    )

    cfg = _ROBOTS.get(robot)
    if cfg is None:
        raise ValueError(f"robot {robot!r} unknown; choose {sorted(_ROBOTS)}")
    if cfg.visibility_kind not in ("steered", "fixed"):
        raise ValueError(
            f"robot {robot!r} has visibility_kind={cfg.visibility_kind!r}; the curator supports only "
            "'steered'/'fixed' (v2/v2_fixed/g1). Dynamic-camera robots need their visibility sidecar.")

    payload_path = pathlib.Path(payload_path) if payload_path else (_CACHE_DIR / cfg.payload)
    data = torch.load(str(payload_path), map_location="cpu", weights_only=False)

    origin = data["grid_origin"].float().numpy()
    spacing = float(data["grid_spacing"])
    n_per = tuple(int(x) for x in data["n_grid_per_axis"].tolist())
    nx, ny, nz = n_per

    agg = _aggregate_by_voxel(data)                       # voxel_id (flat), voxel_pos, dexterity(=D_R)
    vid = agg["voxel_id"].astype(np.int64)
    centers = agg["voxel_pos"].astype(np.float32)

    vis_steer, vis_fixed = _camera_visibility(centers, cfg.model_key)
    vis = np.asarray(vis_steer if cfg.visibility_kind == "steered" else vis_fixed, dtype=bool)

    n_flat = nx * ny * nz
    D = np.zeros(n_flat, dtype=np.float32)
    D[vid] = agg["dexterity"].astype(np.float32)
    V = np.zeros(n_flat, dtype=bool)
    V[vid] = vis

    sidecar = {
        "D_reach": torch.from_numpy(D.reshape(nx, ny, nz)).half(),
        "visible": torch.from_numpy(V.reshape(nx, ny, nz)),
        "grid_origin": torch.as_tensor(origin, dtype=torch.float32),
        "grid_spacing": float(spacing),
        "n_grid_per_axis": torch.tensor(n_per, dtype=torch.long),
        "robot": robot,
        "model_key": cfg.model_key,
        "visibility_kind": cfg.visibility_kind,
    }
    out = pathlib.Path(out) if out else _sidecar_path(cfg.model_key, cfg.visibility_kind)
    torch.save(sidecar, str(out))
    n_reach = int((D > 0).sum())
    n_useful = int(((D > 0) & V).sum())
    mb = out.stat().st_size / 1e6
    print(f"saved {out}  grid={n_per}  reachable_voxels={n_reach}  visible&reachable={n_useful}  "
          f"size={mb:.2f} MB")
    return out


# ---------------------------------------------------------------------------
# Query (light: numpy dense field; no cuRobo / matplotlib)
# ---------------------------------------------------------------------------

class VisibleReachableField:
    """Query the dense visible-reachable sidecar. All positions in ``base_link`` frame.

    ``arm`` in ``{"L","R"}``: ``R`` looks up ``D_R`` directly; ``L`` negates the base-frame ``y`` first
    (exact Y-symmetry mirror). Out-of-grid queries return the not-useful default (blind, D=0).
    """

    def __init__(self, sidecar_path: pathlib.Path):
        s = torch.load(str(sidecar_path), map_location="cpu", weights_only=False)
        self.D = s["D_reach"].float().numpy()            # (nx,ny,nz)
        self.visible = s["visible"].numpy().astype(bool)  # (nx,ny,nz)
        self.origin = s["grid_origin"].numpy().astype(np.float64)
        self.spacing = float(s["grid_spacing"])
        self.n_per = tuple(int(x) for x in s["n_grid_per_axis"].tolist())
        self.robot = s.get("robot", "?")
        self.visibility_kind = s.get("visibility_kind", "?")

    @classmethod
    def from_robot(cls, robot: str = "v2") -> "VisibleReachableField":
        """Load robot's existing sidecar without importing builder-only plotting dependencies.

        The saved ``robot`` field is authoritative for a query.  Scanning the tiny sidecars keeps this
        runtime path free of the plotter, MuJoCo, and CUDA visibility builder; the heavier ``_ROBOTS``
        registry remains build-only.
        """
        for path in _CACHE_DIR.glob("visible_reachable_*_*.pt"):
            sidecar = torch.load(str(path), map_location="cpu", weights_only=False)
            if sidecar.get("robot") == robot:
                return cls(path)
        raise FileNotFoundError(f"no visible-reachable sidecar for robot {robot!r} in {_CACHE_DIR}")

    def _voxel(self, pos_base: np.ndarray, arm: str) -> tuple[int, int, int] | None:
        """Base-frame position -> integer voxel (ix,iy,iz), or None if outside the grid.
        ``arm='L'`` mirrors ``y`` into the R field first."""
        p = np.asarray(pos_base, dtype=np.float64).copy()
        if arm == "L":
            p[1] = -p[1]
        elif arm != "R":
            raise ValueError(f"arm must be 'L' or 'R'; got {arm!r}")
        idx = np.round((p - self.origin) / self.spacing).astype(np.int64)
        if np.any(idx < 0) or np.any(idx >= np.asarray(self.n_per)):
            return None
        return int(idx[0]), int(idx[1]), int(idx[2])

    def useful(self, pos_base, arm: str) -> tuple[bool, float]:
        """(visible, D_reach) at the single base-frame voxel for ``arm``. Out-of-grid -> (False, 0.0)."""
        v = self._voxel(pos_base, arm)
        if v is None:
            return False, 0.0
        return bool(self.visible[v]), float(self.D[v])

    def _robust_radius_voxels(self, delta_m: float) -> int:
        """Conservative lattice radius covering every ``delta_m`` base shift.

        A query rounds to its nearest voxel.  With 2 cm spacing, ``round(0.05
        / 0.02)`` silently gives two (4 cm), while a 5 cm physical shift can
        land in the third neighbor (6 cm center).  ``ceil`` keeps that outer
        lattice shell and therefore never certifies less perturbation than the
        requested physical margin.
        """
        return int(np.ceil(float(delta_m) / self.spacing))

    def robust_score(self, pos_base, arm: str, delta_m: float = 0.05) -> float:
        """Base-shift-tolerance metric: ``min D_reach`` over the ``+-delta_m`` base-frame voxel ball,
        0 if any voxel in the ball is out-of-grid or blind. High => a small base drift keeps the goal
        both reachable and visible."""
        c = self._voxel(pos_base, arm)
        if c is None:
            return 0.0
        r = self._robust_radius_voxels(delta_m)
        nx, ny, nz = self.n_per
        d2 = r * r
        worst = np.inf
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx * dx + dy * dy + dz * dz > d2:
                        continue                      # ball, not cube
                    ix, iy, iz = c[0] + dx, c[1] + dy, c[2] + dz
                    if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                        return 0.0                    # ball pokes out of grid -> not robust
                    if not self.visible[ix, iy, iz]:
                        return 0.0                    # any blind neighbor -> not robust
                    worst = min(worst, float(self.D[ix, iy, iz]))
        return 0.0 if worst is np.inf else worst

    # -- world-frame convenience (transform cube world pose -> base-frame grasp point) --------------

    def _grasp_point_base(self, cube_world_pos, base_link_pos, base_link_quat) -> np.ndarray:
        """Cube world XYZ -> live ``base_link`` grasp target.

        The field was built in ``base_link`` coordinates.  Query with the live
        root pose, including roll/pitch; a yaw-projected planner pose is a
        different frame and can select a score for a point the body does not
        currently see or reach.
        """
        from mj_envs.tasks.visual_manipulation.curobo.planner import GRASP_Z_ABOVE_M
        grasp_world = np.asarray(cube_world_pos, dtype=np.float32).copy()
        # ``GRASP_Z_ABOVE_M`` is defined along the cube/world +Z axis. Apply it
        # before transforming: adding it after world->base would incorrectly
        # use base +Z while the walker is pitched or rolled.
        grasp_world[2] += GRASP_Z_ABOVE_M
        pos_b, _ = world_to_base_link_frame(
            torch.as_tensor(grasp_world, dtype=torch.float32).view(1, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),                 # quat unused for position
            torch.as_tensor(base_link_pos, dtype=torch.float32).view(3),
            torch.as_tensor(base_link_quat, dtype=torch.float32).view(4),
        )
        return pos_b.numpy()[0].astype(np.float64)

    def useful_world(self, cube_world_pos, base_pos, base_quat, arm: str) -> tuple[bool, float]:
        return self.useful(self._grasp_point_base(cube_world_pos, base_pos, base_quat), arm)

    def score_world(self, cube_world_pos, base_pos, base_quat, arm: str, delta_m: float = 0.05) -> float:
        return self.robust_score(self._grasp_point_base(cube_world_pos, base_pos, base_quat), arm, delta_m)

    def robust_grasp_points(self, grasp_z_base: float, tau: float, max_points: int = 12,
                            delta_m: float = 0.05, min_separation_m: float = 0.08) -> list[tuple[np.ndarray, str, float]]:
        """Return separated high-margin body-frame grasp points near one grasp height.

        This is the inverse-map source for base-stance selection: if a cube is at world ``g`` and a
        fixed-yaw body should see it at point ``p``, the desired base XY is ``g_xy - R(yaw) p_xy``.
        The field is only a candidate generator. Runtime fast IK and full cuRobo still decide whether a
        candidate is executable. ``tau`` is absolute, never a relative gain over a bad current stance.

        The dense field is right-arm canonical. Left-arm candidates mirror ``y`` exactly like
        ``_voxel``. Vectorized neighborhood minima make this setup-time query cheap enough to run once
        per target, without scanning Python ``robust_score`` calls over the full grid.
        """
        if tau <= 0.0:
            raise ValueError(f"tau must be positive; got {tau}")
        z = int(np.round((float(grasp_z_base) - self.origin[2]) / self.spacing))
        nx, ny, nz = self.n_per
        radius = self._robust_radius_voxels(delta_m)
        if z < radius or z >= nz - radius:
            return []
        score = np.full((nx, ny), np.inf, dtype=np.float32)
        valid = np.ones((nx, ny), dtype=bool)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dx * dx + dy * dy + dz * dz > radius * radius:
                        continue
                    src_x0, src_x1 = max(0, dx), min(nx, nx + dx)
                    src_y0, src_y1 = max(0, dy), min(ny, ny + dy)
                    dst_x0, dst_x1 = max(0, -dx), min(nx, nx - dx)
                    dst_y0, dst_y1 = max(0, -dy), min(ny, ny - dy)
                    patch_d = self.D[src_x0:src_x1, src_y0:src_y1, z + dz]
                    patch_v = self.visible[src_x0:src_x1, src_y0:src_y1, z + dz]
                    score[dst_x0:dst_x1, dst_y0:dst_y1] = np.minimum(
                        score[dst_x0:dst_x1, dst_y0:dst_y1], patch_d)
                    valid[dst_x0:dst_x1, dst_y0:dst_y1] &= patch_v
        score[~valid] = 0.0
        points: list[tuple[np.ndarray, str, float]] = []
        for ix, iy in np.argwhere(score >= tau)[np.argsort(score[score >= tau])[::-1]]:
            p_r = self.origin + self.spacing * np.array([ix, iy, z], dtype=np.float64)
            value = float(score[ix, iy])
            for arm, point in (("R", p_r), ("L", p_r * np.array([1.0, -1.0, 1.0]))):
                if all(np.linalg.norm(point[:2] - prior[0][:2]) >= min_separation_m
                       or arm != prior[1] for prior in points):
                    points.append((point, arm, value))
                    if len(points) >= max_points:
                        return points
        return points


# ---------------------------------------------------------------------------
# Curator entry
# ---------------------------------------------------------------------------

def curate_scenario(scenario, base_pose, robot: str = "v2", tau: float = _DEFAULT_TAU,
                    delta_m: float = 0.05, field: VisibleReachableField | None = None) -> list[dict]:
    """Score every ``scenario.pick`` cube per-arm from a nominal ``base_pose = (pos, quat_wxyz)``.

    Returns one dict per cube (ranked by best-arm ``robust`` score, descending)::

        {name, per_arm={"L":{visible,D_reach,robust}, "R":{...}}, best_arm, keep}

    ``keep`` = some arm is visible AND ``D_reach >= tau``. Pure analysis; the scenario is not mutated.
    """
    field = field or VisibleReachableField.from_robot(robot)
    base_pos, base_quat = base_pose
    results = []
    for m in scenario.pick:
        per_arm = {}
        for arm in ("L", "R"):
            vis, d = field.useful_world(m.pos, base_pos, base_quat, arm)
            rs = field.score_world(m.pos, base_pos, base_quat, arm, delta_m)
            per_arm[arm] = {"visible": vis, "D_reach": d, "robust": rs}
        best_arm = max(("L", "R"), key=lambda a: per_arm[a]["robust"])
        keep = any(per_arm[a]["visible"] and per_arm[a]["D_reach"] >= tau for a in ("L", "R"))
        results.append({"name": m.name, "per_arm": per_arm, "best_arm": best_arm, "keep": keep})
    results.sort(key=lambda r: r["per_arm"][r["best_arm"]]["robust"], reverse=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="v2", help="v2 | v2_fixed | g1")
    parser.add_argument("--payload", type=pathlib.Path, default=None, help="override fat payload path")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="override sidecar out path")
    args = parser.parse_args()
    build_field(args.robot, args.payload, args.out)


if __name__ == "__main__":
    main()
