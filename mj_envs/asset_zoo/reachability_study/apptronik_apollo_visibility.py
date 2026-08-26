"""Dynamic stereo visibility for Apollo reachability rows.

Apollo's source MJCF omits physical eye calibration. This scorer therefore evaluates only the
user-approved provisional 16:9 pinhole envelope (91.5° horizontal × 60° vertical), never a
sensor-performance claim. It scores each workspace voxel once at the approved symmetric-down
source ``qpos0`` arm pose, solves one legal shared 3-DOF neck pose, then requires each physical
eye to pass its FOV envelope and a MuJoCo raycast against source group-1 visual meshes. Stereo
visibility is the eye-wise OR and is copied to every successful wrist orientation in that voxel.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import pathlib
import sys

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_EYES = (
    ("head_cam_left", "head_cam_left_site"),
    ("head_cam_right", "head_cam_right_site"),
)
_HEAD_AIM_SITE = "head_cam_left_site"
_HEAD_JOINTS = ("neck_yaw", "neck_roll", "neck_pitch")
_ASPECT_RATIO = 16.0 / 9.0
_NEAR_PLANE_M = 0.10
_LOS_EPS = 1e-3
_VISUAL_GROUP = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
_VISIBILITY_CHUNK_ROWS = 2_000
_AIM_FALLBACK_FRACTIONS = (0.0, 0.5, 1.0)
_AIM_FALLBACK_MARGIN_RAD = np.deg2rad(3.0)

# Forked workers inherit these read-only CPU tensors copy-on-write. MuJoCo model/data are created
# in each worker, because they are mutated while reconstructing one successful arm pose at a time.
_WORKSPACE_PAYLOAD: dict | None = None
_WORKSPACE_SPEC = None
_WORKSPACE_ROOT_OFFSET: np.ndarray | None = None
_WORKER_MODEL: mujoco.MjModel | None = None
_WORKER_DATA: mujoco.MjData | None = None


def _camera_half_angles(model: mujoco.MjModel, camera_name: str) -> tuple[float, float]:
    """Return provisional horizontal/vertical half angles from approved 16:9 envelope."""
    camera_id = model.camera(camera_name).id
    v_half = np.deg2rad(float(model.cam_fovy[camera_id])) / 2.0
    return np.arctan(_ASPECT_RATIO * np.tan(v_half)), v_half


def eye_in_fov(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    site_name: str,
    target_world: np.ndarray,
) -> bool:
    """Test one target against one MuJoCo ``-Z`` camera's provisional FOV envelope."""
    camera_id = model.camera(camera_name).id
    site_id = model.site(site_name).id
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(target_world, dtype=np.float64) - data.site_xpos[site_id])
    forward = -float(local[2])
    if forward <= 1e-9 or np.linalg.norm(local) < _NEAR_PLANE_M:
        return False
    h_half, v_half = _camera_half_angles(model, camera_name)
    return abs(float(np.arctan2(local[0], forward))) <= h_half and abs(float(np.arctan2(local[1], forward))) <= v_half


def eye_los_clear(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    target_world: np.ndarray,
    eye_body: int,
) -> bool:
    """Raycast one eye through source visual group 1, excluding its carrying head body."""
    vector = np.asarray(target_world, dtype=np.float64) - origin
    distance = float(np.linalg.norm(vector))
    if distance < 1e-9:
        return True
    geomid = np.full(1, -1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, origin, vector / distance, _VISUAL_GROUP, True, eye_body, geomid)
    return hit < 0.0 or hit >= distance - _LOS_EPS


def visible_eyes(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> tuple[str, ...]:
    """Return physical eyes that independently pass FOV and visual-mesh LOS."""
    result = []
    for camera_name, site_name in _EYES:
        if not eye_in_fov(model, data, camera_name, site_name, target_world):
            continue
        site_id = model.site(site_name).id
        if eye_los_clear(model, data, data.site_xpos[site_id], target_world, int(model.site_bodyid[site_id])):
            result.append(camera_name)
    return tuple(result)


def eyes_in_fov_batch(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> np.ndarray:
    """Return FOV-only `(N,2)` eye mask at current shared neck state for live viewing.

    Deliberately omits raycasts: live slider changes must redraw without rebuilding a mesh-BVH.
    Offline ``score_workspace_dynamic`` remains the occlusion-aware authority.
    """
    targets = np.atleast_2d(np.asarray(target_world, dtype=np.float64))
    mask = np.zeros((targets.shape[0], len(_EYES)), dtype=bool)
    for column, (camera_name, site_name) in enumerate(_EYES):
        camera_id = model.camera(camera_name).id
        site_id = model.site(site_name).id
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        local = (targets - data.site_xpos[site_id]) @ rotation
        forward = -local[:, 2]
        h_half, v_half = _camera_half_angles(model, camera_name)
        mask[:, column] = (
            (forward > 1e-9)
            & (np.linalg.norm(targets - data.site_xpos[site_id], axis=1) >= _NEAR_PLANE_M)
            & (np.abs(np.arctan2(local[:, 0], np.where(forward > 0.0, forward, 1.0))) <= h_half)
            & (np.abs(np.arctan2(local[:, 1], np.where(forward > 0.0, forward, 1.0))) <= v_half)
        )
    return mask


def aim_head_at_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    *,
    iterations: int = 40,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Aim shared legal Apollo neck at target using source site angular Jacobian.

    The left eye's local ``+X`` site axis is optical forward by construction. The same solved
    neck state then drives both physical camera frames. Damped least squares avoids singular
    pitch/roll directions while retaining source joint limits.
    """
    joints = [model.joint(name) for name in _HEAD_JOINTS]
    qpos = np.array([model.jnt_qposadr[joint.id] for joint in joints], dtype=np.int32)
    dofs = np.array([model.jnt_dofadr[joint.id] for joint in joints], dtype=np.int32)
    limits = np.array([model.jnt_range[joint.id] for joint in joints], dtype=np.float64)
    site_id = model.site(_HEAD_AIM_SITE).id
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    for _ in range(iterations):
        origin = data.site_xpos[site_id]
        direction = np.asarray(target_world, dtype=np.float64) - origin
        distance = float(np.linalg.norm(direction))
        if distance < 1e-9:
            break
        desired = direction / distance
        forward = data.site_xmat[site_id].reshape(3, 3)[:, 0]
        error = np.cross(forward, desired)
        if float(np.linalg.norm(error)) < tolerance:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        angular = jacr[:, dofs]
        delta = angular.T @ np.linalg.solve(angular @ angular.T + 1e-5 * np.eye(3), error)
        data.qpos[qpos] = np.clip(data.qpos[qpos] + delta, limits[:, 0], limits[:, 1])
        mujoco.mj_forward(model, data)
    return data.qpos[qpos].astype(np.float32).copy()


def _near_fov_boundary(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> bool:
    """Whether a DLS miss is close enough to an eye FOV edge to justify fallback search."""
    for camera_name, site_name in _EYES:
        camera_id = model.camera(camera_name).id
        site_id = model.site(site_name).id
        rotation = data.cam_xmat[camera_id].reshape(3, 3)
        local = rotation.T @ (np.asarray(target_world, dtype=np.float64) - data.site_xpos[site_id])
        forward = -float(local[2])
        if forward <= 1e-9:
            continue
        h_half, v_half = _camera_half_angles(model, camera_name)
        h_overrun = abs(float(np.arctan2(local[0], forward))) - h_half
        v_overrun = abs(float(np.arctan2(local[1], forward))) - v_half
        if max(h_overrun, v_overrun) <= _AIM_FALLBACK_MARGIN_RAD:
            return True
    return False


def _fallback_visible_eyes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    primary_q: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Try legal neck-limit lattice when DLS misses a target inside an eye FOV.

    The dynamic policy asks whether *any* legal common-neck state sees a voxel. A single local
    DLS solve can stop short near coupled roll/pitch limits, creating isolated blind labels even
    with clear LOS. This fallback tests 27 deterministic low/mid/high joint states only after
    the primary solution fails both eyes, and returns the closest successful state to that solve.
    It deliberately does not alter FOV, near-plane, or occluder policy. Candidates are probed
    nearest-first and stop at first visible state; caller invokes this only within 3 degrees of
    an eye boundary, so true rear blind regions do not pay this search cost.
    """
    joints = [model.joint(name) for name in _HEAD_JOINTS]
    qpos = np.array([model.jnt_qposadr[joint.id] for joint in joints], dtype=np.int32)
    limits = np.array([model.jnt_range[joint.id] for joint in joints], dtype=np.float64)
    candidates = limits[:, 0, None] + (limits[:, 1] - limits[:, 0])[:, None] * np.asarray(
        _AIM_FALLBACK_FRACTIONS, dtype=np.float64
    )
    span = limits[:, 1] - limits[:, 0]
    lattice = [
        candidates[np.arange(len(_HEAD_JOINTS)), selection]
        for selection in itertools.product(range(len(_AIM_FALLBACK_FRACTIONS)), repeat=len(_HEAD_JOINTS))
    ]
    lattice.sort(key=lambda candidate: float(np.sum(((candidate - primary_q) / span) ** 2)))
    for candidate_q in lattice:
        data.qpos[qpos] = candidate_q
        mujoco.mj_forward(model, data)
        eyes = visible_eyes(model, data, target_world)
        if eyes:
            return eyes, candidate_q.astype(np.float32)
    data.qpos[qpos] = primary_q
    mujoco.mj_forward(model, data)
    return (), primary_q


def _init_visibility_worker() -> None:
    """Create worker-local mutable MuJoCo state after forked payload inheritance."""
    global _WORKER_MODEL, _WORKER_DATA
    if _WORKSPACE_SPEC is None:
        raise RuntimeError("visibility worker missing workspace spec")
    _WORKER_MODEL = _WORKSPACE_SPEC.model_loader()
    _WORKER_DATA = mujoco.MjData(_WORKER_MODEL)


def _score_success_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score representative voxel rows at symmetric-down source rest.

    Wrist orientation must not change camera sightline classification. Reaching-arm self-occlusion
    made duplicate target voxels disagree, so source arms remain in approved neutral pose.
    """
    if any(value is None for value in (
        _WORKSPACE_PAYLOAD, _WORKSPACE_SPEC, _WORKSPACE_ROOT_OFFSET,
        _WORKER_MODEL, _WORKER_DATA,
    )):
        raise RuntimeError("visibility worker not initialized")
    payload = _WORKSPACE_PAYLOAD
    spec = _WORKSPACE_SPEC
    root_offset = _WORKSPACE_ROOT_OFFSET
    model = _WORKER_MODEL
    data = _WORKER_DATA
    visible_left = np.zeros(indices.shape[0], dtype=bool)
    visible_right = np.zeros(indices.shape[0], dtype=bool)
    head_q = np.empty((indices.shape[0], len(_HEAD_JOINTS)), dtype=np.float32)
    target_pos = payload["target_pos"]
    for row, idx in enumerate(indices.tolist()):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root_offset
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        spec.apply_source_coupling(model, data)
        mujoco.mj_forward(model, data)
        target_world = target_pos[idx].numpy() + root_offset
        primary_q = aim_head_at_target(model, data, target_world)
        eyes = visible_eyes(model, data, target_world)
        if not eyes and _near_fov_boundary(model, data, target_world):
            eyes, primary_q = _fallback_visible_eyes(model, data, target_world, primary_q)
        head_q[row] = primary_q
        visible_left[row] = "head_cam_left" in eyes
        visible_right[row] = "head_cam_right" in eyes
    return indices, visible_left, visible_right, head_q


def _score_isolated_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Re-score topologically isolated blind voxels with exhaustive legal neck candidates.

    This is solver recovery, not label smoothing: a voxel changes only when a concrete shared-neck
    configuration passes the unchanged physical-eye FOV, 0.10 m near plane, and visual-mesh LOS.
    """
    if any(value is None for value in (
        _WORKSPACE_PAYLOAD, _WORKSPACE_SPEC, _WORKSPACE_ROOT_OFFSET,
        _WORKER_MODEL, _WORKER_DATA,
    )):
        raise RuntimeError("visibility worker not initialized")
    payload = _WORKSPACE_PAYLOAD
    spec = _WORKSPACE_SPEC
    root_offset = _WORKSPACE_ROOT_OFFSET
    model = _WORKER_MODEL
    data = _WORKER_DATA
    visible_left = np.zeros(indices.shape[0], dtype=bool)
    visible_right = np.zeros(indices.shape[0], dtype=bool)
    head_q = np.empty((indices.shape[0], len(_HEAD_JOINTS)), dtype=np.float32)
    target_pos = payload["target_pos"]
    for row, idx in enumerate(indices.tolist()):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root_offset
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        spec.apply_source_coupling(model, data)
        mujoco.mj_forward(model, data)
        target_world = target_pos[idx].numpy() + root_offset
        primary_q = aim_head_at_target(model, data, target_world)
        eyes, primary_q = _fallback_visible_eyes(model, data, target_world, primary_q)
        visible_left[row] = "head_cam_left" in eyes
        visible_right[row] = "head_cam_right" in eyes
        head_q[row] = primary_q
    return indices, visible_left, visible_right, head_q


def _isolated_blind_compact_indices(
    unique_voxels: np.ndarray,
    visible_any: np.ndarray,
    n_grid_per_axis: np.ndarray,
) -> np.ndarray:
    """Return successful R-workspace voxels with at least five visible face-neighbours."""
    nx, ny, nz = (int(value) for value in n_grid_per_axis)
    compact_for_voxel = {int(voxel): compact for compact, voxel in enumerate(unique_voxels.tolist())}
    isolated: list[int] = []
    for compact, voxel in enumerate(unique_voxels.tolist()):
        if visible_any[compact]:
            continue
        ix = voxel // (ny * nz)
        iy = (voxel % (ny * nz)) // nz
        iz = voxel % nz
        neighbours = 0
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            xx, yy, zz = ix + dx, iy + dy, iz + dz
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                neighbour = compact_for_voxel.get(xx * ny * nz + yy * nz + zz)
                neighbours += neighbour is not None and bool(visible_any[neighbour])
        if neighbours >= 5:
            isolated.append(compact)
    return np.asarray(isolated, dtype=np.int64)


def _bimanual_isolated_blind_compact_indices(
    unique_voxels: np.ndarray,
    visible_any: np.ndarray,
    n_grid_per_axis: np.ndarray,
) -> np.ndarray:
    """Return source-R rows behind isolated blind cells in displayed R∪mirrored-L field."""
    nx, ny, nz = (int(value) for value in n_grid_per_axis)

    def mirror(voxel: int) -> int:
        ix = voxel // (ny * nz)
        iy = (voxel % (ny * nz)) // nz
        iz = voxel % nz
        return ix * ny * nz + (ny - 1 - iy) * nz + iz

    compact_for_voxel = {int(voxel): compact for compact, voxel in enumerate(unique_voxels.tolist())}
    bimanual_visible: dict[int, bool] = {}
    for voxel, compact in compact_for_voxel.items():
        mirrored = mirror(voxel)
        bimanual_visible[voxel] = bool(visible_any[compact]) or bimanual_visible.get(voxel, False)
        bimanual_visible[mirrored] = bool(visible_any[compact]) or bimanual_visible.get(mirrored, False)
    repair: set[int] = set()
    for voxel, is_visible in bimanual_visible.items():
        if is_visible:
            continue
        ix = voxel // (ny * nz)
        iy = (voxel % (ny * nz)) // nz
        iz = voxel % nz
        visible_neighbours = 0
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            xx, yy, zz = ix + dx, iy + dy, iz + dz
            neighbour = xx * ny * nz + yy * nz + zz
            visible_neighbours += (
                0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz and bimanual_visible.get(neighbour, False)
            )
        if visible_neighbours < 5:
            continue
        for source_voxel in (voxel, mirror(voxel)):
            compact = compact_for_voxel.get(source_voxel)
            if compact is not None:
                repair.add(compact)
    return np.fromiter(sorted(repair), dtype=np.int64)


def score_workspace_dynamic(
    workspace: pathlib.Path,
    out: pathlib.Path,
    *,
    log_every: int = 2_000,
    workers: int = 1,
) -> None:
    """Score every successful Apollo voxel with dynamic shared-neck stereo FOV∧LOS.

    Arm geometry stays at approved symmetric-down source rest. Failed IK rows remain false/NaN
    so visible dexterity preserves the workspace denominator.
    """
    for path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs"), str(_REPO_ROOT / "curobo")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from mj_envs.asset_zoo.reachability_study.generate_workspace_curobo import get_robot_spec
    payload = torch.load(str(workspace), map_location="cpu", weights_only=False)
    if payload.get("robot") != "apptronik_apollo":
        raise ValueError(f"expected apptronik_apollo workspace, got {payload.get('robot')!r}")
    success_idx = torch.where(payload["success"].bool())[0]
    if success_idx.numel() == 0:
        raise ValueError("workspace contains no successful arm-IK rows")
    if workers <= 0:
        raise ValueError("workers must be positive")
    spec = get_robot_spec("apptronik_apollo")
    n_rows = int(payload["success"].numel())
    visible_left = torch.zeros(n_rows, dtype=torch.bool)
    visible_right = torch.zeros(n_rows, dtype=torch.bool)
    head_q = torch.full((n_rows, len(_HEAD_JOINTS)), float("nan"), dtype=torch.float32)
    root_offset = np.asarray(spec.home_root_pos, dtype=np.float64)
    global _WORKSPACE_PAYLOAD, _WORKSPACE_SPEC, _WORKSPACE_ROOT_OFFSET
    _WORKSPACE_PAYLOAD = payload
    _WORKSPACE_SPEC = spec
    _WORKSPACE_ROOT_OFFSET = root_offset
    success_rows = success_idx.numpy()
    voxel_index = payload["voxel_index"].numpy()
    unique_voxels, representative_idx = np.unique(voxel_index[success_rows], return_index=True)
    representative_rows = success_rows[representative_idx]
    row_to_voxel = np.searchsorted(unique_voxels, voxel_index)
    chunks = [chunk for chunk in np.array_split(
        representative_rows,
        max(1, int(np.ceil(representative_rows.size / _VISIBILITY_CHUNK_ROWS))),
    ) if chunk.size]
    scorer = _score_success_indices
    if workers == 1:
        _init_visibility_worker()
        results = map(scorer, chunks)
    else:
        # CUDA is never initialized here; fork shares the 1.4 GB CPU payload without a copy.
        context = mp.get_context("fork")
        pool = context.Pool(processes=workers, initializer=_init_visibility_worker)
        results = pool.imap_unordered(scorer, chunks, chunksize=1)
    voxel_visible_left = np.zeros(unique_voxels.size, dtype=bool)
    voxel_visible_right = np.zeros(unique_voxels.size, dtype=bool)
    voxel_head_q = np.empty((unique_voxels.size, len(_HEAD_JOINTS)), dtype=np.float32)
    processed = 0
    next_log = log_every
    try:
        for indices, left, right, solved_head_q in results:
            compact = row_to_voxel[indices]
            voxel_visible_left[compact] = left
            voxel_visible_right[compact] = right
            voxel_head_q[compact] = solved_head_q
            processed += int(indices.size)
            if processed >= next_log or processed == representative_rows.size:
                visible_any = voxel_visible_left | voxel_visible_right
                print(
                    f"Apollo dynamic visibility {processed}/{representative_rows.size} voxels: "
                    f"visible={int(visible_any.sum())} ({100 * float(visible_any.mean()):.1f}%)",
                    flush=True,
                )
                next_log += log_every
    finally:
        if workers > 1:
            pool.close()
            pool.join()
    isolated = _isolated_blind_compact_indices(
        unique_voxels,
        voxel_visible_left | voxel_visible_right,
        payload["n_grid_per_axis"].numpy(),
    )
    bimanual_isolated = _bimanual_isolated_blind_compact_indices(
        unique_voxels,
        voxel_visible_left | voxel_visible_right,
        payload["n_grid_per_axis"].numpy(),
    )
    isolated = np.unique(np.concatenate((isolated, bimanual_isolated)))
    if isolated.size:
        _init_visibility_worker()
        indices, left, right, solved_head_q = _score_isolated_indices(representative_rows[isolated])
        compact = row_to_voxel[indices]
        voxel_visible_left[compact] = left
        voxel_visible_right[compact] = right
        voxel_head_q[compact] = solved_head_q
    success_compact = row_to_voxel[success_rows]
    visible_left[success_idx] = torch.from_numpy(voxel_visible_left[success_compact])
    visible_right[success_idx] = torch.from_numpy(voxel_visible_right[success_compact])
    head_q[success_idx] = torch.from_numpy(voxel_head_q[success_compact])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "robot": "apptronik_apollo",
        "workspace": str(workspace.resolve()),
        "workspace_rows": n_rows,
        "success_rows": int(success_idx.numel()),
        "visible_left": visible_left,
        "visible_right": visible_right,
        "head_q": head_q,
        "head_joint_names": _HEAD_JOINTS,
        "camera_mode": "dynamic_shared_3dof_neck_aim_per_voxel_symmetric_down_arms",
        "camera_projection": "provisional_16_9_pinhole_fovy60_fovx91p49_near0p10m",
        "occluders": "source_group1_visual_meshes_excluding_eye_carrying_body",
    }, out)
    visible_any = visible_left | visible_right
    print(
        f"saved {out}: {int(visible_any[success_idx].sum())}/{int(success_idx.numel())} "
        "successful arm-IK rows visible by at least one provisional eye"
    )
