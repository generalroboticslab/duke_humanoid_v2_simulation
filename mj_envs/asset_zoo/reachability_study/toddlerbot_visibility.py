"""Per-eye dynamic-head visibility for ToddlerBot reachability targets.

The source MJCF camera intrinsics encode the raw 160-degree-diagonal fisheye's pinhole envelope.
Visibility (`visible_eyes`/`target_visible`) requires legal dynamic head aim, each eye's FOV cone,
AND a clear line of sight to the target -- raycast against the robot's own group-2 visual mesh
(`eye_los_clear`), matching V2/G1's occlusion policy (`_camera_visibility` in
`plot_workspace_curobo.py`). No stereo matching, rectification, depth estimate, or midpoint camera
is involved. `eyes_in_fov_batch` (the live interactive viewer path) stays FOV-only by design --
see its docstring.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MJCF = _REPO_ROOT / "asset" / "toddlerbot_2xm_gripper" / "toddlerbot_2xm_gripper_pos.xml"
_EYES = (
    ("head_cam_left", "head_cam_left_site"),
    ("head_cam_right", "head_cam_right_site"),
)
_HEAD_AIM_SITE = "head_cam_left_site"
_HEAD_YAW_DRIVEN = "neck_yaw_driven"
_HEAD_YAW_DRIVE = "neck_yaw_drive"
_HEAD_PITCH = "neck_pitch"
_HEAD_YAW_DRIVE_RATIO = -1.0 / 0.9090909091
_GEOMGROUP_VISUAL = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)  # group-2 visual mesh only
_NEAR_PLANE_M = 0.10
_LOS_EPS = 1e-3


def load_model() -> mujoco.MjModel:
    """Compile source MJCF containing the authoritative head-camera sites."""
    return mujoco.MjModel.from_xml_path(str(_MJCF))


def eye_in_fov(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    site_name: str,
    target_world: np.ndarray,
) -> bool:
    """Return whether target center lies in one eye's raw-fisheye pinhole envelope."""
    camera_id = model.camera(camera_name).id
    site_id = model.site(site_name).id
    rotation = data.site_xmat[site_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(target_world, dtype=np.float64) - data.site_xpos[site_id])
    if local[2] <= 0.0 or np.linalg.norm(local) < _NEAR_PLANE_M:
        return False
    fx, fy, cx_offset, cy_offset = model.cam_intrinsic[camera_id]
    width, height = model.cam_resolution[camera_id]
    cx = width / 2.0 - 0.5 - cx_offset
    cy = height / 2.0 - 0.5 - cy_offset
    pixel_x = fx * local[0] / local[2] + cx
    pixel_y = fy * local[1] / local[2] + cy
    return 0.0 <= pixel_x <= width - 1.0 and 0.0 <= pixel_y <= height - 1.0


def eye_los_clear(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    target_world: np.ndarray,
    eye_body: int,
) -> bool:
    """Raycast origin->target_world against group-2 visual meshes; True if unoccluded.

    Mirrors `_camera_visibility`'s `_los_clear` (V2/G1, `plot_workspace_curobo.py`): same occluder
    definition (group-2 visual mesh, the true body surface) and same per-ray exclusion of the
    eye's own carrying body (so the lens housing can't self-occlude its own target).
    """
    vec = np.asarray(target_world, dtype=np.float64) - origin
    dist = float(np.linalg.norm(vec))
    if dist < 1e-9:
        return True
    geomid = np.zeros(1, dtype=np.int32)
    hit = mujoco.mj_ray(model, data, origin, vec / dist, _GEOMGROUP_VISUAL, True, eye_body, geomid)
    return hit < 0 or hit >= dist - _LOS_EPS


def visible_eyes(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> tuple[str, ...]:
    """Return names of eyes with target in FOV cone AND clear line of sight (mesh raycast)."""
    target_world = np.asarray(target_world, dtype=np.float64)
    result = []
    for camera_name, site_name in _EYES:
        if not eye_in_fov(model, data, camera_name, site_name, target_world):
            continue
        site_id = model.site(site_name).id
        origin = data.site_xpos[site_id]
        eye_body = int(model.site_bodyid[site_id])
        if eye_los_clear(model, data, origin, target_world, eye_body):
            result.append(camera_name)
    return tuple(result)


def eyes_in_fov_batch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
) -> np.ndarray:
    """Vectorized FOV test: return `(N, 2)` bool mask of which eye sees each target at current head pose.

    Mirrors `eye_in_fov`'s raw-fisheye pinhole-envelope test exactly: project target into the
    eye's site-local frame (cols are `data.site_xmat`), require `local[2] > 0` (in front), then
    pinhole-project with `cam_intrinsic`/`cam_resolution` and check pixel bounds. No occlusion --
    intentional, matches `_camera_visibility_instantaneous_v2`'s live-viewer tradeoff (BVH/raycast
    setup per slider change is too costly for interactive use); the offline scorer
    (`score_workspace_dynamic` -> `visible_eyes`) is occlusion-aware. Column order matches `_EYES`
    (`left`, `right`).
    """
    targets = np.asarray(target_world, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[None, :]
    mask = np.zeros((targets.shape[0], len(_EYES)), dtype=bool)
    for column, (camera_name, site_name) in enumerate(_EYES):
        camera_id = model.camera(camera_name).id
        site_id = model.site(site_name).id
        rotation = data.site_xmat[site_id].reshape(3, 3)
        local = (targets - data.site_xpos[site_id]) @ rotation  # (N, 3), site-local
        in_front = (local[:, 2] > 0.0) & (np.linalg.norm(local, axis=1) >= _NEAR_PLANE_M)
        fx, fy, cx_offset, cy_offset = model.cam_intrinsic[camera_id]
        width, height = model.cam_resolution[camera_id]
        cx = width / 2.0 - 0.5 - cx_offset
        cy = height / 2.0 - 0.5 - cy_offset
        pixel_x = fx * local[:, 0] / np.where(in_front, local[:, 2], 1.0) + cx
        pixel_y = fy * local[:, 1] / np.where(in_front, local[:, 2], 1.0) + cy
        inside = (
            in_front
            & (pixel_x >= 0.0)
            & (pixel_x <= width - 1.0)
            & (pixel_y >= 0.0)
            & (pixel_y <= height - 1.0)
        )
        mask[:, column] = inside
    return mask


def target_visible(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> bool:
    """Visibility OR over left/right physical fisheye eyes."""
    return bool(visible_eyes(model, data, target_world))


def aim_head_at_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    iterations: int = 40,
) -> tuple[float, float]:
    """Aim shared ToddlerBot neck yaw/pitch at target within physical joint limits.

    Uses source MuJoCo angular site Jacobian, so camera optical convention, joint-axis signs, and
    moving lens center remain source-authoritative. The left eye supplies the aim residual; both
    eyes are evaluated afterward at this one shared physical head state. Forty iterations are
    needed near coupled-neck ROM edges; six left false FOV holes along otherwise continuous rays.
    """
    yaw_joint = model.joint(_HEAD_YAW_DRIVEN)
    pitch_joint = model.joint(_HEAD_PITCH)
    yaw_drive_joint = model.joint(_HEAD_YAW_DRIVE)
    yaw_qpos = int(model.jnt_qposadr[yaw_joint.id])
    pitch_qpos = int(model.jnt_qposadr[pitch_joint.id])
    yaw_drive_qpos = int(model.jnt_qposadr[yaw_drive_joint.id])
    dof_index = np.array([model.jnt_dofadr[yaw_joint.id], model.jnt_dofadr[pitch_joint.id]], dtype=np.int32)
    yaw_limits = model.jnt_range[yaw_joint.id]
    pitch_limits = model.jnt_range[pitch_joint.id]
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
        forward = data.site_xmat[site_id].reshape(3, 3)[:, 2]
        error = np.cross(forward, desired)
        if float(np.linalg.norm(error)) < 1e-5:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        joint_jacobian = jacr[:, dof_index]
        delta = joint_jacobian.T @ np.linalg.solve(
            joint_jacobian @ joint_jacobian.T + 1e-5 * np.eye(3), error
        )
        data.qpos[yaw_qpos] = np.clip(data.qpos[yaw_qpos] + delta[0], *yaw_limits)
        data.qpos[pitch_qpos] = np.clip(data.qpos[pitch_qpos] + delta[1], *pitch_limits)
        data.qpos[yaw_drive_qpos] = _HEAD_YAW_DRIVE_RATIO * data.qpos[yaw_qpos]
        mujoco.mj_forward(model, data)

    return float(data.qpos[yaw_qpos]), float(data.qpos[pitch_qpos])


def score_workspace_dynamic(
    workspace: pathlib.Path,
    out: pathlib.Path,
    *,
    log_every: int = 2_000,
) -> None:
    """Solve head aim and stereo visibility for every successful arm-IK row.

    `workspace` targets are expressed in exported `base_link`; source MuJoCo keeps the robot at
    `spec.home_root_pos`, so targets are translated once into source world coordinates. The score
    requires FOV cone AND clear line of sight (group-2 visual-mesh raycast via `visible_eyes`),
    matching V2/G1's occlusion policy. Neck yaw/pitch remains dynamic and is solved separately for
    every reachable target voxel. SO(3) orientations sharing one voxel have identical camera
    geometry, so their result is broadcast exactly into row-aligned output.

    Output remains row-aligned with `workspace`: false/NaN entries are failed arm IK rows and are
    never considered by the visible-reachable aggregation.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    for path in (str(repo_root), str(repo_root / "mj_envs"), str(repo_root.parent / "curobo")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from mj_envs.asset_zoo.reachability_study.generate_workspace_curobo import get_robot_spec

    payload = torch.load(str(workspace), map_location="cpu", weights_only=False)
    if payload.get("robot") != "toddlerbot":
        raise ValueError(f"expected toddlerbot workspace, got {payload.get('robot')!r}")
    success_idx = torch.where(payload["success"].bool())[0]
    if success_idx.numel() == 0:
        raise ValueError("workspace contains no successful arm-IK rows")

    spec = get_robot_spec("toddlerbot")
    model = spec.model_loader()
    data = mujoco.MjData(model)
    n_rows = int(payload["success"].numel())
    visible_left = torch.zeros(n_rows, dtype=torch.bool)
    visible_right = torch.zeros(n_rows, dtype=torch.bool)
    head_q = torch.full((n_rows, 2), float("nan"), dtype=torch.float32)
    target_offset = np.asarray(spec.home_root_pos, dtype=np.float64)
    success_voxel = payload["voxel_index"][success_idx].numpy()
    _, first_offset, inverse = np.unique(success_voxel, return_index=True, return_inverse=True)
    representative_idx = success_idx.numpy()[first_offset]
    n_voxels = len(representative_idx)
    voxel_row_count = np.bincount(inverse, minlength=n_voxels)
    voxel_left = np.zeros(n_voxels, dtype=bool)
    voxel_right = np.zeros(n_voxels, dtype=bool)
    voxel_head_q = np.full((n_voxels, 2), np.nan, dtype=np.float32)

    for number, idx in enumerate(representative_idx, start=1):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = target_offset
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        spec.apply_source_coupling(model, data)
        mujoco.mj_forward(model, data)
        target_world = payload["target_pos"][idx].numpy() + target_offset
        voxel_head_q[number - 1] = aim_head_at_target(model, data, target_world)
        eyes = visible_eyes(model, data, target_world)
        voxel_left[number - 1] = "head_cam_left" in eyes
        voxel_right[number - 1] = "head_cam_right" in eyes
        if number % log_every == 0 or number == n_voxels:
            visible_rows = int(voxel_row_count[voxel_left | voxel_right].sum())
            print(
                f"dynamic visibility {number}/{n_voxels} voxels: "
                f"visible rows={visible_rows}/{int(success_idx.numel())} "
                f"({100 * visible_rows / int(success_idx.numel()):.1f}%)",
                flush=True,
            )

    visible_left[success_idx] = torch.from_numpy(voxel_left[inverse])
    visible_right[success_idx] = torch.from_numpy(voxel_right[inverse])
    head_q[success_idx] = torch.from_numpy(voxel_head_q[inverse])

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "robot": "toddlerbot",
        "workspace": str(workspace.resolve()),
        "workspace_rows": n_rows,
        "success_rows": int(success_idx.numel()),
        "visible_left": visible_left,
        "visible_right": visible_right,
        "head_q": head_q,
        "camera_mode": "dynamic_neck_aim_per_success_voxel",
        "camera_projection": "arducam_imx291_fisheye_envelope_near0p10m",
    }, out)
    visible_any = visible_left | visible_right
    print(
        f"saved {out}: {int(visible_any[success_idx].sum())}/{int(success_idx.numel())} "
        "successful arm-IK rows visible by at least one eye"
    )


__all__ = [
    "aim_head_at_target", "eye_in_fov", "eye_los_clear", "eyes_in_fov_batch", "load_model",
    "score_workspace_dynamic", "target_visible", "visible_eyes",
]
