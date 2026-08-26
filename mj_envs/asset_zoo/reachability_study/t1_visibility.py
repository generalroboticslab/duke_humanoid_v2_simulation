"""Per-target dynamic-head visibility for Booster T1 reachability rows.

Booster T1 RealSense version = single Intel RealSense D455 module on the head (verified 2026-07-18
via Booster T1 Instruction Manual V1.0 + D455 datasheet). Source MJCF declares one
``<camera name="head_cam">`` at H2 forward (0.01, 0, 0.11) with fovy=65 matching the D455 RGB
vertical FOV (90° h × 65° v RGB, 87° h × 59° v depth). The module uses the RGB envelope as the
declared workspace visibility field; depth-frame scoring would require a separate, calibrated
depth-camera intrinsic. Documented envelope approximation per SOP §3, NOT a calibrated claim.

T1 has a single physical eye so per-target scoring is single-eye (no stereo OR). The head is
actuated: ``AAHead_yaw`` (±1.57 rad z-axis) + ``Head_pitch`` (-0.35 to 1.22 rad y-axis). Both
joints sit on the serial chain (no sibling drive/equality branch in source), so head state is
written directly into source qpos and FK'd through ``mj_forward`` -- no gearbox writeback needed.

Visibility per row = (per-target closed-form head aim within joint limits) AND (target inside
RGB envelope FOV) AND (no source collision geom between head_cam_site and target).
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MJCF = _REPO_ROOT / "asset" / "booster_t1" / "t1.xml"
_EYES = (("head_cam", "head_cam_site"),)
_HEAD_AIM_SITE = "head_cam_site"
_HEAD_YAW_JOINT = "AAHead_yaw"
_HEAD_PITCH_JOINT = "Head_pitch"


def load_model() -> mujoco.MjModel:
    """Compile source MJCF carrying the head_cam + head_cam_site declarations."""
    return mujoco.MjModel.from_xml_path(str(_MJCF))


def _camera_intrinsics(model: mujoco.MjModel, camera_name: str) -> tuple[float, float, float, float, int, int]:
    """Resolve (fx, fy, cx, cy, width, height) for the D455 envelope.

    T1 RealSense version uses a single Intel RealSense D455 module. Per SOP §3, the FOV envelope
    is documented but NOT calibrated -- always derive from the declared vertical FOV and
    resolution. Pixel coords assume centered principal point.
    """
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    assert camera_id >= 0, f"camera {camera_name!r} absent from source MJCF"
    width, height = model.cam_resolution[camera_id]
    fovy_deg = float(model.cam_fovy[camera_id])
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    # Isotropic (square-pixel) pinhole: fx = fy, NOT fy*(width/height). Horizontal FOV is then
    # DERIVED from fovy + aspect ratio (atan((width/2)/fx) ~= 45.5 deg, matching the declared 90 deg
    # h spec), not independently declared -- fy*(width/height) would instead force fovx == fovy
    # (65 deg), silently narrowing the horizontal envelope by ~25 deg.
    fx = fy
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return fx, fy, cx, cy, width, height


def eye_in_fov(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    site_name: str,
    target_world: np.ndarray,
) -> bool:
    """Return whether target center projects inside one eye's RGB envelope.

    Uses MuJoCo camera convention: cameras look along their local ``-Z`` axis. The site is
    co-located with the camera body so its local frame matches the camera frame; we therefore
    require the target's local Z component to be NEGATIVE (in front of the camera) for the
    FOV gate to be evaluated.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    assert site_id >= 0, f"site {site_name!r} absent from source MJCF"
    rotation = data.site_xmat[site_id].reshape(3, 3)
    local = rotation.T @ (np.asarray(target_world, dtype=np.float64) - data.site_xpos[site_id])
    if local[2] >= 0.0:
        return False  # target behind camera (MuJoCo camera looks along local -Z)
    dist = np.linalg.norm(np.asarray(target_world, dtype=np.float64) - data.site_xpos[site_id])
    if dist < 0.1:
        return False  # target closer than shared workspace visibility near clip (0.1 m)
    fx, fy, cx, cy, width, height = _camera_intrinsics(model, camera_name)
    # T1's head_cam mount quat (-90 deg about Y, to point local -Z along +X) leaves local X =
    # vertical, local Y = horizontal (verified: rot[:,0]=[0,0,1], rot[:,1]=[0,1,0] in H2 frame) --
    # opposite of the generic X=horizontal/Y=vertical camera convention. fx (from width) pairs
    # with local[1]; fy (from height) pairs with local[0].
    pixel_x = fx * local[1] / (-local[2]) + cx
    pixel_y = fy * local[0] / (-local[2]) + cy
    return 0.0 <= pixel_x <= width - 1.0 and 0.0 <= pixel_y <= height - 1.0


def eye_ray_clear(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    target_world: np.ndarray,
) -> bool:
    """Return whether no source visual mesh (group 2) before target occludes this eye.

    The originating camera body is excluded from the raycast (``bodyexclude``) so the camera's own
    housing is not treated as an occluder.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    origin = data.site_xpos[site_id]
    direction = np.asarray(target_world, dtype=np.float64) - origin
    distance = float(np.linalg.norm(direction))
    if distance < 1e-9:
        return True
    hit_geom = np.full(1, -1, dtype=np.int32)
    geomgroup = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)  # group-2 visual meshes only
    hit_distance = mujoco.mj_ray(
        model,
        data,
        origin,
        direction / distance,
        geomgroup,
        True,
        int(model.site_bodyid[site_id]),
        hit_geom,
    )
    return hit_distance < 0.0 or hit_distance >= distance - 1e-3


def visible_eyes(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> tuple[str, ...]:
    """Return names of eyes that independently pass FOV and occlusion gates."""
    return tuple(
        camera_name
        for camera_name, site_name in _EYES
        if eye_in_fov(model, data, camera_name, site_name, target_world)
        and eye_ray_clear(model, data, site_name, target_world)
    )


def target_visible(model: mujoco.MjModel, data: mujoco.MjData, target_world: np.ndarray) -> bool:
    """Visibility for the single T1 D455 module (no stereo OR needed)."""
    return bool(visible_eyes(model, data, target_world))


def eye_in_fov_batch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
) -> np.ndarray:
    """Vectorized FOV test for T1's single D455 module: return `(N, 1)` bool mask.

    Mirrors `eye_in_fov`'s envelope test exactly: project target into the site-local frame (cols
    are `data.site_xmat`), require `local[2] < 0` (in front of the camera -- MuJoCo camera looks
    along local `-Z`), then pinhole-project with intrinsics derived from `cam_fovy`/
    `cam_resolution` and check pixel bounds. No occlusion -- intentional, matches the live-viewer
    tradeoff (BVH/raycast setup per slider change is too costly for interactive use); the offline
    scorer (`score_workspace_dynamic` -> `visible_eyes`) is occlusion-aware. Column 0 matches
    `head_cam`/`head_cam_site`.
    """
    targets = np.asarray(target_world, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[None, :]
    camera_name, site_name = _EYES[0]
    camera_id = model.camera(camera_name).id
    site_id = model.site(site_name).id
    rotation = data.site_xmat[site_id].reshape(3, 3)
    local = (targets - data.site_xpos[site_id]) @ rotation  # (N, 3), site-local
    in_front = local[:, 2] < 0.0
    fx, fy, cx, cy, width, height = _camera_intrinsics(model, camera_name)
    z = np.where(in_front, -local[:, 2], 1.0)  # use -Z (forward distance)
    # local X = vertical, local Y = horizontal for T1's mount quat -- see eye_in_fov.
    pixel_x = fx * local[:, 1] / z + cx
    pixel_y = fy * local[:, 0] / z + cy
    dist = np.linalg.norm(targets - data.site_xpos[site_id], axis=1)
    inside = (
        in_front
        & (dist >= 0.1)
        & (pixel_x >= 0.0)
        & (pixel_x <= width - 1.0)
        & (pixel_y >= 0.0)
        & (pixel_y <= height - 1.0)
    )
    return inside.reshape(-1, 1)


def aim_head_at_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    iterations: int = 40,
    tolerance: float = 1e-4,
) -> tuple[float, float]:
    """Aim T1 head yaw/pitch at target analytically (closed-form)."""
    yaw_joint = model.joint(_HEAD_YAW_JOINT)
    pitch_joint = model.joint(_HEAD_PITCH_JOINT)
    yaw_qpos = int(model.jnt_qposadr[yaw_joint.id])
    pitch_qpos = int(model.jnt_qposadr[pitch_joint.id])
    yaw_limits = model.jnt_range[yaw_joint.id]
    pitch_limits = model.jnt_range[pitch_joint.id]

    # Pitch joint pivot in the base frame:
    # x_p = 0.0625, y_p = 0.0, z_p = 0.30485
    # Camera site offset: z_s = 0.12 (upward)
    x_p, y_p, z_p = 0.0625, 0.0, 0.30485
    z_s = 0.12

    base_pos = data.qpos[:3]
    t_base = np.asarray(target_world, dtype=np.float64) - base_pos

    dx = t_base[0] - x_p
    dy = t_base[1] - y_p
    dz = t_base[2] - z_p

    yaw = np.atan2(dy, dx)
    yaw = float(np.clip(yaw, *yaw_limits))

    dx_yawed = dx * np.cos(yaw) + dy * np.sin(yaw)
    R = np.sqrt(dx_yawed**2 + dz**2)
    if R < z_s:
        pitch = np.atan2(dx_yawed, dz)
    else:
        pitch = np.atan2(dx_yawed, dz) - np.arccos(z_s / R)

    pitch = float(np.clip(pitch, *pitch_limits))

    data.qpos[yaw_qpos] = yaw
    data.qpos[pitch_qpos] = pitch
    mujoco.mj_forward(model, data)

    return yaw, pitch


def score_workspace_dynamic(
    workspace: pathlib.Path,
    out: pathlib.Path,
    *,
    log_every: int = 2_000,
) -> None:
    """Solve head aim and per-target visibility for every successful arm-IK row.

    Target positions are expressed in the robot base frame. During visibility checking,
    the robot is held at the neutral/arms-down pose (qpos0) rather than the reaching posture,
    matching G1/V2/Toddlerbot occlusion policies. Head yaw/pitch is solved dynamically
    per target voxel. SO(3) orientations sharing one voxel have identical camera geometry,
    so visibility is broadcast into row-aligned output.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    for path in (str(repo_root), str(repo_root / "mj_envs"), str(repo_root.parent / "curobo")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from mj_envs.asset_zoo.reachability_study.generate_workspace_curobo import get_robot_spec

    payload = torch.load(str(workspace), map_location="cpu", weights_only=False)
    if payload.get("robot") != "booster_t1":
        raise ValueError(f"expected booster_t1 workspace, got {payload.get('robot')!r}")
    success_idx = torch.where(payload["success"].bool())[0]
    if success_idx.numel() == 0:
        raise ValueError("workspace contains no successful arm-IK rows")

    spec = get_robot_spec("booster_t1")
    model = spec.model_loader()
    data = mujoco.MjData(model)
    n_rows = int(payload["success"].numel())
    visible = torch.zeros(n_rows, dtype=torch.bool)
    head_q = torch.full((n_rows, 2), float("nan"), dtype=torch.float32)
    target_offset = np.asarray(spec.home_root_pos, dtype=np.float64)

    success_voxel = payload["voxel_index"][success_idx].numpy()
    target_pos = payload["target_pos"][success_idx].numpy()
    target_pos_sym = target_pos.copy()
    target_pos_sym[:, 1] = np.abs(target_pos_sym[:, 1])
    target_pos_rounded = np.round(target_pos_sym, 4)
    _, first_offset, inverse = np.unique(target_pos_rounded, axis=0, return_index=True, return_inverse=True)
    representative_idx = success_idx.numpy()[first_offset]
    n_voxels = len(representative_idx)
    voxel_row_count = np.bincount(inverse, minlength=n_voxels)
    voxel_visible = np.zeros(n_voxels, dtype=bool)
    voxel_head_q = np.full((n_voxels, 2), np.nan, dtype=np.float32)

    arms_down = {
        "Left_Shoulder_Roll": -np.pi / 2,
        "Right_Shoulder_Roll": np.pi / 2,
        "Left_Elbow_Yaw": 0.0,
        "Right_Elbow_Yaw": 0.0,
    }

    for number, idx in enumerate(representative_idx, start=1):
        data.qpos[:] = model.qpos0
        for jname, val in arms_down.items():
            data.qpos[model.jnt_qposadr[model.joint(jname).id]] = val
        data.qpos[:3] = target_offset
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        spec.apply_source_coupling(model, data)
        mujoco.mj_forward(model, data)
        t_base = payload["target_pos"][idx].numpy().copy()
        t_base[1] = np.abs(t_base[1])
        target_world = t_base + target_offset
        voxel_head_q[number - 1] = aim_head_at_target(model, data, target_world)
        voxel_visible[number - 1] = target_visible(model, data, target_world)
        if number % log_every == 0 or number == n_voxels:
            visible_rows = int(voxel_row_count[voxel_visible].sum())
            print(
                f"dynamic visibility {number}/{n_voxels} voxels: "
                f"visible rows={visible_rows}/{int(success_idx.numel())} "
                f"({100 * visible_rows / int(success_idx.numel()):.1f}%)",
                flush=True,
            )

    visible[success_idx] = torch.from_numpy(voxel_visible[inverse])
    mapped_head_q = voxel_head_q[inverse]
    original_y = payload["target_pos"][success_idx, 1].numpy()
    mapped_head_q[:, 0] *= np.sign(original_y)
    head_q[success_idx] = torch.from_numpy(mapped_head_q)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "robot": "booster_t1",
        "workspace": str(workspace.resolve()),
        "workspace_rows": n_rows,
        "success_rows": int(success_idx.numel()),
        "visible_left": visible.clone(),
        "visible_right": visible.clone(),
        "visible": visible,
        "head_q": head_q,
        "camera_mode": "dynamic_d455_head_aim_per_success_voxel",
        "camera_intrinsic_source": "d455_rgb_envelope_fovy65_resolution_1280x800",
        "camera_projection": "d455_rgb_pinhole_fovy65_near0p10m",
    }, out)
    print(
        f"saved {out}: {int(visible[success_idx].sum())}/{int(success_idx.numel())} "
        "successful arm-IK rows visible by D455 head cam"
    )


__all__ = [
    "aim_head_at_target",
    "eye_in_fov",
    "eye_in_fov_batch",
    "eye_ray_clear",
    "load_model",
    "score_workspace_dynamic",
    "target_visible",
    "visible_eyes",
]
