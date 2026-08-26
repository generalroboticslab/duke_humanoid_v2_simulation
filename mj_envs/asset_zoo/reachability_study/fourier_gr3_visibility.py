"""Dynamic head-camera visibility for the Fourier GR-3.

GR-3 carries a 2-DOF actuated head (``head_yaw_joint`` about +Z, +-80 deg;
``head_pitch_joint`` about +Y, -30..+8.5 deg) with a single head camera, so visibility is a
DYNAMIC question: a target counts as visible if SOME legal head state brings it inside the
camera envelope with an unobstructed line of sight. The policy mirrors Booster T1's --
FOV cone AND an explicit raycast against the robot's own group-2 visual meshes -- because a
pure-FOV test silently counts targets the robot's own chest is blocking.

CAMERA. Two cameras are declared in the source MJCF and either may be scored, because the
vendor's own documents disagree about the mount pitch by 25 deg (``head_cam`` = the URDF's
15 deg nose-down, ``head_cam_fovfig`` = the 40 deg the official FOV figure draws). See
``asset/fourier_gr3/gr3_import.py`` for the evidence; results are reported as a sensitivity
pair, never as a calibrated number. Both share the vendor's published 128 deg x 80 deg
envelope, encoded as ``fovy`` + resolution and read back by the shared intrinsics helper.

HEAD AIM. Closed-form, exact, one ``mj_forward`` per target -- no iterative IK. T1's
closed form does not transfer: it assumes the lens sits straight up the pitch-rotated axis,
whereas GR-3's lens is offset both forward (90.3 mm) and up (51.2 mm) from the pitch pivot,
and its optical axis is itself pitched 15/40 deg inside that link. The derivation for the
general case is in :func:`aim_head_at_target`.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MJCF = _REPO_ROOT / "asset" / "fourier_gr3" / "gr3.xml"
_HEAD_YAW_JOINT = "head_yaw_joint"
_HEAD_PITCH_JOINT = "head_pitch_joint"
DEFAULT_CAMERA = "head_cam"
SENSITIVITY_CAMERA = "head_cam_fovfig"
NEAR_CLIP_M = 0.10  # shared workspace near plane (Euclidean lens-to-target), SOP section 3


def load_model() -> mujoco.MjModel:
    """Compile the source MJCF carrying both head cameras and their raycast sites."""
    return mujoco.MjModel.from_xml_path(str(_MJCF))


def camera_intrinsics(model: mujoco.MjModel, camera_name: str):
    """(fx, fy, cx, cy, width, height) for the declared envelope.

    Isotropic square-pixel pinhole: ``fy`` follows from the declared vertical ``fovy`` and
    the image height, and ``fx = fy`` so the horizontal envelope is DERIVED from the aspect
    ratio (128.01 deg for GR-3's 1955x800). Using ``fy * width / height`` instead would force
    fovx == fovy and silently throw away 48 deg of the vendor's published horizontal field.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    assert cam_id >= 0, f"camera {camera_name!r} absent from source MJCF"
    width, height = (int(v) for v in model.cam_resolution[cam_id])
    fovy = float(model.cam_fovy[cam_id])
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    return fy, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0, width, height


def _head_geometry(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str):
    """Constants for the closed-form aim, measured off the model at the standing pose.

    Returns the yaw and pitch pivots in the ``base_link`` frame, the lens offset from the
    pitch pivot, and the optical axis, both expressed in the pitch link's zero-pitch frame.
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _HEAD_YAW_JOINT)
    pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _HEAD_PITCH_JOINT)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    base = data.xpos[base_id]
    return {
        "yaw_pivot": data.xanchor[yaw_id] - base,
        "pitch_pivot": data.xanchor[pitch_id] - base,
        "lens_offset": data.cam_xpos[cam_id] - data.xanchor[pitch_id],
        "optical_axis": -data.cam_xmat[cam_id].reshape(3, 3)[:, 2],
        "yaw_limits": model.jnt_range[yaw_id].copy(),
        "pitch_limits": model.jnt_range[pitch_id].copy(),
    }


def aim_head_at_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    camera_name: str = DEFAULT_CAMERA,
) -> tuple[float, float]:
    """Point the head camera at ``target_world``; return the (yaw, pitch) it commanded.

    Closed form, exact, and applied with a single ``mj_forward``.

    YAW is exact because the lens lies ON the yaw axis' sagittal plane (its y offset is 0):
    rotating by ``atan2(dy, dx)`` measured from the yaw pivot therefore brings the target
    into the plane that contains both the pitch pivot and the lens.

    PITCH then solves a planar problem. Write everything in that sagittal plane relative to
    the pitch pivot: the lens sits at ``R(phi) r`` and looks along ``R(phi) a``, where r is
    the lens offset, a the optical axis, and R a rotation about +Y. Requiring the look
    direction to be parallel to (target - lens) is

        cross(R a, u - R r) = 0,   u = target - pitch_pivot

    and because a planar rotation preserves the scalar cross product, ``cross(R a, R r)``
    collapses to the constant ``cross(a, r)``. What remains is

        u_z cos(alpha) + u_x sin(alpha) = cross(a, r),   alpha = phi + theta0

    a single sinusoid in alpha, solved by the standard amplitude-phase reduction. Of the two
    roots the one aiming TOWARDS the target is kept. When |cross(a, r)| exceeds the amplitude
    the target lies inside the circle the lens sweeps and no exact aim exists; the closest
    attainable alpha is used. Joint limits are applied last, so an out-of-reach target simply
    ends up outside the FOV rather than being silently marked visible.
    """
    geom = _head_geometry(model, data, camera_name)
    yaw_adr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _HEAD_YAW_JOINT)])
    pitch_adr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _HEAD_PITCH_JOINT)])

    base = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")]
    target = np.asarray(target_world, dtype=np.float64) - base

    d = target - geom["yaw_pivot"]
    yaw = float(np.clip(np.arctan2(d[1], d[0]), *geom["yaw_limits"]))

    # Sagittal coordinates after the yaw rotation, relative to the pitch pivot.
    x_sag = d[0] * np.cos(yaw) + d[1] * np.sin(yaw)
    offset = geom["pitch_pivot"] - geom["yaw_pivot"]
    u = np.array([x_sag - offset[0], target[2] - geom["yaw_pivot"][2] - offset[2]])

    r = geom["lens_offset"]
    a = geom["optical_axis"]
    r2 = np.array([r[0], r[2]])
    a2 = np.array([a[0], a[2]])
    theta0 = np.arctan2(-a2[1], a2[0])          # optical axis' own nose-down angle
    k = a2[0] * r2[1] - a2[1] * r2[0]           # cross(a, r), invariant under the rotation
    amplitude = float(np.hypot(u[0], u[1]))
    delta = np.arctan2(u[0], u[1])              # phase of (u_z cos + u_x sin)

    def alignment(alpha: float) -> float:
        """cos-like score: how well the lens looks at the target at this alpha."""
        look = np.array([np.cos(alpha), -np.sin(alpha)])
        rot = np.array([[np.cos(alpha - theta0), np.sin(alpha - theta0)],
                        [-np.sin(alpha - theta0), np.cos(alpha - theta0)]])
        to_target = u - rot @ r2
        norm = float(np.linalg.norm(to_target))
        return float(look @ to_target) / norm if norm > 1e-12 else -1.0

    if amplitude > 1e-9 and abs(k) <= amplitude:
        # Regular case: two exact roots; keep the one that looks TOWARDS the target.
        half = np.arccos(k / amplitude)
        alpha = max((delta + half, delta - half), key=alignment)
    else:
        # Degenerate: the target lies inside the circle the lens sweeps about the pitch
        # pivot (|u| < |cross(a, r)|), so no orientation can put it on the optical axis --
        # it is nearer the pivot than the 103.8 mm lens offset. Fall back to the
        # best-aligned alpha on a fine sweep. Such targets sit well inside the 0.10 m near
        # clip and are rejected by `eye_in_fov` regardless; this branch only keeps the
        # returned head state sane instead of letting it face backwards.
        sweep = np.linspace(-np.pi, np.pi, 721)
        alpha = float(max(sweep, key=alignment))
    pitch = float(np.clip(alpha - theta0, *geom["pitch_limits"]))

    data.qpos[yaw_adr] = yaw
    data.qpos[pitch_adr] = pitch
    mujoco.mj_forward(model, data)
    return yaw, pitch


def eye_in_fov(model, data, target_world, camera_name: str = DEFAULT_CAMERA) -> bool:
    """Whether the target projects inside the camera envelope and past the near clip."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    rot = data.cam_xmat[cam_id].reshape(3, 3)
    delta = np.asarray(target_world, dtype=np.float64) - data.cam_xpos[cam_id]
    local = rot.T @ delta
    if local[2] >= 0.0:
        return False                      # behind the lens (MuJoCo cameras look along -Z)
    if np.linalg.norm(delta) < NEAR_CLIP_M:
        return False
    fx, fy, cx, cy, width, height = camera_intrinsics(model, camera_name)
    # The mount quat puts camera-local X along the world vertical and Y along the horizontal
    # (Ry(-90) maps local X -> +Z_link, local Y -> +Y_link), so fx pairs with local[1] and fy
    # with local[0] -- the same pairing Booster T1 uses for its identically-mounted camera.
    pixel_x = fx * local[1] / (-local[2]) + cx
    pixel_y = fy * local[0] / (-local[2]) + cy
    return 0.0 <= pixel_x <= width - 1.0 and 0.0 <= pixel_y <= height - 1.0


def eye_ray_clear(model, data, target_world, camera_name: str = DEFAULT_CAMERA) -> bool:
    """Whether no group-2 visual mesh blocks the lens-to-target segment.

    The lens' own carrying body is excluded so the camera housing cannot occlude itself.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{camera_name}_site")
    assert site_id >= 0, f"raycast site for {camera_name!r} absent from source MJCF"
    origin = data.site_xpos[site_id]
    direction = np.asarray(target_world, dtype=np.float64) - origin
    distance = float(np.linalg.norm(direction))
    if distance < 1e-9:
        return True
    hit_geom = np.full(1, -1, dtype=np.int32)
    geomgroup = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)   # group-2 visual meshes only
    hit_distance = mujoco.mj_ray(
        model, data, origin, direction / distance, geomgroup, True,
        int(model.site_bodyid[site_id]), hit_geom,
    )
    return hit_distance < 0.0 or hit_distance >= distance - 1e-3


def target_visible(model, data, target_world, camera_name: str = DEFAULT_CAMERA) -> bool:
    """FOV cone AND line of sight, at whatever head state the caller has already aimed."""
    return (eye_in_fov(model, data, target_world, camera_name)
            and eye_ray_clear(model, data, target_world, camera_name))


def eye_in_fov_batch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    camera_name: str = DEFAULT_CAMERA,
) -> np.ndarray:
    """Vectorized FOV test (no raycast): return `(N,)` bool mask, mirrors `eye_in_fov`.

    FOV-only, no occlusion -- matches the live-viewer tradeoff Booster T1/ToddlerBot already use
    (a BVH raycast per slider change is too costly for interactive use). The offline scorer
    (`target_visible`) remains occlusion-aware. Reads `data.cam_xpos/cam_xmat` directly, so it is
    correct at whatever head state the caller already commanded (e.g. live yaw/pitch sliders).
    """
    targets = np.atleast_2d(np.asarray(target_world, dtype=np.float64))
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    rot = data.cam_xmat[cam_id].reshape(3, 3)
    delta = targets - data.cam_xpos[cam_id]
    local = delta @ rot  # (N, 3); row-vector form of rot.T @ delta_i per target
    in_front = local[:, 2] < 0.0
    dist = np.linalg.norm(delta, axis=1)
    fx, fy, cx, cy, width, height = camera_intrinsics(model, camera_name)
    z = np.where(in_front, -local[:, 2], 1.0)
    # Same local-axis pairing as `eye_in_fov`: fx with local[1], fy with local[0].
    pixel_x = fx * local[:, 1] / z + cx
    pixel_y = fy * local[:, 0] / z + cy
    return (
        in_front
        & (dist >= NEAR_CLIP_M)
        & (pixel_x >= 0.0) & (pixel_x <= width - 1.0)
        & (pixel_y >= 0.0) & (pixel_y <= height - 1.0)
    )


__all__ = [
    "DEFAULT_CAMERA",
    "NEAR_CLIP_M",
    "SENSITIVITY_CAMERA",
    "aim_head_at_target",
    "camera_intrinsics",
    "eye_in_fov",
    "eye_in_fov_batch",
    "eye_ray_clear",
    "load_model",
    "target_visible",
]
