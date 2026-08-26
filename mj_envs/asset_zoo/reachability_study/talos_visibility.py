"""Dynamic head-camera visibility for the PAL Robotics TALOS.

TALOS carries a 2-DOF actuated head (``head_1_joint`` = TILT about +Y, -12..+45 deg;
``head_2_joint`` = PAN about +Z, +-75 deg) with a head-mounted RGB-D camera, so visibility is
a DYNAMIC question: a target counts as visible if SOME legal head state brings it inside the
camera envelope with an unobstructed line of sight. The policy mirrors Booster T1's and
GR-3's -- FOV cone AND an explicit raycast against the robot's own group-2 visual meshes --
because a pure-FOV test silently counts targets the robot's own chest is blocking.

CAMERA. The sensor is an **Orbbec Astra Pro**, confirmed from two independent primary
sources: PAL's own ``talos_description/urdf/head/head.urdf.xacro`` instantiates the
``orbbec_astra_pro`` macro on ``head_2_link``, and Stasse et al., "TALOS: A new humanoid
research platform targeted for industrial applications" (Humanoids 2017) names the same part.
Orbbec's published RGB envelope is 63.1 deg x 49.4 deg at 640x480, which the source MJCF
encodes as ``head_cam``, matching how every other robot in this study is scored on its
primary RGB stream. The mount pose is PAL's own ``rgbd_link`` offset, not an estimate.

HEAD AIM. Closed-form, exact, one ``mj_forward`` per target -- no iterative IK. GR-3's closed
form does not transfer: GR-3's head is yaw-outer/pitch-inner, TALOS's is the opposite
(tilt-outer, pan-inner), which breaks GR-3's "yaw is exact on its own" step. TALOS instead
admits a cleaner exact solution, because its two head axes INTERSECT (verified on the model:
the tilt and pan anchors coincide to 0.0 m at z=1.47025), making the head a true gimbal about
a single point. The derivation is in :func:`aim_head_at_target`.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MJCF = _REPO_ROOT / "asset" / "pal_talos" / "talos_study.xml"
_HEAD_TILT_JOINT = "head_1_joint"   # outer, +Y
_HEAD_PAN_JOINT = "head_2_joint"    # inner, +Z
_CAMERA_BODY = "head_2_link"
DEFAULT_CAMERA = "head_cam"             # Orbbec Astra Pro RGB stream (primary result)
NEAR_CLIP_M = 0.10  # shared workspace near plane (Euclidean lens-to-target), SOP section 3


def load_model() -> mujoco.MjModel:
    """Compile the study MJCF carrying the head camera and its raycast site."""
    return mujoco.MjModel.from_xml_path(str(_MJCF))


def camera_intrinsics(model: mujoco.MjModel, camera_name: str):
    """(fx, fy, cx, cy, width, height) for the declared envelope.

    Isotropic square-pixel pinhole: ``fy`` follows from the declared vertical ``fovy`` and the
    image height, and ``fx = fy`` so the horizontal envelope is DERIVED from the 4:3 aspect
    ratio. That reproduces Orbbec's published horizontal RGB figure to within 0.1 deg (63.04
    vs 63.1), which is the consistency check that the vendor's published angle really does
    describe a square-pixel pinhole.
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    assert cam_id >= 0, f"camera {camera_name!r} absent from source MJCF"
    width, height = (int(v) for v in model.cam_resolution[cam_id])
    fovy = float(model.cam_fovy[cam_id])
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    return fy, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0, width, height


def _head_geometry(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str):
    """Constants for the closed-form aim, measured off the model at zero head joints.

    Returns the shared gimbal pivot and the frame it rotates, plus the lens offset and optical
    axis expressed in that zero frame. Reading them from the live model rather than hardcoding
    them means a change to the mount in the import script cannot silently desync this scorer.
    """
    tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _HEAD_TILT_JOINT)
    pan_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _HEAD_PAN_JOINT)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _CAMERA_BODY)
    tilt_adr = int(model.jnt_qposadr[tilt_id])
    pan_adr = int(model.jnt_qposadr[pan_id])

    saved = data.qpos.copy()
    data.qpos[tilt_adr] = 0.0
    data.qpos[pan_adr] = 0.0
    mujoco.mj_forward(model, data)
    pivot = data.xanchor[tilt_id].copy()
    separation = float(np.linalg.norm(data.xanchor[pan_id] - pivot))
    assert separation < 1e-9, (
        f"TALOS head axes no longer intersect (separation {separation:.6g} m); the closed-form "
        "gimbal aim in aim_head_at_target is not valid for an offset pair"
    )
    frame = data.xmat[body_id].reshape(3, 3).copy()
    lens_offset = frame.T @ (data.cam_xpos[cam_id] - pivot)
    optical_axis = frame.T @ (-data.cam_xmat[cam_id].reshape(3, 3)[:, 2])
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return {
        "pivot": pivot,
        "frame": frame,
        "lens_offset": lens_offset,
        "optical_axis": optical_axis / np.linalg.norm(optical_axis),
        "tilt_limits": model.jnt_range[tilt_id].copy(),
        "pan_limits": model.jnt_range[pan_id].copy(),
        "tilt_adr": tilt_adr,
        "pan_adr": pan_adr,
    }


def aim_head_at_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_world: np.ndarray,
    camera_name: str = DEFAULT_CAMERA,
) -> tuple[float, float]:
    """Point the head camera at ``target_world``; return the (tilt, pan) it commanded.

    Closed form, exact, and applied with a single ``mj_forward``.

    Because both head axes pass through one point, the head applies a pure rotation
    ``R = Ry(tilt) Rz(pan)`` about the pivot. Let ``u`` be the target relative to the pivot in
    the zero-head frame, ``r`` the lens offset and ``a`` the unit optical axis, both constant
    in the camera link's own frame. Perfect aim means the target lies on the lens ray IN THAT
    LINK FRAME, i.e. ``R^T u = r + t a`` for some ``t > 0``: the head-relative target sits a
    distance ``t`` down the optical axis from the lens.

    STEP 1 -- solve ``t`` alone. A rotation preserves length, so ``|r + t a| = |u|``, giving

        t^2 + 2 (r.a) t - (|u|^2 - |r|^2) = 0,  t = -(r.a) + sqrt((r.a)^2 + |u|^2 - |r|^2)

    The discriminant is ``|u|^2`` minus the squared pivot-to-ray distance (0.187 m here), so
    it fails only for targets nearer the pivot than the lens ray ever passes -- all of which
    sit far inside the 0.10 m near clip and are rejected by :func:`eye_in_fov` anyway.

    STEP 2 -- ``s = r + t a`` is now a KNOWN vector, and ``R^T u = s`` is an ordinary
    two-angle aiming problem. ``Rz(-pan)`` preserves the z component, so tilt alone must carry
    ``u`` to the right height:

        u_x sin(tilt) + u_z cos(tilt) = s_z

    one sinusoid in tilt, solved by amplitude-phase reduction; of the two roots the one giving
    the better alignment is kept. Pan is then read straight off the residual azimuth. Joint
    limits are applied last, so an out-of-reach target simply ends up outside the FOV rather
    than being silently marked visible.
    """
    geom = _head_geometry(model, data, camera_name)
    u = geom["frame"].T @ (np.asarray(target_world, dtype=np.float64) - geom["pivot"])
    r = geom["lens_offset"]
    a = geom["optical_axis"]

    def commanded(tilt: float, pan: float) -> tuple[float, float]:
        tilt = float(np.clip(tilt, *geom["tilt_limits"]))
        pan = float(np.clip(pan, *geom["pan_limits"]))
        data.qpos[geom["tilt_adr"]] = tilt
        data.qpos[geom["pan_adr"]] = pan
        mujoco.mj_forward(model, data)
        return tilt, pan

    r_dot_a = float(r @ a)
    discriminant = r_dot_a**2 + float(u @ u) - float(r @ r)
    amplitude = float(np.hypot(u[0], u[2]))
    if discriminant < 0.0 or amplitude < 1e-12:
        # Target inside the sphere the lens ray never enters, or straight along the tilt axis:
        # no exact aim exists. Both cases sit inside the near clip; hold the head level so the
        # returned state stays sane rather than snapping to an arbitrary extreme.
        return commanded(0.0, 0.0)

    s = r + (-r_dot_a + np.sqrt(discriminant)) * a
    sin_shift = float(s[2]) / amplitude
    if abs(sin_shift) > 1.0:
        # |s_z| exceeds what tilt can reach because the target is far off the sagittal plane;
        # take the closest attainable height and let pan do the rest.
        sin_shift = float(np.clip(sin_shift, -1.0, 1.0))
    phase = np.arctan2(u[2], u[0])
    base = np.arcsin(sin_shift) - phase

    def solution(tilt: float) -> tuple[float, float, float]:
        """(alignment, tilt, pan) for one tilt root; alignment is cos of the aim error.

        Both roots of the tilt equation can score a perfect 1.0 in this UNCONSTRAINED math (the
        classic two-fold gimbal ambiguity: e.g. tilt~166 deg / pan~178 deg reproduces the same
        optical-axis direction as tilt~39 deg / pan~-6 deg). TALOS's tilt range is only
        -12..+45 deg, so the "wrap-around" root is always physically unreachable and gets
        clamped into a garbage pose -- but comparing UNCLAMPED scores lets it tie (or win by
        rounding noise) against the reachable root, which flips visible/blind at nearby targets
        with no physical cause (verified: adjacent voxels 3 cm apart flip solely from which root
        won the tie). Scoring the CLAMPED angles instead means an unreachable root is scored by
        what it actually achieves once limited, so it can never out-rank a root that needs no
        clamping at all.
        """
        clamped_tilt = float(np.clip(tilt, *geom["tilt_limits"]))
        cos_t, sin_t = np.cos(clamped_tilt), np.sin(clamped_tilt)
        w = np.array([u[0] * cos_t - u[2] * sin_t, u[1], u[0] * sin_t + u[2] * cos_t])
        pan = float(np.arctan2(w[1], w[0]) - np.arctan2(s[1], s[0]))
        pan = float((pan + np.pi) % (2.0 * np.pi) - np.pi)
        clamped_pan = float(np.clip(pan, *geom["pan_limits"]))
        rot = _rotation(clamped_tilt, clamped_pan)
        to_target = u - rot @ r
        norm = float(np.linalg.norm(to_target))
        score = float((rot @ a) @ to_target) / norm if norm > 1e-12 else -1.0
        return score, clamped_tilt, clamped_pan

    best = max((solution(base), solution(np.pi - np.arcsin(sin_shift) - phase)),
               key=lambda item: item[0])
    return commanded(best[1], best[2])


def _rotation(tilt: float, pan: float) -> np.ndarray:
    """``Ry(tilt) Rz(pan)`` -- the head's outer-tilt, inner-pan composition."""
    ct, st = np.cos(tilt), np.sin(tilt)
    cp, sp = np.cos(pan), np.sin(pan)
    return np.array([
        [ct * cp, -ct * sp, st],
        [sp, cp, 0.0],
        [-st * cp, st * sp, ct],
    ])


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
    # The mount's xyaxes put camera-local X along the robot's right and local Y along its up,
    # the conventional MuJoCo orientation, so local[0] pairs with fx and local[1] with fy.
    pixel_x = fx * local[0] / (-local[2]) + cx
    pixel_y = fy * local[1] / (-local[2]) + cy
    return 0.0 <= pixel_x <= width - 1.0 and 0.0 <= pixel_y <= height - 1.0


def eye_ray_clear(model, data, target_world, camera_name: str = DEFAULT_CAMERA) -> bool:
    """Whether no group-2 visual mesh blocks the lens-to-target segment.

    Excludes the WHOLE head/neck chain (``head_1_link`` + ``head_2_link``), not just the lens'
    own carrying body: with only a head camera, no part of the robot's own head assembly can
    ever be a real obstacle to its own lens. ``mujoco.mj_ray`` only accepts ONE excluded body per
    call, so a self-hit on the OTHER chain body (e.g. a ray that grazes the tilt-joint housing on
    ``head_1_link`` while excluding only ``head_2_link``) is walked past -- advance the origin
    just beyond the hit and re-cast -- rather than treated as blocking. Before this, such grazes
    near the tilt limit produced isolated, physically-baseless blind flips between
    voxels millimetres apart (visible `--reach-visible-compare` artifact).
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{camera_name}_site")
    assert site_id >= 0, f"raycast site for {camera_name!r} absent from source MJCF"
    head_bodies = (
        int(model.body_weldid[model.joint(_HEAD_TILT_JOINT).bodyid.item()]),
        int(model.body_weldid[model.joint(_HEAD_PAN_JOINT).bodyid.item()]),
        int(model.body_weldid[model.site_bodyid[site_id]]),
    )
    head_bodies = tuple(dict.fromkeys(head_bodies))  # dedupe, keep order
    origin = data.site_xpos[site_id].copy()
    direction = np.asarray(target_world, dtype=np.float64) - origin
    remaining = float(np.linalg.norm(direction))
    if remaining < 1e-9:
        return True
    unit = direction / remaining
    geomgroup = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)   # group-2 visual meshes only
    hit_geom = np.full(1, -1, dtype=np.int32)
    for exclude_body in head_bodies:
        hit_distance = mujoco.mj_ray(model, data, origin, unit, geomgroup, True, exclude_body, hit_geom)
        if hit_distance < 0.0 or hit_distance >= remaining - 1e-3:
            return True
        hit_body = int(model.body_weldid[model.geom_bodyid[int(hit_geom[0])]])
        if hit_body not in head_bodies:
            return False  # genuine block
        origin = origin + unit * (hit_distance + 1e-3)
        remaining -= hit_distance + 1e-3
    return False  # exhausted the chain depth and still blocked


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
    correct at whatever head state the caller already commanded (e.g. live tilt/pan sliders).
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
    # Same local-axis pairing as `eye_in_fov`: fx with local[0], fy with local[1].
    pixel_x = fx * local[:, 0] / z + cx
    pixel_y = fy * local[:, 1] / z + cy
    return (
        in_front
        & (dist >= NEAR_CLIP_M)
        & (pixel_x >= 0.0) & (pixel_x <= width - 1.0)
        & (pixel_y >= 0.0) & (pixel_y <= height - 1.0)
    )


__all__ = [
    "DEFAULT_CAMERA",
    "NEAR_CLIP_M",
    "aim_head_at_target",
    "camera_intrinsics",
    "eye_in_fov",
    "eye_in_fov_batch",
    "eye_ray_clear",
    "load_model",
    "target_visible",
]
