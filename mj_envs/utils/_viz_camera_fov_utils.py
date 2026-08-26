"""Shared helpers for the FOV/vision viz scripts (plan A0).

Used by:
  - mj_envs/probe/viz_camera_fov.py        (static probe grid + panels)
  - mj_envs/probe/viz_camera_fov_video.py  (animated orbit mp4)

Renders FOV cones / probe spheres / optic-axis overlay into a MuJoCo scene.
The geometry math assumes CAMERA +Z OUT (the convention `camera_perception.py`
follows; MuJoCo `<camera>` itself looks down -Z — the v83 world sites already
align the camera frame so +Z is forward).

Reused externally by `tasks/visual_manipulation/pickplace_reach_env.py`, which
imports `_draw_frustum` from this module to overlay cones on rollouts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import mujoco

_MJ_ENVS = os.path.dirname(os.path.abspath(__file__))
if _MJ_ENVS not in sys.path:
    sys.path.insert(0, _MJ_ENVS)

from tasks.camera_perception import fov_detect  # noqa: F401  (re-exported for callers)
from tasks.camera_terms import H_HALF, V_HALF, NEAR, FAR, _CAM_SITES  # noqa: F401
from asset_zoo.fov_frustum import frustum_corner_dirs

# Namespaced for the merged MjModel that `env.sim.mj_model` exposes.
SITES = tuple(f"robot/{n}" for n in _CAM_SITES)
TASK = "HumanoidRmaVelEstArmFlashSacv83L2TActuatedCam"

# MuJoCo site optical axis = +Z column of `site_xmat`. v83 sites are built so +Z
# points OUT of the head (i.e. the camera's forward direction). If a future asset
# flips this, `fov_detect` will disagree with the rendered cone and the script's
# green/red classification will mismatch the visible pixels.
DEFAULT_OUT_DIR = os.path.join(_MJ_ENVS, "..", "runs", "_viz_camera")


def _add_sphere(scn, pos, rgba, size=0.045):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, size, size]), np.asarray(pos, np.float64),
        np.eye(3).flatten(), np.asarray(rgba, np.float32),
    )
    scn.ngeom += 1


def _add_segment(scn, a, b, rgba, width=0.006):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
        np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(a, np.float64), np.asarray(b, np.float64))
    scn.ngeom += 1


def _add_cylinder(scn, a, b, rgba, radius=0.003):
    """Fourier GR-3 path: explicit cylinder (vs the capsule used for D436/v83)."""
    if scn.ngeom >= scn.maxgeom:
        return
    diff = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    length = float(np.linalg.norm(diff))
    if length < 1e-9:
        return
    z_axis = np.array([0.0, 0.0, 1.0])
    diff_n = diff / length
    cross = np.cross(z_axis, diff_n)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-9:
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if diff_n[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        angle = np.arctan2(cross_norm, float(np.dot(z_axis, diff_n)))
        axis = cross / cross_norm
        quat = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat)
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, length / 2.0, 0.0]),
        ((np.asarray(a, np.float64) + np.asarray(b, np.float64)) / 2.0),
        mat, np.asarray(rgba, np.float32),
    )
    scn.ngeom += 1


def _draw_frustum(
    scn, site_pos, R_wc, rgba, *,
    draw_near: bool = False,                       # was True — near-rect clips the head mesh
    axis: bool = False,                            # was True — opaque yellow pole pokes through cubes
    axis_length_m: float = 0.6,
    axis_width_m: float = 0.012,
    axis_rgba: tuple = (1.0, 1.0, 0.0, 1.0),
    basis: str = "capsule",                        # "capsule" | "cyl" — fourier drawer merge
):
    """FOV cone: 4 edge rays + (optional) near/far rects + (optional) +Z axis.

    R_wc: world-from-camera rotation (3x3). Corners sit at EUCLIDEAN lens range
    NEAR/FAR along the unit corner directions -- `fov_detect` gates on
    ``rng = ||p_cam||``, not on depth z. This previously placed them at DEPTH
    NEAR/FAR (``(sx*d*tan H, sy*d*tan V, d)``), over-drawing the far corners to
    ~8.1 m of true range against a 5.0 m gate, so the cone claimed detections the
    detector rejects. Corner directions come from the same helper the baked
    model-geom overlay uses, so the two cannot drift apart again.

    Defaults are cone-only — most call sites (chase-cam mp4, orbit mp4, pickplace
    playback) want the geometric envelope without the yellow pole or the near-rect
    that clips the head mesh. Static probe (`viz_camera_fov.py`) opts back into
    both via explicit kwargs.

    `basis="cyl"` draws edges as mjGEOM_CYLINDER (Fourier GR-3 path); default
    `capsule` uses mjv_connector (D436 / v83 path). Both share the corner math.
    """
    corner_dirs = frustum_corner_dirs(H_HALF, V_HALF)
    def corners(r):
        return [u * r for u in corner_dirs]
    far_w = [site_pos + R_wc @ c for c in corners(FAR)]
    if basis == "capsule":
        for i in range(4):
            _add_segment(scn, site_pos, far_w[i], rgba)              # edge rays
            _add_segment(scn, far_w[i], far_w[(i + 1) % 4], rgba)    # far rect
        if draw_near:
            near_w = [site_pos + R_wc @ c for c in corners(NEAR)]
            for i in range(4):
                _add_segment(scn, near_w[i], near_w[(i + 1) % 4], rgba)  # near rect
        if axis:
            _add_segment(scn, site_pos, site_pos + R_wc[:, 2] * axis_length_m,
                         axis_rgba, width=axis_width_m)
    elif basis == "cyl":
        # Fourier GR-3 path: explicit near-rect + far-rect + 4 connecting rays as
        # mjGEOM_CYLINDER (the fourier code draws the 12 edges — kept identical
        # to its pre-merge behavior so the reachability figure does not change).
        near_w = [site_pos + R_wc @ c for c in corners(NEAR)]
        edges = []
        if draw_near:
            edges += [(near_w[i], near_w[(i + 1) % 4]) for i in range(4)]
        edges += [(far_w[i], far_w[(i + 1) % 4]) for i in range(4)]
        edges += [(near_w[i], far_w[i]) for i in range(4)]
        for a, b in edges:
            _add_cylinder(scn, a, b, rgba, radius=0.003)
        if axis:
            _add_segment(scn, site_pos, site_pos + R_wc[:, 2] * axis_length_m,
                         axis_rgba, width=axis_width_m)
    else:
        raise ValueError(f"_draw_frustum basis={basis!r}; expected 'capsule' or 'cyl'")


def _eye_camera(cp, R):
    """Free MjCamera at the site, looking 1 m down the camera's +Z axis."""
    axis = R[:, 2]
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = cp + axis
    c.distance = 1.0
    c.azimuth = float(np.degrees(np.arctan2(axis[1], axis[0])))
    c.elevation = float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1))))
    return c


def get_camera_site_poses(m, d):
    """World-frame position+rotation for each RGB camera site (uses merged model)."""
    poses = {}
    for name in SITES:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid >= 0, f"site {name!r} not found in merged MjModel"
        poses[name] = (d.site_xpos[sid].copy(),
                       d.site_xmat[sid].reshape(3, 3).copy())
    return poses


def get_head_pose(m, d):
    """Position of cam_left_rgb site (head reference) and base z height."""
    left_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITES[0])
    cp = d.site_xpos[left_id].copy()
    base_z = float(d.qpos[2])
    return cp, base_z
