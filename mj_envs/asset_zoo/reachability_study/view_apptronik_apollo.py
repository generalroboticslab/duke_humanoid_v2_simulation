"""Live Apollo source-model viewer, beginning with collision-model approval.

Scope: source visual mesh (group 1), native source collision primitives (group 3),
camera/end-effector audit markers (group 4), and cuRobo-config collision spheres (group 5).
Do not create a second Apollo viewer.

Run live (needs a display):
    MUJOCO_GL=glfw python mj_envs/asset_zoo/reachability_study/view_apptronik_apollo.py
Headless approval strip (front | side | isometric):
    MUJOCO_GL=egl python mj_envs/asset_zoo/reachability_study/view_apptronik_apollo.py --png

Native MuJoCo keys: `1` source visual mesh, `3` native source collision, `4` camera/EE audit
markers, `5` cuRobo spheres. Diagnostic palette is runtime-only: collision primitives opaque
green, head-eye axes cyan/orange, palm-center EE sites magenta/orange, head-eye FOV frusta thin
cyan/orange wireframes, spheres translucent blue.

``_add_audit_markers`` is imported by ``plot_workspace_curobo.py``; keep it importable here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mujoco
import numpy as np

from mj_envs.asset_zoo.reachability_study import view_common as vc
from mj_envs.utils.mj_collision_spheres import CapsuleSphereFit, build_collision_spheres


_XML_PATH = _REPO / "asset" / "apptronik_apollo" / "apptronik_apollo.xml"
_PNG_PATH = _HERE.parent / "result" / "apptronik_apollo_source_collision_model.png"
_VISUAL_GROUP = 1
_COLLISION_GROUP = 3
_AUDIT_GROUP = 4
_CUROBO_SPHERE_GROUP = 5
_CUROBO_SPHERE_RGBA = np.array([0.1, 0.55, 1.0, 0.55], dtype=np.float32)
_HEAD_CAMERA_ASPECT_RATIO = 16.0 / 9.0
_HEAD_CAMERA_FOV_RANGE_M = 0.45
_HEAD_EYE_CAMERAS = (
    ("head_cam_left", np.array([0.1, 0.85, 1.0, 0.95], dtype=np.float32)),
    ("head_cam_right", np.array([1.0, 0.55, 0.1, 0.95], dtype=np.float32)),
)
_END_EFFECTOR_SITES = (
    ("end_effector_L_site", np.array([0.95, 0.15, 0.85, 0.95], dtype=np.float32)),
    ("end_effector_R_site", np.array([1.0, 0.60, 0.10, 0.95], dtype=np.float32)),
)
_BASE_PELVIS_SPHERE_RADIUS_M = 0.08
_BASE_PELVIS_SPHERE_COUNT = 5
_CAPSULE_SPHERE_FITS = {
    "base_link": CapsuleSphereFit(radius_scale=_BASE_PELVIS_SPHERE_RADIUS_M / 0.10, sphere_count=_BASE_PELVIS_SPHERE_COUNT),
}


def _load_model() -> mujoco.MjModel:
    """Load source model and apply diagnostic-only colors for collision coverage review."""
    model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
    model.geom_rgba[np.flatnonzero(model.geom_group == _COLLISION_GROUP)] = np.array([0.1, 1.0, 0.1, 1.0])
    return model


def _collision_summary(model: mujoco.MjModel) -> str:
    """Return source group-3 primitive coverage summary, without approximating it."""
    geoms = np.flatnonzero(model.geom_group == _COLLISION_GROUP)
    type_names = {mujoco.mjtGeom.mjGEOM_CAPSULE: "capsules", mujoco.mjtGeom.mjGEOM_BOX: "boxes"}
    counts = Counter(type_names.get(model.geom_type[g], f"type-{model.geom_type[g]}") for g in geoms)
    bodies = len({int(model.geom_bodyid[g]) for g in geoms})
    count_text = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    return f"Apollo source collision: {len(geoms)} group-3 geoms on {bodies} bodies ({count_text})"


def _curobo_spheres(model: mujoco.MjModel) -> list[tuple[int, np.ndarray, float]]:
    """Derive cuRobo collision spheres from source group-3 primitives.

    Boxes use inscribed spheres so hand and sole proxies do not extend beyond their source
    collision boxes. Capsules retain source radius, except the broad pelvis uses a smaller chain
    per visual review. This source-local dictionary is the collision-sphere payload for the
    eventual cuRobo robot config.
    """
    source_local = build_collision_spheres(model, box_inscribed=True, capsule_sphere_overrides=_CAPSULE_SPHERE_FITS)
    spheres = []
    for body_name, body_spheres in source_local.items():
        body_id = model.body(body_name).id
        for sphere in body_spheres:
            spheres.append((body_id, np.asarray(sphere["center"], dtype=np.float64), float(sphere["radius"])))
    if len(spheres) > 500:
        raise RuntimeError(f"Apollo cuRobo collision sphere budget exceeded: {len(spheres)} > 500")
    return spheres


def _add_curobo_spheres(data: mujoco.MjData, scene, spheres: list[tuple[int, np.ndarray, float]]) -> None:
    """Draw source-FK transformed cuRobo collision spheres in viewer group 5.

    Spheres are welded to source bodies (no cuRobo Kinematics FK), so transform each by its
    parent body's live pose -- unlike the cfg-dict FK path in ``view_common``.
    """
    for body_id, local_center, radius in spheres:
        world_center = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local_center
        vc.add_spheres(scene, world_center[None], [radius], _CUROBO_SPHERE_RGBA[None])


def _add_audit_markers(model: mujoco.MjModel, data: mujoco.MjData, scene) -> None:
    """Draw provisional head-eye FOV frusta and selected palm-center end-effector sites.

    The source OAK-D frames remain torso mapping sensors, not Apollo's head stereo pair, so they
    are intentionally absent from this primary perception overlay.
    """
    for camera_name, rgba in _HEAD_EYE_CAMERAS:
        camera_id = model.camera(camera_name).id
        origin = data.cam_xpos[camera_id]
        camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
        forward = -camera_rotation[:, 2]
        vc.add_spheres(scene, origin[None], [0.004], rgba[None])
        vc.add_segment(scene, origin, origin + 0.18 * forward, rgba, width=0.001)
        vertical_half_width = _HEAD_CAMERA_FOV_RANGE_M * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2.0)
        horizontal_half_width = vertical_half_width * _HEAD_CAMERA_ASPECT_RATIO
        far_center = origin + _HEAD_CAMERA_FOV_RANGE_M * forward
        corners = (
            far_center - horizontal_half_width * camera_rotation[:, 0] - vertical_half_width * camera_rotation[:, 1],
            far_center + horizontal_half_width * camera_rotation[:, 0] - vertical_half_width * camera_rotation[:, 1],
            far_center + horizontal_half_width * camera_rotation[:, 0] + vertical_half_width * camera_rotation[:, 1],
            far_center - horizontal_half_width * camera_rotation[:, 0] + vertical_half_width * camera_rotation[:, 1],
        )
        for corner in corners:
            vc.add_segment(scene, origin, corner, rgba, width=0.001)
        for start, end in zip(corners, corners[1:] + corners[:1]):
            vc.add_segment(scene, start, end, rgba, width=0.001)
    for site_name, rgba in _END_EFFECTOR_SITES:
        center = data.site_xpos[model.site(site_name).id]
        vc.add_spheres(scene, center[None], [0.015], rgba[None])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", action="store_true", help="write headless approval PNG instead of opening live viewer")
    parser.add_argument("--out", type=pathlib.Path, default=_PNG_PATH, help="output path with --png")
    args = parser.parse_args()
    model = _load_model()
    data = mujoco.MjData(model)
    spheres = _curobo_spheres(model)
    mujoco.mj_forward(model, data)
    print(f"{_collision_summary(model)}; cuRobo spheres={len(spheres)}")

    draw_audit = _add_audit_markers
    draw_spheres = lambda m, d, scene: _add_curobo_spheres(d, scene, spheres)
    if args.png:
        vc.render_png_strip(
            model, data, args.out,
            views=((90.0, -6.0), (0.0, -6.0), (42.0, -18.0)),
            overlays=[(None, draw_audit), (None, draw_spheres)],
            lookat=(0.0, 0.0, 0.95), distance=2.8,
        )
        return
    print("Apollo live viewer: group 1=source visual, 3=source collision, 4=camera/EE audit, 5=cuRobo spheres.")
    vc.run_live(
        model, data,
        init_groups={_VISUAL_GROUP: 1, _COLLISION_GROUP: 0, _AUDIT_GROUP: 1, _CUROBO_SPHERE_GROUP: 0},
        overlays=[(_AUDIT_GROUP, draw_audit), (_CUROBO_SPHERE_GROUP, draw_spheres)],
    )


if __name__ == "__main__":
    main()
