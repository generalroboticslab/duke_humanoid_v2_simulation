"""Open ToddlerBot with source visual/collision, stereo FOV, and cuRobo spheres.

Viewer groups: 2=source visuals, 3=source collision, 4=stereo FOV, 5=cuRobo collision spheres.
Toggle groups with the native MuJoCo viewer digit keys. ``--workspace <sidecar.pt>`` overlays
one reachability-shaded sphere per reachable voxel.

``_add_fov``/``FOV_SEGMENT_COUNT`` are imported by ``plot_workspace_curobo.py``;
keep them importable from this module.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Live viewer needs GLFW; select before importing MuJoCo (setdefault preserves an explicit backend).
os.environ.setdefault("MUJOCO_GL", "glfw")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mujoco
import numpy as np

from mj_envs.asset_zoo.reachability_study import view_common as vc

_MJCF = _REPO_ROOT / "asset" / "toddlerbot_2xm_gripper" / "toddlerbot_2xm_gripper_pos.xml"
_EYES = (("head_cam_left", "head_cam_left_site", (0.2, 0.7, 1.0, 0.9)),
         ("head_cam_right", "head_cam_right_site", (1.0, 0.55, 0.1, 0.9)))
_FOV_NEAR_M = 0.10
_FOV_FAR_M = 0.55
# Segments `_add_fov` appends per eye: near square (4) + far square (4) + near-far rays (4) +
# optical axis (1). Callers reserve this many scene-geom slots before invoking `_add_fov`.
FOV_SEGMENT_COUNT = len(_EYES) * (4 + 4 + 4 + 1)
_CUROBO_REGIONS = ((r"left_(shoulder|elbow|wrist|gripper)", (0.95, 0.2, 0.2, 0.42)),
                   (r"right_(shoulder|elbow|wrist|gripper)", (0.2, 0.45, 1.0, 0.42)),
                   (r"base_link|torso|head|neck|waist|pelvis", (0.1, 0.9, 0.9, 0.42)),
                   (r"hip|knee|ankle", (0.7, 0.7, 0.7, 0.34)))


def _camera_rays(model: mujoco.MjModel, camera_id: int) -> list[np.ndarray]:
    """Return raw-fisheye pinhole-envelope corner rays in site (+Z forward) frame."""
    fx, fy, cx_offset, cy_offset = model.cam_intrinsic[camera_id]
    width, height = model.cam_resolution[camera_id]
    cx, cy = width / 2.0 - 0.5 - cx_offset, height / 2.0 - 0.5 - cy_offset
    rays = []
    for u, v in ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)):
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        rays.append(ray / np.linalg.norm(ray))
    return rays


def _add_fov(model: mujoco.MjModel, data: mujoco.MjData, scene) -> None:
    """Draw left/right 160-degree-diagonal fisheye-envelope wireframes in group 4."""
    for camera_name, site_name, rgba in _EYES:
        camera_id, site_id = model.camera(camera_name).id, model.site(site_name).id
        pos, rotation = data.site_xpos[site_id], data.site_xmat[site_id].reshape(3, 3)
        rays = [rotation @ ray for ray in _camera_rays(model, camera_id)]
        near = [pos + _FOV_NEAR_M * ray for ray in rays]
        far = [pos + _FOV_FAR_M * ray for ray in rays]
        for points in (near, far):
            for index in range(4):
                vc.add_segment(scene, points[index], points[(index + 1) % 4], rgba)
        for index in range(4):
            vc.add_segment(scene, pos, far[index], rgba)
        vc.add_segment(scene, pos, pos + 0.20 * rotation[:, 2], (1.0, 1.0, 0.0, 1.0), width=0.004)


def _spheres_state(model: mujoco.MjModel, data: mujoco.MjData):
    from mj_envs.tasks.visual_manipulation.curobo.toddlerbot_curobo_robot_cfg import BASE_LINK, build_robot_cfg_dict
    return vc.curobo_fk_state(model, data, build_robot_cfg_dict, BASE_LINK, _CUROBO_REGIONS)


def _load_workspace(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load reachable voxel centers and paper-consistent RdYlGn reachability colors."""
    import torch
    from matplotlib import colormaps

    payload = torch.load(path, map_location="cpu", weights_only=False)
    success = payload["success"].bool().numpy()
    points = payload["target_pos"][success].numpy()
    if "voxel_index" not in payload:
        raise ValueError(f"workspace lacks voxel_index: {path}")
    voxel_index = payload["voxel_index"][success].numpy()
    voxel_ids, first = np.unique(voxel_index, return_index=True)
    points = points[first]
    counts = np.bincount(voxel_index, minlength=int(payload["n_grid_per_axis"].prod()))
    n_orientations = int(payload["n_orientations"])
    dexterity = counts[voxel_ids].astype(np.float32) / n_orientations
    rgba = colormaps["RdYlGn"](np.clip(dexterity / 0.7, 0.0, 1.0)).astype(np.float32)
    rgba[:, 3] = 0.60
    if points.size == 0:
        raise ValueError(f"workspace has no successful targets: {path}")
    return points, dexterity, rgba


def _add_workspace(scene, points_base: np.ndarray, colors: np.ndarray, data: mujoco.MjData, root_id: int) -> None:
    """Draw source-world workspace spheres shaded by reachability index D."""
    rotation = data.xmat[root_id].reshape(3, 3)
    centers = points_base @ rotation.T + data.xpos[root_id]
    vc.add_spheres(scene, centers, np.full(len(centers), 0.018), colors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=pathlib.Path, default=None,
                        help="optional workspace sidecar; draws one sphere per reachable voxel")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    state = _spheres_state(model, data)
    root_id = state[-1]
    workspace = None if args.workspace is None else _load_workspace(args.workspace)
    mujoco.mj_forward(model, data)
    print("ToddlerBot viewer; group 2=visual, 3=MuJoCo collision, 4=stereo FOV, 5=cuRobo spheres")
    if workspace is not None:
        print(f"workspace reachable voxels ({len(workspace[0])}), "
              f"D=[{workspace[1].min():.3f}, {workspace[1].max():.3f}] RdYlGn [0, 0.7]")

    overlays = [(4, _add_fov), (5, lambda m, d, s: vc.add_curobo_fk_spheres(d, s, *state))]
    if workspace is not None:
        overlays.append((None, lambda m, d, s: _add_workspace(s, workspace[0], workspace[2], d, root_id)))
    vc.run_live(model, data, init_groups={2: 1, 3: 1, 4: 1, 5: 1}, overlays=overlays)


if __name__ == "__main__":
    main()
