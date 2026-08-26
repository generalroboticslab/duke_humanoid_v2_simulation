"""Shared MuJoCo viewer scaffolding for the reachability-study per-robot viewers.

Owns rendering/overlay/loop/PNG plumbing only, never sphere derivation. The robots
disagree on how their cuRobo collision spheres are produced (Kinematics-FK from a cfg
dict; ``CuroboArmPlanner.get_robot_as_spheres``; source-primitive fitting), so that math
stays in each ``view_<robot>.py``; here we only draw already-computed world-frame geometry
and run the viewer/offscreen loops. See readme "Adding a new robot" step 6.

Overlay convention: an overlay is ``(group, fn)`` where ``fn(model, data, scene)`` draws
into a MuJoCo scene each frame. It runs iff ``group is None`` (always) or that viewer geom
group is toggled on. Group-5 spheres, group-4 FOV, etc. are gated this way so the native
digit keys hide/show them.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time
from typing import Callable, Iterable, Sequence

import mujoco
import numpy as np
from mujoco import viewer

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

Overlay = tuple[int | None, Callable[[mujoco.MjModel, mujoco.MjData, object], None]]


def add_segment(scene, start, end, rgba, width: float = 0.002) -> None:
    """Append one visual-only wireframe edge (capsule connector) to a MuJoCo scene."""
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo scene lacks capacity for another segment")
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64))
    scene.ngeom += 1


def add_spheres(scene, centers, radii, colors) -> None:
    """Draw precomputed world-frame spheres. ``colors`` is (N,4) or one rgba to broadcast."""
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim == 1:
        colors = np.broadcast_to(colors, (len(centers), 4))
    for center, radius, rgba in zip(centers, radii, colors):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError(f"MuJoCo scene lacks capacity for {len(centers)} spheres")
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0.0, 0.0]), center, np.eye(3).ravel(),
                            np.ascontiguousarray(rgba, dtype=np.float32))
        scene.ngeom += 1


def add_ee_sites(model, data, scene, sites: Iterable[tuple[str, Sequence[float]]], radius: float = 0.02) -> None:
    """Draw fixed tool-frame sites as solid spheres (raw MuJoCo sites render at zero size)."""
    for site_name, rgba in sites:
        center = data.site_xpos[model.site(site_name).id]
        add_spheres(scene, center[None], [radius], np.asarray(rgba, dtype=np.float32)[None])


def run_live(model, data, init_groups: dict[int, int], overlays: Iterable[Overlay],
             camera: dict | None = None, frame: int | None = None, fps: int = 50) -> None:
    """Open a passive viewer; redraw group-gated overlays every frame.

    init_groups: {group_index: 0/1} initial visibility.
    overlays:    (group_index|None, fn) list; fn runs iff group None or toggled on.
    camera:      optional {lookat, distance, azimuth, elevation}.
    frame:       optional mjtFrame for native site/body axis display.
    """
    overlays = list(overlays)
    dt = 1.0 / fps
    with viewer.launch_passive(model, data) as handle:
        for group, on in init_groups.items():
            handle.opt.geomgroup[group] = on
        if frame is not None:
            handle.opt.frame = frame
        if camera is not None:
            for key, value in camera.items():
                setattr(handle.cam, key, value)
        while handle.is_running():
            mujoco.mj_forward(model, data)
            handle.user_scn.ngeom = 0
            for group, fn in overlays:
                if group is None or handle.opt.geomgroup[group]:
                    fn(model, data, handle.user_scn)
            handle.sync()
            time.sleep(dt)


def render_png_strip(model, data, out, views: Iterable[tuple[float, float]], overlays: Iterable[Overlay],
                     *, lookat: Sequence[float], distance: float,
                     geomgroups: Iterable[int] | None = None, size: tuple[int, int] = (1200, 900)) -> None:
    """Offscreen multi-view strip (front|side|iso...). views: (azimuth, elevation) list.

    geomgroups: model geom groups to force visible in the base scene (else MuJoCo defaults).
    A fresh MjvOption hides groups 3/5 by default, so a static PNG must list every group it
    needs. Overlays are gated against this same set (group None always draws).
    """
    overlays = list(overlays)
    width, height = size
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    option = mujoco.MjvOption()
    if geomgroups is None:
        on_groups = set(range(len(option.geomgroup)))
    else:
        on_groups = set(geomgroups)
        option.geomgroup[:] = 0
        for group in on_groups:
            option.geomgroup[group] = 1
    camera = mujoco.MjvCamera()
    camera.lookat[:] = lookat
    camera.distance = distance
    images = []
    for azimuth, elevation in views:
        camera.azimuth, camera.elevation = azimuth, elevation
        renderer.update_scene(data, camera=camera, scene_option=option)
        for group, fn in overlays:
            if group is None or group in on_groups:
                fn(model, data, renderer.scene)
        images.append(renderer.render())
    renderer.close()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    imageio.imwrite(str(out), np.concatenate(images, axis=1))
    print(f"saved {out}")


# --- Family A: cuRobo collision spheres from a robot cfg dict via analytic FK ------------
# Shared verbatim by view_t1 / view_toddlerbot / view_berkeley_humanoid_lite; they differ only
# in which ``*_curobo_robot_cfg`` module they import and their per-link color regions.

def curobo_fk_state(model, data, build_robot_cfg_dict: Callable[[], dict], base_link: str,
                    regions: Sequence[tuple[str, Sequence[float]]]):
    """Build cuRobo Kinematics + source-qpos addresses + per-sphere colors for group-5 display.

    regions: (regex, rgba) list mapping link name -> color; first match wins, unmatched -> gray.
    Returns (kin, qpos_adr, colors, root_body_id) to feed ``add_curobo_fk_spheres``.
    """
    for path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs"), "/home/grl/repo/curobo"):
        if path not in sys.path:
            sys.path.insert(0, path)
    from curobo._src.robot.kinematics.kinematics import Kinematics
    from curobo._src.types.robot import RobotCfg

    cfg = build_robot_cfg_dict()
    data.qpos[:] = model.qpos0
    for name, value in cfg["robot_cfg"]["kinematics"]["lock_joints"].items():
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = value
    kin = Kinematics(RobotCfg.create(cfg).kinematics)
    qpos_adr = [int(model.jnt_qposadr[model.joint(name).id]) for name in kin.joint_names]
    colors = np.asarray([
        next((rgba for pattern, rgba in regions if re.search(pattern, link)), (0.7, 0.7, 0.7, 0.42))
        for link, spheres in cfg["robot_cfg"]["kinematics"]["collision_spheres"].items() for _ in spheres
    ], dtype=np.float32)
    return kin, qpos_adr, colors, model.body(base_link).id


def add_curobo_fk_spheres(data, scene, kin, qpos_adr, colors: np.ndarray, root_id: int) -> None:
    """Draw live cuRobo collision spheres in group 5 by FK-ing current qpos through cuRobo."""
    import torch
    from curobo._src.state.state_joint import JointState

    q = np.asarray([data.qpos[address] for address in qpos_adr], dtype=np.float32)
    state = JointState.from_position(torch.from_numpy(q).to(kin.device_cfg.device).unsqueeze(0),
                                     joint_names=kin.joint_names)
    spheres = kin.compute_kinematics(state).robot_spheres[0, 0].cpu().numpy()
    if len(spheres) != len(colors):
        raise RuntimeError(f"cuRobo sphere count {len(spheres)} != configured {len(colors)}")
    centers = spheres[:, :3] @ data.xmat[root_id].reshape(3, 3).T + data.xpos[root_id]
    add_spheres(scene, centers, spheres[:, 3], colors)


# --- Family B: cuRobo spheres straight off the planner (get_robot_as_spheres) --------------
# Shared verbatim by view_g1 / view_fourier_gr3; they differ only in the robot key and how they
# render the returned spheres (overlay vs baked worldbody geoms).

def planner_fk_spheres(model, robot_key: str, device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """cuRobo IK collision spheres at the home keyframe, base pinned to the origin.

    These are the spheres cuRobo actually collides during the dexterity IK solve, read straight
    off ``CuroboArmPlanner``'s kinematics rather than re-FK'd from a cfg dict (the Family-A path).
    Returns ``(centers Nx3, radii N)`` in the cuRobo base frame (== world when the base sits at
    the origin). Prints a radius summary for the collision-model approval strip.
    """
    import torch

    from mj_envs.asset_zoo.reachability_study import generate_workspace_curobo as gwc

    gwc._require_curobo()
    spec = gwc.get_robot_spec(robot_key)
    planner = gwc.CuroboArmPlanner(
        model, device=device, max_attempts=1, enable_graph_attempt=0, hand_z_floor=None,
        robot_cfg_dict=spec.robot_cfg_dict_fn(), base_link=spec.base_link, home_base_pos=spec.home_root_pos,
    )
    qpos_home = model.key_qpos[0].copy() if model.nkey else model.qpos0.copy()
    q_np = np.array(
        [qpos_home[planner._qpos_adr[i]] - planner._ref_offsets[i] for i in range(len(planner._joint_names))],
        dtype=np.float32,
    )
    result = planner._kin.get_robot_as_spheres(torch.from_numpy(q_np).unsqueeze(0).to(device))
    spheres = result[0] if (len(result) and isinstance(result[0], list)) else result
    centers = np.array([[s.position[0], s.position[1], s.position[2]] for s in spheres], dtype=np.float64)
    radii = np.array([s.radius for s in spheres], dtype=np.float64)
    print(f"cuRobo IK collision spheres: {len(spheres)} spheres; "
          f"radius min/mean/max = {radii.min():.3f}/{radii.mean():.3f}/{radii.max():.3f} m")
    return centers, radii
