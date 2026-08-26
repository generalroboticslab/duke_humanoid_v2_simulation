"""Generate collision-free arm-seed caches for the three standalone-MJCF study robots.

Berkeley Humanoid Lite, ToddlerBot, and Apollo ship as standalone source MJCFs, so the
project-wide mujoco-warp sampler (``mj_envs/asset_zoo/generate_safe_arm_poses.py``) does not
apply.  This lighter cuRobo-Kinematics sampler holds root/legs/torso at source ``qpos0``,
samples only the arm hinges within their exact MJCF limits, rejects self-collision, and writes
the cache schema consumed by ``generate_workspace_curobo.py``.

Per-robot safety gate differs (why one file, three configs):
  - berkeley   : MuJoCo contact count (``ncon``); no cuRobo model.
  - toddlerbot : BOTH ``ncon`` and cuRobo sphere pairs, plus sibling-gearbox drive coupling
                 written before FK.
  - apptronik_apollo : cuRobo sphere pairs only (source group-3 geoms mask contacts off, so
                 ``ncon`` cannot certify safety).

Usage:
    python mj_envs/asset_zoo/reachability_study/generate_curobo_safe_arm_poses.py \\
        --robot toddlerbot --samples 4096 --seed 42
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_CACHE_DIR = _REPO_ROOT / "mj_envs" / "asset_zoo" / "cache"

Robot = Literal["berkeley", "toddlerbot", "apptronik_apollo"]


@dataclass(frozen=True)
class RobotSpec:
    """Per-robot sampler configuration; ``build_cfg`` None disables the cuRobo gate."""

    mjcf: pathlib.Path
    arm_joint_names: tuple[str, ...]
    cache_path: pathlib.Path
    check_ncon: bool
    build_cfg: Callable[[], dict] | None
    drive_coupling: Callable[[mujoco.MjModel, mujoco.MjData], None] | None
    extra_meta: dict = field(default_factory=dict)


def _spec(robot: Robot) -> RobotSpec:
    """Build the selected robot's spec, importing its cuRobo cfg lazily.

    The lazy import keeps the berkeley path (no cuRobo model) free of the cuRobo toolchain.
    """
    if robot == "berkeley":
        from mj_envs.tasks.visual_manipulation.curobo.berkeley_humanoid_lite_curobo_robot_cfg import (
            BERKELEY_MJCF,
            LEFT_ARM_JOINT_NAMES,
            RIGHT_ARM_JOINT_NAMES,
        )

        return RobotSpec(
            mjcf=BERKELEY_MJCF,
            arm_joint_names=LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES,
            cache_path=_CACHE_DIR / "safe_arm_poses_berkeley_humanoid_lite.pt",
            check_ncon=True,
            build_cfg=None,
            drive_coupling=None,
        )
    if robot == "toddlerbot":
        from mj_envs.tasks.visual_manipulation.curobo.toddlerbot_curobo_robot_cfg import (
            LEFT_ARM_JOINT_NAMES,
            RIGHT_ARM_JOINT_NAMES,
            TODDLERBOT_MJCF,
            apply_source_drive_coupling,
            build_robot_cfg_dict,
        )

        return RobotSpec(
            mjcf=TODDLERBOT_MJCF,
            arm_joint_names=LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES,
            cache_path=_CACHE_DIR / "safe_arm_poses_toddlerbot.pt",
            check_ncon=True,
            build_cfg=build_robot_cfg_dict,
            drive_coupling=apply_source_drive_coupling,
            extra_meta={"source_drive_coupling": "six arm *_drive qpos values set to negative *_driven qpos"},
        )
    if robot == "apptronik_apollo":
        from mj_envs.tasks.visual_manipulation.curobo.apptronik_apollo_curobo_robot_cfg import (
            APOLLO_MJCF,
            LEFT_ARM_JOINT_NAMES,
            RIGHT_ARM_JOINT_NAMES,
            build_robot_cfg_dict,
        )

        return RobotSpec(
            mjcf=APOLLO_MJCF,
            arm_joint_names=LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES,
            cache_path=_CACHE_DIR / "safe_arm_poses_apptronik_apollo.pt",
            check_ncon=False,
            build_cfg=build_robot_cfg_dict,
            drive_coupling=None,
            extra_meta={"safety_gate": "approved source-group-3 cuRobo sphere self-collision pairs"},
        )
    raise ValueError(f"unknown robot {robot!r}")


def _site_pose(data: mujoco.MjData, site_id: int, home_root_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoint pose in root ``base_link`` coordinates (wxyz quat), not world coordinates."""
    xyzw = Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat()
    return (data.site_xpos[site_id] - home_root_pos).astype(np.float32), np.array(
        [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32
    )


@torch.inference_mode()
def _curobo_collision_free(
    kin,
    collision_pairs: torch.Tensor,
    sphere_padding: torch.Tensor,
    kin_qpos_indices: np.ndarray,
    data: mujoco.MjData,
) -> bool:
    """Require every configured cuRobo sphere pair to be disjoint at this source pose.

    ``kin_qpos_indices`` maps ``kin.joint_names`` order to source qpos slots, so the sampled
    ``q`` stays aligned with the joint names cuRobo expects.
    """
    from curobo._src.state.state_joint import JointState

    q = torch.as_tensor(data.qpos[kin_qpos_indices], dtype=torch.float32, device=kin.device_cfg.device)
    spheres = kin.compute_kinematics(
        JointState.from_position(q.unsqueeze(0), joint_names=kin.joint_names)
    ).robot_spheres[0, 0]
    first, second = collision_pairs[:, 0], collision_pairs[:, 1]
    separation = torch.linalg.vector_norm(spheres[first, :3] - spheres[second, :3], dim=1)
    required = spheres[first, 3] + spheres[second, 3] + sphere_padding[first] + sphere_padding[second]
    return bool(torch.all(separation >= required))


def generate(spec: RobotSpec, samples: int, seed: int, out: pathlib.Path) -> None:
    """Write ``samples`` collision-free both-arm poses, sorted nearest source rest first."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    arm_names = spec.arm_joint_names
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf))
    data = mujoco.MjData(model)
    home_root_pos = model.qpos0[:3].copy()
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_names])
    if np.any(joint_ids < 0):
        missing = [n for n, jid in zip(arm_names, joint_ids) if jid < 0]
        raise ValueError(f"source MJCF missing arm joints: {missing}")
    qpos_indices = model.jnt_qposadr[joint_ids].astype(np.int64)
    lower, upper = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_L_site")
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector_R_site")
    if left_site < 0 or right_site < 0:
        raise ValueError("source MJCF lacks endpoint sites end_effector_{L,R}_site")

    kin = collision_pairs = sphere_padding = kin_qpos_indices = None
    if spec.build_cfg is not None:
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.types.robot import RobotCfg

        kin = Kinematics(RobotCfg.create(spec.build_cfg()).kinematics)
        self_collision = kin.get_self_collision_config()
        collision_pairs = self_collision.collision_pairs.long()
        sphere_padding = self_collision.sphere_padding
        kin_qpos_indices = np.array(
            [model.jnt_qposadr[model.joint(n).id] for n in kin.joint_names], dtype=np.int64
        )

    # Intentional static rest contacts (adjacent-link housing overlap, e.g. Berkeley's shoulder
    # pitch<->yaw, and hip_yaw<->base) exist at qpos0 and are not real self-collisions; baseline
    # them so the gate rejects only NEW contact pairs a sampled pose introduces. Matches the readme
    # "Adding a new robot" step 8: reject real self-collision *except named, intentional static rest
    # contacts*. Empty for robots whose rest pose is already contact-free (e.g. toddlerbot), so their
    # gate stays identical to a plain ncon!=0 reject.
    rest_contacts: frozenset[tuple[int, int]] = frozenset()
    if spec.check_ncon:
        data.qpos[:] = model.qpos0
        if spec.drive_coupling is not None:
            spec.drive_coupling(model, data)
        mujoco.mj_forward(model, data)
        rest_contacts = frozenset(
            (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
            for c in data.contact[: data.ncon]
        )

    rng = np.random.default_rng(seed)
    poses: list[np.ndarray] = []
    ee_l_pos: list[np.ndarray] = []
    ee_l_quat: list[np.ndarray] = []
    ee_r_pos: list[np.ndarray] = []
    ee_r_quat: list[np.ndarray] = []
    attempts = source_contact_rejections = curobo_collision_rejections = 0
    while len(poses) < samples:
        attempts += 1
        data.qpos[:] = model.qpos0
        data.qpos[qpos_indices] = rng.uniform(lower, upper)
        if spec.drive_coupling is not None:
            spec.drive_coupling(model, data)
        mujoco.mj_forward(model, data)
        if spec.check_ncon and any(
            (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2))) not in rest_contacts
            for c in data.contact[: data.ncon]
        ):
            source_contact_rejections += 1
            continue
        if kin is not None and not _curobo_collision_free(
            kin, collision_pairs, sphere_padding, kin_qpos_indices, data
        ):
            curobo_collision_rejections += 1
            continue
        l_pos, l_quat = _site_pose(data, left_site, home_root_pos)
        r_pos, r_quat = _site_pose(data, right_site, home_root_pos)
        poses.append(data.qpos[qpos_indices].astype(np.float32).copy())
        ee_l_pos.append(l_pos)
        ee_l_quat.append(l_quat)
        ee_r_pos.append(r_pos)
        ee_r_quat.append(r_quat)

    pose_array = np.stack(poses)
    order = np.linalg.norm(pose_array - model.qpos0[qpos_indices], axis=1).argsort()
    payload = {
        "poses": torch.from_numpy(pose_array[order]),
        "joint_names": list(arm_names),
        "qpos_indices": torch.from_numpy(qpos_indices),
        "ee_l_pos": torch.from_numpy(np.stack(ee_l_pos)[order]),
        "ee_l_quat": torch.from_numpy(np.stack(ee_l_quat)[order]),
        "ee_r_pos": torch.from_numpy(np.stack(ee_r_pos)[order]),
        "ee_r_quat": torch.from_numpy(np.stack(ee_r_quat)[order]),
        "source_mjcf": str(spec.mjcf),
        "source_snapshot": (model.nbody, model.njnt, model.ngeom, model.nmesh, model.nsite, model.nu),
        "seed": seed,
        "attempts": attempts,
        "acceptance_rate": samples / attempts,
        "sorted_by_l2_from_source_rest": True,
        **spec.extra_meta,
    }
    if spec.check_ncon:
        payload["source_contact_rejections"] = source_contact_rejections
    if kin is not None:
        payload["curobo_collision_rejections"] = curobo_collision_rejections
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(
        f"saved {samples} {spec.mjcf.stem} safe arm poses to {out} "
        f"({samples / attempts:.1%} accepted; {attempts} attempts)"
    )


def main(
    robot: Robot,
    samples: int = 4096,
    seed: int = 42,
    out: pathlib.Path | None = None,
) -> None:
    """Sample one robot's collision-free arm-seed cache (``--robot`` selects the safety gate)."""
    spec = _spec(robot)
    generate(spec, samples, seed, spec.cache_path if out is None else out)


if __name__ == "__main__":
    tyro.cli(main)
