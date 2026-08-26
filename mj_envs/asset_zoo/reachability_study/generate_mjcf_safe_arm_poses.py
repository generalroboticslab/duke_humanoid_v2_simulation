"""Collision-free arm-pose caches for the study robots whose source MJCF self-certifies.

Companion to ``generate_curobo_safe_arm_poses.py``. Both write the same cache schema; they
differ in how a sampled pose is certified safe:

  * **this file** -- ``data.ncon == 0`` alone. Valid only because each of these three source
    MJCFs declares *all* of its structural rest overlaps as ``<contact><exclude>`` pairs, so
    any reported contact is a real, avoidable self-collision. No cuRobo toolchain needed.
  * **generate_curobo_safe_arm_poses.py** -- cuRobo sphere-pair gating, for sources that
    cannot make that guarantee (Apollo's group-3 geoms mask contacts off entirely) or that
    need drive coupling written before FK (ToddlerBot).

Keeping them separate is the point: importing cuRobo to sample a robot that does not need it
is what made the per-robot copies diverge in the first place.

The cache is the seed pool the cuRobo IK solver draws its restarts from, and its end-effector
cloud is what ``generate_workspace_curobo.py --grid-bounds-source legacy-symmetric`` derives
the voxel-grid bounds from, so it must span the arms' full range, not a comfortable subset.

Method: rejection-sample all 14 arm joints uniformly inside their source limits, hold every
other joint at the source home keyframe, keep a sample only if MuJoCo reports zero contacts.
Poses are stored as the raw 14 arm coordinates; the two end-effector poses are stored in the
``base_link`` frame (the floating root is subtracted) because every downstream consumer --
grid bounds, IK targets, payload positions -- lives in that frame.

Usage:
    python mj_envs/asset_zoo/reachability_study/generate_mjcf_safe_arm_poses.py \\
        --robot pal_talos --samples 4096 --seed 42
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
import torch
import tyro

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

_CACHE_DIR = _REPO_ROOT / "mj_envs" / "asset_zoo" / "cache"

# Uniform across all three sources -- kept as module constants rather than RobotSpec fields
# because a source that disagreed would need a different frame convention downstream anyway.
BASE_LINK = "base_link"
EE_SITES = ("end_effector_L_site", "end_effector_R_site")

Robot = Literal["booster_t1", "fourier_gr3", "pal_talos"]


@dataclass(frozen=True)
class RobotSpec:
    """Per-robot sampler configuration.

    ``limit_margin`` shrinks each joint's sampling interval by that fraction of its range at
    both ends. Booster T1 uses 0.05 because its wrist geoms graze the forearm exactly at the
    limits and a hard-limit sample wastes solver restarts; GR-3 and TALOS sample the full
    declared range. ``rest_overlap_fix`` is quoted in the home-pose assertion failure so a
    stale-excludes error names its own remedy -- an import script for the two robots that
    have one, and a plain statement of fact for vendor-shipped T1, which has none.
    """

    mjcf: pathlib.Path
    arm_joints: tuple[str, ...]
    cache_path: pathlib.Path
    rest_overlap_fix: str
    limit_margin: float = 0.0


def _spec(robot: Robot) -> RobotSpec:
    """Build the selected robot's spec, importing T1's cuRobo joint list lazily.

    T1's arm joint names live in its cuRobo cfg so the sampler and the solver cannot drift;
    GR-3 and TALOS have no cuRobo cfg, so their names are declared here.
    """
    if robot == "booster_t1":
        from mj_envs.tasks.visual_manipulation.curobo.booster_t1_curobo_robot_cfg import (
            LEFT_ARM_JOINT_NAMES,
            RIGHT_ARM_JOINT_NAMES,
        )

        return RobotSpec(
            mjcf=_REPO_ROOT / "asset" / "booster_t1" / "t1.xml",
            arm_joints=tuple(LEFT_ARM_JOINT_NAMES) + tuple(RIGHT_ARM_JOINT_NAMES),
            cache_path=_CACHE_DIR / "safe_arm_poses_booster_t1.pt",
            rest_overlap_fix=(
                "t1.xml is vendor-shipped with no import script and no static rest overlaps; "
                "a contact here means the asset changed"
            ),
            limit_margin=0.05,
        )
    if robot == "fourier_gr3":
        left = (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
        )
        return RobotSpec(
            mjcf=_REPO_ROOT / "asset" / "fourier_gr3" / "gr3.xml",
            arm_joints=left + tuple(n.replace("left_", "right_") for n in left),
            cache_path=_CACHE_DIR / "safe_arm_poses_fourier_gr3.pt",
            rest_overlap_fix="re-run asset/fourier_gr3/gr3_import.py",
        )
    if robot == "pal_talos":
        # arm_*_1..3 = shoulder pitch/roll/yaw, arm_*_4 = elbow pitch, arm_*_5..7 = wrist
        # yaw/pitch/roll. TALOS's home is the `standing` keyframe, NOT qpos0: the vendor zero
        # pose is not even legal, since arm_*_2 and arm_*_4 have limit intervals excluding
        # zero. Sampling is against the source limits either way, so only held joints care --
        # and every robot here reads key_qpos[0], so nothing robot-specific is needed.
        return RobotSpec(
            mjcf=_REPO_ROOT / "asset" / "pal_talos" / "talos_study.xml",
            arm_joints=tuple(f"arm_left_{i}_joint" for i in range(1, 8))
            + tuple(f"arm_right_{i}_joint" for i in range(1, 8)),
            cache_path=_CACHE_DIR / "safe_arm_poses_pal_talos.pt",
            rest_overlap_fix="re-run asset/pal_talos/talos_study_import.py",
        )
    raise ValueError(f"unknown robot {robot!r}")


def _joint_info(
    model: mujoco.MjModel, spec: RobotSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (qpos addresses, lower bounds, upper bounds) for the arm joints."""
    addrs, lo, hi = [], [], []
    for name in spec.arm_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name!r} missing from {spec.mjcf}")
        if not model.jnt_limited[jid]:
            raise ValueError(f"joint {name!r} is unlimited; cannot sample uniformly")
        j_lo, j_hi = (float(v) for v in model.jnt_range[jid])
        margin = (j_hi - j_lo) * spec.limit_margin
        addrs.append(int(model.jnt_qposadr[jid]))
        lo.append(j_lo + margin)
        hi.append(j_hi - margin)
    return np.array(addrs, dtype=np.int64), np.array(lo), np.array(hi)


def _quat_of(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix (flat or nested) -> wxyz quaternion."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, mat.reshape(9))
    return q


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[1:] *= -1.0
    return out


def generate(spec: RobotSpec, samples: int, batch: int, seed: int, out: pathlib.Path, force: bool) -> None:
    """Write ``samples`` contact-free both-arm poses with EE poses in the base_link frame."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if out.exists() and not force:
        cache = torch.load(str(out), map_location="cpu", weights_only=False)
        if cache["ee_l_pos"].shape[0] >= samples:
            print(f"cache already has {cache['ee_l_pos'].shape[0]:,} poses -- use --force to regen")
            return

    model = mujoco.MjModel.from_xml_path(str(spec.mjcf))
    data = mujoco.MjData(model)
    if model.nkey < 1:
        raise ValueError(f"{spec.mjcf.name} has no home keyframe")
    home_qpos = model.key_qpos[0].copy()

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_LINK)
    if base_id < 0:
        raise ValueError(f"body {BASE_LINK!r} missing from {spec.mjcf}")
    site_ids = []
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"site {name!r} missing from {spec.mjcf}")
        site_ids.append(sid)

    addrs, lo, hi = _joint_info(model, spec)
    n_dof = len(addrs)

    # The untouched home pose must itself be contact-free, else the ncon==0 criterion is
    # meaningless and every sample would be rejected.
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)
    if data.ncon > 0:
        raise ValueError(
            f"home keyframe already reports {data.ncon} contacts; the rest-overlap excludes in "
            f"{spec.mjcf.name} are stale -- {spec.rest_overlap_fix}"
        )

    rng = np.random.default_rng(seed)
    poses = np.empty((samples, n_dof), dtype=np.float32)
    ee_pos = [np.empty((samples, 3), dtype=np.float32) for _ in EE_SITES]
    ee_quat = [np.empty((samples, 4), dtype=np.float32) for _ in EE_SITES]

    collected = attempts = 0
    t0 = time.time()
    print(f"sampling {samples:,} contact-free {spec.mjcf.stem} arm poses ({n_dof} DOF, batch={batch})")
    while collected < samples:
        q_batch = rng.uniform(lo, hi, size=(batch, n_dof)).astype(np.float32)
        attempts += batch
        for row in q_batch:
            if collected >= samples:
                break
            data.qpos[:] = home_qpos
            data.qpos[addrs] = row
            mujoco.mj_forward(model, data)
            if data.ncon > 0:
                continue
            base_pos = data.xpos[base_id].copy()
            inv = _quat_conj(_quat_of(data.xmat[base_id]))
            for k, sid in enumerate(site_ids):
                rel = data.site_xpos[sid] - base_pos
                pos_b = np.empty(3)
                mujoco.mju_rotVecQuat(pos_b, rel, inv)
                quat_b = np.empty(4)
                mujoco.mju_mulQuat(quat_b, inv, _quat_of(data.site_xmat[sid]))
                ee_pos[k][collected] = pos_b
                ee_quat[k][collected] = quat_b
            poses[collected] = row
            collected += 1
        rate = collected / max(time.time() - t0, 1e-6)
        print(f"  {collected:>7,}/{samples:,}  yield={100.0 * collected / attempts:5.1f}%  "
              f"{rate:6.0f} poses/s", end="\r")

    print(f"\ncollected {collected:,} from {attempts:,} samples "
          f"(yield {100.0 * collected / attempts:.1f}%)")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            poses=torch.from_numpy(poses),
            joint_names=list(spec.arm_joints),
            qpos_indices=torch.from_numpy(addrs),
            ee_l_pos=torch.from_numpy(ee_pos[0]),
            ee_l_quat=torch.from_numpy(ee_quat[0]),
            ee_r_pos=torch.from_numpy(ee_pos[1]),
            ee_r_quat=torch.from_numpy(ee_quat[1]),
            mjcf_hash=None,
            mjcf_path=str(spec.mjcf),
            home_qpos=torch.from_numpy(home_qpos.astype(np.float32)),
            sorted_by_l2_from_default=False,
            gen_attempts=attempts,
            gen_yield=collected / max(attempts, 1),
        ),
        str(out),
    )
    lo_ee = np.minimum(ee_pos[0].min(axis=0), ee_pos[1].min(axis=0))
    hi_ee = np.maximum(ee_pos[0].max(axis=0), ee_pos[1].max(axis=0))
    print(f"saved -> {out}")
    print(f"EE cloud AABB (base_link frame): min {np.round(lo_ee, 4)}  max {np.round(hi_ee, 4)}")


def main(
    robot: Robot,
    samples: int = 4096,
    batch: int = 512,
    seed: int = 42,
    out: pathlib.Path | None = None,
    force: bool = False,
) -> None:
    """Sample one robot's contact-free arm-seed cache (``--out`` overrides the default path)."""
    spec = _spec(robot)
    generate(spec, samples, batch, seed, spec.cache_path if out is None else out, force)


if __name__ == "__main__":
    tyro.cli(main)
