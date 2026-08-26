"""
Generate a cached dataset of collision-free arm poses using GPU-accelerated MuJoCo (mujoco_warp).

Pipeline:
  1. Auto-identify arm joints via kinematic tree traversal (anchor-based DFS).
  2. Batch-sample random arm poses across parallel GPU envs; discard self-collisions.
  3. Compute pose difficulties (2D CoM-to-support-center distance) and EE poses in pelvis frame.
  4. Sort by L2 distance from default pose; save to cache.

Yield & Speed (humanoid_v21, 14 arm DOF, batch=8192, RTX 4090):
  - Yield rate: ~0.6% (collision-free / total sampled). Low because the safe manifold is many
    small disconnected pockets in a 14-DOF joint space.
  - Throughput: ~0.73s per 16k samples → ~1.7 min for 2M poses.

Optimization history — what was tried:
  WORKED:
    - ccd_iterations=0: default is 50, explicit 0 disables CCD (continuous collision detection).
      CCD is irrelevant for static FK-only snapshots; removing it gave the primary speedup.
    - fwd_position CUDA graph: captures FK-only pass (no dynamics, no collision solve) as a
      replayable CUDA graph — eliminates kernel launch overhead per batch.

  DID NOT WORK:
    - MCMC sampling (Metropolis-Hastings on joint space): tested two variants:
        1. All chains seeded from q_default → q_default is in/near collision for humanoid_v21,
           so chains produced 3-4 accepted samples per 8192 (worse than uniform's ~50).
        2. Two-phase warm-up: seed chains from uniform-found safe poses, then run MCMC.
           Yield stayed ~0.6% — same as uniform. MCMC gives no advantage when the safe
           manifold is small disconnected pockets (chains can't diffuse between them).
        3. Smaller step_size (0.05/sqrt(ndof)): yield dropped to 0.3% (over-local exploration).

  NOT YET TRIED (future ideas):
    - Rejection sampling with learned collision classifier (predict collision before sim call).
    - Workspace decomposition: partition joint space into regions, sample each with separate bias.
    - Larger batch (>16k): limited by GPU VRAM; mjwarp allocates contact buffers per-env.

Usage:
    python mj_envs/asset_zoo/generate_safe_arm_poses.py --robot unitree_g1 --vis
    python mj_envs/asset_zoo/generate_safe_arm_poses.py --robot humanoid_v21 --vis
    python mj_envs/asset_zoo/generate_safe_arm_poses.py --robot humanoid_v21 --rebuild-fk
    python mj_envs/asset_zoo/generate_safe_arm_poses.py --robot unitree_g1 --manipulation --quick-vis
        (live random-pose sanity check against the grafted welded+parallel_gripper+actuated-cam
         G1 used by G1RmaVelEstArmFlashSacL2T/ActuatedCam -- no cache, no generation)
    python mj_envs/asset_zoo/generate_safe_arm_poses.py --robot openarm_v2

Viewer Controls: TAB sidebar, F1 help.
"""

import sys
import time
from pathlib import Path
from typing import Literal

import torch
import tyro
import xxhash
import mujoco

import warp as wp
import mujoco_warp as mjwarp

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from mjlab.entity import Entity
from mjlab.sim.sim import Simulation, SimulationCfg
from mj_envs.utils.torch_math_utils import _rot_matrix_to_quat


# humanoid_v21 arm bone collision capsules → child body whose offset defines the bone.
# Used by realistic_collision to rebuild each capsule as a full joint-to-joint shaft.
_ARM_BONE_MAP_HUMANOID = {
    f"{seg}_{s}_collision": f"{child}_{s}"
    for s in ("L", "R")
    for seg, child in (
        ("shoulder_2", "shoulder_3"),
        ("shoulder_3", "elbow"),
        ("elbow", "wrist_1"),
        ("wrist_1", "wrist_2"),
        ("wrist_2", "wrist_3"),
        ("wrist_3", "end_effector"),
    )
}


def _quat_z_to(d):
    """wxyz quaternion rotating local +z onto unit vector d."""
    import numpy as np
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis)); c = float(np.dot(z, d))
    if s < 1e-8:
        return [1.0, 0.0, 0.0, 0.0] if c > 0 else [0.0, 1.0, 0.0, 0.0]
    axis /= s; ang = np.arctan2(s, c)
    return [np.cos(ang / 2.0), *(axis * np.sin(ang / 2.0))]


def _apply_spec_geometry(spec, link_deltas: dict, realistic_collision: bool, robot: str) -> None:
    """Edit arm geometry in the MjSpec BEFORE compile (so derived collision quantities
    like geom_rbound / aabb are recomputed — editing the compiled model's geom_size
    leaves broadphase bounds stale and silently drops collisions).

    1. link_deltas: lengthen each named bone by moving its body.pos along local −y.
    2. realistic_collision: rebuild every arm collision capsule as a full joint-to-joint
       shaft (pos=midpoint of child offset, half-length=|offset|/2, quat aligning +z to
       the bone). Radius preserved. Read AFTER deltas so longer bones get longer shafts.
    """
    import numpy as np
    bodies = {b.name: b for b in spec.bodies}

    for nm, delta in link_deltas.items():
        if nm not in bodies:
            raise ValueError(f"link_deltas body '{nm}' not found in spec")
        p = np.array(bodies[nm].pos, dtype=float); p[1] -= delta
        bodies[nm].pos = p

    if realistic_collision:
        if robot != "humanoid_v21":
            raise ValueError("realistic_collision only supported for humanoid_v21")
        geoms = {g.name: g for b in spec.bodies for g in b.geoms}
        for gname, cname in _ARM_BONE_MAP_HUMANOID.items():
            if gname not in geoms or cname not in bodies:
                raise ValueError(f"realistic_collision: missing geom '{gname}' or body '{cname}'")
            g = geoms[gname]
            cp = np.array(bodies[cname].pos, dtype=float)  # child offset in geom's body frame
            L = float(np.linalg.norm(cp))
            if L < 1e-6:
                continue
            g.pos = cp / 2.0
            g.size = [float(g.size[0]), L / 2.0, 0.0]
            g.quat = _quat_z_to(cp / L)


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz convention throughout)
# ---------------------------------------------------------------------------

def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of wxyz quaternion. (..., 4) -> (..., 4)."""
    return q * q.new_tensor([1., -1., -1., -1.])

def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q using Rodrigues' formula. q: (..., 4), v: (..., 3)."""
    w, xyz = q[..., 0:1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)

def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1*q2. Both (..., 4) wxyz."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


# ---------------------------------------------------------------------------
# EE body ID lookup
# ---------------------------------------------------------------------------

def _get_ee_frame_ids(model: mujoco.MjModel, robot: str) -> tuple[int, int, int, bool]:
    """Return (pelvis_id, ee_l_id, ee_r_id, ee_is_site) for the given robot.

    pelvis_id: body index for the reference frame.
    ee_l_id / ee_r_id: frame indices (body or site) for left/right EE.
    ee_is_site: True when EE frames are MuJoCo sites (humanoid_v21).
    """
    if robot == "unitree_g1":
        pelvis_name = "pelvis"
        ee_l_name   = "left_palm"
        ee_r_name   = "right_palm"
        ee_is_site  = True
    else:  # humanoid_v21 — EE frames are sites on wrist_3_L/R
        pelvis_name = "base_link"
        ee_l_name   = "end_effector_L_site"
        ee_r_name   = "end_effector_R_site"
        ee_is_site  = True

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pelvis_name)
    obj_type  = mujoco.mjtObj.mjOBJ_SITE if ee_is_site else mujoco.mjtObj.mjOBJ_BODY
    ee_l_id   = mujoco.mj_name2id(model, obj_type, ee_l_name)
    ee_r_id   = mujoco.mj_name2id(model, obj_type, ee_r_name)

    if pelvis_id < 0 or ee_l_id < 0 or ee_r_id < 0:
        frame_type = "site" if ee_is_site else "body"
        raise ValueError(
            f"{frame_type} lookup failed for robot='{robot}': "
            f"pelvis={pelvis_id} ({pelvis_name}), "
            f"ee_l={ee_l_id} ({ee_l_name}), ee_r={ee_r_id} ({ee_r_name})"
        )
    return pelvis_id, ee_l_id, ee_r_id, ee_is_site


def _extract_ee_in_pelvis(
    xpos: torch.Tensor,    # (B, nbody, 3) world body positions
    xquat: torch.Tensor,   # (B, nbody, 4) world body orientations wxyz
    pelvis_id: int,
    ee_l_id: int,
    ee_r_id: int,
    site_xpos: torch.Tensor | None = None,   # (B, nsite, 3) — required when ee_is_site
    site_xmat: torch.Tensor | None = None,   # (B, nsite, 3, 3) — required when ee_is_site
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute EE positions and orientations expressed in pelvis frame.

    Args:
        xpos / xquat: body FK arrays — used for pelvis frame and (when site_xpos is None) EE pose.
        pelvis_id, ee_l_id, ee_r_id: frame indices (body when site_xpos is None, else site).
        site_xpos / site_xmat: site FK arrays; supply for robots where EE frames are sites.

    Returns:
        (ee_l_pos, ee_l_quat, ee_r_pos, ee_r_quat) — each (B, 3) or (B, 4) in pelvis frame.
    """
    pelvis_pos  = xpos[:, pelvis_id, :]    # (B, 3)
    pelvis_quat = xquat[:, pelvis_id, :]   # (B, 4)
    inv_pelvis  = _quat_conjugate(pelvis_quat)

    results = []
    for ee_id in (ee_l_id, ee_r_id):
        if site_xpos is not None:
            ee_pos_w  = site_xpos[:, ee_id, :]
            ee_quat_w = _rot_matrix_to_quat(site_xmat[:, ee_id])
        else:
            ee_pos_w  = xpos[:, ee_id, :]
            ee_quat_w = xquat[:, ee_id, :]
        pos_p  = _quat_apply(inv_pelvis, ee_pos_w - pelvis_pos)
        quat_p = _quat_multiply(inv_pelvis, ee_quat_w)
        results.extend([pos_p, quat_p])

    return results[0], results[1], results[2], results[3]



def get_arm_joint_info(model, device, include_waist: bool = True) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    """Identify arm joints via kinematic tree traversal. Returns joints in natural model order.

    Strategy: anchor on bodies named "shoulder/elbow/arm" → verify connected to torso →
    DFS-collect entire subtree.  This captures any leaf body regardless of its name.

    Args:
        model: Compiled MuJoCo model.
        device: Device to allocate tensors on.
        include_waist: If True, also include waist/spine joints.

    Returns:
        (qpos_indices, arm_dof_names, lower_bounds, upper_bounds)
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Find the torso body (body with the most children)
    child_counts = [0] * model.nbody
    for i in range(model.nbody):
        p = model.body_parentid[i]
        if p >= 0:
            child_counts[p] += 1

    torso_body = max(range(1, model.nbody), key=lambda i: child_counts[i])
    print(f"\033[92m[generate_safe_arm_poses] Torso body: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, torso_body)}\033[0m")

    # Build children list for downward (torso→leaf) traversal.
    children = [[] for _ in range(model.nbody)]
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if p >= 0:
            children[p].append(b)

    # Anchor on upper-limb bodies (shoulder/elbow/arm), then collect the full subtree
    # downward to leaves.  This captures all arm bodies regardless of how the EE body
    # is named — avoiding silent omissions when leaves use non-standard names.
    arm_bodies = set()
    arm_anchor_keywords = ["shoulder", "elbow", "arm"]

    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and any(k in name.lower() for k in arm_anchor_keywords):
            # Verify this anchor body is actually connected to the torso
            curr = i
            while curr != torso_body and curr > 0:
                curr = model.body_parentid[curr]
            if curr != torso_body:
                continue

            # Collect the entire subtree rooted at this anchor (DFS downward)
            stack = [i]
            while stack:
                b = stack.pop()
                arm_bodies.add(b)
                stack.extend(children[b])

    arm_qpos_indices = []
    arm_joint_ids = []
    arm_dof_names = []

    for j in range(model.njnt):
        if model.jnt_type[j] not in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            continue

        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        is_upper_body = model.jnt_bodyid[j] in arm_bodies
        is_waist_or_spine = "waist" in name.lower() or "spine" in name.lower()

        if is_waist_or_spine and include_waist:
            arm_qpos_indices.append(model.jnt_qposadr[j])
            arm_joint_ids.append(j)
            arm_dof_names.append(name)
        elif is_upper_body and not is_waist_or_spine:
            arm_qpos_indices.append(model.jnt_qposadr[j])
            arm_joint_ids.append(j)
            arm_dof_names.append(name)

    if not arm_qpos_indices:
        raise ValueError("No arm joints identified. Check robot configuration.")

    qpos_indices = torch.tensor(arm_qpos_indices, device=device, dtype=torch.long)

    jnt_range = torch.tensor(model.jnt_range, device=device, dtype=torch.float32)
    arm_jnt_range = jnt_range[arm_joint_ids]

    margin = (arm_jnt_range[:, 1] - arm_jnt_range[:, 0]) * 0.05  # 5% inset — avoid extremes
    lower_bounds = arm_jnt_range[:, 0] + margin
    upper_bounds = arm_jnt_range[:, 1] - margin
    return qpos_indices, arm_dof_names, lower_bounds, upper_bounds

def visualize_poses(model: mujoco.MjModel,
                    qpos_indices: torch.Tensor,
                    lower_bounds: torch.Tensor,
                    upper_bounds: torch.Tensor,
                    device: str,
                    batch_size: int,
                    foot_l_id: int,
                    foot_r_id: int,
                    all_safe_poses: torch.Tensor = None) -> None:
    """Interactive 3D visualization of arm poses (safe=green, collision=red).
    If all_safe_poses is provided, plays them back sequentially (sorted order)."""
    import mujoco.viewer as viewer
    import numpy as np
    sim_cfg = SimulationCfg(nconmax=128, njmax=384)  # bumped from 32/64: welded parallel_gripper capsule colliders raise ncon past 32 (overflowed at ncon=62)
    sim_cfg.mujoco.ccd_iterations = 0  # static snapshot — no motion, CCD unnecessary
    sim = Simulation(num_envs=batch_size, cfg=sim_cfg, model=model, device=device)

    wp.capture_begin(sim.wp_device)
    mjwarp._src.forward.fwd_position(sim.wp_model, sim.wp_data, factorize=False)
    fwd_pos_graph = wp.capture_end(sim.wp_device)

    cpu_data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    model.geom_rgba[:, 3] = 0.3  # Make the robot translucent to see the CoM indicators
    model.vis.scale.com = 0.1     # shrink the mjVIS_COM marker (default 0.4 is huge at the pelvis)
    model.vis.rgba.com[3] = 0.5   # 50% transparent so it doesn't occlude the body

    # Pre-allocate rendering vectors
    sc_3d_np = np.zeros(3, dtype=np.float32)
    com_3d_np = np.zeros(3, dtype=np.float32)
    geom_mat = np.eye(3).flatten()
    rgba_blue = np.array([0, 0, 1, 0.5], dtype=np.float32)
    rgba_red = np.array([1, 0, 0, 0.5], dtype=np.float32)

    random_noise = torch.empty((batch_size, len(qpos_indices)), device=device, dtype=torch.float32)
    env_collision = torch.zeros(batch_size, dtype=torch.bool, device=device)
    geom_bodyid = torch.tensor(model.geom_bodyid, device=device)

    print(f"Visualizing {'sorted' if all_safe_poses is not None else 'random'} poses. Press Ctrl+C to stop.")
    pose_idx = 0
    with viewer.launch_passive(model, cpu_data) as v_handle:
        while v_handle.is_running():
            sim.reset()
            if all_safe_poses is not None:
                # Playback sorted poses sequentially
                current_batch_size = min(batch_size, all_safe_poses.shape[0] - pose_idx)
                if current_batch_size <= 0:
                    pose_idx = 0 # Loop back
                    current_batch_size = min(batch_size, all_safe_poses.shape[0])

                sim.data.qpos[:current_batch_size, qpos_indices] = all_safe_poses[pose_idx : pose_idx + current_batch_size]
                pose_idx += current_batch_size
            else:
                # Original random sampling behavior
                random_noise.uniform_()
                random_arm_poses = lower_bounds + random_noise * (upper_bounds - lower_bounds)
                sim.data.qpos[:, qpos_indices] = random_arm_poses

            with wp.ScopedDevice(device):
                wp.capture_launch(fwd_pos_graph)

            env_collision.zero_()
            nacon = sim.data.nacon[0].item()
            is_collision_contact = (sim.data.contact.dist[:nacon] < 0.0) & (sim.data.contact.worldid[:nacon] >= 0)

            # Filter collisions: only consider internal robot collisions
            geom1 = sim.data.contact.geom[:nacon, 0][is_collision_contact]
            geom2 = sim.data.contact.geom[:nacon, 1][is_collision_contact]
            is_self_collision = (geom_bodyid[geom1] > 0) & (geom_bodyid[geom2] > 0)

            valid_worldids = sim.data.contact.worldid[:nacon][is_collision_contact][is_self_collision]
            valid_mask = valid_worldids < batch_size
            env_collision[valid_worldids[valid_mask]] = True

            safe_indices = (~env_collision).nonzero(as_tuple=True)[0]
            unsafe_indices = env_collision.nonzero(as_tuple=True)[0]

            if safe_indices.numel() > 0:
                indices_to_show = safe_indices
                is_safe = True
            else:
                indices_to_show = unsafe_indices
                is_safe = False

            for idx in indices_to_show.tolist():
                cpu_data.qpos[:] = sim.data.qpos[idx].cpu().numpy()
                mujoco.mj_forward(model, cpu_data)

                # Recompute dynamic support center for the current visualization pose
                foot_l_pos_3d = sim.data.xpos[idx, foot_l_id, :].cpu().numpy()  # full XYZ
                foot_r_pos_3d = sim.data.xpos[idx, foot_r_id, :].cpu().numpy()  # full XYZ
                # Use actual foot height so the sphere sits on the foot, not at z=0
                sc_3d_np[:2] = (foot_l_pos_3d[:2] + foot_r_pos_3d[:2]) / 2.0
                sc_3d_np[2] = (foot_l_pos_3d[2] + foot_r_pos_3d[2]) / 2.0

                # Visualize the geometrical support center with a blue sphere
                mujoco.mjv_initGeom(v_handle.user_scn.geoms[0],
                                    mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0],
                                    sc_3d_np, geom_mat, rgba_blue)

                # Visualize the difficulty metric: draw a cylinder from the ground-projected CoM
                # to the ground-projected support center (purely 2D XY displacement)
                com_3d_np[:] = sim.data.subtree_com[idx, 0, :].cpu().numpy()
                com_3d_np[2] = sc_3d_np[2]  # project both endpoints to the foot plane
                mujoco.mjv_connector(v_handle.user_scn.geoms[1],
                                     mujoco.mjtGeom.mjGEOM_CYLINDER, 0.015,
                                     com_3d_np, sc_3d_np)
                v_handle.user_scn.geoms[1].rgba[:] = rgba_red

                v_handle.user_scn.ngeom = 2

                v_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0 if is_safe else 1
                v_handle.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1
                v_handle.sync()
                time.sleep(0.1)
                if not v_handle.is_running():
                    break

# ---------------------------------------------------------------------------
# OpenArm v2 bimanual — fixed-base, no legs/torso/waist. Kinematically incompatible
# with the GPU dual-arm cross-collision pipeline above (no shared torso for
# get_arm_joint_info's anchor traversal; "arm" keyword false-positives on every body
# since "openarm" contains "arm" as a substring; finger-finger contact is an expected
# closed-gripper state, not the cross-*left/right*-arm exception the RL/single-arm-safe
# split models). Kept as its own CPU rejection-sampling helper (ported from the former
# standalone generate_openarm_poses.py) rather than forced through the shared loop.
# ---------------------------------------------------------------------------

_OPENARM_XML = Path(__file__).resolve().parents[2] / "asset" / "openarm_v2" / "openarm_bimanual.xml"
_OPENARM_LEFT_JOINTS  = [f"openarm_left_joint{i}"  for i in range(1, 8)]
_OPENARM_RIGHT_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
_OPENARM_LEFT_EE_SITE  = "left_fingertip"
_OPENARM_RIGHT_EE_SITE = "right_fingertip"


def _rot_matrix_to_quat_np(mat) -> "np.ndarray":
    """3x3 rotation matrix -> wxyz quaternion, numpy (N,3,3) -> (N,4) or (3,3) -> (4,)."""
    import numpy as np
    if mat.ndim == 2:
        mat = mat[None]
    N = mat.shape[0]
    q = np.empty((N, 4), dtype=np.float64)
    for i in range(N):
        m = mat[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            q[i] = [0.25 / s, (m[2, 1] - m[1, 2]) * s,
                    (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                    (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                    0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q.squeeze(0) if N == 1 else q


def _generate_openarm_poses(samples: int, batch: int, seed: int, out_suffix: str, force: bool) -> None:
    """CPU rejection-sampling generator for OpenArm v2 (world-frame EE poses; fixed-base
    so no pelvis/support-center concepts apply). Accepts a config iff no non-finger geom
    is in contact (closed-gripper finger-finger contact is expected, not rejected).
    """
    import numpy as np

    out_path = Path(__file__).parent / "cache" / f"safe_arm_poses_openarm_v2{out_suffix}.pt"
    if out_path.exists() and not force:
        cache = torch.load(str(out_path), map_location="cpu", weights_only=False)
        if cache["ee_l_pos"].shape[0] >= samples:
            print(f"Cache exists with {cache['ee_l_pos'].shape[0]:,} poses — skipping. Use --force to regen.")
            return

    model = mujoco.MjModel.from_xml_path(str(_OPENARM_XML))
    data = mujoco.MjData(model)

    def _joint_info(names):
        addrs, lows, highs = [], [], []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint not found: {name}")
            adr = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            margin = (hi - lo) * 0.05
            addrs.append(adr); lows.append(lo + margin); highs.append(hi - margin)
        return np.array(addrs, int), np.array(lows), np.array(highs)

    l_addrs, l_lo, l_hi = _joint_info(_OPENARM_LEFT_JOINTS)
    r_addrs, r_lo, r_hi = _joint_info(_OPENARM_RIGHT_JOINTS)
    all_addrs = np.concatenate([l_addrs, r_addrs])
    all_lo    = np.concatenate([l_lo, r_lo])
    all_hi    = np.concatenate([l_hi, r_hi])
    n_dof = len(all_addrs)

    ee_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _OPENARM_LEFT_EE_SITE)
    ee_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _OPENARM_RIGHT_EE_SITE)
    if ee_l_id < 0 or ee_r_id < 0:
        raise ValueError(f"EE sites not found: {_OPENARM_LEFT_EE_SITE}, {_OPENARM_RIGHT_EE_SITE}")

    # Finger-finger contact is expected (closed gripper) — exclude from rejection.
    finger_geom_ids: set[int] = set()
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "finger" in name.lower():
            finger_geom_ids.add(i)

    rng = np.random.default_rng(seed)

    ee_l_pos_buf  = np.empty((samples, 3), np.float32)
    ee_l_quat_buf = np.empty((samples, 4), np.float32)
    ee_r_pos_buf  = np.empty((samples, 3), np.float32)
    ee_r_quat_buf = np.empty((samples, 4), np.float32)

    collected = 0
    total_sampled = 0
    t0 = time.time()

    print(f"Generating {samples:,} collision-free OpenArm v2 poses (batch={batch}) ...")
    print(f"DOFs: {n_dof} ({len(_OPENARM_LEFT_JOINTS)} left + {len(_OPENARM_RIGHT_JOINTS)} right)")
    print(f"Ignoring {len(finger_geom_ids)} finger geoms from collision check (closed-gripper contacts expected)")

    q_base = data.qpos.copy()  # default qpos (fingers stay at default — closed)

    while collected < samples:
        q_batch = rng.uniform(all_lo, all_hi, size=(batch, n_dof)).astype(np.float32)
        total_sampled += batch

        for i in range(batch):
            if collected >= samples:
                break
            q = q_base.copy()
            q[all_addrs] = q_batch[i]
            data.qpos[:] = q
            mujoco.mj_forward(model, data)

            has_arm_collision = any(
                data.contact[c].geom1 not in finger_geom_ids or
                data.contact[c].geom2 not in finger_geom_ids
                for c in range(data.ncon)
            )
            if has_arm_collision:
                continue

            ee_l_pos_buf[collected]  = data.site_xpos[ee_l_id].astype(np.float32)
            mat_l = data.site_xmat[ee_l_id].reshape(3, 3)
            ee_l_quat_buf[collected] = _rot_matrix_to_quat_np(mat_l).astype(np.float32)

            ee_r_pos_buf[collected]  = data.site_xpos[ee_r_id].astype(np.float32)
            mat_r = data.site_xmat[ee_r_id].reshape(3, 3)
            ee_r_quat_buf[collected] = _rot_matrix_to_quat_np(mat_r).astype(np.float32)

            collected += 1

        elapsed = time.time() - t0
        rate    = collected / max(elapsed, 1e-6)
        yield_pct = 100.0 * collected / max(total_sampled, 1)
        print(f"  {collected:>8,}/{samples:,}  "
              f"yield={yield_pct:.1f}%  "
              f"rate={rate:.0f} poses/s  "
              f"eta={max(0, (samples - collected) / max(rate, 1)):.0f}s",
              end="\r")

    print(f"\nDone. Collected {collected:,} poses from {total_sampled:,} samples "
          f"(yield={100.*collected/total_sampled:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        ee_l_pos  = torch.from_numpy(ee_l_pos_buf),
        ee_l_quat = torch.from_numpy(ee_l_quat_buf),
        ee_r_pos  = torch.from_numpy(ee_r_pos_buf),
        ee_r_quat = torch.from_numpy(ee_r_quat_buf),
    ), str(out_path))
    print(f"Saved -> {out_path}")


def generate_safe_arm_poses(
    robot: Literal["unitree_g1", "humanoid_v21", "openarm_v2"] = "unitree_g1",
    samples: int = 2097152,
    batch: int = 32768,
    device: str = "cuda:0",
    include_waist: bool = False,
    vis: bool = False,
    force: bool = False,
    rebuild_fk: bool = False,
    link_deltas: dict | None = None,
    out_suffix: str = "",
    realistic_collision: bool = False,
    manipulation: bool = False,
    quick_vis: bool = False,
    seed: int = 42,
) -> None:
    """
    Main generator for collision-free arm poses.

    Args:
        robot: The robot model to generate poses for ("unitree_g1", "humanoid_v21", or
                    "openarm_v2"). openarm_v2 dispatches to a separate CPU rejection-sampling
                    path (_generate_openarm_poses) and ignores manipulation/quick_vis/vis/
                    include_waist/rebuild_fk/link_deltas/realistic_collision.
        samples: The total number of safe poses to generate. If the cache contains more, it's reused.
        batch: The GPU simulation batch size.
        device: The computing device.
        include_waist: If True, includes waist joints. NOTE: Changing this requires --force to refresh cache
                      as the model hash won't change but the dimensionality of stored poses will.
        vis: If True, visualizes random poses interactively instead of saving.
        force: If True, ignores existing cache and regenerates poses.
        rebuild_fk: If True, augments an existing cache with FK-computed EE poses without
                    regenerating joint poses. Useful when the cache exists but lacks EE data.
        link_deltas: Optional {body_name: delta_m}. After compile, body_pos[id,1] -= delta
                    (local −y) to elongate structural bones for the arm-length sweep. Re-filters
                    self-collision at the new geometry. NOTE: the bone collision CAPSULE keeps
                    its original length, so the elongated bone's self-collision is slightly
                    under-modeled; inter-link / torso clearance is still captured.
        out_suffix: Appended to output cache filenames so per-config sweep caches don't
                    clobber the baseline (e.g. "_sweep_u10_f5"). Use with force=True.
        realistic_collision: If True, rebuild every arm collision capsule as a full
                    joint-to-joint shaft (midpoint pos, half-length = |child_offset|/2,
                    quat aligning local +z to the bone) instead of the asset's near-zero
                    pucks at joints. Applied AFTER link_deltas so lengthened bones get
                    correspondingly longer capsules. Re-baselines self-collision honestly
                    (humanoid_v21 only).
        manipulation: unitree_g1 only. If True, use this project's grafted welded
                    end-effector + parallel_gripper hand + actuated head camera cfg
                    (mj_envs.asset_zoo.g1.g1_constants.get_g1_robot_cfg) instead of mjlab's
                    stock bare-wrist G1 -- the actual geometry G1RmaVelEstArmFlashSacL2T/
                    ActuatedCam train against. Only quick_vis=True is supported for this cfg:
                    it has no static XML (built via spec_fn) so the mjcf_hash cache-validity
                    check doesn't apply, and its post-graft EE sites aren't named
                    left_palm/right_palm so _get_ee_frame_ids doesn't resolve.
        quick_vis: If True, skip the cache/generation/EE-frame pipeline entirely and launch
                    a live viewer cycling true-random arm poses (no cache read or write).
                    The only way to visualize the manipulation=True cfg.
        seed: RNG seed, openarm_v2 only.
    """
    if robot == "openarm_v2":
        _generate_openarm_poses(samples=samples, batch=batch, seed=seed, out_suffix=out_suffix, force=force)
        return

    if robot == "unitree_g1":
        if manipulation:
            if not quick_vis:
                raise ValueError(
                    "manipulation=True only supports quick_vis=True: full pose generation/caching "
                    "against the manipulation-config G1 isn't wired up (mjcf_hash needs a static XML, "
                    "which this spec_fn-built cfg doesn't have; _get_ee_frame_ids' left_palm/right_palm "
                    "sites don't exist post-graft). Use --quick-vis for a live random-pose sanity check, "
                    "or build_arm_collision_graph.py for the actual collision-free arm graph."
                )
            from mj_envs.asset_zoo.g1.g1_constants import get_g1_robot_cfg
            cfg = get_g1_robot_cfg(end_effector="welded", hand="parallel_gripper", head_camera="actuated")
            xml_path = None
        else:
            from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg, G1_XML
            cfg = get_g1_robot_cfg()
            xml_path = G1_XML
    elif robot == "humanoid_v21":
        from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_humanoid_v21_robot_cfg, HUMANOID_V21_XML
        cfg = get_humanoid_v21_robot_cfg()
        xml_path = HUMANOID_V21_XML
    else:
        raise ValueError(f"Unsupported robot specified: {robot}")

    entity = Entity(cfg)
    if link_deltas or realistic_collision:
        # Geometry edits MUST happen in the spec, before compile, so collision broadphase
        # bounds are recomputed (editing the compiled model's geom_size leaves rbound stale
        # and silently drops collisions).
        _apply_spec_geometry(entity.spec, link_deltas or {}, realistic_collision, robot)
        msg = []
        if link_deltas:
            msg.append(f"link_deltas={link_deltas}")
        if realistic_collision:
            msg.append("realistic capsules (joint-to-joint shafts)")
        print(f"\033[93m[generate_safe_arm_poses] Spec geometry: {'; '.join(msg)}\033[0m")
    model = entity.spec.compile()
    mjcf_hash = xxhash.xxh64(xml_path.read_bytes()).hexdigest() if xml_path is not None else None

    output_path = Path(__file__).parent / "cache" / f"safe_arm_poses_{robot}{out_suffix}.pt"
    output_path_single = Path(__file__).parent / "cache" / f"safe_arm_poses_{robot}{out_suffix}_single_arm_safe.pt"

    qpos_indices, arm_names, lower_bounds, upper_bounds = get_arm_joint_info(model, device, include_waist)

    print(f"\033[92m[generate_safe_arm_poses] Arm names: {arm_names}\033[0m")

    if robot == "unitree_g1":
        foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    else:  # humanoid_v21
        foot_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_L")
        foot_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot_R")

    if quick_vis:
        # Bypass cache/EE-frame/generation entirely -- true-random live pose cycling.
        # Placed before _get_ee_frame_ids since manipulation cfg's post-graft EE sites
        # aren't named left_palm/right_palm (see manipulation docstring above).
        visualize_poses(model, qpos_indices, lower_bounds, upper_bounds, device,
                        min(batch, 32), foot_l_id, foot_r_id)
        return

    # Split arm joints into left and right arm joints
    left_joint_mask = torch.tensor([name.startswith("left_") for name in arm_names], device=device, dtype=torch.bool)
    right_joint_mask = torch.tensor([name.startswith("right_") for name in arm_names], device=device, dtype=torch.bool)
    print(f"[generate_safe_arm_poses] Left arm joints: {[arm_names[i] for i in range(len(arm_names)) if left_joint_mask[i]]}")
    print(f"[generate_safe_arm_poses] Right arm joints: {[arm_names[i] for i in range(len(arm_names)) if right_joint_mask[i]]}")

    # Trace left and right arm bodies for independent collision tracking
    child_counts = [0] * model.nbody
    for i in range(model.nbody):
        p = model.body_parentid[i]
        if p >= 0:
            child_counts[p] += 1
    torso_body = max(range(1, model.nbody), key=lambda i: child_counts[i])

    children = [[] for _ in range(model.nbody)]
    for b in range(1, model.nbody):
        p = model.body_parentid[b]
        if p >= 0:
            children[p].append(b)

    arm_anchor_keywords = ["shoulder", "elbow", "arm"]
    left_arm_bodies = set()
    right_arm_bodies = set()
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and any(k in name.lower() for k in arm_anchor_keywords):
            curr = i
            while curr != torso_body and curr > 0:
                curr = model.body_parentid[curr]
            if curr != torso_body:
                continue
            
            is_left = "left" in name.lower() or name.endswith("_L") or "_l_" in name.lower()
            is_right = "right" in name.lower() or name.endswith("_R") or "_r_" in name.lower()
            
            stack = [i]
            while stack:
                b = stack.pop()
                if is_left:
                    left_arm_bodies.add(b)
                elif is_right:
                    right_arm_bodies.add(b)
                stack.extend(children[b])

    pelvis_id, ee_l_id, ee_r_id, ee_is_site = _get_ee_frame_ids(model, robot)
    print(f"\033[92m[generate_safe_arm_poses] EE frame ids: pelvis={pelvis_id}, ee_l={ee_l_id}, ee_r={ee_r_id}, is_site={ee_is_site}\033[0m")

    # ------------------------------------------------------------------
    # Cache check — determine what work is needed
    # ------------------------------------------------------------------
    needs_generation = True
    needs_fk_rebuild = False
    poses_for_vis    = None

    if not force and output_path.exists() and output_path_single.exists():
        cache_rl = torch.load(str(output_path), map_location="cpu", weights_only=False)
        cache_single = torch.load(str(output_path_single), map_location="cpu", weights_only=False)
        has_ee = "ee_l_pos" in cache_rl and "ee_r_pos" in cache_rl and "ee_l_pos" in cache_single and "ee_r_pos" in cache_single
        is_valid = (
            cache_rl.get("poses") is not None
            and cache_rl.get("poses").shape[1] == len(qpos_indices)
            and "difficulties" in cache_rl
            and cache_single.get("poses") is not None
            and cache_single.get("poses").shape[1] == len(qpos_indices)
            and "difficulties" in cache_single
        )
        if is_valid:
            if cache_rl.get("mjcf_hash") != mjcf_hash:
                print(f"Warning: Cache MJCF hash mismatch. Using existing cache anyway. Use --force to regenerate.")
            if cache_rl.get("poses").shape[0] < samples or cache_single.get("poses").shape[0] < samples:
                print(f"Warning: Cache has fewer poses than {samples} requested. Use --force to regenerate.")
            
            if has_ee and not rebuild_fk:
                print("Valid caches found (with EE poses).")
                needs_generation = False
                if vis:
                    poses_for_vis = cache_rl["poses"].to(device)
            else:
                print("Cache found — EE poses missing or rebuild requested. Running FK pass...")
                needs_generation = False
                needs_fk_rebuild = True
        else:
            print("Caches invalid or shape mismatch. Regenerating...")

    if not needs_generation and not needs_fk_rebuild:
        if vis and poses_for_vis is not None:
            visualize_poses(model, qpos_indices, lower_bounds, upper_bounds, device,
                            min(batch, 32), foot_l_id, foot_r_id, poses_for_vis)
        return

    # ------------------------------------------------------------------
    # FK rebuild: augment existing caches with EE poses (no regeneration)
    # ------------------------------------------------------------------
    if needs_fk_rebuild:
        for path in (output_path, output_path_single):
            if not path.exists():
                continue
            cache = torch.load(str(path), map_location="cpu", weights_only=False)
            poses_cpu = cache["poses"]
            n_poses   = poses_cpu.shape[0]

            sim_cfg = SimulationCfg(nconmax=128, njmax=384)  # bumped from 32/64: welded parallel_gripper capsule colliders raise ncon past 32 (overflowed at ncon=62)
            sim_cfg.mujoco.ccd_iterations = 0  # static snapshot — no motion, CCD unnecessary
            sim = Simulation(num_envs=batch, cfg=sim_cfg, model=model, device=device)
            sim.reset()

            # Capture FK-only CUDA graph
            wp.capture_begin(sim.wp_device)
            mjwarp._src.forward.fwd_position(sim.wp_model, sim.wp_data, factorize=False)
            fwd_pos_graph = wp.capture_end(sim.wp_device)

            ee_l_pos_all  = torch.empty((n_poses, 3), dtype=torch.float32)
            ee_l_quat_all = torch.empty((n_poses, 4), dtype=torch.float32)
            ee_r_pos_all  = torch.empty((n_poses, 3), dtype=torch.float32)
            ee_r_quat_all = torch.empty((n_poses, 4), dtype=torch.float32)

            print(f"Running FK on {n_poses} poses for {path.name} in chunks of {batch}...")
            start_time = time.time()
            for chunk_start in range(0, n_poses, batch):
                chunk_end  = min(chunk_start + batch, n_poses)
                chunk_size = chunk_end - chunk_start

                sim.reset()
                if model.nkey > 0:
                    key_qpos = torch.tensor(model.key_qpos[0], device=device, dtype=torch.float32)
                    sim.data.qpos[:chunk_size] = key_qpos.unsqueeze(0)
                sim.data.qpos[:chunk_size, qpos_indices] = poses_cpu[chunk_start:chunk_end].to(device)
                with wp.ScopedDevice(device):
                    wp.capture_launch(fwd_pos_graph)

                l_pos, l_quat, r_pos, r_quat = _extract_ee_in_pelvis(
                    sim.data.xpos[:chunk_size],
                    sim.data.xquat[:chunk_size],
                    pelvis_id, ee_l_id, ee_r_id,
                    site_xpos=sim.data.site_xpos[:chunk_size] if ee_is_site else None,
                    site_xmat=sim.data.site_xmat[:chunk_size] if ee_is_site else None,
                )
                ee_l_pos_all[chunk_start:chunk_end]  = l_pos.cpu()
                ee_l_quat_all[chunk_start:chunk_end] = l_quat.cpu()
                ee_r_pos_all[chunk_start:chunk_end]  = r_pos.cpu()
                ee_r_quat_all[chunk_start:chunk_end] = r_quat.cpu()

                if (chunk_start // batch) % 50 == 0 or chunk_end == n_poses:
                    print(f"  FK: {chunk_end}/{n_poses} ({chunk_end/n_poses*100:.1f}%) | "
                          f"{time.time()-start_time:.1f}s")

            cache["ee_l_pos"]  = ee_l_pos_all
            cache["ee_l_quat"] = ee_l_quat_all
            cache["ee_r_pos"]  = ee_r_pos_all
            cache["ee_r_quat"] = ee_r_quat_all
            torch.save(cache, str(path))
            print(f"FK pass complete. Cache augmented at {path}")

        if vis:
            visualize_poses(model, qpos_indices, lower_bounds, upper_bounds, device,
                            min(batch, 32), foot_l_id, foot_r_id, poses_cpu.to(device))
        return

    # ------------------------------------------------------------------
    # Generation (caches missing or force=True)
    # ------------------------------------------------------------------
    if needs_generation:
        sim_cfg = SimulationCfg(nconmax=128, njmax=384)  # bumped from 32/64: welded parallel_gripper capsule colliders raise ncon past 32 (overflowed at ncon=62)
        sim_cfg.mujoco.ccd_iterations = 0  # static snapshot — no motion, CCD unnecessary
        sim = Simulation(num_envs=batch, cfg=sim_cfg, model=model, device=device)
        sim.reset()
        if model.nkey > 0:
            key_qpos = torch.tensor(model.key_qpos[0], device=device, dtype=torch.float32)
            sim.data.qpos[:] = key_qpos.unsqueeze(0)
        sim.forward()

        # Capture FK-only CUDA graph (no dynamics, no collision solve) for speed
        wp.capture_begin(sim.wp_device)
        mjwarp._src.forward.fwd_position(sim.wp_model, sim.wp_data, factorize=False)
        fwd_pos_graph = wp.capture_end(sim.wp_device)

        # Pre-compute geom arm labels on GPU
        geom_arm_label = torch.zeros(model.ngeom, dtype=torch.long, device=device)
        geom_bodyid = torch.tensor(model.geom_bodyid, device=device)
        for g in range(model.ngeom):
            b = model.geom_bodyid[g]
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if b in left_arm_bodies:
                if "shoulder" in b_name.lower():
                    geom_arm_label[g] = 0  # shoulder is treated as body
                else:
                    geom_arm_label[g] = 1  # left arm links (moving)
            elif b in right_arm_bodies:
                if "shoulder" in b_name.lower():
                    geom_arm_label[g] = 0  # shoulder is treated as body
                else:
                    geom_arm_label[g] = 2  # right arm links (moving)

        num_left_joints = left_joint_mask.sum().item()
        num_right_joints = right_joint_mask.sum().item()
        candidates_L = []
        candidates_R = []

        verified_safe_poses_rl = torch.empty((samples, len(qpos_indices)), device=device, dtype=torch.float32)
        verified_safe_poses_single = torch.empty((samples, len(qpos_indices)), device=device, dtype=torch.float32)
        total_verified_rl = 0
        total_verified_single = 0

        random_noise = torch.empty((batch, len(qpos_indices)), device=device, dtype=torch.float32)
        env_l_collision = torch.zeros(batch, dtype=torch.bool, device=device)
        env_r_collision = torch.zeros(batch, dtype=torch.bool, device=device)

        attempts = 0
        start_time = time.time()
        last_print = start_time

        print(f"Generating safe arm poses in loops on {device}...")

        while total_verified_rl < samples:
            random_noise.uniform_()
            random_arm_poses = lower_bounds + random_noise * (upper_bounds - lower_bounds)
            sim.data.qpos[:, qpos_indices] = random_arm_poses

            torch.cuda.synchronize()
            with wp.ScopedDevice(device):
                wp.capture_launch(fwd_pos_graph)
            torch.cuda.synchronize()

            # Get contacts that are self-collisions
            nacon = sim.data.nacon[0].item()
            is_collision_contact = (sim.data.contact.dist[:nacon] < 0.0) & (sim.data.contact.worldid[:nacon] >= 0)
            geom1 = sim.data.contact.geom[:nacon, 0][is_collision_contact]
            geom2 = sim.data.contact.geom[:nacon, 1][is_collision_contact]
            is_self_collision = (geom_bodyid[geom1] > 0) & (geom_bodyid[geom2] > 0)

            valid_worldids = sim.data.contact.worldid[:nacon][is_collision_contact][is_self_collision]
            valid_mask = valid_worldids < batch

            c_geom1 = geom1[is_self_collision][valid_mask]
            c_geom2 = geom2[is_self_collision][valid_mask]
            c_worldids = valid_worldids[valid_mask]

            c_l1 = geom_arm_label[c_geom1]
            c_l2 = geom_arm_label[c_geom2]

            # Stage 1 collision logic: ignore cross-arm moving link collisions {1, 2} to avoid center bias
            is_l_collision = ((c_l1 == 1) & (c_l2 != 2)) | ((c_l2 == 1) & (c_l1 != 2))
            is_r_collision = ((c_l1 == 2) & (c_l2 != 1)) | ((c_l2 == 2) & (c_l1 != 1))

            env_l_collision.zero_()
            env_r_collision.zero_()
            env_l_collision[c_worldids[is_l_collision]] = True
            env_r_collision[c_worldids[is_r_collision]] = True

            safe_l_mask = ~env_l_collision
            safe_r_mask = ~env_r_collision

            num_safe_L = safe_l_mask.sum().item()
            num_safe_R = safe_r_mask.sum().item()

            if num_safe_L > 0:
                safe_indices_L = safe_l_mask.nonzero(as_tuple=True)[0]
                candidates_L.append(random_arm_poses[safe_indices_L][:, left_joint_mask].cpu())
            if num_safe_R > 0:
                safe_indices_R = safe_r_mask.nonzero(as_tuple=True)[0]
                candidates_R.append(random_arm_poses[safe_indices_R][:, right_joint_mask].cpu())

            attempts += batch

            # Check candidate sizes
            total_cand_L = sum(c.shape[0] for c in candidates_L) if candidates_L else 0
            total_cand_R = sum(c.shape[0] for c in candidates_R) if candidates_R else 0

            # Run Stage 2 validation in chunks of size validation_batch
            validation_batch = min(batch, 32768)
            if total_cand_L >= validation_batch and total_cand_R >= validation_batch:
                cat_L = torch.cat(candidates_L, dim=0)
                cat_R = torch.cat(candidates_R, dim=0)

                chunk_L = cat_L[:validation_batch].to(device)
                chunk_R = cat_R[:validation_batch].to(device)

                candidates_L = [cat_L[validation_batch:]] if cat_L.shape[0] > validation_batch else []
                candidates_R = [cat_R[validation_batch:]] if cat_R.shape[0] > validation_batch else []

                shuffled_idx_L = torch.randperm(validation_batch, device=device)
                shuffled_idx_R = torch.randperm(validation_batch, device=device)

                paired_poses = torch.empty((validation_batch, len(qpos_indices)), device=device, dtype=torch.float32)
                paired_poses[:, left_joint_mask] = chunk_L[shuffled_idx_L]
                paired_poses[:, right_joint_mask] = chunk_R[shuffled_idx_R]

                sim.data.qpos[:validation_batch, qpos_indices] = paired_poses
                torch.cuda.synchronize()
                with wp.ScopedDevice(device):
                    wp.capture_launch(fwd_pos_graph)
                torch.cuda.synchronize()

                nacon_val = sim.data.nacon[0].item()
                is_collision_contact_val = (sim.data.contact.dist[:nacon_val] < 0.0) & (sim.data.contact.worldid[:nacon_val] >= 0)
                geom1_val = sim.data.contact.geom[:nacon_val, 0][is_collision_contact_val]
                geom2_val = sim.data.contact.geom[:nacon_val, 1][is_collision_contact_val]
                is_self_collision_val = (geom_bodyid[geom1_val] > 0) & (geom_bodyid[geom2_val] > 0)

                valid_worldids_val = sim.data.contact.worldid[:nacon_val][is_collision_contact_val][is_self_collision_val]
                valid_mask_val = valid_worldids_val < validation_batch
                
                # Check cross-arm moving link collisions
                val_g1 = geom1_val[is_self_collision_val][valid_mask_val]
                val_g2 = geom2_val[is_self_collision_val][valid_mask_val]
                val_worldids = valid_worldids_val[valid_mask_val]

                c_l1_val = geom_arm_label[val_g1]
                c_l2_val = geom_arm_label[val_g2]
                is_cross_arm = ((c_l1_val == 1) & (c_l2_val == 2)) | ((c_l1_val == 2) & (c_l2_val == 1))

                # RL collision (any self-collision)
                env_collision_rl = torch.zeros(validation_batch, dtype=torch.bool, device=device)
                env_collision_rl[val_worldids] = True

                # Reachability collision (ignore 1 <-> 2 cross-arm moving collisions)
                env_collision_reach = torch.zeros(validation_batch, dtype=torch.bool, device=device)
                env_collision_reach[val_worldids[~is_cross_arm]] = True

                safe_mask_rl = ~env_collision_rl
                # single-arm-safe is a supplement where only single arm is safe (i.e. cross-arm collision exists)
                safe_mask_single = (~env_collision_reach) & env_collision_rl

                num_safe_rl = safe_mask_rl.sum().item()
                if num_safe_rl > 0 and total_verified_rl < samples:
                    copy_cnt = min(num_safe_rl, samples - total_verified_rl)
                    safe_indices = safe_mask_rl.nonzero(as_tuple=True)[0][:copy_cnt]
                    verified_safe_poses_rl[total_verified_rl : total_verified_rl + copy_cnt] = paired_poses[safe_indices]
                    total_verified_rl += copy_cnt

                num_safe_single = safe_mask_single.sum().item()
                if num_safe_single > 0 and total_verified_single < samples:
                    copy_cnt = min(num_safe_single, samples - total_verified_single)
                    safe_indices = safe_mask_single.nonzero(as_tuple=True)[0][:copy_cnt]
                    verified_safe_poses_single[total_verified_single : total_verified_single + copy_cnt] = paired_poses[safe_indices]
                    total_verified_single += copy_cnt

            curr_time = time.time()
            if curr_time - last_print > 0.5 or (total_verified_rl >= samples and total_verified_single >= samples):
                print(f"Verified RL: {total_verified_rl:<7}/{samples:<7} | "
                      f"Single-Arm-Safe: {total_verified_single:<7}/{samples:<7} | "
                      f"Candidates L: {total_cand_L:<6} | R: {total_cand_R:<6} | "
                      f"Time: {curr_time - start_time:.2f}s")
                last_print = curr_time

        # Compute EE poses and difficulties for both datasets
        print("Computing final kinematics and end-effector coordinates for both datasets...")
        
        def process_and_save_dataset(verified_safe_poses, is_single_arm_safe):
            all_safe_poses       = verified_safe_poses
            num_actual           = all_safe_poses.shape[0]
            all_safe_com_xy      = torch.empty((num_actual, 2), device=device, dtype=torch.float32)
            all_support_centers_xy = torch.empty((num_actual, 2), device=device, dtype=torch.float32)
            all_ee_l_pos         = torch.empty((num_actual, 3), device=device, dtype=torch.float32)
            all_ee_l_quat        = torch.empty((num_actual, 4), device=device, dtype=torch.float32)
            all_ee_r_pos         = torch.empty((num_actual, 3), device=device, dtype=torch.float32)
            all_ee_r_quat        = torch.empty((num_actual, 4), device=device, dtype=torch.float32)

            for chunk_start in range(0, num_actual, validation_batch):
                chunk_end = min(chunk_start + validation_batch, num_actual)
                chunk_size = chunk_end - chunk_start

                # Pad chunk to full batch size to avoid shape wrapper mismatches
                sim_batch_size = sim.data.qpos.shape[0]
                qpos_chunk = torch.zeros((sim_batch_size, sim.data.qpos.shape[1]), device=device, dtype=torch.float32)
                if model.nkey > 0:
                    key_qpos = torch.tensor(model.key_qpos[0], device=device, dtype=torch.float32)
                    qpos_chunk[:] = key_qpos.unsqueeze(0)
                qpos_chunk[:chunk_size, qpos_indices] = all_safe_poses[chunk_start:chunk_end]
                sim.data.qpos[:] = qpos_chunk

                torch.cuda.synchronize()
                with wp.ScopedDevice(device):
                    wp.capture_launch(fwd_pos_graph)
                torch.cuda.synchronize()

                foot_l_pos = sim.data.xpos[:chunk_size, foot_l_id, :2]
                foot_r_pos = sim.data.xpos[:chunk_size, foot_r_id, :2]

                l_pos, l_quat, r_pos, r_quat = _extract_ee_in_pelvis(
                    sim.data.xpos[:chunk_size],
                    sim.data.xquat[:chunk_size],
                    pelvis_id, ee_l_id, ee_r_id,
                    site_xpos=sim.data.site_xpos[:chunk_size] if ee_is_site else None,
                    site_xmat=sim.data.site_xmat[:chunk_size] if ee_is_site else None,
                )

                all_safe_com_xy[chunk_start:chunk_end]        = sim.data.subtree_com[:chunk_size, 0, :2]
                all_support_centers_xy[chunk_start:chunk_end] = (foot_l_pos + foot_r_pos) / 2.0
                all_ee_l_pos[chunk_start:chunk_end]           = l_pos
                all_ee_l_quat[chunk_start:chunk_end]          = l_quat
                all_ee_r_pos[chunk_start:chunk_end]           = r_pos
                all_ee_r_quat[chunk_start:chunk_end]          = r_quat

            all_difficulties = torch.norm(all_safe_com_xy - all_support_centers_xy, dim=-1)

            # Sort by L2 from default pose so index 0 = nearest to rest.
            data_default = mujoco.MjData(model)
            if model.nkey > 0:
                mujoco.mj_resetDataKeyframe(model, data_default, 0)
            q_default_arm = torch.tensor(
                data_default.qpos[qpos_indices.cpu().numpy()], dtype=torch.float32, device=device
            )
            sort_idx = (all_safe_poses - q_default_arm).norm(dim=1).argsort()
            all_safe_poses          = all_safe_poses[sort_idx]
            all_safe_com_xy         = all_safe_com_xy[sort_idx]
            all_support_centers_xy  = all_support_centers_xy[sort_idx]
            all_ee_l_pos            = all_ee_l_pos[sort_idx]
            all_ee_l_quat           = all_ee_l_quat[sort_idx]
            all_ee_r_pos            = all_ee_r_pos[sort_idx]
            all_ee_r_quat           = all_ee_r_quat[sort_idx]
            all_difficulties        = torch.norm(all_safe_com_xy - all_support_centers_xy, dim=-1)

            suffix = "_single_arm_safe" if is_single_arm_safe else ""
            out_path = Path(__file__).parent / "cache" / f"safe_arm_poses_{robot}{out_suffix}{suffix}.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "poses":                    all_safe_poses.cpu(),
                "joint_names":              arm_names,
                "qpos_indices":             qpos_indices.cpu(),
                "difficulties":             all_difficulties.cpu(),
                "ee_l_pos":                 all_ee_l_pos.cpu(),
                "ee_l_quat":                all_ee_l_quat.cpu(),
                "ee_r_pos":                 all_ee_r_pos.cpu(),
                "ee_r_quat":                all_ee_r_quat.cpu(),
                "mjcf_hash":                mjcf_hash,
                "mjcf_path":                str(xml_path),
                "sorted_by_l2_from_default": True,
                "gen_attempts":             attempts,
                "gen_yield":                total_verified_rl / max(attempts, 1),
            }, str(out_path))
            print(f"Success! Saved to {out_path}")

        process_and_save_dataset(verified_safe_poses_rl, is_single_arm_safe=False)
        process_and_save_dataset(verified_safe_poses_single[:total_verified_single], is_single_arm_safe=True)

    if vis:
        visualize_poses(model, qpos_indices, lower_bounds, upper_bounds, device, min(batch, 32), foot_l_id, foot_r_id, verified_safe_poses_rl)

if __name__ == "__main__":
    tyro.cli(generate_safe_arm_poses)
