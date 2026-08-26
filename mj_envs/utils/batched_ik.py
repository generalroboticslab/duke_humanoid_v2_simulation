"""Batched DLS inverse kinematics for two-armed humanoids.

Solves EE position + orientation targets for left and right arms simultaneously
across N parallel environments. Supports GPU path (MuJoCo Warp) and
CPU fallback (sequential mujoco.mj_kinematics).

Key features:
  - Warm-started iterative DLS with Tikhonov regularisation and posture spring.
  - Joint-space leaky bias integrator to cancel steady-state PD gravity sag.
  - Optional soft joint-limit barrier (cuRobo BOUNDS_SMOOTH style).
  - Optional self-collision repulsion via contact-normal Jacobian.
  - Early-exit convergence check to skip Cholesky solve when already converged.

See BatchedAnalyticalIK for full design documentation.
"""

import os
import re
import numpy as np
import torch
import warp as wp
import mujoco
import mujoco_warp as mjwarp
from typing import List, Literal, Optional

from .torch_math_utils import (
    _quat_conjugate, _quat_apply, _quat_multiply,
    _axis_angle_to_quat, _quat_to_axis_angle, _quat_to_rot_matrix,
    _jlog, _colldist_from_sdf, _capsule_seg_seg_dist,
    _rot_matrix_to_quat,
)

EEFrameType = Literal["body", "site"]
_MJ_FRAME_TYPES: dict[str, mujoco.mjtObj] = {
    "body": mujoco.mjtObj.mjOBJ_BODY,
    "site": mujoco.mjtObj.mjOBJ_SITE,
}


# ---------------------------------------------------------------------------
# MuJoCo Warp Math Utilities (wxyz convention)
# ---------------------------------------------------------------------------

@wp.struct
class CapsuleDistResult:
    dist: float
    delta: wp.vec3
    c1: wp.vec3

@wp.func
def wp_quat_conjugate(q: wp.quat) -> wp.quat:
    return wp.quat(q[0], -q[1], -q[2], -q[3])

@wp.func
def wp_quat_apply(q: wp.quat, v: wp.vec3) -> wp.vec3:
    # Rodrigues formula: v' = v + 2w(q_vec x v) + 2(q_vec x (q_vec x v))
    qv = wp.vec3(q[1], q[2], q[3])
    t = wp.cross(qv, v) * 2.0
    return v + q[0] * t + wp.cross(qv, t)

@wp.func
def wp_seg_pt_closest(p: wp.vec3, q: wp.vec3, x: wp.vec3) -> wp.vec3:
    pq = q - p
    l2 = wp.dot(pq, pq)
    if l2 < 1e-8:
        return p
    t = wp.clamp(wp.dot(x - p, pq) / l2, 0.0, 1.0)
    return p + t * pq

@wp.func
def wp_capsule_seg_seg_dist(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3) -> CapsuleDistResult:
    """Compute the shortest distance between two line segments [p1, p2] and [p3, p4].
    
    Implements Shene's closest-point algorithm. Handles parallel segments by 
    falling back to the minimum of four point-to-segment projections.
    
    Returns:
        CapsuleDistResult containing distance, the delta vector (points away 
        from torso), and the point of application on segment 1.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    b = wp.dot(d1, d2)
    c = wp.dot(d1, r)
    denom = a * e - b * b
    
    s = 0.0
    t = 0.0
    if denom > 1e-8:
        # Non-parallel: standard Shene algorithm
        s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
        t = wp.clamp((b * s + f) / e, 0.0, 1.0)
        s = wp.clamp((b * t - c) / a, 0.0, 1.0)
    else:
        # Parallel: find min distance from endpoints to segments to avoid division by zero
        cl_a1_B = wp_seg_pt_closest(p3, p4, p1)
        cl_a2_B = wp_seg_pt_closest(p3, p4, p2)
        cl_b1_A = wp_seg_pt_closest(p1, p2, p3)
        cl_b2_A = wp_seg_pt_closest(p1, p2, p4)
        
        d_a1 = wp.length(p1 - cl_a1_B)
        d_a2 = wp.length(p2 - cl_a2_B)
        d_b1 = wp.length(cl_b1_A - p3)
        d_b2 = wp.length(cl_b2_A - p4)
        
        d_min = d_a1
        c1 = p1
        c2 = cl_a1_B
        if d_a2 < d_min:
            d_min = d_a2; c1 = p2; c2 = cl_a2_B
        if d_b1 < d_min:
            d_min = d_b1; c1 = cl_b1_A; c2 = p3
        if d_b2 < d_min:
            d_min = d_b2; c1 = cl_b2_A; c2 = p4
        
        res = CapsuleDistResult()
        res.dist = d_min
        res.delta = c1 - c2
        res.c1 = c1
        return res

    c1 = p1 + s * d1
    c2 = p3 + t * d2
    res = CapsuleDistResult()
    res.dist = wp.length(c1 - c2)
    res.delta = c1 - c2
    res.c1 = c1
    return res


@wp.kernel
def _fused_repulsion_kernel(
    # Model info:
    body_parentid: wp.array(dtype=int),
    body_rootid: wp.array(dtype=int),
    dof_bodyid: wp.array(dtype=int),
    # Capsule data:
    arm_body_ids: wp.array(dtype=int),
    arm_radii: wp.array(dtype=float),
    arm_local_centers: wp.array(dtype=wp.vec3),
    arm_local_axes: wp.array(dtype=wp.vec3),
    arm_half_lens: wp.array(dtype=float),
    torso_body_ids: wp.array(dtype=int),
    torso_radii: wp.array(dtype=float),
    torso_local_centers: wp.array(dtype=wp.vec3),
    torso_local_axes: wp.array(dtype=wp.vec3),
    torso_half_lens: wp.array(dtype=float),
    # Pair mask:
    pair_mask: wp.array2d(dtype=float),
    # DOF mapping:
    left_dof_addr: wp.array(dtype=int),
    right_dof_addr: wp.array(dtype=int),
    K: int,
    # Counts:
    num_arm: int,
    num_torso: int,
    # Constants:
    activation_dist: float,
    k_coll: float,
    # Data in (world frame from FK):
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array2d(dtype=wp.spatial_vector),
    # Out:
    dq_out: wp.array2d(dtype=float), # (N, 2K)
):
    """Fused self-collision repulsion kernel.
    
    Parallelized across (nworld, 2*K) threads. Each thread computes the net 
    repulsion torque for a single arm DOF by checking all capsule pairs.
    
    Physics:
      1. For each arm-torso capsule pair (i, j):
         a. Transform capsules to world frame using xpos/xquat.
         b. Compute segment-segment distance and closest point C1 on arm.
         c. If gap < activation_dist, compute repulsion force magnitude F (unit direction).
         d. Project F to joint torque: tau = dot(F, jac_p(C1))
            where jac_p(C1) is the analytic point-Jacobian at C1.
      2. Accumulate tau * k_coll into dq_out for the current DOF.
    """
    worldid, arm_dof_idx = wp.tid()
    
    # Map arm_dof_idx to model dofid
    dofid = 0
    if arm_dof_idx < K:
        dofid = left_dof_addr[arm_dof_idx]
    else:
        dofid = right_dof_addr[arm_dof_idx - K]
        
    dof_body_id = dof_bodyid[dofid]
    dof_root_id = body_rootid[dof_body_id]
        
    # Pre-fetch DOF axis (expressed at Subtree COM)
    cdof_val = cdof[worldid, dofid]
    cdof_ang = wp.vec3(cdof_val[0], cdof_val[1], cdof_val[2])
    cdof_lin = wp.vec3(cdof_val[3], cdof_val[4], cdof_val[5])
    
    # Subtree COM world position
    com_subtree = subtree_com[worldid, dof_root_id]
    
    total_qfrc = float(0.0)
    
    # Iterate through all arm capsules
    for i in range(num_arm):
        abid = arm_body_ids[i]
        
        # Check if this arm body is in the subtree of the current DOF
        curr = int(abid)
        is_descendant = int(0)
        while curr > 0:
            if curr == dof_body_id:
                is_descendant = 1
                curr = 0 # Break
            else:
                curr = body_parentid[curr]
            
        if is_descendant == 0:
            continue
            
        # 1. Transform arm capsule to world space
        a_body_pos  = xpos[worldid, abid]
        a_body_quat = xquat[worldid, abid]
        
        a_ctr_w = a_body_pos + wp_quat_apply(a_body_quat, arm_local_centers[i])
        a_ax_w  = wp_quat_apply(a_body_quat, arm_local_axes[i])
        a_p1    = a_ctr_w - a_ax_w * arm_half_lens[i]
        a_p2    = a_ctr_w + a_ax_w * arm_half_lens[i]
        a_rad   = arm_radii[i]
        
        # Step 2: Iterate through torso capsules to find pairwise repulsion
        for j in range(num_torso):
            if pair_mask[i, j] < 0.5:
                continue

            tbid = torso_body_ids[j]
            t_body_pos  = xpos[worldid, tbid]
            t_body_quat = xquat[worldid, tbid]
            
            t_ctr_w = t_body_pos + wp_quat_apply(t_body_quat, torso_local_centers[j])
            t_ax_w  = wp_quat_apply(t_body_quat, torso_local_axes[j])
            t_p1    = t_ctr_w - t_ax_w * torso_half_lens[j]
            t_p2    = t_ctr_w + t_ax_w * torso_half_lens[j]
            t_rad   = torso_radii[j]
            
            # Step 2a: Analytical segment-segment distance
            res = wp_capsule_seg_seg_dist(a_p1, a_p2, t_p1, t_p2)
            dist = res.dist
            delta = res.delta # Points away from torso (C1 - C2)
            
            gap = dist - (a_rad + t_rad)
            if gap < activation_dist:
                # Step 3: Compute repulsion force magnitude (pyroki smooth C1 cost)
                rep_mag = float(0.0)
                if gap < 0.0:
                    rep_mag = -(gap - 0.5 * activation_dist)
                else:
                    d_rel = gap - activation_dist
                    rep_mag = 0.5 / (activation_dist + 1e-6) * (d_rel * d_rel)
                
                # Step 3a: World-frame force vector
                force_w = wp.vec3(0.0, 0.0, 0.0)
                if dist > 1e-6:
                    force_w = delta * (rep_mag / dist)
                else:
                    d_body = a_body_pos - t_body_pos
                    if wp.length(d_body) > 1e-6:
                        force_w = wp.normalize(d_body) * rep_mag
                    else:
                        force_w = wp.vec3(0.0, 0.0, rep_mag)
                
                # Step 4: Project to joint space (analytic point Jacobian)
                # Formula: J_p(C1) = cdof_lin + cross(cdof_ang, C1 - SubtreeCOM)
                offset = res.c1 - com_subtree
                jac_p_c1 = cdof_lin + wp.cross(cdof_ang, offset)
                total_qfrc = total_qfrc + wp.dot(force_w, jac_p_c1)

    # Step 5: Accumulate into result buffer (dq += gain * qfrc)
    dq_out[worldid, arm_dof_idx] += k_coll * total_qfrc


# ---------------------------------------------------------------------------
# MuJoCo Warp DLS Math Kernels
# ---------------------------------------------------------------------------

@wp.func
def wp_cholesky_factor(A: wp.array2d(dtype=float), L: wp.array2d(dtype=float), K: int, env_id: int) -> int:
    """Computes L such that L*L^T = A for a KxK positive definite matrix.
    
    Matrices are stored flattened in (N, K*K) arrays to allow dynamic sizing.
    Uses standard Cholesky-Banachiewicz algorithm.
    
    Args:
        A: (N, K*K) flattened system matrices.
        L: (N, K*K) result buffer for lower-triangular factor.
        K: system dimension.
        env_id: thread environment index.
        
    Returns:
        0 if successful, 1 if matrix is not positive definite.
    """
    for i in range(K):
        for j in range(i + 1):
            s = float(0.0)
            for k in range(j):
                s = s + L[env_id, i*K + k] * L[env_id, j*K + k]
            if i == j:
                val = float(A[env_id, i*K + i] - s)
                if val > 0.0:
                    L[env_id, i*K + i] = wp.sqrt(val)
                else:
                    return 1 # not positive definite
            else:
                l_diag = float(L[env_id, j*K + j])
                if l_diag > 1e-12:
                    L[env_id, i*K + j] = (A[env_id, i*K + j] - s) / l_diag
                else:
                    L[env_id, i*K + j] = 0.0
    return 0

@wp.func
def wp_cholesky_solve(L: wp.array2d(dtype=float), b: wp.array2d(dtype=float), x: wp.array2d(dtype=float), K: int, env_id: int):
    """Solves L*L^T*x = b via forward and backward substitution.
    
    L is KxK flattened in (N, K*K); b and x are (N, K).
    """
    # 1. Forward substitution: L * y = b  (store y in x temporarily)
    for i in range(K):
        s = float(0.0)
        for j in range(i):
            s = s + L[env_id, i*K + j] * x[env_id, j]
        l_diag = float(L[env_id, i*K + i])
        if l_diag > 1e-12:
            x[env_id, i] = (b[env_id, i] - s) / l_diag
        else:
            x[env_id, i] = 0.0
        
    # 2. Backward substitution: L^T * x = y
    for i in range(K-1, -1, -1):
        s = float(0.0)
        for j in range(i + 1, K):
            s = s + L[env_id, j*K + i] * x[env_id, j]
        l_diag = float(L[env_id, i*K + i])
        if l_diag > 1e-12:
            x[env_id, i] = (x[env_id, i] - s) / l_diag
        else:
            x[env_id, i] = 0.0

@wp.kernel
def _fused_dls_kernel(
    Jw: wp.array3d(dtype=float),      # (N, 6, K)
    werr: wp.array2d(dtype=float),    # (N, 6)
    joint_reg: wp.array(dtype=float), # (K)
    w_smooth_sq: float,
    q_prev_diff: wp.array2d(dtype=float), # (N, K)
    lambda_eff: wp.array(dtype=float),    # (N)
    v: wp.array2d(dtype=float),           # (N, K)
    w_post_sq: float,
    dq_min: wp.array2d(dtype=float),      # (N, K)
    dq_max: wp.array2d(dtype=float),      # (N, K)
    K: int,
    # Scratchpads (pre-allocated per environment)
    JTJ: wp.array2d(dtype=float),     # (N, K*K)
    JTdx: wp.array2d(dtype=float),    # (N, K)
    L_buf: wp.array2d(dtype=float),   # (N, K*K)
    JJT: wp.array2d(dtype=float),     # (N, 36)
    L6: wp.array2d(dtype=float),      # (N, 36)
    Jw_v: wp.array2d(dtype=float),    # (N, 6)
    Jw_v_solved: wp.array2d(dtype=float), # (N, 6)
    proj: wp.array2d(dtype=float),    # (N, K)
    # Result
    dq_out: wp.array2d(dtype=float),  # (N, K)
):
    """Fused DLS normal equations + null-space posture + active-set re-solve.
    
    Parallelized across (nworld,) environments. Each thread handles the 
    matrix math for a single arm independently. Parameterized by K DOFs.
    
    Physics:
      1. Assemble Normal Equations: JTJ = Jw^T @ Jw + damping.
      2. Solve Step 1 (Task): JTJ @ dq_task = Jw^T @ werr.
      3. Solve Step 2 (Posture): Projects posture spring into task null-space.
         N = I - J^T (JJT + lam*I)^-1 J.
      4. Active-Set: Checks dq_min/max. If violated, re-solves reduced system.
    """
    env_id = wp.tid()
    
    # --- Step 1: Compute JTJ = Jw^T @ Jw (K x K system) ---
    for i in range(K):
        for j in range(K):
            s = float(0.0)
            for k in range(6):
                s = s + Jw[env_id, k, i] * Jw[env_id, k, j]
            JTJ[env_id, i*K + j] = s
            
    # Add diagonal terms: reg + lambda + smooth
    lam = float(lambda_eff[env_id])
    for i in range(K):
        JTJ[env_id, i*K + i] = JTJ[env_id, i*K + i] + joint_reg[i] + lam + w_smooth_sq

    # --- Step 2: Compute JTdx = Jw^T @ werr + smoothing bias ---
    for i in range(K):
        s = float(0.0)
        for k in range(6):
            s = s + Jw[env_id, k, i] * werr[env_id, k]
        JTdx[env_id, i] = s + w_smooth_sq * q_prev_diff[env_id, i]
        
    # --- Step 3: Solve Task tracking: JTJ * dq_task = JTdx ---
    for i in range(K*K): L_buf[env_id, i] = 0.0
    fail_k = int(wp_cholesky_factor(JTJ, L_buf, K, env_id))
    
    # Temporary storage for unconstrained task solution
    dq_task = wp.vec(length=10, dtype=float) # constant safe upper bound
    if fail_k == 0:
        wp_cholesky_solve(L_buf, JTdx, dq_out, K, env_id)
    else:
        for i in range(K): dq_out[env_id, i] = 0.0

    for i in range(K):
        dq_task[i] = dq_out[env_id, i]
        
    # --- Step 4: Null-space posture projection (uses 6x6 Task-space solve) ---
    # 4a. Jw_v = Jw @ v (N, 6)
    for i in range(6):
        s = float(0.0)
        for j in range(K):
            s = s + Jw[env_id, i, j] * v[env_id, j]
        Jw_v[env_id, i] = s
        
    # 4b. JJT = Jw @ Jw^T + lam*I (6x6 system)
    for i in range(6):
        for j in range(6):
            s = float(0.0)
            for k in range(K):
                s = s + Jw[env_id, i, k] * Jw[env_id, j, k]
            JJT[env_id, i*6 + j] = s
            if i == j:
                JJT[env_id, i*6 + j] = JJT[env_id, i*6 + j] + lam
                
    for i in range(36): L6[env_id, i] = 0.0
    fail6 = int(wp_cholesky_factor(JJT, L6, 6, env_id))
    if fail6 == 0:
        wp_cholesky_solve(L6, Jw_v, Jw_v_solved, 6, env_id)
    else:
        for i in range(6): Jw_v_solved[env_id, i] = 0.0
        
    # 4c. proj = Jw^T @ Jw_v_solved (N, K)
    for i in range(K):
        s = float(0.0)
        for k in range(6):
            s = s + Jw[env_id, k, i] * Jw_v_solved[env_id, k]
        proj[env_id, i] = s
        
    # 4d. dq_unconstrained = dq_task + w_post_sq * (v - proj)
    for i in range(K):
        dq_out[env_id, i] = dq_task[i] + w_post_sq * (v[env_id, i] - proj[env_id, i])
        
    # --- Step 5: Active-Set Logic (Hard Box Constraints) ---
    has_active = int(0)
    for i in range(K):
        if dq_out[env_id, i] < dq_min[env_id, i] or dq_out[env_id, i] > dq_max[env_id, i]:
            has_active = 1
            
    if has_active == 1:
        # Restore original JTJ + damping for constrained re-solve
        for i in range(K):
            for j in range(K):
                s = float(0.0)
                for k in range(6):
                    s = s + Jw[env_id, k, i] * Jw[env_id, k, j]
                JTJ[env_id, i*K + j] = s
        for i in range(K):
            JTJ[env_id, i*K + i] = JTJ[env_id, i*K + i] + joint_reg[i] + lam + w_smooth_sq

        # Effective RHS including posture bias: rhs_eff = JTJ @ dq_unconstrained
        rhs_eff = wp.vec(length=10, dtype=float)
        for i in range(K):
            s = float(0.0)
            for j in range(K):
                s = s + JTJ[env_id, i*K + j] * dq_out[env_id, j]
            rhs_eff[i] = s
            
        dq_bounded = wp.vec(length=10, dtype=float)
        active_f = wp.vec(length=10, dtype=float)
        free_f = wp.vec(length=10, dtype=float)
        for i in range(K):
            val = float(dq_out[env_id, i])
            b_val = float(wp.clamp(val, dq_min[env_id, i], dq_max[env_id, i]))
            dq_bounded[i] = b_val
            if val < dq_min[env_id, i] or val > dq_max[env_id, i]:
                active_f[i] = 1.0; free_f[i] = 0.0
            else:
                active_f[i] = 0.0; free_f[i] = 1.0
                
        # Reduced RHS calculation
        rhs_c = wp.vec(length=10, dtype=float)
        for i in range(K):
            if active_f[i] == 1.0:
                rhs_c[i] = dq_bounded[i]
            else:
                s = float(0.0)
                for j in range(K):
                    s = s + JTJ[env_id, i*K + j] * active_f[j] * dq_bounded[j]
                rhs_c[i] = rhs_eff[i] - s
                
        # Modify system matrix H_c in-place: zero active rows/cols, diag=1 for active
        for i in range(K):
            for j in range(K):
                JTJ[env_id, i*K + j] = JTJ[env_id, i*K + j] * free_f[i] * free_f[j]
            JTJ[env_id, i*K + i] = JTJ[env_id, i*K + i] + active_f[i]
            
        for i in range(K*K): L_buf[env_id, i] = 0.0
        fail_kc = int(wp_cholesky_factor(JTJ, L_buf, K, env_id))
        
        if fail_kc == 0:
            for i in range(K): JTdx[env_id, i] = rhs_c[i]
            wp_cholesky_solve(L_buf, JTdx, dq_out, K, env_id)
            for i in range(K):
                dq_out[env_id, i] = wp.clamp(dq_out[env_id, i], dq_min[env_id, i], dq_max[env_id, i])
        else:
            # Fallback to clamped solution if re-solve fails
            for i in range(K):
                dq_out[env_id, i] = dq_bounded[i]


# ---------------------------------------------------------------------------
# Collision capsule builder (pure function, no env state)
# ---------------------------------------------------------------------------

def build_collision_capsules(mj_model, left_joint_names: list, right_joint_names: list,
                              radius_scale: float = 1.0):
    """Build arm/torso collision capsule lists for BatchedAnalyticalIK.

    All quaternions: wxyz (MuJoCo convention).

    Scans all mj_model bodies; matches:
      arm distal links: pattern r"elbow|wrist|forearm|upperarm" (case-insensitive)
      torso links:      pattern r"base_link|waist|torso|trunk|chest|pelvis|abdomen|spine"
    Builds capsule proxies from each body's geoms (capsule, cylinder, sphere, box).
    Left/right arm chain membership determined by descent from left_joint_names[0] /
    right_joint_names[0] parent body.
    Radii clamped to [0.008, 0.08] m after radius_scale applied.

    Args:
        mj_model:           compiled mujoco.MjModel
        left_joint_names:   ordered left arm joint names (proximal→distal)
        right_joint_names:  ordered right arm joint names (proximal→distal)
        radius_scale:       uniform radius multiplier (default 1.0)

    Returns:
        arm_capsules:   list of (body_id, radius, cx, cy, cz, ax, ay, az, half_len, is_left)
        torso_capsules: list of (body_id, radius, cx, cy, cz, ax, ay, az, half_len)
        Returns ([], []) gracefully when no matching bodies found.
    """
    arm_distal_pat = re.compile(r"elbow|wrist|forearm|upperarm", re.IGNORECASE)
    torso_pat      = re.compile(r"base_link|waist|torso|trunk|chest|pelvis|abdomen|spine",
                                re.IGNORECASE)

    left_root_body  = mj_model.joint(left_joint_names[0]).bodyid  if left_joint_names  else -1
    right_root_body = mj_model.joint(right_joint_names[0]).bodyid if right_joint_names else -1

    def _is_descendant(body_id: int, ancestor_id: int) -> bool:
        cur = body_id
        while cur > 0:
            if cur == ancestor_id:
                return True
            cur = mj_model.body_parentid[cur]
        return False

    def _quat_to_rotmat_wxyz(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def _geom_capsule_proxy(geom_id: int, default_radius: float):
        """Approximate geom as capsule proxy in body-local frame.

        Returns (radius, center_local[3], axis_local_unit[3], half_len).
        """
        gtype = int(mj_model.geom_type[geom_id])
        gsize = mj_model.geom_size[geom_id]
        gpos  = mj_model.geom_pos[geom_id].astype(np.float64)
        grot  = _quat_to_rotmat_wxyz(mj_model.geom_quat[geom_id])

        if gtype in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            radius   = float(gsize[0])
            half_len = float(gsize[1])
            axis_local = grot[:, 2]
        elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius   = float(gsize[0])
            half_len = 0.0
            axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            ext    = np.asarray(gsize[:3], dtype=np.float64)
            ax_idx = int(np.argmax(ext))
            oth    = [j for j in (0, 1, 2) if j != ax_idx]
            radius   = float(max(ext[oth[0]], ext[oth[1]]))
            half_len = float(ext[ax_idx])
            axis_local = grot[:, ax_idx]
        else:
            radius   = float(default_radius)
            half_len = 0.0
            axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        axis_norm = np.linalg.norm(axis_local)
        if axis_norm < 1e-8:
            axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis_local = axis_local / axis_norm

        radius = max(0.008, min(radius * radius_scale, 0.08))
        return radius, gpos, axis_local, half_len

    arm_capsules   = []
    torso_capsules = []

    for i in range(mj_model.nbody):
        name     = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        geom_adr = int(mj_model.body_geomadr[i])
        geom_num = int(mj_model.body_geomnum[i])
        geom_ids = range(geom_adr, geom_adr + geom_num)

        if arm_distal_pat.search(name):
            is_left  = (left_root_body  > 0 and _is_descendant(i, left_root_body))
            is_right = (right_root_body > 0 and _is_descendant(i, right_root_body))

            if not is_left and not is_right:
                print(f"[IK] Warning: body '{name}' matches arm pattern but is not descendant "
                      "of either arm root. Skipping.")
                continue

            added = False
            for gid in geom_ids:
                r, c, axis, half_len = _geom_capsule_proxy(gid, default_radius=0.05)
                arm_capsules.append((i, r,
                                     float(c[0]), float(c[1]), float(c[2]),
                                     float(axis[0]), float(axis[1]), float(axis[2]),
                                     float(half_len), is_left))
                added = True
            if not added:
                arm_capsules.append((i, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, is_left))

        elif torso_pat.search(name):
            default_r = 0.08 if "base_link" in name.lower() else 0.06
            added = False
            for gid in geom_ids:
                r, c, axis, half_len = _geom_capsule_proxy(gid, default_radius=default_r)
                torso_capsules.append((i, r,
                                       float(c[0]), float(c[1]), float(c[2]),
                                       float(axis[0]), float(axis[1]), float(axis[2]),
                                       float(half_len)))
                added = True
            if not added:
                torso_capsules.append((i, default_r, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0))

    print(f"[IK] Collision arm capsules  ({len(arm_capsules)}): "
          f"{[(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b[0]) or b[0], b[9]) for b in arm_capsules]}")
    print(f"[IK] Collision torso capsules ({len(torso_capsules)}): "
          f"{[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b[0]) or b[0] for b in torso_capsules]}")

    return arm_capsules, torso_capsules


# ---------------------------------------------------------------------------
# EE frame helpers (body vs site)
# ---------------------------------------------------------------------------

def _xmat_to_quat(xmat: np.ndarray) -> np.ndarray:
    """site_xmat (9,) row-major → wxyz quaternion via MuJoCo mju_mat2Quat."""
    q = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(q, xmat)
    return q.astype(np.float32)


_batch_mat3x3_to_quat = _rot_matrix_to_quat  # backward-compat alias


# ---------------------------------------------------------------------------
# BatchedAnalyticalIK
# ---------------------------------------------------------------------------

class BatchedAnalyticalIK:
    """Iterative damped least-squares (DLS) IK solver for two arms, batched across N environments.

    Numerical (not closed-form) solver. Each solve() refines internal joint state q_ik
    through num_iters Newton-style DLS iterations until both EEs converge.

    Algorithm (one solve() call)
    ==============================
    1. Root sync: copy non-arm (torso/floating-base) DOFs from physics into q_ik
       so FK is evaluated at actual root pose.

    2. Cartesian target rate limiting:
       pos_targets clamped toward internal committed state by at most
       max_target_pos_step per solve() — prevents IK chasing discontinuous policy output in one step.
       quat_targets passed through unmodified; quaternion interpolation is caller's responsibility.

    3. DLS iterations (num_iters times, default 3). Early exit when ALL envs satisfy pos_tol
       and ori_tol — checked after FK, before Jacobian+Cholesky
       (mirrors cuRobo check_convergence() in newton_base.py:436):
       a. FK: compute current EE position and orientation from q_ik.
       b. Convergence check on EVERY iteration (including iter 0): break immediately when
          max position error ≤ pos_tol AND max orientation error ≤ ori_tol across all envs+arms.
          At steady state warm-start FK is already within tolerance — entire Jacobian+Cholesky
          path skipped, cost reduces to single FK call.
          Uses 2 GPU→CPU syncs (L+R fused) rather than 4.
          dq early-exit GPU tensor (dq_max_gpu) resolved here at zero marginal cost.
       c. Jacobian: compute J_pos (3×K) and J_ori (3×K) for each arm at q_ik.
       d. Weighted task space: stack into J (6×K); row weights
          w = [1, 1, 1, 1, 1, 1] for [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z].
          IMPORTANT: w_ori² must be >> w_post² or DLS reaches fixed-point equilibrium
          where posture spring counteracts orientation correction
          (2.31° floor with w_ori=0.1; 0.025° with w_ori=1.0).
          Orientation error and Jacobian transformed into local EE body frame before weighting.
          Mathematically critical: down-weighting local Z (roll) relaxes wrist roll penalty
          regardless of arm global pose. World-frame weighting would only relax roll when arm
          points straight up, causing solver to fight limits for arbitrary global roll.
       e. Two-step hierarchical solve:
          Step 1 — Task (K×K Cholesky):
              (J^T W² J + diag(joint_reg) + w_smooth²·I + λ²I) dq = J^T W² err + w_smooth²·(q_prev−q)
              => dq_task = cholesky_solve(JTdx, L)
          Step 2 — Null-space posture bias:
              dq = dq_task + w_post² · N @ (q_default − q)
              N = I − J^T(JJ^T + λI)⁻¹ J  (projects onto task null-space)
          where W = diag(w), λ² is Tikhonov damping (keeps JTJ PD near singularities),
          joint_reg = w_post² * linspace(prox_scale, dist_scale, K) is ONLY Tikhonov damping
          on JTJ diagonal (proximal joints damped more — singularity avoidance).
          Posture bias via scalar w_post² applied to N @ (default − q), acting only
          in DOFs the task does not use — decouples tracking quality from posture_weight.
       f. Active-set hard joint-limit enforcement: any joint whose dq violates
          dq_min = max(q_min−q, −max_dq) or dq_max = min(q_max−q, max_dq) frozen at bound.
          Single K×K re-solve gives free joints optimal tracking under fixed constraints.
          See _active_set_constrain for details.
       g. Self-collision repulsion:
              For each active arm-body/torso-body sphere pair closer than activation_dist:
                  CPU path: mj_geomDistance for exact signed dist; J_coll = n @ (jac_torso - jac_arm)
                             (torso-arm ensures Jacobian represents separation increase rate).
                  Warp path: sphere-sphere dist from body positions; mjwarp.jac at arm body COM.
              dq_coll = k_coll * J^T @ force  (force = repulsion * normal pointing away).
       h. Combined update, step-size scaled and clamped to joint limits:
              q_ik += gain * (dq + dq_coll)

    4. Return q_ik[:, arm_qpos_addr] — absolute arm joint positions (N, 2K) ready for
       physics actuator targets.

    Absolute vs relative output
    ============================
    Returns ABSOLUTE joint positions, not deltas.
    mjlab's DifferentialIKAction uses relative: q_target = q_physics_current + dq (one step).

    Why absolute:
    - Multi-iteration convergence handles large target jumps in one policy step.
    - q_ik warm-started from previous solution; subsequent calls start near-converged.
    - Joint-bias integrator built around persistent q_ik; re-linearising at physics q
      would remove previous-command baseline for measuring PD tracking deficit.
    - Arm actuators are stiff; q_ik rarely diverges from physics.
      When it does (external disturbance, reset), call reset() to resync.

    Trade-off: if arm is forcibly displaced, q_ik FK is evaluated at wrong config
    until reset() is called. Acceptable for rigid velocity-controlled arm;
    compliant/force-controlled arm would benefit from re-linearising at physics.

    Persistent state (carries across solve() calls)
    =================================================
    - q_ik (N, nq): internal qpos warm-started each call. Only arm DOFs updated by DLS;
      non-arm DOFs synced from physics each call.
    - ee_integral / ori_integral (N, 2, 3): Cartesian EE integrator. Accumulates
      (actual_ee_pos_w − IK FK prediction) in world frame to correct persistent hardware
      tracking gap from motor lag/friction. World frame avoids rotating-frame drift over
      ~200-step window; re-expressed to body frame at apply time using R_root_T.
      Active only when ki > 0 (default 0.0). Enable on hardware with ki=0.003–0.005.
      Reset per-episode. Replaces removed joint_bias_integral (caused locomotion oscillation).

    Key hyperparameters
    ====================
    damping              = 1e-3   λ_init in Tikhonov regularisation; _lm_lambda starts here per env
    lm_adapt             = False  LM adaptive damping. When True, _lm_lambda (N,) updated
                                  per-iteration: λ halved on good steps, doubled on bad steps.
                                  Reuses FK from next iteration — zero extra FK overhead.
    lm_lambda_min        = 1e-4   minimum λ (allows near-Newton steps when converging well)
    lm_lambda_max        = 1.0    maximum λ (prevents exploding steps near singularities)
    use_jlog             = True   Apply SO(3) left Jacobian inverse (jlog) to orientation Jacobian.
                                  Corrects manifold curvature at large errors (>20°).
                                  Near-identity for small errors (θ < 5°) — negligible overhead.
    posture_weight       = 0.05   w_post base; gain for null-space posture bias.
                                  Two-step hierarchical solve:
                                    Step 1: (J^T W² J + diag(joint_reg) + λI) dq_task = J^T W² err
                                    Step 2: dq = dq_task + w_post² · N @ (q_default − q)
                                             N = I − J^T(JJ^T + λI)⁻¹ J  (null-space projector)
                                  Posture bias projected through N — only acts on DOFs task doesn't use.
                                  posture_weight can be increased without degrading Cartesian tracking.
                                  prox_scale/dist_scale control Tikhonov damping on JTJ diagonal,
                                  NOT posture gain — independent.
    prox_scale           = 4.0    Tikhonov damping multiplier for most proximal joint (shoulder).
                                  Higher → proximal joints damped more near singularities.
                                  Does NOT affect posture bias (decoupled by null-space projection).
    dist_scale           = 0.1    Tikhonov damping multiplier for most distal joint (wrist_3).
                                  Lower → distal joints free near singularities.
                                  Does NOT affect posture bias.
    max_dq               = 0.5    per-step velocity cap (rad): folded into box bounds as
                                  dq_min = max(q_min−q, −max_dq), dq_max = min(q_max−q, max_dq).
                                  Enforced as hard constraint via active-set, not post-hoc clamping.
    gain                 = 1.0    DLS step size scale. Cholesky solution already optimal for
                                  linearized Jacobian; max_dq bounds overshoot.
    pos_tol              = 1e-3   early-exit: max position error across all envs (m)
    ori_tol              = 1e-2   early-exit: max orientation error across all envs (rad, ≈ 0.6°)
    w_smooth             = 0.0    inter-call velocity smoothing weight (0 = disabled). Adds
                                  w_smooth²·I to JTJ penalising deviation from previous solve() result.
                                  WARNING: for short arms (max reach ~41cm, J elements < 0.01),
                                  w_smooth²=0.01 is 100× larger than J^T J tracking term (~0.0001),
                                  causing 10–20× slower convergence and steady-state position error.
                                  Prefer output-level EMA smoothing in caller (HumanoidEnv).
                                  Tuning if used: 0.005–0.02.
    cond_damp            = 0.0    condition-number cap (disabled by default). When > 0, sets
                                  λ_eff = max(λ, max_diag(JTJ)/cond_damp) after all regularisation.
                                  Useful only for TRUE kinematic singularities (wrist-lock, elbow-lock).
                                  NOT for workspace-boundary shaking — caused by body motion shifting
                                  closest reachable point; arm manipulability typical at boundary (w≈9e-3)
                                  and cond_damp slows convergence 5× without helping.
                                  For boundary shaking, use output-level EMA smoothing in caller.
                                  Tuning: 20–200.
    yaw_weight           = 1.0    Orientation weight for base-link yaw axis [0, 1].
                                  1.0 = full yaw tracking (default).
                                  0.0 = yaw ignored; EE free about base-link z; roll/pitch still tracked.
                                  Intermediate values (e.g. 0.5) partially soften yaw tracking.
                                  IMPORTANT: applied in base-link (body) frame, not EE frame.
                                  When yaw_weight < 1, DLS math uses body-frame yaw-attenuated weights
                                  so DLS z-row corresponds to base yaw regardless of EE orientation.                                  Mechanically: task-weight for 6th row (body-frame z) scaled by
                                  yaw_weight; convergence and LM-error metrics similarly discounted.
                                  To change at runtime: edit default in source file.

    Backends
    =========
    - GPU (wp_model provided): all N environments solved in parallel via MuJoCo Warp.
    - CPU (wp_model=None): environments solved sequentially via mujoco.mj_kinematics.

    Integration with RL policy (--use-ik in HumanoidEnv)
    ======================================================
    Policy controls ALL joints (legs + arms) and is unaware of IK. After policy writes
    joint targets, HumanoidEnv._ik_compute_arm_targets() runs IK and OVERWRITES arm columns
    of _full_joint_targets:

      - Arm joints NOT in controlled_joints: pure IK solution, no policy contribution.
      - Arm joints IN controlled_joints: policy arm action added as ADDITIVE RESIDUAL
        on top of IK target (policy_target = ik_target + action * scale).

    Policy observes target_arm_joint_pos = pre-residual IK joint targets.

    Gravity compensation (arms)
    ============================
    By default (deploy_config.py: gravity_compensation=["shoulder","elbow","wrist"]),
    HumanoidEnv injects qfrc_bias feedforward torques (gravity + Coriolis) into arm
    actuators each step. SEPARATE from joint-bias integrator above:
      - Actuator grav-comp: holds arm against gravity at TORQUE level.
      - Joint-bias integrator: corrects persistent JOINT tracking error not fully cancelled
        by torque feedforward (PD tracking lag, model mismatch, contact).
    Leg joints NOT compensated — locomotion policy trained expecting gravity on legs.

    Self-collision avoidance
    =========================
    Inspired by QP collision avoidance constraints via
    mj_geomDistance + contact normal Jacobian). Adapted as gradient term in DLS update:
      - CPU path: exact signed distances via mj_geomDistance; contact normal Jacobian
        n @ (jac_arm - jac_torso) sliced to arm DOFs.
      - Warp path: sphere-sphere distances from body COM positions; mjwarp.jac at arm body
        COM sliced to arm DOFs.
    Collision repulsion accumulated INSIDE each DLS iteration so every FK update sees geometry.

    Quaternion/frame conventions
    =============================
    - All quaternions: wxyz (MuJoCo convention).
    - All poses: world frame unless noted as body/local frame.
    - Joint order in solve() output: [left_arm..., right_arm...], proximal-to-distal.

    Args:
        mj_model: MuJoCo model.
        wp_model: MuJoCo Warp model, or None for CPU backend.
        num_envs: Number of parallel environments.
        device: Torch device string.
        ee_left_name: MuJoCo body name of left end-effector.
        ee_right_name: MuJoCo body name of right end-effector.
        left_joint_names: Left arm joint names, proximal-to-distal (shoulder → wrist).
        right_joint_names: Right arm joint names, proximal-to-distal.
        arm_collision_bodies: List of (body_id, radius_m, is_left) for arm links to protect.
        torso_collision_bodies: List of (body_id, radius_m) for torso obstacles.
        collision_activation_dist: Repulsion activates when sphere gap < this (m).
        k_coll: Scale on repulsive joint velocity.
        limit_margin: Inset from hard joint limits applied to IK clamp (rad).
        damping: Initial λ per env in Tikhonov regularisation; _lm_lambda starts at damping².
        lm_adapt: Enable LM adaptive damping. λ halved when error decreased, doubled when increased.
        lm_lambda_min: Minimum λ when adapting (default 1e-4).
        lm_lambda_max: Maximum λ when adapting (default 1.0).
        posture_weight: w_post base; gain for null-space posture bias.
            Applied via N @ (q_default − q) after task solve — does not fight Cartesian tracking.
        prox_scale: Tikhonov damping multiplier for most proximal joint (default 4.0).
        dist_scale: Tikhonov damping multiplier for most distal joint (default 0.1).
        max_dq: Per-step joint velocity clamp after Cholesky solve (rad, default 0.5).
        gain: DLS step size scale in _apply_arm_update (default 1.0).
        pos_tol: Early-exit: max position error across all envs/arms (m, default 1e-3).
        ori_tol: Early-exit: max orientation error across all envs/arms (rad, default 1e-2).
        ki: Cartesian integrator gain. 0.0 = disabled. Enable with 0.003–0.005 on hardware.
        ki_max: Per-component cap on ee_integral (m, default 0.03).
        ki_ori: Orientation integrator gain. 0.0 = disabled. Enable with 0.002–0.003.
        ki_ori_max: Per-vector norm cap on ori_integral (rad, default 0.05 ≈ 3°).
        ki_decay: Leaky decay per step (default 0.995, ~200-step memory at 50 Hz).
        max_target_pos_step: Per-solve target step cap (m) applied before DLS. Default 0.05.
        w_smooth: Inter-call velocity smoothing weight (0 = disabled). Adds w_smooth²·I to JTJ.
            See class docstring warning about short-arm steady-state error. Range: 0.005–0.02.
        use_jlog: Apply SO(3) left Jacobian inverse (jlog) to orientation Jacobian in DLS math.
            Default True; disable to compare convergence.
        cond_damp: Condition-number cap (0 = off). When > 0, sets effective λ after regularisation.
            Useful only for TRUE kinematic singularities. Range: 20–200.
        yaw_weight: EE yaw orientation weight in [0, 1] in base-link frame.
            1.0 = full yaw tracking. 0.0 = yaw free; roll/pitch still tracked.
            When < 1.0, solve() auto-switches to body-frame mode.
        debug: Print EE position/orientation errors every 50 steps.
    """

    def __init__(self, 
                 mj_model, 
                 wp_model, 
                 num_envs: int, 
                 device: str,
                 ee_left_name: str, ee_right_name: str,
                 left_joint_names: List[str], right_joint_names: List[str],
                 ee_left_type: EEFrameType = "body", ee_right_type: EEFrameType = "body",
                 collision_radius_scale: float = 1.0,
                 collision_activation_dist: float = 0.05,
                 k_coll: float = 1.5,
                 limit_margin: float = 0.1,
                 damping: float = 1e-3,
                 lm_adapt: bool = False,
                 lm_lambda_min: float = 1e-4,
                 lm_lambda_max: float = 1.0,
                 posture_weight: float = 0.02,
                 gain: float = 1.0,
                 pos_tol: float = 1e-3,
                 ori_tol: float = 1e-2,
                 prox_scale: float = 4.0,
                 dist_scale: float = 0.1,
                 ki: float = 0.00001,
                 ki_max: float = 0.03,
                 ki_ori: float = 0.0,
                 ki_ori_max: float = 0.05,
                 w_smooth: float = 0.0,
                 use_jlog: bool = True,
                 cond_damp: float = 0.0,
                 ki_decay: float = 0.995,
                 max_dq: float = 0.5,
                 max_target_pos_step: float = 0.01,
                 max_target_ori_step_rad: float = float("inf"),
                 yaw_weight: float = 0.0,
                 smooth_near_alpha: float = 0.7,
                 smooth_near_thresh: float = 0.01,  # rad
                 smooth_far_thresh: float = 0.05,   # rad
                 debug: bool = False,
                 step_dt: float = 0.02,
                 backend: str = "auto"):
        self.mj_model  = mj_model
        self.wp_model  = wp_model
        self.num_envs  = num_envs
        self.device    = device
        self.wp_device = str(device)
        self.debug     = debug
        self._debug_step = 0
        # backend="auto": warp if wp_model provided, else cpu.
        # backend="warp"/"cpu": explicit override (wp_model must match).
        if backend == "auto":
            self._use_warp = (wp_model is not None)
        elif backend == "warp":
            self._use_warp = True
        elif backend == "cpu":
            self._use_warp = False
        else:
            raise ValueError(f"Unsupported IK backend '{backend}'")
        self._yaw_weight = max(0.0, min(float(yaw_weight), 1.0))
        self._yaw_axis_weight = torch.tensor([1.0, 1.0, self._yaw_weight], dtype=torch.float32, device=device)

        # Optional deep diagnostics (single-file toggle, no caller changes required).
        # Env vars:
        #   IK_DIAG=1                       enable diagnostics
        #   IK_DIAG_FORCE_NEUTRAL=1         force posture/collision/smoothing/cond damping to zero
        #   IK_DIAG_SCENARIO=zero_error     set targets to current FK pose each solve (sanity check)
        #   IK_DIAG_SCENARIO=mirror         enforce mirrored Cartesian targets for L/R arms
        #   IK_DIAG_CHECK_FD=1              finite-difference check for right-arm J_pos columns
        #   IK_DIAG_FD_VERBOSE=1            print per-column FD rows (default prints max error only)
        #   IK_DIAG_CHECK_FK_LIN=1          compare FK delta vs J*dq on right arm
        #   IK_DIAG_K_COLL_SCALE=<float>    scale collision gain (diagnostic ablation)
        #   IK_DIAG_POSTURE_SCALE=<float>   scale posture weight (diagnostic ablation)
        self._diag_enabled = os.getenv("IK_DIAG", "0") == "1"
        if self._diag_enabled:
            self._diag_every        = int(os.getenv("IK_DIAG_EVERY", "1"))
            self._diag_env          = int(os.getenv("IK_DIAG_ENV", "0"))
            self._diag_eps          = float(os.getenv("IK_DIAG_FD_EPS", "1e-4"))
            self._diag_print_jac    = os.getenv("IK_DIAG_PRINT_JAC", "0") == "1"
            self._diag_check_fd     = os.getenv("IK_DIAG_CHECK_FD", "1") == "1"
            self._diag_check_fk_lin = os.getenv("IK_DIAG_CHECK_FK_LIN", "1") == "1"
            self._diag_fd_verbose   = os.getenv("IK_DIAG_FD_VERBOSE", "0") == "1"
            self._diag_scenario     = os.getenv("IK_DIAG_SCENARIO", "").strip().lower()
            self._diag_mirror_x     = float(os.getenv("IK_DIAG_MIRROR_X", "0.15"))
            self._diag_mirror_y     = float(os.getenv("IK_DIAG_MIRROR_Y", "0.20"))
            self._diag_mirror_z     = float(os.getenv("IK_DIAG_MIRROR_Z", "0.05"))
            # Local-only: used for init-time param scaling, not stored as attributes
            _force_neutral   = os.getenv("IK_DIAG_FORCE_NEUTRAL", "0") == "1"
            _k_coll_scale    = float(os.getenv("IK_DIAG_K_COLL_SCALE", "1.0"))
            _posture_scale   = float(os.getenv("IK_DIAG_POSTURE_SCALE", "1.0"))
        else:
            self._diag_every = 1   # only field read outside diag-gated code paths
            _force_neutral   = False
            _k_coll_scale    = 1.0
            _posture_scale   = 1.0
        assert smooth_far_thresh > smooth_near_thresh, (
            f"smooth_far_thresh ({smooth_far_thresh}) must exceed "
            f"smooth_near_thresh ({smooth_near_thresh})"
        )
        self._smooth_near_alpha  = smooth_near_alpha
        self._smooth_near_thresh = smooth_near_thresh
        self._smooth_far_thresh  = smooth_far_thresh

        posture_weight *= _posture_scale
        k_coll         *= _k_coll_scale

        if _force_neutral:
            posture_weight = 0.0
            k_coll         = 0.0
            w_smooth       = 0.0
            cond_damp      = 0.0

        K = len(left_joint_names)   # DOFs per arm
        self.K = K

        # DLS step size and regularisation
        # _lm_lambda replaces old scalar _damping_sq. Initialized to damping² for all envs.
        # When lm_adapt=True, updated per-iteration via _solve_warp/_solve_cpu;
        # otherwise stays constant (identical to old fixed Tikhonov λ²).
        lm_lambda_init   = damping ** 2
        self._lm_lambda  = torch.full((num_envs,), lm_lambda_init, dtype=torch.float32, device=device)
        self._lm_lambda_init = lm_lambda_init  # kept for reset()
        self._lm_adapt   = lm_adapt
        self._lm_lambda_min = lm_lambda_min
        self._lm_lambda_max = lm_lambda_max
        self._gain       = gain                # Step size scale on the DLS joint update
        self._w_post_sq  = posture_weight ** 2 # Posture regularisation weight squared (scalar base)
        self._max_dq     = max_dq              # Per-step joint velocity clamp (rad)

        # Per-joint posture regularisation weights: proximal joints (shoulder) get higher damping
        # than distal joints (wrist) — DLS prefers distal joints first.
        # Without this, uniform w_post²·I spreads EE correction across all 7 joints by Jacobian
        # magnitude — shoulder participates unnecessarily, causing oscillation for wrist-roll corrections.
        # Scale range: shoulder ×prox_scale → wrist_3 ×dist_scale.
        _scales = torch.linspace(prox_scale, dist_scale, K, device=device)  # (K,) decreasing
        self._joint_reg = self._w_post_sq * _scales                          # (K,) per-joint λ

        # --- EE frame indices (body or site) ---
        self._ee_left_type  = ee_left_type
        self._ee_right_type = ee_right_type
        self._ee_left_id  = mujoco.mj_name2id(mj_model, _MJ_FRAME_TYPES[ee_left_type],  ee_left_name)
        self._ee_right_id = mujoco.mj_name2id(mj_model, _MJ_FRAME_TYPES[ee_right_type], ee_right_name)
        assert self._ee_left_id  != -1, f"[IK] EE {ee_left_type} '{ee_left_name}' not found"
        assert self._ee_right_id != -1, f"[IK] EE {ee_right_type} '{ee_right_name}' not found"
        print(f"[IK] EE frames: left={self._ee_left_id} ('{ee_left_name}', {ee_left_type}), "
              f"right={self._ee_right_id} ('{ee_right_name}', {ee_right_type})")
        if self._use_warp:
            _backend_name = "warp/GPU"
        else:
            _backend_name = "mujoco/CPU"
        print(f"[IK] Backend: {_backend_name}")

        # --- Joint address maps ---
        def _qpos_addr(names):
            return torch.tensor([int(mj_model.joint(j).qposadr[0]) for j in names],
                                dtype=torch.long, device=device)
        def _dof_addr(names):
            return torch.tensor([int(mj_model.joint(j).dofadr[0]) for j in names],
                                dtype=torch.long, device=device)

        self._left_qpos_addr  = _qpos_addr(left_joint_names)
        self._right_qpos_addr = _qpos_addr(right_joint_names)
        self._arm_qpos_addr   = torch.cat([self._left_qpos_addr, self._right_qpos_addr])
        self._left_dof_addr   = _dof_addr(left_joint_names)
        self._right_dof_addr  = _dof_addr(right_joint_names)

        # --- Joint limits and null-space default pose ---
        all_names = left_joint_names + right_joint_names
        self._joint_lower = torch.tensor([mj_model.joint(j).range[0] for j in all_names],
                                         dtype=torch.float32, device=device)
        self._joint_upper = torch.tensor([mj_model.joint(j).range[1] for j in all_names],
                                         dtype=torch.float32, device=device)
        self._joint_lower_ik = self._joint_lower + limit_margin
        self._joint_upper_ik = self._joint_upper - limit_margin
        self._arm_default = torch.tensor([mj_model.joint(j).qpos0[0] for j in all_names],
                                         dtype=torch.float32, device=device)
        self._posture_initialized = False

        # Early-exit convergence tolerances (mirrors cuRobo check_convergence())
        self._pos_tol = pos_tol
        self._ori_tol = ori_tol

        # Per-arm limit slices — avoids indexing the 2K tensor inside the hot loop
        self._left_lower_ik  = self._joint_lower_ik[:K]
        self._left_upper_ik  = self._joint_upper_ik[:K]
        self._right_lower_ik = self._joint_lower_ik[K:]
        self._right_upper_ik = self._joint_upper_ik[K:]

        nv = mj_model.nv

        # --- Backend-specific FK + Jacobian buffers ---
        if self._use_warp:
            # Warp FK scratch buffers (minimal contacts, IK-only data)
            with wp.ScopedDevice(self.wp_device):
                self._wp_data = mjwarp.make_data(mj_model, nworld=num_envs, nconmax=1, njmax=1)
                self._wp_xpos  = wp.to_torch(self._wp_data.xpos)
                self._wp_xquat = wp.to_torch(self._wp_data.xquat)
                self._wp_qpos  = wp.to_torch(self._wp_data.qpos)

            self._jacp = torch.zeros(num_envs, 3, nv, device=device)
            self._jacr = torch.zeros(num_envs, 3, nv, device=device)
            self._wp_jacp = wp.from_torch(self._jacp, dtype=wp.float32)
            self._wp_jacr = wp.from_torch(self._jacr, dtype=wp.float32)

            # Contiguous buffers for passing world-frame EE positions to mjwarp.jac
            # (slices of self._wp_xpos are non-contiguous, which triggers copies in from_torch).
            self._left_pos_w_contig  = torch.zeros(num_envs, 3, device=device)
            self._right_pos_w_contig = torch.zeros(num_envs, 3, device=device)

            # Body ID buffers for mjwarp.jac. For site EE: use parent body; for body EE: use EE body.
            _left_jac_body  = (mj_model.site_bodyid[self._ee_left_id]
                               if ee_left_type == "site" else self._ee_left_id)
            _right_jac_body = (mj_model.site_bodyid[self._ee_right_id]
                                if ee_right_type == "site" else self._ee_right_id)
            self._left_body_buf  = torch.full((num_envs,), _left_jac_body,  dtype=torch.int32, device=device)
            self._right_body_buf = torch.full((num_envs,), _right_jac_body, dtype=torch.int32, device=device)
            self._wp_left_body  = wp.from_torch(self._left_body_buf,  dtype=wp.int32)
            self._wp_right_body = wp.from_torch(self._right_body_buf, dtype=wp.int32)
            # Site pose arrays (zero-copy torch views; only accessed when frame type is "site")
            if ee_left_type == "site" or ee_right_type == "site":
                self._wp_site_xpos = wp.to_torch(self._wp_data.site_xpos)   # (num_envs, nsite, 3)
                self._wp_site_xmat = wp.to_torch(self._wp_data.site_xmat)   # (num_envs, nsite, 3, 3)

            # Fused DLS Kernel Scratchpads (Warp Arrays)
            # Parallelized per env; pre-allocated to avoid kernel-time allocation.
            with wp.ScopedDevice(self.wp_device):
                self._wp_JTJ         = wp.zeros((num_envs, K * K), dtype=float)
                self._wp_JTdx        = wp.zeros((num_envs, K),     dtype=float)
                self._wp_L_buf       = wp.zeros((num_envs, K * K), dtype=float)
                self._wp_JJT         = wp.zeros((num_envs, 36),    dtype=float) # 6x6 Task Space
                self._wp_L6          = wp.zeros((num_envs, 36),    dtype=float)
                self._wp_Jw_v        = wp.zeros((num_envs, 6),     dtype=float)
                self._wp_Jw_v_solved = wp.zeros((num_envs, 6),     dtype=float)
                self._wp_proj        = wp.zeros((num_envs, K),     dtype=float)
                self._wp_dq_out      = wp.zeros((num_envs, K),     dtype=float)
        else:
            # CPU: one mj_data per env for sequential FK + mj_jacBody
            self._mj_datas = [mujoco.MjData(mj_model) for _ in range(num_envs)]
            self._cpu_jacp = np.zeros((3, nv), dtype=np.float64)
            self._cpu_jacr = np.zeros((3, nv), dtype=np.float64)
            # Pre-allocated EE state tensors (CPU fallback, float32)
            self._cpu_left_pos   = torch.zeros(num_envs, 3, device=device)
            self._cpu_left_quat  = torch.zeros(num_envs, 4, device=device)
            self._cpu_right_pos  = torch.zeros(num_envs, 3, device=device)
            self._cpu_right_quat = torch.zeros(num_envs, 4, device=device)

        # Dedicated FK scratch data for finite-difference diagnostics.
        self._diag_mj_data = mujoco.MjData(mj_model) if self._diag_enabled else None

        # Arm Jacobian slices (shared across backends)
        self._J_pos_L = torch.zeros(num_envs, 3, K, device=device)
        self._J_ori_L = torch.zeros(num_envs, 3, K, device=device)
        self._J_pos_R = torch.zeros(num_envs, 3, K, device=device)
        self._J_ori_R = torch.zeros(num_envs, 3, K, device=device)

        # Shared scratch buffers for DLS math — arms solved sequentially so one set suffices.
        # Pre-allocating avoids repeated GPU heap allocations in hot loop (2 arms × num_iters per solve()).
        self._R_ee_buf       = torch.empty(num_envs, 3, 3,  dtype=torch.float32, device=device)
        self._J_full         = torch.empty(num_envs, 6, K,  dtype=torch.float32, device=device)
        self._err_full       = torch.empty(num_envs, 6,     dtype=torch.float32, device=device)
        self._dq_buf         = torch.empty(num_envs, 2 * K, dtype=torch.float32, device=device)

        # --- DLS constants ---
        # Task weights: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z].
        # NOTE: ee_quat_targets must be set to natural wrist orientation at position target
        # (not identity) before solving; HumanoidEnv.reset() handles this automatically.
        #
        # CRITICAL: w_ori² must be >> w_post² to break DLS equilibrium where posture spring
        # counteracts orientation correction. With w_ori=0.1 (w_ori²=0.01) and w_post=0.05
        # (w_post²=0.0025), ratio is only 4×, creating permanent 2.31° error floor.
        # Setting w_ori=1.0 (w_ori²=1.0) gives 400× ratio, floor reduces to 0.025°.
        # All orientation axes equally weighted (including wrist roll / local Z).
        self.task_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
        self._use_jlog    = use_jlog  # apply jlog Jacobian correction for large orientation errors
        # Body-frame yaw-attenuated task weights: same as task_weights but ori z-row (base yaw)
        # scaled by yaw_weight. Pre-computed once; used in DLS math when yaw_weight < 1.
        if self._yaw_weight < 1.0:
            self._task_weights_bf = self.task_weights.clone()
            self._task_weights_bf[5] *= self._yaw_weight
        else:
            self._task_weights_bf = self.task_weights

        # q_ik: IK internal qpos, updated each DLS iteration.
        # Non-arm DOFs (torso, floating base) synced from physics at start of each solve;
        # only arm qpos indices modified by DLS.
        self.q_ik = torch.zeros(num_envs, mj_model.nq, device=device)
        self._non_arm_mask = torch.ones(mj_model.nq, dtype=torch.bool, device=device)
        self._non_arm_mask[self._arm_qpos_addr] = False  # True = torso/root DOF, False = arm DOF

        # Workspace saturation flag: set by _solve_warp/_solve_cpu on dq early-exit (arm at
        # workspace boundary). Diagnostics only.
        self._ik_saturated = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Cartesian EE integrator — corrects persistent hardware tracking gap from motor lag/friction.
        # Accumulated in world frame (avoids rotating-frame drift; ~200-step memory from ki_decay=0.995).
        # Re-expressed to body frame at apply time using R_root_T. Zero overhead when ki=0.0 (default).
        # Error source: _last_ee_pos_ik_w - actual_ee_pos_w (desired − actual; positive when arm lags).
        # Reset per-episode via reset(). Enable on hardware with ki=0.003–0.005.
        self._ki          = ki
        self._ki_max      = ki_max
        self._ki_ori      = ki_ori
        self._ki_ori_max  = ki_ori_max
        self._ki_decay    = ki_decay
        self.ee_integral  = torch.zeros(num_envs, 2, 3, device=device)
        self.ori_integral = torch.zeros(num_envs, 2, 3, device=device)
        # World-frame IK FK result cached after last DLS iteration (one-step lag).
        # Used by integrator to measure IK-predicted vs actual EE gap.
        self._last_ee_pos_ik_w  = torch.zeros(num_envs, 2, 3, device=device)
        self._last_ee_quat_ik_w = torch.zeros(num_envs, 2, 4, device=device)
        self._last_ee_quat_ik_w[..., 0] = 1.0  # init to identity quaternion

        # Cartesian target rate limit (position only): internal committed target state.
        self._max_target_pos_step = max_target_pos_step
        self._pos_targets_committed = torch.zeros(num_envs, 2, 3, device=device)
        self._pos_target_init_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Cartesian target rate limit (orientation): internal committed target state.
        self._max_target_ori_step_rad = max_target_ori_step_rad
        self._quat_targets_committed  = torch.zeros(num_envs, 2, 4, device=device)
        self._quat_targets_committed[..., 0] = 1.0   # identity
        self._quat_target_init_mask   = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # --- Inter-call velocity smoothing (pyroki smoothness_residual) ---
        # Limits frame-to-frame joint velocity by penalising deviation from previous solve() result.
        # Adds w_smooth²·I to JTJ and w_smooth²·(q_prev-q) to RHS.
        # Equivalent to posture regularisation with previous call's result as target (not fixed default).
        # Zero overhead when disabled (w_smooth=0).
        self._w_smooth_sq   = w_smooth ** 2
        self._q_prev_call   = torch.zeros(num_envs, 2 * K, device=device)  # arm DOFs only
        # Output-level adaptive EMA — damps workspace-boundary jitter without slowing convergence.
        # DISTINCT from _q_prev_call (intra-solve w_smooth velocity penalty).
        self._smooth_prev   = torch.zeros(num_envs, 2 * K, device=device)

        # --- Velocity/acceleration targets for inverse-dynamics feedforward ---
        # Finite differences of solve() output: dq = LPF(Δq/step_dt), ddq = Δdq/step_dt (clamped).
        # DISTINCT from _q_prev_call (intra-solve velocity smoothing).
        # Seeded at reset from physics qpos so first post-reset step gives dq ≈ 0.
        self._ff_q_prev     = torch.zeros(num_envs, 2 * K, device=device)  # q_target from previous solve
        self._ff_dq_prev    = torch.zeros(num_envs, 2 * K, device=device)  # LPF-filtered dq from previous solve
        self._step_dt       = float(step_dt)
        self._vel_lpf_alpha = 0.3   # LPF weight on new dq: 0=frozen, 1=no filter; 0.3 preserves 1-2 Hz bandwidth
        # Public outputs — written each solve(), consumed by control layer.
        self.dq_target  = torch.zeros(num_envs, 2 * K, device=device)
        self.ddq_target = torch.zeros(num_envs, 2 * K, device=device)

        # --- Condition-number damping (Nakamura–Wampler variable damping) ---
        # After building JTJ (with posture/smoothing terms), sets
        # λ_eff = max(lm_lambda, max_diag / cond_damp). Caps effective condition number at
        # ~cond_damp regardless of Jacobian singularity. 0 = disabled.
        self._cond_damp = cond_damp

        # --- Self-collision avoidance ---
        # Capsule proxies auto-derived from mj_model geoms; no caller config required.
        # Repulsion activates when capsule-capsule gap < collision_activation_dist.
        self._setup_collision_avoidance(
            left_joint_names, right_joint_names,
            collision_radius_scale, collision_activation_dist, k_coll, nv,
        )

        if self._diag_enabled:
            print("[IK-DIAG] enabled")
            print(f"[IK-DIAG] params posture_weight={posture_weight} k_coll={k_coll} w_smooth={w_smooth} cond_damp={cond_damp}")
            print(f"[IK-DIAG] addrs left_dof={self._left_dof_addr.tolist()} right_dof={self._right_dof_addr.tolist()}")
            print(f"[IK-DIAG] addrs left_qpos={self._left_qpos_addr.tolist()} right_qpos={self._right_qpos_addr.tolist()}")
            print(f"[IK-DIAG] ee_ids left={self._ee_left_id} right={self._ee_right_id}")
            print(f"[IK-DIAG] joint_names_left={left_joint_names}")
            print(f"[IK-DIAG] joint_names_right={right_joint_names}")

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor, current_physics_qpos: torch.Tensor):
        """Sync IK warm-start state for given environments.

        Copies current physics qpos into q_ik so FK in next solve() starts from
        actual robot pose rather than stale solution.

        ee_integral/ori_integral also cleared for compatibility.

        On first call (posture not yet initialized), captures actual deployment default
        pose from env 0 as null-space posture target — ensures posture spring pulls toward
        the elbow/wrist position hardware actually holds at rest, not model's qpos0.

        Args:
            env_ids:              1-D tensor of environment indices to reset.
            current_physics_qpos: (N, nq) full qpos from physics sim (all envs).
        """
        if not self._posture_initialized:
            self._arm_default = current_physics_qpos[0, self._arm_qpos_addr].clone()
            print(f"[IK] Null-space posture from deployment default: {self._arm_default.cpu().tolist()}")
            self._posture_initialized = True
        self.q_ik[env_ids]                 = current_physics_qpos[env_ids].clone()
        self.ee_integral[env_ids]          = 0.0
        self.ori_integral[env_ids]         = 0.0
        self._last_ee_pos_ik_w[env_ids]    = 0.0
        self._last_ee_quat_ik_w[env_ids]   = torch.tensor([1., 0., 0., 0.], device=self.device)
        self._ik_saturated[env_ids]        = False
        # Re-anchor velocity smoothing baseline so first post-reset solve does not
        # penalise arm for moving away from pre-reset configuration.
        self._q_prev_call[env_ids]  = current_physics_qpos[env_ids][:, self._arm_qpos_addr]
        # Seed output EMA so first post-reset step sees dq≈0 — no spurious full-blend suppression.
        self._smooth_prev[env_ids]  = current_physics_qpos[env_ids][:, self._arm_qpos_addr]
        # Reset LM damping to initial value; avoids carrying over inflated λ from failed
        # trajectory into next episode (would cause unnecessarily cautious first steps).
        self._lm_lambda[env_ids]    = self._lm_lambda_init
        self._pos_target_init_mask[env_ids] = False
        self._quat_target_init_mask[env_ids] = False
        # Seed FD state from actual arm qpos so dq_target ≈ 0 on first post-reset step.
        # Without seeding: dq_raw = (q_target - 0) / step_dt → spike.
        self._ff_q_prev[env_ids]  = current_physics_qpos[env_ids][:, self._arm_qpos_addr]
        self._ff_dq_prev[env_ids] = 0.0
        self.dq_target[env_ids]   = 0.0
        self.ddq_target[env_ids]  = 0.0
    def solve(self, physics_qpos: torch.Tensor,
              pos_targets: torch.Tensor,
              quat_targets: torch.Tensor,
              num_iters: int = 2,
              actual_ee_pos:  Optional[torch.Tensor] = None,
              actual_ee_quat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run IK and return absolute arm joint positions.

        All pos_targets/quat_targets must be in root body (base_link) frame.
        FK outputs and Jacobians are always rotated to body frame internally.
        Integrator corrections accumulated in world frame and re-expressed to body frame at apply time.

        Args:
            physics_qpos:    (N, nq) current physics state — syncs torso DOFs for FK.
            pos_targets:     (N, 2, 3) EE position targets [left, right] in root body frame.
            quat_targets:    (N, 2, 4) EE orientation targets [left, right] (wxyz) in root body frame.
            num_iters:       DLS Newton steps per call (default 2). Warm-started continuous
                             motion typically converges in 0 iters (steady state, iter-0 FK
                             exits before Jacobians) or 1 iter (small target change).
                             Use 3 for large discontinuous target jumps.
                             With gain=1.0+jlog, 3 iters from warm start converges to <1mm.
                             Use 5+ for cold start or large target jumps.
            actual_ee_pos:   (N, 2, 3) actual physics EE positions in world frame.
                             When ki > 0, drives Cartesian position integrator:
                             error = actual_ee_pos − _last_ee_pos_ik_w,
                             accumulated in world frame, applied as pos_targets offset.
                             Pass None or ki=0 to disable (zero overhead).
            actual_ee_quat:  (N, 2, 4) actual physics EE quaternions in world frame.
                             When ki_ori > 0, drives orientation integrator similarly.
                             Pass None or ki_ori=0 to disable.

        Returns:
            (N, 2*K) absolute arm joint positions in [left…, right…] order.
        """
        # Sync torso/floating-base DOFs from physics so FK sees correct root pose.
        self.q_ik[:, self._non_arm_mask] = physics_qpos[:, self._non_arm_mask]

        if self._diag_enabled:
            pos_targets, quat_targets = self._diag_prepare_targets(pos_targets, quat_targets)

        # Cartesian target rate limit (position only): clamp input targets
        # toward internal committed target by at most max_target_pos_step each solve.
        init_mask = self._pos_target_init_mask
        if (~init_mask).any():
            self._pos_targets_committed[~init_mask] = pos_targets[~init_mask]
            init_mask[~init_mask] = True
        if self._max_target_pos_step < float('inf'):
            delta = pos_targets - self._pos_targets_committed
            dist = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = (self._max_target_pos_step / dist).clamp(max=1.0)
            self._pos_targets_committed = self._pos_targets_committed + delta * scale
        else:
            self._pos_targets_committed.copy_(pos_targets)
        pos_targets = self._pos_targets_committed

        # Cartesian target rate limit (orientation): clamp input targets
        # toward internal committed target by at most max_target_ori_step_rad each solve.
        init_mask = self._quat_target_init_mask
        if (~init_mask).any():
            self._quat_targets_committed[~init_mask] = quat_targets[~init_mask]
            init_mask[~init_mask] = True
        if self._max_target_ori_step_rad < float('inf'):
            q_delta   = _quat_multiply(quat_targets, _quat_conjugate(self._quat_targets_committed))
            aa_delta  = _quat_to_axis_angle(q_delta)                    # (N, 2, 3)
            ang       = aa_delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale     = (self._max_target_ori_step_rad / ang).clamp(max=1.0)
            aa_step   = aa_delta * scale
            q_step    = _axis_angle_to_quat(aa_step)
            self._quat_targets_committed = _quat_multiply(q_step, self._quat_targets_committed)
        else:
            self._quat_targets_committed.copy_(quat_targets)
        quat_targets = self._quat_targets_committed

        # Cartesian EE integrator — corrects persistent hardware tracking gap.
        # Accumulates in world frame (avoids rotating-frame drift; ~200-step memory from ki_decay).
        # Re-expressed to body frame at apply time. Zero overhead when ki=0 or actual_ee_pos is None.
        _ki_R_root_T = None  # computed once; reused by both position and orientation integrators
        if actual_ee_pos is not None and self._ki > 0:
            err_w = self._last_ee_pos_ik_w - actual_ee_pos          # (N, 2, 3) desired - actual
            self.ee_integral.mul_(self._ki_decay).add_(self._ki * err_w)
            self.ee_integral.clamp_(-self._ki_max, self._ki_max)
            _ki_R_root_T = _quat_to_rot_matrix(self.q_ik[:, 3:7]).transpose(-2, -1)  # (N,3,3)
            correction = (_ki_R_root_T[:, None] @ self.ee_integral.unsqueeze(-1)).squeeze(-1)
            pos_targets = pos_targets + correction

        if actual_ee_quat is not None and self._ki_ori > 0:
            err_aa_w = _quat_to_axis_angle(
                _quat_multiply(self._last_ee_quat_ik_w, _quat_conjugate(actual_ee_quat))
            )                                                         # (N, 2, 3) world frame
            self.ori_integral.mul_(self._ki_decay).add_(self._ki_ori * err_aa_w)
            # Per-vector norm cap (not per-component) for orientation
            nrm = self.ori_integral.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.ori_integral.mul_(nrm.clamp(max=self._ki_ori_max) / nrm)
            if _ki_R_root_T is None:
                _ki_R_root_T = _quat_to_rot_matrix(self.q_ik[:, 3:7]).transpose(-2, -1)
            correction_aa   = (_ki_R_root_T[:, None] @ self.ori_integral.unsqueeze(-1)).squeeze(-1)
            correction_quat = _axis_angle_to_quat(correction_aa)     # (N, 2, 4)
            quat_targets    = _quat_multiply(correction_quat, quat_targets)

        # Multi-iteration solve (fresh Jacobian + collision check each step).
        if self._use_warp:
            self._solve_warp(pos_targets, quat_targets, num_iters)
        else:
            self._solve_cpu(pos_targets, quat_targets, num_iters)

        if self._diag_enabled and (self._debug_step % self._diag_every == 0):
            self._diag_log_step(pos_targets, quat_targets)

        if self.debug and self._debug_step % 50 == 0:
            self._print_debug(pos_targets, quat_targets, num_iters)
            if self._ik_saturated.any():
                print(f"[IK {self._debug_step}] WARNING: {self._ik_saturated.sum()} env(s) at workspace boundary — "
                      "target may be outside reachable space (arm drives to closest feasible pose)")
        self._debug_step += 1

        # Enforce IK joint limits on the DLS output.
        q_pre = self.q_ik[:, self._arm_qpos_addr]
        q_cmd = torch.clamp(
            q_pre,
            self._joint_lower_ik,
            self._joint_upper_ik,
        )
        if self._diag_enabled and (self._debug_step % self._diag_every == 0):
            hit = (q_pre < self._joint_lower_ik) | (q_pre > self._joint_upper_ik)
            if hit.any().item():
                ei = min(max(self._diag_env, 0), self.num_envs - 1)
                hit_idx = torch.where(hit[ei])[0].cpu().tolist()
                print(f"[IK-DIAG step={self._debug_step}] output_clamp env={ei} idx={hit_idx}")
        self.q_ik[:, self._arm_qpos_addr] = q_cmd

        # Adaptive output-level EMA — damps workspace-boundary jitter without slowing convergence.
        # Per-env max joint change distinguishes steady-state noise (dq small) from intentional
        # motion (dq large). For N>1, one near-target env does not suppress smoothing in all others.
        # DISTINCT from _q_prev_call (intra-solve w_smooth velocity penalty).
        result = q_cmd
        q_prev = self._smooth_prev                                    # (N, 2K)
        dq_max = (result - q_prev).abs().max(dim=-1).values           # (N,) per-env
        t = (dq_max - self._smooth_near_thresh).div(
                self._smooth_far_thresh - self._smooth_near_thresh).clamp(0.0, 1.0)
        alpha = (self._smooth_near_alpha + (1.0 - self._smooth_near_alpha) * t
                 ).unsqueeze(-1)                                       # (N, 1)
        smooth_mask = (alpha < 0.999).squeeze(-1)                     # (N,) bool
        if smooth_mask.any():
            idx = smooth_mask.nonzero(as_tuple=True)[0]
            result = result.clone()
            result[idx] = alpha[idx] * result[idx] + (1.0 - alpha[idx]) * q_prev[idx]
            # Sync q_ik warm-start to smoothed output — prevents Jacobian drift when
            # actuators follow smoothed target but q_ik holds unsmoothed solution.
            self.q_ik[idx[:, None], self._arm_qpos_addr] = result[idx]
        self._smooth_prev.copy_(result)

        # Update velocity-smoothing baseline: next call penalises deviating from this result.
        if self._w_smooth_sq > 0:
            self._q_prev_call.copy_(result)

        # --- Velocity/acceleration targets for inverse-dynamics feedforward ---
        # IK solves independently each step (not time-consistent), so raw FD is noisy.
        # LPF (alpha=0.3) suppresses step-to-step jitter while preserving 1-2 Hz arm bandwidth.
        dt    = self._step_dt
        alpha = self._vel_lpf_alpha
        dq_raw = (result - self._ff_q_prev) / dt
        dq     = alpha * dq_raw + (1.0 - alpha) * self._ff_dq_prev

        # Clamp ddq: τ_max=48 Nm (RS03×0.8), I_eff≈0.3-0.5 kg·m² → peak 96-160 rad/s².
        # Clamp at 100 rad/s² to prevent torque spikes from IK jumps; not a trajectory shaper.
        _MAX_DDQ = 100.0  # rad/s²
        ddq = ((dq - self._ff_dq_prev) / dt).clamp_(-_MAX_DDQ, _MAX_DDQ)

        self.dq_target.copy_(dq)
        self.ddq_target.copy_(ddq)
        self._ff_dq_prev.copy_(dq)      # store filtered dq (LPF state for next step)
        self._ff_q_prev.copy_(result)

        return result

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _setup_collision_avoidance(self, left_joint_names, right_joint_names,
                                   radius_scale, activation_dist, k_coll, nv):
        """Pre-allocate capsule-capsule collision avoidance state.

        Capsule proxies auto-built from mj_model geoms via build_collision_capsules().

        Args:
            left_joint_names:  ordered left arm joint names (proximal→distal)
            right_joint_names: ordered right arm joint names (proximal→distal)
            radius_scale:      uniform radius multiplier applied to all capsules
            activation_dist:   repulsion activates when capsule-capsule gap < this (m)
            k_coll:            repulsion scale
            nv:                number of velocity DOFs in model
        """
        self._collision_activation_dist = activation_dist
        self.k_coll = k_coll

        arm_collision_capsules, torso_collision_capsules = build_collision_capsules(
            self.mj_model, left_joint_names, right_joint_names, radius_scale,
        )

        if not arm_collision_capsules or not torso_collision_capsules:
            self._coll_enabled = False
            return
        self._coll_enabled = True

        # Unpack arm capsules: (body_id, radius, cx, cy, cz, ax, ay, az, half_len, is_left)
        self._coll_arm_body_ids      = [s[0] for s in arm_collision_capsules]
        self._coll_arm_radii         = [s[1] for s in arm_collision_capsules]
        self._coll_arm_local_center  = [[s[2], s[3], s[4]] for s in arm_collision_capsules]
        self._coll_arm_local_axis    = [[s[5], s[6], s[7]] for s in arm_collision_capsules]
        self._coll_arm_half_len      = [s[8] for s in arm_collision_capsules]
        self._coll_arm_is_left       = [s[9] for s in arm_collision_capsules]

        # Unpack torso capsules: (body_id, radius, cx, cy, cz, ax, ay, az, half_len)
        self._coll_torso_body_ids     = [s[0] for s in torso_collision_capsules]
        self._coll_torso_radii        = [s[1] for s in torso_collision_capsules]
        self._coll_torso_local_center = [[s[2], s[3], s[4]] for s in torso_collision_capsules]
        self._coll_torso_local_axis   = [[s[5], s[6], s[7]] for s in torso_collision_capsules]
        self._coll_torso_half_len     = [s[8] for s in torso_collision_capsules]

        # Tensors for vectorized computation
        self._coll_arm_body_ids_t       = torch.tensor(self._coll_arm_body_ids,     dtype=torch.long,    device=self.device)
        self._coll_arm_radii_t          = torch.tensor(self._coll_arm_radii,        dtype=torch.float32, device=self.device)
        self._coll_arm_local_center_t   = torch.tensor(self._coll_arm_local_center, dtype=torch.float32, device=self.device)  # (A, 3)
        self._coll_arm_local_axis_t     = torch.tensor(self._coll_arm_local_axis,   dtype=torch.float32, device=self.device)  # (A, 3)
        self._coll_arm_half_len_t       = torch.tensor(self._coll_arm_half_len,     dtype=torch.float32, device=self.device)  # (A,)

        self._coll_torso_body_ids_t     = torch.tensor(self._coll_torso_body_ids,     dtype=torch.long,    device=self.device)
        self._coll_torso_radii_t        = torch.tensor(self._coll_torso_radii,        dtype=torch.float32, device=self.device)
        self._coll_torso_local_center_t = torch.tensor(self._coll_torso_local_center, dtype=torch.float32, device=self.device)  # (T, 3)
        self._coll_torso_local_axis_t   = torch.tensor(self._coll_torso_local_axis,   dtype=torch.float32, device=self.device)  # (T, 3)
        self._coll_torso_half_len_t     = torch.tensor(self._coll_torso_half_len,     dtype=torch.float32, device=self.device)  # (T,)

        # Pre-computed combined radii (A, T)
        self._coll_combined_radii = (
            self._coll_arm_radii_t[:, None] + self._coll_torso_radii_t[None, :])

        # Pair filter: remove direct joint-neighbor pairs (same body or parent-child) —
        # structural adjacency, not meaningful self-collision constraints.
        parent = self.mj_model.body_parentid
        A, T = len(self._coll_arm_body_ids), len(self._coll_torso_body_ids)
        pair_mask = np.ones((A, T), dtype=np.float32)
        for ai, a_bid in enumerate(self._coll_arm_body_ids):
            a_parent = int(parent[a_bid])
            for ti, t_bid in enumerate(self._coll_torso_body_ids):
                t_parent = int(parent[t_bid])
                if a_bid == t_bid or a_parent == t_bid or t_parent == a_bid:
                    pair_mask[ai, ti] = 0.0
        self._coll_pair_mask_t = torch.tensor(pair_mask, dtype=torch.float32, device=self.device)

        # Extract unique arm bodies for Jacobian calls (one call per unique parent body).
        # Multiple capsules on same body share body_id — aggregate via index_add_.
        unique_arm_body_ids    = []
        unique_arm_bodies_info = []  # list of (body_id, is_left)
        cap_to_unique_idx      = []
        for i, bid in enumerate(self._coll_arm_body_ids):
            if bid not in unique_arm_body_ids:
                unique_arm_body_ids.append(bid)
                unique_arm_bodies_info.append((bid, self._coll_arm_is_left[i]))
            cap_to_unique_idx.append(unique_arm_body_ids.index(bid))

        self._unique_arm_bodies_info = unique_arm_bodies_info
        self._cap_to_unique_idx      = torch.tensor(cap_to_unique_idx, dtype=torch.long, device=self.device)
        self._num_unique_arm_bodies  = len(unique_arm_body_ids)

        total_pairs = A * T
        kept_pairs = int(pair_mask.sum())
        print(f"[IK] Capsule-capsule collision: {len(arm_collision_capsules)} arm capsules "
              f"({self._num_unique_arm_bodies} bodies) x {len(torso_collision_capsules)} torso capsules, "
              f"d_act={activation_dist:.3f}m, k={k_coll}")
        print(f"[IK] Collision pair mask: kept {kept_pairs}/{total_pairs} pairs "
              f"(dropped {total_pairs - kept_pairs} same-body or parent-child pairs)")

        if self._use_warp:
            with wp.ScopedDevice(self.wp_device):
                self._wp_arm_body_ids      = wp.from_torch(self._coll_arm_body_ids_t.int())
                self._wp_arm_radii         = wp.from_torch(self._coll_arm_radii_t)
                self._wp_arm_local_centers = wp.from_torch(self._coll_arm_local_center_t, dtype=wp.vec3)
                self._wp_arm_local_axes    = wp.from_torch(self._coll_arm_local_axis_t,   dtype=wp.vec3)
                self._wp_arm_half_lens     = wp.from_torch(self._coll_arm_half_len_t)

                self._wp_torso_body_ids     = wp.from_torch(self._coll_torso_body_ids_t.int())
                self._wp_torso_radii        = wp.from_torch(self._coll_torso_radii_t)
                self._wp_torso_local_center = wp.from_torch(self._coll_torso_local_center_t, dtype=wp.vec3)
                self._wp_torso_local_axis   = wp.from_torch(self._coll_torso_local_axis_t,   dtype=wp.vec3)
                self._wp_torso_half_lens     = wp.from_torch(self._coll_torso_half_len_t)

                self._wp_left_dof_addr  = wp.from_torch(self._left_dof_addr.int())
                self._wp_right_dof_addr = wp.from_torch(self._right_dof_addr.int())
                self._wp_pair_mask      = wp.from_torch(self._coll_pair_mask_t)
        else:
            # CPU: Pre-allocated scratch buffers
            self._coll_jacp1 = np.zeros((3, nv), dtype=np.float64)
            self._coll_jacr1 = np.zeros((3, nv), dtype=np.float64)
            # Index maps for slicing
            self._coll_left_dof_np  = self._left_dof_addr.cpu().numpy()
            self._coll_right_dof_np = self._right_dof_addr.cpu().numpy()

    def _solve_warp(self, pos_targets, quat_targets, num_iters):
        """Warp path: batched FK + DLS solve across all N envs using MuJoCo Warp.

        Targets are always in root body (base_link) frame.
        FK outputs and Jacobians rotated into root body frame each iteration.
        Root transforms computed once per solve() call (root DOFs fixed while
        arm DOFs are updated during IK iterations).

        Each iteration:
          1. Copy q_ik → wp_data.qpos; run mjwarp.kinematics + com_pos for FK.
          2. LM λ update (if lm_adapt): compare err_after (this FK) vs err_before (saved
             pre-step). Reuses FK already needed for convergence checking — zero extra overhead.
          3. Two early-exit checks:
             a. Position check: all envs within pos_tol and ori_tol → break.
                Checked on EVERY iteration (including iter 0). At steady state warm-start
                already within pos_tol → break after single FK call, saving Jacobian+Cholesky (~1.6ms).
             b. dq check (iter ≥ 1 only): max joint delta from previous step < 1e-5 rad → break.
                Catches workspace-boundary fixed points: arm at closest reachable pose (dq≈0)
                but residual task error remains. Sets _ik_saturated=True so solve() decays/gates
                position PI on next call, preventing integral windup.
          4. Compute arm Jacobians via mjwarp.jac; slice to arm DOF addresses.
          5. Run DLS normal equations (Cholesky) for both arms → dq (N, 2K).
          6. Optionally add self-collision repulsion → update q_ik via _apply_arm_update.

        Modifies self.q_ik in-place. Results read by solve() via q_ik[:, arm_qpos_addr].
        """
        root_pos      = self.q_ik[:, 0:3]
        root_quat     = self.q_ik[:, 3:7]
        root_quat_inv = _quat_conjugate(root_quat)
        R_root_T      = _quat_to_rot_matrix(root_quat).transpose(-2, -1)

        self._ik_saturated.fill_(False)  # reset each call; set True only on dq early-exit
        with wp.ScopedDevice(self.wp_device):
            err_before    = None   # set before each step; compared with next iter's FK result
            prev_err_max  = None   # scalar tensor: previous iter max(err_sq) for stagnation check
            dq_max_gpu    = None   # GPU tensor: max|dq| from previous step (no sync yet)
            for i in range(num_iters):
                self._wp_qpos.copy_(self.q_ik)
                mjwarp.kinematics(self.wp_model, self._wp_data)
                mjwarp.com_pos(self.wp_model, self._wp_data)
                # FK runs async on GPU while CPU continues below.

                if self._ee_left_type == "body":
                    left_pos_w  = self._wp_xpos[:, self._ee_left_id, :3]
                    left_quat_w = self._wp_xquat[:, self._ee_left_id]
                else:
                    left_pos_w  = self._wp_site_xpos[:, self._ee_left_id, :]
                    left_quat_w = _batch_mat3x3_to_quat(self._wp_site_xmat[:, self._ee_left_id])
                if self._ee_right_type == "body":
                    right_pos_w  = self._wp_xpos[:, self._ee_right_id, :3]
                    right_quat_w = self._wp_xquat[:, self._ee_right_id]
                else:
                    right_pos_w  = self._wp_site_xpos[:, self._ee_right_id, :]
                    right_quat_w = _batch_mat3x3_to_quat(self._wp_site_xmat[:, self._ee_right_id])

                # Cache world-frame FK (last iteration wins). Used by Cartesian integrator
                # to measure IK-predicted vs actual hardware EE position.
                self._last_ee_pos_ik_w[:, 0].copy_(left_pos_w)
                self._last_ee_pos_ik_w[:, 1].copy_(right_pos_w)
                self._last_ee_quat_ik_w[:, 0].copy_(left_quat_w)
                self._last_ee_quat_ik_w[:, 1].copy_(right_quat_w)

                left_pos,  left_quat  = self._world_to_body(left_pos_w,  left_quat_w,  root_pos, root_quat_inv, R_root_T)
                right_pos, right_quat = self._world_to_body(right_pos_w, right_quat_w, root_pos, root_quat_inv, R_root_T)

                # LM update: this iteration's FK gives err_after from previous step.
                # Zero overhead — reuses positions already read for convergence check.
                if self._lm_adapt and err_before is not None:
                    err_after = self._compute_err_sq(
                        left_pos, left_quat, right_pos, right_quat,
                        pos_targets, quat_targets)
                    improved = err_after < err_before
                    self._lm_lambda = torch.where(
                        improved,
                        (self._lm_lambda * 0.5).clamp(min=self._lm_lambda_min),
                        (self._lm_lambda * 2.0).clamp(max=self._lm_lambda_max),
                    )

                # Position early-exit: break before Jacobian+Cholesky when all envs within tolerance.
                # Checked on iter 0 too — at steady state warm-start FK already accurate, saves ~1.6ms.
                # Forces GPU→CPU sync; by then FK above has finished, so dq_max_gpu is also complete.
                if self._converged(left_pos, left_quat, right_pos, right_quat,
                                   pos_targets, quat_targets):
                    break
                # Workspace-saturation early-exit (iter ≥ 1): detect either
                #   (A) true fixed-point in joint space (dq≈0), OR
                #   (B) Cartesian error stagnation (closest reachable point with non-zero dq).
                # (B): unreachable targets keep producing small non-zero dq while error stops improving
                # (boundary orbiting). dq-only checks miss this, allowing PI windup.
                # Stagnation check marks saturated early so solve() decays/gates PI on next call.
                cur_err_sq  = self._compute_err_sq(
                    left_pos, left_quat, right_pos, right_quat,
                    pos_targets, quat_targets)
                cur_err_max = cur_err_sq.amax()
                stagnant = (i > 0 and prev_err_max is not None and
                            torch.abs(prev_err_max - cur_err_max).item() < 1e-8)
                dq_fixed = i > 0 and dq_max_gpu is not None and dq_max_gpu.item() < 1e-5
                if dq_fixed or stagnant:
                    self._ik_saturated.fill_(True)
                    break
                prev_err_max = cur_err_max.detach()

                # Record err_before this step so NEXT iteration can update λ.
                if self._lm_adapt:
                    err_before = cur_err_sq

                # mjwarp.jac expects world-frame point coordinates.
                # Keep sampling point in world frame; rotate rows to body frame when requested.
                self._left_pos_w_contig.copy_(left_pos_w)
                self._right_pos_w_contig.copy_(right_pos_w)
                wp_lp = wp.from_torch(self._left_pos_w_contig,  dtype=wp.vec3f)
                wp_rp = wp.from_torch(self._right_pos_w_contig, dtype=wp.vec3f)

                mjwarp.jac(self.wp_model, self._wp_data, self._wp_jacp, self._wp_jacr, wp_lp, self._wp_left_body)
                torch.bmm(R_root_T, self._jacp[:, :, self._left_dof_addr], out=self._J_pos_L)
                torch.bmm(R_root_T, self._jacr[:, :, self._left_dof_addr], out=self._J_ori_L)

                mjwarp.jac(self.wp_model, self._wp_data, self._wp_jacp, self._wp_jacr, wp_rp, self._wp_right_body)
                torch.bmm(R_root_T, self._jacp[:, :, self._right_dof_addr], out=self._J_pos_R)
                torch.bmm(R_root_T, self._jacr[:, :, self._right_dof_addr], out=self._J_ori_R)

                # DLS step (EE tracking + null-space posture + optional limit barrier)
                dq = self._compute_dls_dq_fused_warp(
                    left_pos, left_quat, right_pos, right_quat,
                    pos_targets, quat_targets,
                )

                # Collision repulsion (Warp path): fused kernel computes + projects to joint space
                if self._coll_enabled:
                    self._compute_collision_repulsion_fused_warp(dq)

                self._apply_arm_update(dq)

                # Queue GPU max reduction for next iter's dq early-exit check.
                # No .item() — stays on GPU so FK of next iter runs in parallel.
                # Sync deferred to after _converged() next iter, which already stalls CPU
                # for FK results — dq_max_gpu ready at zero marginal cost.
                dq_max_gpu = torch.amax(torch.abs(dq))

    def _solve_cpu(self, pos_targets, quat_targets, num_iters):
        """CPU path: sequential FK + DLS solve, one mj_data per environment.

        Targets are always in root body (base_link) frame.
        FK outputs and Jacobians rotated into root body frame each iteration.

        Each iteration:
          1. Loop over envs: copy q_ik[i] → mj_data.qpos; run mj_kinematics + mj_comPos.
          2. LM λ update (if lm_adapt): same logic as _solve_warp — reuses existing FK.
          3. Position early-exit if all envs converged (checked on iter 0 too).
          4. dq early-exit on fixed-point (prev_dq_max < 1e-5): marks _ik_saturated=True
             so position PI decays/gates instead of winding up.
          5. Loop over envs: compute arm Jacobians via mj_jacBody; convert to float32 torch.
          6. Run batched DLS normal equations for both arms → dq (N, 2K).
          7. Per-env collision repulsion (mj_geomDistance path).
          8. Update q_ik via _apply_arm_update.

        Substantially slower than Warp path for N > 1, but useful for debugging or
        environments without GPU. Modifies self.q_ik in-place.
        """
        left_dof_idx  = self._left_dof_addr.cpu().numpy()
        right_dof_idx = self._right_dof_addr.cpu().numpy()

        err_before = None
        prev_err_max = None  # scalar tensor: previous iter max(err_sq) for stagnation check
        prev_dq_max = None  # scalar Python float; CPU path has no async benefit
        self._ik_saturated.fill_(False)

        root_transforms = []
        for i in range(self.num_envs):
            root_pos      = self.q_ik[i, 0:3].cpu().float()
            root_quat     = self.q_ik[i, 3:7].cpu().float()
            root_quat_inv = _quat_conjugate(root_quat.unsqueeze(0)).squeeze(0)
            R_root_T      = _quat_to_rot_matrix(root_quat.unsqueeze(0)).squeeze(0).T
            root_transforms.append((root_pos, root_quat_inv, R_root_T))

        for iter_i in range(num_iters):
            for i, mj_data in enumerate(self._mj_datas):
                mj_data.qpos[:] = self.q_ik[i].detach().cpu().numpy()
                mujoco.mj_kinematics(self.mj_model, mj_data)
                mujoco.mj_comPos(self.mj_model, mj_data)

                if self._ee_left_type == "body":
                    left_pos_w  = torch.from_numpy(mj_data.xpos[self._ee_left_id].astype(np.float32))
                    left_quat_w = torch.from_numpy(mj_data.xquat[self._ee_left_id].astype(np.float32))
                else:
                    left_pos_w  = torch.from_numpy(mj_data.site_xpos[self._ee_left_id].astype(np.float32))
                    left_quat_w = torch.from_numpy(_xmat_to_quat(mj_data.site_xmat[self._ee_left_id]))
                if self._ee_right_type == "body":
                    right_pos_w  = torch.from_numpy(mj_data.xpos[self._ee_right_id].astype(np.float32))
                    right_quat_w = torch.from_numpy(mj_data.xquat[self._ee_right_id].astype(np.float32))
                else:
                    right_pos_w  = torch.from_numpy(mj_data.site_xpos[self._ee_right_id].astype(np.float32))
                    right_quat_w = torch.from_numpy(_xmat_to_quat(mj_data.site_xmat[self._ee_right_id]))

                # Cache world-frame FK (last iteration wins). Used by Cartesian integrator.
                self._last_ee_pos_ik_w[i, 0] = left_pos_w
                self._last_ee_pos_ik_w[i, 1] = right_pos_w
                self._last_ee_quat_ik_w[i, 0] = left_quat_w
                self._last_ee_quat_ik_w[i, 1] = right_quat_w

                root_pos, root_quat_inv, R_root_T = root_transforms[i]
                self._cpu_left_pos[i] = R_root_T @ (left_pos_w - root_pos)
                self._cpu_right_pos[i] = R_root_T @ (right_pos_w - root_pos)
                self._cpu_left_quat[i] = _quat_multiply(
                    root_quat_inv.unsqueeze(0), left_quat_w.unsqueeze(0)).squeeze(0)
                self._cpu_right_quat[i] = _quat_multiply(
                    root_quat_inv.unsqueeze(0), right_quat_w.unsqueeze(0)).squeeze(0)

            # LM update: this iteration's FK gives err_after from previous step.
            if self._lm_adapt and err_before is not None:
                err_after = self._compute_err_sq(
                    self._cpu_left_pos, self._cpu_left_quat,
                    self._cpu_right_pos, self._cpu_right_quat,
                    pos_targets, quat_targets)
                improved = err_after < err_before
                self._lm_lambda = torch.where(
                    improved,
                    (self._lm_lambda * 0.5).clamp(min=self._lm_lambda_min),
                    (self._lm_lambda * 2.0).clamp(max=self._lm_lambda_max),
                )

            # Position early-exit: break before Jacobians+Cholesky when all envs within tolerance.
            # Checked on iter 0 too — at steady state warm-start already accurate, saves full jac+Cholesky.
            if self._converged(self._cpu_left_pos,  self._cpu_left_quat,
                               self._cpu_right_pos, self._cpu_right_quat,
                               pos_targets, quat_targets):
                break
            # Workspace-saturation early-exit (iter ≥ 1): detect either
            #   (A) true fixed-point in joint space (dq≈0), OR
            #   (B) Cartesian error stagnation (closest reachable point, non-zero dq).
            cur_err_sq  = self._compute_err_sq(
                self._cpu_left_pos, self._cpu_left_quat,
                self._cpu_right_pos, self._cpu_right_quat,
                pos_targets, quat_targets)
            cur_err_max = cur_err_sq.amax()
            stagnant = (iter_i > 0 and prev_err_max is not None and
                        torch.abs(prev_err_max - cur_err_max).item() < 1e-8)
            dq_fixed = iter_i > 0 and prev_dq_max is not None and prev_dq_max < 1e-5
            if dq_fixed or stagnant:
                self._ik_saturated.fill_(True)
                break
            prev_err_max = cur_err_max.detach()

            # Record err_before for LM λ update in next iteration.
            if self._lm_adapt:
                err_before = cur_err_sq

            for i, mj_data in enumerate(self._mj_datas):
                R_root_T = root_transforms[i][2]

                if self._ee_left_type == "body":
                    mujoco.mj_jacBody(self.mj_model, mj_data, self._cpu_jacp, self._cpu_jacr, self._ee_left_id)
                else:
                    mujoco.mj_jacSite(self.mj_model, mj_data, self._cpu_jacp, self._cpu_jacr, self._ee_left_id)
                J_pos_l = torch.from_numpy(self._cpu_jacp[:, left_dof_idx].astype(np.float32))
                J_ori_l = torch.from_numpy(self._cpu_jacr[:, left_dof_idx].astype(np.float32))
                self._J_pos_L[i] = R_root_T @ J_pos_l
                self._J_ori_L[i] = R_root_T @ J_ori_l

                if self._ee_right_type == "body":
                    mujoco.mj_jacBody(self.mj_model, mj_data, self._cpu_jacp, self._cpu_jacr, self._ee_right_id)
                else:
                    mujoco.mj_jacSite(self.mj_model, mj_data, self._cpu_jacp, self._cpu_jacr, self._ee_right_id)
                J_pos_r = torch.from_numpy(self._cpu_jacp[:, right_dof_idx].astype(np.float32))
                J_ori_r = torch.from_numpy(self._cpu_jacr[:, right_dof_idx].astype(np.float32))
                self._J_pos_R[i] = R_root_T @ J_pos_r
                self._J_ori_R[i] = R_root_T @ J_ori_r

            dq = self._compute_dls_dq(
                self._cpu_left_pos, self._cpu_left_quat,
                self._cpu_right_pos, self._cpu_right_quat,
                pos_targets, quat_targets,
            )

            # Collision repulsion (CPU path)
            if self._coll_enabled:
                dq += self._compute_collision_repulsion_cpu()

            self._apply_arm_update(dq)
            prev_dq_max = dq.abs().max().item()

    def _compute_dls_dq_fused_warp(
        self, left_pos, left_quat, right_pos, right_quat, pos_targets, quat_targets
    ) -> torch.Tensor:
        """Fused Warp path: run DLS math kernels for both arms.
        
        Returns (N, 2K) joint delta.
        """
        # 1. Joint displacement bounds (dq_min, dq_max)
        q_arm = self.q_ik[:, self._arm_qpos_addr]
        dq_min = (self._joint_lower_ik - q_arm).clamp_(min=-self._max_dq)
        dq_max = (self._joint_upper_ik - q_arm).clamp_(max=self._max_dq)

        # 2. Cartesian errors in root body frame
        err_pos_L = pos_targets[:, 0] - left_pos
        err_pos_R = pos_targets[:, 1] - right_pos
        # Use wxyz convention for quaternion multiplication
        err_ori_L = _quat_to_axis_angle(_quat_multiply(quat_targets[:, 0], _quat_conjugate(left_quat)))
        err_ori_R = _quat_to_axis_angle(_quat_multiply(quat_targets[:, 1], _quat_conjugate(right_quat)))

        # 3. Solve Left Arm
        wp.launch(
            kernel=_fused_dls_kernel,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(torch.cat([self._J_pos_L, self._J_ori_L], dim=1)), # (N, 6, K)
                wp.from_torch(torch.cat([err_pos_L, err_ori_L], dim=1)),         # (N, 6)
                wp.from_torch(self._joint_reg),
                self._w_smooth_sq,
                wp.from_torch(self._q_prev_call[:, :self.K] - q_arm[:, :self.K]),
                wp.from_torch(self._lm_lambda),
                wp.from_torch(self._arm_default[:self.K] - q_arm[:, :self.K]),
                float(self._w_post_sq),
                wp.from_torch(dq_min[:, :self.K]),
                wp.from_torch(dq_max[:, :self.K]),
                self.K,
                self._wp_JTJ, self._wp_JTdx, self._wp_L_buf, self._wp_JJT, self._wp_L6,
                self._wp_Jw_v, self._wp_Jw_v_solved, self._wp_proj,
            ],
            outputs=[self._wp_dq_out],
            device=self.wp_device
        )
        self._dq_buf[:, :self.K].copy_(wp.to_torch(self._wp_dq_out))

        # 4. Solve Right Arm
        wp.launch(
            kernel=_fused_dls_kernel,
            dim=self.num_envs,
            inputs=[
                wp.from_torch(torch.cat([self._J_pos_R, self._J_ori_R], dim=1)), # (N, 6, K)
                wp.from_torch(torch.cat([err_pos_R, err_ori_R], dim=1)),         # (N, 6)
                wp.from_torch(self._joint_reg),
                self._w_smooth_sq,
                wp.from_torch(self._q_prev_call[:, self.K:] - q_arm[:, self.K:]),
                wp.from_torch(self._lm_lambda),
                wp.from_torch(self._arm_default[self.K:] - q_arm[:, self.K:]),
                float(self._w_post_sq),
                wp.from_torch(dq_min[:, self.K:]),
                wp.from_torch(dq_max[:, self.K:]),
                self.K,
                self._wp_JTJ, self._wp_JTdx, self._wp_L_buf, self._wp_JJT, self._wp_L6,
                self._wp_Jw_v, self._wp_Jw_v_solved, self._wp_proj,
            ],
            outputs=[self._wp_dq_out],
            device=self.wp_device
        )
        self._dq_buf[:, self.K:].copy_(wp.to_torch(self._wp_dq_out))

        return self._dq_buf

    def _compute_dls_dq(self, left_pos, left_quat, right_pos, right_quat,
                         pos_targets, quat_targets) -> torch.Tensor:
        """Run DLS step independently for each arm and concatenate results.

        Arms decoupled at this level: each gets its own Jacobian, limits, and posture default.
        Collision repulsion (if enabled) added by callers (_solve_warp/_solve_cpu) after this call.

        Args:
            left_pos:      (N, 3) current left EE position in body frame.
            left_quat:     (N, 4) current left EE orientation (wxyz) in body frame.
            right_pos:     (N, 3) current right EE position in body frame.
            right_quat:    (N, 4) current right EE orientation (wxyz) in body frame.
            pos_targets:   (N, 2, 3) EE position targets [left, right] in body frame.
            quat_targets:  (N, 2, 4) EE orientation targets [left, right] in body frame.

        Returns:
            (N, 2K) joint delta: left arm DOFs first, right arm DOFs second.
        """
        self._dq_buf[:, :self.K] = self._dls_step_cpu(
            left_pos,  left_quat,  pos_targets[:, 0], quat_targets[:, 0],
            self._J_pos_L, self._J_ori_L, self._left_qpos_addr,  self._arm_default[:self.K],
            self._left_lower_ik,  self._left_upper_ik,
            self._q_prev_call[:, :self.K], arm_name="L")
        self._dq_buf[:, self.K:] = self._dls_step_cpu(
            right_pos, right_quat, pos_targets[:, 1], quat_targets[:, 1],
            self._J_pos_R, self._J_ori_R, self._right_qpos_addr, self._arm_default[self.K:],
            self._right_lower_ik, self._right_upper_ik,
            self._q_prev_call[:, self.K:], arm_name="R")
        return self._dq_buf

    def _apply_arm_update(self, dq: torch.Tensor):
        """Apply scaled joint delta to q_ik, clamped within IK joint limits.

        dq from _dls_step_cpu (or fused kernel) is already feasible (box-constrained by active-set).
        Clamp here is safety net only for collision repulsion overflow:
        _compute_collision_repulsion_{fused_warp,cpu}() adds dq_coll after DLS math,
        which can push a joint slightly past its limit in rare cases.

        Args:
            dq: (N, 2K) joint delta from _compute_dls_dq (+ optional collision repulsion).
        """
        arm_q = (self.q_ik[:, self._arm_qpos_addr] + self._gain * dq).clamp(
            self._joint_lower_ik, self._joint_upper_ik)
        self.q_ik[:, self._arm_qpos_addr] = arm_q

    # ---- Frame helpers ----

    @staticmethod
    def _world_to_body(
        pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat_inv: torch.Tensor,
        R_root_T: torch.Tensor,
    ) -> tuple:
        """Transform batched world-frame pose into root body frame.

        Args:
            pos_w:         (N, 3) world-frame position.
            quat_w:        (N, 4) world-frame orientation (wxyz).
            root_pos:      (N, 3) root body world position.
            root_quat_inv: (N, 4) conjugate of root body quaternion (wxyz).
            R_root_T:      (N, 3, 3) transpose of root rotation matrix.

        Returns:
            pos_b:  (N, 3) position in body frame.
            quat_b: (N, 4) orientation in body frame (wxyz).
        """
        pos_b  = (R_root_T @ (pos_w - root_pos).unsqueeze(-1)).squeeze(-1)
        quat_b = _quat_multiply(root_quat_inv, quat_w)
        return pos_b, quat_b

    # ---- Collision repulsion: shared tensor math ----

    def _collision_repulsion_from_tensors(
        self, xpos: torch.Tensor, xquat: torch.Tensor
    ) -> tuple:
        """Shared math for capsule-capsule collision repulsion (steps 1–4).

        Computes world-frame repulsion forces and torques at each unique arm body
        from pairwise capsule-capsule distances via Shene's algorithm.
        Called by both Warp and CPU paths, which differ only in how xpos/xquat are sourced.

        Args:
            xpos:  (N, nbody, 3) world-frame body positions.
            xquat: (N, nbody, 4) world-frame body quaternions (wxyz).

        Returns:
            force_per_unique:  (N, U, 3) net force at each unique arm body COM.
            torque_per_unique: (N, U, 3) net torque at each unique arm body COM.
        """
        # 1. Transform capsule centers and axes to world space
        arm_body_pos_w  = xpos[:,  self._coll_arm_body_ids_t, :3]   # (N, A, 3)
        arm_body_quat_w = xquat[:, self._coll_arm_body_ids_t]        # (N, A, 4)
        arm_ctr_w = arm_body_pos_w + _quat_apply(arm_body_quat_w, self._coll_arm_local_center_t)   # (N, A, 3)
        arm_ax_w  = _quat_apply(arm_body_quat_w, self._coll_arm_local_axis_t)                      # (N, A, 3)
        arm_p1 = arm_ctr_w - arm_ax_w * self._coll_arm_half_len_t[None, :, None]                   # (N, A, 3)
        arm_p2 = arm_ctr_w + arm_ax_w * self._coll_arm_half_len_t[None, :, None]                   # (N, A, 3)

        torso_body_pos_w  = xpos[:,  self._coll_torso_body_ids_t, :3]  # (N, T, 3)
        torso_body_quat_w = xquat[:, self._coll_torso_body_ids_t]       # (N, T, 4)
        torso_ctr_w = torso_body_pos_w + _quat_apply(torso_body_quat_w, self._coll_torso_local_center_t)  # (N, T, 3)
        torso_ax_w  = _quat_apply(torso_body_quat_w, self._coll_torso_local_axis_t)                       # (N, T, 3)
        torso_p1 = torso_ctr_w - torso_ax_w * self._coll_torso_half_len_t[None, :, None]                  # (N, T, 3)
        torso_p2 = torso_ctr_w + torso_ax_w * self._coll_torso_half_len_t[None, :, None]                  # (N, T, 3)

        # 2. Pairwise capsule-capsule centerline distances and surface gaps
        # dist: (N,A,T)  delta=C1-C2: (N,A,T,3)  C1: (N,A,T,3) arm closest point
        centerline_dist, delta, C1 = _capsule_seg_seg_dist(arm_p1, arm_p2, torso_p1, torso_p2)
        gap = centerline_dist - self._coll_combined_radii.unsqueeze(0)  # (N, A, T)

        # 3. Per-pair repulsion force and torque (sum over T before aggregating)
        repulsion = -_colldist_from_sdf(gap, self._collision_activation_dist)  # (N, A, T) ≥ 0
        repulsion = repulsion * self._coll_pair_mask_t.unsqueeze(0)
        normals   = delta / centerline_dist.clamp(min=1e-8).unsqueeze(-1)      # (N, A, T, 3)

        force_per_pair  = repulsion.unsqueeze(-1) * normals                    # (N, A, T, 3)
        # Moment arm: C1 (arm closest point) relative to arm body COM.
        # C1 varies per torso capsule T — sum torques before reducing over T.
        moment_per_pair = C1 - arm_body_pos_w.unsqueeze(2)                    # (N, A, T, 3)
        torque_per_pair = torch.linalg.cross(moment_per_pair, force_per_pair, dim=-1)  # (N, A, T, 3)

        force_per_cap  = force_per_pair.sum(dim=2)   # (N, A, 3)
        torque_per_cap = torque_per_pair.sum(dim=2)  # (N, A, 3)

        # 4. Aggregate to unique parent bodies (multi-capsule bodies handled via index_add_)
        force_per_unique  = torch.zeros(self.num_envs, self._num_unique_arm_bodies, 3, device=self.device)
        torque_per_unique = torch.zeros(self.num_envs, self._num_unique_arm_bodies, 3, device=self.device)
        force_per_unique.index_add_(1, self._cap_to_unique_idx, force_per_cap)
        torque_per_unique.index_add_(1, self._cap_to_unique_idx, torque_per_cap)
        return force_per_unique, torque_per_unique

    # ---- Collision repulsion: Warp path ----

    def _compute_collision_repulsion_fused_warp(self, dq: torch.Tensor):
        """Fused Warp path: computes and projects repulsion directly to joint space.
        
        Modifies dq in-place.
        """
        wp_dq = wp.from_torch(dq)
        wp.launch(
            kernel=_fused_repulsion_kernel,
            dim=(self.num_envs, 2 * self.K),
            inputs=[
                self.wp_model.body_parentid,
                self.wp_model.body_rootid,
                self.wp_model.dof_bodyid,
                self._wp_arm_body_ids,
                self._wp_arm_radii,
                self._wp_arm_local_centers,
                self._wp_arm_local_axes,
                self._wp_arm_half_lens,
                self._wp_torso_body_ids,
                self._wp_torso_radii,
                self._wp_torso_local_center,
                self._wp_torso_local_axis,
                self._wp_torso_half_lens,
                self._wp_pair_mask,
                self._wp_left_dof_addr,
                self._wp_right_dof_addr,
                self.K,
                len(self._coll_arm_body_ids),
                len(self._coll_torso_body_ids),
                self._collision_activation_dist,
                self.k_coll,
                self._wp_data.xpos,
                self._wp_data.xquat,
                self._wp_data.subtree_com,
                self._wp_data.cdof,
            ],
            outputs=[wp_dq],
            device=self.wp_device
        )

    # ---- Collision repulsion: CPU path ----

    def _compute_collision_repulsion_cpu(self) -> torch.Tensor:
        """Capsule-capsule collision repulsion for the CPU path. Returns (N, 2K) dq."""
        xpos_list  = [d.xpos.copy()  for d in self._mj_datas]
        xquat_list = [d.xquat.copy() for d in self._mj_datas]
        xpos  = torch.from_numpy(np.stack(xpos_list)).float().to(self.device)   # (N, nbody, 3)
        xquat = torch.from_numpy(np.stack(xquat_list)).float().to(self.device)  # (N, nbody, 4)

        force_per_unique, torque_per_unique = self._collision_repulsion_from_tensors(xpos, xquat)

        dq = torch.zeros(self.num_envs, 2 * self.K, device=self.device)
        for u_idx, (bid, is_left) in enumerate(self._unique_arm_bodies_info):
            dof_addr = self._left_dof_addr if is_left else self._right_dof_addr
            dof_idx  = dof_addr.cpu().numpy()
            for i in range(self.num_envs):
                f_net   = force_per_unique[i, u_idx].cpu().double().numpy()
                t_net   = torque_per_unique[i, u_idx].cpu().double().numpy()
                com_pos = xpos[i, bid].cpu().double().numpy()
                mujoco.mj_jac(self.mj_model, self._mj_datas[i],
                               self._coll_jacp1, self._coll_jacr1, com_pos, bid)
                dq_i      = self._coll_jacp1[:, dof_idx].T @ f_net + self._coll_jacr1[:, dof_idx].T @ t_net
                arm_slice = slice(0, self.K) if is_left else slice(self.K, 2 * self.K)
                dq[i, arm_slice] += self.k_coll * torch.from_numpy(dq_i).float().to(self.device)
        return dq

    def visualize_collision_spheres(self, mj_data, scn):
        """Draw capsule collision geoms into the mjvScene for debugging."""
        if not self._coll_enabled:
            return

        def _draw_capsule(bid, local_center, local_axis, half_len, radius, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            quat_wxyz = torch.tensor(mj_data.xquat[bid], dtype=torch.float32, device=self.device)
            lc = torch.tensor(local_center, dtype=torch.float32, device=self.device)
            la = torch.tensor(local_axis,   dtype=torch.float32, device=self.device)
            ctr_w  = mj_data.xpos[bid] + _quat_apply(quat_wxyz.unsqueeze(0), lc.unsqueeze(0)).squeeze(0).cpu().numpy()
            ax_w   = _quat_apply(quat_wxyz.unsqueeze(0), la.unsqueeze(0)).squeeze(0).cpu().numpy()
            q_rot = np.zeros(4)
            mujoco.mju_quatZ2Vec(q_rot, ax_w)
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, q_rot)
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.array([radius, half_len, 0]),
                                ctr_w, mat, np.array(rgba))
            scn.ngeom += 1

        for i, bid in enumerate(self._coll_arm_body_ids):
            _draw_capsule(bid, self._coll_arm_local_center[i], self._coll_arm_local_axis[i],
                          self._coll_arm_half_len[i], self._coll_arm_radii[i], [0, 1, 0, 0.3])

        for i, bid in enumerate(self._coll_torso_body_ids):
            _draw_capsule(bid, self._coll_torso_local_center[i], self._coll_torso_local_axis[i],
                          self._coll_torso_half_len[i], self._coll_torso_radii[i], [1, 0, 0, 0.3])

    def _compute_err_sq(self, left_pos, left_quat, right_pos, right_quat,
                        pos_targets, quat_targets) -> torch.Tensor:
        """Per-env squared tracking error (pos² + ori²) for both arms. Returns (N,).

        Used by LM adaptive damping to decide whether previous step improved error.
        Cheap: four norm-squareds, no Jacobian involved.
        When yaw_weight<1, orientation error weighted so yaw is discounted
        (consistent with DLS objective yaw-row weighting).
        """
        l_aa = _quat_to_axis_angle(_quat_multiply(quat_targets[:, 0], _quat_conjugate(left_quat)))
        r_aa = _quat_to_axis_angle(_quat_multiply(quat_targets[:, 1], _quat_conjugate(right_quat)))
        if self._yaw_weight < 1.0:
            # Discount yaw (body-frame z) so LM adaptation sees the same weighted objective.
            l_aa = l_aa * self._yaw_axis_weight   # (N, 3) * (3,)
            r_aa = r_aa * self._yaw_axis_weight
        return (
            (pos_targets[:, 0] - left_pos ).pow(2).sum(dim=-1) +
            (pos_targets[:, 1] - right_pos).pow(2).sum(dim=-1) +
            l_aa.pow(2).sum(dim=-1) +
            r_aa.pow(2).sum(dim=-1)
        )

    def _converged(self, left_pos, left_quat, right_pos, right_quat,
                   pos_targets, quat_targets) -> bool:
        """True when ALL envs and both arms are within pos_tol / ori_tol.

        Analogous to cuRobo's check_convergence() (newton_base.py:436). Called after FK,
        before Jacobian+Cholesky — saves compute when warm-started solution already accurate.
        Uses exactly 2 GPU→CPU syncs (one for position, one for orientation) regardless of N:
        left+right errors fused via torch.max before .item().

        Args:
            left_pos:     (N, 3) current left EE position in body frame.
            left_quat:    (N, 4) current left EE quaternion (wxyz) in body frame.
            right_pos:    (N, 3) current right EE position in body frame.
            right_quat:   (N, 4) current right EE quaternion (wxyz) in body frame.
            pos_targets:  (N, 2, 3) EE position targets [left, right] in body frame.
            quat_targets: (N, 2, 4) EE orientation targets [left, right] in body frame.

        Returns:
            True if max error across all envs and both arms is within tolerance.
        """
        # Fuse L+R position into one .item() sync (was 2 separate).
        # torch.max(scalar_tensor, scalar_tensor): element-wise max, no alloc.
        pos_L = (pos_targets[:, 0] - left_pos ).norm(dim=-1).max()
        pos_R = (pos_targets[:, 1] - right_pos).norm(dim=-1).max()
        if torch.max(pos_L, pos_R).item() > self._pos_tol:
            return False
        # Fuse L+R orientation into one .item() sync (was 2 separate).
        aa_L = _quat_to_axis_angle(_quat_multiply(quat_targets[:, 0], _quat_conjugate(left_quat)))
        aa_R = _quat_to_axis_angle(_quat_multiply(quat_targets[:, 1], _quat_conjugate(right_quat)))
        if self._yaw_weight < 1.0:
            # Discount yaw so convergence check matches DLS objective (avoids stalling on
            # yaw error solver intentionally ignores).
            aa_L = aa_L * self._yaw_axis_weight   # (N, 3) * (3,)
            aa_R = aa_R * self._yaw_axis_weight
        ori_L = aa_L.norm(dim=-1).max()
        ori_R = aa_R.norm(dim=-1).max()
        return torch.max(ori_L, ori_R).item() <= self._ori_tol

    def _dls_step_cpu(self, pos, quat, pos_tgt, quat_tgt, J_pos, J_ori, qpos_addr, default_q,
                  lower_ik, upper_ik, q_prev_call, arm_name: str = "?") -> torch.Tensor:
        """One weighted DLS step for a single arm using joint-space normal equations (CPU path).

        Two-step hierarchical solve:
          Step 1 (task): (J^T W² J + diag(joint_reg) + w_smooth²·I + λ·I) dq = J^T W² err + w_smooth²·(q_prev−q)
                         => dq_task = cholesky_solve(JTdx, L)
          Step 2 (posture): dq = dq_task + w_post² · N @ (q_default − q)
                     N = I − J^T(JJ^T + λI)⁻¹ J  (projects onto task null-space)
        Then enforces hard joint-limit + velocity constraints via active-set re-solve:
            dq_min ≤ dq ≤ dq_max   where dq_min = max(q_min−q, −max_dq), dq_max = min(q_max−q, max_dq)
        Violated joints frozen to bound; free joints re-solved in reduced K×K system for
        optimal tracking under constraints.

        When use_jlog=True, orientation Jacobian corrected for SO(3) manifold curvature
        before normal equations: J_ori_ee ← jlog(err_ori_ee) @ J_ori_ee.

        joint_reg (self._joint_reg): per-joint Tikhonov damping on JTJ diagonal — proximal joints
        (shoulder) get higher damping than distal joints (wrist_3), so DLS prefers wrist for
        wrist-rotation corrections. Does NOT act as posture spring (decoupled by null-space projector).

        Uses self._J_full and self._err_full as pre-allocated (N,6,K)/(N,6) scratch buffers to
        avoid GPU heap allocations in hot loop. Arms solved sequentially so one shared set suffices.

        Args:
            pos:       (N, 3) current EE position.
            quat:      (N, 4) current EE orientation (wxyz).
            pos_tgt:   (N, 3) target EE position.
            quat_tgt:  (N, 4) target EE orientation (wxyz).
            J_pos:     (N, 3, K) position Jacobian.
            J_ori:     (N, 3, K) orientation Jacobian.
            qpos_addr: (K,) joint qpos indices for this arm.
            default_q: (K,) default posture for posture regularisation.
            lower_ik:  (K,) per-arm lower hard limits (joint_lower + limit_margin).
            upper_ik:  (K,) per-arm upper hard limits (joint_upper - limit_margin).
            q_prev_call: (N, K) arm joint positions from previous solve() call.
                         Used for velocity smoothing: adds w_smooth²·(q_prev_call - q) to RHS.
                         Zero overhead when w_smooth=0.

        Returns:
            dq: (N, K) joint velocity delta satisfying dq_min ≤ dq ≤ dq_max.
        """
        q_arm   = self.q_ik[:, qpos_addr]                                               # (N, K)
        # Box bounds for dq: joint displacement cannot exceed remaining range to each limit,
        # capped by max_dq for velocity safety. Folding both into one tensor lets
        # active-set check cover both joint limits AND velocity limiting in a single pass.
        dq_min  = (lower_ik - q_arm).clamp(min=-self._max_dq)                         # (N, K)
        dq_max  = (upper_ik - q_arm).clamp(max= self._max_dq)                         # (N, K)

        err_pos = pos_tgt - pos                                                        # (N, 3)
        err_ori = _quat_to_axis_angle(_quat_multiply(quat_tgt, _quat_conjugate(quat))) # (N, 3)

        # Write position (world/body frame, no rotation needed).
        self._err_full[:, :3] = err_pos
        self._J_full[:, :3, :] = J_pos

        if self._yaw_weight < 1.0:
            # Yaw-attenuated path (body frame):
            # J_ori rows express angular velocity in root body frame (rotated by R_root_T).
            # Skipping EE rotation keeps row 2 (z-axis) aligned with base-link yaw.
            # task_weights_bf attenuates this row by yaw_weight.
            # jlog correction still applied in body frame (valid — still SO(3) manifold fix).
            J_ori_bf = J_ori
            if self._use_jlog:
                J_ori_bf = _jlog(-err_ori) @ J_ori_bf   # (N, 3, 3) @ (N, 3, K) = (N, 3, K)
            self._err_full[:, 3:] = err_ori
            self._J_full[:, 3:, :] = J_ori_bf
            w = self._task_weights_bf                    # (6,) — ori z row * yaw_weight
        else:
            # EE-frame path (default): transform orientation to local EE frame.
            # Ensures task_weights penalise local wrist roll rather than a global axis.
            R_ee = _quat_to_rot_matrix(quat, out=self._R_ee_buf)           # (N, 3, 3) — reuses scratch
            R_ee_T = R_ee.transpose(-2, -1)                                # (N, 3, 3) — view, no alloc
            err_ori_ee = (R_ee_T @ err_ori.unsqueeze(-1)).squeeze(-1)  # (N, 3) ori error in EE frame
            J_ori_ee   = R_ee_T @ J_ori                                 # (N, 3, K) ori Jac in EE frame
            # SE3 log-map Jacobian correction (pyroki jlog): adjusts J_ori for manifold curvature.
            # Correct correction: J_R^{-1}(err_ee) = J_L^{-1}(-err_ee) = _jlog(-err_ori_ee).
            # Derivation: EE-frame right perturbation R_actual → R_actual·exp(J_ee·δq),
            # d(err_ee)/dq = -J_R^{-1}(err_ee) @ J_ori_ee, corrected Jacobian:
            # J_R^{-1}(err_ee) @ J_ori_ee = _jlog(-err_ee) @ J_ori_ee.
            # Correction ≈ I for small errors (<5°); meaningful above ~20°.
            if self._use_jlog:
                J_ori_ee = _jlog(-err_ori_ee) @ J_ori_ee   # (N, 3, 3) @ (N, 3, K) = (N, 3, K)
            self._err_full[:, 3:] = err_ori_ee
            self._J_full[:, 3:, :] = J_ori_ee
            w = self.task_weights                            # (6,)  — row weights

        # Jw   = w @ J (N, 6, K) — Jacobian rows scaled by task weights
        # werr = w * err (N, 6)   — Cartesian error scaled by task weights
        Jw   = w.unsqueeze(-1) * self._J_full
        werr = w * self._err_full

        # --- Step 1: Unconstrained DLS Solve ---
        # JTJ = J^T diag(w²) J + diag(reg) + diag(smooth) + diag(λ)
        JTJ = torch.bmm(Jw.transpose(-2, -1), Jw)
        
        # Build effective diagonal damping functionally (avoids loops/multi-launches)
        diag_add = self._joint_reg + self._lm_lambda.unsqueeze(-1)
        if self._w_smooth_sq > 0:
            diag_add = diag_add + self._w_smooth_sq
        
        # Tikhonov / condition-number damping (Nakamura–Wampler)
        if self._cond_damp > 0:
            # max_diag approximated from scaled Jacobian power
            max_diag = (Jw.pow(2).sum(dim=1) + self._joint_reg).amax(dim=-1)
            lambda_cond = max_diag / self._cond_damp
            diag_add = diag_add + torch.max(torch.zeros_like(lambda_cond), 
                                            lambda_cond - self._lm_lambda).unsqueeze(-1)

        JTJ = JTJ + torch.diag_embed(diag_add)
        
        # JTdx = J^T diag(w²) err + w_smooth² * (q_prev - q)
        JTdx = torch.bmm(Jw.transpose(-2, -1), werr.unsqueeze(-1)).squeeze(-1)
        if self._w_smooth_sq > 0:
            JTdx = JTdx + self._w_smooth_sq * (q_prev_call - q_arm)

        L, info = torch.linalg.cholesky_ex(JTJ)
        dq = torch.cholesky_solve(JTdx.unsqueeze(-1), L).squeeze(-1)

        # --- Step 2: Null-space posture bias ---
        # dq = dq_task + N @ posture_spring
        # N = I - J^T(JJ^T + λI)⁻¹J projects spring onto task null-space.
        v = default_q - q_arm
        Jw_v = torch.bmm(Jw, v.unsqueeze(-1)).squeeze(-1)
        JJT  = torch.bmm(Jw, Jw.transpose(-2, -1))
        # Use existing LM lambda for JJT damping
        JJT  = JJT + torch.diag_embed(self._lm_lambda.unsqueeze(-1).expand(-1, 6))
        
        L_jjt, _ = torch.linalg.cholesky_ex(JJT)
        Jw_v_solved = torch.cholesky_solve(Jw_v.unsqueeze(-1), L_jjt).squeeze(-1)
        proj = torch.bmm(Jw.transpose(-2, -1), Jw_v_solved.unsqueeze(-1)).squeeze(-1)

        dq = dq + self._w_post_sq * (v - proj)

        failed = info != 0
        if self.debug and failed.any():
            print(f"[IK] WARNING: cholesky_ex failed for {failed.sum().item()} env(s) — "
                  f"dq zeroed (check for NaN in qpos or degenerate Jacobian)")
        dq = torch.where(failed.unsqueeze(-1), torch.zeros_like(dq), dq)

        # --- Step 3: Box Constraints (Active-Set) ---
        dq_raw = dq.clone()
        active = (dq < dq_min) | (dq > dq_max)
        if active.any().item():
            # Augment RHS with posture contribution so re-solve respects null-space bias.
            rhs_eff = torch.bmm(JTJ, dq.unsqueeze(-1)).squeeze(-1)

            if self._diag_enabled and (self._debug_step % self._diag_every == 0):
                self._diag_active_set(arm_name, dq_raw, dq_min, dq_max, active, q_arm, lower_ik, upper_ik)

            # Functional one-iteration active-set solve
            dq_bounded = dq.clamp(dq_min, dq_max)
            active_f   = active.float()
            free_f     = 1.0 - active_f

            # Reduce RHS: remove contribution of fixed joints
            H_active_cols = JTJ * active_f.unsqueeze(-2)
            rhs_c = rhs_eff - torch.bmm(H_active_cols, dq_bounded.unsqueeze(-1)).squeeze(-1)
            rhs_c = torch.where(active, dq_bounded, rhs_c)

            # Build reduced system H_c functionally
            H_c = JTJ * free_f.unsqueeze(-1) * free_f.unsqueeze(-2)
            H_c = H_c + torch.diag_embed(active_f)

            L_c, info_c = torch.linalg.cholesky_ex(H_c)
            dq = torch.cholesky_solve(rhs_c.unsqueeze(-1), L_c).squeeze(-1)
            dq = torch.where(info_c.unsqueeze(-1) != 0, dq_bounded, dq).clamp(dq_min, dq_max)

            if self._diag_enabled and (self._debug_step % self._diag_every == 0):
                ei = min(max(self._diag_env, 0), self.num_envs - 1)
                print(f"[IK-DIAG step={self._debug_step}] dq_after_active[{arm_name}]={dq[ei].detach().cpu().numpy().tolist()}")

        if self._diag_enabled and (self._debug_step % self._diag_every == 0):
            if self._diag_print_jac:
                ei = min(max(self._diag_env, 0), self.num_envs - 1)
                print(f"[IK-DIAG step={self._debug_step}] J_pos_{arm_name}={J_pos[ei].detach().cpu().numpy().tolist()}")
            if arm_name == "R":
                self._diag_check_right_jacobian_fd(J_pos, pos)
                self._diag_check_right_fk_linearization(J_pos, pos)

        return dq

    # -------------------------
    # Diagnostic helpers
    # -------------------------

    def _diag_active_set(self, arm_name, dq_raw, dq_min, dq_max, active, q_arm, lower_ik, upper_ik):
        ei = min(max(self._diag_env, 0), self.num_envs - 1)
        aidx = torch.where(active[ei])[0].cpu().tolist()
        print(f"[IK-DIAG step={self._debug_step}] active arm={arm_name} env={ei} idx={aidx}")
        if len(aidx) > 0:
            print(f"[IK-DIAG step={self._debug_step}] dq_raw[{arm_name}]={dq_raw[ei].detach().cpu().numpy().tolist()}")
            print(f"[IK-DIAG step={self._debug_step}] dq_min[{arm_name}]={dq_min[ei].detach().cpu().numpy().tolist()}")
            print(f"[IK-DIAG step={self._debug_step}] dq_max[{arm_name}]={dq_max[ei].detach().cpu().numpy().tolist()}")
            q_e = q_arm[ei]
            lo_e = lower_ik
            hi_e = upper_ik
            q_cpu = q_e.detach().cpu().numpy().tolist()
            lo_cpu = lo_e.detach().cpu().numpy().tolist()
            hi_cpu = hi_e.detach().cpu().numpy().tolist()
            cause_rows = []
            for j in aidx:
                lo_margin = q_cpu[j] - lo_cpu[j]
                hi_margin = hi_cpu[j] - q_cpu[j]
                hit_low_range = lo_margin <= self._max_dq + 1e-6
                hit_high_range = hi_margin <= self._max_dq + 1e-6
                if hit_low_range and hit_high_range:
                    cause = "joint_range+max_dq"
                elif hit_low_range or hit_high_range:
                    cause = "joint_range"
                else:
                    cause = "max_dq"
                cause_rows.append({
                    "idx": int(j),
                    "q": float(q_cpu[j]),
                    "lower": float(lo_cpu[j]),
                    "upper": float(hi_cpu[j]),
                    "lo_margin": float(lo_margin),
                    "hi_margin": float(hi_margin),
                    "cause": cause,
                })
            print(f"[IK-DIAG step={self._debug_step}] active_detail[{arm_name}]={cause_rows}")

    def _diag_prepare_targets(self, pos_targets: torch.Tensor, quat_targets: torch.Tensor):
        """Inject test scenarios at solve() entry for repeatable diagnostics."""
        if self._diag_scenario == "zero_error":
            lp, lq, rp, rq = self._diag_fk_env(min(max(self._diag_env, 0), self.num_envs - 1))
            pos_targets = pos_targets.clone()
            quat_targets = quat_targets.clone()
            pos_targets[:, 0] = lp.unsqueeze(0).expand(self.num_envs, -1)
            pos_targets[:, 1] = rp.unsqueeze(0).expand(self.num_envs, -1)
            quat_targets[:, 0] = lq.unsqueeze(0).expand(self.num_envs, -1)
            quat_targets[:, 1] = rq.unsqueeze(0).expand(self.num_envs, -1)
        elif self._diag_scenario == "mirror":
            pos_targets = pos_targets.clone()
            quat_targets = quat_targets.clone()
            pos_targets[:, 0, 0] = self._diag_mirror_x
            pos_targets[:, 0, 1] = self._diag_mirror_y
            pos_targets[:, 0, 2] = self._diag_mirror_z
            pos_targets[:, 1, 0] = self._diag_mirror_x
            pos_targets[:, 1, 1] = -self._diag_mirror_y
            pos_targets[:, 1, 2] = self._diag_mirror_z
            quat_targets[:, 1] = quat_targets[:, 0]
        return pos_targets, quat_targets

    def _diag_fk_env(self, env_idx: int):
        """FK for one env from current q_ik. Returns (lpos, lquat, rpos, rquat) in body frame."""
        env_idx = min(max(env_idx, 0), self.num_envs - 1)
        root_pos = self.q_ik[env_idx, 0:3]
        root_qi  = _quat_conjugate(self.q_ik[env_idx:env_idx+1, 3:7]).squeeze(0)
        if self._use_warp:
            with wp.ScopedDevice(self.wp_device):
                self._wp_qpos.copy_(self.q_ik)
                mjwarp.kinematics(self.wp_model, self._wp_data)
            if self._ee_left_type == "body":
                lp = self._wp_xpos[env_idx, self._ee_left_id, :3].detach().clone()
                lq = self._wp_xquat[env_idx, self._ee_left_id].detach().clone()
            else:
                lp = self._wp_site_xpos[env_idx, self._ee_left_id, :].detach().clone()
                lq = _batch_mat3x3_to_quat(self._wp_site_xmat[env_idx:env_idx+1, self._ee_left_id]).squeeze(0).detach().clone()
            if self._ee_right_type == "body":
                rp = self._wp_xpos[env_idx, self._ee_right_id, :3].detach().clone()
                rq = self._wp_xquat[env_idx, self._ee_right_id].detach().clone()
            else:
                rp = self._wp_site_xpos[env_idx, self._ee_right_id, :].detach().clone()
                rq = _batch_mat3x3_to_quat(self._wp_site_xmat[env_idx:env_idx+1, self._ee_right_id]).squeeze(0).detach().clone()
        else:
            d = self._diag_mj_data if self._diag_mj_data is not None else self._mj_datas[env_idx]
            d.qpos[:] = self.q_ik[env_idx].detach().cpu().numpy()
            mujoco.mj_kinematics(self.mj_model, d)
            mujoco.mj_comPos(self.mj_model, d)
            if self._ee_left_type == "body":
                lp = torch.from_numpy(d.xpos[self._ee_left_id].astype(np.float32)).to(self.device)
                lq = torch.from_numpy(d.xquat[self._ee_left_id].astype(np.float32)).to(self.device)
            else:
                lp = torch.from_numpy(d.site_xpos[self._ee_left_id].astype(np.float32)).to(self.device)
                lq = torch.from_numpy(_xmat_to_quat(d.site_xmat[self._ee_left_id])).to(self.device)
            if self._ee_right_type == "body":
                rp = torch.from_numpy(d.xpos[self._ee_right_id].astype(np.float32)).to(self.device)
                rq = torch.from_numpy(d.xquat[self._ee_right_id].astype(np.float32)).to(self.device)
            else:
                rp = torch.from_numpy(d.site_xpos[self._ee_right_id].astype(np.float32)).to(self.device)
                rq = torch.from_numpy(_xmat_to_quat(d.site_xmat[self._ee_right_id])).to(self.device)
        # Convert to body frame
        lp = _quat_apply(root_qi.unsqueeze(0), (lp - root_pos).unsqueeze(0)).squeeze(0)
        rp = _quat_apply(root_qi.unsqueeze(0), (rp - root_pos).unsqueeze(0)).squeeze(0)
        lq = _quat_multiply(root_qi.unsqueeze(0), lq.unsqueeze(0)).squeeze(0)
        rq = _quat_multiply(root_qi.unsqueeze(0), rq.unsqueeze(0)).squeeze(0)
        return lp, lq, rp, rq

    def _diag_log_step(self, pos_targets: torch.Tensor, quat_targets: torch.Tensor):
        """Per-step L/R error vector log for requested evidence."""
        ei = min(max(self._diag_env, 0), self.num_envs - 1)
        lp, lq, rp, rq = self._diag_fk_env(ei)
        l_err = pos_targets[ei, 0] - lp
        r_err = pos_targets[ei, 1] - rp
        l_ang = _quat_to_axis_angle(_quat_multiply(quat_targets[ei:ei+1, 0], _quat_conjugate(lq.unsqueeze(0))))[0].norm().item()
        r_ang = _quat_to_axis_angle(_quat_multiply(quat_targets[ei:ei+1, 1], _quat_conjugate(rq.unsqueeze(0))))[0].norm().item()
        print(
            f"[IK-DIAG step={self._debug_step}] env={ei} "
            f"L_err_vec={l_err.detach().cpu().numpy().tolist()} L_err={l_err.norm().item():.6f} L_ang={l_ang:.6f} "
            f"R_err_vec={r_err.detach().cpu().numpy().tolist()} R_err={r_err.norm().item():.6f} R_ang={r_ang:.6f}"
        )

    def _diag_check_right_jacobian_fd(self, J_pos_R: torch.Tensor, right_pos_batch: torch.Tensor):
        """Finite-difference check for right-arm position Jacobian columns (env selected)."""
        if not self._diag_check_fd:
            return
        ei = min(max(self._diag_env, 0), self.num_envs - 1)
        eps = self._diag_eps
        if eps <= 0:
            return

        x0 = right_pos_batch[ei].detach().clone()
        q_saved = self.q_ik[ei].detach().clone()
        ana = J_pos_R[ei].detach().clone()  # (3, K)

        col_err = []
        for j in range(self.K):
            q1 = q_saved.clone()
            q1[self._right_qpos_addr[j]] += eps
            d = self._diag_mj_data if self._diag_mj_data is not None else self._mj_datas[ei]
            d.qpos[:] = q1.detach().cpu().numpy()
            mujoco.mj_kinematics(self.mj_model, d)
            mujoco.mj_comPos(self.mj_model, d)
            if self._ee_right_type == "body":
                x1 = torch.from_numpy(d.xpos[self._ee_right_id].astype(np.float32)).to(self.device)
            else:
                x1 = torch.from_numpy(d.site_xpos[self._ee_right_id].astype(np.float32)).to(self.device)
            root_pos = self.q_ik[ei, 0:3]
            root_qi  = _quat_conjugate(self.q_ik[ei:ei+1, 3:7]).squeeze(0)
            x1 = _quat_apply(root_qi.unsqueeze(0), (x1 - root_pos).unsqueeze(0)).squeeze(0)
            fd = (x1 - x0) / eps
            err = (fd - ana[:, j]).norm().item()
            col_err.append(err)
            if self._diag_fd_verbose:
                print(
                    f"[IK-DIAG step={self._debug_step}] FD_JR col={j} "
                    f"fd={fd.detach().cpu().numpy().tolist()} ana={ana[:, j].detach().cpu().numpy().tolist()} err={err:.6e}"
                )
        print(f"[IK-DIAG step={self._debug_step}] FD_JR max_col_err={max(col_err) if col_err else 0.0:.6e}")

    def _diag_check_right_fk_linearization(self, J_pos_R: torch.Tensor, right_pos_batch: torch.Tensor):
        """Check FK delta vs J*dq for right arm (env selected)."""
        if not self._diag_check_fk_lin:
            return
        ei = min(max(self._diag_env, 0), self.num_envs - 1)
        dq_test = torch.zeros(self.K, device=self.device)
        dq_test[0] = self._diag_eps

        x0 = right_pos_batch[ei].detach().clone()
        q_saved = self.q_ik[ei].detach().clone()
        q1 = q_saved.clone()
        q1[self._right_qpos_addr] = q1[self._right_qpos_addr] + dq_test

        d = self._diag_mj_data if self._diag_mj_data is not None else self._mj_datas[ei]
        d.qpos[:] = q1.detach().cpu().numpy()
        mujoco.mj_kinematics(self.mj_model, d)
        mujoco.mj_comPos(self.mj_model, d)
        if self._ee_right_type == "body":
            x1 = torch.from_numpy(d.xpos[self._ee_right_id].astype(np.float32)).to(self.device)
        else:
            x1 = torch.from_numpy(d.site_xpos[self._ee_right_id].astype(np.float32)).to(self.device)
        root_pos = self.q_ik[ei, 0:3]
        root_qi  = _quat_conjugate(self.q_ik[ei:ei+1, 3:7]).squeeze(0)
        x1 = _quat_apply(root_qi.unsqueeze(0), (x1 - root_pos).unsqueeze(0)).squeeze(0)

        fk_dx = x1 - x0
        j_dx = J_pos_R[ei] @ dq_test
        err = (fk_dx - j_dx).norm().item()
        print(
            f"[IK-DIAG step={self._debug_step}] FKvsJ_R "
            f"fk_dx={fk_dx.detach().cpu().numpy().tolist()} jdx={j_dx.detach().cpu().numpy().tolist()} err={err:.6e}"
        )

    def _print_debug(self, pos_targets, quat_targets, num_iters):
        """Print EE position/orientation tracking error for env 0.

        Called by solve() every 50 steps when debug=True. Re-runs FK on q_ik (without
        modifying state) so reported error reflects final converged state after all DLS
        iterations, not pre-solve warm-start.

        Warp FK world-frame positions always converted to body frame before comparison.
        CPU path: _cpu_left_pos etc. are already in body frame.

        Output format per arm:
            [IK <step> iters=<N>] L/R pos=[x,y,z] tgt=[x,y,z] err=<m>m
                                        quat=[w,x,y,z] tgt=[w,x,y,z] ang=<deg>°
        """
        def _fmt(t): return [f"{v:.3f}" for v in t[0].cpu().tolist()]

        if self._use_warp:
            with wp.ScopedDevice(self.wp_device):
                self._wp_qpos.copy_(self.q_ik)
                mjwarp.kinematics(self.wp_model, self._wp_data)
            if self._ee_left_type == "body":
                ee_pos_w_l  = self._wp_xpos[:, self._ee_left_id, :3]
                ee_quat_w_l = self._wp_xquat[:, self._ee_left_id]
            else:
                ee_pos_w_l  = self._wp_site_xpos[:, self._ee_left_id, :]
                ee_quat_w_l = _batch_mat3x3_to_quat(self._wp_site_xmat[:, self._ee_left_id])
            if self._ee_right_type == "body":
                ee_pos_w_r  = self._wp_xpos[:, self._ee_right_id, :3]
                ee_quat_w_r = self._wp_xquat[:, self._ee_right_id]
            else:
                ee_pos_w_r  = self._wp_site_xpos[:, self._ee_right_id, :]
                ee_quat_w_r = _batch_mat3x3_to_quat(self._wp_site_xmat[:, self._ee_right_id])
            root_pos  = self.q_ik[:, 0:3]
            rqi       = _quat_conjugate(self.q_ik[:, 3:7])
            ee_pos_l  = _quat_apply(rqi, ee_pos_w_l  - root_pos)
            ee_quat_l = _quat_multiply(rqi, ee_quat_w_l)
            ee_pos_r  = _quat_apply(rqi, ee_pos_w_r  - root_pos)
            ee_quat_r = _quat_multiply(rqi, ee_quat_w_r)
            ee_positions = [(ee_pos_l, ee_quat_l), (ee_pos_r, ee_quat_r)]
        else:
            # CPU path: _cpu_left_pos etc. are already in body frame
            ee_positions = [(self._cpu_left_pos,  self._cpu_left_quat),
                            (self._cpu_right_pos, self._cpu_right_quat)]

        for (ee_pos, ee_quat), side, pt, qt in zip(
            ee_positions,
            ["L", "R"],
            [pos_targets[:, 0], pos_targets[:, 1]],
            [quat_targets[:, 0], quat_targets[:, 1]],
        ):
            ee_pos  = ee_pos.to(self.device)
            ee_quat = ee_quat.to(self.device)
            pe = (pt[0] - ee_pos[0]).norm().item()
            ae = _quat_to_axis_angle(_quat_multiply(qt[0:1], _quat_conjugate(ee_quat[0:1])))[0].norm().item()
            print(f"[IK {self._debug_step} iters={num_iters}] {side} "
                  f"pos={_fmt(ee_pos)} tgt={_fmt(pt)} err={pe:.4f}m  "
                  f"quat={_fmt(ee_quat)} tgt={_fmt(qt)} ang={ae*57.3:.1f}°")
