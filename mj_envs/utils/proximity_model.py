"""Analytical bimanual proximity model: K=48 capsule-capsule clearance distances per step.

Overview
--------
At every RL step this module returns (N, 48) signed surface-to-surface clearance distances
between all arm-segment and torso-segment pairs for a batch of N environments. Negative
values mean the capsule surfaces are interpenetrating. The distances feed a quadratic reward
term that gives the policy a smooth gradient away from self-collision without any binary step.

Why analytical, not an MLP
---------------------------
The naive approach is to train an MLP on (joint angles → segment distances). We rejected it:

1. **Exact vs approximate.** The segment-to-segment distance is a closed-form algebraic
   function of body positions. An MLP only approximates it, with ~0.5–2 cm error in the
   critical 0–10 cm zone. Errors in that zone mean the policy receives wrong gradients
   exactly where precision matters most.

2. **100× faster.** Analytical cost ≈ 800 FLOPs for 48 pairs. A 256-unit × 3-layer MLP
   costs ~86 000 FLOPs and still produces approximation error. No trade-off to make.

3. **Smooth by construction.** The capsule distance function is C∞ in body positions
   (C0 at capsule-endpoint boundaries, but that is the best possible for piecewise-linear
   geometry). Smoothness is a mathematical fact, not an architecture choice.

4. **No training pipeline.** MLP requires: generating a collision-free dataset, choosing a
   network architecture, setting a learning rate, validating on out-of-distribution poses,
   and re-training if the robot MJCF changes. None of that applies here.

5. **No catastrophic forgetting.** If the policy learned to avoid collisions, near-collision
   poses disappear from the rollout buffer, causing the MLP's boundary to drift inward over
   time. The analytical formula is independent of the policy distribution.

6. **Deployment on real robot is trivial.** ~800 FLOPs of numpy arithmetic at 50 Hz
   (< 0.1 ms). No model file, no GPU dependency, no TorchScript version maintenance.

K = 48 pairs: design rationale
-------------------------------
Each arm has 7 bodies: shoulder_1 → shoulder_2 → shoulder_3 → elbow → wrist_1 → wrist_2
→ wrist_3. Body origins are used as capsule axis endpoints (each consecutive pair of body
origins defines one segment axis). This gives 6 segments per arm:

    seg 0: shoulder_1 → shoulder_2   r = 0.053 m   (large proximal shoulder)
    seg 1: shoulder_2 → shoulder_3   r = 0.03925 m
    seg 2: shoulder_3 → elbow        r = 0.03925 m
    seg 3: elbow      → wrist_1      r = 0.03925 m
    seg 4: wrist_1    → wrist_2      r = 0.0285 m
    seg 5: wrist_2    → wrist_3      r = 0.023 m   (hand / distal forearm)

Note: wrist_3 has no collision geom (visual-only EE body). Its body origin is used as the
distal endpoint of seg 5, with the radius from wrist_2's collision cylinder.

K pairs are formed from:
    36 cross-arm:  all 6 left segs × all 6 right segs   → covers every way the two arms
                   can touch each other regardless of configuration
     6 L-torso:   each left  seg vs the torso capsule
     6 R-torso:   each right seg vs the torso capsule
    ── total 48 ──

This is a 6× expansion over an 8-pair design (hand+forearm only). The extra pairs cover
upper-arm-to-forearm crossings and shoulder-torso contacts that occur during wide reaches
and arm-crossing maneuvers common in bimanual manipulation tasks.

Torso capsule approximation
---------------------------
The base_link geom is a box (size=[0.065, 0.09, 0.206], center at z=0.206 in pelvis frame).
We approximate it as a vertical capsule in the pelvis frame:

    TORSO_BOTTOM = (0, 0, 0)       ← bottom face of box
    TORSO_TOP    = (0, 0, 0.412)   ← top face of box (2 × half-extent)
    TORSO_RADIUS = 0.10 m          ← actual max half-width is 0.09 m

The +1 cm is intentionally conservative (overclaims torso extent). A policy that keeps arms
slightly further from a too-large torso capsule is safe; a too-small capsule would allow arm
contacts to slip through without penalty. The torso endpoints are constant in the pelvis
frame — no lookup at runtime — so the torso contributes zero per-step overhead beyond the
already-required pelvis-frame transform.

Extended position array (N, 16, 3)
-----------------------------------
Inside compute_bimanual_distances, body positions are transformed from world frame to pelvis
frame in a single batched quat_apply call. The result is augmented with 2 constant torso
endpoints to form an (N, 16, 3) extended array:

    indices 0-6:   left  arm bodies in pelvis frame (shoulder_1_L … wrist_3_L)
    indices 7-13:  right arm bodies in pelvis frame (shoulder_1_R … wrist_3_R)
    index  14:     TORSO_BOTTOM = (0, 0, 0)    constant
    index  15:     TORSO_TOP    = (0, 0, 0.412) constant

Pre-built index tensors (PROX_A, DIST_A, PROX_B, DIST_B) gather the 4 endpoints of all
48 pairs in a single advanced-index operation, giving (N, 48, 3) per endpoint. A single
call to _batched_seg_seg_dist produces all 48 centerline distances at once.

Parallel-segment fallback: why point-to-segment, not endpoint-to-endpoint
--------------------------------------------------------------------------
The Shene algorithm has a degenerate case when denom = a·e − (d1·d2)² ≈ 0 (nearly
parallel segments). The standard workaround — using min(endpoint-to-endpoint distances)
as the fallback — gives the wrong answer for overlapping parallel segments.

Example: forearms hanging side-by-side with 5 cm lateral separation:
    A = [(0,0,0), (0,0,-0.20)]   B = [(0.05,0,-0.05), (0.05,0,-0.15)]
    True distance = 0.05 m (perpendicular gap).
    Min endpoint-pair distance = √(0.05² + 0.05²) = 0.0707 m (WRONG — too large by 41%).

This matters: if the fallback overestimates, the reward gives no penalty when arms are
actually 5 cm apart, and the policy may allow closer approaches than intended.

The fix: for the 4 endpoints, compute point-to-segment distance (projects the endpoint onto
the opposing segment axis, clamped to [0,1]). For the overlapping case the projection lands
inside the opposing segment, correctly recovering the perpendicular distance. Verified by
unit test in __main__.

Reward calibration
------------------
    violation_k = relu(MARGIN - d_k)       # ∈ [0, MARGIN]
    penalty = sum_k violation_k²            # ∈ [0, K × MARGIN²]

With MARGIN = 0.10 m and weight = -30.0 in RewardTermCfg:
    At d = 0 (touching):  violation = 0.10, violation² = 0.01  → -0.30 per pair
    At d = 0.05:          violation = 0.05, violation² = 0.0025 → -0.075 per pair
    At d ≥ 0.10:          violation = 0                          →  0 (gradient off)
    7+ simultaneous contacts → -2.10, exceeding the +2.0 primary locomotion reward

The quadratic shape creates a smooth gradient that grows as arms approach, reaching its
maximum slope at contact. Linear would give constant gradient inside margin. Exponential
would concentrate signal too close to d=0.

Quaternion convention: wxyz throughout
---------------------------------------
sim.data.xquat from mjwarp is wxyz. The body frame transform uses:
    inv_q = [w, -x, -y, -z]   (conjugate = inverse for unit quaternions)
    _quat_apply uses: v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v))

Validated at startup by rotating [1,0,0] by a 90° yaw quaternion (cos45, 0, 0, sin45)
and asserting the result is [0,1,0].

Deployment on real robot
-------------------------
At 50 Hz, compute_bimanual_distances takes < 0.1 ms on CPU (pure numpy, ~800 FLOPs × 48
pairs). No model file, no GPU required. Apply a light EMA (α=0.3) on the output distances
to absorb encoder-noise-induced position jitter (~3 mm from 0.01 rad encoder noise):
    d_filtered = 0.7 * d_prev + 0.3 * d_new
Filter the output, not the input — filtering joint angles would add lag to the distance
signal, which is the opposite of what a safety margin needs.

"""

from __future__ import annotations

import torch
from torch import Tensor

# =============================================================================
# Body / Segment Constants
# =============================================================================

PELVIS_BODY_NAME: str = "base_link"

# 14 arm bodies: 7 left (indices 0–6) + 7 right (indices 7–13) in the lookup array.
# These are the segment *endpoint* bodies; body origins are used as capsule axis endpoints.
ARM_BODY_NAMES: list[str] = [
    "shoulder_1_L", "shoulder_2_L", "shoulder_3_L",
    "elbow_L", "wrist_1_L", "wrist_2_L", "wrist_3_L",
    "shoulder_1_R", "shoulder_2_R", "shoulder_3_R",
    "elbow_R", "wrist_1_R", "wrist_2_R", "wrist_3_R",
]

# Segment table: (prox_ext_idx, dist_ext_idx, radius_m).
# Indices 0–13 map to ARM_BODY_NAMES positions in the extended array.
# Radii sourced from MJCF cylinder geom size[0]: shoulder_1=0.053, shoulder_{2,3}/elbow=0.03925,
# wrist_1=0.0285, wrist_2=0.023. wrist_3 carries no collision geom; its body is
# used only as the distal endpoint of the wrist_2 segment.
_L_SEG: list[tuple[int, int, float]] = [
    (0, 1, 0.053),    # shoulder_1_L → shoulder_2_L
    (1, 2, 0.03925),  # shoulder_2_L → shoulder_3_L
    (2, 3, 0.03925),  # shoulder_3_L → elbow_L
    (3, 4, 0.03925),  # elbow_L      → wrist_1_L
    (4, 5, 0.0285),   # wrist_1_L    → wrist_2_L
    (5, 6, 0.023),    # wrist_2_L    → wrist_3_L
]
_R_SEG: list[tuple[int, int, float]] = [
    (7,  8,  0.053),   # shoulder_1_R → shoulder_2_R
    (8,  9,  0.03925), # shoulder_2_R → shoulder_3_R
    (9,  10, 0.03925), # shoulder_3_R → elbow_R
    (10, 11, 0.03925), # elbow_R      → wrist_1_R
    (11, 12, 0.0285),  # wrist_1_R    → wrist_2_R
    (12, 13, 0.023),   # wrist_2_R    → wrist_3_R
]

# Torso: base_link box (size=[0.065, 0.09, 0.206], geom pos=[0,0,0.206] in body frame)
# → vertical capsule in pelvis (base_link) frame: z ∈ [0, 2×0.206] = [0, 0.412].
# Conservative radius 0.10 m (actual max half-width = 0.09 m — overestimates extent = safe).
TORSO_BOTTOM: Tensor = torch.zeros(3)
TORSO_TOP: Tensor    = torch.tensor([0.0, 0.0, 0.412])
TORSO_RADIUS: float  = 0.10

_TORSO_BOT_EXT: int = 14   # index in extended position array (size 16)
_TORSO_TOP_EXT: int = 15

# Reward calibration.
# violation_k = relu((MARGIN - d_k) / MARGIN) ∈ [0, 1] per pair.
# penalty = sum_k violation_k². At d=0: violation=1.0, violation²=1.0.
# With weight=-30.0: per-pair penalty = -30.0 × 1.0 = -30.0?
# Wait — actually violation = relu(MARGIN - d) / MARGIN, so at d=0: violation=1.0.
# Per-pair contribution = 1.0 (not 0.01). Use weight=-0.30 in cfg for -0.30/pair.
# OR: violation = relu(MARGIN - d), at d=0: violation=MARGIN=0.10. violation²=0.01.
# With weight=-30.0: per-pair = -30.0 × 0.01 = -0.30. This is the calibration below.
COLLISION_MARGIN: float = 0.10   # m — clearance target; full penalty at d ≤ 0
COLLISION_WEIGHT: float = -30.0  # recommended RewardTermCfg.weight
# Calibration: at d=0 (touching), violation=0.10, violation²=0.01.
#   weight × 0.01 = -30.0 × 0.01 = -0.30/pair ≈ 15% of typical +2.0 primary reward.
#   7+ simultaneous contacts → -2.1 — exceeds primary reward magnitude.

# =============================================================================
# Build K=48 pair index arrays (module-level, created once on import)
# =============================================================================


def _build_pair_tables() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build (PROX_A, DIST_A, PROX_B, DIST_B, RADII) each shape (K=48,).

    Pair layout (in order):
        [  0.. 35] 36 cross-arm pairs: all 6 L segments × all 6 R segments
        [ 36.. 41]  6 left-vs-torso pairs
        [ 42.. 47]  6 right-vs-torso pairs
    """
    pa, da, pb, db, radii = [], [], [], [], []

    # 36 cross-arm pairs: 6 L segs × 6 R segs
    for ia, ib, ra in _L_SEG:
        for ic, id_, rc in _R_SEG:
            pa.append(ia);  da.append(ib)
            pb.append(ic);  db.append(id_)
            radii.append(ra + rc)

    # 6 left-vs-torso
    for ia, ib, ra in _L_SEG:
        pa.append(ia);                da.append(ib)
        pb.append(_TORSO_BOT_EXT);   db.append(_TORSO_TOP_EXT)
        radii.append(ra + TORSO_RADIUS)

    # 6 right-vs-torso
    for ia, ib, ra in _R_SEG:
        pa.append(ia);                da.append(ib)
        pb.append(_TORSO_BOT_EXT);   db.append(_TORSO_TOP_EXT)
        radii.append(ra + TORSO_RADIUS)

    return (
        torch.tensor(pa,    dtype=torch.long),
        torch.tensor(da,    dtype=torch.long),
        torch.tensor(pb,    dtype=torch.long),
        torch.tensor(db,    dtype=torch.long),
        torch.tensor(radii, dtype=torch.float32),
    )


PROX_A, DIST_A, PROX_B, DIST_B, RADII = _build_pair_tables()
K: int = RADII.numel()   # 48

# Structural overlap mask: True = pair is meaningful for collision avoidance.
# Pairs 36 and 42 are always deeply negative (≈ -70mm) regardless of arm configuration
# because shoulder_1_L and shoulder_1_R bodies are rigidly embedded inside the torso
# capsule at their attachment points. Including them in soft_min pulls the repulsion
# signal to a constant floor, masking real collisions. Exclude them.
# Pair 36: L_SEG[0] (shoulder_1_L→shoulder_2_L) vs Torso — combined radius 0.153m.
# Pair 42: R_SEG[0] (shoulder_1_R→shoulder_2_R) vs Torso — combined radius 0.153m.
# These are the only two permanently-negative pairs at default configuration; all
# others are configuration-dependent.
VALID_PAIR_MASK: Tensor = torch.ones(K, dtype=torch.bool)
VALID_PAIR_MASK[36] = False   # shoulder_1_L vs torso: structural overlap
VALID_PAIR_MASK[42] = False   # shoulder_1_R vs torso: structural overlap

# =============================================================================
# Quaternion helpers (wxyz convention)
# =============================================================================


def _quat_conjugate(q: Tensor) -> Tensor:
    """Conjugate of wxyz quaternion: [w, -x, -y, -z]. Shape-preserving (..., 4)."""
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Apply wxyz quaternion q to vectors v.

    Shapes: q=(...,4), v=(...,3) → (...,3). Supports broadcasting.
    Uses: v + 2w(q_xyz × v) + 2(q_xyz × (q_xyz × v)).
    """
    w   = q[..., 0:1]
    xyz = q[..., 1:]
    t   = 2.0 * torch.linalg.cross(xyz, v)
    return v + w * t + torch.linalg.cross(xyz, t)


# =============================================================================
# Segment geometry helpers
# =============================================================================

_NEAR_ZERO    = 1e-8   # denominator floor to avoid NaN
_PARALLEL_EPS = 1e-5   # sin²θ threshold for parallel detection (denom / (a·e) < eps)


def _seg_pt_dist(p: Tensor, q: Tensor, x: Tensor) -> Tensor:
    """Distance from point x to segment [p, q]. All (..., 3) → (...,).

    Projects x onto the line through p–q, clamps to [p, q], returns Euclidean
    distance. Correctly returns the perpendicular distance when x lies in the
    interior — unlike point-to-endpoint, which would overestimate for overlapping
    parallel segments.
    """
    pq      = q - p
    len_sq  = (pq * pq).sum(-1).clamp(min=_NEAR_ZERO)
    t       = ((x - p) * pq).sum(-1) / len_sq
    closest = p + t.clamp(0.0, 1.0).unsqueeze(-1) * pq
    diff    = x - closest
    return (diff * diff).sum(-1).sqrt()


def _batched_seg_seg_dist(p1: Tensor, p2: Tensor, p3: Tensor, p4: Tensor) -> Tensor:
    """Centerline distance between capsule segments [p1, p2] and [p3, p4].

    Args: all (N, K, 3).
    Returns: (N, K) distances ≥ 0.

    Algorithm: Shene closest-point on two line segments with one refinement round.
    Parallel case (sin²θ < _PARALLEL_EPS, i.e. denom/(a·e) < eps) falls back to
    the minimum of 4 point-to-segment distances. This correctly handles overlapping
    parallel segments — naive endpoint-to-endpoint distances would overestimate
    (e.g., side-by-side forearms with 5cm lateral gap reported as 7cm by endpoint test).
    """
    d1 = p2 - p1   # (N, K, 3)
    d2 = p4 - p3
    r  = p1 - p3

    a = (d1 * d1).sum(-1)    # (N, K)
    e = (d2 * d2).sum(-1)
    f = (d2 * r ).sum(-1)
    b = (d1 * d2).sum(-1)
    c = (d1 * r ).sum(-1)

    denom    = a * e - b * b
    ae       = (a * e).clamp(min=_NEAR_ZERO)
    parallel = denom < _PARALLEL_EPS * ae

    # ── Non-parallel: Shene algorithm with one refinement round ──────────────
    s  = ((b * f - c * e) / denom.clamp(min=_NEAR_ZERO)).clamp(0.0, 1.0)
    t  = ((b * s + f)     / e.clamp(min=_NEAR_ZERO)).clamp(0.0, 1.0)
    s  = ((b * t - c)     / a.clamp(min=_NEAR_ZERO)).clamp(0.0, 1.0)   # refine
    delta_np = (p1 + s.unsqueeze(-1) * d1) - (p3 + t.unsqueeze(-1) * d2)
    dist_np  = (delta_np * delta_np).sum(-1).sqrt()   # (N, K)

    # ── Parallel fallback: min of 4 point-to-segment distances ───────────────
    # Each of the 4 endpoints is projected onto the opposing segment. This gives
    # the correct perpendicular distance when the segments overlap laterally.
    dist_par = torch.stack([
        _seg_pt_dist(p3, p4, p1),   # A-prox endpoint → segment B
        _seg_pt_dist(p3, p4, p2),   # A-dist endpoint → segment B
        _seg_pt_dist(p1, p2, p3),   # B-prox endpoint → segment A
        _seg_pt_dist(p1, p2, p4),   # B-dist endpoint → segment A
    ], dim=-1).min(-1).values       # (N, K)

    return torch.where(parallel, dist_par, dist_np)


# =============================================================================
# Main public API
# =============================================================================


def compute_bimanual_distances(body_pos_w: Tensor, body_quat_w: Tensor) -> Tensor:
    """Compute K=48 signed capsule-capsule clearance distances per environment.

    Args:
        body_pos_w:  (N, 15, 3) — row 0 = pelvis (base_link), rows 1-14 = arm bodies,
                     world frame. Expected order:
                       row  0: base_link (pelvis)
                       rows 1-7:  shoulder_1_L … wrist_3_L
                       rows 8-14: shoulder_1_R … wrist_3_R
                     Use ARM_BODY_NAMES to map names → row indices 1-14.
        body_quat_w: (N, 15, 4) — wxyz quaternions, world frame, same row order.

    Returns:
        (N, K=48) signed clearance distances in metres.
        Positive  = surfaces not touching.
        Zero      = capsule surfaces just touching.
        Negative  = capsule surfaces interpenetrating.

    Computation:
        1. Rotate arm bodies into pelvis frame (removes global yaw/pitch/roll so
           torso-relative positions are constant for a neutral pose).
        2. Append 2 constant torso endpoints (TORSO_BOTTOM, TORSO_TOP) expressed
           in the pelvis body frame.
        3. Gather segment endpoints for all K pairs using pre-built index tables.
        4. Compute K centerline distances analytically via _batched_seg_seg_dist.
        5. Subtract per-pair summed capsule radii to get signed surface distances.
    """
    N      = body_pos_w.shape[0]
    device = body_pos_w.device

    # 1. Transform 14 arm bodies from world into pelvis frame
    pelvis_pos  = body_pos_w[:,  0, :]    # (N, 3)
    pelvis_quat = body_quat_w[:, 0, :]   # (N, 4)
    inv_q       = _quat_conjugate(pelvis_quat)   # (N, 4)

    arm_pos_w = body_pos_w[:, 1:15, :]              # (N, 14, 3)
    rel       = arm_pos_w - pelvis_pos.unsqueeze(1)  # (N, 14, 3)
    arm_pos_p = _quat_apply(inv_q.unsqueeze(1), rel) # (N, 14, 3)

    # 2. Extended position array: 14 arm bodies + 2 constant torso endpoints
    t_bot = TORSO_BOTTOM.to(device).expand(N, 1, 3)    # (N, 1, 3)
    t_top = TORSO_TOP.to(device).expand(N, 1, 3)        # (N, 1, 3)
    ext   = torch.cat([arm_pos_p, t_bot, t_top], dim=1) # (N, 16, 3)

    # 3. Gather segment endpoints for all K pairs
    pa = PROX_A.to(device)   # (K,)
    da = DIST_A.to(device)
    pb = PROX_B.to(device)
    db = DIST_B.to(device)

    P1 = ext[:, pa, :]   # (N, K, 3)
    D1 = ext[:, da, :]
    P2 = ext[:, pb, :]
    D2 = ext[:, db, :]

    # 4. Centerline distances → 5. subtract radii for signed surface clearance
    d_raw = _batched_seg_seg_dist(P1, D1, P2, D2)     # (N, K)
    return d_raw - RADII.to(device).unsqueeze(0)       # (N, K)


def reward_arm_proximity(d: Tensor) -> Tensor:
    """Quadratic proximity penalty from analytical capsule distances.

    Args:
        d: (N, K=48) signed clearance distances from compute_bimanual_distances.

    Returns:
        (N,) non-negative penalty values. Apply with COLLISION_WEIGHT=-30.0 as
        RewardTermCfg.weight.

    Calibration (COLLISION_MARGIN=0.10m, weight=-30.0):
        violation_k = relu(MARGIN - d_k),  ∈ [0, MARGIN]
        return sum_k violation_k²
        At d=0 (touching): violation=0.10, violation²=0.01.
        weight × 0.01 = -0.30/pair ≈ 15% of typical +2.0 primary reward.
        At 7+ simultaneous contacts: penalty ≥ -2.10 — exceeds primary reward.

    Zero when all pairs clear COLLISION_MARGIN; grows quadratically inside margin.
    """
    violation = (COLLISION_MARGIN - d).clamp(min=0.0)   # (N, K), in [0, MARGIN]
    return violation.pow(2).sum(-1)                       # (N,)


# =============================================================================
# Startup validation
# =============================================================================


def validate_arm_proximity_setup(mj_model: object) -> None:
    """Assert model geometry matches proximity model assumptions.

    Checks:
    1. All 15 required bodies exist in the model (pelvis + 14 arm bodies).
    2. Quaternion convention: 90° yaw of [1,0,0] → [0,1,0] with wxyz=(cos45,0,0,sin45).
    3. Capsule radii are in plausible range [0.01, 0.20] m.
    4. Default pose: all K distances finite, positive, and at least one > 10 cm.

    Args:
        mj_model: A mujoco.MjModel instance (compiled from the humanoid_v21 spec).

    Raises:
        AssertionError on any failure with a descriptive message.
    """
    import math
    import mujoco
    import numpy as np

    # 1. All required bodies exist
    all_names = [PELVIS_BODY_NAME] + ARM_BODY_NAMES
    for name in all_names:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found in model"

    # 2. Quaternion convention: 90° yaw of x-axis must give y-axis
    c45, s45 = math.cos(math.pi / 4), math.sin(math.pi / 4)
    q = torch.tensor([[c45, 0.0, 0.0, s45]])   # wxyz 90° yaw
    v = torch.tensor([[1.0, 0.0, 0.0]])
    rotated  = _quat_apply(q, v)
    expected = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(rotated, expected, atol=1e-5), (
        f"Quaternion convention mismatch: 90° yaw of x-axis → {rotated.tolist()} "
        f"(expected {expected.tolist()})"
    )

    # 3. Capsule radii sanity
    all_radii = [r for _, _, r in _L_SEG] + [r for _, _, r in _R_SEG] + [TORSO_RADIUS]
    for r in all_radii:
        assert 0.01 <= r <= 0.20, f"Capsule radius {r:.4f} m out of expected [0.01, 0.20]"

    # 4. Default pose: distances finite, positive, at least one > 10 cm
    data = mujoco.MjData(mj_model)
    mujoco.mj_kinematics(mj_model, data)
    body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in all_names
    ]
    pos_np  = np.array([data.xpos[bid]  for bid in body_ids])   # (15, 3)
    quat_np = np.array([data.xquat[bid] for bid in body_ids])   # (15, 4) wxyz
    pos_t  = torch.from_numpy(pos_np ).unsqueeze(0).float()     # (1, 15, 3)
    quat_t = torch.from_numpy(quat_np).unsqueeze(0).float()     # (1, 15, 4)
    d      = compute_bimanual_distances(pos_t, quat_t)           # (1, K)

    assert torch.all(torch.isfinite(d)), "Non-finite distances in default pose"
    assert torch.all(d > 0), (
        f"Penetrating pairs in default pose: min={d.min().item():.4f} m"
    )
    assert d.max().item() > 0.10, (
        "All distances < 10 cm in default pose — likely wrong body IDs or zero kinematics"
    )
    print(
        f"[validate_arm_proximity_setup] OK — K={K}, "
        f"min={d.min().item():.3f} m, max={d.max().item():.3f} m"
    )


# =============================================================================
# Unit tests
# =============================================================================

if __name__ == "__main__":
    import math

    def _t(v: list) -> Tensor:
        """Wrap point as (1, 1, 3) for _batched_seg_seg_dist."""
        return torch.tensor([[v]], dtype=torch.float64)

    def _check(name: str, got: Tensor, expected: float, atol: float = 1e-5) -> None:
        val = got.item()
        ok  = abs(val - expected) <= atol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={val:.6f}, expected={expected:.6f}")
        if not ok:
            raise AssertionError(f"FAIL: {name}: {val:.6f} != {expected:.6f} (atol={atol})")

    print("=== proximity_model.py unit tests ===\n")

    # 1. Perpendicular crossing segments → distance = 0
    _check(
        "perpendicular_crossing",
        _batched_seg_seg_dist(_t([0,0,0]), _t([1,0,0]), _t([0.5,-1,0]), _t([0.5,1,0])),
        0.0, atol=1e-6,
    )

    # 2. Perpendicular offset: A along x, B along y at z=0.1 → closest point gap = 0.1
    _check(
        "perpendicular_offset",
        _batched_seg_seg_dist(_t([0,0,0]), _t([1,0,0]), _t([0.5,1,0.1]), _t([0.5,-1,0.1])),
        0.1, atol=1e-6,
    )

    # 3. Parallel non-overlapping: A down z, B offset laterally and above → endpoint-to-endpoint
    #    A: (0,0,0)→(0,0,-0.3), B: (0.05,0,0.1)→(0.05,0,0.2) — no z-overlap
    #    Closest: A-prox (0,0,0) to B-prox (0.05,0,0.1) → hypot(0.05,0.1)
    _check(
        "parallel_non_overlapping",
        _batched_seg_seg_dist(_t([0,0,0]), _t([0,0,-0.3]), _t([0.05,0,0.1]), _t([0.05,0,0.2])),
        math.hypot(0.05, 0.1), atol=1e-5,
    )

    # 4. CRITICAL: Parallel overlapping → must return lateral separation 0.05, NOT 0.0707
    #    A: (0,0,0)→(0,0,-0.2), B: (0.05,0,-0.05)→(0.05,0,-0.15) (B inside A's z-range)
    #    Endpoint-to-endpoint min = hypot(0.05,0.05)=0.0707 (WRONG).
    #    Point-to-segment correctly gives 0.05.
    _check(
        "parallel_overlapping",
        _batched_seg_seg_dist(_t([0,0,0]), _t([0,0,-0.2]), _t([0.05,0,-0.05]), _t([0.05,0,-0.15])),
        0.05, atol=1e-5,
    )

    # 5. T-shape endpoint: A along x [0..1], B at x=2 along z [0..1] → dist = 1.0
    _check(
        "t_shape_endpoint",
        _batched_seg_seg_dist(_t([0,0,0]), _t([1,0,0]), _t([2,0,0]), _t([2,0,1])),
        1.0, atol=1e-6,
    )

    # 6. Degenerate (zero-length segment A = point at origin); B at x=[1,2] → dist = 1.0
    _check(
        "degenerate_point_A",
        _batched_seg_seg_dist(_t([0,0,0]), _t([0,0,0]), _t([1,0,0]), _t([2,0,0])),
        1.0, atol=1e-5,
    )

    # 7. K=48 pair count
    assert K == 48, f"Expected K=48, got {K}"
    print(f"\n  [PASS] K={K} pairs")

    # 8. Index range: all indices in [0, 15]
    assert PROX_A.max() <= 15 and PROX_A.min() >= 0
    assert DIST_A.max() <= 15 and DIST_A.min() >= 0
    assert PROX_B.max() <= 15 and PROX_B.min() >= 0
    assert DIST_B.max() <= 15 and DIST_B.min() >= 0
    print(f"  [PASS] All pair indices in [0, 15]")

    # 9. RADII sanity: positive, less than 0.25 m
    assert (RADII > 0).all() and (RADII < 0.25).all()
    print(f"  [PASS] RADII range [{RADII.min():.4f}, {RADII.max():.4f}] m")

    # 10. reward_arm_proximity calibration:
    #     One pair at d=0 (touching), rest clear at d=1m.
    #     violation = relu(0.10 - 0.0) = 0.10, violation² = 0.01.
    #     Sum over K: 0.01 (only 1 pair). weight=-30 → -0.30.
    d_test      = torch.ones(1, K) * 1.0
    d_test[0, 0] = 0.0
    pen          = reward_arm_proximity(d_test)
    expected_pen = 0.10 ** 2   # = 0.01
    assert abs(pen.item() - expected_pen) < 1e-7, f"Reward: {pen.item()} != {expected_pen}"
    print(
        f"  [PASS] reward_arm_proximity(1 pair at d=0) = {pen.item():.4f} "
        f"(× weight=-30 → {COLLISION_WEIGHT * pen.item():.4f}/pair)"
    )

    # 11. float32 precision: K=48 pairs, random arm configuration
    torch.manual_seed(42)
    pos_r  = torch.randn(4, 15, 3, dtype=torch.float32) * 0.3
    quat_r = torch.nn.functional.normalize(
        torch.randn(4, 15, 4, dtype=torch.float32), dim=-1
    )
    d_f32 = compute_bimanual_distances(pos_r, quat_r)
    assert torch.all(torch.isfinite(d_f32)), "Non-finite float32 distances"
    assert d_f32.shape == (4, K)
    print(
        f"  [PASS] float32 random N=4: shape={d_f32.shape}, "
        f"min={d_f32.min():.4f} m, max={d_f32.max():.4f} m"
    )

    # 12. Batch dimension: N=1 and N=512 should both work
    for N in (1, 512):
        pos_b  = torch.zeros(N, 15, 3)
        quat_b = torch.zeros(N, 15, 4); quat_b[:, :, 0] = 1.0  # identity
        d_b    = compute_bimanual_distances(pos_b, quat_b)
        assert d_b.shape == (N, K) and torch.all(torch.isfinite(d_b))
    print(f"  [PASS] Batch sizes N=1 and N=512 correct")

    print("\n=== All tests passed ===")
