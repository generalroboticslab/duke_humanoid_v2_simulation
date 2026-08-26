"""Analytic capsule occlusion for active-vision detection (A4+, plan/ACTIVE_VISION_PICK_PLACE_PLAN.md).

FAST replacement for the ``mujoco_warp.rays`` raycast (`camera_occlusion.py`). Tests each
(camera, target) line of sight against a handful of CAPSULES that approximate the robot's own body
(torso + upper-arms + forearms) in pure batched torch — no warp, no kernel launch, no torch↔warp
boundary. A camera scores a target (A4 coverage reward) only if it both FRAMES it (`fov_detect`) AND
has clear line of sight, so v83's moving arms/torso self-occlude targets and force the cameras to aim
where they can see.

Why capsules over the raycast (profiled 2026-06-24, N=4096): the cost of `mujoco_warp.rays` is the
KERNEL itself (16384 rays × 119 scene geoms, nearest-hit) — 28.9 ms/step vs ~26 µs total for the
torch↔warp conversion/alloc, i.e. 2.5× training wall-clock (64.4k→25.4k fps). The ONLY occluder in
A4–A6 is the robot's own body (no external geometry until Stage B), so ~5 capsules capture it; the
test is then nrays × ncapsule closed-form segment–segment distances, ~170× cheaper and in-graph.
Approximation, not the mesh: a binary credit gate tolerates limb-edge slop; capsule RADII are fit
(`smoke_occlusion_capsule.py`) to reproduce the raycast's per-target clear-fraction statistics.

Occlusion window (same optics rule as the raycast): a capsule blocks a ray only if the closest
approach lies strictly between the NEAR plane and the target — ``s ∈ (near/range, 1-EPS)`` along the
cam→target segment. ``s ≤ near/range`` = inside the near plane (nothing there is "seen" anyway);
``s ≥ 1-EPS`` = at the target (EPS pull-in so a capsule sitting AT the target does not self-block,
matching the raycast's ``_EPS``).
"""

from __future__ import annotations

import torch

_EPS = 0.02  # along-ray pull-in at the target end (matches camera_occlusion._EPS)


def _seg_seg_closest_s_dist(p1, d1, a, p2, d2):
    """Closest-approach param on segment 1 and the segment–segment distance (clamped, vectorised).

    Standard one-iteration clamped solution (Ericson, Real-Time Collision Detection §5.1.9).
    Segment 1 = ``p1 + s·d1`` (s∈[0,1]), segment 2 = ``p2 + t·d2`` (t∈[0,1]).

    Args:
        p1, d1: (..., 3) start + (end-start) of segment 1 (the cam→target ray).
        a:      (...)    ``dot(d1, d1)`` (passed in: caller already has range²).
        p2, d2: (..., 3) start + (end-start) of segment 2 (the capsule axis).
    Returns:
        s:    (...) closest-approach parameter along segment 1, in [0, 1].
        dist: (...) Euclidean distance between the two segments.
    """
    r = p1 - p2
    e = (d2 * d2).sum(-1)                      # |d2|²  (capsule length², > 0)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = a * e - b * b                       # ≥ 0; 0 when the segments are parallel
    s = ((b * f - c * e) / denom.clamp_min(1e-9)).clamp(0.0, 1.0)
    t = ((b * s + f) / e.clamp_min(1e-9)).clamp(0.0, 1.0)
    s = ((t * b - c) / a.clamp_min(1e-9)).clamp(0.0, 1.0)
    closest = (p1 + s.unsqueeze(-1) * d1) - (p2 + t.unsqueeze(-1) * d2)
    return s, closest.norm(dim=-1)


def cast_unoccluded_capsule(cam_pos, targets, caps_p, caps_q, caps_r, near, ground_z=None):
    """Per (camera, target) clear-line-of-sight test vs self-body capsules + the ground plane.

    Occluders: (1) self-body capsules (torso/arms — self-occlusion, the A4 premise), and (2) the
    flat ground plane z=``ground_z``. The ground DOMINATES here (measured 2026-06-24: ~99% of raycast
    occlusions are terrain): full-sphere targets sampled below the root sit at/under the floor, and a
    camera cannot see through it. Both must be modelled to match the raycast ground truth.

    Args:
        cam_pos: (N, ncam, 3) camera optical-site world positions.
        targets: (N, ntgt, 3) world target positions.
        caps_p:  (N, C, 3) capsule axis start (world).
        caps_q:  (N, C, 3) capsule axis end   (world).
        caps_r:  (C,)      capsule radii.
        near:    near-clip distance (m); blockers closer than this along the ray are ignored.
        ground_z: flat-ground world height; ``None`` disables the ground term.
    Returns:
        (N, ncam, ntgt) bool, True = unoccluded line of sight.
    """
    n, ncam, _ = cam_pos.shape
    ntgt = targets.shape[1]
    nray = ncam * ntgt
    o = cam_pos.unsqueeze(2).expand(n, ncam, ntgt, 3)
    delta = targets.unsqueeze(1).expand(n, ncam, ntgt, 3) - o     # cam→target
    rng = delta.norm(dim=-1)                                      # (N, ncam, ntgt)

    p1 = o.reshape(n, nray, 1, 3)
    d1 = delta.reshape(n, nray, 1, 3)
    a = (rng * rng).reshape(n, nray, 1)                           # |d1|² = range²
    p2 = caps_p.unsqueeze(1)                                      # (N, 1, C, 3)
    d2 = (caps_q - caps_p).unsqueeze(1)                           # (N, 1, C, 3)

    s, dist = _seg_seg_closest_s_dist(p1, d1, a, p2, d2)          # (N, nray, C)
    s_near = (near / rng.reshape(n, nray, 1)).clamp(max=1.0)      # near plane as a ray fraction
    blocked = (dist < caps_r.view(1, 1, -1)) & (s > s_near) & (s < 1.0 - _EPS)
    occluded = blocked.any(dim=2)                                # (N, nray)

    if ground_z is not None:
        oz = o.reshape(n, nray, 3)[..., 2]
        dz = delta.reshape(n, nray, 3)[..., 2]
        sg = (ground_z - oz) / dz.clamp_max(-1e-6)               # descending crossing param (dz<0)
        sn = (near / rng.reshape(n, nray)).clamp(max=1.0)
        hit_ground = (dz < 0.0) & (sg > sn) & (sg < 1.0 - _EPS)  # floor crossed before the target
        occluded = occluded | hit_ground

    return (~occluded).reshape(n, ncam, ntgt)
