#!/usr/bin/env python3
"""
Visualize replacing collision geometry (box, cylinder, mesh) with capsules/spheres.

WHY CAPSULES?
  Capsules and spheres are the fastest collision primitives in MuJoCo/mjwarp:
  - O(1) narrow-phase test  (vs O(faces) for meshes)
  - Fewer contacts per pair  → fewer solver iterations
  - GPU-friendly             — no branching over triangle sets

ALGORITHMS  (one per shape type)
  capsules_from_box(box_size)
      Longest axis → capsule axis, shortest → radius, middle → row packing.
      Sphere fallback for near-cubic boxes (aspect ratio < 1.3).

  capsules_from_cylinder(radius, height)
      Single inscribed capsule (or sphere if height < 2·radius).

  capsules_from_mesh_coverage(mesh)
      Greedy set-cover with medial-axis-guided candidates:
        1. Simplify mesh  (quadric decimation)
        2. Sample interior medial-axis points  (voxelised EDT, scipy)
        3. Build capsule candidates  (spheres + random segment pairs)
        4. Compute distance matrix  (fully vectorised, no Python loops)
        5. Greedy set-cover  → pick capsule covering most uncovered points

  capsules_from_coacd(mesh)    ← BEST FOR REAL ROBOT MESHES
      CoACD (SIGGRAPH 2022) → convex decomposition → one capsule per hull.
      Uses Vt[0] (longest PCA axis) as capsule direction per hull.
      Results cached to disk (xxhash key) so reruns skip the slow MCTS solver.

QUALITY METRICS  (printed per example)
  volume_ratio   : capsule_volume / reference_volume     (target: 0.6–1.0)
  hausdorff_norm : max(surface→capsule gap) / diagonal   (target: < 0.10)
  near_coverage  : frac of surface pts within 5% of diagonal  (target: > 0.85)

USAGE
  python visualize_capsule_replacement.py
  Output: capsule_visualization.png
"""
import os
import pickle

import xxhash
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from scipy.ndimage import distance_transform_edt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import coacd
    HAS_COACD = True
except ImportError:
    HAS_COACD = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _distance_matrix(points, p1s, p2s):
    """Distance from every point to every line segment.  Fully vectorised.

    Given N points and M segments, computes all N×M distances in one shot
    using (N,1,3) vs (1,M,3) broadcasting.  No Python loops.

    Args:
        points : (N, 3)  query points
        p1s    : (M, 3)  segment start points
        p2s    : (M, 3)  segment end points

    Returns: (N, M) float32 distance matrix
    """
    # segment vectors and squared lengths
    v = p2s - p1s                                           # (M, 3)
    L2 = np.sum(v * v, axis=1)                              # (M,)

    # vector from each segment start to each point
    diff = points[:, None, :] - p1s[None, :, :]             # (N, M, 3)

    # project each point onto each segment, clamp to [0, 1]
    t = np.sum(diff * v[None, :, :], axis=2)                # (N, M)
    t = np.clip(t / np.maximum(L2[None, :], 1e-12), 0., 1.)

    # closest point on each segment
    proj = p1s[None, :, :] + t[:, :, None] * v[None, :, :]  # (N, M, 3)

    return np.linalg.norm(points[:, None, :] - proj, axis=2).astype(np.float32)


def _capsule_volume(p1, p2, r):
    """Volume of one capsule = cylinder + sphere (two hemicaps)."""
    return np.pi * r**2 * np.linalg.norm(p2 - p1) + (4 / 3) * np.pi * r**3


def _union_volume(capsules):
    """Sum of individual capsule volumes (upper bound — ignores overlaps)."""
    return sum(_capsule_volume(c['p1'], c['p2'], c['radius']) for c in capsules)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(capsules, surface_points,
                             reference_volume=None, mesh_diagonal=None):
    """Measure how well capsules approximate the original shape.

    For each surface sample point, computes signed distance to the nearest
    capsule (positive = outside, negative = inside).  Then derives:

      volume_ratio  : total capsule volume / reference volume
      hausdorff_norm: worst-case surface gap, normalised by mesh diagonal
      near_coverage : fraction of surface within 5% of diagonal

    Args:
        capsules        : list of {'p1': (3,), 'p2': (3,), 'radius': float}
        surface_points  : (N, 3) ground-truth surface samples
        reference_volume: mesh or box volume (None → skip volume_ratio)
        mesh_diagonal   : bounding-box diagonal length (None → skip hausdorff)

    Returns: dict with volume_ratio, hausdorff_norm, near_coverage
    """
    N = len(surface_points)
    if N == 0 or len(capsules) == 0:
        return dict(volume_ratio=None, hausdorff_norm=None, near_coverage=0.0)

    # Stack capsule endpoints into (M, 3) arrays for batch distance
    p1s = np.array([c['p1'] for c in capsules])
    p2s = np.array([c['p2'] for c in capsules])
    radii = np.array([c['radius'] for c in capsules])

    # (N, M) distances, then subtract radii → signed distance per capsule
    dists = _distance_matrix(surface_points, p1s, p2s)     # (N, M)
    signed = dists - radii[None, :]                         # (N, M)
    min_dist = signed.min(axis=1)                           # (N,)  nearest capsule

    hausdorff_gap = float(np.maximum(min_dist, 0).max())
    hausdorff_norm = (hausdorff_gap / mesh_diagonal) if mesh_diagonal else None

    dilation = (mesh_diagonal * 0.05) if mesh_diagonal else 0.0
    near_coverage = float(np.mean(min_dist <= dilation))

    cap_vol = _union_volume(capsules)
    volume_ratio = (cap_vol / reference_volume) if reference_volume else None

    return dict(volume_ratio=volume_ratio, hausdorff_norm=hausdorff_norm,
                near_coverage=near_coverage)


# ---------------------------------------------------------------------------
# Cylinder → capsule (or sphere)
# ---------------------------------------------------------------------------

def capsules_from_cylinder(radius, height):
    """Inscribe a single capsule (or sphere) inside a cylinder.

    Perfect fit: capsule radius = cylinder radius, capsule length fills
    the remaining height after the two hemicaps.
    Sphere fallback for flat discs (height < 2·radius).

    Args:
        radius : cylinder radius (m)
        height : cylinder full height (m)

    Returns: dict(capsules, n)
    """
    if height < 2 * radius:
        cap = {'p1': np.zeros(3), 'p2': np.zeros(3), 'radius': radius}
    else:
        half_cyl = (height / 2) - radius
        cap = {'p1': np.array([0., 0., -half_cyl]),
               'p2': np.array([0., 0., +half_cyl]),
               'radius': radius}
    return dict(capsules=[cap], n=1)


# ---------------------------------------------------------------------------
# Box → capsules
# ---------------------------------------------------------------------------

def capsules_from_box(box_size, max_capsules=None):
    """Pack inscribed capsules into an axis-aligned box.

    Sort the 3 dimensions longest → shortest:
      longest  → capsule cylinder axis
      middle   → pack multiple capsules side by side
      shortest → capsule radius  (touches the two closest faces)

    Row count = ceil(middle / shortest).  Sphere fallback when the box is
    nearly cubic (max/min extent < 1.3).

    Args:
        box_size     : [lx, ly, lz] full extents in metres
        max_capsules : hard cap on number of capsules (optional)

    Returns: dict(capsules, axis_idx, arrange_idx, n)
    """
    sizes = np.array(box_size, dtype=float)

    # Near-cubic → single sphere
    if sizes.max() / sizes.min() < 1.3:
        r = sizes.min() / 2
        return dict(capsules=[{'p1': np.zeros(3), 'p2': np.zeros(3), 'radius': r}],
                    axis_idx=int(np.argmax(sizes)),
                    arrange_idx=int(np.argsort(sizes)[1]), n=1)

    long_i, mid_i, short_i = np.argsort(sizes)[::-1]
    sz, sy, sx = sizes[long_i], sizes[mid_i], sizes[short_i]

    r = sx / 2
    cyl_len = sz - sx  # cylindrical part (MuJoCo hemicaps add r at each end)
    n = max(1, int(np.ceil(sy / sx)))
    if max_capsules is not None:
        n = min(n, max_capsules)

    centres_1d = (np.array([0.0]) if n == 1
                  else np.linspace(-(sy / 2 - r), +(sy / 2 - r), n))

    capsules = []
    for c_val in centres_1d:
        center = np.zeros(3)
        center[mid_i] = c_val
        p1 = center.copy(); p1[long_i] -= cyl_len / 2
        p2 = center.copy(); p2[long_i] += cyl_len / 2
        capsules.append({'p1': p1, 'p2': p2, 'radius': r})

    return dict(capsules=capsules, axis_idx=int(long_i),
                arrange_idx=int(mid_i), n=n)


def _box_surface_samples(sizes, n=800):
    """Uniform random samples on all 6 faces of a box (for metric evaluation)."""
    h = np.array(sizes) / 2
    pts = []
    for dim in range(3):
        for sign in (-1, 1):
            face_pts = np.random.default_rng(dim * 2 + (sign + 1) // 2).uniform(
                -h, h, (n // 6, 3))
            face_pts[:, dim] = sign * h[dim]
            pts.append(face_pts)
    return np.vstack(pts)


# ---------------------------------------------------------------------------
# Mesh → capsules  (medial axis + vectorised set-cover)
# ---------------------------------------------------------------------------

def _sample_medial_axis(mesh, n_points=200, voxel_frac=0.05):
    """Find interior points near the mesh's medial axis (shape skeleton).

    The medial axis is the set of points equidistant to ≥2 surface patches
    — exactly where capsule endpoints should sit for good coverage.

    Primary method (needs scipy):
      1. Voxelise the mesh at ~5% of bounding-box diagonal pitch.
      2. Compute the Euclidean Distance Transform (EDT) on the voxel grid.
         High EDT = far from surface = on or near the medial axis.
      3. Rank voxels by EDT, pick top candidates with spatial diversity.

    Fallback (no scipy):
      Sample along the 3 PCA axes, keep only interior points.

    Args:
        mesh       : trimesh.Trimesh
        n_points   : target number of returned points
        voxel_frac : voxel pitch as fraction of bounding-box diagonal

    Returns: (K, 3) interior points near the medial axis
    """
    pitch = max(voxel_frac * np.linalg.norm(mesh.extents), 1e-3)

    if HAS_SCIPY:
        try:
            vox = mesh.voxelized(pitch=pitch).fill()
            edt = distance_transform_edt(vox.matrix.astype(np.float32))
            flat_idx = np.argsort(edt.ravel())[::-1]
            top_k = min(n_points * 10, len(flat_idx))
            chosen = flat_idx[np.linspace(0, top_k - 1, n_points, dtype=int)]
            ijk = np.array(np.unravel_index(chosen, edt.shape)).T
            return vox.origin + (ijk + 0.5) * pitch
        except Exception:
            pass  # fall through to PCA fallback

    # PCA fallback: stratified sampling along principal axes
    verts = mesh.vertices
    centroid = verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(verts - centroid, full_matrices=False)
    pts = []
    for axis in Vt:
        proj = (verts - centroid) @ axis
        for t in np.linspace(proj.min(), proj.max(), n_points // 3):
            pts.append(centroid + t * axis)
    candidates = np.array(pts)
    try:
        candidates = candidates[mesh.contains(candidates)]
    except Exception:
        pass
    if len(candidates) == 0:
        candidates = np.array([mesh.centroid])
    idx = np.linspace(0, len(candidates) - 1,
                      min(n_points, len(candidates)), dtype=int)
    return candidates[idx]


def capsules_from_mesh_coverage(mesh, max_capsules=5, num_surface=1500,
                                 num_inner=250, num_candidates=600,
                                 simplify_target=300, radius_percentile=3):
    """Approximate a mesh with ≤ max_capsules inscribed capsules via set-cover.

    Pipeline:
      1. Simplify mesh to ~300 faces  (speeds up all downstream ops)
      2. Sample surface points        (ground truth for coverage)
      3. Find medial-axis points      (skeleton of the shape)
      4. Build candidate capsules     (spheres at medial pts + random segments)
      5. Distance matrix              (fully vectorised via _distance_matrix)
      6. Greedy set-cover             (pick capsule covering most uncovered pts)

    The radius for each candidate = percentile(surface_dists, 3):
    a low percentile keeps the capsule inscribed while being robust to
    outlier vertices on the mesh surface.

    Args:
        mesh              : trimesh.Trimesh
        max_capsules      : max capsules to return
        num_surface       : number of surface sample points
        num_inner         : number of medial-axis interior points
        num_candidates    : number of random segment pairs to try
        simplify_target   : target face count after decimation
        radius_percentile : percentile of surface distances → capsule radius

    Returns: dict(capsules, n, metrics)
    """
    if not HAS_TRIMESH:
        raise RuntimeError("trimesh required for mesh conversion.")

    # 1. Simplify
    work = mesh
    if len(mesh.faces) > simplify_target:
        try:
            work = mesh.simplify_quadric_decimation(simplify_target)
        except Exception:
            pass

    # 2. Surface samples
    surface_pts, _ = trimesh.sample.sample_surface(work, num_surface)

    # 3. Medial-axis interior points
    inner_pts = _sample_medial_axis(work, n_points=num_inner)

    # 4. Build candidates — spheres (p1==p2) + random segment pairs
    cands_p1 = list(inner_pts)
    cands_p2 = list(inner_pts)
    if len(inner_pts) > 1:
        rng = np.random.default_rng(42)
        for _ in range(num_candidates):
            i1, i2 = rng.choice(len(inner_pts), 2, replace=False)
            cands_p1.append(inner_pts[i1])
            cands_p2.append(inner_pts[i2])
    p1s = np.array(cands_p1)  # (M, 3)
    p2s = np.array(cands_p2)

    # 5. Vectorised distance matrix — single call, no Python loop
    all_dists = _distance_matrix(surface_pts, p1s, p2s)     # (N, M)

    # Inscribed radius per candidate: low percentile = robust to outlier verts
    radii = np.percentile(all_dists, radius_percentile, axis=0).astype(np.float32)

    diag = float(np.linalg.norm(work.bounds[1] - work.bounds[0]))
    dilation = diag * 0.05
    D = all_dists <= (radii[None, :] + dilation)  # (N, M) bool coverage mask

    # 6. Greedy set-cover
    uncovered = np.ones(len(surface_pts), dtype=bool)
    selected = []
    for _ in range(max_capsules):
        if not uncovered.any():
            break
        scores = D[uncovered].sum(axis=0)
        best_j = int(scores.argmax())
        if scores[best_j] == 0:
            break
        selected.append(best_j)
        uncovered &= ~D[:, best_j]

    capsules = [{'p1': p1s[j], 'p2': p2s[j], 'radius': float(radii[j])}
                for j in selected]

    mesh_vol = float(work.volume) if work.is_watertight else None
    metrics = compute_quality_metrics(capsules, surface_pts,
                                      reference_volume=mesh_vol,
                                      mesh_diagonal=diag)
    return dict(capsules=capsules, n=len(capsules), metrics=metrics)


# ---------------------------------------------------------------------------
# CoACD disk cache  (skips the slow MCTS decomposition on reruns)
# ---------------------------------------------------------------------------

_COACD_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".coacd_cache")
_coacd_used_keys: set[str] = set()


def _coacd_cache_key(vertices, faces, threshold, max_convex_hull):
    """Hash mesh geometry + CoACD params → 16-char hex key for cache filename."""
    h = xxhash.xxh128()
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
    h.update(f"t={threshold},m={max_convex_hull}".encode())
    return h.hexdigest()[:16]


def _coacd_cached(mesh, threshold, max_convex_hull):
    """Run CoACD, caching results to .coacd_cache/<hash>.pkl.

    On cache hit:  loads pickle → instant.
    On cache miss: runs CoACD MCTS solver, saves result, returns hulls.
    Tracks which keys were used so _coacd_cache_cleanup() can purge stale files.

    Returns: list of (vertices, faces) tuples, one per convex hull.
    """
    key = _coacd_cache_key(mesh.vertices, mesh.faces, threshold, max_convex_hull)
    _coacd_used_keys.add(key)
    os.makedirs(_COACD_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_COACD_CACHE_DIR, f"{key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    coacd.set_log_level("error")
    c_mesh = coacd.Mesh(vertices=mesh.vertices.astype(np.float64),
                        indices=mesh.faces.astype(np.int32))
    hulls = coacd.run_coacd(c_mesh, threshold=threshold,
                             max_convex_hull=max_convex_hull, seed=0)
    hulls_np = [(np.array(v), np.array(f)) for v, f in hulls]
    with open(cache_path, "wb") as f:
        pickle.dump(hulls_np, f)
    return hulls_np


def _coacd_cache_cleanup():
    """Delete cached .pkl files that weren't used in this run (stale entries)."""
    if not os.path.isdir(_COACD_CACHE_DIR):
        return
    for fname in os.listdir(_COACD_CACHE_DIR):
        if fname.endswith(".pkl") and fname.removesuffix(".pkl") not in _coacd_used_keys:
            os.remove(os.path.join(_COACD_CACHE_DIR, fname))


# ---------------------------------------------------------------------------
# CoACD decomposition → capsules
# ---------------------------------------------------------------------------

def fit_capsule_to_hull(vertices, radius_percentile=3):
    """Fit an inscribed capsule to a convex hull along its longest axis.

    Uses PCA: Vt[0] = first principal component = direction of maximum
    variance = the longest axis of the point cloud.  This is the natural
    capsule direction for any shape:
      - Elongated (shank, arm link): Vt[0] aligns with the limb
      - Isotropic (sphere-like): all axes equivalent, any works

    The capsule radius is set to a low percentile of perpendicular distances
    so it stays inscribed while being robust to outlier hull vertices.

    If the hull is shorter than 2·radius along its longest axis (nearly
    spherical), the capsule degenerates to a sphere at the centroid.

    Args:
        vertices          : (N, 3) convex hull vertices
        radius_percentile : percentile of perp-distances → radius  (2–5)

    Returns: {'p1': (3,), 'p2': (3,), 'radius': float}
    """
    verts = np.asarray(vertices, dtype=float)
    centroid = verts.mean(axis=0)
    _, _, Vt = np.linalg.svd(verts - centroid, full_matrices=False)

    axis = Vt[0]
    t = (verts - centroid) @ axis
    t_min, t_max = float(t.min()), float(t.max())

    # Perpendicular distance from each vertex to the axis line
    on_axis = centroid + t[:, None] * axis
    perp = np.linalg.norm(verts - on_axis, axis=1)
    r = max(float(np.percentile(perp, radius_percentile)), 1e-3)

    # Pull endpoints inward by r so the hemicaps stay inside the hull
    t_p1, t_p2 = t_min + r, t_max - r

    if t_p2 < t_p1:
        # Hull too short for a capsule → degenerate to sphere
        r_s = (t_max - t_min) / 2.0
        mid = centroid + ((t_min + t_max) / 2.0) * axis
        return {'p1': mid.copy(), 'p2': mid.copy(), 'radius': r_s}

    return {'p1': centroid + t_p1 * axis,
            'p2': centroid + t_p2 * axis, 'radius': r}


def capsules_from_coacd(mesh, threshold=0.05, max_convex_hull=8):
    """Decompose mesh into convex hulls (CoACD), fit one capsule per hull.

    CoACD (SIGGRAPH 2022) finds a near-optimal convex decomposition using
    collision-aware concavity + MCTS search.  Each hull is then approximated
    by an inscribed capsule aligned to its longest PCA axis.

    Much better than greedy set-cover for complex/non-convex robot parts
    because CoACD respects the geometry's natural sub-parts.

    Args:
        mesh            : trimesh.Trimesh
        threshold       : concavity threshold (0.01=fine, 0.1=coarse)
        max_convex_hull : max hulls  (-1 = unlimited)

    Returns: dict(capsules, n, n_hulls, metrics)
    """
    if not HAS_COACD:
        raise RuntimeError("coacd is required: pip install coacd")
    if not HAS_TRIMESH:
        raise RuntimeError("trimesh is required for surface sampling.")

    hulls = _coacd_cached(mesh, threshold, max_convex_hull)
    capsules = [fit_capsule_to_hull(hull_verts) for hull_verts, _ in hulls]

    surface_pts, _ = trimesh.sample.sample_surface(mesh, 1500)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    mesh_vol = float(mesh.volume) if mesh.is_watertight else None
    metrics = compute_quality_metrics(capsules, surface_pts,
                                      reference_volume=mesh_vol,
                                      mesh_diagonal=diag)
    return dict(capsules=capsules, n=len(capsules),
                n_hulls=len(hulls), metrics=metrics)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_box_wireframe(ax, size, color='black', lw=1.8):
    """Draw 12 edges of an axis-aligned box."""
    h = np.array(size) / 2
    corners = np.array([[((-1)**((i >> k) & 1)) * h[k] for k in range(3)]
                        for i in range(8)])
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                ax.plot(*zip(corners[i], corners[j]), color=color, lw=lw)


def draw_cylinder_wireframe(ax, radius, height, color='black', lw=1.0):
    """Draw top/bottom circles + 8 vertical lines."""
    theta = np.linspace(0, 2 * np.pi, 40)
    for z in (-height / 2, height / 2):
        ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                np.full_like(theta, z), color=color, lw=lw)
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([radius * np.cos(t)] * 2, [radius * np.sin(t)] * 2,
                [-height / 2, height / 2], color=color, lw=lw)


def draw_mesh_wireframe(ax, mesh, color='black', alpha=0.3, max_edges=500):
    """Draw mesh edges as a single Line3DCollection (one draw call)."""
    edges = mesh.edges_unique
    if len(edges) > max_edges:
        edges = edges[np.random.default_rng(0).choice(len(edges), max_edges,
                                                       replace=False)]
    verts = mesh.vertices
    segments = verts[edges]  # (E, 2, 3)
    ax.add_collection3d(Line3DCollection(segments, colors=color,
                                          alpha=alpha, linewidths=0.5))


def draw_capsule(ax, p1, p2, radius, color, alpha=0.7):
    """Draw a capsule (cylinder + 2 hemicaps) at arbitrary orientation."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    v = p2 - p1
    length = np.linalg.norm(v)
    v_hat = v / length if length > 1e-6 else np.array([0., 0., 1.])

    # Rotation matrix from Z-axis to v_hat  (Rodrigues)
    z_ax = np.array([0., 0., 1.])
    k = np.cross(z_ax, v_hat)
    sin_a = np.linalg.norm(k)
    cos_a = float(np.dot(z_ax, v_hat))

    if sin_a > 1e-6:
        k /= sin_a
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    elif cos_a < 0:
        R = np.diag([1., -1., -1.])
    else:
        R = np.eye(3)

    N_theta, N_phi = 16, 8  # enough for visualisation, fast to render
    theta = np.linspace(0, 2 * np.pi, N_theta)

    def _surf(X, Y, Z):
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) @ R.T + p1
        ax.plot_surface(pts[:, 0].reshape(X.shape),
                        pts[:, 1].reshape(X.shape),
                        pts[:, 2].reshape(X.shape),
                        color=color, alpha=alpha, linewidth=0, antialiased=True)

    # Cylinder barrel
    s = np.linspace(0, length, 2)
    T, S = np.meshgrid(theta, s)
    _surf(radius * np.cos(T), radius * np.sin(T), S)

    # Hemicaps (top and bottom)
    phi = np.linspace(0, np.pi / 2, N_phi)
    T2, P = np.meshgrid(theta, phi)
    Xh = radius * np.sin(P) * np.cos(T2)
    Yh = radius * np.sin(P) * np.sin(T2)
    _surf(Xh, Yh,  length + radius * np.cos(P))
    _surf(Xh, Yh, -radius * np.cos(P))


def draw_scene(ax, obj, p, obj_type='box'):
    """Render capsules overlaid on the original shape wireframe."""
    for i, c in enumerate(p['capsules']):
        draw_capsule(ax, c['p1'], c['p2'], c['radius'],
                     color=CAPSULE_COLORS[i % len(CAPSULE_COLORS)])

    if obj_type == 'box':
        draw_box_wireframe(ax, obj)
        half = max(obj) * 0.8
        center = np.zeros(3)
    elif obj_type == 'cylinder':
        r, h = obj
        draw_cylinder_wireframe(ax, r, h)
        half = max(r, h / 2) * 1.5
        center = np.zeros(3)
    else:  # mesh
        draw_mesh_wireframe(ax, obj)
        bounds = obj.bounds
        half = np.max(bounds[1] - bounds[0]) * 0.7
        center = obj.centroid

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')


# ---------------------------------------------------------------------------
# Examples registry
# ---------------------------------------------------------------------------

MESHES_DIR = os.path.join(os.path.dirname(__file__), "meshes")
VIEW_ALONG = {0: (0, 90), 1: (0, 0), 2: (90, 0)}
AXIS_NAMES = ['X', 'Y', 'Z']
CAPSULE_COLORS = ['#e07b54', '#5b9e6e', '#5574b8', '#b855a0', '#c9a227']

EXAMPLES = [
    {"type": "box", "obj": [0.014, 0.067, 0.187], "name": "Foot",          "max_cap": None},
    {"type": "box", "obj": [0.054, 0.195, 0.014], "name": "Knee guard",    "max_cap": None},
    {"type": "box", "obj": [0.13,  0.18,  0.412], "name": "Torso (max 2)", "max_cap": 2},
    {"type": "cylinder", "obj": (0.04, 0.15),      "name": "Shin cylinder", "max_cap": None},
    {"type": "cylinder", "obj": (0.05, 0.04),      "name": "Flat disc",     "max_cap": None},
]

if HAS_TRIMESH:
    # Synthetic: elongated rotated ellipsoid
    ell = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    ell.vertices[:, 0] *= 0.5
    ell.vertices[:, 2] *= 2.0
    ell.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 1, 0]))
    EXAMPLES.append({"type": "mesh", "obj": ell,
                     "name": "Synth ellipsoid", "max_cap": 3})

    # Real robot parts: set-cover (baseline) + CoACD (improved)
    _robot_parts = [
        ("shank.obj", "Shank", 4, 4),
        ("knee.stl",  "Knee",  3, 4),
        ("hip_3.stl", "Hip",   3, 4),
        ("waist.stl", "Waist", 3, 4),
    ]
    for fname, label, max_c, max_coacd in _robot_parts:
        fpath = os.path.join(MESHES_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            m = trimesh.load(fpath, force='mesh')
            EXAMPLES.append({"type": "mesh", "obj": m,
                             "name": f"{label} (set-cover)", "max_cap": max_c})
            if HAS_COACD:
                EXAMPLES.append({"type": "coacd", "obj": m,
                                 "name": f"{label} (CoACD)",
                                 "threshold": 0.05, "max_hull": max_coacd})
        except Exception as e:
            print(f"[warn] could not load {fname}: {e}")


# ---------------------------------------------------------------------------
# Metric formatting helper  (shared by mesh and coacd branches)
# ---------------------------------------------------------------------------

def _fmt_metrics(m):
    """Format volume_ratio, hausdorff_norm, near_coverage for printing."""
    vr = f"{m['volume_ratio']:.2f}" if m['volume_ratio'] else "N/A"
    hd = f"{m['hausdorff_norm']:.3f}" if m['hausdorff_norm'] else "N/A"
    nc = f"{m['near_coverage']:.2f}"
    return vr, hd, nc


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capsule_visualizations")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving visualizations to {out_dir}/ ...")

    for row, ex in enumerate(EXAMPLES):
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle("Box / Cylinder / Mesh → Capsule Replacement\n"
                     "(wireframe = original, coloured = capsule approximation)",
                     fontsize=12, fontweight='bold', y=1.01)

        name     = ex["name"]
        obj_type = ex["type"]
        obj      = ex["obj"]
        max_cap  = ex.get("max_cap")

        if obj_type == 'box':
            p = capsules_from_box(obj, max_capsules=max_cap)
            diag = np.linalg.norm(obj)
            surf = _box_surface_samples(np.array(obj))
            m = compute_quality_metrics(p['capsules'], surf,
                                        reference_volume=float(np.prod(obj)),
                                        mesh_diagonal=diag)
            a, r_ax = AXIS_NAMES[p['axis_idx']], AXIS_NAMES[p['arrange_idx']]
            vr, hd, nc = _fmt_metrics(m)
            print(f"{name:22s}  box:{obj}  → {p['n']} cap(s)  "
                  f"vol={vr}  hd={hd}  near_cov={nc}")
            subtitle = (f"{name}  {obj}\nn={p['n']}  vol={vr}  hd={hd}")
            titles = [subtitle, f"along capsule ({a})", f"along arrange ({r_ax})"]
            views = [(20, 30), VIEW_ALONG[p['axis_idx']],
                     VIEW_ALONG[p['arrange_idx']]]

        elif obj_type == 'cylinder':
            radius, height = obj
            p = capsules_from_cylinder(radius, height)
            print(f"{name:22s}  cyl r={radius} h={height}  → {p['n']} cap(s)")
            subtitle = f"{name}  r={radius} h={height}\nn={p['n']}"
            titles = [subtitle, "end-on (X)", "side (Z)"]
            views = [(20, 30), (0, 90), (0, 0)]

        elif obj_type == 'mesh':
            p = capsules_from_mesh_coverage(obj, max_capsules=max_cap)
            m = p['metrics']
            vr, hd, nc = _fmt_metrics(m)
            print(f"{name:22s}  mesh F={len(obj.faces)}  → {p['n']} cap(s)  "
                  f"vol={vr}  hd={hd}  near_cov={nc}")
            subtitle = f"{name}\nn={p['n']}  vol={vr}  hd={hd}  cov={nc}"
            titles = [subtitle, "View X-axis", "View Y-axis"]
            views = [(20, 30), (0, 90), (0, 0)]

        else:  # coacd
            threshold = ex.get("threshold", 0.05)
            max_hull = ex.get("max_hull", 8)
            p = capsules_from_coacd(obj, threshold=threshold,
                                    max_convex_hull=max_hull)
            m = p['metrics']
            vr, hd, nc = _fmt_metrics(m)
            print(f"{name:22s}  CoACD hulls={p['n_hulls']}  → {p['n']} cap(s)  "
                  f"vol={vr}  hd={hd}  near_cov={nc}")
            subtitle = (f"{name}\nn={p['n']} ({p['n_hulls']} hulls)  "
                        f"vol={vr}  hd={hd}  cov={nc}")
            titles = [subtitle, "View X-axis", "View Y-axis"]
            views = [(20, 30), (0, 90), (0, 0)]
            obj_type = 'mesh'  # reuse mesh wireframe drawing

        for col, (title, (elev, azim)) in enumerate(zip(titles, views)):
            ax = fig.add_subplot(1, 3, col + 1, projection='3d')
            draw_scene(ax, obj, p, obj_type=obj_type)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, pad=6, fontsize=8)

        plt.tight_layout()
        safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        out = os.path.join(out_dir, f"{row:02d}_{safe_name}.png")
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)

    _coacd_cache_cleanup()
    print(f"\nDone. Saved individual images to {out_dir}/")
