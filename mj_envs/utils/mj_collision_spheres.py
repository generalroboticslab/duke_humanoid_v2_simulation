"""Convert a MuJoCo model's collision-class geoms into cuRobo collision spheres.

Replaces the previous MORPHIT-on-visual-mesh fit (RobotBuilder + yourdfpy + ~60s + JSON
cache) with a direct, readable primitive->sphere decomposition of the actual MuJoCo
collision geoms. Every collision geom in this repo's robots is a primitive
(capsule/box/sphere), so the conversion is analytic -- no mesh fitting, no cache, pose-
independent per link.

Group roles (matches asset/create/export_mjspec_to_urdf.py::GROUP_ROLE):
    g2 visual mesh, g3 collision capsule, g4 FOV cone (viewer-only), g5 rack ikproxy.
Collision spheres come from g3 + g5 only. Per body, g5 (the simplified rack proxy)
supersedes g3 -- same rule the URDF exporter uses.

Sphere policy: sphere radius is ALWAYS the true primitive radius, never inflated -- an enlarged
sphere leaves the true geometry and causes spurious collisions / falsely-shrunk reachability.
Capsule axis spheres are therefore an under-approximation (the wall between center rings bulges
slightly past the union); denser ``spacing`` shrinks it, sealing needs off-axis spheres not a
bigger radius. Stubby capsules (cylinder length <= radius) collapse to one true-radius sphere.
See ``_capsule_spheres``.

Optional coarsening (``build_collision_spheres(coarsen_links=...)``): bodies matching the regex
whose collision geoms are a bundle of >=2 parallel capsules get the bundle collapsed to one
enclosing capsule before sphere-ization (``merge_capsule_bundle``). Used to strip the ~300
spheres in humanoid_v21's locked shank/foot 6-capsule cages (~75% of planner collision cost,
profiled) with no arm-reach fidelity loss. Everything else stays faithful.

Output format is cuRobo's collision_spheres dict, keyed by link (== MuJoCo body) name:
    {link_name: [{"center": [x, y, z], "radius": r}, ...]}
with centers in the body/link frame (cuRobo transforms them via FK per config).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class CapsuleSphereFit:
    """Per-body override for capsule->sphere fitting (see ``build_collision_spheres``).

    ``radius_scale`` multiplies the true capsule radius; ``sphere_count`` forces the number of
    spheres placed along the capsule axis (else the usual gap-derived count). Used where a
    broad locked link (e.g. Apollo's pelvis) needs a hand-tuned coarse chain instead of the
    faithful per-geom fit. Only valid for bodies whose collision geoms are capsules.
    """

    radius_scale: float = 1.0
    sphere_count: int | None = None

# geom_group roles that contribute collision geometry. g5 supersedes g3 per body.
COLLISION_GROUPS = (3, 5)
# Fail-loud ceiling on total spheres (self-collision cost is O(n^2)); tune via `spacing` if a
# long thin capsule overshoots.
SPHERE_BUDGET = 500
# Inter-sphere gap = max(spacing * radius, MIN_GAP).
#   spacing (in units of the primitive radius) = 1.0 -> centers one radius apart, overlap.
#     Radius is NEVER inflated (that would push the model outside the true capsule and cause
#     spurious collisions), so on-axis spheres of radius r under-cover the wall between center
#     rings; the ONLY honest way to shrink those holes is more, closer spheres. spacing 1.0 cuts
#     the worst base_link (thick, r~0.085 m) wall bulge 28.8 mm -> 8.0 mm (0.8 -> 5.2, 0.6 -> 2.7).
#     Total is nearly flat vs spacing (358..440): only the few THICK base_link capsules densify;
#     thin arm/leg capsules are floored by MIN_GAP and don't. (Old 2.0 = tangent, 28.8 mm holes.)
#   MIN_GAP (absolute, m) floors the gap so hair-thin locked capsules (feet r=0.007, shanks
#     r=0.01) don't spawn hundreds of sub-cm spheres -- coarser there is fine (far from the arm,
#     locked). Keeps humanoid_v21's whole-body g3/g5 fit under SPHERE_BUDGET (416 at spacing 1.0).
DEFAULT_SPACING = 1.0
DEFAULT_MIN_GAP = 0.015
_MJ = mujoco.mjtGeom


def _capsule_spheres(size: np.ndarray, gap: float) -> list[tuple[np.ndarray, float]]:
    """Capsule -> spheres along its local +z axis. ``size`` = [radius, half_length].

    Sphere radius is ALWAYS the true capsule radius -- never inflated. Inflating to seal the
    inter-sphere wall gap would push the collision model OUTSIDE the true geometry and cause
    spurious collisions / falsely-shrunk reachability, so it is disallowed. The on-axis chain
    is therefore an under-approximation: the cylinder wall between center rings bulges slightly
    beyond the sphere union (unavoidable for radius-r axis spheres). Denser ``gap`` shrinks that
    bulge; genuinely sealing it needs off-axis surface spheres, not a bigger radius.

    ``n`` is floored at 2 (not derived purely from ``gap``) so short/fat capsules always place
    a sphere at each true axial tip ``+-hl`` -- exact tip coverage, no protrusion. A previous
    version collapsed short capsules (``2*hl <= r``) to ONE enclosing sphere of radius ``r+hl``:
    since a single sphere is isotropic, that radius also bulged out perpendicular to the axis,
    where the true capsule is only ``r`` wide -- e.g. GR-3's upper_arm_roll_link (r=0.0596,
    hl=0.0242) over-covered by 2.4 cm on every side, a real violation of the "never inflate"
    rule above, not a negligible one. ``hl == 0`` is the exact sphere of radius ``r``.
    """
    r, hl = float(size[0]), float(size[1])
    if hl <= 0.0:
        return [(np.zeros(3), r)]
    n = max(2, math.ceil(2.0 * hl / gap) + 1)
    zs = np.linspace(-hl, hl, n)
    return [(np.array([0.0, 0.0, z]), r) for z in zs]


def _cylinder_spheres(size: np.ndarray, gap: float) -> list[tuple[np.ndarray, float]]:
    """Cylinder -> spheres along its local +z axis. ``size`` = [radius, half_length].

    Unlike a capsule, a cylinder's ends are FLAT, not domed. Chaining it through
    ``_capsule_spheres`` (end centers at ``z = +-hl``) pokes each end sphere a full radius
    past the flat cap, over-approximating the true length by up to ``2r`` -- large for GR-3's
    short/fat cylinders (e.g. base_link r=0.08, hl=0.075: ~107% extra length per cap). Axial
    centers are inset by ``r`` instead (same principle as ``_box_spheres``' per-axis inset),
    so the sphere union stays inside the true flat-capped cylinder.

    Stubby cylinder (``hl <= r``): insetting would put the chain interval below zero. The
    minimal ENCLOSING sphere here is ``sqrt(r^2+hl^2)`` (rim-to-center distance), but as with
    the old capsule stub bug (see ``_capsule_spheres``), a single isotropic sphere at that
    radius protrudes past the true flat caps AND the true cylinder wall everywhere off that
    one rim direction -- e.g. GR-3's base_link (r=0.08, hl=0.075) would still bulge ~2 cm past
    its true radius. Use radius ``hl`` instead (never inflated: ``hl <= r`` here, so a
    radius-``hl`` sphere at the origin touches both flat caps exactly and stays strictly
    inside the true side wall) -- an under-approximation of the rim annulus, consistent with
    this module's "never protrude past true geometry" rule.
    """
    r, hl = float(size[0]), float(size[1])
    if hl <= r:
        return [(np.zeros(3), hl)]
    hl_in = hl - r
    n = math.ceil(2.0 * hl_in / gap) + 1
    zs = np.linspace(-hl_in, hl_in, n)
    return [(np.array([0.0, 0.0, z]), r) for z in zs]


def _sphere_spheres(size: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Sphere -> itself. ``size`` = [radius]."""
    return [(np.zeros(3), float(size[0]))]


def _box_spheres(size: np.ndarray, gap: float, radius_scale: float = 1.0) -> list[tuple[np.ndarray, float]]:
    """Box -> grid of overlapping spheres. ``size`` = [hx, hy, hz] half-extents.

    Sphere radius = ``radius_scale`` * smallest half-extent; each axis is covered by a centered
    grid spaced ~``gap`` apart. Robot-agnostic path -- humanoid_v21 has no collision boxes; kept
    minimal for G1 generality.

    ``radius_scale`` < 1 shrinks the balls below the min half-extent (used where the source box
    is a tight envelope and the default r=min-half-extent balls read too fat on review). The
    inset uses the scaled r, so the ball surfaces stay flush with the true faces regardless of
    scale (inset ``h - r`` grows as r shrinks); smaller balls just need a denser grid to cover.

    Grid centers are inset by ``r`` per axis (span ``+-(h - r)``, not ``+-h``): a sphere
    centered exactly on a box face/edge/corner would stick out by ``r`` regardless of how
    small r is, since the center itself already sits on the boundary. Insetting keeps every
    sphere surface flush with (never past) the true box faces. ``h - r`` is always >= 0 since
    r is the min half-extent over all three axes.

    Density (n) is derived from the INSET span ``2*h_in``, not the raw half-extent ``2*h``:
    deriving it from ``h`` while gridding over ``h_in`` mismatches count to span. On the very
    axis that sets ``r`` (``h_in == 0``), that bug alone forced n>1 with a zero span, emitting
    the same coincident center ``n`` times -- e.g. GR-3's torso box duplicated every point 3x
    (36 spheres, most stacked on top of each other) before this fix.
    """
    r = radius_scale * float(min(size))
    axes = []
    for h in size:
        h_in = float(h) - r
        if h_in <= 0.0:
            axes.append(np.array([0.0]))
            continue
        n = max(1, math.ceil(2.0 * h_in / gap) + 1)
        axes.append(np.linspace(-h_in, h_in, n) if n > 1 else np.array([0.0]))
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return [(c, r) for c in grid]


def _local_spheres(geom_type: int, size: np.ndarray, spacing: float, min_gap: float,
                   box_radius_scale: float = 1.0):
    """Dispatch primitive -> local-frame spheres. Inter-sphere gap = max(spacing * radius,
    min_gap). ``box_radius_scale`` shrinks box sphere radii (see ``_box_spheres``)."""
    if geom_type == _MJ.mjGEOM_SPHERE:
        return _sphere_spheres(size)
    r = float(size[0]) if geom_type != _MJ.mjGEOM_BOX else box_radius_scale * float(min(size))
    gap = max(spacing * r, min_gap)
    if geom_type == _MJ.mjGEOM_CAPSULE:
        return _capsule_spheres(size, gap)
    if geom_type == _MJ.mjGEOM_CYLINDER:
        return _cylinder_spheres(size, gap)
    if geom_type == _MJ.mjGEOM_BOX:
        return _box_spheres(size, gap, box_radius_scale)
    raise ValueError(f"unsupported collision geom type {geom_type} (need sphere/capsule/cylinder/box)")


def geom_to_spheres(model: mujoco.MjModel, g: int, spacing: float = DEFAULT_SPACING,
                    min_gap: float = DEFAULT_MIN_GAP, box_radius_scale: float = 1.0):
    """Spheres for collision geom ``g``, transformed local -> **body frame**.

    Returns a list of ``(center3, radius)``. Inter-sphere gap = max(spacing * radius, min_gap):
    ``spacing`` (fraction of radius) sets density on thick geoms; ``min_gap`` (m) floors it so
    hair-thin geoms stay bounded. See DEFAULT_SPACING / DEFAULT_MIN_GAP. ``box_radius_scale``
    shrinks box sphere radii only.
    """
    q = model.geom_quat[g]
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pos = model.geom_pos[g]
    locals_ = _local_spheres(int(model.geom_type[g]), model.geom_size[g], spacing, min_gap, box_radius_scale)
    return [(pos + rot @ c, r) for c, r in locals_]


def _override_capsule_spheres(model: mujoco.MjModel, g: int, fit: CapsuleSphereFit):
    """Fit ONE capsule geom to a hand-tuned chain per ``fit``, transformed local -> body frame.

    Radius = ``fit.radius_scale`` * true capsule radius. ``fit.sphere_count`` spheres are placed
    evenly along the capsule axis (tips at +-hl); if ``sphere_count`` is None the count falls
    back to the usual gap-derived chain at the scaled radius.
    """
    if int(model.geom_type[g]) != _MJ.mjGEOM_CAPSULE:
        raise ValueError(f"capsule_sphere_overrides target geom {g} is not a capsule")
    r = fit.radius_scale * float(model.geom_size[g][0])
    hl = float(model.geom_size[g][1])
    if fit.sphere_count is not None:
        n = fit.sphere_count
    else:
        gap = max(DEFAULT_SPACING * r, DEFAULT_MIN_GAP)
        n = max(2, math.ceil(2.0 * hl / gap) + 1)
    zs = np.linspace(-hl, hl, n) if n > 1 else np.array([0.0])
    q = model.geom_quat[g]
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    pos = model.geom_pos[g]
    return [(pos + rot @ np.array([0.0, 0.0, z]), r) for z in zs]


def _geom_axis(model: mujoco.MjModel, g: int) -> np.ndarray:
    """Capsule local +z axis expressed in the body frame (unit vector)."""
    q = model.geom_quat[g]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()[:, 2]


def _all_parallel(model: mujoco.MjModel, geoms: list[int], tol: float = 1e-2) -> bool:
    a0 = _geom_axis(model, geoms[0])
    return all(abs(float(np.dot(_geom_axis(model, g), a0))) > 1.0 - tol for g in geoms)


def merge_capsule_bundle(model: mujoco.MjModel, geoms: list[int], spacing: float,
                         min_gap: float) -> list[tuple[np.ndarray, float]]:
    """Collapse a bundle of PARALLEL body-frame capsules into ONE enclosing capsule, then
    sphere it. Cheap conservative bound for locked multi-capsule cages (shank/foot) that carry
    6 thin capsules -> ~100 spheres each at full fidelity but sit far from the arm workspace.

    Enclosing capsule (all in the body frame): shared axis ``a`` = the bundle's common capsule
    axis. Axial span = union of every capsule's [center.a - hl, center.a + hl]; new half-length
    = span/2, new axial center = span midpoint. Radius = max over capsules of (perpendicular
    distance from the bundle's radial centroid + that capsule's radius) -- so the single capsule
    swallows the bundle's width too, never under-covers. Returns body-frame ``(center, radius)``
    spheres spaced by the usual ``max(spacing*r, min_gap)`` gap.
    """
    axis = _geom_axis(model, geoms[0])
    centers = np.array([model.geom_pos[g] for g in geoms])
    radii = np.array([float(model.geom_size[g][0]) for g in geoms])
    half_lengths = np.array([float(model.geom_size[g][1]) for g in geoms])
    axial = centers @ axis                                  # each capsule center's coord along a
    lo, hi = float((axial - half_lengths).min()), float((axial + half_lengths).max())
    axial_mid, hl = 0.5 * (lo + hi), 0.5 * (hi - lo)
    radial = centers - np.outer(axial, axis)               # perpendicular components
    radial_centroid = radial.mean(0)
    r = float(np.max(np.linalg.norm(radial - radial_centroid, axis=1) + radii))
    center = radial_centroid + axial_mid * axis
    gap = max(spacing * r, min_gap)
    n = max(1, math.ceil(2.0 * hl / gap) + 1) if hl > 0 else 1
    zs = np.linspace(-hl, hl, n) if n > 1 else [0.0]
    return [(center + z * axis, r) for z in zs]


def collision_geoms_for_body(model: mujoco.MjModel, b: int) -> list[int]:
    """Collision geom IDs on body ``b``: g5 if the body has any (rack simplified proxy),
    else g3. Empty if the body carries neither."""
    geoms = range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b])
    by_group = {gp: [g for g in geoms if int(model.geom_group[g]) == gp] for gp in COLLISION_GROUPS}
    return by_group[5] if by_group[5] else by_group[3]


def build_collision_spheres(model: mujoco.MjModel, spacing: float = DEFAULT_SPACING,
                            min_gap: float = DEFAULT_MIN_GAP,
                            coarsen_links: str | None = None,
                            box_inscribed: bool = True,
                            box_radius_scale: float = 1.0,
                            capsule_sphere_overrides: dict[str, CapsuleSphereFit] | None = None,
                            skip_mesh: bool = False,
                            ) -> dict[str, list[dict]]:
    """cuRobo collision_spheres dict from every collision-bearing body. Asserts the total is
    within ``SPHERE_BUDGET``.

    ``coarsen_links`` (regex, optional): bodies whose name matches AND that carry >=2 parallel
    collision capsules get their bundle collapsed to one enclosing capsule (see
    ``merge_capsule_bundle``) instead of the faithful per-geom fit. Used to strip the ~300
    spheres in humanoid_v21's locked shank/foot 6-capsule cages -- ~75% of planner collision
    cost for zero arm-planning value. Non-matching or non-parallel bodies stay faithful.

    ``box_inscribed``: box sphere grids always inset by the (scaled) radius so the union stays
    inside the true box faces -- the only supported mode; ``False`` (a face-centered grid that
    protrudes one radius past each face) is not implemented, so it is rejected loudly rather
    than silently ignored.

    ``box_radius_scale`` (<=1): shrink box sphere radii below the min half-extent for robots
    whose source boxes are already tight envelopes (Berkeley/Booster T1 use 0.75). See
    ``_box_spheres``.

    ``capsule_sphere_overrides`` (body-name -> ``CapsuleSphereFit``): replace the faithful
    per-geom fit of the named bodies with a hand-tuned scaled/fixed-count capsule chain (e.g.
    Apollo's broad pelvis). Overridden bodies bypass ``coarsen_links``.

    ``skip_mesh``: drop mesh collision geoms (this module fits primitives only; a mesh has no
    analytic sphere decomposition). Contributing no spheres is an under-approximation -- safe
    per the module's "never protrude past true geometry" rule -- valid only where the mesh link
    is irrelevant to arm planning (e.g. Toddlerbot's locked feet, pruned below-knee anyway). Off
    by default so a mesh on an unhandled robot still fails loud.
    """
    if not box_inscribed:
        raise ValueError("box_inscribed=False (face-centered box spheres) is not supported")
    overrides = capsule_sphere_overrides or {}
    coarsen_re = re.compile(coarsen_links) if coarsen_links else None
    out: dict[str, list[dict]] = {}
    for b in range(1, model.nbody):
        geoms = collision_geoms_for_body(model, b)
        if skip_mesh:
            geoms = [g for g in geoms if int(model.geom_type[g]) != _MJ.mjGEOM_MESH]
        if not geoms:
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bname in overrides:
            fit = overrides[bname]
            local = [cr for g in geoms for cr in _override_capsule_spheres(model, g, fit)]
        elif coarsen_re and coarsen_re.search(bname) and len(geoms) > 1 and _all_parallel(model, geoms):
            local = merge_capsule_bundle(model, geoms, spacing, min_gap)
        else:
            local = [cr for g in geoms for cr in geom_to_spheres(model, g, spacing, min_gap, box_radius_scale)]
        spheres = [
            {"center": [float(c[0]), float(c[1]), float(c[2])], "radius": float(r)}
            for c, r in local
        ]
        out[bname] = spheres
    total = sum(len(v) for v in out.values())
    assert total <= SPHERE_BUDGET, (
        f"collision->sphere produced {total} spheres, budget {SPHERE_BUDGET}. "
        f"Increase `spacing` (>1.0 -> fewer spheres) and re-run."
    )
    return out


def _sphere_intersection_volume(ra: float, rb: float, d: float) -> float:
    """Lens volume where two spheres (radii ``ra``, ``rb``, center distance ``d``) intersect.

    Standard two-sphere intersection: 0 when disjoint (``d >= ra+rb``), the smaller sphere's
    full volume when one contains the other (``d <= |ra-rb|``), else the closed-form lens.
    """
    if d >= ra + rb:
        return 0.0
    if d <= abs(ra - rb):
        return 4.0 / 3.0 * math.pi * min(ra, rb) ** 3
    return (
        math.pi
        * (ra + rb - d) ** 2
        * (d * d + 2.0 * d * (ra + rb) - 3.0 * (ra - rb) ** 2)
        / (12.0 * d)
    )


def _enclosing_sphere(
    ca: np.ndarray, ra: float, cb: np.ndarray, rb: float
) -> tuple[np.ndarray, float]:
    """Minimal sphere containing spheres ``(ca,ra)`` and ``(cb,rb)``: returns ``(center, radius)``.

    When one already contains the other it IS the larger sphere; otherwise the enclosing sphere
    spans from A's far surface to B's far surface (radius ``(d+ra+rb)/2``).
    """
    diff = cb - ca
    d = float(np.linalg.norm(diff))
    if d <= abs(ra - rb):
        return (ca.copy(), ra) if ra >= rb else (cb.copy(), rb)
    u = diff / d
    r = 0.5 * (d + ra + rb)
    center = 0.5 * (ca + cb) + u * (0.5 * (rb - ra))
    return center, r


def merge_strongly_overlapping_spheres(
    collision_spheres: dict[str, list[dict]],
    max_radius_growth: float = 1.35,
    max_nonoverlap_volume: float = 1e-5,
) -> dict[str, list[dict]]:
    """Merge strongly-overlapping same-link sphere pairs into one enclosing sphere.

    Capsule/box chain fits emit spheres that heavily overlap their neighbours; each redundant
    sphere costs O(n) inside cuRobo's O(n^2) self-collision check for negligible geometric gain.
    Two spheres on the SAME link (same FK frame) merge into their minimal enclosing sphere when
    BOTH hold:

      - non-overlap (symmetric-difference) volume ``vol(A)+vol(B)-2*vol(A∩B)`` <=
        ``max_nonoverlap_volume`` (m^3): they already cover nearly the same space, so replacing
        them barely changes the collision model; and
      - the enclosing sphere's radius grows <= ``max_radius_growth`` x the larger input radius:
        the replacement never balloons past the true geometry (this module's "never protrude"
        rule).

    Never merges across links. Greedy first-qualifying merge, repeated to a fixed point per link.

    Inverse of nothing -- callers run it right after ``build_collision_spheres`` and before any
    link pruning; strictly reduces or preserves the sphere count, so it never breaches
    ``SPHERE_BUDGET``.
    """
    # ponytail: O(n^3) per link (rescan on every merge); n is a few dozen spheres per link, so
    # this is instant. Switch to a spatial index only if a link ever carries hundreds of spheres.
    merged: dict[str, list[dict]] = {}
    for link, spheres in collision_spheres.items():
        items = [(np.asarray(s["center"], dtype=np.float64), float(s["radius"])) for s in spheres]
        changed = True
        while changed and len(items) > 1:
            changed = False
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    ci, ri = items[i]
                    cj, rj = items[j]
                    d = float(np.linalg.norm(ci - cj))
                    inter = _sphere_intersection_volume(ri, rj, d)
                    nonoverlap = (
                        4.0 / 3.0 * math.pi * (ri ** 3 + rj ** 3) - 2.0 * inter
                    )
                    center, r = _enclosing_sphere(ci, ri, cj, rj)
                    if nonoverlap <= max_nonoverlap_volume and r <= max_radius_growth * max(ri, rj):
                        items[i] = (center, r)
                        items.pop(j)
                        changed = True
                        break
                if changed:
                    break
        merged[link] = [
            {"center": [float(c[0]), float(c[1]), float(c[2])], "radius": float(r)}
            for c, r in items
        ]
    return merged


def _selfcheck() -> None:
    """Assert the merge geometry on hand-checkable cases."""
    # Two coincident identical 0.1 m spheres -> one 0.1 m sphere (zero non-overlap, growth 1.0).
    out = merge_strongly_overlapping_spheres(
        {"l": [{"center": [0, 0, 0], "radius": 0.1}, {"center": [0, 0, 0], "radius": 0.1}]}
    )
    assert len(out["l"]) == 1 and abs(out["l"][0]["radius"] - 0.1) < 1e-9, out

    # Two 0.1 m spheres 0.5 m apart (disjoint) -> never merged (huge non-overlap + radius growth).
    out = merge_strongly_overlapping_spheres(
        {"l": [{"center": [0, 0, 0], "radius": 0.1}, {"center": [0.5, 0, 0], "radius": 0.1}]}
    )
    assert len(out["l"]) == 2, out

    # One sphere fully inside a larger one -> collapses to the larger (radius unchanged).
    out = merge_strongly_overlapping_spheres(
        {"l": [{"center": [0, 0, 0], "radius": 0.1}, {"center": [0.01, 0, 0], "radius": 0.02}]},
        max_nonoverlap_volume=1.0,
    )
    assert len(out["l"]) == 1 and abs(out["l"][0]["radius"] - 0.1) < 1e-9, out
    print("mj_collision_spheres self-check OK")


if __name__ == "__main__":
    _selfcheck()
