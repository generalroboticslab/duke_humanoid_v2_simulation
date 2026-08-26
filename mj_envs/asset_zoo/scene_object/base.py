"""Shared primitives for procedural scene-object assets.

A scene object (workbench, shelf, ...) is a list of ``Prop`` pieces — rigid geoms welded to the
scene worldbody. ``PartBuilder`` accumulates those pieces for one object placed at world (x, y)
with a yaw about z (so a builder writes geometry in a convenient local frame). ``add_props_to_spec``
welds a finished piece list onto a ``mujoco.MjSpec`` worldbody (e.g. an mjlab ``SceneCfg.spec_fn``).

Articulated objects (drawers/doors) need bodies + joints, not just welded geoms; they will get
their own builder and do not use ``PartBuilder``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

GREY = (0.5, 0.5, 0.5, 1.0)            # neutral default color
INVIS = (0.0, 0.0, 0.0, 0.0)           # invisible collision proxy

WOOD_MATERIAL = "scene_wood"           # set Prop.material to this; add_props_to_spec builds it
_WOOD_TEX_PX = 256
# sRGB source colours. They stay deliberately close: this is quiet finished oak, not high-contrast
# growth rings. Blender decodes these before physically based shading, which otherwise makes a
# numerically "light" texture look unexpectedly dark.
# Pale birch: near-white under Blender's linear/PBR lighting, with only a warm wood hint.
_WOOD_LIGHT = (0.96, 0.94, 0.88)
_WOOD_DARK = (0.91, 0.88, 0.81)


@dataclass
class Prop:
    """One static geom welded to the world.

    Box by default (``center`` + ``half`` + ``quat``). If ``fromto`` is set it is a CAPSULE
    spanning those two world endpoints with radius ``half[0]`` (``center`` is the midpoint, kept
    consistent so position checks still hold); ``quat`` is then ignored. ``shape="ellipsoid"``
    keeps the box's ``center``/``half``/``quat`` semantics but renders as an ellipsoid (semi-axes
    = ``half``) -- a rounded, organic pad instead of a flat-faced box.
    """
    name: str
    center: tuple[float, float, float]
    half: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = GREY
    collide: bool = True
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # (w,x,y,z)
    fromto: tuple[float, float, float, float, float, float] | None = None  # capsule endpoints
    group: int | None = None  # None -> derive from alpha (see add_props_to_spec); set to override
    shape: str = "box"  # "box" | "ellipsoid"; ignored when fromto is set (always capsule)
    # Material name; ``add_props_to_spec`` registers the shared wood material on the spec on
    # demand (``_ensure_wood``). MuJoCo multiplies material texture by rgba, so rgba stays the
    # piece's base tint rather than being replaced.
    material: str | None = None


def _yawq(yaw_deg: float) -> tuple[float, float, float, float]:
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def frame_from_axes(x_dir, z_dir) -> np.ndarray:
    """Rotation sending local +x to ``x_dir`` and local +z to ``z_dir``, for ``PartBuilder(rot=...)``.

    ``z_dir`` is orthonormalized against ``x_dir``, so callers may pass any non-parallel pair -- an
    asset is usually placed by "point it THIS way, with its face THAT way" and asking for an exactly
    orthogonal pair just moves the Gram-Schmidt into every call site.
    """
    x = np.asarray(x_dir, float)
    x = x / np.linalg.norm(x)
    z = np.asarray(z_dir, float)
    z = z - x * float(z @ x)
    z = z / np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


class PartBuilder:
    """Accumulate ``Prop`` pieces of one object, each given in a local frame that is rotated and
    translated to world. ``.box`` / ``.ellipsoid`` / ``.capsule`` append a piece; ``.pieces`` is the
    result list.

    The frame was YAW-ONLY until 2026-08-12, which silently capped what this module could express:
    any asset that had to TILT was unbuildable, and the one that needed it (the human figure's
    hanging hand) got a bespoke 50-line copy of an existing asset instead -- a copy that was wrong
    in every revision it had and drifted from the original whenever either changed. Passing ``rot``
    lifts that cap, so an asset is authored once in its natural frame and placed at any orientation.

    Args:
        name: piece-name stem; every piece is ``{name}_{suffix}``.
        x, y, z: world origin of the local frame.
        yaw: rotation about z, degrees. Convenience for the floor-standing case (shelf, workbench,
            figure) -- exactly equivalent to the matching ``rot``, and ignored when ``rot`` is given.
        rot: full 3x3 world rotation of the local frame; see ``frame_from_axes``.
    """

    def __init__(self, name: str, x: float, y: float, yaw: float = 0.0,
                 z: float = 0.0, rot: np.ndarray | None = None):
        self.name = name
        self.t = np.array([x, y, z], dtype=float)
        if rot is None:
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            self.q = _yawq(yaw)
        else:
            self.R = np.asarray(rot, dtype=float)
            q = np.empty(4)
            mujoco.mju_mat2Quat(q, self.R.flatten())
            self.q = tuple(q)
        self.pieces: list[Prop] = []

    def to_world(self, p) -> tuple[float, float, float]:
        """Map a point from the builder's local frame to world."""
        return tuple(self.R @ np.asarray(p, float) + self.t)

    def to_local(self, p) -> tuple[float, float, float]:
        """Inverse of ``to_world``."""
        return tuple(self.R.T @ (np.asarray(p, float) - self.t))

    def box(self, suffix, loff, half, rgba, collide, group=None, material=None):
        self.pieces.append(Prop(f"{self.name}_{suffix}", self.to_world(loff),
                                half, rgba, collide, self.q, group=group, material=material))

    def ellipsoid(self, suffix, loff, half, rgba, collide, group=None):
        self.pieces.append(Prop(f"{self.name}_{suffix}", self.to_world(loff),
                                half, rgba, collide, self.q, group=group, shape="ellipsoid"))

    def local_bounds(self, *suffixes):
        """Tight ``(center, half)`` in the BUILDER's own frame around the named pieces already added.

        For sizing a collision proxy from the parts it stands for instead of transcribing their
        dimensions: a hand-written half-extent goes stale the moment a body dimension changes, and
        it fails silently -- the proxy simply stops covering what it claims to cover. Every
        primitive here has an exact local box (box/ellipsoid: ``center +/- half``; capsule: both
        endpoints +/- radius), so the union is exact rather than an over-bound.

        Asserts every requested suffix exists, which makes a renamed or removed part a build-time
        error instead of a silently shrunken obstacle.
        """
        want = {f"{self.name}_{s}" for s in suffixes}
        owned = [p for p in self.pieces if p.name in want]
        missing = want - {p.name for p in owned}
        assert not missing, f"local_bounds: no such piece(s) {sorted(missing)}"
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for p in owned:
            ends = ((p.fromto[:3], p.fromto[3:]) if p.fromto else (p.center,))
            ext = (p.half[0],) * 3 if p.fromto else p.half
            for e in ends:
                for i, v in enumerate(self.to_local(e)):
                    lo[i] = min(lo[i], v - ext[i])
                    hi[i] = max(hi[i], v + ext[i])
        return (tuple((a + b) / 2.0 for a, b in zip(lo, hi)),
                tuple((b - a) / 2.0 for a, b in zip(lo, hi)))

    def capsule(self, suffix, p0, p1, radius, rgba, collide, group=None):
        w0, w1 = self.to_world(p0), self.to_world(p1)
        ft = (*w0, *w1)
        mid = tuple((a + b) / 2.0 for a, b in zip(w0, w1))
        self.pieces.append(Prop(f"{self.name}_{suffix}", mid, (radius, 0.0, 0.0),
                                rgba, collide, self.q, ft, group=group))


def _ensure_wood(spec: mujoco.MjSpec) -> None:
    """Register the shared wood-grain material on ``spec``. Idempotent.

    Generated into memory rather than shipped as a PNG: it is a handful of numpy ops, so a file
    would be one more asset path to resolve and keep in sync for nothing.

    One low-contrast longitudinal grain signal: irregular enough not to read as graphic stripes,
    quiet enough to remain a tabletop rather than become scene texture. It is deliberately not a
    growth-ring simulation; procedural rings looked dark and artificial under Blender lighting.

    ``texuniform`` scales grain by geom world size instead of UVs, so every bench keeps the same
    physical density whatever its top dimensions. This lives in MuJoCo model rather than Blender
    bridge: native renderer draws it and bridge exports ``mat_texid`` through normal material path.
    One source of truth for look.
    """
    if any(m.name == WOOD_MATERIAL for m in spec.materials):
        return
    length, width = np.meshgrid(
        np.linspace(0.0, 1.0, _WOOD_TEX_PX, dtype=np.float32),
        np.linspace(0.0, 1.0, _WOOD_TEX_PX, dtype=np.float32),
        indexing="ij",
    )
    grain = 0.5 + 0.10 * np.sin(width * 30.0 + 0.22 * np.sin(length * 9.0))
    grain += 0.04 * np.sin(width * 83.0 + length * 3.0)
    rgb = np.stack([lo + grain * (hi - lo) for hi, lo in zip(_WOOD_LIGHT, _WOOD_DARK)], axis=-1)

    tex = spec.add_texture()
    tex.name = f"{WOOD_MATERIAL}_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.width = tex.height = _WOOD_TEX_PX
    tex.nchannel = 3
    tex.data = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()

    mat = spec.add_material()
    mat.name = WOOD_MATERIAL
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    mat.texuniform = True
    mat.texrepeat = [2.0, 1.0]
    mat.specular = 0.15     # satin varnish; the 0.5 default reads as wet plastic
    mat.shininess = 0.3
    mat.reflectance = 0.0


def add_props_to_spec(spec: mujoco.MjSpec, pieces: list[Prop], prefix: str = "") -> None:
    """Weld ``Prop`` pieces (box or capsule) onto ``spec.worldbody``.

    Geom group follows the robot convention so the viewer toggles them together: visible parts
    (rgba alpha > 0) -> group 2, invisible collision proxies -> group 3. ``prefix`` namespaces
    geom names (e.g. ``"pp_"``).
    """
    for f in pieces:
        g = spec.worldbody.add_geom()
        g.name = f"{prefix}{f.name}"
        if f.fromto is not None:
            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.fromto = list(f.fromto)        # endpoints; compiler sets pos/quat + length
            g.size = [f.half[0], 0.0, 0.0]   # radius
        else:
            g.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID if f.shape == "ellipsoid" else mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(f.half)
            g.pos = list(f.center)
            g.quat = list(f.quat)
        g.rgba = list(f.rgba)
        if f.material is not None:
            # Built here rather than by the caller: props are declared at import time, long before
            # any spec exists, and a caller that forgot would only fail at compile().
            if f.material == WOOD_MATERIAL:
                _ensure_wood(spec)
            g.material = f.material
        g.contype = 1 if f.collide else 0
        g.conaffinity = 1 if f.collide else 0
        g.group = f.group if f.group is not None else (2 if f.rgba[3] > 0.0 else 3)
