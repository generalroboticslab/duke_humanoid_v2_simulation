"""Stand-alone shelf-rack asset: an N-tier open shelf rack on 4 corner posts.

Procedural like the workbench (no mesh files, no generated MJCF). ``shelf(...)`` returns ``Prop``
pieces welded onto any ``mujoco.MjSpec`` worldbody via ``add_props_to_spec`` (see ``base.py``;
e.g. an mjlab ``SceneCfg.spec_fn``). Run this file directly to preview one rack (``--render`` /
``--viewer``).

Geometry (bottom-up, floor at z=0): ``tier_gaps[0]`` is the clear gap UNDER the bottom shelf;
``tier_gaps[i>=1]`` is the clear gap between shelf i-1 and shelf i. The top shelf's top face is
the top of the unit (no headroom above). Shelves are boxes; the 4 corner posts are vertical
capsules spanning floor->top (hemispherical caps inset by the radius so the z-extent is 0..H).

Model split follows the workbench convention via ``add_props_to_spec`` (visible rgba -> group 2).
Shelves and posts are simple primitives whose collider IS their visual shape, so each piece is a
single visible + collidable geom (no separate invisible proxy — that split only pays off when the
visual and collision shapes differ, as on the workbench legs/apron).

Frame: built in a local frame centered on the rack footprint (depth toward robot = local +x,
width = local y, +z up), then yawed ``yaw`` deg about z and translated to world (x, y) by
``PartBuilder``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

if __package__ in (None, ""):   # run as a script: put mj_envs/ on the path for the asset_zoo import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco

from asset_zoo.scene_object.base import PartBuilder, Prop, add_props_to_spec

SHELF_RGBA = (0.95, 0.95, 0.95, 1.0)   # white shelf boards, standard utility-rack look
POST_RGBA = (0.85, 0.85, 0.85, 1.0)    # light-grey metal posts

_POST_CORNERS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class ShelfSpec:
    """Shape + appearance of one shelf rack (full extents in m unless noted).

    Defaults reproduce the 2-tier utility rack. ``len(tier_gaps)`` MUST equal ``n_tiers``:
    ``tier_gaps[0]`` is the ground clearance under the bottom shelf; ``tier_gaps[i>=1]`` is the
    clear vertical gap between consecutive shelves. Pass a custom instance to ``shelf(..., spec=)``
    for a different rack (more tiers, other footprint) without touching the builder.
    """
    n_tiers: int = 2
    length: float = 0.60                             # shelf y extent (width, faces robot left-right)
    width: float = 0.28                              # shelf x extent (depth, front-to-back)
    thick: float = 0.015                             # shelf board thickness
    tier_gaps: tuple[float, ...] = (0.25, 0.25)      # [0]=ground gap; [i>=1]=gap below shelf i
    post_radius: float = 0.012                       # corner-post radius
    post_inset: float = 0.02                         # post-center inset from each shelf edge
    shelf_rgba: tuple[float, float, float, float] = SHELF_RGBA
    post_rgba: tuple[float, float, float, float] = POST_RGBA


DEFAULT_SHELF = ShelfSpec()


def _tier_center_z(spec: ShelfSpec) -> list[float]:
    """Center z of each shelf board, bottom-up. ``len(tier_gaps)`` must equal ``n_tiers``."""
    gaps = spec.tier_gaps
    assert len(gaps) == spec.n_tiers, \
        f"len(tier_gaps)={len(gaps)} must equal n_tiers={spec.n_tiers}"
    bases = [gaps[0]]                                  # bottom-face z of shelf 0
    for k in range(1, spec.n_tiers):
        bases.append(bases[k - 1] + spec.thick + gaps[k])
    return [b + spec.thick / 2.0 for b in bases]


def tier_surface_z(spec: ShelfSpec = DEFAULT_SHELF) -> list[float]:
    """Top-face z of each shelf tier, bottom-up — where objects placed on the rack rest."""
    return [z + spec.thick / 2.0 for z in _tier_center_z(spec)]


def shelf(name: str, x: float, y: float, yaw: float = 0.0,
          spec: ShelfSpec = DEFAULT_SHELF) -> list[Prop]:
    """Return the ``Prop`` pieces of one N-tier shelf rack, footprint centered at world (x, y).

    Pieces (local frame, then yawed + translated): ``{name}_tier{k}`` boxes bottom-up +
    ``{name}_post{i}`` capsules at the 4 corners. Every piece is visible (group 2) and collidable.
    """
    hx, hy, hz = spec.width / 2.0, spec.length / 2.0, spec.thick / 2.0
    zc = _tier_center_z(spec)
    height = zc[-1] + hz                              # top face of the top shelf
    r = spec.post_radius
    px, py = hx - spec.post_inset, hy - spec.post_inset

    b = PartBuilder(name, x, y, yaw)
    for k, z in enumerate(zc):
        b.box(f"tier{k}", (0.0, 0.0, z), (hx, hy, hz), spec.shelf_rgba, True)
    for i, (sx, sy) in enumerate(_POST_CORNERS):
        cx, cy = sx * px, sy * py
        b.capsule(f"post{i}", (cx, cy, r), (cx, cy, height - r), r, spec.post_rgba, True)
    return b.pieces


# --------------------------------------------------------------------------- #
# Stand-alone preview (no robot/env): one rack on a floor.
# --------------------------------------------------------------------------- #
def _preview_model(yaw: float, spec: ShelfSpec) -> mujoco.MjModel:
    s = mujoco.MjSpec()
    s.visual.global_.offwidth = 1280        # default offscreen buffer is 640x480
    s.visual.global_.offheight = 720
    s.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
    floor = s.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4.0, 4.0, 0.1]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]
    add_props_to_spec(s, shelf("rack", 0.0, 0.0, yaw, spec))
    return s.compile()


def main() -> None:
    yaw = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--yaw"):
            yaw = float(a.split("=", 1)[1]) if "=" in a else yaw
    model = _preview_model(yaw, DEFAULT_SHELF)
    print(f"shelf rack preview: tiers={DEFAULT_SHELF.n_tiers} "
          f"surfaces={[round(z, 3) for z in tier_surface_z()]} yaw={yaw} ngeom={model.ngeom}")

    if "--viewer" in sys.argv:
        from mujoco import viewer
        model.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                                   | mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        viewer.launch(model, data)
        return

    if "--render" in sys.argv:
        # requires MUJOCO_GL=egl in the environment (mujoco reads it at import time)
        import imageio.v3 as iio
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 1.8, 135.0, -15.0
        with mujoco.Renderer(model, height=720, width=1280) as r:
            r.update_scene(data, cam)
            img = r.render()
        out = os.path.join(os.path.dirname(__file__), "media", "shelf_preview.png")
        iio.imwrite(out, img)
        print(f"rendered -> {out}")


if __name__ == "__main__":
    main()
