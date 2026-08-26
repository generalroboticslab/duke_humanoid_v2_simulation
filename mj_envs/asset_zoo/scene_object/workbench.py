"""Stand-alone LARBANKE-style adjustable workbench asset (visual-realistic, collision-simple).

A reusable static-prop asset: a height-adjustable workbench modeled after the LARBANKE
bench (wood top 48 x 20.1 x 1.2 in, black steel legs + apron frame, 5 height detents
33.5-41.3 in). It is independent of any robot/env — ``workbench(...)`` returns a list of
``Prop`` pieces (built via the shared ``PartBuilder``) in a world frame, welded onto any
``mujoco.MjSpec`` worldbody with ``add_props_to_spec`` (see ``base.py``; e.g. an mjlab
``SceneCfg.spec_fn``). Run this file directly to preview a single bench (``--render`` / ``--viewer``).

Model split (mirrors the robot's visual/collision geom-group convention):
  - VISUAL (group 2, non-colliding): wood top + 4 corner legs + a 4-rail apron frame.
  - COLLISION (group 3, invisible): a flat top plate (box — the rest-on surface) + 4 VERTICAL
    leg capsules sitting FLUSH on the floor (bottom cap tangent at z=0) up to the top-underside,
    co-located with the visible legs. Capsules emit fewer
    contact points than box legs -> lighter constraint solve / less njmax pressure; the top
    must stay a box to present a flat surface.

Frame: a bench is built in its local frame (long 48-in axis = local y, depth toward viewer =
local x, +z up), then rotated ``yaw`` deg about z and translated to world (x, y). ``top_z`` is
the top SURFACE height (default ``DEFAULT_TABLE_Z`` = 24 in, the adjustable minimum).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

if __package__ in (None, ""):   # run as a script: put mj_envs/ on the path for the asset_zoo import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco

from asset_zoo.scene_object.base import (
    INVIS,
    WOOD_MATERIAL,
    PartBuilder,
    Prop,
    add_props_to_spec,
)

# --------------------------------------------------------------------------- #
# Geometry (m). LARBANKE top 48 x 20.1 x 1.2 in.
# --------------------------------------------------------------------------- #
TOP_LEN_H = 1.2192 / 2.0           # 48 in   -> half-length (long axis, faces user)
TOP_DEPTH_H = 0.5105 / 2.0         # 20.1 in -> half-depth (toward user)
TOP_THICK_H = 0.03048 / 2.0        # 1.2 in  -> half-thickness
LEG_H = 0.02                       # visible leg square half-width (0.04 m tube)
LEG_COL_R = 0.03                   # leg-capsule COLLIDER radius (wider than the visual leg)
INSET = 0.05                       # leg inset from the top edge
APRON_HZ = 0.04                    # apron rail half-height

DEFAULT_TABLE_Z = 0.6096           # default top surface height = 24 in (robot-workspace height,
                                   # below the product's 33.5-41.3 in detent range)

# White, so the wood texture's own colour comes through unmultiplied (MuJoCo multiplies material
# texture by geom rgba). Kept as a constant because WorkbenchSpec still exposes it as a knob.
TABLE_RGBA = (1.0, 1.0, 1.0, 1.0)      # wood grain comes from WOOD_MATERIAL
LEG_RGBA = (0.10, 0.10, 0.10, 1.0)     # black steel


@dataclass(frozen=True)
class WorkbenchSpec:
    """Shape + appearance of one workbench (all lengths are HALF-extents, m).

    Defaults reproduce the LARBANKE bench (the module-level ``*_H`` constants). Pass a custom
    instance to ``workbench(..., spec=...)`` to build a different table (other footprint, leg
    thickness, collider radius, or colors) without touching the builder. ``top_z`` (top surface
    height) stays a per-call arg since the same bench is used at different heights.
    """
    len_h: float = TOP_LEN_H          # long axis (faces user) half-length
    depth_h: float = TOP_DEPTH_H      # depth (toward user) half-length
    thick_h: float = TOP_THICK_H      # top half-thickness
    leg_h: float = LEG_H              # visible leg square half-width
    leg_col_r: float = LEG_COL_R      # collider capsule radius
    inset: float = INSET              # leg inset from top edge
    apron_hz: float = APRON_HZ        # apron rail half-height
    top_rgba: tuple[float, float, float, float] = TABLE_RGBA
    leg_rgba: tuple[float, float, float, float] = LEG_RGBA


LARBANKE = WorkbenchSpec()             # default bench


def workbench(name: str, x: float, y: float, top_z: float = DEFAULT_TABLE_Z,
              yaw: float = 0.0, spec: WorkbenchSpec = LARBANKE) -> list[Prop]:
    """Return the ``Prop`` pieces of one bench at world (x, y), top at ``top_z``, yaw deg.

    ``spec`` (default ``LARBANKE``) sets the bench shape/appearance; pass a custom
    ``WorkbenchSpec`` for a different table. Pieces (local frame, then yawed + translated):
    wood top, 4 legs, 4 apron rails (all visual), a collision top plate, and 4 vertical leg
    capsules (collision).
    """
    len_h, depth_h, thick_h = spec.len_h, spec.depth_h, spec.thick_h
    leg_h, col_r, inset, apron_hz = spec.leg_h, spec.leg_col_r, spec.inset, spec.apron_hz
    top_rgba, leg_rgba = spec.top_rgba, spec.leg_rgba

    b = PartBuilder(name, x, y, yaw)
    under = top_z - 2.0 * thick_h                # top underside z
    lz = under / 2.0                             # leg mid-height (floor -> underside)
    lx_edge = depth_h - inset
    ly_edge = len_h - inset

    # VISUAL (group 2): wood top + 4 corner legs + apron frame.
    b.box("top", (0.0, 0.0, top_z - thick_h), (depth_h, len_h, thick_h), top_rgba, False,
          material=WOOD_MATERIAL)
    for sx in (+1, -1):
        for sy in (+1, -1):
            b.box(f"leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                  (sx * lx_edge, sy * ly_edge, lz), (leg_h, leg_h, lz), leg_rgba, False)
    arz = under - apron_hz
    for sx in (+1, -1):
        b.box(f"apronL_{'p' if sx > 0 else 'n'}", (sx * lx_edge, 0.0, arz),
              (leg_h, ly_edge, apron_hz), leg_rgba, False)
    for sy in (+1, -1):
        b.box(f"apronS_{'p' if sy > 0 else 'n'}", (0.0, sy * ly_edge, arz),
              (lx_edge, leg_h, apron_hz), leg_rgba, False)

    # COLLISION (group 3, invisible): flat top plate (box) + 4 vertical leg capsules
    # (bottom cap flush at floor z=0 -> top underside).
    b.box("col_top", (0.0, 0.0, top_z - thick_h), (depth_h, len_h, thick_h), INVIS, True)
    for sx in (+1, -1):
        for sy in (+1, -1):
            b.capsule(f"col_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                      (sx * lx_edge, sy * ly_edge, col_r),
                      (sx * lx_edge, sy * ly_edge, under), col_r, INVIS, True)
    return b.pieces


# --------------------------------------------------------------------------- #
# Stand-alone preview (no robot/env): one bench on a floor.
# --------------------------------------------------------------------------- #
def _preview_model(top_z: float, yaw: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280     # default offscreen buffer is 640x480
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4.0, 4.0, 0.1]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]
    add_props_to_spec(spec, workbench("bench", 0.0, 0.0, top_z, yaw))
    return spec.compile()


def main() -> None:
    top_z = DEFAULT_TABLE_Z
    yaw = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--table-z"):
            top_z = float(a.split("=", 1)[1]) if "=" in a else top_z
        elif a.startswith("--yaw"):
            yaw = float(a.split("=", 1)[1]) if "=" in a else yaw
    model = _preview_model(top_z, yaw)
    print(f"workbench preview: top_z={top_z:.3f} yaw={yaw} ngeom={model.ngeom}")

    if "--viewer" in sys.argv:
        from mujoco import viewer
        model.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                                   | mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        for i in range(model.ngeom):
            gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if "_col_" in gn:
                model.geom_rgba[i] = [1.0, 0.2, 0.2, 0.5]   # reveal colliders (group 3)
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
        cam.lookat[:] = [0.0, 0.0, 0.4]
        cam.distance, cam.azimuth, cam.elevation = 2.2, 135.0, -15.0
        with mujoco.Renderer(model, height=720, width=1280) as r:
            r.update_scene(data, cam)
            img = r.render()
        out = os.path.join(os.path.dirname(__file__), "media", "workbench_preview.png")
        iio.imwrite(out, img)
        print(f"rendered -> {out}")


if __name__ == "__main__":
    main()
