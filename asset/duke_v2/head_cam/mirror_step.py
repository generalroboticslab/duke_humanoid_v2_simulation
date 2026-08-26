"""CAD stage 1 (PROVENANCE — not part of the model build; see README.md).

Create source/HeadCameraV2_dual.step: add a twin of the head-camera group,
rotated 180 deg about the vertical axis through the plate's central hole
(same physical parts mounted on the opposite side — confirmed by the user;
a true mirror would flip chirality and need mirror-image motors).

The axis passes through the center of the LARGEST inner cutout of the top
face of CNC_body03_x1_top_plate.  Everything except the top plate and the
empty 'schematics' component is duplicated.

  IN    source/HeadCameraV2.step            (committed, 11.7 MB Fusion AP214 export)
  OUT   source/HeadCameraV2_dual.step       (NOT committed — regenerate before stage 2)
  DEPS  build123d (NOT installed in the project env; needs a CAD env)
  PINS  the "right column = left rotated 180z, NOT mirrored" decision that
        head_camera_creation.RZ180 / cam_fusion_info.mirror_180z encode, plus the
        plate hole center used as the rotation axis.

Run:  python mirror_step.py
"""
from __future__ import annotations

import copy
import pathlib

import numpy as np
from build123d import Axis, Compound, Plane, export_step, import_step, mirror

HERE = pathlib.Path(__file__).parent
SRC = HERE / "source" / "HeadCameraV2.step"
DST = HERE / "source" / "HeadCameraV2_dual.step"
MODE = "rotate180"  # or "mirror"


def main():
    print("loading", SRC, flush=True)
    asm = import_step(str(SRC))
    children = list(asm.children)

    plate = next(c for c in children if "top_plate" in (c.label or "").lower())
    group = [c for c in children
             if c is not plate and len(list(c.solids())) > 0]
    print("plate:", plate.label, flush=True)
    print("group to duplicate:", [c.label for c in group], flush=True)

    # ---- hole center: largest inner wire of the plate's top face ----------
    bb = plate.bounding_box()
    top_faces = [f for f in plate.faces()
                 if abs(f.center().Z - bb.max.Z) < 0.2]
    top = max(top_faces, key=lambda f: f.area)
    inner = top.inner_wires()
    if not inner:
        raise RuntimeError("no inner wires found on plate top face")
    big = max(inner, key=lambda w: w.length)
    wb = big.bounding_box()
    hole = np.array([(wb.min.X + wb.max.X) / 2, (wb.min.Y + wb.max.Y) / 2])
    print(f"hole center: x={hole[0]:.3f} y={hole[1]:.3f} "
          f"(cutout bbox {wb.max.X - wb.min.X:.1f} x {wb.max.Y - wb.min.Y:.1f}, "
          f"{len(inner)} inner wires on top face)", flush=True)

    # ---- duplicate the group ------------------------------------------------
    pl = Plane(origin=(hole[0], hole[1], 0), z_dir=(0, 1, 0))
    twins = []
    for c in group:
        if MODE == "mirror":
            t = mirror(c, about=pl)
        else:
            t = copy.copy(c).rotate(  # 180 deg about vertical axis through hole
                axis=Axis((hole[0], hole[1], 0), (0, 0, 1)), angle=180)
        t.label = (c.label or "part") + "_twin"
        twins.append(t)

    out = Compound(label="twincities_v2_dual",
                   children=[copy.copy(plate)] + [copy.copy(c) for c in group]
                            + twins)
    export_step(out, str(DST))
    print("wrote", DST, flush=True)

    # ---- verification: reload and compare ----------------------------------
    chk = import_step(str(DST))
    print("\nverification (reloaded output):", flush=True)
    total_orig = total_twin = 0.0
    for ch in chk.children:
        sl = list(ch.solids())
        vol = sum(s.volume for s in sl)
        b = ch.bounding_box()
        c = np.round([(b.min.X + b.max.X) / 2, (b.min.Y + b.max.Y) / 2,
                      (b.min.Z + b.max.Z) / 2], 1)
        print(f"  {ch.label}: solids={len(sl)} vol={vol/1000:.2f}cm3 "
              f"center=({c[0]}, {c[1]}, {c[2]})", flush=True)
        if (ch.label or "").endswith("_twin"):
            total_twin += vol
        elif "top_plate" not in (ch.label or ""):
            total_orig += vol
    print(f"group volume original={total_orig/1000:.2f} cm3, "
          f"twin={total_twin/1000:.2f} cm3 (must match)", flush=True)


if __name__ == "__main__":
    main()
