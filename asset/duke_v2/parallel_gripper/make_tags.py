"""Paste AprilTag (tag36h11) images onto the gripper's 4 tag plates as ZERO-THICKNESS decals.

The plate thickness is already modelled in the CAD, so the tag must be a flat picture
glued to the plate's outer face -- NOT 3D geometry. Each tag is therefore:
  - a PNG of the tag36h11 pattern (white + black), and
  - a flat 4-vertex quad mesh (2 triangles, no thickness) with UV coords,
placed 0.4 mm off the plate's outer face (a decal offset, not tag thickness; 0.4 mm keeps
the quad clear of depth-buffer fighting with the base-mesh pad faces at typical camera
distances - 0.1 mm was inside the z-buffer resolution once the pads moved into base.obj) and textured
with the PNG. Rendered with material=texture -> a true flat image.

8x8 grids come from tag_grids.py (TAG_GRIDS; cv2.aruco DICT_APRILTAG_36h11). Quads are baked
in the gripper WORLD frame so they attach under the rack bodies (world-coord meshes at
pos 0 0 0) and ride with the jaws.

CONFIGURABLE/REPLACEABLE: change the tag IDs + placement in TAGS below (and the grids in
tag_grids.py), then re-run this script + tag_finalize.py to re-skin the gripper.

    python3 make_tags.py        # -> meshes/tags/tag_<id>.png + tag_<id>.obj
"""
import pathlib

import numpy as np
from PIL import Image

from tag_grids import TAG_GRIDS   # {id: 8x8 (1=white,0=black,row0=top)}
from tag_layout import tag_specs  # per-hand slot->id layout (shared by the loader)

HERE = pathlib.Path(__file__).parent

PLATE_MM = 20.0     # plate side (tag fills it)
EPS_MM = 0.4        # decal stand-off above the face (placement only; >= 4x the offscreen z-buffer
                    # resolution at 0.4 m so the quad never depth-fights the base-mesh pad plane)
PNG_PX = 512        # nearest-upscaled so edges stay crisp
N = 8

# Every tag (both hands) at its plate slot. Source of truth = tag_layout.py:
#   left_rack / right_rack jaws, +Z (upper) / -Z (lower) plate faces, rot=180 (cv2.aruco vs
#   pupil_apriltags 180deg decode offset, so detected tag +X = +gripper X / robot-forward).
# The gripper XML is tag-FREE; the loader composes each hand's 4 tags at load.
TAGS = tag_specs()


def save_png(tag_id, out):
    g = np.asarray(TAG_GRIDS[tag_id])                     # 8x8, 1=white 0=black, row0 top
    img = np.where(g == 1, 255, 0).astype(np.uint8)
    big = np.kron(img, np.ones((PNG_PX // N, PNG_PX // N), np.uint8))   # nearest upscale
    Image.fromarray(big, "L").save(out / f"tag_{tag_id}.png")


def write_quad(t, out):
    """Flat quad on the plate face. UV maps the PNG so the tag decodes from the outer side.
    +Z face viewed from above: image right->+X, image up->+Y.
    -Z face viewed from below: mirror X so it is not a (undecodable) mirror image.

    In-plane decal rotation `rot` (deg, 0/90/180/270): rotates the PRINTED IMAGE on the
    plate. The AprilTag detector locks onto the pattern's canonical orientation, so the
    detected frame rotates WITH the decal. Use this to align the detected +X (image-right)
    to the physical direction you want. NOTE: cv2.aruco-generated 36h11 markers are decoded
    by the reference AprilRobotics library with a fixed offset (commonly 180°); `rot`
    compensates it. The exact value must be confirmed against your real detector."""
    cx, cy, zf = t["cx"], t["cy"], t["zface"]
    up = t["face"] == "+Z"
    rot = int(t.get("rot", 0))
    h = PLATE_MM / 2
    z = zf + EPS_MM if up else zf - EPS_MM

    if up:
        # 3D corner positions TL,TR,BR,BL on the readable (+Z) face
        pos = [(cx - h, cy + h), (cx + h, cy + h), (cx + h, cy - h), (cx - h, cy - h)]
        faces = [(0, 3, 2), (0, 2, 1)]        # CCW seen from +Z -> normal +Z
    else:
        # -Z face viewed from below: screen-right=+X, screen-up=-Y (mirror so it is readable)
        pos = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        faces = [(0, 2, 1), (0, 3, 2)]        # normal -Z (outward)

    base_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]   # TL,TR,BR,BL (col 0..1 L->R, row 0..1 top->bottom)
    k = (rot // 90) % 4
    uv = base_uv[k:] + base_uv[:k]            # rotate the printed image on the plate by `rot`
    corners = list(zip(pos, uv))

    lines = []
    for (x, y), _ in corners:
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
    for _, (u, v) in corners:
        lines.append(f"vt {u:.4f} {1.0 - v:.4f}")          # OBJ vt origin = bottom-left
    for a, b, c in faces:
        lines.append(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}")
    (out / f"tag_{t['id']}.obj").write_text("\n".join(lines) + "\n")


def main():
    out = HERE / "meshes" / "tags"
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("tag_*"):           # clear old slab/png/obj artifacts
        p.unlink()
    for t in TAGS:
        save_png(t["id"], out)
        write_quad(t, out)
        print(f"tag {t['id']} on {t['body']} {t['face']}: flat decal (png+quad), 0 thickness")


if __name__ == "__main__":
    main()
