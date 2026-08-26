"""Measurement tool (PROVENANCE — not part of the model build; see README.md).

Extract component placements from a Fusion STEP and report poses relative
to the CNC top plate (the robot's body-top mounting plate).

Method (same as the Reference PositionDeter doc): each assembly instance is
placed via NEXT_ASSEMBLY_USAGE_OCCURRENCE -> (PRODUCT_DEFINITION_SHAPE ->
CONTEXT_DEPENDENT_SHAPE_REPRESENTATION) -> REPRESENTATION_RELATIONSHIP_WITH_
TRANSFORMATION -> ITEM_DEFINED_TRANSFORMATION -> AXIS2_PLACEMENT_3D
(origin point + Z axis + X axis).  Relative pose = inv(T_plate) * T_child.

  IN    source/HeadCameraV2.step (default) or any STEP passed as argv[1]
  OUT   stdout only; the 2026-06-12 run is transcribed in
        PositionDeter/RELATIVE_POSITION_top_plate__head_cameras.md
  DEPS  numpy only — pure-regex STEP parsing, NO CAD kernel. Runs in the project env.
  PINS  the mount geometry the head_camera_creation constants sit on
        (YAW_ANCHOR_Y=0.065 column spacing, YAW_Z, PITCH_Z heights).

Usage: python extract_relative_positions.py [file.step]
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STEP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "source" / "HeadCameraV2.step"


def parse_entities(text):
    data = text[text.index("DATA;"):]
    body = data.replace("\r", "").replace("\n", " ")
    ents = {}
    for rec in body.split(";"):
        m = re.match(r"\s*#(\d+)\s*=\s*([A-Z_0-9]+)\s*\((.*)\)\s*$", rec, re.S)
        if m:
            ents[int(m.group(1))] = (m.group(2), m.group(3))
            continue
        # complex (multi-typed) instance: #id=(TYPE1(..)TYPE2(..)...)
        m = re.match(r"\s*#(\d+)\s*=\s*\((.*)\)\s*$", rec, re.S)
        if m:
            ents[int(m.group(1))] = ("$COMPLEX", m.group(2))
    return ents


def refs(args):
    return [int(x) for x in re.findall(r"#(\d+)", args)]


def first_string(args):
    m = re.match(r"\s*'((?:[^']|'')*)'", args)
    return m.group(1).replace("''", "'") if m else None


def main():
    text = STEP.read_text(encoding="utf-8", errors="replace")
    ents = parse_entities(text)
    rev = defaultdict(list)
    for eid, (t, a) in ents.items():
        for r in refs(a):
            rev[r].append(eid)

    def axis2_frame(eid):
        """AXIS2_PLACEMENT_3D -> (origin, z, x) as numpy arrays."""
        rs = refs(ents[eid][1])
        pt = next(r for r in rs if ents[r][0] == "CARTESIAN_POINT")
        dirs = [r for r in rs if ents[r][0] == "DIRECTION"]
        def vec(e):
            return np.array([float(x) for x in
                             re.findall(r"[-+0-9.Ee]+", ents[e][1].split("(", 1)[1])][:3])
        origin = vec(pt)
        z = vec(dirs[0]) if dirs else np.array([0.0, 0, 1])
        x = vec(dirs[1]) if len(dirs) > 1 else np.array([1.0, 0, 0])
        return origin, z, x

    def pose_matrix(origin, z, x):
        z = z / np.linalg.norm(z)
        x = x - (x @ z) * z
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, origin
        return T

    def etype_of(r):
        return ents.get(r, ("", ""))[0]

    # SHAPE_REPRESENTATION id -> product name, via
    # SHAPE_DEFINITION_REPRESENTATION(#PRODUCT_DEFINITION_SHAPE, #rep)
    rep_product = {}
    for eid, (t, a) in ents.items():
        if t != "SHAPE_DEFINITION_REPRESENTATION":
            continue
        rs = refs(a)
        pds = [r for r in rs if etype_of(r) == "PRODUCT_DEFINITION_SHAPE"]
        reps = [r for r in rs if etype_of(r) == "SHAPE_REPRESENTATION"]
        if not pds or not reps:
            continue
        name = None
        for pd in refs(ents[pds[0]][1]):
            if etype_of(pd) == "PRODUCT_DEFINITION":
                for f in refs(ents[pd][1]):
                    if etype_of(f) == "PRODUCT_DEFINITION_FORMATION":
                        for pr in refs(ents[f][1]):
                            if etype_of(pr) == "PRODUCT":
                                name = first_string(ents[pr][1])
        if name:
            rep_product[reps[0]] = name

    # complex RRWT entities: (parent_rep, child_rep) + IDT -> child pose
    poses = {}
    for eid, (t, a) in ents.items():
        if "COMPLEX" not in t or "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION" not in a:
            continue
        reps = [r for r in refs(a) if etype_of(r) == "SHAPE_REPRESENTATION"]
        idts = [r for r in refs(a) if etype_of(r) == "ITEM_DEFINED_TRANSFORMATION"]
        if len(reps) != 2 or len(idts) != 1:
            continue
        names = [rep_product.get(r, f"rep#{r}") for r in reps]
        # the root assembly is the parent; keep only direct children of it
        if "twincities" not in names[0] and "twincities" not in names[1]:
            continue
        child = names[0] if "twincities" in names[1] else names[1]
        placements = [r for r in refs(ents[idts[0]][1])
                      if etype_of(r) == "AXIS2_PLACEMENT_3D"]
        if len(placements) != 2:
            continue
        src, dst = placements
        key = child
        n = 2
        while key in poses:  # second instance of same product (the two motors)
            key = f"{child}#{n}"
            n += 1
        poses[key] = (pose_matrix(*axis2_frame(dst)),
                      pose_matrix(*axis2_frame(src)), idts[0], dst)

    print(f"{len(poses)} placed instances found:")
    for name, (T, Ts, idt, dst) in sorted(poses.items()):
        o = np.round(T[:3, 3], 4)
        rot_is_identity = np.allclose(T[:3, :3], np.eye(3), atol=1e-9)
        print(f"  {name}: origin={tuple(o)} rot={'identity' if rot_is_identity else 'NON-IDENTITY'}"
              f"  (IDT #{idt}, target AXIS2 #{dst})")
        if not rot_is_identity:
            print(np.round(T[:3, :3], 6))

    # relative pose: plate -> others
    plate_key = next(k for k in poses if "top_plate" in k.lower())
    Tp = poses[plate_key][0]
    print(f"\nrelative to plate instance '{plate_key}':")
    for name, (T, *_ ) in sorted(poses.items()):
        rel = np.linalg.inv(Tp) @ T
        o = np.round(rel[:3, 3], 4)
        ident = np.allclose(rel[:3, :3], np.eye(3), atol=1e-9)
        print(f"  {name}: t={tuple(o)} mm  rot={'identity' if ident else 'see matrix'}")
        if not ident:
            print(np.round(rel[:3, :3], 6))


if __name__ == "__main__":
    main()
