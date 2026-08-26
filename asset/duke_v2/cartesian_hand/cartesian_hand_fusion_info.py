"""Gripper-link physical properties from Fusion 360 -> cartesian_hand.xml <inertial>.

The gripper analog of asset/duke_v2/head_cam/cam_fusion_info.py: it converts
Fusion 360 per-link Properties into the <inertial> elements of cartesian_hand.xml, the
single source of truth for the gripper's mass/COM/inertia.

WHY
---
The inertials shipped in cartesian_hand.xml are CAD uniform-density estimates from the
source model. Replace them with real per-link values read in Fusion 360 (mass is
material-aware there). This bridge does the unit conversion + MuJoCo formatting.

FRAME (read Fusion in the gripper's OWN assembly frame)
-------------------------------------------------------
Every gripper body in cartesian_hand.xml is declared at identity (`pos="0 0 0"`, no
quat), so all 10 bodies share ONE frame -- the gripper assembly/base frame (the
pose you see in cartesian_hand.xml: `base` at the origin, jaws reaching +X). Read each
component's Properties in THAT assembled frame (Option A, same as the head
camera's global CNC frame) -- NOT isolated at the origin, NOT a per-body local
frame. The mount offset onto the wrist is applied separately
(humanoid_v21_constants.GRIPPER_MOUNT_POS), so it does NOT enter here.

NO MIRROR
---------
Unlike cam_fusion_info.py, there is NO left/right mirror. The gripper is ONE
part; its left/right jaws are distinct real components, so every one of the 10
bodies gets its own independent Fusion block. (Mirroring here would be a bug.)

HOW TO USE
----------
1. In Fusion, with the gripper in its assembled pose, select the component(s)
   that make up each body (the grouping must match cartesian_hand.xml's <body> split),
   open Properties, and copy the text. Each block needs at least: Mass, Center
   of Mass, and 'Moment of Inertia at Center of Mass'.
2. Paste each block into the matching entry of GRIPPER_INFO below (replace the
   "PASTE ..." placeholder).
3. Run:  ~/miniconda3/envs/mjhand/bin/python cartesian_hand_fusion_info.py            # parse + print
         ~/miniconda3/envs/mjhand/bin/python cartesian_hand_fusion_info.py --write     # write into cartesian_hand.xml

Units auto-convert to SI:  mass g->kg (1e-3) | length mm->m (1e-3) | inertia
g.mm^2->kg.m^2 (1e-9). MuJoCo fullinertia order is  ixx iyy izz ixy ixz iyz.
You can fill and --write the blocks incrementally; only filled bodies are touched.
"""
from __future__ import annotations

import re
import sys
import pathlib

import numpy as np

# ── Fusion 360 Properties, ONE block per cartesian_hand.xml <body> (assembly frame, NO
#    mirror). Replace each "PASTE ..." with a fresh Fusion export. Body names MUST
#    match <body name="..."> in cartesian_hand.xml (the 10 moving links below). ──
#
#    Template of a real block (see asset/duke_v2/head_cam/cam_fusion_info.py):
#        Part Name	base
#        Physical
#            Mass	69.5 g
#            Center of Mass	-23.57 mm, 8.66 mm, 15.27 mm
#            Moment of Inertia at Center of Mass (g mm^2)
#                ixx	... ixy	... ixz	...
#                iyx	... iyy	... iyz	...
#                izx	... izy	... izz	...
GRIPPER_INFO: dict[str, str] = {
    "base":              "\nPASTE Fusion Properties for body 'base' (assembly frame, NO mirror).\n",
    "bridge":            "\nPASTE Fusion Properties for body 'bridge' (assembly frame, NO mirror).\n",
    "left_up_rack":      "\nPASTE Fusion Properties for body 'left_up_rack' (assembly frame, NO mirror).\n",
    "left_up_finger":    "\nPASTE Fusion Properties for body 'left_up_finger' (assembly frame, NO mirror).\n",
    "right_up_rack":     "\nPASTE Fusion Properties for body 'right_up_rack' (assembly frame, NO mirror).\n",
    "right_up_finger":   "\nPASTE Fusion Properties for body 'right_up_finger' (assembly frame, NO mirror).\n",
    "left_down_rack":    "\nPASTE Fusion Properties for body 'left_down_rack' (assembly frame, NO mirror).\n",
    "left_down_finger":  "\nPASTE Fusion Properties for body 'left_down_finger' (assembly frame, NO mirror).\n",
    "right_down_rack":   "\nPASTE Fusion Properties for body 'right_down_rack' (assembly frame, NO mirror).\n",
    "right_down_finger": "\nPASTE Fusion Properties for body 'right_down_finger' (assembly frame, NO mirror).\n",
}

XML = pathlib.Path(__file__).parent / "cartesian_hand.xml"


def parse_fusion(info: str):
    """Fusion text -> (mass[kg], com[3] m, I[3x3] kg.m^2 at COM), in the read frame.

    Accepts either Fusion unit system and converts to SI:
      mass    g  -> kg (1e-3)        |  kg stays
      length  mm -> m  (1e-3)        |  cm -> m (1e-2)  |  m stays
      inertia g.mm^2 -> kg.m^2 (1e-9)|  kg.m^2 stays
    Case-insensitive keys; raises a clear error on an unfilled placeholder.
    """
    if "PASTE" in info or "Mass" not in info:
        raise ValueError("looks like a placeholder -- paste the Fusion text first "
                         "(gripper assembly frame, NO mirror)")
    mm = re.search(r'Mass\s+([\d.Ee+]+)\s*(kg|g)\b', info, re.IGNORECASE)
    if mm is None:
        raise ValueError("could not find 'Mass <value> g|kg' in the Fusion block")
    mass = float(mm.group(1)) * (1e-3 if mm.group(2).lower() == 'g' else 1.0)

    cm = re.search(r'Center of Mass\s+([-\d.Ee+]+)\s*(mm|cm|m)\s*,\s*'
                   r'([-\d.Ee+]+)\s*(?:mm|cm|m)\s*,\s*([-\d.Ee+]+)\s*(?:mm|cm|m)',
                   info, re.IGNORECASE)
    if cm is None:
        raise ValueError("could not find 'Center of Mass x, y, z' in the Fusion block")
    lscale = {'mm': 1e-3, 'cm': 1e-2, 'm': 1.0}[cm.group(2).lower()]
    com = np.array([float(cm.group(i)) for i in (1, 3, 4)]) * lscale

    hdr = re.search(r'Moment of Inertia at Center of Mass\s*\(([^)]*)\)', info, re.IGNORECASE)
    if hdr is None:
        raise ValueError("could not find 'Moment of Inertia at Center of Mass (units)'")
    iscale = 1e-9 if hdr.group(1).replace(' ', '').lower().startswith('g') else 1.0

    sec = re.search(r'Moment of Inertia at Center of Mass.*?Izz\s+[-\d.Ee+]+',
                    info, re.DOTALL | re.IGNORECASE)
    if sec is None:
        raise ValueError("could not find the full ixx..izz inertia table")
    sec = sec.group(0)

    def g(key: str) -> float:
        mt = re.search(rf'\b{key}\s+([-\d.Ee+]+)', sec, re.IGNORECASE)
        if mt is None:
            raise ValueError(f"missing inertia component {key} in the Fusion block")
        return float(mt.group(1)) * iscale

    I = np.array([[g('Ixx'), g('Ixy'), g('Ixz')],
                  [g('Ixy'), g('Iyy'), g('Iyz')],
                  [g('Ixz'), g('Iyz'), g('Izz')]])
    return mass, com, I


def inertial_xml(mass, com, I) -> str:
    # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
    fi = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
    return (f'<inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" '
            f'mass="{mass:.5f}" '
            f'fullinertia="{fi[0]:.6e} {fi[1]:.6e} {fi[2]:.6e} '
            f'{fi[3]:.6e} {fi[4]:.6e} {fi[5]:.6e}"/>')


def _filled() -> dict[str, str]:
    """The subset of GRIPPER_INFO that has real Fusion data pasted in."""
    return {b: t for b, t in GRIPPER_INFO.items() if "PASTE" not in t}


def build_inertials() -> dict[str, str]:
    """body name -> inertial XML string, for every FILLED body. NO mirror."""
    return {b: inertial_xml(*parse_fusion(t)) for b, t in _filled().items()}


def write_to_xml() -> None:
    """Replace each filled body's first <inertial .../> in cartesian_hand.xml, by name."""
    t = XML.read_text()
    inertials = build_inertials()
    for body, new_inertial in inertials.items():
        pat = re.compile(rf'(<body name="{re.escape(body)}"[^>]*>\s*)<inertial[\s\S]*?/>')
        t, n = pat.subn(lambda mm: mm.group(1) + new_inertial, t, count=1)
        if n != 1:
            raise RuntimeError(
                f"{body}: expected to replace exactly 1 <inertial>, did {n} "
                f"(check the body exists in {XML.name} and has an <inertial> child)")
    XML.write_text(t)
    print(f"wrote {len(inertials)} inertials into {XML.name}")


def main() -> None:
    filled = _filled()
    todo = [b for b in GRIPPER_INFO if b not in filled]
    for body, xml in build_inertials().items():
        print(f"{body}:\n  {xml}\n")
    masses = {b: parse_fusion(t)[0] for b, t in filled.items()}
    if masses:
        print(f"filled {len(filled)}/{len(GRIPPER_INFO)} bodies; "
              f"total mass (filled) = {sum(masses.values()) * 1000:.1f} g")
        print("per-body: " + ", ".join(f"{b}={m * 1000:.1f}g" for b, m in masses.items()))
    if todo:
        print(f"still PASTE-needed ({len(todo)}): " + ", ".join(todo))
    if "--write" in sys.argv:
        write_to_xml()


if __name__ == "__main__":
    main()
