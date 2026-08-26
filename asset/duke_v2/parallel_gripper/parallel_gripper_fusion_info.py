"""Gripper per-body physical properties from Fusion 360 -> parallel_gripper.xml <inertial>.

SINGLE SOURCE OF TRUTH for the gripper mass/COM/inertia, stored the SAME way as
asset/duke_v2/head_cam/cam_fusion_info.py and cartesian_hand_v2: raw Fusion 360
"Properties" text in Python string constants, parsed by parse_fusion(). NO JSON. The 3
gripper-link inertials are baked into parallel_gripper.xml by write_to_xml()
(run with --write); the decoupled mount flange (FLANGE_INFO) is parsed live by the loader
(mj_envs/asset_zoo/parallel_gripper.py:_attach_flanges).

parallel_gripper = LONG-STROKE jaws (travel 0.0847 m). 2026-07-10 ParallelGripper0710
re-export, real-world-synced materials: base 170.6->171.6 g (now INCLUDES the two tag
holders, their 4 pads and the usb_c_protector as CAD parts), each rack 76.7->74.4 g;
cnc_flange is the silver mounting disc (unchanged, kept from the 0626 measurement).
Grouping: mini_gripper_old/ParallelGripper0710/parallelgripper_groups.json.

Assembly frame, NO mirror: every body is declared at identity in the XML, so all blocks
are read in the one gripper base/assembly frame (same world frame as the STEP exports, so
COM is world-coord). To update: in Fusion select the components for a body, open
Properties, copy Mass / Center of Mass / 'Moment of Inertia at Center of Mass', paste into
the matching string below, then re-run parallel_gripper_creation.py.

AUTHORED-POSE REPOSE (racks only): the 0710 STEP authors the jaws ~3.025 mm wider per
side than the canonical pose the repo meshes/XML use (q=0 = canonical). The rack meshes
were KEPT at the canonical pose (vertex-identical after the shift — see
mini_gripper_old/ParallelGripper0710/verify_repose.py), so the rack COMs measured by
Fusion in the 0710 pose must be shifted back by REPOSE_SHIFT_M below. The inertia tensor
at the COM is translation-invariant, so only the COM moves. Consumers must read gripper
links through gripper_inertial(), never parse_fusion() directly.

Units auto-convert to SI:  mass g->kg | length mm->m | inertia g.mm^2->kg.m^2."""
from __future__ import annotations
import re
import sys
import pathlib
import numpy as np


# === gripper links: Fusion 360 Properties (assembly frame, NO mirror) ===
# ParallelGripper0710 export (2026-07-10, real-world-synced). Volume kept as a grouping
# cross-check: it must equal the summed STEP solid volumes of the body's component group
# (base 99745.1 / rack 76522.7 mm^3 — verified against the probe CSV).
base_info = """
Part Name	base
Material Name	(Various)
Physical
	Mass	171.609431065 g
	Volume	99745.117104298 mm^3
	Center of Mass	0.733467541 mm, 11.956278225 mm, -9.645511698 mm
	Moment of Inertia at Center of Mass (g mm^2)
		Ixx	2.073E+05
		Ixy	8373.269579709
		Ixz	11.670569426
		Iyx	8373.269579709
		Iyy	50753.915875713
		Iyz	-7398.304312593
		Izx	11.670569426
		Izy	-7398.304312593
		Izz	2.230E+05
"""

# Rack COMs below are in the 0710 AUTHORED pose — gripper_inertial() applies REPOSE_SHIFT_M.
left_rack_info = """
Part Name	left_rack
Material Name	(Various)
Physical
	Mass	74.36563558 g
	Volume	76522.709555269 mm^3
	Center of Mass	46.202481705 mm, -25.025568322 mm, -3.857337014 mm
	Moment of Inertia at Center of Mass (g mm^2)
		Ixx	93892.122707037
		Ixy	34513.757907052
		Ixz	13199.304934191
		Iyx	34513.757907052
		Iyy	78361.884365731
		Iyz	-13218.490543166
		Izx	13199.304934191
		Izy	-13218.490543166
		Izz	1.504E+05
"""

right_rack_info = """
Part Name	right_rack
Material Name	(Various)
Physical
	Mass	74.36563558 g
	Volume	76522.709555269 mm^3
	Center of Mass	46.161822086 mm, 38.971296059 mm, -16.142662986 mm
	Moment of Inertia at Center of Mass (g mm^2)
		Ixx	93845.969855863
		Ixy	-34424.758862223
		Ixz	-13180.733443389
		Iyx	-34424.758862223
		Iyy	78190.499591115
		Iyz	-13209.204797765
		Izx	-13180.733443389
		Izy	-13209.204797765
		Izz	1.502E+05
"""

# === decoupled mount flange: Fusion 360 Properties (visual-only welded body) ===
# The "cnc" part (CNC.step shaft coupler) IS this gripper's mounting flange — the silver
# disc with the bolt-hole ring at the -X end.
# It is DECOUPLED from the gripper XML (lives in flanges/cnc_flange.obj, composed onto `base`
# at load by the loader's _attach_flanges). Parsed live by the loader (NOT baked into the XML).
cnc_flange_info = """
Part Name	cnc_flange
Material Name	Aluminum 6061
Physical
	Mass	22.054752 g
	Center of Mass	-23.426420 mm, 5.999983 mm, -9.999999 mm
	Moment of Inertia at Center of Mass (g mm^2)
		ixx	5647.755085
		ixy	0.000000
		ixz	0.000000
		iyx	0.000000
		iyy	3517.588985
		iyz	0.000000
		izx	0.000000
		izy	0.000000
		izz	3218.100749
"""

# 3 gripper links are baked into the XML by the creation script; flange parsed live by loader.
GRIPPER_INFO = {
    "base": base_info,
    "left_rack": left_rack_info,
    "right_rack": right_rack_info,
}

# Canonical-pose compensation for the rack COMs (see module docstring). The 0710 STEP
# authors the LEFT jaw 3.025382 mm further -Y (and RIGHT +Y) than the canonical mesh
# pose; canonical = authored + shift. Exact value from the vertex-set alignment of the
# 0710 export against the canonical rack meshes (residual 7e-6 mm).
REPOSE_SHIFT_M = {
    "left_rack":  np.array([0.0, +3.025382e-3, 0.0]),
    "right_rack": np.array([0.0, -3.025382e-3, 0.0]),
}


def gripper_inertial(key: str):
    """(mass, com, I) for a gripper link in the CANONICAL mesh frame.

    parse_fusion() + REPOSE_SHIFT_M on the COM (inertia at COM is translation-invariant).
    The only correct way to consume GRIPPER_INFO — parse_fusion() alone returns the rack
    COMs in the 0710 authored pose, which does NOT match the repo meshes.
    """
    mass, com, inertia = parse_fusion(GRIPPER_INFO[key])
    if key in REPOSE_SHIFT_M:
        com = com + REPOSE_SHIFT_M[key]
    return mass, com, inertia

FLANGE_INFO = {
    "cnc_flange": cnc_flange_info,
}


XML = pathlib.Path(__file__).parent / "parallel_gripper.xml"


def parse_fusion(info):
    # Fusion text -> (mass[kg], com[3] m, I[3x3] kg.m^2 at COM). g/mm/g.mm^2 -> SI.
    if "PASTE" in info or "Mass" not in info:
        raise ValueError("looks like a placeholder -- paste the Fusion text first")
    mm = re.search(r"Mass\s+([\d.Ee+]+)\s*(kg|g)\b", info)
    mass = float(mm.group(1)) * (1e-3 if mm.group(2) == "g" else 1.0)
    cm = re.search(r"Center of Mass\s+([-\d.Ee+]+)\s*(mm|cm|m)\s*,\s*"
                   r"([-\d.Ee+]+)\s*(?:mm|cm|m)\s*,\s*([-\d.Ee+]+)\s*(?:mm|cm|m)", info)
    lscale = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[cm.group(2)]
    com = np.array([float(cm.group(i)) for i in (1, 3, 4)]) * lscale
    hdr = re.search(r"Moment of Inertia at Center of Mass\s*\(([^)]*)\)", info).group(1).replace(" ", "").lower()
    iscale = 1e-9 if hdr.startswith("g") else 1.0   # g.mm^2 -> kg.m^2 (kg.m^2 starts with 'k')
    sec = re.search(r"Moment of Inertia at Center of Mass.*?Izz\s+[-\d.Ee+]+", info, re.DOTALL | re.IGNORECASE).group(0)
    g = lambda k: float(re.search(rf"\b{k}\s+([-\d.Ee+]+)", sec, re.IGNORECASE).group(1)) * iscale
    I = np.array([[g("Ixx"), g("Ixy"), g("Ixz")],
                  [g("Ixy"), g("Iyy"), g("Iyz")],
                  [g("Ixz"), g("Iyz"), g("Izz")]])
    return mass, com, I


def inertial_xml(mass, com, I):
    # MuJoCo fullinertia order: ixx iyy izz ixy ixz iyz
    fi = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
    return (f'<inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{mass:.5f}" '
            f'fullinertia="{fi[0]:.6e} {fi[1]:.6e} {fi[2]:.6e} {fi[3]:.6e} {fi[4]:.6e} {fi[5]:.6e}"/>')


def build_inertials():
    return {b: inertial_xml(*gripper_inertial(b)) for b in GRIPPER_INFO}


def write_to_xml():
    # Replace each gripper link's first <inertial .../> in parallel_gripper.xml, by name.
    t = XML.read_text()
    inertials = build_inertials()
    for body, new_inertial in inertials.items():
        pat = re.compile(rf'(<body name="{re.escape(body)}"[^>]*>\s*)<inertial[\s\S]*?/>')
        t, n = pat.subn(lambda mm: mm.group(1) + new_inertial, t, count=1)
        if n != 1:
            raise RuntimeError(f"{body}: expected to replace exactly 1 <inertial>, did {n}")
    XML.write_text(t)
    print(f"wrote {len(inertials)} link inertials into {XML.name}")


def main():
    tot = 0.0
    for b in GRIPPER_INFO:
        m, c, _ = gripper_inertial(b); tot += m
        print(f"  {b:14s} mass={m * 1000:7.2f} g  com(m)={np.round(c, 4)}  (canonical pose)")
    for b, t in FLANGE_INFO.items():
        m, c, _ = parse_fusion(t)
        print(f"  {b:14s} mass={m * 1000:7.2f} g  com(m)={np.round(c, 4)}  (flange, parsed live by loader)")
    print(f"\n{len(GRIPPER_INFO)} gripper links total mass = {tot * 1000:.1f} g")
    if "--write" in sys.argv:
        write_to_xml()


if __name__ == "__main__":
    main()
