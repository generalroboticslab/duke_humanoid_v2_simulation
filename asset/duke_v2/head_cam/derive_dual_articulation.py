"""CAD stage 2 (PROVENANCE — not part of the model build; see README.md).

Derive the 4-DOF articulation for source/HeadCameraV2_dual.step.

  IN    source/HeadCameraV2_dual.step        (regenerate with mirror_step.py first)
        meshes/components_high_res/part_*.obj + small_parts.obj  (NOT committed)
  OUT   stdout (joint axes / pitch range / inertials) + small_parts_<body>.obj
  DEPS  build123d, OCP, cadquery, trimesh, scipy — NOT installed in the project env.
  PINS  head_camera_creation.YAW_Z (0.52), PITCH_Z (0.625442), PITCH_RANGE, YAW_RANGE
        and the joint-axis directions. Re-run only if the CAD changes; nothing in the
        model imports this file.
  NOTE  both inputs are missing from the tree today, so this does NOT run as-is —
        it is kept as the audit trail for those constants.

Two identical head-camera chains (left = original at y<0, right = "_twin"
rotated 180 deg about z) on one shared CNC top plate:

  base (plate + 2 mounts + 2 yaw stators)
    yaw_left / yaw_right   (yaw rotor flange + neck + pitch stator)
      pitch_left / pitch_right  (pitch rotor flange + arm + D435 + U-joint)

Pitch travel range: symmetric +-pi/2 about a level (0 deg) zero, span pi (180 deg).
(Was an off-center baseline transcribed from the Reference project's saved CAD
pose, [-1.5657, 1.5343] rad; corrected to symmetric +-90 deg.)  Each chain's range
is that baseline shifted by the measured arm-angle difference about the pitch axis,
using the neck-down direction as the angular reference in both models (the shift is
~0 here, so the output stays symmetric).

Masses/inertia: SUPERSEDED.  Per-link mass/COM/inertia now come from Fusion 360
via cam_fusion_info.py (the single source of truth, written into the 6
<inertial> of head_camera_dual.xml).  This script is kept only for the
joint-axis and pitch-range derivation below; its mass constants
(USER_MASS_G / MOTOR_MASS_G / D435_MASS_G / EST_RHO) are no longer used by the
model and remain only as provenance for that earlier estimate.
"""
from __future__ import annotations

import numpy as np
import trimesh
from build123d import import_step
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).parent
STEP = HERE / "source" / "HeadCameraV2_dual.step"
PARTS_DIR = HERE / "meshes" / "components_high_res"

# ---- baseline transcribed from the Reference project (NewCameraModel) -------
REF_PITCH_RANGE = (-np.pi / 2, np.pi / 2)    # symmetric +-90 deg about level zero
REF_AXIS_DIR = np.array([0.0, 1.0, 0.0])     # out_dir was +y (toward arm hub)
REF_AXIS_PT = np.array([0.0, 0.0, 164.0])
REF_NECK_COM = np.array([0.001088, -5.309483, 118.844830])   # gimbal_tower
REF_D435_COM = np.array([-12.987516, 0.248753, 199.938850])  # Component5
YAW_RANGE = (-1.5 * np.pi, 1.5 * np.pi)   # ±1.5π (±270°), narrowed from ±2π to ease RL

MOTOR_MASS_G = 194.076340085   # RS05, same part as previous model
D435_MASS_G = 70.999808109     # same camera
# user-provided masses (Fusion Properties readout, 2026-06-12), grams,
# one chain side; effective density = mass / exact CAD volume per component
USER_MASS_G = {"mount": 52.0, "neck": 39.0, "arm": 25.0, "ujoint": 4.0}
# plate is the static base: aluminum ESTIMATE (dynamically irrelevant)
EST_RHO = {"plate": 2.70 / 1000.0}  # g/mm^3


def mass_props(solid):
    """(volume mm^3, com mm, inertia ABOUT COM, unit density) — OCC exact."""
    p = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid.wrapped, p)
    m = p.MatrixOfInertia()
    return (p.Mass(),
            np.array([p.CentreOfMass().X(), p.CentreOfMass().Y(),
                      p.CentreOfMass().Z()]),
            np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]))


def radial_extent(solid, axis_pt, axis_dir):
    bb = solid.bounding_box()
    corners = np.array([[x, y, z] for x in (bb.min.X, bb.max.X)
                        for y in (bb.min.Y, bb.max.Y)
                        for z in (bb.min.Z, bb.max.Z)])
    rel = corners - axis_pt
    rel -= np.outer(rel @ axis_dir, axis_dir)
    return np.linalg.norm(rel, axis=1).max()


def arm_angle(axis_pt, axis_dir, neck_com, cam_com):
    """Angle of the camera about the pitch axis, measured from neck-down."""
    def perp(v):
        v = v - (v @ axis_dir) * axis_dir
        return v / np.linalg.norm(v)
    r = perp(neck_com - axis_pt)
    u = np.cross(axis_dir, r)
    c = cam_com - axis_pt
    c = c - (c @ axis_dir) * axis_dir
    return np.arctan2(c @ u, c @ r)


def main():
    print("loading", STEP, flush=True)
    asm = import_step(str(STEP))

    chains = {"left": {}, "right": {}}
    plate = None
    for ch in asm.children:
        label = ch.label or "?"
        side = "right" if label.endswith("_twin") else "left"
        sl = list(ch.solids())
        if "top_plate" in label.lower():
            plate = ch
            continue
        key = ("mount" if "mount" in label else
               "neck" if "neck" in label else
               "arm" if "arm" in label else
               "d435" if "D435" in label else
               "ujoint" if "joint" in label.lower() else
               "motor" if "robstride" in label else None)
        if key is None:
            print("  WARNING unrecognized:", label)
            continue
        if key == "motor":
            zc = np.mean([mass_props(s)[1][2] for s in sl])
            key = "motor_yaw" if zc < 570 else "motor_pitch"
        chains[side][key] = sl
    assert plate is not None
    for side in chains:
        missing = {"mount", "neck", "arm", "d435", "ujoint",
                   "motor_yaw", "motor_pitch"} - set(chains[side])
        assert not missing, f"{side} missing {missing}"

    zhat = np.array([0.0, 0.0, 1.0])
    body_of_solid, solid_handles = [], []

    def add(solids, body, rho=None):
        for s in solids:
            v, c, i = mass_props(s)
            body_of_solid.append((c, v, i, body, rho))
            solid_handles.append(s)

    add([s for s in plate.solids()], "base", EST_RHO["plate"])

    results = {}
    for side in ("left", "right"):
        cset = chains[side]
        props = {k: [mass_props(s) for s in sl] for k, sl in cset.items()}

        # --- yaw axis from lower-motor rotor flange ---------------------------
        my = cset["motor_yaw"]
        com_xy = np.array([c[:2] for v, c, _ in props["motor_yaw"]])
        vols = np.array([v for v, _, _ in props["motor_yaw"]])
        guess = (com_xy * vols[:, None]).sum(0) / vols.sum()
        zc = np.array([c[2] for v, c, _ in props["motor_yaw"]])
        z_thr = (zc * vols).sum() / vols.sum() + 8.0  # above motor mid = output end
        guess_pt = np.array([guess[0], guess[1], 0.0])
        flange = [(s, v, c) for s, (v, c, _) in zip(my, props["motor_yaw"])
                  if radial_extent(s, guess_pt, zhat) < 19.5 and c[2] > z_thr]
        assert flange, f"{side}: no yaw rotor flange found"
        w = np.array([v for _, v, _ in flange])
        yaw_xy = (np.array([c[:2] for _, _, c in flange]) * w[:, None]).sum(0) / w.sum()
        print(f"{side}: yaw axis x={yaw_xy[0]:.3f} y={yaw_xy[1]:.3f} "
              f"({len(flange)} flange solids)", flush=True)

        # --- pitch axis: dir snapped to +-y, point from flange or fallback ----
        mp = cset["motor_pitch"]
        pcoms = np.array([c for v, c, _ in props["motor_pitch"]])
        pvols = np.array([v for v, _, _ in props["motor_pitch"]])
        centroid = (pcoms * pvols[:, None]).sum(0) / pvols.sum()
        yhat = np.array([0.0, 1.0, 0.0])
        # out_dir: toward arm hub; arm wraps the motor so use the U-joint side
        uj_com = props["ujoint"][0][1]
        out_dir = yhat if (uj_com - centroid) @ yhat > 0 else -yhat
        off = abs((uj_com - centroid) @ yhat)
        assert off > 3.0, f"{side}: U-joint axial offset ambiguous ({off:.1f}mm)"
        ax_end = [(s, v, c) for s, (v, c, _) in zip(mp, props["motor_pitch"])
                  if radial_extent(s, centroid, yhat) < 19.5
                  and (c - centroid) @ out_dir > 10.0]
        if ax_end:
            w = np.array([v for _, v, _ in ax_end])
            fl = (np.array([c for _, _, c in ax_end]) * w[:, None]).sum(0) / w.sum()
            pitch_pt = np.array([fl[0], centroid[1], fl[2]])
            src = f"{len(ax_end)} flange solids"
        else:
            pitch_pt = centroid.copy()
            src = "motor centroid FALLBACK (no flange solids in this instance)"
        print(f"{side}: pitch axis dir={out_dir} through x={pitch_pt[0]:.3f} "
              f"z={pitch_pt[2]:.3f} ({src})", flush=True)

        # --- pitch range from Reference baseline ------------------------------
        neck_com = props["neck"][0][1]
        d435_main = max(props["d435"], key=lambda t: t[0])[1]
        th_ref = arm_angle(REF_AXIS_PT, REF_AXIS_DIR, REF_NECK_COM, REF_D435_COM)
        th_new = arm_angle(pitch_pt, out_dir, neck_com, d435_main)
        d = th_new - th_ref
        rng = (REF_PITCH_RANGE[0] - d, REF_PITCH_RANGE[1] - d)
        print(f"{side}: arm angle {np.degrees(th_new):.2f}deg vs baseline "
              f"{np.degrees(th_ref):.2f}deg -> pitch range "
              f"[{rng[0]:.4f}, {rng[1]:.4f}]", flush=True)

        # --- body assignment ---------------------------------------------------
        yaw_body, pitch_body = f"yaw_{side}", f"pitch_{side}"
        # g/mm^3: known part mass over exact CAD volume
        rho_my = MOTOR_MASS_G / sum(v for v, _, _ in props["motor_yaw"])
        rho_mp = MOTOR_MASS_G / sum(v for v, _, _ in props["motor_pitch"])
        rho_d435 = D435_MASS_G / sum(v for v, _, _ in props["d435"])
        rho_user = {k: USER_MASS_G[k] / sum(v for v, _, _ in props[k])
                    for k in USER_MASS_G}
        flange_ids = {id(s) for s, _, _ in flange}
        add([s for s in my if id(s) not in flange_ids], "base", rho_my)
        add([s for s in my if id(s) in flange_ids], yaw_body, rho_my)
        add(cset["mount"], "base", rho_user["mount"])
        add(cset["neck"], yaw_body, rho_user["neck"])
        end_ids = {id(s) for s, _, _ in ax_end}
        add([s for s in mp if id(s) not in end_ids], yaw_body, rho_mp)
        add([s for s in mp if id(s) in end_ids], pitch_body, rho_mp)
        add(cset["arm"], pitch_body, rho_user["arm"])
        add(cset["d435"], pitch_body, rho_d435)
        add(cset["ujoint"], pitch_body, rho_user["ujoint"])
        results[side] = {"yaw_xy": yaw_xy, "pitch_pt": pitch_pt,
                         "pitch_dir": out_dir, "pitch_range": rng}

    # ---- part -> body via vertex vote ------------------------------------------
    import cadquery as cq
    part_files = sorted(PARTS_DIR.glob("part_*.obj"))
    part_names, pts_all, owner = [], [], []
    part_centroid = {}
    for pf in part_files:
        m = trimesh.load(pf, process=False)
        part_centroid[pf.stem] = m.bounds.mean(axis=0)
        pts_all.append(np.asarray(m.vertices))
        owner.append(np.full(len(m.vertices), len(part_names)))
        part_names.append(pf.stem)
    tree = cKDTree(np.vstack(pts_all))
    owner = np.concatenate(owner)

    votes = {n: {} for n in part_names}
    for s, (c, v, _, b, _) in zip(solid_handles, body_of_solid):
        verts, _ = cq.Solid(s.wrapped).tessellate(2.0)
        pts = np.array([(p.x, p.y, p.z) for p in verts])
        if len(pts) > 80:
            pts = pts[:: len(pts) // 80][:80]
        dist, j = tree.query(pts)
        good = j[dist < 1.0]
        if len(good) == 0:
            continue
        stem = part_names[int(np.bincount(owner[good]).argmax())]
        votes[stem][b] = votes[stem].get(b, 0.0) + v

    print("\npart -> body:", flush=True)
    part_body = {}
    for stem in part_names:
        if votes[stem]:
            best = max(votes[stem], key=votes[stem].get)
            tot = sum(votes[stem].values())
            flag = "" if votes[stem][best] / tot > 0.9 else f"  WARNING mixed {votes[stem]}"
        else:
            coms = np.array([c for c, *_ in body_of_solid])
            i = int(np.argmin(np.linalg.norm(coms - part_centroid[stem], axis=1)))
            best, flag = body_of_solid[i][3], "  (nearest-solid fallback)"
        part_body[stem] = best
        print(f"  {stem}: {best}  center={np.round(part_centroid[stem], 1)}{flag}",
              flush=True)

    # ---- small parts split -------------------------------------------------------
    small = trimesh.load(PARTS_DIR / "small_parts.obj", process=False)
    pieces = small.split(only_watertight=False)
    cents = np.array([part_centroid[n] for n in part_names])
    buckets = {}
    for p in pieces:
        c = p.bounds.mean(axis=0)
        b = part_body[part_names[np.argmin(np.linalg.norm(cents - c, axis=1))]]
        buckets.setdefault(b, []).append(p)
    for b, ps in buckets.items():
        trimesh.util.concatenate(ps).export(PARTS_DIR / f"small_parts_{b}.obj")
        print(f"small_parts_{b}: {len(ps)} pieces", flush=True)

    # ---- inertials -----------------------------------------------------------------
    print("\n<inertial> per body (kg, m, kg*m^2) — structural masses are "
          "ESTIMATES pending user data:", flush=True)
    for body in ("yaw_left", "pitch_left", "yaw_right", "pitch_right"):
        sel = [(v, c, i, rho) for c, v, i, b, rho in body_of_solid if b == body]
        masses = np.array([v * rho * 1e-3 for v, _, _, rho in sel])  # g -> kg
        M = masses.sum()
        com = sum(m * c for m, (_, c, _, _) in zip(masses, sel)) / M / 1000.0
        I = np.zeros((3, 3))
        for m, (v, c, i_com, rho) in zip(masses, sel):
            d = c / 1000.0 - com
            # i_com: unit-density mm^5 * (g/mm^3) = g*mm^2 -> kg*m^2 is 1e-9
            I += i_com * rho * 1e-9 + m * ((d @ d) * np.eye(3) - np.outer(d, d))
        evals, evecs = np.linalg.eigh(I)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        if np.linalg.det(evecs) < 0:
            evecs[:, 2] *= -1
        q = Rotation.from_matrix(evecs).as_quat()
        print(f"  {body}: mass={M:.6f} pos=({com[0]:.6f} {com[1]:.6f} {com[2]:.6f}) "
              f"quat(wxyz)=({q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}) "
              f"diaginertia=({np.format_float_scientific(evals[0], 4)} "
              f"{np.format_float_scientific(evals[1], 4)} "
              f"{np.format_float_scientific(evals[2], 4)})", flush=True)
    tot = sum(v * rho * 1e-3 for _, v, _, _, rho in body_of_solid)
    print(f"  total model mass (incl. base): {tot:.3f} kg", flush=True)

    print("\njoints:", flush=True)
    for side in ("left", "right"):
        r = results[side]
        print(f"  yaw_{side}:  axis 0 0 -1  pos ({r['yaw_xy'][0]/1000:.4f} "
              f"{r['yaw_xy'][1]/1000:.4f} 0.5)  range {YAW_RANGE}", flush=True)
        # Yaw sign flipped (axis 0 0 -1) so positive ctrl yaws in the
        # conventional sense; pitch axis signs mirror the XML after the
        # post-flip convention (left=0 -1 0, right=0 1 0) — the
        # pre-flip out_dir from the mesh analysis is the opposite of
        # this for both sides, so flip uniformly.
        pitch_sign = -1
        pitch_axis = tuple((pitch_sign * r["pitch_dir"]).astype(int))
        print(f"  pitch_{side}: axis {pitch_axis}  "
              f"pos ({r['pitch_pt'][0]/1000:.4f} 0 {r['pitch_pt'][2]/1000:.4f})  "
              f"range ({r['pitch_range'][0]:.4f}, {r['pitch_range'][1]:.4f})",
              flush=True)


if __name__ == "__main__":
    main()
