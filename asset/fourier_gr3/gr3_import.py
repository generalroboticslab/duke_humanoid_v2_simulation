"""Import the vendor Fourier GR-3 v2.1.1 MJCF into this repo's study conventions.

    python asset/fourier_gr3/gr3_import.py          # -> asset/fourier_gr3/gr3.xml

``gr3.xml`` is a GENERATED artifact — never hand-edit it; edit this script and re-run.
The vendor tree (``~/Documents/Wiki-GRx-Models``) is READ-ONLY and is never written to.

WHY A SCRIPT AND NOT A HAND-EDITED COPY
---------------------------------------
The vendor ships GR-3 as a display model: fixed base, no actuators, no sites, no camera,
and geom groups that clash with this study's conventions. Every transform below is a
mechanical, re-runnable rule with a stated reason, so a vendor model update is a re-run
rather than a re-audit. Provenance and the numbers' derivations live here, next to the
code that applies them.

SOURCE
------
``Wiki-GRx-Models/GRX/GR3/gr3v2_1_1/mjcf/gr3v2_1_1_dummy_hand.xml`` (FFTAI / Fourier
Intelligence). The ``_dummy_hand`` variant is used, not the bare-wrist one: the study's
SOP requires a FIXED end-effector site at the tool centre and forbids planning to a moving
finger link, and the rigid dummy hand is exactly such a fixed body (it also carries the
hand's real 0.4686 kg and a collision box, both of which the bare-wrist file lacks).

TRANSFORMS APPLIED (each with its justification)
------------------------------------------------
1. FLOATING ROOT + STANDING HEIGHT. The vendor model is fixed-base. A ``<freejoint>`` is
   added to ``base_link`` and the body is raised by ``STAND_HEIGHT_M`` so the feet touch
   z=0. 0.9270331 m is the exact analytic lowest point of any COLLISION geom at qpos0
   (``cylinder_foot_1`` on ``right_foot_pitch_link``), cross-checked by contact bisection.
   NOTE the visual foot meshes reach 1.5 mm lower (0.9285355); the collision value is used
   because the payload, the IK collision model and the section-cut floor all key off
   collision geometry. The 1.5 mm visual interpenetration is a documented, deliberate gap.

2. STANDING KEYFRAME. The vendor model declares no keyframe (nkey=0), so consumers had no
   canonical pose. ``standing`` pins the free root at the height above and every hinge at 0.

3. TOOL SITES. ``end_effector_{L,R}_site`` on the ``*_end_effector_link`` bodies (NOT on
   the dummy-hand bodies, although they are coincident: the end-effector links also exist
   in the bare-wrist vendor file, so the site definition survives swapping in a real
   gripper). Position = the centroid of the dummy hand's distal end-face vertex cluster
   (all vertices within 5 mm of z_min), the same "mesh distal tip" convention Booster T1
   uses. Measured R=(0.00872, +0.01911, -0.17634), L=(0.00828, -0.01931, -0.17631); the two
   agree under Y-mirroring to 0.44 mm in x, 0.21 mm in y and 0.02 mm in z (mesh
   triangulation noise), so the values are SYMMETRISED to make the L/R mirror exact — the
   whole downstream chain (solve-R-mirror-L, ``--mirror-left``, the visibility Y-fold)
   assumes exact symmetry.

4. GEOM GROUPS. Vendor uses group 1 for the 37 visual meshes and no group (=0) for the 22
   collision primitives. This study requires visual=2 / collision=3 / camera-FOV=4 /
   cuRobo-spheres=5: the URDF exporter's GROUP_ROLE map has no entry for 0 or 1 and would
   reject every body, and the visibility raycast probes group 2 only — leaving the meshes
   in group 1 would silently report every target as unoccluded.

5. HEAD CAMERA. The vendor model has no ``<camera>`` at all (ncam=0). Two cameras are
   declared on the existing ``camera_link`` body (see CAMERA CONTRACT below), plus a
   co-located site per camera to serve as the occlusion-raycast origin, plus a group-4
   wireframe of the primary camera's frustum for visual inspection.

CAMERA CONTRACT (SOP §3)
------------------------
Fourier publishes NO model number, resolution, intrinsics or depth range for the GR-3 head
camera. What IS first-hand vendor data, from the official developer docs
(https://support.fftai.com/en/getting-started/general-information, GR-3 overview §7):

  * the two FOV figures are captioned "OAK Camera - FOV - Vertical" / "- Horizontal", i.e.
    the vendor names it a Luxonis OAK (DepthAI) camera. GR-1's page instead says
    "Depth Camera: Realsense" and carries no OAK figure, so this is GR-3-specific and not a
    copied template;
  * ``overview_image7.png`` (side view) is annotated 80° and ``overview_image8.png`` (top
    view) 128°. Both were downloaded and measured directly: the side view's upper ray is
    horizontal to 0.13° and its lower ray sits 80.07° below horizontal; the top view shows
    TWO lens apertures ~70 mm apart, each with its own 128° cone.

128° H x 80° V with a ~70 mm baseline matches the Luxonis OAK-D W (127.7° x 79.5°, 75 mm)
closely, but the vendor never states a part number, so no SKU is claimed anywhere in this
repo. The spec table's "Monocular camera" row contradicts the two apertures its own figure
draws; we model a SINGLE eye, which is the conservative reading (a stereo OR could only
increase visibility, and at a 70 mm baseline the difference is negligible at workspace
range).

⚠ MOUNT-PITCH CONFLICT — the reason two cameras are declared. The vendor URDF's
``camera_joint`` is rpy (0, 0.2618, 0) = 15.0° nose-down, but the official side-view FOV
figure is drawn with the optical axis 40° nose-down (its cone runs from horizontal to
80.07° down). The figure is NOT drawn with the neck pitched: its camera-height-to-total-
height ratio is 0.9701 against the model's 0.9718 at neutral pitch, whereas pitching the
neck down drives that ratio to 0.985+. So the 25° gap is a genuine vendor inconsistency,
plausibly ``camera_link`` being the mounting-face frame rather than the optical frame. At
1 m range 25° displaces the visible cone by >0.4 m, so it cannot be waved away:

  * ``head_cam``        — optical axis = ``camera_link`` +X, i.e. the URDF's 15°. PRIMARY.
  * ``head_cam_fovfig`` — pitched a further 25° down, reproducing the official figure's 40°.

Both are scored and reported as a sensitivity pair; neither is presented as calibrated.

FOV ENCODING. 128° x 80° is not a pinhole-consistent pair (tan64/tan40 = 2.443 is no
standard sensor aspect), which is expected of a wide/fisheye lens spec. This study's FOV
gate (``t1_visibility._camera_intrinsics``) builds an isotropic pinhole from ``fovy`` and
``resolution``: fy = (h/2)/tan(fovy/2), fx = fy, so the vertical envelope is ``fovy`` and
the horizontal one is derived from the aspect ratio. ``fovy=80`` with resolution
1955 x 800 therefore reproduces 80.0° V and 128.01° H exactly. The RESOLUTION IS AN
ENVELOPE-ENCODING DEVICE, NOT A CLAIMED SENSOR RESOLUTION — the vendor publishes none.
The resulting rectangular envelope has a 131.4° diagonal, comfortably inside the OAK-D W
class's ~150° diagonal, so it does not over-claim at the corners.
"""
from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
VENDOR = pathlib.Path.home() / "repo/Wiki-GRx-Models/GRX/GR3/gr3v2_1_1"
SRC_XML = VENDOR / "mjcf" / "gr3v2_1_1_dummy_hand.xml"
OUT_XML = HERE / "gr3.xml"

# --- measured constants (derivations in the module docstring) ---------------------------
STAND_HEIGHT_M = 0.9270331          # analytic lowest collision point at qpos0
TOOL_SITE_XZ = (0.00850, -0.17632)  # symmetrised (x, z) of the dummy-hand distal tip
TOOL_SITE_Y = 0.01921               # +y on the RIGHT hand, -y on the LEFT

# Camera. quat maps the MuJoCo camera's local -Z (its view axis) onto the desired direction
# expressed in camera_link coordinates; camera_link's own +X already carries the URDF's 15°.
CAM_FOVY_DEG = 80.0
CAM_RES = (1955, 800)               # encodes 128.01° H (see docstring); NOT a sensor spec
QUAT_15 = (0.70710678, 0.0, -0.70710678, 0.0)          # Ry(-90°): view axis = camera_link +X
QUAT_40 = (0.84339145, 0.0, -0.53729961, 0.0)          # Ry(-65°): 25° further nose-down
FOV_NEAR_M, FOV_FAR_M = 0.10, 0.60  # group-4 wireframe extent (diagnostic, not a range claim)


def _q_to_str(q) -> str:
    return " ".join(f"{v:.8g}" for v in q)


def _iter_bodies(root: ET.Element):
    for body in root.iter("body"):
        yield body


def build() -> ET.ElementTree:
    tree = ET.parse(SRC_XML)
    root = tree.getroot()
    root.set("model", "gr3v2_1_1")

    # 1. meshes resolve against our own assets/ copy, by basename
    compiler = root.find("compiler")
    compiler.set("meshdir", "assets")
    for mesh in root.iter("mesh"):
        mesh.set("file", pathlib.PurePosixPath(mesh.get("file")).name)

    # 2. geom groups -> study convention (visual 2 / collision 3)
    n_vis = n_col = 0
    for geom in root.iter("geom"):
        if geom.get("group") == "1":
            geom.set("group", "2")
            n_vis += 1
        elif geom.get("group") is None:
            geom.set("group", "3")
            n_col += 1
        else:
            raise SystemExit(f"unexpected vendor geom group {geom.get('group')!r}")

    bodies = {b.get("name"): b for b in _iter_bodies(root)}

    # 3. floating root + standing height
    base = bodies["base_link"]
    if base.get("pos") not in (None, "0 0 0"):
        raise SystemExit("vendor base_link already carries a pos; re-derive STAND_HEIGHT_M")
    base.set("pos", f"0 0 {STAND_HEIGHT_M}")
    free = ET.Element("freejoint", {"name": "floating_base"})
    base.insert(0, free)

    # 4. tool sites (fixed, on the end-effector links)
    x, z = TOOL_SITE_XZ
    for side, y in (("R", +TOOL_SITE_Y), ("L", -TOOL_SITE_Y)):
        parent = bodies[f"{'right' if side == 'R' else 'left'}_end_effector_link"]
        ET.SubElement(parent, "site", {
            "name": f"end_effector_{side}_site",
            "pos": f"{x:.5f} {y:+.5f} {z:.5f}",
            "size": "0.006", "rgba": "1 0.2 0.2 0.6" if side == "R" else "0.2 0.4 1 0.6",
        })

    # 5. cameras + raycast-origin sites on camera_link
    cam_body = bodies["camera_link"]
    for name, quat, rgba in (("head_cam", QUAT_15, "0 0 1 0.6"),
                             ("head_cam_fovfig", QUAT_40, "1 0.6 0 0.6")):
        ET.SubElement(cam_body, "camera", {
            "name": name, "pos": "0 0 0", "quat": _q_to_str(quat),
            "fovy": f"{CAM_FOVY_DEG:g}", "resolution": f"{CAM_RES[0]} {CAM_RES[1]}",
        })
        ET.SubElement(cam_body, "site", {
            "name": f"{name}_site", "pos": "0 0 0", "quat": _q_to_str(quat),
            "size": "0.005", "rgba": rgba,
        })
    _add_fov_wireframe(cam_body, QUAT_15)

    # 6. exact L/R mirror symmetry
    n_sym = _symmetrise_lr(bodies)

    # 7. refit the upper-body collision model to the visual meshes
    n_fit = _refit_upper_body_collision(bodies)

    # 8. standing keyframe (free root + 31 hinges at 0)
    n_hinge = sum(1 for j in root.iter("joint"))
    qpos = f"0 0 {STAND_HEIGHT_M} 1 0 0 0 " + " ".join(["0"] * n_hinge)
    kf = ET.Element("keyframe")
    ET.SubElement(kf, "key", {"name": "standing", "qpos": qpos})
    root.append(kf)

    print(f"[import] geom groups: {n_vis} visual -> 2, {n_col} collision -> 3")
    print(f"[import] symmetrised {n_sym} left/right body pair(s)")
    print(f"[import] refitted collision on {n_fit} upper-body link(s)")
    print(f"[import] hinges found: {n_hinge} (keyframe qpos length {7 + n_hinge})")
    return tree


def _mesh_aabb(body: ET.Element) -> tuple[np.ndarray, np.ndarray] | None:
    """AABB (centre, half-extents) of a link's group-2 visual meshes, in the LINK frame.

    Every vendor visual geom sits at the body origin with identity orientation (verified:
    no ``pos``/``quat`` on any ``type="mesh"`` geom), so the raw STL vertices are already
    link-frame coordinates and no geom transform is needed.
    """
    import trimesh

    pts = []
    for geom in body.findall("geom"):
        if geom.get("type") != "mesh" or geom.get("group") != "2":
            continue
        assert geom.get("pos") is None and geom.get("quat") is None, "visual geom has a pose"
        pts.append(trimesh.load(HERE / "assets" / f"{geom.get('mesh')}.STL",
                                force="mesh").vertices)
    if not pts:
        return None
    v = np.vstack(pts)
    lo, hi = v.min(axis=0), v.max(axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _refit_upper_body_collision(bodies: dict[str, ET.Element]) -> int:
    """Replace the vendor's under-sized upper-body collision primitives with mesh fits.

    WHY. Measured against the visual meshes, the vendor collision model encloses only 20.4%
    of the robot's surface vertices (16.4% across the arm links) — against 50.3% / 67.6% for
    Booster T1, the closest already-integrated robot in this study. The torso is the worst
    case (a 200x150x200 mm box inside a 273x298x471 mm shell) and the forearm, wrist-pitch
    and wrist-roll links carry NO collision geometry at all. Left alone, the arm sweeps
    through the chest and through its own elbow undetected, and the reachable workspace —
    the quantity this study reports — is over-stated exactly in front of the body.

    HOW. Fits are INSCRIBED, never circumscribed, so an under-approximation is corrected
    without creating an over-approximation: a limb capsule runs along the mesh AABB's
    longest axis and takes as its radius the SMALLER of the two remaining half-extents, so
    it can never be wider than the link itself; blocky links take the mesh AABB as a box.
    Nothing is inflated.

    SCOPE. Legs keep their vendor primitives — the arm-only cuRobo model prunes everything
    below the hips regardless. The head keeps its vendor sphere (68.6% coverage already, and
    a sphere is the right shape for it).

    SYMMETRY. Each pair is fitted on the RIGHT link and mirrored to the left: a per-side
    argmax flips the chosen capsule axis whenever two extents are near-equal (it does, on
    ``upper_arm_pitch``), which would silently undo the exact L/R symmetry set above.
    """
    capsule_links = ("upper_arm_pitch_link", "upper_arm_roll_link", "upper_arm_yaw_link",
                     "lower_arm_pitch_link", "hand_yaw_link", "hand_pitch_link",
                     "hand_roll_link")
    box_links = ("base_link", "waist_yaw_link", "waist_pitch_link", "torso_link")
    hand_pair = ("dummy_right_hand_link", "left_dummy_hand_link")  # vendor names are asymmetric

    def clear_collision(body: ET.Element) -> None:
        for geom in [g for g in body.findall("geom") if g.get("group") == "3"]:
            body.remove(geom)

    def add_box(body: ET.Element, name: str, c, h) -> None:
        ET.SubElement(body, "geom", {
            "name": name, "type": "box", "group": "3",
            "size": " ".join(f"{v:.6g}" for v in h),
            "pos": " ".join(f"{v:.6g}" for v in c),
            "rgba": "1 1 1 1",
        })

    def add_capsule(body: ET.Element, name: str, c, h) -> None:
        axis = int(np.argmax(h))
        radius = float(min(h[i] for i in range(3) if i != axis))
        half = max(float(h[axis]) - radius, 1e-4)
        d = np.zeros(3)
        d[axis] = half
        ET.SubElement(body, "geom", {
            "name": name, "type": "capsule", "group": "3",
            "size": f"{radius:.6g}",
            "fromto": " ".join(f"{v:.6g}" for v in (*(c - d), *(c + d))),
            "rgba": "1 1 1 1",
        })

    n = 0
    for suffix in capsule_links:
        right, left = bodies[f"right_{suffix}"], bodies[f"left_{suffix}"]
        fit = _mesh_aabb(right)
        assert fit is not None, f"right_{suffix} has no visual mesh to fit"
        c, h = fit
        for body, sign in ((right, +1.0), (left, -1.0)):
            clear_collision(body)
            add_capsule(body, f"{'right' if sign > 0 else 'left'}_{suffix}_col",
                        c * np.array([1.0, sign, 1.0]), h)
            n += 1
    for name in box_links:
        body = bodies[name]
        fit = _mesh_aabb(body)
        assert fit is not None, f"{name} has no visual mesh to fit"
        clear_collision(body)
        add_box(body, f"{name}_col", *fit)
        n += 1
    fit = _mesh_aabb(bodies[hand_pair[0]])
    assert fit is not None, "dummy hand has no visual mesh to fit"
    c, h = fit
    for name, sign in zip(hand_pair, (+1.0, -1.0)):
        clear_collision(bodies[name])
        add_box(bodies[name], f"{name}_col", c * np.array([1.0, sign, 1.0]), h)
        n += 1
    return n


def _symmetrise_lr(bodies: dict[str, ET.Element]) -> int:
    """Force every ``left_*``/``right_*`` body pair to be an exact mirror about y=0.

    The vendor model is *nearly* symmetric but not exactly: ``hand_roll_link`` differs by
    0.4997 mm in y (right carries ``pos="0 -0.00049967 0"``, left carries none),
    ``upper_arm_pitch_link`` by 0.01 mm in y and ``foot_pitch_link`` by 0.08 mm in x. The
    whole downstream chain assumes exact symmetry -- reachability is solved for the RIGHT
    arm only and the left is derived by Y-mirroring, ``plot_workspace_curobo --mirror-left``
    takes ``max(D_R, mirror(D_R))``, and the dynamic visibility scorer folds targets into
    the +y half-space. A sub-millimetre bias is physically irrelevant but makes those
    identities false, so it is removed here rather than silently tolerated.

    x and z are averaged, y becomes +-mean(|y|). Body quaternions are already exact mirrors
    ((w,x,y,z) -> (w,-x,y,-z)) and are left untouched.
    """
    n = 0
    for name, left in bodies.items():
        if not name.startswith("left_"):
            continue
        right = bodies.get("right_" + name[5:])
        if right is None:
            continue
        lp = np.fromstring(left.get("pos", "0 0 0"), sep=" ")
        rp = np.fromstring(right.get("pos", "0 0 0"), sep=" ")
        if np.allclose(lp, rp * np.array([1.0, -1.0, 1.0]), atol=0.0):
            continue
        x, z = (lp[0] + rp[0]) / 2.0, (lp[2] + rp[2]) / 2.0
        y = (abs(lp[1]) + abs(rp[1])) / 2.0
        left.set("pos", f"{x:.9g} {y:.9g} {z:.9g}")
        right.set("pos", f"{x:.9g} {-y:.9g} {z:.9g}")
        n += 1
    return n


def _add_fov_wireframe(cam_body: ET.Element, quat) -> None:
    """Group-4 frustum outline for the PRIMARY camera, drawn in camera_link coordinates."""
    w, h = CAM_RES
    fy = (h / 2.0) / np.tan(np.deg2rad(CAM_FOVY_DEG) / 2.0)
    th, tv = (w / 2.0) / fy, (h / 2.0) / fy          # tan of the half-angles
    qw, qx, qy, qz = quat
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])

    def corner(depth, sh, sv):
        # camera-local: view along -Z, image right +X, image up +Y
        return R @ np.array([sv * tv * depth, sh * th * depth, -depth])

    def cap(name, a, b, size="0.004"):
        ET.SubElement(cam_body, "geom", {
            "name": name, "type": "capsule", "size": size,
            "fromto": " ".join(f"{v:.5f}" for v in (*a, *b)),
            "rgba": "1 0.55 0.1 0.9", "contype": "0", "conaffinity": "0",
            "mass": "0", "group": "4",
        })

    for depth, tag in ((FOV_NEAR_M, "near"), (FOV_FAR_M, "far")):
        c = [corner(depth, s, v) for s, v in ((+1, +1), (+1, -1), (-1, -1), (-1, +1))]
        for i in range(4):
            cap(f"cam_fov_{tag}_{i}", c[i], c[(i + 1) % 4])
    for i, (s, v) in enumerate(((+1, +1), (+1, -1), (-1, -1), (-1, +1))):
        cap(f"cam_fov_strut_{i}", np.zeros(3), corner(FOV_FAR_M, s, v))
    cap("cam_fov_axis", np.zeros(3), R @ np.array([0.0, 0.0, -1.2]), size="0.006")


def _add_rest_contact_excludes(tree: ET.ElementTree) -> int:
    """Declare ``<contact><exclude>`` for body pairs that overlap in the standing pose.

    Refitting the upper body to its visual meshes makes neighbouring links in a kinematic
    chain overlap where they meet — the shoulder ball sits inside the torso shell, the three
    waist boxes nest, the two wrist capsules nest. These are STRUCTURAL: no arm motion can
    separate them, they are present in every pose, and they are not information about
    self-collision. The SOP calls for exactly this ("intended static/adjacent contacts that
    safe-pose collision filtering must ignore by name"), and declaring them here keeps
    ``data.ncon == 0`` a valid safety test for the safe-pose sampler rather than forcing
    every consumer to carry its own whitelist. cuRobo separately reaches the same set through
    ``_rest_overlapping_link_pairs``.

    The pairs are DISCOVERED, not hand-listed: the model is compiled at the standing keyframe
    and whatever touches is excluded, so a change to the fits updates this automatically.
    """
    import tempfile

    import mujoco

    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp) / "probe.xml"
        probe.write_text(ET.tostring(tree.getroot(), encoding="unicode"))
        # meshes resolve via meshdir="assets" relative to the XML -> point it at ours
        model = mujoco.MjSpec.from_string(
            probe.read_text().replace('meshdir="assets"', f'meshdir="{HERE / "assets"}"')
        ).compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    pairs = sorted({
        tuple(sorted((
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                              model.geom_bodyid[data.contact[i].geom1]),
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                              model.geom_bodyid[data.contact[i].geom2]),
        )))
        for i in range(data.ncon)
    })
    if not pairs:
        return 0
    contact = ET.SubElement(tree.getroot(), "contact")
    for first, second in pairs:
        ET.SubElement(contact, "exclude", {"body1": first, "body2": second})
    return len(pairs)


def main() -> None:
    tree = build()
    n_excl = _add_rest_contact_excludes(tree)
    print(f"[import] excluded {n_excl} structural rest-overlap body pair(s)")
    ET.indent(tree, space="  ")
    header = (f"<!-- GENERATED by {pathlib.Path(__file__).name} from "
              f"{SRC_XML.relative_to(pathlib.Path.home())} - do not hand-edit. -->\n")
    OUT_XML.write_text(header + ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    print(f"[import] wrote {OUT_XML}")


if __name__ == "__main__":
    main()
