"""Derive the study-ready PAL TALOS model from the vendor MuJoCo Menagerie drop.

    python asset/pal_talos/talos_study_import.py       # -> asset/pal_talos/talos_study.xml

``talos_study.xml`` is a GENERATED artifact — never hand-edit it; edit this script and
re-run. It sits beside the vendor model and reuses its ``assets/`` meshes directly, so
nothing is duplicated.

WHY NOT USE THE VENDOR MODEL DIRECTLY
-------------------------------------
``talos.xml`` is an excellent display/simulation model but three of its properties make it
unusable as-is for this study:

1. Its collision geometry is 45 triangle MESHES. Both consumers of collision here take
   primitives only — the URDF exporter emits ``<sphere>/<cylinder>/<box>`` and
   ``mj_collision_spheres`` sphere-ises primitives analytically — so meshes cannot reach
   cuRobo at all.
2. ``qpos0`` is not a legal configuration: ``arm_{left,right}_{2,4}_joint`` are all pinned
   at 0, which lies OUTSIDE their own limits ([0.0087, 2.8711] and [-2.2340, -0.0035] and
   their mirrors). Sampling or planning from it is meaningless.
3. There is no keyframe at all (``nkey=0``) and no head camera (``ncam=0``), so no canonical
   pose and no visibility geometry.

WHAT THIS SCRIPT DOES, AND WHY EACH STEP
-----------------------------------------
1. COLLISION REFIT. Every mesh collider is replaced by a capsule INSCRIBED in that link's own
   collision mesh: axis = the mesh AABB's longest edge, radius = the SMALLER of the two
   remaining half-extents, so the capsule can never be wider than the link. Radius is never
   inflated — an over-fat proxy makes the planner refuse legal poses and under-reports the
   reachable workspace, which is the number this study exists to measure. The feet keep
   their vendor boxes (already primitives) and are pruned from the arm model regardless.

   Each side is fitted from ITS OWN declared mesh. The vendor mirrors the right side by
   declaring ``<mesh name="r_arm_Nc" file="arm/arm_N_collision.stl" scale="1 -1 1"/>``, so
   fitting from the compiled geometry reproduces the mirror automatically; the script then
   ASSERTS that every arm/leg left-right pair came out an exact mirror, because getting this
   wrong is silent — a right-side capsule fitted from the left mesh sits on the wrong side of
   the limb and nothing in a screenshot shows it.

2. TOOL SITES. ``end_effector_{L,R}_site`` on ``arm_{left,right}_7_link`` — the last ARM
   link. Everything below it (``gripper_*``) carries a joint and therefore moves with the
   gripper, which the SOP forbids as a planning endpoint. Position (0, 0, -0.11337) in the
   wrist frame is the palm centre: the vendor mounts a fixed, jointless palm housing mesh
   (``gripper_base_link``, both visual and collision variants) directly on ``arm_*_7_link``
   at the top of the gripper kinematic chain -- unlike every ``gripper_*`` body below it,
   its own position never moves with the gripper's open/close joints, so it is a legal
   planning endpoint. Its AABB in the wrist frame spans z = -0.1474 to -0.0793 (mid-depth
   -0.1134) with x/y centres of -0.0035/-0.0039 -- an asymmetric-mesh artifact an order of
   magnitude smaller than the housing's own 100+ mm extent, so x and y are pinned to 0 to
   keep the point on the wrist axis, same as the rejected finger-tip centroid below.
   (An earlier version of this site sat at the mid-depth of the finger PADS instead,
   z = -0.21365, spanning -0.2441 to -0.1832 -- effectively fingertip depth, not the palm;
   moved because the study's grasp endpoint should be where the hand's structure is, not
   where the fingers happen to close.) The point is on the wrist axis, so it is
   mirror-invariant.

3. STANDING KEYFRAME. Every joint clamped into its own limits, then both shoulders abducted
   15 deg (``arm_2``) with the elbows at -0.2 deg. Clamping alone leaves the hands buried in
   the thighs (8 body pairs, up to -64 mm); 15 deg clears everything with margin (10 deg
   already suffices) and puts the palms 92-97 deg off the camera axis, far outside its field
   of view, so they cannot occlude the forward workspace. Root height is solved so the lowest
   collision point sits exactly on z = 0.

4. HEAD CAMERA. Mount pose and optical convention are the vendor's own: PAL's
   ``talos_description/urdf/head/head.urdf.xacro`` puts ``rgbd_link`` at
   (0.0621, 0.0375, 0.1832) on ``head_2_link``, and ``xyaxes="0 -1 0 0 0 1"`` makes the
   MuJoCo camera's view axis (-Z) point along world +X with image-up along +Z, which is that
   xacro's ``rpy=(-90, 0, -90)`` optical frame expressed in MuJoCo. See CAMERA CONTRACT.

CAMERA CONTRACT (SOP section 3)
--------------------------------
The head camera is an **Orbbec Astra Pro**, established by two independent primary sources:
PAL's own robot description instantiates ``<xacro:orbbec_astra_pro name="rgbd" parent="head_2">``
for the default head, and the TALOS platform paper (Stasse et al., Humanoids 2017) states the
first unit "is equipped with an ORBBEC Astra Pro RGB-D camera". Orbbec's own product page
publishes, for the Astra series: **RGB FOV H63.1 deg x V49.4 deg**, depth FOV H58.4 x V45.5,
depth range 0.6-8 m.

``fovy = 49.4`` with resolution 640x480 reproduces that RGB envelope exactly: the study's FOV
gate builds an isotropic pinhole (fy from fovy and image height, fx = fy), giving a derived
horizontal 63.04 deg against Orbbec's published 63.1. No fictitious resolution is needed
because Orbbec's own pair is 4:3-consistent (tan(63.1/2)/tan(49.4/2) = 1.335).

Only the RGB stream is modelled (``head_cam``): the study scores every other robot on its
primary RGB envelope, and TALOS follows the same convention. An earlier revision also
declared a second camera for the depth envelope (``fovy = 45.5`` -> 58.4 deg horizontal,
``head_cam_depth``) as a documented sensitivity twin; dropped because the study's reachable
and visible-reachable results should use one consistent camera definition across all robots.

⚠ SUPERSEDED VALUE. An earlier configuration on this branch declared ``fovy=58.0`` with
``fovx=72.935``. That pair is internally consistent but matches no Orbbec product: fx and fy
agree to 7 ppm and 72.935 is exactly 2*atan((4/3)*tan(29 deg)), i.e. the horizontal was
DERIVED from the vertical by the renderer, leaving 58.0 as the only free parameter — and no
Orbbec model has a 58 deg vertical RGB field. The likeliest origin is PAL's
``realsense_d435.gazebo.xacro``, whose ``vertical_fov`` is 58.0 deg and which belongs to a
RealSense D435 DEPTH stream measured at 16:9. Its solid angle is 33% larger than the real
Astra RGB cone (1.169 sr vs 0.882 sr; 65% more image area at 1 m), so it would have
systematically over-stated the visible workspace.
"""
from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC_XML = HERE / "talos.xml"
OUT_XML = HERE / "talos_study.xml"

# --- measured constants (derivations in the module docstring) ---------------------------
SHOULDER_ABDUCTION_RAD = np.deg2rad(15.0)   # arm_2, +left / -right
# arm_4 (elbow) is parked at its own nearest-to-straight LIMIT rather than a hard-coded
# angle: the limit is -0.0035 rad (-0.2005 deg), so writing "-0.2 deg" is 0.0005 deg outside
# it and MuJoCo reports the keyframe as violating the joint's range.
ELBOW_AT_LIMIT = "upper"
TOOL_SITE_WRIST = (0.0, 0.0, -0.11337)      # palm centre in arm_*_7_link

# Orbbec Astra Pro. RGB stream only -- see CAMERA CONTRACT for why.
CAM_MOUNT_POS = (0.0621, 0.0375, 0.1832)    # PAL talos_description rgbd_link on head_2_link
CAM_MOUNT_XYAXES = "0 -1 0 0 0 1"           # view axis (-Z) -> world +X, image up -> +Z
CAM_RESOLUTION = (640, 480)
CAM_FOVY_RGB = 49.4                         # -> 63.04 deg horizontal (Orbbec publishes 63.1)
FOV_NEAR_M, FOV_FAR_M = 0.10, 0.60          # group-4 wireframe extent (diagnostic, matches GR-3)

# Links whose vendor collider is already a primitive; left untouched.
KEEP_PRIMITIVE_BODIES = ("leg_left_6_link", "leg_right_6_link")


def _compiled_source() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(SRC_XML))
    return model, mujoco.MjData(model)


def _collider_vertices(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """One mesh collider's vertices in its BODY frame.

    Uses the COMPILED vertices, so the vendor's ``scale="1 -1 1"`` on every right-side mesh is
    already baked in and the right limb is fitted from genuinely right-side geometry.
    """
    mesh_id = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
    return model.mesh_vert[start:start + count] @ rot.reshape(3, 3).T + model.geom_pos[geom_id]


def _inscribed_capsule_pca(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(p0, p1, radius) of a capsule inscribed in a mesh, along the mesh's PRINCIPAL axis.

    An axis-ALIGNED fit is badly wrong for this robot: TALOS's arm links run at an angle to
    their own body frames, so the axis-aligned box is far larger than the link and the
    inscribed capsule inside it collapses to a stub. Measured per-collider self-coverage over
    all 45 colliders is 14.1% axis-aligned versus 53.2% principal-axis (``arm_left_1``: 12.0%
    -> 94.1%), for the same inscribed radius rule.

    The radius stays INSCRIBED: it is the smaller of the two perpendicular principal extents,
    so the capsule is never wider than the link in its narrow direction. Nothing is inflated;
    an over-fat proxy would make the planner reject legal poses and under-report the reachable
    workspace, which is exactly what this study measures.

    Mirror-safety: the principal axis' SIGN is arbitrary in an SVD, but the capsule is stored
    as an unordered endpoint pair and the radius comes from absolute extents, so a mirrored
    point cloud yields the mirrored capsule regardless of the sign the solver happens to pick.
    """
    centre = vertices.mean(axis=0)
    centred = vertices - centre
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    along = centred @ axes[0]
    radius = float(min(np.abs(centred @ axes[1]).max(), np.abs(centred @ axes[2]).max()))
    reach = max(float(np.abs(along).max()) - radius, 1e-4)
    return centre - axes[0] * reach, centre + axes[0] * reach, radius


def _mesh_collider_aabb(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """(centre, half-extents) of one mesh collider's vertices, in its BODY frame."""
    mesh_id = int(model.geom_dataid[geom_id])
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    verts = model.mesh_vert[start:start + count]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
    verts = verts @ rot.reshape(3, 3).T + model.geom_pos[geom_id]
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def _inscribed_capsule(centre: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(p0, p1, radius) of the capsule inscribed in an AABB, along its longest axis."""
    axis = int(np.argmax(half))
    radius = float(min(half[i] for i in range(3) if i != axis))
    reach = max(float(half[axis]) - radius, 1e-4)
    offset = np.zeros(3)
    offset[axis] = reach
    return centre - offset, centre + offset, radius


def build() -> ET.ElementTree:
    model, data = _compiled_source()
    tree = ET.parse(SRC_XML)
    root = tree.getroot()
    root.set("model", "talos_study")

    bodies = {b.get("name"): b for b in root.iter("body")}
    fits: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    # 1. collision refit -----------------------------------------------------------------
    n_fit = 0
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name in KEEP_PRIMITIVE_BODIES or name not in bodies:
            continue
        mesh_colliders = [
            g for g in range(model.body_geomadr[body_id],
                             model.body_geomadr[body_id] + model.body_geomnum[body_id])
            if model.geom_group[g] == 3 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        if not mesh_colliders:
            continue
        body = bodies[name]
        for geom in [g for g in body.findall("geom")
                     if g.get("class") == "collision" and g.get("mesh")]:
            body.remove(geom)
        # One capsule PER mesh collider rather than one per body: `arm_*_7_link` carries both
        # the wrist shell and the gripper base, whose union AABB would be a loose blob, and
        # the gripper base is mounted non-mirrored between the two hands, so a per-body union
        # would also destroy the arm's left/right symmetry.
        for k, g in enumerate(mesh_colliders):
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                          int(model.geom_dataid[g]))
            fit = _inscribed_capsule_pca(_collider_vertices(model, g))
            fits[f"{name}::{mesh_name}"] = fit
            p0, p1, radius = fit
            ET.SubElement(body, "geom", {
                "name": f"{name}_col{k}", "class": "collision", "type": "capsule",
                "size": f"{radius:.6g}",
                "fromto": " ".join(f"{v:.6g}" for v in (*p0, *p1)),
            })
            n_fit += 1

    _assert_mirrored_fits(fits)

    # 2. tool sites on the last ARM link --------------------------------------------------
    x, y, z = TOOL_SITE_WRIST
    for side, tag in (("left", "L"), ("right", "R")):
        ET.SubElement(bodies[f"arm_{side}_7_link"], "site", {
            "name": f"end_effector_{tag}_site",
            "pos": f"{x:.6g} {y:.6g} {z:.6g}",
            "size": "0.008", "rgba": "1 0.2 0.2 0.6" if tag == "R" else "0.2 0.4 1 0.6",
        })

    # 3. head cameras + their raycast-origin sites ----------------------------------------
    head = bodies["head_2_link"]
    xyaxes_vals = [float(v) for v in CAM_MOUNT_XYAXES.split()]
    x_axis, y_axis = xyaxes_vals[:3], xyaxes_vals[3:]
    view_dir = -np.cross(x_axis, y_axis)
    # `head_cam` stays at PAL's own rgbd_link origin (CAM_MOUNT_POS) -- this is the pose
    # `talos_visibility.py` reads (`data.cam_xpos`, `{camera_name}_site`) to SCORE
    # reachability/visibility, so it must stay vendor-accurate, not a display convenience.
    #
    # The group-4 wireframe is display-only (`--view`, never read by the scorer), so it is drawn
    # from a separate, pushed-out `lens_pos` instead. Reason: CAM_MOUNT_POS is the Orbbec housing
    # MESH's own origin, not its lens -- assets/sensors/orbbec/orbbec.stl extends only 5.9 mm
    # forward of it, so a wireframe apex placed there sits INSIDE the housing (every ray across
    # the full FOV cone self-hits the housing mesh at ~2 mm from CAM_MOUNT_POS). That is invisible
    # at whole-body `--view` zoom and reads as "inside the head". x=0.1175 m (the head's GLOBAL
    # front-most vertex, from unrelated geometry 80 mm away) overshot -- looked like a horn
    # floating off the face. x=0.075 m clears the LOCAL visor rim (head_2_default.stl vertices
    # within 20 mm of the mount reach x=0.087 m; verified zero self-hits over the full FOV cone at
    # this position) and reads as a sensor sitting flush in its slit -- picked by eye against a
    # render after the overshoot.
    CAM_LENS_FORWARD_M = 0.075 - CAM_MOUNT_POS[0]
    lens_pos = tuple(np.asarray(CAM_MOUNT_POS) + CAM_LENS_FORWARD_M * np.asarray(view_dir))
    common = {"pos": " ".join(f"{v:.6g}" for v in CAM_MOUNT_POS), "xyaxes": CAM_MOUNT_XYAXES}
    ET.SubElement(head, "camera", {
        "name": "head_cam", **common, "fovy": f"{CAM_FOVY_RGB:g}",
        "resolution": f"{CAM_RESOLUTION[0]} {CAM_RESOLUTION[1]}",
    })
    ET.SubElement(head, "site", {
        "name": "head_cam_site", **common, "size": "0.006", "rgba": "0 0 1 0.6",
    })
    _add_fov_wireframe(head, lens_pos, x_axis, y_axis)

    # 4. legal, contact-free standing keyframe -------------------------------------------
    qpos, height = _standing_qpos(model, data)
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", {
        "name": "standing", "qpos": " ".join(f"{v:.9g}" for v in qpos),
    })

    # NOTE ON qpos0. The `standing` keyframe -- not `qpos0` -- is this model's canonical home,
    # and every consumer must load it. `qpos0` stays at the vendor's all-zeros, which for TALOS
    # is not a legal configuration at all: `arm_*_2` and `arm_*_4` have limit intervals that
    # EXCLUDE zero, so the vendor's own nominal pose is mechanically unreachable. Shifting each
    # hinge's `ref` onto the standing value is not a fix -- MuJoCo applies `qpos - ref` as the
    # joint's spatial transform, so that renumbers qpos0 without moving a single link. The real
    # consumer at risk is the URDF export (cuRobo's IK seed), which therefore reads the keyframe.

    print(f"[import] refitted {n_fit} mesh colliders to inscribed capsules")
    print(f"[import] standing height {height:.7f} m")
    return tree


def _assert_mirrored_fits(fits: dict[str, tuple[np.ndarray, np.ndarray, float]]) -> None:
    """Fail loudly if any arm/leg left-right capsule pair is not an exact y-mirror.

    A right-side capsule accidentally fitted from left-side geometry sits on the wrong side of
    the limb and is invisible in any render, so this is checked rather than trusted.
    """
    checked = 0
    for name, (p0, p1, radius) in fits.items():
        body_name, _, mesh_name = name.partition("::")
        if "_left_" not in body_name or not body_name.startswith(("arm_", "leg_")):
            continue
        # The gripper base rides on arm_7 with a non-mirrored mount (the three-finger hand is
        # rotated, not reflected, between sides) -- a genuine hardware asymmetry, not a fit bug.
        if not mesh_name.startswith(("l_", "r_")):
            continue
        mirror = fits.get(f"{body_name.replace('_left_', '_right_')}::{mesh_name.replace('l_', 'r_', 1)}")
        assert mirror is not None, f"{name} has no right-side counterpart"
        q0, q1, r2 = mirror
        flip = np.array([1.0, -1.0, 1.0])
        assert abs(radius - r2) < 1e-9, f"{name}: radius {radius} vs mirror {r2}"
        # A capsule's endpoints are unordered, and mirroring one whose axis is y swaps them,
        # so compare the pair as a set rather than element-wise.
        mirrored = sorted((tuple(p0 * flip), tuple(p1 * flip)))
        assert np.allclose(mirrored, sorted((tuple(q0), tuple(q1))), atol=1e-9), (
            f"{name}: capsule is not an exact mirror of its right-side pair"
        )
        checked += 1
    print(f"[import] verified {checked} arm/leg capsule pairs are exact y-mirrors")


def _add_fov_wireframe(body: ET.Element, pos, x_axis, y_axis) -> None:
    """Group-4 frustum outline for the PRIMARY (RGB) camera, drawn in `body`'s local frame.

    `x_axis`/`y_axis` are the camera's own right/up axes -- the same vectors given as MuJoCo's
    `xyaxes` -- so, unlike GR-3's quat-derived version, no local-axis convention needs
    rederiving: the view direction is `-cross(x_axis, y_axis)` per MuJoCo's documented camera
    frame (right=+X, up=+Y, view=-Z), applied directly.
    """
    w, h = CAM_RESOLUTION
    fy = (h / 2.0) / np.tan(np.deg2rad(CAM_FOVY_RGB) / 2.0)
    th, tv = (w / 2.0) / fy, (h / 2.0) / fy          # tan of the half-angles
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    z_axis = np.cross(x_axis, y_axis)                # camera -Z is the view direction
    p = np.asarray(pos, dtype=np.float64)

    def corner(depth, sh, sv):
        return p + sh * th * depth * x_axis + sv * tv * depth * y_axis - depth * z_axis

    def cap(name, a, b, size="0.004"):
        ET.SubElement(body, "geom", {
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
        cap(f"cam_fov_strut_{i}", p, corner(FOV_FAR_M, s, v))
    cap("cam_fov_axis", p, p - 1.2 * z_axis, size="0.006")


def _standing_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    """Clamp qpos0 into its limits, abduct the shoulders, and drop the feet onto z=0."""
    qpos = model.qpos0.copy()
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            adr = int(model.jnt_qposadr[j])
            qpos[adr] = np.clip(qpos[adr], *model.jnt_range[j])
    for name, value in (("arm_left_2_joint", +SHOULDER_ABDUCTION_RAD),
                        ("arm_right_2_joint", -SHOULDER_ABDUCTION_RAD),
                        ("arm_left_4_joint", None),
                        ("arm_right_4_joint", None)):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if value is None:                       # park the elbow at its own straightest limit
            value = float(model.jnt_range[joint_id][1])
        qpos[model.jnt_qposadr[joint_id]] = value
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, f"standing pose still has {data.ncon} contacts before the refit"
    qpos[2] -= _lowest_collision_z(model, data)
    return qpos, float(qpos[2])


def _lowest_collision_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Exact lowest point of any collision geom in the current configuration."""
    geom = mujoco.mjtGeom
    lowest = np.inf
    for g in range(model.ngeom):
        if model.geom_group[g] != 3:
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        centre, size, kind = data.geom_xpos[g], model.geom_size[g], model.geom_type[g]
        if kind == geom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            z = (model.mesh_vert[start:start + count] @ rot.T + centre)[:, 2].min()
        elif kind == geom.mjGEOM_BOX:
            z = centre[2] - float(np.abs(rot[2]) @ size[:3])
        elif kind == geom.mjGEOM_CYLINDER:
            z = centre[2] - (abs(rot[2, 2]) * size[1] + float(np.hypot(*rot[2, :2])) * size[0])
        elif kind == geom.mjGEOM_CAPSULE:
            z = centre[2] - (abs(rot[2, 2]) * size[1] + size[0])
        elif kind == geom.mjGEOM_SPHERE:
            z = centre[2] - size[0]
        else:
            continue
        lowest = min(lowest, z)
    return float(lowest)


def _add_rest_contact_excludes(tree: ET.ElementTree) -> int:
    """Declare ``<contact><exclude>`` for body pairs that touch in the standing pose.

    Inscribed capsules on neighbouring links inevitably meet where the links meet: here the
    upper arm and forearm graze across the elbow by 4.1 mm, and the two ankles graze by
    2.0 mm at the stance width. Both are structural -- present in every pose, carrying no
    information about self-collision -- and the SOP provides for ignoring exactly this class
    by name, which also keeps ``data.ncon == 0`` a valid test for the safe-pose sampler.

    Excluding the elbow pair costs NOTHING here, and that was verified rather than assumed:
    driving ``arm_left_4_joint`` to its full -128 deg flexion limit against the VENDOR's own
    triangle-mesh colliders (the fidelity ground truth) produces zero contacts at every step,
    i.e. TALOS's forearm genuinely never reaches its upper arm.

    Pairs are DISCOVERED by compiling at the standing keyframe, so a change to the fits
    updates this automatically instead of leaving a stale hand-written list.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        probe = pathlib.Path(tmp) / "probe.xml"
        probe.write_text(ET.tostring(tree.getroot(), encoding="unicode"))
        spec = mujoco.MjSpec.from_string(
            probe.read_text().replace('meshdir="assets"', f'meshdir="{HERE / "assets"}"')
        )
        model = spec.compile()
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
    contact = tree.getroot().find("contact")
    if contact is None:
        contact = ET.SubElement(tree.getroot(), "contact")
    for first, second in pairs:
        ET.SubElement(contact, "exclude", {"body1": first, "body2": second})
    return len(pairs)


def main() -> None:
    tree = build()
    n_excl = _add_rest_contact_excludes(tree)
    print(f"[import] excluded {n_excl} structural rest-overlap body pair(s)")
    ET.indent(tree, space="  ")
    header = (f"<!-- GENERATED by {pathlib.Path(__file__).name} from {SRC_XML.name} "
              "- do not hand-edit. -->\n")
    OUT_XML.write_text(header + ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    print(f"[import] wrote {OUT_XML}")


if __name__ == "__main__":
    main()
