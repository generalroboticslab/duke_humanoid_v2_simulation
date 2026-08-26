"""Head-camera RGB-FOV frustum overlay — shared by every robot that carries a camera.

Was duplicated verbatim in humanoid_v21_constants.py and g1_constants.py (the two copies
differed only in comments, one local variable name, and g1's color lookup raising KeyError
on an unlisted module tag). One copy here; both import it. g1_constants already reaches
into humanoid_v21_constants for CAMERA_DIR / CAMERA_MOTORS / _apply_joint_properties, so
the split bought no isolation.

A wireframe frustum (thin capsule edges) rigidly fixed in each camera's parent body, so it
tracks a gimbal automatically — no runtime viewer hook. D436 RGB FOV 90 deg H x 65 deg V.
Edges, not a solid mesh: a translucent stage-light volume was tried and reverted per review
(too visually distracting), so the outline is visible without obscuring the scene.

The wireframe is paired with a baked ``mjLIGHT_SPOT`` per ``*_rgb`` site
(``add_fov_spotlights``), added to shade whatever surfaces fall inside the cone so the FOV
would read through lighting boundaries and not only outlines. It is baked but INACTIVE as of
2026-08-04 (``FOV_SPOT_ACTIVE``): the cone is unusable at lens width and the dot's edge is
unfixably faceted, so the wireframe is the sole FOV channel again. See ``FOV_SPOT_ACTIVE``
and ``FOV_SPOT_CUTOFF_DEG`` for the measurements behind that.

The envelope MUST stay identical to ``tasks.camera_terms.fov_detect``'s range gate — the
overlay is a sensor audit aid and must never claim a range the detector does not have. The
literals are kept local on purpose: asset construction must not import task code.

Note: MuJoCo (3.11) has no per-geom or per-material shadow toggle — ``castshadow`` exists
only on lights — so overlay geoms DO cast shadows when their group is shown. Transparent
geoms and runtime decor geoms cast too (measured). ``add_fov_frustum_hull`` therefore clears
``castshadow`` on the spec's lights, which is the only lever available; see its docstring for
the measurements and for the one case it cannot reach (lights added to the spec afterwards).
"""

import math

import mujoco
import numpy as np

FOV_H_HALF_RAD: float = 0.7854
FOV_V_HALF_RAD: float = 0.5672
FOV_NEAR_M: float = 0.20
FOV_FAR_M: float = 5.0
FOV_EDGE_RADIUS: float = 0.002
# Optical-axis ray reaches the detector far plane too (drawn out to `far`). Yellow and
# slightly thicker than the edges: it is the image-center sight line, the one line that says
# WHERE the camera is aimed, so it stays legible against the per-side edge color rather than
# blending into it. 1.5x, not the original 3x -- that read as an opaque pole running through
# the gripper and cubes in chase-cam video.
FOV_AXIS_RADIUS: float = FOV_EDGE_RADIUS * 1.5
FOV_AXIS_RGBA: tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.10)  # legacy; axis pyramid now uses module color
# Center-cone pyramid: scaled-down FOV at ``FOV_AXIS_CONE_RATIO`` (default 0.1 = 1/10th).
# 0.1 * 90 deg H = 9 deg H, 0.1 * 65 deg V = 6.5 deg V -- tracks the camera's lens instead of
# being a fixed cone that drifts from the wireframe / detector envelope when a caller changes
# the lens. Replaces the thin yellow axis capsule as a soft, low-alpha pyramid so the aim
# direction reads at a glance instead of being mistaken for a thin edge.
FOV_AXIS_CONE_RATIO: float = 0.1
FOV_AXIS_ALPHA: float = 0.10
# Dedicated geom group so the FOV toggles independently of the robot meshes
# (group 2). Groups 3+ are hidden by default, so the FOV starts off; the
# native MuJoCo viewer toggles it with the matching digit key (press 4).
# NOTE: gaze_search_eval's LOS raycast masks OUT this group — the FOV overlay
# must never occlude detection rays.
FOV_GEOM_GROUP: int = 4
# When True, ``add_fov_frustums`` (the 13 capsule edges per ``*_rgb`` site) is called next to
# ``add_fov_frustum_hull`` in every asset's ``get_spec``. When False, only the hull is baked
# and the wireframe is dropped entirely. Static probes / orbit mp4s that want the envelope
# outline keep this True; chase-cam playback where the edges blur against the world flips it
# off. The wireframe can't be hidden at runtime: MuJoCo's only edge-visibility lever is
# ``MjvOption.geomgroup`` and group 4 is also the hull's, so a runtime toggle would either
# hide both or neither. Compile-time is the only correct split.
FOV_WIREFRAME: bool = False
# When True, bake the outer + stereo shells (the full cone at far and FOV_HULL_STEREO_M).
# When False, only the tip (apex->near, mixed-color) and the axis pyramid ship. Use False for chase-cam
# playback where the large fill hides the gripper / cubes.
# Stays True as the DEFAULT: the live viewer, the orbit mp4s and ``probe/fov_hull_check.py`` all want the
# full cone, and it is the honest depiction of the envelope. The paper keyframe figure is the one consumer
# that cannot afford it -- at its 1.35 m chase-cam distance the outer shells cover most of the frame, which
# is invisible at 2560x1440 but washes the panel out at 1.2 in on the page -- so ``curobo_reach_verify.py``
# patches this to False for ``--record-keyframes`` only. Flipping the default here instead was tried on
# 2026-08-06 and reverted: it silently stripped the cone from every viewer session too.
FOV_LARGE_VIS: bool = True
# Tip (apex->near, mixed-color) is independent of FOV_LARGE_VIS: the large shells can be off while the
# tip stays on, marking the lens when the cone fill would occlude the scene.
FOV_TIP_VIS: bool = True
# Spotlight parameters. The spot is an AIM-POINT marker, not a depiction of the FOV — the
# wireframe is the only honest depiction of the envelope. Brightness comes from `diffuse`,
# NOT `intensity`: light_intensity is read only in physical-light-units mode (default 0.0),
# so the classic renderer every surface here uses ignores it.
FOV_SPOT_DIFFUSE: tuple[float, float, float] = (0.60, 0.60, 0.48)
# MuJoCo `cutoff` is DEGREES, and this is deliberately FAR narrower than the lens
# (H/2=45, V/2=32.5). A cone matching the real FOV is useless: the optical axis sits ~47.6
# deg below horizontal on the welded rig, so any cutoff >= 47.6 opens the cone ABOVE
# horizontal and floods the ground plane to infinity, and even 32.5 washes several metres
# at grazing incidence. A few degrees puts a tight dot where the camera is pointed.
# Raise `FOV_SPOT_DIFFUSE` rather than this if the dot is hard to see.
FOV_SPOT_CUTOFF_DEG: float = 3.0
# OpenGL GL_SPOT_EXPONENT: brightness falls off as cos(theta)^exponent about the axis, so
# this narrows the beam INSIDE `cutoff` rather than sharpening the cutoff edge. The
# original 20.0 killed the beam outright (measured 0 lit pixels at every brightness) and
# was one of the three bugs behind "the spotlight does not work". Keep at 1.0: with a
# cutoff of a few degrees the cone is already tight, and any extra exponent only dims it.
FOV_SPOT_EXPONENT: float = 1.0
FOV_SPOT_ATTENUATION: tuple[float, float, float] = (1.0, 0.0, 0.0)
# (No FOV_SPOT_GROUP — MjsLight has no `group` attr; lights are always active.)
# The spot renders with a visibly faceted edge because MuJoCo's classic renderer shades
# PER-VERTEX (Gouraud): the beam boundary is the floor's triangulation, not the cone. Three
# smoothing fixes were measured and all rejected (2026-08-04):
#   * mjLIGHT_IMAGE — an environment/area light, not a projected spot. Lights the whole
#     frame (303k px against the same scene's 18.5k spot).
#   * shrinking the ground plane's `size[2]` render-grid spacing — a NO-OP on an INFINITE
#     plane (identical lit-pixel count at 1.0 / 0.25 / 0.05 / 0.01), which is what every
#     standalone viewer uses.
#   * swapping those viewers to a finite tessellated floor — does refine it (an infinite
#     plane over-lights the footprint ~29%), but only shrinks the spot toward its true
#     size; the edge stays as ragged. Not worth editing 8 viewers for.
# No per-pixel lighting switch exists (`offsamples` MSAA and `shadowsize` do not apply).
# Hence the spot stays an aim-point marker and the WIREFRAME carries the FOV envelope.
# DISABLED by default (2026-08-04): with the cone unusable at lens width and the dot's edge
# unfixably faceted, the spot earns less than the shading confusion it adds. The lights are
# still BAKED (so `nlight` and every light index are unchanged whichever way this flag sits
# — flipping it can never shift a light id out from under a caller); only `active` differs.
# Flip to True to get the aim-point dot back. Per-model override after compile:
# `model.light_active[i] = 1` for the `*_fov_spot` ids.
FOV_SPOT_ACTIVE: bool = False
# Translucent solid frustum, an OPT-IN companion to the wireframe (`add_fov_frustum_hull`).
# PER-FACE alpha, and the hull is double-sided, so a look-through hits it twice: the effective
# tint is ~2x this. 0.12 was the single-sided pick (measured against 0.06 / 0.10 / 0.18 / 0.20 on
# the real rig: 0.06 vanished into the lit floor near the apex, 0.20 started tinting objects seen
# through it). 0.01: per-face value; doubled-winding makes near ~0.02 and nested stereo makes
# the confident zone ~0.04. Fill is a hint, edges (FOV_EDGE_RADIUS=2 mm capsule wireframe) carry
# the read. Axis pyramid reads at 0.10 (its own constant) so the optical-center sight line is
# the strongest cue in the scene.
FOV_HULL_ALPHA: float = 0.01
# Shared name suffix so the hull geoms are findable by name after compile.
FOV_HULL_SUFFIX: str = "_fov_hull"
# Truncated frustum triangles over verts [near corner 0..3, far corner 0..3]:
# 4 side quads (split into tris), 1 near cap, 1 far cap. Caps are 2 triangles each (0-2-1 +
# 0-3-2 winds consistent with the corner order), sides match (i,j,j+1) and (i,j+1,i+1).
_FRUSTUM_FACES = (
    (0, 4, 5), (0, 5, 1),    # wall 0 -> 4
    (1, 5, 6), (1, 6, 2),    # wall 1 -> 5
    (2, 6, 7), (2, 7, 3),    # wall 2 -> 6
    (3, 7, 4), (3, 4, 0),    # wall 3 -> 7
    (0, 1, 2), (0, 2, 3),    # near cap
    (4, 6, 5), (4, 7, 6),    # far cap
)
# Inner shell depth. tasks/camera_terms.py:45 records that ~3 m is where stereo depth is optimal
# and 5 m only the usable-detection edge -- a single uniform slab throws that distinction away.
# Nesting a second hull here makes the confident zone twice as dense for free and puts a visible
# step at the boundary, so tint reads as CONFIDENCE, not just extent.
FOV_HULL_STEREO_M: float = 3.0
# Per-module frustum colors, matched against the site name's module tag. A rig whose
# tags are not listed (e.g. humanoid's un-tagged single rig, or a ring rig's cam0..camN)
# falls back to _FOV_RGBA_DEFAULT — colors only disambiguate modules visually, they gate
# nothing.
_FOV_SIDE_RGBA: dict[str, tuple[float, float, float, float]] = {
    "head":   (0.2, 0.7, 1.0, 0.10),  # cyan/blue (g1's torso-mounted D435)
    "left":   (0.2, 0.7, 1.0, 0.10),  # cyan
    "right":  (1.0, 0.55, 0.1, 0.10), # orange
    "center": (0.4, 1.0, 0.4, 0.10),  # green
}
_FOV_RGBA_DEFAULT: tuple[float, float, float, float] = _FOV_SIDE_RGBA["right"]
# Color for the apex->near band: stereo depth is unreliable there (any return that lands inside
# the near plane is below the detector's range gate). The tip echoes the module's own color
# mixed 50% toward black, so cyan stays cyan, orange stays orange — visually same family,
# darker shade flags "not measurement space". Alpha matches the main shell.
FOV_TIP_MIX: float = 0.5
FOV_TIP_ALPHA: float = FOV_HULL_ALPHA


def frustum_corner_dirs(h_half: float, v_half: float) -> list[np.ndarray]:
    """UNIT camera-frame directions to the 4 frustum corners (x right, y down, z out).

    ``dir ∝ (tan h_half, tan v_half, 1)`` reproduces fov_detect's corner angles
    (``h=atan2(x, z)``, ``v=atan2(y, z)``). NORMALIZED on purpose: fov_detect gates on
    ``rng = ||p_cam||`` (Euclidean lens distance, camera_perception.py), NOT on depth z,
    so a corner must be placed at ``unit_dir * range`` to land on the real gate. Scaling
    a non-unit ``(tan h, tan v, 1)`` by the range instead over-draws the corners by
    ``||(tan h, tan v, 1)|| ≈ 1.6x`` at this FOV — the bug this helper exists to prevent.

    Shared with the runtime decor drawer in ``_viz_camera_fov_utils._draw_frustum`` so the
    baked overlay and the viz-script overlay cannot drift apart.
    """
    tan_h, tan_v = math.tan(h_half), math.tan(v_half)
    dirs = []
    for s_x, s_y in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        d = np.array([s_x * tan_h, s_y * tan_v, 1.0])
        dirs.append(d / np.linalg.norm(d))
    return dirs


def add_fov_frustums(
    spec: mujoco.MjSpec,
    h_half: float = FOV_H_HALF_RAD,
    v_half: float = FOV_V_HALF_RAD,
    near: float = FOV_NEAR_M,
    far: float = FOV_FAR_M,
    group: int = FOV_GEOM_GROUP,
) -> None:
    """Add a wireframe RGB-FOV frustum anchored at each ``*_rgb`` site (RGB optical centre).

    The ``*_rgb`` site is the SAME site fov_detect / the gaze servo read
    (``tasks/camera_terms._CAM_SITES``), so the drawn cone is exactly the computed one
    (an earlier anchor on ``*_cam_front_center``, the glass centre, was ~3.3 cm off the
    real apex).

    Frustum opens along site +Z (standard camera convention: z forward/out, x right,
    y down). Rect corners sit at EUCLIDEAN lens distance ``near``/``far`` along the corner
    directions — the same range gate fov_detect applies — so the drawn extent matches what
    is detectable. Corner endpoints computed in site frame then transformed to the parent
    body frame via the site's pos/quat. Geoms added to the parent body so the frustum
    tracks the gimbal automatically. Visual only (contype=conaffinity=mass=0); toggled with
    the ``group`` digit key.

    The lens/group arguments default to the D436 rig every current caller uses; they exist
    because a second lens is already in the repo (reachability_study's Fourier GR3 runs a
    D455 and reads its FOV from the model's own ``cam_fovy``). Pass them per robot rather
    than adding a second copy of this function.
    """
    # Wireframe is compile-time-gated by ``FOV_WIREFRAME``. Runtime hiding is impossible because
    # the wireframe and the hull share ``FOV_GEOM_GROUP`` (group 4 is the only free slot in the
    # ``MjvOption.geomgroup`` array that's NOT already owned by fit/collision/drop/ground), and
    # there is no per-geom visibility lever on either side. Flip the flag to drop the 13
    # edge capsules (and their two meshes) from the model entirely; the hull stays.
    if not FOV_WIREFRAME:
        return

    corner_dirs = frustum_corner_dirs(h_half, v_half)

    for site in spec.sites:
        if not site.name.endswith("_rgb"):
            continue
        rgba = next((c for tag, c in _FOV_SIDE_RGBA.items() if tag in site.name),
                    _FOV_RGBA_DEFAULT)

        # Rotation v_body = R @ v_cam (site/camera frame → parent body frame). Taken from
        # MuJoCo's own quat convention rather than hand-expanded, so it cannot silently
        # disagree with how the compiler interprets the same site.quat.
        p = np.asarray(site.pos, dtype=float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(site.quat, dtype=float))
        R = R.reshape(3, 3)

        def to_body(v_cam) -> np.ndarray:
            return p + R @ np.asarray(v_cam, dtype=float)

        body = site.parent

        def add_edge(name: str, a, b, radius: float, color) -> None:
            geom = body.add_geom()
            geom.name = name
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.fromto = [*a, *b]
            geom.size = [radius, 0.0, 0.0]
            geom.rgba = list(color)
            geom.contype = 0
            geom.conaffinity = 0
            geom.mass = 0.0
            geom.group = group

        near_c = [to_body(u * near) for u in corner_dirs]
        far_c  = [to_body(u * far)  for u in corner_dirs]
        lens = to_body((0.0, 0.0, 0.0))

        edges  = [(near_c[i], near_c[(i + 1) % 4]) for i in range(4)]  # near rect
        edges += [(far_c[i],  far_c[(i + 1) % 4])  for i in range(4)]  # far rect
        edges += [(lens, far_c[i]) for i in range(4)]                   # rays from lens
        for i, (a, b) in enumerate(edges):
            add_edge(f"{site.name}_fov{i}", a, b, FOV_EDGE_RADIUS, rgba)
        # Optical-axis ray is now drawn as a 5 deg pyramid by add_fov_frustum_hull — same
        # toggle group, no thin line to overdraw the cone.


def add_fov_spotlights(
    spec: mujoco.MjSpec,
    diffuse: tuple[float, float, float] = FOV_SPOT_DIFFUSE,
    cutoff_deg: float = FOV_SPOT_CUTOFF_DEG,
    exponent: float = FOV_SPOT_EXPONENT,
    active: bool = FOV_SPOT_ACTIVE,
) -> None:
    """Add a baked ``mjLIGHT_SPOT`` per ``*_rgb`` site, anchored on its parent body.

    Pairs with ``add_fov_frustums`` so the FOV reads through shading too — anything inside
    the cone is brighter than outside, giving a visible footprint without an outline.
    Anchored on the camera's parent body (same body the wireframe geoms live on) so the
    cone tracks the gimbal automatically. ``pos`` is the camera site offset slightly along
    its +Z (apex just past the lens, so the spotlight beam doesn't shade the head mesh
    from inside). ``dir`` is the site +Z — the camera's forward axis.

    ``active`` defaults to ``FOV_SPOT_ACTIVE`` (currently False — see that constant for why).
    The lights are ADDED either way, so ``nlight`` and every light index are identical
    whichever way the flag sits; only their emission differs. There is no digit-key toggle
    for lights — MjsLight has no ``group`` attribute in MuJoCo 3.11 — so this flag and
    ``model.light_active[i]`` after compile are the only levers. Digit-key "4" toggles the
    wireframe geoms, which are unaffected.

    Brightness comes from ``diffuse``, NOT ``light.intensity``: intensity is only read when
    the renderer runs in physical light units (its default is 0.0), so setting it alone
    leaves the light at the stock diffuse and looks like the spotlight "does not work".
    Likewise ``cutoff`` is in DEGREES (MuJoCo default 45) — passing radians collapses the
    cone to ~1 deg and it disappears. Both were the original bug.

    Tune ``diffuse`` / ``exponent`` per scene if two head-mounted cameras overlap on the
    workspace and double-brighten the table.
    """
    R_local = np.zeros(9)
    # Reuse the same site quat→mat path as add_fov_frustums so the two helpers cannot
    # silently disagree about the camera frame.
    for site in spec.sites:
        if not site.name.endswith("_rgb"):
            continue
        body = site.parent
        if body is None:
            continue
        p = np.asarray(site.pos, dtype=float)
        mujoco.mju_quat2Mat(R_local, np.asarray(site.quat, dtype=float))
        R = R_local.reshape(3, 3)

        # Apex just past the lens along +Z (camera forward). 0.05 m matches the
        # typical glass-thickness offset; tuned to keep the spot origin outside
        # the head mesh so the cone doesn't shade from inside the camera body.
        apex_cam = np.array([0.0, 0.0, 0.05])
        apex_world = p + R @ apex_cam

        light = body.add_light()
        light.name = f"{site.name}_fov_spot"
        light.type = mujoco.mjtLightType.mjLIGHT_SPOT
        light.pos = apex_world.tolist()
        # `dir` in MuJoCo is the unit direction the spot aims (apex -> target).
        light.dir = (R @ np.array([0.0, 0.0, 1.0])).tolist()
        light.cutoff = cutoff_deg          # degrees — MuJoCo's own unit for this field
        light.exponent = exponent
        light.diffuse = list(diffuse)
        light.attenuation = list(FOV_SPOT_ATTENUATION)
        light.castshadow = False  # wireframe already shadows; don't double up
        # Baked-but-dark by default. Kept as a light rather than skipped entirely so the
        # model's light table is the same shape either way (see `active` in the docstring).
        light.active = active


def _bake_hull_shell(
    spec: mujoco.MjSpec, site: mujoco.MjsSite, corner_dirs, R, p,
    *, near: float, far: float, suffix: str, rgb, alpha, group: int,
) -> None:
    """Add one truncated frustum shell (apex_ring at ``near`` -> far_ring at ``far``).

    Both rings wound both ways to defeat the scene-wide GL_CULL_FACE — see the docstring on
    ``add_fov_frustum_hull``. Emissive so the tint is identical on every face; doubled winding
    zeros the averaged vertex normals so Lambert shading is meaningless on these geoms anyway.
    """
    near_pts = [p + R @ (u * near) for u in corner_dirs]
    far_pts = [p + R @ (u * far) for u in corner_dirs]
    verts = near_pts + far_pts
    base = f"{site.name}{suffix}{FOV_HULL_SUFFIX}"

    mesh = spec.add_mesh()
    mesh.name = f"{base}_mesh"
    mesh.uservert = np.asarray(verts, dtype=float).ravel().tolist()
    mesh.userface = [i for a, b, c in _FRUSTUM_FACES for i in (a, b, c)] + \
                    [i for a, b, c in _FRUSTUM_FACES for i in (a, c, b)]

    mat = spec.add_material()
    mat.name = f"{base}_mat"
    mat.rgba = [*rgb, alpha]
    mat.emission = 1.0
    mat.specular = 0.0
    mat.shininess = 0.0

    geom = site.parent.add_geom()
    geom.name = base
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh.name
    geom.material = mat.name
    geom.rgba = [*rgb, alpha]
    geom.contype = 0
    geom.conaffinity = 0
    geom.mass = 0.0
    geom.group = group


def add_fov_frustum_hull(
    spec: mujoco.MjSpec,
    h_half: float = FOV_H_HALF_RAD,
    v_half: float = FOV_V_HALF_RAD,
    near: float = FOV_NEAR_M,
    far: float = FOV_FAR_M,
    alpha: float = FOV_HULL_ALPHA,
    group: int = FOV_GEOM_GROUP,
) -> None:
    """Add a translucent SOLID frustum per ``*_rgb`` site — the volume the wireframe outlines.

    Called by every asset that calls ``add_fov_frustums``, right next to it, so the hull is in
    the model wherever the wireframe is: training, cuRobo, ``run.py play``, every viewer.

    Why a volume at all: in a static figure the 13 capsule edges blend with object
    silhouettes and read as noise (advisor feedback: "messy and distracting"). A filled cone
    reads instantly. Measured on the real rig it is also CHEAPER than the wireframe — one
    mesh geom versus 13 capsules, 0.21 vs 0.24 ms/frame.

    Geometry is a TRUNCATED frustum (8 verts: near ring at ``near``, far ring at ``depth``; 4
    side quads + 2 caps = 12 triangles emitted once, then mirrored for backface = 24 in the
    mesh). The near ring matters because the cone IS visible from the camera's near plane out,
    not as a wedge glued to the lens — and the previous pyramid concentrated its heaviest
    overlap exactly where two shells stack in front of the camera, which read as a blob. With
    a real near plane the eye sees a frustum and the depth-confidence step lands on a quad,
    not a point. Corner directions come from the shared ``frustum_corner_dirs`` at the SAME
    ``h_half``/``v_half`` as the wireframe, so the hull cannot drift from it or from
    fov_detect's range gate.

    THREE shells per site: outer at ``far``, inner at ``FOV_HULL_STEREO_M``, plus an apex->near
    "tip" in saturated red (see ``FOV_TIP_RGBA``). The tip is the band inside the detector's
    near plane — stereo depth is unreliable there, so it reads as "not measurement space".
    Nested shells are separate geoms rather than one mesh because MuJoCo has no per-face alpha;
    nesting is what makes the depth-confidence gradient fall out of ordinary alpha compositing.

    Why baking this into every spec is free: a geom in a HIDDEN group casts no shadow and
    costs no pixels. Measured against a no-hull build of the same scene, with
    ``FOV_GEOM_GROUP`` off: 0 changed pixels, 0 darkened. So the model carries +2 geoms and
    +2 meshes PER SHELL (the URDF exporter strips all of them by name before its snapshot
    tuples are asserted — they are ``uservert``, inline verts URDF cannot express) and renders
    bit-identically until the group is switched on. That measurement is the whole reason this
    is no longer opt-in; an earlier reading of it was wrong and confined the hull to
    viewer-only ``__main__`` blocks, where no real entrypoint could reach it.

    THE ONE CATCH, and it only bites when the group is VISIBLE: MuJoCo's shadow toggle is
    per-LIGHT, never per-geom (``MjsGeom``/``MjsMaterial`` expose nothing), and the shadow
    pass ignores alpha, so a lit hull casts a fully opaque slab — 52403 darkened pixels
    against 522 of actual tint, which buries the floor. This function therefore also clears
    ``castshadow`` on the spec's lights (see the loop at the bottom), which drops it to 1562
    darkened and lets the tint through (842 -> 1920 blue-raised pixels).
    """
    corner_dirs = frustum_corner_dirs(h_half, v_half)
    baked = False
    # Idempotent: the assets bake this inside get_spec, so a caller that adds it again on an
    # already-built spec would otherwise die on "repeated name ... in mesh".
    existing = {mesh.name for mesh in spec.meshes}

    for site in spec.sites:
        if not site.name.endswith("_rgb"):
            continue
        if f"{site.name}{FOV_HULL_SUFFIX}_mesh" in existing:
            continue
        rgb = next((c for tag, c in _FOV_SIDE_RGBA.items() if tag in site.name),
                   _FOV_RGBA_DEFAULT)[:3]

        # Same site quat->mat path as add_fov_frustums so the two cannot disagree on the
        # camera frame. Verts are baked in PARENT-BODY coordinates (like the wireframe
        # edges), so the hull tracks the gimbal with no runtime work.
        p = np.asarray(site.pos, dtype=float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(site.quat, dtype=float))
        R = R.reshape(3, 3)

        # Outer shell always; inner one only when it actually sits inside (a caller shortening
        # `far` past the stereo depth would otherwise get two coincident, z-fighting hulls).
        # Main shells: outer at ``far``, inner at FOV_HULL_STEREO_M (only when strictly inside).
        if FOV_LARGE_VIS:
            for tag, depth in (("", far), ("_stereo", FOV_HULL_STEREO_M)):
                if depth >= far and tag == "_stereo":
                    continue
                _bake_hull_shell(spec, site, corner_dirs, R, p,
                                 near=near, far=depth, suffix=tag, rgb=rgb, alpha=alpha, group=group)
                baked = True

        # Apex->near tip echoes the module's color mixed toward black so cyan stays cyan,
        # orange stays orange. Same truncated-frustum geometry (apex is at distance 0 from ``p``).
        # Gated by FOV_TIP_VIS, independent of FOV_LARGE_VIS: the tip marks the lens even when the
        # cone fill is off.
        if FOV_TIP_VIS:
            tip_rgb = [c * FOV_TIP_MIX for c in rgb]
            _bake_hull_shell(spec, site, corner_dirs, R, p,
                             near=0.0, far=near, suffix="_tip",
                             rgb=tip_rgb, alpha=FOV_TIP_ALPHA, group=group)
            baked = True

        # Center-cone pyramid: scaled-down FOV (FOV_AXIS_CONE_RATIO * camera's H/V half-angle).
        # Optical-axis sight line. NOT gated by ``FOV_LARGE_VIS`` -- the tip marks the lens but
        # not its direction, and this is the only piece that shows where the camera is AIMED.
        axis_dirs = frustum_corner_dirs(h_half * FOV_AXIS_CONE_RATIO,
                                        v_half * FOV_AXIS_CONE_RATIO)
        _bake_hull_shell(spec, site, axis_dirs, R, p,
                         near=0.0, far=far, suffix="_axis",
                         rgb=rgb, alpha=FOV_AXIS_ALPHA, group=group)
        baked = True

    # Shadows off, HERE -- see THE ONE CATCH above. MuJoCo's only shadow lever is per-light, so
    # the fix must touch lights wherever it lives; doing it in the same function that bakes the
    # hull means no viewer, recorder or entrypoint has to know the hull exists.
    #
    # Unconditional rather than "only while group 4 is visible": the group is toggled by the
    # native digit-key handler, which exposes no hook to react to. Costs the robot's own ground
    # shadow -- accepted, and already the status quo in the standalone viewers, which zeroed
    # model.light_castshadow for exactly this reason.
    #
    # LIMIT: this reaches only the lights on THIS spec. A host that adds lights afterwards is
    # not covered -- mjlab's terrain generator does exactly that (terrain_generator.py, an
    # mjLIGHT_DIRECTIONAL on the "terrain" body, castshadow defaulting True), so every
    # mjlab-built env needs the post-compile `model.light_castshadow[:] = 0` its viewer applies.
    if baked:
        for light in spec.lights:
            light.castshadow = False

