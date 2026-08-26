"""MuJoCo -> Newton (ViewerGL) live PBR bridge.

Why
---
``photoreal/bridge.py``'s ``BlenderViewer`` proved the pattern: MuJoCo stays the single source of
truth for physics, an alternate renderer only ever receives already-resolved WORLD-SPACE poses per
frame, never steps its own physics. Newton fits the same pattern, and more simply: it runs
in-process (no subprocess/UDP hop like Blender needs), and ``ViewerGL`` reads body poses from
maximal coordinates (``state.body_q``, see ``newton/_src/viewer/viewer_gl.py::log_state``), so no
joint-space reconstruction (per-joint-type qpos layout, quaternion convention per joint) is needed
either -- just copy ``mj_forward``-resolved ``xpos``/``xquat`` straight into ``body_q``.

Scope
-----
Scene only: robot + terrain/props with PBR shading, real-time. No debug overlays (route markers,
FOV frustum, planner collision-sphere toggle) -- ``ViewerGL`` has the primitives
(``log_lines``/``log_arrows``/``log_capsules``) to add these later without rearchitecting, but
porting the existing ``mjv_addGeoms``-based overlay code is a separate pass.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp

import newton
import newton.viewer
from mjlab.viewer.base import BaseViewer

from asset_zoo.fov_frustum import (
    FOV_FAR_M,
    FOV_H_HALF_RAD,
    FOV_NEAR_M,
    FOV_V_HALF_RAD,
    _FOV_RGBA_DEFAULT,
    _FOV_SIDE_RGBA,
    frustum_corner_dirs,
)
from photoreal.bridge import visible_geom_ids

_GROUND_HALF_EXTENT_M = 50.0  # substitute finite box footprint for MuJoCo's infinite ground plane
_GROUND_HALF_THICK_M = 0.01


def _materialize_scene_xml(mj_model: mujoco.MjModel, xml_str: str) -> str:
    """Rewrite ``spec.to_xml()`` output into a form Newton's importer resolves correctly, using
    only data already sitting in the compiled ``mj_model``:

    1. Every ``<mesh name="..." file="...">`` element's ``file`` -> an absolute temp path written
       from the mesh data ALREADY triangulated in ``mj_model`` (``mesh_vert``/``mesh_face``, same
       arrays ``photoreal.bridge.export_scene`` reads). ``spec.to_xml()`` re-emits each mesh's
       ORIGINAL relative ``file`` string (e.g. ``"pelvis.STL"``) with no ``<compiler meshdir>`` to
       anchor it -- that context lived in whatever directory the source MJCF was loaded from and
       is not preserved through ``MjSpec``'s attach-based composition (confirmed empty
       ``<compiler meshdir>`` on a real mjlab-composed spec).

    2. Every ``<texture name="..." file="...">`` element's ``file`` -> an absolute temp PNG,
       written from the pixel data already decoded into ``mj_model`` (``tex_data``, indexed via
       ``tex_adr``/``tex_height``/``tex_width``/``tex_nchannel``). Same rationale as meshes: e.g.
       the Mars terrain's baked albedo map (``terrains/srb_baked_mesh.py``) is authored as an
       ABSOLUTE path, but MuJoCo's ``strippath``/serialization can still leave a bare filename in
       ``spec.to_xml()`` -- regenerating from the compiled model sidesteps path provenance
       entirely, same as meshes.

    3. Every ``<geom>`` element gets an EXPLICIT ``pos``/``quat``, copied from the compiled
       ``mj_model.geom_pos``/``geom_quat``. A geom that references a mesh with no explicit
       ``pos``/``quat`` of its own relies on MuJoCo's compiler implicitly folding the mesh's own
       ``mesh_pos``/``mesh_quat`` (its centering/inertial-alignment correction) into the geom's
       placement -- ``spec.to_xml()`` reproduces that by simply omitting the attributes (matching
       the source authoring), but Newton's importer treats a bare ``<geom mesh="X"/>`` as sitting
       at IDENTITY local offset, not at the mesh's own frame. Confirmed empirically: an omitted-
       pos/quat mesh geom's ``model.shape_transform`` came back identity while MuJoCo's own
       ``geom_xpos``/``geom_xmat`` for the same geom was a real offset+rotation, producing visibly
       wrong limb orientations. Writing the fully-resolved values explicitly removes the ambiguity
       for every geom (a no-op for primitives, which have no such implicit fold-in).

    Relies on ``<geom>`` elements appearing in the SAME order as ``mj_model``'s geom arrays
    (true for a compiled model: bodies/geoms compile in XML declaration order).
    """
    root = ET.fromstring(xml_str)
    tmpdir = tempfile.mkdtemp(prefix="newton_pbr_mesh_")
    for i, elem in enumerate(root.iter("mesh")):
        name = elem.get("name")
        if name is None or not elem.get("file"):
            continue
        mesh_id = mj_model.mesh(name).id
        va, vn = mj_model.mesh_vertadr[mesh_id], mj_model.mesh_vertnum[mesh_id]
        fa, fn = mj_model.mesh_faceadr[mesh_id], mj_model.mesh_facenum[mesh_id]
        verts = mj_model.mesh_vert[va : va + vn]
        faces = mj_model.mesh_face[fa : fa + fn]
        import trimesh  # local: only needed for this one-time mesh materialization

        tc_num = mj_model.mesh_texcoordnum[mesh_id]
        if tc_num > 0:
            # STL carries no UVs at all, so a textured mesh (e.g. the Mars terrain's baked
            # albedo map) needs an OBJ export instead, with UV coords -- MuJoCo indexes texcoords
            # PER FACE CORNER independently of vertex indices (mesh_facetexcoord), so corners
            # sharing a vertex but differing UVs must become separate OBJ vertices: dedupe on the
            # (vertex_index, texcoord_index, normal_index) triple, not on vertex_index alone.
            #
            # Normals matter here, not just UVs: MuJoCo also precomputes per-face-corner SMOOTH
            # normals (mesh_normal/mesh_facenormal, same indexing pattern as texcoords). Without
            # passing these through, trimesh's OBJ export writes no vertex normals at all, and
            # Newton's loader (newton/_src/utils/mesh.py) falls back to its own
            # compute_vertex_normals() -- which averages face normals PER OUTPUT (post-dedup)
            # vertex. Since the UV dedup above already splits vertices at every UV seam, that
            # fallback recompute only sees a partial ring of neighboring faces at each seam and
            # produces FACETED (non-smooth) normals concentrated exactly at UV seams -- confirmed
            # as the cause of a hard-edged checkerboard/blocky specular artifact on the Mars
            # terrain's rock geometry (which needs far more UV seams than the flat ground).
            # Using MuJoCo's own precomputed smooth normals sidesteps the recompute entirely.
            tca = mj_model.mesh_texcoordadr[mesh_id]
            texcoord = mj_model.mesh_texcoord[tca : tca + tc_num]
            face_tc = mj_model.mesh_facetexcoord[fa : fa + fn]
            na = mj_model.mesh_normaladr[mesh_id]
            nn = mj_model.mesh_normalnum[mesh_id]
            normal = mj_model.mesh_normal[na : na + nn]
            face_n = mj_model.mesh_facenormal[fa : fa + fn]
            triples = np.stack([faces.ravel(), face_tc.ravel(), face_n.ravel()], axis=1)
            uniq_triples, inverse = np.unique(triples, axis=0, return_inverse=True)
            out_verts = verts[uniq_triples[:, 0]]
            out_uv = texcoord[uniq_triples[:, 1]]
            out_normals = normal[uniq_triples[:, 2]]
            out_faces = inverse.reshape(-1, 3)
            mesh = trimesh.Trimesh(
                vertices=out_verts, faces=out_faces, vertex_normals=out_normals, process=False
            )
            mesh.visual = trimesh.visual.TextureVisuals(uv=out_uv)
            path = f"{tmpdir}/mesh_{i}.obj"
            mesh.export(path, include_normals=True)
        else:
            path = f"{tmpdir}/mesh_{i}.stl"
            trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)
        elem.set("file", path)
        # ``mj_model.mesh_vert`` is ALREADY scaled by the mesh's ``mesh_scale`` (confirmed
        # empirically: a mesh with ``mesh_scale=[0.001]*3`` -- i.e. authored in millimeters --
        # still has a real-world-meter-sized ``mesh_vert`` bbox, e.g. a humanoid torso at
        # +-0.065/0.09/0.24 m, not +-65/90/240). ``spec.to_xml()`` still re-emits that mesh's
        # ORIGINAL ``scale="0.001 0.001 0.001"`` attribute (a separate, independent piece of
        # mesh metadata from the vertex data itself) -- left in place, Newton's importer applies
        # it AGAIN on top of already-scaled vertices, shrinking every mesh 1000x into sub-pixel
        # invisibility. Confirmed as the cause of "no humanoid visible, only 4 capsule primitives"
        # on a robot with mm-authored meshes: G1/Argus Mini never hit this because their meshes
        # happen to use ``scale="1 1 1"`` (a no-op double-apply). Explicit "1 1 1" here, matching
        # this module's running pattern of never trusting Newton to replicate a MuJoCo-specific
        # implicit default -- the file's data is final-scale, so any further scale must be an
        # identity.
        elem.set("scale", "1 1 1")

    for i, elem in enumerate(root.iter("texture")):
        name = elem.get("name")
        if name is None or not elem.get("file"):
            continue
        from PIL import Image  # local: only needed for this one-time texture materialization

        tex_id = mj_model.texture(name).id
        adr = mj_model.tex_adr[tex_id]
        h, w, nc = int(mj_model.tex_height[tex_id]), int(mj_model.tex_width[tex_id]), int(
            mj_model.tex_nchannel[tex_id]
        )
        data = mj_model.tex_data[adr : adr + h * w * nc].reshape(h, w, nc)
        mode = {1: "L", 3: "RGB", 4: "RGBA"}[nc]
        path = f"{tmpdir}/tex_{i}.png"
        Image.fromarray(data, mode=mode).save(path)
        elem.set("file", path)

    # MuJoCo 3.x materials can bind textures via nested <layer texture="X" role="rgb"/> children
    # instead of a flat texture="X" attribute (seen on the Mars terrain's material, which uses
    # separate albedo/normal/roughness layers). Newton's importer only reads the flat attribute
    # (``material_info.get("texture")`` in ``import_mjcf.py``), so the base-color layer is
    # flattened onto the <material> element itself; normal/roughness layers are dropped -- Newton
    # has no material slot for them regardless (see ``terrains/srb_baked_mesh.py``'s own comment
    # that only the photoreal bridge, i.e. this module, reads them, and it currently only wires
    # the base color).
    for mat_elem in root.iter("material"):
        for layer in list(mat_elem.iter("layer")):
            if layer.get("role") == "rgb" and layer.get("texture"):
                mat_elem.set("texture", layer.get("texture"))
            mat_elem.remove(layer)

    # <geom> elements also appear as <default> class templates (not instantiated bodies) --
    # restrict to <worldbody>'s tree, the actual body/geom instance hierarchy, to keep the count
    # aligned with mj_model.ngeom.
    geoms = list(root.find("worldbody").iter("geom"))
    assert len(geoms) == mj_model.ngeom, (
        f"XML <geom> count ({len(geoms)}) != mj_model.ngeom ({mj_model.ngeom}); "
        "positional geom_pos/geom_quat matching would silently mismatch."
    )
    # Elements have no .getparent() in xml.etree; build the lookup once for the removal below.
    parent_of = {child: parent for parent in root.iter() for child in parent}
    visible_gids = set(visible_geom_ids(mj_model).tolist())
    for gid, elem in enumerate(geoms):
        if gid not in visible_gids:
            # A non-visible-group geom (collision proxy group 3, curobo/mink IK-avoidance proxy
            # group 5 -- see ``asset/duke_v2/parallel_gripper/parallel_gripper_creation.py``'s
            # IKPROXY_GROUP, FOV hull group 4 -- see ``fov_frustum_lines``'s docstring for why
            # THAT one specifically must not render either) can still share a BODY with a real
            # visible geom (e.g. a gripper rack's visual mesh + its collision boxes + its ikproxy
            # capsule all live on the same body) -- ``build_newton_model_and_body_map``'s
            # ``ignore_names`` only excludes bodies with NO visible geoms at all, so it keeps the
            # whole body, hull/proxy geoms included. Dropped here at the GEOM level instead,
            # against the same visible-group definition ``visible_geom_ids`` uses everywhere else
            # in this bridge, so every group outside (0, 1, 2) is excluded uniformly regardless of
            # which body it landed on. ``fov_frustum_lines`` draws the FOV cone's equivalent as a
            # live overlay instead of the (now-dropped) baked hull.
            parent_of[elem].remove(elem)
            continue
        px, py, pz = mj_model.geom_pos[gid]
        qw, qx, qy, qz = mj_model.geom_quat[gid]
        r, g, b, a = mj_model.geom_rgba[gid]
        if mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            # Newton's importer hardcodes width=length=0.0 for every <geom type="plane">
            # (``import_mjcf.py``: "MuJoCo planes are always infinite for collision; pass 0
            # extents") -- correct for physics, but ``create_mesh_plane`` takes those literally
            # and emits a ZERO-AREA quad (all 4 verts collapse to the origin) -- the ground was
            # never hidden by any visibility toggle, it's geometrically degenerate. Rewritten to
            # a large finite BOX instead: Newton's box importer uses the real size, no special-
            # casing. Shifted down by the half-thickness so the top face still sits at the
            # plane's original z (MuJoCo planes are an infinitesimal sheet at local z=0).
            elem.set("type", "box")
            elem.set("size", f"{_GROUND_HALF_EXTENT_M} {_GROUND_HALF_EXTENT_M} {_GROUND_HALF_THICK_M}")
            pz -= _GROUND_HALF_THICK_M
            # MuJoCo's groundplane is ONE geom serving both visual and collision roles (real
            # contype/conaffinity), unlike every other geom in this codebase's own asset scripts,
            # which always split visual (contype=0) from collision proxies. Newton's importer
            # buckets any geom with real contype/conaffinity and no class= as a COLLISION-ONLY
            # proxy (import_mjcf.py: `collides_with_anything` -> `colliders` list), which renders
            # only if `show_colliders` -- False whenever the model also has proper visual shapes
            # (confirmed: this box's ShapeFlags had COLLIDE_SHAPES set but VISIBLE unset). Newton
            # never simulates physics here (display-only), so stripping contype/conaffinity on
            # this render-only copy is free and routes it into the always-visible visuals bucket.
            elem.set("contype", "0")
            elem.set("conaffinity", "0")
        elem.set("pos", f"{px} {py} {pz}")
        elem.set("quat", f"{qw} {qx} {qy} {qz}")
        # Explicit rgba, always -- a geom/material whose color equals MuJoCo's default is
        # omitted from spec.to_xml() as redundant (same reasoning as pos/quat above), and
        # Newton's importer falls back to an auto-generated per-shape rainbow palette (see
        # newton/_src/utils/import_mjcf.py:1068-1078) when neither the geom nor its material
        # supplies rgba text -- confirmed empirically on Argus Mini, where several geoms/materials
        # (e.g. "robot/white", no rgba attribute at all) rendered as arbitrary saturated hues
        # instead of their real color.
        elem.set("rgba", f"{r} {g} {b} {a}")

    return ET.tostring(root, encoding="unicode")


def build_newton_model_and_body_map(
    mj_model: mujoco.MjModel, spec: mujoco.MjSpec
) -> tuple["newton.Model", np.ndarray]:
    """Parse ``spec`` (the SAME spec ``mj_model`` was compiled from) into a Newton ``Model``, plus
    a ``(mj_body_id, newton_body_id)`` map for per-frame pose sync.

    Newton's importer prefixes body labels with their full ancestor path (e.g.
    ``"mjlab scene/worldbody/robot/pelvis"``) -- and MuJoCo body names themselves already contain
    a ``/`` when composed via mjlab's entity-prefixed attach (e.g. ``"robot/pelvis"``), so bodies
    are matched by checking the newton label ENDS WITH the mj body's full name at a ``/``
    boundary, not by taking the label's last path segment (which would wrongly strip the
    ``"robot/"`` prefix and never match).

    Non-visible-group geoms (MuJoCo groups 3+: collision proxies, IK-avoidance proxies, FOV hull
    -- see ``_materialize_scene_xml``'s geom-group filter) are dropped from the XML directly,
    at the GEOM level, before this ever reaches Newton's importer. A body left with zero geoms
    that way (e.g. a pure mount/frame body) still gets imported as an empty body -- harmless,
    nothing to render -- rather than excluded via ``add_mjcf``'s ``ignore_names``. That was tried
    first and reverted: Newton matches ``ignore_names`` against GEOM names with ``re.match``
    (prefix, not exact -- confirmed in ``import_mjcf.py``), so a hidden zero-geom body's bare
    name is a false-positive PREFIX match against any sibling body's geom names that happen to
    start with it (e.g. hidden mount body ``"robot/cam_base"`` silently ate the real, visible
    ``"robot/cam_base_left_visual"``/``"...cam_base_right_visual"`` geoms too, since both names
    start with the mount body's). Geom-level filtering has no such collision risk.
    """
    xml_str = _materialize_scene_xml(mj_model, spec.to_xml())
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        xml_str,
        floating=True,
        ignore_inertial_definitions=False,
    )
    model = builder.finalize()

    mj_names = [(i, mj_model.body(i).name) for i in range(mj_model.nbody) if mj_model.body(i).name]

    def _match(label: str) -> int | None:
        for mj_id, mj_name in mj_names:
            if label == mj_name or label.endswith("/" + mj_name):
                return mj_id
        return None

    pairs = [
        (mj_id, newton_id)
        for newton_id, label in enumerate(model.body_label)
        if (mj_id := _match(label)) is not None
    ]
    body_map = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    _apply_pbr_materials(mj_model, model, body_map)
    return model, body_map


def _apply_pbr_materials(mj_model: mujoco.MjModel, model: "newton.Model", body_map: np.ndarray) -> None:
    """Set real roughness/metallic on every MESH shape, from the MuJoCo material of its body's
    (first) visible mesh geom -- one material per body, matching this codebase's one-visual-
    mesh-per-link convention.

    Newton's MJCF importer never reads ``mat_shininess``/``mat_specular``/``mat_reflectance`` at
    all (confirmed: zero references in ``import_mjcf.py``), so every imported shape silently
    falls back to the PBR shader's hardcoded default (``wp.vec4(0.5, 0.0, ...)`` in
    ``viewer.py``) regardless of what's authored in the MJCF -- a uniform mid-gloss plastic look
    on everything, which is why the "PBR" renderer barely reads as different from MuJoCo's
    Blinn-Phong: the shader (real Cook-Torrance GGX + Fresnel, ``shaders.py``) never gets real
    material data to work with. The renderer's ONLY override lever is a plain ``.roughness``/
    ``.metallic`` attribute read off the shape's source ``Mesh`` object each frame if present
    (``viewer.py:2406-2409``) -- set once here (static per session, no per-frame cost) rather
    than every ``sync_env_to_viewer`` call. Primitives (sphere/capsule/box/plane) have no such
    override path in Newton's own code, so this only reaches mesh-type shapes -- acceptable,
    meshes are the overwhelming majority of visible surface area in these scenes.
    """
    shape_type = model.shape_type.numpy()
    shape_body = model.shape_body.numpy()
    for mj_body_id, newton_body_id in body_map.tolist():
        geom_ids = [
            g for g in range(mj_model.ngeom)
            if mj_model.geom_bodyid[g] == mj_body_id and mj_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        ]
        if not geom_ids:
            continue
        matid = mj_model.geom_matid[geom_ids[0]]
        if matid < 0:
            continue
        roughness = 1.0 - float(mj_model.mat_shininess[matid])
        metallic = float(mj_model.mat_reflectance[matid])
        for i in np.flatnonzero((shape_body == newton_body_id) & (shape_type == int(newton.GeoType.MESH))):
            src = model.shape_source[i]
            src.roughness = roughness
            src.metallic = metallic


def sync_mjdata_to_newton_state(mjd: mujoco.MjData, body_map: np.ndarray, state) -> None:
    """Copy world-space body poses from a post-``mj_forward`` ``MjData`` into ``state.body_q``.

    Maximal-coordinate sync: no joint-space reconstruction. ``mjd.xquat`` is MuJoCo's wxyz;
    ``wp.transform``'s rotation is xyzw, hence the column reorder.
    """
    if body_map.shape[0] == 0:
        return
    mj_ids, newton_ids = body_map[:, 0], body_map[:, 1]
    pos = mjd.xpos[mj_ids]
    quat_xyzw = mjd.xquat[mj_ids][:, [1, 2, 3, 0]]
    xforms = np.concatenate([pos, quat_xyzw], axis=1).astype(np.float32)
    body_q = state.body_q.numpy()
    body_q[newton_ids] = xforms
    state.body_q.assign(body_q)


_FOV_CORNER_DIRS = np.array(frustum_corner_dirs(FOV_H_HALF_RAD, FOV_V_HALF_RAD))  # (4, 3), unit, camera frame


def fov_camera_sites(mj_model: mujoco.MjModel) -> list[tuple[int, tuple[float, float, float]]]:
    """(site_id, rgb) for every ``*_rgb`` camera site, colored the same way
    ``asset_zoo.fov_frustum.add_fov_frustums`` picks per-module colors."""
    sites = []
    for i in range(mj_model.nsite):
        name = mj_model.site(i).name
        if not name.endswith("_rgb"):
            continue
        rgb = next((c[:3] for tag, c in _FOV_SIDE_RGBA.items() if tag in name), _FOV_RGBA_DEFAULT[:3])
        sites.append((i, rgb))
    return sites


def fov_frustum_lines(
    mjd: mujoco.MjData, sites: list[tuple[int, tuple[float, float, float]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """World-space wireframe edges (4 near-rect + 4 far-rect + 4 lens rays per site) -- the
    same 12-edge cone ``add_fov_frustums`` bakes -- computed LIVE from ``mjd.site_xpos``/
    ``site_xmat`` each frame instead of baked as MJCF geoms.

    Why not just render the baked hull (``add_fov_frustum_hull``, group ``FOV_GEOM_GROUP``)
    like the native viewer's default-on toggle does: Newton's shape pipeline has no alpha
    channel at all (confirmed empirically -- ``Model.shape_color`` is ``wp.vec3``, no
    per-shape alpha uniform anywhere, no ``GL_BLEND`` in the main scene render pass in
    ``viewer_gl.py``/``opengl.py``, only in the text/FPS overlay). The hull's shells are
    ``alpha=0.01-0.10`` and ``emission=1.0`` (flat, unlit) BY DESIGN -- dumping them in
    unfiltered would render fully OPAQUE at full saturation: a solid bright cyan/orange/green
    pyramid over the scene. That is exactly the "translucent stage-light volume ... too
    visually distracting" design this codebase already tried and reverted in favor of a thin
    wireframe outline (see ``fov_frustum.py``'s module docstring) -- so this reproduces that
    same wireframe-only choice for Newton rather than the alpha-only hull, using
    ``ViewerGL.log_lines`` (screen-space line quads, opaque, no alpha required) instead of
    baked geoms.
    """
    if not sites:
        return None
    starts, ends, colors = [], [], []
    for site_id, rgb in sites:
        pos = mjd.site_xpos[site_id]
        rot = mjd.site_xmat[site_id].reshape(3, 3)
        near_c = pos + (rot @ (_FOV_CORNER_DIRS * FOV_NEAR_M).T).T
        far_c = pos + (rot @ (_FOV_CORNER_DIRS * FOV_FAR_M).T).T
        for i in range(4):
            starts.append(near_c[i]); ends.append(near_c[(i + 1) % 4]); colors.append(rgb)
        for i in range(4):
            starts.append(far_c[i]); ends.append(far_c[(i + 1) % 4]); colors.append(rgb)
        for i in range(4):
            starts.append(pos); ends.append(far_c[i]); colors.append(rgb)
    return (
        np.array(starts, dtype=np.float32),
        np.array(ends, dtype=np.float32),
        np.array(colors, dtype=np.float32),
    )


def look_at_pitch_yaw(cam_pos: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Pitch/yaw (degrees) so ``ViewerGL``'s front vector
    ``(cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))`` points from ``cam_pos`` at
    ``target``."""
    dx, dy, dz = target - cam_pos
    yaw = np.degrees(np.arctan2(dy, dx))
    pitch = np.degrees(np.arctan2(dz, np.hypot(dx, dy)))
    return float(pitch), float(yaw)


_FOCUS_RADIUS_M = 2.0  # fixed robot-scale framing radius when a body to focus on is known


def auto_frame_camera(viewer: "newton.viewer.ViewerGL", state, focus: np.ndarray | None = None) -> np.ndarray:
    """Frame the camera. Returns the world-space camera offset used, so a caller tracking a
    single body (see ``track_body_id``) can re-apply the same relative offset.

    ``focus`` (a known body's world position, e.g. the tracked body) frames at a FIXED robot-scale
    radius (``_FOCUS_RADIUS_M``) around it. Without it, falls back to the whole scene's
    body-position bounding sphere -- fine for small scenes (curobo's tabletop harness), but on a
    large terrain (e.g. a 64 m Mars grid) that sphere is dominated by the terrain body's own
    extent, framing the camera terrain-scale with the robot reduced to a speck. Always pass
    ``focus`` when a trackable body exists.
    """
    if focus is not None:
        offset = _FOCUS_RADIUS_M * np.array([0.6, -0.6, 0.5])
        cam_pos = focus + offset
        pitch, yaw = look_at_pitch_yaw(cam_pos, focus)
        viewer.set_camera(wp.vec3(*cam_pos.tolist()), pitch, yaw)
        return offset

    positions = state.body_q.numpy()[:, :3]
    if positions.shape[0] == 0:
        return np.array([2.0, -2.0, 1.5])
    center = positions.mean(axis=0)
    radius = max(float(np.linalg.norm(positions - center, axis=1).max()), 0.5)
    offset = radius * 3.0 * np.array([0.6, -0.6, 0.5])
    cam_pos = center + offset
    pitch, yaw = look_at_pitch_yaw(cam_pos, center)
    viewer.set_camera(wp.vec3(*cam_pos.tolist()), pitch, yaw)
    return offset


def track_body_id(mj_model: mujoco.MjModel) -> int | None:
    """First non-welded (i.e. dynamically simulated, typically the floating base) body id, mirroring
    ``mjlab.viewer.native.viewer.NativeMujocoViewer._set_camera_auto_track``'s selection so the
    Newton viewer chases the same body the native viewer's tracking camera would. ``None`` if
    every body is welded to the world (e.g. a static scene)."""
    for body_id in range(mj_model.nbody):
        is_weld = mj_model.body_weldid[body_id] == 0
        root_id = mj_model.body_rootid[body_id]
        root_is_mocap = mj_model.body_mocapid[root_id] >= 0
        if not (is_weld and not root_is_mocap):
            return body_id
    return None


class NewtonPbrViewer(BaseViewer):
    """Drives a Newton ``ViewerGL`` window as the display for a policy rollout.

    Steps physics through ``BaseViewer``'s budget accumulator exactly like the native viewer, but
    instead of drawing via MuJoCo's rasterizer, copies one environment's state onto a CPU
    ``MjData``, runs forward kinematics for world body poses, and pushes them into a Newton
    ``State`` that ``ViewerGL`` renders with PBR lighting.

    Assumptions:
        * A display is available; this is an interactive viewer by definition.
        * Single environment shown at a time (``env_idx``), matching the native viewer default.
    """

    def __init__(self, env, policy, frame_rate: float = 60.0, env_idx: int = 0, **kwargs):
        super().__init__(env, policy, frame_rate=frame_rate, **kwargs)
        self.env_idx = env_idx
        self._mjm: mujoco.MjModel | None = None
        self._mjd: mujoco.MjData | None = None
        self._model = None
        self._state = None
        self._body_map: np.ndarray | None = None
        self._viewer: "newton.viewer.ViewerGL | None" = None
        self._track_body_id: int | None = None
        self._track_last_pos: np.ndarray | None = None
        self._fov_sites: list[tuple[int, tuple[float, float, float]]] = []

    def setup(self) -> None:
        self._mjm = self.env.unwrapped.sim.mj_model
        self._mjd = mujoco.MjData(self._mjm)
        spec = self.env.unwrapped.scene.spec
        self._model, self._body_map = build_newton_model_and_body_map(self._mjm, spec)
        self._state = self._model.state()

        self._viewer = newton.viewer.ViewerGL()
        self._viewer.set_model(self._model)

        self._pull_state()
        sync_mjdata_to_newton_state(self._mjd, self._body_map, self._state)
        self._track_body_id = track_body_id(self._mjm)
        focus = self._mjd.xpos[self._track_body_id].copy() if self._track_body_id is not None else None
        auto_frame_camera(self._viewer, self._state, focus=focus)
        self._track_last_pos = focus
        self._fov_sites = fov_camera_sites(self._mjm)

        print("[INFO] Newton PBR viewer ready")

    def _pull_state(self) -> None:
        """Copy one env's state off the GPU and run forward kinematics for world body poses."""
        sim_data = self.env.unwrapped.sim.data
        self._mjd.qpos[:] = sim_data.qpos[self.env_idx].cpu().numpy()
        self._mjd.qvel[:] = sim_data.qvel[self.env_idx].cpu().numpy()
        mujoco.mj_forward(self._mjm, self._mjd)

    def _track_camera(self) -> None:
        """Chase-cam: translate BOTH the camera position and its orbit pivot by exactly how far
        the tracked body moved since last frame, so the view follows the robot across large/
        scrolling terrain (e.g. the Mars mesh scenes) without fighting manual mouse orbit/zoom.

        Deliberately does NOT call ``set_camera`` (which overwrites pos/pitch/yaw outright): a
        first version did, and every frame it stomped whatever angle/distance the user had just
        dragged/scrolled to, making the view impossible to adjust. Translating ``camera.pos`` and
        ``camera.pivot`` by the same delta instead preserves the user's current orbit angle and
        zoom -- exactly how a third-person chase cam is expected to behave.
        """
        if self._track_body_id is None:
            return
        target = self._mjd.xpos[self._track_body_id].copy()
        delta = target - self._track_last_pos
        self._track_last_pos = target
        if not np.any(delta):
            return
        from pyglet.math import Vec3 as PyVec3

        dv = PyVec3(*delta.tolist())
        cam = self._viewer.camera
        cam.pos = cam.pos + dv
        cam.pivot = cam.pivot + dv

    def sync_env_to_viewer(self) -> None:
        self._pull_state()
        sync_mjdata_to_newton_state(self._mjd, self._body_map, self._state)
        self._track_camera()
        self._viewer.begin_frame(time=self._step_count * self.env.unwrapped.step_dt)
        self._viewer.log_state(self._state)
        lines = fov_frustum_lines(self._mjd, self._fov_sites)
        if lines is not None:
            starts, ends, colors = lines
            self._viewer.log_lines(
                "fov_frustum",
                wp.array(starts, dtype=wp.vec3),
                wp.array(ends, dtype=wp.vec3),
                wp.array(colors, dtype=wp.vec3),
            )
        self._viewer.end_frame()

    def sync_viewer_to_env(self) -> None:
        """No inbound channel: the Newton window is display-only, all control stays in the
        terminal/gamepad, matching ``BlenderViewer``."""

    def is_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
