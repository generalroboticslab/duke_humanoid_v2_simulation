"""Generate the head-camera MJCF rigs (currently dual + single) the HUMANOID way
(frame-on-joint), via robot_builder.

This SUPERSEDES the hand-authored head_camera_dual.xml. The hand-authored file put
every body at identity in one global CNC frame (body_pos=0, anchors shoved into
jnt_pos, fullinertia, no site) — violating the humanoid normalization regimen. This
script routes the camera through the SAME emitter as the humanoid (asset/duke_v2/humanoid_v21/
robot_builder.py), so the output gets, for free:
  * frame-on-joint: each link's body frame sits ON its joint (jnt_pos=0, +Z axis)
  * inertia emitted as diaginertia + principal-axis quat in the joint-local frame
  * identity-pos/quat suppression, %.6g formatting, deterministic ordering
  * visual/collision childclass split, <body>_collision[N] naming

Every rig is built from ONE `build_column`: each camera module is the SAME physical
part, only placed by a different `Mount` pose. A rig is pure config (`RIGS` below):
N mounts, any planar arrangement, any orientation -- adding one is adding a `Rig(...)`
row, never an engine edit.

PROVENANCE (Path B): the per-link mass/COM/inertia are the ACCURATE values already
captured in cam_fusion_info.py (global CNC frame). We do NOT re-export from Fusion;
instead we analytically rotate/translate them into each joint-local frame. Verified
2026-06-17: the global-frame eigenvalues equal the compiled-model diaginertia exactly,
and a rotation preserves eigenvalues — so this is physically identical to a perfect
joint-local Fusion re-export, only the COM vector and frame orientation change.

DECISIONS (locked 2026-06-17):
  (1) the static mount + yaw-motor mass stays in base_left/base_right (0.24606 kg
      each); the module root 'base' is a massless attach frame.
  (2) joint axes are pure local +Z (the body frame is oriented so +Z is the spin
      axis; sign chosen to reproduce the current physical positive direction:
      yaw +Z = world -z, pitch_left +Z = world -y, pitch_right +Z = world +y).
  (3) the two dual columns differ by a PROPER 180-deg rotation about z, not a mirror.
      Generalized 2026-07-28: that is no longer a special case, it is the
      `rpy=(0,0,pi)` value of a general `Mount` rotation (see MOUNT PLACEMENT).
      Meshes are SHARED across all mounts (the body frame supplies the rotation), so
      only 3 mesh assets are emitted no matter how many modules a rig has.

Geometry note: the 3 OBJ meshes and the collision fromto endpoints are defined in
the global CNC frame (= base_link frame at the identity attach). frame-on-joint moves
each body frame to its joint, so every visual/collision geom carries origin =
inv(T_body) [ @ Rz180 for the imaged column], pinning the CNC geometry back to its
true place. This is the same transform used for COM/inertia, so everything stays
consistent and the compiled WORLD positions match the old model exactly.

MOUNT PLACEMENT (generalized 2026-07-28, supersedes the SINGLE-MODULE CENTERING /
FACING notes): a Mount is a rigid CNC->world map W that lands the CANONICAL column's
yaw axis (the anchor A = (0,-YAW_ANCHOR_Y,0)) on `xyz` with rotation `rpy`. The
canonical column (W=I) has a vertical yaw shaft and looks BACKWARD (-X); rpy=(0,0,pi)
looks FORWARD (+X).

The key invariant: every LOCAL quantity is independent of W. Each body frame is
T_body = W @ T_canonical, so a geom origin inv(T_body) @ W @ X collapses to
inv(T_canonical) @ X — W cancels exactly. Mesh origins, collision capsules, camera
sites and the yaw->pitch relative offset are therefore byte-identical for every
module of every rig, and W survives in only two places: the root fixed joint's origin,
and the physics (COM by the full affine W, inertia by its rotation).

That is why the old single-module hack (compute the left column, then patch the root
joint's y to 0) is gone: it was hand-applying exactly this rigid translation. The
earlier trap it worked around still holds and is worth remembering — recomputing
everything from a NEW anchor (YAW_ANCHOR_Y=0) is a NO-OP, because the anchor is used
both to build each body's local frame and to place it, and the two uses cancel.
YAW_ANCHOR_Y is now purely the canonical column's own anchor offset; placement lives
only in Mount.xyz.

Run:  python head_camera_creation.py     # writes every rig in RIGS
"""
from __future__ import annotations
import sys
import re
import pathlib
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))                                   # cam_fusion_info
sys.path.insert(0, str(HERE.parents[2] / "asset" / "create"))  # robot_builder, builder_helpers

from robot_builder import (
    Robot, Link, Joint, Geom, Origin, Inertial, Mesh, Capsule, Site, JointLimit, Material,
    rot_with_zaxis, mat_to_rpy, capsule_transform,   # shared SE(3) / fromto-capsule helpers
    _rpy_to_transform,                               # rpy -> matrix (inverse of mat_to_rpy)
)
from builder_helpers import STANDARD_MATERIALS
import cam_fusion_info as CF

# Tones and finishes track humanoid_v21_creation_v3 EXACTLY (dark anodized graphite; see the
# note there on why specular is 0.75 -- the Blender photoreal grade reads mat_specular as
# Principled Metallic and 1-mat_shininess as Roughness) so a head cam mounted on the humanoid
# reads as the same build. Re-toned HERE ONLY -- STANDARD_MATERIALS is shared with argus /
# ball_circle / the humanoid v2 script and must not be re-toned for them.
#
# The whole actuated column (mount, yaw motor, pitch bracket) is now ONE dark shade. The
# earlier light/dark alternation existed to keep the links individually readable, but the
# D436 is a far stronger read: it is the only bright object on the column, so the silhouette
# is legible from the camera alone and the motors stop competing with it.
CAM_MATERIALS = [m for m in STANDARD_MATERIALS if m.name not in ("light_grey", "dark_grey")] + [
    Material("light_grey", rgba=[0.19, 0.20, 0.22, 1], specular=0.75, shininess=0.05),
    Material("dark_grey", rgba=[0.11, 0.115, 0.125, 1], specular=0.35, shininess=0.05),
    # Intel RealSense D436: bare anodized-aluminium housing, noticeably brighter than the
    # motors and less rough than their bead-blasted finish (shininess 0.2 -> Roughness 0.8).
    Material("d436", rgba=[0.58, 0.60, 0.62, 1], specular=0.75, shininess=0.2),
]
LINK_MATERIAL = {"base_left": "dark_grey", "yaw_left_link": "dark_grey",
                 "pitch_left_link": "dark_grey"}

# ── geometry constants (global CNC frame = base_link frame at identity attach) ──
YAW_ANCHOR_Y   = 0.065        # -y of the CANONICAL column's yaw axis (its own anchor, NOT a rig layout knob)
YAW_Z          = 0.52         # yaw axis height
PITCH_Z        = 0.625442     # pitch axis height
YAW_AXIS_W     = np.array([0.0, 0.0, -1.0])   # yaw shaft (blue +Z) -> world -Z (DOWN), both sides (flipped: yaw positive-rotation sense reversed)
PITCH_AXIS_W_L = np.array([0.0, -1.0, 0.0])   # pitch shaft of the CANONICAL (-y) column -> -Y;
                                              # +y (imaged) column gets Rz180 -> +Y. Sign chosen so
                                              # dragging the pitch slider tilts the camera the intuitive
                                              # way (hinge: shaft +Z direction == positive-rotation sense).
YAW_RANGE      = (-4.7124, 4.7124)            # +-1.5pi
PITCH_RANGE    = (-1.5708, 1.5708)            # +-pi/2
JOINT_DAMPING  = 0.5
MESH_SCALE     = [0.001, 0.001, 0.001]

# collision capsules per LEFT link: (radius, p1, p2) with endpoints in CNC frame.
# Transcribed from the prior hand-authored head_camera_dual.xml (humanoid-convention
# capsule fits of the link meshes); base mount simplified to one conservative capsule.
# NOTE: unlike mass/COM/inertia (sourced from cam_fusion_info), these are NOT derived
# from the meshes — re-fit them if the OBJ meshes are re-exported.
COLLISION_L = {
    "base_left": [
        # (0.035, (0, -0.065, 0.435), (0, -0.065, 0.525)),
        (0.035, (0, -0.065, 0.435), (0, -0.065, 0.55)),

    ],
    "yaw_left_link": [
        # (0.018, (0, -0.065, 0.552), (0, -0.065, 0.596)),
        (0.023, (0, -0.085, 0.6254), (0, -0.045, 0.6254)),
    ],
    "pitch_left_link": [
        (0.016, (-0.012, -0.094, 0.662), (-0.012, -0.036, 0.662)),
        # (0.0155, (0, -0.083, 0.628), (0, -0.04, 0.628)),
    ],
}
# ── D436 camera reference frames (canonical -y column) ─────────────────────────
# cam_front_center := the FRONT-GLASS centre, measured DIRECTLY in the user's Fusion
# assembly (Inspect>Measure on the D436 front glass face) in the canonical (-y) CNC
# frame: X=-0.025, Y=-0.065, Z=0.662 m. This is the reference the RGB site offsets from.
GLASS_CENTER_L = np.array([-0.025, -0.065, 0.662])
# canonical (-y) optical frame (+Z out lens=-X, +Y down=-Z, +X right=+Y); maps optical-
# frame offsets into the canonical world frame (W=Rz180 then mirrors them for the +y col).
_FWD_C, _DOWN_C = np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
R_OPT_CANON = np.column_stack([np.cross(_DOWN_C, _FWD_C), _DOWN_C, _FWD_C])
# RGB optical centre relative to the GLASS centre = two optical-frame legs added up, so the
# RGB site lands on the real RGB lens while cam_front_center stays at the user's glass point:
#   (a) glass -> depth origin (left IR): official D400 datasheet — 17.5 mm toward the left
#       imager (Table 4-19, -X optical) and 4.2 mm BEHIND the front glass (Table 4-16,
#       D435/D435i, -Z optical).
#   (b) depth origin -> RGB: per-unit factory color->depth extrinsic read off the REAL
#       D436 (serial 408122071763) via rs2_get_extrinsics (read_d436_extrinsics.py),
#       t_c2d in the DEPTH OPTICAL frame; rotation 0.16 deg ~ identity (RGB ~parallel).
DEPTH_FROM_GLASS_OPT = np.array([-0.0175, 0.0, -0.0042])              # (a) optical frame
RGB_C2D_T_OPT        = np.array([-0.0147991, 0.0001231, -0.0001451])  # (b) optical frame
RGB_FROM_GLASS_CANON = R_OPT_CANON @ (DEPTH_FROM_GLASS_OPT + RGB_C2D_T_OPT)

VISUAL_MESH = {"base_left": "base_left.obj", "yaw_left_link": "yaw_left.obj",
               "pitch_left_link": "pitch_left_bracket.obj"}
# Second visual geom on a link, carrying its own material: (mesh file, material).
EXTRA_VISUAL = {"pitch_left_link": ("pitch_left_d436.obj", "d436")}

MESH_DIR = HERE / "meshes" / "links"
# D436 housing bounds in the canonical CNC frame, metres, with ~1 mm margin. The camera is a
# separate solid in the Fusion assembly, so it survives OBJ export as its own connected
# components (one 25 x 90 x 25 mm shell plus the sliver faces of its lens/label decals); this
# box selects them by containment. Centred on GLASS_CENTER_L's lens plane, 25 mm deep.
D436_BOX = (np.array([-0.026, -0.111, 0.6490]), np.array([0.001, -0.019, 0.6755]))

RZ180 = CF.RZ180                              # 180 deg about z (proper rotation); the forward-facing mount rotation

# link/joint names per column variant. An un-tagged column drops the L/R suffix
# entirely since there is nothing to disambiguate.
_CNC_LINKS = ("base_left", "yaw_left_link", "pitch_left_link")
SINGLE_LINK_NAMES = {"base_left": "base_mount", "yaw_left_link": "yaw_link",
                     "pitch_left_link": "pitch_link"}

_BASE_ATTACH_MASS = 1e-9
_BASE_ATTACH_EDGE_M = 1e-3  # equivalent-box edge length target

TRIPLE_SPAN_Y = 2 * YAW_ANCHOR_Y                     # +-y of the triple rig's outer pair (see RIGS)

CANON_ANCHOR = np.array([0.0, -YAW_ANCHOR_Y, 0.0])   # canonical column's yaw axis in the CNC frame
CANON_FWD    = np.array([-1.0, 0.0, 0.0])            # canonical column's lens direction at qpos=0
CANON_DOWN   = np.array([0.0, 0.0, -1.0])            # canonical column's image-down direction


# ── small SE(3) glue (rot_with_zaxis / mat_to_rpy / capsule_transform now in robot_builder) ──
def split_pitch_mesh():
    """Split pitch_left.obj into its bracket and its D436 camera, as two OBJ files.

    MuJoCo binds one material per geom, so the camera can only be shaded apart from the
    motors if it is its own mesh. Rejected the alternative of shading the whole pitch link
    as camera: that link is mostly the pitch bracket and its motor housing.

    Classification is by CONNECTED COMPONENT against D436_BOX, not by a coordinate cut: the
    bracket wraps behind the camera and shares its z band, so any single-plane split takes
    bracket faces with it. The camera is a distinct solid upstream, so components are exactly
    the part boundary.

    Idempotent and mtime-guarded — re-exporting pitch_left.obj regenerates both outputs, so
    the derived files cannot go stale against their source.
    """
    import trimesh
    src = MESH_DIR / "pitch_left.obj"
    outs = [MESH_DIR / VISUAL_MESH["pitch_left_link"], MESH_DIR / EXTRA_VISUAL["pitch_left_link"][0]]
    if all(o.exists() and o.stat().st_mtime >= src.stat().st_mtime for o in outs):
        return

    lo, hi = (b * 1000.0 for b in D436_BOX)   # OBJs are in millimetres (MESH_SCALE)
    parts = trimesh.load(src, process=False).split(only_watertight=False)
    groups = [[], []]
    for p in parts:
        groups[bool((p.bounds[0] >= lo).all() and (p.bounds[1] <= hi).all())].append(p)
    assert all(groups), f"D436_BOX selected {len(groups[1])}/{len(parts)} components of {src.name}"
    for out, group in zip(outs, groups):
        trimesh.util.concatenate(group).export(out)
        print(f"✓ wrote {out}")


def T_of(R, p) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(p, float); return T

def origin_of(T) -> Origin:
    return Origin(list(T[:3, 3]), mat_to_rpy(T[:3, :3]))

def _lim(lo, hi):
    return JointLimit(lo, hi, effort=5.0, velocity=12.0)   # effort unused (no actuators emitted)


# ── rig configuration ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Mount:
    """Where one camera module sits, in the global CNC frame.

    tag:  name infix ('left'/'right'/'cam0'/...); None emits the un-suffixed
          SINGLE_LINK_NAMES and is only legal on a one-module rig.
    xyz:  where this module's YAW AXIS lands (the canonical anchor CANON_ANCHOR
          is mapped onto it).
    rpy:  rotation applied to the canonical column — EITHER an extrinsic-XYZ
          [r,p,y] triple OR a 3x3 rotation matrix, the same dual form
          robot_builder.Origin accepts, so a matrix from rot_with_zaxis / CF.RZ180
          can be dropped in without hand-deriving angles. Identity looks BACKWARD
          (-X); RZ180 looks FORWARD (+X). A roll/pitch component cants the whole
          module: the yaw shaft tilts with it and the optical site frame follows.
    """
    tag: str | None
    xyz: Sequence[float] = (0.0, 0.0, 0.0)
    rpy: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Rig:
    """One emitted MJCF file: model name, output filename, and its N mounts."""
    name: str
    mounts: tuple[Mount, ...]
    out: pathlib.Path = field(default=None)
    yaw_range: tuple[float, float] = YAW_RANGE
    pitch_range: tuple[float, float] = PITCH_RANGE
    damping: float = JOINT_DAMPING

    def __post_init__(self):
        object.__setattr__(self, "out", self.out or HERE / f"{self.name}.xml")


def ring(n, radius, z=0.0, start_deg=0.0, tag=lambda i: f"cam{i}", face_outward=True):
    """N mounts evenly spaced on a circle of the given radius, each facing radially.

    Sugar over Mount: returns plain Mounts, so a ring rig is config like any other.
    Angle 0 = +X; a module facing outward at angle a needs rpy yaw = a + pi, since
    the canonical column looks -X.
    """
    turn = np.pi if face_outward else 0.0
    angles = np.deg2rad(start_deg) + 2 * np.pi * np.arange(n) / n
    return tuple(Mount(tag(i), (radius * np.cos(a), radius * np.sin(a), z), (0.0, 0.0, a + turn))
                 for i, a in enumerate(angles))


RIGS = (
    # Dual: the canonical (-y, backward-looking) column plus its forward-looking
    # 180z partner. RZ180 is passed as a MATRIX so the emitted numbers stay bit-exact
    # against the historical mirror_180z path (an rpy=(0,0,pi) round-trip would inject
    # ~1e-16 noise into the inertias).
    Rig("head_camera_dual", (
        Mount("left",  (0.0,  YAW_ANCHOR_Y, 0.0), RZ180),
        Mount("right", (0.0, -YAW_ANCHOR_Y, 0.0)),
    )),
    # Single: the same forward-looking module, centered on the plate.
    Rig("head_camera_single", (
        Mount(None, (0.0, 0.0, 0.0), RZ180),
    )),
    # Triple: an outward-looking pair pushed out to make room for a centered
    # module between them. Both outer modules face FORWARD (RZ180) and the center
    # one faces BACKWARD (identity), so the parked rig covers front and rear
    # without relying on yaw travel; the +-270 deg sweep still lets any module
    # reach any azimuth. Spacing is set by that sweep, not the parked footprint:
    # a module claims a cylinder of radius 0.0577 m about its own yaw axis
    # (measured off the compiled dual rig) and neighbours must sit >= 0.1154 m
    # apart. TRIPLE_SPAN_Y keeps the dual's proven 0.130 m pitch between adjacent
    # modules, which is why the outer pair doubles its offset instead of staying
    # at YAW_ANCHOR_Y (0.065 would put the center module inside both neighbours'
    # sweeps).
    Rig("head_camera_triple", (
        Mount("left",   (0.0,  TRIPLE_SPAN_Y, 0.0), RZ180),
        Mount("center", (0.0,  0.0,           0.0)),
        Mount("right",  (0.0, -TRIPLE_SPAN_Y, 0.0), RZ180),
    )),
)


def mount_rot(m: Mount) -> np.ndarray:
    """Mount rotation as a matrix, accepting a 3x3 matrix or an [r,p,y] triple."""
    arr = np.asarray(m.rpy, dtype=float)
    return arr if arr.shape == (3, 3) else _rpy_to_transform(list(arr), [0, 0, 0])[:3, :3]


def mount_transform(m: Mount) -> np.ndarray:
    """Rigid CNC->world map W: rotate the canonical column, land its anchor on xyz."""
    return T_of(mount_rot(m), m.xyz) @ T_of(np.eye(3), -CANON_ANCHOR)


def build_column(mount: Mount, rig: Rig):
    """Return (links, joints) for the one camera module placed by `mount`.

    Every module is the same physical column (the CNC-source parts of
    cam_fusion_info '_left', the OBJ meshes, COLLISION_L, all living at y=-0.065 and
    looking -X); the mount supplies the rigid CNC->world map W that places it. Local
    geometry is W-invariant (see the module docstring's MOUNT PLACEMENT note), so W
    only reaches the root fixed joint's origin and the physics.

    Naming (corrected 2026-06-17): the +y column (GREEN) is the robot's LEFT and at
    qpos=0 looks FORWARD (+X); the -y column (BLUE) is the RIGHT and looks BACKWARD (-X).
    A mount with tag=None emits the un-suffixed SINGLE_LINK_NAMES."""
    W  = mount_transform(mount)                  # CNC->world placement (Decision 3, generalized)
    Rw = W[:3, :3]
    tag = mount.tag

    # Canonical body frames at the CNC-source side (y=-0.065, the un-rotated column);
    # every module's frames are W @ these, so each link's LOCAL inertia/geom is
    # byte-identical across modules and the placement lives only in the base-fixed
    # joint ("mirror at the limb root").
    T_yaw_C   = T_of(rot_with_zaxis(YAW_AXIS_W),     (0.0, -YAW_ANCHOR_Y, YAW_Z))
    T_pitch_C = T_of(rot_with_zaxis(PITCH_AXIS_W_L), (0.0, -YAW_ANCHOR_Y, PITCH_Z))
    T_yaw   = W @ T_yaw_C
    T_pitch = W @ T_pitch_C
    T_base  = T_yaw                               # fixed mount co-located with yaw frame
    Tbody = {"base_left": T_base, "yaw_left_link": T_yaw, "pitch_left_link": T_pitch}

    # physics: parse the CNC-source parts, then carry them to this mount's placement
    # (COM by the full affine W, inertia by its rotation). Reduces to CF.mirror_180z
    # exactly when W is the 180z rotation.
    phys = {}
    for lk, info in CF.LEFT_INFO.items():
        m, com_g, I_g = CF.parse_fusion(info)
        com_g = (W @ np.append(np.asarray(com_g, float), 1.0))[:3]
        I_g = Rw @ np.asarray(I_g, float) @ Rw.T
        phys[lk] = (m, com_g, I_g)

    def to_local(T, com_g, I_g):
        Rg2l = T[:3, :3].T                        # base -> body-local rotation
        com_l = Rg2l @ (com_g - T[:3, 3])
        I_l = Rg2l @ I_g @ Rg2l.T
        return com_l, I_l

    # CNC-source key -> emitted link name; joint/site names follow the same rule
    rn = SINGLE_LINK_NAMES if tag is None else {lk: lk.replace("left", tag) for lk in _CNC_LINKS}
    jn = ("base_mount_fixed", "yaw", "pitch") if tag is None else \
         (f"base_{tag}_fixed", f"yaw_{tag}", f"pitch_{tag}")
    site_name = (lambda nm: nm) if tag is None else (lambda nm: f"{tag}_{nm}")

    links, joints = [], []

    for lk in _CNC_LINKS:
        T = Tbody[lk]; Tinv = np.linalg.inv(T)
        m, com_g, I_g = phys[lk]
        com_l, I_l = to_local(T, com_g, I_g)

        # visual mesh: pin CNC mesh back to its true world place (W) inside body frame
        vis = Geom(Mesh(VISUAL_MESH[lk], scale=MESH_SCALE), origin=origin_of(Tinv @ W),
                   material=LINK_MATERIAL[lk], is_visual=True, is_collision=False)
        geoms = [vis]
        if lk in EXTRA_VISUAL:   # same body frame, own material (the D436 shell)
            mesh_file, mat = EXTRA_VISUAL[lk]
            geoms.append(Geom(Mesh(mesh_file, scale=MESH_SCALE), origin=origin_of(Tinv @ W),
                              material=mat, is_visual=True, is_collision=False))
        # collision capsules
        for r, p1, p2 in COLLISION_L[lk]:
            Tcw, L = capsule_transform(p1, p2)
            geoms.append(Geom(Capsule(r, L), origin=origin_of(Tinv @ W @ Tcw),
                              is_visual=False, is_collision=True,
                              group=3, contype=1, conaffinity=1))
        link = Link(name=rn[lk], geoms=geoms, mass=m,
                    inertial=Inertial(m, com_l.tolist(), I_l))
        # camera sites on the pitch link — OpenCV/ROS optical frame:
        #   +Z = optical axis (out the lens front), +X = image right, +Y = image down.
        # At qpos=0 the canonical column looks -X with up = world +Z; both directions
        # are carried through this mount's rotation, so a canted mount tilts the optical
        # frame with it. Image-right = down x forward. Two sites per column, placed by W:
        #   {col}_cam_front_center = the D436 FRONT-GLASS centre (user's Fusion reading)
        #   {col}_rgb              = the true RGB optical centre = glass + (glass->leftIR,
        #                            datasheet) + (leftIR->RGB, measured); lands on the RGB lens
        # both share the optical orientation (RGB ~parallel to depth, 0.16 deg ignored).
        if lk == "pitch_left_link":
            fwd  = Rw @ CANON_FWD
            down = Rw @ CANON_DOWN
            R_site_world = np.column_stack([np.cross(down, fwd), down, fwd])  # +X = down x +Z
            site_rpy = mat_to_rpy(T[:3, :3].T @ R_site_world)                 # optical frame in pitch-local

            def _add_site(nm, p_canon):                        # p_canon in canonical (-y) CNC coords
                p_world = (W @ np.append(p_canon, 1.0))[:3]    # W places it at this mount
                p_local = (Tinv @ np.append(p_world, 1.0))[:3]
                link.sites.append(Site(name=site_name(nm), origin=Origin(list(p_local), site_rpy)))

            _add_site("cam_front_center", GLASS_CENTER_L)
            _add_site("rgb", GLASS_CENTER_L + RGB_FROM_GLASS_CANON)
        links.append(link)

    # joints (all origins relative to PARENT body frame)
    joints.append(Joint(name=jn[0], parent="base", child=rn["base_left"],
                        type="fixed", origin=origin_of(T_base)))   # base frame is identity
    joints.append(Joint(name=jn[1], parent=rn["base_left"], child=rn["yaw_left_link"],
                        type="hinge", origin=origin_of(np.linalg.inv(T_base) @ T_yaw),
                        axis=[0, 0, 1], limit=_lim(*rig.yaw_range), damping=rig.damping))
    joints.append(Joint(name=jn[2], parent=rn["yaw_left_link"], child=rn["pitch_left_link"],
                        type="hinge", origin=origin_of(np.linalg.inv(T_yaw) @ T_pitch),
                        axis=[0, 0, 1], limit=_lim(*rig.pitch_range), damping=rig.damping))
    return links, joints


def write_rig(rig: Rig):
    """Assemble the rig's modules under a massless root and emit its MJCF file."""
    tags = [m.tag for m in rig.mounts]
    # Robot.add_link is keyed by name and silently OVERWRITES, so duplicate tags would
    # emit a quietly truncated model instead of failing.
    assert len(set(tags)) == len(tags), f"{rig.name}: duplicate mount tags {tags}"
    assert None not in tags or len(tags) == 1, f"{rig.name}: untagged mount needs a 1-module rig"

    robot = Robot(rig.name, materials=CAM_MATERIALS)
    # Massless module root (the attach anchor on base_link). mass and diaginertia
    # must both be > 0 for MuJoCo, but picking them independently as 1e-9 gave
    # I/mass = 1e-9/1e-9 = 1 m^2 -- i.e. an equivalent-box edge of ~1 m (I/mass ~
    # edge^2/6), making the "Inertia" viewer overlay draw a room-size red box on
    # a body that's physically a point. Derive diaginertia from mass via a fixed
    # tiny reference edge so the equivalent box stays visually negligible.
    robot.add_link(Link(name="base", geoms=[], mass=_BASE_ATTACH_MASS,
                        inertial=Inertial(_BASE_ATTACH_MASS, [0, 0, 0],
                                          np.eye(3) * (_BASE_ATTACH_MASS * _BASE_ATTACH_EDGE_M**2 / 6))))
    for mount in rig.mounts:
        links, joints = build_column(mount, rig)
        for l in links: robot.add_link(l)
        for j in joints: robot.add_joint(j)

    xml = robot.to_mjcf_string(add_freejoint=False, add_light_camera=False,
                               imu_site_name=None, actuators=None, sensors=None,
                               contact_body_names=None)
    # builder strips mesh file= to basename, but meshes live in meshes/links/
    xml = re.sub(r'file="([^"/]+\.(?:obj|stl|dae|ply))"', r'file="links/\1"', xml, flags=re.IGNORECASE)
    rig.out.write_text(xml)
    print(f"✓ wrote {rig.out}")


def main():
    split_pitch_mesh()
    for rig in RIGS:
        write_rig(rig)


if __name__ == "__main__":
    main()
