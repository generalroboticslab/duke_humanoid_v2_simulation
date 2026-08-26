"""Robot Builder Helper Functions - Human-Friendly Pipeline

Provides intuitive, readable wrappers for assembling robots:
- Auto-import physics from Fusion360 (via fusion_info.py)
- Section-by-section link and joint creation
- Symmetry helpers for mirroring left limbs to right
- Standardized material definitions and joint parameters

✨ KEY BENEFITS:
- DRY Principle: Define left side once, mirror right automatically.
- Readability: Linear, modular code instead of giant dataclass blocks.
- Safety: Standardized revolute joint defaults and limit handling.

Example:
    from builder_helpers import link_from_fusion, simple_joint, deg

    # Create link with auto-imported physics
    ankle = link_from_fusion("ankle_1",
        visual_mesh="meshes/ankle_1.obj",
        collision=Geom(Cylinder(0.044, 0.0345),
                      Origin([-0.017, 0, -0.0255], [0, deg(90), 0]))
    )

    # Build joint with clear parameters
    hip_joint = simple_joint("left_hip_1_joint",
        parent="waist", child="hip_1_L",
        xyz=[0, 0.06, -0.08], rpy=[deg(75), 0, 0],
        lower=deg(-105), upper=deg(105)
    )
"""

import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import replace
from robot_builder import (
    Robot, Link, Joint, JointLimit, Geom, Origin, Inertial,
    Box, Cylinder, Capsule, Sphere, Mesh, Material,
    _rpy_to_transform
)
from fusion_info import PHYSICS_DB
from functools import lru_cache

# ── Constants ─────────────────────────────────────────────────

deg = np.deg2rad  # Shorthand: deg(45) instead of np.deg2rad(45)

# Standard materials
STANDARD_MATERIALS = [
    Material("silver", rgba=[0.75, 0.75, 0.75, 1]),
    Material("blue", rgba=[0, 0, 0.8, 1]),
    Material("black", rgba=[0, 0, 0, 1]),
    Material("purple", rgba=[0.8, 0, 0.8, 1]),
    Material("light_grey", rgba=[0.4, 0.4, 0.4, 1]),
    Material("dark_grey", rgba=[0.3, 0.3, 0.3, 1]),
    Material("white", rgba=[1, 1, 1, 1]),
]

# ── Link Creation Helpers ─────────────────────────────────────

def link_from_fusion(
    fusion_key: str,
    visual_mesh: str,
    collision: Union[Geom, List[Geom], None] = None,
    visual_material: str = "dark_grey",
    name_override: Optional[str] = None,
    mesh_scale: List[float] = [0.001, 0.001, 0.001],
) -> Link:
    """Create link with auto-imported Fusion360 physics

    Args:
        fusion_key: Key in PHYSICS_DB (e.g., "ankle_1", "hip_2")
        visual_mesh: Path to visual mesh file
        collision: Single Geom or list of collision geoms
        visual_material: Material name for visual mesh
        name_override: Override link name (default: fusion_key)
        mesh_scale: Scale factor for mesh (default: [0.001, 0.001, 0.001])

    Returns:
        Link with physics from Fusion360 and specified geometry

    Example:
        ankle = link_from_fusion("ankle_1",
            visual_mesh="meshes/ankle_1.obj",
            collision=Geom(Cylinder(0.044, 0.0345),
                          Origin([-0.017, 0, -0.0255], [0, deg(90), 0]),
                          is_collision=True)
        )
    """
    phys = PHYSICS_DB[fusion_key]

    # Ensure collision is a list
    if collision is None:
        collision_geoms = []
    elif isinstance(collision, list):
        collision_geoms = collision
    else:
        collision_geoms = [collision]

    # Build geom list: visual mesh + collision geoms
    geoms = [
        Geom(Mesh(visual_mesh, scale=mesh_scale),
             material=visual_material,
             is_visual=True, is_collision=False)
    ]
    geoms.extend(collision_geoms)

    return Link(
        name=name_override or fusion_key,
        geoms=geoms,
        mass=phys.mass,
        inertial=Inertial(phys.mass, phys.com.tolist(), phys.inertia)
    )


def simple_link(
    name: str,
    visual_mesh: str,
    collision: Union[Geom, List[Geom]],
    mass: float,
    com: List[float],
    inertia: np.ndarray,
    visual_material: str = "dark_grey",
    mesh_scale: List[float] = [0.001, 0.001, 0.001],
) -> Link:
    """Create link with manually specified physics (for parts not in Fusion DB)

    Args:
        name: Link name
        visual_mesh: Path to visual mesh file
        collision: Single Geom or list of collision geoms
        mass: Mass in kg
        com: Center of mass [x, y, z] in meters
        inertia: 3x3 inertia tensor at COM
        visual_material: Material name for visual mesh
        mesh_scale: Scale factor for mesh

    Returns:
        Link with specified physics and geometry
    """
    collision_geoms = collision if isinstance(collision, list) else [collision]

    geoms = [
        Geom(Mesh(visual_mesh, scale=mesh_scale),
             material=visual_material,
             is_visual=True, is_collision=False)
    ]
    geoms.extend(collision_geoms)

    return Link(
        name=name,
        geoms=geoms,
        mass=mass,
        inertial=Inertial(mass, com, inertia)
    )


# ── Joint Creation Helpers ────────────────────────────────────

def simple_joint(
    name: str,
    parent: str,
    child: str,
    xyz: List[float],
    rpy: List[float],
    axis: List[float] = [0, 0, 1],
    effort: float = 55.0,
    velocity: float = 12.0,
    lower: float = -np.pi,
    upper: float = np.pi,
    damping: float = 0.0,
    ref: float = 0.0,
) -> Joint:
    """Create revolute joint with common defaults

    Args:
        name: Joint name
        parent: Parent link name
        child: Child link name
        xyz: Position [x, y, z] of joint in parent frame
        rpy: Orientation [roll, pitch, yaw] of joint in parent frame
        axis: Joint axis in child frame (default: [0, 0, 1])
        effort: Torque limit in Nm (default: 55)
        velocity: Velocity limit in rad/s (default: 12)
        lower: Lower joint limit in radians (default: -π)
        upper: Upper joint limit in radians (default: π)
        damping: Joint damping (default: 0)
        ref: MuJoCo `ref` attribute — zero-position offset in radians. Shifts the
            joint's q=0 pose without changing body frames or inertia. Useful when the
            neutral pose differs from the URDF/XML geometric origin (default: 0)

    Returns:
        Joint configured as revolute with specified limits

    Example:
        hip_joint = simple_joint("left_hip_1_joint",
            parent="waist", child="hip_1_L",
            xyz=[0, 0.06, -0.08], rpy=[deg(75), 0, 0],
            lower=deg(-105), upper=deg(105)
        )
    """
    return Joint(
        name=name,
        parent=parent,
        child=child,
        type="hinge",
        origin=Origin(xyz, rpy),
        axis=axis,
        limit=JointLimit(lower, upper, effort, velocity),
        damping=damping,
        ref=ref,
    )


# ── Symmetry Helpers ──────────────────────────────────────────

def mirror_geom(geom: Geom, y_flip: bool = True) -> Geom:
    """Mirror a geom across Y-axis or keep identical

    Args:
        geom: Geom to mirror
        y_flip: If True, flip Y-coordinate and certain RPY angles

    Returns:
        Mirrored copy of geom
    """
    if not y_flip:
        return geom

    new_origin = Origin(
        xyz=[geom.origin.xyz[0], -geom.origin.xyz[1], geom.origin.xyz[2]],
        rpy=geom.origin.rpy if not y_flip else
            [geom.origin.rpy[0], -geom.origin.rpy[1], -geom.origin.rpy[2]]
    )

    return replace(geom, origin=new_origin)


def mirror_link(link: Link, name_map: Dict[str, str], y_flip: bool = True) -> Link:
    """Create mirrored copy of link for L/R symmetry

    Args:
        link: Link to mirror
        name_map: Dict mapping old names to new (e.g., {"_L": "_R", "left": "right"})
        y_flip: If True, mirror across Y-axis

    Returns:
        Mirrored link with reflected geometry and inertia

    Example:
        hip_R = mirror_link(hip_L, {"_L": "_R", "left": "right"})

    Note:
        For symmetric robots, inertial properties should be IDENTICAL (not mirrored)
        because the physical properties are measured in the local body frame.
        The parent joint orientation handles the mirroring, not the inertia itself.
    """
    # Mirror geoms
    new_geoms = [mirror_geom(g, y_flip) for g in link.geoms]

    # DO NOT mirror inertial! Fusion360 exports are in local body frame
    # The mirroring is handled by joint placement and orientation
    new_inertial = link.inertial

    # Update name
    new_name = link.name
    for old, new in name_map.items():
        new_name = new_name.replace(old, new)

    return Link(
        name=new_name,
        geoms=new_geoms,
        mass=link.mass,
        inertial=new_inertial
    )


def mirror_link_about_plane(link: Link, normal, point=(0.0, 0.0, 0.0)) -> Link:
    """Reflect a link's own geometry and physics about a plane in its local frame.

    Use for chiral hardware: a left/right pair that is one part and its mirror image, not
    one part placed at two poses. Applies to an already-built Link, so it composes with
    both `link_from_fusion` and `simple_link` without either needing a mirror argument.

    Args:
        link: Link to reflect (not modified).
        normal: Plane normal in the link's local frame; need not be unit length.
        point: Any point on the plane, link-local. Default = plane through the origin.

    Returns:
        New Link, same name, with mass unchanged and everything else reflected.

    Reflection about plane (n̂, p0) is the affine map x ↦ H x + t, with
    H = I - 2 n̂ n̂ᵀ and t = (I - H) p0. det H = -1, so H is not a rotation and cannot be
    a geom quat on its own. It factors exactly as H = (H S) S with S = diag(-1, 1, 1),
    where H S is a proper rotation — so the reflection is an ordinary geom pose plus a
    sign flip on the shape's local x:

        origin.xyz  ->  H p + t
        orientation ->  R' = H R S          (det R' = +1)
        Mesh only   ->  scale[0] negated

    One formula covers every shape: the mirrored point set is H(R V + p) + t, and writing
    it as R' V' + p' with R' = H R S needs S V' = V. Box/Cylinder/Capsule/Sphere are all
    mirror-symmetric about their own local x = 0 plane, so V' = V and nothing else changes.
    A Mesh is not, so V' = S V is realized by the negative x scale — MuJoCo re-winds the
    faces itself, verified vertex-exact against an offline trimesh reflection.

    Physics (mass properties are about the COM in the local frame, so t drops out of the
    inertia): mass unchanged, com -> H c + t, inertia -> H J Hᵀ. Checked against trimesh's
    own mass properties of the reflected solid to machine precision, for both an
    axis-aligned and a random offset plane.

    Assumptions:
        - Child joint frames are NOT touched. Reflecting a link whose child mount does not
          lie on the plane moves that mount, and the joint origin in the build script must
          then be hand-edited to match — the child part is generally not chiral, so there
          is no correct automatic choice of which child axis to flip.
        - Rejected: pre-reflecting the .obj offline. Slower to compile, duplicates megabytes
          of mesh per part, and needs a separate script.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    H = np.eye(3) - 2.0 * np.outer(n, n)
    t = (np.eye(3) - H) @ np.asarray(point, dtype=float)
    S = np.diag([-1.0, 1.0, 1.0])

    new_geoms = []
    for g in link.geoms:
        R = _rpy_to_transform(g.origin.rpy, [0, 0, 0])[:3, :3]
        R_mirrored = H @ R @ S
        assert np.isclose(np.linalg.det(R_mirrored), 1.0), \
            f"mirrored orientation of geom on '{link.name}' is not a rotation"
        shape = g.shape
        if isinstance(shape, Mesh):
            scale = list(shape.scale) if shape.scale else [1.0, 1.0, 1.0]
            scale[0] = -scale[0]
            shape = replace(shape, scale=scale)
        new_geoms.append(replace(
            g,
            shape=shape,
            origin=Origin(xyz=(H @ np.asarray(g.origin.xyz, dtype=float) + t).tolist(),
                          rpy=R_mirrored),
        ))

    new_sites = [replace(s, origin=Origin(
        xyz=(H @ np.asarray(s.origin.xyz, dtype=float) + t).tolist(),
        rpy=H @ _rpy_to_transform(s.origin.rpy, [0, 0, 0])[:3, :3] @ S,
    )) for s in link.sites]

    new_inertial = link.inertial
    if new_inertial is not None:
        new_inertial = replace(
            new_inertial,
            com=(H @ np.asarray(new_inertial.com, dtype=float) + t).tolist(),
            inertia=H @ new_inertial.inertia @ H.T,
        )

    return replace(link, geoms=new_geoms, sites=new_sites, inertial=new_inertial)


def mirror_joint(joint: Joint, name_map: Dict[str, str], xyz_flip_y: bool = True, rpy_adjust: Optional[List[float]] = None) -> Joint:
    """Create mirrored copy of joint for L/R symmetry

    Args:
        joint: Joint to mirror
        name_map: Dict mapping old names to new
        xyz_flip_y: If True, flip Y-coordinate of xyz
        rpy_adjust: Optional RPY adjustment for mirrored joint

    Returns:
        Mirrored joint

    Example:
        right_hip = mirror_joint(left_hip,
            {"left": "right", "_L": "_R"},
            rpy_adjust=[0, 0, 0]  # Adjust RPY as needed
        )
    """
    new_xyz = [joint.origin.xyz[0],
               -joint.origin.xyz[1] if xyz_flip_y else joint.origin.xyz[1],
               joint.origin.xyz[2]]

    new_rpy = joint.origin.rpy if rpy_adjust is None else rpy_adjust

    new_name = joint.name
    new_parent = joint.parent
    new_child = joint.child
    for old, new in name_map.items():
        new_name = new_name.replace(old, new)
        new_parent = new_parent.replace(old, new)
        new_child = new_child.replace(old, new)

    return Joint(
        name=new_name,
        parent=new_parent,
        child=new_child,
        type=joint.type,
        origin=Origin(new_xyz, new_rpy),
        axis=joint.axis,
        limit=joint.limit,
        damping=joint.damping,
        armature=joint.armature,
        friction=joint.friction,
        ref=joint.ref,  # Caller should negate if effective axis direction flips under mirroring
    )


# ── Performance Optimizations ─────────────────────────────────

# Cache mesh loading (avoids redundant trimesh.load() calls)
@lru_cache(maxsize=128)
def _cached_mesh_load(filepath: str):
    """Internal cache for mesh loading - used by robot_builder"""
    pass  # Implemented in robot_builder.py
