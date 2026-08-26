"""
URDF Helper Utilities

This module provides helper functions and classes for creating URDF files programmatically.
It simplifies the process of defining robot links with visual, collision, and inertial properties.

Key Components:
    - GeometryData: Dataclass for specifying geometry properties with validation
    - link_helper: Function to create complete URDF links from geometry specifications
    - Material definitions: Pre-defined materials (silver, blue, black, purple)

Features:
    - Automatic inertia computation from geometry
    - Support for box, cylinder, and mesh geometries
    - Validation of geometry parameters
    - Diagnostic output comparing visual and collision properties
"""

from typing import Optional, Union, List
import six
import xml.etree.ElementTree as ET
import copy
import inspect
import sys
import collections.abc
import numpy as np
import os
from odio_urdf import *
from dataclasses import dataclass
from urdf_util import (
    cylinder_inertia, sphere_inertia, box_inertia, mesh_inertia,
    rpy_to_transform, parallel_axis_term, rotate_inertia
)
import trimesh

def capsules_from_box(box_size):
    """
    Compute capsule parameters that best approximate a box collision shape.

    Strategy: r = sx/2 (maximum radius touching the shortest-axis faces),
    n = ceil(sy/sx) (minimum count to span the full sy-extent; capsules may
    overlap, which is fine — same-body geoms form a union in MuJoCo).

    Args:
        box_size: array-like of 3 positive floats (full dimensions, any order)

    Returns dict:
        radius      (float)   sx / 2
        cyl_len     (float)   sz - sx  (URDF cylinder length; MuJoCo adds caps
                              of radius r, giving total capsule length = sz)
        centers     (n, 3)   capsule centres in box-local frame
        axis_idx    (int)    box-local axis (0/1/2) capsules are aligned along
        arrange_idx (int)    box-local axis (0/1/2) capsules are arranged along
        n           (int)    ceil(sy / sx)
    """
    sizes = np.array(box_size, dtype=float)
    longest_idx, second_idx, shortest_idx = np.argsort(sizes)[::-1]
    sx, sy, sz = sizes[shortest_idx], sizes[second_idx], sizes[longest_idx]

    r       = sx / 2
    cyl_len = sz - sx                           # total = cyl_len + 2r = sz
    n       = max(1, int(np.ceil(sy / sx)))     # enough to span sy with r=sx/2

    if n == 1:
        centres_1d = np.array([0.0])
    else:
        centres_1d = np.linspace(-(sy / 2 - r), +(sy / 2 - r), n)

    centers = np.zeros((n, 3))
    centers[:, second_idx] = centres_1d

    return dict(
        radius=r, cyl_len=cyl_len, centers=centers,
        axis_idx=int(longest_idx), arrange_idx=int(second_idx), n=n,
    )


# RPY to align a cylinder (default axis=Z) with each world axis (in local frame)
_CAPSULE_AXIS_RPY = {
    0: [0.0,  np.pi / 2, 0.0],   # Z → X
    1: [-np.pi / 2, 0.0, 0.0],   # Z → Y
    2: [0.0,  0.0,       0.0],   # Z → Z (no rotation)
}


def _rpy_compose(rpy1, rpy2):
    """Compose two URDF RPY rotations: R = R(rpy1) @ R(rpy2)."""
    R1 = rpy_to_transform(rpy1, [0, 0, 0])[:3, :3]
    R2 = rpy_to_transform(rpy2, [0, 0, 0])[:3, :3]
    R  = R1 @ R2
    # Extract extrinsic XYZ (= URDF RPY) from rotation matrix
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:                       # gimbal lock: fix yaw=0
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw  = 0.0
    return [roll, pitch, yaw]


def geom_to_vis_coll(origin, geometry, material_name):
    """Helper function to create visual and collision elements"""
    return [
        Visual(origin, geometry, Material(name=material_name)),
        Collision(origin, geometry),
    ]

# Define materials that can be reused (Positional args first)
silver = Material(Color(rgba=[0.75, 0.75, 0.75, 1]), name="silver")
blue = Material(Color(rgba=[0, 0, 0.8, 1]), name="blue")
black = Material(Color(rgba=[0, 0, 0, 1]), name="black")
purple = Material(Color(rgba=[0.8, 0, 0.8, 1]), name="purple")
light_grey = Material(Color(rgba=[0.4, 0.4, 0.4, 1]), name="light_grey")
dark_grey = Material(Color(rgba=[0.3, 0.3, 0.3, 1]), name="dark_grey")


@dataclass
class GeometryData:
    """
    Data class for defining robot link geometry properties.
    """
    origin: List # [x, y, z] position of geometry relative to link frame
    rpy: List # [roll, pitch, yaw] orientation in radians
    type: str  # 'box', 'cylinder', 'mesh', etc.
    size: Optional[List] = None #  [x, y, z] dimensions for box type
    radius: Optional[float] = None # Radius for cylinder type
    length: Optional[float] = None # Length/height for cylinder type
    mesh_filename: Optional[str] = None  # Path to mesh file (relative to URDF location)
    mesh_scale: Optional[List] = None    # [x, y, z] scale factors for mesh
    mass: float = 0  # Mass in kg
    inertia: Optional[Union[str, np.ndarray]] = None # Either 'infer' to calculate from geometry, or 3x3 numpy array
    inertia_origin: Optional[List] = None # [x, y, z] COM position relative to link frame
    enable_visual: bool = True # Whether to include visual geometry
    enable_collision: bool = True # Whether to include collision geometry
    material: str = "silver" # Material name for visual appearance
    use_capsule_collision: bool = False  # Replace box collision with capsules (box type only)

    def __post_init__(self):
        """Validate geometry data after initialization."""
        # Validate mass
        if self.mass < 0:
            raise ValueError(f"Mass must be non-negative, got {self.mass}")

        if self.use_capsule_collision and self.type != "box":
            raise ValueError(f"use_capsule_collision only supported for type='box', got '{self.type}'")

        # Validate geometry type and required fields
        if self.type == "box":
            if self.size is None or len(self.size) != 3:
                raise ValueError(f"Box geometry requires size=[x,y,z], got {self.size}")
            if any(s <= 0 for s in self.size):
                raise ValueError(f"Box dimensions must be positive, got {self.size}")

        elif self.type == "cylinder":
            if self.radius is None or self.radius <= 0:
                raise ValueError(f"Cylinder requires positive radius, got {self.radius}")
            if self.length is None or self.length <= 0:
                raise ValueError(f"Cylinder requires positive length, got {self.length}")

        elif self.type == "mesh":
            if self.mesh_filename is None:
                raise ValueError("Mesh geometry requires mesh_filename")

        # Validate origin and rpy
        if len(self.origin) != 3:
            raise ValueError(f"Origin must be [x,y,z], got {self.origin}")
    _MESH_CACHE = {}

def _get_geometry_inertia_data(g, script_dir):
    """
    Extract inertia data for a single geometry.

    Returns:
        tuple: (mass, com_position, local_inertia_about_com, rotation_matrix)
        - mass: geometry mass
        - com_position: COM position in link frame
        - local_inertia_about_com: inertia tensor about geometry's own COM, in local frame
        - rotation_matrix: rotation from geometry's local frame to link frame
    """
    # Get rotation matrix from rpy
    transform = rpy_to_transform(g.rpy, [0, 0, 0])  # only rotation, no translation
    rotation = transform[:3, :3]
    
    # ... code continues exactly as before for box and cylinder ...

    if g.type == "box":
        # Box COM is at g.origin, inertia is about box center
        com_pos = np.array(g.origin)
        x, y, z = g.size
        ixx = (1/12) * g.mass * (y**2 + z**2)
        iyy = (1/12) * g.mass * (x**2 + z**2)
        izz = (1/12) * g.mass * (x**2 + y**2)
        local_inertia = np.diag([ixx, iyy, izz])
        # volume and volume-centroid for constant-density comparison
        vol = x * y * z
        vol_centroid = np.array(g.origin)
        # no surface area centroid for primitive
        area = None
        area_centroid = None
        return g.mass, com_pos, local_inertia, rotation, area, area_centroid, vol, vol_centroid

    elif g.type == "cylinder":
        # Cylinder COM is at g.origin, inertia is about cylinder center (z-axis along cylinder)
        com_pos = np.array(g.origin)
        r, h = g.radius, g.length
        ixx = (1/12) * g.mass * h**2 + (1/4) * g.mass * r**2
        iyy = ixx
        izz = (1/2) * g.mass * r**2
        local_inertia = np.diag([ixx, iyy, izz])
        # volume and centroid
        vol = np.pi * r**2 * h
        vol_centroid = np.array(g.origin)
        area = None
        area_centroid = None
        return g.mass, com_pos, local_inertia, rotation, area, area_centroid, vol, vol_centroid

    elif g.type == "mesh":
        mesh_path = f"{script_dir}/{g.mesh_filename}"
        mesh_result = mesh_inertia(mesh_path, g.mass, about='com')

        # If user provided inertia_origin, use that as COM position
        # Otherwise, compute from mesh COM transformed to link frame
        if g.inertia_origin is not None:
            com_pos = np.array(g.inertia_origin)
        else:
            # Mesh COM in link frame = g.origin + R @ mesh_com
            mesh_com = mesh_result['center_of_mass']
            com_pos = np.array(g.origin) + rotation @ mesh_com

        # Use provided inertia or inferred from mesh
        if isinstance(g.inertia, np.ndarray):
            local_inertia = g.inertia
        else:
            local_inertia = mesh_result['inertia_tensor']

        # Use trimesh to get area and volume and their centroids
        try:
            if mesh_path not in _MESH_CACHE:
                _MESH_CACHE[mesh_path] = trimesh.load_mesh(mesh_path)
            # Need to copy because we apply scale in-place
            m = _MESH_CACHE[mesh_path].copy()
            # apply scale if provided
            if g.mesh_scale is not None:
                m.apply_scale(np.array(g.mesh_scale))
            # trimesh reports center_mass (volume centroid) and volume
            vol = float(m.volume) if getattr(m, 'volume', None) is not None else None
            vol_centroid_local = np.array(m.center_mass) if vol is not None else None
            # surface area centroid: area-weighted average of triangle centers
            try:
                tri_centers = m.triangles_center
                tri_areas = m.area_faces
                if tri_centers is not None and tri_areas is not None and len(tri_areas) > 0:
                    area_centroid_local = np.average(tri_centers, weights=tri_areas, axis=0)
                    area = float(m.area)
                else:
                    area_centroid_local = None
                    area = None
            except Exception:
                area_centroid_local = None
                area = None

            # transform local centroids into link frame
            vol_centroid = (np.array(g.origin) + rotation @ vol_centroid_local) if vol_centroid_local is not None else None
            area_centroid = (np.array(g.origin) + rotation @ area_centroid_local) if area_centroid_local is not None else None
        except Exception:
            vol = None
            vol_centroid = None
            area = None
            area_centroid = None

        return g.mass, com_pos, local_inertia, rotation, area, area_centroid, vol, vol_centroid

    else:
        raise ValueError(f"Unsupported geometry type: {g.type}")


def _compute_combined_inertia(geom_data_list):
    """
    Compute combined COM and inertia about combined COM from multiple geometries.

    Args:
        geom_data_list: List of (mass, com_position, local_inertia, rotation) tuples

    Returns:
        tuple: (total_mass, combined_com, combined_inertia_about_com)
    """
    if not geom_data_list:
        return 0.0, np.zeros(3), np.zeros((3, 3)), None, None

    # Phase 1: Compute combined COM (mass-weighted average)
    # geom_data_list entries have shape: (mass, com_pos, local_inertia, rotation, area, area_centroid, vol, vol_centroid)
    total_mass = sum(item[0] for item in geom_data_list)
    if total_mass < 1e-12:
        return 0.0, np.zeros(3), np.zeros((3, 3)), None, None

    combined_com = sum(item[0] * item[1] for item in geom_data_list) / total_mass

    # Compute combined area-weighted centroid (surface constant-mass assumption) if any areas available
    area_sum = 0.0
    area_centroid_sum = np.zeros(3)
    for item in geom_data_list:
        area = item[4]
        a_cent = item[5]
        if area is not None and a_cent is not None:
            area_sum += area
            area_centroid_sum += area * a_cent
    combined_area_centroid = (area_centroid_sum / area_sum) if area_sum > 1e-12 else None

    # Compute combined volume-weighted centroid (constant-density assumption) if any volumes available
    vol_sum = 0.0
    vol_centroid_sum = np.zeros(3)
    for item in geom_data_list:
        vol = item[6]
        v_cent = item[7]
        if vol is not None and v_cent is not None:
            vol_sum += vol
            vol_centroid_sum += vol * v_cent
    combined_vol_centroid = (vol_centroid_sum / vol_sum) if vol_sum > 1e-12 else None

    # Phase 2: Compute combined inertia about combined COM
    combined_inertia = np.zeros((3, 3))
    for item in geom_data_list:
        mass = item[0]
        com_pos = item[1]
        local_inertia = item[2]
        rotation = item[3]
        # Rotate inertia from local frame to link frame
        I_rotated = rotate_inertia(rotation, local_inertia)

        # Displacement from this geometry's COM to combined COM
        displacement = com_pos - combined_com

        # Contribution: rotated inertia + parallel axis term
        I_contrib = I_rotated + parallel_axis_term(mass, displacement)
        combined_inertia += I_contrib

    return total_mass, combined_com, combined_inertia, combined_area_centroid, combined_vol_centroid


def link_helper(link_name: str, geom_list: List[GeometryData], inertia_origin: Optional[List] = None, verbose=True):
    """
    Create a URDF link from a list of geometry specifications.

    This function constructs a complete URDF link with visual, collision, and inertial
    properties derived from the provided geometry data. It automatically computes
    combined inertial properties (mass, COM, inertia tensor) from all geometries.

    The inertia tensor is computed ABOUT THE COMBINED COM, which is the correct
    reference point for URDF and MJCF formats.

    Args:
        link_name: Name of the link
        geom_list: List of GeometryData objects defining link geometries
        inertia_origin: [x, y, z] COM override (if None, auto-computed from geometries)
        verbose: Print diagnostic info

    Returns:
        Link object ready to be added to a Robot

    Raises:
        ValueError: If geometry types are invalid or required parameters are missing

    Note:
        - Visual geometries are used for URDF inertial properties
        - Collision geometries are shown for comparison in verbose output
        - Inertia is computed about the combined COM (not link origin)
    """
    if not link_name:
        raise ValueError("link_name cannot be empty")

    if not geom_list:
        raise ValueError(f"geom_list for link '{link_name}' cannot be empty")

    link = Link(link_name)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Geometry object constructors for URDF
    geom_objs = {
        "box": lambda g: Geometry(Box(size=g.size)),
        "cylinder": lambda g: Geometry(Cylinder(radius=g.radius, length=g.length)),
        "mesh": lambda g: Geometry(Mesh(filename=g.mesh_filename, scale=g.mesh_scale) if g.mesh_scale else Mesh(filename=g.mesh_filename)),
    }

    # Collect inertia data for visual and collision geometries
    visual_geom_data = []
    collision_geom_data = []

    for g in geom_list:
        if g.type not in geom_objs:
            raise ValueError(f"unimplemented geometry type: {g.type}")

        geom_obj = geom_objs[g.type](g)
        origin = Origin(g.origin, g.rpy)

        # Add visual element
        if g.enable_visual:
            link(Visual(origin, geom_obj, Material(name=g.material)))
            if isinstance(g.inertia, np.ndarray) or g.inertia == "infer":
                geom_data = _get_geometry_inertia_data(g, script_dir)
                visual_geom_data.append(geom_data)

        # Add collision element(s)
        if g.enable_collision:
            if g.use_capsule_collision:
                # Expand box into N cylinders approximating the box collision shape.
                caps    = capsules_from_box(g.size)
                R_box   = rpy_to_transform(g.rpy, [0, 0, 0])[:3, :3]
                cap_rpy = _rpy_compose(g.rpy, _CAPSULE_AXIS_RPY[caps['axis_idx']])
                for c in caps['centers']:
                    cap_origin = (np.array(g.origin) + R_box @ c).tolist()
                    link(Collision(
                        Origin(cap_origin, cap_rpy),
                        Geometry(Cylinder(radius=caps['radius'], length=caps['cyl_len'])),
                        # Pass a special name that urdf_to_mujoco.py intercepts.
                        # This generates a standard cylinder in the URDF (for compatibility)
                        # but cues the MJCF converter to replace it with a capsule geometry.
                        name="capsule_collision"
                    ))
                # Inertia comparison still uses the equivalent box volume
                if isinstance(g.inertia, np.ndarray) or g.inertia == "infer":
                    collision_geom_data.append(_get_geometry_inertia_data(g, script_dir))
            else:
                link(Collision(origin, geom_obj))
                if isinstance(g.inertia, np.ndarray) or g.inertia == "infer":
                    geom_data = _get_geometry_inertia_data(g, script_dir)
                    collision_geom_data.append(geom_data)

    # Compute combined inertia for visual geometries (used in URDF)
    total_mass_visual, combined_com_visual, inertia_tensor_visual, combined_area_visual, combined_vol_visual = _compute_combined_inertia(visual_geom_data)

    # Compute combined inertia for collision geometries (for comparison only)
    total_mass_collision, combined_com_collision, inertia_tensor_collision, combined_area_collision, combined_vol_collision = _compute_combined_inertia(collision_geom_data)

    # Determine final COM:
    # 1. Use explicit override if provided
    # 2. Otherwise use computed combined COM
    if inertia_origin is not None:
        com = inertia_origin
    else:
        com = combined_com_visual.tolist() if total_mass_visual > 1e-12 else [0, 0, 0]

    # Compute COM under a constant-density assumption using visual mesh volumes (if available)
    if combined_vol_visual is not None:
        com_const_density = combined_vol_visual.tolist()
    elif combined_area_visual is not None:
        com_const_density = combined_area_visual.tolist()
    else:
        com_const_density = combined_com_visual.tolist() if total_mass_visual > 1e-12 else [0, 0, 0]

    # Build URDF Inertial element
    inertial_args = [
        Mass(value=total_mass_visual),
        Inertia(
            ixx=inertia_tensor_visual[0, 0],
            iyy=inertia_tensor_visual[1, 1],
            izz=inertia_tensor_visual[2, 2],
            ixy=inertia_tensor_visual[0, 1],
            ixz=inertia_tensor_visual[0, 2],
            iyz=inertia_tensor_visual[1, 2]
        )
    ]
    inertial_args.insert(0, Origin(xyz=com, rpy=[0, 0, 0]))
    link(Inertial(*inertial_args))

    # Print diagnostic info
    if verbose:
        def get_inertia_components(tensor):
            return [tensor[0,0], tensor[1,1], tensor[2,2], tensor[0,1], tensor[0,2], tensor[1,2]]

        vis_inertia = get_inertia_components(inertia_tensor_visual)
        col_inertia = get_inertia_components(inertia_tensor_collision)

        print(f"\n{link_name} (COM: [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}])")
        print(f"  Constant Density COM: [{com_const_density[0]:.4f}, {com_const_density[1]:.4f}, {com_const_density[2]:.4f}]")
        print("  {:>10} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12} | {:>12}".format(
            "mass", "ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        print("V {:10.4f} | {:12.6f} | {:12.6f} | {:12.6f} | {:12.6f} | {:12.6f} | {:12.6f}".format(
            total_mass_visual,
            vis_inertia[0], vis_inertia[1], vis_inertia[2], vis_inertia[3], vis_inertia[4], vis_inertia[5]
        ))

        # Show collision row or note if collision disabled
        if total_mass_collision < 1e-12:
            print(f"C \033[93mNo collision mesh provided\033[0m")
        else:
            print("C {:10.4f} | {:12.6f} | {:12.6f} | {:12.6f} | {:12.6f} | {:12.6f} | {:12.6f}".format(
                total_mass_collision,
                col_inertia[0], col_inertia[1], col_inertia[2], col_inertia[3], col_inertia[4], col_inertia[5]
            ))
            # Calculate percent differences
            if total_mass_visual > 1e-12:
                mass_pct = 100.0 * (total_mass_collision - total_mass_visual) / total_mass_visual
                inertia_pcts = [
                    100.0 * (col - vis) / abs(vis) if abs(vis) > 1e-6 else 0.0
                    for vis, col in zip(vis_inertia, col_inertia)
                ]
                print(f"%{mass_pct:10.2f}  | {inertia_pcts[0]:12.2f} | {inertia_pcts[1]:12.2f} | "
                      f"{inertia_pcts[2]:12.2f} | {inertia_pcts[3]:12.2f} | {inertia_pcts[4]:12.2f} | "
                      f"{inertia_pcts[5]:12.2f}")

    return link
