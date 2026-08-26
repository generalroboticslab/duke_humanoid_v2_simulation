"""Robot Model: dataclass-based robot description with URDF and MuJoCo XML export.

Defines a robot as a tree of Body objects connected by Joints, with graph
features (tendons, equality constraints, contact excludes) stored at the
model level. Exports to both URDF and MuJoCo XML from the same model.

MuJoCo/Warp GPU optimizations:
- All collision cylinders exported as capsules (analytical contact, faster broadphase)
- Contact excludes for adjacent bodies
- Explicit condim on collision defaults
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np
import os
import xml.etree.ElementTree as ET


# ============================================================================
# Data Model
# ============================================================================

@dataclass
class Geom:
    """Geometry primitive attached to a body.

    Args:
        name: Unique identifier for this geom.
        type: box | cylinder | capsule | sphere | mesh
        origin: [x,y,z] position in parent body frame.
        rpy: [roll,pitch,yaw] orientation in radians.
        role: "visual" | "collision" | "both" — determines which XML elements are generated.
        mass: Mass in kg (used for inertia computation).
        inertia: 3x3 array, "infer" to compute from geometry, or None to skip.
        use_capsule_collision: If True and type="box", expand into N capsules at export.
    """
    name: str
    type: str
    origin: list
    rpy: list
    # Shape params (type-dependent)
    size: Optional[list] = None           # box [x,y,z] full extents
    radius: Optional[float] = None        # cylinder/capsule/sphere
    length: Optional[float] = None        # cylinder/capsule
    mesh_filename: Optional[str] = None
    mesh_scale: Optional[list] = None
    # Inertia
    mass: float = 0.0
    inertia: Union[np.ndarray, str, None] = None  # 3x3, "infer", or None
    inertia_origin: Optional[list] = None          # COM override for mesh
    # Role & appearance
    role: str = "both"
    material: Optional[str] = None
    use_capsule_collision: bool = False
    # MuJoCo-specific overrides (None = use class defaults)
    contype: Optional[int] = None
    conaffinity: Optional[int] = None
    condim: Optional[int] = None
    friction: Optional[list] = None
    group: Optional[int] = None

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError(f"Mass must be non-negative, got {self.mass}")
        if self.use_capsule_collision and self.type != "box":
            raise ValueError(f"use_capsule_collision only for type='box', got '{self.type}'")
        if self.type == "box":
            if self.size is None or len(self.size) != 3:
                raise ValueError(f"Box requires size=[x,y,z], got {self.size}")
        elif self.type in ("cylinder", "capsule"):
            if self.radius is None or self.radius <= 0:
                raise ValueError(f"{self.type} requires positive radius")
            if self.length is None or self.length <= 0:
                raise ValueError(f"{self.type} requires positive length")
        elif self.type == "mesh":
            if self.mesh_filename is None:
                raise ValueError("Mesh requires mesh_filename")
        if len(self.origin) != 3:
            raise ValueError(f"origin must be [x,y,z], got {self.origin}")
        if self.role not in ("visual", "collision", "both"):
            raise ValueError(f"role must be visual/collision/both, got '{self.role}'")


@dataclass
class Joint:
    """Joint connecting a child body to its parent.

    Args:
        type: revolute | continuous | prismatic | fixed | free
        axis: Rotation/translation axis in child frame.
        origin/rpy: Pose of child frame relative to parent.
    """
    name: str
    type: str
    axis: list = field(default_factory=lambda: [0, 0, 1])
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    effort: float = 0.0
    velocity: float = 12.0
    lower: Optional[float] = None
    upper: Optional[float] = None
    damping: float = 0.0
    armature: Optional[float] = None
    frictionloss: Optional[float] = None


@dataclass
class Body:
    """Rigid body in the kinematic tree.

    Children are (Joint, Body) pairs. Inertial properties are computed
    by finalize() from visual-role geoms.
    """
    name: str
    geoms: list = field(default_factory=list)
    children: list = field(default_factory=list)  # list[tuple[Joint, Body]]
    # Computed by finalize()
    mass: Optional[float] = None
    com: Optional[np.ndarray] = None
    inertia_tensor: Optional[np.ndarray] = None


# --- Graph features (model-level) ---

@dataclass
class Actuator:
    name: str
    joint: str
    type: str = "motor"
    kp: Optional[float] = None
    kv: Optional[float] = None
    gear: float = 1.0
    forcerange: Optional[list] = None
    ctrlrange: Optional[list] = None


@dataclass
class Sensor:
    name: str
    type: str                          # gyro | accelerometer | velocimeter | subtreeangmom | ...
    site: Optional[str] = None
    body: Optional[str] = None


@dataclass
class Site:
    name: str
    body: str                          # body name reference
    origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: float = 0.01


@dataclass
class Tendon:
    """Fixed tendon coupling multiple joints."""
    name: str
    joints: list = field(default_factory=list)  # list[tuple[str, float]]  (joint_name, coef)
    limited: bool = False
    range: Optional[list] = None


@dataclass
class EqualityConstraint:
    type: str                          # weld | connect | joint
    name: Optional[str] = None
    body1: Optional[str] = None
    body2: Optional[str] = None
    joint1: Optional[str] = None
    joint2: Optional[str] = None
    polycoef: Optional[list] = None
    anchor: Optional[list] = None
    solref: Optional[list] = None
    solimp: Optional[list] = None


@dataclass
class ContactExclude:
    body1: str
    body2: str


@dataclass
class DefaultClass:
    """MuJoCo default class for geom/joint properties."""
    name: str
    geom: dict = field(default_factory=dict)
    joint: dict = field(default_factory=dict)
    children: list = field(default_factory=list)  # list[DefaultClass]


@dataclass
class RobotModel:
    """Complete robot description.

    Body tree rooted at root_body. Graph features reference tree elements by name.
    """
    name: str
    root_body: Body
    materials: dict = field(default_factory=dict)       # name -> [r,g,b,a]
    actuators: list = field(default_factory=list)
    sensors: list = field(default_factory=list)
    sites: list = field(default_factory=list)
    tendons: list = field(default_factory=list)
    equality_constraints: list = field(default_factory=list)
    contact_excludes: list = field(default_factory=list)
    defaults: list = field(default_factory=list)         # list[DefaultClass]
    compiler: dict = field(default_factory=lambda: {
        "angle": "radian", "meshdir": "meshes", "autolimits": "true"
    })
    floating_base: bool = True
    freejoint_name: str = "floating_base_joint"


# ============================================================================
# Math Utilities
# ============================================================================

def euler_to_quat(rpy):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)."""
    roll, pitch, yaw = rpy
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def format_array(arr, precision=6):
    """Format array as space-separated string."""
    return " ".join(f"{x:.{precision}g}" for x in arr)


def rpy_to_rotation(rpy):
    """RPY (extrinsic XYZ) to 3x3 rotation matrix."""
    rpy = np.asarray(rpy, dtype=float)
    sr, sp, sy = np.sin(rpy)
    cr, cp, cy = np.cos(rpy)
    return np.array([
        [cp*cy, cy*sr*sp - cr*sy, sr*sy + cr*cy*sp],
        [cp*sy, cr*cy + sr*sp*sy, cr*sp*sy - cy*sr],
        [-sp,   cp*sr,            cr*cp]
    ])


def _rpy_compose(rpy1, rpy2):
    """Compose two RPY rotations: R = R(rpy1) @ R(rpy2), return result as RPY."""
    R = rpy_to_rotation(rpy1) @ rpy_to_rotation(rpy2)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw  = 0.0
    return [roll, pitch, yaw]


# ============================================================================
# Capsule Expansion
# ============================================================================

# RPY to align capsule (default Z-axis) with each world axis
_CAPSULE_AXIS_RPY = {
    0: [0.0, np.pi / 2, 0.0],   # Z → X
    1: [-np.pi / 2, 0.0, 0.0],  # Z → Y
    2: [0.0, 0.0, 0.0],         # Z → Z
}


def capsules_from_box(box_size):
    """Compute capsule parameters approximating a box collision shape.

    r = sx/2 (shortest dimension), n = ceil(sy/sx), capsules along longest axis.

    Returns dict with: radius, cyl_len, centers (n,3), axis_idx, arrange_idx, n.
    """
    sizes = np.array(box_size, dtype=float)
    longest_idx, second_idx, shortest_idx = np.argsort(sizes)[::-1]
    sx, sy, sz = sizes[shortest_idx], sizes[second_idx], sizes[longest_idx]

    r       = sx / 2
    cyl_len = sz - sx  # total capsule length = cyl_len + 2r = sz
    n       = max(1, int(np.ceil(sy / sx)))

    if n == 1:
        centres_1d = np.array([0.0])
    else:
        centres_1d = np.linspace(-(sy / 2 - r), +(sy / 2 - r), n)

    centers = np.zeros((n, 3))
    centers[:, second_idx] = centres_1d

    return dict(radius=r, cyl_len=cyl_len, centers=centers,
                axis_idx=int(longest_idx), arrange_idx=int(second_idx), n=n)


def expand_capsule_geoms(geom):
    """Expand a box Geom with use_capsule_collision into a list of capsule Geoms.

    Returns list of capsule Geom objects with role="collision".
    """
    assert geom.type == "box" and geom.use_capsule_collision
    caps = capsules_from_box(geom.size)
    R_box = rpy_to_rotation(geom.rpy)
    cap_rpy = _rpy_compose(geom.rpy, _CAPSULE_AXIS_RPY[caps['axis_idx']])
    result = []
    for i, c in enumerate(caps['centers']):
        cap_origin = (np.array(geom.origin) + R_box @ c).tolist()
        result.append(Geom(
            name=f"{geom.name}_{i}", type="capsule",
            origin=cap_origin, rpy=cap_rpy,
            radius=caps['radius'], length=caps['cyl_len'],
            mass=geom.mass / caps['n'], inertia="infer",
            role="collision", material=geom.material,
        ))
    return result


# ============================================================================
# Inertia Computation
# ============================================================================

_MESH_CACHE = {}


def _get_geom_inertia(g, script_dir):
    """Compute inertia data for a single Geom.

    Returns (mass, com_pos, local_inertia_about_com, rotation_3x3).
    """
    R = rpy_to_rotation(g.rpy)

    if g.type == "box":
        com = np.array(g.origin)
        x, y, z = g.size
        local_I = np.diag([
            (1/12) * g.mass * (y**2 + z**2),
            (1/12) * g.mass * (x**2 + z**2),
            (1/12) * g.mass * (x**2 + y**2),
        ])
        return g.mass, com, local_I, R

    elif g.type in ("cylinder", "capsule"):
        com = np.array(g.origin)
        r, h = g.radius, g.length
        ixx = (1/12) * g.mass * h**2 + (1/4) * g.mass * r**2
        izz = (1/2) * g.mass * r**2
        local_I = np.diag([ixx, ixx, izz])
        return g.mass, com, local_I, R

    elif g.type == "mesh":
        from urdf_util import mesh_inertia
        mesh_path = os.path.join(script_dir, g.mesh_filename)
        mesh_result = mesh_inertia(mesh_path, g.mass, about='com')
        if g.inertia_origin is not None:
            com = np.array(g.inertia_origin)
        else:
            com = np.array(g.origin) + R @ mesh_result['center_of_mass']
        if isinstance(g.inertia, np.ndarray):
            local_I = g.inertia
        else:
            local_I = mesh_result['inertia_tensor']
        return g.mass, com, local_I, R

    else:
        raise ValueError(f"Unsupported geometry type: {g.type}")


def compute_body_inertia(body, script_dir):
    """Compute mass, COM, and inertia tensor for a body from its visual geoms.

    Uses parallel axis theorem to combine multiple geometry contributions.
    Modifies body.mass, body.com, body.inertia_tensor in place.
    """
    # Collect inertia from visual-role geoms (matching old behavior)
    geom_data = []
    for g in body.geoms:
        if g.role in ("visual", "both"):
            if isinstance(g.inertia, np.ndarray) or g.inertia == "infer":
                geom_data.append(_get_geom_inertia(g, script_dir))

    if not geom_data:
        body.mass = 0.0
        body.com = np.zeros(3)
        body.inertia_tensor = np.zeros((3, 3))
        return

    total_mass = sum(m for m, _, _, _ in geom_data)
    if total_mass < 1e-12:
        body.mass = 0.0
        body.com = np.zeros(3)
        body.inertia_tensor = np.zeros((3, 3))
        return

    combined_com = sum(m * c for m, c, _, _ in geom_data) / total_mass

    # Combine inertia about combined COM via parallel axis theorem
    combined_I = np.zeros((3, 3))
    for mass, com, local_I, R in geom_data:
        I_rotated = R @ local_I @ R.T
        d = com - combined_com
        I_parallel = mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        combined_I += I_rotated + I_parallel

    body.mass = total_mass
    body.com = combined_com
    body.inertia_tensor = combined_I


def finalize(model, script_dir=None, verbose=True):
    """Compute inertia for all bodies in the tree.

    Call after building the tree and before exporting.
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))

    def _walk(body):
        compute_body_inertia(body, script_dir)
        if verbose:
            _print_body_inertia(body)
        for _, child in body.children:
            _walk(child)

    _walk(model.root_body)


def _print_body_inertia(body):
    """Print diagnostic inertia info for a body."""
    if body.mass is None or body.mass < 1e-12:
        return
    I = body.inertia_tensor
    c = body.com
    print(f"\n{body.name} (COM: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}])")
    print(f"  mass={body.mass:.4f}  I_diag=[{I[0,0]:.6f}, {I[1,1]:.6f}, {I[2,2]:.6f}]")


# ============================================================================
# Tree Utilities
# ============================================================================

def find_body(root, name):
    """Find a body by name in the tree. Returns None if not found."""
    if root.name == name:
        return root
    for _, child in root.children:
        found = find_body(child, name)
        if found is not None:
            return found
    return None


def collect_joints(root):
    """Collect all non-fixed joints in tree traversal order."""
    joints = []
    for joint, child in root.children:
        if joint.type not in ("fixed",):
            joints.append(joint)
        joints.extend(collect_joints(child))
    return joints


def _collect_meshes(root, seen=None):
    """Collect unique (mesh_name, filename, scale) from tree."""
    if seen is None:
        seen = {}
    for g in root.geoms:
        if g.type == "mesh" and g.mesh_filename:
            key = g.mesh_filename.replace("meshes/", "").replace(".stl", "")
            if key not in seen:
                seen[key] = (key, g.mesh_filename, g.mesh_scale)
    for _, child in root.children:
        _collect_meshes(child, seen)
    return sorted(seen.values())


def _collect_body_excludes(root):
    """Generate contact excludes for parent-child body pairs (can't self-collide)."""
    excludes = []
    for _, child in root.children:
        excludes.append(ContactExclude(root.name, child.name))
        excludes.extend(_collect_body_excludes(child))
    return excludes


# ============================================================================
# MuJoCo XML Export
# ============================================================================

def to_mujoco(model, include_actuators=False):
    """Generate MuJoCo XML string from a RobotModel.

    All collision cylinders are exported as capsules for GPU-optimized contact.
    Contact excludes for adjacent bodies are automatically added.
    """
    root = ET.Element("mujoco")
    root.set("model", model.name)

    # Compiler
    compiler = ET.SubElement(root, "compiler")
    for k, v in model.compiler.items():
        compiler.set(k, v)

    # Defaults
    if model.defaults:
        default_section = ET.SubElement(root, "default")
        _emit_defaults(default_section, model.defaults)

    # Assets
    asset = ET.SubElement(root, "asset")
    for mat_name, rgba in model.materials.items():
        mat = ET.SubElement(asset, "material")
        mat.set("name", mat_name)
        mat.set("rgba", format_array(rgba))
    for mesh_name, filename, scale in _collect_meshes(model.root_body):
        mesh = ET.SubElement(asset, "mesh")
        mesh.set("name", mesh_name)
        mesh.set("file", filename.replace("meshes/", ""))
        if scale:
            mesh.set("scale", format_array(scale))

    # Worldbody
    worldbody = ET.SubElement(root, "worldbody")

    # Build site lookup: body_name -> [Site, ...]
    site_lookup = {}
    for site in model.sites:
        site_lookup.setdefault(site.body, []).append(site)

    root_body_elem = _emit_body_mujoco(model.root_body, None, site_lookup, is_root=True)
    root_body_elem.set("childclass", model.name)

    # Light and camera on root body
    light = ET.Element("light")
    light.set("pos", "0 0 2")
    light.set("mode", "trackcom")
    root_body_elem.insert(0, light)

    camera = ET.Element("camera")
    camera.set("name", "tracking")
    camera.set("pos", "1.5 -1.5 1")
    camera.set("xyaxes", "0.707 0.707 0 -0.3 0.3 0.9")
    camera.set("mode", "trackcom")
    root_body_elem.insert(1, camera)

    # Freejoint
    if model.floating_base:
        fj = ET.Element("freejoint")
        fj.set("name", model.freejoint_name)
        # Insert after inertial
        insert_idx = 2
        for i, child in enumerate(root_body_elem):
            if child.tag == "inertial":
                insert_idx = i + 1
                break
        root_body_elem.insert(insert_idx, fj)

    worldbody.append(root_body_elem)

    # Contact excludes (auto-generated + user-specified)
    all_excludes = _collect_body_excludes(model.root_body) + model.contact_excludes
    if all_excludes:
        contact = ET.SubElement(root, "contact")
        for exc in all_excludes:
            ex = ET.SubElement(contact, "exclude")
            ex.set("body1", exc.body1)
            ex.set("body2", exc.body2)

    # Tendons
    if model.tendons:
        tendon_section = ET.SubElement(root, "tendon")
        for t in model.tendons:
            fixed = ET.SubElement(tendon_section, "fixed")
            fixed.set("name", t.name)
            if t.limited:
                fixed.set("limited", "true")
            if t.range:
                fixed.set("range", format_array(t.range))
            for jname, coef in t.joints:
                jt = ET.SubElement(fixed, "joint")
                jt.set("joint", jname)
                jt.set("coef", str(coef))

    # Equality constraints
    if model.equality_constraints:
        eq_section = ET.SubElement(root, "equality")
        for eq in model.equality_constraints:
            elem = ET.SubElement(eq_section, eq.type)
            if eq.name: elem.set("name", eq.name)
            if eq.body1: elem.set("body1", eq.body1)
            if eq.body2: elem.set("body2", eq.body2)
            if eq.joint1: elem.set("joint1", eq.joint1)
            if eq.joint2: elem.set("joint2", eq.joint2)
            if eq.polycoef: elem.set("polycoef", format_array(eq.polycoef))
            if eq.anchor: elem.set("anchor", format_array(eq.anchor))
            if eq.solref: elem.set("solref", format_array(eq.solref))
            if eq.solimp: elem.set("solimp", format_array(eq.solimp))

    # Actuators
    if model.actuators:
        act_section = ET.SubElement(root, "actuator")
        for a in model.actuators:
            elem = ET.SubElement(act_section, a.type)
            elem.set("name", a.name)
            elem.set("joint", a.joint)
            if a.gear != 1.0: elem.set("gear", str(a.gear))
            if a.kp is not None: elem.set("kp", str(a.kp))
            if a.kv is not None: elem.set("kv", str(a.kv))
            if a.forcerange: elem.set("forcerange", format_array(a.forcerange))
            if a.ctrlrange: elem.set("ctrlrange", format_array(a.ctrlrange))

    # Sensors
    if model.sensors:
        sensor_section = ET.SubElement(root, "sensor")
        for s in model.sensors:
            elem = ET.SubElement(sensor_section, s.type)
            elem.set("name", s.name)
            if s.site: elem.set("site", s.site)
            if s.body: elem.set("body", s.body)

    _indent_xml(root)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def _emit_defaults(parent, defaults_list):
    """Recursively emit <default> elements."""
    for dc in defaults_list:
        elem = ET.SubElement(parent, "default")
        elem.set("class", dc.name)
        if dc.geom:
            geom = ET.SubElement(elem, "geom")
            for k, v in dc.geom.items():
                geom.set(k, v)
        if dc.joint:
            joint = ET.SubElement(elem, "joint")
            for k, v in dc.joint.items():
                joint.set(k, v)
        if dc.children:
            _emit_defaults(elem, dc.children)


def _emit_body_mujoco(body, joint, site_lookup, is_root=False):
    """Recursively convert Body to MuJoCo <body> element."""
    elem = ET.Element("body")
    elem.set("name", body.name)

    # Position/orientation from joint
    if joint is not None:
        pos = joint.origin
        if not np.allclose(pos, [0, 0, 0], atol=1e-9):
            elem.set("pos", format_array(pos))
        quat = euler_to_quat(joint.rpy)
        if not np.allclose(quat, [1, 0, 0, 0], atol=1e-9):
            elem.set("quat", format_array(quat))

    # Inertial
    if body.mass is not None and body.mass > 1e-12:
        inertial = ET.SubElement(elem, "inertial")
        inertial.set("pos", format_array(body.com.tolist()))
        inertial.set("mass", str(body.mass))
        I = body.inertia_tensor
        ixx, iyy, izz = I[0, 0], I[1, 1], I[2, 2]
        ixy, ixz, iyz = I[0, 1], I[0, 2], I[1, 2]
        if np.allclose([ixy, ixz, iyz], [0, 0, 0], atol=1e-9):
            inertial.set("diaginertia", format_array([ixx, iyy, izz]))
        else:
            inertial.set("fullinertia", format_array([ixx, iyy, izz, ixy, ixz, iyz]))

    # Collision geoms first (before visual, matches old ordering)
    for g in body.geoms:
        if g.role in ("collision", "both"):
            if g.use_capsule_collision:
                for cap_geom in expand_capsule_geoms(g):
                    _emit_geom_mujoco(elem, cap_geom, is_collision=True)
            elif g.type == "cylinder":
                # GPU optimization: all collision cylinders → capsules
                _emit_geom_mujoco(elem, g, is_collision=True, force_capsule=True)
            else:
                _emit_geom_mujoco(elem, g, is_collision=True)

    # Visual geoms
    for g in body.geoms:
        if g.role in ("visual", "both"):
            _emit_geom_mujoco(elem, g, is_collision=False)

    # Joint (if not root)
    if joint is not None and joint.type not in ("fixed",):
        j = ET.SubElement(elem, "joint")
        j.set("name", joint.name)
        if joint.type == "revolute" or joint.type == "continuous":
            j.set("type", "hinge")
        elif joint.type == "prismatic":
            j.set("type", "slide")
        j.set("axis", format_array(joint.axis))
        if joint.type == "revolute" and joint.lower is not None and joint.upper is not None:
            j.set("range", f"{joint.lower} {joint.upper}")
        if joint.damping > 0:
            j.set("damping", str(joint.damping))
        if joint.armature is not None:
            j.set("armature", str(joint.armature))
        if joint.frictionloss is not None:
            j.set("frictionloss", str(joint.frictionloss))

    # Sites belonging to this body
    for site in site_lookup.get(body.name, []):
        s = ET.SubElement(elem, "site")
        s.set("name", site.name)
        s.set("pos", format_array(site.origin))
        s.set("size", str(site.size))
        quat = euler_to_quat(site.rpy)
        if not np.allclose(quat, [1, 0, 0, 0], atol=1e-9):
            s.set("quat", format_array(quat))

    # Children
    for child_joint, child_body in body.children:
        child_elem = _emit_body_mujoco(child_body, child_joint, site_lookup)
        elem.append(child_elem)

    return elem


def _emit_geom_mujoco(parent, g, is_collision, force_capsule=False):
    """Emit a single <geom> element for MuJoCo."""
    geom = ET.SubElement(parent, "geom")
    geom.set("class", "collision" if is_collision else "visual")
    if not is_collision and g.name:
        geom.set("name", g.name)

    # Position
    pos = g.origin
    if not np.allclose(pos, [0, 0, 0], atol=1e-9):
        geom.set("pos", format_array(pos))

    # Orientation
    quat = euler_to_quat(g.rpy)
    if not np.allclose(quat, [1, 0, 0, 0], atol=1e-9):
        geom.set("quat", format_array(quat))

    # Type + shape
    if g.type == "box" and not g.use_capsule_collision:
        geom.set("type", "box")
        geom.set("size", format_array([s / 2.0 for s in g.size]))
    elif g.type in ("cylinder", "capsule"):
        if force_capsule or g.type == "capsule":
            geom.set("type", "capsule")
        else:
            geom.set("type", "cylinder")
        geom.set("size", f"{g.radius} {g.length / 2.0}")
    elif g.type == "sphere":
        geom.set("type", "sphere")
        geom.set("size", str(g.radius))
    elif g.type == "mesh":
        if not is_collision:
            # Visual mesh — type comes from default class
            pass
        else:
            geom.set("type", "mesh")
        mesh_name = g.mesh_filename.replace("meshes/", "").replace(".stl", "")
        geom.set("mesh", mesh_name)

    # Material (visual only)
    if not is_collision and g.material:
        geom.set("material", g.material)

    # MuJoCo overrides
    if g.contype is not None: geom.set("contype", str(g.contype))
    if g.conaffinity is not None: geom.set("conaffinity", str(g.conaffinity))
    if g.condim is not None: geom.set("condim", str(g.condim))
    if g.friction: geom.set("friction", format_array(g.friction))
    if g.group is not None: geom.set("group", str(g.group))

    # Name for collision geoms
    if is_collision:
        geom.set("name", g.name)


# ============================================================================
# URDF Export
# ============================================================================

def to_urdf(model):
    """Generate URDF XML string from a RobotModel.

    Graph features (tendons, equality) are silently ignored.
    """
    robot = ET.Element("robot")
    robot.set("name", model.name)

    # Materials
    for mat_name, rgba in model.materials.items():
        mat = ET.SubElement(robot, "material")
        mat.set("name", mat_name)
        color = ET.SubElement(mat, "color")
        color.set("rgba", format_array(rgba))

    # Emit root link
    _emit_link_urdf(robot, model.root_body)

    # Emit children recursively
    _emit_children_urdf(robot, model.root_body)

    _indent_xml(robot)
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")


def _emit_link_urdf(robot, body):
    """Emit a <link> element for URDF."""
    link = ET.SubElement(robot, "link")
    link.set("name", body.name)

    # Visual geoms
    for g in body.geoms:
        if g.role in ("visual", "both"):
            visual = ET.SubElement(link, "visual")
            _emit_origin_urdf(visual, g.origin, g.rpy)
            _emit_geometry_urdf(visual, g)
            if g.material:
                mat = ET.SubElement(visual, "material")
                mat.set("name", g.material)

    # Collision geoms
    for g in body.geoms:
        if g.role in ("collision", "both"):
            if g.use_capsule_collision:
                # Expand box into cylinders with capsule_collision marker
                caps = capsules_from_box(g.size)
                R_box = rpy_to_rotation(g.rpy)
                cap_rpy = _rpy_compose(g.rpy, _CAPSULE_AXIS_RPY[caps['axis_idx']])
                for c in caps['centers']:
                    cap_origin = (np.array(g.origin) + R_box @ c).tolist()
                    coll = ET.SubElement(link, "collision")
                    coll.set("name", "capsule_collision")
                    _emit_origin_urdf(coll, cap_origin, cap_rpy)
                    geom_elem = ET.SubElement(coll, "geometry")
                    cyl = ET.SubElement(geom_elem, "cylinder")
                    cyl.set("radius", str(caps['radius']))
                    cyl.set("length", str(caps['cyl_len']))
            else:
                coll = ET.SubElement(link, "collision")
                _emit_origin_urdf(coll, g.origin, g.rpy)
                _emit_geometry_urdf(coll, g)

    # Inertial
    if body.mass is not None and body.mass > 1e-12:
        inertial = ET.SubElement(link, "inertial")
        _emit_origin_urdf(inertial, body.com.tolist(), [0, 0, 0])
        mass_elem = ET.SubElement(inertial, "mass")
        mass_elem.set("value", str(body.mass))
        I = body.inertia_tensor
        inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", str(I[0, 0]))
        inertia.set("iyy", str(I[1, 1]))
        inertia.set("izz", str(I[2, 2]))
        inertia.set("ixy", str(I[0, 1]))
        inertia.set("ixz", str(I[0, 2]))
        inertia.set("iyz", str(I[1, 2]))


def _emit_children_urdf(robot, parent_body):
    """Recursively emit child links and joints."""
    for joint, child in parent_body.children:
        # Joint
        j = ET.SubElement(robot, "joint")
        j.set("name", joint.name)
        j.set("type", joint.type)
        parent_elem = ET.SubElement(j, "parent")
        parent_elem.set("link", parent_body.name)
        child_elem = ET.SubElement(j, "child")
        child_elem.set("link", child.name)
        _emit_origin_urdf(j, joint.origin, joint.rpy)
        axis = ET.SubElement(j, "axis")
        axis.set("xyz", format_array(joint.axis))
        if joint.type == "revolute":
            limit = ET.SubElement(j, "limit")
            limit.set("effort", str(joint.effort))
            limit.set("velocity", str(joint.velocity))
            if joint.lower is not None:
                limit.set("lower", str(joint.lower))
            if joint.upper is not None:
                limit.set("upper", str(joint.upper))

        # Child link
        _emit_link_urdf(robot, child)

        # Recurse
        _emit_children_urdf(robot, child)


def _emit_origin_urdf(parent, xyz, rpy):
    """Emit <origin xyz="..." rpy="..."/> element."""
    origin = ET.SubElement(parent, "origin")
    origin.set("xyz", format_array(xyz))
    origin.set("rpy", format_array(rpy))


def _emit_geometry_urdf(parent, g):
    """Emit <geometry> element for URDF."""
    geom_elem = ET.SubElement(parent, "geometry")
    if g.type == "box":
        box = ET.SubElement(geom_elem, "box")
        box.set("size", format_array(g.size))
    elif g.type in ("cylinder", "capsule"):
        cyl = ET.SubElement(geom_elem, "cylinder")
        cyl.set("radius", str(g.radius))
        cyl.set("length", str(g.length))
    elif g.type == "sphere":
        sph = ET.SubElement(geom_elem, "sphere")
        sph.set("radius", str(g.radius))
    elif g.type == "mesh":
        mesh = ET.SubElement(geom_elem, "mesh")
        mesh.set("filename", g.mesh_filename)
        if g.mesh_scale:
            mesh.set("scale", format_array(g.mesh_scale))


# ============================================================================
# XML Helpers
# ============================================================================

def _indent_xml(elem, level=0):
    """Pretty-print XML with 2-space indentation."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent
