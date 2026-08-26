import xml.dom.minidom
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Tuple
import math
import numpy as np
import scipy.linalg

def _parse_array(arr: List[float]) -> str:
    return " ".join(f"{x:.6g}" for x in arr)

def _euler_to_quat(rpy: List[float]) -> List[float]:
    roll, pitch, yaw = rpy
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]

def _rpy_to_transform(rpy: List[float], pos: List[float]) -> np.ndarray:
    sr, sp, sy = np.sin(rpy)
    cr, cp, cy = np.cos(rpy)
    rot = np.array([
        [cp*cy, cy*sr*sp - cr*sy, sr*sy + cr*cy*sp],
        [cp*sy, cr*cy + sr*sp*sy, cr*sp*sy - cy*sr],
        [-sp,   cp*sr,             cr*cp]
    ])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

def _rpy_compose(rpy1: List[float], rpy2: List[float]) -> List[float]:
    R1 = _rpy_to_transform(rpy1, [0,0,0])[:3, :3]
    R2 = _rpy_to_transform(rpy2, [0,0,0])[:3, :3]
    R = R1 @ R2
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    cp = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw  = 0.0
    return [roll, pitch, yaw]

def mat_to_rpy(R: np.ndarray) -> List[float]:
    """Inverse of `_rpy_to_transform`: rotation matrix -> extrinsic-XYZ rpy
    (R = Rz(yaw) Ry(pitch) Rx(roll)). Same extraction inlined in `_rpy_compose`."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return [roll, pitch, yaw]


def rot_with_zaxis(zaxis) -> np.ndarray:
    """Right-handed rotation whose 3rd COLUMN (local +Z) is `zaxis` (roll arbitrary)."""
    z = np.asarray(zaxis, float); z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0.0, 1, 0])
    x = np.cross(helper, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def capsule_transform(p0, p1) -> Tuple[np.ndarray, float]:
    """Endpoints -> (SE(3) transform with local +Z along the axis, midpoint center),
    cylinder length. `fromto`/hemisphere-center semantics (the two endpoints are the
    capsule's two hemisphere centres). Roll about +Z is arbitrary (capsule is radially
    symmetric)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    T = np.eye(4)
    T[:3, :3] = rot_with_zaxis(d / L)
    T[:3, 3] = (p0 + p1) / 2.0
    return T, L


def capsule_fromto(p0, p1, radius: float, **geom_kwargs) -> "Geom":
    """`Geom(Capsule)` authored from two endpoints (MuJoCo `fromto` semantics), for arbitrary
    orientation. Converts endpoints -> center + axis-rpy via `capsule_transform`; `geom_kwargs`
    pass through to `Geom` (name/is_visual/is_collision/contype/conaffinity/group/friction/...).
    """
    T, L = capsule_transform(p0, p1)
    origin = Origin(T[:3, 3].tolist(), mat_to_rpy(T[:3, :3]))
    return Geom(Capsule(radius, L), origin=origin, **geom_kwargs)


_CAPSULE_AXIS_RPY = {
    0: [0.0, np.pi/2, 0.0],
    1: [-np.pi/2, 0.0, 0.0],
    2: [0.0, 0.0, 0.0],
}

_CIRCLE_FIT_GAUSS_X, _CIRCLE_FIT_GAUSS_W = np.polynomial.legendre.leggauss(24)

def _circle_union_area(centers_2d: List[Tuple[float, float]], radius: float) -> float:
    """Approximate union area of equal circles with fixed-order Gaussian integration.

    Assumes circles already lie inside the target rectangle, so no clipping is needed.
    This is used only to choose capsule grid spacing during asset export; fixed 24-point
    quadrature keeps small grids in the low-ms range while avoiding dense pixel brute force.
    """
    if radius <= 0.0 or not centers_2d:
        return 0.0

    xmin = min(cx - radius for cx, _ in centers_2d)
    xmax = max(cx + radius for cx, _ in centers_2d)
    x_samples = 0.5 * (xmax - xmin) * _CIRCLE_FIT_GAUSS_X + 0.5 * (xmax + xmin)

    total = 0.0
    r2 = radius * radius
    for x, weight in zip(x_samples, _CIRCLE_FIT_GAUSS_W):
        intervals = []
        for cx, cy in centers_2d:
            dx = x - cx
            rem = r2 - dx * dx
            if rem >= 0.0:
                dy = math.sqrt(rem)
                intervals.append((cy - dy, cy + dy))
        if not intervals:
            continue

        intervals.sort()
        lo, hi = intervals[0]
        length = 0.0
        for a, b in intervals[1:]:
            if a <= hi:
                hi = max(hi, b)
            else:
                length += hi - lo
                lo, hi = a, b
        total += weight * (length + hi - lo)

    return 0.5 * (xmax - xmin) * total

def _fit_circle_grid_no_overrun(sx: float, sy: float, n_rows: int, n_cols: int) -> Tuple[float, float, float]:
    """Choose grid spacing that maximizes circle union area inside a rectangle.

    Returns `(row_ratio, col_ratio, radius)` where ratios scale original cell-center
    offsets. Objective is maximum filled area with every circle fully contained in
    the `sx` by `sy` rectangle. This optimizes spacing, not radius padding; callers can
    apply overfill afterwards if they intentionally want boundary overrun.
    """
    row_cell_cs = -sx / 2 + (sx / n_rows) * (np.arange(n_rows) + 0.5)
    col_cell_cs = -sy / 2 + (sy / n_cols) * (np.arange(n_cols) + 0.5)

    if n_rows == 1 and n_cols == 1:
        return 1.0, 1.0, min(sx, sy) / 2.0

    def eval_area(ratios: np.ndarray) -> float:
        row_ratio = float(np.clip(ratios[0], 0.0, 1.0)) if n_rows > 1 else 1.0
        col_ratio = float(np.clip(ratios[1], 0.0, 1.0)) if n_cols > 1 else 1.0
        row_cs = row_ratio * row_cell_cs
        col_cs = col_ratio * col_cell_cs
        centers = [(float(x), float(y)) for x in row_cs for y in col_cs]
        radius = min(
            [sx / 2 - abs(x) for x, _ in centers] +
            [sy / 2 - abs(y) for _, y in centers]
        )
        return _circle_union_area(centers, radius)

    start = np.array([
        0.7 if n_rows > 1 else 1.0,
        0.7 if n_cols > 1 else 1.0,
    ], dtype=float)
    simplex = [
        start,
        np.clip(start + np.array([-0.08 if n_rows > 1 else 0.0, 0.0]), 0.0, 1.0),
        np.clip(start + np.array([0.0, -0.08 if n_cols > 1 else 0.0]), 0.0, 1.0),
    ]
    values = [eval_area(x) for x in simplex]

    for _ in range(35):
        order = np.argsort(values)[::-1]
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        best, second, worst = simplex
        centroid = (best + second) / 2.0
        reflected = np.clip(centroid + (centroid - worst), 0.0, 1.0)
        reflected_value = eval_area(reflected)

        if reflected_value > values[0]:
            expanded = np.clip(centroid + 2.0 * (reflected - centroid), 0.0, 1.0)
            expanded_value = eval_area(expanded)
            simplex[2] = expanded if expanded_value > reflected_value else reflected
            values[2] = max(expanded_value, reflected_value)
        elif reflected_value > values[1]:
            simplex[2] = reflected
            values[2] = reflected_value
        else:
            contracted = np.clip(centroid + 0.5 * (worst - centroid), 0.0, 1.0)
            contracted_value = eval_area(contracted)
            if contracted_value > values[2]:
                simplex[2] = contracted
                values[2] = contracted_value
            else:
                simplex[1] = np.clip(best + 0.5 * (simplex[1] - best), 0.0, 1.0)
                simplex[2] = np.clip(best + 0.5 * (simplex[2] - best), 0.0, 1.0)
                values[1] = eval_area(simplex[1])
                values[2] = eval_area(simplex[2])

        if max(np.linalg.norm(simplex[i] - simplex[0]) for i in (1, 2)) < 1e-4:
            break

    row_ratio, col_ratio = simplex[int(np.argmax(values))]
    if n_rows == 1:
        row_ratio = 1.0
    if n_cols == 1:
        col_ratio = 1.0

    row_cs = float(row_ratio) * row_cell_cs
    col_cs = float(col_ratio) * col_cell_cs
    centers = [(float(x), float(y)) for x in row_cs for y in col_cs]
    radius = min(
        [sx / 2 - abs(x) for x, _ in centers] +
        [sy / 2 - abs(y) for _, y in centers]
    )
    return float(row_ratio), float(col_ratio), float(radius)

def _capsules_from_box(box_size: List[float], grid_shape: Optional[Tuple[int, int]] = None,
                       overfill_ratio: float = 0.0,
                       distance_ratio: Optional[Union[float, Tuple[float, float]]] = None) -> Dict:
    if overfill_ratio < 0.0:
        raise ValueError("capsule overfill_ratio must be non-negative")
    if distance_ratio is not None:
        if isinstance(distance_ratio, tuple):
            if len(distance_ratio) != 2 or any(r < 0.0 for r in distance_ratio):
                raise ValueError("capsule distance_ratio tuple must be two non-negative values")
        elif distance_ratio < 0.0:
            raise ValueError("capsule distance_ratio must be non-negative")

    sizes = np.array(box_size, dtype=float)
    longest_idx, second_idx, shortest_idx = np.argsort(sizes)[::-1]
    sx, sy, sz = sizes[shortest_idx], sizes[second_idx], sizes[longest_idx]

    if grid_shape is None:
        # Original: single row along second_idx
        r = sx / 2
        cyl_len = sz - sx
        n = max(1, int(np.ceil(sy / sx)))
        centres_1d = np.array([0.0]) if n == 1 else np.linspace(-(sy/2 - r), +(sy/2 - r), n)
        centers = np.zeros((n, 3))
        centers[:, second_idx] = centres_1d
    else:
        # 2D grid: n_rows along shortest_idx, n_cols along second_idx.
        # Auto placement maximizes circle union area inside the box without boundary overrun.
        # Explicit distance_ratio remains as a diagnostic override: 1 keeps cell centers, 0
        # collapses all capsules to the box center; a tuple gives (row_ratio, col_ratio).
        # overfill_ratio applies positive radius padding after the no-overrun fit.
        n_rows, n_cols = grid_shape
        if n_rows < 1 or n_cols < 1:
            raise ValueError("capsule_grid dimensions must be positive")
        cell_x, cell_y = sx / n_rows, sy / n_cols
        row_cell_cs = -sx/2 + cell_x * (np.arange(n_rows) + 0.5)
        col_cell_cs = -sy/2 + cell_y * (np.arange(n_cols) + 0.5)

        if distance_ratio is None:
            row_ratio, col_ratio, r_inside = _fit_circle_grid_no_overrun(sx, sy, n_rows, n_cols)
        elif isinstance(distance_ratio, tuple):
            row_ratio, col_ratio = distance_ratio
            row_cs_tmp = row_ratio * row_cell_cs
            col_cs_tmp = col_ratio * col_cell_cs
            centers_tmp = [(float(x), float(y)) for x in row_cs_tmp for y in col_cs_tmp]
            r_inside = min(
                [sx / 2 - abs(x) for x, _ in centers_tmp] +
                [sy / 2 - abs(y) for _, y in centers_tmp]
            )
        else:
            row_ratio = col_ratio = distance_ratio
            row_cs_tmp = row_ratio * row_cell_cs
            col_cs_tmp = col_ratio * col_cell_cs
            centers_tmp = [(float(x), float(y)) for x in row_cs_tmp for y in col_cs_tmp]
            r_inside = min(
                [sx / 2 - abs(x) for x, _ in centers_tmp] +
                [sy / 2 - abs(y) for _, y in centers_tmp]
            )

        row_cs = row_ratio * row_cell_cs
        col_cs = col_ratio * col_cell_cs
        r_cover = 0.0
        for row_i, rc in enumerate(row_cs):
            row_bounds = (-sx/2 + cell_x * row_i, -sx/2 + cell_x * (row_i + 1))
            for col_i, cc in enumerate(col_cs):
                col_bounds = (-sy/2 + cell_y * col_i, -sy/2 + cell_y * (col_i + 1))
                for row_edge in row_bounds:
                    for col_edge in col_bounds:
                        r_cover = max(r_cover, float(np.hypot(row_edge - rc, col_edge - cc)))
        r = r_inside + overfill_ratio * max(0.0, r_cover - r_inside)
        cyl_len = max(0.0, sz - 2 * r)               # caps flush with box ends along longest axis
        centers = []
        for rc in row_cs:
            for cc in col_cs:
                c = np.zeros(3)
                c[shortest_idx] = rc
                c[second_idx] = cc
                centers.append(c)
        centers = np.array(centers)

    return dict(radius=r, cyl_len=cyl_len, centers=centers, axis_idx=int(longest_idx))

@dataclass
class Box:
    size: List[float]
    
@dataclass
class Cylinder:
    radius: float
    length: float
    
@dataclass
class Capsule:
    radius: float
    length: float
    
@dataclass
class Sphere:
    radius: float
    
@dataclass
class Mesh:
    filename: str
    scale: Optional[List[float]] = None


def mesh_asset_name(shape: "Mesh") -> str:
    """MJCF `<mesh>` asset name for a Mesh geom.

    The asset name doubles as the de-duplication key, so it must distinguish one source
    file loaded at different scales — MuJoCo bakes scale into the compiled vertices, and
    a mirrored link reuses its twin's .obj with a negated axis (see
    `builder_helpers.mirror_link_about_plane`). Keying on the file stem alone would
    silently collapse the two and render the mirrored link unmirrored.

    Only the sign pattern is encoded, since that is what mirroring varies; an all-positive
    scale returns the bare stem, so existing asset names are unchanged.
    """
    stem = shape.filename.split("/")[-1].split(".")[0]
    if not shape.scale:
        return stem
    flipped = "".join(ax for ax, s in zip("xyz", shape.scale) if s < 0)
    return f"{stem}_mirror{flipped}" if flipped else stem

GeomShape = Union[Box, Cylinder, Capsule, Sphere, Mesh]

@dataclass
class Origin:
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        # Accept a 3x3 rotation matrix in place of rpy — auto-convert (hard to
        # hand-derive rpy from a known matrix; skip mat_to_rpy call at every callsite).
        arr = np.asarray(self.rpy, dtype=float)
        if arr.shape == (3, 3):
            self.rpy = mat_to_rpy(arr)

@dataclass
class JointLimit:
    lower: float
    upper: float
    effort: float
    velocity: float
    frictionloss: Optional[float] = None
    springref: Optional[float] = None
    springstiffness: Optional[float] = None

@dataclass
class Inertial:
    mass: float
    com: List[float]
    inertia: np.ndarray

@dataclass
class Geom:
    shape: GeomShape
    origin: Origin = field(default_factory=Origin)
    name: Optional[str] = None
    material: Optional[str] = None
    rgba: Optional[List[float]] = None
    friction: Optional[List[float]] = None
    solref: Optional[List[float]] = None
    margin: Optional[float] = None
    density: float = 1000.0
    is_visual: bool = True
    is_collision: bool = False
    contype: Optional[int] = None
    conaffinity: Optional[int] = None
    group: Optional[int] = None
    use_capsule_approximation: bool = False
    capsule_grid: Optional[Tuple[int, int]] = None
    capsule_overfill_ratio: float = 0.0  # grid approx only: 0=inside cell, 1=cell-corner cover, >1=extra padding
    capsule_distance_ratio: Optional[Union[float, Tuple[float, float]]] = None  # grid approx only: None=auto max-fill, float=(row=col), tuple=(row,col)

@dataclass
class Site:
    name: str
    origin: Origin = field(default_factory=Origin)

@dataclass
class Link:
    name: str
    geoms: List[Geom] = field(default_factory=list)
    sites: List[Site] = field(default_factory=list)
    mass: Optional[float] = None
    inertial: Optional[Inertial] = None

@dataclass
class Actuator:
    name: str
    target_joint: str
    type: str = "motor"
    gear: float = 1.0
    ctrlrange: Optional[List[float]] = None

@dataclass
class Sensor:
    name: str
    type: str
    site: Optional[str] = None
    body: Optional[str] = None
    joint: Optional[str] = None

@dataclass
class Equality:
    """Joint-coupling equality constraint: joint1 = poly(joint2).

    Emits <equality><joint joint1 joint2 polycoef .../></equality>. Distinct from the
    site-site `connect` constraint the builder synthesizes for loop-closure joints; this is
    the 1:1 (or polynomial) mimic used to gang two actuated joints onto one DOF (e.g. a
    rack-and-pinion gripper's two jaws). polycoef defaults to 1:1 coupling [0,1,0,0,0].
    """
    joint1: str
    joint2: str
    polycoef: List[float] = field(default_factory=lambda: [0.0, 1.0, 0.0, 0.0, 0.0])
    solref: Optional[List[float]] = None
    solimp: Optional[List[float]] = None

@dataclass
class Joint:
    name: str
    parent: str
    child: str
    type: str = "hinge"
    origin: Origin = field(default_factory=Origin)
    axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    limit: Optional[JointLimit] = None
    damping: float = 0.0
    armature: float = 0.0
    friction: float = 0.0
    is_loop_closure: bool = False
    ref: float = 0.0  # Zero-position offset (rad). Shifts q=0 without touching body frames or inertia.
    solreflimit: Optional[List[float]] = None  # [timeconst, dampratio] for the limit constraint;
                                               # stiffen (e.g. ~2*dt) when a light DOF is rammed
                                               # into its range at high actuator force.
    solimplimit: Optional[List[float]] = None  # [dmin, dmax, width, midpoint, power] for the limit.

@dataclass
class Texture:
    """A MuJoCo texture asset, referenced by name from Material.texture.

    `file` is resolved by MuJoCo relative to the emitted XML's own directory (no `texturedir` is
    set on the compiler), so pass it relative to that -- e.g. "meshes/foo.png" for an image sitting
    beside the generated meshes. Fields map 1:1 onto MJCF attributes.
    """
    name: str
    type: str = "2d"
    file: Optional[str] = None
    builtin: Optional[str] = None
    rgb1: Optional[List[float]] = None
    rgb2: Optional[List[float]] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class Material:
    name: str
    rgba: Optional[List[float]] = None
    emission: Optional[float] = None
    specular: Optional[float] = None
    shininess: Optional[float] = None
    reflectance: Optional[float] = None
    texture: Optional[str] = None       # name of a Texture passed to Robot(textures=...)
    texrepeat: Optional[List[float]] = None  # tiles per metre when texuniform, else per geom face
    texuniform: Optional[bool] = None   # True = project in metres, so tile size is size-independent

def rotate_inertia(rotation, inertia):
    return rotation @ inertia @ rotation.T

def parallel_axis_term(mass, displacement):
    d = np.asarray(displacement, dtype=float)
    return mass * ((np.dot(d, d) * np.eye(3)) - np.outer(d, d))

def _compute_link_inertial(link: Link) -> Inertial:
    masses = []
    coms = []
    inertias = []
    rotations = []
    
    for g in link.geoms:
        if not g.is_visual: continue
        rotation = _rpy_to_transform(g.origin.rpy, [0,0,0])[:3, :3]
        
        if isinstance(g.shape, Box):
            x, y, z = g.shape.size
            v = x*y*z
            m = v * g.density
            ix = (m/12.0) * (y**2 + z**2)
            iy = (m/12.0) * (x**2 + z**2)
            iz = (m/12.0) * (x**2 + y**2)
            I = np.diag([ix, iy, iz])
            com = np.array(g.origin.xyz)
            
        elif isinstance(g.shape, Cylinder):
            r, h = g.shape.radius, g.shape.length
            v = np.pi * (r**2) * h
            m = v * g.density
            ix = (m/12.0) * (3*r**2 + h**2)
            iy = ix
            iz = (m/2.0) * r**2
            I = np.diag([ix, iy, iz])
            com = np.array(g.origin.xyz)
            
        elif isinstance(g.shape, Capsule):
            r, h = g.shape.radius, g.shape.length
            v = np.pi * (r**2) * (h) + (4./3.) * np.pi * (r**3)
            m = v * g.density
            ix = (m/12.0) * (3*r**2 + h**2)
            iy = ix
            iz = (m/2.0) * r**2
            I = np.diag([ix, iy, iz])
            com = np.array(g.origin.xyz)
            
        elif isinstance(g.shape, Sphere):
            r = g.shape.radius
            v = (4.0/3.0) * np.pi * (r**3)
            m = v * g.density
            ix = (2.0/5.0) * m * r**2
            I = np.diag([ix, ix, ix])
            com = np.array(g.origin.xyz)
            
        elif isinstance(g.shape, Mesh):
            import trimesh
            import os
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            except:
                base_dir = "."
            path = os.path.join(base_dir, g.shape.filename)
            try:
                tm = trimesh.load(path)
                if g.shape.scale:
                    tm.apply_scale(g.shape.scale)
                v = tm.volume
                m = v * g.density
                com_mesh = tm.center_mass
                I = tm.moment_inertia * (m / v)
                com = np.array(g.origin.xyz) + rotation @ com_mesh
            except Exception as e:
                m = 0.001
                com = np.array(g.origin.xyz)
                I = np.eye(3) * 1e-6

        masses.append(m)
        coms.append(com)
        inertias.append(I)
        rotations.append(rotation)

    if sum(masses) < 1e-9:
        return Inertial(mass=1e-3, com=[0,0,0], inertia=np.eye(3)*1e-6)
        
    total_mass = sum(masses)
    combined_com = sum(m * c for m, c in zip(masses, coms)) / total_mass
    
    combined_I = np.zeros((3,3))
    for m, c, rot, I_l in zip(masses, coms, rotations, inertias):
        I_rot = rotate_inertia(rot, I_l)
        disp = c - combined_com
        combined_I += I_rot + parallel_axis_term(m, disp)
        
    if link.mass is not None:
        scale = link.mass / total_mass
        total_mass = link.mass
        combined_I *= scale

    return Inertial(mass=total_mass, com=combined_com.tolist(), inertia=combined_I)

class Robot:
    def __init__(self, name: str, materials: Optional[List[Material]] = None,
                 textures: Optional[List[Texture]] = None):
        self.name = name
        self.materials = materials or []
        self.textures = textures or []
        self._links: Dict[str, Link] = {}
        self._joints: Dict[str, Joint] = {}
        
    def add_link(self, link: Link) -> 'Robot':
        self._links[link.name] = link
        return self
        
    def add_joint(self, joint: Joint) -> 'Robot':
        self._joints[joint.name] = joint
        return self
        
    def validate(self):
        pass

    def _spanning_tree_dfs(self, joint_order: Optional[List[str]] = None):
        adj = {n: [] for n in self._links}
        for j in self._joints.values():
            if not j.is_loop_closure:
                adj[j.parent].append((j.child, j))

        if joint_order:
            # Emission order (and therefore the qpos / actuator index layout) otherwise falls out
            # of the order `add_joint` happened to be called in, so restructuring the build code
            # silently permutes the model. Ranking siblings by `joint_order` makes it declared.
            # Stable sort: joints absent from the list keep their `add_joint` order.
            rank = {name: i for i, name in enumerate(joint_order)}
            for children in adj.values():
                children.sort(key=lambda cj: rank.get(cj[1].name, len(rank)))

        all_children = {j.child for j in self._joints.values() if not j.is_loop_closure}
        roots = sorted([n for n in self._links if n not in all_children])
        return roots[0] if roots else None, adj

    def _dfs_joint_names(self, root_name: str, adj: dict) -> List[str]:
        """Tree-joint names in the order `_emit_body` will emit them (pre-order DFS)."""
        order = []

        def walk(link_name):
            for child_name, child_joint in adj[link_name]:
                order.append(child_joint.name)
                walk(child_name)

        walk(root_name)
        return order


    def _emit_body(self, link: Link, joint: Optional[Joint], adj: dict, parent_body: ET.Element):
        body = ET.SubElement(parent_body, "body", name=link.name)
        
        if joint:
            body.set("pos", _parse_array(joint.origin.xyz))
            quat = _euler_to_quat(joint.origin.rpy)
            if not np.allclose(quat, [1,0,0,0], atol=1e-6):
                body.set("quat", _parse_array(quat))
                
        inert = link.inertial
        if not inert:
            inert = _compute_link_inertial(link)
            
        inert_el = ET.SubElement(body, "inertial", mass=str(inert.mass), pos=_parse_array(inert.com))
        w, v = np.linalg.eigh(inert.inertia)
        if np.linalg.det(v) < 0:
            v[:, 0] *= -1
            
        inert_el.set("diaginertia", _parse_array(w))
        
        trace = np.trace(v)
        if trace > 0:
            S = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * S
            qx = (v[2,1] - v[1,2]) / S
            qy = (v[0,2] - v[2,0]) / S
            qz = (v[1,0] - v[0,1]) / S
        elif v[0,0] > v[1,1] and v[0,0] > v[2,2]:
            S = 2.0 * np.sqrt(1.0 + v[0,0] - v[1,1] - v[2,2])
            qw = (v[2,1] - v[1,2]) / S
            qx = 0.25 * S
            qy = (v[0,1] + v[1,0]) / S
            qz = (v[0,2] + v[2,0]) / S
        elif v[1,1] > v[2,2]:
            S = 2.0 * np.sqrt(1.0 + v[1,1] - v[0,0] - v[2,2])
            qw = (v[0,2] - v[2,0]) / S
            qx = (v[0,1] + v[1,0]) / S
            qy = 0.25 * S
            qz = (v[1,2] + v[2,1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + v[2,2] - v[0,0] - v[1,1])
            qw = (v[1,0] - v[0,1]) / S
            qx = (v[0,2] + v[2,0]) / S
            qy = (v[1,2] + v[2,1]) / S
            qz = 0.25 * S
            
        inert_el.set("quat", _parse_array([qw, qx, qy, qz]))
        
        if joint and joint.type != "fixed":
            j_el = ET.SubElement(body, "joint", name=joint.name, type=joint.type, pos="0 0 0", axis=_parse_array(joint.axis))
            if joint.limit:
                j_el.set("range", f"{joint.limit.lower} {joint.limit.upper}")
            if joint.solreflimit is not None: j_el.set("solreflimit", _parse_array(joint.solreflimit))
            if joint.solimplimit is not None: j_el.set("solimplimit", _parse_array(joint.solimplimit))
            if joint.ref: j_el.set("ref", f"{joint.ref:.6g}")
            if joint.damping: j_el.set("damping", str(joint.damping))
            if joint.armature: j_el.set("armature", str(joint.armature))
            if joint.friction: j_el.set("frictionloss", str(joint.friction))

        num_cols = 0
        for g in link.geoms:
            if g.is_collision:
                if g.use_capsule_approximation and isinstance(g.shape, Box):
                    num_cols += len(_capsules_from_box(g.shape.size, g.capsule_grid)['centers'])
                else:
                    num_cols += 1
                    
        num_vis = sum(1 for g in link.geoms if g.is_visual)
        col_idx = 0
        vis_idx = 0

        for i, g in enumerate(link.geoms):
            if not g.is_collision and not g.is_visual:
                continue

            if g.use_capsule_approximation and isinstance(g.shape, Box) and g.is_collision:
                capsules_info = _capsules_from_box(
                    g.shape.size, g.capsule_grid, g.capsule_overfill_ratio, g.capsule_distance_ratio
                )
                R_box = _rpy_to_transform(g.origin.rpy, [0,0,0])[:3, :3]
                cap_rpy = _rpy_compose(g.origin.rpy, _CAPSULE_AXIS_RPY[capsules_info['axis_idx']])
                for j, c in enumerate(capsules_info['centers']):
                    cap_origin = np.array(g.origin.xyz) + R_box @ c
                    name_str = f"{link.name}_collision{col_idx}" if num_cols > 1 else f"{link.name}_collision"
                    gel = ET.SubElement(body, "geom", type="capsule", size=f"{capsules_info['radius']} {capsules_info['cyl_len']/2.0}",
                                       name=name_str,
                                       pos=_parse_array(cap_origin.tolist()))
                    col_idx += 1
                    quat = _euler_to_quat(cap_rpy)
                    if not np.allclose(quat, [1,0,0,0], atol=1e-6):
                        gel.set("quat", _parse_array(quat))
                    gel.set("class", "collision")
                    if g.contype is not None: gel.set("contype", str(g.contype))
                    if g.conaffinity is not None: gel.set("conaffinity", str(g.conaffinity))
                    if g.group is not None: gel.set("group", str(g.group))
                    if g.margin is not None: gel.set("margin", str(g.margin))
                    if g.friction is not None: gel.set("friction", _parse_array(g.friction))
                    if g.solref is not None: gel.set("solref", _parse_array(g.solref))
            else:
                if g.is_collision:
                    name_str = g.name or (f"{link.name}_collision{col_idx}" if num_cols > 1 else f"{link.name}_collision")
                    col_idx += 1
                else:
                    name_str = g.name or (f"{link.name}_visual{vis_idx}" if num_vis > 1 else f"{link.name}_visual")
                    vis_idx += 1
                    
                gel = ET.SubElement(body, "geom", name=name_str)
                if isinstance(g.shape, Box):
                    gel.set("type", "box")
                    gel.set("size", _parse_array([s/2.0 for s in g.shape.size]))
                elif isinstance(g.shape, Cylinder):
                    gel.set("type", "cylinder")
                    gel.set("size", f"{g.shape.radius} {g.shape.length/2.0}")
                elif isinstance(g.shape, Capsule):
                    gel.set("type", "capsule")
                    gel.set("size", f"{g.shape.radius} {g.shape.length/2.0}")
                elif isinstance(g.shape, Sphere):
                    gel.set("type", "sphere")
                    gel.set("size", f"{g.shape.radius}")
                elif isinstance(g.shape, Mesh):
                    gel.set("type", "mesh")
                    gel.set("mesh", mesh_asset_name(g.shape))
                    
                gel.set("pos", _parse_array(g.origin.xyz))
                quat = _euler_to_quat(g.origin.rpy)
                if not np.allclose(quat, [1,0,0,0], atol=1e-6):
                    gel.set("quat", _parse_array(quat))
                    
                if g.is_collision and not g.is_visual:
                    gel.set("class", "collision")
                elif g.is_visual and not g.is_collision:
                    gel.set("class", "visual")
                    if g.material: gel.set("material", g.material)
                    if g.rgba: gel.set("rgba", _parse_array(g.rgba))
                else:
                    raise Exception(f"Geom {g.name} must be exclusively visual or collision right now")
                
                if g.contype is not None: gel.set("contype", str(g.contype))
                if g.conaffinity is not None: gel.set("conaffinity", str(g.conaffinity))
                if g.group is not None: gel.set("group", str(g.group))
                if g.margin is not None: gel.set("margin", str(g.margin))
                if g.friction is not None: gel.set("friction", _parse_array(g.friction))
                if g.solref is not None: gel.set("solref", _parse_array(g.solref))

        for s in link.sites:
            sel = ET.SubElement(body, "site", name=s.name, pos=_parse_array(s.origin.xyz))
            quat = _euler_to_quat(s.origin.rpy)
            if not np.allclose(quat, [1,0,0,0], atol=1e-6):
                sel.set("quat", _parse_array(quat))

        # preserve insertion order for deterministic DFS and backward compatibility (RL obs)
        for child_name, child_joint in adj[link.name]:
            self._emit_body(self._links[child_name], child_joint, adj, body)

    def to_mjcf_string(self, add_freejoint=True, add_light_camera=True, actuators:List[Actuator]=None, imu_site_name="imu_site", contact_body_names=None, sensors:List[Sensor]=None, equalities:List[Equality]=None, joint_order:List[str]=None) -> str:
        """
        Args:
            joint_order: Full list of tree-joint names, in the order they must appear in the
                emitted model — i.e. the qpos / actuator index layout. Supplied here rather than
                at construction so build code stays order-agnostic: assemble the robot however is
                convenient, declare the layout once at export. Siblings are ranked by this list
                and the realized order is then asserted against it, so any permutation from any
                cause fails loudly instead of silently invalidating checkpoints and keyframes.
                Excludes the free joint (added by `add_freejoint`, always index 0).
                Ordering is sibling-level only; depth is fixed by kinematics, so an order that
                contradicts the tree is unrealizable and raises.
        """
        actuators = actuators or []
        contact_body_names = contact_body_names or []
        sensors = sensors or []
        equalities = equalities or []

        self.validate()
        root_name, adj = self._spanning_tree_dfs(joint_order)

        if joint_order:
            realized = self._dfs_joint_names(root_name, adj)
            if realized != list(joint_order):
                raise ValueError(
                    f"joint_order not realized.\n  requested: {list(joint_order)}\n  emitted:   {realized}"
                )

        mujoco = ET.Element("mujoco", model=self.name)
        ET.SubElement(mujoco, "compiler", angle="radian", meshdir="meshes", autolimits="true")
        
        default = ET.SubElement(mujoco, "default")
        rc = ET.SubElement(default, "default", {"class": self.name})
        
        vd = ET.SubElement(rc, "default", {"class": "visual"})
        ET.SubElement(vd, "geom", group="2", type="mesh", contype="0", conaffinity="0", material="silver")
        
        cd = ET.SubElement(rc, "default", {"class": "collision"})
        ET.SubElement(cd, "geom", group="3", contype="1", conaffinity="1")
        
        asset = ET.SubElement(mujoco, "asset")
        for t in self.textures:  # before the materials that reference them
            tex = ET.SubElement(asset, "texture", name=t.name, type=t.type)
            if t.file is not None: tex.set("file", t.file)
            if t.builtin is not None: tex.set("builtin", t.builtin)
            if t.rgb1: tex.set("rgb1", _parse_array(t.rgb1))
            if t.rgb2: tex.set("rgb2", _parse_array(t.rgb2))
            if t.width is not None: tex.set("width", str(t.width))
            if t.height is not None: tex.set("height", str(t.height))
        for m in self.materials:
            mat = ET.SubElement(asset, "material", name=m.name)
            if m.rgba: mat.set("rgba", _parse_array(m.rgba))
            if m.emission is not None: mat.set("emission", str(m.emission))
            if m.specular is not None: mat.set("specular", str(m.specular))
            if m.shininess is not None: mat.set("shininess", str(m.shininess))
            if m.reflectance is not None: mat.set("reflectance", str(m.reflectance))
            if m.texture is not None: mat.set("texture", m.texture)
            if m.texrepeat is not None: mat.set("texrepeat", _parse_array(m.texrepeat))
            if m.texuniform is not None: mat.set("texuniform", str(m.texuniform).lower())
            
        mesh_names = set()
        for link in self._links.values():
            for g in link.geoms:
                if isinstance(g.shape, Mesh):
                    name = mesh_asset_name(g.shape)
                    if name not in mesh_names:
                        me = ET.SubElement(asset, "mesh", name=name, file=g.shape.filename.split("/")[-1])
                        if g.shape.scale:
                            me.set("scale", _parse_array(g.shape.scale))
                        mesh_names.add(name)
                        
        worldbody = ET.SubElement(mujoco, "worldbody")
        root_body = ET.Element("body", name=root_name, childclass=self.name)
        
        if add_light_camera:
            ET.SubElement(root_body, "light", pos="0 0 2", mode="trackcom")
            ET.SubElement(root_body, "camera", name="tracking", pos="1.5 -1.5 1", xyaxes="0.707 0.707 0 -0.3 0.3 0.9", mode="trackcom")
            
        if add_freejoint:
            ET.SubElement(root_body, "freejoint", name="floating_base_joint")
            
        worldbody.append(root_body)
        
        dummy_root = ET.Element("dummy")
        self._emit_body(self._links[root_name], None, adj, dummy_root)
        
        rb_filled = list(dummy_root)[0]
        for child in rb_filled:
            root_body.append(child)
            
        if imu_site_name:
            ET.SubElement(root_body, "site", name=imu_site_name, pos="0 0 0")
            
        if contact_body_names:
            for b_name in contact_body_names:
                b_elems = worldbody.findall(f".//body[@name='{b_name}']")
                for b in b_elems:
                    geoms = [g for g in b.findall("geom") if g.get("class") == "collision"]
                    if geoms:
                        pos = np.mean([np.array([float(x) for x in g.get("pos", "0 0 0").split()]) for g in geoms], axis=0)
                        ET.SubElement(b, "site", name=f"{b_name}_contact", pos=_parse_array(pos))

        eq_joints = [j for j in self._joints.values() if j.is_loop_closure]
        if eq_joints or equalities:
            equality = ET.SubElement(mujoco, "equality")
            for j in eq_joints:
                c_body = worldbody.find(f".//body[@name='{j.child}']")
                ET.SubElement(c_body, "site", name=f"{j.name}_c_site", pos="0 0 0")
                p_body = worldbody.find(f".//body[@name='{j.parent}']")
                ET.SubElement(p_body, "site", name=f"{j.name}_p_site", pos=_parse_array(j.origin.xyz))
                ET.SubElement(equality, "connect", site1=f"{j.name}_c_site", site2=f"{j.name}_p_site")
            for eq in equalities:
                jel = ET.SubElement(equality, "joint", joint1=eq.joint1, joint2=eq.joint2,
                                    polycoef=_parse_array(eq.polycoef))
                if eq.solref is not None: jel.set("solref", _parse_array(eq.solref))
                if eq.solimp is not None: jel.set("solimp", _parse_array(eq.solimp))

        if sensors:
            sensor = ET.SubElement(mujoco, "sensor")
            for s in sensors:
                sel = ET.SubElement(sensor, s.type, name=s.name)
                if s.site: sel.set("site", s.site)
                if s.body: sel.set("body", s.body)
                if s.joint: sel.set("joint", s.joint)
                
        if actuators:
            act = ET.SubElement(mujoco, "actuator")
            for a in actuators:
                ael = ET.SubElement(act, a.type, name=a.name, joint=a.target_joint, gear=str(a.gear), ctrllimited="true")
                if a.ctrlrange:
                    ael.set("ctrlrange", _parse_array(a.ctrlrange))
                else: 
                     j_ref = self._joints.get(a.target_joint)
                     if j_ref and j_ref.limit:
                         ael.set("ctrlrange", f"{-j_ref.limit.effort} {j_ref.limit.effort}")
        
        xmlstr = xml.dom.minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="  ")
        return "\n".join([line for line in xmlstr.split('\n') if line.strip()])

    def export_mjcf(self, filepath: str, **kwargs):
        content = self.to_mjcf_string(**kwargs)
        with open(filepath, "w") as f:
            f.write(content)
