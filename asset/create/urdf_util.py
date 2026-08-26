"""
modified from https://github.com/mikedh/trimesh/blob/main/trimesh/inertia.py
-------------

Functions for dealing with inertia tensors.

Results validated against known geometries and checked for
internal consistency.
"""

import abc
import numpy as np
from numpy import float64, floating, int64, integer, unsignedinteger
from functools import cached_property

from typing import Optional, Union
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import multi_dot

Integer = Union[int, integer, unsignedinteger]
Floating = Union[float, floating]
Number = Union[Floating, Integer]

from trimesh import caching, creation, inertia, sample, triangles, util
from trimesh import transformations as tf
from trimesh.base import Trimesh
from trimesh.constants import log, tol


# immutable identity matrix for checks
_IDENTITY = np.eye(4)
_IDENTITY.flags.writeable = False


def rpy_to_transform(rpy, pos):
    """
    Fast RPY + position to 4x4 transform (extrinsic XYZ).
    """
    rpy = np.asarray(rpy, dtype=float)
    pos = np.asarray(pos, dtype=float)

    sr, sp, sy = np.sin(rpy)
    cr, cp, cy = np.cos(rpy)

    rot = np.array([
        [cp*cy, cy*sr*sp - cr*sy, sr*sy + cr*cy*sp],
        [cp*sy, cr*cy + sr*sp*sy, cr*sp*sy - cy*sr],
        [-sp,   cp*sr,             cr*cp]
    ])

    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform

def parallel_axis_term(mass, displacement):
    """
    Compute the parallel axis theorem term for translating inertia.

    I_new = I_old + parallel_axis_term(mass, displacement)
    where displacement = new_point - old_point (vector from old reference to new reference)

    Parameters
    ----------
    mass : float
        Mass of the body.
    displacement : (3,) array-like
        Vector from old reference point to new reference point.

    Returns
    -------
    term : (3,3) ndarray
        The parallel axis contribution to add to the inertia tensor.
    """
    d = np.asarray(displacement, dtype=float)
    return mass * ((np.dot(d, d) * np.eye(3)) - np.outer(d, d))


def rotate_inertia(rotation, inertia):
    """
    Rotate an inertia tensor from local frame to world frame.

    Parameters
    ----------
    rotation : (3,3) ndarray
        Rotation matrix from local frame to world frame.
    inertia : (3,3) ndarray
        Inertia tensor in local frame.

    Returns
    -------
    I_rotated : (3,3) ndarray
        Inertia tensor in world frame.
    """
    return rotation @ inertia @ rotation.T


def transform_inertia(transform, inertia, parallel_axis, mass):
    """
    Transform inertia tensor from body frame to world frame.

    Parameters
    ----------
    transform : (4,4) ndarray
        Homogeneous transform from body frame to world frame.
    inertia : (3,3) ndarray
        Inertia tensor in body coordinates (about COM).
    parallel_axis : bool
        If True, apply parallel axis theorem to translate inertia to world origin.
        If False, only rotate the inertia (keeps it about the body's COM).
    mass : float
        Mass of the body (only used if parallel_axis=True).

    Returns
    -------
    I_world : (3,3) ndarray
        Inertia tensor in world coordinates.
        If parallel_axis=True: about world origin.
        If parallel_axis=False: about body COM (just rotated).
    """
    R = transform[:3, :3]
    r = transform[:3, 3]

    I_world_com = rotate_inertia(R, inertia)

    if parallel_axis:
        return I_world_com + parallel_axis_term(mass, r)
    else:
        return I_world_com


def box_inertia(mass: Number, size: ArrayLike, transform: Optional[ArrayLike] = None) -> NDArray[float64]:
    """
    Return the inertia tensor of a box.
    Parameters
    ------------
    mass : float
      Mass of box
    size : (3,) float
      Size of box
    transform : (4, 4) float
      Transformation of box
    Returns
    ------------
    inertia : (3, 3) float
      Inertia tensor
    """
    # make sure the size is a 3D vector
    x_dim, y_dim, z_dim = size
    ixx = (1/12) * mass * (y_dim**2 + z_dim**2)
    iyy = (1/12) * mass * (x_dim**2 + z_dim**2)
    izz = (1/12) * mass * (x_dim**2 + y_dim**2)
    inertia =  np.array([
        [ixx, 0.0, 0.0],
        [0.0, iyy, 0.0],
        [0.0, 0.0, izz]
    ])
    if transform is not None:
        inertia = transform_inertia(transform, inertia, parallel_axis=False, mass=mass)
    return inertia


def cylinder_inertia(
    mass: Number, radius: Number, height: Number, transform: Optional[ArrayLike] = None
) -> NDArray[float64]:
    """
    Return the inertia tensor of a cylinder.

    Parameters
    ------------
    mass : float
      Mass of cylinder
    radius : float
      Radius of cylinder
    height : float
      Height of cylinder
    transform : (4, 4) float
      Transformation of cylinder

    Returns
    ------------
    inertia : (3, 3) float
      Inertia tensor
    """
    h2, r2 = height**2, radius**2
    diagonal = np.array(
        [
            ((mass * h2) / 12) + ((mass * r2) / 4),
            ((mass * h2) / 12) + ((mass * r2) / 4),
            (mass * r2) / 2,
        ]
    )
    inertia = diagonal * np.eye(3)

    if transform is not None:
        # !diffrent from trimesh!
        inertia = transform_inertia(transform, inertia, parallel_axis=True, mass=mass)

    return inertia


def sphere_inertia(mass: Number, radius: Number, transform: Optional[ArrayLike] = None) -> NDArray[float64]:
    """
    Return the inertia tensor of a sphere.

    Parameters
    ------------
    mass : float
      Mass of sphere
    radius : float
      Radius of sphere

    Returns
    ------------
    inertia : (3, 3) float
      Inertia tensor
    """
    inertia =  (2.0 / 5.0) * (radius**2) * mass * np.eye(3)
    if transform is not None:
        # !diffrent from trimesh!
        inertia = transform_inertia(transform, inertia, parallel_axis=True, mass=mass)
    return inertia

import numpy as np
import trimesh

def mesh_inertia(file_path, mass, units='mm', about='com'):
    """
    Compute inertia tensor and center of mass for an STL file assuming uniform mass distribution.

    Parameters:
        file_path (str): Path to the STL file.
        mass (float): Total mass of the object in kg.
        units (str): 'mm' or 'm' (default: 'mm').
        about (str): 'com' or 'origin' for inertia reference point (default: 'com').
                     'com' - inertia about center of mass (correct for URDF/MJCF)
                     'origin' - inertia about mesh origin (legacy behavior)

    Returns:
        dict: {
            'mass': mass in kg,
            'center_of_mass': np.array([x, y, z]) in mesh coordinates,
            'inertia_tensor': 3x3 np.array in kg·m² (about specified reference point)
        }
    """
    
    # Load mesh
    mesh = trimesh.load_mesh(file_path)
    
    # Scale to meters if necessary
    if units == 'mm':
        mesh.apply_scale(0.001)
    
    # Get volume and inertia for unit density
    inertia_matrix = mesh.moment_inertia
    com = mesh.center_mass

    # Compute scale factor based on given mass
    density = mass / mesh.volume  # kg/m³
    inertia_scaled = inertia_matrix * density  # scales linearly with density
    
    # If about origin is requested, use parallel axis theorem
    if about == 'origin':
        displacement = -com
        translation_matrix = np.array([
            [displacement[1]**2 + displacement[2]**2, -displacement[0]*displacement[1], -displacement[0]*displacement[2]],
            [-displacement[0]*displacement[1], displacement[0]**2 + displacement[2]**2, -displacement[1]*displacement[2]],
            [-displacement[0]*displacement[2], -displacement[1]*displacement[2], displacement[0]**2 + displacement[1]**2]
        ])
        inertia_scaled += mass * translation_matrix
    
    return {
        'mass': mass,
        'center_of_mass': com,
        'inertia_tensor': inertia_scaled
    }


def points_inertia(
    points: ArrayLike,
    weights: Union[None, ArrayLike, Number] = None,
    at_center_mass: bool = True,
) -> NDArray[float64]:
    """
    Calculate an inertia tensor for an array of point masses
    at the center of mass.

    Parameters
    ----------
    points : (n, 3)
      Points in space.
    weights : (n,) or number
      Per-point weight to use.
    at_center_mass
      Calculate at the center of mass of the points, or if False
      at the original origin.

    Returns
    -----------
    tensor : (3, 3)
      Inertia tensor for point masses.
    """
    if weights is None:
        # by default make the total weight 1.0 to match
        # the default mass in other functions, and so that
        # if a user didn't specify anything it doesn't blow
        # up the scale depending on the number of points
        weights = np.full(len(points), 1.0 / float(len(points)), dtype=np.float64)
    elif isinstance(weights, (float, np.integer, int)):
        # "is it a number" check
        weights = np.full(len(points), float(weights), dtype=np.float64)
    else:
        weights = np.array(weights)
        if len(weights) != len(points):
            raise ValueError(
                f"Weights must correspond to points! {len(weights)} != {len(points)}"
            )

    # make sure the points are an array of correct shape
    points = np.asanyarray(points, dtype=np.float64)
    if len(points.shape) != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be `(n, 3)` not {points.shape}")

    if at_center_mass:
        # get the center of mass of the points
        center_mass = np.average(points, weights=weights, axis=0)
        # get the points with the origin at their center of mass
        points_com = points - center_mass
    else:
        # calculate at original origin
        points_com = points

    # expand into shorthand for the expressions
    x, y, z = points_com.T
    x2, y2, z2 = (points_com**2).T

    # calculate tensors per-point in a flattened (9, n) array
    # from physics.stackexchange.com/questions/614094
    tensors = np.array(
        [y2 + z2, -x * y, -x * z, -x * y, x2 + z2, -y * z, -x * z, -y * z, x2 + y2],
        dtype=np.float64,
    )

    # combine the weighted tensors and reshape
    tensor = (tensors * weights).sum(axis=1).reshape((3, 3))

    return tensor


def principal_axis(inertia: ArrayLike):
    """
    Find the principal components and principal axis
    of inertia from the inertia tensor.

    Parameters
    ------------
    inertia : (3, 3) float
      Inertia tensor

    Returns
    ------------
    components : (3,) float
      Principal components of inertia
    vectors : (3, 3) float
      Row vectors pointing along the
      principal axes of inertia
    """
    inertia = np.asanyarray(inertia, dtype=np.float64)
    if inertia.shape != (3, 3):
        raise ValueError("inertia tensor must be (3, 3)!")

    # you could any of the following to calculate this:
    # np.linalg.svd, np.linalg.eig, np.linalg.eigh
    # moment of inertia is square symmetric matrix
    # eigh has the best precision in tests
    components, vectors = np.linalg.eigh(inertia)

    # eigh returns them as column vectors, change them to row vectors
    vectors = vectors.T

    return components, vectors


# def transform_inertia(
#     transform: ArrayLike,
#     inertia_tensor: ArrayLike,
#     parallel_axis: bool = False,
#     mass: Optional[Number] = None,
# ):
#     """
#      Transform an inertia tensor to a new frame.

#      Note that in trimesh `mesh.moment_inertia` is *axis aligned*
#      and at `mesh.center_mass`.

#      So to transform to a new frame and get the moment of inertia at
#      the center of mass the translation should be ignored and only
#      rotation applied.

#      If parallel axis is enabled it will compute the inertia
#      about a new location.

#      More details in the MIT OpenCourseWare PDF:
#     ` MIT16_07F09_Lec26.pdf`


#      Parameters
#      ------------
#      transform : (3, 3) or (4, 4) float
#        Transformation matrix
#      inertia_tensor : (3, 3) float
#        Inertia tensor.
#      parallel_axis : bool
#        Apply the parallel axis theorum or not.
#        If the passed inertia tensor is at the center of mass
#        and you want the new post-transform tensor also at the
#        center of mass you DON'T want this enabled as you *only*
#        want to apply the rotation. Use this to get moment of
#        inertia at an arbitrary frame that isn't the center of mass.

#      Returns
#      ------------
#      transformed : (3, 3) float
#        Inertia tensor in new frame.
#     """
#     # check inputs and extract rotation
#     transform = np.asanyarray(transform, dtype=np.float64)
#     if transform.shape == (4, 4):
#         rotation = transform[:3, :3]
#     elif transform.shape == (3, 3):
#         rotation = transform
#     else:
#         raise ValueError("transform must be (3, 3) or (4, 4)!")

#     inertia_tensor = np.asanyarray(inertia_tensor, dtype=np.float64)
#     if inertia_tensor.shape != (3, 3):
#         raise ValueError("inertia_tensor must be (3, 3)!")

#     if parallel_axis:
#         if transform.shape == (3, 3):
#             # shorthand for "translation"
#             a = np.zeros(3, dtype=np.float64)
#         else:
#             # get the translation
#             a = transform[:3, 3]
#         # First the changed origin of the new transform is taken into
#         # account. To calculate the inertia tensor
#         # the parallel axis theorem is used
#         M = np.array(
#             [
#                 [a[1] ** 2 + a[2] ** 2, -a[0] * a[1], -a[0] * a[2]],
#                 [-a[0] * a[1], a[0] ** 2 + a[2] ** 2, -a[1] * a[2]],
#                 [-a[0] * a[2], -a[1] * a[2], a[0] ** 2 + a[1] ** 2],
#             ]
#         )
#         aligned_inertia = inertia_tensor + mass * M

#         return multi_dot([rotation.T, aligned_inertia, rotation])

#     return multi_dot([rotation, inertia_tensor, rotation.T])


def radial_symmetry(mesh):
    """
    Check whether a mesh has radial symmetry.

    Returns
    -----------
    symmetry : None or str
         None         No rotational symmetry
         'radial'     Symmetric around an axis
         'spherical'  Symmetric around a point
    axis : None or (3,) float
      Rotation axis or point
    section : None or (3, 2) float
      If radial symmetry provide vectors
      to get cross section
    """

    # shortcuts to avoid typing and hitting cache
    scalar = mesh.principal_inertia_components.copy()

    # exit early if inertia components are all zero
    if (scalar < 1e-30).any():
        return None, None, None

    # normalize the PCI so we can compare them
    scalar = scalar / np.linalg.norm(scalar)
    vector = mesh.principal_inertia_vectors
    # the sorted order of the principal components
    order = scalar.argsort()

    # we are checking if a geometry has radial symmetry
    # if 2 of the PCI are equal, it is a revolved 2D profile
    # if 3 of the PCI (all of them) are equal it is a sphere
    diff = np.abs(np.diff(scalar[order]))
    # diffs that are within tol of zero
    diff_zero = diff < 1e-4

    if diff_zero.all():
        # this is the case where all 3 PCI are identical
        # this means that the geometry is symmetric about a point
        # examples of this are a sphere, icosahedron, etc
        axis = vector[0]
        section = vector[1:]

        return "spherical", axis, section

    elif diff_zero.any():
        # this is the case for 2/3 PCI are identical
        # this means the geometry is symmetric about an axis
        # probably a revolved 2D profile

        # we know that only 1/2 of the diff values are True
        # if the first diff is 0, it means if we take the first element
        # in the ordered PCI we will have one of the non- revolve axis
        # if the second diff is 0, we take the last element of
        # the ordered PCI for the section axis
        # if we wanted the revolve axis we would just switch [0,-1] to
        # [-1,0]

        # since two vectors are the same, we know the middle
        # one is one of those two
        section_index = order[np.array([[0, 1], [1, -1]])[diff_zero]].flatten()
        section = vector[section_index]

        # we know the rotation axis is the sole unique value
        # and is either first or last of the sorted values
        axis_index = order[np.array([-1, 0])[diff_zero]][0]
        axis = vector[axis_index]
        return "radial", axis, section

    return None, None, None


def scene_inertia(scene, transform: Optional[ArrayLike] = None) -> NDArray[float64]:
    """
    Calculate the inertia of a scene about a specific frame.

    Parameters
    ------------
    scene : trimesh.Scene
      Scene with geometry.
    transform : None or (4, 4) float
      Homogeneous transform to compute inertia at.

    Returns
    ----------
    moment : (3, 3)
      Inertia tensor about requested frame
    """
    # shortcuts for tight loop
    graph = scene.graph
    geoms = scene.geometry

    # get the matrix ang geometry name for
    nodes = [graph[n] for n in graph.nodes_geometry]
    # get the moment of inertia with the mesh moved to a location
    moments = np.array(
        [
            geoms[g].moment_inertia_frame(np.dot(np.linalg.inv(mat), transform))
            for mat, g in nodes
            if hasattr(geoms[g], "moment_inertia_frame")
        ],
        dtype=np.float64,
    )

    return moments.sum(axis=0)





##############################################################33
# https://github.com/mikedh/trimesh/blob/main/trimesh/primitives.py
# Subclasses of Trimesh objects that are parameterized as primitives.
# Useful because you can move boxes and spheres around
# and then use trimesh operations on them at any point.


class Primitive(Trimesh):
    """
    Geometric Primitives which are a subclass of Trimesh.
    Mesh is generated lazily when vertices or faces are requested.
    """

    # ignore superclass copy directives
    __copy__ = None
    __deepcopy__ = None

    def __init__(self):
        # run the Trimesh constructor with no arguments
        super().__init__()

        # remove any data
        self._data.clear()
        self._validate = False

        # make sure any cached numpy arrays have
        # set `array.flags.writable = False`
        self._cache.force_immutable = True

    def __repr__(self):
        return f"<trimesh.primitives.{type(self).__name__}>"

    @property
    def faces(self):
        stored = self._cache["faces"]
        if util.is_shape(stored, (-1, 3)):
            return stored
        self._create_mesh()
        return self._cache["faces"]

    @faces.setter
    def faces(self, values):
        if values is not None:
            raise ValueError("primitive faces are immutable: not setting!")

    @property
    def vertices(self):
        stored = self._cache["vertices"]
        if util.is_shape(stored, (-1, 3)):
            return stored

        self._create_mesh()
        return self._cache["vertices"]

    @vertices.setter
    def vertices(self, values):
        if values is not None:
            raise ValueError("primitive vertices are immutable: not setting!")

    @property
    def face_normals(self):
        # if the mesh hasn't been created yet do that
        # before checking to see if the mesh creation
        # already populated the face normals
        if "vertices" not in self._cache:
            self._create_mesh()

        # we need to avoid the logic in the superclass that
        # is specific to the data model prioritizing faces
        stored = self._cache["face_normals"]
        if util.is_shape(stored, (-1, 3)):
            return stored

        # if the creation did not populate normals we have to do it
        # just calculate if not stored
        unit, valid = triangles.normals(self.triangles)
        normals = np.zeros((len(valid), 3))
        normals[valid] = unit
        # store and return
        self._cache["face_normals"] = normals
        return normals

    @face_normals.setter
    def face_normals(self, values):
        if values is not None:
            log.warning("Primitive face normals are immutable!")

    @property
    def transform(self):
        """
        The transform of the Primitive object.

        Returns
        -------------
        transform : (4, 4) float
          Homogeneous transformation matrix
        """
        return self.primitive.transform

    @abc.abstractmethod
    def to_dict(self):
        """
        Should be implemented by each primitive.
        """
        raise NotImplementedError()

    def copy(self, include_visual=True, **kwargs):
        """
        Return a copy of the Primitive object.

        Returns
        -------------
        copied : object
          Copy of current primitive
        """
        # get the constructor arguments
        kwargs.update(self.to_dict())
        # remove the type indicator, i.e. `Cylinder`
        kwargs.pop("kind")
        # create a new object with kwargs
        primitive_copy = type(self)(**kwargs)

        if include_visual:
            # copy visual information
            primitive_copy.visual = self.visual.copy()

        # copy metadata
        primitive_copy.metadata = self.metadata.copy()

        for k, v in self._data.data.items():
            if k not in primitive_copy._data:
                primitive_copy._data[k] = v

        return primitive_copy

    def to_mesh(self, **kwargs):
        """
        Return a copy of the Primitive object as a Trimesh.

        Parameters
        -----------
        kwargs : dict
          Passed to the Trimesh object constructor.

        Returns
        ------------
        mesh : trimesh.Trimesh
          Tessellated version of the primitive.
        """
        result = Trimesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            face_normals=self.face_normals.copy(),
            process=kwargs.pop("process", False),
            **kwargs,
        )
        return result

    def apply_transform(self, matrix):
        """
        Apply a transform to the current primitive by
        applying a new transform on top of existing
        `self.primitive.transform`. If the matrix
        contains scaling it will change parameters
        like `radius` or `height` automatically.

        Parameters
        ------------
        matrix: (4, 4) float
          Homogeneous transformation
        """
        matrix = np.asanyarray(matrix, order="C", dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix must be `(4, 4)`!")
        if util.allclose(matrix, _IDENTITY, 1e-8):
            # identity matrix is a no-op
            return self

        prim = self.primitive
        # copy the current transform
        current = prim.transform.copy()
        # see if matrix has scaling from the matrix
        scale = np.linalg.det(matrix[:3, :3]) ** (1.0 / 3.0)

        # the objects we handle re-scaling for
        # note that `Extrusion` is NOT supported
        kinds = (Box, Cylinder, Capsule, Sphere)
        if isinstance(self, kinds) and abs(scale - 1.0) > 1e-8:
            # scale the primitive attributes
            if hasattr(prim, "height"):
                prim.height *= scale
            if hasattr(prim, "radius"):
                prim.radius *= scale
            if hasattr(prim, "extents"):
                prim.extents *= scale
            # scale the translation of the current matrix
            current[:3, 3] *= scale
            # apply new matrix, rescale, translate, current
            updated = util.multi_dot([matrix, tf.scale_matrix(1.0 / scale), current])
        else:
            # without scaling just multiply
            updated = np.dot(matrix, current)

        # make sure matrix is a rigid transform
        if not tf.is_rigid(updated):
            raise ValueError("Couldn't produce rigid transform!")

        # apply the new matrix
        self.primitive.transform = updated

        return self

    def _create_mesh(self):
        raise ValueError("Primitive doesn't define mesh creation!")


class PrimitiveAttributes:
    """
    Hold the mutable data which defines a primitive.
    """

    def __init__(self, parent, defaults, kwargs, mutable=True):
        """
        Hold the attributes for a Primitive.

        Parameters
        ------------
        parent : Primitive
          Parent object reference.
        defaults : dict
          The default values for this primitive type.
        kwargs : dict
          User-passed values, i.e. {'radius': 10.0}
        """
        # store actual data in parent object
        self._data = parent._data
        # default values define the keys
        self._defaults = defaults
        # store a reference to the parent ubject
        self._parent = parent
        # start with a copy of all default objects
        self._data.update(defaults)
        # store whether this data is mutable after creation
        self._mutable = mutable
        # assign the keys passed by the user only if
        # they are a property of this primitive
        for key, default in defaults.items():
            value = kwargs.get(key, None)
            if value is not None:
                # convert passed data into type of defaults
                self._data[key] = util.convert_like(value, default)
        # make sure stored values are immutable after setting
        if not self._mutable:
            self._data.mutable = False

    @property
    def __doc__(self):
        # this is generated dynamically as the format
        # operation can be surprisingly slow and most
        # people never call it
        import pprint

        doc = (
            "Store the attributes of a {name} object.\n\n"
            + "When these values are changed, the mesh geometry will \n"
            + "automatically be updated to reflect the new values.\n\n"
            + "Available properties and their default values are:\n {defaults}"
            + "\n\nExample\n---------------\n"
            + "p = trimesh.primitives.{name}()\n"
            + "p.primitive.radius = 10\n"
            + "\n"
        ).format(
            name=self._parent.__class__.__name__,
            defaults=pprint.pformat(self._defaults, width=-1)[1:-1],
        )
        return doc

    def __getattr__(self, key):
        if key.startswith("_"):
            return super().__getattr__(key)
        elif key == "center":
            # this whole __getattr__ is a little hacky
            return self._data["transform"][:3, 3]
        elif key in self._defaults:
            return util.convert_like(self._data[key], self._defaults[key])
        raise AttributeError(f"primitive object has no attribute '{key}' ")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            return super().__setattr__(key, value)
        elif key == "center":
            value = np.array(value, dtype=np.float64)
            transform = np.eye(4)
            transform[:3, 3] = value
            self._data["transform"] = transform
            return
        elif key in self._defaults:
            if self._mutable:
                self._data[key] = util.convert_like(value, self._defaults[key])
            else:
                raise ValueError(
                    "Primitive is configured as immutable! Cannot set attribute!"
                )
        else:
            keys = list(self._defaults.keys())
            raise ValueError(f"Only default attributes {keys} can be set!")

    def __dir__(self):
        result = sorted(dir(type(self)) + list(self._defaults.keys()))
        return result


class Cylinder(Primitive):
    def __init__(self, radius=1.0, height=1.0, mass=1.0, transform=None, sections=32, mutable=True):
        """
        Create a Cylinder Primitive, a subclass of Trimesh.

        Parameters
        -------------
        radius : float
          Radius of cylinder
        height : float
          Height of cylinder
        mass : float
          Mass of cylinder
        transform : (4, 4) float
          Homogeneous transformation matrix
        sections : int
          Number of facets in circle.
        mutable : bool
          Are extents and transform mutable after creation.
        """
        super().__init__()

        defaults = {"height": 10.0, "radius": 1.0, "transform": np.eye(4), "sections": 32}
        self.primitive = PrimitiveAttributes(
            self,
            defaults=defaults,
            kwargs={
                "height": height,
                "radius": radius,
                "mass": mass,
                "transform": transform,
                "sections": sections,
            },
            mutable=mutable,
        )

    @caching.cache_decorator
    def volume(self):
        """
        The analytic volume of the cylinder primitive.

        Returns
        ---------
        volume : float
          Volume of the cylinder
        """
        volume = (np.pi * self.primitive.radius**2) * self.primitive.height
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the cylinder primitive.

        Returns
        ----------
        tensor: (3, 3) float
          3D inertia tensor
        """

        tensor = cylinder_inertia(
            mass=self.primitive.mass,
            radius=self.primitive.radius,
            height=self.primitive.height,
            transform=self.primitive.transform,
        )
        return tensor

    @caching.cache_decorator
    def direction(self):
        """
        The direction of the cylinder's axis.

        Returns
        --------
        axis: (3,) float, vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    @property
    def segment(self):
        """
        A line segment which if inflated by cylinder radius
        would represent the cylinder primitive.

        Returns
        -------------
        segment : (2, 3) float
          Points representing a single line segment
        """
        # half the height
        half = self.primitive.height / 2.0
        # apply the transform to the Z- aligned segment
        points = np.dot(
            self.primitive.transform, np.transpose([[0, 0, -half, 1], [0, 0, half, 1]])
        ).T[:, :3]
        return points

    def to_dict(self):
        """
        Get a copy of the current Cylinder primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/cylinder.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "cylinder",
            "transform": self.primitive.transform.tolist(),
            "radius": float(self.primitive.radius),
            "height": float(self.primitive.height),
        }

    def _create_mesh(self):
        log.debug("creating mesh for Cylinder primitive")
        mesh = creation.cylinder(
            radius=self.primitive.radius,
            height=self.primitive.height,
            sections=self.primitive.sections,
            transform=self.primitive.transform,
        )

        self._cache["vertices"] = mesh.vertices
        self._cache["faces"] = mesh.faces
        self._cache["face_normals"] = mesh.face_normals


class Capsule(Primitive):
    def __init__(
        self, radius=1.0, height=10.0, transform=None, sections=32, mutable=True
    ):
        """
        Create a Capsule Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius : float
          Radius of cylinder
        height : float
          Height of cylinder
        transform : (4, 4) float
          Transformation matrix
        sections : int
          Number of facets in circle
        mutable : bool
          Are extents and transform mutable after creation.
        """
        super().__init__()

        defaults = {"height": 1.0, "radius": 1.0, "transform": np.eye(4), "sections": 32}
        self.primitive = PrimitiveAttributes(
            self,
            defaults=defaults,
            kwargs={
                "height": height,
                "radius": radius,
                "transform": transform,
                "sections": sections,
            },
            mutable=mutable,
        )

    @property
    def transform(self):
        return self.primitive.transform

    def to_dict(self):
        """
        Get a copy of the current Capsule primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/capsule.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "capsule",
            "transform": self.primitive.transform.tolist(),
            "height": float(self.primitive.height),
            "radius": float(self.primitive.radius),
        }

    @caching.cache_decorator
    def direction(self):
        """
        The direction of the capsule's axis.

        Returns
        --------
        axis : (3,) float
          Vector along the cylinder axis
        """
        axis = np.dot(self.primitive.transform, [0, 0, 1, 0])[:3]
        return axis

    def _create_mesh(self):
        log.debug("creating mesh for `Capsule` primitive")

        mesh = creation.capsule(
            radius=self.primitive.radius, height=self.primitive.height
        )
        mesh.apply_transform(self.primitive.transform)

        self._cache["vertices"] = mesh.vertices
        self._cache["faces"] = mesh.faces
        self._cache["face_normals"] = mesh.face_normals


class Sphere(Primitive):
    def __init__(
        self,
        radius: Number = 1.0,
        center: Optional[ArrayLike] = None,
        transform: Optional[ArrayLike] = None,
        subdivisions: Integer = 3,
        mutable: bool = True,
    ):
        """
        Create a Sphere Primitive, a subclass of Trimesh.

        Parameters
        ----------
        radius
          Radius of sphere
        center : None or (3,) float
          Center of sphere.
        transform : None or (4, 4) float
          Full homogeneous transform. Pass `center` OR `transform.
        subdivisions
          Number of subdivisions for icosphere.
        mutable
          Are extents and transform mutable after creation.
        """

        super().__init__()

        constructor = {"radius": float(radius), "subdivisions": int(subdivisions)}
        # center is a helper method for "transform"
        # since a sphere is rotationally symmetric
        if center is not None:
            if transform is not None:
                raise ValueError("only one of `center` and `transform` may be passed!")
            translate = np.eye(4)
            translate[:3, 3] = center
            constructor["transform"] = translate
        elif transform is not None:
            constructor["transform"] = transform

        # create the attributes object
        self.primitive = PrimitiveAttributes(
            self,
            defaults={"radius": 1.0, "transform": np.eye(4), "subdivisions": 3},
            kwargs=constructor,
            mutable=mutable,
        )

    @property
    def center(self):
        return self.primitive.center

    @center.setter
    def center(self, value):
        self.primitive.center = value

    def to_dict(self):
        """
        Get a copy of the current Sphere primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/sphere.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "sphere",
            "transform": self.primitive.transform.tolist(),
            "radius": float(self.primitive.radius),
        }

    @property
    def bounds(self):
        # no docstring so will inherit Trimesh docstring
        # return exact bounds from primitive center and radius (rather than faces)
        # self.extents will also use this information
        bounds = np.array(
            [
                self.primitive.center - self.primitive.radius,
                self.primitive.center + self.primitive.radius,
            ]
        )
        return bounds

    @property
    def bounding_box_oriented(self):
        # for a sphere the oriented bounding box is the same as the axis aligned
        # bounding box, and a sphere is the absolute slowest case for the OBB calculation
        # as it is a convex surface with a ton of face normals that all need to
        # be checked
        return self.bounding_box

    @caching.cache_decorator
    def area(self):
        """
        Surface area of the current sphere primitive.

        Returns
        --------
        area: float, surface area of the sphere Primitive
        """

        area = 4.0 * np.pi * (self.primitive.radius**2)
        return area

    @caching.cache_decorator
    def volume(self):
        """
        Volume of the current sphere primitive.

        Returns
        --------
        volume: float, volume of the sphere Primitive
        """

        volume = (4.0 * np.pi * (self.primitive.radius**3)) / 3.0
        return volume

    @caching.cache_decorator
    def moment_inertia(self):
        """
        The analytic inertia tensor of the sphere primitive.

        Returns
        ----------
        tensor: (3, 3) float
          3D inertia tensor.
        """
        return sphere_inertia(mass=self.volume, radius=self.primitive.radius, transform=self.primitive.transform)

    def _create_mesh(self):
        log.debug("creating mesh for Sphere primitive")
        unit = creation.icosphere(
            subdivisions=self.primitive.subdivisions, radius=self.primitive.radius
        )

        # apply the center offset here
        self._cache["vertices"] = unit.vertices + self.primitive.center
        self._cache["faces"] = unit.faces
        self._cache["face_normals"] = unit.face_normals


class Box(Primitive):
    def __init__(self, extents=None, transform=None, mutable=True):
        """
        Create a Box Primitive as a subclass of Trimesh

        Parameters
        ----------
        extents : Optional[ndarray] (3,) float
          Length of each side of the 3D box.
        transform : Optional[ndarray] (4, 4) float
          Homogeneous transformation matrix for box center.
        mutable : bool
          Are extents and transform mutable after creation.
        """
        super().__init__()
        defaults = {"transform": np.eye(4), "extents": np.ones(3)}

        self.primitive = PrimitiveAttributes(
            self,
            defaults=defaults,
            kwargs={"extents": extents, "transform": transform},
            mutable=mutable,
        )

    def to_dict(self):
        """
        Get a copy of the current Box primitive as
        a JSON-serializable dict that matches the schema
        in `trimesh/resources/schema/box.schema.json`

        Returns
        ----------
        as_dict : dict
          Serializable data for this primitive.
        """
        return {
            "kind": "box",
            "transform": self.primitive.transform.tolist(),
            "extents": self.primitive.extents.tolist(),
        }

    @property
    def transform(self):
        return self.primitive.transform

    def sample_volume(self, count):
        """
        Return random samples from inside the volume of the box.

        Parameters
        -------------
        count : int
          Number of samples to return

        Returns
        ----------
        samples : (count, 3) float
          Points inside the volume
        """
        samples = sample.volume_rectangular(
            extents=self.primitive.extents,
            count=count,
            transform=self.primitive.transform,
        )
        return samples

    def sample_grid(self, count=None, step=None):
        """
        Return a 3D grid which is contained by the box.
        Samples are either 'step' distance apart, or there are
        'count' samples per box side.

        Parameters
        -----------
        count : int or (3,) int
          If specified samples are spaced with np.linspace
        step : float or (3,) float
          If specified samples are spaced with np.arange

        Returns
        -----------
        grid : (n, 3) float
          Points inside the box
        """

        if count is not None and step is not None:
            raise ValueError("only step OR count can be specified!")

        # create pre- transform bounds from extents
        bounds = np.array([-self.primitive.extents, self.primitive.extents]) * 0.5

        if step is not None:
            grid = util.grid_arange(bounds, step=step)
        elif count is not None:
            grid = util.grid_linspace(bounds, count=count)
        else:
            raise ValueError("either count or step must be specified!")

        transformed = tf.transform_points(grid, matrix=self.primitive.transform)
        return transformed

    @property
    def is_oriented(self):
        """
        Returns whether or not the current box is rotated at all.
        """
        if util.is_shape(self.primitive.transform, (4, 4)):
            return not np.allclose(self.primitive.transform[0:3, 0:3], np.eye(3))
        else:
            return False

    @caching.cache_decorator
    def volume(self):
        """
        Volume of the box Primitive.

        Returns
        --------
        volume : float
          Volume of box.
        """
        volume = float(np.prod(self.primitive.extents))
        return volume

    def _create_mesh(self):
        log.debug("creating mesh for Box primitive")
        box = creation.box(
            extents=self.primitive.extents, transform=self.primitive.transform
        )

        self._cache.cache.update(box._cache.cache)
        self._cache["vertices"] = box.vertices
        self._cache["faces"] = box.faces
        self._cache["face_normals"] = box.face_normals

    def as_outline(self):
        """
        Return a Path3D containing the outline of the box.

        Returns
        -----------
        outline : trimesh.path.Path3D
          Outline of box primitive
        """
        # do the import in function to keep soft dependency
        from trimesh.path.creation import box_outline

        # return outline with same size as primitive
        return box_outline(
            extents=self.primitive.extents, transform=self.primitive.transform
        )