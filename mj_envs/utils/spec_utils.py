"""MuJoCo MjSpec utility helpers.

Pure-mujoco utilities for modifying MjSpec objects — no mjlab dependency.
"""

from __future__ import annotations

import mujoco


def attach_cube_to_link(
    spec: mujoco.MjSpec,
    body_name: str,
    half_extents: tuple[float, float, float],
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    *,
    mass: float | None = None,
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
    name: str = "cube_tool",
    collisions_enabled: bool = True,
    condim: int = 3,
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
    solref: tuple[float, float] = (0.02, 1.0),
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.005, 0.5, 2),
) -> mujoco.MjsGeom:
    """Attach a box geom at a fixed pose to a named body in a MjSpec.

    The geom is rigidly fixed to the body — no additional joint is created.
    Intended for simulating end-effector tools (grippers, sensors, cameras).

    Args:
        spec:               MjSpec to modify in-place.
        body_name:          Name of the body to attach the cube to.
        half_extents:       Box half-extents (x, y, z) in metres — MuJoCo box convention.
        pos:                Position offset from the body frame origin (metres).
        quat:               Orientation quaternion (w, x, y, z) in the body frame.
        mass:               Geom mass in kg. None → MuJoCo infers from density (default 1000 kg/m³).
        rgba:               RGBA colour tuple.
        name:               Geom name; must be unique within the spec.
        collisions_enabled: If True, contype=conaffinity=1 (participates in contacts).
                            If False, contype=conaffinity=0 (visual-only, no contacts).
        condim:             Contact dimensionality (3 = full box friction). Ignored when
                            collisions_enabled=False.
        friction:           (sliding, spinning, rolling) friction coefficients.
        solref:             MuJoCo solref (timeconst, dampratio).
        solimp:             MuJoCo solimp (dmin, dmax, width, midpoint, power).

    Returns:
        The created MjsGeom.

    Raises:
        KeyError: If body_name is not found in spec.
    """
    body = spec.body(body_name)
    if body is None:
        raise KeyError(f"Body {body_name!r} not found in spec.")

    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = list(half_extents)
    geom.pos = list(pos)
    geom.quat = list(quat)
    geom.rgba = list(rgba)
    geom.condim = condim
    geom.contype = 1 if collisions_enabled else 0
    geom.conaffinity = 1 if collisions_enabled else 0
    geom.friction = list(friction)
    geom.solref = list(solref)
    geom.solimp = list(solimp)
    if mass is not None:
        geom.mass = mass

    return geom
