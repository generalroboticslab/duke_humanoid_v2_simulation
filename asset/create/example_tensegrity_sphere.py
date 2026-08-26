"""Programmatically build the Argus v1.1 12-bar tensegrity sphere as a MuJoCo robot.

Uses the declarative builder in robot_builder.py (Robot / Link / Joint / Geom -> export_mjcf). The
ANALYTICAL geometry (12 twisted-cuboctahedron bars, motor-inside roll) is imported UNCHANGED from the
viser design tool plot_viser_tensegrity12_symball.py, so this MJCF tracks the same RHO/TAU/ROLL the
interactive search converged to (motors inside, R = 281 mm). Nothing in the design tools is modified.

STRUCTURE (one free body + 12 prismatic DOF):
  core            free-floating hub. Carries the SHELL collision sphere (the body envelope) and a tiny
                  visual hub for mass.
  stator{i}       welded (fixed joint) to core at bar midpoint m_i, oriented by _leg_basis(m_i,a_i):
                  guide cylinder + pinion motor + motor housing + drum disk, ALL visual-only.
  rod{i}          slide joint along the bar axis (local +Z), range +-STROKE: rod cylinder + a foot ball
                  at each end.

COLLISION MODEL (sim-speed constraint from the user):
  ONLY the foot balls and the core SHELL sphere are collision geoms (class "collision", contype=1).
  Every stator / rod / drum / motor geom is visual-only (contype=0) -> the broadphase sees just
  1 shell + 24 feet instead of ~70 bodies. Because the builder forbids a geom being both visual and
  collision, each foot (and the shell) is emitted as a co-located pair: a visual sphere (gives mass +
  render) and a collision sphere (gives contact).

Run:  python example_tensegrity_sphere.py
      -> writes argus_tensegrity/argus_tensegrity12.xml, then compiles + steps it to validate.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARGUS = os.path.join(_HERE, "..", "argus_tensegrity")
sys.path.insert(0, _HERE)
sys.path.insert(0, _ARGUS)

from robot_builder import (  # noqa: E402  (path set above)
    Actuator, Cylinder, Geom, Inertial, Joint, JointLimit, Link, Material, Origin, Robot, Sphere,
)
# Single source of truth for the analytical layout + the validated motors-inside params.
from plot_viser_tensegrity12_symball import (  # noqa: E402
    DRUM_OFFSET, DRUM_R, HOUSING_HALF, HOUSING_R, MHOUSING_OFFSET, MHOUSING_R, MOTOR_OFFSET, MOTOR_R,
    RHO, ROD_LEN, ROD_RADIUS, STROKE, TAU, _leg_basis, _struts,
)
from plot_viser_symball import FOOT_R, FOOT_Z  # noqa: E402

# Fixed hardware lengths (cylinder heights) taken from add_symball_module's render; not exported as
# constants there, so restated here as the part dimensions.
GUIDE_LEN = 0.080        # rod-guide housing cylinder height
MOTOR_LEN = 0.051        # pinion motor cylinder height
MHOUSING_LEN = 0.056     # motor housing cylinder height
DRUM_LEN = 0.010         # drum disk height
HUB_R = 0.020            # tiny central hub sphere (core mass)

# Shell collision sphere: the body envelope, = housing-cluster outer radius (midpoint + housing OD).
# Feet stick out past this, so feet are the ground contacts and the shell is the body-down contact.
SHELL_R = RHO + MHOUSING_R

# ── Mass budget (explicit per-Link mass; geom geometry only sets inertia shape) ──
# Geoms default to density 1000 (water), so the solid r=0.164 shell sphere alone
# auto-derived ~18.5 kg of phantom mass. The real robot is an open cage with a
# printed structural shell, not a water ball. Link.mass overrides each link's
# total to the real component sum and rescales the geom-derived inertia tensor to
# match (robot_builder). For the rolling shell, lumping the full core mass onto
# the solid-sphere geom (I=2/5 m R^2) approximates the true hollow-shell+central
# inertia within ~12% — closer than per-geom density (which would apply 2/5 to the
# shell-only mass and badly underestimate the 2/3 m R^2 thin-shell term).

# Printed shell: hollow CF-PLA sphere, 10 mm wall, 80% infill. Mass from the wall
# volume so it tracks SHELL_R automatically.
SHELL_WALL      = 0.010  # m printed wall thickness
CFPLA_DENSITY   = 1240.0 # kg/m^3 carbon-fiber PLA (solid)
SHELL_INFILL    = 0.80   # 80% infill
SHELL_MASS      = ((4.0 / 3.0) * np.pi
                   * (SHELL_R ** 3 - (SHELL_R - SHELL_WALL) ** 3)
                   * CFPLA_DENSITY * SHELL_INFILL)        # ~3.15 kg

BATTERY_MASS    = 1.0    # kg
CONTROLLER_MASS = 0.5    # kg
CORE_MASS       = BATTERY_MASS + CONTROLLER_MASS + SHELL_MASS   # ~4.65 kg

# Core inertia computed analytically rather than lumped onto the solid-sphere geom
# (which would apply the solid 2/5 m R^2 form and under-count the rolling inertia
# by ~12%). The shell is a thick hollow sphere; the battery+controller are a
# compact payload near the center (estimated radius of gyration 0.05 m). Isotropic.
_R_OUT = SHELL_R
_R_IN  = SHELL_R - SHELL_WALL
_I_SHELL   = 0.4 * SHELL_MASS * (_R_OUT ** 5 - _R_IN ** 5) / (_R_OUT ** 3 - _R_IN ** 3)
_PAYLOAD_RGYR = 0.05     # m radius of gyration of packed battery+controller
_I_PAYLOAD = 0.4 * (BATTERY_MASS + CONTROLLER_MASS) * _PAYLOAD_RGYR ** 2
CORE_INERTIA = (_I_SHELL + _I_PAYLOAD) * np.eye(3)             # ~0.0547 kg·m² isotropic

MOTOR_MASS      = 0.31   # kg RobStride 00 (asset/argus_v2/argus_v2_actuator_plan.md)
CONNECTOR_MASS  = 0.2    # kg per-motor wiring/bracket
STATOR_MASS     = MOTOR_MASS + CONNECTOR_MASS            # 0.51 kg (stator welded to core)

ROD_CARBON_MASS = 0.025  # kg hollow 14x16 carbon tube (actuator plan)
FOOT_MASS       = 0.03   # kg each TPU/foam contact pad (2 per rod)
ROD_ASSY_MASS   = ROD_CARBON_MASS + 2.0 * FOOT_MASS      # 0.085 kg moving slide assembly

_AX_X = [0.0, np.pi / 2.0, 0.0]   # rpy that points a builder cylinder's local +Z along local +X


def _mat_to_rpy(R):
    """Rotation matrix (columns = local x,y,z in world) -> URDF/builder rpy [roll,pitch,yaw]."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return [roll, pitch, yaw]


def build():
    """Assemble and return (Robot, actuators) for the 12-bar tensegrity sphere."""
    mats = [
        Material(name="silver", rgba=[0.7, 0.7, 0.75, 1.0]),
        Material(name="pink", rgba=[1.0, 0.45, 0.65, 1.0]),
        Material(name="purple", rgba=[0.6, 0.4, 0.9, 1.0]),
        Material(name="green", rgba=[0.4, 0.8, 0.45, 0.6]),
        Material(name="orange", rgba=[1.0, 0.6, 0.2, 0.9]),
        Material(name="shell", rgba=[0.4, 0.45, 0.65, 0.15]),
    ]
    robot = Robot("argus_tensegrity12", materials=mats)

    # ── core: hub mass + shell collision envelope ────────────────────────────
    core = Link(name="core",
                inertial=Inertial(mass=CORE_MASS, com=[0.0, 0.0, 0.0], inertia=CORE_INERTIA),
                geoms=[
        Geom(Sphere(HUB_R), name="hub", material="silver", is_visual=True),
        Geom(Sphere(SHELL_R), name="shell_vis", material="shell", is_visual=True),
        Geom(Sphere(SHELL_R), name="shell_col", is_collision=True, is_visual=False),
    ])
    robot.add_link(core)

    actuators = []
    for i, (m, a) in enumerate(_struts()):
        rpy = _mat_to_rpy(_leg_basis(m, a))

        # stator: welded to core at the bar midpoint, oriented along the bar axis. Visual-only.
        stator = Link(name=f"stator{i}", mass=STATOR_MASS, geoms=[
            Geom(Cylinder(HOUSING_R, GUIDE_LEN), name=f"guide{i}", material="silver", is_visual=True),
            Geom(Cylinder(MOTOR_R, MOTOR_LEN), origin=Origin(xyz=MOTOR_OFFSET.tolist(), rpy=_AX_X),
                 name=f"motor{i}", material="pink", is_visual=True),
            Geom(Cylinder(MHOUSING_R, MHOUSING_LEN),
                 origin=Origin(xyz=MHOUSING_OFFSET.tolist(), rpy=_AX_X),
                 name=f"mhousing{i}", material="silver", is_visual=True),
            Geom(Cylinder(DRUM_R, DRUM_LEN), origin=Origin(xyz=DRUM_OFFSET.tolist(), rpy=_AX_X),
                 name=f"drum{i}", material="purple", is_visual=True),
        ])
        robot.add_link(stator)
        robot.add_joint(Joint(name=f"stator{i}_fix", parent="core", child=f"stator{i}",
                              type="fixed", origin=Origin(xyz=m.tolist(), rpy=rpy)))

        # rod: prismatic along the bar axis (local +Z), foot ball at each end (visual + collision).
        rod = Link(name=f"rod{i}", mass=ROD_ASSY_MASS, geoms=[
            Geom(Cylinder(ROD_RADIUS, ROD_LEN), name=f"rod{i}", material="green", is_visual=True),
            Geom(Sphere(FOOT_R), origin=Origin(xyz=[0, 0, FOOT_Z]), name=f"foot_p{i}",
                 material="orange", is_visual=True),
            Geom(Sphere(FOOT_R), origin=Origin(xyz=[0, 0, FOOT_Z]), name=f"foot_p{i}_col",
                 is_collision=True, is_visual=False),
            Geom(Sphere(FOOT_R), origin=Origin(xyz=[0, 0, -FOOT_Z]), name=f"foot_n{i}",
                 material="orange", is_visual=True),
            Geom(Sphere(FOOT_R), origin=Origin(xyz=[0, 0, -FOOT_Z]), name=f"foot_n{i}_col",
                 is_collision=True, is_visual=False),
        ])
        robot.add_link(rod)
        robot.add_joint(Joint(name=f"slide{i}", parent=f"stator{i}", child=f"rod{i}",
                              type="slide", axis=[0, 0, 1], origin=Origin(xyz=[0, 0, 0]),
                              limit=JointLimit(lower=-STROKE, upper=STROKE, effort=100.0,
                                               velocity=10.0)))
        actuators.append(Actuator(name=f"act{i}", target_joint=f"slide{i}", type="motor",
                                  ctrlrange=[-100.0, 100.0]))

    return robot, actuators


def main():
    robot, actuators = build()
    out = os.path.join(_ARGUS, "argus_tensegrity12.xml")
    robot.export_mjcf(out, actuators=actuators)
    print(f"[write] {out}")
    print(f"[spec]  ROD={ROD_LEN*1e3:.0f}mm RHO={RHO*1e3:.0f}mm TAU={np.rad2deg(TAU):.0f}deg "
          f"shell R={SHELL_R*1e3:.0f}mm  STROKE=+-{STROKE*1e3:.0f}mm")

    import mujoco  # validate: compile + step
    model = mujoco.MjModel.from_xml_path(out)
    data = mujoco.MjData(model)
    n_col = sum(1 for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g])
    print(f"[mjcf]  nbody={model.nbody} njnt={model.njnt} nu={model.nu} ngeom={model.ngeom} "
          f"collision_geoms={n_col} (expect 1 shell + 24 feet = 25)")
    for _ in range(200):
        mujoco.mj_step(model, data)
    print(f"[step]  200 steps OK, core z={data.qpos[2]:+.4f} m")


if __name__ == "__main__":
    main()
