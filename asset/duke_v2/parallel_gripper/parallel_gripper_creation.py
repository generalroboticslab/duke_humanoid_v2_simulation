"""parallel_gripper — declarative MJCF builder (auto-generates parallel_gripper.xml).

Same pattern as asset/duke_v2/humanoid_v21/humanoid_v21_creation_v3.py: the XML is a pure build artifact,
never hand-edited. Run this to regenerate it.

    python parallel_gripper_creation.py

WHAT IS AND ISN'T IN THE GENERATED XML
--------------------------------------
The gripper is a graft module: the asset_zoo loader (mj_envs/asset_zoo/parallel_gripper.py)
loads this XML, composes the cnc_flange + per-hand AprilTag decals onto it, and adds the real
position actuator (BuiltinPositionActuatorCfg). So the generated XML carries ONLY the pieces the
loader keeps and rides onto the arm:

  - 3 bodies (base + 2 racks) with <inertial> from the Fusion-360 text (single source of truth:
    parallel_gripper_fusion_info.GRIPPER_INFO, parsed live — NOT duplicated here).
  - 2 slide joints (left/right rack), 1:1 coupled by a joint equality so one DOF drives both.
  - capsule colliders on the moving jaws and fixed base slider, emitted by the builder's native
    box->capsule approximation (names `<rack>_collisionN` and `base_collision`).

It deliberately OMITS (the loader/arm owns these, or they are preview-only):
  - floor / lights / <visual> / <statistic>  — standalone-scene furniture, never grafted.
  - the <position> actuator                   — re-added as a Builtin by the loader (gains live
                                                 in parallel_gripper.py, not the XML).
  - cnc_flange + AprilTag decals              — composed at load by the loader.

DESIGN INVARIANTS (break silently if violated)
  - Body names base/left_rack/right_rack are referenced by the loader (flange parent, tag bodies,
    EE site). Do not rename.
  - Jaw coordinate q (m): q=0 is the half-open spawn default (authored CAD pose, ref=0),
    q=-0.05 is fully open (~184 mm gap), fingers touch ~q=0.018; range [-0.05, 0.0347].
    The spawn opening lives in the consumer's keyframe, keyed on the driven joint
    left_rack_y / follower right_rack_y.
  - Jaw collision = two AXIS-ALIGNED boxes per rack, tiled into capsules along +X. Fixed slider
    collision belongs to `base`, tiled along Y. `_capsules_from_box` assumes axis-aligned boxes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).parent
# robot_builder / builder_helpers live in asset/create.
sys.path.insert(0, str(HERE.parents[1] / "create"))

from robot_builder import (  # noqa: E402
    Robot, Link, Joint, JointLimit, Geom, Origin, Box, Mesh, Material, Equality, Inertial,
    capsule_fromto,       # endpoints->Geom(Capsule) helper (arbitrary orientation, ikproxy only)
)

# Fusion-360 physics (single source of truth) — parse_fusion + GRIPPER_INFO from the asset module.
_fi_path = HERE / "parallel_gripper_fusion_info.py"
_fi_spec = importlib.util.spec_from_file_location("_pg_fusion_info", _fi_path)
_FI = importlib.util.module_from_spec(_fi_spec)
_fi_spec.loader.exec_module(_FI)

# ── tuning (NOT in CAD): slide-joint dynamics, jaw travel, collision proxies ──
LIMIT_LOW = -0.05          # m, fully open
LIMIT_HIGH = 0.0347        # m, fully closed (fingers touch)
JOINT_DAMPING = 12.0
JOINT_ARMATURE = 0.02
JAW_FRICTION = [2.0, 0.005, 0.0001]
JAW_SOLREF = [0.005, 1.0]
MESH_SCALE = [0.001, 0.001, 0.001]
IKPROXY_GROUP = 5          # geom group for IK-only collision-avoidance proxies (physics-inert)
# group 4 is taken by FOV_GEOM_GROUP (humanoid_v21_constants.py) + copy-pasted ground-plane debug
# geoms across asset_zoo/*_constants.py -- group 5 confirmed unused repo-wide, use that instead.
# RIGHT jaw = LEFT rotated 180 deg about the X-parallel axis at (Y, Z) below. The two jaws are
# ROTATED copies, NOT mirrored (confirmed: rotating the left mesh maps it exactly onto the right;
# from the Fusion inertia data the COMs map by this same rotation). A reflection would flip
# handedness and miss the Z flip -> the inner-face collision mesh would mis-align with the visual.
JAW_ROT_Y = 0.00697        # m
JAW_ROT_Z = -0.010         # m

MATERIALS = [
    Material("silver", rgba=[0.75, 0.75, 0.75, 1]),   # required by builder's hardcoded visual default class
    # Finishes match humanoid_v21_creation_v3 -- all matte (see the note there on why specular
    # is pinned low; MuJoCo's 0.5/0.5 default reads as wet plastic on these meshes).
    Material("base_mat", rgba=[0.38, 0.38, 0.38, 1], specular=0.15, shininess=0.1),
    Material("left_rack_mat", rgba=[0.30, 0.31, 0.34, 1], specular=0.05, shininess=0.05),
    Material("right_rack_mat", rgba=[0.33, 0.30, 0.34, 1], specular=0.05, shininess=0.05),
]

def _fusion_inertial(key: str):
    """(mass[kg], com[list,m], inertia[3x3]) for body `key`, CANONICAL mesh frame.

    Must go through gripper_inertial (parse_fusion + the authored-pose REPOSE_SHIFT_M on
    the rack COMs) — see the fusion_info module docstring."""
    mass, com, inertia = _FI.gripper_inertial(key)
    return mass, com.tolist(), inertia


def capsule_collider(name: str, p0, p1, radius: float, contype: int | None = None,
                      group: int | None = None) -> Geom:
    """Jaw capsule collider from two endpoints (MuJoCo `fromto` semantics).

    Thin wrapper over robot_builder.capsule_fromto (the shared endpoints->Geom geometry) that
    layers on the jaw PHYSICS: conaffinity=0 so the jaws collide with grasped objects/world but
    NOT each other or the base; JAW_FRICTION/JAW_SOLREF; collision (not visual). group/contype
    default to the builder's collision class (group 3, contype 1). Pass `contype`/`group` to
    override (e.g. the IK-only proxy capsules below use contype=0, group=5 -> physics-inert, same
    convention as the `visual` class, but still a real compiled geom so mink's
    CollisionAvoidanceLimit can query it).
    """
    return capsule_fromto(p0, p1, radius, name=name,
                          is_visual=False, is_collision=True,
                          contype=contype, conaffinity=0, group=group,
                          friction=JAW_FRICTION, solref=JAW_SOLREF)


def link(key: str, material: str, collision=None) -> Link:
    """Gripper link: visual mesh (meshes/<key>.obj) + Fusion inertial + optional colliders."""
    mass, com, inertia = _fusion_inertial(key)
    geoms = [Geom(Mesh(f"meshes/{key}.obj", scale=MESH_SCALE), material=material,
                  is_visual=True, is_collision=False)]
    geoms += collision or []
    return Link(name=key, geoms=geoms, mass=mass, inertial=Inertial(mass, com, inertia))


def rack_joint(name: str, child: str, axis) -> Joint:
    """Slide joint driving one jaw. effort/velocity are unused (no actuator in this XML) but
    JointLimit requires them; effort set to the servo force limit for documentation."""
    return Joint(name=name, parent="base", child=child, type="slide",
                 origin=Origin([0, 0, 0]), axis=axis,
                 limit=JointLimit(LIMIT_LOW, LIMIT_HIGH, 150.0, 1.0),
                 damping=JOINT_DAMPING, armature=JOINT_ARMATURE, ref=0.0)


def build_gripper() -> Robot:
    robot = Robot("parallel_gripper", materials=MATERIALS)

    # ── Moving jaw collision: 2 axis-aligned BOXES per rack, each tiled into capsules at emit by the
    # builder's use_capsule_approximation (_capsules_from_box: radius = half the shortest edge,
    # capsule axis = longest edge, single row stacked along the middle edge). Collision serves two
    # jobs: (1) grasp CONTACT (the inner pad, aligned with the visual); (2) arm-motion collision
    # (coarse coverage of the finger back/slider). The three groups (LEFT jaw, m):
    #   - PAD:         box [X 0.0816, Y 0.010, Z 0.040] -> 4 caps r0.005 along +X, stacked over Z;
    #                  zero overfill keeps every contact capsule inside this box.
    #                  +Y face at Y=-0.028 = the visual contact plane.
    #   - FINGER-BACK: box [X 0.067,  Y 0.022, Z 0.042] -> 2 caps r0.011 along +X, behind the pad.
    # Fixed base slider: 0.18 m total length, 0.052 m diameter (+30/+10 mm vs prior base capsule).
    # Center is 10 mm forward of base.obj AABB center: moving from the former (15, 7.5, 0) mm down
    # to z=-10 mm and back to x=-1.7 mm. It belongs to `base`, not either translating
    # rack, so self/world collision does not follow jaw opening.
    # RIGHT = the LEFT jaw ROTATED 180 deg about the X-axis at (JAW_ROT_Y, JAW_ROT_Z) — jaws are
    # rotated copies, NOT mirrored (so y AND z flip). For an axis-aligned box that 180-deg-X
    # rotation maps it onto an axis-aligned box at the mirrored CENTER (extents unchanged, capsules
    # axis-symmetric), so mir() on the center alone reproduces the rotated jaw; box rpy stays 0.
    _r = lambda v: round(v, 4)
    mir = lambda c: [_r(c[0]), _r(2 * JAW_ROT_Y - c[1]), _r(2 * JAW_ROT_Z - c[2])]

    #                 size,                    center
    LEFT_BOXES = [
        ([0.0816, 0.010, 0.040], [0.0808, -0.033, -0.010]),   # pad         -> 4 caps r0.005
        ([0.087,  0.042, 0.042], [0.0675, -0.049, -0.010]),   # finger-back -> 1 cap  r0.021
    ]
    BASE_SLIDER_BOX = ([0.052, 0.180, 0.052], [-0.0017, 0.0075, -0.0100])
    # finger-back = ONE capsule enclosing the former 2-cap column (r0.011 at z=-0.02, 0.0):
    # smallest enclosing circle -> r0.021 centered z=-0.01; y=-0.049 so the inner (+Y) face sits at
    # -0.028 = the pad contact plane, keeping it BEHIND the grasping face (no contact-shadow).
    # ── IK-only collision-avoidance proxy: 1 capsule/rack, covering the tip+mid finger rows of the
    # real jaw capsules above (pad = tip, finger-back = mid). Axis fixed to the rows' shared
    # orientation (local X) rather than PCA-fit -- PCA would tilt toward the tip/mid rows' 15mm
    # Y-offset instead of their true physical axis, inflating the radius needed to cover both.
    # Center/radius solved as the weighted smallest-enclosing-circle (each row = a point at its
    # perpendicular offset + its own capsule radius) in the plane perpendicular to that axis --
    # minimizes worst-case over-coverage vs the naive centroid + farthest-point bound. Length then
    # manually stretched at the base end to reach the slider's x-position (partial base coverage;
    # the slider's own axis runs perpendicular local Y and can't be captured by one straight
    # capsule without re-tilting it) and trimmed 10mm at the tip end (rounded cap still reaches past
    # the visual mesh's actual fingertip). Verified against a rendered overlay of the real geoms and
    # visual mesh (not guessed) -- mj_envs/probe/render_gripper_single_capsule_candidate.py. Shifted
    # outward, then 5 mm toward the grasp interior from the physical pad plane at Y=-0.028. Fixed
    # slider coverage is now separate on `base`; this proxy covers moving jaw rows only. group=5,
    # contype=0 -> never
    # collides physically (mirrors the `visual` class's contype=0 conaffinity=0 convention); exists
    # as a real compiled geom so mink's CollisionAvoidanceLimit can query it via mj_geomDistance.
    IKPROXY = ([0.0150, -0.0486, -0.01], [0.1066, -0.0486, -0.01], 0.0226)   # p0, p1, r

    slider_size, slider_center = BASE_SLIDER_BOX
    robot.add_link(link("base", "base_mat", collision=[
        Geom(Box(slider_size), Origin(slider_center), is_visual=False, is_collision=True,
             use_capsule_approximation=True),
    ]))

    def rack(name: str, m):
        """Build one rack link: 2 jaw boxes (m() maps center: identity for left, mir for right) +
        the ikproxy capsule (emitted LAST so the box caps get `<rack>_collision0..4` and the proxy
        keeps its explicit `<rack>_ikproxy` name, which the loader's collision regex excludes).

        Each jaw box carries the jaw PHYSICS (conaffinity=0 so jaws collide with grasped
        objects/world but NOT each other or the base; JAW_FRICTION/JAW_SOLREF ride on the Geom, the
        emitter passes them through to every tiled capsule) and use_capsule_approximation=True so
        the builder's _capsules_from_box fills it (radius = half shortest edge, axis = longest edge,
        row stacked along the middle edge)."""
        p0, p1, r = IKPROXY
        return link(name, f"{name}_mat", collision=(
            [Geom(Box(size), Origin(m(center)), is_visual=False, is_collision=True, conaffinity=0,
                  friction=JAW_FRICTION, solref=JAW_SOLREF, use_capsule_approximation=True,
                  capsule_grid=(1, 4) if i == 0 else None,
                  capsule_overfill_ratio=0.0)
             for i, (size, center) in enumerate(LEFT_BOXES)]
            + [capsule_collider(f"{name}_ikproxy", m(p0), m(p1), r, contype=0, group=IKPROXY_GROUP)]
        ))

    robot.add_link(rack("left_rack", lambda c: c))
    robot.add_link(rack("right_rack", mir))

    robot.add_joint(rack_joint("left_rack_y", "left_rack", [0, 1, 0]))
    robot.add_joint(rack_joint("right_rack_y", "right_rack", [0, -1, 0]))
    return robot


def export():
    # inner grasping face is now a box->capsule fill (build_gripper), no mesh hull to generate.
    robot = build_gripper()
    # 1:1 mimic: right_rack_y = left_rack_y (one servo drives both jaws via rack-and-pinion).
    coupling = Equality(joint1="right_rack_y", joint2="left_rack_y",
                        polycoef=[0, 1, 0, 0, 0], solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001])
    xml = robot.to_mjcf_string(
        add_freejoint=False,      # graft module: the arm owns the floating base
        add_light_camera=False,   # no preview furniture
        imu_site_name=None,       # no IMU
        actuators=[],             # loader re-adds the position servo as a Builtin
        equalities=[coupling],
    )
    out = HERE / "parallel_gripper.xml"
    out.write_text(xml)
    print(f"wrote {out}")


if __name__ == "__main__":
    export()
