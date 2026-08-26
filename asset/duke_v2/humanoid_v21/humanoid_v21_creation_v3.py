"""Humanoid V2.1 Robot Creation Script - Human-Friendly Version

This script implements the improved robot creation workflow for the Humanoid V2.1.
It focuses on making robot creation easy to read and adjust for humans while
eliminating manual data entry errors and improving accuracy.

ARCHITECTURE:
1. asset/create/fusion_info.py: Fusion360 export strings + parser + PHYSICS_DB.
2. asset/create/builder_helpers.py: Human-friendly wrappers for link/joint creation and mirroring.
3. humanoid_v21_creation_v3.py: This script - the clean, section-by-section assembly.

The builder library lives in asset/create/ (shared with argus / ball_circle / duke_v2's other
components); only this robot's own data lives here. See asset/duke_v2/humanoid_v21/README.md.

USAGE:
    # Single robot export (humanoid_v21.xml + humanoid_v21_high_res.xml)
    python asset/duke_v2/humanoid_v21/humanoid_v21_creation_v3.py

    # Custom output path
    python asset/duke_v2/humanoid_v21/humanoid_v21_creation_v3.py --output my_robot.xml

    # How to modify:
    - Change joint limits: Find the joint in build_legs() or build_arms()
    - Change collision: Find the link and update the Geom(Box/Cylinder/...)
    - Add a sensor: Add to the sensors list in export_single()

27 DOF humanoid: waist (1), legs (6x2), arms (7x2)
"""

import argparse
import mujoco
import numpy as np
import os
import sys

# The builder library is a sibling of this robot's directory, not a sibling of this file, and
# it is imported bare (`from builder_helpers import ...`) the same way argus / ball_circle /
# parallel_gripper import it. Same idiom as parallel_gripper_creation.py.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "create"))

from builder_helpers import (
    link_from_fusion, simple_link, simple_joint,
    mirror_link_about_plane, deg,
    STANDARD_MATERIALS
)
from robot_builder import (
    Robot, Link, Joint, JointLimit, Geom, Origin, Box, Cylinder, Capsule, Sphere,
    Sensor, Actuator, Site, Material
)

# Adjacent links alternate light_grey / dark_grey (the builder default). The shared
# STANDARD_MATERIALS shades (0.40 / 0.30) are too close to tell apart in render, so override
# both to a wider spread HERE ONLY -- STANDARD_MATERIALS itself is shared with argus /
# ball_circle / duke_v2 and must not be re-toned for them.
#
# Both finishes are MATTE. MuJoCo's material default (specular 0.5 / shininess 0.5) puts a
# hard plastic-looking highlight on every mesh; an earlier pass here went the other way and
# made the light shade a high-specular machined-aluminium, which blew out under the headlight.
# The parts are bead-blasted alu and anodized housings -- both diffuse. Specular is therefore
# pinned LOW on both, and the small remaining gap only softens the light links' edge falloff.
# Link separation is carried by the brightness gap (0.20 vs 0.12), not by finish.
#
# Tone is dark graphite, not mid grey: hard-anodized 6061 measures ~0.2 diffuse albedo and
# reads slightly COOL (blue channel a touch above red), so the values are neutral-cool rather
# than flat equal-channel. The earlier 0.50/0.30 pair was above any real anodized finish and
# washed out under the tracking headlight.
# The Blender photoreal grade reads mat_specular as Principled *Metallic* and
# (1 - mat_shininess) as *Roughness* (mj_envs/photoreal/blender_view.py). Anodized alu is a
# metal with a bead-blasted finish, so that is metallic ~0.75 + roughness ~0.95, i.e.
# specular 0.75 / shininess 0.05. Shininess stays near zero on both finishes: in MuJoCo's own
# viewer it is the highlight exponent, so a low value keeps the sheen broad and dull instead of
# the hard plastic dot the 0.5/0.5 default gives.
_METAL = dict(specular=0.75, shininess=0.05)
_SHELL = dict(specular=0.35, shininess=0.05)
MATERIALS = [m for m in STANDARD_MATERIALS if m.name not in ("light_grey", "dark_grey")] + [
    Material("light_grey", rgba=[0.19, 0.20, 0.22, 1], **_METAL),
    Material("dark_grey", rgba=[0.11, 0.115, 0.125, 1], **_SHELL),
]

# Per-arm EE site offsets in wrist_3 local frame. Define independently — L and R may diverge.
# x = 0.0325 m is the GEOMETRIC CENTER of the 5-face AprilTag wrist cube (tag_cube_0/1)
# once mounted via its flange adapter bracket, measured from the wrist_3 / end_effector link
# origin (hand-measured: ~12.5 mm adapter + 20 mm half of the 40 mm cube = 32.5 mm). This is
# the physically meaningful bare-wrist end-effector reference for the tag-cube configuration.
# When the parallel_gripper is attached instead, the pipeline's place_ee_site_at_grasp_center
# relocates this site to the gripper grasp center (x≈0.1176), so this value only takes effect
# on the hand-less ("builtin") build.
EE_SITE_OFFSET_L = [0.0325, 0, 0]
EE_SITE_OFFSET_R = [0.0325, 0, 0]


# Toggle for high-res visual meshes
is_high_res = True
# is_high_res = False
hig_res_str="_high_res" if is_high_res else ""


# ════
#  TORSO (Base + Waist)
# ════

def build_torso():
    """Build base_link and waist with joint"""
    links = []
    joints = []

    # ── Base Link ──────────────────────────────────────────────
    base_link = link_from_fusion("base_link",
        visual_mesh="meshes/base_link.stl",
        collision=[
            # # Torso collision box
            # Geom(Box([0.13, 0.18, 0.412]),
            #      Origin([0, 0, 0.206]),
            #      is_visual=False, is_collision=True,use_capsule_approximation=True),
            
            Geom(Box([0.12, 0.17, 0.48]), Origin([0, 0, 0.20]),
                 is_visual=False, is_collision=True, use_capsule_approximation=True,
                 capsule_grid=(1, 2), capsule_overfill_ratio=1.0),

            # Geom(Capsule(radius=0.07, length=0.36),
            #      Origin([0, -0.03, 0.16]),
            #      is_visual=False, is_collision=True
            # ),
            # Geom(Capsule(radius=0.07, length=0.36),
            #      Origin([0, 0.03, 0.16]),
            #      is_visual=False, is_collision=True
            # ),


            # # Waist motor (collision disabled)
            # Geom(Cylinder(0.053, 0.0361),
            #      Origin([0, 0, -0.01805]),
            #      is_visual=False, is_collision=False)
        ]
    )
    links.append(base_link)

    # ── Waist ──────────────────────────────────────────────────
    waist = link_from_fusion("waist",
        visual_mesh=f"meshes/waist{hig_res_str}.obj",
        collision=[
            # # Left hip motor
            # Geom(Capsule(0.053, 0.042),
            #      Origin([0, 0.039649, -0.0762], [deg(75), 0, 0]),
            #      is_visual=False, is_collision=True),
            # # Right hip motor
            # Geom(Capsule(0.053, 0.042),
            #      Origin([0, -0.039649, -0.0762], [deg(-75), 0, 0]),
            #      is_visual=False, is_collision=True)
        ]
    )
    links.append(waist)

    # ── Waist Joint ────────────────────────────────────────────
    waist_joint = simple_joint("waist_joint",
        parent="base_link", child="waist",
        xyz=[0, 0, -0.0361], rpy=[0, 0, 0],
        lower=-np.pi/2, upper=np.pi/2
    )
    joints.append(waist_joint)

    return links, joints


# ════
#  LEGS (both sides, flat — neither side derived from the other)
# ════

def build_legs():
    """Build both legs (12 links, 12 joints).

    Written flat, segment by segment down the kinematic chain: left link, right link,
    left joint, right joint — same style as build_arms(). No mirroring helper, so every
    number is at its use site and the two legs can diverge freely.

    Replaced a mirror_link/mirror_joint pair whose right leg was derived from the left
    via an index-keyed `rpy_adjustments` table plus two implicit rules: only joint index
    0 flips its xyz Y, and right_hip_2's asymmetric limits were stamped on *after*
    mirroring. Both rules were comment-only invariants that a reordering of the segments
    would have broken silently. Rejected keeping the mirror and adding asserts: the
    per-side rpy values are already literals in that table, so flattening costs nothing
    but line count and removes the patch-after-mirror step entirely.

    Only these differ between sides; everything else is identical by construction:
      hip_1 xyz y sign, hip_1/hip_2/hip_3/ankle_1/ankle_2 joint rpy, hip_2 limits.
    knee is identical on both sides.

    Emission order of these lists does NOT set the body/qpos layout — JOINT_ORDER in
    export_single() ranks siblings and to_mjcf_string asserts the realized DFS order.

    Returns:
        (links, joints)
    """
    links = []
    joints = []

    # ── Hip 1 (Roll) ───────────────────────────────────────────
    hip_2_L = mirror_link_about_plane(link_from_fusion("hip_2",
        visual_mesh=f"meshes/hip_2{hig_res_str}.obj",
        # collision=Geom(Cylinder(0.053, 0.0612),Origin([0.0312 - 0.0612/2, 0, -0.069],[deg(90), deg(-75), deg(-90)]),
        collision=Geom(Capsule(0.06, 0.01),Origin([0.0312 - 0.0612/2, 0, -0.069],[deg(90), deg(-75), deg(-90)]),
                      is_visual=False, is_collision=True,use_capsule_approximation=True),
        name_override="hip_2_L",
        visual_material="light_grey"
    ), normal=[0, 1, 0])
    hip_2_R = link_from_fusion("hip_2",
        visual_mesh=f"meshes/hip_2{hig_res_str}.obj",
        collision=Geom(Capsule(0.06, 0.01),Origin([0.0312 - 0.0612/2, 0, -0.069],[deg(90), deg(-75), deg(-90)]),
                      is_visual=False, is_collision=True,use_capsule_approximation=True),
        name_override="hip_2_R",
        visual_material="light_grey"
    )
    links += [hip_2_L, hip_2_R]

    left_hip_1_joint = simple_joint("left_hip_1_joint",
        parent="waist", child="hip_2_L",
        xyz=[0, 0.059946, -0.081628], rpy=[deg(75), 0, 0],
        lower=deg(-105), upper=deg(105)
    )
    right_hip_1_joint = simple_joint("right_hip_1_joint",
        parent="waist", child="hip_2_R",
        xyz=[0, -0.059946, -0.081628], rpy=[deg(-75), 0, 0],
        lower=deg(-105), upper=deg(105)
    )
    joints += [left_hip_1_joint, right_hip_1_joint]

    # ── Hip 2 (Pitch) ──────────────────────────────────────────
    hip_3_L = mirror_link_about_plane(link_from_fusion("hip_3",
        visual_mesh=f"meshes/hip_3{hig_res_str}.obj",
        collision=Geom(Capsule(0.053, 0.042),
                      Origin([0, -0.0945, 0.0287], [deg(-90), 0, 0]),
                      is_visual=False, is_collision=True),
        name_override="hip_3_L"
    ), normal=[1, 0, 0])
    hip_3_R = link_from_fusion("hip_3",
        visual_mesh=f"meshes/hip_3{hig_res_str}.obj",
        collision=Geom(Capsule(0.053, 0.042),
                      Origin([0, -0.0945, 0.0287], [deg(-90), 0, 0]),
                      is_visual=False, is_collision=True),
        name_override="hip_3_R"
    )
    links += [hip_3_L, hip_3_R]

    # Asymmetric limits: pitch swings forward on both sides, so the sign convention flips.
    left_hip_2_joint = simple_joint("left_hip_2_joint",
        parent="hip_2_L", child="hip_3_L",
        xyz=[0.0287, 0, -0.0691], rpy=[deg(90), deg(-75), deg(-90)],
        lower=deg(-105), upper=deg(30)
    )
    right_hip_2_joint = simple_joint("right_hip_2_joint",
        parent="hip_2_R", child="hip_3_R",
        xyz=[0.0287, 0, -0.0691], rpy=[deg(90), deg(75), deg(-90)],
        lower=deg(-30), upper=deg(105)
    )
    joints += [left_hip_2_joint, right_hip_2_joint]

    # ── Hip 3 (Yaw) ────────────────────────────────────────────
    knee_L = mirror_link_about_plane(link_from_fusion("knee",  # "knee" in fusion_info is actually hip_3
        visual_mesh=f"meshes/knee{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.06, 0.01),
            #Cylinder(0.06, 0.0555), # original,do not remove
                      Origin([0.0255 - 0.0555/2, 0, -0.0795],
                            [deg(-90), 0, deg(90)]),
                      is_visual=False, is_collision=True),
        name_override="knee_L",
        visual_material="light_grey"
    ), normal=[0, 1, 0])
    knee_R = link_from_fusion("knee",
        visual_mesh=f"meshes/knee{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.06, 0.01),
                      Origin([0.0255 - 0.0555/2, 0, -0.0795],
                            [deg(-90), 0, deg(90)]),
                      is_visual=False, is_collision=True),
        name_override="knee_R",
        visual_material="light_grey"
    )
    links += [knee_L, knee_R]

    left_hip_3_joint = simple_joint("left_hip_3_joint",
        parent="hip_3_L", child="knee_L",
        xyz=[0, -0.1156, 0.0287], rpy=[deg(-90), 0, 0],
        lower=deg(-90), upper=deg(90)
    )
    right_hip_3_joint = simple_joint("right_hip_3_joint",
        parent="hip_3_R", child="knee_R",
        xyz=[0, -0.1156, 0.0287], rpy=[deg(-90), deg(180), 0],
        lower=deg(-90), upper=deg(90)
    )
    joints += [left_hip_3_joint, right_hip_3_joint]

    # ── Knee ───────────────────────────────────────────────────
    shank_L = mirror_link_about_plane(link_from_fusion("shank",
        visual_mesh=f"meshes/shank{hig_res_str}.obj",
        collision=[
            # Knee guard top (with capsule approximation)
            Geom(Box([0.06, 0.18, 0.02]),
                 Origin([0, 0.1, -0.025]),
                 is_visual=False, is_collision=True,
                 use_capsule_approximation=True),
            # Knee guard bottom
            Geom(Box([0.06, 0.18, 0.02]),
                 Origin([0, 0.1, 0.075]),
                 is_visual=False, is_collision=True,
                 use_capsule_approximation=True),
        ],
        name_override="shank_L"
    ), normal=[1, 0, 0])
    shank_R = link_from_fusion("shank",
        visual_mesh=f"meshes/shank{hig_res_str}.obj",
        collision=[
            Geom(Box([0.06, 0.18, 0.02]),
                 Origin([0, 0.1, -0.025]),
                 is_visual=False, is_collision=True,
                 use_capsule_approximation=True),
            Geom(Box([0.06, 0.18, 0.02]),
                 Origin([0, 0.1, 0.075]),
                 is_visual=False, is_collision=True,
                 use_capsule_approximation=True),
        ],
        name_override="shank_R"
    )
    links += [shank_L, shank_R]

    # Knee is the one segment whose joint frame is identical on both sides.
    left_knee_joint = simple_joint("left_knee_joint",
        parent="knee_L", child="shank_L",
        xyz=[0.0255, 0, -0.0795], rpy=[deg(-90), 0, deg(90)],
        effort=120,  # Knee has higher torque limit
        lower=deg(-130), upper=deg(130)
    )
    right_knee_joint = simple_joint("right_knee_joint",
        parent="knee_R", child="shank_R",
        xyz=[0.0255, 0, -0.0795], rpy=[deg(-90), 0, deg(90)],
        effort=120,
        lower=deg(-130), upper=deg(130)
    )
    joints += [left_knee_joint, right_knee_joint]

    # ── Ankle 1 (Pitch) ────────────────────────────────────────
    ankle_1_L = mirror_link_about_plane(link_from_fusion("ankle_1",
        visual_mesh=f"meshes/ankle_1{hig_res_str}.obj",
        collision=Geom(Capsule(0.044, 0.001), # Cylinder(0.044, 0.0345),
                      Origin([-0.1 + 0.0345/2, 0, -0.0255],
                            [0, deg(90), 0]),
                      is_visual=False, is_collision=True),
        name_override="ankle_1_L",
        visual_material="light_grey"
    ), normal=[0, 1, 0])
    ankle_1_R = link_from_fusion("ankle_1",
        visual_mesh=f"meshes/ankle_1{hig_res_str}.obj",
        collision=Geom(Capsule(0.044, 0.001),
                      Origin([-0.1 + 0.0345/2, 0, -0.0255],
                            [0, deg(90), 0]),
                      is_visual=False, is_collision=True),
        name_override="ankle_1_R",
        visual_material="light_grey"
    )
    links += [ankle_1_L, ankle_1_R]

    left_ankle_1_joint = simple_joint("left_ankle_1_joint",
        parent="shank_L", child="ankle_1_L",
        xyz=[0, 0.195, 0.0], rpy=[0, deg(180), 0],
        lower=deg(-50), upper=deg(50)
    )
    right_ankle_1_joint = simple_joint("right_ankle_1_joint",
        parent="shank_R", child="ankle_1_R",
        xyz=[0, 0.195, 0.0], rpy=[0, deg(180), deg(180)],
        lower=deg(-50), upper=deg(50)
    )
    joints += [left_ankle_1_joint, right_ankle_1_joint]

    # ── Ankle 2 (Roll) - Foot ──────────────────────────────────
    foot_L = mirror_link_about_plane(link_from_fusion("ankle_2",
        visual_mesh=f"meshes/ankle_2{hig_res_str}.obj",
        collision=Geom(Box([0.014, 0.072, 0.235]), # width 0.067->0.072 to match the visual sole (72mm; was 2.5mm/side narrow); original: Box([0.014, 0.067, 0.187])
                      Origin([-0.061, 0, 0.100]), # original: Origin([-0.061, 0, 0.076])
                      is_visual=False, is_collision=True,
                      use_capsule_approximation=True),
        name_override="foot_L",
        visual_material="dark_grey"
    ), normal=[0, 1, 0])
    foot_R = link_from_fusion("ankle_2",
        visual_mesh=f"meshes/ankle_2{hig_res_str}.obj",
        collision=Geom(Box([0.014, 0.072, 0.235]),
                      Origin([-0.061, 0, 0.100]),
                      is_visual=False, is_collision=True,
                      use_capsule_approximation=True),
        name_override="foot_R",
        visual_material="dark_grey"
    )
    links += [foot_L, foot_R]

    left_ankle_2_joint = simple_joint("left_ankle_2_joint",
        parent="ankle_1_L", child="foot_L",
        xyz=[-0.1, 0, -0.0255], rpy=[deg(-90), 0, deg(-90)],
        effort=27,  # Ankle has lower torque limit
        lower=deg(-60), upper=deg(60)
    )
    right_ankle_2_joint = simple_joint("right_ankle_2_joint",
        parent="ankle_1_R", child="foot_R",
        xyz=[-0.1, 0, -0.0255], rpy=[deg(90), 0, deg(90)],
        effort=27,
        lower=deg(-60), upper=deg(60)
    )
    joints += [left_ankle_2_joint, right_ankle_2_joint]

    return links, joints


# ════
#  ARMS (both sides, flat — neither side derived from the other)
# ════

def build_arms():
    """Build both arms (14 links, 14 joints).

    Written flat, segment by segment down the kinematic chain: left link, right link,
    left joint, right joint. No mirroring helper — every number is at its use site, so
    the two arms can diverge freely (different collision, limits, EE hardware) without
    unpicking a mirror + patch chain.

    Only these differ between sides; everything else is identical by construction:
      shoulder_1 xyz y sign, shoulder_1/2/3 joint rpy, shoulder_2 limits, EE site offset.

    Emission order of these lists does NOT set the body/qpos layout — JOINT_ORDER in
    export_single() ranks siblings and to_mjcf_string asserts the realized DFS order.

    Returns:
        (links, joints)
    """
    links = []
    joints = []

    # ── Shoulder 1 (Roll) ──────────────────────────────────────
    shoulder_2_L = mirror_link_about_plane(link_from_fusion("shoulder_2",
        visual_mesh=f"meshes/shoulder_2{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.053, 1e-4),
            # Cylinder(0.053, 0.058), # original, do not remove
                      Origin([0, 0, -0.090], [0, deg(90), 0]),
                      is_visual=False, is_collision=True),
        name_override="shoulder_2_L"
    ), normal=[0, 1, 0])
    shoulder_2_R = link_from_fusion("shoulder_2",
        visual_mesh=f"meshes/shoulder_2{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.053, 1e-4),
                      Origin([0, 0, -0.090], [0, deg(90), 0]),
                      is_visual=False, is_collision=True),
        name_override="shoulder_2_R"
    )
    links += [shoulder_2_L, shoulder_2_R]

    left_shoulder_1_joint = simple_joint("left_shoulder_1_joint",
        parent="base_link", child="shoulder_2_L",
        xyz=[0, 0.083, 0.3475], rpy=[deg(90), 0, 0],
        lower=deg(-180), upper=deg(180)
    )
    right_shoulder_1_joint = simple_joint("right_shoulder_1_joint",
        parent="base_link", child="shoulder_2_R",
        xyz=[0, -0.083, 0.3475], rpy=[deg(-90), 0, 0],
        lower=deg(-180), upper=deg(180)
    )
    joints += [left_shoulder_1_joint, right_shoulder_1_joint]

    # ── Shoulder 2 (Pitch) ─────────────────────────────────────
    shoulder_3_L = mirror_link_about_plane(link_from_fusion("shoulder_3_long",
        visual_mesh=f"meshes/shoulder_3{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.042, 0.02),  # extended down from length 1e-4; origin shifted -Y to keep top anchored
            # Cylinder(0.0785/2, 0.033), # original, do not remove
                      Origin([0, -0.093, 0.02475], [deg(90), 0, 0]),
                      is_visual=False, is_collision=True),
        name_override="shoulder_3_L",
        visual_material="light_grey"
    ), normal=[1, 0, 0])
    shoulder_3_R = link_from_fusion("shoulder_3_long",
        visual_mesh=f"meshes/shoulder_3{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.042, 0.02),
                      Origin([0, -0.093, 0.02475], [deg(90), 0, 0]),
                      is_visual=False, is_collision=True),
        name_override="shoulder_3_R",
        visual_material="light_grey"
    )
    links += [shoulder_3_L, shoulder_3_R]

    # Asymmetric limits: pitch swings forward on both sides, so the sign convention flips.
    left_shoulder_2_joint = simple_joint("left_shoulder_2_joint",
        parent="shoulder_2_L", child="shoulder_3_L",
        xyz=[0.02475, 0, -0.09], rpy=[0, deg(-90), 0],
        effort=27,
        lower=deg(-180), upper=deg(30)
    )
    right_shoulder_2_joint = simple_joint("right_shoulder_2_joint",
        parent="shoulder_2_R", child="shoulder_3_R",
        xyz=[0.02475, 0, -0.09], rpy=[deg(180), deg(90), 0],
        effort=27,
        lower=deg(-30), upper=deg(180)
    )
    joints += [left_shoulder_2_joint, right_shoulder_2_joint]

    # ── Shoulder 3 (Yaw) ───────────────────────────────────────
    elbow_L = mirror_link_about_plane(link_from_fusion("elbow_long",
        visual_mesh=f"meshes/elbow{hig_res_str}.obj",
        collision=Geom(Capsule(0.042, 0.01),
                    #    Cylinder(0.0785/2, 0.047), # original, do not remove
                      Origin([0, 0, -0.06-0.042-0.009], [0, deg(90), 0]),
                      is_visual=False, is_collision=True),
        name_override="elbow_L"
    ), normal=[0, 1, 0])
    elbow_R = link_from_fusion("elbow_long",
        visual_mesh=f"meshes/elbow{hig_res_str}.obj",
        collision=Geom(Capsule(0.042, 0.01),
                      Origin([0, 0, -0.06-0.042-0.009], [0, deg(90), 0]),
                      is_visual=False, is_collision=True),
        name_override="elbow_R"
    )
    links += [elbow_L, elbow_R]

    left_shoulder_3_joint = simple_joint("left_shoulder_3_joint",
        parent="shoulder_3_L", child="elbow_L",
        xyz=[0, -0.096, 0.02475], rpy=[deg(-90), deg(180), 0],
        effort=15.5,
        lower=deg(-180), upper=deg(180)
    )
    right_shoulder_3_joint = simple_joint("right_shoulder_3_joint",
        parent="shoulder_3_R", child="elbow_R",
        xyz=[0, -0.096, 0.02475], rpy=[deg(-90), 0, 0],
        effort=15.5,
        lower=deg(-180), upper=deg(180)
    )
    joints += [left_shoulder_3_joint, right_shoulder_3_joint]

    # ── Elbow ──────────────────────────────────────────────────
    # Chiral: reflect about its own YZ plane so both sides share one mesh + Fusion entry.
    # The child joint (left_wrist_1_joint) sits at x=0, on the plane, so it is unaffected.
    wrist_1_L = mirror_link_about_plane(link_from_fusion("wrist_1_long",
        visual_mesh=f"meshes/wrist_1{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.042, 1e-4),Origin([0, -0.0785-0.01, 0], [deg(90), 0, 0]),
            # Cylinder(0.0785/2, 0.033),Origin([0, -0.0785, 0.0235], [deg(90), 0, 0]), # original, do not remove
                      is_visual=False, is_collision=True),
        name_override="wrist_1_L",
        visual_material="light_grey"
    ), normal=[1, 0, 0])
    wrist_1_R = link_from_fusion("wrist_1_long",
        visual_mesh=f"meshes/wrist_1{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.042, 1e-4),Origin([0, -0.0785-0.01, 0], [deg(90), 0, 0]),
                      is_visual=False, is_collision=True),
        name_override="wrist_1_R",
        visual_material="light_grey"
    )
    links += [wrist_1_L, wrist_1_R]

    left_elbow_joint = simple_joint("left_elbow_joint",
        parent="elbow_L", child="wrist_1_L",
        xyz=[0, 0, -0.06-0.042-0.009], rpy=[deg(90), 0, deg(-90)],
        effort=15.5,
        lower=deg(-125), upper=deg(125)
    )
    right_elbow_joint = simple_joint("right_elbow_joint",
        parent="elbow_R", child="wrist_1_R",
        xyz=[0, 0, -0.06-0.042-0.009], rpy=[deg(90), 0, deg(-90)],
        effort=15.5,
        lower=deg(-125), upper=deg(125)
    )
    joints += [left_elbow_joint, right_elbow_joint]

    # ── Wrist 1 (Pitch) ────────────────────────────────────────
    wrist_2_L = mirror_link_about_plane(link_from_fusion("wrist_2_long",
        visual_mesh=f"meshes/wrist_2{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.038, 0.02),
            # Cylinder(0.057/2, 0.051), # original, do not remove
            Origin([0, 0, -0.0435-0.042], [0, deg(90), 0]),
            is_visual=False, is_collision=True),
        name_override="wrist_2_L",
        visual_material="dark_grey"
    ), normal=[0, 1, 0])
    wrist_2_R = link_from_fusion("wrist_2_long",
        visual_mesh=f"meshes/wrist_2{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.038, 0.02),
            Origin([0, 0, -0.0435-0.042], [0, deg(90), 0]),
            is_visual=False, is_collision=True),
        name_override="wrist_2_R",
        visual_material="dark_grey"
    )
    links += [wrist_2_L, wrist_2_R]

    left_wrist_1_joint = simple_joint("left_wrist_1_joint",
        parent="wrist_1_L", child="wrist_2_L",
        xyz=[0, -0.096, 0], rpy=[deg(-90), deg(-90), 0],
        effort=15.5,
        lower=deg(-180), upper=deg(180)
    )
    right_wrist_1_joint = simple_joint("right_wrist_1_joint",
        parent="wrist_1_R", child="wrist_2_R",
        xyz=[0, -0.096, 0], rpy=[deg(-90), deg(-90), 0],
        effort=15.5,
        lower=deg(-180), upper=deg(180)
    )
    joints += [left_wrist_1_joint, right_wrist_1_joint]

    # ── Wrist 2 (Yaw) ──────────────────────────────────────────
    # Left wrist_3 is the chiral mirror of the right part, not the same part re-posed:
    # reflect it about its own YZ plane so it reuses the one mesh and Fusion entry.
    # The child joint (left_wrist_3_joint) sits at x=0, on the plane, so it is unaffected.
    wrist_3_L = mirror_link_about_plane(link_from_fusion("wrist_3_new",
        visual_mesh=f"meshes/wrist_3_new{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.04, 0.02),Origin([0, 0, -0.07], [deg(90), 0, 0]),
            # Cylinder(0.046/2, 0.04875), Origin([0, -0.10925/2, 0.00], [deg(90), 0, 0]), # original, do not remove
                      is_visual=False, is_collision=True),
        name_override="wrist_3_L",
        visual_material="light_grey"
    ), normal=[1, 0, 0])
    wrist_3_R = link_from_fusion("wrist_3_new",
        visual_mesh=f"meshes/wrist_3_new{hig_res_str}.obj",
        collision=Geom(
            Capsule(0.04, 0.02),Origin([0, 0, -0.07], [deg(90), 0, 0]),
                      is_visual=False, is_collision=True),
        name_override="wrist_3_R",
        visual_material="light_grey"
    )
    links += [wrist_3_L, wrist_3_R]

    left_wrist_2_joint = simple_joint("left_wrist_2_joint",
        parent="wrist_2_L", child="wrist_3_L",
        xyz=[0.00, 0, -0.0435-0.042], rpy=[deg(0), 0, deg(180)],
        axis=[-1, 0, 0],
        effort=12,
        lower=deg(-92), upper=deg(92)
    )
    right_wrist_2_joint = simple_joint("right_wrist_2_joint",
        parent="wrist_2_R", child="wrist_3_R",
        xyz=[0.00, 0, -0.0435-0.042], rpy=[deg(0), 0, deg(0)],
        axis=[-1, 0, 0],
        effort=12,
        lower=deg(-92), upper=deg(92)
    )
    joints += [left_wrist_2_joint, right_wrist_2_joint]

    # ── Wrist 3 (Roll) - End Effector ─────────────────────────
    # Physics key must match the mesh actually rendered: the "_new" attachment (0.158 kg),
    # not the superseded "end_effector_attachment" (0.041 kg) — a 3.8x mass error otherwise.
    # Capsule collision (commented out) would cover the full hand: center at EE midpoint,
    # extending from wrist joint to EE tip, axis along local X.
    end_effector_L = mirror_link_about_plane(link_from_fusion("end_effector_attachment_new",
        visual_mesh=f"meshes/end_effector_attachment_new{hig_res_str}.obj",
        # collision=[Geom(Capsule(0.03, 0.05),
        #                 Origin(EE_SITE_OFFSET_L, [deg(90), 0, 0]),is_collision=True, is_visual=False)],
        name_override="end_effector_L",
        visual_material="dark_grey"
    ), normal=[0, 1, 0])
    end_effector_R = link_from_fusion("end_effector_attachment_new",
        visual_mesh=f"meshes/end_effector_attachment_new{hig_res_str}.obj",
        # collision=[Geom(Capsule(0.03, 0.05),
        #                 Origin(EE_SITE_OFFSET_R, [deg(90), 0, 0]), is_collision=True, is_visual=False)],
        name_override="end_effector_R",
        visual_material="dark_grey"
    )
    links += [end_effector_L, end_effector_R]

    # axis = roll (negated to match physical positive direction)
    # ref=deg(0): when arm is up, the wrist_3 frame is aligned with the base_link frame
    left_wrist_3_joint = simple_joint("left_wrist_3_joint",
        parent="wrist_3_L", child="end_effector_L",
        xyz=[0, 0.0, -0.064], rpy=[[0,1,0],[0,0,-1],[-1,0,0]],
        effort=12,
        lower=deg(-90), upper=deg(90)
    )
    right_wrist_3_joint = simple_joint("right_wrist_3_joint",
        parent="wrist_3_R", child="end_effector_R",
        xyz=[0, 0.0, -0.064], rpy=[[0,1,0],[0,0,-1],[-1,0,0]],
        effort=12,
        lower=deg(-90), upper=deg(90)
    )
    joints += [left_wrist_3_joint, right_wrist_3_joint]

    # ── End-Effector Sites ─────────────────────────────────────
    # Massless site at EE_SITE_OFFSET_* along wrist_3_* local x — IK target frame.
    end_effector_L.sites.append(Site(name="end_effector_L_site", origin=Origin(EE_SITE_OFFSET_L)))
    end_effector_R.sites.append(Site(name="end_effector_R_site", origin=Origin(EE_SITE_OFFSET_R)))

    return links, joints


# ════
#  MAIN ROBOT BUILDER
# ════

def build_humanoid() -> Robot:
    """Build complete humanoid robot

    Returns:
        Robot with 28 links, 27 joints (27 DOF)
    """
    robot = Robot("humanoid_v2", materials=MATERIALS)

    # Build all body parts
    torso_links, torso_joints = build_torso()
    leg_links, leg_joints = build_legs()
    arm_links, arm_joints = build_arms()

    # Add to robot
    for link in (torso_links + leg_links + arm_links):
        robot.add_link(link)

    for joint in (torso_joints + leg_joints + arm_joints):
        robot.add_joint(joint)

    return robot


# ════
#  BATCH GENERATION & DOMAIN RANDOMIZATION
# ════

def strip_high_res(xml: str) -> str:
    """Replace _high_res.obj references with .obj for low-res XML variant."""
    return xml.replace("_high_res.obj", ".obj")


def assert_mirror_symmetry(xml_path: str, tol_mm: float = 1e-6):
    """Fail-loud gate: every ``*_L`` body must be the exact mirror of its ``*_R`` twin.

    The bare robot is symmetric about the xz plane, so this must hold by construction.
    It guards the ``mirror_link_about_plane`` normals in build_legs()/build_arms(): each
    normal is a function of that segment's joint rpy (H = R_L^T M R_R), so editing a joint
    frame silently invalidates it. Nothing else catches that — the model still compiles,
    mass and inertia eigenvalues still validate, and the only symptom is a policy that
    favours one side. A wrong-handed left leg went unnoticed here for exactly that reason.

    Pairs are found by name suffix, so a new limb segment is covered without editing this.

    Compares world positions only (body frame, COM, geom origins). Geom *orientations* are
    not compared: a mirrored rotation is not a rotation, so it needs the same H R S
    factorization as the reflection itself, and every defect seen so far also moves a
    position. Widen this if that stops being true.

    Args:
        xml_path: written MJCF to reload (path, not string, so relative mesh refs resolve).
        tol_mm: max allowed residual, millimetres.

    Raises:
        AssertionError: listing each broken link and its residual.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    flip_y = np.array([1.0, -1.0, 1.0])

    broken = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if not name or not name.endswith("_L"):
            continue
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name[:-2] + "_R")
        if j < 0:
            continue
        err = max(np.abs(data.xpos[i] - data.xpos[j] * flip_y).max(),
                  np.abs(data.xipos[i] - data.xipos[j] * flip_y).max())
        # Match geoms as SETS within each type, never by index. A capsule grid enumerates in
        # local order, so the mirrored body lists the same points back-to-front (foot_R runs
        # y=-0.1557..-0.0977 against foot_L's +0.0977..+0.1557) and index pairing reports a
        # spurious 58 mm on a body that is exactly mirrored. Sorting positions alone is also
        # wrong: it can pair a visual mesh against a collision capsule (ankle_1, spurious
        # 50 mm). Type first, then position.
        sel_l, sel_r = model.geom_bodyid == i, model.geom_bodyid == j
        pos_l, typ_l = data.geom_xpos[sel_l], model.geom_type[sel_l]
        pos_r, typ_r = data.geom_xpos[sel_r] * flip_y, model.geom_type[sel_r]
        if len(pos_l) != len(pos_r):
            err = float("inf")
        else:
            ord_l, ord_r = np.lexsort((*pos_l.T, typ_l)), np.lexsort((*pos_r.T, typ_r))
            if not np.array_equal(typ_l[ord_l], typ_r[ord_r]):
                err = float("inf")
            else:
                err = max(err, np.abs(pos_l[ord_l] - pos_r[ord_r]).max())
        if err * 1000 > tol_mm:
            broken.append(f"    {name[:-2]:16s} {err * 1000:9.3f} mm")

    assert not broken, (
        "L/R mirror symmetry broken about the xz plane:\n" + "\n".join(broken) +
        "\n    -> check the mirror_link_about_plane normal on these links"
    )


def export_single():
    """Export high-res and low-res robot XMLs. Always builds with high-res mesh
    paths, then strips _high_res suffix for the low-res variant."""
    robot = build_humanoid()

    # Standard sensors for RL
    sensors = [
        Sensor("imu_ang_vel", "gyro", site="imu_site"),
        Sensor("imu_lin_vel", "velocimeter", site="imu_site"),
        Sensor("imu_lin_acc", "accelerometer", site="imu_site"),
        Sensor("root_angmom", "subtreeangmom", body="base_link")
    ]

    contact_bodies = ["foot_L", "foot_R"]

    # Declared qpos / actuator layout, independent of the order the build functions run in.
    # Downstream consumers index by it (keyframes in humanoid_v21_constants.py, trained policy
    # checkpoints, cuRobo arm poses), so changing it is a breaking change requiring their
    # migration — it is pinned here so a build refactor cannot permute it by accident.
    # The free joint is added by add_freejoint and is not listed.
    JOINT_ORDER = [
        "waist_joint",
        *(f"{side}_{j}_joint" for side in ("left", "right")
          for j in ("hip_1", "hip_2", "hip_3", "knee", "ankle_1", "ankle_2")),
        *(f"{side}_{j}_joint" for side in ("left", "right")
          for j in ("shoulder_1", "shoulder_2", "shoulder_3", "elbow", "wrist_1", "wrist_2", "wrist_3")),
    ]

    xml = robot.to_mjcf_string(
        add_freejoint=True,
        add_light_camera=True,
        imu_site_name="imu_site",
        contact_body_names=contact_bodies,
        sensors=sensors,
        joint_order=JOINT_ORDER,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    high_res_path = os.path.join(script_dir, "humanoid_v21_high_res.xml")
    low_res_path = os.path.join(script_dir, "humanoid_v21.xml")

    with open(high_res_path, "w") as f:
        f.write(xml)
    print(f"Exported: {high_res_path}")

    with open(low_res_path, "w") as f:
        f.write(strip_high_res(xml))
    print(f"Exported: {low_res_path}")

    assert_mirror_symmetry(low_res_path)
    print("Mirror symmetry OK (all L/R pairs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-n", type=int, default=0,
                       help="Generate N randomized variants")
    parser.add_argument("--batch-dir", default="tmp/batch/")
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()

    if args.batch_n > 0:
        print(f"Batch generation not yet implemented in v2")
        print(f"Use old script for batch: humanoid_v21_urdf_creation.py")
    else:
        export_single()
