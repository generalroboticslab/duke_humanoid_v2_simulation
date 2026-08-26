"""Stand-alone human-figure asset: a STANDING person offering an object (capsules, no joints).

Extends ``human_hand`` from a detached palm to a full body, so a handover scene contains a
HUMAN-SCALE occluder instead of a book-sized floating plate. The occlusion matters:
``curobo.scene._scene_occluder_geoms`` treats every ``pp_*`` geom as a sight-line blocker, and a bare
palm blocks almost no rays, which makes any "was the human monitored" metric vacuous.

``human_figure(...)`` returns a list of ``Prop`` pieces (shared ``PartBuilder``), welded onto any
``mujoco.MjSpec`` worldbody with ``add_props_to_spec`` (see ``base.py``). Run this file directly to
preview (``--render`` / ``--viewer``).

Frame and placement: (x, y, z) is the PALM position, identical in meaning to ``human_hand``, and the
hand pieces come from ``human_hand`` itself -- so a scenario can swap a hand marker for a figure
without moving the palm plate one millimetre, and the cube-on-palm support math
(``pickplace_scenarios._HAND_PALM_H`` / ``_handoff_palm_z``) stays exactly valid. The body is placed
BEHIND the palm along local -x (before yaw), i.e. the figure faces the direction the fingers point.

POSTURE: STANDING UPRIGHT, feet on the floor, at a stature of 1.10 m. Both halves of that are forced
by arithmetic, not taste. The target scenes pin the palm at 0.582 m (humanoid v2) to 0.710 m (G1).
For a proportional figure of stature H the shoulder sits at
0.818 H and the whole arm is 0.332 H long, so the LOWEST palm a standing figure can present is

    shoulder - arm = 0.486 H  ,  and the palm must also clear the chest:  reach > chest_r

Solving both at a 0.582 m palm caps the stature at ~1.18 m, and only near 1.10 m does the arm stop
being dead straight (here it closes at ~91% extension). A 1.75 m adult is short by 5 cm of arm at ANY
pose: shoulder 1.43, arm 0.582, required drop 0.85. Earlier revisions dodged this by kneeling (rejected:
a 1.0 m x-footprint sprawl, wider than the robot's reach envelope) and by leaning (rejected: needs a
55 deg bow). So this figure is deliberately child-scaled -- 1.10 m tall, head top 1.08, 0.20 m
footprint -- which is also what puts it at the same scale as a robot whose shoulder is 0.92-1.05 m.

Segment lengths below are stature fractions of that 1.10 m figure (upper arm 0.186 H, forearm
0.146 H, hip-to-shoulder 0.288 H, thigh/shin 0.245 H, shoulder height 0.818 H -- the skeleton
proportions used by SMPL-derived MuJoCo humanoids such as UHC), written out as absolute metres
because that is what the builder consumes.

FLOOR-ANCHORED, not palm-anchored: ``spec.shoulder_z`` is an absolute world height, so the feet stay
on the floor for any palm height and the ARM BEND absorbs the difference (~91% extension at v2's
0.582 m palm, ~62% at G1's 0.710 m). A palm further away than the arm is long would silently detach
the hand from the wrist, so that case asserts instead.

BUILT FROM CAPSULES, not boxes. Capsules read as limbs and are what a reviewer expects of a
simplified human; boxes read as a stack of crates. No joints and no mesh: the pose is scripted, so
there is never IK for the human, and the geometry stays cheap (see cost note below).

Collision: ONE real obstacle geom, a VERTICAL capsule from the floor to the shoulder, so cuRobo must
plan around the person's body (plus the hand's own palm proxy, see ``human_hand``). Both join the
"press 5" debug overlay (``_PLANNER_OVERLAY_GROUP``) rather than staying permanently invisible/visible;
they render nothing by default, same as every other collision-only geom. Everything else is decoration
(``contype = 0``). Three notes:
  * yawed BOXES sized from the visible geoms, not a vertical capsule. A capsule's circular section
    cannot be both 0.091 deep and 0.142 wide, so the single capsule earlier revisions used was 16 mm
    too fat in front AND 35 mm too narrow at the shoulders. It existed only because ``ik_curobo``
    once asserted box frames were world-axis-aligned while this figure yaws freely; cuRobo cuboids
    are oriented boxes, so the true shape is now passed straight through.
  * torso and legs are separate boxes, split at the hip: the feet reach 0.100 forward while the
    chest is 0.091 deep, and one box would carry the foot's depth up to shoulder height.
  * cost. Every collidable ``pp_*`` geom becomes a cuRobo obstacle cuboid; one per figure keeps the
    planner load identical to the old single-box torso. The 33 visible geoms are static with
    collision off, so they cost a draw call each and nothing in the physics or planning loop.
The head stays an ellipsoid (rounder than a capsule at head scale) precisely because it does not
collide -- ``build_curobo_scene`` raises on any collidable type other than box/capsule.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace

if __package__ in (None, ""):   # run as a script: put mj_envs/ on the path for the asset_zoo import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco
import numpy as np

from asset_zoo.scene_object.base import PartBuilder, Prop, add_props_to_spec, frame_from_axes
from asset_zoo.scene_object.human_hand import DEFAULT_HAND, HAND_RGBA, RESTING_CURL, HandSpec, human_hand

# Re-toned 2026-08-12 for the dark-anodized robot / dark floor. Dyed cotton and denim measure
# 0.10-0.25 diffuse albedo; the old 0.58-blue shirt was above that and, being the brightest
# object left in the frame, pulled the eye off the robot it stands next to. Both stay BLUE and
# keep their value gap so the waist line still reads at chase-cam distance.
SHIRT_RGBA = (0.20, 0.25, 0.35, 1.0)    # muted work shirt -- distinct from the skin-tone hand
PANTS_RGBA = (0.11, 0.12, 0.16, 1.0)    # darker than the shirt so the waist line reads

# Mirrors ``curobo_reach_harness.PLANNER_SPHERE_GROUP`` / ``_PLANNER_SPHERE_RGBA`` by value, not import
# (see the ``torso_col`` comment below for why): the human's two real obstacle proxies join the same
# "press 5" debug overlay the robot's own collision spheres already use.
_PLANNER_OVERLAY_GROUP = 5
_PLANNER_OVERLAY_RGBA = (0.1, 0.9, 0.9, 0.35)

# Which visible parts each collision proxy stands for. ``PartBuilder.local_bounds`` measures the
# box from these, so a changed body dimension moves the obstacle with it and a renamed part is a
# build-time error -- the proxy can no longer silently stop covering what it claims to cover.
# The limbs are absent on purpose: they have no proxy yet (see the module docstring).
_TORSO_PARTS = ("pelvis", "abdomen", "chest", "shoulders")
_LEG_PARTS = ("thighR", "thighL", "shinR", "shinL", "footR", "footL")


@dataclass(frozen=True)
class HumanSpec:
    """A standing 1.10 m figure. Lengths m. See the module docstring for why 1.10 and not 1.75."""
    # --- floor-anchored pose. shoulder_z = 0.818 * 1.10 stature; arm_reach is the horizontal palm ->
    # shoulder standoff and is the ONE tuned number here: larger clears the chest better but
    # straightens the arm (the vertical drop is already fixed by the stature).
    #
    # 0.15 put the palm proxy PENETRATING the torso collision capsule by 4.6 cm at the
    # nominal pose -- confirmed with ``mujoco.mj_geomDistance`` -- because an earlier revision squared up
    # BOTH proxies to a conservative isotropic radius for yaw invariance (a capsule/box legal at every
    # facing needs the larger of its two true half-extents; see ``torso_col``'s and ``human_hand``'s own
    # comments) without re-checking whether the two enlarged shapes still cleared each other. Neither
    # proxy can shrink back down without reopening that rotation-legality bug, so the fix is standoff:
    # arm_reach 0.15 -> 0.22 (paired with ``upper_arm_len``/``forearm_len`` below, since the elbow-length
    # budget at the old 0.15 had no headroom left -- the arm was already at ~96% of ``upper_arm_len +
    # forearm_len`` for the worst case (v2's lower shoulder height puts the palm further below the
    # figure's own 0.9 m shoulder, the biggest vertical drop of any robot)). Re-verified at 0.22: palm-vs-
    # torso clearance +0.023 m (was -0.046 m), arm at ~93% of the new, longer chord budget -- comfortable
    # margin on both sides, worst case (v2/v2_fixed); g1's shorter drop has even more room.
    shoulder_z: float = 0.900
    arm_reach: float = 0.22
    shoulder_half_span: float = 0.107   # shoulder joint offset from the body midline
    arm_side: float = -1.0
    """Which shoulder the offering arm leaves from: -1 = the figure's RIGHT (local -y), +1 = left.

    HANDEDNESS IS NOT FREE. ``human_hand`` builds a RIGHT hand by default -- fingers along +x, palm
    up (+z), thumb at -y -- and the figure faces +x, so +y is its left. Hanging that hand off the +y
    shoulder puts a right hand on a left arm and the thumb points outward, away from the body, which
    is anatomically impossible and was the first thing a reader noticed. ``human_hand(mirror=True)``
    now builds the left hand, and the RELAXED arm uses it; this side still stays at -1 because the
    offering pose is authored right-handed (see ``_relaxed_hand``)."""

    # --- segment lengths (stature fractions of the 1.10 m figure; see module docstring)
    # upper_arm_len/forearm_len scaled x1.05 alongside the ``arm_reach`` increase above -- restores the
    # elbow-length headroom the longer standoff consumes (measured: 0.22 m standoff at the old segment
    # lengths left only ~4% of the chord budget spare for the worst-case robot).
    upper_arm_len: float = 0.205 * 1.05
    forearm_len: float = 0.161 * 1.05
    torso_len: float = 0.317            # hip joint -> shoulder joint
    thigh_len: float = 0.270
    shin_len: float = 0.270
    foot_len: float = 0.069
    # ``hand_len`` lived here to size a bespoke relaxed hand. Both hands are now the one
    # ``human_hand`` asset (see ``_relaxed_hand``), so its length comes from ``HandSpec`` and the
    # two can no longer be sized apart.
    neck_len: float = 0.055     # 0.05 H; shorter gets hidden under the shoulder bar's own cap

    # --- radii. Chest 0.091 gives a 0.18 m chest; with the 0.035 shoulder bar the figure is 0.28 m
    # across the deltoids, which is shoulder breadth at this stature.
    chest_r: float = 0.091
    waist_r: float = 0.072
    pelvis_r: float = 0.063
    shoulder_r: float = 0.035
    upper_arm_r: float = 0.030
    forearm_r: float = 0.025
    thigh_r: float = 0.047
    shin_r: float = 0.037
    foot_r: float = 0.031
    neck_r: float = 0.031
    head_half: tuple[float, float, float] = (0.063, 0.052, 0.068)

    # --- posture
    hip_half_width: float = 0.053
    relaxed_hand_out: float = 0.080     # relaxed wrist, this far outboard of the hip (clears the thigh)

    rgba: tuple[float, float, float, float] = SHIRT_RGBA
    pants_rgba: tuple[float, float, float, float] = PANTS_RGBA
    skin_rgba: tuple[float, float, float, float] = HAND_RGBA
    hand: HandSpec = DEFAULT_HAND


DEFAULT_HUMAN = HumanSpec()


def _elbow(shoulder, wrist, upper_len: float, fore_len: float, bend) -> tuple[float, float, float]:
    """Elbow position with EXACT segment lengths: intersect the two joint spheres, pick a bend side.

    Two spheres (radius ``upper_len`` at the shoulder, ``fore_len`` at the wrist) meet in a circle;
    ``bend`` selects a point on it by pointing roughly where the elbow should flare. Solving this
    rather than hand-placing the elbow is what keeps the upper arm and forearm at true anthropometric
    length -- hand-picked joint points silently stretch or crush the limbs whenever the palm moves.
    Asserts the chord is reachable: a floor-anchored figure whose palm is beyond arm's length would
    otherwise render with the hand detached from the wrist, which is easy to miss in a small preview.
    """
    sx, sy, sz = shoulder
    dx, dy, dz = wrist[0] - sx, wrist[1] - sy, wrist[2] - sz
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    assert d <= upper_len + fore_len, (
        f"palm is {d:.3f} m from the shoulder but the arm is only {upper_len + fore_len:.3f} m long; "
        f"lower shoulder_z or shorten arm_reach"
    )
    ux, uy, uz = dx / d, dy / d, dz / d
    t = (upper_len ** 2 - fore_len ** 2 + d * d) / (2.0 * d)
    h = math.sqrt(max(upper_len ** 2 - t * t, 0.0))
    # component of the bend hint perpendicular to the arm axis
    bx, by, bz = bend
    dot = bx * ux + by * uy + bz * uz
    px, py, pz = bx - dot * ux, by - dot * uy, bz - dot * uz
    pn = math.sqrt(px * px + py * py + pz * pz)
    px, py, pz = px / pn, py / pn, pz / pn
    return (sx + t * ux + h * px, sy + t * uy + h * py, sz + t * uz + h * pz)


_RELAXED_EXTENSION = 0.97   # relaxed arm's chord as a fraction of its own length; see the wrist below


def _relaxed_wrist_drop(spec: HumanSpec) -> float:
    """How far below the shoulder the relaxed wrist hangs, for a near-straight arm.

    The wrist is already fixed laterally (outboard of the hip, to clear the thigh), so only the
    vertical leg is free: solve the chord at _RELAXED_EXTENSION of full arm length for it.
    """
    chord = _RELAXED_EXTENSION * (spec.upper_arm_len + spec.forearm_len)
    lateral = spec.shoulder_half_span - (spec.hip_half_width + spec.relaxed_hand_out)
    return math.sqrt(max(chord * chord - lateral * lateral, 0.0))


def _relaxed_hand(b: PartBuilder, spec: HumanSpec, wrist, axis) -> list[Prop]:
    """The hanging hand: the SAME ``human_hand`` asset, re-posed to point down ``axis`` from ``wrist``.

    This used to be a second, hand-written hand (palm pad + 4 digits + thumb, ~50 lines) because
    ``PartBuilder`` carries a yaw only and could not pitch the real asset into a hanging pose. That
    copy was wrong in every revision it had -- an axis-aligned ellipsoid palm on a tilted wrist,
    finger roots spread along world +x instead of across the palm, no taper, no curl -- and each fix
    only moved it further from the offering hand it was meant to match. ``PartBuilder`` now carries a
    full rotation, so there is ONE hand asset, placed twice, with no copy to drift.

    Frame: ``human_hand`` builds palm-UP with fingers along +x, so +x maps to the hand axis and +z
    (the palm normal) maps INBOARD, toward the thigh -- the way a relaxed palm faces.

    ``wrist``/``axis`` arrive in ``b``'s local frame (that is what the arm chain is solved in) and
    are mapped to world through ``b`` itself -- directions by its rotation, the wrist point by the
    full transform -- so the figure's own placement is applied exactly once.

    Returns the pieces; they do NOT go through ``b``, which builds in a different frame.
    """
    axis_w = b.R @ np.asarray(axis, float)
    # Palm faces inboard. The relaxed arm hangs opposite the offering arm, so inboard is +arm_side.
    palm_n = b.R @ np.array([0.0, spec.arm_side, 0.0])
    # The asset's origin is the palm CENTRE, half a palm ahead of the wrist along the hand axis.
    origin = np.asarray(b.to_world(wrist)) + axis_w * spec.hand.palm_half[0]
    # MIRRORED: the offering arm is ``arm_side``, so this one is the figure's OTHER hand. Placing
    # the unmirrored asset here put a right hand on the left arm, thumb pointing away from the body.
    # RESTING_CURL because this hand holds nothing -- the default spec is the presenting pose.
    return human_hand(f"{b.name}_hand2", *origin, yaw_deg=0.0,
                      spec=replace(spec.hand, **RESTING_CURL),
                      collidable_palm=False, rot=frame_from_axes(axis_w, palm_n), mirror=True)


def human_figure(name: str, x: float, y: float, z: float,
                 yaw_deg: float = 0.0, spec: HumanSpec = DEFAULT_HUMAN,
                 collidable_palm: bool = False,
                 collidable_torso: bool = True) -> list[Prop]:
    """Return the ``Prop`` pieces of one standing figure whose palm is at world (x, y, z), yaw deg.

    Pieces: the full ``human_hand`` cluster at exactly (x, y, z, yaw), then pelvis / abdomen / chest /
    shoulder-bar / neck capsules, an ellipsoid head, both arms (the offering arm reaching the palm and
    a relaxed arm hanging at the side), both legs (thigh + shin + foot), and one invisible collision
    capsule.

    The body stands one shoulder half-span to the side of the palm (see ``body_y`` in the source) so the
    offering arm lies in a single sagittal plane; the figure is NOT centred behind its own palm.

    The figure stands on z = 0, so ``z`` only sets how far the arm reaches down, not the body height.

    Args:
        name: geom-name stem; every piece is ``{name}_{suffix}`` and the caller is expected to weld
            with ``prefix="pp_"`` so the occluder and cuRobo-obstacle scans pick the figure up.
        x, y, z: world palm centre -- same semantics as ``human_hand``.
        yaw_deg: rotation about z; the figure faces +x before yaw (fingers point away from the body).
            ANY angle is legal: both collidable proxies are yaw-invariant by construction (the torso is a
            vertical capsule, the palm proxy a square axis-aligned box), so ``ik_curobo``'s axis-alignment
            assertions hold at every facing. An earlier revision restricted this to multiples of 90.
        collidable_palm: forwarded to ``human_hand``.
        collidable_torso: add the invisible torso collision capsule (the cuRobo obstacle).
    """
    pieces = human_hand(name, x, y, z, yaw_deg, spec.hand, collidable_palm)
    b = PartBuilder(name, x, y, yaw_deg)

    # Skeleton. Local x is "behind the palm", z is world height; the body stands upright at
    # x = -arm_reach, so shoulder, hip, knee and ankle all share that x.
    #
    # body_y offsets the WHOLE body sideways by one shoulder half-span, which lands the offering
    # shoulder at y = 0, directly behind the palm. Derived, not tuned: it is the only offset for which
    # the offering arm lies in a single sagittal plane. Centring the body instead (earlier revision)
    # left the shoulder 0.107 m to one side of a palm on the midline, so the arm had to reach ACROSS
    # the body -- the forearm arrived at the palm 52% laterally at v2's palm height and 94% at G1's,
    # which put the hand on sideways and forced the wrist 68-71 deg off the palm axis, at the human
    # extension limit. Offsetting the body drops that to 45 deg / 2 deg with zero lateral component.
    body_x = -spec.arm_reach
    body_y = -spec.arm_side * spec.shoulder_half_span
    shoulder_z = spec.shoulder_z
    hip_z = shoulder_z - spec.torso_len
    knee_z = hip_z - spec.thigh_len
    ankle_z = knee_z - spec.shin_len

    def spine(t: float) -> tuple[float, float, float]:
        """Point a fraction t up the hip->shoulder line (t=0 hip, t=1 shoulder)."""
        return (body_x, body_y, hip_z + t * (shoulder_z - hip_z))

    # Torso as three overlapping capsules of increasing radius: narrow pelvis, waist, broad chest.
    # A single capsule would be a uniform barrel -- the taper is what makes it read as a person.
    b.capsule("pelvis", (body_x, body_y - spec.hip_half_width, hip_z),
              (body_x, body_y + spec.hip_half_width, hip_z),
              spec.pelvis_r, spec.pants_rgba, False)
    b.capsule("abdomen", spine(0.05), spine(0.55), spec.waist_r, spec.rgba, False)
    # Chest stops at 0.8 of the spine, NOT at the shoulder joint. A capsule endpoint carries a
    # hemispherical cap, so a chest ending at the shoulder bulges chest_r (0.091) ABOVE it and
    # swallows the whole neck and the base of the head -- the head then looks welded to the torso.
    # Ending at 0.8 tops the cap out just under the shoulder bar, which carries the width from there.
    b.capsule("chest", spine(0.5), spine(0.8), spec.chest_r, spec.rgba, False)
    b.capsule("shoulders", (body_x, body_y - spec.shoulder_half_span, shoulder_z),
              (body_x, body_y + spec.shoulder_half_span, shoulder_z), spec.shoulder_r, spec.rgba, False)
    neck_top = (body_x, body_y, shoulder_z + spec.neck_len)
    b.capsule("neck", (body_x, body_y, shoulder_z), neck_top, spec.neck_r, spec.skin_rgba, False)
    b.ellipsoid("head", (body_x, body_y, neck_top[2] + spec.head_half[2]), spec.head_half,
                spec.skin_rgba, False)

    # Offering arm: shoulder -> elbow -> wrist at the BACK of the palm. Two things are load-bearing.
    #
    # Wrist at 1.1x the palm semi-length, i.e. just behind its rear pole, not buried at 0.6 inside it
    # (earlier revision). The forearm is 0.025 in radius but the palm plate is only 0.011 half-thick,
    # so a wrist at the palm centre pushed the forearm's hemispherical cap 14 mm above the palm's top
    # surface -- straight through the cube resting there. At the rear pole the cap sits behind the palm
    # and clears the cube by 2 mm, and a real wrist is at the back of the palm anyway.
    #
    # Bend hint points straight BACKWARD, so the elbow tucks behind the shoulder and the forearm slopes
    # down-and-forward into the upturned palm (wrist 45 deg off the palm axis at v2's palm height, 2 deg
    # at G1's -- both inside the human extension range). The elbow grazes the flank by 3 mm, which reads
    # as an arm resting against the body. An outboard hint was tried first: it swings the elbow wide and
    # makes the forearm approach the palm from the SIDE, which is what made the wrist look broken.
    side = spec.arm_side
    for tag, sh, wr, bend in (
        ("", (body_x, body_y + side * spec.shoulder_half_span, shoulder_z),
         (-spec.hand.palm_half[0] * 1.1, 0.0, z), (-1.0, 0.0, 0.0)),
        # Relaxed arm: hangs at the side, wrist held outboard of the hip so it clears the thigh.
        # Wrist DEPTH is derived, not picked: _elbow puts the elbow h = sqrt(upper^2 - t^2) off the
        # shoulder-wrist line, so a wrist short of the arm's reach buys a big h and swings the whole
        # forearm forward. The old `hip_z - 0.02` sat at 88% extension -> elbow 90 mm off the line
        # and a forearm 32 deg off vertical, which is what threw the hand out in front of the thigh.
        # _RELAXED_EXTENSION hangs it near-straight; the residual bend still reads as a relaxed arm.
        ("2", (body_x, body_y - side * spec.shoulder_half_span, shoulder_z),
         (body_x, body_y - side * (spec.hip_half_width + spec.relaxed_hand_out),
          shoulder_z - _relaxed_wrist_drop(spec)),
         (-1.0, 0.0, 0.0)),
    ):
        el = _elbow(sh, wr, spec.upper_arm_len, spec.forearm_len, bend)
        b.capsule(f"upperarm{tag}", sh, el, spec.upper_arm_r, spec.rgba, False)
        b.capsule(f"forearm{tag}", el, wr, spec.forearm_r, spec.rgba, False)
        if tag:     # the offering side already has a full hand from ``human_hand``
            # Carry the hand on along the forearm axis. Hanging it straight down instead makes it
            # cut across the thigh from a side view.
            v = [w - e for w, e in zip(wr, el)]
            n = math.sqrt(sum(c * c for c in v))
            relaxed = _relaxed_hand(b, spec, wr, tuple(c / n for c in v))

    # Legs: straight and vertical under the hips, toes pointing the way the figure faces (+x).
    # L/R follow the FIGURE's anatomy, not the world: facing +x with +z up puts its left at +y (which
    # is also why the offering right arm leaves from -y -- see ``HumanSpec.arm_side``).
    for sy in (-1.0, 1.0):
        yw = body_y + sy * spec.hip_half_width
        b.capsule(f"thigh{'R' if sy < 0 else 'L'}", (body_x, yw, hip_z), (body_x, yw, knee_z),
                  spec.thigh_r, spec.pants_rgba, False)
        b.capsule(f"shin{'R' if sy < 0 else 'L'}", (body_x, yw, knee_z), (body_x, yw, ankle_z),
                  spec.shin_r, spec.pants_rgba, False)
        b.capsule(f"foot{'R' if sy < 0 else 'L'}", (body_x, yw, ankle_z),
                  (body_x + spec.foot_len, yw, spec.foot_r), spec.foot_r, spec.pants_rgba, False)

    if collidable_torso:
        # Two yawed BOXES, MEASURED off the visible parts they stand for (hence built last, once
        # those parts exist) rather than transcribed. A person is wider than deep, so the single
        # vertical capsule earlier revisions used was wrong in both directions at once: its radius
        # had to cover the WIDER half-extent, putting 16 mm of phantom depth in front, and it still
        # stopped 35 mm short of the shoulder tips it was meant to bound. That capsule existed only
        # because ``ik_curobo`` asserted box frames were world-axis-aligned while this figure yaws
        # freely; cuRobo cuboids are oriented boxes, so the true shape passes straight through now.
        #
        # Split at the hip because the FEET reach further forward than the chest is deep. One box
        # would carry the foot's depth all the way up to shoulder height, putting phantom exactly
        # where the gripper closes. Two boxes cost one extra cuRobo obstacle and keep both tight.
        #
        # group=5 / translucent cyan matches ``curobo_reach_harness.PLANNER_SPHERE_GROUP`` (not imported --
        # that module imports this one via ``curobo/scene.py``, so importing back would cycle): the SAME
        # "press 5" debug overlay that already shows the robot's own collision spheres and pick-object
        # padding. Before this the torso obstacle was alpha=0 and had no visible form at any toggle.
        for suffix, parts in (("torso_col", _TORSO_PARTS), ("legs_col", _LEG_PARTS)):
            center, half = b.local_bounds(*parts)
            b.box(suffix, center, half, _PLANNER_OVERLAY_RGBA, True, group=_PLANNER_OVERLAY_GROUP)

    return pieces + b.pieces + relaxed


# --------------------------------------------------------------------------- #
# Stand-alone preview (no robot/env): one figure on a floor.
# --------------------------------------------------------------------------- #
def _preview_model(yaw: float, palm_z: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4.0, 4.0, 0.1]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]
    add_props_to_spec(spec, human_figure("human", 0.0, 0.0, palm_z, yaw), prefix="pp_")
    return spec.compile()


def main() -> None:
    yaw, palm_z = 0.0, 0.582
    for a in sys.argv[1:]:
        if a.startswith("--yaw="):
            yaw = float(a.split("=", 1)[1])
        if a.startswith("--palm-z="):
            palm_z = float(a.split("=", 1)[1])
    model = _preview_model(yaw, palm_z)
    print(f"human_figure preview: yaw={yaw} palm_z={palm_z} ngeom={model.ngeom}")

    if "--viewer" in sys.argv:
        from mujoco import viewer
        model.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_CONTACT
                                   | mujoco.mjtDisableBit.mjDSBL_GRAVITY)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        viewer.launch(model, data)
        return

    if "--render" in sys.argv:
        # requires MUJOCO_GL=egl in the environment (mujoco reads it at import time)
        import imageio.v3 as iio
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [-0.2, 0.0, 0.6]
        cam.distance, cam.azimuth, cam.elevation = 2.2, 150.0, -12.0
        with mujoco.Renderer(model, height=720, width=1280) as r:
            r.update_scene(data, cam)
            img = r.render()
        out = os.path.join(os.path.dirname(__file__), "media", "human_figure_preview.png")
        iio.imwrite(out, img)
        print(f"rendered -> {out}")


if __name__ == "__main__":
    main()
