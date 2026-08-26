"""Stand-alone human-hand asset: a decorative capsule-cluster hand (no joints, no arm).

A reusable static-prop asset for "a person is offering something here" scene context —
palm + fingers + thumb, all non-colliding visual geoms (matches the abstract-target-marker
convention: visual context, no physics interaction). ``human_hand(...)`` returns a list of
``Prop`` pieces (built via the shared ``PartBuilder``), welded onto any ``mujoco.MjSpec``
worldbody with ``add_props_to_spec`` (see ``base.py``). Run this file directly to preview a
single hand (``--render`` / ``--viewer``).

Superseded design: grafting MuJoCo's bundled full-body ``humanoid.xml`` onto the scene —
rejected because its 3-DOF arm cannot reach an arbitrary scene target (grid-searched, best
case ~0.33 m off); a standalone hand prop has no such reach constraint since it is placed
directly.

Frame: a hand is built in its local frame (+x, before yaw, is the direction fingers point /
the hand presents toward), then rotated ``yaw_deg`` about z and translated to world (x, y, z).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

if __package__ in (None, ""):   # run as a script: put mj_envs/ on the path for the asset_zoo import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mujoco

from asset_zoo.scene_object.base import PartBuilder, Prop, add_props_to_spec

# Skin tone. Lowered from 0.90 red 2026-08-12: measured light-skin diffuse albedo is ~0.35-0.45,
# i.e. sRGB near 0.72, and the old value was a near-blown orange that clipped to a flat silhouette
# under the render's filmic tonemap -- the fingers lost their shading and read as one mitten. Kept
# deliberately the BRIGHTEST thing in the scene: this is also the grasp-target marker, so it must
# stay findable against the dark robot and floor.
HAND_RGBA = (0.72, 0.55, 0.44, 1.0)   # skin tone


@dataclass(frozen=True)
class HandSpec:
    """Shape + appearance of one hand (ellipsoid/capsule semi-axes or radius, m)."""
    # Palm ellipsoid semi-axes: wrist->knuckle length, width, thickness. Real-hand proportions
    # (~10cm long x ~8.5cm wide) put length slightly *ahead* of width, not behind it -- earlier
    # draft had width > length, which reads as a short/wide stub from top-down. Ellipsoid (not a
    # box) also gives rounded edges instead of a flat-faced slab.
    # Trimmed from (0.05, 0.042) 2026-08-12: at 100 x 84 mm the outline was very nearly a circle and
    # read as a paddle. THICKNESS IS UNCHANGED -- ``pickplace_scenarios._HAND_PALM_H`` is
    # ``palm_half[2]`` and sets where a cube rests, so that number is load-bearing and stays put.
    # Width stays well clear of ``CUBE_HALF`` (0.030); the palm must still support the cube it offers.
    palm_half: tuple[float, float, float] = (0.046, 0.038, 0.011)
    # Real fingers are roughly as long as the palm itself (not a short stub) and packed close
    # together at the base -- adjacent fingers TOUCH, they do not fan.
    finger_len: float = 0.085
    # FOUR FINGER BREADTHS MAKE A PALM BREADTH -- that is the proportion the eye actually checks.
    # 0.008 (16 mm) gave a 64 mm finger array under an 84 mm palm, so the palm flared out past the
    # fingers on both sides as bare surface and read as a disc with rods attached. 20 mm fingers
    # under a 76 mm palm close that gap; it is also simply the right girth for a hand this size.
    finger_r: float = 0.010
    n_fingers: int = 4
    # Total width spanned by the finger CENTRES. Slightly under 3 x diameter, so adjacent fingers
    # overlap and merge into one hand. An earlier 0.055-over-16 mm left a 2.3 mm air gap down every
    # seam, which read as four loose rods parked beside a palm.
    finger_spread: float = 0.057
    # Relative length of each finger (index -> pinky order), longest in the middle -- real hand
    # proportions, so a uniform-rod look doesn't read as mechanical/robotic.
    finger_length_ratios: tuple[float, ...] = (0.85, 1.0, 0.92, 0.70)

    # --- digit articulation. A finger is THREE phalanges (proximal/middle/distal) over three
    # joints (MCP/PIP/DIP), and the flexion COMPOUNDS down the chain -- that compounding is what
    # produces a finger's curve. Until 2026-08-12 a digit was two capsules with a single bend,
    # which cannot make that shape at any parameter value; the hand read as jointed tubing and no
    # amount of retuning the one angle fixed it. Fractions are of the finger's own length and are
    # near-invariant across real hands.
    phalanx_fractions: tuple[float, float, float] = (0.45, 0.30, 0.25)
    # Girth tapers toward the tip. Gentle: each capsule carries a hemispherical cap, so a big step
    # between segments shows up as a bulge at every knuckle rather than as taper.
    phalanx_r_scale: tuple[float, float, float] = (1.0, 0.92, 0.82)
    # MCP, PIP, DIP for the index finger. The DEFAULT is the OFFERING pose: FLAT, because this hand
    # presents an object resting on its palm -- a closed hand fights the thing it is holding out,
    # and the cube's support face is the flat palm. Not exactly zero: a real open hand still carries
    # a few degrees of residual tone, and dead-straight digits read as machined rod.
    # ``RESTING_CURL`` below is the same hand hanging free. Pose is a parameter, not a second asset.
    # Slightly NEGATIVE: on a flat presented hand the digits continue the palm's line and the tips
    # droop a hair under their own weight. Curling them up cups the palm, which is the one thing a
    # hand offering an object on that palm must not do.
    joint_curl_deg: tuple[float, float, float] = (-1.0, -2.0, -1.5)
    # Curl grows index -> pinky (a hand closes progressively across the span), as a multiple of
    # ``joint_curl_deg`` at the last finger. 1.0 (flat pose) keeps the four level with each other.
    finger_curl_ramp: float = 1.0
    # Fingertips draw toward the hand's midline as they extend, the way a relaxed hand's do,
    # instead of running as four parallel rails. Degrees at the outermost finger.
    finger_converge_deg: float = 7.0

    # --- thumb. Length is now along ITS OWN CHAIN (metacarpal + 2 phalanges), not a sideways
    # offset: the thumb's metacarpal is mobile and sits in the palm, which is why the thumb has an
    # extra segment the fingers do not. ``opposition`` lifts it OUT of the palm plane toward the
    # fingers -- the single most recognisable thing about a human thumb, and the reason a thumb
    # drawn flat in the palm plane reads as a fifth finger stuck on sideways.
    thumb_len: float = 0.058
    thumb_fractions: tuple[float, float, float] = (0.42, 0.32, 0.26)
    # Flat-hand defaults, matching ``joint_curl_deg``: the thumb lies IN the palm plane, abducted to
    # the side, exactly as flat as the fingers. Opposition and the thumb's own flexion STACK (the
    # lift is the chain's starting heading, the joints add to it), so even the modest 10/15 pair
    # tipped it 18 deg off an otherwise flat hand and it read as a raised thumb.
    # ``RESTING_CURL`` rolls it into opposition, which is a GRASPING posture, not an open one.
    thumb_joint_curl_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    thumb_splay_deg: float = 34.0        # away from the fingers, in the palm plane
    thumb_opposition_deg: float = 0.0    # out of the palm plane, toward the palm side
    thumb_r_scale: float = 1.15          # thumb is visibly thicker than the fingers
    # Thenar eminence -- the muscle pad at the thumb's base. Semi-axes as fractions of palm_half.
    # A palm without it is a flat slab; it is most of a real palm's volume from the side. Kept
    # BELOW 1.0 in z so it swells the palm from within instead of sitting on it as a lens.
    thenar_half_frac: tuple[float, float, float] = (0.52, 0.34, 0.92)
    # Wrist end. A real palm is widest at the knuckles and narrows to the wrist; one ellipsoid is
    # the same width at both ends, which is what reads as an egg. This narrower ellipsoid, set back
    # behind the palm, supplies the taper. Fractions of palm_half; x is also its setback.
    wrist_half_frac: tuple[float, float, float] = (0.42, 0.66, 0.90)
    rgba: tuple[float, float, float, float] = HAND_RGBA


DEFAULT_HAND = HandSpec()
# A hand hanging free is NOT the presenting pose: with nothing to hold, the digits fall to their
# resting tonus, roughly 60 deg cumulative at the fingertip, and the thumb rolls into opposition.
# Same asset, different joint angles.
RESTING_CURL = dict(joint_curl_deg=(16.0, 28.0, 16.0), finger_curl_ramp=1.6,
                    thumb_joint_curl_deg=(8.0, 16.0, 12.0), thumb_opposition_deg=34.0)


def _digit(b: PartBuilder, suffix: str, root, total_len: float, fractions, joint_deg,
           radius: float, r_scales, rgba, yaw_deg: float, lift_deg: float = 0.0,
           hand_y: float = 1.0) -> tuple[float, float, float]:
    """Add one articulated digit as a chain of capsules from ``root``. Returns the tip.

    The chain is what makes this read as a finger. Each segment's direction is the previous one
    rotated by that JOINT's flexion angle, so the angles ACCUMULATE (MCP, then MCP+PIP, then
    MCP+PIP+DIP) and the digit sweeps a real arc. A two-capsule digit with a single bend -- what
    this replaced -- has one direction change and reads as a hinge, not a finger.

    Flexion is toward +z, the palm side, about the axis perpendicular to the digit's own heading:
    the digit is built in its own plane, so ``yaw_deg`` (heading in the palm plane) and the flexion
    are independent and neither distorts the other.

    Args:
        root: chain start, in the hand's canonical RIGHT-hand local frame.
        total_len: length of the whole chain; ``fractions`` splits it per segment.
        joint_deg: flexion added AT the start of each segment, degrees, one per segment.
        r_scales: radius multiplier per segment, tapering toward the tip.
        yaw_deg: heading in the palm plane, away from +x (the finger axis).
        lift_deg: heading OUT of the palm plane toward the palm side -- thumb opposition. Zero for
            fingers, which stay in plane.
        hand_y: -1 mirrors the finished chain to the left hand. Applied at emit, after the whole
            chain is solved, so the reflection cannot interact with the rotation sense.
    """
    yaw, lift = math.radians(yaw_deg), math.radians(lift_deg)
    # Heading, and the in-plane axis it flexes about. ``side`` is perpendicular to ``fwd`` and
    # chosen so that a positive joint angle rotates the digit toward +z.
    fwd = (math.cos(yaw) * math.cos(lift), math.sin(yaw) * math.cos(lift), math.sin(lift))
    side = (math.sin(yaw), -math.cos(yaw), 0.0)

    p = tuple(root)
    cum = 0.0
    for i, (frac, jd, rs) in enumerate(zip(fractions, joint_deg, r_scales)):
        cum += math.radians(jd)
        c, s = math.cos(cum), math.sin(cum)
        # Rodrigues about ``side``; ``fwd`` is perpendicular to it, so the u(u.v) term vanishes.
        cross = (side[1] * fwd[2] - side[2] * fwd[1],
                 side[2] * fwd[0] - side[0] * fwd[2],
                 side[0] * fwd[1] - side[1] * fwd[0])
        d = tuple(f * c + x * s for f, x in zip(fwd, cross))
        nxt = tuple(pc + dc * total_len * frac for pc, dc in zip(p, d))
        b.capsule(f"{suffix}{i}", (p[0], hand_y * p[1], p[2]), (nxt[0], hand_y * nxt[1], nxt[2]),
                  radius * rs, rgba, False)
        p = nxt
    return p


def human_hand(name: str, x: float, y: float, z: float,
               yaw_deg: float = 0.0, spec: HandSpec = DEFAULT_HAND,
               collidable_palm: bool = False, rot=None, mirror: bool = False) -> list[Prop]:
    """Return the ``Prop`` pieces of one hand at world (x, y, z), yaw deg.

    ``rot`` (3x3, see ``base.frame_from_axes``) replaces the yaw with a full orientation, so the
    SAME hand serves the palm-up offering pose and the human figure's tilted hanging arm. Local +x
    is the finger direction and local +z the palm normal; ``(x, y, z)`` is the palm centre in both
    cases. Before this the builder was yaw-only and the figure carried a bespoke second hand.

    ``mirror`` gives the LEFT hand. This is authored here, by negating local y, and NOT by any
    transform at placement time: handedness is a REFLECTION, and ``Prop.quat`` can only hold a
    rotation, so no amount of re-orienting a right hand produces a left one. The default (right)
    hand has its thumb at -y, which with +x forward and +z up is a right hand palm-up. A figure
    that needs both hands must mirror one of them -- putting the unmirrored asset on both arms
    points one thumb outward, away from the body, which is anatomically impossible.

    Pieces (local frame, then yawed + translated): an ellipsoid palm (rounded pad, not a flat-
    faced box), ``n_fingers`` 2-segment curled+tapered digits fanned out from the palm's domed
    front, and one thicker 2-segment opposable thumb off to one side. Each digit's root is buried
    inside the palm ellipsoid (at 75% of the local surface depth for its lateral offset) so there
    is no visible box/tube seam, and -- since an ellipsoid's front surface pulls back toward the
    edges -- the knuckle line falls into a natural arc for free, instead of the dead-straight row
    a flat palm face gives. ``collidable_palm`` adds one thin, conservative collision box that
    encloses palm, fingers, and thumb. Visual hand geoms stay non-colliding. A sphere is unsuitable
    because it cannot support the cube's flat base and cuRobo's static-scene bridge accepts
    boxes/capsules only.
    """
    hand_y = -1.0 if mirror else 1.0     # every local y flips for the left hand; see ``mirror``
    palm_half, finger_len, finger_r = spec.palm_half, spec.finger_len, spec.finger_r
    n_fingers, spread, rgba = spec.n_fingers, spec.finger_spread, spec.rgba
    ratios = spec.finger_length_ratios if len(spec.finger_length_ratios) == n_fingers else (1.0,) * n_fingers
    a, palm_w, palm_c = palm_half

    def root_x(fy: float, embed: float = 0.75) -> float:
        """x on the palm ellipsoid's front surface at lateral offset ``fy``, pulled ``embed``
        of the way back toward center so the digit root is buried, not seamed onto the surface."""
        ratio = max(0.0, 1.0 - (fy / palm_w) ** 2)
        return a * math.sqrt(ratio) * embed

    # (x, y, z) is the builder ORIGIN, so every piece below is authored about the palm centre.
    # Keeping z as a local offset instead would tilt it with ``rot`` and slide the hand off its
    # own placement point.
    b = PartBuilder(name, x, y, yaw_deg, z=z, rot=rot)
    b.ellipsoid("palm", (0.0, 0.0, 0.0), palm_half, rgba, False)
    if collidable_palm:
        # ``palm_half`` exactly, and yawed with the builder (hence ``b.box``): the proxy IS the palm.
        # Earlier revisions squared the footprint to its larger half-extent and stamped an identity
        # quat, because ``ik_curobo`` asserted every box frame was world-axis-aligned and the
        # live-handover scene presents this hand at arbitrary yaw. cuRobo cuboids are oriented boxes,
        # so that assert is gone and the padding it forced (0.085 square vs a true 0.050 x 0.042 --
        # a hand 80% as wide as the person's chest, straddling the very cube the gripper must take)
        # goes with it. group=5 / translucent cyan mirrors
        # ``curobo_reach_harness.PLANNER_SPHERE_GROUP`` by value (importing back would cycle) -- the
        # "press 5" debug overlay, same as every other collision-only proxy, so it renders nothing by
        # default instead of being a permanent opaque-red box on top of every offered cube.
        #
        # NOT shrunk further toward the cube's own 0.030: below the measured palm the planner would
        # believe part of a real hand is absent. The decorative fingertips already reach ~0.125 m and
        # are ``contype=0``, so the hand is under-modelled in that direction already -- trimming the
        # palm compounds that gap rather than removing slack.
        b.box("collision", (0.0, 0.0, 0.0), palm_half, (0.1, 0.9, 0.9, 0.35), True, group=5)

    for i in range(n_fingers):
        fy = -spread / 2.0 + spread * i / (n_fingers - 1)
        t = i / (n_fingers - 1)
        # Curl ramps index -> pinky; the digit also heads toward the midline as it extends.
        ramp = 1.0 + (spec.finger_curl_ramp - 1.0) * t
        _digit(b, f"finger{i}", (root_x(fy), fy, 0.0), finger_len * ratios[i],
               spec.phalanx_fractions, [j * ramp for j in spec.joint_curl_deg],
               finger_r, spec.phalanx_r_scale, rgba,
               yaw_deg=-spec.finger_converge_deg * fy / (spread / 2.0), hand_y=hand_y)

    # Thumb: rooted at the palm BASE (behind the knuckle line, on the thenar side) and lifted out
    # of the palm plane by opposition. Rooting it level with the fingers -- what this did before --
    # is what made it read as a fifth finger sticking out sideways.
    ty = -palm_w * 0.62
    _digit(b, "thumb", (-a * 0.22, ty, palm_c * 0.35), spec.thumb_len,
           spec.thumb_fractions, spec.thumb_joint_curl_deg,
           finger_r * spec.thumb_r_scale, spec.phalanx_r_scale, rgba,
           yaw_deg=-spec.thumb_splay_deg, lift_deg=spec.thumb_opposition_deg, hand_y=hand_y)
    # Thenar eminence: the thumb-base muscle pad, most of a real palm's volume seen edge-on.
    b.ellipsoid("thenar", (-a * 0.12, hand_y * ty * 0.62, 0.0),
                tuple(h * f for h, f in zip(palm_half, spec.thenar_half_frac)), rgba, False)
    # Wrist end, set back behind the palm so the silhouette narrows toward the arm.
    b.ellipsoid("wrist", (-a * (1.0 - spec.wrist_half_frac[0] * 0.75), 0.0, 0.0),
                tuple(h * f for h, f in zip(palm_half, spec.wrist_half_frac)), rgba, False)
    return b.pieces


# --------------------------------------------------------------------------- #
# Stand-alone preview (no robot/env): one hand on a floor.
# --------------------------------------------------------------------------- #
def _preview_model(yaw: float) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.worldbody.add_light(pos=[0.0, 0.0, 3.0], dir=[0.0, 0.0, -1.0])
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [4.0, 4.0, 0.1]
    floor.rgba = [0.3, 0.4, 0.5, 1.0]
    add_props_to_spec(spec, human_hand("hand", 0.0, 0.0, 0.9, yaw))
    return spec.compile()


def main() -> None:
    yaw = 0.0
    for a in sys.argv[1:]:
        if a.startswith("--yaw"):
            yaw = float(a.split("=", 1)[1]) if "=" in a else yaw
    model = _preview_model(yaw)
    print(f"human_hand preview: yaw={yaw} ngeom={model.ngeom}")

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
        cam.lookat[:] = [0.0, 0.0, 0.9]
        cam.distance, cam.azimuth, cam.elevation = 0.5, 135.0, -15.0
        with mujoco.Renderer(model, height=720, width=1280) as r:
            r.update_scene(data, cam)
            img = r.render()
        out = os.path.join(os.path.dirname(__file__), "media", "human_hand_preview.png")
        iio.imwrite(out, img)
        print(f"rendered -> {out}")


if __name__ == "__main__":
    main()
