"""Pick-place design-validation scenarios + scene injection.

Each scenario places PHYSICAL
static props (stand-alone workbenches from ``workbench.py`` / a shelf) plus abstract TARGET
points (objects to pick / pads to place / a human-hand point) for the heuristic controller
wired in later steps.

Frame convention (env-local world; the robot spawns at the origin per env, base at
HOME_KEYFRAME pos=(0,0,0.59), identity quat -> faces +x):
  +x = robot forward,  +y = robot left,  +z = up (floor at z=0).
A target at x=0.40 sits 0.40 m in front of the base. Props replicate per env via
mujoco_warp world replication (the geom lives in the single scene worldbody every env copies).

Props are injected through ``SceneCfg.spec_fn`` (the post-merge MjSpec hook), NOT the robot
entity spec -- mjlab strips an entity's worldbody geoms during scene merge, so props must
ride the scene spec (same mechanism as the ball-circle water-surface plane). The bench geometry
(visual-realistic top/legs/apron + simple top-plate-and-leg-capsule collision) lives in the
reusable ``workbench`` asset; this module only places benches and abstract markers.

Design decisions:
  - PHYSICAL props plus FREE pick cubes: a pick target rests on its support under MuJoCo contact and
    can be contacted/grasped. Place/hand markers remain abstract scene context.
  - Distances are BASE-relative: ``close`` near-edge ~0.20 m (stationary, in reach),
    ``far`` near-edge ~1.0 m (out of reach -> walk in first).
  - Long edge (48 in) faces the robot. Flank tables (left/right) are yawed 90 deg so their long
    edge faces the robot when it turns to that side; a 1.219 m bench cannot sit 0.1 m apart
    across the front, so flanking is the faithful reading of "table left and right".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import sqrt

import mujoco

from asset_zoo.scene_object import (
    DEFAULT_HAND,
    DEFAULT_TABLE_Z,
    TOP_DEPTH_H,
    Prop,
    ShelfSpec,
    add_props_to_spec,
    human_figure,
    shelf,
    tier_surface_z,
    workbench,
)

OBJ_CLEAR = 0.02                        # place-pad / hand marker clearance above a surface
CUBE_SIZE = 0.06                        # graspable object = 60 mm cube (user spec)
CUBE_HALF = CUBE_SIZE / 2.0             # 30 mm half-extent; cube center rests this far above surface
PICK_CUBE_MASS_KG = 0.03                # light foam/plastic manipulation block
PICK_CUBE_FRICTION = (1.2, 0.01, 0.001) # sliding, torsional, rolling contact friction
TABLE_FRICTION = (0.9, 0.01, 0.001)     # workbench support contact friction
# Stiffer than MuJoCo's 0.02 default: the gripper rack rakes the cube top during the tilted approach
# descent and a position-servo arm (kp=800, forcerange 150 N) drove the cube 50 mm into the 30 mm-thick
# table collision plate, i.e. clean through it, after which the cube free-falls to the floor and the
# mission can never grasp it. 0.005 s is still 2.5 physics steps at dt=0.002 (stable), and caps
# penetration well under the plate thickness so a graze stays recoverable instead of terminal.
PICK_CUBE_SOLREF = (0.005, 1.0)         # (time constant s, damping ratio)
LATERAL_CUBE_X = 0.25                   # bimanual_mixed_close table/handoff forward X (tuned; matches FRONT_BACK_CUBE_X reach distance)
LATERAL_CUBE_STD_M = 0.05                # exact per-axis standard deviation for lateral benchmark
LATERAL_CUBE_HALF_RANGE_M = sqrt(3.0) * LATERAL_CUBE_STD_M
# FAR cases only: widen the jitter along the table's own LONG axis (world x for the yawed left/right
# benches, world y for the yaw=0 front/back benches). Shared by left_right_far and front_back_far.
# Both far cases center their cube mean on that axis (see ``lr_far_cube_x``), so both share one ceiling:
# TOP_LEN_H - CUBE_HALF = 0.5796 of half-range, i.e. std 0.335. 0.30 -> half-range 0.520 keeps 0.060 m
# of table beyond the outermost cube FACE. Do not raise past 0.33 without shortening the cube or
# lengthening the bench -- the tail starts hanging off the end of the top.
FAR_LONG_SIDE_STD_M = 0.30
FAR_LONG_SIDE_HALF_RANGE_M = sqrt(3.0) * FAR_LONG_SIDE_STD_M
# Mean cube-center inset from table's NEAR edge (user-directed 2026-07-22, widened 2026-08-18). 0.11 m
# moved the distribution 35 mm inward from the prior CUBE_HALF + std placement and, with symmetric-uniform
# half-range 0.087 m, put the outermost <1% tail at up to 1.6 mm of overhang -- accepted at the time as a
# rare, small excursion. `bimanual_mixed_close`'s cube_0 (which sits at `lateral_near_y` directly, unlike
# left_right_close/far's `lateral_near_y + LATERAL_SPREAD_M`, so it carries the FULL tail risk) reliably
# knocked cube_0 off `table_L` at that worst case (2026-08-18: seed 46 reproduced 7/7 across 6 independent
# hosts/3 GPU architectures -- not a fluke, not a robot-collision, not host-specific; `knock_witness()`
# showed the cube's only contact was the table it was already resting on, confirmed geometrically: seed
# 46's draw left only 7 mm of support margin, well inside the documented tail). Something outside this
# file (contact solver, most likely) made a margin the design explicitly tolerated no longer stable in
# practice; 0.125 m restores a positive worst-case margin (8.4 mm) with the same half-range, so trial
# diversity is unchanged -- only the tail that was already flagged as marginal moves to safe. Verified:
# seed 46 clean post-fix (see `MEMORY.md` 2026-08-18). The depth distribution is shared by left_right_close
# AND left_right_far (extra `LATERAL_SPREAD_M` there already gave them more margin than bimanual_mixed_close
# had); only left_right_far widens its table-long-axis (world-x) distribution.
LATERAL_CUBE_EDGE_INSET_M = 0.125
FRONT_BACK_CUBE_STD_M = 0.03             # per-axis standard deviation, continuous symmetric-uniform x AND y
FRONT_BACK_CUBE_HALF_RANGE_M = sqrt(3.0) * FRONT_BACK_CUBE_STD_M
# Mirror the left_right EDGE_INSET convention: cube mean = near edge + CUBE_HALF + std, so the 1-std-
# inward sample sits near-face flush at the 0.20 near edge and the pair HUGS the edge. Continuous x
# (replaces the former asymmetric two-point support hack, which was only 2 discrete x-values). Support
# holds without the hack here: min center = mean - half_range = 0.203 > 0.20 edge, so the cube COM stays
# over the table (no tip); the >1-std inward tail shows only a harmless front-face overhang.
FRONT_BACK_CUBE_EDGE_INSET_M = CUBE_HALF + FRONT_BACK_CUBE_STD_M
FRONT_BACK_CUBE_X = 0.20 + FRONT_BACK_CUBE_EDGE_INSET_M   # = 0.255; front/back mirrored means across YZ plane
HANDOFF_CUBE_Z_STD_M = 0.01              # vertical handoff variation, bounded by G1 table-clearance floor
HANDOFF_CUBE_Z_HALF_RANGE_M = sqrt(3.0) * HANDOFF_CUBE_Z_STD_M
_MARK_R = 0.03                          # place-pad / hand marker sphere radius
_PICK_RGBA = (0.15, 0.80, 0.15, 1.0)    # green = object to pick
_PLACE_RGBA = (0.20, 0.45, 0.90, 1.0)   # blue = place pad
_HAND_RGBA = (0.90, 0.70, 0.55, 1.0)    # skin = human-hand point

# Robot invariant: shoulder z (base + arm mount). Hardcoded for now; refine later from mj_model.
# Reach envelope assumes this; cube/hand positions depend on it.
_SHOULDER_Z_M = 0.94

# Bench top sits this far below the reaching robot's shoulder joint, so a target resting on the bench
# presents at the SAME shoulder-relative reach height for every robot despite different stature. One
# value across ALL scenes for a given robot. 0.328 = the humanoid's original DEFAULT_TABLE_Z relative
# to its shoulder (its bimanual reach ceiling); a taller-shouldered robot (g1) gets a proportionally
# higher bench and keeps the humanoid's exact validated table when it stands at 0.9375 shoulder.
TABLE_BELOW_SHOULDER_M = 0.328


def table_z_for(shoulder_z: float, table_below: float = TABLE_BELOW_SHOULDER_M) -> float:
    """Per-robot bench-top world z: a fixed drop below the robot's shoulder-joint world z."""
    return shoulder_z - table_below

# Vertical half-thickness of the human_hand palm plate; an object resting IN a hand (offered
# for handoff, no table) sits this far above the hand marker's z, same "surface + CUBE_HALF"
# convention as a table (see ``oz`` below), just with the palm as the surface.
_HAND_PALM_H = DEFAULT_HAND.palm_half[2]

# Custom ShelfSpec for place_shelf: tier surfaces z=0.565 / 0.760, both within arm reach.
# Differs from DEFAULT_SHELF (tier_gaps=(0.25,0.25) -> surfaces 0.265/0.530, unreachable).
# Lowered from (0.65, 0.18) [surfaces 0.665/0.860] 2026-07-01: top tier sat only 0.08 m below
# shoulder (_SHOULDER_Z_M=0.94) -- too high a reach. First pass ground gap 0.65->0.45 (surfaces
# 0.465/0.652) went too low; bumped back up +0.10 -> 0.55. Ground gap only changed (inter-tier
# gap unchanged) so both tiers shift together; top tier now 0.18 m below shoulder.
# length (left-right span) DOUBLED 0.60 -> 1.20 m 2026-07-01 (user-directed, "2x wider"): the robot
# was straddling the narrow rack, its gripper clipping through the corner posts/tier edges
# (`scene_place_shelf.png` showed the left gripper + carried objects overlapping a post). Posts move
# from y=+-0.28 to y=+-0.58 (post_inset unchanged), clearing the arm span. width/depth (x, toward
# robot) untouched -- reach distance from the robot (PLACE_SHELF_X_M) is unaffected.
_SHELF_SPEC = ShelfSpec(
    n_tiers=2, length=1.20, width=0.28, thick=0.015,
    tier_gaps=(0.6, 0.2),
)
PLACE_SHELF_X_M = 0.4                  # shelf structure world x; front edge (0.30 m) clears torso

# Shelf tiers are shoulder-relative, same rule as the per-robot bench (``table_z_for``): the TOP tier
# surface sits this fixed drop below the reaching shoulder. Absolute floor-anchored tiers put g1's
# taller shoulder 0.114 m OVER a humanoid-sized rack, dropping the lower tier out of g1's reach (g1
# grasped the upper cube every seed but the lower cube almost never). Anchoring to the shoulder lifts
# g1's whole rack so BOTH tiers stay in reach; the humanoid keeps its validated shelf byte-for-byte.
# 0.1075 = the humanoid's current top-surface drop (shoulder 0.9375 - top surface 0.830).
SHELF_UPPER_BELOW_SHOULDER_M = 0.1075

def shelf_spec_for(shoulder_z: float, base: ShelfSpec = _SHELF_SPEC) -> ShelfSpec:
    """Per-stature shelf: pick the GROUND gap so the top tier surface lands ``SHELF_UPPER_BELOW_SHOULDER_M``
    below ``shoulder_z``. Inter-tier gap and every other field stay stature-invariant, so only the whole
    rack shifts vertically. Assumes 2 tiers. Inverts ``tier_surface_z`` for the top tier:
    top_surface = ground + 2*thick + inter_gap  ->  ground = shoulder - drop - 2*thick - inter_gap."""
    inter_gap = base.tier_gaps[1]
    ground = shoulder_z - SHELF_UPPER_BELOW_SHOULDER_M - 2.0 * base.thick - inter_gap
    return replace(base, tier_gaps=(ground, inter_gap))


@dataclass
class Marker:
    """One abstract target point, drawn as a non-colliding visual sphere (or, for ``hand``
    markers, a capsule-cluster hand shape — see ``make_scene_spec_fn``)."""
    name: str
    pos: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = _PICK_RGBA
    yaw_deg: float = 0.0    # hand markers only: local +x (finger-point direction) after this yaw
    collidable: bool = True     # True -> geom participates in MuJoCo collision world (default physical pick object)
    jitter_half_range_xyz: tuple[float, float, float] | None = None
    """Optional symmetric-uniform XYZ half-range. ``None`` uses caller's legacy XY rule and dz=0."""
    jitter_x_offsets: tuple[float, float] | None = None
    """Optional two-point X offsets paired with ``jitter_x_first_prob``."""
    jitter_x_first_prob: float = 0.0
    pos_follows_hand: str | None = None
    """Hand marker this pick object rests IN. Its pose is DERIVED from that hand's sampled palm pose
    and never sampled on its own, so an offered object cannot drift off the palm that holds it."""


@dataclass
class Scenario:
    """A pick-place validation case: static props + target markers + control regime."""
    name: str
    props: list[Prop]                                    # benches / shelf slabs
    pick: list[Marker] = field(default_factory=list)    # objects to grasp
    place: list[Marker] = field(default_factory=list)   # place-target markers (pad-like); unused when shelf tier IS target
    hand: list[Marker] = field(default_factory=list)    # human-hand points (no table)
    carried: list[Marker] = field(default_factory=list)  # objects the env starts holding (place scenarios)
    stationary: bool = True                              # False -> walk in (far cases)
    holding: bool = False                                # True -> start with objects in hand (each carried marker -> arm)


def _handoff_pieces(hand_xyz: tuple[float, float, float], xy_half_range: float,
                     yaw_deg: float) -> tuple[Marker, Marker]:
    """Build the (cube_1, hand_R) pair for a "handed off from a person" pick object.

    Single source of truth for every bimanual_mixed_* scenario: one palm (``hand_R``) at
    ``hand_xyz``, one cube resting on it (z = palm z + palm half-thickness + cube half-extent,
    same (x, y) as the palm).

    THE HAND IS THE RANDOMIZATION LEADER: it carries ``jitter_half_range_xyz`` (xy_half_range
    in-plane, HANDOFF_CUBE_Z_HALF_RANGE_M vertical) and the cube's pose is DERIVED from the sampled
    palm via ``pos_follows_hand``. The dependency used to run the other way (cube sampled, palm
    following it), which put the object -- not the person offering it -- at the root of the scene and
    let any future hand-only randomization (a rotating figure) slide the palm out from under the cube.
    Leading with the hand makes "the object is in the human's hand" true by construction, and the
    invisible palm collision box that physically supports the cube is part of the hand it moves with.
    The rng draw order is unchanged (one sample per handoff scene, in scenario-hand order after the
    freely-sampled picks), so every seed reproduces the exact scene it did before.

    Scenarios differ only in ``hand_xyz`` / ``yaw_deg`` (which way the palm faces) / ``xy_half_range``
    (their own lateral-vs-front/back jitter scale).
    """
    hand_r = Marker("hand_R", hand_xyz, _HAND_RGBA, yaw_deg=yaw_deg, collidable=True,
                     jitter_half_range_xyz=(xy_half_range, xy_half_range, HANDOFF_CUBE_Z_HALF_RANGE_M))
    cube_1 = Marker("cube_1", (hand_xyz[0], hand_xyz[1], hand_xyz[2] + _HAND_PALM_H + CUBE_HALF),
                     pos_follows_hand=hand_r.name)
    return cube_1, hand_r


# --------------------------------------------------------------------------- #
# The six scenarios (parametrized on table height; tune layout in later steps).
# --------------------------------------------------------------------------- #
def make_scenarios(table_z: float = DEFAULT_TABLE_Z,
                   shoulder_z: float = _SHOULDER_Z_M,
                   far_dist: float = 0.80) -> dict[str, Scenario]:
    """Build all scenarios with workbench tops at ``table_z`` (default 24-in minimum).

    ``shoulder_z`` is the reaching robot's shoulder-joint world z; the handoff hand height is
    defined RELATIVE to it (shoulder - 0.10), not as an absolute constant, so a taller robot (g1)
    is offered the object at a proportionally higher, same-shoulder-relative handoff height -- the
    same per-robot-stature rule the per-robot ``table_z`` uses. Defaults to the humanoid shoulder
    for standalone/module-global use (SCENARIOS).
    """
    s: dict[str, Scenario] = {}
    oz = table_z + CUBE_HALF             # pick-cube CENTER z: 50 mm cube resting on the top surface
    # Close near-edge 0.20 m: a 0.51 m-deep bench at 0.10 m would put its visits under the robot
    # (torso overlaps -> contact blowup); 0.20 m clears the body, marker still in reach.
    near = TOP_DEPTH_H + 0.20            # close: near edge ~0.20 m -> center this far out
    far = TOP_DEPTH_H + far_dist         # far: near edge ~far_dist m (default 0.80 m)
    # Pick markers sit nearer the table's near edge than the table center, so the arm does not
    # reach OVER the table top to grasp (forearm/gripper collision risk with the table surface).
    # 2026-07-01 (user-directed): was 0.12 m in from the edge. Tried flush-with-edge (cube
    # half-extent + 1 cm, ~0.035 m) first, but that put the target close enough to the robot's
    # own torso (front face ~0.13 m out, near edge at 0.20 m) that the scripted-IK reach eval
    # produced a cramped, hunched-over arm/body pose reaching in under itself -- confirmed by
    # A/B rendering both insets (0.035 and the original 0.12) that the eval's squat-like pose is
    # a pre-existing v83/IK-controller artifact (near-identical at both insets, final EE error
    # ~0.10 m either way -- see pickplace_reach_env.py's main()/Args docstring: OOD-reference risk, verdict
    # never PASS even at 0.12), NOT caused by this inset. Settled on 0.08 m: visibly closer to
    # the edge than the 0.12 m baseline without pushing the target into the cramped near-torso
    # zone.
    EDGE_INSET = 0.08
    on = 0.20 + EDGE_INSET               # marker: at the near edge (~0.235 m out)
    lateral_near_y = 0.20 + LATERAL_CUBE_EDGE_INSET_M   # lateral cube mean inset from near edge
    lateral_far_y = far_dist + LATERAL_CUBE_EDGE_INSET_M
    # Extra lateral separation for left_right_close ONLY (both sides). Widens the L/R gap so the g1
    # FIXED head camera visibly cannot frame both flank cubes simultaneously (exposes the see-to-reach
    # limit on given hardware). Table center and cube shift by the SAME delta, so the cube stays on the
    # table top. Kept local to left_right_close: ``near`` is shared with front_back_close/bimanual and
    # must not move.
    LATERAL_SPREAD_M = 0.05
    lr_table_y = near + LATERAL_SPREAD_M
    lr_cube_y = lateral_near_y + LATERAL_SPREAD_M
    # Widening y pushes the cubes farther from the base overall; pull the forward (x) placement CLOSER so
    # each arm still comfortably reaches (single-arm per-cube), leaving the FIXED-FOV see-gate as the ONLY
    # failure. Local to left_right_close (LATERAL_CUBE_X is shared with far/bimanual). Still on the flank
    # table top (long 48-in edge faces the robot -> ample x extent).
    lr_cube_x = 0.20
    # Front/back table inner edges sit 0.20 m from the base. Cube mean = edge + CUBE_HALF + std
    # (FRONT_BACK_CUBE_EDGE_INSET_M), so both x and y jitter are continuous symmetric uniform at
    # FRONT_BACK_CUBE_STD_M: the 1-std-inward x sample is near-face flush at the 0.20 edge and the pair
    # hugs the edge. COM stays over the table (min center 0.203 > 0.20), so no tip -- support holds
    # without the former two-point x hack.

    # 1. Left + Right, close, stationary. Diagonal front-left (+x, +y) vs rear-right (-x, -y).
    s["left_right_close"] = Scenario(
        name="left_right_close",
        props=(workbench("table_L", 0.0, lr_table_y, table_z, yaw=90.0)
                   + workbench("table_R", 0.0, -lr_table_y, table_z, yaw=90.0)),
        pick=[Marker("cube_0", (lr_cube_x, lr_cube_y, oz),
                     jitter_half_range_xyz=(LATERAL_CUBE_HALF_RANGE_M, LATERAL_CUBE_HALF_RANGE_M, 0.0)),
              Marker("cube_1", (-lr_cube_x, -lr_cube_y, oz),
                     jitter_half_range_xyz=(LATERAL_CUBE_HALF_RANGE_M, LATERAL_CUBE_HALF_RANGE_M, 0.0))],
    )

    # 2. Left + Right, far (1 m) -> walk in, then pick. Unlike close, widen world x, the yawed bench's
    # long axis, while retaining the shared depth jitter and near-edge inset.
    lr_far_table_y = far + LATERAL_SPREAD_M                    # mirror lr_table_y (near -> far)
    lr_far_cube_y = lateral_far_y + LATERAL_SPREAD_M           # mirror lr_cube_y (same inset, far)
    # Cube mean CENTERED on the long axis (world x), unlike left_right_close's +-lr_cube_x diagonal. The
    # close case offsets fore/aft so each arm reaches its own cube without a cross-body sweep; here the
    # robot walks to each table separately, so the diagonal buys nothing and its 0.20 m offset spent a
    # third of the table's long half-length (0.6096) before jitter even started -- capping the widening
    # at std 0.219. Centered, both far cases share the same 0.335 ceiling (user-directed 2026-08-05).
    lr_far_cube_x = 0.0
    s["left_right_far"] = Scenario(
        name="left_right_far",
        props=(workbench("table_L", 0.0, lr_far_table_y, table_z, yaw=90.0)
                   + workbench("table_R", 0.0, -lr_far_table_y, table_z, yaw=90.0)),
        pick=[Marker("cube_0", (lr_far_cube_x, lr_far_cube_y, oz),
                     jitter_half_range_xyz=(FAR_LONG_SIDE_HALF_RANGE_M,
                                            LATERAL_CUBE_HALF_RANGE_M, 0.0)),
              Marker("cube_1", (-lr_far_cube_x, -lr_far_cube_y, oz),
                     jitter_half_range_xyz=(FAR_LONG_SIDE_HALF_RANGE_M,
                                            LATERAL_CUBE_HALF_RANGE_M, 0.0))],
        stationary=False,
    )

    # 3. Front + Back, close, stationary. HEADLINE: rear camera + rear reach.
    s["front_back_close"] = Scenario(
        name="front_back_close",
        props=(workbench("table_F", near, 0.0, table_z, yaw=0.0)
                   + workbench("table_B", -near, 0.0, table_z, yaw=0.0)),
        pick=[Marker("cube_0", (FRONT_BACK_CUBE_X, 0.0, oz), collidable=True,
                     jitter_half_range_xyz=(FRONT_BACK_CUBE_HALF_RANGE_M,
                                            FRONT_BACK_CUBE_HALF_RANGE_M, 0.0)),
              Marker("cube_1", (-FRONT_BACK_CUBE_X, 0.0, oz), collidable=True,
                     jitter_half_range_xyz=(FRONT_BACK_CUBE_HALF_RANGE_M,
                                            FRONT_BACK_CUBE_HALF_RANGE_M, 0.0))],
    )

    # 4. Front + Back, far -> walk/turn to service both. Keeps front_back_close's near-edge inset and
    #    depth (world-x) jitter; ``fb_far_x`` = close cube x shifted out by the same near->far table
    #    displacement. Like left_right_far, the FAR case widens the table's LONG axis only -- here the
    #    tables are yaw=0, so the long axis is world y (half-length TOP_LEN_H = 0.6096, so the 0.173 m
    #    half-range leaves ~0.44 m margin).
    fb_far_x = FRONT_BACK_CUBE_X + (far - near)
    s["front_back_far"] = Scenario(
        name="front_back_far",
        props=(workbench("table_F", far, 0.0, table_z, yaw=0.0)
                   + workbench("table_B", -far, 0.0, table_z, yaw=0.0)),
        pick=[Marker("cube_0", (fb_far_x, 0.0, oz), collidable=True,
                     jitter_half_range_xyz=(FRONT_BACK_CUBE_HALF_RANGE_M,
                                            FAR_LONG_SIDE_HALF_RANGE_M, 0.0)),
              Marker("cube_1", (-fb_far_x, 0.0, oz), collidable=True,
                     jitter_half_range_xyz=(FRONT_BACK_CUBE_HALF_RANGE_M,
                                            FAR_LONG_SIDE_HALF_RANGE_M, 0.0))],
        stationary=False,
    )

    # 4b. Front, BEYOND camera range (stationary=False -> walk in). The single cube sits past the
    #     camera FAR clip (camera_terms.FAR) at spawn, so under --camera the head camera CANNOT SEE it --
    #     hidden by RANGE, not angle. This is the ONLY scenario that exercises the DISCOVER walk-to-SEE
    #     path: the body-FSM must DRIVE the base forward until the cube enters the FAR clip, THEN
    #     APPROACH + REACH. It is also the sole regression test for removing the gimbal instant-TERMINATE
    #     (an actuated-gimbal robot that quit on a hidden pending cube would abandon this mission at
    #     spawn). Same near-edge cube inset + x/y jitter as front_back (support/reach identical); ONLY the
    #     table distance differs. ``disc_edge`` (near edge) is set well past FAR so the cube center stays
    #     range-hidden even at the inward jitter tail (min center ~= disc_edge, still > FAR).
    disc_edge = 5.20                                      # near-edge > camera_terms.FAR (5.0) -> range-hidden at spawn
    s["front_far_discover"] = Scenario(
        name="front_far_discover",
        props=workbench("table_F", disc_edge + TOP_DEPTH_H, 0.0, table_z, yaw=0.0),
        pick=[Marker("cube_0", (disc_edge + FRONT_BACK_CUBE_EDGE_INSET_M, 0.0, oz), collidable=True,
                     jitter_half_range_xyz=(FRONT_BACK_CUBE_HALF_RANGE_M,
                                            FRONT_BACK_CUBE_HALF_RANGE_M, 0.0))],
        stationary=False,
    )

    # 5. Bimanual mixed pick (stationary): left object on a table, right object handed off from
    #    a human's hand (floating point, no table) on the right side. Both picked simultaneously.
    #    hand_*.z = _SHOULDER_Z_M - 0.30 (table/reach height, same constant place_shelf uses for
    #    carried objects) -- first pass used -0.10 (chest height) but that floated the hand up
    #    near the robot's head/cameras (too high, user-reported 2026-07-01); -0.30 lands it near
    #    the table top (cube_0 z=0.635), a natural "handing something across the table" height.
    #    The hand marker (hand_R, via ``_handoff_pieces``) renders as a WHOLE STANDING PERSON
    #    (asset_zoo.scene_object.human_figure) whose palm is exactly at the marker. It used to be a
    #    detached palm+fingers cluster; the body was added because the paper's monitoring metric is
    #    vacuous against a book-sized plate -- a bare palm blocks almost no sight line, so "was the
    #    human watched" always scored visible -- and because a person is something the planner should
    #    have to route around. Consequences: ~23 extra decorative geoms and ONE new cuRobo obstacle
    #    (the torso box). The palm plate itself is byte-identical, so the cube-on-palm support math
    #    below is untouched, and the hand is still not an IK target (see plan glowing-mixing-river.md).
    #    yaw_deg=180 points fingers toward the robot (-x); the figure stands one shoulder half-span to
    #    the side of its own palm, so its offering arm is not reaching across its own body.
    # Mirror left_right_close EXACTLY: cube_1 shares cube_0's mirrored position MEAN (same XY mean,
    # z-mean = oz) and the same XY std; the ONLY handoff-specific difference is a nonzero z-std
    # (HANDOFF_CUBE_Z_HALF_RANGE_M) modelling handoff-height uncertainty. Palm sits so its top meets
    # the cube bottom at oz (_handoff_palm_z = oz - _HAND_PALM_H - CUBE_HALF), i.e. the cube rests on
    # the hand at table height -- removes the ~53 mm z-mean elevation that made bimanual_mixed_close
    # artificially asymmetric (unreachable under the tilted grasp; see MEMORY 2026-07-14).
    _handoff_palm_z = oz - _HAND_PALM_H - CUBE_HALF
    _hand_r_xyz = (LATERAL_CUBE_X, -lateral_near_y, _handoff_palm_z)
    _cube_1, _hand_r = _handoff_pieces(_hand_r_xyz, LATERAL_CUBE_HALF_RANGE_M, yaw_deg=180.0)
    s["bimanual_mixed_close"] = Scenario(
        name="bimanual_mixed_close",
        props=workbench("table_L", 0.0, near, table_z, yaw=90.0),
        pick=[Marker("cube_0", (LATERAL_CUBE_X, lateral_near_y, oz),
                     jitter_half_range_xyz=(LATERAL_CUBE_HALF_RANGE_M,
                                            LATERAL_CUBE_HALF_RANGE_M, 0.0)),
              _cube_1],
        hand=[_hand_r],
    )

    # 5b. Bimanual mixed pick, FRONT/BACK geometry (stationary): front object on a table (+x),
    #     back object handed off from a human's hand (-x) reaching toward the robot. Same handoff
    #     model as bimanual_mixed_close (palm at table height, cube rests on palm, z-jitter models
    #     handoff-height uncertainty); the layout rotates 90 deg so the two targets sit fore/aft
    #     instead of left/right -- this is the mixed-family Front/Back cell, exercising the rear
    #     camera + rear reach on the handoff side. Table cube reuses front_back_close's exact
    #     asymmetric x-jitter (keeps the footprint 10 mm inside the near edge). hand fingers point
    #     +x toward the robot (yaw_deg=0), the mirror of the -y hand's yaw_deg=180 in mixed_close.
    _fb_hand_r_xyz = (-FRONT_BACK_CUBE_X, 0.0, _handoff_palm_z)
    _fb_cube_1, _fb_hand_r = _handoff_pieces(_fb_hand_r_xyz, FRONT_BACK_CUBE_HALF_RANGE_M, yaw_deg=0.0)
    s["bimanual_mixed_front_back_close"] = Scenario(
        name="bimanual_mixed_front_back_close",
        props=workbench("table_F", near, 0.0, table_z, yaw=0.0),
        pick=[Marker("cube_0", (FRONT_BACK_CUBE_X, 0.0, oz), collidable=True,
                     jitter_half_range_xyz=(FRONT_BACK_CUBE_HALF_RANGE_M,
                                            FRONT_BACK_CUBE_HALF_RANGE_M, 0.0)),
              _fb_cube_1],
        hand=[_fb_hand_r],
    )

    # Bimanual pick on shelf: one cube on each tier, each on the natural side of its gripper so
    # L arm reaches cube_0 (at -y) and R arm reaches cube_1 (at +y) without cross-body sweep.
    # Cubes offset 0.30 m laterally so each arm wraps around the shelf to its own cube.
    spec_shelf = shelf_spec_for(shoulder_z)
    lower_z, upper_z = tier_surface_z(spec_shelf)
    cube_x = PLACE_SHELF_X_M - spec_shelf.width / 2.0 + 0.06
    s["shelf"] = Scenario(
        name="shelf",
        props=shelf("shelf", PLACE_SHELF_X_M, 0.0, spec=spec_shelf),
        pick=[Marker("cube_0", (cube_x, -0.30, lower_z + CUBE_HALF), collidable=True),
              Marker("cube_1", (cube_x, +0.30, upper_z + CUBE_HALF), collidable=True)],
    )
    s["shelf_pick_both"] = s["shelf"]   # backwards-compatibility alias
    return s


SCENARIOS: dict[str, Scenario] = make_scenarios()

# Authoritative scenario-name set. Keys are table_z/shoulder_z-independent (make_scenarios builds the
# same names for any heights), so this is the single source of valid names for callers that must
# validate a name WITHOUT importing the default-height SCENARIOS objects (e.g. per-robot phase-3/4,
# which get their height-correct scenario from scene.robot_scene, not from this dict).
SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIOS)


def all_markers(scn: Scenario) -> list[Marker]:
    """Pick + place + hand markers, in a stable order (controller-facing index order)."""
    return list(scn.pick) + list(scn.place) + list(scn.hand)


def make_scene_spec_fn(scn: Scenario, base_spec_fn=None):
    """Return a ``SceneCfg.spec_fn`` that injects this scenario's props + target markers.

    Chains ``base_spec_fn`` first, then welds scenario props (workbench/shelf pieces) via
    ``add_props_to_spec`` and adds abstract target markers. Robot velocity environments already own
    the terrain plane; this hook must not add another ground geom.

    Args:
        scn: the scenario to render.
        base_spec_fn: optional existing scene spec_fn to run before adding props.
    """
    def _add_marker(parent, m, geom_type, size, *, local_pos=None):
        g = parent.add_geom()
        g.name = f"pp_mark_{m.name}"
        g.type = geom_type
        g.size = size
        g.pos = list(m.pos if local_pos is None else local_pos)
        g.rgba = list(m.rgba)
        g.contype = 1 if m.collidable else 0
        g.conaffinity = 1 if m.collidable else 0
        g.group = 2            # robot visual group
        return g

    def _add_free_pick_cube(spec, m):
        """Add one collidable pick target as a free body initialized on its support.

        Pick cubes are dynamic in every caller, including the kinematic harness. The kinematic path
        resets their free-joint qpos deterministically and never integrates them; physics lets table and
        gripper contacts move them. Keep the established ``pp_mark_*`` geom namespace: perception,
        cuRobo's dynamic object cuboids, and diagnostics resolve those names.
        """
        body = spec.worldbody.add_body()
        body.name = f"pp_obj_{m.name}"
        body.pos = list(m.pos)
        joint = body.add_freejoint()
        joint.name = f"pp_free_{m.name}"
        geom = _add_marker(body, m, mujoco.mjtGeom.mjGEOM_BOX, [CUBE_HALF, CUBE_HALF, CUBE_HALF],
                           local_pos=(0.0, 0.0, 0.0))
        geom.mass = PICK_CUBE_MASS_KG
        geom.friction = PICK_CUBE_FRICTION
        geom.solref = PICK_CUBE_SOLREF
        geom.priority = 1       # cube's stiff solref wins the pair, not a solmix average with the table

    def spec_fn(spec: mujoco.MjSpec) -> None:
        if base_spec_fn is not None:
            base_spec_fn(spec)
        add_props_to_spec(spec, scn.props, prefix="pp_")
        # Workbench visual pieces stay non-colliding; only its invisible ``_col_*`` support geometry
        # participates in cube contact. Set it explicitly instead of inheriting MuJoCo's default.
        for geom in spec.worldbody.geoms:
            if geom.name.startswith("pp_table_") and "_col_" in geom.name:
                geom.friction = TABLE_FRICTION
        # Pick cubes are free physical bodies. Carried cubes remain static until the separate
        # carry/attachment task owns them; place pads remain abstract spheres.
        for m in scn.pick:
            _add_free_pick_cube(spec, m)
            # Add dormant equality weld constraints for dynamic grasp attachment.
            # Look up hand base body dynamically: supports both namespaced ('robot/L_base')
            # in DynamicHarness and bare ('L_base') in KinematicHarness.
            for arm_prefix in ["L", "R"]:
                base_body = spec.body(f"robot/{arm_prefix}_base") or spec.body(f"{arm_prefix}_base")
                if base_body is None:
                    continue
                eq = spec.add_equality()
                eq.name = f"pp_weld_{arm_prefix}_{m.name}"
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                eq.name1 = base_body.name
                eq.name2 = f"pp_obj_{m.name}"
                eq.active = False
        for m in scn.carried:
            _add_marker(spec.worldbody, m, mujoco.mjtGeom.mjGEOM_BOX, [CUBE_HALF, CUBE_HALF, CUBE_HALF])
        for m in scn.place:
            _add_marker(spec.worldbody, m, mujoco.mjtGeom.mjGEOM_SPHERE, [_MARK_R, 0.0, 0.0])
        for m in scn.hand:
            add_props_to_spec(spec, human_figure(m.name, *m.pos, yaw_deg=m.yaw_deg,
                                                 collidable_palm=m.collidable), prefix="pp_")

    return spec_fn
