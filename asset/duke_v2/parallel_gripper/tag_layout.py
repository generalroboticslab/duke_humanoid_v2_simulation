"""Per-hand AprilTag layout for parallel_gripper (pure data, no heavy deps).

Left and right grippers SHARE one gripper XML but carry DIFFERENT tag IDs so an AprilTag
detector can tell the two hands apart. The 8 physical decal SLOTS (4 jaw plates + 4 base-holder pads) are identical on both
hands; only which tag ID sits in each slot differs.

Imported by:
  - make_tags.py  : generate the PNG + flat-quad OBJ for every tag id, at its slot.
  - parallel_gripper.py (loader): compose the chosen hand's 8 tags onto the gripper
    at load time (the gripper XML itself is tag-free).
"""
from __future__ import annotations

# 8 physical decal slots per hand (slot body + plate face + center/face-Z, mm).
# ALL slots are CAD ground truth since the ParallelGripper0710.step re-export (probe_step
# solid centers, ±0.001 mm; the 0626-era jaw values re-verified to <=5 um after the
# authored-jaw-pose repose — see mini_gripper_old/ParallelGripper0710/).
# JAW slots: the 4 jaw tag plates (each 20x20x2 mm, normal = Z); cx/cy = plate center,
# zface = the OUTER plate face (top +10, bottom -30). Same both hands.
# BASE slots: the 4 pocket-seated pads on the base tag-holder plates (holders + pads are
# CAD parts of the base link, baked into base.obj); pad outer faces FLUSH with the holder
# surfaces at z=+15.5 (top holder) / -35.5 (bottom holder, mirrored across the gripper's
# z=-10 symmetry plane).
SLOTS = {
    "left_top":      {"body": "left_rack",  "face": "+Z", "cx": 43.45, "cy": -45.04, "zface":  10.0},
    "left_bottom":   {"body": "left_rack",  "face": "-Z", "cx": 44.61, "cy": -44.99, "zface": -30.0},
    "right_top":     {"body": "right_rack", "face": "+Z", "cx": 44.61, "cy":  58.96, "zface":  10.0},
    "right_bottom":  {"body": "right_rack", "face": "-Z", "cx": 43.45, "cy":  59.01, "zface": -30.0},
    "base_top_a":    {"body": "base", "face": "+Z", "cx": -4.743, "cy": -5.00, "zface":  15.5},
    "base_top_b":    {"body": "base", "face": "+Z", "cx": -4.743, "cy": 17.00, "zface":  15.5},
    "base_bottom_a": {"body": "base", "face": "-Z", "cx": -4.743, "cy": -5.00, "zface": -35.5},
    "base_bottom_b": {"body": "base", "face": "-Z", "cx": -4.743, "cy": 17.00, "zface": -35.5},
}

# In-plane decal rotation (deg) compensating the cv2.aruco-gen vs pupil_apriltags-decode
# 180deg offset, so detected tag +X = +gripper X (robot forward). See make_tags.write_quad.
# Applies to EVERY slot (jaw and base alike) — the +X-forward convention is universal.
TAG_ROT = 180

# hand -> {slot: tag_id}. LEFT and RIGHT use DISJOINT id sets (distinguishable).
# FULL RENUMBER 2026-07-10 (user-chosen, all 16 ids new; verified free of every id
# block in BOTH repos: visual_servoing bodies 0-5/12-25/56-61/67-69/401-420/501-518/
# 577-586 and all prior gripper ids 563-575, now retired/deregistered):
#   L jaws 80-83, L base holder 84-87, R jaws 90-93, R base holder 94-97.
HAND_TAGS = {
    "L": {"left_top": 80, "left_bottom": 81, "right_top": 82, "right_bottom": 83,
          "base_top_a": 84, "base_top_b": 85, "base_bottom_a": 86, "base_bottom_b": 87},
    "R": {"left_top": 90, "left_bottom": 91, "right_top": 92, "right_bottom": 93,
          "base_top_a": 94, "base_top_b": 95, "base_bottom_a": 96, "base_bottom_b": 97},
}


def tag_specs() -> list[dict]:
    """Every tag (both hands) as a generation/placement spec dict."""
    out = []
    for hand, mapping in HAND_TAGS.items():
        for slot, tid in mapping.items():
            out.append({"id": tid, "hand": hand, "slot": slot, "rot": TAG_ROT, **SLOTS[slot]})
    return out


def hand_tag_ids(hand: str) -> list[int]:
    """Ordered tag ids for one hand ('L' or 'R')."""
    m = HAND_TAGS[hand]
    return [m[slot] for slot in SLOTS]


def tag_body(tag_id: int) -> str:
    """Jaw body a given tag id rides on."""
    for mapping in HAND_TAGS.values():
        for slot, tid in mapping.items():
            if tid == tag_id:
                return SLOTS[slot]["body"]
    raise KeyError(f"unknown tag id {tag_id}")
