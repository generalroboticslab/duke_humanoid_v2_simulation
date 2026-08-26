"""The canonical home configuration of a compiled MuJoCo model.

Split out so every consumer of a study MJCF -- URDF export, cuRobo config, visibility scoring,
figure overlays -- agrees on one answer, because for one robot the obvious answer is wrong.
"""
from __future__ import annotations

import mujoco

STANDING_KEYFRAME = "standing"


def home_qpos(model: mujoco.MjModel):
    """The model's home qpos: its ``standing`` keyframe if it declares one, else ``qpos0``.

    ``qpos0`` is the right answer for every robot whose zero pose is its standing pose, which
    is all of them except PAL TALOS: the vendor gives TALOS ``arm_*_2`` and ``arm_*_4`` limit
    intervals that EXCLUDE zero, so its ``qpos0`` is not merely a different pose but a
    mechanically unreachable one, and anything seeded from it starts out of bounds. TALOS
    therefore carries an explicit ``standing`` keyframe. Of the robots currently in the study
    only TALOS and GR-3 declare a keyframe under that name, and GR-3's is byte-identical to its
    ``qpos0``, so preferring the keyframe changes nothing for any existing robot.
    """
    for k in range(model.nkey):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) == STANDING_KEYFRAME:
            return model.key_qpos[k].copy()
    return model.qpos0.copy()


__all__ = ["STANDING_KEYFRAME", "home_qpos"]
