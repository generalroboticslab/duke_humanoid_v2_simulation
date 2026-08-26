"""Canonical home-pose setter for every reachability-study render path.

This is the single "stand the robot the way the paper solved it" helper, shared by
``plot_workspace_curobo`` (all figure/video/viewer entrypoints) and
``probe_orientation_quiver``. It previously lived in ``probe_orientation_quiver``, which
forced every caller into a function-local import to dodge that module's ``torch`` +
``matplotlib.use("Agg")`` module-scope side effects; keeping it here (mujoco/numpy only,
humanoid_v21 constants imported lazily) lets callers import it at module scope.
"""
from __future__ import annotations

import re

import mujoco
import numpy as np

# init_state joint names that belong to the arm/gripper chain (shoulder->wrist + gripper
# rack). Used by arm_only to keep legs at qpos0 while the arms take the home reaching pose.
_ARM_JOINT_RE = re.compile(r"shoulder|elbow|wrist|rack")


def set_home_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_only: bool = False,
    pose_overrides: dict[str, float] | None = None,
    home_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Stand the welded model at the home pose; return the base world position.

    Base at ``home_pos`` (default ``HOME_KEYFRAME.pos``, humanoid_v21) with identity orientation
    (matches the IK frame). Arm joints from the mjlab cfg init_state so the gripper sits in a natural
    reaching stance. Only affects visual context — line placement uses the returned base translation.
    Pass ``home_pos`` (e.g. G1's ``HOME_KEYFRAME.pos``) to overlay a different floating-base robot.

    arm_only: if True, apply init_state only to arm/gripper joints (shoulder..wrist, rack);
    legs and every other joint stay at qpos0 (zero pose). The workspace overlay uses this so
    the figure shows the reaching arm without a distracting standing leg stance.

    pose_overrides: if given, bypass init_state entirely — every joint stays at qpos0 (zero)
    except the named joints, set to the given radian values. For an explicit display pose.
    """
    from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import HOME_KEYFRAME, get_humanoid_v21_robot_cfg

    from mj_envs.utils.mj_home_pose import home_qpos

    base_pos = np.asarray(HOME_KEYFRAME.pos if home_pos is None else home_pos, dtype=np.float64)
    data.qpos[:] = home_qpos(model)
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[:3] = base_pos
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    jnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    src = pose_overrides if pose_overrides is not None else get_humanoid_v21_robot_cfg().init_state.joint_pos
    for key, val in src.items():
        if pose_overrides is None and arm_only and not _ARM_JOINT_RE.search(key):
            continue
        for jid, jn in enumerate(jnames):
            if jn is not None and (jn == key or re.fullmatch(key, jn)):
                data.qpos[model.jnt_qposadr[jid]] = float(val)
    mujoco.mj_forward(model, data)
    return base_pos
