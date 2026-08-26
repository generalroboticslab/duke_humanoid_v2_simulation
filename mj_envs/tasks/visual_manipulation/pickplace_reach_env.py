"""PickPlaceReachEnv: cuRobo arm reach composed on frozen humanoid policy through physics.

The robot stands and ONE arm reaches a fixed world
target (a front OR rear pick cube; symmetric shoulder ROM makes the behind-the-back reach the same
solve as the front one). A scripted POLICY (``policies.HeuristicReachPolicy``) writes the arm +
camera setpoints each step; the frozen humanoid policy (loco + arm/camera tracking) then runs on this
env's live student obs and outputs the full 31D action, its arm residual tracking the IK reference
we injected as ``arm_ref``. Composition seam is identical to CameraLearnerEnv (the trainable
active-vision env), so a learner can later replace the scripted IK exactly as A1 replaced the gaze
random-walk — this env stays FlashSAC-trainable.

Per step:
  1. transform the fixed world target into the live base_link frame (base sways while balancing),
  2. hand physics_qpos + target_in_base + arm_home_pose to ``self._policy.compute(...)``,
  3. write both commands into the passive holders via ``set_command``,
  4. run the frozen policy and step.

Design decisions:
  - Only the REACHING arm is IK-driven; the idle arm + waist are held at the default (home) pose so
    the demo isolates a single-arm reach. The solver still solves both arms (block-independent), but
    the idle arm's solution is discarded, so its target value is irrelevant.
  - Orientation target = identity in base frame (a plain forward grasp). This slice scores POSITION
    error only; a task-specific grasp orientation is a later slice.
  - The key risk being tested: the frozen policy was trained on collision-graph arm references;
    a reach pose toward a table may be mildly OOD, so the frozen policy's arm tracking (EE-to-target
    error) is the reported signal, not an assumed success.
  - All per-step control lives in ``policies.HeuristicReachPolicy``; the env is the holder writer.
    Future learned arm/camera policies subclass ``HeuristicReachPolicy`` and override .compute().

This module ALSO carries the eval/render CLI driver (``main``, run as
``python pickplace_reach_env.py [scenario] --render ...``) -- merged from the former
``eval_pickplace_reach.py`` 2026-07-01 (user-directed). Reason: the driver's render camera needs
env 0's world-frame origin (``env.scene.env_origins[0]``; mujoco_warp grid-replicates envs, so env 0
is NOT generally at world (0,0,0)), which is the same quantity the env itself uses to compute
``reach_target_w``. Keeping the driver in a separate file let that origin math be re-derived
independently in each place and drift apart (the bug that prompted this merge: the driver's camera
assumed env 0 sat at the origin, framing the wrong point in world space). One file, one place that
touches ``env_origins`` -- the CLI driver reads it straight off the ``env`` it just built.

This module ALSO carries the ``--walk`` locomotion mission (formerly walk_to_reach_eval.py, merged
2026-07-16): un-gag the frozen policy's twist command and drive the base in with HeuristicMovingPolicy
+ ReachabilityGate, SEARCH(gaze) -> DECIDE -> [walk -> reach (direct IK) -> park] per cube. Same env
(PickPlaceReachEnv), same frozen policy; the only difference from the stationary reach is that the
twist channel is live and a mission state machine schedules the base motion + per-visit reach.

Run (from repo root; args via tyro, ``--help`` for all):
  python mj_envs/tasks/visual_manipulation/pickplace_reach_env.py \
      --scenario NAME --steps N --envs N --side left|right --target front|rear [--render] [--view]
  scenario defaults to "front_back_close"; --target front = largest +x cube, rear = smallest -x cube.
  --view opens a live real-time mujoco viewer of env 0; --render writes an offscreen MP4.
  --walk runs the locomotion mission instead (needs a >= 2-cube scenario, e.g. front_back_far).
"""

from __future__ import annotations

import math
import os
import sys
import time
import inspect
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import torch
import tyro
import warp as wp

# ``mj_envs`` (so ``from tasks.x`` and ``from utils.ik_mink`` resolve) + repo root (so
# ``from mj_envs.utils.ik_mink_local`` inside ``utils/ik_mink.py`` resolves). Only needed when run
# directly as a script (``__package__`` unset); harmless no-op when imported as part of the package.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import glfw  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from mjlab.utils.lab_api.math import matrix_from_quat  # noqa: E402

from asset_zoo.humanoid_v21.humanoid_v21_constants import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES  # noqa: E402
from asset_zoo.humanoid_v21.humanoid_v21_constants import HOME_KEYFRAME  # noqa: E402
from asset_zoo.g1.g1_constants import KNEES_BENT_KEYFRAME  # noqa: E402
from flash_sac.env_wrapper import ManagerBasedRlEnvWithFinalObs  # noqa: E402
from tasks.camera_terms import _CAM_SITES  # noqa: E402
from tasks.frozen_policy import build_configured_env_cfg, load_frozen_policy  # noqa: E402
from tasks.g1_velocity.experiments import _ARM_ONLY_PATTERN  # noqa: E402
from tasks.visual_manipulation.curobo.scene import robot_scene  # noqa: E402
from tasks.visual_manipulation.curobo.ik_curobo import CuroboArmPlanner  # noqa: E402
from tasks.visual_manipulation.dynamics_executor import DynamicArmReferenceExecutor  # noqa: E402
from tasks.visual_manipulation.pickplace_scenarios import make_scene_spec_fn  # noqa: E402
from tasks.visual_manipulation.contact_metrics import TableContactMonitor  # noqa: E402
from tasks.visual_manipulation.policies import DirectReachPolicy, DualCubeSearchGaze, HeuristicReachPolicy  # noqa: E402
from tasks.visual_manipulation.moving_policy import HeuristicMovingPolicy, ReachabilityGate  # noqa: E402
from utils.ik_mink import BatchedMinkIK, HUMANOID_V21_MINK_COLLISION_PAIRS  # noqa: E402
from utils._viz_camera_fov_utils import _draw_frustum  # noqa: E402 (shared FOV-cone overlay; renderers live in mj_envs/probe/viz_camera_fov{,_video}.py)
from asset_zoo.fov_frustum import FOV_GEOM_GROUP  # noqa: E402 (single source for FOV geom group digit-key)

_EE_SITE_NAME = {"left": "end_effector_L_site", "right": "end_effector_R_site"}
_CAM_SITE_NAME = {"left": _CAM_SITES[0], "right": _CAM_SITES[1]}   # (cam_left_rgb, cam_right_rgb)
# mjlab prefixes every entity name with "<key>/" in the merged scene model that the IK solver reads
# directly; command-term joint names are entity-local (bare). Bridge the two with this namespace.
_ENTITY_NAMESPACE = "robot/"

TASK = "HumanoidRmaVelEstArmFlashSacv145MixedArmsCam"


@dataclass(frozen=True)
class RobotSpec:
    """Per-robot phase-4 capability descriptor. Drives every robot-conditional branch in the env +
    CLI so the control spine stays identical: which frozen policy, which phase-3 scene, and which
    optional holders (camera gaze / actuated gripper / reactive mink-IK) that robot supports.

    ``uses_mink=False`` marks the lean executor robot (g1): cuRobo plans one arm route, the
    DynamicArmReferenceExecutor retimes it, and no camera/gripper/mink machinery is built. The
    humanoid variants (``uses_mink=True``) keep the dual-arm reactive-IK + gaze + gripper stack.
    """
    task: str
    scene_key: str
    has_camera: bool          # build the camera_ref gaze holder + drive it in step()
    has_gripper: bool         # build the zero-dim runtime_gripper term (actuated parallel gripper)
    uses_mink: bool           # build BatchedMinkIK for reactive direct/gaze/walk arm control
    expected_action_dim: int
    arm_actuator_names: tuple | None = None       # build_configured_env_cfg reach seam (None = default)
    camera_actuator_names: tuple | None = None
    stand_keyframe_z: float | None = None         # base qpos0 z lift so njmax sizing pose stands (g1)


ROBOT_SPECS: dict[str, RobotSpec] = {
    "v2": RobotSpec(task=TASK, scene_key="v2", has_camera=True, has_gripper=True, uses_mink=True,
                    expected_action_dim=31),
    "v2_fixed": RobotSpec(task=TASK, scene_key="v2_fixed", has_camera=True, has_gripper=True,
                          uses_mink=True, expected_action_dim=31),
    "g1": RobotSpec(task="G1RmaVelEstArmFlashSacL2T", scene_key="g1", has_camera=False,
                    has_gripper=False, uses_mink=False, expected_action_dim=29,
                    arm_actuator_names=(_ARM_ONLY_PATTERN,), camera_actuator_names=None,
                    stand_keyframe_z=float(KNEES_BENT_KEYFRAME.pos[2])),
}


class PickPlaceReachEnv(ManagerBasedRlEnvWithFinalObs):
    """Frozen v123 tracks a cuRobo arm reference through physics on Phase-3 scene geometry.

    The Phase-3 ``v2`` descriptor remains scene authority: it supplies the humanoid-relative
    table height, props, and markers through ``robot_scene``. Its actuated parallel grippers are
    installed as metadata-derived zero-dimension runtime targets, while v123 retains its trained
    31-dimensional observation/action contract.
    """

    def __init__(self, *, cfg, device, policy_task: str, policy_checkpoint: str,
                 target_offset, robot: str = "v2", robot_cfg=None,
                 reaching_side: str = "right", ik_num_iters: int = 8,
                 mink_min_ee_height: float | None = None, mink_min_ee_height_cost: float = 1.0,
                 near_target_height_margin: float | None = None,
                 near_target_radius: float = 0.15,
                 near_target_height_cost: float = 1.0,
                 record_contacts: bool = False, actuated_gripper: bool = False,
                 **kwargs):
        super().__init__(cfg=cfg, device=device, **kwargs)
        assert reaching_side in ("left", "right"), reaching_side
        self.reaching_side = reaching_side
        self._spec = ROBOT_SPECS[robot]

        self._policy_fn, self._policy_obs_group, action_dim = load_frozen_policy(
            policy_task, policy_checkpoint, device=str(device), num_oracle_envs=2, close_oracle=True)
        assert action_dim == self._spec.expected_action_dim, (
            f"frozen {robot} action_dim {action_dim} != {self._spec.expected_action_dim}")

        # Lean executor robot (g1): cuRobo plan-once + DynamicArmReferenceExecutor, no camera/gripper/
        # mink stack. Returns early; the rest of this ctor is the humanoid reactive-IK + gaze + gripper
        # build. ``self._policy is None`` selects the executor path in step()/reset() below.
        if not self._spec.uses_mink:
            self._init_executor_reach(robot_cfg=robot_cfg, target_offset=target_offset,
                                      reaching_side=reaching_side, record_contacts=record_contacts)
            return
        # Passive command holders the policy writes each step.
        self._arm_ref = self.command_manager.get_term("arm_ref")
        self._camera_ref = self.command_manager.get_term("camera_ref")
        self._runtime_gripper = None
        self.gripper_home_target = None
        entity = self.scene["robot"]
        if actuated_gripper:
            from asset_zoo.humanoid_v21.humanoid_v21_constants import HAND_REGISTRY

            self._runtime_gripper = self.action_manager.get_term("runtime_gripper")
            assert self._runtime_gripper.action_dim == 0, "runtime gripper changed v123 action shape"
            gripper_joints = tuple(
                prefix + joint.name
                for prefix, hand in HAND_REGISTRY["parallel_gripper"].hands.items()
                for joint in hand.load_module().joints
            )
            gripper_joint_ids, resolved = entity.find_joints(gripper_joints, preserve_order=True)
            assert tuple(resolved) == gripper_joints, (resolved, gripper_joints)
            # The robot's 0.9 soft-limit margin is for locomotion. Restore the gripper's full physical
            # range so its configured home (rack_y unset in HOME_KEYFRAME = natural open, q=0.0) and any
            # commanded jaw pose is reachable without reset silently clipping to the locomotion margin.
            entity.data.soft_joint_pos_limits[:, gripper_joint_ids] = entity.data.joint_pos_limits[:, gripper_joint_ids]
            from asset_zoo.parallel_gripper import PARALLEL_GRIPPER_OPEN_RACK_POS_M

            # Hold the gripper at its configured SSOT open target:
            self.gripper_home_target = (
                entity.data.default_joint_pos[:, self._runtime_gripper.target_ids]
                + PARALLEL_GRIPPER_OPEN_RACK_POS_M
            )
            self._runtime_gripper.set_targets(self.gripper_home_target)
        assert hasattr(self._arm_ref, "set_command"), "arm_ref must be PassiveArmRefCommandTerm"
        assert hasattr(self._camera_ref, "set_command"), "camera_ref must be PassiveGazeCommandTerm"

        # Arm pose the robot starts in (and the policy falls back to when no live plan is active).
        # Sourced from the Phase-3 planning-home (``robot_cfg.planning_home_joint_pos``): this is the
        # exact pose cuRobo seeds its cspace from, so the planner's first route begins from the same
        # physical state instead of from mjlab's hanging-arm default. The frozen policy holds the
        # non-reaching arm + waist at this snapshot each step (it only overwrites the reaching-arm
        # slice). Non-arm columns (waist + gimbal + gripper rack) keep ``entity.data.default_joint_pos``.
        from tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import HUMANOID_ARM_JOINT_HOME
        planning_home_map = (robot_cfg.planning_home_joint_pos if robot_cfg is not None
                             else HUMANOID_ARM_JOINT_HOME)
        self._arm_home_pose = entity.data.default_joint_pos[:, self._arm_ref.target_ids].clone()
        for i, name in enumerate(self._arm_ref.target_names):
            joint_name = name.rsplit("/", 1)[-1]
            if joint_name in planning_home_map:
                self._arm_home_pose[:, i] = planning_home_map[joint_name]
        # Planning-arm snapshot mirrors the home so a fresh policy reset reads the same arm pose the
        # planner seeded; the cuRobo planned policy uses this for its IK warm-start.
        self._planning_arm_pose = self._arm_home_pose.clone()

        # Mink QP IK over both arms (root/base_link-frame targets). A global QP with hard joint-limit
        # + self-collision constraints, solved per env — replaces the local-gradient DLS (batched_ik),
        # which got trapped in a local minimum / wrong kinematic branch reaching behind the torso. The
        # IK reads the merged scene model, whose body/joint/site names carry the "robot/" namespace, so
        # prefix the root body + collision-pair names too. use_yaw_frame=False keeps targets in the
        # base_link frame (as step() supplies), so no base_quat is needed in solve().
        left_arm_joint_names = [f"left_{s}_joint"  for s in ("shoulder_1", "shoulder_2", "shoulder_3",
                                                              "elbow", "wrist_1", "wrist_2", "wrist_3")]
        right_arm_joint_names = [f"right_{s}_joint" for s in ("shoulder_1", "shoulder_2", "shoulder_3",
                                                              "elbow", "wrist_1", "wrist_2", "wrist_3")]
        def _prefix_body(body: str | tuple[str, ...]) -> str | tuple[str, ...]:
            if isinstance(body, str):
                return _ENTITY_NAMESPACE + body
            return tuple(_ENTITY_NAMESPACE + name for name in body)

        mink_collision_pairs = [(_prefix_body(a), _prefix_body(b))
                                for a, b in HUMANOID_V21_MINK_COLLISION_PAIRS]
        self._ik = BatchedMinkIK(
            mj_model=self.sim.mj_model,
            num_envs=self.num_envs, device=str(self.device),
            ee_left_name=_ENTITY_NAMESPACE + _EE_SITE_NAME["left"],
            ee_right_name=_ENTITY_NAMESPACE + _EE_SITE_NAME["right"],
            left_joint_names=[_ENTITY_NAMESPACE + n for n in left_arm_joint_names],
            right_joint_names=[_ENTITY_NAMESPACE + n for n in right_arm_joint_names],
            ee_left_type="site", ee_right_type="site",
            root_body_name=_ENTITY_NAMESPACE + "base_link",
            collision_body_pairs=mink_collision_pairs,
            use_yaw_frame=False,
            mink_position_cost=5.0,
            mink_min_ee_height=mink_min_ee_height,
            mink_min_ee_height_cost=mink_min_ee_height_cost,
            mink_min_ee_height_side=reaching_side,
            mink_near_target_height_margin=near_target_height_margin,
            mink_near_target_radius=near_target_radius,
            mink_near_target_height_cost=near_target_height_cost,
            mink_near_target_height_side=reaching_side)

        # Reach target per env. The scenario coords assume "robot at its env origin, facing +x"; each
        # env's robot spawns at scene.env_origins[env] (grid-spaced so envs don't overlap), so the
        # world target = env_origin + scenario offset. z is absolute (table height), env_origin z = 0.
        # The drawn cube geom lives at the shared scenario coords (env_origin (0,0) only), so it lines
        # up visually with the env-0 robot when env 0 sits at the origin (num_envs=1).
        target_offset_w = torch.as_tensor(target_offset, device=self.device, dtype=torch.float32).view(1, 3)
        self.reach_target_w = self.scene.env_origins + target_offset_w
        self._reaching_site_id = entity.find_sites(_EE_SITE_NAME[reaching_side])[0][0]

        # Per-step controller: IK for the arm, closed-loop visual servo for the camera (aim side
        # flows from the target's x sign -- front -> left camera, rear -> right camera).
        target_x_forward = float(torch.as_tensor(target_offset, dtype=torch.float32).view(-1)[0])
        self._policy = HeuristicReachPolicy(
            ik=self._ik, ik_num_iters=int(ik_num_iters),
            reaching_side=reaching_side, target_x_forward=target_x_forward,
            arm_ref=self._arm_ref, camera_ref=self._camera_ref,
            device=self.device)
        # Aim-side camera SITE (for the real pose fov_detect servos on) -- must match the policy's
        # own aim_side, resolved AFTER construction since aim_side depends on target_x_forward.
        self._cam_site_id = entity.find_sites(_CAM_SITE_NAME[self._policy.aim_side])[0][0]

        # Reported metrics (updated each step, post-physics).
        self.reach_ee_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.reach_err = torch.zeros(self.num_envs, device=self.device)
        self.cam_aim_error = torch.zeros(self.num_envs, device=self.device)
        self.cam_in_fov = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ik_target_w = self.reach_target_w.clone()  # live waypoint-adjusted IK target, world frame
        self.arm_reference = self._arm_home_pose.clone()
        self.arm_tracking_error = torch.zeros_like(self._arm_home_pose)
        self.plan_id = 0
        self._contact_monitor = (
            TableContactMonitor(self.sim.mj_model, self.num_envs, self.step_dt) if record_contacts else None
        )
        self.table_contact_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.table_normal_impulse = torch.zeros(self.num_envs, device=self.device)
        self.table_contact_pairs: list[tuple[tuple[str, str], ...]] = [()] * self.num_envs

    def _init_executor_reach(self, *, robot_cfg, target_offset, reaching_side: str,
                             record_contacts: bool) -> None:
        """Lean cuRobo-plan-once executor reach (g1). Folded from the former g1_pickplace_reach_env.

        cuRobo plans one table-aware arm route from live qpos; DynamicArmReferenceExecutor retimes it
        into ``arm_ref`` at the control cadence; the frozen policy emits low-level actions and MuJoCo
        integrates. No camera_ref, gripper, or mink-IK is built. ``self._policy`` stays None.
        """
        assert self.num_envs == 1, "executor-reach planning is B=1 until its static-reference gate passes"
        assert robot_cfg is not None, "executor-reach robot needs a phase-3 RobotDescriptor (robot_cfg)"
        self._runtime_gripper = None
        self.gripper_home_target = None
        self._arm_ref = self.command_manager.get_term("arm_ref")
        assert hasattr(self._arm_ref, "set_command"), "arm_ref must be a passive holder"

        self._robot_cfg = replace(robot_cfg, reaching_side=reaching_side[0].upper())  # "right"->"R"
        self._controlled_joint_names = list(self._robot_cfg.arm_joints_by_side[self._robot_cfg.reaching_side])
        arm_names = list(self._arm_ref.target_names)
        missing = [name for name in self._controlled_joint_names if name not in arm_names]
        assert not missing, f"arm_ref missing descriptor joints: {missing}"
        # Hold the arm at cuRobo's reach-ready home (carried on the descriptor's home_joint_pos), NOT
        # mjlab's default arm pose: it is the pose phase-3 plans from, and the frozen policy holds it
        # upright. See the former g1 env docstring for the plan-hostile-default rationale.
        self._arm_home_pose = self.scene["robot"].data.default_joint_pos[:, self._arm_ref.target_ids].clone()
        home_map = self._robot_cfg.home_joint_pos
        for i, name in enumerate(self._arm_ref.target_names):
            key = name.split("/")[-1]
            if key in home_map:
                self._arm_home_pose[0, i] = home_map[key]
        self._executor = DynamicArmReferenceExecutor(
            self._controlled_joint_names, control_dt=float(self.step_dt))
        # planner_kwargs may carry tuning keys the CuroboArmPlanner ctor does not accept; keep only
        # accepted params and surface any dropped keys (g1's are identity-valued, so dropping is neutral).
        planner_params = set(inspect.signature(CuroboArmPlanner.__init__).parameters)
        planner_kwargs = {k: v for k, v in self._robot_cfg.planner_kwargs.items() if k in planner_params}
        dropped = set(self._robot_cfg.planner_kwargs) - set(planner_kwargs)
        if dropped:
            print(f"[pickplace_reach] dropped unsupported planner_kwargs: {sorted(dropped)}")
        self._planner = CuroboArmPlanner(
            mj_model=self.sim.mj_model,
            device=str(self.device),
            robot_cfg_dict=self._robot_cfg.build_robot_cfg_dict(
                self._robot_cfg.planning_home_joint_pos),
            base_link=self._robot_cfg.base_link,
            home_base_pos=self._robot_cfg.home_base_pos,
            robot_descriptor=self._robot_cfg,
            **planner_kwargs,
        )
        self.reach_target_w = self.scene.env_origins + torch.as_tensor(
            target_offset, device=self.device, dtype=torch.float32).view(1, 3)
        self._reaching_site_id = self.scene["robot"].find_sites(self._robot_cfg.reach_frame)[0][0]
        self.reach_ee_pos_w = torch.zeros(1, 3, device=self.device)
        self.reach_err = torch.zeros(1, device=self.device)
        self.arm_reference = self._arm_home_pose.clone()
        self.arm_tracking_error = torch.zeros_like(self._arm_home_pose)
        self.plan_id = 0
        self._contact_monitor = (
            TableContactMonitor(self.sim.mj_model, 1, self.step_dt) if record_contacts else None
        )
        self.table_contact_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.table_normal_impulse = torch.zeros(1, device=self.device)
        self.table_contact_pairs: list[tuple[tuple[str, str], ...]] = [()]
        self._planned = False
        self._policy = None

    def _target_in_base(self, entity) -> torch.Tensor:
        base_pos_w = entity.data.root_link_pos_w
        base_rot_bw = matrix_from_quat(entity.data.root_link_quat_w).transpose(-1, -2)
        return torch.matmul(base_rot_bw, (self.reach_target_w - base_pos_w).unsqueeze(-1)).squeeze(-1)

    def _plan_reach(self, target_in_base: torch.Tensor) -> None:
        # Seed the single open-loop plan from the planner's canonical reach-ready home (phase-3
        # parity), NOT the live arm: plan_pose is seed-sensitive and the frozen controller's
        # steady-state tracking error off that home alone can flip a reachable target to infeasible.
        self._planner.plan(
            target_in_base[0].detach().cpu().numpy(),
            side=self._robot_cfg.reaching_side,
            current_state=self._planner.home_state,
        )
        self._executor.load(
            self._planner.joint_names,
            self._planner.trajectory_mjlab_qpos,
            self._planner.interpolation_dt,
        )
        self._planned = True

    def _step_executor(self, hold_reference: bool):
        """Executor-reach step (g1): write the planned arm reference, run frozen inference, integrate."""
        entity = self.scene["robot"]
        if hold_reference:
            arm_command = self._arm_home_pose
        else:
            target_in_base = self._target_in_base(entity)
            if not self._planned:
                self._plan_reach(target_in_base)
            arm_command = self._executor.command(self._arm_home_pose, list(self._arm_ref.target_names))
        self._arm_ref.set_command(arm_command)
        self.arm_reference = arm_command.detach().clone()
        action = self._policy_fn({self._policy_obs_group: self.obs_buf[self._policy_obs_group]}).detach()
        out = super().step(action)
        self.reach_ee_pos_w = entity.data.site_pose_w[:, self._reaching_site_id, :3]
        self.reach_err = torch.linalg.norm(self.reach_ee_pos_w - self.reach_target_w, dim=-1)
        self.arm_tracking_error = torch.abs(
            entity.data.joint_pos[:, self._arm_ref.target_ids] - self.arm_reference)
        self.plan_id = self._executor.plan_id
        if self._contact_monitor is not None:
            qpos = wp.to_torch(self.sim.wp_data.qpos).detach().cpu().numpy()
            qvel = wp.to_torch(self.sim.wp_data.qvel).detach().cpu().numpy()
            counts, impulses = self._contact_monitor.sample(qpos, qvel)
            self.table_contact_count = torch.as_tensor(counts, device=self.device)
            self.table_normal_impulse = torch.as_tensor(impulses, device=self.device)
            self.table_contact_pairs = self._contact_monitor.last_pairs
        return out

    def reset(self, *args, **kwargs):
        out = super().reset(*args, **kwargs)
        if self._policy is None:
            self._executor.clear()
            self._planned = False
        else:
            self._policy.reset()
            if self._runtime_gripper is not None:
                self._runtime_gripper.set_targets(self.gripper_home_target)
        return out

    def step(self, _action=None, *, hold_reference: bool = False):
        if self._policy is None:
            return self._step_executor(hold_reference)
        entity = self.scene["robot"]
        base_pos_w = entity.data.root_link_pos_w                              # (N, 3)
        base_rot_wb = matrix_from_quat(entity.data.root_link_quat_w)          # (N, 3, 3) base->world
        base_rot_bw = base_rot_wb.transpose(-1, -2)                           # world->base
        target_in_base = torch.matmul(
            base_rot_bw, (self.reach_target_w - base_pos_w).unsqueeze(-1)
        ).squeeze(-1)                                                         # (N, 3) target in base frame

        # Aim-side camera's REAL pose, base frame (servo reads the actual mounted/tracked pose,
        # not an analytic assumption) -- same base_rot_bw transform as the target above.
        cam_pose_w = entity.data.site_pose_w[:, self._cam_site_id]            # (N, 7) pos+quat
        cam_pos_base = torch.matmul(
            base_rot_bw, (cam_pose_w[:, :3] - base_pos_w).unsqueeze(-1)
        ).squeeze(-1)                                                         # (N, 3)
        cam_mat_base = torch.matmul(base_rot_bw, matrix_from_quat(cam_pose_w[:, 3:7]))  # (N, 3, 3)

        # Reaching EE's REAL pose (physics site, base frame) -- feeds the waypoint's actual-progress
        # gate. Must be the REAL simulated pose, not the IK's own internal virtual FK: the IK model has
        # no table (only self-collision pairs), so if the real arm is physically blocked by the table
        # its virtual solve still "succeeds" and reports fake progress -- only the real site pose can
        # see the arm is actually stuck.
        ee_pose_w = entity.data.site_pose_w[:, self._reaching_site_id, :3]
        ee_pos_base = torch.matmul(
            base_rot_bw, (ee_pose_w - base_pos_w).unsqueeze(-1)
        ).squeeze(-1)                                                         # (N, 3)

        # Reach object's orientation in base frame (planner-policy grasp-goalset transform;
        # mink paths ignore it). The pick cube is an axis-aligned marker (identity world quat -- see
        # pickplace_scenarios._add_marker sets pos only), so object-in-base = conj(base_quat). NOTE:
        # cubes axis-aligned + grasp yaw-symmetric => this has no observable effect on the current
        # scenarios; it is correctness plumbing for a future tilted object.
        obj_quat_base = entity.data.root_link_quat_w.clone()                   # (N, 4) wxyz base->world
        obj_quat_base[:, 1:] *= -1.0                                           # conjugate = world->base

        physics_qpos = wp.to_torch(self.sim.wp_data.qpos)                     # (N, nq) live qpos in model order
        arm_command, camera_command = self._policy.compute(
            physics_qpos=physics_qpos,
            target_in_base=target_in_base,
            arm_home_pose=self._arm_home_pose,
            cam_pos_base=cam_pos_base,
            cam_mat_base=cam_mat_base,
            reach_ee_pos_base=ee_pos_base,
            # gimbal joints' REAL qpos (camera_ref column order) -> the policy's integral
            # trim, which cancels the frozen policy residual's steady gimbal offset.
            cam_joint_pos=entity.data.joint_pos[:, self._camera_ref.target_ids],
            obj_quat_base=obj_quat_base,
            planning_arm_pose=(self._planning_arm_pose
                               if getattr(self._policy, "_uses_planning_home", False) else None))
        self._arm_ref.set_command(arm_command)
        self._camera_ref.set_command(camera_command)
        self.arm_reference = arm_command.detach().clone()

        # Waypoint-adjusted IK target (base frame -> world), same base pose used to compute it above
        # -- may sit above/short of reach_target_w mid-approach (lift-transit-descend); see
        # HeuristicReachPolicy._waypoint_target.
        self.ik_target_w = base_pos_w + torch.matmul(
            base_rot_wb, self._policy.last_reach_target_base.unsqueeze(-1)
        ).squeeze(-1)

        policy_action = self._policy_fn(
            {self._policy_obs_group: self.obs_buf[self._policy_obs_group]}).detach()
        out = super().step(policy_action)

        self.reach_ee_pos_w = entity.data.site_pose_w[:, self._reaching_site_id, :3]
        self.reach_err = torch.linalg.norm(self.reach_ee_pos_w - self.reach_target_w, dim=-1)
        self.arm_tracking_error = torch.abs(
            entity.data.joint_pos[:, self._arm_ref.target_ids] - self.arm_reference)
        if self._contact_monitor is not None:
            qpos = wp.to_torch(self.sim.wp_data.qpos).detach().cpu().numpy()
            qvel = wp.to_torch(self.sim.wp_data.qvel).detach().cpu().numpy()
            counts, impulses = self._contact_monitor.sample(qpos, qvel)
            self.table_contact_count = torch.as_tensor(counts, device=self.device)
            self.table_normal_impulse = torch.as_tensor(impulses, device=self.device)
            self.table_contact_pairs = self._contact_monitor.last_pairs
        executor = getattr(self._policy, "_executor", None)
        self.plan_id = 0 if executor is None else executor.plan_id
        self.cam_aim_error = self._policy.last_aim_error
        self.cam_in_fov = self._policy.last_in_fov
        return out


class DirectReachSearchGaze(DualCubeSearchGaze):
    """DualCubeSearchGaze with DirectReachPolicy's arm: straight-line IK to the
    raw target every step (no lift/transit/descend waypoint). The mission sets
    ``_target_name`` when a visit's reach begins and clears it to park the arm."""

    def _waypoint_target(self, target_in_base: torch.Tensor, ee_now: torch.Tensor) -> torch.Tensor:
        return target_in_base


# --------------------------------------------------------------------------- #
# Eval/render CLI driver (``python pickplace_reach_env.py [scenario] --render ...``).
# --------------------------------------------------------------------------- #
@dataclass
class Args:
    """Roll out the phase-4 dynamic reach and report EE-to-cube distance."""
    robot: Literal["v2", "v2_fixed", "g1"] = "v2"
    """Which phase-4 robot. v2/v2_fixed = humanoid (dual-arm reactive-IK + gaze + gripper). g1 = lean
    cuRobo-plan-once executor reach (single-arm, no camera/gripper); g1 supports the executor reach
    only (not --walk/--gaze/--direct)."""
    scenario: str = "front_back_close"
    """Pick-place scenario from the shared Phase-3 ``robot_scene(robot, ...)`` factory."""
    steps: int | None = None
    """Rollout length. Resolved to 300 (default) or 600 (under --gaze, since search time eats
    into the budget before reach even starts) unless explicitly set."""
    envs: int = 1
    """Parallel environments. FULL PARALLEL EXECUTION works (verified 2026-07-16: --envs 32 = 32/32
    envs upright and reaching, mean final-err on par with --envs 1, ~19x rollout throughput). The
    props (tables/cubes, injected via scene.spec_fn) DO replicate per env -- mujoco_warp copies the
    scene worldbody into every world at its env_origin, and each env's reach_target_w = env_origin +
    offset lines up with its own replicated props (the pickplace_scenarios header documents this). The
    old '2026-07-01 defaulted 4->1, props not replicated' note was a MISDIAGNOSIS: the real blocker
    was mjlab's None njmax/nconmax heuristic sizing the per-world constraint/contact buffers for the
    bare robot, so adding the scenario props overflowed them ('njmax overflow') at compile. Fixed by
    setting env_cfg.sim.njmax/nconmax to fit robot+props (see build below). Default stays 1 for a
    clean single --view/--render (env 0); raise freely for batched eval. Reach residual (~0.08 m,
    verdict CHECK) is a controller-tuning matter, identical at 1 and 32 envs -- not a parallelism bug.
    """
    side: Literal["left", "right"] = "right"
    """Reaching arm."""
    target: Literal["front", "rear"] = "front"
    """front = largest +x pick cube, rear = smallest -x cube (headline symmetric-reach case)."""
    render: bool = False
    """Write an offscreen MP4 of env 0 (side view)."""
    view: bool = False
    """Open a live passive mujoco viewer (env 0, real-time)."""
    hold_home: bool = False
    """Debug: skip policy/IK/physics entirely; freeze env 0 at its post-reset default (home) joint
    pose for visual verification (see HOME_KEYFRAME in humanoid_v21_constants.py)."""
    walk: bool = False
    """Locomotion mission: SEARCH (gaze) -> DECIDE -> [walk (twist) -> reach (direct IK) -> park] per
    cube. Un-gags the frozen policy's twist command so the base walks in on far scenarios (cubes
    ~1 m out, beyond the arm envelope). Needs a scenario with >= 2 pick/carried cubes
    (default front_back_far). Mutually exclusive with --direct/--gaze/--hold-home."""
    perturb: bool = False
    """--walk only: keep the training env's runtime perturbations (interval DR + push_robot + 20 s
    episode timeout). Default False = play-style demo (same strip as gaze_search_eval)."""
    blur_omega: float = 2.0
    """--walk only: motion-blur detection gate (rad/s physical gimbal speed); 0 disables."""
    table_collision: bool | None = None
    """Real table contact. Default (None) resolves per-mode: ON for stationary reach, OFF for --walk
    (the demo walks/reaches close to the tables; the cubes are visual-only already). Explicit
    ``--table-collision`` / ``--no-table-collision`` overrides. Disabling is diagnostic-only for a
    stationary Phase-4 result: it zeroes contype/conaffinity on scenario table collision geoms."""
    gaze: bool = False
    """Gate the arm reach behind camera search: the target isn't reached from step 0 (GT), the
    arm stays home until DualCubeSearchGaze's SEARCH->ASSIGN detector actually registers+assigns
    the target cube to some camera (policies.DualCubeSearchGaze). Single-env only."""
    direct: bool = False
    """Debug: skip the lift/transit/descend waypoint -- IK straight-lines to the raw GT target
    every step (policies.DirectReachPolicy). Sanity-check for whether the open final-err >
    min-err finding is inherent to the IK/controller or introduced by the waypoint gating.
    Mutually exclusive with --gaze."""
    min_height: float | None = None
    """Soft EE-height floor (base_link frame, both arms) -- see ik_mink.MinHeightTask. None
    disables (default, no behavior change). Experimental knob for the reach-from-below bug."""
    min_height_cost: float = 1.0
    """Position-cost weight (z-axis) for --min-height when enabled."""
    near_target_height_margin: float | None = None
    """Soft EE-height floor gated on horizontal proximity to the LIVE commanded (waypoint-
    adjusted) target, not a fixed base_link value -- see ik_mink.NearTargetHeightTask. None
    disables (default, no behavior change). Replaces the structurally superseded
    --min-height mechanism (memoryless + fixed floor couldn't discriminate transit-from-home
    vs wrongly-low-near-target); gated to the reaching side only, same idle-arm rationale."""
    near_target_radius: float = 0.15
    """Horizontal (x/y) distance from the live target within which --near-target-height-
    margin can activate. Unused if that flag is None."""
    near_target_height_cost: float = 1.0
    """Position-cost weight (z-axis) for --near-target-height-margin when enabled."""
    record_contacts: bool = False
    """Read table contact pairs/normal impulse from shadow MuJoCo data. Evaluation-only CPU sync."""
    target_index: int = 1
    """--robot g1 only: index into scenario pick markers to reach (single arm cannot cross-body reach
    the opposite cube). Default left_right_close cube_1 (-y, right)."""
    seed: int | None = None
    """--robot g1 only: seed torch+numpy before env build for a reproducible cube jitter / reach."""
    debug_target_frames: int = 0
    """Per-step print of target + EE + cube in world and base frames for the first N steps
    (0 disables). Use to verify the world-frame cube and the env's reach_target_w agree, and
    that the base-frame transform lands where the arm is actually trying to go."""
    no_cube_collision: bool = False
    """Sanity-check: disable collision on the pick/cube geoms too (contype/conaffinity=0). Lets the
    arm physically contact the cube without bouncing off. Use with --table-collision False to
    isolate the arm's reachability from obstacle contacts. Evaluation-only; do not use for
    grading a real reach."""


def _wrap_no_table_collision(spec_fn):
    """Wrap a scene spec_fn so scenario table collision geoms are disabled (contype/conaffinity=0).

    The robot walks/reaches close to the tables; disabling robot-vs-table contact judges the
    scripted motion on its own (the cubes are visual-only already). Shared by the stationary reach
    (--no-table-collision) and the --walk mission (its default)."""
    def _spec_fn_no_table_collision(spec, _base=spec_fn):
        _base(spec)
        for g in spec.worldbody.geoms:
            if g.name.startswith("pp_table_") and "_col_" in g.name:
                g.contype = 0
                g.conaffinity = 0
    return _spec_fn_no_table_collision


def _wrap_no_cube_collision(spec_fn):
    """Wrap a scene spec_fn so pick/cube geoms are non-collidable (contype/conaffinity=0).

    The pick cubes remain VISIBLE (the geom itself is intact, just non-colliding). Use alongside
    --no-table-collision for a fully obstacle-free sanity check that isolates the arm's
    reachability from any contact response. Matches ``pp_mark_cube_*`` (pick) and ``pp_mark_*_shelf``
    (carried/place hand-off) geom names produced by pickplace_scenarios._add_marker."""
    def _spec_fn_no_cube_collision(spec, _base=spec_fn):
        _base(spec)
        for g in spec.worldbody.geoms:
            if g.name.startswith("pp_mark_"):
                g.contype = 0
                g.conaffinity = 0
    return _spec_fn_no_cube_collision


# arm-reach success: real EE within 5 cm of the cube (reach-line PASS bar).
REACH_OK_M = 0.05
REACH_TIMEOUT_STEPS = 250
PARK_STEPS = 60          # arm-home settle time between visits


def _run_walk_mission(args: Args, device: str) -> None:
    """Locomotion mission (formerly walk_to_reach_eval.py): find both cubes, walk (forward OR
    backward) until each is arm-reachable, then REACH it with the direct straight-line IK arm.

    SEARCH -> DECIDE -> [GO -> REACH -> PARK] x2 -> DONE. The frozen v123 policy was TRAINED to
    track twist commands; this un-gags the command channel the Reach experiment pins to zero."""
    assert not (args.direct or args.gaze or args.hold_home), (
        "--walk owns the reach path; it is mutually exclusive with --direct/--gaze/--hold-home")
    steps = args.steps if args.steps is not None else 2200
    no_table_collision = (not args.table_collision) if args.table_collision is not None else True

    # Build the scene through the SAME per-robot Phase-3 factory the reach paths use (robot_scene),
    # NOT the module-default SCENARIOS dict: the default assumes shoulder_z=0.94 / table_z=0.612, but
    # v2's shoulder is 0.921 -> table_z=0.593. Sourcing from SCENARIOS put the walk mission on a
    # different table height (and cube z) than Phase-3 and the Phase-4 reach -- they must be identical.
    spec = ROBOT_SPECS[args.robot]
    try:
        robot_cfg, _, scn, _ = robot_scene(spec.scene_key, args.scenario)
    except KeyError as exc:
        raise ValueError(f"unknown Phase-3 scenario {args.scenario!r}") from exc
    markers = list(scn.pick) + list(scn.carried)
    assert len(markers) >= 2, f"scenario {args.scenario!r} has {len(markers)} cube(s); need >= 2"

    env_cfg, exp = build_configured_env_cfg(TASK, num_envs=1, device=device, reach=True)
    from run import find_latest_checkpoint
    policy_checkpoint = find_latest_checkpoint(TASK)
    assert policy_checkpoint is not None, f"no checkpoint found under runs/{TASK}/*/model_*.pt"
    if not args.perturb:
        # Same play-mode strip as gaze_search_eval (see its comment for the full story).
        env_cfg.episode_length_s = 3600.0
        env_cfg.events = {n: t for n, t in env_cfg.events.items()
                          if getattr(t, "mode", "") != "interval"}
    # ---- un-gag the twist command channel (verified recipe; cfg-local) ---------
    # The Reach experiment pins twist ranges to (0,0) and rel_standing_envs=1.0 to
    # make a standing base. Standing envs get vel_command_b ZEROED every step and
    # heading envs get wz overwritten (mjlab velocity_command._update_command), so
    # a scripted write would be erased. Additionally the velocity-range curriculum
    # would un-pin the (0,0) ranges at the first reset (random command for a frame)
    # and velocity_tracking_failure would hard-reset a temporarily-blocked walk.
    tw_cfg = env_cfg.commands["twist"]
    tw_cfg.resampling_time_range = (1.0e9, 1.0e9)   # resample timer never fires
    tw_cfg.rel_standing_envs = 0.0                  # no standing zero-mask
    tw_cfg.rel_heading_envs = 0.0                   # no heading wz override
    env_cfg.curriculum.pop("command_vel", None)     # keep (0,0) ranges: reset draws = 0
    env_cfg.terminations.pop("velocity_tracking_failure", None)
    env_cfg.scene.spec_fn = make_scene_spec_fn(scn, base_spec_fn=env_cfg.scene.spec_fn)

    # put_data sizes the mjwarp constraint buffers by forwarding qpos0, but mjlab keeps a
    # floating base's standing height ONLY in the init_state keyframe -- qpos0 leaves the
    # robot pancaked on the floor (~292 floor contacts). Harmless for a plain humanoid (the
    # njmax heuristic absorbs it) but pickplace's extra table/cube geoms push init nefc past
    # the buffer -> "njmax overflow". Lift the base body's qpos0 to the keyframe height so the
    # sizing pose stands. Free-joint qpos is absolute, so reset (which writes the keyframe
    # qpos) is unaffected -- no double-count.
    def _spec_fn_stand_qpos0(spec, _base=env_cfg.scene.spec_fn):
        _base(spec)
        b = spec.body("robot/base_link")
        b.pos = [b.pos[0], b.pos[1], HOME_KEYFRAME.pos[2]]
    env_cfg.scene.spec_fn = _spec_fn_stand_qpos0

    if no_table_collision:
        env_cfg.scene.spec_fn = _wrap_no_table_collision(env_cfg.scene.spec_fn)
        print("  [no_table_collision] table collision geoms disabled (contype/conaffinity=0)")
    env = PickPlaceReachEnv(cfg=env_cfg, device=device, robot=args.robot, robot_cfg=robot_cfg,
                            policy_task=TASK, policy_checkpoint=policy_checkpoint,
                            target_offset=markers[0].pos, reaching_side="right")
    env.reset()

    origin0 = env.scene.env_origins[0]
    cubes_world = {mk.name: (origin0 + torch.as_tensor(mk.pos, device=device)).tolist()
                   for mk in markers}
    print(f"==== walk-to-reach eval: scenario={args.scenario!r} cubes="
          f"{ {k: [round(v, 3) for v in p] for k, p in cubes_world.items()} } steps={steps} ====")

    pol = DirectReachSearchGaze(ik=env._ik, ik_num_iters=8, reaching_side="right",
                                target_x_forward=1.0, arm_ref=env._arm_ref,
                                camera_ref=env._camera_ref, device=device,
                                blur_omega_max=args.blur_omega)

    # ---- line-of-sight gate: EXACT mj_ray (same as gaze_search_eval) -----------
    los_model = env.sim.mj_model
    los_data = mujoco.MjData(los_model)
    _sid = {s: mujoco.mj_name2id(los_model, mujoco.mjtObj.mjOBJ_SITE, f"robot/cam_{s}_rgb")
            for s in ("left", "right")}
    _own_body = {s: int(los_model.site_bodyid[_sid[s]]) for s in ("left", "right")}
    _cube_np = {cid: np.asarray(p, dtype=np.float64) for cid, p in cubes_world.items()}
    _ray_groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    _synced_at = [-1]
    blockers: dict[str, int] = {}

    def los_fn(side: str, cid: str) -> bool:
        if _synced_at[0] != pol._t:
            los_data.qpos[:] = wp.to_torch(env.sim.wp_data.qpos)[0].detach().cpu().numpy()
            mujoco.mj_forward(los_model, los_data)
            _synced_at[0] = pol._t
        cam = los_data.site_xpos[_sid[side]]
        d = _cube_np[cid] - cam
        rng = float(np.linalg.norm(d))
        if rng < 1e-6:
            return True
        u = d / rng
        pnt = cam + 0.03 * u
        gid = np.zeros(1, dtype=np.int32)
        hit = mujoco.mj_ray(los_model, los_data, pnt, u, _ray_groups, 1,
                            _own_body[side], gid)
        clear = hit < 0 or hit >= rng - 0.03 - 0.06
        if not clear:
            name = mujoco.mj_id2name(los_model, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or f"geom{int(gid[0])}"
            blockers[name] = blockers.get(name, 0) + 1
        return clear

    pol.bind(env, cubes_world, los_fn=los_fn)
    env._policy = pol

    # ---- moving stack -----------------------------------------------------------
    step_dt = float(getattr(env, "step_dt", 0.02))
    mover = HeuristicMovingPolicy(step_dt=step_dt)
    gates = {cid: ReachabilityGate(device=device) for cid in cubes_world}
    twist = env.command_manager.get_term("twist")
    entity = env.scene["robot"]

    cube_t = {cid: torch.as_tensor(p, dtype=torch.float32, device=device)
              for cid, p in cubes_world.items()}

    def target_in_base_xy(cid: str) -> tuple[float, float, float]:
        """(x_base, y_base, dist_xy) of cube `cid` from the live base pose."""
        root_pos = entity.data.root_link_pos_w[0]
        R_bw = matrix_from_quat(entity.data.root_link_quat_w).transpose(-1, -2)[0]
        p_b = R_bw @ (cube_t[cid] - root_pos)
        return float(p_b[0]), float(p_b[1]), float(torch.linalg.norm(p_b[:2]))

    def visit_cost_s(cid: str) -> float:
        """Estimated walk time to the reach floor (no turn term: nothing ever turns)."""
        _, _, d = target_in_base_xy(cid)
        return max(0.0, d - ReachabilityGate.GEOMETRIC_FLOOR_M) / HeuristicMovingPolicy.CRUISE_VX

    def set_reaching_side(side: str) -> None:
        """Re-point the reach machinery at `side`'s arm at runtime.

        The reach line fixes the reaching arm at construction (a CLI choice
        there); here the ReachabilityGate decides PER LEG which arm's both-ring
        score won, so the stop decision and the reach execution always talk
        about the same arm. This recomputes exactly the side-derived state that
        HeuristicReachPolicy.__init__ and PickPlaceReachEnv.__init__ bake in:
        the policy's IK/arm_ref column indices + EE slot, and the env's real-EE
        readout site (feeds the reach_err metric and waypoint feedback).
        """
        left, right = list(LEFT_ARM_JOINT_NAMES), list(RIGHT_ARM_JOINT_NAMES)
        ik_out_names = left + right
        reach_names = right if side == "right" else left
        arm_ref_names = list(env._arm_ref.target_names)
        pol._reaching_side = side
        pol._reaching_ik_cols = torch.tensor([ik_out_names.index(n) for n in reach_names],
                                             device=device, dtype=torch.long)
        pol._reaching_armref_cols = torch.tensor([arm_ref_names.index(n) for n in reach_names],
                                                 device=device, dtype=torch.long)
        pol._reach_ee_idx = 0 if side == "left" else 1
        env._reaching_site_id = entity.find_sites(_EE_SITE_NAME[side])[0][0]

    # ---- viewer (same skeleton as gaze_search_eval; lookat follows the robot) ---
    viewer = view_data = mj_model = None
    if args.view:
        mj_model = env.sim.mj_model
        view_data = mujoco.MjData(mj_model)
        o = origin0.detach().cpu().numpy()
        viewer = mujoco.viewer.launch_passive(mj_model, view_data)
        viewer.opt.geomgroup[FOV_GEOM_GROUP] = 1
        viewer.cam.lookat[:] = [o[0], o[1], 0.75]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 150, -12

    _mj = env.sim.mj_model
    _edge_gids = {s: [g for g in range(_mj.ngeom)
                      if (n := mujoco.mj_id2name(_mj, mujoco.mjtObj.mjOBJ_GEOM, g))
                      and n.startswith(f"robot/cam_{s}_rgb_fov") and not n.endswith("_axis")]
                  for s in ("left", "right")}
    assert all(len(g) == 12 for g in _edge_gids.values()), f"frustum edges not found: {_edge_gids}"
    _base_alpha = {s: float(_mj.geom_rgba[_edge_gids[s][0], 3]) for s in ("left", "right")}
    _LOCK_AIM = np.radians(2.0)

    def _fov_flash() -> None:
        for s in ("left", "right"):
            seeing = pol.last_sharp.get(s, True) and any(
                pol.last_in_fov_pair.get((s, cid)) and not pol.last_occluded.get((s, cid))
                for cid in cubes_world)
            locked = (pol.assigned[s] is not None
                      and pol.last_aim_to.get((s, pol.assigned[s]), np.pi) < _LOCK_AIM)
            if (seeing or pol.assigned[s] is not None) and not locked:
                pulse = 0.5 + 0.5 * np.sin(2 * np.pi * pol._t / 16)
                alpha = 0.25 + 0.75 * pulse
            else:
                alpha = _base_alpha[s]
            for g in _edge_gids[s]:
                mj_model.geom_rgba[g, 3] = alpha

    from tasks.visual_manipulation.curobo_reach_harness import draw_target_axes
    _target_frames = [(np.asarray(p, dtype=np.float64), np.eye(3)) for p in cubes_world.values()]

    def sync_view() -> bool:
        view_data.qpos[:] = wp.to_torch(env.sim.wp_data.qpos)[0].detach().cpu().numpy()
        mujoco.mj_forward(mj_model, view_data)
        _fov_flash()
        # Waypoint trajectory: world-fixed RGB grasp-goal triads at every cube, via the SAME shared
        # ``draw_target_axes`` primitive the kinematic walk viewer calls -- so both walk paths show the
        # same trajectory and differ only in physics. Redrawn each frame (user_scn is otherwise unused;
        # the FOV frustums are model geoms, not user_scn).
        viewer.user_scn.ngeom = 0
        draw_target_axes(viewer.user_scn, _target_frames)
        # Follow the robot: the mission spans ~1.6 m of floor; exponential smoothing.
        r = entity.data.root_link_pos_w[0].detach().cpu().numpy()
        viewer.cam.lookat[0] = 0.95 * viewer.cam.lookat[0] + 0.05 * r[0]
        viewer.cam.lookat[1] = 0.95 * viewer.cam.lookat[1] + 0.05 * r[1]
        viewer.sync()
        return viewer.is_running()

    # ---- mission loop -------------------------------------------------------------
    phase = "SEARCH"
    order: list[str] = []
    current: str | None = None
    visits: dict[str, dict] = {}          # cid -> report fields
    falls = 0
    aim_log: list[tuple[str, dict]] = []  # (phase, {side: aim_rad to assigned})
    prev_pos = entity.data.root_link_pos_w[0, :2].clone()
    walked = 0.0
    last_assign = dict(pol.assigned)
    phase_t0 = 0

    def start_visit(cid: str, t: int) -> None:
        nonlocal current, phase, phase_t0
        current = cid
        visits[cid]["start_step"] = t
        mover.retarget()
        phase = "GO"
        phase_t0 = t

    for t in range(steps):
        # -- task state machine (decides this step's twist / arm target) -----------
        vx = vy = wz = 0.0
        if phase == "SEARCH" and pol.assign_final:
            costs = {cid: visit_cost_s(cid) for cid in cubes_world}
            order = sorted(costs, key=costs.get)
            for cid in order:
                root_pos = entity.data.root_link_pos_w[0]
                root_quat = entity.data.root_link_quat_w[0]
                s0 = gates[cid].score(cube_t[cid], root_pos, root_quat)
                visits[cid] = {"cost_s": costs[cid], "score0": s0}
            cost_str = {c: f"{costs[c]:.1f}s" for c in order}
            score_str = {c: f"{visits[c]['score0']:.2f}" for c in order}
            print(f"  step {t}: DECIDE — est. walk time {cost_str} -> go {order[0]!r} first "
                  f"(initial reach scores {score_str})", flush=True)
            start_visit(order[0], t)
        if phase == "GO":
            x_b, y_b, dist = target_in_base_xy(current)
            root_pos = entity.data.root_link_pos_w[0]
            root_quat = entity.data.root_link_quat_w[0]
            if gates[current].step(cube_t[current], root_pos, root_quat, dist):
                mover.stop()
            vx, vy, wz = mover.compute(x_b, y_b)
            if gates[current].latched and mover.settled:
                g = gates[current]
                arm = g.latched_arm or "right"
                visits[current].update(stop_step=t, reason=g.latch_reason,
                                     score=g.last_score, dist=dist, arm=arm,
                                     direction=mover.direction_label)
                print(f"  step {t}: {current}: REACHABLE (score {g.last_score:.2f}, "
                      f"dist {dist:.2f} m, walked {visits[current]['direction']}, "
                      f"via {g.latch_reason}) — reaching with the {arm} arm (gate winner, "
                      f"L {g.last_arm_scores['left']:.2f} / R {g.last_arm_scores['right']:.2f})", flush=True)
                # arm on: the gate-winning arm, direct straight-line IK to THIS cube.
                # Re-seed the IK warm start from the LIVE physics pose first: the
                # solver's internal joint state still holds the previous visit's
                # (possibly contorted) reach solution while the physical arm is
                # back home -- solving a new target from that stale state is how
                # second-visit reaches froze at home (EE-to-cube stuck ~0.5 m).
                pol._ik.reset(torch.tensor([0], device=device),
                              wp.to_torch(env.sim.wp_data.qpos))
                set_reaching_side(arm)
                env.reach_target_w[:] = cube_t[current]
                pol._target_name = current
                visits[current]["reach_min"] = float("inf")
                phase, phase_t0 = "REACH", t
        elif phase == "REACH":
            err = float(env.reach_err[0])
            visits[current]["reach_min"] = min(visits[current]["reach_min"], err)
            if err < REACH_OK_M or (t - phase_t0) >= REACH_TIMEOUT_STEPS:
                visits[current].update(reach_final=err, reach_steps=t - phase_t0,
                                     reach_ok=visits[current]["reach_min"] < REACH_OK_M)
                tag = "OK" if visits[current]["reach_ok"] else "TIMEOUT"
                print(f"  step {t}: {current}: reach {tag} — EE-to-cube min "
                      f"{visits[current]['reach_min']:.3f} m (now {err:.3f} m, "
                      f"{t - phase_t0} steps) — parking arm", flush=True)
                pol._target_name = None            # arm back to home pose
                phase, phase_t0 = "PARK", t
        elif phase == "PARK":
            if (t - phase_t0) >= PARK_STEPS:
                nxt = [c for c in order if "stop_step" not in visits.get(c, {})]
                if nxt:
                    start_visit(nxt[0], t)
                else:
                    phase = "DONE"
                    print(f"  step {t}: DONE — all targets visited, holding position", flush=True)

        # -- write the twist BEFORE env.step so this step's obs carry it -----------
        twist.vel_command_b[0, 0] = vx
        twist.vel_command_b[0, 1] = vy
        twist.vel_command_b[0, 2] = wz
        twist.is_standing_env[:] = False   # auto-resets re-roll these masks
        twist.is_heading_env[:] = False
        _, _, terminated, _, _ = env.step()
        if bool(terminated[0]):
            falls += 1
            print(f"  step {t}: TERMINATION (fell over) — env auto-reset; restarting mission", flush=True)
            pol.reset()
            pol._target_name = None
            for g in gates.values():
                g.reset()
            mover.retarget()
            phase, order, current = "SEARCH", [], None
            prev_pos = entity.data.root_link_pos_w[0, :2].clone()

        # -- metrics ---------------------------------------------------------------
        cur_pos = entity.data.root_link_pos_w[0, :2]
        walked += float(torch.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos.clone()
        if pol.assigned != last_assign:
            print(f"  step {t}: assigned={pol.assigned}  registry={sorted(pol.registry)}", flush=True)
            last_assign = dict(pol.assigned)
        if pol.assign_final:
            aim_log.append((phase, {s: pol.last_aim_to[(s, pol.assigned[s])] for s in ("left", "right")}))
        if viewer is not None and not sync_view():
            break

    # ---- report + verdict ---------------------------------------------------------
    print(f"\nfound: { {cid: f'step {st}' for cid, st in sorted(pol.found_step.items(), key=lambda x: x[1])} }")
    print(f"assignment: {pol.assigned}   visit order: {order}")
    for cid in order:
        L = visits.get(cid, {})
        if "stop_step" in L:
            reach = (f"reach[{L.get('arm', '?')}] min {L['reach_min']:.3f} m final {L['reach_final']:.3f} m "
                     f"({'OK' if L.get('reach_ok') else 'TIMEOUT'})"
                     if "reach_min" in L and "reach_final" in L else "reach not run")
            print(f"  {cid}: score {L['score0']:.2f} -> {L['score']:.2f}   walked {L['direction']}   "
                  f"stop dist {L['dist']:.2f} m ({L['reason']})   {reach}")
        else:
            print(f"  {cid}: NOT reached (score0 {L.get('score0', float('nan')):.2f})")
    print(f"base path length {walked:.2f} m   falls {falls}")
    if blockers:
        top = sorted(blockers.items(), key=lambda kv: -kv[1])[:3]
        print(f"top blockers: {', '.join(f'{n} x{c}' for n, c in top)}")

    all_latched = order and all("stop_step" in visits.get(c, {}) for c in order)
    all_reached = order and all(visits.get(c, {}).get("reach_ok") for c in order)
    distinct = (None not in pol.assigned.values()
                and pol.assigned["left"] != pol.assigned["right"])
    done_aims = [d for ph, d in aim_log if ph == "DONE"]
    move_aims = [d for ph, d in aim_log if ph in ("GO", "REACH", "PARK")]
    locked = False
    if done_aims:
        means = {}
        for s in ("left", "right"):
            tail = [math.degrees(d[s]) for d in done_aims][-100:]
            means[s] = float(np.mean(tail))
            move_mean = (float(np.mean([math.degrees(d[s]) for d in move_aims]))
                         if move_aims else float("nan"))
            print(f"  {s:5s} cam -> {pol.assigned[s]}: settled aim (post-mission) "
                  f"{means[s]:.2f} deg   during mission {move_mean:.2f} deg (not judged)")
        locked = all(m < 1.0 for m in means.values())
    verdict = "PASS" if (all_latched and all_reached and distinct and locked and falls == 0) else "CHECK"
    print(f"verdict={verdict}  (PASS = all cubes reach-latched AND arm-reached < {REACH_OK_M} m, "
          f"distinct camera assignment, post-mission settled aim < 1 deg, zero falls)")

    if viewer is not None:
        print("[view] metrics done — sim keeps running LIVE; close the viewer to exit", flush=True)
        while viewer.is_running():
            twist.vel_command_b[:] = 0.0
            env.step()
            if not sync_view():
                break
        viewer.close()
    env.close()


def _run_g1_reach(args: Args, device: str) -> None:
    """Lean cuRobo-plan-once executor reach (g1). Folded from the former g1_pickplace_reach_env.

    Preflight the frozen policy at the arm home (zero terminations required), plan one route, run the
    dynamic reach, and (under --view) hold the window open with ENTER-restart.
    """
    spec = ROBOT_SPECS["g1"]
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    robot_cfg, _, scenario, _ = robot_scene(spec.scene_key, args.scenario)
    assert 0 <= args.target_index < len(scenario.pick), "target_index outside scenario pick markers"
    target = scenario.pick[args.target_index]
    env_cfg, _ = build_configured_env_cfg(
        spec.task, num_envs=1, device=device, reach=True,
        reach_nominal=True,                              # deterministic eval: no obs corruption / DR
        reach_arm_actuator_names=spec.arm_actuator_names,
        reach_camera_actuator_names=spec.camera_actuator_names)   # g1 L2T has no gaze command
    env_cfg.scene.spec_fn = make_scene_spec_fn(scenario, base_spec_fn=env_cfg.scene.spec_fn)

    # put_data sizes the mjwarp constraint buffers from qpos0, but mjlab keeps a floating base's
    # standing height only in the init keyframe -- qpos0 leaves the robot pancaked, and pickplace's
    # props then push init nefc past the buffer ("njmax overflow"). Lift the base body's qpos0 to the
    # keyframe height so the sizing pose stands. Free-joint qpos is absolute, so reset is unaffected.
    base_body = f"robot/{robot_cfg.base_link}"
    def _spec_fn_stand_qpos0(spec_obj, _base=env_cfg.scene.spec_fn):
        _base(spec_obj)
        b = spec_obj.body(base_body)
        b.pos = [b.pos[0], b.pos[1], spec.stand_keyframe_z]
    env_cfg.scene.spec_fn = _spec_fn_stand_qpos0
    from run import find_latest_checkpoint
    checkpoint = find_latest_checkpoint(spec.task)
    assert checkpoint is not None, f"no checkpoint found under runs/{spec.task}/*/model_*.pt"
    env = PickPlaceReachEnv(
        cfg=env_cfg, device=device, robot="g1", robot_cfg=robot_cfg,
        policy_task=spec.task, policy_checkpoint=checkpoint,
        target_offset=target.pos, reaching_side=args.side, record_contacts=args.record_contacts)

    # Live viewer: passive window driven by env-0 qpos each step (read-only mirror, never sim state).
    # ENTER re-runs the whole episode (reset -> plan -> reach).
    viewer = view_data = mj_model = None
    reset_requested = [False]

    def _on_key(key: int) -> None:
        if key == glfw.KEY_ENTER:
            reset_requested[0] = True

    if args.view:
        mj_model = env.sim.mj_model
        view_data = mujoco.MjData(mj_model)
        viewer = mujoco.viewer.launch_passive(mj_model, view_data, key_callback=_on_key)
        tgt = env.reach_target_w[0].cpu().numpy()
        viewer.cam.lookat[:] = [tgt[0], tgt[1], 0.75]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 150.0, -12.0
        print("  [view] press ENTER to restart the episode")

    def _sync():
        if viewer is None:
            return True
        view_data.qpos[:] = wp.to_torch(env.sim.wp_data.qpos)[0].detach().cpu().numpy()
        mujoco.mj_forward(mj_model, view_data)
        viewer.sync()
        return viewer.is_running()

    steps = args.steps if args.steps is not None else 300

    def _run_episode() -> None:
        env.reset()
        preflight_falls = 0
        for _ in range(300):
            _, _, terminated, _, _ = env.step(hold_reference=True)
            preflight_falls += int(terminated.sum().item())
            if not _sync():
                return
        assert preflight_falls == 0, (
            f"g1 home-reference preflight terminated {preflight_falls} times; dynamic reach is blocked")
        peak_p95 = torch.zeros(1, device=device)
        falls = 0
        for _ in range(steps):
            _, _, terminated, _, _ = env.step()
            peak_p95 = torch.maximum(peak_p95, torch.quantile(env.arm_tracking_error, 0.95, dim=1))
            falls += int(terminated.sum().item())
            if not _sync() or reset_requested[0]:
                break
        print(f"g1 dynamic reach: target={target.name} final_err={env.reach_err.item():.3f} m "
              f"arm_p95={peak_p95.item():.4f} rad plans={env.plan_id} falls={falls} "
              f"table_contacts={int(env.table_contact_count.item())} "
              f"normal_impulse={env.table_normal_impulse.item():.5f} N*s")

    _run_episode()
    if viewer is not None:
        print("  [view] rollout done -- ENTER to restart, close the window to exit")
        while viewer.is_running():
            if reset_requested[0]:
                reset_requested[0] = False
                _run_episode()
            _sync()
    env.close()


def main(args: Args) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.robot == "g1":
        assert not (args.walk or args.gaze or args.direct), (
            "--robot g1 supports the executor reach only (not --walk/--gaze/--direct)")
        _run_g1_reach(args, device)
        return
    if args.walk:
        _run_walk_mission(args, device)
        return
    scenario = args.scenario
    steps = args.steps if args.steps is not None else (600 if args.gaze else 300)
    num_envs = args.envs
    reaching_side, target_which = args.side, args.target
    render, view, hold_home = args.render, args.view, args.hold_home
    no_table_collision = bool(args.table_collision is False)
    assert not (args.gaze and args.direct), "--gaze and --direct are mutually exclusive"
    spec = ROBOT_SPECS[args.robot]

    try:
        phase3_robot_cfg, _, scn, table_z = robot_scene(spec.scene_key, scenario)
    except KeyError as exc:
        raise ValueError(f"unknown Phase-3 scenario {scenario!r}") from exc
    assert scn.pick, f"scenario {scenario!r} has no pick object to reach"
    pick_by_x = (max if target_which == "front" else min)   # front = largest +x, rear = smallest -x
    target_marker = pick_by_x(scn.pick, key=lambda m: m.pos[0])
    reach_target = target_marker.pos
    # Lateral scenarios (left_right_close/far) place the chosen cube on ONE side of the robot
    # (cube_y > 0 = robot's left, cube_y < 0 = robot's right). Without cross-body motion, the
    # reaching arm must be on the cube's side -- otherwise the IK drives the arm down/sideways
    # into the table and never closes the gap. Auto-pair when the explicit --side is wrong, log it.
    if abs(target_marker.pos[1]) > 1e-3:
        cube_side = "left" if target_marker.pos[1] > 0 else "right"
        if cube_side != reaching_side:
            print(f"  [side-auto] cube {target_marker.name!r} sits at y={target_marker.pos[1]:+.3f} m"
                  f" ({cube_side} side) -- overriding --side {reaching_side} -> {cube_side}")
            reaching_side = cube_side
    print(f"==== reach eval: scenario={scenario!r} table_z={table_z:.3f} "
          f"target={target_which}:{target_marker.name!r}@{tuple(round(v,3) for v in reach_target)} "
          f"side={reaching_side} envs={num_envs} steps={steps} ====")

    env_cfg, exp = build_configured_env_cfg(
        spec.task, num_envs=num_envs, device=device, reach=True, reach_nominal=True,
        reach_actuated_gripper=True)
    # Per-env static props (workbenches/shelf) replicate across every mujoco_warp world (see
    # pickplace_scenarios header), so the per-world constraint/contact buffers must fit the robot PLUS
    # this scenario's props. mjlab's None heuristic sizes for the bare robot and overflows once props
    # are added ("njmax overflow"), which is what pinned this eval to num_envs=1. Size the buffers to
    # the props' extra collidable geoms so --envs>1 compiles and every world runs a real reach.
    env_cfg.sim.njmax = 4096
    env_cfg.sim.nconmax = 1024
    from run import find_latest_checkpoint
    policy_checkpoint = find_latest_checkpoint(spec.task)
    assert policy_checkpoint is not None, f"no checkpoint found under runs/{spec.task}/*/model_*.pt"
    env_cfg.scene.spec_fn = make_scene_spec_fn(scn, base_spec_fn=env_cfg.scene.spec_fn)
    if no_table_collision:
        env_cfg.scene.spec_fn = _wrap_no_table_collision(env_cfg.scene.spec_fn)
        print("  [no_table_collision] table collision geoms disabled (contype/conaffinity=0)")
    if args.no_cube_collision:
        env_cfg.scene.spec_fn = _wrap_no_cube_collision(env_cfg.scene.spec_fn)
        print("  [no_cube_collision] pick cube geoms disabled (contype/conaffinity=0)")
    env = PickPlaceReachEnv(cfg=env_cfg, device=device, robot=args.robot, robot_cfg=phase3_robot_cfg,
                            policy_task=spec.task, policy_checkpoint=policy_checkpoint,
                            target_offset=reach_target, reaching_side=reaching_side,
                            mink_min_ee_height=args.min_height, mink_min_ee_height_cost=args.min_height_cost,
                            near_target_height_margin=args.near_target_height_margin,
                            near_target_radius=args.near_target_radius,
                            near_target_height_cost=args.near_target_height_cost,
                            record_contacts=args.record_contacts, actuated_gripper=True)
    env.reset()

    mj_model = env.sim.mj_model
    render_data = mujoco.MjData(mj_model) if render else None
    # env 0's world-frame origin (mujoco_warp grid-replicates envs, so env 0 is NOT generally at
    # world (0,0,0)) -- the SAME env.scene.env_origins the class above uses for reach_target_w, read
    # straight off the env instead of re-derived (that duplication is what caused this to drift out
    # of sync with the real robot position before the merge).
    origin0 = env.scene.env_origins[0].cpu().numpy()

    if args.direct:
        env._policy = DirectReachPolicy(
            ik=env._ik, ik_num_iters=8, reaching_side=reaching_side,
            target_x_forward=reach_target[0], arm_ref=env._arm_ref,
            camera_ref=env._camera_ref, device=device)
        print("  [direct] waypoint bypassed -- IK straight-lines to the raw target")

    if args.gaze:
        cubes_world = {mk.name: (origin0 + np.asarray(mk.pos)).tolist() for mk in scn.pick}
        pol = DualCubeSearchGaze(
            ik=env._ik, ik_num_iters=8, reaching_side=reaching_side,
            target_x_forward=reach_target[0], arm_ref=env._arm_ref,
            camera_ref=env._camera_ref, device=device, target_name=target_marker.name)
        pol.bind(env, cubes_world)
        env._policy = pol
        print(f"  [gaze] cubes={ {k: [round(v, 3) for v in p] for k, p in cubes_world.items()} }")
    # Frame the whole scenario (robot + BOTH tables it stands between), not just a tight crop
    # toward the single reach target -- distance=2.6/lookat-toward-target cropped the other table
    # out of frame entirely, making the robot look like it was standing off away from a lone
    # table instead of between the front/back pair (matches the smoke-test scene camera: wide,
    # lookat centered on the robot).
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [origin0[0], origin0[1], 0.75]
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135.0, -15.0
    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(mj_model, height=720, width=1280) if render else None

    # Live viewer: passive window driven by env-0 qpos each step, paced to real time.
    # ENTER resets the env on the next loop iteration (same key as mjlab's NativeMujocoViewer /
    # run.py, whose native viewer binds KEY_ENTER -> request_reset -- matched here for muscle-memory
    # consistency even though this driver's viewer is launch_passive, not NativeMujocoViewer).
    reset_requested = [False]

    def _on_key(key: int) -> None:
        if key == glfw.KEY_ENTER:
            reset_requested[0] = True

    viewer = view_data = None
    step_dt = getattr(env, "step_dt", mj_model.opt.timestep)
    if view:
        view_data = mujoco.MjData(mj_model)
        viewer = mujoco.viewer.launch_passive(mj_model, view_data, key_callback=_on_key)
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = cam.distance, cam.azimuth, cam.elevation
        print("  [view] press ENTER to reset the env")

    # hold_home: freeze at the single post-reset qpos snapshot (== HOME_KEYFRAME, no policy/IK/
    # physics ever runs) so the viewer/render shows exactly the default pose, not a rollout of it.
    qpos_home = wp.to_torch(env.sim.wp_data.qpos)[0].detach().cpu().numpy().copy() if hold_home else None
    if hold_home:
        names, vals = env._arm_ref.target_names, env._arm_home_pose[0].cpu().numpy()
        print("  [hold_home] arm_home_pose (waist + both arms, entity.data.default_joint_pos):")
        for n, v in zip(names, vals):
            print(f"    {n:30s} {v:+.4f}")
        if env._runtime_gripper is not None:
            names = env._runtime_gripper.target_names
            targets = env.gripper_home_target[0].cpu().numpy()
            actual = env.scene["robot"].data.joint_pos[0, env._runtime_gripper.target_ids].cpu().numpy()
            print("  [hold_home] gripper qpos / configured target:")
            for n, q, target in zip(names, actual, targets):
                print(f"    {n:30s} {q:+.4f} / {target:+.4f}")

    min_err = torch.full((num_envs,), float("inf"), device=device)
    aim_err_sum = torch.zeros(num_envs, device=device)
    in_fov_count = torch.zeros(num_envs, device=device)
    arm_track_p95 = torch.zeros(num_envs, device=device)
    falls = 0
    for step_idx in range(steps):
        if view and not viewer.is_running():
            break
        if reset_requested[0]:
            reset_requested[0] = False
            env.reset()
            min_err[:] = float("inf")
            aim_err_sum.zero_()
            in_fov_count.zero_()
            falls = 0
            if hold_home:
                qpos_home = wp.to_torch(env.sim.wp_data.qpos)[0].detach().cpu().numpy().copy()
            print("  [view] env reset")
        if hold_home:
            qpos_now = qpos_home
        else:
            _obs, _rew, term, _timeout, _ = env.step()
            min_err = torch.minimum(min_err, env.reach_err)
            aim_err_sum += env.cam_aim_error
            in_fov_count += env.cam_in_fov.float()
            arm_track_p95 = torch.maximum(arm_track_p95, torch.quantile(env.arm_tracking_error, 0.95, dim=1))
            falls += int(term.sum().item())
            qpos_now = wp.to_torch(env.sim.wp_data.qpos)[0].detach().cpu().numpy()
        if args.debug_target_frames and step_idx < args.debug_target_frames:
            entity = env.scene["robot"]
            base_pos_w = entity.data.root_link_pos_w[0]
            R_bw = matrix_from_quat(entity.data.root_link_quat_w[0]).transpose(-1, -2)
            tgt_w = env.reach_target_w[0]
            tgt_b = R_bw @ (tgt_w - base_pos_w)
            ee_w = env.reach_ee_pos_w[0]
            ee_b = R_bw @ (ee_w - base_pos_w)
            wp_b = env._policy.last_reach_target_base[0] if env._policy.last_reach_target_base is not None else None
            # Compare reach_target_w against the LIVE MuJoCo geom pose of the chosen cube. The visual
            # cube is the user-perceived target; reach_target_w is the IK target. They MUST agree.
            cube_geom_name = f"pp_mark_{target_marker.name}"
            gid = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, cube_geom_name)
            if gid >= 0:
                geom_pos_w = env.sim.mj_data.geom_xpos[gid].copy()
                env0_origin = env.scene.env_origins[0].detach().cpu().numpy()
                geom_offset = geom_pos_w - env0_origin
                tgt_offset = tgt_w.detach().cpu().numpy() - env0_origin
                diff = float(np.linalg.norm(geom_pos_w - tgt_w.detach().cpu().numpy()))
                cube_str = f"  geom[{cube_geom_name}]_w=({geom_pos_w[0]:+.3f},{geom_pos_w[1]:+.3f},{geom_pos_w[2]:+.3f})  offset=({geom_offset[0]:+.3f},{geom_offset[1]:+.3f},{geom_offset[2]:+.3f})  ||geom-tgt||={diff:.3f}  target_offset=({tgt_offset[0]:+.3f},{tgt_offset[1]:+.3f},{tgt_offset[2]:+.3f})"
            else:
                cube_str = f"  geom[{cube_geom_name}] NOT FOUND in mj_model"
            print(f"  [dbg {step_idx:3d}] base_pos_w=({base_pos_w[0].item():+.3f},{base_pos_w[1].item():+.3f},{base_pos_w[2].item():+.3f})"
                  f"  cube_w=({tgt_w[0].item():+.3f},{tgt_w[1].item():+.3f},{tgt_w[2].item():+.3f})"
                  f"  EE_w=({ee_w[0].item():+.3f},{ee_w[1].item():+.3f},{ee_w[2].item():+.3f})"
                  f"  ||cube_w-EE_w||={float((tgt_w - ee_w).norm()):.3f}",
                  flush=True)
            print(f"           tgt_b=({tgt_b[0].item():+.3f},{tgt_b[1].item():+.3f},{tgt_b[2].item():+.3f})"
                  f"  EE_b=({ee_b[0].item():+.3f},{ee_b[1].item():+.3f},{ee_b[2].item():+.3f})"
                  + (f"  wp_b=({wp_b[0].item():+.3f},{wp_b[1].item():+.3f},{wp_b[2].item():+.3f})" if wp_b is not None else "  wp_b=None")
                  + cube_str,
                  flush=True)
        if render:
            render_data.qpos[:] = qpos_now
            mujoco.mj_forward(mj_model, render_data)
            renderer.update_scene(render_data, cam)
            # Overlay the aim-side camera's FOV cone (green = target in-frustum this frame, red =
            # not) so the collision fix's video also answers "can the tracked camera even see the
            # target while the arm reaches" -- previously only a numeric aim-error was reported.
            cam_pos_w = render_data.site_xpos[env._cam_site_id].copy()
            cam_mat_w = render_data.site_xmat[env._cam_site_id].reshape(3, 3).copy()
            in_fov = bool(env.cam_in_fov[0].item())
            rgba = (0.15, 0.9, 0.15, 0.35) if in_fov else (0.9, 0.15, 0.15, 0.35)
            _draw_frustum(renderer.scene, cam_pos_w, cam_mat_w, rgba,
                          axis=False, draw_near=False)
            frames.append(renderer.render())
        if view:
            view_data.qpos[:] = qpos_now
            mujoco.mj_forward(mj_model, view_data)
            viewer.sync()

    if hold_home:
        print("  [hold_home] froze at post-reset default (home) joint pose -- no policy/IK ran, no metrics")
    else:
        final_err = env.reach_err
        root_z = env.scene["robot"].data.root_link_pos_w[:, 2]
        upright = (root_z > 0.40)
        print(f"  reaching-arm EE-to-cube distance (m):  min={min_err.mean().item():.3f} "
              f"final={final_err.mean().item():.3f}  (mean over {num_envs} envs)")
        print(f"  best env final={final_err.min().item():.3f}  worst env final={final_err.max().item():.3f}")
        print(f"  upright envs: {int(upright.sum().item())}/{num_envs}  fall_rate={falls/(steps*num_envs):.4f}")
        aim_err_mean_deg = math.degrees((aim_err_sum / steps).mean().item())
        final_aim_err_deg = math.degrees(env.cam_aim_error.mean().item())
        in_fov_frac = (in_fov_count / steps).mean().item()
        print(f"  camera aim error (deg): rollout-mean={aim_err_mean_deg:.1f} final={final_aim_err_deg:.1f}  "
              f"in_fov_frac={in_fov_frac:.3f}  (closed-form parallax-corrected solve; should stay near 0)")
        print(f"  arm reference tracking: p95={arm_track_p95.mean().item():.4f} rad  plan_id={env.plan_id}")
        plan_error = getattr(env._policy, "last_plan_error", None)
        if plan_error is not None:
            print(f"  planned reach blocked: {plan_error}")
        if args.record_contacts:
            print(f"  table contacts: count={int(env.table_contact_count.sum().item())} "
                  f"normal_impulse={env.table_normal_impulse.sum().item():.5f} N*s "
                  f"pairs={env.table_contact_pairs[0]}")
        if args.gaze:
            print(f"  [gaze] assigned={pol.assigned}  found_step={pol.found_step}")
        verdict = "PASS" if (final_err.mean().item() < 0.05 and bool(upright.all())) else "CHECK"
        print(f"  verdict={verdict}  (PASS = mean final < 0.05 m and all upright)")

    if render:
        renderer.close()
        out = os.path.join(os.path.dirname(__file__), "media", f"reach_{scenario}_{target_which}_{reaching_side}.mp4")
        import imageio.v3 as iio
        iio.imwrite(out, np.stack(frames), fps=30)
        print(f"  rendered -> {out}")

    if view and viewer.is_running():
        print("  [view] rollout done — holding window open; close the viewer to exit")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 120)
    if viewer is not None:
        viewer.close()

    close_policy = getattr(env._policy, "close", None)
    if close_policy is not None:
        close_policy()
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
