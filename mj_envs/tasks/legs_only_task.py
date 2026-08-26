"""Shared utilities for legs-only environment configurations."""
from dataclasses import dataclass

import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


@wp.kernel
def process_randomized_actions_kernel(
    current_target: wp.array2d(dtype=float),
    next_target: wp.array2d(dtype=float),
    interpolation_rate: wp.array2d(dtype=float),
    steps_remaining: wp.array(dtype=int),
    safe_poses: wp.array2d(dtype=float),
    num_safe_poses: int,
    min_steps: int,
    max_steps: int,
    seed: int,
):
    env_id = wp.tid()
    num_joints = current_target.shape[1]
    
    # 1. Update steps
    steps = steps_remaining[env_id] - 1
    
    # 2. Check if target reached
    if steps <= 0:
        # Snap to exact next target to prevent drift
        for j in range(num_joints):
            current_target[env_id, j] = next_target[env_id, j]
            
        # Sample new next target
        state = wp.rand_init(seed, env_id)
        next_idx = wp.randi(state, 0, num_safe_poses)
        new_steps = wp.randi(state, min_steps, max_steps)
        steps_remaining[env_id] = new_steps
        
        inv_steps = 1.0 / float(new_steps)
        
        for j in range(num_joints):
            val_next = safe_poses[next_idx, j]
            next_target[env_id, j] = val_next
            interpolation_rate[env_id, j] = (val_next - current_target[env_id, j]) * inv_steps
    else:
        # Just interpolate
        steps_remaining[env_id] = steps
        for j in range(num_joints):
            current_target[env_id, j] += interpolation_rate[env_id, j]


@wp.kernel
def reset_randomized_actions_kernel(
    env_ids: wp.array(dtype=int),
    current_target: wp.array2d(dtype=float),
    next_target: wp.array2d(dtype=float),
    interpolation_rate: wp.array2d(dtype=float),
    steps_remaining: wp.array(dtype=int),
    safe_poses: wp.array2d(dtype=float),
    num_safe_poses: int,
    min_steps: int,
    max_steps: int,
    seed: int,
):
    tid = wp.tid()
    env_id = env_ids[tid]
    num_joints = current_target.shape[1]
    
    state = wp.rand_init(seed, env_id)
    
    # Sample current target
    curr_idx = wp.randi(state, 0, num_safe_poses)
    # Sample next target
    next_idx = wp.randi(state, 0, num_safe_poses)
    # Sample duration
    new_steps = wp.randi(state, min_steps, max_steps)
    
    steps_remaining[env_id] = new_steps
    inv_steps = 1.0 / float(new_steps)
    
    for j in range(num_joints):
        val_curr = safe_poses[curr_idx, j]
        val_next = safe_poses[next_idx, j]
        
        current_target[env_id, j] = val_curr
        next_target[env_id, j] = val_next
        interpolation_rate[env_id, j] = (val_next - val_curr) * inv_steps


class FixedJointPositionAction(JointPositionAction):
    """Joints held at default pose; contributes zero action dimensions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._action_dim = 0
        self._processed_actions = self._offset.clone()  # pre-allocate; reused every step

    def process_actions(self, actions):
        self._processed_actions.copy_(self._offset)  # in-place: no allocation, tracks offset changes


@dataclass(kw_only=True)
class FixedJointPositionActionCfg(JointPositionActionCfg):
    """Config for fixed joint position action."""

    def build(self, env: ManagerBasedRlEnv) -> FixedJointPositionAction:
        return FixedJointPositionAction(self, env)


@dataclass(kw_only=True)
class RandomizedJointPositionActionCfg(JointPositionActionCfg):
    """Config for randomized joint position action."""
    robot_name: str = "g1"
    min_steps: int = 100
    max_steps: int = 500

    def build(self, env: ManagerBasedRlEnv) -> "RandomizedJointPositionAction":
        return RandomizedJointPositionAction(self, env)


class RandomizedJointPositionAction(JointPositionAction):
    """Joints randomly drifting between safe poses; contributes zero action dimensions."""

    def __init__(self, cfg: RandomizedJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._action_dim = 0

        # Cache default arm joint positions for stable reset.
        # Robot resets to q_default; current_target must match to prevent a sudden jump
        # (nearest safe pose to q_default is ~0.91 rad — large enough to destabilize).
        # Same indexing as JointPositionAction uses when use_default_offset=True.
        self._q_default_arm = self._entity.data.default_joint_pos[:, self._target_ids].clone()

        # Align Warp stream with Torch's default stream for zero-overhead sync
        self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        
        # Load safe poses
        from pathlib import Path
        
        robot_name = cfg.robot_name
        if robot_name == "g1":
            robot_name = "unitree_g1"
        
        # Try to find the safe poses file in the cache directory
        file_path = Path(__file__).parent.parent / "asset_zoo" / "cache" / f"safe_arm_poses_{robot_name}.pt"
        if not file_path.exists():
            # Fallback to g1 if specific one doesn't exist
            file_path = Path(__file__).parent.parent / "asset_zoo" / "cache" / "safe_arm_poses_unitree_g1.pt"
             
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find safe arm poses file at {file_path}")

        print(f"[RandomizedJointPositionAction] Loading safe poses from {file_path}")
        data = torch.load(file_path, map_location=self.device)
        
        # Match safe pose joints to our target joints by name
        safe_joint_names = data["joint_names"]
        try:
            joint_indices = [safe_joint_names.index(name) for name in self._target_names]
            self.safe_poses = data["poses"][:, joint_indices].contiguous()
        except ValueError as e:
            # If some joints are missing, it's a critical mismatch
            print(f"[RandomizedJointPositionAction] ERROR: Joint mismatch! {e}")
            print(f"Target joints: {self._target_names}")
            print(f"Safe joints in file: {safe_joint_names}")
            raise e

        self.num_safe_poses = self.safe_poses.shape[0]
        
        # Interpolation state (pinned memory for zero-copy)
        self.current_target = torch.zeros(self.num_envs, self._num_targets, device=self.device)
        self.next_target = torch.zeros(self.num_envs, self._num_targets, device=self.device)
        self.steps_remaining = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.interpolation_rate = torch.zeros(self.num_envs, self._num_targets, device=self.device)
        
        self.min_steps = cfg.min_steps
        self.max_steps = cfg.max_steps
        
        # Warp state wrapping (zero-copy)
        self.current_target_wp = wp.from_torch(self.current_target)
        self.next_target_wp = wp.from_torch(self.next_target)
        self.interpolation_rate_wp = wp.from_torch(self.interpolation_rate)
        self.steps_remaining_wp = wp.from_torch(self.steps_remaining)
        self.safe_poses_wp = wp.from_torch(self.safe_poses)
        
        # Bind _processed_actions to current_target directly to avoid copies in process_actions
        self._processed_actions = self.current_target
        
        self._frame_count = 0

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            # Reset all
            target_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        else:
            target_ids = env_ids.to(torch.int32)
            
        num_resets = len(target_ids)
        if num_resets == 0:
            return

        wp.launch(
            kernel=reset_randomized_actions_kernel,
            dim=num_resets,
            inputs=[
                wp.from_torch(target_ids),
                self.current_target_wp,
                self.next_target_wp,
                self.interpolation_rate_wp,
                self.steps_remaining_wp,
                self.safe_poses_wp,
                self.num_safe_poses,
                self.min_steps,
                self.max_steps,
                self._frame_count  # use frame count as part of seed
            ],
            device=self.device,
            stream=self._wp_stream
        )

        # Override current_target with q_default: robot resets to q_default, arms must start
        # there too. Without this, arms jump from q_default to a random safe pose (up to
        # 0.91 rad gap) causing an instantaneous large command that can destabilize the robot.
        # Arms then interpolate smoothly from q_default toward next_target (set by kernel).
        # current_target_wp is a zero-copy alias — in-place write is immediately visible to Warp.
        # Use long() for PyTorch indexing; target_ids is int32 (required by Warp wp.from_torch).
        target_ids_pt = target_ids.long()
        self.current_target[target_ids_pt] = self._q_default_arm[target_ids_pt]
        # Recompute interpolation_rate from q_default to next_target over the sampled step count.
        steps = self.steps_remaining[target_ids_pt].float().unsqueeze(-1)  # (N, 1)
        self.interpolation_rate[target_ids_pt] = (
            self.next_target[target_ids_pt] - self._q_default_arm[target_ids_pt]
        ) / steps

    def process_actions(self, actions):
        self._frame_count += 1
        
        # Launch fused Warp kernel to update all envs in one GPU dispatch
        # All inputs are pre-cached wp.array objects for maximum performance
        wp.launch(
            kernel=process_randomized_actions_kernel,
            dim=self.num_envs,
            inputs=[
                self.current_target_wp,
                self.next_target_wp,
                self.interpolation_rate_wp,
                self.steps_remaining_wp,
                self.safe_poses_wp,
                self.num_safe_poses,
                self.min_steps,
                self.max_steps,
                self._frame_count
            ],
            device=self.device,
            stream=self._wp_stream
        )
        # Note: self._processed_actions is already aliased to self.current_target in __init__


@dataclass(kw_only=True)
class ExternalJointPositionActionCfg(JointPositionActionCfg):
    """Config for externally controlled joint position action."""

    def build(self, env: ManagerBasedRlEnv) -> "ExternalJointPositionAction":
        return ExternalJointPositionAction(self, env)


class ExternalJointPositionAction(JointPositionAction):
    """Joints controlled by an external source; contributes zero policy action dimensions.
    
    Used for integrating separate upper-body controllers (IK, Teleop, or another policy)
    with the legs-only locomotion base. Targets are set via `set_targets()`.
    """

    def __init__(self, cfg: ExternalJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._action_dim = 0
        # Initialize targets to the default offset pose
        self.targets = self._offset.clone()
        # Bind processed actions directly to the targets buffer to eliminate copies
        self._processed_actions = self.targets

    def set_targets(self, targets: torch.Tensor):
        """Sets new targets for the joints.
        
        Args:
            targets: Tensor of shape (num_envs, num_joints).
        """
        self.targets.copy_(targets)

    def process_actions(self, actions: torch.Tensor):
        """No-op as processed actions are aliased to targets buffer."""
        pass


@dataclass(kw_only=True)
class RandomizedOffsetJointPositionActionCfg(RandomizedJointPositionActionCfg):
    """Config for policy-controlled arm action additively offset from random target."""

    def build(self, env: ManagerBasedRlEnv) -> "RandomizedOffsetJointPositionAction":
        return RandomizedOffsetJointPositionAction(self, env)


class RandomizedOffsetJointPositionAction(RandomizedJointPositionAction):
    """Arms smoothly interpolate between safe poses; policy adds an additive offset on top.

    processed_target = random_interpolated_target + policy_action * scale

    Zero policy action → arm stays exactly at the random pose. A strong action penalty
    encourages near-zero outputs, so the locomotion policy minimally disturbs the arm
    trajectory driven by an external controller in deployment.
    """

    def __init__(self, cfg: RandomizedOffsetJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Restore full policy action dimensions (parent zeroed them out)
        self._action_dim = self._num_targets
        # Un-alias _processed_actions so we can compute target + delta separately
        self._processed_actions = torch.zeros_like(self.current_target)

    def process_actions(self, actions: torch.Tensor):
        # 1. Advance warp random interpolation (updates self.current_target in-place)
        super().process_actions(None)
        # 2. Add policy delta on top of the updated random target
        self._raw_actions[:] = actions
        self._processed_actions.copy_(self.current_target + self._raw_actions * self._scale)


@dataclass(kw_only=True)
class CommandGatedRandomizedArmActionCfg(RandomizedOffsetJointPositionActionCfg):
    locomotion_vel_threshold: float = 0.3  # m/s; above this → locomotion mode
    loco_arm_range: float = 0.5            # rad; max per-joint deviation from loco anchor
    loco_anchor_range: float = 0.3         # rad; per-episode anchor sampled from default ± this
    loco_min_steps: int = 80               # min hold steps during locomotion (1.6s at 50Hz)
    command_name: str = "twist"

    def build(self, env: ManagerBasedRlEnv) -> "CommandGatedRandomizedArmAction":
        return CommandGatedRandomizedArmAction(self, env)


class CommandGatedRandomizedArmAction(RandomizedOffsetJointPositionAction):
    """Arm randomization soft-gated by locomotion command magnitude.

    During locomotion (|cmd_vel_xy| > locomotion_vel_threshold): clamps next_target
    to loco_anchor ± loco_arm_range and enforces steps_remaining >= loco_min_steps.
    During stationary: normal random target sampling, full range, normal hold duration.

    loco_anchor is re-randomized each episode (default_joint_pos ± loco_anchor_range),
    adding diversity to the locomotion arm pose distribution so the policy never trains
    exclusively at the exact default pose during locomotion.

    Motivation: in deployment the arm moves only when the robot is stationary. Training
    large random arm excursions during locomotion is off-distribution and degrades
    velocity tracking. Reactive margin (loco_arm_range) is preserved so the policy
    can still make small arm corrections for balance.

    Implementation: post-super hook. Warp kernel runs first (advances current_target,
    may resample next_target); we then clamp next_target and recompute interpolation_rate
    toward the clamped target. Changes take effect on the next kernel step (zero-copy
    PyTorch↔Warp aliasing means writes are immediately visible to the next wp.launch).
    """

    cfg: CommandGatedRandomizedArmActionCfg

    def __init__(self, cfg: CommandGatedRandomizedArmActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._loco_vel_threshold: float = cfg.locomotion_vel_threshold
        self._loco_arm_range: float = cfg.loco_arm_range
        self._loco_anchor_range: float = cfg.loco_anchor_range
        self._loco_min_steps: int = cfg.loco_min_steps
        self._twist_command_name: str = cfg.command_name
        # default_joint_pos: fixed geometry reference, safe to cache at init
        self._default_arm_pos: torch.Tensor = (
            self._entity.data.default_joint_pos[:, self._target_ids].clone()
        )
        # loco_anchor: per-episode randomized neutral pose; re-sampled in reset()
        self._loco_anchor: torch.Tensor = self._default_arm_pos.clone()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        # Re-sample locomotion anchor: default ± uniform(-loco_anchor_range, +loco_anchor_range)
        if env_ids is None or isinstance(env_ids, slice):
            ids = slice(None)
        else:
            ids = env_ids.long()
        noise = torch.empty_like(self._loco_anchor[ids]).uniform_(
            -self._loco_anchor_range, self._loco_anchor_range
        )
        self._loco_anchor[ids] = self._default_arm_pos[ids] + noise

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)  # Warp kernel: advance interpolation, maybe resample

        twist = self._env.command_manager.get_command(self._twist_command_name)
        locomoting = twist[:, :2].norm(dim=1) > self._loco_vel_threshold  # (num_envs,) bool
        if not locomoting.any():
            return

        anchor = self._loco_anchor[locomoting]                        # (n_loco, n_joints)

        # Clamp sampled next_target to anchor ± loco_arm_range
        loco_next_target = anchor + (self.next_target[locomoting] - anchor).clamp(
            -self._loco_arm_range, self._loco_arm_range
        )
        self.next_target[locomoting] = loco_next_target

        # Redirect interpolation_rate toward clamped target (takes effect next Warp step)
        steps_left = self.steps_remaining[locomoting].float().unsqueeze(1).clamp(min=1.0)
        self.interpolation_rate[locomoting] = (
            loco_next_target - self.current_target[locomoting]
        ) / steps_left

        # Enforce minimum hold: single clamp avoids extra boolean tensor + branch
        self.steps_remaining[locomoting] = self.steps_remaining[locomoting].clamp(
            min=self._loco_min_steps
        )


def apply_legs_only_modifiers(
    cfg: ManagerBasedRlEnvCfg,
    leg_pattern: str,
    arm_pattern: str,
    robot_name: str,
    randomize_arms: bool = True,
    observe_com: bool = True,
    observe_full_joints: bool = False,
    arm_action_cfg=None,
):
    """Integrates legs-only locomotion modifications into a standard environment config.

    Restricts primary actions, observations, and rewards to leg joints, while holding
    arm/other joints at randomized safe poses (or default pose) accelerated by Warp kernels.

    Args:
        cfg: The environment configuration to modify in-place.
        leg_pattern: Regex matching leg joints to be controlled/observed.
        arm_pattern: Regex matching arm/other joints to be randomized.
        robot_name: Name of the robot (for loading safe poses).
        randomize_arms: If True, uses RandomizedJointPositionAction for arms.
                        If False, uses FixedJointPositionAction (holding default pose).
                        Ignored when arm_action_cfg is provided.
        arm_action_cfg: If provided, assigned directly to cfg.actions["joint_pos_arms"]
                        instead of creating a default RandomizedJointPositionActionCfg or
                        FixedJointPositionActionCfg. Use this when the caller needs a custom
                        arm action (e.g. RandomizedOffsetJointPositionActionCfg) to avoid a
                        redundant intermediate assignment.
    """
    import re

    # 1. Action modification
    # Restrict legs to leg_pattern and add arm actions (randomized, fixed, or custom)
    orig_scale = cfg.actions["joint_pos"].scale
    orig_use_default_offset = cfg.actions["joint_pos"].use_default_offset

    # Filter the original scale dictionary to only include arm/uncontrolled joints
    arm_scale = {}
    if isinstance(orig_scale, dict):
        for k, v in orig_scale.items():
            if re.search(arm_pattern, k):
                arm_scale[k] = v
    else:
        arm_scale = orig_scale

    cfg.actions["joint_pos"].actuator_names = (leg_pattern,)

    if arm_action_cfg is not None:
        cfg.actions["joint_pos_arms"] = arm_action_cfg
    elif randomize_arms:
        cfg.actions["joint_pos_arms"] = RandomizedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(arm_pattern,),
            scale=arm_scale,
            use_default_offset=orig_use_default_offset,
            robot_name=robot_name,
            min_steps=100,
            max_steps=500
        )
    else:
        cfg.actions["joint_pos_arms"] = FixedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(arm_pattern,),
            scale=arm_scale,
            use_default_offset=orig_use_default_offset,
        )

    # 2. Observation modification
    # Restrict joint pos/vel to legs and add fused COM observation
    if "actor" in cfg.observations:
        terms = cfg.observations["actor"].terms
        if not observe_full_joints:
            for term in ["joint_pos", "joint_vel"]:
                if term in terms:
                    terms[term].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(leg_pattern,))
        
        if observe_com:
            from mjlab.managers.observation_manager import ObservationTermCfg
            from tasks.humanoid_velocity.observation import robot_com_b
            terms["robot_com_b"] = ObservationTermCfg(func=robot_com_b)

    # 3. Reward modification
    # Restrict pose and joint limit rewards to legs
    for term in ["pose", "dof_pos_limits"]:
        if term in cfg.rewards:
            reward_cfg = cfg.rewards[term]
            reward_cfg.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(leg_pattern,))
            
            # For 'pose' reward, we must ensure the 'std' dictionaries don't contain 
            # keys that won't match any joints in the restricted asset.
            # The RewardManager/PoseReward will fail if a regex in 'std' matches 0 joints.
            if term == "pose":
                p = reward_cfg.params
                
                for std_key in ["std_standing", "std_walking", "std_running"]:
                    if std_key in p and isinstance(p[std_key], dict):
                        filtered_std = {}
                        for k, v in p[std_key].items():
                            if k == ".*":
                                filtered_std[k] = v
                                continue
                            
                            # Check if this regex matches ANY joint in our restricted leg set
                            # We use re.search on the leg_pattern itself or check if any string 
                            # matching 'k' would be matched by 'leg_pattern'.
                            # Simplest: only keep if 'k' contains keywords matching joints in leg_pattern.
                            is_relevant = False
                            for kw in ["hip", "knee", "ankle", "waist"]:
                                if kw in k.lower() and kw in leg_pattern.lower():
                                    is_relevant = True
                                    break
                            
                            if is_relevant:
                                filtered_std[k] = v
                        p[std_key] = filtered_std
