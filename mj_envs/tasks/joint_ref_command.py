"""Joint-reference commands + a thin residual action.

Deployment model:
    operator cmd (task space)  -- async, high level
       |  conversion (IK / graph-nav / gaze-walk / heuristic / RL)   onboard
       v
    joint_ref (joint space)    -- emitted by a CommandTerm, this module
       |  + scale * action     -- policy residual, JointPosRefResidualAction
       v
    joint_target

The reference generator (warp graph nav / gaze walk) was historically baked INSIDE
the arm/camera action subclasses. That is conceptually a *command*, not action
processing: the env accepts a joint reference and the policy adds a residual on top.
This module owns the reference as a CommandTerm; the action is a thin residual reading
``.command``. How the reference is generated is interchangeable behind that interface.

Legs are intentionally NOT migrated to this pattern: they have no operator reference
today, so they stay on the base ``JointPositionAction`` whose ``offset`` (=
``default_joint_pos``, the constant home pose) IS the implicit, skipped reference --
``target = home + scale*action``, the degenerate const-ref case. Building a
const-emitting CommandTerm for legs now would ship a do-nothing placeholder (an extra
per-step ``compute`` tick, zero info) for a feature that does not exist. When legs DO
gain an operator reference (squat / knee config / crouch-height posture), un-skip them:
add a real posture ``CommandTerm`` here + flip ``cfg.actions["joint_pos"]`` to
``JointPosRefResidualActionCfg`` in ``legs_only_task.apply_legs_only_modifiers`` (one
helper edit; the residual seam below is already proven on arm/camera). That migration
retrains anyway (new behavior) and adds the moving-ref observation then -- so nothing is
pre-wired now.

Timing (vs the old action-baked generator): a CommandTerm self-ticks in
``_update_command`` at the END of the env step (after physics), so the reference the
policy acts on is the one produced by the *previous* step's compute. This is a 1-step
(20 ms) latency vs the old in-action tick -- which is exactly the real onboard pipeline
latency (conversion runs, policy consumes next tick). Intentional; makes sim match
deployment. The reset-compute pass (scalar ``dt==0``) must NOT tick (it would add a
phantom advance and phase-shift the RNG/frame cadence), hence the ``_reset_pass`` guard
in ``_update_command``. Note mjlab >=1.6.0 also passes a per-env *tensor* dt on the
auto_reset step path (zero for envs that just reset); that path is a normal step and
must still tick, so the guard tests for a scalar zero specifically.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

# Sentinel resampling interval: reference changes are event-driven inside the generator
# (graph arrival / per-env step counter), never time-driven. Only reset -> _resample_command
# seeding matters. Large value so the base CommandTerm timer never fires a resample.
_NO_TIME_RESAMPLE = (1.0e9, 1.0e9)


# =============================================================================
# Warp kernels (collision-free arm graph traversal)
# =============================================================================


@wp.kernel
def process_graph_nav_kernel(
    current_node:    wp.array(dtype=int),
    target_node:     wp.array(dtype=int),
    next_hop_node:   wp.array(dtype=int),
    steps_remaining: wp.array(dtype=int),
    current_target:  wp.array2d(dtype=float),
    interp_rate:     wp.array2d(dtype=float),
    graph_poses:     wp.array2d(dtype=float),
    next_hop_table:  wp.array2d(dtype=int),   # [N, N] int32
    N:               int,
    steps_per_edge:  int,
    seed:            int,
):
    """Per-step graph navigation kernel. One thread per env."""
    env_id = wp.tid()
    num_joints = current_target.shape[1]
    steps_left = steps_remaining[env_id] - 1

    if steps_left <= 0:
        # Advance to next_hop node and snap current_target to its pose.
        next_node = next_hop_node[env_id]
        current_node[env_id] = next_node
        for j in range(num_joints):
            current_target[env_id, j] = graph_poses[next_node, j]

        # Resample target when we have arrived.
        tgt = target_node[env_id]
        if next_node == tgt:
            state = wp.rand_init(seed, env_id)
            tgt = wp.randi(state, 0, N)
            target_node[env_id] = tgt

        # Compute next hop and set interpolation rate toward it.
        nh = next_hop_table[next_node, tgt]
        next_hop_node[env_id] = nh
        steps_remaining[env_id] = steps_per_edge
        inv_steps = 1.0 / float(steps_per_edge)
        for j in range(num_joints):
            interp_rate[env_id, j] = (graph_poses[nh, j] - current_target[env_id, j]) * inv_steps
    else:
        steps_remaining[env_id] = steps_left
        for j in range(num_joints):
            current_target[env_id, j] += interp_rate[env_id, j]


@wp.kernel
def reset_graph_nav_kernel(
    env_ids:         wp.array(dtype=int),
    current_node:    wp.array(dtype=int),
    target_node:     wp.array(dtype=int),
    next_hop_node:   wp.array(dtype=int),
    steps_remaining: wp.array(dtype=int),
    current_target:  wp.array2d(dtype=float),
    interp_rate:     wp.array2d(dtype=float),
    graph_poses:     wp.array2d(dtype=float),
    next_hop_table:  wp.array2d(dtype=int),
    N:               int,
    steps_per_edge:  int,
    seed:            int,
):
    """Reset navigation state for a subset of envs (one thread per reset env)."""
    tid = wp.tid()
    env_id = env_ids[tid]
    num_joints = current_target.shape[1]

    state = wp.rand_init(seed, env_id)
    start_node = wp.randi(state, 0, N)
    tgt = wp.randi(state, 0, N)

    current_node[env_id] = start_node
    target_node[env_id] = tgt

    nh = next_hop_table[start_node, tgt]
    next_hop_node[env_id] = nh
    steps_remaining[env_id] = steps_per_edge
    inv_steps = 1.0 / float(steps_per_edge)

    for j in range(num_joints):
        current_target[env_id, j] = graph_poses[start_node, j]
        interp_rate[env_id, j] = (graph_poses[nh, j] - graph_poses[start_node, j]) * inv_steps


# =============================================================================
# Arm-graph joint-reference command
# =============================================================================


@dataclass(kw_only=True)
class ArmGraphRefCommandTermCfg(CommandTermCfg):
    """Collision-free graph-nav joint-reference command.

    Emits ``joint_ref`` (absolute arm joint positions) by traversing a precomputed
    collision-free graph. Holds the reference half of the arm pipeline; the policy
    residual lives in JointPosRefResidualAction.
    """

    entity_name: str = "robot"
    actuator_names: tuple[str, ...] | list[str] = ()
    robot_name: str = "humanoid_v21"
    steps_per_edge: int = 10
    resampling_time_range: tuple[float, float] = _NO_TIME_RESAMPLE
    # Per-episode mixed regime: at every env reset a Bernoulli(home_ratio) fraction of envs hold the
    # arm at HOME for the whole episode (frozen, no swing), the rest get normal random graph-nav swing.
    # Frozen envs keep a clean backward-locomotion gradient alive (full arm swing reaction otherwise
    # erases backward tracking); the policy observes joint_ref (= home for frozen envs) and can
    # condition its gait on the regime. 0 = disabled (all envs swing).
    home_ratio: float = 0.0
    # Optional curriculum on home_ratio: linearly anneal from home_ratio_start to home_ratio_end over
    # [0, home_ratio_curriculum_steps] env-control-steps, then hold home_ratio_end.
    # curriculum_steps == 0 => no curriculum, the static home_ratio above is used.
    home_ratio_start: float = 0.0
    home_ratio_end: float = 0.0
    home_ratio_curriculum_steps: int = 0
    # Fraction of FROZEN envs that hold a random graph node instead of home (0 = legacy: frozen
    # always means home). At 0 the arm reference is static iff it equals home, so the policy learns
    # "target != home" => "arm is slewing" and pre-yaws the waist. Deploy holds a far IK target
    # STATIC, so the brace never stops firing: injecting a reach target while the arm physically
    # stays home yaws the waist 5.19 deg vs 0.61 at rest (mj_envs/probe/waist_cause_probe.py) --
    # more than a real reach, i.e. the observation drives it, not arm mass. > 0 makes "far target,
    # holding still" in-distribution.
    frozen_at_graph_node: float = 0.0
    # Env-control-steps before frozen_at_graph_node switches on (0 = on from step 0, legacy).
    # A held far pose deepens the low-command standing optimum (memory/grid_low_command_deadzone.md):
    # a policy that cannot yet walk stands, trips velocity_tracking_failure, and the grid curriculum
    # never cascades (cells frozen at 64/148 vs bursting 64 -> 402). Delay past locomotion
    # acquisition and both survive. Callers write ``<iters> * self.num_steps_per_env`` (the
    # weight_stages idiom) so this stays a CONSTANT iteration count -- acquiring locomotion costs a
    # fixed ~4000 iterations, and a horizon fraction would push the switch out on a longer run.
    frozen_at_graph_node_warmup_steps: int = 0

    def build(self, env: ManagerBasedRlEnv) -> "ArmGraphRefCommandTerm":
        return ArmGraphRefCommandTerm(self, env)


class ArmGraphRefCommandTerm(CommandTerm):
    """Joint reference via collision-free graph traversal.

    Lifecycle (was previously baked into the arm action term):
      __init__              -> graph load, warp views, nav-state tensors (NO sim write)
      _resample_command     <- old _reset_envs(q_default=True): sample start/target nodes,
                               set nav state, write start pose to sim (overrides engine default)
      _update_command       <- old process_actions warp launch (per-step graph tick), dt-guarded
      command property      -> joint_ref (was current_target)
    """

    cfg: ArmGraphRefCommandTermCfg

    def __init__(self, cfg: ArmGraphRefCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._dt: float | torch.Tensor | None = None
        # True only on the explicit reset-compute pass (scalar dt=0.0). The
        # auto_reset step path passes a per-env tensor dt, which must still tick.
        self._reset_pass: bool = False

        # Resolve target joints (mirror BaseAction._find_targets, JOINT transmission).
        self._entity = env.scene[cfg.entity_name]
        target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        self._target_names = target_names

        # Load precomputed graph artifact (path identical to old action: parents[1] = mj_envs).
        artifact_path = (
            Path(__file__).resolve().parents[1]
            / "asset_zoo" / "cache"
            / f"arm_collision_graph_{cfg.robot_name}.pt"
        )
        # Reconstitute from a shipped slim artifact (edges only) before falling back to a full
        # rebuild: the BFS closure costs ~30 s, whereas regenerating the graph re-runs the MuJoCo
        # collision check over every kNN edge. No-op when neither the slim nor full file exists.
        from utils.slim_cache import expand as _expand_slim_cache
        _expand_slim_cache(artifact_path)
        if not artifact_path.exists():
            import os
            import time
            lock_path = Path(str(artifact_path) + ".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    print(f"Collision graph not found: {artifact_path}. Generating on the fly...")
                    import sys
                    repo_root = str(Path(__file__).resolve().parents[2])
                    if repo_root not in sys.path:
                        sys.path.append(repo_root)
                    from mj_envs.asset_zoo.scripts.build_arm_collision_graph import main as build_graph, BuildArgs
                    device_str = str(env.device) if hasattr(env, "device") else "cuda:0"
                    args = BuildArgs(robot=cfg.robot_name, device=device_str)
                    build_graph(args)
                finally:
                    os.close(fd)
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
            except FileExistsError:
                print("Another process is generating the collision graph. Waiting...")
                while lock_path.exists():
                    time.sleep(0.1)

        artifact = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        # Artifact schema (built by asset_zoo/scripts/build_arm_collision_graph.py):
        #   poses     (N, 15) f32  - sampled collision-free arm configs (graph nodes).
        #   edges     list         - collision-free transitions between near poses (graph adjacency).
        #   next_hop  (N, N) u16    - ALL-PAIRS shortest-path routing table. next_hop[i, j] = node to
        #                            step to FROM i when heading TO j. Runtime nav follows it greedily
        #                            (i -> next_hop[i,j] -> ... -> j), O(1)/step, no live pathfinding.
        #                            Quadratic in N: 9990^2 * 2B = ~200 MB, dominates the artifact size.
        #                            Space-for-time vs running Dijkstra on `edges` per step.
        #   joint_names, n_poses, k, method, robot - metadata.

        # Align graph joints to reference joints by name.
        graph_joint_names: list[str] = artifact["joint_names"]
        try:
            joint_col = [graph_joint_names.index(n) for n in self._target_names]
        except ValueError as e:
            raise ValueError(
                f"Joint mismatch between collision graph and command targets. {e}\n"
                f"Command targets: {self._target_names}\n"
                f"Graph joints:    {graph_joint_names}"
            ) from e

        graph_poses = artifact["poses"][:, joint_col].contiguous()   # (N, D) float32
        N, D = graph_poses.shape
        self.N = N
        self.steps_per_edge = cfg.steps_per_edge
        self._inv_steps_per_edge = 1.0 / float(cfg.steps_per_edge)

        self.graph_poses = graph_poses.to(self.device)
        self.next_hop = artifact["next_hop"].to(dtype=torch.int32, device=self.device)

        # Per-env navigation state.
        self.current_node    = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.target_node     = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.next_hop_node   = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.steps_remaining = torch.full(
            (self.num_envs,), self.steps_per_edge, dtype=torch.int32, device=self.device
        )
        # joint_ref: the emitted command (was current_target).
        self._joint_ref         = torch.zeros(self.num_envs, D, device=self.device)
        self.interpolation_rate = torch.zeros(self.num_envs, D, device=self.device)

        # Warp stream + zero-copy array views.
        self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))
        self._wp_device = wp.get_device(str(self.device))
        self._cur_node_wp       = wp.from_torch(self.current_node)
        self._tgt_node_wp       = wp.from_torch(self.target_node)
        self._next_hop_node_wp  = wp.from_torch(self.next_hop_node)
        self._steps_rem_wp      = wp.from_torch(self.steps_remaining)
        self._cur_target_wp     = wp.from_torch(self._joint_ref)
        self._interp_rate_wp    = wp.from_torch(self.interpolation_rate)
        self._graph_poses_wp    = wp.from_torch(self.graph_poses)
        self._next_hop_table_wp = wp.from_torch(self.next_hop)

        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self._zero_arm_vel = torch.zeros(self.num_envs, D, device=self.device)
        self._frame_count = 0

        # Per-episode mixed-regime state (see cfg.home_ratio). _home_pose is the constant HOME arm
        # config (default_joint_pos never changes); _frozen_mask marks envs holding home this episode.
        self._home_ratio = cfg.home_ratio
        self._home_ratio_start = cfg.home_ratio_start
        self._home_ratio_end = cfg.home_ratio_end
        self._home_ratio_curriculum_steps = cfg.home_ratio_curriculum_steps
        self._home_pose = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._frozen_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Pose each frozen env actually holds this episode: home, or a random graph node when
        # cfg.frozen_at_graph_node fires. Kept as its own buffer rather than mutating _home_pose,
        # because _home_pose is the constant every non-frozen path (and the probes) reads.
        self._frozen_at_graph_node = cfg.frozen_at_graph_node
        self._frozen_at_graph_node_warmup_steps = cfg.frozen_at_graph_node_warmup_steps
        self._frozen_pose = self._home_pose.clone()

    def _current_graph_node_mix(self) -> float:
        """Held-at-node fraction, gated off until the warmup window has elapsed.

        A STEP, not a ramp: the point is to keep the harder regime entirely absent while locomotion
        is acquired, and a ramp would reintroduce it during exactly that window. The probes read
        ``_frozen_at_graph_node`` directly and zero it, so they are unaffected either way.

        Reads ``common_step_counter``, which eval/play RESTORES from the checkpoint, so a loaded
        policy is past any warmup and runs the mix it was trained to end with.
        """
        if self._env.common_step_counter < self._frozen_at_graph_node_warmup_steps:
            return 0.0
        return self._frozen_at_graph_node

    def _current_home_ratio(self) -> float:
        """Return static or common-step-counter-annealed frozen-arm fraction."""
        if self._home_ratio_curriculum_steps <= 0:
            return self._home_ratio
        frac = min(self._env.common_step_counter / self._home_ratio_curriculum_steps, 1.0)
        return self._home_ratio_start + (self._home_ratio_end - self._home_ratio_start) * frac

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        return self._joint_ref

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        # Capture dt so _update_command can skip the reset (dt==0) compute pass.
        self._dt = dt
        self._reset_pass = not isinstance(dt, torch.Tensor) and dt == 0.0
        super().compute(dt, env_ids)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Seed nav state for env_ids and write the start pose to sim.

        Samples random start/target nodes; writes start pose + zero vel to sim so the
        actual arm joints match joint_ref from frame 1 (overrides the engine's default
        reset). Relocated from the old action.reset; still lands after scene.reset and
        before sim.forward, so the override propagates.
        """
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
        n = ids.numel()

        # Split resetting envs into frozen (hold home) and swing (random graph nav). With
        # home_ratio == 0 every env follows the legacy swing path.
        hr = self._current_home_ratio()
        if hr > 0.0:
            frozen = torch.rand(n, device=self.device) < hr
        else:
            frozen = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._frozen_mask[ids] = frozen

        frozen_ids = ids[frozen]
        if frozen_ids.numel() > 0:
            # Held pose is home by default; a frozen_at_graph_node fraction instead holds a random
            # graph node, so "arm reference is static" stops implying "arm reference is home".
            # Legacy path must stay byte-exact: held == _home_pose, which the probes overwrite and
            # rely on being reasserted. Only the sampled subset departs from it.
            held = self._home_pose[frozen_ids].clone()
            node = torch.zeros(frozen_ids.numel(), device=self.device, dtype=self.current_node.dtype)
            node_mix = self._current_graph_node_mix()
            if node_mix > 0.0:
                at_node = torch.rand(frozen_ids.numel(), device=self.device) < node_mix
                k = int(at_node.sum())
                if k > 0:
                    picked = torch.randint(0, self.N, (k,), device=self.device)
                    node[at_node] = picked.to(self.current_node.dtype)
                    held[at_node] = self.graph_poses[picked.long()]
            self._frozen_pose[frozen_ids]       = held
            self.current_node[frozen_ids]       = node
            self.target_node[frozen_ids]        = node
            self.next_hop_node[frozen_ids]      = node
            self._joint_ref[frozen_ids]         = held
            self.interpolation_rate[frozen_ids] = 0.0
            self.steps_remaining[frozen_ids]    = self.steps_per_edge
            self._entity.write_joint_position_to_sim(held, joint_ids=self._target_ids, env_ids=frozen_ids)
            self._entity.write_joint_velocity_to_sim(
                self._zero_arm_vel[:frozen_ids.numel()], joint_ids=self._target_ids, env_ids=frozen_ids
            )

        swing_ids = ids[~frozen]
        n_swing = swing_ids.numel()
        if n_swing == 0:
            return
        start_nodes = torch.randint(0, self.N, (n_swing,), device=self.device, dtype=torch.int32)
        tgt_nodes   = torch.randint(0, self.N, (n_swing,), device=self.device, dtype=torch.int32)
        start_l     = start_nodes.long()
        nh_nodes    = self.next_hop[start_l, tgt_nodes.long()].int()
        start_poses = self.graph_poses[start_l]
        nh_poses    = self.graph_poses[nh_nodes.long()]

        self.current_node[swing_ids]       = start_nodes
        self.target_node[swing_ids]        = tgt_nodes
        self.next_hop_node[swing_ids]      = nh_nodes
        self._joint_ref[swing_ids]         = start_poses
        self.interpolation_rate[swing_ids] = (nh_poses - start_poses) * self._inv_steps_per_edge
        self.steps_remaining[swing_ids]    = self.steps_per_edge

        self._entity.write_joint_position_to_sim(start_poses, joint_ids=self._target_ids, env_ids=swing_ids)
        self._entity.write_joint_velocity_to_sim(
            self._zero_arm_vel[:n_swing], joint_ids=self._target_ids, env_ids=swing_ids
        )

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if self._reset_pass:   # reset compute pass: no tick (avoid phantom advance)
            return
        self._frame_count += 1
        wp.launch(
            process_graph_nav_kernel,
            dim=self.num_envs,
            inputs=[
                self._cur_node_wp, self._tgt_node_wp, self._next_hop_node_wp,
                self._steps_rem_wp, self._cur_target_wp, self._interp_rate_wp,
                self._graph_poses_wp, self._next_hop_table_wp,
                self.N, self.steps_per_edge, self._frame_count,
            ],
            device=self._wp_device,
            stream=self._wp_stream,
        )
        # Kernel ticks every env. Reassert the held pose after launch so frozen envs never swing;
        # the shared CUDA stream preserves launch-then-clamp order. _frozen_pose equals _home_pose
        # unless cfg.frozen_at_graph_node put this env on a graph node for the episode.
        if self._frozen_mask.any():
            self._joint_ref[self._frozen_mask] = self._frozen_pose[self._frozen_mask]
            self.interpolation_rate[self._frozen_mask] = 0.0


@dataclass(kw_only=True)
class CommandGatedArmGraphRefCommandTermCfg(ArmGraphRefCommandTermCfg):
    """ArmGraphRefCommandTermCfg + locomotion gating (reads the twist command).

    Two speed regimes, both gated on the steps_remaining == steps_per_edge node-advance edge:
      * Locomotion (|twist_xy| > locomotion_vel_threshold): loco_min_steps dwell per edge.
      * Stationary (|twist_xy| <= stationary_vel_threshold, optional): stand_min_steps dwell
        per edge. When stand_min_steps <= 0 (default), stationary regime uses full-speed
        traversal (steps_per_edge). Set stand_min_steps > 0 to slow the arm at zero base cmd.

    Training-time vs deploy note (Wave-19, 2026-07-25): stand_min_steps is a TRAINING-TIME
    lever, NOT a deploy-side slowdown. The v2ybsk keeper is trained with stand_min_steps=80
    and reverted to the default 0 at deploy. Same-policy probe (sway_probe with
    --stand-min-steps 0/80/120 override) confirmed the v2ybsk ckpt is invariant to deploy
    sms in [0, 120] (drift_peak 0.063-0.068 m, fell 0); the slow-arm training taught
    arm-motion-agnostic anticipation that generalizes. By contrast a fast-arm-trained
    ckpt (v2ybdkc) probed at runtime sms>0 is OOD and degrades (sms=120 makes it fall).
    Lever added for training-only use; deploy must reset to default 0 unless the user
    explicitly wants a deploy-time slow arm.
    """

    locomotion_vel_threshold: float = 0.3
    loco_min_steps: int = 80
    stationary_vel_threshold: float = 0.05
    stand_min_steps: int = 0  # 0 = no stationary slowdown (legacy behavior)
    command_name: str = "twist"

    def build(self, env: ManagerBasedRlEnv) -> "CommandGatedArmGraphRefCommandTerm":
        return CommandGatedArmGraphRefCommandTerm(self, env)


class CommandGatedArmGraphRefCommandTerm(ArmGraphRefCommandTerm):
    """ArmGraphRefCommandTerm with locomotion + stationary gating.

    During locomotion (|twist_xy| > locomotion_vel_threshold): graph traversal continues but
    each edge is re-extended to loco_min_steps (slower dwell). Stationary
    (|twist_xy| <= stationary_vel_threshold, when stand_min_steps > 0): each edge is
    re-extended to stand_min_steps (slower swing). Otherwise (mid-cmd, not loco and not stand):
    full-speed traversal (steps_per_edge). Gate triggers only on the steps_remaining ==
    steps_per_edge (node-advance) edge — the v29 bug-history (HumanoidRmaVelEstArmFlashSacv29
    docstring) showed gating mid-edge caused arm-target oscillation.

    Reads the twist command -> register `twist` BEFORE this term in cfg.commands so the
    command manager (dict-insertion compute order) has the current twist available.
    """

    cfg: CommandGatedArmGraphRefCommandTermCfg

    def __init__(self, cfg: CommandGatedArmGraphRefCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._loco_thresh_sq: float = cfg.locomotion_vel_threshold ** 2
        self._loco_min_steps: int = cfg.loco_min_steps
        self._inv_loco_min_steps: float = 1.0 / float(cfg.loco_min_steps)
        self._stand_thresh_sq: float = cfg.stationary_vel_threshold ** 2
        self._stand_min_steps: int = cfg.stand_min_steps
        self._inv_stand_min_steps: float = 1.0 / float(cfg.stand_min_steps) if cfg.stand_min_steps > 0 else 0.0
        self._twist_command_name: str = cfg.command_name

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if self._reset_pass:
            return
        super()._update_command(env_ids)  # graph tick

        twist = self._env.command_manager.get_command(self._twist_command_name)
        vx, vy = twist[:, 0], twist[:, 1]
        cmd_norm_sq = vx * vx + vy * vy
        locomoting = cmd_norm_sq > self._loco_thresh_sq
        locomoting &= ~self._frozen_mask

        if self._stand_min_steps > 0:
            # Locomotion branch: extend dwell to loco_min_steps.
            if locomoting.any():
                loco_ids = (locomoting & (self.steps_remaining == self.steps_per_edge)).nonzero(as_tuple=True)[0]
                if loco_ids.numel() > 0:
                    nh_poses = self.graph_poses[self.next_hop_node[loco_ids].long()]
                    self.interpolation_rate[loco_ids] = (nh_poses - self._joint_ref[loco_ids]) * self._inv_loco_min_steps
                    self.steps_remaining[loco_ids] = self._loco_min_steps
            # Stationary branch: extend dwell to stand_min_steps (only for envs in the
            # stationary band, not frozen, not locomoting).
            standing = (cmd_norm_sq <= self._stand_thresh_sq) & ~self._frozen_mask & ~locomoting
            if standing.any():
                stand_ids = (standing & (self.steps_remaining == self.steps_per_edge)).nonzero(as_tuple=True)[0]
                if stand_ids.numel() > 0:
                    nh_poses = self.graph_poses[self.next_hop_node[stand_ids].long()]
                    self.interpolation_rate[stand_ids] = (nh_poses - self._joint_ref[stand_ids]) * self._inv_stand_min_steps
                    self.steps_remaining[stand_ids] = self._stand_min_steps
        else:
            # Legacy behavior: locomotion-only gating, stationary uses full-speed traversal.
            if not locomoting.any():
                return
            ext_ids = (locomoting & (self.steps_remaining == self.steps_per_edge)).nonzero(as_tuple=True)[0]
            if ext_ids.numel() == 0:
                return
            nh_poses = self.graph_poses[self.next_hop_node[ext_ids].long()]
            self.interpolation_rate[ext_ids] = (nh_poses - self._joint_ref[ext_ids]) * self._inv_loco_min_steps
            self.steps_remaining[ext_ids] = self._loco_min_steps


# =============================================================================
# Camera gaze joint-reference command
# =============================================================================


@dataclass(kw_only=True)
class JointWalkRefCommandTermCfg(CommandTermCfg):
    """Generic single-or-multi-DOF uniform random-walk joint reference (no graph nav).

    Emits ``joint_ref`` (absolute joint positions) for any actuator set. Each episode
    the reference is seeded uniform in [-walk_range, walk_range]^D per joint, then
    takes bounded random-walk steps every [min_steps, max_steps] env-steps. Used for
    joints that need a moving reference but lack a collision graph (waist_joint).

    Lives next to GazeRefCommandTermCfg because the implementation is structurally
    the same minus the yaw/pitch naming split; gaze yaw/pitch remain the rich path.
    """

    entity_name: str = "robot"
    actuator_names: tuple[str, ...] | list[str] = ()
    min_steps: int = 30
    max_steps: int = 120
    walk_range: float = 1.0   # per-joint half-range; seeded uniform in [-walk_range, +walk_range]
    walk_step: float = 0.5    # per-step delta half-range (clamped by walk_range)
    resampling_time_range: tuple[float, float] = _NO_TIME_RESAMPLE

    def build(self, env: ManagerBasedRlEnv) -> "JointWalkRefCommandTerm":
        return JointWalkRefCommandTerm(self, env)


class JointWalkRefCommandTerm(CommandTerm):
    """Joint reference via uniform random-walk stepping (no warp; slow Python counter)."""

    cfg: JointWalkRefCommandTermCfg

    def __init__(self, cfg: JointWalkRefCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._dt: float | torch.Tensor | None = None
        # True only on the explicit reset-compute pass (scalar dt=0.0). The
        # auto_reset step path passes a per-env tensor dt, which must still tick.
        self._reset_pass: bool = False

        self._entity = env.scene[cfg.entity_name]
        target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        self._target_names = target_names
        D = len(target_ids)

        self._joint_ref = torch.zeros(self.num_envs, D, device=self.device)
        self._steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._joint_ref

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self._dt = dt
        self._reset_pass = not isinstance(dt, torch.Tensor) and dt == 0.0
        super().compute(dt, env_ids)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
        n = ids.numel()
        targets = torch.empty(n, self._joint_ref.shape[1], device=self.device).uniform_(
            -self.cfg.walk_range, self.cfg.walk_range
        )
        self._joint_ref[ids] = targets
        self._reset_counters(ids, n)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if self._reset_pass:
            return
        self._steps_remaining -= 1
        expired = (self._steps_remaining <= 0).nonzero(as_tuple=False).flatten()
        if expired.numel() > 0:
            self._walk(expired)

    def _walk(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.long()
        n = ids.numel()
        cur = self._joint_ref[ids]
        delta = torch.empty_like(cur).uniform_(-self.cfg.walk_step, self.cfg.walk_step)
        self._joint_ref[ids] = (cur + delta).clamp_(-self.cfg.walk_range, self.cfg.walk_range)
        self._reset_counters(ids, n)

    def _reset_counters(self, ids: torch.Tensor, n: int) -> None:
        self._steps_remaining[ids] = torch.randint(
            self.cfg.min_steps, self.cfg.max_steps + 1,
            (n,), device=self.device, dtype=torch.long,
        )


@dataclass(kw_only=True)
class GazeRefCommandTermCfg(CommandTermCfg):
    """Random-walk gaze joint-reference command for the head-camera gimbal.

    Emits ``joint_ref`` (absolute yaw/pitch joint positions). Each episode the reference
    is seeded absolute-uniform in the ROM box, then takes bounded random-walk steps every
    [min_steps, max_steps] env-steps. Holds the reference half of the gaze pipeline; the
    policy residual lives in JointPosRefResidualAction.
    """

    entity_name: str = "robot"
    actuator_names: tuple[str, ...] | list[str] = ()
    min_steps: int = 30
    max_steps: int = 120
    yaw_range: float = 1.0
    pitch_range: float = 0.6
    yaw_walk: float = 1.0
    pitch_walk: float = 0.4
    resampling_time_range: tuple[float, float] = _NO_TIME_RESAMPLE

    def build(self, env: ManagerBasedRlEnv) -> "GazeRefCommandTerm":
        return GazeRefCommandTerm(self, env)


class GazeRefCommandTerm(CommandTerm):
    """Joint reference via random-walk gaze stepping (no warp; slow Python counter).

    Lifecycle (was previously baked into the gaze action term):
      _resample_command  <- seed: absolute-uniform yaw/pitch + counter reset
      _update_command    <- decrement counters; random-walk expired envs (dt-guarded)
      command property   -> joint_ref
    """

    cfg: GazeRefCommandTermCfg

    def __init__(self, cfg: GazeRefCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._dt: float | torch.Tensor | None = None
        # True only on the explicit reset-compute pass (scalar dt=0.0). The
        # auto_reset step path passes a per-env tensor dt, which must still tick.
        self._reset_pass: bool = False

        self._entity = env.scene[cfg.entity_name]
        target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        self._target_names = target_names
        D = len(target_ids)

        self._yaw_cols: list[int] = [i for i, n in enumerate(target_names) if "yaw" in n]
        self._pitch_cols: list[int] = [i for i, n in enumerate(target_names) if "pitch" in n]

        self._joint_ref = torch.zeros(self.num_envs, D, device=self.device)
        self._steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._joint_ref

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        self._dt = dt
        self._reset_pass = not isinstance(dt, torch.Tensor) and dt == 0.0
        super().compute(dt, env_ids)

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
        n = ids.numel()
        targets = torch.empty(n, self._joint_ref.shape[1], device=self.device)
        if self._yaw_cols:
            targets[:, self._yaw_cols] = torch.empty(
                n, len(self._yaw_cols), device=self.device
            ).uniform_(-self.cfg.yaw_range, self.cfg.yaw_range)
        if self._pitch_cols:
            targets[:, self._pitch_cols] = torch.empty(
                n, len(self._pitch_cols), device=self.device
            ).uniform_(-self.cfg.pitch_range, self.cfg.pitch_range)
        self._joint_ref[ids] = targets
        self._reset_counters(ids, n)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if self._reset_pass:
            return
        self._steps_remaining -= 1
        expired = (self._steps_remaining <= 0).nonzero(as_tuple=False).flatten()
        if expired.numel() > 0:
            self._walk(expired)

    def _walk(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.long()
        n = ids.numel()
        cur = self._joint_ref[ids]
        if self._yaw_cols:
            delta = torch.empty(
                n, len(self._yaw_cols), device=self.device
            ).uniform_(-self.cfg.yaw_walk, self.cfg.yaw_walk)
            cur[:, self._yaw_cols] = (cur[:, self._yaw_cols] + delta).clamp_(
                -self.cfg.yaw_range, self.cfg.yaw_range)
        if self._pitch_cols:
            delta = torch.empty(
                n, len(self._pitch_cols), device=self.device
            ).uniform_(-self.cfg.pitch_walk, self.cfg.pitch_walk)
            cur[:, self._pitch_cols] = (cur[:, self._pitch_cols] + delta).clamp_(
                -self.cfg.pitch_range, self.cfg.pitch_range)
        self._joint_ref[ids] = cur
        self._reset_counters(ids, n)

    def _reset_counters(self, ids: torch.Tensor, n: int) -> None:
        self._steps_remaining[ids] = torch.randint(
            self.cfg.min_steps, self.cfg.max_steps + 1,
            (n,), device=self.device, dtype=torch.long,
        )


@dataclass(kw_only=True)
class PassiveGazeCommandTermCfg(CommandTermCfg):
    """Externally-set gaze joint reference (active-vision A1+, plan/valiant-giggling-hoare.md).

    Drop-in replacement for GazeRefCommandTermCfg on the ``camera_ref`` command when a
    HIGH-LEVEL learner (not a random walk) sources the gaze setpoint. The term is a passive
    HOLDER: it emits whatever was last written via ``set_command`` and never self-generates.
    Same public interface (``command``, ``target_names``) so v83's ``target_camera_joint_pos``
    obs + ``camera_joint_tracking`` reward still bind unchanged.
    """

    entity_name: str = "robot"
    actuator_names: tuple[str, ...] | list[str] = ()
    resampling_time_range: tuple[float, float] = _NO_TIME_RESAMPLE

    def build(self, env: ManagerBasedRlEnv) -> "PassiveGazeCommandTerm":
        return PassiveGazeCommandTerm(self, env)


class PassiveGazeCommandTerm(CommandTerm):
    """Holds an externally-written gaze joint reference; no self-generation.

    Lifecycle:
      _resample_command  <- reset envs to neutral (zeros) so a freshly reset env looks
                            forward until the learner writes the next setpoint.
      _update_command    <- no-op: the last ``set_command`` value persists across the step
                            (mjlab order: learner writes BEFORE env.step's action processing,
                            then command_manager.compute ticks this no-op, holding the value).
      set_command        <- the learner's per-step write (clamped to ROM by the caller).
      command property   -> joint_ref (absolute joint reference; the residual action adds on top).
    """

    cfg: PassiveGazeCommandTermCfg

    def __init__(self, cfg: PassiveGazeCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._entity = env.scene[cfg.entity_name]
        target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        self._target_names = target_names
        self._joint_ref = self._entity.data.default_joint_pos[:, self._target_ids].clone()

    @property
    def command(self) -> torch.Tensor:
        return self._joint_ref

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    @property
    def target_ids(self) -> torch.Tensor:
        """Entity joint ids of the held reference's columns (consumers index joint state)."""
        return self._target_ids

    def set_command(self, value: torch.Tensor) -> None:
        """Write the gaze setpoint for ALL envs (clamping to ROM is the caller's job)."""
        self._joint_ref.copy_(value)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """No-op: passive holder never self-generates or time-resamples.

        Skips base CommandTerm.compute's time_left decrement + .nonzero() (a GPU sync
        point) every step -- pointless here since resampling_time_range is the sentinel
        _NO_TIME_RESAMPLE and the only resample path is the explicit env-reset call.
        """

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
        self._joint_ref[ids] = self._entity.data.default_joint_pos[ids][:, self._target_ids]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass


# =============================================================================
# Arm joint-reference holder (externally set, e.g. by scripted IK)
# =============================================================================


@dataclass(kw_only=True)
class PassiveArmRefCommandTermCfg(CommandTermCfg):
    """Externally-set arm joint reference (pick-place reach, plan/ACTIVE_VISION_DESIGN_VALIDATION_PLAN.md).

    Drop-in replacement for (Command)ArmGraphRefCommandTermCfg on the ``arm_ref`` command when a
    scripted controller (or a future learner) sources the arm setpoint instead of the collision
    graph. The term is a passive HOLDER: it emits whatever was last written via ``set_command`` and
    never self-generates. Same public interface (``command``, ``target_names``) so v83's
    ``target_arm_joint_pos`` obs + arm tracking reward bind unchanged and the residual action still
    adds the frozen policy's residual on top.
    """

    entity_name: str = "robot"
    actuator_names: tuple[str, ...] | list[str] = ()
    resampling_time_range: tuple[float, float] = _NO_TIME_RESAMPLE

    def build(self, env: ManagerBasedRlEnv) -> "PassiveArmRefCommandTerm":
        return PassiveArmRefCommandTerm(self, env)


class PassiveArmRefCommandTerm(CommandTerm):
    """Holds an externally-written arm joint reference; no self-generation.

    Neutral (reset) value is the robot's DEFAULT arm pose (home), NOT zeros: a freshly reset arm
    then looks like the standing home pose until the controller writes the first IK setpoint —
    unlike the gaze holder whose zero neutral is a valid look-forward. Mirrors PassiveGazeCommandTerm
    otherwise (set_command write, no-op tick).

    Lifecycle:
      _resample_command  <- reset envs to the default arm pose (home).
      _update_command    <- no-op: the last set_command value persists across the step.
      set_command        <- the controller's per-step write (absolute arm joint positions).
      command property   -> joint_ref (the residual action adds the policy residual on top).
    """

    cfg: PassiveArmRefCommandTermCfg

    def __init__(self, cfg: PassiveArmRefCommandTermCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._entity = env.scene[cfg.entity_name]
        target_ids, target_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
        self._target_ids = torch.tensor(target_ids, device=self.device, dtype=torch.long)
        self._target_names = target_names
        self._joint_ref = torch.zeros(self.num_envs, len(target_ids), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._joint_ref

    @property
    def target_names(self) -> list[str]:
        return self._target_names

    @property
    def target_ids(self) -> torch.Tensor:
        """Column indices of the reference joints into the entity's joint arrays (default_joint_pos)."""
        return self._target_ids

    def set_command(self, value: torch.Tensor) -> None:
        """Write the arm joint reference for ALL envs (absolute positions; caller's job to clamp)."""
        self._joint_ref.copy_(value)

    def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """No-op: passive holder never self-generates or time-resamples.

        Skips base CommandTerm.compute's time_left decrement + .nonzero() (a GPU sync
        point) every step -- pointless here since resampling_time_range is the sentinel
        _NO_TIME_RESAMPLE and the only resample path is the explicit env-reset call.
        """

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
        self._joint_ref[ids] = self._entity.data.default_joint_pos[ids][:, self._target_ids]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        pass


# =============================================================================
# Thin residual action: joint_target = joint_ref + scale * action
# =============================================================================


@dataclass(kw_only=True)
class JointPosRefResidualActionCfg(JointPositionActionCfg):
    """Residual joint-position action on top of a joint-reference command.

    processed = command_manager.get_term(command_name).command + scale * action

    Replaces the bespoke arm/camera action subclasses; the reference generator now lives
    in a CommandTerm (e.g. ArmGraphRefCommandTerm, GazeRefCommandTerm). The inherited
    `offset` is unused (the command IS the absolute joint reference).
    """

    command_name: str = ""

    def build(self, env: ManagerBasedRlEnv) -> "JointPosRefResidualAction":
        return JointPosRefResidualAction(self, env)


class JointPosRefResidualAction(JointPositionAction):
    """Adds the policy residual to a joint-reference command.

    Contract: the command IS the absolute joint reference; processed = command +
    scale*action. The inherited ``offset`` is UNUSED -- ``process_actions`` overwrites
    ``_processed_actions`` and inherited ``apply_actions`` writes that straight to sim, so
    ``use_default_offset`` has no effect here (set it False to skip the dead offset alloc).
    Keeps ``raw_action`` (read by the arm_action_l2 reward).
    """

    cfg: JointPosRefResidualActionCfg

    def __init__(self, cfg: JointPosRefResidualActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._cmd = env.command_manager.get_term(cfg.command_name)
        # Column-order invariant: the command's joint order must match the action's targets.
        assert tuple(self._target_names) == tuple(self._cmd.target_names), (
            f"Joint order mismatch between action '{cfg.command_name}' targets and command.\n"
            f"Action: {self._target_names}\nCommand: {self._cmd.target_names}"
        )
        self._scale_is_scalar = isinstance(self._scale, float)
        self._processed_actions = torch.zeros(self.num_envs, self._num_targets, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        ref = self._cmd.command
        self._raw_actions.copy_(actions)
        if self._scale_is_scalar:
            torch.add(ref, self._raw_actions, alpha=self._scale, out=self._processed_actions)
        else:
            torch.addcmul(ref, self._raw_actions, self._scale, out=self._processed_actions)
