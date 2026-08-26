"""Heuristic coupled arm+camera step for the scripted pick-place reach.

The arm-target decision and the camera-aim decision share the SAME target +
side + step. Calling HeuristicReachPolicy.compute() returns BOTH commands in
one shot so the env stops orchestrating two separate computations.

Today: Mink QP IK for the arm, closed-form parallax-corrected analytic solve for the
camera (see _compute_camera — recomputed fresh each step from the real camera pose, not
integrated/nudged, so it can't gimbal-lock or drift). Tomorrow: a learned arm policy (B1)
and/or learned camera policy subclass HeuristicReachPolicy and override .compute() — the
env holds _policy and calls .compute() unchanged either way.
"""

import mujoco
import numpy as np
import torch

from asset_zoo.humanoid_v21.humanoid_v21_constants import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from tasks.camera_perception import fov_detect
from tasks.camera_terms import FAR, H_HALF, NEAR, V_HALF
from tasks.visual_manipulation.control_vec import gaze_aim
from tasks.visual_manipulation.dynamics_executor import DynamicArmReferenceExecutor
from utils.ik_mink import BatchedMinkIK

_GAZE_YAW_ROM, _GAZE_PITCH_ROM = 4.7124, 1.5708     # yaw +-270 deg, pitch +-90 deg


def _camera_column(camera_joint_names: list[str], axis: str, side: str) -> int:
    """Index of the camera_ref column for one (axis, side)."""
    for i, name in enumerate(camera_joint_names):
        if axis in name and side in name:
            return i
    raise ValueError(f"no camera joint matches axis={axis!r} side={side!r} in {camera_joint_names}")


class HeuristicReachPolicy:
    """Per-step arm + camera commands for the scripted pick-place reach.

    Single coupled decision. Today: IK for arm (Mink QP), closed-form parallax-
    corrected azimuth/elevation for camera. Both leaves are private; future
    learned replacements subclass and override .compute().

    All bookkeeping is in __init__; compute() is the hot path.
    """

    def __init__(self, *, ik: BatchedMinkIK, ik_num_iters: int,
                 reaching_side: str, target_x_forward: float,
                 arm_ref, camera_ref,
                 device: torch.device | str,
                 cruise_z_base: float = 0.10, xy_near: float = 0.05,
                 xy_far: float = 0.15, z_tol: float = 0.03):
        # --- arm bookkeeping ---
        self._ik = ik                                 # Mink QP solver (both arms)
        self._ik_num_iters = ik_num_iters             # 8 saturates (see internals.md)
        self._arm_ref = arm_ref                       # writer target
        arm_ref_joint_names = list(arm_ref.target_names)
        # IK output is [L7, R7] in the suffix order; arm_ref is whatever order
        # the command term configured. Build column-index tensors once.
        left_names  = list(LEFT_ARM_JOINT_NAMES)
        right_names = list(RIGHT_ARM_JOINT_NAMES)
        ik_out_names = left_names + right_names       # solver's column order
        reach_names = right_names if reaching_side == "right" else left_names
        self._reaching_side = reaching_side
        self._reaching_ik_cols = torch.tensor(
            [ik_out_names.index(n) for n in reach_names], device=device, dtype=torch.long)
        self._reaching_armref_cols = torch.tensor(
            [arm_ref_joint_names.index(n) for n in reach_names], device=device, dtype=torch.long)
        self._grasp_quat_base = torch.tensor(         # identity in base frame (IK param, policy holds it)
            [1.0, 0.0, 0.0, 0.0], device=device)

        # ik.solve's pos_targets / actual gripper-site readout slot order is [left, right] (index 0/1) --
        # see BatchedMinkIK._last_ee_pos_ik_base.
        self._reach_ee_idx = 0 if reaching_side == "left" else 1
        # --- lift/transit/descend waypoint. Root cause: the grasp target sits only ~2.5cm above
        # the table (cube height); rate-limited IK converges the SHORT z-delta long before the
        # LONG xy-delta (e.g. rear reach crosses the full table gap), so for most of the rollout
        # the gripper site is already at near-table height while still sweeping sideways across both
        # tabletops -- that's the collision. cruise_z_base=0.10 matches the existing ready-arm
        # hover height (world ~0.69m, 8cm clear of the 0.61m tabletops) -- same height the arm
        # already starts at, so holding it costs no extra vertical travel, only defers the final
        # descend until xy is aligned above the target. Stateless -- recomputed every step from
        # the IK's actual gripper-site feedback, no phase counter/reset hook.
        self._cruise_z = float(cruise_z_base)
        self._xy_near = float(xy_near)
        self._xy_far = float(xy_far)
        self._z_tol = float(z_tol)

        # --- camera bookkeeping (aim side picked once from target sign) ---
        self._camera_ref = camera_ref
        camera_joint_names = list(camera_ref.target_names)
        # Public: the env resolves which camera SITE (cam_left_rgb/cam_right_rgb) to read the
        # real pose from each step — must match this side (opposed mount: left=front hemi).
        self.aim_side = "left" if target_x_forward >= 0.0 else "right"
        self._aim_cam_yaw_col   = _camera_column(camera_joint_names, "yaw",   self.aim_side)
        self._aim_cam_pitch_col = _camera_column(camera_joint_names, "pitch", self.aim_side)
        # Integral trim on the gaze reference (see _compute_camera): cancels the frozen
        # policy's steady residual offset on the gimbal joints. Diagnosed on v83
        # (front+right): joints settle yaw -8.1 deg / pitch +3.9 deg SHORT of the reference
        # (the residual action holds a constant offset on top of it), which fully explains
        # the measured 5.2 deg aim error (prediction/measurement ratio 1.04). The trim
        # integrates the shortfall so (reference + residual) lands ON the ideal angles:
        # 5.2 deg -> 0.2 deg settled, in_fov 100%. Gain per step; bias clamped for
        # anti-windup (transients integrate briefly; the clamp bounds the excursion).
        n_envs = camera_ref.command.shape[0]
        self._trim_gain = 0.3
        self._trim_max = 0.35                                  # rad, |bias| cap (~20 deg)
        self._trim = torch.zeros(n_envs, 2, device=device)     # [:, 0]=yaw, [:, 1]=pitch
        # Diagnostics from the last _compute_camera call (env/eval driver reads these for
        # verification — not used by compute() itself).
        self.last_aim_error: torch.Tensor | None = None
        self.last_in_fov: torch.Tensor | None = None
        self.last_reach_target_base: torch.Tensor | None = None
        self._obj_quat_base: torch.Tensor | None = None   # stashed for planner-policy compatibility

    def reset(self) -> None:
        """Clear per-episode state (integral trim, diagnostics). Callers must invoke this
        alongside env.reset() -- the env's reset never touches the policy object, so without
        this the trim/diagnostics from the previous episode silently carry over (see
        DualCubeSearchGaze.reset for the search-state analog)."""
        self._trim.zero_()
        self.last_aim_error = None
        self.last_in_fov = None
        self.last_reach_target_base = None

    def compute(self, *, physics_qpos: torch.Tensor,
                target_in_base: torch.Tensor,
                arm_home_pose: torch.Tensor,
                cam_pos_base: torch.Tensor,
                cam_mat_base: torch.Tensor,
                reach_ee_pos_base: torch.Tensor,
                cam_joint_pos: torch.Tensor | None = None,
                obj_quat_base: torch.Tensor | None = None,
                planning_arm_pose: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (arm_command, camera_command) for the existing set_command interface.

        `arm_home_pose` is owned by the env (snapshot of entity default joint pos);
        passed in so policy stays stateless re: env-derived snapshots. `cam_pos_base`/
        `cam_mat_base` are the AIM-SIDE camera site's live pose, base frame (env computes
        this from `site_pose_w` — see pickplace_reach_env.step()). `reach_ee_pos_base` is
        the REACHING arm's REAL (physics) gripper-site pose, base frame, same source — used as
        the waypoint's actual-progress feedback (see _waypoint_target's docstring for why
        this must be the real pose, not the IK's own internal virtual forward-kinematics pose).
        `cam_joint_pos` (N, n_cam_joints; camera_ref column order) is the gimbal joints'
        REAL qpos — feeds the integral trim that cancels the frozen policy's steady
        residual offset (None = trim off, raw closed-form command).
        `obj_quat_base` (N, 4 wxyz) is the reach object's orientation in base frame; only the
        Planner policies may consume it for grasp-candidate goalsets. The mink paths
        ignore it (their grasp orientation is a fixed identity quat). Stashed for _compute_arm.
        """
        self._obj_quat_base = obj_quat_base
        arm_command = self._compute_arm(
            physics_qpos, target_in_base,
            arm_home_pose if planning_arm_pose is None else planning_arm_pose,
            reach_ee_pos_base)
        camera_command = self._compute_camera(target_in_base, cam_pos_base, cam_mat_base, cam_joint_pos)
        return arm_command, camera_command

    def _compute_arm(self, physics_qpos: torch.Tensor,
                     target_in_base: torch.Tensor,
                     arm_home_pose: torch.Tensor,
                     reach_ee_pos_base: torch.Tensor) -> torch.Tensor:
        N = target_in_base.shape[0]
        reach_target = self._waypoint_target(target_in_base, reach_ee_pos_base)  # (N,3), collision-avoiding
        self.last_reach_target_base = reach_target.detach()  # (N,3), base frame -- diagnostics only
        pos_targets = target_in_base.unsqueeze(1).expand(N, 2, 3).clone()
        pos_targets[:, self._reach_ee_idx] = reach_target
        quat_targets = self._grasp_quat_base.view(1, 1, 4).expand(N, 2, 4).contiguous()
        sol = self._ik.solve(physics_qpos, pos_targets, quat_targets,
                             num_iters=self._ik_num_iters)             # (N, 14)
        arm_command = arm_home_pose.clone()                           # (N, 15) waist + idle arm baseline
        arm_command[:, self._reaching_armref_cols] = sol[:, self._reaching_ik_cols]
        return arm_command

    def _waypoint_target(self, target_in_base: torch.Tensor,
                         ee_now: torch.Tensor) -> torch.Tensor:
        """Lift -> transit -> descend, computed fresh each step from REAL gripper-site feedback.

        `ee_now` must be the REAL (physics site) gripper pose, base frame -- NOT the IK's own internal
        virtual forward-kinematics pose (`ik._last_ee_pos_ik_base`). The IK's collision model is self-collision only (no
        table), so if the real arm is physically wedged against the table, the virtual solve still
        converges fine in its own table-free universe and reports fake progress while the real gripper site
        doesn't move at all -- confirmed by instrumenting a rear-reach rollout where the virtual forward kinematics
        converged to ~0mm error while the real site pose stayed put ~0.4m from the target the whole
        rollout (arm pinned against the near table's edge).

        Z descends as a CONTINUOUS function of remaining xy distance (linear ramp between `xy_far`
        and `xy_near`), not a hard xy_tol gate. A binary gate deadlocks: holding Z at cruise height
        changes the arm's reachable-xy manifold under the level-gripper orientation constraint, so
        xy error plateaus just above the gate threshold and never crosses it. The continuous ramp
        starts lowering z as soon as xy makes ANY progress inside xy_far, so there's no fixed point
        to get stuck on.

        `at_cruise` gates xy the same way (hold current xy while still below cruise height; chase
        the real target once at/above it) but is deliberately ONE-SIDED (`ee_now.z > cruise_z -
        z_tol`), not a symmetric `abs(...) < z_tol` band. A symmetric band straddles cruise_z on
        BOTH sides, so the moment real descent lowers ee below the band -- which is the intended,
        designed outcome of phase 3 -- the flag flips back to "still lifting" and FREEZES wp_xy at
        whatever xy the arm happened to reach, discarding any remaining xy error permanently (z
        keeps ramping down per descend_frac, but xy never gets to close the gap). Worse, if ee_now.z
        settles/noises right at the band edge (physics settling, warm-started QP jitter), the flag
        can flicker step to step, snapping wp_xy between "frozen at current position" and "the full,
        fixed target xy" -- a visible teleport, not the intended smooth chase. The one-sided form
        only trips "still lifting" while genuinely BELOW cruise height, so it stays continuously
        true through TRANSIT and the start of DESCEND, and only flips once descent has committed
        (by which point xy should already be converged per the ramp's own design) -- confirmed via
        `--view`'s ik_target_w marker, which stopped visibly jumping after this change.
        """
        at_cruise = ee_now[:, 2] > (self._cruise_z - self._z_tol)
        wp_xy = torch.where(at_cruise.unsqueeze(-1), target_in_base[:, :2], ee_now[:, :2])
        xy_err = (ee_now[:, :2] - target_in_base[:, :2]).norm(dim=-1)
        descend_frac = ((self._xy_far - xy_err) / (self._xy_far - self._xy_near)).clamp(0.0, 1.0)
        wp_z = self._cruise_z + (target_in_base[:, 2] - self._cruise_z) * descend_frac
        return torch.cat([wp_xy, wp_z.unsqueeze(-1)], dim=-1)

    def _compute_camera(self, target_in_base: torch.Tensor,
                        cam_pos_base: torch.Tensor, cam_mat_base: torch.Tensor,
                        cam_joint_pos: torch.Tensor | None = None) -> torch.Tensor:
        """Closed-form absolute gaze, recomputed fresh every step from the REAL camera position
        (parallax-corrected — the superseded open-loop formula measured azimuth/elevation from the
        BASE ORIGIN, off by the ~0.6m head-mount height; for this scenario's close target (0.32m)
        that error is tens of degrees). Stateless — no integration of the previous command, so no
        gimbal-lock accumulation either: the superseded closed-loop version corrected yaw/pitch
        independently in the camera's OWN rotating frame (image-plane h_angle/v_angle), which loses
        yaw authority entirely as pitch nears the +-90 deg ROM boundary — it plateaued there at a
        real, un-decaying ~30-40 deg aim error (pitch pinned at the clamp). This version can't do
        that: yaw/pitch are each a single atan2/asin of the desired look direction, not a per-axis
        nudge, so there's no local linearization to break down.

        Kinematics (asset/duke_v2/head_cam/head_camera_dual.xml) — VERIFIED against MjModel forward
        kinematics (5 (yaw, pitch) samples/side, exact match to float precision, not just derived):
        yaw hinges on base_link's +Z; pitch hinges on the yaw-rotated frame's local +X (one relation
        for both cams — the right camera's yaw datum is flipped 180 deg since it's rear-mounted).
        Forward direction (unit vector, base frame) as a function of (yaw, pitch) — REMEASURED against
        the compiled model (both the standalone head_camera_dual.xml AND the humanoid-attached gimbal):
        a POSITIVE yaw joint swings the optical axis toward -y, not +y (the prior sign was a latent bug
        that only agreed with FK at yaw=0; off-axis it mis-aimed up to ~57 deg):
            left:  forward = ( cos(yaw)*cos(pitch), -sin(yaw)*cos(pitch), -sin(pitch))
            right: forward = (-cos(yaw)*cos(pitch),  sin(yaw)*cos(pitch), -sin(pitch))
        Inverting for the desired forward direction d = normalize(target - cam_pos), base frame:
            pitch = asin(-d_z)                    # in [-90, 90] — matches the joint ROM exactly
            yaw   = atan2(-d_y, d_x)     (left)    # cos(pitch) cancels — no divide-by-zero
            yaw   = atan2( d_y, -d_x)    (right)   # unified: atan2(-sign*d_y, sign*d_x)
        """
        sign = 1.0 if self.aim_side == "left" else -1.0
        # Closed-form solve via the shared, deploy-safe primitive (single source of truth for the
        # aim math + yaw sign). ROM=inf here so the raw (ideal) angles feed the trim below; the final
        # ROM clamp stays at the command write.
        yaw, pitch = gaze_aim(target_in_base, cam_pos_base, sign, float("inf"), float("inf"))

        det = fov_detect(cam_pos_base, cam_mat_base, target_in_base, H_HALF, V_HALF, NEAR, FAR)
        self.last_aim_error = det["aim_error"]
        self.last_in_fov = det["in_fov"]

        # Integral trim (see __init__): the frozen policy's residual action settles the
        # gimbal at a constant offset from the written reference, so writing the ideal
        # angles alone leaves a steady ~5 deg aim error. Integrate the observed shortfall
        # (ideal - actual joint) and pre-compensate the reference; steady state then puts
        # the ACTUAL joints on the ideal angles. Anti-windup: bias hard-clamped.
        if cam_joint_pos is not None:
            act_yaw = cam_joint_pos[:, self._aim_cam_yaw_col]
            act_pitch = cam_joint_pos[:, self._aim_cam_pitch_col]
            self._trim[:, 0] = (self._trim[:, 0] + self._trim_gain * (yaw - act_yaw)
                                ).clamp(-self._trim_max, self._trim_max)
            self._trim[:, 1] = (self._trim[:, 1] + self._trim_gain * (pitch - act_pitch)
                                ).clamp(-self._trim_max, self._trim_max)
            yaw = yaw + self._trim[:, 0]
            pitch = pitch + self._trim[:, 1]

        cam_cmd = self._camera_ref.command.clone()
        cam_cmd[:, self._aim_cam_yaw_col] = yaw.clamp(-_GAZE_YAW_ROM, _GAZE_YAW_ROM)
        cam_cmd[:, self._aim_cam_pitch_col] = pitch.clamp(-_GAZE_PITCH_ROM, _GAZE_PITCH_ROM)
        return cam_cmd


class DirectReachPolicy(HeuristicReachPolicy):
    """Same as HeuristicReachPolicy but skips the lift/transit/descend waypoint -- IK
    straight-lines to the raw target every step. Debug/sanity-check only: isolates whether
    the open final-err > min-err finding (see plan doc, 2026-07-04 target-jump-bug entry)
    is inherent to the IK/controller or introduced by the waypoint's cruise-height gating."""

    def _waypoint_target(self, target_in_base: torch.Tensor, ee_now: torch.Tensor) -> torch.Tensor:
        return target_in_base


class DualCubeSearchGaze(HeuristicReachPolicy):
    """Dual-camera dual-cube SEARCH -> ASSIGN -> TRACK gaze — no GT in the CONTROL path.

    Task premise (user-specified): the robot KNOWS the task has K cubes (K=2 today). Both
    cameras raster-scan their own hemisphere simultaneously; each camera must end up locked
    on a DIFFERENT cube.

    Perception abstraction (faithful to the real robot): a SIMULATED DETECTOR per camera
    reports (ID, position) for every cube inside that camera's 90x65 deg / 0.28-3.0 m
    frustum. In sim the ID/position come from the GT cube table handed to ``bind()`` — the
    stand-in for AprilTag registration (real hardware: the tag decoder emits the same
    (ID, pose) tuples; the tagged-cube assets exist under scene_object/tag_cube). The gaze
    CONTROL never reads that table directly — only detector reports gate what it may use,
    so the policy transfers to a real detector unchanged.

    The heuristic (all rules, no learning):
      REGISTRY   detections register cubes BY ID -> "who is who" is inferred, not given;
                 immune to cubes placed arbitrarily close (IDs can't collide).
      CLAIM      first cube found is claimed by its finder camera (the other camera keeps
                 scanning and, by ID exclusion, can only be triggered by the OTHER cube).
      ASSIGN     once all K cubes are registered, ONE 2x2 min-total-yaw-travel comparison
                 fixes the final camera<->cube pairing (hysteresis: never re-assigned).
      TRACK      each camera aims at ITS cube with the closed-form solve + its OWN
                 integral trim (the frozen v83 residual offset is per-joint — see
                 HeuristicReachPolicy._compute_camera).

    Arm: held at the home pose while the target is unassigned (``target_name=None``, the
    default — this policy then evaluates FINDING only, reach/grasp out of scope). Pass
    ``target_name`` to release the arm to the normal ``HeuristicReachPolicy`` IK reach once
    that cube is registered+assigned to this policy's aim-side camera (see ``_compute_arm``).
    Single-env only (asserted): scenario props exist once at the world origin (see
    pickplace_reach_env.Args.envs).
    """

    # Blur-safe gimbal speed cap (user requirement: smooth — no motion blur from fast
    # slews, but not slow either), calibrated from the D435: ~10 ms indoor exposure,
    # ~1 deg tolerable smear -> ~1.7 rad/s; cap at 1.5 rad/s (86 deg/s) = 0.03 rad per
    # 50 Hz control step. EVERY written gimbal command is slew-limited to this rate
    # (scan cruise, pitch row hops, reversal, TRACK acquisition swings — all smoothed).
    SLEW_MAX_STEP = 0.03              # rad per control step (= 1.5 rad/s at 50 Hz)
    # Motion-blur DETECTION gate (A-series blur_omega_max semantics), restored per
    # review: a frame taken while the PHYSICAL gimbal moves faster than this is
    # unusable — that step's detections drop. The scan motion is blur-BUDGETED to
    # stay under the gate (cruise 1.0 rad/s; row hops move pitch alone at
    # HOP_PITCH_STEP with yaw parked), leaving ~1 rad/s margin for the frozen v83
    # residual's jitter (typ. ±0.8, spikes 3-4 rad/s — spike frames still drop,
    # honestly). 2.0 rad/s assumes a shorter exposure than the 1.7 estimate above;
    # re-calibrate against the real camera before hardware. Instance-configurable:
    # __init__(blur_omega_max=...), 0 disables the gate.
    BLUR_OMEGA_MAX = 2.0              # rad/s physical gimbal speed above which frames drop
    # Row-hop pitch rate — DELIBERATELY under the slew cap, with yaw parked during
    # the hop (hops start at a sweep reversal, where yaw speed passes through zero
    # anyway). Rationale: close cubes only become visible on the steep row RIGHT
    # AFTER a hop, and a cap-speed hop (1.5 pitch + yaw reversal + jitter > gate)
    # used to blind the detector exactly at first sight (audited: in-frustum at step
    # ~165, registered only ~182). Costs ~0.2 s per hop, buys detection DURING it.
    HOP_PITCH_STEP = 0.02             # rad per control step (= 1.0 rad/s at 50 Hz)
    # Two raster rows instead of three (the 65 deg vertical FOV overlaps them into a
    # contiguous -12..+92 deg band) — recovers the sweep time the lower cruise costs.
    SCAN_PITCH = (0.35, 1.05)         # rad rows, top->down (tabletop cubes ~0.4-1.2 rad)
    # EACH camera sweeps the FULL circle (+-180 deg about its own datum; the +-270 deg
    # yaw ROM affords it): a single camera must be able to search the entire space by
    # itself (user requirement — robustness to the other camera being occluded/failed
    # or both cubes sitting in one hemisphere), not merely the two cameras' union.
    # The opposed datums still make the pair anti-phased in world azimuth, so joint
    # coverage stays fast while each camera is individually complete.
    YAW_LIM = 3.1416                  # rad sweep amplitude (+-180 deg = full circle each)
    YAW_RATE = 0.02                   # rad/step scan cruise (1.0 rad/s, comfortably under
                                      # the slew cap so the sweep stays smooth)
    def __init__(self, blur_omega_max: float | None = None,
                 target_name: str | None = None, **kw):
        super().__init__(**kw)
        # Cube id the ARM may reach for once a camera has it assigned (see _compute_arm);
        # None = find-only contract, arm holds home. Declared HERE (not swallowed by
        # **kw) -- HeuristicReachPolicy.__init__ doesn't accept it, and find-only
        # callers (gaze_search_eval) don't pass it.
        self._target_name = target_name
        # None -> class default; 0 -> gate disabled (every frame counts as sharp).
        self.blur_omega_max = self.BLUR_OMEGA_MAX if blur_omega_max is None else float(blur_omega_max)
        cam_names = list(self._camera_ref.target_names)
        self._cols = {s: (_camera_column(cam_names, "yaw", s), _camera_column(cam_names, "pitch", s))
                      for s in ("left", "right")}
        self._side_sign = {"left": 1.0, "right": -1.0}   # opposed mount: right yaw datum flipped
        self._side_trim = {s: [0.0, 0.0] for s in ("left", "right")}
        self._yaw_ref, self._yaw_dir, self._row = 0.0, 1.0, 0
        self._pitch_ref = float(self.SCAN_PITCH[0])      # smoothed scan pitch (row-hop state)
        self.cubes_world: dict = {}    # DETECTOR's GT table: id -> world pos (bind())
        self.registry: dict = {}       # id -> world pos, filled on first detection
        self.assigned = {"left": None, "right": None}
        self.assign_final = False
        self.found_step: dict = {}     # id -> step of first detection
        self.last_aim_to: dict = {}    # (side, id) -> aim angle (diagnostics, every pair)
        self.last_occluded: dict = {}  # (side, id) -> in-FOV but line of sight blocked
        self.last_sharp: dict = {}     # side -> frame sharp enough to detect (blur gate)
        self._los_fn = None
        self._cmd_last = None          # last written gimbal command (slew-limiter state)
        self._t = 0

    def reset(self) -> None:
        """Clear SEARCH/ASSIGN/TRACK state on top of the base trim/diagnostics reset --
        without this, a post-episode ENTER-reset restarts physics but the policy stays
        latched on the previous episode's assignment/scan-sweep position, so it never
        re-searches (looks like "reset does nothing" from the SEARCH phase's perspective)."""
        super().reset()
        self._side_trim = {s: [0.0, 0.0] for s in ("left", "right")}
        self._yaw_ref, self._yaw_dir, self._row = 0.0, 1.0, 0
        self.registry = {}
        self.assigned = {"left": None, "right": None}
        self.assign_final = False
        self.found_step = {}
        self.last_aim_to = {}
        self.last_occluded = {}
        self.last_sharp = {}
        self._cmd_last = None
        self._t = 0

    def bind(self, env, cubes_world: dict, los_fn=None) -> None:
        """Attach the env (both cameras' site poses are read directly — the env only feeds
        one side's) and the detector's GT cube table {id: world pos}.

        ``los_fn(side, cube_id) -> bool`` is the LINE-OF-SIGHT gate: detection requires
        in_fov AND clear line of sight (same rule as the A-series:
        camera_learner_env "detection = in_fov ∧ unoccluded"). None = geometry-only
        detector (sees through obstacles — fine for open-tabletop scenes, wrong behind
        shelf boards / a reaching arm). The eval driver supplies an EXACT mj_ray test
        against the full scene (tables/shelf/self-body all occlude); a training consumer
        would plug the fast analytic capsule test (camera_occlusion_capsule) instead —
        same slot, different fidelity/cost point."""
        assert env.num_envs == 1, "DualCubeSearchGaze is single-env (scenario props exist once)"
        self._env = env
        self._los_fn = los_fn
        self.cubes_world = {k: torch.as_tensor(v, dtype=torch.float32, device=env.device)
                            for k, v in cubes_world.items()}
        entity = env.scene["robot"]
        self._site_ids = {s: entity.find_sites(f"cam_{s}_rgb")[0][0] for s in ("left", "right")}

    # ------------------------------------------------------------------ frames
    def _frames(self):
        """Both cameras' (pos, mat) in base frame + a world->base point transform."""
        entity = self._env.scene["robot"]
        base_pos = entity.data.root_link_pos_w
        from mjlab.utils.lab_api.math import matrix_from_quat
        R_bw = matrix_from_quat(entity.data.root_link_quat_w).transpose(-1, -2)
        cams = {}
        for s, sid in self._site_ids.items():
            pose = entity.data.site_pose_w[:, sid]
            pos_b = torch.matmul(R_bw, (pose[:, :3] - base_pos).unsqueeze(-1)).squeeze(-1)
            mat_b = torch.matmul(R_bw, matrix_from_quat(pose[:, 3:7]))
            cams[s] = (pos_b, mat_b)

        def to_base(p_w: torch.Tensor) -> torch.Tensor:
            return torch.matmul(R_bw, (p_w.view(1, 3) - base_pos).unsqueeze(-1)).squeeze(-1)

        return cams, to_base

    # ------------------------------------------------------------ arm: gated reach
    def _compute_arm(self, physics_qpos, target_in_base, arm_home_pose, reach_ee_pos_base):
        # NOTE: gate on "assigned to ANY side", not self.aim_side -- aim_side is fixed at
        # construction from the target's x-sign (front/rear convention inherited from
        # HeuristicReachPolicy, which has only one real camera decision to make). The ASSIGN
        # step here picks camera<->cube pairing by real min-total-yaw-travel geometry and can
        # come out either way, independent of that x-sign convention.
        if self._target_name is None or self._target_name not in self.assigned.values():
            # Still searching (or no target_name -- find-only contract). Diagnostics marker
            # (env.ik_target_w) sits at the idle gripper site, not None: HeuristicReachPolicy.__init__
            # only defaults last_reach_target_base to None, and pickplace_reach_env.step()
            # unconditionally .unsqueeze()s it every step -- leaving it None crashes the
            # first step of a --gaze rollout.
            self.last_reach_target_base = reach_ee_pos_base.detach()
            return arm_home_pose.clone()
        return HeuristicReachPolicy._compute_arm(
            self, physics_qpos, target_in_base, arm_home_pose, reach_ee_pos_base)

    # ------------------------------------------------------------------- gaze
    def _compute_camera(self, target_in_base, cam_pos_base, cam_mat_base, cam_joint_pos=None):
        import math
        self._t += 1
        cams, to_base = self._frames()
        cube_base = {cid: to_base(w) for cid, w in self.cubes_world.items()}

        # Per-camera physical angular speed -> frame sharpness (blur gate). Proxy: the
        # optical-axis sweep rate is bounded by |yaw_rate|·cos(pitch) + |pitch_rate|;
        # we use the conservative |yaw|+|pitch| of the REAL joint velocities.
        entity = self._env.scene["robot"]
        jvel = entity.data.joint_vel[0]
        tid = self._camera_ref.target_ids
        for s in ("left", "right"):
            yc, pc = self._cols[s]
            omega = float(abs(jvel[tid[yc]]) + abs(jvel[tid[pc]]))
            self.last_sharp[s] = self.blur_omega_max <= 0 or omega < self.blur_omega_max

        # SIMULATED DETECTOR — the only place the GT table is read (AprilTag stand-in).
        # detection = in_fov ∧ clear line of sight ∧ sharp frame (A-series rules).
        det = {s: [] for s in ("left", "right")}
        self.last_aim_to = {}
        self.last_in_fov_pair = {}                                   # (side, id) -> frustum test
        self.last_occluded = {}                                      # (side, id) -> LOS blocked?
        for s in ("left", "right"):
            p, m = cams[s]
            for cid, pb in cube_base.items():
                r = fov_detect(p, m, pb, H_HALF, V_HALF, NEAR, FAR)
                self.last_aim_to[(s, cid)] = float(r["aim_error"][0])
                self.last_in_fov_pair[(s, cid)] = bool(r["in_fov"][0])
                blocked = (self._los_fn is not None
                           and bool(r["in_fov"][0])                  # LOS only decides in-FOV cases
                           and not self._los_fn(s, cid))
                self.last_occluded[(s, cid)] = blocked
                if bool(r["in_fov"][0]) and not blocked and self.last_sharp[s]:
                    det[s].append(cid)
                    if cid not in self.registry:                     # REGISTRY (by ID)
                        self.registry[cid] = self.cubes_world[cid]
                        self.found_step[cid] = self._t

        # ASSIGN: all K registered -> one 2x2 min-total-yaw-travel comparison (final).
        if not self.assign_final and len(self.registry) == len(self.cubes_world) >= 2:
            ids = sorted(self.registry)
            cost = {}
            for s in ("left", "right"):
                p, _ = cams[s]
                cur = float(cam_joint_pos[0, self._cols[s][0]]) if cam_joint_pos is not None else 0.0
                for cid in ids:
                    need_yaw, _ = gaze_aim(cube_base[cid], p, self._side_sign[s],
                                           float("inf"), float("inf"))
                    need = float(need_yaw[0])
                    delta = (need - cur + math.pi) % (2 * math.pi) - math.pi
                    cost[(s, cid)] = abs(delta)
            keep = cost[("left", ids[0])] + cost[("right", ids[1])]
            swap = cost[("left", ids[1])] + cost[("right", ids[0])]
            self.assigned = ({"left": ids[0], "right": ids[1]} if keep <= swap else
                             {"left": ids[1], "right": ids[0]})
            self.assign_final = True
        elif not self.assign_final and len(self.registry) == 1:
            cid = next(iter(self.registry))
            if cid not in self.assigned.values():                    # CLAIM (first finder)
                finder = "left" if cid in det["left"] else "right"
                self.assigned[finder] = cid

        # diagnostics kept compatible with the reach env's metrics plumbing
        s0 = "left"
        cid0 = self.assigned[s0] or next(iter(self.cubes_world))
        p0, m0 = cams[s0]
        r0 = fov_detect(p0, m0, cube_base[cid0], H_HALF, V_HALF, NEAR, FAR)
        self.last_aim_error, self.last_in_fov = r0["aim_error"], r0["in_fov"]

        cam_cmd = self._camera_ref.command.clone()

        # TRACK: each assigned camera aims at ITS cube (closed-form + per-side trim).
        for s in ("left", "right"):
            cid = self.assigned[s]
            if cid is None:
                continue
            yc, pc = self._cols[s]
            sign = self._side_sign[s]
            p, _m = cams[s]
            yaw, pitch = gaze_aim(cube_base[cid], p, sign, float("inf"), float("inf"))
            if cam_joint_pos is not None:
                # Unwrap the closed-form yaw onto the branch nearest the CURRENT joint
                # angle. A cube sitting right behind this camera's datum (the crossed
                # assignment) puts atan2 on its ±pi cut: base sway flips the command
                # between +180 and -180 deg, and chasing that flip is a 2*pi swing
                # through the slew limiter while the trim integrates garbage. The
                # ±270 deg yaw ROM lets the joint simply hold e.g. +185 deg instead.
                cur_yaw = float(cam_joint_pos[0, yc])
                err_yaw = (float(yaw[0]) - cur_yaw + math.pi) % (2 * math.pi) - math.pi
                tgt_yaw = cur_yaw + err_yaw
                # The nearest branch may lie OUTSIDE the yaw ROM (found on G1: ideal
                # -55 deg seen from a joint near +265 -> nearest branch +305 > +270
                # limit; the ROM clamp then pins the joint at the stop FOREVER while
                # every step re-picks the same out-of-range branch and the trim
                # rails). Fold onto the in-ROM branch and let the slew limiter take
                # the long way around.
                if abs(tgt_yaw) > _GAZE_YAW_ROM:
                    tgt_yaw -= math.copysign(2.0 * math.pi, tgt_yaw)
                    err_yaw = tgt_yaw - cur_yaw
                yaw = torch.full_like(yaw, tgt_yaw)
                tr = self._side_trim[s]
                # Trim is a STEADY-STATE corrector: integrate only near lock, so a
                # long acquisition swing (or the fold above) can't wind it to the
                # clamp with transit error it was never meant to fix.
                if abs(err_yaw) < 0.3:
                    tr[0] = min(0.35, max(-0.35, tr[0] + 0.3 * err_yaw))
                err_pitch = float(pitch[0]) - float(cam_joint_pos[0, pc])
                if abs(err_pitch) < 0.3:
                    tr[1] = min(0.35, max(-0.35, tr[1] + 0.3 * err_pitch))
                yaw, pitch = yaw + tr[0], pitch + tr[1]
            cam_cmd[:, yc] = yaw.clamp(-_GAZE_YAW_ROM, _GAZE_YAW_ROM)
            cam_cmd[:, pc] = pitch.clamp(-_GAZE_PITCH_ROM, _GAZE_PITCH_ROM)

        # SEARCH: every unassigned camera runs the serpentine raster (shared phase).
        # Blur-budgeted motion: cruise sits under the detection gate, and a ROW HOP
        # moves pitch alone at HOP_PITCH_STEP while yaw parks (hops begin exactly at
        # a sweep reversal, where yaw speed passes through zero anyway) — frames
        # taken DURING the hop stay sharp, so the steep row's first sight registers.
        searching = [s for s in ("left", "right") if self.assigned[s] is None]
        if searching:
            hop = self.SCAN_PITCH[self._row] - self._pitch_ref
            if abs(hop) > 1e-9:                        # ROW HOP: pitch only, yaw parked
                self._pitch_ref += max(-self.HOP_PITCH_STEP, min(self.HOP_PITCH_STEP, hop))
            else:
                self._yaw_ref += self._yaw_dir * self.YAW_RATE
                if abs(self._yaw_ref) >= self.YAW_LIM:
                    self._yaw_ref = max(-self.YAW_LIM, min(self.YAW_LIM, self._yaw_ref))
                    self._yaw_dir *= -1.0
                    self._row = (self._row + 1) % len(self.SCAN_PITCH)
            for s in searching:
                yc, pc = self._cols[s]
                cam_cmd[:, yc] = self._yaw_ref
                cam_cmd[:, pc] = self._pitch_ref

        # SLEW LIMITER — smoothness/no-blur guarantee (see SLEW_MAX_STEP). Bounds the
        # per-step change of EVERY written gimbal command: pitch row hops, sweep
        # reversals, SEARCH->TRACK acquisition and cross-hemisphere assignment swings
        # all become <= 1.5 rad/s pans. Steady-state tracking is untouched (its
        # per-step deltas are far below the cap). Seeded from the REAL joint angles so
        # the very first command cannot jump either.
        if self._cmd_last is None:
            self._cmd_last = (cam_joint_pos[:, :].clone() if cam_joint_pos is not None
                              else cam_cmd.clone())
        cam_cmd = self._cmd_last + (cam_cmd - self._cmd_last).clamp(
            -self.SLEW_MAX_STEP, self.SLEW_MAX_STEP)
        self._cmd_last = cam_cmd.clone()
        return cam_cmd
