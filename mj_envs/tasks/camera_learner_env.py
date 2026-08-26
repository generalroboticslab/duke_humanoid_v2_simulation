"""CameraLearnerEnv: high-level gaze policy composed on a FROZEN v83 (active-vision A1).

The learner is HIGH-LEVEL: env.step receives only the 4D gaze
SETPOINT; it is written to the ``camera_ref`` command, then the frozen v83 policy runs on this
env's live student obs and outputs the FULL 31D action (legs + arms + a camera-tracking residual
on top of the setpoint). v83 is thus folded into the transition function — FlashSAC trains one
flat 4D actor and never sees v83 as a policy (no hierarchy in the trainer; runner reads
``policy_action_dim``).

Two state pieces this env owns:
  - frozen v83 ``policy_fn`` (loaded once; batch-agnostic, runs on our student-obs window).
  - ``cam_target_w`` (N, 2, 3): per-episode world targets, one per camera (left=front hemi,
    right=rear hemi). Allocated by the obs terms (camera_terms), resampled here on reset.

Design decisions:
  - Targets sampled from the just-reset base STATE (root_link_pos/quat are primary free-joint
    state, set by the reset events before sim.forward), mapped base→world so they stay
    world-fixed (do not drift with the base). Front/rear hemispheres match the opposed mount so
    each gimbal yaws within ROM.
  - Setpoint clamped per-column to the gimbal ROM (yaw ±4.7124, pitch ±1.5708) using the
    camera_ref term's joint-name order, so the actor's raw output maps to reachable references.
  - policy_fn reads the PREVIOUS step's obs_buf (1-step control latency) — the natural loop.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import mdp
from mjlab.utils.lab_api.math import matrix_from_quat

from flash_sac.env_wrapper import ManagerBasedRlEnvWithFinalObs
from tasks.frozen_policy import load_frozen_policy
from tasks.camera_perception import fov_detect, xmat_optical_axis, BeliefBufferBase
from tasks.camera_occlusion_capsule import cast_unoccluded_capsule
from tasks.camera_terms import H_HALF, V_HALF, NEAR, FAR, _CAM_SITES, _N_TARGETS

from mjlab_util.observation_terms import (
    JointPosTerm, JointVelTerm, LastActionTerm, BaseAngVelTerm,
    ProjectedGravityTerm, CommandTerm, TargetJointPosTerm,
    CamTargetPosBaseTerm, CamJointPosTerm, CamJointVelTerm,
    CamBaseAngVelTerm, CamProjectedGravityTerm
)
from mjlab_util.obs_buffer import ObsBuffer, PolicyObsSpec

_YAW_ROM, _PITCH_ROM = 4.7124, 1.5708
_EL_HALF = math.radians(40.0)        # target elevation half-range (within pitch reach)
_RANGE_MIN, _RANGE_MAX = 0.6, 2.5    # target distance shell (within [near, far])
_FLOOR_MARGIN = 0.1                  # A5: min target height above ground (m); below = nonsensical
                                     # (buried) + reads as bogus "occluded" (camera ray hits floor)
# Per-camera azimuth windows (base frame, +x forward): left↔front, right↔rear.
_AZ_WINDOWS = ((-0.5 * math.pi, 0.5 * math.pi), (0.5 * math.pi, 1.5 * math.pi))

# A4 occlusion: capsules approximating the self-occluder set (torso + upper-arms + forearms) as
# (anchor_a, anchor_b, radius). Each anchor = one body name, or "shoulder_mid" = mean of the two
# shoulders (a synthetic torso top). Camera bodies are NOT included — the mount must not self-occlude.
# Radii FIT to the raycast clear-fraction in smoke_occlusion_capsule.py (binary gate; mesh-edge slop
# tolerated). Endpoints read live from body_link_pos_w each step.
_OCC_CAPSULES = (
    ("base_link", "shoulder_mid", 0.22),   # torso trunk (fat; base_link geom rbound ~0.24)
    ("shoulder_2_L", "elbow_L", 0.06),     # left upper arm
    ("elbow_L", "wrist_1_L", 0.06),        # left forearm
    ("shoulder_2_R", "elbow_R", 0.06),     # right upper arm
    ("elbow_R", "wrist_1_R", 0.06),        # right forearm
)


class CameraLearnerEnv(ManagerBasedRlEnvWithFinalObs):
    """Frozen v83 (loco+arm+cam-tracking) ⊕ learned high-level gaze setpoint (4D)."""

    def __init__(self, *, cfg, device, v83_task: str, v83_ckpt: str,
                 target_resample_steps: int = 0, target_full_sphere: bool = False,
                 occlusion: bool = False, pomdp: bool = False, belief_decay: float = 0.95,
                 blur_omega_max: float | None = None, deployable_belief: bool = False,
                 **kwargs):
        # A5 (POMDP): set BEFORE super so the belief obs term — built during the obs-manager
        # construction inside super().__init__ — reads the right decay when it lazily allocates
        # env.belief (camera_terms._ensure_belief). pomdp ⇒ occlusion (detection needs the
        # line-of-sight gate); asserted after super once _occlusion is resolved.
        self._pomdp = bool(pomdp)
        self._belief_decay = float(belief_decay)

        # Motion-blur detection gate (realism lever): a detection is ACCEPTED only when this
        # camera's optical-axis angular speed (world frame) is below blur_omega_max — a fast slew
        # smears the image past usability. None = OFF (A1–A5 byte-stable). Set pre-super (read in
        # _reset_idx, which super().__init__ triggers) alongside the prev-axis cache.
        self._blur_omega_max = None if blur_omega_max is None else float(blur_omega_max)
        self._prev_optic_axis = None                 # (N, ncam, 3) world optical axes, prev step

        super().__init__(cfg=cfg, device=device, **kwargs)

        self.policy_action_dim = 4  # runner.setup reads this -> trains a 4D actor

        # A2 (locomotion): re-draw both targets every N control steps so the walking base cannot
        # carry them out of the reachable depth shell (option C). 0 = off (A1 standing: targets
        # fixed for the whole episode, byte-stable). Staggered per-env via episode_length_buf.
        self._target_resample_steps = int(target_resample_steps)

        # A3 (learned assignment): sample BOTH targets over the FULL azimuth sphere (not the A1/A2
        # front/rear hemisphere split). With the hemisphere split the camera<->target pairing is
        # forced by geometry (front cam sees front target); full-sphere makes either camera able to
        # cover either target, so the assignment-free max-over-cameras reward must LEARN the split.
        self._target_full_sphere = bool(target_full_sphere)

        # A4 (occlusion gate): after each step, test cam->target line of sight against self-body
        # capsules (camera_occlusion_capsule) and store self.cam_unoccluded (N, 2, 2) bool, read by
        # the coverage reward (camera_terms) to zero a blocked camera's aim so the max-over-cameras
        # picks the unoccluded one (v83 arm self-occlusion forces reassign). Capsule (not raycast):
        # the rays kernel costs 2.5x training wall-clock; self-body is the only occluder in A4-A6.
        # Resolve the per-camera optical-site indices (entity-local, into site_pose_w) and the
        # capsule anchor BODY indices (into body_link_pos_w) by NAME.
        self._occlusion = bool(occlusion)
        if self._occlusion:
            asset = self.scene["robot"]
            self._occ_site_ids = [asset.find_sites(n)[0][0] for n in _CAM_SITES]
            names = list(asset.body_names)

            def _anchor(a):  # body name -> [idx]; "shoulder_mid" -> [idxL, idxR] (averaged at use)
                if a == "shoulder_mid":
                    return [names.index("shoulder_2_L"), names.index("shoulder_2_R")]
                return [names.index(a)]

            self._cap_a = [_anchor(a) for a, _b, _r in _OCC_CAPSULES]
            self._cap_b = [_anchor(b) for _a, b, _r in _OCC_CAPSULES]
            self._cap_r = torch.tensor([r for _a, _b, r in _OCC_CAPSULES], device=self.device)
            self.cam_unoccluded = torch.ones(self.num_envs, len(_CAM_SITES), _N_TARGETS,
                                             dtype=torch.bool, device=self.device)

        # A5: detection = in_fov ∧ unoccluded -> POMDP needs the occlusion gate. The belief obj
        # itself was already lazily built by the belief obs term during super() (env.belief).
        if self._pomdp:
            assert self._occlusion, "pomdp=True requires occlusion=True (detection needs line-of-sight)"

        # Route A (deployability proof): replace the world-store belief with the BASE-frame
        # BeliefBufferBase, which uses ONLY deployable signals (PnP-chain base-frame measurement +
        # gyro egomotion bridge; no GT world target, no absolute base pose). The belief obs term reads
        # ``env.belief`` live each compute (camera_terms.camera_belief_obs), so reassigning here — after
        # the term lazily built the world-store object in super() — makes the term use the new object
        # with no term change. _update_belief branches on this flag. Eval-only knob (default False =
        # world-store A5, byte-stable); same 29D actor schema -> a FROZEN CamA5* policy runs unchanged.
        self._deployable_belief = bool(deployable_belief)
        if self._deployable_belief:
            assert self._pomdp, "deployable_belief requires pomdp=True (it replaces the A5 belief)"
            self.belief = BeliefBufferBase(self.num_envs, _N_TARGETS, self._belief_decay, self.device)

        self._policy_fn, self._student_group, adim = load_frozen_policy(
            v83_task, v83_ckpt, device=str(device), num_oracle_envs=2, close_oracle=True
        )
        assert adim == 31, f"frozen v83 action_dim {adim} != 31"

        # camera_ref must be the passive holder; camera action term must be last (so the learner's
        # 4D maps to the camera tail and v83's 27 legs+arms are absolute).
        self._camera_ref = self.command_manager.get_term("camera_ref")
        assert hasattr(self._camera_ref, "set_command"), "camera_ref must be PassiveGazeCommandTerm"
        assert self.action_manager.active_terms[-1] == "joint_pos_camera", \
            self.action_manager.active_terms

        # Per-column ROM clamp aligned to the camera_ref joint order.
        rom = [_YAW_ROM if "yaw" in n else _PITCH_ROM for n in self._camera_ref.target_names]
        self._rom = torch.tensor(rom, device=self.device)

    def _reset_idx(self, env_ids):
        self._resample_targets(env_ids)
        super()._reset_idx(env_ids)
        # A5: clear the belief for the reset envs (new episode, new hidden targets). Guard hasattr
        # because the first mjlab reset can run before the belief term's lazy allocation in edge
        # build orders; under pomdp the term builds env.belief during obs construction, so it exists.
        if self._pomdp and hasattr(self, "belief"):
            self.belief.reset(env_ids)
        # Blur gate: re-seed the prev optical axis for the reset envs to the CURRENT axis, so the
        # post-reset head teleport is not read as a huge angular speed that would spuriously gate the
        # first detection. Guard hasattr(_occ_site_ids): the first mjlab reset fires inside super()
        # before __init__ resolves the occlusion sites.
        if self._blur_omega_max is not None and hasattr(self, "_occ_site_ids"):
            axes = self._optic_axes()
            if self._prev_optic_axis is None:
                self._prev_optic_axis = axes
            else:
                self._prev_optic_axis[env_ids] = axes[env_ids]

    def _resample_targets(self, env_ids) -> None:
        """Sample 2 world-fixed targets for the given envs.

        A1/A2: front/rear hemisphere split (_AZ_WINDOWS), one per camera. A3 (target_full_sphere):
        both targets over the full azimuth -> learned assignment (either camera may cover either).
        """
        ids = env_ids.long()
        n = ids.numel()
        if n == 0:
            return
        asset = self.scene["robot"]
        root_pos = asset.data.root_link_pos_w[ids]                       # (n, 3)
        root_mat = matrix_from_quat(asset.data.root_link_quat_w[ids])    # (n, 3, 3)
        root_z = root_pos[:, 2]                                          # (n,) base height (world)
        az_windows = ((-math.pi, math.pi),) * _N_TARGETS if self._target_full_sphere else _AZ_WINDOWS
        for k, (az_lo, az_hi) in enumerate(az_windows):
            az = torch.empty(n, device=self.device).uniform_(az_lo, az_hi)
            if self._pomdp:
                # A5 above-floor clamp: raise the per-sample elevation floor so target_z =
                # root_z + rng·sin(el) ≥ _FLOOR_MARGIN (no buried targets). Clamping el (not z) keeps
                # the spawn ON its sampled ray. For standing root_z≈0.57 the numerator is negative so
                # el_lo ≤ 0 (most of [-40°,40°] survives; only deep-downward+far rays are lifted); the
                # clamp(-_EL_HALF,_EL_HALF) also guards a fallen-base resample (root_z<margin → el_lo>0,
                # bounded to _EL_HALF so the uniform width never goes negative). NOTE: rng MUST be
                # drawn here (before el) since el_lo depends on it — this reorders the RNG stream, so it
                # lives ONLY on the pomdp branch to keep A1-A4 target sampling byte-identical.
                rng = torch.empty(n, device=self.device).uniform_(_RANGE_MIN, _RANGE_MAX)
                sin_lo = ((_FLOOR_MARGIN - root_z) / rng).clamp(-1.0, 1.0)
                el_lo = torch.asin(sin_lo).clamp(-_EL_HALF, _EL_HALF)
                el = torch.rand(n, device=self.device) * (_EL_HALF - el_lo) + el_lo
            else:
                # A1-A4 byte-stable original: el then rng, both via empty().uniform_ (legacy stream).
                el = torch.empty(n, device=self.device).uniform_(-_EL_HALF, _EL_HALF)
                rng = torch.empty(n, device=self.device).uniform_(_RANGE_MIN, _RANGE_MAX)
            d_base = torch.stack([                                       # (n, 3) base-frame unit
                torch.cos(el) * torch.cos(az),
                torch.cos(el) * torch.sin(az),
                torch.sin(el),
            ], dim=-1)
            target_base = (rng.unsqueeze(-1) * d_base).unsqueeze(-1)     # (n, 3, 1)
            target_w = root_pos + torch.matmul(root_mat, target_base).squeeze(-1)
            self.cam_target_w[ids, k] = target_w

    def step(self, cam_4d: torch.Tensor):
        cam_4d = cam_4d.to(self.device)
        clamped = cam_4d.clamp(-self._rom, self._rom).detach()
        self._camera_ref.set_command(clamped)
        
        # Frozen policy uses student obs from the previous step
        v83_31d = self._policy_fn({self._student_group: self.obs_buf[self._student_group]}).detach()
        
        out = super().step(v83_31d)
        
        # A2 periodic resample (option C): after the base has stepped, re-draw both targets for any
        # env that just crossed a multiple of _target_resample_steps. Keyed on the per-env step
        # counter so jumps stagger across envs (not a synchronized refresh). Fresh-reset envs
        # (buf==0, already resampled by _reset_idx) are excluded by the buf>0 guard.
        if self._target_resample_steps > 0:
            buf = self.episode_length_buf
            due = (buf % self._target_resample_steps == 0) & (buf > 0)
            ids = due.nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() > 0:
                self._resample_targets(ids)
                # A5: a periodic resample TELEPORTS both targets to new world positions. The belief
                # must forget the old estimate — otherwise it keeps pointing the actor at the stale
                # ghost position (decaying conf) instead of triggering a re-search, which inflates
                # re-acquisition latency and trains the policy on a belief that contradicts reality.
                # Same env-granularity as belief.reset (resample redraws BOTH targets per env).
                # _update_belief below re-detects this step if a fresh target is already in view.
                if self._pomdp:
                    self.belief.reset(ids)
        # A4: refresh occlusion from the just-stepped (post-physics) scene. The coverage reward read
        # it during super().step (this step's reward used the PREVIOUS occlusion) -> 1-step latency,
        # same convention as the frozen policy_fn reading the prior obs_buf; negligible at 50 Hz.
        if self._occlusion:
            self._update_occlusion()
        # A5: fold this step's detections into the belief AFTER occlusion (detection needs the fresh
        # cam_unoccluded). Same 1-step latency as occlusion: the belief obs the actor reads NEXT step
        # reflects detections from THIS step — consistent with the reward's occlusion convention.
        if self._pomdp:
            self._update_belief()

        return (
            self.obs_buf,
            out[1],
            out[2],
            out[3],
            out[4]
        )

    def _update_occlusion(self) -> None:
        """Test each camera->each target vs self-body capsules; store self.cam_unoccluded (N,2,2)."""
        data = self.scene["robot"].data
        origins = torch.stack([data.site_pose_w[:, s, :3] for s in self._occ_site_ids], dim=1)  # (N,2,3)
        bp = data.body_link_pos_w                                        # (N, nbody, 3)
        caps_p = torch.stack([bp[:, ids].mean(dim=1) for ids in self._cap_a], dim=1)  # (N, C, 3)
        caps_q = torch.stack([bp[:, ids].mean(dim=1) for ids in self._cap_b], dim=1)  # (N, C, 3)
        self.cam_unoccluded = cast_unoccluded_capsule(
            origins, self.cam_target_w, caps_p, caps_q, self._cap_r, NEAR, ground_z=0.0,
        )

    def _optic_axes(self) -> torch.Tensor:
        """Per-camera optical axis (world frame), (N, ncam, 3) — third column of each cam site's R_wc.

        Used by the motion-blur gate to measure how fast each line of sight sweeps the world (gimbal
        slew ⊕ base/head rotation, both folded into the world-frame site orientation)."""
        pose = self.scene["robot"].data.site_pose_w                      # (N, S, 7)
        axes = [xmat_optical_axis(matrix_from_quat(pose[:, sid, 3:7])) for sid in self._occ_site_ids]
        return torch.stack(axes, dim=1)                                  # (N, ncam, 3)

    def _update_belief(self) -> None:
        """A5 detection -> belief.update. detected[k] = OR_cameras (in_fov_ck ∧ unoccluded_ck ∧ sharp_c).

        Per camera optical site, fov_detect vs both targets (batched (N,K)); AND with the just-
        computed cam_unoccluded[:, c] line-of-sight; OR over the 2 cameras. Measurement = GT world
        target pos (read only where detected; BeliefBuffer.update ignores undetected entries). Reuses
        the occlusion optical-site indices (_occ_site_ids); requires occlusion=True (asserted in init).

        Motion-blur gate (if blur_omega_max set): a camera's detections this step are dropped when its
        optical axis swept faster than blur_omega_max rad/s (finite-diff vs the previous step's axis) —
        a blurred frame is unusable. Gated so the no-blur path keeps the A5 RNG stream + detection bytes
        untouched.
        """
        data = self.scene["robot"].data
        pose = data.site_pose_w                                          # (N, S, 7)

        # Per-camera blur sharpness mask (N, ncam): True = slew slow enough to read.
        sharp = None
        if self._blur_omega_max is not None:
            axis_now = self._optic_axes()                               # (N, ncam, 3)
            if self._prev_optic_axis is None:
                omega = torch.zeros(self.num_envs, len(self._occ_site_ids), device=self.device)
            else:
                cos = (axis_now * self._prev_optic_axis).sum(-1).clamp(-1.0, 1.0)
                omega = torch.acos(cos) / self.step_dt                  # (N, ncam) rad/s
            self._prev_optic_axis = axis_now.detach()
            sharp = omega < self._blur_omega_max                        # (N, ncam)

        detected = torch.zeros(self.num_envs, _N_TARGETS, dtype=torch.bool, device=self.device)
        for ci, sid in enumerate(self._occ_site_ids):
            cam_pos = pose[:, sid, :3].unsqueeze(1)                      # (N, 1, 3)
            cam_mat = matrix_from_quat(pose[:, sid, 3:7]).unsqueeze(1)   # (N, 1, 3, 3)
            in_fov = fov_detect(cam_pos, cam_mat, self.cam_target_w,
                                H_HALF, V_HALF, NEAR, FAR)["in_fov"]      # (N, K)
            gate = self.cam_unoccluded[:, ci]                           # (N, K)
            if sharp is not None:
                gate = gate & sharp[:, ci].unsqueeze(-1)                # blurred cam contributes nothing
            detected = detected | (in_fov & gate)                       # (N, K)

        if self._deployable_belief:
            # Deployable path: bridge HELD bearings forward by gyro egomotion BEFORE snapping fresh
            # detections (the snap then overwrites the rotated value for detected targets). Measurement
            # is the base-frame target pose from the deployable PnP+encoder chain (world cancels — see
            # _target_base_meas / BeliefBufferBase). No world target, no absolute base pose used.
            self.belief.apply_egomotion(self._base_gyro(), self.step_dt)
            dir_base, dist = self._target_base_meas()
            self.belief.update(detected, dir_base, dist)
        else:
            self.belief.update(detected, self.cam_target_w)

    def _base_gyro(self) -> torch.Tensor:
        """Body-frame base angular velocity (N,3) from the IMU gyro sensor — the SAME signal the
        actor's base_ang_vel obs reads (CamBaseAngVelTerm: ``robot/imu_ang_vel``). Deployable."""
        return mdp.builtin_sensor(self, sensor_name="robot/imu_ang_vel")

    def _target_base_meas(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Deployable base-frame measurement of both targets -> (dir_base (N,K,3), dist (N,K)).

        On hardware = PnP target-in-camera composed with the gimbal+base encoder kinematics; the
        absolute base pose cancels in that chain. In sim the identical number is computed via
        R_bw @ (target_w - root_pos) (world cancels), matching camera_belief_obs's fresh-detect dir/dist
        exactly, so a freshly detected target's obs is bit-equal to the world-store path."""
        data = self.scene["robot"].data
        root_pos = data.root_link_pos_w                                  # (N, 3)
        root_mat = matrix_from_quat(data.root_link_quat_w)               # (N, 3, 3)
        rel = self.cam_target_w - root_pos.unsqueeze(1)                  # (N, K, 3) world
        rel_base = torch.matmul(
            root_mat.transpose(-1, -2).unsqueeze(1), rel.unsqueeze(-1)
        ).squeeze(-1)                                                    # (N, K, 3) base frame
        dist = torch.linalg.norm(rel_base, dim=-1)                       # (N, K)
        dir_base = rel_base / dist.clamp_min(1e-6).unsqueeze(-1)
        return dir_base, dist

    def update_visualizers(self, visualizer) -> None:
        """Draw the 2 gaze targets in the play viewer (run.py play), green = in a camera's FOV.

        Hooked by NativeMujocoViewer each frame (manager_based_rl_env.update_visualizers contract).
        Without this the viewer shows the gimbal slewing but not WHERE it should look, so tracking
        can't be eyeballed. Each target sphere is GREEN if either camera currently frames it
        (fov_detect), else RED — same visibility test as the reward. Drawn for the selected env only.
        """
        super().update_visualizers(visualizer)
        if not hasattr(self, "cam_target_w"):
            return
        idx = int(getattr(visualizer, "env_idx", 0))
        asset = self.scene["robot"]
        sids = [asset.find_sites(n)[0][0] for n in _CAM_SITES]
        pose = asset.data.site_pose_w                                    # (N, S, 7)
        unocc = getattr(self, "cam_unoccluded", None)                    # gate green by line-of-sight
        for k in range(_N_TARGETS):
            tgt = self.cam_target_w[idx, k]                              # (3,)
            seen = False
            for ci, sid in enumerate(sids):
                det = fov_detect(pose[idx:idx + 1, sid, :3],
                                 matrix_from_quat(pose[idx:idx + 1, sid, 3:7]),
                                 tgt.unsqueeze(0), H_HALF, V_HALF, NEAR, FAR)
                framed = bool(det["in_fov"][0])
                if unocc is not None:
                    framed = framed and bool(unocc[idx, ci, k])
                seen = seen or framed
            color = (0.1, 0.9, 0.1, 1.0) if seen else (0.9, 0.1, 0.1, 1.0)
            visualizer.add_sphere(tgt.detach().cpu().numpy(), 0.08, color)
