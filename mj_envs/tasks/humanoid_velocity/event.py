"""Event helpers: fast resets, push events, deferred DR, event parameter curriculum."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.entity import Entity
from mjlab.envs.mdp.dr._core import _get_entity_indices, _select_default_values
from mjlab.envs.mdp.dr.body import (
    _decompose_pseudo_inertia_J,
    _reconstruct_pseudo_inertia_J,
)
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
import torch.nn.functional as F
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, sample_uniform, yaw_quat


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# =============================================================================
# Gait Phase Clock Reset
# =============================================================================


class randomize_gait_period:
    """Reset event: allocate gait phase buffers and randomize gait period per env.

    Allocates three env-level buffers on the correct device:
        env._gait_period        [B]: cycle duration (s), randomized per episode.
        env._gait_phase         [B]: current phase ∈ [0, 1).
        env._phase_updated_step [B]: episode_length_buf value at last phase update.

    _gait_period initialized to period_min (not zero) to prevent div-by-zero in
    GaitPhase when the obs term runs before the first reset event fires.
    _phase_updated_step initialized to -1 so the first call always triggers an update.

    Params:
        period_range: (min, max) gait period in seconds (default (0.55, 0.75)).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        period_range = cfg.params.get("period_range", (0.55, 0.75))
        self.period_min = float(period_range[0])
        self.period_max = float(period_range[1])
        env._gait_period = torch.full((env.num_envs,), self.period_min, device=env.device)
        # Assign a fixed gait polarity to each env, set once here and restored
        # on every episode reset (not re-randomized). Even envs: phase_init=0.0
        # (L-leading: L in stance first). Odd envs: phase_init=0.5 (R-leading:
        # R in stance first). This ensures the rollout buffer always contains an
        # equal mix of both gait polarities, correcting the L-leading bias that
        # would arise if all envs reset to 0.
        #
        # Reward safety: foot_phase_contact_match and foot_stance_slip_penalty
        # both derive phase_R = (phase_L + 0.5) % 1.0, so the reward is purely
        # relative to _gait_phase — it produces identical signal for phase_init=0
        # and phase_init=0.5 (the two cases are symmetric by construction).
        #
        # Overhead: one arange at init; per-reset is an indexed gather vs a
        # scalar fill — identical CUDA cost. Zero per-step overhead.
        env._gait_phase_init = (torch.arange(env.num_envs, device=env.device) % 2) * 0.5
        env._gait_phase = env._gait_phase_init.clone()
        env._phase_updated_step = torch.full(
            (env.num_envs,), -1, device=env.device, dtype=torch.long
        )
        # Pre-allocated scratch buffer for in-place uniform sampling; avoids a
        # per-reset allocation when torch.rand would otherwise create a new tensor.
        self._rand_buf = torch.empty(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **_) -> None:
        n = len(env_ids)
        self._rand_buf[:n].uniform_(self.period_min, self.period_max)
        env._gait_period[env_ids] = self._rand_buf[:n]
        env._gait_phase[env_ids] = env._gait_phase_init[env_ids]
        env._phase_updated_step[env_ids] = -1


# =============================================================================
# Fast Reset Events (previously local_events.py)
# =============================================================================


class RootResetFast:
    """Optimized root reset event.

    Pre-allocates range tensors to avoid Python loop overhead and dictionary lookups
    during the reset cycle. Supports fast-path for zero rotations or velocities.
    """
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.pose_range = cfg.params.get("pose_range", {})
        self.vel_range = cfg.params.get("velocity_range", {})
        self.asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

        # Pre-allocate pose range tensor
        range_list = []
        self.has_rot = False
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
            r = self.pose_range.get(key, (0.0, 0.0))
            range_list.append(r)
            if key in ["roll", "pitch", "yaw"] and (r[0] != 0 or r[1] != 0):
                self.has_rot = True
        self.pose_ranges = torch.tensor(range_list, device=env.device)

        # Pre-allocate velocity range tensor
        v_range_list = []
        self.has_vel = False
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
            r = self.vel_range.get(key, (0.0, 0.0))
            v_range_list.append(r)
            if r[0] != 0 or r[1] != 0:
                self.has_vel = True
        self.vel_ranges = torch.tensor(v_range_list, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        asset: Entity = env.scene[self.asset_cfg.name]
        num_resets = len(env_ids)

        # 1. Pose sampling
        pose_samples = sample_uniform(
            self.pose_ranges[:, 0], self.pose_ranges[:, 1], (num_resets, 6), device=env.device
        )

        default_root_state = asset.data.default_root_state
        assert default_root_state is not None
        root_states = default_root_state[env_ids]

        positions = (
            root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
        )

        if self.has_rot:
            orientations_delta = quat_from_euler_xyz(
                pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
            )
            orientations = quat_mul(root_states[:, 3:7], orientations_delta)
        else:
            orientations = root_states[:, 3:7]

        # 2. Velocity sampling
        if self.has_vel:
            vel_samples = sample_uniform(
                self.vel_ranges[:, 0], self.vel_ranges[:, 1], (num_resets, 6), device=env.device
            )
            velocities = root_states[:, 7:13] + vel_samples
        else:
            velocities = root_states[:, 7:13]

        # 3. Write to sim
        asset.write_root_link_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


class JointResetFast:
    """Optimized joint reset event.

    Pre-allocates joint indexing and avoids redundant clones during the reset cycle.
    """
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        self.pos_range = cfg.params.get("position_range", (0.0, 0.0))
        self.vel_range = cfg.params.get("velocity_range", (0.0, 0.0))

        asset = env.scene[self.asset_cfg.name]
        self.joint_ids = self.asset_cfg.joint_ids
        if self.joint_ids is None:
            self.joint_ids = asset.default_joint_ids
        if isinstance(self.joint_ids, list):
            self.joint_ids = torch.tensor(self.joint_ids, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        asset: Entity = env.scene[self.asset_cfg.name]

        # Sample positions
        default_pos = asset.data.default_joint_pos[env_ids][:, self.joint_ids]
        joint_pos = default_pos + sample_uniform(*self.pos_range, default_pos.shape, env.device)

        # Clamp to limits
        limits = asset.data.soft_joint_pos_limits[env_ids][:, self.joint_ids]
        joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

        # Sample velocities
        default_vel = asset.data.default_joint_vel[env_ids][:, self.joint_ids]
        if self.vel_range[0] != 0 or self.vel_range[1] != 0:
            joint_vel = default_vel + sample_uniform(*self.vel_range, default_vel.shape, env.device)
        else:
            joint_vel = default_vel

        asset.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.joint_ids
        )

# HACK, TODO: MAKE IT PRINCIPLED!
# Measured cmd-0.4 walking marginals of the grid policy (probe/gaitinit_feasibility.py, n=102400
# on-manifold states). Ordered to the 12 leg joints below. center=p50, half=(p90-p10)/2, vpk=|vel|_p90.
# Left legs swing in phase with sin(2π·φ); right legs antiphase (KF_SIGN negates sin & cos).
# hip_1/knee/ankle_1 dominate the sagittal swing (hip_1 is NOT a yaw joint — measured, not assumed).
_KF_LEG_NAMES = [
    "left_hip_1_joint", "left_hip_2_joint", "left_hip_3_joint",
    "left_knee_joint", "left_ankle_1_joint", "left_ankle_2_joint",
    "right_hip_1_joint", "right_hip_2_joint", "right_hip_3_joint",
    "right_knee_joint", "right_ankle_1_joint", "right_ankle_2_joint",
]
_KF_CENTER = [0.385, 0.028, -0.103, 0.778, 0.346, -0.022,
              -0.360, -0.046, 0.043, -0.777, -0.389, 0.056]
_KF_HALF = [0.120, 0.063, 0.098, 0.202, 0.091, 0.054,
            0.095, 0.060, 0.119, 0.156, 0.111, 0.046]
_KF_VPK = [2.166, 0.895, 1.554, 4.285, 1.113, 0.560,
           1.672, 0.882, 2.119, 3.336, 1.472, 0.729]
_KF_SIGN = [1., 1., 1., 1., 1., 1., -1., -1., -1., -1., -1., -1.]
_KF_CAPTURE_CMD = 0.4   # speed the marginals were measured at; keyframe scales linearly off this


class GaitKeyframeResetFast:
    """Reset event: put a fraction ``f`` of resetting envs into an ON-manifold WALKING state
    instead of the standing crouch, for deploy-safe reference-state-initialization (Exp2).

    Motivation. The grid policy
    stands at low command because every training episode resets FROM standing — the critic never
    samples a low-cmd WALKING state, so it never grounds a high value there. This event injects a
    broad, on-manifold gait state at reset; a fraction of those draw a low command from the (unchanged,
    independent) grid sampler, giving exactly the low-cmd-walking coverage the dead zone lacks.

    On-manifold, not uniform-random (the manifold trap): a walker's viable states are a thin correlated
    surface (pose, joint velocity, root velocity coupled by the gait). Uniform-random init lands off it
    → stumble → trains recovery-TO-standing (wrong sign). This keyframe is a parametric stride whose
    per-joint pose centers/amplitudes and velocity peaks are the MEASURED cmd-0.4 marginals of THIS
    policy (constants above), so it sits on the real gait surface. Verified on-manifold by
    probe/gaitinit_feasibility.py: SYNTH keyframe re-entry matched the REAL captured-state term-rate
    (0.585 vs 0.535 at cmd 0.20) and velocity-sustain, both far below the standing (COLD) baseline.

    Deploy-safe: reset-mode only. The keyframe is written at t=0 and never referenced by any obs or
    reward (gait_phase is absent from this task's actor/critic obs, phase rewards are zero-weighted), so
    this is NOT the rejected control-side phase clock — the policy runs free after reset.

    Parametrization: for each masked env sample speed s~U(0, s_max) and phase φ~U(0, 2π); set legs to
    center + (s/0.4)·half·(sign·sinφ), joint velocities (s/0.4)·vpk·(sign·cosφ), and forward root
    velocity s. Root pose (incl. HOME height 0.59) is left as reset_base set it. Unmasked envs keep the
    standing reset — the deploy start condition and zero-cmd calm stay the majority.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.f = float(cfg.params.get("fraction", 0.3))
        self.s_max = float(cfg.params.get("s_max", 0.6))
        self.asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        asset = env.scene[self.asset_cfg.name]
        self.leg_ids = torch.tensor([asset.joint_names.index(n) for n in _KF_LEG_NAMES], device=env.device)
        dev = env.device
        self.center = torch.tensor(_KF_CENTER, device=dev)
        self.half = torch.tensor(_KF_HALF, device=dev)
        self.vpk = torch.tensor(_KF_VPK, device=dev)
        self.sign = torch.tensor(_KF_SIGN, device=dev)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **_) -> None:
        asset: Entity = env.scene[self.asset_cfg.name]
        # Mask a fraction f of the resetting envs (Bernoulli — no fixed count, cheap).
        pick = env_ids[torch.rand(len(env_ids), device=env.device) < self.f]
        k = pick.numel()
        if k == 0:
            return
        s = sample_uniform(0.0, self.s_max, (k, 1), device=env.device)      # (k,1) init speed
        phi = torch.rand(k, 1, device=env.device) * (2 * torch.pi)
        scale = s / _KF_CAPTURE_CMD
        sph, cph = torch.sin(phi), torch.cos(phi)
        pos = self.center + scale * self.half * (self.sign * sph)           # (k,12) phase-split pose
        vel = scale * self.vpk * (self.sign * cph)                          # (k,12) phase-deriv velocity

        jp = asset.data.joint_pos[pick].clone()
        jv = asset.data.joint_vel[pick].clone()
        jp[:, self.leg_ids] = pos
        jv[:, self.leg_ids] = vel
        limits = asset.data.soft_joint_pos_limits[pick][:, self.leg_ids]
        jp[:, self.leg_ids] = jp[:, self.leg_ids].clamp(limits[..., 0], limits[..., 1])
        asset.write_joint_state_to_sim(jp, jv, env_ids=pick)

        # write_root_link_velocity_to_sim takes WORLD-frame velocity, but the stride pose is body-frame
        # forward. reset_base randomizes yaw, so rotate the forward speed into world by the root yaw —
        # else the robot glides world-x while facing elsewhere (off-manifold sideways drift).
        fwd_b = torch.zeros(k, 3, device=env.device)
        fwd_b[:, 0] = s[:, 0]
        root_vel = torch.zeros(k, 6, device=env.device)
        root_vel[:, :3] = quat_apply(yaw_quat(asset.data.root_link_quat_w[pick]), fwd_b)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=pick)


_BANK_DTYPE = torch.float16


class GaitBankResetFast:
    """Reset event: inject a fraction ``f`` of resetting envs into a REAL walking state
    harvested from this run's own successful episodes.

    Supersedes GaitKeyframeResetFast, which SYNTHESIZES the pose from hardcoded cmd-0.4
    marginals (_KF_* above). Those constants are absolute joint angles in one MJCF's frame:
    edit a link length or default_joint_pos and they silently go off-manifold (names still
    resolve, and the limit clamp at line 276 hides the drift), training recovery-TO-standing,
    the wrong sign. This class holds no MJCF-derived constant -- every width comes from
    asset.num_joints at init.

    Three phases, all inside this one reset callback, exploiting that ``env_ids`` is a SUBSET:
      COMMIT  the finished envs' pending rows -> ring, but only if they ended by time_out.
      HARVEST the LIVE (non-resetting) envs, whose qpos/qvel here are real mid-episode state.
      INJECT  draw a row from a reset-speed stratum and write it over the standing reset.
    Success is not knowable at snapshot time -- a state twenty steps before a fall looks
    healthy instantaneously -- hence the deferred per-env pending buffer.

    Raw qpos/qvel only, never the root_link_* properties: those read xpos/xquat/cvel, which
    refresh on sim.forward() AFTER this event, so they are a substep stale on read and report
    the PRE-reset x,y on write-back. Raw slices are current, need no frame conversion, and
    carry no pose-before-velocity ordering hazard. Root x,y are never stored and never
    written, so whatever reset_base placed survives untouched.

    The v_floor motion gate is load-bearing, not a refinement: time_out alone is satisfied by
    standing still, and at low command this policy DOES stand, so a timeout-only gate would
    fill the slow bins with standing states and silently no-op at exactly the speeds this
    exists to fix.

    Cold start is a no-op: an unwritten slot leaves the standing reset alone, so behaviour
    equals the parent until the policy can walk well enough to fill the bank. The next
    episode's command is deliberately INDEPENDENT of the harvested state's command (reset
    events run before command_manager.reset). This is a previous-success reset curriculum,
    not command-conditioned gait replay: it creates continued-motion, braking, redirection and
    reversal rollouts from a viable walking basin. Speed bins preserve diversity in the reset
    distribution; they do not predict the new command.

    Throughput and cost. Runs on every step that resets any env, so it is written to issue
    ZERO host-device syncs -- no .nonzero(), no boolean-mask compaction, no Python loop over
    bins, no ``if t.numel()`` guard. Rejected rows are routed to a write-only trash bin and
    injection is masked with torch.where instead of being compacted, trading a dense
    O(num_envs) pass (~2 MB of traffic, single-digit microseconds) for the ~9 syncs a
    compacting version needs. Storage is fp16 for the same reason as the FlashSAC replay
    frames (replay_buffer.py:93): quantities here are O(1e1) with 1e-3 relative resolution,
    far below actuator resolution and observation noise, and it halves both footprint and the
    per-step bandwidth of the harvest write. The quaternion is renormalized on read because
    fp16 rounding pushes it off the unit sphere.

    Params:
        fraction:    share of resetting envs to inject (default 0.3, matches GaitKeyframeResetFast).
        s_max:       top of the injected speed range (default 0.6, matches).
        v_floor:     minimum planar speed to harvest (default 0.1, reuses VelocityTrackingFailure's
                     calibrated cmd_xy_threshold rather than inventing a constant).
        n_bins:      speed bins spanning [v_floor, s_max] (default 6).
        depth:       rows per bin (default 1024).
        per_episode: candidate snapshots kept per episode, evenly spaced over its length
                     (default 8). This is the throughput knob: a single reservoir sample per
                     episode fills the bank ~8x slower and leaves it that much staler than the
                     policy generating it.
        warm_start:  serve the SYNTHETIC keyframe (the _KF_* marginals) to envs selected for
                     injection whose drawn stratum is still empty, instead of leaving them at the
                     standing reset. Default False = unchanged behaviour.

                     Motivated by measurement, not tidiness.
                     Under the REAL training command distribution the ring fills far slower than a
                     forced-slow probe suggests: at checkpoint 2500, 2048 envs, 3000 steps, bin
                     occupancy reaches only [81, 95, 148, 174, 215, 231] of depth 1024, and bin 0 --
                     the slow band -- is the SLOWEST to fill, not the fullest. Extrapolated, bin 0
                     does not saturate until roughly two thirds through a 15k run. Without this
                     flag the class therefore trains most of the run "as the parent MINUS the
                     keyframe", i.e. with the reset intervention DELETED rather than replaced, and
                     it is deleted longest in the slow band that forms dead-zone competence.

                     ONE Bernoulli draw decides whether an env is intervened on; slot validity only
                     picks the source. So the intervention RATE is exactly ``fraction`` at every
                     point in training, matching both GaitKeyframeResetFast and the warm_start=False
                     bank, and the only variable is which source served it. Splitting this into two
                     independent draws would make the total rate a function of bank occupancy and
                     reintroduce the confound.

                     The keyframe path writes strictly less state than a bank row -- leg joints plus
                     a yaw-rotated forward speed, root pose left standing as reset_base placed it --
                     which is deliberate: it is GaitKeyframeResetFast's write set, unchanged, so the
                     handover is a source swap and not also a write-set change.

                     Cost: reintroduces the _KF_* constants this class exists to retire. Bounded --
                     they are a transient bootstrap, not the steady-state source, and their share
                     decays to zero as the ring fills.

                     REJECTED as a design. A hardcoded sinusoid
                     with hand-measured per-joint amplitudes IS a gait model, which is the thing this
                     class exists to remove; its numbers do not redeem it. Kept only so the 10.12 run
                     stays reproducible. Use ``sample_filled`` instead -- it closes the same cold-bank
                     hole using real harvested states.
        sample_filled: draw the injected row from the rows the bank actually HOLDS, instead of
                     uniformly over the slot address space. Default False = unchanged.

                     The bug this fixes (10.11 measured it without naming it): the incumbent draws
                     ``slot ~ U{0, depth-1}`` and gives up if that slot is unwritten, so the chance a
                     selected env is actually served equals that bin's FILL FRACTION. At the measured
                     81/1024 in the slow bin, 92% of slow-band injections silently no-op back to the
                     standing reset -- the exact reset distribution the dead zone is made of. The bank
                     already held 81 real slow-walking states; the sampler could not reach them.

                     Also serves an empty bin from the rest of the ring rather than declining: any
                     real state beats no state. That asserts nothing about gait shape, and the class
                     already treats the speed stratum as "not a target" (see above) -- the next
                     episode's command is independent of it either way.
        priority:    FALSIFIED -- do not build on this. ...BankPriNormOpen ran a completely full bank
                     (6144/6144) with the best available form of this weighting and still lost to
                     uniform-over-filled on both dead-zone speeds, at identical termination (0.011)
                     and episode length. The mechanism is unsound, not undertuned: the score below
                     is accrued from a snapshot to EPISODE END, and with
                     ``resampling_time_range=(2, 8)`` against a 20 s episode ~4 command periods fall
                     inside that window, so it measures a random command SEQUENCE rather than the
                     state it is filed under. No choice of reference repairs an estimand that is
                     dominated by command draw. Kept as the record of a falsified experiment; the
                     working form of this idea moved to the command axis, where one command period,
                     one reward window and one cell are already aligned -- see
                     ``GridAdaptiveVelocityCommand._grid_probabilities`` in command.py.

                     weight the ``sample_filled`` draw by MEASURED performance instead of uniformly:
                     rows the policy handled WORST are replayed most. Requires sample_filled.

                     Each stored row carries the realized ``perf_term`` reward rate over the remainder
                     of its own episode -- reward accumulated after the snapshot, divided by steps
                     lived after it. Weight is the deficit against the bank's own population mean,
                     ``clamp(mean_rate - rate, min=0)``, so the reference is measured from this run
                     rather than set by hand, and it tracks the policy as it improves. Zero total
                     deficit falls back to uniform-over-filled.

                     Why a deficit and not the raw score: replaying what already works teaches nothing
                     and narrows the bank (11, survivorship). Why the population mean and not a
                     constant: any fixed target is a hand pick, and a perfect policy would pin the
                     weights forever. This also subsumes the v_floor gate's job on the standing case --
                     a standing state harvested under a standing command scores WELL, so it earns a
                     small weight and is rarely replayed, where the hand-set floor had to guess.
        cmd_norm:    reference the ``priority`` deficit against a fit in COMMAND MAGNITUDE instead of
                     against a single population mean. Requires priority. Default False = the
                     falsified global-mean form, kept only so BankPri stays reproducible.

                     Why. The global-mean deficit was
                     predicted to concentrate replay on the slow band and instead concentrated it on
                     FAST states: drawable rows averaged root speed 0.423 against 0.406 for the whole
                     bank, and the arm posted the worst dead zone of the wave. The reason is that a
                     raw reward rate confounds two things -- a fast command is intrinsically harder to
                     track, so "low tracking reward" reads mostly as "moving fast", not as "handled
                     badly". Selection by an unnormalized score therefore selects speed.

                     Intended fix: regress the stored rate on the command magnitude the row was
                     harvested under, ordinary least squares over the live bank, and take the deficit
                     against that LINE rather than against a scalar, leaving the residual -- how much
                     worse this state went than states at the same command generally go.

                     THAT CLAIM IS FALSE AS IMPLEMENTED, and the paragraph above stated it as fact.
                     The regression is errors-in-variables: the rate spans ~4 command periods while
                     the regressor ``cmag`` is the command at the snapshot INSTANT, one of them. With
                     n periods in the window the fitted slope recovers only ``beta / n``, so the
                     correction is attenuated toward slope 0 -- which is exactly the scalar-mean form
                     it exists to replace. The measured speed-selection bias moved +0.017 (scalar) ->
                     -0.006 (here) -> -0.025 (Open): a sign flip but roughly a third of a correction,
                     which is what beta/n predicts. Accumulating cmag over the window would remove
                     the BIAS but not the VARIANCE, and the residual would still rank rows mostly by
                     which commands they happened to draw.

                     The fit is estimated from the bank each call, closed form, no bandwidth, no bins,
                     no hand-set target: nothing here is chosen, which is the whole point of preferring
                     it to a per-magnitude bucket table.
        perf_term:   reward term read for ``priority`` (default "track_linear_velocity"). Read from
                     reward_manager._episode_sums, which mjlab zeroes in reward_manager.reset() --
                     called AFTER the reset event manager (manager_based_rl_env.py:558 vs 572), so the
                     finished episode's totals are still live when _commit reads them.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.f = float(cfg.params.get("fraction", 0.3))
        self.s_max = float(cfg.params.get("s_max", 0.6))
        self.v_floor = float(cfg.params.get("v_floor", 0.1))
        self.n_bins = int(cfg.params.get("n_bins", 6))
        self.depth = int(cfg.params.get("depth", 1024))
        self.per_episode = int(cfg.params.get("per_episode", 8))
        self.warm_start = bool(cfg.params.get("warm_start", False))
        self.sample_filled = bool(cfg.params.get("sample_filled", False))
        self.priority = bool(cfg.params.get("priority", False))
        self.cmd_norm = bool(cfg.params.get("cmd_norm", False))
        self.perf_term = str(cfg.params.get("perf_term", "track_linear_velocity"))
        assert not (self.priority and not self.sample_filled), "priority needs sample_filled"
        assert not (self.cmd_norm and not self.priority), "cmd_norm needs priority"
        self.asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

        asset = env.scene[self.asset_cfg.name]
        dev = env.device
        # Address arrays are torch.int; advanced indexing needs long. Forced onto the device:
        # indexing a CUDA tensor with a CPU index tensor synchronizes, every step.
        idx = asset.indexing
        self.jq = idx.joint_q_adr.long().to(dev)
        self.jv = idx.joint_v_adr.long().to(dev)
        self.rq = idx.free_joint_q_adr[2:7].long().to(dev)  # z + quat wxyz; x,y excluded
        self.rv = idx.free_joint_v_adr.long().to(dev)
        self.rv2 = self.rv[:2].clone()  # planar linear velocity, materialized once

        self.nj = asset.num_joints
        self.W = 2 * self.nj + 11
        self._Z, self._V = 2 * self.nj, 2 * self.nj + 5
        self.stride = max(1, int(env.max_episode_length) // self.per_episode)

        # Row n_bins is a write-only trash row. Commits that failed the time_out or motion
        # gate are routed there rather than compacted out, which would cost a device sync.
        self.ring = torch.zeros(self.n_bins + 1, self.depth, self.W, device=dev, dtype=_BANK_DTYPE)
        self.valid = torch.zeros(self.n_bins + 1, self.depth, dtype=torch.bool, device=dev)
        self.ptr = torch.zeros(self.n_bins + 1, dtype=torch.long, device=dev)
        self.pend = torch.zeros(env.num_envs, self.per_episode, self.W, device=dev, dtype=_BANK_DTYPE)
        self.pend_bin = torch.full((env.num_envs, self.per_episode), -1, dtype=torch.long, device=dev)
        # Realized post-snapshot performance, fp32: it is compared against a population mean, and
        # fp16's 1e-3 relative resolution would quantize the small deficits that carry the signal.
        self.rate = torch.zeros(self.n_bins + 1, self.depth, device=dev)
        # Command magnitude each row was harvested under: the regressor that separates "handled
        # badly" from "moving fast" (see cmd_norm).
        self.cmag = torch.zeros(self.n_bins + 1, self.depth, device=dev)
        self.pend_cmag = torch.zeros(env.num_envs, self.per_episode, device=dev)
        # Reward sum and episode length AT each pending snapshot; the difference against the values
        # at episode end is what "performance AFTER this state" means.
        self.pend_rs = torch.zeros(env.num_envs, self.per_episode, device=dev)
        self.pend_tl = torch.zeros(env.num_envs, self.per_episode, device=dev)
        self.env_idx = torch.arange(env.num_envs, device=dev)
        # Assigning a PYTHON scalar into an advanced-indexed view synchronizes (verified with
        # torch.cuda.set_sync_debug_mode). A 0-dim device tensor broadcasts identically and
        # does not, so every scalar written through a fancy index below is one of these.
        self._true = torch.ones((), dtype=torch.bool, device=dev)
        self._false = torch.zeros((), dtype=torch.bool, device=dev)
        self._neg1 = torch.full((), -1, dtype=torch.long, device=dev)

        if self.warm_start:
            self.leg_ids = torch.tensor(
                [asset.joint_names.index(n) for n in _KF_LEG_NAMES], device=dev
            )
            self.kf_center = torch.tensor(_KF_CENTER, device=dev)
            self.kf_half = torch.tensor(_KF_HALF, device=dev)
            self.kf_vpk = torch.tensor(_KF_VPK, device=dev)
            self.kf_sign = torch.tensor(_KF_SIGN, device=dev)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # Needed so event_manager.reset() drops pending rows on a manual reset; without it a
        # stale snapshot survives into the next episode and is committed under its verdict.
        self.pend_bin[slice(None) if env_ids is None else env_ids] = self._neg1

    def _bin(self, v: torch.Tensor) -> torch.Tensor:
        t = (v - self.v_floor) / (self.s_max - self.v_floor)
        return (t * self.n_bins).long().clamp_(0, self.n_bins - 1)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **_) -> None:
        asset: Entity = env.scene[self.asset_cfg.name]
        d = asset.data.data  # raw mjwarp Data (see foot_height_fast for the same access)
        self._commit(env, env_ids)
        self._harvest(env, env_ids, d)
        self._inject(env, env_ids, asset, d)

    def _commit(self, env, env_ids: torch.Tensor) -> None:
        # env.reset_time_outs is assigned only in step(), so a manual env.reset() would
        # AttributeError on it. Read the manager, which always exists after setup.
        tm = getattr(env, "termination_manager", None)
        b = self.pend_bin[env_ids]  # (R, K); -1 where nothing was harvested at that slot
        good = (b >= 0) if tm is None else (tm.time_outs[env_ids].unsqueeze(1) & (b >= 0))
        b = torch.where(good, b, self.n_bins).flatten()
        rows = self.pend[env_ids].flatten(0, 1)

        # Sorting by bin makes each bin's rows contiguous, so a row's rank within its bin is
        # its position minus the position of its run's first element; offset that by the bin
        # pointer. More than ``depth`` rows can reach one bin when synchronized envs time out
        # together, so modulo slots can repeat. Select one whole-row winner per destination
        # before writing: duplicate advanced-index writes are unordered on CUDA and can tear a
        # physical state into fields from different source rows.
        # Realized reward RATE over the episode remainder after each snapshot. Rate, not total, so a
        # snapshot late in an episode is not penalized for having less episode left to earn in.
        if self.priority:
            rs_end = env.reward_manager._episode_sums[self.perf_term][env_ids].unsqueeze(1)
            tl_end = env.episode_length_buf[env_ids].float().unsqueeze(1)
            rate = ((rs_end - self.pend_rs[env_ids]) / (tl_end - self.pend_tl[env_ids]).clamp_min(1.0)).flatten()
        else:
            rate = torch.zeros_like(b, dtype=torch.float32)

        order = torch.argsort(b, stable=True)
        bs = b[order]
        pos = torch.arange(bs.numel(), device=bs.device)
        first = torch.full((self.n_bins + 1,), bs.numel(), dtype=torch.long, device=bs.device)
        first.scatter_reduce_(0, bs, pos, "amin", include_self=True)
        slot = (self.ptr[bs] + pos - first[bs]) % self.depth
        dest = bs * self.depth + slot
        n_dest = (self.n_bins + 1) * self.depth
        winner = torch.full((n_dest,), -1, dtype=torch.long, device=bs.device)
        winner.scatter_reduce_(0, dest, pos, "amax", include_self=True)
        write = winner >= 0
        ordered_rows = rows[order]
        # Sentinel makes the zero-commit path valid without a host-side ``if``: winner=-1
        # gathers row 0, while real source position p gathers p+1.
        candidates = torch.cat((torch.zeros_like(self.ring.view(-1, self.W)[:1]), ordered_rows), dim=0)
        selected = candidates[winner + 1]
        ring = self.ring.view(n_dest, self.W)
        ring[:] = torch.where(write.unsqueeze(1), selected, ring)
        # Same winner index, so a row and its score can never come from different source episodes.
        cand_rate = torch.cat((torch.zeros(1, device=bs.device), rate[order]))
        rate_ring = self.rate.view(n_dest)
        rate_ring[:] = torch.where(write, cand_rate[winner + 1], rate_ring)
        if self.cmd_norm:
            cmag = self.pend_cmag[env_ids].flatten()
            cand_c = torch.cat((torch.zeros(1, device=bs.device), cmag[order]))
            cmag_ring = self.cmag.view(n_dest)
            cmag_ring[:] = torch.where(write, cand_c[winner + 1], cmag_ring)
        valid = self.valid.view(n_dest)
        valid[:] = torch.where(write, self._true, valid)
        self.ptr.index_add_(0, b, torch.ones_like(b))  # bincount would sync; index_add_ does not
        self.ptr %= self.depth
        self.pend_bin[env_ids] = self._neg1

    def _harvest(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, d) -> None:
        # Slot index is DERIVED from episode progress, so per_episode snapshots land evenly
        # spread over the episode with no pointer state and no reservoir bookkeeping.
        # episode_length_buf is incremented before _reset_idx and zeroed after it, so it is
        # live here for every env outside env_ids.
        k = (env.episode_length_buf // self.stride).clamp_(0, self.per_episode - 1)
        v = d.qvel[:, self.rv2].norm(dim=1)  # world planar speed, current (not stale)
        gate = (env.episode_length_buf % self.stride) == 0
        gate[env_ids] = self._false  # mid-reset state, already overwritten by reset_base
        gate &= (v >= self.v_floor) & (v <= self.s_max)

        row = torch.cat(
            (d.qpos[:, self.jq], d.qvel[:, self.jv], d.qpos[:, self.rq], d.qvel[:, self.rv]),
            dim=1,
        ).to(_BANK_DTYPE)
        self.pend[self.env_idx, k] = torch.where(gate.unsqueeze(1), row, self.pend[self.env_idx, k])
        self.pend_bin[self.env_idx, k] = torch.where(gate, self._bin(v), self.pend_bin[self.env_idx, k])
        if self.priority:
            rs = env.reward_manager._episode_sums[self.perf_term]
            tl = env.episode_length_buf.float()
            self.pend_rs[self.env_idx, k] = torch.where(gate, rs, self.pend_rs[self.env_idx, k])
            self.pend_tl[self.env_idx, k] = torch.where(gate, tl, self.pend_tl[self.env_idx, k])
            if self.cmd_norm:
                # Command magnitude AT the snapshot. The rate measures what happened after it under
                # roughly this command, so this is the regressor the residual must be taken against.
                c = env.command_manager.get_term("twist").vel_command_b[:, :2].norm(dim=1)
                self.pend_cmag[self.env_idx, k] = torch.where(gate, c, self.pend_cmag[self.env_idx, k])

    def _weights(self) -> torch.Tensor:
        """Per-slot sampling weight over the real bins, flattened. Zero on unwritten slots.

        Uniform-over-filled unless ``priority``, in which case the weight is each row's deficit
        against the bank's own population mean reward rate. The mean is recomputed here rather than
        tracked incrementally: it is a (n_bins, depth) reduction, cheaper than the reset traffic it
        rides along with, and it needs no smoothing constant or staleness argument.
        """
        w = self.valid[: self.n_bins].float()
        if self.priority:
            n = w.sum().clamp_min(1.0)
            rate = self.rate[: self.n_bins]
            if self.cmd_norm:
                # OLS of rate on command magnitude over the live bank, weighted by validity so
                # unwritten slots contribute nothing. Closed form, so no iteration and no sync.
                x = self.cmag[: self.n_bins]
                mx = (x * w).sum() / n
                my = (rate * w).sum() / n
                var = (w * (x - mx) ** 2).sum()
                slope = torch.where(var > 0, (w * (x - mx) * (rate - my)).sum() / var.clamp_min(1e-12),
                                    torch.zeros((), device=w.device))
                ref = my + slope * (x - mx)   # per-row expectation at ITS OWN command magnitude
            else:
                ref = (rate * w).sum() / n    # falsified scalar form, kept for reproducibility
            d = (ref - rate).clamp_min(0.0) * w
            # A policy uniformly good across the bank has zero deficit everywhere; fall back to
            # uniform rather than to an all-zero distribution that would inject nothing.
            w = torch.where(d.sum() > 0, d, w)
        return w.view(-1)

    def _draw(self, b: torch.Tensor, r: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one flat ring index per env, weighted, preferring bin ``b``'s slice.

        Inverse-CDF over the whole ring: the per-bin slice is selected by sampling the uniform
        variate inside that bin's cumulative interval, so bins and weights are handled by the same
        two lines and neither needs a compaction or a rejection loop. An empty bin has a zero-width
        interval and its draw lands in the global pool instead, which is the intended degradation --
        a real state at another speed beats the standing reset.
        """
        cdf = self._weights().cumsum(0)
        end = (b + 1) * self.depth - 1
        hi = cdf[end]
        lo = torch.where(b > 0, cdf[(b * self.depth - 1).clamp_min(0)], torch.zeros((), device=dev))
        span = hi - lo
        # Zero-width bin -> draw from the entire populated ring instead of declining.
        empty = span <= 0
        lo = torch.where(empty, torch.zeros((), device=dev), lo)
        span = torch.where(empty, cdf[-1], span)
        idx = torch.searchsorted(cdf, lo + torch.rand(r, device=dev) * span)
        idx = idx.clamp_max(cdf.numel() - 1)
        return idx, self.valid.view(-1)[idx]

    def _inject(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, asset: Entity, d) -> None:
        r = env_ids.numel()
        dev = env_ids.device
        b = self._bin(torch.rand(r, device=dev) * self.s_max)  # reset-speed stratum, not a target
        # ONE Bernoulli draw decides intervention, and slot validity only decides WHICH source
        # serves it. Keeping the two separate would make the total intervention rate depend on bank
        # occupancy, which is exactly the confound warm_start exists to remove.
        sel = torch.rand(r, device=dev) < self.f
        if self.sample_filled:
            flat, hit = self._draw(b, r, dev)
            rows = self.ring.view(-1, self.W)[flat].float()
        else:
            slot = torch.randint(self.depth, (r,), device=dev)
            hit = self.valid[b, slot]
            rows = self.ring[b, slot].float()
        # An unwritten slot declines to inject, so a cold or sparse bin degrades to the parent's
        # standing reset (or, under warm_start, to the synthetic keyframe) instead of needing an
        # occupancy count and a rejection loop. Under sample_filled only a wholly empty bank misses.
        m = (sel & hit).unsqueeze(1)
        e = env_ids.unsqueeze(1)

        # No soft-limit clamp on bank rows: these joint angles came out of the simulator and are
        # already legal. Clamping is what let the hardcoded keyframe drift off-manifold unnoticed.
        jp = torch.where(m, rows[:, : self.nj], d.qpos[e, self.jq])
        jv = torch.where(m, rows[:, self.nj : self._Z], d.qvel[e, self.jv])
        rz = rows[:, self._Z : self._V]  # (z, qw, qx, qy, qz)
        rz[:, 1:] /= rz[:, 1:].norm(dim=1, keepdim=True).clamp_min(1e-6)  # undo fp16 rounding
        root_q = torch.where(m, rz, d.qpos[e, self.rq])
        root_v = torch.where(m, rows[:, self._V :], d.qvel[e, self.rv])

        if self.warm_start:
            # Selected but unserved: the bank had nothing in the drawn stratum. Masks are disjoint
            # by construction, so this overwrites only envs the bank left alone.
            k = (sel & ~hit).unsqueeze(1)
            s = torch.rand(r, 1, device=dev) * self.s_max
            phi = torch.rand(r, 1, device=dev) * (2 * torch.pi)
            scale = s / _KF_CAPTURE_CMD
            pos = self.kf_center + scale * self.kf_half * (self.kf_sign * torch.sin(phi))
            vel = scale * self.kf_vpk * (self.kf_sign * torch.cos(phi))
            lim = asset.data.soft_joint_pos_limits[env_ids][:, self.leg_ids]
            jp[:, self.leg_ids] = torch.where(k, pos.clamp(lim[..., 0], lim[..., 1]), jp[:, self.leg_ids])
            jv[:, self.leg_ids] = torch.where(k, vel, jv[:, self.leg_ids])
            # Root POSE stays as reset_base placed it (standing, upright) -- only forward speed is
            # written, and in WORLD frame, so the body-forward speed is rotated by the env's own
            # randomized root yaw. Skipping the rotation would slide the robot along world +x while
            # facing elsewhere, which is off-manifold and is the bug this mirrors from
            # GaitKeyframeResetFast.
            fwd = torch.zeros(r, 3, device=dev)
            fwd[:, 0] = s[:, 0]
            world_v = quat_apply(yaw_quat(root_q[:, 1:]), fwd)
            root_v[:, :3] = torch.where(k, world_v, root_v[:, :3])

        asset.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        d.qpos[e, self.rq] = root_q
        d.qvel[e, self.rv] = root_v


class PushRobotFast:
    """Optimized push event with curriculum-ready flat parameters.

    Uses flat parameter names like 'x_range', 'y_range' to facilitate
    EventParamCurriculum updates without nested dictionary complexity.
    """
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset_cfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
        self.record = cfg.params.get("record", False)
        if self.record:
            # Initialize buffers for push-aware rewards (e.g. PushGatedLinearVelocityTracking).
            # _last_push_delta_vel stores world-frame 6D delta velocity (linear+angular).
            env._last_push_step = torch.full((env.num_envs,), -1e6, device=env.device)
            env._last_push_delta_vel = torch.zeros((env.num_envs, 6), device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # No-op: timer reset is handled by event_manager._interval_term_time_left.
        # This method must exist so event_manager.reset() includes PushRobotFast in
        # _mode_class_term_cfgs["interval"] and resets the push timer to [5,15]s on
        # each episode reset, preventing a push from firing immediately after reset.
        pass

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

        asset: Entity = env.scene[self.asset_cfg.name]
        num_resets = len(env_ids)

        # Get current world-frame velocity
        vel_w = asset.data.root_link_vel_w[env_ids]

        # Sample delta velocity from flat parameter ranges passed via kwargs (updated by curriculum)
        range_list = []
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]:
            # Default to zero if not specified
            r = kwargs.get(f"{key}_range", (0.0, 0.0))
            range_list.append(r)

        ranges = torch.tensor(range_list, device=env.device)
        delta_vel = sample_uniform(ranges[:, 0], ranges[:, 1], (num_resets, 6), device=env.device)

        # Rotate linear push from base_link yaw frame to world frame.
        # The yaw quaternion strips roll/pitch so push direction is relative to robot heading.
        quat_yaw = yaw_quat(asset.data.root_link_quat_w[env_ids])
        delta_vel_w = torch.empty_like(delta_vel)
        delta_vel_w[:, :3] = quat_apply(quat_yaw, delta_vel[:, :3])
        delta_vel_w[:, 3:] = delta_vel[:, 3:]  # angular delta unchanged by yaw rotation

        # Write updated velocity (original + delta) to sim
        asset.write_root_link_velocity_to_sim(vel_w + delta_vel_w, env_ids=env_ids)

        if self.record:
            env._last_push_step[env_ids] = float(env.common_step_counter)
            env._last_push_delta_vel[env_ids] = delta_vel_w  # world-frame delta


def foot_height_fast(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Optimized foot height observation that skips redundant quaternion math.

    Directly accesses site_xpos from the GPU bridge, avoiding the 5ms quat_from_matrix
    overhead in mjlab's default site_pose_w implementation.
    """
    # Access mjwarp.Data via asset.data.data
    asset = env.scene[asset_cfg.name]
    return asset.data.data.site_xpos[:, asset_cfg.site_ids, 2]


# =============================================================================
# Deferred Domain Randomization (avoids per-reset mjwarp.set_const() cost)
# =============================================================================


@requires_model_fields(
    "body_mass", "body_ipos", "body_inertia", "body_iquat",
    recompute=RecomputeLevel.set_const,
)
def ee_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float],
    offset_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Rigidly attach a random point mass at a tool-mount site (gripper/payload DR).

    Replaces the ee_payload_mass + ee_com_offset pair, which used ``dr.body_mass`` and
    ``dr.body_com_offset``. Those write ``body_mass`` and ``body_ipos`` but never
    ``body_inertia``, so the added kilogram carried zero rotational inertia — for a 1 kg
    payload 5 cm out that omits 2.5e-3 kg·m², ~6x the wrist's entire principal inertia.
    They also both derive from defaults, so on a shared body the later event silently
    discarded the earlier one's write. One term writes all four fields, so neither
    problem can recur.

    Adding a point mass m_p at p is exactly a rank-1 update of the 4x4 pseudo-inertia
    J = [[sigma, h], [h^T, m]] used by mjlab's dr.pseudo_inertia:

        J' = J + m_p * v v^T,   v = [p; 1]

    (substituting I_p = m_p(|p|^2 I - p p^T) into sigma = 0.5 Tr(I) I - I gives m_p p p^T).
    Reusing mjlab's reconstruct/decompose keeps the parallel-axis algebra in one place and
    guarantees a physically realizable result — rejected hand-rolling the shift because a
    sign slip there produces a valid-looking but non-PSD tensor that MuJoCo accepts.

    Args:
        mass_range: (lo, hi) payload mass in kg, sampled per env and site.
        offset_range: (lo, hi) per-axis jitter around the site, in metres. Models unknown
            tool length/geometry.
        asset_cfg: Selects ``site_names`` — the tool-mount sites. The loaded bodies and
            the nominal payload location both come from the compiled model
            (``site_bodyid`` and ``site_pos``), so nothing about the asset's geometry is
            duplicated here; retargeting the payload means moving the site in the asset.

    Assumes the payload is rigidly attached (no slosh/compliance) and that each selected
    site belongs to a distinct body.

    Known overlap: the global ``body_mass`` scale DR also writes ``body_mass`` from
    defaults, so on the selected bodies whichever event fires last wins and the EE gets
    the payload instead of the +-15% density scale. Left alone deliberately — +-0.026 kg
    on a 0.173 kg body is noise next to a 0-1 kg payload, and making this term read
    current values instead of defaults would compound across firings.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    site_ids = _get_entity_indices(asset.indexing, asset_cfg, "site", False)
    body_ids = env.sim.model.site_bodyid[site_ids].to(env_ids.dtype)

    J = _reconstruct_pseudo_inertia_J(
        _select_default_values(env, "body_mass", env_ids, body_ids),
        _select_default_values(env, "body_ipos", env_ids, body_ids),
        _select_default_values(env, "body_inertia", env_ids, body_ids),
        _select_default_values(env, "body_iquat", env_ids, body_ids),
    )

    shape = (len(env_ids), len(site_ids))
    m_p = sample_uniform(mass_range[0], mass_range[1], shape, env.device)
    p = env.sim.model.site_pos[env_ids.long()[:, None], site_ids[None, :]] + sample_uniform(
        offset_range[0], offset_range[1], (*shape, 3), env.device
    )

    v = torch.cat([p, torch.ones(*shape, 1, device=env.device)], dim=-1)
    J = J + m_p[..., None, None] * (v.unsqueeze(-1) * v.unsqueeze(-2))

    mass, ipos, inertia, iquat = _decompose_pseudo_inertia_J(J)
    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    env.sim.model.body_mass[env_grid, body_grid] = mass
    env.sim.model.body_ipos[env_grid, body_grid] = ipos
    env.sim.model.body_inertia[env_grid, body_grid] = inertia
    env.sim.model.body_iquat[env_grid, body_grid] = iquat


class DeferredModelFieldsWrapper:
    """Wraps a @requires_model_fields DR function, suppressing its recompute trigger.

    WHY: mjwarp.set_const() costs ~40ms for ALL envs each time any requires_model_fields
    event fires on reset (not just the resetting envs — it's all-or-nothing). This wrapper
    lets the GPU field write happen per-episode (cheap, ~0.1ms) while deferring set_const
    to a PeriodicPhysicsRecompute interval event. Saves ~40ms/step at the cost of derived
    quantities (subtree mass, inv weights) being stale for up to N steps.

    Preserves .model_fields so sim.expand_model_fields() still allocates per-world GPU
    buffers. Sets .recompute = RecomputeLevel.none so event_manager skips set_const.
    """

    def __init__(self, fn):
        self.model_fields = getattr(fn, "model_fields", ())
        self.recompute = RecomputeLevel.none  # suppress set_const after each reset call
        self._fn = fn

    def __call__(self, env, env_ids, **kwargs):
        self._fn(env, env_ids, **kwargs)


class PeriodicPhysicsRecompute:
    """Periodic mode="interval" event that triggers mjwarp.set_const() globally.

    Used with DeferredModelFieldsWrapper. Mass/COM writes happen per-reset (cheap),
    but derived quantities (body_subtreemass, dof_invweight0, etc.) only sync after
    set_const. This event fires every N seconds globally, amortizing the ~40ms cost.

    The event_manager reads .recompute after __call__ and calls
    env.sim.recompute_constants(set_const) automatically — __call__ is a no-op.
    Configure as: mode="interval", is_global_time=True, interval_range_s=(N_s, N_s)
    where N_s is a multiple of num_steps_per_env × control_dt to align with rollout
    boundaries and keep physics constants stable within each training batch.
    """

    recompute = RecomputeLevel.set_const  # event_manager triggers set_const after __call__
    model_fields = ()  # no GPU fields written here

    def __init__(self, cfg, env):
        pass

    def __call__(self, env, env_ids, **kwargs):
        pass  # recompute handled by event_manager via .recompute attribute above


class randomize_effort_limits_with_delay:
    """Randomize effort limits, handling DelayedActuator."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        from mjlab.actuator import BuiltinPositionActuator, DelayedActuator, XmlPositionActuator
        from mjlab.envs.mdp.events import _sample_distribution

        self._sample = _sample_distribution
        self.model = env.sim.model
        self.device = env.device
        self.operation = cfg.params.get("operation", "scale")
        self.distribution = cfg.params.get("distribution", "uniform")

        effort_range = cfg.params["effort_limit_range"]
        self.lo = torch.tensor(effort_range[0], device=env.device)
        self.hi = torch.tensor(effort_range[1], device=env.device)

        # Collect ctrl_ids from valid actuators (unwrap DelayedActuator for type check)
        asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
        ctrl_ids = []
        for act in asset.actuators:
            base = act.base_actuator if isinstance(act, DelayedActuator) else act
            if isinstance(base, (BuiltinPositionActuator, XmlPositionActuator)):
                ctrl_ids.extend(act.ctrl_ids.tolist())
        self.ctrl_ids = torch.tensor(ctrl_ids, dtype=torch.long, device=env.device)

        # Store original force range to prevent unbounded drift with scale operation
        self.original_forcerange = self.model.actuator_forcerange[:, self.ctrl_ids].clone()

    def __call__(self, _env, env_ids, **_):
        samples = self._sample(self.distribution, self.lo, self.hi, (len(env_ids), len(self.ctrl_ids)), self.device)
        if self.operation == "scale":
            # Scale from original baseline to prevent unbounded drift
            # Index: [num_envs, num_ctrl_ids] -> broadcast multiply with samples [num_envs, num_ctrl_ids]
            self.model.actuator_forcerange[env_ids[:, None], self.ctrl_ids, 0] = \
                self.original_forcerange[env_ids, :, 0] * samples
            self.model.actuator_forcerange[env_ids[:, None], self.ctrl_ids, 1] = \
                self.original_forcerange[env_ids, :, 1] * samples
        else:
            self.model.actuator_forcerange[env_ids[:, None], self.ctrl_ids, 0] = -samples
            self.model.actuator_forcerange[env_ids[:, None], self.ctrl_ids, 1] = samples


def randomize_foot_solimp(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    d0_range: tuple[float, float] = (0.85, 0.99),
    width_range: tuple[float, float] = (0.002, 0.01),
):
    """Randomize foot solimp: axis 0 (d0, minimum impedance) and axis 2 (width, transition band).

    solimp controls the force–penetration curve shape; smaller d0 = softer contact onset.
    Axes 0 and 2 are non-contiguous so two writes are needed.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    geom_ids = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
    n, m = len(env_ids), len(geom_ids)
    e, g = env_ids[:, None], geom_ids[None, :]  # broadcast indices: no meshgrid allocation

    env.sim.model.geom_solimp[e, g, 0] = torch.empty(n, m, device=env.device).uniform_(*d0_range)
    env.sim.model.geom_solimp[e, g, 2] = torch.empty(n, m, device=env.device).uniform_(*width_range)

randomize_foot_solimp.model_fields = ("geom_solimp",)


def randomize_foot_solref(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    timeconst_range: tuple[float, float] = (0.01, 0.04),  # lower bound = 2×sim_dt for stability
    dampratio_range: tuple[float, float] = (0.8, 1.2),
):
    """Randomize foot solref: axis 0 (timeconst, s) and axis 1 (dampratio).

    solref controls the contact spring: smaller timeconst = stiffer/harder floor
    (0.01s tile → 0.04s carpet; MuJoCo default = 0.02s). Lower bound is 2×sim_dt (0.005s)
    — MuJoCo requires timeconst ≥ 2–3× timestep for stable contact integration.
    dampratio = 1.0 is critically damped; <1 bouncy, >1 overdamped (MuJoCo default = 1.0).
    Axes 0 and 1 are contiguous so both params are written in a single tensor assignment.
    Complements randomize_foot_solimp: solimp = force–penetration curve, solref = spring dynamics.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    geom_ids = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
    n, m = len(env_ids), len(geom_ids)
    e, g = env_ids[:, None], geom_ids[None, :]  # broadcast indices: no meshgrid allocation

    samples = torch.empty(n, m, 2, device=env.device)
    samples[..., 0].uniform_(*timeconst_range)
    samples[..., 1].uniform_(*dampratio_range)
    env.sim.model.geom_solref[e, g, :2] = samples  # single write: axes 0 and 1 are contiguous

randomize_foot_solref.model_fields = ("geom_solref",)


# =============================================================================
# Push Recording Events (used with push-aware reward gating)
# =============================================================================


class reset_last_push_step:
    """Event term: resets the push timer for environments being reset.

    Prevents new episodes from spawning with suppressed rewards if the
    previous episode in that environment ended shortly after a push.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        pass

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **_kwargs):
        if hasattr(env, "_last_push_step"):
            env._last_push_step[env_ids] = -1e6
        if hasattr(env, "_last_push_delta_vel"):
            env._last_push_delta_vel[env_ids] = 0.0


class push_and_record:
    """Event term: applies a push and records the step counter and delta velocity per env.

    Directly implements push_by_setting_velocity sampling to capture the exact
    6D delta velocity instead of computing before/after differences. Stores
    env._last_push_step and env._last_push_delta_vel (world-frame) so reward
    terms can compute time-since-push and magnitude for exponential suppression gating.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        env._last_push_step = torch.full((env.num_envs,), -1e6, device=env.device)
        env._last_push_delta_vel = torch.zeros((env.num_envs, 6), device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **kwargs):
        asset_cfg = kwargs.get("asset_cfg", _DEFAULT_ASSET_CFG)
        velocity_range = kwargs.get("velocity_range", {})

        asset = env.scene[asset_cfg.name]
        vel_w = asset.data.root_link_vel_w[env_ids]

        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=env.device)

        delta_vel = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)

        # Rotate linear push from base_link yaw frame to world frame.
        quat_yaw = yaw_quat(asset.data.root_link_quat_w[env_ids])
        delta_vel_w = torch.empty_like(delta_vel)
        delta_vel_w[:, :3] = quat_apply(quat_yaw, delta_vel[:, :3])
        delta_vel_w[:, 3:] = delta_vel[:, 3:]  # angular delta unchanged by yaw rotation

        asset.write_root_link_velocity_to_sim(vel_w + delta_vel_w, env_ids=env_ids)

        env._last_push_step[env_ids] = float(env.common_step_counter)
        env._last_push_delta_vel[env_ids] = delta_vel_w  # world-frame delta


# =============================================================================
# Event Parameter Curriculum
# =============================================================================


class EventParamCurriculum:
    """Curriculum for event parameters with linear interpolation.

    Smoothly interpolates event parameters between curriculum stages.
    Uses caching to avoid repeated lookups (similar to reward_weight_linear).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.decimation = cfg.params.get("decimation", 2400)  # default matches CURRICULUM_DECIMATION
        self._events = []  # List of (event_name, params, param_data)
        self._state: dict = {}  # cached logging state; rebuilt only when curriculum ticks

        for event_name, stages in cfg.params["events"].items():
            # Gracefully skip curriculum for events that are not active (e.g., disabled in play mode)
            is_active = False
            for term_names in env.event_manager.active_terms.values():
                if event_name in term_names:
                    is_active = True
                    break

            if not is_active:
                continue

            event_cfg = env.event_manager.get_term_cfg(event_name)
            params = event_cfg.params

            # Build per-parameter data: {name: (intervals, cache_idx, is_tuple, boundary_vals)}
            param_data = {}
            param_names = {k for stage in stages for k in stage if k != "step"}

            for param_name in param_names:
                # Build intervals with pre-computed slopes and type flags
                intervals = []
                for i in range(len(stages) - 1):
                    start_step, end_step = stages[i]["step"], stages[i + 1]["step"]
                    start_val, end_val = stages[i].get(param_name), stages[i + 1].get(param_name)

                    if start_val is not None and end_val is not None:
                        is_tuple = isinstance(start_val, tuple)
                        is_dict = isinstance(start_val, dict)
                        if is_dict:
                            # Per-axis dict: {axis_idx: (lo, hi), ...} — interpolate each axis independently
                            slope = {k: tuple((end_val[k][j] - start_val[k][j]) / (end_step - start_step)
                                             for j in range(len(start_val[k]))) for k in start_val}
                        elif is_tuple:
                            slope = tuple((end_val[j] - start_val[j]) / (end_step - start_step)
                                         for j in range(len(start_val)))
                        else:
                            slope = (end_val - start_val) / (end_step - start_step)
                        # Store: (start_step, end_step, start_val, slope, is_tuple, is_dict)
                        intervals.append((start_step, end_step, start_val, slope, is_tuple, is_dict))

                # Get boundary values (first and last stage values)
                first_val = next((s[param_name] for s in stages if param_name in s), None)
                last_val = next((s[param_name] for s in reversed(stages) if param_name in s), None)

                param_data[param_name] = {
                    "intervals": intervals,
                    "cache_idx": 0,  # Last used interval index
                    "boundary": (first_val, last_val),
                }

            self._events.append((event_name, params, param_data))

    @staticmethod
    def _interpolate(start_val, slope, delta, is_tuple, is_dict=False):
        """Fast interpolation helper (eliminates duplicate code)."""
        if is_dict:
            return {k: tuple(start_val[k][j] + delta * slope[k][j] for j in range(len(start_val[k])))
                    for k in start_val}
        if is_tuple:
            return tuple(start_val[j] + delta * slope[j] for j in range(len(start_val)))
        return start_val + delta * slope

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor, **_) -> dict:
        step = env.common_step_counter

        if step % self.decimation == 0:
            for event_name, params, param_data in self._events:
                for param_name, data in param_data.items():
                    intervals = data["intervals"]
                    if not intervals:
                        continue

                    # Try cached interval first
                    idx = data["cache_idx"]
                    if idx < len(intervals):
                        start_step, end_step, start_val, slope, is_tuple, is_dict = intervals[idx]
                        if start_step <= step < end_step:
                            # Cache hit - fast path
                            val = self._interpolate(start_val, slope, step - start_step, is_tuple, is_dict)
                            params[param_name] = val
                            continue

                    # Cache miss - search for current interval
                    for i, (start_step, end_step, start_val, slope, is_tuple, is_dict) in enumerate(intervals):
                        if start_step <= step < end_step:
                            data["cache_idx"] = i
                            val = self._interpolate(start_val, slope, step - start_step, is_tuple, is_dict)
                            params[param_name] = val
                            break
                    else:
                        # Outside all intervals - use boundary values
                        first_val, last_val = data["boundary"]
                        params[param_name] = first_val if step < intervals[0][0] else last_val

            # Rebuild logging state only when curriculum ticks (not every call)
            self._state = {}
            for event_name, params, _ in self._events:
                for key, value in params.items():
                    if isinstance(value, tuple) and len(value) == 2:
                        self._state[f"{event_name}_{key}_min"] = torch.tensor(value[0])
                        self._state[f"{event_name}_{key}_max"] = torch.tensor(value[1])

        return self._state


# =============================================================================
# End-Effector Target Resampling
# =============================================================================


class EETargetsResample:
    """Batched periodic event: resample env._ee_targets_h (B, K, 3) for K end-effectors.

    Single event replaces K separate EETargetResample events. Each EE gets an
    independent uniform-in-ball sample of radius `radius` around its FK default.
    Sampling is vectorized over both batch and K dimensions.

    Params:
        ee_body_names: List of K MuJoCo body names.
        radius:        Sphere radius in meters (default 0.05 = 5cm).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._radius = float(cfg.params.get("radius", 0.05))
        self._asset = env.scene["robot"]
        body_ids, _ = self._asset.find_bodies(cfg.params["ee_body_names"])
        self._body_ids = body_ids
        self._K = len(body_ids)
        self._device = env.device
        self._env = env
        if not hasattr(env, "_ee_targets_h"):
            env._ee_targets_h = torch.zeros(env.num_envs, self._K, 3, device=env.device)
        self._default_pos_h: torch.Tensor | None = None  # (B, K, 3), cached on first call

    def _compute_pos_h(self) -> torch.Tensor:
        from tasks.humanoid_velocity.observation import _ee_pos_in_h
        return _ee_pos_in_h(self._asset, self._body_ids, self._env)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # No-op: timer reset handled by event_manager. Must exist so the resample
        # timer resets to [2,4]s on episode reset instead of persisting across episodes.
        pass

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **_) -> None:
        if self._default_pos_h is None:
            self._default_pos_h = self._compute_pos_h().clone()  # (B, K, 3)

        target = env._ee_targets_h
        if env_ids is None:
            n, idx = target.shape[0], slice(None)
        else:
            n, idx = len(env_ids), env_ids

        # (n, K, 3): independent ball sample per env per EE
        direction = F.normalize(torch.randn(n, self._K, 3, device=self._device), dim=-1)
        r = self._radius * torch.rand(n, self._K, 1, device=self._device).pow(1.0 / 3.0)
        target[idx] = self._default_pos_h[idx] + direction * r


# =============================================================================
# EE Target Radius Curriculum
# =============================================================================


class ee_target_radius_linear:
    """Curriculum: linearly interpolate EETargetResample._radius between stages.

    Mirrors the reward_weight_linear pattern, but targets the sampling radius of
    the EETargetsResample event term instead of a reward weight. Enables the
    target sphere to grow over training (Phase 1: ±5cm → Phase 2: ±15cm).

    Registered as a CurriculumTermCfg with mode="step" alongside the reward
    curricula. Because EETargetsResample is a class-based event, its instance is
    stored in term_cfg.func after _prepare_terms — we grab it once at __init__
    via event_manager.get_term_cfg().

    Params (via CurriculumTermCfg.params):
        event_name:    Key in env.event_manager for the EETargetsResample term.
        radius_stages: List of {"step": int, "radius": float} dicts, ascending.
        decimation:    Update frequency (default 1 = every curriculum step).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        event_term_cfg = env.event_manager.get_term_cfg(cfg.params["event_name"])
        # After _prepare_terms, func is the live EETargetsResample instance.
        self._ee_event: EETargetsResample = event_term_cfg.func  # type: ignore[assignment]

        stages = cfg.params["radius_stages"]
        self.decimation = cfg.params.get("decimation", 1)
        self.steps   = tuple(s["step"]   for s in stages)
        self.radii   = tuple(s["radius"] for s in stages)
        self.slopes  = tuple(
            (self.radii[i + 1] - self.radii[i]) / (self.steps[i + 1] - self.steps[i])
            for i in range(len(stages) - 1)
        )
        self._last_idx = 0
        self._return_tensor = torch.zeros(1, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, _env_ids, **_kwargs) -> torch.Tensor:
        step = env.common_step_counter
        if step % self.decimation != 0:
            return self._return_tensor

        i = self._last_idx
        if i < len(self.slopes) and self.steps[i] <= step < self.steps[i + 1]:
            radius = self.radii[i] + (step - self.steps[i]) * self.slopes[i]
        else:
            radius = None
            for i in range(len(self.slopes)):
                if self.steps[i] <= step < self.steps[i + 1]:
                    self._last_idx = i
                    radius = self.radii[i] + (step - self.steps[i]) * self.slopes[i]
                    break
            if radius is None:
                radius = self.radii[-1] if step >= self.steps[-1] else self.radii[0]

        self._ee_event._radius = radius
        self._return_tensor[0] = radius
        return self._return_tensor


# =============================================================================
# Dead-Zone Termination
# =============================================================================


class VelocityTrackingFailure:
    """Terminate if robot stands still despite non-zero XY command for K consecutive steps.

    Turns dead-zone behavior into an absorbing punishment: episode ends when robot fails
    to produce locomotion for longer than the patience window. No gait schedule; no reward
    landscape modification.

    K=70 steps at 50Hz = 1.4s. Counter resets on episode reset (via reset()), on self-fire,
    and whenever active=False (robot moves OR cmd drops below threshold).

    Bug note: without reset(), counter persists across episode boundaries — fell_over at step
    5 leaves counter=5, which carries into the next episode and causes spurious early fires.
    TerminationManager.reset() calls func.reset(env_ids) only for functions that have reset().

    Threshold modes (mutually exclusive; tracking_ratio takes priority):
      tracking_ratio (preferred): threshold = tracking_ratio * |cmd_xy| per env.
        Principled — "robot must achieve X% of commanded velocity" — transfers across
        robot morphologies without recalibration. Fixes the bug in absolute mode where
        threshold can exceed cmd at the gate boundary (e.g., 0.13 m/s > 0.105 m/s cmd).
        Calibrated at 0.3: fires when robot achieves <30% of cmd, which is 36-56%
        degradation below typical training tracking (ratio ~0.65 from ep_track_lin_vel).
      base_vel_xy_threshold (legacy): fixed absolute threshold in m/s. Use only for
        backward compatibility with existing checkpoints trained on the old threshold.

    reorient_heading_threshold (rad, optional): exempt heading envs mid-pivot. A heading-
      controlled env with |heading_error| above this is turning to face its commanded
      direction, which legitimately stalls forward velocity — counting it as a dead-zone
      failure is a false positive (probed 2026-07-10: conservative policies fired here at
      |heading_error|≈1.7 rad while a genuine aligned dead zone sits at ≈0). None disables.

    vel_window_steps (W, optional): switch the sub-threshold test from INSTANTANEOUS to a
      trailing-window MEAN. None (default) = legacy consecutive-counter (fire after K
      consecutive instantaneous sub-threshold steps). When set, maintain a hard W-step ring
      buffer per env and fire when the WINDOW-MEAN body vel is below tracking_ratio × the
      window-mean cmd (both averaged over the same W steps), once the window is full.
      Rationale (audit 2026-07-10, F1). The consecutive-counter resets on ANY single above-
      threshold instant, so two policies with the SAME mean low-cmd tracking are scored
      oppositely by velocity NOISE: a steady stander accumulates K and fires; a shuffler
      crosses threshold every few steps, resets the counter, never fires. VTF thus punishes
      gait STEADINESS, not tracking failure (v83StudentOnly 91 fires vs v83L2T 1, despite
      equal/better held-command tracking). The windowed mean is noise-invariant: fire ⇔ the
      robot genuinely failed to move on AVERAGE over the window. Reuse W = old K (70 ≈ 1.4 s)
      to keep the patience-window semantics; the ONLY change is instantaneous → mean.
      Running-sum ring is O(1)/step; float32 sum may drift over very long episodes but stays
      O(1) magnitude — acceptable for a termination gate. reorient exemption still applies
      (pivoting heading envs are treated as tracking perfectly, freezing failure accrual).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.cmd_threshold = cfg.params.get("cmd_xy_threshold", 0.1)
        self.tracking_ratio = cfg.params.get("tracking_ratio", None)
        self.vel_threshold = cfg.params.get("base_vel_xy_threshold", 0.07)
        self.K = cfg.params.get("consecutive_steps", 70)
        # Reorientation exemption (rad). When set, heading-controlled envs whose |heading_error|
        # exceeds this are exempt from the active mask: a robot pivoting to face its commanded
        # direction legitimately suppresses forward velocity, and VTF targets the STANDING dead
        # zone (see class docstring), not turn-in-place. None = disabled (legacy behavior).
        # Probed 2026-07-10: this exempts only the heading-env pivot subset (~1/3 of one policy's
        # spurious fires); the majority were plain-env low-cmd steady stands, a separate issue.
        # Note: no yaw dead-zone TERMINATION is wired for the humanoid (YawTrackingFailure is unused
        # here) — only the heading REWARD pressures alignment, so this exemption does weaken the
        # dead-zone guard for mis-headed envs. Kept gated (default None) pending the wider VTF audit.
        self.reorient_heading_threshold = cfg.params.get("reorient_heading_threshold", None)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.command_name = cfg.params.get("command_name", "twist")
        self.counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        # Pre-allocated int32 buffer for in-place counter update — avoids zeros_like each step.
        self._active_int = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        # Windowed-mean mode (F1 fix). None = legacy instantaneous consecutive-counter.
        self.W = cfg.params.get("vel_window_steps", None)
        if self.W is not None:
            B, dev = env.num_envs, env.device
            self.vel_buf = torch.zeros(B, self.W, device=dev)  # trailing W samples of |vel_xy|
            self.cmd_buf = torch.zeros(B, self.W, device=dev)  # trailing W samples of |cmd_xy|
            self.vel_sum = torch.zeros(B, device=dev)          # running sum of vel_buf row
            self.cmd_sum = torch.zeros(B, device=dev)          # running sum of cmd_buf row
            self.fill = torch.zeros(B, dtype=torch.int32, device=dev)  # valid samples/env (caps W)
            self.ptr = 0                                        # shared ring index (per-env fill gates)
            # Firing only when fill==W lets the mean-tests reduce to sum-tests scaled by W (no
            # per-step divide/alloc): cmd_mean>thr ⇔ cmd_sum>thr·W; ratio test's denom cancels.
            self._cmd_thr_W = self.cmd_threshold * self.W
            self._vel_thr_W = self.vel_threshold * self.W       # only used in legacy-absolute mode

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.counter[env_ids] = 0
        if self.W is not None:
            # Clear the window for reset envs so a new episode starts with an empty (invalid)
            # ring; fill<W then suppresses firing until W fresh samples accumulate.
            self.vel_buf[env_ids] = 0.0
            self.cmd_buf[env_ids] = 0.0
            self.vel_sum[env_ids] = 0.0
            self.cmd_sum[env_ids] = 0.0
            self.fill[env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        cmd = env.command_manager.get_command(self.command_name)   # [B, 3]
        vel = self.asset.data.root_link_lin_vel_b                  # [B, 3]
        cmd_norm = torch.norm(cmd[:, :2], dim=1)
        vel_norm = torch.norm(vel[:, :2], dim=1)
        # Reorientation exemption (both modes): a heading env still pivoting to its target
        # legitimately suppresses forward velocity, so treat it as tracking (not a dead zone).
        reorienting = None
        if self.reorient_heading_threshold is not None:
            cmd_term = env.command_manager.get_term(self.command_name)
            if getattr(cmd_term.cfg, "heading_command", False):
                reorienting = cmd_term.is_heading_env & (
                    cmd_term.heading_error.abs() > self.reorient_heading_threshold
                )

        if self.W is not None:
            # Windowed-mean mode (F1 fix): fire on trailing-window MEAN, noise-invariant.
            # Pivoting heading envs contribute a perfect-tracking sample (vel = cmd) so they
            # never drag the window mean down — freezes failure accrual, resumes once aligned.
            if reorienting is not None:
                vel_norm = torch.where(reorienting, cmd_norm, vel_norm)
            p = self.ptr
            # Running sums, all in-place (read old col BEFORE overwrite): zero per-step alloc.
            self.vel_sum.sub_(self.vel_buf[:, p]).add_(vel_norm)
            self.cmd_sum.sub_(self.cmd_buf[:, p]).add_(cmd_norm)
            self.vel_buf[:, p] = vel_norm
            self.cmd_buf[:, p] = cmd_norm
            self.ptr = (p + 1) % self.W
            self.fill.add_(1).clamp_(max=self.W)
            # Fire only once the window is full (fill==W); the mean-tests then reduce to
            # sum-tests scaled by W — no divide, no denom alloc (see __init__).
            #   vel_mean < ratio·cmd_mean  ⇔  vel_sum < ratio·cmd_sum   (denom cancels)
            #   cmd_mean > cmd_threshold   ⇔  cmd_sum > cmd_threshold·W
            if self.tracking_ratio is not None:
                vel_fail = self.vel_sum < self.tracking_ratio * self.cmd_sum
            else:
                vel_fail = self.vel_sum < self._vel_thr_W
            return (self.fill >= self.W) & (self.cmd_sum > self._cmd_thr_W) & vel_fail

        # Legacy mode: K consecutive instantaneous sub-threshold steps.
        if self.tracking_ratio is not None:
            threshold = self.tracking_ratio * cmd_norm             # per-env, scales with cmd
        else:
            threshold = self.vel_threshold                         # scalar, absolute (legacy)
        active = (cmd_norm > self.cmd_threshold) & (vel_norm < threshold)
        if reorienting is not None:
            # Pause the dead-zone clock while a heading env is still turning to its target.
            active = active & ~reorienting
        # In-place update: counter = active * (counter + 1), zero allocation.
        # add_ first (increment), then mul_ (zero non-active): active*(counter+1)
        self._active_int.copy_(active)
        self.counter.add_(self._active_int)
        self.counter.mul_(self._active_int)
        terminated = self.counter >= self.K
        self.counter[terminated] = 0
        return terminated


class YawTrackingFailure:
    """Terminate if robot fails to turn despite non-zero yaw command for K consecutive steps.

    Yaw analogue of VelocityTrackingFailure. Fixes yaw dead zone via absorbing punishment
    rather than reward shaping (which causes kinematic conflict: tight yaw reward forces
    asymmetric leg corrections → body roll/pitch → torque increase, as seen in v33).

    Signed comparison: checks that yaw velocity is aligned with commanded direction AND
    exceeds tracking_ratio × |cmd_yaw|. Pure-magnitude check (like XY norm) would pass a
    robot spinning in the wrong direction at high speed.

    active condition: |cmd_yaw| > cmd_yaw_threshold
                      AND sign(cmd_yaw) × yaw_vel < tracking_ratio × |cmd_yaw|

    K=100 steps at 50Hz = 2s. Longer than XY VTF (70 steps) to account for early training
    where robot walks but has not yet learned to turn — gives 2s to initiate rotation before
    termination. Counter self-regulates: short early-training episodes (fell_over) prevent
    VTF from firing until ep_length exceeds K.

    Calibration (v30 eval, 2026-05-26): v30 achieves 7–12% yaw tracking during walking
    → tracking_ratio=0.3 fires consistently for dead-zone behavior. cmd_yaw_threshold=0.15
    avoids spurious fires at near-zero commands.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.cmd_threshold = cfg.params.get("cmd_yaw_threshold", 0.15)
        self.tracking_ratio = cfg.params.get("tracking_ratio", 0.3)
        self.K = cfg.params.get("consecutive_steps", 100)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.command_name = cfg.params.get("command_name", "twist")
        self.counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.counter[env_ids] = 0

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        command_velocity = env.command_manager.get_command(self.command_name)   # [B, 3]
        command_yaw = command_velocity[:, 2]                                   # signed, rad/s
        measured_yaw_velocity = self.asset.data.root_link_ang_vel_b[:, 2]       # signed, rad/s

        command_yaw_abs = command_yaw.abs()
        # Project yaw velocity onto commanded direction: positive = turning with cmd.
        # Mathematically, sign(command_yaw) * measured_yaw_velocity < tracking_ratio * |command_yaw|
        # is equivalent to command_yaw * measured_yaw_velocity < tracking_ratio * command_yaw.square()
        # when |command_yaw| > cmd_threshold > 0, which avoids slow sign() and extra abs() calls.
        tracking_failed = (command_yaw * measured_yaw_velocity) < (self.tracking_ratio * command_yaw.square())
        failure_active = (command_yaw_abs > self.cmd_threshold) & tracking_failed

        # Apply boolean active mask directly in-place to avoid copy/conversion overhead to a buffer.
        self.counter.add_(failure_active)
        self.counter.mul_(failure_active)
        
        failure_terminated = self.counter >= self.K
        self.counter[failure_terminated] = 0
        return failure_terminated
