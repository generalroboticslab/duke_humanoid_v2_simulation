"""A1 camera-task obs/reward terms (active vision, plan/valiant-giggling-hoare.md).

HIGH-LEVEL gaze policy on frozen v83: each opposed camera independently tracks its OWN
known world point (left = front hemisphere, right = rear hemisphere; fixed pairing). These
mjlab obs/reward TERMS read the per-env targets held on the env as ``env.cam_target_w``
(shape (N, 2, 3), world frame; written by CameraLearnerEnv._reset_idx) and the live camera
site poses, and expose:

  obs   camera_target_pos_base : (N, 6)  base-frame POSITIONS (range-carrying) of [target0, target1]
        camera_joint_pos       : (N, 4)  the 4 gimbal joint angles
        camera_joint_vel       : (N, 4)  the 4 gimbal joint velocities
  reward camera_aim_reward     : (N,)    aim_L(→target0) + aim_R(→target1)

Geometry primitives reused from ``tasks.camera_perception`` (fov_detect); the +Z-out optical
convention + FOV design match the A0 viz scripts. Term classes follow the established
(cfg, env)->__call__ pattern (see humanoid_velocity/reward.py:arm_joint_tracking).

Design notes:
  - Target obs are base-frame POSITIONS (not unit dirs): a unit dir from the BASE origin is
    ambiguous for the CAMERA, which sits ~0.5-0.7 m above the base — the required aim depends on
    the range along the base-ray (parallax). Position carries range so the camera (fixed head
    offset) can resolve its own aim. Both targets given symmetrically (order = label) so the SAME
    obs supports A3 LEARNED camera↔target assignment (only the reward's fixed pairing is A1-scoped).
  - Left site ↔ target0, right site ↔ target1 is a FIXED assignment in the REWARD only (A1 gate);
    learned/assignment-free reward (max-over-cameras) = A3. The obs does not bake the pairing.
  - ``env.cam_target_w`` is allocated lazily by the target-pos obs term so the obs manager can build
    before CameraLearnerEnv finishes __init__ (obs-term init runs before the first obs compute);
    same setattr-on-env pattern as humanoid_velocity/observation.py:joint_target.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from tasks.camera_perception import fov_detect, BeliefBuffer

# Gimbal actuator name pattern (matches experiments.py _CAMERA_PATTERN) and RGB optical sites.
_CAMERA_PATTERN = r"cam_(yaw|pitch)_(left|right)"
_CAM_SITES = ("cam_left_rgb", "cam_right_rgb")  # left↔target0 (front), right↔target1 (rear)

# FOV design (plan A0 / mj_envs/probe/viz_camera_fov*.py): H 90deg, V 65deg, near 0.1 m, far 5.0 m.
# Far relaxed 3.0 -> 5.0 (detection horizon; ~3 m is where stereo depth is optimal, 5 m is the
# usable-detection edge) so the walk-search DISCOVER/APPROACH range gate sees far cubes and drives
# them into reach. GLOBAL: also widens the gaze-policy training reward's in_fov framing envelope.
H_HALF, V_HALF, NEAR, FAR = 0.7854, 0.5672, 0.2, 5.0

# The baked FOV frustum overlay carries its own copy of these four numbers: asset
# construction must not import task code (asset_zoo would gain a torch/mjlab dependency),
# so the values cannot simply be shared. Pin them here instead -- the overlay is a sensor
# audit aid and must never claim a range the detector does not have, and a silent drift
# between the drawn cone and this gate is exactly the failure it would hide.
def _assert_fov_overlay_matches() -> None:
    from asset_zoo import fov_frustum as _f
    drawn = (_f.FOV_H_HALF_RAD, _f.FOV_V_HALF_RAD, _f.FOV_NEAR_M, _f.FOV_FAR_M)
    if drawn != (H_HALF, V_HALF, NEAR, FAR):
        raise AssertionError(
            f"FOV overlay envelope {drawn} != detector gate "
            f"{(H_HALF, V_HALF, NEAR, FAR)}; update asset_zoo/fov_frustum.py"
        )


_assert_fov_overlay_matches()

_N_TARGETS = 2


def _ensure_cam_target_w(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Lazily allocate the (N, 2, 3) world-frame target buffer on the env (shared by all terms)."""
    if not hasattr(env, "cam_target_w"):
        env.cam_target_w = torch.zeros(env.num_envs, _N_TARGETS, 3, device=env.device)
    return env.cam_target_w


def _ensure_belief(env: ManagerBasedRlEnv) -> BeliefBuffer:
    """Lazily allocate the A5 POMDP belief on the env (shared by the obs term and the env's
    ``_update_belief``). Same setattr-on-env / lazy-build pattern as ``_ensure_cam_target_w``: the
    belief obs term __init__ runs during obs-manager build (inside CameraLearnerEnv super().__init__),
    BEFORE the env finishes wiring, so the term — not the env — owns first allocation. The env's
    ``_update_belief`` then reads the SAME object via ``env.belief``. Decay read from
    ``env._belief_decay`` (set pre-super by CameraLearnerEnv) else 0.95."""
    if not hasattr(env, "belief"):
        decay = float(getattr(env, "_belief_decay", 0.95))
        env.belief = BeliefBuffer(env.num_envs, _N_TARGETS, decay, env.device)
    return env.belief


class camera_target_pos_base:
    """Obs: base-frame target POSITIONS (range-carrying) to both targets -> (N, 6).

    POSITION, not unit direction. A unit dir from the base origin is AMBIGUOUS for the
    camera: the camera site sits ~0.5-0.7 m above the base (root_link), so a base-origin
    ray under-determines where the actual target point is — required camera aim depends on
    the range along that ray (parallax up to ~1 rad at near targets). The full base-frame
    position (target_w - root_pos, rotated into base axes, NOT normalized) carries range, so
    the camera (fixed offset on the head) can resolve its own aim. Frame = base (not camera):
    frame-stable, does not move with the action; the constant neck offset is learnable.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        _ensure_cam_target_w(env)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        root_pos = self.asset.data.root_link_pos_w                       # (N, 3)
        root_mat = matrix_from_quat(self.asset.data.root_link_quat_w)    # (N, 3, 3)
        rel_world = env.cam_target_w - root_pos.unsqueeze(1)             # (N, 2, 3) base->target, world
        pos_base = torch.matmul(
            root_mat.transpose(-1, -2).unsqueeze(1), rel_world.unsqueeze(-1)
        ).squeeze(-1)                                                    # (N, 2, 3) in base frame
        return pos_base.reshape(env.num_envs, _N_TARGETS * 3)           # (N, 6) [posL, posR]


class camera_belief_obs:
    """A5 obs: POMDP belief over both targets, base-frame -> (N, 12). ACTOR-ONLY.

    Replaces ``camera_target_pos_base`` (GT, 6D) in the A5 ACTOR group; the privileged critic keeps
    the GT term (asymmetric actor-critic, plan A5 §2.3). Reads the env-side ``BeliefBuffer`` (world
    store) and projects to the CURRENT base frame each compute (``expose_base``), so a target unseen
    since detection stays world-fixed in the estimate while its base-frame bearing/range update as
    the base moves. A never-detected target leaks NO position (confidence=0, seen_ever=0, est=0).

    Layout (12) = dir_base[6] (unit base->est, 2 targets) ++ dist[2] ++ confidence[2] ++ seen_ever[2].
    ``seen_ever`` is bool in the buffer; cast to float here (the policy buffer is float32 — a raw bool
    concat would up/down-cast unexpectedly). Order is fixed; the smoke test pins these slices.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        _ensure_belief(env)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        root_pos = self.asset.data.root_link_pos_w                       # (N, 3)
        root_mat = matrix_from_quat(self.asset.data.root_link_quat_w)    # (N, 3, 3)
        b = env.belief.expose_base(root_pos, root_mat)
        return torch.cat([
            b["dir_base"].reshape(env.num_envs, _N_TARGETS * 3),         # (N, 6)
            b["dist"],                                                   # (N, 2)
            b["confidence"],                                             # (N, 2)
            b["seen_ever"].float(),                                      # (N, 2)
        ], dim=-1)                                                       # (N, 12)


class camera_belief_uncertainty:
    """A5 search-shaping reward (penalty): total belief uncertainty over both targets -> (N,).

    Returns ``Σ_k (1 − confidence_k)`` ∈ [0, 2]; registered with a NEGATIVE weight so every step a
    target is unknown/stale bleeds reward. This is the SOTA active-perception "locate" signal (dense
    info-gain / belief-uncertainty, cf. AAWR arXiv:2512.01188, voxel-discovery arXiv:2602.01266) in
    belief form — the lever that breaks the IDLE local-optimum the pure coverage reward could not:
    coverage is satisfied by holding ONE visible target, but this penalty charges the UNSEEN target
    every step and holding the seen one cannot reduce it, so Q(search) > Q(hold-one) once weighted.

    HACK-PROOF: ``confidence`` rises to 1 ONLY on a real GT-gated detection (in_fov ∧ unoccluded ∧
    sharp, ``CameraLearnerEnv._update_belief``); else decays ×decay; reset to 0 on episode reset and
    on target resample. There is no path to inflate confidence without genuinely framing the true
    target, and the penalty rewards a STATE (monotone in fresh-time) so look-away/look-back cycles are
    strictly worse — no farmable transition (unlike a re-acquisition event bonus, deliberately NOT used).

    Reads the already-resident ``env.belief.confidence`` (N, K) — no new sensor/compute/state. Shares
    the SAME BeliefBuffer the A5 obs term + env ``_update_belief`` use (``_ensure_belief``).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        _ensure_belief(env)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return (1.0 - env.belief.confidence).sum(dim=1)                   # (N,), range [0, 2]


class camera_joint_pos:
    """Obs: the 4 gimbal joint angles -> (N, 4)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        joint_ids, _ = self.asset.find_joints(_CAMERA_PATTERN)
        self.joint_ids = joint_ids

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self.asset.data.joint_pos[:, self.joint_ids]


class camera_joint_vel:
    """Obs: the 4 gimbal joint velocities -> (N, 4)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        joint_ids, _ = self.asset.find_joints(_CAMERA_PATTERN)
        self.joint_ids = joint_ids

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self.asset.data.joint_vel[:, self.joint_ids]


class camera_aim_reward:
    """Reward: each camera vs its OWN target (fixed pairing), summed -> (N,).

    aim_k = exp(-aim_error_k^2 / std^2) + w_fov * in_fov_k, where aim_error / in_fov come from
    fov_detect on camera site k vs env.cam_target_w[:, k]. Dense aim term pulls each gimbal onto
    its target; in_fov bonus rewards actually framing it. Left site -> target0, right -> target1.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        std = float(cfg.params.get("std", 0.3))
        self._inv_std_sq = 1.0 / std ** 2
        self._w_fov = float(cfg.params.get("w_fov", 1.0))
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        # Site indices into site_pose_w's num_sites dim (find_sites resolves into site_names
        # order = the site_pose_w indexing order).
        self._site_ids = []
        for name in _CAM_SITES:
            ids, _names = self.asset.find_sites(name)
            assert len(ids) == 1, f"camera site {name!r} resolved to {len(ids)} sites: {_names}"
            self._site_ids.append(ids[0])

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        pose = self.asset.data.site_pose_w                               # (N, num_sites, 7)
        reward = torch.zeros(env.num_envs, device=env.device)
        for k, sid in enumerate(self._site_ids):
            cam_pos = pose[:, sid, :3]                                   # (N, 3)
            cam_mat = matrix_from_quat(pose[:, sid, 3:7])                # (N, 3, 3)
            det = fov_detect(cam_pos, cam_mat, env.cam_target_w[:, k],
                             H_HALF, V_HALF, NEAR, FAR)
            reward = reward + torch.exp(-det["aim_error"] ** 2 * self._inv_std_sq)
            reward = reward + self._w_fov * det["in_fov"].float()
        return reward


class camera_aim_component:
    """DIAGNOSTIC reward = ONE component of camera_aim, for per-term logging -> (N,).

    Splits camera_aim into its 4 additive pieces so the trainer's Episode_Reward log shows
    the breakdown (aiming vs framing, per camera) without changing the objective: the 4 terms
    (camera k in {0,1}) x (kind in {'dense','fov'}) sum bit-identically to camera_aim_reward.
    Lets a single training run reveal whether 2.0/4.0 is an AIM failure (dense low) or a FRAME
    failure (fov low), and whether one camera is dead (its pair both low). Params:
      camera: 0 (left->target0/front) or 1 (right->target1/rear)
      kind:   'dense' = exp(-aim_error^2/std^2);  'fov' = w_fov * in_fov
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._k = int(cfg.params["camera"])
        self._kind = str(cfg.params["kind"])
        assert self._kind in ("dense", "fov"), self._kind
        std = float(cfg.params.get("std", 0.3))
        self._inv_std_sq = 1.0 / std ** 2
        self._w_fov = float(cfg.params.get("w_fov", 1.0))
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        ids, _names = self.asset.find_sites(_CAM_SITES[self._k])
        assert len(ids) == 1, f"camera site {_CAM_SITES[self._k]!r} -> {len(ids)} sites"
        self._sid = ids[0]

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        pose = self.asset.data.site_pose_w[:, self._sid]                 # (N, 7)
        det = fov_detect(pose[:, :3], matrix_from_quat(pose[:, 3:7]),
                         env.cam_target_w[:, self._k], H_HALF, V_HALF, NEAR, FAR)
        if self._kind == "dense":
            return torch.exp(-det["aim_error"] ** 2 * self._inv_std_sq)
        return self._w_fov * det["in_fov"].float()


class camera_target_coverage:
    """A3 reward = coverage of ONE target by the BEST camera (max over cameras) -> (N,).

    ASSIGNMENT-FREE. For target k, reward = max_c [exp(-aim_err_ck^2/std^2) + w_fov*in_fov_ck] over
    the 2 cameras c. Summed over the 2 targets (= 2 logged terms, one per target) gives the A3
    objective: every target is rewarded by whichever camera frames it best, so the cameras
    SELF-ASSIGN (each target pulls its best-aligned camera; the other is free for the other target).
    Contrast A1/A2 `camera_aim_component`: there target k was HARD-PAIRED to camera k (left->0,
    right->1); here either camera may cover either target -> the assignment is LEARNED. Requires
    full-sphere targets (CameraLearnerEnv target_full_sphere=True) to be non-trivial: with the A1/A2
    hemisphere split, max would just recover the fixed pairing. Params:
      target: 0 or 1 (which world target this term scores)
      std, w_fov: same shaping as camera_aim_component (std 0.3, w_fov 1.0).
      std_narrow, w_track (optional): fine-track tier — a second, narrow aim Gaussian
        (exp(-aim_err^2/std_narrow^2)) weighted by w_track, stacked on the frame term so the policy
        is pulled to CENTER a framed target, not merely keep it in-frustum. std_narrow None (default)
        -> tier absent -> byte-stable. Occlusion-gated like the rest (multiplied by line-of-sight).
    Logging both target terms reveals the split: both high = both targets covered (assignment works);
    one low = a target left uncovered (assignment failed / cameras collide on one target).

    A4 OCCLUSION (auto, no param): if the env exposes ``cam_unoccluded`` (N, ncam, ntgt) bool
    (CameraLearnerEnv with occlusion=True), each camera's aim is MULTIPLIED by its line-of-sight to
    this target, so a camera whose view is blocked by v83's arm/torso scores 0 and the max picks
    the UNOCCLUDED camera -> occlusion forces reassignment. Absent the buffer (A1/A2/A3) the gate is
    a no-op -> byte-stable. Cam order in ``cam_unoccluded`` matches ``_CAM_SITES`` (left, right).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._k = int(cfg.params["target"])
        std = float(cfg.params.get("std", 0.3))
        self._inv_std_sq = 1.0 / std ** 2
        self._w_fov = float(cfg.params.get("w_fov", 1.0))
        # FINE-TRACK tier (optional): a second, NARROW aim Gaussian stacked on the frame term, so
        # once a target is framed the policy is pulled to CENTER it (camera-tracking) -- the honest
        # analog of "AprilTag decodes only a well-centered tag". std_narrow None => term absent =>
        # byte-stable with A1-A4 / the non-track A5 lineage. std_narrow ~0.1 rad (~5.7 deg cone),
        # w_track its weight. A SMOOTH narrow Gaussian (not a hard `aim_err < theta` indicator) keeps
        # a dense gradient toward center; an indicator is flat outside the cone -> the policy cannot
        # climb in.
        std_narrow = cfg.params.get("std_narrow", None)
        self._inv_std_narrow_sq = (1.0 / float(std_narrow) ** 2) if std_narrow is not None else None
        self._w_track = float(cfg.params.get("w_track", 0.0))
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self._sids = []
        for name in _CAM_SITES:
            ids, _names = self.asset.find_sites(name)
            assert len(ids) == 1, f"camera site {name!r} -> {len(ids)} sites"
            self._sids.append(ids[0])

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        tgt = env.cam_target_w[:, self._k]                               # (N, 3)
        unocc = getattr(env, "cam_unoccluded", None)                     # (N, ncam, ntgt) or None
        aims = []
        for ci, sid in enumerate(self._sids):
            pose = self.asset.data.site_pose_w[:, sid]                   # (N, 7)
            det = fov_detect(pose[:, :3], matrix_from_quat(pose[:, 3:7]),
                             tgt, H_HALF, V_HALF, NEAR, FAR)
            aim = (torch.exp(-det["aim_error"] ** 2 * self._inv_std_sq)
                   + self._w_fov * det["in_fov"].float())
            if self._inv_std_narrow_sq is not None:
                aim = aim + self._w_track * torch.exp(-det["aim_error"] ** 2 * self._inv_std_narrow_sq)
            if unocc is not None:
                aim = aim * unocc[:, ci, self._k].float()
            aims.append(aim)
        return torch.stack(aims, dim=-1).max(dim=-1).values             # (N,) best camera for target k
