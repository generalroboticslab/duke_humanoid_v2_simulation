"""Reward term helpers: actuator/foot dynamics, potential-based shaping, push-aware tracking."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.lab_api.math import quat_apply_inverse
from asset_zoo.humanoid_v21.generate_leg_joint_pos import (
    LEG_JOINT_NAMES,
    vertical_translate_lower_body_joints_batch,
)

from utils.proximity_model import (
    ARM_BODY_NAMES,
    PELVIS_BODY_NAME,
    PROX_A, DIST_A, PROX_B, DIST_B, RADII,
    compute_bimanual_distances,
    reward_arm_proximity,
)


# =============================================================================
# Reward Weight Curriculum
# =============================================================================


class reward_weight_linear:
    """Piecewise-linear interpolation of a reward weight over curriculum stages.

    Params (cfg.params):
        reward_name:  Single reward key (str). Mutually exclusive with reward_names.
        reward_names: List of reward keys sharing one weight schedule (list[str]).
                      Use when multiple rewards should be ramped in lockstep.
        weight_stages: List of {step, weight} dicts defining the ramp. Weight is held
                       constant before the first stage and after the last stage.
        decimation:   Control steps between updates (default: 480 = 20 iters).
                      Weight changes on 9.6s-of-iter timescales; finer granularity
                      is wasted recomputation.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        names = cfg.params.get("reward_names") or [cfg.params["reward_name"]]
        self.reward_term_cfgs = [env.reward_manager.get_term_cfg(n) for n in names]
        stages = cfg.params["weight_stages"]
        self.decimation = cfg.params.get("decimation", 480)

        self.steps = tuple(s["step"] for s in stages)
        self.weights = tuple(s["weight"] for s in stages)
        self.slopes = tuple(
            (self.weights[i + 1] - self.weights[i]) / (self.steps[i + 1] - self.steps[i])
            for i in range(len(stages) - 1)
        )

        self._last_idx = 0
        self._cached_weight = self.weights[0]
        self._return_tensor = torch.zeros(1, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, _env_ids, **_kwargs) -> torch.Tensor:
        step = env.common_step_counter
        if step % self.decimation == 0:
            # Fast path: cached interval is still valid
            i = self._last_idx
            if i < len(self.slopes) and self.steps[i] <= step < self.steps[i + 1]:
                self._cached_weight = self.weights[i] + (step - self.steps[i]) * self.slopes[i]
            else:
                # Slow path: search for current interval (runs on stage transitions only)
                weight = None
                for i in range(len(self.slopes)):
                    if self.steps[i] <= step < self.steps[i + 1]:
                        self._last_idx = i
                        weight = self.weights[i] + (step - self.steps[i]) * self.slopes[i]
                        break
                if weight is None:  # before first or after last stage
                    weight = self.weights[-1] if step >= self.steps[-1] else self.weights[0]
                self._cached_weight = weight

            for term_cfg in self.reward_term_cfgs:
                term_cfg.weight = self._cached_weight
            self._return_tensor[0] = self._cached_weight

        return self._return_tensor


# =============================================================================
# Actuator / Energy Rewards
# =============================================================================


@torch.jit.script
def _sum_square_excess(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """Sum of squared excess above threshold: sum(max(|x| - threshold, 0)^2) per batch."""
    return (x.abs() - threshold).relu().square().sum(dim=1)


@torch.jit.script
def _sum_abs_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sum of absolute element-wise product: sum(|a * b|) per batch."""
    return (a * b).abs().sum(dim=1)


class actuator_force_reward:
    """L2 penalty on actuator forces, only above threshold_ratio of limit."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.data = env.scene[asset_cfg.name].data
        ctrl_ids = self.data.indexing.ctrl_ids

        # threshold=0 means penalize all torque; threshold>0 means only penalize excess
        threshold_ratio = cfg.params.get("threshold_ratio", 0.0)
        if threshold_ratio > 0:
            force_limit = env.sim.model.actuator_forcerange[:, ctrl_ids, 1]
            # threshold is fixed at init from original force limits; does not track DR changes (intentional)
            self.threshold = threshold_ratio * force_limit
        else:
            self.threshold = torch.zeros(env.num_envs, len(ctrl_ids), device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return _sum_square_excess(self.data.actuator_force, self.threshold)


class joint_power_reward:
    """Penalize mechanical power: sum(|joint_vel * joint_torque|)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.data = env.scene[asset_cfg.name].data
        idx = self.data.indexing

        # Build joint→actuator mapping (handles robots where not all joints are actuated)
        # For each actuator, get which joint it drives
        joint_driven_by_act = env.sim.model.actuator_trnid[idx.ctrl_ids, 0]
        # match[j, a] = True if joint j is driven by actuator a
        match = (idx.joint_ids.unsqueeze(1) == joint_driven_by_act.unsqueeze(0))
        # Indices into joint_vel for actuated joints
        self.joint_idx = match.any(dim=1).nonzero(as_tuple=True)[0]
        # Corresponding indices into actuator_force
        self.act_idx = match.int().argmax(dim=1)[self.joint_idx]

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return _sum_abs_product(
            self.data.joint_vel[:, self.joint_idx],
            self.data.actuator_force[:, self.act_idx],
        )


class arm_action_l2:
    """L2 penalty on arm policy actions to minimize deviation from the random arm target.

    Encourages the locomotion policy to output near-zero arm actions, keeping arms
    at the externally-driven (random/deployment) pose. Caches the action term at init
    to avoid dict lookup overhead each step.

    Args (via cfg.params):
        arm_action_name: Key of the arm action group in env.action_manager (default: "joint_pos_arms").
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        arm_action_name = cfg.params.get("arm_action_name", "joint_pos_arms")
        self._term = env.action_manager._terms[arm_action_name]

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return torch.sum(self._term.raw_action.square(), dim=-1)


class arm_action_rate_l2:
    """L2 penalty on consecutive arm action differences (smoothness).

    r = sum((a_t - a_{t-1})²) for arm joints only.
    Penalizes jerky arm commands; encourages smooth transitions when the arm target changes.

    At episode start (episode_length_buf == 1), prev_action is set to the current action
    so the first-step penalty is always zero.

    Design: when the task has a dedicated arm action group (e.g. "joint_pos_arms" in
    humanoid_legs_only), set arm_action_name and leave arm_joint_pattern unset — the
    entire raw_action is already arm-only. When all joints share one group ("joint_pos"
    in humanoid_velocity), set arm_joint_pattern to a regex that selects arm joints by
    name within _target_names; only those columns of raw_action are penalized.

    Args (via cfg.params):
        arm_action_name:    Key of the action group (default: "joint_pos_arms").
        arm_joint_pattern:  Optional regex to select arm-joint columns within the action
                            vector by name (matched against _target_names). If None, all
                            columns in the action group are used.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        import re as _re
        action_name = cfg.params.get("arm_action_name", "joint_pos_arms")
        arm_pattern = cfg.params.get("arm_joint_pattern", None)
        self._term = env.action_manager._terms[action_name]

        if arm_pattern is not None:
            target_names = list(self._term._target_names)
            arm_ids = [i for i, name in enumerate(target_names) if _re.search(arm_pattern, name)]
            if not arm_ids:
                raise ValueError(
                    f"arm_joint_pattern={arm_pattern!r} matched no joints in action term "
                    f"{action_name!r}. Available names: {target_names}"
                )
            self._arm_ids: torch.Tensor | None = torch.tensor(
                arm_ids, dtype=torch.long, device=env.device
            )
        else:
            self._arm_ids = None  # use all dims in the action group

        self._prev_action: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        raw = self._term.raw_action  # (B, D_all_or_arm)
        current = raw[:, self._arm_ids] if self._arm_ids is not None else raw

        if self._prev_action is None:
            self._prev_action = torch.zeros_like(current)

        # Zero penalty on first step of each episode so the initial pose transition is free.
        # No if-guard: masked scatter is a no-op when mask is all-False; guard forces GPU→CPU sync.
        reset_mask = env.episode_length_buf == 1  # True for first step after reset
        self._prev_action[reset_mask] = current[reset_mask]

        diff = current - self._prev_action
        self._prev_action.copy_(current)
        return diff.square().sum(dim=-1)  # (B,)


# =============================================================================
# Foot Dynamics Rewards
# =============================================================================


@torch.jit.script
def _compute_foot_impact(
    foot_vel_z: torch.Tensor,    # [B, N]  world-frame foot z-velocity (already sliced)
    in_contact: torch.Tensor,    # [B, N] float
    threshold_vel_sq: float,     # downward velocity threshold squared (m²/s²)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Threshold-based foot impact penalty on downward velocity only.

    Penalizes clamp(v_z_down² - threshold², 0) * in_contact.
    v_z_down = max(0, -v_z): positive when foot moving downward.
    Free zone below threshold matches human walking (0.1–0.3 m/s downward at contact).
    Forward speed completely ignored — no hesitance risk.
    foot_slip already covers horizontal sliding during stance.

    Returns: (penalty [B], mean_downward_speed scalar for logging)
    """
    v_z_down = (-foot_vel_z).clamp(min=0.0)                             # [B, N] downward speed
    excess_sq = (v_z_down * v_z_down - threshold_vel_sq).clamp(min=0.0) # [B, N]
    penalty = (excess_sq * in_contact).sum(dim=1)                        # [B]
    n_contact = in_contact.sum().clamp(min=1.0)
    mean_vel = (v_z_down * in_contact).sum() / n_contact                 # scalar
    return penalty, mean_vel


class foot_impact_velocity:
    """Penalize downward foot velocity above threshold at ground contact.

    Uses threshold-based penalty: clamp(v_z_down² - threshold², 0) * in_contact.
    This creates a free zone for gentle landings (human walking: 0.1–0.3 m/s downward),
    so the weight can be made much stronger without causing hesitant walking.
    Forward speed is ignored — foot_slip already handles horizontal sliding.

    Ablation A5 (2026-05-26): confirmed redundant for v29 tracking objective. Safe to
    remove from future baselines.

    Implements log_metrics(env) so Metrics/foot_impact_velocity_mean is logged even
    when weight=0 (RewardManager protocol).

    Params:
        sensor_name: contact sensor name
        asset_cfg: foot site config
        threshold_vel: downward velocity free zone in m/s (default 0.3)
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.site_ids = asset_cfg.site_ids   # list[int] | slice
        self.sensor = env.scene[cfg.params["sensor_name"]]
        threshold_vel = cfg.params.get("threshold_vel", 0.3)  # m/s downward
        self.threshold_vel_sq = threshold_vel ** 2

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        in_contact = (self.sensor.data.found > 0).float()
        # Slice z-component before JIT dispatch: [B,N] vs [B,N,3] reduces kernel data by 3×
        foot_vel_z = self.asset.data.site_lin_vel_w[:, self.site_ids, 2]  # [B, N]
        penalty, mean_vel = _compute_foot_impact(foot_vel_z, in_contact, self.threshold_vel_sq)
        env.extras["log"]["Metrics/foot_impact_velocity_mean"] = mean_vel
        return penalty

    def log_metrics(self, env: ManagerBasedRlEnv) -> None:
        in_contact = (self.sensor.data.found > 0).float()
        foot_vel_z = self.asset.data.site_lin_vel_w[:, self.site_ids, 2]
        _, mean_vel = _compute_foot_impact(foot_vel_z, in_contact, self.threshold_vel_sq)
        env.extras["log"]["Metrics/foot_impact_velocity_mean"] = mean_vel


@torch.jit.script
def _compute_foot_balance(
    cct: torch.Tensor,            # [B, 2] current_contact_time — live sensor buffer view
    last_air_time: torch.Tensor,  # [B, 2] — live sensor buffer view
    ang_vel_w: torch.Tensor,      # [B, 3] root angular velocity world frame (cvel view)
    step_dt_tol: float,
    inv_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused foot balance penalty + L-R step asymmetry metric.

    Returns: (penalty [B], mean_step_asymmetry scalar)
    penalty = (last_air_L - last_air_R)² × landing_gate × stability_gate
    """
    just_landed = (cct > 0.0) & (cct < step_dt_tol)        # [B, 2]
    any_landing = just_landed.any(dim=1)                    # [B]
    both_valid  = (last_air_time > 0.0).all(dim=1)         # [B]
    diff        = last_air_time[:, 0] - last_air_time[:, 1]# [B]
    ang_vel_xy  = (ang_vel_w[:, 0].square() + ang_vel_w[:, 1].square()).sqrt()  # [B]
    stability   = torch.exp(-ang_vel_xy * inv_std)         # [B]
    return diff.square() * any_landing * both_valid * stability, diff.abs().mean()


class foot_contact_balance:
    """Penalize L-R step duration asymmetry, fired once at each foot landing.

    Compares completed swing durations (last_air_time) at the landing event.
    Gated by XY angular velocity — suppressed when robot is tilting/pushed so
    stability recovery takes priority over gait symmetry.

    Ablation A3 (2026-05-26): confirmed redundant for v29 tracking objective. Gait
    symmetry emerges from velocity tracking, air-time, slip, and fall constraints.
    Safe to remove from future baselines.

    Implements log_metrics(env) so Metrics/foot_step_asymmetry_mean is logged even
    when weight=0 (RewardManager protocol).

    Params:
        sensor_name: contact sensor with track_air_time=True.
        ang_vel_std: XY angular velocity gate scale in rad/s (default: 0.5).
    Returns:
        [N_envs] penalty — apply with negative weight.
    """
    def __init__(self, cfg, env):
        self.sensor      = env.scene[cfg.params["sensor_name"]]
        asset_cfg        = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset       = env.scene[asset_cfg.name]
        self.step_dt_tol = env.step_dt + 1e-8   # cached: first-contact threshold
        self.inv_std     = 1.0 / float(cfg.params.get("ang_vel_std", 0.5))

    def __call__(self, env, **_):
        data = self.sensor.data                  # single cache hit for both fields
        penalty, asymmetry = _compute_foot_balance(
            data.current_contact_time,           # live view into _air_time_state buffer
            data.last_air_time,                  # live view into _air_time_state buffer
            self.asset.data.root_link_ang_vel_w, # zero-copy slice of mjwarp cvel
            self.step_dt_tol,
            self.inv_std,
        )
        env.extras["log"]["Metrics/foot_step_asymmetry_mean"] = asymmetry
        return penalty

    def log_metrics(self, env) -> None:
        data = self.sensor.data
        _, asymmetry = _compute_foot_balance(
            data.current_contact_time,
            data.last_air_time,
            self.asset.data.root_link_ang_vel_w,
            self.step_dt_tol,
            self.inv_std,
        )
        env.extras["log"]["Metrics/foot_step_asymmetry_mean"] = asymmetry


@torch.jit.script
def _compute_foot_distance_penalty(
    foot_pos: torch.Tensor,  # [B, 2, 2] — pre-sliced to xy
    cmd: torch.Tensor,       # [B, 3]
    max_dist: float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided squared penalty + mean stance width. Returns (penalty [B], mean_dist scalar)."""
    diff = foot_pos[:, 0] - foot_pos[:, 1]   # [B, 2]
    dist = torch.norm(diff, dim=1)            # [B]
    excess = (dist - max_dist).clamp(min=0.0)
    penalty = excess.square()
    cmd_norm = (cmd[:, 0] * cmd[:, 0] + cmd[:, 1] * cmd[:, 1]).sqrt() + cmd[:, 2].abs()
    active = (cmd_norm < threshold).float()
    return penalty * active, dist.mean()      # mean fused into JIT kernel


class foot_distance:
    """One-sided squared penalty on excess foot separation, active when standing (|cmd| < threshold).

    Zero gradient at normal stance (no happy-feet incentive), no positive honeypot (no flamingo).
    max_dist = env-0 default stance + spread_margin, measured at init so threshold is robot-agnostic.

    Params:
        asset_cfg: SceneEntityCfg with site_names for foot contact sites.
        command_name: velocity command key.
        command_threshold: |cmd| below which penalty is active (default 0.1 — wider than the
            standard 0.05 to absorb command noise near zero without disabling the penalty).
        spread_margin: allowed excess above default stance in meters (default 0.07).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.site_ids = asset_cfg.site_ids
        self.command_name = cfg.params["command_name"]
        self.threshold = float(cfg.params.get("command_threshold", 0.1))

        foot_pos = self.asset.data.site_pos_w[0, self.site_ids, :2]  # [2, 2]
        default_dist = float(torch.norm(foot_pos[0] - foot_pos[1]))
        self.max_dist = default_dist + float(cfg.params.get("spread_margin", 0.07))

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        foot_pos = self.asset.data.site_pos_w[:, self.site_ids, :2]  # [B, 2, 2]
        cmd = env.command_manager.get_command(self.command_name)
        penalty, mean_dist = _compute_foot_distance_penalty(foot_pos, cmd, self.max_dist, self.threshold)
        env.extras["log"]["Metrics/stance_width_mean"] = mean_dist
        return penalty


class rel_height_control:
    """Track a commanded relative height offset around nominal standing height.

    reward = exp(-error² / std²)  where  error = base_z - (nominal_height + cmd)

    Params:
        command_name: Command term key (1D offset tensor). Default: "rel_height".
        nominal_height: Reference standing height in metres. Default: 0.59 m.
        std: Gaussian std in metres. Default: 0.05 m.
            std=0.10 m: reward at full 5 cm offset with no crouch = exp(-0.25)=0.779,
              insufficient gradient — policy can hover near nominal and score well.
            std=0.05 m: reward at full 5 cm offset with no crouch = exp(-1.0)=0.368,
              2.1× stronger pressure, still tolerant of 1.3 cm gait bob (exp(-0.07)=0.93).
        asset_cfg: SceneEntityCfg for the robot. Default: robot/base_link.

    Throughput: neg_inv_std_sq = -1/std² is precomputed so __call__ uses fma-friendly
    multiply instead of divide on every step.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        asset_cfg = p.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.command_name = p.get("command_name", "rel_height")
        self.nominal_height = float(p.get("nominal_height", 0.59))
        std = float(p.get("std", 0.05))
        self._neg_inv_std_sq = -1.0 / (std * std)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        command = env.command_manager.get_command(self.command_name)
        if command is None:
            raise RuntimeError(f"Command '{self.command_name}' not found in command_manager.")
        target_height = self.nominal_height + command[:, 0]
        base_z = self.asset.data.root_link_pos_w[:, 2]
        return torch.exp((base_z - target_height).square() * self._neg_inv_std_sq)


base_height_control = rel_height_control


@torch.jit.script
def _compute_foot_flat_reward(
    foot_quat_w: torch.Tensor,       # (B, N_feet, 4)
    gravity_expanded: torch.Tensor,  # (B, N_feet, 3) — pre-broadcast by caller
    ref_gravity_foot: torch.Tensor,  # (N_feet, 3)
    std: float,
) -> torch.Tensor:
    """Compute foot flat reward using cached reference gravity direction.

    Measures alignment between gravity projected into each foot frame and the
    reference direction when feet are flat. Uses perpendicular component squared:
    1 - dot² = 0 when perfectly aligned.
    """
    B, N_feet = foot_quat_w.shape[:2]

    # Project gravity into foot frame
    projected_gravity = quat_apply_inverse(
        foot_quat_w.reshape(-1, 4), gravity_expanded.reshape(-1, 3)
    ).reshape(B, N_feet, 3)

    # dot[b, f] = alignment with reference; 1 - dot² = 0 when aligned
    dot = torch.sum(projected_gravity * ref_gravity_foot.unsqueeze(0), dim=-1)
    perp_squared = 1.0 - dot * dot

    return torch.mean(torch.exp(-perp_squared / (std * std)), dim=-1)


class foot_flat_orientation:
    """Reward flat foot orientation (feet being horizontal).

    Penalizes deviation from the expected gravity direction in the foot's local frame.
    When feet are flat on the ground, gravity should project to a specific direction
    in the foot frame - this direction depends on how the foot frame is defined in
    the URDF/MJCF model.

    !! IMPORTANT: You MUST set flat_gravity_dir based on your robot's foot frame !!

    To determine flat_gravity_dir for a new robot:
        1. Check the foot body's local coordinate frame in your URDF/MJCF
        2. When the foot is flat on ground, which local axis points UP?
        3. Gravity points DOWN, so flat_gravity_dir is the negative of that axis

    Common conventions:
        - Foot z-axis up (most common): flat_gravity_dir = (0, 0, -1)
        - Foot x-axis up (humanoid_v21):  flat_gravity_dir = (-1, 0, 0)
        - Foot y-axis up:                 flat_gravity_dir = (0, -1, 0)

    Params:
        std: Standard deviation for exponential reward shaping.
        flat_gravity_dir: Gravity direction in foot frame when foot is flat.
            MUST match your robot's foot frame convention!
        asset_cfg: Asset config with body_names for foot bodies.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        asset_cfg = p.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids   # list[int] | slice
        self.std = float(p["std"])
        flat_gravity_dir: tuple[float, float, float] = p.get("flat_gravity_dir", (-1.0, 0.0, 0.0))

        # Derive N_feet from tensor shape — safe for both list[int] and slice body_ids
        foot_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids, :]
        N_feet = foot_quat_w.shape[1]

        self.ref_gravity = torch.tensor(
            [flat_gravity_dir] * N_feet,
            device=foot_quat_w.device,
            dtype=foot_quat_w.dtype,
        )  # (N_feet, 3)

        # Verification (once at init): check env 0 gravity aligns with configured reference
        gravity_dir = self.asset.data.gravity_vec_w[0]                    # (3,) from env 0
        gravity_expanded = gravity_dir.unsqueeze(0).expand(N_feet, -1)    # (N_feet, 3)
        actual_gravity = quat_apply_inverse(foot_quat_w[0], gravity_expanded)

        dot = torch.sum(actual_gravity * self.ref_gravity, dim=-1)
        if torch.any(dot < 0.95):
            import warnings
            warnings.warn(
                f"[foot_flat_orientation] low alignment (dot={dot.tolist()}) — "
                f"check flat_gravity_dir setting (configured: {flat_gravity_dir}, "
                f"actual env 0: {actual_gravity.tolist()})"
            )

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        foot_quat_w = self.asset.data.body_link_quat_w[:, self.body_ids, :]
        # gravity_vec_w is (B, 3); expand to (B, N_feet, 3) before JIT call
        gravity_expanded = self.asset.data.gravity_vec_w.unsqueeze(1).expand_as(foot_quat_w[..., :3])
        return _compute_foot_flat_reward(foot_quat_w, gravity_expanded, self.ref_gravity, self.std)


@torch.jit.script
def _compute_feet_clearance_tanh_reward(
    foot_pos_z: torch.Tensor,
    foot_vel_xy: torch.Tensor,
    target_height: float,
    tanh_scale: float,
) -> torch.Tensor:
    """JIT-compiled helper for feet clearance reward."""
    # Velocity factor: tanh(v * scale)
    # Saturation at low speeds prevents "cheap" foot dragging
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    vel_factor = torch.tanh(vel_norm * tanh_scale)

    # Height error
    delta = torch.abs(foot_pos_z - target_height)

    # Weighted cost
    return torch.sum(delta * vel_factor, dim=1)


class feet_clearance_tanh:
    """Penalize deviation from target clearance height, weighted by tanh of foot velocity.

    This version remains active even at lower speeds compared to the linear velocity weighting.
    Caches asset reference and parameters to avoid per-step dict lookups.

    Params:
        target_height: Target foot clearance height (m).
        tanh_scale: Scales foot velocity before tanh; higher = saturates at lower speeds.
        command_name: If set, mask cost to zero when command magnitude is below threshold.
        command_threshold: Minimum total command magnitude to activate the reward.
        asset_cfg: Asset config with site_names for foot sites.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        asset_cfg = p.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.site_ids = asset_cfg.site_ids   # list[int] | slice
        self.target_height = float(p["target_height"])
        self.tanh_scale = float(p.get("tanh_scale", 5.0))
        self.command_name: str | None = p.get("command_name")
        self.command_threshold = float(p.get("command_threshold", 0.01))

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        foot_z = self.asset.data.site_pos_w[:, self.site_ids, 2]        # [B, N]
        foot_vel_xy = self.asset.data.site_lin_vel_w[:, self.site_ids, :2]  # [B, N, 2]
        cost = _compute_feet_clearance_tanh_reward(
            foot_z, foot_vel_xy, self.target_height, self.tanh_scale
        )
        if self.command_name is not None:
            cmd = env.command_manager.get_command(self.command_name)
            active = (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs() > self.command_threshold).float()
            return cost * active
        return cost


class feet_swing_height_softplus:
    """Penalize insufficient foot swing height at landing via a soft-plus barrier.

    Replaces the symmetric squared-error of mjlab's feet_swing_height with a one-sided
    soft-plus barrier:  penalty = softplus((h_min - h_peak) / sigma)  evaluated at landing.

    sigma is the e-folding decay length above h_min — analogous to the standard deviation
    of a Gaussian: it sets the width of the transition band in meters.  The penalty decays
    as exp(-(h - h_min) / sigma) for h > h_min, so at h = h_min + sigma the value has
    dropped to 1/e ≈ 37%, and at h = h_min + 3*sigma it is negligible (~5%).

    Shape properties (h_min=0.045 m):
                       sigma=5 mm   sigma=10 mm
      h=0.000  →         ≈9.00        ≈4.51   (foot never lifted — strong push)
      h=0.025  →         ≈4.02        ≈2.13
      h=0.045  →   log(2) ≈ 0.693    (at threshold, independent of sigma)
      h=0.050  →         ≈0.368       ≈0.607  (1 sigma above threshold, = 1/e)
      h=0.060  →         ≈0.049       ≈0.201  (3 sigma / 1.5 sigma above threshold)
      h=0.100  →         ≈1.7e-5      ≈0.004

    Tuning rule: set sigma to the clearance margin within which you still want the
    policy to feel a push.  sigma=5 mm → negligible above h_min+15 mm; sigma=10 mm →
    negligible above h_min+30 mm.

    Advantages over squared-error:
    - One-sided: no penalty for lifting higher than h_min (no symmetric pull back down)
    - No boundary camping: gradient is highest at h≈0, the most dangerous region
    - True saturation: penalty is negligible for h >> h_min, unlike log-barrier which grows
      without bound (ref: 2409.15780 uses log-barrier; soft-plus preferred here)

    Landing-peak mechanism (identical to mjlab feet_swing_height):
    - peak_heights accumulates maximum height during each swing phase
    - Barrier evaluated at first_contact (landing), then peak_heights reset
    - Prevents micro-lift hacks where policy briefly twitches foot without real clearance

    A1 result (2026-05-25): flat-plane peak height is a 22 mm hard attractor regardless
    of this penalty. Kept as motion-quality shaper — removing raises slip +9% and impact
    +5%. Do not remove on flat terrain; purpose is gait shaping, not height control.

    Implements log_metrics(env) so Metrics/peak_height_mean is logged even when weight=0.
    log_metrics also updates peak_heights state so the tracker stays consistent if weight
    is later re-enabled. _update_peak_state() is the shared helper for both paths.

    Params:
        sensor_name: ContactSensor key in scene.
        height_sensor_name: TerrainHeightSensor key in scene (num_frames = num_feet).
        h_min: Minimum acceptable peak swing height (m). Default: 0.045.
        sigma: e-folding decay length above h_min (m); penalty is negligible above
               h_min + 3*sigma. Default: 0.005 (5 mm). Stored as inv_sigma = 1/sigma
               to replace per-call division with a multiply.
        command_name: Command key used for activity gating.
        command_threshold: Minimum |cmd| to activate penalty. Default: 0.05.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        p = cfg.params
        height_sensor = env.scene[p["height_sensor_name"]]
        self.contact_sensor = env.scene[p["sensor_name"]]
        self.height_sensor = height_sensor
        self.h_min = float(p.get("h_min", 0.045))
        self.inv_sigma = 1.0 / float(p.get("sigma", 0.005))  # precomputed 1/sigma; avoids division in hot path
        self.command_name: str = p["command_name"]
        self.command_threshold = float(p.get("command_threshold", 0.05))
        self.step_dt = env.step_dt
        self.peak_heights = torch.zeros(
            (env.num_envs, height_sensor.num_frames),
            device=env.device,
            dtype=torch.float32,
        )
        self._peak_zeros = torch.zeros_like(self.peak_heights)  # preallocated; avoids per-step allocation
        self._peak_max_buf = torch.zeros_like(self.peak_heights)  # scratch buffer for fused in-place update

    def _update_peak_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Update peak height tracker and return (first_contact bool, first_contact float)."""
        foot_heights = self.height_sensor.data.heights
        in_air = self.contact_sensor.data.found == 0
        torch.maximum(self.peak_heights, foot_heights, out=self._peak_max_buf)
        torch.where(in_air, self._peak_max_buf, self.peak_heights, out=self.peak_heights)
        first_contact = self.contact_sensor.compute_first_contact(dt=self.step_dt)
        return first_contact, first_contact.float()

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        first_contact, first_contact_f = self._update_peak_state()

        cmd = env.command_manager.get_command(self.command_name)
        active = (cmd[:, :2].norm(dim=1) + cmd[:, 2].abs() > self.command_threshold).float()

        barrier = torch.nn.functional.softplus((self.h_min - self.peak_heights) * self.inv_sigma)
        cost = torch.sum(barrier * first_contact_f, dim=1) * active

        num_landings = first_contact_f.sum()
        env.extras["log"]["Metrics/peak_height_mean"] = (
            (self.peak_heights * first_contact_f).sum() / num_landings.clamp(min=1)
        )

        self.peak_heights.masked_fill_(first_contact, 0.0)
        return cost

    def log_metrics(self, env: ManagerBasedRlEnv) -> None:
        # State must update every step even at weight=0 so peak_heights stays
        # consistent if weight is later re-enabled.
        first_contact, first_contact_f = self._update_peak_state()
        num_landings = first_contact_f.sum()
        env.extras["log"]["Metrics/peak_height_mean"] = (
            (self.peak_heights * first_contact_f).sum() / num_landings.clamp(min=1)
        )
        self.peak_heights.masked_fill_(first_contact, 0.0)


class body_angular_velocity_penalty:
    """Body XY angular velocity penalty with dual init signatures.

    Signature A (base, identical to vel_mdp):
      params={"asset_cfg": ...}

    Signature B (command-gated):
      params={"asset_cfg": ..., "command_name": "twist", "cmd_std": 0.1,
              "standing_gain": 3.0, "walking_gain": 1.0}

    Runtime path is fixed at init (base or gated) to avoid per-step branching.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        params = cfg.params
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot", body_names=("base_link",)))
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids

        has_gate_cfg = (
            "command_name" in params
            or "cmd_std" in params
            or "standing_gain" in params
            or "walking_gain" in params
        )

        if has_gate_cfg:
            self.command_name: str = params.get("command_name", "twist")
            cmd_std = float(params.get("cmd_std", 0.1))
            self.inv_cmd_var = 1.0 / (cmd_std * cmd_std)
            self.standing_gain = float(params.get("standing_gain", 3.0))
            self.walking_gain = float(params.get("walking_gain", 1.0))
            self._call_impl = self._call_gated
        else:
            self._call_impl = self._call_base

    def _ang_vel_xy_sq(self) -> torch.Tensor:
        return self.asset.data.body_link_ang_vel_w[:, self.body_ids, :2].squeeze(1).square().sum(dim=1)

    def _call_base(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._ang_vel_xy_sq()

    def _call_gated(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        ang_vel_xy_sq = self._ang_vel_xy_sq()
        cmd_lin_sq = env.command_manager.get_command(self.command_name)[:, :2].square().sum(dim=1)
        gate = torch.exp(-cmd_lin_sq * self.inv_cmd_var)
        scale = self.walking_gain + (self.standing_gain - self.walking_gain) * gate
        return ang_vel_xy_sq * scale

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self._call_impl(env) # fixed at init for efficiency/readability


# Backward-compatible alias; remove after all references migrate.
body_angular_velocity_penalty_cmd_gated = body_angular_velocity_penalty


class body_angular_velocity_penalty_base_frame:
    """Body XY angular velocity penalty in BASE frame (yaw-invariant).

    The default mjlab / humanoid `body_angular_velocity_penalty` reads
    `body_link_ang_vel_w` (WORLD frame) and squares its XY components. When the
    base is yawed 90°, a pure-Z yaw rotation projects onto world-X and world-Y,
    inflating the world-XY penalty even though the BASE-frame angular velocity
    is purely Z. This couples yaw turning to a tilt-rate penalty and biases the
    policy against commanded heading changes (esp. under `heading_command=True`,
    where the base must turn to face `heading_target`).

    Fix: rotate `body_link_ang_vel_w` into the base frame via
    `quat_apply_inverse(root_quat_w, ang_vel_w)`. In base frame, only true roll
    and pitch rates remain in XY → the penalty becomes invariant under yaw.
    Gating semantics identical to `body_angular_velocity_penalty`.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        params = cfg.params
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot", body_names=("base_link",)))
        self.asset = env.scene[asset_cfg.name]
        self.body_ids = asset_cfg.body_ids

        has_gate_cfg = (
            "command_name" in params
            or "cmd_std" in params
            or "standing_gain" in params
            or "walking_gain" in params
        )

        if has_gate_cfg:
            self.command_name: str = params.get("command_name", "twist")
            cmd_std = float(params.get("cmd_std", 0.1))
            self.inv_cmd_var = 1.0 / (cmd_std * cmd_std)
            self.standing_gain = float(params.get("standing_gain", 3.0))
            self.walking_gain = float(params.get("walking_gain", 1.0))
            self._call_impl = self._call_gated
        else:
            self._call_impl = self._call_base

    def _ang_vel_xy_sq_b(self) -> torch.Tensor:
        # world-frame ang vel of base_link → base-frame via quat_apply_inverse
        ang_vel_w = self.asset.data.body_link_ang_vel_w[:, self.body_ids, :].squeeze(1)
        root_quat_w = self.asset.data.root_link_quat_w
        ang_vel_b = quat_apply_inverse(root_quat_w, ang_vel_w)
        return ang_vel_b[:, :2].square().sum(dim=1)

    def _call_base(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        return self._ang_vel_xy_sq_b()

    def _call_gated(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        ang_vel_xy_sq = self._ang_vel_xy_sq_b()
        cmd_lin_sq = env.command_manager.get_command(self.command_name)[:, :2].square().sum(dim=1)
        gate = torch.exp(-cmd_lin_sq * self.inv_cmd_var)
        scale = self.walking_gain + (self.standing_gain - self.walking_gain) * gate
        return ang_vel_xy_sq * scale

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self._call_impl(env)


class base_linear_velocity_stationary_penalty:
    """Penalize base horizontal velocity when stationary AND the arm is swinging.

    Targets the deploy failure "base sways when the robot holds a zero locomotion
    command but the arm follows a trajectory" (arm-reaction disturbance). Probe evidence
    (mj_envs/probe/sway_probe.py): at zero base command the graph-nav gate lets the arm
    traverse at FULL speed, so the reaction impulse is largest exactly when the base
    should hold still; the arm adds +52% peak XY drift / +28% base-speed RMS over the
    frozen-arm case, and it is TRANSLATIONAL (yaw ~unaffected). The existing (saturated)
    track_linear reward does not fully reject it. This term adds squared-velocity pressure
    EXACTLY in that regime and nowhere else:

        penalty = ||root_link_lin_vel_b[:, :2]||^2 * stand_gate * arm_active

    stand_gate = exp(-|cmd_xy|^2 / cmd_std^2): 1 at zero cmd, ->0 under locomotion, so the
      policy is NOT penalized for the base motion it is commanded to produce (leaves the
      locomotion regime untouched — no re-litigation of the closed gait/track axes).
    arm_active = (~arm_ref._frozen_mask): 1 only for envs whose arm is actually swinging;
      frozen-arm standing already holds still via track_linear, so this avoids piling
      redundant pressure on the already-good frozen case and keeps the term single-purpose.

    Base-frame (yaw-invariant) horizontal velocity only; yaw rate is left to body_ang_vel
    (probe showed the arm reaction is translational, not yaw). Single new reward term whose
    WEIGHT is the experiment's single variable.

    Design decisions / rejected alternatives:
      * squared velocity (not exp-kernel): the disturbance is a velocity to be driven to
        zero; a quadratic penalty grows with sway magnitude (punishes big excursions
        harder) and needs no std to tune.
      * arm_active gate reads the arm_ref term's _frozen_mask directly (research env; the
        reward manager already reaches into command_manager). Rejected inferring frozen via
        interpolation_rate==0 — that also fires momentarily at swing-node snaps.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        params = cfg.params
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.command_name: str = params.get("command_name", "twist")
        cmd_std = float(params.get("cmd_std", 0.1))
        self.inv_cmd_var = 1.0 / (cmd_std * cmd_std)
        self.arm_command_name: str = params.get("arm_command_name", "arm_ref")
        self._arm_term = None  # resolved lazily (built after reward manager)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        v_xy_sq = self.asset.data.root_link_lin_vel_b[:, :2].square().sum(dim=1)
        cmd_lin_sq = env.command_manager.get_command(self.command_name)[:, :2].square().sum(dim=1)
        stand_gate = torch.exp(-cmd_lin_sq * self.inv_cmd_var)
        if self._arm_term is None:
            self._arm_term = env.command_manager._terms[self.arm_command_name]
        arm_active = (~self._arm_term._frozen_mask).float()
        return v_xy_sq * stand_gate * arm_active


class upright_gated:
    """Upright (base-flatness) reward with a command-gated standing boost.

    Base reward is identical to mjlab ``vel_mdp.upright`` on flat ground:
        flat = exp(-tilt_xy^2 / std^2)
    where tilt_xy^2 = squared XY of the base-frame projected gravity (roll/pitch
    magnitude). Uses ``projected_gravity_b`` directly (base_link is the floating
    root), matching the tilt-cache path used elsewhere in this module.

    Gate mirrors ``body_angular_velocity_penalty`` (tilt-RATE gate), but applied
    to tilt-MAGNITUDE and as a POSITIVE reward scale:
        gate  = exp(-|cmd_lin_xy|^2 / cmd_std^2)      in [0,1], 1 at standing
        scale = walking_gain + (standing_gain - walking_gain) * gate
        reward = flat * scale

    Purpose / rationale. The v124-v126 GLOBAL upright-weight sweep raised the
    reward but barely moved base flatness (exp-term ceiling ~0.895) and, at w3.0,
    shaved walking yaw-tracking — a flat weight demands flatness while WALKING too,
    where commanded tilt is legitimate. This gate demands a flatter base ONLY in
    the grasp window (cmd~0) and relaxes to ``walking_gain`` under locomotion, so
    the flatness pressure lands where EE precision needs it without taxing gait.

    Params (via RewardTermCfg.params):
        std: float                 # flatness kernel, unchanged from upright
        asset_cfg: SceneEntityCfg  # base body (default robot/base_link)
        command_name: str = "twist"
        cmd_std: float = 0.1       # standing detection width (matches bav gate)
        standing_gain: float = 3.0 # flatness scale at cmd~0
        walking_gain: float = 1.0  # flatness scale under full command
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        params = cfg.params
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot", body_names=("base_link",)))
        self.asset = env.scene[asset_cfg.name]
        std = float(params["std"])
        self.inv_std_sq = 1.0 / (std * std)
        self.command_name: str = params.get("command_name", "twist")
        cmd_std = float(params.get("cmd_std", 0.1))
        self.inv_cmd_var = 1.0 / (cmd_std * cmd_std)
        self.standing_gain = float(params.get("standing_gain", 3.0))
        self.walking_gain = float(params.get("walking_gain", 1.0))

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        tilt_xy_sq = self.asset.data.projected_gravity_b[:, :2].square().sum(dim=1)
        flat = torch.exp(-tilt_xy_sq * self.inv_std_sq)
        cmd_lin_sq = env.command_manager.get_command(self.command_name)[:, :2].square().sum(dim=1)
        gate = torch.exp(-cmd_lin_sq * self.inv_cmd_var)
        scale = self.walking_gain + (self.standing_gain - self.walking_gain) * gate
        return flat * scale



# =============================================================================
# Push-Aware Tracking
# =============================================================================


@torch.jit.script
def _compute_push_suppression(
    base: torch.Tensor,
    delta_vel: torch.Tensor,
    vel_scale_sq: torch.Tensor,
    last_push_step: torch.Tensor,
    current_step: int,
    step_dt: float,
    max_tau: float,
) -> torch.Tensor:
    magnitude_sq = (delta_vel.square() * vel_scale_sq).sum(dim=1)
    tau = magnitude_sq.clamp(0.0, 1.0) * max_tau
    tau = tau.clamp(min=1e-6)

    elapsed = (current_step - last_push_step) * step_dt
    suppression = torch.exp(-elapsed / tau)
    return base + suppression * (1.0 - base)


class PushGatedLinearVelocityTracking:
    """track_linear_velocity with magnitude-aware exponential suppression after a push.

    Immediately after a push the reward is 0; it recovers to the base value
    with time constant tau seconds:
        reward = track_linear_velocity(...) * (1 - exp(-elapsed / tau))

    tau is computed dynamically by weighting the 6D delta velocity of the push:
        magnitude = norm(delta_vel * vel_scale)
        tau = clamp(magnitude, 0.0, 1.0) * max_tau

    Requires push_and_record (in event.py) to be used as the push_robot event.

    Params (via RewardTermCfg.params):
        std: Gaussian std for velocity error (same as track_linear_velocity).
        command_name: Name of velocity command.
        max_tau: Maximum decay time constant in seconds.
        vel_scale: Tuple[float, float, float, float, float, float] scaling multiplier
                   for each of the 6 components of the delta velocity [lin_xyz, ang_xyz].
        asset_cfg: SceneEntityCfg (optional, default robot).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.std = cfg.params["std"]
        self.command_name = cfg.params["command_name"]
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.max_tau = float(cfg.params.get("max_tau", 1.0))
        scale = cfg.params.get("vel_scale", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.vel_scale = torch.tensor(scale, device=env.device, dtype=torch.float32)
        self.vel_scale_sq = self.vel_scale ** 2  # pre-squared for JIT helper

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        base = vel_mdp.track_linear_velocity(env, self.std, self.command_name, self.asset_cfg)
        if not hasattr(env, "_last_push_step"):
            return base
        return _compute_push_suppression(
            base, env._last_push_delta_vel, self.vel_scale_sq,
            env._last_push_step, env.common_step_counter, env.step_dt, self.max_tau
        )


class PushGatedAngularVelocityTracking:
    """track_angular_velocity with exponential suppression after a push.

    Same suppression logic as PushGatedLinearVelocityTracking.

    Params (via RewardTermCfg.params):
        std: Gaussian std for velocity error (fixed-std form).
        OR std_rel + std_min: command-proportional std (relative-form, e.g.
                              `track_angular_velocity_relative`). std_eff = max(std_rel*|cmd|, std_min).
        command_name: Name of velocity command.
        max_tau: Maximum decay time constant in seconds.
        vel_scale: Tuple[float, float, float, float, float, float] scaling multiplier
                   for each of the 6 components of the delta velocity [lin_xyz, ang_xyz].
        asset_cfg: SceneEntityCfg (optional, default robot).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.command_name = cfg.params["command_name"]
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.max_tau = float(cfg.params.get("max_tau", 1.0))
        scale = cfg.params.get("vel_scale", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        self.vel_scale = torch.tensor(scale, device=env.device, dtype=torch.float32)
        self.vel_scale_sq = self.vel_scale ** 2  # pre-squared for JIT helper
        # std form: either fixed (std) or relative (std_rel + std_min)
        if "std" in cfg.params:
            self.use_relative = False
            self.std = float(cfg.params["std"])
        elif "std_rel" in cfg.params and "std_min" in cfg.params:
            self.use_relative = True
            self.std_rel = float(cfg.params["std_rel"])
            self.std_min = float(cfg.params["std_min"])
        else:
            raise KeyError("PushGatedAngularVelocityTracking requires 'std' OR 'std_rel'+'std_min' params")

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if self.use_relative:
            # Replicate track_angular_velocity_relative's std_eff computation
            asset = env.scene[self.asset_cfg.name]
            command = env.command_manager.get_command(self.command_name)
            actual = asset.data.root_link_ang_vel_b
            z_error = torch.square(command[:, 2] - actual[:, 2])
            xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
            ang_vel_error = z_error + xy_error
            std_eff = torch.clamp(self.std_rel * command[:, 2].abs(), min=self.std_min)
            base = torch.exp(-ang_vel_error / (std_eff ** 2))
        else:
            base = vel_mdp.track_angular_velocity(env, self.std, self.command_name, self.asset_cfg)
        if not hasattr(env, "_last_push_step"):
            return base
        return _compute_push_suppression(
            base, env._last_push_delta_vel, self.vel_scale_sq,
            env._last_push_step, env.common_step_counter, env.step_dt, self.max_tau
        )


class TrackAngularVelocityCmdGated:
    """track_angular_velocity gated by |cmd_yaw| > threshold.

    When |command[:, 2]| <= cmd_gate_threshold, the reward is zero (no gradient).
    When |command[:, 2]| > cmd_gate_threshold, the reward is the full
    `vel_mdp.track_angular_velocity(env, std, command_name, asset_cfg)` kernel.

    Purpose: decouple stand-noise gradient (cmd_yaw ~ 0) from turn-tracking
    gradient (cmd_yaw != 0). The default track_angular_velocity fires at all
    cmds; at stand the reward is ~1.0 (trivial "do nothing" gradient pressure).
    This gated form gives a clean tracking signal ONLY during turns, freeing the
    critic's stand-state value from competing yaw pressure.

    Params (via RewardTermCfg.params):
        std: Gaussian std for velocity error (same as track_angular_velocity).
        command_name: Name of velocity command (e.g. "twist").
        cmd_gate_threshold: |cmd_yaw| threshold below which the reward is zero.
        asset_cfg: SceneEntityCfg (optional, default robot).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.std = float(cfg.params["std"])
        self.command_name = cfg.params["command_name"]
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.cmd_gate_threshold = float(cfg.params.get("cmd_gate_threshold", 0.05))

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        base = vel_mdp.track_angular_velocity(env, self.std, self.command_name, self.asset_cfg)
        command = env.command_manager.get_command(self.command_name)
        # |cmd_yaw| > threshold mask; zero gradient at stand.
        gate = (command[:, 2].abs() > self.cmd_gate_threshold).float()
        return base * gate


# =============================================================================
# Arm Tracking
# =============================================================================


class arm_pose:
    """Arm joint tracking reward for HumanoidLocoArmFollow.

    r = exp(-mean((q_arm - q_arm_target)² / σ²)) × exp(-‖v_cmd_xy‖² / σ_gate²)

    Reads env._arm_joint_target (B, K) written by arm_joint_target obs term. Uses
    the same asset_cfg joint pattern so joint ordering is guaranteed to match.

    Velocity gate (σ_gate²=0.16) suppresses the reward when walking fast, where
    arm swing from gait dynamics would otherwise create conflicting gradients.
    At standing (‖v_cmd‖=0): gate=1.0. At 0.8 m/s speed cap: gate≈0.01.

    Fixed σ=0.1 rad — same as baseline pose term standing tolerance. Not speed-
    dependent because the gate already handles locomotion-phase suppression.

    This term replaces the arm component of the baseline pose reward (which is
    restricted to leg joints in HumanoidLocoArmFollow), so weight=1.0 keeps the
    total reward budget unchanged.

    Params: asset_cfg, command_name, std (default 0.1), gate_sigma_sq (default 0.16).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        joint_ids, _ = self.asset.find_joints(asset_cfg.joint_names)
        self.joint_ids = joint_ids
        self._inv_std_sq = 1.0 / float(cfg.params.get("std", 0.1)) ** 2
        self._inv_gate_sq = 1.0 / float(cfg.params.get("gate_sigma_sq", 0.16))
        self._command_name = cfg.params.get("command_name", "twist")

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        q = self.asset.data.joint_pos[:, self.joint_ids]    # (B, K)
        target = env._arm_joint_target                       # (B, K)
        cmd = env.command_manager.get_command(self._command_name)
        speed_sq = cmd[:, 0].square() + cmd[:, 1].square()
        # fuse exp(reward_arg) * exp(gate_arg) → exp(reward_arg + gate_arg)
        return torch.exp(
            -(q - target).square().mean(dim=1) * self._inv_std_sq
            - speed_sq * self._inv_gate_sq
        )


class arm_joint_tracking:
    """Reward for tracking the target arm joint positions.

    Computes exp(-mean(error^2)/std^2) between simulated arm joint positions and the target arm joint positions.
    """
    def __init__(self, cfg, env: ManagerBasedRlEnv):
        std = float(cfg.params.get("std", 0.1))
        self._inv_std_sq = 1.0 / std ** 2

        # Two sources (resolved once): command_name (preferred, a joint-reference CommandTerm,
        # reads .command) or arm_action_name (legacy, reference-baked action, reads
        # .current_target). Both expose .target_names for joint-id resolution.
        command_name = cfg.params.get("command_name", None)
        self._from_command = command_name is not None
        if self._from_command:
            self._term = env.command_manager.get_term(command_name)
        else:
            arm_action_name = cfg.params.get("arm_action_name", "joint_pos_arms")
            self._term = env.action_manager._terms[arm_action_name]

        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]

        # Resolve target actuator names to joint ids
        joint_ids, _ = self.asset.find_joints(self._term.target_names)
        self.joint_ids = joint_ids

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        target_pos = self._term.command if self._from_command else self._term.current_target
        sim_pos = self.asset.data.joint_pos[:, self.joint_ids]

        error_sq = torch.square(sim_pos - target_pos)
        return torch.exp(-torch.mean(error_sq, dim=1) * self._inv_std_sq)


# =============================================================================
# Lower-Body Joint Target Tracking
# =============================================================================


class joint_target_tracking:
    """Reward for tracking a lower-body joint target derived from relative height.

    Computes exp(-mean(error^2)/std^2) between simulated lower-body joint positions
    and the height-command-derived target joint positions. The target order matches
    LEG_JOINT_NAMES.

    Cache coupling: if the `joint_target` obs term is active, this reward reads its
    cached IK result from `env.<target_attr>` (step-guarded) instead of recomputing.
    `target_attr` param must match between this reward and the obs term (both default
    to `"_joint_target"`). Mismatch causes silent double-compute, not wrong values.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._command_name = cfg.params.get("command_name", "rel_height")
        self._nominal_z_travel_mm = float(cfg.params.get("nominal_z_travel_mm", -40.0))
        self._inv_std_sq = 1.0 / float(cfg.params.get("std", 0.2)) ** 2
        self._joint_names = tuple(cfg.params.get("joint_names", LEG_JOINT_NAMES))
        # Cache attrs must match joint_target obs term (same target_attr, same command_name).
        self._target_attr = cfg.params.get("target_attr", "_joint_target")
        self._step_attr = f"{self._target_attr}_step"

        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]

        joint_ids = []
        for joint_name in self._joint_names:
            found_ids, _ = self.asset.find_joints([joint_name])
            joint_ids.append(found_ids[0])
        # Tensor index is faster than Python list fancy indexing in the hot-path slice.
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        # Reuse IK result cached by joint_target obs term if computed this step.
        if getattr(env, self._step_attr, -1) == env.common_step_counter:
            target_pos = getattr(env, self._target_attr)
        else:
            command = env.command_manager.get_command(self._command_name)
            z_travel = torch.clamp(
                self._nominal_z_travel_mm + command[:, 0] * 1000.0,
                -150.0,
                0.0,
            )
            target_pos = vertical_translate_lower_body_joints_batch(z_travel)
        sim_pos = self.asset.data.joint_pos[:, self.joint_ids]

        error_sq = torch.square(sim_pos - target_pos)
        return torch.exp(-torch.mean(error_sq, dim=1) * self._inv_std_sq)


# =============================================================================
# Bimanual Arm Proximity (Analytical)
# =============================================================================


class arm_proximity_reward:
    """Analytical K=48 capsule-capsule proximity penalty for bimanual arm collision avoidance.

    Computes exact signed clearance distances between all pairs of:
        - 36 cross-arm segment pairs (6 left × 6 right)
        - 6 left arm vs torso pairs
        - 6 right arm vs torso pairs

    No MLP, no approximation error. Uses the Shene segment-to-segment algorithm
    (~800 FLOPs/pair) with a parallel-segment fallback via 4 point-to-segment
    distances (correct for overlapping parallel segments).

    Reward signal: weight × sum_k relu(MARGIN - d_k)² where MARGIN=0.10m.
    Calibration with weight=-30.0: -0.30 per touching pair (≈15% of +2.0 primary).

    Args (via cfg.params): none required; "asset_cfg" uses robot by default.

    See utils/proximity_model.py for full geometry details and unit tests.
    See plan/ARM_PROXIMITY_MODEL.md for design rationale.

    *** KNOWN BUGS — DO NOT ENABLE IN TRAINING UNTIL FIXED ***

    Bug 1 (CRITICAL): find_bodies ordering not guaranteed.
        `self.asset.find_bodies(all_names)` returns body IDs sorted by the
        articulation's internal model order, NOT by the query list order. If any
        arm body has a lower model-index than base_link (pelvis), the row layout
        seen by compute_bimanual_distances will be scrambled. compute_bimanual_
        distances expects row 0 = pelvis, rows 1-7 = L arm, rows 8-14 = R arm.
        Fix: re-order body_ids to match all_names order after find_bodies returns.
        Diagnostic: print body_ids and cross-check with mj_name2id in validate_
        arm_proximity_setup.

    Bug 2 (UNKNOWN): quaternion convention of body_link_quat_w.
        compute_bimanual_distances expects wxyz. If body_link_quat_w returns xyzw
        (common in ROS/quaternion_xyzw convention used by some mjlab versions),
        the pelvis-frame rotation will be completely wrong.
        Fix: verify convention by logging body_link_quat_w[0, 0, :] against known
        neutral pose and comparing to mujoco.MjData.xquat (which is wxyz).

    Bug 3 (MINOR): dead init code.
        self._prox_a/b, _dist_a/b, _radii are set in __init__ but never used;
        compute_bimanual_distances uses its own module-level constants and moves
        them to device per call. Delete dead code once Bugs 1/2 are resolved.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg  = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]

        # BUG 1: find_bodies may return IDs in model order, not all_names order.
        # Needs reorder fix before production use.
        all_names = [PELVIS_BODY_NAME] + ARM_BODY_NAMES
        body_ids, _ = self.asset.find_bodies(all_names)
        self.body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)

        # BUG 3: these are set but never used in __call__ (dead code).
        self._prox_a = PROX_A.to(env.device)
        self._dist_a = DIST_A.to(env.device)
        self._prox_b = PROX_B.to(env.device)
        self._dist_b = DIST_B.to(env.device)
        self._radii  = RADII.to(env.device)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        # BUG 2: body_link_quat_w convention (wxyz vs xyzw) unverified.
        pos  = self.asset.data.body_link_pos_w[:, self.body_ids, :]    # (N, 15, 3)
        quat = self.asset.data.body_link_quat_w[:, self.body_ids, :]   # (N, 15, 4) — wxyz assumed

        d    = compute_bimanual_distances(pos, quat)                    # (N, 48)

        # Log diagnostics (picked up by wandb/compact_logger if extras["log"] is set)
        log = env.extras.get("log")
        if log is not None:
            log["proximity/min_clearance_mean"] = d.min(-1).values.mean()
            log["proximity/collision_rate"]     = (d.min(-1).values < 0).float().mean()

        return reward_arm_proximity(d)   # (N,), ≥ 0; apply with weight < 0 in cfg


# =============================================================================
# End-Effector Tracking Reward
# =============================================================================


class ee_tracking_coarse:
    """Coarse EE position tracking reward in the horizontal frame.

    r = exp(-||ee_error_h||² / σ²), σ = 5cm (default).

    Reads env._ee_target_pos_h, which must be set by the experiment (M3+).
    At default arm pose with target initialized to default EE position, this
    reward is ≈1.0.

    Optional velocity gating (M4+): when gate_command_name is set, the reward
    is multiplied by gate = exp(-||v_cmd_xy||² / gate_sigma_sq). This softly
    suppresses the EE reward while the robot is walking (gate≈0.018 at 0.4 m/s),
    preventing the arm from fighting locomotion body motion, while still rewarding
    EE tracking when standing (gate=1.0 at v=0). Default gate_sigma_sq=0.04
    matches the ee_gate observation term (blueprint §5.2).

    Cache coupling: shares `_EEHFrameComputer` (stored at `env._ee_h_frame_{body_id}`)
    with `ee_current_pos_h` / `ee_error_h` obs terms. Horizontal frame computed at most
    once per step regardless of which terms are active. If ee_body_name differs between
    this reward and the obs terms, separate computers are created (no interference).

    Params:
        ee_body_name:      MuJoCo body name of the end-effector.
        std:               Gaussian σ in meters (default 0.05 = 5cm, from §6.3).
        gate_command_name: Velocity command key to derive gate from (optional).
                           When None (default), no gating is applied (M3 behaviour).
        gate_sigma_sq:     Gate width σ² (default 0.04 = σ=0.2 m/s, from §5.2).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        from tasks.humanoid_velocity.observation import _EEHFrameComputer
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        body_ids, _ = self.asset.find_bodies([cfg.params["ee_body_name"]])
        self._ee_body_id = body_ids[0]
        sigma = float(cfg.params.get("std", 0.05))
        self._inv_sigma_sq = 1.0 / sigma ** 2
        self._gate_command: str | None = cfg.params.get("gate_command_name", None)
        self._gate_inv_sigma_sq = 1.0 / float(cfg.params.get("gate_sigma_sq", 0.04))
        self._target_attr = cfg.params.get("target_attr", "_ee_target_pos_h")
        if not hasattr(env, self._target_attr):
            setattr(env, self._target_attr, torch.zeros(env.num_envs, 3, device=env.device))
        # Reuse _EEHFrameComputer shared with ee_current_pos_h / ee_error_h obs terms.
        # Avoids recomputing yaw_quat + quat_apply_inverse when both obs and reward are active.
        hframe_attr = f"_ee_h_frame_{self._ee_body_id}"
        if not hasattr(env, hframe_attr):
            setattr(env, hframe_attr, _EEHFrameComputer(self.asset, self._ee_body_id, env))
        self._ee_h_computer: _EEHFrameComputer = getattr(env, hframe_attr)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        current_h = self._ee_h_computer.get(env)                  # (B, 3) — cached per step
        error_h = getattr(env, self._target_attr) - current_h     # (B, 3)
        reward = torch.exp(-error_h.square().sum(dim=-1) * self._inv_sigma_sq)  # (B,)
        if self._gate_command is not None:
            cmd = env.command_manager.get_command(self._gate_command)  # (B, ≥2)
            speed_sq = cmd[:, 0].square() + cmd[:, 1].square()
            gate = torch.exp(-speed_sq * self._gate_inv_sigma_sq)
            reward = reward * gate
        return reward


class ee_tracking_coarse_batched:
    """Batched EE tracking reward: mean over K end-effectors, shape (B,).

    Reads env._ee_targets_h (B, K, 3) and computes per-EE Gaussian reward,
    then averages across EEs. Single call replaces K separate ee_tracking_coarse terms.

    Params:
        ee_body_names:     List of K MuJoCo body names (must match EETargetsResample order).
        std:               Gaussian σ in meters (default 0.05 = 5cm).
        gate_command_name: Velocity command key for gate (optional).
        gate_sigma_sq:     Gate width σ² (default 0.04).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset = env.scene["robot"]
        body_ids, _ = self.asset.find_bodies(cfg.params["ee_body_names"])
        self._body_ids = body_ids
        self._K = len(body_ids)
        sigma = float(cfg.params.get("std", 0.05))
        self._inv_sigma_sq = 1.0 / sigma ** 2
        self._gate_command: str | None = cfg.params.get("gate_command_name", None)
        self._gate_inv_sigma_sq = 1.0 / float(cfg.params.get("gate_sigma_sq", 0.04))
        if not hasattr(env, "_ee_targets_h"):
            env._ee_targets_h = torch.zeros(env.num_envs, self._K, 3, device=env.device)
        from tasks.humanoid_velocity.observation import _ee_pos_in_h
        self._ee_pos_in_h = _ee_pos_in_h

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        current_h = self._ee_pos_in_h(self.asset, self._body_ids, env)    # (B, K, 3)
        error_h = env._ee_targets_h - current_h                            # (B, K, 3)
        reward = torch.exp(
            -error_h.square().sum(dim=-1) * self._inv_sigma_sq
        ).mean(dim=-1)                                                      # (B,)
        if self._gate_command is not None:
            cmd = env.command_manager.get_command(self._gate_command)
            speed_sq = cmd[:, 0].square() + cmd[:, 1].square()
            reward = reward * torch.exp(-speed_sq * self._gate_inv_sigma_sq)
        return reward


# =============================================================================
# Gait Phase Rewards
# =============================================================================


@torch.jit.script
def _compute_tilt_mag(proj_grav: torch.Tensor) -> torch.Tensor:
    """Tilt magnitude from projected gravity: |roll| + |pitch|.

    Shared by foot_phase_contact_match and foot_stance_slip_penalty. Cached on env
    via _tilt_mag_cache / _tilt_mag_cache_step to avoid duplicate computation when
    both reward terms are active in the same step.
    """
    roll  = torch.atan2(proj_grav[:, 1], -proj_grav[:, 2])
    pitch = torch.atan2(proj_grav[:, 0], torch.sqrt(proj_grav[:, 1].square() + proj_grav[:, 2].square()))
    return roll.abs() + pitch.abs()


def _resolve_foot_slots(sensor, device: torch.device) -> torch.Tensor:
    """Return a LongTensor([foot_L_slot, foot_R_slot]) from a ContactSensor.

    sensor._slots lists fields grouped by primary body in declaration order;
    unique primary names are the column indices of data.found.
    Pre-built as a tensor to avoid per-step Python list indexing.

    NOTE: relies on ContactSensor private attributes _slots and .primary_name.
    If mjlab changes this internal layout, update this helper accordingly.
    """
    primary_order: list[str] = []
    for s in sensor._slots:
        if s.primary_name not in primary_order:
            primary_order.append(s.primary_name)
    return torch.tensor(
        [primary_order.index("foot_L"), primary_order.index("foot_R")],
        device=device, dtype=torch.long,
    )


class foot_phase_contact_match:
    """Reward foot contact state matching the gait phase schedule.

    r = exp(-match_error) * is_moving * stability ∈ [0, 1] per env.

    match_error = |target_contact - actual_contact|.sum ∈ {0, 1, 2}.
    Anti-phase (φ_R = φ_L + 0.5): one foot always in target-stance at stance_ratio=0.55.

    Stability gate: exp(-tilt_mag/tilt_thresh) suppresses reward during genuine tilt events.
    ang_vel_std param retained for compatibility but no longer used in the gate.
    Normal gait: tilt < 0.05 rad → gate≈1 → rewards fire.
    Push-induced tilt > 0.15 rad: gate≈0 → rewards suppressed.
    ang_vel-only gate was rejected (false suppress during gait bob in 21-23% of timesteps).
    tilt_thresh=0.25 rad (≈14°) chosen to catch pushes while letting normal gait through.

    Foot slot ordering resolved explicitly via robot.find_bodies — not assumed from
    sensor declaration order.

    Params:
        sensor_name:    "feet_ground_contact"
        command_name:   "twist"
        lin_thresh:     0.05 m/s (default) — must match GaitPhase
        yaw_thresh:     0.05 rad/s (default) — must match GaitPhase
        stance_ratio:   0.55 (default)
        ang_vel_std:    0.5 rad/s (default)
        tilt_thresh:    0.25 rad (default)
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        from tasks.humanoid_velocity.observation import gait_is_moving as _gait_is_moving
        self._gait_is_moving = _gait_is_moving
        self.sensor = env.scene[cfg.params["sensor_name"]]
        self.robot = env.scene["robot"]
        self.cmd_name = cfg.params["command_name"]
        self.lin_thresh = float(cfg.params.get("lin_thresh", 0.05))
        self.yaw_thresh = float(cfg.params.get("yaw_thresh", 0.05))
        self.stance_ratio = float(cfg.params.get("stance_ratio", 0.55))
        self.inv_ang_vel_std = 1.0 / float(cfg.params.get("ang_vel_std", 0.5))
        self.tilt_thresh = float(cfg.params.get("tilt_thresh", 0.25))
        self.foot_slots = _resolve_foot_slots(self.sensor, env.device)
        # Phase offsets [L=0, R=+0.5] — broadcast pattern matches foot_stance_slip_penalty.
        self.phase_offsets = torch.tensor([0.0, 0.5], device=env.device)
        # Shared tilt magnitude cache: computed once per step, reused by foot_stance_slip_penalty.
        if not hasattr(env, '_tilt_mag_cache'):
            env._tilt_mag_cache = torch.zeros(env.num_envs, device=env.device)
            env._tilt_mag_cache_step = -1

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if not hasattr(env, '_gait_phase'):
            return torch.zeros(env.num_envs, device=env.device)

        # Broadcast: [B,1] + [2] → [B,2]; avoids torch.stack allocation.
        phases = (env._gait_phase.unsqueeze(-1) + self.phase_offsets) % 1.0
        target_contact = (phases <= self.stance_ratio).float()            # [B, 2]

        actual_contact = (self.sensor.data.found[:, self.foot_slots] > 0).float()  # [B, 2]
        match_error = (target_contact - actual_contact).abs().sum(dim=-1)
        reward = torch.exp(-match_error)

        cmd_vel = env.command_manager.get_command(self.cmd_name)
        is_moving = self._gait_is_moving(cmd_vel, self.lin_thresh, self.yaw_thresh).float()

        # Pure tilt gate: suppresses phase rewards when robot is genuinely tilted.
        # ang_vel-only gate rejected: fires in 21-23% of normal walking (gait bob).
        # tilt_gate at 0.25 rad: fires <1% during normal gait, 100% during push events.
        # Cached on env: foot_stance_slip_penalty reuses this if active in the same step.
        if env._tilt_mag_cache_step != env.common_step_counter:
            env._tilt_mag_cache.copy_(_compute_tilt_mag(self.robot.data.projected_gravity_b))
            env._tilt_mag_cache_step = env.common_step_counter

        stability = torch.exp(-env._tilt_mag_cache / self.tilt_thresh)

        return reward * is_moving * stability


class foot_stance_slip_penalty:
    """Penalize foot horizontal slip during prescribed stance phase.

    Returns positive magnitude; weight should be negative in cfg to make it a penalty.
    r = sum(target_stance * actual_contact * |foot_xy_vel|) * is_moving * stability ≥ 0

    Double-gated by target_stance (phase schedule) AND actual_contact (sensor):
    no penalty if foot is airborne even when phase says stance — allows terrain
    adaptation and brief corrective lifts without penalty.

    site_lin_vel_w: horizontal xy velocity at foot contact sites. Consistent with
    foot_impact_velocity and feet_clearance_tanh. z excluded: compliance is intentional.

    Stability gate: same pure tilt gate as foot_phase_contact_match — suppressed when robot
    is genuinely tilted. See foot_phase_contact_match docstring for rationale.

    TODO (reward hacking): Three known exploit vectors:
      1. Hover exploit: policy can zero slip penalty by lifting stance foot (actual_contact=0),
         costing only exp(-1)≈0.37 drop in foot_phase_contact_match. Break-even at
         |w_slip|*foot_slip > w_contact_match*0.63 — easily triggered at fast walking.
         Fix: also penalise target_stance*(1-actual_contact)*|foot_xy_vel|, or remove the
         actual_contact gate and accept all stance-phase foot velocity as penalty.
      2. Stability gate escape: deliberate torso wobble drives tilt_gate below 1.0,
         suppressing penalty. See foot_phase_contact_match for mitigations.
      3. Swing-phase drag: target_stance=0 so dragging foot incurs no slip penalty;
         foot_phase_contact_match only penalises contact existence, not foot velocity.
         Fix: add a swing-phase foot-velocity penalty, or confirm contact sensor fires.

    Params:
        sensor_name:    "feet_ground_contact"
        asset_cfg:      SceneEntityCfg with site_names=("foot_L_contact","foot_R_contact")
                        Must be passed explicitly — no default to avoid circular import.
        command_name:   "twist"
        lin_thresh:     0.05 m/s (default) — must match GaitPhase
        yaw_thresh:     0.05 rad/s (default) — must match GaitPhase
        stance_ratio:   0.55 (default)
        ang_vel_std:    0.5 rad/s (default)
        tilt_thresh:    0.25 rad (default)
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        from tasks.humanoid_velocity.observation import gait_is_moving as _gait_is_moving
        self._gait_is_moving = _gait_is_moving
        self.sensor = env.scene[cfg.params["sensor_name"]]
        self.robot = env.scene["robot"]
        asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.site_ids = asset_cfg.site_ids    # [foot_L_contact, foot_R_contact]
        self.cmd_name = cfg.params["command_name"]
        self.lin_thresh = float(cfg.params.get("lin_thresh", 0.05))
        self.yaw_thresh = float(cfg.params.get("yaw_thresh", 0.05))
        self.stance_ratio = float(cfg.params.get("stance_ratio", 0.55))
        self.inv_ang_vel_std = 1.0 / float(cfg.params.get("ang_vel_std", 0.5))
        self.tilt_thresh = float(cfg.params.get("tilt_thresh", 0.25))
        self.foot_slots_t = _resolve_foot_slots(self.sensor, env.device)
        self.phase_offsets = torch.tensor([0.0, 0.5], device=env.device)  # L=0, R=+0.5
        # Shared tilt magnitude cache: computed once per step, reused from foot_phase_contact_match.
        if not hasattr(env, '_tilt_mag_cache'):
            env._tilt_mag_cache = torch.zeros(env.num_envs, device=env.device)
            env._tilt_mag_cache_step = -1

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if not hasattr(env, '_gait_phase'):
            return torch.zeros(env.num_envs, device=env.device)

        # Broadcast phase offsets [0, 0.5] across batch: [B,1] + [2] → [B,2]
        phases = (env._gait_phase.unsqueeze(-1) + self.phase_offsets) % 1.0
        target_stance = (phases <= self.stance_ratio).float()                         # [B, 2]

        actual_contact = (self.sensor.data.found[:, self.foot_slots_t] > 0).float()  # [B, 2]
        foot_vel_xy = self.asset.data.site_lin_vel_w[:, self.site_ids, :2]           # [B, 2, 2]
        foot_slip = torch.linalg.vector_norm(foot_vel_xy, dim=-1)                     # [B, 2]
        penalty = (target_stance * actual_contact * foot_slip).sum(dim=-1)            # [B]

        cmd_vel = env.command_manager.get_command(self.cmd_name)
        is_moving = self._gait_is_moving(cmd_vel, self.lin_thresh, self.yaw_thresh).float()

        # Pure tilt gate: same as foot_phase_contact_match.
        # Reads from shared env cache; no recompute if foot_phase_contact_match already ran this step.
        if env._tilt_mag_cache_step != env.common_step_counter:
            env._tilt_mag_cache.copy_(_compute_tilt_mag(self.robot.data.projected_gravity_b))
            env._tilt_mag_cache_step = env.common_step_counter

        stability = torch.exp(-env._tilt_mag_cache / self.tilt_thresh)

        return penalty * is_moving * stability


# =============================================================================
# Low-Velocity Walk Initiation
# =============================================================================


def track_linear_velocity_relative(
    env: ManagerBasedRlEnv,
    std_rel: float,
    std_min: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std_z: float | None = None,
    std_stand: float | None = None,
) -> torch.Tensor:
    """Velocity tracking with command-proportional std.

    Shared-std (legacy, std_z is None):
        std_eff = max(std_rel * ||cmd_xy||, std_min)
        reward  = exp(-(||v_err_xy||² + v_z²) / std_eff²)

    Decoupled (std_z given): commanded XY error and UNcommanded vertical bob get
    separate Gaussians, so the XY floor can be sharp without over-penalising bob:
        std_xy  = max(std_rel * ||cmd_xy||, std_min)   # std_min is now an XY-only floor
        reward  = exp(-(||v_err_xy||²/std_xy²  +  v_z²/std_z²))

    Replaces mjlab track_linear_velocity (fixed std). Parameterization convention
    matches mjlab: divisor is std², not 2*std².

    Motivation: fixed std=0.5 gives 6.6× weaker initiation gradient at cmd=0.2
    vs cmd=1.0 → standing is locally optimal at low commands. Proportional std
    equalises gradient across all command magnitudes.

    Why decouple (grid dead-zone fix, memory/grid_low_command_deadzone.md): the
    shared std forced std_min ≥ 0.3 ONLY to keep the ~0.2 m/s gait vz-bob from
    scoring exp(-0.04/0.09) ≈ 0.64 instead of a punishing ≈0.17 at std_min=0.15.
    (Precision on the "decoupling at the old floor is a no-op" claim: that holds
    where the floor BINDS, ||cmd_xy|| < std_min/std_rel = 0.6, since both terms
    then divide by std_min². Above 0.6 the shared form scales vz by the relative
    std while the decoupled form pins it at std_z, so they differ. The no-op covers
    the region the lever was argued over, not the whole range —
    probe/track_std_check.py tests 2 and 2b.)
    That same 0.3 floor makes XY tracking toothless at low cmd (walk barely out-
    scores stand on the linear term, and the yaw penalty cancels it). Splitting vz
    onto its own fixed std_z=0.3 removes that coupling, so std_min can drop to ~0.1
    on the XY axis and widen the walk-vs-stand linear gap past the yaw cancellation.
    Reward reads root_link_lin_vel_b (true sim velocity, not the noisy estimate),
    so sharpening the XY floor carries no noise-chasing risk. The earlier rejection
    of std_min=0.1 (v2ybsk wave-20) was for the SHARED std, which the decoupling
    dissolves — new premise, not a re-test. std_z is None reproduces the exact old
    formula for every legacy caller (v30 chain, g1); only opt-in callers decouple.

    Why std_stand (the zero-command atom, requires std_z): with std_rel=0.5 the floor
    binds wherever ||cmd_xy|| < 2*std_min, so at std_min=0.1 its binding region is
    exactly [0, 0.2) — the dead zone TOGETHER WITH exact zero. One constant then serves
    two regimes that want opposite things. On (0, 0.2) the sharp floor breaks the
    walk-versus-stand tie and is the whole mechanism, worth +19% on ±0.20 disp. At
    cmd == 0 there is no tie: standing is both the target and the optimum, so the
    sharpening buys nothing and only adds gradient pressure against the irreducible
    gait ripple. Measured cost: cmd-0 joint_power 9.38 vs 6.05, the invariant that
    kept ...GridDecoupledTrack a lever rather than the base despite it being the only
    policy on that lineage to beat the ybsk reference on the dead zone.

    Zero command is a DISCRETE atom, not the limit of small command — standing envs
    are zeroed exactly in the command term's _update_command, while grid cells sample
    strictly inside their cell — so the predicate is `cmd_norm == 0` with no epsilon
    and no threshold to pick. Setting std_stand to the pre-decoupling floor (0.3)
    makes the standing population BIT-IDENTICAL to the old shared-std formula,
    exp(-(xy² + vz²)/0.09), i.e. it restores the exact configuration that measured
    the 6.05 calm rather than introducing a new operating point.

    REJECTED here, recorded so it is not re-proposed: swapping the exponential for a
    heavy-tailed kernel (inverse-quadratic 1/(1+e²), e = err/std) to escape "exponential
    saturation". There is no saturation to escape. The relative std makes e SELF-NORMALISING:
    a policy standing while ||cmd|| is commanded has err = ||cmd|| and std = std_rel·||cmd||,
    so e = 1/std_rel = 2 for EVERY command above the floor, and below the floor e = ||cmd||/
    std_min is smaller still. The standing policy therefore never leaves e <= 2, where exp is
    nowhere near flat. Measured on probe/track_std_check.py: the kernel swap buys 2.2x
    gradient at the standing point, not the orders of magnitude the saturation story predicts,
    and it RAISES standing's own score at cmd 0.5 from 0.018 to 0.2 -- shrinking the
    walk-versus-stand value gap that the std levers exist to widen. Wrong tool, and the
    self-normalisation is the reason.
    """
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise RuntimeError(f"Command '{command_name}' not found in command_manager.")

    cmd_xy = command[:, :2]
    actual_vel = robot.data.root_link_lin_vel_b
    cmd_norm = torch.linalg.vector_norm(cmd_xy, dim=1)
    xy_err_sq = torch.sum(torch.square(cmd_xy - actual_vel[:, :2]), dim=1)
    z_err_sq = torch.square(actual_vel[:, 2])

    if std_z is None:
        std_eff = torch.clamp(std_rel * cmd_norm, min=std_min)
        return torch.exp(-(xy_err_sq + z_err_sq) / std_eff.square())

    std_xy = torch.clamp(std_rel * cmd_norm, min=std_min)
    if std_stand is not None:
        std_xy = torch.where(cmd_norm == 0.0, torch.full_like(std_xy, std_stand), std_xy)
    return torch.exp(-(xy_err_sq / std_xy.square() + z_err_sq / (std_z * std_z)))


def track_angular_velocity_relative(
    env: ManagerBasedRlEnv,
    std_rel: float,
    std_min: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw velocity tracking with command-proportional std.

    std_eff = max(std_rel * |cmd_yaw|, std_min)
    reward  = exp(-(yaw_err²) / std_eff²)

    Analogous to track_linear_velocity_relative for XY. Fixes yaw dead zone:
    fixed std=sqrt(0.5) gives 0.980 standing reward at cmd=0.1 rad/s.
    With std_rel=0.5, std_min=0.1 rad/s: standing at cmd=0.1 scores 0.368.

    Parameterisation convention matches mjlab: divisor is std², not 2*std².
    """
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise RuntimeError(f"Command '{command_name}' not found in command_manager.")

    cmd_yaw = command[:, 2]
    actual_wyaw = robot.data.root_link_ang_vel_b[:, 2]
    std_eff = torch.clamp(std_rel * cmd_yaw.abs(), min=std_min)
    yaw_err_sq = torch.square(cmd_yaw - actual_wyaw)
    return torch.exp(-yaw_err_sq / std_eff.square())


def feet_air_time_smooth_gate(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str = "twist",
    cmd_low: float = 0.05,
    cmd_high: float = 0.25,
) -> torch.Tensor:
    """Mirror of mjlab feet_air_time with a linear-ramp cmd gate.

    mjlab original (rewards.py:236):
        scale = (||cmd_xy|| + |cmd_yaw| > command_threshold).float()  # binary
    This version:
        scale = clamp((total_cmd - cmd_low) / (cmd_high - cmd_low), 0, 1)  # ramp

    Motivation: the hard gate zeroes air_time reward for all ||cmd_xy|| < 0.25,
    removing positive liftoff gradient in the dead zone (0.05–0.25 m/s).

    Preserves the Metrics/air_time_mean log for monitoring parity with mjlab.
    """
    contact_sensor = env.scene[sensor_name]
    current_air_time = contact_sensor.data.current_air_time
    assert current_air_time is not None

    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    air_time_reward = torch.sum(in_range.float(), dim=1)

    in_air_mask = (current_air_time > 0).float()
    n_in_air = in_air_mask.sum()
    mean_air_time = (current_air_time * in_air_mask).sum() / torch.clamp(n_in_air, min=1)
    env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    total_cmd = torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
    gate_scale = torch.clamp((total_cmd - cmd_low) / (cmd_high - cmd_low), 0.0, 1.0)
    return air_time_reward * gate_scale


class com_support_metric:
    """Log whole-body CoM projection distance to foot-site midpoint.

    Purpose: diagnose whether CoM/support deviation predicts falls or torque spikes before
    adding reward pressure. Uses asset.data.data (mjwarp Data object) for subtree_com access.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.site_ids = asset_cfg.site_ids
        self.root_body_id = self.asset.data.indexing.root_body_id

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        foot_pos = self.asset.data.site_pos_w[:, self.site_ids, :2]
        support_center = foot_pos.mean(dim=1)
        com_xy = self.asset.data.data.subtree_com[:, self.root_body_id, :2]
        err = torch.linalg.norm(com_xy - support_center, dim=-1)
        if "log" in env.extras:
            env.extras["log"]["Metrics/com_support_error_mean"] = err.mean()
        return err


class com_support_reward:
    """Reward for keeping the projected CoM near the center of the foot support polygon.

    Uses a Gaussian kernel exp(-deviation^2 / std^2) to provide a smooth gradient.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.site_ids = asset_cfg.site_ids
        self.std = float(cfg.params.get("std", 0.12))
        self.root_body_id = self.asset.data.indexing.root_body_id

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        foot_pos = self.asset.data.site_pos_w[:, self.site_ids, :2]
        support_center = foot_pos.mean(dim=1)
        com_xy = self.asset.data.data.subtree_com[:, self.root_body_id, :2]
        deviation_sq = torch.sum(torch.square(com_xy - support_center), dim=-1)
        return torch.exp(-deviation_sq / (self.std ** 2))


def track_integral_error(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
    """Reward driving the bounded integral tracking error to zero (PI integral term).

    reward = exp(−‖I‖² / std²), I = IntegralErrorVelocityCommand.integral_error (3D).

    Supplies the time pressure the instantaneous velocity-tracking reward lacks at small
    commands: standing while a command is unmet accrues debt in I, so the policy must
    eventually step/turn to discharge it → delivers the average sub-v_min rate and kills
    the low-speed dead zone, without changing the velocity command interface.
    """
    I = env.command_manager.get_term(command_name).integral_error
    return torch.exp(-torch.sum(I * I, dim=1) / (std * std))

