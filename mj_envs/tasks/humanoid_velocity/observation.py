"""Observation term helpers: IMU model, COM, joint/arm observations, EE tracking."""

from __future__ import annotations

import math as _math
from dataclasses import dataclass

import xxhash
import torch
import torch.nn.functional as F
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import yaw_quat, quat_apply_inverse
from asset_zoo.humanoid_v21.generate_leg_joint_pos import (
    LEG_JOINT_NAMES,
    vertical_translate_lower_body_joints_batch,
)
from mjlab_util.optimized_entity import OptimizedDelayBuffer as DelayBuffer

_TWO_PI = 2.0 * _math.pi


# =============================================================================
# Gait Phase Clock
# =============================================================================


@torch.jit.script
def gait_is_moving(cmd_vel: torch.Tensor, lin_thresh: float, yaw_thresh: float) -> torch.Tensor:
    """True [B] where command exceeds activation thresholds.

    Single definition shared by GaitPhase (obs) and both phase reward classes to
    guarantee identical is_moving semantics from one place.
    JIT-compiled; uses squared comparison to avoid sqrt.
    """
    return (
        (cmd_vel[:, 0].square() + cmd_vel[:, 1].square() > lin_thresh * lin_thresh) |
        (cmd_vel[:, 2].abs() > yaw_thresh)
    )


@torch.jit.script
def _update_gait_phase(
    phase: torch.Tensor,
    period: torch.Tensor,
    phase_updated_step: torch.Tensor,
    episode_length_buf: torch.Tensor,
    cmd_vel: torch.Tensor,
    step_dt: float,
    lin_thresh_sq: float,
    yaw_thresh: float,
    snap_window: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled gait phase update.

    Returns (new_phase, new_phase_updated_step). Caller reassigns env._gait_phase and
    env._phase_updated_step. Uses squared linear threshold to avoid sqrt.

    Multi-call guard: envs whose episode_length_buf == phase_updated_step are already
    up-to-date and are left unchanged (torch.where with needs_update mask).
    """
    needs_update = episode_length_buf != phase_updated_step
    is_moving = (
        (cmd_vel[:, 0].square() + cmd_vel[:, 1].square()) > lin_thresh_sq
    ) | (cmd_vel[:, 2].abs() > yaw_thresh)

    should_tick = is_moving | (phase > 1e-6)
    new_phase = (phase + (step_dt / period) * should_tick.float()) % 1.0

    snap_to_home = (~is_moving) & (new_phase < snap_window) & (phase > 1.0 - snap_window)
    new_phase = torch.where(snap_to_home, torch.zeros_like(new_phase), new_phase)

    new_phase = torch.where(needs_update, new_phase, phase)
    new_updated = torch.where(needs_update, episode_length_buf, phase_updated_step)
    return new_phase, new_updated


class GaitPhase:
    """Homing phase clock: sin/cos encoding of left and right foot gait phases.

    Output: [sin(2π·φ_L), cos(2π·φ_L), sin(2π·φ_R), cos(2π·φ_R)] ∈ R^4.
    Right leg is anti-phase: φ_R = (φ_L + 0.5) % 1.0.

    Phase dynamics (per control step):
        is_moving   = |cmd_xy|² > lin_thresh² OR |cmd_yaw| > yaw_thresh
        should_tick = is_moving OR (phase > 1e-6)   # homing: complete cycle after cmd=0
        d_phase     = (step_dt / period) * should_tick
        new_phase   = (phase + d_phase) % 1.0
        snap_to_home: when not moving AND new_phase < snap_window AND old_phase > (1 - snap_window)

    Multi-call guard: phase updated at most once per step via episode_length_buf comparison.
    Rewards read _gait_phase from the previous step (1-step lag, ~20ms — negligible vs
    gait period ~650ms). This is an approximation: reward coupling is not exact.

    Anti-phase identity: sin(2π(φ+0.5)) = −sin(2πφ), cos(2π(φ+0.5)) = −cos(2πφ).
    R-phase output is the negation of L-phase — halves trig ops.

    Design notes:
    - is_not_home threshold 1e-6 prevents stall: phase in (0, small] that is neither
      ticking (> large threshold) nor snapping (old_phase > 1-window).
    - snap_window is a heuristic tolerance covering max-step overshoot.
    - _gait_phase/_gait_period buffers allocated by randomize_gait_period event; fallback
      zeros returned until that event runs (guarded by _ready flag after first success).
    - Phase update lives here because mjlab has no per-step non-event callback.
    - _out (B,4) and _angle (B,) are pre-allocated; __call__ has zero tensor allocations.

    Params:
        period_min: float  — minimum gait period (s), used for snap_window (default 0.55)
        lin_thresh: float  — linear speed threshold (m/s) to activate gait (default 0.05)
        yaw_thresh: float  — yaw speed threshold (rad/s) to activate gait (default 0.05)
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        period_min = float(cfg.params.get("period_min", 0.55))
        self._snap_window = 2.0 * env.step_dt / period_min  # heuristic tolerance
        lin_thresh = float(cfg.params.get("lin_thresh", 0.05))
        self._lin_thresh_sq = lin_thresh * lin_thresh         # avoid sqrt in hot path
        self._yaw_thresh = float(cfg.params.get("yaw_thresh", 0.05))
        # Pre-allocated buffers: eliminate per-step allocations in __call__
        self._out   = torch.zeros(env.num_envs, 4, device=env.device)  # (B, 4) output
        self._angle = torch.zeros(env.num_envs,    device=env.device)  # (B,) 2π·φ_L
        # _ready: becomes True once randomize_gait_period has run; avoids hasattr every step
        self._ready = False

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if not self._ready:
            if not hasattr(env, '_gait_phase'):
                return self._out  # pre-allocated zeros; safe before randomize_gait_period fires
            self._ready = True

        cmd_vel = env.command_manager.get_command("twist")   # [B, 3]: [vx, vy, wz]
        env._gait_phase, env._phase_updated_step = _update_gait_phase(
            env._gait_phase, env._gait_period,
            env._phase_updated_step, env.episode_length_buf,
            cmd_vel, env.step_dt,
            self._lin_thresh_sq, self._yaw_thresh, self._snap_window,
        )

        # Anti-phase identity: sin/cos(2π(φ+0.5)) = −sin/cos(2πφ).
        # Compute L-phase only; R-phase is the negation — halves trig ops.
        self._angle.copy_(env._gait_phase).mul_(_TWO_PI)
        torch.sin(self._angle, out=self._out[:, 0])
        torch.cos(self._angle, out=self._out[:, 1])
        torch.neg(self._out[:, 0], out=self._out[:, 2])
        torch.neg(self._out[:, 1], out=self._out[:, 3])
        return self._out


def joint_pos_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute joint positions (radians). Normalizer learns the mean; no manual centering needed."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute joint velocities (rad/s). Default joint vel is zero so this equals joint_vel_rel."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


class joint_actuator_force:
    """Actuator forces in joint order. Passive joints return zero."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.data = env.scene[asset_cfg.name].data
        idx = self.data.indexing

        # Match joints to actuators: actuator_trnid[i, 0] = joint driven by actuator i
        actuator_joints = env.sim.model.actuator_trnid[idx.ctrl_ids, 0]
        match = (idx.joint_ids.unsqueeze(1) == actuator_joints.unsqueeze(0))

        self.actuated = match.any(dim=1)  # joints with actuators
        self.act_idx = match.int().argmax(dim=1)[self.actuated]  # actuator index per joint
        self.force = torch.zeros(env.num_envs, len(idx.joint_ids), device=self.data.joint_pos.device)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        self.force[:, self.actuated] = self.data.actuator_force[:, self.act_idx]
        return self.force


def foot_friction_gt(
    env: ManagerBasedRlEnv,
    left_cfg: SceneEntityCfg,
    right_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Ground-truth per-foot tangential friction coefficient (privileged critic obs).

    Returns (B, 2) = [left, right], the mean over each foot's collision capsules of
    ``geom_friction[:, :, 0]``. The `foot_friction` DR event (startup, per-geom independent
    Uniform(0.3, 1.5), 5 capsules/foot) sets a hidden per-episode dynamics latent the deploy
    actor never observes and a 1-frame Markov critic cannot infer (fixed at startup, no direct
    signal until a slip). Feeding it directly tests the asymmetric-critic "observe the latent"
    premise. Per-foot mean (not the raw 10-vector) because the 5 capsules tile one contact patch
    → the mean is the effective foot friction, and 2-dim keeps the paper story clean.

    geom_friction is per-env (B, ngeom, 3) after DR; geom ids resolved via indexing map exactly as
    the DR write path (event.randomize_foot_solimp): indexing.geom_ids[asset_cfg.geom_ids].
    """
    idx = env.scene["robot"].indexing
    lg = idx.geom_ids[left_cfg.geom_ids]
    rg = idx.geom_ids[right_cfg.geom_ids]
    f = env.sim.model.geom_friction[:, :, 0]                       # (B, ngeom) tangential
    return torch.stack([f[:, lg].mean(dim=1), f[:, rg].mean(dim=1)], dim=1)   # (B, 2)


# =============================================================================
# Simulated IMU Model
# =============================================================================


@torch.jit.script
def _compute_imu_corruption(
    R_mount: torch.Tensor,
    gravity: torch.Tensor,
    ang_vel: torch.Tensor,
    additive_bias: torch.Tensor,
    drift_accum: torch.Tensor,
    drift_std: torch.Tensor,  # Per-env tensor for domain randomization
    drift_beta: float,         # Gauss-Markov AR(1) coefficient
) -> tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled IMU corruption pipeline.

    Applies rotation, bias, and bounded drift to gravity and angular velocity.
    Uses Gauss-Markov process (AR(1)) for realistic bounded drift behavior.
    Returns: (signal_6d [grav(3)|ang_vel(3)], updated_drift_accum)
    Returns the full 6-dim signal to avoid a redundant slice→cat at the call site.
    """
    # 1. Rotate by mounting misalignment
    rot_gravity = torch.bmm(R_mount, gravity.unsqueeze(-1)).squeeze(-1)
    rot_ang_vel = torch.bmm(R_mount, ang_vel.unsqueeze(-1)).squeeze(-1)

    # Re-normalize gravity to preserve unit length
    rot_gravity = F.normalize(rot_gravity, dim=1)

    # 2. Concatenate to 6-dim signal
    signal = torch.cat([rot_gravity, rot_ang_vel], dim=1)

    # 3. Add additive bias (constant per episode)
    signal = signal + additive_bias

    # 4. Accumulate drift (bounded Gauss-Markov AR(1) process)
    # drift[t] = beta * drift[t-1] + noise
    # This prevents unbounded random walk while maintaining realistic autocorrelation
    drift_accum = drift_beta * drift_accum + torch.randn_like(drift_accum) * drift_std.unsqueeze(-1)
    signal = signal + drift_accum

    return signal, drift_accum


@dataclass
class IMUModelCfg:
    """Configuration for the simulated IMU model.

    Models 5 real-world IMU error sources:
      1. Mounting misalignment — static rotation per episode (Rodrigues)
      2. Additive bias — constant per-channel offset per episode (split accel/gyro)
      3. Drift — bounded Gauss-Markov process (AR(1) model)
      4. Transport delay — shared FIFO delay for all IMU channels
      5. Measurement noise — handled externally by ObservationTermCfg.noise

    Real IMU specs (1x baseline):
      - Mounting misalignment: 0.3% (~0.0047 rad = 0.27°)
      - Accel bias: 0.05 mg = 5e-5 g (fractional)
      - Gyro bias: 5.5 °/h = 2.67e-5 rad/s
      - Drift: ~6e-7 rad/s per step @ 50Hz
    """

    max_mount_angle: tuple[float, float] | float = (0.003, 0.015)  # rad, 5x margin: 0.17-0.86°
    accel_bias_range: tuple[float, float] | float = (5e-5, 2.5e-4)  # fractional g, 5x margin on 5e-5
    gyro_bias_range: tuple[float, float] | float = (2.67e-5, 1.5e-4)  # rad/s, 5-6x margin on 2.67e-5
    drift_std_range: tuple[float, float] | float = (5e-6, 5e-5)  # Gauss-Markov std, 5x margin
    drift_beta: float = 0.99  # AR(1) coefficient: 0.99 = ~100 step correlation (~2s @ 50Hz)
    delay_min_lag: int = 1
    delay_max_lag: int = 3
    delay_update_period: int = 0
    apply_corruption: bool = True  # False = passthrough (perfect IMU observations)


class IMUModel:
    """Simulated IMU that applies rotation, bias, drift, and delay.

    Corruption pipeline (order matters):
      raw gravity + ang_vel
        → rotate by R_mount
        → add additive_bias
        → add drift (bounded Gauss-Markov AR(1) process)
        → push into DelayBuffer → read delayed output

    Shared across observation terms via env._imu_model.
    """

    def __init__(self, cfg: IMUModelCfg, env: ManagerBasedRlEnv):
        self.cfg = cfg
        self._num_envs = env.num_envs
        self._device = env.device

        # Mounting misalignment: (num_envs, 3, 3)
        self._R_mount = torch.eye(3, device=self._device).unsqueeze(0).expand(self._num_envs, -1, -1).clone()

        # Additive bias: (num_envs, 6) — [gravity(3), ang_vel(3)]
        self._additive_bias = torch.zeros(self._num_envs, 6, device=self._device)

        # Drift accumulator: (num_envs, 6)
        self._drift_accum = torch.zeros(self._num_envs, 6, device=self._device)

        # Per-env drift std (for domain randomization): (num_envs,)
        self._drift_std = torch.zeros(self._num_envs, device=self._device)

        # Shared delay buffer for 6-dim concatenated signal
        self._delay_buffer = DelayBuffer(
            min_lag=cfg.delay_min_lag,
            max_lag=cfg.delay_max_lag,
            batch_size=self._num_envs,
            device=self._device,
            update_period=cfg.delay_update_period,
        )

        # Step counter guard: prevent double computation per step
        self._last_step = -1

        # Cached outputs
        self.cached_gravity = torch.zeros(self._num_envs, 3, device=self._device)
        self.cached_ang_vel = torch.zeros(self._num_envs, 3, device=self._device)

        # Initialize per-env state
        self.reset()

    def _sample_from_range(self, param: tuple[float, float] | float, n: int, dim: int = 1) -> torch.Tensor:
        """Sample uniform values from range or use constant.

        Args:
            param: Either (min, max) tuple or single float
            n: Number of samples (batch size)
            dim: Output dimension per sample

        Returns:
            (n, dim) tensor with uniform samples in [-val, +val] for each dimension
        """
        if isinstance(param, tuple):
            # Domain randomization: sample magnitude from range, then uniform direction
            range_vals = torch.rand(n, dim, device=self._device) * (param[1] - param[0]) + param[0]
            return (torch.rand(n, dim, device=self._device) * 2 - 1) * range_vals
        else:
            # Fixed value: uniform in [-param, +param]
            return (torch.rand(n, dim, device=self._device) * 2 - 1) * param

    def _sample_rodrigues_rotation(self, n: int, angles: torch.Tensor | None = None) -> torch.Tensor:
        """Sample random rotation matrices via Rodrigues formula.

        Args:
            n: Number of rotation matrices to sample.
            angles: Optional (n,) tensor of rotation angles. If None, samples from cfg.

        Returns:
            (n, 3, 3) rotation matrices.
        """
        # Random axis: uniform on unit sphere
        axis = torch.randn(n, 3, device=self._device)
        axis = F.normalize(axis, dim=1)

        # Random angle: use provided or sample from config
        if angles is None:
            if isinstance(self.cfg.max_mount_angle, tuple):
                # Sample from range
                angle = torch.rand(n, device=self._device) * \
                    (self.cfg.max_mount_angle[1] - self.cfg.max_mount_angle[0]) + \
                    self.cfg.max_mount_angle[0]
            else:
                # Use fixed value
                angle = torch.rand(n, device=self._device) * self.cfg.max_mount_angle
        else:
            angle = angles

        # Rodrigues: R = I + sin(θ)·K + (1 - cos(θ))·K²
        # where K is the skew-symmetric matrix of the axis
        K = torch.zeros(n, 3, 3, device=self._device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]

        sin_a = angle.sin().unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)
        cos_a = angle.cos().unsqueeze(-1).unsqueeze(-1)
        I = torch.eye(3, device=self._device).unsqueeze(0)

        return I + sin_a * K + (1 - cos_a) * (K @ K)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset IMU state for specified environments.

        Re-samples R_mount and additive_bias, zeros drift, resets delay buffer.
        """
        if env_ids is None:
            idx = slice(None)
            n = self._num_envs
        else:
            idx = env_ids
            n = len(env_ids)

        # Re-sample mounting rotation (handles both tuple and float configs)
        self._R_mount[idx] = self._sample_rodrigues_rotation(n)

        # Re-sample additive bias: split accel (channels 0-2) and gyro (channels 3-5)
        accel_bias = self._sample_from_range(self.cfg.accel_bias_range, n, dim=3)
        gyro_bias = self._sample_from_range(self.cfg.gyro_bias_range, n, dim=3)
        self._additive_bias[idx] = torch.cat([accel_bias, gyro_bias], dim=1)

        # Re-sample drift_std per environment
        self._drift_std[idx] = self._sample_from_range(self.cfg.drift_std_range, n, dim=1).abs().squeeze(-1)

        # Zero drift accumulator
        self._drift_accum[idx] = 0.0

        # Reset delay buffer
        self._delay_buffer.reset(batch_ids=env_ids)

        # Reset cached outputs for reset envs
        self.cached_gravity[idx] = 0.0
        self.cached_ang_vel[idx] = 0.0

    def compute_once(self, env: ManagerBasedRlEnv):
        """Compute IMU outputs, guarded to run once per env step.

        Reads projected_gravity and imu_ang_vel from env, applies the corruption
        pipeline, and caches the results in cached_gravity and cached_ang_vel.
        """
        step = env.common_step_counter
        if step == self._last_step:
            return  # Already computed this step
        self._last_step = step

        asset = env.scene["robot"]

        # Raw ground truth
        gravity = asset.data.projected_gravity_b  # (num_envs, 3)
        ang_vel = env.scene["robot/imu_ang_vel"].data  # (num_envs, 3)

        if not self.cfg.apply_corruption:
            # Passthrough: no corruption (perfect observations)
            self.cached_gravity = gravity
            self.cached_ang_vel = ang_vel
            return

        # Apply corruption pipeline (JIT-optimized for 1.55x speedup)
        # Returns full (B,6) signal directly — avoids a redundant slice→cat
        signal, self._drift_accum = _compute_imu_corruption(
            self._R_mount, gravity, ang_vel,
            self._additive_bias, self._drift_accum, self._drift_std, self.cfg.drift_beta
        )

        # Push through delay buffer and read delayed output
        self._delay_buffer.append(signal)
        delayed = self._delay_buffer.compute()

        # Cache output slices
        self.cached_gravity = delayed[:, :3]
        self.cached_ang_vel = delayed[:, 3:]


class imu_projected_gravity:
    """Observation term: projected gravity from simulated IMU.

    Creates or reuses env._imu_model. Owns reset (calls IMUModel.reset).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        if not hasattr(env, "_imu_model"):
            env._imu_model = IMUModel(cfg.params["imu_cfg"], env)
        self._imu = env._imu_model

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        self._imu.compute_once(env)
        return self._imu.cached_gravity

    def reset(self, env_ids: torch.Tensor | None = None):
        self._imu.reset(env_ids)


class imu_ang_vel:
    """Observation term: angular velocity from simulated IMU.

    Shares env._imu_model with imu_projected_gravity if both are active. If used alone,
    owns the reset. If used alongside imu_projected_gravity, double-reset is harmless
    (both calls just re-sample valid random values).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        if not hasattr(env, "_imu_model"):
            env._imu_model = IMUModel(cfg.params["imu_cfg"], env)
        self._imu = env._imu_model

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        self._imu.compute_once(env)
        return self._imu.cached_ang_vel

    def reset(self, env_ids: torch.Tensor | None = None):
        self._imu.reset(env_ids)


# =============================================================================
# COM Observation
# =============================================================================


@wp.kernel
def compute_robot_com_b_kernel(
    subtree_com_w: wp.array2d(dtype=float),
    root_pos_w: wp.array2d(dtype=float),
    root_quat_w: wp.array2d(dtype=float),
    root_com_vel_w: wp.array2d(dtype=float),
    obs_out: wp.array2d(dtype=float),
):
    tid = wp.tid()

    # 1. Load data
    com_w = wp.vec3(subtree_com_w[tid, 0], subtree_com_w[tid, 1], subtree_com_w[tid, 2])
    pos_w = wp.vec3(root_pos_w[tid, 0], root_pos_w[tid, 1], root_pos_w[tid, 2])
    quat_w = wp.quat(root_quat_w[tid, 1], root_quat_w[tid, 2], root_quat_w[tid, 3], root_quat_w[tid, 0])  # (x,y,z,w) for warp
    vel_w = wp.vec3(root_com_vel_w[tid, 0], root_com_vel_w[tid, 1], root_com_vel_w[tid, 2])

    # 2. System COM offset in base frame (joint-configuration-dependent, no global pos needed)
    offset_w = com_w - pos_w
    q_inv = wp.quat_inverse(quat_w)  # rotates world -> base
    pos_b = wp.quat_rotate(q_inv, offset_w)

    # 3. COM velocity in base frame
    vel_b = wp.quat_rotate(q_inv, vel_w)

    # 4. Store interleaved [pos, vel]
    obs_out[tid, 0] = pos_b[0]
    obs_out[tid, 1] = pos_b[1]
    obs_out[tid, 2] = pos_b[2]
    obs_out[tid, 3] = vel_b[0]
    obs_out[tid, 4] = vel_b[1]
    obs_out[tid, 5] = vel_b[2]


class robot_com_b:
    """Fused observation: whole-robot COM offset and velocity in base frame.

    Position (3D): R_root⁻¹ × (subtree_com[root] − xpos[root])
        Displacement of the whole-robot COM from the root joint origin, expressed in the
        root body frame. Purely joint-configuration-dependent; no global position or
        orientation needed. Identical across training, sim deployment, and real robot.

    Velocity (3D): R_root⁻¹ × cvel[root, 3:6]
        Root body velocity expressed at the subtree_com reference point, in root body frame.
        Includes locomotion speed; missing joint-induced COM velocity contributions
        (acceptable approximation — joint contribution is small vs. locomotion speed).
        Real-robot gap: approximated by finite-diff of the position term (joint-only,
        no locomotion speed). Accepted, analogous to base_lin_vel = 0 on hardware.

    Returns: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]
        self.num_envs = env.num_envs
        self.device = env.device

        # Pre-allocate interleaved output buffer (B, 6)
        self._obs_interleaved = torch.zeros(self.num_envs, 6, device=self.device)
        self._obs_interleaved_wp = wp.from_torch(self._obs_interleaved)

        # Pre-cache Warp arrays as LIVE views into the mjwarp data buffers.
        # CRITICAL: only direct slices of mjwarp tensors (data.data.*) are live views.
        # Properties like data.root_link_pos_w call torch.cat() → return new tensors each
        # time → wp.from_torch() on them would freeze the value at init. Always use raw buffers.
        raw = self.asset.data.data  # mjwarp Data object
        root_body_id = self.asset.data.indexing.root_body_id
        self._subtree_com_w_wp = wp.from_torch(raw.subtree_com[:, root_body_id])  # (B,3) live
        self._root_pos_w_wp    = wp.from_torch(raw.xpos[:, root_body_id])          # (B,3) live
        self._root_quat_w_wp   = wp.from_torch(raw.xquat[:, root_body_id])         # (B,4) live
        self._root_cvel_lin_wp = wp.from_torch(raw.cvel[:, root_body_id, 3:6])     # (B,3) live

        self._wp_stream = wp.stream_from_torch(torch.cuda.current_stream(self.device))

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        wp.launch(
            kernel=compute_robot_com_b_kernel,
            dim=self.num_envs,
            inputs=[
                self._subtree_com_w_wp,
                self._root_pos_w_wp,
                self._root_quat_w_wp,
                self._root_cvel_lin_wp,
                self._obs_interleaved_wp,
            ],
            device=self.device,
            stream=self._wp_stream,
        )
        return self._obs_interleaved


# =============================================================================
# Arm Observation
# =============================================================================


class target_arm_joint_pos:
    """Observation term: current arm/camera joint reference for the residual tracking policy.

    Two sources (resolved once at init):
      - command_name (preferred): a joint-reference CommandTerm; reads ``.command``.
      - arm_action_name (legacy): a reference-baked action term; reads ``.current_target``.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command_name = cfg.params.get("command_name", None)
        self._from_command = command_name is not None
        if self._from_command:
            self._term = env.command_manager.get_term(command_name)
        else:
            arm_action_name = cfg.params.get("arm_action_name", "joint_pos_arms")
            self._term = env.action_manager._terms[arm_action_name]

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if self._from_command:
            return self._term.command
        return self._term.current_target


class joint_target:
    """Observation term: lower-body joint target derived from the relative-height command.

    The command is a relative height offset in meters. This term maps the
    offset to a vertical translation target, converts that target to leg joint
    angles, and caches the result on env so actor/critic can share one tensor.
    The returned tensor uses LEG_JOINT_NAMES order.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._command_name = cfg.params.get("command_name", "rel_height")
        self._nominal_z_travel_mm = float(cfg.params.get("nominal_z_travel_mm", -40.0))
        self._target_attr = cfg.params.get("target_attr", "_joint_target")
        self._step_attr = f"{self._target_attr}_step"
        self._joint_names = tuple(cfg.params.get("joint_names", LEG_JOINT_NAMES))

        if not hasattr(env, self._target_attr):
            setattr(env, self._target_attr, torch.zeros(env.num_envs, len(self._joint_names), device=env.device))
        if not hasattr(env, self._step_attr):
            setattr(env, self._step_attr, -1)
        self._target = getattr(env, self._target_attr)

    def _update_target(self, env: ManagerBasedRlEnv) -> None:
        command = env.command_manager.get_command(self._command_name)
        z_travel = torch.clamp(
            self._nominal_z_travel_mm + command[:, 0] * 1000.0,
            -150.0,
            0.0,
        )
        self._target.copy_(vertical_translate_lower_body_joints_batch(z_travel))
        setattr(env, self._step_attr, env.common_step_counter)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if getattr(env, self._step_attr) != env.common_step_counter:
            self._update_target(env)
        return self._target


# =============================================================================
# End-Effector Observations (Horizontal Frame)
# =============================================================================


class _EEHFrameComputer:
    """Compute-once per step: single EE position in the pelvis horizontal frame.

    Shared between ee_current_pos_h and ee_error_h for the same EE body to avoid
    computing yaw_quat + quat_apply_inverse twice when both terms are active.
    Stored on env as `_ee_h_frame_{body_id}` so any number of obs terms can reuse it.

    Pre-allocates _offset_w (B,3) and _cache (B,3); zero new tensor allocations per
    step except for yaw_quat and quat_apply_inverse which have no out= API.
    """

    def __init__(self, asset, body_id: int, env: ManagerBasedRlEnv):
        self._asset = asset
        self._body_id = body_id
        self._cache    = torch.zeros(env.num_envs, 3, device=env.device)
        self._offset_w = torch.zeros(env.num_envs, 3, device=env.device)  # reused each step
        self._last_step = -1

    def get(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        if env.common_step_counter != self._last_step:
            self._last_step = env.common_step_counter
            torch.sub(
                self._asset.data.body_link_pos_w[:, self._body_id, :],
                self._asset.data.root_link_pos_w,
                out=self._offset_w,
            )
            self._cache.copy_(
                quat_apply_inverse(yaw_quat(self._asset.data.root_link_quat_w), self._offset_w)
            )
        return self._cache


def ee_pos_in_horizontal_frame(env: ManagerBasedRlEnv, ee_body_name: str) -> torch.Tensor:
    """Compute EE position in the pelvis-centered yaw-only (horizontal) frame.

    Frame definition: origin = pelvis world position, axes = world axes rotated by
    pelvis yaw only (pitch/roll stripped). This frame is heading-invariant — rotating
    the robot in place does not translate the EE target, preventing locomotion/
    manipulation fighting.

    Unlike the full body frame, this does NOT cancel pitch/roll disturbances. If the
    pelvis tilts, ee_pos_h shifts because FK uses the full robot state. This is the
    correct behavior: the policy should observe tilt-induced EE drift and correct it.

    Args:
        env: ManagerBasedRlEnv with scene["robot"].
        ee_body_name: Name of the EE body (e.g. "right_hand").

    Returns:
        (B, 3) EE position in horizontal frame [meters].
    """
    asset = env.scene["robot"]
    # Resolve body index once; this function is called at each step so IDs must be cached
    # by the calling class (see ee_current_pos_h).
    body_ids, _ = asset.find_bodies([ee_body_name])
    ee_pos_w = asset.data.body_link_pos_w[:, body_ids[0], :]   # (B, 3) world
    pelvis_pos_w  = asset.data.root_link_pos_w                  # (B, 3) world
    pelvis_quat_w = asset.data.root_link_quat_w                 # (B, 4) wxyz

    # R_yaw^-1 @ (ee_pos_w - pelvis_pos_w): rotate offset into horizontal frame
    offset_w = ee_pos_w - pelvis_pos_w                          # (B, 3)
    q_yaw = yaw_quat(pelvis_quat_w)                             # (B, 4) yaw-only quaternion
    return quat_apply_inverse(q_yaw, offset_w)                  # (B, 3)


class ee_current_pos_h:
    """Observation term: EE position in the pelvis-centered horizontal frame.

    Caches the body ID at init to avoid per-step string resolution.
    Shares _EEHFrameComputer with ee_error_h (via env._ee_h_frame_{body_id}) so the
    yaw_quat + quat_apply_inverse computation runs at most once per step.

    Params:
        ee_body_name: MuJoCo body name of the end-effector (e.g. "right_hand").
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset = env.scene["robot"]
        body_ids, _ = asset.find_bodies([cfg.params["ee_body_name"]])
        body_id = body_ids[0]
        attr = f"_ee_h_frame_{body_id}"
        if not hasattr(env, attr):
            setattr(env, attr, _EEHFrameComputer(asset, body_id, env))
        self._computer: _EEHFrameComputer = getattr(env, attr)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self._computer.get(env)  # (B, 3)


class ee_target_pos_h:
    """Observation term: EE target position in the horizontal frame.

    Reads env.<target_attr> (B, 3), which is managed by the experiment:
      - Initialized at reset to the current EE default position in horizontal frame.
      - Updated by the EE command term (M3+) when the robot is standing.

    _target is a direct reference to the env tensor (written in-place, never replaced
    after init), so __call__ is a zero-cost attribute return.

    Params:
        ee_body_name: Used at reset to initialize the target from the FK default pose.
        target_attr:  Name of the env attribute holding the target buffer
                      (default "_ee_target_pos_h"). Set to a unique name per EE for
                      bimanual use (e.g. "_ee_target_pos_h_R", "_ee_target_pos_h_L").
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset = env.scene["robot"]
        body_ids, _ = self.asset.find_bodies([cfg.params["ee_body_name"]])
        self._ee_body_id = body_ids[0]
        self._target_attr = cfg.params.get("target_attr", "_ee_target_pos_h")
        if not hasattr(env, self._target_attr):
            setattr(env, self._target_attr, torch.zeros(env.num_envs, 3, device=env.device))
        # Direct reference: target tensor is written in-place (never replaced after init).
        self._target: torch.Tensor = getattr(env, self._target_attr)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Re-initialize EE target to current FK default pose for reset envs."""
        idx = slice(None) if env_ids is None else env_ids
        offset_w = (
            self.asset.data.body_link_pos_w[:, self._ee_body_id, :]
            - self.asset.data.root_link_pos_w
        )
        self._target[idx] = quat_apply_inverse(
            yaw_quat(self.asset.data.root_link_quat_w), offset_w
        )[idx]

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self._target  # (B, 3)


class ee_error_h:
    """Observation term: EE tracking error in the horizontal frame.

    Returns (B, 3): ee_target_pos_h − ee_current_pos_h.
    Both in the pelvis-centered yaw-only frame — see ee_pos_in_horizontal_frame.
    Shares _EEHFrameComputer with ee_current_pos_h (via env._ee_h_frame_{body_id}) so
    the horizontal frame position is computed at most once per step.
    _target is a direct reference (written in-place); _out is pre-allocated so
    torch.sub(out=) avoids a (B,3) allocation every call.

    Params:
        ee_body_name: MuJoCo body name of the end-effector.
        target_attr:  Name of the env attribute holding the target buffer
                      (default "_ee_target_pos_h"). Must match the value used in
                      ee_target_pos_h and EETargetsResample for the same EE.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset = env.scene["robot"]
        body_ids, _ = asset.find_bodies([cfg.params["ee_body_name"]])
        body_id = body_ids[0]
        attr = f"_ee_h_frame_{body_id}"
        if not hasattr(env, attr):
            setattr(env, attr, _EEHFrameComputer(asset, body_id, env))
        self._computer: _EEHFrameComputer = getattr(env, attr)
        target_attr = cfg.params.get("target_attr", "_ee_target_pos_h")
        if not hasattr(env, target_attr):
            setattr(env, target_attr, torch.zeros(env.num_envs, 3, device=env.device))
        # Direct reference: target tensor is written in-place (never replaced after init).
        self._target: torch.Tensor = getattr(env, target_attr)
        self._out = torch.zeros(env.num_envs, 3, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        torch.sub(self._target, self._computer.get(env), out=self._out)
        return self._out  # (B, 3)


def _ee_pos_in_h(asset, body_ids, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return EE positions in horizontal frame: (B, K, 3).

    Shared helper for batched EE obs classes. Broadcasts the yaw quaternion over K bodies.
    """
    ee_pos_w      = asset.data.body_link_pos_w[:, body_ids, :]   # (B, K, 3)
    pelvis_pos_w  = asset.data.root_link_pos_w                    # (B, 3)
    pelvis_quat_w = asset.data.root_link_quat_w                   # (B, 4)
    B, K, _ = ee_pos_w.shape
    offset_w = ee_pos_w - pelvis_pos_w[:, None, :]                # (B, K, 3)
    yaw_q = yaw_quat(pelvis_quat_w).unsqueeze(1).expand(B, K, 4).reshape(B * K, 4)
    return quat_apply_inverse(yaw_q, offset_w.reshape(B * K, 3)).view(B, K, 3)


class ee_targets_h:
    """Batched obs: EE target positions in horizontal frame, shape (B, K*3).

    Reads env._ee_targets_h (B, K, 3). Initialized at reset to current FK positions.
    Resampled by EETargetsResample event.
    _targets is a direct reference to the env tensor; _targets_flat is a pre-computed
    (B, K*3) view alias — __call__ returns it with zero allocation.

    Params:
        ee_body_names: List of K MuJoCo body names.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset = env.scene["robot"]
        body_ids, _ = self.asset.find_bodies(cfg.params["ee_body_names"])
        self._body_ids = body_ids
        B, K = env.num_envs, len(body_ids)
        if not hasattr(env, "_ee_targets_h"):
            env._ee_targets_h = torch.zeros(B, K, 3, device=env.device)
        # Direct reference + pre-computed flat view to avoid reshape every call.
        self._targets    = env._ee_targets_h          # (B, K, 3) — written in-place
        self._targets_flat = self._targets.view(B, K * 3)  # (B, K*3) — zero-copy alias
        self._env = env  # kept only for reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._targets[idx] = _ee_pos_in_h(self.asset, self._body_ids, self._env)[idx]

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        return self._targets_flat  # (B, K*3) — zero allocation


class ee_currents_h:
    """Batched obs: current EE positions in horizontal frame, shape (B, K*3).

    _out is a flat (B, K*3) buffer; _current_h is a (B, K, 3) view alias into it.
    FK result is written into _current_h via copy_; _out is returned directly —
    zero reshape allocation per call. Step guard skips FK on repeated calls within
    the same step (e.g. actor + critic obs groups).

    Params:
        ee_body_names: List of K MuJoCo body names.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset = env.scene["robot"]
        body_ids, _ = self.asset.find_bodies(cfg.params["ee_body_names"])
        self._body_ids = body_ids
        B, K = env.num_envs, len(body_ids)
        # Flat output buffer; _current_h is a (B,K,3) view into it for writing.
        # Returning the flat buffer directly avoids reshape every call.
        self._out       = torch.zeros(B, K * 3, device=env.device)
        self._current_h = self._out.view(B, K, 3)
        self._last_step = -1

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if env.common_step_counter != self._last_step:
            self._last_step = env.common_step_counter
            self._current_h.copy_(_ee_pos_in_h(self.asset, self._body_ids, env))
        return self._out


class ee_errors_h:
    """Batched obs: EE tracking errors in horizontal frame, shape (B, K*3).

    Returns env._ee_targets_h − current EE positions for all K bodies.
    _target is a direct reference to env._ee_targets_h (written in-place, never replaced).
    _current_h is updated once per step (step guard); _error_3d is a (B,K,3) view of
    the flat _out buffer — torch.sub(out=_error_3d) writes the result with zero extra
    allocation, and _out is returned directly (no reshape).
    The subtraction runs every call (not step-guarded) since _target may change intra-step
    via EETargetsResample events.

    Params:
        ee_body_names: List of K MuJoCo body names.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.asset = env.scene["robot"]
        body_ids, _ = self.asset.find_bodies(cfg.params["ee_body_names"])
        self._body_ids = body_ids
        B, K = env.num_envs, len(body_ids)
        if not hasattr(env, "_ee_targets_h"):
            env._ee_targets_h = torch.zeros(B, K, 3, device=env.device)
        # Direct reference: target tensor written in-place (never replaced after init).
        self._target: torch.Tensor = env._ee_targets_h
        self._current_h = torch.zeros(B, K, 3, device=env.device)
        # Flat output buffer; _error_3d is a (B,K,3) view used as torch.sub out=.
        self._out      = torch.zeros(B, K * 3, device=env.device)
        self._error_3d = self._out.view(B, K, 3)
        self._last_step = -1

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if env.common_step_counter != self._last_step:
            self._last_step = env.common_step_counter
            self._current_h.copy_(_ee_pos_in_h(self.asset, self._body_ids, env))
        torch.sub(self._target, self._current_h, out=self._error_3d)
        return self._out


# =============================================================================
# Arm Joint Target (HumanoidLocoArmFollow)
# =============================================================================


class arm_joint_target:
    """Moving arm joint target for loco-arm-follow training.

    Each step, advances a Warp linear-interpolation kernel between collision-free
    arm poses drawn from safe_arm_poses_{robot_name}.pt. Writes env._arm_joint_target
    (B, K) — a direct alias to the internal current_target tensor so the arm_pose
    reward reads the updated value at zero copy cost.

    Two pose pools:
    - Active pool (mid-episode): sorted_poses[:active_pool_size], grows 5%→100%.
      Sorting by L2 from q_default makes consecutive draws naturally close early in
      training without an explicit jump limit (which could pick unsafe paths).
    - Reset pool (episode start): sorted_poses[:reset_pool_size], fixed at 5%.
      Keeps episode-start tracking error small regardless of curriculum phase.

    Curriculum schedule is baked into __call__ via stages in params — no separate
    CurriculumTermCfg needed. Step-counter guard ensures the kernel launches exactly
    once per env step even when actor and critic obs groups both call this term.

    Params (ObservationTermCfg.params):
        arm_joint_pattern:  Regex pattern for arm joints (must match asset joint names).
        robot_name:         Cache file suffix (default "humanoid_v21").
        reset_pool_fraction: Fixed episode-reset pool fraction (default 0.05).
        pool_stages:        [{"step", "value"}] for active_pool_fraction schedule.
        min_steps_stages:   [{"step", "value"}] for min hold-duration schedule.
        max_steps_stages:   [{"step", "value"}] for max hold-duration schedule.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        # Guard: if a second obs group (e.g. critic) instantiates this term, share the
        # first instance's state rather than reinitializing. The step-guard in __call__
        # ensures the Warp kernel runs exactly once per env step regardless of how many
        # obs groups call this term.
        if hasattr(env, "_arm_joint_target"):
            self._is_secondary = True
            return
        self._is_secondary = False

        from pathlib import Path
        from tasks.legs_only_task import process_randomized_actions_kernel

        self._process_kernel = process_randomized_actions_kernel
        self.num_envs = env.num_envs
        self.device = env.device
        asset = env.scene["robot"]

        # --- Resolve arm joints in model order (same order arm_pose reward uses) ---
        arm_pattern = cfg.params["arm_joint_pattern"]
        joint_ids, joint_names = asset.find_joints((arm_pattern,))
        self.joint_ids = joint_ids
        K = len(joint_ids)

        # q_default_arm: robot arm joint positions after keyframe reset.
        # Used as the starting point for every episode reset — guarantees near-zero initial
        # tracking error (q_arm = q_default_arm = _current). The pose library is only used
        # for the NEXT target, not the starting point. This sidesteps the fact that the
        # library may not contain q_default (many collision-free libraries are uniformly
        # sampled and may never land exactly on the default pose).
        self._q_default_arm = asset.data.default_joint_pos[0, joint_ids].clone()  # (K,)

        # --- Load and align pose cache ---
        robot_name = cfg.params.get("robot_name", "humanoid_v21")
        cache_path = (
            Path(__file__).resolve().parents[2]
            / "asset_zoo" / "cache"
            / f"safe_arm_poses_{robot_name}.pt"
        )
        from utils.slim_cache import expand as _expand_slim_cache
        _expand_slim_cache(cache_path)
        data = torch.load(str(cache_path), map_location=self.device, weights_only=False)
        # `mjcf_path` is stored absolute by the generator, so a cache produced on another machine
        # names a path that does not exist here. Resolve a relative entry against the repo root,
        # and skip the staleness hash when the MJCF cannot be located at all -- an unverifiable
        # cache is still usable, whereas raising makes a distributed cache unusable everywhere.
        mjcf_path = Path(data["mjcf_path"])
        if not mjcf_path.is_absolute():
            mjcf_path = Path(__file__).resolve().parents[3] / mjcf_path
        if mjcf_path.is_file():
            live_hash = xxhash.xxh64(mjcf_path.read_bytes()).hexdigest()
            if live_hash != data.get("mjcf_hash"):
                raise RuntimeError(
                    f"safe_arm_poses_{robot_name}.pt stale — MJCF changed. "
                    f"Regenerate: python mj_envs/asset_zoo/generate_safe_arm_poses.py --robot {robot_name}"
                )
        cache_names = data["joint_names"]

        # Reorder cache columns to model joint order; silently drops waist if present.
        try:
            col_indices = [cache_names.index(n) for n in joint_names]
        except ValueError as e:
            raise ValueError(
                f"arm_joint_target: arm joint missing from cache.\n"
                f"  Env arm joints : {joint_names}\n"
                f"  Cache joints   : {cache_names}\n"
                f"  Missing        : {e}"
            )
        poses = data["poses"][:, col_indices].contiguous().to(self.device)  # (N, K)
        N = poses.shape[0]

        # Sort by L2 from q_default if cache is not already sorted.
        # One-time cost at init; handles both legacy (unsorted) and new (presorted) caches.
        if not data.get("sorted_by_l2_from_default", False):
            q_default_arm = asset.data.default_joint_pos[0, joint_ids]  # (K,)
            order = (poses - q_default_arm).norm(dim=1).argsort()
            poses = poses[order].contiguous()
        self.sorted_poses = poses  # (N, K), index 0 nearest to q_default
        self.N = N

        # Fixed reset pool
        reset_frac = cfg.params.get("reset_pool_fraction", 0.05)
        self._reset_pool_size = max(1, int(N * reset_frac))

        # Curriculum schedules: list of (step, value) pairs, ascending by step.
        _N = 24  # _STEPS_PER_ENV — used as multiplier in default stages
        self._pool_stages = [
            (s["step"], s["value"])
            for s in cfg.params.get("pool_stages", [
                {"step": 0,         "value": 0.05},
                {"step": 500  * _N, "value": 0.05},
                {"step": 2000 * _N, "value": 1.0},
            ])
        ]
        self._min_steps_stages = [
            (s["step"], s["value"])
            for s in cfg.params.get("min_steps_stages", [
                {"step": 0,         "value": 200},
                {"step": 2000 * _N, "value": 50},
            ])
        ]
        self._max_steps_stages = [
            (s["step"], s["value"])
            for s in cfg.params.get("max_steps_stages", [
                {"step": 0,         "value": 500},
                {"step": 2000 * _N, "value": 200},
            ])
        ]
        # Live curriculum values — updated each step in __call__
        self.active_pool_fraction = self._pool_stages[0][1]
        self.min_steps = int(self._min_steps_stages[0][1])
        self.max_steps = int(self._max_steps_stages[0][1])

        # Warp interpolation buffers (persistent — safe to wrap with wp.from_torch)
        self._current   = torch.zeros(self.num_envs, K, device=self.device)
        self._next      = torch.zeros(self.num_envs, K, device=self.device)
        self._rate      = torch.zeros(self.num_envs, K, device=self.device)
        self._steps_rem = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._current_wp   = wp.from_torch(self._current)
        self._next_wp      = wp.from_torch(self._next)
        self._rate_wp      = wp.from_torch(self._rate)
        self._steps_rem_wp = wp.from_torch(self._steps_rem)
        self._poses_wp     = wp.from_torch(self.sorted_poses)
        self._wp_stream    = wp.stream_from_torch(torch.cuda.current_stream(self.device))

        self._frame_count = 0
        self._last_step = -1  # step-guard: kernel runs once per env step

        # Initialize all envs synchronously at q_default.
        # Current = q_default gives zero initial tracking error.
        # Next = random library pose; smoothly approached over hold_steps.
        # Reset() uses the same logic — see reset() docstring.
        self._init_reset_all()

        # Expose shared buffer and self for reward access and debugging.
        # env._arm_joint_target aliases _current — Warp writes in-place, reward reads live.
        env._arm_joint_target = self._current
        env._arm_joint_target_term = self

    @staticmethod
    def _interp(step: int, stages: list[tuple[int, float]]) -> float:
        """Linear interpolation over (step, value) stages."""
        for i in range(len(stages) - 1):
            s0, v0 = stages[i]
            s1, v1 = stages[i + 1]
            if s0 <= step < s1:
                return v0 + (step - s0) * (v1 - v0) / (s1 - s0)
        return stages[-1][1] if step >= stages[-1][0] else stages[0][1]

    def _init_reset_all(self) -> None:
        """Initialize all envs: current=q_default, next=random library pose.

        Called once at init to pre-load all envs before the first training step.
        The RL runner calls get_observations() (not env.reset()) before the first step,
        so obs_manager.reset() is never called before the first reward computation.
        This prevents arm_pose from seeing huge initial tracking error (q_default vs zeros).
        """
        B = self.num_envs
        q0 = self._q_default_arm.unsqueeze(0).expand(B, -1)  # (B, K)
        rng_next = torch.randint(0, self._reset_pool_size, (B,), device=self.device)
        nxt = self.sorted_poses[rng_next]  # (B, K)
        steps = torch.randint(self.min_steps, self.max_steps + 1, (B,),
                              device=self.device, dtype=torch.int32)
        self._current.copy_(q0)
        self._next.copy_(nxt)
        self._steps_rem.copy_(steps)
        self._rate.copy_((nxt - q0) / steps.float().unsqueeze(-1).clamp(min=1))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset arm targets for specified envs.

        Sets current=q_default_arm for each reset env so the episode always starts
        with near-zero tracking error (robot arms also reset to q_default). The next
        target is sampled from the active pool; it is approached smoothly over
        hold_steps. This avoids the large initial error that occurs if the current
        target is a random library pose (which can be far from q_default).
        """
        if self._is_secondary:
            return  # primary instance owns reset
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = env_ids.to(torch.long)
        if len(ids) == 0:
            return
        B = len(ids)
        active_pool_size = max(1, int(self.N * self.active_pool_fraction))
        q0 = self._q_default_arm.unsqueeze(0).expand(B, -1)  # (B, K)
        rng_next = torch.randint(0, active_pool_size, (B,), device=self.device)
        nxt = self.sorted_poses[rng_next]  # (B, K)
        steps = torch.randint(self.min_steps, self.max_steps + 1, (B,),
                              device=self.device, dtype=torch.int32)
        self._current[ids] = q0
        self._next[ids] = nxt
        self._steps_rem[ids] = steps
        self._rate[ids] = (nxt - q0) / steps.float().unsqueeze(-1).clamp(min=1)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        if self._is_secondary:
            return env._arm_joint_target  # read-only: primary already advanced
        step = env.common_step_counter
        if step != self._last_step:
            # First call this step (actor obs). Advance kernel and update curriculum.
            self._last_step = step
            self._frame_count += 1
            self.active_pool_fraction = self._interp(step, self._pool_stages)
            self.min_steps = max(1, int(self._interp(step, self._min_steps_stages)))
            self.max_steps = max(self.min_steps + 1, int(self._interp(step, self._max_steps_stages)))
            active_pool_size = max(1, int(self.N * self.active_pool_fraction))
            wp.launch(
                kernel=self._process_kernel,
                dim=self.num_envs,
                inputs=[
                    self._current_wp, self._next_wp, self._rate_wp, self._steps_rem_wp,
                    self._poses_wp,
                    active_pool_size,
                    self.min_steps, self.max_steps,
                    self._frame_count,
                ],
                device=self.device,
                stream=self._wp_stream,
            )
        # Subsequent calls this step (e.g. critic obs group) return the same tensor.
        return self._current  # env._arm_joint_target aliases this


class ee_gate:
    """Observation term: velocity-gated EE precision scalar.

    ee_gate = exp(-||v_cmd_xy||² / σ²), σ = 0.2 m/s → ee_gate = 0.04.
    Values: ≈1.0 at v_cmd=0 (standing), ≈0.18 at v_cmd=0.4 m/s.

    Informs the policy when EE precision is demanded (near standing).
    Also used as a reward multiplier in ee_tracking_coarse (M3+).
    _out is pre-allocated to eliminate the per-step unsqueeze allocation.

    Params:
        command_name: Velocity command key in env.command_manager.
        sigma_sq:     Gate scale (default 0.04 = σ=0.2 m/s from §5.2).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._command_name = cfg.params["command_name"]
        self._inv_sigma_sq = 1.0 / float(cfg.params.get("sigma_sq", 0.04))
        self._out = torch.zeros(env.num_envs, 1, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        cmd = env.command_manager.get_command(self._command_name)  # (B, ≥2)
        speed_sq = cmd[:, 0].square() + cmd[:, 1].square()         # ||v_cmd_xy||²
        torch.exp(-speed_sq * self._inv_sigma_sq, out=self._out[:, 0])
        return self._out  # (B, 1)


# =============================================================================
# DR Parameter Observation (estimator_target privileged labels)
# =============================================================================


class RandFootFriction:
    """Current DR tangential friction of foot geoms (mean), normalized to ~[−1, 1].

    Reads env.sim.model.geom_friction[:, foot_geom_ids, 0] each step.
    Returns (B, 1): (mean_friction − center) / half_range.

    Params:
        asset_cfg: SceneEntityCfg with geom_names set to foot geom pattern (required).
        center:     Nominal friction (default 1.4 = midpoint of DR range [0.3, 2.5]).
        half_range: Half DR range (default 1.1).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self._geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
        self._center = float(cfg.params.get("center", 1.4))
        self._half_range = float(cfg.params.get("half_range", 1.1))
        self._device = env.device

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        env_ids = torch.arange(env.num_envs, device=self._device, dtype=torch.int)
        # geom_friction: (n_envs, n_geoms, 3); axis 0 = tangential friction
        friction = env.sim.model.geom_friction[env_ids[:, None], self._geom_ids[None, :], 0]
        return (friction.mean(dim=-1, keepdim=True) - self._center) / self._half_range


class RandBodyMassScale:
    """Total body mass normalized by default total mass, mapped to ~[−1, 1].

    Reads env.sim.model.body_mass[:, body_ids] each step and computes
    sum(current_mass) / default_total_mass − 1.0, then divides by half_range.
    Returns (B, 1).

    Params:
        asset_cfg:  SceneEntityCfg with body_names=".*" (all bodies, required).
        half_range: Half DR scale range (default 0.15 for DR scale [0.85, 1.15]).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset = env.scene[asset_cfg.name]
        self._body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
        self._half_range = float(cfg.params.get("half_range", 0.15))
        self._device = env.device
        # Snapshot default total mass before any DR (all envs identical at init)
        env_0 = torch.zeros(1, dtype=torch.int, device=env.device)
        self._default_total = float(env.sim.model.body_mass[env_0, self._body_ids].sum().item())

    def __call__(self, env: ManagerBasedRlEnv, **_) -> torch.Tensor:
        env_ids = torch.arange(env.num_envs, device=self._device, dtype=torch.int)
        mass = env.sim.model.body_mass[env_ids[:, None], self._body_ids[None, :]]
        scale = mass.sum(dim=-1, keepdim=True) / self._default_total
        return (scale - 1.0) / self._half_range
