"""Command terms for humanoid velocity variants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp.curriculums import VelocityStage
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(kw_only=True)
class RelativeHeightCommandCfg(CommandTermCfg):
    """Sample a relative height offset around the nominal standing height.

    The command is a 1D offset in meters, not an absolute height. The policy
    observes the offset, while the reward converts it back to an absolute target
    by adding `nominal_height`.
    """

    entity_name: str
    nominal_height: float = 0.59
    offset_range: tuple[float, float] = (-0.1, 0.1)
    # Optional stepped curriculum: list of (step_threshold, (low, high)) sorted ascending.
    # _resample_command picks the range for the largest threshold reached.
    # If None, offset_range is used throughout training.
    offset_curriculum: list | None = None

    def build(self, env: ManagerBasedRlEnv) -> "RelativeHeightCommand":
        return RelativeHeightCommand(self, env)

    def __post_init__(self):
        low, high = self.offset_range
        if low > high:
            raise ValueError(f"offset_range must satisfy low <= high, got {self.offset_range}")
        if self.nominal_height <= 0.0:
            raise ValueError(f"nominal_height must be positive, got {self.nominal_height}")


class RelativeHeightCommand(CommandTerm):
    """Per-episode relative height offset command."""

    cfg: RelativeHeightCommandCfg

    def __init__(self, cfg: RelativeHeightCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._env = env
        self.robot = env.scene[cfg.entity_name]
        self._offset_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["rel_height_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._offset_command

    def _current_range(self) -> tuple[float, float]:
        if not self.cfg.offset_curriculum:
            return self.cfg.offset_range
        step = self._env.common_step_counter
        current = self.cfg.offset_range
        for threshold, rng in self.cfg.offset_curriculum:
            if step >= threshold:
                current = rng
        return current

    def _update_metrics(self) -> None:
        target_height = self.cfg.nominal_height + self._offset_command[:, 0]
        current_height = self.robot.data.root_link_pos_w[:, 2]
        self.metrics["rel_height_error"] += (current_height - target_height).abs()
        self.metrics["base_height_error"] += (current_height - self.cfg.nominal_height).abs()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        low, high = self._current_range()
        offsets = torch.empty(len(env_ids), device=self.device).uniform_(low, high)
        self._offset_command[env_ids, 0] = offsets

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        return


# =============================================================================
# Low-Velocity Walk Initiation
# =============================================================================


class LowVelBiasedVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand with a mixture sampler that over-represents low-magnitude xy cmds.

    On each resample, a `p_low` fraction of non-standing envs get their xy cmd
    replaced by a uniform sample from the annulus [low_min, low_max] at a
    uniform angle.

    Motivation: the default uniform sampler covers cmd ∈ (−1,1)² so the
    low-magnitude annulus (0.1–0.4 m/s) occupies only ~12% of episodes.
    Oversampling populates the buffer with (standing_history, cmd_low, *)
    transitions, calibrating Q(walk | standing_history, cmd_low).

    Design:
    - Standing envs excluded from mask: _update_command zeros their cmd anyway.
    - vel_command_w kept in sync with vel_command_b so world-frame envs stay
      consistent on the next _update_command call.
    - ang_vel_z left unchanged so heading-controlled envs keep normal yaw signal.
    - Annulus [low_min, low_max] not disk: avoids over-weighting near-zero cmd,
      which is effectively equivalent to standing and adds nothing.
    """

    cfg: "LowVelBiasedVelocityCommandCfg"

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)

        # Exclude standing envs — their cmd is zeroed in _update_command.
        is_standing = self.is_standing_env[env_ids]
        select_mask = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.p_low
        ) & ~is_standing
        low_env_ids = env_ids[select_mask]
        if low_env_ids.numel() == 0:
            return

        n_low = low_env_ids.numel()
        angle = torch.rand(n_low, device=self.device) * (2.0 * math.pi)
        mag_span = self.cfg.low_max - self.cfg.low_min
        magnitude = self.cfg.low_min + torch.rand(n_low, device=self.device) * mag_span

        new_vx = magnitude * torch.cos(angle)
        new_vy = magnitude * torch.sin(angle)
        self.vel_command_b[low_env_ids, 0] = new_vx
        self.vel_command_b[low_env_ids, 1] = new_vy
        # Sync world-frame mirror so is_world_env path in _update_command stays consistent.
        self.vel_command_w[low_env_ids, 0] = new_vx
        self.vel_command_w[low_env_ids, 1] = new_vy


@dataclass(kw_only=True)
class LowVelBiasedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for LowVelBiasedVelocityCommand.

    Extra fields:
        p_low:   Fraction of non-standing envs resampled into the low-vel annulus.
        low_min: Annulus inner radius in m/s.
        low_max: Annulus outer radius in m/s.
    """

    p_low: float = 0.3
    low_min: float = 0.1
    low_max: float = 0.4

    def build(self, env: ManagerBasedRlEnv) -> LowVelBiasedVelocityCommand:
        return LowVelBiasedVelocityCommand(self, env)


# =============================================================================
# Velocity command with bounded integral tracking error (low-speed dead-zone fix)
# =============================================================================


class IntegralErrorVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand that also maintains a bounded, leaky integral of the
    velocity tracking error, exposed as `integral_error` (3D: [Ix, Iy, Iyaw]).

    Why: at small commands the instantaneous tracking reward is flat (standing ≈
    stepping), so the policy stands → low-speed dead zone. The *integral* of the
    tracking error accrues debt the longer a command is unmet, giving time pressure;
    rewarding |I|→0 forces intermittent stepping/turning that delivers the average
    sub-v_min rate. The command interface stays pure velocity. This is the integral
    term of a PI controller (existing velocity tracking is the proportional term).

    Per step (after the base updates vel_command_b incl. heading→ang-vel):
        e  = vel_command_b − v_meas                       # v_meas = [vx_b, vy_b, wz]
        I ← clamp((1−leak)·I + e·dt, −I_max, I_max)
    I is zeroed for envs that just resampled (periodic resample OR episode reset), so a
    new command starts debt-free (no cross-command carryover). The per-step penalty in
    the paired reward already charges the debt continuously, so no pre-reset snapshot
    is needed — the single resample step's post-reset I≈0 is immaterial.

    v_meas (sim training): true body velocity + domain randomization, so the policy is
    robust to the onboard estimator used at deploy:
        xy = root_link_lin_vel_b[:, :2] + per-episode bias + per-step noise
        wz = root_link_ang_vel_b[:, 2]   (gyro: reliable sim & real → light noise)
    At deploy the integrator is mirrored with v_meas = v_est (estimator head). DR
    magnitudes come from probe_vest_odometry (bias ~−0.008 m/s, noise ~0.05 m/s xy).

    Off-policy (FlashSAC): `integral_error` is exposed as an observation term so the
    critic state is Markov in the rewarded quantity; the reward reads the same buffer.
    Bounded + leaky keeps replay staleness finite.
    """

    cfg: "IntegralErrorVelocityCommandCfg"

    def __init__(self, cfg: "IntegralErrorVelocityCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.integral_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._v_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._I_max = torch.tensor(cfg.I_max, device=self.device)
        self._dr_bias = torch.tensor(cfg.dr_bias, device=self.device)
        self._dr_noise = torch.tensor(cfg.dr_noise, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # New command → debt-free start (covers periodic resample + episode reset).
        self.integral_error[env_ids] = 0.0
        # Per-episode estimator bias on the measured velocity used by the integral.
        self._v_bias[env_ids] = (
            torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0
        ) * self._dr_bias

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        v_meas = torch.empty(self.num_envs, 3, device=self.device)
        v_meas[:, :2] = self.robot.data.root_link_lin_vel_b[:, :2]
        v_meas[:, 2] = self.robot.data.root_link_ang_vel_b[:, 2]
        v_meas = v_meas + self._v_bias + torch.randn_like(v_meas) * self._dr_noise
        e = self.vel_command_b - v_meas
        self.integral_error = torch.clamp(
            (1.0 - self.cfg.leak) * self.integral_error + e * self._env.step_dt,
            -self._I_max, self._I_max,
        )


@dataclass(kw_only=True)
class IntegralErrorVelocityCommandCfg(UniformVelocityCommandCfg):
    """Config for IntegralErrorVelocityCommand.

    Extra fields:
        leak:     Per-step integral leak (0 = pure bounded integral, reset on resample).
        I_max:    Per-axis |I| cap (Ix, Iy in m; Iyaw in rad).
        dr_bias:  Per-axis per-episode bias half-range on v_meas (m/s, m/s, rad/s).
        dr_noise: Per-axis per-step Gaussian std on v_meas (m/s, m/s, rad/s).
    """

    leak: float = 0.0
    I_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
    dr_bias: tuple[float, float, float] = (0.02, 0.02, 0.005)
    dr_noise: tuple[float, float, float] = (0.05, 0.05, 0.01)

    def build(self, env: ManagerBasedRlEnv) -> IntegralErrorVelocityCommand:
        return IntegralErrorVelocityCommand(self, env)


def integral_error_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Observation: bounded integral tracking error from IntegralErrorVelocityCommand."""
    return env.command_manager.get_term(command_name).integral_error

# =============================================================================
# Grid-adaptive velocity curriculum
# =============================================================================

# Termination term whose firing must NOT void a command period's mastery evidence under
# ``stability_only_survivorship`` -- it is a tracking cutoff, and mastery already scores tracking.
_TRACKING_CUTOFF_TERM = "velocity_tracking_failure"


class GridAdaptiveVelocityCommand(IntegralErrorVelocityCommand):
    """Joint 3D velocity-command curriculum driven by measured velocity-tracking error.

    Non-standing commands sample one ``(vx, vy, wz)`` cell from ``grid_weights``
    then uniformly sample inside that cell. Exact-zero standing commands retain the
    base sampler's separate standing branch, use selected-cell sentinel ``-1``, and
    never affect curriculum weights. At the next resample, a moving cell whose mean
    normalised linear and angular tracking error both fall below ``master_error_frac``
    activates itself and its valid six face-neighbors. This preserves a joint
    distribution, so it cannot expand independent per-axis ranges into unmastered
    command combinations.

    Mastery is measured directly from state (true root velocity vs command), NOT from
    the reward manager, so reshaping the tracking reward (std, kernel, weights) cannot
    silently redefine "mastered". ``master_error_frac`` is the one physical bar: mean
    tracking error over a command period below ``k``·(commanded speed), with a floor
    (``lin_speed_floor`` / ``yaw_floor``) so near-zero commands are judged on absolute
    error instead of an unachievable relative one. Stability needs no term of its own:
    a fall terminates the episode, ``reset`` zeroes the period's accumulators before
    ``_update_grid_weights`` reads them, so only periods survived without early
    termination can ever credit a cell.

    Heading, world-frame, forward-only, and initial-velocity modes are intentionally
    rejected: each mutates a sampled command after cell assignment and would invalidate
    the correspondence between reward history and a grid cell.

    The optional turn-in-place lane (``turn_lane_mass_range``) obeys that same rule rather
    than breaking it: it is a DISTINCT cell set sampled INSTEAD of the grid, not a grid
    command mutated after assignment. A lane env never receives a grid cell, keeps
    ``selected_cells`` at the ``-1`` sentinel, and accumulates its tracking reward in its own
    counters, so no grid cell is ever credited for a command it did not produce.

    """

    cfg: "GridAdaptiveVelocityCommandCfg"

    def __init__(self, cfg: "GridAdaptiveVelocityCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._grid_shape = torch.tensor(cfg.grid_bins, dtype=torch.long, device=self.device)
        self._grid_lows = torch.tensor(
            (cfg.ranges.lin_vel_x[0], cfg.ranges.lin_vel_y[0], cfg.ranges.ang_vel_z[0]),
            device=self.device,
        )
        grid_highs = torch.tensor(
            (cfg.ranges.lin_vel_x[1], cfg.ranges.lin_vel_y[1], cfg.ranges.ang_vel_z[1]),
            device=self.device,
        )
        self._cell_widths = (grid_highs - self._grid_lows) / self._grid_shape
        self.grid_weights = torch.zeros(tuple(cfg.grid_bins), device=self.device)
        self.selected_cells = torch.full((self.num_envs, 3), -1, dtype=torch.long, device=self.device)
        self._neighbor_offsets = torch.tensor(
            ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)),
            device=self.device,
        )
        self._safe_cell_mask = self._planar_speed_mask(cfg.max_planar_speed)
        self._stage_masks = self._build_stage_masks()
        self._deadline_stage_count = 0
        self._stage_deadline_steps: tuple[int, ...] | None = None
        # Per-period accumulators of ABSOLUTE tracking error (linear ‖·‖, angular |·|), summed over
        # the command period and normalised by the command magnitude at resample. Not rewards.
        self._tracking_err_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self._tracking_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tracking_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Turn-in-place lane. Separate error counters from the grid's are REQUIRED, not tidiness:
        # lane envs keep ``selected_cells == -1``, and _update_grid_weights filters candidate cells
        # only by ``_tracking_count > 0``. Counting lane envs there would pass -1 rows into
        # the neighbor scatter and corrupt grid weights.
        self._turn_yaw_bin = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._turn_err_sum = torch.zeros(self.num_envs, device=self.device)
        self._turn_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # One EMA PER YAW BIN. Init 0 = maximum deficit, so every bin opens at the upper mass bound
        # and decays only as that bin itself is learned.
        self._turn_track_ema = torch.zeros(int(self._grid_shape[2]), device=self.device)
        self.metrics["grid_active_cells"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["grid_stage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["grid_final_coverage"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["turn_lane_mass"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["turn_lane_worst_ema"] = torch.zeros(self.num_envs, device=self.device)
        self._activate_initial_cells()
        self._set_active_cell_metric()

    def _set_active_cell_metric(self) -> None:
        """Expose grid progress without synchronizing the training GPU."""
        weights = self.grid_weights
        self.metrics["grid_active_cells"][:] = weights.count_nonzero()
        completed = torch.stack([(weights[mask] > 0.0).all() for mask in self._stage_masks]).sum()
        self.metrics["grid_stage"][:] = completed
        self.metrics["grid_final_coverage"][:] = (weights[self._safe_cell_mask] > 0.0).all()

    def _axis_centers(self) -> list[torch.Tensor]:
        return [
            self._grid_lows[axis]
            + (torch.arange(self._grid_shape[axis].item(), device=self.device) + 0.5)
            * self._cell_widths[axis]
            for axis in range(3)
        ]

    def _planar_speed_mask(self, speed_limit: float) -> torch.Tensor:
        centers = self._axis_centers()
        planar_speed = torch.sqrt(centers[0][:, None].square() + centers[1][None, :].square())
        return (planar_speed[:, :, None] <= speed_limit).expand(-1, -1, self._grid_shape[2])

    def _initial_cell_mask(self) -> torch.Tensor:
        centers = self._axis_centers()
        planar_speed = torch.sqrt(centers[0][:, None].square() + centers[1][None, :].square())
        if bool(torch.all(self._cell_widths == 0)):
            # Zero-VOLUME envelope: every range collapsed to a point, so all cells share the same
            # command and ``initial_planar_speed_range`` is unsatisfiable by construction. This is an
            # explicit "no command sampling" configuration, not a misconfiguration -- the frozen-policy
            # reach seam (tasks/frozen_policy.py) pins the twist ranges to (0,0,0) with
            # ``rel_standing_envs=1.0`` and drives ``vel_command_b`` itself. There is no envelope to
            # explore, so seed the whole (degenerate) grid: ``_build_stage_masks`` then starts with a
            # full frontier and collapses to a single stage, and ``_activate_initial_cells`` has cells
            # to activate. Requires ALL THREE axes degenerate -- a partially-collapsed envelope that
            # still misses the initial band is a real misconfiguration and must keep raising.
            return self._safe_cell_mask.clone()
        return (
            (planar_speed[:, :, None] >= self.cfg.initial_planar_speed_range[0])
            & (planar_speed[:, :, None] <= self.cfg.initial_planar_speed_range[1])
            & (centers[2][None, None, :] >= self.cfg.initial_yaw_range[0])
            & (centers[2][None, None, :] <= self.cfg.initial_yaw_range[1])
        )

    def _build_stage_masks(self) -> tuple[torch.Tensor, ...]:
        initial = self._initial_cell_mask()
        distance = torch.full_like(self.grid_weights, -1, dtype=torch.long)
        distance[initial] = 0
        frontier = initial
        masks = [initial]
        depth = 0
        while torch.any(distance[self._safe_cell_mask] < 0):
            depth += 1
            cells = frontier.nonzero(as_tuple=False)
            neighbors = cells[:, None, :] + self._neighbor_offsets[1:]
            valid = ((neighbors >= 0) & (neighbors < self._grid_shape)).all(dim=-1)
            neighbors = neighbors[valid]
            next_frontier = torch.zeros_like(frontier)
            if neighbors.numel() > 0:
                safe_unvisited = self._safe_cell_mask[tuple(neighbors.T)] & (distance[tuple(neighbors.T)] < 0)
                neighbors = neighbors[safe_unvisited]
                if neighbors.numel() > 0:
                    next_frontier[tuple(neighbors.T)] = True
                    distance[tuple(neighbors.T)] = depth
            frontier = next_frontier
            if not torch.any(frontier) and torch.any(distance[self._safe_cell_mask] < 0):
                raise ValueError("Grid stage graph does not reach every safe command cell.")
            masks.append((distance >= 0) & (distance <= depth))
        if not torch.all(masks[-1] == self._safe_cell_mask):
            raise ValueError("Final grid stage must cover every safe planar command cell.")
        if any(not torch.all(earlier <= later) for earlier, later in zip(masks, masks[1:], strict=False)):
            raise ValueError("Grid stages must be nested.")
        return tuple(masks)

    def _activate_initial_cells(self) -> None:
        active = self._initial_cell_mask()
        if not active.any():
            raise ValueError("Initial planar-speed and yaw ranges must contain at least one grid-cell center.")
        self.grid_weights[active] = self.cfg.max_weight

    def _advance_deadline_stages(self) -> None:
        if self._stage_deadline_steps is None:
            return
        while (
            self._deadline_stage_count < len(self._stage_deadline_steps)
            and self._env.common_step_counter >= self._stage_deadline_steps[self._deadline_stage_count]
        ):
            self._deadline_stage_count += 1
            self.grid_weights[self._stage_masks[self._deadline_stage_count]] = self.cfg.max_weight
        self._set_active_cell_metric()

    def set_training_horizon(self, total_control_steps: int) -> None:
        """Derive coverage-proportional stage deadlines from runner training horizon.

        Deadlines allocate equal control-step budget per safe grid cell introduced,
        rather than hard-coding iteration counts. The final graph shell activates at
        ``final_stage_fraction`` of training, leaving the remaining fraction as
        mandatory final-stage dwell.
        """
        if total_control_steps <= 0:
            raise ValueError("Grid curriculum training horizon must be positive.")
        initial_count = self._stage_masks[0].count_nonzero().item()
        safe_count = self._safe_cell_mask.count_nonzero().item()
        added_cells = safe_count - initial_count
        self._stage_deadline_steps = tuple(
            round(
                total_control_steps
                * self.cfg.final_stage_fraction
                * (mask.count_nonzero().item() - initial_count)
                / added_cells
            )
            for mask in self._stage_masks[1:]
        )

    def _tracking_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env absolute tracking error this control step, from TRUE root velocity vs command.

        Same velocity source and body frame the tracking reward reads (``root_link_lin_vel_b`` /
        ``root_link_ang_vel_b[2]``), so sharpening the gate carries no noise-chasing risk and it
        stays independent of whatever kernel/std the reward uses.
        """
        v = self.robot.data.root_link_lin_vel_b
        w_yaw = self.robot.data.root_link_ang_vel_b[:, 2]
        lin_err = torch.linalg.vector_norm(self.vel_command_b[:, :2] - v[:, :2], dim=1)
        ang_err = (self.vel_command_b[:, 2] - w_yaw).abs()
        return lin_err, ang_err

    def _update_metrics(self) -> None:
        assigned = self.selected_cells[:, 0] >= 0
        active = assigned & self._tracking_ready
        in_lane = self._turn_yaw_bin >= 0
        lane_active = in_lane & self._tracking_ready
        self._tracking_ready[assigned | in_lane] = True
        lin_err, ang_err = self._tracking_errors()
        self._tracking_err_sum[:, 0].add_(lin_err * active)
        self._tracking_err_sum[:, 1].add_(ang_err * active)
        self._tracking_count.add_(active.to(dtype=torch.long))
        self._turn_err_sum.add_(ang_err * lane_active)
        self._turn_count.add_(lane_active.to(dtype=torch.long))

    def _update_grid_weights(self, env_ids: torch.Tensor) -> None:
        eligible = self._tracking_count[env_ids] > 0
        env_ids = env_ids[eligible]
        if env_ids.numel() == 0:
            return
        # Mean absolute error over the period, normalised by the command magnitude (constant over the
        # period). Floors switch the bar from relative to absolute near zero command: above the floor
        # e ≤ k means "within k of the commanded speed"; below it, "within k·floor" absolute.
        mean_err = self._tracking_err_sum[env_ids] / self._tracking_count[env_ids, None]
        cmd = self.vel_command_b[env_ids]
        cmd_xy = torch.linalg.vector_norm(cmd[:, :2], dim=1).clamp(min=self.cfg.lin_speed_floor)
        cmd_yaw = cmd[:, 2].abs().clamp(min=self.cfg.yaw_floor)
        e_lin = mean_err[:, 0] / cmd_xy
        e_ang = mean_err[:, 1] / cmd_yaw
        successful_cells = self.selected_cells[env_ids][
            (e_lin <= self.cfg.master_error_frac)
            & (e_ang <= self.cfg.master_error_frac)
        ]
        if successful_cells.numel() == 0:
            return
        neighbors = successful_cells[:, None, :] + self._neighbor_offsets
        valid = ((neighbors >= 0) & (neighbors < self._grid_shape)).all(dim=-1)
        neighbors = neighbors[valid]
        neighbors = neighbors[self._safe_cell_mask[tuple(neighbors.T)]]
        mirrored = self._grid_shape - 1 - neighbors
        neighbors = torch.cat((neighbors, mirrored), dim=0)
        flat_indices = (
            neighbors[:, 0] * self._grid_shape[1] * self._grid_shape[2]
            + neighbors[:, 1] * self._grid_shape[2]
            + neighbors[:, 2]
        )
        flat_weights = self.grid_weights.flatten()
        increments = torch.zeros_like(flat_weights)
        increments.scatter_add_(0, flat_indices, torch.full_like(flat_indices, self.cfg.neighbor_weight, dtype=flat_weights.dtype))
        flat_weights.add_(increments).clamp_(max=self.cfg.max_weight)
        self._set_active_cell_metric()

    def _grid_probabilities(self) -> torch.Tensor:
        """Uniform sampling distribution over the currently unlocked cells.

        ``grid_weights`` is binary (``neighbor_weight == max_weight == 1.0``), so it is a pure 0/1
        reachability gate: a cell the curriculum has not opened stays at zero, so an unreachable
        command is never issued. Normalising the gate yields uniform probability over exactly the
        opened cells.
        """
        gate = self.grid_weights.flatten()
        return gate / gate.sum()

    def _sample_grid_commands(self, env_ids: torch.Tensor) -> None:
        probabilities = self._grid_probabilities()
        flat_cells = torch.multinomial(probabilities, len(env_ids), replacement=True)
        ny, nz = self._grid_shape[1], self._grid_shape[2]
        cells = torch.stack((flat_cells // (ny * nz), (flat_cells // nz) % ny, flat_cells % nz), dim=1)
        commands = self._grid_lows + (cells + torch.rand_like(cells, dtype=torch.float32)) * self._cell_widths
        valid = torch.linalg.vector_norm(commands[:, :2], dim=-1) <= self.cfg.max_planar_speed
        while not torch.all(valid):
            invalid = ~valid
            commands[invalid] = self._grid_lows + (
                cells[invalid] + torch.rand_like(cells[invalid], dtype=torch.float32)
            ) * self._cell_widths
            valid = torch.linalg.vector_norm(commands[:, :2], dim=-1) <= self.cfg.max_planar_speed
        self.selected_cells[env_ids] = cells
        self.vel_command_b[env_ids] = commands
        self.vel_command_w[env_ids] = commands

    def _turn_lane_mass(self, bins: torch.Tensor | None) -> tuple[float, torch.Tensor | None]:
        """Lane share of moving envs, plus how that share splits across the active yaw bins.

        Exact turn-in-place is measure-zero in the command box, so equal-weight cell sampling can
        never give it usable mass: a 9-cell zero-linear lane among 468 safe grid cells would be 1.9%
        of commands, indistinguishable from the ~1% density that caused the dead zone. An explicit
        mass term is therefore irreducible. What is avoidable is hardcoding its OPERATING POINT, so
        only the bounds are configured and the point inside them tracks measured performance:

            mass    = clamp(hi * MAX_k(1 - ema_k), lo, hi)
            p(bin k) = (1 - ema_k) / sum_j(1 - ema_j)

        ``hi`` doubles as the gain, so no separate gain constant exists. Angular reward alone drives
        it because the lane's linear command is zero, where the linear reward scores standing still
        and would report success exactly when the lane is failing.

        MAX, not mean, and deficit-weighted bins, not uniform -- this is the v2 -> v3 fix, and both
        halves are load-bearing. v2 kept ONE pooled EMA over all 8 lane bins and split the mass
        uniformly. Fast turn-in-place is easy, so the 7 easy bins learned quickly, dragged the pooled
        EMA to ~0.94, and collapsed mass to its floor by iter 3750 -- while the one bin holding the
        regime that actually fails (wz~0.2) was still broken. Measured on both v2 seeds: that bin
        received 0.61% of envs, BELOW the ~1% natural density that caused the dead zone in the first
        place. The lane starved the only thing it existed to train, and neither seed improved
        (frozen 0.20 / 0.38 vs parent 0.41 -- i.e. no effect, the 0.20 was a lucky draw).

        MAX keeps the lane open while ANY turn speed still fails; deficit weighting points it at the
        one that does. A bin the policy has mastered decays out of the split on its own.
        """
        if self.cfg.turn_lane_mass_range is None or bins is None or bins.numel() == 0:
            return 0.0, None
        low, high = self.cfg.turn_lane_mass_range
        deficit = (1.0 - self._turn_track_ema[bins]).clamp(min=0.0)
        mass = float(torch.clamp(high * deficit.max(), min=low, max=high))
        total = deficit.sum()
        # All bins mastered: fall back to uniform so the floor mass stays a valid distribution.
        probs = deficit / total if total > 0.0 else torch.full_like(deficit, 1.0 / deficit.numel())
        return mass, probs

    def _active_turn_bins(self) -> torch.Tensor:
        """Yaw bins the lane may sample: grid-active bins that exclude zero yaw.

        Reuses the grid's own activation instead of carrying a second curriculum, so lane yaw
        coverage widens exactly as the grid's does -- slow turn-in-place before fast.

        The zero-crossing bin is excluded because a near-zero yaw command with zero linear command
        is not turn-in-place, it is standing, which the inherited ``rel_standing_envs`` branch
        already samples at 20%. Including it would also break the mass feedback at exactly the wrong
        moment: with the usual ``initial_yaw_range`` that bin is the ONLY one active at init, so the
        lane would open at its upper bound, score the high angular reward that standing earns at
        zero yaw command, and collapse to its lower bound BEFORE any real turning bin ever opened.
        """
        active = (self.grid_weights > 0.0).any(dim=0).any(dim=0)
        edges = self._grid_lows[2] + torch.arange(self._grid_shape[2] + 1, device=self.device) * self._cell_widths[2]
        active &= (edges[:-1] > 0.0) | (edges[1:] < 0.0)
        return active.nonzero(as_tuple=False).flatten()

    def _sample_turn_lane(self, env_ids: torch.Tensor, bins: torch.Tensor, probs: torch.Tensor) -> None:
        """Command these envs to rotate with EXACTLY zero linear velocity.

        Zero must be exact: a measured companion linear command as small as 0.05 m/s collapses the
        frozen fraction from 51% to 9%, so any epsilon here trains a different regime than the one
        that fails at deploy.

        Bins are drawn from ``probs`` (deficit-weighted), not uniformly -- see _turn_lane_mass.
        """
        picks = bins[torch.multinomial(probs, env_ids.numel(), replacement=True)]
        yaw = self._grid_lows[2] + (picks + torch.rand(picks.shape, device=self.device)) * self._cell_widths[2]
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 2] = yaw
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self._turn_yaw_bin[env_ids] = picks

    def _update_turn_ema(self, env_ids: torch.Tensor) -> None:
        """Fold the finishing lane envs' mean angular COMPETENCE into that bin's running estimate.

        Competence is ``clamp(1 - e_ang, 0, 1)`` where ``e_ang`` is the normalised angular tracking
        error; 1 = mastered, 0 = standing (error equals the command). Same [0,1], 1=good convention
        the old mean-angular-reward carried, so ``_turn_lane_mass``'s ``1 - ema`` deficit is unchanged.
        Angular only: the lane's linear command is zero, where linear error would reward standing.

        PER BIN, not pooled: a single scalar averaged the easy fast-turn bins together with the one
        slow-turn bin that actually fails, so the average read "mastered" while the failing bin was
        untouched. See _turn_lane_mass for the measured consequence.

        Each bin's EMA rate is the fraction of ITS OWN current population being folded in -- the
        per-bin analogue of the pooled version's rate, so the time constant stays roughly one
        turnover of that bin regardless of how the mass is split. Using finished/num_envs here
        instead would slow every bin by the number of bins.
        """
        finished = env_ids[(self._turn_yaw_bin[env_ids] >= 0) & (self._turn_count[env_ids] > 0)]
        if finished.numel() == 0:
            return
        n_bins = int(self._grid_shape[2])
        bins_f = self._turn_yaw_bin[finished]
        mean_err = self._turn_err_sum[finished] / self._turn_count[finished]
        cmd_yaw = self.vel_command_b[finished, 2].abs().clamp(min=self.cfg.yaw_floor)
        competence = (1.0 - mean_err / cmd_yaw).clamp(0.0, 1.0)
        sums = torch.zeros(n_bins, device=self.device).index_add_(0, bins_f, competence)
        counts = torch.zeros(n_bins, device=self.device).index_add_(0, bins_f, torch.ones_like(competence))
        # Population is read BEFORE _resample_command clears _turn_yaw_bin, so it includes these envs.
        pop = torch.bincount(self._turn_yaw_bin[self._turn_yaw_bin >= 0], minlength=n_bins).clamp(min=1)
        seen = counts > 0
        alpha = (counts[seen] / pop[seen]).clamp(max=1.0)
        means = sums[seen] / counts[seen]
        self._turn_track_ema[seen] += alpha * (means - self._turn_track_ema[seen])

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._advance_deadline_stages()
        self._update_grid_weights(env_ids)
        self._update_turn_ema(env_ids)
        self._tracking_err_sum[env_ids] = 0.0
        self._tracking_count[env_ids] = 0
        self._tracking_ready[env_ids] = False
        self.selected_cells[env_ids] = -1
        self._turn_err_sum[env_ids] = 0.0
        self._turn_count[env_ids] = 0
        self._turn_yaw_bin[env_ids] = -1
        super()._resample_command(env_ids)
        moving_env_ids = env_ids[~self.is_standing_env[env_ids]]
        if moving_env_ids.numel() == 0:
            return
        # The lane is empty until the grid opens a yaw bin clear of zero; _turn_lane_mass returns 0
        # for that case, so the metric reports the mass the lane is actually running at.
        bins = self._active_turn_bins() if self.cfg.turn_lane_mass_range is not None else None
        mass, probs = self._turn_lane_mass(bins)
        self.metrics["turn_lane_mass"][:] = mass
        if mass > 0.0:
            assert bins is not None and probs is not None
            # Worst active bin drives the mass; log it so a starved bin is visible during training
            # rather than only in a post-hoc eval, which is how the v2 pooled-EMA bug escaped.
            self.metrics["turn_lane_worst_ema"][:] = float(self._turn_track_ema[bins].min())
            to_lane = torch.rand(moving_env_ids.numel(), device=self.device) < mass
            lane_ids, moving_env_ids = moving_env_ids[to_lane], moving_env_ids[~to_lane]
            if lane_ids.numel() > 0:
                self._sample_turn_lane(lane_ids, bins, probs)
        if moving_env_ids.numel() > 0:
            self._sample_grid_commands(moving_env_ids)

    def _unstable(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Bool mask over ``env_ids``: episode ended from instability, not a tracking cutoff.

        Every non-timeout termination term counts EXCEPT ``velocity_tracking_failure``, so a new
        stability term added to the task is vetoing by default and only the one term this fix is
        about is exempt. The termination manager loads after the command manager, so the first
        reset of the run can arrive before it exists; nothing has terminated yet, so no veto.
        """
        manager = getattr(self._env, "termination_manager", None)
        if manager is None:
            return torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)
        names = [
            name for name in manager.active_terms
            if name != _TRACKING_CUTOFF_TERM and not manager.get_term_cfg(name).time_out
        ]
        unstable = torch.zeros(env_ids.shape, dtype=torch.bool, device=self.device)
        for name in names:
            unstable |= manager.get_term(name)[env_ids]
        return unstable

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        """Discard the in-flight command period's mastery evidence, then resample.

        ``super().reset`` calls ``_resample`` -> ``_update_grid_weights``, so whatever is zeroed
        here is exactly what the period is NOT credited for. Zeroing everything (legacy) implements
        "only periods survived without early termination can credit a cell".

        That rule was written to keep a fall from counting as mastery, but the humanoid task also
        terminates on ``velocity_tracking_failure``, and MEASURED on this env that term -- not
        instability -- is what the rule actually filters on. Over iters 1k-6k, where the mastery
        cascade has to fire, ``...WaistStandFrozenAtNode`` runs ``velocity_tracking_failure`` at
        0.59 per episode against ``...WaistStand``'s 0.05 (12x), while ``illegal_contact`` is
        0.84 vs 0.86 and ``fell_over`` 0.15 vs 0.06 -- i.e. the excess terminations that stalled its
        curriculum were entirely the tracking cutoff. So the gate double-counts tracking: a cell must
        clear a hard error cliff (the termination) AND then ``master_error_frac``. The cliff adds no
        information the mean-error test lacks -- a period bad enough to trip it carries a large
        ``mean_err`` and fails ``master_error_frac`` on its own -- but it does veto every period that
        merely dipped, so any change raising tracking-failure rate silently freezes cell expansion.

        ``stability_only_survivorship`` narrows the veto to instability (``fell_over`` /
        ``illegal_contact``), letting a tracking-cut period be scored normally on the steps it did
        run. Default False keeps the legacy behaviour bit-exact.

        Rejected: dropping the veto entirely (a fall mid-period would then credit the cell from the
        good steps before it, which is the failure mode the original rule correctly prevents), and
        weighting credit by surviving step fraction (changes what ``master_error_frac`` means for
        every already-graded run, breaking this campaign's band comparisons).
        """
        assert isinstance(env_ids, torch.Tensor)
        voided = env_ids if not self.cfg.stability_only_survivorship else env_ids[self._unstable(env_ids)]
        self._tracking_err_sum[voided] = 0.0
        self._tracking_count[voided] = 0
        self._tracking_ready[voided] = False
        self.selected_cells[voided] = -1
        self._turn_err_sum[voided] = 0.0
        self._turn_count[voided] = 0
        self._turn_yaw_bin[voided] = -1
        extras = super().reset(env_ids)
        self._set_active_cell_metric()
        return extras

    def curriculum_state_dict(self) -> dict[str, torch.Tensor]:
        """Return persistent grid state; per-command histories reset on resume."""
        return {
            "grid_weights": self.grid_weights.detach().cpu(),
            "turn_track_ema": self._turn_track_ema.detach().cpu(),
        }

    def load_curriculum_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore persistent weights when their configured grid shape matches.

        ``turn_track_ema`` is optional so checkpoints written before the turn lane existed still
        load; absent, the lane re-warms from maximum deficit. A scalar value (the v2 pooled EMA) is
        also dropped rather than broadcast: broadcasting the pooled figure would seed every bin with
        the easy bins' average and reproduce the starvation the per-bin split exists to fix.
        """
        weights = state["grid_weights"].to(self.device)
        if weights.shape != self.grid_weights.shape:
            raise ValueError(
                f"Grid checkpoint shape {tuple(weights.shape)} does not match configured shape {tuple(self.grid_weights.shape)}."
            )
        self.grid_weights.copy_(weights)
        ema = state.get("turn_track_ema")
        if ema is not None and ema.shape == self._turn_track_ema.shape:
            self._turn_track_ema.copy_(ema.to(self.device))
        self._set_active_cell_metric()


@dataclass(kw_only=True)
class GridAdaptiveVelocityCommandCfg(IntegralErrorVelocityCommandCfg):
    """Configuration for a local joint ``(vx, vy, wz)`` command curriculum."""

    grid_bins: tuple[int, int, int] = (8, 8, 9)
    # MAGNITUDE annulus on ‖v_xy‖ (not a signed interval like initial_yaw_range): the BFS seeds every
    # cell whose center speed lies in [lo, hi], in every direction. Spans the whole low-speed band
    # rather than a thin ring, so the curriculum starts where the gait keyframe / state bank already
    # injects the robot (event.py s_max = 0.6) instead of leaving the slowest commands to a later BFS
    # stage -- that mismatch is the low-command dead zone. ``lo`` is a positive epsilon only because a
    # literal 0.0 is rejected below; no grid in use has a cell center that slow.
    initial_planar_speed_range: tuple[float, float] = (0.05, 0.6)
    initial_yaw_range: tuple[float, float] = (-0.1, 0.1)
    # Mastery bar, in PHYSICAL error, reward-independent. A cell (and a lane yaw bin) is mastered when
    # mean tracking error over a command period is within master_error_frac of the commanded speed on
    # BOTH axes: linear ‖v_err_xy‖ ≤ k·‖cmd_xy‖ and angular |w_err| ≤ k·|cmd_yaw|. k=0.25 reproduces
    # the retired 0.8-kernel semantic (0.472·std_rel, std_rel=0.5) as a clean "within a quarter".
    master_error_frac: float = 0.25
    # Denominator floors: below these command magnitudes the bar switches from relative to ABSOLUTE
    # (k·floor), so near-zero commands are not held to an unachievable relative precision and cmd=0
    # never divides by zero. Own knobs, deliberately NOT tied to the reward's std_min.
    lin_speed_floor: float = 0.1
    yaw_floor: float = 0.1
    neighbor_weight: float = 1.0
    max_weight: float = 1.0
    max_planar_speed: float = 0.8
    final_stage_fraction: float = 0.8
    # (lo, hi) bounds on the turn-in-place lane's share of MOVING envs, or None to disable the lane
    # entirely. None = bit-identical to the pre-lane sampler, so every existing child is unchanged;
    # only a class that opts in differs. The operating point inside these bounds is DERIVED from the
    # lane's own angular tracking deficit -- see GridAdaptiveVelocityCommand._turn_lane_mass.
    turn_lane_mass_range: tuple[float, float] | None = None
    # Which episode endings void a command period's mastery evidence. False (legacy) = ANY early
    # termination does, because ``reset`` zeroes the accumulators before ``_update_grid_weights``
    # reads them. True = only INSTABILITY does; a period cut short by ``velocity_tracking_failure``
    # is still scored by ``master_error_frac``. See GridAdaptiveVelocityCommand.reset for why the
    # legacy rule double-counts tracking and starves the curriculum.
    stability_only_survivorship: bool = False

    def build(self, env: ManagerBasedRlEnv) -> GridAdaptiveVelocityCommand:
        return GridAdaptiveVelocityCommand(self, env)

    def __post_init__(self):
        super().__post_init__()
        if self.heading_command or self.rel_world_envs or self.rel_forward_envs or self.init_velocity_prob:
            raise ValueError("GridAdaptiveVelocityCommand requires direct velocity commands without heading, world, forward, or initial-velocity overrides.")
        if any(bins <= 0 for bins in self.grid_bins):
            raise ValueError(f"grid_bins must be positive, got {self.grid_bins}")
        if max(abs(limit) for limit in (*self.ranges.lin_vel_x, *self.ranges.lin_vel_y)) > self.max_planar_speed:
            raise ValueError(
                f"GridAdaptiveVelocityCommand linear command ranges must remain within "
                f"+/-{self.max_planar_speed} m/s (the configured max_planar_speed)."
            )
        if self.initial_planar_speed_range[0] <= 0.0 or self.initial_planar_speed_range[0] > self.initial_planar_speed_range[1]:
            raise ValueError("initial_planar_speed_range must be positive and ordered.")
        box_diagonal_speed = math.hypot(
            max(abs(limit) for limit in self.ranges.lin_vel_x),
            max(abs(limit) for limit in self.ranges.lin_vel_y),
        )
        if self.initial_planar_speed_range[1] > box_diagonal_speed:
            raise ValueError("initial_planar_speed_range exceeds configured linear command limits.")
        if self.initial_yaw_range[0] > self.initial_yaw_range[1] or self.initial_yaw_range[0] < self.ranges.ang_vel_z[0] or self.initial_yaw_range[1] > self.ranges.ang_vel_z[1]:
            raise ValueError("initial_yaw_range must lie within configured yaw command limits.")
        if self.turn_lane_mass_range is not None:
            low, high = self.turn_lane_mass_range
            if not 0.0 < low <= high < 1.0:
                raise ValueError(
                    f"turn_lane_mass_range must satisfy 0 < lo <= hi < 1, got {self.turn_lane_mass_range}"
                )
        if not 0.0 < self.neighbor_weight <= self.max_weight:
            raise ValueError("neighbor_weight must be positive and no greater than max_weight.")
        if self.master_error_frac <= 0.0:
            raise ValueError("master_error_frac must be positive.")
        if self.lin_speed_floor <= 0.0 or self.yaw_floor <= 0.0:
            raise ValueError("lin_speed_floor and yaw_floor must be positive.")
        # Upper bound is the configured command box diagonal: a disk larger than the box
        # is a no-op. The hardware planar-speed cap is NOT enforced here -- it is set
        # explicitly per experiment at the callsite.
        if not 0.0 < self.max_planar_speed <= box_diagonal_speed:
            raise ValueError(
                f"max_planar_speed must be in (0.0, {box_diagonal_speed}], the configured "
                "linear command box diagonal."
            )
        if not 0.0 < self.final_stage_fraction < 1.0:
            raise ValueError("final_stage_fraction must be in (0.0, 1.0).")



class PiecewiseLinearVelocityRangeCurriculum:
    """Piecewise-linear interpolation of command velocity ranges over stages.

    Vanilla mjlab's `commands_vel` (mjlab.tasks.velocity.mdp.curriculums) does
    step-jump range updates: at each stage's trigger step, ranges snap directly
    to that stage's values. That produces a discontinuous jump in the sampled
    command distribution, visible as a policy behavior shift right at the
    curriculum boundary. This version interpolates linearly between consecutive
    stages instead, so the range grows smoothly over [steps[i], steps[i+1]).

    Boundary behavior:
        - Before first stage: hold the first stage ranges.
        - After last stage: hold last stage ranges.

    Params:
        command_name: Name of the command term (e.g., "twist").
        velocity_stages: List of dicts, each with keys: step, lin_vel_x,
            lin_vel_y, ang_vel_z. Values are (lo, hi) float tuples. Missing axes
            carry forward the previous stage value.
        decimation: Control steps between cache updates (default: 2400 = 100 iters).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        command_name = cfg.params["command_name"]
        self._command_term = env.command_manager.get_term(command_name)
        assert self._command_term is not None
        command_cfg = cast(UniformVelocityCommandCfg, self._command_term.cfg)
        stages: list[VelocityStage] = cfg.params["velocity_stages"]
        self.decimation = cfg.params.get("decimation", 2400)

        self.steps = tuple(stage["step"] for stage in stages)

        lin_vel_x = []
        lin_vel_y = []
        ang_vel_z = []
        cur_x = command_cfg.ranges.lin_vel_x
        cur_y = command_cfg.ranges.lin_vel_y
        cur_z = command_cfg.ranges.ang_vel_z
        for stage in stages:
            if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
                cur_x = stage["lin_vel_x"]
            if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
                cur_y = stage["lin_vel_y"]
            if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
                cur_z = stage["ang_vel_z"]
            lin_vel_x.append(cur_x)
            lin_vel_y.append(cur_y)
            ang_vel_z.append(cur_z)

        self.lin_vel_x = tuple(lin_vel_x)
        self.lin_vel_y = tuple(lin_vel_y)
        self.ang_vel_z = tuple(ang_vel_z)

        n = len(stages) - 1
        dt = tuple(float(self.steps[i + 1] - self.steps[i]) for i in range(n))
        self._sx = tuple(
            (
                (self.lin_vel_x[i + 1][0] - self.lin_vel_x[i][0]) / dt[i],
                (self.lin_vel_x[i + 1][1] - self.lin_vel_x[i][1]) / dt[i],
            )
            for i in range(n)
        )
        self._sy = tuple(
            (
                (self.lin_vel_y[i + 1][0] - self.lin_vel_y[i][0]) / dt[i],
                (self.lin_vel_y[i + 1][1] - self.lin_vel_y[i][1]) / dt[i],
            )
            for i in range(n)
        )
        self._sz = tuple(
            (
                (self.ang_vel_z[i + 1][0] - self.ang_vel_z[i][0]) / dt[i],
                (self.ang_vel_z[i + 1][1] - self.ang_vel_z[i][1]) / dt[i],
            )
            for i in range(n)
        )

        self._first_x = self.lin_vel_x[0]
        self._first_y = self.lin_vel_y[0]
        self._first_z = self.ang_vel_z[0]

        self._last_idx = 0
        self._cx = self._first_x
        self._cy = self._first_y
        self._cz = self._first_z

        dev = env.device
        self._ret: dict[str, torch.Tensor] = {
            "lin_vel_x_min": torch.zeros(1, device=dev),
            "lin_vel_x_max": torch.zeros(1, device=dev),
            "lin_vel_y_min": torch.zeros(1, device=dev),
            "lin_vel_y_max": torch.zeros(1, device=dev),
            "ang_vel_z_min": torch.zeros(1, device=dev),
            "ang_vel_z_max": torch.zeros(1, device=dev),
        }

    def _update_cache(self, step: int) -> None:
        i = self._last_idx
        n = len(self._sx)
        if i < n and self.steps[i] <= step < self.steps[i + 1]:
            ds = step - self.steps[i]
            self._cx = (
                self.lin_vel_x[i][0] + ds * self._sx[i][0],
                self.lin_vel_x[i][1] + ds * self._sx[i][1],
            )
            self._cy = (
                self.lin_vel_y[i][0] + ds * self._sy[i][0],
                self.lin_vel_y[i][1] + ds * self._sy[i][1],
            )
            self._cz = (
                self.ang_vel_z[i][0] + ds * self._sz[i][0],
                self.ang_vel_z[i][1] + ds * self._sz[i][1],
            )
            return

        for i in range(n):
            if self.steps[i] <= step < self.steps[i + 1]:
                self._last_idx = i
                ds = step - self.steps[i]
                self._cx = (
                    self.lin_vel_x[i][0] + ds * self._sx[i][0],
                    self.lin_vel_x[i][1] + ds * self._sx[i][1],
                )
                self._cy = (
                    self.lin_vel_y[i][0] + ds * self._sy[i][0],
                    self.lin_vel_y[i][1] + ds * self._sy[i][1],
                )
                self._cz = (
                    self.ang_vel_z[i][0] + ds * self._sz[i][0],
                    self.ang_vel_z[i][1] + ds * self._sz[i][1],
                )
                return

        if step >= self.steps[-1]:
            self._cx, self._cy, self._cz = self.lin_vel_x[-1], self.lin_vel_y[-1], self.ang_vel_z[-1]
        else:
            self._cx, self._cy, self._cz = self._first_x, self._first_y, self._first_z

    def __call__(self, env: ManagerBasedRlEnv, _env_ids, **_kwargs) -> dict[str, torch.Tensor]:
        step = env.common_step_counter
        if step % self.decimation == 0:
            self._update_cache(step)
            cfg = self._command_term.cfg
            cfg.ranges.lin_vel_x = self._cx
            cfg.ranges.lin_vel_y = self._cy
            cfg.ranges.ang_vel_z = self._cz
            self._ret["lin_vel_x_min"][0] = self._cx[0]
            self._ret["lin_vel_x_max"][0] = self._cx[1]
            self._ret["lin_vel_y_min"][0] = self._cy[0]
            self._ret["lin_vel_y_max"][0] = self._cy[1]
            self._ret["ang_vel_z_min"][0] = self._cz[0]
            self._ret["ang_vel_z_max"][0] = self._cz[1]
        return self._ret
