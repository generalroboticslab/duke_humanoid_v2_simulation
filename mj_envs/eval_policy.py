#!/usr/bin/env python3
"""Headless deterministic policy evaluation across fixed velocity commands.

Runs N episodes per command without noise or DR, giving a weight-independent
behavioral comparison across experiments. Complements training log analysis by
catching issues that only appear in deterministic play (e.g., v8 oscillation
invisible in training logs but visible at eval time).

No training overhead — runs entirely post-hoc on saved checkpoints.

Usage:
  DISPLAY="" python mj_envs/eval_policy.py \\
      --task HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv8 \\
      --checkpoint runs/HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv8/2026-03-31_02-04-39/model_14999.pt \\
      --episodes 50

  # Compare multiple checkpoints from the same run (curriculum progression):
  DISPLAY="" python mj_envs/eval_policy.py \\
      --task HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv10 \\
      --checkpoint runs/.../model_5000.pt runs/.../model_10000.pt runs/.../model_14999.pt

  # Custom command set:
  DISPLAY="" python mj_envs/eval_policy.py \\
      --task HumanoidVelocityRMACNNShortEstimatorPhaseEEDRv8 \\
      --checkpoint runs/.../model_14999.pt \\
      --cmds "0.5,0,0" "1.0,0,0" "0.8,0,0.5" "-0.3,0,0"

  # Dead zone eval (§7.1 SOP):
  DISPLAY="" python mj_envs/eval_policy.py \\
      --task HumanoidRmaVelEstArmFlashSacv30 \\
      --checkpoint runs/.../model_0015000.pt \\
      --episodes 10 --max-steps 500 \\
      --cmds "0.05,0,0" "0.10,0,0" "0.20,0,0" "0.30,0,0"

Metrics reported (all weight-independent):
  fell_over_rate   — fraction of episodes where `fell_over` termination fired
  termination_rate — fraction of episodes ending in any non-timeout termination
  ep_length        — mean steps per episode
  displacement_m   — mean XY displacement from reset position (m); §7.1 gate metric
  track_vx_err     — mean |cmd_x - actual_vx| (m/s)
  track_vy_err     — mean |cmd_y - actual_vy| (m/s)
  track_xy_err     — mean ||cmd_xy - actual_vxy||₂ (m/s)
  track_ang_err    — mean |cmd_yaw - actual_wyaw| (rad/s)
  action_rate      — mean ||a_t - a_{t-1}||² per step (smoothness)
  arm_delta_l2     — mean arm action delta magnitude per step

§7.1 dead zone thresholds (displacement in 10s / 500 steps at 50Hz):
  cmd=0.05 m/s → displacement ≥ 0.10 m  (dead zone if < 0.10 m)
  cmd=0.10 m/s → displacement ≥ 0.50 m  (dead zone if < 0.10 m)
  cmd=0.20 m/s → displacement ≥ 1.50 m  (dead zone if < 0.50 m)
  cmd=0.30 m/s → displacement ≥ 2.00 m  (dead zone if < 1.00 m)
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Default eval commands: (vx m/s, vy m/s, wz rad/s)
# Covers: forward slow/fast, turning, lateral, stop. Keep sign-mirrored so
# directional asymmetry is visible in the default benchmark.
# ---------------------------------------------------------------------------
_DEFAULT_POSITIVE_CMDS = [
    (0.5,  0.0,  0.0),
    (1.0,  0.0,  0.0),
    (0.0,  0.0,  0.5),
    (0.5,  0.0,  0.5),
    (0.0,  0.0,  0.0),   # standing still
]


def symmetrize_commands(
    cmds: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Return unique commands plus their sign inverses in stable order."""
    symmetric = []
    for cmd in cmds:
        for candidate in (cmd, tuple(-value for value in cmd)):
            if candidate not in symmetric:
                symmetric.append(candidate)
    return symmetric


DEFAULT_CMDS = symmetrize_commands(_DEFAULT_POSITIVE_CMDS)
DEFAULT_EPISODES = 50
DEFAULT_MAX_STEPS = 1000   # max steps per episode


@dataclass
class EvalResult:
    cmd: tuple[float, float, float]
    fell_over_rate: float
    termination_rate: float
    termination_causes: dict[str, float]
    ep_length: float
    displacement_m: float   # mean XY displacement from reset pos; §7.1 dead zone gate
    track_vx_err: float
    track_vy_err: float
    track_xy_err: float
    track_ang_err: float
    action_rate: float
    arm_delta_l2: float
    n_episodes: int
    # Raw (weight-divided) mean value of each reward term, so two policies with
    # different term weights stay comparable. Weight-0 terms are omitted.
    reward_terms: dict[str, float] = field(default_factory=dict)


def parse_cmds(cmd_strs: list[str]) -> list[tuple[float, float, float]]:
    cmds = []
    for s in cmd_strs:
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Command must be 'vx,vy,wz', got: {s!r}")
        cmds.append(tuple(parts))
    return cmds


def load_policy(task: str, checkpoint: str, device: str, num_envs: int = 1):
    """Load trained policy from checkpoint. Returns (policy_fn, env_wrapper)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from run import (
        get_env_cfg, apply_experiment, base_env, create_agent_cfg,
        apply_play_overrides, PlayConfig, _load_trained_policy,
    )
    from mjlab.envs import ManagerBasedRlEnv
    from rsl_rl.runners import OnPolicyRunner
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab_util.patched_observation_manager import ChronologicalObservationManager

    play_cfg = PlayConfig(task=task, checkpoint=checkpoint, device=device,
                          num_envs=num_envs, verbose=False)
    exp = apply_experiment(task, play_cfg, silent=True)
    env_cfg = base_env(play_cfg, exp=exp, enable_reward_curriculum=False, play=True)
    agent_cfg = create_agent_cfg(task)

    if exp is not None:
        exp.configure(env_cfg, agent_cfg)

    apply_play_overrides(env_cfg, play_cfg)

    # Disable interval/startup DR — keep reset events (per-episode randomization).
    if env_cfg.events is not None:
        env_cfg.events = {
            k: v for k, v in env_cfg.events.items()
            if getattr(v, "mode", None) not in ("interval", "startup")
        }

    env_cfg.scene.num_envs = num_envs

    torch.set_float32_matmul_precision("high")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env.observation_manager = ChronologicalObservationManager(env_cfg.observations, env)
    env._configure_gym_env_spaces()
    env.observation_manager.reset()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy = _load_trained_policy(play_cfg, env, agent_cfg,
                                  Path(checkpoint), None, env_cfg, device)
    if policy is None:
        raise RuntimeError(f"Failed to load policy from {checkpoint}")

    return policy, env


MAX_LANES = 4096   # proven safe on a 24GB card — training runs this exact env count


def eval_cmds(policy, env, chunk_cmds: list[tuple[float, float, float]],
              block_count: int, episodes: int, max_steps: int,
              device: str) -> list[EvalResult]:
    """Evaluate several commands inside a single rollout.

    Lanes are contiguous per-command blocks: lanes [i*episodes, (i+1)*episodes)
    all run command i. Every command therefore advances together in one forward
    pass of max_steps, instead of one full pass per command.

    Why batched: a 29-command x 50-episode sweep is only ~725k env-steps, which
    the physics does in seconds. Running commands serially left all but
    `episodes` lanes idle and cost ~29x more wall time than the work requires.

    `block_count` is the lane capacity in blocks and is fixed for the life of
    the env, so a caller sweeping more commands than fit passes them in chunks
    of at most `block_count`. A final short chunk simply leaves the tail blocks
    unread rather than reallocating the env.

    Each lane still gets independent reset-mode DR (PD gains, friction, initial
    pose), so per-command numbers keep the same episode-level variance as
    before; only the scheduling changed.
    """
    n_envs = block_count * episodes
    assert env.unwrapped.scene.num_envs == n_envs, (
        f"env has {env.unwrapped.scene.num_envs} lanes, batch layout needs {n_envs}"
    )
    act_dim = env.unwrapped.action_space.shape[-1]
    arm_start = max(0, act_dim - 14)
    unwrapped = env.unwrapped
    n_cmds = len(chunk_cmds)

    # Lane -> command map. Tail blocks beyond this chunk's commands are never
    # read back, but still need a valid command to step with.
    cmd_lane = torch.empty(n_envs, 3, device=device)
    for i in range(block_count):
        cmd = chunk_cmds[min(i, n_cmds - 1)]
        cmd_lane[i * episodes:(i + 1) * episodes] = torch.tensor(cmd, device=device)

    obs, _ = env.reset()

    # Capture start positions for displacement (world frame XY)
    pos_starts = None
    try:
        pos_starts = unwrapped.scene["robot"].data.root_link_pos_w[:, :2].clone()
    except Exception:
        pass

    # Resolve which command terms carry a velocity buffer once. The buffer is
    # re-fetched each step (a term may reallocate on resample) but the attribute
    # scan itself is hoisted out of the hot loop. Deliberately unguarded: if the
    # command manager cannot be scanned, no command is ever applied and every
    # number below is meaningless, so fail loudly rather than silently.
    override_attrs = []
    for term in unwrapped.command_manager._terms.values():
        for attr in ("_vel_command_b", "vel_command_b", "_vel_command_w", "vel_command_w"):
            if hasattr(term, attr):
                override_attrs.append((term, attr))

    def _override_commands():
        for term, attr in override_attrs:
            val = getattr(term, attr)
            width = min(val.shape[-1], 3)
            val[:, :width] = cmd_lane[:, :width]

    _override_commands()

    prev_action = torch.zeros(n_envs, act_dim, device=device)
    finished = torch.zeros(n_envs, dtype=torch.bool, device=device)
    fell_flags = torch.zeros(n_envs, dtype=torch.bool, device=device)
    termination_flags = torch.zeros(n_envs, dtype=torch.bool, device=device)
    termination_cause_flags = {
        name: torch.zeros(n_envs, dtype=torch.bool, device=device)
        for name in unwrapped.termination_manager.active_terms
    }
    ep_lengths = torch.full((n_envs,), max_steps, dtype=torch.long, device=device)
    # Track last valid positions (before reset on fall) per env
    pos_ends = pos_starts.clone() if pos_starts is not None else None

    # Accumulators are per-lane so they can be reduced per command block at the
    # end. The environment auto-resets a done lane before returning from step().
    # Never include that reset trajectory in its original episode's metrics.
    # Terminal state velocity is unavailable through this wrapper, so exclude
    # that one transition from tracking metrics rather than score reset-state
    # velocity.
    tracking_sums = torch.zeros(4, n_envs, device=device)
    tracking_count = torch.zeros(n_envs, dtype=torch.long, device=device)
    action_rate_sum = torch.zeros(n_envs, device=device)
    arm_delta_sum = torch.zeros(n_envs, device=device)
    action_count = torch.zeros(n_envs, dtype=torch.long, device=device)

    # Detect if we should measure errors in world frame (for tasks like Argus/Ballbot)
    is_world = False
    try:
        obs_cfg = unwrapped.cfg.observations.get("actor", None)
        if obs_cfg is not None and "commands_xy" in obs_cfg.terms:
            is_world = True
    except Exception:
        pass

    # Reward terms. `_step_reward` is (n_envs, n_terms) holding raw*weight, so
    # dividing by weight recovers the raw term value — the weight-independent
    # quantity that survives comparison across experiment classes that retune
    # weights. Weight-0 terms carry no recoverable signal and are dropped.
    reward_mgr = unwrapped.reward_manager
    reward_names = list(reward_mgr.active_terms)
    reward_weights = torch.tensor(
        [cfg.weight for cfg in reward_mgr._term_cfgs], device=device
    )
    reward_keep = [i for i, w in enumerate(reward_weights.tolist()) if w != 0.0]
    reward_sums = torch.zeros(len(reward_names), n_envs, device=device)

    for step in range(max_steps):
        active_lanes = ~finished
        # `.any()` on a device tensor forces a host sync. Poll periodically instead of
        # every step: retired lanes are already masked out of every accumulator, so a
        # late break only costs a few idle physics steps, not correctness.
        if step % 50 == 0 and not active_lanes.any():
            break

        with torch.no_grad():
            action = policy(obs)

        obs, _, dones, info = env.step(action)
        dones = dones.bool().view(-1)

        # After step: re-override commands (env may resample on reset)
        _override_commands()

        action_rate_sum += (action - prev_action).pow(2).sum(dim=-1) * active_lanes
        arm_delta_sum += (
            (action[:, arm_start:] - prev_action[:, arm_start:]).pow(2).sum(dim=-1) * active_lanes
        )
        action_count += active_lanes.long()
        prev_action.copy_(action)

        # Same mask as the action metrics: a retired lane is
        # replaying a reset trajectory and must not score.
        reward_sums += reward_mgr._step_reward.T * active_lanes

        # Done lanes now contain a reset state. Only lanes that neither retired
        # earlier nor terminated this step have a physical post-action state
        # belonging to this episode.
        still_in_episode = active_lanes & ~dones

        try:
            if is_world:
                root_vel = unwrapped.scene["robot"].data.root_link_lin_vel_w[:, :2]
                ang_vel = unwrapped.scene["robot"].data.root_link_ang_vel_w[:, 2]
            else:
                root_vel = unwrapped.scene["robot"].data.root_link_lin_vel_b[:, :2]
                ang_vel = unwrapped.scene["robot"].data.root_link_ang_vel_b[:, 2]
            xy_error = root_vel - cmd_lane[:, :2]
            tracking_sums[0] += xy_error[:, 0].abs() * still_in_episode
            tracking_sums[1] += xy_error[:, 1].abs() * still_in_episode
            tracking_sums[2] += torch.linalg.vector_norm(xy_error, dim=-1) * still_in_episode
            tracking_sums[3] += (ang_vel - cmd_lane[:, 2]).abs() * still_in_episode
            tracking_count += still_in_episode.long()
        except Exception:
            pass

        # Record first termination, then retire every done lane. This evaluates
        # exactly one episode per lane instead of silently concatenating resets.
        time_outs = info.get("time_outs", torch.zeros_like(dones))
        ended = active_lanes & dones
        true_term = ended & ~time_outs.bool().view(-1)
        # masked_fill_/where instead of boolean-index assignment: the latter runs
        # nonzero() and syncs the device every step.
        ep_lengths.masked_fill_(ended, step + 1)
        termination_flags |= true_term
        # `dones` combines every failure termination. Read individual manager
        # bits before next compute() overwrites them; otherwise illegal_contact
        # and velocity_tracking_failure are mislabeled as falls.
        for name, flags in termination_cause_flags.items():
            flags |= ended & unwrapped.termination_manager.get_term(name)
        fell_flags |= termination_cause_flags.get(
            "fell_over", torch.zeros_like(fell_flags)
        )
        finished |= ended

        # Keep last nonterminal physical position. Done lanes have been reset.
        if pos_ends is not None:
            try:
                curr_pos = unwrapped.scene["robot"].data.root_link_pos_w[:, :2]
                pos_ends = torch.where(still_in_episode[:, None], curr_pos, pos_ends)
            except Exception:
                pass

    def block_mean(per_lane: torch.Tensor) -> list[float]:
        """Mean over the episodes of each command block."""
        return per_lane.view(block_count, episodes).float().mean(dim=1).tolist()

    def block_ratio(sums: torch.Tensor, counts: torch.Tensor) -> list[float]:
        """Per-block sum/count, matching the old sum-over-all / count-over-all."""
        s = sums.view(block_count, episodes).sum(dim=1)
        c = counts.view(block_count, episodes).sum(dim=1).float()
        return torch.where(c > 0, s / c, torch.full_like(s, float("nan"))).tolist()

    if pos_starts is not None and pos_ends is not None:
        disp = block_mean((pos_ends - pos_starts).norm(dim=-1))
    else:
        disp = [float("nan")] * block_count

    fell = block_mean(fell_flags)
    term = block_mean(termination_flags)
    causes = {name: block_mean(flags) for name, flags in termination_cause_flags.items()}
    ep_len = block_mean(ep_lengths)
    vx_err = block_ratio(tracking_sums[0], tracking_count)
    vy_err = block_ratio(tracking_sums[1], tracking_count)
    xy_err = block_ratio(tracking_sums[2], tracking_count)
    ang_err = block_ratio(tracking_sums[3], tracking_count)
    act_rate = block_ratio(action_rate_sum, action_count)
    arm_dl2 = block_ratio(arm_delta_sum, action_count)
    rewards = {
        reward_names[i]: block_ratio(reward_sums[i] / reward_weights[i], action_count)
        for i in reward_keep
    }

    return [
        EvalResult(
            cmd=chunk_cmds[i],
            fell_over_rate=fell[i],
            termination_rate=term[i],
            termination_causes={name: vals[i] for name, vals in causes.items()},
            ep_length=ep_len[i],
            displacement_m=disp[i],
            track_vx_err=vx_err[i],
            track_vy_err=vy_err[i],
            track_xy_err=xy_err[i],
            track_ang_err=ang_err[i],
            action_rate=act_rate[i],
            arm_delta_l2=arm_dl2[i],
            n_episodes=episodes,
            reward_terms={name: vals[i] for name, vals in rewards.items()},
        )
        for i in range(n_cmds)
    ]


def print_results(label: str, results: list[EvalResult]) -> None:
    # (column header, EvalResult attribute) for the headline table.
    metric_fields = {
        "fell_over": "fell_over_rate", "term_rate": "termination_rate",
        "ep_len": "ep_length", "disp_m": "displacement_m",
        "vx_err": "track_vx_err", "vy_err": "track_vy_err",
        "xy_err": "track_xy_err", "ang_err": "track_ang_err",
        "act_rate": "action_rate", "arm_dl2": "arm_delta_l2",
    }

    def table(title: str | None, names, get, mean_row: bool = False) -> None:
        """One command-per-row table. `get(result, name) -> float` supplies each cell."""
        col = max(10, *(len(name) + 1 for name in names))
        print(f"\n{title}" if title else "")
        print(f"{'':32}" + "".join(f"{name:>{col}}" for name in names))
        print("─" * (32 + col * len(names)))
        for r in results:
            cmd_str = f"vx={r.cmd[0]:+.2f} vy={r.cmd[1]:+.2f} wz={r.cmd[2]:+.2f}"
            print(f"{cmd_str:<32}" + "".join(f"{get(r, name):>{col}.4f}" for name in names))
        if mean_row:
            row = f"{'MEAN':<32}"
            for name in names:
                vals = [v for v in (get(r, name) for r in results) if not math.isnan(v)]
                mean = sum(vals) / len(vals) if vals else float("nan")
                row += f"{mean:>{col}.4f}"
            print(row)

    table(None, list(metric_fields), lambda r, name: getattr(r, metric_fields[name]))

    cause_names = sorted({name for r in results for name in r.termination_causes})
    if cause_names:
        table("Termination-cause episode rates:", cause_names,
              lambda r, name: r.termination_causes.get(name, 0.0))

    reward_names = sorted({name for r in results for name in r.reward_terms})
    if reward_names:
        table("Reward terms (raw, weight-divided — comparable across weight retunes):",
              reward_names, lambda r, name: r.reward_terms.get(name, float("nan")),
              mean_row=True)

    # §7.1 dead zone check
    thresholds = {0.05: 0.10, 0.10: 0.50, 0.20: 1.50, 0.30: 2.00}
    dz_lines = []
    for r in results:
        cmd_vx = round(r.cmd[0], 2)
        if cmd_vx in thresholds and not math.isnan(r.displacement_m):
            thresh = thresholds[cmd_vx]
            status = "PASS" if r.displacement_m >= thresh else "FAIL"
            dz_lines.append(f"  §7.1 cmd={cmd_vx:.2f}: disp={r.displacement_m:.3f}m  thresh={thresh:.2f}m  [{status}]")
    if dz_lines:
        print("\nDead zone eval (§7.1):")
        for line in dz_lines:
            print(line)

    print(f"\n[{label}] summary over {results[0].n_episodes} episodes per command:")
    fell_mean = sum(r.fell_over_rate for r in results) / len(results)
    termination_mean = sum(r.termination_rate for r in results) / len(results)
    def metric_mean(values):
        values = [value for value in values if not math.isnan(value)]
        return sum(values) / len(values) if values else float("nan")

    vx_mean = metric_mean(r.track_vx_err for r in results)
    vy_mean = metric_mean(r.track_vy_err for r in results)
    xy_mean = metric_mean(r.track_xy_err for r in results)
    disp_mean = [r.displacement_m for r in results if not math.isnan(r.displacement_m)]
    disp_mean = sum(disp_mean) / len(disp_mean) if disp_mean else float("nan")
    print(f"  mean fell_over_rate: {fell_mean:.4f}")
    print(f"  mean termination_rate: {termination_mean:.4f}")
    print(f"  mean displacement:   {disp_mean:.4f} m")
    print(f"  mean track_vx_err:   {vx_mean:.4f} m/s")
    print(f"  mean track_vy_err:   {vy_mean:.4f} m/s")
    print(f"  mean track_xy_err:   {xy_mean:.4f} m/s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="Experiment class name")
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="Checkpoint path(s) to evaluate")
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                    help="Episodes per command per checkpoint")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                    help="Max steps per episode")
    ap.add_argument("--cmds", nargs="+", default=None,
                    help="Commands as 'vx,vy,wz' (default: sign-symmetric benchmark set)")
    ap.add_argument("--symmetric", action="store_true",
                    help="Add each supplied command's sign inverse, deduplicating zero commands")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-envs", type=int, default=None,
                    help=f"Cap on total parallel lanes (default: {MAX_LANES}). Commands are "
                         "packed into one rollout at `episodes` lanes each; sweeps needing "
                         "more lanes than the cap run in sequential chunks.")
    ap.add_argument("--stand-min-steps", type=int, default=None,
                    help="Override arm_ref.stand_min_steps at runtime. Use 0 to evaluate "
                         "the v2ybsk ckpt under deploy-equivalent (full-speed arm) and "
                         "isolate training-vs-deploy effect. Omit to use the ckpt-trained value.")
    ap.add_argument("--home-ratio", type=float, default=None,
                    help="Override arm_ref home_ratio at runtime, cancelling its curriculum. "
                         "Use 0.0 for the deploy condition (arm ACTIVE in every episode); 1.0 "
                         "freezes every arm at home. The MixedArms lineage anneals home_ratio "
                         "1.0 -> 0.5 but the 15k budget stops it at 0.583, so a default eval "
                         "scores ~58%% frozen-arm episodes and under-samples the arm-active "
                         "regime that deploy runs 100%% of the time. Omit to use the "
                         "ckpt-trained curriculum value.")
    args = ap.parse_args()

    cmds = parse_cmds(args.cmds) if args.cmds else DEFAULT_CMDS
    if args.symmetric:
        cmds = symmetrize_commands(cmds)
    episodes = args.episodes
    # Pack as many commands as fit under the lane cap into one rollout. Block
    # count is fixed for the env's lifetime, so longer sweeps run in chunks.
    lane_cap = args.num_envs if args.num_envs is not None else MAX_LANES
    block_count = max(1, min(len(cmds), lane_cap // episodes))
    num_envs = block_count * episodes

    for ckpt in args.checkpoint:
        # Label from the checkpoint path, never from ``args.task``: comparing two
        # variants under one --task (same obs/action space, different training cfg)
        # otherwise prints an identical header for every block, which silently
        # inverts A/B readings. Run dir carries the timestamp+tag that disambiguates.
        ckpt_path = Path(ckpt)
        label = f"{ckpt_path.parent.parent.name}/{ckpt_path.parent.name}/{ckpt_path.stem}"
        n_chunks = math.ceil(len(cmds) / block_count)
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"Commands: {len(cmds)}  Episodes/cmd: {episodes}  Max steps: {args.max_steps}")
        print(f"Lanes: {num_envs} ({block_count} cmds/rollout, {n_chunks} rollout(s))")
        print(f"{'='*60}")

        policy, env = load_policy(args.task, ckpt, args.device, num_envs=num_envs)

        # Same-condition probe: override arm_ref.stand_min_steps at runtime so the eval
        # runs under identical deploy-equivalent conditions regardless of how the ckpt
        # was trained. Restored after this ckpt's eval loop.
        arm_ref = env.unwrapped.command_manager._terms.get("arm_ref", None)
        trained_sms = arm_ref._stand_min_steps if arm_ref is not None else None
        if args.stand_min_steps is not None and arm_ref is not None:
            sms = args.stand_min_steps
            arm_ref._stand_min_steps = sms
            arm_ref._inv_stand_min_steps = 1.0 / float(sms) if sms > 0 else 0.0
            print(f"  arm_ref.stand_min_steps override: trained={trained_sms} -> runtime={sms}")

        # home_ratio is set by a CURRICULUM on common_step_counter, so pinning it also
        # requires zeroing curriculum_steps -- _current_home_ratio() ignores _home_ratio
        # whenever _home_ratio_curriculum_steps > 0.
        trained_hr = (arm_ref._current_home_ratio(), arm_ref._home_ratio,
                      arm_ref._home_ratio_curriculum_steps) if arm_ref is not None else None
        if args.home_ratio is not None and arm_ref is not None:
            arm_ref._home_ratio = args.home_ratio
            arm_ref._home_ratio_curriculum_steps = 0
            print(f"  arm_ref.home_ratio override: trained={trained_hr[0]:.3f} "
                  f"-> runtime={args.home_ratio:.3f}")

        results = []
        for chunk_idx in range(n_chunks):
            chunk = cmds[chunk_idx * block_count:(chunk_idx + 1) * block_count]
            print(f"  rollout {chunk_idx + 1}/{n_chunks} ({len(chunk)} cmds) ...",
                  end="", flush=True)
            t0 = time.perf_counter()
            results.extend(eval_cmds(policy, env, chunk, block_count, episodes,
                                     args.max_steps, args.device))
            print(f" {time.perf_counter() - t0:.1f}s")

        print_results(label, results)

        if args.stand_min_steps is not None and arm_ref is not None:
            arm_ref._stand_min_steps = trained_sms
            arm_ref._inv_stand_min_steps = 1.0 / float(trained_sms) if trained_sms > 0 else 0.0
        if args.home_ratio is not None and arm_ref is not None:
            _, arm_ref._home_ratio, arm_ref._home_ratio_curriculum_steps = trained_hr
        env.unwrapped.close()


if __name__ == "__main__":
    main()
