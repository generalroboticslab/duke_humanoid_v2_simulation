"""FlashSAC configuration for mjlab environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlashSACConfig:
    """Configuration for FlashSAC training."""

    # Training
    num_learning_iterations: int = 25000
    learning_starts: int = 10

    # Network architecture
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    use_layer_norm: bool = True

    # SAC hyperparameters
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4

    # --- Muon optimizer for actor + critic 2D params (screen, 2026-08-07) ---
    # Kimi-K3 / GLM-5 / DeepSeek-V4 all replaced AdamW with Muon for matrix params. When True,
    # actor and qnet use mj_envs.flash_sac.muon.Muon: Newton-Schulz-orthogonalized momentum for
    # ndim==2 params, plain AdamW for biases/norms/conv kernels. alpha keeps fused AdamW (it is
    # a single scalar, nothing to orthogonalize). Muon's step is spectrally normalized, so its
    # useful lr does NOT match AdamW's -- muon_learning_rate is the axis under screen; the
    # AdamW-path lr inside the same optimizer stays at the per-net actor/critic lr.
    use_muon: bool = False
    muon_learning_rate: float = 3e-4
    muon_momentum: float = 0.95
    # Cosine-anneal the MUON GROUP ONLY (the AdamW fallback group and alpha keep their constant
    # baseline lr, so this stays a single-variable change). 0 = constant, matching the first screen.
    #
    # Why this exists: the 2026-08-07 screen found Muon leads the baseline early (+7.30 @1800) and
    # then reverses (-4.63 @15000). Mechanism: a Newton-Schulz update is orthogonalized to FIXED
    # norm, so unlike AdamW -- whose effective step anneals implicitly as gradients shrink -- Muon
    # takes a literally constant-size step for all 15000 iterations. Good far from the optimum,
    # harmful near it. This knob supplies the missing annealing.
    #
    # Units are OUTER ITERATIONS, not gradient steps: the schedulers are stepped once per iteration
    # in the training loop, whereas actor and critic take different numbers of gradient steps
    # (actor only every policy_frequency), so a step counter inside Muon would anneal the two nets
    # at different rates.
    muon_lr_decay_iters: int = 0
    muon_lr_min_factor: float = 0.05
    gamma: float = 0.99
    tau: float = 0.033  # Scaled from 0.125 for 8 updates → 0.033 for 32 (maintains same effective target convergence rate)
    alpha_init: float = 0.01
    use_autotune: bool = True
    target_entropy_ratio: float = 0.5
    # Target entropy via target_sigma (preferred — matches reference FlashSAC).
    # When > 0, overrides target_entropy_ratio: target_entropy = 0.5*n_act*log(2πe·σ²).
    # σ=0.15 → for |A|=22, target_entropy ≈ -10.5 (vs ratio=0.25 giving -5.5).
    # Set to 0.0 to fall back to target_entropy_ratio formulation.
    target_sigma: float = 0.0

    # Replay buffer
    buffer_size: int = 1024  # Per environment
    batch_size: int = 8192  # Global batch size
    num_steps: int = 1  # N-step returns
    # Frame-ring replay (plan/REPLAY_FRAME_RING_PLAN.md): store one frame per timestep per history
    # view and reconstruct the L-frame window at sample time, instead of storing the full overlapping
    # window per transition. ~L× less replay VRAM, lossless, same sampling distribution. Default ON —
    # runner.py degenerates any flat/non-history obs group to a bit-exact 1-frame ring, so this is
    # safe even for experiments without use_sequence_encoder=True (zero savings there, but no harm).
    frame_ring_history: bool = True

    # Update schedule
    num_updates: int = 8  # Updates per env step
    policy_frequency: int = 2  # Actor updates every N critic updates
    num_collect_steps: int = 4  # Env steps collected per iteration (increases data 4x)

    # Distributional critic (C51)
    num_atoms: int = 501
    v_min: float = -20.0
    v_max: float = 20.0
    num_q_networks: int = 2

    # Action handling
    use_tanh: bool = True

    # Optimization
    compile: bool = True
    # torch.compile mode: "default" (no cudagraphs), "reduce-overhead" (cudagraphs on the
    # per-step teacher/student fwd + the learn-step loss fns), "max-autotune" (also picks
    # best matmul). cudagraphs eliminate per-launch CPU overhead on the hot inner loops
    # (teacher/student fwd is called 4× per collect step). Trade-off: first-call warmup is
    # much longer (graphs are captured lazily) and dynamic shapes break capture (we keep
    # shape-static on the fwd side). Default "default" preserves byte-identical numerics
    # to date; flip to "reduce-overhead" to A/B test capture overhead.
    compile_mode: str = "default"
    amp: bool = True
    amp_dtype: str = "bf16"
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    # LR schedule (applied to actor, critic, alpha optimizers).
    # linear warmup 0 → base_lr over lr_warmup_iters, then cosine decay to
    # base_lr * lr_min_factor over lr_decay_iters. Both 0 = constant lr.
    lr_warmup_iters: int = 0
    lr_decay_iters: int = 0
    lr_min_factor: float = 0.1

    # Observation normalization
    obs_normalization: bool = True

    # Reward normalization (matches reference FlashSAC).
    # Tracks discounted returns G_t per env, divides batch rewards by max(sqrt(Var[G]), max|G|/G_max).
    # When True, set v_min=-G_max, v_max=G_max so critic support matches normalized scale.
    normalize_reward: bool = False
    normalized_G_max: float = 5.0

    # Log std bounds
    log_std_max: float = 0.0
    log_std_min: float = -5.0

    # Logging & saving
    save_interval: int = 2500
    logging_interval: int = 50
    logger: str = "wandb"  # "wandb" or "tensorboard"
    wandb_project: str = "mjlab"

    # mjlab-specific: observation group names
    actor_obs_group: str = "actor"
    critic_obs_group: str = "critic"

    # Alive reward bonus (added per step for off-policy stability)
    alive_reward: float = 0.0

    # Sequence encoder (RMA-CNN / TCN) for history obs. Disabled by default (flat MLP).
    use_sequence_encoder: bool = False
    encoder_type: str = "rma_cnn"       # "rma_cnn" | "tcn" | "attn_rma_cnn"
    encoder_embed_dim: int = 32
    encoder_latent_dim: int = 128

    # Temporal encoder over the PRIVILEGED critic history window (L_c, D_c). Off ⇒ 1-frame Markov
    # critic (flat MLP), byte-identical to today. On ⇒ the critic reuses the actor's RMA-CNN
    # (encoder_type/embed/latent above) to implicitly system-ID per-episode DR latents (mass,
    # friction) from the state response, instead of the flat-concat window (the v57 dead-lever
    # mechanism that lacked a temporal inductive bias). Requires the critic obs group to carry a
    # history window (env observations["critic"].history_length > 1).
    use_critic_sequence_encoder: bool = False

    # Velocity estimator head on the sequence encoder latent (feature-gated).
    # Requires use_sequence_encoder=True. When enabled, SequenceActor owns a
    # VelocityEstimatorHead and the actor update adds estimator MSE to actor_loss.
    # estimator_target_key must name a key present in the critic obs TensorDict
    # (e.g. "estimator_target") providing ground-truth base_lin_vel (B, 3).
    # Default-off: exact current behavior when use_velocity_estimator=False.
    use_velocity_estimator: bool = False
    estimator_loss_coef: float = 1.0
    estimator_target_key: str = "estimator_target"
    # Output dimension of the velocity estimator head. Default 3 = (vx, vy, vz).
    # Increase to include privileged physical latents (friction, mass scale, payload).
    vel_head_output_dim: int = 3
    # Stop-gradient between the estimator head and the encoder.
    # True (default): estimator MSE gradient stops at the vel_head — encoder is not updated
    #   by the estimator loss (probe mode, no side effects on locomotion).
    # False: estimator MSE gradient flows back into the encoder, forcing the encoder to
    #   represent the estimator targets (e.g. friction, mass scale).
    estimator_head_stop_gradient: bool = True

    # --- L2T privileged teacher → proprio student distillation (feature-gated) ---
    # When use_distilled_student=True the runner builds a SECOND SequenceActor (the student)
    # alongside the RL actor (the privileged teacher). The student reads its own obs
    # group (student_obs_group), is trained OFF-POLICY from the shared replay by
    # regressing its deterministic action onto the teacher's deterministic action on the
    # same transition (privileged_obs_t), and is the deployable. The teacher RL path is
    # byte-identical when use_distilled_student=False. See plan/DEAD_ZONE_LOW_CMD_VEL_plan.md §7.
    use_distilled_student: bool = False
    student_obs_group: str = "student"          # env obs group the student consumes
    student_learning_rate: float = 3e-4
    imitation_coef: float = 1.0             # weight on the imitation loss in the student loss
    # Imitation loss form. False ⇒ uniform MSE on the squashed action ‖a_s − a_t‖² (discards the
    # teacher's log_std). True ⇒ closed-form reverse KL(π_s‖π_t) between the pre-tanh Gaussians, a
    # teacher-precision-weighted mean match: the mean error (μ_s−μ_t)² is weighted by 1/σ_t² (the
    # DETACHED teacher precision), so the student matches the teacher hardest on the dims the teacher
    # commits to (small σ_t) — the recovery-critical components on the student's own near-fall states.
    # Same optimum (μ_s=μ_t) ⇒ dead-zone closure preserved; only the per-dim/state weighting changes.
    # Reverse (not forward) KL so the weight is the ungameable teacher σ_t, not the student's own σ_s.
    # False ⇒ byte-identical to the prior MSE student loss. See plan/KL_IMITATION_DISTILL_PLAN.md.
    # REJECTED (v58L2T, 2026-06-15): falls REGRESSED 0.00143→0.00215 + dead zone regressed. The SAC
    # teacher σ_t is the EXPLORATION spread, large on balance/recovery dims, so 1/σ_t² down-weights
    # exactly the fall-critical components. Kept off by default; do not enable without a new premise.
    imitation_kl: bool = False
    # Sample-mixing: per-env, execute the student's action w.p. α_mix (else the teacher's
    # explore action); α_mix ramps 0 → student_action_prob_max linearly over the warmup iters.
    # Injects student-visited states into the shared buffer to shrink the imitation gap.
    student_action_prob_max: float = 0.2
    student_action_prob_warmup_iters: int = 3000
    # Teacher-warmup gate: skip ALL student-relevant work (the imitation update AND the
    # collect-time student forward that feeds the mix) while global_step < student_start_iters.
    # The teacher's action target is the student's only supervision; before the teacher's task
    # competence saturates the student would chase a fast-moving label. Measured on the v59 teacher
    # curve, track_linear_velocity (hence the action) saturates only at ~3000 iters (= the energy-
    # curriculum final stage = the α-mix warmup end), so distilling earlier wastes gradient on an
    # immature teacher. Gating also yields a wall-clock speedup over the warmup window (one fewer
    # actor-sized fwd/bwd per learn batch + the skipped collect fwd). The α-mix ramp is measured
    # from this offset so the student never DRIVES before it has been TRAINED. 0 ⇒ byte-identical to
    # the prior always-on student. Only meaningful with use_distilled_student=True. See plan/
    # L2T_STUDENT_RL_REVISIT_PLAN.md Exp 0. Default 3000 (Exp 0 = baseline-grade fall +
    # warmup speedup); set 0 to restore the prior always-on student (byte-identical).
    student_start_iters: int = 3000
    # Delayed-α: extra iters the student BC-warms (updates from the teacher-driven buffer) BEFORE it
    # ever DRIVES. The collect-time student fwd + α-mix ramp are anchored at
    # drive_start = student_start_iters + student_drive_delay_iters (the student *update* gate stays at
    # student_start_iters). Motivated by a probe: at student_start+~1000 the student is still
    # semi-garbage yet already drives ~α·N envs, contaminating the buffer; delaying the drive removes
    # that window. Default 2000 = ON for all L2T runs (student clean by then). 0 ⇒ drive_start ==
    # student_start_iters ⇒ byte-identical to the prior schedule. Only meaningful with
    # use_distilled_student=True.
    student_drive_delay_iters: int = 2000
    # Episode-level sample-mixing: draw the executed driver (student vs teacher) ONCE per episode
    # (resampled on env reset) instead of a fresh per-step coin. Per-step Bernoulli tethers the rollout
    # to the teacher — mean consecutive student run = 1/(1−α_mix) steps ≪ episode length — so the buffer
    # never holds sustained-drift (deploy-horizon) student states, and the student's last_action obs is
    # the teacher's action ~α_mix of steps (train≠deploy). Per-episode driving gives full-horizon student
    # rollouts the teacher relabels — coherent DAgger coverage WITHOUT freezing the teacher; α_mix then ⇒
    # fraction of envs running pure-student per episode. Only meaningful with use_distilled_student=True.
    # False ⇒ per-step mixing (byte-identical to v54).
    episode_level_mixing: bool = False
    # Freeze the teacher during Phase-2 distillation: skip the teacher's critic/actor/alpha
    # updates (and Polyak), run only the student imitation update (every batch, not gated by
    # the actor cadence). Protects the teacher's fragile low-cmd skill from regressing under
    # the buffer-reset-on-load + continued RL — the action labels stay fixed at the loaded
    # checkpoint. Only meaningful with use_distilled_student=True.
    freeze_teacher: bool = False
    # Mid-run teacher freeze for a 3-STAGE curriculum (v79): freeze the teacher ONCE global_step
    # crosses this iter (0 = never; static freeze_teacher above freezes from the start instead).
    # Stage-1/2 = normal v59 co-train (teacher live, per-episode mixing) until the threshold;
    # Stage-3 = teacher held fixed + ALL envs student-driven (alpha_mix forced to 1.0) + student
    # keeps its BC imitation update on the frozen labels. The teacher's heavy C51-critic + SAC-actor
    # updates are skipped after the threshold (the no-op speedup); only the cheap teacher forward
    # for the BC label remains. Monotonic → one-shot teacher-normalizer eval() at the transition
    # stops the labels drifting. Only meaningful with use_distilled_student=True.
    freeze_teacher_after_iters: int = 0
    # v88+ stage-3 α-mix ramp curve (L2T_ALPHA_MIX_AND_FREEZE_EXPERIMENT_PLAN §1). Replaces the
    # v79 step-jump at freeze_teacher_after_iters with a smooth ramp from student_action_prob_max
    # → 1.0 over ramp_iters. ORTHOGONAL to freeze_teacher: when ramp_iters>0 the ramp applies in
    # collect-time regardless of whether the teacher is live or frozen (live-teacher mode anchors
    # the ramp start at student_start_iters+warmup_iters so the ramp begins at end-of-Phase-2).
    # shape: "linear" (uniform), "ease_in" (p^2, convex — mild-start: slow first 30% of window,
    # fast last 70%), "ease_out" (p^0.5, concave — aggressive-start). 0 = current step-jump,
    # byte-identical to v82/v83/v79. See plan/L2T_ALPHA_MIX_AND_FREEZE_EXPERIMENT_PLAN.md.
    student_action_prob_stage3_ramp_iters: int = 0
    student_action_prob_stage3_ramp_shape: str = "linear"  # {"linear","ease_in","ease_out"}
    # --- v62 full-PG student (L2T_STUDENT_RL_REVISIT_PLAN.md Exp 1+2, unified) ---
    # Turn the BC-only student into an asymmetric SAC actor on the SHARED privileged critic, with
    # the BC term kept as a trust-region anchor. student_pg_coef>0 adds, to the student loss, the
    # teacher's own actor objective evaluated on the STUDENT (reparam-sampled action, log_std live):
    #   sac_loss = (α_s·logπ_s − qf_norm).mean(),  student_loss = imitation_coef·BC + pg_coef·sac_loss
    # where qf = min-over-ensemble E[Q](critic_obs, a_s) and qf_norm = qf/(qf.detach().std()+1e-6).
    # The detached batch-std normalization makes pg_coef SCALE-INVARIANT: the PG gradient competes
    # with BC through the critic's action-SLOPE ∂Q/∂a, which grows as the critic sharpens over
    # training and differs across reward configs (plain vs cam). Dividing by the per-batch Q spread
    # rescales that slope to O(1) so a fixed pg_coef holds the same PG-vs-BC balance throughout and
    # transfers across variants. The critic's params are NOT in student_optimizer → Q-grads are
    # discarded, the critic is uncorrupted (same as the teacher actor step). Deploy stays the
    # deterministic mean (PG only trains log_std, which deploy ignores) → export parity preserved.
    # Validity rests on episode_level_mixing: the student's (s,a) is in the critic's Bellman support
    # so Q(s,a_s) is interpolation, not extrapolation. MANDATE student_start_iters>0 (a trained
    # critic). 0 ⇒ byte-identical (PG block skipped, no extra student fwd / RNG draw).
    student_pg_coef: float = 0.0
    # Critic-warmup gate for the PG term (v66, L2T_STUDENT_RL_REVISIT_PLAN.md §Exp 5). The PG term in
    # _update_student_loss stays OFF until global_step >= max(student_start_iters, student_pg_start_iters),
    # while the student critic (student_own_critic) trains from student_start_iters regardless. On a
    # warm-start the student_qnet inits FRESH (the ckpt has none) and engaging PG immediately makes the
    # student climb a near-random Q for thousands of steps → directional noise during the fragile early
    # distillation window (the suspected source of v64's speed/stability trade). Delaying PG lets
    # Q^{π_student} Bellman-converge first, so PG starts from a good value estimate. 0 ⇒ no extra delay
    # (effective gate = student_start_iters) = byte-identical to v64. Only meaningful with
    # student_pg_coef>0 + student_own_critic (a separate critic that must mature; with the shared teacher
    # critic there is nothing to warm up).
    student_pg_start_iters: int = 0
    # Student own entropy autotune for the PG term. True ⇒ adapt α_s to hold student_target_entropy
    # (mirror teacher recipe); False with pg_coef>0 ⇒ fixed α_s = student_alpha_init. Student optimal
    # exploration ≠ teacher's, so it gets its own log_alpha (not the teacher's). Only meaningful with
    # student_pg_coef>0.
    student_use_autotune: bool = True
    student_alpha_init: float = 0.1
    student_target_entropy_ratio: float = 0.25   # target_entropy = −ratio·|A| (= teacher default)
    student_alpha_learning_rate: float = 3e-4
    # Stochastic student collect: during α-mix collect, execute the student's SAMPLED action (its own
    # live dist) instead of the deterministic mean. PG trains log_std, so sampling here keeps collect
    # on-distribution for the student's entropy objective (the synergy the old per-step Exp 2 lacked).
    # Deploy is always the mean regardless. False ⇒ deterministic collect (byte-identical).
    student_collect_stochastic: bool = False
    # Student-OWN critic for the PG term (v64, L2T_STUDENT_RL_REVISIT_PLAN.md). When True AND
    # student_pg_coef>0, the PG term scores the student action a_s under a SEPARATE C51 critic that
    # evaluates π_STUDENT — its Bellman target bootstraps the STUDENT's next action on the
    # next_student_obs (not the teacher's). The shared teacher qnet instead evaluates π_TEACHER (it
    # bootstraps the teacher's next action), which is the documented objection (b) behind every
    # rejected Q-term/PG attempt (MEMORY entry 33): teacher-Q pulls the student toward the teacher's
    # optimum, not the student's own deploy optimum (different obs / sensor noise / achievable policy).
    # A student-own critic estimating Q^{π_student} removes that misalignment — the student PG climbs
    # the value of what the DEPLOYABLE student can actually achieve. Critic obs stay the PRIVILEGED
    # critic_obs (asymmetric actor-critic: clean-state value of the partially-observed student → low
    # variance, standard). Builds student_qnet/_target/_optimizer (own C51, own polyak target) + a
    # next_student replay slot (frame-ring student_next_frame, the bootstrap obs). Default False ⇒ PG
    # (if on) uses the shared teacher critic = byte-identical to v63. Only meaningful with
    # student_pg_coef>0 + use_distilled_student; requires frame_ring_history (the next-student slot is
    # implemented on the frame-ring student path only; supports the 1-step and n-step sample paths).
    student_own_critic: bool = False
    # Multi-scale strided student window (dead-zone fallback #2 / Path B, plan §7.7). The
    # student's 0.4 s dense window (history_length=20) cannot see the multi-second velocity-
    # error build-up that drives low-cmd discharge. Give it ~2.9 s of reach at 20-frame cost by
    # GATHERING the env's longer history (history_length=L=max(idx)+1) at multiple strides into
    # T=len(idx) frames before the net. student_strided_idx = ascending buffer indices (chronological,
    # oldest→newest; the env CircularBuffer.buffer is oldest→newest). student_strided_scale_sizes =
    # per-block lengths (coarse,mid,fine) summing to T, each a uniform-Δt block → the student
    # backbone is MultiScaleSequenceBackbone (one shared encoder per block, concat). The gather
    # is applied at the runner obs-reads AND inside DeployedPolicyFlashSAC (deploy parity); the
    # replay/net/normalizer are sized at T (compact). () ⇒ disabled ⇒ dense path byte-identical
    # to v51/v54. Only meaningful with use_distilled_student=True.
    student_strided_idx: tuple[int, ...] = ()       # ascending env-history indices to gather (T frames)
    student_strided_scale_sizes: tuple[int, ...] = ()  # per-scale block lengths, sum == len(idx)

    # Per-Q-net target distributions (ablation: matches old fast_sac behavior).
    # When False (default): both Q-nets train against the min-Q distribution (current behavior).
    # When True: each Q-net trains against its own projected distribution (old fast_sac runner).
    use_per_q_target: bool = False

    # Zeta-distributed noise repetition (matches reference FlashSAC).
    # Per env, sample N ~ Zeta(zeta_mu) ∈ [1, zeta_max_n]; reuse same noise ε
    # for N consecutive steps; resample at boundaries. Eliminates IID per-step
    # jitter in replay buffer transitions, preventing critic from learning to
    # value action chatter. PMF for mu=2.0: P(N=1)≈61%, P(N=2)≈15%, mean≈1.85.
    use_zeta_noise: bool = False
    zeta_mu: float = 2.0
    zeta_max_n: int = 16

    # Mini-batches per rb.sample() call.
    # More mini-batches per sample → fewer kernel launches (gather is memory-bandwidth-bound,
    # not compute-bound, so throughput is unchanged) but higher transition-peak GPU memory.
    # Transition peak = 2 × (chunk_size × batch_size_local) rows of obs (old chunk still alive
    # when new chunk is allocated).
    # - 1: 32 gathers/iter, ~23.2 GB peak.
    # - 2: 16 gathers/iter, ~23.4 GB peak (safe fallback on 24 GB GPU).
    # - 4 (default): 8 gathers/iter, ~23.7–23.9 GB peak (tight; reduce to 2 if OOM).
    # - 8: OOM on 24 GB GPU.
    # Must divide total_updates = num_collect_steps × num_updates evenly.
    sample_chunk_size: int = 4

    @classmethod
    def from_saved_args(cls, saved: dict) -> "FlashSACConfig":
        """Build a config from a checkpoint's saved `args`, dropping any unknown keys.

        Fresh-format checkpoints store current field names directly. Legacy l2t_-prefixed
        checkpoints are migrated once by scripts/upgrade_flash_sac_checkpoints.py (the rename
        table lives there), so no per-load key remapping is needed here.
        """
        valid = cls.__dataclass_fields__
        return cls(**{k: v for k, v in saved.items() if k in valid})
