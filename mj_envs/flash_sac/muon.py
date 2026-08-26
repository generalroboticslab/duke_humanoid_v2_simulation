"""Muon optimizer (Newton-Schulz orthogonalized momentum) with an AdamW fallback group.

Purpose
-------
Screen the one idea that survived the Kimi-K3 / GLM-5 / DeepSeek-V4 read-through: all three
labs independently replaced AdamW with Muon for the *matrix* parameters. Muon momentum-averages
the gradient, then replaces the resulting update with its nearest orthogonal matrix (via a
quintic Newton-Schulz iteration), so every singular value of the step is ~1. The step direction
stops being dominated by whichever few singular directions happen to carry the largest gradient.

I/O
---
Constructed like any ``torch.optim.Optimizer``. Params are split by rank at construction:
``p.ndim == 2`` (Linear weights) take the Muon path; everything else (biases, LayerNorm scales,
Conv2d kernels, the log-alpha scalar) takes a plain AdamW path. Exposes the standard
``step`` / ``zero_grad`` / ``state_dict`` / ``param_groups`` surface, so it is a drop-in for the
existing ``optim.AdamW`` construction and works unchanged under ``LambdaLR``.

Assumptions
-----------
* Only ``ndim == 2`` is orthogonalized. Conv kernels (ndim 4) *could* be folded to 2D the way
  Keller Jordan's reference does, but our only conv stack is the camera encoder and folding it
  would confound an optimizer screen with an encoder change. Left on AdamW deliberately.
* Newton-Schulz runs in bfloat16. The iteration only needs to drive singular values toward 1,
  not to converge precisely, so the reduced precision is free; this matches the reference.

Design decisions
----------------
* **One optimizer object, two update rules** rather than two optimizer objects. ``LambdaLR``
  type-checks ``isinstance(optimizer, Optimizer)`` and mutates ``group['lr']``, and the runner
  calls ``.step()`` / ``.state_dict()`` on a single handle in eight places. A single subclass
  keeps all of that untouched; the alternative was editing every callsite to drive a pair.
* **AdamW math is reimplemented here** instead of delegating to ``optim.AdamW``. Cost: the
  non-Muon params lose ``fused=True``. Baseline runs keep fused AdamW untouched -- this file is
  only reached when ``use_muon=True``.

Performance
-----------
The naive spelling of this optimizer -- a Python loop calling a single-matrix Newton-Schulz per
param -- cost **+17.8% wall-clock per training iteration**, and that is what the wave-2 runs
paid. It was LAUNCH-bound, not flop-bound: 5 steps x 3 matmuls x 8 matrices = 120 serial matmul
launches per step, on matrices whose largest is 384x768, reaching 4.84 TFLOP/s where the same GPU
does 120 TFLOP/s on one 2048x2048 bf16 matmul (4% of achievable), with cost scaling 7.90x from 1
matrix to 8 -- essentially per-matrix constant overhead.

Fixed by (a) batching every 2D param into ONE zero-padded buffer so the iteration is 15 batched
matmuls total instead of 15 per matrix (see :meth:`Muon._plan`), and (b) driving the AdamW
fallback group with ``torch._foreach_*`` instead of a per-param loop. Measured on the real
actor/qnet shapes loaded from a trained checkpoint (RTX 4090, uncontended, per optimizer step):

==========  ============  ==========  ==========  ==============
net         fused AdamW   Muon old    Muon new    vs fused AdamW
==========  ============  ==========  ==========  ==============
actor       0.292 ms      2.132 ms    0.419 ms    7.3x -> 1.4x
qnet        0.298 ms      2.125 ms    0.501 ms    7.1x -> 1.7x
==========  ============  ==========  ==========  ==============

End-to-end, the number that actually matters: a matched A/B on ONE idle L40S, both arms run
sequentially at 4096 envs for 600 iterations, with collect confirmed equal (0.2253 vs 0.2261 s)
so the comparison is clean:

=================  =========  =========  =========
arm                collect    learn      iteration
=================  =========  =========  =========
Cosine (AdamW)     0.2253 s   0.1118 s   0.3375 s
Muon (batched)     0.2261 s   0.1158 s   0.3417 s
delta              +0.4%      +3.6%      **+1.2%**
=================  =========  =========  =========

So Muon went from +17.8% to **+1.2%** per iteration -- within noise of the AdamW baseline.

How often this runs is set by the EXPERIMENT, not by the defaults in ``config.py``: the wave-2
runs override ``num_updates`` to 3, giving ``num_collect_steps`` 4 x 3 = 12 critic steps per
iteration and, via ``policy_frequency`` 2, 6 actor steps -- 18 total, confirmed against the saved
optimizer step counters (179880 and 89940 over 15000 iterations). Read the run's own ``args``
before scaling a per-step cost to a per-iteration one; the config defaults (8 -> 48 steps) would
overstate it 2.7x. Scaling a 4090 microbenchmark to an L40S run mispredicts by ~2x in the other
direction -- only the matched same-GPU A/B above settles it.

Two things measured and deliberately NOT done:

* **Bucketing by shape to cut padding waste.** Splitting into per-shape buckets cuts FLOPs 3-4x
  but runs 2-3x SLOWER (actor 0.227 -> 1.570 ms at 7 buckets); one padded batch at 55-70% waste
  beats four tight ones because these matrices cannot fill the GPU. Still true under CUDA graphs,
  so it is execution latency, not launch overhead.
* **CUDA graph capture of the NS block.** Captures cleanly and is bit-identical, but worth only
  1.05-1.26x on top of batching (0.3% of iteration time) -- not worth the capture machinery or
  its failure modes inside a live training process. ``torch.compile`` on ``step`` is similar and
  additionally cannot use ``mode="reduce-overhead"`` here ("skipping cudagraphs due to mutated
  inputs"), inherent to in-place optimizer state.

An earlier version of this docstring claimed the overhead was ~5%; that figure was taken under
collect contention which inflated the denominator. Measure per-iteration cost only on an idle GPU.
* **Nesterov momentum on by default**, matching the reference Muon; it is the variant all three
  papers' descriptions correspond to.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

# Quintic coefficients from Keller Jordan's reference Muon. Tuned so the iteration drives
# singular values into a neighbourhood of 1 in ~5 steps; they do NOT converge to exactly 1
# (the quintic has a deliberately flat, slightly-overshooting fixed point) and must not be
# "corrected" toward a cleaner-looking triple.
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315
_NS_STEPS = 5


def _ns_quintic(x: torch.Tensor, steps: int = _NS_STEPS) -> torch.Tensor:
    """Run the quintic Newton-Schulz iteration on a BATCH of matrices.

    ``x`` is ``(B, m, n)`` with ``m <= n``, already cast to bfloat16 and normalized. Every
    matrix in the batch shares one set of matmul launches, which is the whole point -- see the
    module docstring for why this path is launch-bound rather than flop-bound.
    """
    for _ in range(steps):
        a = x @ x.transpose(-2, -1)
        b = _NS_B * a + _NS_C * (a @ a)
        x = _NS_A * x + b @ x
    return x


def _orthogonalize(grad: torch.Tensor, steps: int = _NS_STEPS, eps: float = 1e-7) -> torch.Tensor:
    """Return the ~nearest semi-orthogonal matrix to ``grad`` via quintic Newton-Schulz.

    Single-matrix convenience wrapper over :func:`_ns_quintic`. ``step`` does NOT call this --
    it batches every 2D param through one padded buffer instead -- but keeping one spelling of
    the iteration means the self-check below validates the same code the optimizer runs.

    Operates on the short side: if rows > cols the matrix is transposed first so the iterated
    Gram matrix ``X @ X.T`` stays the smaller of the two, then transposed back.
    """
    x = grad.bfloat16()
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = _ns_quintic(x.unsqueeze(0), steps).squeeze(0)
    if transposed:
        x = x.T
    return x.to(grad.dtype)


class Muon(Optimizer):
    """Muon for 2D params, AdamW for the rest, behind one Optimizer handle.

    Args:
        params: iterable of parameters (not param groups -- the split is done here).
        lr: step size for the Muon (2D) group.
        adamw_lr: step size for the fallback group. Defaults to ``lr`` when None.
        momentum: Muon momentum coefficient.
        nesterov: use Nesterov-style lookahead on the Muon momentum buffer.
        weight_decay: decoupled weight decay, applied to both groups.
        betas / eps: AdamW hyperparameters for the fallback group.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        adamw_lr: float | None = None,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ):
        params = list(params)
        muon_params = [p for p in params if p.ndim == 2]
        other_params = [p for p in params if p.ndim != 2]
        groups = []
        if muon_params:
            groups.append(
                dict(
                    params=muon_params, use_muon=True, lr=lr, momentum=momentum,
                    nesterov=nesterov, weight_decay=weight_decay,
                )
            )
        if other_params:
            groups.append(
                dict(
                    params=other_params, use_muon=False,
                    lr=lr if adamw_lr is None else adamw_lr,
                    betas=betas, eps=eps, weight_decay=weight_decay,
                )
            )
        super().__init__(groups, defaults={})
        # Padded-batch layout per Muon group, built lazily and reused. Deliberately an instance
        # attribute rather than a param_group key: ``Optimizer.state_dict()`` serializes every
        # group key except ``params``, so a buffer stored there would be written into every
        # checkpoint.
        self._plans: dict[int, tuple] = {}

    def _plan(self, gi: int, params: list[torch.Tensor]) -> tuple:
        """Return the cached padded-batch layout for Muon group ``gi``.

        Every 2D param is transposed to short-side-first, then written into ONE
        ``(B, max_rows, max_cols)`` buffer so the Newton-Schulz iteration runs as 15 batched
        matmuls instead of 15 per matrix. Zero padding is exact, not an approximation: a padded
        row/column contributes nothing to ``X @ X.T`` and stays zero through the quintic, so each
        matrix evolves exactly as it would alone. The norm used to normalize is likewise
        unaffected because the padding is zero.

        Padding is deliberately NOT minimized by bucketing similar shapes. Measured on the real
        actor/qnet shapes, splitting into per-shape buckets cuts FLOPs 3-4x but runs 2-3x SLOWER
        (actor 0.227 -> 1.570 ms at 7 buckets) because these matrices are far too small to fill
        the GPU; one padded batch with 55-70% waste beats four tight ones. Still true with CUDA
        graphs removing the launch cost, so this is execution latency, not launch overhead.
        """
        key = tuple(id(p) for p in params)
        plan = self._plans.get(gi)
        if plan is not None and plan[0] == key:
            return plan
        transposed = [p.size(0) > p.size(1) for p in params]
        rows = [min(p.shape) for p in params]
        cols = [max(p.shape) for p in params]
        buf = torch.zeros(len(params), max(rows), max(cols),
                          device=params[0].device, dtype=torch.bfloat16)
        dst = [buf[i, :rows[i], :cols[i]] for i in range(len(params))]
        # Newton-Schulz gives every singular value ~1 regardless of shape, so a wide matrix would
        # otherwise take a much larger Frobenius step than a tall one at the same lr. Rescale by
        # the aspect ratio to equalize them.
        scales = [max(1.0, p.size(0) / p.size(1)) ** 0.5 for p in params]
        plan = (key, buf, dst, transposed, rows, cols, scales)
        self._plans[gi] = plan
        return plan

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for gi, group in enumerate(self.param_groups):
            lr = group["lr"]
            wd = group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]

            if group["use_muon"]:
                _, buf, dst, transposed, rows, cols, scales = self._plan(gi, params)
                momentum = group["momentum"]
                bufs = []
                for p in params:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    bufs.append(state["momentum_buffer"])
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, grads)
                gs = torch._foreach_add(grads, bufs, alpha=momentum) if group["nesterov"] else bufs

                torch._foreach_copy_(dst, [g.T if t else g for g, t in zip(gs, transposed)])
                buf.div_(buf.flatten(1).norm(dim=1).add_(1e-7).view(-1, 1, 1))
                out = _ns_quintic(buf)

                if wd:
                    torch._foreach_mul_(params, 1.0 - lr * wd)
                for i, p in enumerate(params):
                    u = out[i, :rows[i], :cols[i]]
                    p.add_((u.T if transposed[i] else u).to(p.dtype), alpha=-lr * scales[i])
            else:
                beta1, beta2 = group["betas"]
                exp_avgs, exp_avg_sqs = [], []
                for p in params:
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                # Every param in this group is stepped by the same backward, so their step
                # counters never diverge; one bias correction covers the batch.
                t = self.state[params[0]]["step"]
                torch._foreach_mul_(exp_avgs, beta1)
                torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t
                denom = torch._foreach_div(exp_avg_sqs, bias2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, group["eps"])
                if wd:
                    torch._foreach_mul_(params, 1.0 - lr * wd)
                torch._foreach_addcdiv_(params, exp_avgs, denom, value=-lr / bias1)

        return loss


if __name__ == "__main__":
    torch.manual_seed(0)

    # Newton-Schulz must pull an ill-conditioned matrix into the quintic's fixed-point band.
    # That band is ~[0.67, 1.15], NOT exactly 1 -- the coefficients trade exactness for speed
    # of approach, so asserting near-1 here would be asserting a property Muon does not have.
    BAND = (0.6, 1.3)
    g = torch.randn(64, 32) @ torch.diag(torch.logspace(0, -2, 32))  # cond 1e2, gradient-like
    s = torch.linalg.svdvals(_orthogonalize(g).float())
    assert BAND[0] < s.min() and s.max() < BAND[1], f"outside band: {s.min()} .. {s.max()}"

    # cond 1e3 does NOT reach the band in the default 5 steps (measured min ~0.14); it does by
    # 10. Pinned so a future coefficient/step edit that changes this trade-off is visible.
    g_stiff = torch.randn(64, 32) @ torch.diag(torch.logspace(0, -3, 32))
    assert torch.linalg.svdvals(_orthogonalize(g_stiff).float()).min() < 0.5
    s = torch.linalg.svdvals(_orthogonalize(g_stiff, steps=10).float())
    assert BAND[0] < s.min() and s.max() < BAND[1], f"10 steps: {s.min()} .. {s.max()}"

    # Non-square both ways, to exercise the transpose branch.
    for shape in [(16, 64), (64, 16)]:
        s = torch.linalg.svdvals(_orthogonalize(torch.randn(*shape)).float())
        assert BAND[0] < s.min() and s.max() < BAND[1], f"{shape}: {s.min()} .. {s.max()}"

    # The AdamW fallback group must track torch's own AdamW closely on a 1D param.
    ref_p = torch.nn.Parameter(torch.randn(32))
    our_p = torch.nn.Parameter(ref_p.detach().clone())
    ref_opt = torch.optim.AdamW([ref_p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.01)
    our_opt = Muon([our_p], lr=1e-2, betas=(0.9, 0.95), weight_decay=0.01)
    for _ in range(20):
        grad = torch.randn(32)
        ref_p.grad, our_p.grad = grad.clone(), grad.clone()
        ref_opt.step()
        our_opt.step()
    assert torch.allclose(ref_p, our_p, atol=1e-5), (ref_p - our_p).abs().max()

    # A 2D param must take the Muon path, a 1D param the AdamW path.
    m = torch.nn.Linear(8, 4)
    opt = Muon(m.parameters(), lr=1e-3)
    assert [g["use_muon"] for g in opt.param_groups] == [True, False]
    m(torch.randn(2, 8)).sum().backward()
    opt.step()
    assert "momentum_buffer" in opt.state[m.weight]
    assert "exp_avg" in opt.state[m.bias]

    # The batched/padded Muon path must match a literal per-matrix reference. This is the check
    # that the zero-padding argument in `_plan` is actually exact -- a wrong pad, a wrong
    # transpose flag, or a scale applied to the wrong row of the batch all show up here.
    shapes = [(64, 16), (16, 64), (32, 32), (8, 48)]   # tall, wide, square, and the batch max
    ref_ps = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
    bat_ps = [torch.nn.Parameter(p.detach().clone()) for p in ref_ps]
    bat_opt = Muon(bat_ps, lr=1e-2, momentum=0.9, weight_decay=0.01)
    ref_state = [torch.zeros_like(p) for p in ref_ps]
    for _ in range(5):
        gs = [torch.randn_like(p) for p in ref_ps]
        for p, g in zip(bat_ps, gs):
            p.grad = g.clone()
        bat_opt.step()
        with torch.no_grad():                            # literal transcription of the old loop
            for p, g, mbuf in zip(ref_ps, gs, ref_state):
                mbuf.mul_(0.9).add_(g)
                u = _orthogonalize(g.add(mbuf, alpha=0.9))
                p.mul_(1.0 - 1e-2 * 0.01)
                p.add_(u, alpha=-1e-2 * max(1.0, p.size(0) / p.size(1)) ** 0.5)
    for r, b in zip(ref_ps, bat_ps):
        assert torch.allclose(r, b, atol=1e-4), f"batched != reference: {(r - b).abs().max()}"

    # A param whose grad is None must be skipped without disturbing the batch layout of the rest.
    partial = Muon([torch.nn.Parameter(torch.randn(8, 4)) for _ in range(3)], lr=1e-3)
    g0, g2 = partial.param_groups[0]["params"][0], partial.param_groups[0]["params"][2]
    g0.grad, g2.grad = torch.randn(8, 4), torch.randn(8, 4)
    before = partial.param_groups[0]["params"][1].detach().clone()
    partial.step()
    assert torch.equal(partial.param_groups[0]["params"][1], before), "grad-less param was updated"

    # LambdaLR compatibility -- the runner wraps this handle in one.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5)
    sched.step()
    assert all(abs(g["lr"] - 5e-4) < 1e-12 for g in opt.param_groups)

    # Per-group lambdas: the runner anneals the Muon group ONLY, leaving the AdamW fallback at a
    # constant baseline lr. If group ORDER ever changed, or LambdaLR stopped accepting a list, the
    # cosine would silently land on the wrong group and the experiment would be a plain re-seed.
    import math as _math

    m2 = torch.nn.Linear(8, 4)
    opt2 = Muon(m2.parameters(), lr=1e-3, adamw_lr=2e-4)
    assert [g["use_muon"] for g in opt2.param_groups] == [True, False], "group order changed"
    DECAY, MINF = 100, 0.05

    def _cos(step, total=DECAY, mn=MINF):
        return mn + (1.0 - mn) * 0.5 * (1.0 + _math.cos(_math.pi * min(step / total, 1.0)))

    sched2 = torch.optim.lr_scheduler.LambdaLR(
        opt2, [_cos if g["use_muon"] else (lambda s: 1.0) for g in opt2.param_groups]
    )
    for _ in range(DECAY):
        sched2.step()
    muon_g, adamw_g = opt2.param_groups
    assert abs(muon_g["lr"] - 1e-3 * MINF) < 1e-12, f"muon lr not annealed: {muon_g['lr']}"
    assert abs(adamw_g["lr"] - 2e-4) < 1e-12, f"adamw lr must stay constant: {adamw_g['lr']}"
    # Halfway through, cosine is at the midpoint -- catches an off-by-one or a linear-vs-cosine mixup.
    opt3 = Muon(torch.nn.Linear(8, 4).parameters(), lr=1e-3, adamw_lr=2e-4)
    sched3 = torch.optim.lr_scheduler.LambdaLR(
        opt3, [_cos if g["use_muon"] else (lambda s: 1.0) for g in opt3.param_groups]
    )
    for _ in range(DECAY // 2):
        sched3.step()
    mid = 1e-3 * (MINF + (1.0 - MINF) * 0.5)
    assert abs(opt3.param_groups[0]["lr"] - mid) < 1e-9, opt3.param_groups[0]["lr"]

    print("muon.py self-check OK")
