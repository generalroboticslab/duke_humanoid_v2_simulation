#!/usr/bin/env python
"""Aggregate a dyn_sweep log directory into the dynamic-table numbers
(P / Tbar / E / Tsearch / Tapproach / Treach).

Per cell (``<robot>__<scenario>__rN.log``): pair the 10 per-trial ``METRICS2`` lines (those carrying
``t_sim_s``; the 11th summary line has none and is skipped) with the 10 ``mission_verdict`` lines in
trial order. P = SUCCESS fraction; Tbar = mean ``t_sim_s`` over SUCCESS trials; E = mean ``energy_J`` over
SUCCESS trials (the paper's success-only convention). Per variant: trial-weighted over all its trials.

Tbar SPLIT, from the per-trial ``PHASES`` line that follows each verdict, into THREE groups by WHY the
mission is spending time, not by which actuator group happens to move:
  * ``Tsearch`` = ACQUIRE (perception-first search+decide, mostly stationary) + DISCOVER (blind drive to
    an unvisited support anchor because the target is not yet located) -- time spent LOCATING the target.
  * ``Tapproach`` = APPROACH alone -- locomotion toward a target that has ALREADY been located (drives
    toward "Point C", the anchor shifted to the cube's live observed position; see
    ``reach_policy.py:_handle_approach``). Verified by reading ``_handle_discover``/``_handle_approach``:
    APPROACH only runs once ``_walk_current`` is committed, so it never fires before the target is known.
  * ``Treach`` = EXTEND + GRASP + RETRACT_PLAN + RETRACT + TERMINATE (the zero-twist mission-close tick) --
    physically executing the grasp once a target is both located and reached.
Every FSM phase lands in exactly one group, so Tsearch + Tapproach + Treach == Tbar exactly (mod float
rounding) for every trial. Motivation for splitting Tsearch further (2026-08-19): screening ``dyn19``
showed DISCOVER is exactly 0.00s for every actuated-camera (Act1/Act2) cell, including "far" scenarios --
the gimbal decouples visibility from body bearing entirely, so its ONLY remaining trigger (out-of-RANGE)
never fires in this benchmark's geometry. APPROACH, by contrast, is comparable in magnitude across ALL
five robots at a given target separation -- actuation eliminates the SEARCH walk, not the walk needed to
close distance on a target once it's found. Folding APPROACH into Tsearch (the 2-way split used through
2026-08-18) obscured this: a chunk of what looked like "search cost" was actually distance-to-target cost,
present even for the actuated robots.

The split's exactness only holds for logs written after 2026-08-03. Before then ``phase_ticks`` counted
planner-stalled ticks while ``t_sim_s`` did not, so the split OVERSHOT the total it decomposes by
exactly the plan wait that fell inside a search/reach phase (measured +1.480 s on a g1 trial whose
``plan_wait_s`` was 1.480). Any Tsearch/Tapproach/Treach read off ``dyn8``/``dyn9`` carries that inflation
and is not a decomposition of that sweep's Tbar.

That defect reached the paper -- Table IV printed cells whose ARM term alone exceeded the mission total --
because nothing in this output mentioned the sum. Two checks now run on every aggregate and report at the
TOP, next to the numbers they invalidate:
  * ``resid = Tbar - Tsearch - Tapproach - Treach`` per cell and per variant. Must be ~0 (all 8 phases are
    covered, so a nonzero resid means a phase name is missing from all three groups); a negative one
    raises ``INVALID DECOMPOSITION`` and means the split is not citable for that sweep.
  * the clock audit ``sum(ALL phases) - t_sim_s``, which must be <= 0 when the phase histogram and the
    mission clock are charged on the same ticks, and is POSITIVE (by that trial's ``plan_wait_s``) when
    they are not. It is what names the CAUSE of a failed resid rather than only its symptom.

Usage: python dyn_aggregate.py ~/tmp/dyn8
"""
import ast
import collections
import glob
import json
import os
import re
import sys

VARIANTS = ["g1", "v2_fixed", "v2", "v2_single_fixed", "v2_single"]           # column order
LABEL = {"g1": "G1", "v2_fixed": "Fix2", "v2": "Act2", "v2_single_fixed": "Fix1", "v2_single": "Act1"}
SCENARIOS = ["left_right_close", "left_right_far", "front_back_close", "front_back_far",
             "bimanual_mixed_close", "bimanual_mixed_front_back_close"]
SCEN_LABEL = {"left_right_close": "L/R close", "left_right_far": "L/R far",
              "front_back_close": "F/B close", "front_back_far": "F/B far",
              "bimanual_mixed_close": "L/R Human", "bimanual_mixed_front_back_close": "F/B Human"}


SEARCH_PHASES = ("ACQUIRE", "DISCOVER")                                      # locating the target
APPROACH_PHASES = ("APPROACH",)                                              # locomotion toward a located target
REACH_PHASES = ("EXTEND", "GRASP", "RETRACT_PLAN", "RETRACT", "TERMINATE")   # executing the grasp


def pin_provenance(root):
    """Report the checkpoints the sweep in ``root`` loaded, from the ``checkpoints.txt`` each host appends.

    Returns ``(lines, split)``: report lines, and True if any variant was measured under MORE THAN ONE md5.
    A split is not cosmetic -- it means that variant's column pools two policies, which is what `dyn8` did
    undetected for a day. It is reported at the TOP of the aggregate, next to the numbers it invalidates,
    because the failure mode of the previous design was that nothing in the output dir mentioned weights at
    all. Absent record (a pre-2026-08-03 sweep) is reported as unknown, not as clean.
    """
    path = os.path.join(root, "checkpoints.txt")
    if not os.path.isfile(path):
        return ["checkpoints: NO RECORD (sweep predates pin logging -- provenance unverifiable here)"], False
    seen = collections.defaultdict(set)          # robot -> {(md5, path)}
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 4:
            _host, robot, digest, ckpt = parts
            seen[robot].add((digest, ckpt))
    lines, split = ["checkpoints:"], False
    for robot in sorted(seen):
        for digest, ckpt in sorted(seen[robot]):
            lines.append(f"  {robot:16s} {digest[:8]}  {os.path.basename(ckpt)}")
        if len(seen[robot]) > 1:
            split = True
            lines.append(f"  ^^ SPLIT: {robot} ran under {len(seen[robot])} DISTINCT checkpoints -- its "
                         f"column below POOLS THAT MANY POLICIES and is not a single-policy measurement")
    return lines, split


BASE_SEED = 42          # dyn_sweep pins --seed 42; --record-trials N sweeps base_seed + trial_index
MANIFEST_PREFIX = "dyn_manifest_host"


def manifest_provenance(root):
    """Validate per-host dynamic manifests merged alongside their logs.

    Legacy sweeps predate manifests and remain readable under their documented pinned-checkpoint record.
    New sweeps must have compatible source/protocol/pin records before their numerical aggregate can support
    a paper claim.
    """
    paths = sorted(glob.glob(os.path.join(root, f"{MANIFEST_PREFIX}*.json")))
    if not paths:
        return ["manifest: NO RECORD (legacy sweep; use checkpoint record below)"], True
    manifests = []
    for path in paths:
        try:
            with open(path) as f:
                manifests.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"manifest: INVALID {os.path.basename(path)}: {exc}"], False
    canonical = manifests[0].copy()
    canonical.pop("host", None)
    for manifest in manifests[1:]:
        comparable = manifest.copy()
        comparable.pop("host", None)
        if comparable != canonical:
            return ["manifest: MISMATCH across hosts; source, protocol, or checkpoint state differs"], False
    pins = canonical.get("pins", {})
    lines = [f"manifest: {len(manifests)} host record(s), format {canonical.get('format', '?')}"]
    for robot in VARIANTS:
        pin = pins.get(robot)
        if pin is None:
            return [f"manifest: missing pin for {robot}"], False
        lines.append(f"  {robot:16s} {pin.get('md5', '')[:8]}  {os.path.basename(pin.get('path', ''))}")
    return lines, True


def repro_commands(root, failures):
    """One ``--view`` command per FAILED trial, so a failure can be watched instead of only counted.

    A trial is addressed by (robot, scenario, seed): ``dyn_sweep`` runs every cell with ``--seed 42
    --record-trials 10``, so the i-th trial block in a log is seed ``42 + i``. The three repeats share
    those seeds -- they differ only by CUDA contact-solver nondeterminism -- so the repeat number is
    reported for traceability but is NOT part of the command; a replay reproduces the seed, not the
    scheduler state. ``--record-trials`` is dropped so ``--view`` opens on that single seed.

    The checkpoint is taken from the sweep's own ``checkpoints.txt`` rather than re-resolved, since
    "latest" is a per-host per-day answer that has already produced three distinct wrong-weights
    failures here (see ``dyn_sweep``'s pin table).
    """
    ckpt = {}
    path = os.path.join(root, "checkpoints.txt")
    if os.path.isfile(path):
        for line in open(path):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 4:
                ckpt[parts[1]] = parts[3]
    lines = []
    for robot, scenario, rep, seed, reason in failures:
        pin = f" --checkpoint {ckpt[robot]}" if robot in ckpt else ""
        lines.append(f"# {robot}/{scenario} r{rep} seed={seed}: {reason}\n"
                     f"PYTHONPATH=.:mj_envs {sys.executable} "
                     f"mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py "
                     f"--robot {robot} --scenario {scenario} --dynamic --mpc --walk --camera "
                     f"--seed {seed} --steps 5000{pin} --view")
    return lines


def parse_log(path):
    """Return list of (success, t_sim_s, energy_J, t_search_s, t_approach_s, t_reach_s, t_all_s,
    e_search_J, e_approach_J, e_manip_J, e_all_J, plan_wait_s, fail_reason|None) per trial. ``t_all_s``
    sums EVERY phase (should equal ``t_search_s + t_approach_s + t_reach_s`` exactly, since the three
    groups now cover all 8 phase names) and feeds the clock audit; ``e_all_J`` is the same check for the
    ``PHASE_ENERGY`` line (should equal ``energy_J`` exactly -- both are the same per-tick rectangles,
    ``PHASE_ENERGY`` just split by phase instead of pre-summed; see ``accumulate_energy`` in
    ``curobo_reach_harness.py``). ``PHASE_ENERGY`` is absent from logs written before 2026-08-19; those
    trials get ``(0.0, 0.0, 0.0, 0.0)`` and are excluded from the energy-split decomposition check below."""
    metrics, verdicts, phases, energies = [], [], [], []
    for line in open(path, errors="ignore"):
        if line.startswith("METRICS2:") and "t_sim_s=" in line:
            e = float(re.search(r"energy_J=([\d.]+)", line).group(1))
            t = float(re.search(r"t_sim_s=([\d.]+)", line).group(1))
            pw = float(re.search(r"plan_wait_s=([\d.]+)", line).group(1))
            metrics.append((t, e, pw))
        elif line.startswith("mission_verdict:"):
            ok = "'SUCCESS'" in line
            reason = None if ok else line.split("mission_verdict:", 1)[1].strip()
            verdicts.append((ok, reason))
        elif line.startswith("PHASES:"):
            d = ast.literal_eval(line.split("PHASES:", 1)[1].strip())
            phases.append((sum(d.get(k, 0.0) for k in SEARCH_PHASES),
                           sum(d.get(k, 0.0) for k in APPROACH_PHASES),
                           sum(d.get(k, 0.0) for k in REACH_PHASES),
                           sum(d.values())))
        elif line.startswith("PHASE_ENERGY:"):
            d = ast.literal_eval(line.split("PHASE_ENERGY:", 1)[1].strip())
            energies.append((sum(d.get(k, 0.0) for k in SEARCH_PHASES),
                             sum(d.get(k, 0.0) for k in APPROACH_PHASES),
                             sum(d.get(k, 0.0) for k in REACH_PHASES),
                             sum(d.values())))
    if not energies:
        energies = [(0.0, 0.0, 0.0, 0.0)] * len(metrics)
    trials = []
    for (t, e, pw), (ok, reason), (tl, tp, tr, ta), (el, ep, er, ea) in zip(
            metrics, verdicts, phases, energies):
        trials.append((ok, t, e, tl, tp, tr, ta, el, ep, er, ea, pw, reason))
    return trials


def cell_stats(trials):
    n = len(trials)
    succ = [(t, e, tl, tp, tr, el, ep, er)
            for ok, t, e, tl, tp, tr, _ta, el, ep, er, _ea, _pw, _ in trials if ok]
    p = len(succ) / n if n else float("nan")
    mean = lambda i: sum(s[i] for s in succ) / len(succ) if succ else float("nan")  # noqa: E731
    return p, mean(0), mean(1), mean(2), mean(3), mean(4), mean(5), mean(6), mean(7), n, len(succ)


def clock_skew(trials):
    """Worst ``sum(ALL phases) - t_sim_s`` over the trials, and the plan wait on that same trial.

    ``<= 0`` is the invariant: every phase tick must also be a mission tick, so the phase histogram cannot
    outrun the clock it decomposes. The deficit is the ticks taken before the FSM entered its first phase.
    A POSITIVE skew is the pre-2026-08-03 defect -- ``phase_ticks`` charged on planner-stalled ticks that
    ``t_sim_s`` excludes -- and it equals that trial's ``plan_wait_s``, which is why the pair is returned
    together: the match is what identifies the cause rather than only flagging the symptom. (Measured
    post-fix 2026-08-06: skew exactly 0.00 with ``plan_wait_s`` 0.20/0.92, i.e. stalls now excluded from
    both sides.) Over ALL trials, not successes only -- the bug is in the instrumentation, so a failed
    trial witnesses it just as well.
    """
    worst, at_wait = 0.0, 0.0
    for _ok, t, _e, _tl, _tp, _tr, ta, _el, _ep, _er, _ea, pw, _r in trials:
        skew = ta - t
        if abs(skew) > abs(worst):
            worst, at_wait = skew, pw
    return worst, at_wait


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_repro = "--repro" in sys.argv[1:]
    root = os.path.expanduser(args[0] if args else "~/tmp/dyn7")
    by_cell = {}
    incomplete, fails, failed_trials = [], collections.Counter(), []
    for r in VARIANTS:
        for s in SCENARIOS:
            trials = []
            for rep in (1, 2, 3):
                path = os.path.join(root, f"{r}__{s}__r{rep}.log")
                if not os.path.isfile(path):
                    incomplete.append(f"{r}__{s}__r{rep} (missing)")
                    continue
                t = parse_log(path)
                if len(t) != 10:
                    incomplete.append(f"{r}__{s}__r{rep} ({len(t)} trials)")
                for i, trial in enumerate(t):
                    if not trial[0]:
                        failed_trials.append((r, s, rep, BASE_SEED + i, trial[-1]))
                trials += t
            by_cell[(r, s)] = trials
            for ok, _, _, _, _, _, _, _, _, _, _, _, reason in trials:
                if not ok:
                    fails[reason] += 1

    print(f"# aggregate {root}\n")
    manifest_lines, manifested = manifest_provenance(root)
    print("\n".join(manifest_lines))
    prov, split = pin_provenance(root)
    if not manifested:
        print("!! NON-CITABLE: manifest missing or invalid")
    elif split:
        print("!! CHECKPOINT SPLIT -- do not cite these numbers as single-policy (see above)")
    elif prov[0] != "checkpoints: NO RECORD (sweep predates pin logging -- provenance unverifiable here)":
        # Legacy append-only checkpoint records are informational once manifests are authoritative.
        print("\n".join(prov))

    # Decomposition check, reported HERE rather than beside the Tsearch/Treach table, for the same reason the
    # checkpoint split is: a defect nothing prints is a defect that ships. `dyn9` published cells whose arm
    # term alone exceeded the mission total because this sum was never taken.
    # Tolerance: t_sim_s and each phase duration are independently rounded to 3 decimals in the log text
    # (see ``round(ticks_in * dt, 3)`` at the print site), so a resid within that noise floor is not a
    # decomposition defect -- only a resid that could not arise from rounding is.
    _RESID_TOL_S = 0.01
    _RESID_TOL_J = 0.5   # energy sums round at 2 decimals in PHASE_ENERGY vs METRICS2's own 2 decimals
    alltrials = [t for r in VARIANTS for s in SCENARIOS for t in by_cell[(r, s)]]
    bad, bad_e = [], []
    for r in VARIANTS:
        for s in SCENARIOS:
            _p, tb, eb, tl, tp, tr, el, ep, er, _n, ns = cell_stats(by_cell[(r, s)])
            if ns and tb - tl - tp - tr < -_RESID_TOL_S:
                bad.append((f"{LABEL[r]}/{SCEN_LABEL[s]}", tb - tl - tp - tr))
            # Legacy logs (pre-2026-08-19) have no PHASE_ENERGY line -- el/ep/er all 0.0 by parse_log's
            # fallback, which never trips this (eb - 0 - 0 - 0 = eb, never negative past the tolerance).
            if ns and eb - el - ep - er < -_RESID_TOL_J:
                bad_e.append((f"{LABEL[r]}/{SCEN_LABEL[s]}", eb - el - ep - er))
    skew, at_wait = clock_skew(alltrials)
    print(f"\nclock audit: worst sum(all phases) - t_sim = {skew:+.2f} s "
          f"(plan_wait_s on that trial {at_wait:.2f})")
    if bad:
        print(f"!! INVALID DECOMPOSITION -- Tsearch + Tapproach + Treach EXCEEDS Tbar in {len(bad)} cell(s); "
              f"the split is NOT citable for this sweep")
        for name, resid in sorted(bad, key=lambda b: b[1]):
            print(f"     {name:<22} resid {resid:+.2f} s")
        if skew > 0:
            print(f"     cause: phase histogram outruns the mission clock by {skew:+.2f} s -- it is being "
                  f"charged on planner-stalled ticks that t_sim_s excludes (fixed 2026-08-03; re-measure, "
                  f"the inflation is not removable post hoc)")
    if bad_e:
        print(f"!! INVALID ENERGY DECOMPOSITION -- Esearch + Eapproach + Emanip EXCEEDS E in {len(bad_e)} "
              f"cell(s); the energy split is NOT citable for this sweep")
        for name, resid in sorted(bad_e, key=lambda b: b[1]):
            print(f"     {name:<22} resid {resid:+.2f} J")
    print()
    print("P / Tbar / E")
    print("scenario".ljust(11) + "".join(LABEL[r].center(20) for r in VARIANTS))
    for s in SCENARIOS:
        row = SCEN_LABEL[s].ljust(11)
        for r in VARIANTS:
            p, tb, eb, _tl, _tp, _tr, _el, _ep, _er, _n, _ = cell_stats(by_cell[(r, s)])
            row += f"{p:.2f} / {tb:5.2f} / {eb:4.0f}".center(20)
        print(row)
    print("\nTsearch / Tapproach / Treach (s, success-only; every phase covered, so the triple sums to Tbar)")
    print("scenario".ljust(11) + "".join(LABEL[r].center(24) for r in VARIANTS))
    for s in SCENARIOS:
        row = SCEN_LABEL[s].ljust(11)
        for r in VARIANTS:
            _p, _tb, _eb, tl, tp, tr, _el, _ep, _er, _n, _ = cell_stats(by_cell[(r, s)])
            row += f"{tl:4.2f}/{tp:4.2f}/{tr:5.2f}".center(24)
        print(row)
    print("\nEsearch / Eapproach / Emanip (J, success-only; sums to E)")
    print("scenario".ljust(11) + "".join(LABEL[r].center(24) for r in VARIANTS))
    for s in SCENARIOS:
        row = SCEN_LABEL[s].ljust(11)
        for r in VARIANTS:
            _p, _tb, _eb, _tl, _tp, _tr, el, ep, er, _n, _ = cell_stats(by_cell[(r, s)])
            row += f"{el:4.0f}/{ep:4.0f}/{er:5.0f}".center(24)
        print(row)
    print("\nper-variant (trial-weighted):")
    print("variant".ljust(9) + "P".center(16) + "Tbar".center(9) + "E".center(7)
          + "Tsearch".center(9) + "Tapproach".center(11) + "Treach".center(9) + "resid".center(9)
          + "Esearch".center(9) + "Eapproach".center(11) + "Emanip".center(9) + "Eresid".center(9))
    for r in VARIANTS:
        allt = [t for s in SCENARIOS for t in by_cell[(r, s)]]
        p, tb, eb, tl, tp, tr, el, ep, er, n, ns = cell_stats(allt)
        print(f"{LABEL[r]:<9}{f'{ns}/{n}={p:.3f}':^16}{tb:^9.2f}{eb:^7.0f}{tl:^9.2f}{tp:^11.2f}{tr:^9.2f}"
              f"{tb - tl - tp - tr:^+9.2f}{el:^9.0f}{ep:^11.0f}{er:^9.0f}{eb - el - ep - er:^+9.1f}")
    total = sum(len(by_cell[(r, s)]) for r in VARIANTS for s in SCENARIOS)
    nfail = sum(fails.values())
    print(f"\ntrials {total}, failures {nfail}:")
    for reason, c in fails.most_common():
        print(f"  {c:3d}  {reason}")
    if incomplete:
        print("\nINCOMPLETE cells:")
        for c in incomplete:
            print("  " + c)
    if want_repro:
        print(f"\n--view repro for each of the {len(failed_trials)} failed trial(s), run from the repo root:")
        for line in repro_commands(root, failed_trials):
            print(line)
    if not manifested or split or incomplete or bad or bad_e:
        sys.exit(2)


if __name__ == "__main__":
    main()
