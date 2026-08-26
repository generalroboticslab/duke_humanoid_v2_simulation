#!/usr/bin/env python
"""Parallel launcher for the dynamic two-target reach/grasp benchmark.

Replaces the retired per-run shell scripts (dyn5/dyn5b/dyn5c/dyn7_launch.sh), which differed only in
their robot set and output directory -- every other knob (seed 42, 3 same-seed repeats, 10 record-trials,
``--steps 5000``, the single-cam checkpoint pin, the GPU set) was byte-identical.

Run ONE instance per host, giving every instance the SAME --hosts topology (defaults to a two-host
pair) and its own --this id:
    python dyn_sweep.py --this 15 --out dyn7 --robots g1,v2_fixed,v2,v2_single_fixed,v2_single
    python dyn_sweep.py --this 16 --out dyn7 --robots g1,v2_fixed,v2,v2_single_fixed,v2_single
    # other machines / GPU sets:
    python dyn_sweep.py --this A --hosts 'A=0-3;B=0-7' --out dyn8 --robots g1,v2

Wall-time design (minimize makespan, not maximize concurrency):
  1. DYNAMIC DISPATCH to the least-busy GPU, not a static ``n % gpus``. Cell runtimes span ~2.3x (far
     cells ~40 s/trial vs close ~17 s), so a fixed round-robin can strand a GPU with two slow cells while
     others idle. Each job instead goes to whichever GPU has the most free slots, and a GPU that clears a
     fast cell immediately pulls the next job -- self-balancing regardless of cell cost. (Pattern adapted
     from ``envs/parallel_train.Launcher``; that copy is outdated, so this is a fresh self-contained one.)
  2. BOUNDED POOL DEPTH per GPU (``--cap``, default 3). Each job alternates CPU-bound MuJoCo stepping with
     GPU-bound cuRobo trajopt; a small pool overlaps one job's CPU phase with another's GPU phase to keep
     the GPU busy WITHOUT the thrash of firing all ~6/GPU at once (pure time-slicing = overhead, longer
     makespan). At ~17 GiB/job on 46 GiB cards, 3 fits with headroom.
  3. LONGEST-CELL-FIRST (LPT) dispatch order: the expensive cells start first and the short ones backfill
     the tail; the classic makespan-minimizing heuristic for a bounded pool.

Parallel is CORRECT (not serial): the belief-freshness commit gate is TICK-based
(``_belief_tick = self._ticks``, ``_BELIEF_FRESH_TICKS=50``) with per-tick detection computed
synchronously from gaze+scene geometry, so GPU contention (slower wall-time/tick) cannot age the belief.
Loaded-vs-unloaded P flips are CUDA contact-solver nondeterminism at the margin (the paper discloses
+-0.15 on a 10-trial P), handled by pooling 3 repeats x 10 trials = 30/cell, exactly as the cited dyn3
did with parallel workers.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
VERIFY = os.path.join(os.path.dirname(__file__), "curobo_reach_verify.py")

# ============================== THE PIN TABLE -- edit HERE and only here ==============================
# One entry per variant, repo-relative. EVERY variant is pinned; there is no auto-resolution path left in
# this launcher. To retarget a variant, change its line below -- nothing else in this file, and nothing in
# ``dyn_aggregate.py``, needs touching.
#
# WHY pinning is not optional. Auto-resolution globs ``runs/<task>/*/model_*.pt`` and takes the newest,
# which is a per-host, per-day answer. It has failed three distinct ways here:
#   * HOSTS DISAGREED, silently, in a CITED sweep. one sweep ran host A on the dual-cam
#     ``2026-07-28_18-10-03`` and host B on ``2026-07-29_07-31-01`` -- md5-distinct -- so its published
#     Fix2/Act2 columns pool two policies at an unrecorded ratio. Found only by md5-ing both hosts a day
#     later.
#   * "LATEST" MOVED ONTO A LIVE RUN. `v2_best` is an ALIAS; by 2026-08-03 it pointed at
#     ``...GridGaitInitTurnInPlace``, still training. Hosts held different step counts, one held none,
#     and 14 cells died on launch with ``AssertionError: no checkpoint under runs/v2_best/*/model_*.pt``.
#   * WRONG ARCHITECTURE. ``v2_best_single``'s chain walks past the single-cam parent onto a DUAL-cam
#     ancestor, loading a 31-action net into a 29-action model (``size mismatch for action_scale``).
# g1 was the last unpinned column and had the first shape latent: one host held FOUR candidate run dirs for
# its task (two only 4 minutes apart, ``14-50-43`` and ``14-54-46``), the other held ONE. All 18 g1
# cells did land on ``14-54-46``, so those numbers are single-policy -- by luck, not construction.
#
# The pinned files live IN THE REPO under ``CKPT_DIR``, version-controlled, not under ``runs/`` (which is
# both gitignored and rsync-excluded). Two reasons that matters more than the disk cost (~20 MB each):
#   * ``runs/`` copies had to be HAND-COPIED to every sweep host, and a hand copy is exactly how two hosts
#     end up holding different bytes under the same path. In-repo files arrive by the same sync as the code,
#     so the weights and the tree that loads them move together or not at all.
#   * A ``runs/`` path names a directory that gets deleted, renamed, or overwritten by the next training
#     run. An in-repo file with the task class in its NAME survives that, and says what it came from
#     without needing this file's git history: ``<role>__<TaskClass>__<run-id>__<iteration>.pt``.
# See ``checkpoints/README.md`` for each file's md5 and the exact command that trained it.
CKPT_DIR = "mj_envs/tasks/visual_manipulation/test/checkpoints"
_G1 = f"{CKPT_DIR}/g1__G1RmaVelEstArmFlashSacStudentOnlyg1bsk2__2026-07-25_14-54-46__model_0015000.pt"
_DUAL = f"{CKPT_DIR}/dual__HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4MixedArmsCam__2026-07-28_18-10-03__model_0015000.pt"
_SINGLE = f"{CKPT_DIR}/single__HumanoidRmaVelEstArmFlashSacv2ybsk_yaw_s4SingleCam__grl2_s0__model_0015000.pt"
PINS = {
    "g1":              _G1,
    "v2":              _DUAL,
    "v2_fixed":        _DUAL,
    "v2_single":       _SINGLE,
    "v2_single_fixed": _SINGLE,
}
# Provenance record written into every sweep's output dir, so which weights produced a number is answerable
# from the data alone rather than from this file's git history at an unknown date.
PIN_RECORD = "checkpoints.txt"

# Canonical scenario order used ONLY to make the cross-host Bresenham split deterministic; dispatch order
# is the LPT ranking below.
SCENARIOS = ["left_right_close", "left_right_far", "front_back_close", "front_back_far",
             "bimanual_mixed_close", "bimanual_mixed_front_back_close"]
# Slow -> fast, from the dyn3 timings: the two "far" cells dominate, the bimanual pair next, close L/R
# shortest. Lower rank = dispatched earlier.
LPT_RANK = {"left_right_far": 0, "front_back_far": 1, "bimanual_mixed_front_back_close": 2,
            "bimanual_mixed_close": 3, "front_back_close": 4, "left_right_close": 5}

REPEATS = (1, 2, 3)


# Default topology: a two-host pair. One host's GPU 6 has an uncorrectable L2 SRAM ECC fault
# and is dropped -- an auto-detect (nvidia-smi -L) could not know a present
# device is faulty, so the usable set stays explicit. Override with --hosts for any other machine set.
DEFAULT_HOSTS = "15=0-5,7;16=0-7"
MANIFEST_PREFIX = "dyn_manifest_host"
EVAL_SOURCES = (
    "mj_envs/tasks/visual_manipulation/test/curobo_reach_verify.py",
    "mj_envs/tasks/visual_manipulation/reach_policy.py",
    "mj_envs/tasks/visual_manipulation/jacobian_reach_tracker.py",
    "mj_envs/tasks/visual_manipulation/curobo_reach_harness.py",
    "mj_envs/tasks/visual_manipulation/curobo/scene.py",
    "mj_envs/tasks/visual_manipulation/curobo/planner.py",
    "mj_envs/tasks/visual_manipulation/pickplace_scenarios.py",
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(out_dir, host):
    return os.path.join(out_dir, f"{MANIFEST_PREFIX}{host}.json")


def _manifest(host, hosts, robots, pins):
    """Immutable provenance for one host's assigned share of a dynamic sweep.

    Each host writes its own file because their ``~/tmp`` directories are distinct until results are
    collected. Keeping host identity in the filename prevents a later rsync merge from replacing another
    host's provenance record.
    """
    return {
        "format": 1,
        "host": host,
        "hosts": hosts,
        "robots": robots,
        "scenarios": SCENARIOS,
        "repeats": list(REPEATS),
        "base_seed": 42,
        "record_trials": 10,
        "steps": 5000,
        "pins": {robot: {"path": path, "md5": digest} for robot, path, digest in pins},
        "sources": {rel: _sha256(os.path.join(REPO, rel)) for rel in EVAL_SOURCES},
    }


def _prepare_output(out_dir, manifest, resume):
    """Create fresh output or validate an explicit compatible resume.

    Old cell logs and append-only checkpoint records are indistinguishable from new results after a
    cross-host rsync. Refuse them by default. Resume permits only the identical source/protocol/pin state.
    """
    path = _manifest_path(out_dir, manifest["host"])
    if resume:
        if not os.path.isfile(path):
            sys.exit(f"--resume requires matching {path}")
        with open(path) as f:
            existing = json.load(f)
        if existing != manifest:
            sys.exit("--resume manifest mismatch: source, protocol, topology, or checkpoint changed")
        return
    if os.path.exists(out_dir) and os.listdir(out_dir):
        sys.exit(f"refusing nonempty sweep output {out_dir}; choose new --out or pass --resume")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "x") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def ckpt_for(robot):
    """The pinned checkpoint for a variant, repo-relative. Unknown variant is a hard error, not a fallback
    to auto-resolution: a new variant must be pinned deliberately, and the failure mode of guessing is a
    silently mismatched policy (see ``PINS``)."""
    if robot not in PINS:
        sys.exit(f"UNPINNED VARIANT {robot!r}: add it to PINS in {os.path.basename(__file__)}. "
                 f"Known: {', '.join(sorted(PINS))}")
    return PINS[robot]


def pin_report(robots):
    """``[(robot, path, md5|MISSING)]`` for the requested variants -- what this sweep will actually load.

    md5 is computed here rather than trusted from the path because the path is what is pinned and the FILE
    is what is measured: two hosts can hold different bytes under the same name (hand-copied, ``runs/`` is
    outside the rsync), which is exactly the `dyn8` defect in a different disguise.
    """
    out = []
    for r in robots:
        rel = ckpt_for(r)
        full = os.path.join(REPO, rel)
        digest = "MISSING"
        if os.path.isfile(full):
            h = hashlib.md5()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        out.append((r, rel, digest))
    return out


def parse_gpus(spec):
    """'0-5,7' -> [0,1,2,3,4,5,7]. Accepts comma-separated ids and inclusive 'a-b' ranges."""
    ids = []
    for tok in spec.split(","):
        if "-" in tok:
            a, b = tok.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(tok))
    return ids


def parse_hosts(spec):
    """'15=0-5,7;16=0-7' -> ordered dict {host_id: [gpu ids]}. Order fixes the deterministic split."""
    hosts = {}
    for part in spec.split(";"):
        hid, gpus = part.split("=")
        hosts[hid] = parse_gpus(gpus)
    return hosts


def owner_sequence(hosts):
    """Yield the owning host for each successive job, split across hosts in proportion to their GPU count.

    Smooth weighted round-robin (weight = usable GPUs per host): interleaved so no cell's repeats clump
    onto one host, exact ratio over any window, and deterministic given the host order. Generalizes the old
    hard-wired 7:8 two-host Bresenham to any N hosts with any GPU counts.
    """
    weight = {h: len(g) for h, g in hosts.items()}
    total = sum(weight.values())
    cur = {h: 0 for h in hosts}
    while True:
        for h in hosts:
            cur[h] += weight[h]
        pick = max(hosts, key=lambda h: cur[h])
        cur[pick] -= total
        yield pick


def build_jobs(this_host, hosts, robots, out_dir, only=None):
    """Return this host's jobs as (name, cmd_tokens, log_path), in LPT dispatch order.

    Split is the weighted round-robin above over the CANONICAL (robot x scenario x repeat) enumeration;
    dispatch order is then sorted LPT. The split itself is independent of that sort.

    ``only`` (a set of ``robot__scenario__rep`` names) bypasses the host split and keeps exactly those
    cells -- used to re-run specific crashed cells through the retry path on one host.
    """
    owners = owner_sequence(hosts)
    jobs = []
    for r in robots:
        ckpt = ["--checkpoint", ckpt_for(r)]
        for s in SCENARIOS:
            for rep in REPEATS:
                owner = next(owners)
                name = f"{r}__{s}__r{rep}"
                if only is not None:
                    if name not in only:
                        continue
                elif owner != this_host:
                    continue
                cmd = [sys.executable, VERIFY, "--dynamic", "--robot", r, "--scenario", s,
                       "--walk", "--camera", "--seed", "42", "--record-trials", "10",
                       "--steps", "5000", *ckpt]
                jobs.append((LPT_RANK[s], name, cmd, os.path.join(out_dir, name + ".log")))
    jobs.sort(key=lambda j: j[0])  # LPT: slowest cells dispatched first
    return [(name, cmd, log) for _rank, name, cmd, log in jobs]


class Launcher:
    """Least-busy-GPU dispatcher with a per-GPU concurrency cap. Blocks until all jobs finish.

    Retries a job that exits nonzero (``max_retries`` times) instead of abandoning it. This is essential
    on packed L40S sweeps: ~0.1% of cuRobo trajopt runs die with `cudaErrorIllegalAddress`, a transient
    contact-solver / driver fault (MEMORY: CUDA-crash infra class) that a sibling repeat of the SAME cell
    clears. A crashed cell exits nonzero, so a fresh attempt -- requeued to the tail, so it lands on
    whatever GPU frees next rather than the same one -- almost always completes. Only cells that fail
    ``max_retries+1`` times in a row are reported failed.
    """

    def __init__(self, gpu_ids, cap, max_retries=2):
        self.gpu_ids = list(gpu_ids)
        self.cap = cap
        self.max_retries = max_retries
        self.load = {g: 0 for g in self.gpu_ids}
        self.running = []          # live subprocess.Popen, each tagged with ._gpu/._job
        self.failed = []

    def _least_busy(self):
        best, free = None, 0
        for g in self.gpu_ids:
            avail = self.cap - self.load[g]
            if avail > free:
                best, free = g, avail
        return best

    def run(self, jobs):
        queue = [(name, cmd, log, 0) for name, cmd, log in jobs]   # trailing int = attempt count
        started = 0
        while queue or self.running:
            while queue and self._least_busy() is not None:
                name, cmd, log, attempt = queue.pop(0)
                gpu = self._least_busy()
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONPATH"] = "mj_envs"
                f = open(log, "w")
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO)
                p._gpu, p._job, p._f = gpu, (name, cmd, log, attempt), f
                self.running.append(p)
                self.load[gpu] += 1
                started += 1
                tag = f" retry {attempt}" if attempt else ""
                print(f"start {name}{tag} gpu{gpu} (running {len(self.running)})", flush=True)
            still = []
            for p in self.running:
                if p.poll() is None:
                    still.append(p)
                    continue
                self.load[p._gpu] -= 1
                p._f.close()
                name, cmd, log, attempt = p._job
                if p.returncode == 0:
                    print(f"done {name}", flush=True)
                elif attempt < self.max_retries:
                    queue.append((name, cmd, log, attempt + 1))
                    print(f"RETRY {name} rc={p.returncode} (attempt {attempt + 1}/{self.max_retries})", flush=True)
                else:
                    self.failed.append((name, p.returncode))
                    print(f"FAIL {name} rc={p.returncode} after {self.max_retries} retries", flush=True)
            self.running = still
            time.sleep(0.5)
        return self.failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--this", required=True, help="id of THIS host, must be a key in --hosts")
    ap.add_argument("--hosts", default=DEFAULT_HOSTS,
                    help="topology 'id=gpus;id=gpus', gpus as '0-5,7'; same value on every host. "
                         f"default: {DEFAULT_HOSTS}")
    ap.add_argument("--out", required=True, help="directory name under ~/tmp for per-cell logs")
    ap.add_argument("--robots", required=True, help="comma-separated variants")
    ap.add_argument("--cap", type=int, default=3, help="max concurrent jobs per GPU")
    ap.add_argument("--retries", type=int, default=2, help="retries per cell on nonzero exit (CUDA crash)")
    ap.add_argument("--only", default=None,
                    help="comma-separated cell names (robot__scenario__rN) to run, bypassing host split; "
                         "re-run specific crashed cells on one host; requires --resume")
    ap.add_argument("--resume", action="store_true",
                    help="resume only an identical manifested sweep; required with --only")
    ap.add_argument("--dry", action="store_true", help="print the plan and exit without writing artifacts")
    args = ap.parse_args()

    hosts = parse_hosts(args.hosts)
    if args.this not in hosts:
        sys.exit(f"--this {args.this!r} not in --hosts {list(hosts)}")
    robots = args.robots.split(",")
    only = set(args.only.split(",")) if args.only else None
    out_dir = os.path.join(os.path.expanduser("~/tmp"), args.out)
    pins = pin_report(robots)
    missing = [r for r, _p, d in pins if d == "MISSING"]
    if missing:
        sys.exit("MISSING CHECKPOINT(S) under {}:\n{}".format(
            REPO, "\n".join(f"  {r}: {p}" for r, p, d in pins if d == "MISSING")))
    if only is not None and not args.resume:
        sys.exit("--only requires --resume")

    manifest = _manifest(args.this, args.hosts, robots, pins)
    if not args.dry:
        _prepare_output(out_dir, manifest, args.resume)
    gpu_ids = hosts[args.this]
    jobs = build_jobs(args.this, hosts, robots, out_dir, only=only)

    print(f"host {args.this}: {len(jobs)} jobs, gpus {gpu_ids}, cap {args.cap}/gpu, "
          f"retries {args.retries} -> {out_dir}", flush=True)
    # Manifest pins replace the old append-only checkpoints.txt. Each host owns a separate manifest, so
    # a repeated output name cannot mix an earlier run's checkpoint record into this sweep.
    print("pinned checkpoints:", flush=True)
    for r, path, digest in pins:
        print(f"  {r:16s} {digest[:8]}  {path}", flush=True)
    if args.dry:
        for name, _cmd, _log in jobs:
            print("  " + name)
        return

    failed = Launcher(gpu_ids, args.cap, args.retries).run(jobs)
    print(f"DYN_SWEEP_DONE host {args.this} out {args.out}: {len(jobs)} jobs, {len(failed)} failed", flush=True)
    if failed:
        print("failed: " + ", ".join(f"{n}(rc{c})" for n, c in failed), flush=True)


if __name__ == "__main__":
    main()
