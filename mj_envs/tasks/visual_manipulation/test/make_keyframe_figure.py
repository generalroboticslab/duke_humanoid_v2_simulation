"""Assemble the paper figure: one row per scenario, one column per mission stage, from the FSM keyframe
PNGs + manifests written by ``curobo_reach_verify.py --record-keyframes``.

Why a script and not a manual crop: the keyframe capture emits a VARIABLE number of frames per scenario (a
hidden cube adds ``DISCOVER``, a two-visit mission repeats the ``EXTEND..RETRACT`` block, and
``--keyframe-every`` adds intra-phase samples), so no fixed index maps to the same mission stage across rows.
Columns are therefore resolved SEMANTICALLY -- "the frame 70% of the way through EXTEND" -- against each
manifest's own phase spans, which is what makes the six rows comparable.

Only a manifest with ``"pass": true`` is eligible: a figure panel must come from a mission the grader
accepted, and each scenario contributes the lowest passing seed (deterministic, not cherry-picked).

Capture runs FIRST and in PARALLEL from inside this script. The six scenarios are independent processes, so
serializing them (the old readme shell loop) paid six cold starts back to back for no reason; fanning them out
makes the sweep GPU-bound instead. ``--capture missing`` (the default) REUSES any scenario that already has a
passing manifest, so re-running to retune the layout costs zero simulation.

Run (one command -- captures fan out, then the figure assembles):
  /home/grl/repo/micromamba/envs/py312/bin/python \
      mj_envs/tasks/visual_manipulation/test/make_keyframe_figure.py \
      --keyframe-dir mj_envs/tasks/visual_manipulation/media/keyframes_6scenario
Writes ``<keyframe-dir>/keyframe_figure.pdf`` (vector text over raster panels) and ``.png`` at 300 dpi.

Main figure is 6x5 (initial, extend, pre-grasp, grasp, retract): every manipulation panel contains
visible arm motion and the active object. Manifests retain ALL FSM transition frames for a supplementary
full-state sequence.
"""

import argparse
import concurrent.futures
import glob
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

# Row order + display labels. Keys are ``--scenario`` values; the figure drops ``shelf`` (six rows).
# Labels are the SCENARIO NAMES PRINTED IN ``main.tex`` Table IV, verbatim. An earlier draft said "(near)"
# for ``close`` and "Mixed" for the human-handoff rows, so no row label in the figure matched any row name in
# the table it illustrates. Row ORDER also matches the table.
SCENARIOS: dict[str, str] = {
    "left_right_close": "L/R\nclose",
    "left_right_far": "L/R\nfar",
    "front_back_close": "F/B\nclose",
    "front_back_far": "F/B\nfar",
    "bimanual_mixed_close": "L/R\nHuman",
    "bimanual_mixed_front_back_close": "F/B\nHuman",
}

# Paper panel spec: all four columns are manipulation frames, never static FSM bookkeeping.
#
# The old 6-column draft had ACQUIRE / APPROACH / TERMINATE. They were semantically valid FSM states but bad
# figure panels: ACQUIRE and TERMINATE repeat the parked home pose, APPROACH is absent in near-spawn rows, and
# none prove arm motion. Four manipulation moments make every cell satisfy the figure invariant: robot, task
# object, and moving arm are all visible. Each row depicts its FIRST completed visit -- a second visit repeats
# the same reach-grasp-retract mechanism and belongs in a supplementary sequence, not a mixed-time main panel.
#
# Fractions are within a phase's first contiguous run. ``--keyframe-every 0.5`` supplies the interior samples;
# phase transitions alone cannot show the reach motion because EXTEND entry is still the ready posture.
STAGES: tuple[tuple[str, tuple[tuple[tuple[str, ...], float], ...]], ...] = (
    # Initial frame: mission tick 0, robot at its ready posture. Carries the scenario's task layout -- which
    # supports, how many cubes, where they sit -- which the manipulation columns crop too tightly to read.
    ("Initial",   ((("ACQUIRE",), 0.00),)),
    ("Extend",    ((("EXTEND",), 0.30),)),
    ("Pre-grasp", ((("EXTEND",), 0.70),)),
    ("Grasp",     ((("GRASP",), 1.00),)),
    ("Retract",   ((("RETRACT",), 0.50),)),
)


# Chase-cam calibration for a FIGURE PANEL, not for the MP4: pulled in from the shipped 1.9 m so a 60 mm cube
# and the jaw gap read at panel size, and aimed above the table plate (0.68) because at 1.35 m that aim crops
# the head off. Handoff scenarios need the lens on the robot's left-front (215) or the human model occludes the
# arm. Passed as env overrides, and only when the caller has not already exported them.
#
# ``WALK_CAM_DISTANCE_M`` is only the TIGHT end of the ramp: the recorder pulls the lens back to
# ``RECORD_WALK_CAMERA_DISTANCE_WIDE`` while the robot is acquiring/walking, so the Initial column frames the
# whole task layout while the manipulation columns still frame a 60 mm cube. A per-scenario distance table was
# tried first and dropped -- it fixed the Initial panel by shrinking every OTHER panel in the same row.
CAM_ENV = {"WALK_CAM_DISTANCE_M": "1.35", "WALK_CAM_LOOKAT_Z": "0.95", "MUJOCO_GL": "egl"}
YAW_DEG = {s: "215" if s.startswith("bimanual") else "145" for s in SCENARIOS}

VERIFY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_reach_verify.py")


def _capture_one(keyframe_dir: str, robot: str, scenario: str, seed: int, every: float) -> tuple[str, bool, float]:
    """Run ONE graded mission as a subprocess, writing that scenario's keyframe PNGs + manifest.

    Subprocess rather than in-process loop: cuRobo/MuJoCo hold global GPU state, so six scenarios in one
    interpreter must run one after another, while six processes overlap on the same GPU. Env (not CLI flags)
    carries the camera because that is the override surface ``curobo_reach_verify`` already exposes."""
    env = os.environ.copy()
    for k, v in CAM_ENV.items():
        env.setdefault(k, v)
    env.setdefault("WALK_CAM_YAW_OFFSET_DEG", YAW_DEG[scenario])   # per-scenario, still caller-overridable
    cmd = [sys.executable, VERIFY, "--robot", robot, "--scenario", scenario,
           "--walk", "--camera", "--dynamic", "--mpc", "--seed", str(seed),
           "--record-keyframes", keyframe_dir, "--keyframe-every", str(every)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    ok = "VERDICT: PASS" in res.stdout
    if not ok:
        tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-5:])
        print(f"[capture] {scenario}: FAIL ({dt:.0f}s)\n{tail}", flush=True)
    else:
        print(f"[capture] {scenario}: PASS ({dt:.0f}s)", flush=True)
    return scenario, ok, dt


def _capture(keyframe_dir: str, robot: str, scenarios: list[str], seed: int, every: float, jobs: int) -> None:
    """Fan the pending captures across ``jobs`` concurrent subprocesses (threads only wait on subprocesses).

    ``jobs`` is a GPU-memory budget, not a core count: each worker holds its own cuRobo world + EGL context."""
    if not scenarios:
        print("[capture] nothing to capture (all scenarios already have a passing manifest)")
        return
    os.makedirs(keyframe_dir, exist_ok=True)
    print(f"[capture] {len(scenarios)} scenario(s), {jobs} at a time: {', '.join(scenarios)}", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda s: _capture_one(keyframe_dir, robot, s, seed, every), scenarios))
    serial = sum(r[2] for r in results)
    wall = time.time() - t0
    failed = [r[0] for r in results if not r[1]]
    print(f"[capture] done in {wall:.0f}s wall ({serial:.0f}s serial, {serial / max(wall, 1e-9):.1f}x)"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""), flush=True)


def _load_manifest(keyframe_dir: str, robot: str, scenario: str) -> dict | None:
    """Lowest-seed PASSING manifest for one scenario, or None.

    Sorted by seed number rather than filename so ``seed9`` does not outrank ``seed10`` lexically."""
    paths = glob.glob(os.path.join(keyframe_dir, f"{robot}_{scenario}_seed*_keyframes.json"))

    def _seed(p: str) -> int:
        return int(os.path.basename(p).split("_seed")[1].split("_")[0])

    for path in sorted(paths, key=_seed):
        with open(path) as f:
            manifest = json.load(f)
        if manifest.get("pass"):
            manifest["_path"] = path
            manifest["_seed"] = _seed(path)
            return manifest
    return None


def _pick(rows: list[dict], phases: tuple[str, ...], frac: float, t_min: float) -> dict | None:
    """Frame nearest ``frac`` through the FIRST contiguous run of any ``phases`` entry at or after ``t_min``.

    Two constraints, both learned from a wrong first version. (1) Contiguous run, not "every frame with this
    phase": a two-visit mission re-enters EXTEND/GRASP/RETRACT, and pooling the occurrences made ``frac``
    interpolate ACROSS visits -- the RETRACT column showed visit 1 while the GRASP column beside it showed
    visit 2. (2) ``t_min`` from the previously chosen column, so the row reads left-to-right in mission time;
    without it the same pooling produced a RETRACT panel timestamped 12 s BEFORE its GRASP panel. Together
    they pin every column but the last to a single visit, which is the story one row can actually tell.

    ``frac`` is over the run's own frames (entry + ``--keyframe-every`` samples), so it means the same thing
    whether the phase lasted 0.04 s or 4 s."""
    for phase in phases:
        run: list[dict] = []
        for row in rows:                          # rows are emitted in mission order
            if row["phase"] == phase and row["t_s"] >= t_min:
                run.append(row)
            elif run:
                break                             # run ended; take the first one, not a later re-entry
        if run:
            return run[min(int(round(frac * (len(run) - 1))), len(run) - 1)]
    return None


def _crop(img: np.ndarray, w_frac: float, h_frac: float) -> np.ndarray:
    """Centered crop. Default ``0.5625 x 1.0`` turns the 2560x1440 capture into a 1:1 panel.

    Square is the paper aspect: the figure is a 6-row strip at ``\\columnwidth``, so panel width is the scarce
    axis and the 16:9 side margins spend it on empty floor rather than on the robot.

    KNOWN COST, do not rediscover it: the FAR scenarios park their second bench within a few percent of the
    frame edge (measured: coloured content spans columns 0..0.95 of the width on ``left_right_far``'s opening
    frame), so a 0.5625 centre crop drops that bench's cube from the ``Initial`` panel of ``L/R far``, and
    clips the human to a shoulder in ``L/R Human``'s. Both are layout context, not mission evidence, and the
    manipulation columns are unaffected. Pass ``--crop 1.0 1.0`` to get the full frame back."""
    h, w = img.shape[:2]
    cw, ch = int(w * w_frac), int(h * h_frac)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keyframe-dir", required=True, help="directory written by --record-keyframes")
    ap.add_argument("--robot", default="v2")
    ap.add_argument("--out", default=None, help="output stem (default <keyframe-dir>/keyframe_figure)")
    ap.add_argument("--width-in", type=float, default=3.5,
                    help="figure width in inches. MUST equal the width it is included at, or every font "
                         "rescales: IEEEtran \\columnwidth is 3.5 in, \\textwidth is 7.14 in. The 13.0 in "
                         "default this replaced was included at \\columnwidth, shrinking 9 pt titles to ~2 pt")
    ap.add_argument("--crop", type=float, nargs=2, default=(0.5625, 1.0), metavar=("W", "H"),
                    help="centered crop as a fraction of each 2560x1440 panel; the 0.5625 default is 1:1 "
                         "(1440x1440). Pass '1.0 1.0' for the full 16:9 frame")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--capture", choices=("missing", "all", "none"), default="missing",
                    help="missing: reuse scenarios that already have a passing manifest (default); "
                         "all: re-simulate every row; none: assemble from what is on disk")
    ap.add_argument("--jobs", type=int, default=3,
                    help="concurrent capture subprocesses; each holds its own cuRobo world + EGL context")
    ap.add_argument("--seed", type=int, default=42, help="mission seed for captures")
    ap.add_argument("--keyframe-every", type=float, default=0.5,
                    help="intra-phase sampling period (s); phase-entry frames alone miss the reach motion")
    args = ap.parse_args()

    if args.capture != "none":
        pending = [s for s in SCENARIOS
                   if args.capture == "all" or _load_manifest(args.keyframe_dir, args.robot, s) is None]
        _capture(args.keyframe_dir, args.robot, pending, args.seed, args.keyframe_every, args.jobs)

    out_stem = args.out or os.path.join(args.keyframe_dir, "keyframe_figure")
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

    panels: list[list[tuple[np.ndarray, float] | None]] = []
    row_labels: list[str] = []
    seeds: list[int] = []
    for scenario, label in SCENARIOS.items():
        manifest = _load_manifest(args.keyframe_dir, args.robot, scenario)
        if manifest is None:
            print(f"[figure] SKIP {scenario}: no passing manifest in {args.keyframe_dir}")
            continue
        rows = manifest["keyframes"]
        picked: list[dict | None] = []
        t_min = 0.0
        for _header, candidates in STAGES:
            hit = None
            for phases, frac in candidates:
                hit = _pick(rows, phases, frac, t_min)
                if hit is not None:
                    break
            picked.append(hit)
            if hit is not None:
                t_min = hit["t_s"]
        cells: list[tuple[np.ndarray, float] | None] = []
        for row in picked:
            if row is None:
                cells.append(None)
                continue
            img = plt.imread(os.path.join(args.keyframe_dir, row["file"]))
            cells.append((_crop(img, *args.crop), row["t_s"]))
        panels.append(cells)
        row_labels.append(label)
        seeds.append(manifest["_seed"])
        chosen = ["n/a" if r is None else "{}@{}s".format(r["phase"], r["t_s"]) for r in picked]
        print(f"[figure] {scenario} seed {manifest['_seed']}: {chosen}")

    if not panels:
        raise SystemExit(f"no passing manifests under {args.keyframe_dir}")

    n_rows, n_cols = len(panels), len(STAGES)
    first = next(cell for row in panels for cell in row if cell is not None)
    panel_h, panel_w = first[0].shape[:2]
    # Margins are budgeted in INCHES, not axes fractions: the row-label gutter and the title band hold text at
    # a fixed point size, so a fraction that fits at 13 in starves them at 3.5 in.
    # 0.39 in = widest row label ("Human", 6.5 pt) at 0.315 in plus the 4 pt labelpad. Was 0.46, which
    # left a visible blank strip at the far left of the figure once it was set at \columnwidth.
    label_in, title_in = 0.39, 0.26
    cell_w = (args.width_in - label_in) / n_cols
    fig_h = cell_w * panel_h / panel_w * n_rows + title_in
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.width_in, fig_h))
    axes = np.atleast_2d(axes)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("0.75")
            cell = panels[r][c]
            if cell is None:
                ax.set_facecolor("0.94")
                ax.text(0.5, 0.5, "phase not entered", ha="center", va="center", color="0.45",
                        style="italic", fontsize=5.5, transform=ax.transAxes)
                continue
            img, t_s = cell
            ax.imshow(img)
            # Time is the point of a "timed keyframe": it is the mission clock the METRICS tables quote.
            ax.text(0.04, 0.96, f"{t_s:.1f} s", transform=ax.transAxes, fontsize=5.5, color="white",
                    va="top", bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.6))
            if r == 0:
                ax.set_title(STAGES[c][0], fontsize=7, pad=2, linespacing=0.95)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=6.5, rotation=0, ha="right", va="center",
                              labelpad=4, linespacing=1.0)
    fig.subplots_adjust(wspace=0.012, hspace=0.012, left=label_in / args.width_in, right=0.998,
                        top=1.0 - title_in / fig_h, bottom=0.004)
    fig.savefig(f"{out_stem}.pdf", dpi=args.dpi)
    fig.savefig(f"{out_stem}.png", dpi=args.dpi)
    # Seed is provenance, so it must reach the caption even though it is no longer printed per row (six
    # identical "seed 42" labels cost gutter width that the row names need at \columnwidth).
    seed_note = f"seed {seeds[0]}" if len(set(seeds)) == 1 else f"MIXED SEEDS per row: {dict(zip(row_labels, seeds))}"
    print(f"[figure] wrote {out_stem}.pdf and {out_stem}.png ({n_rows}x{n_cols}, {args.width_in} in wide, {seed_note})")
    print(f"[figure] include at EXACTLY {args.width_in} in or the fonts rescale: "
          f"\\includegraphics[width={'\\columnwidth' if abs(args.width_in - 3.5) < 0.05 else '\\textwidth'}]{{...}}")


if __name__ == "__main__":
    main()
