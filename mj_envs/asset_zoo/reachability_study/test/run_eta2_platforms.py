"""Compute pairwise visible-reachable coverage eta_2 for every column of the paper's Fig. 2.

Single-target eta answers "can some camera see this one point?", which a neck that swings both eyes
together answers well -- so eta is structurally blind to camera INDEPENDENCE, the property the
hardware argument actually rests on. eta_2 asks "can the rig see two points at once?", where a
coupled neck has one viewing direction and two separated targets must share a frustum. This script
produces that number for all eight platforms from one kernel (`gpu_visibility.score_pairs`), so the
comparison is not an artifact of scoring our own rigs differently from everyone else's.

Sampling matches `camera_count_ablation._rig_stats`'s headline scalar exactly, per platform: x_i
dexterity-weighted over the reachable set, x_j uniform in the bounding box of all voxel centers,
n_pairs=20000, seed=0. That is the convention the published ours-actuated 0.9556 uses, so the new
columns are directly comparable to it.

Three invariants are checked rather than assumed:

  * eta_2 <= eta on the SAME x_i sample. Covering a pair requires seeing x_a, so this cannot fail
    unless the kernel is wrong. eta is recomputed here with `score_targets` rather than read from a
    sidecar, so both sides come from the same kernel and the bound is exact.
  * As separation goes to zero, eta_2 approaches eta: with x_j a few centimetres from x_i, seeing
    both is nearly the same event as seeing x_i. The paper states this degeneracy as a Remark; here
    it is measured.
  * Rigs with no aim group are invariant to the aim candidates, since they have exactly one
    realizable state. Verified by construction (`score_pairs` runs only the rest candidate for them)
    and reported so the claim is visible in the output.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import tyro

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mj_envs.asset_zoo.reachability_study.gpu_visibility import _aim_groups, _ADAPTERS, score_pairs, score_targets
from mj_envs.asset_zoo.reachability_study.plot_workspace_curobo import _reach_visible_case, _mirror_to_paper
# Owned here, NOT imported from `camera_count_ablation`. These edges are not only the plot binning:
# `_separation_curve` draws its pair radii uniformly over [edges[0], edges[-1]], so they also set the
# separation distribution behind every scalar in the manifest -- including the eta_2 that
# `plot_workspace_curobo` prints in the Fig. 2 titles. While this was imported, retuning the
# camera-count figure's bins (`arange(0.1, 1.301, 0.2)`) silently moved that sampling range from
# 0.0-1.2 m to 0.1-1.3 m and invalidated the published manifest, which `_cached_rows` then rejected
# wholesale. The camera-count figure is free to rebin; this sampling convention is frozen because the
# paper cites numbers drawn from it. Change it only with `--refresh` and a manifest regeneration.
_SEP_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2])

_N_PAIRS = 20_000
_SEED = 0
_NEAR_SEP_M = 0.03  # degeneracy probe: x_j this far from x_i, isotropic
_MIN_BIN = 50       # below this a bin fraction is noisier than the effect, so it is left unplotted
# Fig. 2 column order, as (cli key, gpu_visibility adapter key, display name). Names match the Fig. 2
# panel titles exactly, so the two figures cannot disagree about what a platform is called.
_COLUMNS = (
    ("v2_fixed", "humanoid_v21_fixed", "Ours (fixed)"),
    ("v2", "humanoid_v21", "Ours (actuated)"),
    ("g1", "unitree_g1", "G1"),
    ("toddlerbot", "toddlerbot", "ToddlerBot"),
    ("booster_t1", "booster_t1", "Booster T1"),
    ("fourier_gr3", "fourier_gr3", "Fourier GR-3"),
    ("pal_talos", "pal_talos", "PAL TALOS"),
    ("apptronik_apollo", "apptronik_apollo", "Apptronik Apollo"),
)
# Ours carries a confidence interval and a heavy stroke; the six external platforms are drawn thin so
# the reading is "one flat line against a falling bundle" rather than eight equally weighted curves.
_OURS_STYLE = {
    "v2": dict(color="tab:green", lw=1.9, marker="o", ms=3.5, zorder=5),
    "v2_fixed": dict(color="tab:red", lw=1.9, marker="s", ms=3.5, zorder=5),
}
_EXTERNAL_COLORS = ("#7f7f7f", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf")


def _separation_curve(adapter_key: str, reach_pos: np.ndarray, weights, lo, hi, rng) -> tuple:
    """Pairwise coverage conditioned on target separation, for one platform.

    The headline scalar draws x_j uniformly in the workspace box, which almost never lands within
    0.2 m of x_i and so leaves the near bins empty. Here x_j is proposed at a radius drawn uniformly
    over the bin range in an isotropic direction instead. Every plotted value is a mean CONDITIONAL
    on separation, which is what the x axis already conditions on, so reweighting the proposal in
    that same variable leaves each bin unbiased and only moves samples to where they are needed.

    Bins past a small robot's own extent lose their samples to the box rejection below and fall under
    `_MIN_BIN`, so short platforms truncate rather than reporting a noisy tail. That truncation is
    informative and is left visible.
    """
    ia = rng.choice(len(reach_pos), size=_N_PAIRS, p=weights)
    radius = rng.uniform(_SEP_EDGES[0], _SEP_EDGES[-1], size=_N_PAIRS)
    step = rng.normal(size=(_N_PAIRS, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    aux = (reach_pos[ia] + radius[:, None] * step).astype(np.float32)
    keep = ((aux >= lo) & (aux <= hi)).all(1)   # x_j must stay inside the workspace box
    aux, ia = aux[keep], ia[keep]

    covered = score_pairs(adapter_key, reach_pos[ia], aux)
    sep = np.linalg.norm(reach_pos[ia] - aux, axis=1)
    bin_id = np.digitize(sep, _SEP_EDGES) - 1
    curve, curve_n = [], []
    for b in range(len(_SEP_EDGES) - 1):
        m = bin_id == b
        curve.append(float(covered[m].mean()) if m.sum() >= _MIN_BIN else None)
        curve_n.append(int(m.sum()))
    return curve, curve_n


def _run(cli_key: str, adapter_key: str, name: str) -> dict:
    case = _reach_visible_case(cli_key)
    reach = case["reached"]
    pos = case["points"]
    reach_pos = pos[reach].astype(np.float32)
    dex = np.nan_to_num(case["dexterity"][reach], nan=0.0)
    weights = dex / dex.sum() if dex.sum() > 0 else None
    lo, hi = pos.min(0), pos.max(0)

    rng = np.random.default_rng(_SEED)
    ia = rng.choice(len(reach_pos), size=_N_PAIRS, p=weights)
    aux = rng.uniform(lo, hi, size=(_N_PAIRS, 3)).astype(np.float32)
    x_i = reach_pos[ia]

    eta2 = score_pairs(adapter_key, x_i, aux)
    eta = score_targets(adapter_key, x_i)[0]

    # Mean separation of the scalar sampler's own pairs, reported because the scalar is CONFOUNDED by
    # it: x_j is uniform in each platform's own voxel box, eta_2 approaches eta as separation goes to
    # zero, so a physically small robot is asked easier pairs and scores higher for a reason that has
    # nothing to do with its cameras. ToddlerBot sits at ~0.49 m against 0.74-0.98 m for every other
    # column and its scalar is flattered accordingly. Printing this next to eta_2 is what stops the
    # ranking being read off the scalar alone; the separation curve is the comparison that is sound.
    sep_mean = float(np.linalg.norm(x_i - aux, axis=1).mean())

    # Degeneracy probe: same x_i, x_j a fixed small distance away in an isotropic direction.
    step = rng.normal(size=(_N_PAIRS, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    near = (x_i + _NEAR_SEP_M * step).astype(np.float32)
    eta2_near = score_pairs(adapter_key, x_i, near)

    curve, curve_n = _separation_curve(adapter_key, reach_pos, weights, lo, hi, rng)

    return dict(
        column=cli_key, name=name, groups=len(_aim_groups(_ADAPTERS[adapter_key])),
        eyes=len(_ADAPTERS[adapter_key].eyes),
        eta=float(eta.mean()), eta2=float(eta2.mean()), eta2_near=float(eta2_near.mean()),
        bound_ok=bool(eta2.mean() <= eta.mean()), witness_ok=bool((eta2 & ~eta).sum() == 0),
        sep_mean=sep_mean, curve=curve, curve_n=curve_n,
    )


def _plot_curves(rows: list[dict], out) -> None:
    """Pairwise coverage against target separation, all eight Fig. 2 columns on one axis.

    The scalar eta_2 says every single-direction platform collapses; this says WHY, by showing the
    collapse is a monotone decay in separation rather than an occlusion artifact. Separation is in
    metres, not normalized by robot size: the platforms differ in scale, but ToddlerBot is the
    smallest in the set and retains the most coverage of any external column, which is the opposite
    of what a size artifact would produce, so it serves as the disclosed control.

    CIs are drawn for our two columns only. All eight are measured identically, but eight intervals
    at column width is unreadable and the externals are not the quantity a reader checks for
    resolution -- the Ours(fixed) against Ours(actuated) gap is.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 6})
    centers = 0.5 * (_SEP_EDGES[:-1] + _SEP_EDGES[1:])
    fig, ax = plt.subplots(1, 1, figsize=(3.45, 3.2))
    ext = 0
    for r in rows:
        p = np.array([np.nan if v is None else v for v in r["curve"]])
        if r["column"] in _OURS_STYLE:
            n = np.array(r["curve_n"])
            err = np.where(n >= _MIN_BIN, 1.96 * np.sqrt(np.abs(p * (1 - p)) / np.maximum(n, 1)), np.nan)
            ax.errorbar(centers, p, yerr=err, elinewidth=0.7, capsize=1.5, label=r["name"],
                        **_OURS_STYLE[r["column"]])
        else:
            ax.plot(centers, p, color=_EXTERNAL_COLORS[ext], lw=1.0, marker="o", ms=2.2,
                    alpha=0.85, label=r["name"])
            ext += 1
    ax.set_xlabel(r"separation $\|x_i - x_j\|$ (m)")
    ax.set_ylabel(r"pairwise VR fraction $\eta_2$")
    ax.set_ylim(0, 1)
    # Legend goes ABOVE the axes, not inside. There is no empty corner: the flat ours-actuated curve
    # occupies the top, and ours-fixed and G1 hold the bottom left, so any in-axes box covers data.
    # Handles are reordered to put our two columns first -- matplotlib sorts errorbar containers
    # after plain Line2Ds, which otherwise buries the two curves the figure exists to contrast.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: not labels[i].startswith("Ours"))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False,
              handlelength=1.2, borderpad=0.2, labelspacing=0.25, columnspacing=1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(f"{out}.{suffix}", dpi=200)
    print(f"wrote {out}.png / .pdf")
    _mirror_to_paper(out, paper_stem="fig_eta2_separation")


_MANIFEST = _REPO_ROOT / "mj_envs/asset_zoo/reachability_study/result/eta2_manifest.json"


def _cached_rows(refresh: bool) -> dict:
    """Previously computed per-column rows that are still valid for the current settings.

    A full pass is eight platform payload loads plus three GPU kernel passes each, which is minutes,
    and almost every rerun in practice is a figure tweak rather than a numbers change. The manifest
    already stores every value the plot needs, so it doubles as the cache and no second artifact is
    introduced. Reuse is gated on the settings that change the numbers -- seed, pair count, and the
    bin edges -- so a cached row cannot silently survive a convention change.

    NOT gated on the kernel or the reach payloads: a change to `gpu_visibility` or a re-solved
    payload leaves stale rows that look valid. Pass `--refresh` after touching either.
    """
    if refresh or not _MANIFEST.is_file():
        return {}
    m = json.loads(_MANIFEST.read_text())
    stale = (m.get("n_pairs") != _N_PAIRS or m.get("seed") != _SEED
             or m.get("near_sep_m") != _NEAR_SEP_M or m.get("sep_edges") != _SEP_EDGES.tolist())
    if stale:
        print("cache settings differ from current constants -- recomputing every column.")
        return {}
    return {k: v for k, v in m.get("columns", {}).items()
            if v.get("curve") is not None and "sep_mean" in v}


def main(refresh: bool = False, plot_only: bool = False) -> None:
    """Score every Fig. 2 column, write the manifest, and render the separation figure.

    Args:
        refresh: Ignore cached rows and recompute every column. Required after any change to
            `gpu_visibility` or to a reach payload, neither of which the cache can detect.
        plot_only: Replot from the manifest without touching a GPU. Fails if it is incomplete.
    """
    cache = _cached_rows(refresh)
    if plot_only:
        missing = [c for c, _, _ in _COLUMNS if c not in cache]
        if missing:
            raise SystemExit(f"--plot-only needs a complete manifest; missing {missing}. "
                             f"Run without --plot-only first.")
        _plot_curves([cache[c] for c, _, _ in _COLUMNS], _MANIFEST.parent / "eta2_separation")
        return

    rows = []
    for column in _COLUMNS:
        hit = cache.get(column[0])
        if hit is not None:
            print(f"cached {column[0]}")
            rows.append(hit)
        else:
            rows.append(_run(*column))
    print(f"{'column':<18}{'K':>3}{'eyes':>5}{'eta':>8}{'eta_2':>8}{'eta_2|sep=3cm':>15}"
          f"{'mean sep':>10}{'eta2<=eta':>10}{'witness':>9}")
    for r in rows:
        print(f"{r['column']:<18}{r['groups']:>3}{r['eyes']:>5}{r['eta']:>8.4f}{r['eta2']:>8.4f}"
              f"{r['eta2_near']:>15.4f}{r['sep_mean']:>10.3f}"
              f"{str(r['bound_ok']):>10}{str(r['witness_ok']):>9}")
    spread = max(r["sep_mean"] for r in rows) / min(r["sep_mean"] for r in rows)
    print(f"mean-separation spread across columns: {spread:.2f}x. Scalar eta_2 is NOT comparable "
          f"across columns whose sampled separations differ; read the separation curve instead.")
    print("\nFig. 2 titles (eta / eta_2, percent):")
    for r in rows:
        print(f"  {r['column']:<18}{100 * r['eta']:>6.1f}{100 * r['eta2']:>8.1f}")

    # Write the manifest the figure reads, so no eta_2 is transcribed by hand into a title or a
    # caption. `plot_workspace_curobo` loads this and refuses to label a column it cannot find here.
    # The eta recorded alongside is on THIS script's dexterity-weighted x_i sample and is NOT the
    # published eta (uniform over reachable voxels); it is stored only so a later reader can tell the
    # two conventions apart rather than assume the figure mixed them by accident.
    manifest = dict(
        n_pairs=_N_PAIRS, seed=_SEED, near_sep_m=_NEAR_SEP_M, sep_edges=_SEP_EDGES.tolist(),
        source="mj_envs/asset_zoo/reachability_study/test/run_eta2_platforms.py",
        columns={r["column"]: r for r in rows},
    )
    out = _MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out}")

    _plot_curves(rows, out.parent / "eta2_separation")


if __name__ == "__main__":
    tyro.cli(main)
