"""K = 1/2/3 head-camera count ablation: reach falls with K, coverage rises and saturates.

Produces the paper's ablation figure and table from the three per-rig reachability payloads
(`workspace_curobo_ik_humanoid_v21_{single,dual,triple}_R_so3_dex_0p02.pt`), which must all be
solved on the SAME grid (pad 0.10, 73x91x73 = 484,939 voxels) so a voxel index means the same target
in each. The grid is anchored to integer multiples of `--grid-spacing` on all three axes, which also
makes it `--pad`-independent -- pad decides how many cells, never where they sit.

Why the two curves oppose:

* `W_R(K)` is NON-INCREASING in K. Camera hardware is a third of the rig's cuRobo collision spheres
  (26 of 104 on the dual rig) and is active in self-collision, so every module added removes arm
  configurations. This is measured, not assumed -- the rigs are NOT nested (K=1 sits at y=0, K=2 at
  y=+-0.065, K=3 at y=+-0.130), so no rig's obstacle set contains another's. The K=3 outer pair is
  not free to sit at +-0.065: a module sweeps a cylinder of radius 0.0577 m about its own yaw axis,
  so neighbours must clear 0.1154 m, and both rigs sit at the same 0.130 m adjacent pitch (see
  `asset/duke_v2/head_cam/head_camera_creation.py` RIGS). There is no cheaper K=3, so the measured
  drop is the true cost of a third camera rather than an artifact of a layout choice.
* `eta(K)` and `eta_2(K)` are NON-DECREASING in K, by set inclusion over the per-eye masks.

`W_VR = eta * W_R` is what penalizes added hardware: coverage is a fraction OF a reachable set the
same hardware shrinks. On the whole-body set that product eliminates K=1 outright (-2.35%) but leaves
K=2 and K=3 within 0.41%, a few times the noise floor below, so the metric declares them equivalent
and cost breaks the tie. Do not restate this as an interior optimum: the turnover at K=2 exists only
when scoring one arm, and that convention was retired on 2026-07-29 for understating the robot by a
third relative to `fig:workspace`. An earlier version of this docstring credited a
"mutual-occlusion tax"; that mechanism was measured and is ZERO (0/600 camera-blocking rays at K=2,
0/1200 at K=3), which is precisely why the penalty has to arrive through `W_R`.

MEASUREMENT NOISE. cuRobo's `success` thresholds `pos_err`/`rot_err` reduced over 32 GPU seeds, so
re-solving an identical window flips ~0.1% of rows. Differences between rigs below that floor are
not reportable. The table prints the floor next to the deltas rather than quoting bare percentages.

AUXILIARY-REGION VARIANT (paper Definition 5). The pairwise metric requires only `x_i` to be
REACHABLE; the partner `x_j` is any point in the workspace box. The claim is "manipulate here while
watching there". Two reasons, in order of weight. (1) The metric selects camera COUNT, so requiring
`x_j` to be reachable would fold arm kinematics into a sensing number that `W_R` already carries, and
a low score would then have two causes. (2) It matches the order the platform acts in: both regions
must be OBSERVED before a two-handed assignment exists at all (`curobo/scene.py`: both cubes
reachable -> bimanual, one -> single, none -> hold), so the observation test is the one that gates the
decision. Immaterial either way: restricting `x_j` to the reachable set moves `eta_2` by at
most 0.014 at every K, with the sign flipping under dexterity weighting. Do NOT justify this by
saying the platform has no bimanual IK -- it does (`set_bimanual_goalset`); it is only the
reachability PAYLOAD that solves one arm at a time, with the idle tool pose-pinned at its rest FK
pose rather than its joints locked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Same bootstrap every sibling script carries: `python <this file>` puts the SCRIPT's dir on
# sys.path, not the repo root, so `mj_envs` is unimportable without it (the readme's documented
# invocation is exactly that, and it only worked with PYTHONPATH already set).
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mj_envs.asset_zoo.reachability_study.plot_workspace_curobo import (  # noqa: E402
    _ROBOTS,
    _mirror_to_paper,
    _pairwise_vrw,
    _plot_reach_visible_grid,
    _reach_visible_case,
    _shoulder_section_payload_z,
)

# cli_key per camera count. Keys resolve payload + model_key through `_ROBOTS`, so the rig table
# stays the single source of truth.
_K_TO_KEY = {1: "v2_single", 2: "v2", 3: "v2_triple"}
# Separation bins for the curve, CENTERED on 0.2, 0.4, ..., 1.2 m so every plotted point lands on a
# labelled tick and the only convention left to defend is "binned estimate at bin center". Uniform
# width, no ragged end bin: the top edge runs to 1.3 rather than stopping the sampler at 1.2, which
# the workspace box admits (its diagonal is ~2.7 m). Sub-0.1 m separations drop out, which is the
# degenerate near-coincident regime. Zero-width bins at exactly 0.2/0.4/... were considered and
# rejected: they cost no statistical power, since the sampler generates rather than subsets, but
# they estimate the curve at one separation instead of over the neighbourhood a design figure cares
# about.
_SEP_EDGES = np.arange(0.1, 1.301, 0.2)


def _rig_cost(model_key: str) -> tuple[float, int]:
    """Head-camera mass (kg) and gimbal DOF for one rig.

    The coverage metric is non-decreasing in K by construction, so it can never argue against a
    third camera -- the remark in the paper concedes exactly this. Cost is what closes the argument,
    so it is measured from the same compiled models the solver used rather than quoted from a BOM.
    """
    import mujoco
    from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_spec
    from mj_envs.asset_zoo.reachability_study.plot_workspace_curobo import _HUMANOID_RIGS

    head_camera, _eyes = _HUMANOID_RIGS[model_key]
    m = get_spec(head_camera=head_camera, end_effector="welded", hand="parallel_gripper").compile()
    mass = sum(float(m.body_mass[i]) for i in range(m.nbody) if m.body(i).name.startswith("cam_"))
    dof = sum(int(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
              for j in range(m.njnt) if m.joint(j).name.startswith("cam_"))
    return mass, dof


def _rig_stats(k: int, n_pairs: int, per_bin: int, seed: int) -> dict:
    """Reach volume, single-target coverage, and pairwise coverage vs separation for one rig.

    Scored on the WHOLE-BODY reachable set (right arm unioned with its sagittal reflection), which is
    what `_reach_visible_case` returns and what Fig. `fig:workspace` already reports for every robot.
    An earlier version scored the right arm alone; that understated the machine by a third against
    the paper's own comparison figure, and it changed the answer -- the union recovers a lost voxel
    whenever the opposite arm still reaches it, which halves the third module's reach penalty and
    moves the `W_VR` maximum from K=2 to K=3. Reach and dexterity are mirrored, sound because the
    arms and body are symmetric; visibility is NOT mirrored but re-evaluated against the real cameras
    at the reflected positions, so the coverage columns are measured rather than assumed.
    """
    # TWO generators, both seeded from `seed` alone, so the headline scalar can never be perturbed
    # by a change on the curve side. One generator threaded through the three rigs made `eta_pair`
    # for K=2/K=3 depend on how many draws K=1 had consumed, which silently moved published numbers
    # when the curve sampler changed and put them out of step with `test/pairwise_gate.py`, which
    # re-seeds per rig. Reusing the SAME scalar stream for every rig is also a paired comparison:
    # differences between K are then not sampling noise.
    rng = np.random.default_rng(seed)
    crng = np.random.default_rng([seed, k])
    cfg = _ROBOTS[_K_TO_KEY[k]]
    case = _reach_visible_case(_K_TO_KEY[k])

    reach = case["reached"]
    pos = case["points"]
    reach_pos = pos[reach]
    n_reach = int(reach.sum())

    vis_steer, vis_fixed = case["vis_steer"][reach], case["vis_fixed"][reach]

    # x_i is drawn from the REACHABLE set weighted by dexterity -- the paper's manipulation point is
    # not merely reachable, it is well-conditioned.
    dex = np.nan_to_num(case["dexterity"][reach], nan=0.0)
    w = dex / dex.sum() if dex.sum() > 0 else None
    lo, hi = pos.min(0), pos.max(0)

    def _cov(aux: np.ndarray, ia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Steered and locked-gimbal pairwise masks, from one solve.

        The locked family is free here: a fixed rig has one realizable head state, so the pairwise
        definition collapses to a conjunction of single-target fixed masks that `_camera_visibility`
        already produces for both members of every pair.
        """
        pts = np.concatenate([reach_pos, aux], axis=0).astype(np.float32)
        return _pairwise_vrw(pts, cfg.model_key, np.stack([ia, np.arange(len(ia)) + n_reach], 1),
                             return_fixed=True)

    # Headline scalar: x_j uniform in the workspace box. Unconditional, so it is comparable across
    # K without reference to any separation distribution.
    ia = rng.choice(n_reach, size=n_pairs, p=w)
    aux = rng.uniform(lo, hi, size=(n_pairs, 3)).astype(np.float32)
    _pair_steer, _pair_fixed = _cov(aux, ia)
    eta_pair = float(_pair_steer.mean())
    eta_pair_fixed = float(_pair_fixed.mean())

    # Curve: the SAME estimator, but with x_j proposed at a chosen separation instead of from the
    # box. A uniform-box x_j lands within 0.2 m of x_i almost never, which left the first three bins
    # empty and truncated the curve exactly where the K=1 penalty is largest. Each bin reports a
    # conditional mean GIVEN separation -- which is what the x axis already conditions on -- so
    # reweighting the proposal in that variable leaves every plotted value unbiased; it only moves
    # samples to where they are needed.
    #
    # STRATIFIED, one draw loop per bin, rather than one uniform draw over the whole range. Uniform-r
    # proposals arrive uniform per bin but do not SURVIVE uniformly: the box rejection below removes
    # a growing share as r grows, because more of the sphere of radius r around x_i falls outside the
    # workspace. At 0.1 m bins that left 1,696 pairs in the first bin and 319 in the last, and the
    # far bins then jittered by more than the effect they were drawn to show. Filling each bin to the
    # same count fixes the CI width across the axis, and costs no kernel time: the rejected
    # proposals never reach `_cov`, so the kernel sees `per_bin * nbin` pairs, all of them used.
    nbin = len(_SEP_EDGES) - 1
    cdf = np.cumsum(w)   # inverse-CDF sampling of x_i, built ONCE
    lo_e, w_e = _SEP_EDGES[:-1], np.diff(_SEP_EDGES)

    def _propose(need: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw `need[b]` proposals for every bin b in ONE vectorized pass, return the accepted."""
        b = np.repeat(np.arange(nbin), need)
        i = np.searchsorted(cdf, crng.random(b.size))
        r = lo_e[b] + w_e[b] * crng.random(b.size)
        u = crng.normal(size=(b.size, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        x = (reach_pos[i] + r[:, None] * u).astype(np.float32)
        keep = ((x >= lo) & (x <= hi)).all(1)   # x_j must stay inside the workspace box
        return b[keep], i[keep], x[keep]

    # Acceptance falls with r (more of the sphere of radius r about x_i leaves the box), so the
    # overdraw is per bin and measured rather than guessed: pass 1 draws 2x, pass 2 asks for the
    # shortfall divided by that bin's own observed rate. Two passes cover every bin in practice; the
    # loop is a guard, not the normal path.
    b_acc = np.zeros(0, int); i_acc = np.zeros(0, int); x_acc = np.zeros((0, 3), np.float32)
    need = np.full(nbin, 2 * per_bin)
    while need.any():
        b_n, i_n, x_n = _propose(need)
        rate = np.maximum(np.bincount(b_n, minlength=nbin), 1) / need.clip(1)
        b_acc = np.concatenate([b_acc, b_n]); i_acc = np.concatenate([i_acc, i_n])
        x_acc = np.concatenate([x_acc, x_n])
        have = np.bincount(b_acc, minlength=nbin)
        need = np.where(have < per_bin, np.ceil(1.3 * (per_bin - have) / rate), 0).astype(int)

    # Keep the first `per_bin` accepted per bin. Truncating an iid accepted stream is unbiased for
    # the bin's conditional mean: the survivors are still uniform in r over the bin and in direction.
    order = np.argsort(b_acc, kind="stable")
    b_sorted = b_acc[order]
    rank = np.arange(b_sorted.size) - np.searchsorted(b_sorted, b_sorted, side="left")
    sel = order[rank < per_bin]
    ia_c, aux_c = i_acc[sel], x_acc[sel]
    cov_c, cov_fixed_c = _cov(aux_c, ia_c)

    bin_id = b_acc[sel]

    def _binned(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-bin mean coverage and sample count.

        The count is kept so the plot can draw a binomial CI. The stratified proposal above fills
        every bin to exactly `per_bin`, so the gate never fires and the interval width depends only
        on p, not on where the point sits on the axis. Worth drawing anyway: it is what makes "K=2
        and K=3 nearly coincide" a measured statement rather than an eyeball one, since their gap
        (0.017) is the same size.
        """
        curve = np.full(len(_SEP_EDGES) - 1, np.nan)
        curve_n = np.zeros(len(curve), dtype=int)
        for b in range(len(curve)):
            m = bin_id == b
            if m.sum() >= 50:  # below this the bin fraction is noisier than the effect measured
                curve[b] = cov[m].mean()
                curve_n[b] = int(m.sum())
        return curve, curve_n

    curve, curve_n = _binned(cov_c)
    curve_fixed, curve_n_fixed = _binned(cov_fixed_c)

    cam_mass, cam_dof = _rig_cost(cfg.model_key)
    volume = n_reach * case["voxel_vol"]
    eta_steer, eta_fixed = float(vis_steer.mean()), float(vis_fixed.mean())
    return {
        "k": k,
        "cam_mass": cam_mass,
        "cam_dof": cam_dof,
        "n_reach": n_reach,
        "volume_m3": volume,
        "eta_steer": eta_steer,
        "eta_fixed": eta_fixed,
        "eta_pair": eta_pair,
        "eta_pair_fixed": eta_pair_fixed,
        "curve": curve,
        "curve_n": curve_n,
        "curve_fixed": curve_fixed,
        "curve_n_fixed": curve_n_fixed,
        # The VRW volume itself, which is what the paper's definition names. `eta` is a fraction OF
        # the reachable set, so the product is exact rather than an approximation. It is the entry
        # that separates the rigs without an external weighting of coverage against mass: it rejects
        # K=1 by a margin no cost argument is needed to read, and leaves K=2/K=3 inside 0.41%.
        "wvr_m3": volume * eta_steer,
        "wvr_fixed_m3": volume * eta_fixed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pairs", type=int, default=20000,
                    help="pairs for the headline scalar eta_2; 20k is the published convention "
                         "and what test/pairwise_gate.py re-derives")
    ap.add_argument("--curve-pairs-per-bin", type=int, default=5000,
                    help="pairs per separation bin for the curve. Set explicitly rather than "
                         "derived from --n-pairs: the two estimators are independent, and "
                         "dividing the scalar budget by the bin count made the per-bin number a "
                         "residual (20000//6 = 3333) that moved whenever either input changed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "result" / "camera_count_ablation")
    args = ap.parse_args()

    stats = [_rig_stats(k, args.n_pairs, args.curve_pairs_per_bin, args.seed)
             for k in sorted(_K_TO_KEY)]

    base = stats[1]  # K=2, the shipped rig
    print(f"{'K':>2} {'reach vox':>10} {'W_R m^3':>9} {'dW_R':>7} {'eta_fix':>8} {'eta':>7} "
          f"{'W_VR m^3':>9} {'dW_VR':>7} {'W_VRfix':>8} {'eta_pair':>9} {'cam kg':>8} {'cam DOF':>8}")
    for s in stats:
        dr = 100.0 * (s["n_reach"] - base["n_reach"]) / base["n_reach"]
        dv = 100.0 * (s["wvr_m3"] - base["wvr_m3"]) / base["wvr_m3"]
        print(f"{s['k']:>2} {s['n_reach']:>10,} {s['volume_m3']:>9.4f} {dr:>6.2f}% "
              f"{s['eta_fixed']:>8.4f} {s['eta_steer']:>7.4f} {s['wvr_m3']:>9.4f} {dv:>6.2f}% "
              f"{s['wvr_fixed_m3']:>8.4f} {s['eta_pair']:>9.4f} "
              f"{s['cam_mass']:>8.3f} {s['cam_dof']:>8}")
    for s in stats:
        print(f"K={s['k']} curve n per bin: {s['curve_n'].tolist()}")
        print(f"K={s['k']} eta_2 steered: {np.round(s['curve'], 4).tolist()}")
        print(f"K={s['k']} eta_2 fixed:   {np.round(s['curve_fixed'], 4).tolist()}")
    print("\nsolver noise floor ~0.1% of rows; deltas below that are not reportable.")
    print("whole-body scoring: W_VR rejects K=1 outright, K=2/K=3 differ by <0.5% and cost decides.")

    import matplotlib
    matplotlib.use("Agg")
    # TrueType, not matplotlib's default Type 3: this figure is mirrored into the manuscript and
    # IEEE PDF eXpress rejects a submission carrying any Type 3 font.
    matplotlib.rcParams["pdf.fonttype"] = 42
    from matplotlib.lines import Line2D

    centers = 0.5 * (_SEP_EDGES[:-1] + _SEP_EDGES[1:])

    def _wilson(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
        """95% Wilson score interval as (lower, upper) OFFSETS from p, shape (2, len(p)).

        Wilson rather than the normal approximation because the smallest bin is not large-count: the
        K=1 fixed family reaches ~48 successes in 5,000 pairs at 1.2 m separation, which fails the
        n*p > 50 rule the normal form relies on. Wilson inverts the test instead of assuming the
        observed fraction is the true one, so it stays valid at small n*p, never leaves [0, 1], and
        is correctly ASYMMETRIC near zero. Closed form, so no bootstrap. At the other 35 points the
        two agree to three decimals; this exists so the one small-count bar is the right shape.
        """
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return np.stack([p - (c - h), (c + h) - p])

    def draw_curve(ax) -> None:
        """Draw the separation curves into the section-cut grid's bottom strip.

        Two families on one axis: colour carries K, line style carries actuation, and the marker is
        deliberately the SAME for both so nothing but the line distinguishes them. Both come from the
        one kernel and the one stratified pair sample, so a vertical gap between a solid line and its
        own dotted line is the value of steering at that separation and nothing else. Both families
        carry intervals; the fixed bins are no longer a redundant copy of the steered counts now that
        every bin holds exactly `per_bin` pairs.

        The locked K>=2 curves FLATTEN near 0.14 rather than decaying to zero, and that is mechanism,
        not noise: past the separation where the two visibility events decorrelate, a rig with two
        cameras can assign one target to each and the joint probability factorizes into a product
        that does not depend on separation. K=1 has no second camera to assign, so both targets must
        share one frustum and it keeps falling. Do not "fix" the plateau.

        Type sizes are the grid's, not this figure's: the composite is drawn at ~11 in wide and
        placed at one column, so a point size chosen for a 3.45 in standalone would render at a
        third of its nominal size here.
        """
        line = dict(marker="o", ms=3.5, lw=2.2)          # shared by the curves and the legend keys
        bars = dict(elinewidth=1.2, capsize=2.5)         # errorbar-only, rejected by Line2D
        handles = []
        for s, color in zip(stats, ("tab:blue", "tab:orange", "tab:green")):
            ax.errorbar(centers, s["curve"], yerr=_wilson(s["curve"], s["curve_n"]),
                        color=color, **line, **bars)
            ax.errorbar(centers, s["curve_fixed"], yerr=_wilson(s["curve_fixed"], s["curve_n"]),
                        color=color, ls=":", **line, **bars)
            handles.append(Line2D([], [], color=color, lw=2.6, label=f"$K$={s['k']}"))
        # TWO legends, not one two-row block: a single legend forces both rows onto shared column
        # widths, so the three K entries would be spaced by the word "actuated" instead of by their
        # own labels. Mid-left is the only clear region: the K=1 solid line sweeps the upper right
        # and the fixed family occupies the bottom.
        kw = dict(bbox_transform=ax.transAxes, loc="upper left", handlelength=1.6, borderpad=0.3,
                  columnspacing=1.2, fontsize=14, frameon=False)
        ax.add_artist(ax.legend(
            handles=[Line2D([], [], color="0.35", ls=":", label="fixed", **line),
                     Line2D([], [], color="0.35", ls="-", label="actuated", **line)],
            bbox_to_anchor=(0.02, 0.74), ncol=2, **kw))
        ax.legend(handles=handles, bbox_to_anchor=(0.02, 0.58), ncol=3, **kw)
        ax.set_xlabel(r"separation $\|x_i - x_j\|$ (m)")
        ax.set_ylabel(r"pairwise VR fraction $\eta_2$")
        ax.set_ylim(0, 1)
        # Right limit trails the last POINT, not the last bin edge: the top bin runs to 1.3 only
        # so its centre lands on 1.2 like every other, and showing that empty half-bin marooned
        # the final marker.
        ax.set_xlim(_SEP_EDGES[0], centers[-1] + 0.05)
        ax.tick_params(labelsize=15)
        ax.grid(alpha=0.3)

    # One composite figure rather than two placed side by side in LaTeX. The section cuts and the
    # separation curve are read together, and stacked as two floats at single-column width they cost
    # 6.1 in of a 9.25 in text column. Merging them into one grid whose bottom strip carries the
    # curve BESIDE the colorbars, where the old full-width bar band held nothing but air, pays for
    # the curve almost entirely out of that reclaimed band. `--camera-count-cuts` still writes the
    # standalone cut grid; only the paper's copy is merged.
    # ONE letter for the whole cut grid, not one per rig: the three rigs are already named by their
    # column headings, and (a)-(c)-per-column plus (d) made the caption key four panels where the
    # figure only makes two claims. Only the first case carries `panel`, which puts the letter at
    # column 0's top-left, the grid's own corner.
    # `eta2` is THIS script's scalar, not the platform manifest's: the manifest has no K=1/K=3 entry
    # and its kernel scores the shipped rig 0.9572 against 0.9556 here, so the two must never appear
    # in one figure. Carrying it in the headings is what lets the results text drop the ablation
    # table, whose every other column the headings already print (eta_fix, eta, W_VR^fix, W_VR, W_R).
    cases = [
        {"name": f"$K = {k}$", "panel": "a" if i == 1 else "", "visible": c["vis_steer"],
         "visible_static": c["vis_fixed"], "eta2": s["eta_pair"], "eta2_fixed": s["eta_pair_fixed"],
         "top_cutoff_z": _shoulder_section_payload_z("humanoid_v21"),
         **{key: c[key] for key in ("robot", "points", "reached", "dexterity", "voxel_vol")}}
        for i, (s, (k, c)) in enumerate(
            zip(stats, ((k, _reach_visible_case(_K_TO_KEY[k])) for k in sorted(_K_TO_KEY))), start=1)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _plot_reach_visible_grid(cases, args.out, vis_cmap="YlGn", blind_cmap="Blues",
                             mid_cmap="YlOrBr", extra_row=draw_curve, extra_panel="b")
    _mirror_to_paper(args.out, paper_stem="fig_camera_count")


if __name__ == "__main__":
    main()
