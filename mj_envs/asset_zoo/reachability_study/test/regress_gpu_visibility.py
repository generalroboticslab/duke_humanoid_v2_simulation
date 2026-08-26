"""Regression gate for `gpu_visibility` refactors, on the robots that have a sidecar to check.

Each case rescores from the raw workspace and compares against that robot's committed-by-convention
sidecar. All five dynamic-visibility robots are covered, which between them exercise the
coupled-neck branch (`_apply_coupling`, ToddlerBot's -1/0.909 gear), the 3-DOF-neck two-eye
`cam_intrinsic` branch (Apollo), and the plain 2-DOF necks. An absent input SKIPS rather than
crashing: `mj_envs/asset_zoo/cache/` is gitignored, so a fresh checkout has none of these, and a
sidecar can outlive the multi-GB payload it was scored from -- hence both paths are checked, not
just the sidecar.

`expected_diff` stays per-case rather than a hardcoded 0 so a future change that is measured and
deliberately accepted can be pinned to its exact row count without immediately rescoring; pinning
to the count still fails on any FURTHER drift, which a loose `diff <= n` would not. All five are 0
today. Only Apollo's ever was not: it carried 43 rows from the aim-solver fix (`_capped_residual` +
`_lattice_seed`), borderline-FOV voxels behind the head where no legal aim exists and the solver
settled on a different arbitrary pose among the joint-limit-saturated ones. Its sidecar has since
been rescored. T1 and TALOS were re-scored under the same fix and came back bit-identical, which is
the empirical form of the claim that neither singularity can fire on a 2-DOF neck.

The sidecars are local; what the figures actually read is the TRACKED
`aggregated_cache/<stem>_aggregated.pt` derived from them. Rescoring a sidecar without rerunning
`plot_workspace_curobo.py --cache --robot <robot>` leaves that derived file stale, and this gate
cannot see it -- it never loads the aggregate.

`visible` is compared AFTER the isolated-blind repair pass. Comparing raw `score_targets` output
against a saved sidecar does not work: the sidecar is written post-repair, so the two differ by
exactly `isolated_blind_directly_visible` rows and the mismatch looks like a regression when
nothing is wrong.

`head_q` is REPORTED, never gated, and `max|dq|` is expected to be large on some robots (T1 prints
1.57 rad = pi/2 while matching `visible` exactly). Aim is underdetermined: a neck can frame the
same target from more than one pose, so two runs may pick different ones without disagreeing about
what is visible. Only `visible` is the observable this study publishes, so only `visible` is
asserted. `nan_match` is the useful companion -- it catches a row flipping between "aim found" and
"no aim exists", which `max|dq|` cannot see.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mj_envs.asset_zoo.reachability_study.gpu_visibility import (
    _bimanual_isolated_blind,
    _lattice_rescore,
    score_targets,
)

_CACHE = _REPO_ROOT / "mj_envs/asset_zoo/cache"
# (robot, workspace, reference sidecar, expected `visible` row difference -- see module docstring).
# Every dynamic-visibility robot with a sidecar is listed; GR-3 is here so that a checkout which
# does have its payload gets the coverage, and SKIPs cleanly on one that does not.
_CASES = (
    ("apptronik_apollo", _CACHE / "workspace_curobo_ik_apptronik_apollo_R_so3_dex_0p02_symrest.pt",
     _CACHE / "workspace_curobo_ik_apptronik_apollo_R_so3_dex_0p02_symrest_visibility_gpu.pt", 0),
    ("toddlerbot", _CACHE / "workspace_curobo_ik_toddlerbot_R_so3_dex_0p02.pt",
     _CACHE / "workspace_curobo_ik_toddlerbot_R_so3_dex_0p02_visibility_gpu.pt", 0),
    ("booster_t1", _CACHE / "workspace_curobo_ik_booster_t1_R_so3_dex_0p02.pt",
     _CACHE / "t1_visibility_R_dex_0p02_gpu.pt", 0),
    ("pal_talos", _CACHE / "workspace_curobo_ik_pal_talos_R_so3_dex_0p02.pt",
     _CACHE / "pal_talos_visibility_R_dex_0p02_gpu.pt", 0),
    ("fourier_gr3", _CACHE / "workspace_curobo_ik_fourier_gr3_R_so3_dex_0p02.pt",
     _CACHE / "fourier_gr3_visibility_R_dex_0p02_gpu.pt", 0),
)


def main() -> None:
    failures = []
    ran = 0
    for robot, workspace, sidecar, expected_diff in _CASES:
        # Both inputs must be checked: a sidecar can outlive the multi-GB payload it was scored
        # from (GR-3 is in exactly that state locally), and rescoring needs the payload.
        missing = [p for p in (workspace, sidecar) if not p.exists()]
        if missing:
            print(f"{robot:<18} SKIP  absent: {', '.join(p.name for p in missing)}")
            continue
        payload = torch.load(workspace, map_location="cpu", weights_only=False)
        reference = torch.load(sidecar, map_location="cpu", weights_only=False)
        success = payload["success"].bool(); success_idx = torch.where(success)[0]
        voxel = payload["voxel_index"][success_idx].numpy()
        _u, first, inverse = np.unique(voxel, return_index=True, return_inverse=True)
        reps = success_idx.numpy()[first]
        targets = payload["target_pos"][reps].numpy()
        visible, head_q = score_targets(robot, targets)
        ran += 1
        repaired = 0
        for compact in _bimanual_isolated_blind(voxel[first], visible, payload["n_grid_per_axis"].numpy()):
            if _lattice_rescore(robot, targets[compact]):
                visible[compact] = True; repaired += 1
        want = reference["visible_left"][success_idx].numpy()[first]
        want_q = reference["head_q"][success_idx].numpy()[first]
        finite = np.isfinite(head_q) & np.isfinite(want_q)
        diff = int((visible != want).sum())
        ok = diff == expected_diff
        if not ok:
            failures.append(f"{robot}: diff={diff}, expected {expected_diff}")
        print(f"{robot:<18} {'PASS' if ok else 'FAIL'}  rows={len(visible):>7} "
              f"diff={diff}/{expected_diff} repaired={repaired}/{reference['isolated_blind_directly_visible']} "
              f"nvis={int(visible.sum())}/{int(want.sum())} "
              f"max|dq|={float(np.abs(head_q[finite] - want_q[finite]).max()):.3e} "
              f"nan_match={np.array_equal(np.isnan(head_q), np.isnan(want_q))}")
    if failures:
        raise SystemExit("regression: " + "; ".join(failures))
    # An all-SKIP run is the default state of a fresh checkout, where `mj_envs/asset_zoo/cache/` is
    # empty. Exiting 0 there would report "no regressions" for a gate that scored nothing, which is
    # the one outcome a reproducer must not be handed quietly.
    if ran == 0:
        raise SystemExit(
            "every case SKIPped -- nothing was scored, so this is NOT a pass. The raw workspace "
            "payloads are multi-GB and are not distributed; regenerate at least one with\n"
            "  python mj_envs/asset_zoo/reachability_study/generate_workspace_curobo.py --robot <robot>")
    print(f"{ran}/{len(_CASES)} cases scored, no regressions.")


if __name__ == "__main__":
    main()
