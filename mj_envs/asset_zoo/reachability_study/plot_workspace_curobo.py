"""Render an arm reachability workspace figure (any robot via ``--robot``; and a live 3D viewer).

Reads a precomputed workspace sidecar written by ``generate_workspace_curobo.py`` (per-voxel,
per-orientation IK reachability over the arm workspace). Never runs IK; never edits the sidecar.
Run from the repository root with the project python. Robot selected by ``--robot`` (humanoid_v21 or
g1); the overlaid robot mesh comes from the payload's ``robot`` field, so figure and payload always match.

Reachability
------------
``D = (# feasible sampled orientations) / (# sampled orientations)`` per voxel — orientation
reachability: fraction of a fixed near-uniform SO(3) orientation set the arm can reach at that
position. Colored RdYlGn on a fixed range ``R ∈ [0, 0.7]`` (rounded ceiling above bimanual peak).

Default: paper section figure
-----------------------------
Pick the robot with ``--robot`` (resolves its canonical payload); add ``--view`` for the live viewer:

.. code-block:: bash

    PY=python
    # figure (PNG+PDF -> result/, PDF mirrored to paper):
    $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --robot g1
    # live interactive viewer (needs a display; --section is the Z cut, 2.0 = show all):
    MUJOCO_GL=glfw $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --robot g1 --view --section 2.0

``--robot`` accepts ``v2``, ``v2_fixed``, or ``g1`` and resolves that robot's
canonical 0.02 m payload in ``cache/``; ``--input`` overrides it; a bare run (neither) uses ``v2``. It
derives the left arm by Y-mirroring the right (``--mirror-left``, on by default) for the bimanual field
``max(D_R, D_L)``, writes 350-dpi PNG + vector PDF to
``reachability_study/result/<input_stem>_reachability_index_sections.{png,pdf}``, then copies the PDF (the paper asset)
into ``paper_writing_humanoid_v2/figures/`` when that dir exists.

Figure = one row of three orthographic cutaways with the welded robot overlaid for scale, plus a
horizontal colorbar under (a):

* (a) Top view  (X-Y): retain ``z <= --section`` (default 0.30 m).
* (b) Front view (Y-Z): retain ``x <= 0``.
* (c) Side view  (X-Z): retain ``y >= 0``.

Each panel retains reachable voxels by its camera-depth cut, then projects overlapping voxels by
``max(D)`` (max = "is any voxel on this ray high reachability-index"; mean would dilute, frontmost would
occlude). The vertical axis is floor-referenced (foot sole at Z=0); the ``--section`` cut itself
stays in the base/root frame.

Live viewer
-----------
``--view`` opens an interactive MuJoCo viewer instead of writing files: every reachable voxel with
``z <= --section`` as a touching sphere, robot at home pose. Needs a display (``MUJOCO_GL=glfw``; over SSH
use ``ssh -X``). ``--section`` is the Z cut — raise it (``2.0``) to show the whole cloud, lower it to peel
from the top. MuJoCo lighting makes sphere colors differ from the unlit figure; it is for inspection, not the paper.

.. code-block:: bash

    MUJOCO_GL=glfw ... plot_workspace_curobo.py --robot g1 --view --section 2.0

Live viewer options
-------------------
* ``--view-mode {reachable,visible,instantaneous}`` — ``reachable`` (default) colors every reachable
  voxel by its ``D`` along RdYlGn (single-cloud, paper palette). ``visible`` splits the reachable
  cloud into reachable+visible (Greens family) and reachable+blind (Reds family) and gamma-curves
  the ``D`` shading so mid-D voxels get more visual contrast (matches ``--reach-visible-compare``;
  the colorbar at tick 0 in that figure exactly equals the color of a ``D``=0 voxel). Per
  robot: ``v2`` uses steered/actuated visibility, ``v2_fixed`` uses fixed-gimbal visibility
  (paper's "Ours (fixed)" column), ``g1`` uses its single fixed head cam (steered == fixed).
  ``instantaneous`` (ToddlerBot live `--view` only) re-evaluates per-voxel FOV at the current
  head-pose sliders every control-state change and renders Greens-vs-Reds by what the eyes
  actually see right now (no offline IK aiming).
* ``--sphere-r FLOAT`` — voxel sphere radius (m, geom size; rendered diameter is 2x). Default
  picks ``0.5 * grid_spacing`` (touching packing) or ``0.02`` for legacy payloads without
  ``grid_spacing``. Larger values overshoot the scene ``maxgeom`` cap and force auto-stride;
  smaller values make individual voxels visible.
* ``--stride INT`` — keep every Nth voxel on the per-axis grid (after spatial dedup), trading
  density for staying under the viewer's ``maxgeom`` cap (~1000). Default 1 = no thinning.
  ``--stride 2`` keeps ~1/8 of the cloud; a warning is printed if even the kept set exceeds
  the cap.

Other options
-------------
* ``--left-input PATH`` — use an independently solved L payload instead of the mirror (needed to
  *measure* L/R asymmetry; the mirror makes it exact by construction). Overrides ``--mirror-left``.
* ``--no-mirror-left`` — right arm only (no bimanual max).
* ``--symmetry`` — render the L/R symmetry comparison figure (needs ``--left-input`` or
  ``--mirror-left``) to ``<input_stem>_symmetry.png`` (or ``--out``).
* ``--out PATH`` — symmetry-figure output path.

A standalone 3D perspective of the reachable slab (robot + colored voxel boxes) is available via
``_render_reachable_slab_3d`` but is intentionally not part of the default figure.

Reach-and-visible figure (``--reach-visible-compare``)
------------------------------------------------------------------------------
``--reach-visible-compare``
writes paper ``fig:workspace``: Top and Side section-cut rows for Ours fixed, Ours actuated, and G1;
green means reachable-visible, red means reachable-blind. Side-by-side colorbars map both hues to the
same reachability-index range ``R ∈ [0, 0.7]``. Both show which reachable voxels real head camera(s)
can SEE.

.. code-block:: bash

    # Paper reachability map, humanoid V2 (bare run also selects V2):
    MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --robot humanoid_v21
    # Paper reachability map, G1:
    MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --robot g1
    # Paper reachability-visible comparison, V2 fixed / V2 actuated / G1:
    MUJOCO_GL=egl $PY mj_envs/asset_zoo/reachability_study/plot_workspace_curobo.py --reach-visible-compare

Commands require canonical ``cache/workspace_curobo_ik_*_R_so3_dex_0p02.pt`` payloads. Each writes
PNG/PDF under ``reachability_study/result/`` and mirrors PDF into ``paper_writing_humanoid_v2/figures/``.
The reach-visible comparison requires GPU; do not set ``CUDA_VISIBLE_DEVICES=""``.

``_camera_visibility`` dispatches by the payload's robot
and per eye/point tests range + FOV cone + line-of-sight (self-occlusion) via batched GPU raycasts
against the body meshes, with the body held at the arms-down pose (``_ARMS_DOWN_POSE``, forearms clear
of the cone):

* **humanoid_v21 (V2)** -- the **actuated dual back-to-back gimbal**: each voxel is **blind** (no eye),
  **fixed** (seen by the G1-matched 0.8308-rad-down back-to-back baseline), or **steerable-only** (seen only
  when an eye aims at it). The compare figure shows V2 fixed (fixed only) and V2 actuated (fixed OR
  steerable) as two of its three columns.
* **g1 (``_camera_visibility_g1``)** -- the **single FIXED builtin head camera** (official D435 mount:
  0.8308 rad / 47.6-degree down, no
  steering): a voxel is either head-cam visible or blind (``vis_steer == vis_fixed``, no steerable-only).
  The honest baseline the paper contrasts against V2.

The compare figure renders two orthographic SECTION CUTS -- Top=XY, Side=XZ -- inequality
half-spaces (top z<=cutoff, side y>=0, SAME convention as the reachability figure) rendered as the exposed cut
FACE: each cell shows the FRONTMOST voxel nearest the cut plane, so its category is exact and no depth-collapse
can bury a visible voxel behind a blind one (max-D projection would). ``--rv-flat`` drops the per-cell D
shading. Runs on
**GPU** (do not set ``CUDA_VISIBLE_DEVICES=""``); RGB near clip ``_RV_NEAR_M`` = 0.1 m, FAR 3.0 m.

Paper caption (ICRA-ready)
--------------------------
    **Reachability-index map of the humanoid dual arms.** At each point of a 20 mm regular grid we
    solve inverse kinematics (cuRobo; success = end-effector within 10 mm and 0.5 rad,
    self-collision-free, torso fixed) toward a fixed set of 64 near-uniform SO(3) orientations,
    and define reachability index R as the fraction reached; the head camera and parallel gripper are
    included in the arm geometry. Given the established left/right symmetry of the arms, the
    bimanual field D = max(D_L, D_R) is formed with the left arm mirrored from the right. Three
    orthographic cutaways are shown with the robot overlaid for scale: (a) top (X-Y, z <= 0.30 m),
    (b) front (Y-Z, x <= 0), (c) side (X-Z, y >= 0); along each view axis overlapping points are
    projected by their maximum D. The vertical axis is referenced to the foot (Z = 0); panels
    render each sample point as a filled cell, color encoding R in [0, 0.7].

Notes for the caption/methods: the 10 mm / 0.5 rad (~29 deg) figures are the cuRobo success gate
(``_POSITION_TOLERANCE`` / ``_ORIENTATION_TOLERANCE`` in ``ik_curobo_robot_cfg.py``); 0.5 rad is a
permissive *acceptance* gate, not a tracking bound (the cost still pulls fully to the nearest
orientation candidate, and 0.5 rad < the 45 deg half-gap between adjacent samples). The grid is
sampled at POINTS, not cells -- ``pcolormesh`` renders each point as a filled cell for a continuous
look, but D is a point measurement (say "grid point", not "voxel", in the paper).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Section figures are committed deliverables (any robot): write them under result/, then mirror the
# vector PDF (the paper asset) into the manuscript's figures/ dir. The input .pt lives in gitignored
# cache/, so the figure must NOT be written beside it.
_RESULT_DIR = Path(__file__).resolve().parent / "result"
# The manuscript moved to paper_writing_humanoid_v2/; the old path stopped existing and the mirror,
# guarded by is_dir(), silently became a no-op, so figures regenerated here quietly stopped reaching
# the paper. Use `_mirror_to_paper` rather than an inline copy so a skip is always reported.
_PAPER_FIG_DIR = _REPO_ROOT / "paper_writing_humanoid_v2" / "figures"


def _mirror_to_paper(path_base: Path, paper_stem: str | None = None) -> None:
    """Copy a rendered figure's PDF into the manuscript's figures/ dir.

    `paper_stem` overrides the filename where the paper includes the figure under a different name
    (`fig_camera_count.pdf` vs `camera_count_ablation.pdf`).

    The manuscript directory is not part of this export, so the mirror is a no-op here.
    """
    dst = _PAPER_FIG_DIR / f"{paper_stem or path_base.name}.pdf"
    if not _PAPER_FIG_DIR.is_dir():
        return
    shutil.copy2(path_base.with_suffix(".pdf"), dst)
    print(f"copied {dst}")


def _eta2_manifest() -> dict:
    """Pairwise coverage per figure column, written by `test/run_eta2_platforms.py`.

    Read from a manifest rather than recomputed here for two reasons. Scoring eta_2 for all eight
    columns needs CUDA and about ten minutes, which would make regenerating a plot a GPU job; and the
    alternative of a hard-coded table in this file is a hand-transcribed number, which is exactly what
    a reader cannot verify. A missing manifest is tolerated and drops the eta_2 field from the titles,
    so the figure still renders on a machine with no GPU, but it says so.
    """
    path = _RESULT_DIR / "eta2_manifest.json"
    if not path.is_file():
        print(f"WARNING: {path} absent -- titles will omit eta_2. Regenerate with "
              f"`python mj_envs/asset_zoo/reachability_study/test/run_eta2_platforms.py`.")
        return {}
    return json.loads(path.read_text())["columns"]

# Robot registry + payload aggregation live in `workspace_data` (this file owns rendering
# only). Underscore names kept: package-internal, not a public API.
from mj_envs.asset_zoo.reachability_study.workspace_data import (  # noqa: E402
    _CACHE_DIR,
    _DEFAULT_INPUT,
    _DEFAULT_ROBOT_KEY,
    _DYNAMIC_VISIBILITY_SIDECAR,
    _ROBOTS,
    _aggregate_by_voxel,
    _load_workspace_aggregate,
    _mirror_agg_l,
    _resolve_robot,
    _toddlerbot_visible_field,
    _write_aggregated_cache,
)
from mj_envs.asset_zoo.reachability_study.study_pose import set_home_pose  # noqa: E402
from mj_envs.asset_zoo.reachability_study.view_common import add_segment  # noqa: E402

import matplotlib

matplotlib.use("Agg")
# Embed PDF text as TrueType, not matplotlib's default Type 3. The figures here are mirrored into
# the manuscript, and IEEE PDF eXpress rejects a submission carrying any Type 3 font -- one such
# font in one included figure taints the whole main.pdf.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro


# Rounded ceiling above current bimanual peak D=0.672.  Shared by paper and live
# views so color remains quantitative while using the available dynamic range.
_DEXTERITY_VMAX = 0.7

# `_render_vrw_video`'s per-layer frontier boundary search tolerance -- see its use for why an
# exact-equality searchsorted boundary is unsafe here.
_FULL_EPS = 1e-7


def _greens_reds_rgba(values: np.ndarray, visible: np.ndarray, alpha: float = 1.0,
                      vis_cmap: str = "Greens", blind_cmap: str = "Greys") -> np.ndarray:
    """Vis vs Blind RGBA split, gamma-curved by D (matches --reach-visible-compare).

    `values` is the per-voxel D (reachability index) and `visible` is a `(N,)` bool mask
    aligned with it. Voxels with `visible == True` shade `vis_cmap` by D, the rest shade `blind_cmap` by D.
    """
    dex_norm = (torch.as_tensor(values, dtype=torch.float32) / _DEXTERITY_VMAX).clamp(0.0, 1.0)
    cmap_t = (_CMAP_T_MIN + (_CMAP_T_MAX - _CMAP_T_MIN) * dex_norm ** _VIEW_GAMMA).clamp(0.0, 1.0)
    cmap_idx = (cmap_t * 255.0).long().clamp(0, 255)
    rgba_t = torch.where(
        torch.as_tensor(visible, dtype=torch.bool).unsqueeze(1),
        _CMAP_LUT[vis_cmap][cmap_idx],
        _CMAP_LUT[blind_cmap][cmap_idx],
    )
    rgba = rgba_t.clone()
    rgba[:, 3] = alpha
    return rgba.numpy()

# Visible-mode D-to-cmap curve: `t = _CMAP_T_MIN + (_CMAP_T_MAX - _CMAP_T_MIN) * clip(D/vmax) ** _VIEW_GAMMA`.
# With _VIEW_GAMMA < 1 (default 0.5 = square root), mid-D voxels map higher into each shaded range,
# gaining intra-hue contrast; low-D and high-D endpoints stay fixed. Affects both live viewer and
# `--reach-visible-compare` (voxels AND colorbars use the same gamma-baked shaded cmap, so the
# colorbar tick at "0" still exactly matches a voxel at D=0 — no cheat, just a curved gradient).
_VIEW_GAMMA = 0.5

# Visible-mode cmap lookup range: dexterity D in [0,1] is stretched into [_CMAP_T_MIN,
# _CMAP_T_MAX] before indexing the Greens/Reds LUTs. Skipping [_CMAP_T_MIN, 0) keeps low-D voxels
# out of each cmap's dimmest entries (Greens(0) is washed-out; Reds(0) is too dark to read).
# Skipping (_CMAP_T_MAX, 1] avoids the near-white end of each colormap. The actual transform is
# gamma-curved (see _VIEW_GAMMA above) so mid-D voxels gain perceptual contrast.
_CMAP_T_MIN = 0.35
_CMAP_T_MAX = 0.95


def _shaded_cmap(base_name: str, gamma: float = _VIEW_GAMMA):
    """Matplotlib LinearSegmentedColormap sliced to [_CMAP_T_MIN, _CMAP_T_MAX] of `base_name`,
    then gamma-curved so cmap(s) = original(_CMAP_T_MIN + (_CMAP_T_MAX - _CMAP_T_MIN) * s ** gamma).

    Apply with ``Normalize(0.0, _DEXTERITY_VMAX)`` for BOTH voxels and colorbars -- the colorbar
    tick at "0" / "0.7" still matches a voxel at D=0 / D=vmax exactly (same LUT walk), and
    gamma < 1 just reshapes the in-between gradient, giving mid-D voxels more visual contrast.
    """
    import matplotlib.colors as mcolors
    s_grid = np.linspace(0.0, 1.0, 256)
    t_grid = _CMAP_T_MIN + (_CMAP_T_MAX - _CMAP_T_MIN) * s_grid ** gamma
    try:
        cmap_obj = plt.get_cmap(base_name)
    except ValueError:
        try:
            cmap_obj = plt.get_cmap(base_name.lower())
        except ValueError:
            cmap_obj = plt.get_cmap(base_name.capitalize())
    colors = cmap_obj(t_grid)[:, :3]
    return mcolors.LinearSegmentedColormap.from_list(f"{base_name}_shaded_g{gamma}", colors)

# Matplotlib cmap lookup tables (256-entry RGBA) as torch tensors. Sampled at startup via
# `(t * 255).long()` for fast per-voxel indexing in the live viewer -- avoids the matplotlib
# per-call + astopy overhead on the 10k+ voxel startup path.
class _CmapLutDict(dict):
    def __getitem__(self, key: str) -> "torch.Tensor":
        if key not in self:
            try:
                cmap_obj = plt.get_cmap(key)
            except ValueError:
                try:
                    cmap_obj = plt.get_cmap(key.lower())
                except ValueError:
                    cmap_obj = plt.get_cmap(key.capitalize())
            self[key] = torch.as_tensor(cmap_obj(np.linspace(0.0, 1.0, 256)).astype(np.float32))
        return super().__getitem__(key)

_CMAP_LUT: dict[str, "torch.Tensor"] = _CmapLutDict({
    name: torch.as_tensor(plt.get_cmap(name)(np.linspace(0.0, 1.0, 256)).astype(np.float32))
    for name in ("RdYlGn", "Reds", "Greens", "Greys", "Blues", "viridis", "YlGnBu")
})



# Head-camera rigs of humanoid_v21, keyed by `model_key`. Maps to the `get_spec(head_camera=...)`
# mode plus the per-eye (yaw joint, pitch joint, rgb site) triples that `_camera_visibility`
# iterates. A table rather than an `if single: ... else: ...` chain because that boolean silently
# scored a 3-camera rig as 2 (any non-single key fell into the dual branch), and because eye COUNT
# is the independent variable of the K ablation -- it must be data, not control flow.
_HUMANOID_RIGS: dict[str, tuple[str, tuple[tuple[str, str, str], ...]]] = {
    "humanoid_v21": ("actuated", (
        ("cam_yaw_left", "cam_pitch_left", "cam_left_rgb"),
        ("cam_yaw_right", "cam_pitch_right", "cam_right_rgb"),
    )),
    "humanoid_v21_single": ("actuated_single", (
        ("cam_yaw", "cam_pitch", "cam_rgb"),
    )),
    "humanoid_v21_triple": ("actuated_triple", (
        ("cam_yaw_left", "cam_pitch_left", "cam_left_rgb"),
        ("cam_yaw_center", "cam_pitch_center", "cam_center_rgb"),
        ("cam_yaw_right", "cam_pitch_right", "cam_right_rgb"),
    )),
}

# Head-camera site(s) per non-humanoid `model_key`, for `_robot_forward_azimuth`. humanoid_v21
# variants instead derive their site names from `_HUMANOID_RIGS` (3rd element of each eye triple)
# rather than duplicating them here.
_HEAD_CAM_SITES: dict[str, tuple[str, ...]] = {
    "unitree_g1": ("head_camera_rgb",),
    "toddlerbot": ("head_cam_left_site", "head_cam_right_site"),
    "booster_t1": ("head_cam_site",),
    "fourier_gr3": ("head_cam_site",),
    "pal_talos": ("head_cam_site",),
    "apptronik_apollo": ("head_cam_left_site", "head_cam_right_site"),
}


_HEAD_CAM_Z_INTO_HEAD = frozenset({"booster_t1", "fourier_gr3", "pal_talos"})
"""Robots whose head-camera site has local +Z pointing INTO the head (toward the robot's own
skull), not OUT along the optical axis like every other rig here -- the opposite of the repo-wide
"+Z = optical axis OUT" convention (`tasks/camera_perception.py`). Found by extracting a frame from
`vrw_video_fourier_gr3.mp4` and finding the robot's BACKPACK panel facing the camera at the azimuth
`_robot_forward_azimuth` computed as "front" -- v2's own video, computed the same way, correctly
shows its head-camera eyes facing the viewer, so this is specific to these 3 robots' site
authoring, not a bug in the general formula. `pts_debug/toddlerbot`/`apptronik_apollo` (same
dual-site-averaging code path as these 3) were checked too and do NOT need this: raw azimuth -180
matches the group that's already correct, only these 3 land at exactly 0 (the un-corrected
"backwards" value)."""


def _robot_forward_azimuth(model, data, model_key: str) -> float:
    """World-frame MuJoCo camera azimuth that looks at `model_key`'s FRONT (head cameras facing
    the viewer), derived from its head-camera site(s)' local +Z axis -- repo-wide convention "+Z =
    optical axis OUT" (`tasks/camera_perception.py`), averaged over stereo pairs. Needs `mj_forward`
    already run on `data` at the pose whose framing you want (arm pose doesn't matter -- head/neck
    kinematics are independent of it for every robot here).

    Not a hardcoded constant because it ISN'T uniform: measured, humanoid_v21/unitree_g1/toddlerbot/
    apptronik_apollo face world +X, but booster_t1/fourier_gr3/pal_talos face world -X (2026-08-19).
    A single fixed azimuth put half these robots' videos in a back view. Solving per-robot from the
    actual camera geometry is what makes `--vrw-video` correct across the whole `_ROBOTS` table
    without auditing each new asset's authoring convention by hand.
    """
    import mujoco

    site_names = ([triple[2] for triple in _HUMANOID_RIGS[model_key][1]] if model_key in _HUMANOID_RIGS
                  else list(_HEAD_CAM_SITES[model_key]))
    dirs = [data.site_xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)]
            .reshape(3, 3)[:, 2] for name in site_names]
    horiz = np.mean(dirs, axis=0)[:2]
    # humanoid_v21's dual rig mounts its L/R eyes toe'd fully OUT at rest -- opposite horizontal
    # signs (measured: left [+0.674, 0], right [-0.674, 0]) -- so the average cancels to exactly
    # zero instead of bisecting to "forward". Fall back to one eye alone rather than divide by 0;
    # verified against a direct render (side-by-side az=-60 vs az=180 snapshot) that the single-eye
    # direction IS the robot's front, not an arbitrary L/R pick.
    if np.linalg.norm(horiz) < 1e-6:
        horiz = dirs[0][:2]
    horiz = horiz / np.linalg.norm(horiz)
    if model_key in _HEAD_CAM_Z_INTO_HEAD:
        horiz = -horiz  # see _HEAD_CAM_Z_INTO_HEAD's own docstring
    # MuJoCo azimuth->world-direction mapping (empirically probed, matches the convention
    # `mjv_defaultCamera` uses): dir = (-cos(az), -sin(az)). Inverting for a desired `horiz`.
    return float(np.degrees(np.arctan2(-horiz[1], -horiz[0])))



@dataclass
class Args:
    """Reachability-index workspace figure / viewer inputs (see module docstring)."""

    robot: str | None = None
    """Robot to plot: one of `v2` (V2 dual-camera reach + steered/actuated visibility), `v2_fixed`
    (same V2 reach but fixed-gimbal visibility -- the paper's "Ours (fixed)" column, no steering
    expansion), `v2_single` / `v2_single_fixed` (K=1, one centered forward-facing module) and
    `v2_triple` / `v2_triple_fixed` (K=3, modules at y = +-0.130 and 0) -- every module carries
    collision spheres, so each K rig is solved against its own obstruction and owns its payload;
    they do NOT share v2's reach cloud, and scoring one rig against another's payload is exactly
    the error that produced the retracted single-camera column, `g1` (G1 reach + its single fixed
    head cam), `toddlerbot`, or `apptronik_apollo` (per-success-row dynamic neck/head visibility).
    `--input` overrides. Bare run (neither) uses v2."""

    input: Path = _DEFAULT_INPUT
    """R-arm workspace `.pt` sidecar. Overrides `--robot`. Defaults to the humanoid_v21 0.02 m
    reachability-index payload, so a bare run reproduces the paper figure."""

    view: bool = False
    """Open the interactive MuJoCo viewer (voxel spheres) instead of writing the figure;
    `section` is its upper Z cutoff. Default (unset) writes the PNG/PDF section figure."""

    view_mode: Literal["reachable", "visible", "instantaneous"] = "reachable"
    """`reachable` (default) = every reachable voxel colored by D. For ToddlerBot, `visible`
    renders only dynamic-FOV visible-reachable voxels, colored by `D_visible`, in either live
    `--view` mode or the static section render. V2/G1 `visible` remains live-view-only.
    `instantaneous` (ToddlerBot live `--view` only): at the slider-driven live head pose,
    re-evaluate the FOV envelope per voxel every control-state change and split the reachable
    cloud into Greens (instantaneously visible) vs Reds (instantaneously blind). Reveals what
    the eyes can actually see right now versus what offline IK says is reachable from a
    per-voxel legal head aim."""

    vis_cmap: str = "plasma"
    """Colormap for reachable-and-visible workspace (e.g. Greens, Blues, Viridis, YlGnBu)."""

    blind_cmap: str = "Blues"
    """Colormap for reachable-but-blind workspace (e.g. Greys for a neutral grayscale gradient, or Reds)."""

    visibility_input: Path | None = None
    """ToddlerBot dynamic-visibility sidecar. Default is matching canonical cache file. Required
    only when `--input` is a noncanonical ToddlerBot workspace and `--view-mode visible` is used."""

    sphere_r: float | None = None
    """`--view` only: initial voxel sphere radius (metres, geom size — rendered diameter is 2x),
    seeds the panel's live "sphere r" slider (range up to 3x this value) so it can be dragged
    during the session. Default (unset) picks `0.5 * grid_spacing * stride` (touching packing at
    the surviving `--stride` density) or `0.02` for legacy payloads without `grid_spacing`. Pass
    e.g. `0.01` for sparser look or `0.005` for individual-sphere visibility; above ~0.03 spheres
    clip and the cut panel subsamples past the scene `maxgeom` cap."""

    stride: int = 1
    """`--view` only: keep every Nth candidate voxel (after spatial-bucket dedup) for additional
    thinning. Voxels are deduped to one per ~1-sphere bucket at startup (since `pos` is
    aggregation-mean, not on a regular grid — raw index stride would scatter); stride then
    thins that already-uniform set. Trade density for staying under the viewer's `maxgeom`
    cap (~1000 in mujoco 3.10). Default 1 = no extra stride. Combined with --sphere-r it
    controls visual density."""

    alpha: float = 1.0
    """`--view` only: initial voxel-sphere opacity (0-1), seeds the panel's live "opacity"
    slider — drag it during the session to change translucency without restarting. <1 lets
    interior voxels and the welded robot mesh show through the sphere shell."""

    white_bg: bool = False
    """`--view` only: set initial viewer background color to white."""

    shadows: bool = False
    """`--view` only: enable shadows in live viewer / snapshot exports (default: False)."""

    flat: bool = True
    """`--view` only: enable flat unlit shading for bright uniform sphere/box colors (default: True)."""

    geom_type: Literal["sphere", "box"] = "box"
    """`--view` only: geometry shape for rendering points (sphere or box). (default: box)."""

    custom_viewer: bool = False
    """`--view` only: launch high-capacity custom GLFW viewer (supports 300,000+ geoms without downsampling warning)."""

    max_geom: int = 300000
    """`--view` only: maximum geometry capacity limit when using `--custom-viewer` (default: 300000)."""

    left_input: Path | None = None
    """Independently solved L-arm sidecar; bimanual field = per-voxel max(D_R, D_L). Overrides
    `mirror_left` when given."""

    mirror_left: bool = True
    """Derive L in-memory by Y-mirroring R (sagittal reflection on the symmetric grid), giving the
    bimanual field with no second payload. Instant. Sets L/R symmetry true by construction (IoU=1);
    pass an independently solved `left_input` to *measure* the residual, or `--no-mirror-left` for
    the R arm alone. Ignored if `left_input` is given."""

    section: float = 0.3
    """Upper Z cutoff (base frame, metres) for the (a) top view AND the paper figure Z cut; retained
    voxels project by max reachability index R. The live `--view` cut is instead set by
    `cut_point`/`cut_normal` (arbitrary plane), so this only drives the figure paths."""

    cut_point: tuple[float, float, float] = (0.0, 0.0, 0.3)
    """Live `--view` only: a point (base frame, metres) the cut plane passes through. With the
    default `cut_normal=(0,0,1)` this reproduces the Z<=0.3 top slab."""

    cut_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Live `--view` only: cut-plane normal (base frame; any direction, auto-normalized). Keep the
    voxels BEHIND the normal — `(p - cut_point) . n_hat <= 0`. Flip the normal to peel the other
    side; e.g. `(1,0,0)` keeps the rear (x<=cut_point.x), `(-1,0,0)` keeps the front."""

    symmetry: bool = False
    """Render the L/R symmetry figure instead of the section figure; needs `left_input` or
    `mirror_left`. Writes `<input_stem>_symmetry.png` (or `out`)."""

    out: Path | None = None
    """symmetry mode: output PNG path (default `<input_stem>_symmetry.png` beside `input`)."""

    rv_flat: bool = False
    """reach_visible_compare mode: drop the per-cell reachability shading (flat category colors)."""

    rv_video: bool = False
    """With `--reach-visible-compare`/`--camera-count-cuts`: ALSO render an animated `.mp4` beside the
    static `.png`/`.pdf`, same figure/layout/silhouette/colorbars, but each row's fixed section-cut
    boundary (top z<=shoulder height, side y>=0) sweeps across the FULL shared display window instead
    of sitting at one plane -- the top plane top-to-bottom, the side plane near-to-far, both driven by
    one shared progress fraction so every column's cut moves in lockstep and the two rows animate
    simultaneously, then reverse for a clean loop. See `_plot_reach_visible_grid`'s `video` branch."""

    rv_video_seconds: float = 6.0
    """`--rv-video` only: duration of ONE sweep pass (top/left extreme to bottom/right extreme); the
    written file ping-pongs there and back, so total length is ~2x this plus the hold pauses."""

    rv_video_fps: int = 30
    """`--rv-video` only: output frame rate."""

    rv_video_hold_frames: int = 8
    """`--rv-video` only: extra frames repeating each sweep extreme so playback pauses there instead
    of reversing direction immediately."""

    rv_video_3d: bool = False
    """With `--reach-visible-compare`: render a SEPARATE `reach_visible_compare_3d.mp4` (independent of
    `--rv-video` -- does not require it) -- a third top row, a per-column MuJoCo 3D perspective render
    of the reachable+visible voxels, PLUS switches all three rows to `--vrw-video`'s own SEQUENTIAL
    cut order (Z "xy" section, then Y "yz" section, one axis at a time, mirrored return for a
    jump-free loop) instead of the plain video's simultaneous sweep -- see `_plot_reach_visible_grid`'s
    `video_3d_row` docstring. Full resolution by default (see `--rv-video-3d-stride`), same box-per-
    voxel look and SDF sub-voxel interpolation as `--vrw-video`. Written separately (not merged into
    `reach_visible_compare.mp4`) since the 3-row layout is wider than the figure this repo ships, and
    the cut order differs from the plain video's. Only wired for `--reach-visible-compare`, not
    `--camera-count-cuts`."""

    rv_video_3d_stride: int = 1
    """`--rv-video-3d` only: voxel grid stride for the 3D row (independent of `--stride`, which this
    mode doesn't otherwise use). Default 1 = full resolution, matching `--vrw-video`'s own default --
    the per-frame cost is O(voxels + frames), not O(voxels x frames) (see `_rv3d_render_worker`), so
    the full reachable set (1-2e5 voxels/case) is tractable. Higher = coarser/faster, same convention
    as `--vrw-video` (box edge = grid_spacing x stride)."""

    reach_visible_compare: bool = False
    """Render the 8-case CROSS-ROBOT comparison figure (rows = top/side section cuts; columns =
    Ours-fixed / Ours-actuated / G1 / ToddlerBot / Booster T1 / Apptronik Apollo / Fourier GR-3 /
    PAL TALOS), each column binary green=visible / red=blind. Writes
    `reach_visible_compare.{png,pdf}` under result/ and mirrors the PDF to the paper. The K=1/2/3
    rigs live in `--camera-count-cuts`, not here."""

    camera_count_cuts: bool = False
    """Render the K = 1/2/3 camera-count section-cut figure: columns K=1 / K=2 / K=3 / lost 2->3,
    the paper's `fig:camera-count` spatial row. The fourth column is the reach a third camera costs
    (reachable at K=2, not at K=3) in its own flat palette, since its two categories are
    lost/retained rather than visible/blind. Column headers carry no volumes -- these are mirrored
    whole-body sets while the companion table is right-arm-only. Writes
    `camera_count_cuts.{png,pdf}` under result/ and mirrors the PDF to the paper."""

    reach_visible_single: bool = False
    """Render a single-column reach-visible figure for `--robot` (any `_ROBOTS` key, e.g.
    v2/v2_fixed/v2_single/v2_single_fixed/g1/toddlerbot) -- same Greens=visible/Reds=blind
    convention and column layout as `--reach-visible-compare`, a preview of one column before
    splicing it into that grid. Writes `reach_visible_<robot>.{png,pdf}` under result/ and mirrors
    the PDF to the paper."""

    vrw_3d: bool = False
    """Render teaser panel (b): the V2 `z <= shoulder` visible-reachable slab in 3D as a two-column
    figure, gimbals locked vs steered, to `result/teaser_vrw_3d.{png,pdf}`. Same plasma/Blues
    colormap, colorbars and cut as `--reach-visible-compare`, so the teaser and `fig:workspace`
    agree. Framing knobs: `--vrw-alpha`, `--vrw-azimuth`, `--vrw-elevation`, `--vrw-distance`,
    `--stride` (leave at 1 for the full-quality cloud). Needs a GPU (the visibility raycast)."""

    vrw_alpha: float = 1.0
    """`--vrw-3d` only: voxel opacity. Default is opaque: at the width panel (b) gets in the 1x4
    teaser row, any alpha below 1 averages voxels along the view ray and smears the visible/blind
    boundary -- which is the one thing the panel has to show -- and hides the flat shoulder cut
    face that marks this as a section. Values near 0.01 reveal the whole body through the cloud
    (a nicer silhouette, no readable coverage claim); see `_render_vrw_3d`."""

    vrw_azimuth: float = -60.0
    vrw_elevation: float = -25.0
    vrw_distance: float = 2.4
    """`--vrw-3d` only: camera framing (degrees, degrees, metres). Elevation < 0 looks down."""

    vrw_video: bool = False
    """Animate `--robot`'s visible-reachable slab: same pose/colors as `--vrw-3d` (one column, not
    the fixed-vs-actuated pair). Two chained sweep phases, no pause between them: (1) Z, lowest voxel
    to highest voxel; (2) world-X (a YZ-plane cut), nearest voxel to farthest voxel from the actual
    camera position. Writes `result/vrw_video_<robot>.mp4`. Shares `--vrw-alpha`/`--vrw-distance`/
    `--stride` with `--vrw-3d`, but NOT `--vrw-azimuth`/`--vrw-elevation` -- camera azimuth is instead
    solved per-robot from its head-camera geometry so every robot is shown from its own front (see
    `_robot_forward_azimuth`; a single fixed azimuth showed 3 of 12 robots from the back, since they
    don't share a world-facing convention). `--vrw-video-tilt`/`--vrw-video-elevation` steer the shot
    relative to that auto-detected front. See `_render_vrw_video`."""

    vrw_video_steps_per_layer: int = 2
    """`--vrw-video` only: interpolation frames per voxel layer along EACH phase's sweep axis (Z for
    phase 1, world-X for phase 2), grid_spacing apart -- NOT a total frame count. Each voxel's box
    grows from 0 to full size over its own layer's steps -- a 1D SDF cut (`signed_distance = cutoff -
    entry_face`, clamped to [0, full size]), not a per-frame binary include/exclude. Default 2 =
    "half point": one halfway frame then one full-size frame per layer -- a fixed, EVEN cadence
    deliberately chosen over a plain `linspace(lo, hi, n_frames)`, which does not divide evenly by
    `grid_spacing` (measured: 180/62 layers = 2.9/layer on v2) and produced a periodic pixel-diff
    spike (a whole layer's worth of new area popping in
    once every 2-3 frames, unevenly). 1 = instant pop, no interpolation. Higher smooths each layer's
    OWN growth further but does not shrink the once-per-layer new-area pop, which is a property of
    the voxel grid, not the frame rate -- see `_render_vrw_video`. Total sweep frames = (phase 1
    Z-layer count + phase 2 X-layer count) x this value; cheap regardless, since the render loop only
    touches a geom slot on the frame it first appears and while its own frac is still <1, never once
    it's fully grown."""

    vrw_video_fps: int = 60
    """`--vrw-video` only: output frame rate. Purely a playback-speed knob -- the render loop always
    produces the same (layers x steps_per_layer) frame count regardless of fps, so raising this
    costs nothing extra to render, just plays the same content back faster/shorter."""

    vrw_video_hold_frames: int = 5
    """`--vrw-video` only: extra frames repeating the final (uncut) frame so playback pauses on the
    full cloud instead of ending mid-sweep."""

    vrw_video_res: int = 1080
    """`--vrw-video` only: square output resolution in pixels (width == height). The cloud's tallest
    stance is a narrow standing figure and its widest is a torso-height disc, neither of which use a
    16:9 frame's extra width -- square wastes nothing."""

    vrw_video_tilt: float = 20.0
    """`--vrw-video` only: azimuth offset (degrees) from the robot's auto-detected front-facing
    direction (`_robot_forward_azimuth`) -- 0 is a dead-on front view, this gives a slight 3/4 turn.
    Same value applies to every `--robot`, which is what makes the renders comparable side by side
    despite different robots facing different world directions."""

    vrw_video_elevation: float = -25.0
    """`--vrw-video` only: camera elevation (degrees, < 0 looks down), shared across every robot for
    the same side-by-side comparability as `--vrw-video-tilt`."""

    cache: bool = False
    """Write the small aggregated fallback cache
    (aggregated_cache/<payload_stem>_aggregated.pt) for the resolved --robot/--input payload after
    loading it from the raw checkpoint, then exit. Requires the raw payload (and, for
    dynamic-visibility robots, its visibility sidecar) to be present -- this is the one-time step
    that produces the small, git-committable file other checkouts/collaborators can run this
    script against without the multi-GB raw checkpoints. Mutually exclusive with every other run
    mode."""

    cache_all: bool = False
    """Same as `--cache`, but iterates every entry in `_ROBOTS` and writes each
    one's aggregated_cache/*.pt in turn -- the one-shot way to refresh the whole committable cache
    set after regenerating one or more raw payloads. Robots whose raw payload (or, for
    dynamic-visibility robots, its visibility sidecar) isn't present locally are skipped with a
    printed warning rather than aborting the batch, since not every machine holds every robot's
    multi-GB checkpoint. Mutually exclusive with every other run mode."""

    rv3d_worker_in: Path | None = None
    rv3d_worker_out: Path | None = None
    """Internal -- `_plot_reach_visible_grid`'s `video_3d_row` re-invokes this script with these two
    set, as a subprocess, to render the 3D row's frames in a fresh process (see `_rv3d_render_worker`
    docstring). Not meant to be passed by hand."""

    orientation_debug: bool = False
    """Debug aid, not a shipped figure: base-link XYZ axes (red/green/blue) overlaid on each of the
    6 `--reach-visible-compare` robots, no voxels, across the same Top/Side/3D camera views the
    comparison video uses. Writes `result/orientation_debug.png`. See `_render_orientation_debug`."""

    orientation_debug_worker_robot: str | None = None
    orientation_debug_worker_view: int = 0
    orientation_debug_worker_bbox_radius: float = 0.0
    orientation_debug_worker_out: Path | None = None
    """Internal -- `_render_orientation_debug` re-invokes this script with these set, as a
    subprocess per (robot, view) pair, to render each in a fresh process (see
    `_orientation_debug_worker` docstring for why one pair per process). Not meant to be passed by
    hand."""




# Explicit display pose for the workspace overlay: arms abducted to horizontal via
# shoulder_2 = +-pi/2, every other joint at qpos0 (zero). Keeps the body out of the
# voxel volume so the reachable set reads cleanly.
_OVERLAY_POSE = {
    "right_shoulder_2_joint": np.pi / 2,
    "left_shoulder_2_joint": -np.pi / 2,
}

# Per-robot overlay pose: same intent (abduct arms clear of the voxel volume), different joint
# names. G1 abducts via shoulder_roll (+ left / - right); every other joint at qpos0.
_OVERLAY_POSE_BY_ROBOT = {
    "humanoid_v21": _OVERLAY_POSE,
    "unitree_g1": {
        "left_shoulder_roll_joint": np.pi / 2,
        "right_shoulder_roll_joint": -np.pi / 2,
    },
}

# reach-visible occluder / silhouette pose per robot: arms hung DOWN so the forearms + gripper sit
# clear of the head-camera FOV (the least-occluding canonical pose -- a fair visibility baseline).
# The SAME pose is the raycast occluder in `_camera_visibility*` and the drawn silhouette in
# the compare figure, so the body that produces the blind regions is the body shown.
# humanoid_v21 is already arms-down at qpos0 (empty override). G1's arm zero-config bends the elbow
# 90 deg FORWARD (forearms jut into the front cone); rotating elbow +90 deg drops the forearms
# alongside the thighs and wrist_roll +90 deg turns the gripper flat to the leg (both within ROM).
_ARMS_DOWN_POSE = {
    "humanoid_v21": {},
    "unitree_g1": {
        "left_elbow_joint": np.pi / 2,
        "right_elbow_joint": np.pi / 2,
        "left_wrist_roll_joint": np.pi / 2,
        "right_wrist_roll_joint": np.pi / 2,
    },
    "toddlerbot": {},
    "booster_t1": {
        "Left_Shoulder_Roll": -np.pi / 2,
        "Right_Shoulder_Roll": np.pi / 2,
        "Left_Elbow_Yaw": 0.0,
        "Right_Elbow_Yaw": 0.0,
    },
    # GR-3's source qpos0 already hangs both arms straight down at the sides, so no override
    # is needed to clear the head camera's field of view.
    "fourier_gr3": {},
    # TALOS's arms-down pose is its `standing` keyframe, which `set_home_pose` now resolves
    # via `mj_home_pose.home_qpos`; its qpos0 is an illegal configuration and must not be used.
    "pal_talos": {},
    "apptronik_apollo": {},
}


def _attach_white_skybox_to_spec(spec) -> None:
    """Attach a flat white skybox texture to MjSpec for toggling white/dark viewer background."""
    import mujoco
    tex = spec.add_texture()
    tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
    tex.rgb1 = [1.0, 1.0, 1.0]
    tex.rgb2 = [1.0, 1.0, 1.0]
    tex.mark = mujoco.mjtMark.mjMARK_NONE
    tex.width = 16
    tex.height = 16


def _overlay_assets(robot: str):
    """Resolve (welded_model, home_pos, overlay_pose, visual_mask_fn) for the payload's robot.

    Deferred imports keep the humanoid / no-overlay paths from pulling G1 assets. G1's welded model
    and its ``HOME_KEYFRAME.pos`` come from ``g1_constants``; the humanoid default is unchanged.

    ``visual_mask_fn(model) -> bool[ngeom]`` selects the meshes kept in the paper figure (collision
    proxies / camera frusta hidden). humanoid_v21 tags visual meshes with a ``_visual`` name suffix;
    G1's mjlab meshes are unnamed but sit in MuJoCo visual group 2 — selecting by suffix there would
    hide the entire body. Kept per-robot so the humanoid figure stays byte-identical."""
    import mujoco

    def _compile_with_skybox(xml_path: Path):
        spec = mujoco.MjSpec.from_file(str(xml_path))
        _attach_white_skybox_to_spec(spec)
        return spec.compile()

    if robot == "unitree_g1":
        from mj_envs.asset_zoo.g1.g1_constants import get_g1_robot_cfg, HOME_KEYFRAME
        from mjlab.entity import Entity
        cfg = get_g1_robot_cfg(head_camera="builtin", end_effector="welded", hand="parallel_gripper")
        entity = Entity(cfg)
        _attach_white_skybox_to_spec(entity.spec)
        return (entity.compile(), np.asarray(HOME_KEYFRAME.pos, np.float64),
                _OVERLAY_POSE_BY_ROBOT["unitree_g1"], lambda m: np.asarray(m.geom_group) == 2)
    if robot == "toddlerbot":
        toddler_mjcf = _REPO_ROOT / "asset" / "toddlerbot_2xm_gripper" / "toddlerbot_2xm_gripper_pos.xml"
        return (_compile_with_skybox(toddler_mjcf), np.array([0.0, 0.0, 0.315053]),
                {}, lambda m: np.asarray(m.geom_group) == 2)
    if robot == "booster_t1":
        t1_mjcf = _REPO_ROOT / "asset" / "booster_t1" / "t1.xml"
        return (_compile_with_skybox(t1_mjcf), np.array([0.0, 0.0, 0.665]),
                {}, lambda m: np.asarray(m.geom_group) == 2)
    if robot == "fourier_gr3":
        gr3_mjcf = _REPO_ROOT / "asset" / "fourier_gr3" / "gr3.xml"
        return (_compile_with_skybox(gr3_mjcf), np.array([0.0, 0.0, 0.9270331]),
                {}, lambda m: np.asarray(m.geom_group) == 2)
    if robot == "pal_talos":
        talos_mjcf = _REPO_ROOT / "asset" / "pal_talos" / "talos_study.xml"
        return (_compile_with_skybox(talos_mjcf), np.array([0.0, 0.0, 1.08205]),
                {}, lambda m: np.asarray(m.geom_group) == 2)
    if robot == "apptronik_apollo":
        apollo_mjcf = _REPO_ROOT / "asset" / "apptronik_apollo" / "apptronik_apollo.xml"
        return (_compile_with_skybox(apollo_mjcf), np.array([0.0, 0.0, 1.0813]),
                {}, lambda m: np.asarray(m.geom_group) == 1)

    from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_spec, HOME_KEYFRAME
    # Dual keeps "welded" (its gimbal needs no DOF to render at rest, and this is the shipped path
    # whose figures must not shift); the K=1/K=3 rigs only exist in actuated form, so they take it
    # from the rig table. Without this every humanoid overlay drew the DUAL head regardless of K.
    head_camera = "welded" if robot == "humanoid_v21" else _HUMANOID_RIGS[robot][0]
    spec = get_spec(head_camera=head_camera, end_effector="welded", hand="parallel_gripper")
    _attach_white_skybox_to_spec(spec)

    def _humanoid_visual(m):
        # The head camera's pitch-link mesh is split into multi-material sub-geoms
        # (`cam_pitch_{left,right}_link_visual0`/`_visual1` -- silver eye-box vs black housing),
        # so a bare `.endswith("_visual")` drops half the camera module's geometry. `_visual`
        # optionally followed by digits, anchored at the end, catches both the plain and the
        # split-mesh names without matching `_visual_fov*`/collision/site names elsewhere.
        pattern = re.compile(r"_visual\d*$")
        return np.array([
            bool(pattern.search(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""))
            for i in range(m.ngeom)
        ])

    return (spec.compile(), np.asarray(HOME_KEYFRAME.pos, np.float64),
            _OVERLAY_POSE_BY_ROBOT["humanoid_v21"], _humanoid_visual)


def _render_robot_overlay(
    points: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
    path: Path,
    sphere_r: float = 0.02,
    alpha: float = 1.0,
    azimuths: tuple[int, ...] = (135, 200, 270),
    azimuth: int | None = None,
) -> None:
    """Render the welded robot at home pose with one RdYlGn sphere per reachable voxel.

    Points are voxel centers in the cuRobo base frame (world - HOME_KEYFRAME.pos, identity
    base orientation), so world placement is a pure translation by HOME_KEYFRAME.pos — the
    same convention probe_orientation_quiver uses. Spheres are colored by `values` (D in
    [0,1]) with RdYlGn; a matplotlib colorbar strip is composited on the right. Heavy
    cuRobo/warp imports are local so the matplotlib fallback path needs no GPU toolchain.

    `alpha` < 1 makes the spheres translucent so the INTERIOR voxels and the robot mesh
    behind the outer layer stay visible (the volume reads as a solid cloud, not an opaque
    shell) — MuJoCo depth-sorts and blends per-geom rgba.
    """
    import mujoco

    from mj_envs.asset_zoo.reachability_study.generate_workspace_curobo import _load_welded_model
    from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import HOME_KEYFRAME

    finite = np.isfinite(values)
    pts = points[finite]
    vals = values[finite]
    if pts.shape[0] == 0:
        raise ValueError("no finite voxels to render")
    vmin, vmax = 0.0, max(float(np.nanmax(vals)), 1e-6)
    cmap = plt.get_cmap("RdYlGn")
    norm = (vals - vmin) / (vmax - vmin)
    rgba = cmap(np.clip(norm, 0.0, 1.0)).astype(np.float32)
    rgba[:, 3] = alpha  # translucent voxels expose interior samples + robot mesh

    model = _load_welded_model()
    hres, wres = 900, 1100
    model.vis.global_.offwidth = wres
    model.vis.global_.offheight = hres
    data = mujoco.MjData(model)
    base_pos = set_home_pose(model, data, pose_overrides=_OVERLAY_POSE)
    world = pts.astype(np.float64) + base_pos

    r = mujoco.Renderer(model, hres, wres, max_geom=max(20000, pts.shape[0] + 1000))
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # Fit vertically over BOTH the voxel volume (top spheres above the head) and the robot
    # body (feet at ground) so neither the volume top nor the feet get clipped.
    z_lo = min(world[:, 2].min(), data.geom_xpos[:, 2].min())
    z_hi = max(world[:, 2].max(), data.geom_xpos[:, 2].max())
    zc = 0.5 * (z_lo + z_hi)
    zspan = z_hi - z_lo
    cam.lookat[:] = [0.1, 0.0, zc - 0.1]
    cam.distance = max(1.9, 1.25 * zspan)
    cam.elevation = -10
    eye = np.eye(3, dtype=np.float64).flatten()
    size = np.array([sphere_r, 0.0, 0.0], dtype=np.float64)

    az_list = (azimuth,) if azimuth is not None else azimuths
    frames = []
    for az in az_list:
        cam.azimuth = az
        r.update_scene(data, cam)
        scn = r.scene
        n0 = scn.ngeom
        n_add = min(world.shape[0], scn.maxgeom - n0)
        for i in range(n_add):
            g = scn.geoms[n0 + i]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size,
                                world[i], eye, rgba[i])
        scn.ngeom = n0 + n_add
        rgb = r.render()
        # Depth pass on the same scene to build a transparent-background alpha mask:
        # background pixels sit at the far plane (max depth) -> alpha 0.
        r.enable_depth_rendering()
        depth = r.render()
        r.disable_depth_rendering()
        alpha = np.where(depth >= depth.max() - 1e-6, 0, 255).astype(np.uint8)
        frames.append(np.dstack([rgb, alpha]))
    render = np.concatenate(frames, axis=1)

    # Colorbar strip via matplotlib, then composite to the right of the render (opaque).
    fig, ax = plt.subplots(figsize=(1.6, hres / 100.0), dpi=100)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=ax, label=label)
    fig.tight_layout()
    fig.canvas.draw()
    bar = np.asarray(fig.canvas.buffer_rgba())  # RGBA, alpha 255
    plt.close(fig)
    if bar.shape[0] != render.shape[0]:
        idx = (np.linspace(0, bar.shape[0] - 1, render.shape[0])).astype(int)
        bar = bar[idx]
    canvas = np.concatenate([render, bar], axis=1)

    import imageio
    imageio.imwrite(str(path), canvas)
    print(f"saved {path}  ({canvas.shape[1]}x{canvas.shape[0]}, {pts.shape[0]} voxels)")


def _halfspace_mask(pos: np.ndarray, p0, n) -> np.ndarray:
    """Keep voxels BEHIND the cut plane through `p0` with normal `n`: `(pos - p0) . n_hat <= 0`.

    A zero-length normal defines no plane, so keep everything (viewer shows the full cloud).
    """
    n = np.asarray(n, np.float64)
    norm = np.linalg.norm(n)
    if norm == 0.0:
        return np.ones(pos.shape[0], dtype=bool)
    return (pos - np.asarray(p0, np.float64)) @ (n / norm) <= 0.0


def _normal_to_mat(n) -> np.ndarray:
    """Row-major 3x3 rotation (flattened) whose local +Z maps to unit `n` — orients the cut-plane
    box so its thin axis is the plane normal. Arbitrary in-plane roll (Gram-Schmidt off a seed axis).
    """
    n = np.asarray(n, np.float64)
    norm = np.linalg.norm(n)
    n = n / norm if norm > 0.0 else np.array([0.0, 0.0, 1.0])
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(seed, n); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return np.column_stack([t1, t2, n]).flatten()


def _v2_fov_corners(h_half: float, v_half: float, depth: float) -> list[np.ndarray]:
    """Return the 4 rectangle corners in camera-local (+Z optical) frame at given depth (m)."""
    h = depth * np.tan(h_half)
    v = depth * np.tan(v_half)
    return [np.array([+h, +v, depth]), np.array([-h, +v, depth]),
            np.array([-h, -v, depth]), np.array([+h, -v, depth])]


def _add_v2_fov(model: mujoco.MjModel, data: mujoco.MjData, scene) -> None:
    """Draw V2 dual gimbal FOV wireframe at the pose already baked into `data` by `mj_forward`.

    Uses `data.site_xpos` / `data.site_xmat` directly -- MuJoCo's FK has applied the slider
    values to `cam_yaw_{side}` / `cam_pitch_{side}` and any home-pose root translation, so this
    pose matches the lens pose the visibility fn reads from the same `data`. Mirrors
    `view_toddlerbot._add_fov`. 2 eyes x 13 segments = 26 geoms.
    """
    from tasks.camera_terms import H_HALF, V_HALF, FAR

    eyes = (("left", "cam_left_rgb", (0.2, 0.7, 1.0, 0.9)),
            ("right", "cam_right_rgb", (1.0, 0.55, 0.1, 0.9)))
    near_d, far_d = _RV_NEAR_M, float(FAR)
    for side, site_name, rgba in eyes:
        st = model.site(site_name)
        center = data.site_xpos[st.id]
        rot = data.site_xmat[st.id].reshape(3, 3)  # columns: camera-local x, y, z axes in world
        for depth in (near_d, far_d):
            corners_local = _v2_fov_corners(H_HALF, V_HALF, depth)
            corners_world = [center + rot @ c for c in corners_local]
            for i in range(4):
                add_segment(scene, corners_world[i], corners_world[(i + 1) % 4], rgba)
        corners_far_local = _v2_fov_corners(H_HALF, V_HALF, far_d)
        corners_far_world = [center + rot @ c for c in corners_far_local]
        for i in range(4):
            add_segment(scene, center, corners_far_world[i], rgba)
        add_segment(scene, center, center + 0.20 * rot[:, 2], (1.0, 1.0, 0.0, 1.0), width=0.004)


V2_FOV_SEGMENT_COUNT = 2 * (4 + 4 + 4 + 1)


T1_FOV_SEGMENT_COUNT = (4 + 4 + 4 + 1)  # single D455 eye on Booster T1 head


def _add_t1_fov(model: mujoco.MjModel, data: mujoco.MjData, scene) -> None:
    """Draw T1 single D455 FOV wireframe at the pose already baked into `data` by `mj_forward`.

    Mirrors `_add_v2_fov` but for a single eye (`head_cam`/`head_cam_site`). Corners along
    MuJoCo's local `-Z` (camera forward) -- opposite of V2's `+Z` convention. Per-eye cost:
    4 (near square) + 4 (far square) + 4 (rays) + 1 (axis) = 13 segments. Read fovy/resolution
    straight from the MJCF, so changing the D455 intrinsics in source XML propagates here.
    """
    from mj_envs.asset_zoo.reachability_study.t1_visibility import _camera_intrinsics

    camera_name, site_name, rgba = "head_cam", "head_cam_site", (0.2, 0.7, 1.0, 0.9)
    fx, fy, cx, cy, width, height = _camera_intrinsics(model, camera_name)
    fovy_rad = np.deg2rad(float(model.cam_fovy[model.camera(camera_name).id]))
    v_half = fovy_rad / 2.0
    h_half = np.arctan(width / height * np.tan(v_half))
    near_d, far_d = 0.10, 1.00  # covers T1's reachable-voxel extent (~0.74 m from base_link)

    site_id = model.site(site_name).id
    pos = data.site_xpos[site_id]
    rot = data.site_xmat[site_id].reshape(3, 3)
    forward = -rot[:, 2]  # MuJoCo camera looks along local -Z

    def corners(depth):
        h = depth * np.tan(h_half)
        v = depth * np.tan(v_half)
        # local point (x, y, -depth) [MuJoCo -Z forward] -> world: pos + x*rot[:,0] + y*rot[:,1]
        # + depth*forward (since forward = -rot[:,2]). T1's head_cam mount quat leaves local X =
        # vertical, local Y = horizontal (opposite of the generic convention) -- see
        # t1_visibility.eye_in_fov -- so v pairs with rot[:,0] and h pairs with rot[:,1].
        return [
            pos + depth * forward + h * rot[:, 1] + v * rot[:, 0],
            pos + depth * forward - h * rot[:, 1] + v * rot[:, 0],
            pos + depth * forward - h * rot[:, 1] - v * rot[:, 0],
            pos + depth * forward + h * rot[:, 1] - v * rot[:, 0],
        ]

    for depth in (near_d, far_d):
        rect = corners(depth)
        for i in range(4):
            add_segment(scene, rect[i], rect[(i + 1) % 4], rgba)
    far_rect = corners(far_d)
    for i in range(4):
        add_segment(scene, pos, far_rect[i], rgba)
    add_segment(scene, pos, pos + 0.20 * forward, (1.0, 1.0, 0.0, 1.0), width=0.004)


def _export_viewer_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: mujoco.MjvCamera,
    n_to_render: int,
    kept_world: np.ndarray,
    kept_rgba: np.ndarray,
    sphere_r: float,
    out_path: Path,
    transparent: bool = True,
    shadows: bool = False,
    flat: bool = True,
    geom_type: str = "box",
    draw_fov_fn: Callable | None = None,
    resolution: tuple[int, int] = (1920, 1080),
) -> Path:
    """Render the exact current viewer scene state offscreen and export as high-res PNG."""
    import mujoco
    from PIL import Image

    w, h = resolution
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)

    renderer = mujoco.Renderer(model, h, w, max_geom=max(1, n_to_render) + model.ngeom + 1000)
    # Sites (e.g. the camera module's unsized `cam_*_rgb`/`cam_*_cam_front_center` sensor sites)
    # default to group 0, which `MjvOption`'s default `sitegroup` renders as small spheres --
    # visible as two stray dots beside the head-camera housings in every figure this feeds. Not a
    # meaningful marker for any of these renders, so site visualization is off entirely.
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = 0
    snap_cam = mujoco.MjvCamera()
    snap_cam.lookat[:] = cam.lookat
    snap_cam.distance = cam.distance
    snap_cam.azimuth = cam.azimuth
    snap_cam.elevation = cam.elevation
    snap_cam.type = cam.type

    renderer.update_scene(data, camera=snap_cam, scene_option=scene_option)
    scn = renderer.scene
    if geom_type == "box":
        g_type = mujoco.mjtGeom.mjGEOM_BOX
        size = np.array([sphere_r, sphere_r, sphere_r], dtype=np.float64)
    else:
        g_type = mujoco.mjtGeom.mjGEOM_SPHERE
        size = np.array([sphere_r, 0.0, 0.0], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64).flatten()

    for i in range(n_to_render):
        if scn.ngeom >= scn.maxgeom:
            break
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            g_type,
            size,
            kept_world[i],
            eye,
            kept_rgba[i],
        )
        if flat:
            geom.emission = 0.4
            geom.specular = 0.0
        else:
            geom.emission = 0.0
        scn.ngeom += 1

    if draw_fov_fn is not None:
        draw_fov_fn(model, data, scn)

    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
    if not transparent:
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    else:
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0

    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    renderer.close()

    bg_mask = depth >= (depth.max() - 1e-5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if transparent:
        alpha = np.where(bg_mask, 0, 255).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        Image.fromarray(rgba).save(str(out_path))
    else:
        rgb_copy = rgb.copy()
        rgb_copy[bg_mask] = [255, 255, 255]
        Image.fromarray(rgb_copy).save(str(out_path))

    print(f"[Snapshot] Exported high-res PNG ({w}x{h}, transparent={transparent}, shadows={shadows}, flat={flat}, shape={geom_type}) to {out_path}")
    return out_path


def _show_robot_overlay_live(
    points: np.ndarray,
    values: np.ndarray,
    success: np.ndarray,
    *,
    cut_point: tuple[float, float, float],
    cut_normal: tuple[float, float, float],
    sphere_r: float = 0.02,
    alpha: float = 1.0,
    white_bg: bool = False,
    shadows: bool = False,
    flat: bool = True,
    geom_type: str = "box",
    robot: str = "humanoid_v21",
    stride: int = 1,
    view_mode: str = "reachable",
    visible: np.ndarray | None = None,
    instantaneous_visible_fn: Callable[[np.ndarray, np.ndarray, "mujoco.MjModel", "mujoco.MjData"], np.ndarray] | None = None,
    available_modes: tuple[str, ...] = ("reachable", "visible", "instantaneous"),
    vis_cmap: str = "Greens",
    blind_cmap: str = "Greys",
    custom_viewer: bool = False,
    max_geom_cap: int = 300000,
) -> None:
    """MuJoCo passive viewer (welded robot + voxel spheres) with a live DearPyGui cut panel."""
    import datetime
    import os
    import time

    import mujoco
    import mujoco.viewer as mjviewer


    if not os.environ.get("DISPLAY"):
        raise RuntimeError("--view needs an X11 DISPLAY (opens MuJoCo + DearPyGui windows)")
    import dearpygui.dearpygui as dpg

    finite = success & np.isfinite(values)
    pos = points[finite].astype(np.float64)  # base-frame voxel centers (masked against the cut plane)
    vals = values[finite]
    if pos.shape[0] == 0:
        raise ValueError("no finite voxels to show")
    vis_mask = np.asarray(visible, dtype=bool)[np.asarray(finite, dtype=bool)] if visible is not None else None

    model, home_pos, overlay_pose, _ = _overlay_assets(robot)
    if robot == "humanoid_v21" and "instantaneous" in available_modes:
        from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_spec
        spec = get_spec(head_camera="actuated", end_effector="welded", hand="parallel_gripper")
        _attach_white_skybox_to_spec(spec)
        model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    base_pos = set_home_pose(model, data, pose_overrides=overlay_pose, home_pos=home_pos)
    world = pos + base_pos

    pos_t = torch.as_tensor(pos, dtype=torch.float64)
    axis_coords = tuple(torch.unique(pos_t[:, a]) for a in range(3))
    grid_xi = torch.searchsorted(axis_coords[0], pos_t[:, 0].contiguous())
    grid_yi = torch.searchsorted(axis_coords[1], pos_t[:, 1].contiguous())
    grid_zi = torch.searchsorted(axis_coords[2], pos_t[:, 2].contiguous())
    if stride > 1:
        grid_stride_mask = (grid_xi % stride == 0) & (grid_yi % stride == 0) & (grid_zi % stride == 0)
    else:
        grid_stride_mask = torch.ones(pos.shape[0], dtype=torch.bool)

    eye = np.eye(3, dtype=np.float64).flatten()
    lo, hi = pos.min(0), pos.max(0)

    show_fov = robot in ("toddlerbot", "apptronik_apollo") or (
        robot in ("humanoid_v21", "booster_t1", "fourier_gr3", "pal_talos")
        and "instantaneous" in available_modes
    )
    fov_segment_count = 0
    apply_pose_fn = None
    draw_fov_fn = None
    neck_tags: list[str] = []
    neck_labels: list[str] = []
    neck_limits: list[tuple[float, float]] = []
    if show_fov:
        if robot == "toddlerbot":
            from mj_envs.asset_zoo.reachability_study.toddlerbot_visibility import (
                _HEAD_PITCH, _HEAD_YAW_DRIVE, _HEAD_YAW_DRIVE_RATIO, _HEAD_YAW_DRIVEN,
            )
            from mj_envs.asset_zoo.reachability_study.view_toddlerbot import _add_fov, FOV_SEGMENT_COUNT as _FS_COUNT
            direct_joints = [("neck_yaw", "neck yaw", _HEAD_YAW_DRIVEN), ("neck_pitch", "neck pitch", _HEAD_PITCH)]
            coupled_joints = [(_HEAD_YAW_DRIVE, _HEAD_YAW_DRIVEN, _HEAD_YAW_DRIVE_RATIO)]
            fov_segment_count = _FS_COUNT
            draw_fov_fn = _add_fov
        elif robot == "apptronik_apollo":
            from mj_envs.asset_zoo.reachability_study.view_apptronik_apollo import _add_audit_markers
            direct_joints = [
                ("neck_yaw", "neck yaw", "neck_yaw"),
                ("neck_roll", "neck roll", "neck_roll"),
                ("neck_pitch", "neck pitch", "neck_pitch"),
            ]
            coupled_joints = []
            fov_segment_count = 22
            draw_fov_fn = _add_audit_markers
        elif robot == "humanoid_v21":
            from tasks.camera_terms import H_HALF, V_HALF  # noqa
            direct_joints = [
                ("cam_yaw_left", "left yaw", "cam_yaw_left"), ("cam_pitch_left", "left pitch", "cam_pitch_left"),
                ("cam_yaw_right", "right yaw", "cam_yaw_right"), ("cam_pitch_right", "right pitch", "cam_pitch_right"),
            ]
            coupled_joints = []
            fov_segment_count = V2_FOV_SEGMENT_COUNT
            draw_fov_fn = _add_v2_fov
        elif robot == "booster_t1":
            direct_joints = [
                ("AAHead_yaw", "head yaw", "AAHead_yaw"),
                ("Head_pitch", "head pitch", "Head_pitch"),
            ]
            coupled_joints = []
            fov_segment_count = T1_FOV_SEGMENT_COUNT
            draw_fov_fn = _add_t1_fov
        elif robot == "fourier_gr3":
            direct_joints = [
                ("head_yaw_joint", "head yaw", "head_yaw_joint"),
                ("head_pitch_joint", "head pitch", "head_pitch_joint"),
            ]
            coupled_joints = []
        elif robot == "pal_talos":
            direct_joints = [
                ("head_1_joint", "head tilt", "head_1_joint"),
                ("head_2_joint", "head pan", "head_2_joint"),
            ]
            coupled_joints = []

        neck_tags = [tag for tag, _, _ in direct_joints]
        neck_labels = [label for _, label, _ in direct_joints]
        neck_limits = [tuple(model.joint(joint_name).range) for _, _, joint_name in direct_joints]
        direct_qpos = [int(model.jnt_qposadr[model.joint(joint_name).id]) for _, _, joint_name in direct_joints]
        coupled_qpos = [
            (int(model.jnt_qposadr[model.joint(drive_name).id]),
             int(model.jnt_qposadr[model.joint(driven_name).id]), ratio)
            for drive_name, driven_name, ratio in coupled_joints
        ]

        def apply_pose_fn(*values, _addrs=direct_qpos, _coupled=coupled_qpos, _m=model, _d=data):
            for addr, value in zip(_addrs, values):
                _d.qpos[addr] = value
            for drive_addr, driven_addr, ratio in _coupled:
                _d.qpos[drive_addr] = ratio * _d.qpos[driven_addr]
            mujoco.mj_forward(_m, _d)

    hide_blind_available = view_mode in ("visible", "instantaneous")
    available_modes = tuple(available_modes) or ("reachable",)
    initial_mode = view_mode if view_mode in available_modes else available_modes[0]
    mode_labels = {"reachable": "reachable (RdYlGn D)",
                   "visible": "visible (Greens=vis, Reds=blind)",
                   "instantaneous": "instantaneous (live FOV)"}
    dpg.create_context()
    with dpg.window(label="workspace cut", tag="cut_win", no_collapse=True):
        dpg.add_checkbox(label="enable cut", tag="enable", default_value=True)
        dpg.add_checkbox(label="white background", tag="white_bg", default_value=bool(white_bg))
        dpg.add_checkbox(label="shadows", tag="shadows", default_value=bool(shadows))
        dpg.add_checkbox(label="flat shading", tag="flat", default_value=bool(flat))
        if custom_viewer:
            # custom_viewer draws its own GLFW window with no native mujoco UI, so the usual
            # "press 0-5 to toggle geom group" key binding from the built-in viewer doesn't exist
            # here -- replicate it as checkboxes, wired to opt.geomgroup[i] below.
            dpg.add_text("geom groups (mujoco default: 0-2 on, 3-5 off)")
            for g in range(6):
                dpg.add_checkbox(label=f"group {g}", tag=f"geomgroup{g}", default_value=g < 3)
        dpg.add_combo(label="shape", tag="geom_type", items=["sphere", "box"], default_value=geom_type)
        if len(available_modes) > 1:
            dpg.add_text("render mode")
            dpg.add_radio_button(tag="mode", items=tuple(mode_labels[m] for m in available_modes),
                                 default_value=mode_labels[initial_mode], horizontal=False)
        if hide_blind_available:
            dpg.add_checkbox(label="hide blind (reds)", tag="hide_blind", default_value=False)
        dpg.add_drag_float(label="opacity", tag="alpha", default_value=float(alpha),
                           min_value=0.02, max_value=1.0, clamped=True, format="%.3f", speed=0.002)
        dpg.add_slider_float(label="sphere r (m)", tag="sphere_r", default_value=float(sphere_r),
                             min_value=0.002, max_value=3.0 * sphere_r, clamped=True, format="%.4f")
        dpg.add_slider_float(label="min reachability D", tag="min_d", default_value=0.0,
                             min_value=0.0, max_value=1.0, clamped=True, format="%.3f")
        dpg.add_text("cut point (m, base frame)")
        for i, ax in enumerate("xyz"):
            dpg.add_slider_float(label=f"p{ax}", tag=f"p{ax}", default_value=float(cut_point[i]),
                                 min_value=float(lo[i]), max_value=float(hi[i]), clamped=True, format="%.3f")
        dpg.add_text("cut normal (auto-normalized)")
        for i, ax in enumerate("xyz"):
            dpg.add_slider_float(label=f"n{ax}", tag=f"n{ax}", default_value=float(cut_normal[i]),
                                 min_value=-1.0, max_value=1.0, clamped=True, format="%.2f")
        if show_fov:
            dpg.add_text("camera aim (rad)")
            for tag, label, limits in zip(neck_tags, neck_labels, neck_limits):
                dpg.add_slider_float(label=label, tag=tag, default_value=0.0,
                                     min_value=float(limits[0]), max_value=float(limits[1]),
                                     clamped=True, format="%.2f")
        dpg.add_text("snapshot export")
        dpg.add_button(label="Save PNG (Transparent BG)", tag="btn_snap_trans")
        dpg.add_button(label="Save PNG (White BG)", tag="btn_snap_white")
        dpg.add_text("", tag="count")

    panel_height = 540 + 30 * len(neck_tags) + (30 * len(available_modes) if len(available_modes) > 1 else 0) + (30 * 7 if custom_viewer else 0)
    dpg.create_viewport(title="workspace cut", width=380, height=panel_height, resizable=True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    def read_controls():
        cut_origin = (dpg.get_value("px"), dpg.get_value("py"), dpg.get_value("pz"))
        cut_normal = (dpg.get_value("nx"), dpg.get_value("ny"), dpg.get_value("nz"))
        neck = tuple(dpg.get_value(tag) for tag in neck_tags) if show_fov else ()
        hide_blind = bool(dpg.get_value("hide_blind")) if hide_blind_available else False
        geomgroup = tuple(bool(dpg.get_value(f"geomgroup{g}")) for g in range(6)) if custom_viewer else ()
        if len(available_modes) > 1:
            label = dpg.get_value("mode")
            mode_by_label = {mode_labels[m]: m for m in available_modes}
            mode = mode_by_label.get(label, initial_mode)
        else:
            mode = available_modes[0]
        return (bool(dpg.get_value("enable")), cut_origin, cut_normal, neck, hide_blind, mode,
                float(dpg.get_value("alpha")), float(dpg.get_value("sphere_r")), float(dpg.get_value("min_d")),
                bool(dpg.get_value("white_bg")), bool(dpg.get_value("shadows")), bool(dpg.get_value("flat")),
                str(dpg.get_value("geom_type")), geomgroup)

    if custom_viewer:
        import ctypes
        import glfw
        # DearPyGui statically links its own private copy of GLFW (confirmed via `nm -D
        # _dearpygui.so`: glfwMakeContextCurrent is a *defined* symbol, not an import), so pip's
        # `glfw` module has zero visibility into DPG's window/context -- glfw.get_current_context()
        # reads pip's own disjoint bookkeeping, which was never told about DPG's context. DPG's
        # Linux backend (mvViewport_linux.cpp) calls glfwMakeContextCurrent once at viewport setup
        # and never again per-frame, assuming it stays current. Creating a second (pip-glfw) window
        # below and switching to it steals the real GLX current-context away from DPG permanently,
        # so dpg.render_dearpygui_frame() draws into the wrong framebuffer -- DPG panel goes black.
        # Fix: read the real GLX handles via ctypes (bypasses both GLFW copies' private state) and
        # re-select them before every DPG frame.
        libgl = ctypes.CDLL("libGL.so.1")
        libgl.glXGetCurrentDisplay.restype = ctypes.c_void_p
        libgl.glXGetCurrentDrawable.restype = ctypes.c_void_p
        libgl.glXGetCurrentContext.restype = ctypes.c_void_p
        libgl.glXMakeCurrent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        dpg_display = libgl.glXGetCurrentDisplay()
        dpg_drawable = libgl.glXGetCurrentDrawable()
        dpg_context = libgl.glXGetCurrentContext()
        if not glfw.init():
            raise RuntimeError("could not initialize GLFW for custom viewer")
        window = glfw.create_window(1280, 720, f"MuJoCo Custom Viewer — {robot} (maxgeom={max_geom_cap})", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("failed to create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        scene = mujoco.MjvScene(model, max_geom_cap)
        cam = mujoco.MjvCamera()
        opt = mujoco.MjvOption()
        mjr_ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        cam.azimuth = 90.0
        cam.elevation = -45.0
        cam.distance = 2.5
        cam.lookat[:] = base_pos

        button_left = False
        button_middle = False
        button_right = False
        last_x = 0.0
        last_y = 0.0

        def mouse_button_cb(win, button, action, mods):
            nonlocal button_left, button_middle, button_right, last_x, last_y
            if action == glfw.PRESS:
                last_x, last_y = glfw.get_cursor_pos(win)
            if button == glfw.MOUSE_BUTTON_LEFT:
                button_left = (action == glfw.PRESS)
            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                button_middle = (action == glfw.PRESS)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                button_right = (action == glfw.PRESS)

        def cursor_pos_cb(win, xpos, ypos):
            nonlocal last_x, last_y
            dx = xpos - last_x
            dy = ypos - last_y
            last_x, last_y = xpos, ypos

            if not (button_left or button_middle or button_right):
                return

            width, height = glfw.get_window_size(win)
            if height <= 0:
                return

            mod_shift = (glfw.get_key(win, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                         glfw.get_key(win, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

            if button_right:
                act = mujoco.mjtMouse.mjMOUSE_ZOOM
            elif button_middle or (button_left and mod_shift):
                act = mujoco.mjtMouse.mjMOUSE_MOVE_H if abs(dx) > abs(dy) else mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif button_left:
                act = mujoco.mjtMouse.mjMOUSE_ROTATE_H if abs(dx) > abs(dy) else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                return

            mujoco.mjv_moveCamera(model, act, dx / height, dy / height, scene, cam)

        def scroll_cb(win, xoffset, yoffset):
            mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

        glfw.set_mouse_button_callback(window, mouse_button_cb)
        glfw.set_cursor_pos_callback(window, cursor_pos_cb)
        glfw.set_scroll_callback(window, scroll_cb)

        scene_geom_cap = max_geom_cap - (fov_segment_count if show_fov else 0)
        prev_control_state = None
        prev_warned_rec_stride = None
        n_to_render_curr = 0
        kept_world_curr = world[:0]
        kept_rgba_curr = np.zeros((0, 4), dtype=np.float32)

        print(f"\nLaunching Custom High-Cap MuJoCo GLFW viewer + cut panel — {pos.shape[0]} reachable voxels, "
              f"capacity={max_geom_cap}, R in [0, {_DEXTERITY_VMAX:g}]. Close window to exit.\n")

        _prof = {"poll": 0.0, "dpg_ctx": 0.0, "dpg_frame": 0.0, "mj_ctx": 0.0, "rebuild": 0.0, "render": 0.0, "swap": 0.0}
        _prof_frames = 0
        _prof_t0 = time.perf_counter()

        try:
            while not glfw.window_should_close(window) and dpg.is_dearpygui_running():
                _t = time.perf_counter()
                glfw.poll_events()
                _t, _prof["poll"] = time.perf_counter(), _prof["poll"] + time.perf_counter() - _t
                libgl.glXMakeCurrent(dpg_display, dpg_drawable, dpg_context)
                _t, _prof["dpg_ctx"] = time.perf_counter(), _prof["dpg_ctx"] + time.perf_counter() - _t
                dpg.render_dearpygui_frame()
                _t, _prof["dpg_frame"] = time.perf_counter(), _prof["dpg_frame"] + time.perf_counter() - _t
                glfw.make_context_current(window)
                _t, _prof["mj_ctx"] = time.perf_counter(), _prof["mj_ctx"] + time.perf_counter() - _t
                cut_enabled, cut_origin, cut_normal, neck, hide_blind, mode, alpha, sphere_r, min_d, white_bg_val, shadows_val, flat_val, geom_type_val, geomgroup_val = read_controls()
                if mode == "instantaneous" and instantaneous_visible_fn is None:
                    mode = "visible"
                if mode == "visible" and visible is None:
                    mode = "reachable"
                control_state = (cut_enabled, cut_origin, cut_normal, neck, hide_blind, mode, alpha, sphere_r, min_d, white_bg_val, shadows_val, flat_val, geom_type_val, geomgroup_val)
                _t = time.perf_counter()
                if control_state != prev_control_state:
                    for _g, _v in enumerate(geomgroup_val):
                        opt.geomgroup[_g] = 1 if _v else 0
                    prev_control_state = control_state
                    normal_hat = torch.as_tensor(cut_normal, dtype=torch.float64)
                    normal_hat = normal_hat / torch.linalg.norm(normal_hat)
                    plane_point = torch.as_tensor(cut_origin, dtype=torch.float64)
                    if cut_enabled:
                        keep_mask = (pos_t @ normal_hat - plane_point @ normal_hat) <= 0.0
                    else:
                        keep_mask = torch.ones(pos.shape[0], dtype=torch.bool)
                    dex_mask = torch.as_tensor(vals, dtype=torch.float32) >= min_d
                    combined_mask = keep_mask & grid_stride_mask & dex_mask
                    kept_indices = torch.nonzero(combined_mask, as_tuple=False).squeeze(1).numpy()
                    if kept_indices.size > scene_geom_cap:
                        rec_stride = max(2, kept_indices.size // scene_geom_cap + 1)
                        if rec_stride != prev_warned_rec_stride:
                            prev_warned_rec_stride = rec_stride
                            print(f"warning: {kept_indices.size} voxels > cap {scene_geom_cap}; rerun with --stride {rec_stride}", file=sys.stderr)
                        overflow_step = -(-kept_indices.size // scene_geom_cap)
                        render_indices = kept_indices[::overflow_step][:scene_geom_cap]
                        n_to_render = render_indices.size
                    else:
                        prev_warned_rec_stride = None
                        render_indices = kept_indices
                        n_to_render = kept_indices.size

                    if geom_type_val == "box":
                        g_type = mujoco.mjtGeom.mjGEOM_BOX
                        size = np.array([sphere_r, sphere_r, sphere_r], dtype=np.float64)
                    else:
                        g_type = mujoco.mjtGeom.mjGEOM_SPHERE
                        size = np.array([sphere_r, 0.0, 0.0], dtype=np.float64)

                    kept_world = world[render_indices]
                    if show_fov:
                        apply_pose_fn(*neck)
                    if mode == "instantaneous":
                        kept_visible = np.asarray(
                            instantaneous_visible_fn(kept_world, vals[render_indices], model, data),
                            dtype=bool,
                        )
                        if hide_blind:
                            kept_world = kept_world[kept_visible]
                            kept_rgba = _greens_reds_rgba(
                                vals[render_indices][kept_visible], kept_visible[kept_visible], alpha=alpha,
                                vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                            n_to_render = int(kept_visible.sum())
                        else:
                            kept_rgba = _greens_reds_rgba(vals[render_indices], kept_visible, alpha=alpha,
                                                         vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                    elif mode == "visible":
                        keep = vis_mask[render_indices] if hide_blind else None
                        if hide_blind:
                            kept_world = kept_world[keep]
                            kept_rgba = _greens_reds_rgba(
                                vals[render_indices][keep], vis_mask[render_indices][keep], alpha=alpha,
                                vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                            n_to_render = int(keep.sum())
                        else:
                            kept_rgba = _greens_reds_rgba(vals[render_indices], vis_mask[render_indices], alpha=alpha,
                                                         vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                    else:  # reachable
                        dex_norm = (torch.as_tensor(vals[render_indices], dtype=torch.float32) / _DEXTERITY_VMAX).clamp(0.0, 1.0)
                        cmap_t = (_CMAP_T_MIN + (_CMAP_T_MAX - _CMAP_T_MIN) * dex_norm ** _VIEW_GAMMA).clamp(0.0, 1.0)
                        cmap_idx = (cmap_t * 255.0).long().clamp(0, 255)
                        kept_rgba = _CMAP_LUT[vis_cmap][cmap_idx].numpy()
                        kept_rgba[:, 3] = alpha
                    n_to_render_curr = n_to_render
                    kept_world_curr = kept_world
                    kept_rgba_curr = kept_rgba

                    dpg.set_value("count", f"shown {n_to_render} / {kept_indices.size} kept  ({pos.shape[0]} total, mode={mode})")

                    # Rebuild scene with robot + voxel geoms. Only needed when control_state
                    # changes -- robot pose/voxel set is otherwise static, so re-running this
                    # (up to max_geom_cap mjv_initGeom Python calls) every idle/camera-only frame
                    # was the main perf bottleneck.
                    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    robot_geoms_count = scene.ngeom
                    for render_slot in range(n_to_render_curr):
                        if robot_geoms_count + render_slot >= scene.maxgeom:
                            break
                        geom = scene.geoms[robot_geoms_count + render_slot]
                        mujoco.mjv_initGeom(geom, g_type, size, kept_world_curr[render_slot], eye, kept_rgba_curr[render_slot])
                        if flat_val:
                            geom.emission = 0.4
                            geom.specular = 0.0
                        else:
                            geom.emission = 0.0
                    scene.ngeom = robot_geoms_count + min(n_to_render_curr, scene.maxgeom - robot_geoms_count)
                    scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1 if white_bg_val else 0
                    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows_val else 0

                    if show_fov and draw_fov_fn is not None:
                        draw_fov_fn(model, data, scene)
                _t, _prof["rebuild"] = time.perf_counter(), _prof["rebuild"] + time.perf_counter() - _t

                # mjv_updateScene (above) also refreshes scene.camera from cam, but is gated behind
                # control_state and cam is NOT part of it (mouse-drag rotation doesn't touch cut/mode
                # sliders) -- so mouse drag would never move the view. mjv_updateCamera is the cheap
                # camera-only counterpart; run it unconditionally so drag/zoom stay live every frame
                # without paying for the full geom rebuild.
                mujoco.mjv_updateCamera(model, data, cam, scene)

                # Snapshot export checks
                if dpg.is_item_clicked("btn_snap_trans") or dpg.is_item_clicked("btn_snap_white"):
                    is_trans = dpg.is_item_clicked("btn_snap_trans")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    kind = "transparent" if is_trans else "white"
                    out_path = _RESULT_DIR / f"reachability_view_{robot}_{ts}_{kind}.png"
                    _export_viewer_snapshot(
                        model, data, cam, n_to_render_curr, kept_world_curr, kept_rgba_curr, sphere_r,
                        out_path, transparent=is_trans, shadows=shadows_val, flat=flat_val, geom_type=geom_type_val, draw_fov_fn=draw_fov_fn if show_fov else None
                    )
                    dpg.set_value("count", f"Saved {out_path.name}")

                width, height = glfw.get_framebuffer_size(window)
                viewport = mujoco.MjrRect(0, 0, width, height)
                _t = time.perf_counter()
                mujoco.mjr_render(viewport, scene, mjr_ctx)
                _t, _prof["render"] = time.perf_counter(), _prof["render"] + time.perf_counter() - _t
                glfw.swap_buffers(window)
                _prof["swap"] += time.perf_counter() - _t

                _prof_frames += 1
                if time.perf_counter() - _prof_t0 > 1.0:
                    fps = _prof_frames / (time.perf_counter() - _prof_t0)
                    parts = " ".join(f"{k}={1000 * v / _prof_frames:.2f}ms" for k, v in _prof.items())
                    print(f"[perf] fps={fps:.1f} {parts}", file=sys.stderr)
                    _prof = {k: 0.0 for k in _prof}
                    _prof_frames = 0
                    _prof_t0 = time.perf_counter()
        finally:
            dpg.destroy_context()
            glfw.destroy_window(window)
            glfw.terminate()
        return

    print(f"\nLaunching MuJoCo viewer + cut panel — {pos.shape[0]} reachable voxels, "
          f"R in [0, {_DEXTERITY_VMAX:g}] (RdYlGn). Close a window or Ctrl+C to exit.\n")
    with mjviewer.launch_passive(model, data) as viewer:
        scene_geom_cap = viewer.user_scn.maxgeom - (fov_segment_count if show_fov else 0)
        prev_control_state = None
        prev_warned_rec_stride = None
        n_to_render_curr = 0
        kept_world_curr = world[:0]
        kept_rgba_curr = np.zeros((0, 4), dtype=np.float32)

        try:
            while viewer.is_running() and dpg.is_dearpygui_running():
                dpg.render_dearpygui_frame()
                cut_enabled, cut_origin, cut_normal, neck, hide_blind, mode, alpha, sphere_r, min_d, white_bg_val, shadows_val, flat_val, geom_type_val, _ = read_controls()
                if mode == "instantaneous" and instantaneous_visible_fn is None:
                    mode = "visible"
                if mode == "visible" and visible is None:
                    mode = "reachable"
                control_state = (cut_enabled, cut_origin, cut_normal, neck, hide_blind, mode, alpha, sphere_r, min_d, white_bg_val, shadows_val, flat_val, geom_type_val)
                if control_state != prev_control_state:
                    prev_control_state = control_state
                    normal_hat = torch.as_tensor(cut_normal, dtype=torch.float64)
                    normal_hat = normal_hat / torch.linalg.norm(normal_hat)
                    plane_point = torch.as_tensor(cut_origin, dtype=torch.float64)
                    if cut_enabled:
                        keep_mask = (pos_t @ normal_hat - plane_point @ normal_hat) <= 0.0
                    else:
                        keep_mask = torch.ones(pos.shape[0], dtype=torch.bool)
                    dex_mask = torch.as_tensor(vals, dtype=torch.float32) >= min_d
                    combined_mask = keep_mask & grid_stride_mask & dex_mask
                    kept_indices = torch.nonzero(combined_mask, as_tuple=False).squeeze(1).numpy()
                    if kept_indices.size > scene_geom_cap:
                        rec_stride = max(2, kept_indices.size // scene_geom_cap + 1)
                        if rec_stride != prev_warned_rec_stride:
                            prev_warned_rec_stride = rec_stride
                            print(f"warning: {kept_indices.size} voxels > cap {scene_geom_cap}; rerun with --stride {rec_stride}", file=sys.stderr)
                        overflow_step = -(-kept_indices.size // scene_geom_cap)
                        render_indices = kept_indices[::overflow_step][:scene_geom_cap]
                        n_to_render = render_indices.size
                    else:
                        prev_warned_rec_stride = None
                        render_indices = kept_indices
                        n_to_render = kept_indices.size

                    if geom_type_val == "box":
                        g_type = mujoco.mjtGeom.mjGEOM_BOX
                        size = np.array([sphere_r, sphere_r, sphere_r], dtype=np.float64)
                    else:
                        g_type = mujoco.mjtGeom.mjGEOM_SPHERE
                        size = np.array([sphere_r, 0.0, 0.0], dtype=np.float64)

                    user_scn = viewer.user_scn
                    geoms = user_scn.geoms
                    kept_world = world[render_indices]
                    if show_fov:
                        apply_pose_fn(*neck)
                    if mode == "instantaneous":
                        kept_visible = np.asarray(
                            instantaneous_visible_fn(kept_world, vals[render_indices], model, data),
                            dtype=bool,
                        )
                        if hide_blind:
                            kept_world = kept_world[kept_visible]
                            kept_rgba = _greens_reds_rgba(
                                vals[render_indices][kept_visible], kept_visible[kept_visible], alpha=alpha,
                                vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                            n_to_render = int(kept_visible.sum())
                        else:
                            kept_rgba = _greens_reds_rgba(vals[render_indices], kept_visible, alpha=alpha,
                                                         vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                    elif mode == "visible":
                        keep = vis_mask[render_indices] if hide_blind else None
                        if hide_blind:
                            kept_world = kept_world[keep]
                            kept_rgba = _greens_reds_rgba(
                                vals[render_indices][keep], vis_mask[render_indices][keep], alpha=alpha,
                                vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                            n_to_render = int(keep.sum())
                        else:
                            kept_rgba = _greens_reds_rgba(vals[render_indices], vis_mask[render_indices], alpha=alpha,
                                                         vis_cmap=vis_cmap, blind_cmap=blind_cmap)
                    else:  # reachable
                        dex_norm = (torch.as_tensor(vals[render_indices], dtype=torch.float32) / _DEXTERITY_VMAX).clamp(0.0, 1.0)
                        cmap_t = (_CMAP_T_MIN + (_CMAP_T_MAX - _CMAP_T_MIN) * dex_norm ** _VIEW_GAMMA).clamp(0.0, 1.0)
                        cmap_idx = (cmap_t * 255.0).long().clamp(0, 255)
                        kept_rgba = _CMAP_LUT[vis_cmap][cmap_idx].numpy()
                        kept_rgba[:, 3] = alpha
                    n_to_render_curr = n_to_render
                    kept_world_curr = kept_world
                    kept_rgba_curr = kept_rgba
                    for render_slot in range(n_to_render):
                        geom = geoms[render_slot]
                        mujoco.mjv_initGeom(geom, g_type, size, kept_world[render_slot], eye, kept_rgba[render_slot])
                        if flat_val:
                            geom.emission = 0.4
                            geom.specular = 0.0
                        else:
                            geom.emission = 0.0
                    user_scn.ngeom = n_to_render
                    user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1 if white_bg_val else 0
                    user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows_val else 0

                    if show_fov and draw_fov_fn is not None:
                        draw_fov_fn(model, data, user_scn)
                    dpg.set_value("count", f"shown {n_to_render} / {kept_indices.size} kept  ({pos.shape[0]} total, mode={mode})")

                # Snapshot export checks
                if dpg.is_item_clicked("btn_snap_trans") or dpg.is_item_clicked("btn_snap_white"):
                    is_trans = dpg.is_item_clicked("btn_snap_trans")
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    kind = "transparent" if is_trans else "white"
                    out_path = _RESULT_DIR / f"reachability_view_{robot}_{ts}_{kind}.png"
                    _export_viewer_snapshot(
                        model, data, viewer.cam, n_to_render_curr, kept_world_curr, kept_rgba_curr, sphere_r,
                        out_path, transparent=is_trans, shadows=shadows_val, flat=flat_val, geom_type=geom_type_val, draw_fov_fn=draw_fov_fn if show_fov else None
                    )
                    dpg.set_value("count", f"Saved {out_path.name}")

                viewer.sync()
                time.sleep(1 / 30)
        finally:
            dpg.destroy_context()


def _grid_stride_mask(points: np.ndarray, stride: int) -> np.ndarray:
    """Keep voxels whose index on every regular-grid axis is a multiple of `stride`.

    Downsamples a dense n_grid^3 volume to (n_grid/stride)^3 for a lighter live view.
    Per-axis rank comes from the sorted unique coordinate, so it needs no payload grid
    internals and works on the already-aggregated voxel centers.
    """
    mask = np.ones(points.shape[0], dtype=bool)
    for ax in range(3):
        c = np.round(points[:, ax], 5)
        u = np.unique(c)
        idx = np.searchsorted(u, c)
        mask &= (idx % stride == 0)
    return mask


def _plot_scatter(points: np.ndarray, values: np.ndarray, title: str, label: str, path: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    finite = np.isfinite(values)
    if finite.any():
        shown = finite
        colors = values[shown]
        vmin = 0.0 if np.nanmin(colors) >= 0.0 else float(np.nanmin(colors))
        vmax = float(np.nanmax(colors))
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-6
    else:
        shown = np.ones(values.shape[0], dtype=bool)
        colors = np.zeros(values.shape[0], dtype=np.float32)
        vmin, vmax = 0.0, 1.0

    sc = ax.scatter(
        points[shown, 0],
        points[shown, 1],
        points[shown, 2],
        c=colors,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        s=22,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_xlabel("X fwd (m)")
    ax.set_ylabel("Y left (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.65, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cell_edges(coords: np.ndarray) -> np.ndarray:
    """Return cell boundaries around sorted, regularly-spaced voxel centers."""
    if coords.size < 2:
        raise ValueError("workspace section needs at least two grid coordinates per axis")
    step = np.diff(coords)
    if not np.allclose(step, step[0], rtol=1e-4, atol=1e-6):
        raise ValueError("workspace section plot requires a regular voxel grid")
    half_step = 0.5 * step[0]
    return np.concatenate(([coords[0] - half_step], coords[:-1] + half_step, [coords[-1] + half_step]))


def _plot_sections(
    points: np.ndarray,
    values: np.ndarray,
    reached: np.ndarray,
    cutoff_z: float,
    path_base: Path,
    robot: str = "humanoid_v21",
    vis_cmap: str = "Greens",
) -> None:
    """Write paper-ready orthographic and perspective views of axis-aligned cutaways.

    XY and D retain `z <= cutoff_z`; YZ retains `x <= 0`; XZ retains `y >= 0`. Retained
    voxels project by maximum reachability index along camera depth, so high-D interior is visible.
    Fixed scale keeps colors comparable while each cut follows that camera's depth axis.
    """
    import mujoco


    axis_coords = tuple(np.unique(points[:, axis]) for axis in range(3))
    cmap = _shaded_cmap(vis_cmap)
    cmap.set_bad("white")
    panels = (
        # One row: top / front / side. Camera directions make screen-right/screen-up match the
        # labelled axes exactly; each panel keeps its own axis labels (no shared-edge CAD stacking).
        ("(a) Top view", 1, 0, "Y left (m)", "X forward (m)", 180, -90, "lower"),
        ("(b) Front view", 1, 2, "Y left (m)", "Z up (m)", 180, 0, "upper"),
        ("(c) Side view", 0, 2, "X forward (m)", "Z up (m)", 90, 0, "upper"),
    )

    model, home_pos, overlay_pose, visual_mask_fn = _overlay_assets(robot)
    visual = visual_mask_fn(model)
    # The welded model also carries collision proxies and camera frusta.  They do not
    # belong in a paper figure and can otherwise occlude its robot silhouette.
    model.geom_rgba[~visual, 3] = 0.0
    data = mujoco.MjData(model)
    base_pos = set_home_pose(model, data, pose_overrides=overlay_pose, home_pos=home_pos)
    mujoco.mj_forward(model, data)
    xmat = data.geom_xmat[visual].reshape(-1, 3, 3)
    half_extent = np.einsum("nij,nj->ni", np.abs(xmat), model.geom_size[visual])
    robot_lo = np.min(data.geom_xpos[visual] - half_extent - base_pos, axis=0)
    robot_hi = np.max(data.geom_xpos[visual] + half_extent - base_pos, axis=0)
    # geom_size is a conservative mesh bound. Use actual visual-mesh vertices for the sole, so
    # display Z=0 aligns with the lowest rendered foot point rather than below it.
    mesh_z_lo = []
    for gid in np.flatnonzero(visual & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
        mid = int(model.geom_dataid[gid])
        start = int(model.mesh_vertadr[mid]); count = int(model.mesh_vertnum[mid])
        verts = model.mesh_vert[start:start + count]
        mesh_z_lo.append(np.min(verts @ data.geom_xmat[gid].reshape(3, 3)[2] + data.geom_xpos[gid, 2]))
    if mesh_z_lo:
        robot_lo[2] = min(mesh_z_lo) - base_pos[2]
    # Crop the displayed extent (and panel size, below) to where the robot is actually REACHABLE,
    # not the full probed grid: `points`/`axis_coords` include every probed voxel (reached or not),
    # so a generously-padded probe grid (e.g. ToddlerBot's) would otherwise inflate panel size with
    # blank margin. G1/V2 payloads happen to have a tight probe pad already (reach fills to the
    # grid edge), so this crop is a no-op for them; matches their look for any payload.
    reached_coords = tuple(np.unique(points[reached, axis]) for axis in range(3))
    axis_bounds = tuple(
        (min(_cell_edges(coords)[0], robot_lo[axis]), max(_cell_edges(coords)[-1], robot_hi[axis]))
        for axis, coords in enumerate(reached_coords)
    )
    x_span = axis_bounds[0][1] - axis_bounds[0][0]
    y_span = axis_bounds[1][1] - axis_bounds[1][0]
    z_span = axis_bounds[2][1] - axis_bounds[2][0]

    # Floor-reference the vertical axis: shift displayed Z so the foot sole sits at Z=0 (height
    # above ground reads more naturally than the root frame). Display-only — the payload Z, the
    # mujoco render, and the `--section` cut all stay in the base/root frame; only Z tick values
    # and Z axis limits are offset by this amount for the front/side panels.
    z_offset = -float(robot_lo[2])

    cut_masks = (
        reached & (points[:, 2] <= cutoff_z),
        reached & (points[:, 0] <= 0.0),
        reached & (points[:, 1] >= 0.0),
    )
    view_center = base_pos + 0.5 * np.array([sum(bounds) for bounds in axis_bounds])

    def project_max(mask: np.ndarray, horizontal_axis: int, vertical_axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project retained voxels by max reachability index along camera depth.

        Max answers whether any voxel on a camera ray offers high reachability index. Mean would
        dilute that reachable interior; frontmost would recreate opaque-sphere occlusion.
        """
        horizontal = axis_coords[horizontal_axis]
        vertical = axis_coords[vertical_axis]
        selected = mask & np.isfinite(values)
        h_idx = np.searchsorted(horizontal, points[selected, horizontal_axis])
        v_idx = np.searchsorted(vertical, points[selected, vertical_axis])
        flat = np.full(horizontal.size * vertical.size, -np.inf, dtype=np.float32)
        np.maximum.at(flat, v_idx * horizontal.size + h_idx, values[selected])
        grid = flat.reshape(vertical.size, horizontal.size)
        grid[~np.isfinite(grid)] = np.nan
        return _cell_edges(horizontal), _cell_edges(vertical), grid

    cut_grids = tuple(
        project_max(mask, horizontal_axis, vertical_axis)
        for mask, (_, horizontal_axis, vertical_axis, *_) in zip(cut_masks, panels)
    )

    with plt.rc_context({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10}):
        # Single row: top / front / side, all at the same inches-per-metre scale so every panel
        # shares metres-per-inch. Panels are bottom-aligned; heights follow each view's vertical
        # span (top=x, front/side=z). Vertical colorbar to the right, spanning the full row height.
        inch_per_m = 2.15
        gap, left, bottom, top, right = 0.55, 0.95, 0.6, 0.52, 0.45
        cb_gap, cb_w = 0.35, 0.25
        panel_w = (inch_per_m * y_span, inch_per_m * y_span, inch_per_m * x_span)
        panel_h = (inch_per_m * x_span, inch_per_m * z_span, inch_per_m * z_span)
        max_h = max(panel_h)
        x0 = (left, left + panel_w[0] + gap, left + panel_w[0] + gap + panel_w[1] + gap)
        fig_w = x0[2] + panel_w[2] + cb_gap + cb_w + right
        fig_h = bottom + max_h + top
        fig = plt.figure(figsize=(fig_w, fig_h))
        # Top-align panels (heights differ: top=x span < front/side=z span).
        axes = tuple(
            fig.add_axes((x0[i] / fig_w, (bottom + max_h - panel_h[i]) / fig_h,
                          panel_w[i] / fig_w, panel_h[i] / fig_h))
            for i in range(3)
        )
        colorbar_ax = fig.add_axes(
            ((x0[2] + panel_w[2] + cb_gap) / fig_w, bottom / fig_h, cb_w / fig_w, max_h / fig_h)
        )

        mesh = None
        for ax, (name, horizontal_axis, vertical_axis, xlabel, ylabel, azimuth, elevation, origin), (h_edges, v_edges, grid) in zip(axes, panels, cut_grids):
            h_bounds = axis_bounds[horizontal_axis]
            v_bounds = axis_bounds[vertical_axis]
            h_span, v_span = h_bounds[1] - h_bounds[0], v_bounds[1] - v_bounds[0]
            # Floor-reference display: offset whichever plotted axis is Z so the foot is at Z=0
            # (spans unchanged; the mujoco render stays in base frame and is placed via the shifted
            # extent, so robot + field move together).
            h_shift = z_offset if horizontal_axis == 2 else 0.0
            v_shift = z_offset if vertical_axis == 2 else 0.0
            h_bounds = (h_bounds[0] + h_shift, h_bounds[1] + h_shift)
            v_bounds = (v_bounds[0] + v_shift, v_bounds[1] + v_shift)
            h_edges = h_edges + h_shift
            v_edges = v_edges + v_shift
            mesh = ax.pcolormesh(
                h_edges, v_edges, grid, cmap=cmap, vmin=0.0, vmax=_DEXTERITY_VMAX,
                shading="flat", rasterized=True,
            )
            hres = 1000
            wres = max(2, round(hres * h_span / v_span))
            model.vis.global_.orthographic = 1
            model.vis.global_.fovy = v_span
            model.vis.global_.offwidth = wres
            model.vis.global_.offheight = hres
            renderer = mujoco.Renderer(model, hres, wres)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.lookat[:] = view_center
            camera.distance = 5.0
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.orthographic = 1
            # Default-group sites (e.g. the head camera's unsized sensor sites) render as stray
            # dots otherwise; see `_export_viewer_snapshot`'s scene_option for the same fix.
            site_off = mujoco.MjvOption(); site_off.sitegroup[:] = 0
            renderer.update_scene(data, camera, scene_option=site_off)
            rgb = renderer.render()
            renderer.enable_depth_rendering()
            depth = renderer.render()
            renderer.disable_depth_rendering()
            renderer.close()
            alpha = np.where(depth >= depth.max() - 1e-6, 0, 230).astype(np.uint8)
            ax.imshow(np.dstack((rgb, alpha)), extent=(*h_bounds, *v_bounds), origin=origin, zorder=3)
            ax.set_title(name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect("equal")
            ax.set_xlim(h_bounds)
            ax.set_ylim(v_bounds)
            ax.tick_params(direction="out", length=3)

        assert mesh is not None

        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0.0, _DEXTERITY_VMAX), cmap=cmap),
            cax=colorbar_ax,
            orientation="vertical",
        )
        colorbar.set_label("Reachability index")
        fig.savefig(path_base.with_suffix(".png"), dpi=350)
        fig.savefig(path_base.with_suffix(".pdf"))
        plt.close(fig)
    print(f"saved {path_base.with_suffix('.png')} and {path_base.with_suffix('.pdf')}")


def _los_clear(wm, wd, rc, origin: "torch.Tensor", target: "torch.Tensor", tdist: "torch.Tensor",
               eye_body: int, dev, wp_dev) -> "torch.Tensor":
    """Batched line-of-sight: True where nothing occludes the segment origin->target.

    Shared by both robots' reach-visible raycasts (V2 dual gimbal, G1 single fixed cam). Raycasts
    only the group-2 VISUAL MESHES (the true body surface) via the BVH render context; excludes the
    camera's own body per ray (an in-cone target is always in front of the lens). `-1` dist = no hit;
    a hit at/after the target distance is clear (occluder is behind the target).
    """
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src.types import vec6

    N = origin.shape[0]
    GROUP_MESH = vec6(0, 0, 1, 0, 0, 0)  # raycast the group-2 visual meshes only (the true surface)
    EPS = 1e-3
    vecw = (target - origin) / tdist.unsqueeze(1)
    wp_pnt = wp.from_torch(origin.contiguous().view(1, N, 3), dtype=wp.vec3f)
    wp_vec = wp.from_torch(vecw.contiguous().view(1, N, 3), dtype=wp.vec3f)
    be = wp.from_torch(torch.full((N,), eye_body, dtype=torch.int32, device=dev))
    dist = wp.empty((1, N), dtype=wp.float32, device=wp_dev)
    gid = wp.empty((1, N), dtype=wp.int32, device=wp_dev)
    nrm = wp.empty((1, N), dtype=wp.vec3f, device=wp_dev)
    mjw.rays(wm, wd, wp_pnt, wp_vec, GROUP_MESH, True, be, dist, gid, nrm, rc)
    dd = wp.to_torch(dist).view(N)
    return (dd < 0) | (dd >= tdist - EPS)


def _camera_visibility(points: np.ndarray, robot: str = "humanoid_v21",
                       pairs: np.ndarray | None = None,
                       fixed_tilt_rad: float | None = None) -> tuple[np.ndarray, ...]:
    """Per-point visibility of the reachable cloud by the head camera(s).

    Dispatches by robot: humanoid_v21 has the actuated dual back-to-back gimbal (fixed baseline +
    steerable expansion, below); humanoid_v21_single has the SAME hardware/policy but a single
    centered, forward-facing eye (`head_camera="actuated_single"`, joints `cam_yaw`/`cam_pitch`,
    site `cam_rgb` -- no `_left`/`_right` suffix, one iteration of the eye loop below instead of
    two); G1 ships a single FIXED builtin head camera (`_camera_visibility_g1`, no steering ->
    vis_steer == vis_fixed). All return `(vis_steer, vis_fixed, inst_steer, inst_fixed)` bool
    `(N,)`. NOTE: unlike the dual module, humanoid_v21_single's `vis_fixed`/`inst_fixed` has NO
    rear coverage -- the dual's un-actuated baseline still sees front+rear from its back-to-back
    mount, single only ever looks forward. This is an expected physical asymmetry from having one
    eye instead of two, not a bug.

    Actuated dual head cameras (humanoid_v21) [identical eye-by-eye math for humanoid_v21_single]:

    `points` are voxel centers in the cuRobo base frame `(N, 3)`. base_link sits at the world
    origin at zero qpos (asserted), so base frame == world frame and these are the ray targets
    directly. Returns `(vis_steer, vis_fixed, inst_steer, inst_fixed)` bool `(N,)`, each OR-ed
    over the eye(s):

    * `vis_fixed` -- seen by the fixed baseline: the two eyes back-to-back (left forward, right
      rearward) but pitched DOWN by `_BASELINE_TILT_RAD` (0.8308 rad, matching G1, rather than dead
      horizontal -- it aims the cone at the below-eye-height manipulation workspace): in range
      `[NEAR, FAR]` AND inside the FOV cone at that tilt AND line-of-sight clear.
    * `vis_steer` -- seen when the eye AIMS at the point (closed-form pan/tilt, refined from the
      actual aimed lens position, clamped to ROM +-270 deg yaw / +-90 deg pitch): in range AND
      inside the FOV cone AT THE AIMED POSE AND line-of-sight clear. The gimbal moves both the rgb
      center and the optical axis, so the cone is evaluated at the aimed pose, not assumed centered;
      a point the gimbal cannot bring within the FOV half-angle fails. `vis_fixed` subset `vis_steer`.
    * `inst_fixed`/`inst_steer` -- the SAME fixed/aimed poses, but range+FOV-cone ONLY, no LOS
      raycast (matches this module's other `instantaneous` concept -- `_camera_visibility_
      instantaneous_v2`, the live-viewer FOV-only mode -- but reported here as a batch number
      instead of read off viewer sliders). An upper bound over `vis_fixed`/`vis_steer`: what the
      lens geometrically could see if the robot's own body never occluded it.

    Design decisions:
      - Body held at ZERO qpos (arms down), the most un-occluded canonical pose (per user). The
        drawn body IS the occluder.
      - Occluder = the group-2 VISUAL MESHES (the true body surface), raycast with a BVH
        (`create_render_context`) so the ~2.9 M triangles are traversed in log time (~0.6 ms for
        294 k rays). Collision capsules were rejected: they under-cover the torso/head in places,
        letting rays slip past and marking clearly-occluded points visible (user-reported); the
        mesh is exact.
      - The optical center is JOINT-DRIVEN: the *_rgb site is offset from the gimbal axes, so it
        both translates and rotates as the eye aims. An explicit 2-DOF serial FK cam_pose(qy, qp)
        (inner pitch then outer yaw about the zero-config joint axes/anchors) reproduces the site
        pose to 1e-15 vs mj_forward, avoiding a per-point mj_forward.
      - Everything on GPU; visibility is 4 batched `mujoco_warp` raycasts (fixed+steer x 2 eyes),
        one put_model/put_data.

    Only the group-2 meshes are raycast (excludes FOV frustums and collision proxies); the eye's
    own origin body is excluded per ray so the ray does not self-hit its own housing at the lens
    (an aimed/in-cone target is always in front of the lens, so the housing never occludes a
    genuine target).

    PAIRWISE (`pairs` given, consumed by `_pairwise_vrw`). Two ways a pair can be covered at one
    instant, OR-ed: distinct eyes (one aims at each target; exact, decided from the per-eye masks by
    the counting identity in `_pairwise_vrw`), and ONE eye holding both, which is the branch below
    and the ENTIRE result at K=1.

    That single-eye branch aims at the angular BISECTOR of the two target directions. The bisector
    minimizes the larger of the two angles, so it is the optimal aim for a CIRCULAR cone -- but this
    FOV is a 45 x 32.5 deg RECTANGLE and the gimbal has no roll, so vertical spread cannot be traded
    into the wider horizontal budget. The test is therefore SOUND but INCOMPLETE:

      - Sound: a True is a verified witness. Cone membership and line of sight are re-evaluated at
        the actual clamped aimed pose (`cpp`), never at the idealized aim.
      - Incomplete: measured against a dense brute-force aim grid over 20,000 random direction
        pairs, it misses 473 of 8,994 coverable pairs (5.26%), every one at wide separation
        (65.5-96.6 deg, median 79.6). An az/el-midpoint alternative aim recovers 3 of those 473, so
        there is no cheap closed-form fix; closing the gap needs a search over aims.

    Consequence, since this is deliberately not fixed: eta_2 is a LOWER bound, and materially so
    only for K=1 (K>=2 draws its pairwise coverage from the exact distinct-eye branch). The misses
    concentrate exactly where the paper reports eta_2 for K=1 decaying at wide separation, so the
    bias runs in FAVOR of that conclusion -- state it as a limitation rather than lean on it. Single
    -target `vis_steer`/`vis_fixed`, and hence W_VR = eta * W_R, never touch this branch.
    """
    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw

    wp.init()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_dev = "cuda:0" if torch.cuda.is_available() else "cpu"  # warp needs an indexed cuda id
    if robot in ("g1", "unitree_g1"):
        return _camera_visibility_g1(points, dev, wp_dev)

    from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_spec
    from tasks.camera_terms import H_HALF, V_HALF, FAR

    tilt = _BASELINE_TILT_RAD if fixed_tilt_rad is None else fixed_tilt_rad
    NEAR = _RV_NEAR_M  # RGB visibility near clip (0.1 m), NOT the 0.28 m depth-perception near

    head_camera, eye_names = _HUMANOID_RIGS[robot]
    model = get_spec(head_camera=head_camera, end_effector="welded", hand="parallel_gripper").compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # all joints at qpos0 (zero): neutral body + back-to-back cameras
    assert np.allclose(data.xpos[model.body("base_link").id], 0.0, atol=1e-6), "base_link not at world origin"
    wm = mjw.put_model(model)
    wd = mjw.put_data(model, data, nworld=1)  # copies the forwarded geom poses (occluder state)
    rc = mjw.create_render_context(model)  # per-mesh BVH; enables fast exact mesh raycast

    # Occluder scene for the FIXED baseline, posed at the pose it actually evaluates.
    # `wd` above freezes every camera module at zero gimbal, but the fixed baseline looks from
    # (qy=0, qp=_BASELINE_TILT_RAD) -- so the housings in the raycast scene sat 0.83 rad away from
    # where the lens was, and each eye's own base/yaw links occluded from the wrong place. The
    # published 38% fixed-coverage number was computed against that mis-posed scene.
    # This is exact, not an approximation: the fixed baseline uses ONE gimbal pose for ALL points,
    # so one extra `put_data` fixes it outright. (The STEERED path aims per point -- ~100k distinct
    # poses -- and still uses the frozen scene; that remains the open half of the problem.)
    data_fixed = mujoco.MjData(model)
    for _yj, pitch_joint, _st in eye_names:
        data_fixed.qpos[model.joint(pitch_joint).qposadr] = tilt
    mujoco.mj_forward(model, data_fixed)
    wd_fixed = mjw.put_data(model, data_fixed, nworld=1)

    P = torch.as_tensor(points, dtype=torch.float32, device=dev)  # (N,3), world == base frame
    N = P.shape[0]
    EPS = 1e-3
    yaw_max = np.deg2rad(270.0)
    pit_max = np.deg2rad(90.0)

    def rod(axis: torch.Tensor, angle: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation of rows of `v` (N,3) about unit `axis` (3,) by per-row `angle` (N,)."""
        ct = torch.cos(angle).unsqueeze(1)
        st = torch.sin(angle).unsqueeze(1)
        cr = torch.cross(axis.expand_as(v), v, dim=1)
        return v * ct + cr * st + axis * (v @ axis).unsqueeze(1) * (1.0 - ct)

    vis_fixed = torch.zeros(N, dtype=torch.bool, device=dev)
    vis_steer = torch.zeros(N, dtype=torch.bool, device=dev)
    inst_fixed = torch.zeros(N, dtype=torch.bool, device=dev)  # FOV+range only, no LOS raycast
    inst_steer = torch.zeros(N, dtype=torch.bool, device=dev)
    per_eye: list[torch.Tensor] = []       # (E,) of (N,) steered masks, kept UN-collapsed
    per_eye_pair: list[torch.Tensor] = []  # (E,) of (M,) "one pose covers both targets" masks
    M = 0
    if pairs is not None:
        pairs = torch.as_tensor(np.asarray(pairs), dtype=torch.long, device=dev)
        M = int(pairs.shape[0])

    for yaw_joint, pitch_joint, site_name in eye_names:
        yj = model.joint(yaw_joint)
        pj = model.joint(pitch_joint)
        st = model.site(site_name)
        ay = torch.as_tensor(data.xaxis[yj.id], dtype=torch.float32, device=dev)
        oy = torch.as_tensor(data.xanchor[yj.id], dtype=torch.float32, device=dev)
        ap = torch.as_tensor(data.xaxis[pj.id], dtype=torch.float32, device=dev)
        op = torch.as_tensor(data.xanchor[pj.id], dtype=torch.float32, device=dev)
        R0 = torch.as_tensor(data.site_xmat[st.id].reshape(3, 3), dtype=torch.float32, device=dev)
        c0 = torch.as_tensor(data.site_xpos[st.id], dtype=torch.float32, device=dev)
        x0, y0, z0 = R0[:, 0], R0[:, 1], R0[:, 2]
        # `aim` needs to know which way the optical axis (site +Z) faces at zero gimbal, because the
        # closed-form yaw solution differs by a pi offset between a +X-facing and a -X-facing module.
        # Derive it from the GEOMETRY rather than the eye's name. Measured at zero pose: dual is
        # (left +1, right -1) -- what the old name-keyed table hardcoded -- but triple is
        # (left +1, CENTER -1, right +1), so that table would have mis-signed two of its three
        # modules. Modules are back-to-back about X, so z0 is near-parallel to x_hat.
        zx = float(z0[0])
        assert abs(zx) > 0.9, f"{site_name}: optical axis not X-aligned at zero gimbal (z0.x={zx:.3f})"
        zsign = 1.0 if zx > 0 else -1.0
        eye_body = int(model.site_bodyid[st.id])
        # The closed-form aim below assumes yaw about world +-Z, pitch about world +-Y (verified).
        assert torch.allclose(ay.abs().cpu(), torch.tensor([0.0, 0.0, 1.0]), atol=1e-3)
        assert torch.allclose(ap.abs().cpu(), torch.tensor([0.0, 1.0, 0.0]), atol=1e-3)

        def aim(gdir: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Gimbal (qy, qp) that point the optical axis along unit dirs `gdir` (N,3)."""
            qp = torch.arcsin((-gdir[:, 2]).clamp(-1.0, 1.0))  # pitch sets elevation (yaw keeps +-Z)
            base = torch.atan2(gdir[:, 1], gdir[:, 0])
            if zsign > 0:  # left eye (optical axis +X at zero)
                qy = -base
            else:  # right eye (optical axis -X at zero): pi offset, wrapped to (-pi, pi]
                qy = torch.atan2(torch.sin(np.pi - base), torch.cos(np.pi - base))
            return qy, qp

        def cam_pose(qy: torch.Tensor, qp: torch.Tensor, n: int = N):
            """Joint-driven rgb pose at gimbal (qy, qp): inner pitch then outer yaw about the
            zero-config axes/anchors. Returns (center, x_axis, y_axis, z_axis) in world, all (n,3).
            When the gimbal moves, BOTH the center and the optical axis (FOV cone) move with it.
            `n` is the row count: the single-target path passes one gimbal per POINT (n = N), the
            pairwise path one per PAIR (n = M), so the broadcast width cannot be captured as N."""
            center = oy + rod(ay, qy, op + rod(ap, qp, (c0 - op).expand(n, 3)) - oy)

            def col(u: torch.Tensor) -> torch.Tensor:
                return rod(ay, qy, rod(ap, qp, u.expand(n, 3)))

            return center, col(x0), col(y0), col(z0)

        # ---- fixed baseline: both eyes tilted DOWN by _BASELINE_TILT_RAD (0.8308 rad, matching G1) -- fairer than
        # dead horizontal: the tilt drops each optical axis so the cone
        # covers the manipulation workspace below the ~0.66 m eye height instead of aiming at the
        # far wall. Same scalar tilts both eyes down (the pitch-axis sign flip per eye cancels the
        # optical-axis sign flip).
        qy0 = torch.zeros(N, device=dev)
        qp0 = torch.full((N,), tilt, device=dev)
        cbf, xbf, ybf, zbf = cam_pose(qy0, qp0)
        vbf = P - cbf
        rngf = vbf.norm(dim=1)
        vzf, vxf, vyf = (vbf * zbf).sum(1), (vbf * xbf).sum(1), (vbf * ybf).sum(1)
        conef = (vzf > 0) & (torch.atan2(vxf, vzf).abs() <= H_HALF) & (torch.atan2(vyf, vzf).abs() <= V_HALF)
        candf = (rngf >= NEAR) & (rngf <= FAR) & conef
        inst_fixed |= candf
        # Validate the analytic lens pose against MuJoCo's own FK at the same gimbal state before
        # trusting the co-posed occluder scene: if these disagree, `wd_fixed` and `cbf` describe
        # different configurations and the raycast is worse than the frozen one it replaces.
        mj_cf = data_fixed.site_xpos[st.id]
        assert np.allclose(cbf[0].cpu().numpy(), mj_cf, atol=1e-5), (
            f"{site_name}: analytic fixed lens {cbf[0].cpu().numpy()} != MuJoCo FK {mj_cf}")
        clearf = _los_clear(wm, wd_fixed, rc, cbf, P, rngf.clamp_min(EPS), eye_body, dev, wp_dev)
        vis_fixed |= candf & clearf

        # ---- steerable: aim the optical axis AT each point, then test the aimed FOV cone ----
        # Aim once from the zero center, then REFINE from the actual aimed center: the rgb center
        # moves with the gimbal, so aiming from c0 leaves a small residual offset; one refine step
        # points the axis at the target from the true lens position.
        v0 = P - c0
        qy, qp = aim(v0 / v0.norm(dim=1).clamp_min(EPS).unsqueeze(1))
        c_tmp, *_ = cam_pose(qy.clamp(-yaw_max, yaw_max), qp.clamp(-pit_max, pit_max))
        gc = P - c_tmp
        qy, qp = aim(gc / gc.norm(dim=1).clamp_min(EPS).unsqueeze(1))
        qy = qy.clamp(-yaw_max, yaw_max)  # ROM limits
        qp = qp.clamp(-pit_max, pit_max)
        # Same gate as the fixed case, at the AIMED pose: range AND FOV cone (a point the gimbal
        # cannot center -- ROM-clamped past the FOV half-angle -- fails the cone test) AND LOS.
        cs, xs, ys, zs = cam_pose(qy, qp)
        v2 = P - cs
        dc = v2.norm(dim=1)
        vz2, vx2, vy2 = (v2 * zs).sum(1), (v2 * xs).sum(1), (v2 * ys).sum(1)
        cone2 = (vz2 > 0) & (torch.atan2(vx2, vz2).abs() <= H_HALF) & (torch.atan2(vy2, vz2).abs() <= V_HALF)
        cand2 = (dc >= NEAR) & (dc <= FAR) & cone2
        inst_steer |= cand2
        clear2 = _los_clear(wm, wd, rc, cs, P, dc.clamp_min(EPS), eye_body, dev, wp_dev)
        eye_steer = cand2 & clear2
        vis_steer |= eye_steer
        per_eye.append(eye_steer)

        if pairs is None:
            continue

        # ---- same-camera pair branch: ONE gimbal pose must cover BOTH targets ----
        # Needed because the pairwise definition permits k_i == k_j. The per-eye masks above cannot
        # answer this: they are computed with the eye aimed AT each target separately, and a single
        # camera cannot hold two aims at once. So aim at the angular bisector of the two targets and
        # require both to fall inside that one cone. For K=1 this branch is the ENTIRE result -- the
        # distinct-camera branch is empty -- so the K=1 curve depends on it alone.
        pa, pb = P[pairs[:, 0]], P[pairs[:, 1]]
        da = pa - c0
        db = pb - c0
        bis = da / da.norm(dim=1).clamp_min(EPS).unsqueeze(1) + db / db.norm(dim=1).clamp_min(EPS).unsqueeze(1)
        # Antipodal targets (bisector degenerates to ~0) can never share a <180 deg cone; the
        # normalize below would amplify noise into an arbitrary aim, so mark them unusable.
        bn = bis.norm(dim=1)
        ok_bis = bn > 1e-3
        qyp, qpp = aim(bis / bn.clamp_min(EPS).unsqueeze(1))
        cp_, *_ = cam_pose(qyp.clamp(-yaw_max, yaw_max), qpp.clamp(-pit_max, pit_max), n=M)
        # Refine from the true lens position, as the single-target path does: the lens translates
        # with the gimbal, so the bisector taken at c0 is not the bisector at the aimed center.
        ba = pa - cp_
        bb = pb - cp_
        bis2 = ba / ba.norm(dim=1).clamp_min(EPS).unsqueeze(1) + bb / bb.norm(dim=1).clamp_min(EPS).unsqueeze(1)
        bn2 = bis2.norm(dim=1)
        ok_bis &= bn2 > 1e-3
        qyp, qpp = aim(bis2 / bn2.clamp_min(EPS).unsqueeze(1))
        qyp = qyp.clamp(-yaw_max, yaw_max)
        qpp = qpp.clamp(-pit_max, pit_max)
        cpp, xpp, ypp, zpp = cam_pose(qyp, qpp, n=M)

        def _covers(tgt: torch.Tensor) -> torch.Tensor:
            v = tgt - cpp
            r = v.norm(dim=1)
            vz, vx, vy = (v * zpp).sum(1), (v * xpp).sum(1), (v * ypp).sum(1)
            cone = (vz > 0) & (torch.atan2(vx, vz).abs() <= H_HALF) & (torch.atan2(vy, vz).abs() <= V_HALF)
            cand = (r >= NEAR) & (r <= FAR) & cone
            return cand & _los_clear(wm, wd, rc, cpp, tgt, r.clamp_min(EPS), eye_body, dev, wp_dev)

        per_eye_pair.append(ok_bis & _covers(pa) & _covers(pb))

    vis_steer |= vis_fixed  # numerical guarantee vis_fixed subset of vis_steer
    inst_steer |= inst_fixed
    out = (vis_steer.cpu().numpy(), vis_fixed.cpu().numpy(),
           inst_steer.cpu().numpy(), inst_fixed.cpu().numpy())
    if pairs is None:
        return out
    return out + (torch.stack(per_eye).cpu().numpy(), torch.stack(per_eye_pair).cpu().numpy())


def _pairwise_vrw(
    points: np.ndarray, robot: str, pairs: np.ndarray, return_fixed: bool = False,
    fixed_tilt_rad: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Pairwise visible-reachable mask: can the rig cover BOTH targets of each pair at once?

    Implements the corrected §3 definition, which permits ``k_i == k_j``. Returns `(M,)` bool over
    `pairs` `(M,2)` of indices into `points`. With `return_fixed`, additionally returns the
    LOCKED-GIMBAL counterpart of the same mask, from the same solve.

    The locked-gimbal case needs no aim search: a fixed rig has exactly ONE realizable head state,
    so the definition's existential over head states has a single term and collapses to a
    conjunction of the two single-target fixed masks, `vis_fixed[a] & vis_fixed[b]`. Those masks are
    already computed here for every point (both members of every pair live in `points`), against the
    co-posed `wd_fixed` occluder scene, so the second family costs no extra raycast. The independent
    kernel reduces the same way: `gpu_visibility.score_pairs` on a zero-aim-group adapter builds one
    pseudo-group holding every eye with `aims = ("rest",)`, which zeroes its `distinct` term and
    leaves exactly this conjunction.

    Two disjoint ways a pair can be covered, OR-ed:

    * **Distinct cameras** -- eye `e` aims at `a` while eye `e' != e` aims at `b`. Read straight off
      the per-eye steered masks `A (E,N)`; no extra raycast. Counting trick: the number of ordered
      eye pairs covering `(a,b)` is `n_a * n_b`, of which `|A_.(a) & A_.(b)|` reuse the SAME eye, so
      a distinct pair exists iff `n_a * n_b - overlap > 0`. Empty when `E == 1`.
    * **One camera, one pose** -- a single eye holds both targets in one cone (`_camera_visibility`'s
      bisector branch). This is the whole story at K=1.

    The auxiliary-region variant of the paper (L288): only `x_i` need be reachable, so no
    simultaneous bimanual IK is required -- which matters because the payload does not contain it
    (single-arm solve, opposite arm pinned).
    """
    pairs = np.asarray(pairs)
    assert pairs.ndim == 2 and pairs.shape[1] == 2, f"pairs must be (M,2), got {pairs.shape}"
    _, vis_fixed, _, _, per_eye, per_eye_pair = _camera_visibility(
        points, robot, pairs=pairs, fixed_tilt_rad=fixed_tilt_rad)
    a, b = pairs[:, 0], pairs[:, 1]
    va = per_eye[:, a]            # (E,M) eye sees target a (aimed at a)
    vb = per_eye[:, b]
    n_a = va.sum(0).astype(np.int64)
    n_b = vb.sum(0).astype(np.int64)
    overlap = (va & vb).sum(0).astype(np.int64)
    distinct = (n_a * n_b - overlap) > 0
    steer = distinct | per_eye_pair.any(0)
    if not return_fixed:
        return steer
    return steer, vis_fixed[a] & vis_fixed[b]


def _camera_visibility_instantaneous_v2(
    points: np.ndarray,
    lens_poses: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """V2 instantaneous FOV visibility using the viewer's own lens poses (post-`mj_forward`).

    `lens_poses` is a list of `(site_xpos, site_xmat)` tuples in world frame -- one per eye (left,
    right). The viewer already wrote the slider-driven gimbal qpos and ran `mj_forward`, so these
    poses reflect the live head pose AND any home-pose translation applied to the floating root.
    Each eye is tested against its own cone (range + half-angle); the mask is the OR over eyes.

    No occluder raycast: instantaneous mode is for interactive exploration and the BVH setup per
    slider change is too costly. The static `--view-mode visible` and `--reach-visible-compare`
    paths use the full cone+range+LOS via `_camera_visibility` (offline, GPU raycast); the live
    mode trades that accuracy for sub-second response.

    `points` and the lens poses must share a world frame. The viewer passes `kept_world` (already
    translated by `set_home_pose`'s `base_pos`) plus the viewer's live `data.site_xpos/site_xmat`,
    so both are in the same shifted world frame.
    """
    import torch

    from tasks.camera_terms import H_HALF, V_HALF, FAR

    NEAR = _RV_NEAR_M
    P = np.asarray(points, dtype=np.float64)
    vis = np.zeros(P.shape[0], dtype=bool)
    for site_xpos, site_xmat in lens_poses:
        c = site_xpos
        rot = site_xmat.reshape(3, 3)
        xs, ys, zs = rot[:, 0], rot[:, 1], rot[:, 2]
        v = P - c
        vz = v @ zs; vx = v @ xs; vy = v @ ys
        d = np.linalg.norm(v, axis=1)
        cone = (vz > 0) & (np.abs(np.arctan2(vx, vz)) <= H_HALF) & (np.abs(np.arctan2(vy, vz)) <= V_HALF)
        cand = (d >= NEAR) & (d <= FAR) & cone
        vis |= cand
    return vis


def _camera_visibility_g1(points: np.ndarray, dev, wp_dev) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Visibility of the reachable cloud by G1's SINGLE FIXED builtin head camera.

    G1's real platform fixes its D435/D436-compatible head-camera mount to ``torso_link``, pitched
    0.8308 rad (47.6 deg) down, NOT V2's dual steerable gimbal. So there is no steering expansion: a voxel is
    either inside that one fixed cone (with clear line-of-sight) or blind. Returns ``(vis, vis, inst,
    inst)`` -- ``vis_steer == vis_fixed`` and ``inst_steer == inst_fixed`` -- so the steerable-only
    category is empty and the G1 compare column is just visible (head-cam) vs blind. `inst` is the
    same fixed cone, range+FOV only, no LOS raycast (see `_camera_visibility`'s `inst_fixed` doc).
    This is the honest baseline the paper contrasts against V2's back-to-back steerable coverage.

    Same faithful visibility model as the V2 path: body at the arms-down pose (``_ARMS_DOWN_POSE``,
    forearms clear of the forward cone), group-2 mesh occluder via BVH (``_los_clear``), RGB range
    ``[_RV_NEAR_M, FAR]``, FOV 90x65 half-angles. The G1 payload is in the ``pelvis`` base frame, and
    the pelvis is NOT at the world origin at qpos0 (z~0.79 m), so ``points`` are shifted to world by
    the pelvis translation; base orientation is identity (asserted), so no rotation is needed.
    """
    import mujoco
    import mujoco_warp as mjw

    from mj_envs.asset_zoo.g1.g1_constants import get_spec
    from tasks.camera_terms import H_HALF, V_HALF, FAR

    NEAR = _RV_NEAR_M
    model = get_spec(head_camera="builtin", end_effector="welded", hand="parallel_gripper").compile()
    data = mujoco.MjData(model)
    for jname, val in _ARMS_DOWN_POSE["unitree_g1"].items():  # forearms down, clear of the cone
        data.qpos[model.jnt_qposadr[model.joint(jname).id]] = val
    mujoco.mj_forward(model, data)
    base_id = model.body("pelvis").id
    assert np.allclose(data.xquat[base_id], [1.0, 0.0, 0.0, 0.0], atol=1e-6), "pelvis base not axis-aligned"
    base_off = torch.as_tensor(data.xpos[base_id], dtype=torch.float32, device=dev)

    wm = mjw.put_model(model)
    wd = mjw.put_data(model, data, nworld=1)  # forwarded occluder state (arms-down body)
    rc = mjw.create_render_context(model)  # per-mesh BVH for the fast exact mesh raycast

    P = torch.as_tensor(points, dtype=torch.float32, device=dev) + base_off  # base -> world
    N = P.shape[0]
    EPS = 1e-3

    st = model.site("head_camera_rgb")
    R0 = torch.as_tensor(data.site_xmat[st.id].reshape(3, 3), dtype=torch.float32, device=dev)
    c0 = torch.as_tensor(data.site_xpos[st.id], dtype=torch.float32, device=dev)
    x0, y0, z0 = R0[:, 0], R0[:, 1], R0[:, 2]  # +Z optical axis out of the lens
    eye_body = int(model.site_bodyid[st.id])  # torso_link: excluded per ray (never self-occludes)

    v = P - c0
    rng = v.norm(dim=1)
    vz, vx, vy = (v * z0).sum(1), (v * x0).sum(1), (v * y0).sum(1)
    cone = (vz > 0) & (torch.atan2(vx, vz).abs() <= H_HALF) & (torch.atan2(vy, vz).abs() <= V_HALF)
    cand = (rng >= NEAR) & (rng <= FAR) & cone
    clear = _los_clear(wm, wd, rc, c0.expand(N, 3), P, rng.clamp_min(EPS), eye_body, dev, wp_dev)
    vis = (cand & clear).cpu().numpy()
    inst = cand.cpu().numpy()
    return vis, vis, inst, inst


def _payload_base_height_above_foot(robot: str) -> float:
    """Return payload-base Z above lowest rendered foot vertex for `robot`.

    The main-paper comparison top plane is fixed in the common foot-referenced world frame,
    not at a camera-relative offset.  Payload coordinates use `base_link` (V2) or `pelvis`
    (G1), so derive their conversion from the exact arms-down model and its visual foot mesh.
    """
    import mujoco


    model, home_pos, _overlay_pose, visual_mask_fn = _overlay_assets(robot)
    data = mujoco.MjData(model)
    set_home_pose(model, data, pose_overrides=_ARMS_DOWN_POSE[robot], home_pos=home_pos)
    mujoco.mj_forward(model, data)
    visual = visual_mask_fn(model)
    foot_z = float("inf")
    for gid in np.flatnonzero(visual & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
        mesh_id = int(model.geom_dataid[gid])
        start = int(model.mesh_vertadr[mesh_id])
        verts = model.mesh_vert[start:start + int(model.mesh_vertnum[mesh_id])]
        foot_z = min(foot_z, float(np.min(verts @ data.geom_xmat[gid].reshape(3, 3)[2]
                                         + data.geom_xpos[gid, 2])))
    base_name = "pelvis" if robot == "unitree_g1" else "base_link"
    return float(data.xpos[model.body(base_name).id, 2] - foot_z)


# Fixed-baseline downward tilt: both eyes pitched down by this from horizontal, a fairer baseline
# than dead-horizontal (aims the cone at the below-eye-height manipulation workspace). This is
# exactly G1's official D435-mount pitch, so fixed V2 and G1 use one physical inclination.
_BASELINE_TILT_RAD = 0.8307767239493009
_BASELINE_TILT_DEG = float(np.rad2deg(_BASELINE_TILT_RAD))

# Per-robot shoulder body names: first shoulder joint body (pitch / ab-adduction) L and R.
# The top section cut in the comparison figure slices at each robot's own shoulder-center
# height rather than a shared fixed height, so the most-interesting cross section is shown
# for every morphology.
_SHOULDER_BODIES: dict[str, tuple[str, str]] = {
    "humanoid_v21": ("shoulder_2_L", "shoulder_2_R"),
    "unitree_g1":   ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
    "toddlerbot":   ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
    "booster_t1":   ("AL1", "AR1"),
    "apptronik_apollo": ("l_shoulder_aa_link", "r_shoulder_aa_link"),
    "fourier_gr3": ("left_upper_arm_pitch_link", "right_upper_arm_pitch_link"),
    "pal_talos": ("arm_left_1_link", "arm_right_1_link"),
}


def _shoulder_section_payload_z(robot: str) -> float:
    """Return `robot`'s physical shoulder-center plane in workspace base coordinates.

    The comparison figure's top row slices through each robot's shoulder-pitch center rather
    than a shared fixed height.  This uses source-MJCF FK at the overlay pose to derive the
    world Z of the two shoulder-pitch bodies and returns that height in the payload's base
    frame (world_z - base_pos_z), ready for use as ``top_cutoff_z``.
    """
    import mujoco


    model, home_pos, overlay_pose, _visual_mask_fn = _overlay_assets(robot)
    data = mujoco.MjData(model)
    base_pos = set_home_pose(model, data, pose_overrides=overlay_pose, home_pos=home_pos)
    mujoco.mj_forward(model, data)
    left_body, right_body = _SHOULDER_BODIES[robot]
    shoulder_z = [
        data.xpos[model.body(name).id, 2]
        for name in (left_body, right_body)
    ]
    return float(np.mean(shoulder_z) - base_pos[2])


def _shoulder_foot_referenced_z(robot: str) -> float:
    """Return `robot`'s shoulder-center height referenced to the foot sole (Z=0 in the figure)."""
    return _shoulder_section_payload_z(robot) + _payload_base_height_above_foot(robot)


# `_plot_reach_visible_grid` column-width floor (inches): V2/G1's own x_span happens to produce
# ~2.39 in columns, which is what the fixed 18-pt title/tick sizing was tuned against. Smaller-reach
# robots (ToddlerBot) would otherwise get a much narrower column and start overlapping text.
_MIN_COL_W_IN = 2.4
_COMPARE_VOLUME_MIN_COL_W_IN = 3.05
_COMPARE_TICK_M = 0.25
# Inset (inches) of a sub-panel letter from the top-left corner of its block. X is small rather than
# zero: the letter belongs outside the y label and tick column, at the block's own corner, so a
# reader scanning the left margin finds a,b before any axis furniture. Y is near zero because the
# letter is set `va="top"`, so this is the gap above the glyph, not below it.
_PANEL_X, _PANEL_Y = 0.35, 0.01


def _format_compare_tick(value: float) -> str:
    """Format comparison ticks exactly, without unnecessary trailing zeros."""
    return f"{value:.2f}".rstrip("0").rstrip(".")

# Near clip (m) for the reach-visible RGB analysis. Decoupled from tasks.camera_terms.NEAR (0.28 m,
# the D436 depth IDEAL min-Z used by the perception/detection gate): this figure asks whether the
# RGB camera can SEE the target, and RGB reads closer than the depth ideal range (confirmed: april
# tags detected below 0.3 m). 0.1 m is the shared workspace-visibility near clip.
_RV_NEAR_M = 0.1

def _reach_visible_case(robot_key: str, visibility_override: Path | None = None) -> dict:
    """Load a robot's reach payload (bimanual max over R + mirrored-L) and its camera visibility.

    Factored from main's single-robot reach-visible path so the comparison grid can build a case per
    robot. Returns points/reached/dexterity in the payload base frame plus the voxel volume. Static-
    camera robots (`visibility_kind` steered/fixed: V2/G1) get one binary per-voxel camera label and
    return `vis_steer`/`vis_fixed` over ALL points (caller picks). Dynamic robots (ToddlerBot/
    booster_t1: head aims per IK row, no single per-voxel camera label) return `visible` directly --
    the same per-row aggregation `--view-mode visible` uses (voxel counts visible if >=1 reachable
    orientation finds a legal head aim that sees it), bimanual-mirrored like `dexterity`.
    """
    cfg = _ROBOTS[robot_key]
    payload_path = _CACHE_DIR / cfg.payload
    dynamic_visibility_path = None
    if cfg.visibility_kind == "dynamic":
        # `visibility_override` lets a sensitivity sidecar be plotted without touching the
        # canonical one -- GR-3 declares two head-camera mount pitches (15 deg URDF vs 40 deg
        # official FOV figure) and PAL TALOS has an RGB-vs-depth stream pair; both report both.
        # Every other dynamic robot always uses its canonical sidecar.
        dynamic_visibility_path = (
            (visibility_override or _DYNAMIC_VISIBILITY_SIDECAR[cfg.cli_key])
            if cfg.cli_key in ("fourier_gr3", "pal_talos")
            else _DYNAMIC_VISIBILITY_SIDECAR[cfg.cli_key]
        )
    agg, dynamic, meta = _load_workspace_aggregate(
        payload_path, cfg, need_visible=True, dynamic_visibility_path=dynamic_visibility_path,
    )
    visible_agg = (
        _toddlerbot_visible_field(dynamic, agg, meta["n_grid_per_axis"], mirror_left=True)
        if dynamic is not None else None
    )
    aggL = _mirror_agg_l(agg, meta["n_grid_per_axis"])
    ids = np.union1d(agg["voxel_id"], aggL["voxel_id"])
    DR = np.full(ids.size, np.nan); DL = np.full(ids.size, np.nan)
    pos = np.full((ids.size, 3), np.nan, dtype=np.float32)
    rR = np.searchsorted(ids, agg["voxel_id"]); rL = np.searchsorted(ids, aggL["voxel_id"])
    DR[rR] = np.where(agg["success_voxel"], agg["dexterity"], np.nan)
    DL[rL] = np.where(aggL["success_voxel"], aggL["dexterity"], np.nan)
    pos[rR] = agg["voxel_pos"]; pos[rL] = aggL["voxel_pos"]
    D_bi = np.fmax(DR, DL)
    # `cfg.model_key`, NOT `meta["robot"]`. The latter is the BASE name baked in at generation time,
    # so all three K rigs report "humanoid_v21"; consumers use `robot` to dispatch `_overlay_assets`,
    # which then drew the DUAL head on the K=1 and K=3 panels. Identical for every other robot, whose
    # payload string already equals its model_key. Same trap the visibility dispatch below avoids.
    case = dict(robot=cfg.model_key, points=pos, reached=~np.isnan(D_bi), dexterity=np.nan_to_num(D_bi),
                voxel_vol=(meta["grid_spacing"] or 0.02) ** 3)
    if cfg.visibility_kind == "dynamic":
        if visible_agg is None:
            raise FileNotFoundError(
                f"dynamic visibility sidecar/aggregated cache missing for {cfg.cli_key}: "
                f"{dynamic_visibility_path}"
            )
        visible = np.zeros(ids.size, dtype=bool)
        visible[rR] = visible_agg["success_voxel"]
        visL = np.zeros(ids.size, dtype=bool)
        visL[rL] = visible_agg["success_voxel"]
        visible |= visL
        case["visible"] = visible
    else:
        # `cfg.model_key`, NOT `meta["robot"]`, for the reason given above the `case` dict: dispatching
        # on the payload string would score every K=1 and K=3 case with the DUAL camera's math.
        (case["vis_steer"], case["vis_fixed"],
         case["inst_steer"], case["inst_fixed"]) = _camera_visibility(pos, cfg.model_key)
    return case


def _rv3d_render_worker(io_in: Path, io_out: Path) -> None:
    """Subprocess entry point for `_plot_reach_visible_grid`'s `video_3d_row` -- renders every case's
    3D-panel frame sequence in a FRESH process, then exits.

    Exists because `mujoco.Renderer` create/close cycles are CUMULATIVE within one process in this
    environment: the Top/Side silhouette renders already create-and-close 12 of them (6 cases x 2
    rows), and a standalone repro confirmed the 13th renderer's `.render()` call then segfaults
    natively -- no Python exception, `Fatal Python error: Segmentation fault` inside
    `mujoco.mjr_render`. The same repro with NO prior renderers in the process did not crash on an
    equivalent perspective render, so the ceiling is per-process, not something about this render in
    particular. A fresh subprocess gets its own EGL/GL context budget, so the 3D row's renderers never
    share a process with those 12 (or with each other, across separate `--reach-visible-compare`
    calls in the same `main()` invocation for the plain 2-row video).

    I/O is two `.npz` files rather than a pipe/queue: the per-case point clouds are the only bulky
    payload (thousands of rows after `video_3d_stride`, tiny by video standards), and `MjModel`/
    `MjData` aren't picklable, so this rebuilds them fresh per case via `_overlay_assets` exactly like
    the parent process does, instead of trying to hand them across the process boundary.
    """
    import mujoco


    data = np.load(io_in, allow_pickle=False)
    robots = [str(r) for r in data["robots"]]
    z_boundary_displays = data["z_boundary_displays"]
    y_boundaries = data["y_boundaries"]
    stride = int(data["stride"].item()); tilt = float(data["tilt"].item())
    elevation = float(data["elevation"].item()); alpha = float(data["alpha"].item())
    res = int(data["res"].item())
    vis_cmap = str(data["vis_cmap"].item()); blind_cmap = str(data["blind_cmap"].item())
    view_radius = float(data["view_radius"].item())  # shared across ALL cases -- see the camera block
    n_frames = z_boundary_displays.size

    scene_option = mujoco.MjvOption(); scene_option.sitegroup[:] = 0
    eye = np.eye(3, dtype=np.float64).flatten()
    half_edge = 0.5 * 0.02 * max(stride, 1)  # voxel box half-extent (box edge == grid spacing x stride)
    box = np.array([half_edge] * 3, dtype=np.float64)
    out = {}
    for i, robot in enumerate(robots):
        pts_native = data[f"points_{i}"]
        dex = data[f"dexterity_{i}"]
        vis = data[f"visible_{i}"]
        base_pos = data[f"base_pos_{i}"]
        z_off = float(data[f"z_off_{i}"].item())
        view_center = data[f"view_center_{i}"]

        model, home_pos, _ov, visual_mask_fn = _overlay_assets(robot)
        model.geom_rgba[~visual_mask_fn(model), 3] = 0.0
        d = mujoco.MjData(model)
        set_home_pose(model, d, pose_overrides=_ARMS_DOWN_POSE.get(robot, {}), home_pos=home_pos)
        mujoco.mj_forward(model, d)

        world = pts_native.astype(np.float64) + base_pos
        rgba = _greens_reds_rgba(dex, vis, alpha=alpha, vis_cmap=vis_cmap, blind_cmap=blind_cmap)
        model.vis.global_.orthographic = 0
        model.vis.global_.fovy = 45.0
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, res)
        model.vis.global_.offheight = max(model.vis.global_.offheight, res)
        # Camera frames the SHARED display window the Top/Side rows already use (`view_center_i` /
        # `view_radius`, both computed once by the parent from `disp_bounds`), NOT this case's own
        # voxel bbox, which is what an earlier version fitted. That fit had two defects. It excluded
        # the ROBOT MESH -- `world` is the voxel cloud alone -- so every column's legs and feet were
        # clipped off the bottom of the panel and each robot floated at whatever height its own
        # workspace centroid put it, instead of standing on the common foot-referenced ground line
        # the Side row shows. And it auto-zoomed PER CASE, so a small rig and a large one were drawn
        # the same apparent size and the row could not be read as a size comparison at all.
        #
        # `view_radius` is the display box's bounding-SPHERE radius (half the space diagonal), not
        # half its largest span: this is a perspective camera at an oblique azimuth, so a corner of
        # the box sits further from the lookat point than any face does, and half-max-span left
        # those corners of the workspace outside the frustum -- visibly cut-off voxels at the
        # widest part of the sweep. A sphere bound is azimuth-independent and cannot clip.
        fit_distance = view_radius / np.tan(np.radians(45.0) / 2.0)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = view_center
        cam.distance = fit_distance
        # `_robot_forward_azimuth`, not a fixed value: an EARLIER version of this line used ONE
        # shared azimuth (`tilt` alone) because `_robot_forward_azimuth` put booster_t1/fourier_gr3/
        # pal_talos on the opposite world side from humanoid_v21/unitree_g1/apptronik_apollo,
        # mirroring the Y-cut sweep's screen direction between the two groups. That was chasing a
        # SYMPTOM: those 3 robots' head-camera sites have local +Z pointing INTO the head, the
        # opposite of every other rig's "+Z = optical axis out" (`_HEAD_CAM_Z_INTO_HEAD`, found by
        # comparing `vrw_video_fourier_gr3.mp4` against `vrw_video_v2.mp4` -- v2 showed its camera
        # eyes facing the viewer, GR-3 showed its BACKPACK facing the viewer, at what the
        # uncorrected formula called each one's own "front"). With that sign bug fixed at the
        # source, all 6 robots' true front-facing azimuth measures the SAME (180 deg, verified) --
        # so per-robot `_robot_forward_azimuth` now gives BOTH a consistent viewpoint AND each
        # robot's correct mesh-front, and the two goals were never actually in conflict.
        cam.azimuth = _robot_forward_azimuth(model, d, robot) + tilt
        cam.elevation = elevation

        n = pts_native.shape[0]
        renderer = mujoco.Renderer(model, res, res, max_geom=n + model.ngeom + 100)
        frames = np.empty((n_frames, res, res, 3), np.uint8)
        base = model.ngeom

        # Same box-per-voxel look and SDF sub-voxel interpolation as `_render_vrw_video`, and SAME
        # SEQUENTIAL structure (one axis sweeps at a time, matching its grow-then-erase phases)
        # instead of the two boundaries moving together -- see `_plot_reach_visible_grid`'s
        # `video_3d_row` docstring for why the parent builds the timeline this way (every row bookends
        # at its OWN start state, so the whole video loops with no jump, which a simultaneous
        # intersection could not do per-row). The parent pins whichever boundary ISN'T currently
        # sweeping at its fully-permissive extreme, so within any given frame exactly one axis
        # constrains and the other is trivially full for every voxel -- no per-voxel "which axis
        # binds" logic needed; classify by FRAME instead.
        z_hi_disp = float(z_boundary_displays.max())
        z_native = z_boundary_displays - z_off  # per-frame Z boundary, native (case) frame
        is_y_moving = np.abs(z_native - (z_hi_disp - z_off)) <= _FULL_EPS  # Z pinned full -> Y active

        z_order = np.argsort(np.round(pts_native[:, 2], 6), kind="stable")
        z_sorted, z_world, z_rgba = pts_native[z_order, 2], world[z_order], rgba[z_order]
        z_bot_s, z_top_s = z_sorted - half_edge, z_sorted + half_edge

        y_order = np.argsort(np.round(pts_native[:, 1], 6), kind="stable")
        y_sorted, y_world, y_rgba = pts_native[y_order, 1], world[y_order], rgba[y_order]
        y_bot_s, y_top_s = y_sorted - half_edge, y_sorted + half_edge

        # Z's keep condition (coord<=boundary) makes survivors a PREFIX of ascending-Z order, so
        # direct index==slot already keeps them low & contiguous. Y's condition (coord>=boundary)
        # makes survivors a SUFFIX of ascending-Y order, so slots are reversed to map that suffix
        # into the same low range -- exactly `_render_vrw_video`'s grow-vs-erase slot convention,
        # just chosen by AXIS here since either axis can be run with its boundary moving in either
        # temporal direction (forward sweep, then the mirrored return) and the same fixed mapping
        # is valid both ways (a pure function of which original-sorted voxels currently pass, not of
        # which direction the boundary is currently moving).
        def _z_slot(k: int) -> int:
            return base + k

        def _y_slot(k: int) -> int:
            return base + (n - 1 - k)

        def _init_full(world_s, rgba_s, slot_fn) -> None:
            for k in range(n):
                geom = scn.geoms[slot_fn(k)]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, box, world_s[k], eye, rgba_s[k])
                geom.emission, geom.specular = 0.4, 0.0

        # Slots are shared (same `n`-wide range) between the Z and Y representations, so switching
        # axis clobbers whichever one isn't active -- reinit only fires on an actual axis CHANGE (3
        # times total over the whole sequence: start-hold+Z-forward, then Y-forward+end-hold+
        # Y-reverse, then Z-reverse), not every frame.
        current_axis = None
        for fi in range(n_frames):
            renderer.update_scene(d, camera=cam, scene_option=scene_option)
            scn = renderer.scene
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1  # opaque white, matches _render_vrw_3d
            if is_y_moving[fi]:
                if current_axis != "y":
                    _init_full(y_world, y_rgba, _y_slot); current_axis = "y"
                b = y_boundaries[fi]
                n_empty = min(int(np.searchsorted(y_top_s, b, side="right")), n)
                n_full = min(int(np.searchsorted(y_bot_s, b, side="left")), n)
                for k in range(n_empty, n_full):
                    frac = np.clip((y_top_s[k] - b) / (2.0 * half_edge), 0.0, 1.0)
                    geom = scn.geoms[_y_slot(k)]
                    geom.size[1] = half_edge * frac
                    geom.pos[1] = y_world[k, 1] + half_edge * (1.0 - frac)
                scn.ngeom = base + (n - n_empty)
            else:
                if current_axis != "z":
                    _init_full(z_world, z_rgba, _z_slot); current_axis = "z"
                b = z_native[fi]
                n_full = min(int(np.searchsorted(z_top_s, b, side="right")), n)
                n_empty = min(int(np.searchsorted(z_bot_s, b, side="left")), n)
                for k in range(n_full, n_empty):
                    frac = np.clip((b - z_bot_s[k]) / (2.0 * half_edge), 0.0, 1.0)
                    geom = scn.geoms[_z_slot(k)]
                    geom.size[2] = half_edge * frac
                    geom.pos[2] = z_world[k, 2] - half_edge * (1.0 - frac)
                scn.ngeom = base + n_empty
            frames[fi] = renderer.render()
        renderer.close()
        out[f"frames_{i}"] = frames
        print(f"rv3d worker: {robot} done ({n_frames} frames, {n} voxels)")
    io_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(io_out, **out)


def _plot_reach_visible_grid(cases: list[dict], path_base: Path, flat: bool = False,
                             vis_cmap: str = "Greens", blind_cmap: str = "Greys",
                             wrap: int | None = None,
                             rows_sel: tuple[int, ...] = (0, 1),
                             mid_cmap: str | None = None,
                             extra_row=None, extra_panel: str = "",
                             video: bool = False, video_seconds: float = 6.0,
                             video_fps: int = 30, video_hold_frames: int = 8,
                             video_3d_row: bool = False, video_3d_stride: int = 1,
                             video_3d_alpha: float = 1.0, video_3d_tilt: float = 20.0,
                             video_3d_elevation: float = -25.0) -> None:
    """N-case comparison: rows = Top/Side section cuts, columns = cases (a robot + a visibility mask).

    `video=True` writes `path_base.mp4` INSTEAD of the static `.png`/`.pdf` (call this function twice,
    once each way, to get both): identical figure -- same layout, silhouettes, titles, colorbars -- but
    each row's section-cut boundary sweeps the FULL shared display window (`disp_bounds`) instead of
    sitting at its one fixed plane, so the animation reads as the static figure's cut PLANE physically
    moving through the volume rather than a different figure. Top row sweeps Z top->bottom (native
    boundary per case = display Z minus that case's own `z_off`, since `z_off` differs per robot but the
    displayed sweep height must not); side row sweeps Y near->far (already common frame, no offset).
    Without `video_3d_row` (the plain 2-row video), both boundaries move SIMULTANEOUSLY, chosen so
    EVERY row holds its full reachable set at frac=0 and drains to empty at frac=1 -- top's `keep` is
    Z<=boundary (drains as boundary falls) and side's is Y>=boundary (drains as boundary RISES), so
    the two boundaries move opposite ways in raw coordinates to move the same way in fill state.
    Matching fill state, not matching raw sweep direction, is what "synchronized" means here -- two
    rows retreating together reads as one coherent cut sweeping through the body; one row filling
    while the other drains over the same shared `frac` reads as unsynced even though both still
    update every frame. ONE shared progress fraction per frame (`frac`), so a viewer sees one plane
    crossing every column at once in each row. Ping-pongs (forward then reverse) with
    `video_hold_frames` pauses at each extreme, for a jump-free loop. Only the per-(case,row)
    `cut_image` RESULT changes per frame -- titles/colorbars/silhouettes are drawn once and reused via
    `im.set_data`, since those depend on the fixed robot pose and full reachable/visible SETS, not on
    any one cut plane's position.

    `video_3d_row=True` (video-only -- asserted) switches the timeline to SEQUENTIAL instead --
    matching `_render_vrw_video`'s own grow-then-erase phases (Z cut, "xy" section, THEN Y cut, "yz"
    section, never both moving at once) -- and prepends a THIRD row above Top/Side: a perspective
    MuJoCo render per case of the same reachable+visible voxels as boxes, full resolution by default
    (`video_3d_stride=1`) with the same SDF sub-voxel interpolation as `_render_vrw_video` (see
    `_rv3d_render_worker`'s docstring for the per-axis clip). Phase A sweeps Z low->high (Top row and
    the 3D row's Z extent reveal top-down; Y pinned at its fully-permissive extreme, so Side row and
    the 3D row's Y extent sit at their full, uncut state). Phase B holds Z at fully-grown while Y
    sweeps low->high (Side row cuts away; Top/3D-row's Z extent hold still, fully revealed). The
    mirrored return (Y un-cuts, then Z un-reveals) brings every row back to EXACTLY its phase-A start
    state, so the loop closes with no jump for EVERY row -- the OLD simultaneous ping-pong only
    bookended the 3D row's own intersection at empty; Top and Side individually crossed a mismatched
    pair of states at the wrap (Top empty<->full, Side full<->empty), which a viewer would see as a
    pop. Camera: `_robot_forward_azimuth(model, d, robot) + video_3d_tilt`, `video_3d_elevation` --
    per-robot front-facing, same as `_render_vrw_video`'s single-robot convention. An earlier
    version used ONE shared azimuth here instead, because `_robot_forward_azimuth` put three robots
    (booster_t1/fourier_gr3/pal_talos) on the opposite world side from the other three, mirroring
    the Y-cut sweep's screen direction between them -- that turned out to be `_robot_forward_azimuth`
    itself picking up a sign flip specific to those 3 robots' head-camera site convention
    (`_HEAD_CAM_Z_INTO_HEAD`), not a real disagreement about which way is front. With that fixed at
    the source, all 6 robots' true front-facing azimuth measures the same, so per-robot
    `_robot_forward_azimuth` gives both a consistent viewpoint and correct per-robot orientation.
    The 3D row's camera POSITION and zoom, unlike its azimuth, are NOT per-case: all six frame the
    same shared `disp_bounds` window the Top/Side rows use, at one shared distance, so the row reads
    as a size comparison and every robot stands on the same foot-referenced ground line. See
    `_rv3d_render_worker`'s docstring.

    Binary color per cell: vis_cmap = reachable AND visible, blind_cmap = reachable but blind, D-shaded (`flat`
    drops shading). Section-cut FACE (frontmost voxel nearest the cut plane), same convention as the
    single figure (top z<=cutoff_z, side y>=0). All columns share one metres-per-inch scale AND common
    display axis limits (foot-referenced Z), so the three cases are directly comparable; each column
    carries its own robot silhouette at the arms-down pose. Top-plane height is stored per case in
    `top_cutoff_z`, derived from that robot's shoulder-pitch center (FK-computed from source MJCF) and
    converted to payload base coordinates. Each case dict needs: name, robot, points, reached, dexterity,
    visible, top_cutoff_z.

    An optional per-case `head` string is drawn as a second title line naming the articulation that
    aims that column's cameras. The cross-robot figure needs it because five of its six external
    columns are scored DYNAMIC, over every legal neck state, so an untagged "Fourier GR-3 70%" beside
    "G1 16%" reads as two fixed heads and inverts the figure's point. Title height is sized from this
    key (see `title_lines`), so adding it to one case grows the margin for all.

    Optional per-case keys `vis_cmap` / `blind_cmap` / `flat` override the figure-wide palette and
    D-shading for one column, so a column whose two categories are NOT visible/blind can opt out of
    the shared colorbars. No caller uses them today; `--camera-count-cuts` did, for a lost-voxel diff
    column (reached=K=2 reach, visible=lost going to K=3) dropped when that figure went to three
    panels at single-column width.

    Volumes in the column headers are the MIRRORED whole-body set, in both figures this draws, and
    the ablation table uses that same convention as of 2026-07-29, so headers and table agree by
    construction (the shipped dual rig reads 1.366/1.414 m³ in both). An earlier revision of this
    docstring said the table was right-arm-only; it was, and the disagreement that created is
    exactly why the convention was unified. Do not reintroduce a per-figure convention.

    `wrap` lays the cases out in that many columns, adding case-ROWS as needed, instead of one long
    strip; `rows_sel` picks which section cuts to draw (0 = top XY, 1 = side XZ). Both exist for the
    single-column paper figure, where panel width is the binding constraint. Defaults reproduce the
    full two-row wide strip, so the cross-robot comparison figure is unaffected.

    `mid_cmap` plus a per-case `visible_static` mask splits the visible category in two, giving a
    THREE-way partition of the reachable set: `visible_static` (seen with the cameras at rest),
    `visible & ~visible_static` (seen only once the gimbal aims), and `~visible` (blind). This is
    exact rather than a display convention, because `vis_fixed` is a subset of `vis_steer` by
    construction; verified 0 violations on all three K rigs. It exists so the camera-count figure can
    show what articulation contributes in ONE panel per rig instead of differencing a fixed row
    against an actuated row, which cost twice the panels and made the reader do the subtraction.
    `visible_static` must be a subset of `visible`; a stray voxel outside it would silently render as
    steer-only. Omit `mid_cmap` and the two-category behaviour is unchanged.

    `extra_row` is a callback `(matplotlib.axes.Axes) -> None` drawing one FOREIGN panel into a strip
    added BELOW the colorbar band, so the reading order is cuts, then the legend that decodes them,
    then the extra panel. That is what lets the paper's camera-count figure carry six section cuts
    AND a separation curve in one graphic. Two other arrangements were tried and rejected: a fourth
    grid column widened the source to 15.2 in, so at single-column placement the reduction was 0.22
    and 18 pt panel type set at 4 pt; sharing one strip between the panel and a vertically stacked
    bar block was shorter still but moved the bars away from the cuts they label. The bottom strip
    keeps the source at its three-column width (~11.0 in, reduction 0.31, ~5.6 pt final) and leaves
    the bar band exactly where the standalone cut figure puts it. The callback owns its axes
    completely, including any legend it places outside them, so `left`/`right` reserve nothing for
    it; this function only positions the axes and labels the strip with `extra_panel`. Leave it None
    and every margin, font and colorbar position is exactly as before, which keeps `fig:workspace`
    untouched.
    """
    import mujoco


    # rows: (h_axis, v_axis, ylabel, azimuth, elevation, imshow-origin). Top = XY (depth Z, look down),
    # Side = XZ (depth Y, look along Y); both put X-forward on the horizontal axis so rows align.
    _ALL_ROWS = ((0, 1, "Y left (m)", 90, -90, "lower"), (0, 2, "Z up (m)", 90, 0, "upper"))
    rows = tuple(_ALL_ROWS[i] for i in rows_sel)
    if video_3d_row:
        assert video, "video_3d_row only makes sense with video=True (see docstring)"
        # Sentinel row: h_axis=None marks it as the perspective render, not a `cut_image` section --
        # every other tuple slot (v_axis/ylabel/az/el/origin) is unused for it, so all five stay None.
        rows = ((None, None, None, None, None, None),) + rows
    def shade(base_cmap: str, dvals: np.ndarray, case_flat: bool) -> np.ndarray:
        if case_flat:
            # Single tint per category, no D-shaded gradient -- the constant `t=0.75` lands in
            # the middle of the shaded cmap, so the voxel colors and the (single-tint) bars stay
            # visually consistent.
            return _shaded_cmap(base_cmap)(np.full(dvals.shape, 0.75)).astype(np.float32)
        # Use the shaded cmap with Normalize(0, _DEXTERITY_VMAX) so the colorbar AND the voxels
        # display the SAME palette -- avoids the bar-tick-at-0-showing-Greens(0)-white mismatch
        # with voxels-at-D=0-showing-Greens(0.35).
        return _shaded_cmap(base_cmap)(np.clip(dvals / _DEXTERITY_VMAX, 0.0, 1.0)).astype(np.float32)

    for c in cases:
        model, home_pos, _ov, visual_mask_fn = _overlay_assets(c["robot"])
        visual = visual_mask_fn(model)
        model.geom_rgba[~visual, 3] = 0.0
        d = mujoco.MjData(model)
        base_pos = set_home_pose(model, d, pose_overrides=_ARMS_DOWN_POSE.get(c["robot"], {}), home_pos=home_pos)
        mujoco.mj_forward(model, d)
        xmat = d.geom_xmat[visual].reshape(-1, 3, 3)
        he = np.einsum("nij,nj->ni", np.abs(xmat), model.geom_size[visual])
        c["model"] = model; c["data"] = d; c["base_pos"] = base_pos
        c["robot_lo"] = np.min(d.geom_xpos[visual] - he - base_pos, axis=0)
        c["robot_hi"] = np.max(d.geom_xpos[visual] + he - base_pos, axis=0)
        # geom_size bounds a mesh conservatively. Foot-reference must use actual rendered mesh
        # vertices, otherwise the sole visibly floats above displayed Z=0.
        mesh_z_lo = []
        for gid in np.flatnonzero(visual & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
            mid = int(model.geom_dataid[gid])
            start = int(model.mesh_vertadr[mid]); count = int(model.mesh_vertnum[mid])
            verts = model.mesh_vert[start:start + count]
            mesh_z_lo.append(np.min(verts @ d.geom_xmat[gid].reshape(3, 3)[2] + d.geom_xpos[gid, 2]))
        if mesh_z_lo:
            c["robot_lo"][2] = min(mesh_z_lo) - base_pos[2]
        c["axis_coords"] = tuple(np.unique(c["points"][:, a]) for a in range(3))
        c["z_off"] = -float(c["robot_lo"][2])  # foot-reference: shift Z so the sole sits at 0

    def disp_range(c: dict, a: int) -> tuple[float, float]:
        # Window the SOLVED lattice by default. With `extra_row` window the REACHED voxels instead:
        # the lattice is padded past the workspace on every axis, so a solved-extent window prints a
        # blank margin on all four sides of every section, and the composite cannot afford it. The
        # robot bounds still enter, so the silhouette is never clipped.
        coords = (np.unique(c["points"][c["reached"], a]) if extra_row is not None
                  else c["axis_coords"][a])
        lo = min(_cell_edges(coords)[0], c["robot_lo"][a])
        hi = max(_cell_edges(coords)[-1], c["robot_hi"][a])
        return (lo + c["z_off"], hi + c["z_off"]) if a == 2 else (lo, hi)

    disp_bounds = {a: (min(disp_range(c, a)[0] for c in cases), max(disp_range(c, a)[1] for c in cases))
                   for a in range(3)}
    x_span = disp_bounds[0][1] - disp_bounds[0][0]
    y_span = disp_bounds[1][1] - disp_bounds[1][0]
    z_span = disp_bounds[2][1] - disp_bounds[2][0]

    def cut_image(c: dict, h_axis: int, v_axis: int, boundary: float | None = None):
        pts = c["points"]; hc = c["axis_coords"][h_axis]; vc = c["axis_coords"][v_axis]
        depth_axis = 3 - h_axis - v_axis
        if boundary is None:
            boundary = float(c["top_cutoff_z"]) if depth_axis == 2 else 0.0
        keep = c["reached"] & ((pts[:, depth_axis] <= boundary) if depth_axis == 2 else (pts[:, depth_axis] >= boundary))
        sel = np.flatnonzero(keep)
        h_idx = np.searchsorted(hc, pts[sel, h_axis]); v_idx = np.searchsorted(vc, pts[sel, v_axis])
        cell = v_idx * hc.size + h_idx
        order = np.argsort(-np.abs(pts[sel, depth_axis] - boundary), kind="stable")  # frontmost wins
        best = np.full(hc.size * vc.size, -1, np.int64); best[cell[order]] = sel[order]
        img = np.zeros((hc.size * vc.size, 4), np.float32)
        has = best >= 0; gi = best[has]; dd = c["dexterity"][gi]; vis = c["visible"][gi]
        col = np.zeros((gi.size, 4), np.float32)
        # Palette and D-shading are per case, falling back to the figure-wide setting. A column whose
        # two categories are not visible/blind (the lost-voxel diff) needs both: its own hues so the
        # reader cannot read it against the shared colorbars, and flat tints because a binary
        # lost/retained split has no gradient to decode and therefore earns no bar of its own.
        c_flat = flat or c.get("flat", False)
        static = c.get("visible_static")
        if mid_cmap is None or static is None:
            col[vis] = shade(c.get("vis_cmap", vis_cmap), dd[vis], c_flat)
        else:
            st = static[gi]
            rest, steer_only = vis & st, vis & ~st
            col[rest] = shade(c.get("vis_cmap", vis_cmap), dd[rest], c_flat)
            col[steer_only] = shade(mid_cmap, dd[steer_only], c_flat)
        col[~vis] = shade(c.get("blind_cmap", blind_cmap), dd[~vis], c_flat)
        img[has] = col
        he = _cell_edges(hc) + (c["z_off"] if h_axis == 2 else 0.0)
        ve = _cell_edges(vc) + (c["z_off"] if v_axis == 2 else 0.0)
        return he, ve, img.reshape(vc.size, hc.size, 4)

    ncol = min(wrap, len(cases)) if wrap else len(cases)
    nblock = -(-len(cases) // ncol)   # case-rows; 1 reproduces the original single strip
    # This three-column source PDF is placed at single-column width in the paper.
    # Size typography before that ~0.41x reduction so final labels remain legible.
    src_scale = 1.0
    with plt.rc_context({"font.size": 18, "axes.labelsize": 18, "axes.titlesize": 18}):
        inch_per_m = 1.9
        # `bottom` holds the X tick labels, the X label and the colorbar band, plus the foreign
        # panel's own strip when there is one. The band keeps its place inside that 2.0 in either
        # way, so the bars stay directly under the cuts they decode.
        left, right, bottom, vgap, hgap = 1.35, 0.25, 2.0, 0.12, 0.12
        # Height of the bar band's own baseline inside `bottom`. Standalone it is 0.83, which leaves
        # the bars' tick labels and category names ~0.5 in and the rest as the figure's bottom
        # margin. With a strip below, that margin is dead space between the category names and the
        # strip's title, so the band drops to what its labels actually need.
        strip_h, band_y = 4.05, 0.83
        if extra_row is not None:
            band_y = 0.52
            bottom += strip_h - (0.83 - band_y)
        # Titles are a name line plus one stats line, plus one for each optional line any case
        # carries: the `head` articulation tag and the pairwise-coverage line. A margin hard-coded
        # for the 2-line case silently CLIPPED the top line off once titles grew, so it is derived:
        # 0.3 in is one 18 pt line at default linespacing and 0.2 is the pad the 2-line figure had
        # left over, so the plain case reproduces the historical 0.8 exactly. The pairwise term was
        # missing while only `fig:workspace` used that line, which left its top line touching the
        # figure edge (531 inked pixels in row 0 of the 350 dpi render, against ~0 with the term).
        title_lines = (2 + any(c.get("head") for c in cases)
                       + any(c.get("eta2") is not None or c.get("eta2_key") for c in cases))
        top_margin = (0.2 + 0.3 * title_lines) * src_scale
        # Must clear the block above's X ticks + X label AND the block below's two-line title, or
        # they overprint. Measured at 18 pt: ~0.25 + 0.30 above, ~0.60 below, so 1.30 is the floor
        # and this sits just over it. Only paid when `wrap` splits the cases across case-rows.
        block_gap = 1.4
        # Small-reach robots (e.g. ToddlerBot, solo column) have a much smaller x_span than V2/G1,
        # which would otherwise starve the column of the physical width this font/tick sizing needs
        # (title and X-tick labels start overlapping). Clamp to the historical V2/G1 column width
        # (~2.39 in at their x_span) so solo/narrow-robot columns get the same breathing room; a
        # no-op for V2/G1 sized columns. Scale top_h/side_h by the SAME factor so the allocated box
        # aspect still matches the data aspect -- clamping col_w alone desyncs box vs. data aspect,
        # and `ax.set_aspect("equal")` (adjustable="box", the default) then shrinks the axes box to
        # compensate, dragging the y-label into the tick labels.
        col_scale = max(1.0, _COMPARE_VOLUME_MIN_COL_W_IN / (inch_per_m * x_span))
        col_w = inch_per_m * x_span * col_scale
        top_h = inch_per_m * y_span * col_scale; side_h = inch_per_m * z_span * col_scale
        # Heights of the SELECTED rows, in draw order (top of the block downward).
        row_h_all = {0: top_h, 1: side_h}
        row_hs = [row_h_all[i] for i in rows_sel]
        if video_3d_row:
            row_hs = [col_w] + row_hs  # square 3D panel, reusing the column width as its side length
        block_h = sum(row_hs) + vgap * (len(rows) - 1)
        fig_w = left + ncol * col_w + (ncol - 1) * hgap + right
        fig_h = bottom + nblock * block_h + (nblock - 1) * block_gap + top_margin
        # Video frames are captured straight off the canvas (`fig.canvas.buffer_rgba()`), unlike the
        # static path, which asks `savefig(dpi=350)` for its own resolution independent of the
        # Figure's own `dpi`. Left at the rcParams default (~100), the composed figure's ~11 in width
        # rendered at ~1100-2050 px -- well under 4K. Force >=3840 px on the wide edge (every case here
        # is wider than tall) so every video matches the static figure's resolution class.
        dpi = max(plt.rcParams["figure.dpi"], int(np.ceil(3840.0 / fig_w))) if video else None
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        def row_y0(bi: int, ri: int) -> float:
            """Bottom edge (inches) of case-row `bi`'s section-cut row `ri`, counting rows downward."""
            block_bottom = bottom + (nblock - 1 - bi) * (block_h + block_gap)
            below = sum(row_hs[ri + 1:]) + vgap * (len(rows) - 1 - ri)
            return block_bottom + below

        frame_targets = []  # video only: (im, c, h_axis, v_axis, depth_axis) to update per frame
        frame_targets_3d = []  # video_3d_row only: (im, c) to re-render via MuJoCo per frame
        for ci, c in enumerate(cases):
            bi, col = divmod(ci, ncol)
            x0 = left + col * (col_w + hgap)
            reachable_count = int(c["reached"].sum())
            visible_count = int((c["reached"] & c["visible"]).sum())
            pct = 100.0 * visible_count / max(reachable_count, 1)
            visible_volume_m3 = visible_count * c["voxel_vol"]
            reachable_volume_m3 = reachable_count * c["voxel_vol"]
            for ri, (h_axis, v_axis, ylabel, az, el, origin) in enumerate(rows):
                row_h, y0 = row_hs[ri], row_y0(bi, ri)
                ax = fig.add_axes((x0 / fig_w, y0 / fig_h, col_w / fig_w, row_h / fig_h))
                if ri == 0 and c.get("panel"):
                    # Sub-panel letter at the outer top-left CORNER of the panel's block, ABOVE the
                    # title and OUTSIDE the y-label column, the journal convention. Keeping it inside
                    # the title centred it with the text and made it read as part of the column's
                    # name; keeping it at the axes' left edge left it sitting over the tick labels.
                    fig.text(_PANEL_X / fig_w, (fig_h - _PANEL_Y) / fig_h, c["panel"], ha="left",
                             va="top", fontweight="bold", fontsize=22)
                if h_axis is None:
                    # 3D perspective row: a full MuJoCo scene (robot + voxel boxes), not a
                    # `cut_image` flat section -- no data-coordinate axes to speak of, so ticks/
                    # spines are off and the placeholder is filled in per-frame below, in
                    # `frame_targets_3d` rather than `frame_targets`.
                    im = ax.imshow(np.zeros((2, 2, 3), np.uint8), zorder=1)
                    ax.axis("off")
                    frame_targets_3d.append((im, c))
                elif video:
                    # Placeholder: filled in per-frame below, once the whole static figure (titles,
                    # colorbars, silhouettes) is built. Extent/edges never change across frames --
                    # only which voxel is frontmost at the swept boundary does -- so `_cell_edges` is
                    # computed once here rather than inside the per-frame loop.
                    hc, vc = c["axis_coords"][h_axis], c["axis_coords"][v_axis]
                    he = _cell_edges(hc) + (c["z_off"] if h_axis == 2 else 0.0)
                    ve = _cell_edges(vc) + (c["z_off"] if v_axis == 2 else 0.0)
                    im = ax.imshow(np.zeros((vc.size, hc.size, 4), np.float32),
                                    extent=(he[0], he[-1], ve[0], ve[-1]), origin="lower",
                                    interpolation="nearest", zorder=1)
                    frame_targets.append((im, c, h_axis, v_axis, 3 - h_axis - v_axis))
                else:
                    he, ve, img = cut_image(c, h_axis, v_axis)
                    ax.imshow(img, extent=(he[0], he[-1], ve[0], ve[-1]), origin="lower",
                              interpolation="nearest", zorder=1)
                if h_axis is not None:
                    hb = disp_bounds[h_axis]; vb = disp_bounds[v_axis]
                    h_span, v_span = hb[1] - hb[0], vb[1] - vb[0]
                    # Silhouette: camera frames the COMMON display window (Z un-shifted back to base frame by
                    # this case's foot offset), so every column shares one world window and stays aligned.
                    base_center = [c["base_pos"][a] + 0.5 * (disp_bounds[a][0] + disp_bounds[a][1]
                                   - (2 * c["z_off"] if a == 2 else 0.0)) for a in range(3)]
                    hres = 1000; wres = max(2, round(hres * h_span / v_span))
                    model = c["model"]
                    model.vis.global_.orthographic = 1
                    model.vis.global_.fovy = v_span
                    model.vis.global_.offwidth = wres
                    model.vis.global_.offheight = hres
                    renderer = mujoco.Renderer(model, hres, wres)
                    camera = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(camera)
                    camera.lookat[:] = base_center
                    camera.distance = 5.0
                    camera.azimuth = az; camera.elevation = el; camera.orthographic = 1
                    # Default-group sites (e.g. the head camera's unsized sensor sites) render as
                    # stray dots otherwise; see `_export_viewer_snapshot`'s scene_option for the fix.
                    site_off = mujoco.MjvOption(); site_off.sitegroup[:] = 0
                    renderer.update_scene(c["data"], camera, scene_option=site_off)
                    rgb = renderer.render()
                    renderer.enable_depth_rendering(); depth = renderer.render(); renderer.disable_depth_rendering()
                    renderer.close()
                    alpha = np.where(depth >= depth.max() - 1e-6, 0, 230).astype(np.uint8)
                    ax.imshow(np.dstack((rgb, alpha)), extent=(*hb, *vb), origin=origin, zorder=3)
                    # Both axes use a shared comparison tick cadence. Set before explicit bounds:
                    # Matplotlib otherwise expands a small robot's view to accommodate off-range ticks.
                    tick_start = np.ceil(vb[0] / _COMPARE_TICK_M) * _COMPARE_TICK_M
                    v_ticks = np.arange(tick_start, vb[1] + 1e-9, _COMPARE_TICK_M)
                    ax.set_yticks(v_ticks, [_format_compare_tick(tick) for tick in v_ticks])
                    tick_start = np.ceil(hb[0] / _COMPARE_TICK_M) * _COMPARE_TICK_M
                    h_ticks = np.arange(tick_start, hb[1] + 1e-9, _COMPARE_TICK_M)
                    # Blank the OUTERMOST x labels once columns sit side by side. Only `hgap` (0.12 in)
                    # separates one column's last label from the next column's first, so at the compare
                    # figure's 2.4 in columns "0.75" and "-0.75" collide outright. The ticks stay, so the
                    # axis extent is still readable; only the two labels that overlap are dropped.
                    h_labels = [_format_compare_tick(t) for t in h_ticks]
                    if ncol > 1:
                        h_labels[0] = h_labels[-1] = ""
                    ax.set_xticks(h_ticks, h_labels)
                    ax.set_aspect("equal"); ax.set_xlim(hb); ax.set_ylim(vb)
                    ax.tick_params(direction="out", length=4, labelsize=16 * src_scale)
                if ri == 0:
                    # Optional head-articulation tag. Without it "Fourier GR-3 70%" next to "G1 16%"
                    # reads as two fixed heads, when GR-3, TALOS, T1, Apollo and ToddlerBot are all
                    # scored DYNAMIC -- over every legal neck state (`gpu_visibility._ADAPTERS`). The
                    # figure's claim is not "articulation beats none", which those columns would
                    # contradict, but that a shared neck aims both eyes together while our two
                    # gimbals aim separately, so the tag has to be on the panel to be read with it.
                    head = f"\n{c['head']}" if c.get("head") else ""
                    # Pairwise coverage, the field that makes this figure state the hardware claim
                    # instead of implying it. Single-target eta is structurally blind to camera
                    # INDEPENDENCE: a 2-DoF neck can point at any ONE named voxel, which is why the
                    # neck columns score 48-80% and a reviewer answers "buy a neck". eta_2 asks
                    # whether the rig can hold TWO points at once, where one shared viewing direction
                    # has to fit both in a single frustum, and the neck columns fall to 20-58% while
                    # ours holds 96%. Rendered on its own line: appending it to the percentage line
                    # runs that line past the 2.39 in column width and collides with the neighbour.
                    # `eta2` on the case wins over the manifest lookup. The manifest is written by
                    # `test/run_eta2_platforms.py`, which scores the SHIPPED rig and the external
                    # platforms; it has no K=1/K=3 entries and should not grow any, because the
                    # camera-count ablation solves those rigs itself and its own eta_2 is the number
                    # the results text quotes. Two kernels, so never mix them in one figure.
                    eta2 = c.get("eta2", _eta2_manifest().get(c.get("eta2_key", ""), {}).get("eta2"))
                    # `eta2_fixed`, when the case carries it, drives the same fixed->actuated arrow the
                    # single-target coverage uses above: the camera-count columns are one rig with
                    # gimbals locked vs steered, so a lone actuated number would hide the articulation
                    # gain. Cross-robot columns (Fig. 2) have no locked counterpart and stay one number.
                    eta2_fixed = c.get("eta2_fixed")
                    if eta2 is None:
                        pair = ""
                    elif eta2_fixed is not None:
                        pair = f"\npairwise {100 * eta2_fixed:.0f}%→{100 * eta2:.0f}%"
                    else:
                        # One decimal on THIS branch only. The two kernels above disagree by 0.0032
                        # on the shipped rig (manifest 0.9572, `_pairwise_vrw` 0.9540), which at
                        # `.0f` straddles 95.5 and prints 96% here against the camera-count figure's
                        # 95% and the paper's 0.95 -- read as a contradiction rather than as two
                        # scorers. At `.1f` they read 95.7 and 95.4, the same number at the precision
                        # the text quotes. The arrow branch keeps `.0f`: its numbers are the ones
                        # already published in the video and on the site.
                        pair = f"\npairwise {100 * eta2:.1f}%"
                    if c.get("visible_static") is not None:
                        # Both coverage fractions, because the panel draws both categories and a
                        # single "(97%)" would leave the fixed region unlabelled. These are eta_fix
                        # and eta of the ablation table, so the header is its cross-check. The arrow
                        # form is what fits: spelling them out as "fixed 19%" / "actuated 94%" needs
                        # two more lines (four total, which crowded the panels), while putting those
                        # words on one line runs ~2.5 in at 18 pt against a 2.39 in column and
                        # collides with the neighbouring title. "19%->94%" is 13 chars, ~1.8 in, and
                        # reads as the direction the figure argues: articulation, not K, is what
                        # moves coverage. The legend below names the two categories.
                        fixed_count = int((c["reached"] & c["visible_static"]).sum())
                        fixed_pct = 100.0 * fixed_count / max(reachable_count, 1)
                        # Both lines carry the SAME fixed->actuated step, as a fraction then as a
                        # volume, so the header cross-checks both ablation-table columns instead of
                        # one: the volumes are W_VR^fix and W_VR, and the fractions are eta_fix and
                        # eta. Strictly the volume line is the fraction line times the reachable
                        # volume, but W_VR^fix carries its own paragraph in the results and a reader
                        # comparing figure to table should not have to do that multiply.
                        ax.set_title(f"{c['name']}{head} ({fixed_pct:.0f}%→{pct:.0f}%)\n"
                                     f"{fixed_count * c['voxel_vol']:.3f}→{visible_volume_m3:.3f}"
                                     f"/{reachable_volume_m3:.3f} m³{pair}")
                    elif c['name'] == "Ours (actuated)":
                        ax.set_title(f"{c['name']}{head}\n" + rf"$\bf{{({pct:.0f}\%)\ {visible_volume_m3:.3f}/{reachable_volume_m3:.3f}\ m^3}}$" + pair)
                    else:
                        ax.set_title(f"{c['name']}{head}\n({pct:.0f}%) {visible_volume_m3:.3f}/{reachable_volume_m3:.3f} m³{pair}")
                if h_axis is not None:
                    if col == 0:
                        ax.set_ylabel(ylabel)
                        # Default placement follows tick-label width, which differs between rows.
                        # Pin both vertical labels to one column-aligned coordinate instead.
                        ax.yaxis.set_label_coords(-0.22, 0.5)
                    else:
                        ax.tick_params(labelleft=False)
                    # Every block's bottom row carries the X label: with a wrap the blocks are separated
                    # vertically, so an unlabelled upper block would read as an axis-free strip. The 3D
                    # row is never the bottom row (it only exists alongside Top/Side), so this never
                    # needs to fire for it.
                    if ri == len(rows) - 1:
                        ax.set_xlabel("X forward (m)")
                    else:
                        ax.tick_params(labelbottom=False)

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        # Every category hue encodes the same reachability index. Each bar uses the SAME shaded
        # cmap + Normalize the voxels use -- what you see at tick "0" IS the color a low-D voxel
        # has in the figure (no Greens(0)-white mismatch). Greens-shaded range [_CMAP_T_MIN,
        # _CMAP_T_MAX], Reds-shaded range, both displayed over D in [0, _DEXTERITY_VMAX].
        norm = Normalize(0.0, _DEXTERITY_VMAX)
        # Colorbar geometry in inches (matches the historical ncol=3 V2/G1 figure's size), centered
        # under the plotted columns and shrunk-to-fit for narrower figures (solo columns, e.g.
        # ToddlerBot) -- the old fixed fig-FRACTION geometry (0.20/0.57, width 0.25) assumed a
        # ~8.7 in ncol=3 figure and overlapped for any narrower one.
        cbar_w, cbar_h, cbar_gap, cbar_y = 2.17, 0.20, 1.0, band_y + (strip_h if extra_row else 0.0)
        # Three bars share the same width two used, so the labels name only the CATEGORY and the
        # shared line above names the scale once. Do NOT expand these to "Reachable visible-fixed
        # index" and friends: it asserts three indices where there is one R on three hues, and it
        # does not fit -- measured at 16 pt those run 3.25 and 3.69 in against a 3.17 in bar pitch,
        # and `font_scale` cannot save them because it is derived from BAR widths, not label widths.
        # "Fixed"/"Actuated" rather than "at rest"/"steered" because these are the SAME quantities
        # the comparison figure's "Ours fixed" column and Table IV's eta_fix report -- Fig. 2's fixed
        # column and this figure's K=2 fixed category are both 66,534 voxels, one mask. The middle
        # bar must keep "only": it is actuated MINUS fixed (56-75%), not the actuated set (94-98%).
        # Two-bar form: "Visible-reachable" / "Blind-reachable", NOT the historical
        # "Reachable-visible index" / "Reachable-blind index". Three separate fixes. Word order is
        # the paper's own term (visible-reachable workspace, W_VR), which the figure had inverted.
        # The "index" suffix is gone because two labels ending in "index" read as two indices when
        # both bars carry one R; the shared scale line below now runs for both forms and names it
        # once. And BOTH keep the "-reachable" half: the figure draws only reachable voxels and
        # partitions them, so a bare "Blind" would drop the qualifier the two categories share and
        # read as blind-anywhere rather than reachable-but-blind.
        bars = ([(vis_cmap, "Visible, fixed"), (mid_cmap, "Visible, actuated only"),
                 (blind_cmap, "Blind")]
                if mid_cmap is not None else
                [(vis_cmap, "Visible-reachable"), (blind_cmap, "Blind-reachable")])
        nbar = len(bars)
        avail_w = fig_w - left - right
        total_w = nbar * cbar_w + (nbar - 1) * cbar_gap
        font_scale = min(1.0, avail_w / total_w)
        cbar_w *= font_scale; cbar_gap *= font_scale
        x_start = left + (avail_w - (nbar * cbar_w + (nbar - 1) * cbar_gap)) / 2
        boxes = [(x_start + i * (cbar_w + cbar_gap), cbar_y, cbar_w, cbar_h) for i in range(nbar)]
        for (x0, y0_bar, w_bar, h_bar), (cmap, label) in zip(boxes, bars):
            cax = fig.add_axes((x0 / fig_w, y0_bar / fig_h, w_bar / fig_w, h_bar / fig_h))
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=_shaded_cmap(cmap)), cax=cax,
                               orientation="horizontal")
            bar.set_ticks((0.0, _DEXTERITY_VMAX))
            bar.set_ticklabels(("0", f"{_DEXTERITY_VMAX:g}"))
            bar.ax.tick_params(labelsize=15 * font_scale, length=3, pad=2)
            bar.set_label(label, fontsize=16 * font_scale, labelpad=3)
        if extra_row is not None:
            # The foreign panel owns the strip below the bar band, and is INSET inside it, unlike the
            # section-cut axes: those get their tick labels, axis labels and titles from the figure's
            # own `left` / `bottom` / `top_margin`, while a foreign panel carries all four itself and
            # would otherwise print them over its neighbours. Measured at this type size: x label
            # plus ticks 0.95 in, and 0.40 in of head room so the panel letter clears the y label,
            # which is 23 characters and taller than the axes would otherwise be. Its LEFT and RIGHT
            # edges are the grid's, so the strip and the grid share both vertical rules and the
            # panel's y label sits in the same `left` margin the grid's does.
            pad_b, pad_t = 0.95, 0.40
            ax_extra = fig.add_axes((left / fig_w, pad_b / fig_h,
                                     (fig_w - left - right) / fig_w,
                                     (strip_h - pad_b - pad_t) / fig_h))
            # Same x as the grid letter, so the two share one vertical rule. Set 0.12 in ABOVE the
            # strip's own top, unlike the grid letter, which cannot rise past the figure edge: this
            # panel carries no title, so the space over its axes is empty, and the letter otherwise
            # floats low against a tall y label. The lift is bounded by the bar category names above,
            # whose descenders reach ~0.12 in below the band.
            fig.text(_PANEL_X / fig_w, (strip_h + 0.12 - _PANEL_Y) / fig_h, extra_panel, ha="left",
                     va="top", fontweight="bold", fontsize=22)
            extra_row(ax_extra)
            # Pin the foreign panel's y label to the SAME figure x as the section cuts', which sit at
            # -0.22 of a COLUMN width. The two panels have different axes widths, so leaving this to
            # matplotlib (which places the label off its own tick-label extent) or copying the grid's
            # axes-fraction offset both leave the two labels a few hundredths of an inch apart, which
            # is visible against the shared left rule.
            ax_extra.yaxis.set_label_coords(-0.22 * col_w / (fig_w - left - right), 0.5)
        # Every bar label names only its CATEGORY, so without this line nothing says what the shared
        # 0-0.7 scale is and the reader has numbered bars with no quantity. One centred line carries
        # it once for BOTH forms; folding the scale into the labels instead is not available, since
        # the compound names overrun the bar pitch (see `bars`). Sits ABOVE the bars, in the gap
        # between them and the X label: the band below is taken by the tick labels and the category
        # names, measured at 0.29-0.51 in.
        # "reachability index" is Zacharias's own name for the solved-orientation fraction, so
        # "orientation" was redundant and put the figure at odds with its caption, which already
        # says "shade is the reachability index R". Do NOT retitle this "conditional reachability
        # index": nothing here is conditioned, and the invented term would need a definition the
        # figure has no room for.
        fig.text(0.5, (cbar_y + cbar_h + 0.06) / fig_h, "shade = reachability index $R$",
                 ha="center", va="bottom", fontsize=16 * font_scale)
        if video:
            import imageio.v2 as imageio

            # Sweep range = the SAME common display window (`disp_bounds`) the axes/silhouettes are
            # framed to, so the moving plane never crosses outside what's drawn. Top boundary is
            # native-frame per case (subtract that case's own foot-reference `z_off` back out); side
            # boundary has no per-case offset, so it's shared as-is.
            z_lo, z_hi = disp_bounds[2]; y_lo, y_hi = disp_bounds[1]
            n_sweep = max(2, round(video_fps * video_seconds))
            if video_3d_row:
                # SEQUENTIAL two-phase cut, matching `_render_vrw_video`'s own grow-then-erase
                # pattern (Z first, "xy" section, then Y, "yz" section) instead of animating both
                # boundaries at once -- Top row cuts, THEN Side row cuts, exactly one axis moving at
                # any given frame. Phase A: Z sweeps low->high (Top row reveals top-down; Y pinned
                # at `y_lo`, i.e. fully permissive, so Side row -- and the 3D row's Y extent -- sit
                # at their full, uncut state the whole phase). Phase B: Z stays pinned at `z_hi`
                # (fully permissive; Top row and the 3D row's Z extent hold still, fully revealed)
                # while Y sweeps low->high (Side row cuts away). The mirrored return (Y un-cuts, then
                # Z un-reveals) brings every row back to EXACTLY its phase-A start state, so the loop
                # closes with no jump -- a plain ping-pong of the OLD simultaneous ramp could not do
                # this per-row (Top and Side would each cross a different, mismatched pair of states
                # at the wrap), only the combined 3D intersection happened to bookend at empty.
                n_phase = max(2, round(n_sweep / 2.0))
                phase_a_z = np.linspace(z_lo, z_hi, n_phase)
                phase_a_y = np.full(n_phase, y_lo)
                phase_b_z = np.full(n_phase, z_hi)
                phase_b_y = np.linspace(y_lo, y_hi, n_phase)
                # Phase A's last frame and phase B's first are the SAME state (Z fully grown, Y
                # still untouched) -- drop the duplicate at that internal seam.
                z_fwd = np.concatenate([phase_a_z, phase_b_z[1:]])
                y_fwd = np.concatenate([phase_a_y, phase_b_y[1:]])
            else:
                # Both boundaries move the SAME sense (full reachable set at f=0, draining to empty at
                # f=1) even though their `keep` predicates point opposite ways (top keeps Z<=boundary
                # so it drains as boundary FALLS; side keeps Y>=boundary so it drains as boundary
                # RISES) -- without this the two rows looked unsynced: one filling while the other
                # drained over the same shared `f`, instead of both retreating together.
                frac_fwd = np.linspace(0.0, 1.0, n_sweep)
                z_fwd = z_hi - frac_fwd * (z_hi - z_lo)
                y_fwd = y_lo + frac_fwd * (y_hi - y_lo)
            # One forward pass, hold, reverse pass (endpoints deduped), hold -- ends where it starts
            # for a jump-free loop.
            z_boundary_displays = np.concatenate([
                np.full(video_hold_frames, z_fwd[0]), z_fwd,
                np.full(video_hold_frames, z_fwd[-1]), z_fwd[-2::-1],
            ])
            y_boundaries = np.concatenate([
                np.full(video_hold_frames, y_fwd[0]), y_fwd,
                np.full(video_hold_frames, y_fwd[-1]), y_fwd[-2::-1],
            ])

            three_d = [None] * len(frame_targets_3d)  # (im, frames_rgb) per 3D panel, index-aligned
            if video_3d_row:
                import subprocess
                import tempfile
                from concurrent.futures import ThreadPoolExecutor

                res_3d = 900  # fixed square resolution -- a sub-panel thumbnail, not the hero shot
                # Rendered in a FRESH SUBPROCESS, not in-process -- see `_rv3d_render_worker`'s
                # docstring. This process already spent its `mujoco.Renderer` budget on the 12
                # Top/Side silhouette renders above (6 cases x 2 rows); a 13th here segfaulted natively.
                #
                # ONE case per subprocess call, each independently retried, rather than one call for
                # all 6: MuJoCo's offscreen EGL renderer segfaults sporadically in this environment --
                # confirmed via a standalone repro (same script, same inputs; succeeded once, then
                # crashed on the next 4 consecutive attempts; not a geom-count or camera-parameter bug,
                # just transient GL/driver flakiness, worse under concurrent GPU load from other
                # processes on the box). A segfault has no Python exception to catch, so the only lever
                # is relaunching a fresh process and hoping the next one lands clean -- matches this
                # file's existing note on the batch `--vrw-video` renders needing the same mitigation.
                # Per-case isolation means a case that keeps failing doesn't force every ALREADY-
                # rendered case to re-render on every retry.
                #
                # The 6 cases are otherwise fully independent (own model, own renderer, own EGL
                # context), so they run CONCURRENTLY via a thread pool -- `subprocess.run` blocks with
                # the GIL released, so N Python threads each waiting on their own OS process is real
                # parallelism, not fake. Uncapped (one worker per case): measured at 3 concurrent, GPU
                # sat at 83% (`nvidia-smi`), so 3 wasn't the ceiling, and the per-case retry loop below
                # already absorbs a transient EGL segfault regardless of how many workers run together
                # -- so a lower cap would only trade wall time for a robustness margin this environment
                # hasn't needed. Revisit with an explicit cap if a future run shows it thrashing.
                tmp_dir = Path(tempfile.mkdtemp(prefix="rv3d_"))
                worker_script = str(Path(__file__).resolve())
                max_attempts = 10
                # ONE shared camera framing for the whole 3D row, handed to every worker: the SAME
                # world window (`disp_bounds`) the Top/Side rows are drawn to, so the three rows are
                # three views of one box at one scale, and Z=0 -- the foot-referenced ground line --
                # lands at the same panel height in every column. Radius is the box's bounding
                # SPHERE (half the space diagonal), which no oblique camera azimuth can clip; see
                # `_rv3d_render_worker`'s camera block for what the discarded per-case bbox fit got
                # wrong.
                view_radius = 0.5 * float(np.linalg.norm([x_span, y_span, z_span]))

                def _render_one_case(i: int, im, c: dict):
                    sel = c["reached"] & _grid_stride_mask(c["points"], video_3d_stride)
                    # Window centre in THIS case's world frame: undo the case's own foot offset on Z
                    # so one display window maps onto six differently-rooted robots. Identical
                    # formula to the Top/Side silhouette camera's `base_center`.
                    view_center = [c["base_pos"][a] + 0.5 * (disp_bounds[a][0] + disp_bounds[a][1]
                                   - (2 * c["z_off"] if a == 2 else 0.0)) for a in range(3)]
                    io_in, io_out = tmp_dir / f"in_{i}.npz", tmp_dir / f"out_{i}.npz"
                    np.savez(io_in, robots=np.array([c["robot"]], dtype="<U40"),
                            z_boundary_displays=z_boundary_displays, y_boundaries=y_boundaries,
                            stride=video_3d_stride, tilt=video_3d_tilt, elevation=video_3d_elevation,
                            alpha=video_3d_alpha, res=res_3d, vis_cmap=vis_cmap, blind_cmap=blind_cmap,
                            view_radius=view_radius, view_center_0=np.asarray(view_center),
                            points_0=c["points"][sel], dexterity_0=c["dexterity"][sel],
                            visible_0=c["visible"][sel], base_pos_0=c["base_pos"], z_off_0=c["z_off"])
                    worker_cmd = [sys.executable, worker_script,
                                 "--rv3d-worker-in", str(io_in), "--rv3d-worker-out", str(io_out)]
                    for attempt in range(1, max_attempts + 1):
                        proc = subprocess.run(worker_cmd)
                        if proc.returncode == 0 and io_out.is_file():
                            break
                        print(f"rv3d worker ({c['robot']}) attempt {attempt}/{max_attempts} failed "
                              f"(exit {proc.returncode}), retrying...")
                    else:
                        raise RuntimeError(f"_rv3d_render_worker failed {max_attempts} times in a "
                                           f"row for {c['robot']}")
                    return i, im, np.load(io_out)["frames_0"]

                with ThreadPoolExecutor(max_workers=len(frame_targets_3d)) as pool:
                    futures = [pool.submit(_render_one_case, i, im, c)
                              for i, (im, c) in enumerate(frame_targets_3d)]
                    for fut in futures:
                        i, im, frames = fut.result()  # re-raises any worker's RuntimeError here
                        three_d[i] = (im, frames)
                shutil.rmtree(tmp_dir, ignore_errors=True)

            out_path = path_base.with_suffix(".mp4")
            # `-preset veryfast`: measured 273 -> 186 ms/frame encode time at this figure's ~4K frame
            # size (default x264 preset optimizes compression ratio, not speed; this trades a larger
            # file for faster encode, no change to the RENDERED pixels). At 373 frames that's the
            # difference between ~100 s and ~70 s of encode time -- a real fraction of total render
            # time once `fig.canvas.draw()` itself is also ~300 ms/frame at this resolution.
            writer = imageio.get_writer(str(out_path), fps=video_fps, codec="libx264", quality=8,
                                        output_params=["-preset", "veryfast"])
            for fi in range(z_boundary_displays.size):
                for im, c, h_axis, v_axis, depth_axis in frame_targets:
                    boundary = ((z_boundary_displays[fi] - c["z_off"]) if depth_axis == 2
                               else y_boundaries[fi])
                    _, _, img = cut_image(c, h_axis, v_axis, boundary=boundary)
                    im.set_data(img)
                # 3D row: same two boundaries, but as an INTERSECTION (both conditions on the same
                # points) rather than two separate sections -- this panel is the literal 3D shape the
                # Top/Side rows are each showing a cut of, not a third independent sweep.
                for im, frames_rgb in three_d:
                    im.set_data(frames_rgb[fi])
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
            writer.close()
            plt.close(fig)
            print(f"saved {out_path}")
        else:
            fig.savefig(path_base.with_suffix(".png"), dpi=350)
            fig.savefig(path_base.with_suffix(".pdf"))
            plt.close(fig)
            print(f"saved {path_base.with_suffix('.png')} and {path_base.with_suffix('.pdf')}")


# (view_name, azimuth, elevation, orthographic) -- same three camera conventions as
# `_plot_reach_visible_grid`'s Top/Side rows (`_ALL_ROWS`) and `_rv3d_render_worker`'s 3D-row
# camera. Azimuth `None` for the 3D view is a sentinel: `_orientation_debug_worker` computes it
# per-robot via `_robot_forward_azimuth` + tilt, same as the real pipeline now does. Module-level so
# both the worker and the parent (which only needs the labels) read the identical list.
_ORIENTATION_DEBUG_VIEWS = [
    ("Top (az=90,el=-90, ortho)", 90.0, -90.0, True),
    ("Side (az=90,el=0, ortho)", 90.0, 0.0, True),
    ("3D (az=fwd+20,el=-25, persp)", None, -25.0, False),
]


def _orientation_debug_worker(robot: str, view_idx: int, shared_bbox_radius: float, out_path: Path) -> None:
    """Subprocess entry point for `_render_orientation_debug` -- renders ONE robot's ONE view (no
    voxels, base-link XYZ axes overlaid) and saves the RGB array to `out_path` (`.npz`), then exits.
    `shared_bbox_radius` is the MAX bounding radius over all 6 robots (computed once by the parent,
    before any subprocess is spawned) -- every scale decision (Top/Side fovy, 3D distance, axis
    arrow length) reads this ONE shared value, not this robot's own bbox, so panels are directly
    size-comparable across both rows and columns instead of each auto-fitting independently.

    ONE (robot, view) pair per process, not all 3 views (or all 6 robots) in one loop -- isolated by
    bisection: a renderer holding CUSTOM geoms (axis arrows in extra slots past `model.ngeom`),
    closed, followed by a SECOND `mujoco.Renderer` construction in the SAME process -- for the SAME
    model, regardless of reuse-vs-fresh-instance, regardless of `mjv_connector` vs. manually
    computed geom pos/mat (both tested) -- segfaults deterministically (5/5) on that second
    renderer's `.render()` call inside `mujoco.mjr_render`. A renderer that never touches any geom
    slot past `model.ngeom` does not have this problem (also verified, 3/3 fresh renderers with only
    the model's own static geoms). So the working regime -- proven by `_rv3d_render_worker`, which
    touches custom geom slots on ONE renderer used for hundreds of frames, closed exactly once at
    the very end -- is "at most one CUSTOM-geom-touching renderer lifetime per process." Three
    views x custom axis geoms means three per-process render calls here, hence one subprocess per
    view instead of per robot.
    """
    import mujoco


    axis_rgba = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]  # X, Y, Z

    model, home_pos, _ov, visual_mask_fn = _overlay_assets(robot)
    visual = visual_mask_fn(model)
    model.geom_rgba[~visual, 3] = 0.0
    d = mujoco.MjData(model)
    set_home_pose(model, d, pose_overrides=_ARMS_DOWN_POSE.get(robot, {}), home_pos=home_pos)
    mujoco.mj_forward(model, d)

    # "Base link" = the root body directly under world (mjOBJ world is body 0); the last such body
    # if more than one is world-parented, matching how a floating-base humanoid's actual
    # torso/pelvis is typically the final one added after any world-fixed fixtures.
    base_id = int(np.flatnonzero(model.body_parentid == 0)[-1])
    base_world_pos = d.xpos[base_id].copy()
    base_world_mat = d.xmat[base_id].reshape(3, 3).copy()

    xmat = d.geom_xmat[visual].reshape(-1, 3, 3)
    he = np.einsum("nij,nj->ni", np.abs(xmat), model.geom_size[visual])
    bbox_lo = np.min(d.geom_xpos[visual] - he, axis=0)
    bbox_hi = np.max(d.geom_xpos[visual] + he, axis=0)
    bbox_center = 0.5 * (bbox_lo + bbox_hi)
    # `shared_bbox_radius` (max over all 6 robots, computed once by the parent) drives EVERY scale
    # decision below -- Top/Side fovy (world-metres spanned by the frame) AND the 3D row's distance
    # fit all read the SAME value, not each robot's own bbox or each row's own axis extent. Top and
    # Side are orthographic projections of the SAME object and have to share a scale to be read
    # AS projections of each other (a mechanical-drawing convention: 1 m reads as the same pixel
    # span in both), and the 3D row needs to match that same apparent size too, not auto-zoom
    # independently -- otherwise the grid looks like every robot/view was individually cropped to
    # fill its panel, which defeats a size comparison across columns AND across rows.
    diameter = 2.0 * shared_bbox_radius * 1.3  # 30% margin, matches the old per-row fit's own margin
    # Sized off the SHARED radius too (a fixed 0.35 m was invisible against ToddlerBot-scale robots
    # and barely visible against TALOS-scale ones; a per-robot radius would make the arrows
    # themselves incomparable across columns the same way the old per-robot camera fit did).
    axis_len, axis_rad = 0.55 * shared_bbox_radius, 0.015 * shared_bbox_radius

    res = 500
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, res)
    model.vis.global_.offheight = max(model.vis.global_.offheight, res)
    site_off = mujoco.MjvOption(); site_off.sitegroup[:] = 0

    _view_name, az, el, ortho = _ORIENTATION_DEBUG_VIEWS[view_idx]
    if az is None:  # 3D row: per-robot front-facing azimuth, matching `_rv3d_render_worker`
        az = _robot_forward_azimuth(model, d, robot) + 20.0
    model.vis.global_.orthographic = 1 if ortho else 0
    model.vis.global_.fovy = diameter if ortho else 45.0
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = bbox_center
    cam.distance = shared_bbox_radius / np.tan(np.radians(45.0) / 2.0) * 1.3 if not ortho else 3.0
    cam.azimuth = az; cam.elevation = el; cam.orthographic = 1 if ortho else 0
    renderer = mujoco.Renderer(model, res, res, max_geom=model.ngeom + 13)
    renderer.update_scene(d, camera=cam, scene_option=site_off)
    scn = renderer.scene
    scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    base = model.ngeom
    for k in range(3):
        geom = scn.geoms[base + k]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array(axis_rgba[k], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, axis_rad,
                             base_world_pos, base_world_pos + axis_len * base_world_mat[:, k])
    scn.ngeom = base + 3
    frame = renderer.render()
    renderer.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, frame=frame)


def _render_orientation_debug() -> None:
    """Debug aid, NOT a shipped figure: one robot per column (the 6 `--reach-visible-compare`
    robots), rendered in their home/rest pose with NO voxels, base-link XYZ axes overlaid
    (red=X, green=Y, blue=Z), across the SAME three camera views `--reach-visible-compare
    --rv-video-3d` uses for its Top/Side/3D rows. Exists because a suspected 180-degree-about-Z
    orientation mismatch between robots (apptronik_apollo, fourier_gr3, possibly others) was too
    subtle to read off the voxel silhouette alone -- the axis colors make each robot's own
    "forward"/"left"/"up" convention explicit at a glance instead. Writes
    `result/orientation_debug.png`. Each robot's rendering happens in its own retried subprocess --
    see `_orientation_debug_worker`'s docstring for why.
    """
    import subprocess
    import tempfile

    import mujoco


    robots = ["humanoid_v21", "unitree_g1", "booster_t1", "apptronik_apollo", "fourier_gr3", "pal_talos"]
    names = ["Ours (actuated)", "G1", "Booster T1", "Apptronik Apollo", "Fourier GR-3", "PAL TALOS"]
    views = _ORIENTATION_DEBUG_VIEWS

    # Shared scale across every column, computed ONCE here (cheap: forward kinematics only, no
    # rendering) and passed into every worker -- each worker used to auto-fit ITS OWN robot's
    # bounding box, so a small robot (ToddlerBot-scale) and a large one (TALOS-scale) filled the
    # same panel size, making the grid look like they're comparable size when they aren't. The MAX
    # bounding radius across all 6 becomes the one shared "zoom" for camera distance/fovy AND axis
    # arrow length, so panel-to-panel size differences are real, not an artifact of independent
    # auto-fitting. Each robot still LOOKS AT its own bbox center (only the zoom is shared).
    shared_bbox_radius = 0.0
    for robot in robots:
        model, home_pos, _ov, visual_mask_fn = _overlay_assets(robot)
        visual = visual_mask_fn(model)
        d = mujoco.MjData(model)
        set_home_pose(model, d, pose_overrides=_ARMS_DOWN_POSE.get(robot, {}), home_pos=home_pos)
        mujoco.mj_forward(model, d)
        xmat = d.geom_xmat[visual].reshape(-1, 3, 3)
        he = np.einsum("nij,nj->ni", np.abs(xmat), model.geom_size[visual])
        bbox_lo = np.min(d.geom_xpos[visual] - he, axis=0)
        bbox_hi = np.max(d.geom_xpos[visual] + he, axis=0)
        shared_bbox_radius = max(shared_bbox_radius, 0.5 * float(np.max(bbox_hi - bbox_lo)))

    tmp_dir = Path(tempfile.mkdtemp(prefix="orientdbg_"))
    worker_script = str(Path(__file__).resolve())
    max_attempts = 10
    results = [[None] * len(views) for _ in robots]
    for ci, robot in enumerate(robots):
        for vi in range(len(views)):
            out_path = tmp_dir / f"{robot}_{vi}.npz"
            worker_cmd = [sys.executable, worker_script,
                         "--orientation-debug-worker-robot", robot,
                         "--orientation-debug-worker-view", str(vi),
                         "--orientation-debug-worker-bbox-radius", str(shared_bbox_radius),
                         "--orientation-debug-worker-out", str(out_path)]
            for attempt in range(1, max_attempts + 1):
                proc = subprocess.run(worker_cmd)
                if proc.returncode == 0 and out_path.is_file():
                    break
                print(f"orientation-debug worker ({robot}, view {vi}) attempt {attempt}/"
                      f"{max_attempts} failed (exit {proc.returncode}), retrying...")
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(f"_orientation_debug_worker failed {max_attempts} times in a "
                                   f"row for {robot} view {vi}")
            results[ci][vi] = np.load(out_path)["frame"]
    shutil.rmtree(tmp_dir, ignore_errors=True)

    fig, axes = plt.subplots(len(views), len(robots), figsize=(3.0 * len(robots), 3.2 * len(views)))
    for ci, name in enumerate(names):
        for ri, (view_name, _az, _el, _ortho) in enumerate(views):
            ax = axes[ri, ci]
            ax.imshow(results[ci][ri])
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(name, fontsize=11)
            if ci == 0:
                ax.set_ylabel(view_name, fontsize=9)
    fig.suptitle("Base-link XYZ axes: red=X, green=Y, blue=Z", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULT_DIR / "orientation_debug.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def _render_vrw_3d(
    alpha: float,
    stride: int,
    azimuth: float,
    elevation: float,
    distance: float,
    vis_cmap: str,
    blind_cmap: str,
) -> None:
    """Render teaser panel (b): the V2 visible-reachable slab in 3D, fixed and actuated gimbals.

    Writes a two-column figure (`teaser_vrw_3d.{png,pdf}`) plus the two source renders
    (`teaser_vrw_3d_{fixed,actuated}.png`) under `result/`. Both columns share camera, cut, pose
    and framing, so the ONLY difference between them is which voxels are visible.

    Design decisions, and what breaks if they are changed:

    * Colors come from `_greens_reds_rgba` and the colorbars from `_shaded_cmap`, both over
      `Normalize(0, _DEXTERITY_VMAX)` with the compare figure's `plasma`/`Blues` pair -- so a voxel
      of a given reachability index R is the same color here as in `fig:workspace`, and the bar
      tick at "0" is exactly the color of an R=0 voxel. Hand-rolling either would desynchronise
      the teaser from Fig. 2.
    * Body is drawn at `_ARMS_DOWN_POSE`, which is the pose `_camera_visibility` raycasts against.
      Drawing `_OVERLAY_POSE` (arms out, as the reachability figure does) would show a body that
      never produced these blind voxels.
    * The slab is the `z <= shoulder-center` half-space of `_shoulder_section_payload_z`, the same
      cut as the compare figure's top row, so the two figures agree on what is shown as well as
      how it is colored.
    * Rendered `transparent=False`, i.e. over the model's WHITE skybox. A translucent voxel blends
      against whatever is behind it, so on the transparent path (skybox off, dark backdrop) every
      alpha below ~0.4 darkened the hues into mud. White backdrop is what makes a low `alpha`
      usable, and it matches the paper page the panel lands on.
    * `alpha` < 1 lets the interior read through the outer shell. MuJoCo sorts translucent geoms
      per frame, so a dense cloud can blend imperfectly; `stride` thins it if that shows, at the
      cost of a gappier cloud -- stride 1 (every voxel) is the quality setting.
    """
    import mujoco
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize


    case = _reach_visible_case("v2")
    cut_z = _shoulder_section_payload_z("humanoid_v21")
    sel = case["reached"] & (case["points"][:, 2] <= cut_z)
    if stride > 1:
        sel &= _grid_stride_mask(case["points"], stride)
    points, values = case["points"][sel], case["dexterity"][sel]

    model, home_pos, _overlay_pose, visual_fn = _overlay_assets("humanoid_v21")
    model.geom_rgba[~visual_fn(model), 3] = 0.0  # hide collision proxies / camera frusta
    data = mujoco.MjData(model)
    base_pos = set_home_pose(
        model, data, pose_overrides=_ARMS_DOWN_POSE["humanoid_v21"], home_pos=home_pos,
    )
    mujoco.mj_forward(model, data)
    world = points.astype(np.float64) + base_pos

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # Frame on the BODY, not the cloud centroid: the cloud is asymmetric between the two columns
    # (visible vs blind voxels shift its XY extent), and the body is what the reader anchors on.
    # Centering on the cloud would let the head wander between the two panels.
    cam.lookat[:] = base_pos
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation

    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    # "cameras", not "gimbals": the paper's figure-facing vocabulary is "camera articulation" /
    # "fixed camera" / "actuated camera" (abstract, Fig. 2 headings, Table IV). Gimbal is the
    # mechanism and stays in the hardware prose.
    for tag, title, visible in (
        ("fixed", "Fixed cameras", case["vis_fixed"][sel]),
        ("actuated", "Actuated cameras", case["vis_steer"][sel]),
    ):
        rgba = _greens_reds_rgba(values, visible, alpha=alpha,
                                 vis_cmap=vis_cmap, blind_cmap=blind_cmap)
        png = _export_viewer_snapshot(
            model, data, cam, world.shape[0], world, rgba,
            sphere_r=0.5 * 0.02 * max(stride, 1),
            out_path=_RESULT_DIR / f"teaser_vrw_3d_{tag}.png",
            transparent=False, geom_type="box",
            resolution=(3840, 2160),
        )
        panels.append((title, _crop_white(plt.imread(str(png)))))
        print(f"{tag}: {int(visible.sum())}/{visible.size} slab voxels visible "
              f"({100.0 * visible.mean():.1f}%)")

    # One axes holding both renders side by side, rather than two subplots. `imshow` locks the
    # aspect, so with two axes the leftover width becomes dead space INSIDE each axes and no
    # `wspace` can close it. Concatenating first makes the gutter an explicit pixel count.
    gutter = int(0.05 * max(img.shape[1] for _t, img in panels))
    h = max(img.shape[0] for _t, img in panels)
    strips = []
    for i, (_title, img) in enumerate(panels):
        pad = np.ones((h - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
        strips.append(np.vstack([img, pad]) if pad.shape[0] else img)
        if i == 0:
            strips.append(np.ones((h, gutter, img.shape[2]), dtype=img.dtype))
    joined = np.hstack(strips)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.imshow(joined)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.26)
    # Cropped panels can differ in width by a pixel or two, so derive each column's data-x range
    # from the hstack geometry and map to figure coords via the axes' left/right. Anchored to the
    # BODY's horizontal center, not the ink-bbox center: the cloud's visible/blind mask shifts its
    # silhouette asymmetrically, so a title on the ink-bbox lands ~3% left of the head.
    panel_widths = [img.shape[1] for _t, img in panels]
    x_starts = [0]
    for w in panel_widths[:-1]:
        x_starts.append(x_starts[-1] + w + gutter)
    axes_left, axes_right = 0.0, 1.0
    span = axes_right - axes_left
    for i, ((title, img), x_start, w) in enumerate(zip(panels, x_starts, panel_widths)):
        dark = (img[:, :, :3] < 0.25).all(axis=2)
        body_cols = np.where(dark.any(axis=0))[0]
        body_center = 0.5 * (body_cols[0] + body_cols[-1]) if body_cols.size else 0.5 * w
        center = axes_left + (x_start + body_center) / joined.shape[1] * span
        # Slight outward nudge so the two titles don't read as one block over the body.
        center += (0.04, -0.04)[i]
        fig.text(center, 0.93, title, ha="center", fontsize=11)
    norm = Normalize(0.0, _DEXTERITY_VMAX)
    for i, (cmap, label) in enumerate(((vis_cmap, "Visible-reachable"),
                                       (blind_cmap, "Blind-reachable"))):
        cax = fig.add_axes((0.14 + 0.42 * i, 0.14, 0.30, 0.035))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=_shaded_cmap(cmap)), cax=cax,
                           orientation="horizontal")
        bar.set_ticks((0.0, _DEXTERITY_VMAX))
        bar.set_ticklabels(("0", f"{_DEXTERITY_VMAX:g}"))
        bar.ax.tick_params(labelsize=8, length=3, pad=2)
        bar.set_label(label, fontsize=9, labelpad=2)
    fig.text(0.5, 0.22, "Reachability index $R$", ha="center", fontsize=9)
    path_base = _RESULT_DIR / "teaser_vrw_3d"
    fig.savefig(path_base.with_suffix(".png"), dpi=600)
    fig.savefig(path_base.with_suffix(".pdf"), dpi=600)
    plt.close(fig)
    print(f"saved {path_base.with_suffix('.png')} and {path_base.with_suffix('.pdf')}")


def _render_vrw_video(
    robot_key: str,
    alpha: float,
    stride: int,
    tilt: float,
    elevation: float,
    distance: float,
    vis_cmap: str,
    blind_cmap: str,
    steps_per_layer: int,
    fps: int,
    hold_frames: int,
    res: int,
) -> None:
    """Animate `robot_key`'s visible-reachable slab in two chained sweep phases, no pause between
    them: (1) Z, lowest voxel to highest voxel (world-up), growing from nothing to the full cloud;
    (2) world-X, a YZ-plane cut pushed from the front of the robot to the back (direction set by the
    actual camera position) -- starts at phase 1's full cloud and CUTS AWAY as the plane passes each
    voxel, ending empty, not a second reveal (see `_sweep_axis` mode="erase").

    Same arms-down pose and Greens/Blues color mapping as `_render_vrw_3d` (one column instead of
    the fixed-vs-actuated pair, animated instead of a fixed shoulder-height cut), so a still frame
    from this video is color-identical to the paper teaser. Camera framing DIFFERS from
    `_render_vrw_3d` in two ways. First, distance: that teaser only ever shows the z<=shoulder half,
    so a fixed distance tuned for it clips the FULL cloud shown here (whose top can be well above
    the head, at max overhead reach) -- `lookat`/`distance` are instead solved from the cloud's own
    bounding box every call, centered on its midpoint (not the root) and backed off until its
    bounding radius fits the (square, so fovx==fovy) frame with margin, so framing stays correct
    across robots of very different reach envelopes without per-robot hand-tuned constants. Second,
    azimuth: `--vrw-3d` uses one fixed world angle; this instead solves each robot's own front-facing
    azimuth from its head-camera geometry (`_robot_forward_azimuth`) and applies `tilt` as an offset
    from it -- measured, `humanoid_v21`/`unitree_g1`/`toddlerbot`/`apptronik_apollo` face world +X but
    `booster_t1`/`fourier_gr3`/`pal_talos` face -X, so one fixed azimuth necessarily shows some of
    them from the back. `distance` is still a floor (`max(distance, auto_fit_distance)`), never a
    clip.

    Reuses ONE `mujoco.Renderer` across both phases (unlike `_export_viewer_snapshot`, which builds
    a fresh renderer + PNG per call) since this loop calls it hundreds of times. Writes
    `result/vrw_video_<robot_key>.mp4`.
    """
    import mujoco
    import imageio.v2 as imageio


    cfg = _ROBOTS[robot_key]
    case = _reach_visible_case(robot_key)
    if cfg.visibility_kind == "dynamic":
        visible = case["visible"]
    else:
        visible = case["vis_steer"] if cfg.visibility_kind == "steered" else case["vis_fixed"]

    sel = case["reached"]
    if stride > 1:
        sel = sel & _grid_stride_mask(case["points"], stride)
    points, values, visible = case["points"][sel], case["dexterity"][sel], visible[sel]

    model, home_pos, _overlay_pose, visual_fn = _overlay_assets(cfg.model_key)
    model.geom_rgba[~visual_fn(model), 3] = 0.0  # hide collision proxies / camera frusta
    data = mujoco.MjData(model)
    base_pos = set_home_pose(
        model, data, pose_overrides=_ARMS_DOWN_POSE.get(cfg.model_key, {}), home_pos=home_pos,
    )
    mujoco.mj_forward(model, data)
    world = points.astype(np.float64) + base_pos
    rgba = _greens_reds_rgba(values, visible, alpha=alpha, vis_cmap=vis_cmap, blind_cmap=blind_cmap)

    half_edge = 0.5 * 0.02 * max(stride, 1)  # voxel box half-extent (box edge == grid spacing)
    steps_per_layer = max(1, steps_per_layer)
    step_fracs = np.arange(1, steps_per_layer + 1) / steps_per_layer  # (0, 1], "half point" == [0.5, 1]

    # Fit the camera to the FULL cloud's bounding box (not just the root/lookat point), since the
    # cloud's own vertical extent (feet to overhead reach) is what was clipping at the top. Square
    # `res` makes fovx == fovy, so one bounding radius covers both frame axes.
    bbox_lo, bbox_hi = points.min(axis=0), points.max(axis=0)
    bbox_center = 0.5 * (bbox_lo + bbox_hi)
    bbox_radius = 0.5 * float(np.max(bbox_hi - bbox_lo))
    fovy_deg = model.vis.global_.fovy if model.vis.global_.fovy > 0 else 45.0
    fit_distance = bbox_radius / np.tan(np.radians(fovy_deg) / 2.0) * 1.25  # 25% margin

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = base_pos + bbox_center
    cam.distance = max(distance, fit_distance)
    cam.azimuth = _robot_forward_azimuth(model, data, cfg.model_key) + tilt
    cam.elevation = elevation

    w = h = res
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    renderer = mujoco.Renderer(model, h, w, max_geom=points.shape[0] + model.ngeom + 1000)
    box = np.array([half_edge] * 3, dtype=np.float64)
    eye = np.eye(3, dtype=np.float64).flatten()
    # Unsized sites (camera module's `cam_*_rgb`/`cam_*_cam_front_center`) default to group 0,
    # which `MjvOption`'s default `sitegroup` renders as small spheres -- two stray dots beside
    # each head-camera housing. Off entirely; see `_export_viewer_snapshot` for the same fix.
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = 0

    # Which world-X extreme is nearer the camera? Read the ACTUAL stereo-averaged eye position off
    # the scene (one throwaway update_scene) rather than hand-deriving it from azimuth/elevation --
    # robust to whatever camera convention MuJoCo uses internally.
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    cam_world_pos = np.mean([np.array(c.pos) for c in renderer.scene.camera], axis=0)
    probe_lo = base_pos + np.array([bbox_lo[0], bbox_center[1], bbox_center[2]])
    probe_hi = base_pos + np.array([bbox_hi[0], bbox_center[1], bbox_center[2]])
    x_near_is_lo = np.linalg.norm(cam_world_pos - probe_lo) < np.linalg.norm(cam_world_pos - probe_hi)

    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULT_DIR / f"vrw_video_{cfg.cli_key}.mp4"
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8,
                                output_params=["-preset", "veryfast"])

    # `renderer.update_scene()` only resets `scn.ngeom` to the model's base geom count -- the extra
    # (voxel) geom SLOTS keep whatever pos/size/rgba they were last written with (confirmed: writing
    # a slot, calling update_scene again, and re-including that index reads back the old write).
    # That means a slot only needs (a) ONE `mjv_initGeom` the first frame it appears, and (b) a
    # pos/size touch on the frames its own frac is still <1 -- total Python-side geom-touches is
    # then O(voxels + steps_per_layer * layers) instead of O(voxels * total frames) (every
    # already-fully-grown voxel re-touched every single frame). Each phase below resets `n_added`
    # to 0 and re-inits every slot from scratch, since a different sweep axis means a different sort
    # order -- slot i names a different voxel each phase.
    slot_cap = renderer.scene.maxgeom - model.ngeom
    base = model.ngeom
    rgb = None
    phase_stats = []

    def _sweep_axis(axis: int, ascending: bool, mode: str, label: str) -> None:
        """Sweep a cutting plane along world-frame `axis` (0=X, 1=Y, 2=Z) from its `ascending`-True
        low extreme (or high extreme if False) to the opposite one, half-point SDF interpolated per
        voxel layer along that axis.

        `mode="grow"`: starts empty; a voxel's near (entry) face is fixed and its far face grows in
        as the plane reaches it -- reveals from nothing to the full cloud. Used for phase 1 (Z,
        lowest to highest voxel).
        `mode="erase"`: starts with every voxel already at full size (the state `mode="grow"` left
        behind); a voxel's far face is fixed and its near face retreats as the plane reaches it --
        "whatever is still ahead of the plane stays visible, whatever it has already passed is cut
        away." Used for phase 2 (X, nearest to farthest voxel from the actual camera position: a
        YZ-plane cut pushed front to back, chained directly onto phase 1's last frame with no reset
        and no static pause in between).
        """
        nonlocal rgb
        sign = 1.0 if ascending else -1.0
        # Rounded up front, not just for `layers` -- the raw per-voxel coordinate (from the
        # aggregated/mirrored cache) carries ~1e-5 float noise around its nominal grid value, while
        # `cutoffs` below is built from `layers`' ROUNDED, deduplicated values. Comparing raw
        # `top_sorted`/`bottom_sorted` against those cutoffs left every voxel's threshold off by
        # ~1e-5 -- far past `_FULL_EPS` (tuned for ~1e-16 double-precision noise, not float32-scale
        # data noise) -- so `n_full` silently stalled partway through the sweep on real payloads
        # (confirmed: 106,389/175,237 voxels on v2's X axis stuck at their halfway frac forever,
        # some from the very FIRST layer). Rounding both sides the same way removes the mismatch at
        # its source; the ~1e-5 m position error this introduces in `world_sorted` is far sub-pixel.
        coord = np.round(sign * points[:, axis].astype(np.float64), 6)
        order = np.argsort(coord, kind="stable")
        c_sorted, world_sorted, rgba_sorted = coord[order], world[order], rgba[order]
        bottom_sorted = c_sorted - half_edge  # near (entry) face in the sweep direction
        top_sorted = c_sorted + half_edge  # far face in the sweep direction
        n = c_sorted.shape[0]
        layers = np.unique(c_sorted)
        cutoffs = ((layers[:, None] - half_edge) + step_fracs[None, :] * (2.0 * half_edge)).ravel()

        def _frac_and_coord(idx: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
            if mode == "grow":  # near face fixed, far face grows in: frac 0 -> 1
                frac = np.clip((cutoff - bottom_sorted[idx]) / (2.0 * half_edge), 0.0, 1.0)
                return frac, bottom_sorted[idx] + half_edge * frac
            frac = np.clip((top_sorted[idx] - cutoff) / (2.0 * half_edge), 0.0, 1.0)  # far fixed: 1 -> 0
            return frac, top_sorted[idx] - half_edge * frac

        def _write(i: int, frac: float, c_eff: float, geom) -> None:
            geom.size[axis] = half_edge * frac
            geom.pos[axis] = world_sorted[i, axis] + sign * (c_eff - c_sorted[i])

        n_cap = min(n, slot_cap)
        # Erase mode assigns slots in REVERSE math-index order (math-index 0 = front = erased
        # first -> slot n_cap-1; math-index n_cap-1 = back = erased last -> slot 0), so the
        # not-yet-erased set [n_full, n_cap) always occupies the LOW, CONTIGUOUS slot range
        # [0, n_cap - n_full). That lets `scn.ngeom` actually SHRINK as voxels are cut away
        # (excluding fully-erased slots from rendering entirely) instead of staying at n_cap for
        # the whole phase with most boxes merely zero-sized -- confirmed this rendered ~2x the
        # total boxes of the grow phase (which ramps 0->n_cap) since it stayed flat at n_cap the
        # entire time. Grow mode keeps direct index==slot (its own ngeom already ramps correctly).
        def _slot(i: int) -> int:
            return base + (n_cap - 1 - i) if mode == "erase" else base + i

        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        scn = renderer.scene
        if mode == "erase":
            # Every voxel starts fully visible (frac=1) -- one upfront O(n) init pass, matching the
            # state `mode="grow"` ends in, instead of re-deriving it frame by frame.
            for i in range(n_cap):
                geom = scn.geoms[_slot(i)]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, box, world_sorted[i], eye,
                                    rgba_sorted[i])
                geom.emission, geom.specular = 0.4, 0.0
            # (`scn.ngeom` gets reset by the next `update_scene()` call regardless -- restored
            # inside the per-frame loop below, every frame.)
        n_added = 0  # `mode="grow"` only: highest voxel index ever assigned a geom slot so far

        for cutoff in cutoffs:
            n_visible = min(int(np.searchsorted(bottom_sorted, cutoff, side="right")), slot_cap)
            # `cutoffs` is built from `layers` (rounded, deduplicated) while `top_sorted` is built
            # from `c_sorted` (raw, per-voxel) -- mathematically the same value at a layer's own
            # final step, but reached via different floating-point paths, so exact equality can land
            # 1 ULP either side. An exact `side="right"` boundary then sometimes counts a voxel as
            # "already full" on the very cutoff meant to WRITE it full, skipping that write forever
            # (confirmed: left every layer permanently stuck at its half-point frac). `-_FULL_EPS`
            # makes the boundary lenient by a margin far above float noise (~1e-16) and far below
            # the smallest real step (>= half_edge), so the layer's own last cutoff always still
            # lands in the frontier and gets its exact terminal value written.
            n_full = min(int(np.searchsorted(top_sorted, cutoff - _FULL_EPS, side="right")), n_visible)

            renderer.update_scene(data, camera=cam, scene_option=scene_option)
            scn = renderer.scene
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1  # opaque white, matches _render_vrw_3d

            if mode == "grow":
                # Frontier: slots already added but still short of full size -- touch pos/size only.
                idx = np.arange(n_full, min(n_added, n_visible))
                if idx.size:
                    frac, c_eff = _frac_and_coord(idx, cutoff)
                    for k, i in enumerate(idx):
                        _write(i, frac[k], c_eff[k], scn.geoms[base + i])
                # New this frame: first appearance -- full init (type/mat/rgba never touched again).
                idx = np.arange(n_added, n_visible)
                if idx.size:
                    frac, c_eff = _frac_and_coord(idx, cutoff)
                    for k, i in enumerate(idx):
                        size_i, pos_i = box.copy(), world_sorted[i].copy()
                        geom = scn.geoms[base + i]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, size_i, pos_i, eye,
                                            rgba_sorted[i])
                        geom.emission, geom.specular = 0.4, 0.0
                        _write(i, frac[k], c_eff[k], geom)
                n_added = max(n_added, n_visible)
                scn.ngeom = base + n_added
            else:
                # Frontier band only -- everything below n_full is fully erased (excluded from
                # `scn.ngeom` below, not individually zeroed); everything at/above n_visible is
                # still full size from the upfront init, untouched until the plane reaches it.
                idx = np.arange(n_full, n_visible)
                if idx.size:
                    frac, c_eff = _frac_and_coord(idx, cutoff)
                    for k, i in enumerate(idx):
                        _write(i, frac[k], c_eff[k], scn.geoms[_slot(i)])
                # Reversed slot order (see `_slot`) puts the still-visible set [n_full, n_cap) at
                # the LOW, contiguous slots [0, n_cap - n_full) -- shrink ngeom to exactly that
                # range so fully-erased voxels stop costing render time, instead of staying in the
                # scene at zero size for the rest of the phase.
                scn.ngeom = base + (n_cap - n_full)

            rgb = renderer.render()
            writer.append_data(rgb)
        if mode == "erase":
            # A slot at exactly frac=0 has size 0 along `axis` but STILL full size on the other two
            # -- a fully cut box is a flat plate (zero thickness, full width/height), not a point, so
            # it's still visible face-on unless also excluded from `scn.ngeom`. The per-frame loop's
            # `n_full` (epsilon-bounded so a layer's own last cutoff isn't excluded before it can be
            # written -- see above) can end the phase a few voxels short of `n_cap` for exactly the
            # LAST layer, since there's no subsequent cutoff left to let it catch up. Confirmed on
            # v2 (X axis): 56 of the farthest, most isolated voxels lingered as visible flat plates
            # this way (nothing nearby to hide their now-degenerate silhouette). One extra frame,
            # past every voxel's far face, forces the true count and gives a genuinely empty final
            # frame for the hold/loop-back to repeat.
            n_full_final = int(np.searchsorted(top_sorted, top_sorted.max() + half_edge, side="right"))
            renderer.update_scene(data, camera=cam, scene_option=scene_option)
            scn = renderer.scene
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
            scn.ngeom = base + (n_cap - min(n_full_final, n_cap))
            rgb = renderer.render()
            writer.append_data(rgb)
        phase_stats.append(f"{label}: {layers.size} layers x {steps_per_layer} steps = {cutoffs.size} frames")

    # Phase 1: lowest voxel to highest voxel (Z, world-up), revealing from nothing.
    _sweep_axis(axis=2, ascending=True, mode="grow", label="Z low->high")
    # Phase 2: YZ-plane cut pushed from the front of the robot to the back (direction set by the
    # actual camera position) -- cuts away from the full cloud phase 1 leaves behind, chained with
    # no reset and no static pause in between.
    _sweep_axis(axis=0, ascending=x_near_is_lo, mode="erase", label="X front->back cut")

    for _ in range(hold_frames):  # pause on the final frame only, instead of ending abruptly
        writer.append_data(rgb)
    writer.close()
    renderer.close()
    print(f"saved {out_path} ({'; '.join(phase_stats)}; + {hold_frames} hold @ {fps} fps, "
          f"{points.shape[0]} voxels)")


def _crop_white(img: np.ndarray) -> np.ndarray:
    """Trim the uniform white border a MuJoCo snapshot leaves around the rendered content.

    The offscreen render is a fixed 16:9 frame while the slab is a wide, short volume, so roughly
    half of each panel is margin. Cropping here rather than by tuning the camera keeps both columns
    on the identical camera (the comparison's whole point) while still packing them tightly.
    """
    ink = (img[:, :, :3] < 0.99).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if rows.size == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _render_reachable_slab_3d(
    points: np.ndarray,
    values: np.ndarray,
    reached: np.ndarray,
    cutoff_z: float,
    out_path: Path,
    azimuth: float = -60.0,
    elevation: float = -25.0,
) -> None:
    """Render the z<=cutoff_z reachable slab as colored voxel boxes WITH the welded robot.

    Kept for reuse: NOT wired into the default section figure. It was trialled as panel (d) then
    dropped because a 3D slab of the same z-cut duplicates the top-view information. Call directly
    to produce a standalone 3D perspective of the reachable set (all D, depth-correct occlusion
    against the mesh). `points` are voxel centers in the base frame, `values` per-voxel reachability index
    (may be the bimanual max), `reached` the reachable mask. Writes `out_path` (PNG, transparent
    background). Same mujoco offscreen path as the 2D panels, so style matches.
    """
    import mujoco

    from mj_envs.asset_zoo.reachability_study.generate_workspace_curobo import _load_welded_model

    cmap = plt.colormaps["RdYlGn"].copy()
    model = _load_welded_model()
    visual = np.array([
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_visual")
        for i in range(model.ngeom)
    ])
    model.geom_rgba[~visual, 3] = 0.0
    data = mujoco.MjData(model)
    base_pos = set_home_pose(model, data, pose_overrides=_OVERLAY_POSE)
    mujoco.mj_forward(model, data)

    axis_x = np.unique(points[:, 0])
    half = 0.5 * float(np.min(np.diff(axis_x))) if axis_x.size > 1 else 0.01
    box = np.array([half, half, half], dtype=np.float64)
    xmat_eye = np.eye(3).flatten()

    sel = reached & np.isfinite(values) & (points[:, 2] <= cutoff_z)
    sel_pos = points[sel] + base_pos  # base frame -> world (robot mesh is in world)
    sel_col = cmap(np.clip(values[sel] / _DEXTERITY_VMAX, 0.0, 1.0)).astype(np.float32)

    lo = points[reached].min(axis=0)
    hi = points[reached].max(axis=0)
    view_center = base_pos + 0.5 * (lo + hi)
    cres = 1400
    model.vis.global_.orthographic = 0
    model.vis.global_.fovy = 45
    model.vis.global_.offwidth = cres
    model.vis.global_.offheight = cres
    renderer = mujoco.Renderer(model, cres, cres, max_geom=int(sel.sum()) + model.ngeom + 1000)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = view_center
    camera.distance = 2.4
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera)
    scn = renderer.scene
    for p, c in zip(sel_pos, sel_col):
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, box, p, xmat_eye, c)
        scn.ngeom += 1
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    renderer.close()
    alpha = np.where(depth >= depth.max() - 1e-6, 0, 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.dstack((rgb, alpha)))
    ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, transparent=True)
    plt.close(fig)
    print(f"saved {out_path}")


def _symmetry_to_grid(agg: dict) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Scatter per-voxel D / reachability into dense per-axis (nx,ny,nz) arrays.

    Per-axis index comes from ranking the unique voxel_pos coordinate, so it needs no payload
    grid internals and works for both the legacy cubic grid AND the isotropic spacing grid
    (whose per-axis point counts differ, e.g. 17x21x17). Returns (D_grid, reach_grid,
    axis_coords) with axis_coords the ascending (X, Y, Z) meter values; unreached voxels are
    NaN (D) / False (reach).
    """
    pos = agg["voxel_pos"].astype(np.float64)
    D = agg["dexterity"].astype(np.float64)
    reach = agg["success_voxel"].astype(bool)
    coords: list[np.ndarray] = []
    idx = np.empty((pos.shape[0], 3), dtype=np.int64)
    for ax in range(3):
        c = np.round(pos[:, ax], 5)
        u = np.unique(c)
        coords.append(u)
        idx[:, ax] = np.searchsorted(u, c)
    shape = tuple(c.size for c in coords)
    Dg = np.full(shape, np.nan, dtype=np.float64)
    Rg = np.zeros(shape, dtype=bool)
    Dg[idx[:, 0], idx[:, 1], idx[:, 2]] = D
    Rg[idx[:, 0], idx[:, 1], idx[:, 2]] = reach
    return Dg, Rg, coords


def _symmetry_masked(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy with entries outside `mask` set NaN (imshow renders NaN as the cmap bad color)."""
    out = field.copy()
    out[~mask] = np.nan
    return out


def _plot_symmetry(aggR: dict, aggL: dict, out: Path) -> None:
    """L/R reachability-index workspace symmetry figure from two INDEPENDENTLY solved payloads.

    EVIDENCE (not assert) of left/right symmetry: reflect the measured L field across the
    sagittal plane (base-frame Y) and compare voxel-wise to R. Mirroring R to fake L would make
    symmetry true by construction; L is measured, so a small residual is real evidence and a
    large one flags genuine asymmetry. Quantifies reachable-envelope IoU + mean/median |ΔD| and
    renders occlusion-free orthogonal slice panels (matplotlib only, deterministic).
    """
    DR, RR, cR = _symmetry_to_grid(aggR)
    DL, RL, cL = _symmetry_to_grid(aggL)
    if DR.shape != DL.shape:
        raise ValueError(f"grid mismatch R{DR.shape} vs L{DL.shape}")
    # Grids must coincide (same bounds/side-independent) and Y must be symmetric about 0.
    for a in range(3):
        if not np.allclose(cR[a], cL[a]):
            raise ValueError(f"axis {a} coords differ between R and L payloads")
    if not np.allclose(cR[1], -cR[1][::-1], atol=1e-4):
        raise ValueError("Y axis not symmetric about 0; sagittal reflection invalid")

    # Sagittal reflection of L: reverse Y (index 1). Mirror is an involution.
    DL_ref = DL[:, ::-1, :]
    RL_ref = RL[:, ::-1, :]

    inter = RR & RL_ref
    union = RR | RL_ref
    iou = inter.sum() / max(int(union.sum()), 1)
    dsig = np.nan_to_num(DR) - np.nan_to_num(DL_ref)  # signed residual (R minus mirrored L)
    dd = np.abs(dsig)
    mean_dd = float(dd[union].mean())
    med_dd = float(np.median(dd[union]))
    print(f"reachable voxels: R={int(RR.sum())}  L(mirror)={int(RL_ref.sum())}  "
          f"inter={int(inter.sum())}  union={int(union.sum())}")
    print(f"symmetry: IoU={iou:.4f}  mean|dD|={mean_dd:.4f}  median|dD|={med_dd:.4f}")

    X, Y, Z = cR  # ascending meter values per axis
    dmax = float(np.nanmax([np.nanmax(DR), np.nanmax(DL_ref)]))
    ddmax = float(np.nanpercentile(dd[union], 99)) if union.any() else 0.1
    ddmax = max(ddmax, 1e-3)

    # Slice index per plane = densest reachable slice in R (through the shoulder band).
    jY = int(RR.sum(axis=(0, 2)).argmax())   # fix Y -> sagittal (X-Z)
    iX = int(RR.sum(axis=(1, 2)).argmax())   # fix X -> coronal  (Y-Z)
    kZ = int(RR.sum(axis=(0, 1)).argmax())   # fix Z -> transverse (X-Y)

    rdylgn = plt.get_cmap("RdYlGn").copy(); rdylgn.set_bad("white")
    diverge = plt.get_cmap("RdBu_r").copy(); diverge.set_bad("white")

    planes = [
        ("Sagittal (X-Z)", lambda g: g[:, jY, :], [X[0], X[-1], Z[0], Z[-1]], "X fwd (m)", "Z up (m)", f"Y={Y[jY]:+.2f}"),
        ("Coronal (Y-Z)",  lambda g: g[iX, :, :], [Y[0], Y[-1], Z[0], Z[-1]], "Y left (m)", "Z up (m)", f"X={X[iX]:+.2f}"),
        ("Transverse (X-Y)", lambda g: g[:, :, kZ], [X[0], X[-1], Y[0], Y[-1]], "X fwd (m)", "Y left (m)", f"Z={Z[kZ]:+.2f}"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5))
    im_v = im_d = None
    for r, (pname, slc, extent, xl, yl, at) in enumerate(planes):
        dr = _symmetry_masked(DR, RR)
        dl = _symmetry_masked(DL_ref, RL_ref)
        dv = _symmetry_masked(dsig, union)
        cols = [
            (slc(dr), rdylgn, 0.0, dmax, f"R arm\n{pname} @ {at}"),
            (slc(dl), rdylgn, 0.0, dmax, f"L arm (mirrored)\n{pname}"),
            (slc(dv), diverge, -ddmax, ddmax, f"ΔD = D_R − D_L\n{pname}"),
        ]
        for c, (field, cmap, vmin, vmax, title) in enumerate(cols):
            ax = axes[r, c]
            im = ax.imshow(field.T, origin="lower", extent=extent, cmap=cmap,
                           vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7)
            if c < 2:
                im_v = im
            else:
                im_d = im

    fig.suptitle(
        f"humanoid_v21 reachability-index workspace L/R symmetry\n"
        f"IoU={iou:.3f}   mean|ΔD|={mean_dd:.3f}   median|ΔD|={med_dd:.3f}   "
        f"(D over K={aggR['n_orientations']} SO(3) orientations; L independently solved)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=2.2)
    cax_v = fig.add_axes((0.13, 0.02, 0.45, 0.015))
    cax_d = fig.add_axes((0.68, 0.02, 0.22, 0.015))
    fig.colorbar(im_v, cax=cax_v, orientation="horizontal", label="Reachability index D (reachable orientations / K)")
    fig.colorbar(im_d, cax=cax_d, orientation="horizontal", label="D_R − D_L (mirrored)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    args = tyro.cli(Args)

    if args.rv3d_worker_in is not None:
        _rv3d_render_worker(args.rv3d_worker_in, args.rv3d_worker_out)
        return

    if args.orientation_debug_worker_robot is not None:
        _orientation_debug_worker(args.orientation_debug_worker_robot, args.orientation_debug_worker_view,
                                  args.orientation_debug_worker_bbox_radius,
                                  args.orientation_debug_worker_out)
        return

    if args.orientation_debug:
        _render_orientation_debug()
        return

    other_modes = (args.reach_visible_compare or args.reach_visible_single or args.symmetry
                   or args.view or args.camera_count_cuts)
    if args.cache and args.cache_all:
        raise ValueError("--cache and --cache-all are mutually exclusive")
    if (args.cache or args.cache_all) and other_modes:
        raise ValueError(
            "--cache[-all] cannot be combined with --reach-visible-compare/"
            "--camera-count-cuts/--reach-visible-single/--symmetry/--view"
        )

    if args.cache_all:
        written, skipped = 0, 0
        for cfg in _ROBOTS.values():
            payload_path = _CACHE_DIR / cfg.payload
            if not payload_path.is_file():
                print(f"skip {cfg.cli_key}: raw payload missing ({payload_path})")
                skipped += 1
                continue
            sidecar = _DYNAMIC_VISIBILITY_SIDECAR.get(cfg.cli_key)
            if sidecar is not None and not sidecar.is_file():
                print(f"skip {cfg.cli_key}: visibility sidecar missing ({sidecar})")
                skipped += 1
                continue
            out_path = _write_aggregated_cache(cfg)
            print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
            written += 1
        print(f"aggregated cache: {written} written, {skipped} skipped")
        return

    # Resolve `--robot` (defaults to _DEFAULT_ROBOT_KEY on bare run). If the user gave --robot AND
    # didn't override --input, point args.input at the resolved payload. If they DID override
    # --input, leave it alone -- they want a custom payload.
    if args.robot is not None and args.input == _DEFAULT_INPUT:
        cfg = _resolve_robot(args.robot.lower())
        args.input = _CACHE_DIR / cfg.payload

    if args.cache:
        robot_cfg = _resolve_robot(args.robot)
        out_path = _write_aggregated_cache(robot_cfg)
        print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
        return

    if args.vrw_3d:
        _render_vrw_3d(
            args.vrw_alpha, args.stride, args.vrw_azimuth, args.vrw_elevation, args.vrw_distance,
            args.vis_cmap, args.blind_cmap,
        )
        return

    if args.vrw_video:
        robot_cfg = _resolve_robot(args.robot)
        _render_vrw_video(
            robot_cfg.cli_key, args.vrw_alpha, args.stride, args.vrw_video_tilt,
            args.vrw_video_elevation, args.vrw_distance, args.vis_cmap, args.blind_cmap,
            args.vrw_video_steps_per_layer, args.vrw_video_fps, args.vrw_video_hold_frames,
            args.vrw_video_res,
        )
        return

    if args.reach_visible_compare:
        # CROSS-ROBOT comparison: Ours-actuated (the shipped dual rig with steerable gaze), G1 (fixed
        # head camera), Booster T1, Apptronik Apollo,
        # Fourier GR-3, PAL TALOS (per-IK-row dynamic head aim with an occlusion-aware FOV+LOS
        # mask). Each column's top cut is at that robot's own shoulder-center height so
        # silhouette/workspace scales stay comparable.
        #
        # The K=1/2/3 rigs deliberately do NOT appear here. They are an ours-internal ablation, they
        # differ from each other by ~2% of reach where these columns differ by factors, and carrying
        # them made this figure 10 columns wide with colliding tick labels. `--camera-count-cuts`
        # owns that comparison; this figure owns "ours vs other humanoids".
        v2 = _reach_visible_case("v2")
        g1 = _reach_visible_case("g1")
        t1 = _reach_visible_case("booster_t1")
        apollo = _reach_visible_case("apptronik_apollo")
        # Fourier GR-3 (head_cam = 15 deg URDF mount, the PRIMARY of its two documented pitches)
        # and PAL TALOS (head_cam = Orbbec Astra Pro RGB stream). Both are dynamic-head robots, so
        # they use their `visible` field exactly like ToddlerBot / T1 / Apollo. GR-3 once carried a
        # 16-seed payload, which retained only 98.54% of 32-seed successes and was therefore barred
        # from this cross-robot figure; the 2026-07-27 recompute re-solved every robot at 32 seeds
        # throughout, so the gate is met (see readme_reachability.md, "Recompute (2026-07-27)").
        gr3 = _reach_visible_case("fourier_gr3")
        talos = _reach_visible_case("pal_talos")
        v2_top_cutoff_z = _shoulder_section_payload_z("humanoid_v21")
        g1_top_cutoff_z = _shoulder_section_payload_z("unitree_g1")
        t1_top_cutoff_z = _shoulder_section_payload_z("booster_t1")
        apollo_top_cutoff_z = _shoulder_section_payload_z("apptronik_apollo")
        gr3_top_cutoff_z = _shoulder_section_payload_z("fourier_gr3")
        talos_top_cutoff_z = _shoulder_section_payload_z("pal_talos")
        print(f"top section: per-robot shoulder-center height "
              f"(V2 z<={v2_top_cutoff_z:.6f} m [{_shoulder_foot_referenced_z('humanoid_v21'):.3f} m foot-ref], "
              f"G1 z<={g1_top_cutoff_z:.6f} m [{_shoulder_foot_referenced_z('unitree_g1'):.3f} m foot-ref], "
              f"T1 z<={t1_top_cutoff_z:.6f} m [{_shoulder_foot_referenced_z('booster_t1'):.3f} m foot-ref], "
              f"Apollo z<={apollo_top_cutoff_z:.6f} m [{_shoulder_foot_referenced_z('apptronik_apollo'):.3f} m foot-ref])")
        # `head` counts the joints that actually aim the camera, i.e. the `head_joints` tuple each
        # robot declares in `gpu_visibility._ADAPTERS` (T1 2, Apollo 3, GR-3 2, TALOS 2), and G1's
        # zero. Apollo carries TWO cameras but on ONE neck, so it reads
        # "neck" like the monocular heads: the count that matters here is independent aim points,
        # which is 1 for every external column and 2 only for ours.
        cases = [
            {"name": "Ours (actuated)", "eta2_key": "v2", "head": "2×2-DoF gimbals", "visible": v2["vis_steer"], "instantaneous": v2.get("inst_steer"), "top_cutoff_z": v2_top_cutoff_z, **{k: v2[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}},
            {"name": "G1", "eta2_key": "g1", "head": "fixed head", "visible": g1["vis_steer"], "instantaneous": g1.get("inst_steer"), "top_cutoff_z": g1_top_cutoff_z, **{k: g1[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}},
            {"name": "Booster T1", "eta2_key": "booster_t1", "head": "2-DoF neck", "visible": t1["visible"], "top_cutoff_z": t1_top_cutoff_z, **{k: t1[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}},
            {"name": "Apptronik Apollo", "eta2_key": "apptronik_apollo", "head": "3-DoF neck", "visible": apollo["visible"], "top_cutoff_z": apollo_top_cutoff_z, **{k: apollo[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}},
            {"name": "Fourier GR-3", "eta2_key": "fourier_gr3", "head": "2-DoF neck", "visible": gr3["visible"], "top_cutoff_z": gr3_top_cutoff_z, **{k: gr3[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}},
            {"name": "PAL TALOS", "eta2_key": "pal_talos", "head": "2-DoF neck", "visible": talos["visible"], "top_cutoff_z": talos_top_cutoff_z, **{k: talos[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}},
        ]
        for cc in cases:
            n = int(cc["reached"].sum()); vv = int((cc["reached"] & cc["visible"]).sum())
            # `instantaneous` (FOV+range, no LOS raycast) is only precomputed for static-camera
            # robots (V2/G1); dynamic robots (ToddlerBot/T1/Apollo/GR-3/TALOS) have no batch
            # FOV-only sidecar, only their live-viewer slider-driven instantaneous mode.
            inst = cc.get("instantaneous")
            inst_str = (f"  instantaneous={int((cc['reached'] & inst).sum())} "
                        f"({100 * int((cc['reached'] & inst).sum()) / max(n, 1):.1f}%)") if inst is not None else ""
            print(f"{cc['name']}: reachable={n}  visible={vv} ({100 * vv / max(n, 1):.1f}%){inst_str}  "
                  f"visible_volume={vv * cc['voxel_vol']:.3f} m³  "
                  f"reachable_volume={n * cc['voxel_vol']:.3f} m³")
        _RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path_base = _RESULT_DIR / "reach_visible_compare"
        _plot_reach_visible_grid(cases, path_base, flat=args.rv_flat,
                                 vis_cmap=args.vis_cmap, blind_cmap=args.blind_cmap)
        _mirror_to_paper(path_base)
        if args.rv_video:
            _plot_reach_visible_grid(cases, path_base, flat=args.rv_flat,
                                     vis_cmap=args.vis_cmap, blind_cmap=args.blind_cmap,
                                     video=True, video_seconds=args.rv_video_seconds,
                                     video_fps=args.rv_video_fps,
                                     video_hold_frames=args.rv_video_hold_frames)
        if args.rv_video_3d:
            _plot_reach_visible_grid(cases, _RESULT_DIR / "reach_visible_compare_3d", flat=args.rv_flat,
                                     vis_cmap=args.vis_cmap, blind_cmap=args.blind_cmap,
                                     video=True, video_seconds=args.rv_video_seconds,
                                     video_fps=args.rv_video_fps,
                                     video_hold_frames=args.rv_video_hold_frames,
                                     video_3d_row=True, video_3d_stride=args.rv_video_3d_stride)
        return

    if args.camera_count_cuts:
        # Spatial rows of the paper's camera-count figure: the SAME three rigs the ablation table
        # scores. The table shows eta rising with K but cannot show WHERE, and "the blind cap above
        # the head shrinks" is otherwise a bare scalar.
        #
        # One panel per rig, three-way: seen at rest / seen only when steered / blind. An earlier
        # revision drew an actuated row over a fixed row, six panels, and made the reader difference
        # them to see what articulation contributes. The partition is exact (vis_fixed subset
        # vis_steer), so the same information fits in half the panels, and the width that frees is
        # what pays for keeping BOTH section cuts. The trade it exposes is 5.3's argument: the
        # rest-visible region grows with K (19/38/42%) while the steered-only band shrinks
        # (75/59/56%), i.e. added cameras substitute for articulation rather than adding coverage.
        k1, k2, k3 = (_reach_visible_case(k) for k in ("v2_single", "v2", "v2_triple"))
        # Every rig shares the arm, so the shoulder-height top cut is the dual rig's for all three.
        cut_z = _shoulder_section_payload_z("humanoid_v21")
        keys = ("robot", "points", "reached", "dexterity", "voxel_vol")
        cases = [
            {"name": f"$K = {i}$", "visible": c["vis_steer"], "visible_static": c["vis_fixed"],
             "top_cutoff_z": cut_z, **{k: c[k] for k in keys}}
            for i, c in enumerate((k1, k2, k3), start=1)
        ]
        for cc in cases:
            n = int(cc["reached"].sum())
            fixed = int((cc["reached"] & cc["visible_static"]).sum())
            act = int((cc["reached"] & cc["visible"]).sum())
            # The partition is only exact while vis_fixed stays a subset of vis_steer. Assert rather
            # than trust it: a violation would render as actuated-only and understate fixed.
            assert not (cc["reached"] & cc["visible_static"] & ~cc["visible"]).any(), \
                f"{cc['name']}: visible_static not a subset of visible"
            # "fixed"/"actuated" to match the figure legend, the paper, and Table IV's eta_fix/eta.
            # `actuated` is the UNION (fixed + actuated-only), which is what the panel heading's
            # second number reports; the legend's middle band is the difference.
            print(f"{cc['name']}: reachable={n:,}  fixed={fixed:,} ({100 * fixed / n:.2f}%)  "
                  f"actuated-only={act - fixed:,} ({100 * (act - fixed) / n:.2f}%)  "
                  f"blind={n - act:,} ({100 * (n - act) / n:.2f}%)  actuated={100 * act / n:.2f}%")
        print("counts are the MIRRORED whole-body set, matching --reach-visible-compare AND the "
              "ablation table, which uses the same convention as of 2026-07-29.")
        _RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path_base = _RESULT_DIR / "camera_count_cuts"
        # Three columns, BOTH section cuts. The XY shoulder cut is back because the K=2-to-3 fixed
        # gain is lateral (8,076 voxels at mean |y| = 0.39 m, only 0.4% of them near the y~0 plane),
        # so the XZ cutaway alone cannot show it and the two rigs looked identical there.
        # `mid_cmap` splits visible into rest / steered-only; see `_plot_reach_visible_grid`.
        # Palette is fixed here rather than taken from `--vis-cmap`/`--blind-cmap`, whose plasma/Blues
        # defaults suit the TWO-category comparison figure and fail at three. plasma sweeps magenta
        # to yellow, so under deuteranopia it collapses onto both other ramps (min dE 1.5 against
        # Greens) and no third choice recovers it; three single-hue ramps are needed. Ranking single-
        # hue triples by worst-case dE between ANY two shades of two categories, but that ranking is
        # overridden by inspection. Blind stays Blues, non-negotiable: Fig. 2's legend already means
        # blue = blind and the two figures are read together. Given that, YlGn / YlOrBr / Blues is
        # both the best-scoring set (43.2 normal, 24.8 deuteranopic, against plasma-Greens-Blues'
        # 43.2 / 1.5) and the one that survived visual review; warm alternatives that avoided green
        # entirely scored worse on both counts (RdPu/YlOrBr at 39.1 / 8.1) and looked garish.
        # Green is rest-visible and orange steered-only so the panels read good-to-caution-to-blind.
        _plot_reach_visible_grid(cases, path_base, flat=args.rv_flat,
                                 vis_cmap="YlGn", blind_cmap="Blues",
                                 mid_cmap="YlOrBr")
        _mirror_to_paper(path_base)
        if args.rv_video:
            _plot_reach_visible_grid(cases, path_base, flat=args.rv_flat,
                                     vis_cmap="YlGn", blind_cmap="Blues", mid_cmap="YlOrBr",
                                     video=True, video_seconds=args.rv_video_seconds,
                                     video_fps=args.rv_video_fps,
                                     video_hold_frames=args.rv_video_hold_frames)
        return

    if args.reach_visible_single:
        robot_cfg = _resolve_robot(args.robot)
        c = _reach_visible_case(robot_cfg.cli_key, args.visibility_input)
        visible = {"fixed": c.get("vis_fixed"), "steered": c.get("vis_steer"),
                   "dynamic": c.get("visible")}[robot_cfg.visibility_kind]
        instantaneous = {"fixed": c.get("inst_fixed"), "steered": c.get("inst_steer"),
                          "dynamic": None}[robot_cfg.visibility_kind]
        name = {"v2": "Ours (actuated)", "v2_fixed": "Ours (fixed)",
                "v2_single": "Ours (single, actuated)", "v2_single_fixed": "Ours (single, fixed)",
                "g1": "G1", "toddlerbot": "ToddlerBot"}.get(robot_cfg.cli_key, robot_cfg.cli_key)
        top_cutoff_z = _shoulder_section_payload_z(c["robot"])
        case = {"name": name, "visible": visible, "instantaneous": instantaneous, "top_cutoff_z": top_cutoff_z,
                **{k: c[k] for k in ("robot", "points", "reached", "dexterity", "voxel_vol")}}
        n = int(case["reached"].sum()); vv = int((case["reached"] & case["visible"]).sum())
        foot_ref = _shoulder_foot_referenced_z(c["robot"])
        print(f"top section: {name} shoulder center (payload z={top_cutoff_z:.6f} m, "
              f"foot-referenced={foot_ref:.3f} m)")
        # `instantaneous` (FOV+range, no LOS raycast) is only precomputed for static-camera
        # robots (V2/G1) -- dynamic robots have no batch FOV-only sidecar.
        inst_str = ""
        if instantaneous is not None:
            iv = int((case["reached"] & instantaneous).sum())
            inst_str = f"  instantaneous={iv} ({100 * iv / max(n, 1):.1f}%)"
        print(f"{name}: reachable={n}  visible={vv} ({100 * vv / max(n, 1):.1f}%){inst_str}")
        _RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path_base = (_RESULT_DIR / f"reach_visible_{robot_cfg.cli_key}" if args.out is None
                     else Path(args.out).with_suffix(""))
        _plot_reach_visible_grid([case], path_base, flat=args.rv_flat,
                                 vis_cmap=args.vis_cmap, blind_cmap=args.blind_cmap)
        _mirror_to_paper(path_base)
        return

    # Resolve the robot the user asked for; the payload's own "robot" field is the asset's native
    # model_loader key (informational, exposed via meta["robot"]). The CLI choice drives
    # everything: payload, model, visibility flavor.
    robot_cfg = _resolve_robot(args.robot)
    # Visibility is only ever consumed by the --view live viewer and the static --view-mode
    # visible render below; a plain reachability figure must never pay for loading/aggregating it.
    need_visible = robot_cfg.visibility_kind == "dynamic" and (args.view or args.view_mode == "visible")
    dynamic_visibility_path = (
        args.visibility_input or _DYNAMIC_VISIBILITY_SIDECAR[robot_cfg.cli_key]
        if robot_cfg.visibility_kind == "dynamic" else None
    )
    agg, dynamic, meta = _load_workspace_aggregate(
        args.input, robot_cfg, need_visible=need_visible, dynamic_visibility_path=dynamic_visibility_path,
    )
    agg_right = agg  # retain row-ID mapping for ToddlerBot's dynamic per-IK visibility sidecar

    # L aggregate: independently solved payload (--left-input) OR instant Y-mirror of R
    # (--mirror-left). left_input wins if both given.
    aggL = None
    if args.left_input is not None:
        aggL = _aggregate_by_voxel(torch.load(str(args.left_input), map_location="cpu", weights_only=False))
    elif args.mirror_left:
        aggL = _mirror_agg_l(agg, meta["n_grid_per_axis"])

    if args.symmetry:
        if aggL is None:
            raise ValueError("--symmetry requires --left-input or --mirror-left")
        out = args.out or args.input.with_name(f"{args.input.stem}_symmetry.png")
        _plot_symmetry(agg, aggL, out)
        return

    if aggL is not None:
        # True bimanual field: per shared voxel, D = max(D_R, D_L) over the two independent
        # solves (aligned by voxel_id; both use the legacy-symmetric side-independent grid).
        ids = np.union1d(agg["voxel_id"], aggL["voxel_id"])
        DR = np.full(ids.size, np.nan); DL = np.full(ids.size, np.nan)
        pos = np.full((ids.size, 3), np.nan, dtype=np.float32)
        rR = np.searchsorted(ids, agg["voxel_id"]); rL = np.searchsorted(ids, aggL["voxel_id"])
        DR[rR] = np.where(agg["success_voxel"], agg["dexterity"], np.nan)
        DL[rL] = np.where(aggL["success_voxel"], aggL["dexterity"], np.nan)
        pos[rR] = agg["voxel_pos"]; pos[rL] = aggL["voxel_pos"]  # shared grid -> same coords
        D_bi = np.fmax(DR, DL)  # nan-ignoring max: better-suited arm per voxel
        agg = {"voxel_pos": pos, "dexterity": np.nan_to_num(D_bi),
               "success_voxel": ~np.isnan(D_bi), "n_orientations": agg["n_orientations"]}
    points = agg["voxel_pos"]

    if args.view:
        # Live viewer + interactive cut panel: pass the FULL reachable cloud so the cut can be
        # steered live; `cut_point`/`cut_normal` seed the panel sliders. Sphere radius defaults
        # to half the *thinned* voxel edge (touching packing at the surviving `--stride` density,
        # not the raw grid edge -- otherwise a stride>1 view under-fills with gapped spheres);
        # legacy fixed-count payloads (grid_spacing=0) keep the original 0.02.
        spacing = meta["grid_spacing"]
        sphere_r = args.sphere_r if args.sphere_r is not None else (
            0.5 * spacing * args.stride if spacing > 0.0 else 0.02)
        # Precompute everything the panel's mode radio might select, so flipping reachable <-> visible
        # <-> instantaneous at runtime doesn't restart the viewer. Always-on cost is one static BVH
        # raycast (visible_mask) which is cheap; instantaneous fn is just a Python closure.
        available_modes: list[str] = ["reachable"]
        visible_mask = None
        instantaneous_visible_fn = None
        success = agg["success_voxel"]
        if robot_cfg.visibility_kind == "dynamic" and dynamic is None and (
                args.visibility_input is not None or args.view_mode == "visible"):
            raise FileNotFoundError(
                f"dynamic visibility sidecar missing: {dynamic_visibility_path}; "
                "run the robot visibility scorer or select --view-mode reachable"
            )

        # Static visible mask: dispatch by robot's visibility_kind
        if dynamic is not None:
            if args.left_input is not None:
                raise ValueError("dynamic visibility supports R plus --mirror-left only; "
                                 "an independently solved L workspace needs its own dynamic sidecar")
            if robot_cfg.cli_key in ("toddlerbot", "booster_t1"):
                visible_agg = _toddlerbot_visible_field(
                    dynamic, agg_right, meta["n_grid_per_axis"], mirror_left=args.mirror_left,
                )
            else:
                visible_agg = dict(agg_right)
                visible_agg["dexterity"] = dynamic["dexterity"]
                visible_agg["success_voxel"] = dynamic["success_voxel"]
            print(
                f"{robot_cfg.cli_key} dynamic visibility: {dynamic['visible_rows']} visible successful rows; "
                f"{int(visible_agg['success_voxel'].sum())} visible-reachable voxels "
                f"(sidecar {dynamic['sidecar']})"
            )
            # Rebuild bimanual reachable cloud with a visible mask aligned to it (matches
            # `_reach_visible_case` shape so live + static figures stay consistent).
            aggL_dyn = _mirror_agg_l(agg_right, meta["n_grid_per_axis"])
            ids = np.union1d(agg_right["voxel_id"], aggL_dyn["voxel_id"])
            DR = np.full(ids.size, np.nan); DL = np.full(ids.size, np.nan)
            pos_bi = np.full((ids.size, 3), np.nan, dtype=np.float32)
            rR = np.searchsorted(ids, agg_right["voxel_id"])
            rL = np.searchsorted(ids, aggL_dyn["voxel_id"])
            DR[rR] = np.where(agg_right["success_voxel"], agg_right["dexterity"], np.nan)
            DL[rL] = np.where(aggL_dyn["success_voxel"], aggL_dyn["dexterity"], np.nan)
            pos_bi[rR] = agg_right["voxel_pos"]; pos_bi[rL] = aggL_dyn["voxel_pos"]
            D_bi = np.nan_to_num(np.fmax(DR, DL))
            success_bi = ~np.isnan(np.fmax(DR, DL))
            vis_dyn = np.zeros(ids.size, dtype=bool)
            vis_dyn[rR] = visible_agg["success_voxel"]
            if args.mirror_left:
                visL = np.zeros(ids.size, dtype=bool)
                visL[rL] = visible_agg["success_voxel"]
                vis_dyn |= visL
            points, agg["dexterity"], success, visible_mask = pos_bi, D_bi, success_bi, vis_dyn
            available_modes.append("visible")
        elif robot_cfg.visibility_kind in ("steered", "fixed"):
            # Precompute both gimbal clouds up front so the radio can pick either; only the
            # payload's requested `visibility_kind` is exposed (matches what the static figure uses).
            vis_steer, vis_fixed, _inst_steer, _inst_fixed = _camera_visibility(points, robot_cfg.model_key)
            visible_mask = vis_fixed if robot_cfg.visibility_kind == "fixed" else vis_steer
            available_modes.append("visible")

        # Instantaneous closure: FOV-only, no raycast (interactive; per-slider-change BVH setup
        # is too costly). Only these two robots ship actuated cameras.
        if robot_cfg.model_key == "toddlerbot":
            from mj_envs.asset_zoo.reachability_study.toddlerbot_visibility import eyes_in_fov_batch as _eyes_in_fov_batch
            def instantaneous_visible_fn(targets_world, _values, model, data):
                del _values
                return _eyes_in_fov_batch(model, data, np.asarray(targets_world, dtype=np.float64)).any(axis=1)
            available_modes.append("instantaneous")
        elif robot_cfg.model_key == "humanoid_v21":
            def instantaneous_visible_fn(targets_world, _values, model, data):
                del model
                left_id = data.model.site("cam_left_rgb").id
                right_id = data.model.site("cam_right_rgb").id
                lens_poses = [
                    (data.site_xpos[left_id], data.site_xmat[left_id]),
                    (data.site_xpos[right_id], data.site_xmat[right_id]),
                ]
                return _camera_visibility_instantaneous_v2(
                    np.asarray(targets_world, dtype=np.float64), lens_poses,
                )
            available_modes.append("instantaneous")
        elif robot_cfg.model_key == "booster_t1":
            from mj_envs.asset_zoo.reachability_study.t1_visibility import eye_in_fov_batch as _t1_eye_in_fov_batch
            def instantaneous_visible_fn(targets_world, _values, model, data):
                del _values
                return _t1_eye_in_fov_batch(
                    model, data, np.asarray(targets_world, dtype=np.float64),
                ).reshape(-1)
            available_modes.append("instantaneous")
        elif robot_cfg.model_key == "apptronik_apollo":
            from mj_envs.asset_zoo.reachability_study.apptronik_apollo_visibility import (
                eyes_in_fov_batch as _apollo_eyes_in_fov_batch,
            )
            def instantaneous_visible_fn(targets_world, _values, model, data):
                del _values
                return _apollo_eyes_in_fov_batch(
                    model, data, np.asarray(targets_world, dtype=np.float64),
                ).any(axis=1)
            available_modes.append("instantaneous")
        elif robot_cfg.model_key == "fourier_gr3":
            from mj_envs.asset_zoo.reachability_study.fourier_gr3_visibility import (
                eye_in_fov_batch as _gr3_eye_in_fov_batch,
            )
            def instantaneous_visible_fn(targets_world, _values, model, data):
                del _values
                return _gr3_eye_in_fov_batch(model, data, np.asarray(targets_world, dtype=np.float64))
            available_modes.append("instantaneous")
        elif robot_cfg.model_key == "pal_talos":
            from mj_envs.asset_zoo.reachability_study.talos_visibility import (
                eye_in_fov_batch as _talos_eye_in_fov_batch,
            )
            def instantaneous_visible_fn(targets_world, _values, model, data):
                del _values
                return _talos_eye_in_fov_batch(model, data, np.asarray(targets_world, dtype=np.float64))
            available_modes.append("instantaneous")

        initial_mode = args.view_mode if args.view_mode in available_modes else available_modes[0]
        _show_robot_overlay_live(
            points, agg["dexterity"], success,
            cut_point=args.cut_point, cut_normal=args.cut_normal,
            sphere_r=sphere_r, alpha=args.alpha, white_bg=args.white_bg, shadows=args.shadows, flat=args.flat,
            robot=robot_cfg.model_key, stride=args.stride,
            view_mode=initial_mode, visible=visible_mask,
            instantaneous_visible_fn=instantaneous_visible_fn,
            available_modes=tuple(available_modes),
            vis_cmap=args.vis_cmap, blind_cmap=args.blind_cmap,
            custom_viewer=args.custom_viewer, max_geom_cap=args.max_geom,
        )
        return

    if args.view_mode == "visible":
        if robot_cfg.cli_key != "toddlerbot":
            raise ValueError("static --view-mode visible currently supports ToddlerBot only")
        if args.left_input is not None:
            raise ValueError("ToddlerBot dynamic visibility supports R plus --mirror-left only; "
                             "an independently solved L workspace needs its own dynamic sidecar")
        if dynamic is None:
            raise FileNotFoundError(
                f"dynamic visibility sidecar/aggregated cache missing: {dynamic_visibility_path}"
            )
        visible_agg = _toddlerbot_visible_field(
            dynamic, agg_right, meta["n_grid_per_axis"], mirror_left=args.mirror_left,
        )
        _RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path_base = _RESULT_DIR / f"{args.input.stem}_visible_reachable_fov_sections"
        _plot_sections(
            visible_agg["voxel_pos"], visible_agg["dexterity"], visible_agg["success_voxel"],
            args.section, path_base, robot=robot_cfg.model_key, vis_cmap=args.vis_cmap,
        )
        print(
            f"ToddlerBot FOV-visible render: {dynamic['visible_rows']} visible successful R rows; "
            f"{int(visible_agg['success_voxel'].sum())} visible-reachable bimanual voxels"
        )
        return

    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path_base = _RESULT_DIR / f"{args.input.stem}_reachability_index_sections"
    _plot_sections(
        points,
        agg["dexterity"],
        agg["success_voxel"],
        args.section,
        path_base,
        robot=robot_cfg.model_key,
        vis_cmap=args.vis_cmap,
    )
    # Mirror the vector PDF (paper asset) into the manuscript figures dir.
    _mirror_to_paper(path_base)


if __name__ == "__main__":
    main()
