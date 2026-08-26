"""Robot registry + workspace-payload aggregation for the reachability study.

The study's *data* layer, split out of ``plot_workspace_curobo`` (which had grown to hold four
concerns in one file). Everything here maps a ``--robot`` key to its cuRobo IK payload and reduces
that payload to the per-voxel reachability field ``D`` the figures consume; nothing here draws.
Depends on torch/numpy only -- no matplotlib, no MuJoCo -- so ``--cache``/``--cache-all`` and any
downstream consumer can aggregate on a machine with no rendering toolchain.

Two on-disk tiers: ``_CACHE_DIR`` holds the multi-GB raw payloads (gitignored) and
``_AGGREGATED_CACHE_DIR`` the small committed per-voxel reductions. ``_load_workspace_aggregate``
prefers the raw payload and silently falls back to the small cache, so a checkout with neither a
GPU nor the raw files still renders every figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]

_CACHE_DIR = _REPO_ROOT / "mj_envs" / "asset_zoo" / "cache"

# Small, git-committable fallback for _CACHE_DIR's multi-GB raw payloads / visibility sidecars
# (both gitignored via the repo-wide "cache/" rule -- this dir is NOT, since it isn't named
# "cache"). Populated by `--cache[-all]`; consumed by `_load_workspace_aggregate` whenever the raw
# payload is absent, so the plotting/viewer paths run without ever touching the huge originals.
_AGGREGATED_CACHE_DIR = Path(__file__).resolve().parent / "aggregated_cache"

# Single source of truth for `--robot` choices. Each entry is one CLI key + the four pieces of
# config that depend on it. Adding a new robot:
#   1. Append one entry below (cli_key, model_key, payload filename, visibility_kind).
#   2. Add a `case` branch in `_overlay_assets(robot)` so the welded model loads.
#   3. Add a branch in `_camera_visibility(points, robot)` only after that robot has a complete
#      visibility sidecar. Reachability-only view/figure needs no camera implementation.
#   4. Add an `_OVERLAY_POSE_BY_ROBOT` / `_ARMS_DOWN_POSE` entry if the arm-clears-FOV pose differs.
# After that, `--help` lists it, `--robot <key>` resolves it, and the rest of the code paths adapt.
@dataclass(frozen=True)
class RobotConfig:
    cli_key: str            # what the user writes after --robot (e.g. "v2_fixed")
    model_key: str          # _overlay_assets / _camera_visibility dispatch key ("humanoid_v21"/"unitree_g1")
    payload: str            # filename inside _CACHE_DIR
    visibility_kind: str    # "steered" | "fixed" | "dynamic"


_ROBOTS: dict[str, RobotConfig] = {
    # The four V2 rigs (K = 1/2/3 cameras) each carry their OWN reachability payload. Camera hardware
    # is 26 of the dual rig's 104 cuRobo collision spheres and is active in self-collision, so reach
    # genuinely depends on K -- the older comment here ("camera is IK-independent") was wrong, and
    # v2_single was scored against the DUAL rig's obstruction. All four are solved on one shared grid
    # (pad 0.10, 73x91x73), anchored to integer multiples of the spacing on every axis, so a voxel
    # index means the same target in every payload. The shipped pad-0.05 payload cannot be mixed in
    # (smaller grid), nor can any payload predating the Z-axis snap -- its Z origin is -0.3662, i.e.
    # -18.31 cells, so its voxels sit off-lattice relative to these.
    "v2":       RobotConfig("v2",       "humanoid_v21", "workspace_curobo_ik_humanoid_v21_dual_R_so3_dex_0p02.pt", "steered"),
    "v2_fixed": RobotConfig("v2_fixed", "humanoid_v21", "workspace_curobo_ik_humanoid_v21_dual_R_so3_dex_0p02.pt", "fixed"),
    "v2_single":       RobotConfig("v2_single",       "humanoid_v21_single", "workspace_curobo_ik_humanoid_v21_single_R_so3_dex_0p02.pt", "steered"),
    "v2_single_fixed": RobotConfig("v2_single_fixed", "humanoid_v21_single", "workspace_curobo_ik_humanoid_v21_single_R_so3_dex_0p02.pt", "fixed"),
    "v2_triple":       RobotConfig("v2_triple",       "humanoid_v21_triple", "workspace_curobo_ik_humanoid_v21_triple_R_so3_dex_0p02.pt", "steered"),
    "v2_triple_fixed": RobotConfig("v2_triple_fixed", "humanoid_v21_triple", "workspace_curobo_ik_humanoid_v21_triple_R_so3_dex_0p02.pt", "fixed"),
    "g1":       RobotConfig("g1",       "unitree_g1",   "workspace_curobo_ik_unitree_g1_R_so3_dex_0p02.pt",  "fixed"),
    "toddlerbot": RobotConfig("toddlerbot", "toddlerbot", "workspace_curobo_ik_toddlerbot_R_so3_dex_0p02.pt", "dynamic"),
    "booster_t1": RobotConfig("booster_t1", "booster_t1", "workspace_curobo_ik_booster_t1_R_so3_dex_0p02.pt", "dynamic"),
    # Apollo exposes an occlusion-aware dynamic sidecar for its approved provisional FOV envelope.
    # It remains diagnostic, never calibrated sensor performance.
    # GR-3 ships a real 2-DOF head, so its visibility is dynamic. Its two declared cameras
    # (15 deg URDF mount vs 40 deg official-FOV-figure mount) are scored as a sensitivity pair.
    "fourier_gr3": RobotConfig(
        "fourier_gr3", "fourier_gr3",
        "workspace_curobo_ik_fourier_gr3_R_so3_dex_0p02.pt", "dynamic",
    ),
    # TALOS ships a real 2-DOF head carrying an Orbbec Astra Pro. Its two co-located streams
    # (RGB 63.1x49.4 deg, depth 58.4x45.5 deg) are scored as a primary plus sensitivity twin.
    "pal_talos": RobotConfig(
        "pal_talos", "pal_talos",
        "workspace_curobo_ik_pal_talos_R_so3_dex_0p02.pt", "dynamic",
    ),
    "apptronik_apollo": RobotConfig(
        "apptronik_apollo", "apptronik_apollo",
        "workspace_curobo_ik_apptronik_apollo_R_so3_dex_0p02_symrest.pt", "dynamic",
    ),
}

_TODDLERBOT_DYNAMIC_VISIBILITY = _CACHE_DIR / "workspace_curobo_ik_toddlerbot_R_so3_dex_0p02_visibility_gpu.pt"
_BOOSTER_T1_DYNAMIC_VISIBILITY = _CACHE_DIR / "t1_visibility_R_dex_0p02_gpu.pt"
_FOURIER_GR3_DYNAMIC_VISIBILITY = _CACHE_DIR / "fourier_gr3_visibility_R_dex_0p02_gpu.pt"
_PAL_TALOS_DYNAMIC_VISIBILITY = _CACHE_DIR / "pal_talos_visibility_R_dex_0p02_gpu.pt"
_APOLLO_DYNAMIC_VISIBILITY = (
    _CACHE_DIR / "workspace_curobo_ik_apptronik_apollo_R_so3_dex_0p02_symrest_visibility_gpu.pt"
)

# Canonical dynamic-visibility sidecar per dynamic-visibility --robot cli_key. Single source of
# truth shared by `_reach_visible_case` and `main()`'s --view / --view-mode visible branches,
# which previously duplicated this same if/elif dispatch.
_DYNAMIC_VISIBILITY_SIDECAR: dict[str, Path] = {
    "toddlerbot": _TODDLERBOT_DYNAMIC_VISIBILITY,
    "booster_t1": _BOOSTER_T1_DYNAMIC_VISIBILITY,
    "fourier_gr3": _FOURIER_GR3_DYNAMIC_VISIBILITY,
    "pal_talos": _PAL_TALOS_DYNAMIC_VISIBILITY,
    "apptronik_apollo": _APOLLO_DYNAMIC_VISIBILITY,
}

_DEFAULT_ROBOT_KEY = "v2"  # bare-run (no --robot) default

_DEFAULT_INPUT = _CACHE_DIR / _ROBOTS[_DEFAULT_ROBOT_KEY].payload


def _resolve_robot(cli_key: str | None) -> RobotConfig:
    """Validate the user's --robot choice and return its config. Single source of validation."""
    cfg = _ROBOTS.get(cli_key if cli_key is not None else _DEFAULT_ROBOT_KEY)
    if cfg is None:
        raise ValueError(f"--robot {cli_key!r} unknown; choose {sorted(_ROBOTS)}")
    return cfg



def _aggregate_by_voxel(data: dict) -> dict:
    # Scatter-aggregate on GPU when available (~2M rows): CUDA scatter is ~10x the CPU path.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_pos = data["target_pos"].float().to(dev)
    success = data["success"].bool().to(dev)
    voxel_index = data.get("voxel_index")
    orient_index = data.get("orient_index")
    if voxel_index is None:
        voxel_index = torch.arange(target_pos.shape[0], dtype=torch.long, device=dev)
    else:
        voxel_index = voxel_index.long().to(dev)
    if orient_index is None:
        orient_index = torch.zeros(target_pos.shape[0], dtype=torch.long, device=dev)
    else:
        orient_index = orient_index.long().to(dev)

    unique_voxel, compact_index = torch.unique(voxel_index, sorted=True, return_inverse=True)
    n_vox = int(unique_voxel.numel())
    n_ori = int(data.get("n_orientations", int(orient_index.max().item()) + 1 if orient_index.numel() else 1))

    # Vectorized voxel aggregation (was two Python loops over ~2M rows -> minutes; now
    # torch scatter over the whole tensor). compact_index maps each row to its voxel.
    pos_count = torch.bincount(compact_index, minlength=n_vox).clamp_min(1).float()
    pos_sum = torch.zeros((n_vox, 3), dtype=torch.float32, device=dev)
    pos_sum.index_add_(0, compact_index, target_pos)
    voxel_pos = pos_sum / pos_count.unsqueeze(1)

    succ = success.bool()
    vv = compact_index[succ]
    oo = orient_index[succ].clamp(max=n_ori - 1)
    hit = torch.zeros((n_vox, n_ori), dtype=torch.bool, device=dev)
    hit[vv, oo] = True  # per-voxel per-orientation feasibility

    # Per-voxel max of the manipulability terms over successful rows (amax scatter, then
    # voxels with no success -> NaN, matching the prior init-NaN semantics).
    trans = torch.full((n_vox,), float("-inf"), dtype=torch.float32, device=dev)
    rot = torch.full((n_vox,), float("-inf"), dtype=torch.float32, device=dev)
    trans.scatter_reduce_(0, vv, data["manip_trans"][succ.cpu()].float().to(dev), reduce="amax", include_self=True)
    rot.scatter_reduce_(0, vv, data["manip_rot"][succ.cpu()].float().to(dev), reduce="amax", include_self=True)
    reached = torch.zeros(n_vox, dtype=torch.bool, device=dev)
    reached[vv] = True
    trans[~reached] = float("nan")
    rot[~reached] = float("nan")

    dexterity = hit.sum(dim=1).float() / max(n_ori, 1)
    return {
        "voxel_id": unique_voxel.cpu().numpy(),
        "voxel_pos": voxel_pos.cpu().numpy(),
        "dexterity": dexterity.cpu().numpy(),
        "trans": trans.cpu().numpy(),
        "rot": rot.cpu().numpy(),
        "success_voxel": hit.any(dim=1).cpu().numpy(),
        "n_orientations": n_ori,
    }


def _aggregate_toddlerbot_visible(data: dict, agg: dict, visibility_path: Path) -> dict:
    """Aggregate dynamic per-IK-row stereo visibility into exact visible-reachable D.

    A row contributes only when its arm IK succeeded *and* its source-MuJoCo neck-aimed head state
    sees the target through either physical eye. This preserves orientation multiplicity: a voxel's
    visible index is visible successful orientations divided by the workspace SO(3) count, not a
    binary static-camera label.
    """
    vis = torch.load(str(visibility_path), map_location="cpu", weights_only=False)
    n_rows = int(data["success"].numel())
    # Whitelist the robots whose sidecars this aggregator understands. Keep the two failure
    # modes separate: an unlisted robot and a row-count mismatch used to raise the same
    # "does not match workspace rows" message, which sent a row-count hunt after what was
    # really a missing registration.
    if vis.get("robot") not in ("toddlerbot", "booster_t1", "apptronik_apollo", "fourier_gr3", "pal_talos"):
        raise ValueError(f"visibility sidecar {visibility_path} is for unsupported robot {vis.get('robot')!r}")
    if int(vis.get("workspace_rows", -1)) != n_rows:
        raise ValueError(
            f"visibility sidecar {visibility_path} has workspace_rows={vis.get('workspace_rows')}, "
            f"expected {n_rows}"
        )
    visible_rows = vis["visible_left"].bool() | vis["visible_right"].bool()
    if visible_rows.numel() != n_rows:
        raise ValueError(f"visibility sidecar {visibility_path} has {visible_rows.numel()} rows, expected {n_rows}")

    voxel_index = data["voxel_index"].long().numpy()
    compact = np.searchsorted(agg["voxel_id"], voxel_index)
    if not np.array_equal(agg["voxel_id"][compact], voxel_index):
        raise ValueError("workspace voxel IDs disagree with aggregate")
    contributing = (data["success"].bool() & visible_rows).numpy()
    visible_count = np.bincount(compact[contributing], minlength=agg["voxel_id"].size)
    visible_dexterity = visible_count.astype(np.float32) / max(int(agg["n_orientations"]), 1)
    return {
        "dexterity": visible_dexterity,
        "success_voxel": visible_count > 0,
        "visible_rows": int(contributing.sum()),
        "sidecar": visibility_path,
    }


def _toddlerbot_visible_field(
    dynamic: dict,
    agg_right: dict,
    n_grid_per_axis,
    *,
    mirror_left: bool,
) -> dict:
    """Bimanual-mirror an unmirrored per-R-voxel visible-dexterity dict into `agg_right`'s shape.

    `dynamic` is `_aggregate_toddlerbot_visible`'s return value (or an aggregated-cache copy of
    it) -- decoupled from the raw payload so this mirroring step works identically whether
    `dynamic` was just computed from a raw visibility sidecar or loaded from the small cache.
    """
    visible_agg = dict(agg_right)
    visible_agg["dexterity"] = dynamic["dexterity"]
    visible_agg["success_voxel"] = dynamic["success_voxel"]
    if mirror_left:
        # User-established L/R symmetry: reflect targets/visibility aggregate, never head commands.
        visible_left = _mirror_agg_l(visible_agg, n_grid_per_axis)
        left_order = np.argsort(visible_left["voxel_id"])
        left_ids = visible_left["voxel_id"][left_order]
        left_at_right = left_order[np.searchsorted(left_ids, visible_agg["voxel_id"])]
        visible_agg["dexterity"] = np.fmax(
            visible_agg["dexterity"], visible_left["dexterity"][left_at_right]
        )
        visible_agg["success_voxel"] |= visible_left["success_voxel"][left_at_right]
    return visible_agg


def _mirror_agg_l(agg: dict, n_grid_per_axis) -> dict:
    """Y-mirror an R per-voxel aggregate into an L aggregate (sagittal reflection).

    Operates on the ~voxel-count aggregate (not the 18.6 M-row payload) so it is instant and needs
    no second load. The legacy-symmetric grid is symmetric about Y=0, so the mirror is an exact Y
    index reversal `iy' = (ny-1)-iy`; voxel_pos Y is negated and voxel_id recomputed on the same
    row-major lattice `ix*(ny*nz)+iy*nz+iz`. D / reachability / manipulability copy unchanged.
    Valid for plotting only (makes L/R symmetry true by construction).
    """
    ny = int(n_grid_per_axis[1]); nz = int(n_grid_per_axis[2])
    vid = agg["voxel_id"].astype(np.int64)
    ix = vid // (ny * nz); iy = (vid % (ny * nz)) // nz; iz = vid % nz
    pos = agg["voxel_pos"].copy(); pos[:, 1] = -pos[:, 1]
    out = dict(agg)
    out["voxel_id"] = ix * (ny * nz) + (ny - 1 - iy) * nz + iz
    out["voxel_pos"] = pos
    return out


def _aggregated_cache_path(payload_path: Path) -> Path:
    """Small-cache counterpart of a raw `cache/<payload>.pt` path: `aggregated_cache/<stem>_aggregated.pt`."""
    return _AGGREGATED_CACHE_DIR / f"{payload_path.stem}_aggregated.pt"


def _workspace_meta(data: dict, cfg: RobotConfig) -> dict:
    """The three raw-payload fields anything downstream of aggregation still needs."""
    return {
        "n_grid_per_axis": data["n_grid_per_axis"],
        "grid_spacing": float(data.get("grid_spacing", 0.0) or 0.0),
        "robot": str(data.get("robot", cfg.model_key)),
    }


def _load_workspace_aggregate(
    payload_path: Path,
    cfg: RobotConfig,
    *,
    need_visible: bool,
    dynamic_visibility_path: Path | None = None,
) -> tuple[dict, dict | None, dict]:
    """Load a robot's per-voxel reachability (+ optional visibility) field, raw payload or fallback.

    Returns `(agg, dynamic, meta)`: `agg` is `_aggregate_by_voxel`'s per-voxel dict; `meta` is
    `{n_grid_per_axis, grid_spacing, robot}` (the only raw-payload fields anything downstream of
    aggregation still needs); `dynamic` is `_aggregate_toddlerbot_visible`'s UNMIRRORED per-R-voxel
    visible-dexterity dict for `cfg.visibility_kind == "dynamic"` robots, or `None` if visibility
    wasn't requested or wasn't available. Mirroring is deliberately left to the caller (via
    `_toddlerbot_visible_field`, or by combining `dynamic` with `agg` directly) rather than done
    here: the three call sites already apply different mirror policies (`_reach_visible_case`
    always mirrors; `main()`'s `--view` branch only mirrors toddlerbot/booster_t1, not
    apollo/gr3/talos) -- baking one policy into the loader would silently change that.

    Tries, in order: (1) the raw payload at `payload_path` if present -- computes everything fresh,
    exactly as before this cache existed; (2) `_aggregated_cache_path(payload_path)` if present --
    a small pre-aggregated file, no raw payload or raw visibility sidecar touched; (3) raises,
    naming both paths tried. `need_visible=False` skips even attempting visibility (no sidecar
    load, no extra aggregation pass) -- callers that don't need it (e.g. a plain reachability
    figure) must not pay for it. A missing/absent dynamic-visibility source (sidecar or cached
    field) is never fatal here -- `dynamic` is left `None` and the decision to treat that as an
    error belongs to the caller, exactly as it does today (see `main()`'s `--view-mode visible`
    check).
    """
    if payload_path.is_file():
        data = torch.load(str(payload_path), map_location="cpu", weights_only=False)
        agg = _aggregate_by_voxel(data)
        meta = _workspace_meta(data, cfg)
        dynamic = None
        if need_visible and cfg.visibility_kind == "dynamic" and dynamic_visibility_path is not None \
                and dynamic_visibility_path.is_file():
            dynamic = _aggregate_toddlerbot_visible(data, agg, dynamic_visibility_path)
        return agg, dynamic, meta

    cache_path = _aggregated_cache_path(payload_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"neither raw payload {payload_path} nor aggregated cache {cache_path} exists; "
            "run the cuRobo IK generator or write the aggregated cache with --cache"
        )
    cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    dynamic = cached["visible_agg"] if need_visible else None
    return cached["agg"], dynamic, cached["meta"]


def _write_aggregated_cache(cfg: RobotConfig) -> Path:
    """Write `aggregated_cache/<payload_stem>_aggregated.pt` for `cfg` from its raw checkpoint.

    Requires the raw payload (and, for dynamic-visibility robots, its visibility sidecar) to be
    present locally -- this is the one-time regeneration step; `_load_workspace_aggregate` is what
    later reads the result back without either raw file.
    """
    payload_path = _CACHE_DIR / cfg.payload
    data = torch.load(str(payload_path), map_location="cpu", weights_only=False)
    agg = _aggregate_by_voxel(data)
    meta = _workspace_meta(data, cfg)
    visible = None
    if cfg.visibility_kind == "dynamic":
        sidecar = _DYNAMIC_VISIBILITY_SIDECAR[cfg.cli_key]
        visible = _aggregate_toddlerbot_visible(data, agg, sidecar)
    out_path = _aggregated_cache_path(payload_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agg": agg, "visible_agg": visible, "meta": meta}, str(out_path))
    return out_path
