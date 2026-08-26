"""Seeded cuRobo IK workspace generator smoke path.

This is the target-conditioned upgrade path for dexterous-space maps:
sample an EE target pose, seed cuRobo IK from a nearby sampled joint pose, solve
IK, then compute MuJoCo Jacobian manipulability at the returned configuration.

Initial scope is humanoid_v21 only. It uses full deployed geometry with welded
head camera and welded parallel gripper, so camera/gripper collision geometry is
present without extra non-arm DOFs. cuRobo's current robot cfg still locks all
non-arm joints internally; this script uses direct IK (`ik_solver.solve_pose`),
not full `plan_pose` trajectory optimization.

Throughput (`--solver batched`, the default)
--------------------------------------------
`--solver sequential` is the faithful ground truth: one target per `solve_pose`
call, per-target nearest safe-arm seed, 512 internal seeds. It is ~70-80 solves/s
because the GPU sits ~88% idle (kernel-launch / Python-overhead bound), NOT compute
bound. `--solver batched` (default) solves B targets per `solve_pose` call with one
constant home idle-pin + optimizer seed and fewer seeds; measured 99.2% per-voxel D
agreement vs sequential. On one free 24GB 4090: ~2258 solves/s (30x). Stable ceiling
there is `--batch-size 512 --num-seeds 32` (batch 1024 = cudaErrorIllegalAddress,
2048 = OOM); memory grows with batch x num_seeds x collision-spheres. On 46GB L40S
there is headroom to push batch higher.

Multi-GPU data-parallel (used for the paper's 40^3 x 32 = 2.048M-solve grid)
---------------------------------------------------------------------------
Split the target rows across GPUs with native `--target-start` / `--limit`, one
process per GPU pinned by `CUDA_VISIBLE_DEVICES`, then `merge_workspace_curobo.py`.
ser16 recipe (8x L40S 46GB, user bx35, rsync-only, no sshfs):
  1. Provision cuRobo: it is pure-Python + warp JIT (no compiled .so), so `rsync` the
     curobo tree over; then `pip install qpsolvers cuda-core cuda-bindings
     cuda-pathfinder` (the only non-rsyncable deps). warp JIT-compiles kernels on
     first run (~90s).
  2. Ensure the target GPUs are free.
  3. Launch 8 chunks: TOTAL/8 rows each, per-GPU `CUDA_VISIBLE_DEVICES=$g` +
     `--target-start $((g*CHUNK)) --limit CHUNK`, `--batch-size 512 --num-seeds 32`.
     Do NOT use `set -e`: the post-save teardown segfault (exit 139) is a benign
     cuRobo/torch exit crash AFTER the file is written; gate success on file
     existence, not exit code. 2.048M solves finished in ~4 min (14.6k solves/s
     aggregate) vs ~8 h single-GPU sequential.
  4. `merge_workspace_curobo.py` the 8 sidecars into one payload, pull to grl1 cache,
     clean remote scratch.

Two-host 16-GPU extension (ser10 + ser16, G1 fine 0.02 production, 2026-07-11):
  Same as above but 16 chunks over both hosts. G1 `--pad 0.15` -> 63x75x62 = 292950 voxels;
  `CHUNK = ceil(292950/16)=18310 voxels x64 = 1171840 rows`; global shard `G=0..15` (ser10
  G=0..7, ser16 G=8..15), `--target-start G*CHUNK`. 18.75M solves in ~10 min wall. Gotchas:
    - GRID MUST ENCLOSE the full reachable set. legacy-symmetric bounds = seed-cache EE cloud
      + `--pad`; too-tight pad caps the low-D fringe flat (G1 needed 0.15, not 0.05). After
      the solve, bin `success` into `n_grid_per_axis` and confirm every one of the 6 faces has
      0 reachable voxels; if not, raise `--pad` and re-solve.
    - cache/ is EXCLUDED from the rsync push -> push `safe_arm_poses_unitree_g1.pt` (the
      seed cache) EXPLICITLY, else the solve has no seed pool.
    - ser10 launch MUST export `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` (CXXABI_1.3.15).
    - Pull with the remote glob QUOTED (`"$H:.../shard_${H}_*.pt"`) or zsh expands it
      locally and skips the rsync.
  Full copy-paste recipe: reachability_study/readme_reachability.md ("Reproduce the fine 0.02 figure").
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import (  # noqa: E402
    HOME_KEYFRAME,
    get_humanoid_v21_robot_cfg,
)
from mjlab.entity import Entity  # noqa: E402


def _require_curobo() -> None:
    """Import curobo + IK-config names into module globals on first solver use.

    Deferred so importing this module only for _load_welded_model (the plotting path)
    does not pay curobo's ~1s import cost. `from __future__ import annotations` keeps the
    curobo type hints as strings, so deferring these does not break function signatures.
    Idempotent; solver entry points call it before touching curobo objects.
    """
    global IKSolver, IKSolverCfg, JointState, GoalToolPose, ToolPoseCriteria, CuroboArmPlanner
    global _ORIENTATION_TOLERANCE, _POSITION_TOLERANCE, build_robot_cfg_dict_from_urdf
    if "IKSolver" in globals():
        return
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    from curobo._src.state.state_joint import JointState
    from curobo._src.types.tool_pose import GoalToolPose
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
    from tasks.visual_manipulation.curobo.ik_curobo import CuroboArmPlanner
    from tasks.visual_manipulation.curobo.ik_curobo_robot_cfg import (
        _ORIENTATION_TOLERANCE,
        _POSITION_TOLERANCE,
        build_robot_cfg_dict_from_urdf,
    )


def _load_welded_model(head_camera: str = "welded") -> mujoco.MjModel:
    """Compile full deployed humanoid_v21 geometry with welded camera/gripper DOFs.

    ``head_camera`` picks the camera rig ("none" for the camera-free upper bound, "welded" for the
    shipped dual, "actuated_single"/"actuated_triple" for the K=1/K=3 ablation arms). The rig's
    geometry obstructs the arm, so it changes FK/collision and must be reflected in the payload
    name; default preserves the shipped dual."""
    cfg = get_humanoid_v21_robot_cfg(
        head_camera=head_camera,
        end_effector="welded",
        hand="parallel_gripper",
    )
    return Entity(cfg).compile()


def _load_welded_model_g1() -> mujoco.MjModel:
    """Compile full deployed G1 geometry with welded parallel gripper (builtin head).

    Matches the kinematics of the G1 cuRobo cfg (``g1_curobo_robot_cfg._build_mj_model_resolved``,
    builtin head): this model is used only for FK site poses + Jacobian metrics + qpos mapping, not
    for cuRobo collision (cuRobo carries its own spheres), so ``welded`` vs the cfg's ``actuated``
    (locked gripper DOFs either way) is immaterial while the grasp sites + arm chain match."""
    from mj_envs.asset_zoo.g1.g1_constants import get_g1_robot_cfg
    cfg = get_g1_robot_cfg(head_camera="builtin", end_effector="welded", hand="parallel_gripper")
    return Entity(cfg).compile()


def _load_berkeley_humanoid_lite_model() -> mujoco.MjModel:
    """Compile Berkeley Lite source MJCF with its paired bare-hand endpoint sites."""
    path = _REPO_ROOT / "asset" / "berkeley_humanoid_lite" / "berkeley_humanoid_lite.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def _load_toddlerbot_model() -> mujoco.MjModel:
    """Compile ToddlerBot source MJCF with stereo/tool sites and gearbox couplings."""
    path = _REPO_ROOT / "asset" / "toddlerbot_2xm_gripper" / "toddlerbot_2xm_gripper_pos.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def _humanoid_robot_cfg_dict(head_camera: str = "actuated") -> dict:
    _require_curobo()
    return build_robot_cfg_dict_from_urdf(head_camera=head_camera)


def _g1_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.g1_curobo_robot_cfg import build_robot_cfg_dict
    return build_robot_cfg_dict()


def _berkeley_humanoid_lite_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.berkeley_humanoid_lite_curobo_robot_cfg import (
        build_robot_cfg_dict,
    )
    return build_robot_cfg_dict()


def _toddlerbot_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.toddlerbot_curobo_robot_cfg import build_robot_cfg_dict
    return build_robot_cfg_dict()


def _booster_t1_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.booster_t1_curobo_robot_cfg import build_robot_cfg_dict
    return build_robot_cfg_dict()


def _fourier_gr3_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.fourier_gr3_curobo_robot_cfg import build_robot_cfg_dict
    return build_robot_cfg_dict()


def _pal_talos_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.talos_curobo_robot_cfg import build_robot_cfg_dict
    return build_robot_cfg_dict()


def _apptronik_apollo_robot_cfg_dict() -> dict:
    from tasks.visual_manipulation.curobo.apptronik_apollo_curobo_robot_cfg import build_robot_cfg_dict
    return build_robot_cfg_dict()


def _load_booster_t1_model() -> mujoco.MjModel:
    """Compile T1 source MJCF with its paired bare-hand endpoint sites and head_cam."""
    path = _REPO_ROOT / "asset" / "booster_t1" / "t1.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def _load_fourier_gr3_model() -> mujoco.MjModel:
    """Compile the GR-3 source MJCF with its tool sites, head cameras and refitted collision."""
    path = _REPO_ROOT / "asset" / "fourier_gr3" / "gr3.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def _load_pal_talos_model() -> mujoco.MjModel:
    """Compile the TALOS study MJCF with its tool sites, head cameras and refitted collision."""
    path = _REPO_ROOT / "asset" / "pal_talos" / "talos_study.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def _load_apptronik_apollo_model() -> mujoco.MjModel:
    """Compile Apollo source MJCF with approved palm-center endpoint sites."""
    path = _REPO_ROOT / "asset" / "apptronik_apollo" / "apptronik_apollo.xml"
    return mujoco.MjModel.from_xml_path(str(path))


def _no_source_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """No-op for robots whose planner coordinates directly fill source qpos."""


def _apply_toddlerbot_source_coupling(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    from tasks.visual_manipulation.curobo.toddlerbot_curobo_robot_cfg import apply_source_drive_coupling
    apply_source_drive_coupling(model, data)


_CACHE_DIR = _REPO_ROOT / "mj_envs" / "asset_zoo" / "cache"


@dataclass(frozen=True)
class RobotSpec:
    """Per-robot bindings so one solve path serves multiple arms.

    ``home_root_pos`` is the root translation both the FK read (``_site_pose``) and the seed
    write (``_write_seed_qpos``) apply; with identity root orientation it cancels out of
    ``target_pos``, so only self-consistency matters. ``site_name`` / ``tool_frame`` map a side ('L'/'R') to the MuJoCo
    grasp-site name and the cuRobo tool-frame name (equal for both current robots, named apart for
    clarity). ``robot_cfg_dict_fn`` is a deferred import so the humanoid / plot paths never pull the
    other robot's assets."""

    name: str
    model_tag: str
    cache_path: pathlib.Path
    home_root_pos: tuple
    base_link: str
    model_loader: Callable[[], mujoco.MjModel]
    robot_cfg_dict_fn: Callable[[], dict]
    site_name: Callable[[str], str]
    tool_frame: Callable[[str], str]
    apply_source_coupling: Callable[[mujoco.MjModel, mujoco.MjData], None] = _no_source_coupling
    idle_from_source_qpos0: bool = False


# head_camera rig -> (FK/model-loader mode, cuRobo-collision mode) for humanoid_v21. The shipped
# dual splits these ("welded" geometry for FK, "actuated" for the solver's spheres); the K=1/K=3
# ablation rigs have no welded variant, so both sides use the actuated rig. Keyed by the CLI name so
# a payload can never be built with one rig's FK and another's collision model.
_HUMANOID_RIG_MODES = {
    "dual": ("welded", "actuated"),
    "none": ("none", "none"),
    "single": ("actuated_single", "actuated_single"),
    "triple": ("actuated_triple", "actuated_triple"),
}


def get_robot_spec(robot: str, rig: str = "dual") -> RobotSpec:
    """Resolve a RobotSpec by name; G1 imports are deferred inside so the humanoid path (and the
    curobo-free plot path) never pull G1 assets.

    ``rig`` selects the humanoid's head-camera layout (see ``_HUMANOID_RIG_MODES``). It changes the
    collision model and therefore the reachable set, so it is stamped into ``model_tag``.

    The arm-pose cache is deliberately SHARED across rigs. It is not a result -- it fixes the grid
    bounds (``_grid_bounds_from_cache``), the target ordering, the SO(3) reference set, and the IK
    seed pool. Per-rig caches would move the grid, so row ``i`` of two payloads would no longer be
    the same target, which breaks the witness/contested classification the ablation is built on. The
    cache's poses were sampled self-collision-free against the dual rig, so as SEEDS they are mildly
    conservative for the camera-free rig -- but identically so for every rig, so the bias is common
    mode and cancels in the K-to-K comparison. Seeds are start points only; cuRobo validates the
    solution against the rig's own collision model."""
    if robot == "humanoid_v21":
        loader_mode, cfg_mode = _HUMANOID_RIG_MODES[rig]
        return RobotSpec(
            name="humanoid_v21",
            model_tag=f"head_camera={loader_mode},end_effector=welded,hand=parallel_gripper",
            cache_path=_CACHE_DIR / "safe_arm_poses_humanoid_v21.pt",
            home_root_pos=tuple(HOME_KEYFRAME.pos),
            base_link="base_link",
            model_loader=lambda: _load_welded_model(loader_mode),
            robot_cfg_dict_fn=lambda: _humanoid_robot_cfg_dict(cfg_mode),
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
        )
    if robot in ("g1", "unitree_g1"):
        from mj_envs.asset_zoo.g1.g1_constants import HOME_KEYFRAME as G1_HOME
        _g1_grasp = {"L": "left_hand_grasp", "R": "right_hand_grasp"}
        return RobotSpec(
            name="unitree_g1",
            model_tag="head_camera=builtin,end_effector=welded,hand=parallel_gripper",
            cache_path=_CACHE_DIR / "safe_arm_poses_unitree_g1.pt",
            home_root_pos=tuple(G1_HOME.pos),
            base_link="pelvis",
            model_loader=_load_welded_model_g1,
            robot_cfg_dict_fn=_g1_robot_cfg_dict,
            site_name=lambda s: _g1_grasp[s.upper()],
            tool_frame=lambda s: _g1_grasp[s.upper()],
        )
    if robot in ("berkeley_humanoid_lite", "berkeley_lite"):
        return RobotSpec(
            name="berkeley_humanoid_lite",
            model_tag="source_mjcf,bare_hand_endpoint=-0.10m_local_z",
            cache_path=_CACHE_DIR / "safe_arm_poses_berkeley_humanoid_lite.pt",
            home_root_pos=(0.0, 0.0, 0.68),
            base_link="base_link",
            model_loader=_load_berkeley_humanoid_lite_model,
            robot_cfg_dict_fn=_berkeley_humanoid_lite_robot_cfg_dict,
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
        )
    if robot in ("toddlerbot", "toddlerbot_2xm_gripper"):
        return RobotSpec(
            name="toddlerbot",
            model_tag="source_mjcf,stereo_head,physical_driven_arm_outputs",
            cache_path=_CACHE_DIR / "safe_arm_poses_toddlerbot.pt",
            home_root_pos=(0.0, 0.0, 0.315053),
            base_link="base_link",
            model_loader=_load_toddlerbot_model,
            robot_cfg_dict_fn=_toddlerbot_robot_cfg_dict,
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
            apply_source_coupling=_apply_toddlerbot_source_coupling,
        )
    if robot in ("booster_t1", "t1"):
        from mj_envs.tasks.visual_manipulation.curobo.booster_t1_curobo_robot_cfg import (
            HOME_ROOT_POS,
        )
        return RobotSpec(
            name="booster_t1",
            model_tag="source_mjcf,base_link=base_link,D455_head_cam,no_gripper",
            cache_path=_CACHE_DIR / "safe_arm_poses_booster_t1.pt",
            home_root_pos=HOME_ROOT_POS,
            base_link="base_link",
            model_loader=_load_booster_t1_model,
            robot_cfg_dict_fn=_booster_t1_robot_cfg_dict,
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
        )
    if robot in ("fourier_gr3", "gr3"):
        from mj_envs.tasks.visual_manipulation.curobo.fourier_gr3_curobo_robot_cfg import (
            HOME_ROOT_POS,
        )
        return RobotSpec(
            name="fourier_gr3",
            model_tag="source_mjcf,base_link=base_link,OAK_head_cam,dummy_hand_tools,refit_g3",
            cache_path=_CACHE_DIR / "safe_arm_poses_fourier_gr3.pt",
            home_root_pos=HOME_ROOT_POS,
            base_link="base_link",
            model_loader=_load_fourier_gr3_model,
            robot_cfg_dict_fn=_fourier_gr3_robot_cfg_dict,
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
        )
    if robot in ("pal_talos", "talos"):
        from mj_envs.tasks.visual_manipulation.curobo.talos_curobo_robot_cfg import (
            HOME_ROOT_POS,
        )
        return RobotSpec(
            name="pal_talos",
            model_tag="source_mjcf,base_link=base_link,orbbec_astra_pro_head_cam,"
                      "wrist_grasp_tools,refit_g3_pca",
            cache_path=_CACHE_DIR / "safe_arm_poses_pal_talos.pt",
            home_root_pos=HOME_ROOT_POS,
            base_link="base_link",
            model_loader=_load_pal_talos_model,
            robot_cfg_dict_fn=_pal_talos_robot_cfg_dict,
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
        )
    if robot in ("apptronik_apollo", "apollo"):
        from mj_envs.tasks.visual_manipulation.curobo.apptronik_apollo_curobo_robot_cfg import (
            HOME_ROOT_POS,
        )
        return RobotSpec(
            name="apptronik_apollo",
            model_tag="source_mjcf,palm_center_tools,approved_group3_spheres",
            cache_path=_CACHE_DIR / "safe_arm_poses_apptronik_apollo.pt",
            home_root_pos=HOME_ROOT_POS,
            base_link="base_link",
            model_loader=_load_apptronik_apollo_model,
            robot_cfg_dict_fn=_apptronik_apollo_robot_cfg_dict,
            site_name=lambda s: f"end_effector_{s.upper()}_site",
            tool_frame=lambda s: f"end_effector_{s.upper()}_site",
            idle_from_source_qpos0=True,
        )
    raise ValueError(
        f"unknown robot: {robot!r} (expected humanoid_v21, g1, berkeley_humanoid_lite, "
        f"toddlerbot, booster_t1, fourier_gr3, pal_talos, or apptronik_apollo)"
    )


def robot_cfg_dict_for_side(spec: RobotSpec, side: str) -> dict:
    """Return one-arm Apollo config with inactive arm hard-locked at source rest.

    Other robots retain their existing dual-arm configuration. This must be used by
    generation and validation together because Apollo workspace payloads then contain
    seven target-arm coordinates, rather than an unconstrained fourteen-arm solution.
    """
    if spec.name != "apptronik_apollo":
        return spec.robot_cfg_dict_fn()
    from mj_envs.tasks.visual_manipulation.curobo.apptronik_apollo_curobo_robot_cfg import (
        build_robot_cfg_dict,
    )

    return build_robot_cfg_dict(active_side=side)


def _load_cache(cache_path: pathlib.Path, limit: int, stride: int) -> dict:
    if not cache_path.exists():
        raise FileNotFoundError(f"missing pose cache: {cache_path}")
    cache = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    sl = slice(0, None if limit <= 0 else limit * stride, stride)
    cache["poses"] = cache["poses"][sl]
    for key in ("ee_l_pos", "ee_l_quat", "ee_r_pos", "ee_r_quat"):
        if key in cache:
            cache[f"seed_{key}"] = cache[key][sl]
    if limit > 0:
        cache["poses"] = cache["poses"][:limit]
        for key in ("ee_l_pos", "ee_l_quat", "ee_r_pos", "ee_r_quat"):
            seed_key = f"seed_{key}"
            if seed_key in cache:
                cache[seed_key] = cache[seed_key][:limit]
    return cache


def _grid_bounds_from_cache(cache: dict, pad: float, source: str, seed_pool: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return workspace grid bounds for target-space IK sampling.

    `legacy-symmetric` follows the old dexterous-space sampler's dependency
    chain: build one grid from combined left/right cached EE positions, then use
    that same grid for both arms. X/Y are symmetrized around the root frame so
    front/back and left/right chunks select identical coordinates. Z remains the
    physical cache min/max range because workspace height is not symmetric about
    the base origin.

    `seed-pool` is kept for local debugging only; it uses the active arm's seed
    cloud and therefore must not be used for left/right comparisons.
    """
    if source == "seed-pool":
        seed_pos = seed_pool["pos"]
        return seed_pos.min(dim=0).values - pad, seed_pos.max(dim=0).values + pad
    if source != "legacy-symmetric":
        raise ValueError(f"unknown grid bounds source: {source}")
    all_ee = torch.cat([cache["ee_l_pos"], cache["ee_r_pos"]], dim=0).float()
    lo_raw = all_ee.min(dim=0).values - pad
    hi_raw = all_ee.max(dim=0).values + pad
    x_abs = torch.maximum(lo_raw[0].abs(), hi_raw[0].abs())
    y_abs = torch.maximum(lo_raw[1].abs(), hi_raw[1].abs())
    lo = torch.tensor([-float(x_abs), -float(y_abs), float(lo_raw[2])], dtype=torch.float32)
    hi = torch.tensor([float(x_abs), float(y_abs), float(hi_raw[2])], dtype=torch.float32)
    return lo, hi


def _grid_axes(
    lo: torch.Tensor,
    hi: torch.Tensor,
    n_grid: int,
    grid_spacing: float,
) -> tuple[list[torch.Tensor], tuple[int, int, int], torch.Tensor]:
    """Per-axis lattice POINTS (endpoint-inclusive), per-axis counts, and lattice origin.

    Two modes:
    * ``grid_spacing <= 0``: fixed count — ``linspace(lo, hi, n_grid)`` per axis
      (legacy; anisotropic when the bounds are non-cubic). Origin = ``lo``.
    * ``grid_spacing > 0``: fixed metric spacing (cubic voxels edge ``s``). For an axis whose
      bounds are symmetric about 0 (``|lo+hi| < s/2``, the legacy-symmetric X/Y case) the
      points are centered on 0: ``s*arange(-m, m+1)`` with ``m = round(max|lo|,|hi|/s)`` —
      symmetric about 0 and containing 0, so the L/R (y) and front/back (x) mirrors land on
      lattice points. Other axes (Z) use ``lo + s*arange(n)``.

    Returns ``origin`` = per-axis first point; voxel indexing must anchor to ``origin`` (NOT
    ``lo``) so ``round((p-origin)/s)`` is integer for the centered axes. These are POINTS, not
    cell centers; the hierarchical solve nests such lattices and warm-starts from the nearest
    coarse point.
    """
    if grid_spacing > 0.0:
        axes = []
        n_per_axis = []
        origin = []
        for i in range(3):
            lo_i, hi_i = float(lo[i]), float(hi[i])
            if abs(lo_i + hi_i) < 0.5 * grid_spacing:  # symmetric about 0 (X/Y)
                m = int(round(max(abs(lo_i), abs(hi_i)) / grid_spacing))
                pts = grid_spacing * torch.arange(-m, m + 1, dtype=torch.float32)
            else:
                # Anchor to integer multiples of the spacing (..., -2s, -s, 0, s, 2s, ...) exactly
                # as the symmetric X/Y branch above does, instead of starting at the raw padded
                # bound. Without this Z's origin inherits the cache extent (-0.3662 at pad 0.10,
                # i.e. -18.31 cells) and changing --pad slides every Z coordinate by a fraction of
                # a cell, so two payloads solved at different pads cannot be compared voxel to
                # voxel -- which is exactly what a K-to-K camera ablation does. Snapping makes the
                # lattice pad-INDEPENDENT: pad decides how many cells, never where they sit.
                lo_s = math.floor(lo_i / grid_spacing + 1e-9) * grid_spacing
                n_i = int(math.ceil((hi_i - lo_s) / grid_spacing - 1e-9)) + 1
                pts = lo_s + grid_spacing * torch.arange(n_i, dtype=torch.float32)
            axes.append(pts)
            n_per_axis.append(pts.shape[0])
            origin.append(float(pts[0]))
        return axes, (n_per_axis[0], n_per_axis[1], n_per_axis[2]), torch.tensor(origin, dtype=torch.float32)
    axes = [torch.linspace(float(lo[i]), float(hi[i]), n_grid) for i in range(3)]
    return axes, (n_grid, n_grid, n_grid), lo.clone().float()


def _ordered_grid_centers(
    lo: torch.Tensor,
    hi: torch.Tensor,
    n_grid: int,
    target_order: str,
    seed_pos: torch.Tensor,
    order_pos: torch.Tensor,
    grid_spacing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create voxel centers and deterministic voxel ids for a workspace grid."""
    if grid_spacing > 0.0 and target_order != "grid":
        raise ValueError(
            "spacing-based grid requires --target-order grid "
            "(legacy-symmetric ordering assumes a cubic n_grid count)"
        )
    axes, _, _ = _grid_axes(lo, hi, n_grid, grid_spacing)
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    centers = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    voxel_ids = torch.arange(centers.shape[0], dtype=torch.long)
    if target_order == "grid":
        return centers, voxel_ids
    if target_order == "center-out":
        order = torch.argsort(torch.linalg.norm(centers, dim=1))
    elif target_order == "nearest-seed":
        nearest_dist = _nearest_distance(centers, seed_pos)
        order = torch.argsort(nearest_dist)
    elif target_order == "legacy-nearest":
        nearest_dist = _nearest_distance(centers, order_pos)
        order = torch.argsort(nearest_dist)
    elif target_order == "legacy-symmetric-nearest":
        order = _legacy_symmetric_order(centers, n_grid, order_pos)
    else:
        raise ValueError(f"unknown target order: {target_order}")
    return centers[order], voxel_ids[order]


def _nearest_distance(points: torch.Tensor, refs: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    """Compute nearest reference distance without allocating full grid x ref matrix."""
    out = torch.empty(points.shape[0], dtype=torch.float32)
    refs = refs.float()
    for start in range(0, points.shape[0], batch_size):
        end = min(start + batch_size, points.shape[0])
        out[start:end] = torch.cdist(points[start:end].float(), refs).min(dim=1).values
    return out


def _legacy_order_positions(cache: dict) -> torch.Tensor:
    """Use the same L/R sampled EE cloud for both arms when ordering target chunks."""
    return torch.cat([cache["seed_ee_l_pos"], cache["seed_ee_r_pos"]], dim=0).float()


def _legacy_symmetric_order(centers: torch.Tensor, n_grid: int, order_pos: torch.Tensor) -> torch.Tensor:
    """Order chunks by old-cache proximity while preserving X/Y mirror groups."""
    nearest_dist = _nearest_distance(centers, order_pos)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for flat in range(centers.shape[0]):
        ix = flat // (n_grid * n_grid)
        iy = (flat // n_grid) % n_grid
        iz = flat % n_grid
        key = (min(ix, n_grid - 1 - ix), min(iy, n_grid - 1 - iy), iz)
        groups.setdefault(key, []).append(flat)

    def group_sort_key(key: tuple[int, int, int]) -> tuple[float, int, int, int]:
        members = groups[key]
        return (float(nearest_dist[members].min()), key[2], key[0], key[1])

    order: list[int] = []
    for key in sorted(groups, key=group_sort_key):
        order.extend(sorted(groups[key]))
    return torch.tensor(order, dtype=torch.long)


def _orientation_refs(n_orientations: int) -> np.ndarray:
    """Deterministic wxyz orientation-bin quaternions for target-grid smoke.

    This intentionally stays simple: yaw coverage first, then two pitch levels
    once more than four bins are requested. Roll stays zero for now because the
    first validation target is seeded cuRobo plumbing, not final SO(3) sampling.
    """
    if n_orientations <= 0:
        raise ValueError("n_orientations must be positive")
    if n_orientations == 1:
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    if n_orientations <= 4:
        yaws = np.linspace(-np.pi, np.pi, n_orientations, endpoint=False)
        pitches = np.array([0.0])
    else:
        n_pitch = 2
        n_yaw = int(np.ceil(n_orientations / n_pitch))
        yaws = np.linspace(-np.pi, np.pi, n_yaw, endpoint=False)
        pitches = np.linspace(-np.pi / 4.0, np.pi / 4.0, n_pitch)

    quats = []
    for yaw in yaws:
        for pitch in pitches:
            xyzw = Rotation.from_euler("zyx", [yaw, pitch, 0.0]).as_quat()
            quats.append([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            if len(quats) == n_orientations:
                return np.asarray(quats, dtype=np.float32)
    return np.asarray(quats, dtype=np.float32)


def _orientation_refs_so3(n_orientations: int, pair_x_reflection: bool = False) -> np.ndarray:
    """Near-uniform SO(3) orientation set as wxyz quaternions, shape (n, 4).

    Super-Fibonacci spiral (Alexa, "Super-Fibonacci Spirals", CVPR 2022): deterministic,
    low-discrepancy coverage of the full rotation group with no pole clustering or roll
    bias — unlike the RPY grid in `_orientation_refs`. ``pair_x_reflection`` is only for
    humanoid_v21's parallel gripper: it pairs every rotation with its tool-frame-correct image
    under base-frame X reflection, ``R' = F_x R F_y``. Other robots retain the unpaired
    Super-Fibonacci set because their tool-frame reflection convention is embodiment-specific.
    Used for the paper dexterity-D volume so D = (reachable / n) remains a fraction over one
    shared orientation set.
    """
    if n_orientations <= 0:
        raise ValueError("n_orientations must be positive")
    if pair_x_reflection:
        # A Z-half-turn maps to itself under F_x R F_y. It closes an odd-sized set
        # without introducing an unpaired orientation.
        fixed_xyzw = Rotation.from_euler("z", -np.pi / 2.0).as_quat()
        if n_orientations == 1:
            return np.array([[fixed_xyzw[3], *fixed_xyzw[:3]]], dtype=np.float32)
        n_base = n_orientations // 2
    else:
        n_base = n_orientations
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    q = np.empty((n_base, 4), dtype=np.float64)  # xyzw
    for i in range(n_base):
        s = i + 0.5
        r = np.sqrt(s / n_base)
        rr = np.sqrt(1.0 - s / n_base)
        a = 2.0 * np.pi * s / phi
        b = 2.0 * np.pi * s / psi
        q[i] = [r * np.sin(a), r * np.cos(a), rr * np.sin(b), rr * np.cos(b)]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if not pair_x_reflection:
        wxyz = np.empty_like(q)
        wxyz[:, 0] = q[:, 3]
        wxyz[:, 1:] = q[:, :3]
        return wxyz.astype(np.float32)
    rot = Rotation.from_quat(q).as_matrix()
    reflect_base_x = np.diag([-1.0, 1.0, 1.0])
    reflect_tool_y = np.diag([1.0, -1.0, 1.0])
    mirror = Rotation.from_matrix(reflect_base_x @ rot @ reflect_tool_y).as_quat()
    paired = np.empty((2 * n_base, 4), dtype=np.float64)
    paired[0::2] = q
    paired[1::2] = mirror
    if n_orientations % 2:
        paired = np.vstack((paired, fixed_xyzw))
    wxyz = np.empty_like(paired)
    wxyz[:, 0] = paired[:, 3]
    wxyz[:, 1:] = paired[:, :3]
    return wxyz.astype(np.float32)


def _orientation_refs_from_seed_pool(seed_pool: dict, n_orientations: int) -> np.ndarray:
    """Use reachable seed-pool gripper quaternions as orientation bins."""
    if n_orientations <= 0:
        raise ValueError("n_orientations must be positive")
    seed_quat = seed_pool["quat"]
    if seed_quat.shape[0] == 0:
        raise ValueError("empty seed pool")
    idx = torch.linspace(0, seed_quat.shape[0] - 1, n_orientations).round().long()
    return seed_quat[idx].numpy().astype(np.float32)


def _joint_id_by_suffix(model: mujoco.MjModel, name: str) -> int:
    for jid in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if n is not None and (n == name or n.endswith(f"/{name}") or n.split("/")[-1] == name):
            return jid
    raise ValueError(f"joint not found in MuJoCo model: {name}")


def _site_id(model: mujoco.MjModel, side: str, spec: RobotSpec) -> int:
    name = spec.site_name(side)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"site not found: {name}")
    return sid


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, side: str, spec: RobotSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return EE pose in cuRobo's base frame.

    MuJoCo `site_xpos` is world-frame after the floating base is placed at
    `spec.home_root_pos`. The cuRobo arm cfg locks the root at its root frame, so
    target positions must subtract that home base translation. Base orientation is
    identity in `_write_seed_qpos`, so no rotation transform is needed here.
    """
    sid = _site_id(model, side, spec)
    pos = data.site_xpos[sid].copy() - np.asarray(spec.home_root_pos, dtype=np.float64)
    mat = data.site_xmat[sid].reshape(3, 3).copy()
    xyzw = Rotation.from_matrix(mat).as_quat()
    quat = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    return pos.astype(np.float32), quat


def _seed_pool_from_cache(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cache: dict,
    side: str,
    spec: RobotSpec,
) -> dict:
    """FK cached joint poses through full welded model to build target seeds."""
    poses = cache["poses"]
    qpos_indices = cache["qpos_indices"]
    pos = torch.empty((poses.shape[0], 3), dtype=torch.float32)
    quat = torch.empty((poses.shape[0], 4), dtype=torch.float32)
    for i, pose in enumerate(poses):
        _write_seed_qpos(model, data, qpos_indices, pose, spec)
        p, q = _site_pose(model, data, side, spec)
        pos[i] = torch.from_numpy(p)
        quat[i] = torch.from_numpy(q)
    return {"poses": poses, "qpos_indices": qpos_indices, "pos": pos, "quat": quat}


def _grid_targets_from_seed_pool(
    seed_pool: dict,
    cache: dict,
    n_grid: int,
    refs: np.ndarray,
    pad: float,
    max_targets: int,
    target_start: int,
    grid_bounds_source: str,
    target_order: str,
    grid_spacing: float = 0.0,
    gate_voxel_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create target poses from a shared workspace grid, optionally after position gating."""
    if n_grid <= 0 and grid_spacing <= 0.0:
        raise ValueError("n_grid must be positive")
    if target_start < 0:
        raise ValueError("target_start must be nonnegative")
    seed_pos = seed_pool["pos"]
    lo, hi = _grid_bounds_from_cache(cache, pad, grid_bounds_source, seed_pool)
    order_pos = _legacy_order_positions(cache)
    centers, voxel_ids = _ordered_grid_centers(
        lo, hi, n_grid, target_order, seed_pos, order_pos, grid_spacing
    )
    if gate_voxel_indices is not None:
        keep = torch.isin(voxel_ids, gate_voxel_indices)
        centers, voxel_ids = centers[keep], voxel_ids[keep]
        if centers.numel() == 0:
            raise ValueError("position gate excludes every grid voxel")
    refs_t = torch.from_numpy(refs)

    target_pos = centers.repeat_interleave(refs_t.shape[0], dim=0)
    target_quat = refs_t.repeat(centers.shape[0], 1)
    voxel_index = voxel_ids.repeat_interleave(refs_t.shape[0])
    orient_index = torch.arange(refs_t.shape[0]).repeat(centers.shape[0])
    stop = None if max_targets <= 0 else target_start + max_targets
    target_pos = target_pos[target_start:stop]
    target_quat = target_quat[target_start:stop]
    voxel_index = voxel_index[target_start:stop]
    orient_index = orient_index[target_start:stop]
    return target_pos, target_quat, voxel_index, orient_index


def _load_position_gate(
    path: pathlib.Path | None,
    *,
    robot: str,
    grid_spacing: float,
    pad: float,
    grid_lo: torch.Tensor,
    grid_hi: torch.Tensor,
    n_grid_per_axis: tuple[int, int, int],
) -> tuple[torch.Tensor | None, dict]:
    """Load a reproducible position-only gate and reject any lattice mismatch."""
    if path is None:
        return None, {}
    gate = torch.load(str(path), map_location="cpu", weights_only=False)
    checks = {
        "robot": robot,
        "grid_spacing": float(grid_spacing),
        "pad": float(pad),
        "n_grid_per_axis": torch.tensor(n_grid_per_axis, dtype=torch.long),
        "grid_lo": grid_lo.float(),
        "grid_hi": grid_hi.float(),
    }
    for key, expected in checks.items():
        actual = gate.get(key)
        if actual is None:
            raise ValueError(f"position gate missing {key}: {path}")
        if isinstance(expected, torch.Tensor):
            if not torch.allclose(torch.as_tensor(actual), expected):
                raise ValueError(f"position gate {key} mismatch: {path}")
        elif actual != expected:
            raise ValueError(f"position gate {key} mismatch: {path}")
    voxel_indices = torch.unique(torch.as_tensor(gate["voxel_indices"], dtype=torch.long), sorted=True)
    if voxel_indices.numel() == 0 or int(voxel_indices[-1]) >= int(np.prod(n_grid_per_axis)):
        raise ValueError(f"position gate voxel indices invalid: {path}")
    return voxel_indices, {
        "position_gate": str(path.resolve()),
        "position_gate_voxels": int(voxel_indices.numel()),
        "position_gate_total_voxels": int(np.prod(n_grid_per_axis)),
        "position_gate_halo": float(gate.get("halo", float("nan"))),
        "position_gate_coarse_workspace": str(gate.get("coarse_workspace", "")),
    }


def _grid_targets_from_local_seeds(
    seed_pool: dict,
    cache: dict,
    n_grid: int,
    n_orientations: int,
    pad: float,
    max_targets: int,
    target_start: int,
    target_alpha: float,
    grid_bounds_source: str,
    target_order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create grid targets whose orientations/seeds come from nearest local cache poses.

    For each voxel center, pick the nearest `n_orientations` seed-pool poses by
    EE position. Each selected seed contributes both the target orientation and
    the IK seed. This is the practical dexterity smoke mode: it tests whether
    cuRobo can move a nearby full-gripper pose to the voxel center while
    preserving one locally observed gripper attitude.
    """
    if n_grid <= 0:
        raise ValueError("n_grid must be positive")
    if n_orientations <= 0:
        raise ValueError("n_orientations must be positive")
    if target_start < 0:
        raise ValueError("target_start must be nonnegative")
    if target_alpha < 0.0 or target_alpha > 1.0:
        raise ValueError("target_alpha must be in [0, 1]")
    seed_pos = seed_pool["pos"]
    seed_quat = seed_pool["quat"]
    lo, hi = _grid_bounds_from_cache(cache, pad, grid_bounds_source, seed_pool)
    order_pos = _legacy_order_positions(cache)
    centers, voxel_ids = _ordered_grid_centers(lo, hi, n_grid, target_order, seed_pos, order_pos)

    dist = torch.cdist(centers, seed_pos)

    k = min(n_orientations, seed_pos.shape[0])
    local_seed_idx = torch.topk(dist, k=k, largest=False, dim=1).indices
    voxel_center_pos = centers.repeat_interleave(k, dim=0)
    selected_seed_idx = local_seed_idx.reshape(-1)
    selected_seed_pos = seed_pos[selected_seed_idx]
    if target_alpha == 0.0:
        target_pos = selected_seed_pos.clone()
    elif target_alpha == 1.0:
        target_pos = voxel_center_pos.clone()
    else:
        target_pos = selected_seed_pos + target_alpha * (voxel_center_pos - selected_seed_pos)
    target_quat = seed_quat[selected_seed_idx].clone()
    voxel_index = voxel_ids.repeat_interleave(k)
    orient_index = torch.arange(k).repeat(centers.shape[0])
    seed_indices = selected_seed_idx.clone()
    seed_target_distance = torch.linalg.norm(target_pos - selected_seed_pos, dim=1)

    stop = None if max_targets <= 0 else target_start + max_targets
    target_pos = target_pos[target_start:stop]
    target_quat = target_quat[target_start:stop]
    voxel_index = voxel_index[target_start:stop]
    orient_index = orient_index[target_start:stop]
    seed_indices = seed_indices[target_start:stop]
    voxel_center_pos = voxel_center_pos[target_start:stop]
    seed_target_distance = seed_target_distance[target_start:stop]
    return (
        target_pos,
        target_quat,
        voxel_index,
        orient_index,
        seed_indices,
        voxel_center_pos,
        seed_target_distance,
    )


def _nearest_seed_indices(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    seed_pool: dict,
    orientation_weight: float,
) -> torch.Tensor:
    """Pick nearest cache seed in position plus quaternion-bin distance."""
    return _seed_candidate_indices(
        target_pos, target_quat, seed_pool, orientation_weight, attempts=1
    ).squeeze(1)


def _seed_candidate_indices(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    seed_pool: dict,
    orientation_weight: float,
    attempts: int,
    preferred: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick one or more IK seeds per target, nearest in position plus orientation.

    Direct cuRobo IK can return a false negative from one local basin even when a
    neighboring cached pose solves the same target. Keep the locally chosen seed
    first when provided, then try the next best distinct candidates. Score rows in
    bounded batches: production shards contain millions of targets, so a Python
    loop over targets makes seed assignment dominate the GPU solve.
    """
    if attempts <= 0:
        raise ValueError("seed attempts must be positive")
    seed_pos = seed_pool["pos"]
    seed_quat = seed_pool["quat"]
    k = min(attempts, seed_pos.shape[0])
    out = torch.empty((target_pos.shape[0], k), dtype=torch.long, device=target_pos.device)
    preferred_t = None if preferred is None else preferred.to(device=target_pos.device, dtype=torch.long)
    batch_size = 16_384
    seed_quat_t = seed_quat.T
    for start in range(0, target_pos.shape[0], batch_size):
        stop = min(start + batch_size, target_pos.shape[0])
        pos_dist = torch.cdist(target_pos[start:stop], seed_pos)
        quat_dist = 1.0 - torch.abs(target_quat[start:stop] @ seed_quat_t)
        scores = pos_dist + orientation_weight * quat_dist
        if preferred_t is None:
            out[start:stop] = torch.topk(scores, k=k, largest=False).indices
            continue
        selected = preferred_t[start:stop]
        out[start:stop, 0] = selected
        if k > 1:
            scores.scatter_(1, selected.unsqueeze(1), float("inf"))
            out[start:stop, 1:] = torch.topk(scores, k=k - 1, largest=False).indices
    return out


def _seed_candidates_for_solver(
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    seed_pool: dict,
    orientation_weight: float,
    attempts: int,
    solver: str,
    preferred: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return real candidates for sequential IK or schema placeholders for batched IK.

    ``_solve_targets_batched`` seeds every direct GPU IK call from one fixed home pose, so it
    never reads per-target candidates. Keep one zero column for payload compatibility instead of
    spending CPU time ranking millions of unused rows.
    """
    if solver == "batched":
        return torch.zeros((target_pos.shape[0], 1), dtype=torch.long)
    return _seed_candidate_indices(
        target_pos, target_quat, seed_pool, orientation_weight, attempts, preferred=preferred,
    )


def _active_dof_indices(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    return np.array([model.jnt_dofadr[_joint_id_by_suffix(model, j)] for j in joint_names], dtype=np.int64)


def _write_seed_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_indices: torch.Tensor,
    pose: torch.Tensor,
    spec: RobotSpec,
) -> None:
    data.qpos[:] = model.qpos0
    # Match generator behavior: start from standing HOME base if model has a free root.
    data.qpos[:3] = np.asarray(spec.home_root_pos, dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[qpos_indices.cpu().numpy()] = pose.cpu().numpy()
    spec.apply_source_coupling(model, data)
    mujoco.mj_forward(model, data)


def _idle_tool_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    planner: CuroboArmPlanner,
    idle_frame: str,
    spec: RobotSpec,
    seed_state: JointState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed idle-tool pose, using source rest only for robots that require it.

    Apollo's cache samples are safe but arbitrary asymmetric dual-arm configurations. Its idle
    hand must instead be pinned to source ``qpos0`` so every target is evaluated against the
    user-approved symmetric-down arm. Other robots retain their cache-home idle convention.
    """
    if spec.idle_from_source_qpos0:
        data.qpos[:] = model.qpos0
        data.qpos[:3] = np.asarray(spec.home_root_pos, dtype=np.float64)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        spec.apply_source_coupling(model, data)
        mujoco.mj_forward(model, data)
        idle_state = planner.current_state_from_qpos(data.qpos)
    else:
        idle_state = seed_state
    pose = planner._kin.compute_kinematics(idle_state).tool_poses.get_link_pose(idle_frame)
    return pose.position[0], pose.quaternion[0]


def _goal_pose_from_seed(
    planner: CuroboArmPlanner,
    current_state: JointState,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    side: str,
    spec: RobotSpec,
    idle_pose: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> GoalToolPose:
    """Build 2-tool goal: active EE to target, idle EE pinned at supplied/rest FK pose."""
    _require_curobo()
    side = side.upper()
    idle = "L" if side == "R" else "R"
    idle_frame = spec.tool_frame(idle)
    active_frame = spec.tool_frame(side)
    if idle_pose is None:
        kin_state = planner._kin.compute_kinematics(current_state)
        source_idle_pose = kin_state.tool_poses.get_link_pose(idle_frame)
        idle_position, idle_quaternion = source_idle_pose.position[0], source_idle_pose.quaternion[0]
    else:
        idle_position, idle_quaternion = idle_pose
    device = planner._device
    position = torch.empty((1, 1, 2, 1, 3), device=device, dtype=torch.float32)
    quaternion = torch.empty((1, 1, 2, 1, 4), device=device, dtype=torch.float32)
    position[0, 0, 0, 0] = idle_position
    quaternion[0, 0, 0, 0] = idle_quaternion
    position[0, 0, 1, 0] = torch.as_tensor(target_pos, device=device, dtype=torch.float32)
    quaternion[0, 0, 1, 0] = torch.as_tensor(target_quat, device=device, dtype=torch.float32)
    return GoalToolPose(
        tool_frames=[idle_frame, active_frame],
        position=position,
        quaternion=quaternion,
    )


def _reach_sphere(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    active_dof_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Sphere outside which the tool site is unreachable for EVERY joint configuration.

    Centre is the arm-base body's world origin; radius is the sum of the link offsets from that body
    down to the tool site. A serial chain cannot place its tip farther from its base than the sum of
    its link lengths, so a target outside the sphere provably has no IK solution and need not be
    dispatched at all. On the pad-0.10 humanoid_v21 grid that is 59% of all rows -- validated
    against the shipped payload, where 0 of 17.9 M solved rows succeeded beyond the bound and the
    largest successful radius was 0.7171 m against a 0.7180 m bound.

    Every translation in the chain is counted at full magnitude regardless of joint angle, so the
    radius is an upper bound rather than an estimate; joint anchors are included for models that
    offset them from body origins (humanoid_v21 does not -- its `jnt_pos` sum is exactly 0).

    Rejected alternative: centring on the floating base needs no assumption about which joints are
    locked, but inflates the radius by the shoulder offset (0.718 -> 1.075 m), which encloses nearly
    the whole grid and filters almost nothing. The assert below is what buys the tighter centre.
    """
    sid = model.site(site_name).id
    chain: list[int] = []
    b = int(model.site_bodyid[sid])
    while b != 0:
        chain.append(b)
        b = int(model.body_parentid[b])
    arm_base = chain[-2] if len(chain) >= 2 else chain[-1]

    # The centre is read once at qpos0, so it must not depend on any joint being solved for.
    active = set(int(i) for i in np.asarray(active_dof_idx).ravel())
    b = int(model.body_parentid[arm_base])
    while b != 0:
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            dofs = set(range(model.jnt_dofadr[j], model.jnt_dofadr[j] + 6))
            assert not (dofs & active), (
                f"{model.body(b).name} carries an active DOF above the arm base "
                f"{model.body(arm_base).name}; the reach-sphere centre would move"
            )
        b = int(model.body_parentid[b])

    radius = float(np.linalg.norm(model.site(sid).pos))
    for bb in chain[: chain.index(arm_base)]:
        radius += float(np.linalg.norm(model.body(bb).pos))
        for j in range(model.body_jntadr[bb], model.body_jntadr[bb] + model.body_jntnum[bb]):
            radius += float(np.linalg.norm(model.jnt_pos[j]))

    # Targets are expressed in the base-link frame (the grid is built with the base at the origin),
    # while `data` here has the floating base at `home_root_pos` -- so world `xpos` would offset the
    # centre by the root height and filter the wrong shell. Express the centre in the base frame.
    base = chain[-1]
    centre = data.xmat[base].reshape(3, 3).T @ (data.xpos[arm_base] - data.xpos[base])
    return centre, radius


def _jacobian_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    active_dof_idx: np.ndarray,
    spec: RobotSpec,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, _site_id(model, side, spec))
    jv = jacp[:, active_dof_idx]
    jw = jacr[:, active_dof_idx]
    sv = np.linalg.svd(jv, compute_uv=False)
    sw = np.linalg.svd(jw, compute_uv=False)
    mt = float(np.sqrt(max(0.0, np.linalg.det(jv @ jv.T))))
    mr = float(np.sqrt(max(0.0, np.linalg.det(jw @ jw.T))))
    return sv.astype(np.float32), sw.astype(np.float32), mt, mr, float(sv[-1]), float(sw[-1])


def _build_batched_ik_solver(scene_dict: dict, num_seeds: int, max_batch: int, robot_cfg_dict: dict) -> IKSolver:
    """Dedicated IK solver that solves ``max_batch`` targets per ``solve_pose`` call.

    Same welded robot_cfg + table scene + success gate (0.01 m / 0.5 rad) the motion
    planner's ik_solver uses, but ``max_batch_size > 1`` so cuRobo solves the whole batch
    in one GPU call. This is the speed win vs the per-target loop (``_solve_targets``),
    which fires one batch=1 solve per target and leaves the GPU ~88% idle. ``robot_cfg_dict``
    is the per-robot cuRobo cfg (humanoid_v21 or G1).
    """
    _require_curobo()
    cfg = IKSolverCfg.create(
        robot=robot_cfg_dict,
        scene_model=scene_dict,
        num_seeds=num_seeds,
        position_tolerance=_POSITION_TOLERANCE,
        orientation_tolerance=_ORIENTATION_TOLERANCE,
        self_collision_check=True,
        use_cuda_graph=True,
        max_batch_size=max_batch,
        max_goalset=1,
    )
    return IKSolver(cfg)


def _solve_targets_batched(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    planner: CuroboArmPlanner,
    active_dof_idx: np.ndarray,
    seed_pool: dict,
    seed_candidate_indices: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    side: str,
    source: str,
    out: pathlib.Path,
    extra: dict,
    log_every: int,
    batch_size: int,
    num_seeds: int,
    spec: RobotSpec,
    position_only_active: bool = False,
) -> None:
    """Batched IK: solve ``batch_size`` targets per GPU call, then Jacobian metrics on CPU.

    Payload schema is IDENTICAL to ``_solve_targets`` so the plotter / merge script are
    unchanged. Two deliberate simplifications vs the sequential path (measured 99.8%
    per-target agreement on the coarse g10 payload):

    * The idle-hand pin is pinned at ONE constant home pose (``seed_pool[0]``), not each
      target's nearest safe pose.
    * ``num_seeds`` defaults below the sequential 512. Fewer seeds trade a few percent of
      marginal-voxel recall for the batch parallelism that makes 26^3 grids tractable.

    ``seed_candidate_indices`` / ``winning_seed_indices`` are kept in the payload for schema
    compatibility but carry no per-target seed meaning here (filled 0 / -1).
    """
    _require_curobo()
    n = target_pos.shape[0]
    n_dof = len(planner.joint_names)
    device = planner._device
    idle = "L" if side.upper() == "R" else "R"
    idle_frame = spec.tool_frame(idle)
    active_frame = spec.tool_frame(side)

    # Safe cache pose initializes optimization. Apollo's idle goal separately uses source qpos0.
    _write_seed_qpos(model, data, seed_pool["qpos_indices"], seed_pool["poses"][0], spec)
    home = planner.current_state_from_qpos(data.qpos)
    idle_pos, idle_quat = _idle_tool_pose(model, data, planner, idle_frame, spec, home)
    idle_pos = idle_pos.to(device)
    idle_quat = idle_quat.to(device)
    seed_cfg = home.position.view(1, 1, -1).to(device)

    solver = _build_batched_ik_solver(
        planner._scene_dict, num_seeds, batch_size, robot_cfg_dict_for_side(spec, side)
    )
    if position_only_active:
        # Feasibility gate: active tool has only XYZ constraints. Keep idle hand fully
        # pinned so this matches final two-tool collision and joint-limit problem.
        solver.update_tool_pose_criteria(
            {
                idle_frame: ToolPoseCriteria.track_position_and_orientation(),
                active_frame: ToolPoseCriteria.track_position(),
            }
        )
    tgt_pos = target_pos.to(device=device, dtype=torch.float32)
    tgt_quat = target_quat.to(device=device, dtype=torch.float32)

    q_solution = torch.full((n, n_dof), float("nan"), dtype=torch.float32)
    success = torch.zeros(n, dtype=torch.bool)
    pos_err = torch.full((n,), float("nan"), dtype=torch.float32)
    rot_err = torch.full((n,), float("nan"), dtype=torch.float32)
    solve_time = torch.full((n,), float("nan"), dtype=torch.float32)
    sigma_trans = torch.full((n, 3), float("nan"), dtype=torch.float32)
    sigma_rot = torch.full((n, 3), float("nan"), dtype=torch.float32)
    manip_trans = torch.full((n,), float("nan"), dtype=torch.float32)
    manip_rot = torch.full((n,), float("nan"), dtype=torch.float32)
    sigma_min_trans = torch.full((n,), float("nan"), dtype=torch.float32)
    sigma_min_rot = torch.full((n,), float("nan"), dtype=torch.float32)

    # Dispatch only rows that are not provably out of reach. Skipped rows keep their initialized
    # values (success False, errors NaN) -- identical to what a failed solve would leave behind,
    # because a target outside the reach sphere HAS no solution. Row order and payload shape are
    # untouched, so chunks remain merge-compatible with payloads solved before this filter existed.
    centre, radius = _reach_sphere(model, data, active_frame, active_dof_idx)
    centre_t = torch.as_tensor(centre, dtype=torch.float32, device=device)
    feasible = (tgt_pos - centre_t).norm(dim=1) <= radius
    order = torch.nonzero(feasible).flatten()
    n_solve = int(order.numel())
    print(f"reach sphere r={radius:.4f} about {np.round(centre, 4).tolist()}: "
          f"dispatching {n_solve:,}/{n:,} rows ({100.0 * (1 - n_solve / max(n, 1)):.1f}% skipped)",
          flush=True)

    t0 = time.time()
    for start in range(0, n_solve, batch_size):
        stop = min(start + batch_size, n_solve)
        sel = order[start:stop]
        b = stop - start
        position = torch.empty((b, 1, 2, 1, 3), device=device, dtype=torch.float32)
        quaternion = torch.empty((b, 1, 2, 1, 4), device=device, dtype=torch.float32)
        position[:, 0, 0, 0] = idle_pos
        quaternion[:, 0, 0, 0] = idle_quat
        position[:, 0, 1, 0] = tgt_pos[sel]
        quaternion[:, 0, 1, 0] = tgt_quat[sel]
        goal = GoalToolPose(tool_frames=[idle_frame, active_frame], position=position, quaternion=quaternion)
        seed_batch = seed_cfg.expand(b, 1, -1).clone()
        res = solver.solve_pose(goal, seed_config=seed_batch, return_seeds=1)

        ok = res.success.reshape(b, -1)[:, 0].cpu()
        sel_cpu = sel.cpu()
        success[sel_cpu] = ok
        pos_err[sel_cpu] = res.position_error.reshape(b, -1)[:, 0].float().cpu()
        rot_err[sel_cpu] = res.rotation_error.reshape(b, -1)[:, 0].float().cpu()
        st = float(getattr(res, "solve_time", float("nan")))
        q_batch = res.solution.detach().reshape(b, -1, n_dof)[:, 0].cpu().numpy()
        for j in range(b):
            i = int(sel_cpu[j])
            solve_time[i] = st
            if not bool(ok[j]):
                continue
            q_curobo = q_batch[j].astype(np.float32)
            q_solution[i] = torch.from_numpy(q_curobo)
            data.qpos[:] = model.qpos0
            data.qpos[:3] = np.asarray(spec.home_root_pos, dtype=np.float64)
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            data.qpos[planner._qpos_adr] = planner.to_mjlab_qpos(q_curobo)
            spec.apply_source_coupling(model, data)
            mujoco.mj_forward(model, data)
            sv, sw, mt, mr, smin_t, smin_r = _jacobian_metrics(model, data, side, active_dof_idx, spec)
            sigma_trans[i] = torch.from_numpy(sv)
            sigma_rot[i] = torch.from_numpy(sw)
            manip_trans[i] = mt
            manip_rot[i] = mr
            sigma_min_trans[i] = smin_t
            sigma_min_rot[i] = smin_r
        if log_every > 0:
            el = time.time() - t0
            print(f"{stop}/{n_solve} ok={int(success.sum())} {stop / el:.0f} solves/s elapsed={el:.1f}s")

    winning_seed_indices = torch.full((n,), -1, dtype=torch.long)
    attempt_count = torch.ones(n, dtype=torch.long)
    _save_workspace_payload(
        out, planner, side, source, target_pos, target_quat, seed_candidate_indices,
        winning_seed_indices, attempt_count, success, q_solution, pos_err, rot_err, solve_time,
        sigma_trans, sigma_rot, manip_trans, manip_rot, sigma_min_trans, sigma_min_rot, extra, spec,
    )


def _save_workspace_payload(
    out, planner, side, source, target_pos, target_quat, seed_candidate_indices,
    winning_seed_indices, attempt_count, success, q_solution, pos_err, rot_err, solve_time,
    sigma_trans, sigma_rot, manip_trans, manip_rot, sigma_min_trans, sigma_min_rot, extra, spec,
) -> None:
    """Write the workspace sidecar. Shared by sequential + batched solve paths."""
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot": spec.name,
        "model": spec.model_tag,
        "side": side.upper(),
        "source": source,
        "joint_names": planner.joint_names,
        "target_pos": target_pos,
        "target_quat": target_quat,
        "seed_indices": seed_candidate_indices[:, 0],
        "seed_candidate_indices": seed_candidate_indices,
        "winning_seed_indices": winning_seed_indices,
        "attempt_count": attempt_count,
        "success": success,
        "q_solution": q_solution,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "solve_time": solve_time,
        "sigma_trans": sigma_trans,
        "sigma_rot": sigma_rot,
        "manip_trans": manip_trans,
        "manip_rot": manip_rot,
        "sigma_min_trans": sigma_min_trans,
        "sigma_min_rot": sigma_min_rot,
    }
    payload.update(extra)
    torch.save(payload, str(out))
    print(f"saved {out}")
    print(f"success {int(success.sum())}/{success.shape[0]}")


def _solve_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    planner: CuroboArmPlanner,
    active_dof_idx: np.ndarray,
    seed_pool: dict,
    seed_candidate_indices: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    side: str,
    source: str,
    out: pathlib.Path,
    extra: dict,
    log_every: int,
    spec: RobotSpec,
) -> None:
    qpos_indices = seed_pool["qpos_indices"]
    seed_poses = seed_pool["poses"]
    n = target_pos.shape[0]
    if seed_candidate_indices.ndim == 1:
        seed_candidate_indices = seed_candidate_indices.unsqueeze(1)
    q_solution = torch.full((n, len(planner.joint_names)), float("nan"), dtype=torch.float32)
    success = torch.zeros(n, dtype=torch.bool)
    winning_seed_indices = torch.full((n,), -1, dtype=torch.long)
    attempt_count = torch.zeros(n, dtype=torch.long)
    pos_err = torch.full((n,), float("nan"), dtype=torch.float32)
    rot_err = torch.full((n,), float("nan"), dtype=torch.float32)
    solve_time = torch.full((n,), float("nan"), dtype=torch.float32)
    sigma_trans = torch.full((n, 3), float("nan"), dtype=torch.float32)
    sigma_rot = torch.full((n, 3), float("nan"), dtype=torch.float32)
    manip_trans = torch.full((n,), float("nan"), dtype=torch.float32)
    manip_rot = torch.full((n,), float("nan"), dtype=torch.float32)
    sigma_min_trans = torch.full((n,), float("nan"), dtype=torch.float32)
    sigma_min_rot = torch.full((n,), float("nan"), dtype=torch.float32)
    fixed_idle_pose = None
    if spec.idle_from_source_qpos0:
        data.qpos[:] = model.qpos0
        data.qpos[:3] = np.asarray(spec.home_root_pos, dtype=np.float64)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        spec.apply_source_coupling(model, data)
        mujoco.mj_forward(model, data)
        rest_state = planner.current_state_from_qpos(data.qpos)
        fixed_idle_pose = _idle_tool_pose(
            model, data, planner, spec.tool_frame("L" if side.upper() == "R" else "R"), spec, rest_state,
        )

    t0 = time.time()
    for i in range(n):
        pos = target_pos[i].numpy()
        quat = target_quat[i].numpy()
        total_solve_time = 0.0
        for attempt_i, seed_tensor in enumerate(seed_candidate_indices[i]):
            seed_i = int(seed_tensor)
            attempt_count[i] = attempt_i + 1
            _write_seed_qpos(model, data, qpos_indices, seed_poses[seed_i], spec)
            current_state = planner.current_state_from_qpos(data.qpos)
            goal = _goal_pose_from_seed(planner, current_state, pos, quat, side, spec, fixed_idle_pose)
            seed_config = current_state.position.view(1, 1, -1).clone()
            result = planner._planner.ik_solver.solve_pose(
                goal,
                current_state=current_state,
                seed_config=seed_config,
                return_seeds=1,
            )
            ok = bool(result.success.flatten()[0].item())
            pos_err[i] = float(result.position_error.flatten()[0].item())
            rot_err[i] = float(result.rotation_error.flatten()[0].item())
            step_solve_time = float(getattr(result, "solve_time", float("nan")))
            if np.isfinite(step_solve_time):
                total_solve_time += step_solve_time
                solve_time[i] = total_solve_time
            if not ok:
                continue
            success[i] = True
            winning_seed_indices[i] = seed_i
            q_curobo = result.solution.reshape(-1, len(planner.joint_names))[0].detach().cpu().numpy()
            q_solution[i] = torch.from_numpy(q_curobo.astype(np.float32))
            data.qpos[:] = model.qpos0
            data.qpos[:3] = np.asarray(spec.home_root_pos, dtype=np.float64)
            data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            data.qpos[planner._qpos_adr] = planner.to_mjlab_qpos(q_curobo)
            spec.apply_source_coupling(model, data)
            mujoco.mj_forward(model, data)
            sv, sw, mt, mr, smin_t, smin_r = _jacobian_metrics(model, data, side, active_dof_idx, spec)
            sigma_trans[i] = torch.from_numpy(sv)
            sigma_rot[i] = torch.from_numpy(sw)
            manip_trans[i] = mt
            manip_rot[i] = mr
            sigma_min_trans[i] = smin_t
            sigma_min_rot[i] = smin_r
            break
        should_log = log_every > 0 and (
            i == 0 or i + 1 == n or (i + 1) % log_every == 0 or not bool(success[i])
        )
        if should_log:
            print(
                f"{i + 1:4d}/{n:<4d} ok={int(success[i])} "
                f"seed={int(winning_seed_indices[i]) if success[i] else int(seed_candidate_indices[i, 0]):<5d} "
                f"tries={int(attempt_count[i])}/{seed_candidate_indices.shape[1]} "
                f"pos={pos_err[i].item():.5f} rot={rot_err[i].item():.5f} "
                f"mt={manip_trans[i].item():.5f} elapsed={time.time() - t0:.1f}s"
            )

    _save_workspace_payload(
        out, planner, side, source, target_pos, target_quat, seed_candidate_indices,
        winning_seed_indices, attempt_count, success, q_solution, pos_err, rot_err, solve_time,
        sigma_trans, sigma_rot, manip_trans, manip_rot, sigma_min_trans, sigma_min_rot, extra, spec,
    )


def run(
    limit: int,
    stride: int,
    side: str,
    device: str,
    out: pathlib.Path,
    target_source: str,
    n_grid: int,
    n_orientations: int,
    seed_pool_size: int,
    pad: float,
    orientation_seed_weight: float,
    orientation_source: str,
    seed_attempts: int,
    target_alpha: float,
    target_start: int,
    grid_bounds_source: str,
    target_order: str,
    log_every: int,
    solver: str = "batched",
    batch_size: int = 256,
    num_seeds: int = 32,
    grid_spacing: float = 0.0,
    robot: str = "humanoid_v21",
    position_gate: pathlib.Path | None = None,
    rig: str = "dual",
) -> None:
    _require_curobo()
    spec = get_robot_spec(robot, rig)
    model = spec.model_loader()
    data = mujoco.MjData(model)
    planner = CuroboArmPlanner(
        model,
        device=device,
        max_attempts=1,
        enable_graph_attempt=0,
        hand_z_floor=None,
        robot_cfg_dict=robot_cfg_dict_for_side(spec, side),
        base_link=spec.base_link,
        home_base_pos=spec.home_root_pos,
    )
    active_dof_idx = _active_dof_indices(model, planner.joint_names)

    if target_source == "cache":
        cache = _load_cache(spec.cache_path, limit, stride)
        seed_pool = _seed_pool_from_cache(model, data, cache, side, spec)
        target_pos = seed_pool["pos"]
        target_quat = seed_pool["quat"]
        preferred = torch.arange(target_pos.shape[0], dtype=torch.long)
        seed_candidate_indices = _seed_candidates_for_solver(
            target_pos,
            target_quat,
            seed_pool,
            orientation_seed_weight,
            seed_attempts,
            solver,
            preferred=preferred,
        )
        extra = {"limit": limit, "stride": stride, "seed_attempts": seed_attempts}
        source = "seeded_curobo_direct_ik_from_safe_arm_pose_targets"
    elif target_source == "grid":
        pool_limit = seed_pool_size if seed_pool_size > 0 else max(limit, 1)
        cache = _load_cache(spec.cache_path, pool_limit, stride)
        seed_pool = _seed_pool_from_cache(model, data, cache, side, spec)
        grid_lo, grid_hi = _grid_bounds_from_cache(cache, pad, grid_bounds_source, seed_pool)
        _, n_grid_per_axis, grid_origin = _grid_axes(grid_lo, grid_hi, n_grid, grid_spacing)
        gate_voxel_indices, gate_extra = _load_position_gate(
            position_gate,
            robot=spec.name,
            grid_spacing=grid_spacing,
            pad=pad,
            grid_lo=grid_lo,
            grid_hi=grid_hi,
            n_grid_per_axis=n_grid_per_axis,
        )
        if orientation_source == "local-seed":
            if gate_voxel_indices is not None:
                raise ValueError("position gate currently supports fixed/so3/seed-pool/nearest-seed grid targets")
            (
                target_pos,
                target_quat,
                voxel_index,
                orient_index,
                seed_indices,
                voxel_center_pos,
                seed_target_distance,
            ) = (
                _grid_targets_from_local_seeds(
                    seed_pool,
                    cache,
                    n_grid,
                    n_orientations,
                    pad,
                    limit,
                    target_start,
                    target_alpha,
                    grid_bounds_source,
                    target_order,
                )
            )
            seed_candidate_indices = _seed_candidates_for_solver(
                target_pos,
                target_quat,
                seed_pool,
                orientation_seed_weight,
                seed_attempts,
                solver,
                preferred=seed_indices,
            )
        elif orientation_source == "position-only":
            if n_orientations != 1:
                raise ValueError("position-only orientation source requires --n-orientations 1")
            refs = _orientation_refs(1)
            target_pos, target_quat, voxel_index, orient_index = _grid_targets_from_seed_pool(
                seed_pool, cache, n_grid, refs, pad, limit, target_start, grid_bounds_source,
                target_order, grid_spacing, gate_voxel_indices
            )
            seed_candidate_indices = _seed_candidates_for_solver(
                target_pos, target_quat, seed_pool, orientation_seed_weight, seed_attempts, solver,
            )
        elif orientation_source == "fixed":
            refs = _orientation_refs(n_orientations)
            target_pos, target_quat, voxel_index, orient_index = _grid_targets_from_seed_pool(
                seed_pool, cache, n_grid, refs, pad, limit, target_start, grid_bounds_source,
                target_order, grid_spacing, gate_voxel_indices
            )
            seed_candidate_indices = _seed_candidates_for_solver(
                target_pos, target_quat, seed_pool, orientation_seed_weight, seed_attempts, solver,
            )
        elif orientation_source == "so3":
            refs = _orientation_refs_so3(n_orientations, pair_x_reflection=(spec.name == "humanoid_v21"))
            target_pos, target_quat, voxel_index, orient_index = _grid_targets_from_seed_pool(
                seed_pool, cache, n_grid, refs, pad, limit, target_start, grid_bounds_source,
                target_order, grid_spacing, gate_voxel_indices
            )
            seed_candidate_indices = _seed_candidates_for_solver(
                target_pos, target_quat, seed_pool, orientation_seed_weight, seed_attempts, solver,
            )
        elif orientation_source == "seed-pool":
            refs = _orientation_refs_from_seed_pool(seed_pool, n_orientations)
            target_pos, target_quat, voxel_index, orient_index = _grid_targets_from_seed_pool(
                seed_pool, cache, n_grid, refs, pad, limit, target_start, grid_bounds_source,
                target_order, grid_spacing, gate_voxel_indices
            )
            seed_candidate_indices = _seed_candidates_for_solver(
                target_pos, target_quat, seed_pool, orientation_seed_weight, seed_attempts, solver,
            )
        elif orientation_source == "nearest-seed":
            if n_orientations != 1:
                raise ValueError("nearest-seed orientation source requires --n-orientations 1")
            refs = _orientation_refs(1)
            target_pos, target_quat, voxel_index, orient_index = _grid_targets_from_seed_pool(
                seed_pool, cache, n_grid, refs, pad, limit, target_start, grid_bounds_source,
                target_order, grid_spacing, gate_voxel_indices
            )
            seed_indices = _nearest_seed_indices(
                target_pos, target_quat, seed_pool, orientation_seed_weight
            )
            target_quat = seed_pool["quat"][seed_indices].clone()
            seed_candidate_indices = _seed_candidates_for_solver(
                target_pos,
                target_quat,
                seed_pool,
                orientation_seed_weight,
                seed_attempts,
                solver,
                preferred=seed_indices,
            )
            orient_index.zero_()
        else:
            raise ValueError(f"unknown orientation source: {orientation_source}")
        extra = {
            "limit": limit,
            "stride": stride,
            "n_grid": n_grid,
            "n_grid_per_axis": torch.tensor(n_grid_per_axis, dtype=torch.long),
            "grid_origin": grid_origin,
            "grid_spacing": float(grid_spacing),
            "n_orientations": n_orientations,
            "orientation_source": orientation_source,
            "seed_attempts": seed_attempts,
            "target_alpha": target_alpha,
            "target_start": target_start,
            "grid_bounds_source": grid_bounds_source,
            "target_order": target_order,
            "grid_lo": grid_lo,
            "grid_hi": grid_hi,
            "pad": pad,
            "orientation_seed_weight": orientation_seed_weight,
            "voxel_index": voxel_index,
            "orient_index": orient_index,
        }
        extra.update(gate_extra)
        if orientation_source == "local-seed":
            extra["voxel_center_pos"] = voxel_center_pos
            extra["seed_target_distance"] = seed_target_distance
        source = "seeded_curobo_direct_ik_from_target_grid"
    else:
        raise ValueError(f"unknown target source: {target_source}")

    if target_pos.numel() == 0:
        raise ValueError("no targets selected")
    if solver == "batched":
        _solve_targets_batched(
            model, data, planner, active_dof_idx, seed_pool, seed_candidate_indices, target_pos,
            target_quat, side, source, out, extra, log_every, batch_size, num_seeds, spec,
            position_only_active=(orientation_source == "position-only"),
        )
    else:
        if orientation_source == "position-only":
            raise ValueError("position-only orientation source requires --solver batched")
        _solve_targets(
            model, data, planner, active_dof_idx, seed_pool, seed_candidate_indices, target_pos,
            target_quat, side, source, out, extra, log_every, spec,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--target-source", choices=["cache", "grid"], default="cache")
    parser.add_argument("--n-grid", type=int, default=3)
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=0.0,
        help="grid mode: isotropic metric voxel edge (m); >0 overrides --n-grid with cubic "
        "spacing (requires --target-order grid). 0 = legacy fixed-count.",
    )
    parser.add_argument("--n-orientations", type=int, default=4)
    parser.add_argument(
        "--orientation-source",
        choices=["seed-pool", "fixed", "so3", "nearest-seed", "local-seed", "position-only"],
        default="seed-pool",
    )
    parser.add_argument("--seed-pool-size", type=int, default=512)
    parser.add_argument("--pad", type=float, default=0.1)
    parser.add_argument("--orientation-seed-weight", type=float, default=0.25)
    parser.add_argument("--seed-attempts", type=int, default=1)
    parser.add_argument("--target-start", type=int, default=0)
    parser.add_argument(
        "--grid-bounds-source",
        choices=["legacy-symmetric", "seed-pool"],
        default="legacy-symmetric",
        help="grid mode: legacy-symmetric uses combined L/R cache bounds; seed-pool is side-specific debug",
    )
    parser.add_argument(
        "--target-order",
        choices=["legacy-symmetric-nearest", "legacy-nearest", "center-out", "grid", "nearest-seed"],
        default=None,
        help="grid mode: defaults to 'grid' with --grid-spacing, else 'legacy-symmetric-nearest' "
             "(which keeps chunk prefixes mirrored); nearest-seed is side-specific debug",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--target-alpha",
        type=float,
        default=1.0,
        help="local-seed only: 0 keeps seed EE position, 1 moves to voxel center",
    )
    parser.add_argument(
        "--solver",
        choices=["batched", "sequential"],
        default="batched",
        help="batched: many targets per IK call (default, ~7-30x faster); "
        "sequential: one target per call with 512 seeds (faithful, slow)",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="batched solver: targets per GPU call")
    parser.add_argument("--num-seeds", type=int, default=32, help="batched solver: IK restarts per target")
    parser.add_argument("--side", choices=["L", "R", "l", "r"], default="R")
    parser.add_argument(
        "--robot",
        choices=["humanoid_v21", "g1", "unitree_g1", "berkeley_humanoid_lite", "berkeley_lite", "toddlerbot", "toddlerbot_2xm_gripper", "booster_t1", "t1", "fourier_gr3", "gr3", "pal_talos", "talos", "apptronik_apollo", "apollo"],
        default="humanoid_v21",
        help="which arm to map: humanoid_v21 (default), g1 (== unitree_g1), "
        "berkeley_humanoid_lite (== berkeley_lite), "
        "toddlerbot (== toddlerbot_2xm_gripper), or "
        "booster_t1 (== t1), or apptronik_apollo (== apollo). Selects model, seed cache, cuRobo cfg, "
        "and endpoint/tool frames via get_robot_spec.",
    )
    parser.add_argument(
        "--rig",
        choices=sorted(_HUMANOID_RIG_MODES),
        default="dual",
        help="humanoid_v21 head-camera layout: dual (default, shipped K=2), none (camera-free "
        "upper bound), single (K=1), triple (K=3). The modules carry collision spheres, so each "
        "rig has its own reachable set -- payloads must not be shared across rigs.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--position-gate",
        type=pathlib.Path,
        default=None,
        help="grid-only position gate sidecar; solve all orientations only at its preserved voxel indices",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=_REPO_ROOT / "mj_envs" / "asset_zoo" / "cache" / "workspace_curobo_ik_humanoid_v21_smoke.pt",
    )
    args = parser.parse_args()
    if args.target_order is None:
        # A spacing-based grid is generally non-cubic, and every legacy ordering assumes a cubic
        # n_grid count -- `_ordered_grid_centers` rejects the combination outright. "grid" is the
        # only compatible ordering, so requiring the user to also type it made every documented
        # --grid-spacing command fail on a constraint they cannot resolve any other way.
        args.target_order = "grid" if args.grid_spacing > 0.0 else "legacy-symmetric-nearest"
    run(
        args.limit, args.stride, args.side.upper(), args.device, args.out,
        args.target_source, args.n_grid, args.n_orientations, args.seed_pool_size,
        args.pad, args.orientation_seed_weight, args.orientation_source, args.seed_attempts,
        args.target_alpha, args.target_start, args.grid_bounds_source, args.target_order, args.log_every,
        args.solver, args.batch_size, args.num_seeds, args.grid_spacing, args.robot, args.position_gate,
        args.rig,
    )


if __name__ == "__main__":
    main()
