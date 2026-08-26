"""Shared GPU reachability visibility scorer for dynamic-head robots.

Every robot uses one protocol: hold the documented canonical arms-down pose, solve a legal
camera-group head pose per reachable voxel, then test ordered eyes by FOV/range and visual-mesh
raycast.  Later eyes receive only targets still unseen.  Source-MJCF MuJoCo-Warp FK moves the
head meshes per target, unlike the legacy CPU sidecars.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "mj_envs")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The study's own camera envelope (90 x 65 deg, near/far), shared with the closed-form scorer in
# `plot_workspace_curobo`. Only the rigs whose eyes are bare sites read it -- see `_Adapter.fov_half`.
from tasks.camera_terms import FAR as _FAR_M, H_HALF as _H_HALF, V_HALF as _V_HALF  # noqa: E402

_NEAR_M = 0.10
_RAY_EPS_M = 1e-3
# Fixed-baseline head tilt, 47.6 deg down, matching G1's mount pitch. Duplicated from
# `plot_workspace_curobo._BASELINE_TILT_RAD` rather than imported because that module pulls
# matplotlib into a scorer that otherwise needs none; `test/pairwise_gate.py` asserts they agree.
_BASELINE_TILT_RAD = 0.8307767239493009
# Targets per GPU pass. Both hot kernels are launch-bound at 2,048 and near-asymptotic here:
# measured per-target aim cost 38 -> 19 us and per-ray cost 62 -> 13 us going from 2,048 to 16,384.
#
# NOT fully chunk-invariant, and the exception is worth knowing. `visible` is: rows never interact
# (`seen`/`unresolved` below are per row), and re-scoring ToddlerBot and Apollo at 2,048 vs 16,384
# gives bitwise-identical masks. `head_q` is NOT, because `_aim`'s convergence break tests the
# chunk-wide MAX residual, so a row that converged early keeps taking sub-tolerance steps until its
# slowest neighbour catches up, and which rows share a chunk depends on this constant. Measured
# spread is 7.3e-06 rad on 5% of ToddlerBot rows, i.e. below the loop's own 1e-5 tolerance, and it
# moves no visibility bit. Do not "fix" it by making the break per row: that would change the
# published head states for real.
_CHUNK = 16_384
# Largest rotation `_aim_group` asks for in one DLS step. See that function for why a cap is needed
# at all; pi/2 specifically because it is the angle at which the raw chord residual's useful
# (forward-perpendicular) component peaks, so capping there costs nothing in convergence rate while
# removing the degeneracy beyond it.
_AIM_STEP_CAP_RAD = float(np.pi / 2)
# Aim error above which `_aim_group` treats a row as stuck and restarts it from the joint lattice.
# Four orders of magnitude above the DLS loop's own 1e-5 break, so a merely slow row is never
# restarted -- the real failures sit past 5 deg (0.087 rad), three orders above this.
_AIM_STUCK_RAD = 1e-3
# Spacing of the restart lattice, in degrees of JOINT travel. A fixed node COUNT (what
# `_lattice_rescore` uses, 5) is the wrong knob: on a +-270 deg yaw it leaves 135 deg between
# nodes, and a near-vertical target then finds no node off the gimbal pole to restart from, which
# stranded the last 0.1% of aims. Spacing the nodes uniformly in angle instead makes the seed's
# resolution independent of how wide the joint happens to be. The cap bounds the FK to a few
# thousand worlds on the widest neck, and it is paid only on chunks that have a stuck row.
_LATTICE_SPACING_DEG = 10.0
_LATTICE_MAX_NODES = 61


@dataclass(frozen=True)
class _Adapter:
    robot: str
    # (MJCF camera name, site name) per physical eye. The camera is None on rigs whose eyes are
    # bare sites -- see `fov_half`.
    eyes: tuple[tuple[str | None, str], ...]
    aim_site: str
    aim_axis: int
    aim_sign: float
    head_joints: tuple[str, ...]
    visual_group: int
    dls_steps: int = 40
    # Which site-frame axes carry the image's horizontal and vertical. T1 and GR-3 mount their
    # cameras rolled 90 deg, so their site X runs up the image and Y across it -- hence the
    # (1, 0) default. A conventionally mounted camera (site X right, Y up, e.g. TALOS) is
    # (0, 1). Getting this backwards silently swaps the H and V field of view, which on a
    # non-square sensor is a real error and not a cosmetic one.
    image_axes: tuple[int, int] = (1, 0)
    # Independently aimable groups, ordered: (aim site, joints, indices into `eyes` this group
    # carries). None means the one-group default every external neck robot uses, derived from
    # `aim_site` / `head_joints` -- an empty tuple means a FIXED head that cannot aim at all.
    # This is the field that distinguishes "two cameras on one neck" from "two independent
    # gimbals", which single-target coverage is structurally blind to and pairwise coverage is not.
    aim_groups: tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...] | None = None
    # (horizontal, vertical) FOV half-angles in radians. Set on rigs whose eyes carry no MJCF
    # camera, so `cam_fovy`/`cam_resolution`/`cam_intrinsic` cannot supply the envelope: our own
    # modules and G1's head declare only an rgb SITE, and the study's 90 x 65 deg envelope lives
    # in `tasks.camera_terms`. When set, `_fov` takes its site-frame cone branch and `far_m` caps
    # the range (the intrinsics branches have no far clip because their targets never reach one).
    fov_half: tuple[float, float] | None = None
    far_m: float | None = None
    # Joint values written into the canonical pose before any aiming: the fixed-baseline rigs'
    # downward head tilt, and G1's bent elbows that clear its own forearms out of the head cone.
    pose_overrides: tuple[tuple[str, float], ...] = ()
    # `get_spec(head_camera=...)` mode for our own rigs, which must compile their own model --
    # see `_load_rig`. None routes through `get_robot_spec` like every external platform.
    head_camera: str | None = None


def _aim_groups(adapter: _Adapter) -> tuple[tuple[str, tuple[str, ...], tuple[int, ...]], ...]:
    """Resolve `aim_groups`, defaulting to the single group implied by `aim_site`/`head_joints`."""
    if adapter.aim_groups is not None:
        return adapter.aim_groups
    return ((adapter.aim_site, adapter.head_joints, tuple(range(len(adapter.eyes)))),)


_ADAPTERS = {
    "toddlerbot": _Adapter(
        "toddlerbot", (("head_cam_left", "head_cam_left_site"), ("head_cam_right", "head_cam_right_site")),
        "head_cam_left_site", 2, 1.0, ("neck_yaw_driven", "neck_pitch"), 2,
    ),
    "booster_t1": _Adapter(
        "booster_t1", (("head_cam", "head_cam_site"),), "head_cam_site", 2, -1.0,
        ("AAHead_yaw", "Head_pitch"), 2,
    ),
    # GR-3: one head camera on a 2-DOF neck. Its site carries the camera's own quat, so the
    # view axis is the site's -Z (axis 2, sign -1) exactly as on Booster T1, and the generic
    # isotropic-pinhole branch of `_fov` reproduces the vendor's 128x80 deg envelope from
    # `fovy` + resolution. No drive/driven coupling, so `_apply_coupling` stays a no-op.
    "fourier_gr3": _Adapter(
        "fourier_gr3", (("head_cam", "head_cam_site"),), "head_cam_site", 2, -1.0,
        ("head_yaw_joint", "head_pitch_joint"), 2,
    ),
    "apptronik_apollo": _Adapter(
        "apptronik_apollo", (("head_cam_left", "head_cam_left_site"), ("head_cam_right", "head_cam_right_site")),
        "head_cam_left_site", 0, 1.0, ("neck_yaw", "neck_roll", "neck_pitch"), 1,
    ),
    # TALOS: one Orbbec Astra Pro on a 2-DOF neck, listed OUTER-FIRST (`head_1_joint` is the
    # tilt and the parent of `head_2_joint`, the pan) -- the reverse of GR-3's yaw-outer head.
    # Its site carries the camera's own quat, so the view axis is the site's -Z exactly as on
    # T1 and GR-3, and the generic isotropic-pinhole branch of `_fov` reproduces the vendor's
    # 63.1 x 49.4 deg RGB envelope from `fovy` + resolution. No coupling, so `_apply_coupling`
    # stays a no-op.
    "pal_talos": _Adapter(
        "pal_talos", (("head_cam", "head_cam_site"),), "head_cam_site", 2, -1.0,
        ("head_1_joint", "head_2_joint"), 2, image_axes=(0, 1),
    ),
    # --- our own rigs and G1 -------------------------------------------------------------------
    # These differ from the five external platforms in three ways, all handled by adapter fields
    # rather than by branches: their eyes are bare SITES with the optical axis on site +Z and the
    # envelope supplied by `tasks.camera_terms` (`fov_half`/`far_m`, `aim_axis`/`aim_sign` = 2/+1
    # against the externals' -Z); each camera module is its OWN aim group; and the visibility model
    # is not the model their IK ran against (`head_camera`, see `_load_rig`).
    "humanoid_v21": _Adapter(
        "humanoid_v21", ((None, "cam_left_rgb"), (None, "cam_right_rgb")),
        "cam_left_rgb", 2, 1.0,
        ("cam_yaw_left", "cam_pitch_left", "cam_yaw_right", "cam_pitch_right"), 2,
        image_axes=(0, 1),
        aim_groups=(
            ("cam_left_rgb", ("cam_yaw_left", "cam_pitch_left"), (0,)),
            ("cam_right_rgb", ("cam_yaw_right", "cam_pitch_right"), (1,)),
        ),
        fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M), head_camera="actuated",
    ),
    "humanoid_v21_single": _Adapter(
        "humanoid_v21_single", ((None, "cam_rgb"),), "cam_rgb", 2, 1.0,
        ("cam_yaw", "cam_pitch"), 2, image_axes=(0, 1),
        aim_groups=(("cam_rgb", ("cam_yaw", "cam_pitch"), (0,)),),
        fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M), head_camera="actuated_single",
    ),
    "humanoid_v21_triple": _Adapter(
        "humanoid_v21_triple",
        ((None, "cam_left_rgb"), (None, "cam_center_rgb"), (None, "cam_right_rgb")),
        "cam_left_rgb", 2, 1.0,
        ("cam_yaw_left", "cam_pitch_left", "cam_yaw_center", "cam_pitch_center",
         "cam_yaw_right", "cam_pitch_right"), 2, image_axes=(0, 1),
        aim_groups=(
            ("cam_left_rgb", ("cam_yaw_left", "cam_pitch_left"), (0,)),
            ("cam_center_rgb", ("cam_yaw_center", "cam_pitch_center"), (1,)),
            ("cam_right_rgb", ("cam_yaw_right", "cam_pitch_right"), (2,)),
        ),
        fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M), head_camera="actuated_triple",
    ),
    # The un-actuated baseline of the SAME dual hardware: both modules frozen at the downward
    # baseline tilt, back-to-back, so a front/rear pair is still coverable by distinct eyes even
    # though nothing can aim. One pitch scalar tilts both eyes down, because the per-eye pitch-axis
    # sign flip cancels the per-eye optical-axis sign flip.
    "humanoid_v21_fixed": _Adapter(
        "humanoid_v21_fixed", ((None, "cam_left_rgb"), (None, "cam_right_rgb")),
        "cam_left_rgb", 2, 1.0, (), 2, image_axes=(0, 1), aim_groups=(),
        fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M), head_camera="actuated",
        pose_overrides=(("cam_pitch_left", _BASELINE_TILT_RAD), ("cam_pitch_right", _BASELINE_TILT_RAD)),
    ),
    # The K=1 / K=3 counterparts of `humanoid_v21_fixed`, built by the same rule: same eyes, no head
    # joints, no aim group, every pitch frozen at the baseline tilt and every yaw left at qpos0
    # (rest). They exist so the camera-count ablation's fixed band and actuated band come from ONE
    # kernel; without them only the K=2 rig had a fixed adapter, and scoring the K=1/K=3 fixed bands
    # by the closed-form path would split kernels *inside a single figure*.
    "humanoid_v21_single_fixed": _Adapter(
        "humanoid_v21_single_fixed", ((None, "cam_rgb"),), "cam_rgb", 2, 1.0, (), 2,
        image_axes=(0, 1), aim_groups=(),
        fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M), head_camera="actuated_single",
        pose_overrides=(("cam_pitch", _BASELINE_TILT_RAD),),
    ),
    "humanoid_v21_triple_fixed": _Adapter(
        "humanoid_v21_triple_fixed",
        ((None, "cam_left_rgb"), (None, "cam_center_rgb"), (None, "cam_right_rgb")),
        "cam_left_rgb", 2, 1.0, (), 2, image_axes=(0, 1), aim_groups=(),
        fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M), head_camera="actuated_triple",
        pose_overrides=(("cam_pitch_left", _BASELINE_TILT_RAD), ("cam_pitch_center", _BASELINE_TILT_RAD),
                        ("cam_pitch_right", _BASELINE_TILT_RAD)),
    ),
    # G1's real platform bolts its head camera to `torso_link` pitched 47.6 deg down, so there is
    # no aim group at all and its single cone is the whole story. Its `head_camera` MJCF camera is
    # declared at 1x1 resolution, which the intrinsics branch cannot use, and the study
    # deliberately lends it OUR 90 x 65 envelope (flattering, and recorded as such) -- both reasons
    # to take the site-cone branch. The elbow/wrist overrides swing its forearms clear of the cone.
    "unitree_g1": _Adapter(
        "unitree_g1", ((None, "head_camera_rgb"),), "head_camera_rgb", 2, 1.0, (), 2,
        image_axes=(0, 1), aim_groups=(), fov_half=(_H_HALF, _V_HALF), far_m=float(_FAR_M),
        pose_overrides=(
            ("left_elbow_joint", np.pi / 2), ("right_elbow_joint", np.pi / 2),
            ("left_wrist_roll_joint", np.pi / 2), ("right_wrist_roll_joint", np.pi / 2),
        ),
    ),
}


def _canonical_qpos(model: mujoco.MjModel, spec, robot: str, root_pos: np.ndarray) -> np.ndarray:
    """Return source qpos for the approved visibility pose, including root and coupling.

    `spec` is None for our own rigs, which compile their own model (see `_load_rig`) and have no
    source coupling to apply.
    """
    from mj_envs.utils.mj_home_pose import home_qpos

    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos(model)
    data.qpos[:3] = np.asarray(root_pos, dtype=np.float64)
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    if robot == "booster_t1":
        for name, value in {
            "Left_Shoulder_Roll": -np.pi / 2,
            "Right_Shoulder_Roll": np.pi / 2,
            "Left_Elbow_Yaw": 0.0,
            "Right_Elbow_Yaw": 0.0,
        }.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    for name, value in _ADAPTERS[robot].pose_overrides:
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    if spec is not None:
        spec.apply_source_coupling(model, data)
    return data.qpos.copy()


def _load_rig(robot: str) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
    """Return the visibility model, its base-frame origin in world, and the canonical qpos.

    Two sources, and the split changes results rather than being cosmetic. Every external platform
    resolves through `get_robot_spec`, whose `model_loader` IS the model its IK solved against.
    Our own rigs cannot: `_HUMANOID_RIG_MODES["dual"]` loads the WELDED camera variant, in which
    the gimbal is rigid for collision purposes and `cam_yaw_*`/`cam_pitch_*` DO NOT EXIST as
    joints, so nothing can be aimed against it. `_Adapter.head_camera` names the actuated variant
    to compile instead; the two variants share arms and torso, so the occluding body is the same
    and only the head modules differ.

    The root translation is bookkeeping, not geometry: `score_targets` shifts its targets by the
    same vector, so only the target-relative-to-body pose matters and that is invariant to it. It
    is kept anyway so head_q and any saved sidecar stay in the same world frame as before.
    """
    adapter = _ADAPTERS[robot]
    if adapter.head_camera is not None:
        from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import HOME_KEYFRAME, get_spec

        model = get_spec(head_camera=adapter.head_camera, end_effector="welded",
                         hand="parallel_gripper").compile()
        root = np.asarray(HOME_KEYFRAME.pos, dtype=np.float64)
        return model, root, _canonical_qpos(model, None, robot, root)
    from mj_envs.asset_zoo.reachability_study.generate_workspace_curobo import get_robot_spec

    spec = get_robot_spec(robot)
    model = spec.model_loader()
    root = np.asarray(spec.home_root_pos, dtype=np.float64)
    return model, root, _canonical_qpos(model, spec, robot, root)


class _WarpBatch:
    """One target chunk with dynamic source-MJCF FK and compact candidate-ray batches."""

    def __init__(self, model: mujoco.MjModel, qpos: torch.Tensor, wm=None):
        import mujoco_warp as mjw
        import warp as wp

        self.model, self.qpos, self.n = model, qpos, qpos.shape[0]
        self.mjw, self.wp = mjw, wp
        self.wm = mjw.put_model(model) if wm is None else wm
        self.wd = mjw.make_data(model, nworld=self.n)
        self.forward(qpos)

    def forward(self, qpos: torch.Tensor, camlight: bool = True) -> None:
        """Re-run FK for `qpos`. `camlight` is optional because `_aim`'s DLS loop reads only sites,
        so the camera/light pass is a kernel launch per step for a result nothing consumes."""
        self.qpos = qpos
        self.wp.copy(self.wd.qpos, self.wp.from_torch(qpos.contiguous()))
        self.mjw.kinematics(self.wm, self.wd)
        if camlight:
            self.mjw.camlight(self.wm, self.wd)

    def site(self, site_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wp.to_torch(self.wd.site_xpos)[:, site_id], self.wp.to_torch(self.wd.site_xmat)[:, site_id]

    def camera(self, camera_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wp.to_torch(self.wd.cam_xpos)[:, camera_id], self.wp.to_torch(self.wd.cam_xmat)[:, camera_id]

    def ray_clear(self, origins: torch.Tensor, targets: torch.Tensor, eye_bodies: tuple[int, ...], group: int) -> torch.Tensor:
        """Return visual-mesh LOS for this compact, already-FOV-filtered target batch.

        `eye_bodies` lists every body in the camera's own head/neck mount chain (each
        `head_joints` body plus the site's own carrying body) -- with only a head camera, the
        robot's own head assembly can never be a real obstacle to its own lens, so self-hits on
        any of them are not occlusion. `mujoco_warp.rays` only accepts ONE excluded body per
        call, so a multi-link neck (TALOS: head_1_link + head_2_link) is walked past iteratively:
        each pass excludes one chain body; a self-hit on any OTHER chain body advances the ray
        origin just past that hit and retries, up to `len(eye_bodies)` passes (deep enough for
        any neck this study models). A hit that is not on the chain is a genuine block.
        """
        from mujoco_warp._src.types import vec6

        device = origins.device
        delta = targets - origins
        total_dist = delta.norm(dim=1).clamp_min(1e-9)
        direction = delta / total_dist[:, None]
        cur_origin = origins.clone()
        remaining = total_dist.clone()
        clear = torch.zeros(self.n, dtype=torch.bool, device=device)
        first_hit_geom = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        active = torch.ones(self.n, dtype=torch.bool, device=device)
        groups = [0] * 6; groups[group] = 1
        weld_of_body = torch.as_tensor(self.model.body_weldid, dtype=torch.int64, device=device)
        weld_of_geom = weld_of_body[torch.as_tensor(self.model.geom_bodyid, dtype=torch.int64, device=device)]
        self_body_set = torch.as_tensor(eye_bodies, dtype=torch.int64, device=device)  # already weld ids
        ngeom = weld_of_geom.shape[0]

        for exclude_id in eye_bodies:
            if not active.any():
                break
            # `rays()` requires pnt.shape[0] == d.nworld (== self.n, fixed at construction) so
            # that worldid lines up with self.wd's per-world geom_xpos/geom_xmat. A compacted,
            # active-only array here breaks that alignment -- besides reading/writing past the
            # end of the compacted buffers once the active set shrinks below self.n (a real
            # illegal-memory-access, not a mujoco_warp bug), it silently pairs each ray with the
            # WRONG world's FK pose whenever `idx` isn't a plain 0..n_idx-1 prefix. So every pass
            # ray-tests all self.n worlds and masks the result by `active` instead of compacting.
            pnt = self.wp.from_torch(cur_origin.contiguous().view(self.n, 1, 3), dtype=self.wp.vec3f)
            vec = self.wp.from_torch(direction.contiguous().view(self.n, 1, 3), dtype=self.wp.vec3f)
            excluded = self.wp.zeros(1, dtype=self.wp.int32, device="cuda:0")
            excluded.fill_(int(exclude_id))
            dist = self.wp.empty((self.n, 1), dtype=self.wp.float32, device="cuda:0")
            geomid = self.wp.empty((self.n, 1), dtype=self.wp.int32, device="cuda:0")
            normal = self.wp.empty((self.n, 1), dtype=self.wp.vec3, device="cuda:0")
            self.mjw.rays(self.wm, self.wd, pnt, vec, vec6(*groups), True, excluded, dist, geomid, normal)
            d = self.wp.to_torch(dist).view(-1)
            g = self.wp.to_torch(geomid).view(-1).to(torch.int64)
            # mujoco_warp's ray kernel pads its geom loop up to a tile-size multiple; on a
            # genuine miss every candidate (real geoms with no hit, plus the padding indices)
            # ties at MJ_MAXVAL, and the tile-argmin can report one of those padding indices
            # as `geomid_out` even though `dist_out` correctly reports -1. So `g` alone is not
            # bounds-safe -- it can be >= ngeom on a miss. `dist_out` is always correct, so gate
            # the geom lookup on both `g >= 0` and `g < ngeom` and treat anything else as a miss.
            valid_hit = (g >= 0) & (g < ngeom)
            hit_weld = torch.where(valid_hit, weld_of_geom[g.clamp(0, ngeom - 1)], torch.full_like(g, -1))
            is_clear = active & ((d < 0) | (d >= remaining - _RAY_EPS_M))
            is_self_hit = active & ~is_clear & valid_hit & torch.isin(hit_weld, self_body_set)
            # Capture the first non-self hit's geom id for the LOS post-filter path.
            new_block = active & ~is_clear & valid_hit & ~is_self_hit
            first_hit_geom = torch.where(new_block & (first_hit_geom < 0), g, first_hit_geom)

            clear = clear | is_clear
            active = is_self_hit  # only self-hit rays continue to the next pass

            adv_d = d[is_self_hit] + _RAY_EPS_M
            cur_origin[is_self_hit] = cur_origin[is_self_hit] + direction[is_self_hit] * adv_d[:, None]
            remaining[is_self_hit] = remaining[is_self_hit] - adv_d
        return clear

    def ray_clear_with_hit(self, origins: torch.Tensor, targets: torch.Tensor,
                           eye_bodies: tuple[int, ...], group: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Same as ``ray_clear`` but also returns the per-ray FIRST non-self-hit geom id.

        ``hit_geom`` is -1 for clear rays (no nearest blocker) and the geom index of the
        nearest non-``eye_bodies`` blocker otherwise. The harness LOS path uses the geom id
        to post-filter against the CPU's ``_scene_occluder_geoms`` set (the GPU's broader
        group mask includes visual meshes the CPU prefix-filter excludes -- see plan note)."""
        # Reuse ray_clear with the LOS group mask (groups 2, 3, 5 are visual + collision + planner
        # spheres; the CPU's `pp_*` prefix covers all three). Bitwise OR of the three masks.
        from mujoco_warp._src.types import vec6
        # Inline a copy of ray_clear with the LOS mask hardcoded; this avoids allocating a
        # second group list per call. Mirrors the structure of ray_clear with LOS groups.
        device = origins.device
        delta = targets - origins
        total_dist = delta.norm(dim=1).clamp_min(1e-9)
        direction = delta / total_dist[:, None]
        cur_origin = origins.clone()
        remaining = total_dist.clone()
        clear = torch.zeros(self.n, dtype=torch.bool, device=device)
        first_hit_geom = torch.full((self.n,), -1, dtype=torch.int64, device=device)
        active = torch.ones(self.n, dtype=torch.bool, device=device)
        groups = [0, 0, 1, 1, 0, 1]   # visual(2) + collision(3) + planner-sphere(5)
        weld_of_body = torch.as_tensor(self.model.body_weldid, dtype=torch.int64, device=device)
        weld_of_geom = weld_of_body[torch.as_tensor(self.model.geom_bodyid, dtype=torch.int64, device=device)]
        self_body_set = torch.as_tensor(eye_bodies, dtype=torch.int64, device=device)
        ngeom = weld_of_geom.shape[0]
        for exclude_id in eye_bodies:
            if not active.any():
                break
            pnt = self.wp.from_torch(cur_origin.contiguous().view(self.n, 1, 3), dtype=self.wp.vec3f)
            vec = self.wp.from_torch(direction.contiguous().view(self.n, 1, 3), dtype=self.wp.vec3f)
            excluded = self.wp.zeros(1, dtype=self.wp.int32, device="cuda:0")
            excluded.fill_(int(exclude_id))
            dist = self.wp.empty((self.n, 1), dtype=self.wp.float32, device="cuda:0")
            geomid = self.wp.empty((self.n, 1), dtype=self.wp.int32, device="cuda:0")
            normal = self.wp.empty((self.n, 1), dtype=self.wp.vec3, device="cuda:0")
            self.mjw.rays(self.wm, self.wd, pnt, vec, vec6(*groups), True, excluded, dist, geomid, normal)
            d = self.wp.to_torch(dist).view(-1)
            g = self.wp.to_torch(geomid).view(-1).to(torch.int64)
            valid_hit = (g >= 0) & (g < ngeom)
            hit_weld = torch.where(valid_hit, weld_of_geom[g.clamp(0, ngeom - 1)], torch.full_like(g, -1))
            is_clear = active & ((d < 0) | (d >= remaining - _RAY_EPS_M))
            is_self_hit = active & ~is_clear & valid_hit & torch.isin(hit_weld, self_body_set)
            new_block = active & ~is_clear & valid_hit & ~is_self_hit
            first_hit_geom = torch.where(new_block & (first_hit_geom < 0), g, first_hit_geom)
            clear = clear | is_clear
            active = is_self_hit
            adv_d = d[is_self_hit] + _RAY_EPS_M
            cur_origin[is_self_hit] = cur_origin[is_self_hit] + direction[is_self_hit] * adv_d[:, None]
            remaining[is_self_hit] = remaining[is_self_hit] - adv_d
        return clear, first_hit_geom


def _apply_coupling(robot: str, qpos: torch.Tensor, model: mujoco.MjModel) -> None:
    if robot != "toddlerbot":
        return
    driven = model.jnt_qposadr[model.joint("neck_yaw_driven").id]
    drive = model.jnt_qposadr[model.joint("neck_yaw_drive").id]
    qpos[:, drive] = -qpos[:, driven] / 0.9090909091


def _aim(adapter: _Adapter, model: mujoco.MjModel, qpos: torch.Tensor, targets: torch.Tensor, wm=None) -> tuple[torch.Tensor, _WarpBatch]:
    """Batched source-FK DLS aim, one head state per target, every aim group pointed at that target.

    Returns the solved qpos and the `_WarpBatch` forwarded at it. A rig with no aim group (a bolted
    head) returns its canonical pose untouched, which is the whole of its coverage.

    `wm` accepts a `put_model` handle from an earlier chunk; it is pure reuse, and passing None only
    costs another upload.

    PERFORMANCE, and why the probe batch sits outside the loop. The finite-difference probe used to
    be CONSTRUCTED per head joint per step, so a 2-DOF neck at `dls_steps=40` called
    `mujoco_warp.make_data` 80 times per chunk. At ~6 us per allocated world that was 99% of this
    function's runtime, against 0.11 ms for the FK it existed to run. Allocating it once and
    re-forwarding is bitwise-neutral, since `forward` is a pure function of qpos, and measured
    15x/25x/29x faster at n = 2k/16k/64k with `torch.equal` on the solved qpos at every size.
    `camlight` is skipped inside the loop for the same reason -- only `batch.site` is read here --
    and one full forward after the loop restores the camera poses `_fov` reads on Apollo.
    """
    batch = _WarpBatch(model, qpos, wm)
    groups = _aim_groups(adapter)
    if not groups:  # fixed head: nothing to solve, and the constructor already forwarded camlight
        return qpos, batch
    probe = _WarpBatch(model, qpos.clone(), batch.wm)
    for site_name, joint_names, _eye_idx in groups:
        _aim_group(adapter, model, qpos, targets, site_name, joint_names, batch, probe)
    batch.forward(qpos)
    return qpos, batch


def _capped_residual(forward: torch.Tensor, desired: torch.Tensor, chord: torch.Tensor) -> torch.Tensor:
    """Replace the chord residual `desired - forward` past a right angle, leaving it alone before it.

    WHY. `_aim_group` drives the chord through a Jacobian whose every column is `w x forward`, so
    the whole range space is orthogonal to `forward` and DLS silently discards the chord's
    forward-parallel part. What survives is the perpendicular part, of magnitude `sin(theta)` --
    which peaks at theta = 90 deg and then COLLAPSES back to zero as the target swings behind the
    lens. At theta = 180 deg exactly the chord is purely parallel to `forward`, the projection is
    identically zero, and the loop is a fixed point: it returns the canonical pose, 180 deg off
    target, having "converged" on nothing. Measured on `humanoid_v21_single` over a uniform sphere
    of directions: 2.8% of aims ended >5 deg off and 1.6% ended >45 deg off, all of them in the
    cone behind the head, on a gimbal whose +-270 deg yaw can physically reach every one of them.
    Downstream this reads as an occlusion, and it cost ~2.1 points of the K=1 coverage number.

    WHAT. Rotate `forward` toward `desired` about their common perpendicular by at most
    `_AIM_STEP_CAP_RAD` and solve for THAT intermediate direction. Past the cap the ask is a clean
    right-angle turn -- maximum perpendicular drive, no parallel component to be discarded -- and
    the aim walks around to the target over several steps instead of stalling.

    BITWISE. Rows at or below the cap are returned as the untouched `chord` object rather than a
    recomputed Rodrigues value, so every aim that never swings past a right angle -- which is every
    aim on a rig whose neck limits cannot reach that far anyway -- produces the identical head pose
    it did before this function existed. `test/verify_ours_g1_port.py` covers this.

    Exactly-antipodal rows have no defined `forward x desired` axis. Any perpendicular is a legal
    rotation axis there, so take the one built from `forward`'s smallest-magnitude component, whose
    cross product is bounded away from degenerate (norm >= sqrt(2/3)).
    """
    cos = (forward * desired).sum(1).clamp(-1.0, 1.0)
    capped = cos < float(np.cos(_AIM_STEP_CAP_RAD))
    if not bool(capped.any()):
        return chord
    axis = torch.linalg.cross(forward, desired, dim=1)
    degenerate = axis.norm(dim=1) < 1e-6
    if bool(degenerate.any()):
        basis = torch.zeros_like(forward).scatter_(1, forward.abs().argmin(1, keepdim=True), 1.0)
        axis = torch.where(degenerate[:, None], torch.linalg.cross(forward, basis, dim=1), axis)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # Rodrigues about an axis perpendicular to `forward`, so the (1 - cos) term drops out.
    goal = forward * float(np.cos(_AIM_STEP_CAP_RAD)) + torch.linalg.cross(axis, forward, dim=1) * float(np.sin(_AIM_STEP_CAP_RAD))
    return torch.where(capped[:, None], goal - forward, chord)


def _aim_group(adapter: _Adapter, model: mujoco.MjModel, qpos: torch.Tensor, targets: torch.Tensor,
               site_name: str, joint_names: tuple[str, ...], batch: _WarpBatch, probe: _WarpBatch) -> None:
    """Drive one aim group's joints until its site's optical axis points at `targets`.

    Mutates `qpos` in place and leaves `batch` forwarded at the solved state (camlight skipped --
    `_aim` runs one full forward after every group). Groups are solved in sequence on the SHARED
    qpos, which is exact rather than an approximation: a group's joints move only its own site, so
    a later group reads an FK state its own solve does not depend on.

    A second, lattice-seeded pass rescues the rows the first one leaves stuck; see
    `_lattice_seed` for what gets stuck and why, and note that rows the FIRST pass solved are
    restored bit for bit afterwards, so the rescue can only add solutions, never perturb them.
    """
    device = qpos.device
    joint_qpos = [int(model.jnt_qposadr[model.joint(name).id]) for name in joint_names]
    limits = torch.as_tensor(np.asarray([model.jnt_range[model.joint(name).id] for name in joint_names]), device=device)
    site_id = model.site(site_name).id
    _dls_aim(adapter, model, qpos, targets, joint_qpos, limits, site_id, batch, probe)

    stuck = _aim_residual(adapter, batch, site_id, targets) > _AIM_STUCK_RAD
    if not bool(stuck.any()):
        return
    solved = qpos[:, joint_qpos].clone()
    qpos[:, joint_qpos] = _lattice_seed(adapter, model, qpos, targets, joint_qpos, limits, site_id, batch.wm)
    _apply_coupling(adapter.robot, qpos, model)
    batch.forward(qpos, camlight=False)
    _dls_aim(adapter, model, qpos, targets, joint_qpos, limits, site_id, batch, probe)
    # Keep whatever the first pass already achieved, so this pass is additive by construction.
    rescued = stuck & (_aim_residual(adapter, batch, site_id, targets) < _AIM_STUCK_RAD)
    qpos[:, joint_qpos] = torch.where(rescued[:, None], qpos[:, joint_qpos], solved)
    _apply_coupling(adapter.robot, qpos, model)
    batch.forward(qpos, camlight=False)


def _aim_residual(adapter: _Adapter, batch: _WarpBatch, site_id: int, targets: torch.Tensor) -> torch.Tensor:
    """Per-row angle, in radians, between an aim site's optical axis and its target direction."""
    origin, rotation = batch.site(site_id)
    forward = adapter.aim_sign * rotation[:, :, adapter.aim_axis]
    desired = targets - origin
    desired = desired / desired.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return torch.arccos((forward * desired).sum(1).clamp(-1.0, 1.0))


def _lattice_seed(adapter: _Adapter, model: mujoco.MjModel, qpos: torch.Tensor, targets: torch.Tensor,
                  joint_qpos: list[int], limits: torch.Tensor, site_id: int, wm) -> torch.Tensor:
    """Best-aligned head state on the joint-limit lattice, per target: a restart for stuck rows.

    WHY. Even with `_capped_residual` removing the antipodal fixed point, DLS takes the LEAST-NORM
    joint motion at every step, which is locally correct and globally a trap on a yaw/pitch gimbal.
    A target high and behind the lens is closer in joint space over the top than around the side,
    so pitch saturates at its limit -- and at pitch = +-90 deg the yaw axis has rotated onto the
    optical axis, where yaw only rolls the image and no longer steers it. The solve is then pinned
    in a corner it cannot leave. Measured on `humanoid_v21_single`, this stranded 83 of 4,096
    uniform directions with `cam_pitch` at exactly +-90 deg, every one of which a brute-force sweep
    of the same limits aims to within 0.41 deg median. Only rigs whose pitch range reaches the pole
    can hit this; no external platform's neck comes close (widest is ToddlerBot's -35..80 deg).

    WHAT. Score the same limit lattice `_lattice_rescore` already uses -- 5 fractions per joint,
    spanning each limit -- by FK'ing it once and taking, per target, the node whose optical axis is
    best aligned. Lens position moves with the joints, so alignment is measured from each node's own
    origin rather than a nominal one. Restarting DLS there crosses the pole instead of stalling
    against it. Non-group joints are read from row 0 because they are identical across rows by
    construction: only head joints vary per target, and other groups' joints do not move this site.
    """
    from itertools import product

    device = qpos.device
    axes = []
    for column in range(len(joint_qpos)):
        lo, hi = float(limits[column, 0]), float(limits[column, 1])
        count = min(_LATTICE_MAX_NODES, max(5, int(round(np.rad2deg(hi - lo) / _LATTICE_SPACING_DEG)) + 1))
        axes.append(torch.linspace(lo, hi, count, dtype=limits.dtype, device=device))
    nodes = torch.stack([torch.stack(choice) for choice in product(*[list(axis) for axis in axes])])
    lattice = qpos[0].repeat(nodes.shape[0], 1)
    lattice[:, joint_qpos] = nodes.to(lattice.dtype)
    _apply_coupling(adapter.robot, lattice, model)

    origin, rotation = _WarpBatch(model, lattice, wm).site(site_id)
    forward = adapter.aim_sign * rotation[:, :, adapter.aim_axis]
    desired = targets[:, None, :] - origin[None, :, :]
    desired = desired / desired.norm(dim=2, keepdim=True).clamp_min(1e-9)
    best = (desired * forward[None, :, :]).sum(2).argmax(dim=1)
    return lattice[best][:, joint_qpos]


def _dls_aim(adapter: _Adapter, model: mujoco.MjModel, qpos: torch.Tensor, targets: torch.Tensor,
             joint_qpos: list[int], limits: torch.Tensor, site_id: int,
             batch: _WarpBatch, probe: _WarpBatch) -> None:
    """One damped-least-squares aim sweep from wherever `qpos` currently stands."""
    device = qpos.device
    eps, damping = 1e-4, 1e-5
    eye = torch.eye(3, device=device).expand(qpos.shape[0], 3, 3)
    for _ in range(adapter.dls_steps):
        origin, rotation = batch.site(site_id)
        forward = adapter.aim_sign * rotation[:, :, adapter.aim_axis]
        desired = targets - origin
        desired = desired / desired.norm(dim=1, keepdim=True).clamp_min(1e-9)
        residual = desired - forward
        if float(residual.norm(dim=1).max()) < 1e-5:
            break
        residual = _capped_residual(forward, desired, residual)
        cols = []
        for qadr in joint_qpos:
            perturbed = qpos.clone(); perturbed[:, qadr] += eps
            _apply_coupling(adapter.robot, perturbed, model)
            probe.forward(perturbed, camlight=False)
            _p, rot_p = probe.site(site_id)
            relative = torch.bmm(rot_p, rotation.transpose(1, 2))
            # World-frame angular velocity: vee((R(q+eps)-R(q)) R(q)^T) / eps.
            # This matches MuJoCo's `mj_jacSite(..., jacr)` used by source scorers.
            cols.append(torch.stack((
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ), dim=1) / (2 * eps))
        angular_jac = torch.stack(cols, dim=2)
        # Camera-ray Jacobian: a joint angular velocity w moves forward direction by w x forward.
        # Solve this direct directional residual, not an angular residual.  The latter is singular
        # far from alignment and created discontinuous head states at adjacent voxels.
        jac = torch.linalg.cross(angular_jac.transpose(1, 2), forward[:, None, :], dim=2).transpose(1, 2)
        delta = torch.bmm(jac.transpose(1, 2), torch.linalg.solve(torch.bmm(jac, jac.transpose(1, 2)) + damping * eye, residual[:, :, None])).squeeze(2)
        delta.clamp_(-0.15, 0.15)
        for column, qadr in enumerate(joint_qpos):
            qpos[:, qadr] = qpos[:, qadr] + delta[:, column]
            qpos[:, qadr].clamp_(limits[column, 0], limits[column, 1])
        _apply_coupling(adapter.robot, qpos, model)
        batch.forward(qpos, camlight=False)


def _aim_bisector(adapter: _Adapter, model: mujoco.MjModel, qpos: torch.Tensor, a: torch.Tensor,
                  b: torch.Tensor, wm=None) -> tuple[torch.Tensor, _WarpBatch]:
    """Aim every group at the angular bisector of the two target DIRECTIONS from its own lens.

    The single best-effort state for holding two targets in one cone. Direction bisector rather than
    the midpoint of the two points, and the distinction is worth 13 points of pairwise coverage at
    K=1: with x_j drawn uniformly from the workspace box the two targets routinely sit at very
    different ranges, and the midpoint of the POINTS is dragged toward the far one, throwing the near
    one out of frame. The bisector of the DIRECTIONS minimizes the larger of the two angles, which is
    optimal for a circular cone -- this FOV is a rectangle and the gimbal has no roll, so vertical
    spread cannot be traded into the wider horizontal budget and the aim stays sound but incomplete.

    Bisector, lens position, and aim are mutually dependent, because the lens translates as the
    gimbal turns, so the bisector taken at the rest pose is not the bisector at the aimed pose. Two
    passes per group: aim at the bisector measured from where the lens currently is, then re-measure
    and re-aim from where it ended up. Same refine structure the closed-form scorer uses.

    Antipodal pairs have no bisector -- and cannot share any cone under 180 deg -- so they fall back
    to aiming at `a`, which costs nothing: field of view is re-tested at the solved pose regardless,
    so an unusable aim simply reports not-covered instead of producing a false witness.
    """
    batch = _WarpBatch(model, qpos, wm)
    groups = _aim_groups(adapter)
    if not groups:
        return qpos, batch
    probe = _WarpBatch(model, qpos.clone(), batch.wm)
    for site_name, joint_names, _eye_idx in groups:
        site_id = model.site(site_name).id
        for _refine in range(2):
            origin, _rotation = batch.site(site_id)
            to_a = a - origin
            to_a = to_a / to_a.norm(dim=1, keepdim=True).clamp_min(1e-9)
            to_b = b - origin
            to_b = to_b / to_b.norm(dim=1, keepdim=True).clamp_min(1e-9)
            bisector = to_a + to_b
            norm = bisector.norm(dim=1, keepdim=True)
            direction = torch.where(norm > 1e-3, bisector / norm.clamp_min(1e-9), to_a)
            _aim_group(adapter, model, qpos, origin + direction, site_name, joint_names, batch, probe)
    batch.forward(qpos)
    return qpos, batch


def _eye_bodies(adapter: _Adapter, model: mujoco.MjModel) -> tuple[int, ...]:
    """Every body in the camera's own head/neck mount chain: each eye site's own carrying body,
    then each `head_joints` body (covers a site on a further, joint-less welded child).

    Order matters for `ray_clear`'s pass budget, not just correctness: the site's own carrying
    body is listed FIRST because it is overwhelmingly the common self-hit (the lens sits close
    to its own housing mesh -- see `talos_study_import.py`'s Orbbec-housing note), so pass 1
    already excludes it and most rays resolve there, exactly like the old single-exclude
    behavior. The other neck-chain bodies (rare stragglers, e.g. a ray grazing the tilt-joint
    housing) only cost extra passes for the few rays that actually need them.
    """
    ids: list[int] = []
    for _camera_name, site_name in adapter.eyes:
        bid = int(model.body_weldid[model.site(site_name).bodyid.item()])
        if bid not in ids:
            ids.append(bid)
    for name in adapter.head_joints:
        bid = int(model.body_weldid[model.joint(name).bodyid.item()])
        if bid not in ids:
            ids.append(bid)
    return tuple(ids)


def _fov(adapter: _Adapter, model: mujoco.MjModel, batch: _WarpBatch, eye: tuple[str, str], targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return FOV/range mask, lens origins, and carrying body for one physical eye.

    `targets` is a free parameter read against whatever FK state `batch` carries, so "aim at A, ask
    about B" needs no restructuring -- which is what the pairwise kernel is built on.
    """
    camera_name, site_name = eye
    site_id = model.site(site_name).id
    site_pos, site_rot = batch.site(site_id)
    if adapter.fov_half is not None:
        # Eyes that are bare sites: rectangular cone from the shared envelope, optical axis on
        # `aim_axis`/`aim_sign` (site +Z for ours and G1, against the intrinsics branches' -Z), and
        # a far clip, which the intrinsics branches omit because their targets never reach one.
        local = torch.bmm(site_rot.transpose(1, 2), (targets - site_pos)[:, :, None]).squeeze(2)
        h_axis, v_axis = adapter.image_axes
        h_half, v_half = adapter.fov_half
        forward = adapter.aim_sign * local[:, adapter.aim_axis]
        distance = local.norm(dim=1)
        inside = ((forward > 0) & (distance >= _NEAR_M) & (distance <= adapter.far_m)
                  & (torch.atan2(local[:, h_axis], forward).abs() <= h_half)
                  & (torch.atan2(local[:, v_axis], forward).abs() <= v_half))
        return inside, site_pos, int(model.site_bodyid[site_id])
    camera_id = model.camera(camera_name).id
    cam_pos, cam_rot = batch.camera(camera_id)
    if adapter.robot == "apptronik_apollo":
        local = torch.bmm(cam_rot.transpose(1, 2), (targets - cam_pos)[:, :, None]).squeeze(2)
        forward = -local[:, 2]
        v_half = torch.deg2rad(torch.tensor(float(model.cam_fovy[camera_id]) / 2, device=targets.device))
        h_half = torch.atan(torch.tensor(16.0 / 9.0, device=targets.device) * torch.tan(v_half))
        inside = (forward > 0) & (local.norm(dim=1) >= _NEAR_M) & (torch.atan2(local[:, 0], forward).abs() <= h_half) & (torch.atan2(local[:, 1], forward).abs() <= v_half)
        return inside, cam_pos, int(model.site_bodyid[site_id])
    local = torch.bmm(site_rot.transpose(1, 2), (targets - site_pos)[:, :, None]).squeeze(2)
    camera = model.camera(camera_name)
    if adapter.robot == "toddlerbot":
        fx, fy, cx_off, cy_off = model.cam_intrinsic[camera.id]; width, height = model.cam_resolution[camera.id]
        z = local[:, 2]
        px = fx * local[:, 0] / z.clamp_min(1e-9) + width / 2 - .5 - cx_off
        py = fy * local[:, 1] / z.clamp_min(1e-9) + height / 2 - .5 - cy_off
        inside = (z > 0) & (local.norm(dim=1) >= _NEAR_M) & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    else:
        width, height = model.cam_resolution[camera.id]; fy = (height / 2) / np.tan(np.deg2rad(model.cam_fovy[camera.id]) / 2); fx = fy
        h_axis, v_axis = adapter.image_axes
        z = -local[:, 2]
        px = fx * local[:, h_axis] / z.clamp_min(1e-9) + (width - 1) / 2
        py = fy * local[:, v_axis] / z.clamp_min(1e-9) + (height - 1) / 2
        inside = (z > 0) & (local.norm(dim=1) >= _NEAR_M) & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    return inside, site_pos, int(model.site_bodyid[site_id])


def _eye_seen(adapter: _Adapter, model: mujoco.MjModel, batch: _WarpBatch, qpos: torch.Tensor,
              eye: tuple[str | None, str], targets: torch.Tensor, eye_bodies: tuple[int, ...],
              pending: torch.Tensor) -> torch.Tensor:
    """FOV-and-line-of-sight mask for one eye, restricted to the rows indexed by `pending`.

    Returns a full-width `(batch.n,)` mask that is False outside `pending`. `pending` exists because
    both callers evaluate eyes in order and only ask later eyes about targets still unseen.

    The dense/sparse split is a measured tradeoff, not a micro-optimization. Compacting the
    candidate rows into their own batch trades one `make_data` for fewer rays, because `ray_clear`
    tests every world of whatever batch it is handed (see its docstring). On this GPU: ~6 us per
    allocated world against ~13 us per ray, so compacting c of n rows only pays while c < 0.68 n.
    Above that, ray-test the aim batch directly -- its FK already IS the aimed state and `inside` /
    `origins` are already in hand, so the dense branch also deletes a redundant second `_fov` call.
    First eyes are normally dense and later eyes sparse, so both branches see real traffic.
    """
    seen = torch.zeros(batch.n, dtype=torch.bool, device=targets.device)
    inside, origins, _body = _fov(adapter, model, batch, eye, targets)
    candidate = pending[inside[pending]]
    if not candidate.numel():
        return seen
    if candidate.numel() >= 0.68 * batch.n:
        seen[candidate] = batch.ray_clear(origins, targets, eye_bodies, adapter.visual_group)[candidate]
    else:
        ray_batch = _WarpBatch(model, qpos[candidate], batch.wm)
        _inside, ray_origins, _body = _fov(adapter, model, ray_batch, eye, targets[candidate])
        seen[candidate] = ray_batch.ray_clear(ray_origins, targets[candidate], eye_bodies, adapter.visual_group)
    return seen


def score_targets(robot: str, targets_base: np.ndarray, base_qpos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """GPU-score target centers. Returns visible union and legal solved head qpos per target.

    `base_qpos` optionally supplies a PER-TARGET body pose `(N, nq)` to aim and raycast from, instead
    of the one canonical arms-down pose every published number uses. Its purpose is to measure the
    canonical-pose approximation rather than to replace it: `main.tex` claims the arms-down mask is an
    upper bound because "a reaching arm can only lower" coverage, which is a directional claim with no
    containment proof -- an arms-down forearm hanging across the torso occludes too, so a reaching arm
    can also UNBLOCK a line of sight. Passing each voxel's own reaching solution turns that claim into
    a measurement. See `test/armpose_bias.py`.

    Incompatible with the sagittal fold below, which mirrors the target but could not mirror an
    asymmetric arm pose, so the two are mutually exclusive by assertion.
    """
    adapter = _ADAPTERS[robot]
    model, root_pos, base_q = _load_rig(robot)
    assert base_qpos is None or robot not in ("booster_t1", "fourier_gr3"), \
        f"{robot} folds targets sagittally, which is unsound against a per-row arm pose"
    eye_bodies = _eye_bodies(adapter, model)
    visible = np.zeros(len(targets_base), bool); head_q = np.full((len(targets_base), len(adapter.head_joints)), np.nan, np.float32)
    wm = None  # uploaded once by the first chunk's `_aim`, then handed back to every later chunk
    for start in range(0, len(targets_base), _CHUNK):
        stop = min(start + _CHUNK, len(targets_base)); target = torch.as_tensor(targets_base[start:stop], dtype=torch.float32, device="cuda") + torch.tensor(root_pos, dtype=torch.float32, device="cuda")
        # Fold into the +y half-space for robots whose source is EXACTLY mirror-symmetric and
        # whose camera sits on the sagittal plane: visibility(x, y, z) == visibility(x, -y, z),
        # so this halves the unique work with no approximation. GR-3 qualifies because
        # gr3_import.py removes the vendor's sub-millimetre L/R bias.
        if robot in ("booster_t1", "fourier_gr3"): target[:, 1].abs_()
        rows = np.tile(base_q, (stop - start, 1)) if base_qpos is None else base_qpos[start:stop]
        qpos = torch.tensor(rows, dtype=torch.float32, device="cuda")
        qpos, batch = _aim(adapter, model, qpos, target, wm)
        wm = batch.wm
        seen = torch.zeros(stop - start, dtype=torch.bool, device="cuda")
        for eye in adapter.eyes:
            unresolved = torch.nonzero(~seen, as_tuple=False).squeeze(1)
            if not unresolved.numel(): break
            seen |= _eye_seen(adapter, model, batch, qpos, eye, target, eye_bodies, unresolved)
        visible[start:stop] = seen.cpu().numpy()
        head_q[start:stop] = qpos[:, [model.jnt_qposadr[model.joint(name).id] for name in adapter.head_joints]].cpu().numpy()
    return visible, head_q


def score_pairs(robot: str, pairs_a: np.ndarray, pairs_b: np.ndarray) -> np.ndarray:
    """Pairwise coverage: can the rig see BOTH targets of each pair at one instant?

    `pairs_a` / `pairs_b` are `(M, 3)` target centers in the payload base frame. Returns `(M,)` bool.

    THE STATEMENT BEING TESTED, which is the whole reason this function exists:

        a pair is covered iff there exists a SINGLE REALIZABLE HEAD STATE at which some eye sees
        x_a and some eye sees x_b (the two eyes need not be distinct).

    That one sentence specializes correctly to every rig in the study, which is why it replaces the
    two hard-coded branches of `plot_workspace_curobo._pairwise_vrw`. For independent gimbals the
    state is the tuple of per-module angles and "left on x_a, right on x_b" is one such state. For a
    coupled neck a single state constrains every eye at once, which is exactly the physics the
    metric must charge for -- and is the reason single-target coverage cannot separate a 2-DOF neck
    carrying two eyes from two independently aimed eyes, while this can.

    Structure. `_Adapter.aim_groups` partitions the eyes into independently steerable groups, so the
    head state factorizes and each group picks its own aim. Per group `g` and candidate aim `c` the
    kernel records whether group `g` sees each target at `c`, then:

      * same group, same candidate: `sees_a[g,c] and sees_b[g,c]` -- one state, either eye. This is
        the ENTIRE result for a coupled neck or a bolted head, and for a fixed back-to-back rig it
        is what lets a front/rear pair be covered by opposite eyes.
      * different groups: group `g` holds x_a while `g' != g` holds x_b, each at its own candidate,
        because the groups are mechanically independent. Counted with the same identity
        `_pairwise_vrw` uses: `n_a * n_b - overlap > 0` over the per-group any-candidate masks,
        which is identically zero at one group, as it must be.

    Candidate aims are `{x_a, x_b, bisector, rest}` (see `_aim_bisector`), every group aimed at the
    same one. SOUND: a True is a witness, since field of view and line of sight are re-evaluated at
    the actual joint-limit-clamped solved state, never at the requested aim. INCOMPLETE: no search
    over aims, so a pair coverable only from some third direction reads False. The incompleteness is
    identical for every rig -- in particular the coupled platforms get the same best-effort bisector
    attempt our own K=1 rig gets, so a surviving gap is not an artifact of searching harder on our
    own behalf. This is the same bias already declared in the paper's Limitations.

    NO SAGITTAL FOLD, unlike `score_targets`, and the omission is mandatory rather than lazy.
    Folding y onto one half-space is exact for a single target under mirror symmetry, but folding
    the two targets of a pair independently CHANGES THEIR SEPARATION and can collapse them onto the
    same side, which inflates coverage. Booster T1 and GR-3 therefore cost about twice here.

    Occlusion between camera modules is ignored, because `_eye_bodies` excludes every mount-chain
    body from every eye's rays. That is inherited from `score_targets` rather than chosen here, and
    it is why `sees_a[g,c]` is well defined without knowing what the other groups are doing.
    """
    adapter = _ADAPTERS[robot]
    model, root_pos, base_q = _load_rig(robot)
    eye_bodies = _eye_bodies(adapter, model)
    groups = _aim_groups(adapter)
    # A fixed head still needs one pseudo-group to hold its eyes, at its one realizable state.
    pair_groups = groups or ((adapter.aim_site, (), tuple(range(len(adapter.eyes)))),)
    root = torch.tensor(root_pos, dtype=torch.float32, device="cuda")
    covered = np.zeros(len(pairs_a), bool)
    wm = None  # uploaded once, then reused by every later candidate and chunk
    for start in range(0, len(pairs_a), _CHUNK):
        stop = min(start + _CHUNK, len(pairs_a)); n = stop - start
        a = torch.as_tensor(pairs_a[start:stop], dtype=torch.float32, device="cuda") + root
        b = torch.as_tensor(pairs_b[start:stop], dtype=torch.float32, device="cuda") + root
        # Candidates: aim at each endpoint, aim at the bisector, and the canonical rest pose. A
        # bolted head cannot aim, so all four collapse to the rest pose and only that one is run.
        aims = ("a", "b", "bisector", "rest") if groups else ("rest",)
        sees = torch.zeros(len(aims), 2, len(pair_groups), n, dtype=torch.bool, device="cuda")
        for candidate, kind in enumerate(aims):
            qpos = torch.tensor(np.tile(base_q, (n, 1)), dtype=torch.float32, device="cuda")
            if kind == "rest":
                batch = _WarpBatch(model, qpos, wm)
            elif kind == "bisector":
                qpos, batch = _aim_bisector(adapter, model, qpos, a, b, wm)
            else:
                qpos, batch = _aim(adapter, model, qpos, a if kind == "a" else b, wm)
            wm = batch.wm
            for which, target in enumerate((a, b)):
                for gi, (_site, _joints, eye_idx) in enumerate(pair_groups):
                    seen = sees[candidate, which, gi]  # a view, so `|=` writes through
                    for e in eye_idx:
                        unresolved = torch.nonzero(~seen, as_tuple=False).squeeze(1)
                        if not unresolved.numel(): break
                        seen |= _eye_seen(adapter, model, batch, qpos, adapter.eyes[e], target,
                                          eye_bodies, unresolved)
        same = (sees[:, 0] & sees[:, 1]).any(0).any(0)
        a_any, b_any = sees[:, 0].any(0), sees[:, 1].any(0)  # (G, n): group sees it at SOME candidate
        n_a, n_b = a_any.sum(0).long(), b_any.sum(0).long()
        distinct = (n_a * n_b - (a_any & b_any).sum(0).long()) > 0
        covered[start:stop] = (same | distinct).cpu().numpy()
    return covered


def _visible_at_states(adapter: _Adapter, model: mujoco.MjModel, qpos: torch.Tensor, target: torch.Tensor) -> bool:
    """Exact GPU FOV/LOS union for one target over concrete, legal source qpos states."""
    eye_bodies = _eye_bodies(adapter, model)
    batch = _WarpBatch(model, qpos)
    targets = target.expand(qpos.shape[0], -1)
    seen = torch.zeros(qpos.shape[0], dtype=torch.bool, device=qpos.device)
    for eye in adapter.eyes:
        unresolved = torch.nonzero(~seen, as_tuple=False).squeeze(1)
        if not unresolved.numel():
            break
        inside, _origins, _body = _fov(adapter, model, batch, eye, targets)
        candidate = unresolved[inside[unresolved]]
        if candidate.numel():
            ray_batch = _WarpBatch(model, qpos[candidate], batch.wm)
            _inside, origins, _body = _fov(adapter, model, ray_batch, eye, targets[candidate])
            seen[candidate] = ray_batch.ray_clear(origins, targets[candidate], eye_bodies, adapter.visual_group)
    return bool(seen.any())


def _bimanual_isolated_blind(unique_voxels: np.ndarray, visible: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Return source compact IDs behind a bimanual 3-D isolated blind cell; no label is inferred."""
    nx, ny, nz = map(int, grid)
    compact = {int(v): i for i, v in enumerate(unique_voxels.tolist())}
    def mirror(v: int) -> int:
        return (v // (ny * nz)) * ny * nz + (ny - 1 - (v % (ny * nz)) // nz) * nz + v % nz
    displayed = {}
    for voxel, idx in compact.items():
        displayed[voxel] = bool(visible[idx]) or displayed.get(voxel, False)
        mirrored = mirror(voxel)
        displayed[mirrored] = bool(visible[idx]) or displayed.get(mirrored, False)
    repair = set()
    for voxel, is_visible in displayed.items():
        if is_visible:
            continue
        x = voxel // (ny * nz); y = (voxel % (ny * nz)) // nz; z = voxel % nz
        seen_neighbors = 0
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            xx, yy, zz = x + dx, y + dy, z + dz
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                seen_neighbors += displayed.get(xx * ny * nz + yy * nz + zz, False)
        if seen_neighbors >= 5:
            for source in (voxel, mirror(voxel)):
                if source in compact:
                    repair.add(compact[source])
    return np.asarray(sorted(repair), dtype=np.int64)


def _lattice_rescore(robot: str, target_base: np.ndarray) -> bool:
    """Return true only if an actual legal lattice head state sees this one target."""
    from itertools import product

    adapter = _ADAPTERS[robot]
    model, root_pos, base = _load_rig(robot)
    joints = [int(model.jnt_qposadr[model.joint(name).id]) for name in adapter.head_joints]
    limits = [model.jnt_range[model.joint(name).id] for name in adapter.head_joints]
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    candidates = np.tile(base, (len(fractions) ** len(joints), 1))
    for row, choice in enumerate(product(fractions, repeat=len(joints))):
        for column, qadr in enumerate(joints):
            lo, hi = limits[column]; candidates[row, qadr] = lo + choice[column] * (hi - lo)
    qpos = torch.as_tensor(candidates, dtype=torch.float32, device="cuda")
    _apply_coupling(robot, qpos, model)
    target = torch.as_tensor(target_base, dtype=torch.float32, device="cuda") + torch.as_tensor(root_pos, dtype=torch.float32, device="cuda")
    if robot in ("booster_t1", "fourier_gr3"):
        target[1].abs_()
    return _visible_at_states(adapter, model, qpos, target[None])


def use_camera(robot: str, camera: str) -> None:
    """Point a robot's adapter at a different declared camera, for sensitivity runs.

    GR-3's vendor documentation disagrees with itself about the head-camera mount pitch by
    25 deg, so its source MJCF declares both variants (`head_cam` at the URDF's 15 deg,
    `head_cam_fovfig` at the 40 deg its official FOV figure draws) and both are scored. The
    override is a module-level rebind rather than a parameter thread because every helper
    below reads the adapter from `_ADAPTERS`.
    """
    base = _ADAPTERS[robot]
    _ADAPTERS[robot] = _Adapter(
        base.robot, ((camera, f"{camera}_site"),), f"{camera}_site",
        base.aim_axis, base.aim_sign, base.head_joints, base.visual_group,
        base.dls_steps, base.image_axes,
    )


def score_workspace_gpu(workspace: pathlib.Path, out: pathlib.Path) -> None:
    """Score each successful workspace voxel once and atomically save a row-aligned GPU sidecar."""
    payload = torch.load(workspace, map_location="cpu", weights_only=False); robot = str(payload["robot"])
    if robot not in _ADAPTERS: raise ValueError(f"GPU visibility unsupported for {robot!r}")
    success = payload["success"].bool(); success_idx = torch.where(success)[0]
    voxel = payload["voxel_index"][success_idx].numpy(); _, first, inverse = np.unique(voxel, return_index=True, return_inverse=True)
    reps = success_idx.numpy()[first]; representative_targets = payload["target_pos"][reps].numpy()
    visible_voxel, head_q_voxel = score_targets(robot, representative_targets)
    repair = _bimanual_isolated_blind(voxel[first], visible_voxel, payload["n_grid_per_axis"].numpy())
    repaired = 0
    for compact in repair:
        if _lattice_rescore(robot, representative_targets[compact]):
            visible_voxel[compact] = True; repaired += 1
    nrows = int(success.numel()); visible = torch.zeros(nrows, dtype=torch.bool); head_q = torch.full((nrows, head_q_voxel.shape[1]), float("nan"))
    visible[success_idx] = torch.from_numpy(visible_voxel[inverse]); head_q[success_idx] = torch.from_numpy(head_q_voxel[inverse])
    digest = hashlib.sha256(pathlib.Path(workspace).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save({"robot": robot, "workspace": str(workspace.resolve()), "workspace_rows": nrows, "visible_left": visible, "visible_right": visible, "head_q": head_q, "head_joint_names": _ADAPTERS[robot].head_joints, "scorer": "gpu_warp_dynamic_mesh_v3_directional_dls_lattice_rescore", "head_aim": "source_mjcf_directional_dls", "workspace_sha256": digest, "canonical_pose": "reachability_visibility_arms_down_v1", "isolated_blind_candidates": int(repair.size), "isolated_blind_directly_visible": repaired}, tmp)
    tmp.replace(out)


if __name__ == "__main__":  # pragma: no cover - `score_visibility.py` is the entrypoint
    raise SystemExit("run `score_visibility.py <workspace> <out>` (add --device cpu for the reference scorer)")
