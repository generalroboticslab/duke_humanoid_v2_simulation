"""Robot-agnostic, dual-format serializer for a resolved dynamic ``MjSpec`` variant.

Resolves the grafted planner/sim variant (actuated head camera + actuated parallel
grippers) of a target robot and serializes it as either:

- ``--format mjcf`` -- a **lossless** native dump of the resolved variant
  (``Entity.write_xml`` / ``Entity.to_zip``). MuJoCo->MuJoCo, no information loss;
  the grafted camera/gripper bodies are already in the spec, no compile needed to
  serialize. Value: a flattened, standalone-loadable snapshot of the exact variant
  the planner/sim uses, which today only exists by re-running the graft code.
- ``--format urdf`` -- the **minimum-information-loss** conversion for cuRobo
  collision-aware planning. Emits two
  artifacts (faithful floating-base ``_full.urdf`` + fixed-base ``_curobo.urdf``) and
  hard-gates them through yourdfpy scene-graph FK + cuRobo ``RobotBuilder`` (needs
  yourdfpy >= 0.0.60 under numpy>=2; ``_ensure_yourdfpy`` auto-upgrades for reproducibility).

Robot-agnostic: the only robot-specific input is the small ``ROBOTS`` table
(cfg-fn, root body, expected compile snapshot, tool sites, output dir). The core
resolve/serialize path hardcodes no robot name. Artifacts land in each robot's own
``asset/<robot>/`` via ``RobotSpec.out_dir``; ``--output`` overrides it. Both
``get_humanoid_v21_robot_cfg`` and
``get_g1_robot_cfg`` accept the same keyword args (``head_camera``/``end_effector``/
``hand``); they are always called by keyword so their differing defaults/arg-order
do not matter.

GROUP_ROLE map (URDF geom emit, documented here for the whole exporter; used by the
URDF path only). MuJoCo geom ``group`` is the role selector:

    group | type    | name pattern      | role
    ------+---------+-------------------+---------------------------------------
      2   | MESH    | ``*_visual``      | fit       (MORPHIT source AND <visual>)
      3   | CAPSULE | ``*_collision*``  | collision (<collision> provenance)
      4   | CAPSULE | ``cam_*_fov*``    | drop      (viewer-only FOV cones)
      5   | CAPSULE | ``*_rack_ikproxy``| collision (simplified; racks only, g5>g3)

    GROUP_ROLE = {2: "fit", 3: "collision", 4: "drop", 5: "collision"}

Fail-loud rule (URDF path): a body with geom but no ``fit``-role geom is a bug.
A body with ZERO geoms (census: ``cam_base``, a bare gimbal pivot frame) is legal --
it emits a fixed jointless link and no spheres. Guard is
``has_geom and not has_fit_geom`` -> fail; ``not has_geom`` -> skip.

Run:
    python asset/create/export_mjspec_to_urdf.py \
        --robot humanoid_v21 --format mjcf
    python asset/create/export_mjspec_to_urdf.py --robot g1 --head-camera builtin
"""

import argparse
import os
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "mj_envs"))

from mjlab.entity.entity import Entity  # noqa: E402

from mj_envs.asset_zoo.fov_frustum import FOV_HULL_SUFFIX  # noqa: E402 (viewer overlay, stripped before export)

# First yourdfpy release whose _forward_kinematics_joint uses q.item() instead of float(q),
# so scene-graph FK (and cuRobo's internal FK) survives numpy>=2. 0.0.58 crashes with
# "only 0-dimensional arrays can be converted to Python scalars".
_MIN_YOURDFPY = (0, 0, 60)

# Snapshot = (nbody, njnt, ngeom, nmesh, nsite, nu). Per-robot verify gate captured
# live 2026-07-05 for the actuated/actuated/parallel_gripper variant. A mismatch means
# the variant changed upstream; fail-loud and update the plan before continuing.
_Snapshot = tuple[int, int, int, int, int, int]

GROUP_ROLE = {2: "fit", 3: "collision", 4: "drop", 5: "collision"}


def _humanoid_cfg_fn():
    from asset_zoo.humanoid_v21.humanoid_v21_constants import get_humanoid_v21_robot_cfg
    return get_humanoid_v21_robot_cfg


def _g1_cfg_fn():
    from asset_zoo.g1.g1_constants import get_g1_robot_cfg
    return get_g1_robot_cfg


class _SourceMjcfEntity:
    """Minimal :class:`Entity` stand-in for a robot that IS a plain source MJCF file.

    Third-party robots (Booster T1, Fourier GR-3, ...) enter this repo as a standalone MJCF
    under ``asset/<robot>/`` rather than as an mjlab ``EntityCfg`` variant assembled from
    grafted modules, so there is no ``cfg_fn(head_camera=, end_effector=, hand=)`` to call.
    The exporter core only ever touches ``.spec``, ``.compile()`` and ``.write_xml()``, so a
    three-method shim is enough and keeps that core untouched.
    """

    def __init__(self, path: pathlib.Path):
        self.spec = mujoco.MjSpec.from_file(str(path))

    def compile(self) -> mujoco.MjModel:
        return self.spec.compile()

    def write_xml(self, path) -> None:
        self.spec.compile()  # to_xml needs a compiled spec
        pathlib.Path(path).write_text(self.spec.to_xml())


class RobotSpec:
    """Per-robot config: everything the agnostic core needs and nothing more.

    Exactly one of ``cfg_fn_getter`` (mjlab variant) or ``source_mjcf`` (standalone MJCF)
    must be given.
    """

    def __init__(self, name, cfg_fn_getter=None, *, root_body, snapshot, tool_sites,
                 out_dir, source_mjcf=None):
        assert (cfg_fn_getter is None) != (source_mjcf is None), (
            f"{name}: give exactly one of cfg_fn_getter / source_mjcf"
        )
        self.name = name
        self._cfg_fn_getter = cfg_fn_getter  # deferred import; only load the robot asked for
        self.source_mjcf = pathlib.Path(source_mjcf) if source_mjcf else None
        self.root_body = root_body
        self.snapshot: _Snapshot = snapshot
        self.tool_sites = tool_sites
        # Where this robot's artifacts belong, so regeneration is correct without --output.
        # Previously the default was asset/create for every robot and callers had to remember
        # `--output asset/<robot>`; forgetting it silently wrote robot data into the library dir.
        self.out_dir = pathlib.Path(out_dir)

    @property
    def cfg_fn(self):
        return self._cfg_fn_getter()


ROBOTS = {
    "humanoid_v21": RobotSpec(
        "humanoid_v21", _humanoid_cfg_fn, root_body="base_link",
        # nmesh 40 -> 42: wrist_1_L and wrist_3_L are each mirrored about their own YZ
        # plane, so both load their twin's .obj a second time at a negated x scale
        # (see builder_helpers.mirror_link_about_plane).
        # 42 -> 53: the same reflection was then applied to the remaining 11 left-side
        # links (hip_2 hip_3 knee shank ankle_1 foot shoulder_2 shoulder_3 elbow wrist_2
        # end_effector), which had been reusing their right-side twin's mesh and inertia
        # unreflected -- left/right differed by up to 12.6 mm. Each adds one _mirrorx asset.
        # 53 -> 54, ngeom 158 -> 160 (pin refreshed 2026-08-23): head_camera_creation.py splits
        # pitch_left.obj into pitch_left_bracket.obj + pitch_left_d436.obj (`EXTRA_VISUAL`), so
        # the mesh set gains one asset and each of the two pitch links gains one visual geom.
        # The pin had gone stale against that change while the rig-baseline comment below was
        # updated, which is why the two disagreed on ngeom (158 vs 160).
        snapshot=(42, 36, 160, 54, 9, 33),
        tool_sites=("end_effector_L", "end_effector_R"),
        out_dir=_REPO_ROOT / "asset" / "duke_v2" / "humanoid_v21",
    ),
    "g1": RobotSpec(
        "g1", _g1_cfg_fn, root_body="pelvis",
        # head_camera="builtin" variant (original G1 head mesh kept); export/run this robot
        # with --head-camera builtin so this snapshot holds. The parallel_gripper graft now carries
        # per-arm grasp-center sites (`{left,right}_hand_grasp`), the cuRobo tool frames.
        # nsite 11 -> 13 (2026-08-05): the graft also emits `end_effector_{L,R}`, the wrist-roll
        # output flanges, matching humanoid_v21's own tool_sites naming. Sites only -- no geom,
        # joint or actuator moved, which is what says this is the gripper graft and not drift.
        snapshot=(37, 34, 124, 57, 13, 31),
        tool_sites=("left_hand_grasp", "right_hand_grasp"),
        out_dir=_REPO_ROOT / "asset" / "unitree_g1",
    ),
    "fourier_gr3": RobotSpec(
        "fourier_gr3", root_body="base_link",
        # Standalone vendor import (Fourier GR-3 v2.1.1 dummy-hand variant), regenerated by
        # asset/fourier_gr3/gr3_import.py. nu=0 by design: the vendor ships no actuators and
        # this study only needs kinematics/collision. ncam=2 (the 15-deg URDF mount and the
        # 40-deg official-FOV-figure sensitivity twin) are not part of the snapshot tuple.
        source_mjcf=_REPO_ROOT / "asset" / "fourier_gr3" / "gr3.xml",
        snapshot=(40, 32, 81, 37, 4, 0),
        tool_sites=("end_effector_L_site", "end_effector_R_site"),
        out_dir=_REPO_ROOT / "asset" / "fourier_gr3",
    ),
    "pal_talos": RobotSpec(
        "pal_talos", root_body="base_link",
        # Standalone vendor import (PAL Robotics TALOS), regenerated by
        # asset/pal_talos/talos_study_import.py from the pristine vendor asset/pal_talos/talos.xml.
        # nu=0 by design (kinematics/collision only). The 45 group-3 capsules are PCA
        # principal-axis inscribed refits of the vendor mesh colliders; ncam=1 (Orbbec Astra Pro
        # RGB only, since 2026-07-25 -- the depth-stream sensitivity twin was dropped) is not
        # part of the snapshot tuple. ngeom also carries the 13 display-only group-4 FOV
        # wireframe capsules (`_add_fov_wireframe` in talos_study_import.py).
        source_mjcf=_REPO_ROOT / "asset" / "pal_talos" / "talos_study.xml",
        snapshot=(46, 45, 117, 74, 8, 0),
        tool_sites=("end_effector_L_site", "end_effector_R_site"),
        out_dir=_REPO_ROOT / "asset" / "pal_talos",
    ),
}


def build_variant(robot: RobotSpec, head_camera: str, end_effector: str, hand: str):
    """Resolve the variant to export.

    Standalone-MJCF robots ignore the graft flags (they have no swappable modules); mjlab
    robots get the keyword call, which stays robot-agnostic despite the cfg-fns' differing
    signatures/defaults."""
    if robot.source_mjcf is not None:
        print(f"[variant] {robot.name}: source MJCF {robot.source_mjcf}")
        return _SourceMjcfEntity(robot.source_mjcf)
    cfg = robot.cfg_fn(head_camera=head_camera, end_effector=end_effector, hand=hand)
    return Entity(cfg)


def _snapshot_of(model: mujoco.MjModel) -> _Snapshot:
    return (model.nbody, model.njnt, model.ngeom, model.nmesh, model.nsite, model.nu)


def _force_wireframe_for_preflight() -> None:
    """The exporter's snapshot tuples include the 13-per-cam wireframe capsules (they are part
    of the compiled model, even though viewer code defaults FOV_WIREFRAME=False to hide them).
    Flipping the flag here ONCE for preflight keeps the URDF-model snapshot pin valid without
    forcing every runtime viewer to bake wireframes it never renders."""
    import mj_envs.asset_zoo.fov_frustum as _fov
    _fov.FOV_WIREFRAME = True


def _force_wireframe_for_preflight() -> None:
    # Snapshot tuples were captured with the wireframe baked; without this override the runtime
    # default FOV_WIREFRAME=False would shift ngeom and break every preflight. The override is
    # exporter-local; viewers continue to default to no-wireframe.
    import mj_envs.asset_zoo.fov_frustum as _fov
    _fov.FOV_WIREFRAME = True


def strip_fov_hull(spec) -> int:
    """Drop the translucent FOV hull (geoms + their inline meshes) from ``spec``. Returns geoms removed.

    The hull is a VIEWER artifact baked into ``get_spec`` by ``asset_zoo.fov_frustum`` so the
    head-camera cone is visible in every viewer. It has no business in a URDF: its geom sits in
    the dropped group (``GROUP_ROLE[4] == "drop"``) so nothing referenced it in the emit, but its
    MESH is ``uservert`` (inline verts, no file on disk), which URDF cannot express at all --
    URDF meshes are file references.

    Removing it here rather than teaching the census to skip it keeps the exporter's snapshot
    tuples measuring the URDF-relevant model, so adding a viewer overlay upstream cannot move a
    pin that exists to catch KINEMATIC drift. The wireframe cone is deliberately NOT stripped: it
    is already inside the pinned ``ngeom`` and removing it would move every snapshot.
    """
    removed = 0
    for geom in list(spec.geoms):
        if geom.name.endswith(FOV_HULL_SUFFIX):
            spec.delete(geom)
            removed += 1
    for mesh in list(spec.meshes):
        if mesh.name.endswith(f"{FOV_HULL_SUFFIX}_mesh"):
            spec.delete(mesh)
    for mat in list(spec.materials):
        if mat.name.endswith(f"{FOV_HULL_SUFFIX}_mat"):
            spec.delete(mat)
    return removed


# Compile snapshots for the humanoid_v21 head-camera rig variants used by the K=1/2/3 camera-count
# ablation. `robot.snapshot` pins the shipped dual rig; a different rig is an INTENTIONAL variant,
# not drift, so it needs its own pin rather than a relaxed assert. Each entry differs from the dual
# baseline (42, 36, 160, 54, 9, 33) by exactly one camera module -- nbody +-3 (base/yaw/pitch links),
# njnt +-2, ngeom +-19, nsite +-2 (front_center + rgb), nu +-2 -- which is the check that these are
# the right numbers rather than whatever the compiler happened to emit. nmesh is camera-independent,
# so all three share it.
#
# ngeom refreshed 2026-08-23 (140 -> 141, 176 -> 179) alongside the dual pin: the pitch_left
# bracket/d436 split adds one visual geom PER CAMERA, so the shift is 1/2/3 for K=1/2/3. The
# +-19-per-module relation still holds exactly (160-141 = 19, 179-160 = 19), which is what says
# these are the split's numbers and not drift.
_HUMANOID_V21_RIG_SNAPSHOTS = {
    "actuated_single": (39, 34, 141, 54, 7, 31),
    "actuated_triple": (45, 38, 179, 54, 11, 35),
}


def preflight(robot: RobotSpec, entity: Entity, expected: _Snapshot | None = None) -> mujoco.MjModel:
    """Phase 0: compile, assert the per-robot snapshot, assert every resolved mesh
    exists with a URDF-loadable suffix. Fail-loud on any drift.

    ``expected`` overrides ``robot.snapshot`` for a deliberate variant (see
    ``_HUMANOID_V21_RIG_SNAPSHOTS``); it is still an exact equality assert, just against a
    different pin."""
    model = entity.compile()
    got = _snapshot_of(model)
    expected = expected or robot.snapshot
    assert got == expected, (
        f"{robot.name} snapshot drift: got {got}, expected {expected} "
        f"(nbody,njnt,ngeom,nmesh,nsite,nu). Variant changed -- update plan."
    )
    # Mesh census from the resolved spec (file paths live on the spec, not mj_model).
    mesh_files = [m.file for m in entity.spec.meshes if m.file]
    n_expected = expected[3]
    assert len(mesh_files) == n_expected, (
        f"{robot.name}: {len(mesh_files)} spec mesh files, expected {n_expected}"
    )
    bad = [f for f in mesh_files if pathlib.Path(f).suffix.lower() not in (".stl", ".obj")]
    assert not bad, f"{robot.name}: non-URDF-loadable mesh suffix(es): {bad}"
    print(f"[preflight] {robot.name} snapshot OK {got}; {n_expected} meshes, all .stl/.obj")
    return model


def build_mjcf(robot: RobotSpec, entity: Entity, live_model: mujoco.MjModel,
               out_dir: pathlib.Path) -> None:
    """Lossless MJCF emit + round-trip gate.

    Emits ``<robot>_resolved.xml`` -- a flattened, standalone snapshot of the grafted
    variant. Mesh refs stay file paths (MuJoCo does NOT embed file-ref meshes in
    ``to_zip`` for this asset layout, so a self-contained zip is not produced). To make
    the XML location-independent, the base ``meshdir`` is rewritten to its absolute source
    path before emit; grafted camera/gripper meshes are already absolute. The XML thus
    reloads from any working directory as long as the source mesh tree is present.

    Round-trip gate (lossless): reload the emitted XML and assert the compile snapshot
    (nbody/njnt/ngeom/nmesh/nsite/nu) is identical to the live variant. Counts identical
    = zero structural loss."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = entity.spec
    # Rewrite relative base meshdir -> absolute so the emitted XML is not tied to out_dir.
    if not pathlib.Path(spec.meshdir).is_absolute():
        spec.meshdir = str((pathlib.Path(spec.modelfiledir) / spec.meshdir).resolve())
    xml_path = out_dir / f"{robot.name}_resolved.xml"
    entity.write_xml(xml_path)
    print(f"[mjcf] wrote {xml_path} (abs base meshdir; portable)")
    reloaded = mujoco.MjModel.from_xml_path(str(xml_path))
    got, exp = _snapshot_of(reloaded), _snapshot_of(live_model)
    assert got == exp, f"[mjcf] round-trip snapshot mismatch: {got} vs live {exp}"
    print(f"[mjcf] round-trip OK, identical snapshot {got}")


def _ensure_yourdfpy():
    """Reproducibility guard: auto-upgrade yourdfpy to >= _MIN_YOURDFPY in place.

    The URDF gate below runs full scene-graph FK (yourdfpy) + cuRobo RobotBuilder, both of
    which crash under numpy>=2 on yourdfpy < 0.0.60. Rather than let the gate's result depend
    on whatever version a given machine happens to have, pin the floor here and pip-install it
    if missing, so the export is deterministic. Returns the (possibly reloaded) yourdfpy module."""
    import importlib

    import yourdfpy
    cur = tuple(int(x) for x in yourdfpy.__version__.split(".")[:3])
    if cur >= _MIN_YOURDFPY:
        return yourdfpy
    want = ".".join(map(str, _MIN_YOURDFPY))
    print(f"[env] yourdfpy {yourdfpy.__version__} < {want}; upgrading in place (numpy2 FK fix)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"yourdfpy>={want}"])
    importlib.reload(yourdfpy)
    got = tuple(int(x) for x in yourdfpy.__version__.split(".")[:3])
    assert got >= _MIN_YOURDFPY, f"yourdfpy still {yourdfpy.__version__} after upgrade to {want}"
    print(f"[env] yourdfpy now {yourdfpy.__version__}")
    return yourdfpy


def _name(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i)


def _quat_wxyz_to_rpy(q):
    """MuJoCo quat (wxyz) -> URDF rpy (fixed-axis XYZ). scipy wants xyzw."""
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def _origin(pos, quat):
    r, p, y = _quat_wxyz_to_rpy(quat)
    return f'<origin xyz="{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}" rpy="{r:.9g} {p:.9g} {y:.9g}"/>'


def _pose_matrix(pos, quat):
    """Homogeneous transform from MuJoCo body/geom pose fields."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    out[:3, 3] = pos
    return out


def _origin_from_matrix(transform):
    r, p, y = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")
    pos = transform[:3, 3]
    return f'<origin xyz="{pos[0]:.9g} {pos[1]:.9g} {pos[2]:.9g}" rpy="{r:.9g} {p:.9g} {y:.9g}"/>'


def _mesh_asset_inverse(model, mesh_id):
    """Inverse MuJoCo mesh asset transform.

    MuJoCo renders compiled mesh vertices, not raw OBJ/STL vertices. The compiler applies
    scale, then stores the asset recenter/reorientation in ``mesh_pos/mesh_quat`` such that
    compiled vertices equal ``R.T @ (scaled_source - mesh_pos)``. URDF readers load the raw
    source mesh, so the visual origin must include this inverse asset transform while the
    mesh filename and scale still reference the source file directly.
    """
    rot = Rotation.from_quat([
        model.mesh_quat[mesh_id, 1],
        model.mesh_quat[mesh_id, 2],
        model.mesh_quat[mesh_id, 3],
        model.mesh_quat[mesh_id, 0],
    ]).as_matrix()
    inv = np.eye(4)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ model.mesh_pos[mesh_id]
    return inv


def _mesh_paths(spec, relative_to: pathlib.Path | None = None):
    """Resolve every mesh asset to a SOURCE path, never a copied/generated mesh.

    Absolute paths are correct for cuRobo because its ``asset_path`` can be ``/``. VSCode's
    URDFLoader treats an absolute POSIX filename as relative to the URDF URL, producing
    ``<urdf-dir>//home/...`` and a 404. For the faithful full URDF, emit repo-local relative
    paths from the URDF directory instead: same source mesh files, viewer-friendly URL
    resolution, same per-mesh scale.
    """
    base = pathlib.Path(spec.modelfiledir)
    md = pathlib.Path(spec.meshdir)
    out = {}
    for mesh in spec.meshes:
        f = pathlib.Path(mesh.file)
        p = f if f.is_absolute() else (base / md / f)
        p = p.resolve()
        if relative_to is None:
            out[mesh.name] = str(p)
        else:
            out[mesh.name] = pathlib.Path(os.path.relpath(p, start=relative_to.resolve())).as_posix()
    return out


def _inertial_xml(model, b):
    """URDF <inertial>. MuJoCo gives principal (diagonal) inertia + iquat (principal->body);
    URDF origin rpy = euler(iquat) aligns the tensor frame to principal axes, so the URDF tensor
    is diagonal with the MuJoCo principal values."""
    m = model.body_mass[b]
    ixx, iyy, izz = model.body_inertia[b]
    o = _origin(model.body_ipos[b], model.body_iquat[b])
    return (f'<inertial>{o}<mass value="{m:.9g}"/>'
            f'<inertia ixx="{ixx:.9g}" ixy="0" ixz="0" iyy="{iyy:.9g}" iyz="0" izz="{izz:.9g}"/>'
            f'</inertial>')


def _primitive_geometry_xml(model, g, gname):
    """MuJoCo primitive geom -> URDF ``<geometry>`` body. Shared by the visual and collision
    emitters so a primitive means the same shape in both roles.

    URDF has no capsule primitive, so capsule and cylinder both become a URDF cylinder from the
    same size/pose fields (the capsule accepts only the rounded-cap loss, the cylinder is exact);
    sphere and box map exactly."""
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return f'<geometry><sphere radius="{model.geom_size[g, 0]:.9g}"/></geometry>'
    if model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, length = model.geom_size[g, 0], 2.0 * model.geom_size[g, 1]
        return (f'<geometry><cylinder radius="{radius:.9g}" length="{length:.9g}"/>'
                f'</geometry>')
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        # MuJoCo geom_size holds HALF-extents; URDF <box size> is the full edge lengths.
        s = 2.0 * model.geom_size[g, :3]
        return (f'<geometry><box size="{s[0]:.9g} {s[1]:.9g} {s[2]:.9g}"/>'
                f'</geometry>')
    raise ValueError(f"unsupported primitive geom type {model.geom_type[g]} for {gname}")


def _visual_xml(model, g, mesh_abs, meshname_of):
    """g2 fit geom -> URDF <visual>. Origin from geom_pos/geom_quat (body frame).

    Usually a mesh, but a vendor import may decorate a body with small primitive visuals
    (TALOS ships a 5 mm marker cube on ``torso_2_link`` and four wrist pucks); those emit the
    same URDF primitive the collision path would, so no visual geom is silently dropped.
    Carries the mesh's MuJoCo `mesh_scale` (e.g. 0.001 for mm-source meshes) so cuRobo
    + any URDF consumer gets the same world extent as the MuJoCo model. Without the scale
    the mesh inflates ~1000x and MORPHIT spheres explode correspondingly. The origin also
    carries MuJoCo's compiled mesh asset transform, otherwise URDF viewers show the raw CAD
    frame instead of the rendered MuJoCo mesh frame."""
    mn = meshname_of(g)
    gname = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
    rgba = model.geom_rgba[g]
    mat = (f'<material name="{gname}_mat"><color rgba="{rgba[0]:.4g} {rgba[1]:.4g} '
           f'{rgba[2]:.4g} {rgba[3]:.4g}"/></material>')
    if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        geometry = _primitive_geometry_xml(model, g, gname)
        return f'<visual name="{gname}">{_origin(model.geom_pos[g], model.geom_quat[g])}{geometry}{mat}</visual>'
    mesh_id = model.geom_dataid[g]
    if mesh_id >= 0:
        origin = _pose_matrix(model.geom_pos[g], model.geom_quat[g]) @ _mesh_asset_inverse(model, mesh_id)
        o = _origin_from_matrix(origin)
        s = model.mesh_scale[mesh_id]
        scale_xml = f' scale="{s[0]:.9g} {s[1]:.9g} {s[2]:.9g}"'
    else:
        o = _origin(model.geom_pos[g], model.geom_quat[g])
        scale_xml = ""
    return (f'<visual name="{gname}">{o}'
            f'<geometry><mesh filename="{mesh_abs[mn]}"{scale_xml}/></geometry>{mat}</visual>')


def _collision_xml(model, g):
    """MuJoCo collision capsule/cylinder/sphere -> URDF collision primitive.

    The project collision geoms are group-3/group-5 capsules plus the occasional true
    cylinder (e.g. UR5e's ``eef_collision`` wrist geom) and sphere (e.g. G1's
    ``pelvis_collision``). URDF has no capsule primitive, so capsule/cylinder are both emitted
    as a URDF cylinder from the same size/pose fields (the capsule accepts only the rounded-cap
    loss, the cylinder is exact); a sphere maps to a URDF sphere exactly. These <collision> tags
    are decorative for the faithful twin -- cuRobo ignores them and reads spheres from
    ``mj_collision_spheres.build_collision_spheres``. No mesh fitting or visual mirroring.
    """
    gname = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
    o = _origin(model.geom_pos[g], model.geom_quat[g])
    return f'<collision name="{gname}">{o}{_primitive_geometry_xml(model, g, gname)}</collision>'


def _joint_xml(name, jtype, parent, child, origin_xml, axis=None, rng=None, default=None):
    """Emit one URDF joint in cuRobo coordinates (MuJoCo qpos minus compiled qpos0 ref)."""
    axis_xml = f'<axis xyz="{axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g}"/>' if axis is not None else ""
    if rng is not None:
        limit = f'<limit effort="1000" velocity="10" lower="{rng[0]:.9g}" upper="{rng[1]:.9g}"/>'
    else:
        limit = '<limit effort="1000" velocity="10"/>' if jtype == "continuous" else ""
    # Only revolute joints carry a default; fixed / floating / prismatic emit no default.
    default_xml = "" if default is None or jtype != "revolute" else f'<default value="{default:.9g}"/>'
    return (f'<joint name="{name}" type="{jtype}">'
            f'<parent link="{parent}"/><child link="{child}"/>{origin_xml}{default_xml}{axis_xml}{limit}</joint>')


def _build_one_urdf(model, spec, robot: RobotSpec, floating_base: bool,
                    mesh_relative_to: pathlib.Path | None = None):
    """Emit one URDF string. floating_base=True -> faithful twin (world link + floating base,
    NOT cuRobo-loadable). False -> cuRobo view (base_link is the fixed root)."""
    from mj_envs.utils.mj_home_pose import home_qpos as _home_qpos_of

    mesh_abs = _mesh_paths(spec, relative_to=mesh_relative_to)
    home_qpos = _home_qpos_of(model)

    def meshname_of(g):
        return _name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])

    links, joints = [], []
    if floating_base:
        links.append('<link name="world"/>')

    # Body links + their parent joints. Body tree == URDF link tree (one parent each).
    for b in range(1, model.nbody):
        bname = _name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        pname = _name(model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[b])
        geoms = list(range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]))
        # g2 visual meshes only. They are not collision proxies.
        visuals = "".join(
            _visual_xml(model, g, mesh_abs, meshname_of)
            for g in geoms
            if GROUP_ROLE.get(int(model.geom_group[g])) == "fit"
        )
        group5 = [g for g in geoms if int(model.geom_group[g]) == 5]
        group3 = [g for g in geoms if int(model.geom_group[g]) == 3]
        # If simplified rack proxies (g5) exist on a body, they supersede that body's g3 geoms.
        collision_geoms = group5 if group5 else group3
        collisions = "".join(_collision_xml(model, g) for g in collision_geoms)
        links.append(f'<link name="{bname}">{_inertial_xml(model, b)}{visuals}{collisions}</link>')

        njnt_b = model.body_jntnum[b]
        origin = _origin(model.body_pos[b], model.body_quat[b])
        if njnt_b == 0:
            joints.append(_joint_xml(f"{bname}_fixed", "fixed", pname, bname, origin))
            continue
        j = model.body_jntadr[b]
        jname = _name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or bname
        jtype = model.jnt_type[j]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            if floating_base:
                joints.append(_joint_xml(jname, "floating", pname, bname, origin))
            # cuRobo: base_link is the root -> no joint emitted.
            continue
        axis = model.jnt_axis[j]
        qadr = int(model.jnt_qposadr[j])
        ref_offset = float(model.qpos0[qadr]) if qadr >= 0 else 0.0
        # The shared planner/playback seam is q_curobo = q_mujoco - qpos0. MuJoCo folds XML joint
        # ``ref`` into qpos0 (wrist_3 is ±pi/2), so raw MuJoCo limits would allow cuRobo to plan a
        # q_curobo that maps outside the physical range after playback adds the ref back. Export limits
        # and default in the same cspace coordinates as the route, not raw MuJoCo qpos.
        rng = model.jnt_range[j] - ref_offset if model.jnt_limited[j] else None
        default_val = float(home_qpos[qadr]) - ref_offset if qadr >= 0 else None
        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
            utype = "revolute" if rng is not None else "continuous"
        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            utype = "prismatic"
        else:
            raise ValueError(f"unsupported joint type {jtype} on body {bname}")
        joints.append(_joint_xml(jname, utype, pname, bname, origin, axis=axis, rng=rng, default=default_val))

    # Sites -> uniform fixed frame links, exact MuJoCo names, no special tool site (plan §F).
    for s in range(model.nsite):
        sname = _name(model, mujoco.mjtObj.mjOBJ_SITE, s)
        bname = _name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[s])
        links.append(f'<link name="{sname}"/>')
        origin = _origin(model.site_pos[s], model.site_quat[s])
        joints.append(_joint_xml(f"{sname}_frame", "fixed", bname, sname, origin))

    body = "\n".join(links + joints)
    return f'<?xml version="1.0"?>\n<robot name="{robot.name}">\n{body}\n</robot>\n', len(links), len(joints)


def _assert_joint_limits(model, urdf, robot: RobotSpec):
    """Every exported bound must use q_curobo = q_mujoco - qpos0 coordinates."""
    for j in urdf.robot.joints:
        if j.type not in ("revolute", "prismatic"):
            continue
        mj_j = mujoco.mj_id2name  # local alias unused; kept explicit below
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name)
        assert jid >= 0, f"URDF joint {j.name} absent from mj_model"
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid] - model.qpos0[qadr]
        assert abs(j.limit.lower - lo) < 1e-6 and abs(j.limit.upper - hi) < 1e-6, (
            f"{robot.name} joint {j.name} limit [{j.limit.lower},{j.limit.upper}] "
            f"!= cuRobo-coordinate MuJoCo range [{lo},{hi}]"
        )


def _assert_mesh_refs(path: pathlib.Path, expect_relative: bool) -> None:
    """URDF mesh reference gate.

    Validates the artifact in the same way a normal URDF viewer resolves it: relative
    filenames are resolved from the URDF directory. This catches VSCode's failure mode
    without generating viewer-specific mesh copies.
    """
    root = ET.parse(path).getroot()
    meshes = root.findall(".//mesh")
    assert meshes, f"{path.name}: no mesh refs found"
    bad = []
    for mesh in meshes:
        filename = mesh.attrib["filename"]
        ref = pathlib.Path(filename)
        if expect_relative and ref.is_absolute():
            bad.append(f"absolute:{filename}")
            continue
        if not expect_relative and not ref.is_absolute():
            bad.append(f"relative:{filename}")
            continue
        resolved = (path.parent / ref).resolve() if expect_relative else ref
        if resolved.suffix.lower() not in (".stl", ".obj"):
            bad.append(f"suffix:{filename}")
        if not resolved.exists():
            bad.append(f"missing:{filename}->{resolved}")
        if "scale" not in mesh.attrib:
            bad.append(f"missing-scale:{filename}")
    assert not bad, f"{path.name}: invalid mesh refs: {bad[:8]}"


def build_urdf(robot: RobotSpec, entity: Entity, live_model: mujoco.MjModel,
               out_dir: pathlib.Path) -> None:
    """Two-artifact URDF (plan §B). Both files differ ONLY in the root joint:
    ``<robot>_full.urdf`` = faithful twin (world + floating base, all joints active; NOT
    cuRobo-loadable), ``<robot>_curobo.urdf`` = fixed base (base_link root), for cuRobo.
    Mesh refs point to source files with no copy: relative paths in the faithful full URDF
    for VSCode/RViz-style viewers, absolute paths in the cuRobo URDF for RobotBuilder.

    Those absolute paths are MACHINE-LOCAL and deliberately stay that way. cuRobo is handed
    ``asset_root_path="/"``, so one root cannot cover both this repo's meshes and g1's, which live
    inside the installed ``mjlab`` package -- no relative form expresses both.
    ``tasks/visual_manipulation/curobo/urdf_localize.py`` remaps every mesh onto the current tree at
    LOAD time instead. Re-exporting on another machine rewrites these paths and is expected to; the
    loader absorbs it, so the churn is not a regression.

    Gate (hard, all asserts): full yourdfpy scene-graph FK load of BOTH files (validates the
    link/joint graph, element schema, AND that FK closes) + per-joint limit/count asserts vs
    mj_model; then cuRobo ``RobotBuilder`` load of the fixed-base file. All require yourdfpy
    >= 0.0.60 under numpy>=2 (``_ensure_yourdfpy`` auto-upgrades); 0.0.58 crashes the FK.
    Reproducibility: the version floor is pinned, not machine-dependent."""
    yourdfpy = _ensure_yourdfpy()
    from curobo._src.robot.builder.builder_robot import RobotBuilder

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = entity.spec
    results = {}
    for suffix, floating in (("full", True), ("curobo", False)):
        path = out_dir / f"{robot.name}_{suffix}.urdf"
        mesh_relative_to = out_dir if floating else None
        xml, nlinks, njoints = _build_one_urdf(
            live_model, spec, robot, floating_base=floating,
            mesh_relative_to=mesh_relative_to,
        )
        path.write_text(xml)
        # Full scene-graph load: schema + link/joint graph + FK closure (build_scene_graph default).
        urdf = yourdfpy.URDF.load(str(path))
        _assert_joint_limits(live_model, urdf, robot)
        _assert_mesh_refs(path, expect_relative=floating)
        print(f"[urdf] wrote {path}: {nlinks} links / {njoints} joints "
              f"(yourdfpy scene-graph FK OK, joint limits match mj_model)")
        results[suffix] = (path, nlinks, njoints)

    # cuRobo load: the real FK gate. Hard-assert now that the env is fixed.
    curobo_path = results["curobo"][0]
    RobotBuilder(str(curobo_path), asset_path="/", tool_frames=list(robot.tool_sites))
    print(f"[urdf] cuRobo RobotBuilder loaded {curobo_path.name} (tool_frames={robot.tool_sites})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", choices=list(ROBOTS), required=True)
    ap.add_argument("--format", choices=("mjcf", "urdf", "both"), default="mjcf")
    ap.add_argument("--head-camera", default="actuated")
    ap.add_argument("--end-effector", default="actuated")
    ap.add_argument("--hand", default="parallel_gripper")
    ap.add_argument("--output", default=None,
                    help="override the robot's own out_dir (default: RobotSpec.out_dir)")
    args = ap.parse_args()

    robot = ROBOTS[args.robot]
    out_dir = pathlib.Path(args.output) if args.output else robot.out_dir
    _force_wireframe_for_preflight()
    entity = build_variant(robot, args.head_camera, args.end_effector, args.hand)
    n_hull = strip_fov_hull(entity.spec)
    if n_hull:
        print(f"[variant] {robot.name}: dropped {n_hull} viewer-only FOV hull geom(s) before export")
    live_model = preflight(robot, entity, _HUMANOID_V21_RIG_SNAPSHOTS.get(args.head_camera))

    if args.format in ("mjcf", "both"):
        build_mjcf(robot, entity, live_model, out_dir)
    if args.format in ("urdf", "both"):
        build_urdf(robot, entity, live_model, out_dir)


if __name__ == "__main__":
    main()
