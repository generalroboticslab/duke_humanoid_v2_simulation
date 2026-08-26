"""Unified robot model viewer.

Supports:
1. Live MuJoCo passive viewer for resolved MJCF XML.
2. Interactive trimesh scene viewer for URDF files.
3. Headless GLB export for URDF files (useful in environments without display/OpenGL).

Run examples:
    # Live MuJoCo viewer for humanoid resolved XML
    python asset/create/view_robot.py --robot humanoid_v21 --type mjcf --mode live

    # Interactive trimesh viewer for G1 URDF
    python asset/create/view_robot.py --robot g1 --type curobo --mode interactive

    # Headless GLB export for humanoid URDF
    python asset/create/view_robot.py --robot humanoid_v21 --type full --mode glb --out humanoid_view.glb
"""

import argparse
import pathlib
import sys
import time

import numpy as np

# Path configurations
_HERE = pathlib.Path(__file__).resolve().parent


def _artifact_dir(robot: str) -> pathlib.Path:
    """Where ``robot``'s exported artifacts live, read from the exporter's own ROBOTS table.

    Single source of truth: the viewer must look wherever ``export_mjspec_to_urdf.py`` writes,
    and the two used to agree only because both hardcoded ``asset/create``. They no longer do
    (artifacts now sit in each robot's ``asset/<robot>/``), so the viewer defers to the table."""
    sys.path.insert(0, str(_HERE))
    from export_mjspec_to_urdf import ROBOTS

    return ROBOTS[robot].out_dir


def _fk_numpy2_safe(self, joint, q=None):
    """yourdfpy 0.0.58/0.0.60 forward kinematics shim for NumPy 2 compatibility.
    Coerces q to a python scalar to prevent float/rotation matrix errors.
    """
    import trimesh.transformations as tra
    origin = joint.origin if joint.origin is not None else np.eye(4)
    if joint.mimic is not None and q is None:
        if joint.mimic.joint in self.actuated_joint_names:
            mi = self.actuated_joint_names.index(joint.mimic.joint)
            q = self.cfg[mi] * joint.mimic.multiplier + joint.mimic.offset
    if joint.type in ("revolute", "prismatic", "continuous"):
        if q is None:
            q = self.cfg[self.actuated_dof_indices[self.actuated_joint_names.index(joint.name)]]
        q = float(np.asarray(q).reshape(-1)[0])
        if joint.type == "prismatic":
            matrix = origin @ tra.translation_matrix(q * np.asarray(joint.axis))
        else:
            matrix = origin @ tra.rotation_matrix(q, joint.axis)
    else:  # fixed / floating / planar
        matrix = origin
    return matrix, q


def run_live_mjcf(robot: str):
    """Launches the passive MuJoCo viewer to show the resolved XML model."""
    import mujoco
    from mujoco import viewer

    xml_path = _artifact_dir(robot) / f"{robot}_resolved.xml"
    if not xml_path.exists():
        print(f"Error: Resolved XML not found: {xml_path}")
        print(f"Please run: python asset/create/export_mjspec_to_urdf.py --robot {robot} --format mjcf")
        sys.exit(1)

    print(f"Loading MuJoCo model: {xml_path.name}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    print(f"Opening passive viewer: nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh}")
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(0.02)


def run_urdf(robot: str, urdf_type: str, mode: str, custom_urdf: str = None, out_path: str = None):
    """Loads a URDF and views it interactively or exports it to GLB."""
    import yourdfpy
    import yourdfpy.urdf as YU

    # Apply NumPy 2 safety patch to yourdfpy
    YU.URDF._forward_kinematics_joint = _fk_numpy2_safe

    # Resolve URDF path
    if custom_urdf:
        urdf_path = pathlib.Path(custom_urdf)
    else:
        urdf_path = _artifact_dir(robot) / f"{robot}_{urdf_type}.urdf"

    if not urdf_path.exists():
        print(f"Error: URDF not found: {urdf_path}")
        if not custom_urdf:
            print(f"Please run: python asset/create/export_mjspec_to_urdf.py --robot {robot} --format urdf")
        sys.exit(1)

    print(f"Loading URDF: {urdf_path.name}")
    urdf = yourdfpy.URDF.load(str(urdf_path))
    print(f"Loaded successfully: {len(urdf.link_map)} links, {len(urdf.joint_map)} joints")

    scene = urdf.scene
    print(f"Scene graph built: {len(scene.geometry)} meshes (posed at home config)")

    if mode == "glb":
        # Headless GLB export
        if not out_path:
            out_file = _artifact_dir(robot) / f"{robot}_{urdf_type}_home.glb" if not custom_urdf else urdf_path.with_suffix(".glb")
        else:
            out_file = pathlib.Path(out_path)

        scene.export(str(out_file))
        print(f"Wrote GLB to: {out_file}")
        print("You can view this file in Blender, MeshLab, or any online 3D viewer.")

    elif mode == "interactive":
        # Interactive trimesh viewer window
        print("Opening interactive window via trimesh Scene.show()...")
        try:
            scene.show()
        except Exception as ex:
            print(f"Interactive viewer unavailable ({type(ex).__name__}: {ex})")
            print("To verify coordinates, you can export to GLB instead: use --mode glb")


def main():
    ap = argparse.ArgumentParser(description="Unified robot model viewer")
    ap.add_argument("--robot", default="humanoid_v21", choices=("humanoid_v21", "g1"),
                    help="Robot variant (default: humanoid_v21)")
    ap.add_argument("--type", default="curobo", choices=("mjcf", "full", "curobo"),
                    help="Model type to load: resolved MJCF xml, full URDF, or curobo URDF (default: curobo)")
    ap.add_argument("--mode", default="interactive", choices=("live", "interactive", "glb"),
                    help="Viewer mode (default: interactive)")
    ap.add_argument("--urdf", default=None,
                    help="Optional custom path to a URDF file (bypasses --robot and --type)")
    ap.add_argument("--out", default=None,
                    help="Optional custom output path for GLB export (used with --mode glb)")
    args = ap.parse_args()

    # Input validation
    if args.type == "mjcf" and args.mode != "live":
        print("Error: --type mjcf requires --mode live")
        sys.exit(1)
    if args.type != "mjcf" and args.mode == "live":
        print("Error: --mode live requires --type mjcf")
        sys.exit(1)
    if args.urdf and args.type == "mjcf":
        print("Error: Custom --urdf path cannot be used with --type mjcf")
        sys.exit(1)

    if args.type == "mjcf":
        run_live_mjcf(args.robot)
    else:
        run_urdf(
            robot=args.robot,
            urdf_type=args.type,
            mode=args.mode,
            custom_urdf=args.urdf,
            out_path=args.out
        )


if __name__ == "__main__":
    main()
