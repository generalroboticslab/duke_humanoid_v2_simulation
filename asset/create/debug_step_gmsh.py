#!/usr/bin/env python3
"""Minimal STEP→OBJ via gmsh — debug only.

Uses gmsh's Frontal-Delaunay mesher (not OCCT BRepMesh_IncrementalMesh).
Diagnostic: if thin ring appears here but not in process_stp.py, BRepMesh is culprit.

Usage:
    python debug_step_gmsh.py wrist_1.step [out.obj] [--mesh-size 2.0] [--min-volume 1000]
"""

import argparse
from pathlib import Path

import gmsh


def step_to_obj(
    step_path: str,
    obj_path: str,
    mesh_size_max: float,
    mesh_size_min: float,
    min_volume: float,
) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(3)
    surfaces = gmsh.model.getEntities(2)
    print(f"  STEP entities: {len(volumes)} volumes, {len(surfaces)} surfaces", flush=True)

    if min_volume > 0 and volumes:
        to_remove = []
        for dim, tag in volumes:
            vol = gmsh.model.occ.getMass(dim, tag)
            if vol < min_volume:
                to_remove.append((dim, tag))
        if to_remove:
            gmsh.model.occ.remove(to_remove, recursive=True)
            gmsh.model.occ.synchronize()
            remaining = gmsh.model.getEntities(3)
            print(
                f"  removed {len(to_remove)}/{len(volumes)} small volumes "
                f"(< {min_volume} mm³), {len(remaining)} remain",
                flush=True,
            )

    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.Algorithm", 6)       # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    all_elems = gmsh.model.mesh.getElements(dim=2)
    n_tris = sum(
        len(all_elems[1][i]) for i, et in enumerate(all_elems[0]) if et == 2
    )
    print(f"  mesh: {n_nodes} nodes, {n_tris} triangles", flush=True)

    gmsh.write(obj_path)
    print(f"  written: {obj_path}", flush=True)
    gmsh.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug STEP→OBJ via gmsh")
    parser.add_argument("step", help="Input STEP file")
    parser.add_argument("obj", nargs="?", help="Output OBJ (default: <stem>.obj in same dir)")
    parser.add_argument("--mesh-size", type=float, default=10.0,
                        help="Max mesh element size in mm (default: 2.0)")
    parser.add_argument("--mesh-size-min", type=float, default=0.2,
                        help="Min mesh element size in mm (default: 0.1)")
    parser.add_argument("--min-volume", type=float, default=1000.0,
                        help="Drop volumes smaller than this mm³ (default: 1000, 0=keep all)")
    args = parser.parse_args()

    step_path = Path(args.step).resolve()
    obj_path = args.obj or str(step_path.with_suffix(".obj"))

    step_to_obj(str(step_path), str(obj_path), args.mesh_size, args.mesh_size_min, args.min_volume)


if __name__ == "__main__":
    main()
