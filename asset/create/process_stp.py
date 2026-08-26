from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh


@dataclass
class MeshConfig:
    """Tessellation parameters for one resolution pass (OCC BRepMesh and gmsh).

    linear_defl:     max chord deviation in mm (OCC) / gmsh MeshSizeMax — lower = finer mesh.
    angular_defl:    max angular deviation between adjacent triangles in radians (OCC) /
                     gmsh MinimumCircleNodes derived as round(2π / angular_defl) — lower = smoother curves.
    label:           display name used in progress output.
    """
    linear_defl: float
    angular_defl: float
    label: str


HIGH_RES = MeshConfig(linear_defl=0.5, angular_defl=0.2, label="high_res")
LOW_RES  = MeshConfig(linear_defl=1.0, angular_defl=0.4, label="low_res")


@contextmanager
def timed_stage(label):
    start = time.monotonic()
    print(f"{label}...", flush=True)
    try:
        yield
    finally:
        print(f"{label}: {time.monotonic() - start:.2f}s", flush=True)


def _tessellate_solids(solids, cfg: MeshConfig) -> trimesh.Trimesh:
    """Tessellate list of cadquery Solid objects via OCC BRepMesh, return merged trimesh.

    Uses OCC's BRepMesh_IncrementalMesh which natively handles all analytic surface types
    (cones, spheres, tori) including those with periodic UV parametrization that gmsh
    cannot mesh.
    """
    all_verts, all_faces = [], []
    offset = 0
    for solid in solids:
        verts, tris = solid.tessellate(cfg.linear_defl, angularTolerance=cfg.angular_defl)
        if not verts:
            continue
        v = np.array([(p.x, p.y, p.z) for p in verts], dtype=np.float64)
        f = np.array(tris, dtype=np.int64) + offset
        all_verts.append(v)
        all_faces.append(f)
        offset += len(verts)
    if not all_verts:
        raise RuntimeError(f"tessellation produced no geometry")
    mesh = trimesh.Trimesh(
        vertices=np.vstack(all_verts), faces=np.vstack(all_faces), process=False
    )
    mesh.merge_vertices()
    mesh.fix_normals()
    print(f"  {cfg.label}: {len(mesh.faces)} faces", flush=True)
    return mesh


def _extract_mesh_from_gmsh(cfg: MeshConfig) -> trimesh.Trimesh:
    import gmsh
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    vertices = coords.reshape(-1, 3)
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    faces = []
    for etype, entags in zip(elem_types, elem_node_tags):
        if etype == 2:  # 3-node triangle
            tris = entags.reshape(-1, 3)
            faces.append(np.array([[tag_to_idx[t] for t in tri] for tri in tris], dtype=np.int64))
    if not faces:
        raise RuntimeError("gmsh produced no triangles")
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces), process=False)
    mesh.merge_vertices()
    mesh.fix_normals()
    print(f"  {cfg.label} (gmsh): {len(mesh.faces)} faces", flush=True)
    return mesh


def _gmsh_worker(step_path_str: str, cfg: MeshConfig) -> trimesh.Trimesh:
    """Open own gmsh session, mesh STEP at cfg resolution, return trimesh.

    Module-level so ProcessPoolExecutor can pickle it.
    """
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(step_path_str)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", cfg.linear_defl*20)
        gmsh.option.setNumber("Mesh.MeshSizeMin", cfg.linear_defl)
        nodes_per_circle = max(12, int(np.round(2 * np.pi / cfg.angular_defl)))
        gmsh.option.setNumber("Mesh.MinimumCircleNodes", nodes_per_circle)
        gmsh.model.mesh.generate(2)
        return _extract_mesh_from_gmsh(cfg)
    finally:
        gmsh.finalize()


def _tessellate_gmsh(step_path, high_cfg: MeshConfig, low_cfg: MeshConfig):
    """Tessellate STEP → (high_res, low_res) via gmsh Frontal-Delaunay mesher.

    High-res and low-res run in parallel via two worker processes — gmsh has process-global
    state and cannot share a single session across threads.
    Raises on any failure; caller falls back to OCC BRepMesh.
    linear_defl → MeshSizeMax, linear_defl_min → MeshSizeMin, angular_defl → MinimumCircleNodes.
    """
    step_path_str = str(step_path)
    with ProcessPoolExecutor(max_workers=2) as pool:
        f_high = pool.submit(_gmsh_worker, step_path_str, high_cfg)
        f_low  = pool.submit(_gmsh_worker, step_path_str, low_cfg)
        high_res = f_high.result()
        low_res  = f_low.result()
    return high_res, low_res


def tessellate_step(step_path, high_cfg=HIGH_RES, low_cfg=LOW_RES, min_solid_volume=None):
    """Tessellate STEP file → (high_res, low_res) trimesh.Trimesh pair.

    Filters small solids before meshing so the filter applies to both the gmsh and OCC
    paths.  Tries gmsh Frontal-Delaunay first (better quality); falls back to cadquery
    + OCC BRepMesh on any error.  OCC handles surface types gmsh cannot mesh (cones,
    spheres — periodic UV parametrization).

    min_solid_volume: filter solids whose bounding-box volume (mm³) is below this threshold.
      Note: bounding-box approximation overestimates volume for curved/hollow bodies.
    """
    # Load and filter solids upfront so both gmsh and OCC paths see the same geometry.
    cq_result = cq.importers.importStep(str(step_path))
    all_solids = cq_result.solids().vals()
    n_before = len(all_solids)

    if min_solid_volume:
        def _bbox_vol(s):
            bb = s.BoundingBox()
            return (bb.xmax - bb.xmin) * (bb.ymax - bb.ymin) * (bb.zmax - bb.zmin)
        solids = [s for s in all_solids if _bbox_vol(s) >= min_solid_volume]
        if len(solids) < n_before:
            print(f"  dropped {n_before - len(solids)}/{n_before} small solids "
                  f"(bbox_vol < {min_solid_volume} mm³), {len(solids)} remain", flush=True)
    else:
        solids = all_solids

    # gmsh needs a STEP file; write filtered temp STEP when solids were dropped.
    temp_path = None
    if len(solids) < n_before:
        compound = cq.Compound.makeCompound(solids)
        fd, tmp = tempfile.mkstemp(suffix=".step")
        os.close(fd)
        temp_path = Path(tmp)
        cq.exporters.export(compound, str(temp_path))

    gmsh_input = temp_path if temp_path else step_path

    try:
        return _tessellate_gmsh(gmsh_input, high_cfg, low_cfg)
    except Exception as e:
        print(f"  gmsh failed ({e}), falling back to OCC BRepMesh", flush=True)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()

    # Run coarse first: OCC caches tessellation on shape topology and a coarser request
    # after a finer one returns the cached fine mesh; extracting coarse to numpy first,
    # then running fine, forces the recompute without affecting extracted data.
    low_res  = _tessellate_solids(solids, low_cfg)
    high_res = _tessellate_solids(solids, high_cfg)
    return high_res, low_res


def write_obj(vertices, faces, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}\n" for x, y, z in vertices)
    lines.extend(f"f {a+1} {b+1} {c+1}\n" for a, b, c in faces)
    path.write_text("".join(lines))


def process_step(input_path, output_stem, output_dir, *, min_solid_volume=1000,
                 high_cfg=HIGH_RES, low_cfg=LOW_RES):
    """Convert one STEP file to high_res OBJ + low_res OBJ.

    Pipeline:
      1. Import STEP via cadquery (OCC kernel).
      2. Filter small solids by bounding-box volume approximation.
      3. Tessellate at high_cfg → {stem}_high_res.obj
      4. Tessellate at low_cfg  → {stem}.obj
    """
    output_dir = Path(output_dir)

    with timed_stage(f"{input_path} tessellate"):
        high_res, low_res = tessellate_step(
            input_path,
            high_cfg=high_cfg,
            low_cfg=low_cfg,
            min_solid_volume=min_solid_volume,
        )

    high_res_path = output_dir / f"{output_stem}_high_res.obj"
    with timed_stage(f"{input_path} write high_res"):
        write_obj(high_res.vertices, high_res.faces, high_res_path)

    low_res_path = output_dir / f"{output_stem}.obj"
    with timed_stage(f"{input_path} write"):
        write_obj(low_res.vertices, low_res.faces, low_res_path)

    high_res_mb = high_res_path.stat().st_size / 1e6
    low_res_mb = low_res_path.stat().st_size / 1e6
    print(
        f"{input_path} → {high_res_path} ({len(high_res.faces)}f, {high_res_mb:.1f}MB)"
        f" | {low_res_path} ({len(low_res.faces)}f, {low_res_mb:.1f}MB)",
        flush=True,
    )
    if high_res_mb > 20:
        print(f"  WARNING: high_res exceeds 20MB — increase linear_defl or angular_defl",
              flush=True)


def process_directory(source, output_dir):
    step_paths = sorted(source.glob("*.step")) + sorted(source.glob("*.stp"))

    def run_one(step_path):
        command = [sys.executable, str(Path(__file__)), str(step_path),
                   "--output_dir", str(output_dir)]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.returncode != 0 and result.stderr:
            print(result.stderr, end="", flush=True)
        return step_path, result.returncode

    workers = min(len(step_paths), max(1, (os.cpu_count() or 4) * 3 // 4))
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, p): p for p in step_paths}
        for fut in as_completed(futs):
            path, rc = fut.result()
            if rc != 0:
                print(f"{path}: FAILED with exit code {rc}", flush=True)
                failures.append(str(path))

    if failures:
        raise RuntimeError(f"failed to process {len(failures)} STEP files: {', '.join(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="asset/duke_v2/humanoid_v21/meshes/source_stp")
    parser.add_argument("--output_dir", default="asset/duke_v2/humanoid_v21/meshes")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)

    if source.is_file():
        process_step(source, source.stem, output_dir)
    elif source.is_dir():
        process_directory(source, output_dir)
    else:
        raise FileNotFoundError(source)
