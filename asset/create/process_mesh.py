"""
Mesh simplification pipeline for robot visual meshes.

Source OBJ files are CAD assembly exports: many sub-bodies, non-uniform triangulation,
open-shell geometry (not watertight). Pipeline produces two outputs per mesh:

  {output_dir}/{stem}_high_res.obj  — all survivor bodies concatenated, full resolution
  {output_dir}/{stem}.obj           — simplified version for simulation visuals

Pipeline per file:
  1. Load: extract all Trimesh bodies from OBJ scene.
  2. Size filter: drop bodies whose bbox diagonal < small_frac * max diagonal.
  3. Occlusion filter: drop bodies whose surface samples all land inside another body
     (ray-cast contains(), no convex hull — open-shell safe).
  4. Save high_res: concatenate survivors as-is.
  5. Simplify per body (isotropic remesh → QEC), then concatenate.
     Per-body isolation prevents QEC from merging vertices across body seams.
     Isotropic remesh first redistributes the non-uniform CAD triangulation uniformly,
     eliminating the cone/sharp-edge artifacts that QEC produces on clustered tiny triangles.
  6. Save simplified.

When called directly:
  - source is a directory → process all .obj files in parallel (hash-based skip).
  - source is a single file → process that file only.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
import tyro


def _bbox_diag(mesh: trimesh.Trimesh) -> float:
    return float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))


def _remove_small(bodies: list[trimesh.Trimesh], small_frac: float) -> list[trimesh.Trimesh]:
    max_diag = max(_bbox_diag(b) for b in bodies)
    threshold = small_frac * max_diag
    kept = [b for b in bodies if _bbox_diag(b) >= threshold]
    dropped = len(bodies) - len(kept)
    if dropped:
        print(f"  [small] dropped {dropped}/{len(bodies)} bodies (diag < {threshold:.2f})")
    return kept


def _remove_occluded(bodies: list[trimesh.Trimesh], n_samples: int) -> list[trimesh.Trimesh]:
    # Ray-cast contains() on the actual mesh — no convex hull fallback.
    # Convex hull of a large-extent open-shell body over-extends massively, classifying
    # external sub-features as interior. Direct ray-cast is tight even on non-watertight meshes.
    survivors = []
    for i, body in enumerate(bodies):
        pts, _ = trimesh.sample.sample_surface(body, n_samples)
        occluded = False
        for j, other in enumerate(bodies):
            if i == j:
                continue
            try:
                inside = other.contains(pts)
            except Exception:
                continue
            if inside.all():
                occluded = True
                break
        if not occluded:
            survivors.append(body)
    dropped = len(bodies) - len(survivors)
    if dropped:
        print(f"  [occlude] dropped {dropped}/{len(bodies)} occluded bodies")
    return survivors



def _qec_single(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Simplify one body: isotropic remesh → QEC.

    Direct QEC on CAD exports produces cones and sharp edges because CAD triangulation
    is non-uniform (tiny triangles cluster on curves, large on flats). QEC collapses
    clustered tiny triangles into angular peaks, and open-shell boundary vertices have
    under-constrained quadrics that collapse into cones.

    Fix: isotropic remesh first (uniform triangles, reprojected to original surface),
    then QEC on the clean mesh. featuredeg=30 preserves intentional sharp features.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))

    # Isotropic remesh to ~2× target so QEC has headroom. Edge length from equilateral
    # triangle area: A = (sqrt(3)/4)*l^2 → l = sqrt(4*A / (N*sqrt(3))).
    remesh_target = max(4, target_faces * 2)
    targetlen = float(np.sqrt(4.0 * mesh.area / (remesh_target * np.sqrt(3))))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=5,
        targetlen=pymeshlab.PureValue(targetlen),
        featuredeg=30.0,   # dihedral threshold for sharp-feature preservation
        reprojectflag=True,  # snap remeshed verts back to original surface
    )

    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        qualitythr=0.3,
        autoclean=True,
    )
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


def _simplify_bodies(bodies: list[trimesh.Trimesh], total_target: int) -> list[trimesh.Trimesh]:
    """Simplify each body to a proportional face target, return simplified list.

    Per-body isolation prevents QEC from merging vertices across body seams.
    Concatenating open-shell bodies creates coincident boundary vertices; running QEC
    on the merged mesh welds them, collapsing the seam topology and destroying geometry.
    """
    total_faces = sum(len(b.faces) for b in bodies)
    results = []
    total_before = 0
    total_after = 0
    for body in bodies:
        frac = len(body.faces) / total_faces
        target = max(4, int(total_target * frac))
        simplified = _qec_single(body, target)
        total_before += len(body.faces)
        total_after += len(simplified.faces)
        results.append(simplified)
    print(f"  [simplify] {total_before} → {total_after} faces (target {total_target})")
    return results


def process_mesh(
    input_obj: str,
    output_stem: str,
    output_dir: str = "asset/duke_v2/humanoid_v21/meshes",
    small_frac: float = 0.05,
    occlusion_samples: int = 50,
    simplify_threshold: int = 10000,
    simplify_ratio: float = 0.5,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Process one multi-body OBJ: filter bodies, save high_res, simplify, save simplified.

    Saves:
      {output_dir}/{output_stem}_high_res.obj  — survivors concatenated, full resolution
      {output_dir}/{output_stem}.obj           — per-body isotropic remesh + QEC

    Args:
        input_obj: Path to source OBJ (CAD export, may have many sub-bodies).
        output_stem: Base name for outputs (e.g. "elbow_long").
        output_dir: Directory to write outputs.
        small_frac: Drop bodies whose bbox diagonal < small_frac * max diagonal.
        occlusion_samples: Surface samples per body for ray-cast occlusion test.
        simplify_threshold: Skip simplification if total survivor faces already below this.
        simplify_ratio: Simplified target = simplify_ratio * original total face count.

    Returns:
        (high_res_mesh, simplified_mesh)
    """
    print(f"\n=== {os.path.basename(input_obj)} ===")

    # 1. Load
    scene = trimesh.load(input_obj)
    if isinstance(scene, trimesh.Scene):
        bodies = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    else:
        bodies = [scene]
    print(f"  [load] {len(bodies)} bodies, {sum(len(b.faces) for b in bodies)} total faces")

    total_input_faces = sum(len(b.faces) for b in bodies)

    # 2. Remove small
    bodies = _remove_small(bodies, small_frac)

    # 3. Remove occluded
    if len(bodies) > 1:
        bodies = _remove_occluded(bodies, occlusion_samples)

    # 4. Save high_res: survivors concatenated, no vertex welding across bodies.
    os.makedirs(output_dir, exist_ok=True)
    exterior = trimesh.util.concatenate(bodies) if len(bodies) > 1 else bodies[0]
    high_res_path = os.path.join(output_dir, f"{output_stem}_high_res.obj")
    exterior.export(high_res_path)
    print(f"  [save] high_res → {high_res_path} ({len(exterior.faces)} faces)")

    # 5. Simplify each body independently then concatenate.
    # target_faces is fraction of original total (pre-filter) so ratio is consistent across runs.
    target_faces = max(100, int(total_input_faces * simplify_ratio))
    total_survivor_faces = sum(len(b.faces) for b in bodies)
    if total_survivor_faces > simplify_threshold:
        simplified_bodies = _simplify_bodies(bodies, target_faces)
        simplified = trimesh.util.concatenate(simplified_bodies) if len(simplified_bodies) > 1 else simplified_bodies[0]
    else:
        simplified = exterior
        print(f"  [simplify] skipped ({total_survivor_faces} faces ≤ threshold {simplify_threshold})")

    # 6. Save simplified
    out_path = os.path.join(output_dir, f"{output_stem}.obj")
    simplified.export(out_path)
    print(f"  [save] simplified → {out_path} ({len(simplified.faces)} faces)")

    return exterior, simplified


_HASH_CACHE_FILE = ".mesh_hashes.json"


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_hashes(output_dir: str) -> dict[str, str]:
    cache = os.path.join(output_dir, _HASH_CACHE_FILE)
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    return {}


def _save_hashes(output_dir: str, hashes: dict[str, str]) -> None:
    cache = os.path.join(output_dir, _HASH_CACHE_FILE)
    with open(cache, "w") as f:
        json.dump(hashes, f, indent=2)


def _process_one(
    obj_file: str,
    output_dir: str,
    small_frac: float,
    occlusion_samples: int,
    simplify_threshold: int,
    simplify_ratio: float,
) -> None:
    """Worker: process a single OBJ file. Module-level so it's picklable."""
    process_mesh(
        input_obj=obj_file,
        output_stem=Path(obj_file).stem,
        output_dir=output_dir,
        small_frac=small_frac,
        occlusion_samples=occlusion_samples,
        simplify_threshold=simplify_threshold,
        simplify_ratio=simplify_ratio,
    )


def process_all(
    source_dir: str,
    output_dir: str = "asset/duke_v2/humanoid_v21/meshes",
    small_frac: float = 0.05,
    occlusion_samples: int = 50,
    simplify_threshold: int = 10000,
    simplify_ratio: float = 0.5,
    force: bool = False,
    n_workers: int = -1,
) -> None:
    """Process all .obj files in source_dir in parallel.

    Skips files whose SHA-256 hash matches the cache in {output_dir}/.mesh_hashes.json.
    Hash cache is written in the main process after each worker completes — no race condition.
    """
    source_path = Path(source_dir)
    obj_files = sorted(source_path.glob("*.obj"))
    if not obj_files:
        print(f"No .obj files found in {source_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    hashes = _load_hashes(output_dir)

    # Pre-compute hashes and filter; skip check happens before spawning workers.
    to_process = []
    current_hashes: dict[str, str] = {}
    for obj_file in obj_files:
        h = _file_hash(str(obj_file))
        current_hashes[str(obj_file)] = h
        if not force and hashes.get(str(obj_file)) == h:
            print(f"=== {obj_file.name} — unchanged, skipping ===")
        else:
            to_process.append(obj_file)

    if not to_process:
        return

    workers = os.cpu_count() if n_workers == -1 else n_workers
    kwargs = dict(
        output_dir=output_dir,
        small_frac=small_frac,
        occlusion_samples=occlusion_samples,
        simplify_threshold=simplify_threshold,
        simplify_ratio=simplify_ratio,
    )

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_one, str(f), **kwargs): f for f in to_process}
        for future in as_completed(futures):
            obj_file = futures[future]
            future.result()  # propagate exceptions
            hashes[str(obj_file)] = current_hashes[str(obj_file)]
            _save_hashes(output_dir, hashes)  # written in main process — no race condition


@dataclass
class Args:
    source: str = "asset/duke_v2/humanoid_v21/meshes/source"
    """Source: a directory of .obj files (batch, parallel) or a single .obj file."""
    output_dir: str = "asset/duke_v2/humanoid_v21/meshes"
    """Directory to write processed outputs."""
    small_frac: float = 0.05
    """Drop bodies with bbox diagonal < this fraction of the largest body's diagonal."""
    occlusion_samples: int = 50
    """Number of surface samples per body for occlusion detection."""
    simplify_threshold: int = 10000
    """Skip simplification if face count already below this value."""
    simplify_ratio: float = 0.5
    """Simplification target = ratio * total input face count."""
    force: bool = False
    """Reprocess even if source file is unchanged (directory mode only)."""
    n_workers: int = -1
    """Number of parallel workers (-1 = cpu_count, directory mode only)."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    source = Path(args.source)
    if source.is_dir():
        process_all(
            source_dir=str(source),
            output_dir=args.output_dir,
            small_frac=args.small_frac,
            occlusion_samples=args.occlusion_samples,
            simplify_threshold=args.simplify_threshold,
            simplify_ratio=args.simplify_ratio,
            force=args.force,
            n_workers=args.n_workers,
        )
    elif source.is_file():
        process_mesh(
            input_obj=str(source),
            output_stem=source.stem,
            output_dir=args.output_dir,
            small_frac=args.small_frac,
            occlusion_samples=args.occlusion_samples,
            simplify_threshold=args.simplify_threshold,
            simplify_ratio=args.simplify_ratio,
        )
    else:
        raise FileNotFoundError(f"source not found: {source}")
