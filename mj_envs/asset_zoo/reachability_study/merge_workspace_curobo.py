"""Merge chunked cuRobo workspace sidecars.

`generate_workspace_curobo.py --target-start` lets large sweeps run as smaller
processes when a single large process hits native allocator/CUDA crashes. This
script concatenates those chunks into one sidecar with the same schema expected
by `plot_workspace_curobo.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


_CAT_KEYS = {
    "target_pos",
    "target_quat",
    "seed_indices",
    "seed_candidate_indices",
    "winning_seed_indices",
    "attempt_count",
    "success",
    "q_solution",
    "pos_err",
    "rot_err",
    "solve_time",
    "sigma_trans",
    "sigma_rot",
    "manip_trans",
    "manip_rot",
    "sigma_min_trans",
    "sigma_min_rot",
    "voxel_index",
    "orient_index",
    "voxel_center_pos",
    "seed_target_distance",
}

_CHECK_KEYS = {
    "robot",
    "model",
    "side",
    "source",
    "joint_names",
    "stride",
    "n_grid",
    "n_orientations",
    "orientation_source",
    "seed_attempts",
    "target_alpha",
    "pad",
    "orientation_seed_weight",
    # isotropic spacing / hierarchical grid metadata: all shards of one side must agree, else
    # voxel_index means different things per shard and the merged lattice is corrupt.
    "grid_spacing",
    "n_grid_per_axis",
    "grid_origin",
    "grid_lo",
    "grid_hi",
    "position_gate",
    "position_gate_voxels",
    "position_gate_total_voxels",
    "position_gate_halo",
    "position_gate_coarse_workspace",
    "levels",
    "grid_bounds_source",
}


def _same(a: object, b: object) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    return a == b


def merge(inputs: list[Path], out: Path) -> None:
    if len(inputs) < 2:
        raise ValueError("need at least two input chunks")
    chunks = [torch.load(str(path), map_location="cpu", weights_only=False) for path in inputs]
    first = chunks[0]
    for chunk in chunks[1:]:
        for key in _CHECK_KEYS:
            if key in first and key in chunk and not _same(first[key], chunk[key]):
                raise ValueError(f"chunk metadata mismatch for {key}: {first[key]} != {chunk[key]}")

    merged = {}
    for key, value in first.items():
        if key in _CAT_KEYS and all(key in chunk for chunk in chunks):
            merged[key] = torch.cat([chunk[key] for chunk in chunks], dim=0)
        else:
            merged[key] = value

    merged["limit"] = int(merged["target_pos"].shape[0])
    merged["target_start"] = int(min(int(chunk.get("target_start", 0)) for chunk in chunks))
    merged["chunk_files"] = [str(path) for path in inputs]
    merged["chunk_limits"] = [int(chunk["target_pos"].shape[0]) for chunk in chunks]
    merged["chunk_target_starts"] = [int(chunk.get("target_start", 0)) for chunk in chunks]

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, str(out))
    print(f"saved {out}")
    print(f"success {int(merged['success'].sum())}/{merged['success'].numel()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()
