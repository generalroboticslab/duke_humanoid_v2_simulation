"""Ship-sized cache artifacts: store the minimum, rebuild the rest on first use.

Why
---
Three cache payloads are too large to distribute in a git repository (GitHub hard-rejects any
blob over 100 MB), and regenerating them from scratch is expensive enough that dropping them is
not an option either. Both facts have the same cause: each file stores a *derived* structure
alongside the source data it was derived from.

    arm_collision_graph_<robot>.pt   193 MB   99.7% of it is `next_hop`, the dense N x N
                                              all-pairs routing table -- the transitive closure
                                              of `edges`, which is only ~0.5 MB.
    safe_arm_poses_<robot>.pt        232 MB   `poses` is float32 at 2M rows; the end-effector
                                              columns are forward kinematics of `poses`.

So the slim form keeps the graph and drops its closure, and keeps joint angles at the precision
they are actually used at. `expand()` reconstitutes the original dict on first load and caches
the result next to the slim file, so the cost is paid once per clone rather than per run.

Measured, on the shipped humanoid_v21 graph (9,990 nodes / 131,700 edges):
  * slim artifact          1.07 MB   against 193 MB dense        (178x)
  * next_hop rebuild       ~30 s     single-core scipy BFS
The expensive half of a true regeneration -- MuJoCo collision-checking every kNN edge -- is
preserved in `edges` and never re-run.

Precision, measured rather than assumed
---------------------------------------
`poses` are joint angles in radians over [-2.80, +2.38]; float16 round-trip error is at most
9.8e-4 rad = 0.056 deg, which is far below actuator resolution. Indices are a different matter:
`next_hop` addresses up to 9,989 and float16 cannot represent integers above 2048, so a float16
round-trip corrupts 27.3% of its entries. Index arrays stay integral here and are stored at the
narrowest integer type that holds them -- never a float.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

SLIM_SUFFIX = ".slim.pt"


def _slim_path(full_path: Path) -> Path:
    return full_path.with_name(full_path.name.replace(".pt", SLIM_SUFFIX))


def _min_int_dtype(max_value: int):
    """Narrowest signed/unsigned integer numpy dtype that holds `max_value`."""
    for dt in (np.uint8, np.int16, np.uint16, np.int32):
        if max_value <= np.iinfo(dt).max:
            return dt
    return np.int64


def pack_collision_graph(src: Path, dst: Path, repo_root: Path) -> None:
    """Write the slim form of an arm collision graph: keep `edges`, drop `next_hop`."""
    d = torch.load(str(src), map_location="cpu", weights_only=False)
    edges = np.asarray(d["edges"])
    slim = {
        "poses": d["poses"],
        "edges": torch.from_numpy(edges.astype(_min_int_dtype(int(edges.max())))),
        "joint_names": d["joint_names"],
        "n_poses": d["n_poses"],
        "k": d["k"],
        "method": d["method"],
        "robot": d["robot"],
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, str(dst))


def pack_safe_arm_poses(src: Path, dst: Path, repo_root: Path) -> None:
    """Write the slim form of a safe-arm-pose library.

    Keeps only what the training observation term reads (`poses`, `joint_names`, `qpos_indices`)
    plus the staleness fields, with `poses` demoted to float16. The end-effector columns are
    dropped: they are forward kinematics of `poses` and only a from-scratch VRW generation reads
    them, which is outside what a clone is expected to do.

    `mjcf_path` is rewritten relative to the repo root. It is stored absolute by the generator,
    and the loader hashes the file it names, so an absolute path from the machine that generated
    the cache raises FileNotFoundError on every other machine.
    """
    d = torch.load(str(src), map_location="cpu", weights_only=False)
    slim = {
        "poses": d["poses"].half(),
        "joint_names": d["joint_names"],
        "qpos_indices": d["qpos_indices"],
        "mjcf_hash": d["mjcf_hash"],
        "mjcf_path": _repo_relative_mjcf(d["mjcf_path"], repo_root),
        "sorted_by_l2_from_default": d.get("sorted_by_l2_from_default", False),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, str(dst))


def _repo_relative_mjcf(mjcf_path: str, repo_root: Path) -> str:
    """Repo-relative form of a generator-recorded ``mjcf_path``, or its basename if outside."""
    mjcf = Path(mjcf_path)
    try:
        return str(mjcf.resolve().relative_to(repo_root))
    except ValueError:
        # Lives outside the repo (e.g. the g1 MJCF, which ships inside the mjlab package).
        # Record it unresolved; the loader falls back to skipping the staleness hash.
        return mjcf.name


def rewrite_mjcf_path(src: Path, dst: Path, repo_root: Path) -> None:
    """Copy a pose cache with only `mjcf_path` made repo-relative; every other field is preserved.

    For artifacts small enough that slimming buys nothing, the absolute-path defect still has to be
    fixed: an absolute path from the generating machine makes the loader's staleness check
    unenforceable everywhere else. Rewriting the path alone keeps the end-effector columns and
    float32 poses that `pack_safe_arm_poses` discards.
    """
    d = torch.load(str(src), map_location="cpu", weights_only=False)
    d["mjcf_path"] = _repo_relative_mjcf(d["mjcf_path"], repo_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, str(dst))


def _rebuild_next_hop(slim: dict) -> torch.Tensor:
    """All-pairs next-hop routing table from the edge list, by one BFS per source.

    Mirrors what build_arm_collision_graph.py computes, so the expanded artifact is
    interchangeable with a natively generated one.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order

    n = int(slim["n_poses"])
    e = slim["edges"].numpy().astype(np.int64)
    adj = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    adj = adj + adj.T

    next_hop = np.zeros((n, n), dtype=np.uint16)
    for src in range(n):
        order, pred = breadth_first_order(adj, src, directed=False, return_predecessors=True)
        # Walk the BFS tree outward from `src`. For each node, the first hop on the path from
        # `src` is the first hop of its predecessor, except for src's direct neighbours, whose
        # first hop is themselves. Filling in BFS order guarantees the predecessor is done first.
        hop = np.zeros(n, dtype=np.uint16)
        for node in order[1:]:
            p = pred[node]
            hop[node] = node if p == src else hop[p]
        next_hop[src] = hop
    return torch.from_numpy(next_hop)


def expand(full_path: Path) -> None:
    """Materialise `full_path` from its `.slim.pt` sibling if it is not already present.

    No-op when the full artifact exists, so a machine that generated its own cache natively is
    never touched. Writes atomically: a rebuild interrupted halfway would otherwise leave a
    truncated file that looks valid to `torch.load`.
    """
    full_path = Path(full_path)
    if full_path.exists():
        return
    slim_path = _slim_path(full_path)
    if not slim_path.exists():
        return  # nothing shipped; caller's own error path (or regeneration hook) takes over

    slim = torch.load(str(slim_path), map_location="cpu", weights_only=False)
    if "edges" in slim and "next_hop" not in slim:
        print(f"[slim_cache] rebuilding next_hop for {full_path.name} (~30 s, one time)...")
        slim = dict(slim)
        slim["edges"] = [tuple(int(x) for x in row) for row in slim["edges"]]
        slim["next_hop"] = _rebuild_next_hop(
            {"n_poses": slim["n_poses"], "edges": torch.tensor(slim["edges"])}
        )
    if "poses" in slim and slim["poses"].dtype == torch.float16:
        slim["poses"] = slim["poses"].float()

    tmp = full_path.with_suffix(".tmp")
    torch.save(slim, str(tmp))
    tmp.replace(full_path)
    print(f"[slim_cache] wrote {full_path.name}")


def _self_check() -> None:
    """Round-trip the shipped humanoid_v21 graph and assert the rebuild matches the original."""
    repo = Path(__file__).resolve().parents[2]
    src = repo / "mj_envs/asset_zoo/cache/arm_collision_graph_humanoid_v21.pt"
    assert src.exists(), f"need {src} to self-check"
    orig = torch.load(str(src), map_location="cpu", weights_only=False)

    # A float16 round-trip must NOT be used for indices; assert the failure mode is real so the
    # rule in this module's docstring stays honest if someone "optimises" it later.
    nh = orig["next_hop"].to(torch.int32)
    corrupt = (nh.float().half().float().to(torch.int32) != nh).sum().item()
    assert corrupt > 0, "expected float16 to corrupt large indices"

    sub = {"n_poses": 400, "edges": None}
    e = np.asarray(orig["edges"])
    e = e[(e[:, 0] < 400) & (e[:, 1] < 400)]
    sub["edges"] = torch.tensor(e)
    rebuilt = _rebuild_next_hop(sub)
    # Every rebuilt hop must be an actual neighbour of its source, or zero (unreachable).
    adj = {(int(a), int(b)) for a, b in e} | {(int(b), int(a)) for a, b in e}
    checked = 0
    for i in range(400):
        for j in range(400):
            h = int(rebuilt[i, j])
            if h != 0 and i != j:
                assert (i, h) in adj, f"next_hop[{i},{j}]={h} is not a neighbour of {i}"
                checked += 1
    assert checked > 1000, f"only {checked} hops exercised, subgraph too sparse to be meaningful"
    print(f"self-check ok: {checked} rebuilt hops are valid neighbours; "
          f"float16 would corrupt {corrupt:,} index entries")


if __name__ == "__main__":
    _self_check()
