"""
Build arm collision graph from 2M safe pose library.

Run sweep_graph_connectivity.py first to determine --n-poses.

Steps:
  1. Profile random vs mini-batch FPS subsampling; use faster method.
  2. Subsample N poses from 2M pose library.
  3. Build kNN graph; filter edges with full-body MuJoCo collision check (arm vs all bodies).
  4. Verify graph is connected; abort if not.
  5. Compute all-pairs next_hop via BFS.
  6. Save artifact to cache/arm_collision_graph_{robot}.pt.

Usage:
    python mj_envs/asset_zoo/scripts/build_arm_collision_graph.py --n-poses 10000
    python mj_envs/asset_zoo/scripts/build_arm_collision_graph.py --n-poses 10000 --k 30 --robot humanoid_v21
"""

import sys
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import mujoco
import warp as wp
import mujoco_warp as mjwarp
import networkx as nx
import tyro
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import cKDTree

sys.path.append(str(Path(__file__).resolve().parents[3]))

from mjlab.entity import Entity
from mjlab.sim.sim import Simulation, SimulationCfg


def _arm_body_ids(model: mujoco.MjModel) -> set[int]:
    """Same kinematic-tree traversal as get_arm_joint_info."""
    child_counts = [0] * model.nbody
    for i in range(model.nbody):
        p = model.body_parentid[i]
        if p >= 0:
            child_counts[p] += 1
    torso = max(range(1, model.nbody), key=lambda i: child_counts[i])

    arm_bodies: set[int] = set()
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if any(k in name.lower() for k in ("hand", "wrist", "gripper")):
            curr, branch = i, set()
            while curr != torso and curr >= 0:
                branch.add(curr)
                curr = model.body_parentid[curr]
            if curr == torso:
                arm_bodies.update(branch)
    return arm_bodies


def _fps_minibatch(poses: torch.Tensor, N: int, pool: int = 100_000) -> torch.Tensor:
    """Farthest-point sampling via random pool: random pool_size → FPS → N indices into pool."""
    M = poses.shape[0]
    pool = min(pool, M)
    pool_idx = torch.randperm(M, device=poses.device)[:pool]
    pts = poses[pool_idx]   # (pool, D)

    selected = torch.zeros(N, dtype=torch.long, device=poses.device)
    dists = torch.full((pool,), float("inf"), device=poses.device)
    cur = torch.randint(0, pool, (1,), device=poses.device).item()
    selected[0] = cur

    for i in range(1, N):
        d = ((pts - pts[cur]) ** 2).sum(dim=1)
        dists = torch.minimum(dists, d)
        cur = int(torch.argmax(dists).item())
        selected[i] = cur

    return pool_idx[selected]  # indices into original 2M


def _profile_sampling(poses: torch.Tensor, N: int, device: str) -> tuple[str, torch.Tensor]:
    """Profile random vs mini-batch FPS; return (method_name, indices_into_poses)."""
    M = poses.shape[0]
    poses_gpu = poses.to(device)

    # Random
    REPS = 3
    t0 = time.perf_counter()
    for _ in range(REPS):
        r_idx = torch.randperm(M)[:N]
    t_rand = (time.perf_counter() - t0) / REPS

    # FPS
    t0 = time.perf_counter()
    for _ in range(REPS):
        f_idx = _fps_minibatch(poses_gpu, N)
    t_fps = (time.perf_counter() - t0) / REPS

    print(f"  Random: {t_rand*1000:.1f} ms/call   Mini-batch FPS: {t_fps*1000:.1f} ms/call")

    if t_fps <= t_rand * 1.5:
        print("  → Using mini-batch FPS (comparable speed, better coverage)")
        method = "fps"
        idx = _fps_minibatch(poses_gpu, N).cpu()
    else:
        print("  → Using random (faster)")
        method = "random"
        idx = torch.randperm(M)[:N]

    return method, idx


def _knn_edges(poses: np.ndarray, k: int) -> np.ndarray:
    """Unique undirected kNN edges. Returns (E, 2) int32."""
    tree = cKDTree(poses)
    _, nbrs = tree.query(poses, k=k + 1)
    N = poses.shape[0]
    src = np.repeat(np.arange(N, dtype=np.int32), k)
    dst = nbrs[:, 1:].flatten().astype(np.int32)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


def _test_collisions(
    sim: "Simulation",
    fwd_graph,
    poses_gpu: torch.Tensor,
    edges: np.ndarray,
    qpos_idx: torch.Tensor,
    arm_body_t: torch.Tensor,
    n_interp: int,
    device: str,
) -> np.ndarray:
    """Return bool (E,): True = arm collision detected on interpolated edge."""
    E = edges.shape[0]
    sim_batch = sim.num_envs
    has_col = np.zeros(E, dtype=bool)
    geom_body = torch.tensor(sim.model.geom_bodyid, device=device)

    edges_t = torch.from_numpy(edges).long().to(device)
    pa = poses_gpu[edges_t[:, 0]]
    pb = poses_gpu[edges_t[:, 1]]

    for step in range(n_interp):
        alpha = step / max(n_interp - 1, 1)
        waypoints = (1.0 - alpha) * pa + alpha * pb

        n_done = 0
        for b0 in range(0, E, sim_batch):
            be = min(b0 + sim_batch, E)
            B = be - b0
            wp_batch = waypoints[b0:be]

            sim.reset()
            sim.data.qpos[:B, qpos_idx] = wp_batch
            with wp.ScopedDevice(sim.wp_device):
                wp.capture_launch(fwd_graph)

            is_contact = (sim.data.contact.dist < 0.0) & (sim.data.contact.worldid >= 0)
            if is_contact.any():
                g1 = sim.data.contact.geom[is_contact, 0]
                g2 = sim.data.contact.geom[is_contact, 1]
                b1 = geom_body[g1]
                b2 = geom_body[g2]
                arm_mask = (
                    (torch.isin(b1, arm_body_t) | torch.isin(b2, arm_body_t))
                    & (b1 > 0) & (b2 > 0)
                )
                wids = sim.data.contact.worldid[is_contact][arm_mask]
                valid = (wids >= 0) & (wids < B)
                has_col[b0 + wids[valid].cpu().numpy()] = True

            n_done += B
            if n_done % 20000 == 0:
                print(f"    {n_done}/{E} edges tested (step {step+1}/{n_interp})", end="\r")

    print()
    return has_col


def _compute_next_hop(adj: sp.csr_matrix, N: int) -> np.ndarray:
    """All-pairs next-hop table via BFS. Returns (N, N) uint16.

    next_hop[i][j] = first node to visit from i on shortest path to j.
    Requires connected graph. If j unreachable from i, next_hop[i][j] = i (sentinel).
    """
    next_hop = np.zeros((N, N), dtype=np.uint16)

    for i in range(N):
        order, pred = breadth_first_order(adj, i_start=i, directed=False, return_predecessors=True)
        # Propagate first_hop down BFS tree; BFS order ensures parent processed first.
        first_hop = np.full(N, i, dtype=np.int32)  # default: self (sentinel for unreachable)
        for node in order:
            p = pred[node]
            if p < 0 or node == i:  # root
                continue
            if p == i:
                first_hop[node] = node
            else:
                first_hop[node] = first_hop[p]
        next_hop[i] = first_hop.astype(np.uint16)

        if i % 500 == 0:
            print(f"  BFS {i}/{N}", end="\r")

    print()
    return next_hop


@dataclass
class BuildArgs:
    n_poses: int = 10000
    k: int = 20
    n_interp: int = 5
    sim_batch: int = 2048
    robot: str = "humanoid_v21"
    device: str = "cuda:0"
    force: bool = False


def main(args: BuildArgs) -> None:
    out_path = (
        Path(__file__).resolve().parents[1]
        / "cache"
        / f"arm_collision_graph_{args.robot}.pt"
    )
    if out_path.exists() and not args.force:
        print(f"Artifact exists: {out_path}\nUse --force to rebuild.")
        return

    # --- Model ---
    print("Loading model...")
    if args.robot == "humanoid_v21":
        from mj_envs.asset_zoo.humanoid_v21.humanoid_v21_constants import get_humanoid_v21_robot_cfg
        entity = Entity(get_humanoid_v21_robot_cfg())
    elif args.robot == "unitree_g1":
        from mj_envs.asset_zoo.g1.g1_constants import get_g1_robot_cfg
        entity = Entity(get_g1_robot_cfg(end_effector="welded", hand="parallel_gripper", head_camera="actuated"))
    else:
        raise ValueError(f"Unsupported robot: {args.robot}")

    model = entity.spec.compile()
    arm_bodies = _arm_body_ids(model)
    arm_body_t = torch.tensor(sorted(arm_bodies), device=args.device, dtype=torch.int32)

    # --- Load pose library ---
    cache = Path(__file__).resolve().parents[1] / "cache" / f"safe_arm_poses_{args.robot}.pt"
    if not cache.exists():
        print(f"Safe arm poses library not found: {cache}. Generating on the fly...")
        from mj_envs.asset_zoo.generate_safe_arm_poses import generate_safe_arm_poses
        generate_safe_arm_poses(robot=args.robot, device=args.device)
    raw = torch.load(str(cache), map_location="cpu", weights_only=False)
    all_poses   = raw["poses"]        # (2M, D)
    joint_names = raw["joint_names"]
    qpos_idx = raw["qpos_indices"].to(device=args.device)
    print(f"Arm joints ({len(joint_names)}): {joint_names}")
    print(f"Pose library: {all_poses.shape}")

    # --- Profile + subsample ---
    print(f"\nProfiling subsampling strategies for N={args.n_poses}...")
    method, idx = _profile_sampling(all_poses, args.n_poses, args.device)
    poses_cpu = all_poses[idx].numpy()
    poses_gpu = torch.tensor(poses_cpu, device=args.device)
    print(f"Subsampled {args.n_poses} poses via {method}")

    # --- kNN graph ---
    print(f"\nBuilding kNN graph (k={args.k})...")
    t0 = time.perf_counter()
    edges = _knn_edges(poses_cpu, args.k)
    print(f"Candidate edges: {edges.shape[0]}  ({time.perf_counter()-t0:.1f}s)")

    # --- Sim + collision test ---
    print(f"\nTesting edges (n_interp={args.n_interp}, sim_batch={args.sim_batch})...")
    sim_cfg = SimulationCfg(nconmax=150, njmax=500)
    sim_cfg.mujoco.ccd_iterations = 100
    sim = Simulation(num_envs=args.sim_batch, cfg=sim_cfg, model=model, device=args.device)
    sim.reset()

    wp.capture_begin(sim.wp_device)
    mjwarp._src.forward.fwd_position(sim.wp_model, sim.wp_data, factorize=False)
    fwd_graph = wp.capture_end(sim.wp_device)

    t0 = time.perf_counter()
    has_col = _test_collisions(
        sim, fwd_graph, poses_gpu, edges, qpos_idx, arm_body_t,
        args.n_interp, args.device,
    )
    valid_edges = edges[~has_col]
    print(
        f"Valid edges: {valid_edges.shape[0]} / {edges.shape[0]} "
        f"({100*valid_edges.shape[0]/edges.shape[0]:.1f}% safe)  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

    # --- Connectivity check: keep largest connected component ---
    print("\nChecking connectivity...")
    G = nx.Graph()
    G.add_nodes_from(range(args.n_poses))
    G.add_edges_from(valid_edges.tolist())
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        # Isolated nodes represent arm configs with no collision-free path to any neighbor
        # (all edges cross body geometry). Drop them — they are unreachable from safe space.
        largest_comp = max(comps, key=len)
        n_dropped = args.n_poses - len(largest_comp)
        print(
            f"WARNING: {len(comps)} components. Dropping {n_dropped} isolated nodes "
            f"(arm configs with no collision-free neighbor paths). "
            f"Keeping {len(largest_comp)}/{args.n_poses} nodes."
        )
        keep = sorted(largest_comp)
        keep_set = set(keep)
        remap = {old: new for new, old in enumerate(keep)}
        poses_cpu = poses_cpu[keep]
        poses_gpu = poses_gpu[keep]
        valid_edges = np.array(
            [[remap[a], remap[b]] for a, b in valid_edges.tolist() if a in keep_set and b in keep_set],
            dtype=np.int32,
        )
        args.n_poses = len(keep)
    else:
        print("Graph is connected.")

    # --- All-pairs next_hop via BFS ---
    print(f"\nComputing all-pairs next_hop for N={args.n_poses}...")
    t0 = time.perf_counter()
    rows = valid_edges[:, 0]
    cols = valid_edges[:, 1]
    E = valid_edges.shape[0]
    adj = sp.csr_matrix(
        (np.ones(2 * E, dtype=np.float32),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(args.n_poses, args.n_poses),
    )
    next_hop = _compute_next_hop(adj, args.n_poses)
    print(f"next_hop computed ({time.perf_counter()-t0:.1f}s)")

    memory_mb = next_hop.nbytes / 1e6
    print(f"next_hop: {next_hop.shape}, {memory_mb:.1f} MB")
    if memory_mb > 900:
        raise RuntimeError(f"next_hop exceeds memory budget ({memory_mb:.1f} MB > 900 MB)")

    # --- Save artifact ---
    artifact = {
        "poses":       torch.tensor(poses_cpu, dtype=torch.float32),
        "next_hop":    torch.tensor(next_hop, dtype=torch.uint16),
        "edges":       valid_edges.tolist(),
        "joint_names": joint_names,
        "n_poses":     args.n_poses,
        "k":           args.k,
        "method":      method,
        "robot":       args.robot,
    }
    torch.save(artifact, str(out_path))
    total_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}  ({total_mb:.1f} MB)")


if __name__ == "__main__":
    main(tyro.cli(BuildArgs))
