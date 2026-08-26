"""Supervise deterministic cuRobo reachability shards on a multi-GPU host.

Run this program on the remote host once the tree is in place. It preclaims only its own GPU-guard locks,
waits for sentinel release, launches one independent grid slice per ``GPU:CHUNK`` pair, retains
per-chunk PID/log files, and removes only locks it created after its worker exits. Workers rebuild
the same global grid × SO(3) list, so no target bundle is shared between GPUs.
"""

from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _parse_pair(value: str) -> tuple[int, int]:
    try:
        gpu_text, chunk_text = value.split(":", maxsplit=1)
        gpu, chunk = int(gpu_text), int(chunk_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected GPU:CHUNK, got {value!r}") from exc
    if not 0 <= gpu <= 7 or chunk < 0:
        raise argparse.ArgumentTypeError(f"invalid GPU:CHUNK, got {value!r}")
    return gpu, chunk


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--pad", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--grid-spacing", type=float, default=0.02)
    parser.add_argument("--n-orientations", type=int, default=64)
    parser.add_argument(
        "--orientation-source",
        choices=("so3", "position-only"),
        default="so3",
        help="so3 for final dexterity; position-only for batched spatial feasibility gating",
    )
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--position-gate", type=Path, default=None)
    parser.add_argument("--side", choices=("R", "L"), default="R")
    parser.add_argument("--log-every", type=int, default=200000)
    parser.add_argument("--repo", type=Path, default=Path.home() / "repo" / "legged_env_v2")
    parser.add_argument("--python", dest="python_path", type=Path,
                        default=Path.home() / "repo" / "micromamba" / "envs" / "py312" / "bin" / "python")
    parser.add_argument("--sentinel-wait", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("pairs", metavar="GPU:CHUNK", nargs="+", type=_parse_pair)
    return parser


def _worker_command(args: argparse.Namespace, chunk: int) -> list[str]:
    start = chunk * args.chunk_size
    return [
        str(args.python_path), "-u",
        "mj_envs/asset_zoo/reachability_study/generate_workspace_curobo.py",
        "--robot", args.robot, "--side", args.side,
        "--target-source", "grid", "--orientation-source", args.orientation_source,
        "--n-orientations", str(args.n_orientations),
        "--grid-spacing", str(args.grid_spacing), "--target-order", "grid",
        "--grid-bounds-source", "legacy-symmetric", "--pad", str(args.pad),
        "--target-start", str(start), "--limit", str(args.chunk_size),
        "--solver", "batched", "--num-seeds", str(args.num_seeds),
        "--batch-size", str(args.batch_size), "--device", "cuda:0",
        "--out", str(args.out_dir / f"chunk_{chunk}.pt"),
        "--log-every", str(args.log_every),
    ] + ([] if args.position_gate is None else ["--position-gate", str(args.position_gate)])


def main() -> None:
    args = _parser().parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if not args.repo.is_dir() or not args.python_path.is_file():
        raise FileNotFoundError("remote --repo or --python path missing")
    if len({gpu for gpu, _ in args.pairs}) != len(args.pairs):
        raise ValueError("each GPU may receive one chunk per driver invocation")
    if len({chunk for _, chunk in args.pairs}) != len(args.pairs):
        raise ValueError("chunk ids must be unique")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    owned_locks: set[Path] = set()
    workers: dict[int, tuple[subprocess.Popen, object, Path | None]] = {}

    def cleanup() -> None:
        for process, log, lock in workers.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            log.close()
            if lock is not None:
                lock.unlink(missing_ok=True)
        workers.clear()
        for lock in owned_locks:
            lock.unlink(missing_ok=True)

    atexit.register(cleanup)

    def on_signal(signum: int, _frame) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if args.dry_run:
        for gpu, chunk in args.pairs:
            print(f"gpu={gpu} chunk={chunk} start={chunk * args.chunk_size}: {' '.join(_worker_command(args, chunk))}")
        return

    for gpu, _ in args.pairs:
        # The guard file name is a fixed convention shared by every host, not the local hostname;
        # never remove a pre-existing owner lock.
        lock = Path("/tmp") / f"ser16_user_gpu_{gpu}.lock"
        if not lock.exists():
            lock.touch()
            owned_locks.add(lock)

    time.sleep(args.sentinel_wait)
    for gpu, chunk in args.pairs:
        out = args.out_dir / f"chunk_{chunk}.pt"
        pid_path = args.out_dir / f"chunk_{chunk}.pid"
        if out.is_file() and out.stat().st_size > 0:
            print(f"chunk {chunk} exists, skip")
            continue
        if pid_path.is_file():
            try:
                existing_pid = int(pid_path.read_text().strip())
                os.kill(existing_pid, 0)
            except (OSError, ValueError):
                pass
            else:
                print(f"chunk {chunk} already running as pid {existing_pid}, skip")
                continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = ":".join((str(Path.home() / "repo" / "curobo"), str(args.repo), str(args.repo / "mj_envs")))
        env["LD_LIBRARY_PATH"] = f"{Path.home() / 'repo' / 'micromamba' / 'envs' / 'py312' / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
        log = (args.out_dir / f"chunk_{chunk}.log").open("w")
        process = subprocess.Popen(
            _worker_command(args, chunk), cwd=args.repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        pid_path.write_text(f"{process.pid}\n")
        lock = Path("/tmp") / f"ser16_user_gpu_{gpu}.lock"
        workers[chunk] = (process, log, lock if lock in owned_locks else None)
        print(f"launched chunk {chunk} on gpu {gpu}: pid {process.pid}", flush=True)

    while workers:
        for chunk, (process, log, lock) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            if lock is not None:
                lock.unlink(missing_ok=True)
                owned_locks.discard(lock)
            del workers[chunk]
            print(f"chunk {chunk} {'completed' if code == 0 else f'failed ({code})'}", flush=True)
        if workers:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
