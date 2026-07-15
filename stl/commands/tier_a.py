#!/usr/bin/env python3
"""Tier A build (plan Phase 3): exact-lattice epochs with <= 1 total death.

Stage 1 — 480 deaths=1 epochs: (dier in {hal, baku}) x (ttd in 60..299),
cprs=1, survived-death exits to deaths=2 bracketed [-1, +1]. Epochs are
independent — solved in a process pool, one .npz per epoch, resumable.

Stage 2 — the deaths=0 epoch (canonical opening lattice): its survived-
death exits land exactly on stage-1 epochs (duration in {60..299} ==
the dier's new ttd), so its brackets inherit stage-1 tightness.

Artifacts: checkpoints/tablebase/tier_a/d1_{dier}_{ttd}.npz (lo, hi as
float32), d0.npz, manifest.json with sha256 per file.

Cost (measured pilot numbers): ~7.4-8.8 ms/state, 180,000 states/epoch
=> ~22-26 min/epoch single-core; 481 epochs on N workers ~= 9-10 h at 20
workers. Run detached; progress lines stream per epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints",
    "tablebase",
    "tier_a",
)
TTD_RANGE = range(60, 300)  # survivable death durations == reachable ttds at 1 death


def epoch_path(dier: str, ttd: int) -> str:
    return os.path.join(OUT_DIR, f"d1_{dier}_{ttd}.npz")


def solve_one_d1_epoch(args: tuple[str, int]) -> tuple[str, int, float]:
    """Worker: solve one deaths=1 epoch and save it. Returns timing info."""
    dier, ttd = args
    from stl.solver.tablebase import EpochSpec, solve_epoch
    from stl.solver.tablebase import bracket_survive_value

    spec = EpochSpec(
        ttd_hal=float(ttd) if dier == "hal" else 0.0,
        ttd_baku=float(ttd) if dier == "baku" else 0.0,
        cprs=1,
    )
    start = time.perf_counter()
    V_lo, V_hi = solve_epoch(spec, bracket_survive_value())
    elapsed = time.perf_counter() - start

    path = epoch_path(dier, ttd)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, lo=V_lo.astype(np.float32), hi=V_hi.astype(np.float32))
    os.replace(tmp, path)
    return dier, ttd, elapsed


def stage1(workers: int, limit: int | None, max_tasks_per_child: int = 8) -> None:
    todo = [
        (dier, ttd)
        for dier in ("hal", "baku")
        for ttd in TTD_RANGE
        if not os.path.exists(epoch_path(dier, ttd))
    ]
    if limit is not None:
        todo = todo[:limit]
    total = len(todo)
    print(f"[stage1] {total} epochs to solve ({480 - total} already done)", flush=True)
    if not todo:
        return

    done = 0
    start = time.perf_counter()
    last_beat = start
    with ProcessPoolExecutor(
        max_workers=workers, max_tasks_per_child=max_tasks_per_child
    ) as pool:
        futures = {pool.submit(solve_one_d1_epoch, item): item for item in todo}
        pending = set(futures)
        while pending:
            from concurrent.futures import FIRST_COMPLETED, wait

            finished, pending = wait(pending, timeout=120, return_when=FIRST_COMPLETED)
            now = time.perf_counter()
            for future in finished:
                dier, ttd, elapsed = future.result()
                done += 1
                rate = done / (now - start) * 3600
                eta_h = (total - done) / max(rate, 1e-9)
                print(
                    f"[stage1] d1_{dier}_{ttd} solved in {elapsed/60:.1f} min "
                    f"({done}/{total}, {rate:.1f} epochs/h, ETA {eta_h:.1f} h)",
                    flush=True,
                )
            if now - last_beat >= 300:
                print(f"[heartbeat] alive: {done}/{total} done, {len(pending)} pending, "
                      f"wall {(now - start)/60:.0f} min", flush=True)
                last_beat = now


def stage2() -> None:
    from stl.solver.tablebase import EpochSpec, solve_epoch

    missing = [
        (dier, ttd)
        for dier in ("hal", "baku")
        for ttd in TTD_RANGE
        if not os.path.exists(epoch_path(dier, ttd))
    ]
    if missing:
        print(f"[stage2] blocked: {len(missing)} deaths=1 epochs missing", flush=True)
        return

    cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    def load(dier: str, ttd: int) -> tuple[np.ndarray, np.ndarray]:
        key = (dier, ttd)
        if key not in cache:
            with np.load(epoch_path(dier, ttd)) as data:
                cache[key] = (
                    data["lo"].astype(np.float64),
                    data["hi"].astype(np.float64),
                )
        return cache[key]

    def survive_value(checker_is_hal: bool, duration: int, next_bit: int, other_cyl: int):
        dier = "hal" if checker_is_hal else "baku"
        lo, hi = load(dier, duration)
        if checker_is_hal:  # Hal's cylinder reset to 0
            return float(lo[next_bit, 0, other_cyl]), float(hi[next_bit, 0, other_cyl])
        return float(lo[next_bit, other_cyl, 0]), float(hi[next_bit, other_cyl, 0])

    print("[stage2] solving deaths=0 epoch against the 480 solved epochs", flush=True)
    start = time.perf_counter()
    spec = EpochSpec(ttd_hal=0.0, ttd_baku=0.0, cprs=0)
    V_lo, V_hi = solve_epoch(
        spec, survive_value, progress=lambda msg: print(f"[stage2] {msg}", flush=True)
    )
    elapsed = time.perf_counter() - start

    path = os.path.join(OUT_DIR, "d0.npz")
    np.savez_compressed(path, lo=V_lo.astype(np.float32), hi=V_hi.astype(np.float32))
    width = V_hi - V_lo
    print(f"[stage2] done in {elapsed/60:.1f} min", flush=True)
    print(
        f"[stage2] opening-lattice width: mean={np.nanmean(width):.4f} "
        f"median={np.nanmedian(width):.4f} max={np.nanmax(width):.4f}",
        flush=True,
    )
    print(
        f"[stage2] canonical post-leap fresh-cylinder values: "
        f"Hal-drops [{V_lo[0,0,0]:+.4f}, {V_hi[0,0,0]:+.4f}]  "
        f"Baku-drops [{V_lo[1,0,0]:+.4f}, {V_hi[1,0,0]:+.4f}]",
        flush=True,
    )


def write_manifest() -> None:
    manifest = {}
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(OUT_DIR, name)
        with open(path, "rb") as fh:
            manifest[name] = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[manifest] {len(manifest)} artifacts hashed", flush=True)


def _pid_alive(pid: int) -> bool:
    """Windows-safe liveness check (os.kill(pid, 0) TERMINATES on Windows)."""
    import subprocess

    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, check=False,
    ).stdout
    return str(pid) in out


def acquire_lock() -> bool:
    """Single-instance guard so a scheduled auto-resume never double-runs."""
    lock = os.path.join(OUT_DIR, ".lock")
    if os.path.exists(lock):
        try:
            with open(lock) as fh:
                pid = int(fh.read().strip())
        except (ValueError, OSError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            print(f"[lock] another build (pid {pid}) is running; exiting", flush=True)
            return False
        print(f"[lock] removing stale lock (pid {pid})", flush=True)
        os.remove(lock)
    with open(lock, "w") as fh:
        fh.write(str(os.getpid()))
    return True


def release_lock() -> None:
    lock = os.path.join(OUT_DIR, ".lock")
    try:
        os.remove(lock)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="solve at most N epochs this run (validation)")
    parser.add_argument("--max-tasks-per-child", type=int, default=8)
    parser.add_argument("--stage", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if not acquire_lock():
        return
    try:
        if args.stage in ("1", "all"):
            stage1(args.workers, args.limit, max_tasks_per_child=args.max_tasks_per_child)
        if args.stage in ("2", "all") and args.limit is None:
            if os.path.exists(os.path.join(OUT_DIR, "d0.npz")):
                print("[stage2] d0.npz already exists; skipping", flush=True)
            else:
                stage2()
        write_manifest()
    finally:
        release_lock()


if __name__ == "__main__":
    main()
