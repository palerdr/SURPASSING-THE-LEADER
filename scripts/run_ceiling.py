"""End-to-end driver for the Phase I-2 *ceiling run* (charter:
``docs/SOLVER_EXTENSION_GOAL.md``).

Executes the full pipeline the charter deferred, OOM-safe and resumable:

  Stage 1  Regenerate the full Phase-G training corpus in memory-bounded
           ``baku_cyl`` chunks (sequential — the per-worker exact-subtree memo
           is freed between chunks), then merge → ``ceiling_corpus.npz``.
  Stage 2  Regenerate the held-out ruler with F-2 interior anchors
           (``run_ceiling_holdout.py``) → ``ceiling_holdout_interior.npz``.
  Stage 3  Train hidden=192 + calibration/AlphaZero gate
           (``run_gen_iteration.py``) → ``gen_ceiling_h192/``.
  Stage 4  Global invariant gate (§4 pytest block).

OOM safety is two-layered:
  * Primary — chunking the 14-value ``baku_cyl`` axis into thirds and running
    them **sequentially**, so peak RAM is ~chunk/full of the full-grid footprint
    (the full 7168-grid at 8 workers exhausts 24 GiB + 16 GiB swap; a 5/14 chunk
    is ~9 GiB).
  * Watchdog — a sampler on ``/proc/meminfo`` watching BOTH MemAvailable and
    SwapUsed. On a host with 16 GiB swap, MemAvailable alone is not the OOM
    signal: the kernel pages the workers' anonymous memo tables out and thrashes
    (wall-clock explodes) long before MemAvailable bottoms out, so rising
    SwapUsed is the thrashing signal. (Summed per-process RSS is *not* used — it
    double-counts fork-shared pages and would false-kill the worker pool.) On
    sustained pressure — or a per-chunk wall-clock deadline, which catches the
    silent hang an OOM-killed ``multiprocessing.Pool`` worker leaves behind — the
    chunk's process group is SIGTERM'd, SIGKILL-swept, and retried at fewer
    workers (6→4→2).

Resumable: any stage whose output already exists is skipped, so a Stage-3 gate
failure never forces redoing the ~14h corpus.

Usage (background):
    python scripts/run_ceiling.py 2>&1 | tee checkpoints/ceiling_run.log
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.value_targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_TABLEBASE_INTERIOR,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# ── Stage 1 partition — the union of the three chunk lists is exactly
# PHASE_G_BAKU_CYL_GRID (14 values), so chunking loses nothing relative to the
# full grid and the 239→240→241 memo-reuse triplet stays in one chunk. (Pin
# coverage is a separate matter and NOT a partition concern: the grid sweep
# reaches only the baku_cyl=299 pins; the full 22-pin tablebase set enters
# TRAINING via run_gen_iteration's bootstrap anchor-class registry walk, not this
# sweep.) ─────────────────────────────────────────────────────────────────────
CHUNKS: tuple[tuple[str, str], ...] = (
    ("A", "0,60,120,180,220"),
    ("B", "239,240,241,270,289"),
    ("C", "290,295,297,299"),
)

# Watchdog: preempt on sustained RAM exhaustion OR swap thrashing. Chunking keeps
# a chunk's footprint ~9 GiB (well under 23 GiB RAM), so swap should never be
# touched and this never fires — it is the safety net for the unexpected.
MEM_FLOOR_GIB = 2.5            # true RAM exhaustion (MemAvailable)
SWAP_USED_CEIL_GIB = 2.0       # thrashing onset: >2 GiB of memo spilled to swap
SAMPLE_SECONDS = 10
SUSTAINED_SAMPLES = 3          # 3 × 10s = 30s sustained breach before preempt
# 24h backstop, NOT a per-chunk budget: a healthy chunk at 12 workers finishes
# well under this. It exists only to reap a chunk genuinely hung by an OOM-killed
# Pool worker (pool.map blocks forever). A deadline hit ABORTS-and-reports — it
# never downshifts workers, because a slow-but-healthy chunk is a compute-budget
# problem and fewer workers only makes it slower (the bug that burned the 1st run).
CHUNK_DEADLINE_SECONDS = 86400
# Memory-preempt retry ladder. 32-core box; per-worker memo SHRINKS as workers
# rise (each worker partitions fewer states), so total RAM is ~worker-independent
# (~10-13 GiB peak per chunk). Start high for speed; only a real memory preempt
# walks it down.
WORKER_LADDER = (12, 8, 4)


def log(msg: str) -> None:
    print(f"[ceiling] {msg}", flush=True)


def mem_pressure_gib() -> tuple[float, float]:
    """Return ``(MemAvailable, SwapUsed)`` in GiB.

    MemAvailable alone is not the OOM signal on a swap-backed host: the kernel
    pages the workers' anonymous memo tables out to the 16 GiB swap and thrashes
    long before MemAvailable bottoms out. Rising SwapUsed is the thrashing
    signal; both are watched.
    """
    fields: dict[str, float] = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("MemAvailable", "SwapTotal", "SwapFree"):
                fields[key] = int(rest.split()[0]) / (1024 * 1024)  # kB → GiB
    avail = fields.get("MemAvailable", float("inf"))
    swap_used = fields.get("SwapTotal", 0.0) - fields.get("SwapFree", 0.0)
    return avail, swap_used


def _sigkill_sweep(pgid: int) -> None:
    """SIGKILL the whole process group so no fork-shared worker outlives the
    reaped leader still holding memo RAM. Idempotent — the group may be empty."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_with_mem_guard(
    cmd: list[str],
    *,
    mem_floor_gib: float = MEM_FLOOR_GIB,
    swap_ceil_gib: float = SWAP_USED_CEIL_GIB,
    deadline_seconds: float = CHUNK_DEADLINE_SECONDS,
) -> str:
    """Run ``cmd`` in its own process group, watching memory + swap pressure and
    a wall-clock deadline. Returns one of: ``"ok"`` (exit 0), ``"mem"`` (preempted
    on sustained memory/swap pressure — caller retries at fewer workers),
    ``"deadline"`` (exceeded the wall-clock backstop — caller ABORTS, since fewer
    workers would only be slower), or ``"fail:<rc>"`` (nonzero exit). Every exit
    path SIGKILL-sweeps the group so a straggler can't survive into the next retry."""
    proc = subprocess.Popen(cmd, cwd=str(REPO), start_new_session=True)
    pgid = os.getpgid(proc.pid)
    breaches = 0
    start = time.monotonic()
    while True:
        rc = proc.poll()
        if rc is not None:
            _sigkill_sweep(pgid)  # reap stragglers even on a clean leader exit
            return "ok" if rc == 0 else f"fail:{rc}"
        avail, swap_used = mem_pressure_gib()
        elapsed = time.monotonic() - start
        if avail < mem_floor_gib or swap_used > swap_ceil_gib:
            breaches += 1
            log(f"  ⚠ pressure: MemAvail {avail:.1f} GiB, SwapUsed {swap_used:.1f} GiB "
                f"({breaches}/{SUSTAINED_SAMPLES})")
        else:
            breaches = 0
        over_deadline = elapsed > deadline_seconds
        if breaches >= SUSTAINED_SAMPLES or over_deadline:
            reason = "deadline" if over_deadline else "mem"
            why = "wall-clock deadline" if over_deadline else "sustained memory/swap pressure"
            log(f"  ⚠ preempting chunk process group ({why}, elapsed {elapsed / 3600:.1f}h)")
            try:
                os.killpg(pgid, signal.SIGTERM)
                proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass
            _sigkill_sweep(pgid)
            return reason
        time.sleep(SAMPLE_SECONDS)


def corpus_ok(path: Path) -> bool:
    """A chunk/merge output counts as done only if it loads to a non-empty corpus."""
    if not path.exists():
        return False
    try:
        return len(load_targets_as_records(path)) > 0
    except Exception:
        return False


def run_corpus_chunk(name: str, baku_cyl: str, out: Path) -> None:
    if corpus_ok(out):
        log(f"Stage 1 chunk {name}: {out.name} already present "
            f"({len(load_targets_as_records(out))} records) — skip")
        return
    for workers in WORKER_LADDER:
        log(f"Stage 1 chunk {name}: baku={baku_cyl} workers={workers}")
        cmd = [PY, "scripts/run_phase_g_corpus.py",
               "--out", str(out), "--baku-cyl", baku_cyl, "--workers", str(workers)]
        result = run_with_mem_guard(cmd)
        if result == "ok":
            log(f"Stage 1 chunk {name}: done "
                f"({len(load_targets_as_records(out))} records)")
            return
        if result.startswith("fail"):
            raise SystemExit(f"chunk {name} failed ({result}) — stop-and-report")
        if result == "deadline":
            raise SystemExit(
                f"chunk {name} exceeded the {CHUNK_DEADLINE_SECONDS / 3600:.0f}h deadline at "
                f"workers={workers}. This is a compute-budget/hang condition, NOT memory "
                f"(reducing workers would only be slower). Raise CHUNK_DEADLINE_SECONDS or "
                f"workers and investigate — stop-and-report.")
        # result == "mem": real memory pressure — retry at fewer workers.
        log(f"Stage 1 chunk {name}: MEMORY-preempted at workers={workers}, retrying lower")
    raise SystemExit(f"chunk {name} memory-preempted at every worker level — aborting")


def merge_corpus(chunk_paths: list[Path], out: Path) -> None:
    if corpus_ok(out):
        log(f"Stage 1 merge: {out.name} already present — skip")
        return
    merged: list = []
    for p in chunk_paths:
        recs = load_targets_as_records(p)
        log(f"  merge: +{len(recs)} from {p.name}")
        merged.extend(recs)
    breakdown = source_breakdown(merged)
    # Defensibility: the corpus's job is to supply the exact-LP horizon bulk (the
    # tablebase pins are injected separately by run_gen_iteration's bootstrap
    # anchor-class walk, so tablebase count here is not the right sanity check).
    # A merge missing a horizon source means a chunk produced no LP labels — a
    # broken sweep, not a valid corpus.
    for src in (SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3):
        if breakdown.get(src, 0) == 0:
            raise SystemExit(f"merged corpus missing {src} — broken sweep: {breakdown}")
    save_targets(merged, out)
    log(f"Stage 1 merge: {len(merged)} records → {out.name}  breakdown={breakdown}")


def holdout_has_interior(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return source_breakdown(load_targets_as_records(path)).get(SOURCE_TABLEBASE_INTERIOR, 0) > 0
    except Exception:
        return False


def run_subprocess(cmd: list[str], stage: str) -> None:
    log(f"{stage}: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        raise SystemExit(f"{stage} failed (exit {rc}) — stop-and-report per charter §6.4")
    log(f"{stage}: passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--workers-holdout", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--interior-threshold", type=float, default=0.05)
    parser.add_argument("--skip-gate", action="store_true",
                        help="Stop after data + train; skip Stage 3 gate / Stage 4 pytest.")
    args = parser.parse_args()

    ck = Path(args.ckpt_dir)
    corpus = ck / "ceiling_corpus.npz"
    holdout = ck / "ceiling_holdout_interior.npz"
    out_dir = ck / "gen_ceiling_h192"
    out_targets = ck / "gen_ceiling_h192_targets.npz"

    t0 = time.time()
    avail0, swap0 = mem_pressure_gib()
    log(f"START  MemAvail={avail0:.1f} GiB SwapUsed={swap0:.1f} GiB  corpus→{corpus.name}")

    # ── Stage 1 — corpus (chunked, sequential, OOM-guarded) → merge ──
    # Skip the whole ~31h regen if a valid corpus is already in place. The
    # Phase-G corpus is deterministic (exact LP, no RNG), so a prepared
    # ceiling_corpus.npz — e.g. the existing phase_g_targets.npz — is bit-for-bit
    # what a regen would produce; reusing it also isolates the net-width variable.
    if corpus_ok(corpus):
        log(f"Stage 1: {corpus.name} already present "
            f"({len(load_targets_as_records(corpus))} records) — skipping corpus regen")
    else:
        chunk_paths = []
        for name, baku in CHUNKS:
            cp = ck / f"ceiling_corpus_{name}.npz"
            run_corpus_chunk(name, baku, cp)
            chunk_paths.append(cp)
        merge_corpus(chunk_paths, corpus)

    # ── Stage 2 — interior-aware held-out ruler ──
    if holdout_has_interior(holdout):
        log(f"Stage 2: {holdout.name} already has interior source — skip")
    else:
        run_subprocess([PY, "scripts/run_ceiling_holdout.py",
                        "--out", str(holdout), "--workers", str(args.workers_holdout)],
                       "Stage 2 holdout")

    if args.skip_gate:
        log(f"Data ready (corpus + holdout); --skip-gate set. Elapsed {(time.time()-t0)/3600:.2f}h")
        return 0

    # ── Stage 3 — train hidden=192 + calibration/AlphaZero gate ──
    run_subprocess([
        PY, "scripts/run_gen_iteration.py",
        "--in-checkpoint", str(ck / "gen_phaseG_iter3" / "best.pt"),
        "--out-dir", str(out_dir),
        "--out-targets", str(out_targets),
        "--anchor-targets", str(corpus),
        "--held-out-targets", str(holdout),
        "--hidden-dim", "192",
        "--iterations", str(args.iterations),
        "--epochs", str(args.epochs),
        "--tablebase-replicate", "30",
        "--prev-gen-holdout-mse", "0.05934",
        "--per-source-mse-threshold", f"tablebase_interior:{args.interior_threshold}",
        "--per-source-mse-threshold", "exact_horizon_2:0.05",
        "--per-source-mse-threshold", "exact_horizon_3:0.05",
    ], "Stage 3 train+gate")

    # ── Stage 4 — global invariant gate (§4) ──
    run_subprocess([
        PY, "-m", "pytest",
        "tests/test_cfr_firewall.py", "tests/test_cfr_rigorous_exact.py",
        "tests/test_cfr_selective_search.py", "tests/test_cfr_tablebase.py",
        "tests/test_calibration.py", "tests/test_audit_pack.py", "-q",
    ], "Stage 4 global gate")

    log(f"CEILING RUN COMPLETE — all gates passed. Elapsed {(time.time()-t0)/3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
