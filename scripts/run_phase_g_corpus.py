"""Phase G: stratified wider grid for LP-labeled training corpus.

Phase 9.0's pilot established that asymmetric deaths and high CPR fatigue
don't shift LP equilibrium values at horizons 2-3. This driver focuses
the grid expansion on axes that DO carry signal:

- Cylinder thresholds (densified around 239/240/241 overflow trigger,
  289/290 near-overflow, 295/297/299 critical near-overflow).
- Hal extreme cylinders (200+) included — currently only swept at 0/120/240.
- Denser clock variants spanning opening through leap-window approach.

Skipped (per Phase 9.0 negative finding):
- Asymmetric deaths: kept symmetric (0,0)/(1,1) only.
- High CPR (>=8): kept (0, 5) — values 8-12 don't change h2/h3 LP values.

Estimated yield: 14 × 8 × 8 × 2 × 2 × 2 = 7168 candidates → ~2900 emitted
LP-labeled records (vs current 196), ~15× expansion. Compute: ~8-12 hours
with 14 workers + memoization.

Usage:
    python scripts/run_phase_g_corpus.py --out checkpoints/phase_g_targets.npz --workers 14
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.value_targets import (
    generate_targets,
    save_targets,
    source_breakdown,
)

# Phase G training grid: stratified expansion of cylinder + clock axes.
# Axes are ordered so memoization cache reuses substates across adjacent
# parameter values (e.g. 239 → 240 → 241 share most subtree structure).

PHASE_G_BAKU_CYL_GRID: tuple[float, ...] = (
    0.0, 60.0, 120.0, 180.0, 220.0,              # base coverage
    239.0, 240.0, 241.0,                         # overflow trigger boundary
    270.0, 289.0, 290.0,                         # near-overflow zone
    295.0, 297.0, 299.0,                         # critical near-overflow
)  # 14 values

PHASE_G_HAL_CYL_GRID: tuple[float, ...] = (
    0.0, 60.0, 120.0, 180.0, 220.0, 240.0, 270.0, 290.0,
)  # 8 values; expanded Hal-side coverage (was 0/120/240 only)

PHASE_G_CLOCK_GRID: tuple[float, ...] = (
    720.0, 1500.0, 2000.0, 2500.0, 3000.0, 3300.0, 3450.0, 3540.0,
)  # 8 values; denser sampling through mid-game and pre-leap

PHASE_G_HALF_GRID: tuple[int, ...] = (1, 2)
PHASE_G_DEATHS_GRID: tuple[int, ...] = (0, 1)  # symmetric only
PHASE_G_CPR_GRID: tuple[int, ...] = (0, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="checkpoints/phase_g_targets.npz",
        help="Output .npz path for the LP-labeled corpus.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count. NOTE: total memoization memory scales with "
        "grid size, duplicated across workers. The full 7168-candidate grid at "
        "8 workers exhausts 24 GiB RAM + 16 GiB swap. Chunk the grid via "
        "--baku-cyl to bound peak memory.",
    )
    parser.add_argument(
        "--baku-cyl",
        type=str,
        default=None,
        help="Comma-separated baku_cylinder subset (e.g. '0,60,120,180,220,239,240,241'). "
        "Overrides the full 14-value axis so the grid can be run in memory-bounded "
        "chunks. Merge chunk outputs afterward with load_targets_as_records + save_targets.",
    )
    args = parser.parse_args()

    baku_cyl_grid = PHASE_G_BAKU_CYL_GRID
    if args.baku_cyl is not None:
        baku_cyl_grid = tuple(float(v) for v in args.baku_cyl.split(","))

    candidate_count = (
        len(baku_cyl_grid)
        * len(PHASE_G_HAL_CYL_GRID)
        * len(PHASE_G_CLOCK_GRID)
        * len(PHASE_G_HALF_GRID)
        * len(PHASE_G_DEATHS_GRID)
        * len(PHASE_G_CPR_GRID)
    )
    print(f"Phase G grid: {candidate_count} candidate states", flush=True)
    print(f"  baku_cyl axis ({len(baku_cyl_grid)}): {baku_cyl_grid}", flush=True)
    print(f"  hal_cyl axis ({len(PHASE_G_HAL_CYL_GRID)}): {PHASE_G_HAL_CYL_GRID}", flush=True)
    print(f"  clock axis ({len(PHASE_G_CLOCK_GRID)}): {PHASE_G_CLOCK_GRID}", flush=True)
    print(
        f"Sweeping with {args.workers} workers.",
        flush=True,
    )

    rejected_path = f"{args.out}.rejected.npz"
    t0 = time.time()
    targets = generate_targets(
        baku_cylinder_grid=baku_cyl_grid,
        hal_cylinder_grid=PHASE_G_HAL_CYL_GRID,
        clock_grid=PHASE_G_CLOCK_GRID,
        half_grid=PHASE_G_HALF_GRID,
        deaths_grid=PHASE_G_DEATHS_GRID,
        cpr_grid=PHASE_G_CPR_GRID,
        rejected_pool_path=rejected_path,
        workers=args.workers,
    )
    elapsed = time.time() - t0

    save_targets(targets, args.out)
    breakdown = source_breakdown(targets)

    print(
        f"\nPhase G complete in {elapsed:.1f}s ({elapsed / 3600:.2f}h).",
        flush=True,
    )
    print(f"  Records emitted: {len(targets)}", flush=True)
    print(f"  Source breakdown: {breakdown}", flush=True)
    print(f"  Pass rate: {len(targets) / candidate_count * 100:.1f}%", flush=True)
    print(f"  Saved to: {args.out}", flush=True)
    print(f"  Rejected pool: {rejected_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
