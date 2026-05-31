"""Bounded Phase-G-style corpus regen whose primary purpose is producing a real
**rejected pool** for Phase H reanalysis.

``run_phase_g_corpus.py`` hard-codes the full 7168-candidate grid (a ~31h run).
This driver sweeps a smaller, deliberately contested grid so it finishes in a
background-friendly window while still emitting genuine high-``unresolved``
states (the ones the horizon-2/3 LP can't resolve) into ``{out}.rejected.npz``.
Those are the Phase H input.

Usage:
    python scripts/run_phase_h_regen.py --out checkpoints/phase_h_corpus.npz --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.value_targets import generate_targets, load_rejected_pool, save_targets, source_breakdown

# Contested-leaning grid: low/mid cylinders (where the checker has a safe
# strategy, so lines survive and stay unresolved at shallow horizon) plus
# near-leap clocks (gated to horizon 3). These are exactly the states Phase G
# rejects — the Phase H reanalysis input.
BAKU_CYL = (180.0, 200.0, 220.0, 240.0, 290.0)
HAL_CYL = (0.0, 120.0, 240.0)
CLOCK = (720.0, 3300.0, 3540.0)
HALF = (1, 2)
DEATHS = (0,)
CPR = (0,)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="checkpoints/phase_h_corpus.npz")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    candidate_count = len(BAKU_CYL) * len(HAL_CYL) * len(CLOCK) * len(HALF) * len(DEATHS) * len(CPR)
    print(f"Phase H regen grid: {candidate_count} candidates "
          f"(baku={BAKU_CYL} hal={HAL_CYL} clock={CLOCK})", flush=True)

    rejected_path = f"{args.out}.rejected.npz"
    t0 = time.time()
    targets = generate_targets(
        baku_cylinder_grid=BAKU_CYL,
        hal_cylinder_grid=HAL_CYL,
        clock_grid=CLOCK,
        half_grid=HALF,
        deaths_grid=DEATHS,
        cpr_grid=CPR,
        rejected_pool_path=rejected_path,
        workers=args.workers,
    )
    elapsed = time.time() - t0
    save_targets(targets, args.out)

    rejected = load_rejected_pool(rejected_path)
    print(f"\nPhase H regen complete in {elapsed:.1f}s ({elapsed/3600:.2f}h).", flush=True)
    print(f"  Accepted (LP-labeled) records: {len(targets)}", flush=True)
    print(f"  Source breakdown: {source_breakdown(targets)}", flush=True)
    print(f"  REJECTED pool (Phase H input): {len(rejected)} states -> {rejected_path}", flush=True)
    print(f"  Corpus -> {args.out}", flush=True)
    print(f"\nNext: python scripts/run_phase_h_reanalysis.py "
          f"--rejected-pool {rejected_path} --out checkpoints/phase_h_reanalysis.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
