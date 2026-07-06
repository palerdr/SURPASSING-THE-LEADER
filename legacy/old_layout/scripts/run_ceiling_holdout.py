"""Regenerate the held-out calibration ruler for the I-2 ceiling run.

The May-21 ``holdout_v2_targets.npz`` predates the Phase F-2 interior anchors,
so its tablebase class is 19 all-±1 boundary pins with no interior source — it
cannot test the interior gate (charter success criterion #3). This driver mints
a fresh ruler through ``generate_holdout_targets``, which calls
``_generate_tablebase_targets(split_interior=True)`` and therefore labels the
F-2 interior pins as the distinct ``SOURCE_TABLEBASE_INTERIOR`` class.

The held-out grid (384 candidates, shifted off the Phase-G training grid so the
two never overlap) is small — no OOM risk; the heavy compute is the training
corpus, not this ruler.

Usage:
    python scripts/run_ceiling_holdout.py \
        --out checkpoints/ceiling_holdout_interior.npz --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.value_targets import (
    SOURCE_TABLEBASE_INTERIOR,
    generate_holdout_targets,
    save_targets,
    source_breakdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="checkpoints/ceiling_holdout_interior.npz")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rejected_path = f"{args.out}.rejected.npz"
    t0 = time.time()
    targets = generate_holdout_targets(
        rejected_pool_path=rejected_path,
        workers=args.workers,
    )
    elapsed = time.time() - t0
    save_targets(targets, args.out)

    breakdown = source_breakdown(targets)
    interior_count = breakdown.get(SOURCE_TABLEBASE_INTERIOR, 0)
    print(f"\nHoldout regen complete in {elapsed:.1f}s ({elapsed / 3600:.2f}h).", flush=True)
    print(f"  Records: {len(targets)}", flush=True)
    print(f"  Source breakdown: {breakdown}", flush=True)
    print(f"  Ruler -> {args.out}", flush=True)

    # Defensibility gate: a held-out ruler with no interior source is exactly the
    # blind spot F-2 was built to close. Fail loud rather than silently certify
    # the wider net against a boundary-only ruler.
    if interior_count == 0:
        print(
            f"\nFAIL: no {SOURCE_TABLEBASE_INTERIOR!r} records in the ruler — the "
            "interior gate cannot bite. Refusing to emit a boundary-only holdout.",
            flush=True,
        )
        return 1
    print(f"  Interior anchors present: {interior_count} ({SOURCE_TABLEBASE_INTERIOR})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
