"""Build the clean F-2-expanded held-out ruler for the I-2 re-gate.

The Stage-2 *regenerated* ruler (`ceiling_holdout_interior.npz`) diverged from the
trusted baseline ruler in its exact_horizon_3 set (64 → 80 states; the 16 extra
carry ~1.0 MSE for both nets and inflate the headline). Investigation showed the
divergence is NOT an unresolved-probability artifact — the trusted ruler itself is
full of unresolved>0.35 states and scores cleanly — so filtering on unresolved is
the wrong tool (it would delete half the trusted ruler).

Correct fix: GRAFT. Take the trusted baseline ruler verbatim (the exact 235 states
that defined held-out MSE 0.05934) and add only the 3 Phase-F-2 interior anchors
(exact pinned values, unresolved=0) relabeled `tablebase_interior`. The result is
trustworthy AND directly comparable to the 0.05934 baseline.

Usage:
    python scripts/make_clean_holdout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.value_targets import (
    SOURCE_TABLEBASE_INTERIOR,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)

TRUSTED = "checkpoints/holdout_v2_targets.npz"            # defined the 0.05934 baseline
INTERIOR_SRC = "checkpoints/ceiling_holdout_interior.npz"  # source of the 3 exact anchors
OUT = "checkpoints/ceiling_holdout_clean.npz"


def main() -> int:
    trusted = load_targets_as_records(TRUSTED)
    interior = [r for r in load_targets_as_records(INTERIOR_SRC)
                if r.source == SOURCE_TABLEBASE_INTERIOR]
    if len(interior) != 3:
        raise SystemExit(f"expected 3 interior anchors, found {len(interior)} — abort")

    clean = trusted + interior
    save_targets(clean, OUT)
    print(f"trusted ruler:   {len(trusted)} records  {source_breakdown(trusted)}")
    print(f"interior anchors: {len(interior)} (exact, unresolved=0)")
    print(f"clean ruler →    {OUT}  {len(clean)} records  {source_breakdown(clean)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
