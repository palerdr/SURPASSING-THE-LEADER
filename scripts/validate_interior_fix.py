"""Fast validation of the Phase I-2 interior-balancing fix.

Reuses the existing ceiling corpus's (deterministic, ~2h) MCTS bootstrap instead
of regenerating it: splits the 3 interior pins out of the lumped `tablebase`
class, replicates them up to the boundary count, weights them 2x, retrains
hidden=192 with light weight_decay, and re-gates on the CLEAN grafted ruler.
~15 min vs the full ~2.5h pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.bootstrap_loop import (
    BootstrapConfig,
    CalibrationGateError,
    calibration_check,
    enforce_calibration_gate,
)
from training.train_value_net import TrainConfig, train
from training.value_targets import (
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_TABLEBASE,
    SOURCE_TABLEBASE_INTERIOR,
    SOURCE_TERMINAL,
    load_targets_as_records,
    save_targets,
    source_breakdown,
)
from dataclasses import replace

CORPUS = "checkpoints/gen_ceiling_h192_targets.npz"
HOLDOUT = "checkpoints/ceiling_holdout_clean.npz"
OUT_CORPUS = "checkpoints/gen_ceiling_h192_v2_targets.npz"
OUT_DIR = "checkpoints/gen_ceiling_h192_v2"
INTERIOR_WEIGHT = 2.0
WEIGHT_DECAY = 1e-4


def main() -> int:
    recs = load_targets_as_records(CORPUS)
    # Interior pins were lumped into `tablebase` (|value| < 0.99 distinguishes them
    # from the ±1 boundary pins) and replicated 30x. Pull them out, dedupe to the
    # unique pins, relabel, and re-replicate up to the boundary count.
    boundary = [r for r in recs if r.source == SOURCE_TABLEBASE and abs(r.value) >= 0.99]
    interior_raw = [r for r in recs if r.source == SOURCE_TABLEBASE and abs(r.value) < 0.99]
    others = [r for r in recs if r.source != SOURCE_TABLEBASE]

    seen = {}
    for r in interior_raw:
        seen.setdefault(round(r.value, 6), r)
    interior_unique = [replace(r, source=SOURCE_TABLEBASE_INTERIOR) for r in seen.values()]
    print(f"interior pins found: {len(interior_unique)} unique values "
          f"{[round(r.value, 3) for r in interior_unique]}")

    n_boundary = len(boundary)
    interior_replicate = max(1, round(n_boundary / len(interior_unique)))
    interior = interior_unique * interior_replicate
    print(f"boundary={n_boundary}  interior {len(interior_unique)}×{interior_replicate}={len(interior)} "
          f"(weight {INTERIOR_WEIGHT}), weight_decay={WEIGHT_DECAY}")

    corpus = others + boundary + interior
    save_targets(corpus, OUT_CORPUS)
    print(f"corpus → {OUT_CORPUS}: {len(corpus)} records {source_breakdown(corpus)}")

    cfg = TrainConfig(
        epochs=150, seed=0, device="cpu", hidden_dim=192, weight_decay=WEIGHT_DECAY,
        source_weights=(
            (SOURCE_TERMINAL, 30.0),
            (SOURCE_EXACT_HORIZON_2, 10.0),
            (SOURCE_EXACT_HORIZON_3, 10.0),
            (SOURCE_TABLEBASE_INTERIOR, INTERIOR_WEIGHT),
        ),
        early_stopping_patience=40,
    )
    tr = train(OUT_CORPUS, OUT_DIR, cfg)
    print(f"trained: best val MSE {tr.best_val_mse:.5f} @ epoch {tr.best_epoch}")
    print(f"  per-source val: {tr.best_per_source_mse}")

    ho = load_targets_as_records(HOLDOUT)
    report = calibration_check(checkpoint_path=tr.checkpoint_path, held_out_targets=ho, device="cpu")
    print(f"\nCLEAN-ruler held-out: overall {report.overall_mse:.5f}")
    print(f"  per-source: { {k: round(v,5) for k,v in report.mse_per_source.items()} }")
    print("  baseline h128 reference: overall 0.06986, interior 0.89447, tablebase 0.00666")

    gate = BootstrapConfig(
        tablebase_mse_threshold=0.01,
        per_source_mse_thresholds={
            SOURCE_TABLEBASE_INTERIOR: 0.05,
            SOURCE_EXACT_HORIZON_2: 0.05,
            SOURCE_EXACT_HORIZON_3: 0.05,
        },
        required_sources=(SOURCE_TERMINAL, SOURCE_TABLEBASE, SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3),
    )
    try:
        enforce_calibration_gate(report, gate)
        print("\nGATE PASSED")
        return 0
    except CalibrationGateError as e:
        print(f"\nGATE FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
