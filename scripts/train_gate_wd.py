"""Train hidden=192 on the prebuilt interior-balanced corpus at a given
weight_decay, then gate on the clean ruler. Used to close the 9e-6 tablebase
miss without rebuilding the corpus."""

from __future__ import annotations

import argparse
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
)

CORPUS = "checkpoints/gen_ceiling_h192_v2_targets.npz"
HOLDOUT = "checkpoints/ceiling_holdout_clean.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--interior-weight", type=float, default=2.0)
    ap.add_argument("--tablebase-weight", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    cfg = TrainConfig(
        epochs=150, seed=0, device="cpu", hidden_dim=192, weight_decay=a.weight_decay,
        source_weights=(
            (SOURCE_TERMINAL, 30.0),
            (SOURCE_EXACT_HORIZON_2, 10.0),
            (SOURCE_EXACT_HORIZON_3, 10.0),
            (SOURCE_TABLEBASE, a.tablebase_weight),
            (SOURCE_TABLEBASE_INTERIOR, a.interior_weight),
        ),
        early_stopping_patience=40,
    )
    tr = train(CORPUS, f"checkpoints/gen_ceiling_{a.tag}", cfg)
    ho = load_targets_as_records(HOLDOUT)
    r = calibration_check(checkpoint_path=tr.checkpoint_path, held_out_targets=ho, device="cpu")
    ps = {k: round(v, 5) for k, v in r.mse_per_source.items()}
    beats = "BEATS" if r.overall_mse < 0.06986 else "above"
    print(f"[{a.tag}] wd={a.weight_decay} iw={a.interior_weight} tbw={a.tablebase_weight} → "
          f"overall {r.overall_mse:.5f} ({beats} baseline 0.06986) | {ps}", flush=True)

    gate = BootstrapConfig(
        tablebase_mse_threshold=0.01,
        max_unresolved_per_source=0.35,
        per_source_mse_thresholds={
            SOURCE_TABLEBASE_INTERIOR: 0.05,
            SOURCE_EXACT_HORIZON_2: 0.05,
            SOURCE_EXACT_HORIZON_3: 0.05,
        },
        required_sources=(SOURCE_TERMINAL, SOURCE_TABLEBASE, SOURCE_EXACT_HORIZON_2, SOURCE_EXACT_HORIZON_3),
    )
    try:
        enforce_calibration_gate(r, gate)
        print(f"[{a.tag}] GATE PASSED ✓", flush=True)
        return 0
    except CalibrationGateError as e:
        print(f"[{a.tag}] GATE FAILED: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
