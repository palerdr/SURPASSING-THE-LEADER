"""Phase 9.0 pilot driver: validate that the asymmetric-death and
extended-CPR axes carry training signal *before* paying for the wider
grid expansion in Phase G.

Generates two corpora:
  A: legacy symmetric grid (deaths_grid=(0,1), cpr_grid=(0,5))
  B: asymmetric + extended (baku_deaths_grid=(0,1), hal_deaths_grid=(0,1),
                            cpr_grid=(0,5,10))

For each, reports source breakdown and corpus diagnostics. For B,
additionally reports feature collisions, per-axis coverage, and
value-signal per axis.

Acceptance criteria (gates whether Phase G is worth pursuing):

  1. corpus B has measurably more records per axis than A on the
     death + CPR axes (basic sanity — more states are reachable).
  2. at least one of:
     a. asymmetric death states have value spread > 0.05 vs symmetric
        counterparts at the same (cylinder, clock) cells.
     b. high-CPR states (>= 6) produce distinct training labels from
        their /6.0-scaled feature counterparts.
     c. axis_carries_signal returns True for baku_deaths, hal_deaths,
        or cprs.

Run:
    python scripts/run_corpus_validation.py \
        --out-dir checkpoints/pilot/ \
        --workers 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.corpus_diagnostics import (
    axis_carries_signal,
    feature_collision_report,
    per_axis_coverage,
    summarize_corpus,
    value_distribution_per_axis,
)
from training.value_targets import (
    generate_targets,
    save_targets,
    source_breakdown,
)

# Smaller cylinder/clock grids than production to keep the pilot fast
# (~30-60 min on 14 workers). Both corpora share these axes so the
# only difference is deaths + cprs.
PILOT_BAKU_CYL = (180.0, 240.0, 290.0, 299.0)
PILOT_HAL_CYL = (0.0, 120.0, 240.0)
PILOT_CLOCK = (720.0, 2000.0, 3540.0)
PILOT_HALF = (1, 2)

# Corpus A — legacy symmetric (matches Phase 8 defaults shape)
A_DEATHS = (0, 1)
A_CPRS = (0, 5)

# Corpus B — asymmetric + extended CPR (uses the bc370f3 fixes)
B_BAKU_DEATHS = (0, 1)
B_HAL_DEATHS = (0, 1)
B_CPRS = (0, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="checkpoints/pilot/")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Generating corpus A (legacy symmetric)", flush=True)
    t0 = time.time()
    corpus_a = generate_targets(
        baku_cylinder_grid=PILOT_BAKU_CYL,
        hal_cylinder_grid=PILOT_HAL_CYL,
        clock_grid=PILOT_CLOCK,
        half_grid=PILOT_HALF,
        deaths_grid=A_DEATHS,
        cpr_grid=A_CPRS,
        workers=args.workers,
    )
    t_a = time.time() - t0
    save_targets(corpus_a, out_dir / "corpus_a.npz")
    print(f"  Corpus A: {len(corpus_a)} records in {t_a:.1f}s", flush=True)
    print(f"  Source breakdown: {source_breakdown(corpus_a)}", flush=True)

    print(f"[2/4] Generating corpus B (asymmetric + extended CPR)", flush=True)
    t0 = time.time()
    corpus_b = generate_targets(
        baku_cylinder_grid=PILOT_BAKU_CYL,
        hal_cylinder_grid=PILOT_HAL_CYL,
        clock_grid=PILOT_CLOCK,
        half_grid=PILOT_HALF,
        baku_deaths_grid=B_BAKU_DEATHS,
        hal_deaths_grid=B_HAL_DEATHS,
        cpr_grid=B_CPRS,
        workers=args.workers,
    )
    t_b = time.time() - t0
    save_targets(corpus_b, out_dir / "corpus_b.npz")
    print(f"  Corpus B: {len(corpus_b)} records in {t_b:.1f}s", flush=True)
    print(f"  Source breakdown: {source_breakdown(corpus_b)}", flush=True)

    print(f"[3/4] Diagnostic reports", flush=True)
    summary_a = summarize_corpus(corpus_a)
    summary_b = summarize_corpus(corpus_b)

    print(f"  Corpus A summary:", flush=True)
    print(f"    n_records: {summary_a['n_records']}", flush=True)
    print(f"    axes_with_signal: {summary_a['axes_with_signal']}", flush=True)
    print(f"    divergent_collisions: {summary_a['divergent_collision_count']}", flush=True)

    print(f"  Corpus B summary:", flush=True)
    print(f"    n_records: {summary_b['n_records']}", flush=True)
    print(f"    axes_with_signal: {summary_b['axes_with_signal']}", flush=True)
    print(f"    divergent_collisions: {summary_b['divergent_collision_count']}", flush=True)

    # B is the corpus under test — deeper inspection
    coverage_b = per_axis_coverage(
        corpus_b, axes=("baku_deaths", "hal_deaths", "cprs")
    )
    distribution_b = value_distribution_per_axis(
        corpus_b, axes=("baku_deaths", "hal_deaths", "cprs")
    )
    print(f"  Corpus B coverage on new axes:", flush=True)
    for axis, bins in coverage_b.items():
        print(f"    {axis}: {dict(sorted(bins.items()))}", flush=True)
    print(f"  Corpus B value distribution on new axes:", flush=True)
    for axis, bin_stats in distribution_b.items():
        for bin_key in sorted(bin_stats):
            mean, stddev, count = bin_stats[bin_key]
            print(
                f"    {axis}[{bin_key}]: n={count} mean={mean:+.3f} stddev={stddev:.3f}",
                flush=True,
            )

    # Collision details for B (asymmetric death is where surprises live)
    collisions_b = feature_collision_report(corpus_b, only_divergent=True)
    if collisions_b:
        print(
            f"  Corpus B has {len(collisions_b)} divergent collision groups "
            f"(largest size {max(len(g.record_indices) for g in collisions_b)})",
            flush=True,
        )
        # Show top 3 most-collided groups
        for g in collisions_b[:3]:
            vals = sorted(set(round(v, 3) for v in g.values))
            print(f"    Hash {g.feature_hash[:80]}...  values={vals}", flush=True)
    else:
        print(f"  Corpus B has no divergent collisions ✅", flush=True)

    print(f"[4/4] Acceptance criteria", flush=True)
    death_signal = (
        axis_carries_signal(distribution_b["baku_deaths"])
        or axis_carries_signal(distribution_b["hal_deaths"])
    )
    cpr_signal = axis_carries_signal(distribution_b["cprs"])
    asymmetric_states_reached = (
        coverage_b["baku_deaths"].get(0, 0) > 0
        and coverage_b["baku_deaths"].get(1, 0) > 0
        and coverage_b["hal_deaths"].get(0, 0) > 0
        and coverage_b["hal_deaths"].get(1, 0) > 0
    )
    high_cpr_states_reached = coverage_b["cprs"].get(10, 0) > 0

    verdict = {
        "asymmetric_death_states_reached": asymmetric_states_reached,
        "high_cpr_states_reached": high_cpr_states_reached,
        "baku_or_hal_deaths_carry_value_signal": death_signal,
        "cprs_carries_value_signal": cpr_signal,
        "any_new_axis_carries_signal": death_signal or cpr_signal,
        "phase_g_recommended": (asymmetric_states_reached and high_cpr_states_reached)
        and (death_signal or cpr_signal),
    }
    print(f"  Verdict: {json.dumps(verdict, indent=2)}", flush=True)

    # Persist for posterity
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "corpus_a_summary": summary_a,
                "corpus_b_summary": summary_b,
                "corpus_b_coverage_new_axes": {
                    axis: {str(k): v for k, v in bins.items()}
                    for axis, bins in coverage_b.items()
                },
                "corpus_b_value_distribution_new_axes": {
                    axis: {
                        str(k): {"mean": m, "stddev": s, "count": c}
                        for k, (m, s, c) in bin_stats.items()
                    }
                    for axis, bin_stats in distribution_b.items()
                },
                "verdict": verdict,
                "elapsed_seconds": {"corpus_a": t_a, "corpus_b": t_b},
            },
            f,
            indent=2,
        )
    print(f"  Wrote: {report_path}", flush=True)

    if verdict["phase_g_recommended"]:
        print(
            "\nPhase G (stratified wider grid) RECOMMENDED — new axes carry signal.",
            flush=True,
        )
        return 0
    else:
        print(
            "\nPhase G NOT YET justified by this pilot. Recommend skipping to Phase F "
            "(registry expansion) and revisiting G later with a broader pilot.",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
