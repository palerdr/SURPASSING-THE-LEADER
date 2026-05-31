"""Phase H: tiered reanalysis of a Phase-G rejected pool.

Loads a rejected pool (the ``{corpus}.npz.rejected.npz`` written by
``run_phase_g_corpus.py`` / ``generate_targets``), re-solves each high-variance
state with the two-tier deepening in ``training.reanalysis`` (exact-deepen, then
high-iter MCTS fallback), audits the exact tier against the full-width oracle
(charter gates G5 + G7), and writes an **additive** reanalysis corpus.

The output is a NEW .npz — it never overwrites the Phase-G corpus. Merge it into
the training set at I-2 time via ``load_targets_as_records`` + ``save_targets``.

Usage:
    python scripts/run_phase_h_reanalysis.py \
        --rejected-pool checkpoints/phase_g_targets.npz.rejected.npz \
        --out checkpoints/phase_h_reanalysis.npz \
        --exact-horizon 4 --mcts-iters 4000 --audit-sample 16 --workers 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.cfr.exact import ExactSearchConfig, clear_solve_cache
from training.reanalysis import (
    SOURCE_EXACT_HORIZON_4,
    SOURCE_REANALYSIS_MCTS,
    audit_reanalysis,
    reanalyze_pool,
)
from training.value_targets import load_rejected_pool, save_targets, source_breakdown


def _load_evaluator(checkpoint: str | None):
    """Optional trained-net leaf evaluator for the MCTS tier. None → terminal-only."""
    if not checkpoint:
        return None
    from environment.cfr.evaluator import ValueNetEvaluator
    from training.train_value_net import load_checkpoint, make_predict_fn

    model = load_checkpoint(checkpoint)
    return ValueNetEvaluator(model_fn=make_predict_fn(model))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejected-pool", required=True, help="Input rejected-pool .npz.")
    parser.add_argument("--out", default="checkpoints/phase_h_reanalysis.npz", help="Additive output corpus.")
    parser.add_argument("--exact-horizon", type=int, default=4)
    parser.add_argument("--accept-unresolved", type=float, default=0.05,
                        help="Tier-1 success ceiling on residual unresolved mass.")
    parser.add_argument("--mcts-iters", type=int, default=4000)
    parser.add_argument("--audit-sample", type=int, default=16,
                        help="States to audit full-width for G5/G7 (sampled; cost is full-width depth).")
    parser.add_argument("--limit", type=int, default=None, help="Reanalyze only the first N pool entries.")
    parser.add_argument("--checkpoint", default=None, help="Optional value-net checkpoint for the MCTS tier.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = ExactSearchConfig()
    pool = load_rejected_pool(args.rejected_pool)
    print(f"Loaded rejected pool: {len(pool)} states from {args.rejected_pool}", flush=True)
    if args.limit is not None:
        print(f"  limiting to first {args.limit}", flush=True)

    evaluator = _load_evaluator(args.checkpoint)
    print(f"  MCTS-tier evaluator: {'value-net ' + args.checkpoint if evaluator else 'terminal-only'}", flush=True)

    # ── Gate G5/G7: audit the exact tier BEFORE trusting the corpus ──
    clear_solve_cache()
    t0 = time.time()
    audit = audit_reanalysis(
        pool if args.limit is None else pool[: args.limit],
        config=config,
        exact_horizon=args.exact_horizon,
        accept_unresolved=args.accept_unresolved,
        sample=args.audit_sample,
    )
    print(
        f"\nAUDIT (G5/G7): sampled={audit.sampled} of {audit.exact_tier_total} exact-tier rescues | "
        f"max_value_gap={audit.max_value_gap:.2e} (<= {0.05}) | max_nash_gap={audit.max_nash_gap:.2e} | "
        f"passed={audit.passed}",
        flush=True,
    )
    if not audit.passed:
        print("GATE FAILED — exact tier diverged from the full-width oracle:", flush=True)
        for f in audit.failures:
            print(f"  - {f}", flush=True)
        return 1

    # ── Reanalysis ──
    clear_solve_cache()
    outcomes = reanalyze_pool(
        pool,
        config=config,
        exact_horizon=args.exact_horizon,
        accept_unresolved=args.accept_unresolved,
        mcts_iters=args.mcts_iters,
        evaluator=evaluator,
        seed=args.seed,
        limit=args.limit,
    )
    elapsed = time.time() - t0

    targets = [o.target for o in outcomes]
    n_exact = sum(1 for o in outcomes if o.tier == SOURCE_EXACT_HORIZON_4)
    n_mcts = sum(1 for o in outcomes if o.tier == SOURCE_REANALYSIS_MCTS)
    rescue_rate = (n_exact / len(outcomes) * 100) if outcomes else 0.0

    save_targets(targets, args.out)
    print(
        f"\nPhase H complete in {elapsed:.1f}s.\n"
        f"  Reanalyzed: {len(outcomes)} states\n"
        f"  Exact-tier rescues (SOURCE_EXACT_HORIZON_4): {n_exact} ({rescue_rate:.1f}%)\n"
        f"  MCTS fallback (SOURCE_REANALYSIS_MCTS): {n_mcts}\n"
        f"  Source breakdown: {source_breakdown(targets)}\n"
        f"  Additive corpus written to: {args.out}\n"
        f"  (Merge into training at I-2 via load_targets_as_records + save_targets.)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
