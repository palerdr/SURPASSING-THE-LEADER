"""Benchmark MCTS convergence vs iteration budget on Phase 1 tablebase scenarios.

Runs mcts_search at a schedule of iteration counts on every registered
tablebase scenario, with multiple RNG seeds per cell, and reports the
mean and standard deviation of root_value_for_hal. Useful for deciding
whether Slice 4b (tablebase shortcuts, value net) is the right next
investment vs Slice 4c (transposition cache).

Output is plain-text table on stdout. Run with:

    python scripts/benchmark_mcts_convergence.py

or send to a log file:

    python scripts/benchmark_mcts_convergence.py > /tmp/mcts_bench.txt
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.mcts import MCTSConfig, mcts_search
from environment.cfr.tablebase import REGISTRY


ITERATION_SCHEDULE: tuple[int, ...] = (50, 200, 1000, 2500)
SEEDS: tuple[int, ...] = (0, 1, 2)
EXPLORATION_C: float = 1.0


def _expected_label(scenario) -> str:
    if scenario.expected_value is not None:
        return f"pinned={scenario.expected_value:+.2f}"
    return "relational"


def main() -> None:
    evaluator = TerminalOnlyEvaluator()
    print()
    print(f"{'scenario':<45} | {'iters':>6} | {'avg':>9} | {'std':>8} | {'expected':<14} | {'wall':>7}")
    print("-" * 110)

    for name, factory in REGISTRY.items():
        sample = factory()
        note = _expected_label(sample)

        for n_iter in ITERATION_SCHEDULE:
            t0 = time.time()
            values = []
            for seed in SEEDS:
                scenario = factory()
                cfg = MCTSConfig(
                    iterations=n_iter,
                    exploration_c=EXPLORATION_C,
                    evaluator=None,
                    use_tablebase=False,
                )
                rng = np.random.default_rng(seed)
                result = mcts_search(scenario.game, cfg, evaluator, rng, scenario.config)
                values.append(result.root_value_for_hal)

            avg = float(np.mean(values))
            std = float(np.std(values))
            elapsed = time.time() - t0
            print(
                f"{name:<45} | {n_iter:>6} | {avg:>+9.4f} | {std:>8.4f} | {note:<14} | {elapsed:>6.2f}s",
                flush=True,
            )

        print()


if __name__ == "__main__":
    main()
