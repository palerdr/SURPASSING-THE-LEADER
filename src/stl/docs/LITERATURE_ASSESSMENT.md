# STL solver and learning claim boundary

## Current position

Pure DTH at ladder rung L1 is the repository's completed exact solution and the
default policy provider for arena play. The STL project owns rung L2: the same
public mechanics plus Baku's prospective Dropper action 61 in the leap window.
No complete L2 solve is claimed.

The STL exact, search, learning, evaluation, and promotion modules remain useful
research surfaces. They do not supersede the DTH tablebase and are not a closed
AlphaZero reinforcement-learning system.

## What is rigorous

- The STL engine is the transition and legality authority for the public
  leap-aware game.
- Full-width finite-horizon matrices use literal legal actions, engine-derived
  chance, and LP minimax.
- Candidate search, CFR+, MCTS, learned values, Tier A frontiers, and tablebase
  intervals retain their declared approximation boundaries.
- Finite-depth best-response intervals certify only their named scenario,
  horizon, support, and frontier. They are not global exploitability proofs.

## Algorithmic roles

- **Linear programming** is the matrix-game truth oracle.
- **CFR/CFR+** is a bounded approximate matrix solver and must be audited against
  LP before use in a claim.
- **Simultaneous-move MCTS** produces empirical role marginals from finite search
  budgets; its output is approximate even when its leaf values are exact.
- **Policy/value networks** compress or extend exact anchors but never establish
  their own correctness.
- **DeepStack/ReBeL-style local resolving** is relevant to bounded critical-state
  hardening, not evidence that the full game has been solved.
- **AlphaZero-style training** would require complete MCTS-guided trajectories,
  terminal or explicitly truncated outcomes, replay, reanalysis, and statistical
  promotion. The current project does not claim that closed loop.

## Retained executable surfaces

The current Hydra command configurations cover checkpoint training, saved-corpus
training, evaluation, and report-based promotion. The source tree also retains
finite-depth exploitability measurement, strength gates, exact/search audits,
legacy scripted-opponent play, and learning support modules. Generated reports,
checkpoints, and corpora remain ignored project outputs, not documentation or
proof artifacts.

Historical generation-zero run ledgers and their removed commands are not a
current roadmap. Any future training plan must begin from the present command
surface, identify its rung on `docs/FORMULATION_LADDER.md`, predeclare its gates,
and preserve the exact/approximate/empirical vocabulary of
`docs/FOUNDATIONS.md`.

## Literature library

Primary references live in `docs/papers/references/`: AlphaZero, CFR, CFR+,
DeepStack, ReBeL, and the silent-duel timing paper. The duel paper supplies only
timing-game precedent; its many-player non-constant-sum model is not the DTH or
STL stage game.
