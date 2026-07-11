# Literature Assessment

## Current Verdict

The exact/search foundation is directionally aligned with modern game-solving
systems, but the current repository does not yet have a closed AlphaZero
reinforcement-learning loop. It has two disconnected approximations:

- MCTS expert iteration over a constructed state grid; and
- legacy bucketed `CanonicalHal` outcome training against scripted opponents.

The audited bridge is now `docs/REGEN2RL.md`. It connects the exact oracle,
simultaneous-move MCTS, reconstructable replay, terminal-outcome self-play,
policy/value training, and statistical promotion in Python before any new Rust
kernel work.

## What Is Rigorous

- The exact minimax tower uses literal-second public states, engine-derived
  chance, terminal-only utility, and LP minimax solves.
- Firewall tests prevent abstraction and shaping imports from entering the
  rigorous recursion.
- Tablebase and Tier A intervals tighten declared frontiers without changing
  exact engine transitions.
- Finite-depth best-response intervals are certified for their stated horizon,
  support mass, and frontier bracket. They are not global exploitability
  certificates.

## Where the Repository Matches the Literature

- **AlphaZero / AlphaGo Zero:** the model already has value and policy heads,
  the learner has masked cross-entropy and value loss, and search can produce
  role policies. The missing step is genuine MCTS self-play whose visited-state
  policies and completed outcomes feed replay.
- **Simultaneous-move MCTS:** the playable solver samples average root role
  strategies. This matches the need for mixed play, but the current PUCT-like
  matrix rule does not automatically inherit the convergence theorem for
  Hannan-consistent selection. `REGEN2RL` therefore requires exact-matrix
  budget curves and saddle-gap audits.
- **DeepStack / Libratus / ReBeL:** the repository separates a blueprint
  evaluator from bounded local resolve. This is useful for critical-state
  hardening, but it supplements rather than replaces the self-play loop.
- **CFR / CFR+:** LP remains the certification path, while Python CFR+ is a
  bounded local runtime option. Literal-second actions remain intact.
- **Stockfish / NNUE:** search/evaluator separation is a useful systems analogy.
  It is not evidence that the game is globally solved.

## Gaps and Risks

- The audit found that the former Hydra self-play command emitted flags its
  parser did not accept; distillation D1 removed that misleading surface until
  P5 supplies the real joint-MCTS actor.
- The module named self-play does not call matrix-game MCTS and only improves
  Hal against scripted opponents.
- Static MCTS bootstrap values are not AlphaZero terminal outcomes.
- Replay rows lack exact-state reconstruction, schema versions, and producing
  checkpoint/search hashes.
- The 23-float feature map can collapse distinct exact states.
- The engine permits arbitrarily long safe lines, so RL needs an explicit
  episode-cap and truncation-sensitivity contract.
- Candidate actions, finite MCTS budgets, and learned frontiers each add a
  separate approximation that must be measured.
- Calibration-improving fine-tunes have regressed against `pattern_reader`;
  arena, policy, and finite-depth best-response evidence must remain mandatory.

## Required Next Validation

Do not start another long generation from the old scripts. Complete the
`REGEN2RL` dependency chain in order:

1. freeze the finite training-game and provenance contracts;
2. certify Python engine/exact/search conformance;
3. add reconstructable replay and grouped splits;
4. validate the simultaneous MCTS improvement operator;
5. train the exact-anchor generation-zero checkpoint; and
6. pass the four-game MCTS self-play smoke before scaling.

Promotion remains a multi-signal decision. Exact calibration, search
conformance, paired arena/SPRT, scripted robustness, policy saddle gaps, and
finite-depth best-response intervals must identify the same candidate and
champion artifacts. Overlapping intervals are inconclusive, and no empirical
gate is a proof of full-game equilibrium.
