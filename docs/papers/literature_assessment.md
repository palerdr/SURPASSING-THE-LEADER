# Literature Assessment

## Current Verdict

The solver is directionally aligned with modern game-solving systems, but its
promotion history shows that held-out calibration alone is not enough. The
current machinery is strongest where it follows CFR/DeepStack-style rigor:
exact terminal utilities, audited frontier values, certified exploitability
intervals, and live promotion gates. The next progress step should be bounded
critical-state subgame resolve, because recent candidates improved calibration
while regressing against `pattern_reader`.

## What Is Rigorous

- The exact CFR/minimax tower uses exact-second public states, engine-derived
  chance, terminal-only utility, and LP minimax solves.
- The firewall tests prevent abstraction and shaping imports from entering the
  rigorous recursion.
- Tablebase and Tier A intervals act like small endgame databases: they tighten
  frontier values without changing the exact recursion.
- Promotion now requires three independent signals: held-out ruler improvement,
  deterministic ladder non-regression, and certified exploitability
  non-regression.

## Where The Repo Matches The Literature

- **AlphaZero / AlphaGo Zero:** the repo has the generate -> train -> evaluate
  loop, a compact value/policy net, and MCTS-backed target generation. The
  difference is that this game is simultaneous and tactically brittle, so the
  acceptance gate must include exploitability and scripted live opponents.
- **DeepStack / Libratus / ReBeL:** the repo already separates a blueprint
  evaluator from local search, and it has a `subgame_resolve_at_critical` seam.
  The failed unbounded critical-resolve run shows that this idea needs bounded
  target selection and progress visibility before becoming an overnight job.
- **CFR / CFR+:** the exact tower is consistent with regret/exploitability
  reasoning, even though it uses finite-horizon LP minimax and selective search
  rather than a monolithic CFR training run.
- **Simultaneous-move MCTS:** the agent plays average root strategies, which is
  important because final small-budget LP roots can collapse to exploitable pure
  choices.
- **Stockfish / NNUE:** the repository mirrors the separation of search and fast
  evaluator. It does not yet have Stockfish-like depth/budget discipline for
  every tactical branch; bounded critical resolve is the next step in that
  direction.

## Gaps And Risks

- Calibration-improving fine-tunes can still hurt live strength. This happened
  repeatedly with d1-Hal, d0-only, plain MCTS-refresh, and value-head
  trust-region candidates.
- `pattern_reader` is the current canary for policies that become too readable
  or misvalue repeated timing patterns.
- Unbounded critical subgame resolve is too expensive: the first attempt ran for
  four hours without producing a target corpus.
- The root package layout is import-stable but not fully packaged; moving code
  under a new package root should be a separate migration after UV is in place.

## Required Next Validation

The next generation run should use:

```powershell
python scripts/run_gen_iteration.py --subgame-resolve-at-critical --bootstrap-critical-only --bootstrap-max-states 24 ...
```

Accept it only if:

- held-out MSE strictly beats `0.055905345689954`,
- the 200-iteration `pattern_reader` diagnostic is non-regressing,
- the deterministic 240-game checkpoint ladder is non-regressing overall and by
  opponent,
- certified exploitability has zero candidate-certified-worse cases.

This is the smallest next experiment that directly tests the DeepStack/Libratus
claim: deeper local solving at high-stakes states should improve tactical
robustness without broad policy drift.
