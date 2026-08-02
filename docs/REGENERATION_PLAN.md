# Frozen revival migration record

The repository-wide migration to [`REVIVAL_MODEL.md`](REVIVAL_MODEL.md) is
complete.

## Result

- DTH, abstract, Python STL, Rust STL, and OCaml STL evaluate the same
  `0.95 * (1 - s/240) * 0.75^(t/60)` surface and the same eligibility guards.
- Identity physicality and accumulated revival attempts are not probability
  inputs. Historical counters may remain in match records or learned feature
  schemas, but changing them cannot change a transition probability.
- The TUI layers display the engine's recorded probability; they do not own a
  model.
- The frozen surface is bound into DTH and abstract artifact schemas.

## Artifact outcome

The completed DTH solution is `complete_full_v1`, a dense
289,374,121-class quotient sweep. Earlier partial-solve machinery is not part
of the production contract.
Optional dataset, training, self-play, CFR, and MCTS tooling remains available
as research downstream of the exact solution.

## Structural invariant

The migration changed probabilities but not the eligibility zero-set.
Reachability, quotient counts, and potential ordering therefore remain the
same. Value-bearing artifacts are valid only when their manifest binds the
frozen rule hash; mismatches fail closed.
