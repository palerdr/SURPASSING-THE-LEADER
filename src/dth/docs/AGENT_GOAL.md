# Completed DTH agent contract

Pure DTH is fully solved. The play-time agent is not a bounded resolver or a
learned approximation: `dth.agent.CompleteDTHAgent` reads the canonical
`complete_full_v1` packed tablebase and reconstructs a certified equilibrium
policy for every arena state.

## Canonical artifact

- Directory: `src/dth/artifacts/complete_full_v1/`
- Schema: `dth.complete-tablebase.v1`
- Quotient classes: 289,374,121
- Values: one float64 per class
- Routing audit: one uint8 solver-kind value per class
- Certificate gate: saddle gap at most `1e-6`
- Frozen rules: source-derived DTH schema hash and the repository-wide revival
  model in `docs/REVIVAL_MODEL.md`

The artifact is generated and therefore gitignored. Its manifest binds array
digests, class encoding, rule hash, sweep configuration, and Python/Rust
execution provenance. Missing, corrupt, wrong-schema, or off-domain lookups
fail closed.

## Arena behavior

`python -m arena play` selects DTH by default. Arena projects the canonical
STL role-relative state directly onto the literal-second DTH coordinates and
uses the tablebase policy for actions 1..60. STL remains the referee and owns
all state transitions, revival rolls, clocks, and leap legality. The only
prospective rule the DTH strategy does not contain is Baku's Dropper action 61
inside the public leap window.

There is no play-time compatibility artifact, checkpoint leaf evaluator, or
finite-horizon fallback. Research code may train
or evaluate approximators, but it cannot replace or silently stand in for the
completed artifact in arena play.

## Research status

Dataset generation, neural training, self-play, CFR, and MCTS remain available
for compression, explanation, robustness studies, and comparisons. Their
outputs are experimental measurements downstream of the exact solution. They
do not establish DTH values or policy authority.

## Validation

The complete sweep, packed codec, independent dead-band anchor, Bellman
recertification, resume behavior, and Python/Rust byte parity are locked by the
tests named `test_complete_*` and the contract in
[`DTH_COMPLETE_PARITY.md`](DTH_COMPLETE_PARITY.md).
