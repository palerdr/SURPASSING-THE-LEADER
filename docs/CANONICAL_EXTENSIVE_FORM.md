# Canonical Extensive-Form Contract

This document freezes repository-wide state and transition boundaries. Action
timing is owned separately by [`ACTION_TIMING.md`](ACTION_TIMING.md), the
revival-probability surface by [`REVIVAL_MODEL.md`](REVIVAL_MODEL.md), and the
set of games this repository claims to model by
[`FORMULATION_LADDER.md`](FORMULATION_LADDER.md).

## Documentary boundary

The cylinder capacity and injection cap follow
[E-CYLINDER-CAP](game-sources/EVIDENCE.md#e-cylinder-cap); the strict cumulative
boundary follows [E-TTD-EXACT](game-sources/EVIDENCE.md#e-ttd-exact) and is
measured in a worked case by
[E-TTD-SLACK](game-sources/EVIDENCE.md#e-ttd-slack); crossing the cylinder limit
follows [E-OVERFLOW](game-sources/EVIDENCE.md#e-overflow); the composition of
the injected dose follows
[E-DOSE-COMPOSITION](game-sources/EVIDENCE.md#e-dose-composition).

<!-- canon:C-DEATH-BOUNDARY -->
## Death and revival

Let `q` be the injected duration and `t` prior cumulative time-to-death (TTD).

1. Cylinder capacity is exactly 300 seconds.
2. Reaching capacity triggers injection; the physical dose is capped at 300.
3. A current dose `q >= 300` has zero revival probability.
4. `t + q > 300` is fatal, while `t + q == 300` remains revival-eligible when
   `q < 300`.
5. Revival uses prior TTD `t`; a surviving successor carries `t + q`, resets
   the cylinder, and increments the relevant death/referee state.
6. The injected dose is the cylinder contents plus a fixed 60-second
   failed-check penalty: `q = s + 60`.

These six boundaries are documentary and hold identically at every rung of the
formulation ladder. Rules 3 and 4 together give the eligibility predicate
`s <= 239 and s + t <= 240` at one-second resolution.

<!-- canon:C-REVIVAL-AUTHORITY -->
## Revival probability

The numerical revival surface is **not** documentary. It is frozen once, for
every formulation, in [`REVIVAL_MODEL.md`](REVIVAL_MODEL.md):

```text
P_rev(s, t) = 0.95 * (1 - s / 240) * 0.75^(t/60)
```

for eligible `(s, t)`, and zero otherwise. The dose factor is linear and
reaches zero exactly at the documentary lethal dose `q = 300`; STL's referee
floor is omitted because it provably cannot bind inside the eligible region.
The surface is exactly bucket-invariant,
which is what lets the bucketed abstractions, pure DTH, and the leap-aware
public game be played from one interface against comparable numbers.

Superseded and no longer permitted anywhere: the cubic dose curve
`1 - (q/300)^3`, DTH's local `2^(-t/240)` factor, and the `abstract` `0.80`
baseline with a stretched-exponential TTD term. No project may carry its own
revival constants, and no project may reintroduce an explicit CPR count or a
per-player physicality multiplier below ladder rung L2.

<!-- canon:C-AUTHORITY -->
## Executable authorities

- `src/stl/engine/game.py` owns full-game transitions.
- `src/stl/engine/actions.py` owns STL action legality.
- `src/crates/stl_solver/src/game.rs` must match the Python engine where covered.
- `src/dth/solver.py` owns the no-leap DTH abstraction (ladder rung L1).
- `src/abstract/rules.py` owns only the enumerated role-relative 10-second and
  5-second bucket abstractions (ladder rung L0); their shared exact contract is
  documented in `src/abstract/docs/MODEL.md`.
- `src/ocaml/lib/engine/` owns the imported leap-aware OCaml engine (ladder rung
  L2). It already uses the two-variable revival surface as a parity target; this
  does not silently reclassify older STL/DTH checkpoints.
- `arena/` owns no game rules; it adapts solver policies to the canonical STL
  referee for live play.

Solvers may use audited scalar helpers but must not reimplement divergent game
rules. Rule-bound artifacts must identify their schema and a **source-derived**
digest: a digest computed from a hand-written description of the rules does not
satisfy this requirement, because it cannot detect a change to the rules it
describes.
