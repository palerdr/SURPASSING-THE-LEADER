# Formulation ladder

This file freezes *which games this repository claims to model*, as a strictly
increasing sequence. Each rung adds exactly one mechanic to the rung below it,
keeps every rule the lower rungs froze, and uses the single revival surface in
[`REVIVAL_MODEL.md`](REVIVAL_MODEL.md).

The point of the ladder is that a rung can be solved, played, and retired
without renegotiating the rules. A formulation not on this ladder is not a
supported claim.

Timing is owned by [`ACTION_TIMING.md`](ACTION_TIMING.md); state and transition
boundaries by [`CANONICAL_EXTENSIVE_FORM.md`](CANONICAL_EXTENSIVE_FORM.md).

## Shared core — true at every rung

These hold from L0 to L4 and are not restated per rung.

```text
state            x = (checker_load, checker_ttd, dropper_load, dropper_ttd)
                 role-relative; values from the current Dropper's perspective
actions          positive ordinal units 1..A; action 0 is illegal; no passing
success          check ≥ drop
squandered time  ST = check − drop + 1            (inclusive; never zero)
failure          dose q = checker_load + F        (F = the failed-check penalty)
capacity         C units; load reaching C is an immediate fatal injection
eligibility      q < C and checker_ttd + q ≤ C    (equality at C is eligible)
survival         load resets to 0, ttd becomes ttd + q, roles swap
death            the current Dropper wins
potential        checker_load + checker_ttd + dropper_load + dropper_ttd
                 strictly increases on every live edge ⇒ the graph is a finite DAG
```

Evidence: [E-ST-NONZERO], [E-INSTANT-CHECK], [E-ST-INCLUSIVE-LEDGER],
[E-DOSE-COMPOSITION], [E-CYLINDER-CAP], [E-TTD-EXACT], [E-TTD-SLACK],
[E-MAX-ST], [E-RESET-ON-REVIVAL], [E-OVERFLOW] in
[`game-sources/EVIDENCE.md`](game-sources/EVIDENCE.md).

Every rung is a simultaneous-move zero-sum game. The stage game is a matrix with
exactly `A + 1` degrees of freedom: `A` successful-lag classes indexed by
`check − drop`, plus one action-independent failed-check class.

<!-- canon:C-LADDER -->
## The rungs

### L0 — bucketed pure game · solved

Owner: `src/abstract/`. Bucket width `B`, `A = 60/B` actions, `C = 300/B`,
`F = 60/B`. Nothing but the shared core. No leap second, no private information,
no referee object, no player identity.

| Ruleset | `B` | `A` | `C` | reachable states | with dead-TTD quotient |
|---|---:|---:|---:|---:|---:|
| `bucket6` | 10 s | 6 | 30 | 576,270 | 55,681 |
| `bucket12` | 5 s | 12 | 60 | 8,870,160 | 644,335 |

Both are exactly enumerable and certified end to end. L0 exists to make the
rules falsifiable at a size where every state can be checked.

### L1 — pure DTH · the current target

Owner: `src/dth/`. Identical to L0 with `B = 1`: literal seconds `1..60`,
`C = 300`, `F = 60`. This is the **pure** formulation and it is now frozen —
"pure" means the shared core at one-second resolution and nothing else.

| | count |
|---|---:|
| reachable states | 5,267,489,760 |
| with the per-player dead-TTD quotient | 289,374,121 |
| distinct stage-matrix degrees of freedom | 61 |

L1 is the fidelity ceiling for the *mechanics*: no rung above it changes how ST,
doses, or revival work. Everything above adds information or an extra action.

### L2 — leap-aware public game

Owner: `src/stl/`. Adds one mechanic: within the leap window, the
Dropper may additionally choose second 61; the Checker stays capped at 60, so
`(drop = 61, check ≤ 60)` always fails.

The state gains a leap phase, because the window is a wall-clock event and each
half-round consumes a fixed 60 seconds:

```text
x = (checker_load, checker_ttd, dropper_load, dropper_ttd, phase)
phase ∈ {k half-rounds before the window, …, in the window, after the window}
```

Only the distance to the window matters, and after it the coordinate is
absorbing, so `phase` is bounded by the number of half-rounds a match can reach
the window from. Size is `L1 × |phase|`. Both players know the leap rule from
initialization; knowledge does not alter structural legality.

L2 is still **perfect information**. It is the largest rung that backward
induction can address in principle.

### L3 — private leap knowledge and memory loss

Adds genuinely unobserved state: which player currently holds the leap
realization, and the scheduled memory-loss event that can remove it. Players no
longer share a common posterior, so L3 is an imperfect-information game and
`V(x)` is replaced by values over information sets.

Backward induction does not apply. This is the first rung requiring
CFR/CFR+/ReBeL-class machinery, and the first where the repository's
exactness firewall cannot certify a solution by enumeration.

Not started. Do not claim any result at this rung.

### L4 — sensing and within-turn signalling

Adds echolocation conditioned on ambient volume the players manipulate, reads on
the opponent's tells, and deliberate within-turn signalling. The drop time stops
being a single simultaneous commitment and becomes a partially observable
process.

Research target. No formulation is frozen here; the mechanics in the source are
narrative and are not reducible to a small action set without further modelling
decisions that no evidence constrains.

## Claim boundary

- L0 is solved and certified.
- L1 is the active target. No complete-game claim exists for it yet.
- L2 has a public-state baseline implementation only.
- L3 and L4 are not implemented. Nothing in this repository may claim to solve
  a private-information version of this game.

Changing any rung's mechanic requires updating this file, the canonical docs,
the evidence ledger, the schemas, and the tests in one commit.

[E-ST-NONZERO]: game-sources/EVIDENCE.md#e-st-nonzero
[E-INSTANT-CHECK]: game-sources/EVIDENCE.md#e-instant-check
[E-ST-INCLUSIVE-LEDGER]: game-sources/EVIDENCE.md#e-st-inclusive-ledger
[E-DOSE-COMPOSITION]: game-sources/EVIDENCE.md#e-dose-composition
[E-CYLINDER-CAP]: game-sources/EVIDENCE.md#e-cylinder-cap
[E-TTD-EXACT]: game-sources/EVIDENCE.md#e-ttd-exact
[E-TTD-SLACK]: game-sources/EVIDENCE.md#e-ttd-slack
[E-MAX-ST]: game-sources/EVIDENCE.md#e-max-st
[E-RESET-ON-REVIVAL]: game-sources/EVIDENCE.md#e-reset-on-revival
[E-OVERFLOW]: game-sources/EVIDENCE.md#e-overflow
