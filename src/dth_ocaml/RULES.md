# DTH solver types and revival model

This project defines the pure role-relative DTH state and chance-branch types
and the repository-wide revival-probability function. It does not carry
STL-only leap-window or player-identity mechanics.

Simultaneous matrices are solved by GLPK through `lib/solver/matrix_game.ml`,
which certifies a saddle gap of at most `1e-6` before reporting a value. This
project does not implement a simplex method of its own.

## Actions and successful checks

- Dropper and Checker actions are literal seconds `1..60`.
- A check succeeds when `check >= drop`.
- Successful inclusive Squandered Time is `ST = check - drop + 1`.
- Reaching the 300-second cylinder capacity is a terminal Dropper win.

## Failed checks and revival

For a failed check, let `s` be the Checker’s vial contents immediately before
the injection and `t` be the Checker’s accrued TTD. The injected dose is:

```text
q = s + 60
```

The frozen repository model is:

```text
P_rev(s, t) = 0.95 * (1 - s / 240) * 0.75^(t / 60)
             if q < 300 and t + q <= 300
             0 otherwise
```

The equality `t + q = 300` remains eligible when `q < 300`. A failed-check
death is a terminal Dropper win; a survived failure swaps roles, clears the
revived player’s vial, and adds the dose to that player’s accrued TTD.

These numbers are solver constants, not documentary probabilities. The
repository authority and evidence boundary are documented in
[`docs/REVIVAL_MODEL.md`](../../docs/REVIVAL_MODEL.md).
