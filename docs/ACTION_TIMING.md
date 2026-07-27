# Action Timing Authority

This is the sole repository authority for action seconds and Squandered Time
(ST). Engine, solver, learning, Rust, tests, and artifact schemas must agree.

## Evidence and interpretation

The sources establish that ST cannot be zero and that the narrated immediate
“0 seconds” check produces one second of ST. See
[E-ST-NONZERO](game-sources/EVIDENCE.md#e-st-nonzero) and
[E-INSTANT-CHECK](game-sources/EVIDENCE.md#e-instant-check).
The narrative zero is therefore immediate action in the first second, not a
literal solver action.

The canonical match ledger corroborates the inclusive convention independently:
twelve successful checks all reproduce their stated accumulation only under
`ST = c - d + 1`, and the maximum single half-round contribution observed is
exactly 60 seconds. See
[E-ST-INCLUSIVE-LEDGER](game-sources/EVIDENCE.md#e-st-inclusive-ledger) and
[E-MAX-ST](game-sources/EVIDENCE.md#e-max-st). The 60-second failed-check
penalty is confirmed five times by
[E-DOSE-COMPOSITION](game-sources/EVIDENCE.md#e-dose-composition).

<!-- canon:C-ACTION-NORMAL -->
## Literal-second convention

For a normal half-round:

```text
drop d  in {1, ..., 60}
check c in {1, ..., 60}
success iff c >= d
successful ST = c - d + 1
failure iff c < d
failed-check penalty = 60 seconds
```

Action 1 means acting immediately. Action 0 and passing are illegal. A
same-second pair succeeds and accumulates one second.

<!-- canon:C-ACTION-LEAP -->
## STL leap window

Both players know the leap rule from initialization. Knowledge does not alter
structural legality: only Baku while acting as Dropper may additionally choose
61 in the leap window. Checker remains capped at 60, so `(d=61, c<=60)` fails.
DTH and the canonical abstract example have no leap-second action.

## Dense policy indices

STL policy tensors have width 62: index 0 is illegal padding, indices 1..60 are
literal normal seconds, and index 61 is the conditional Baku drop. DTH policies
have 60 actions corresponding directly to seconds 1..60.

Any artifact produced with action 0, a pass action, exclusive `c-d` ST, or a
different leap mask is incompatible and must fail schema validation.
